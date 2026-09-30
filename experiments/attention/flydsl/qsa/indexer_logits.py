"""QSA indexer prefill logits for gfx942 (FlyDSL): relu(Q_h . K) summed over 4 heads, times scale.

One 4-wave CTA owns up to 128 query rows (32 per wave) of one request and a contiguous key
range. Q stays in VGPRs (4 heads x 64 BF16 per lane). Each 32-key block is staged once per CTA
into double-buffered, XOR-swizzled LDS and fed to v_mfma_f32_32x32x8_bf16 with keys as the
MFMA rows, so every lane holds 4 consecutive keys of one query row (16-byte stores). Lane half
h holds head dims [64h, 64h + 64) of its key/query row, so k-step s pairs dims {4s..4s+3} and
{64+4s..64+4s+3}: a fixed permutation of the D=128 reduction (FP32 order differs from the
Torch einsum reference only in association). Two CTAs share a CU.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import gpu, rocdl

from ..mha._common import _buffer, _buffer_words, _uniform

HEADS, DIM, BLOCK, ITEM = 4, 128, 32, 6
# Byte offset that fails every per-wave range check.
_DROP = 0x40000000


def _lds(address):
    return llvm.inttoptr(ir.Type.parse("!llvm.ptr<3>"), fx.Int32(address).ir_value())


def _lds_load(address):
    return fx.Vector(llvm.load(ir.VectorType.get([4], fx.Int32.ir_type), _lds(address), alignment=16))


def _lds_store(address, value):
    llvm.store(fx.Vector(value).ir_value(), _lds(address), alignment=16)


def _step(words, s):
    pair = (words[2], words[3]) if s & 1 else (words[0], words[1])
    return fx.Vector.from_elements(list(pair), fx.Int32).bitcast(fx.Int16)


def _mfma(a, b, c):
    return fx.Vector(rocdl.mfma_f32_32x32x8bf16_1k(ir.VectorType.get([16], fx.Float32.ir_type),
                                                   [a.ir_value(), b.ir_value(), c.ir_value(), 0, 0, 0]))


def _relu(values):
    # Signed-integer max on the FP32 bits: -0 and negatives become +0.
    bits = fx.Vector(values).bitcast(fx.Int32)
    zero = fx.Int32(0)
    return fx.Vector.from_elements([(bits[i] > zero).select(bits[i], zero) for i in range(bits.numel)],
                                   fx.Int32).bitcast(fx.Float32)


def _scores(key, query):
    # Keys are the MFMA rows and query rows the columns: lane (column, half) ends up holding keys
    # block + 8j + 4 * half + [0, 4) of query row `column`, i.e. one 16-byte store per j.
    acc = [fx.Vector.filled(16, 0.0, fx.Float32) for _ in range(HEADS)]
    for s in range(16):
        for h in range(HEADS):
            acc[h] = _mfma(_step(key[s // 2], s), _step(query[h][s // 2], s), acc[h])
    return [_relu(value) for value in acc]


def _store(relu, output, block, end, column, half, stride, scale):
    for j in range(4):
        words = []
        for p in range(2):
            i = 4 * j + 2 * p
            value = fx.Vector.from_elements([relu[0][i], relu[0][i + 1]], fx.Float32)
            for h in range(1, HEADS):
                value = value + fx.Vector.from_elements([relu[h][i], relu[h][i + 1]], fx.Float32)
            value = value * fx.Vector.from_elements([scale, scale], fx.Float32)
            words.extend(value.bitcast(fx.Int32)[e] for e in range(2))
        # A 4-key group past `end` is dropped; a partial group only spills into columns
        # [end, end + 3], which no row reads as a causal logit.
        group = block + 8 * j + 4 * half
        offset = (group < end).select((column * stride + group) * 4, fx.Int32(_DROP))
        rocdl.raw_ptr_buffer_store(fx.Vector.from_elements(words, fx.Int32).ir_value(), output,
                                   offset.ir_value(), fx.Int32(0).ir_value())


@flyc.kernel(name="qsa_indexer_logits", known_block_size=[256, 1, 1])
def _kernel(Q: fx.Tensor, K: fx.Tensor, LOGITS: fx.Tensor, ITEMS: fx.Tensor, stride: fx.Int32, scale: fx.Float32):
    # Key c's 16-byte chunk j lives at c * 16 + (j ^ (c & 7)): conflict-free ds_read_b128.
    storage = fx.SharedAllocator().allocate(fx.Array[fx.Int8, 2 * BLOCK * 256, 16]).peek()
    shared = fx.Int32(fx.ptrtoint(fx.get_iter(storage.view(fx.make_layout(2 * BLOCK * 256, 1)))))
    tid = fx.Int32(gpu.thread_id("x"))
    wave, lane = _uniform(tid >> 6), tid & 63
    column, half = lane & 31, lane >> 5
    entry = fx.Int32(gpu.block_id("x")) * ITEM
    row0, local0, rows = _uniform(ITEMS[entry]), _uniform(ITEMS[entry + 1]), _uniform(ITEMS[entry + 2])
    key_base, start, end = _uniform(ITEMS[entry + 3]), _uniform(ITEMS[entry + 4]), _uniform(ITEMS[entry + 5])
    wave_rows = rows - 32 * wave
    wave_rows = (wave_rows > 32).select(fx.Int32(32), (wave_rows < 0).select(fx.Int32(0), wave_rows))

    queries = _buffer(fx.make_view(fx.get_iter(Q) + fx.Int64(row0) * (HEADS * DIM), fx.make_layout(1, 1)),
                      rows * (HEADS * DIM * 2))
    row = 32 * wave + column
    row = (row < rows).select(row, rows - 1)
    query = [[_buffer_words(queries, row * (HEADS * DIM * 2) + half * 128 + h * (DIM * 2) + i * 16)
              for i in range(8)] for h in range(HEADS)]

    # Buffer stores are range checked against the wave's rows, so padded rows are dropped without
    # branches; 4-key groups starting past `end` push their offset out of range.
    output = _buffer(fx.make_view(fx.get_iter(LOGITS) + fx.Int64(local0 + 32 * wave) * fx.Int64(stride),
                                  fx.make_layout(1, 1)), wave_rows * stride * 4)
    # Thread t copies 32 bytes (chunks 2(t%8), 2(t%8)+1) of key t/8 of each block into LDS.
    copy_key, copy_chunk = tid >> 3, (tid & 7) * 2
    keys = _buffer(fx.make_view(fx.get_iter(K) + fx.Int64(key_base) * DIM, fx.make_layout(1, 1)), end * (DIM * 2))
    stage0 = copy_key * 256 + ((copy_chunk ^ (copy_key & 7)) * 16)
    stage1 = copy_key * 256 + (((copy_chunk + 1) ^ (copy_key & 7)) * 16)
    read = [column * 256 + (((8 * half + i) ^ (column & 7)) * 16) for i in range(8)]

    first = start + copy_key
    source = (first < end).select(first, end - 1) * (DIM * 2) + copy_chunk * 16
    staged0, staged1 = _buffer_words(keys, source), _buffer_words(keys, source + 16)
    _lds_store(shared + stage0, staged0)
    _lds_store(shared + stage1, staged1)
    gpu.barrier()
    buffer = fx.Int32(0)
    for block in range(start, end, fx.Int32(BLOCK)):
        current = fx.Int32(block)
        more = current + BLOCK < end
        if more:
            following = current + BLOCK + copy_key
            source = (following < end).select(following, end - 1) * (DIM * 2) + copy_chunk * 16
            staged0, staged1 = _buffer_words(keys, source), _buffer_words(keys, source + 16)
        base = shared + buffer * (BLOCK * 256)
        key = [_lds_load(base + address) for address in read]
        _store(_scores(key, query), output, current, end, column, half, stride, scale)
        if more:
            base = shared + (buffer ^ 1) * (BLOCK * 256)
            _lds_store(base + stage0, staged0)
            _lds_store(base + stage1, staged1)
        gpu.barrier()
        buffer = buffer ^ 1


@flyc.jit
def _launch(Q: fx.Tensor, K: fx.Tensor, LOGITS: fx.Tensor, ITEMS: fx.Tensor, stride: fx.Int32, scale: fx.Float32,
            items: fx.Int32, stream: fx.Stream):
    _kernel(Q, K, LOGITS, ITEMS, stride, scale, value_attrs={"rocdl.waves_per_eu": 2}).launch(
        grid=(items, 1, 1), block=(256, 1, 1), stream=stream)


_COMPILED = {}


def launch(q, packed, logits, items, stride, scale):
    """items: [n, 6] int32 (row0, local0, rows <= 128, key_base, start, end), one CTA each; wave w
    owns rows [32w, 32w + 32) and writes logits row local0 + 32w + m. Chunk ends other than the
    tile's causal bound are multiples of 32, and the row stride is a multiple of 4."""
    stream = torch.cuda.current_stream(q.device)
    args = (q.view(-1), packed.view(-1), logits.view(-1), items.view(-1), stride, scale, items.shape[0], stream)
    with torch.cuda.device(q.device):
        compiled = _COMPILED.get(q.device)
        if compiled is None:
            _COMPILED[q.device] = flyc.compile(_launch, *args)
        else:
            compiled(*args)
