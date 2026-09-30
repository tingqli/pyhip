"""QSA indexer paged decode logits for gfx942 (FlyDSL): relu(Q_h . K) summed over 4 heads, times scale.

Decode has one query row (4 heads) per page-table row and reads that row's compressed keys through
its page table (16 keys = 4 KiB per page). Wave w of CTA (x, r) scores pages 4x + w, 4x + w +
4 * splits, ... of row r and stops at the row's device-side length, so one static grid covers
every length up to the table width (CUDA-graph safe, no host sync). Each page is read with
coalesced 1 KiB loads (one register-double-buffered page ahead) and transposed through a per-wave
4 KiB LDS copy: direct loads in the MFMA layout put 16 keys, i.e. 16 cache lines, in every
quarter-wave and ran at about half the bandwidth. A page is one v_mfma_f32_16x16x16_bf16 chain
with the heads as MFMA rows (rows 4..15 read as zero) and the 16 keys as columns: lane j < 16 ends
with the 4 head scores of key j, so ReLU, head sum and scale stay in-lane. Lane (column c, group
g) holds dims [32g, 32g + 32) of head c and of key c, so k-step s pairs dims 32g + 4s + [0, 4): a
fixed permutation of the D=128 reduction (FP32 order differs from the Torch reference only in
association). Keys [length, 16 * pages) get -inf and the rest of the row stays unwritten;
the shared FlyDSL top-k masks candidates outside [0, length).
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir import ir
from flydsl.expr import gpu, rocdl

from pyhip.codegen.flydsl.helpers import rocdl_aux
from pyhip.ops.mha.flydsl._common import _buffer, _buffer_words, _uniform
from .indexer_logits import _lds_load, _lds_store
from .indexer_topk import _readlane, _sync

HEADS, DIM, PAGE, WAVES = 4, 128, 16, 4
PAGE_BYTES = PAGE * DIM * 2
# CTAs per graph launch before splitting a row further stops paying (8 per CU on 80 CUs).
CTAS_PER_CU = 8
# Byte offset that fails every buffer range check.
_DROP = 0x40000000


def _buffer_word(resource, offset):
    return fx.Int32(rocdl.raw_ptr_buffer_load(fx.Int32.ir_type, resource, fx.Int32(offset).ir_value(),
                                              fx.Int32(0).ir_value(), aux=rocdl_aux(0)))


def _page(keys, page, lane, valid):
    # An invalid (past-the-chunk) prefetch is range-dropped instead of reading a stray page.
    offset = valid.select(page * PAGE_BYTES + 16 * lane, fx.Int32(_DROP))
    return [_buffer_words(keys, offset + 1024 * m) for m in range(4)]


def _stage(region, write, words):
    for address, value in zip(write, words):
        _lds_store(region + address, value)


def _step(words, s):
    chunk = words[s // 2]
    pair = (chunk[2], chunk[3]) if s & 1 else (chunk[0], chunk[1])
    return fx.Vector.from_elements(list(pair), fx.Int32).bitcast(fx.Int16)


def _logit(query, key, scale):
    acc = fx.Vector.filled(4, 0.0, fx.Float32)
    for s in range(8):
        acc = fx.Vector(rocdl.mfma_f32_16x16x16bf16_1k(
            ir.VectorType.get([4], fx.Float32.ir_type),
            [_step(query, s).ir_value(), _step(key, s).ir_value(), acc.ir_value(), 0, 0, 0]))
    # Signed-integer max on the FP32 bits: -0 and negatives become +0.
    bits, zero = acc.bitcast(fx.Int32), fx.Int32(0)
    relu = fx.Vector.from_elements([(bits[i] > zero).select(bits[i], zero) for i in range(4)],
                                   fx.Int32).bitcast(fx.Float32)
    return (((relu[0] + relu[1]) + relu[2]) + relu[3]) * scale


@flyc.kernel(name="qsa_indexer_decode_logits", known_block_size=[256, 1, 1])
def _kernel(Q: fx.Tensor, K: fx.Tensor, PAGES: fx.Tensor, LENGTHS: fx.Tensor, LOGITS: fx.Tensor,
            q_stride: fx.Int32, page_stride: fx.Int32, width: fx.Int32, splits: fx.Int32, key_bytes: fx.Int32,
            scale: fx.Float32):
    storage = fx.SharedAllocator().allocate(fx.Array[fx.Int8, WAVES * PAGE_BYTES, 16]).peek()
    shared = fx.Int32(fx.ptrtoint(fx.get_iter(storage.view(fx.make_layout(WAVES * PAGE_BYTES, 1)))))
    tid = fx.Int32(gpu.thread_id("x"))
    wave, lane = _uniform(tid >> 6), tid & 63
    column, group = lane & 15, lane >> 4
    split, row = fx.Int32(gpu.block_id("x")), fx.Int32(gpu.block_id("y"))
    length = _uniform(LENGTHS[row])
    length = (length < width).select(length, width)
    pages = (length + (PAGE - 1)) >> 4
    first = split * WAVES + wave
    if first < pages:
        # Heads 4..15 of the MFMA rows read past the 4-head buffer and load zeros.
        queries = _buffer(fx.make_view(fx.get_iter(Q) + fx.Int64(row) * fx.Int64(q_stride), fx.make_layout(1, 1)),
                          HEADS * DIM * 2)
        source = (column < HEADS).select(column * (DIM * 2) + group * 64, fx.Int32(_DROP))
        query = [_buffer_words(queries, source + 16 * m) for m in range(4)]
        keys = _buffer(K, key_bytes)
        output = _buffer(fx.make_view(fx.get_iter(LOGITS) + fx.Int64(row) * fx.Int64(width), fx.make_layout(1, 1)),
                         width * 4)
        table = _buffer(fx.make_view(fx.get_iter(PAGES) + fx.Int64(row) * fx.Int64(page_stride),
                                     fx.make_layout(1, 1)), pages * 4)
        # Global load m covers keys 4m..4m+3 of the page (1 KiB, coalesced); the per-wave LDS copy
        # puts key k's 16-byte chunk c at k * 256 + (c ^ (k & 7)) * 16 (conflict-free both ways).
        region = shared + wave * PAGE_BYTES
        write = [(4 * m + group) * 256 + ((column ^ ((4 * m + group) & 7)) * 16) for m in range(4)]
        read = [column * 256 + (((4 * group + t) ^ (column & 7)) * 16) for t in range(4)]
        target = (lane < PAGE).select(column * 4, fx.Int32(_DROP))
        step = splits * WAVES
        for chunk in range(first, pages, step * 64):
            # Lane i holds the page id of this wave's i-th page in the chunk (ids past the row load 0).
            begin = fx.Int32(chunk)
            ids = _buffer_word(table, (begin + lane * step) * 4)
            count = (pages - begin + step - 1) // step
            count = (count < 64).select(count, fx.Int32(64))
            w0, w1, w2, w3 = _page(keys, _readlane(ids, 0), lane, count > 0)
            for iteration in range(0, count):
                k = fx.Int32(iteration)
                index = begin + k * step
                # Page k + 1 is in flight while page k is transposed and scored.
                n0, n1, n2, n3 = _page(keys, _readlane(ids, (k + 1) & 63), lane, k + 1 < count)
                _sync()
                _stage(region, write, (w0, w1, w2, w3))
                _sync()
                value = _logit(query, [_lds_load(region + address) for address in read], scale)
                value = (index * PAGE + column < length).select(value, fx.Float32(float("-inf")))
                rocdl.raw_ptr_buffer_store(value.ir_value(), output, (index * (PAGE * 4) + target).ir_value(),
                                           fx.Int32(0).ir_value(), aux=rocdl_aux(0))
                w0, w1, w2, w3 = n0, n1, n2, n3


@flyc.jit
def _launch(Q: fx.Tensor, K: fx.Tensor, PAGES: fx.Tensor, LENGTHS: fx.Tensor, LOGITS: fx.Tensor,
            q_stride: fx.Int32, page_stride: fx.Int32, width: fx.Int32, splits: fx.Int32, key_bytes: fx.Int32,
            scale: fx.Float32, rows: fx.Int32, stream: fx.Stream):
    _kernel(Q, K, PAGES, LENGTHS, LOGITS, q_stride, page_stride, width, splits, key_bytes, scale).launch(
        grid=(splits, rows, 1), block=(256, 1, 1), stream=stream)


_COMPILED = {}


def compiled(device):
    return device in _COMPILED


def launch(q, cache, page_table, lengths, logits, scale):
    """q: [rows, >=4, 128] BF16 (heads 0..3 used, row stride a multiple of 8); cache: compressed K
    [slots, 1, 128] BF16 (any view, < 2 GiB); page_table: [rows, pages] int32 (page p of row r holds
    keys 16p..16p+15 at slots 16 * page_table[r, p] + [0, 16)); lengths: [rows] int32; logits:
    [rows, 16 * pages] FP32, contiguous. Compiles on the first call per device (not graph-capturable)."""
    rows, width = logits.shape
    count = torch.cuda.get_device_properties(q.device).multi_processor_count
    splits = max(1, min(-(-page_table.shape[1] // WAVES), -(-count * CTAS_PER_CU // rows)))
    stream = torch.cuda.current_stream(q.device)
    args = (q.view(-1), cache.view(-1), page_table.view(-1), lengths, logits.view(-1), q.stride(0),
            page_table.stride(0), width, splits, cache.numel() * 2, scale, rows, stream)
    with torch.cuda.device(q.device):
        function = _COMPILED.get(q.device)
        if function is None:
            _COMPILED[q.device] = flyc.compile(_launch, *args)
        else:
            function(*args)
