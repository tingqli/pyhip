"""QSA indexer for gfx942 after ``index_qk_proj``: prefill/decode prep, logits and top-k.

BF16 norm/RoPE/mean steps (Triton) round like SGLang's eager chain, bit for bit.
Logits (FlyDSL MFMA, ``indexer_logits.py``) are the FP32 relu head sum of BF16 dots,
so only accumulation order differs from the Torch einsum reference. Top-k (FlyDSL,
``indexer_topk.py``) keeps the 512 largest ordered FP32 keys; ties keep the
lowest block ids. Each row holds the selected blocks' tokens (block order is
unspecified, like SGLang's radix top-k), then 0..3 causal tail tokens, then -1,
matching SGLang's fixed-width token ABI. Decode (``decode_indexer``) uses paged logits
(``indexer_decode.py``) and the shared FlyDSL top-k/expand; ``decode_forward`` also fuses the
CUDA-graph decode q prep, pending-ring store and group compression into one bit-exact kernel.
"""

from collections import OrderedDict
from itertools import accumulate
import math
import threading

import numpy as np
import torch
import triton
import triton.language as tl

from . import indexer_decode, indexer_logits, indexer_topk

__all__ = ["MAX_COMPRESSED_KEYS", "decode_forward", "decode_indexer", "prefill_indexer"]

_LOGITS_BUDGET_BYTES = 256 * 1024 * 1024
# The top-k kernel keeps chosen block ids as uint16.
MAX_COMPRESSED_KEYS = 65536
# Logits CTA tile rows and keys per work item (a multiple of the kernel's 32-key block).
_BM, _KC = 128, 512
# Tokens per q-prep program and compressed groups per compress program.
_TB, _GB = 8, 16
_TOPK, _RATIO, _WIDTH = 512, 4, 2051
# The top-k kernel reads rows up to the next 512-value boundary (8 slots x 64 lanes).
_PAD = 512


@triton.jit
def _rstd(x, n_cols, eps, R: tl.constexpr):
    # SGLang's Gemma RMSNorm reduces each 128-wide row with DPP row_shr 8/4/2/1, then row pairs,
    # then the two warp partials; summing index bits 3,2,1,0,4,5,6 in that order is bit-exact.
    s = tl.reshape(x * x, [R, 2, 2, 2, 2, 2, 2, 2])
    s = tl.sum(tl.sum(tl.sum(tl.sum(s, axis=4), axis=4), axis=4), axis=4)
    s = tl.sum(tl.sum(tl.sum(s, axis=3), axis=2), axis=1)
    return 1.0 / tl.sqrt(s / n_cols + eps)


@triton.jit
def _partner(cols, ROT: tl.constexpr):
    return tl.where(cols < ROT // 2, cols + ROT // 2, tl.where(cols < ROT, cols - ROT // 2, cols))


@triton.jit
def _norm_rope(x, partner, n_cols, eps, Weight, Cache, Axis, p0, p1, p2, R: tl.constexpr, D: tl.constexpr,
               ROT: tl.constexpr, CACHE_STRIDE: tl.constexpr):
    # Rows of x are normed like SGLang, then NeoX-rotated with its per-op BF16 roundings; partner holds
    # each rotary column's other half (x[i + ROT/2] or x[i - ROT/2]).
    cols = tl.arange(0, D)
    half = cols % (ROT // 2)
    rot = cols < ROT
    rstd = _rstd(x, n_cols, eps, R)[:, None]
    y = (x * rstd * (1.0 + tl.load(Weight + cols).to(tl.float32))[None, :]).to(tl.bfloat16).to(tl.float32)
    yp = (partner * rstd * (1.0 + tl.load(Weight + _partner(cols, ROT)).to(tl.float32))[None, :])
    yp = yp.to(tl.bfloat16).to(tl.float32)
    selector = tl.load(Axis + half, mask=rot, other=0)[None, :]
    pos = tl.where(selector == 0, p0[:, None], tl.where(selector == 1, p1[:, None], p2[:, None]))
    cache = Cache + pos * CACHE_STRIDE + half[None, :]
    c = tl.load(cache, mask=rot[None, :], other=0.0).to(tl.bfloat16).to(tl.float32)
    s = tl.load(cache + ROT // 2, mask=rot[None, :], other=0.0).to(tl.bfloat16).to(tl.float32)
    own = (y * c).to(tl.bfloat16).to(tl.float32)
    other = (yp * s).to(tl.bfloat16).to(tl.float32)
    out = tl.where((cols < ROT // 2)[None, :], own - other, own + other)
    return tl.where(rot[None, :], out, y).to(tl.bfloat16)


@triton.jit
def _indexer_q_prep(QK, Q, Weight, KeyState, RopeState, Slots, Positions, position_stride,
                    Cache, Axis, Valid, tokens, n_cols, eps, TB: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
                    ROT: tl.constexpr, CACHE_STRIDE: tl.constexpr):
    # position_stride is 0 for 1-D positions, so all three rope axes read the same row.
    first = tl.program_id(0) * TB
    if first == 0:
        tl.store(Valid, 1)
    cols = tl.arange(0, D)
    row = tl.arange(0, TB * H)
    token = first + row // H
    live = token < tokens
    p0 = tl.load(Positions + token, mask=live, other=0)
    p1 = tl.load(Positions + position_stride + token, mask=live, other=0)
    p2 = tl.load(Positions + 2 * position_stride + token, mask=live, other=0)
    src = QK + (token.to(tl.int64) * (H + 1) * D + row % H * D)[:, None]
    x = tl.load(src + cols[None, :], mask=live[:, None], other=0.0).to(tl.float32)
    partner = tl.load(src + _partner(cols, ROT)[None, :], mask=live[:, None], other=0.0).to(tl.float32)
    q = _norm_rope(x, partner, n_cols, eps, Weight, Cache, Axis, p0, p1, p2, TB * H, D, ROT, CACHE_STRIDE)
    tl.store(Q + (token.to(tl.int64) * H * D + row % H * D)[:, None] + cols[None, :], q, mask=live[:, None])
    token = first + tl.arange(0, TB)
    live = token < tokens
    slot = tl.load(Slots + token, mask=live, other=0)
    key = tl.load(QK + (token.to(tl.int64) * (H + 1) * D + H * D)[:, None] + cols[None, :], mask=live[:, None])
    tl.store(KeyState + slot[:, None] * D + cols[None, :], key, mask=live[:, None])
    lanes = tl.arange(0, 4)[None, :]
    axes = live[:, None] & (lanes < 3)
    value = tl.load(Positions + lanes * position_stride + token[:, None], mask=axes, other=0)
    tl.store(RopeState + slot[:, None] * 3 + lanes, value, mask=axes)


@triton.jit
def _group_mean(QK, first, step, cols, H: tl.constexpr, D: tl.constexpr, RATIO: tl.constexpr):
    row = QK + (first * (H + 1) * D + H * D)[:, None] + cols[None, :]
    acc = tl.load(row).to(tl.float32)
    for member in tl.static_range(1, RATIO):
        acc = acc + tl.load(row + (member * step * (H + 1) * D)[:, None]).to(tl.float32)
    return (acc * (1.0 / RATIO)).to(tl.bfloat16).to(tl.float32)


@triton.jit
def _indexer_k_compress(QK, Rows, WriteLocs, Sequences, GroupEnds, RopeMatrix, Cache, Axis, Weight,
                        Compressed, Packed, KeyBase, groups, n_cols, eps, GB: tl.constexpr, H: tl.constexpr,
                        D: tl.constexpr, ROT: tl.constexpr, RATIO: tl.constexpr, CACHE_STRIDE: tl.constexpr):
    group = tl.program_id(0) * GB + tl.arange(0, GB)
    live = group < groups
    loc = tl.load(WriteLocs + group, mask=live, other=0).to(tl.int64)
    # Planner padding writes the inert slot 0; ROCm SGLang gathers member row 0 for it.
    first = tl.where(loc != 0, tl.load(Rows + group, mask=live, other=0), 0)
    step = tl.where(loc != 0, 1, 0)
    cols = tl.arange(0, D)
    x = _group_mean(QK, first, step, cols, H, D, RATIO)
    partner = _group_mean(QK, first, step, _partner(cols, ROT), H, D, RATIO)
    p0 = tl.load(RopeMatrix + first * 3)
    p1 = tl.load(RopeMatrix + first * 3 + 1)
    p2 = tl.load(RopeMatrix + first * 3 + 2)
    key = _norm_rope(x, partner, n_cols, eps, Weight, Cache, Axis, p0, p1, p2, GB, D, ROT, CACHE_STRIDE)
    tl.store(Compressed + loc[:, None] * D + cols[None, :], key, mask=live[:, None])
    real = live & (loc != 0)
    packed = tl.load(KeyBase + tl.load(Sequences + group, mask=real, other=0), mask=real, other=0)
    packed = packed + tl.load(GroupEnds + group, mask=real, other=0) // RATIO
    tl.store(Packed + packed.to(tl.int64)[:, None] * D + cols[None, :], key, mask=real[:, None])


@triton.jit
def _ring_member(KeyState, loc, own, key, cols, D: tl.constexpr):
    # SGLang gathers after its ring store: a member in this row's own slot is this token's key.
    return tl.where(own, key, tl.load(KeyState + loc * D + cols)).to(tl.float32)


@triton.jit
def _indexer_decode_prep(QK, Q, QWeight, KWeight, KeyState, RopeState, Compressed, Slots, GroupLocs, WriteLocs,
                         Positions, position_stride, Cache, Axis, n_cols, q_eps, k_eps, H: tl.constexpr,
                         D: tl.constexpr, ROT: tl.constexpr, RATIO: tl.constexpr, CACHE_STRIDE: tl.constexpr):
    # One CUDA-graph decode row: q norm/RoPE, pending-ring store, then SGLang's fixed-shape
    # compression of the group ending at this token (oldest member first, written to WriteLocs).
    row = tl.program_id(0)
    cols = tl.arange(0, D)
    other = _partner(cols, ROT)
    heads = tl.arange(0, H)
    r0 = tl.load(Positions + row)
    r1 = tl.load(Positions + position_stride + row)
    r2 = tl.load(Positions + 2 * position_stride + row)
    src = QK + row.to(tl.int64) * (H + 1) * D
    x = tl.load(src + heads[:, None] * D + cols[None, :]).to(tl.float32)
    partner = tl.load(src + heads[:, None] * D + other[None, :]).to(tl.float32)
    lanes = tl.zeros([H], dtype=tl.int64)
    q = _norm_rope(x, partner, n_cols, q_eps, QWeight, Cache, Axis, r0 + lanes, r1 + lanes, r2 + lanes, H, D, ROT,
                   CACHE_STRIDE)
    tl.store(Q + row.to(tl.int64) * H * D + heads[:, None] * D + cols[None, :], q)
    key = tl.load(src + H * D + cols)
    key_other = tl.load(src + H * D + other)
    slot = tl.load(Slots + row).to(tl.int64)
    tl.store(KeyState + slot * D + cols, key)
    axes = tl.arange(0, 4)
    tl.store(RopeState + slot * 3 + axes, tl.load(Positions + axes * position_stride + row, mask=axes < 3, other=0),
             mask=axes < 3)
    first = tl.load(GroupLocs + row * RATIO).to(tl.int64)
    own = first == slot
    acc = _ring_member(KeyState, first, own, key, cols, D)
    acc_other = _ring_member(KeyState, first, own, key_other, other, D)
    for member in tl.static_range(1, RATIO):
        loc = tl.load(GroupLocs + row * RATIO + member).to(tl.int64)
        acc = acc + _ring_member(KeyState, loc, loc == slot, key, cols, D)
        acc_other = acc_other + _ring_member(KeyState, loc, loc == slot, key_other, other, D)
    mean = (acc * (1.0 / RATIO)).to(tl.bfloat16).to(tl.float32)
    mean_other = (acc_other * (1.0 / RATIO)).to(tl.bfloat16).to(tl.float32)
    one = tl.zeros([1], dtype=tl.int64)
    c0 = tl.where(own, r0, tl.load(RopeState + first * 3)) + one
    c1 = tl.where(own, r1, tl.load(RopeState + first * 3 + 1)) + one
    c2 = tl.where(own, r2, tl.load(RopeState + first * 3 + 2)) + one
    k = _norm_rope(mean[None, :], mean_other[None, :], n_cols, k_eps, KWeight, Cache, Axis, c0, c1, c2, 1, D, ROT,
                   CACHE_STRIDE)
    tl.store(Compressed + tl.load(WriteLocs + row).to(tl.int64) * D + cols[None, :], k)


@triton.jit
def _indexer_gather_prefix(Compressed, Packed, Table, Pairs, table_stride, D: tl.constexpr, RATIO: tl.constexpr):
    pair = tl.program_id(0)
    destination = tl.load(Pairs + pair * 3)
    sequence = tl.load(Pairs + pair * 3 + 1).to(tl.int64)
    block = tl.load(Pairs + pair * 3 + 2)
    slot = tl.load(Table + sequence * table_stride + block * RATIO).to(tl.int64) // RATIO
    cols = tl.arange(0, D)
    tl.store(Packed + destination.to(tl.int64) * D + cols, tl.load(Compressed + slot * D + cols))


class _Chunk:
    def __init__(self, row0, rows, width, items):
        self.row0, self.rows, self.width, self.items = row0, rows, width, items


class _Layout:
    """Host-derived per-forward plan; device tensors are read-only and shared."""

    def __init__(self, seq_lens, extend_lens, device):
        prefixes = [s - e for s, e in zip(seq_lens, extend_lens)]
        compressed = [s // _RATIO for s in seq_lens]
        key_base = list(accumulate(compressed, initial=0))
        self.rows, self.keys = sum(extend_lens), key_base[-1]
        info = np.empty((self.rows, 2), dtype=np.int32)
        tiles, pairs, row = [], [], 0
        for sequence, (length, extend, prefix) in enumerate(zip(seq_lens, extend_lens, prefixes)):
            info[row:row + extend, 0] = prefix + np.arange(extend)
            info[row:row + extend, 1] = length
            for start in range(0, extend, _BM):
                count = min(_BM, extend - start)
                bound = min((prefix + start + count) // _RATIO, compressed[sequence])
                tiles.append((row + start, count, key_base[sequence], bound))
            pairs.extend((key_base[sequence] + block, sequence, block) for block in range(prefix // _RATIO))
            row += extend
        self.chunks, self.logits_elements = [], 0
        begin = 0
        while begin < len(tiles):
            end, width = begin, 4
            while end < len(tiles):
                grown = max(width, -(-tiles[end][3] // 4) * 4)
                rows = tiles[end][0] + tiles[end][1] - tiles[begin][0]
                if end > begin and rows * grown * 4 > _LOGITS_BUDGET_BYTES:
                    break
                width, end = grown, end + 1
            row0 = tiles[begin][0]
            rows = tiles[end - 1][0] + tiles[end - 1][1] - row0
            items = [(first, first - row0, count, base, key, min(key + _KC, bound))
                     for first, count, base, bound in tiles[begin:end] if bound > _TOPK
                     for key in range(0, bound, _KC)]
            tensor = torch.tensor(items, dtype=torch.int32, device=device).reshape(-1, 6)
            self.chunks.append(_Chunk(row0, rows, width, tensor))
            self.logits_elements = max(self.logits_elements, rows * width)
            begin = end
        self.row_info = torch.from_numpy(info).to(device)
        self.key_base = torch.tensor(key_base[:-1], dtype=torch.int32, device=device)
        self.pairs = torch.tensor(pairs, dtype=torch.int32, device=device).reshape(-1, 3) if pairs else None


_layouts = OrderedDict()
_lock = threading.Lock()


def _layout(seq_lens, extend_lens, device):
    key = (device, seq_lens, extend_lens)
    with _lock:
        layout = _layouts.get(key)
        if layout is None:
            layout = _layouts[key] = _Layout(seq_lens, extend_lens, device)
            while len(_layouts) > 8:
                _layouts.popitem(last=False)
        _layouts.move_to_end(key)
        return layout


def _check(tensor, name, dtype, shape=None, contiguous=True):
    if tensor.dtype != dtype or (shape is not None and tuple(tensor.shape) != tuple(shape)) or (
            contiguous and not tensor.is_contiguous()):
        raise ValueError(f"{name} must be {'contiguous ' if contiguous else ''}{dtype} {shape}, got "
                         f"{tensor.dtype} {tuple(tensor.shape)}")


def prefill_indexer(qk, **kwargs):
    """Return int32 [T, 2051] token selections and update the QSA pending/compressed caches.

    qk is the contiguous BF16 ``index_qk_proj`` output [T, (heads + 1) * D]. Host
    seq/extend lengths describe the packed extend batch; request-local query
    positions must equal ``prefix + i`` (checked on device). ``write_locs`` and the
    other group tensors are SGLang's extend write plan, including slot-0 padding.
    """
    return _prefill(qk, **kwargs)[0]


def _prefill(qk, *, heads, positions, logical_positions, state_slots, key_state, rope_state,
             write_locs, member_rows, group_sequences, group_ends, rope_matrix, compressed,
             token_slot_table, cos_sin_cache, axis_map, q_weight, k_weight, q_eps, k_eps,
             seq_lens, extend_lens):
    device, head_dim = qk.device, q_weight.numel()
    rows = sum(extend_lens)
    rotary_dim = cos_sin_cache.shape[1]
    if heads != 4 or head_dim != 128 or rotary_dim % 4 or not 0 < rotary_dim < head_dim:
        raise ValueError("The gfx942 indexer kernels require 4 heads, D128 and 0<rotary_dim<D")
    if any(n & (n - 1) for n in (rotary_dim // 2, head_dim - rotary_dim)):
        raise ValueError("rotary_dim/2 and head_dim-rotary_dim must be powers of two")
    if len(seq_lens) != len(extend_lens) or any(s < e or e < 0 for s, e in zip(seq_lens, extend_lens)):
        raise ValueError("Invalid host sequence/extend lengths")
    if max(seq_lens) // _RATIO > MAX_COMPRESSED_KEYS or rows == 0:
        raise ValueError(f"Rows must be non-empty and have at most {MAX_COMPRESSED_KEYS} compressed keys")
    _check(qk, "qk", torch.bfloat16, (rows, (heads + 1) * head_dim))
    _check(logical_positions, "logical_positions", torch.int64, (rows,))
    _check(state_slots, "state_slots", torch.int64, (rows,))
    _check(key_state, "key_state", torch.bfloat16, (key_state.shape[0], 1, head_dim))
    _check(rope_state, "rope_state", torch.int64, (key_state.shape[0], 3))
    groups = write_locs.numel()
    _check(write_locs, "write_locs", torch.int32, (groups,))
    for tensor, name in ((member_rows, "member_rows"), (group_sequences, "group_sequences"),
                         (group_ends, "group_ends")):
        _check(tensor, name, torch.int64, (groups,))
    _check(rope_matrix, "rope_matrix", torch.int64, (rows, 3))
    _check(compressed, "compressed", torch.bfloat16, (compressed.shape[0], 1, head_dim))
    _check(token_slot_table, "token_slot_table", torch.int32, (len(seq_lens), token_slot_table.shape[1]), False)
    _check(axis_map, "axis_map", torch.int32, (rotary_dim // 2,))
    _check(q_weight, "q_weight", torch.bfloat16, (head_dim,))
    _check(k_weight, "k_weight", torch.bfloat16, (head_dim,))
    if (cos_sin_cache.dtype not in (torch.bfloat16, torch.float32) or not cos_sin_cache.is_contiguous()
            or positions.dtype != torch.int64 or positions.ndim not in (1, 2) or positions.stride(-1) != 1
            or positions.shape[-1] != rows or (positions.ndim == 2 and positions.shape[0] != 3)
            or token_slot_table.stride(1) != 1 or token_slot_table.shape[1] < max(seq_lens)):
        raise ValueError("Unsupported RoPE cache, positions or token-slot table layout")
    tensors = (qk, positions, logical_positions, state_slots, key_state, rope_state, write_locs, member_rows,
               group_sequences, group_ends, rope_matrix, compressed, token_slot_table, cos_sin_cache, axis_map,
               q_weight, k_weight)
    if any(t.device != device for t in tensors):
        raise ValueError("All indexer tensors must be on one device")

    layout = _layout(tuple(seq_lens), tuple(extend_lens), device)
    q = torch.empty((rows, heads, head_dim), dtype=torch.bfloat16, device=device)
    valid = torch.empty(1, dtype=torch.int32, device=device)
    _indexer_q_prep[(-(-rows // _TB),)](qk, q, q_weight, key_state, rope_state, state_slots, positions,
                                        positions.stride(0) if positions.ndim == 2 else 0, cos_sin_cache,
                                        axis_map, valid, rows, head_dim, q_eps, TB=_TB, H=heads, D=head_dim,
                                        ROT=rotary_dim, CACHE_STRIDE=cos_sin_cache.stride(0), num_warps=4)
    packed = torch.empty((max(layout.keys, 1), head_dim), dtype=torch.bfloat16, device=device)
    if groups:
        _indexer_k_compress[(-(-groups // _GB),)](qk, member_rows, write_locs, group_sequences, group_ends,
                                                  rope_matrix, cos_sin_cache, axis_map, k_weight, compressed,
                                                  packed, layout.key_base, groups, head_dim, k_eps, GB=_GB,
                                                  H=heads, D=head_dim, ROT=rotary_dim, RATIO=_RATIO,
                                                  CACHE_STRIDE=cos_sin_cache.stride(0), num_warps=4)
    if layout.pairs is not None:
        _indexer_gather_prefix[(layout.pairs.shape[0],)](compressed, packed, token_slot_table, layout.pairs,
                                                         token_slot_table.stride(0), D=head_dim, RATIO=_RATIO,
                                                         num_warps=1)
    out = torch.empty((rows, _WIDTH), dtype=torch.int32, device=device)
    logits = torch.empty(max(layout.logits_elements, 1) + _PAD, dtype=torch.float32, device=device)
    scale = float(np.float32(1.0) / np.float32(math.sqrt(head_dim)))
    for chunk in layout.chunks:
        view = logits[:chunk.rows * chunk.width].view(chunk.rows, chunk.width)
        if chunk.items.shape[0]:
            indexer_logits.launch(q, packed, view, chunk.items, chunk.width, scale)
        indexer_topk.launch(view, chunk.width, chunk.row0, chunk.rows, logical_positions, layout.row_info, out, valid)
    torch._assert_async(valid, "QSA indexer query positions differ from host prefix+index layout")
    return out, q, packed


def decode_indexer(q, cache, page_table, lengths, query_positions, sequence_lengths):
    """Paged decode selection for 4 BF16 heads: int32 [rows, 2051] token indices.

    q is contiguous [rows, 4 or 8, 128] (heads 4..7 are the fused prep's zero padding), cache the
    [pages, 16, 1, 128] BF16 compressed pool, page_table int32 [rows, P] (rows score 16 * P keys)
    and lengths the int32 compressed lengths. Only the logits change (FlyDSL paged kernel reading
    each row's own keys instead of the whole table width). FlyDSL top-k and expansion share
    the prefill selection core. Everything is device-side, so the call is CUDA-graph capturable
    after eager warmup. Each row has at most MAX_COMPRESSED_KEYS compressed keys.
    """
    return _decode_select(q, cache, page_table, lengths, query_positions, sequence_lengths)[0]


def _decode_select(q, cache, page_table, lengths, query_positions, sequence_lengths):
    rows, width = q.shape[0], page_table.shape[1] * 16
    if width > MAX_COMPRESSED_KEYS:
        raise ValueError(f"Decode supports at most {MAX_COMPRESSED_KEYS} compressed keys per row")
    logits = torch.empty(rows * width + _PAD, dtype=torch.float32, device=q.device)[:rows * width].view(rows, width)
    out = torch.empty((rows, _WIDTH), dtype=torch.int32, device=q.device)
    indexer_decode.launch(q, cache, page_table, lengths, logits, float(np.float32(1.0) / np.float32(math.sqrt(128))))
    indexer_topk.launch_decode(logits, lengths.contiguous(), query_positions.contiguous(),
                               sequence_lengths.contiguous(), out)
    return out, logits


def decode_forward(qk, **kwargs):
    """SGLang's CUDA-graph decode ``QSAIndexer.forward_cuda`` after ``index_qk_proj``: int32 [rows, 2051].

    One Triton kernel reproduces the unfused (BF16 cos/sin) q norm/RoPE, the pending-ring key and
    RoPE-position store at ``state_slots`` and the fixed-shape compression of ``group_locs``
    (oldest member first) into ``write_locs`` bit for bit; ``decode_indexer`` then selects tokens.
    Rows must belong to distinct requests (each row only sees its own ring store).
    """
    return _decode_forward(qk, **kwargs)[0]


def _decode_forward(qk, *, positions, state_slots, group_locs, write_locs, key_state, rope_state, compressed,
                    cos_sin_cache, axis_map, q_weight, k_weight, q_eps, k_eps, cache, page_table, lengths,
                    query_positions, sequence_lengths):
    rows, heads, head_dim = qk.shape[0], 4, q_weight.numel()
    rotary_dim = cos_sin_cache.shape[1]
    q = torch.empty((rows, heads, head_dim), dtype=torch.bfloat16, device=qk.device)
    _indexer_decode_prep[(rows,)](qk, q, q_weight, k_weight, key_state, rope_state, compressed, state_slots,
                                  group_locs, write_locs, positions,
                                  positions.stride(0) if positions.ndim == 2 else 0, cos_sin_cache, axis_map,
                                  head_dim, q_eps, k_eps, H=heads, D=head_dim, ROT=rotary_dim, RATIO=_RATIO,
                                  CACHE_STRIDE=cos_sin_cache.stride(0), num_warps=4)
    tokens, logits = _decode_select(q, cache, page_table, lengths, query_positions, sequence_lengths)
    return tokens, q, logits
