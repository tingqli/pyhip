"""QSA indexer for gfx942 after ``index_qk_proj``: prefill/decode prep, logits and top-k.

BF16 norm/RoPE/mean steps (Triton) round like SGLang's eager chain, bit for bit.
Logits (FlyDSL MFMA, ``indexer_logits.py``) are the FP32 relu head sum of BF16 dots,
so only accumulation order differs from the Torch einsum reference. Top-k (FlyDSL,
``indexer_topk.py``) keeps the 512 largest ordered FP32 keys; ties keep the
lowest block ids. Each row holds the selected blocks' tokens in ascending block
order (PyHIP attention reads such rows without sorting), then 0..3 causal tail
tokens, then -1, matching SGLang's fixed-width token ABI. Decode (``decode_indexer``) fuses the CUDA-graph
q prep, pending-ring store and group compression into one bit-exact kernel (TARGET_VERIFY windows compress
in a second launch), then runs paged logits (``indexer_decode.py``) and the shared FlyDSL top-k/expand.
"""

from collections import OrderedDict
import math
import threading

import numpy as np
import torch
import triton
import triton.language as tl

from . import indexer_decode, indexer_logits, indexer_topk
from .attention_prepare import upload

__all__ = ["MAX_COMPRESSED_KEYS", "decode_indexer", "prefill_indexer"]

_LOGITS_BUDGET_BYTES = 256 * 1024 * 1024
# The top-k kernel keeps chosen block ids as uint16.
MAX_COMPRESSED_KEYS = 65536
# Logits CTA tile rows and keys per work item (a multiple of the kernel's 32-key block).
_BM, _KC = 128, indexer_logits.ITEM_KEYS
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


@triton.jit(do_not_specialize=["position_stride", "tokens"])
def _indexer_q_prep(QK, Q, Weight, KeyState, RopeState, Slots, Positions, position_stride,
                    Cache, Axis, Stats, tokens, n_cols, eps, TB: tl.constexpr, H: tl.constexpr,
                    D: tl.constexpr, ROT: tl.constexpr, CACHE_STRIDE: tl.constexpr):
    # position_stride is 0 for 1-D positions, so all three rope axes read the same row.
    first = tl.program_id(0) * TB
    # Per-row logits (min bits, ~max bits) start all ones; the logits kernel folds into them.
    lanes = tl.arange(0, 2)[None, :]
    rows = first + tl.arange(0, TB)[:, None]
    tl.store(Stats + rows * 2 + lanes, tl.full((TB, 2), -1, tl.int32), rows < tokens)
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


@triton.jit(do_not_specialize=["groups"])
def _indexer_k_compress(QK, Rows, WriteLocs, RopeMatrix, Cache, Axis, Weight, Compressed, groups, n_cols, eps,
                        GB: tl.constexpr, H: tl.constexpr, D: tl.constexpr, ROT: tl.constexpr,
                        RATIO: tl.constexpr, CACHE_STRIDE: tl.constexpr):
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


@triton.jit
def _ring_member(KeyState, loc, own, key, cols, D: tl.constexpr):
    # SGLang gathers after its ring store: a member in this row's own slot is this token's key.
    return tl.where(own, key, tl.load(KeyState + loc * D + cols)).to(tl.float32)


@triton.jit(do_not_specialize=["position_stride"])
def _indexer_decode_prep(QK, Q, QWeight, KWeight, KeyState, RopeState, Compressed, Slots, GroupLocs, WriteLocs,
                         Positions, position_stride, Cache, Axis, n_cols, q_eps, k_eps, H: tl.constexpr,
                         D: tl.constexpr, ROT: tl.constexpr, RATIO: tl.constexpr, CACHE_STRIDE: tl.constexpr,
                         COMPRESS: tl.constexpr):
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
    if COMPRESS:
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
        k = _norm_rope(mean[None, :], mean_other[None, :], n_cols, k_eps, KWeight, Cache, Axis, c0, c1, c2, 1, D,
                       ROT, CACHE_STRIDE)
        tl.store(Compressed + tl.load(WriteLocs + row).to(tl.int64) * D + cols[None, :], k)


@triton.jit
def _indexer_ring_compress(KeyState, RopeState, Compressed, GroupLocs, WriteLocs, Cache, Axis, KWeight, n_cols, k_eps,
                           D: tl.constexpr, ROT: tl.constexpr, RATIO: tl.constexpr, CACHE_STRIDE: tl.constexpr):
    # The fused decode compression after every row's ring store has landed (all members read back).
    row = tl.program_id(0)
    cols = tl.arange(0, D)
    other = _partner(cols, ROT)
    first = tl.load(GroupLocs + row * RATIO).to(tl.int64)
    acc = tl.load(KeyState + first * D + cols).to(tl.float32)
    acc_other = tl.load(KeyState + first * D + other).to(tl.float32)
    for member in tl.static_range(1, RATIO):
        loc = tl.load(GroupLocs + row * RATIO + member).to(tl.int64)
        acc = acc + tl.load(KeyState + loc * D + cols).to(tl.float32)
        acc_other = acc_other + tl.load(KeyState + loc * D + other).to(tl.float32)
    mean = (acc * (1.0 / RATIO)).to(tl.bfloat16).to(tl.float32)
    mean_other = (acc_other * (1.0 / RATIO)).to(tl.bfloat16).to(tl.float32)
    one = tl.zeros([1], dtype=tl.int64)
    c0 = tl.load(RopeState + first * 3) + one
    c1 = tl.load(RopeState + first * 3 + 1) + one
    c2 = tl.load(RopeState + first * 3 + 2) + one
    k = _norm_rope(mean[None, :], mean_other[None, :], n_cols, k_eps, KWeight, Cache, Axis, c0, c1, c2, 1, D, ROT,
                   CACHE_STRIDE)
    tl.store(Compressed + tl.load(WriteLocs + row).to(tl.int64) * D + cols[None, :], k)


class _Chunk:
    def __init__(self, row0, rows, width, items):
        self.row0, self.rows, self.width, self.items = row0, rows, width, items


class _Layout:
    """Host-derived per-forward plan; device tensors are read-only and shared."""

    def __init__(self, seq_lens, extend_lens, device):
        seq, extend = np.asarray(seq_lens, dtype=np.int64), np.asarray(extend_lens, dtype=np.int64)
        prefix, compressed = seq - extend, seq // _RATIO
        row_base = np.cumsum(extend) - extend
        self.rows = int(extend.sum())
        sequence = np.repeat(np.arange(len(seq)), extend)
        info = np.stack((np.arange(self.rows) - (row_base - prefix)[sequence], seq[sequence]), axis=1)
        counts = -(-extend // _BM)
        owner = np.repeat(np.arange(len(seq)), counts)
        start = (np.arange(len(owner)) - np.repeat(np.cumsum(counts) - counts, counts)) * _BM
        sizes = np.minimum(_BM, extend[owner] - start)
        firsts = row_base[owner] + start
        bounds = np.minimum((prefix[owner] + start + sizes) // _RATIO, compressed[owner])
        parts = [info.reshape(-1)]
        chunks, self.logits_elements = [], 0
        begin = 0
        while begin < len(owner):
            end, width = begin, 4
            while end < len(owner):
                grown = max(width, -(-int(bounds[end]) // 4) * 4)
                rows = int(firsts[end] + sizes[end] - firsts[begin])
                if end > begin and rows * grown * 4 > _LOGITS_BUDGET_BYTES:
                    break
                width, end = grown, end + 1
            row0 = int(firsts[begin])
            rows = int(firsts[end - 1] + sizes[end - 1]) - row0
            keep = np.arange(begin, end)[bounds[begin:end] > _TOPK]
            steps = -(-bounds[keep] // _KC)
            tile = np.repeat(keep, steps)
            key = (np.arange(len(tile)) - np.repeat(np.cumsum(steps) - steps, steps)) * _KC
            parts.append(np.stack((firsts[tile], firsts[tile] - row0, sizes[tile], owner[tile], key,
                                   np.minimum(key + _KC, bounds[tile])), axis=1).reshape(-1))
            chunks.append((row0, rows, width, len(tile)))
            self.logits_elements = max(self.logits_elements, rows * width)
            begin = end
        # One pinned asynchronous upload: a new layout never drains the stream.
        data = upload(np.concatenate(parts), device)
        offsets = np.cumsum([0] + [len(p) for p in parts])
        views = [data[offsets[i]:offsets[i + 1]] for i in range(len(parts))]
        self.row_info = views[0].view(-1, 2)
        self.chunks = [_Chunk(row0, rows, width, view.view(count, 6))
                       for (row0, rows, width, count), view in zip(chunks, views[1:])]


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
    dtypes = dtype if isinstance(dtype, tuple) else (dtype,)
    if tensor.dtype not in dtypes or (shape is not None and tuple(tensor.shape) != tuple(shape)) or (
            contiguous and not tensor.is_contiguous()):
        raise ValueError(f"{name} must be {'contiguous ' if contiguous else ''}{' or '.join(map(str, dtypes))} "
                         f"{shape}, got {tensor.dtype} {tuple(tensor.shape)}")


def _check_shared(qk, rows, heads, positions, state_slots, key_state, rope_state, compressed, cos_sin_cache,
                  axis_map, q_weight, k_weight):
    """Checks shared by prefill and decode; returns rotary_dim."""
    head_dim = q_weight.numel()
    if cos_sin_cache.ndim != 2 or cos_sin_cache.dtype not in (torch.bfloat16, torch.float32) or (
            not cos_sin_cache.is_contiguous()):
        raise ValueError("cos_sin_cache must be a contiguous BF16/FP32 [positions, rotary_dim] table")
    rotary_dim = cos_sin_cache.shape[1]
    if heads != 4 or head_dim != 128 or rotary_dim % 4 or not 0 < rotary_dim < head_dim:
        raise ValueError("The gfx942 indexer kernels require 4 heads, D128 and 0<rotary_dim<D")
    if any(n & (n - 1) for n in (rotary_dim // 2, head_dim - rotary_dim)):
        raise ValueError("rotary_dim/2 and head_dim-rotary_dim must be powers of two")
    if (positions.dtype != torch.int64 or positions.ndim not in (1, 2) or positions.stride(-1) != 1
            or positions.shape[-1] != rows or (positions.ndim == 2 and positions.shape[0] != 3)):
        raise ValueError("positions must be int64 [rows] or [3, rows] with a unit token stride")
    _check(qk, "qk", torch.bfloat16, (rows, (heads + 1) * head_dim))
    _check(state_slots, "state_slots", torch.int64, (rows,))
    _check(key_state, "key_state", torch.bfloat16, (key_state.shape[0], 1, head_dim))
    _check(rope_state, "rope_state", torch.int64, (key_state.shape[0], 3))
    _check(compressed, "compressed", torch.bfloat16, (compressed.shape[0], 1, head_dim))
    _check(axis_map, "axis_map", torch.int32, (rotary_dim // 2,))
    _check(q_weight, "q_weight", torch.bfloat16, (head_dim,))
    _check(k_weight, "k_weight", torch.bfloat16, (head_dim,))
    if compressed.numel() * compressed.element_size() >= 2**31:
        raise ValueError("The compressed key pool must be smaller than 2 GiB")
    return rotary_dim


def prefill_indexer(qk, *, heads, positions, logical_positions, state_slots, key_state, rope_state, write_locs,
                    member_rows, group_sequences, group_ends, rope_matrix, compressed, token_slot_table,
                    cos_sin_cache, axis_map, q_weight, k_weight, q_eps, k_eps, seq_lens, extend_lens, q_out=None):
    """Return int32 [T, 2051] token selections and update the QSA pending/compressed caches.

    qk is the contiguous BF16 ``index_qk_proj`` output [T, (heads + 1) * D] (heads=4, D=128). Host
    ``seq_lens``/``extend_lens`` describe the packed extend batch; every prefix (seq_len - extend_len)
    must be a multiple of 4, because a chunk compresses only groups whose four members are its own rows.
    ``positions`` are the RoPE positions (int64 [T] or [3, T]); request-local ``logical_positions``
    (int64 [T]) must equal ``prefix + i`` (checked on device: a mismatch traps on the GPU, which aborts
    the process).

    Updated in place: ``key_state`` (BF16 [slots, 1, D]) and ``rope_state`` (int64 [slots, 3]) at the
    rows' pending-ring ``state_slots`` (int64 [T]), and ``compressed`` (BF16 [compressed_slots, 1, D],
    under 2 GiB). ``write_locs`` (int32 [G]), ``member_rows``, ``group_sequences``, ``group_ends``
    (int64 [G]) and ``rope_matrix`` (int64 [T, 3]) are SGLang's extend write plan, including slot-0
    padding. ``token_slot_table`` (int32 [requests, >= max(seq_lens)]) holds each token's physical slot;
    a group's compressed key sits at its first token's slot // 4. ``cos_sin_cache`` (BF16/FP32
    [positions, rotary_dim]) and ``axis_map`` (int32 [rotary_dim / 2]) give the multi-axis RoPE,
    ``q_weight``/``k_weight`` (BF16 [D]) and ``q_eps``/``k_eps`` the Gemma RMSNorms. If given,
    ``q_out`` (BF16 [T, heads, D]) receives the normalized index queries.
    """
    device, head_dim = qk.device, q_weight.numel()
    rows = sum(extend_lens)
    if len(seq_lens) != len(extend_lens) or any(s < e or e < 0 for s, e in zip(seq_lens, extend_lens)):
        raise ValueError("Invalid host sequence/extend lengths")
    if any((s - e) % _RATIO for s, e in zip(seq_lens, extend_lens)):
        raise ValueError(f"prefill_indexer requires every prefix (seq_len - extend_len) to be aligned to "
                         f"compress_ratio={_RATIO}")
    if max(seq_lens) // _RATIO > MAX_COMPRESSED_KEYS or rows == 0:
        raise ValueError(f"Rows must be non-empty and have at most {MAX_COMPRESSED_KEYS} compressed keys")
    rotary_dim = _check_shared(qk, rows, heads, positions, state_slots, key_state, rope_state, compressed,
                               cos_sin_cache, axis_map, q_weight, k_weight)
    _check(logical_positions, "logical_positions", torch.int64, (rows,))
    groups = write_locs.numel()
    _check(write_locs, "write_locs", torch.int32, (groups,))
    for tensor, name in ((member_rows, "member_rows"), (group_sequences, "group_sequences"),
                         (group_ends, "group_ends")):
        _check(tensor, name, torch.int64, (groups,))
    _check(rope_matrix, "rope_matrix", torch.int64, (rows, 3))
    _check(token_slot_table, "token_slot_table", torch.int32, (len(seq_lens), token_slot_table.shape[1]), False)
    if token_slot_table.stride(1) != 1 or token_slot_table.shape[1] < max(seq_lens):
        raise ValueError("Unsupported token-slot table layout")
    tensors = (qk, positions, logical_positions, state_slots, key_state, rope_state, write_locs, member_rows,
               group_sequences, group_ends, rope_matrix, compressed, token_slot_table, cos_sin_cache, axis_map,
               q_weight, k_weight)
    if q_out is not None:
        _check(q_out, "q_out", torch.bfloat16, (rows, heads, head_dim))
        tensors += (q_out,)
    if any(t.device != device for t in tensors):
        raise ValueError("All indexer tensors must be on one device")

    layout = _layout(tuple(seq_lens), tuple(extend_lens), device)
    q = torch.empty((rows, heads, head_dim), dtype=torch.bfloat16, device=device) if q_out is None else q_out
    stats = torch.empty((rows, 2), dtype=torch.int32, device=device)
    _indexer_q_prep[(-(-rows // _TB),)](qk, q, q_weight, key_state, rope_state, state_slots, positions,
                                        positions.stride(0) if positions.ndim == 2 else 0, cos_sin_cache,
                                        axis_map, stats, rows, head_dim, q_eps, TB=_TB, H=heads,
                                        D=head_dim, ROT=rotary_dim, CACHE_STRIDE=cos_sin_cache.stride(0),
                                        num_warps=4)
    if groups:
        _indexer_k_compress[(-(-groups // _GB),)](qk, member_rows, write_locs, rope_matrix, cos_sin_cache, axis_map,
                                                  k_weight, compressed, groups, head_dim, k_eps, GB=_GB, H=heads,
                                                  D=head_dim, ROT=rotary_dim, RATIO=_RATIO,
                                                  CACHE_STRIDE=cos_sin_cache.stride(0), num_warps=4)
    out = torch.empty((rows, _WIDTH), dtype=torch.int32, device=device)
    logits = torch.empty(max(layout.logits_elements, 1) + _PAD, dtype=torch.float32, device=device)
    scale = float(np.float32(1.0) / np.float32(math.sqrt(head_dim)))
    for chunk in layout.chunks:
        view = logits[:chunk.rows * chunk.width].view(chunk.rows, chunk.width)
        # The first call compiles the logits kernel even without work, so no later request does.
        if chunk.items.shape[0] or q.device not in indexer_logits._COMPILED:
            indexer_logits.launch(q, compressed, token_slot_table, view, chunk.items, chunk.width, scale, stats)
        indexer_topk.launch(view, chunk.width, chunk.row0, chunk.rows, logical_positions, layout.row_info, out,
                            stats)
    return out


def decode_indexer(qk, *, positions, logical_positions, state_slots, key_state, rope_state, write_locs, group_locs,
                   compressed, page_table, lengths, cos_sin_cache, axis_map, q_weight, k_weight, q_eps, k_eps,
                   seq_lens, verify=False, q_out=None, logits_out=None):
    """CUDA-graph decode counterpart of ``prefill_indexer``: int32 [rows, 2051] token selections.

    One row per query token; ``qk``, ``positions``, ``state_slots``, the caches and the RoPE/norm parameters
    are as in ``prefill_indexer``. One Triton kernel reproduces SGLang's unfused (BF16 cos/sin) q norm/RoPE,
    the pending-ring key and RoPE-position store at ``state_slots`` and the fixed-shape compression of
    ``group_locs`` (int32 [rows, 4] ring slots, oldest member first) into ``write_locs`` (int32 [rows], the
    inert slot 0 when the row completes no group) bit for bit. Rows must belong to distinct requests (each
    row only sees its own ring store) unless ``verify=True``: a TARGET_VERIFY window has several rows per
    request, so the compression runs as a second launch after every row's ring store, in SGLang's order.

    Selection reads ``compressed`` as 16-key pages: page p of a row holds keys 16 * page_table[row, p] +
    [0, 16) (``page_table`` int32 [rows, pages], at most 65536 keys per row). ``lengths``, ``logical_positions``
    and ``seq_lens`` (int32 [rows]; ``logical_positions`` may be int64) are each row's compressed-key count,
    query position and sequence length. Everything stays on device, so the call is CUDA-graph capturable
    after an eager warmup. If given, ``q_out`` (BF16 [rows, 4, D]) receives the normalized index queries and
    ``logits_out`` (FP32 [rows, 16 * pages] whose storage extends 512 values past it) the logits.
    """
    if qk.ndim != 2 or page_table.ndim != 2 or 0 in (qk.shape[0], page_table.shape[1]):
        raise ValueError("qk and page_table must be 2-D with at least one row and one page")
    rows, heads, head_dim = qk.shape[0], 4, q_weight.numel()
    width = page_table.shape[1] * 16
    if width > MAX_COMPRESSED_KEYS:
        raise ValueError(f"Decode supports at most {MAX_COMPRESSED_KEYS} compressed keys per row")
    rotary_dim = _check_shared(qk, rows, heads, positions, state_slots, key_state, rope_state, compressed,
                               cos_sin_cache, axis_map, q_weight, k_weight)
    for tensor, name in ((write_locs, "write_locs"), (lengths, "lengths")):
        _check(tensor, name, torch.int32, (rows,))
    _check(logical_positions, "logical_positions", (torch.int32, torch.int64), (rows,), False)
    _check(seq_lens, "seq_lens", torch.int32, (rows,), False)
    _check(group_locs, "group_locs", torch.int32, (rows, _RATIO))
    _check(page_table, "page_table", torch.int32, (rows, page_table.shape[1]))
    tensors = (qk, positions, logical_positions, state_slots, key_state, rope_state, write_locs, group_locs,
               compressed, page_table, lengths, cos_sin_cache, axis_map, q_weight, k_weight, seq_lens)
    if q_out is not None:
        _check(q_out, "q_out", torch.bfloat16, (rows, heads, head_dim))
        tensors += (q_out,)
    if logits_out is not None:
        _check(logits_out, "logits_out", torch.float32, (rows, width))
        if logits_out.untyped_storage().nbytes() < (logits_out.storage_offset() + rows * width + _PAD) * 4:
            raise ValueError(f"logits_out storage must extend {_PAD} values past the logits")
        tensors += (logits_out,)
    if any(t.device != qk.device for t in tensors):
        raise ValueError("All indexer tensors must be on one device")
    q = torch.empty((rows, heads, head_dim), dtype=torch.bfloat16, device=qk.device) if q_out is None else q_out
    logits = logits_out
    if logits is None:
        logits = torch.empty(rows * width + _PAD, dtype=torch.float32, device=qk.device)[:rows * width]
        logits = logits.view(rows, width)
    _indexer_decode_prep[(rows,)](qk, q, q_weight, k_weight, key_state, rope_state, compressed, state_slots,
                                  group_locs, write_locs, positions,
                                  positions.stride(0) if positions.ndim == 2 else 0, cos_sin_cache, axis_map,
                                  head_dim, q_eps, k_eps, H=heads, D=head_dim, ROT=rotary_dim, RATIO=_RATIO,
                                  CACHE_STRIDE=cos_sin_cache.stride(0), COMPRESS=not verify, num_warps=4)
    if verify:
        _indexer_ring_compress[(rows,)](key_state, rope_state, compressed, group_locs, write_locs, cos_sin_cache,
                                        axis_map, k_weight, head_dim, k_eps, D=head_dim, ROT=rotary_dim,
                                        RATIO=_RATIO, CACHE_STRIDE=cos_sin_cache.stride(0), num_warps=4)
    out = torch.empty((rows, _WIDTH), dtype=torch.int32, device=qk.device)
    indexer_decode.launch(q, compressed, page_table, lengths, logits,
                          float(np.float32(1.0) / np.float32(math.sqrt(head_dim))))
    indexer_topk.launch_decode(logits, lengths.contiguous(), logical_positions.contiguous(), seq_lens.contiguous(),
                               out)
    return out
