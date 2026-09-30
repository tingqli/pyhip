"""One-call gfx942 BF16 D256 sparse attention; plans and scratch are private."""

from collections import OrderedDict
from itertools import accumulate
import math
import threading
from types import SimpleNamespace

import torch
import triton
import triton.language as tl

from . import dense, direct, union

__all__ = ["qsa"]


@triton.jit
def _qsa_recover_blocks(Indices, Positions, Lengths, SequenceIds, Blocks, Errors):
    row = tl.program_id(0)
    position = tl.load(Positions + row)
    length = tl.load(Lengths + tl.load(SequenceIds + row))
    visible = position + 1
    complete = tl.minimum(visible // 4, 512)
    columns = tl.arange(0, 512)
    a = tl.load(Indices + row * 2051 + columns * 4)
    b = tl.load(Indices + row * 2051 + columns * 4 + 1)
    c = tl.load(Indices + row * 2051 + columns * 4 + 2)
    d = tl.load(Indices + row * 2051 + columns * 4 + 3)
    full = columns < complete
    valid = (a >= 0) & (a % 4 == 0) & (b == a + 1) & (c == a + 2) & (d == a + 3)
    valid &= (d < visible) & (d < length)
    blocks = tl.where(full & valid, a // 4, -1)
    ordered = tl.sort(tl.where(full, blocks, 2147483647), descending=False)
    previous = tl.gather(ordered, tl.maximum(columns - 1, 0), axis=0)
    duplicate = full & (columns > 0) & (ordered == previous)

    slots = tl.arange(0, 4096)
    tokens = tl.load(Indices + row * 2051 + slots, slots < 2051, -1)
    count = complete * 4 + visible % 4
    tail = (slots >= complete * 4) & (slots < count)
    tail_token = (visible // 4) * 4 + slots - complete * 4
    bad_slots = (slots < 2051) & tl.where(
        slots >= count,
        tokens != -1,
        (tokens < 0) | (tokens >= visible) | (tokens >= length) | (tail & (tokens != tail_token)),
    )
    error = (position < 0) | (visible > length)
    error |= tl.sum((full & ~valid).to(tl.int32), 0) > 0
    error |= tl.sum(duplicate.to(tl.int32), 0) > 0
    error |= tl.sum(bad_slots.to(tl.int32), 0) > 0
    tl.store(Blocks + row * 512 + columns, blocks)
    tl.store(Errors + row, error.to(tl.int32))


class _Workspace:
    def __init__(self, q, k, v, indices, query_lens, prefix_lens, scale):
        lengths = tuple(n + p for n, p in zip(query_lens, prefix_lens))
        kw = {"dtype": torch.int32, "device": q.device}
        self.metadata = dict(
            query_lens=query_lens, prefix_lens=prefix_lens, scale=scale,
            cu_q=torch.tensor(tuple(accumulate(query_lens, initial=0)), **kw),
            cu_k=torch.tensor(tuple(accumulate(lengths, initial=0)), **kw),
            kv_lens=torch.tensor(lengths, **kw),
            query_positions=torch.tensor([p + i for n, p in zip(query_lens, prefix_lens) for i in range(n)], **kw),
            query_sequence_ids=torch.tensor([s for s, n in enumerate(query_lens) for _ in range(n)], **kw),
            block_indices=torch.empty((q.shape[0], 512), **kw),
            max_seqlen_q=max(query_lens, default=0), max_seqlen_k=max(lengths, default=0),
        )
        self.errors = torch.empty(q.shape[0], **kw)
        self.captured = False
        inputs = self.bind(q, k, v, indices)
        self.dense = dense.prepare(inputs=inputs)
        self.union = self.direct = None
        if sum(self.dense.query_counts) != q.shape[0]:
            self.union = union.allocate_plan(inputs=inputs, query_tile=32, grid_multiplier=2,
                                            max_union_inflation=4.0, skip_counts=self.dense.query_counts)
            self.direct = direct.prepare(inputs=inputs, skip_counts=self.dense.query_counts, union=self.union)

    def bind(self, q, k, v, indices):
        return SimpleNamespace(q=q, k=k, v=v, indices=indices, **self.metadata)


_workspaces = OrderedDict()
_lock = threading.RLock()
_device = None


def qsa(q, k, v, indices, *, query_lens=None, prefix_lens=None, softmax_scale=None, out=None):
    """Return sparse attention for BF16 Q[M,H,256], K/V[N,HK,256].

    indices is contiguous int32[M,2051]: up to 512 complete four-token blocks,
    followed by the query's 0..3 causal tail tokens, then -1 padding. Block IDs
    are request-local, unique, and may be unordered. The selected set is exact.

    Host query_lens/prefix_lens describe packed requests. For one request they
    default to (M,) and (N-M,). Caller need not create metadata or manage plans.
    Hot calls rebuild selection using private per-layout/per-stream scratch;
    out is optional and must not alias Q/K/V. Warm the same stream/layout before
    graph capture. Captured workspaces stay pinned for graph pointer lifetime.
    This FlyDSL runtime supports one GPU per process (TP uses separate workers).
    """
    global _device
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("Q/K/V must be contiguous BF16 [tokens, heads, 256]")
    if isinstance(softmax_scale, (torch.Tensor, bool)):
        raise ValueError("softmax_scale must be a host scalar or None")
    scale = 0.0625 if softmax_scale is None else float(softmax_scale)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("softmax_scale must be finite and positive")
    query_lens = (q.shape[0],) if query_lens is None else tuple(query_lens)
    prefix_lens = ((k.shape[0] - q.shape[0],) if len(query_lens) == 1 else (0,) * len(query_lens)) if prefix_lens is None else tuple(prefix_lens)
    if (len(query_lens) != len(prefix_lens)
            or any(type(n) is not int or n < 0 or n >= 2**31 for n in (*query_lens, *prefix_lens))
            or sum(query_lens) != q.shape[0] or sum(query_lens) + sum(prefix_lens) != k.shape[0]):
        raise ValueError("Host query/prefix lengths must describe the packed Q/K/V")
    inputs = SimpleNamespace(q=q, k=k, v=v, scale=scale)
    dense._check_qkv(inputs)
    if q.shape[1] // k.shape[1] > 16:
        raise ValueError("At most 16 Q heads per KV head fit the direct fallback wave")
    if torch.cuda.get_device_properties(q.device).gcnArchName.split(":")[0] != "gfx942":
        raise ValueError("QSA requires gfx942")
    if (indices.shape != (q.shape[0], 2051) or indices.dtype != torch.int32
            or indices.device != q.device or not indices.is_contiguous()
            or indices.numel() * indices.element_size() >= 2**31):
        raise ValueError("indices must be contiguous device int32 [M,2051] with byte span < 2**31")
    if out is None:
        out = torch.empty_like(q)
    if (out.shape != q.shape or out.dtype != q.dtype or out.device != q.device
            or not out.is_contiguous() or out.data_ptr() % 16 or out.requires_grad):
        raise ValueError("out must be aligned, contiguous, inference-only and match Q")
    begin, end = out.data_ptr(), out.data_ptr() + out.numel() * out.element_size()
    for source in (q, k, v, indices):
        if begin < source.data_ptr() + source.numel() * source.element_size() and source.data_ptr() < end:
            raise ValueError("out must not overlap Q/K/V or indices")
    if q.shape[0] == 0:
        return out
    with _lock, torch.cuda.device(q.device):
        if _device is not None and _device != q.device:
            raise ValueError("FlyDSL QSA requires one GPU per process")
        _device = q.device
        stream = torch.cuda.current_stream(q.device)
        key = (q.device, stream.cuda_stream, query_lens, prefix_lens, q.shape[1], k.shape[1], scale)
        workspace = _workspaces.get(key)
        capturing = torch.cuda.is_current_stream_capturing()
        if workspace is None:
            if capturing:
                raise RuntimeError("Warm QSA on this stream and layout before graph capture")
            workspace = _Workspace(q, k, v, indices, query_lens, prefix_lens, scale)
            _workspaces[key] = workspace
            for stale in list(_workspaces):
                if len(_workspaces) <= 8:
                    break
                if stale != key and not _workspaces[stale].captured:
                    del _workspaces[stale]
        _workspaces.move_to_end(key)
        workspace.captured |= capturing
        inputs = workspace.bind(q, k, v, indices)
        with torch.profiler.record_function("pyhip_qsa.recover_blocks"):
            _qsa_recover_blocks[(q.shape[0],)](
                indices, inputs.query_positions, inputs.kv_lens,
                inputs.query_sequence_ids, inputs.block_indices, workspace.errors, num_warps=4)
            torch._assert_async((workspace.errors == 0).all(), "Invalid compressed QSA token/block/tail ABI")
        with torch.profiler.record_function("pyhip_qsa.plan_rebuild"):
            if workspace.union is not None:
                union.rebuild_plan(inputs=inputs, plan=workspace.union)
                direct.rebuild_plan(inputs=inputs, plan=workspace.direct)
        with torch.profiler.record_function("pyhip_qsa.attention"):
            dense.run(inputs=inputs, prepared=workspace.dense, out=out)
            if workspace.union is not None:
                union.run(inputs=inputs, plan=workspace.union, out=out)
                direct.run(inputs=inputs, prepared=workspace.direct, out=out)
    return out
