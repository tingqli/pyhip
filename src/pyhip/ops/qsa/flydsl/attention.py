"""One-call gfx942 BF16 D256 sparse attention; plans and scratch are private."""

from collections import OrderedDict
import math
import threading
from types import SimpleNamespace
import weakref

import numpy as np
import torch

from . import attention_direct as direct
from . import attention_prepare as prepare
from . import attention_union as union

__all__ = ["attention"]

# Below max(UNION_MIN_ROWS, UNION_MIN_ROWS_PER_HEAD * H/HK) query rows, direct-only beats
# union plus direct on gfx942, so no union plan is built.
UNION_MIN_ROWS, UNION_MIN_ROWS_PER_HEAD = 384, 64


def _aligned(nbytes):
    return -(-nbytes // 256) * 256


class _Arena:
    """Scratch shared by one stream's workspaces: every call rewrites what it reads.

    buffers[1] holds dense membership; each completed build clears what it set.
    Growth replaces the buffers; captured workspaces keep theirs alive.
    """

    def __init__(self):
        self.buffers = [None, None]
        self.generation = 0
        self.users = weakref.WeakSet()

    def reserve(self, sizes, device):
        grown = False
        for zeroed, size in enumerate(sizes):
            buffer = self.buffers[zeroed]
            if buffer is None or buffer.numel() < size:
                size = max(size, 0 if buffer is None else buffer.numel() * 5 // 4)
                self.buffers[zeroed] = (torch.zeros if zeroed else torch.empty)(
                    size, dtype=torch.uint8, device=device)
                grown = True
        if grown:
            self.generation += 1
            for user in list(self.users):
                user.release()


_arenas = {}


class _Workspace:
    def __init__(self, q, k, v, indices, query_lens, prefix_lens, scale):
        q_lens = np.asarray(query_lens, dtype=np.int64)
        lengths = q_lens + np.asarray(prefix_lens, dtype=np.int64)
        sequence = np.repeat(np.arange(len(q_lens)), q_lens)
        positions = np.arange(len(sequence)) + (lengths - np.cumsum(q_lens))[sequence]
        self.metadata = dict(
            query_lens=query_lens, prefix_lens=prefix_lens, scale=scale,
            kv_lens=prepare.upload(lengths, q.device),
            query_positions=prepare.upload(positions, q.device),
            query_sequence_ids=prepare.upload(sequence, q.device),
            max_seqlen_k=int(lengths.max(initial=0)),
        )
        self.captured = False
        self.block_indices = None
        inputs = SimpleNamespace(q=q, k=k, v=v, indices=indices, block_indices=None, **self.metadata)
        self.union = None
        if q.shape[0] >= max(UNION_MIN_ROWS, UNION_MIN_ROWS_PER_HEAD * (q.shape[1] // k.shape[1])):
            self.union = prepare.allocate_plan(inputs=inputs, query_tile=32, grid_multiplier=2, scratch=False)
        self.direct = direct.prepare(inputs=inputs, union=self.union, scratch=False)
        self.specs = [(self, "block_indices", (q.shape[0], 512), torch.int32, False)]
        if self.union is not None:
            self.specs += [(self.union, *spec) for spec in prepare.scratch_specs(self.union)]
        self.specs += [(self.direct, *spec) for spec in direct.scratch_specs(self.direct)]
        sizes = [0, 0]
        for *_, shape, dtype, zeroed in self.specs:
            sizes[zeroed] += _aligned(math.prod(shape) * dtype.itemsize)
        self.arena = _arenas.setdefault((q.device, torch.cuda.current_stream(q.device).cuda_stream), _Arena())
        self.stream = torch.cuda.current_stream(q.device)
        self.generation = None
        self.pinned = None
        self.hot = {}
        self.arena.reserve(sizes, q.device)

    def release(self):
        if self.pinned is None:
            for owner, name, *_ in self.specs:
                setattr(owner, name, None)
            self.direct.source_blocks = None
            self.generation = None
            self.hot = {}

    def attach(self):
        """Point the plans at this stream's scratch (again after arena growth); pin it for capture."""
        if self.pinned is None and self.generation != self.arena.generation:
            offsets = [0, 0]
            for owner, name, shape, dtype, zeroed in self.specs:
                nbytes = math.prod(shape) * dtype.itemsize
                view = self.arena.buffers[zeroed][offsets[zeroed]:offsets[zeroed] + nbytes]
                setattr(owner, name, view.view(dtype).view(shape))
                offsets[zeroed] += _aligned(nbytes)
            self.generation = self.arena.generation
            self.arena.users.add(self)
            self.hot = {}
        self.direct.source_blocks = self.block_indices
        if self.pinned is None and torch.cuda.is_current_stream_capturing():
            # Graph pointers must outlive arena growth.
            self.pinned = tuple(self.arena.buffers)

    def bind(self, q, k, v, indices):
        self.attach()
        return SimpleNamespace(q=q, k=k, v=v, indices=indices, block_indices=self.block_indices, **self.metadata)


class _Hot:
    """Replays one workspace binding: Triton K1-K3 without the JIT dispatcher and FlyDSL launchers
    with prebuilt arguments. Building it performs the current call through the normal paths."""

    def __init__(self, workspace, q, k, v, indices, out):
        inputs = workspace.bind(q, k, v, indices)
        self.stream = workspace.stream.cuda_stream
        self.prepare = prepare.replays(inputs=inputs, plan=workspace.union)
        self.compute = []
        if workspace.union is not None:
            union.run(inputs=inputs, plan=workspace.union, out=out)
            self.compute.append(union.launcher(inputs=inputs, plan=workspace.union, out=out))
        direct.run(inputs=inputs, prepared=workspace.direct, out=out)
        self.compute.append(direct.launcher(inputs=inputs, prepared=workspace.direct, out=out))
        self.compute = [launch for launch in self.compute if launch is not None]

    def prepare_call(self, indices):
        recover, *rest = self.prepare
        recover(self.stream, indices)
        for replay in rest:
            replay(self.stream)

    def compute_call(self, q, k, v, out):
        views = (q.view(-1), k.view(-1), v.view(-1), out.view(-1))
        for launch in self.compute:
            launch(*views)


_workspaces = OrderedDict()
_lock = threading.RLock()
_device = None
_warmed = set()


def _causal_indices(rows, device):
    """Each row's first min(visible // 4, 512) blocks, then its causal tail: a valid selection."""
    visible = torch.arange(1, rows + 1, dtype=torch.int32, device=device)[:, None]
    blocks = (visible // 4).clamp(max=512) * 4
    columns = torch.arange(2051, dtype=torch.int32, device=device)
    tail = torch.where(columns < blocks + visible % 4, columns - blocks + visible // 4 * 4, -1)
    return torch.where(columns < blocks, columns, tail).contiguous()


def _warm(device, heads, kv_heads, scale):
    """Compile every variant this head shape can use (no plan, each union tile: BQ16/BQ21 for H6,
    BQ32/BQ42 for H3) with full-causal dummies, so later layouts never JIT mid-serving."""
    group = heads // kv_heads
    rows = [1, max(UNION_MIN_ROWS, UNION_MIN_ROWS_PER_HEAD * group, 1)]
    if kv_heads == 1 and group in (6, 3):
        cus = torch.cuda.get_device_properties(device).multi_processor_count
        rows.append((21 if group == 6 else 42) * cus * 2)
    for n in rows:
        q = torch.zeros((n, heads, 256), dtype=torch.bfloat16, device=device)
        kv = torch.zeros((n, kv_heads, 256), dtype=torch.bfloat16, device=device)
        indices = _causal_indices(n, device)
        _Hot(_Workspace(q, kv, kv, indices, (n,), (0,), scale), q, kv, kv, indices, torch.empty_like(q))


def attention(q, k, v, indices, *, query_lens=None, prefix_lens=None, softmax_scale=None, out=None):
    """Return sparse attention for BF16 Q[M,H,256], K/V[N,HK,256].

    indices is contiguous int32[M,2051]: up to 512 complete four-token blocks,
    followed by the query's 0..3 causal tail tokens, then -1 padding. Block IDs
    are request-local, unique, and may be unordered. The selected set is exact.
    A selection that breaks this layout traps on the GPU, which aborts the process.

    Host query_lens/prefix_lens describe packed requests. For one request they
    default to (M,) and (N-M,). Caller need not create metadata or manage plans.
    Hot calls rebuild selection: per-layout plans share one private per-stream
    scratch arena. out is optional and must not alias Q/K/V. Warm the same
    stream/layout before graph capture. Captured workspaces stay pinned for
    graph pointer lifetime and share their stream's scratch: do not replay them
    concurrently with each other or with that stream's calls.
    The first call for each head shape (H, HK, scale) compiles every kernel variant that shape
    can use, so no later layout compiles; with empty caches this takes about half a minute.
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
    if q.device.type != "cuda" or torch.version.hip is None:
        raise ValueError("QSA attention requires ROCm/gfx942")
    for tensor in (q, k, v):
        if (tensor.dtype != torch.bfloat16 or tensor.device != q.device or tensor.layout != torch.strided
                or not tensor.is_contiguous() or tensor.shape[-1] != 256):
            raise ValueError("Q/K/V must be contiguous BF16 [tokens, heads, 256]")
        if tensor.requires_grad:
            raise ValueError("QSA attention is inference-only")
        if tensor.data_ptr() % 16 or tensor.numel() * tensor.element_size() >= 2**31:
            raise ValueError("Q/K/V require 16-byte alignment and byte spans < 2**31")
    if k.shape != v.shape or q.shape[1] <= 0 or k.shape[1] <= 0 or q.shape[1] % k.shape[1]:
        raise ValueError("K/V must match and Q heads must be a positive multiple of KV heads")
    if q.shape[1] // k.shape[1] > 16:
        raise ValueError("At most 16 Q heads per KV head fit the direct fallback wave")
    if _device != q.device and torch.cuda.get_device_properties(q.device).gcnArchName.split(":")[0] != "gfx942":
        raise ValueError("QSA attention requires gfx942")
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
            raise ValueError("FlyDSL attention requires one GPU per process")
        _device = q.device
        capturing = torch.cuda.is_current_stream_capturing()
        shape = (q.device, q.shape[1], k.shape[1], scale)
        if shape not in _warmed and not capturing:
            _warm(*shape)
            _warmed.add(shape)
        key = (q.device, torch._C._cuda_getCurrentRawStream(q.device.index), query_lens, prefix_lens,
               q.shape[1], k.shape[1], scale)
        workspace = _workspaces.get(key)
        if workspace is None:
            if capturing:
                raise RuntimeError("Warm attention on this stream and layout before graph capture")
            workspace = _Workspace(q, k, v, indices, query_lens, prefix_lens, scale)
            _workspaces[key] = workspace
            for stale in list(_workspaces):
                if len(_workspaces) <= 8:
                    break
                if stale != key and not _workspaces[stale].captured:
                    del _workspaces[stale]
        _workspaces.move_to_end(key)
        workspace.captured |= capturing
        workspace.attach()
        profiling = torch.autograd._profiler_enabled()
        hooked = prepare.hooked()
        # Triton replays are specialized on the indices' 16-byte alignment, like the JIT cache.
        replay_key = indices.data_ptr() % 16 == 0
        hot = None if hooked else workspace.hot.get(replay_key)
        if hot is not None:
            if profiling:
                with torch.profiler.record_function("pyhip_qsa.attention.prepare"):
                    hot.prepare_call(indices)
                with torch.profiler.record_function("pyhip_qsa.attention.compute"):
                    hot.compute_call(q, k, v, out)
            else:
                hot.prepare_call(indices)
                hot.compute_call(q, k, v, out)
        elif hooked or profiling:
            inputs = workspace.bind(q, k, v, indices)
            with torch.profiler.record_function("pyhip_qsa.attention.prepare"):
                prepare.run(inputs=inputs, plan=workspace.union)
            with torch.profiler.record_function("pyhip_qsa.attention.compute"):
                if workspace.union is not None:
                    union.run(inputs=inputs, plan=workspace.union, out=out)
                direct.run(inputs=inputs, prepared=workspace.direct, out=out)
        else:
            workspace.hot[replay_key] = _Hot(workspace, q, k, v, indices, out)
    return out
