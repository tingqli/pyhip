"""Linear D256 BF16 varlen attention on gfx942, without conversion kernels."""

import functools
import importlib
import math
import weakref
from numbers import Real

_VALIDATED = {}


@functools.cache
def _core_run():
    if __package__:
        return importlib.import_module(
            "..mha.mha_pa_bf16_256_linear_942", __package__
        ).run
    name = "experiments.attention.flydsl.mha.mha_pa_bf16_256_linear_942"
    try:
        return importlib.import_module(name).run
    except ModuleNotFoundError as exc:
        if not exc.name or not name.startswith(exc.name + "."):
            raise
    # Only standalone loading may need the project root; never add sibling dirs.
    import sys
    from pathlib import Path

    root = str(Path(__file__).resolve().parents[4])
    if root not in sys.path:
        sys.path.insert(0, root)
    return importlib.import_module(name).run


def _validate_metadata(tensors, shapes, limits):
    cq, ck = tensors[:2]
    table = tensors[2] if len(tensors) == 3 else None
    key = (tuple(map(id, tensors)), shapes, limits)
    try:
        version = tuple((t._version, t.device, tuple(t.shape), t.data_ptr()) for t in tensors)
    except RuntimeError as exc:
        raise ValueError("Metadata must have version counters; create it outside inference_mode().") from exc
    cached = _VALIDATED.get(key)
    if cached is not None and cached[1] == version and all(
        ref() is t for ref, t in zip(cached[0], tensors)
    ):
        return
    import torch

    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Unwarmed or changed metadata: warm this call outside graph capture first.")
    qb = cq.cpu().tolist()
    kb = qb if ck is cq else ck.cpu().tolist()
    nq, nk = shapes[0][0], shapes[1][0]
    max_q, max_k, causal, page = limits
    if (qb[0] != 0 or kb[0] != 0 or qb[-1] != nq
            or table is None and kb[-1] != nk):
        raise ValueError("Bounds must start at zero and end at Q/K token counts (K is logical when paged).")
    qlens = [end - start for start, end in zip(qb, qb[1:])]
    klens = [end - start for start, end in zip(kb, kb[1:])]
    if (any(not 0 <= n <= max_q for n in qlens)
            or any(not 0 <= n <= max_k for n in klens)):
        raise ValueError("Bounds must be monotonic and sequence lengths must not exceed their maxima.")
    if any(qn > 0 and (kn == 0 or causal and kn < qn) for qn, kn in zip(qlens, klens)):
        raise ValueError("Active Q requires nonempty KV; bottom-right causal attention requires KV >= Q.")
    if table is not None:
        if nk % page:
            raise ValueError("With block_table, physical K/V tokens must be a multiple of page_size.")
        for length, row in zip(klens, table.cpu().tolist()):
            count = (length + page - 1) // page
            if count > len(row) or any(p < 0 or p >= nk // page for p in row[:count]):
                raise ValueError("block_table has missing columns or invalid active physical page IDs.")
    _VALIDATED[key] = (
        tuple(weakref.ref(t, lambda _, key=key: _VALIDATED.pop(key, None)) for t in tensors),
        version,
    )


def _overlaps(a, b):
    na, nb = a.numel() * a.element_size(), b.numel() * b.element_size()
    # data_ptr includes view offsets; Python integers preserve full 64-bit pointers.
    return bool(na and nb and a.data_ptr() <= b.data_ptr() + nb - 1
                and b.data_ptr() <= a.data_ptr() + na - 1)


def flash_attn_varlen_func(
    q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
    min_seqlen_q=0, dropout_p=0.0, softmax_scale=None, logits_soft_cap=0.0,
    causal=False, window_size=(-1, -1, 0), bias=None, alibi_slopes=None,
    deterministic=False, return_lse=False, return_attn_probs=False,
    how_v3_bf16_cvt=1, block_table=None, out=None, cu_seqlens_q_padded=None,
    cu_seqlens_k_padded=None, sink_ptr=None, layout="linear", key_layout=None,
    num_waves=8, *, page_size=1, persistent=None, stream=None,
):
    """Return BF16 O[NQ,H,256], or (O, FP32 LSE[NQ,H]) with return_lse.

    Q/K/V are contiguous token-major BF16; H must be a positive multiple of HK.
    Pages are consecutive groups of physical K/V rows; table entries may permute
    them arbitrarily. Without a table, page_size imposes no padding requirement.
    Missing CK means self-attention with CQ boundaries and equal Q/K token counts.
    Causal masking is bottom-right. Q/K/V/O pointers must be 16-byte aligned.
    Warm metadata and the core specialization before capture; rewarm after edits.
    Metadata must track PyTorch versions. persistent=None selects True.
    """
    import torch

    unsupported = [name for name, value, default in (
        ("min_seqlen_q", min_seqlen_q, 0), ("dropout_p", dropout_p, 0.0),
        ("logits_soft_cap", logits_soft_cap, 0.0), ("deterministic", deterministic, False),
        ("return_attn_probs", return_attn_probs, False), ("how_v3_bf16_cvt", how_v3_bf16_cvt, 1),
    ) if not isinstance(value, Real) or value != default]
    unsupported += [name for name, value in (
        ("bias", bias), ("alibi_slopes", alibi_slopes), ("sink_ptr", sink_ptr),
        ("cu_seqlens_q_padded", cu_seqlens_q_padded), ("cu_seqlens_k_padded", cu_seqlens_k_padded),
    ) if value is not None]
    if (not isinstance(window_size, (tuple, list))
            or not all(isinstance(x, Real) for x in window_size)
            or tuple(window_size) not in ((-1, -1), (-1, -1, 0))):
        unsupported.append("window_size/sinks")
    if (not isinstance(layout, str) or layout.lower() != "linear"
            or key_layout is not None and (not isinstance(key_layout, str) or key_layout.lower() != "linear")):
        unsupported.append("nonlinear layout/key_layout")
    if unsupported:
        raise NotImplementedError("D256 varlen attention does not support: " + ", ".join(unsupported))
    if not isinstance(causal, bool) or not isinstance(return_lse, bool):
        raise ValueError("causal and return_lse must be bools")
    if persistent is not None and not isinstance(persistent, bool):
        raise ValueError("persistent must be bool or None")
    if type(num_waves) is not int or num_waves != 8:
        raise NotImplementedError("Only num_waves=8 is supported")
    if type(page_size) is not int or page_size not in (1, 4):
        raise NotImplementedError("Only page_size=1 or 4 is supported")
    for name, value in (("max_seqlen_q", max_seqlen_q), ("max_seqlen_k", max_seqlen_k)):
        if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value < 2**31:
            raise ValueError(f"{name} must be a nonnegative signed-int32 host bound")
    if softmax_scale is not None and (not isinstance(softmax_scale, Real) or isinstance(softmax_scale, bool)):
        raise ValueError("softmax_scale must be a host real scalar or None")
    scale = 1.0 / 16 if softmax_scale is None else float(softmax_scale)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("softmax_scale must be finite and positive")
    if not isinstance(q, torch.Tensor) or not q.is_cuda:
        raise ValueError("Q must be a device tensor on gfx942")
    device = q.device

    def check(name, tensor, dtype):
        if (not isinstance(tensor, torch.Tensor) or tensor.device != device or tensor.dtype != dtype
                or tensor.layout != torch.strided or not tensor.is_contiguous()):
            raise ValueError(f"{name} must be contiguous {dtype} on the Q GPU")
        if tensor.numel() * tensor.element_size() >= 2**31:
            raise ValueError(f"{name} byte span must be < 2**31")
        if tensor.requires_grad:
            raise NotImplementedError("Backward/autograd is not supported")

    for name, tensor in (("Q", q), ("K", k), ("V", v)):
        check(name, tensor, torch.bfloat16)
        if tensor.ndim != 3 or tensor.shape[-1] != 256:
            raise NotImplementedError("Q/K/V must be linear [tokens, heads, 256]")
        if tensor.data_ptr() % 16:
            raise ValueError(f"{name} data_ptr must be 16-byte aligned")
    if k.shape != v.shape or q.shape[1] <= 0 or k.shape[1] <= 0 or q.shape[1] % k.shape[1]:
        raise ValueError("K/V shapes must match; Q heads must be a positive multiple of KV heads")
    if cu_seqlens_k is None:
        if block_table is not None or q.shape[0] != k.shape[0]:
            raise ValueError("cu_seqlens_k=None requires equal-token self-attention without block_table")
        cu_seqlens_k = cu_seqlens_q
    metadata = (cu_seqlens_q, cu_seqlens_k) + (() if block_table is None else (block_table,))
    for name, tensor in zip(("cu_seqlens_q", "cu_seqlens_k", "block_table"), metadata):
        check(name, tensor, torch.int32)
    if (cu_seqlens_q.ndim != 1 or cu_seqlens_q.numel() < 1
            or cu_seqlens_k.shape != cu_seqlens_q.shape):
        raise ValueError("CQ/CK must have equal 1D shapes [B+1]")
    if block_table is not None and (block_table.ndim != 2 or block_table.shape[0] != cu_seqlens_q.numel() - 1):
        raise ValueError("block_table must have shape [B, max_pages]")
    if getattr(torch.cuda.get_device_properties(device), "gcnArchName", "").split(":", 1)[0] != "gfx942":
        raise NotImplementedError("This BF16 D256 backend requires gfx942")
    stream = torch.cuda.current_stream(device) if stream is None else stream
    if not isinstance(stream, torch.cuda.Stream) or stream.device != device:
        raise ValueError("stream must be a torch CUDA/ROCm stream on the Q GPU")
    with torch.cuda.device(device), torch.cuda.stream(stream):
        _validate_metadata(metadata, (tuple(q.shape), tuple(k.shape)),
                           (max_seqlen_q, max_seqlen_k, causal, page_size))
        if out is None:
            out = torch.empty_like(q)
        check("out", out, torch.bfloat16)
        if out.shape != q.shape or out.data_ptr() % 16:
            raise ValueError("out must match Q's shape and have a 16-byte-aligned data_ptr")
        lse = torch.empty(q.shape[:2], dtype=torch.float32, device=device) if return_lse else None
        sources = (q, k, v) + metadata + (() if lse is None else (lse,))
        if any(_overlaps(out, tensor) for tensor in sources):
            raise ValueError("out must not overlap Q/K/V, metadata, or LSE")
        return _core_run()(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                           out=out, lse=lse, page_size=page_size, causal=causal, softmax_scale=scale,
                           block_table=block_table, persistent=persistent is not False, stream=stream)


__all__ = ["flash_attn_varlen_func"]