"""Lazy, explicit AITER references; never suppress a correctness failure."""

import functools

import torch


class ReferenceUnavailable(RuntimeError):
    """The optional package or a specific native specialization is missing."""


@functools.cache
def _aiter():
    try:
        import aiter
    except (ImportError, OSError) as exc:
        raise ReferenceUnavailable(f"AITER import unavailable: {exc}") from exc
    return aiter


def probe_reference(call):
    try:
        out = call()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return out
    except (ImportError, OSError) as exc:
        # Optional backends can import/JIT their native module lazily on the
        # first call. Missing dependencies are not numerical test failures.
        raise ReferenceUnavailable(f"AITER backend dependency unavailable: {exc}") from exc
    except RuntimeError as exc:
        if "no matching kernel found" not in str(exc).lower():
            raise
        raise ReferenceUnavailable(str(exc)) from exc


def aiter_call(case, causal, out=None, softmax_scale=None):
    """Same paged tensors/metadata, with no hidden gather or timing fallback.

    The optional direct-paged CK specialization may not exist in a given AITER
    build. Unit BF16 descales and FP8 native descales are passed explicitly.
    """
    aiter = _aiter()
    if case.sinks is not None and not causal:
        raise ReferenceUnavailable("AITER paged noncausal sink semantics are not supported")
    if out is None:
        out = torch.empty(case.q.shape[0], case.heads, case.dv, device=case.q.device, dtype=torch.bfloat16)
    if case.q.dtype == torch.bfloat16:
        if not all(bool(torch.all(scale == 1)) for scale in (case.qs, case.ks, case.vs)):
            raise ReferenceUnavailable("AITER BF16 comparison requires unit descales (no silent input conversion)")
        scales = {}
    else:
        if case.qs.numel() != 1:
            raise ReferenceUnavailable("AITER paged FP8 supports scalar Q descale, not per-token scales")
        scales = {"q_descale": case.qs, "k_descale": case.ks, "v_descale": case.vs}

    def call():
        return aiter.mha_batch_prefill_func(
            case.q, case.k, case.v, case.cq, case.indptr, case.indices,
            max(case.q_lens, default=0), max(case.kv_lens, default=0),
            causal=causal, softmax_scale=softmax_scale, window_size=(case.window_left, -1),
            sink_ptr=case.sinks, kv_last_page_lens=case.last, out=out, **scales,
        )
    return call


def aiter_linear_call(case, causal, out=None, softmax_scale=None, *, linear_kv=None):
    """Explicit CK linear BF16 comparison; gather is OUTSIDE timing.

    This is not a substitute measurement for a missing 5D specialization.
    Call CK directly, not the public router which can select FlyDSL/ASM.
    ``linear_kv`` allows the explicit gather+linear benchmark to reuse its
    preallocated gather destinations; the caller must populate them first.
    """
    _aiter()
    try:
        from aiter.ops.mha import mha_varlen_fwd
    except (ImportError, OSError) as exc:
        raise ReferenceUnavailable(f"AITER CK entry unavailable: {exc}") from exc
    if case.q.dtype != torch.bfloat16 or case.q_offset or case.table_offset:
        raise ReferenceUnavailable("AITER linear reference requires dense BF16 sequences")
    if not all(bool(torch.all(scale == 1)) for scale in (case.qs, case.ks, case.vs)):
        raise ReferenceUnavailable("AITER linear BF16 comparison requires unit descales")
    if linear_kv is None:
        keys, values = case.logical_kv()
        k = torch.cat(keys).to(torch.bfloat16).contiguous()
        v = torch.cat(values).to(torch.bfloat16).contiguous()
    else:
        if len(linear_kv) != 2:
            raise ValueError("linear_kv must contain K and V")
        k, v = linear_kv
        for name, tensor, dim in (("K", k, case.dq), ("V", v, case.dv)):
            if (not isinstance(tensor, torch.Tensor) or tensor.shape != (sum(case.kv_lens), case.kv_heads, dim)
                    or tensor.dtype != torch.bfloat16 or tensor.device != case.q.device or not tensor.is_contiguous()):
                raise ValueError(f"linear {name} must be contiguous BF16 THD on the input device")
    if out is None:
        out = torch.empty(case.q.shape[0], case.heads, case.dv, device=case.q.device, dtype=torch.bfloat16)

    def call():
        result = mha_varlen_fwd(
            case.q, k, v, case.cq, case.ck, max(case.q_lens), max(case.kv_lens),
            0, 0.0, case.dq**-0.5 if softmax_scale is None else softmax_scale,
            0.0, False, causal, case.window_left, 0 if causal else -1,
            0, False, False, sink_ptr=case.sinks, out=out,
        )
        return result[0]
    return call


def aiter_opus_call(case, causal, out=None, softmax_scale=None):
    """Explicit original gfx950 OPUS comparator, never a router fallback."""
    if case.window_left >= 0 or case.sinks is not None:
        raise ValueError("OPUS D128/D192 comparison has no SWA or sink support")
    if case.q.dtype != torch.bfloat16 or case.q_offset or case.table_offset:
        raise ReferenceUnavailable("OPUS comparison requires dense BF16 sequences")
    if not all(bool(torch.all(scale == 1)) for scale in (case.qs, case.ks, case.vs)):
        raise ReferenceUnavailable("OPUS comparison requires unit descales")
    if case.dq == 128 and len(case.q_lens) != 1:
        raise ReferenceUnavailable("OPUS D128 comparison requires one sequence")
    _aiter()
    try:
        from aiter.ops.mha import fmha_fwd_bf16_opus_fwd, fmha_fwd_bf16_opus_varlen_fwd
    except (ImportError, OSError) as exc:
        raise ReferenceUnavailable(f"AITER OPUS entry unavailable: {exc}") from exc
    keys, values = case.logical_kv()
    k = torch.cat(keys).to(torch.bfloat16).contiguous()
    v = torch.cat(values).to(torch.bfloat16).contiguous()
    if out is None:
        out = torch.empty(case.q.shape[0], case.heads, case.dv, device=case.q.device, dtype=torch.bfloat16)
    scale = case.dq**-0.5 if softmax_scale is None else softmax_scale
    if case.dq == 128:
        q4, k4, v4, out4 = (tensor.unsqueeze(0) for tensor in (case.q, k, v, out))

        def call():
            fmha_fwd_bf16_opus_fwd(q4, k4, v4, scale, causal, out=out4)
            return out
    else:
        def call():
            fmha_fwd_bf16_opus_varlen_fwd(
                case.q, k, v, scale, causal, case.cq, case.ck,
                max(case.q_lens), max(case.kv_lens), out=out,
            )
            return out
    return call