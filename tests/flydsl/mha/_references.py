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


def aiter_linear_call(case, causal, out=None, softmax_scale=None):
    """CK window/sink reference with linear KV prepared outside timing."""
    _aiter()
    try:
        from aiter.ops.mha import mha_varlen_fwd
    except (ImportError, OSError) as exc:
        raise ReferenceUnavailable(f"AITER CK entry unavailable: {exc}") from exc
    if case.q.dtype != torch.bfloat16 or case.q_offset or case.table_offset:
        raise ReferenceUnavailable("AITER linear reference requires dense BF16 sequences")
    if not all(bool(torch.all(scale == 1)) for scale in (case.qs, case.ks, case.vs)):
        raise ReferenceUnavailable("AITER linear BF16 comparison requires unit descales")
    keys, values = case.logical_kv()
    k = torch.cat(keys).to(torch.bfloat16).contiguous()
    v = torch.cat(values).to(torch.bfloat16).contiguous()
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


def aiter_reference_call(case, causal, out=None, softmax_scale=None):
    """AITER reference for the quick benchmark; preparation is outside timing.

    Full BF16 uses AITER's public varlen router, not a forced OPUS backend.
    Window/sink requests use CK explicitly: the local public router treats W0
    as full attention and some ASM paths do not implement sink logits.
    """
    if case.window_left >= 0 or case.sinks is not None:
        call = aiter_linear_call(case, causal, out, softmax_scale)
        call.aiter_entry = "aiter.ops.mha.mha_varlen_fwd (CK window/sink)"
        return call
    aiter = _aiter()
    if case.q.dtype != torch.bfloat16 or case.q_offset or case.table_offset:
        raise ReferenceUnavailable("AITER quick comparison requires BF16 with unpadded prefix metadata")
    if not all(bool(torch.all(scale == 1)) for scale in (case.qs, case.ks, case.vs)):
        raise ReferenceUnavailable("AITER quick comparison requires unit descales; no silent conversion")
    keys, values = case.logical_kv()
    q = case.q.contiguous()
    k = torch.cat(keys).to(torch.bfloat16).contiguous()
    v = torch.cat(values).to(torch.bfloat16).contiguous()
    if out is None:
        out = torch.empty(q.shape[0], case.heads, case.dv, device=q.device, dtype=torch.bfloat16)
    max_q, max_k = max(case.q_lens, default=0), max(case.kv_lens, default=0)

    def call():
        return aiter.flash_attn_varlen_func(
            q, k, v, case.cq, case.ck, max_q, max_k, causal=causal,
            softmax_scale=softmax_scale, out=out,
        )

    call.aiter_entry = "aiter.flash_attn_varlen_func (public router)"
    return call