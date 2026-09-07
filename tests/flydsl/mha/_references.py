"""Lazy, explicit AITER references; never suppress a correctness failure."""

import functools
import contextlib
import hashlib
import importlib.util
from pathlib import Path
import sys
import tempfile

import torch


class ReferenceUnavailable(RuntimeError):
    """The optional package or a specific native specialization is missing."""


REQUESTED_SOURCES = {
    "bf16": ("tests/flydsl/test_attn_8wave_32x32_lkgv.py",
             "c0880420cd10a797c59087d4f73e942aa237020c3f0793b2fa0bcd9a6ac0776a"),
    "fp8": ("tests/flydsl/pa_8wave/pa_prefill_8w32x32.py",
            "620209a023ccb5ea566489774d19edae880dfbcee298613233ed6f01f3b59849"),
}


def requested_identity(kind):
    relative, expected = REQUESTED_SOURCES[kind]
    path = Path(__file__).resolve().parents[3] / relative
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(f"requested reference source changed: {relative}")
    return {"path": relative, "sha256": actual}


@functools.cache
def _requested_module(kind):
    from flydsl.utils import env
    identity = requested_identity(kind)
    path = Path(__file__).resolve().parents[3] / identity["path"]
    saved = {key: getattr(env.debug, key) for key in ("dump_ir", "dump_asm", "enable_debug_info", "dump_dir")}
    device = torch.get_default_device()
    try:
        # The reference imports enable dump/set_device; isolate those side effects.
        with tempfile.TemporaryDirectory(prefix="mha-reference-import-") as temp, contextlib.chdir(temp):
            spec = importlib.util.spec_from_file_location(f"mha_requested_{kind}", path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            return module
    finally:
        torch.set_default_device(device)
        for key, value in saved.items():
            setattr(env.debug, key, value)


def requested_reference_call(case, causal, softmax_scale=None):
    """User-selected source, same logical inputs; layout preparation is untimed.

    Unsupported dense semantics are explicitly unavailable, never resized or
    converted to a different attention problem. Both paths leave LSE disabled.
    """
    kind = "fp8" if case.q.dtype == torch.float8_e4m3fnuz else "bf16"
    if case.window_left >= 0 or case.sinks is not None or case.q_offset or case.table_offset:
        raise ReferenceUnavailable("requested references do not support SWA/sink/prefix offsets")
    if softmax_scale is not None and softmax_scale != case.dq**-0.5:
        raise ReferenceUnavailable("requested reference uses the default softmax scale")
    if kind == "bf16":
        if (case.q.dtype != torch.bfloat16 or causal or case.heads != case.kv_heads
                or case.dq != 128 or case.dv != 128 or len(case.q_lens) != 1
                or case.q_lens[0] <= 0 or case.q_lens[0] % 256
                or case.kv_lens[0] <= 0 or case.kv_lens[0] % 32):
            raise ReferenceUnavailable("dense BF16 reference requires B1/H=HK/Dq=Dv128/NC/Q%256=0/KV%32=0")
        if not all(bool(torch.all(s == 1)) for s in (case.qs, case.ks, case.vs)):
            raise ReferenceUnavailable("dense BF16 reference requires unit descales")
        module = _requested_module(kind)
        keys, values = case.logical_kv()
        q = case.q.transpose(0, 1).contiguous()
        k = keys[0].transpose(0, 1).to(torch.bfloat16).contiguous()
        v = values[0].transpose(0, 1).to(torch.bfloat16).contiguous()
        v = v.reshape(case.heads, case.kv_lens[0] // 8, 8, case.dv).transpose(2, 3).contiguous()
        out = torch.empty_like(q)
        kernel = module.MHA(case.heads, case.dq, 256, 32)
        stream = torch.cuda.current_stream()

        def call():
            kernel(q, k, v, out, stream)
            return out.transpose(0, 1)
    else:
        if case.page not in (32, 64, 128) or any(k < q for q, k in zip(case.q_lens, case.kv_lens) if causal):
            raise ReferenceUnavailable("FP8 BN32 reference requires supported pages and causal KV>=Q")
        module = _requested_module(kind)
        kernel = module.PagedAttention(case.heads, case.kv_heads, case.dq, case.dv, case.page, causal, case.mode)
        out = torch.empty((case.q.shape[0], case.heads, case.dv), dtype=torch.bfloat16, device=case.q.device)

        def call():
            return kernel(case.q, case.k, case.v, case.cq, case.ck, case.indptr, case.indices,
                          max(case.q_lens), max(case.kv_lens), causal, case.qs, case.ks, case.vs,
                          case.last, out=out)
    call.reference_source = requested_identity(kind)
    return call


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


def aiter_linear_call(case, causal, out=None, softmax_scale=None, *, linear_kv=None):
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
    if linear_kv is None:
        keys, values = case.logical_kv()
        k = torch.cat(keys).to(torch.bfloat16).contiguous()
        v = torch.cat(values).to(torch.bfloat16).contiguous()
    else:
        k, v = linear_kv
        for tensor, dim in ((k, case.dq), (v, case.dv)):
            if (tensor.shape != (sum(case.kv_lens), case.kv_heads, dim)
                    or tensor.dtype != torch.bfloat16 or tensor.device != case.q.device or not tensor.is_contiguous()):
                raise ValueError("linear KV must be preallocated contiguous BF16 THD")
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


def aiter_gather_call(case, causal, softmax_scale=None):
    """One timed closure executes full-KV gather + CK, with no cached gather."""
    if __package__:
        from ._gather import gather_kv_call
    else:
        from _gather import gather_kv_call
    gather, linear_kv = gather_kv_call(case)
    # Full-cache identity check is untimed and does not replace timed gathers.
    gather()
    keys, values = case.logical_kv()
    for actual, expected in zip(linear_kv, (torch.cat(keys), torch.cat(values))):
        torch.testing.assert_close(actual.float(), expected, rtol=0, atol=0)
    linear = aiter_linear_call(case, causal, softmax_scale=softmax_scale, linear_kv=linear_kv)

    def call():
        gather()
        return linear()
    call.aiter_entry = "full paged KV Triton gather + aiter.ops.mha.mha_varlen_fwd (CK, no LSE)"
    call.workspace_ptrs = [t.data_ptr() for t in linear_kv]
    call.extra_io_bytes = 2 * sum(t.numel() * t.element_size() for t in linear_kv)
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