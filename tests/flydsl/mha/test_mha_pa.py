"""BF16 MHA, FP8 MHA and SWA: explicit performance sets and functional pytest.

CLI: --suite bf16-mha|fp8-mha|swa|all [--case ID ...] [--list]
pytest: functional tests by default; PYHIP_MHA_PERF=1 enables performance tests.
"""

import argparse
from contextlib import contextmanager
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import fcntl
from functools import cache, partial
import hashlib
import importlib
import importlib.metadata
from itertools import accumulate
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import tempfile
import time
import warnings

import pytest
import torch
import triton
import triton.language as tl


HERE = Path(__file__).resolve().parent


# ---- Backend contracts and explicit performance parameters ----

def gpu_arch():
    return torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] if torch.cuda.is_available() else "cpu"


@dataclass(frozen=True)
class Backend:
    name: str
    module: str
    arch: str
    dtype: torch.dtype
    memory_mode: str = "lds"
    persistent: bool | None = None
    strided: bool = False
    empty_kv: bool = True
    causal_short_kv: bool = True

    def load(self):
        return importlib.import_module((__package__ + "." if __package__ else "") + self.module)

    @property
    def available(self):
        return self.arch == gpu_arch() or self.arch == "both" and gpu_arch() in ("gfx942", "gfx950")

    @property
    def fp8(self):
        return self.dtype == torch.float8_e4m3fnuz


FP8 = Backend("fp8_942", "mha_pa_fp8_942", "gfx942", torch.float8_e4m3fnuz)
BF16_942 = Backend("bf16_942", "mha_pa_bf16_942", "gfx942", torch.bfloat16,
                   empty_kv=False, causal_short_kv=False)
BF16_950 = Backend("bf16_950", "mha_pa_bf16_950", "gfx950", torch.bfloat16, strided=True)
BF16_950_PERSISTENT = Backend("bf16_950_persistent", "mha_pa_bf16_950", "gfx950", torch.bfloat16,
                              persistent=True, strided=True)
SWA = Backend("swa_bf16", "mha_pa_swa_bf16", "both", torch.bfloat16, strided=True)


@dataclass(frozen=True)
class Workload:
    name: str
    q_lens: tuple[int, ...]
    kv_lens: tuple[int, ...]
    dq: int = 192
    dv: int = 128
    heads: int = 16
    kv_heads: int = 1
    page: int = 64
    causal: bool = False
    window: int = -1
    sink: bool = False
    scale_mode: str = "per-token"
    input_kind: str = "bf16-source"
    seed: int = 20260905

    @property
    def flops(self):
        pairs = 0
        for q, k in zip(self.q_lens, self.kv_lens):
            if not self.causal:
                pairs += q * k
            else:
                for row in range(q):
                    diagonal = k - q + row
                    left = max(0, diagonal - self.window) if self.window >= 0 else 0
                    pairs += max(0, min(k, diagonal + 1) - left)
        return 2 * self.heads * pairs * (self.dq + self.dv)

    def unsupported(self, backend):
        if self.window >= 0 and not self.causal:
            return "windowed attention requires bottom-right causal mode"
        if backend.name == "swa_bf16":
            if self.window < 0 or not self.causal:
                return "single-wave backend requires causal SWA"
        elif self.window >= 0 or self.sink:
            if backend.arch != "gfx950":
                return "this full-MHA backend does not support SWA/sink"
        if backend.name != "bf16_942" and self.page != 64:
            return "backend supports page64 only"
        if backend.name != "bf16_942" and self.dv != 128:
            return "backend supports V128 only"
        if not backend.empty_kv and any(q > 0 and k == 0 for q, k in zip(self.q_lens, self.kv_lens)):
            return "original BF16 pipeline requires nonempty active KV"
        if self.causal and not backend.causal_short_kv and any(k < q for q, k in zip(self.q_lens, self.kv_lens)):
            return "original BF16 causal pipeline requires KV>=Q per sequence"
        return None

    def to_dict(self):
        return {**asdict(self), "effective_flops": self.flops}


@dataclass(frozen=True)
class PerfCase:
    workload: Workload
    smoke: bool = False
    arches: tuple[str, ...] = ("gfx942", "gfx950")

    @property
    def id(self):
        return self.workload.name


# B1, V128, contiguous, unit BF16 descales, zero tail padding, seed20260905.
BF16_MHA_PERF_CASES = (
    *(PerfCase(Workload(f"bf16-smoke-d{d}", (65,), (129,), dq=d), smoke=True) for d in (128, 192)),
    *(PerfCase(Workload(f"bf16-full-d{d}", (10240,), (2583,), dq=d)) for d in (128, 192)),
    *(PerfCase(Workload(f"bf16-causal-d{d}", (32768,), (32768,), dq=d, causal=True)) for d in (128, 192)),
    *(PerfCase(Workload(f"bf16-mha-{name}", (q,), (kv,), dq=128, heads=8, kv_heads=8, page=page),
               arches=("gfx942",))
      for name, q, kv, page in (("long-p32", 20480, 20480, 32),
                                ("short-p32", 10240, 2560, 32), ("short-p64", 10240, 2560, 64))),
)

FP8_MHA_PERF_CASES = (
    *(PerfCase(Workload(f"fp8-full-token-d{d}", (10240,), (2583,), dq=d), arches=("gfx942",))
      for d in (128, 192)),
    *(PerfCase(Workload(f"fp8-causal-token-d{d}", (32768,), (32768,), dq=d, causal=True), arches=("gfx942",))
      for d in (128, 192)),
    PerfCase(Workload("fp8-full-tensor-d128", (10240,), (2560,), dq=128, heads=8, scale_mode="per-tensor"),
             arches=("gfx942",)),
    PerfCase(Workload("fp8-full-tensor-d192", (10240,), (2560,), dq=192, scale_mode="per-tensor"),
             arches=("gfx942",)),
    PerfCase(Workload("fp8-causal-tensor-d192", (32768,), (32768,), dq=192, causal=True, scale_mode="per-tensor"),
             arches=("gfx942",)),
)

SWA_PERF_CASES = (
    *(PerfCase(Workload(f"swa-w0-d{d}", (65,), (129,), dq=d, causal=True, window=0, sink=True), smoke=True)
      for d in (128, 192)),
    *(PerfCase(Workload(f"swa-kv{kv}-d{d}", (16384,), (kv,), dq=d, causal=True, window=128, sink=True))
      for kv in (32768, 65536, 131072) for d in (128, 192)),
)

PERF_SUITES = {"bf16-mha": BF16_MHA_PERF_CASES, "fp8-mha": FP8_MHA_PERF_CASES, "swa": SWA_PERF_CASES}


# ---- Independent paged inputs, FP32 O oracle and native dispatch checks ----

def i32(values):
    return torch.tensor(values, device="cuda", dtype=torch.int32)


def vectorize_kv(k_pages, v_pages):
    pages, page, heads, dq = k_pages.shape
    dv = v_pages.shape[-1]
    vector = 16 // k_pages.element_size()
    k = k_pages.reshape(pages, page, heads, dq // vector, vector).permute(0, 2, 3, 1, 4).contiguous()
    v = v_pages.reshape(pages, page // vector, vector, heads, dv).permute(0, 3, 1, 4, 2).contiguous()
    return k, v


def quantize(values, dtype, *, per_token=False):
    scale = values.float().abs().amax(-1, keepdim=True) if per_token else values.float().abs().amax().reshape(1)
    scale = scale / torch.finfo(dtype).max
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    return (values.float() / scale).to(dtype), scale.float()


@dataclass
class Case:
    q: torch.Tensor
    k_pages: torch.Tensor
    v_pages: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    cq: torch.Tensor
    ck: torch.Tensor
    indptr: torch.Tensor
    indices: torch.Tensor
    last: torch.Tensor
    qs: torch.Tensor
    ks: torch.Tensor
    vs: torch.Tensor
    q_lens: tuple
    kv_lens: tuple
    page_order: list
    q_offset: int
    table_offset: int
    mode: str
    window_left: int
    sinks: torch.Tensor | None

    @property
    def heads(self):
        return self.q.shape[1]

    @property
    def kv_heads(self):
        return self.k_pages.shape[2]

    @property
    def dq(self):
        return self.q.shape[-1]

    @property
    def dv(self):
        return self.v_pages.shape[-1]

    @property
    def page(self):
        return self.k_pages.shape[1]

    def logical_kv(self):
        keys, values = [], []
        pos = self.table_offset
        k_pages, v_pages = self.k_pages.float(), self.v_pages.float()
        for length in self.kv_lens:
            count = (length + self.page - 1) // self.page
            ids = self.page_order[pos:pos + count]
            keys.append(k_pages[ids].reshape(-1, self.kv_heads, self.dq)[:length])
            values.append(v_pages[ids].reshape(-1, self.kv_heads, self.dv)[:length])
            pos += count
        return keys, values


def make_case(q_lens=(256,), kv_lens=(256,), *, dtype=torch.bfloat16, dq=192, dv=128,
              page=64, heads=4, kv_heads=1, mode="per-token", layout="contiguous",
              q_offset=0, table_offset=0, nonunit_scales=False, magnitude=1.0,
              window_left=-1, has_sink=False, poison_tail=True, reverse_pages=False,
              quantized=False, seed=20260906, source_dtype=torch.float32):
    assert len(q_lens) == len(kv_lens)
    torch.manual_seed(seed)
    q_lens, kv_lens = tuple(q_lens), tuple(kv_lens)
    tokens = q_offset + sum(q_lens) + (7 if q_offset else 0)
    q = torch.randn(tokens, heads, dq, device="cuda", dtype=source_dtype) * magnitude
    counts = [(n + page - 1) // page for n in kv_lens]
    pages = max(1, sum(counts))
    k_pages = torch.randn(pages, page, kv_heads, dq, device="cuda", dtype=source_dtype) * magnitude
    v_pages = torch.randn(pages, page, kv_heads, dv, device="cuda", dtype=source_dtype)
    if quantized:
        q, qs = quantize(q, dtype, per_token=mode == "per-token")
        k_pages, ks = quantize(k_pages, dtype)
        v_pages, vs = quantize(v_pages, dtype)
    else:
        q, k_pages, v_pages = q.to(dtype), k_pages.to(dtype), v_pages.to(dtype)
        qs = torch.ones((tokens, heads, 1) if mode == "per-token" else (1,), device="cuda")
        ks, vs = torch.ones(1, device="cuda"), torch.ones(1, device="cuda")
    if nonunit_scales:
        qs.uniform_(0.5, 1.25)
        ks.fill_(0.75)
        vs.fill_(1.25)
    if layout == "padded":
        storage = torch.zeros(tokens, heads + 1, dq + 16, device="cuda", dtype=dtype)
        storage[:, :heads, :dq].copy_(q)
        q = storage[:, :heads, :dq]
    elif layout == "head-major":
        q = q.transpose(0, 1).contiguous().transpose(0, 1)
    elif layout != "contiguous":
        raise ValueError(layout)
    order = list(reversed(range(sum(counts)))) if reverse_pages else torch.randperm(sum(counts)).tolist()
    order = [0] * table_offset + order
    pos = table_offset
    for length, count in zip(kv_lens, counts):
        if length % page:
            k_pages[order[pos + count - 1], length % page:] = float("nan") if poison_tail else 0
            v_pages[order[pos + count - 1], length % page:] = float("nan") if poison_tail else 0
        pos += count
    k, v = vectorize_kv(k_pages, v_pages)
    return Case(q, k_pages, v_pages, k, v,
                i32(list(accumulate(q_lens, initial=q_offset))),
                i32(list(accumulate(kv_lens, initial=11 if table_offset else 0))),
                i32(list(accumulate(counts, initial=table_offset))), i32(order),
                i32([(n - 1) % page + 1 if n else 0 for n in kv_lens]), qs, ks, vs,
                q_lens, kv_lens, order, q_offset, table_offset, mode, window_left,
                torch.linspace(-1, 1, heads, device="cuda") if has_sink else None)


def torch_reference(case, causal, softmax_scale=None):
    """Independent FP32 bottom-right attention O; LSE is not tested."""
    output = torch.full((case.q.shape[0], case.heads, case.dv), float("nan"), device=case.q.device)
    scale = case.dq**-0.5 if softmax_scale is None else softmax_scale
    keys, values = case.logical_kv()
    offset = case.q_offset
    for q_len, kv_len, k, v in zip(case.q_lens, case.kv_lens, keys, values):
        k = (k * case.ks).repeat_interleave(case.heads // case.kv_heads, 1).transpose(0, 1)
        v = (v * case.vs).repeat_interleave(case.heads // case.kv_heads, 1).transpose(0, 1)
        for start in range(0, q_len, 256):
            end = min(start + 256, q_len)
            if kv_len == 0:
                output[offset + start:offset + end] = 0
                continue
            q = case.q[offset + start:offset + end].float()
            qs = case.qs if case.qs.numel() == 1 else case.qs[offset + start:offset + end]
            first, last = 0, kv_len
            if case.window_left >= 0:
                first = max(0, kv_len - q_len + start - case.window_left)
                last = max(0, min(kv_len, kv_len - q_len + end))
            logits = ((q * qs).transpose(0, 1) @ k[:, first:last].transpose(1, 2)) * scale
            if causal:
                diagonal = kv_len - q_len + torch.arange(start, end, device=q.device)[:, None]
                cols = torch.arange(first, last, device=q.device)[None, :]
                mask = cols > diagonal
                if case.window_left >= 0:
                    mask |= cols < diagonal - case.window_left
                logits.masked_fill_(mask, -float("inf"))
            if case.sinks is not None:
                logits = torch.cat((logits, case.sinks[:, None, None].expand(-1, end - start, 1)), -1)
            probabilities = logits.softmax(-1).nan_to_num(0.0)
            if case.sinks is not None:
                probabilities = probabilities[..., :-1]
            output[offset + start:offset + end] = (probabilities @ v[:, first:last]).transpose(0, 1)
        offset += q_len
    return output


def output_buffer(case, layout="contiguous"):
    shape = (case.q.shape[0], case.heads, case.dv)
    if layout == "padded":
        backing = torch.full((shape[0], shape[1] + 1, shape[2] + 16), -123, device="cuda", dtype=torch.bfloat16)
        return backing[:, :shape[1], :shape[2]], backing
    if layout == "head-major":
        backing = torch.full((shape[1], shape[0], shape[2]), -123, device="cuda", dtype=torch.bfloat16)
        return backing.transpose(0, 1), backing
    out = torch.full(shape, -123, device="cuda", dtype=torch.bfloat16)
    return out, out


def make_call(case, backend, causal, *, out=None, **options):
    if out is None:
        out = output_buffer(case)[0]
    if backend != SWA:
        options.setdefault("memory_mode", backend.memory_mode)
        options.setdefault("persistent", backend.persistent)
    kernel = backend.load().PagedAttention(case.heads, case.kv_heads, case.dq, case.dv, case.page,
        causal, case.mode, window_left=case.window_left, has_sink=case.sinks is not None, **options)

    def call(**runtime):
        kwargs = {"out": out, "lse": None, "sink_ptr": case.sinks,
                  "max_seqlen_q": max(case.q_lens, default=0), "max_seqlen_k": max(case.kv_lens, default=0)}
        kwargs.update(runtime)
        max_q, max_k = kwargs.pop("max_seqlen_q"), kwargs.pop("max_seqlen_k")
        return kernel(case.q, case.k, case.v, case.cq, case.ck, case.indptr, case.indices,
                      max_q, max_k, causal, case.qs, case.ks, case.vs, case.last, **kwargs)
    return call, out, kernel


@contextmanager
def _profiler_diagnostics():
    """Filter only native profiler start/stop progress; preserve diagnostics.

    ROCm Kineto logs USDT progress above ERROR severity, so its severity switch
    cannot suppress it without also hiding errors. Capture stderr only around
    this synchronous, untimed dispatch check (never around event measurement).
    On failure replay everything. Other warnings/errors always pass through.
    """
    sys.stderr.flush()
    with os.fdopen(os.dup(2), "wb", buffering=0) as original, tempfile.TemporaryFile() as captured:
        failed = True
        try:
            os.dup2(captured.fileno(), 2)
            yield
            failed = False
        finally:
            sys.stderr.flush()
            os.dup2(original.fileno(), 2)
            captured.seek(0)
            for line in captured:
                progress = re.match(rb"^USDT:.*ActivityProfilerController\.cpp:\d+\]\s+profiler_(start|stop)\s*$", line)
                if failed or not progress:
                    original.write(line)


def dispatch_names(call):
    with _profiler_diagnostics(), warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"Warning: Profiler clears events at the end of each cycle\.",
                                category=UserWarning, module=r"torch\.profiler\.profiler")
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as profiler:
            call()
            torch.cuda.synchronize()
        return [e.name for e in profiler.events() if "CUDA" in str(e.device_type)]


# ---- Test-only gather and AITER references (never production dispatch) ----

@triton.jit
def _gather_kv(K, V, SLOTS, LK, LV, HK: tl.constexpr, DQ: tl.constexpr,
               DV: tl.constexpr, PAGE: tl.constexpr, BLOCK: tl.constexpr):
    token, head = tl.program_id(0), tl.program_id(1)
    slot = tl.load(SLOTS + token)
    page, offset = slot // PAGE, slot % PAGE
    d = tl.arange(0, BLOCK)
    # BF16 SHUFFLE-5D: K[P,H,D/8,S,8], V[P,H,S/8,D,8].
    k_offset = (((page * HK + head) * (DQ // 8) + d // 8) * PAGE + offset) * 8 + d % 8
    v_offset = (((page * HK + head) * (PAGE // 8) + offset // 8) * DV + d) * 8 + offset % 8
    k = tl.load(K + k_offset, d < DQ, other=0)
    v = tl.load(V + v_offset, d < DV, other=0)
    tl.store(LK + (token * HK + head) * DQ + d, k, d < DQ)
    tl.store(LV + (token * HK + head) * DV + d, v, d < DV)


def gather_kv_call(case):
    """Preallocate slot mapping/workspace, but read cache on EVERY call."""
    if case.q.dtype != torch.bfloat16 or case.q_offset or case.table_offset:
        raise ValueError("gather reference requires BF16 with no prefix offsets")
    slots, pos = [], 0
    for length in case.kv_lens:
        ids = case.page_order[pos:pos + (length + case.page - 1) // case.page]
        slots.extend(ids[i // case.page] * case.page + i % case.page for i in range(length))
        pos += len(ids)
    slot_tensor = torch.tensor(slots, device=case.q.device, dtype=torch.int64)
    k = torch.empty((len(slots), case.kv_heads, case.dq), device=case.q.device, dtype=torch.bfloat16)
    v = torch.empty((len(slots), case.kv_heads, case.dv), device=case.q.device, dtype=torch.bfloat16)

    def gather():
        if slots:
            _gather_kv[(len(slots), case.kv_heads)](
                case.k, case.v, slot_tensor, k, v, case.kv_heads, case.dq, case.dv,
                case.page, triton.next_power_of_2(max(case.dq, case.dv)), num_warps=4)
        return k, v
    return gather, (k, v)


class ReferenceUnavailable(RuntimeError):
    """The required comparison package or native specialization is missing."""


@cache
def _aiter():
    os.environ.setdefault("AITER_LOG_LEVEL", "ERROR")
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
    gather, linear_kv = gather_kv_call(case)
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
    """Public BF16 varlen router for Full; explicit CK for window/sink."""
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


# ---- Read-only GPU selection, environment and reporting ----

SMI = "/opt/rocm/bin/amd-smi"
_GPU_LEASE = None
_GPU_SELECTION = None


def _smi(*args):
    return json.loads(subprocess.check_output([SMI, *args, "--json"], text=True, timeout=20))


def _gpu_records(data):
    rows = data.get("gpu_data", []) if isinstance(data, dict) else data
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("SMI returned no valid GPU records")
    return {int(row["gpu"]): row for row in rows}


def select_idle_gpu(*, candidates=None, arches=("gfx942", "gfx950"), required_ptl="current",
                    stable_samples=3, record_path=None):
    """Scan eligible physical GPUs; never set policy or interrupt other jobs."""
    global _GPU_LEASE, _GPU_SELECTION
    if stable_samples < 1:
        raise ValueError("stable_samples must be positive")
    started = time.perf_counter()
    devices = _gpu_records(_smi("static", "--asic", "--bus", "--limit"))
    allowed = set(devices) if candidates is None else set(candidates)
    if not allowed or not allowed <= devices.keys():
        raise ValueError(f"invalid GPU pool {sorted(allowed)}; available={sorted(devices)}")
    eligible, rejected = {}, {}
    for index in sorted(allowed):
        row = devices[index]
        arch = row["asic"].get("target_graphics_version", "")
        limits = row.get("limit", {})
        if arch not in arches:
            rejected[index] = f"architecture {arch} not in {arches}"
            continue
        if required_ptl != "current" and (limits.get("ptl_state"), limits.get("ptl_format")) != ("Enabled", required_ptl):
            rejected[index] = f"requires existing PTL Enabled/{required_ptl}; no hardware setter"
            continue
        bdf = row["bus"]["bdf"].lower()
        unique = (Path("/sys/bus/pci/devices") / bdf / "unique_id").read_text().strip().lower().removeprefix("0x")
        if not unique or any(c not in "0123456789abcdef" for c in unique) or int(unique, 16) == 0:
            raise RuntimeError(f"GPU {index}: no trustworthy ROCr unique_id")
        eligible[index] = {"physical_gpu": index, "bdf": bdf, "rocr_uuid": f"GPU-{unique.zfill(16)}",
                           "asic": row["asic"], "limits": limits}
    if not eligible:
        raise RuntimeError(f"no eligible GPU: {rejected}")
    log = open(record_path, "a") if record_path else None
    monitor = None
    stable = {index: 0 for index in eligible}
    lock_dir = Path.home() / ".cache/pyhip-mha/gpu-locks"
    lock_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    if lock_dir.is_symlink() or lock_dir.stat().st_uid != os.getuid() or lock_dir.stat().st_mode & 0o022:
        raise RuntimeError("GPU lock directory must be private and user-owned")

    def record(data):
        if log:
            log.write(json.dumps({"time_utc": datetime.now(timezone.utc).isoformat(), **data}) + "\n")
            log.flush()

    record({"event": "inventory", "eligible": eligible, "rejected": rejected,
            "old_visibility": {key: os.environ.get(key) for key in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")}})
    try:
        command = [SMI, "metric", "--gpu", *(str(i) for i in eligible), "--usage", "--csv", "--watch", "5"]
        monitor = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                   bufsize=1, env={**os.environ, "PYTHONUNBUFFERED": "1"})
        fields, batch, last_stamps = None, {}, {}
        order = list(eligible)
        for values in csv.reader(monitor.stdout):
            if "gpu" in values and "gfx_activity" in values and "umc_activity" in values:
                if batch:
                    raise RuntimeError("incomplete all-GPU monitoring snapshot")
                fields = values
                continue
            if not values or values[0].startswith("'CTRL'"):
                continue
            if fields is None or len(values) != len(fields):
                raise RuntimeError(f"invalid GPU monitor row: {values}")
            row = dict(zip(fields, values))
            index = int(row["gpu"])
            # SMI emits GPUs in requested order, but timestamps each separately.
            if index != order[len(batch)]:
                raise RuntimeError(f"unexpected/duplicate/out-of-order GPU monitoring row: {row}")
            stamp = int(row["timestamp"])
            if index in last_stamps and stamp <= last_stamps[index]:
                raise RuntimeError(f"stale GPU monitoring row: {row}")
            last_stamps[index] = stamp
            batch[index] = row
            if len(batch) != len(eligible):
                continue
            processes = _gpu_records(_smi("process", "--gpu", *(str(i) for i in eligible)))
            busy = {}
            for i, metrics in batch.items():
                if i not in processes or not processes[i].get("process_list"):
                    raise RuntimeError(f"GPU {i}: missing process status")
                others = other_processes(processes[i], os.getpid())
                quiet = float(metrics["gfx_activity"]) == 0 and float(metrics["umc_activity"]) == 0 and not others
                stable[i] = stable[i] + 1 if quiet else 0
                busy[i] = others
            record({"event": "sample", "metrics": batch, "other_processes": busy, "consecutive_idle": stable})
            for i in sorted(eligible):
                if stable[i] < stable_samples:
                    continue
                lease = (lock_dir / (eligible[i]["bdf"].replace(":", "_") + ".lock")).open("a+")
                try:
                    fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    lease.close()
                    continue
                try:
                    ensure_idle(eligible[i]["bdf"])
                    current = _gpu_records(_smi("static", "--gpu", eligible[i]["bdf"], "--limit"))
                    limits = next(iter(current.values()))["limit"]
                    if (limits.get("ptl_state"), limits.get("ptl_format")) != (
                            eligible[i]["limits"].get("ptl_state"), eligible[i]["limits"].get("ptl_format")):
                        raise RuntimeError("PTL changed during device selection")
                except RuntimeError as exc:
                    lease.close()
                    stable[i] = 0
                    record({"event": "candidate_changed", "gpu": i, "reason": str(exc)})
                    continue
                _GPU_LEASE = lease
                _GPU_SELECTION = {**eligible[i], "logical_gpu": 0, "scan_pool": list(eligible),
                                  "selection_seconds": time.perf_counter() - started,
                                  "stable_samples": stable_samples, "required_ptl": required_ptl,
                                  "local_lock_held": True, "scheduler_reserved": False}
                record({"event": "selected", **_GPU_SELECTION})
                return dict(_GPU_SELECTION)
            batch = {}
        raise RuntimeError("SMI monitor ended before an eligible GPU became idle")
    finally:
        if log:
            log.close()
        if monitor is not None:
            monitor.terminate()
            try:
                monitor.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                monitor.kill()


def activate_selected_gpu(selection):
    """Set physical visibility BEFORE HIP initialization and verify actual BDF."""
    if torch.cuda.is_initialized():
        raise RuntimeError("automatic GPU selection must happen before torch GPU initialization")
    os.environ["ROCR_VISIBLE_DEVICES"] = selection["rocr_uuid"]
    os.environ["HIP_VISIBLE_DEVICES"] = "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    if torch.cuda.device_count() != 1:
        raise RuntimeError("selected ROCr UUID did not expose exactly one GPU")
    prop = torch.cuda.get_device_properties(0)
    actual = f"{prop.pci_domain_id:04x}:{prop.pci_bus_id:02x}:{prop.pci_device_id:02x}.0"
    if actual != selection["bdf"]:
        raise RuntimeError(f"physical GPU mapping mismatch: expected {selection['bdf']}, got {actual}")
    require_idle_device()
    selection["bdf_verified"] = True
    if _GPU_SELECTION is not None:
        _GPU_SELECTION["bdf_verified"] = True


def other_processes(data, own_pid):
    """Parse amd-smi output without ignoring memory-resident idle workers."""
    others = []
    def visit(value):
        if isinstance(value, dict):
            info = value.get("process_info")
            if isinstance(info, dict):
                pid = info.get("pid", info.get("process_id"))
                if pid is None or int(pid) != own_pid:
                    others.append(info)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
    visit(data)
    return others


def ensure_idle(device="0"):
    data = json.loads(subprocess.check_output([SMI, "process", "--gpu", device, "--json"], text=True, timeout=15))
    others = other_processes(data, os.getpid())
    if others:
        raise RuntimeError(f"GPU {device} has another process; refusing idle-guarded benchmark: {others}")


def require_idle_device():
    prop = torch.cuda.get_device_properties(0)
    bdf = f"{prop.pci_domain_id:04x}:{prop.pci_bus_id:02x}:{prop.pci_device_id:02x}.0"
    ensure_idle(bdf)


@contextmanager
def wall_stage(timings, name):
    """Synchronized stage wall time, never inserted inside a benchmark event."""
    torch.cuda.synchronize()
    start = time.perf_counter()
    try:
        yield
    finally:
        torch.cuda.synchronize()
        timings[name] = timings.get(name, 0.0) + time.perf_counter() - start


def environment():
    result = {"time_utc": datetime.now(timezone.utc).isoformat(), "arch": gpu_arch(),
              "torch": torch.__version__, "hip": torch.version.hip,
              "flydsl": importlib.metadata.version("flydsl"),
              "python": sys.version, "interpreter": sys.executable,
              "device_environment": {name: os.environ.get(name) for name in
                  ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")},
              "sources_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob("*.py")}}
    if torch.cuda.is_available():
        prop = torch.cuda.get_device_properties(0)
        result.update(gpu=prop.name, compute_units=prop.multi_processor_count)
        bdf = f"{prop.pci_domain_id:04x}:{prop.pci_bus_id:02x}:{prop.pci_device_id:02x}.0"
        result.update(bdf=bdf, gpu_selection=_GPU_SELECTION)
        try:
            result["limits"] = json.loads(subprocess.check_output([SMI, "static", "--gpu", bdf, "--limit", "--json"], text=True, timeout=15))
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
            result["limits_unavailable"] = str(exc)
    return result


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


# ---- Candidate selection and common correctness/event measurement ----

def bf16_mha_candidates(spec, arch):
    """BF16: own kernels only for smoke, otherwise own kernels + AITER."""
    backends = [BF16_942] if arch == "gfx942" else [BF16_950, BF16_950_PERSISTENT]
    return backends, () if spec.smoke else (("aiter", aiter_reference_call),)


def fp8_mha_candidates(spec, arch):
    """FP8: only our LDS kernel; the independent FP32 correctness oracle stays."""
    return [FP8], ()


def swa_candidates(spec, arch):
    """SWA: own smoke; long shapes compare prepared CK and full gather + CK."""
    if spec.smoke:
        return [SWA], ()
    backends = [SWA] if arch == "gfx942" else [BF16_950, BF16_950_PERSISTENT, SWA]
    return backends, (("aiter", aiter_reference_call), ("aiter_gather", aiter_gather_call))


PERF_CANDIDATES = {"bf16-mha": bf16_mha_candidates, "fp8-mha": fp8_mha_candidates, "swa": swa_candidates}


def accuracy(actual, reference, label, tolerance=0.02):
    """acc is normalized squared error, like pyhip.calc_diff: zero is best.

    Keep the elementwise check too: a scalar average must not hide a few bad
    rows. Include acc in errors without printing successful validation progress.
    """
    from pyhip import calc_diff
    value = float(calc_diff(reference, actual, diff_thr=-1))
    finite = math.isfinite(value) and bool(torch.isfinite(actual).all())
    if not finite:
        raise AssertionError(f"{label} acc={value}: nonfinite output")
    try:
        torch.testing.assert_close(actual.float(), reference.float(), rtol=tolerance, atol=tolerance)
    except AssertionError as exc:
        raise AssertionError(f"{label} acc={value:.8g} FAIL (rtol=atol={tolerance})\n{exc}") from exc
    return value


def perf_timer(flops, nbytes, name):
    from pyhip import cudaPerf
    return cudaPerf(flops=flops, rw_bytes=nbytes, name=name, verbose=0)


def measure(calls, references, *, label, flops, nbytes, run_count=5, warmup=10, repeat=1,
            tolerance=0.02, timings=None, verbose_runs=False):
    """Check every independent buffer, then run_count full rotating-buffer rounds.

    One round measures every buffer. repeat>1 advances the buffer on every
    invocation inside one event pair; all candidates see identical indices.
    Median over all event intervals, not the fastest sample or a sum of timers.
    """
    buffers = len(references)
    if not buffers or not calls or any(len(pool) != buffers for pool in calls.values()):
        raise ValueError("every candidate requires the same nonempty buffer pool")
    timings = {} if timings is None else timings
    rows = []
    for name, pool in calls.items():
        accs, outputs = [], []
        for index, (call, reference) in enumerate(zip(pool, references)):
            with wall_stage(timings, "candidate_first_calls_s"):
                actual = call()
            with wall_stage(timings, "output_validation_s"):
                accs.append(accuracy(actual, reference, f"{label}/{name}/buffer{index}", tolerance))
                outputs.append(actual.data_ptr())
                first = actual.clone()
                for _ in range(2):
                    torch.testing.assert_close(call(), first, rtol=0, atol=0)
        if len(set(outputs)) != buffers:
            raise AssertionError(f"{name}: output buffers alias")
        with wall_stage(timings, "dispatch_profiling_s"):
            kernels = dispatch_names(pool[0])
        if not kernels and actual.numel():
            raise AssertionError(f"{name}: expected a native attention dispatch")
        if name == "aiter_gather" and (len(kernels) != 2 or not any("gather" in k for k in kernels)):
            raise AssertionError(f"expected one gather and one CK dispatch: {kernels}")
        rows.append({"backend": name, "acc": max(accs), "acc_per_buffer": accs,
                 "checked": True, "status": "passed",
                     "repeated_bit_exact": True, "output_ptrs": outputs,
                     "workspace_ptrs": [getattr(call, "workspace_ptrs", []) for call in pool],
                     "kernels": kernels, "aiter_entry": getattr(pool[0], "aiter_entry", None),
                     "logical_io_bytes": nbytes[name], "runs": []})
    if run_count:
        with wall_stage(timings, "warmup_s"):
            for index in range(warmup):
                for pool in calls.values():
                    pool[index % buffers]()
    by_name = {row["backend"]: row for row in rows}
    timed_start = time.perf_counter()
    for trial in range(run_count):
        for buffer in range(buffers):
            order = list(calls) if (trial + buffer) % 2 == 0 else list(reversed(calls))
            indices = [(buffer + iteration) % buffers for iteration in range(repeat)]
            for name in order:
                with perf_timer(flops * repeat, nbytes[name] * repeat, name) as perf:
                    for index in indices:
                        calls[name][index]()
                us = perf.dt() * 1e6 / repeat
                if not math.isfinite(us) or us <= 0:
                    raise RuntimeError("cudaPerf returned no positive timing; check CUDAPERF filtering")
                sample = {"run": trial + 1, "buffer_indices": indices, "us": us,
                          "tflops": perf.tflops(), "gbps": perf.bw()}
                row = by_name[name]
                row["runs"].append(sample)
                acc = f"{row['acc']:.8g}"
                if verbose_runs:
                    print(f"{label}/{name} round={trial + 1}/{run_count} buffers={indices} acc={acc} "
                          f"time={us:.3f} us tflops={sample['tflops']:.3f} bw={sample['gbps']:.3f} GB/s", flush=True)
    timings["measurement_wall_s"] = time.perf_counter() - timed_start
    for row in rows:
        us = statistics.median(sample["us"] for sample in row["runs"]) if row["runs"] else None
        row.update(us=us, tflops=flops / us / 1e6 if us else None,
                   gbps=nbytes[row["backend"]] / us / 1e3 if us else None)
    return rows


@torch.inference_mode()
def run_case(workload, backends, *, run_count=5, warmup=10, repeat=1, buffers=10,
             reference_factories=(), layout="contiguous", nonunit_scales=False, softmax_scale=None,
             poison_tail=False, query_tile=None, block_n=None, verbose_runs=False):
    case_start = time.perf_counter()
    timings = {}
    w = workload
    shape = w.to_dict()
    supported = list(backends)
    if not supported:
        raise ValueError("a case must explicitly select at least one backend")
    for backend in supported:
        if not backend.available or w.unsupported(backend):
            raise ValueError(f"{w.name}/{backend.name}: {w.unsupported(backend) or 'unsupported architecture'}")
    if len({b.dtype for b in supported}) != 1:
        raise ValueError("a case cannot mix BF16 and FP8")
    if layout != "contiguous" and any(not b.strided for b in supported):
        raise ValueError("the selected backend requires contiguous Q/O")
    if poison_tail and BF16_942 in supported:
        raise ValueError("bf16_942 uses zero-padded tails; --poison-tail is supported by gfx950/SWA/FP8")
    if buffers < 1:
        raise ValueError("buffers must be positive")
    backend = supported[0]
    cases, references, calls = [], [], {}
    factories = dict(reference_factories)
    for index in range(buffers):
        with wall_stage(timings, "input_preparation_s"):
            case = make_case(w.q_lens, w.kv_lens, dtype=backend.dtype, dq=w.dq, dv=w.dv, page=w.page,
                             heads=w.heads, kv_heads=w.kv_heads, mode=w.scale_mode, layout=layout,
                             window_left=w.window, has_sink=w.sink, nonunit_scales=nonunit_scales,
                             poison_tail=poison_tail, quantized=backend.fp8, source_dtype=torch.bfloat16, seed=w.seed + index)
        cases.append(case)
        with wall_stage(timings, "fp32_reference_s"):
            references.append(torch_reference(case, w.causal, softmax_scale))
        for candidate in supported:
            out, _ = output_buffer(case, layout)
            options = {"query_tile": query_tile, "block_n": block_n} if candidate == SWA else {}
            call = make_call(case, candidate, w.causal, out=out, **options)[0]
            calls.setdefault(candidate.name, []).append(partial(call, softmax_scale=softmax_scale))
        for name, factory in factories.items():
            # Declared references are required: never silently drop a comparison.
            with wall_stage(timings, "reference_setup_and_first_calls_s"):
                call = factory(case, w.causal, softmax_scale=softmax_scale)
                probe_reference(call)
            calls.setdefault(name, []).append(call)
    pointers = [{key: getattr(c, key).data_ptr() for key in ("q", "k", "v", "qs", "ks", "vs", "indices")} for c in cases]
    for key in ("q", "k", "v"):
        if sum(w.q_lens) and len({ptr[key] for ptr in pointers}) != buffers:
            raise AssertionError(f"input {key} buffers alias")
    # Logical Q/K/V/O size, not a hardware traffic counter (especially for SWA).
    nbytes = (sum(w.q_lens) * w.heads * (w.dq * case.q.element_size() + w.dv * 2)
              + sum(w.kv_lens) * w.kv_heads * (w.dq + w.dv) * case.k.element_size())
    candidate_bytes = {name: nbytes + getattr(pool[0], "extra_io_bytes", 0) for name, pool in calls.items()}
    label = f"{w.name} B{len(w.q_lens)} H{w.heads}/{w.kv_heads} Q{w.q_lens} KV{w.kv_lens} W{w.window}"
    rows = measure(calls, references, label=label, flops=w.flops, nbytes=candidate_bytes, run_count=run_count,
                   warmup=warmup, repeat=repeat, tolerance=0.1 if backend.fp8 else 0.02,
                   timings=timings, verbose_runs=verbose_runs)
    timings["case_total_s"] = time.perf_counter() - case_start
    return {"workload": shape, "results": rows, "unavailable": {}, "wall_time_s": timings,
            "input": {"dtype": str(backend.dtype), "layout": layout, "poison_tail": poison_tail,
                      "input_buffers": buffers, "buffer_seeds": [w.seed + i for i in range(buffers)],
                      "buffer_ptrs": pointers, "with_lse": False,
                      "nonunit_scales": nonunit_scales, "softmax_scale": softmax_scale,
                      "query_tile": query_tile, "block_n": block_n,
                      "q_shape": list(case.q.shape), "q_stride": list(case.q.stride()),
                      "flydsl_kv": "SHUFFLE-5D", "aiter_kv": "prepared linear THD; conversion not timed"},
            "flops": w.flops, "logical_io_bytes": nbytes, "candidate_logical_io_bytes": candidate_bytes}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", choices=("all", *PERF_SUITES), default="all")
    parser.add_argument("--case", nargs="+", dest="case_ids", help="exact IDs from --list; no implicit shape changes")
    parser.add_argument("--list", action="store_true", help="list explicit parameters without GPU initialization")
    parser.add_argument("--buffers", type=int, default=10, help="independent input/output/workspace sets; every round measures all buffers")
    parser.add_argument("--run-count", type=int, default=5, help="positive full buffer rounds; default 5x10=50 samples per candidate")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=1, help="rotating-buffer calls per event interval; report per-call time")
    parser.add_argument("--gpu", default=os.environ.get("PYHIP_MHA_GPU", "auto"),
                        help="auto: scan eligible GPUs; physical SMI index: restrict to that GPU; current: existing visibility")
    parser.add_argument("--gpu-pool", type=lambda s: [int(x) for x in s.split(",")],
                        help="optional physical SMI index pool for auto selection, e.g. 0,1,2,3")
    parser.add_argument("--required-ptl", choices=("current", "VECTOR,F8", "VECTOR,BF16"),
                        default=os.environ.get("PYHIP_MHA_REQUIRED_PTL", "current"),
                        help="select only GPUs ALREADY using this policy; never sets hardware")
    parser.add_argument("--verbose-runs", action="store_true", help="print every buffer/sample; JSON always retains all samples")
    parser.add_argument("--output", type=Path, help="optional JSON plus summary Markdown; use a new filename")
    args = parser.parse_args(argv)
    if args.gpu not in ("auto", "current") and not args.gpu.isdecimal():
        parser.error("--gpu must be auto, current or a nonnegative physical SMI index")
    if args.gpu_pool is not None and (not args.gpu_pool or min(args.gpu_pool) < 0 or args.gpu.isdecimal()):
        parser.error("--gpu-pool requires auto/current and nonnegative indices")
    if args.gpu_pool is not None and args.gpu == "current":
        args.gpu = "auto"
    if min(args.run_count, args.repeat, args.buffers) < 1 or args.warmup < 0:
        parser.error("run-count/repeat/buffers must be positive; warmup must be nonnegative")
    available = {spec.id for name, specs in PERF_SUITES.items() if args.suite in ("all", name) for spec in specs}
    if args.case_ids and (len(args.case_ids) != len(set(args.case_ids)) or not set(args.case_ids) <= available):
        parser.error("--case requires distinct IDs in the selected suite; use --list")
    return args


def run(args, selected):
    """Run (suite, PerfCase) pairs; all selection/reference decisions are explicit."""
    if args.output:
        args.output = args.output.resolve()
        if args.output.suffix != ".json":
            raise ValueError("--output must be a .json file; a companion .md is generated")
        if args.output.exists() or args.output.with_suffix(".md").exists():
            raise FileExistsError("choose a new output file; previous evidence must not be overwritten")
    execution_start = time.perf_counter()
    report = {"complete": False, "environment": environment(), "config": {**vars(args), "output": str(args.output) if args.output else None},
              "wall_time_s": {"idle_wait_s": getattr(args, "idle_wait_s", 0.0)},
              "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[3], text=True).strip(),
              "measurement_scope": "idle-guarded diagnostic; not a scheduler reservation",
              "timer": "pyhip.cudaPerf GPU event interval; per-call time; includes launch gaps",
              "protocol": {"warmup": args.warmup, "run_count": args.run_count, "repeat": args.repeat,
                           "input_buffers": args.buffers, "samples_per_candidate": args.run_count * args.buffers,
                           "summary": "median of all per-call event times across all buffers and rounds",
                           "buffer_rotation": args.buffers > 1, "repeat_reuses_same_buffer": args.buffers == 1,
                           "gpu_spin_before_event": "cudaPerf's built-in torch.cuda._sleep(1000000)"},
              "bandwidth": "logical Q/K/V/O bytes; aiter_gather adds full KV read+write; not measured HBM traffic",
              "acc_definition": "sum((ref-out)^2) / sum(ref^2+out^2), smaller is better; elementwise tolerance also required",
              "records": []}
    try:
        for suite, spec in selected:
            if report["environment"]["arch"] not in spec.arches:
                report["records"].append({"suite": suite, "workload": spec.workload.to_dict(), "results": [],
                                           "unavailable": {"architecture": f"requires {spec.arches}"}})
                continue
            require_idle_device()
            backends, references = PERF_CANDIDATES[suite](spec, report["environment"]["arch"])
            row = run_case(spec.workload, backends, reference_factories=references,
                           run_count=args.run_count, warmup=args.warmup, repeat=args.repeat,
                           buffers=args.buffers, verbose_runs=args.verbose_runs)
            row["suite"] = suite
            row["reference_names"] = [name for name, _ in references]
            report["records"].append(row)
            if args.output:
                save(args.output, report)
            require_idle_device()
        if not any(row["results"] for row in report["records"]):
            raise RuntimeError("no supported candidate ran")
        report["complete"] = True
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["wall_time_s"]["execution_s"] = time.perf_counter() - execution_start
        report["suite_wall_time_s"] = {suite: sum(r.get("wall_time_s", {}).get("case_total_s", 0.0)
                                                for r in report["records"] if r["suite"] == suite)
                                       for suite in dict(selected)}
        if args.output:
            save(args.output, report)
    import pandas as pd
    table = pd.DataFrame([{"suite": row["suite"], "case": row["workload"]["name"], **{k: r[k] for k in
                          ("backend", "status", "acc", "us", "tflops", "gbps")}}
                         for row in report["records"] for r in row["results"]]).to_markdown(index=False, floatfmt=".6g")
    print("\nMedian per-call results (all runs retained):\n" + table, flush=True)
    if args.output:
        dispatches = [f"- **{row['workload']['name']}**: `{r['aiter_entry']}`\n  - `" + "`\n  - `".join(r["kernels"]) + "`"
                      for row in report["records"] for r in row["results"] if r["backend"].startswith("aiter")]
        args.output.with_suffix(".md").write_text(
            "# Quick MHA accuracy/performance\n\n"
            f"cudaPerf events; {args.buffers} independent buffers, {args.run_count} full rounds; acc=max across buffers. "
            "Bandwidth is logical Q/K/V/O GB/s; gather+CK adds full KV read/write, not a hardware counter.\n\n"
            + table + "\n\n## AITER dispatch\n\n" + "\n".join(dispatches or ["AITER disabled or unavailable."]) + "\n")
    return report


def main(argv=None):
    args = parse_args(argv)
    selected = [(suite, spec) for suite, specs in PERF_SUITES.items() if args.suite in ("all", suite)
                for spec in specs if args.case_ids is None or spec.id in args.case_ids]
    if args.list:
        for suite, spec in selected:
            print(f"{suite}: {spec.id} arches={spec.arches} smoke={spec.smoke} {spec.workload.to_dict()}")
        return selected
    wait_start = time.perf_counter()
    selecting = args.gpu != "current"
    if selecting:
        log = args.output.with_suffix(".idle.jsonl") if args.output else None
        if log:
            if args.output.exists() or args.output.with_suffix(".md").exists() or log.exists():
                raise FileExistsError("choose new report and idle-log paths before waiting")
            log.parent.mkdir(parents=True, exist_ok=True)
        arches = tuple(sorted({arch for _, spec in selected for arch in spec.arches}))
        selection = select_idle_gpu(candidates=[int(args.gpu)] if args.gpu.isdecimal() else args.gpu_pool,
                                    arches=arches, required_ptl=args.required_ptl, record_path=log)
        activate_selected_gpu(selection)
    args.idle_wait_s = time.perf_counter() - wait_start if selecting else 0.0
    return run(args, selected)


@pytest.fixture(scope="module", autouse=True)
def native_gpu_selection():
    # Collection remains GPU-free. Setup occurs only when native tests run.
    gpu = os.environ.get("PYHIP_MHA_GPU", "current")
    if gpu != "current":
        if gpu != "auto" and not gpu.isdecimal():
            raise ValueError("PYHIP_MHA_GPU must be auto/current/physical index")
        selection = select_idle_gpu(candidates=None if gpu == "auto" else [int(gpu)],
                                    required_ptl=os.environ.get("PYHIP_MHA_REQUIRED_PTL", "current"),
                                    record_path=os.environ.get("PYHIP_MHA_SELECTION_LOG"))
        activate_selected_gpu(selection)


@pytest.mark.parametrize("variant", ("8wave", "persistent"))
@pytest.mark.parametrize("dq", (128, 192))
def test_bf16_mha(dq, variant):
    """Only the short ragged/empty/poisoned cases not covered by perf shapes."""
    arch = gpu_arch()
    if arch not in ("gfx942", "gfx950") or variant == "persistent" and arch != "gfx950":
        pytest.skip("requires a supported native GPU")
    backends = [BF16_942 if arch == "gfx942" else BF16_950_PERSISTENT if variant == "persistent" else BF16_950]
    kv = (63, 0, 193, 257) if arch == "gfx950" else (63, 65, 193, 257)
    w = Workload("ragged_edges", (0, 7, 33, 129), kv, dq=dq, heads=6, kv_heads=2,
                 causal=True)
    run_case(w, backends, run_count=0, layout="padded" if backends[0].strided else "contiguous", nonunit_scales=True,
             poison_tail=backends != [BF16_942], softmax_scale=0.0625)


@pytest.mark.parametrize("dq", (128, 192))
def test_swa(dq):
    """One fused ragged/empty/poisoned/scale case per head dimension."""
    if gpu_arch() not in ("gfx942", "gfx950"):
        pytest.skip("requires a supported native GPU")
    w = Workload("swa_ragged_edges", (0, 7, 33, 129), (63, 0, 193, 257),
                 dq=dq, heads=6, kv_heads=2, causal=True, window=128, sink=True)
    run_case(w, [SWA], run_count=0, layout="padded",
             nonunit_scales=True, poison_tail=True, softmax_scale=0.0625)


@pytest.mark.parametrize("dq", (128, 192))
@pytest.mark.parametrize("causal", (False, True))
@pytest.mark.parametrize("mode", ("per-token", "per-tensor"))
def test_fp8_mha(dq, causal, mode):
    if gpu_arch() != "gfx942":
        pytest.skip("requires native gfx942")
    w = Workload("fp8_lds_edges", (0, 7, 33, 129), (63, 65, 193, 257), dq=dq,
                 heads=6, kv_heads=2, causal=causal, scale_mode=mode)
    run_case(w, [FP8], run_count=0, poison_tail=True)


def _pytest_perf(suite, spec):
    if gpu_arch() not in spec.arches:
        pytest.skip(f"requires {spec.arches}")
    # Shared timing defaults, not a second set of shape/reference selection rules.
    args = parse_args([])
    args.suite, args.case_ids, args.gpu = suite, [spec.id], "current"
    directory = os.environ.get("PYHIP_MHA_OUTPUT")
    if directory:
        args.output = Path(directory) / f"{spec.id}.json"
    run(args, [(suite, spec)])


@pytest.mark.parametrize("spec", BF16_MHA_PERF_CASES, ids=lambda spec: spec.id)
def test_perf_bf16_mha(spec):
    _pytest_perf("bf16-mha", spec)


@pytest.mark.parametrize("spec", FP8_MHA_PERF_CASES, ids=lambda spec: spec.id)
def test_perf_fp8_mha(spec):
    _pytest_perf("fp8-mha", spec)


@pytest.mark.parametrize("spec", SWA_PERF_CASES, ids=lambda spec: spec.id)
def test_perf_swa(spec):
    _pytest_perf("swa", spec)


# Pytest honors __test__ during collection. No plugin hooks/custom options are
# needed: enable before collection, then use standard -k test_perf_ for perf only.
for _performance_test in (test_perf_bf16_mha, test_perf_fp8_mha, test_perf_swa):
    _performance_test.__test__ = os.environ.get("PYHIP_MHA_PERF", "0") == "1"
del _performance_test


if __name__ == "__main__":
    main()