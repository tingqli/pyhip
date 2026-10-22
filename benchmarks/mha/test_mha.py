"""MHA integration benchmark: accuracy and performance of the FlyDSL MHA kernels.

Suites: bf16-mha / fp8-mha / swa (paged SHUFFLE-5D PagedAttention) and linear
(D256 flash_attn_varlen_func, linear/page1/page4). Every candidate is checked on
every independent buffer against an FP32 oracle (plus two bit-exact repeats)
before timing; JIT, allocation and references stay outside the cudaPerf timer.

    python3 benchmarks/mha/test_mha.py --list
    python3 benchmarks/mha/test_mha.py --suite bf16-mha --case bf16-full-d128 --gpu 0
    python3 benchmarks/mha/test_mha.py --suite all --check-only
"""

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import cache
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import pytest
import torch
import triton
import triton.language as tl

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.ops.mha._mha import (BF16_942, BF16_942_GRID, BF16_950, BF16_950_PERSISTENT, FP8, SWA, Workload,
                                check_outputs, gpu_arch, linear_inputs, paged_inputs)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "mytest/mydata"
SOURCES = sorted((ROOT / "src/pyhip/ops/mha/flydsl").glob("*.py"))


# ---- Explicit cases ----

@dataclass(frozen=True)
class PerfCase:
    workload: Workload
    smoke: bool = False
    arches: tuple[str, ...] = ("gfx942", "gfx950")

    @property
    def id(self):
        return self.workload.name

    def to_dict(self):
        return self.workload.to_dict()


@dataclass(frozen=True)
class LinearCase:
    """D256 flash_attn_varlen_func: contiguous BF16 [T,H,256] Q/K/V, optional page1/page4 table."""
    name: str
    q_lens: tuple[int, ...]
    kv_lens: tuple[int, ...]
    heads: int = 24
    kv_heads: int = 2
    causal: bool = False
    page: int | None = None
    smoke: bool = False
    arches: tuple[str, ...] = ("gfx942",)
    seed: int = 20260924

    @property
    def id(self):
        return self.name

    @property
    def flops(self):
        pairs = sum(q * k if not self.causal else sum(min(k, k - q + row + 1) for row in range(q))
                    for q, k in zip(self.q_lens, self.kv_lens))
        return 2 * self.heads * pairs * (256 + 256)

    def to_dict(self):
        return {**asdict(self), "effective_flops": self.flops}


# B1, contiguous, unit BF16 descales, zero tail padding, seed 20260905.
# V128 unless stated; explicit D256 GQA cases use Dq=Dv=256, H24/HK2.
BF16_MHA_CASES = (
    *(PerfCase(Workload(f"bf16-smoke-d{d}", (65,), (129,), dq=d), smoke=True) for d in (128, 192)),
    *(PerfCase(Workload(f"bf16-full-d{d}", (10240,), (2583,), dq=d)) for d in (128, 192)),
    *(PerfCase(Workload(f"bf16-causal-d{d}", (32768,), (32768,), dq=d, causal=True)) for d in (128, 192)),
    *(PerfCase(Workload(f"bf16-mha-{name}", (q,), (kv,), dq=128, heads=8, kv_heads=8, page=page),
               arches=("gfx942",))
      for name, q, kv, page in (("long-p32", 20480, 20480, 32),
                                ("short-p32", 10240, 2560, 32), ("short-p64", 10240, 2560, 64))),
    *(PerfCase(Workload(f"bf16-mha-{name}-d192", (q,), (kv,), dq=192, heads=8, kv_heads=8, page=page),
               arches=("gfx942",))
      for name, q, kv, page in (("long-p32", 20480, 20480, 32),
                                ("short-p32", 10240, 2560, 32), ("short-p64", 10240, 2560, 64))),
    PerfCase(Workload("bf16-gqa-smoke-d256", (65,), (129,), dq=256, dv=256, heads=24, kv_heads=2),
             smoke=True, arches=("gfx942",)),
    *(PerfCase(Workload(f"bf16-gqa-{name}-d256", (q,), (kv,), dq=256, dv=256, heads=24, kv_heads=2,
                        page=page, causal=causal), arches=("gfx942",))
      for name, q, kv, page, causal in (("full", 10240, 2583, 64, False),
                                       ("causal", 32768, 32768, 64, True),
                                       ("long-p32", 20480, 20480, 32, False),
                                       ("short-p32", 10240, 2560, 32, False),
                                       ("short-p64", 10240, 2560, 64, False))),
)

FP8_MHA_CASES = (
    PerfCase(Workload("fp8-smoke-d128", (65,), (129,), dq=128), smoke=True, arches=("gfx942",)),
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

SWA_CASES = (
    *(PerfCase(Workload(f"swa-w0-d{d}", (65,), (129,), dq=d, causal=True, window=0, sink=True), smoke=True)
      for d in (128, 192)),
    *(PerfCase(Workload(f"swa-kv{kv}-d{d}", (16384,), (kv,), dq=d, causal=True, window=128, sink=True))
      for kv in (32768, 65536, 131072) for d in (128, 192)),
)

LINEAR_CASES = (
    LinearCase("linear-smoke", (0, 7, 33, 65), (3, 33, 65, 129), heads=12, kv_heads=1, causal=True, smoke=True),
    LinearCase("linear-full", (10240,), (2583,)),
    LinearCase("linear-full-page1", (10240,), (2583,), page=1),
    LinearCase("linear-full-page4", (10240,), (2583,), page=4),
    *(LinearCase(f"linear-causal-{n}", (n,), (n,), heads=12, kv_heads=1, causal=True) for n in (2048, 8192)),
)

SUITES = {"bf16-mha": BF16_MHA_CASES, "fp8-mha": FP8_MHA_CASES, "swa": SWA_CASES, "linear": LINEAR_CASES}


# ---- Benchmark-only gather and AITER references (never production dispatch) ----

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
    """Preallocate slot mapping/workspace, but read the paged cache on EVERY call."""
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
    call.extra_io_bytes = 2 * sum(t.numel() * t.element_size() for t in linear_kv)
    return call


def aiter_reference_call(case, causal, out=None, softmax_scale=None):
    """Public BF16 varlen router for full attention; explicit CK for window/sink."""
    if case.window_left >= 0 or case.sinks is not None:
        call = aiter_linear_call(case, causal, out, softmax_scale)
        call.aiter_entry = "aiter.ops.mha.mha_varlen_fwd (CK window/sink)"
        return call
    aiter = _aiter()
    if case.q.dtype != torch.bfloat16 or case.q_offset or case.table_offset:
        raise ReferenceUnavailable("AITER comparison requires BF16 with unpadded prefix metadata")
    if not all(bool(torch.all(scale == 1)) for scale in (case.qs, case.ks, case.vs)):
        raise ReferenceUnavailable("AITER comparison requires unit descales; no silent conversion")
    keys, values = case.logical_kv()
    q = case.q.contiguous()
    k = torch.cat(keys).to(torch.bfloat16).contiguous()
    v = torch.cat(values).to(torch.bfloat16).contiguous()
    if out is None:
        out = torch.empty(q.shape[0], case.heads, case.dv, device=q.device, dtype=torch.bfloat16)
    max_q, max_k = max(case.q_lens, default=0), max(case.kv_lens, default=0)

    def call():
        return aiter.flash_attn_varlen_func(q, k, v, case.cq, case.ck, max_q, max_k, causal=causal,
                                            softmax_scale=softmax_scale, out=out)
    call.aiter_entry = "aiter.flash_attn_varlen_func (public router)"
    return call


def aiter_varlen_call(arguments):
    """AITER public router on the same contiguous D256 THD inputs (unpaged linear cases only)."""
    aiter = _aiter()
    a = arguments
    out = torch.empty_like(a["q"])

    def call():
        return aiter.flash_attn_varlen_func(a["q"], a["k"], a["v"], a["cu_seqlens_q"], a["cu_seqlens_k"],
                                            a["max_seqlen_q"], a["max_seqlen_k"], causal=a["causal"],
                                            softmax_scale=a["softmax_scale"], out=out)
    call.aiter_entry = "aiter.flash_attn_varlen_func (public router)"
    return call


# ---- Candidates per suite: (own backends, declared references) ----

def bf16_mha_candidates(spec, arch, aiter=True):
    """BF16: own kernels only for smoke, otherwise own kernels + AITER."""
    backends = [BF16_942, BF16_942_GRID] if arch == "gfx942" else [BF16_950, BF16_950_PERSISTENT]
    return backends, () if spec.smoke or not aiter else (("aiter", aiter_reference_call),)


def fp8_mha_candidates(spec, arch, aiter=True):
    """FP8: only the LDS kernel; the independent FP32 oracle stays."""
    return [FP8], ()


def swa_candidates(spec, arch, aiter=True):
    """SWA: own smoke; long shapes compare prepared CK and full gather + CK."""
    if spec.smoke:
        return [SWA], ()
    backends = [SWA] if arch == "gfx942" else [BF16_950, BF16_950_PERSISTENT, SWA]
    return backends, (("aiter", aiter_reference_call), ("aiter_gather", aiter_gather_call)) if aiter else ()


CANDIDATES = {"bf16-mha": bf16_mha_candidates, "fp8-mha": fp8_mha_candidates, "swa": swa_candidates}


# ---- Measurement ----

def measure(calls, references, *, label, flops, nbytes, run_count, warmup, repeat, tolerance, verbose_runs):
    """Check every buffer, then run_count full rotating-buffer rounds; median of all event intervals.

    repeat > 1 advances the buffer on every call inside one event pair; all candidates see identical
    buffer indices and alternate their order every sample.
    """
    from pyhip import cudaPerf

    buffers = len(references)
    if not buffers or not calls or any(len(pool) != buffers for pool in calls.values()):
        raise ValueError("every candidate requires the same nonempty buffer pool")
    checked = check_outputs(calls, references, label=label, tolerance=tolerance)
    if "aiter_gather" in checked:
        kernels = checked["aiter_gather"]["kernels"]
        if len(kernels) != 2 or not any("gather" in k for k in kernels):
            raise AssertionError(f"expected one gather and one CK dispatch: {kernels}")
    rows = []
    for name, row in checked.items():
        row.update(aiter_entry=getattr(calls[name][0], "aiter_entry", None), logical_io_bytes=nbytes[name], runs=[])
        rows.append(row)
    if run_count:
        for index in range(warmup):
            for pool in calls.values():
                pool[index % buffers]()
    for trial in range(run_count):
        for buffer in range(buffers):
            order = list(calls) if (trial + buffer) % 2 == 0 else list(reversed(calls))
            indices = [(buffer + i) % buffers for i in range(repeat)]
            for name in order:
                with cudaPerf(flops=flops * repeat, rw_bytes=nbytes[name] * repeat, name=name, verbose=0) as perf:
                    for index in indices:
                        calls[name][index]()
                us = perf.dt() * 1e6 / repeat
                if not math.isfinite(us) or us <= 0:
                    raise RuntimeError("cudaPerf returned no positive timing; check CUDAPERF filtering")
                checked[name]["runs"].append({"run": trial + 1, "buffer_indices": indices, "us": us})
                if verbose_runs:
                    print(f"{label}/{name} round={trial + 1}/{run_count} buffers={indices} time={us:.3f} us",
                          flush=True)
    for row in rows:
        us = statistics.median(sample["us"] for sample in row["runs"]) if row["runs"] else None
        row.update(us=us, tflops=flops / us / 1e6 if us else None,
                   gbps=row["logical_io_bytes"] / us / 1e3 if us else None)
    return rows


@torch.inference_mode()
def run_paged(suite, spec, arch, args):
    w = spec.workload
    backends, references = CANDIDATES[suite](spec, arch, not args.skip_aiter)
    cases, oracle, calls = paged_inputs(w, backends, buffers=args.buffers)
    for name, factory in references:
        # Declared references are required: never silently drop a comparison.
        for case in cases:
            call = factory(case, w.causal)
            probe_reference(call)
            calls.setdefault(name, []).append(call)
    for key in ("q", "k", "v"):
        if sum(w.q_lens) and len({getattr(c, key).data_ptr() for c in cases}) != len(cases):
            raise AssertionError(f"input {key} buffers alias")
    case = cases[0]
    # Logical Q/K/V/O bytes, not a hardware traffic counter (especially for SWA).
    nbytes = (sum(w.q_lens) * w.heads * (w.dq * case.q.element_size() + w.dv * 2)
              + sum(w.kv_lens) * w.kv_heads * (w.dq + w.dv) * case.k.element_size())
    candidate_bytes = {name: nbytes + getattr(pool[0], "extra_io_bytes", 0) for name, pool in calls.items()}
    label = f"{w.name} B{len(w.q_lens)} H{w.heads}/{w.kv_heads} Q{w.q_lens} KV{w.kv_lens} W{w.window}"
    rows = measure(calls, oracle, label=label, flops=w.flops, nbytes=candidate_bytes,
                   run_count=0 if args.check_only else args.run_count, warmup=args.warmup, repeat=args.repeat,
                   tolerance=0.1 if backends[0].fp8 else 0.02, verbose_runs=args.verbose_runs)
    return {"results": rows, "flops": w.flops, "logical_io_bytes": nbytes,
            "input": {"dtype": str(backends[0].dtype), "flydsl_kv": "SHUFFLE-5D",
                      "aiter_kv": "prepared linear THD; conversion not timed",
                      "buffer_seeds": [w.seed + i for i in range(args.buffers)]}}


@torch.no_grad()
def run_linear(spec, args):
    """D256 flash_attn_varlen_func (persistent and grid) vs AITER on unpaged inputs; no LSE."""
    from pyhip.ops.mha.flydsl.mha_pa_bf16_942 import flash_attn_varlen_func

    device = torch.device("cuda", torch.cuda.current_device())
    references, calls = [], {}
    # Metadata must keep version counters: build inputs outside inference_mode.
    for index in range(args.buffers):
        arguments, (output, _) = linear_inputs(device, qlens=spec.q_lens, klens=spec.kv_lens, page=spec.page,
                                               heads=spec.heads, kv_heads=spec.kv_heads, causal=spec.causal,
                                               scale=256**-0.5, poison=False, seed=spec.seed + index)
        arguments["return_lse"] = False
        references.append(output)
        for name, persistent in (("linear", True), ("linear_grid", False)):
            kwargs = arguments | {"persistent": persistent, "out": torch.empty_like(arguments["q"])}
            calls.setdefault(name, []).append(lambda kwargs=kwargs: flash_attn_varlen_func(**kwargs))
        if spec.page is None and not spec.smoke and not args.skip_aiter:
            call = aiter_varlen_call(arguments)
            probe_reference(call)
            calls.setdefault("aiter", []).append(call)
    tokens = sum(spec.q_lens)
    nbytes = tokens * spec.heads * 256 * 2 * 2 + sum(spec.kv_lens) * spec.kv_heads * 256 * 2 * 2
    rows = measure(calls, references, label=spec.name, flops=spec.flops, nbytes={name: nbytes for name in calls},
                   run_count=0 if args.check_only else args.run_count, warmup=args.warmup, repeat=args.repeat,
                   tolerance=0.02, verbose_runs=args.verbose_runs)
    return {"results": rows, "flops": spec.flops, "logical_io_bytes": nbytes,
            "input": {"dtype": "torch.bfloat16", "kv": f"page{spec.page}" if spec.page else "linear THD",
                      "page_order": "reversed physical pages" if spec.page else None,
                      "buffer_seeds": [spec.seed + i for i in range(args.buffers)]}}


# ---- Reports ----

def environment():
    prop = torch.cuda.get_device_properties(torch.cuda.current_device())
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    return {"time_utc": datetime.now(timezone.utc).isoformat(), "arch": gpu_arch(), "gpu": prop.name,
            "compute_units": prop.multi_processor_count, "torch": torch.__version__, "hip": torch.version.hip,
            "flydsl": importlib.metadata.version("flydsl"), "python": sys.version, "git_revision": revision,
            "device_environment": {name: os.environ.get(name) for name in
                                   ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")},
            "sources_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in SOURCES}}


def format_table(report):
    header = ("suite", "case", "backend", "acc", "us", "TFLOPS", "GB/s")
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]

    def text(value, fmt):
        return "-" if value is None else format(value, fmt)
    for record in report["records"]:
        for row in record["results"]:
            lines.append("| " + " | ".join((record["suite"], record["case"]["name"], row["backend"],
                                            text(row["acc"], ".3g"), text(row["us"], ".3f"),
                                            text(row["tflops"], ".2f"), text(row["gbps"], ".1f"))) + " |")
        for reason in record.get("unavailable", {}).values():
            lines.append(f"| {record['suite']} | {record['case']['name']} | skipped: {reason} | | | | |")
    return "\n".join(lines)


def write_report(folder, report):
    with (folder / "result.json").open("w") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    dispatches = [f"- {record['case']['name']}: `{row['aiter_entry']}`: " + ", ".join(f"`{k}`" for k in row["kernels"])
                  for record in report["records"] for row in record["results"] if row["backend"].startswith("aiter")]
    protocol = report["protocol"]
    (folder / "summary.md").write_text(
        f"# MHA accuracy/performance ({report['environment']['gpu']}, {report['environment']['arch']})\n\n"
        f"{'complete' if report['complete'] else 'INCOMPLETE: ' + report.get('error', '')}. "
        f"cudaPerf events; {protocol['input_buffers']} independent buffers x {protocol['run_count']} rounds, "
        f"median per call; acc = max over buffers of sum((ref-out)^2)/sum(ref^2+out^2). "
        "GB/s counts logical Q/K/V/O bytes (gather+CK adds the full KV read/write).\n\n"
        + format_table(report) + "\n\n## AITER dispatch\n\n" + ("\n".join(dispatches) or "none") + "\n")


def run(args, selected):
    if args.check_only:
        args.run_count = 0
    folder = args.output
    folder.mkdir(parents=True, exist_ok=False)
    arch = gpu_arch()
    report = {"complete": False, "environment": environment(), "config": {**vars(args), "output": str(folder)},
              "protocol": {"timer": "pyhip.cudaPerf GPU event interval per call, after its built-in GPU spin",
                           "warmup": args.warmup, "run_count": args.run_count, "repeat": args.repeat,
                           "input_buffers": args.buffers, "samples_per_candidate": args.run_count * args.buffers,
                           "summary": "median of all per-call event times across all buffers and rounds",
                           "checks": "FP32 oracle on every buffer, two bit-exact repeats, native dispatch"},
              "records": []}
    start = time.perf_counter()
    try:
        for suite, spec in selected:
            record = {"suite": suite, "case": spec.to_dict(), "results": [], "unavailable": {}}
            report["records"].append(record)
            if arch not in spec.arches:
                record["unavailable"]["architecture"] = f"requires {' or '.join(spec.arches)}"
                continue
            case_start = time.perf_counter()
            record.update(run_linear(spec, args) if suite == "linear" else run_paged(suite, spec, arch, args))
            record["wall_time_s"] = time.perf_counter() - case_start
            for row in record["results"]:
                us = "-" if row["us"] is None else f"{row['us']:.3f} us {row['tflops']:.2f} TFLOPS"
                print(f"{suite}/{spec.id}/{row['backend']}: acc={row['acc']:.3g} {us}", flush=True)
        if not any(record["results"] for record in report["records"]):
            raise RuntimeError("no supported candidate ran")
        report["complete"] = True
    except BaseException as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["wall_time_s"] = time.perf_counter() - start
        write_report(folder, report)
        print("\n" + format_table(report) + f"\n\nResults: {folder}", flush=True)
    return report


def default_output(suite):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return DATA / f"mha_{suite}_{stamp}_{os.getpid()}"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", choices=("all", *SUITES), default="all")
    parser.add_argument("--case", nargs="+", dest="case_ids", help="exact IDs from --list")
    parser.add_argument("--smoke", action="store_true", help="only the small smoke cases")
    parser.add_argument("--list", action="store_true", help="list cases without GPU initialization")
    parser.add_argument("--gpu", type=int, default=0, help="visible device index")
    parser.add_argument("--buffers", type=int, default=10, help="independent input/output sets, rotated per call")
    parser.add_argument("--run-count", type=int, default=5, help="full buffer rounds; default 5x10=50 samples")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=1, help="rotating-buffer calls per event interval")
    parser.add_argument("--check-only", action="store_true", help="validate every buffer without timing")
    parser.add_argument("--skip-aiter", action="store_true", help="time own kernels only")
    parser.add_argument("--verbose-runs", action="store_true", help="print every sample")
    parser.add_argument("--output", type=Path, help="new result directory (default: mytest/mydata/mha_<suite>_*)")
    args = parser.parse_args(argv)
    if min(args.run_count, args.repeat, args.buffers) < 1 or args.warmup < 0 or args.gpu < 0:
        parser.error("run-count/repeat/buffers must be positive; warmup and gpu nonnegative")
    available = {spec.id for name, specs in SUITES.items() if args.suite in ("all", name) for spec in specs}
    if args.case_ids and (len(args.case_ids) != len(set(args.case_ids)) or not set(args.case_ids) <= available):
        parser.error("--case requires distinct IDs in the selected suite; use --list")
    if args.output is None:
        args.output = default_output(args.suite)
    if args.output.exists():
        parser.error("--output must be a new directory")
    return args


def select(args):
    return [(suite, spec) for suite, specs in SUITES.items() if args.suite in ("all", suite)
            for spec in specs if (args.case_ids is None or spec.id in args.case_ids) and (spec.smoke or not args.smoke)]


def main(argv=None):
    args = parse_args(argv)
    selected = select(args)
    if args.list:
        for suite, spec in selected:
            print(f"{suite}: {spec.id} arches={spec.arches} smoke={spec.smoke} {spec.to_dict()}")
        return selected
    torch.cuda.set_device(args.gpu)
    return run(args, selected)


def test_smoke(tmp_path):
    """Integration smoke: every suite's small cases, checked and timed (prints the result table)."""
    if not torch.cuda.is_available() or torch.version.hip is None or gpu_arch() not in ("gfx942", "gfx950"):
        pytest.skip("requires ROCm gfx942 or gfx950")
    gpu = os.environ.get("MHA_TEST_GPU", "0")
    report = main(["--smoke", "--gpu", gpu, "--buffers", "2", "--run-count", "2", "--warmup", "1",
                   "--output", str(tmp_path / "smoke")])
    assert report["complete"]
    timed = [row for record in report["records"] for row in record["results"]]
    assert timed and all(row["us"] > 0 and len(row["runs"]) == 4 for row in timed)
    assert (tmp_path / "smoke/result.json").exists() and (tmp_path / "smoke/summary.md").exists()


if __name__ == "__main__":
    main()
