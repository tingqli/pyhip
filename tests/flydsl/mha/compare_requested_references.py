"""Same-machine performance gates against the two user-selected reference files.

Planning, source verification, health preflight and gate tests use only stdlib.
ROCm imports occur only after a bounded hardware/process check succeeds. No
PTL/clock/power controls, fallback kernels or reference-source edits are used.
"""

import argparse
import contextlib
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from fnmatch import fnmatchcase
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
REFERENCE_SOURCES = {
    "bf16": ("tests/flydsl/test_attn_8wave_32x32_lkgv.py",
             "c0880420cd10a797c59087d4f73e942aa237020c3f0793b2fa0bcd9a6ac0776a"),
    "fp8": ("tests/flydsl/pa_8wave/pa_prefill_8w32x32.py",
            "620209a023ccb5ea566489774d19edae880dfbcee298613233ed6f01f3b59849"),
}
PROTOCOL_SOURCES = (
    "tests/flydsl/pa_8wave/test_pa_prefill.py",
    "src/misc.py",
    "src/contrib/flydsl/helpers.py",
)


@dataclass(frozen=True)
class ComparisonCase:
    name: str
    kind: str
    q: int
    kv: int
    heads: int
    kv_heads: int
    dq: int = 128
    dv: int = 128
    page: int = 64
    causal: bool = False
    scale_mode: str = "per-tensor"
    seed: int = 20260730

    @property
    def flops(self):
        if self.causal:
            if self.kv < self.q:
                raise ValueError("the requested prefill reference requires causal KV>=Q")
            pairs = self.q * (self.kv - self.q) + self.q * (self.q + 1) // 2
        else:
            pairs = self.q * self.kv
        return 2 * self.heads * pairs * (self.dq + self.dv)

    @property
    def buffers(self):
        if self.kind == "bf16":
            return 10
        # Same 4e9-byte / (warmup+samples) cap as pyhip.run_perftest;
        # count FP8 Q/K/V, BF16 O, FP32 scales, and int32 metadata.
        pages = (self.kv + self.page - 1) // self.page
        scales = self.q * self.heads if self.scale_mode == "per-token" else 1
        size = (self.q * self.heads * (self.dq + 2 * self.dv)
                + pages * self.page * self.kv_heads * (self.dq + self.dv)
                + 4 * (scales + 2 + 2 + 2 + 2 + pages + 1))
        return min(12, max(int(4e9 / size), 1))

    def to_dict(self):
        return {**asdict(self), "q_lens": [self.q], "kv_lens": [self.kv],
                "effective_flops": self.flops, "with_lse": False,
                "reference_standalone_protocol_exact": False,
                "matched_pair_protocol": True,
                "protocol": {"timer": "cudaPerf events", "warmup": 2, "iterations": 10, "rounds": 5,
                             "buffers": self.buffers, "sample_order": "alternating reference/current each sample",
                             "round_statistic": "upper_median_us" if self.kind == "bf16" else "mean_us",
                             "aggregate": "median of five round latency statistics",
                             "pre_interval": "unmodified pyhip.cudaPerf; GPU delay before start event",
                             "layout_conversion_timed": False},
                "input_protocol": ("BF16 random independent buffers; dense reference and paged candidate hold identical logical values; "
                                   "identity page order, unit descales, H=HK, Dq=Dv, full noncausal"
                                   if self.kind == "bf16" else
                                   "BF16 random -> FNUZ quantization; scalar K/V descales, Q mode explicit; "
                                   "reverse pages, zero tail after quantization, identical cloned-content buffer sets"),
                "reference_default": self.name in ("bf16_lkgv_native_default", "fp8_prefill_main")}


def cases(compute_units):
    if isinstance(compute_units, bool) or not isinstance(compute_units, int) or compute_units < 1:
        raise ValueError("compute_units must be an explicit positive integer")
    values = [ComparisonCase("bf16_lkgv_native_default", "bf16", 256 * compute_units, 256 * compute_units,
                             8, 8, page=32, seed=0),
              ComparisonCase("bf16_lkgv_q10240_k2560_p32", "bf16", 10240, 2560, 8, 8, page=32, seed=0),
              ComparisonCase("bf16_lkgv_q10240_k2560_p64", "bf16", 10240, 2560, 8, 8, page=64, seed=0),
              ComparisonCase("fp8_prefill_main", "fp8", 32768, 32768, 16, 1, dq=192, causal=True,
                             scale_mode="per-tensor")]
    for dq, heads in ((128, 8), (192, 16)):
        for mode in ("per-token", "per-tensor"):
            values.append(ComparisonCase(f"fp8_prefill_full_d{dq}_{mode}", "fp8", 10240, 2560,
                                         heads, 1, dq=dq, scale_mode=mode))
    values.append(ComparisonCase("fp8_prefill_causal_d192_per-token", "fp8", 32768, 32768,
                                 16, 1, dq=192, causal=True, scale_mode="per-token"))
    return values


def select_cases(compute_units, kinds, patterns=()):
    available = [case for case in cases(compute_units) if case.kind in kinds]
    for pattern in patterns:
        if not any(fnmatchcase(case.name, pattern) for case in available):
            raise ValueError(f"unknown or unselected case: {pattern}")
    return [case for case in available if not patterns or any(fnmatchcase(case.name, p) for p in patterns)]


def reference_identity(kind, root=ROOT):
    relative, expected = REFERENCE_SOURCES[kind]
    path = root / relative
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(f"reference SHA changed: {relative}; do not silently substitute another source")
    return {"path": relative, "sha256": actual, "factory": "MHA" if kind == "bf16" else "PagedAttention"}


def gate(kind, current_us, reference_us, *, bf16_max_regression_pct=5.0,
         native=False, correctness=False, isolated=False, source_matched=False, protocol_matched=False):
    if kind not in REFERENCE_SOURCES:
        raise ValueError("unknown reference kind")
    if not math.isfinite(bf16_max_regression_pct) or bf16_max_regression_pct < 0:
        raise ValueError("BF16 regression allowance must be finite and nonnegative")
    reasons = [name for name, ok in (("native measurement required", native), ("correctness required", correctness),
               ("isolation required", isolated), ("requested source hash required", source_matched),
               ("matching input/timer protocol required", protocol_matched)) if not ok]
    threshold = 1 + bf16_max_regression_pct / 100 if kind == "bf16" else 1.0
    result = {"reference": REFERENCE_SOURCES[kind][0], "max_latency_ratio": threshold,
              "criterion": "roughly equal (explicit BF16 allowance)" if kind == "bf16" else "not slower than reference",
              "status": "unmatched", "mismatch_reasons": reasons, "latency_ratio": None, "delta_pct": None}
    if current_us is None or reference_us is None:
        result["status"] = "unmatched" if reasons else "not_measured"
        return result
    for value in (current_us, reference_us):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError("latencies must be finite positive measurements")
    ratio = current_us / reference_us
    result.update(latency_ratio=ratio, delta_pct=100 * (ratio - 1))
    if not reasons:
        result["status"] = "passed" if ratio <= threshold else "failed"
    return result


def round_statistic(raw_us, kind):
    if not raw_us or any(not math.isfinite(us) or us <= 0 for us in raw_us):
        raise ValueError("positive finite raw samples required")
    return sorted(raw_us)[len(raw_us) // 2] if kind == "bf16" else statistics.mean(raw_us)


def measure(candidates, case, timer_factory, synchronize):
    """Same timer/rotating buffers for both; all conversion/validation is outside."""
    names = list(candidates)
    if set(names) != {"reference", "current"} or any(len(pool) != case.buffers for pool in candidates.values()):
        raise ValueError("both candidates require the full identical buffer protocol")
    rounds = {name: [] for name in names}
    for trial in range(5):
        for index in range(2):
            for name in names:
                candidates[name][0 if case.kind == "bf16" else index % case.buffers]()
        synchronize()
        raw = {name: [] for name in names}
        for index in range(10):
            order = names if (trial + index) % 2 == 0 else names[::-1]
            buffer = (index if case.kind == "bf16" else index + 2) % case.buffers
            for name in order:
                with timer_factory(case.flops, name) as timer:
                    candidates[name][buffer]()
                raw[name].append(timer.dt() * 1e6)
        for name in names:
            statistic = round_statistic(raw[name], case.kind)
            rounds[name].append({"raw_us": raw[name], "statistic_us": statistic,
                                 "raw_tflops": [case.flops / us / 1e6 for us in raw[name]]})
    return {name: statistics.median(r["statistic_us"] for r in samples) for name, samples in rounds.items()}, rounds


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def health_read(command, timeout_seconds=10):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        process.kill()
        try:
            process.communicate(timeout=1)
        except subprocess.TimeoutExpired:
            pass  # An uninterruptible driver wait cannot justify a GPU reset.
        raise RuntimeError(f"GPU health command timed out (pid {process.pid}); stop, no ROCm imports: {command}") from exc
    if process.returncode:
        raise RuntimeError(f"GPU health command failed ({process.returncode}): {stderr}")
    return json.loads(stdout)


def preflight(smi):
    if any(os.environ.get(name) != "0" for name in
           ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")):
        raise RuntimeError("this comparison requires explicit physical GPU0 mapping")
    if os.environ.get("FLYDSL_COMPILE_ONLY", "0") not in ("0", "") or os.environ.get("FLYDSL_COMPILE_ARCH"):
        raise RuntimeError("unset compile-only/cross-target overrides before native performance comparison")
    if os.environ.get("CUDAPERF") is not None:
        raise RuntimeError("unset CUDAPERF filtering so both candidates use the same enabled timer")
    if __package__:
        from ._hardware import other_processes
    else:
        from _hardware import other_processes
    processes = health_read([smi, "process", "--gpu", "0", "--json"])
    if other_processes(processes, os.getpid()):
        raise RuntimeError("another GPU process exists; refusing the comparison")
    static = health_read([smi, "static", "--gpu", "0", "--asic", "--limit", "--json"])
    return {"processes": processes, "static": static, "settings_changed": False}


def load_reference(kind, directory):
    identity = reference_identity(kind)
    path = directory / f"requested_{kind}_reference.py"
    path.write_bytes((ROOT / identity["path"]).read_bytes())
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def dense_reference_call(case, module, torch):
    # The dense reference has no GQA, causal, scales or unequal Dv support.
    if (case.heads != case.kv_heads or case.dq != 128 or case.dv != 128
            or len(case.q_lens) != 1 or case.q_offset or case.q_lens[0] % 256 or case.kv_lens[0] % 32):
        raise ValueError("requested BF16 reference comparison requires dense H=HK, Dq=Dv=128 and aligned lengths")
    keys, values = case.logical_kv()
    q = case.q.transpose(0, 1).contiguous()
    k = keys[0].transpose(0, 1).to(torch.bfloat16).contiguous()
    v = values[0].transpose(0, 1).to(torch.bfloat16).contiguous()
    v = v.reshape(case.heads, case.kv_lens[0] // 8, 8, case.dv).transpose(2, 3).contiguous()
    out = torch.empty_like(q)
    factory = module.MHA(case.heads, case.dq, 256, 32)
    stream = torch.cuda.current_stream()

    def call():
        factory(q, k, v, out, stream)
        return out.transpose(0, 1)
    return call


def paged_reference_call(case, module, causal, torch):
    # This is the selected BN32 reference, NOT the previous native-BN64
    # source. Its factory deliberately has no memory_mode keyword.
    factory = module.PagedAttention(case.heads, case.kv_heads, case.dq, case.dv, case.page,
                                   causal, case.mode)
    out = torch.empty((case.q.shape[0], case.heads, case.dv), device=case.q.device, dtype=torch.bfloat16)

    def call():
        return factory(case.q, case.k, case.v, case.cq, case.ck, case.indptr, case.indices,
                       max(case.q_lens), max(case.kv_lens), causal, case.qs, case.ks, case.vs,
                       case.last, out=out)
    return call


def run_native(args, result):
    # This function is intentionally unreachable on failed health preflight.
    import torch
    import pyhip
    from flydsl.utils import env
    if __package__:
        from ._testing import BF16_942, FP8, make_case, make_call, assert_close, i32
        from ._runner import environment
        from ._hardware import require_idle_device
    else:
        from _testing import BF16_942, FP8, make_case, make_call, assert_close, i32
        from _runner import environment
        from _hardware import require_idle_device
    actual_cu = torch.cuda.get_device_properties(0).multi_processor_count
    if args.compute_units is not None and actual_cu != args.compute_units:
        raise ValueError("planned CU count differs from native hardware; do not silently resize the default case")
    if not torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx942"):
        raise RuntimeError("both requested references require native gfx942")
    selected = select_cases(actual_cu, args.backend, args.case)
    result["environment"] = environment()
    result["compute_units"] = actual_cu
    result["workloads"] = [case.to_dict() for case in selected]
    kernel_sources = {"bf16": "mha_pa_bf16_942.py", "fp8": "mha_pa_fp8_942.py"}
    saved_debug = {name: getattr(env.debug, name) for name in ("dump_ir", "dump_asm", "enable_debug_info", "dump_dir")}
    default_device = torch.get_default_device()
    try:
        with tempfile.TemporaryDirectory(prefix="mha-requested-references-") as temp, contextlib.chdir(temp):
            for workload in selected:
                result["progress"] = {"case": workload.name, "stage": "load_requested_reference"}
                save(args.output, result)
                module = load_reference(workload.kind, Path(temp))
                # Import side effects are outside timing and never authorize
                # hardware settings. Both implementations compile without dumps.
                env.debug.dump_ir = env.debug.dump_asm = env.debug.enable_debug_info = False
                (module.MHA if workload.kind == "bf16" else module.PagedAttention).cache_clear()
                backend = BF16_942 if workload.kind == "bf16" else FP8
                require_idle_device()
                pools = {"reference": [], "current": []}
                for index in range(workload.buffers):
                    result["progress"] = {"case": workload.name, "stage": "prepare_and_validate", "buffer": index}
                    save(args.output, result)
                    case = make_case((workload.q,), (workload.kv,), dtype=backend.dtype, dq=workload.dq, dv=workload.dv,
                        page=workload.page, heads=workload.heads, kv_heads=workload.kv_heads, mode=workload.scale_mode,
                        poison_tail=False, quantized=workload.kind == "fp8", source_dtype=torch.bfloat16,
                        reverse_pages=workload.kind == "fp8", seed=workload.seed + (index if workload.kind == "bf16" else 0))
                    if workload.kind == "bf16":
                        case.page_order[:] = range(len(case.page_order))
                        case.indices.copy_(i32(case.page_order))
                        reference = dense_reference_call(case, module, torch)
                    else:
                        reference = paged_reference_call(case, module, workload.causal, torch)
                    current = make_call(case, backend, workload.causal)[0]
                    for name, call in (("reference", reference), ("current", current)):
                        result["progress"] = {"case": workload.name, "stage": "correctness", "buffer": index, "candidate": name}
                        save(args.output, result)
                        first = call().clone()
                        assert_close(case, backend, first, None, workload.causal)
                        for _ in range(3):
                            torch.testing.assert_close(call(), first, rtol=0, atol=0)
                        pools[name].append(call)
                require_idle_device()
                result["progress"] = {"case": workload.name, "stage": "measure"}
                save(args.output, result)
                times, rounds = measure(pools, workload,
                    lambda flops, name: pyhip.cudaPerf(flops=flops, name=f"requested_{name}", verbose=0), torch.cuda.synchronize)
                require_idle_device()
                # Re-verify the user's reference on disk; no changed file may
                # retain an earlier label merely because its module was cached.
                reference_identity(workload.kind)
                for relative, expected in result["protocol_sources_sha256"].items():
                    if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected:
                        raise ValueError(f"protocol/helper source changed during comparison: {relative}")
                kernel_source = kernel_sources[workload.kind]
                if hashlib.sha256((HERE / kernel_source).read_bytes()).hexdigest() != result["environment"]["sources_sha256"][kernel_source]:
                    raise ValueError(f"candidate source changed during comparison: {kernel_source}")
                check = gate(workload.kind, times["current"], times["reference"],
                    bf16_max_regression_pct=args.bf16_max_regression_pct, native=True, correctness=True,
                    isolated=True, source_matched=True, protocol_matched=True)
                row = {**workload.to_dict(), "backend": backend.name, "event_interval_us": times,
                       "tflops": {name: workload.flops / us / 1e6 for name, us in times.items()}, "rounds": rounds,
                       "requested_reference": reference_identity(workload.kind), "requested_reference_check": check,
                       "candidate_source_sha256": result["environment"]["sources_sha256"][kernel_source],
                       "correctness_passed": True, "repeated_bit_exact_per_candidate": True,
                       "cross_implementation_bit_exact_required": False, "isolated": True,
                       "reference_unavailable": {}, "baseline_checks": [],
                       "note": "Different lazy-softmax algorithms: each candidate independently checked against FP32. "
                               "All layout conversion outside timing; all per-call auxiliary work inside. "
                               "BF16 upper-median latency TFLOPS derived from that latency, not separately sorted TFLOPS."}
                result["records"].append(row)
                save(args.output, result)
                print("REQUESTED_REFERENCE_RESULT", workload.name, times, row["tflops"], check, flush=True)
    finally:
        torch.set_default_device(default_device)
        for name, value in saved_debug.items():
            setattr(env.debug, name, value)
    result["complete"] = True
    result["acceptance_passed"] = bool(result["records"]) and all(r["requested_reference_check"]["status"] == "passed" for r in result["records"])
    result["progress"] = {"stage": "complete"}
    save(args.output, result)
    return 0 if result["acceptance_passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", nargs="+", choices=("bf16", "fp8"), default=["bf16", "fp8"])
    parser.add_argument("--case", nargs="+", default=[])
    parser.add_argument("--compute-units", type=int, help="explicit plan metadata; actual device must match during native execution")
    parser.add_argument("--bf16-max-regression-pct", type=float, default=5.0,
                        help="explicit interpretation of roughly equal; FP8 always has a zero-regression threshold")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--list-cases", action="store_true")
    action.add_argument("--preflight-only", action="store_true", help="bounded hardware reads only; do not import ROCm or run kernels")
    parser.add_argument("--smi", default="/opt/rocm/bin/amd-smi")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.output.exists():
        parser.error("output already exists; choose a new report")
    if not math.isfinite(args.bf16_max_regression_pct) or args.bf16_max_regression_pct < 0:
        parser.error("BF16 allowance must be finite and nonnegative")
    if args.compute_units is not None and args.compute_units < 1:
        parser.error("compute-units must be positive")
    identities = {kind: reference_identity(kind) for kind in args.backend}
    result = {"time_utc": datetime.now(timezone.utc).isoformat(), "requested_references": identities,
              "complete": False, "acceptance_passed": False, "records": [], "settings_changed": False,
              "acceptance_scope": "selected same-machine cases only; not all source-supported shapes",
              "protocol_sources_sha256": {relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
                                          for relative in PROTOCOL_SOURCES},
              "bf16_max_regression_pct": args.bf16_max_regression_pct, "fp8_max_regression_pct": 0.0,
              "not_a_historical_mi308_ptl_gate": True}
    if args.list_cases:
        if args.compute_units is None:
            parser.error("CPU-only planning requires --compute-units (MI325=304); never guess an 80-CU default")
        selected = select_cases(args.compute_units, args.backend, args.case)
        result.update(gpu_queried=False, executed=False, status="planned; not measured", compute_units=args.compute_units,
                      workloads=[case.to_dict() for case in selected])
        save(args.output, result)
        print(json.dumps(result, indent=2))
        return 0
    result["progress"] = {"stage": "health_preflight"}
    save(args.output, result)
    try:
        result["health_before"] = preflight(args.smi)
        if args.preflight_only:
            result.update(executed=False, status="health reads succeeded; no native or performance validation",
                          progress={"stage": "health_preflight_complete"})
            save(args.output, result)
            return 0
        result["progress"] = {"stage": "native_import"}
        save(args.output, result)
        return run_native(args, result)
    except Exception as exc:
        result.update(error={"type": type(exc).__name__, "message": str(exc)}, complete=False, acceptance_passed=False)
        save(args.output, result)
        raise


if __name__ == "__main__":
    raise SystemExit(main())