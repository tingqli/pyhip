"""Historical documented targets with their original source/protocol provenance.

For the user's newer dense-BF16 / BN32-FP8 relative gates, use
compare_requested_references.py; do not relabel these historical comparisons.
"""

import argparse
import contextlib
from dataclasses import replace
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile

import torch

if __package__:
    from ._hardware import ptl_experiment, require_idle_device
    from ._runner import environment, profile_round, event_round, save, make_performance_case
    from ._perf_cases import select_workloads, baseline_status
    from ._testing import FP8, BF16_942, SWA, make_call, assert_close, effective_flops
    from .validate_preservation import load_original, original_call, GIT_ROOT, SOURCE_BRANCH, SOURCES
else:
    from _hardware import ptl_experiment, require_idle_device
    from _runner import environment, profile_round, event_round, save, make_performance_case
    from _perf_cases import select_workloads, baseline_status
    from _testing import FP8, BF16_942, SWA, make_call, assert_close, effective_flops
    from validate_preservation import load_original, original_call, GIT_ROOT, SOURCE_BRANCH, SOURCES


def four_wave(directory):
    relative = "tests/flydsl/pa_4wave/pa_prefill_4wave.py"
    source = subprocess.check_output(["git", "show", f"{SOURCE_BRANCH}:{relative}"], cwd=GIT_ROOT)
    path = directory / "documented_four_wave.py"
    path.write_bytes(source)
    spec = importlib.util.spec_from_file_location("documented_four_wave", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, hashlib.sha256(source).hexdigest()


def branch_four_wave_call(case, module, causal):
    factory = module.MHA(case.heads, case.kv_heads, case.dq, case.dv, case.page, causal,
                         window_left=case.window_left, has_sink=case.sinks is not None)
    out = torch.empty(case.q.shape[0], case.heads, case.dv, dtype=torch.bfloat16, device=case.q.device)

    def call():
        return factory(case.q, case.k, case.v, case.cq, case.ck, case.indptr, case.indices,
                       max(case.q_lens), max(case.kv_lens), causal, case.qs, case.ks, case.vs, case.last,
                       out, sink_ptr=case.sinks)
    return call


def original_swa_call(case, module, causal):
    factory = module.PagedAttention(case.heads, case.kv_heads, case.dq, case.dv, case.page,
                                   causal, case.mode, window_left=case.window_left, has_sink=case.sinks is not None)
    out = torch.empty(case.q.shape[0], case.heads, case.dv, dtype=torch.bfloat16, device=case.q.device)

    def call():
        return factory(case.q, case.k, case.v, case.cq, case.ck, case.indptr, case.indices,
                       max(case.q_lens), max(case.kv_lens), causal, case.qs, case.ks, case.vs, case.last,
                       out=out, sink_ptr=case.sinks)
    return call


def run(args, directory):
    from flydsl.utils import env
    names = ("dump_ir", "dump_asm", "enable_debug_info")
    saved = {name: getattr(env.debug, name) for name in names}
    try:
        return _run(args, directory, env)
    finally:
        for name, value in saved.items():
            setattr(env.debug, name, value)


def _run(args, directory, env):
    backend = {"fp8": FP8, "bf16": BF16_942, "swa": SWA}[args.backend]
    def check_isolation():
        if not args.allow_contention:
            require_idle_device()
    check_isolation()
    result = {"environment": environment(), "ptl_experiment": args.ptl, "records": [], "complete": False,
              "isolated": not args.allow_contention,
              "measurement_scope": "non-exclusive diagnostic only" if args.allow_contention else "isolated comparison"}
    save(args.output, result)
    if not backend.available:
        raise RuntimeError(f"{backend.name} is unavailable on this GPU; no substitute backend is allowed")
    if backend == SWA and result["environment"]["arch"] != "gfx950":
        raise RuntimeError("the original SWA baseline requires gfx950; use the matrix runner for gfx942 diagnostics")
    if args.backend == "fp8":
        patterns = ("fp8_native_410t",)
    elif args.backend == "bf16":
        patterns = ("historical_bf16_250t_shape", "historical_bf16_d192_events", "full_d128_p64", "full_d192_p64")
    else:
        patterns = ("swa_kv131072_d128", "swa_kv131072_d192")
    workloads = select_workloads(args.case or patterns)
    for workload in workloads:
        reason = workload.unsupported(backend)
        if reason:
            raise ValueError(f"{workload.name}: {reason}")
    original = load_original(backend.module, directory)
    original_name = "original_1wave" if backend == SWA else "original_8wave"
    result["original_source"] = SOURCES[backend.module]
    module4, hash4 = four_wave(directory) if args.four_wave else (None, None)
    # Original main's import enables debug dumping. Keep both measured kernels
    # in the same compiler mode, and keep dump side effects out of the checkout.
    env.debug.dump_ir, env.debug.dump_asm, env.debug.enable_debug_info = False, False, False
    for workload in workloads:
        if backend == BF16_942:
            # Original main caches its compiled launch by dtype/device only.
            # Reset the temporary original factory BETWEEN workloads, never
            # between rotating buffers or inside a measured interval.
            original.PagedAttention.cache_clear()
        check_isolation()
        protocol = replace(workload.protocol,
            warmup=workload.protocol.warmup if args.warmup is None else args.warmup,
            rounds=workload.protocol.rounds if args.rounds is None else args.rounds,
            iterations=workload.protocol.iterations if args.iterations is None else args.iterations)
        warmup, trials, iterations = protocol.warmup, protocol.rounds, protocol.iterations
        dq, q, kv, heads, page = workload.dq, max(workload.q_lens), max(workload.kv_lens), workload.heads, workload.page
        cases = [make_performance_case(workload, backend, index) for index in range(protocol.buffers)]
        calls = {"current": []}
        for case in cases:
            pool = {"current": make_call(case, backend, workload.causal)[0]}
            pool[original_name] = (original_swa_call(case, original, workload.causal) if backend == SWA else
                                   original_call(case, backend, original, workload.causal))
            if module4 is not None:
                pool["branch_4wave_comparator"] = branch_four_wave_call(case, module4, workload.causal)
            first = {}
            for name, call in pool.items():
                first[name] = call().clone()
                assert_close(case, backend, first[name], None, workload.causal)
                for _ in range(3):
                    torch.testing.assert_close(call(), first[name], rtol=0, atol=0)
                calls.setdefault(name, []).append(call)
            torch.testing.assert_close(first["current"], first[original_name], rtol=0, atol=0)
        check_isolation()
        for index in range(warmup):
            for pool in calls.values():
                pool[index % len(pool)]()
        rounds = {name: [] for name in calls}
        metric = "attention" if protocol.timer == "profiler" else "interval"
        for trial in range(trials):
            for name in list(calls) if trial % 2 == 0 else reversed(calls):
                sample = (profile_round(calls[name][0], iterations=iterations, warmup=protocol.sample_warmup)
                          if protocol.timer == "profiler" else event_round(calls[name], iterations=iterations, warmup=0))
                rounds[name].append(sample)
                print("BASELINE_ROUND", args.backend, dq, q, kv, heads, page, name, trial,
                      sample[metric]["mean_us"], flush=True)
        check_isolation()
        flops = effective_flops(cases[0], workload.causal)
        assert flops == workload.flops
        latency = {name: statistics.median(r[metric]["mean_us"] for r in samples) for name, samples in rounds.items()}
        if any(not math.isfinite(us) or us <= 0 for us in latency.values()):
            raise ValueError("measurement latency must be finite and positive")
        totals = ({name: statistics.median(r["total"]["mean_us"] for r in samples) for name, samples in rounds.items()}
                  if protocol.timer == "profiler" else None)
        comparison_checks = {name: baseline_status(workload,
            ("branch_4wave_static_comparator" if len(workload.q_lens) == 1 else "branch_4wave_dynamic_comparator")
                if name == "branch_4wave_comparator" else backend.name,
            result["environment"], flops / latency[name] / 1e6, protocol=protocol,
            isolated=not args.allow_contention) for name in calls}
        row = {"backend": backend.name, "dq": dq, "q": q, "kv": kv, "heads": heads, "page": page,
               "effective_flops": flops, "latency_us": latency,
               "attention_us": latency if protocol.timer == "profiler" else None,
               "event_interval_us": latency if protocol.timer == "events" else None, "total_gpu_us": totals,
               "tflops": {name: flops / us / 1e6 for name, us in latency.items()}, "rounds": rounds,
               "four_wave_source_sha256": hash4,
               "four_wave_schedule": None if module4 is None else "static" if len(workload.q_lens) == 1 else "dynamic",
               "original_current_bit_exact": True,
               "workload": workload.to_dict(),
               "baseline_checks": comparison_checks["current"], "comparison_baseline_checks": comparison_checks,
               "protocol": {"warmup": warmup, "rounds": trials, "iterations": iterations,
                            "timer": protocol.timer, "buffers": protocol.buffers, "sample_warmup": protocol.sample_warmup},
               "baseline_note": "branch 4-wave comparator is separate from current 8-wave; the exact August binary/compiler was not archived"}
        result["records"].append(row)
        save(args.output, result)
        print("BASELINE_RESULT", json.dumps({k:v for k,v in row.items() if k != "rounds"}), flush=True)
    result["complete"] = True
    save(args.output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("fp8", "bf16", "swa"), required=True)
    parser.add_argument("--ptl", choices=("current", "VECTOR,F8", "VECTOR,BF16"), default="current")
    parser.add_argument("--four-wave", action="store_true", help="explicit historical comparator only, never a production fallback")
    parser.add_argument("--allow-contention", action="store_true", help="current-PTL diagnostic only; never baseline acceptance")
    parser.add_argument("--case", nargs="+", default=[])
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.four_wave and args.backend != "bf16":
        parser.error("the historical 250T four-wave comparator is BF16 only")
    if ((args.warmup is not None and args.warmup < 0) or (args.rounds is not None and args.rounds < 1)
            or (args.iterations is not None and args.iterations < 2)):
        parser.error("warmup>=0, rounds>=1 and iterations>=2 required")
    if args.ptl != "current" and (args.backend == "fp8") != (args.ptl == "VECTOR,F8"):
        parser.error("PTL format must match the selected input dtype")
    if args.allow_contention and args.ptl != "current":
        parser.error("non-exclusive diagnostics must not change PTL")
    args.output = args.output.resolve()
    with tempfile.TemporaryDirectory(prefix="mha_documented_baselines_") as temp:
        with ptl_experiment(args.ptl, args.output.with_suffix(".hardware.json")), contextlib.chdir(temp):
            run(args, Path(temp))


if __name__ == "__main__":
    main()