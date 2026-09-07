"""Shared functional/performance/resource CLI for paged MHA and SWA."""

import argparse
from collections import Counter
import contextlib
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import itertools
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys

import torch

if __package__:
    from ._testing import (BACKENDS, SWA, BF16_942, make_case, make_call, assert_close,
                          effective_flops, gpu_arch)
    from ._references import aiter_call, aiter_linear_call, aiter_opus_call, probe_reference, ReferenceUnavailable
    from ._perf_cases import Workload, Protocol, select_workloads, baseline_status
    from ._hardware import ptl_experiment, require_idle_device
else:
    from _testing import (BACKENDS, SWA, BF16_942, make_case, make_call, assert_close,
                         effective_flops, gpu_arch)
    from _references import aiter_call, aiter_linear_call, aiter_opus_call, probe_reference, ReferenceUnavailable
    from _perf_cases import Workload, Protocol, select_workloads, baseline_status
    from _hardware import ptl_experiment, require_idle_device


HERE = Path(__file__).resolve().parent


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
        try:
            result["limits"] = json.loads(subprocess.check_output(["/opt/rocm/bin/amd-smi", "static", "--gpu", bdf, "--limit", "--json"], text=True))
        except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
            result["limits_unavailable"] = str(exc)
    return result


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def _quantile(values, fraction):
    position = (len(values) - 1) * fraction
    low = int(position)
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (position - low)


def profile_round(call, *, iterations=100, warmup=20):
    """Same IQR statistic for every backend, retaining all raw GPU events.

    Per-call totals include auxiliary GPU work (e.g. original BF16 counter
    initialization); attention-only latency is reported separately, not hidden.
    """
    if iterations < 2 or warmup < 0:
        raise ValueError("profiling requires iterations >= 2 and warmup >= 0")
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as profiler:
        for _ in range(iterations):
            call()
        torch.cuda.synchronize()
    events = [event for event in profiler.events() if "CUDA" in str(event.device_type)]
    attention = [event for event in events if any(name in event.name.lower() for name in
                 ("attention", "attn_kernel", "_swa", "fmha", "mha_", "flash"))]
    # The first result also records dispatches for external reference kernels
    # whose names may not follow our attention convention.
    count = len(events) // iterations
    if count < 1 or len(events) != count * iterations:
        raise AssertionError(f"nonuniform dispatch count {len(events)} for {iterations} calls")
    elapsed = lambda event: event.time_range.elapsed_us()
    total = [sum(elapsed(e) for e in events[i * count:(i + 1) * count]) for i in range(iterations)]
    primary = [elapsed(event) for event in attention] if len(attention) == iterations else total

    def aggregate(samples):
        ordered = sorted(samples[1:])
        q1, q3 = _quantile(ordered, 0.25), _quantile(ordered, 0.75)
        lower, upper = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
        kept = [i for i in range(1, len(samples)) if lower <= samples[i] <= upper]
        return {"mean_us": statistics.mean(samples[i] for i in kept), "raw_us": samples, "kept_indices": kept}
    return {"total": aggregate(total), "attention": aggregate(primary),
            "dispatches_per_call": count, "dispatch_names": sorted({event.name for event in events})}


def result_path(args, kind):
    if args.output is None:
        return HERE / "results" / f"{kind}.json"
    if args.mode == "all":
        return args.output.with_name(f"{args.output.stem}.{kind}{args.output.suffix or '.json'}")
    return args.output


def event_round(calls, *, iterations=50, warmup=10):
    """Historical CUDA/HIP event median, rotating independent input buffers.

    This interval includes GPU-side auxiliary work and dispatch gaps. Do not
    mislabel it as the profiler's attention-only kernel latency.
    """
    if iterations < 1 or warmup < 0 or not calls:
        raise ValueError("invalid rotating-event measurement parameters")
    for index in range(warmup):
        calls[index % len(calls)]()
    torch.cuda.synchronize()
    events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(iterations)]
    raw = []
    for index, (start, end) in enumerate(events):
        start.record()
        calls[index % len(calls)]()
        end.record()
        end.synchronize()
        raw.append(start.elapsed_time(end) * 1000)
    median = statistics.median(raw)
    values = {"mean_us": median, "median_us": median, "raw_us": raw,
              "kept_indices": list(range(iterations))}
    return {"interval": values, "timer": "events", "buffer_count": len(calls)}


def benchmark_workloads(args):
    if args.matrix == "documented":
        return select_workloads(args.case)
    protocol = Protocol("profiler", 1200, 5, 100, 1)
    if args.matrix == "custom":
        if not args.q or not args.kv:
            raise ValueError("--matrix custom requires --q and --kv")
    if (args.heads or 16) % (args.kv_heads or 1):
        raise ValueError("heads must be divisible by KV heads")
    result = []
    if args.suite in ("pa", "all"):
        for dq, q, kv, causal, page, dv, window in itertools.product(
                args.dq or (128, 192), args.q or (10240,), args.kv or (2560, 2583),
                args.causal if args.causal is not None else (0, 1),
                args.page or (64,), args.dv or (128,), args.window or (-1,)):
            result.append(Workload(f"custom_d{dq}_v{dv}_q{q}_k{kv}_c{causal}_p{page}_w{window}",
                (q,) * (args.batch or 1), (kv,) * (args.batch or 1), dq=dq, dv=dv,
                heads=args.heads or 16, kv_heads=args.kv_heads or 1, page=page, causal=bool(causal),
                window=window, sink=args.sink is True,
                scale_mode=args.scale_mode or "per-token", input_kind="native-cast", protocol=protocol,
                backends=tuple(backend.name for backend in BACKENDS), source="explicit CLI / quick smoke"))
    if args.suite in ("swa", "all"):
        for dq, q, kv, window, page, dv in itertools.product(args.dq or (128, 192), args.q or (16384,),
                    args.kv or (131072,), args.window or (128,), args.page or (64,), args.dv or (128,)):
            result.append(Workload(f"custom_swa_d{dq}_v{dv}_q{q}_k{kv}_w{window}_p{page}",
                (q,) * (args.batch or 1), (kv,) * (args.batch or 1), dq=dq, dv=dv, page=page,
                heads=args.heads or 16, kv_heads=args.kv_heads or 1, window=window, causal=True,
                sink=args.sink is not False, scale_mode=args.scale_mode or "per-token",
                protocol=protocol, backends=(SWA.name,), source="explicit CLI / quick smoke"))
    return tuple(result)


def measurement_protocol(args, workload):
    base = workload.protocol
    timer = args.timer or base.timer
    return Protocol(timer,
                    base.warmup if args.warmup is None else args.warmup,
                    base.rounds if args.rounds is None else args.rounds,
                    base.iterations if args.iterations is None else args.iterations,
                    base.buffers if args.buffers is None else args.buffers,
                    base.sample_warmup if timer == "profiler" else 0)


def make_performance_case(workload, backend, buffer_index=0):
    """One input protocol for the matrix runner and original/current reproducer."""
    bf16_source = workload.input_kind == "bf16-source"
    return make_case(workload.q_lens, workload.kv_lens, dtype=backend.dtype, dq=workload.dq,
                     dv=workload.dv, page=workload.page, heads=workload.heads, kv_heads=workload.kv_heads,
                     mode=workload.scale_mode, window_left=workload.window, has_sink=workload.sink,
                     seed=workload.seed + buffer_index, poison_tail=False,
                     quantized=backend.fp8 and bf16_source, padding_before_quantization=bf16_source,
                     source_dtype=torch.bfloat16 if bf16_source else torch.float32)


def case_selected(args, workload):
    # Documented sweeps are filtered, never silently rewritten into a shape
    # that happens to be legal for a backend.
    for name, value in (("dq", workload.dq), ("dv", workload.dv), ("page", workload.page),
                        ("causal", int(workload.causal)), ("window", workload.window)):
        choice = getattr(args, name, None)
        if choice is not None and value not in choice:
            return False
    for name, value in (("batch", len(workload.q_lens)), ("heads", workload.heads), ("kv_heads", workload.kv_heads)):
        choice = getattr(args, name, None)
        if choice is not None and choice != value:
            return False
    for name, values in (("q", workload.q_lens), ("kv", workload.kv_lens)):
        choice = getattr(args, name, None)
        if choice is not None and any(value not in choice for value in values):
            return False
    if args.scale_mode is not None and args.scale_mode != workload.scale_mode:
        return False
    return args.sink is None or workload.sink == args.sink


def resource_fields(text):
    """AMDHSA vgpr_count includes AGPRs on CDNA; retain both raw and split counts."""
    names = ("group_segment_fixed_size", "private_segment_fixed_size", "vgpr_count",
             "sgpr_count", "vgpr_spill_count", "sgpr_spill_count", "agpr_count")
    result = {}
    for name in names:
        match = re.search(r"\." + name + r":\s*(\d+)", text)
        if match is None and name != "agpr_count":
            raise ValueError(f"missing AMDHSA metadata: {name}")
        result[name] = int(match[1]) if match else 0
    result["vector_register_count"] = result["vgpr_count"] - result["agpr_count"]
    match = re.search(r"\.amdhsa_accum_offset\s+(\d+)", text)
    result["accum_offset"] = int(match[1]) if match else None
    return result


def benchmark(args, backends):
    result = {"environment": environment(), "records": [], "unavailable": [], "complete": False,
              "matrix": args.matrix}
    output = result_path(args, "performance")
    workloads = [case for case in benchmark_workloads(args) if case_selected(args, case)]
    if not workloads:
        raise ValueError("no workload matches the selected filters")
    if args.require_baseline:
        eligible = [check for workload in workloads for backend in backends
                    if workload.unsupported(backend) is None
                    for check in baseline_status(workload, backend.name, result["environment"],
                        protocol=measurement_protocol(args, workload), isolated=not args.allow_contention)
                    if check["backend"] == backend.name]
        if not eligible or any(check["mismatch_reasons"] or check["minimum_tflops"] is None for check in eligible):
            result["baseline_preflight"] = eligible
            result["baseline_error"] = "requires a comparable baseline with a documented numeric acceptance threshold"
            save(output, result)
            raise RuntimeError("documented baseline conditions do not match; no timed kernel was launched")
    save(output, result)
    for backend in backends:
        if not backend.available:
            result["unavailable"].append({"backend": backend.name, "reason": f"requires {backend.arch}; detected {gpu_arch()}"})
            continue
        for workload in workloads:
            reason = workload.unsupported(backend)
            if reason:
                result["unavailable"].append({"backend": backend.name, "case": workload.name, "reason": reason})
                continue
            if not args.allow_contention:
                require_idle_device()
            protocol = measurement_protocol(args, workload)
            cases = [make_performance_case(workload, backend, index) for index in range(protocol.buffers)]
            configurations = [("auto", {})]
            if backend == SWA and args.tiles:
                configurations += [(f"q{qt}_bn{bn}", {"query_tile": qt, "block_n": bn})
                                   for qt, bn in itertools.product((16, 32), (16, 32, 64))]
            for config, options in configurations:
                calls = {"flydsl": []}
                unavailable = {}
                factories = {}
                if args.aiter != "off":
                    factories["aiter_5d"] = aiter_call
                    if not backend.fp8:
                        factories["aiter_ck_linear_prepared"] = aiter_linear_call
                    if backend.arch == "gfx950" and workload.window < 0 and not workload.sink:
                        factories["aiter_opus_linear_prepared"] = aiter_opus_call
                for case in cases:
                    call, out, _ = make_call(case, backend, workload.causal, **options)
                    call()
                    reference, _ = assert_close(case, backend, out, None, workload.causal)
                    first = out.clone()
                    for _ in range(3):
                        torch.testing.assert_close(call(), first, rtol=0, atol=0)
                    calls["flydsl"].append(call)
                    for name, factory in factories.items():
                        if name in unavailable:
                            continue
                        try:
                            ref_call = factory(case, workload.causal)
                            actual = probe_reference(ref_call)
                        except ReferenceUnavailable as exc:
                            unavailable[name] = str(exc)
                            calls.pop(name, None)
                            if args.aiter == "required":
                                raise
                            continue
                        tolerance = 0.1 if backend.fp8 else 0.02
                        torch.testing.assert_close(actual.float(), reference, rtol=tolerance, atol=tolerance)
                        calls.setdefault(name, []).append(ref_call)
                assert all(len(pool) == protocol.buffers for pool in calls.values())
                if not args.allow_contention:
                    require_idle_device()
                for index in range(protocol.warmup):
                    for pool in calls.values():
                        pool[index % len(pool)]()
                rounds = {name: [] for name in calls}
                metric = "attention" if protocol.timer == "profiler" else "interval"
                for trial in range(protocol.rounds):
                    order = list(calls) if trial % 2 == 0 else list(reversed(calls))
                    for name in order:
                        measured = (profile_round(calls[name][0], iterations=protocol.iterations, warmup=protocol.sample_warmup)
                                    if protocol.timer == "profiler" else event_round(calls[name], iterations=protocol.iterations, warmup=0))
                        rounds[name].append(measured)
                        print("MHA_ROUND", workload.name, backend.name, config, name, trial,
                              measured[metric]["mean_us"], "us", flush=True)
                if not args.allow_contention:
                    require_idle_device()
                times = {name: statistics.median(x[metric]["mean_us"] for x in values) for name, values in rounds.items()}
                if any(not math.isfinite(us) or us <= 0 for us in times.values()):
                    raise ValueError("measurement latency must be finite and positive")
                totals = ({name: statistics.median(x["total"]["mean_us"] for x in values) for name, values in rounds.items()}
                          if protocol.timer == "profiler" else None)
                flops = workload.flops
                assert effective_flops(cases[0], workload.causal) == flops
                checks = baseline_status(workload, backend.name, result["environment"], flops / times["flydsl"] / 1e6,
                                         protocol=protocol, isolated=not args.allow_contention, config=config)
                entry = {"backend": backend.name, "config": config, **workload.to_dict(),
                         "q": max(workload.q_lens), "kv": max(workload.kv_lens), "batch": len(workload.q_lens),
                         "attention_us": times if protocol.timer == "profiler" else None,
                         "event_interval_us": times if protocol.timer == "events" else None,
                         "total_gpu_us": totals,
                         "tflops": {name: flops / us / 1e6 for name, us in times.items()},
                         "effective_flops": flops, "rounds": rounds, "reference_unavailable": unavailable,
                         "reference_note": "5D uses identical caches; explicit linear comparison is prepared before timing",
                         "baseline_checks": checks, "isolated": not args.allow_contention,
                         "protocol": {"initial_warmup": protocol.warmup, "rounds": protocol.rounds,
                                      "sample_warmup": protocol.sample_warmup, "iterations": protocol.iterations,
                                      "buffers": protocol.buffers, "timer": protocol.timer,
                                      "statistic": "drop first;1.5IQR mean;median rounds" if protocol.timer == "profiler" else "median of all rotating-buffer event samples"}}
                result["records"].append(entry)
                save(output, result)
                print("MHA_RESULT", json.dumps({k: v for k, v in entry.items() if k != "rounds"}), flush=True)
                if args.require_baseline:
                    eligible = [check for check in checks if check["backend"] == backend.name]
                    if eligible and any(check["status"] != "passed" for check in eligible):
                        raise RuntimeError(f"documented baseline not reproduced: {eligible}")
    if args.require_baseline and not any(check["backend"] == row["backend"] and check["status"] == "passed"
            for row in result["records"] for check in row["baseline_checks"]):
        raise RuntimeError("no directly comparable documented baseline selected; a different kernel/shape is not a pass")
    result["complete"] = True
    save(output, result)
    return result


def audit(args, backends):
    from flydsl.utils import env

    output = result_path(args, "resources")
    result = {"environment": environment(), "records": [], "unavailable": []}
    saved = (env.compile.arch, env.compile.compile_only, env.debug.dump_ir, env.debug.dump_dir, env.runtime.enable_cache)
    env.runtime.enable_cache = False
    try:
        for backend in backends:
            arches = (gpu_arch(),) if backend == SWA and not args.cross_compile else ("gfx942", "gfx950") if backend == SWA else (backend.arch,)
            for arch in arches:
                if arch not in ("gfx942", "gfx950"):
                    continue
                if arch != gpu_arch() and not args.cross_compile:
                    result["unavailable"].append({"backend": backend.name, "arch": arch, "reason": "use --cross-compile for nonexecuted ISA audit"})
                    continue
                for dq, causal, with_lse in itertools.product(args.dq or (128, 192), (True,) if backend == SWA else (False, True), (False, True)):
                    configs = itertools.product((16, 32), (16, 32, 64)) if backend == SWA else ((None, None),)
                    for qt, bn in configs:
                        case = make_case((10240,), (10240 if backend == BF16_942 and causal else 2560,),
                            dtype=backend.dtype, dq=dq, heads=16, window_left=128 if backend == SWA else -1,
                            has_sink=backend == SWA, poison_tail=False)
                        module = backend.load()
                        out = torch.empty(case.q.shape[0], 16, 128, device="cuda", dtype=torch.bfloat16)
                        lse = torch.empty(case.q.shape[:2], device="cuda") if with_lse else case.ks
                        stream = torch.cuda.current_stream()
                        tag = f"{backend.name}_{arch}_d{dq}_c{int(causal)}_lse{int(with_lse)}" + (f"_q{qt}_bn{bn}" if qt else "")
                        directory = args.dump_root / tag
                        env.compile.arch, env.compile.compile_only = arch, True
                        env.debug.dump_ir, env.debug.dump_dir = True, str(directory)
                        common = (case.q.view(-1), case.k.view(-1), case.v.view(-1), out.view(-1), lse.view(-1),
                                  case.cq, case.indptr, case.indices, case.last, case.qs.view(-1), case.ks, case.vs)
                        if backend.fp8:
                            launch, values = module._launch_attention, (*common, 16, 1, case.k.shape[0], 1, 10240, dq,
                                causal, True, with_lse, dq**-0.5, backend.memory_mode == "lds", stream)
                        elif backend == SWA:
                            launch, values = module._launch, (*common, case.sinks, 16, 1, case.k.shape[0], 1, 10240, dq,
                                16*dq, dq, 16*128, 128, 128, True, True, with_lse, dq**-0.5, bn, qt, int(arch[3:]), stream)
                        elif backend == BF16_942:
                            launch = module._build_attention(16, 1, dq, 128, 64, causal, with_lse=with_lse)
                            count = torch.zeros(torch.cuda.get_device_properties(0).multi_processor_count + 1, device="cuda", dtype=torch.int32)
                            values = (case.q, case.k, case.v, case.cq, case.ck, case.indptr, case.indices,
                                      case.qs.view(-1), case.ks, case.vs, case.last, out, lse, count,
                                      torch.cuda.get_device_properties(0).multi_processor_count, stream)
                        else:
                            values = (*common, case.ks, 16, 1, case.k.shape[0], 1, 10240, dq, -1, False,
                                      16*dq, dq, 16*128, 128, True, causal, with_lse, dq**-0.5)
                            if backend.persistent:
                                grid = torch.cuda.get_device_properties(0).multi_processor_count
                                count = torch.zeros(2, device="cuda", dtype=torch.int32)
                                launch, values = module._launch_persistent, (*values, count, grid, stream)
                            else:
                                launch, values = module._launch_attention, (*values, stream)
                        directory.mkdir(parents=True, exist_ok=True)
                        with (directory / "compile.log").open("w") as log, contextlib.redirect_stdout(log):
                            assert launch(*values) is None
                        files = list(directory.rglob("*final_isa.s"))
                        if len(files) != 1:
                            raise AssertionError((tag, files))
                        text = files[0].read_text()
                        fields = resource_fields(text)
                        # Resource reporting is not a blanket zero-spill
                        # promise: BF16/persistent and optional SWA tile sizes
                        # must be compared with their same-compiler baseline.
                        spills = bool(fields["private_segment_fixed_size"] or fields["vgpr_spill_count"] or fields["sgpr_spill_count"])
                        if args.strict_resources or backend.fp8:
                            assert not spills, (tag, fields)
                        assert fields["group_segment_fixed_size"] <= (65536 if arch == "gfx942" else 163840)
                        if arch == "gfx942":
                            for forbidden in ("permlane32_swap", "permlane16_swap", "ds_read_tr", "v_mfma_scale", "v_cvt_pk_bf16_f32"):
                                assert forbidden not in text, (tag, forbidden)
                        counts = Counter(re.findall(r"^\s+((?:v_|s_|ds_|buffer_|global_)\w+)", text, re.M))
                        entry = {"backend": backend.name, "target": arch, "executed": False, "dq": dq,
                                 "causal": causal, "with_lse": with_lse, "query_tile": qt, "block_n": bn,
                                 "resources": fields, "has_spills": spills,
                                 "instructions": dict(counts), "isa": str(files[0]),
                                 "isa_sha256": hashlib.sha256(text.encode()).hexdigest()}
                        result["records"].append(entry)
                        save(output, result)
                        print("MHA_AUDIT", tag, fields, flush=True)
    finally:
        env.compile.arch, env.compile.compile_only, env.debug.dump_ir, env.debug.dump_dir, env.runtime.enable_cache = saved
    save(output, result)
    return result


def main(test_file, suite="pa"):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("functional", "performance", "audit", "all"), default="functional")
    parser.add_argument("--suite", choices=("pa", "swa", "all"), default=suite)
    parser.add_argument("--backend", nargs="+", choices=("auto", *(b.name for b in BACKENDS), SWA.name), default=["auto"])
    parser.add_argument("--matrix", choices=("documented", "quick", "custom"), default="documented",
                        help="documented: original reports; quick: small smoke matrix; custom: explicit shapes")
    parser.add_argument("--case", nargs="+", default=[], help="documented case names or glob patterns")
    parser.add_argument("--list-cases", action="store_true", help="CPU-only plan; no imports of kernels or GPU queries")
    parser.add_argument("--dq", "--head-dim", type=int, nargs="+", choices=(128, 192))
    parser.add_argument("--dv", type=int, nargs="+", choices=(128, 192))
    parser.add_argument("--page", type=int, nargs="+", choices=(32, 64, 128))
    parser.add_argument("--q", "--q-len", type=int, nargs="+")
    parser.add_argument("--kv", "--kv-len", type=int, nargs="+")
    parser.add_argument("--batch", type=int)
    parser.add_argument("--heads", type=int)
    parser.add_argument("--kv-heads", type=int)
    parser.add_argument("--causal", type=int, nargs="+", choices=(0, 1))
    parser.add_argument("--window", type=int, nargs="+")
    parser.add_argument("--sink", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--scale-mode", choices=("per-token", "per-tensor"))
    parser.add_argument("--tiles", action="store_true", help="benchmark all SWA QT/BN combinations as well as auto")
    parser.add_argument("--aiter", choices=("auto", "required", "off"), default="auto")
    parser.add_argument("--gather-linear", action="store_true",
                        help="SWA: compare direct, gather-only, CK linear-only and gather+linear using original event protocol")
    parser.add_argument("--timer", choices=("profiler", "events"))
    parser.add_argument("--buffers", type=int, help="independent rotating input buffers (events only)")
    parser.add_argument("--warmup", type=int, help="override original protocol (recorded as unmatched)")
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--require-baseline", action="store_true", help="fail a documented target or mismatched conditions")
    parser.add_argument("--allow-contention", action="store_true", help="diagnostic timings only, never baseline acceptance")
    parser.add_argument("--ptl", choices=("current", "VECTOR,F8", "VECTOR,BF16"), default="current",
                        help="explicit GPU0 policy experiment; restore original Disabled state on exit")
    parser.add_argument("--cross-compile", action="store_true")
    parser.add_argument("--strict-resources", action="store_true", help="reject scratch/spills for every backend, including optional tile configurations")
    parser.add_argument("--dump-root", type=Path, default=Path("/tmp/mha_resource_audit"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--pytest-args", nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args()
    if ((args.iterations is not None and args.iterations < 2) or
            (args.rounds is not None and args.rounds < 1) or (args.warmup is not None and args.warmup < 0) or
            (args.buffers is not None and args.buffers < 1)):
        parser.error("iterations >= 2, rounds >= 1 and warmup >= 0 are required")
    if any(value is not None and value < 1 for value in (args.batch, args.heads, args.kv_heads)) or (
            args.heads is not None and args.kv_heads is not None and args.heads % args.kv_heads):
        parser.error("batch must be positive; heads must be a positive multiple of KV heads")
    if any(value < 1 for value in (*(args.q or []), *(args.kv or []))) or any(value < -1 for value in args.window or []):
        parser.error("performance lengths must be positive and windows >= -1")
    if args.require_baseline and args.allow_contention:
        parser.error("baseline acceptance requires an isolated GPU")
    if args.matrix != "documented" and args.case:
        parser.error("--case selects the documented matrix only")
    candidates = (*BACKENDS, SWA) if args.suite == "all" else (SWA,) if args.suite == "swa" else BACKENDS
    backends = [b for b in candidates if "auto" in args.backend or b.name in args.backend]
    if not backends:
        parser.error("no selected backend belongs to this suite; use --suite all if needed")
    if args.gather_linear:
        if backends != [SWA] or args.aiter == "off" or args.timer not in (None, "events") or args.buffers not in (None, 1):
            parser.error("--gather-linear requires the SWA suite/backend, AITER enabled, events and one buffer set")
        if args.require_baseline:
            parser.error("gather+linear event comparison is not the documented profiler throughput gate")
        if not args.list_cases and args.mode not in ("performance", "all"):
            parser.error("--gather-linear is a performance comparison")
        if __package__:
            from .test_mha_pa_swa import benchmark_gather_linear, gather_linear_workloads
        else:
            from test_mha_pa_swa import benchmark_gather_linear, gather_linear_workloads
    if args.list_cases:
        try:
            workloads = gather_linear_workloads(args) if args.gather_linear else benchmark_workloads(args)
            def planned_protocol(workload):
                if args.gather_linear:
                    return {"timer": "events", "sample_warmup": 20 if args.warmup is None else args.warmup,
                            "iterations": 100 if args.iterations is None else args.iterations,
                            "rounds": 5 if args.rounds is None else args.rounds, "buffers": 1,
                            "candidate_order": "rotated every sample"}
                return asdict(measurement_protocol(args, workload))
            plan = [{**workload.to_dict(), "measurement_protocol": planned_protocol(workload),
                     "plan": {backend.name: workload.unsupported(backend) or "ready; native hardware required"
                     for backend in backends}} for workload in workloads if case_selected(args, workload)]
            if not plan:
                raise ValueError("no workload matches the selected filters")
        except ValueError as exc:
            parser.error(str(exc))
        print(json.dumps({"matrix": args.matrix, "workloads": plan, "gpu_queried": False}, indent=2))
        if args.output:
            save(args.output, {"matrix": args.matrix, "workloads": plan, "gpu_queried": False,
                               "executed": False, "status": "planned; GPU rerun pending"})
        return
    if args.ptl != "current" and args.mode != "performance":
        parser.error("PTL experiments are only supported in standalone --mode performance")
    if args.ptl != "current" and any(backend.fp8 != (args.ptl == "VECTOR,F8") for backend in backends):
        parser.error("select only FP8 or only BF16 backends for a matching PTL experiment")
    if args.mode in ("functional", "all"):
        import pytest
        files = []
        if any(backend != SWA for backend in backends):
            files.append(HERE / "test_mha_pa.py")
        if SWA in backends:
            files.append(HERE / "test_mha_pa_swa.py")
        if BF16_942 in backends:
            files.append(HERE / "test_bf16_spills.py")
        if args.suite == "all":
            files.append(HERE / "test_perf_cases.py")
            files.append(HERE / "test_mha_stress.py")
            files.append(HERE / "test_export_performance.py")
            files.append(HERE / "test_validation_manifest.py")
            files.append(HERE / "test_requested_references.py")
        class Filter:
            def pytest_collection_modifyitems(self, items, config):
                if "auto" in args.backend:
                    return
                keep, drop = [], []
                for item in items:
                    selected = getattr(item, "callspec", None)
                    backend = selected.params.get("backend", selected.params.get("selected")) if selected else None
                    if backend is None or not hasattr(backend, "name") or backend.name in args.backend:
                        keep.append(item)
                    else:
                        drop.append(item)
                items[:] = keep
                config.hook.pytest_deselected(items=drop)
        status = pytest.main([*(str(path) for path in files), "-q", *args.pytest_args], plugins=[Filter()])
        if status:
            raise SystemExit(status)
    if args.mode in ("audit", "all"):
        audit(args, backends)
    if args.mode in ("performance", "all"):
        output = result_path(args, "swa_gather_linear" if args.gather_linear else "performance")
        with ptl_experiment(args.ptl, output.with_suffix(".hardware.json")):
            if args.gather_linear:
                benchmark_gather_linear(args)
            else:
                benchmark(args, backends)