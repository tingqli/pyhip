"""Quick full/causal/SWA attention: FP32 accuracy, then cudaPerf metrics.

Examples: python test_mha_pa.py --q 10240 --kv 2583 --dq 192
          python test_mha_pa.py --backend swa --window 128 --sink
          python test_mha_pa.py --preset basic --dq 128 192 --backend 8wave persistent swa
"""

import argparse
from functools import partial
import math
from pathlib import Path
import statistics
import subprocess

import pytest
import torch

if __package__:
    from ._testing import (BACKENDS, BF16_942, BF16_950, BF16_950_PERSISTENT, SWA,
                          dispatch_names, gpu_arch, make_call, make_case, output_buffer, torch_reference)
    from ._perf_cases import Workload
    from ._references import ReferenceUnavailable, aiter_reference_call, probe_reference
else:
    from _testing import (BACKENDS, BF16_942, BF16_950, BF16_950_PERSISTENT, SWA,
                         dispatch_names, gpu_arch, make_call, make_case, output_buffer, torch_reference)
    from _perf_cases import Workload
    from _references import ReferenceUnavailable, aiter_reference_call, probe_reference


def select_backends(names, arch):
    aliases = {"auto": BF16_950 if arch == "gfx950" else BF16_942,
               "8wave": BF16_950 if arch == "gfx950" else BF16_942,
               "persistent": BF16_950_PERSISTENT, "swa": SWA}
    choices = {backend.name: backend for backend in (*BACKENDS, SWA)}
    selected = list(dict.fromkeys(aliases[name] if name in aliases else choices[name] for name in names))
    for backend in selected:
        if backend.arch not in (arch, "both") or arch not in ("gfx942", "gfx950"):
            raise ValueError(f"{backend.name} is unavailable on {arch}")
    if len({backend.dtype for backend in selected}) != 1:
        raise ValueError("benchmark BF16 and FP8 separately to keep the same logical inputs")
    return selected


def workloads(args, backends):
    """One small preset and one custom-shape path for all attention modes."""
    if args.preset == "basic":
        shapes = (("full", (10240,), (2583,), False, -1, False),
                  ("causal", (32768,), (32768,), True, -1, False),
                  ("swa", (16384,), (131072,), True, 128, True))
        plan = [Workload(f"{name}_d{dq}", q * args.batch, kv * args.batch, dq=dq, causal=causal,
                         window=window, sink=sink, heads=args.heads, kv_heads=args.kv_heads,
                         page=args.page, scale_mode=args.scale_mode)
                for dq in args.dq for name, q, kv, causal, window, sink in shapes]
        return [w for w in plan if any(w.unsupported(b) is None for b in backends)]
    swa = SWA in backends or (args.window is not None and args.window >= 0)
    defaults = (16384, 131072, 128) if swa else (10240, 2583, -1)
    q = args.q if args.q is not None else defaults[0]
    kv = args.kv if args.kv is not None else defaults[1]
    window = args.window if args.window is not None else defaults[2]
    q_lens = tuple(args.q_lens) if args.q_lens is not None else (q,) * args.batch
    kv_lens = tuple(args.kv_lens) if args.kv_lens is not None else (kv,) * args.batch
    if len(q_lens) != len(kv_lens):
        raise ValueError("q-lens and kv-lens must describe the same batch")
    return [Workload(f"custom_d{dq}", q_lens, kv_lens, dq=dq, heads=args.heads,
                     kv_heads=args.kv_heads, page=args.page, causal=args.causal or window >= 0,
                     window=window, sink=args.sink, scale_mode=args.scale_mode)
            for dq in args.dq]


def accuracy(actual, reference, label, tolerance=0.02):
    """acc is normalized squared error, like pyhip.calc_diff: zero is best.

    Keep the elementwise check too: a scalar average must not hide a few bad
    rows. Print acc before raising, so a failure is visible without timing it.
    """
    from pyhip import calc_diff
    value = float(calc_diff(reference, actual, diff_thr=-1))
    finite = math.isfinite(value) and bool(torch.isfinite(actual).all())
    print(f"{label} acc={value:.8g}" + ("" if finite else " FAIL(nonfinite)"), flush=True)
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


def measure(calls, reference, *, label, flops, nbytes, run_count=5, warmup=5, repeat=1,
            tolerance=0.02):
    """Validate every candidate before any timing; one compact line per run."""
    rows = []
    for name, call in calls.items():
        actual = call()
        torch.cuda.synchronize()  # Finish first compilation/launch before the next candidate.
        acc = accuracy(actual, reference, f"{label}/{name}", tolerance) if reference is not None else None
        kernels = dispatch_names(call)
        if not kernels and actual.numel():
            raise AssertionError(f"{name}: expected a native attention dispatch")
        rows.append({"backend": name, "acc": acc, "checked": reference is not None,
                     "status": "passed" if reference is not None else "unchecked",
                     "kernels": kernels, "aiter_entry": getattr(call, "aiter_entry", None), "runs": []})
        if name == "aiter":
            print(f"aiter entry={getattr(call, 'aiter_entry', 'unknown')}", flush=True)
            for kernel in kernels:
                print(f"aiter kernel={kernel}", flush=True)
    if run_count:
        for _ in range(warmup):
            for call in calls.values():
                call()
        torch.cuda.synchronize()
    by_name = {row["backend"]: row for row in rows}
    for index in range(run_count):
        order = list(calls) if index % 2 == 0 else list(reversed(calls))
        for name in order:
            with perf_timer(flops * repeat, nbytes * repeat, name) as perf:
                for _ in range(repeat):
                    calls[name]()
            us = perf.dt() * 1e6 / repeat
            if not math.isfinite(us) or us <= 0:
                raise RuntimeError("cudaPerf returned no positive timing; check CUDAPERF filtering")
            sample = {"run": index + 1, "us": us, "tflops": perf.tflops(), "gbps": perf.bw()}
            row = by_name[name]
            row["runs"].append(sample)
            acc = "unchecked" if row["acc"] is None else f"{row['acc']:.8g}"
            print(f"{label}/{name} run={index + 1}/{run_count} acc={acc} "
                  f"time={us:.3f} us tflops={sample['tflops']:.3f} bw={sample['gbps']:.3f} GB/s", flush=True)
    for row in rows:
        us = statistics.median(sample["us"] for sample in row["runs"]) if row["runs"] else None
        row.update(us=us, tflops=flops / us / 1e6 if us else None, gbps=nbytes / us / 1e3 if us else None)
    return rows


@torch.inference_mode()
def run_case(workload, backends, *, check=True, run_count=5, warmup=5, repeat=1,
             aiter="auto", layout="contiguous", nonunit_scales=False, softmax_scale=None,
             poison_tail=False, query_tile=None, block_n=None):
    w = workload
    shape = w.to_dict()
    supported = [b for b in backends if w.unsupported(b) is None]
    unavailable = {b.name: w.unsupported(b) for b in backends if b not in supported}
    for name, reason in unavailable.items():
        print(f"{w.name}/{name} N/A: {reason}", flush=True)
    if not supported:
        return {"workload": shape, "results": [], "unavailable": unavailable}
    if layout != "contiguous" and any(not b.strided for b in supported):
        raise ValueError("the selected backend requires contiguous Q/O")
    if poison_tail and BF16_942 in supported:
        raise ValueError("bf16_942 uses zero-padded tails; --poison-tail is supported by gfx950/SWA/FP8")
    backend = supported[0]
    case = make_case(w.q_lens, w.kv_lens, dtype=backend.dtype, dq=w.dq, dv=w.dv, page=w.page,
                     heads=w.heads, kv_heads=w.kv_heads, mode=w.scale_mode, layout=layout,
                     window_left=w.window, has_sink=w.sink, nonunit_scales=nonunit_scales,
                     poison_tail=poison_tail, quantized=backend.fp8, source_dtype=torch.bfloat16, seed=w.seed)
    reference = torch_reference(case, w.causal, softmax_scale)[0] if check else None
    calls = {}
    for candidate in supported:
        out, _ = output_buffer(case, layout)
        options = {"query_tile": query_tile, "block_n": block_n} if candidate == SWA else {}
        call = make_call(case, candidate, w.causal, out=out, **options)[0]
        calls[candidate.name] = partial(call, softmax_scale=softmax_scale)
    if aiter != "off":
        try:
            call = aiter_reference_call(case, w.causal, softmax_scale=softmax_scale)
            probe_reference(call)
            calls["aiter"] = call
        except ReferenceUnavailable as exc:
            if aiter == "on":
                raise
            unavailable["aiter"] = str(exc)
            print(f"{w.name}/aiter N/A: {exc}", flush=True)
    # Logical Q/K/V/O size, not a hardware traffic counter (especially for SWA).
    nbytes = (sum(w.q_lens) * w.heads * (w.dq * case.q.element_size() + w.dv * 2)
              + sum(w.kv_lens) * w.kv_heads * (w.dq + w.dv) * case.k.element_size())
    label = f"{w.name} B{len(w.q_lens)} H{w.heads}/{w.kv_heads} Q{w.q_lens} KV{w.kv_lens} W{w.window}"
    rows = measure(calls, reference, label=label, flops=w.flops, nbytes=nbytes, run_count=run_count,
                   warmup=warmup, repeat=repeat, tolerance=0.1 if backend.fp8 else 0.02)
    return {"workload": shape, "results": rows, "unavailable": unavailable,
            "input": {"dtype": str(backend.dtype), "layout": layout, "poison_tail": poison_tail,
                      "nonunit_scales": nonunit_scales, "softmax_scale": softmax_scale,
                      "query_tile": query_tile, "block_n": block_n,
                      "q_shape": list(case.q.shape), "q_stride": list(case.q.stride()),
                      "flydsl_kv": "SHUFFLE-5D", "aiter_kv": "prepared linear THD; conversion not timed"},
            "flops": w.flops, "logical_io_bytes": nbytes}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", nargs="+", default=["auto"],
                        choices=("auto", "8wave", "persistent", "swa", *(b.name for b in (*BACKENDS, SWA))))
    parser.add_argument("--preset", choices=("single", "basic"), default="single",
                        help="basic: full10240/2583, causal32K, SWA16K/128K/W128/sink for each D")
    parser.add_argument("--q", type=int)
    parser.add_argument("--kv", type=int)
    parser.add_argument("--q-lens", type=lambda s: [int(v) for v in s.split(",")], help="ragged batch, e.g. 33,129")
    parser.add_argument("--kv-lens", type=lambda s: [int(v) for v in s.split(",")])
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--dq", type=int, nargs="+", choices=(128, 192), default=[192])
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=1)
    parser.add_argument("--page", type=int, default=64)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--window", type=int, help="nonnegative window implies causal; SWA default128")
    parser.add_argument("--sink", action="store_true")
    parser.add_argument("--layout", choices=("contiguous", "padded", "head-major"), default="contiguous")
    parser.add_argument("--scale-mode", choices=("per-token", "per-tensor"), default="per-token")
    parser.add_argument("--nonunit-scales", action="store_true")
    parser.add_argument("--softmax-scale", type=float)
    parser.add_argument("--poison-tail", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--query-tile", type=int, choices=(16, 32))
    parser.add_argument("--block-n", type=int, choices=(16, 32, 64))
    parser.add_argument("--check", type=int, choices=(0, 1), default=1, help="1: FP32 accuracy before timing; 0: explicitly unchecked")
    parser.add_argument("--run-count", type=int, default=5, help="0: accuracy only; otherwise print every timed run")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=1, help="calls per event interval; report per-call time")
    parser.add_argument("--aiter", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--require-idle", action="store_true", help="refuse other GPU processes; default is a quick diagnostic")
    parser.add_argument("--output", type=Path, help="optional JSON plus summary Markdown; use a new filename")
    args = parser.parse_args(argv)
    if args.run_count < 0 or args.warmup < 0 or args.repeat < 1 or (not args.check and not args.run_count):
        parser.error("run-count/warmup must be nonnegative, repeat positive; enable check or timing")
    if args.batch < 1 or args.heads < 1 or args.kv_heads < 1 or args.heads % args.kv_heads:
        parser.error("batch/heads must be positive and heads divisible by kv-heads")
    if any(n < 0 for n in (*(args.q_lens or []), *(args.kv_lens or []), args.q or 0, args.kv or 0)):
        parser.error("sequence lengths must be nonnegative")
    if args.preset == "basic" and any(v is not None for v in (args.q, args.kv, args.q_lens, args.kv_lens, args.window)):
        parser.error("use --preset single for custom lengths/window")
    if args.preset == "basic" and (args.causal or args.sink):
        parser.error("basic defines its own masks/sinks; use --preset single to override them")
    return args


def run(args, backends, selected):
    """Report/measurement loop shared by full, causal and SWA cases."""
    if __package__:
        from ._runner import environment, save
        from ._hardware import require_idle_device
    else:
        from _runner import environment, save
        from _hardware import require_idle_device
    if args.output:
        args.output = args.output.resolve()
        if args.output.suffix != ".json":
            raise ValueError("--output must be a .json file; a companion .md is generated")
        if args.output.exists() or args.output.with_suffix(".md").exists():
            raise FileExistsError("choose a new output file; previous evidence must not be overwritten")
    if SWA not in backends and (args.query_tile is not None or args.block_n is not None):
        raise ValueError("query-tile/block-n apply to the SWA backend only")
    report = {"complete": False, "environment": environment(), "config": {**vars(args), "output": str(args.output) if args.output else None},
              "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[3], text=True).strip(),
              "measurement_scope": "idle-guarded diagnostic" if args.require_idle else "quick diagnostic; no exclusive reservation",
              "timer": "pyhip.cudaPerf GPU event interval; per-call time; includes launch gaps",
              "protocol": {"warmup": args.warmup, "run_count": args.run_count, "repeat": args.repeat,
                           "input_buffers": 1, "summary": "median of all per-call event times",
                           "gpu_spin_before_event": "cudaPerf's built-in torch.cuda._sleep(1000000)"},
              "bandwidth": "logical Q/K/V/O bytes divided by time, not measured HBM traffic",
              "acc_definition": "sum((ref-out)^2) / sum(ref^2+out^2), smaller is better; elementwise tolerance also required",
              "records": []}
    print(f"arch={report['environment']['arch']} check={args.check} runs={args.run_count} "
          f"repeat={args.repeat} scope={report['measurement_scope']}", flush=True)
    try:
        for w in selected:
            if args.require_idle:
                require_idle_device()
            row = run_case(w, backends, check=bool(args.check), run_count=args.run_count, warmup=args.warmup,
                           repeat=args.repeat, aiter=args.aiter, layout=args.layout, nonunit_scales=args.nonunit_scales,
                           softmax_scale=args.softmax_scale, poison_tail=args.poison_tail,
                           query_tile=args.query_tile, block_n=args.block_n)
            report["records"].append(row)
            if args.require_idle:
                require_idle_device()
        if not any(row["results"] for row in report["records"]):
            raise RuntimeError("no supported candidate ran")
        report["complete"] = True
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if args.output:
            save(args.output, report)
    import pandas as pd
    table = pd.DataFrame([{"case": row["workload"]["name"], **{k: r[k] for k in
                          ("backend", "status", "acc", "us", "tflops", "gbps")}}
                         for row in report["records"] for r in row["results"]]).to_markdown(index=False, floatfmt=".6g")
    print("\nMedian per-call results (all runs retained):\n" + table, flush=True)
    if args.output:
        dispatches = [f"- **{row['workload']['name']}**: `{r['aiter_entry']}`\n  - `" + "`\n  - `".join(r["kernels"]) + "`"
                      for row in report["records"] for r in row["results"] if r["backend"] == "aiter"]
        args.output.with_suffix(".md").write_text(
            "# Quick MHA accuracy/performance\n\n"
            "cudaPerf GPU-event intervals; acc=0 is best; bandwidth is logical Q/K/V/O GB/s, not a hardware counter.\n\n"
            + table + "\n\n## AITER dispatch\n\n" + "\n".join(dispatches or ["AITER disabled or unavailable."]) + "\n")
    return report


def main(argv=None):
    args = parse_args(argv)
    backends = select_backends(args.backend, gpu_arch())
    return run(args, backends, workloads(args, backends))


@pytest.mark.parametrize("variant", ("8wave", "persistent"))
@pytest.mark.parametrize("dq", (128, 192))
def test_page_boundaries(dq, variant):
    """Only the short ragged/empty/poisoned cases not covered by perf shapes."""
    arch = gpu_arch()
    if arch not in ("gfx942", "gfx950") or variant == "persistent" and arch != "gfx950":
        pytest.skip("requires a supported native GPU")
    backends = select_backends([variant], arch)
    kv = (63, 0, 193, 257) if arch == "gfx950" else (63, 65, 193, 257)
    w = Workload("ragged_edges", (0, 7, 33, 129), kv, dq=dq, heads=6, kv_heads=2,
                 causal=True)
    run_case(w, backends, run_count=0, aiter="off", layout="padded" if backends[0].strided else "contiguous", nonunit_scales=True,
             poison_tail=backends != [BF16_942], softmax_scale=0.0625)


@pytest.mark.parametrize("dq", (128, 192))
def test_swa_page_boundaries(dq):
    """One fused ragged/empty/poisoned/scale case per head dimension."""
    if gpu_arch() not in ("gfx942", "gfx950"):
        pytest.skip("requires a supported native GPU")
    w = Workload("swa_ragged_edges", (0, 7, 33, 129), (63, 0, 193, 257),
                 dq=dq, heads=6, kv_heads=2, causal=True, window=128, sink=True)
    run_case(w, [SWA], run_count=0, aiter="off", layout="padded",
             nonunit_scales=True, poison_tail=True, softmax_scale=0.0625)


if __name__ == "__main__":
    main()