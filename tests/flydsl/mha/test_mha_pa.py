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
    from ._references import ReferenceUnavailable, aiter_reference_call, aiter_gather_call, probe_reference, requested_reference_call, requested_identity
else:
    from _testing import (BACKENDS, BF16_942, BF16_950, BF16_950_PERSISTENT, SWA,
                         dispatch_names, gpu_arch, make_call, make_case, output_buffer, torch_reference)
    from _perf_cases import Workload
    from _references import ReferenceUnavailable, aiter_reference_call, aiter_gather_call, probe_reference, requested_reference_call, requested_identity


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


def measure(calls, references, *, label, flops, nbytes, run_count=5, warmup=10, repeat=1,
            tolerance=0.02):
    """Check every independent buffer, then run_count full rotating-buffer rounds.

    One round measures every buffer. repeat>1 advances the buffer on every
    invocation inside one event pair; all candidates see identical indices.
    Median over all event intervals, not the fastest sample or a sum of timers.
    """
    buffers = len(references)
    if not buffers or not calls or any(len(pool) != buffers for pool in calls.values()):
        raise ValueError("every candidate requires the same nonempty buffer pool")
    rows = []
    for name, pool in calls.items():
        accs, outputs = [], []
        for index, (call, reference) in enumerate(zip(pool, references)):
            actual = call()
            torch.cuda.synchronize()
            accs.append(accuracy(actual, reference, f"{label}/{name}/buffer{index}", tolerance) if reference is not None else None)
            outputs.append(actual.data_ptr())
            first = actual.clone()
            for _ in range(2):
                torch.testing.assert_close(call(), first, rtol=0, atol=0)
        if len(set(outputs)) != buffers:
            raise AssertionError(f"{name}: output buffers alias")
        kernels = dispatch_names(pool[0])
        if not kernels and actual.numel():
            raise AssertionError(f"{name}: expected a native attention dispatch")
        if name == "aiter_gather" and (len(kernels) != 2 or not any("gather" in k for k in kernels)):
            raise AssertionError(f"expected one gather and one CK dispatch: {kernels}")
        checked = all(ref is not None for ref in references)
        rows.append({"backend": name, "acc": max(accs) if checked else None, "acc_per_buffer": accs,
                     "checked": checked, "status": "passed" if checked else "unchecked",
                     "repeated_bit_exact": True, "output_ptrs": outputs,
                     "workspace_ptrs": [getattr(call, "workspace_ptrs", []) for call in pool],
                     "kernels": kernels, "aiter_entry": getattr(pool[0], "aiter_entry", None),
                     "reference_source": getattr(pool[0], "reference_source", None),
                     "logical_io_bytes": nbytes[name], "runs": []})
        if name.startswith("aiter"):
            print(f"{name} entry={getattr(pool[0], 'aiter_entry', 'unknown')} kernels={kernels}", flush=True)
    if run_count:
        for index in range(warmup):
            for pool in calls.values():
                pool[index % buffers]()
        torch.cuda.synchronize()
    by_name = {row["backend"]: row for row in rows}
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
                acc = "unchecked" if row["acc"] is None else f"{row['acc']:.8g}"
                print(f"{label}/{name} round={trial + 1}/{run_count} buffers={indices} acc={acc} "
                      f"time={us:.3f} us tflops={sample['tflops']:.3f} bw={sample['gbps']:.3f} GB/s", flush=True)
    for row in rows:
        us = statistics.median(sample["us"] for sample in row["runs"]) if row["runs"] else None
        row.update(us=us, tflops=flops / us / 1e6 if us else None,
                   gbps=nbytes[row["backend"]] / us / 1e3 if us else None)
    return rows


@torch.inference_mode()
def run_case(workload, backends, *, check=True, run_count=5, warmup=10, repeat=1, buffers=10,
             aiter="auto", layout="contiguous", nonunit_scales=False, softmax_scale=None,
             poison_tail=False, query_tile=None, block_n=None, requested_reference="off"):
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
    if buffers < 1:
        raise ValueError("buffers must be positive")
    backend = supported[0]
    cases, references, calls = [], [], {}
    factories = {}
    if aiter != "off":
        factories["aiter"] = (aiter_reference_call, aiter)
        if w.window >= 0:
            factories["aiter_gather"] = (aiter_gather_call, aiter)
    if requested_reference != "off":
        factories["requested_reference"] = (requested_reference_call, requested_reference)
    for index in range(buffers):
        case = make_case(w.q_lens, w.kv_lens, dtype=backend.dtype, dq=w.dq, dv=w.dv, page=w.page,
                         heads=w.heads, kv_heads=w.kv_heads, mode=w.scale_mode, layout=layout,
                         window_left=w.window, has_sink=w.sink, nonunit_scales=nonunit_scales,
                         poison_tail=poison_tail, quantized=backend.fp8, source_dtype=torch.bfloat16, seed=w.seed + index)
        cases.append(case)
        references.append(torch_reference(case, w.causal, softmax_scale)[0] if check else None)
        for candidate in supported:
            out, _ = output_buffer(case, layout)
            options = {"query_tile": query_tile, "block_n": block_n} if candidate == SWA else {}
            call = make_call(case, candidate, w.causal, out=out, **options)[0]
            calls.setdefault(candidate.name, []).append(partial(call, softmax_scale=softmax_scale))
        for name, (factory, policy) in factories.items():
            if name in unavailable:
                continue
            try:
                call = factory(case, w.causal, softmax_scale=softmax_scale)
                probe_reference(call)
                calls.setdefault(name, []).append(call)
            except ReferenceUnavailable as exc:
                if policy == "on":
                    raise
                unavailable[name] = str(exc)
                calls.pop(name, None)
                print(f"{w.name}/{name} N/A: {exc}", flush=True)
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
                   warmup=warmup, repeat=repeat, tolerance=0.1 if backend.fp8 else 0.02)
    if "requested_reference" in calls:
        if requested_identity("fp8" if backend.fp8 else "bf16") != calls["requested_reference"][0].reference_source:
            raise ValueError("requested reference changed during measurement")
    return {"workload": shape, "results": rows, "unavailable": unavailable,
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
    parser.add_argument("--buffers", type=int, default=10, help="independent input/output/workspace sets; every round measures all buffers")
    parser.add_argument("--run-count", type=int, default=5, help="full buffer rounds; 0: accuracy only; default 5x10=50 samples per candidate")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=1, help="rotating-buffer calls per event interval; report per-call time")
    parser.add_argument("--aiter", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--requested-reference", choices=("auto", "on", "off"), default="off",
                        help="add specified dense BF16 / BN32 paged FP8 reference; never resize unsupported inputs")
    parser.add_argument("--require-idle", action="store_true", help="refuse other GPU processes; default is a quick diagnostic")
    parser.add_argument("--wait-idle", action="store_true", help="physical GPU0: wait indefinitely for no processes and 3 quiet samples before GPU initialization")
    parser.add_argument("--output", type=Path, help="optional JSON plus summary Markdown; use a new filename")
    args = parser.parse_args(argv)
    if args.wait_idle:
        args.require_idle = True
    if args.run_count < 0 or args.warmup < 0 or args.repeat < 1 or args.buffers < 1 or (not args.check and not args.run_count):
        parser.error("run-count/warmup must be nonnegative, repeat/buffers positive; enable check or timing")
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
                           "input_buffers": args.buffers, "samples_per_candidate": args.run_count * args.buffers,
                           "summary": "median of all per-call event times across all buffers and rounds",
                           "buffer_rotation": args.buffers > 1, "repeat_reuses_same_buffer": args.buffers == 1,
                           "gpu_spin_before_event": "cudaPerf's built-in torch.cuda._sleep(1000000)"},
              "bandwidth": "logical Q/K/V/O bytes; aiter_gather adds full KV read+write; not measured HBM traffic",
              "acc_definition": "sum((ref-out)^2) / sum(ref^2+out^2), smaller is better; elementwise tolerance also required",
              "records": []}
    print(f"arch={report['environment']['arch']} check={args.check} runs={args.run_count} "
          f"buffers={args.buffers} repeat={args.repeat} scope={report['measurement_scope']}", flush=True)
    try:
        for w in selected:
            if args.require_idle:
                require_idle_device()
            row = run_case(w, backends, check=bool(args.check), run_count=args.run_count, warmup=args.warmup,
                           repeat=args.repeat, aiter=args.aiter, layout=args.layout, nonunit_scales=args.nonunit_scales,
                           softmax_scale=args.softmax_scale, poison_tail=args.poison_tail,
                           query_tile=args.query_tile, block_n=args.block_n, requested_reference=args.requested_reference,
                           buffers=args.buffers)
            report["records"].append(row)
            if args.output:
                save(args.output, report)
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
                      for row in report["records"] for r in row["results"] if r["backend"].startswith("aiter")]
        args.output.with_suffix(".md").write_text(
            "# Quick MHA accuracy/performance\n\n"
            f"cudaPerf events; {args.buffers} independent buffers, {args.run_count} full rounds; acc=max across buffers. "
            "Bandwidth is logical Q/K/V/O GB/s; gather+CK adds full KV read/write, not a hardware counter.\n\n"
            + table + "\n\n## AITER dispatch\n\n" + "\n".join(dispatches or ["AITER disabled or unavailable."]) + "\n")
    return report


def main(argv=None):
    args = parse_args(argv)
    if args.wait_idle:
        import os
        if any(os.environ.get(key) != "0" for key in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")):
            raise ValueError("--wait-idle requires explicit physical GPU0 visibility")
        if __package__:
            from ._hardware import wait_until_idle
        else:
            from _hardware import wait_until_idle
        log = args.output.with_suffix(".idle.jsonl") if args.output else None
        if log:
            if args.output.exists() or args.output.with_suffix(".md").exists() or log.exists():
                raise FileExistsError("choose new report and idle-log paths before waiting")
            log.parent.mkdir(parents=True, exist_ok=True)
        wait_until_idle(record_path=log)
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


@pytest.mark.parametrize("dq", (128, 192))
def test_full_gather_live_cache(dq):
    if gpu_arch() not in ("gfx942", "gfx950"):
        pytest.skip("requires native GPU")
    from importlib import import_module
    gather_kv_call = import_module((__package__ + "." if __package__ else "") + "_gather").gather_kv_call
    case = make_case((0, 7, 33), (0, 65, 193), dq=dq, heads=6, kv_heads=2,
                     window_left=128, poison_tail=True, reverse_pages=True)
    gather, workspace = gather_kv_call(case)
    ptrs = [t.data_ptr() for t in workspace]
    for index in range(2):
        if index:
            case.k_pages.mul_(0.5)
            case.v_pages.neg_()
            case.pack(copy=True)
        actual = gather()
        assert [t.data_ptr() for t in actual] == ptrs
        for a, expected in zip(actual, case.logical_kv()):
            torch.testing.assert_close(a.float(), torch.cat(expected), rtol=0, atol=0)
    assert len(dispatch_names(gather)) == 1


@pytest.mark.parametrize("dq", (128, 192))
@pytest.mark.parametrize("causal", (False, True))
@pytest.mark.parametrize("mode", ("per-token", "per-tensor"))
def test_fp8_lds_page_boundaries(dq, causal, mode):
    if gpu_arch() != "gfx942":
        pytest.skip("requires native gfx942")
    w = Workload("fp8_lds_edges", (0, 7, 33, 129), (63, 65, 193, 257), dq=dq,
                 heads=6, kv_heads=2, causal=causal, scale_mode=mode)
    run_case(w, select_backends(["fp8_942"], "gfx942"), run_count=0, aiter="off", poison_tail=True)


if __name__ == "__main__":
    main()