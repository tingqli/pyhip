"""Timing: aiter's gfx942 1-stage asm fused MoE (fmoe_fp8_blockscale_g1u1) vs our FlyDSL GateUp + Down.

One EP8 rank of MiMo-V2.6-Flash (hidden 4096, moe_inter 2048, 256 experts, top-8: 32 local experts) on the inputs of
test_fmoe_a8w8_blockscale_asm_gfx942.make_inputs: FP8 activations with per-(token, 128) scales and 128x128
block-scaled FP8 weights, which is what aiter.fused_moe hands the asm kernel on gfx942. Ours: aiter moe_sorting(256)
-> GateUp 2x4_pf1 with fp8_out (FP8 per (row, 128) in sorted-row order) -> Down 4x2_pf1 on the sorted A2, routes added with
BF16 atomics ("atomic"), or stored for tokens with one local route ("B", single_store). Every variant writes its own
output, which the test's check() validates after timing; the summary compares both with the fastest asm kernel.

Method (docs/moe_a8w8_blockscale_gfx942.md):
- 5 clones of a, scales, w1 and w2 rotate over every call that reads weights, so weights are never cache resident.
- --timing queued (default): each variant runs in blocks, --warmup-ms of untimed calls then the timed calls, all
  queued back to back behind a head-start sleep with an event pair per call: the GPU never idles or waits for the
  host. --timing cudaperf times each call with pyhip.cudaPerf (sleep, sync): the GPU idles between calls and short
  kernels then run up to ~30% slower, which hits our three to five short kernels much harder than the one asm kernel.
- "Excluding moe_sorting" is the full path minus its timed sort stage. The variants without a sort have their output
  filled untimed right before the timed region, and that 64-256 MB fill slows our GateUp + Down by up to ~80 us but
  not the asm kernel.
- The block order reverses every round. An amdsmi gate (activity <= 5%, VRAM <= 20%) runs at entry, between rounds
  and at the end; any failure marks the run invalid.

usage: HIP_VISIBLE_DEVICES=0 python bench_moe_a8w8_blockscale_gfx942_vs_asm.py --tokens 32768 [--asm vs_64x256]
           [--json result.json]
"""

import argparse
import functools
import json
import math
import os
import statistics
import subprocess
import sys
import time

I, H, TOPK, E, EP = 2048, 4096, 8, 32, 8


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tokens", type=int, default=32768)
    parser.add_argument("--routing", default="ep_uniform", choices=["ep_uniform", "random", "balanced"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--asm", nargs="+", help="asm kernels to time (default: all of the CSV); auto is always kept")
    parser.add_argument("--control", default="vs_64x256", help="asm kernel of the [prep NaN+sort] control")
    parser.add_argument("--clones", type=int, default=5)
    parser.add_argument("--samples", type=int, default=32, help="per variant, a multiple of --rounds")
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--warmup-ms", type=float, default=50.0)
    parser.add_argument("--timing", default="queued", choices=["queued", "cudaperf"])
    parser.add_argument("--gemm1-layout", default="2x4_pf1")
    parser.add_argument("--gemm2-layout", default="4x2_pf1")
    parser.add_argument("--no-gate", action="store_true", help="skip the amdsmi idle-GPU gate (run marked invalid)")
    parser.add_argument("--dry-run", action="store_true", help="run and check every variant twice, no timing")
    parser.add_argument("--host-time", action="store_true", help="print each variant's host enqueue time and exit")
    parser.add_argument("--json", help="write samples, stages, checks, gates and the environment to this file")
    args = parser.parse_args(argv)
    assert args.samples % args.rounds == 0, "--samples must be a multiple of --rounds"
    return args


def make_gate(enabled, gpu_index):
    """gate(stage, settle) -> passed, and the list of gate records (empty when disabled)."""
    records = []
    if not enabled:
        return (lambda stage, settle=0.0: True), records
    try:
        import amdsmi
    except ImportError:
        sys.exit("amdsmi is not importable (pip install /opt/rocm/share/amd_smi), or pass --no-gate")
    amdsmi.amdsmi_init()
    handle = amdsmi.amdsmi_get_processor_handles()[gpu_index]

    def gate(stage, settle=0.0):
        # Called with this process synchronized: its own gfx_activity decays to <= 1% within 0.5 s.
        time.sleep(settle)
        activity = []
        for _ in range(5):
            activity.append(amdsmi.amdsmi_get_gpu_activity(handle)["gfx_activity"])
            time.sleep(0.1)
        memory = amdsmi.amdsmi_get_gpu_vram_usage(handle)
        record = dict(
            stage=stage,
            bdf=amdsmi.amdsmi_get_gpu_device_bdf(handle),
            activity=activity,
            vram_used_mb=memory["vram_used"],
            vram_total_mb=memory["vram_total"],
            pids=len(amdsmi.amdsmi_get_gpu_process_list(handle)),
        )
        record["pass"] = max(activity) <= 5 and record["vram_used_mb"] <= 0.2 * record["vram_total_mb"]
        records.append(record)
        if not record["pass"] or stage in ("entry", "before sampling", "end"):
            print("GATE", json.dumps(record), flush=True)
        return record["pass"]

    return gate, records


def environment(torch):
    from importlib.metadata import PackageNotFoundError, version

    import aiter

    def git(path):
        try:
            result = subprocess.run(["git", "-C", path, "describe", "--always", "--dirty"], capture_output=True, text=True)
        except OSError:
            return None
        return result.stdout.strip() or None

    props = torch.cuda.get_device_properties(0)
    env = dict(gpu=props.name, arch=props.gcnArchName, cus=props.multi_processor_count, torch=torch.__version__,
               hip=torch.version.hip, pyhip=git(os.path.dirname(os.path.abspath(__file__))),
               aiter=git(os.path.dirname(os.path.dirname(os.path.abspath(aiter.__file__)))))
    try:
        env["flydsl"] = version("flydsl")
    except PackageNotFoundError:
        env["flydsl"] = None
    return env


def main(argv=None):
    args = parse_args(argv)
    gpu = int(os.environ.get("HIP_VISIBLE_DEVICES", "0").split(",")[0])
    gate, gates = make_gate(not args.no_gate, gpu)
    if not gate("entry") and not args.dry_run:
        sys.exit("entry gate failed (GPU busy): no timing")

    import torch

    import pyhip
    import test_fmoe_a8w8_blockscale_asm_gfx942 as asm_t
    from aiter import dtypes
    from aiter.fused_moe import moe_sorting
    from pyhip.contrib.flydsl.moe_down_a8w8_blockscale_gfx942_lds import (
        moe_down_a8w8_blockscale_gfx942_lds, moe_route_counts, moe_zero_rows,
    )
    from pyhip.contrib.flydsl.moe_gateup_a8w8_blockscale_gfx942_lds import moe_gateup_a8w8_blockscale_gfx942_lds

    if gates:
        assert torch.cuda.get_device_properties(0).pci_bus_id == int(gates[0]["bdf"].split(":")[1], 16), (
            "HIP device 0 is not the GPU the gate watches: set HIP_VISIBLE_DEVICES to one physical GPU index"
        )
    unknown = set(args.asm or []) - set(asm_t.KERNELS)
    assert not unknown, f"unknown asm kernels {sorted(unknown)}; choose from {list(asm_t.KERNELS)}"
    env = environment(torch)
    print("ENV", json.dumps(env), flush=True)

    tokens = args.tokens
    inputs = asm_t.make_inputs(tokens, I, H, TOPK, E, EP, args.routing, seed=args.seed)
    routes = inputs["routes"][0].numel()
    flops = 6 * routes * I * H
    expected = asm_t.reference(inputs)
    clones = [inputs] + [
        dict(inputs, **{name: inputs[name].clone() for name in ("a", "a_scale_t", "w1", "w1_scale", "w2", "w2_scale")})
        for _ in range(args.clones - 1)
    ]

    def new_out():
        return torch.empty((tokens, H), device="cuda", dtype=torch.bfloat16)

    no_zero = torch.empty((0, 0), device="cuda", dtype=torch.bfloat16)

    def sort(block_m, out, zero=True):
        """aiter moe_sorting under EP into out, which it zeroes; zero=False hands it an empty moe_buf instead."""
        sorted_ids, sorted_weights, expert_ids, num_valid_ids, _ = moe_sorting(
            inputs["topk_ids"], inputs["topk_weights"], E * EP, H, dtypes.bf16, block_m, inputs["expert_mask"],
            output=out if zero else no_zero,
        )
        if block_m == 256:
            # Older aiter moe_sorting does not round sorted_ids up to whole blocks; the valid blocks fit below it.
            expert_ids = expert_ids[: sorted_ids.numel() // block_m]
        return dict(sorted_ids=sorted_ids, sorted_weights=sorted_weights, sorted_expert_ids=expert_ids,
                    num_valid_ids=num_valid_ids, out=out, block_m=block_m)

    s256 = sort(256, new_out())
    capacity = s256["sorted_expert_ids"].numel() * 256
    a2 = torch.empty((capacity, I), device="cuda", dtype=dtypes.fp8)
    a2_scale_t = torch.empty((I // 128, capacity), device="cuda")
    counts = torch.empty(tokens, device="cuda", dtype=torch.int32)
    counts_timed = torch.empty_like(counts)
    marks = None
    stages = {}

    def mark(label):
        """Ends stage `label` of a timed queued call; a no-op otherwise."""
        if marks is not None:
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            marks.append((label, event))

    def count_routes(s, buf):
        return moe_route_counts(s["sorted_ids"], s["num_valid_ids"], s["sorted_expert_ids"], buf, TOPK)

    def gemm1(c, s):
        moe_gateup_a8w8_blockscale_gfx942_lds(
            c["a"], c["w1"], c["a_scale_t"], c["w1_scale"], s["sorted_ids"], s["sorted_expert_ids"], s["num_valid_ids"],
            a2, layout=args.gemm1_layout, out_scale_t=a2_scale_t, topk=TOPK,
        )

    def gemm2(c, s, out, route_counts=None):
        moe_down_a8w8_blockscale_gfx942_lds(
            a2, c["w2"], a2_scale_t, c["w2_scale"], s["sorted_ids"], s["sorted_weights"], s["sorted_expert_ids"],
            s["num_valid_ids"], out, layout=args.gemm2_layout, reduce="atomic", topk=TOPK, zero_out=False,
            single_store=route_counts is not None, route_counts=route_counts,
        )

    def ours_atomic(c, s, out):
        gemm1(c, s)
        mark("GEMM1")
        gemm2(c, s, out)

    def ours_b(c, s, out, zero_rows):
        count_routes(s, counts_timed)
        if zero_rows:
            moe_zero_rows(out, counts_timed)
        mark("counts")
        gemm1(c, s)
        mark("GEMM1")
        gemm2(c, s, out, counts_timed)

    def after_sort(block_m, out, zero, fn):
        s = sort(block_m, out, zero)
        mark("sort")
        fn(s)

    def poison(out):
        return functools.partial(out.fill_, math.nan)

    def prep_like_full_path(out, block_m):
        # The state the full path leaves before the kernels (NaN poison, then the sort's zeroing), made untimed.
        return lambda: (out.fill_(math.nan), sort(block_m, out))

    # name -> (output checked after timing or None, untimed preparation or None, timed fn(clone), reads clone data).
    variants = {}

    def add(name, fn, prep=None, out=None, reads=True):
        variants[name] = (out, prep, fn, reads)

    asm_kernels = [k for k in asm_t.KERNELS if not args.asm or k in args.asm + ["auto"]]
    block_m_of = {k: asm_t.KERNELS[k]["block_m"] for k in asm_t.KERNELS}
    presorted = {bm: sort(bm, new_out()) for bm in {block_m_of[k] for k in asm_kernels + [args.control]}}
    for k in asm_kernels:
        out = new_out()
        add(f"asm {k}", lambda c, k=k, s=dict(presorted[block_m_of[k]], out=out): asm_t.run_kernel(c, s, k), out.zero_, out)
    out = new_out()
    add(
        f"asm {args.control} [prep NaN+sort]",
        lambda c, s=dict(presorted[block_m_of[args.control]], out=out): asm_t.run_kernel(c, s, args.control),
        prep_like_full_path(out, block_m_of[args.control]),
        out,
    )
    add("GEMM1", lambda c: gemm1(c, s256))
    add("counts", lambda c: count_routes(s256, counts_timed), reads=False)
    add("zero_rows", lambda c, out=new_out(): moe_zero_rows(out, counts), reads=False)
    out = new_out()
    add("GEMM2 atomic", lambda c, out=out: gemm2(c, s256, out), out.zero_, out)
    out = new_out()
    add("GEMM2 B", lambda c, out=out: gemm2(c, s256, out, counts), out.zero_, out)
    out = new_out()
    add("GEMM1+GEMM2 atomic", lambda c, out=out: ours_atomic(c, s256, out), out.zero_, out)
    out = new_out()
    add("GEMM1+GEMM2 atomic [prep NaN+sort]", lambda c, out=out: ours_atomic(c, s256, out), prep_like_full_path(out, 256), out)
    out = new_out()
    add("counts+zero_rows+GEMM1+GEMM2 B", lambda c, out=out: ours_b(c, s256, out, True), poison(out), out)
    # Full paths: the sort is timed; the NaN poison makes check() see every row written or zeroed.
    for k in asm_kernels:
        out = new_out()
        add(
            f"sort+asm {k}",
            lambda c, k=k, out=out: after_sort(block_m_of[k], out, True, lambda s: asm_t.run_kernel(c, s, k)),
            poison(out),
            out,
        )
    out = new_out()
    add("sort+GEMM1+GEMM2 atomic", lambda c, out=out: after_sort(256, out, True, lambda s: ours_atomic(c, s, out)), poison(out), out)
    out = new_out()
    add(
        "sort+counts+GEMM1+GEMM2 B",
        lambda c, out=out: after_sort(256, out, True, lambda s: ours_b(c, s, out, False)),
        poison(out),
        out,
    )
    out = new_out()
    add(
        "sortNZ+counts+zero_rows+GEMM1+GEMM2 B",
        lambda c, out=out: after_sort(256, out, False, lambda s: ours_b(c, s, out, True)),
        poison(out),
        out,
    )
    for bm in sorted({block_m_of[k] for k in asm_kernels} | {256}):
        add(f"sort{bm}", lambda c, bm=bm, out=new_out(): sort(bm, out), reads=False)
    add("sort256NZ", lambda c, out=new_out(): sort(256, out, zero=False), reads=False)

    order = list(variants)
    next_clone = 0
    clone_of = {name: [] for name in order}

    def pick(name):
        nonlocal next_clone
        _, prep, fn, reads = variants[name]
        index = next_clone % len(clones) if reads else -1
        next_clone += reads
        return prep, fn, index, clones[max(index, 0)]

    def run(name, timed):
        prep, fn, index, clone = pick(name)
        if prep:
            prep()
        if not timed:
            return fn(clone)
        clone_of[name].append(index)
        with pyhip.cudaPerf(flops, name=name, verbose=0) as perf:
            fn(clone)
        return perf.latencies[0] * 1e6

    def queued_block(name, warmups, n):
        """warmups untimed and n timed calls (an event pair each) back to back; the 60M-cycle sleep lets the host
        enqueue the whole block before the GPU reaches it."""
        nonlocal marks
        torch.cuda._sleep(60_000_000)
        for _ in range(warmups):
            run(name, False)
        calls = []
        for _ in range(n):
            prep, fn, index, clone = pick(name)
            clone_of[name].append(index)
            if prep:
                prep()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            marks = []
            start.record()
            fn(clone)
            end.record()
            calls.append((start, marks, end))
            marks = None
        torch.cuda.synchronize()
        for start, call_marks, end in calls:
            if call_marks:
                labels = [label for label, _ in call_marks] + ["rest"]
                events = [start] + [event for _, event in call_marks] + [end]
                for label, a, b in zip(labels, events, events[1:]):
                    stages.setdefault(name, {}).setdefault(label, []).append(a.elapsed_time(b) * 1e3)
        return [start.elapsed_time(end) * 1e3 for start, _, end in calls]

    count_routes(s256, counts)
    for _ in range(2):
        for name in order:
            run(name, False)
    torch.cuda.synchronize()

    if args.host_time:
        # Host time to enqueue one call (preparation included) while a long sleep keeps the GPU queue busy.
        for name in order:
            spent = []
            for _ in range(7):
                torch.cuda._sleep(200_000_000)
                t0 = time.perf_counter()
                run(name, False)
                spent.append((time.perf_counter() - t0) * 1e6)
                torch.cuda.synchronize()
            print(f"  host enqueue {name:40s} {statistics.median(spent):8.1f} us", flush=True)
        return

    samples = {name: [] for name in order}
    if not args.dry_run:
        estimate_ms = {name: run(name, True) / 1e3 for name in order}
        for name in order:
            clone_of[name].clear()
        gate("before sampling", settle=0.6)
        for r in range(args.rounds):
            for name in order if r % 2 == 0 else order[::-1]:
                warmups = min(200, max(8, math.ceil(args.warmup_ms / estimate_ms[name])))
                if args.timing == "queued":
                    samples[name] += queued_block(name, warmups, args.samples // args.rounds)
                    continue
                for _ in range(warmups):
                    run(name, False)
                samples[name] += [run(name, True) for _ in range(args.samples // args.rounds)]
            if r < args.rounds - 1:
                gate(f"round {r}", settle=0.6)
        gate("end", settle=0.6)

    checks, diffs = {}, {}
    routed = torch.bincount(inputs["routes"][0], minlength=tokens) > 0
    asm_ref = variants["asm auto"][0][routed].float()
    for name, (out, _, _, _) in variants.items():
        if out is None:
            continue
        try:
            asm_t.check(inputs, out, expected)
            checks[name] = "pass"
        except AssertionError as e:
            checks[name] = f"FAIL {e}"
        o = out[routed].float()
        diffs[name] = (pyhip.calc_diff(o, expected[routed]), pyhip.calc_diff(o, asm_ref))

    # Share of the 6 * routes * I * H FLOPs done by the single-stage variants; sorts and counts have none.
    work = {"GEMM1": 4 / 6, "GEMM2 atomic": 2 / 6, "GEMM2 B": 2 / 6}
    no_flops = {"counts", "zero_rows"} | {name for name in order if name.startswith("sort") and "+" not in name}
    valid = bool(gates) and all(g["pass"] for g in gates)
    print(f"T={tokens} {args.routing}: {routes} local routes, {flops / 1e12:.3f} TFLOP (6*routes*I*H), "
          f"{len(samples[order[0]])} samples/variant, timing {args.timing}, clones {args.clones}, "
          f"gates {'PASS' if valid else 'FAIL' if gates else 'SKIPPED'}")
    print(f"  {'variant':40s} {'median us':>9s} {'p10-p90 us':>15s} {'TFLOPS':>7s}  {'calc_diff vs ref':>16s}  "
          f"{'vs asm auto':>11s}  check")
    for name in order:
        x = sorted(samples[name])
        med = statistics.median(x) if x else math.nan
        spread = f"{x[len(x) // 10]:7.1f}-{x[len(x) * 9 // 10]:<7.1f}" if x else " " * 15
        tf = " " * 7 if name in no_flops else f"{flops * work.get(name, 1) / med / 1e6:7.1f}"
        d = "  ".join(f"{v:{w}.2e}" for v, w in zip(diffs[name], (16, 11))) if name in diffs else " " * 29
        print(f"  {name:40s} {med:9.1f} {spread} {tf}  {d}  {checks.get(name, '')}", flush=True)

    if not args.dry_run:
        med = {name: statistics.median(samples[name]) for name in order}
        for name, parts in stages.items():
            print(f"  stages {name}: " + " | ".join(f"{label} {statistics.median(v):.1f}" for label, v in parts.items()))
        # The tuned asm is the kernel a tuner keeps: the fastest one without the sort.
        tuned = min((name for name in order if name.startswith("asm ") and "[" not in name), key=med.get)
        columns = [
            (f"tuned {tuned}", "sort+" + tuned, tuned),
            ("asm auto (untuned)", "sort+asm auto", "asm auto"),
            ("ours atomic", "sort+GEMM1+GEMM2 atomic", "GEMM1+GEMM2 atomic"),
            ("ours B (single_store)", "sortNZ+counts+zero_rows+GEMM1+GEMM2 B", "counts+zero_rows+GEMM1+GEMM2 B"),
        ]

        def without_sort(full, fallback):
            # Full path minus its timed sort stage; cudaperf mode has no stages and falls back to the no-sort variant.
            if full in stages:
                return statistics.median(t - s for t, s in zip(samples[full], stages[full]["sort"]))
            return med[fallback]

        print("summary, median us (speedup over the tuned asm kernel):")
        print(f"  {'':24s}" + "".join(f"{label:>27s}" for label, _, _ in columns))
        for row, value in (("with moe_sorting", lambda full, fallback: med[full]), ("excluding moe_sorting", without_sort)):
            times = [value(full, fallback) for _, full, fallback in columns]
            print(f"  {row:24s}" + "".join(f"{t:19.1f} ({times[0] / t:.2f}x)" for t in times))
        print("  calc_diff vs reference: " + ", ".join(f"{label} {diffs[full][0]:.2e}" for label, full, _ in columns))

    if args.json:
        with open(args.json, "w") as f:
            json.dump(dict(args=vars(args), env=env, routes=routes, flops=flops, gates=gates, valid=valid, checks=checks,
                           diffs=diffs, samples_us=samples, stages_us=stages, clone_of=clone_of), f)


if __name__ == "__main__":
    main()
