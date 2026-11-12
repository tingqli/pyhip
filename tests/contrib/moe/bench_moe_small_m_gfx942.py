"""Small-M timing: aiter gfx942 1-stage asm fmoe_fp8_blockscale_g1u1 vs our FlyDSL paths.

One EP8 rank of MiMo-V2.6-Flash (hidden 4096, moe_inter 2048, 256 experts, top-8: 32 local experts), inputs of
test_fmoe_a8w8_blockscale_asm_gfx942.make_inputs. Paths (each writes its own out and is checked after timing):
  asm <k>:       aiter moe_sorting(block_m of k, zeroing out) + the asm kernel k ("sort+asm" in the summary)
  bm256:         moe_sorting(256, no zeroing) + route counts + zero rows + GateUp 2x4_pf1 FP8 + Down 4x2_pf1 single_store
                 (fused_moe_a8w8_blockscale_gfx942 with bm128=False)
  bm128:         as bm256 with bm128=True (the entry's default): in both kernels a block with no route at row 128 runs
                 the 128-row pipeline; same grids
  merged:        moe_sorting(256, zeroing out) + fused_moe_a8w8_blockscale_gfx942_mixed (the path
                 fused_moe_a8w8_blockscale_gfx942_sort takes): one merged GateUp and one merged Down launch, BM256 tiles
                 for blocks with a route at row 128 and 8-wave pair tasks for the others, the Down adding with BF16
                 atomics. Checked like the others against the asm test's reference (SiLU output re-quantized per
                 (row, 128)).
Kernel-only variants ("k:" prefix) time one kernel with the routes sorted (256) beforehand.

Method (docs/record/moe_small_m_gfx942.md): queued blocks as in bench_moe_a8w8_blockscale_gfx942_vs_asm.py (head-start
sleep, >= --warmup-ms untimed calls, then timed calls with an event pair each), block order reversed every round,
--clones copies of a / scales / w1 / w2 rotated over the calls that read weights. The amdsmi gate (activity <= 5%,
VRAM <= --max-vram-pct, default 20%) is recorded at entry, between rounds and at the end; with --no-abort a failing
gate marks the run invalid instead of stopping it. Independently, gpu_contention.ExternalLoadMonitor samples the CU
occupancy of other processes during every timed block; a block during which another process ran is discarded and
repeated (counted in the report). KFD does not report the occupancy of every process, so only the activity gates
catch some jobs. A streaming-read probe of a 1 GiB buffer runs in the same process (HBM ceiling).

usage: HIP_VISIBLE_DEVICES=0 python bench_moe_small_m_gfx942.py --tokens 512 1024 2048 4096 [--json out.json]
"""

import argparse
import json
import math
import os
import statistics
import sys
import time

I, H, TOPK, E, EP = 2048, 4096, 8, 32, 8


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tokens", type=int, nargs="+", default=[512, 1024, 2048, 4096])
    parser.add_argument("--routing", default="ep_uniform", choices=["ep_uniform", "random", "balanced"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--asm", nargs="*", help="asm kernels (default: all host-sorted kernels of the CSV); auto is always kept")
    parser.add_argument("--paths", nargs="+", default=["asm", "bm128", "merged"], choices=["asm", "bm256", "bm128", "merged"])
    parser.add_argument("--kernels", action="store_true", help="also time single kernels (sorts, counts, GateUp, Down)")
    parser.add_argument("--clones", type=int, default=2)
    parser.add_argument("--samples", type=int, default=32, help="per variant, a multiple of --rounds")
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--warmup-ms", type=float, default=50.0)
    parser.add_argument("--no-abort", action="store_true", help="time even when the idle gate fails (run marked invalid)")
    parser.add_argument("--max-vram-pct", type=float, default=20.0, help="VRAM use the idle gate allows (other jobs' buffers)")
    parser.add_argument("--json")
    args = parser.parse_args(argv)
    assert args.samples % args.rounds == 0
    return args


def make_gate(gpu_index, max_vram_pct=20.0):
    import amdsmi

    amdsmi.amdsmi_init()
    handle = amdsmi.amdsmi_get_processor_handles()[gpu_index]
    records = []

    def gate(stage, settle=0.0):
        time.sleep(settle)
        activity = []
        for _ in range(5):
            activity.append(amdsmi.amdsmi_get_gpu_activity(handle)["gfx_activity"])
            time.sleep(0.1)
        memory = amdsmi.amdsmi_get_gpu_vram_usage(handle)
        power = amdsmi.amdsmi_get_power_info(handle).get("current_socket_power")
        record = dict(
            stage=stage, time=time.time(), bdf=amdsmi.amdsmi_get_gpu_device_bdf(handle), activity=activity,
            vram_used_mb=memory["vram_used"], vram_total_mb=memory["vram_total"], power_w=power,
            pids=len(amdsmi.amdsmi_get_gpu_process_list(handle)),
        )
        record["pass"] = max(activity) <= 5 and record["vram_used_mb"] <= max_vram_pct / 100 * record["vram_total_mb"]
        records.append(record)
        print("GATE", json.dumps(record), flush=True)
        return record["pass"]

    return gate, records


def hbm_probe(torch, cus, mib=1024, calls=20):
    """Median TB/s of a nontemporal streaming read of a mib MiB buffer (> MALL), 1 workgroup / CU."""
    import pyhip

    hip = pyhip.module(os.path.join(os.path.dirname(os.path.abspath(__file__)), "roofline_probe_gfx942.cpp"))
    buf = torch.ones((mib << 20) // 4, device="cuda", dtype=torch.int32)
    sink = torch.zeros(1024, device="cuda", dtype=torch.int32)
    per_wg = (mib << 20) // 16 // cus
    for _ in range(3):
        hip.read_u4([cus], [256], buf, per_wg, sink)
    torch.cuda.synchronize()
    torch.cuda._sleep(50_000_000)
    events = []
    for _ in range(calls):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        hip.read_u4([cus], [256], buf, per_wg, sink)
        end.record()
        events.append((start, end))
    torch.cuda.synchronize()
    us = sorted(s.elapsed_time(e) * 1e3 for s, e in events)
    return per_wg * cus * 16 / statistics.median(us) / 1e6, us


def main(argv=None):
    args = parse_args(argv)
    gpu = int(os.environ.get("HIP_VISIBLE_DEVICES", "0").split(",")[0])
    gate, gates = make_gate(gpu, args.max_vram_pct)
    if not gate("entry") and not args.no_abort:
        sys.exit("entry gate failed (GPU busy): pass --no-abort to time anyway (marked invalid)")
    from gpu_contention import ExternalLoadMonitor

    monitor = ExternalLoadMonitor(gpu)

    import torch

    torch.zeros(1, device="cuda")
    monitor.claim()

    import pyhip
    import test_fmoe_a8w8_blockscale_asm_gfx942 as asm_t
    from aiter import dtypes
    from aiter.fused_moe import moe_sorting
    from pyhip.contrib.flydsl import moe_merged_a8w8_blockscale_gfx942 as merged
    from pyhip.contrib.flydsl.moe_a8w8_blockscale_gfx942 import (
        fused_moe_a8w8_blockscale_gfx942, fused_moe_a8w8_blockscale_gfx942_mixed, max_sorted_blocks,
    )
    from pyhip.contrib.flydsl.moe_down_a8w8_blockscale_gfx942_lds import (
        moe_down_a8w8_blockscale_gfx942_lds, moe_route_counts, moe_zero_rows,
    )
    from pyhip.contrib.flydsl.moe_gateup_a8w8_blockscale_gfx942_lds import moe_gateup_a8w8_blockscale_gfx942_lds

    assert torch.cuda.get_device_properties(0).pci_bus_id == int(gates[0]["bdf"].split(":")[1], 16), (
        "HIP device 0 is not the GPU the gate watches: set HIP_VISIBLE_DEVICES to one physical GPU index"
    )
    props = torch.cuda.get_device_properties(0)
    env = dict(gpu=props.name, arch=props.gcnArchName, cus=props.multi_processor_count, torch=torch.__version__, hip=torch.version.hip)
    try:
        from importlib.metadata import version

        env["flydsl"] = version("flydsl")
    except Exception:
        env["flydsl"] = None
    print("ENV", json.dumps(env), flush=True)
    hbm_tbps, hbm_us = hbm_probe(torch, props.multi_processor_count)
    print(f"HBM probe: {hbm_tbps:.2f} TB/s median (1 GiB nontemporal read)", flush=True)
    report = dict(args=vars(args), env=env, gates=gates, hbm_probe=dict(median_tbps=hbm_tbps, samples_us=hbm_us), runs=[])

    asm_names = [k for k in asm_t.KERNELS if args.asm is None or k in args.asm or k == "auto"]
    no_zero = torch.empty((0, 0), device="cuda", dtype=torch.bfloat16)
    for tokens in args.tokens:
        torch.cuda.empty_cache()
        inputs = asm_t.make_inputs(tokens, I, H, TOPK, E, EP, args.routing, seed=args.seed)
        expected = asm_t.reference(inputs)
        out = torch.empty((tokens, H), device="cuda", dtype=torch.bfloat16)
        del inputs["w1_raw"], inputs["w2_raw"]
        torch.cuda.empty_cache()
        routes = inputs["routes"][0].numel()
        per_expert = torch.bincount(inputs["routes"][2], minlength=E)
        flops = 6 * routes * I * H
        weight_bytes = inputs["w1"].numel() + inputs["w2"].numel()
        clones = [inputs] + [
            dict(inputs, **{n: inputs[n].clone() for n in ("a", "a_scale_t", "w1", "w1_scale", "w2", "w2_scale")})
            for _ in range(args.clones - 1)
        ]
        marks = None

        def mark(label):
            if marks is not None:
                event = torch.cuda.Event(enable_timing=True)
                event.record()
                marks.append((label, event))

        def sort(block_m, zero):
            sorted_ids, sorted_weights, expert_ids, num_valid_ids, _ = moe_sorting(
                inputs["topk_ids"], inputs["topk_weights"], E * EP, H, dtypes.bf16, block_m, inputs["expert_mask"],
                output=out if zero else no_zero,
            )
            return dict(sorted_ids=sorted_ids, sorted_weights=sorted_weights, sorted_expert_ids=expert_ids,
                        num_valid_ids=num_valid_ids, out=out, block_m=block_m)

        def path_asm(c, k):
            s = sort(asm_t.KERNELS[k]["block_m"], True)
            mark("sort")
            asm_t.run_kernel(c, s, k)

        def path_bm256(c, bm128=False):
            s = sort(256, False)
            mark("sort")
            fused_moe_a8w8_blockscale_gfx942(
                c["a"], c["a_scale_t"], c["w1"], c["w1_scale"], c["w2"], c["w2_scale"], s["sorted_ids"], s["sorted_weights"],
                s["sorted_expert_ids"], s["num_valid_ids"], out, TOPK, bm128=bm128,
            )

        def path_merged(c):
            s = sort(256, True)
            mark("sort")
            fused_moe_a8w8_blockscale_gfx942_mixed(
                c["a"], c["a_scale_t"], c["w1"], c["w1_scale"], c["w2"], c["w2_scale"], s["sorted_ids"], s["sorted_weights"],
                s["sorted_expert_ids"], s["num_valid_ids"], out, TOPK,
            )

        # name -> (prep or None, fn(clone), reference to check against or None, reads weights)
        variants = {}
        if "asm" in args.paths:
            for k in asm_names:
                variants[f"asm {k}"] = (None, lambda c, k=k: path_asm(c, k), expected, True)
        if "bm256" in args.paths:
            variants["bm256"] = (None, path_bm256, expected, True)
        if "bm128" in args.paths:
            variants["bm128"] = (None, lambda c: path_bm256(c, True), expected, True)
        if "merged" in args.paths:
            variants["merged"] = (None, path_merged, expected, True)
        if args.kernels:
            s256 = sort(256, False)
            blocks = min(s256["sorted_expert_ids"].numel(), max_sorted_blocks(tokens, TOPK, E))
            e256 = s256["sorted_expert_ids"][:blocks]
            a2 = torch.empty((blocks * 256, I), device="cuda", dtype=dtypes.fp8)
            a2_scale_t = torch.empty((I // 128, blocks * 256), device="cuda")
            counts = torch.empty(tokens, device="cuda", dtype=torch.int32)
            moe_route_counts(s256["sorted_ids"], s256["num_valid_ids"], e256, counts, TOPK)

            def gateup256(c, bm128):
                moe_gateup_a8w8_blockscale_gfx942_lds(
                    c["a"], c["w1"], c["a_scale_t"], c["w1_scale"], s256["sorted_ids"], e256, s256["num_valid_ids"], a2,
                    a2_scale_t, TOPK, bm128=bm128,
                )

            def down256(c, bm128):
                moe_down_a8w8_blockscale_gfx942_lds(
                    a2, c["w2"], a2_scale_t, c["w2_scale"], s256["sorted_ids"], s256["sorted_weights"], e256,
                    s256["num_valid_ids"], out, TOPK, zero_out=False, single_store=True, route_counts=counts, bm128=bm128,
                )

            variants["k: sort256NZ"] = (None, lambda c: sort(256, False), None, False)
            variants["k: sort256Z"] = (None, lambda c: sort(256, True), None, False)
            variants["k: counts"] = (None, lambda c: moe_route_counts(s256["sorted_ids"], s256["num_valid_ids"], e256, counts, TOPK), None, False)
            variants["k: zero_rows"] = (None, lambda c: moe_zero_rows(out, counts), None, False)
            if "bm256" in args.paths:
                variants["k: gateup256"] = (None, lambda c: gateup256(c, False), None, True)
                variants["k: down256"] = (None, lambda c: down256(c, False), None, True)
            if "bm128" in args.paths:
                variants["k: gateup_bm128"] = (None, lambda c: gateup256(c, True), None, True)
                variants["k: down_bm128"] = (None, lambda c: down256(c, True), None, True)
            if "merged" in args.paths:
                variants["k: gateup_merged"] = (None, lambda c: merged.moe_gateup_merged(
                    c["a"], c["w1"], c["a_scale_t"], c["w1_scale"], s256["sorted_ids"], e256, s256["num_valid_ids"], a2, a2_scale_t,
                    TOPK), None, True)
                variants["k: down_merged"] = (None, lambda c: merged.moe_down_merged(
                    a2, c["w2"], a2_scale_t, c["w2_scale"], s256["sorted_ids"], s256["sorted_weights"], e256, s256["num_valid_ids"],
                    out, TOPK), None, True)
            for k in asm_names if "asm" in args.paths else []:
                sk = sort(asm_t.KERNELS[k]["block_m"], True)
                variants[f"k: asm {k}"] = (out.zero_, lambda c, k=k, sk=sk: asm_t.run_kernel(c, sk, k), None, True)

        order = list(variants)
        next_clone = 0
        checks, diffs, samples, stages = {}, {}, {n: [] for n in order}, {}
        contaminated = {n: 0 for n in order}

        def pick(name):
            nonlocal next_clone
            prep, fn, _, reads = variants[name]
            index = next_clone % len(clones) if reads else 0
            next_clone += reads
            return prep, fn, clones[index]

        def call(name):
            prep, fn, clone = pick(name)
            if prep:
                prep()
            fn(clone)

        def check(name):
            reference = variants[name][2]
            if reference is None:
                return
            torch.cuda.synchronize()
            try:
                asm_t.check(inputs, out, reference)
                result = "pass"
            except AssertionError as error:
                result = f"FAIL {error}"
            if checks.get(name, "pass") == "pass":
                checks[name] = result
            routed = torch.bincount(inputs["routes"][0], minlength=tokens) > 0
            diffs[name] = pyhip.calc_diff(out[routed].float(), reference[routed])

        def poison():
            out.fill_(math.nan)

        def queued_block(name, warmups, n):
            """Times n calls; repeats the block (up to 30 times) while another process runs kernels on the GPU."""
            nonlocal marks
            for attempt in range(30):
                with monitor:
                    torch.cuda._sleep(60_000_000)
                    for _ in range(warmups):
                        call(name)
                    calls = []
                    for _ in range(n):
                        prep, fn, clone = pick(name)
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
                if monitor.clean:
                    break
                contaminated[name] += 1
                time.sleep(1.0)
            for start, call_marks, end in calls:
                if call_marks:
                    labels = [label for label, _ in call_marks] + ["rest"]
                    events = [start] + [event for _, event in call_marks] + [end]
                    for label, a, b in zip(labels, events, events[1:]):
                        stages.setdefault(name, {}).setdefault(label, []).append(a.elapsed_time(b) * 1e3)
            return [start.elapsed_time(end) * 1e3 for start, _, end in calls]

        for name in order:
            poison()
            for _ in range(2):
                call(name)
            check(name)
        torch.cuda.synchronize()
        estimate_ms = {}
        for name in order:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(5):
                call(name)
            torch.cuda.synchronize()
            estimate_ms[name] = (time.perf_counter() - t0) * 1e3 / 5
        gate(f"T={tokens} before sampling", settle=0.3)
        for r in range(args.rounds):
            for name in order if r % 2 == 0 else order[::-1]:
                warmups = min(200, max(8, math.ceil(args.warmup_ms / estimate_ms[name])))
                if variants[name][2] is not None:
                    poison()
                samples[name] += queued_block(name, warmups, args.samples // args.rounds)
                check(name)
            gate(f"T={tokens} round {r}", settle=0.3)

        med = {n: statistics.median(samples[n]) for n in order}

        def without_sort(name):
            if name in stages and "sort" in stages[name]:
                return statistics.median(t - s for t, s in zip(samples[name], stages[name]["sort"]))
            return med[name]

        asm_full = [n for n in order if n.startswith("asm ")]
        tuned = min(asm_full, key=med.get) if asm_full else None
        print(f"\nT={tokens} {args.routing}: {routes} local routes, routes/expert min {per_expert.min().item()} "
              f"max {per_expert.max().item()}, {flops / 1e9:.1f} GFLOP, weights {weight_bytes / 2**20:.0f} MiB")
        print(f"  {'variant':28s} {'median us':>9s} {'p10-p90 us':>17s} {'no-sort us':>10s} {'TFLOPS':>7s} {'TB/s(w)':>7s}  calc_diff  check")
        for name in order:
            x = sorted(samples[name])
            spread = f"{x[len(x) // 10]:8.1f}-{x[len(x) * 9 // 10]:<8.1f}"
            redo = f" (+{contaminated[name]} contaminated blocks redone)" if contaminated[name] else ""
            print(f"  {name:28s} {med[name]:9.1f} {spread} {without_sort(name):10.1f} {flops / med[name] / 1e6:7.1f} "
                  f"{weight_bytes / med[name] / 1e6:7.2f}  {diffs.get(name, float('nan')):9.2e}  {checks.get(name, '')}{redo}", flush=True)
        if tuned:
            for ours in ("bm256", "bm128", "merged"):
                if ours in med:
                    print(f"  {ours}: {med[tuned] / med[ours]:.3f}x vs tuned {tuned} (with sort), "
                          f"{without_sort(tuned) / without_sort(ours):.3f}x without sort; "
                          f"{med['asm auto'] / med[ours]:.3f}x vs asm auto", flush=True)
        report["runs"].append(dict(
            tokens=tokens, routing=args.routing, routes=routes, routes_per_expert=per_expert.tolist(), flops=flops,
            weight_bytes=weight_bytes, samples_us=samples, stages_us=stages, checks=checks, diffs=diffs, tuned_asm=tuned,
            median_us=med, without_sort_us={n: without_sort(n) for n in order}, contaminated_blocks=contaminated,
        ))
        del clones, inputs, expected, variants
        torch.cuda.empty_cache()

    gate("end", settle=0.3)
    report["valid"] = all(g["pass"] for g in gates)
    print(f"gates {'PASS' if report['valid'] else 'FAIL (results contaminated by other GPU work)'}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f)


if __name__ == "__main__":
    main()
