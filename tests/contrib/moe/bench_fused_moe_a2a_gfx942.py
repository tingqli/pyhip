"""aiter.fused_moe on gfx942 with MORI-style all-to-all EP8 FP8 input and balanced routing: asm vs FlyDSL (GateUp
pair_blocks off / on) timing, per-kernel times, and the ATT prolog / K-loop / epilog split of the FlyDSL kernels.

  time:   HIP_VISIBLE_DEVICES=3 python bench_fused_moe_a2a_gfx942.py time --tokens 16384 32768 --json a2a.json
  ATT:    HIP_VISIBLE_DEVICES=3 rocprofv3 --att --att-library-path DECODER_DIR --att-target-cu 1 \\
              --att-shader-engine-mask 0xffff --att-simd-select 0x1 --att-buffer-size 0x40000000 \\
              --kernel-include-regex "gateup_kernel_0|down_kernel_0" --kernel-iteration-range "[7, 9]" -d OUT -o att \\
              -- python bench_fused_moe_a2a_gfx942.py run --tokens 16384 --calls 10
          asm kernel instead: --kernel-include-regex "fmoe_bf16_blockscaleFp8" and run ... --path asm
          python bench_fused_moe_a2a_gfx942.py phases OUT/ui_output_agent_*_dispatch_* --json phases.json

Needs an aiter with the FlyDSL hook (see test_aiter_fused_moe_flydsl_gfx942.py); DECODER_DIR holds
librocprof-trace-decoder.so. Calls rotate through clone sets of the weights and the receive buffers, so each starts cold.

ATT conventions as tests/flydsl/attn_4wave/tools/stall_analysis.md: an instruction issues at first_attempt + stall,
`code[pc_index]` names it, an MFMA occupies its SIMD for 16 cycles from issue. The K loop is the most-taken backward
branch. Each workgroup's G0 and G1 wave share the traced SIMD (one 64 KiB-LDS workgroup per CU); prolog lasts until
both are in the loop, mainloop while both are, epilog from the first one leaving it. MFMA efficiency = union of the
two waves' MFMA windows / phase cycles.
"""

import argparse
import itertools
import json
import os
import re
import statistics
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

ENV = "AITER_FMOE_FP8_BLOCKSCALE_FLYDSL"
I, H, TOPK, E, EP = 2048, 4096, 8, 32, 8
PATHS = ("asm", "flydsl", "flydsl+pair")
MFMA_CYCLES = 16
REALTIME_HZ = 100e6
WAVE_RE = re.compile(r"se(\d+)_sm(\d+)_sl(\d+)_wv(\d+)\.json")
PHASES = ("prolog", "mainloop", "epilog")

_pair_blocks = {"on": False}


def select(path):
    os.environ[ENV] = "0" if path == "asm" else "1"
    _pair_blocks["on"] = path == "flydsl+pair"


def prepare(tokens, clones):
    """(case, clone sets, call) for an a2a receive buffer of `tokens` rows."""
    import aiter.fused_moe as fm
    import pyhip.contrib.flydsl.moe_a8w8_blockscale_gfx942 as fused
    from aiter import ActivationType, QuantType, dtypes
    from test_aiter_fused_moe_flydsl_gfx942 import a2a_inputs
    from test_fmoe_a8w8_blockscale_asm_gfx942 import make_inputs

    if not getattr(fused, "_bench_pair_blocks", False):
        gateup = fused.moe_gateup_a8w8_blockscale_gfx942_lds
        fused.moe_gateup_a8w8_blockscale_gfx942_lds = lambda *a, **kw: gateup(*a, pair_blocks=_pair_blocks["on"], **kw)
        fused._bench_pair_blocks = True
    os.environ[f"{ENV}_MIN_TOKENS"] = "1"
    inputs = make_inputs(tokens, I, H, TOPK, E, EP, "balanced")
    case, buffers = a2a_inputs(inputs, tokens)
    sets = []
    for i in range(clones):
        s = {k: inputs[k] if i == 0 else inputs[k].clone() for k in ("w1", "w2", "w1_scale", "w2_scale")}
        s.update({k: v if i == 0 else v.clone() for k, v in buffers.items()})
        # clone() drops shuffle_weight's tag, which aiter's config lookup reads.
        s["w1"].is_shuffled = s["w2"].is_shuffled = True
        sets.append(s)

    def call(s):
        return fm.fused_moe(
            s["a"], s["w1"], s["w2"], s["topk_weights"], s["topk_ids"], expert_mask=case["expert_mask"],
            activation=ActivationType.Silu, quant_type=QuantType.per_128x128, w1_scale=s["w1_scale"],
            w2_scale=s["w2_scale"], a1_scale=s["a1_scale"], dtype=dtypes.bf16, num_local_tokens=s["num_local_tokens"],
        )

    return case, sets, call


def time_paths(args):
    import torch
    from test_fmoe_a8w8_blockscale_asm_gfx942 import check, reference
    from torch.profiler import ProfilerActivity, profile

    results = []
    for tokens in args.tokens:
        case, sets, call = prepare(tokens, args.clones)
        n, routes = case["shape"][0], case["routes"][0].numel()
        expected = reference(case)
        outs, diffs = {}, {}
        for key, path in (("asm", "asm"), ("flydsl", "flydsl"), ("flydsl+pair", "flydsl+pair"), ("rerun", "flydsl")):
            select(path)
            outs[key] = call(sets[0])[:n].clone()
            torch.cuda.synchronize()
            diffs[key] = check(case, outs[key], expected)
        # Tokens with one local route are stored once; the others add BF16 atomics in run-dependent order.
        single = torch.bincount(case["routes"][0], minlength=n) == 1
        bits = {k: v.view(torch.int16) for k, v in outs.items()}
        assert torch.equal(bits["flydsl"][single], bits["flydsl+pair"][single]), "pair_blocks changed a single-route row"
        multi_differ = {k: (bits["flydsl"][~single] != bits[k][~single]).float().mean().item() for k in ("rerun", "flydsl+pair")}
        del outs, bits, expected

        def segment(start):
            for i in range(args.warmup):
                call(sets[(start + i) % len(sets)])
            torch.cuda.synchronize()
            events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(args.iters)]
            for i, (begin, end) in enumerate(events):
                begin.record()
                call(sets[(start + args.warmup + i) % len(sets)])
                end.record()
            torch.cuda.synchronize()
            return [begin.elapsed_time(end) * 1000 for begin, end in events]

        rounds = []
        for r, order in enumerate(list(itertools.permutations(PATHS)) * args.repeat):
            rounds.append({})
            for path in order:
                select(path)
                rounds[-1][path] = segment(r)

        kernels = {}
        for path in PATHS:
            select(path)
            for i in range(3):
                call(sets[i % len(sets)])
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                for i in range(args.profile_calls):
                    call(sets[i % len(sets)])
                torch.cuda.synchronize()
            with tempfile.NamedTemporaryFile(suffix=".json") as trace:
                prof.export_chrome_trace(trace.name)
                events = sorted((e for e in json.load(open(trace.name))["traceEvents"] if e.get("cat") == "kernel"),
                                key=lambda e: e["ts"])
            per_call = len(events) // args.profile_calls
            assert per_call * args.profile_calls == len(events), f"{len(events)} kernels over {args.profile_calls} calls"
            kernels[path] = [(events[k]["name"], statistics.median(events[c * per_call + k]["dur"] for c in range(args.profile_calls)))
                             for k in range(per_call)]

        med = {p: statistics.median(s for r in rounds for s in r[p]) for p in PATHS}
        paired = [statistics.median(r["flydsl"]) / statistics.median(r["flydsl+pair"]) for r in rounds]
        print(f"T={tokens}: {n} valid rows, {routes} local routes", flush=True)
        for p in PATHS:
            values = sorted(s for r in rounds for s in r[p])
            print(f"  {p:12s} median {med[p]:8.1f} us  p10-p90 {values[len(values) // 10]:.1f}-{values[len(values) * 9 // 10]:.1f}"
                  f"  calc_diff {diffs[p]:.2e}  asm/this {med['asm'] / med[p]:.3f}x")
        print(f"  pair_blocks: {(med['flydsl+pair'] / med['flydsl'] - 1) * 100:+.2f}%, faster in {sum(x > 1 for x in paired)}/{len(paired)} "
              f"rounds; multi-route elements differing from flydsl: rerun {multi_differ['rerun'] * 100:.2f}%, "
              f"pair {multi_differ['flydsl+pair'] * 100:.2f}%")
        for p in PATHS:
            print(f"  kernels {p}: " + ", ".join(f"{name[:40]} {dur:.1f}" for name, dur in kernels[p]), flush=True)
        results.append(dict(tokens=tokens, valid_rows=n, routes=routes, calc_diff=diffs, multi_route_differ=multi_differ,
                            rounds=rounds, kernels=kernels))
        del case, sets
        torch.cuda.empty_cache()
    return results


def run_calls(args):
    import torch

    _, sets, call = prepare(args.tokens, args.clones)
    select(args.path)
    for i in range(args.calls):
        call(sets[i % len(sets)])
    torch.cuda.synchronize()


def load_dispatch(dispatch):
    rows = json.loads((dispatch / "code.json").read_text())["code"]
    asm = [""] * (max(int(r[2]) for r in rows) + 1)
    for r in rows:
        asm[int(r[2])] = str(r[0]).strip()
    waves = []
    for path in sorted(dispatch.glob("se*_sm*_sl*_wv*.json")):
        payload = json.loads(path.read_text())
        assert payload["num_insts"] == payload["num_stitched"], f"{path} not fully stitched"
        wave = payload["wave"]
        pcs = [int(r[4]) for r in wave["instructions"]]
        issue = [int(r[0]) + int(r[2]) for r in wave["instructions"]]
        waves.append(dict(se=int(WAVE_RE.fullmatch(path.name).group(1)), cu=int(wave["cu"]), simd=int(wave["simd"]),
                          begin=int(wave["begin"]), end=int(wave["end"]), pcs=pcs, issue=issue,
                          mfma=[t for pc, t in zip(pcs, issue) if asm[pc].startswith("v_mfma")]))
    realtime = {int(k[2:]): v for k, v in json.loads((dispatch / "realtime.json").read_text()).items() if k.startswith("SE")}
    return asm, waves, realtime


def wave_loop(w, head, tail):
    """(loop entry, loop exit, MFMA count before / in / after the loop) of a wave, None if it never enters the loop."""
    pcs = w["pcs"]
    if head not in pcs:
        return None
    entry = w["issue"][pcs.index(head)]
    exit_index = len(pcs) - pcs[::-1].index(tail)
    exit_ = w["issue"][exit_index] if exit_index < len(pcs) else w["end"]
    counts = [sum(lo <= t < hi for t in w["mfma"]) for lo, hi in ((w["begin"], entry), (entry, exit_), (exit_, w["end"] + 1))]
    return entry, exit_, counts


def union_busy(windows, lo, hi):
    busy, cursor = 0, lo
    for a, b in windows:
        a, b = max(a, cursor), min(b, hi)
        if b > a:
            busy += b - a
            cursor = b
    return busy


def analyze_dispatch(dispatch):
    asm, waves, realtime = load_dispatch(dispatch)
    (tail, head), back_edges = Counter((a, b) for w in waves for a, b in zip(w["pcs"], w["pcs"][1:]) if b < a).most_common(1)[0]
    by_simd = defaultdict(list)
    for w in waves:
        w["loop"] = wave_loop(w, head, tail)
        by_simd[(w["se"], w["cu"], w["simd"])].append(w)
    workgroups, simds = [], []
    for (se, cu, simd), group in sorted(by_simd.items()):
        group.sort(key=lambda w: w["begin"])
        runs = []
        for w in group:
            if runs and w["begin"] < max(x["end"] for x in runs[-1]):
                runs[-1].append(w)
            else:
                runs.append([w])
        rows = []
        for run in runs:
            assert len(run) == 2, f"{len(run)} resident waves on SIMD {(se, cu, simd)}"
            begin, end = min(w["begin"] for w in run), max(w["end"] for w in run)
            row = dict(begin=begin, end=end, lifetime=end - begin, working=all(w["loop"] for w in run))
            if row["working"]:
                entry, exit_ = max(w["loop"][0] for w in run), min(w["loop"][1] for w in run)
                windows = sorted((t, t + MFMA_CYCLES) for w in run for t in w["mfma"])
                for phase, (lo, hi) in zip(PHASES, ((begin, entry), (entry, exit_), (exit_, end))):
                    row[phase], row[f"{phase}_mfma"] = hi - lo, union_busy(windows, lo, hi)
                row["wave_mfma"] = [tuple(w["loop"][2]) for w in run]
            rows.append(row)
        workgroups += rows
        (c0, r0), (c1, r1) = realtime[se][0], realtime[se][-1]
        simds.append(dict(se=se, first=rows[0]["begin"], last=rows[-1]["end"], ghz=(c1 - c0) / ((r1 - r0) / REALTIME_HZ) / 1e9,
                          rt_first=r0 + (rows[0]["begin"] - c0) * (r1 - r0) / (c1 - c0),
                          rt_last=r0 + (rows[-1]["end"] - c0) * (r1 - r0) / (c1 - c0),
                          working=sum(r["lifetime"] for r in rows if r["working"]), empty=sum(r["lifetime"] for r in rows if not r["working"]),
                          working_wgs=sum(r["working"] for r in rows), empty_wgs=sum(not r["working"] for r in rows)))
    working = [w for w in workgroups if w["working"]]
    empty = [w["lifetime"] for w in workgroups if not w["working"]]
    lifetime = sum(w["lifetime"] for w in working)
    ghz = statistics.median(s["ghz"] for s in simds)
    span = sum(s["last"] - s["first"] for s in simds)
    phases = {}
    for phase in PHASES:
        cycles = sorted(w[phase] for w in working)
        phases[phase] = dict(share=sum(cycles) / lifetime, mean_cycles=statistics.mean(cycles), p50=statistics.median(cycles),
                             p95=cycles[int(0.95 * (len(cycles) - 1))], mean_us=statistics.mean(cycles) / ghz / 1e3,
                             mfma_eff=sum(w[f"{phase}_mfma"] for w in working) / sum(cycles))
    return dict(
        kernel=asm[0].lstrip("; ").strip(), dispatch=str(dispatch), loop_iterations=back_edges // (2 * len(working)) + 1,
        traced_simds=len(simds), working_wgs=len(working), empty_wgs=len(empty), clock_ghz=ghz,
        span_us=(max(s["rt_last"] for s in simds) - min(s["rt_first"] for s in simds)) / REALTIME_HZ * 1e6,
        wg_cycles=statistics.mean(w["lifetime"] for w in working), empty_wg_cycles=statistics.mean(empty) if empty else 0.0,
        phases=phases, wg_mfma_eff=sum(w[f"{p}_mfma"] for w in working for p in PHASES) / lifetime,
        wave_mfma=Counter(m for w in working for m in w["wave_mfma"]).most_common(2),
        simd_time=dict(working=sum(s["working"] for s in simds) / span, empty=sum(s["empty"] for s in simds) / span),
    )


def phases_report(r):
    lines = [f"{r['kernel']}: {r['traced_simds']} traced SIMDs, {r['working_wgs']} working + {r['empty_wgs']} empty workgroups, "
             f"K loop {r['loop_iterations']} iterations, {r['clock_ghz']:.3f} GHz, span {r['span_us']:.0f} us",
             f"  workgroup {r['wg_cycles']:.0f} cycles ({r['wg_cycles'] / r['clock_ghz'] / 1e3:.1f} us), "
             f"empty workgroup {r['empty_wg_cycles']:.0f} cycles"]
    for phase in PHASES:
        q = r["phases"][phase]
        lines.append(f"  {phase:8s} {q['share'] * 100:5.1f}%  mean {q['mean_cycles']:8.0f} cycles ({q['mean_us']:6.1f} us)  "
                     f"p50 {q['p50']:.0f} p95 {q['p95']:.0f}  MFMA {q['mfma_eff'] * 100:5.1f}%")
    lines.append(f"  workgroup MFMA {r['wg_mfma_eff'] * 100:.1f}%; MFMA per wave (prolog, loop, epilog): {r['wave_mfma']}; "
                 f"traced-SIMD time: working {r['simd_time']['working'] * 100:.1f}%, empty {r['simd_time']['empty'] * 100:.1f}%")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    timing = sub.add_parser("time", help="asm / FlyDSL / FlyDSL+pair_blocks, all 6 orders interleaved")
    timing.add_argument("--tokens", type=int, nargs="+", default=[16384, 32768], help="receive-buffer rows")
    timing.add_argument("--clones", type=int, default=5)
    timing.add_argument("--iters", type=int, default=16)
    timing.add_argument("--warmup", type=int, default=8)
    timing.add_argument("--repeat", type=int, default=2, help="passes over the 6 path orders")
    timing.add_argument("--profile-calls", type=int, default=20)
    timing.add_argument("--json")
    run = sub.add_parser("run", help="fused_moe calls of one path, for rocprofv3 --att")
    run.add_argument("--tokens", type=int, required=True)
    run.add_argument("--calls", type=int, default=10)
    run.add_argument("--clones", type=int, default=5)
    run.add_argument("--path", choices=PATHS, default="flydsl")
    att = sub.add_parser("phases", help="prolog / K-loop / epilog and MFMA efficiency from rocprofv3 ATT UI folders")
    att.add_argument("dispatches", nargs="+")
    att.add_argument("--json")
    args = parser.parse_args()
    if args.command == "time":
        results = time_paths(args)
    elif args.command == "run":
        return run_calls(args)
    else:
        results = [analyze_dispatch(Path(d)) for d in args.dispatches]
        for r in results:
            print(phases_report(r), flush=True)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
