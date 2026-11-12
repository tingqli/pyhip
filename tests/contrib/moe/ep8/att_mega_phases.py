"""Phase and stall breakdown of the EP8 mega kernel from rocprofv3 ATT UI folders (one per dispatch).

    python tests/contrib/moe/ep8/att_mega_phases.py OUT/ui_output_agent_*_dispatch_* [--json att.json]

Collect (rank 0 traced, the other ranks plain; see docs/megamoe_ep8_942/mega_moe_ep8_att.md):
    torchrun --nproc_per_node=8 --no-python WRAPPER tests/contrib/moe/ep8/bench_ep8_mori_flydsl.py --mode mega ...
    WRAPPER (LOCAL_RANK 0): rocprofv3 --att --att-library-path DECODER_DIR --att-target-cu 1
        --att-shader-engine-mask 0xffff --att-simd-select 0x1 --att-buffer-size 0x40000000
        --kernel-include-regex mega_kernel --kernel-iteration-range "[30, 31]" -d OUT -o att -- python "$@"

ATT conventions as tests/contrib/moe/bench_fused_moe_a2a_gfx942.py: an instruction issues at first attempt + stall,
`code[pc]` names it, an MFMA occupies its SIMD for 16 cycles from issue. The traced SIMD holds waves 0 (G0) and 4
(G1) of the CU's only workgroup. Landmarks are found in the code: the persistent loop is the longest back edge, the
two K loops are the long back edges with MFMAs (GateUp's epilogue starts with SiLU's v_exp_f32), the spin waits are
the short ones with s_sleep, the producer loop is the long back edge without MFMAs.

Per task and wave: ticket (persistent-loop head .. task dispatch), dispatch (.. the spin wait), wait (spin and the
barrier after it), prologue (.. K loop), K loop, epilogue (.. the take-ticket barrier). The K loop of a task on the
SIMD lasts while both waves are in it; MFMA efficiency = union of both waves' MFMA windows / cycles.
"""

import argparse
import bisect
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

MFMA_CYCLES = 16
WAVE_RE = re.compile(r"se(\d+)_sm(\d+)_sl(\d+)_wv(\d+)\.json")
PHASES = ("ticket", "dispatch", "wait", "prologue", "kloop", "epilogue")


def category(text):
    op = text.split()[0] if text else ""
    if op == "s_waitcnt":
        if "vmcnt" in text and "lgkmcnt" in text:
            return "waitcnt vm+lgkm"
        return "waitcnt vmcnt" if "vmcnt" in text else "waitcnt lgkmcnt"
    if op == "s_barrier":
        return "barrier"
    if op.startswith("v_mfma"):
        return "mfma"
    if op.startswith("ds_"):
        return "lds"
    if op.startswith(("buffer_load", "global_load", "s_load", "s_buffer_load")):
        return "load"
    if "atomic" in op:
        return "atomic"
    if op.startswith(("buffer_store", "global_store")):
        return "store"
    if op == "s_sleep":
        return "sleep"
    if op.startswith("v_"):
        return "valu"
    if op.startswith("s_"):
        return "salu"
    return "other"


def landmarks(asm, waves):
    back = Counter()
    for w in waves:
        for a, b in zip(w["pcs"], w["pcs"][1:]):
            if b < a:
                back[(a, b)] += 1
    has = lambda lo, hi, prefix: any(asm.get(pc, "").startswith(prefix) for pc in range(lo, hi + 1))
    longest = max(a - b for a, b in back)
    # The persistent loop's most taken back edge (waves other than 0 jump right after the take-ticket barrier).
    outer = max((e for e in back if e[0] - e[1] > 0.6 * longest), key=lambda e: back[e])
    heads = {b for a, b in back if a - b > 0.6 * longest}
    kloops = [(b, a) for a, b in back if 300 < a - b < 1500 and has(b, a, "v_mfma")]
    assert len(kloops) == 2, kloops
    gateup = [k for k in kloops if has(k[1] + 1, k[1] + 30, "v_exp_f32")]
    assert len(gateup) == 1, kloops
    down = [k for k in kloops if k != gateup[0]]
    producer = [(b, a) for a, b in back if a - b > 300 and not has(b, a, "v_mfma") and has(b, a, "buffer_store")]
    sleeps = sorted(pc for pc, text in asm.items() if text.startswith("s_sleep"))
    marks = {}
    for name, (head, tail) in (("G", gateup[0]), ("D", down[0])):
        sleep = max(pc for pc in sleeps if pc < head)
        ready = next(pc for pc in range(sleep, head) if asm[pc].startswith("s_barrier"))
        first = max(pc for pc in range(sleep - 12, sleep) if asm[pc].startswith("global_load_dword"))
        marks[name] = dict(head=head, tail=tail, wait_first=first, ready=ready)
    marks["P"] = dict(head=producer[0][0], tail=producer[0][1]) if producer else None
    ticket_barrier = outer[0] - 1
    assert asm[ticket_barrier].startswith("s_barrier"), asm[ticket_barrier]
    return dict(heads=heads, ticket_barrier=ticket_barrier, **marks)


def load_dispatch(dispatch):
    rows = json.loads((dispatch / "code.json").read_text())["code"]
    asm = {int(r[2]): str(r[0]).strip() for r in rows}
    waves = []
    for path in sorted(dispatch.glob("se*_sm*_sl*_wv*.json")):
        payload = json.loads(path.read_text())
        assert payload["num_insts"] == payload["num_stitched"], f"{path} not fully stitched"
        wave = payload["wave"]
        rec = wave["instructions"]
        waves.append(dict(se=int(WAVE_RE.fullmatch(path.name).group(1)), slot=int(wave["slot"]), cu=int(wave["cu"]),
                          simd=int(wave["simd"]), begin=int(wave["begin"]), end=int(wave["end"]), pcs=[int(r[4]) for r in rec],
                          attempt=[int(r[0]) for r in rec], stall=[int(r[2]) for r in rec]))
    realtime = {int(k[2:]): v for k, v in json.loads((dispatch / "realtime.json").read_text()).items() if k.startswith("SE")}
    return asm, waves, realtime


def split_tasks(w, lm, asm):
    """Per persistent-loop iteration of a wave: kind and phase boundaries (issue times)."""
    pcs, attempt, stall = w["pcs"], w["attempt"], w["stall"]
    issue = [a + s for a, s in zip(attempt, stall)]
    starts = [0] + [i for i in range(1, len(pcs)) if pcs[i] in lm["heads"] and pcs[i] < pcs[i - 1]]
    tasks = []
    for n, lo in enumerate(starts):
        hi = starts[n + 1] if n + 1 < len(starts) else len(pcs)
        seg = range(lo, hi)
        kind = next((k for k in ("G", "D") if any(pcs[i] == lm[k]["head"] for i in seg)), None)
        if kind is None and lm["P"] and any(pcs[i] == lm["P"]["head"] for i in seg):
            kind = "P"
        end = next((i for i in seg if pcs[i] == lm["ticket_barrier"]), None)
        t = dict(kind=kind or "-", lo=lo, hi=hi, start=issue[lo], next_start=issue[hi] if hi < len(pcs) else w["end"],
                 end=attempt[end] if end is not None else (issue[hi - 1] if hi > lo else issue[lo]))
        if kind in ("G", "D"):
            m = lm[kind]
            first = next((i for i in seg if pcs[i] == m["wait_first"]), None)
            k_in = next(i for i in seg if pcs[i] == m["head"])
            # An s_barrier has two records (arrival, release); the release carries the wait as its stall.
            ready = max(i for i in range(lo, k_in) if pcs[i] == m["ready"])
            k_tail = max(i for i in seg if pcs[i] == m["tail"])
            t.update(dispatch_end=issue[first] if first is not None else attempt[ready], ready=issue[ready], k_in=issue[k_in],
                     k_out=issue[k_tail + 1], k_in_index=k_in, k_out_index=k_tail + 1,
                     mfma=[issue[i] for i in range(k_in, k_tail + 1) if asm[pcs[i]].startswith("v_mfma")])
        tasks.append(t)
    return tasks, issue


def union_busy(windows, lo, hi):
    busy, cursor = 0, lo
    for a, b in sorted(windows):
        a, b = max(a, cursor), min(b, hi)
        if b > a:
            busy += b - a
            cursor = b
    return busy


def segments(win, lo, hi):
    """(start, end, label) pieces of a wave's time in [lo, hi): up to an instruction's attempt '<category> gap',
    then its stall '<category>'."""
    issue, attempt, cats = win
    j = bisect.bisect_right(issue, lo)
    cursor = lo
    while j < len(issue) and cursor < hi:
        for part, end in ((" gap", min(attempt[j], hi)), ("", min(issue[j], hi))):
            if end > cursor:
                yield cursor, end, cats[j] + part
                cursor = end
        j += 1
    if cursor < hi:
        yield cursor, hi, "outside loop"


def idle_attribution(wins, mfmas, lo, hi, at_barrier):
    """MFMA-idle cycles of the SIMD in [lo, hi) by what the wave issuing the next MFMA was doing; its barrier waits go
    through at_barrier(wave, start, end, out)."""
    idle, busy_end = Counter(), lo
    for t, who in mfmas:
        if lo < t <= hi and t > busy_end:
            for s, e, label in segments(wins[who], busy_end, t):
                if label == "barrier":
                    at_barrier(who, s, e, idle)
                else:
                    idle[label] += e - s
        busy_end = max(busy_end, t + MFMA_CYCLES)
    if busy_end < hi:
        idle["after last MFMA"] += hi - busy_end
    return idle


def barriers(w):
    """(arrival, release) of every s_barrier of a wave, in order: ATT gives a barrier two records, the second one
    (release) carries the wait as its stall."""
    out, i, n = [], 0, len(w["pcs"])
    while i < n:
        if w["cats"][i] == "barrier":
            assert i + 1 < n and w["pcs"][i + 1] == w["pcs"][i], (i, w["pcs"][i])
            out.append((w["attempt"][i], w["issue"][i + 1]))
            i += 2
        else:
            i += 1
    return out


def phase_bounds(t):
    if t["kind"] not in ("G", "D"):
        return {}
    return dict(ticket=(t["start"], t["start"]), dispatch=(t["start"], t["dispatch_end"]), wait=(t["dispatch_end"], t["ready"]),
                prologue=(t["ready"], t["k_in"]), kloop=(t["k_in"], t["k_out"]), epilogue=(t["k_out"], t["end"]))


def analyze(dispatch):
    asm, waves, realtime = load_dispatch(dispatch)
    lm = landmarks(asm, waves)
    cat_of = {pc: category(text) for pc, text in asm.items()}
    by_cu = defaultdict(list)
    for w in waves:
        w["tasks"], w["issue"] = split_tasks(w, lm, asm)
        w["cats"] = [cat_of[pc] for pc in w["pcs"]]
        w["wave0"] = any(pc == lm["G"]["wait_first"] or pc == lm["D"]["wait_first"] for pc in w["pcs"])
        by_cu[(w["se"], w["cu"])].append(w)
    phase_cycles = defaultdict(lambda: defaultdict(list))
    stalls = defaultdict(lambda: defaultdict(Counter))
    kloop = defaultdict(list)
    idle = defaultdict(Counter)
    kinds, stall_waves = Counter(), Counter()
    full_cus = 0
    for key, cu_waves in sorted(by_cu.items()):
        assert sum(w["wave0"] for w in cu_waves) <= 1, key
        kind_seq = [t["kind"] for t in cu_waves[0]["tasks"]]
        assert all([t["kind"] for t in w["tasks"]] == kind_seq for w in cu_waves), key
        full = len(cu_waves) == 8
        if full:
            # All 8 waves traced: a barrier wait is charged to the wave that arrived last, while it had not arrived.
            full_cus += 1
            bars = [barriers(w) for w in cu_waves]
            assert len({len(b) for b in bars}) == 1, (key, [len(b) for b in bars])
            index_of = {id(w): v for v, w in enumerate(cu_waves)}
            wins_all = [(w["issue"], w["attempt"], w["cats"]) for w in cu_waves]
        simds = defaultdict(list)
        for w in cu_waves:
            simds[w["simd"]].append(w)
        for simd, pair in sorted(simds.items()):
            assert len(pair) == 2, (key, simd)
            pair.sort(key=lambda w: w["slot"])
            if full:
                def at_barrier(who, s, e, out, pair=pair):
                    v = index_of[id(pair[who])]
                    k = bisect.bisect_right([b[0] for b in bars[v]], s) - 1
                    late = max(range(8), key=lambda u: bars[u][k][0])
                    arrived = bars[late][k][0]
                    for s2, e2, label in segments(wins_all[late], s, min(e, arrived)):
                        out["late wave: " + label] += e2 - s2
                    if e > max(s, arrived):
                        out["barrier release"] += e - max(s, arrived)
            else:
                def at_barrier(who, s, e, out, pair=pair):
                    other = pair[1 - who]
                    for s2, e2, label in segments((other["issue"], other["attempt"], other["cats"]), s, e):
                        out["other wave on SIMD: " + label] += e2 - s2
            for t0, t1 in zip(pair[0]["tasks"], pair[1]["tasks"]):
                if t0["kind"] not in ("G", "D"):
                    continue
                lo, hi = max(t0["k_in"], t1["k_in"]), min(t0["k_out"], t1["k_out"])
                windows = [(t, t + MFMA_CYCLES) for t in t0["mfma"] + t1["mfma"]]
                kloop[t0["kind"]].append(dict(cycles=hi - lo, busy=union_busy(windows, lo, hi), mfma=len(windows),
                                              span=max(t0["k_out"], t1["k_out"]) - min(t0["k_in"], t1["k_in"])))
                wins = [(w["issue"], w["attempt"], w["cats"]) for w in pair]
                mfmas = sorted([(x, 0) for x in t0["mfma"]] + [(x, 1) for x in t1["mfma"]])
                idle[t0["kind"]].update(idle_attribution(wins, mfmas, lo, hi, at_barrier))
        # Phase lengths from wave 0 (else the first traced wave); stall sums over every traced wave.
        lead = next((w for w in cu_waves if w["wave0"]), cu_waves[0])
        for t in lead["tasks"]:
            kinds[t["kind"]] += 1
            for phase, (lo, hi) in phase_bounds(t).items():
                if phase != "ticket":
                    phase_cycles[t["kind"]][phase].append(hi - lo)
        for prev, nxt in zip(lead["tasks"], lead["tasks"][1:]):
            phase_cycles[nxt["kind"]]["ticket"].append(nxt["start"] - prev["end"])
        for w in cu_waves:
            who = "wave 0" if w["wave0"] else ("G0" if w["slot"] == 0 else "G1")
            for t in w["tasks"]:
                stall_waves[(t["kind"], who)] += 1
                bounds = phase_bounds(t)
                for i in range(t["lo"] + 1, t["hi"]):
                    phase = next((p for p, (lo, hi) in bounds.items() if lo < w["issue"][i] <= hi),
                                 "ticket" if t["kind"] in ("G", "D") else t["kind"])
                    c = w["cats"][i]
                    stalls[(t["kind"], who)][phase][c + " stall"] += w["stall"][i]
                    stalls[(t["kind"], who)][phase][c + " gap"] += w["attempt"][i] - w["issue"][i - 1]
    clocks = []
    for se, samples in realtime.items():
        (c0, r0), (c1, r1) = samples[0], samples[-1]
        clocks.append((c1 - c0) / ((r1 - r0) / 1e8) / 1e9)
    summary = dict(dispatch=str(dispatch), kernel=asm.get(0, "").lstrip("; "), landmarks={k: v for k, v in lm.items() if k != "heads"},
                   traced_cus=len(by_cu), full_cus=full_cus, traced_waves=len(waves), tasks=dict(kinds),
                   clock_ghz=statistics.median(clocks), phases={}, kloop={}, stalls={})
    for kind, phases in phase_cycles.items():
        summary["phases"][kind] = {p: dict(n=len(v), mean=statistics.mean(v), p50=statistics.median(v), p95=sorted(v)[int(0.95 * (len(v) - 1))])
                                   for p, v in phases.items() if v}
    for kind, rows in kloop.items():
        summary["kloop"][kind] = dict(n=len(rows), cycles=statistics.mean(r["cycles"] for r in rows),
                                      span=statistics.mean(r["span"] for r in rows), mfma=statistics.mean(r["mfma"] for r in rows),
                                      eff=sum(r["busy"] for r in rows) / sum(r["cycles"] for r in rows),
                                      idle={c: v / len(rows) for c, v in idle[kind].most_common() if v})
    for (kind, who), phases in stalls.items():
        n = max(stall_waves[(kind, who)], 1)
        summary["stalls"][f"{kind} {who}"] = {p: {c: v / n for c, v in sorted(cnt.items(), key=lambda kv: -kv[1]) if v} for p, cnt in phases.items()}
    return summary


def report(s):
    lines = [f"{s['kernel']} {s['dispatch']}: {s['traced_cus']} CUs ({s['full_cus']} with all 8 waves), {s['traced_waves']} waves, "
             f"tasks {s['tasks']}, clock {s['clock_ghz']:.3f} GHz"]
    for kind in ("G", "D"):
        if kind not in s["phases"]:
            continue
        ph = s["phases"][kind]
        lines.append(f"  {kind}: " + "  ".join(f"{p} {ph[p]['mean'] / 1e3:.1f}k" for p in PHASES if p in ph) + " cycles (wave 0, mean)")
        k = s["kloop"][kind]
        lines.append(f"     K loop on a SIMD {k['cycles'] / 1e3:.1f}k cycles, {k['mfma']:.0f} MFMA, MFMA efficiency {k['eff'] * 100:.1f}%")
        lines.append("     MFMA idle by next issuer: " + ", ".join(f"{c} {v / 1e3:.2f}k" for c, v in list(k["idle"].items())[:12]))
        for who in ("wave 0", "G0", "G1"):
            st = s["stalls"].get(f"{kind} {who}", {})
            for phase in ("kloop", "epilogue", "prologue"):
                top = list(st.get(phase, {}).items())[:6]
                if top:
                    lines.append(f"     {who:6s} {phase:8s} " + ", ".join(f"{c} {v / 1e3:.1f}k" for c, v in top))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dispatches", nargs="+")
    parser.add_argument("--json")
    args = parser.parse_args()
    results = [analyze(Path(d)) for d in args.dispatches]
    for r in results:
        print(report(r))
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
