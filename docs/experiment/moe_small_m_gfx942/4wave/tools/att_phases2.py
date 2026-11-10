"""Per-K-block phases for the 2-barrier K loop of every traced wave (ATT records an s_barrier twice: issue, then
release after the wait) and per-SIMD MFMA pipe utilization while two waves are resident.
Usage: att_phases2.py <ui_output dir> [k_blocks]"""
import collections
import json
import statistics
import sys
from pathlib import Path

d = Path(sys.argv[1])
KB = int(sys.argv[2]) if len(sys.argv) > 2 else 32
asm = {int(r[2]): str(r[0]).strip() for r in json.loads((d / "code.json").read_text())["code"]}
kind = lambda r: asm.get(r[4], "")  # noqa: E731
stats = collections.defaultdict(list)
mfma_by_simd = collections.defaultdict(list)
spans = collections.defaultdict(list)
stall_by_op = collections.Counter()
for path in sorted(d.glob("se*_sm*_sl*_wv*.json")):
    insts = json.loads(path.read_text())["wave"]["instructions"]
    if len(insts) < 1000:
        continue
    simd = path.name.split("_sl")[0]
    raw = [i for i, r in enumerate(insts) if kind(r).startswith("s_barrier")]
    # (issue index, release time, wait)
    bars = []
    for i in raw:
        if bars and i == bars[-1][0] + 1:
            bars[-1] = (i, insts[i][0] + insts[i][2] + insts[i][3], insts[i][2])
        else:
            bars.append((i, insts[i][0] + insts[i][2] + insts[i][3], insts[i][2]))
    if len(bars) < 2 * KB + 1:
        continue
    for j in range(2, KB - 2):
        (p2, t_p2, _), (b1, t_b1, w1), (b2, t_b2, w2) = bars[2 * j], bars[2 * j + 1], bars[2 * j + 2]
        comp = insts[p2 + 1 : b1 - 1]
        mem = insts[b1 + 1 : b2 - 1]
        mf = [r for r in comp if kind(r).startswith("v_mfma")]
        stats["period"].append(t_b2 - t_p2)
        stats["compute"].append(insts[b1 - 1][0] - t_p2)
        stats["bar1_wait"].append(w1)
        stats["memory"].append(insts[b2 - 1][0] - t_b1)
        stats["bar2_wait"].append(w2)
        if mf:
            stats["mfma_span"].append(mf[-1][0] + mf[-1][2] - (mf[0][0] + mf[0][2]))
        for r in comp + mem:
            op = kind(r).split(" ")[0]
            group = ("mfma" if op.startswith("v_mfma") else "vmem" if op.startswith("buffer_") else "lds" if op.startswith("ds_")
                     else "waitcnt" if op.startswith("s_waitcnt") else "valu" if op.startswith("v_") else "salu/other")
            stall_by_op[group] += r[2]
    spans[simd].append((bars[2][1], bars[2 * KB - 2][1]))
    for r in insts:
        if kind(r).startswith("v_mfma"):
            mfma_by_simd[simd].append(r[0] + r[2])
for k, v in stats.items():
    q = statistics.quantiles(v, n=10)
    print(f"{k:12s} n={len(v):5d} median {statistics.median(v):7.0f}  p10 {q[0]:7.0f}  p90 {q[-1]:7.0f}  mean {statistics.mean(v):7.0f}")
total = sum(stall_by_op.values())
print("stall ticks by instruction group (K loop):", {k: f"{v / total:.1%}" for k, v in stall_by_op.most_common()})
for simd, ts in sorted(mfma_by_simd.items()):
    # Window where two waves are in their K loops at once.
    s = sorted(spans[simd])
    lo, hi = None, None
    for a, b in zip(s, s[1:]):
        o_lo, o_hi = max(a[0], b[0]), min(a[1], b[1])
        if o_hi > o_lo and (hi is None or o_hi - o_lo > hi - lo):
            lo, hi = o_lo, o_hi
    if lo is None:
        print(simd, "no overlap window")
        continue
    busy, end = 0, lo
    for t in sorted(ts):
        a, b = max(t, end, lo), min(t + 16, hi)
        if b > a:
            busy += b - a
            end = b
    print(f"{simd}: {len(spans[simd])} waves, 2-wave window {hi - lo} ticks, MFMA pipe busy {busy / (hi - lo):.1%}")
