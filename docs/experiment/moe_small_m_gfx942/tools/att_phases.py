"""Per-K-block phases of every wave in an ATT dispatch (one barrier per K block): release->first MFMA, MFMA span,
last MFMA->barrier arrival, barrier wait. Usage: att_phases.py <ui_output dir>."""
import json
import statistics
import sys
from pathlib import Path

d = Path(sys.argv[1])
asm = {int(r[2]): str(r[0]).strip() for r in json.loads((d / "code.json").read_text())["code"]}
stats = {"to_first_mfma": [], "mfma_span": [], "tail_to_barrier": [], "barrier_wait": [], "period": [], "vm_wait": [], "issue_stall": []}
for path in sorted(d.glob("se*_sm*_sl*_wv*.json")):
    insts = json.loads(path.read_text())["wave"]["instructions"]
    if len(insts) < 1000:
        continue
    bars = [i for i, r in enumerate(insts) if asm.get(r[4], "").startswith("s_barrier")]
    # Skip the prologue barrier and the epilogue ones: K-loop barriers are the long regular run.
    for a, b in zip(bars[1:-4], bars[2:-3]):
        release = insts[a][0] + insts[a][2] + insts[a][3]
        mfmas = [r for r in insts[a + 1 : b] if asm.get(r[4], "").startswith("v_mfma")]
        if not mfmas:
            continue
        first = mfmas[0][0] + mfmas[0][2]
        last = mfmas[-1][0] + mfmas[-1][2]
        arrive = insts[b][0]
        stats["to_first_mfma"].append(first - release)
        stats["mfma_span"].append(last - first)
        stats["tail_to_barrier"].append(arrive - last)
        stats["barrier_wait"].append(insts[b][2])
        stats["period"].append(insts[b][0] + insts[b][2] - (insts[a][0] + insts[a][2]))
        stats["vm_wait"].append(sum(r[2] for r in insts[a + 1 : b] if asm.get(r[4], "").startswith("s_waitcnt") and "vmcnt" in asm.get(r[4], "")))
        stats["issue_stall"].append(sum(r[2] for r in insts[a + 1 : b] if asm.get(r[4], "").startswith("buffer_load")))
for k, v in stats.items():
    if v:
        q = statistics.quantiles(v, n=10)
        print(f"{k:16s} n={len(v):5d} median {statistics.median(v):7.0f}  p10 {q[0]:7.0f}  p90 {q[-1]:7.0f}  mean {statistics.mean(v):7.0f}")
