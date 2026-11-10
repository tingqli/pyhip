"""Summary tables of the 4-wave benchmark JSONs (bench_ab.py output): per T the tuned asm (fastest asm path incl.
sort), small (current path), old8 (previous 8-wave path), their ratios to the tuned asm, and the kernel-level
medians when present. Also the per-round medians of small / old8 (sustained runs switch between power states).

usage: python summarize4.py ../final_ep_uniform.json [...]
"""
import json
import statistics
import sys

KERNELS = ["k: sort256Z", "k: sort128Z", "k: gateup_small", "k: down_small", "k: gateup_old8", "k: down_old8",
           "k: gateup256_big", "k: down256_big"]

for path in sys.argv[1:]:
    report = json.load(open(path))
    gates = report["gates"]
    print(f"== {path}: gates {'PASS' if all(g['pass'] for g in gates) else 'FAIL'} "
          f"(max activity {max(max(g['activity']) for g in gates)} %, max VRAM {max(g['vram_used_mb'] for g in gates)} MB)")
    rounds = report["args"]["rounds"]
    print(f"{'T':>5} {'routing':10s} {'tuned asm':>22s} {'small':>8s} {'old8':>8s} {'bm256':>8s} {'small/asm':>9s} {'old8/asm':>8s}"
          f" {'small/old8':>10s}  per-round small | old8")
    for run in report["runs"]:
        med = run["median_us"]
        tuned = run["tuned_asm"]
        cells = [med.get(n) for n in ("small", "old8", "bm256")]
        per_round = {}
        for name in ("small", "old8"):
            samples = run["samples_us"].get(name)
            if samples:
                n = len(samples) // rounds
                per_round[name] = " ".join(f"{statistics.median(samples[r * n:(r + 1) * n]):.0f}" for r in range(rounds))
        ratio = lambda name: f"{med[tuned] / med[name]:.3f}" if name in med else "-"  # noqa: E731
        print(f"{run['tokens']:5d} {run['routing']:10s} {tuned[4:] + ' ' + format(med[tuned], '.1f'):>22s} "
              + " ".join(f"{c:8.1f}" if c else f"{'-':>8s}" for c in cells)
              + f" {ratio('small'):>9s} {ratio('old8'):>8s} "
              + (f"{med['old8'] / med['small']:10.3f}" if "old8" in med and "small" in med else f"{'-':>10s}")
              + f"  {per_round.get('small', '-')} | {per_round.get('old8', '-')}")
    kernel_rows = [run for run in report["runs"] if any(k in run["median_us"] for k in KERNELS)]
    if kernel_rows:
        print("  kernels (median us): " + "  ".join(k[3:] for k in KERNELS))
        for run in kernel_rows:
            print(f"  T={run['tokens']:<5d} " + "  ".join(f"{run['median_us'].get(k, float('nan')):7.1f}" for k in KERNELS))
