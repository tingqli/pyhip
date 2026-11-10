"""Summary tables of the 4wave_128 benchmark JSONs (bench_ab.py output): per T the tuned asm (fastest asm path incl.
sort) and the medians of small (current path), old8 (8-wave), c64 (4-wave with 64-column GateUp tiles) and bm256
when present, as speedups over the tuned asm; then the single-kernel medians.

usage: python summarize128.py ../../4wave_128/final_ep_uniform.json [...]
"""
import json
import sys

PATHS = ["small", "old8", "c64", "bm256"]
KERNELS = ["k: sort256Z", "k: sort128Z", "k: gateup_small", "k: down_small", "k: gateup_old8", "k: down_old8",
           "k: gateup256_big", "k: down256_big"]

for path in sys.argv[1:]:
    report = json.load(open(path))
    gates = report["gates"]
    print(f"== {path}: gates {'PASS' if all(g['pass'] for g in gates) else 'FAIL'} "
          f"(max activity {max(max(g['activity']) for g in gates)} %, max VRAM {max(g['vram_used_mb'] for g in gates)} MB)")
    print(f"{'T':>5} {'tuned asm':>24s} " + " ".join(f"{p:>16s}" for p in PATHS))
    for run in report["runs"]:
        med, tuned = run["median_us"], run["tuned_asm"]
        cells = [f"{med[p]:7.1f} ({med[tuned] / med[p]:.3f}x)" if p in med else f"{'-':>16s}" for p in PATHS]
        print(f"{run['tokens']:5d} {tuned[4:] + ' ' + format(med[tuned], '.1f'):>24s} " + " ".join(f"{c:>16s}" for c in cells))
    kernel_rows = [run for run in report["runs"] if any(k in run["median_us"] for k in KERNELS)]
    if kernel_rows:
        print("  kernels (median us): " + "  ".join(k[3:] for k in KERNELS))
        for run in kernel_rows:
            print(f"  T={run['tokens']:<5d} " + "  ".join(f"{run['median_us'].get(k, float('nan')):7.1f}" for k in KERNELS))
