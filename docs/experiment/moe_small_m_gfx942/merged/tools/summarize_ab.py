"""Tables of run_ab.sh (bench_merged.py JSON): path medians (us, with sort) and speedups vs the tuned asm (the fastest
asm kernel of the run, sort included), and the GateUp / Down kernel medians of the --kernels runs: two-launch
("small" + "256_big"), option B ("split") and option A ("merged").

usage: python summarize_ab.py [names...]   (default: every run of run_ab.sh present in ..)
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.normpath(os.path.join(HERE, ".."))
names = sys.argv[1:] or ["final_ep_uniform", "final_random", "burst_ep_uniform", "burst_random", "final_large_random",
                         "final_large_ep_uniform"]
PATHS = ("bm256", "small", "split", "merged")
for name in names:
    path = os.path.join(D, name + ".json")
    if not os.path.exists(path):
        continue
    report = json.load(open(path))
    print(f"\n## {name} (gates {'PASS' if report.get('valid') else 'FAIL'})\n")
    print("| T | routes/expert | tuned asm | " + " | ".join(PATHS) + " | " + " | ".join(f"{p} / asm" for p in PATHS) + " |")
    print("|" + "---|" * (3 + 2 * len(PATHS)))
    for run in report["runs"]:
        med, tuned, per = run["median_us"], run["tuned_asm"], run["routes_per_expert"]
        cells = [f"{med[p]:.1f}" if p in med else "" for p in PATHS]
        ratios = [f"{med[tuned] / med[p]:.3f}x" if p in med else "" for p in PATHS]
        print(f"| {run['tokens']} | {min(per)}-{max(per)} | {med[tuned]:.1f} ({tuned[4:]}) | " + " | ".join(cells) + " | "
              + " | ".join(ratios) + " |")
    if "k: gateup_merged" in report["runs"][0]["median_us"]:
        print("\n| T | GateUp small+big | split | merged | Down small+big | split | merged |")
        print("|---|---|---|---|---|---|---|")
        for run in report["runs"]:
            med = run["median_us"]
            row = [run["tokens"]]
            for w in ("gateup", "down"):
                row += [f"{med[f'k: {w}_small'] + med.get(f'k: {w}256_big', 0):.1f}", f"{med.get(f'k: {w}_split', float('nan')):.1f}",
                        f"{med[f'k: {w}_merged']:.1f}"]
            print("| " + " | ".join(str(x) for x in row) + " |")
