"""Tables of the option B runs (run_split.sh JSON): path medians (us, with sort) and speedups vs the tuned asm (the
fastest asm kernel of the run, sort included), and the single-kernel medians of the --kernels runs.

usage: python summarize_split.py [names...]   (default: every run of run_split.sh present in ..)
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.normpath(os.path.join(HERE, ".."))
names = sys.argv[1:] or ["final_ep_uniform", "final_random", "burst_ep_uniform", "burst_random", "final_large_random",
                         "final_large_ep_uniform"]
PATHS = ("bm256", "small", "split")
KERNELS = ("k: gateup_small", "k: gateup256_big", "k: gateup_split", "k: down_small", "k: down256_big", "k: down_split")
for name in names:
    path = os.path.join(D, name + ".json")
    if not os.path.exists(path):
        continue
    report = json.load(open(path))
    gates = "PASS" if report.get("valid") else "FAIL"
    print(f"\n## {name} (gates {gates})\n")
    print("| T | routes/expert | tuned asm | " + " | ".join(PATHS) + " | " + " | ".join(f"{p} / asm" for p in PATHS) + " |")
    print("|" + "---|" * (3 + 2 * len(PATHS)))
    for run in report["runs"]:
        med = run["median_us"]
        tuned = run["tuned_asm"]
        per = run["routes_per_expert"]
        cells = [f"{med[p]:.1f}" if p in med else "" for p in PATHS]
        ratios = [f"{med[tuned] / med[p]:.3f}x" if p in med else "" for p in PATHS]
        print(f"| {run['tokens']} | {min(per)}-{max(per)} | {med[tuned]:.1f} ({tuned[4:]}) | " + " | ".join(cells) + " | "
              + " | ".join(ratios) + " |")
    if any(k in report["runs"][0]["median_us"] for k in KERNELS):
        print("\n| T | " + " | ".join(k[3:] for k in KERNELS) + " | small+big GateUp | small+big Down |")
        print("|" + "---|" * (3 + len(KERNELS)))
        for run in report["runs"]:
            med = run["median_us"]
            cells = [f"{med[k]:.1f}" if k in med else "" for k in KERNELS]
            both = [med.get(f"k: {w}_small", 0) + med.get(f"k: {w}256_big", 0) for w in ("gateup", "down")]
            print(f"| {run['tokens']} | " + " | ".join(cells) + f" | {both[0]:.1f} | {both[1]:.1f} |")
