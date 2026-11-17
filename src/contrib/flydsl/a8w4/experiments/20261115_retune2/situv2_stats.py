"""NOTE: uses lookup_config / config_name, removed in refactor D (now
``ops.moe.pick / name_of``); run this at commit 5d0b861 or earlier.

SiTUv2 vs SiLU statistics of configs/a8w4_moe_tuned.csv for README 6.3 (20261115 table).

    cd src/contrib/flydsl && python a8w4/experiments/20261115_retune2/situv2_stats.py
"""
import csv
import os
import statistics
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from a8w4 import moe  # noqa: E402

CSV = os.path.join(os.path.dirname(moe.__file__), "configs", "a8w4_moe_tuned.csv")
SHAPES = {"TP8": (4096, 256, 256, 256, 8), "EP8": (4096, 2048, 32, 256, 8)}

rows = {}
with open(CSV) as f:
    for r in csv.DictReader(f):
        key = tuple(int(r[k]) for k in ("model_dim", "inter_dim", "experts", "experts_global", "topk"))
        rows.setdefault((key, r["act"]), {}).setdefault(int(r["num_tokens"]), {})[r["config"]] = float(r["us"])

for name, shape in SHAPES.items():
    si, sv = rows[(shape, "silu")], rows[(shape, "situv2")]
    ratios, diff, gain = [], [], []
    for t in sorted(si):
        for c in si[t].keys() & sv[t].keys():
            ratios.append(sv[t][c] / si[t][c])
        w_si = min(si[t], key=si[t].get)
        w_sv = min(sv[t], key=sv[t].get)
        if w_si != w_sv:
            diff.append(t)
            gain.append((t, w_si, w_sv, 100 * (sv[t][w_si] / sv[t][w_sv] - 1)))
    ratios.sort()
    q = statistics.quantiles(ratios, n=20)
    print(f"{name}: situv2/silu same config: median {statistics.median(ratios):.3f}, "
          f"p5 {q[0]:.3f} p95 {q[-1]:.3f} (n={len(ratios)})")
    print(f"  points with different winner ({len(diff)}/{len(si)}): {diff}")
    for t, a, b, g in gain:
        print(f"    {t:6d}: silu {a:28s} situv2 {b:28s} situv2 gain {g:+.2f}%")
    for t in (4096,):
        a = moe.lookup_config(shape, t, act="silu", capturing=True)
        b = moe.lookup_config(shape, t, act="situv2", capturing=True)
        print(f"  capture {t}: silu {moe.config_name(a) if hasattr(moe, 'config_name') else a} | "
              f"situv2 {moe.config_name(b) if hasattr(moe, 'config_name') else b}")
