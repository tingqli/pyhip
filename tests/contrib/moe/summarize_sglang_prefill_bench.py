"""Summarize sglang bench_serving prefill logs: one row per input length, optional side-by-side comparison.

usage: python summarize_sglang_prefill_bench.py NAME=DIR [NAME=DIR ...]  (DIR: LOG_DIR of run_benchmark_mimo_pro_prefill.sh)
"""

import re
import sys
from pathlib import Path

FIELDS = {
    "requests": r"Successful requests:\s+(\d+)",
    "duration_s": r"Benchmark duration \(s\):\s+([\d.]+)",
    "input_tok_s": r"Input token throughput \(tok/s\):\s+([\d.]+)",
    "mean_ttft_ms": r"Mean TTFT \(ms\):\s+([\d.]+)",
    "median_ttft_ms": r"Median TTFT \(ms\):\s+([\d.]+)",
    "p99_ttft_ms": r"P99 TTFT \(ms\):\s+([\d.]+)",
}


def load(directory):
    rows = {}
    for log in Path(directory).glob("benchmark_*_con*.log"):
        isl, con = re.match(r"benchmark_(\d+)_con(\d+)", log.name).groups()
        text = log.read_text(errors="replace")
        row = {"concurrency": int(con)}
        for name, pattern in FIELDS.items():
            found = re.findall(pattern, text)
            row[name] = float(found[-1]) if found else None
        if row["input_tok_s"] is not None:
            rows[int(isl)] = row
    return rows


runs = [arg.split("=", 1) for arg in sys.argv[1:]]
data = {name: load(directory) for name, directory in runs}
lengths = sorted(set().union(*(d.keys() for d in data.values())))
for name, rows in data.items():
    print(f"\n### {name}\n")
    print("| ISL | C | requests | input tok/s | mean TTFT ms | median TTFT ms | P99 TTFT ms |")
    print("|---:|---:|---:|---:|---:|---:|---:|")
    for isl in lengths:
        r = rows.get(isl)
        if r:
            print(f"| {isl} | {r['concurrency']} | {r['requests']:.0f} | {r['input_tok_s']:.0f} | {r['mean_ttft_ms']:.0f} "
                  f"| {r['median_ttft_ms']:.0f} | {r['p99_ttft_ms']:.0f} |")
if len(data) > 1:
    base_name, base = next(iter(data.items()))
    print(f"\n### input tok/s vs {base_name}\n")
    print("| ISL | " + " | ".join(data) + " | " + " | ".join(f"{n}/{base_name}" for n in list(data)[1:]) + " |")
    print("|---:|" + "---:|" * (2 * len(data) - 1))
    for isl in lengths:
        values = [data[n].get(isl, {}).get("input_tok_s") for n in data]
        ratios = [f"{v / values[0]:.3f}" if v and values[0] else "-" for v in values[1:]]
        print(f"| {isl} | " + " | ".join(f"{v:.0f}" if v else "-" for v in values) + " | " + " | ".join(ratios) + " |")
