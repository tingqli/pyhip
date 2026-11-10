"""HBM bandwidth of strided multi-stream reads, 1 MiB per workgroup (one 512-thread workgroup per CU per round).

Usage: python stream_probe_gfx942.py [--json out.json]; compare the 16 x 2 KiB pattern of the small-M kernels with
fewer / longer streams (docs/experiment/moe_small_m_gfx942/README.md).
"""
import argparse
import json
import os
import statistics

import pyhip
import torch

CONFIGS = [(16, 128), (16, 256), (8, 128), (8, 256), (8, 512), (4, 512), (1, 2048)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=int, nargs="+", default=[304, 512, 608])
    parser.add_argument("--json")
    args = parser.parse_args()
    hip = pyhip.module(os.path.join(os.path.dirname(os.path.abspath(__file__)), "stream_probe_gfx942.cpp"))
    region = 1 << 20
    buf = torch.ones(max(args.tasks) * region // 4, device="cuda", dtype=torch.int32)
    sink = torch.zeros(1024, device="cuda", dtype=torch.int32)
    report = {}
    for tasks in args.tasks:
        for streams, chunk in CONFIGS:
            kernel = getattr(hip, f"streams_{streams}_{chunk}")
            steps = region // 16 // (streams * chunk)
            fn = lambda: kernel([tasks], [512], buf, steps, sink)  # noqa: E731
            for _ in range(5):
                fn()
            times = []
            for _ in range(30):
                s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                s.record()
                fn()
                e.record()
                e.synchronize()
                times.append(s.elapsed_time(e) * 1e3)
            med = statistics.median(times)
            key = f"tasks={tasks} streams={streams} chunk={chunk * 16 // 1024}KiB"
            report[key] = dict(median_us=med, tbps=tasks * region / med / 1e6)
            print(f"{key:40s} {med:7.1f} us  {tasks * region / med / 1e6:5.2f} TB/s", flush=True)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
