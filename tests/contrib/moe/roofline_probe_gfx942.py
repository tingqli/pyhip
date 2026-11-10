"""Roofline ceilings of this MI300X for the small-M MoE study (docs/record/moe_small_m_gfx942.md).

  HBM:  sustained read bandwidth of a 1 GiB buffer (> 256 MB MALL), several grid sizes / loads in flight.
  MFMA: v_mfma_f32_16x16x32_fp8_fp8 from registers on every SIMD, burst (a few ms) and sustained (power capped).
  GEMM: torch._scaled_mm FP8 (hipBLASLt) M=N=K=8192, warm.

usage: HIP_VISIBLE_DEVICES=0 python roofline_probe_gfx942.py [--json out.json]
"""

import argparse
import json
import os
import statistics
import time

import torch

import pyhip


def timed(fn, calls, warmup=3):
    """Median and min us of `calls` back-to-back launches, each with its own event pair."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    torch.cuda._sleep(50_000_000)
    events = []
    for _ in range(calls):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        events.append((start, end))
    torch.cuda.synchronize()
    us = sorted(s.elapsed_time(e) * 1e3 for s, e in events)
    return statistics.median(us), us[0], us


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--buffer-mib", type=int, default=1024)
    parser.add_argument("--sustained-s", type=float, default=2.0)
    parser.add_argument("--json")
    args = parser.parse_args()
    hip = pyhip.module(os.path.join(os.path.dirname(os.path.abspath(__file__)), "roofline_probe_gfx942.cpp"))
    props = torch.cuda.get_device_properties(0)
    cus = props.multi_processor_count
    report = dict(gpu=props.name, arch=props.gcnArchName, cus=cus, hbm={}, mfma={}, gemm={})

    nbytes = args.buffer_mib << 20
    buf = torch.ones(nbytes // 4, device="cuda", dtype=torch.int32)
    sink = torch.zeros(1024, device="cuda", dtype=torch.int32)
    vecs = nbytes // 16
    for name, block in (("read_u4", 256), ("read_u8", 256), ("read512_u8", 512)):
        kernel = getattr(hip, name)
        for per_cu in (1, 2, 4, 8):
            grid = cus * per_cu
            per_wg = vecs // grid
            med, best, _ = timed(lambda: kernel([grid], [block], buf, per_wg, sink), 30)
            moved = per_wg * grid * 16
            key = f"{name} grid={grid}"
            report["hbm"][key] = dict(median_us=med, best_us=best, median_tbps=moved / med / 1e6, best_tbps=moved / best / 1e6)
            print(f"HBM {key:28s} median {med:8.1f} us {moved / med / 1e6:5.2f} TB/s  best {moved / best / 1e6:5.2f} TB/s", flush=True)
    del buf

    flop_per_wave_iter = 8 * 16 * 16 * 32 * 2
    out = torch.zeros(256, device="cuda")
    for per_cu, iters in ((4, 4000), (8, 2000)):
        grid = cus * per_cu
        flops = grid * 4 * iters * flop_per_wave_iter
        med, best, _ = timed(lambda: hip.mfma_fp8([grid], [256], iters, 0x3C3C3C3C3C3C3C3C, 0x3B3B3B3B3B3B3B3B, out), 20)
        report["mfma"][f"burst waves/SIMD={per_cu}"] = dict(median_us=med, median_tflops=flops / med / 1e6, best_tflops=flops / best / 1e6)
        print(f"MFMA burst {per_cu} waves/SIMD: median {med:8.1f} us {flops / med / 1e6:7.1f} TF  best {flops / best / 1e6:7.1f} TF", flush=True)
    # Sustained: back-to-back launches for several seconds; the power cap lowers the clock.
    grid, iters = cus * 4, 4000
    flops = grid * 4 * iters * flop_per_wave_iter
    count, t0 = 0, time.time()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    while time.time() - t0 < args.sustained_s:
        for _ in range(20):
            hip.mfma_fp8([grid], [256], iters, 0x3C3C3C3C3C3C3C3C, 0x3B3B3B3B3B3B3B3B, out)
            count += 1
        torch.cuda.synchronize()
    end.record()
    end.synchronize()
    tf = flops * count / (start.elapsed_time(end) * 1e-3) / 1e12
    report["mfma"]["sustained"] = dict(seconds=args.sustained_s, tflops=tf)
    print(f"MFMA sustained {args.sustained_s:.0f} s: {tf:7.1f} TF", flush=True)

    fp8 = torch.float8_e4m3fnuz
    for m in (8192, 4096):
        a = (torch.randn(m, 8192, device="cuda") * 0.3).to(fp8)
        b = (torch.randn(8192, 8192, device="cuda") * 0.3).to(fp8)
        one = torch.ones((), device="cuda")
        flops = 2 * m * 8192 * 8192
        med, best, _ = timed(lambda: torch._scaled_mm(a, b.t(), scale_a=one, scale_b=one, out_dtype=torch.bfloat16), 20)
        report["gemm"][f"scaled_mm M={m} N=K=8192"] = dict(median_us=med, median_tflops=flops / med / 1e6, best_tflops=flops / best / 1e6)
        print(f"GEMM _scaled_mm M={m} N=K=8192: median {med:8.1f} us {flops / med / 1e6:7.1f} TF  best {flops / best / 1e6:7.1f} TF", flush=True)
        del a, b

    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
