"""Interleaved kernel A/B of the small-M kernels with all_blocks False ("small") and True ("split"), and the BM256
kernels on big blocks only ("bm256big"), on synthetic sorted blocks: --blocks blocks of --rows routes (rows > 128:
big blocks), expert = block % 32, plus --extra capacity blocks past num_valid (the sort's unused capacity: grid
workgroups that exit at once). MiMo-V2.6-Flash EP8 rank shapes (I 2048, H 4096, top-8). Rounds as in
4wave/tools/ab_kernels.py: each starts after 3 s of an idle GPU and is redone if another process ran meanwhile.

usage: HIP_VISIBLE_DEVICES=2 python ab_split.py --rows 128 --blocks 32 --extra 0 128 --impls small split
"""
import argparse
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "../../../../../tests/contrib/moe"))
from gpu_contention import ExternalLoadMonitor  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--rows", type=int, nargs="+", default=[128])
parser.add_argument("--blocks", type=int, default=32)
parser.add_argument("--extra", type=int, nargs="+", default=[0])
parser.add_argument("--which", nargs="+", default=["gateup", "down"])
parser.add_argument("--impls", nargs="+", default=["small", "split"])
parser.add_argument("--rounds", type=int, default=6)
parser.add_argument("--calls", type=int, default=10)
parser.add_argument("--max-vram-pct", type=float, default=20.0)
args = parser.parse_args()
mon = ExternalLoadMonitor(int(os.environ.get("HIP_VISIBLE_DEVICES", "0").split(",")[0]))

import torch  # noqa: E402

torch.zeros(1, device="cuda")
mon.claim()
from aiter import dtypes  # noqa: E402

import pyhip.contrib.flydsl.moe_small_m_a8w8_blockscale_gfx942 as small  # noqa: E402
from pyhip.contrib.flydsl.moe_down_a8w8_blockscale_gfx942_lds import moe_down_a8w8_blockscale_gfx942_lds  # noqa: E402
from pyhip.contrib.flydsl.moe_gateup_a8w8_blockscale_gfx942_lds import moe_gateup_a8w8_blockscale_gfx942_lds  # noqa: E402

I, H, TOPK, E, T = 2048, 4096, 8, 32, 8192
torch.manual_seed(0)
a = (torch.randn(T, H, device="cuda") * 0.5).to(dtypes.fp8)
a_scale_t = torch.rand(H // 128, T, device="cuda") + 0.5
w1 = (torch.randn(E, 2 * I, H, device="cuda") * 0.5).to(dtypes.fp8)
w1_scale = torch.rand(E, 2 * I // 128, H // 128, device="cuda") * 0.01
w2 = (torch.randn(E, H, I, device="cuda") * 0.5).to(dtypes.fp8)
w2_scale = torch.rand(E, H // 128, I // 128, device="cuda") * 0.01
out = torch.zeros(T, H, device="cuda", dtype=torch.bfloat16)


def timed(fn):
    torch.cuda._sleep(10_000_000)
    for _ in range(3):
        fn()
    events = []
    for _ in range(args.calls):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        events.append((start, end))
    torch.cuda.synchronize()
    return statistics.median(s.elapsed_time(e) * 1e3 for s, e in events)


def gpu_busy():
    vram = mon.amdsmi.amdsmi_get_gpu_vram_usage(mon.handle)
    return (mon.amdsmi.amdsmi_get_gpu_activity(mon.handle)["gfx_activity"] > 2 or mon.foreign_occupancy() > 0
            or vram["vram_used"] > args.max_vram_pct / 100 * vram["vram_total"])


def wait_idle(quiet_s=3.0, timeout_s=7200):
    start, quiet_since = time.time(), None
    while time.time() - start < timeout_s:
        if not gpu_busy():
            quiet_since = quiet_since or time.time()
            if time.time() - quiet_since >= quiet_s:
                return
        else:
            quiet_since = None
        time.sleep(0.1)


def busy_after():
    time.sleep(0.3)
    return any(gpu_busy() or time.sleep(0.1) for _ in range(3))


for rows in args.rows:
    for extra in args.extra:
        b = args.blocks
        capacity = (b + extra) * 256
        sorted_ids = torch.full((capacity,), T | (TOPK << 24), device="cuda", dtype=torch.int32)
        for block in range(b):
            sorted_ids[block * 256 : block * 256 + rows] = torch.randperm(T, device="cuda")[:rows].int()
        sorted_ids[b * 256 :] = torch.randint(0, 1 << 30, (extra * 256,), device="cuda", dtype=torch.int32)
        expert_ids = (torch.arange(b + extra, device="cuda") % E).int()
        num_valid = torch.tensor([b * 256, T], device="cuda", dtype=torch.int32)
        weights = torch.rand(capacity, device="cuda")
        a2 = torch.zeros((capacity, I), device="cuda", dtype=dtypes.fp8)
        a2_scale = torch.ones((I // 128, capacity), device="cuda")
        for which in args.which:
            fns = {}
            for name in args.impls:
                if which == "gateup" and name in ("small", "split"):
                    fns[name] = lambda all_blocks=name == "split": small.moe_gateup_small_m(
                        a, w1, a_scale_t, w1_scale, sorted_ids, expert_ids, num_valid, a2, a2_scale, TOPK, all_blocks=all_blocks)
                elif name in ("small", "split"):
                    fns[name] = lambda all_blocks=name == "split": small.moe_down_small_m(
                        a2, w2, a2_scale, w2_scale, sorted_ids, weights, expert_ids, num_valid, out, TOPK, all_blocks=all_blocks)
                elif which == "gateup":
                    fns[name] = lambda: moe_gateup_a8w8_blockscale_gfx942_lds(
                        a, w1, a_scale_t, w1_scale, sorted_ids, expert_ids, num_valid, a2, layout="2x4_pf1", out_scale_t=a2_scale,
                        topk=TOPK, big_blocks_only=True)
                else:
                    fns[name] = lambda: moe_down_a8w8_blockscale_gfx942_lds(
                        a2, w2, a2_scale, w2_scale, sorted_ids, weights, expert_ids, num_valid, out, layout="4x2_pf1",
                        reduce="atomic", zero_out=False, topk=TOPK, big_blocks_only=True)
                fns[name]()
            results = {name: [] for name in fns}
            dirty = 0
            for r in range(args.rounds):
                order = list(fns) if r % 2 == 0 else list(fns)[::-1]
                for attempt in range(30):
                    wait_idle()
                    with mon:
                        values = {name: timed(fns[name]) for name in order}
                    clean = mon.clean and not busy_after()
                    if clean:
                        break
                dirty += not clean
                for name in fns:
                    results[name].append(values[name])
            med = {name: statistics.median(v) for name, v in results.items()}
            base = med[args.impls[0]]
            print(f"rows={rows:3d} blocks={b} extra={extra:3d} {which:6s} "
                  + "  ".join(f"{n}={med[n]:7.1f}us ({base / med[n]:.3f}x)" for n in fns)
                  + (f"  [{dirty} rounds with foreign kernels]" if dirty else ""), flush=True)
