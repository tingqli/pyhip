"""Synthetic sorted blocks (`--rows` routes each, expert = block % 32) for the small-M kernels: time and weight
bandwidth vs the number of valid blocks (tasks = blocks x N tiles; one 8-wave workgroup per CU, 304 CUs per round).
Ex.: python sweep_blocks.py --rows 32 --blocks 19 32 38 (--grid-blocks 96 adds empty workgroups like a real sort)."""
import argparse
import os
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../../tests/contrib/moe"))
from gpu_contention import ExternalLoadMonitor  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--blocks", type=int, nargs="+", default=[1, 4, 8, 16, 19, 24, 32, 38, 48, 57, 64, 76])
parser.add_argument("--rows", type=int, default=32)
parser.add_argument("--grid-blocks", type=int, default=0, help="expert_ids.numel() (0 = exactly the valid blocks)")
parser.add_argument("--which", nargs="+", default=["gateup", "down"])
parser.add_argument("--configs", nargs="+", default=["ring=4"])
parser.add_argument("--calls", type=int, default=30)
parser.add_argument("--experts", type=int, default=32)
parser.add_argument("--module", default="moe_small_m_a8w8_blockscale_gfx942")
parser.add_argument("--tokens", type=int, default=8192)
args = parser.parse_args()
mon = ExternalLoadMonitor(int(os.environ.get("HIP_VISIBLE_DEVICES", "0").split(",")[0]))

import torch  # noqa: E402

torch.zeros(1, device="cuda")
mon.claim()
from aiter import dtypes  # noqa: E402
import importlib  # noqa: E402

small = importlib.import_module(f"pyhip.contrib.flydsl.{args.module}")

I, H, TOPK, E = 2048, 4096, 8, args.experts
T = args.tokens
torch.manual_seed(0)
a = (torch.randn(T, H, device="cuda") * 0.5).to(dtypes.fp8)
a_scale_t = torch.rand(H // 128, T, device="cuda") + 0.5
w1 = (torch.randn(E, 2 * I, H, device="cuda") * 0.5).to(dtypes.fp8)
w1_scale = torch.rand(E, 2 * I // 128, H // 128, device="cuda") * 0.01
w2 = (torch.randn(E, H, I, device="cuda") * 0.5).to(dtypes.fp8)
w2_scale = torch.rand(E, H // 128, I // 128, device="cuda") * 0.01
out = torch.zeros(T, H, device="cuda", dtype=torch.bfloat16)


def parse(cfg):
    kw = {}
    for kv in cfg.split(","):
        k, v = kv.split("=")
        kw[k] = frozenset(v.split("+")) if k == "ablate" else int(v)
    return kw


def timed(fn):
    torch.cuda._sleep(20_000_000)
    for _ in range(5):
        fn()
    ev = []
    for _ in range(args.calls):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        ev.append((s, e))
    torch.cuda.synchronize()
    return statistics.median(s.elapsed_time(e) * 1e3 for s, e in ev)


for b in args.blocks:
    nb = max(b, args.grid_blocks)
    sid = torch.full((nb * 128,), T | (TOPK << 24), device="cuda", dtype=torch.int32)
    tokens = torch.randperm(T, device="cuda")[: args.rows].int()
    for blk in range(b):
        sid[blk * 128 : blk * 128 + args.rows] = (tokens + blk * 7) % T
    sw = torch.rand(nb * 128, device="cuda")
    eid = (torch.arange(nb, device="cuda") % E).int()
    nv = torch.tensor([b * 128], device="cuda", dtype=torch.int32)
    a2 = torch.empty((nb * 128, I), device="cuda", dtype=dtypes.fp8)
    a2s = torch.empty((I // 128, nb * 128), device="cuda")
    for name in args.which:
        for cfg in args.configs:
            kw = parse(cfg)
            if name == "gateup":
                fn = lambda: small.moe_gateup_small_m(a, w1, a_scale_t, w1_scale, sid, eid, nv, a2, a2s, TOPK, **kw)
                tiles, wbytes = I // 128, 1 << 20
            else:
                fn = lambda: small.moe_down_small_m(a2, w2, a2s, w2_scale, sid, sw, eid, nv, out, TOPK, None, **kw)
                tiles, wbytes = H // 256, 1 << 19
            fn()
            with mon:
                us = timed(fn)
            tasks = b * tiles
            print(
                f"{name:6s} {cfg:28s} blocks {b:3d} (grid {nb:3d}) tasks {tasks:5d} per-XCD {-(-tasks // 8):4d} "
                f"rounds {-(-tasks // 8) / 38:5.2f}  {us:7.1f} us  {tasks * wbytes / us / 1e6:5.2f} TB/s"
                + ("" if mon.clean else "  CONTAMINATED"),
                flush=True,
            )
