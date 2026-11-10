"""Interleaved kernel A/B of small-M implementations on synthetic sorted blocks (--blocks blocks of --rows routes,
expert = block % 32, MiMo-V2.6-Flash EP8 rank shapes: I 2048, H 4096, top-8).

--impls: cur (pyhip.contrib.flydsl.moe_small_m_a8w8_blockscale_gfx942), old8 (variants/moe_small_m_8wave_sort128.py:
8 waves, 128-row sort blocks, per-128 A2 scales [I/128, cap]), lane (variants/moe_small_m_4wave_lane_scales.py: 4
waves, per-wave A-scale loads, A2 scales [cap, I/64]), c64 (variants/moe_small_m_4wave_64col.py: 64-column GateUp
tiles, A2 [I/64, cap]), plain128 (variants/moe_small_m_4wave_128col_plain.py), down128
(variants/moe_small_m_4wave_down128.py: the current kernels with a 128-column Down tile) or a path to a module file with the
cur API (A2 [I/A2_SCALE_COLUMNS, cap]); "name@ring" sets the B ring. Same weights every call (no rotation): absolute
times are optimistic, the ratios are the point. Each round starts after 3 s of an idle GPU (GFX activity <= 2 %, no
CU occupancy of other processes, VRAM <= --max-vram-pct) and is redone (up to 30 times) if another process ran kernels during
it or the GPU is busy right after it; rounds still contaminated are counted.

usage: HIP_VISIBLE_DEVICES=1 python ab_kernels.py --impls c64 down128 cur old8 --rows 16 32 64 96 112 128
"""
import argparse
import importlib
import importlib.util
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "../../../../../tests/contrib/moe"))
from gpu_contention import ExternalLoadMonitor  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--rows", type=int, nargs="+", default=[16, 32, 64, 128])
parser.add_argument("--blocks", type=int, default=32)
parser.add_argument("--which", nargs="+", default=["gateup", "down"])
parser.add_argument("--rounds", type=int, default=6)
parser.add_argument("--calls", type=int, default=10)
parser.add_argument("--ring", type=int, default=2)
parser.add_argument("--impls", nargs="+", default=["old8", "lane", "cur"])
parser.add_argument("--max-vram-pct", type=float, default=20.0, help="VRAM use an idle GPU may show (other jobs' buffers)")
args = parser.parse_args()
mon = ExternalLoadMonitor(int(os.environ.get("HIP_VISIBLE_DEVICES", "0").split(",")[0]))

import torch  # noqa: E402

torch.zeros(1, device="cuda")
mon.claim()
from aiter import dtypes  # noqa: E402

FILES = {
    "old8": os.path.join(HERE, "../variants/moe_small_m_8wave_sort128.py"),
    "lane": os.path.join(HERE, "../variants/moe_small_m_4wave_lane_scales.py"),
    "c64": os.path.join(HERE, "../variants/moe_small_m_4wave_64col.py"),
    "plain128": os.path.join(HERE, "../variants/moe_small_m_4wave_128col_plain.py"),
    "down128": os.path.join(HERE, "../variants/moe_small_m_4wave_down128.py"),
}
LAYOUT = {"old8": "t128", "lane": "row", "cur": "col"}
mods, ring_of, layout_of = {}, {}, {}
for name in args.impls:
    base = name.split("@")[0]
    ring_of[name] = int(name.split("@")[1]) if "@" in name else args.ring
    layout_of[name] = LAYOUT.get(base, "col")
    if base == "cur":
        mods[name] = importlib.import_module("pyhip.contrib.flydsl.moe_small_m_a8w8_blockscale_gfx942")
        continue
    spec = importlib.util.spec_from_file_location(f"pyhip.contrib.flydsl._ab_{len(mods)}", FILES.get(base, base))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    mods[name] = mod

I, H, TOPK, E, T = 2048, 4096, 8, 32, 4096
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
    """GFX activity above 2 %, another process's CU occupancy, or VRAM above --max-vram-pct (another job's buffers)
    while this process is idle. KFD does not report the CU occupancy of every process, hence the activity counter too."""
    vram = mon.amdsmi.amdsmi_get_gpu_vram_usage(mon.handle)
    return (mon.amdsmi.amdsmi_get_gpu_activity(mon.handle)["gfx_activity"] > 2 or mon.foreign_occupancy() > 0
            or vram["vram_used"] > args.max_vram_pct / 100 * vram["vram_total"])


def wait_idle(quiet_s=3.0, timeout_s=7200):
    """Until the GPU has been idle (gpu_busy false) for quiet_s."""
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
    """Activity once this process's kernels have drained: another job started during the round."""
    time.sleep(0.3)
    return any(gpu_busy() or time.sleep(0.1) for _ in range(3))


b = args.blocks
for rows in args.rows:
    tokens = [torch.randperm(T, device="cuda")[:rows].int() for _ in range(b)]
    weights = torch.rand(b * 256, device="cuda")
    expert_ids = (torch.arange(b, device="cuda") % E).int()
    bufs = {}
    for block_m in (128, 256):
        sorted_ids = torch.full((b * block_m,), T | (TOPK << 24), device="cuda", dtype=torch.int32)
        for block in range(b):
            sorted_ids[block * block_m : block * block_m + rows] = tokens[block]
        num_valid = torch.tensor([b * block_m], device="cuda", dtype=torch.int32)
        a2 = torch.zeros((b * block_m, I), device="cuda", dtype=dtypes.fp8)
        bufs[block_m] = (sorted_ids, num_valid, a2)
    scales = {"t128": torch.ones((I // 128, b * 128), device="cuda"), "row": torch.ones((b * 256, I // 64), device="cuda"),
              "col": None}
    for which in args.which:
        fns = {}
        for name, mod in mods.items():
            sid, nv, a2 = bufs[128 if name.startswith("old8") else 256]
            sc, ring = scales[layout_of[name]], ring_of[name]
            if sc is None:
                sc = torch.ones((I // getattr(mod, "A2_SCALE_COLUMNS", 64), b * 256), device="cuda")
            if which == "gateup":
                fns[name] = lambda mod=mod, sid=sid, nv=nv, a2=a2, sc=sc, ring=ring: mod.moe_gateup_small_m(
                    a, w1, a_scale_t, w1_scale, sid, expert_ids, nv, a2, sc, TOPK, ring)
            elif name.startswith("old8"):
                fns[name] = lambda mod=mod, sid=sid, nv=nv, a2=a2, sc=sc, ring=ring: mod.moe_down_small_m(
                    a2, w2, sc, w2_scale, sid, weights, expert_ids, nv, out, TOPK, None, ring)
            else:
                fns[name] = lambda mod=mod, sid=sid, nv=nv, a2=a2, sc=sc, ring=ring: mod.moe_down_small_m(
                    a2, w2, sc, w2_scale, sid, weights, expert_ids, nv, out, TOPK, ring)
            fns[name]()
        results = {name: [] for name in mods}
        dirty = 0
        for r in range(args.rounds):
            order = list(mods) if r % 2 == 0 else list(mods)[::-1]
            for attempt in range(30):
                wait_idle()
                with mon:
                    values = {name: timed(fns[name]) for name in order}
                clean = mon.clean and not busy_after()
                if clean:
                    break
            dirty += not clean
            for name in mods:
                results[name].append(values[name])
        med = {name: statistics.median(v) for name, v in results.items()}
        base = med[list(mods)[0]]
        print(f"rows={rows:3d} {which:6s} " + "  ".join(f"{n}={med[n]:7.1f}us ({base / med[n]:.3f}x)" for n in mods)
              + (f"  [{dirty} rounds with foreign kernels]" if dirty else ""), flush=True)
