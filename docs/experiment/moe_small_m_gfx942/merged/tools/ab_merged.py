"""Interleaved kernel A/B of the merged (option A) GateUp / Down launches against the BM256 kernels on synthetic sort
blocks (--blocks blocks of --rows routes, expert = block % 32, --extra capacity blocks past num_valid), MiMo-V2.6-Flash
EP8 rank shapes. Impls: bm256 (BM256 kernels, big_blocks_only), merged (variants/moe_small_m_8wave_pair.py),
direct (the merged kernels with the BM256 task mapping instead of the task scan: valid only when every block has a
route at row 128, i.e. --rows > 128; isolates the cost of the scan), nobody (direct with an empty pair-task arm: the
BM256 arm alone in the merged kernel structure), exact (nobody launched with --blocks * 16 workgroups, i.e. without the
trailing workgroups of the spare sort capacity), small (4-wave small-M kernels, rows <= 128).
Rounds as in 4wave/tools/ab_kernels.py (3 s idle before each, redone if another process ran).

usage: HIP_VISIBLE_DEVICES=2 python ab_merged.py --rows 256 --blocks 128 --extra 928 --impls bm256 merged direct
"""
import argparse
import importlib.util
import os
import statistics
import sys
import time
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "../../../../../tests/contrib/moe"))
from gpu_contention import ExternalLoadMonitor  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--rows", type=int, nargs="+", default=[256])
parser.add_argument("--blocks", type=int, default=128)
parser.add_argument("--extra", type=int, default=0)
parser.add_argument("--a2-pad", type=int, default=0, help="allocate A2 with this many spare 256-row blocks past the capacity")
parser.add_argument("--nb-override", type=int, default=0,
                    help="num_blocks passed to the direct / nobody / exact launches (A2-scale K stride = 256 * this)")
parser.add_argument("--which", nargs="+", default=["gateup", "down"])
parser.add_argument("--impls", nargs="+", default=["bm256", "merged", "direct"])
parser.add_argument("--rounds", type=int, default=6)
parser.add_argument("--calls", type=int, default=10)
parser.add_argument("--max-vram-pct", type=float, default=20.0)
args = parser.parse_args()
mon = ExternalLoadMonitor(int(os.environ.get("HIP_VISIBLE_DEVICES", "0").split(",")[0]))

import flydsl.expr as fx  # noqa: E402
import torch  # noqa: E402

torch.zeros(1, device="cuda")
mon.claim()
from aiter import dtypes  # noqa: E402

import pyhip.contrib.flydsl.moe_small_m_a8w8_blockscale_gfx942 as small  # noqa: E402
from pyhip.contrib.flydsl.helpers import run_compiled  # noqa: E402
from pyhip.contrib.flydsl.moe_down_a8w8_blockscale_gfx942_lds import (  # noqa: E402
    compile_moe_down_a8w8_blockscale_gfx942_lds, moe_down_a8w8_blockscale_gfx942_lds,
)
from pyhip.contrib.flydsl.moe_gateup_a8w8_blockscale_gfx942_lds import (  # noqa: E402
    compile_moe_gateup_a8w8_blockscale_gfx942_lds, moe_gateup_a8w8_blockscale_gfx942_lds,
)

spec = importlib.util.spec_from_file_location("pyhip.contrib.flydsl._pair", os.path.join(HERE, "../variants/moe_small_m_8wave_pair.py"))
pair = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = pair
spec.loader.exec_module(pair)

I, H, TOPK, E, T = 2048, 4096, 8, 32, 32768
# FlyDSL cache key of the plugins below (see pair.MergedPlugin).
KEY = pair.source_key(os.path.abspath(__file__))


def direct_plugin(mode, n_dim, k_dim):
    """merged_plugin with the BM256 task mapping (every block big)."""
    base = pair.merged_plugin(mode, n_dim, k_dim, TOPK, E)

    def task(ctx):
        num_tasks = ctx.valid_blocks * ctx.big_tiles
        chunk = (num_tasks + 7) >> 3
        slot = ctx.pid >> 3
        index = (ctx.pid & 7) * chunk + slot
        block = index // ctx.big_tiles
        return SimpleNamespace(active=(slot < chunk) & (index < num_tasks), big=slot >= 0, block=block,
                               tile=index - block * ctx.big_tiles)

    return pair.MergedPlugin(task, base.body, base.grid, f"direct:{mode}:{KEY}")


DIRECT = {"gateup": direct_plugin("gateup", I, H), "down": direct_plugin("down", H, I)}
# The direct mapping with no pair task body at all (an empty else arm): is the BM256 arm itself slower when merged?
NOBODY = {mode: pair.MergedPlugin(p.task, lambda ctx, block, tile: None, p.grid, f"nobody:{mode}:{KEY}")
          for mode, p in DIRECT.items()}
# nobody with the grid cut to the valid tasks (--blocks * 16, host-known only in this test): no trailing empty workgroups.
EXACT = {mode: pair.MergedPlugin(p.task, lambda ctx, block, tile: None, lambda num_blocks: args.blocks * 16,
                                 f"exact{args.blocks}:{mode}:{KEY}")
         for mode, p in DIRECT.items()}


def ptr(dtype, tensor):
    import flydsl.compiler as flyc
    return flyc.from_c_void_p(dtype, tensor.data_ptr())


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
    b = args.blocks
    capacity = (b + args.extra) * 256
    sorted_ids = torch.full((capacity,), T | (TOPK << 24), device="cuda", dtype=torch.int32)
    for block in range(b):
        sorted_ids[block * 256 : block * 256 + rows] = torch.randperm(T, device="cuda")[:rows].int()
    expert_ids = (torch.arange(b + args.extra, device="cuda") % E).int()
    num_valid = torch.tensor([b * 256, T], device="cuda", dtype=torch.int32)
    weights = torch.rand(capacity, device="cuda")
    a2 = torch.zeros((capacity + args.a2_pad * 256, I), device="cuda", dtype=dtypes.fp8)[:capacity]
    a2_scale = torch.ones((I // 128, capacity), device="cuda")
    for which in args.which:
        fns = {}
        for name in args.impls:
            if which == "gateup" and name == "bm256":
                fns[name] = lambda: moe_gateup_a8w8_blockscale_gfx942_lds(
                    a, w1, a_scale_t, w1_scale, sorted_ids, expert_ids, num_valid, a2, layout="2x4_pf1", out_scale_t=a2_scale,
                    topk=TOPK, big_blocks_only=True)
            elif name == "bm256":
                fns[name] = lambda: moe_down_a8w8_blockscale_gfx942_lds(
                    a2, w2, a2_scale, w2_scale, sorted_ids, weights, expert_ids, num_valid, out, layout="4x2_pf1",
                    reduce="atomic", zero_out=False, topk=TOPK, big_blocks_only=True)
            elif name == "small" and which == "gateup":
                fns[name] = lambda: small.moe_gateup_small_m(a, w1, a_scale_t, w1_scale, sorted_ids, expert_ids, num_valid, a2, a2_scale, TOPK)
            elif name == "small":
                fns[name] = lambda: small.moe_down_small_m(a2, w2, a2_scale, w2_scale, sorted_ids, weights, expert_ids, num_valid, out, TOPK)
            elif name == "merged" and which == "gateup":
                fns[name] = lambda: pair.moe_gateup_merged(a, w1, a_scale_t, w1_scale, sorted_ids, expert_ids, num_valid, a2, a2_scale, TOPK)
            elif name == "merged":
                fns[name] = lambda: pair.moe_down_merged(a2, w2, a2_scale, w2_scale, sorted_ids, weights, expert_ids, num_valid, out, TOPK)
            elif which == "gateup":
                plugin = {"direct": DIRECT, "nobody": NOBODY, "exact": EXACT}[name]["gateup"]
                launch = compile_moe_gateup_a8w8_blockscale_gfx942_lds(I, H, TOPK, E, "2x4_pf1", True, False, True, False, plugin)
                fns[name] = lambda launch=launch: run_compiled(
                    launch, ptr(fx.Int32, a), ptr(fx.Int32, w1), ptr(fx.Int32, a2), ptr(fx.Float32, a2_scale), ptr(fx.Float32, a_scale_t),
                    ptr(fx.Float32, w1_scale), ptr(fx.Int32, sorted_ids), ptr(fx.Int32, expert_ids), ptr(fx.Int32, num_valid),
                    T, args.nb_override or b + args.extra, torch.cuda.current_stream())
            else:
                plugin = {"direct": DIRECT, "nobody": NOBODY, "exact": EXACT}[name]["down"]
                launch = compile_moe_down_a8w8_blockscale_gfx942_lds(H, I, TOPK, E, "4x2_pf1", True, True, True, False, False, plugin)
                fns[name] = lambda launch=launch: run_compiled(
                    launch, ptr(fx.Int32, a2), ptr(fx.Int32, w2), ptr(fx.Int32, out), ptr(fx.Float32, a2_scale), ptr(fx.Float32, w2_scale),
                    ptr(fx.Int32, sorted_ids), ptr(fx.Float32, weights), ptr(fx.Int32, expert_ids), ptr(fx.Int32, num_valid),
                    ptr(fx.Int32, sorted_ids), ptr(fx.Int32, sorted_ids), T, args.nb_override or b + args.extra, torch.cuda.current_stream())
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
        print(f"rows={rows:3d} blocks={b} extra={args.extra}" + (f" a2_pad={args.a2_pad}" if args.a2_pad else "")
              + (f" nb_override={args.nb_override}" if args.nb_override else "") + f" {which:6s} "
              + "  ".join(f"{n}={med[n]:7.1f}us ({base / med[n]:.3f}x)" for n in fns)
              + (f"  [{dirty} rounds with foreign kernels]" if dirty else ""), flush=True)
