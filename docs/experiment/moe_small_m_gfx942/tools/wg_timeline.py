"""Per-workgroup timeline of the small-M kernels: an instrumented copy of moe_small_m_a8w8_blockscale_gfx942 records
s_memrealtime (100 MHz) at workgroup start / end with HW_ID and XCC_ID into the (unused) route-count pointer.
Ex.: python wg_timeline.py --rows 128 --blocks 19 (one S=8 task per CU) or --blocks 32 (1.68 rounds)."""
import argparse
import collections

import torch

parser = argparse.ArgumentParser()
parser.add_argument("--rows", type=int, default=128)
parser.add_argument("--blocks", type=int, default=19)
parser.add_argument("--which", default="gateup")
args = parser.parse_args()
from aiter import dtypes  # noqa: E402
import atexit  # noqa: E402
import importlib  # noqa: E402
import os  # noqa: E402

import pyhip.contrib.flydsl as package  # noqa: E402

PATCHES = [
    ("        tid = fx.thread_idx.x\n",
     "        tid = fx.thread_idx.x\n"
     "        t_start = llvm.inline_asm(T.i64, [], \"s_memrealtime $0\\ns_waitcnt lgkmcnt(0)\", \"=s\", has_side_effects=True)\n"),
    ("            dispatch(0)\n",
     "            dispatch(0)\n"
     "        t_end = llvm.inline_asm(T.i64, [], \"s_memrealtime $0\\ns_waitcnt lgkmcnt(0)\", \"=s\", has_side_effects=True)\n"
     "        hw_id = llvm.inline_asm(T.i32, [], \"s_getreg_b32 $0, hwreg(HW_REG_HW_ID)\", \"=s\", has_side_effects=True)\n"
     "        xcc_id = llvm.inline_asm(T.i32, [], \"s_getreg_b32 $0, hwreg(HW_REG_XCC_ID)\", \"=s\", has_side_effects=True)\n"
     "        record = Vec.from_elements([fx.Int32(fx.arith.trunci(T.i32, t_start)), fx.Int32(fx.arith.trunci(T.i32, t_end)),\n"
     "                                    fx.Int32(hw_id), fx.Int32(xcc_id)], fx.Int32)\n"
     "        rocdl.raw_ptr_buffer_store(record.ir_value(), _rsrc_of(pRouteCounts, 1 << 24),\n"
     "                                   (tid == 0).select(pid * 16, fx.Int32(OOB_OFFSET)).ir_value(), zero, aux=ir.IntegerAttr.get(T.i32, 0))\n"),
    ("a2_scale_t, topk, ring=4, b_aux=2\n):", "a2_scale_t, topk, ring=4, b_aux=2, timing=None\n):"),
    ("_ptr(fx.Int32, num_valid_ids), _ptr(fx.Int32, num_valid_ids), tokens,", "_ptr(fx.Int32, num_valid_ids), _ptr(fx.Int32, timing), tokens,"),
    ("route_counts=None, ring=4,\n    b_aux=2,\n):", "route_counts=None, ring=4,\n    b_aux=2, timing=None,\n):"),
    ("_ptr(fx.Int32, route_counts if single_store else num_valid_ids)", "_ptr(fx.Int32, timing)"),
]
source = open(os.path.join(package.__path__[0], "moe_small_m_a8w8_blockscale_gfx942.py")).read()
for old, new in PATCHES:
    assert source.count(old) == 1, old
    source = source.replace(old, new)
instrumented = os.path.join(package.__path__[0], "_wg_timeline_instrumented.py")
with open(instrumented, "w") as f:
    f.write(source)
atexit.register(os.remove, instrumented)
small = importlib.import_module("pyhip.contrib.flydsl._wg_timeline_instrumented")

I, H, TOPK, E, T = 2048, 4096, 8, 32, 4096
torch.manual_seed(0)
a = (torch.randn(T, H, device="cuda") * 0.5).to(dtypes.fp8)
a_scale_t = torch.rand(H // 128, T, device="cuda") + 0.5
w1 = (torch.randn(E, 2 * I, H, device="cuda") * 0.5).to(dtypes.fp8)
w1_scale = torch.rand(E, 2 * I // 128, H // 128, device="cuda") * 0.01
w2 = (torch.randn(E, H, I, device="cuda") * 0.5).to(dtypes.fp8)
w2_scale = torch.rand(E, H // 128, I // 128, device="cuda") * 0.01
out = torch.zeros(T, H, device="cuda", dtype=torch.bfloat16)
b = args.blocks
sid = torch.full((b * 128,), T | (TOPK << 24), device="cuda", dtype=torch.int32)
for blk in range(b):
    sid[blk * 128 : blk * 128 + args.rows] = torch.randperm(T, device="cuda")[: args.rows].int()
sw = torch.rand(b * 128, device="cuda")
eid = (torch.arange(b, device="cuda") % E).int()
nv = torch.tensor([b * 128], device="cuda", dtype=torch.int32)
a2 = torch.empty((b * 128, I), device="cuda", dtype=dtypes.fp8)
a2s = torch.empty((I // 128, b * 128), device="cuda")
tiles = I // 128 if args.which == "gateup" else H // 256
grid = (b * tiles + 7) // 8 * 8
timing = torch.zeros(grid * 4, device="cuda", dtype=torch.int32)
def call():
    if args.which == "gateup":
        small.moe_gateup_small_m(a, w1, a_scale_t, w1_scale, sid, eid, nv, a2, a2s, TOPK, timing=timing)
    else:
        small.moe_down_small_m(a2, w2, a2s, w2_scale, sid, sw, eid, nv, out, TOPK, None, timing=timing)
for _ in range(5):
    call()
torch.cuda.synchronize()
s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
s.record(); call(); e.record(); torch.cuda.synchronize()
rec = timing.view(grid, 4).cpu().long()
t0 = rec[:, 0].min()
start, end = (rec[:, 0] - t0) * 10, (rec[:, 1] - t0) * 10  # ns
hw, xcc = rec[:, 2], rec[:, 3]
cu = (hw >> 8) & 15
se = ((hw >> 13) & 3) * 2 + ((hw >> 12) & 1)
print(f"kernel event time {s.elapsed_time(e) * 1e3:.1f} us; WG span {(end.max() - start.min()) / 1e3:.1f} us; grid {grid}")
dur = (end - start) / 1e3
print(f"task duration us: min {dur.min():.1f} median {dur.median():.1f} max {dur.max():.1f}")
per_cu = collections.defaultdict(list)
for i in range(grid):
    per_cu[(int(xcc[i]), int(se[i]), int(cu[i]))].append((float(start[i]) / 1e3, float(end[i]) / 1e3))
counts = collections.Counter(len(v) for v in per_cu.values())
print(f"CUs used {len(per_cu)}; WGs per CU histogram {sorted(counts.items())}")
starts = sorted(float(x) / 1e3 for x in start)
print("start times us (percentiles 0,10,50,90,100):", [round(starts[int(p * (len(starts) - 1))], 1) for p in (0, .1, .5, .9, 1)])
ends = sorted(float(x) / 1e3 for x in end)
print("end times us (percentiles 0,10,50,90,100):", [round(ends[int(p * (len(ends) - 1))], 1) for p in (0, .1, .5, .9, 1)])
by_xcc = collections.defaultdict(list)
for i in range(grid):
    by_xcc[int(xcc[i])].append(float(end[i]) / 1e3)
print("per-XCC: WGs, last end us:", {k: (len(v), round(max(v), 1)) for k, v in sorted(by_xcc.items())})
