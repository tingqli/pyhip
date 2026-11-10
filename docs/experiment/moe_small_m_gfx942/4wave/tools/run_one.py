"""One small-M kernel (8-wave variant on 128-row sort blocks, or the current 4-wave kernel on 256-row blocks) for
rocprofv3 counter collection or ATT. 32 synthetic blocks of --rows routes, as tools/ab_kernels.py.

usage (ATT of one CU, then the per-K-block phases):
  rocprofv3 --att --att-library-path <decoder dir> --att-target-cu 1 --att-shader-engine-mask 0x1 \
      --att-simd-select 0xf --att-buffer-size 0x10000000 --kernel-include-regex small_m -d att_out -o att \
      -- python run_one.py --impl cur --rows 128 --calls 1
  python att_phases2.py att_out/ui_output_*  32
"""
import argparse
import importlib
import importlib.util
import os
import sys

import torch
from aiter import dtypes

HERE = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser()
parser.add_argument("--impl", default="cur", choices=["old8", "lane", "c64", "down128", "cur"])
parser.add_argument("--which", default="gateup", choices=["gateup", "down"])
parser.add_argument("--rows", type=int, default=128)
parser.add_argument("--blocks", type=int, default=32)
parser.add_argument("--calls", type=int, default=3)
parser.add_argument("--ring", type=int, default=2)
args = parser.parse_args()

files = {"old8": "moe_small_m_8wave_sort128.py", "lane": "moe_small_m_4wave_lane_scales.py", "c64": "moe_small_m_4wave_64col.py",
         "down128": "moe_small_m_4wave_down128.py"}
if args.impl == "cur":
    mod = importlib.import_module("pyhip.contrib.flydsl.moe_small_m_a8w8_blockscale_gfx942")
else:
    spec = importlib.util.spec_from_file_location(f"pyhip.contrib.flydsl._{args.impl}", os.path.join(HERE, "../variants", files[args.impl]))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
block_m = 128 if args.impl == "old8" else 256

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
sorted_ids = torch.full((b * block_m,), T | (TOPK << 24), device="cuda", dtype=torch.int32)
for block in range(b):
    sorted_ids[block * block_m : block * block_m + args.rows] = torch.randperm(T, device="cuda")[: args.rows].int()
weights = torch.rand(b * block_m, device="cuda")
expert_ids = (torch.arange(b, device="cuda") % E).int()
num_valid = torch.tensor([b * block_m], device="cuda", dtype=torch.int32)
a2 = torch.zeros((b * block_m, I), device="cuda", dtype=dtypes.fp8)
a2_scale = {"old8": torch.ones((I // 128, b * block_m), device="cuda"), "lane": torch.ones((b * block_m, I // 64), device="cuda")}.get(
    args.impl, torch.ones((I // getattr(mod, "A2_SCALE_COLUMNS", 64), b * block_m), device="cuda"))
for _ in range(args.calls):
    if args.which == "gateup":
        mod.moe_gateup_small_m(a, w1, a_scale_t, w1_scale, sorted_ids, expert_ids, num_valid, a2, a2_scale, TOPK, args.ring)
    elif args.impl == "old8":
        mod.moe_down_small_m(a2, w2, a2_scale, w2_scale, sorted_ids, weights, expert_ids, num_valid, out, TOPK, None, args.ring)
    else:
        mod.moe_down_small_m(a2, w2, a2_scale, w2_scale, sorted_ids, weights, expert_ids, num_valid, out, TOPK, args.ring)
torch.cuda.synchronize()
