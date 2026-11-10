"""Bitwise equivalence and run-to-run determinism of the current small-M kernels against the lane-scale variant
(variants/moe_small_m_4wave_lane_scales.py: same arithmetic, per-wave A-scale loads), on aiter moe_sorting(256) of
test_fmoe_a8w8_blockscale_asm_gfx942.make_inputs. This is how the LDS race of the single-tile K loop showed up
(GateUp differing in 8 of 20 calls) before the explicit lgkmcnt(0) wait ahead of the first barrier.

usage: HIP_VISIBLE_DEVICES=1 python check_equal.py [tokens] [routing]
"""
import importlib
import importlib.util
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "../../../../../tests/contrib/moe"))
import test_fmoe_a8w8_blockscale_asm_gfx942 as asm_t  # noqa: E402
from aiter import dtypes  # noqa: E402
from aiter.fused_moe import moe_sorting  # noqa: E402

spec = importlib.util.spec_from_file_location("pyhip.contrib.flydsl._lane", os.path.join(HERE, "../variants/moe_small_m_4wave_lane_scales.py"))
lane = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = lane
spec.loader.exec_module(lane)
cur = importlib.import_module("pyhip.contrib.flydsl.moe_small_m_a8w8_blockscale_gfx942")

tokens = int(sys.argv[1]) if len(sys.argv) > 1 else 256
routing = sys.argv[2] if len(sys.argv) > 2 else "ep_uniform"
I, H, TOPK, E, EP = 2048, 4096, 8, 32, 8
inputs = asm_t.make_inputs(tokens, I, H, TOPK, E, EP, routing)
out = torch.zeros((tokens, H), device="cuda", dtype=torch.bfloat16)
sorted_ids, sorted_weights, expert_ids, num_valid_ids, _ = moe_sorting(
    inputs["topk_ids"], inputs["topk_weights"], E * EP, H, dtypes.bf16, 256, inputs["expert_mask"], output=out
)
blocks = min(expert_ids.numel(), -(-tokens * TOPK // 256) + E)
expert_ids = expert_ids[:blocks]
capacity = blocks * 256
a2_ref = torch.zeros((capacity, I), device="cuda", dtype=dtypes.fp8)
scale_ref = torch.zeros((capacity, I // 64), device="cuda")
a2 = torch.zeros_like(a2_ref)
scale = torch.zeros((I // 64, capacity), device="cuda")
args = (inputs["a"], inputs["w1"], inputs["a_scale_t"], inputs["w1_scale"], sorted_ids, expert_ids, num_valid_ids)
lane.moe_gateup_small_m(*args, a2_ref, scale_ref, TOPK)
bad = 0
for call in range(20):
    a2.zero_()
    scale.zero_()
    cur.moe_gateup_small_m(*args, a2, scale, TOPK)
    torch.cuda.synchronize()
    if not (torch.equal(a2_ref.view(torch.uint8), a2.view(torch.uint8)) and torch.equal(scale_ref, scale.t())):
        bad += 1
        rows = (a2_ref.view(torch.uint8) != a2.view(torch.uint8)).any(1).nonzero().squeeze(1)
        print(f"GateUp call {call}: {rows.numel()} rows differ (first {rows[:8].tolist()})")
print(f"GateUp: {bad} of 20 calls differ from the lane-scale variant")

reference = None
bad = 0
for impl, a2_in, scale_in in [(lane, a2_ref, scale_ref)] + [(cur, a2_ref, scale_ref.t().contiguous())] * 10:
    result = torch.zeros((tokens, H), device="cuda", dtype=torch.bfloat16)
    impl.moe_down_small_m(a2_in, inputs["w2"], scale_in, inputs["w2_scale"], sorted_ids, sorted_weights, expert_ids,
                          num_valid_ids, result, TOPK)
    torch.cuda.synchronize()
    if reference is None:
        reference = result
        continue
    worst = (result.float() - reference.float()).abs().max().item()
    bad += worst > 0
    if worst > 0:
        print(f"Down: max |difference| {worst:.4g} (BF16 atomics may reorder sums of several routes per token)")
print(f"Down: {bad} of 10 calls differ from the lane-scale variant")
