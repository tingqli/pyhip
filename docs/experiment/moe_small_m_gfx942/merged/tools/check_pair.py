"""Bitwise equivalence of the 8-wave pair variant (variants/moe_small_m_8wave_pair.py) with the current 4-wave small-M
kernels, and its run-to-run determinism, on aiter moe_sorting(256) of test_fmoe_a8w8_blockscale_asm_gfx942.make_inputs
(MiMo-V2.6-Flash EP8 rank). Both take the blocks with no route at row 128 and must write the same FP8 rows, scales
and (one route per token) Down sums; with several routes per token BF16 atomics may add in another order.

usage: HIP_VISIBLE_DEVICES=2 python check_pair.py [tokens routing] ...   (default: 256 ep_uniform 1024 random 4096 random)
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

spec = importlib.util.spec_from_file_location("pyhip.contrib.flydsl._pair", os.path.join(HERE, "../variants/moe_small_m_8wave_pair.py"))
pair = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = pair
spec.loader.exec_module(pair)
cur = importlib.import_module("pyhip.contrib.flydsl.moe_small_m_a8w8_blockscale_gfx942")

cases = sys.argv[1:] or ["256", "ep_uniform", "1024", "random", "4096", "random"]
I, H, TOPK, E, EP = 2048, 4096, 8, 32, 8
failures = 0
for tokens, routing in zip(map(int, cases[::2]), cases[1::2]):
    inputs = asm_t.make_inputs(tokens, I, H, TOPK, E, EP, routing)
    out = torch.zeros((tokens, H), device="cuda", dtype=torch.bfloat16)
    sorted_ids, sorted_weights, expert_ids, num_valid_ids, _ = moe_sorting(
        inputs["topk_ids"], inputs["topk_weights"], E * EP, H, dtypes.bf16, 256, inputs["expert_mask"], output=out
    )
    blocks = min(expert_ids.numel(), -(-tokens * TOPK // 256) + E)
    expert_ids = expert_ids[:blocks]
    capacity = blocks * 256
    args = (inputs["a"], inputs["w1"], inputs["a_scale_t"], inputs["w1_scale"], sorted_ids, expert_ids, num_valid_ids)
    a2_ref = torch.full((capacity, I), 0x55, device="cuda", dtype=torch.uint8).view(dtypes.fp8)
    scale_ref = torch.full((I // 128, capacity), -1.0, device="cuda")
    cur.moe_gateup_small_m(*args, a2_ref, scale_ref, TOPK)
    a2 = torch.empty_like(a2_ref)
    scale = torch.empty_like(scale_ref)
    bad = 0
    for call in range(10):
        a2.view(torch.uint8).fill_(0x55)
        scale.fill_(-1.0)
        pair.moe_gateup_small_m(*args, a2, scale, TOPK)
        torch.cuda.synchronize()
        if not (torch.equal(a2_ref.view(torch.uint8), a2.view(torch.uint8)) and torch.equal(scale_ref, scale)):
            bad += 1
            rows = (a2_ref.view(torch.uint8) != a2.view(torch.uint8)).any(1).nonzero().squeeze(1)
            print(f"  GateUp call {call}: {rows.numel()} rows differ (first {rows[:8].tolist()})")
    written = (scale_ref != -1.0).any(0).sum().item()
    print(f"T={tokens} {routing}: GateUp {bad} of 10 calls differ from the 4-wave kernel ({written} rows written)")
    failures += bad

    reference = torch.zeros((tokens, H), device="cuda", dtype=torch.bfloat16)
    down_args = (inputs["w2"], scale_ref, inputs["w2_scale"], sorted_ids, sorted_weights, expert_ids, num_valid_ids)
    cur.moe_down_small_m(a2_ref, *down_args, reference, TOPK)
    one_route = routing == "ep_uniform"
    bad = 0
    for call in range(10):
        result = torch.zeros_like(reference)
        pair.moe_down_small_m(a2_ref, *down_args, result, TOPK)
        torch.cuda.synchronize()
        worst = (result.float() - reference.float()).abs().max().item()
        scale_ = reference.float().abs().max().item()
        if worst > 0:
            bad += 1
            if call == 0:
                print(f"  Down: max |difference| {worst:.4g} (max |out| {scale_:.4g})")
    note = "must be 0" if one_route else "BF16 atomics may reorder several routes per token"
    print(f"T={tokens} {routing}: Down {bad} of 10 calls differ from the 4-wave kernel ({note})")
    failures += bad if one_route else 0
print("PASS" if failures == 0 else f"FAIL ({failures})")
