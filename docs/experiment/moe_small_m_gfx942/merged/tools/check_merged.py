"""Option A (merged GateUp / Down launches, variants/moe_small_m_8wave_pair.py) on aiter moe_sorting(256) of
test_fmoe_a8w8_blockscale_asm_gfx942.make_inputs (MiMo-V2.6-Flash EP8 rank):
  - GateUp: the merged launch must write the same FP8 rows and scales as the two-launch path (4-wave small-M GateUp +
    BM256 GateUp with big_blocks_only), bitwise (pair tasks equal the 4-wave tasks; BM256 tasks are the same code);
  - fused path (sort zeroing out, merged GateUp, merged Down) against the asm test's reference (asm_t.check).

usage: HIP_VISIBLE_DEVICES=2 python check_merged.py [tokens routing] ...
"""
import importlib.util
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "../../../../../tests/contrib/moe"))
import test_fmoe_a8w8_blockscale_asm_gfx942 as asm_t  # noqa: E402
from aiter import dtypes  # noqa: E402
from aiter.fused_moe import moe_sorting  # noqa: E402

import pyhip  # noqa: E402
from pyhip.contrib.flydsl.moe_a8w8_blockscale_gfx942 import max_sorted_blocks  # noqa: E402
from pyhip.contrib.flydsl.moe_gateup_a8w8_blockscale_gfx942_lds import moe_gateup_a8w8_blockscale_gfx942_lds  # noqa: E402
from pyhip.contrib.flydsl.moe_small_m_a8w8_blockscale_gfx942 import moe_gateup_small_m  # noqa: E402

spec = importlib.util.spec_from_file_location("pyhip.contrib.flydsl._pair", os.path.join(HERE, "../variants/moe_small_m_8wave_pair.py"))
pair = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = pair
spec.loader.exec_module(pair)

cases = sys.argv[1:] or ["64", "random", "256", "ep_uniform", "1000", "random", "4096", "random", "6144", "random",
                         "8192", "ep_uniform", "8192", "random"]
I, H, TOPK, E, EP = 2048, 4096, 8, 32, 8
failures = 0
for tokens, routing in zip(map(int, cases[::2]), cases[1::2]):
    inputs = asm_t.make_inputs(tokens, I, H, TOPK, E, EP, routing)
    expected = asm_t.reference(inputs)
    out = torch.empty((tokens, H), device="cuda", dtype=torch.bfloat16)
    sorted_ids, sorted_weights, expert_ids, num_valid_ids, _ = moe_sorting(
        inputs["topk_ids"], inputs["topk_weights"], E * EP, H, dtypes.bf16, 256, inputs["expert_mask"], output=out
    )
    blocks = min(expert_ids.numel(), max_sorted_blocks(tokens, TOPK, E))
    eid = expert_ids[:blocks]
    capacity = blocks * 256
    args = (inputs["a"], inputs["w1"], inputs["a_scale_t"], inputs["w1_scale"], sorted_ids, eid, num_valid_ids)
    a2_ref = torch.full((capacity, I), 0x55, device="cuda", dtype=torch.uint8).view(dtypes.fp8)
    scale_ref = torch.full((I // 128, capacity), -1.0, device="cuda")
    moe_gateup_small_m(*args, a2_ref, scale_ref, TOPK)
    if tokens > 128:
        moe_gateup_a8w8_blockscale_gfx942_lds(*args, a2_ref, layout="2x4_pf1", out_scale_t=scale_ref, topk=TOPK, big_blocks_only=True)
    a2 = torch.empty_like(a2_ref)
    scale = torch.empty_like(scale_ref)
    bad = 0
    for call in range(5):
        a2.view(torch.uint8).fill_(0x55)
        scale.fill_(-1.0)
        pair.moe_gateup_merged(*args, a2, scale, TOPK)
        torch.cuda.synchronize()
        if not (torch.equal(a2_ref.view(torch.uint8), a2.view(torch.uint8)) and torch.equal(scale_ref, scale)):
            bad += 1
            rows = (a2_ref.view(torch.uint8) != a2.view(torch.uint8)).any(1).nonzero().squeeze(1)
            srows = (scale_ref != scale).any(0).nonzero().squeeze(1)
            print(f"  GateUp call {call}: {rows.numel()} rows, {srows.numel()} scale rows differ (first {rows[:6].tolist()} / {srows[:6].tolist()})")
    print(f"T={tokens} {routing}: merged GateUp {bad} of 5 calls differ from the two-launch GateUp")
    failures += bad
    for call in range(2):
        out.fill_(float("nan"))
        sorted_ids, sorted_weights, expert_ids, num_valid_ids, _ = moe_sorting(
            inputs["topk_ids"], inputs["topk_weights"], E * EP, H, dtypes.bf16, 256, inputs["expert_mask"], output=out
        )
        pair.fused_moe_merged(
            inputs["a"], inputs["a_scale_t"], inputs["w1"], inputs["w1_scale"], inputs["w2"], inputs["w2_scale"], sorted_ids,
            sorted_weights, expert_ids, num_valid_ids, out, TOPK,
        )
        torch.cuda.synchronize()
        try:
            asm_t.check(inputs, out, expected)
            result = "pass"
        except AssertionError as error:
            result = f"FAIL {error}"
            failures += 1
        routed = torch.bincount(inputs["routes"][0], minlength=tokens) > 0
        print(f"T={tokens} {routing}: fused merged call {call}: {result}, calc_diff {pyhip.calc_diff(out[routed].float(), expected[routed]):.2e}")
print("PASS" if failures == 0 else f"FAIL ({failures})")
