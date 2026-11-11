"""Correctness of the merged FP8 block-scale MoE path on gfx942 (fused_moe_a8w8_blockscale_gfx942_mixed / _sort): one
GateUp and one Down launch over a 256-row sort table, BM256 tiles for the blocks with a route at row 128 and 8-wave
pair tasks for the others."""

import argparse

import pytest
import torch

import pyhip
from aiter import dtypes
from aiter.ops.shuffle import shuffle_weight
from test_gateup_a8w8_blockscale_gfx942 import is_gfx942, make_routes

BLOCK_M = 256
SMALL_ROWS = 128
SCALE_COLUMNS = 128


def sort_routes(token, slot, expert, tokens, topk, experts, block_m=BLOCK_M, extra_blocks=0):
    """aiter moe_sorting(block_m) layout: per expert its routes by token, padded to block_m; extra_blocks of garbage
    capacity must never be read."""
    order = torch.argsort(expert * tokens + token)
    token, slot, expert = token[order], slot[order], expert[order]
    counts = torch.bincount(expert, minlength=experts).tolist()
    pad_id = (topk << 24) | tokens
    ids, block_experts, start = [], [], 0
    for e, count in enumerate(counts):
        if count == 0:
            continue
        blocks = (count + block_m - 1) // block_m
        padded = torch.full((blocks * block_m,), pad_id, dtype=torch.int64)
        padded[:count] = (slot[start : start + count] << 24) | token[start : start + count]
        ids.append(padded)
        block_experts += [e] * blocks
        start += count
    num_valid = sum(len(x) for x in ids)
    ids.append(torch.randint(0, 1 << 31, (extra_blocks * block_m,)))
    block_experts += [1 << 30] * extra_blocks
    sorted_ids = torch.cat(ids).to(torch.int32)
    expert_ids = torch.tensor(block_experts, dtype=torch.int32)
    num_valid_ids = torch.tensor([num_valid, tokens], dtype=torch.int32)
    return sorted_ids.cuda(), expert_ids.cuda(), num_valid_ids.cuda()


def block_quant(experts, n, k, std):
    fp8, fp8_max = dtypes.fp8, torch.finfo(dtypes.fp8).max
    scale = torch.empty((experts, n // 128, k // 128), device="cuda")
    weight = torch.empty((experts, n, k), device="cuda", dtype=fp8)
    for e in range(experts):
        source = torch.randn((n // 128, 128, k // 128, 128), device="cuda") * std
        scale[e] = source.abs().amax(dim=(1, 3)).clamp_min(1e-8) / fp8_max
        weight[e] = (source / scale[e][:, None, :, None]).to(fp8).view(n, k)
    # Distinct FP32 scales per block.
    return weight, scale * (0.5 + torch.rand_like(scale))


def make_inputs(tokens, inter_dim, hidden_dim, topk, experts, routing, seed=0, extra_blocks=0):
    generator = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    fp8, fp8_max = dtypes.fp8, torch.finfo(dtypes.fp8).max
    a_source = torch.randn((tokens, hidden_dim // 128, 128), device="cuda") * 0.33
    a_scale = a_source.abs().amax(dim=-1).clamp_min(1e-8) / fp8_max
    a = (a_source / a_scale[..., None]).to(fp8).view(tokens, hidden_dim)
    del a_source
    a_scale *= 0.5 + torch.rand_like(a_scale)
    w1, w1_scale = block_quant(experts, 2 * inter_dim, hidden_dim, 0.2)
    w2, w2_scale = block_quant(experts, hidden_dim, inter_dim, 0.02)
    token, slot, expert = make_routes(tokens, topk, experts, routing, generator)
    sorted_ids, expert_ids, num_valid_ids = sort_routes(token, slot, expert, tokens, topk, experts, extra_blocks=extra_blocks)
    topk_weights = torch.rand((tokens, topk), generator=generator) + 0.1
    topk_weights = (topk_weights / topk_weights.sum(dim=1, keepdim=True)).cuda()
    ids = sorted_ids.long()
    sorted_token, sorted_slot = ids & 0xFFFFFF, ids >> 24 & 0xFF
    routed = (sorted_token < tokens) & (sorted_slot < topk)
    sorted_weights = torch.where(routed, topk_weights[sorted_token.clamp(max=tokens - 1), sorted_slot.clamp(max=topk - 1)], 0.0)
    return {
        "a": a,
        "a_scale_t": a_scale.t().contiguous(),
        "w1_raw": w1,
        "w1": shuffle_weight(w1, layout=(16, 16)),
        "w1_scale": w1_scale,
        "w2_raw": w2,
        "w2": shuffle_weight(w2, layout=(16, 16)),
        "w2_scale": w2_scale,
        "topk_weights": topk_weights,
        "routes": (token.cuda(), slot.cuda(), expert.cuda()),
        "sorted_ids": sorted_ids,
        "sorted_weights": sorted_weights.float().contiguous(),
        "expert_ids": expert_ids,
        "num_valid_ids": num_valid_ids,
        "shape": (tokens, inter_dim, hidden_dim, topk, experts),
    }


def dequant_blocks(weight, scale):
    n, k = weight.shape
    return (weight.float().view(n // 128, 128, k // 128, 128) * scale[:, None, :, None]).view(n, k)


def valid_positions(inputs):
    """Mask of the routed sorted positions in [0, num_valid)."""
    tokens, _, _, topk, _ = inputs["shape"]
    num_valid = int(inputs["num_valid_ids"][0])
    ids = inputs["sorted_ids"][:num_valid].long()
    return ((ids & 0xFFFFFF) < tokens) & ((ids >> 24 & 0xFF) < topk)


def big_rows(inputs):
    """Mask of the positions in [0, num_valid) of the blocks with a route at row 128 (the BM256 tasks)."""
    return valid_positions(inputs).view(-1, BLOCK_M)[:, SMALL_ROWS].repeat_interleave(BLOCK_M)


def gateup_reference(inputs):
    """FP32 SiLU(gate) * up at sorted rows [0, num_valid) (zeros at padding) and its per-128-column amax / 240 scales."""
    tokens, inter_dim, hidden_dim, topk, experts = inputs["shape"]
    kb = hidden_dim // 128
    a = (inputs["a"].float().view(tokens, kb, 128) * inputs["a_scale_t"].t()[:, :, None]).view(tokens, hidden_dim)
    num_valid = int(inputs["num_valid_ids"][0])
    ids = inputs["sorted_ids"][:num_valid].long()
    token, slot = ids & 0xFFFFFF, ids >> 24 & 0xFF
    valid = (token < tokens) & (slot < topk)
    rows = torch.zeros((num_valid, inter_dim), device="cuda")
    block_expert = inputs["expert_ids"].long().repeat_interleave(BLOCK_M)[:num_valid]
    for e in range(experts):
        positions = (valid & (block_expert == e)).nonzero().squeeze(1)
        if positions.numel() == 0:
            continue
        gate, up = (a[token[positions]] @ dequant_blocks(inputs["w1_raw"][e], inputs["w1_scale"][e]).t()).split(inter_dim, dim=-1)
        rows[positions] = torch.nn.functional.silu(gate) * up
    scale = rows.view(num_valid, -1, SCALE_COLUMNS).abs().amax(-1) / torch.finfo(dtypes.fp8).max
    return rows, scale


def check_gateup(inputs, a2, a2_scale, sentinel):
    """Every routed row of a2 against the reference; a2_scale [I/128, capacity] as the kernel writes it. The capacity
    past the valid blocks is left alone."""
    rows, scale = gateup_reference(inputs)
    num_valid = rows.shape[0]
    valid = valid_positions(inputs)
    row_scale = a2_scale.t()
    kernel_scale = row_scale[:num_valid][valid]
    torch.testing.assert_close(kernel_scale, scale[valid], rtol=1e-4, atol=0)
    got = a2[:num_valid][valid]
    dequant = (got.float().view(-1, scale.shape[1], SCALE_COLUMNS) * kernel_scale[:, :, None]).view(got.shape[0], -1)
    diff = pyhip.calc_diff(dequant, rows[valid])
    assert diff < 1e-3, f"calc_diff {diff}"
    # Requantized with the kernel's scales, only FP32 rounding differences may flip an FP8 rounding.
    fp8_max = torch.finfo(dtypes.fp8).max
    requant = (rows[valid].view(got.shape[0], -1, SCALE_COLUMNS) / kernel_scale[:, :, None].clamp_min(1e-30))
    requant = requant.clamp(-fp8_max, fp8_max).to(dtypes.fp8).view(got.shape)
    mismatch = (requant.view(torch.uint8) != got.view(torch.uint8)).float().mean().item()
    assert mismatch < 0.01, f"FP8 mismatch {mismatch:.4f}"
    assert (a2[num_valid:].view(torch.uint8) == sentinel).all()
    assert row_scale[num_valid:].isnan().all()
    return diff


def down_reference(inputs, a2, a2_scale):
    """test_down_a8w8_blockscale_gfx942.reference over the routes of the valid blocks, A dequantized per 128 columns:
    BF16 per route (weight applied), each token's routes summed in FP32; the routed mask and sum of |route|."""
    tokens, inter_dim, hidden_dim, topk, experts = inputs["shape"]
    positions = valid_positions(inputs).nonzero().squeeze(1)
    ids = inputs["sorted_ids"][positions].long()
    token, slot = ids & 0xFFFFFF, ids >> 24 & 0xFF
    expert = inputs["expert_ids"].long().repeat_interleave(BLOCK_M)[positions]
    rows = a2.view(torch.uint8)[positions].view(dtypes.fp8).float().view(-1, inter_dim // SCALE_COLUMNS, SCALE_COLUMNS)
    a_dequant = (rows * a2_scale.t()[positions][:, :, None]).view(-1, inter_dim)
    routed_y = torch.empty((positions.numel(), hidden_dim), device="cuda", dtype=torch.bfloat16)
    for e in range(experts):
        index = (expert == e).nonzero().squeeze(1)
        if index.numel() == 0:
            continue
        weight = inputs["topk_weights"][token[index], slot[index]]
        w_dequant = dequant_blocks(inputs["w2_raw"][e], inputs["w2_scale"][e])
        routed_y[index] = ((a_dequant[index] @ w_dequant.t()) * weight[:, None]).to(torch.bfloat16)
    out = torch.zeros((tokens, hidden_dim), device="cuda")
    magnitude = torch.zeros((tokens, hidden_dim), device="cuda")
    for s in range(topk):
        in_slot = slot == s
        out.index_add_(0, token[in_slot], routed_y[in_slot].float())
        magnitude.index_add_(0, token[in_slot], routed_y[in_slot].float().abs())
    routed = torch.zeros(tokens, device="cuda", dtype=torch.bool)
    routed[token] = True
    return out.to(torch.bfloat16), routed, magnitude


@pytest.mark.parametrize(
    "tokens,routing", [(64, "random"), (256, "ep_uniform"), (1000, "random"), (4096, "random"), (6144, "random"), (8192, "ep_uniform")]
)
def test_fused_mixed_aiter_sort(tokens, routing):
    """fused_moe_a8w8_blockscale_gfx942_mixed on aiter's EP moe_sorting(256) of a MiMo-V2.6-Flash EP8 rank (sort zeroes
    out): small blocks only, mixed (T=4096/6144 random) and big blocks only (T=8192), against the asm test's
    reference."""
    import test_fmoe_a8w8_blockscale_asm_gfx942 as asm_t
    from aiter.fused_moe import moe_sorting
    from pyhip.contrib.flydsl.moe_a8w8_blockscale_gfx942 import fused_moe_a8w8_blockscale_gfx942_mixed

    if not is_gfx942():
        pytest.skip("gfx942 kernel")
    inter_dim, hidden_dim, topk, experts, ep = 2048, 4096, 8, 32, 8
    inputs = asm_t.make_inputs(tokens, inter_dim, hidden_dim, topk, experts, ep, routing)
    expected = asm_t.reference(inputs)
    out = torch.empty((tokens, hidden_dim), device="cuda", dtype=torch.bfloat16)
    for _ in range(2):
        out.fill_(float("nan"))
        sorted_ids, sorted_weights, expert_ids, num_valid_ids, _ = moe_sorting(
            inputs["topk_ids"], inputs["topk_weights"], experts * ep, hidden_dim, dtypes.bf16, BLOCK_M, inputs["expert_mask"],
            output=out,
        )
        fused_moe_a8w8_blockscale_gfx942_mixed(
            inputs["a"], inputs["a_scale_t"], inputs["w1"], inputs["w1_scale"], inputs["w2"], inputs["w2_scale"], sorted_ids,
            sorted_weights, expert_ids, num_valid_ids, out, topk,
        )
        torch.cuda.synchronize()
        asm_t.check(inputs, out, expected)


@pytest.mark.parametrize("tokens,routing", [(1024, "random"), (8192, "ep_uniform"), (8192, "random")])
def test_dispatch_sort(tokens, routing):
    """fused_moe_a8w8_blockscale_gfx942_sort: one 256-row sort, then the merged kernels."""
    import test_fmoe_a8w8_blockscale_asm_gfx942 as asm_t
    from pyhip.contrib.flydsl.moe_a8w8_blockscale_gfx942 import fused_moe_a8w8_blockscale_gfx942_sort

    if not is_gfx942():
        pytest.skip("gfx942 kernel")
    inter_dim, hidden_dim, topk, experts, ep = 2048, 4096, 8, 32, 8
    inputs = asm_t.make_inputs(tokens, inter_dim, hidden_dim, topk, experts, ep, routing)
    out = torch.full((tokens, hidden_dim), float("nan"), device="cuda", dtype=torch.bfloat16)
    expected = asm_t.reference(inputs)
    fused_moe_a8w8_blockscale_gfx942_sort(
        inputs["a"], inputs["a_scale_t"], inputs["w1"], inputs["w1_scale"], inputs["w2"], inputs["w2_scale"],
        inputs["topk_ids"], inputs["topk_weights"], out, inputs["expert_mask"],
    )
    torch.cuda.synchronize()
    asm_t.check(inputs, out, expected)


def test_merged_capacity_fallback(monkeypatch):
    """Past the merged task scan's sort blocks, fused_moe_a8w8_blockscale_gfx942_mixed runs the BM256 kernels (bm128)."""
    import test_fmoe_a8w8_blockscale_asm_gfx942 as asm_t
    from pyhip.contrib.flydsl import moe_merged_a8w8_blockscale_gfx942 as merged
    from pyhip.contrib.flydsl.moe_a8w8_blockscale_gfx942 import fused_moe_a8w8_blockscale_gfx942_sort

    if not is_gfx942():
        pytest.skip("gfx942 kernel")
    monkeypatch.setattr(merged, "MAX_BLOCKS", 1)
    monkeypatch.setattr(merged, "moe_gateup_merged", None)
    inputs = asm_t.make_inputs(1000, 2048, 4096, 8, 32, 8, "random")
    out = torch.full((1000, 4096), float("nan"), device="cuda", dtype=torch.bfloat16)
    expected = asm_t.reference(inputs)
    fused_moe_a8w8_blockscale_gfx942_sort(
        inputs["a"], inputs["a_scale_t"], inputs["w1"], inputs["w1_scale"], inputs["w2"], inputs["w2_scale"],
        inputs["topk_ids"], inputs["topk_weights"], out, inputs["expert_mask"],
    )
    torch.cuda.synchronize()
    asm_t.check(inputs, out, expected)


MERGED_SHAPES = [
    # tokens, inter, hidden, topk, experts, routing, extra blocks: small blocks only, big blocks with rows 128..255 partly
    # used, full blocks with small tails, a MiMo EP8 mix, big blocks only.
    (512, 2048, 4096, 8, 32, "random", 1),
    (400, 256, 512, 2, 4, "random", 1),
    (600, 256, 512, 2, 4, "random", 1),
    (1100, 256, 512, 4, 16, "random", 0),
    (2048, 2048, 4096, 8, 32, "random", 0),
    (6144, 2048, 4096, 8, 32, "ep_uniform", 1),
]


@pytest.mark.parametrize("tokens,inter_dim,hidden_dim,topk,experts,routing,extra_blocks", MERGED_SHAPES)
def test_gateup_down_merged(tokens, inter_dim, hidden_dim, topk, experts, routing, extra_blocks):
    """The merged GateUp writes the BM256 GateUp's bytes for the blocks with a route at row 128, routed rows within the
    reference's tolerance for the others, and nothing past the valid blocks; the merged Down adds every route."""
    from test_down_a8w8_blockscale_gfx942 import check as check_down
    from pyhip.contrib.flydsl.moe_gateup_a8w8_blockscale_gfx942_lds import moe_gateup_a8w8_blockscale_gfx942_lds
    from pyhip.contrib.flydsl.moe_merged_a8w8_blockscale_gfx942 import moe_down_merged, moe_gateup_merged

    if not is_gfx942():
        pytest.skip("gfx942 kernel")
    inputs = make_inputs(tokens, inter_dim, hidden_dim, topk, experts, routing, extra_blocks=extra_blocks)
    capacity = inputs["expert_ids"].numel() * BLOCK_M
    num_valid = int(inputs["num_valid_ids"][0])
    big = big_rows(inputs)
    gateup_inputs = (inputs["a"], inputs["w1"], inputs["a_scale_t"], inputs["w1_scale"], inputs["sorted_ids"],
                     inputs["expert_ids"], inputs["num_valid_ids"])

    def buffers():
        return (torch.full((capacity, inter_dim), 0x55, device="cuda", dtype=torch.uint8),
                torch.full((inter_dim // SCALE_COLUMNS, capacity), float("nan"), device="cuda"))

    a2_ref, a2_scale_ref = buffers()
    moe_gateup_a8w8_blockscale_gfx942_lds(*gateup_inputs, a2_ref.view(dtypes.fp8), a2_scale_ref, topk)
    a2, a2_scale = buffers()
    previous = None
    for _ in range(2):
        moe_gateup_merged(*gateup_inputs, a2.view(dtypes.fp8), a2_scale, topk)
        torch.cuda.synchronize()
        assert torch.equal(a2[:num_valid][big], a2_ref[:num_valid][big])
        assert torch.equal(a2_scale.t()[:num_valid][big].view(torch.int32), a2_scale_ref.t()[:num_valid][big].view(torch.int32))
        if previous is not None:
            assert torch.equal(a2, previous[0]) and torch.equal(a2_scale.view(torch.int32), previous[1].view(torch.int32))
        previous = a2.clone(), a2_scale.clone()
    check_gateup(inputs, a2.view(dtypes.fp8), a2_scale, 0x55)
    expected, routed, magnitude = down_reference(inputs, a2, a2_scale)
    out = torch.empty((tokens, hidden_dim), device="cuda", dtype=torch.bfloat16)
    for _ in range(2):
        out.zero_()
        moe_down_merged(
            a2.view(dtypes.fp8), inputs["w2"], a2_scale, inputs["w2_scale"], inputs["sorted_ids"], inputs["sorted_weights"],
            inputs["expert_ids"], inputs["num_valid_ids"], out, topk,
        )
        torch.cuda.synchronize()
        check_down(out, expected, routed, magnitude)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=int, default=0)
    args = parser.parse_args()
    shape = MERGED_SHAPES[args.case]
    test_gateup_down_merged(*shape)
    print("merged ok", shape)
