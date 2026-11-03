"""Correctness and performance of the gfx942 FP8 block-scale MoE Down (GEMM2) FlyDSL kernels."""

import argparse
import statistics

import pytest
import torch

import pyhip
from aiter import dtypes
from aiter.ops.shuffle import shuffle_weight
from test_gateup_a8w8_blockscale_gfx942 import SORT_BLOCK_M, check_fp8_sorted, is_gfx942, make_routes, sort_routes
from test_gateup_a8w8_blockscale_gfx942 import make_inputs as make_gateup_inputs
from test_gateup_a8w8_blockscale_gfx942 import run_kernel as run_gateup


def make_inputs(tokens, hidden_dim, inter_dim, topk, experts, routing="random", seed=0, extra_blocks=0):
    generator = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    fp8, fp8_max = dtypes.fp8, torch.finfo(dtypes.fp8).max
    kb = inter_dim // 128
    token, slot, expert = make_routes(tokens, topk, experts, routing, generator)
    sorted_ids, expert_ids, num_valid_ids = sort_routes(token, slot, expert, tokens, topk, experts, extra_blocks)
    # Only routed (token, slot) rows of A2 hold data.
    rows = (token * topk + slot).cuda()
    a_source = torch.randn((rows.numel(), kb, 128), device="cuda") * 0.33
    a_scale = a_source.abs().amax(dim=-1).clamp_min(1e-8) / fp8_max
    a2 = torch.zeros((tokens * topk, inter_dim), device="cuda", dtype=fp8)
    a2[rows] = (a_source / a_scale[..., None]).to(fp8).view(-1, inter_dim)
    del a_source
    a2_scale = torch.ones((tokens * topk, kb), device="cuda")
    # Distinct FP32 scales per row and block.
    a2_scale[rows] = a_scale * (0.5 + torch.rand_like(a_scale))
    w_scale = torch.empty((experts, hidden_dim // 128, kb), device="cuda")
    w = torch.empty((experts, hidden_dim, inter_dim), device="cuda", dtype=fp8)
    for e in range(experts):
        w_source = torch.randn((hidden_dim // 128, 128, kb, 128), device="cuda") * 0.2
        w_scale[e] = w_source.abs().amax(dim=(1, 3)).clamp_min(1e-8) / fp8_max
        w[e] = (w_source / w_scale[e][:, None, :, None]).to(fp8).view(hidden_dim, inter_dim)
    del w_source
    w_scale *= 0.5 + torch.rand_like(w_scale)
    topk_weights = torch.rand((tokens, topk), generator=generator) + 0.1
    topk_weights = (topk_weights / topk_weights.sum(dim=1, keepdim=True)).cuda()
    ids = sorted_ids.long()
    sorted_token, sorted_slot = ids & 0xFFFFFF, ids >> 24 & 0xFF
    routed = (sorted_token < tokens) & (sorted_slot < topk)
    sorted_weights = torch.where(
        routed, topk_weights[sorted_token.clamp(max=tokens - 1), sorted_slot.clamp(max=topk - 1)], 0.0
    ).float()
    # Sorted-row A2 for sorted_a2=True: position p holds the A2 row of sorted_ids[p], zeros for padding.
    position_rows = torch.where(routed, sorted_token * topk + sorted_slot, 0)
    a2_sorted = a2.view(torch.uint8)[position_rows]
    a2_sorted[~routed] = 0
    return {
        "a2": a2.view(tokens, topk, inter_dim),
        "a2_sorted": a2_sorted.view(fp8),
        "a2_scale_sorted_t": a2_scale[position_rows].t().contiguous(),
        "w2_raw": w,
        "w2": shuffle_weight(w, layout=(16, 16)),
        "a2_scale_t": a2_scale.t().contiguous(),
        "w2_scale": w_scale,
        "topk_weights": topk_weights,
        "routes": (token.cuda(), slot.cuda(), expert.cuda()),
        "sorted_ids": sorted_ids,
        "sorted_weights": sorted_weights,
        "expert_ids": expert_ids,
        "num_valid_ids": num_valid_ids,
        "shape": (tokens, hidden_dim, inter_dim, topk, experts),
    }


def reference(inputs):
    """BF16 per route (weight applied), then each token's routes summed in FP32 in slot order.

    Also returns the routed-token mask and the sum of |route| per element (the scale of rounding errors).
    """
    tokens, hidden_dim, inter_dim, topk, experts = inputs["shape"]
    kb = inter_dim // 128
    token, slot, expert = inputs["routes"]
    a2 = inputs["a2"].view(tokens * topk, kb, 128)
    a2_scale = inputs["a2_scale_t"].t()
    routed_y = torch.empty((token.numel(), hidden_dim), device="cuda", dtype=torch.bfloat16)
    for e in range(experts):
        index = (expert == e).nonzero().squeeze(1)
        if index.numel() == 0:
            continue
        rows = token[index] * topk + slot[index]
        a_dequant = (a2[rows].float() * a2_scale[rows][:, :, None]).view(-1, inter_dim)
        w_dequant = (
            inputs["w2_raw"][e].float().view(hidden_dim // 128, 128, kb, 128) * inputs["w2_scale"][e][:, None, :, None]
        ).view(hidden_dim, inter_dim)
        weight = inputs["topk_weights"][token[index], slot[index]]
        routed_y[index] = ((a_dequant @ w_dequant.t()) * weight[:, None]).to(torch.bfloat16)
    out = torch.zeros((tokens, hidden_dim), device="cuda")
    magnitude = torch.zeros((tokens, hidden_dim), device="cuda")
    for s in range(topk):
        in_slot = slot == s
        out.index_add_(0, token[in_slot], routed_y[in_slot].float())
        magnitude.index_add_(0, token[in_slot], routed_y[in_slot].float().abs())
    routed = torch.zeros(tokens, device="cuda", dtype=torch.bool)
    routed[token] = True
    return out.to(torch.bfloat16), routed, magnitude


def kernel_args(inputs, sorted_a2=False):
    a2, a2_scale = ("a2_sorted", "a2_scale_sorted_t") if sorted_a2 else ("a2", "a2_scale_t")
    return [inputs[n] for n in (a2, "w2", a2_scale, "w2_scale", "sorted_ids", "sorted_weights", "expert_ids", "num_valid_ids")]


def run_kernel(inputs, out, sorted_a2=False, **kw):
    from pyhip.contrib.flydsl.moe_down_a8w8_blockscale_gfx942_lds import moe_down_a8w8_blockscale_gfx942_lds

    return moe_down_a8w8_blockscale_gfx942_lds(*kernel_args(inputs, sorted_a2), out, topk=inputs["shape"][3], **kw)


def check(out, expected, routed, magnitude):
    assert (out[~routed] == 0).all(), "tokens without routes must be zero"
    actual, wanted = out[routed].float(), expected[routed].float()
    assert torch.isfinite(actual).all(), "non-finite or unwritten output"
    # BF16 rounding of each route (and of the running sum for atomics) scales with the routes, not the
    # result, which is much smaller where routes cancel.
    excess = (actual - wanted).abs() - (0.02 + 0.05 * wanted.abs() + 2**-6 * magnitude[routed])
    assert excess.max() <= 0, f"{(excess > 0).sum().item()} elements off, worst by {excess.max().item():.3f}"
    diff = pyhip.calc_diff(actual, wanted)
    assert diff < 1e-5, f"calc_diff {diff}"
    # Truncating instead of rounding to nearest biases every value toward zero by ~0.3%.
    bias = ((actual - wanted) * wanted.sign()).mean() / wanted.abs().mean()
    assert abs(bias) < 5e-4, f"BF16 rounding bias {bias:.2e}"
    return diff


@pytest.mark.parametrize(
    "tokens,hidden_dim,inter_dim,topk,experts,routing,extra_blocks",
    [
        (200, 512, 256, 2, 4, "random", 2),
        (300, 256, 512, 2, 3, "random", 1),
        (1000, 768, 384, 4, 8, "random", 0),
        (4096, 256, 128, 8, 32, "balanced", 1),
        (1024, 4096, 2048, 8, 32, "ep_uniform", 3),
        (2048, 4096, 2048, 8, 32, "ep8", 0),
    ],
)
@pytest.mark.parametrize(
    "config",
    [
        dict(layout="2x4", reduce="atomic"),
        dict(layout="2x4", reduce="kernel"),
        dict(layout="4x2", reduce="atomic"),
        dict(layout="4x2", reduce="kernel"),
        dict(layout="2x4", reduce="kernel", xcd_swizzle=False),
        dict(layout="4x2", reduce="atomic", xcd_swizzle=False),
        dict(layout="4x2", reduce="atomic", sorted_a2=True),
        dict(layout="2x4", reduce="kernel", sorted_a2=True),
        dict(layout="4x2", reduce="atomic", sorted_a2=True, single_store=True),
        dict(layout="2x4", reduce="atomic", single_store=True),
    ],
    ids=[
        "2x4-atomic", "2x4-kernel", "4x2-atomic", "4x2-kernel", "2x4-kernel-linear", "4x2-atomic-linear",
        "4x2-atomic-sorted", "2x4-kernel-sorted", "4x2-atomic-sorted-single", "2x4-atomic-single",
    ],
)
def test_down(tokens, hidden_dim, inter_dim, topk, experts, routing, extra_blocks, config):
    if not is_gfx942():
        pytest.skip("gfx942 kernel")
    inputs = make_inputs(tokens, hidden_dim, inter_dim, topk, experts, routing, extra_blocks=extra_blocks)
    expected, routed, magnitude = reference(inputs)
    out = torch.empty((tokens, hidden_dim), device="cuda", dtype=torch.bfloat16)
    previous = None
    for _ in range(2):
        out.fill_(float("nan"))
        run_kernel(inputs, out, **config)
        torch.cuda.synchronize()
        check(out, expected, routed, magnitude)
        # BF16 atomics sum in arrival order; the reduce kernel's order is fixed.
        if config["reduce"] == "kernel" and previous is not None:
            assert torch.equal(out.view(torch.int16), previous.view(torch.int16))
        previous = out.clone()


@pytest.mark.parametrize("tokens,topk,experts,routing,extra_blocks", [(300, 2, 3, "random", 1), (2048, 8, 32, "ep8", 2)])
def test_route_counts(tokens, topk, experts, routing, extra_blocks):
    from pyhip.contrib.flydsl.moe_down_a8w8_blockscale_gfx942_lds import moe_route_counts

    if not is_gfx942():
        pytest.skip("gfx942 kernel")
    generator = torch.Generator().manual_seed(0)
    token, slot, expert = make_routes(tokens, topk, experts, routing, generator)
    sorted_ids, expert_ids, num_valid_ids = sort_routes(token, slot, expert, tokens, topk, experts, extra_blocks)
    counts = torch.full((tokens,), -7, device="cuda", dtype=torch.int32)
    moe_route_counts(sorted_ids, num_valid_ids, expert_ids, counts, topk)
    torch.cuda.synchronize()
    assert torch.equal(counts.cpu(), torch.bincount(token, minlength=tokens).to(torch.int32))


def test_validate_extents():
    from pyhip.contrib.flydsl.moe_down_a8w8_blockscale_gfx942_lds import validate_parameters

    # aiter's EP sort of 32K tokens over 256 global experts at block 256 has 327680 rows: a 2.68 GB rows buffer,
    # which only the reduce mode allocates.
    validate_parameters(32768, 4096, 2048, 8, 32, 327680, sorted_a2=True, atomic=True)
    with pytest.raises(ValueError, match="rows="):
        validate_parameters(32768, 4096, 2048, 8, 32, 327680, sorted_a2=True, atomic=False)


@pytest.mark.parametrize(
    "tokens,hidden_dim,inter_dim,topk,experts,routing",
    [(300, 512, 256, 2, 3, "random"), (1024, 4096, 2048, 8, 32, "ep_uniform"), (2048, 4096, 2048, 8, 32, "ep8")],
)
@pytest.mark.parametrize("gateup_layout,down_layout", [("2x4", "4x2"), ("4x2", "2x4")])
def test_gateup_down_fp8_pipeline(tokens, hidden_dim, inter_dim, topk, experts, routing, gateup_layout, down_layout):
    """GEMM1 writes the sorted-row FP8 A2 and its scales; GEMM2 reads them with sorted_a2."""
    if not is_gfx942():
        pytest.skip("gfx942 kernel")
    gateup = make_gateup_inputs(tokens, inter_dim, hidden_dim, topk, experts, routing, extra_blocks=1)
    inputs = make_inputs(tokens, hidden_dim, inter_dim, topk, experts, routing, extra_blocks=1)
    assert torch.equal(gateup["sorted_ids"][: -SORT_BLOCK_M], inputs["sorted_ids"][: -SORT_BLOCK_M])
    capacity = inputs["expert_ids"].numel() * SORT_BLOCK_M
    a2 = torch.full((capacity, inter_dim), 0x55, device="cuda", dtype=torch.uint8)
    a2_scale_t = torch.full((inter_dim // 128, capacity), float("nan"), device="cuda")
    run_gateup(gateup, a2.view(dtypes.fp8), out_scale_t=a2_scale_t, topk=topk, layout=gateup_layout)
    torch.cuda.synchronize()
    check_fp8_sorted(gateup, a2.view(dtypes.fp8), a2_scale_t, 0x55)
    # The reference reads the same FP8 rows in [tokens, topk] order.
    num_valid = int(inputs["num_valid_ids"][0])
    ids = inputs["sorted_ids"][:num_valid].long()
    token, slot = ids & 0xFFFFFF, ids >> 24 & 0xFF
    positions = ((token < tokens) & (slot < topk)).nonzero().squeeze(1)
    routed_rows = token[positions] * topk + slot[positions]
    a2_routed = torch.zeros((tokens * topk, inter_dim), device="cuda", dtype=torch.uint8)
    a2_routed[routed_rows] = a2[positions]
    scale_routed = torch.ones((tokens * topk, inter_dim // 128), device="cuda")
    scale_routed[routed_rows] = a2_scale_t.t()[positions]
    inputs.update(
        a2=a2_routed.view(dtypes.fp8).view(tokens, topk, inter_dim),
        a2_scale_t=scale_routed.t().contiguous(),
        a2_sorted=a2.view(dtypes.fp8),
        a2_scale_sorted_t=a2_scale_t,
    )
    expected, routed, magnitude = reference(inputs)
    out = torch.empty((tokens, hidden_dim), device="cuda", dtype=torch.bfloat16)
    for reduce, single_store in (("atomic", False), ("atomic", True), ("kernel", False)):
        out.fill_(float("nan"))
        run_kernel(inputs, out, sorted_a2=True, layout=down_layout, reduce=reduce, single_store=single_store)
        torch.cuda.synchronize()
        check(out, expected, routed, magnitude)


def benchmark(args):
    from pyhip.contrib.flydsl.moe_down_a8w8_blockscale_gfx942_lds import (
        moe_down_a8w8_blockscale_gfx942_lds,
        moe_down_reduce,
        moe_down_rows_a8w8_blockscale_gfx942_lds,
    )

    tokens, hidden_dim, inter_dim, topk, experts = args.tokens, args.hidden_dim, args.inter_dim, args.topk, args.experts
    inputs = make_inputs(tokens, hidden_dim, inter_dim, topk, experts, args.routing, seed=args.seed)
    routes = inputs["routes"][0].numel()
    blocks = int(inputs["num_valid_ids"][0].item()) // SORT_BLOCK_M
    flops = 2 * routes * hidden_dim * inter_dim
    out = torch.empty((tokens, hidden_dim), device="cuda", dtype=torch.bfloat16)
    rows = torch.empty((inputs["expert_ids"].numel() * SORT_BLOCK_M, hidden_dim), device="cuda", dtype=torch.bfloat16)
    inverse = torch.empty((tokens, topk), device="cuda", dtype=torch.int32)
    variants = [(layout, sorted_a2) for sorted_a2 in ((False, True) if args.sorted_a2 else (False,)) for layout in args.layouts]
    if args.check:
        expected, routed, magnitude = reference(inputs)
        for layout, sorted_a2 in variants:
            for reduce in ("atomic", "kernel"):
                out.fill_(float("nan"))
                run_kernel(inputs, out, sorted_a2=sorted_a2, layout=layout, reduce=reduce)
                torch.cuda.synchronize()
                print(f"check {layout} sorted_a2={sorted_a2} {reduce}: calc_diff={check(out, expected, routed, magnitude):.3e}", flush=True)
    cases = {}
    for layout, sorted_a2 in variants:
        args_ = kernel_args(inputs, sorted_a2)
        tag = f"{layout}{' sorted A2' if sorted_a2 else ''}"
        cases[f"{tag} atomic (zero + down)"] = (
            None, lambda args_=args_, layout=layout: moe_down_a8w8_blockscale_gfx942_lds(*args_, out, layout=layout, topk=topk)
        )
        cases[f"{tag} atomic down only"] = (
            out.zero_,
            lambda args_=args_, layout=layout: moe_down_a8w8_blockscale_gfx942_lds(*args_, out, layout=layout, zero_out=False, topk=topk),
        )
        cases[f"{tag} kernel (fill + down + reduce)"] = (
            None,
            lambda args_=args_, layout=layout: moe_down_a8w8_blockscale_gfx942_lds(
                *args_, out, layout=layout, reduce="kernel", rows=rows, inverse=inverse, topk=topk
            ),
        )
        cases[f"{tag} rows (fill + down)"] = (
            None, lambda args_=args_, layout=layout: moe_down_rows_a8w8_blockscale_gfx942_lds(*args_, rows, inverse, layout=layout)
        )
    moe_down_rows_a8w8_blockscale_gfx942_lds(*kernel_args(inputs), rows, inverse, layout=args.layouts[0])
    cases["reduce only"] = (None, lambda: moe_down_reduce(rows, inverse, out))
    cases["zero out"] = (None, out.zero_)
    # A 1 GiB read before every sample evicts L2 and MALL.
    flush = torch.zeros(1 << 28, device="cuda", dtype=torch.int32)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    samples = {name: [] for name in cases}
    for _ in range(args.reps):
        for order in (list(cases), list(cases)[::-1]):
            for name in order:
                setup, fn = cases[name]
                for _ in range(args.samples):
                    if setup is not None:
                        setup()
                    flush.sum()
                    start.record()
                    fn()
                    end.record()
                    end.synchronize()
                    samples[name].append(start.elapsed_time(end) * 1e3)
    print(
        f"moe_down T={tokens} H={hidden_dim} I={inter_dim} E={experts} topk={topk} {args.routing}: routes={routes} "
        f"blocks={blocks} tasks={blocks * hidden_dim // 256} (cold, median of {len(samples[name])})",
        flush=True,
    )
    for name, us in samples.items():
        median = statistics.median(us)
        print(f"  {name:34s} {median:8.1f} us {flops / median / 1e6:7.1f} TF   min {min(us):8.1f} us", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=32768)
    parser.add_argument("--hidden-dim", type=int, default=4096)
    parser.add_argument("--inter-dim", type=int, default=2048)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--experts", type=int, default=32)
    parser.add_argument("--routing", default="ep_uniform", choices=["ep_uniform", "ep_random", "ep8", "random", "balanced"])
    parser.add_argument("--layouts", nargs="+", default=["2x4", "4x2"])
    parser.add_argument("--sorted-a2", action="store_true", help="also time the sorted-row A2 input")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--samples", type=int, default=8)
    benchmark(parser.parse_args())
