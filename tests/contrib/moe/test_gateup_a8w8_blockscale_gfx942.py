"""Correctness and performance of the gfx942 FP8 block-scale MoE GateUp (BM256) FlyDSL kernel."""

import argparse
import json
import statistics
import subprocess
import threading
import time

import pytest
import torch

import pyhip
from aiter import dtypes
from aiter.ops.shuffle import shuffle_weight

SORT_BLOCK_M = 256


def is_gfx942():
    return torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] == "gfx942"


def make_routes(tokens, topk, experts, routing, generator):
    """Return (token, slot, expert) of every route; each (token, slot) appears at most once."""
    if routing == "ep_uniform":
        # One rank of EP over topk * experts global experts: every token has exactly one
        # local route and every local expert gets tokens / experts rows.
        token = torch.randperm(tokens, generator=generator)
        expert = torch.arange(tokens) % experts
        slot = torch.randint(0, topk, (tokens,), generator=generator)
        return token, slot, expert
    if routing == "ep_random":
        # As ep_uniform, but each token's local expert is drawn at random: unequal expert loads.
        token = torch.arange(tokens)
        expert = torch.randint(0, experts, (tokens,), generator=generator)
        slot = torch.randint(0, topk, (tokens,), generator=generator)
        return token, slot, expert
    if routing == "ep8":
        # One EP8 rank of top-k over 8 * experts global experts: each token has 0 to topk local routes.
        ids = torch.rand((tokens, 8 * experts), generator=generator).topk(topk, dim=1).indices
        token, slot = (ids < experts).nonzero(as_tuple=True)
        return token, slot, ids[token, slot]
    if routing == "random":
        topk_ids = torch.rand((tokens, experts), generator=generator).topk(topk, dim=1).indices
    elif routing == "balanced":
        choices = torch.arange(tokens * topk).view(tokens, topk) % experts
        topk_ids = torch.randperm(experts, generator=generator)[choices[torch.randperm(tokens, generator=generator)]]
    else:
        raise ValueError(f"unsupported routing {routing}")
    token = torch.arange(tokens).repeat_interleave(topk)
    slot = torch.arange(topk).repeat(tokens)
    return token, slot, topk_ids.reshape(-1)


def sort_routes(token, slot, expert, tokens, topk, experts, extra_blocks=0):
    """aiter moe_sorting(256) layout; extra_blocks of garbage capacity must never be read."""
    order = torch.argsort(expert * tokens + token)
    token, slot, expert = token[order], slot[order], expert[order]
    counts = torch.bincount(expert, minlength=experts).tolist()
    pad_id = (topk << 24) | tokens
    ids, block_experts, start = [], [], 0
    for e, count in enumerate(counts):
        if count == 0:
            continue
        blocks = (count + SORT_BLOCK_M - 1) // SORT_BLOCK_M
        padded = torch.full((blocks * SORT_BLOCK_M,), pad_id, dtype=torch.int64)
        padded[:count] = (slot[start : start + count] << 24) | token[start : start + count]
        ids.append(padded)
        block_experts += [e] * blocks
        start += count
    num_valid = sum(len(x) for x in ids)
    ids.append(torch.randint(0, 1 << 31, (extra_blocks * SORT_BLOCK_M,)))
    block_experts += [1 << 30] * extra_blocks
    sorted_ids = torch.cat(ids).to(torch.int32)
    expert_ids = torch.tensor(block_experts, dtype=torch.int32)
    num_valid_ids = torch.tensor([num_valid, tokens], dtype=torch.int32)
    return sorted_ids.cuda(), expert_ids.cuda(), num_valid_ids.cuda()


def make_inputs(tokens, inter_dim, hidden_dim, topk, experts, routing="random", seed=0, extra_blocks=0):
    generator = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    fp8, fp8_max = dtypes.fp8, torch.finfo(dtypes.fp8).max
    kb = hidden_dim // 128
    a_source = torch.randn((tokens, kb, 128), device="cuda") * 0.33
    a_scale = a_source.abs().amax(dim=-1).clamp_min(1e-8) / fp8_max
    a = (a_source / a_scale[..., None]).to(fp8).view(tokens, hidden_dim)
    del a_source
    b_scale = torch.empty((experts, 2 * inter_dim // 128, kb), device="cuda")
    b = torch.empty((experts, 2 * inter_dim, hidden_dim), device="cuda", dtype=fp8)
    for e in range(experts):
        b_source = torch.randn((2 * inter_dim // 128, 128, kb, 128), device="cuda") * 0.2
        b_scale[e] = b_source.abs().amax(dim=(1, 3)).clamp_min(1e-8) / fp8_max
        b[e] = (b_source / b_scale[e][:, None, :, None]).to(fp8).view(2 * inter_dim, hidden_dim)
    del b_source
    # Distinct FP32 scales per row, block and gate/up group.
    a_scale *= 0.5 + torch.rand_like(a_scale)
    b_scale *= 0.5 + torch.rand_like(b_scale)
    token, slot, expert = make_routes(tokens, topk, experts, routing, generator)
    sorted_ids, expert_ids, num_valid_ids = sort_routes(token, slot, expert, tokens, topk, experts, extra_blocks)
    return {
        "a": a,
        "w1_raw": b,
        "w1": shuffle_weight(b, layout=(16, 16)),
        "a_scale_t": a_scale.t().contiguous(),
        "w1_scale": b_scale,
        "routes": (token.cuda(), slot.cuda(), expert.cuda()),
        "sorted_ids": sorted_ids,
        "expert_ids": expert_ids,
        "num_valid_ids": num_valid_ids,
        "shape": (tokens, inter_dim, hidden_dim, topk, experts),
    }


def silu_mul_fp32(inputs):
    """FP32 SiLU(gate) * up [tokens, topk, I] of every route and the routed [tokens, topk] mask."""
    tokens, inter_dim, hidden_dim, topk, experts = inputs["shape"]
    kb = hidden_dim // 128
    a_dequant = (inputs["a"].float().view(tokens, kb, 128) * inputs["a_scale_t"].t()[:, :, None]).view(tokens, hidden_dim)
    values = torch.zeros((tokens, topk, inter_dim), device="cuda")
    routed = torch.zeros((tokens, topk), device="cuda", dtype=torch.bool)
    token, slot, expert = inputs["routes"]
    for e in range(experts):
        rows = (expert == e).nonzero().squeeze(1)
        if rows.numel() == 0:
            continue
        b_dequant = (
            inputs["w1_raw"][e].float().view(2 * inter_dim // 128, 128, kb, 128)
            * inputs["w1_scale"][e][:, None, :, None]
        ).view(2 * inter_dim, hidden_dim)
        gate, up = (a_dequant[token[rows]] @ b_dequant.t()).split(inter_dim, dim=-1)
        values[token[rows], slot[rows]] = torch.nn.functional.silu(gate) * up
        routed[token[rows], slot[rows]] = True
    return values, routed


def reference(inputs):
    values, routed = silu_mul_fp32(inputs)
    out = values.to(torch.bfloat16)
    out[~routed] = float("nan")
    return out


def sorted_fp8_reference(inputs):
    """FP32 SiLU(gate) * up at sorted rows [0, num_valid) (zeros at padding) and its scales amax / 240 per 128 columns."""
    tokens, inter_dim, hidden_dim, topk, experts = inputs["shape"]
    values, _ = silu_mul_fp32(inputs)
    num_valid = int(inputs["num_valid_ids"][0])
    ids = inputs["sorted_ids"][:num_valid].long()
    token, slot = ids & 0xFFFFFF, ids >> 24 & 0xFF
    valid = (token < tokens) & (slot < topk)
    rows = torch.where(valid[:, None], values[token.clamp(max=tokens - 1), slot.clamp(max=topk - 1)], 0.0)
    scale = rows.view(num_valid, -1, 128).abs().amax(-1) / torch.finfo(dtypes.fp8).max
    return rows, scale


def quantize_fp8(rows, scale):
    fp8_max = torch.finfo(dtypes.fp8).max
    quantized = (rows.view(rows.shape[0], -1, 128) / scale[:, :, None].clamp_min(1e-30)).clamp(-fp8_max, fp8_max)
    return quantized.to(dtypes.fp8).view(rows.shape)


def run_kernel(inputs, out, layout="2x4", **kw):
    from pyhip.contrib.flydsl.moe_gateup_a8w8_blockscale_gfx942_lds import moe_gateup_a8w8_blockscale_gfx942_lds

    args = (
        inputs["a"], inputs["w1"], inputs["a_scale_t"], inputs["w1_scale"],
        inputs["sorted_ids"], inputs["expert_ids"], inputs["num_valid_ids"], out,
    )
    return moe_gateup_a8w8_blockscale_gfx942_lds(*args, layout=layout, **kw)


def check(inputs, out, expected):
    routed = ~expected[:, :, 0].isnan()
    # Unrouted (token, slot) entries keep their NaN sentinel.
    assert out[~routed].isnan().all(), "kernel wrote an unrouted slot"
    actual, wanted = out[routed].float(), expected[routed].float()
    assert torch.isfinite(actual).all(), "non-finite or unwritten routed output"
    torch.testing.assert_close(actual, wanted, rtol=0.05, atol=0.01)
    diff = pyhip.calc_diff(actual, wanted)
    assert diff < 1e-5, f"calc_diff {diff}"
    # Truncating instead of rounding to nearest biases every value toward zero by ~0.3%.
    bias = ((actual - wanted) * wanted.sign()).mean() / wanted.abs().mean()
    assert abs(bias) < 5e-4, f"BF16 rounding bias {bias:.2e}"
    return diff


def check_fp8_sorted(inputs, a2, a2_scale_t, sentinel):
    rows, scale = sorted_fp8_reference(inputs)
    num_valid = rows.shape[0]
    kernel_scale = a2_scale_t.t()[:num_valid]
    torch.testing.assert_close(kernel_scale, scale, rtol=1e-4, atol=0)
    dequant = (a2[:num_valid].float().view(num_valid, -1, 128) * kernel_scale[:, :, None]).view(num_valid, -1)
    diff = pyhip.calc_diff(dequant, rows)
    assert diff < 1e-3, f"calc_diff {diff}"
    # Requantized with the kernel's scales, only FP32 rounding differences may flip an FP8 rounding.
    requant = quantize_fp8(rows, kernel_scale)
    mismatch = (requant.view(torch.uint8) != a2[:num_valid].view(torch.uint8)).float().mean().item()
    assert mismatch < 0.01, f"FP8 mismatch {mismatch:.4f}"
    # Capacity past the valid blocks is never written.
    assert (a2[num_valid:].view(torch.uint8) == sentinel).all()
    assert a2_scale_t[:, num_valid:].isnan().all()
    return diff


@pytest.mark.parametrize(
    "tokens,inter_dim,hidden_dim,topk,experts,routing,extra_blocks",
    [
        (200, 256, 512, 2, 4, "random", 2),
        (300, 256, 512, 2, 3, "random", 1),
        (1000, 256, 384, 4, 8, "random", 0),
        (4096, 128, 256, 8, 32, "balanced", 1),
        (1024, 2048, 4096, 8, 32, "ep_uniform", 3),
    ],
)
@pytest.mark.parametrize(
    "config",
    [
        dict(layout="2x4"),
        dict(layout="4x2"),
        dict(layout="4x2", xcd_swizzle=False),
        dict(layout="2x4", persistent=True),
        # Few workers so each loops over many tasks (odd K blocks, dead last prefetch, XCD ranges).
        dict(layout="2x4", persistent=8),
        dict(layout="4x2", persistent=8),
        dict(layout="2x4", persistent=16, xcd_swizzle=False),
        dict(layout="2x4", pair_blocks=True),
        dict(layout="4x2", persistent=8, pair_blocks=True),
    ],
    ids=[
        "lds-2x4", "lds-4x2", "lds-4x2-linear", "lds-2x4-persistent", "lds-2x4-persistent8", "lds-4x2-persistent8",
        "lds-2x4-persistent16-linear", "lds-2x4-pair", "lds-4x2-persistent8-pair",
    ],
)
def test_gateup(tokens, inter_dim, hidden_dim, topk, experts, routing, extra_blocks, config):
    if not is_gfx942():
        pytest.skip("gfx942 kernel")
    inputs = make_inputs(tokens, inter_dim, hidden_dim, topk, experts, routing, extra_blocks=extra_blocks)
    expected = reference(inputs)
    out = torch.empty((tokens, topk, inter_dim), device="cuda", dtype=torch.bfloat16)
    previous = None
    for _ in range(2):
        out.fill_(float("nan"))
        run_kernel(inputs, out, **config)
        torch.cuda.synchronize()
        check(inputs, out, expected)
        if previous is not None:
            assert torch.equal(out.view(torch.int16), previous.view(torch.int16))
        previous = out.clone()


@pytest.mark.parametrize(
    "tokens,inter_dim,hidden_dim,topk,experts,routing,extra_blocks",
    [
        (200, 256, 512, 2, 4, "random", 2),
        (300, 256, 512, 2, 3, "random", 1),
        (1000, 256, 384, 4, 8, "random", 0),
        (4096, 128, 256, 8, 32, "balanced", 1),
        (1024, 2048, 4096, 8, 32, "ep_uniform", 3),
        (2048, 2048, 4096, 8, 32, "ep8", 1),
    ],
)
@pytest.mark.parametrize(
    "config",
    [dict(layout="2x4"), dict(layout="4x2"), dict(layout="4x2", xcd_swizzle=False), dict(layout="2x4", pair_blocks=True)],
    ids=["lds-2x4", "lds-4x2", "lds-4x2-linear", "lds-2x4-pair"],
)
def test_gateup_fp8_sorted(tokens, inter_dim, hidden_dim, topk, experts, routing, extra_blocks, config):
    if not is_gfx942():
        pytest.skip("gfx942 kernel")
    inputs = make_inputs(tokens, inter_dim, hidden_dim, topk, experts, routing, extra_blocks=extra_blocks)
    capacity = inputs["expert_ids"].numel() * SORT_BLOCK_M
    a2 = torch.empty((capacity, inter_dim), device="cuda", dtype=torch.uint8)
    a2_scale_t = torch.empty((inter_dim // 128, capacity), device="cuda")
    previous = None
    for _ in range(2):
        a2.fill_(0x55)
        a2_scale_t.fill_(float("nan"))
        run_kernel(inputs, a2.view(dtypes.fp8), out_scale_t=a2_scale_t, topk=topk, **config)
        torch.cuda.synchronize()
        check_fp8_sorted(inputs, a2.view(dtypes.fp8), a2_scale_t, 0x55)
        if previous is not None:
            assert torch.equal(a2, previous[0]) and torch.equal(a2_scale_t.view(torch.int32), previous[1].view(torch.int32))
        previous = a2.clone(), a2_scale_t.clone()


def sustained(fn, flops, seconds):
    samples, stop = [], threading.Event()

    def sampler():
        while not stop.is_set():
            result = subprocess.run(["amd-smi", "metric", "-g", str(torch.cuda.current_device()), "-c", "-p", "--json"], capture_output=True, text=True)
            try:
                gpu = json.loads(result.stdout)["gpu_data"][0]
                clocks = [v["clk"]["value"] for k, v in gpu["clock"].items() if k.startswith("gfx_")]
                samples.append((time.time(), sum(clocks) / len(clocks), gpu["power"]["socket_power"]["value"]))
            except (ValueError, KeyError, IndexError, TypeError):
                pass

    for _ in range(3):
        fn(0)
    torch.cuda.synchronize()
    thread = threading.Thread(target=sampler)
    thread.start()
    start_time = time.time()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    count = 0
    while time.time() - start_time < seconds:
        for _ in range(10):
            fn(count)
            count += 1
        torch.cuda.synchronize()
    end.record()
    end.synchronize()
    stop.set()
    thread.join()
    tflops = flops * count / (start.elapsed_time(end) * 1e-3) / 1e12
    steady = [s for s in samples if s[0] - start_time > 1.0] or samples
    clock = statistics.median(s[1] for s in steady) if steady else float("nan")
    power = statistics.median(s[2] for s in steady) if steady else float("nan")
    cus = torch.cuda.get_device_properties(0).multi_processor_count
    peak = cus * clock * 1e6 * 4096 / 1e12
    return {"tflops": tflops, "clock_mhz": clock, "power_w": power, "peak_at_clock": peak, "mfma_efficiency": tflops / peak}


def benchmark(args):
    tokens, inter_dim, hidden_dim, topk, experts = args.tokens, args.inter_dim, args.hidden_dim, args.topk, args.experts
    inputs = make_inputs(tokens, inter_dim, hidden_dim, topk, experts, args.routing, seed=args.seed)
    routes = inputs["routes"][0].numel()
    flops = 4 * routes * inter_dim * hidden_dim
    out = torch.empty((tokens, topk, inter_dim), device="cuda", dtype=torch.bfloat16)
    out.fill_(float("nan"))
    run_kernel(inputs, out, xcd_swizzle=not args.no_xcd_swizzle)
    torch.cuda.synchronize()
    if args.check:
        print(f"check: calc_diff={check(inputs, out, reference(inputs)):.3e}", flush=True)
    validated = out.clone()
    clones = [inputs] + [
        dict(inputs, **{name: inputs[name].clone() for name in ("a", "w1", "a_scale_t", "w1_scale")})
        for _ in range(args.clones - 1)
    ]

    def call(index):
        run_kernel(clones[index % len(clones)], out, xcd_swizzle=not args.no_xcd_swizzle)

    for index in range(len(clones)):
        call(index)
    torch.cuda.synchronize()
    samples = []
    for index in range(args.samples):
        with pyhip.cudaPerf(flops, name="moe_gateup", verbose=0) as perf:
            call(index)
        samples.append(perf.latencies[0] * 1e6)
    torch.cuda.synchronize()
    assert torch.equal(out.view(torch.int16), validated.view(torch.int16)), "timed output differs"
    median_us = statistics.median(samples)
    blocks = int(inputs["num_valid_ids"][0].item()) // SORT_BLOCK_M
    report = {
        "shape": dict(tokens=tokens, inter_dim=inter_dim, hidden_dim=hidden_dim, topk=topk, experts=experts, routing=args.routing),
        "routes": routes,
        "blocks": blocks,
        "workgroups": blocks * (inter_dim // 128),
        "median_us": median_us,
        "median_tflops": flops / median_us / 1e6,
        "best_tflops": flops / min(samples) / 1e6,
        "samples_us": samples,
    }
    print(
        f"moe_gateup T={tokens} I={inter_dim} K={hidden_dim} E={experts} topk={topk} {args.routing}: "
        f"routes={routes} blocks={blocks} median {median_us:.1f} us {report['median_tflops']:.1f} TF, best {report['best_tflops']:.1f} TF",
        flush=True,
    )
    if args.dense:
        from pyhip.contrib.flydsl.gemm_a8w8_blockscale_gfx942_8x1 import gemm_a8w8_blockscale_gfx942_8x1

        dense = [
            (torch.randn((routes, hidden_dim), device="cuda") / 10).to(dtypes.fp8),
            inputs["w1"][0].contiguous(),
            torch.rand((hidden_dim // 128, routes), device="cuda"),
            inputs["w1_scale"][0].contiguous(),
        ]
        dense_sets = [dense] + [[t.clone() for t in dense] for _ in range(args.clones - 1)]
        dense_out = torch.empty((routes, 2 * inter_dim), device="cuda", dtype=torch.bfloat16)
        for d in dense_sets:
            gemm_a8w8_blockscale_gfx942_8x1(d[0], d[1], dense_out, d[2], d[3], True)
        torch.cuda.synchronize()
        dense_samples = []
        for index in range(args.samples):
            d = dense_sets[index % len(dense_sets)]
            with pyhip.cudaPerf(flops, name="dense_8x1", verbose=0) as perf:
                gemm_a8w8_blockscale_gfx942_8x1(d[0], d[1], dense_out, d[2], d[3], True)
            dense_samples.append(perf.latencies[0] * 1e6)
        dense_median = statistics.median(dense_samples)
        report["dense_median_us"] = dense_median
        report["dense_median_tflops"] = flops / dense_median / 1e6
        print(f"dense 8x1 M={routes} N={2 * inter_dim} K={hidden_dim}: median {dense_median:.1f} us {flops / dense_median / 1e6:.1f} TF", flush=True)
    if args.sustained:
        report["sustained"] = sustained(call, flops, args.sustained)
        s = report["sustained"]
        print(
            f"sustained: {s['tflops']:.1f} TF at {s['clock_mhz']:.0f} MHz, {s['power_w']:.0f} W, "
            f"peak@clock {s['peak_at_clock']:.0f} TF, MFMA efficiency {s['mfma_efficiency'] * 100:.1f}%",
            flush=True,
        )
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=8192)
    parser.add_argument("--inter-dim", type=int, default=2048)
    parser.add_argument("--hidden-dim", type=int, default=4096)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--experts", type=int, default=32)
    parser.add_argument("--routing", default="ep_uniform", choices=["ep_uniform", "ep_random", "ep8", "random", "balanced"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--clones", type=int, default=4)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--dense", action="store_true", help="also time the dense 8x1 GEMM at M=routes")
    parser.add_argument("--sustained", type=float, default=0, help="seconds of back-to-back launches with clock telemetry")
    parser.add_argument("--no-xcd-swizzle", action="store_true")
    parser.add_argument("--json")
    benchmark(parser.parse_args())
