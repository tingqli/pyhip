"""Correctness, performance and kernel selection of aiter's gfx942 FP8 block-scale 1-stage fused MoE
(aiter.fmoe_fp8_blockscale_g1u1, one launch: stage1 GateUp + SiLU, FP8 per-(row,128) quantization of the
stage1 output, stage2 Down, atomically reduced into [tokens, hidden]).

The defaults are one EP8 rank of MiMo-V2.6-Flash (hidden 4096, moe_inter 2048, 256 experts, top-8,
128x128 FP8 block scales) at 32768 tokens: 32 local experts, sorted exactly as aiter.fused_moe does it
under EP (global expert ids + expert_mask, so the sort buffers and non-persistent grids are sized for
all 256 experts).
"""

import argparse
import functools
import json
import statistics
import subprocess
import threading
import time

import pandas as pd
import pytest
import torch
import torch.nn.functional as F

import aiter
import pyhip
from aiter import ActivationType, dtypes
from aiter.fused_moe import moe_sorting
from aiter.jit.core import AITER_ASM_DIR
from aiter.ops.shuffle import shuffle_weight

BLK = 128
KERNEL_CSV = f"{AITER_ASM_DIR}/gfx942/fmoe/silu/fmoe_bf16_blockscaleFp8_g1u1_silu.csv"


def is_gfx942():
    return torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] == "gfx942"


def load_kernels():
    """Host-sorted (flat == 0) kernels selectable by kernelName; "auto" is what untuned fused_moe runs."""
    kernels = {"auto": dict(knl_name="", block_m=32, tile_n=128, ps_tgs_per_cu=0)}
    df = pd.read_csv(KERNEL_CSV)
    for row in df[df["flat"] == 0].itertuples():
        name = row.co_name.removeprefix("fmoe_bf16_blockscaleFp8_g1u1_").removesuffix(".co").replace("silu_1tg_", "")
        kernels[name] = dict(
            knl_name=row.knl_name, block_m=row.subGU_m, tile_n=row.subGU_n, ps_tgs_per_cu=row.tg_num_perCU * row.ps
        )
    return kernels


KERNELS = load_kernels()


def make_routes(tokens, topk, experts, ep, routing, generator):
    """Global top-k ids [tokens, topk] and weights for the EP rank that owns experts [0, experts)."""
    global_experts = experts * ep
    if routing == "ep_uniform":
        # Same workload as the GateUp test: every token has exactly one local route and every local
        # expert gets tokens / experts rows; the other topk - 1 slots go to distinct remote experts.
        assert global_experts - experts >= topk - 1, "ep_uniform needs topk - 1 remote experts"
        local = torch.empty(tokens, dtype=torch.int64)
        local[torch.randperm(tokens, generator=generator)] = torch.arange(tokens) % experts
        remote = torch.rand((tokens, global_experts - experts), generator=generator).topk(topk - 1, dim=1).indices
        is_local = torch.arange(topk) == torch.randint(0, topk, (tokens, 1), generator=generator)
        ids = torch.empty((tokens, topk), dtype=torch.int64)
        ids[is_local] = local
        ids[~is_local] = remote.flatten() + experts
    elif routing == "random":
        ids = torch.rand((tokens, global_experts), generator=generator).topk(topk, dim=1).indices
    elif routing == "balanced":
        choices = torch.arange(tokens * topk).view(tokens, topk) % global_experts
        ids = torch.randperm(global_experts, generator=generator)[choices[torch.randperm(tokens, generator=generator)]]
    else:
        raise ValueError(f"unsupported routing {routing}")
    weights = torch.rand((tokens, topk), generator=generator) + 0.1
    return ids.to(torch.int32), weights / weights.sum(dim=1, keepdim=True)


def block_quant(experts, n, k, std):
    """FP8 weights [experts, n, k] with distinct FP32 scales per 128x128 block."""
    fp8, fp8_max = dtypes.fp8, torch.finfo(dtypes.fp8).max
    scale = torch.empty((experts, n // BLK, k // BLK), device="cuda")
    weight = torch.empty((experts, n, k), device="cuda", dtype=fp8)
    for e in range(experts):
        source = torch.randn((n // BLK, BLK, k // BLK, BLK), device="cuda") * std
        scale[e] = source.abs().amax(dim=(1, 3)).clamp_min(1e-8) / fp8_max
        weight[e] = (source / scale[e][:, None, :, None]).to(fp8).view(n, k)
    return weight, scale * (0.5 + torch.rand_like(scale))


def make_inputs(tokens, inter_dim, hidden_dim, topk, experts, ep, routing="ep_uniform", seed=0):
    generator = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    fp8, fp8_max = dtypes.fp8, torch.finfo(dtypes.fp8).max
    a_source = torch.randn((tokens, hidden_dim // BLK, BLK), device="cuda") * 0.33
    a_scale = a_source.abs().amax(dim=-1).clamp_min(1e-8) / fp8_max
    a = (a_source / a_scale[..., None]).to(fp8).view(tokens, hidden_dim)
    del a_source
    a_scale *= 0.5 + torch.rand_like(a_scale)
    w1, w1_scale = block_quant(experts, 2 * inter_dim, hidden_dim, 0.2)
    w2, w2_scale = block_quant(experts, hidden_dim, inter_dim, 0.02)
    topk_ids, topk_weights = make_routes(tokens, topk, experts, ep, routing, generator)
    token, slot = (topk_ids < experts).nonzero(as_tuple=True)
    return {
        "a": a,
        "a_scale_t": a_scale.t().contiguous(),
        "w1_raw": w1,
        "w1": shuffle_weight(w1, layout=(16, 16)),
        "w1_scale": w1_scale,
        "w2_raw": w2,
        "w2": shuffle_weight(w2, layout=(16, 16)),
        "w2_scale": w2_scale,
        "topk_ids": topk_ids.cuda(),
        "topk_weights": topk_weights.cuda(),
        "expert_mask": (torch.arange(experts * ep) < experts).to(torch.int32).cuda() if ep > 1 else None,
        "routes": (token.cuda(), slot.cuda(), topk_ids[token, slot].long().cuda()),
        "shape": (tokens, inter_dim, hidden_dim, topk, experts, ep),
    }


def sort_routes(inputs, block_m):
    tokens, _, hidden_dim, _, experts, ep = inputs["shape"]
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, out = moe_sorting(
        inputs["topk_ids"], inputs["topk_weights"], experts * ep, hidden_dim, dtypes.bf16, block_m, inputs["expert_mask"]
    )
    return dict(
        sorted_ids=sorted_ids,
        sorted_weights=sorted_weights,
        sorted_expert_ids=sorted_expert_ids,
        num_valid_ids=num_valid_ids,
        out=out,
        block_m=block_m,
    )


def run_kernel(inputs, sorting, kernel):
    cfg = KERNELS[kernel]
    assert sorting["block_m"] == cfg["block_m"], "moe_sorting block_m must match the kernel's M tile"
    aiter.fmoe_fp8_blockscale_g1u1(
        sorting["out"],
        inputs["a"],
        inputs["w1"],
        inputs["w2"],
        sorting["sorted_ids"],
        sorting["sorted_weights"],
        sorting["sorted_expert_ids"],
        sorting["num_valid_ids"],
        inputs["shape"][3],
        inputs["a_scale_t"],
        inputs["w1_scale"],
        inputs["w2_scale"],
        cfg["knl_name"],
        fc_scale_blkn=BLK,
        fc_scale_blkk=BLK,
        activation=ActivationType.Silu,
        block_size_M=cfg["block_m"],
    )
    return sorting["out"]


def dequant_blocks(weight, scale):
    n, k = weight.shape
    return (weight.float().view(n // BLK, BLK, k // BLK, BLK) * scale[:, None, :, None]).view(n, k)


def reference(inputs):
    tokens, inter_dim, hidden_dim, topk, experts, ep = inputs["shape"]
    fp8, fp8_max = dtypes.fp8, torch.finfo(dtypes.fp8).max
    a = (inputs["a"].float().view(tokens, -1, BLK) * inputs["a_scale_t"].t()[:, :, None]).view(tokens, hidden_dim)
    token, slot, expert = inputs["routes"]
    weight = inputs["topk_weights"][token, slot]
    out = torch.zeros((tokens, hidden_dim), device="cuda")
    for e in range(experts):
        rows = (expert == e).nonzero().squeeze(1)
        if rows.numel() == 0:
            continue
        w1 = dequant_blocks(inputs["w1_raw"][e], inputs["w1_scale"][e])
        w2 = dequant_blocks(inputs["w2_raw"][e], inputs["w2_scale"][e])
        gate, up = (a[token[rows]] @ w1.t()).split(inter_dim, dim=-1)
        # The kernel feeds Down with the SiLU output re-quantized to FP8 per (row, 128).
        act = (F.silu(gate) * up).view(rows.numel(), -1, BLK)
        act_scale = act.abs().amax(dim=-1, keepdim=True).clamp_min(1e-10) / fp8_max
        act = ((act / act_scale).to(fp8).float() * act_scale).view(rows.numel(), inter_dim)
        out.index_add_(0, token[rows], act @ w2.t() * weight[rows, None])
    return out


def check(inputs, out, expected):
    token = inputs["routes"][0]
    routes_per_token = torch.bincount(token, minlength=out.shape[0])
    routed = routes_per_token > 0
    # Tokens without a local route keep moe_sorting's zero fill.
    assert (out[~routed] == 0).all(), "kernel wrote a token without local routes"
    actual, wanted = out[routed].float(), expected[routed]
    assert torch.isfinite(actual).all(), "non-finite or unwritten routed output"
    # Residual error is FP8 rounding ties plus bf16 atomic accumulation, which grows with the routes per token.
    rms = wanted.pow(2).mean().sqrt()
    routes = routes_per_token[routed].float()
    tol = 0.02 * wanted.abs() + 0.2 * rms * routes.sqrt()[:, None]
    excess = (actual - wanted).abs() - tol
    assert (excess <= 0).all(), f"{(excess > 0).sum().item()} elements off, worst by {excess.max().item():.3g} (rms {rms.item():.3g})"
    diff = pyhip.calc_diff(actual, wanted)
    assert diff < 1e-4 * routes.mean().sqrt().item(), f"calc_diff {diff}"
    return diff


SHAPES = [
    (256, 256, 512, 2, 4, 2, "random"),
    (1000, 512, 1024, 4, 8, 4, "random"),
    (2048, 256, 512, 8, 8, 1, "balanced"),
    (4096, 2048, 4096, 8, 32, 8, "ep_uniform"),
]


@functools.lru_cache(maxsize=len(SHAPES))
def cached_case(*shape):
    inputs = make_inputs(*shape)
    return inputs, reference(inputs)


@pytest.mark.parametrize("tokens,inter_dim,hidden_dim,topk,experts,ep,routing", SHAPES)
@pytest.mark.parametrize("kernel", list(KERNELS))
def test_fmoe(tokens, inter_dim, hidden_dim, topk, experts, ep, routing, kernel):
    if not is_gfx942():
        pytest.skip("gfx942 kernels")
    inputs, expected = cached_case(tokens, inter_dim, hidden_dim, topk, experts, ep, routing)
    sorting = sort_routes(inputs, KERNELS[kernel]["block_m"])
    for _ in range(2):
        sorting["out"].zero_()
        run_kernel(inputs, sorting, kernel)
        torch.cuda.synchronize()
        check(inputs, sorting["out"], expected)


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
    tokens, inter_dim, hidden_dim, topk, experts, ep = args.tokens, args.inter_dim, args.hidden_dim, args.topk, args.experts, args.ep
    inputs = make_inputs(tokens, inter_dim, hidden_dim, topk, experts, ep, args.routing, seed=args.seed)
    routes = inputs["routes"][0].numel()
    # GateUp 2 * routes * (2 * inter) * hidden + Down 2 * routes * hidden * inter.
    flops = 6 * routes * inter_dim * hidden_dim
    expected = reference(inputs) if args.check else None
    clones = [inputs] + [
        dict(inputs, **{name: inputs[name].clone() for name in ("a", "a_scale_t", "w1", "w1_scale", "w2", "w2_scale")})
        for _ in range(args.clones - 1)
    ]
    cus = torch.cuda.get_device_properties(0).multi_processor_count
    print(
        f"fmoe_fp8_blockscale_g1u1 T={tokens} I={inter_dim} K={hidden_dim} E={experts} EP={ep} topk={topk} {args.routing}: "
        f"routes={routes} ({routes / tokens:.2f}/token, {routes / experts:.0f}/expert) {flops / 1e12:.3f} TFLOP",
        flush=True,
    )
    sortings, rows = {}, []
    for kernel in [k for k in args.kernels if inter_dim % KERNELS[k]["tile_n"] == 0]:
        cfg = KERNELS[kernel]
        if cfg["block_m"] not in sortings:
            sortings[cfg["block_m"]] = sort_routes(inputs, cfg["block_m"])
        sorting = sortings[cfg["block_m"]]

        def call(index, sorting=sorting, kernel=kernel):
            run_kernel(clones[index % len(clones)], sorting, kernel)

        for index in range(len(clones)):
            call(index)
        samples = []
        for index in range(args.samples):
            sorting["out"].zero_()
            with pyhip.cudaPerf(flops, name=kernel, verbose=0) as perf:
                call(index)
            samples.append(perf.latencies[0] * 1e6)
        diff = check(inputs, sorting["out"], expected) if args.check else float("nan")
        median_us = statistics.median(samples)
        blocks = int(sorting["num_valid_ids"][0].item()) // cfg["block_m"]
        if cfg["ps_tgs_per_cu"]:
            grid = cfg["ps_tgs_per_cu"] * cus
        elif cfg["knl_name"]:
            grid = inter_dim // cfg["tile_n"] * sorting["sorted_expert_ids"].numel()
        else:
            grid = None
        row = dict(
            kernel=kernel,
            block_m=cfg["block_m"],
            blocks=blocks,
            grid=grid,
            median_us=median_us,
            median_tflops=flops / median_us / 1e6,
            best_tflops=flops / min(samples) / 1e6,
            calc_diff=diff,
        )
        if args.sustained:
            s = sustained(call, flops, args.sustained)
            row.update(sustained_tflops=s["tflops"], clock_mhz=s["clock_mhz"], power_w=s["power_w"], mfma_efficiency=s["mfma_efficiency"])
        rows.append(row)
        print(f"  {kernel:16} bm={cfg['block_m']:<3} median {median_us:8.1f} us {row['median_tflops']:6.1f} TF, best {row['best_tflops']:6.1f} TF", flush=True)
    df = pd.DataFrame(rows).sort_values("median_us")
    print(df.to_markdown(index=False, floatfmt=".4g"), flush=True)
    best = df.iloc[0]
    if "auto" in df["kernel"].values:
        auto_us = df.loc[df["kernel"] == "auto", "median_us"].item()
        print(f"best {best['kernel']} (block_m={best['block_m']}): {best['median_us']:.1f} us vs untuned auto {auto_us:.1f} us -> {auto_us / best['median_us']:.3f}x", flush=True)
    if args.json:
        report = dict(
            shape=dict(tokens=tokens, inter_dim=inter_dim, hidden_dim=hidden_dim, topk=topk, experts=experts, ep=ep, routing=args.routing),
            routes=routes,
            kernels=rows,
            best=dict(kernel=best["kernel"], knl_name=KERNELS[best["kernel"]]["knl_name"], block_m=int(best["block_m"])),
        )
        with open(args.json, "w") as f:
            json.dump(report, f)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=32768)
    parser.add_argument("--inter-dim", type=int, default=2048)
    parser.add_argument("--hidden-dim", type=int, default=4096)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--experts", type=int, default=32, help="local experts of this EP rank")
    parser.add_argument("--ep", type=int, default=8, help="EP size; global experts = experts * ep")
    parser.add_argument("--routing", default="ep_uniform", choices=["ep_uniform", "random", "balanced"])
    parser.add_argument("--kernels", nargs="+", default=list(KERNELS), choices=list(KERNELS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--clones", type=int, default=4)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--sustained", type=float, default=0, help="seconds of back-to-back launches with clock telemetry, per kernel")
    parser.add_argument("--json")
    benchmark(parser.parse_args())
