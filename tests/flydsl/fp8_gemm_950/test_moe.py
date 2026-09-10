import argparse

import torch

import flydsl.compiler as flyc
from aiter import dtypes
from aiter.fused_moe import moe_sorting
from aiter.ops.flydsl.moe_kernels import flydsl_moe_stage1
from aiter.ops.quant import (
    fused_dynamic_mxfp8_quant_moe_sort,
    per_1x32_mx_quant_hip,
)
from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4
from pyhip import cudaPerf

from test_moe_mxfp8_mxfp4_gateup_4w import (
    SORT_BLOCK_M,
    _convert_aiter_moe_scale,
    _permute_scale,
    compile_moe_gateup_4w,
)

A_INPUT_SCALE = 0.33
B_INPUT_SCALE = 0.2
AITER_TILE_M = 128
AITER_TILE_N = 256
AITER_TILE_K = 256


# python ./test_moe.py --tokens 49152  --gate-up-size 2048  --hidden-size 6144 --topk 8 --experts 384  --warmup 5 --iterations 30 --data-clones 15 --pyhip-group-size-m 8


def div_up(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def make_balanced_topk(
    tokens: int,
    topk: int,
    experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not 0 < topk <= experts:
        raise ValueError("topk must be in [1, experts]")
    num_routes = tokens * topk
    expert_permutation = torch.randperm(experts, device="cuda", dtype=torch.int64)
    route_indices = torch.arange(num_routes, device="cuda", dtype=torch.int64)
    topk_ids = (
        expert_permutation[route_indices % experts].to(torch.int32).view(tokens, topk)
    )
    topk_weights = torch.ones((tokens, topk), device="cuda", dtype=torch.float32)
    counts = torch.bincount(topk_ids.view(-1).to(torch.int64), minlength=experts)
    if int(counts.max() - counts.min()) > 1:
        raise AssertionError("routing is not balanced")
    if topk > 1:
        token_experts = topk_ids.sort(dim=1).values
        if not bool((token_experts[:, 1:] != token_experts[:, :-1]).all()):
            raise AssertionError("a token was routed to the same expert twice")
    return topk_ids, topk_weights, counts


def sort_routes(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    experts: int,
    hidden_size: int,
    block_m: int,
):
    sorted_ids, sorted_weights, expert_ids, valid_ids, _ = moe_sorting(
        topk_ids,
        topk_weights,
        experts,
        hidden_size,
        dtypes.bf16,
        block_m,
        accumulate=False,
    )
    num_sorted = int(valid_ids[0].item())
    if num_sorted % block_m != 0:
        raise AssertionError("sorted route count is not block aligned")
    num_expert_blocks = num_sorted // block_m
    return (
        sorted_ids[:num_sorted].contiguous(),
        sorted_weights[:num_sorted].contiguous(),
        expert_ids[:num_expert_blocks].contiguous(),
        valid_ids,
    )


def prepare_case(
    tokens: int,
    gate_up_size: int,
    hidden_size: int,
    topk: int,
    experts: int,
):
    if gate_up_size % 256 != 0:
        raise ValueError("gate_up_size must be a multiple of 256")
    if hidden_size < 512 or hidden_size % 256 != 0:
        raise ValueError("hidden_size must be a multiple of 256 and at least 512")

    topk_ids, topk_weights, route_counts = make_balanced_topk(tokens, topk, experts)
    aiter_routing = sort_routes(
        topk_ids,
        topk_weights,
        experts,
        hidden_size,
        AITER_TILE_M,
    )
    pyhip_routing = sort_routes(
        topk_ids,
        topk_weights,
        experts,
        hidden_size,
        SORT_BLOCK_M,
    )

    a_source = (
        torch.randn((tokens, hidden_size), device="cuda", dtype=torch.bfloat16)
        * A_INPUT_SCALE
    )
    a_aiter, aiter_scale_a = fused_dynamic_mxfp8_quant_moe_sort(
        a_source,
        sorted_ids=aiter_routing[0],
        num_valid_ids=aiter_routing[3],
        token_num=tokens,
        topk=topk,
        block_size=AITER_TILE_M,
        sorted_weights=aiter_routing[1],
    )
    a_pyhip, pyhip_scale_a_aiter = fused_dynamic_mxfp8_quant_moe_sort(
        a_source,
        sorted_ids=pyhip_routing[0],
        num_valid_ids=pyhip_routing[3],
        token_num=tokens,
        topk=topk,
        block_size=SORT_BLOCK_M,
        sorted_weights=pyhip_routing[1],
    )
    torch.testing.assert_close(a_aiter, a_pyhip, rtol=0, atol=0)
    del a_source, a_pyhip

    weight_source = (
        torch.randn(
            (experts * gate_up_size, hidden_size),
            device="cuda",
            dtype=torch.bfloat16,
        )
        * B_INPUT_SCALE
    )
    weight, scale_b_raw = per_1x32_mx_quant_hip(
        weight_source,
        quant_dtype=dtypes.fp4x2,
        scale_type=dtypes.fp8_e8m0,
        shuffle=False,
    )
    del weight_source
    weight = weight.view(experts, gate_up_size, hidden_size // 2)
    scale_b_raw = scale_b_raw.view(experts, gate_up_size, hidden_size // 32)

    aiter_weight = shuffle_weight_a16w4(weight, 16, True)
    aiter_scale_b = shuffle_scale_a16w4(
        scale_b_raw.view(experts * gate_up_size, hidden_size // 32),
        experts,
        True,
    )
    pyhip_scale_a = _convert_aiter_moe_scale(pyhip_scale_a_aiter)
    scale_b_rows_per_expert = div_up(gate_up_size, 256) * 256
    scale_b_padded = torch.full(
        (experts, scale_b_rows_per_expert, hidden_size // 32),
        127,
        device="cuda",
        dtype=torch.uint8,
    )
    scale_b_padded[:, :gate_up_size].copy_(scale_b_raw.view(torch.uint8))
    pyhip_scale_b = _permute_scale(
        scale_b_padded.view(experts * scale_b_rows_per_expert, hidden_size // 32)
    )

    return {
        "tokens": tokens,
        "gate_up_size": gate_up_size,
        "hidden_size": hidden_size,
        "topk": topk,
        "experts": experts,
        "route_counts": route_counts,
        "a": a_aiter,
        "weight": weight,
        "aiter_weight": aiter_weight,
        "aiter_scale_a": aiter_scale_a,
        "aiter_scale_b": aiter_scale_b,
        "aiter_sorted_ids": aiter_routing[0],
        "aiter_expert_ids": aiter_routing[2],
        "aiter_valid_ids": aiter_routing[3],
        "pyhip_scale_a": pyhip_scale_a,
        "pyhip_scale_b": pyhip_scale_b,
        "pyhip_sorted_ids": pyhip_routing[0],
        "pyhip_expert_ids": pyhip_routing[2],
        "pyhip_valid_ids": pyhip_routing[3],
    }


def make_aiter_args(data, *, clone: bool):
    def tensor(value):
        return value.clone() if clone else value

    weight = tensor(data["aiter_weight"])
    weight.is_shuffled = True
    return {
        "a": tensor(data["a"]),
        "w1": weight,
        "sorted_token_ids": tensor(data["aiter_sorted_ids"]),
        "sorted_expert_ids": tensor(data["aiter_expert_ids"]),
        "num_valid_ids": tensor(data["aiter_valid_ids"]),
        "out": torch.empty(
            (
                data["tokens"],
                data["topk"],
                data["gate_up_size"] // 2,
            ),
            device="cuda",
            dtype=dtypes.bf16,
        ),
        "w1_scale": tensor(data["aiter_scale_b"]),
        "a1_scale": tensor(data["aiter_scale_a"]),
    }


def run_aiter_stage1(args, data, xcd_swizzle: int) -> None:
    flydsl_moe_stage1(
        **args,
        topk=data["topk"],
        tile_m=AITER_TILE_M,
        tile_n=AITER_TILE_N,
        tile_k=AITER_TILE_K,
        a_dtype="fp8",
        b_dtype="fp4",
        out_dtype="bf16",
        act="situv2",
        gate_mode="interleave",
        use_async_copy=True,
        waves_per_eu=1,
        b_nt=0,
        xcd_swizzle=xcd_swizzle,
        swiglu_limit=7.0,
    )


def make_pyhip_args(data, *, clone: bool):
    def tensor(value):
        return value.clone() if clone else value

    output = torch.empty(
        (
            data["tokens"],
            data["topk"],
            data["gate_up_size"] // 2,
        ),
        device="cuda",
        dtype=dtypes.bf16,
    )
    return (
        tensor(data["a"]).view(torch.int8).view(-1),
        tensor(data["weight"]).view(torch.int8),
        tensor(data["pyhip_scale_a"]),
        tensor(data["pyhip_scale_b"]),
        output.view(-1),
        tensor(data["pyhip_sorted_ids"]),
        tensor(data["pyhip_expert_ids"]),
        tensor(data["pyhip_valid_ids"]),
        data["tokens"],
        data["pyhip_expert_ids"].numel(),
        torch.cuda.current_stream(),
    )


def tensor_bytes(values) -> int:
    if isinstance(values, dict):
        values = values.values()
    return sum(
        value.numel() * value.element_size()
        for value in values
        if isinstance(value, torch.Tensor)
    )


def benchmark(
    name: str,
    run,
    arg_sets,
    flops: int,
    rw_bytes: int,
    warmup: int,
    iterations: int,
):
    for iteration in range(warmup):
        run(arg_sets[iteration % len(arg_sets)])
    torch.cuda.synchronize()

    samples = []
    for iteration in range(iterations):
        clone_index = (warmup + iteration) % len(arg_sets)
        with cudaPerf(
            flops,
            rw_bytes,
            name=f"{name}_{clone_index}",
            verbose=0,
        ) as perf:
            run(arg_sets[clone_index])
        samples.append((perf.dt() * 1.0e6, perf.tflops(), perf.bw()))
    samples.sort(key=lambda sample: sample[0])
    return samples[0], samples[len(samples) // 2]


def run_case(
    tokens: int,
    gate_up_size: int,
    hidden_size: int,
    topk: int,
    experts: int,
    warmup: int,
    iterations: int,
    data_clones: int,
    aiter_xcd_swizzle: int,
    pyhip_xcd_swizzle: bool,
    pyhip_group_size_m: int,
):
    data = prepare_case(tokens, gate_up_size, hidden_size, topk, experts)
    route_counts = data["route_counts"]
    print(
        f"routing: routes={tokens * topk} experts={experts} "
        f"min={int(route_counts.min())} max={int(route_counts.max())} "
        f"aiter_blocks={data['aiter_expert_ids'].numel()} "
        f"pyhip_blocks={data['pyhip_expert_ids'].numel()}"
    )

    pyhip_launcher = compile_moe_gateup_4w(
        gate_up_size // 2,
        hidden_size,
        topk,
        experts,
        xcd_swizzle=pyhip_xcd_swizzle,
        group_size_m=pyhip_group_size_m,
    )
    aiter_check_args = make_aiter_args(data, clone=False)
    pyhip_check_args = make_pyhip_args(data, clone=False)
    pyhip_kernel = flyc.compile[{"opt_level": 2}](pyhip_launcher, *pyhip_check_args)
    run_aiter_stage1(aiter_check_args, data, aiter_xcd_swizzle)
    pyhip_kernel(*pyhip_check_args)
    torch.cuda.synchronize()

    aiter_output = aiter_check_args["out"]
    pyhip_output = pyhip_check_args[4].view_as(aiter_output)
    aiter_finite = bool(torch.isfinite(aiter_output).all())
    pyhip_finite = bool(torch.isfinite(pyhip_output).all())
    max_abs = (aiter_output.float() - pyhip_output.float()).abs().max().item()
    print(
        f"output: aiter_finite={aiter_finite} pyhip_finite={pyhip_finite} "
        f"max_abs={max_abs:.6g} "
        "semantics=aiter_silu_mul_vs_pyhip_situv2"
    )
    if not (aiter_finite and pyhip_finite):
        raise AssertionError("stage1 produced a non-finite output")

    pyhip_sorted_tokens = int(data["pyhip_valid_ids"][0].item())
    aiter_sorted_tokens = int(data["aiter_valid_ids"][0].item())
    pyhip_flops = 2 * pyhip_sorted_tokens * gate_up_size * hidden_size
    aiter_flops = 2 * aiter_sorted_tokens * gate_up_size * hidden_size

    def nominal_rw_bytes(sorted_tokens: int) -> int:
        input_bytes = sorted_tokens * hidden_size
        weight_bytes = experts * gate_up_size * hidden_size // 2
        scale_bytes = (
            sorted_tokens * hidden_size + experts * gate_up_size * hidden_size
        ) // 32
        output_bytes = tokens * topk * gate_up_size
        return input_bytes + weight_bytes + scale_bytes + output_bytes

    aiter_arg_sets = [make_aiter_args(data, clone=True) for _ in range(data_clones)]
    aiter_best, aiter_median = benchmark(
        "aiter_stage1",
        lambda args: run_aiter_stage1(args, data, aiter_xcd_swizzle),
        aiter_arg_sets,
        aiter_flops,
        nominal_rw_bytes(aiter_sorted_tokens),
        warmup,
        iterations,
    )
    del aiter_arg_sets
    torch.cuda.empty_cache()

    pyhip_arg_sets = [make_pyhip_args(data, clone=True) for _ in range(data_clones)]
    pyhip_best, pyhip_median = benchmark(
        "pyhip_a8w4_stage1",
        lambda args: pyhip_kernel(*args),
        pyhip_arg_sets,
        pyhip_flops,
        nominal_rw_bytes(pyhip_sorted_tokens),
        warmup,
        iterations,
    )
    del pyhip_arg_sets
    torch.cuda.empty_cache()

    print(
        f"aiter: best={aiter_best[0]:.3f} us "
        f"median={aiter_median[0]:.3f} us "
        f"best={aiter_best[1]:.2f} TFLOPS "
        f"median={aiter_median[1]:.2f} TFLOPS "
        f"best_bw={aiter_best[2]:.2f} GB/s "
        f"median_bw={aiter_median[2]:.2f} GB/s"
    )
    print(
        f"pyhip: best={pyhip_best[0]:.3f} us "
        f"median={pyhip_median[0]:.3f} us "
        f"best={pyhip_best[1]:.2f} TFLOPS "
        f"median={pyhip_median[1]:.2f} TFLOPS "
        f"best_bw={pyhip_best[2]:.2f} GB/s "
        f"median_bw={pyhip_median[2]:.2f} GB/s"
    )
    print(
        f"ratio: latency={pyhip_median[0] / aiter_median[0]:.3f}x "
        f"throughput={pyhip_median[1] / aiter_median[1]:.3%}"
    )
    return aiter_median, pyhip_median, aiter_best, pyhip_best


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare AIter and PyHIP A8W4 MoE stage1 kernels"
    )
    parser.add_argument(
        "--tokens",
        type=int,
        nargs="+",
        default=(8192, 16384, 24576, 12288, 24576, 49152),
    )
    parser.add_argument("--gate-up-size", type=int, default=512)
    parser.add_argument("--hidden-size", type=int, default=6144)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--experts", type=int, default=384)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--data-clones", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--aiter-xcd-swizzle", type=int, default=8)
    parser.add_argument("--pyhip-group-size-m", type=int, default=4)
    parser.add_argument(
        "--no-pyhip-xcd-swizzle",
        action="store_false",
        dest="pyhip_xcd_swizzle",
    )
    parser.set_defaults(pyhip_xcd_swizzle=True)
    args = parser.parse_args()
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.iterations <= 0:
        parser.error("--iterations must be positive")
    if args.data_clones <= 0:
        parser.error("--data-clones must be positive")
    if args.pyhip_group_size_m <= 0:
        parser.error("--pyhip-group-size-m must be positive")

    props = torch.cuda.get_device_properties()
    if "950" not in props.gcnArchName:
        raise RuntimeError("A8W4 scaled MFMA requires gfx950")
    torch.manual_seed(args.seed)

    results = []
    for tokens in args.tokens:
        print(
            f"\ncase: M={tokens} N={args.gate_up_size} "
            f"K={args.hidden_size} topk={args.topk} experts={args.experts} "
            f"clones={args.data_clones} warmup={args.warmup} "
            f"iterations={args.iterations}"
        )
        aiter_median, pyhip_median, aiter_best, pyhip_best = run_case(
            tokens,
            args.gate_up_size,
            args.hidden_size,
            args.topk,
            args.experts,
            args.warmup,
            args.iterations,
            args.data_clones,
            args.aiter_xcd_swizzle,
            args.pyhip_xcd_swizzle,
            args.pyhip_group_size_m,
        )
        results.append((tokens, aiter_median, pyhip_median, aiter_best, pyhip_best))
        torch.cuda.empty_cache()

    print("\nmedian summary")
    for tokens, aiter_median, pyhip_median, aiter_best, pyhip_best in results:
        print(
            f"M={tokens:5d} N={args.gate_up_size} K={args.hidden_size:5d} "
            f"aiter={aiter_median[0]:8.3f} us "
            f"pyhip={pyhip_median[0]:8.3f} us "
            f"pyhip/aiter={pyhip_median[0] / aiter_median[0]:.3f}x "
            f"aiter_best={aiter_best[0]:8.3f} us "
            f"pyhip_best={pyhip_best[0]:8.3f} us "
            f"pyhip_best/aiter_best={pyhip_best[0] / aiter_best[0]:.3f}x"
        )


if __name__ == "__main__":
    main()
