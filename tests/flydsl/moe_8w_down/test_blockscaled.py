# SPDX-License-Identifier: MIT

import argparse

import aiter
import torch
from aiter import dtypes
from aiter.fused_moe import moe_sorting
from aiter.ops.shuffle import shuffle_weight

import pyhip
from pyhip.contrib.moe_gemm_8wave import moe_gemm_8wave_down

from moe_8wave_down import flydsl_moe_gemm_8wave_down


torch.set_default_device("cuda")

ACTIVATION_QUANT = aiter.get_hip_quant(aiter.QuantType.per_1x128)


def print_markdown_table(headers, rows):
    """Print a compact Markdown table without an additional dependency."""
    widths = [len(header) for header in headers]
    for row in rows:
        for column, value in enumerate(row):
            widths[column] = max(widths[column], len(str(value)))

    def format_row(row):
        return "| " + " | ".join(
            str(value).ljust(widths[column])
            for column, value in enumerate(row)
        ) + " |"

    print(format_row(headers))
    print(format_row(["-" * width for width in widths]))
    for row in rows:
        print(format_row(row))


def make_routing(tokens, topk, experts, seed):
    """Create random top-k IDs and normalized routing weights."""
    assert topk <= experts
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)

    # Random scores followed by topk gives independent random routing per token
    # while preserving the real router invariant that expert IDs are unique.
    scores = torch.rand(tokens, experts, generator=generator, dtype=torch.float32)
    topk_ids = torch.topk(scores, topk, dim=-1, sorted=False).indices.to(torch.int32)
    topk_weights = torch.rand(tokens, topk, generator=generator, dtype=torch.float32)
    topk_weights /= topk_weights.sum(dim=1, keepdim=True)
    return topk_ids, topk_weights


def torch_reference_down(
    input_q,
    input_scales_k_major,
    weight_q,
    weight_scales,
    topk_ids,
    topk_weights,
):
    """Block-scale FP8 MoE down reference matching the kernel output layout."""
    tokens, topk, k = input_q.shape
    experts, n, weight_k = weight_q.shape
    assert weight_k == k

    rows = tokens * topk
    input_blocks = input_q.float().reshape(rows, k // 128, 128)
    # transpose_scale=True keeps the logical tensor shape but stores scales in
    # K-major order; recover the row-major [tokens*topk, K/128] view.
    input_scales = input_scales_k_major.view(k // 128, rows).t().float()
    output = torch.empty(rows, n, dtype=torch.bfloat16)
    expert_per_row = topk_ids.reshape(-1)
    routing_per_row = topk_weights.reshape(-1)

    for expert in range(experts):
        row_ids = torch.where(expert_per_row == expert)[0]
        if row_ids.numel() == 0:
            continue

        accum = torch.zeros(row_ids.numel(), n, dtype=torch.float32)
        for bk in range(k // 128):
            a = input_blocks[row_ids, bk, :]
            for bn in range(n // 128):
                w = weight_q[expert, bn * 128 : (bn + 1) * 128, bk * 128 : (bk + 1) * 128]
                partial = a @ w.float().t()
                scale = input_scales[row_ids, bk, None] * weight_scales[expert, bn, bk]
                accum[:, bn * 128 : (bn + 1) * 128] += partial * scale

        # Kernel multiplies in FP32 and then converts each result to BF16.
        output[row_ids] = (accum * routing_per_row[row_ids, None]).to(torch.bfloat16)

    return output.reshape(tokens, topk, n)


def run_test(
    tokens,
    model_dim,
    inter_dim,
    experts,
    topk,
    block_m,
    num_oc_splits,
    seed,
):
    assert model_dim % 128 == 0
    assert inter_dim % 128 == 0
    assert model_dim % num_oc_splits == 0
    assert (model_dim // num_oc_splits) % 64 == 0
    assert (model_dim // num_oc_splits) // 64 >= 3

    torch.manual_seed(seed)
    # Match the real fused-MoE path: Aiter's HIP FP8 quantizers consume BF16.
    input_bf16 = torch.randn(tokens, topk, inter_dim, dtype=torch.bfloat16)
    weight_bf16 = torch.randn(experts, model_dim, inter_dim, dtype=torch.bfloat16)

    input_q, input_scales = ACTIVATION_QUANT(
        input_bf16,
        quant_dtype=dtypes.fp8,
        transpose_scale=True,
    )

    # Re-layout each 128x128 weight tile as one row, then use Aiter's existing
    # per-token quantizer so each row receives exactly one FP8 block scale.
    weight_blocks = weight_bf16.view(
        experts, model_dim // 128, 128, inter_dim // 128, 128
    ).permute(0, 1, 3, 2, 4).contiguous()
    weight_q_blocks, weight_scales = aiter.pertoken_quant(
        weight_blocks.view(experts, -1, 128 * 128),
        quant_dtype=dtypes.fp8,
    )
    weight_q = weight_q_blocks.view(
        experts, model_dim // 128, inter_dim // 128, 128, 128
    ).permute(0, 1, 3, 2, 4).contiguous().view(experts, model_dim, inter_dim)
    weight_scales = weight_scales.view(
        experts, model_dim // 128, inter_dim // 128
    )
    weight_shuffled = shuffle_weight(weight_q, layout=(16, 16))
    topk_ids, topk_weights = make_routing(tokens, topk, experts, seed + 1)
    (
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        _,
    ) = moe_sorting(
        topk_ids,
        topk_weights,
        experts,
        model_dim,
        torch.bfloat16,
        block_m,
        None,
        None,
        0,
    )

    reference = torch_reference_down(
        input_q,
        input_scales,
        weight_q,
        weight_scales,
        topk_ids,
        topk_weights,
    )

    actual = torch.full_like(reference, torch.nan)
    blk_atomic_int = torch.zeros(1, dtype=torch.uint32)
    num_cus = torch.cuda.get_device_properties().multi_processor_count

    def launch(*args, **kwargs):
        blk_atomic_int.zero_()
        moe_gemm_8wave_down(*args, blk_atomic_int)

    flops1 = 2 * tokens * topk * model_dim * inter_dim
    valid_eblocks = num_valid_ids[0].item() // block_m
    flops2 = valid_eblocks * block_m * model_dim * inter_dim * 2
    rw_bytes = (
        valid_eblocks * model_dim * inter_dim
        + input_q.numel() * input_q.element_size()
        + actual.numel() * actual.element_size()
    )

    candidate_errors = {}
    elapsed_us = None
    try:
        _, elapsed_us = pyhip.run_perftest(
            launch,
            [num_cus],
            [8 * 64],
            actual.numel() * actual.element_size() > (1 << 32),
            "fp8",
            block_m,
            64,
            experts,
            model_dim,
            inter_dim,
            num_oc_splits,
            False,
            True,
            topk,
            sorted_ids.data_ptr(),
            sorted_weights.data_ptr(),
            sorted_expert_ids.data_ptr(),
            num_valid_ids.data_ptr(),
            weight_shuffled.data_ptr(),
            weight_scales.data_ptr(),
            input_q.data_ptr(),
            input_scales.data_ptr(),
            actual.data_ptr(),
            tokens,
            num_warmup=2,
            num_iters=10,
            num_copies=1,
            num_flops=flops2,
            num_verbose=1,
            num_bytes=rw_bytes,
            num_name="moe_gemm_8wave_down",
            num_spec_tag=f"M={tokens * topk},N={model_dim},K={inter_dim}",
        )
    except Exception as e:
        candidate_errors["PyHIP"] = str(e)
    else:
        torch.cuda.synchronize()

    flydsl_runs = {}
    for block_n in (32, 64):
        name = f"FlyDSL BN{block_n}"
        output = torch.full_like(reference, torch.nan)
        counter = torch.zeros(1, dtype=torch.int32)
        candidate_us = None
        try:
            flydsl_launch = flydsl_moe_gemm_8wave_down(
                n=model_dim,
                k=inter_dim,
                topk=topk,
                num_experts=experts,
                block_n=block_n,
                num_oc_splits=num_oc_splits,
            )
            output, candidate_us = pyhip.run_perftest(
                flydsl_launch,
                output,
                input_q,
                weight_shuffled,
                input_scales,
                weight_scales,
                sorted_ids,
                sorted_weights,
                sorted_expert_ids,
                num_valid_ids,
                counter,
                num_warmup=2,
                num_iters=10,
                num_copies=1,
                num_flops=flops2,
                num_verbose=1,
                num_bytes=rw_bytes,
                num_name=f"moe_gemm_8wave_down_flydsl_bn{block_n}",
                num_spec_tag=f"M={tokens * topk},N={model_dim},K={inter_dim}",
            )
        except Exception as e:
            candidate_errors[name] = str(e)
        else:
            torch.cuda.synchronize()
        flydsl_runs[name] = (block_n, output, candidate_us)

    ref_f32 = reference.float()
    threshold = 1.0e-2 + 1.0e-2 * ref_f32.abs()

    def make_result(name, block_n, output, candidate_us):
        if candidate_us is None:
            return {
                "name": name,
                "block_n": block_n,
                "status": "JIT FAILED",
                "us": "N/A",
                "effective_tflops": "N/A",
                "padded_tflops": "N/A",
                "tb_per_s": "N/A",
                "max_abs": "N/A",
                "mean_abs": "N/A",
                "diff": "N/A",
                "mismatches": "N/A",
                "failed": False,
            }

        output_f32 = output.float()
        abs_error = (output_f32 - ref_f32).abs()
        mismatch_count = (abs_error > threshold).sum().item()
        has_nan = torch.isnan(output_f32).any().item()
        return {
            "name": name,
            "block_n": block_n,
            "status": "FAIL" if has_nan or mismatch_count else "PASS",
            "us": f"{candidate_us:.3f}",
            "effective_tflops": f"{flops1 / candidate_us / 1e6:.3f}",
            "padded_tflops": f"{flops2 / candidate_us / 1e6:.3f}",
            "tb_per_s": f"{rw_bytes / candidate_us / 1e6:.3f}",
            "max_abs": f"{abs_error.max().item():.6g}",
            "mean_abs": f"{abs_error.mean().item():.6g}",
            "diff": f"{pyhip.calc_diff(ref_f32, output_f32):.6g}",
            "mismatches": f"{mismatch_count}/{abs_error.numel()}",
            "failed": has_nan or mismatch_count > 0,
        }

    results = [
        make_result("PyHIP", 64, actual, elapsed_us),
    ]
    results.extend(
        make_result(name, block_n, output, candidate_us)
        for name, (block_n, output, candidate_us) in flydsl_runs.items()
    )

    if elapsed_us is not None:
        speedups = {
            result["name"]: (
                "1.000x"
                if result["name"] == "PyHIP"
                else (
                    f"{elapsed_us / flydsl_runs[result['name']][2]:.3f}x"
                    if flydsl_runs[result["name"]][2] is not None
                    else "N/A"
                )
            )
            for result in results
        }
    else:
        speedups = {result["name"]: "N/A" for result in results}

    print(
        f"\nShape: tokens={tokens}, topk={topk}, experts={experts}, "
        f"N={model_dim}, K={inter_dim}"
    )
    print(
        f"Config: expert_blocks={sorted_expert_ids.numel()}, "
        f"oc_splits={num_oc_splits}, persistent_workgroups={num_cus}"
    )
    print("Tolerance: rtol=1e-2, atol=1e-2\n")
    headers = ["Metric", "PyHIP", "BN32", "BN64"]
    rows = [
        ["Status", *(result["status"] for result in results)],
        ["Time (us)", *(result["us"] for result in results)],
        ["Effective TF/s", *(result["effective_tflops"] for result in results)],
        ["Padded TF/s", *(result["padded_tflops"] for result in results)],
        ["TB/s", *(result["tb_per_s"] for result in results)],
        ["Speedup vs PyHIP", *(speedups[result["name"]] for result in results)],
        ["Max abs error", *(result["max_abs"] for result in results)],
        ["Mean abs error", *(result["mean_abs"] for result in results)],
        ["calc_diff", *(result["diff"] for result in results)],
        ["Mismatches", *(result["mismatches"] for result in results)],
    ]
    print_markdown_table(headers, rows)

    if candidate_errors:
        print("\nJIT failures:")
        for name, error in candidate_errors.items():
            summary = error.strip().splitlines()[-1] if error.strip() else "unknown error"
            print(f"- {name}: {summary}")

    failed_results = [result["name"] for result in results if result["failed"]]
    if failed_results:
        raise AssertionError(
            "correctness check failed for: " + ", ".join(failed_results)
        )
    if candidate_errors:
        print("\nCompleted with unavailable JIT candidate(s).")
    else:
        print("\nPASS: all kernels match the Torch block-scale reference")


def main():
    parser = argparse.ArgumentParser(description="Compare moe_gemm_8wave_down with Torch")
    parser.add_argument("--tokens", type=int, default=16384)
    parser.add_argument("--model-dim", type=int, default=6144)
    parser.add_argument("--inter-dim", type=int, default=256)
    parser.add_argument("--experts", type=int, default=384)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--block-m", type=int, default=256)
    parser.add_argument("--num-oc-splits", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    run_test(
        args.tokens,
        args.model_dim,
        args.inter_dim,
        args.experts,
        args.topk,
        args.block_m,
        args.num_oc_splits,
        args.seed,
    )


if __name__ == "__main__":
    main()
