# SPDX-License-Identifier: MIT

import argparse

import aiter
import torch
from aiter import dtypes
from aiter.fused_moe import moe_sorting
from aiter.ops.shuffle import shuffle_weight

import pyhip
from pyhip.contrib.moe_gemm_8wave import moe_gemm_8wave_down

from moe_gemm_8wave_down_flydsl_persistent_mfma import PersistentFlyDSLDownMFMA


torch.set_default_device("cuda")

ACTIVATION_QUANT = aiter.get_hip_quant(aiter.QuantType.per_1x128)


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


def run_test(tokens, model_dim, inter_dim, experts, topk, block_m, num_oc_splits, seed):
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
    rw_bytes = valid_eblocks * (model_dim * inter_dim) + \
                input_q.numel() * input_q.element_size() + \
                actual.numel() * actual.element_size() 

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
        num_bytes = rw_bytes,
        num_name="moe_gemm_8wave_down",
        num_spec_tag=f"M={tokens * topk},N={model_dim},K={inter_dim}",
    )
    torch.cuda.synchronize()

    mfma_actual = torch.full_like(reference, torch.nan)
    mfma_counter = torch.zeros(1, dtype=torch.uint32)
    mfma_launch = PersistentFlyDSLDownMFMA(
        mfma_actual,
        input_q,
        weight_shuffled,
        input_scales,
        weight_scales,
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        mfma_counter,
        num_tokens=tokens,
        n=model_dim,
        k=inter_dim,
        topk=topk,
        num_oc_splits=num_oc_splits,
    )
    mfma_actual, mfma_elapsed_us = pyhip.run_perftest(
        mfma_launch,
        num_warmup=2,
        num_iters=10,
        num_copies=1,
        num_flops=flops2,
        num_verbose=1,
        num_bytes=rw_bytes,
        num_name="moe_gemm_8wave_down_flydsl_persistent_mfma",
        num_spec_tag=f"M={tokens * topk},N={model_dim},K={inter_dim}",
    )
    torch.cuda.synchronize()

    ref_f32 = reference.float()
    actual_f32 = actual.float()
    abs_error = (actual_f32 - ref_f32).abs()
    threshold = 1.0e-2 + 1.0e-2 * ref_f32.abs()
    mismatch = abs_error > threshold
    diff = pyhip.calc_diff(ref_f32, actual_f32)
    mfma_f32 = mfma_actual.float()
    mfma_abs_error = (mfma_f32 - ref_f32).abs()
    mfma_mismatch = mfma_abs_error > threshold
    mfma_diff = pyhip.calc_diff(ref_f32, mfma_f32)

    print(f"shape: tokens={tokens}, topk={topk}, experts={experts}, N={model_dim}, K={inter_dim}")
    print(f"expert blocks: {sorted_expert_ids.numel()}, OC splits: {num_oc_splits}")
    print(f"persistent workgroups: {num_cus}, claimed tasks: {blk_atomic_int.item()}")
    print(f"performance: {elapsed_us:.3f} us, {flops1 / elapsed_us / 1e6:.3f} / {flops2 / elapsed_us / 1e6:.3f} TFLOPS")
    print(
        f"FlyDSL persistent MFMA performance: {mfma_elapsed_us:.3f} us, "
        f"{flops1 / mfma_elapsed_us / 1e6:.3f} / {flops2 / mfma_elapsed_us / 1e6:.3f} TFLOPS, "
        f"speedup vs PyHIP: {elapsed_us / mfma_elapsed_us:.3f}x"
    )
    print(f"max abs error: {abs_error.max().item():.6g}")
    print(f"mean abs error: {abs_error.mean().item():.6g}")
    print(f"calc_diff: {diff:.6g}")
    print(f"mismatches (rtol=1e-2, atol=1e-2): {mismatch.sum().item()} / {mismatch.numel()}")
    print(f"FlyDSL persistent MFMA max abs error: {mfma_abs_error.max().item():.6g}")
    print(f"FlyDSL persistent MFMA calc_diff: {mfma_diff:.6g}")
    print(f"FlyDSL persistent MFMA mismatches: {mfma_mismatch.sum().item()} / {mfma_mismatch.numel()}")

    if torch.isnan(actual_f32).any():
        raise AssertionError("kernel left NaNs in the output")
    torch.testing.assert_close(actual_f32, ref_f32, rtol=1.0e-2, atol=1.0e-2)
    if torch.isnan(mfma_f32).any():
        raise AssertionError("FlyDSL persistent MFMA kernel left NaNs in the output")
    torch.testing.assert_close(mfma_f32, ref_f32, rtol=1.0e-2, atol=1.0e-2)
    print("PASS: moe_gemm_8wave_down matches the Torch block-scale reference")


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
