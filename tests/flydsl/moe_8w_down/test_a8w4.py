# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL MOE a8w4 (fp8 activation, fp4 weight, GUI shuffle) regression tests.

Covers stage2 tile_k auto-resolve for non-256-aligned inter_dim (e.g. DSV4
inter=640) and FlyDSL stage2 / E2E with GUI preshuffle on gfx950.

Usage:
    pytest op_tests/flydsl_tests/test_flydsl_moe_a8w4.py -q
    pytest op_tests/flydsl_tests/test_flydsl_moe_a8w4.py -k tile_k
"""

from __future__ import annotations

import os
os.environ.setdefault("FLYDSL_RUNTIME_ENABLE_CACHE", "0")

import pytest
import torch
import re

from aiter.ops.opus import moe_stage2_a8w4_fused_adapter as _opus_a8w4
from aiter.ops.opus.moe_stage2_a8w4_meta import (
    OPUS_A8W4_KID_ROUTE_BF16_BM32_FULL_N7168_SMALL,
    OPUS_A8W4_KID_ROUTE_FP8_BM64_RBN3072,
    opus_a8w4_kid_reduce_block_n,
)

from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import fused_topk, moe_sorting, torch_moe_stage1, torch_moe_stage2
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.moe_kernels import (
    pick_flydsl_stage2_tile_k,
    resolve_flydsl_stage2_tile_k,
)
from aiter.ops.flydsl.utils import is_flydsl_available
from aiter.ops.quant import (
    mxfp4_moe_sort_fwd,
    per_1x32_f4_quant,
    per_1x32_f8_scale_f8_quant,
)
from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight, shuffle_weight_a16w4
from aiter.test_common import checkAllclose
from aiter.utility.fp4_utils import e8m0_shuffle

import pyhip
from moe_8wave_down_a8w4 import flydsl_moe_gemm_8wave_down_a8w4

Q_TYPE = QuantType.per_1x32

_SKIP_GFX950_FLYDSL = pytest.mark.skipif(
    get_gfx() not in ("gfx950",) or not is_flydsl_available(),
    reason="gfx950 FlyDSL required",
)


def _inter_pad(inter_dim: int) -> int:
    return ((inter_dim + 255) // 256 * 256) - inter_dim


def _stage1_tile_k(model_dim: int) -> int:
    return 512 if (model_dim % 512 == 0) else 256


def _check_close(ref, out, label, atol=1.0, rtol=0.05, max_err_ratio=0.05):
    if out.isnan().any():
        return f"NaN!!!"
    if out.isinf().any():
        return f"Inf!!!"
    err = checkAllclose(ref, out, msg=label, atol=atol, rtol=rtol)
    if(err > max_err_ratio):
        return f"{err}!!!"
    return f"{err:6g}!!!"

def _print_stage2_results(rows):
    headers = ("Kernel", "Time (us)", "TFLOPS", "vs FlyDSL", "Error ratio", "Diff")
    widths = [len(header) for header in headers]
    formatted_rows = []
    for name, elapsed_us, tflops, improve, err, diff in rows:
        row = (
            name,
            f"{elapsed_us:.3f}" if elapsed_us is not None else "N/A",
            f"{tflops/elapsed_us:.3f}" if elapsed_us is not None else "N/A",
            f"{improve:.3f}x" if improve is not None else "N/A",
            f"{err}" if err is not None else "N/A",
            f"{diff:.6g}" if diff is not None else "N/A",
        )
        formatted_rows.append(row)
        widths = [max(width, len(value)) for width, value in zip(widths, row)]

    def format_row(row):
        return "| " + " | ".join(
            value.ljust(width) for value, width in zip(row, widths)
        ) + " |"

    print("\nA8W4 stage2 comparison:")
    print(format_row(headers))
    print(format_row(tuple("-" * width for width in widths)))
    for row in formatted_rows:
        print(format_row(row))


def _generate_a8w4_gui_data(
    token: int,
    model_dim: int,
    inter_dim: int,
    E: int,
    topk: int,
    block_m: int,
    seed: int = 0,
    dtype=torch.bfloat16,
    inter_pad_override: int | None = None,
):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    inter_pad = (
        _inter_pad(inter_dim)
        if inter_pad_override is None
        else int(inter_pad_override)
    )

    inp = torch.randn(token, model_dim, dtype=dtype, device="cuda") / 4
    w1 = torch.randn(E, inter_dim * 2, model_dim, dtype=dtype, device="cuda") / 4
    w2 = torch.randn(E, model_dim, inter_dim, dtype=dtype, device="cuda") / 4
    if inter_pad:
        w1[:, -inter_pad:, :] = 0
        w1[:, inter_dim - inter_pad : inter_dim, :] = 0
        w2[:, :, -inter_pad:] = 0

    score = torch.randn(token, E, dtype=dtype, device="cuda")
    topk_weights, topk_ids = fused_topk(inp, score, topk, True)
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, _ = moe_sorting(
        topk_ids, topk_weights, E, model_dim, dtype, block_m
    )

    a_q, a_scale = per_1x32_f8_scale_f8_quant(
        inp, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0
    )
    w1_q, w1_scale = per_1x32_f4_quant(w1, quant_dtype=dtypes.fp4x2)
    w2_q, w2_scale = per_1x32_f4_quant(w2, quant_dtype=dtypes.fp4x2)
    w1_q = w1_q.view(E, inter_dim * 2, model_dim // 2)
    w2_q = w2_q.view(E, model_dim, inter_dim // 2)

    ref_stage1 = torch_moe_stage1(
        a_q,
        w1_q,
        w2_q,
        topk_weights,
        topk_ids,
        dtype=dtype,
        activation=ActivationType.Swiglu,
        quant_type=Q_TYPE,
        a1_scale=a_scale,
        w1_scale=w1_scale,
    )
    ref_stage2 = torch_moe_stage2(
        ref_stage1,
        w1_q,
        w2_q,
        topk_weights,
        topk_ids,
        dtype=dtype,
        quant_type=Q_TYPE,
        w2_scale=w2_scale,
        a2_scale=None,
        doweight=True,
    )

    a2_q, a2_scale = per_1x32_f8_scale_f8_quant(
        ref_stage1, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0
    )
    a2_q = a2_q.view(token, topk, inter_dim)

    a_scale_sort = mxfp4_moe_sort_fwd(
        a_scale,
        sorted_ids=sorted_ids,
        num_valid_ids=num_valid_ids,
        token_num=token,
        cols=model_dim,
    )
    w1_shuf = shuffle_weight_a16w4(w1_q, 16, True)
    w1_scale_shuf = shuffle_scale_a16w4(w1_scale, E, True)
    w2_shuf = shuffle_weight_a16w4(w2_q, 16, False)
    w2_scale_shuf = shuffle_scale_a16w4(w2_scale, E, False)
    a2_scale_sort = mxfp4_moe_sort_fwd(
        a2_scale,
        sorted_ids=sorted_ids,
        num_valid_ids=num_valid_ids,
        token_num=token,
        cols=inter_dim,
    )

    return {
        "inter_pad": inter_pad,
        "topk": topk,
        "a_q": a_q,
        "a_scale_sort": a_scale_sort,
        "w1_shuf": w1_shuf,
        "w1_scale_shuf": w1_scale_shuf,
        "w2_shuf": w2_shuf,
        "w2_scale_shuf": w2_scale_shuf,
        "a2_q": a2_q,
        "a2_scale_sort": a2_scale_sort,
        "sorted_ids": sorted_ids,
        "sorted_weights": sorted_weights,
        "sorted_expert_ids": sorted_expert_ids,
        "num_valid_ids": num_valid_ids,
        "topk_ids": topk_ids,
        "topk_weights": topk_weights,
        "a2_scale": a2_scale,
        "ref_stage1": ref_stage1,
        "ref_stage2": ref_stage2,
        "token": token,
        "inter_dim": inter_dim,
        "model_dim": model_dim,
    }


@pytest.fixture(autouse=True)
def _a8w4_env():
    old_bound = os.environ.get("AITER_BF16_FP8_MOE_BOUND")
    old_aot = os.environ.get("FLYDSL_RUNTIME_RUN_ONLY")
    os.environ["AITER_BF16_FP8_MOE_BOUND"] = "0"
    os.environ.pop("FLYDSL_RUNTIME_RUN_ONLY", None)
    yield
    if old_bound is None:
        os.environ.pop("AITER_BF16_FP8_MOE_BOUND", None)
    else:
        os.environ["AITER_BF16_FP8_MOE_BOUND"] = old_bound
    if old_aot is None:
        os.environ.pop("FLYDSL_RUNTIME_RUN_ONLY", None)
    else:
        os.environ["FLYDSL_RUNTIME_RUN_ONLY"] = old_aot


def test_pick_flydsl_stage2_tile_k():
    assert pick_flydsl_stage2_tile_k(256) == 256
    assert pick_flydsl_stage2_tile_k(512) == 256
    assert pick_flydsl_stage2_tile_k(640) == 128
    assert pick_flydsl_stage2_tile_k(384) == 128
    assert pick_flydsl_stage2_tile_k(896) == 128
    assert pick_flydsl_stage2_tile_k(1024) == 256
    assert resolve_flydsl_stage2_tile_k(640, 256) == 128
    assert resolve_flydsl_stage2_tile_k(256, 256) == 256
    assert resolve_flydsl_stage2_tile_k(512, 128) == 128


@_SKIP_GFX950_FLYDSL
def test_flydsl_stage2_a8w4_gui(seed=1234):
    from aiter.ops.flydsl.moe_kernels import flydsl_moe_stage2

    kernel_selector=r"flydsl_moe_stage2|flydsl_moe_gemm_8wave_down_a8w4|opus"
    kernel_selector=r"flydsl|.*8wave"

    """

    16384
    flydsl_moe_gemm_8wave_down_a8w4_0                     488.897  4602  2.250  0.639  1.611  flydsl_moe_gemm_8wave_down_a8w4_0

    token个数翻倍，权重读取增加并不多，但是写出数据量翻倍
    此时总带宽急剧下降，比较奇怪

    16384*2
    flydsl_moe_gemm_8wave_down_a8w4_0                    1107.840  3618  4.008  0.787  3.221  flydsl_moe_gemm_8wave_down_a8w4_0

    """
    NUM_ITERS = 10
    all_tokens = [4096, 8192, 16384, 16384*2]
    model_dim, inter_dim, E, topk = 6144, 256, 384, 8
    base_block_m = 64
    stage2_results = []

    for token in all_tokens:
        num_oc_splits = 1 if token >= 16384 else 4

        data = _generate_a8w4_gui_data(
            token,
            model_dim,
            inter_dim,
            E,
            topk,
            base_block_m,
            seed=seed,
            # The locally built Opus module currently contains effective-K=384.
            inter_pad_override=0,
        )
        effective_inter_dim = inter_dim - data["inter_pad"]

        candidates = [(flydsl_moe_stage2, 64, 0),(flydsl_moe_gemm_8wave_down_a8w4, 256, 128)]
        for kernel, block_m, block_n in candidates:
            if block_m == base_block_m:
                sorted_ids = data["sorted_ids"]
                sorted_weights = data["sorted_weights"]
                sorted_expert_ids = data["sorted_expert_ids"]
                num_valid_ids = data["num_valid_ids"]
                a2_scale_sort = data["a2_scale_sort"]
            else:
                (
                    sorted_ids,
                    sorted_weights,
                    sorted_expert_ids,
                    num_valid_ids,
                    _,
                ) = moe_sorting(
                    data["topk_ids"],
                    data["topk_weights"],
                    E,
                    model_dim,
                    torch.bfloat16,
                    block_m,
                )
                a2_scale_sort = mxfp4_moe_sort_fwd(
                    data["a2_scale"],
                    sorted_ids=sorted_ids,
                    num_valid_ids=num_valid_ids,
                    token_num=token,
                    cols=inter_dim,
                )

            flops = 2 * token * topk * model_dim * effective_inter_dim

            if kernel is flydsl_moe_stage2:
                try:
                    flydsl_out, flydsl_us = pyhip.run_perftest(
                        flydsl_moe_stage2,
                        inter_states=data["a2_q"],
                        w2=data["w2_shuf"],
                        sorted_token_ids=sorted_ids,
                        sorted_expert_ids=sorted_expert_ids,
                        num_valid_ids=num_valid_ids,
                        topk=topk,
                        tile_m=block_m,
                        tile_n=256,
                        tile_k=256,
                        a_dtype="fp8",
                        b_dtype="fp4",
                        out_dtype="bf16",
                        mode="reduce",
                        w2_scale=data["w2_scale_shuf"],
                        a2_scale=a2_scale_sort,
                        sorted_weights=sorted_weights,
                        inter_dim_pad=data["inter_pad"],
                        model_dim_pad=0,
                        num_warmup=2,
                        num_iters=NUM_ITERS,
                        num_flops=flops,
                        num_verbose=1,
                        num_name=f"flydsl_moe_stage2_a8w4_bm{block_m}",
                        num_spec_tag=f"M={token},N={model_dim},K={effective_inter_dim}",
                    )
                    torch.cuda.synchronize()
                    flydsl_err = _check_close(
                        data["ref_stage2"],
                        flydsl_out,
                        f"flydsl_stage2_a8w4_bm{block_m}_i{inter_dim}",
                    )
                    flydsl_diff = pyhip.calc_diff(flydsl_out, data["ref_stage2"])
                except Exception as e:
                    print(f"Error occurred during FlyDSL BM{block_m} stage2 test: {e}")
                    flydsl_us = None
                    flydsl_err = None
                    flydsl_diff = None

                baseline_us = flydsl_us
                stage2_results.append(
                    (
                        f"{token:6} FlyDSL BM{block_m}",
                        flydsl_us,
                        flops / 1e6,
                        1.0,
                        flydsl_err,
                        flydsl_diff,
                    )
                )

            if kernel is flydsl_moe_gemm_8wave_down_a8w4:
                try:
                    wave8_out = torch.full(
                        (token, topk, model_dim),
                        torch.nan,
                        dtype=torch.bfloat16,
                        device="cuda",
                    )
                    wave8_counter = torch.zeros(1, dtype=torch.int32, device="cuda")
                    wave8_kernel = flydsl_moe_gemm_8wave_down_a8w4(
                        n=model_dim,
                        k=inter_dim,
                        topk=topk,
                        num_experts=E,
                        block_m=block_m,
                        block_n=block_n,
                        num_oc_splits=num_oc_splits
                    )

                    def launch_8wave(*args):
                        args[-1].zero_()
                        wave8_kernel(*args)
                        return wave8_out.sum(dim=1)

                    wave8_reduced, wave8_us = pyhip.run_perftest(
                        launch_8wave,
                        wave8_out,
                        data["a2_q"],
                        data["w2_shuf"],
                        a2_scale_sort,
                        data["w2_scale_shuf"],
                        sorted_ids,
                        sorted_weights,
                        sorted_expert_ids,
                        num_valid_ids,
                        wave8_counter,
                        num_warmup=2,
                        num_iters=NUM_ITERS,
                        num_flops=flops,
                        num_verbose=1,
                        num_name=f"moe_gemm_8wave_down_a8w4_bm{block_m}_bn{block_n}",
                        num_spec_tag=f"M={token},N={model_dim},K={effective_inter_dim}",
                    )
                    torch.cuda.synchronize()
                    wave8_err = _check_close(
                        data["ref_stage2"],
                        wave8_reduced,
                        f"moe_8wave_down_a8w4_bm{block_m}_bn{block_n}_i{inter_dim}",
                    )
                    wave8_diff = pyhip.calc_diff(wave8_reduced, data["ref_stage2"])
                except Exception as e:
                    print(
                        f"Error occurred during 8-wave BM{block_m} BN{block_n} "
                        f"stage2 test: {e}"
                    )
                    wave8_us = None
                    wave8_err = None
                    wave8_diff = None

                stage2_results.append(
                    (
                        f"{token:6} 8-wave BM{block_m} BN{block_n}",
                        wave8_us,
                        flops / 1e6,
                        baseline_us/wave8_us if baseline_us and wave8_us else None,
                        wave8_err,
                        wave8_diff,
                    )
                )

        # Opus uses separate sort block sizes dictated by each route-output kernel.
        opus_results = []
        opus_configs = (
            (
                "bf16",
                32,
                OPUS_A8W4_KID_ROUTE_BF16_BM32_FULL_N7168_SMALL,
                0.05,
            ),
            ("fp8", 64, OPUS_A8W4_KID_ROUTE_FP8_BM64_RBN3072, 0.1),
        ) if inter_dim in [384,] else ()

        for route_dtype, opus_block_m, opus_kernel_id, max_err_ratio in opus_configs:
            try:
                if not re.match(kernel_selector, "opus_a8w4_stage2_wrapper"):
                    raise ValueError("not selected")
                # Opus route kernels require metadata sorted with the kernel's own
                # block_m. Keep these independent of the FlyDSL/8-wave block_m.
                (
                    opus_sorted_ids,
                    opus_sorted_weights,
                    opus_sorted_expert_ids,
                    opus_num_valid_ids,
                    _,
                ) = moe_sorting(
                    data["topk_ids"],
                    data["topk_weights"],
                    E,
                    model_dim,
                    torch.bfloat16,
                    opus_block_m,
                )
                opus_padded_rows = int(opus_num_valid_ids[0].item())
                opus_flops = 2 * opus_padded_rows * model_dim * effective_inter_dim
                opus_a2_scale_sort = mxfp4_moe_sort_fwd(
                    data["a2_scale"],
                    sorted_ids=opus_sorted_ids,
                    num_valid_ids=opus_num_valid_ids,
                    token_num=token,
                    cols=inter_dim,
                )
                opus_out = torch.empty_like(data["ref_stage2"])

                # The kernel id controls whether the per-route intermediate is
                # BF16 or MXFP8; both paths finish with token/top-k reduction.
                opus_out, opus_us = pyhip.run_perftest(
                    _opus_a8w4.opus_a8w4_stage2_wrapper,
                    inter_states=data["a2_q"],
                    w1=None,
                    w2=data["w2_shuf"],
                    sorted_token_ids=opus_sorted_ids,
                    sorted_expert_ids=opus_sorted_expert_ids,
                    num_valid_ids=opus_num_valid_ids,
                    out=opus_out,
                    topk=topk,
                    kernelName=_opus_a8w4.OPUS_A8W4_STAGE2_KERNEL,
                    w2_scale=data["w2_scale_shuf"],
                    a2_scale=opus_a2_scale_sort,
                    sorted_weights=opus_sorted_weights,
                    inter_dim_pad=data["inter_pad"],
                    model_dim_pad=0,
                    block_m=opus_block_m,
                    kernel_id=opus_kernel_id,
                    stage2_reduce_block_n=opus_a8w4_kid_reduce_block_n(
                        opus_kernel_id
                    ),
                    route_out=True,
                    num_warmup=2,
                    num_iters=NUM_ITERS,
                    num_flops=opus_flops,
                    num_verbose=1,
                    num_name=f"opus_{route_dtype}_moe_stage2_a8w4",
                    num_spec_tag=f"M={token},N={model_dim},K={effective_inter_dim}",
                )
                torch.cuda.synchronize()
                opus_err = _check_close(
                    data["ref_stage2"],
                    opus_out,
                    f"opus_{route_dtype}_stage2_a8w4_gui_i{inter_dim}",
                    # MXFP8 has an additional route-output quantize/dequantize step.
                    max_err_ratio=max_err_ratio,
                )
                opus_diff = pyhip.calc_diff(opus_out, data["ref_stage2"])
            except Exception as e:
                print(f"Error occurred during Opus {route_dtype} stage2 test: {e}")
                opus_us = None
                opus_err = None
                opus_diff = None
                opus_flops = None

            stage2_results.append(
                (
                    f"{token:6} Opus {route_dtype.upper():.4s} BM{opus_block_m}",
                    opus_us,
                    opus_flops / 1e6 if opus_flops else None,
                    baseline_us/opus_us if baseline_us and opus_us else None,
                    opus_err,
                    opus_diff,
                )
            )

    _print_stage2_results(
        [
            *stage2_results
        ]
    )


@pytest.mark.parametrize("inter_dim", [256, 384, 640])
@_SKIP_GFX950_FLYDSL
def test_flydsl_e2e_a8w4_gui(inter_dim):
    from aiter.ops.flydsl.moe_kernels import flydsl_moe_stage1, flydsl_moe_stage2

    token, model_dim, E, topk, block_m, seed = 16, 512, 8, 2, 32, 0
    data = _generate_a8w4_gui_data(
        token, model_dim, inter_dim, E, topk, block_m, seed=seed
    )
    stage1_out = flydsl_moe_stage1(
        a=data["a_q"],
        w1=data["w1_shuf"],
        sorted_token_ids=data["sorted_ids"],
        sorted_expert_ids=data["sorted_expert_ids"],
        num_valid_ids=data["num_valid_ids"],
        topk=topk,
        tile_m=32,
        tile_n=256,
        tile_k=_stage1_tile_k(model_dim),
        a_dtype="fp8",
        b_dtype="fp4",
        out_dtype="bf16",
        act="swiglu",
        gate_mode="interleave",
        w1_scale=data["w1_scale_shuf"],
        a1_scale=data["a_scale_sort"],
        inter_dim_pad=data["inter_pad"],
        model_dim_pad=0,
    )
    a2_q, a2_scale = per_1x32_f8_scale_f8_quant(
        stage1_out, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0
    )
    a2_q = a2_q.view(token, topk, inter_dim)
    a2_scale_sort = mxfp4_moe_sort_fwd(
        a2_scale,
        sorted_ids=data["sorted_ids"],
        num_valid_ids=data["num_valid_ids"],
        token_num=token,
        cols=inter_dim,
    )
    out = flydsl_moe_stage2(
        inter_states=a2_q,
        w2=data["w2_shuf"],
        sorted_token_ids=data["sorted_ids"],
        sorted_expert_ids=data["sorted_expert_ids"],
        num_valid_ids=data["num_valid_ids"],
        topk=topk,
        tile_m=32,
        tile_n=256,
        tile_k=256,
        a_dtype="fp8",
        b_dtype="fp4",
        out_dtype="bf16",
        mode="atomic",
        w2_scale=data["w2_scale_shuf"],
        a2_scale=a2_scale_sort,
        sorted_weights=data["sorted_weights"],
        inter_dim_pad=data["inter_pad"],
        model_dim_pad=0,
    )
    torch.cuda.synchronize()
    _check_close(data["ref_stage2"], out, f"e2e_a8w4_gui_i{inter_dim}")


def main():
    test_flydsl_stage2_a8w4_gui()


if __name__ == "__main__":
    main()

