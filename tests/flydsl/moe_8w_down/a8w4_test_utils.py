# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Shared A8W4 test data, validation, and reporting helpers."""

from __future__ import annotations

import torch

from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import fused_topk, moe_sorting, torch_moe_stage1, torch_moe_stage2
from aiter.ops.quant import (
    mxfp4_moe_sort_fwd,
    per_1x32_f4_quant,
    per_1x32_f8_scale_f8_quant,
)
from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4

Q_TYPE = QuantType.per_1x32


def _inter_pad(inter_dim: int) -> int:
    return ((inter_dim + 255) // 256 * 256) - inter_dim


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


def error_stats(output, reference, *, atol=1.0, rtol=0.05, max_err_ratio=0.05):
    assert output.shape == reference.shape
    mismatches, total_abs, max_abs, dot, denominator = 0, 0., 0., 0., 0.
    finite = True
    for begin in range(0, output.shape[0], 64):
        actual, expected = output[begin:begin + 64].float(), reference[begin:begin + 64].float()
        finite &= bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
        error = (actual - expected).abs()
        mismatches += (error > atol + rtol * expected.abs()).sum().item()
        total_abs += error.sum().item()
        max_abs = max(max_abs, error.max().item())
        a64, e64 = actual.double(), expected.double()
        dot += (a64 * e64).sum().item()
        denominator += (a64.square() + e64.square()).sum().item()
    ratio = mismatches / output.numel()
    return dict(status='PASS' if finite and ratio <= max_err_ratio else 'FAIL',
                error_ratio=ratio, mismatches=f'{mismatches}/{output.numel()}',
                max_abs=max_abs, mean_abs=total_abs / output.numel(),
                diff=1 - 2 * dot / denominator if denominator else 0.)


def check_exact(actual, expected):
    for start in range(0, actual.shape[0], 64):
        torch.testing.assert_close(actual[start:start + 64], expected[start:start + 64], atol=0, rtol=0)


def print_markdown_table(headers, rows):
    widths = [max(len(str(header)), *(len(str(row[i])) for row in rows)) for i, header in enumerate(headers)]
    def formatted(row):
        return '| ' + ' | '.join(str(value).ljust(widths[i]) for i, value in enumerate(row)) + ' |'
    print(formatted(headers))
    print(formatted(['-' * width for width in widths]))
    for row in rows:
        print(formatted(row))


def sorted_metadata(data, block_m, experts):
    ids, weights, eids, valid, _ = moe_sorting(data['topk_ids'], data['topk_weights'], experts,
                                              data['model_dim'], torch.bfloat16, block_m)
    scales = mxfp4_moe_sort_fwd(data['a2_scale'], sorted_ids=ids, num_valid_ids=valid,
                               token_num=data['token'], cols=data['inter_dim'])
    return ids, weights, eids, valid, scales