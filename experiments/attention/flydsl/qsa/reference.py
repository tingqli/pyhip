# Copyright 2023-2024 SGLang Team
# SPDX-License-Identifier: Apache-2.0

"""SGLang's batched FP32 sparse GQA oracle, with package-relative imports."""

from __future__ import annotations

import numpy as np
import torch

from .contract import AttentionInputs


def reference_rows(*, inputs: AttentionInputs, sample_count: int) -> list[int]:
    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    if sample_count >= inputs.q.shape[0]:
        return list(range(inputs.q.shape[0]))
    rows = set()
    start = 0
    for length, prefix in zip(inputs.spec.query_lens, inputs.spec.prefix_lens):
        boundaries = (
            0,
            1,
            2,
            3,
            4,
            7,
            8,
            31,
            32,
            2047 - prefix,
            2048 - prefix,
            2050 - prefix,
            2051 - prefix,
            2052 - prefix,
            length - 1,
        )
        rows.update(start + i for i in boundaries if 0 <= i < length)
        start += length
    rows.update(np.linspace(0, inputs.q.shape[0] - 1, sample_count, dtype=int).tolist())
    return sorted(rows)


@torch.inference_mode()
def sparse_reference(*, inputs: AttentionInputs, rows: list[int]) -> torch.Tensor:
    outputs = []
    kv_heads = inputs.k.shape[1]
    group_size = inputs.q.shape[1] // kv_heads
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        for start in range(0, len(rows), 4):
            ids = torch.tensor(rows[start : start + 4], device=inputs.q.device)
            tokens = inputs.indices.index_select(0, ids).long()
            sequences = inputs.query_sequence_ids.index_select(0, ids).long()
            base = inputs.cu_k.index_select(0, sequences).long()
            valid = tokens >= 0
            slots = base[:, None] + tokens.clamp_min(0)
            k = inputs.k[slots].float()
            v = inputs.v[slots].float()
            q = (
                inputs.q[ids]
                .float()
                .reshape(-1, kv_heads, group_size, inputs.q.shape[2])
            )
            scores = torch.einsum("bghd,bkgd->bghk", q, k) * inputs.scale
            scores.masked_fill_(~valid[:, None, None, :], -float("inf"))
            weights = torch.softmax(scores, dim=-1)
            outputs.append(
                torch.einsum("bghk,bkgd->bghd", weights, v).reshape(
                    -1, inputs.q.shape[1], inputs.q.shape[2]
                )
            )
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
    return torch.cat(outputs)


@torch.inference_mode()
def check_output(
    *,
    inputs: AttentionInputs,
    output: torch.Tensor,
    sample_count: int = 32,
    rtol: float = 2e-2,
    atol: float = 2e-2,
) -> dict:
    assert output.shape == inputs.q.shape and output.dtype == inputs.q.dtype
    assert output.device == inputs.q.device and output.is_contiguous()
    assert bool(
        torch.isfinite(output).all()
    ), "Output contains NaN/Inf or unwritten rows"
    rows = reference_rows(inputs=inputs, sample_count=sample_count)
    expected = sparse_reference(inputs=inputs, rows=rows)
    actual = output[rows].float()
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    error = actual - expected
    return {
        "reference_rows": len(rows),
        "checked_row_ids": rows,
        "max_abs_error": float(error.abs().max()),
        "relative_l2_error": float(error.norm() / expected.norm().clamp_min(1e-12)),
        "rtol": rtol,
        "atol": atol,
        "all_output_rows_finite": True,
    }
