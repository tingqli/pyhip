# Copyright 2023-2024 SGLang Team
# SPDX-License-Identifier: Apache-2.0

"""Allocation-free SGLang launch adapters, with package-relative imports."""

from __future__ import annotations

import msgspec
import torch
import triton

from .baseline_kernels import _sparse_gqa_chunk_prefill, _sparse_gqa_prefill
from .contract import AttentionInputs

NAME = "sglang_single_query_baseline"

_H20_CONFIGS = (
    (32, (32, 8, 2)),
    (64, (64, 8, 2)),
    (1024, (32, 4, 2)),
    (float("inf"), (16, 1, 2)),
)
_L20_CONFIGS = (
    (32, (32, 8, 2)),
    (64, (64, 8, 2)),
    (128, (64, 4, 2)),
    (512, (32, 4, 2)),
    (float("inf"), (16, 1, 2)),
)


class BaselinePlan(msgspec.Struct, frozen=True, kw_only=True):
    grid: tuple[int, int]
    strides: tuple[int, ...]
    options: dict[str, int]


def prepare(*, inputs: AttentionInputs) -> BaselinePlan:
    q, k, v = inputs.q, inputs.k, inputs.v
    group_size = q.shape[1] // k.shape[1]
    table = (
        _H20_CONFIGS if "H20" in torch.cuda.get_device_name(q.device) else _L20_CONFIGS
    )
    block_n, warps, stages = next(cfg for limit, cfg in table if q.shape[0] <= limit)
    # Output is contiguous and has Q's shape, as required by the plugin contract.
    output_strides = (q.shape[1] * q.shape[2], q.shape[2], 1)
    strides = (
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *output_strides,
        inputs.indices.stride(0),
        0,
        inputs.indices.stride(1),
    )
    return BaselinePlan(
        grid=(inputs.max_seqlen_q, len(inputs.spec.query_lens) * k.shape[1]),
        strides=strides,
        options={
            "NUM_KV_HEADS": k.shape[1],
            "GROUP_SIZE": group_size,
            "BLOCK_M": max(16, triton.next_power_of_2(group_size)),
            "BLOCK_N": block_n,
            "HEAD_DIM": q.shape[2],
            "num_warps": warps,
            "num_stages": stages,
        },
    )


def run(*, inputs: AttentionInputs, prepared: object, out: torch.Tensor) -> None:
    assert isinstance(prepared, BaselinePlan)
    args = (inputs.q, inputs.k, inputs.v, out, inputs.indices)
    tail = (inputs.scale, inputs.indices.shape[1], *prepared.strides)
    if inputs.spec.name == "no_prefix":
        _sparse_gqa_prefill[prepared.grid](
            *args, inputs.cu_q, *tail, **prepared.options
        )
    else:
        _sparse_gqa_chunk_prefill[prepared.grid](
            *args, inputs.cu_q, inputs.cu_k, inputs.kv_lens, *tail, **prepared.options
        )
