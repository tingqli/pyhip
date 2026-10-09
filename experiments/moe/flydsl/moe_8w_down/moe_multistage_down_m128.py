# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Experiment entry for the packaged M128 block-scale Down kernel."""

from functools import cache

import torch

from pyhip.ops.moe.flydsl.moe_fp8_blockscale_stage2 import (
    M128_CONTINUOUS_VMEM_CONFIG,
    M128_QUEUE_SIZE,
    make_m128_down as _make_m128_down,
)


@cache
def make_m128_down(*, n, k=256, topk, num_experts, persistent=False,
                   is_sorted_bm_256=False, num_oc_splits=8):
    """Preserve the experiment's one-word counter and self-reset workspace.

    K is a positive multiple of256. num_oc_splits is a compile-time N partition;
    N/splits must be 128-aligned.
    This experiment caches private heads, so one callable must not be launched
    concurrently on multiple streams. The src API accepts caller-owned heads
    instead, which the complete tuned MoE allocates independently per call.
    """
    kernel = _make_m128_down(n=n, k=k, topk=topk, num_experts=num_experts,
                            persistent=persistent, is_sorted_bm_256=is_sorted_bm_256,
                            num_oc_splits=num_oc_splits)
    workspace = {}

    def down(output, input_q, weight, input_scales, weight_scales,
             sorted_ids, sorted_weights, expert_ids, valid_ids, counter):
        assert counter.numel() == 1 and counter.dtype == torch.int32
        assert counter.is_cuda and counter.is_contiguous() and counter.device == output.device
        if persistent:
            if "heads" not in workspace or workspace["heads"].device != output.device:
                workspace["heads"] = torch.zeros(M128_QUEUE_SIZE, dtype=torch.int32, device=output.device)
            counter = workspace["heads"]
        return kernel(output, input_q, weight, input_scales, weight_scales,
                      sorted_ids, sorted_weights, expert_ids, valid_ids, counter)

    down.config = dict(kernel.config)
    down.workspace = workspace
    return down