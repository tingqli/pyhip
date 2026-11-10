# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx942 FP8 block-scale fused MoE: GateUp 2x4_pf1 with FP8 sorted-row output, then Down 4x2_pf1 with single_store.

This is what aiter.fused_moe runs in place of the 1-stage asm kernel fmoe_fp8_blockscale_g1u1 when
AITER_FMOE_FP8_BLOCKSCALE_FLYDSL=1 (see docs/moe_a8w8_blockscale_gfx942.md). Inputs use the asm kernel's layouts:
a1 [tokens, H] FP8 with a1_scale_t [H/128, tokens] FP32, w1 / w2 aiter shuffle_weight(16,16) of [E, 2I, H] /
[E, H, I] FP8 with FP32 scales [E, 2I/128, H/128] / [E, H/128, I/128], and the routes as from aiter
moe_sorting(block_size=256) with an empty moe_buf, so out is not zeroed by the sort.
"""

import torch

from .moe_down_a8w8_blockscale_gfx942_lds import (
    moe_down_a8w8_blockscale_gfx942_lds,
    moe_route_counts,
    moe_zero_rows,
)
from .moe_down_a8w8_blockscale_gfx942_lds import validate_parameters as validate_down
from .moe_gateup_a8w8_blockscale_gfx942_lds import (
    SORT_BLOCK_M,
    moe_gateup_a8w8_blockscale_gfx942_lds,
)
from .moe_gateup_a8w8_blockscale_gfx942_lds import (
    validate_parameters as validate_gateup,
)

GATEUP_LAYOUT = "2x4_pf1"
DOWN_LAYOUT = "4x2_pf1"


def max_sorted_blocks(tokens, topk, local_experts):
    """Upper bound on the 256-row blocks the sort emits: at most one partial block per local expert."""
    return -(-tokens * min(topk, local_experts) // SORT_BLOCK_M) + local_experts


def unsupported_reason(tokens, hidden_dim, inter_dim, topk, local_experts):
    """'' when fused_moe_a8w8_blockscale_gfx942 handles this shape, else the reason it does not."""
    if hidden_dim > 4096:
        return "hidden_dim > 4096 (GateUp keeps all K-block B scales in one VGPR)"
    capacity = max_sorted_blocks(tokens, topk, local_experts) * SORT_BLOCK_M
    try:
        validate_gateup(tokens, inter_dim, hidden_dim, topk, local_experts, fp8_out=True)
        # Also bounds the sorted FP8 intermediate [capacity, I] by one buffer descriptor.
        validate_down(tokens, hidden_dim, inter_dim, topk, local_experts, capacity, sorted_a2=True, atomic=True)
    except ValueError as error:
        return str(error)
    return ""


def fused_moe_a8w8_blockscale_gfx942(
    a1, a1_scale_t, w1, w1_scale, w2, w2_scale, sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, out, topk,
    num_rows=None, single_store=True,
):
    """out [tokens, H] BF16 = sum over each token's local routes of routing weight * Down(SiLU(gate) * up).

    num_rows: optional device int32 tensor with the valid token count (EP all-to-all receive buffers); rows of out
    at or past it are left untouched. single_store=False adds every route with BF16 atomics into a zeroed out.
    """
    tokens = a1.shape[0]
    inter_dim = w2.shape[2]
    # aiter's sorted_ids holds tokens * topk + E * 256 - topk entries (not whole blocks); the valid blocks fit in its floor.
    blocks = min(sorted_expert_ids.numel(), sorted_ids.numel() // SORT_BLOCK_M, max_sorted_blocks(tokens, topk, w1.shape[0]))
    expert_ids = sorted_expert_ids[:blocks]
    capacity = blocks * SORT_BLOCK_M
    a2 = torch.empty((capacity, inter_dim), device=a1.device, dtype=a1.dtype)
    a2_scale_t = torch.empty((inter_dim // 128, capacity), device=a1.device, dtype=torch.float32)
    route_counts = None
    if single_store:
        route_counts = torch.empty(tokens, device=a1.device, dtype=torch.int32)
        moe_route_counts(sorted_ids, num_valid_ids, expert_ids, route_counts, topk)
        moe_zero_rows(out, route_counts, num_rows)
    else:
        out.zero_()
    moe_gateup_a8w8_blockscale_gfx942_lds(
        a1, w1, a1_scale_t, w1_scale, sorted_ids, expert_ids, num_valid_ids, a2, layout=GATEUP_LAYOUT,
        out_scale_t=a2_scale_t, topk=topk,
    )
    moe_down_a8w8_blockscale_gfx942_lds(
        a2, w2, a2_scale_t, w2_scale, sorted_ids, sorted_weights, expert_ids, num_valid_ids, out, layout=DOWN_LAYOUT,
        reduce="atomic", zero_out=False, topk=topk, single_store=single_store, route_counts=route_counts,
    )
    return out
