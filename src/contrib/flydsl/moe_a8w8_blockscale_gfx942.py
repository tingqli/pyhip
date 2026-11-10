# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx942 FP8 block-scale fused MoE: GateUp 2x4_pf1 with FP8 sorted-row output, then Down 4x2_pf1 with single_store.

This is what aiter.fused_moe runs in place of the 1-stage asm kernel fmoe_fp8_blockscale_g1u1 when
AITER_FMOE_FP8_BLOCKSCALE_FLYDSL=1 (see docs/moe_a8w8_blockscale_gfx942.md). Inputs use the asm kernel's layouts:
a1 [tokens, H] FP8 with a1_scale_t [H/128, tokens] FP32, w1 / w2 aiter shuffle_weight(16,16) of [E, 2I, H] /
[E, H, I] FP8 with FP32 scales [E, 2I/128, H/128] / [E, H/128, I/128], and the routes as from aiter
moe_sorting(block_size=256) with an empty moe_buf, so out is not zeroed by the sort.

fused_moe_a8w8_blockscale_gfx942_mixed runs on the same 256-row sort table and decides per sorted block at run time
whether the block has a route at row 128 (more than 128 routes): big_blocks="merged" (default) runs every block in
one GateUp and one Down launch, BM256 tiles for those blocks and 8-wave pair tasks for the others
(moe_merged_a8w8_blockscale_gfx942); "bm256" runs the 4-wave small-M kernels (moe_small_m_a8w8_blockscale_gfx942) on
the small blocks and the BM256 kernels on the others, "split" the small-M kernels on all, a big block as two 128-row
tasks (docs/record/moe_small_m_gfx942.md).
fused_moe_a8w8_blockscale_gfx942_sort sorts and runs it.
"""

import torch

from .moe_down_a8w8_blockscale_gfx942_lds import (
    moe_down_a8w8_blockscale_gfx942_lds,
    moe_route_counts,
    moe_zero_rows,
)
from .moe_down_a8w8_blockscale_gfx942_lds import validate_parameters as validate_down
from .moe_gateup_a8w8_blockscale_gfx942_lds import (
    SMALL_M_ROWS,
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
    blocks = min(sorted_expert_ids.numel(), max_sorted_blocks(tokens, topk, w1.shape[0]))
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


def fused_moe_a8w8_blockscale_gfx942_mixed(
    a1, a1_scale_t, w1, w1_scale, w2, w2_scale, sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, out, topk,
    ring=2, big_blocks="merged",
):
    """fused_moe_a8w8_blockscale_gfx942 on routes from aiter moe_sorting(256) that was handed out as moe_buf (so out
    is zeroed), the Downs adding into out with BF16 atomics. Sorted blocks with a route at row 128 (big) and the others:
    "merged": one GateUp and one Down launch, BM256 tiles for big blocks and pair tasks for the others (falls back to
    "bm256" past moe_merged_a8w8_blockscale_gfx942.MAX_BLOCKS sort blocks); "bm256": small-M GateUp / Down for the
    small blocks, BM256 ones for the big ones; "split": small-M ones for all, a big block as two 128-row tasks.
    ring: B ring of the small-M kernels.
    """
    from .moe_merged_a8w8_blockscale_gfx942 import MAX_BLOCKS, moe_down_merged, moe_gateup_merged
    from .moe_small_m_a8w8_blockscale_gfx942 import moe_down_small_m, moe_gateup_small_m

    assert big_blocks in ("merged", "bm256", "split"), big_blocks
    tokens = a1.shape[0]
    inter_dim = w2.shape[2]
    blocks = min(sorted_expert_ids.numel(), max_sorted_blocks(tokens, topk, w1.shape[0]))
    expert_ids = sorted_expert_ids[:blocks]
    capacity = blocks * SORT_BLOCK_M
    a2 = torch.empty((capacity, inter_dim), device=a1.device, dtype=a1.dtype)
    # Both GateUps write their blocks' rows of the same per-128-column scales.
    a2_scale_t = torch.empty((inter_dim // 128, capacity), device=a1.device, dtype=torch.float32)
    if big_blocks == "merged" and blocks <= MAX_BLOCKS:
        moe_gateup_merged(a1, w1, a1_scale_t, w1_scale, sorted_ids, expert_ids, num_valid_ids, a2, a2_scale_t, topk)
        moe_down_merged(a2, w2, a2_scale_t, w2_scale, sorted_ids, sorted_weights, expert_ids, num_valid_ids, out, topk)
        return out
    # An expert gets at most one route per token: with <= 128 tokens no block reaches row 128.
    has_big = tokens > SMALL_M_ROWS
    split = has_big and big_blocks == "split"
    bm256 = has_big and big_blocks != "split"
    moe_gateup_small_m(
        a1, w1, a1_scale_t, w1_scale, sorted_ids, expert_ids, num_valid_ids, a2, a2_scale_t, topk, ring, all_blocks=split
    )
    if bm256:
        moe_gateup_a8w8_blockscale_gfx942_lds(
            a1, w1, a1_scale_t, w1_scale, sorted_ids, expert_ids, num_valid_ids, a2, layout=GATEUP_LAYOUT,
            out_scale_t=a2_scale_t, topk=topk, big_blocks_only=True,
        )
    moe_down_small_m(
        a2, w2, a2_scale_t, w2_scale, sorted_ids, sorted_weights, expert_ids, num_valid_ids, out, topk, ring,
        all_blocks=split,
    )
    if bm256:
        moe_down_a8w8_blockscale_gfx942_lds(
            a2, w2, a2_scale_t, w2_scale, sorted_ids, sorted_weights, expert_ids, num_valid_ids, out, layout=DOWN_LAYOUT,
            reduce="atomic", zero_out=False, topk=topk, big_blocks_only=True,
        )
    return out


def fused_moe_a8w8_blockscale_gfx942_sort(
    a1, a1_scale_t, w1, w1_scale, w2, w2_scale, topk_ids, topk_weights, out, expert_mask=None, num_local_tokens=None,
    big_blocks="merged",
):
    """aiter moe_sorting(256) with out as moe_buf (zeroed), then fused_moe_a8w8_blockscale_gfx942_mixed.

    topk_ids [tokens, topk] holds global expert ids; expert_mask [global experts] marks this rank's (EP), and
    num_local_tokens (device int32) bounds the valid rows of an all-to-all receive buffer; the sort zeroes all of out.
    """
    from aiter import dtypes
    from aiter.fused_moe import moe_sorting

    global_experts = w1.shape[0] if expert_mask is None else expert_mask.numel()
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, _ = moe_sorting(
        topk_ids, topk_weights, global_experts, w2.shape[1], dtypes.bf16, SORT_BLOCK_M, expert_mask, num_local_tokens,
        output=out,
    )
    return fused_moe_a8w8_blockscale_gfx942_mixed(
        a1, a1_scale_t, w1, w1_scale, w2, w2_scale, sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, out,
        topk_ids.shape[1], big_blocks=big_blocks,
    )
