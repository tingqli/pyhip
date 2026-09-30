"""Exact per-query block membership and compact shared K/V traversal plans."""

from __future__ import annotations

import msgspec
import numpy as np
import torch
import triton
import triton.language as tl

from .contract import AttentionInputs


class SparsePlan(msgspec.Struct, kw_only=True):
    metadata: torch.Tensor
    dense_membership: torch.Tensor
    blocks: torch.Tensor
    membership: torch.Tensor
    score_masks: torch.Tensor
    counts: torch.Tensor
    active: torch.Tensor
    query_tiles: torch.Tensor
    query_tile: int
    group_padded: int
    block_capacity: int
    max_blocks: int
    num_tiles: int
    grid: int
    max_union_inflation: float


@triton.jit
def _scatter_membership(
    Blocks,
    Positions,
    Meta,
    Dense,
    QB: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
    TOPK: tl.constexpr,
    C: tl.constexpr,
):
    tile = tl.program_id(0)
    local = tl.program_id(1)
    first = tl.load(Meta + tile * 5)
    rows = tl.load(Meta + tile * 5 + 1)
    if local < rows:
        row = first + local
        cols = tl.arange(0, C)
        blocks = tl.load(Blocks + row * TOPK + cols, cols < TOPK, -1)
        bit = (1 << local).to(tl.int32)
        tl.atomic_or(
            Dense + tile * MAX_BLOCKS + blocks, bit, blocks >= 0, sem="relaxed"
        )
        visible = tl.load(Positions + row) + 1
        if visible % 4 != 0:
            tl.atomic_or(Dense + tile * MAX_BLOCKS + visible // 4, bit, sem="relaxed")


@triton.jit
def _compact_membership(
    Dense,
    Blocks,
    Membership,
    Counts,
    Meta,
    Active,
    MAX_BLOCKS: tl.constexpr,
    CAPACITY: tl.constexpr,
    C: tl.constexpr,
    RHO: tl.constexpr,
):
    tile = tl.program_id(0)
    ids = tl.arange(0, C)
    bits = tl.load(Dense + tile * MAX_BLOCKS + ids, ids < MAX_BLOCKS, 0)
    valid = bits != 0
    rows = tl.load(Meta + tile * 5 + 1)
    first_position = tl.load(Meta + tile * 5 + 4)
    all_queries = tl.full((), 0xFFFFFFFF, tl.uint32) >> (32 - rows)
    queries = tl.arange(0, 32)
    visible = first_position + queries + 1
    selected = tl.minimum(visible // 4, 512) + (visible % 4 != 0).to(tl.int32)
    total = tl.sum(tl.where(queries < rows, selected, 0))
    count = tl.sum(valid.to(tl.int32))
    enabled = count * rows <= RHO * total
    tl.store(Active + tile, enabled)
    tl.store(Counts + tile * 2, count)
    tl.store(Counts + tile * 2 + 1, 0)
    if enabled:
        common = (
            valid
            & (bits.to(tl.uint32) == all_queries)
            & (ids * 4 + 3 <= first_position)
        )
        common_count = tl.sum(common.to(tl.int32))
        other = valid & ~common
        destination = tl.where(
            common,
            tl.cumsum(common.to(tl.int32)) - 1,
            common_count + tl.cumsum(other.to(tl.int32)) - 1,
        )
        tl.store(Blocks + tile * CAPACITY + destination, ids, valid)
        tl.store(Membership + tile * CAPACITY + destination, bits, valid)
        tl.store(Counts + tile * 2 + 1, common_count // 16)


@triton.jit
def _score_masks(
    Blocks,
    Membership,
    Counts,
    Meta,
    Masks,
    Active,
    QB: tl.constexpr,
    CAP: tl.constexpr,
    C: tl.constexpr,
):
    tile = tl.program_id(0)
    if tl.load(Active + tile) == 0:
        return
    r = tl.arange(0, C)
    count = tl.load(Counts + tile * 2)
    common_tiles = tl.load(Counts + tile * 2 + 1)
    rows = tl.load(Meta + tile * 5 + 1)
    position0 = tl.load(Meta + tile * 5 + 4)
    length = tl.load(Meta + tile * 5 + 3)
    elements = tl.cdiv(count, 16) * QB * 4
    for start in range(common_tiles * QB * 4, elements, C):
        index = start + r
        nt = index // (QB * 4)
        query, quarter = (index // 4) % QB, index % 4
        position = position0 + query
        mask = tl.full((C,), 0, tl.int32)
        for group in tl.static_range(4):
            slot = nt * 16 + quarter * 2 + (group // 2) * 8 + group % 2
            valid = (slot < count) & (query < rows) & (index < elements)
            membership = tl.load(Membership + tile * CAP + slot, valid, 0)
            block = tl.load(Blocks + tile * CAP + slot, valid, 0)
            valid = valid & (((membership >> query) & 1) != 0)
            for offset in tl.static_range(4):
                token = block * 4 + offset
                keep = valid & (token <= position) & (token < length)
                mask = mask | (keep.to(tl.int32) << (group * 4 + offset))
        tl.store(Masks + tile * (CAP // 16) * QB * 4 + index, mask, index < elements)


def allocate_plan(
    *,
    inputs: AttentionInputs,
    query_tile: int,
    grid_multiplier: int = 2,
    max_union_inflation: float = float("inf"),
    skip_counts: tuple[int, ...] | None = None,
) -> SparsePlan:
    group = inputs.q.shape[1] // inputs.k.shape[1]
    group_padded = group
    if query_tile < 1 or query_tile > 32:
        raise ValueError("Require 1<=BQ<=32")
    max_queries = 1 << ((128 // group).bit_length() - 1)
    query_tile = min(query_tile, max_queries)
    skips = (0,) * len(inputs.spec.query_lens) if skip_counts is None else skip_counts
    rows, q0, k0 = [], 0, 0
    for q_len, prefix, skip in zip(
        inputs.spec.query_lens, inputs.spec.prefix_lens, skips
    ):
        kv_len = q_len + prefix
        local = skip
        while local < q_len:
            end = min(q_len, (local // query_tile + 1) * query_tile)
            rows.append((q0 + local, end - local, k0, kv_len, prefix + local))
            local = end
        q0 += q_len
        k0 += kv_len
    metadata = torch.from_numpy(np.asarray(rows, dtype=np.int32).reshape(-1, 5)).to(
        inputs.q.device
    )
    query_tiles = np.full(inputs.q.shape[0], -1, dtype=np.int32)
    for i, (start, n, *_) in enumerate(rows):
        query_tiles[start : start + n] = i
    max_blocks = triton.cdiv(inputs.max_seqlen_k, 4)
    capacity = triton.cdiv(min(max_blocks, query_tile * 513), 16) * 16
    tiles = len(rows)
    kwargs = {"dtype": torch.int32, "device": inputs.q.device}
    return SparsePlan(
        metadata=metadata,
        dense_membership=torch.empty((tiles, max_blocks), **kwargs),
        blocks=torch.empty((tiles, capacity), **kwargs),
        membership=torch.empty((tiles, capacity), **kwargs),
        score_masks=torch.empty((tiles, capacity // 16, query_tile, 4), **kwargs),
        counts=torch.empty((tiles, 2), **kwargs),
        active=torch.empty(tiles, **kwargs),
        query_tiles=torch.from_numpy(query_tiles).to(inputs.q.device),
        query_tile=query_tile,
        group_padded=group_padded,
        block_capacity=capacity,
        max_blocks=max_blocks,
        num_tiles=tiles,
        grid=min(
            tiles * inputs.k.shape[1] * triton.cdiv(query_tile * group, 128),
            torch.cuda.get_device_properties(inputs.q.device).multi_processor_count
            * grid_multiplier,
        ),
        max_union_inflation=max_union_inflation,
    )


def rebuild_plan(*, inputs: AttentionInputs, plan: SparsePlan) -> None:
    if plan.num_tiles == 0:
        return
    plan.dense_membership.zero_()
    _scatter_membership[(plan.num_tiles, plan.query_tile)](
        inputs.block_indices,
        inputs.query_positions,
        plan.metadata,
        plan.dense_membership,
        plan.query_tile,
        plan.max_blocks,
        512,
        512,
        num_warps=4,
    )
    _compact_membership[(plan.num_tiles,)](
        plan.dense_membership,
        plan.blocks,
        plan.membership,
        plan.counts,
        plan.metadata,
        plan.active,
        plan.max_blocks,
        plan.block_capacity,
        triton.next_power_of_2(plan.max_blocks),
        plan.max_union_inflation,
        num_warps=4 if plan.max_blocks <= 8192 else 8,
    )
    _score_masks[(plan.num_tiles,)](
        plan.blocks,
        plan.membership,
        plan.counts,
        plan.metadata,
        plan.score_masks,
        plan.active,
        plan.query_tile,
        plan.block_capacity,
        256,
        num_warps=4,
    )
