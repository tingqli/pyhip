"""Exact QSA attention selection, private union scratch and fused preparation launches.

Dense membership is zero when a plan is allocated and after every completed
build. Recovery scatters into it; its owning compact CTA consumes and clears
the same causal prefix. The next launch sorts tasks and builds disjoint mask
partitions. Plans are private to one stream and pinned for graphs.
"""

import msgspec
import numpy as np
import torch
import triton
import triton.language as tl


class SparsePlan(msgspec.Struct, kw_only=True):
    metadata: torch.Tensor
    dense_membership: torch.Tensor
    blocks: torch.Tensor
    membership: torch.Tensor
    score_masks: torch.Tensor
    counts: torch.Tensor
    active: torch.Tensor
    task_order: torch.Tensor
    query_tiles: torch.Tensor
    query_tile: int
    group_padded: int
    block_capacity: int
    max_blocks: int
    num_tiles: int
    grid: int
    max_union_inflation: float
    packed_direct: bool = False


@triton.jit
def _compare_exchange(cube, flip, bit: tl.constexpr):
    right = tl.reshape(tl.arange(0, 2), [1] * (8 - bit) + [2] + [1] * bit)
    peer = cube ^ tl.xor_sum(cube, 8 - bit, keep_dims=True)
    return tl.where((flip ^ right) != 0, tl.maximum(cube, peer), tl.minimum(cube, peer))


@triton.jit
def _sort_blocks(values):
    cube = tl.reshape(values, [2] * 9)
    for stage in tl.static_range(1, 10):
        if stage < 9:
            flip = tl.reshape(tl.arange(0, 2), [1] * (8 - stage) + [2] + [1] * stage)
        else:
            flip = tl.full((), 0, tl.int32)
        for step in tl.static_range(stage):
            cube = _compare_exchange(cube, flip, stage - step - 1)
    return tl.reshape(cube, (512,))


@triton.jit(do_not_specialize=["MAX_BLOCKS"])
def attention_recover_scatter(Indices, Positions, Lengths, SequenceIds, Blocks, Errors,
            Dense=None, QueryTiles=None, Meta=None, MAX_BLOCKS=0,
            REVERSE: tl.constexpr = False):
    row = tl.num_programs(0) - 1 - tl.program_id(0) if REVERSE else tl.program_id(0)
    position = tl.load(Positions + row)
    length = tl.load(Lengths + tl.load(SequenceIds + row))
    visible = position + 1
    complete = tl.minimum(visible // 4, 512)
    columns = tl.arange(0, 512)
    ptr = Indices + row * 2051 + columns * 4
    a, b = tl.load(ptr), tl.load(ptr + 1)
    c, d = tl.load(ptr + 2), tl.load(ptr + 3)
    full = columns < complete
    valid = (a >= 0) & (a % 4 == 0) & (b == a + 1) & (c == a + 2) & (d == a + 3)
    valid &= (d < visible) & (d < length)
    tail = visible % 4
    tail_start = visible - tail
    partial = columns == complete
    padding = a == tl.where(partial & (tail > 0), tail_start, -1)
    padding &= b == tl.where(partial & (tail > 1), tail_start + 1, -1)
    padding &= (c == tl.where(partial & (tail > 2), tail_start + 2, -1)) & (d == -1)
    errors = tl.where(full, ~valid, ~padding)
    extra = tl.arange(0, 4)
    last = tl.load(Indices + row * 2051 + 2048 + extra, extra < 3, -1)
    expected = tl.where((complete == 512) & (extra < tail), tail_start + extra, -1)
    error = (position < 0) | (visible > length)
    error |= tl.sum(errors.to(tl.int32)) != 0
    error |= tl.sum(((extra < 3) & (last != expected)).to(tl.int32)) != 0
    blocks = tl.where(full & valid, a // 4, -1)
    ordered = tl.where(full, blocks, 2147483647)
    canonical = tl.full((), False, tl.int1)
    if visible <= 2051:
        canonical = tl.sum((full & (blocks != columns)).to(tl.int32)) == 0
    if not canonical:
        ordered = _sort_blocks(ordered)
        previous = tl.gather(ordered, tl.maximum(columns - 1, 0), axis=0)
        error |= tl.sum((full & (columns > 0) & (ordered == previous)).to(tl.int32)) != 0
    tl.store(Blocks + row * 512 + columns, tl.where(full, ordered, -1))
    tl.store(Errors + row, error.to(tl.int32))
    if Dense is not None:
        tile = tl.load(QueryTiles + row)
        if tile >= 0:
            first = tl.load(Meta + tile * 5)
            bit = (1 << (row - first)).to(tl.int32)
            tl.atomic_or(Dense + tile * MAX_BLOCKS + ordered, bit,
                         full & (ordered >= 0) & (ordered < MAX_BLOCKS), sem="relaxed")
            if tail != 0:
                tl.atomic_or(Dense + tile * MAX_BLOCKS + visible // 4, bit,
                             (visible >= 0) & (visible // 4 < MAX_BLOCKS), sem="relaxed")


@triton.jit
def _masks(Blocks, Membership, Masks, tile, count, common_tiles, rows, position, length,
           QB: tl.constexpr, CAP, B: tl.constexpr, part, PARTS: tl.constexpr):
    columns = tl.arange(0, B)
    end = tl.cdiv(count, 16) * QB * 4
    for start in tl.range(common_tiles * QB * 4 + part * B, end, B * PARTS, loop_unroll_factor=1):
        index = start + columns
        nt = index // (QB * 4)
        query = (index // 4) % QB
        quarter = index % 4
        visible = tl.minimum(position + query + 1, length)
        result = tl.full((B,), 0, tl.int32)
        for group in tl.static_range(4):
            slot = nt * 16 + quarter * 2 + (group // 2) * 8 + group % 2
            valid = (slot < count) & (query < rows) & (index < end)
            member = tl.load(Membership + tile * CAP + slot, valid, 0).to(tl.uint32)
            block = tl.load(Blocks + tile * CAP + slot, valid, 0)
            valid &= ((member >> query) & 1) != 0
            prefix = tl.minimum(tl.maximum(visible - block * 4, 0), 4)
            result |= tl.where(valid, (1 << prefix) - 1, 0) << (group * 4)
        tl.store(Masks + tile * (CAP // 16) * QB * 4 + index, result, index < end)


@triton.jit(do_not_specialize=["MAX_BLOCKS", "CAPACITY"])
def attention_compact(Dense, Blocks, Membership, Counts, Meta, Active,
            MAX_BLOCKS, CAPACITY, C: tl.constexpr,
            RHO: tl.constexpr, PACKED_DIRECT: tl.constexpr = False,
        CLEAR: tl.constexpr = False, REVERSE: tl.constexpr = False):
    tile = tl.num_programs(0) - 1 - tl.program_id(0) if REVERSE else tl.program_id(0)
    lanes = tl.arange(0, C)
    rows = tl.load(Meta + tile * 5 + 1)
    position = tl.load(Meta + tile * 5 + 4)
    limit = MAX_BLOCKS
    if CLEAR:
        limit = tl.minimum(MAX_BLOCKS, tl.cdiv(position + rows, 4))
    all_queries = tl.full((), 0xFFFFFFFF, tl.uint32) >> (32 - rows)
    count = tl.full((), 0, tl.int32)
    common_count = tl.full((), 0, tl.int32)
    for start in tl.range(0, limit, C, loop_unroll_factor=1):
        ids = start + lanes
        bits = tl.load(Dense + tile * MAX_BLOCKS + ids, ids < MAX_BLOCKS, 0)
        valid = bits != 0
        common = valid & (bits.to(tl.uint32) == all_queries) & (ids * 4 + 3 <= position)
        count += tl.sum(valid.to(tl.int32))
        common_count += tl.sum(common.to(tl.int32))
    queries = tl.arange(0, 32)
    visible = position + queries + 1
    if PACKED_DIRECT:
        tokens = tl.minimum(visible // 4, 512) * 4 + visible % 4
        direct_tiles = tl.sum(tl.where(queries < rows, tl.cdiv(tokens, 32), 0))
        enabled = 160 * tl.cdiv(count, 16) <= 17 * direct_tiles
    else:
        selected = tl.minimum(visible // 4, 512) + (visible % 4 != 0).to(tl.int32)
        total = tl.sum(tl.where(queries < rows, selected, 0))
        enabled = count * rows <= RHO * total
    tl.store(Active + tile, enabled)
    tl.store(Counts + tile * 2, count)
    tl.store(Counts + tile * 2 + 1, tl.where(enabled, common_count // 16, 0))
    if enabled:
        common_base = tl.full((), 0, tl.int32)
        other_base = common_count
        for start in tl.range(0, limit, C, loop_unroll_factor=1):
            ids = start + lanes
            bits = tl.load(Dense + tile * MAX_BLOCKS + ids, ids < MAX_BLOCKS, 0)
            valid = bits != 0
            common = valid & (bits.to(tl.uint32) == all_queries) & (ids * 4 + 3 <= position)
            other = valid & ~common
            destination = tl.where(common, common_base + tl.cumsum(common.to(tl.int32)) - 1,
                                   other_base + tl.cumsum(other.to(tl.int32)) - 1)
            tl.store(Blocks + tile * CAPACITY + destination, ids, valid)
            tl.store(Membership + tile * CAPACITY + destination, bits, valid)
            common_base += tl.sum(common.to(tl.int32))
            other_base += tl.sum(other.to(tl.int32))
            if CLEAR:
                tl.store(Dense + tile * MAX_BLOCKS + ids, 0, ids < MAX_BLOCKS)
    elif CLEAR:
        for start in tl.range(0, limit, C, loop_unroll_factor=1):
            ids = start + lanes
            tl.store(Dense + tile * MAX_BLOCKS + ids, 0, ids < MAX_BLOCKS)


@triton.jit
def _order(Counts, Active, Order, program,
           TASKS, HK: tl.constexpr, SLICES: tl.constexpr, GRID,
           SIZE: tl.constexpr, SHIFT: tl.constexpr, WIDE: tl.constexpr):
    rank = program * SIZE + tl.arange(0, SIZE)
    tile = (rank % (TASKS // HK)) // SLICES
    valid = rank < TASKS
    count = tl.load(Counts + tile * 2, valid, 0)
    active = tl.load(Active + tile, valid, 0)
    cost = tl.where(active != 0, tl.cdiv(count, 16), 0)
    if WIDE:
        cost = cost.to(tl.int64)
    key = tl.where(valid, (cost << SHIFT) + (TASKS - 1 - rank), -1)
    ordered = tl.sort(key, descending=True)
    task = tl.where(ordered >= 0, TASKS - 1 - (ordered & ((1 << SHIFT) - 1)), -1)
    row, column = rank // GRID, rank % GRID
    destination = row * GRID + tl.where(row % 2 == 0, column, GRID - 1 - column)
    tl.store(Order + destination, task.to(tl.int32), destination < tl.cdiv(TASKS, GRID) * GRID)


@triton.jit(do_not_specialize=["ROWS", "TASKS", "GRID", "ORDER_CTAS", "CAP", "TILES"])
def attention_order_masks_validate(Errors, Valid, Counts=None, Active=None, Order=None,
           ROWS=0, TASKS=0, HK: tl.constexpr = 1,
           SLICES: tl.constexpr = 1, GRID=1, SIZE: tl.constexpr = 1,
           SHIFT: tl.constexpr = 1, WIDE: tl.constexpr = False, CHECK: tl.constexpr = True,
           Blocks=None, Membership=None, Meta=None, Masks=None, ORDER_CTAS=1,
           QB: tl.constexpr = 1, CAP=16, B: tl.constexpr = 256,
           PARTS: tl.constexpr = 4, REVERSE: tl.constexpr = False, TILES=1):
    program = tl.program_id(0)
    if Masks is None or program < ORDER_CTAS:
        if Counts is not None:
            _order(Counts, Active, Order, program, TASKS, HK, SLICES, GRID, SIZE, SHIFT, WIDE)
        if CHECK and program == 0:
            lanes = tl.arange(0, 1024)
            error = tl.full((), 0, tl.int32)
            for start in tl.range(0, ROWS, 1024, loop_unroll_factor=1):
                values = tl.load(Errors + start + lanes, start + lanes < ROWS, 0)
                error |= tl.max((values != 0).to(tl.int32))
            tl.store(Valid, error == 0)
    else:
        task = program - ORDER_CTAS
        tile, part = task // PARTS, task % PARTS
        if REVERSE:
            tile = TILES - 1 - tile
        if tl.load(Active + tile):
            count = tl.load(Counts + tile * 2)
            common = tl.load(Counts + tile * 2 + 1)
            rows = tl.load(Meta + tile * 5 + 1)
            position = tl.load(Meta + tile * 5 + 4)
            length = tl.load(Meta + tile * 5 + 3)
            _masks(Blocks, Membership, Masks, tile, count, common, rows, position, length,
                   QB, CAP, B, part, PARTS)


@triton.jit(do_not_specialize=["MAX_BLOCKS"])
def attention_scatter_prepared(Blocks, Positions, Meta, Dense, MAX_BLOCKS):
    # Prepared-only callers may rebuild a separate plan from recovered blocks.
    tile, local = tl.program_id(0), tl.program_id(1)
    first = tl.load(Meta + tile * 5)
    rows = tl.load(Meta + tile * 5 + 1)
    if local < rows:
        row = first + local
        blocks = tl.load(Blocks + row * 512 + tl.arange(0, 512))
        bit = (1 << local).to(tl.int32)
        tl.atomic_or(Dense + tile * MAX_BLOCKS + blocks, bit, (blocks >= 0) & (blocks < MAX_BLOCKS), sem="relaxed")
        visible = tl.load(Positions + row) + 1
        if visible % 4 != 0:
            tl.atomic_or(Dense + tile * MAX_BLOCKS + visible // 4, bit, sem="relaxed")


def allocate_plan(*, inputs, query_tile: int, grid_multiplier: int = 2,
                  max_union_inflation: float = float("inf"), skip_counts=None) -> SparsePlan:
    group = inputs.q.shape[1] // inputs.k.shape[1]
    if query_tile < 1 or query_tile > 32:
        raise ValueError("Require 1<=BQ<=32")
    skips = (0,) * len(inputs.query_lens) if skip_counts is None else skip_counts
    num_cus = torch.cuda.get_device_properties(inputs.q.device).multi_processor_count
    # Full H6 prefill fills 126/128 MFMA rows. Keep the established tiles for
    # prefix/ragged work and batches too small to fill the persistent grid.
    balanced = (group == 6 and inputs.k.shape[1] == 1 and query_tile >= 21
                and len(inputs.query_lens) == 1 and inputs.prefix_lens[0] == 0
                and inputs.query_lens[0] - skips[0] >= 21 * num_cus * grid_multiplier)
    max_queries = 128 // group if group == 12 else 1 << ((128 // group).bit_length() - 1)
    if balanced:
        max_queries = 21
    query_tile = min(query_tile, max_queries)
    rows, q0, k0 = [], 0, 0
    for q_len, prefix, skip in zip(inputs.query_lens, inputs.prefix_lens, skips):
        local = skip
        remaining = triton.cdiv(q_len - skip, query_tile)
        while local < q_len:
            # Avoid a tiny first/last tile selecting direct for only a few rows.
            end = (local + triton.cdiv(q_len - local, remaining) if balanced
                   else min(q_len, (local // query_tile + 1) * query_tile))
            rows.append((q0 + local, end - local, k0, q_len + prefix, prefix + local))
            local = end
            remaining -= 1
        q0 += q_len
        k0 += q_len + prefix
    metadata = torch.from_numpy(np.asarray(rows, dtype=np.int32).reshape(-1, 5)).to(inputs.q.device)
    query_tiles = np.full(inputs.q.shape[0], -1, dtype=np.int32)
    for i, (start, n, *_) in enumerate(rows):
        query_tiles[start:start + n] = i
    max_blocks = triton.cdiv(inputs.max_seqlen_k, 4)
    capacity = triton.cdiv(min(max_blocks, query_tile * 513), 16) * 16
    tiles = len(rows)
    tasks = tiles * inputs.k.shape[1] * triton.cdiv(query_tile * group, 128)
    grid = min(tasks, num_cus * (1 if group == 12 else grid_multiplier))
    kwargs = {"dtype": torch.int32, "device": inputs.q.device}
    return SparsePlan(
        metadata=metadata, dense_membership=torch.zeros((tiles, max_blocks), **kwargs),
        blocks=torch.empty((tiles, capacity), **kwargs), membership=torch.empty((tiles, capacity), **kwargs),
        score_masks=torch.empty((tiles, capacity // 16, query_tile, 4), **kwargs),
        counts=torch.empty((tiles, 2), **kwargs), active=torch.empty(tiles, **kwargs),
        task_order=torch.empty(triton.cdiv(tasks, grid) * grid if grid else 0, **kwargs),
        query_tiles=torch.from_numpy(query_tiles).to(inputs.q.device), query_tile=query_tile,
        group_padded=group, block_capacity=capacity, max_blocks=max_blocks, num_tiles=tiles,
        grid=grid, max_union_inflation=max_union_inflation)


def _finish(inputs, plan, errors=None, valid=None):
    if plan is None or plan.num_tiles == 0:
        if errors is not None:
            attention_order_masks_validate[(1,)](errors, valid, ROWS=inputs.q.shape[0], num_warps=4)
        return
    reverse = len(inputs.query_lens) == 1
    attention_compact[(plan.num_tiles,)](plan.dense_membership, plan.blocks, plan.membership,
        plan.counts, plan.metadata, plan.active, plan.max_blocks, plan.block_capacity,
        min(1024, triton.next_power_of_2(plan.max_blocks)), plan.max_union_inflation,
        plan.packed_direct, CLEAR=True, REVERSE=reverse, num_warps=4)
    slices = triton.cdiv(plan.query_tile * plan.group_padded, 128)
    tasks = plan.num_tiles * inputs.k.shape[1] * slices
    size = min(4096, triton.next_power_of_2(plan.task_order.numel()))
    shift = max(1, (tasks - 1).bit_length())
    wide = ((plan.block_capacity // 16) << shift) + tasks - 1 >= 2**31
    order_ctas = triton.cdiv(plan.task_order.numel(), size)
    warps = 8 if size > 1024 else 4
    attention_order_masks_validate[(order_ctas + plan.num_tiles * 4,)](errors, valid, plan.counts, plan.active,
        plan.task_order, inputs.q.shape[0], tasks, inputs.k.shape[1], slices, plan.grid,
        size, shift, wide, errors is not None, plan.blocks, plan.membership, plan.metadata,
        plan.score_masks, order_ctas, plan.query_tile, plan.block_capacity, warps * 64,
        4, reverse, plan.num_tiles, num_warps=warps)


def run(*, inputs, plan, errors, valid):
    attention_recover_scatter[(inputs.q.shape[0],)](inputs.indices, inputs.query_positions, inputs.kv_lens,
        inputs.query_sequence_ids, inputs.block_indices, errors,
        plan.dense_membership if plan is not None else None,
        plan.query_tiles if plan is not None else None, plan.metadata if plan is not None else None,
        plan.max_blocks if plan is not None else 0, REVERSE=len(inputs.query_lens) == 1, num_warps=1)
    _finish(inputs, plan, errors, valid)
    torch._assert_async(valid, "Invalid compressed QSA token/block/tail ABI")


def rebuild_plan(*, inputs, plan):
    if plan.num_tiles:
        attention_scatter_prepared[(plan.num_tiles, plan.query_tile)](inputs.block_indices, inputs.query_positions,
                                                  plan.metadata, plan.dense_membership, plan.max_blocks, num_warps=4)
        _finish(inputs, plan)