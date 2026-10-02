"""Exact QSA attention selection, private union scratch and fused preparation launches.

Recovery validates every selection row and traps on an invalid one. Dense membership is
zero when a plan is allocated and after every completed build. Recovery scatters into it; its
owning compact CTA consumes and clears the same causal prefix. The next launch sorts tasks
and builds disjoint mask partitions. Plans use one stream's scratch and are pinned for graphs.
"""

import msgspec
import numpy as np
import torch
import triton
from triton import knobs
import triton.language as tl

# Routing latency model in microseconds, fitted on MI308X (gfx942, 80 CUs) phase timings.
UNION_FIXED = tl.constexpr(10.0)
UNION_STEP = tl.constexpr(3.2)  # per N64 step of the slowest union CTA
ROW_STEP = tl.constexpr(0.7)  # per BN32 step of the longest direct row
ROUTE_MARGIN = tl.constexpr(0.05)
DIRECT_FIXED, PACK_BLOCK, DIRECT_CU_STEP = 41.0, 0.0041, 0.304
PROMOTE_RATIO = 1.8
# K3 holds every task-sort width from 2**7 to 2**12 and picks one per launch, so task counts
# never compile a new variant. Four warps keep the mask CTAs fine-grained.
SORT_MIN_LOG, SORT_MAX_LOG = tl.constexpr(7), tl.constexpr(12)
ORDER_WARPS = 4


class SparsePlan(msgspec.Struct, kw_only=True):
    metadata: torch.Tensor
    dense_membership: torch.Tensor
    blocks: torch.Tensor
    membership: torch.Tensor
    score_masks: torch.Tensor
    counts: torch.Tensor
    costs: torch.Tensor
    active: torch.Tensor
    task_order: torch.Tensor
    query_tiles: torch.Tensor
    direct_flag: torch.Tensor
    query_tile: int
    group_padded: int
    block_capacity: int
    max_blocks: int
    num_tiles: int
    grid: int
    route_costs: tuple
    union_ratio: float = 1.7
    routing: bool = True


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


@triton.jit
def _trap(error):
    if error:
        # llvm.trap: halts the wave and puts the queue into the error state.
        tl.inline_asm_elementwise("s_trap 2", "=v,v", [error.to(tl.int32)], dtype=tl.int32, is_pure=False, pack=1)


@triton.jit(do_not_specialize=["MAX_BLOCKS"])
def attention_recover_scatter(Indices, Positions, Lengths, SequenceIds, Blocks, Errors=None,
            Dense=None, QueryTiles=None, Meta=None, MAX_BLOCKS=0, MEMBERS64: tl.constexpr = False):
    # Last rows first: in a causal request they reach the most blocks.
    row = tl.num_programs(0) - 1 - tl.program_id(0)
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
    # Rows that select their first blocks in order (complete causal prefixes) skip the sort.
    # Measured on MI308X, an ascending check on other rows costs about what skipping saves.
    if tl.sum((full & (blocks != columns)).to(tl.int32)) != 0:
        ordered = _sort_blocks(ordered)
        previous = tl.gather(ordered, tl.maximum(columns - 1, 0), axis=0)
        error |= tl.sum((full & (columns > 0) & (ordered == previous)).to(tl.int32)) != 0
    # Calls trap on an invalid row; an error buffer (tests) records it instead.
    if Errors is not None:
        tl.store(Errors + row, error.to(tl.int32))
    else:
        _trap(error)
    tl.store(Blocks + row * 512 + columns, tl.where(full, ordered, -1))
    if Dense is not None:
        tile = tl.load(QueryTiles + row)
        if tile >= 0:
            first = tl.load(Meta + tile * 5)
            if MEMBERS64:
                bit = tl.full((), 1, tl.int64) << (row - first).to(tl.int64)
            else:
                bit = (1 << (row - first)).to(tl.int32)
            tl.atomic_or(Dense + tile * MAX_BLOCKS + ordered, bit,
                         full & (ordered >= 0) & (ordered < MAX_BLOCKS), sem="relaxed")
            if tail != 0:
                tl.atomic_or(Dense + tile * MAX_BLOCKS + visible // 4, bit,
                             (visible >= 0) & (visible // 4 < MAX_BLOCKS), sem="relaxed")


@triton.jit
def _slot_bits(Membership, Blocks, slot, valid, MEMBERS64: tl.constexpr):
    if MEMBERS64:
        member = tl.load(Membership + slot, valid, 0).to(tl.uint64)
    else:
        member = tl.load(Membership + slot, valid, 0).to(tl.uint32)
    block = tl.load(Blocks + slot, valid, 0)
    return member, block * 4


@triton.jit
def _query_bits(member, start, query, visible, live):
    hit = (((member >> query) & 1) != 0) & live
    return tl.where(hit, (1 << tl.minimum(tl.maximum(visible - start, 0), 4)) - 1, 0)


@triton.jit
def _masks(Blocks, Membership, Masks, Counts, Costs, Meta, program, TILES,
           QB, CAP, B: tl.constexpr, PARTS: tl.constexpr, MEMBERS64: tl.constexpr):
    # Entry (nt, query, quarter) holds 4 bits for each of the slots nt*16 + quarter*2 +
    # (group//2)*8 + group%2. One lane owns an (nt, quarter) pair: its four slots are
    # loaded once and shared by every query. Each wave owns one (tile, part); a part
    # takes 16 of every 16*PARTS slot groups.
    lanes = tl.arange(0, B)
    task = program * (B // 64) + lanes // 64
    tile, part = TILES - 1 - task // PARTS, task % PARTS
    live = task < TILES * PARTS
    # Proposal-2 tiles are built too: routing may promote them in this launch.
    live &= tl.load(Costs + tile * 3 + 2, live, 0) != 0
    count = tl.load(Counts + tile, live, 0)
    first = part * 16 + (lanes % 64) // 4
    rows = tl.load(Meta + tile * 5 + 1, live, 0)
    position = tl.load(Meta + tile * 5 + 4, live, 0)
    length = tl.load(Meta + tile * 5 + 3, live, 0)
    quarter = lanes % 4
    Membership += tile * CAP
    Blocks += tile * CAP
    Masks += tile * (CAP // 16) * QB * 4 + quarter
    steps = tl.where(live, tl.cdiv(tl.maximum(tl.cdiv(count, 16) - first, 0), 16 * PARTS), 0)
    for step in tl.range(0, tl.max(steps), loop_unroll_factor=1):
        nt = first + step * 16 * PARTS
        valid = live & (nt * 16 < count)
        slot = nt * 16 + quarter * 2
        m0, s0 = _slot_bits(Membership, Blocks, slot, valid & (slot < count), MEMBERS64)
        m1, s1 = _slot_bits(Membership, Blocks, slot + 1, valid & (slot + 1 < count), MEMBERS64)
        m2, s2 = _slot_bits(Membership, Blocks, slot + 8, valid & (slot + 8 < count), MEMBERS64)
        m3, s3 = _slot_bits(Membership, Blocks, slot + 9, valid & (slot + 9 < count), MEMBERS64)
        for query in tl.range(0, QB, loop_unroll_factor=1):
            visible = tl.minimum(position + query + 1, length)
            ok = query < rows
            result = _query_bits(m0, s0, query, visible, ok)
            result |= _query_bits(m1, s1, query, visible, ok) << 4
            result |= _query_bits(m2, s2, query, visible, ok) << 8
            result |= _query_bits(m3, s3, query, visible, ok) << 12
            tl.store(Masks + nt * QB * 4 + query * 4, result.to(tl.int16), valid)


@triton.jit
def _compact_scan(Dense, Blocks, Membership, tile, end, budget, MAX_BLOCKS, CAPACITY, C: tl.constexpr):
    # Recovery only scatters within the tile's causal range. Copy the union out in block order
    # and zero what was set; past the budget the tile is direct, its count partial, and the
    # rest is only zeroed.
    lanes = tl.arange(0, C)
    count = tl.full((), 0, tl.int32)
    start = tl.full((), 0, tl.int32)
    while (start < end) & ((16 * tl.cdiv(count, 16)).to(tl.float32) <= budget):
        ids = start + lanes
        bits = tl.load(Dense + tile * MAX_BLOCKS + ids, ids < end, 0)
        valid = bits != 0
        destination = count + tl.cumsum(valid.to(tl.int32)) - 1
        tl.store(Blocks + tile * CAPACITY + destination, ids, valid)
        tl.store(Membership + tile * CAPACITY + destination, bits, valid)
        tl.store(Dense + tile * MAX_BLOCKS + ids, tl.zeros_like(bits), valid)
        count += tl.sum(valid.to(tl.int32))
        start += C
    for rest in tl.range(start, end, C, loop_unroll_factor=1):
        tl.store(Dense + tile * MAX_BLOCKS + rest + lanes, 0, rest + lanes < end)
    return count, start


@triton.jit(do_not_specialize=["MAX_BLOCKS", "CAPACITY"])
def attention_compact(Dense, Blocks, Membership, Counts, Costs, Meta, DirectFlag,
                      MAX_BLOCKS, CAPACITY, RATIO, PROMOTE, MEMBERS64: tl.constexpr = False):
    if tl.program_id(0) == 0:
        tl.store(DirectFlag, 0)
    # Last tile first: in a causal request it holds the most blocks.
    tile = tl.num_programs(0) - 1 - tl.program_id(0)
    rows = tl.load(Meta + tile * 5 + 1)
    position = tl.load(Meta + tile * 5 + 4)
    queries = tl.arange(0, 64 if MEMBERS64 else 32)
    visible = position + queries + 1
    # Union pads blocks to N64 steps; one step equals sixteen BN32 direct steps.
    tokens = tl.minimum(visible // 4, 512) * 4 + visible % 4
    steps = tl.where(queries < rows, tl.cdiv(tokens, 32), 0)
    direct_tiles = tl.sum(steps)
    # Candidates up to PROMOTE are compacted too, so routing may promote them.
    budget = PROMOTE * direct_tiles.to(tl.float32)
    # Membership words hold one bit per query row (64-bit for BQ > 32). The scan width follows
    # MAX_BLOCKS at run time, so KV lengths never compile a new variant.
    end = tl.minimum(MAX_BLOCKS, tl.cdiv(position + rows, 4))
    if MAX_BLOCKS <= 256:
        count, start = _compact_scan(Dense, Blocks, Membership, tile, end, budget, MAX_BLOCKS, CAPACITY, 256)
    elif MAX_BLOCKS <= 512:
        count, start = _compact_scan(Dense, Blocks, Membership, tile, end, budget, MAX_BLOCKS, CAPACITY, 512)
    else:
        count, start = _compact_scan(Dense, Blocks, Membership, tile, end, budget, MAX_BLOCKS, CAPACITY, 1024)
    padded = (16 * tl.cdiv(count, 16)).to(tl.float32)
    candidate = (start >= end) & (padded <= budget)
    proposal = tl.where(candidate & (padded <= RATIO * direct_tiles.to(tl.float32)), 1, tl.where(candidate, 2, 0))
    tl.store(Counts + tile, count)
    tl.store(Costs + tile * 3, direct_tiles)
    tl.store(Costs + tile * 3 + 1, tl.max(steps))
    tl.store(Costs + tile * 3 + 2, proposal)


@triton.jit
def _route_cost(union_sum, union_max, direct_sum, direct_max, TASK_SHARE, DIRECT_STEP, DIRECT_FIXED):
    union = UNION_FIXED + UNION_STEP * tl.maximum(union_max.to(tl.float32), union_sum.to(tl.float32) * TASK_SHARE)
    direct = DIRECT_FIXED + tl.maximum(ROW_STEP * direct_max.to(tl.float32), DIRECT_STEP * direct_sum.to(tl.float32))
    return tl.where(union_sum > 0, union, 0.0) + tl.where(direct_sum > 0, direct, 0.0)


@triton.jit
def _route(Counts, Costs, TILES, TASK_SHARE, DIRECT_STEP, DIRECT_FIXED, C: tl.constexpr, J: tl.constexpr):
    # Union and direct run back to back, so one long union task or a few direct
    # rows set a latency floor the per-tile rule cannot see. Compare the
    # per-tile proposal 1 with all-direct, promoting the proposal-2 tiles, and
    # demoting union tiles above J thresholds; return (mode, threshold).
    lanes = tl.arange(0, C)
    shares = tl.exp2(-0.5 * (tl.arange(0, J) + 1).to(tl.float32))
    # One wave of lanes accumulates and reduces without cross-wave exchange,
    # and stays small for the mask CTAs sharing this launch's registers.
    zero = tl.zeros((C,), tl.int32)
    steps1, steps2, direct0, direct1, direct2 = zero, zero, zero, zero, zero
    max1, max2, row0, row1, row2 = zero, zero, zero, zero, zero
    for start in tl.range(0, TILES, C, loop_unroll_factor=1):
        index = start + lanes
        valid = index < TILES
        steps = tl.cdiv(tl.load(Counts + index, valid, 0), 16)
        direct = tl.load(Costs + index * 3, valid, 0)
        row = tl.load(Costs + index * 3 + 1, valid, 0)
        proposal = tl.load(Costs + index * 3 + 2, valid, 0)
        steps1 += tl.where(proposal == 1, steps, 0)
        steps2 += tl.where(proposal == 2, steps, 0)
        direct0 += tl.where(proposal == 0, direct, 0)
        direct1 += tl.where(proposal == 1, direct, 0)
        direct2 += tl.where(proposal == 2, direct, 0)
        max1 = tl.maximum(max1, tl.where(proposal == 1, steps, 0))
        max2 = tl.maximum(max2, tl.where(proposal == 2, steps, 0))
        row0 = tl.maximum(row0, tl.where(proposal == 0, row, 0))
        row1 = tl.maximum(row1, tl.where(proposal == 1, row, 0))
        row2 = tl.maximum(row2, tl.where(proposal == 2, row, 0))
    steps1, steps2, max1, max2 = tl.sum(steps1), tl.sum(steps2), tl.max(max1), tl.max(max2)
    direct0, direct1, direct2 = tl.sum(direct0), tl.sum(direct1), tl.sum(direct2)
    row0, row1, row2 = tl.max(row0), tl.max(row1), tl.max(row2)
    thresholds = (max1.to(tl.float32) * shares).to(tl.int32)
    k_sum = tl.zeros((J, C), tl.int32)
    k_max = tl.zeros((J, C), tl.int32)
    d_sum = tl.zeros((J, C), tl.int32)
    d_max = tl.zeros((J, C), tl.int32)
    # Demotion only shortens union when its longest task, not throughput, sets its time.
    latency_bound = max1.to(tl.float32) > steps1.to(tl.float32) * TASK_SHARE
    for start in tl.range(0, tl.where(latency_bound, TILES, 0), C, loop_unroll_factor=1):
        index = start + lanes
        valid = index < TILES
        steps = tl.cdiv(tl.load(Counts + index, valid, 0), 16)
        direct = tl.load(Costs + index * 3, valid, 0)
        row = tl.load(Costs + index * 3 + 1, valid, 0)
        proposal = tl.load(Costs + index * 3 + 2, valid, 0)
        keep = (proposal[None, :] == 1) & (steps[None, :] <= thresholds[:, None])
        k_sum += tl.where(keep, steps[None, :], 0)
        k_max = tl.maximum(k_max, tl.where(keep, steps[None, :], 0))
        d_sum += tl.where(keep, 0, direct[None, :])
        d_max = tl.maximum(d_max, tl.where(keep, 0, row[None, :]))
    k_sum, k_max = tl.sum(k_sum, axis=1), tl.max(k_max, axis=1)
    d_sum, d_max = tl.sum(d_sum, axis=1), tl.max(d_max, axis=1)
    zero = tl.full((), 0, tl.int32)
    formula = _route_cost(steps1, max1, direct0 + direct2, tl.maximum(row0, row2),
                          TASK_SHARE, DIRECT_STEP, DIRECT_FIXED)
    all_direct = _route_cost(zero, zero, direct0 + direct1 + direct2, tl.maximum(tl.maximum(row0, row1), row2),
                             TASK_SHARE, DIRECT_STEP, DIRECT_FIXED)
    promoted = _route_cost(steps1 + steps2, tl.maximum(max1, max2), direct0, row0,
                           TASK_SHARE, DIRECT_STEP, DIRECT_FIXED)
    promoted = tl.where(steps2 > 0, promoted, float("inf"))
    demoted = _route_cost(k_sum, k_max, d_sum, d_max, TASK_SHARE, DIRECT_STEP, DIRECT_FIXED)
    demoted = tl.where(latency_bound, demoted, float("inf"))
    choice = tl.argmin(demoted, axis=0)
    best_demoted = tl.min(demoted, axis=0)
    threshold = tl.sum(tl.where(tl.arange(0, J) == choice, thresholds, 0))
    best = tl.minimum(tl.minimum(all_direct, promoted), best_demoted)
    # 0 keeps the proposal, 1 all direct, 2 promotes, 3 demotes above threshold.
    mode = tl.where(best >= formula * (1.0 - ROUTE_MARGIN), 0,
                    tl.where(best == all_direct, 1, tl.where(best == promoted, 2, 3)))
    return mode, threshold


@triton.jit
def _routed(proposal, steps, mode, threshold):
    keep = (mode != 1) & tl.where(mode == 2, proposal != 0, proposal == 1)
    return keep & ((mode != 3) | (steps <= threshold))


@triton.jit
def _order(Counts, Costs, Active, Order, DirectFlag, program, mode, threshold,
           TASKS, HK: tl.constexpr, SLICES: tl.constexpr, GRID,
           SIZE: tl.constexpr, SHIFT, WIDE: tl.constexpr):
    rank = program * SIZE + tl.arange(0, SIZE)
    tile = (rank % (TASKS // HK)) // SLICES
    valid = rank < TASKS
    count = tl.load(Counts + tile, valid, 0)
    active = _routed(tl.load(Costs + tile * 3 + 2, valid, 0), tl.cdiv(count, 16), mode, threshold)
    # The first head's first slice of each tile publishes the route.
    publish = valid & (rank < TASKS // HK) & (rank % SLICES == 0)
    tl.store(Active + tile, active.to(tl.int32), publish)
    # Any direct tile: compact cleared the flag; gated pack/direct launches read this one word.
    tl.store(DirectFlag, 1, tl.max((publish & ~active).to(tl.int32)) != 0)
    cost = tl.where(active, tl.cdiv(count, 16), 0)
    if WIDE:
        cost = cost.to(tl.int64)
    key = tl.where(valid, (cost << SHIFT) + (TASKS - 1 - rank), -1)
    ordered = tl.sort(key, descending=True)
    task = TASKS - 1 - (ordered & ((tl.full((), 1, ordered.dtype) << SHIFT) - 1))
    # Sort hkv-major (tie order), publish the tile-major task the union kernel runs.
    task = tl.where(ordered >= 0, (task % (TASKS // HK)) * HK + task // (TASKS // HK), -1)
    row, column = rank // GRID, rank % GRID
    destination = row * GRID + tl.where(row % 2 == 0, column, GRID - 1 - column)
    tl.store(Order + destination, task.to(tl.int32), destination < tl.cdiv(TASKS, GRID) * GRID)


@triton.jit(do_not_specialize=["TASKS", "GRID", "SIZE", "SHIFT", "ORDER_CTAS", "QB", "CAP", "TILES"])
def attention_order_masks(Counts, Costs, Active, Order, DirectFlag, Blocks, Membership, Meta, Masks,
                          TASKS, HK: tl.constexpr, SLICES: tl.constexpr, GRID, SIZE,
                          SHIFT, WIDE: tl.constexpr, ORDER_CTAS, QB, CAP,
                          B: tl.constexpr, PARTS: tl.constexpr, TILES, TASK_SHARE, DIRECT_STEP, DIRECT_FIXED,
                          ROUTE: tl.constexpr, MEMBERS64: tl.constexpr):
    program = tl.program_id(0)
    if program < ORDER_CTAS:
        mode, threshold = tl.full((), 0, tl.int32), tl.full((), 0, tl.int32)
        if ROUTE:
            mode, threshold = _route(Counts, Costs, TILES, TASK_SHARE, DIRECT_STEP, DIRECT_FIXED, 64, 8)
        for exponent in tl.static_range(SORT_MIN_LOG, SORT_MAX_LOG + 1):
            if SIZE == 1 << exponent:
                _order(Counts, Costs, Active, Order, DirectFlag, program, mode, threshold,
                       TASKS, HK, SLICES, GRID, 1 << exponent, SHIFT, WIDE)
    else:
        _masks(Blocks, Membership, Masks, Counts, Costs, Meta, program - ORDER_CTAS, TILES,
               QB, CAP, B, PARTS, MEMBERS64)


def upload(array, device) -> torch.Tensor:
    # Pinned and asynchronous: building a new layout never drains the stream.
    host = torch.from_numpy(np.ascontiguousarray(array, dtype=np.int32))
    return host.pin_memory().to(device, non_blocking=True)


def scratch_specs(plan: SparsePlan) -> list:
    """(field, shape, dtype, zeroed) of the per-call scratch; every launch rewrites what it reads."""
    tiles, capacity = plan.num_tiles, plan.block_capacity
    members = torch.int64 if plan.query_tile > 32 else torch.int32
    return [("dense_membership", (tiles, plan.max_blocks), members, True),
            ("blocks", (tiles, capacity), torch.int32, False),
            ("membership", (tiles, capacity), members, False),
            ("score_masks", (tiles, capacity // 16, plan.query_tile, 4), torch.int16, False)]


def allocate_scratch(owner, specs, device) -> None:
    for name, shape, dtype, zeroed in specs:
        setattr(owner, name, (torch.zeros if zeroed else torch.empty)(shape, dtype=dtype, device=device))


def allocate_plan(*, inputs, query_tile: int, grid_multiplier: int = 2, scratch: bool = True) -> SparsePlan:
    group = inputs.q.shape[1] // inputs.k.shape[1]
    if query_tile < 1 or query_tile > 32:
        raise ValueError("Require 1<=BQ<=32")
    num_cus = torch.cuda.get_device_properties(inputs.q.device).multi_processor_count
    # Full H6/H3 prefill fills 126/128 MFMA rows (BQ21 / BQ42 with 64-bit membership). Keep the
    # established tiles for prefix/ragged work and batches too small to fill the persistent grid.
    full = (inputs.k.shape[1] == 1 and len(inputs.query_lens) == 1 and inputs.prefix_lens[0] == 0)
    if group == 6 and full and query_tile >= 21 and inputs.query_lens[0] >= 21 * num_cus * grid_multiplier:
        query_tile = 21
    elif group == 3 and full and query_tile == 32 and inputs.query_lens[0] >= 42 * num_cus * grid_multiplier:
        query_tile = 42
    else:
        query_tile = min(query_tile, 128 // group if group == 12 else 1 << ((128 // group).bit_length() - 1))
    q_lens = np.asarray(inputs.query_lens, dtype=np.int64)
    lengths = q_lens + np.asarray(inputs.prefix_lens, dtype=np.int64)
    # Equal tiles per request, larger first: a tiny last tile would route its few rows direct, after union.
    counts = -(-q_lens // query_tile)
    request = np.repeat(np.arange(len(q_lens)), counts)
    index = np.arange(len(request)) - np.repeat(np.cumsum(counts) - counts, counts)
    base = (q_lens // np.maximum(counts, 1))[request]
    extra = (q_lens % np.maximum(counts, 1))[request]
    local = index * base + np.minimum(index, extra)
    sizes = base + (index < extra)
    metadata = np.stack(((np.cumsum(q_lens) - q_lens)[request] + local, sizes,
                         (np.cumsum(lengths) - lengths)[request], lengths[request],
                         (lengths - q_lens)[request] + local), axis=1)
    max_blocks = triton.cdiv(inputs.max_seqlen_k, 4)
    capacity = triton.cdiv(min(max_blocks, query_tile * 513), 16) * 16
    tiles = len(request)
    hk = inputs.k.shape[1]
    slices = triton.cdiv(query_tile * group, 128)
    tasks = tiles * hk * slices
    grid = min(tasks, num_cus * (1 if group == 12 else grid_multiplier))
    packed = int((-(-lengths // 4))[q_lens > 0].sum())
    # Per-layout terms of the route model: union step share per tile, direct cost per step, fixed direct cost.
    route_costs = (hk * slices / min(grid, num_cus) if grid else 0.0, DIRECT_CU_STEP * hk / num_cus,
                   DIRECT_FIXED + PACK_BLOCK * packed * hk)
    kwargs = {"dtype": torch.int32, "device": inputs.q.device}
    plan = SparsePlan(
        metadata=upload(metadata, inputs.q.device), dense_membership=None, blocks=None, membership=None,
        score_masks=None, counts=torch.empty(tiles, **kwargs), costs=torch.empty((tiles, 3), **kwargs),
        active=torch.empty(tiles, **kwargs),
        task_order=torch.empty(triton.cdiv(tasks, grid) * grid if grid else 0, **kwargs),
        direct_flag=torch.empty(1, **kwargs),
        query_tiles=upload(np.repeat(np.arange(tiles), sizes), inputs.q.device), query_tile=query_tile,
        group_padded=group, block_capacity=capacity, max_blocks=max_blocks, num_tiles=tiles,
        grid=grid, route_costs=route_costs)
    if scratch:
        allocate_scratch(plan, scratch_specs(plan), inputs.q.device)
    return plan


def _launches(inputs, plan):
    """(kernel, grid, args, kwargs) of K1-K3 in launch order; K1's first argument is the indices."""
    yield attention_recover_scatter, (inputs.q.shape[0],), (inputs.indices, inputs.query_positions,
        inputs.kv_lens, inputs.query_sequence_ids, inputs.block_indices, None,
        plan.dense_membership if plan is not None else None,
        plan.query_tiles if plan is not None else None, plan.metadata if plan is not None else None,
        plan.max_blocks if plan is not None else 0), dict(
        MEMBERS64=plan is not None and plan.query_tile > 32, num_warps=1)
    if plan is None or plan.num_tiles == 0:
        return
    promote = max(PROMOTE_RATIO, plan.union_ratio) if plan.routing else plan.union_ratio
    yield attention_compact, (plan.num_tiles,), (plan.dense_membership, plan.blocks, plan.membership,
        plan.counts, plan.costs, plan.metadata, plan.direct_flag, plan.max_blocks, plan.block_capacity,
        plan.union_ratio, promote), dict(MEMBERS64=plan.query_tile > 32, num_warps=4)
    slices = triton.cdiv(plan.query_tile * plan.group_padded, 128)
    tasks = plan.num_tiles * inputs.k.shape[1] * slices
    numel = plan.task_order.numel()
    size = min(1 << SORT_MAX_LOG.value, max(1 << SORT_MIN_LOG.value, triton.next_power_of_2(numel)))
    shift = max(1, (tasks - 1).bit_length())
    wide = ((plan.block_capacity // 16) << shift) + tasks - 1 >= 2**31
    order_ctas = triton.cdiv(numel, size)
    # Mask CTAs pack one (tile, part) per wave.
    yield attention_order_masks, (order_ctas + triton.cdiv(plan.num_tiles * 4, ORDER_WARPS),), (
        plan.counts, plan.costs, plan.active, plan.task_order, plan.direct_flag, plan.blocks, plan.membership,
        plan.metadata, plan.score_masks, tasks, inputs.k.shape[1], slices, plan.grid, size, shift, wide,
        order_ctas, plan.query_tile, plan.block_capacity, ORDER_WARPS * 64, 4, plan.num_tiles,
        *plan.route_costs), dict(ROUTE=plan.routing, MEMBERS64=plan.query_tile > 32, num_warps=ORDER_WARPS)


def run(*, inputs, plan):
    for kernel, grid, args, kwargs in _launches(inputs, plan):
        kernel[grid](*args, **kwargs)


def hooked():
    """Whether a Triton launch hook is registered (replays would bypass it)."""
    return any(hook is not None and not (isinstance(hook, knobs.HookChain) and not hook.calls)
               for hook in (knobs.runtime.launch_enter_hook, knobs.runtime.launch_exit_hook))


class Replay:
    """One Triton launch, issued once through the JIT dispatcher and then replayed without it.

    The replay reuses the compiled kernel and the bound arguments, so it skips binding,
    specialization and the cache lookup (about 35 of 45 host microseconds). With
    ``per_call_first`` the first argument is supplied on every call (and not retained); it
    must keep the specialization (16-byte alignment) of the first call.
    """

    __slots__ = ("launch", "head", "values")

    def __init__(self, kernel, grid, args, kwargs, per_call_first=False):
        compiled = kernel[grid](*args, **kwargs)
        bound = dict(zip((p.name for p in kernel.params), args), **kwargs)
        values = [bound[p.name] if p.name in bound else p.default for p in kernel.params]
        self.values = values[1:] if per_call_first else values
        grid = (*grid, 1, 1)
        self.head = (grid[0], grid[1], grid[2])
        self.launch = (compiled.run, compiled.function, compiled.packed_metadata)

    def __call__(self, stream, *first):
        run, function, metadata = self.launch
        run(*self.head, stream, function, metadata, None, None, None, *first, *self.values)


def replays(*, inputs, plan):
    """Run once like run() and return the K1-K3 replays for later calls with the same bindings.

    K1 takes the indices per call: ``replays[0](stream, indices)``.
    """
    return [Replay(*launch, per_call_first=index == 0) for index, launch in enumerate(_launches(inputs, plan))]