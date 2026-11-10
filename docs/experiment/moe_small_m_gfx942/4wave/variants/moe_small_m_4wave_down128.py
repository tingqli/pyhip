# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
# Variant: the current kernels with DOWN_TILE_N = 128 (Down 2 x 16 output columns per wave, full register layout for
# every slice count); A/B reference for the 128 x 256 Down tile of 4wave_128 (loaded by tools/ab_kernels.py; not part
# of the pyhip package).

"""gfx942 FP8 block-scale MoE GateUp and Down on 4-wave workgroups for sort blocks with at most 128 routes.

Routes come from aiter moe_sorting(256), the BM256 kernels' table (moe_gateup/down_a8w8_blockscale_gfx942_lds). A
256-row block with no route at row 128 holds at most 128 routes and runs here; the BM256 kernels run the others
(big_blocks_only=True), so the choice is made per block at run time. With <= 128 routes a BM256 tile is at least
half padding and the work is bound by the weights, read once per block whatever its route count
(docs/record/moe_small_m_gfx942.md).

- A workgroup has 4 waves (one per SIMD) and takes one block and one N tile: a 128 x 256 MFMA output tile
  (GateUp: 128 gate and the matching 128 up weight rows, 128 output columns; Down: 256 output columns). Every wave
  owns 4 x 16 weight rows (GateUp: 2 x 16 gate rows and the matching up rows). LDS is <= 18 KiB and VGPRs stay
  <= 256 ("amdgpu-waves-per-eu"="2,2"), so two workgroups share a CU: two waves per SIMD. Accumulators of 7 or 8
  slices (112 or 128 VGPRs) fit only with the light register layout: one partial per weight group, A read from LDS
  half a slice ahead, one A staging slot (schedule()).
- Each K block is a compute phase (LDS reads of A, MFMAs, the dequant of finished partials) and a memory phase
  without VALU work (the A loads of a later K block and, between two barriers, the next A tile into LDS). Each
  workgroup syncs only its own waves, so the two on a CU drift apart and one's MFMAs cover the other's memory phase.
- B is aiter shuffle_weight(16,16): a 16 B load per lane is MFMA-ready, so each wave streams its weight rows
  straight into VGPRs through a ring of 2 K blocks, issued at the start of a compute phase without a wait.
  Loads past the last K block point out of the descriptor and move no data.
- A (GateUp: rows gathered by token; Down: the sorted FP8 rows GateUp wrote) and its scales are loaded into VGPRs
  two K blocks ahead (light layout: one), one scale per thread, and written into one 16 KiB LDS tile (plus a scale
  per row) that all four waves read. Vector memory instruction issue is the scarce resource of the K loop
  (docs/record): per K block a wave issues 4 B loads per weight-group pair, at most 4 A loads and 1 scale load.
- The rows a block uses are found at run time (ballot of its sorted IDs). The K loop and epilogue exist for 1 to
  8 sixteen-row slices and the block takes the smallest that covers its routes.
- Block scales: per K block the 4 chained MFMAs of a (slice, weight group) give a partial retired as
  acc += partial * (scale_a[row] * scale_b), one element per later MFMA.

GateUp writes the Down input: SiLU(gate) * up quantized to FP8 per (row, 128 columns), its N tile, with FP32 scale
amax / 240, at sorted row block * 256 + row of [blocks * 256, I] (scales [I/128, blocks * 256], the BM256 GateUp's
layout). Rows past the used slices are not written: Down never reads padding rows. Down adds routing weight *
(row @ W2^T) into out [tokens, H] with BF16 atomics.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from .gemm_a8w8_blockscale_gfx942_common import pack_bf16x2_rne
from .helpers import BufferTensor, run_compiled
from .moe_gateup_a8w8_blockscale_gfx942_lds import (
    FP8_MAX,
    LOG2E,
    OOB_OFFSET,
    SMALL_M_ROWS,
    SORT_BLOCK_M,
    TILE_K,
    TOKEN_MASK,
    _lds_dwords,
)

WAVES = 4
GATEUP_TILE_N = 128
DOWN_TILE_N = 128
# GateUp's FP8 output scale granularity: one task's N tile, as in the BM256 GateUp.
A2_SCALE_COLUMNS = GATEUP_TILE_N
ROW_DWORDS = TILE_K // 4
A_TILE_DWORDS = SMALL_M_ROWS * ROW_DWORDS
B_BLOCK_BYTES = 16 * TILE_K
# GateUp's FP8 tile in LDS: 128 B per row.
FP8_ROW_DWORDS = GATEUP_TILE_N // 4
# Down epilogue transpose: per wave 16 rows x 16 dwords; an even pitch keeps the 8 B writes aligned.
TRANSPOSE_PITCH = 18
SLICE_VARIANTS = (1, 2, 3, 4, 5, 6, 7, 8)
# From this many slices a kernel takes the light register layout (see compile_moe_small_m.schedule).
LIGHT_SLICES = {"gateup": 7, "down": 9}


def validate_parameters(mode, tokens, n_dim, k_dim, topk, num_experts, capacity):
    """mode "gateup": n_dim = inter_dim, k_dim = hidden_dim; "down": n_dim = hidden_dim, k_dim = inter_dim."""
    tile_n = GATEUP_TILE_N if mode == "gateup" else DOWN_TILE_N
    if not 0 < tokens <= TOKEN_MASK:
        raise ValueError(f"tokens must be in [1, {TOKEN_MASK}]")
    if n_dim <= 0 or n_dim % tile_n:
        raise ValueError(f"{mode}: N must be a positive multiple of {tile_n}")
    k_blocks = k_dim // TILE_K
    if k_dim <= 0 or k_dim % TILE_K or (2 * k_blocks if mode == "gateup" else k_blocks) > 64:
        raise ValueError(f"{mode}: K must be a multiple of 128 and the task's B scales must fit one VGPR")
    if not 0 < topk < 256 or num_experts <= 0:
        raise ValueError("require 0 < topk < 256 and num_experts > 0")
    weight_rows = 2 * n_dim if mode == "gateup" else n_dim
    extents = {"weight": weight_rows * k_dim}
    if mode == "gateup":
        extents.update(a=tokens * k_dim, a2=capacity * n_dim)
    else:
        extents.update(a2=capacity * k_dim, a2_scale=capacity * (k_dim // A2_SCALE_COLUMNS) * 4, output=tokens * n_dim * 2)
    # Offsets past a block still add up to 64 KiB of K-block soffset.
    over = [f"{name}={size:,}" for name, size in extents.items() if size >= OOB_OFFSET - (1 << 20)]
    if over:
        raise ValueError("buffer byte extents too large: " + ", ".join(over))


def _rsrc_of(ptr, num_records_bytes):
    tensor = fx.rocdl.make_buffer_tensor(fx.make_view(ptr, fx.make_layout(1, 1)), max_size=False, num_records_bytes=num_records_bytes)
    return fx.rocdl.get_buffer_rsrc(fx.get_iter(tensor))


@functools.cache
def compile_moe_small_m(mode, n_dim, k_dim, topk, num_experts, ring=2, b_aux=2, slice_variants=SLICE_VARIANTS):
    """Launcher of the GateUp ("gateup") or Down ("down") kernel; see the module doc.

    ring: K blocks of the per-wave B ring, only 2 (one in flight); b_aux: cache policy bits of the B loads (2 = nt).
    """
    assert mode in ("gateup", "down"), mode
    gateup = mode == "gateup"
    validate_parameters(mode, 1, n_dim, k_dim, topk, num_experts, SORT_BLOCK_M)
    KB = k_dim // TILE_K
    # One A staging slot: the prologue mirrors a ring of 2, and the K loop is unrolled by it.
    assert ring == 2 and KB % ring == 0, "ring must be 2 and divide K / 128"
    variants = sorted(set(slice_variants))
    assert variants[-1] == 8 and variants[0] >= 1, variants
    # 16-row weight groups per wave: GateUp its gate subtiles, then the matching up ones; Down consecutive rows.
    groups = 2 * GATEUP_TILE_N // (16 * WAVES) if gateup else DOWN_TILE_N // (16 * WAVES)

    def schedule(light):
        """(MFMA order of a slice as (K half, K quarter, group), retire_at, partial sets, rolling A reads).

        retire_at[p] lists the (slice delta, group, element) of partials retired after MFMA p of a slice. Full: K step
        outer, group inner, the previous slice's element p under MFMA p (two partial sets). Light: K half, group pair,
        K quarter, group of the pair; element e of a chain retires 2 + e MFMAs after its last step (XDL latency), before
        the group's next chain starts (one partial set), and half 0 of the next slice's A is read once this slice's
        half 0 is done.
        """
        if not light:
            order = [(step // 2, step % 2, g) for step in range(4) for g in range(groups)]
            return order, {p: [(1, p // 4, p % 4)] for p in range(len(order))}, 2, False
        order = [(j, h, 2 * pair + member) for j in range(2) for pair in range(groups // 2) for h in range(2) for member in range(2)]
        retire_at = {}
        for g in range(groups):
            last = max(p for p, o in enumerate(order) if o[2] == g)
            for e in range(4):
                w = last + 2 + e
                retire_at.setdefault(w % len(order), []).append((w // len(order), g, e))
        first = {g: min(p for p, o in enumerate(order) if o[2] == g) for g in range(groups)}
        assert all(p < first[g] for p, items in retire_at.items() for d, g, _ in items if d == 1)
        return order, retire_at, 1, True
    if gateup:
        I = n_dim
        n_tiles = I // GATEUP_TILE_N
        weight_rows = 2 * I
    else:
        H = n_dim
        n_tiles = H // DOWN_TILE_N
        weight_rows = H
    scale_groups = weight_rows // 128

    @fx.struct
    class SharedTiles:
        # One A tile (16 KiB). After the K loop: GateUp's FP8 output tile or Down's per-wave transposes.
        a: fx.Array[fx.Int32, A_TILE_DWORDS, 16]
        # The A scales of the tile's K block, one per row; after the K loop GateUp's per-wave row maxima.
        sa: fx.Array[fx.Float32, SMALL_M_ROWS * (WAVES if gateup else 1), 16]

    @flyc.kernel(known_block_size=[64 * WAVES, 1, 1])
    def small_m_kernel(
        pA: fx.Pointer,
        pB: fx.Pointer,
        pOut: fx.Pointer,
        pScaleOut: fx.Pointer,
        pScaleA: fx.Pointer,
        pScaleB: fx.Pointer,
        pSortedIds: fx.Pointer,
        pSortedWeights: fx.Pointer,
        pExpertIds: fx.Pointer,
        pNumValid: fx.Pointer,
        num_tokens: fx.Int32,
        num_blocks: fx.Int32,
    ):
        tid = fx.thread_idx.x
        lane = tid % 64
        wave = fx.Int32(rocdl.readfirstlane(T.i32, (tid // 64).ir_value()))
        pid = fx.Int32(fx.block_idx.x)
        zero = fx.Int32(0).ir_value()

        def load_i32(rsrc, voffset):
            return fx.Int32(rocdl.raw_ptr_buffer_load(T.i32, rsrc, fx.Int32(voffset).ir_value(), zero, zero))

        def uniform(value):
            return fx.Int32(rocdl.readfirstlane(T.i32, value.ir_value()))

        def decode(fused):
            """(token, slot, valid) of a sorted ID."""
            token = fused & TOKEN_MASK
            slot = (fused >> 24) & 255
            return token, slot, (token < num_tokens) & (slot < topk)

        valid_rows = uniform(load_i32(_rsrc_of(pNumValid, 4), 0))
        valid_blocks = (valid_rows + (SORT_BLOCK_M - 1)) // SORT_BLOCK_M
        valid_blocks = (valid_blocks < num_blocks).select(valid_blocks, num_blocks)
        num_tasks = valid_blocks * n_tiles
        # Hardware deals workgroups round-robin over 8 XCDs: each XCD gets a contiguous task range, so the N tiles
        # of a block (the same A rows) share one L2.
        chunk = (num_tasks + 7) >> 3
        xcd_slot = pid >> 3
        task = (pid & 7) * chunk + xcd_slot
        block = task // n_tiles
        sorted_rsrc = _rsrc_of(pSortedIds, num_blocks * (SORT_BLOCK_M * 4))
        # A route at row 128 sends the block to the BM256 kernels.
        probe = uniform(load_i32(sorted_rsrc, (block * SORT_BLOCK_M + SMALL_M_ROWS) * 4))
        small = ((probe & TOKEN_MASK) >= num_tokens) | (((probe >> 24) & 255) >= topk)
        active = (xcd_slot < chunk) & (task < num_tasks) & small

        if active:
            n_tile = task - block * n_tiles
            row_base = block * SORT_BLOCK_M
            capacity = num_blocks * SORT_BLOCK_M
            expert = uniform(load_i32(_rsrc_of(pExpertIds, num_blocks * 4), block * 4))

            # Rows the block uses (highest valid row + 1, at most 128), identical in every wave.
            low_valid = decode(load_i32(sorted_rsrc, (row_base + lane) * 4))[2]
            high_valid = decode(load_i32(sorted_rsrc, (row_base + 64 + lane) * 4))[2]
            low_mask = fx.Int64(rocdl.ballot(fx.Int64.ir_type, low_valid.ir_value()))
            high_mask = fx.Int64(rocdl.ballot(fx.Int64.ir_type, high_valid.ir_value()))
            used_rows = (high_mask != fx.Int64(0)).select(
                fx.Int64(128) - fx.math.ctlz(high_mask), fx.Int64(64) - fx.math.ctlz(low_mask)
            )

            # The B scales of every K block of the task in one VGPR, read with v_readlane.
            if const_expr(gateup):
                scale_side = lane // KB
                scale_offset = ((expert * scale_groups + n_tile + scale_side * (I // 128)) * KB + lane - scale_side * KB) * 4
                scale_valid = lane < 2 * KB
            else:
                # A wave's 64 rows lie in one 128-row scale group; the workgroup's 256 rows span two.
                scale_group = (n_tile * DOWN_TILE_N + wave * (16 * groups)) // 128
                scale_offset = ((expert * scale_groups + scale_group) * KB + lane) * 4
                scale_valid = lane < KB
            sb_lanes = fx.Float32(
                rocdl.raw_ptr_buffer_load(
                    T.f32,
                    _rsrc_of(pScaleB, num_experts * scale_groups * (KB * 4)),
                    scale_valid.select(scale_offset, fx.Int32(OOB_OFFSET)).ir_value(),
                    zero,
                    zero,
                )
            )

            def scale_b(side, k_block):
                index = k_block + side * KB if const_expr(gateup) else k_block
                return fx.Float32(rocdl.readlane(T.f32, sb_lanes.ir_value(), fx.Int32(index).ir_value()))

            b_tensor = fx.rocdl.make_buffer_tensor(
                fx.make_view(pB + fx.Int64(expert) * (weight_rows * k_dim // 4), fx.make_layout(1, 1)),
                max_size=False,
                num_records_bytes=weight_rows * k_dim,
            )
            b_packet = BufferTensor(fx.make_view(fx.get_iter(b_tensor), fx.make_layout(4, 1)))
            if const_expr(gateup):
                half = groups // 2
                b_first_rows = [n_tile * GATEUP_TILE_N + wave * (16 * half) + g % half * 16 + g // half * I for g in range(groups)]
            else:
                b_first_rows = [n_tile * DOWN_TILE_N + wave * (16 * groups) + g * 16 for g in range(groups)]
            # Lane l, load j of a 16-row group's K block: weight row l % 16, K 64 j + 16 (l // 16) + [0, 16). The
            # group's (wave-uniform) row offset goes into soffset.
            b_group_offsets = [b_first_rows[g] * k_dim for g in range(groups)]
            b_voffset = lane * 16
            # The A reads take the same K: row lane % 16 of a slice, 16 B column 4 j + lane // 16.
            a_read = [_lds_dwords(lane % 16, j * 4 + lane // 16) for j in range(2)]

            if const_expr(gateup):
                a_rows_total = num_tokens
            else:
                a_rows_total = capacity
            a_tensor = fx.rocdl.make_buffer_tensor(
                fx.make_view(pA, fx.make_layout(1, 1)), max_size=False, num_records_bytes=a_rows_total * k_dim
            )
            a_packet = BufferTensor(fx.make_view(fx.get_iter(a_tensor), fx.make_layout(4, 1)))
            # GateUp: a_scale_t [K/128, tokens]; Down: GateUp's [K/128, capacity].
            sa_rsrc = _rsrc_of(pScaleA, a_rows_total * (KB * 4))
            lds = fx.SharedAllocator().allocate(SharedTiles).peek()

            def lds_view(dwords, count=4):
                return fx.make_view(lds.a.ptr + dwords, fx.make_layout(count, 1))

            def sa_view(index, count):
                return fx.make_view(lds.sa.ptr + index, fx.make_layout(count, 1))

            def barrier():
                rocdl.sched_barrier(0)
                rocdl.s_barrier()
                rocdl.sched_barrier(0)

            def silu_mul(gate_values, up_values):
                result = []
                for element in range_constexpr(4):
                    gate = gate_values[element]
                    exponent = fx.Float32(rocdl.exp2(T.f32, (gate * -LOG2E).ir_value()))
                    inverse = fx.Float32(rocdl.rcp(T.f32, (exponent + 1.0).ir_value()))
                    result.append(gate * inverse * up_values[element])
                return result

            def opaque(value, tag):
                # A distinct tag per use: SimplifyCFG hoists identical leading instructions of both dispatch arms.
                return fx.Int32(llvm.inline_asm(T.i32, [value.ir_value()], f"v_mov_b32 $0, $1 ; {tag}", "=v,v", has_side_effects=True))

            def run(S):
                """K loop and epilogue for S 16-row slices (S * 16 >= the rows this block uses)."""
                copies = (S + 1) // 2
                # Accumulators of 7 or more slices (112 or 128 VGPRs) fit only the light layout (one partial set, A of
                # the next slice read half by half, one A staging slot); fewer slices take the full one.
                light = S >= LIGHT_SLICES[mode]
                mfma_order, retire_at, pbufs, rolling = schedule(light)
                per_slice = len(mfma_order)
                first_mfma = {g: min(p for p, o in enumerate(mfma_order) if o[2] == g) for g in range(groups)}
                stages = 1 if light else 2
                # Indices from a per-variant opaque tid: LLVM would hoist prologue loads that variants share above the
                # dispatch, and the structurized branches would keep them alive through every variant's K loop.
                vtid = opaque(tid, f"slices {S}")
                copy_rows = [vtid // 8 + 32 * c for c in range_constexpr(copies)]
                a_offsets = []
                for c in range_constexpr(copies):
                    token, _, valid = decode(load_i32(sorted_rsrc, (row_base + copy_rows[c]) * 4))
                    a_row = token if const_expr(gateup) else row_base + copy_rows[c]
                    a_offsets.append(valid.select(a_row * k_dim + vtid % 8 * 16, fx.Int32(OOB_OFFSET)))
                # One A scale per thread and K block, shared through LDS: row tid % 128 (threads t and t + 128 load
                # the same one). Padding rows load zeros.
                sa_row = vtid % SMALL_M_ROWS
                sa_token, _, sa_valid = decode(load_i32(sorted_rsrc, (row_base + sa_row) * 4))
                sa_offset = sa_valid.select((sa_token if gateup else row_base + sa_row) * 4, fx.Int32(OOB_OFFSET))
                a_write = [_lds_dwords(copy_rows[c], vtid % 8) for c in range_constexpr(copies)]

                # B ring slot (buf * groups + g) * 2 + j; the A rows and scale of K block k stage in slot k % stages.
                b_ring = fx.make_rmem_tensor(fx.make_layout((4, ring * groups * 2), (1, 4)), fx.Int32)
                a_stage = fx.make_rmem_tensor(fx.make_layout((4, stages * copies), (1, 4)), fx.Int32)
                sa_stage = fx.make_rmem_tensor(fx.make_layout((1, stages), (1, 1)), fx.Float32)
                accum = fx.make_rmem_tensor(fx.make_layout((4, groups * S), (1, 4)), fx.Float32)
                # Partial parity * groups + g: a K block's MFMA chain of a slice and weight group.
                partial = fx.make_rmem_tensor(fx.make_layout((4, pbufs * groups), (1, 4)), fx.Float32)
                partial_scale = fx.make_rmem_tensor(fx.make_layout((1, pbufs * groups), (1, 1)), fx.Float32)

                def block_soffset(k_block, stride):
                    # Past the last K block the load points out of every descriptor and moves no data.
                    return (k_block < KB).select(k_block * stride, fx.Int32(OOB_OFFSET))

                def issue_b(k_block, buf):
                    soffset = block_soffset(k_block, B_BLOCK_BYTES)
                    for g in range_constexpr(groups):
                        for j in range_constexpr(2):
                            b_ring[None, (buf * groups + g) * 2 + j].store(
                                b_packet.load(
                                    voffset_bytes=b_voffset + j * 1024, soffset_bytes=b_group_offsets[g] + soffset, aux=b_aux
                                )
                            )

                def issue_a(k_block, slot):
                    soffset = block_soffset(k_block, TILE_K)
                    for c in range_constexpr(copies):
                        a_stage[None, slot * copies + c].store(a_packet.load(voffset_bytes=a_offsets[c], soffset_bytes=soffset))
                    soffset = block_soffset(k_block, (num_tokens if gateup else capacity) * 4)
                    value = rocdl.raw_ptr_buffer_load(T.f32, sa_rsrc, fx.Int32(sa_offset).ir_value(), fx.Int32(soffset).ir_value(), zero)
                    sa_stage[None, slot].store(Vec.from_elements([fx.Float32(value)], fx.Float32))

                def write_a(slot):
                    """The A rows and scale staged in slot into the LDS tile."""
                    for c in range_constexpr(copies):
                        lds_view(a_write[c]).store(a_stage[None, slot * copies + c].load())
                    sa_view(sa_row, 1).store(sa_stage[None, slot].load())

                def retire(slice_index, parity, g, element):
                    """acc += partial * scale for one element of a finished chain."""
                    index = (parity if pbufs == 2 else 0) * groups + g
                    target = accum[None, slice_index * groups + g]
                    target[element] = fx.fma(
                        Vec(partial[None, index].load())[element], Vec(partial_scale[None, index].load())[0], target[element]
                    )

                def read_a(s, j):
                    """A words of 64-K half j of slice s."""
                    return Vec(lds_view(a_read[j] + s * 16 * ROW_DWORDS).load()).bitcast(fx.Int64)

                def read_sa(s):
                    return Vec(sa_view(s * 16 + lane % 16, 1).load())[0]

                def compute(k_block, buf, k_local):
                    """LDS reads, MFMAs and dequant of K block k_block (retire_at: when each partial element retires)."""
                    sb = [scale_b(side, k_block) for side in range_constexpr(2 if gateup else 1)]
                    b_words = [
                        [Vec(b_ring[None, (buf * groups + g) * 2 + j].load()).bitcast(fx.Int64) for j in range_constexpr(2)]
                        for g in range_constexpr(groups)
                    ]
                    a_words = [read_a(0, j) for j in range_constexpr(2)]
                    sa = read_sa(0)
                    for s in range_constexpr(S):
                        parity = (k_local * S + s) % 2
                        more = s + 1 < S
                        a_next = [None, None]
                        sa_next = None
                        if const_expr(more and not rolling):
                            # Slice s + 1 is read before the MFMAs of slice s, which hide its LDS latency.
                            a_next = [read_a(s + 1, j) for j in range_constexpr(2)]
                            sa_next = read_sa(s + 1)
                        for p, (j, h, g) in enumerate(mfma_order):
                            if const_expr(more and rolling and p == per_slice // 2):
                                # Half 0 of this slice is done: its registers take half 0 of the next.
                                a_next[0] = read_a(s + 1, 0)
                                sa_next = read_sa(s + 1)
                            index = (parity if pbufs == 2 else 0) * groups + g
                            first = p == first_mfma[g]
                            accumulator = Vec.filled(4, 0.0, fx.Float32) if first else Vec(partial[None, index].load())
                            partial[None, index].store(
                                Vec(rocdl.mfma_f32_16x16x32_fp8_fp8(T.f32x4, [b_words[g][j][h], a_words[j][h], accumulator, 0, 0, 0]))
                            )
                            # Keep MFMA -> FMA order: the retired partials have MFMAs of XDL latency cover.
                            rocdl.sched_barrier(0)
                            for delta, rg, element in retire_at.get(p, ()):
                                # delta 1: the slice before (the previous K block's last for s == 0).
                                retire(s - delta if s >= delta else S - 1, parity ^ delta, rg, element)
                            if const_expr(first):
                                side = g // (groups // 2) if gateup else 0
                                partial_scale[None, index].store(Vec.from_elements([sa * sb[side]], fx.Float32))
                            rocdl.sched_barrier(0)
                        if const_expr(more):
                            if const_expr(rolling):
                                a_next[1] = read_a(s + 1, 1)
                            a_words, sa = a_next, sa_next

                # Prologue in the K loop's VMEM order (A of block k + 1 between the B of blocks k and k + 1): a wait in
                # the loop holds for its first iteration and its back edge alike, and with another order the waits for
                # B or A would cover loads issued after them.
                issue_a(fx.Int32(0), 0)
                rocdl.sched_barrier(0)
                issue_b(fx.Int32(0), 0)
                rocdl.sched_barrier(0)
                if const_expr(stages == 2):
                    issue_a(fx.Int32(1), 1)
                    rocdl.sched_barrier(0)
                accum.fill(0)
                # The first slice retires these zeros as the "previous" one.
                for p in range_constexpr(pbufs * groups):
                    partial[None, p].store(Vec.filled(4, 0.0, fx.Float32))
                    partial_scale[None, p].store(Vec.filled(1, 0.0, fx.Float32))
                write_a(0)
                rocdl.s_waitcnt(lgkmcnt=0)
                barrier()
                if const_expr(stages == 1):
                    issue_a(fx.Int32(1), 0)
                    rocdl.sched_barrier(0)

                # An opaque trip count keeps the K loop a loop: with one iteration (K = 256) LLVM flattens it and the
                # straight-line code spills, and these launches have no scratch.
                k_loops = fx.Int32(
                    llvm.inline_asm(T.i32, [fx.Int32(KB // ring).ir_value()], f"s_mov_b32 $0, $1 ; k loops {S}", "=s,s", has_side_effects=True)
                )
                for k_outer in range(0, k_loops):
                    for k_local in range_constexpr(ring):
                        k_block = fx.Int32(k_outer) * ring + k_local
                        # Compute phase. B of the next K block goes first: no wait.
                        issue_b(k_block + 1, (k_local + 1) % ring)
                        compute(k_block, k_local, k_local)
                        # Memory phase, VMEM and LDS only: the loads of A of k + 2 (with two slots before the barriers,
                        # with one after the tile write that frees it), and once every wave is done with the tile, A of
                        # k + 1 into it. The tile's LDS reads must have returned before another wave overwrites it:
                        # instruction selection may move the MFMAs that consume them past the barriers.
                        if const_expr(stages == 2):
                            issue_a(k_block + 2, k_local)
                        rocdl.s_waitcnt(lgkmcnt=0)
                        barrier()
                        write_a((k_local + 1) % stages)
                        rocdl.s_waitcnt(lgkmcnt=0)
                        barrier()
                        if const_expr(stages == 1):
                            issue_a(k_block + 2, 0)
                            rocdl.sched_barrier(0)

                # The chains of the last slice that retire under the next one.
                last_parity = ((ring - 1) * S + S - 1) % 2
                for p in sorted(retire_at):
                    for delta, rg, element in retire_at[p]:
                        if const_expr(delta == 1):
                            retire(S - 1, last_parity, rg, element)

                if const_expr(gateup):
                    # From an opaque copy of tid: indices equal to the K loop's would otherwise live across it.
                    gateup_epilogue(S, accum, opaque(tid, f"epilogue {S}"))
                else:
                    down_epilogue(S, accum, opaque(tid, f"epilogue {S}"))

            def gateup_epilogue(S, accum, tid):
                """FP8 per (row, 128 columns) of SiLU(gate) * up; every dwordx4 store writes 16 B of a 128 B row."""
                lane = tid % 64
                half = groups // 2
                values = [
                    [
                        silu_mul(Vec(accum[None, s * groups + sub].load()), Vec(accum[None, s * groups + half + sub].load()))
                        for sub in range_constexpr(half)
                    ]
                    for s in range_constexpr(S)
                ]
                for s in range_constexpr(S):
                    peak = fx.Float32(0.0)
                    for sub in range_constexpr(half):
                        for value in values[s][sub]:
                            peak = fx.arith.maxnumf(peak, fx.math.absf(value))
                    # Lanes l, l ^ 16, l ^ 32, l ^ 48 hold this wave's other columns of row l % 16.
                    for shift in (16, 32):
                        other = rocdl.ds_bpermute(T.i32, fx.Int32((lane ^ shift) * 4).ir_value(), peak.bitcast(fx.Int32).ir_value())
                        peak = fx.arith.maxnumf(peak, fx.Int32(other).bitcast(fx.Float32))
                    if lane < 16:
                        sa_view((s * 16 + lane) * WAVES + wave, 1).store(Vec.from_elements([peak], fx.Float32))
                rocdl.s_waitcnt(lgkmcnt=0)
                barrier()

                def row_amax(row):
                    maxima = Vec(sa_view(row * WAVES, WAVES).load())
                    peak = maxima[0]
                    for index in range_constexpr(1, WAVES):
                        peak = fx.arith.maxnumf(peak, maxima[index])
                    return peak

                for s in range_constexpr(S):
                    row = s * 16 + lane % 16
                    amax = row_amax(row)
                    # |x| <= amax keeps x * 240 / amax within the FP8 range after rounding.
                    inverse = (amax > 0.0).select(fx.Float32(rocdl.rcp(T.f32, amax.ir_value())) * FP8_MAX, fx.Float32(0.0))
                    for sub in range_constexpr(half):
                        q = [values[s][sub][element] * inverse for element in range_constexpr(4)]
                        low = rocdl.cvt_pk_fp8_f32(T.i32, q[0], q[1], fx.Int32(0), False)
                        word = fx.Int32(rocdl.cvt_pk_fp8_f32(T.i32, q[2], q[3], low, True))
                        # Subtile `sub` of this wave is 16 B chunk wave * half + sub of the row; lane group g holds
                        # its columns 4 g .. 4 g + 3.
                        chunk = wave * half + sub
                        lds_view(row * FP8_ROW_DWORDS + chunk * 4 + lane // 16, 1).store(Vec.from_elements([word], fx.Int32))
                rocdl.s_waitcnt(lgkmcnt=0)
                barrier()
                out_rsrc = _rsrc_of(pOut, capacity * I)
                row_chunks = FP8_ROW_DWORDS // 4
                rows_per_store = 64 * WAVES // row_chunks
                for part in range_constexpr((S * 16 + rows_per_store - 1) // rows_per_store):
                    out_row = part * rows_per_store + tid // row_chunks
                    offset = (row_base + out_row) * I + n_tile * GATEUP_TILE_N + tid % row_chunks * 16
                    if const_expr(S * 16 % rows_per_store != 0):
                        offset = (out_row < S * 16).select(offset, fx.Int32(OOB_OFFSET))
                    rocdl.raw_ptr_buffer_store(
                        Vec(lds_view(out_row * FP8_ROW_DWORDS + tid % row_chunks * 4).load()).ir_value(),
                        out_rsrc,
                        offset.ir_value(),
                        zero,
                        aux=ir.IntegerAttr.get(T.i32, 0),
                    )
                scale_row = tid % SMALL_M_ROWS
                scale_offset = (n_tile * capacity + row_base + scale_row) * 4
                rocdl.raw_ptr_buffer_store(
                    (row_amax(scale_row) * (1.0 / FP8_MAX)).ir_value(),
                    _rsrc_of(pScaleOut, (I // A2_SCALE_COLUMNS) * capacity * 4),
                    (tid < S * 16).select(scale_offset, fx.Int32(OOB_OFFSET)).ir_value(),
                    zero,
                    aux=ir.IntegerAttr.get(T.i32, 0),
                )

            def down_epilogue(S, accum, tid):
                """routing weight * row in BF16, through a per-wave LDS transpose per group pair: 4 rows x 64 B per atomic."""
                lane = tid % 64
                lane_rows = [s * 16 + lane % 16 for s in range_constexpr(S)]
                weight_rsrc = _rsrc_of(pSortedWeights, capacity * 4)
                weights = [
                    fx.Float32(rocdl.raw_ptr_buffer_load(T.f32, weight_rsrc, ((row_base + row) * 4).ir_value(), zero, zero))
                    for row in lane_rows
                ]
                token_offsets = []
                for row in lane_rows:
                    token, _, valid = decode(load_i32(sorted_rsrc, (row_base + row) * 4))
                    token_offsets.append(valid.select(token * (H * 2), fx.Int32(-1)))
                out_base = fx.Int64(fx.ptrtoint(pOut)).ir_value()
                transpose = wave * (16 * TRANSPOSE_PITCH)
                write_base = transpose + lane % 16 * TRANSPOSE_PITCH + lane // 16 * 2
                read_base = transpose + lane // 16 * TRANSPOSE_PITCH + lane % 16
                column = n_tile * (DOWN_TILE_N * 2) + wave * (32 * groups) + lane % 16 * 4
                # LLVM does not count the asm stores: retire every load first so no later vmcnt wait covers them.
                rocdl.s_waitcnt(vmcnt=0)
                for s in range_constexpr(S):
                    # Store i covers rows 4 i .. 4 i + 3 of the slice (lane // 16), 64 B each.
                    row_offsets = [
                        fx.Int32(rocdl.ds_bpermute(T.i32, ((4 * i + lane // 16) * 4).ir_value(), token_offsets[s].ir_value()))
                        for i in range_constexpr(4)
                    ]
                    for pair in range_constexpr(groups // 2):
                        for side in range_constexpr(2):
                            values = Vec(accum[None, s * groups + pair * 2 + side].load())
                            words = pack_bf16x2_rne([values[element] * weights[s] for element in range_constexpr(4)])
                            lds_view(write_base + side * 8, 2).store(Vec.from_elements(words, fx.Int32))
                        reads = [Vec(lds_view(read_base + 4 * i * TRANSPOSE_PITCH, 1).load())[0].ir_value() for i in range_constexpr(4)]
                        # One opaque copy of the 4 LDS reads: one wait, and LLVM cannot sink a read into a store branch.
                        copies = llvm.inline_asm(
                            ir.Type.parse("!llvm.struct<(" + ", ".join(["i32"] * 4) + ")>"),
                            reads,
                            "",
                            ",".join(["=v"] * 4 + [str(i) for i in range(4)]),
                            has_side_effects=True,
                        )
                        for i in range_constexpr(4):
                            value = fx.Int32(llvm.extractvalue(T.i32, copies, [i])).ir_value()
                            row_offset = row_offsets[i]
                            if row_offset >= 0:
                                llvm.inline_asm(
                                    None, [(row_offset + column + pair * 64).ir_value(), value, out_base],
                                    "global_atomic_pk_add_bf16 $0, $1, $2", "v,v,s", has_side_effects=True,
                                )

            def dispatch(index):
                if const_expr(index == len(variants) - 1):
                    run(variants[index])
                else:
                    if used_rows <= variants[index] * 16:
                        run(variants[index])
                    else:
                        dispatch(index + 1)

            dispatch(0)

    @flyc.jit
    def launch(
        pA: fx.Pointer,
        pB: fx.Pointer,
        pOut: fx.Pointer,
        pScaleOut: fx.Pointer,
        pScaleA: fx.Pointer,
        pScaleB: fx.Pointer,
        pSortedIds: fx.Pointer,
        pSortedWeights: fx.Pointer,
        pExpertIds: fx.Pointer,
        pNumValid: fx.Pointer,
        num_tokens: fx.Int32,
        num_blocks: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        small_m_kernel(
            pA, pB, pOut, pScaleOut, pScaleA, pScaleB, pSortedIds, pSortedWeights, pExpertIds, pNumValid, num_tokens, num_blocks,
            # Two waves per SIMD: two workgroups per CU.
            value_attrs={"llvm.passthrough": [["target-features", "-packed-fp32-ops"], ["amdgpu-waves-per-eu", "2,2"]]},
        ).launch(
            grid=((num_blocks * n_tiles + 7) // 8 * 8, 1, 1),
            block=(64 * WAVES, 1, 1),
            stream=stream,
        )

    launch.compile_hints["target_features"] = "-packed-fp32-ops"
    return launch


def _ptr(dtype, tensor):
    return flyc.from_c_void_p(dtype, tensor.data_ptr())


def moe_gateup_small_m(
    a, w1, a_scale_t, w1_scale, sorted_ids, expert_ids, num_valid_ids, a2, a2_scale, topk, ring=2, b_aux=2
):
    """a [T, H] FP8, w1 shuffle_weight(16,16) of [E, 2I, H] FP8, a_scale_t [H/128, T], w1_scale [E, 2I/128, H/128] FP32;
    routes as from aiter moe_sorting(256): sorted_ids [blocks * 256], expert_ids [blocks], num_valid_ids [>= 1].
    For every block with no route at row 128, writes its rows of a2 [blocks * 256, I] FP8 and their columns of
    a2_scale [I/64, blocks * 256] FP32 (see the module doc); other blocks are left to the BM256 GateUp."""
    tokens, hidden_dim = a.shape
    num_experts, two_inter, _ = w1.shape
    inter_dim = two_inter // 2
    num_blocks = expert_ids.numel()
    capacity = num_blocks * SORT_BLOCK_M
    assert a2.shape == (capacity, inter_dim) and a2.element_size() == 1
    assert a2_scale.shape == (inter_dim // A2_SCALE_COLUMNS, capacity) and a2_scale.dtype == torch.float32
    assert sorted_ids.numel() >= capacity
    validate_parameters("gateup", tokens, inter_dim, hidden_dim, topk, num_experts, capacity)
    launch = compile_moe_small_m("gateup", inter_dim, hidden_dim, topk, num_experts, ring, b_aux)
    run_compiled(
        launch,
        _ptr(fx.Int32, a), _ptr(fx.Int32, w1), _ptr(fx.Int32, a2), _ptr(fx.Float32, a2_scale), _ptr(fx.Float32, a_scale_t),
        _ptr(fx.Float32, w1_scale), _ptr(fx.Int32, sorted_ids), _ptr(fx.Float32, a2_scale), _ptr(fx.Int32, expert_ids),
        _ptr(fx.Int32, num_valid_ids), tokens, num_blocks, torch.cuda.current_stream(),
    )
    return a2


def moe_down_small_m(
    a2, w2, a2_scale, w2_scale, sorted_ids, sorted_weights, expert_ids, num_valid_ids, out, topk, ring=2, b_aux=2
):
    """out [T, H] BF16 += routing weight * (a2 row @ w2[expert]^T) for the routes of every block with no route at
    row 128; a2 / a2_scale from moe_gateup_small_m, w2 shuffle_weight(16,16) of [E, H, I] FP8, w2_scale
    [E, H/128, I/128], sorted_weights [blocks * 256]. BF16 atomics: out must hold zeros (or partial sums)."""
    tokens, hidden_dim = out.shape
    num_experts, _, inter_dim = w2.shape
    num_blocks = expert_ids.numel()
    capacity = num_blocks * SORT_BLOCK_M
    assert a2.shape[0] >= capacity and a2.shape[1] == inter_dim
    assert a2_scale.shape == (inter_dim // A2_SCALE_COLUMNS, capacity) and a2_scale.dtype == torch.float32
    assert sorted_ids.numel() >= capacity and sorted_weights.numel() >= capacity
    validate_parameters("down", tokens, hidden_dim, inter_dim, topk, num_experts, capacity)
    launch = compile_moe_small_m("down", hidden_dim, inter_dim, topk, num_experts, ring, b_aux)
    run_compiled(
        launch,
        _ptr(fx.Int32, a2), _ptr(fx.Int32, w2), _ptr(fx.Int32, out), _ptr(fx.Float32, a2_scale), _ptr(fx.Float32, a2_scale),
        _ptr(fx.Float32, w2_scale), _ptr(fx.Int32, sorted_ids), _ptr(fx.Float32, sorted_weights), _ptr(fx.Int32, expert_ids),
        _ptr(fx.Int32, num_valid_ids), tokens, num_blocks, torch.cuda.current_stream(),
    )
    return out
