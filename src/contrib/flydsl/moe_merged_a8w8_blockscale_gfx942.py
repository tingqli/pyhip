# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx942 FP8 block-scale MoE GateUp / Down with one launch per stage over a 256-row sort table (docs/record/
moe_small_m_gfx942.md §8, option A): the BM256 kernels in their merged mode run every sorted block, a block with a
route at row 128 as BM256 tasks (256 rows x one N tile) and the others as pair tasks.

- Pair task (emit_pair): one 8-wave workgroup takes a block with at most 128 routes and two adjacent N tiles; half h
  (waves 4 h .. 4 h + 3) computes N tile 2 p + h. Every wave owns 4 x 16 weight rows and streams them from global
  memory straight into VGPRs (shuffle_weight(16,16) is MFMA-ready); the A tile and its scales go through LDS, loaded
  once for both halves by all 512 threads; all 8 waves share the barriers. The K loop exists for 1 to 8 sixteen-row
  slices and a block takes the smallest that covers its routes (the light register layout for 7 or 8). GateUp writes
  SiLU(gate) * up as FP8 per (row, 128 columns), Down adds with BF16 atomics. Design history: the record, §6-§8.
- Task list (merged_plugin): every workgroup scans the row-128 IDs of the valid blocks (ballot + mbcnt prefix) and
  takes a slot of a dense list, per XCD the BM256 tasks first, then the pair tasks; workgroups past the last slot
  skip the scan. The grid follows the sort capacity.
"""

import functools
from types import SimpleNamespace

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from .gemm_a8w8_blockscale_gfx942_common import pack_bf16x2_rne
from .helpers import BufferTensor, run_compiled
from .moe_gateup_a8w8_blockscale_gfx942_lds import (
    FP8_MAX,
    LOG2E,
    OOB_OFFSET,
    PACKET_DWORDS,
    SMALL_M_ROWS,
    SORT_BLOCK_M,
    TILE_K,
    TOKEN_MASK,
    _lds_dwords,
)

WAVES = 8
# Waves per N tile (one half of the workgroup).
TILE_WAVES = 4
GATEUP_TILE_N = 128
DOWN_TILE_N = 256
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
# From this many slices a kernel takes the light register layout (see compile_pair_tasks.schedule).
LIGHT_SLICES = {"gateup": 7, "down": 7}


def validate_parameters(mode, tokens, n_dim, k_dim, topk, num_experts, capacity):
    """mode "gateup": n_dim = inter_dim, k_dim = hidden_dim; "down": n_dim = hidden_dim, k_dim = inter_dim."""
    tile_n = GATEUP_TILE_N if mode == "gateup" else DOWN_TILE_N
    if not 0 < tokens <= TOKEN_MASK:
        raise ValueError(f"tokens must be in [1, {TOKEN_MASK}]")
    if n_dim <= 0 or n_dim % (2 * tile_n):
        raise ValueError(f"{mode}: N must be a positive multiple of {2 * tile_n}")
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
def compile_pair_tasks(mode, n_dim, k_dim, topk, num_experts, ring=2, b_aux=2, slice_variants=SLICE_VARIANTS):
    """The pair task of the GateUp ("gateup") or Down ("down") stage: emit_pair, its LDS regions in dwords (a_region,
    then sa_region) and n_pairs per block; see the module doc.

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
    groups = 2 * GATEUP_TILE_N // (16 * TILE_WAVES) if gateup else DOWN_TILE_N // (16 * TILE_WAVES)

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
    n_pairs = n_tiles // 2
    # LDS of a task in dwords: the A tile (GateUp: then two FP8 output tiles), then sa (Float32).
    a_region = max(A_TILE_DWORDS, 2 * SMALL_M_ROWS * FP8_ROW_DWORDS if gateup else 0)
    sa_region = SMALL_M_ROWS * (WAVES if gateup else 1)

    def emit_pair(ctx, block, pair):
        """Task (block, N tiles 2 pair and 2 pair + 1) of a block with at most 128 routes. ctx: the kernel's pointers
        (pA, pB, pOut, pScaleOut, pScaleA, pScaleB, pSortedIds, pSortedWeights, pExpertIds), num_tokens, num_blocks,
        tid (0..511) and lds (a.ptr Int32 and sa.ptr Float32 regions of a_region and sa_region elements)."""
        pA, pB, pOut, pScaleOut, pScaleA, pScaleB = ctx.pA, ctx.pB, ctx.pOut, ctx.pScaleOut, ctx.pScaleA, ctx.pScaleB
        pSortedIds, pSortedWeights, pExpertIds = ctx.pSortedIds, ctx.pSortedWeights, ctx.pExpertIds
        num_tokens, num_blocks, tid, lds = ctx.num_tokens, ctx.num_blocks, ctx.tid, ctx.lds
        lane = tid % 64
        wave = fx.Int32(rocdl.readfirstlane(T.i32, (tid // 64).ir_value()))
        tile_half = wave // TILE_WAVES
        tile_wave = wave % TILE_WAVES
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

        sorted_rsrc = _rsrc_of(pSortedIds, num_blocks * (SORT_BLOCK_M * 4))
        if const_expr(True):
            n_tile = pair * 2 + tile_half
            row_base = block * SORT_BLOCK_M
            capacity = num_blocks * SORT_BLOCK_M
            expert = uniform(load_i32(_rsrc_of(pExpertIds, num_blocks * 4), block * 4))

            # Rows the task uses (highest valid row + 1, at most 128), identical in every wave.
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
                scale_group = (n_tile * DOWN_TILE_N + tile_wave * (16 * groups)) // 128
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
                b_first_rows = [n_tile * GATEUP_TILE_N + tile_wave * (16 * half) + g % half * 16 + g // half * I for g in range(groups)]
            else:
                b_first_rows = [n_tile * DOWN_TILE_N + tile_wave * (16 * groups) + g * 16 for g in range(groups)]
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
                # All 512 threads copy the A tile: 64 rows of 8 x 16 B per copy.
                copies = (S + 3) // 4
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
                copy_rows = [vtid // 8 + 64 * c for c in range_constexpr(copies)]
                a_offsets = []
                for c in range_constexpr(copies):
                    token, _, valid = decode(load_i32(sorted_rsrc, (row_base + copy_rows[c]) * 4))
                    a_row = token if const_expr(gateup) else row_base + copy_rows[c]
                    a_offsets.append(valid.select(a_row * k_dim + vtid % 8 * 16, fx.Int32(OOB_OFFSET)))
                # One A scale per thread and K block, shared through LDS: row tid % 128 (four threads load each one).
                # Padding rows load zeros.
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
                    maxima = Vec(sa_view(row * WAVES + tile_half * TILE_WAVES, TILE_WAVES).load())
                    peak = maxima[0]
                    for index in range_constexpr(1, TILE_WAVES):
                        peak = fx.arith.maxnumf(peak, maxima[index])
                    return peak

                tile_base = tile_half * (SMALL_M_ROWS * FP8_ROW_DWORDS)
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
                        chunk = tile_wave * half + sub
                        lds_view(tile_base + row * FP8_ROW_DWORDS + chunk * 4 + lane // 16, 1).store(Vec.from_elements([word], fx.Int32))
                rocdl.s_waitcnt(lgkmcnt=0)
                barrier()
                out_rsrc = _rsrc_of(pOut, capacity * I)
                row_chunks = FP8_ROW_DWORDS // 4
                rows_per_store = 64 * TILE_WAVES // row_chunks
                htid = tid % (64 * TILE_WAVES)
                for part in range_constexpr((S * 16 + rows_per_store - 1) // rows_per_store):
                    out_row = part * rows_per_store + htid // row_chunks
                    offset = (row_base + out_row) * I + n_tile * GATEUP_TILE_N + htid % row_chunks * 16
                    if const_expr(S * 16 % rows_per_store != 0):
                        offset = (out_row < S * 16).select(offset, fx.Int32(OOB_OFFSET))
                    rocdl.raw_ptr_buffer_store(
                        Vec(lds_view(tile_base + out_row * FP8_ROW_DWORDS + htid % row_chunks * 4).load()).ir_value(),
                        out_rsrc,
                        offset.ir_value(),
                        zero,
                        aux=ir.IntegerAttr.get(T.i32, 0),
                    )
                scale_row = htid % SMALL_M_ROWS
                scale_offset = (n_tile * capacity + row_base + scale_row) * 4
                rocdl.raw_ptr_buffer_store(
                    (row_amax(scale_row) * (1.0 / FP8_MAX)).ir_value(),
                    _rsrc_of(pScaleOut, (I // A2_SCALE_COLUMNS) * capacity * 4),
                    (htid < S * 16).select(scale_offset, fx.Int32(OOB_OFFSET)).ir_value(),
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
                column = n_tile * (DOWN_TILE_N * 2) + tile_wave * (32 * groups) + lane % 16 * 4
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

    emit_pair = ASTRewriter.transform(emit_pair)
    return SimpleNamespace(emit_pair=emit_pair, a_region=a_region, sa_region=sa_region, n_pairs=n_pairs)


def _ptr(dtype, tensor):
    return flyc.from_c_void_p(dtype, tensor.data_ptr())


SCAN_BLOCKS = 4  # sort blocks per thread of the task scan
MAX_BLOCKS = 64 * WAVES * SCAN_BLOCKS  # sort blocks the scan covers: 2048, T <= 64512 for top-8 over 32 local experts


class MergedPlugin:
    """merged= argument of the BM256 compile functions (hashable by identity: one per shape, see merged_plugin).

    key: a str that changes with the plugin's code. The FlyDSL kernel cache keys on closure scalars and on the
    sources next to the BM256 files only, so without it every plugin of a shape would reuse the first compiled
    kernel (also across processes: importing aiter enables a disk cache).
    """

    def __init__(self, task, body, grid, key):
        self.task, self.body, self.grid, self.key = task, body, grid, key


def source_key(*paths):
    """Hash of this module, the pyhip modules it imports and `paths`."""
    import hashlib
    import inspect
    import sys

    digest = hashlib.sha256()
    modules = (__name__, pack_bf16x2_rne.__module__, BufferTensor.__module__, _lds_dwords.__module__)
    for path in [inspect.getsourcefile(sys.modules[name]) for name in modules] + list(paths):
        with open(path, "rb") as f:
            digest.update(f.read())
    return digest.hexdigest()[:16]


@functools.cache
def merged_plugin(mode, n_dim, k_dim, topk, num_experts):
    small = compile_pair_tasks(mode, n_dim, k_dim, topk, num_experts)
    n_pairs, a_region = small.n_pairs, small.a_region
    # N tiles per block: a BM256 task takes one, a pair task two.
    big_tiles = 2 * n_pairs
    assert a_region + small.sa_region <= 8 * PACKET_DWORDS, "the pair task must fit the BM256 kernel's LDS"

    def scan(ctx):
        """Dense task list, big blocks first: per XCD (pid % 8) a contiguous range of BM256 tasks (rank of the big
        block, N tile), then of pair tasks (rank of the small block, pair). Thread t scans blocks SCAN_BLOCKS * t + i."""
        tid, pid, valid_blocks = ctx.tid, ctx.pid, ctx.valid_blocks
        lane = tid % 64
        wave = fx.Int32(rocdl.readfirstlane(T.i32, (tid // 64).ir_value()))
        zero = fx.Int32(0).ir_value()
        sorted_rsrc = _rsrc_of(ctx.pSortedIds, ctx.num_blocks * (SORT_BLOCK_M * 4))

        def scan_view(index, count=1):
            return fx.make_view(ctx.scan + index, fx.make_layout(count, 1))

        first = tid * SCAN_BLOCKS
        flags = []
        for i in range_constexpr(SCAN_BLOCKS):
            in_valid = first + i < valid_blocks
            # Blocks past num_valid (the sort's spare capacity) read nothing.
            probe_offset = in_valid.select(((first + i) * SORT_BLOCK_M + SMALL_M_ROWS) * 4, fx.Int32(OOB_OFFSET))
            fused = fx.Int32(rocdl.raw_ptr_buffer_load(T.i32, sorted_rsrc, fx.Int32(probe_offset).ir_value(), zero, zero))
            routed = ((fused & TOKEN_MASK) < ctx.num_tokens) & (((fused >> 24) & 255) < topk)
            flags.append(routed & in_valid)
        count = fx.Int32(0)
        for flag in flags:
            count = count + flag.select(fx.Int32(1), fx.Int32(0))
        # Exclusive prefix of count over the wave and its total, one bit plane at a time.
        below = fx.Int32(0)
        wave_total = fx.Int32(0)
        for bit in range_constexpr(SCAN_BLOCKS.bit_length()):
            mask = fx.Int64(rocdl.ballot(fx.Int64.ir_type, (((count >> bit) & 1) != 0).ir_value()))
            lanes_below = rocdl.mbcnt_lo(T.i32, fx.Int32(mask).ir_value(), zero)
            lanes_below = fx.Int32(rocdl.mbcnt_hi(T.i32, fx.Int32(mask >> 32).ir_value(), lanes_below))
            below = below + (lanes_below << bit)
            wave_total = wave_total + (fx.Int32(fx.math.ctpop(mask)) << bit)
        if lane == 0:
            scan_view(wave).store(Vec.from_elements([wave_total], fx.Int32))
        rocdl.s_waitcnt(lgkmcnt=0)
        rocdl.s_barrier()
        totals = Vec(scan_view(0, WAVES).load())
        big_before = fx.Int32(0)
        big_count = fx.Int32(0)
        for w in range_constexpr(WAVES):
            big_before = big_before + (wave > w).select(totals[w], fx.Int32(0))
            big_count = big_count + totals[w]
        big_tasks = big_count * ctx.big_tiles
        small_tasks = (valid_blocks - big_count) * n_pairs
        big_chunk = (big_tasks + 7) >> 3
        small_chunk = (small_tasks + 7) >> 3
        xcd_slot = pid >> 3
        big_task = xcd_slot < big_chunk
        big_index = (pid & 7) * big_chunk + xcd_slot
        small_index = (pid & 7) * small_chunk + (xcd_slot - big_chunk)
        active = big_task.select(big_index < big_tasks, (xcd_slot - big_chunk < small_chunk) & (small_index < small_tasks))
        big_rank = big_index // ctx.big_tiles
        small_rank = small_index // n_pairs
        rank = big_task.select(big_rank, small_rank)
        tile = big_task.select(big_index - big_rank * ctx.big_tiles, small_index - small_rank * n_pairs)
        bigs_before = big_before + below
        for i in range_constexpr(SCAN_BLOCKS):
            mine = big_task.select(flags[i] & (bigs_before == rank), ~flags[i] & (first + i - bigs_before == rank))
            if mine & (first + i < valid_blocks):
                scan_view(WAVES).store(Vec.from_elements([first + i], fx.Int32))
            bigs_before = bigs_before + flags[i].select(fx.Int32(1), fx.Int32(0))
        rocdl.s_waitcnt(lgkmcnt=0)
        rocdl.s_barrier()
        block = fx.Int32(rocdl.readfirstlane(T.i32, Vec(scan_view(WAVES).load())[0].ir_value()))
        # Every wave has read the scan words before a task overwrites this LDS.
        rocdl.s_barrier()
        return active, big_task, block, tile

    scan = ASTRewriter.transform(scan)

    def task(ctx):
        # The used slots per XCD, ceil(big_tiles * big / 8) + ceil(n_pairs * small / 8), never exceed
        # ceil(big_tiles * valid_blocks / 8) + 1: the workgroups past that skip the scan.
        assert ctx.big_tiles == big_tiles, (ctx.big_tiles, big_tiles)
        valid_blocks = ctx.valid_blocks
        in_range = (ctx.pid >> 3) < ((big_tiles * valid_blocks + 7) >> 3) + 1
        active = in_range
        big_task = in_range
        block = fx.Int32(0)
        tile = fx.Int32(0)
        if in_range:
            active, big_task, block, tile = scan(ctx)
        return SimpleNamespace(active=active, big=big_task, block=block, tile=tile)

    def body(ctx, block, pair):
        lds_base = ctx.lds_base
        lds = SimpleNamespace(
            a=SimpleNamespace(ptr=lds_base), sa=SimpleNamespace(ptr=fx.recast_iter(fx.Float32, lds_base + a_region))
        )
        # An opaque tid: the pair task's indices must not merge with the BM256 task's (SimplifyCFG / GVN).
        tid = fx.Int32(llvm.inline_asm(T.i32, [fx.Int32(ctx.tid).ir_value()], "v_mov_b32 $0, $1 ; pair task", "=v,v", has_side_effects=True))
        small.emit_pair(SimpleNamespace(**{**vars(ctx), "tid": tid, "lds": lds}), block, pair)

    def grid(num_blocks):
        # The task bound above for every block of the sort capacity valid, 8 XCDs.
        return 8 * ((big_tiles * num_blocks + 7) // 8 + 1)

    return MergedPlugin(ASTRewriter.transform(task), body, grid, f"pair:{mode}:{source_key()}")


def moe_gateup_merged(a, w1, a_scale_t, w1_scale, sorted_ids, expert_ids, num_valid_ids, a2, a2_scale, topk):
    """GateUp of every block in one launch (BM256 tiles for blocks with a route at row 128, pair tasks for the
    others); a2 [blocks * 256, I] FP8 and a2_scale [I/128, blocks * 256] as the BM256 GateUp writes them (rows a pair
    task does not use are not written)."""
    from .moe_gateup_a8w8_blockscale_gfx942_lds import compile_moe_gateup_a8w8_blockscale_gfx942_lds

    tokens, hidden_dim = a.shape
    num_experts, two_inter, _ = w1.shape
    inter_dim = two_inter // 2
    num_blocks = expert_ids.numel()
    capacity = num_blocks * SORT_BLOCK_M
    assert num_blocks <= MAX_BLOCKS, "too many sort blocks for the task scan"
    assert a2.shape == (capacity, inter_dim) and a2.element_size() == 1
    assert a2_scale.shape == (inter_dim // A2_SCALE_COLUMNS, capacity) and a2_scale.dtype == torch.float32
    validate_parameters("gateup", tokens, inter_dim, hidden_dim, topk, num_experts, capacity)
    launch = compile_moe_gateup_a8w8_blockscale_gfx942_lds(
        inter_dim, hidden_dim, topk, num_experts, merged=merged_plugin("gateup", inter_dim, hidden_dim, topk, num_experts)
    )
    run_compiled(
        launch,
        _ptr(fx.Int32, a), _ptr(fx.Int32, w1), _ptr(fx.Int32, a2), _ptr(fx.Float32, a2_scale), _ptr(fx.Float32, a_scale_t),
        _ptr(fx.Float32, w1_scale), _ptr(fx.Int32, sorted_ids), _ptr(fx.Int32, expert_ids), _ptr(fx.Int32, num_valid_ids),
        tokens, num_blocks, torch.cuda.current_stream(),
    )
    return a2


def moe_down_merged(a2, w2, a2_scale, w2_scale, sorted_ids, sorted_weights, expert_ids, num_valid_ids, out, topk):
    """Down of every block in one launch (BM256 tiles with BF16 atomics for blocks with a route at row 128,
    pair tasks for the others) into a zeroed out [tokens, H]."""
    from .moe_down_a8w8_blockscale_gfx942_lds import compile_moe_down_a8w8_blockscale_gfx942_lds

    tokens, hidden_dim = out.shape
    num_experts, _, inter_dim = w2.shape
    num_blocks = expert_ids.numel()
    capacity = num_blocks * SORT_BLOCK_M
    assert num_blocks <= MAX_BLOCKS, "too many sort blocks for the task scan"
    assert a2.shape[0] >= capacity and a2.shape[1] == inter_dim
    assert a2_scale.shape == (inter_dim // A2_SCALE_COLUMNS, capacity) and a2_scale.dtype == torch.float32
    validate_parameters("down", tokens, hidden_dim, inter_dim, topk, num_experts, capacity)
    launch = compile_moe_down_a8w8_blockscale_gfx942_lds(
        hidden_dim, inter_dim, topk, num_experts, merged=merged_plugin("down", hidden_dim, inter_dim, topk, num_experts)
    )
    run_compiled(
        launch,
        _ptr(fx.Int32, a2), _ptr(fx.Int32, w2), _ptr(fx.Int32, out), _ptr(fx.Float32, a2_scale), _ptr(fx.Float32, w2_scale),
        _ptr(fx.Int32, sorted_ids), _ptr(fx.Float32, sorted_weights), _ptr(fx.Int32, expert_ids),
        _ptr(fx.Int32, num_valid_ids), _ptr(fx.Int32, sorted_ids), tokens, num_blocks, torch.cuda.current_stream(),
    )
    return out
