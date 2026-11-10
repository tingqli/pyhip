# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
# Historical variant, kept only as the A/B reference of docs/experiment/moe_small_m_gfx942/4wave (loaded from
# this file by tools/ab_kernels.py and tools/make_bench_ab.py; not part of the pyhip package).

"""gfx942 FP8 block-scale MoE GateUp and Down on 4-wave workgroups for sort blocks with at most 128 routes.

Routes come from aiter moe_sorting(256), the BM256 kernels' table (moe_gateup/down_a8w8_blockscale_gfx942_lds). A
256-row block with no route at row 128 holds at most 128 routes and runs here; the BM256 kernels run the others
(big_blocks_only=True), so the choice is made per block at run time. With <= 128 routes a BM256 tile is at least
half padding and the work is bound by the weights, read once per block whatever its route count
(docs/record/moe_small_m_gfx942.md).

- A workgroup has 4 waves (one per SIMD) and takes one block and one N tile; every wave owns 2 x 16 weight rows
  (GateUp: 16 gate rows and the matching 16 up rows, 64 output columns per task; Down: 32 output columns, 128
  per task). LDS stays at 32 KiB and VGPRs at <= 256, so two workgroups share a CU: two waves per SIMD.
- Each K block is a compute phase (LDS reads of A, MFMAs, the dequant of the slice before) and a memory phase
  (the next A tile into LDS, the loads of later K blocks, one barrier) without VALU work. Each workgroup syncs only
  its own waves, so the two on a CU drift apart and one's MFMAs cover the other's memory phase.
- B is aiter shuffle_weight(16,16): a 16 B load per lane is MFMA-ready, so each wave streams its weight rows
  straight into VGPRs through a ring of `ring` K blocks, issued at the start of a compute phase without a wait.
  Loads past the last K block point out of the descriptor and move no data.
- A (GateUp: rows gathered by token; Down: the sorted FP8 rows GateUp wrote) is loaded two K blocks ahead into
  VGPRs and written into a double-buffered 16 KiB LDS tile. The A scales of a lane's rows go straight into VGPRs.
- The rows a block uses are found at run time (ballot of its sorted IDs). The K loop and epilogue exist for 1 to
  8 sixteen-row slices and the block takes the smallest that covers its routes.
- Block scales: per K block the chained MFMAs of a (slice, side) give partials retired as
  acc += partial * (scale_a[row] * scale_b) under the next slice's MFMAs.

GateUp writes the Down input: SiLU(gate) * up quantized to FP8 per (row, 64 columns), its N tile, with FP32 scale
amax / 240, at sorted row block * 256 + row of [blocks * 256, I] (scales [blocks * 256, I/64]); Down dequantizes
each 64-column half of a K block with its own scale. Rows past the used slices are not written: Down never reads
padding rows. Down adds routing weight * (row @ W2^T) into out [tokens, H] with BF16 atomics.
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
GATEUP_TILE_N = 64
DOWN_TILE_N = 128
# GateUp's FP8 output scale granularity: one task's N tile.
A2_SCALE_COLUMNS = GATEUP_TILE_N
ROW_DWORDS = TILE_K // 4
A_TILE_DWORDS = SMALL_M_ROWS * ROW_DWORDS
B_BLOCK_BYTES = 16 * TILE_K
# GateUp's FP8 tile in LDS: 64 B per row.
FP8_ROW_DWORDS = GATEUP_TILE_N // 4
# Down epilogue transpose: per wave 16 rows x 16 dwords; an even pitch keeps the 8 B writes aligned.
TRANSPOSE_PITCH = 18
SLICE_VARIANTS = (1, 2, 3, 4, 5, 6, 7, 8)


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

    ring: K blocks of B per wave ring (ring - 1 in flight); b_aux: cache policy bits of the B loads (2 = nt).
    """
    assert mode in ("gateup", "down"), mode
    gateup = mode == "gateup"
    validate_parameters(mode, 1, n_dim, k_dim, topk, num_experts, SORT_BLOCK_M)
    KB = k_dim // TILE_K
    assert ring % 2 == 0 and KB % ring == 0, "the K loop is unrolled by ring, which must divide K / 128"
    variants = sorted(set(slice_variants))
    assert variants[-1] == 8 and variants[0] >= 1, variants
    # Down's A has a scale per 64 columns: each half of a K block is its own MFMA chain and dequant.
    parts = 1 if gateup else 2
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
        # A double buffer, 32 KiB so that two workgroups fit a CU. After the K loop: GateUp's FP8 output tile
        # (buffer 0) and row maxima (buffer 1), or Down's per-wave transposes.
        a: fx.Array[fx.Int32, 2 * A_TILE_DWORDS, 16]

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
                scale_offset = ((expert * scale_groups + n_tile // 2 + scale_side * (I // 128)) * KB + lane - scale_side * KB) * 4
                scale_valid = lane < 2 * KB
            else:
                scale_offset = ((expert * scale_groups + n_tile) * KB + lane) * 4
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
                b_first_rows = [n_tile * GATEUP_TILE_N + wave * 16, n_tile * GATEUP_TILE_N + wave * 16 + I]
            else:
                b_first_rows = [n_tile * DOWN_TILE_N + wave * 32, n_tile * DOWN_TILE_N + wave * 32 + 16]
            # Lane l, load j of a 16-row group's K block: weight row l % 16, K 64 j + 16 (l // 16) + [0, 16).
            b_voffsets = [[b_first_rows[side] * k_dim + j * 1024 + lane * 16 for j in range(2)] for side in range(2)]
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
            # GateUp: a_scale_t [K/128, tokens]; Down: GateUp's [capacity, K/64], the two halves of a K block adjacent.
            sa_tensor = fx.rocdl.make_buffer_tensor(
                fx.make_view(pScaleA, fx.make_layout(1, 1)), max_size=False, num_records_bytes=a_rows_total * (KB * parts * 4)
            )
            sa_packet = BufferTensor(fx.make_view(fx.get_iter(sa_tensor), fx.make_layout(parts, 1)))
            sa_rsrc = fx.rocdl.get_buffer_rsrc(fx.get_iter(sa_tensor))
            lds = fx.SharedAllocator().allocate(SharedTiles).peek()

            def lds_view(dwords, count=4):
                return fx.make_view(lds.a.ptr + dwords, fx.make_layout(count, 1))

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

            def run(S):
                """K loop and epilogue for S 16-row slices (S * 16 >= the rows this block uses)."""
                copies = (S + 1) // 2
                lane_rows = [s * 16 + lane % 16 for s in range_constexpr(S)]
                lane_ids = [load_i32(sorted_rsrc, (row_base + row) * 4) for row in lane_rows]
                copy_rows = [tid // 8 + 32 * c for c in range_constexpr(copies)]
                a_offsets = []
                for c in range_constexpr(copies):
                    token, _, valid = decode(load_i32(sorted_rsrc, (row_base + copy_rows[c]) * 4))
                    a_row = token if const_expr(gateup) else row_base + copy_rows[c]
                    a_offsets.append(valid.select(a_row * k_dim + tid % 8 * 16, fx.Int32(OOB_OFFSET)))
                # The A scales of this lane's row in each slice (padding rows load zeros).
                sa_offsets = []
                for s in range_constexpr(S):
                    token, _, valid = decode(lane_ids[s])
                    sa_row = token if const_expr(gateup) else row_base + lane_rows[s]
                    sa_offsets.append(valid.select(sa_row * (4 if gateup else KB * parts * 4), fx.Int32(OOB_OFFSET)))
                a_write = [_lds_dwords(copy_rows[c], tid % 8) for c in range_constexpr(copies)]

                # B ring slot buf * 4 + side * 2 + j; the A rows and scales of K block k stage in slot k % 2.
                b_ring = fx.make_rmem_tensor(fx.make_layout((4, ring * 4), (1, 4)), fx.Int32)
                a_stage = fx.make_rmem_tensor(fx.make_layout((4, 2 * copies), (1, 4)), fx.Int32)
                sa_stage = fx.make_rmem_tensor(fx.make_layout((parts, 2 * S), (1, parts)), fx.Float32)
                accum = fx.make_rmem_tensor(fx.make_layout((4, 2 * S), (1, 4)), fx.Float32)
                # Partial (pbuf * 2 + side) * parts + part: a K block's (or half's) MFMA chain of a slice.
                partial = fx.make_rmem_tensor(fx.make_layout((4, 4 * parts), (1, 4)), fx.Float32)
                partial_scale = fx.make_rmem_tensor(fx.make_layout((1, 4 * parts), (1, 1)), fx.Float32)

                def block_soffset(k_block, stride):
                    # Past the last K block the load points out of every descriptor and moves no data.
                    return (k_block < KB).select(k_block * stride, fx.Int32(OOB_OFFSET))

                def issue_b(k_block, buf):
                    soffset = block_soffset(k_block, B_BLOCK_BYTES)
                    for side, j in ((0, 0), (0, 1), (1, 0), (1, 1)):
                        b_ring[None, buf * 4 + side * 2 + j].store(
                            b_packet.load(voffset_bytes=b_voffsets[side][j], soffset_bytes=soffset, aux=b_aux)
                        )

                def issue_a(k_block, kbuf):
                    soffset = block_soffset(k_block, TILE_K)
                    for c in range_constexpr(copies):
                        a_stage[None, kbuf * copies + c].store(a_packet.load(voffset_bytes=a_offsets[c], soffset_bytes=soffset))

                def issue_scale_a(k_block, kbuf):
                    soffset = block_soffset(k_block, num_tokens * 4 if gateup else parts * 4)
                    for s in range_constexpr(S):
                        if const_expr(parts == 1):
                            value = rocdl.raw_ptr_buffer_load(
                                T.f32, sa_rsrc, fx.Int32(sa_offsets[s]).ir_value(), fx.Int32(soffset).ir_value(), zero
                            )
                            sa_stage[None, kbuf * S + s].store(Vec.from_elements([fx.Float32(value)], fx.Float32))
                        else:
                            sa_stage[None, kbuf * S + s].store(sa_packet.load(voffset_bytes=sa_offsets[s], soffset_bytes=soffset))

                def write_a(kbuf):
                    for c in range_constexpr(copies):
                        lds_view(a_write[c] + kbuf * A_TILE_DWORDS).store(a_stage[None, kbuf * copies + c].load())

                def retire(slice_index, pbuf, element_index):
                    """acc += partial * scale for element (side, part, element) of a finished slice's partials."""
                    side, rest = divmod(element_index, 4 * parts)
                    part, element = divmod(rest, 4)
                    index = (pbuf * 2 + side) * parts + part
                    target = accum[None, slice_index * 2 + side]
                    target[element] = fx.fma(
                        Vec(partial[None, index].load())[element], Vec(partial_scale[None, index].load())[0], target[element]
                    )

                def read_a(s, kbuf):
                    """A words of slice s (both 64-K halves)."""
                    return [
                        Vec(lds_view(a_read[j] + (s * 16 * ROW_DWORDS + kbuf * A_TILE_DWORDS)).load()).bitcast(fx.Int64)
                        for j in range_constexpr(2)
                    ]

                def compute(k_block, buf, kbuf, k_local):
                    """LDS reads, MFMAs and dequant of K block k_block: the slice before retires under each slice."""
                    sb = [scale_b(side, k_block) for side in range_constexpr(2)]
                    b_words = [
                        [Vec(b_ring[None, buf * 4 + side * 2 + j].load()).bitcast(fx.Int64) for j in range_constexpr(2)]
                        for side in range_constexpr(2)
                    ]
                    # A of slice s + 1 is read before the MFMAs of slice s, which hide its LDS latency.
                    a_next = read_a(0, kbuf)
                    for s in range_constexpr(S):
                        pbuf = (k_local * S + s) % 2
                        # Retired under this slice: the slice before (the previous K block's last for s == 0).
                        previous = s - 1 if s > 0 else S - 1
                        a_words = a_next
                        if const_expr(s + 1 < S):
                            a_next = read_a(s + 1, kbuf)
                        sa = Vec(sa_stage[None, kbuf * S + s].load())
                        for step in range_constexpr(4):
                            j, h = divmod(step, 2)
                            part = j if parts == 2 else 0
                            first = h == 0 if parts == 2 else step == 0
                            for side in range_constexpr(2):
                                index = (pbuf * 2 + side) * parts + part
                                accumulator = Vec.filled(4, 0.0, fx.Float32) if first else Vec(partial[None, index].load())
                                partial[None, index].store(
                                    Vec(rocdl.mfma_f32_16x16x32_fp8_fp8(T.f32x4, [b_words[side][j][h], a_words[j][h], accumulator, 0, 0, 0]))
                                )
                                # Keep MFMA -> FMA order: the retired partial has MFMAs of XDL latency cover.
                                rocdl.sched_barrier(0)
                                for r in range_constexpr(parts):
                                    retire(previous, 1 - pbuf, (step * 2 + side) * parts + r)
                                if const_expr(first):
                                    partial_scale[None, index].store(Vec.from_elements([sa[part] * sb[side]], fx.Float32))
                                rocdl.sched_barrier(0)

                # Prologue: ring - 1 B blocks and the A rows / scales of K blocks 0 and 1, issued in the order the K
                # loop issues them (B of block j + ring - 1, then A of block j + 2, for j < 0). A wait in the loop
                # holds for its first iteration and its back edge alike: with another order, the waits for B or the
                # A scales of a K block would cover the loads of the memory phase just before them.
                for j in range_constexpr(-max(ring - 1, 2), 0):
                    if const_expr(j + ring - 1 >= 0):
                        issue_b(fx.Int32(j + ring - 1), j + ring - 1)
                        rocdl.sched_barrier(0)
                    if const_expr(j + 2 >= 0):
                        issue_a(fx.Int32(j + 2), j + 2)
                        issue_scale_a(fx.Int32(j + 2), j + 2)
                        rocdl.sched_barrier(0)
                accum.fill(0)
                # The first slice retires these zeros as the "previous" one.
                for p in range_constexpr(4 * parts):
                    partial[None, p].store(Vec.filled(4, 0.0, fx.Float32))
                    partial_scale[None, p].store(Vec.filled(1, 0.0, fx.Float32))
                write_a(0)
                rocdl.s_waitcnt(lgkmcnt=0)
                barrier()

                for k_outer in range(0, KB // ring):
                    for k_local in range_constexpr(ring):
                        k_block = fx.Int32(k_outer) * ring + k_local
                        kbuf = k_local % 2
                        # Compute phase. B of a later K block goes first: no wait, and the in-order vmcnt wait for
                        # this K block's B then never covers the A loads of the memory phase after it.
                        issue_b(k_block + (ring - 1), (k_local + ring - 1) % ring)
                        compute(k_block, k_local, kbuf, k_local)
                        # Memory phase, VMEM and LDS only: A of k + 1 into LDS, A rows and scales of k + 2.
                        write_a(1 - kbuf)
                        issue_a(k_block + 2, kbuf)
                        issue_scale_a(k_block + 2, kbuf)
                        rocdl.s_waitcnt(lgkmcnt=0)
                        barrier()

                # The last slice of the last K block has no later MFMA shadow.
                last_pbuf = ((ring - 1) * S + S - 1) % 2
                for element_index in range_constexpr(8 * parts):
                    retire(S - 1, last_pbuf, element_index)

                if const_expr(gateup):
                    gateup_epilogue(S, accum)
                else:
                    down_epilogue(S, accum, lane_rows, lane_ids)

            def gateup_epilogue(S, accum):
                """FP8 per (row, 64 columns) of SiLU(gate) * up; every dwordx4 store writes 16 B of a 64 B row."""
                row_max_base = A_TILE_DWORDS
                values = [
                    silu_mul(Vec(accum[None, s * 2].load()), Vec(accum[None, s * 2 + 1].load())) for s in range_constexpr(S)
                ]
                for s in range_constexpr(S):
                    peak = fx.Float32(0.0)
                    for value in values[s]:
                        peak = fx.arith.maxnumf(peak, fx.math.absf(value))
                    # Lanes l, l ^ 16, l ^ 32, l ^ 48 hold this wave's other columns of row l % 16.
                    for shift in (16, 32):
                        other = rocdl.ds_bpermute(T.i32, fx.Int32((lane ^ shift) * 4).ir_value(), peak.bitcast(fx.Int32).ir_value())
                        peak = fx.arith.maxnumf(peak, fx.Int32(other).bitcast(fx.Float32))
                    if lane < 16:
                        lds_view(row_max_base + (s * 16 + lane) * WAVES + wave, 1).store(Vec.from_elements([peak.bitcast(fx.Int32)], fx.Int32))
                rocdl.s_waitcnt(lgkmcnt=0)
                barrier()

                def row_amax(row):
                    maxima = Vec(lds_view(row_max_base + row * WAVES, WAVES).load()).bitcast(fx.Float32)
                    peak = maxima[0]
                    for index in range_constexpr(1, WAVES):
                        peak = fx.arith.maxnumf(peak, maxima[index])
                    return peak

                for s in range_constexpr(S):
                    row = s * 16 + lane % 16
                    amax = row_amax(row)
                    # |x| <= amax keeps x * 240 / amax within the FP8 range after rounding.
                    inverse = (amax > 0.0).select(fx.Float32(rocdl.rcp(T.f32, amax.ir_value())) * FP8_MAX, fx.Float32(0.0))
                    q = [values[s][element] * inverse for element in range_constexpr(4)]
                    low = rocdl.cvt_pk_fp8_f32(T.i32, q[0], q[1], fx.Int32(0), False)
                    word = fx.Int32(rocdl.cvt_pk_fp8_f32(T.i32, q[2], q[3], low, True))
                    # This wave's 16 columns are 16 B chunk `wave` of the 64 B row; lane group g holds 4 g .. 4 g + 3.
                    lds_view(row * FP8_ROW_DWORDS + wave * 4 + lane // 16, 1).store(Vec.from_elements([word], fx.Int32))
                rocdl.s_waitcnt(lgkmcnt=0)
                barrier()
                out_rsrc = _rsrc_of(pOut, capacity * I)
                rows_per_store = 64 * WAVES // 4
                for part in range_constexpr((S * 16 + rows_per_store - 1) // rows_per_store):
                    out_row = part * rows_per_store + tid // 4
                    offset = (row_base + out_row) * I + n_tile * GATEUP_TILE_N + tid % 4 * 16
                    if const_expr(S * 16 % rows_per_store != 0):
                        offset = (out_row < S * 16).select(offset, fx.Int32(OOB_OFFSET))
                    rocdl.raw_ptr_buffer_store(
                        Vec(lds_view(out_row * FP8_ROW_DWORDS + tid % 4 * 4).load()).ir_value(),
                        out_rsrc,
                        offset.ir_value(),
                        zero,
                        aux=ir.IntegerAttr.get(T.i32, 0),
                    )
                scale_row = tid % SMALL_M_ROWS
                scale_offset = ((row_base + scale_row) * (I // A2_SCALE_COLUMNS) + n_tile) * 4
                rocdl.raw_ptr_buffer_store(
                    (row_amax(scale_row) * (1.0 / FP8_MAX)).ir_value(),
                    _rsrc_of(pScaleOut, capacity * (I // A2_SCALE_COLUMNS) * 4),
                    (tid < S * 16).select(scale_offset, fx.Int32(OOB_OFFSET)).ir_value(),
                    zero,
                    aux=ir.IntegerAttr.get(T.i32, 0),
                )

            def down_epilogue(S, accum, lane_rows, lane_ids):
                """routing weight * row in BF16, through a per-wave LDS transpose: 4 rows x 64 B per store/atomic."""
                weight_rsrc = _rsrc_of(pSortedWeights, capacity * 4)
                weights = [
                    fx.Float32(rocdl.raw_ptr_buffer_load(T.f32, weight_rsrc, ((row_base + row) * 4).ir_value(), zero, zero))
                    for row in lane_rows
                ]
                token_offsets = []
                for s in range_constexpr(S):
                    token, _, valid = decode(lane_ids[s])
                    token_offsets.append(valid.select(token * (H * 2), fx.Int32(-1)))
                out_base = fx.Int64(fx.ptrtoint(pOut)).ir_value()
                transpose = wave * (16 * TRANSPOSE_PITCH)
                write_base = transpose + lane % 16 * TRANSPOSE_PITCH + lane // 16 * 2
                read_base = transpose + lane // 16 * TRANSPOSE_PITCH + lane % 16
                column = n_tile * (DOWN_TILE_N * 2) + wave * 64 + lane % 16 * 4
                # LLVM does not count the asm stores: retire every load first so no later vmcnt wait covers them.
                rocdl.s_waitcnt(vmcnt=0)
                for s in range_constexpr(S):
                    for side in range_constexpr(2):
                        values = Vec(accum[None, s * 2 + side].load())
                        words = pack_bf16x2_rne([values[element] * weights[s] for element in range_constexpr(4)])
                        lds_view(write_base + side * 8, 2).store(Vec.from_elements(words, fx.Int32))
                    # Store i covers rows 4 i .. 4 i + 3 of the slice (lane // 16), 64 B each.
                    row_offsets = [
                        fx.Int32(rocdl.ds_bpermute(T.i32, ((4 * i + lane // 16) * 4).ir_value(), token_offsets[s].ir_value()))
                        for i in range_constexpr(4)
                    ]
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
                                None, [(row_offset + column).ir_value(), value, out_base], "global_atomic_pk_add_bf16 $0, $1, $2", "v,v,s",
                                has_side_effects=True,
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
    For every block with no route at row 128, writes its rows of a2 [blocks * 256, I] FP8 and a2_scale
    [blocks * 256, I/64] FP32 (see the module doc); other blocks are left to the BM256 GateUp."""
    tokens, hidden_dim = a.shape
    num_experts, two_inter, _ = w1.shape
    inter_dim = two_inter // 2
    num_blocks = expert_ids.numel()
    capacity = num_blocks * SORT_BLOCK_M
    assert a2.shape == (capacity, inter_dim) and a2.element_size() == 1
    assert a2_scale.shape == (capacity, inter_dim // A2_SCALE_COLUMNS) and a2_scale.dtype == torch.float32
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
    assert a2_scale.shape == (a2.shape[0], inter_dim // A2_SCALE_COLUMNS) and a2_scale.dtype == torch.float32
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
