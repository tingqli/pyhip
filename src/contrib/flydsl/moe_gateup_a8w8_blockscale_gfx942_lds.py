# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx942 FP8 block-scale MoE GateUp (BM256) on a 2x4 wave grid, A and B both staged through LDS.

For every sorted row, SiLU(gate) * up (gate and up from one FP8 GEMM) is quantized to FP8 per (row, 128 columns)
with FP32 scale amax / 240 and written at sorted row block * 256 + row of [blocks * 256, I] (scales
[I/128, blocks * 256]): the A2 of moe_down_a8w8_blockscale_gfx942_lds. B is aiter shuffle_weight(16,16) of
[E, 2I, K] (gate rows, then up rows); scales are FP32: A [K/128, tokens], B [E, 2I/128, K/128]. Sorted IDs are
(slot << 24) | token as from aiter moe_sorting(256); padding rows use an invalid token or slot, and blocks past
ceil(num_valid_ids[0] / 256) are never read, so sort buffers may keep spare capacity. Per K128 block, A rows
are gathered by sorted ID with eight lanes per row (128 B per row per load) into a 32 KiB LDS tile, and the B
packets gate0, gate1, up0, up1 (64 rows each) fill another 32 KiB. The pipeline follows
gemm_a8w8_blockscale_gfx942_2x4_lds_pf1: 2 waves along M (128 rows) x 4 along N (32 gate + 32 up columns); B is
read for the whole K block at stage 0 ("held"), A streams one 64-row packet per stage, and each packet of the
next K block is written to LDS one stage after its load. B rows are placed in LDS so that every wave's gate (and
up) columns are contiguous. The B scales of all K blocks sit in one VGPR and are read with v_readlane, so the K
loop issues no scalar loads. Row maxima meet in LDS across the waves along N, and the FP8 tile goes through LDS
so that every dwordx4 store writes 8 full rows x 128 B.
"""

import functools
from types import SimpleNamespace

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from .helpers import BufferTensor, run_compiled

SORT_BLOCK_M = 256
# A block with no route at this row holds at most 128 routes (bm128, merged pair tasks).
SMALL_M_ROWS = 128
TILE_N = 128
TILE_K = 128
TOKEN_MASK = 0xFFFFFF
# Past every descriptor used here; stays out of range after adding a tile-local byte offset.
OOB_OFFSET = 0x7FFF0000
LOG2E = 1.4426950408889634
PACKET_ROWS = 64
PACKET_DWORDS = PACKET_ROWS * TILE_K // 4
# Largest finite FP8 E4M3FNUZ (gfx942).
FP8_MAX = 240.0


def validate_parameters(tokens, inter_dim, hidden_dim, topk, num_experts):
    if not 0 < tokens <= TOKEN_MASK:
        raise ValueError(f"tokens must be in [1, {TOKEN_MASK}]")
    if inter_dim <= 0 or inter_dim % TILE_N:
        raise ValueError("inter_dim must be a positive multiple of 128")
    if hidden_dim < TILE_K or hidden_dim % TILE_K:
        raise ValueError("hidden_dim must be a positive multiple of 128")
    if not 0 < topk < 256 or num_experts <= 0:
        raise ValueError("require 0 < topk < 256 and num_experts > 0")
    # Descriptor byte counts and every voffset must stay below OOB_OFFSET. The output is sized by the sort
    # capacity, which the caller checks.
    extents = {
        "a": tokens * hidden_dim,
        "a_scale": tokens * (hidden_dim // TILE_K) * 4,
        "weight tile": (inter_dim + TILE_N) * hidden_dim,
    }
    over = [f"{name}={size:,}" for name, size in extents.items() if size >= OOB_OFFSET]
    if over:
        raise ValueError("buffer byte extents too large: " + ", ".join(over))


def _lds_dwords(row, column):
    # XOR swizzle of the 16B column keeps MFMA-pattern ds_read_b128 conflict free.
    return row * (TILE_K // 4) + ((row ^ column) & 7) * 4


@functools.cache
def compile_moe_gateup_a8w8_blockscale_gfx942_lds(inter_dim, hidden_dim, topk, num_experts, merged=None, bm128=False):
    """bm128: a block with no route at row 128 runs a 128-row pipeline in the same workgroup (two stages per K block,
    no loads, MFMAs or stores for rows 128..255, which are not written). Same grid and task order as without it.
    merged: plugin that maps workgroups to tasks itself (merged.task) and runs the blocks with no route at row 128
    in the same launch (merged.body); see docs/record/moe_small_m_gfx942.md. merged.key (str) must change with the
    plugin's code: the FlyDSL kernel cache keys on closure scalars and on sources next to this file only.
    """
    validate_parameters(1, inter_dim, hidden_dim, topk, num_experts)
    assert merged is None or not bm128
    merged_key = None if merged is None else merged.key
    assert merged_key is None or isinstance(merged_key, str)
    I, K = inter_dim, hidden_dim
    KB = K // TILE_K
    assert 2 * KB <= 64, "B scales of all K blocks must fit in one VGPR (hidden_dim <= 4096)"
    n_tiles = I // TILE_N
    b_k_bytes = TILE_K * 16
    scale_rows_b = 2 * I // 128
    # 16-row M / N slices per wave; N slices are gate first, then the matching up slices.
    m_slices = 8
    n_slices = 4
    gate_slices = 2
    side_rows = [0, I]

    def acc_index(m_slice, n_slice):
        return m_slice * n_slices + n_slice

    def stage_pairs(stage):
        """(m_slice, [n_slice, n_slice], side) of the four dequant pairs of a stage, in issue order."""
        return [(stage * 2 + i % 2, [i // 2 * 2, i // 2 * 2 + 1], i // 2) for i in range(4)]

    @fx.struct
    class SharedTiles:
        # A tile in dwords [0, 4 packets), B tile after it.
        ab: fx.Array[fx.Int32, 8 * PACKET_DWORDS, 16]

    B_BASE = 4 * PACKET_DWORDS
    # Per-row |x| maxima of each wave along N, 4 dwords per row after the FP8 tile (A area).
    ROW_MAX_BASE = B_BASE
    col_waves = 4

    @flyc.kernel(known_block_size=[512, 1, 1])
    def gateup_kernel(
        pA: fx.Pointer,
        pB: fx.Pointer,
        pC: fx.Pointer,
        pScaleC: fx.Pointer,
        pScaleA: fx.Pointer,
        pScaleB: fx.Pointer,
        pSortedIds: fx.Pointer,
        pExpertIds: fx.Pointer,
        pNumValid: fx.Pointer,
        num_tokens: fx.Int32,
        num_blocks: fx.Int32,
    ):
        tid = fx.thread_idx.x
        lane = tid % 64
        wave = tid // 64
        group = fx.Int32(rocdl.readfirstlane(T.i32, (tid // 256).ir_value()))
        wave_m = group
        wave_n = fx.Int32(rocdl.readfirstlane(T.i32, (wave % 4).ir_value()))
        pid = fx.Int32(fx.block_idx.x)

        def rsrc_of(ptr, num_records_bytes):
            tensor = fx.rocdl.make_buffer_tensor(
                fx.make_view(ptr, fx.make_layout(1, 1)), max_size=False, num_records_bytes=num_records_bytes
            )
            return fx.rocdl.get_buffer_rsrc(fx.get_iter(tensor))

        def load_i32(rsrc, voffset):
            zero = fx.Int32(0).ir_value()
            return fx.Int32(rocdl.raw_ptr_buffer_load(T.i32, rsrc, fx.Int32(voffset).ir_value(), zero, zero))

        def uniform(value):
            return fx.Int32(rocdl.readfirstlane(T.i32, value.ir_value()))

        valid_rows = uniform(load_i32(rsrc_of(pNumValid, 4), 0))
        valid_blocks = (valid_rows + (SORT_BLOCK_M - 1)) // SORT_BLOCK_M
        valid_blocks = (valid_blocks < num_blocks).select(valid_blocks, num_blocks)
        lds = fx.SharedAllocator().allocate(SharedTiles).peek()
        if const_expr(merged is not None):
            merged_task = merged.task(
                SimpleNamespace(
                    pid=pid, tid=tid, valid_blocks=valid_blocks, num_tokens=num_tokens, num_blocks=num_blocks,
                    pSortedIds=pSortedIds, scan=lds.ab.ptr, big_tiles=n_tiles,
                )
            )
            task = merged_task.block * n_tiles + merged_task.tile
            active = merged_task.active
        else:
            num_tasks = valid_blocks * n_tiles
            # Hardware deals workgroups round-robin over 8 XCDs: give each XCD a contiguous task
            # range so the N tiles of a sorted block (same A rows) share one L2.
            chunk = (num_tasks + 7) >> 3
            xcd_slot = pid >> 3
            task = (pid & 7) * chunk + xcd_slot
            active = (xcd_slot < chunk) & (task < num_tasks)

        def big_body(tile_rows=SORT_BLOCK_M):
            """The BM256 task `task`; tile_rows=128: the 128-row pipeline (bm128)."""
            # A packets streamed per K block; 16-row M slices per wave.
            stages = tile_rows // PACKET_ROWS
            used_slices = m_slices * tile_rows // SORT_BLOCK_M
            sorted_rsrc = rsrc_of(pSortedIds, num_blocks * (SORT_BLOCK_M * 4))
            expert_rsrc = rsrc_of(pExpertIds, num_blocks * 4)

            def route_ids(block, rows):
                return [load_i32(sorted_rsrc, (block * SORT_BLOCK_M + row) * 4) for row in rows]

            def decode(fused):
                """(token, valid) of a sorted ID."""
                token = fused & TOKEN_MASK
                slot = (fused >> 24) & 255
                return token, (token < num_tokens) & (slot < topk)

            # A slice j covers block rows slice_rows[j] + lane % 16, in LDS packet j // 2.
            slice_rows = [j // 2 * PACKET_ROWS + j % 2 * 32 + wave_m * 16 for j in range_constexpr(m_slices)]

            copy_row = tid // 8
            copy_col = tid % 8

            def copy_rows():
                return [tid // 8 + copy_packet * PACKET_ROWS for copy_packet in range_constexpr(stages)]

            def slice_lane_rows():
                return [slice_rows[j] + tid % 16 for j in range_constexpr(used_slices)]

            def task_offsets(a_ids, sa_ids):
                """A copy and scale_a byte offsets of a block's rows."""
                a_offsets, sa_offsets = [], []
                for fused in a_ids:
                    token, valid = decode(fused)
                    a_offsets.append(valid.select(token * K + tid % 8 * 16, fx.Int32(OOB_OFFSET)))
                for fused in sa_ids:
                    token, valid = decode(fused)
                    sa_offsets.append(valid.select(token * 4, fx.Int32(OOB_OFFSET)))
                return a_offsets, sa_offsets

            a_tensor = fx.rocdl.make_buffer_tensor(
                fx.make_view(pA, fx.make_layout(1, 1)), max_size=False, num_records_bytes=num_tokens * K
            )
            a_packet = BufferTensor(fx.make_view(fx.get_iter(a_tensor), fx.make_layout(4, 1)))
            scale_a_rsrc = rsrc_of(pScaleA, num_tokens * (KB * 4))
            scale_b_rsrc = rsrc_of(pScaleB, num_experts * scale_rows_b * (KB * 4))
            out_rsrc = rsrc_of(pC, num_blocks * (SORT_BLOCK_M * I))
            out_scale_rsrc = rsrc_of(pScaleC, num_blocks * (SORT_BLOCK_M * (I // 128) * 4))
            zero = fx.Int32(0).ir_value()

            def task_scale_b(expert, n_tile):
                """B scale of (side, K block) in lane side * KB + block."""
                scale_lane = tid % 64
                gate_scale_row = expert * scale_rows_b + n_tile
                scale_side = scale_lane // KB
                scale_b_offset = (gate_scale_row + scale_side * (I // 128)) * (KB * 4) + (scale_lane - scale_side * KB) * 4
                return fx.Float32(
                    rocdl.raw_ptr_buffer_load(
                        T.f32, scale_b_rsrc, (scale_lane < 2 * KB).select(scale_b_offset, fx.Int32(OOB_OFFSET)).ir_value(), zero, zero
                    )
                )

            # LDS rows [16 g, 16 g + 16) of B packet q hold this 16-row weight group of side q // 2, so
            # each wave's N slices are adjacent gate (and up) columns.
            loader = wave // 2
            weight_groups = [loader * 2 + q % 2 for q in range(4)]
            b_voffsets = [
                (side_rows[q // 2] + weight_groups[q] * 16) * K + wave % 2 * 1024 + lane * 16 for q in range_constexpr(4)
            ]
            b_write = _lds_dwords(loader * 16 + lane % 16, wave % 2 * 4 + lane // 16)
            a_write = _lds_dwords(copy_row, copy_col)
            # Per (distinct packet-relative row, half_k) read addresses; the packet goes in the immediate offset.
            a_read = [[_lds_dwords(r * 32 + wave_m * 16 + lane % 16, lane // 16 + h * 4) for h in range(2)] for r in range(2)]
            b_read = [_lds_dwords(wave_n * 16 + lane % 16, lane // 16 + h * 4) for h in range(2)]

            def lds_view(dwords, count=4):
                return fx.make_view(lds.ab.ptr + dwords, fx.make_layout(count, 1))

            def read_a_slice(j, half_k):
                return lds_view(a_read[j % 2][half_k] + j // 2 * PACKET_DWORDS).load()

            def read_b_slice(n_slice, half_k):
                return lds_view(b_read[half_k] + (B_BASE + n_slice * PACKET_DWORDS)).load()

            def task_ops(n_tile, expert, a_offsets, sa_offsets, sb_lanes):
                """Global loads and B scales of one task."""
                b_tensor = fx.rocdl.make_buffer_tensor(
                    fx.make_view(
                        pB + (fx.Int64(expert) * (2 * I * K // 4) + fx.Int64(n_tile) * (TILE_N * K // 4)),
                        fx.make_layout(1, 1),
                    ),
                    max_size=False,
                    num_records_bytes=(I + TILE_N) * K,
                )
                b_packet = BufferTensor(fx.make_view(fx.get_iter(b_tensor), fx.make_layout(4, 1)))

                def load_a(k_block, packet):
                    return a_packet.load(voffset_bytes=a_offsets[packet], soffset_bytes=k_block * TILE_K)

                def load_b(k_block, packet):
                    return b_packet.load(voffset_bytes=b_voffsets[packet], soffset_bytes=k_block * b_k_bytes)

                def load_scale_a(k_block, j):
                    value = rocdl.raw_ptr_buffer_load(
                        T.f32, scale_a_rsrc, fx.Int32(sa_offsets[j]).ir_value(), fx.Int32(k_block * num_tokens * 4).ir_value(), zero
                    )
                    return Vec.from_elements([fx.Float32(value)], fx.Float32)

                def scale_b(side, k_block):
                    index = fx.Int32(side * KB) + k_block
                    return fx.Float32(rocdl.readlane(T.f32, sb_lanes.ir_value(), index.ir_value()))

                return SimpleNamespace(load_a=load_a, load_b=load_b, load_scale_a=load_scale_a, scale_b=scale_b)

            def barrier():
                rocdl.sched_barrier(0)
                rocdl.s_barrier()
                rocdl.sched_barrier(0)

            # A stages all packets of the next block, B two slots (bm128: four).
            s_stage = fx.make_rmem_tensor(fx.make_layout((4, stages), (1, 4)), fx.Int32)
            h_stage = fx.make_rmem_tensor(fx.make_layout((4, 2 if stages == 4 else 4), (1, 4)), fx.Int32)
            # held column N slice * 2 + half_k for the four B slices of the current block.
            held = fx.make_rmem_tensor(fx.make_layout((4, 8), (1, 4)), fx.Int32)
            # scale_a column (stage parity) * 2 + slice-in-stage.
            scale_a = fx.make_rmem_tensor(fx.make_layout((1, 4), (1, 1)), fx.Float32)
            accum = fx.make_rmem_tensor(fx.make_layout((4, 32), (1, 4)), fx.Float32)
            partial = fx.make_rmem_tensor(fx.make_layout((4, 4), (1, 4)), fx.Float32)
            partial_scale = fx.make_rmem_tensor(fx.make_layout((1, 2), (1, 1)), fx.Float32)

            def load_s(ops, k_block, packet):
                s_stage[None, packet].store(ops.load_a(k_block, packet))

            def load_h(ops, k_block, packet, slot):
                h_stage[None, slot].store(ops.load_b(k_block, packet))

            def write_s(packet):
                lds_view(a_write + packet * PACKET_DWORDS).store(s_stage[None, packet].load())

            def write_h(packet, value):
                lds_view(b_write + (B_BASE + packet * PACKET_DWORDS)).store(value)

            def read_held():
                for held_index in range_constexpr(4):
                    for held_half in range_constexpr(2):
                        held[None, held_index * 2 + held_half].store(read_b_slice(held_index, held_half))

            def held_word(index, half_k, word):
                return Vec(held[None, index * 2 + half_k].load()).bitcast(fx.Int64)[word]

            last_pair = stage_pairs(stages - 1)[3]

            def run_k(ops, k_block):
                # The last block re-fetches clamped blocks so it stays inside the loop body.
                k_next = (k_block + 1 < KB).select(k_block + 1, KB - 1)
                previous = last_pair
                for stage in range_constexpr(stages):
                    if const_expr(stage == 0):
                        write_s(stages - 1)
                    elif const_expr(stages == 4):
                        write_s(stage - 1)
                        write_h(stage - 1, h_stage[None, (stage - 1) % 2].load())
                        if const_expr(stage == 3):
                            write_h(3, h_stage[None, 1].load())
                    else:
                        # bm128: the four B packets of the next block, loaded in stage 0.
                        write_s(0)
                        for packet in range_constexpr(4):
                            write_h(packet, h_stage[None, packet].load())
                    rocdl.sched_barrier(0)
                    if const_expr(stage == 0):
                        read_held()
                    # Streamed A slices 2s, 2s+1 of this stage.
                    streamed = [
                        [Vec(read_a_slice(stage * 2 + i, h)).bitcast(fx.Int64) for h in range_constexpr(2)]
                        for i in range_constexpr(2)
                    ]
                    rocdl.sched_barrier(0)
                    for next_i in range_constexpr(2):
                        if const_expr(stage < stages - 1):
                            scale_a[None, (stage + 1) % 2 * 2 + next_i].store(ops.load_scale_a(k_block, (stage + 1) * 2 + next_i))
                        else:
                            scale_a[None, next_i].store(ops.load_scale_a(k_next, next_i))
                    load_s(ops, k_next, stage)
                    if const_expr(stages == 4):
                        if const_expr(stage < 3):
                            load_h(ops, k_next, stage, stage % 2)
                        if const_expr(stage == 2):
                            load_h(ops, k_next, 3, 1)
                    elif const_expr(stage == 0):
                        for packet in range_constexpr(4):
                            load_h(ops, k_next, packet, packet)
                    rocdl.sched_barrier(0)
                    rocdl.s_waitcnt(lgkmcnt=0)
                    barrier()
                    rocdl.s_setprio(3)
                    for pair_id, (m_slice, n_pair, side) in enumerate(stage_pairs(stage)):
                        # Every pair of a stage alternates partial slots: entries pair_id % 2 * 2 + {0, 1}.
                        base, old_base = pair_id % 2 * 2, (pair_id + 1) % 2 * 2
                        for k_step in range_constexpr(4):
                            half_k, word = divmod(k_step, 2)
                            for inner in range_constexpr(2):
                                mfma_index = k_step * 2 + inner
                                a_value = streamed[m_slice % 2][half_k][word]
                                b_value = held_word(n_pair[inner], half_k, word)
                                accumulator = Vec.filled(4, 0.0, fx.Float32) if k_step == 0 else Vec(partial[None, base + inner].load())
                                partial[None, base + inner].store(
                                    Vec(rocdl.mfma_f32_16x16x32_fp8_fp8(T.f32x4, [b_value, a_value, accumulator, 0, 0, 0]))
                                )
                                # Keep MFMA -> FMA order: the FMA's partial then has two MFMAs of XDL latency cover.
                                rocdl.sched_barrier(0)
                                if const_expr(mfma_index == 0):
                                    a_scale = Vec(scale_a[None, stage % 2 * 2 + m_slice % 2].load())[0]
                                    partial_scale[None, pair_id % 2].store(
                                        Vec.from_elements([a_scale * ops.scale_b(side, k_block)], fx.Float32)
                                    )
                                # Retire one element of the previous pair under each MFMA.
                                old_m, old_n, _ = previous
                                source = old_base + mfma_index // 4
                                target = accum[None, acc_index(old_m, old_n[mfma_index // 4])]
                                element = mfma_index % 4
                                target[element] = fx.fma(
                                    Vec(partial[None, source].load())[element],
                                    Vec(partial_scale[None, (pair_id + 1) % 2].load())[0],
                                    target[element],
                                )
                                rocdl.sched_barrier(0)
                        previous = (m_slice, n_pair, side)
                    rocdl.s_setprio(0)
                    barrier()

            def k_loop(ops):
                if const_expr(KB % 2):
                    run_k(ops, fx.Int32(0))
                for pair in range(0, KB // 2):
                    k_block = fx.Int32(pair) * 2 + KB % 2
                    run_k(ops, k_block)
                    run_k(ops, k_block + 1)

            def start_k_loop():
                accum.fill(0)
                # The first stage retires these zeros as the "previous" pair.
                for pro_index in range_constexpr(4):
                    partial[None, pro_index].store(Vec.filled(4, 0.0, fx.Float32))
                for pro_index in range_constexpr(2):
                    partial_scale[None, pro_index].store(Vec.filled(1, 0.0, fx.Float32))
                rocdl.s_waitcnt(lgkmcnt=0)
                barrier()
                # Waves 4-7 trail by one barrier so their memory phase overlaps waves 0-3 compute.
                if group == 1:
                    barrier()

            def finish_k_loop():
                # The last pair has no later MFMA shadow.
                last_m, last_n, _ = last_pair
                final_scale = Vec(partial_scale[None, 1].load())[0]
                for inner in range_constexpr(2):
                    target = accum[None, acc_index(last_m, last_n[inner])]
                    values = Vec(partial[None, 2 + inner].load())
                    for element in range_constexpr(4):
                        target[element] = fx.fma(values[element], final_scale, target[element])
                if group == 0:
                    barrier()

            def silu_mul(gate_values, up_values):
                result = []
                for element in range_constexpr(4):
                    gate = gate_values[element]
                    exponent = fx.Float32(rocdl.exp2(T.f32, (gate * -LOG2E).ir_value()))
                    inverse = fx.Float32(rocdl.rcp(T.f32, (exponent + 1.0).ir_value()))
                    result.append(gate * inverse * up_values[element])
                return result

            def row_max(row):
                partial = Vec(lds_view(ROW_MAX_BASE + row * 4, col_waves).load()).bitcast(fx.Float32)
                peak = partial[0]
                for wave_col in range_constexpr(1, col_waves):
                    peak = fx.arith.maxnumf(peak, partial[wave_col])
                return peak

            def epilogue(block, n_tile):
                # Padding rows of the tile (zero A rows) are written too: FP8 zeros with scale 0.
                finish_k_loop()
                values = [
                    [
                        silu_mul(
                            Vec(accum[None, acc_index(m_slice, gate)].load()),
                            Vec(accum[None, acc_index(m_slice, gate_slices + gate)].load()),
                        )
                        for gate in range_constexpr(gate_slices)
                    ]
                    for m_slice in range_constexpr(used_slices)
                ]
                for m_slice in range_constexpr(used_slices):
                    peak = fx.Float32(0.0)
                    for gate_values in values[m_slice]:
                        for value in gate_values:
                            peak = fx.arith.maxnumf(peak, fx.math.absf(value))
                    # Lanes l, l ^ 16, l ^ 32, l ^ 48 hold this wave's other columns of row l % 16.
                    for shift in (16, 32):
                        other = rocdl.ds_bpermute(T.i32, fx.Int32((lane ^ shift) * 4).ir_value(), peak.bitcast(fx.Int32).ir_value())
                        peak = fx.arith.maxnumf(peak, fx.Int32(other).bitcast(fx.Float32))
                    if lane < 16:
                        lds_view(ROW_MAX_BASE + (slice_rows[m_slice] + lane) * 4 + wave_n, 1).store(
                            Vec.from_elements([peak.bitcast(fx.Int32)], fx.Int32)
                        )
                rocdl.s_waitcnt(lgkmcnt=0)
                barrier()
                for m_slice in range_constexpr(used_slices):
                    row = slice_rows[m_slice] + lane % 16
                    amax = row_max(row)
                    # |x| <= amax keeps x * 240 / amax within the FP8 range after rounding.
                    inverse = (amax > 0.0).select(fx.Float32(rocdl.rcp(T.f32, amax.ir_value())) * FP8_MAX, fx.Float32(0.0))
                    for gate in range_constexpr(gate_slices):
                        q = [values[m_slice][gate][element] * inverse for element in range_constexpr(4)]
                        low = rocdl.cvt_pk_fp8_f32(T.i32, q[0], q[1], fx.Int32(0), False)
                        word = fx.Int32(rocdl.cvt_pk_fp8_f32(T.i32, q[2], q[3], low, True))
                        # 4 FP8 columns 4 g .. 4 g + 3 of the 16-column chunk; lane group g = lane // 16.
                        lds_view(_lds_dwords(row, wave_n * gate_slices + gate) + lane // 16, 1).store(
                            Vec.from_elements([word], fx.Int32)
                        )
                rocdl.s_waitcnt(lgkmcnt=0)
                barrier()
                for part in range_constexpr(tile_rows // 64):
                    out_row = part * 64 + tid // 8
                    rocdl.raw_ptr_buffer_store(
                        Vec(lds_view(_lds_dwords(out_row, tid % 8)).load()).ir_value(),
                        out_rsrc,
                        ((block * SORT_BLOCK_M + out_row) * I + n_tile * TILE_N + tid % 8 * 16).ir_value(),
                        zero,
                        aux=ir.IntegerAttr.get(T.i32, 0),
                    )
                scale_row = tid % SORT_BLOCK_M
                scale_offset = ((n_tile * num_blocks + block) * SORT_BLOCK_M + scale_row) * 4
                rocdl.raw_ptr_buffer_store(
                    (row_max(scale_row) * (1.0 / FP8_MAX)).ir_value(),
                    out_scale_rsrc,
                    (tid < tile_rows).select(scale_offset, fx.Int32(OOB_OFFSET)).ir_value(),
                    zero,
                    aux=ir.IntegerAttr.get(T.i32, 0),
                )

            block = task // n_tiles
            n_tile = task - block * n_tiles
            expert = uniform(load_i32(expert_rsrc, block * 4))
            a_offsets, sa_offsets = task_offsets(route_ids(block, copy_rows()), route_ids(block, slice_lane_rows()))
            ops = task_ops(n_tile, expert, a_offsets, sa_offsets, task_scale_b(expert, n_tile))
            for pro_packet in range_constexpr(stages):
                load_s(ops, 0, pro_packet)
            for pair_begin in range_constexpr(0, 4, 2):
                for pro_packet in range_constexpr(pair_begin, pair_begin + 2):
                    load_h(ops, 0, pro_packet, pro_packet % 2)
                for pro_packet in range_constexpr(pair_begin, pair_begin + 2):
                    write_h(pro_packet, h_stage[None, pro_packet % 2].load())
                rocdl.sched_barrier(0)
            for pro_slice in range_constexpr(2):
                scale_a[None, pro_slice].store(ops.load_scale_a(0, pro_slice))
            for pro_packet in range_constexpr(stages):
                write_s(pro_packet)
            start_k_loop()
            k_loop(ops)
            epilogue(block, n_tile)

        def small_ctx():
            return SimpleNamespace(
                pA=pA, pB=pB, pOut=pC, pScaleOut=pScaleC, pScaleA=pScaleA, pScaleB=pScaleB, pSortedIds=pSortedIds,
                pSortedWeights=None, pExpertIds=pExpertIds, num_tokens=num_tokens, num_blocks=num_blocks, tid=tid,
                lds_base=lds.ab.ptr,
            )

        if const_expr(merged is not None):
            if active:
                if merged_task.big:
                    big_body()
                else:
                    merged.body(small_ctx(), merged_task.block, merged_task.tile)
        elif const_expr(bm128):
            if active:
                # Valid routes are a prefix of the block: no route at row 128 leaves rows 128..255 empty.
                bottom_id = uniform(load_i32(rsrc_of(pSortedIds, num_blocks * (SORT_BLOCK_M * 4)), ((task // n_tiles) * SORT_BLOCK_M + SMALL_M_ROWS) * 4))
                if ((bottom_id & TOKEN_MASK) < num_tokens) & (((bottom_id >> 24) & 255) < topk):
                    big_body()
                    # Distinct last instructions: SimplifyCFG must not sink the two pipelines' common tail into one block.
                    llvm.inline_asm(None, [], "; bm256 task", "", has_side_effects=True)
                else:
                    big_body(SMALL_M_ROWS)
                    llvm.inline_asm(None, [], "; bm128 task", "", has_side_effects=True)
        else:
            if active:
                big_body()

    @flyc.jit
    def launch(
        pA: fx.Pointer,
        pB: fx.Pointer,
        pC: fx.Pointer,
        pScaleC: fx.Pointer,
        pScaleA: fx.Pointer,
        pScaleB: fx.Pointer,
        pSortedIds: fx.Pointer,
        pExpertIds: fx.Pointer,
        pNumValid: fx.Pointer,
        num_tokens: fx.Int32,
        num_blocks: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        if const_expr(merged_key is not None):
            grid_x = merged.grid(num_blocks)
        else:
            grid_x = (num_blocks * n_tiles + 7) // 8 * 8
        gateup_kernel(
            pA, pB, pC, pScaleC, pScaleA, pScaleB, pSortedIds, pExpertIds, pNumValid, num_tokens, num_blocks,
            value_attrs={"llvm.passthrough": [["target-features", "-packed-fp32-ops"]]},
        ).launch(
            grid=(grid_x, 1, 1),
            block=(512, 1, 1),
            stream=stream,
        )

    launch.compile_hints["target_features"] = "-packed-fp32-ops"
    return launch


def moe_gateup_a8w8_blockscale_gfx942_lds(
    a, w1, a_scale_t, w1_scale, sorted_ids, expert_ids, num_valid_ids, out, out_scale_t, topk, bm128=False
):
    """a[T,K] fp8, w1[E,2I,K] fp8 shuffle_weight(16,16), a_scale_t[K/128,T], w1_scale[E,2I/128,K/128] fp32.

    sorted_ids[blocks*256] / expert_ids[blocks] / num_valid_ids[>=1] int32 as from aiter moe_sorting(256);
    out [blocks * 256, I] FP8 receives SiLU(gate) * up of every sorted row, out_scale_t [I/128, blocks * 256] FP32
    its scales. bm128=True: blocks with at most 128 routes run the 128-row pipeline; their sorted rows 128..255
    are not written.
    """
    tokens, hidden_dim = a.shape
    num_experts, two_inter, _ = w1.shape
    inter_dim = two_inter // 2
    num_blocks = expert_ids.numel()
    capacity = num_blocks * SORT_BLOCK_M
    assert out.shape == (capacity, inter_dim) and out.element_size() == 1
    assert out_scale_t.shape == (inter_dim // 128, capacity) and out_scale_t.dtype == torch.float32
    assert capacity * inter_dim < OOB_OFFSET, "FP8 output too large for one buffer descriptor"
    validate_parameters(tokens, inter_dim, hidden_dim, topk, num_experts)
    assert sorted_ids.numel() >= capacity
    launch = compile_moe_gateup_a8w8_blockscale_gfx942_lds(inter_dim, hidden_dim, topk, num_experts, bm128=bool(bm128))
    run_compiled(
        launch,
        flyc.from_c_void_p(fx.Int32, a.data_ptr()),
        flyc.from_c_void_p(fx.Int32, w1.data_ptr()),
        flyc.from_c_void_p(fx.Int32, out.data_ptr()),
        flyc.from_c_void_p(fx.Float32, out_scale_t.data_ptr()),
        flyc.from_c_void_p(fx.Float32, a_scale_t.data_ptr()),
        flyc.from_c_void_p(fx.Float32, w1_scale.data_ptr()),
        flyc.from_c_void_p(fx.Int32, sorted_ids.data_ptr()),
        flyc.from_c_void_p(fx.Int32, expert_ids.data_ptr()),
        flyc.from_c_void_p(fx.Int32, num_valid_ids.data_ptr()),
        tokens,
        num_blocks,
        torch.cuda.current_stream(),
    )
    return out
