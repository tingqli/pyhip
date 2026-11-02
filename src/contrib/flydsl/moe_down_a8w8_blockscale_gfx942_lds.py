# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx942 FP8 block-scale MoE Down (GEMM2, BM256 x BN256) on 2x4 / 4x2 wave grids, A and B staged through LDS.

Per route y = routing weight * (A2 row @ W2[expert]^T): A2 is FP8 [tokens, topk, I] gathered by sorted ID
(row token * topk + slot), W2 is aiter shuffle_weight(16,16) of [E, H, I]. Scales are FP32: A2
[I/128, tokens * topk], W2 [E, H/128, I/128]. With sorted_a2, A2 is [blocks * 256, I] in sorted-row order (scales
[I/128, blocks * 256]): a block's rows are then contiguous instead of spread over the whole [tokens, topk, I]
tensor, which thrashes the L1 TLB on every K block. The K loop is the one of moe_gateup_a8w8_blockscale_gfx942_lds;
one workgroup takes a 256-row sorted block and 256 hidden columns.
  2x4: 2 waves along M (128 rows) x 4 along N (64 columns); B is held, A streams.
  4x2: 4 waves along M (64 rows) x 2 along N (128 columns); A is held, B streams.
A token's routes are summed in one of two ways:
  atomic=True: BF16 y is added into out [tokens, H] (zeroed beforehand) with global_atomic_pk_add_bf16. Each
      wave transposes its tile through LDS so that one atomic covers two rows x 128 contiguous bytes.
  atomic=False: BF16 y is stored at its sorted position in rows [blocks * 256, H], and inverse[token, slot]
      receives that position (filled with -1 beforehand); compile_moe_down_reduce then sums every token's routes
      in FP32, in slot order.
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
from .helpers import BufferTensor
from .moe_gateup_a8w8_blockscale_gfx942 import OOB_OFFSET, SORT_BLOCK_M, TOKEN_MASK
from .moe_gateup_a8w8_blockscale_gfx942_lds import LAYOUTS, PACKET_DWORDS, PACKET_ROWS, TILE_K, _lds_dwords

TILE_N = 256
# Atomic epilogue transpose tile per wave: 16 rows x 32 dwords; a 34-dword pitch keeps writes and reads conflict free.
TRANSPOSE_PITCH = 34
REDUCE_COLUMNS = 16


def validate_parameters(tokens, hidden_dim, inter_dim, topk, num_experts, capacity=SORT_BLOCK_M, sorted_a2=False):
    if not 0 < tokens <= TOKEN_MASK:
        raise ValueError(f"tokens must be in [1, {TOKEN_MASK}]")
    if hidden_dim <= 0 or hidden_dim % TILE_N or hidden_dim // REDUCE_COLUMNS > 1024:
        raise ValueError("hidden_dim must be a multiple of 256 in [256, 16384]")
    if inter_dim < TILE_K or inter_dim % TILE_K or inter_dim > 64 * TILE_K:
        raise ValueError("inter_dim must be a multiple of 128 in [128, 8192]")
    if not 0 < topk < 256 or num_experts <= 0:
        raise ValueError("require 0 < topk < 256 and num_experts > 0")
    # Descriptor byte counts and every offset must stay below OOB_OFFSET.
    a2_rows = capacity if sorted_a2 else tokens * topk
    extents = {
        "a2": a2_rows * inter_dim,
        "a2_scale": a2_rows * (inter_dim // TILE_K) * 4,
        "weight tile": TILE_N * inter_dim,
        "output": tokens * hidden_dim * 2,
        "rows": capacity * hidden_dim * 2,
    }
    over = [f"{name}={size:,}" for name, size in extents.items() if size >= OOB_OFFSET]
    if over:
        raise ValueError("buffer byte extents too large: " + ", ".join(over))


def _rsrc_of(ptr, num_records_bytes):
    tensor = fx.rocdl.make_buffer_tensor(
        fx.make_view(ptr, fx.make_layout(1, 1)), max_size=False, num_records_bytes=num_records_bytes
    )
    return fx.rocdl.get_buffer_rsrc(fx.get_iter(tensor))


@functools.cache
def compile_moe_down_a8w8_blockscale_gfx942_lds(
    hidden_dim, inter_dim, topk, num_experts, layout="2x4", xcd_swizzle=True, atomic=True, sorted_a2=False
):
    validate_parameters(1, hidden_dim, inter_dim, topk, num_experts, sorted_a2=sorted_a2)
    assert layout in LAYOUTS, layout
    H, K = hidden_dim, inter_dim
    KB = K // TILE_K
    n_tiles = H // TILE_N
    b_k_bytes = TILE_K * 16
    scale_rows_b = H // 128
    stream_a = layout == "2x4"
    # 16-row M / N slices per wave.
    m_slices = 8 if stream_a else 4
    n_slices = 32 // m_slices
    # 64-column groups of a wave: one LDS transpose (atomic) or two 8-row x 128 B stores (rows) each.
    col_groups = n_slices // 4
    wave_bytes = n_slices * 32

    def acc_index(m_slice, n_slice):
        return m_slice * n_slices + n_slice

    def stage_pairs(stage):
        """(m_slice, [n_slice, n_slice]) of the four dequant pairs of a stage, in issue order."""
        if const_expr(stream_a):
            return [(stage * 2 + i % 2, [i // 2 * 2, i // 2 * 2 + 1]) for i in range(4)]
        return [(i, [stage * 2, stage * 2 + 1]) for i in range(4)]

    def weight_group(packet, loader):
        """16-row group of the N tile held in LDS rows [16 loader, 16 loader + 16) of B packet `packet`.

        2x4 waves read N slice s from packet s, rows [16 wave_n, +16); 4x2 waves from packet s // 2,
        rows [32 wave_n + 16 (s % 2), +16). Either way a wave's N slices are its contiguous columns.
        """
        if const_expr(stream_a):
            return loader * 4 + packet
        return loader // 2 * 8 + packet * 2 + loader % 2

    @fx.struct
    class SharedTiles:
        # A tile in dwords [0, 4 packets), B tile after it.
        ab: fx.Array[fx.Int32, 8 * PACKET_DWORDS, 16]

    B_BASE = 4 * PACKET_DWORDS

    @flyc.kernel(known_block_size=[512, 1, 1])
    def down_kernel(
        pA: fx.Pointer,
        pB: fx.Pointer,
        pOut: fx.Pointer,
        pScaleA: fx.Pointer,
        pScaleB: fx.Pointer,
        pSortedIds: fx.Pointer,
        pSortedWeights: fx.Pointer,
        pExpertIds: fx.Pointer,
        pNumValid: fx.Pointer,
        pInverse: fx.Pointer,
        num_tokens: fx.Int32,
        num_blocks: fx.Int32,
    ):
        tid = fx.thread_idx.x
        lane = tid % 64
        wave = tid // 64
        group = fx.Int32(rocdl.readfirstlane(T.i32, (tid // 256).ir_value()))
        if const_expr(stream_a):
            wave_m = group
            wave_n = fx.Int32(rocdl.readfirstlane(T.i32, (wave % 4).ir_value()))
        else:
            wave_m = fx.Int32(rocdl.readfirstlane(T.i32, (wave // 2).ir_value()))
            wave_n = fx.Int32(rocdl.readfirstlane(T.i32, (wave % 2).ir_value()))
        pid = fx.Int32(fx.block_idx.x)

        def load_i32(rsrc, voffset):
            zero = fx.Int32(0).ir_value()
            return fx.Int32(rocdl.raw_ptr_buffer_load(T.i32, rsrc, fx.Int32(voffset).ir_value(), zero, zero))

        def uniform(value):
            return fx.Int32(rocdl.readfirstlane(T.i32, value.ir_value()))

        valid_rows = uniform(load_i32(_rsrc_of(pNumValid, 4), 0))
        valid_blocks = (valid_rows + (SORT_BLOCK_M - 1)) // SORT_BLOCK_M
        valid_blocks = (valid_blocks < num_blocks).select(valid_blocks, num_blocks)
        num_tasks = valid_blocks * n_tiles
        if const_expr(xcd_swizzle):
            # Hardware deals workgroups round-robin over 8 XCDs: give each XCD a contiguous task
            # range so the N tiles of a sorted block (same A rows) share one L2.
            xcd_tasks = (num_tasks + 7) >> 3
            xcd_slot = pid >> 3
            task = (pid & 7) * xcd_tasks + xcd_slot
            active = (xcd_slot < xcd_tasks) & (task < num_tasks)
        else:
            task = pid
            active = task < num_tasks

        if active:
            zero = fx.Int32(0).ir_value()
            sorted_rsrc = _rsrc_of(pSortedIds, num_blocks * (SORT_BLOCK_M * 4))
            weight_rsrc = _rsrc_of(pSortedWeights, num_blocks * (SORT_BLOCK_M * 4))
            route_rows = num_tokens * topk
            a2_rows = num_blocks * SORT_BLOCK_M if sorted_a2 else route_rows
            block = task // n_tiles
            n_tile = task - block * n_tiles
            expert = uniform(load_i32(_rsrc_of(pExpertIds, num_blocks * 4), block * 4))

            def route_ids(rows):
                return [load_i32(sorted_rsrc, (block * SORT_BLOCK_M + row) * 4) for row in rows]

            def decode(fused):
                """(A2 row, token, valid) of a sorted ID."""
                token = fused & TOKEN_MASK
                slot = (fused >> 24) & 255
                return token * topk + slot, token, (token < num_tokens) & (slot < topk)

            # A slice j covers block rows slice_rows[j] + lane % 16, in LDS packet slice_packet[j].
            if const_expr(stream_a):
                slice_packet = [j // 2 for j in range(8)]
                slice_offset = [j % 2 * 32 for j in range(8)]
            else:
                slice_packet = list(range(4))
                slice_offset = [0] * 4
            slice_rows = [slice_packet[j] * PACKET_ROWS + slice_offset[j] + wave_m * 16 for j in range_constexpr(m_slices)]
            lane_rows = [slice_rows[j] + lane % 16 for j in range_constexpr(m_slices)]
            copy_row = tid // 8
            copy_col = tid % 8

            a_offsets, sa_offsets = [], []
            copy_rows = [copy_row + copy_packet * PACKET_ROWS for copy_packet in range_constexpr(4)]
            for row, fused in zip(copy_rows, route_ids(copy_rows)):
                a_row, _, a_valid = decode(fused)
                a_row = block * SORT_BLOCK_M + row if sorted_a2 else a_row
                a_offsets.append(a_valid.select(a_row * K + copy_col * 16, fx.Int32(OOB_OFFSET)))
            for row, fused in zip(lane_rows, route_ids(lane_rows)):
                sa_row, _, sa_valid = decode(fused)
                sa_row = block * SORT_BLOCK_M + row if sorted_a2 else sa_row
                sa_offsets.append(sa_valid.select(sa_row * 4, fx.Int32(OOB_OFFSET)))

            a_tensor = fx.rocdl.make_buffer_tensor(
                fx.make_view(pA, fx.make_layout(1, 1)), max_size=False, num_records_bytes=a2_rows * K
            )
            a_packet = BufferTensor(fx.make_view(fx.get_iter(a_tensor), fx.make_layout(4, 1)))
            scale_a_rsrc = _rsrc_of(pScaleA, a2_rows * (KB * 4))
            b_tensor = fx.rocdl.make_buffer_tensor(
                fx.make_view(
                    pB + (fx.Int64(expert) * (H * K // 4) + fx.Int64(n_tile) * (TILE_N * K // 4)), fx.make_layout(1, 1)
                ),
                max_size=False,
                num_records_bytes=TILE_N * K,
            )
            b_packet = BufferTensor(fx.make_view(fx.get_iter(b_tensor), fx.make_layout(4, 1)))
            # A wave's columns lie in one 128-column scale group: B scale of K block k in lane k.
            scale_group = n_tile * 2 + (wave_n // 2 if stream_a else wave_n)
            scale_b_offset = ((expert * scale_rows_b + scale_group) * KB + lane) * 4
            sb_lanes = fx.Float32(
                rocdl.raw_ptr_buffer_load(
                    T.f32,
                    _rsrc_of(pScaleB, num_experts * scale_rows_b * (KB * 4)),
                    (lane < KB).select(scale_b_offset, fx.Int32(OOB_OFFSET)).ir_value(),
                    zero,
                    zero,
                )
            )

            loader = wave // 2
            b_voffsets = [weight_group(q, loader) * 16 * K + wave % 2 * 1024 + lane * 16 for q in range_constexpr(4)]
            b_write = _lds_dwords(loader * 16 + lane % 16, wave % 2 * 4 + lane // 16)
            a_write = _lds_dwords(copy_row, copy_col)
            # Per (distinct packet-relative row, half_k) read addresses; the packet goes in the immediate offset.
            if const_expr(stream_a):
                a_read = [[_lds_dwords(r * 32 + wave_m * 16 + lane % 16, lane // 16 + h * 4) for h in range(2)] for r in range(2)]
                b_read = [[_lds_dwords(wave_n * 16 + lane % 16, lane // 16 + h * 4) for h in range(2)]]
            else:
                a_read = [[_lds_dwords(wave_m * 16 + lane % 16, lane // 16 + h * 4) for h in range(2)]]
                b_read = [[_lds_dwords(wave_n * 32 + r * 16 + lane % 16, lane // 16 + h * 4) for h in range(2)] for r in range(2)]

            lds = fx.SharedAllocator().allocate(SharedTiles).peek()

            def lds_view(dwords, count=4):
                return fx.make_view(lds.ab.ptr + dwords, fx.make_layout(count, 1))

            def read_a_slice(j, half_k):
                return lds_view(a_read[j % 2 if stream_a else 0][half_k] + slice_packet[j] * PACKET_DWORDS).load()

            def read_b_slice(n_slice, half_k):
                if const_expr(stream_a):
                    return lds_view(b_read[0][half_k] + (B_BASE + n_slice * PACKET_DWORDS)).load()
                return lds_view(b_read[n_slice % 2][half_k] + (B_BASE + n_slice // 2 * PACKET_DWORDS)).load()

            def load_a(k_block, packet):
                return a_packet.load(voffset_bytes=a_offsets[packet], soffset_bytes=k_block * TILE_K)

            def load_b(k_block, packet):
                return b_packet.load(voffset_bytes=b_voffsets[packet], soffset_bytes=k_block * b_k_bytes)

            def load_scale_a(k_block, j):
                value = rocdl.raw_ptr_buffer_load(
                    T.f32, scale_a_rsrc, fx.Int32(sa_offsets[j]).ir_value(), fx.Int32(k_block * a2_rows * 4).ir_value(), zero
                )
                return Vec.from_elements([fx.Float32(value)], fx.Float32)

            def scale_b(k_block):
                return fx.Float32(rocdl.readlane(T.f32, sb_lanes.ir_value(), fx.Int32(k_block).ir_value()))

            def barrier():
                rocdl.sched_barrier(0)
                rocdl.s_barrier()
                rocdl.sched_barrier(0)

            # The streamed operand stages all four packets of the next block, the held one two slots.
            s_stage = fx.make_rmem_tensor(fx.make_layout((4, 4), (1, 4)), fx.Int32)
            h_stage = fx.make_rmem_tensor(fx.make_layout((4, 2), (1, 4)), fx.Int32)
            # held column slice * 2 + half_k for the four held slices of the current block.
            held = fx.make_rmem_tensor(fx.make_layout((4, 8), (1, 4)), fx.Int32)
            # 2x4: scale_a column (stage parity) * 2 + slice-in-stage; 4x2: (block parity) * 4 + slice.
            scale_a = fx.make_rmem_tensor(fx.make_layout((1, 8), (1, 1)), fx.Float32)
            accum = fx.make_rmem_tensor(fx.make_layout((4, 32), (1, 4)), fx.Float32)
            partial = fx.make_rmem_tensor(fx.make_layout((4, 4), (1, 4)), fx.Float32)
            partial_scale = fx.make_rmem_tensor(fx.make_layout((1, 2), (1, 1)), fx.Float32)

            def load_s(k_block, packet):
                s_stage[None, packet].store(load_a(k_block, packet) if stream_a else load_b(k_block, packet))

            def load_h(k_block, packet, slot):
                h_stage[None, slot].store(load_b(k_block, packet) if stream_a else load_a(k_block, packet))

            def write_s(packet):
                if const_expr(stream_a):
                    lds_view(a_write + packet * PACKET_DWORDS).store(s_stage[None, packet].load())
                else:
                    lds_view(b_write + (B_BASE + packet * PACKET_DWORDS)).store(s_stage[None, packet].load())

            def write_h(packet, value):
                if const_expr(stream_a):
                    lds_view(b_write + (B_BASE + packet * PACKET_DWORDS)).store(value)
                else:
                    lds_view(a_write + packet * PACKET_DWORDS).store(value)

            def read_held():
                # 2x4: the four N slices of B; 4x2: the four M slices of A.
                for held_index in range_constexpr(4):
                    for held_half in range_constexpr(2):
                        value = read_b_slice(held_index, held_half) if stream_a else read_a_slice(held_index, held_half)
                        held[None, held_index * 2 + held_half].store(value)

            def held_word(index, half_k, word):
                return Vec(held[None, index * 2 + half_k].load()).bitcast(fx.Int64)[word]

            last_pair = stage_pairs(3)[3]

            def run_k(parity, k_block):
                # The last block re-fetches clamped blocks so it stays inside the loop body.
                k_next = (k_block + 1 < KB).select(k_block + 1, KB - 1)
                previous = last_pair
                for stage in range_constexpr(4):
                    if const_expr(stage == 0):
                        write_s(3)
                    else:
                        write_s(stage - 1)
                        write_h(stage - 1, h_stage[None, (stage - 1) % 2].load())
                        if const_expr(stage == 3):
                            write_h(3, h_stage[None, 1].load())
                    rocdl.sched_barrier(0)
                    if const_expr(stage == 0):
                        read_held()
                    # Streamed slices of this stage: A slices 2s, 2s+1 (2x4) or B slices 2s, 2s+1 (4x2).
                    streamed = [
                        [Vec(read_a_slice(stage * 2 + i, h) if stream_a else read_b_slice(stage * 2 + i, h)).bitcast(fx.Int64) for h in range_constexpr(2)]
                        for i in range_constexpr(2)
                    ]
                    rocdl.sched_barrier(0)
                    if const_expr(stream_a):
                        for next_i in range_constexpr(2):
                            if const_expr(stage < 3):
                                scale_a[None, (stage + 1) % 2 * 2 + next_i].store(load_scale_a(k_block, (stage + 1) * 2 + next_i))
                            else:
                                scale_a[None, next_i].store(load_scale_a(k_next, next_i))
                    elif const_expr(stage == 0):
                        for next_j in range_constexpr(4):
                            scale_a[None, (1 - parity) * 4 + next_j].store(load_scale_a(k_next, next_j))
                    load_s(k_next, stage)
                    if const_expr(stage < 3):
                        load_h(k_next, stage, stage % 2)
                    if const_expr(stage == 2):
                        load_h(k_next, 3, 1)
                    rocdl.sched_barrier(0)
                    rocdl.s_waitcnt(lgkmcnt=0)
                    barrier()
                    rocdl.s_setprio(3)
                    for pair_id, (m_slice, n_pair) in enumerate(stage_pairs(stage)):
                        # Every pair of a stage alternates partial slots: entries pair_id % 2 * 2 + {0, 1}.
                        base, old_base = pair_id % 2 * 2, (pair_id + 1) % 2 * 2
                        for k_step in range_constexpr(4):
                            half_k, word = divmod(k_step, 2)
                            for inner in range_constexpr(2):
                                mfma_index = k_step * 2 + inner
                                n_slice = n_pair[inner]
                                if const_expr(stream_a):
                                    a_value = streamed[m_slice % 2][half_k][word]
                                    b_value = held_word(n_slice, half_k, word)
                                else:
                                    a_value = held_word(m_slice, half_k, word)
                                    b_value = streamed[inner][half_k][word]
                                accumulator = Vec.filled(4, 0.0, fx.Float32) if k_step == 0 else Vec(partial[None, base + inner].load())
                                partial[None, base + inner].store(
                                    Vec(rocdl.mfma_f32_16x16x32_fp8_fp8(T.f32x4, [b_value, a_value, accumulator, 0, 0, 0]))
                                )
                                # Keep MFMA -> FMA order: the FMA's partial then has two MFMAs of XDL latency cover.
                                rocdl.sched_barrier(0)
                                if const_expr(mfma_index == 0):
                                    if const_expr(stream_a):
                                        a_scale = Vec(scale_a[None, stage % 2 * 2 + m_slice % 2].load())[0]
                                    else:
                                        a_scale = Vec(scale_a[None, parity * 4 + m_slice].load())[0]
                                    partial_scale[None, pair_id % 2].store(Vec.from_elements([a_scale * scale_b(k_block)], fx.Float32))
                                # Retire one element of the previous pair under each MFMA.
                                old_m, old_n = previous
                                source = old_base + mfma_index // 4
                                target = accum[None, acc_index(old_m, old_n[mfma_index // 4])]
                                element = mfma_index % 4
                                target[element] = fx.fma(
                                    Vec(partial[None, source].load())[element],
                                    Vec(partial_scale[None, (pair_id + 1) % 2].load())[0],
                                    target[element],
                                )
                                rocdl.sched_barrier(0)
                        previous = (m_slice, n_pair)
                    rocdl.s_setprio(0)
                    barrier()

            def finish_k_loop():
                # The last pair has no later MFMA shadow.
                last_m, last_n = last_pair
                final_scale = Vec(partial_scale[None, 1].load())[0]
                for inner in range_constexpr(2):
                    target = accum[None, acc_index(last_m, last_n[inner])]
                    values = Vec(partial[None, 2 + inner].load())
                    for element in range_constexpr(4):
                        target[element] = fx.fma(values[element], final_scale, target[element])
                # Pairs with group 1's last K-loop barrier: after it no wave reads the A/B tiles any more.
                if group == 0:
                    barrier()

            def weighted_words(m_slice, n_slice, weight):
                """BF16 of routing weight * the 4 consecutive columns this lane holds (row lane % 16)."""
                values = Vec(accum[None, acc_index(m_slice, n_slice)].load())
                return pack_bf16x2_rne([values[element] * weight for element in range_constexpr(4)])

            # Prologue: K block 0 into LDS.
            for pro_packet in range_constexpr(4):
                load_s(0, pro_packet)
            for pair_begin in range_constexpr(0, 4, 2):
                for pro_packet in range_constexpr(pair_begin, pair_begin + 2):
                    load_h(0, pro_packet, pro_packet % 2)
                for pro_packet in range_constexpr(pair_begin, pair_begin + 2):
                    write_h(pro_packet, h_stage[None, pro_packet % 2].load())
                rocdl.sched_barrier(0)
            for pro_slice in range_constexpr(2 if stream_a else 4):
                scale_a[None, pro_slice].store(load_scale_a(0, pro_slice))
            for pro_packet in range_constexpr(4):
                write_s(pro_packet)
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

            if const_expr(KB % 2):
                run_k(0, fx.Int32(0))
            for k_pair in range(0, KB // 2):
                k_block = fx.Int32(k_pair) * 2 + KB % 2
                run_k(KB % 2, k_block)
                run_k(1 - KB % 2, k_block + 1)

            # Output rows are reloaded here (not kept through the K loop) so the loads overlap the drain.
            out_ids = route_ids(lane_rows)
            weights = [
                fx.Float32(rocdl.raw_ptr_buffer_load(T.f32, weight_rsrc, ((block * SORT_BLOCK_M + row) * 4).ir_value(), zero, zero))
                for row in lane_rows
            ]
            if const_expr(atomic):
                token_offsets = []
                for fused in out_ids:
                    _, out_token, out_valid = decode(fused)
                    token_offsets.append(out_valid.select(out_token * (H * 2), fx.Int32(-1)))
                out_base = fx.Int64(fx.ptrtoint(pOut)).ir_value()
                transpose = wave * (16 * TRANSPOSE_PITCH)
                write_base = transpose + lane % 16 * TRANSPOSE_PITCH + lane // 16 * 2
                read_base = transpose + lane // 32 * TRANSPOSE_PITCH + lane % 32
                column_base = n_tile * (TILE_N * 2) + wave_n * wave_bytes + lane % 32 * 4
                finish_k_loop()
                # LLVM does not count the asm atomics: retire every load first so no later vmcnt wait covers them.
                rocdl.s_waitcnt(vmcnt=0)
                for j in range_constexpr(m_slices):
                    # Atomic i of a group covers rows 2i (lanes 0-31) and 2i + 1 (lanes 32-63) of slice j.
                    row_offsets = [
                        fx.Int32(rocdl.ds_bpermute(T.i32, ((lane // 32 + 2 * i) * 4).ir_value(), token_offsets[j].ir_value()))
                        for i in range_constexpr(8)
                    ]
                    for c in range_constexpr(col_groups):
                        for n_local in range_constexpr(4):
                            lds_view(write_base + n_local * 8, 2).store(
                                Vec.from_elements(weighted_words(j, c * 4 + n_local, weights[j]), fx.Int32)
                            )
                        # One opaque copy of all 8 LDS reads: one wait, and LLVM cannot sink a read into its
                        # atomic's branch (which serializes 8 LDS round trips).
                        reads = [Vec(lds_view(read_base + 2 * i * TRANSPOSE_PITCH, 1).load())[0].ir_value() for i in range_constexpr(8)]
                        copies = llvm.inline_asm(
                            ir.Type.parse("!llvm.struct<(" + ", ".join(["i32"] * 8) + ")>"),
                            reads,
                            "",
                            ",".join(["=v"] * 8 + [str(i) for i in range(8)]),
                            has_side_effects=True,
                        )
                        for i in range_constexpr(8):
                            value = fx.Int32(llvm.extractvalue(T.i32, copies, [i]))
                            row_offset = row_offsets[i]
                            if row_offset >= 0:
                                llvm.inline_asm(
                                    None,
                                    [(row_offset + column_base + c * 128).ir_value(), value.ir_value(), out_base],
                                    "global_atomic_pk_add_bf16 $0, $1, $2",
                                    "v,v,s",
                                    has_side_effects=True,
                                )
            else:
                row_offsets_out = []
                for row, fused in zip(lane_rows, out_ids):
                    _, _, out_valid = decode(fused)
                    row_offsets_out.append(
                        out_valid.select((block * SORT_BLOCK_M + row) * (H * 2) + n_tile * (TILE_N * 2), fx.Int32(OOB_OFFSET))
                    )
                rows_rsrc = _rsrc_of(pOut, num_blocks * (SORT_BLOCK_M * H * 2))
                lane_group = lane // 16
                odd_group = lane_group % 2 == 1
                low_rows = lane % 16 < 8
                exchange_lane = fx.Int32((lane ^ 16) * 4)
                # Rows r < 8 store columns [0, 64) B of the group, rows r >= 8 columns [64, 128) B of row r ^ 8.
                column_bytes = (lane_group % 2 * 2 + lane_group // 2) * 16 + wave_n * wave_bytes
                column_bytes = column_bytes + low_rows.select(fx.Int32(0), fx.Int32(64))
                finish_k_loop()

                def row_swap(value):
                    # DPP row_ror:8 within 16 lanes = lane ^ 8: the same columns of row r ^ 8.
                    return fx.Int32(rocdl.update_dpp(T.i32, value.ir_value(), value.ir_value(), 0x128, 0xF, 0xF, True))

                def column_pair(m_slice, first, weight):
                    """16 B per lane: 8 consecutive BF16 columns of N slice `first` (even lane groups) or first + 1."""
                    left, right = [weighted_words(m_slice, first + side, weight) for side in range_constexpr(2)]
                    # Even 16-lane groups keep `first` and take the partner's half; odd groups do first + 1.
                    exchanged = [
                        fx.Int32(rocdl.ds_bpermute(T.i32, exchange_lane.ir_value(), odd_group.select(left[pair], right[pair]).ir_value()))
                        for pair in range_constexpr(2)
                    ]
                    return [odd_group.select(exchanged[pair], left[pair]) for pair in range_constexpr(2)] + [
                        odd_group.select(right[pair], exchanged[pair]) for pair in range_constexpr(2)
                    ]

                for j in range_constexpr(m_slices):
                    partner = row_swap(row_offsets_out[j])
                    offsets = [low_rows.select(row_offsets_out[j], partner), low_rows.select(partner, row_offsets_out[j])]
                    for c in range_constexpr(col_groups):
                        first, second = column_pair(j, c * 4, weights[j]), column_pair(j, c * 4 + 2, weights[j])
                        words = [
                            [low_rows.select(first[i], row_swap(second[i])) for i in range_constexpr(4)],
                            [low_rows.select(row_swap(first[i]), second[i]) for i in range_constexpr(4)],
                        ]
                        for dwords, offset in zip(words, offsets):
                            rocdl.raw_ptr_buffer_store(
                                Vec.from_elements(dwords, fx.Int32).ir_value(),
                                rows_rsrc,
                                (offset + column_bytes + c * 128).ir_value(),
                                zero,
                                aux=ir.IntegerAttr.get(T.i32, 0),
                            )

                def write_inverse():
                    inverse_row = tid % SORT_BLOCK_M
                    inverse_a_row, _, inverse_valid = decode(load_i32(sorted_rsrc, (block * SORT_BLOCK_M + inverse_row) * 4))
                    inverse_valid = inverse_valid & (tid < SORT_BLOCK_M)
                    rocdl.raw_ptr_buffer_store(
                        (block * SORT_BLOCK_M + inverse_row).ir_value(),
                        _rsrc_of(pInverse, route_rows * 4),
                        inverse_valid.select(inverse_a_row * 4, fx.Int32(OOB_OFFSET)).ir_value(),
                        zero,
                        aux=ir.IntegerAttr.get(T.i32, 0),
                    )

                if n_tile == 0:
                    write_inverse()

    @flyc.jit
    def launch(
        pA: fx.Pointer,
        pB: fx.Pointer,
        pOut: fx.Pointer,
        pScaleA: fx.Pointer,
        pScaleB: fx.Pointer,
        pSortedIds: fx.Pointer,
        pSortedWeights: fx.Pointer,
        pExpertIds: fx.Pointer,
        pNumValid: fx.Pointer,
        pInverse: fx.Pointer,
        num_tokens: fx.Int32,
        num_blocks: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        down_kernel(
            pA, pB, pOut, pScaleA, pScaleB, pSortedIds, pSortedWeights, pExpertIds, pNumValid, pInverse, num_tokens, num_blocks,
            value_attrs={"llvm.passthrough": [["target-features", "-packed-fp32-ops"]]},
        ).launch(
            grid=((num_blocks * n_tiles + 7) // 8 * 8, 1, 1),
            block=(512, 1, 1),
            stream=stream,
        )

    launch.compile_hints["target_features"] = "-packed-fp32-ops"
    return launch


@functools.cache
def compile_moe_down_reduce(hidden_dim, topk):
    """out[token] = BF16(sum over slots of rows[inverse[token, slot]]) in FP32 slot order; -1 = no route."""
    H = hidden_dim
    threads = H // REDUCE_COLUMNS
    # Thread t sums columns [8 t, 8 t + 8) and the same of the right half: 16 B loads, contiguous across lanes.
    half_bytes = H

    @flyc.kernel(known_block_size=[threads, 1, 1])
    def reduce_kernel(pRows: fx.Pointer, pInverse: fx.Pointer, pOut: fx.Pointer, num_tokens: fx.Int32, num_rows: fx.Int32):
        token = fx.Int32(fx.block_idx.x)
        tid = fx.thread_idx.x
        zero = fx.Int32(0).ir_value()
        rows_tensor = fx.rocdl.make_buffer_tensor(
            fx.make_view(pRows, fx.make_layout(1, 1)), max_size=False, num_records_bytes=num_rows * (H * 2)
        )
        rows = BufferTensor(fx.make_view(fx.get_iter(rows_tensor), fx.make_layout(4, 1)))
        inverse = fx.get_iter(
            fx.rocdl.make_buffer_tensor(
                fx.make_view(pInverse, fx.make_layout(1, 1)), max_size=False, num_records_bytes=num_tokens * (topk * 4)
            )
        )
        out_rsrc = _rsrc_of(pOut, num_tokens * (H * 2))
        acc = fx.make_rmem_tensor(fx.make_layout((8, 2), (1, 8)), fx.Float32)
        acc.fill(0)
        for slot in range_constexpr(topk):
            position = fx.Int32(rocdl.s_buffer_load(inverse, token * topk + slot))
            if position >= 0:
                for half in range_constexpr(2):
                    words = Vec(rows.load(voffset_bytes=tid * 16 + half * half_bytes, soffset_bytes=position * (H * 2)))
                    old = Vec(acc[None, half].load())
                    values = []
                    for word in range_constexpr(4):
                        values.append(old[word * 2] + (words[word] << 16).bitcast(fx.Float32))
                        values.append(old[word * 2 + 1] + (words[word] & -65536).bitcast(fx.Float32))
                    acc[None, half].store(Vec.from_elements(values, fx.Float32))
        for half in range_constexpr(2):
            total = Vec(acc[None, half].load())
            dwords = pack_bf16x2_rne([total[i] for i in range(4)]) + pack_bf16x2_rne([total[i] for i in range(4, 8)])
            rocdl.raw_ptr_buffer_store(
                Vec.from_elements(dwords, fx.Int32).ir_value(),
                out_rsrc,
                (tid * 16 + half * half_bytes).ir_value(),
                (token * (H * 2)).ir_value(),
                aux=ir.IntegerAttr.get(T.i32, 0),
            )

    @flyc.jit
    def launch(
        pRows: fx.Pointer,
        pInverse: fx.Pointer,
        pOut: fx.Pointer,
        num_tokens: fx.Int32,
        num_rows: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        reduce_kernel(pRows, pInverse, pOut, num_tokens, num_rows).launch(
            grid=(num_tokens, 1, 1), block=(threads, 1, 1), stream=stream
        )

    return launch


def moe_down_reduce(rows, inverse, out):
    """out [tokens, H] = sum of each token's rows (inverse [tokens, topk] = row index or -1)."""
    tokens, hidden_dim = out.shape
    topk = inverse.shape[1]
    compile_moe_down_reduce(hidden_dim, topk)(
        flyc.from_c_void_p(fx.Int32, rows.data_ptr()),
        flyc.from_c_void_p(fx.Int32, inverse.data_ptr()),
        flyc.from_c_void_p(fx.Int32, out.data_ptr()),
        tokens,
        rows.shape[0],
        torch.cuda.current_stream(),
    )
    return out


def _launch_down(
    a2, w2, a2_scale_t, w2_scale, sorted_ids, sorted_weights, expert_ids, num_valid_ids, target, inverse, layout, xcd_swizzle,
    atomic, tokens, topk,
):
    sorted_a2 = a2.dim() == 2
    inter_dim = a2.shape[-1]
    num_experts, hidden_dim, _ = w2.shape
    num_blocks = expert_ids.numel()
    capacity = num_blocks * SORT_BLOCK_M
    validate_parameters(tokens, hidden_dim, inter_dim, topk, num_experts, capacity, sorted_a2)
    assert sorted_ids.numel() >= capacity and sorted_weights.numel() >= capacity
    assert a2.shape[0] >= capacity if sorted_a2 else a2.shape[:2] == (tokens, topk)
    launch = compile_moe_down_a8w8_blockscale_gfx942_lds(
        hidden_dim, inter_dim, topk, num_experts, layout, bool(xcd_swizzle), atomic, sorted_a2
    )
    launch(
        flyc.from_c_void_p(fx.Int32, a2.data_ptr()),
        flyc.from_c_void_p(fx.Int32, w2.data_ptr()),
        flyc.from_c_void_p(fx.Int32, target.data_ptr()),
        flyc.from_c_void_p(fx.Float32, a2_scale_t.data_ptr()),
        flyc.from_c_void_p(fx.Float32, w2_scale.data_ptr()),
        flyc.from_c_void_p(fx.Int32, sorted_ids.data_ptr()),
        flyc.from_c_void_p(fx.Float32, sorted_weights.data_ptr()),
        flyc.from_c_void_p(fx.Int32, expert_ids.data_ptr()),
        flyc.from_c_void_p(fx.Int32, num_valid_ids.data_ptr()),
        flyc.from_c_void_p(fx.Int32, inverse.data_ptr()),
        tokens,
        num_blocks,
        torch.cuda.current_stream(),
    )


def moe_down_rows_a8w8_blockscale_gfx942_lds(
    a2, w2, a2_scale_t, w2_scale, sorted_ids, sorted_weights, expert_ids, num_valid_ids, rows, inverse,
    layout="2x4", xcd_swizzle=True,
):
    """Down kernel only: rows [blocks * 256, H] BF16 per sorted position, inverse [tokens, topk] (refilled)."""
    assert rows.shape[0] >= expert_ids.numel() * SORT_BLOCK_M
    inverse.fill_(-1)
    _launch_down(
        a2, w2, a2_scale_t, w2_scale, sorted_ids, sorted_weights, expert_ids, num_valid_ids, rows, inverse, layout, xcd_swizzle,
        False, *inverse.shape,
    )
    return rows, inverse


def moe_down_a8w8_blockscale_gfx942_lds(
    a2, w2, a2_scale_t, w2_scale, sorted_ids, sorted_weights, expert_ids, num_valid_ids, out,
    layout="2x4", xcd_swizzle=True, reduce="atomic", zero_out=True, rows=None, inverse=None, topk=None,
):
    """out [tokens, H] BF16 = sum over each token's routes of routing weight * (a2 row @ w2[expert]^T).

    a2 [tokens, topk, I] FP8 with a2_scale_t [I/128, tokens * topk], or sorted-row a2 [blocks * 256, I] with
    a2_scale_t [I/128, blocks * 256] (then pass topk); w2 shuffle_weight(16,16) of [E, H, I] FP8,
    w2_scale [E, H/128, I/128], sorted_weights [blocks * 256] FP32 alongside sorted_ids.
    reduce="atomic": the Down kernel adds every route into out with BF16 atomics (zero_out=False keeps out's
    contents). reduce="kernel": the Down kernel writes rows and inverse (optional preallocated buffers),
    then moe_down_reduce writes out.
    """
    assert reduce in ("atomic", "kernel"), reduce
    topk = a2.shape[1] if a2.dim() == 3 else topk
    assert topk is not None, "pass topk with a sorted-row a2"
    if reduce == "atomic":
        if zero_out:
            out.zero_()
        _launch_down(
            a2, w2, a2_scale_t, w2_scale, sorted_ids, sorted_weights, expert_ids, num_valid_ids, out, out, layout, xcd_swizzle,
            True, out.shape[0], topk,
        )
        return out
    if rows is None:
        rows = torch.empty((expert_ids.numel() * SORT_BLOCK_M, out.shape[1]), device=out.device, dtype=torch.bfloat16)
    if inverse is None:
        inverse = torch.empty((out.shape[0], topk), device=out.device, dtype=torch.int32)
    moe_down_rows_a8w8_blockscale_gfx942_lds(
        a2, w2, a2_scale_t, w2_scale, sorted_ids, sorted_weights, expert_ids, num_valid_ids, rows, inverse, layout, xcd_swizzle
    )
    return moe_down_reduce(rows, inverse, out)
