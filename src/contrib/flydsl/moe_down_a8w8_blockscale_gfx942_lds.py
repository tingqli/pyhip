# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx942 FP8 block-scale MoE Down (GEMM2, BM256 x BN256) on a 4x2 wave grid, A and B staged through LDS.

out[token] += routing weight * (A2 row @ W2[expert]^T) for every route, added in BF16 with
global_atomic_pk_add_bf16 into out [tokens, H] (zeroed beforehand). A2 is the sorted-row FP8 output of
moe_gateup_a8w8_blockscale_gfx942_lds, [blocks * 256, I] with FP32 scales [I/128, blocks * 256]; W2 is aiter
shuffle_weight(16,16) of [E, H, I] with FP32 scales [E, H/128, I/128]. One workgroup takes a 256-row sorted block
and 256 hidden columns: 4 waves along M (64 rows) x 2 along N (128 columns). The K loop is the one of
moe_gateup_a8w8_blockscale_gfx942_lds with the operands swapped: A is read for the whole K block at stage 0
("held"), B streams one 64-row packet per stage. Each wave transposes its tile through LDS so that one atomic
covers two rows x 128 contiguous bytes. With single_store, rows of tokens with one local route (route_counts from
compile_moe_route_counts) are plain stores: nothing else adds to them.
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

from .gemm_a8w8_blockscale_gfx942_common import pack_bf16x2_rne
from .helpers import BufferTensor, run_compiled
from .moe_gateup_a8w8_blockscale_gfx942_lds import (
    OOB_OFFSET, PACKET_DWORDS, PACKET_ROWS, SMALL_M_ROWS, SORT_BLOCK_M, TILE_K, TOKEN_MASK, _lds_dwords,
)

TILE_N = 256
# Atomic epilogue transpose tile per wave: 16 rows x 32 dwords; a 34-dword pitch keeps writes and reads conflict free.
TRANSPOSE_PITCH = 34


def validate_parameters(tokens, hidden_dim, inter_dim, topk, num_experts, capacity=SORT_BLOCK_M):
    """capacity: sorted rows of A2 (blocks * 256)."""
    if not 0 < tokens <= TOKEN_MASK:
        raise ValueError(f"tokens must be in [1, {TOKEN_MASK}]")
    if hidden_dim <= 0 or hidden_dim % TILE_N:
        raise ValueError("hidden_dim must be a positive multiple of 256")
    if inter_dim < TILE_K or inter_dim % TILE_K or inter_dim > 64 * TILE_K:
        raise ValueError("inter_dim must be a multiple of 128 in [128, 8192]")
    if not 0 < topk < 256 or num_experts <= 0:
        raise ValueError("require 0 < topk < 256 and num_experts > 0")
    # Descriptor byte counts and every offset must stay below OOB_OFFSET.
    extents = {
        "a2": capacity * inter_dim,
        "a2_scale": capacity * (inter_dim // TILE_K) * 4,
        "weight tile": TILE_N * inter_dim,
        "output": tokens * hidden_dim * 2,
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
def compile_moe_down_a8w8_blockscale_gfx942_lds(hidden_dim, inter_dim, topk, num_experts, single_store=False, merged=None, bm128=False):
    """single_store: rows of tokens whose route count (pRouteCounts) is 1 are plain stores instead of atomics.
    bm128: a block with no route at row 128 runs a 128-row pipeline in the same workgroup (A rows 128..255 neither
    loaded nor multiplied). Same grid and task order as without it.
    merged: plugin that maps workgroups to tasks itself (merged.task) and runs the blocks with no route at row 128
    in the same launch (merged.body); see docs/record/moe_small_m_gfx942.md. merged.key (str) must change with the
    plugin's code: the FlyDSL kernel cache keys on closure scalars and on sources next to this file only."""
    validate_parameters(1, hidden_dim, inter_dim, topk, num_experts)
    assert merged is None or not (single_store or bm128)
    merged_key = None if merged is None else merged.key
    assert merged_key is None or isinstance(merged_key, str)
    H, K = hidden_dim, inter_dim
    KB = K // TILE_K
    n_tiles = H // TILE_N
    b_k_bytes = TILE_K * 16
    scale_rows_b = H // 128
    # 16-row M / N slices per wave.
    m_slices = 4
    n_slices = 8
    # 64-column groups of a wave: one LDS transpose each.
    col_groups = 2
    wave_bytes = n_slices * 32

    def acc_index(m_slice, n_slice):
        return m_slice * n_slices + n_slice

    def weight_group(packet, loader):
        """16-row group of the N tile held in LDS rows [16 loader, 16 loader + 16) of B packet `packet`.

        Waves read N slice s from packet s // 2, rows [32 wave_n + 16 (s % 2), +16): a wave's N slices are its
        contiguous columns.
        """
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
        pRouteCounts: fx.Pointer,
        num_tokens: fx.Int32,
        num_blocks: fx.Int32,
    ):
        tid = fx.thread_idx.x
        lane = tid % 64
        wave = tid // 64
        group = fx.Int32(rocdl.readfirstlane(T.i32, (tid // 256).ir_value()))
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
            xcd_tasks = (num_tasks + 7) >> 3
            xcd_slot = pid >> 3
            task = (pid & 7) * xcd_tasks + xcd_slot
            active = (xcd_slot < xcd_tasks) & (task < num_tasks)

        def big_body(tile_rows=SORT_BLOCK_M):
            """The BM256 task `task`; tile_rows=128: the 128-row pipeline (bm128)."""
            # A packets held per K block; each gives every wave one 16-row M slice.
            held_packets = tile_rows // PACKET_ROWS
            zero = fx.Int32(0).ir_value()
            sorted_rsrc = _rsrc_of(pSortedIds, num_blocks * (SORT_BLOCK_M * 4))
            weight_rsrc = _rsrc_of(pSortedWeights, num_blocks * (SORT_BLOCK_M * 4))
            a2_rows = num_blocks * SORT_BLOCK_M
            block = task // n_tiles
            n_tile = task - block * n_tiles
            expert = uniform(load_i32(_rsrc_of(pExpertIds, num_blocks * 4), block * 4))

            def route_ids(rows):
                return [load_i32(sorted_rsrc, (block * SORT_BLOCK_M + row) * 4) for row in rows]

            def decode(fused):
                """(token, valid) of a sorted ID."""
                token = fused & TOKEN_MASK
                slot = (fused >> 24) & 255
                return token, (token < num_tokens) & (slot < topk)

            # A slice j covers block rows slice_rows[j] + lane % 16, in LDS packet j.
            slice_rows = [j * PACKET_ROWS + wave_m * 16 for j in range_constexpr(m_slices)]
            lane_rows = [slice_rows[j] + lane % 16 for j in range_constexpr(held_packets)]
            copy_row = tid // 8
            copy_col = tid % 8

            # A2 row of sorted row r: block * 256 + r.
            a_offsets, sa_offsets = [], []
            copy_rows = [copy_row + copy_packet * PACKET_ROWS for copy_packet in range_constexpr(held_packets)]
            for row, fused in zip(copy_rows, route_ids(copy_rows)):
                _, a_valid = decode(fused)
                a_offsets.append(a_valid.select((block * SORT_BLOCK_M + row) * K + copy_col * 16, fx.Int32(OOB_OFFSET)))
            for row, fused in zip(lane_rows, route_ids(lane_rows)):
                _, sa_valid = decode(fused)
                sa_offsets.append(sa_valid.select((block * SORT_BLOCK_M + row) * 4, fx.Int32(OOB_OFFSET)))

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
            scale_group = n_tile * 2 + wave_n
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
            a_read = [_lds_dwords(wave_m * 16 + lane % 16, lane // 16 + h * 4) for h in range(2)]
            b_read = [[_lds_dwords(wave_n * 32 + r * 16 + lane % 16, lane // 16 + h * 4) for h in range(2)] for r in range(2)]

            def lds_view(dwords, count=4):
                return fx.make_view(lds.ab.ptr + dwords, fx.make_layout(count, 1))

            def read_a_slice(j, half_k):
                return lds_view(a_read[half_k] + j * PACKET_DWORDS).load()

            def read_b_slice(n_slice, half_k):
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

            # B stages all four packets of the next block, A two slots.
            s_stage = fx.make_rmem_tensor(fx.make_layout((4, 4), (1, 4)), fx.Int32)
            h_stage = fx.make_rmem_tensor(fx.make_layout((4, 2), (1, 4)), fx.Int32)
            # held column M slice * 2 + half_k for the held A slices of the current block.
            held = fx.make_rmem_tensor(fx.make_layout((4, 8), (1, 4)), fx.Int32)
            # scale_a column (block parity) * 4 + M slice.
            scale_a = fx.make_rmem_tensor(fx.make_layout((1, 8), (1, 1)), fx.Float32)
            accum = fx.make_rmem_tensor(fx.make_layout((4, 32), (1, 4)), fx.Float32)
            partial = fx.make_rmem_tensor(fx.make_layout((4, 4), (1, 4)), fx.Float32)
            partial_scale = fx.make_rmem_tensor(fx.make_layout((1, 2), (1, 1)), fx.Float32)

            def load_s(k_block, packet):
                s_stage[None, packet].store(load_b(k_block, packet))

            def load_h(k_block, packet, slot):
                h_stage[None, slot].store(load_a(k_block, packet))

            def write_s(packet):
                lds_view(b_write + (B_BASE + packet * PACKET_DWORDS)).store(s_stage[None, packet].load())

            def write_h(packet, value):
                lds_view(a_write + packet * PACKET_DWORDS).store(value)

            def read_held():
                for held_index in range_constexpr(held_packets):
                    for held_half in range_constexpr(2):
                        held[None, held_index * 2 + held_half].store(read_a_slice(held_index, held_half))

            def held_word(index, half_k, word):
                return Vec(held[None, index * 2 + half_k].load()).bitcast(fx.Int64)[word]

            def stage_pairs(stage):
                """(m_slice, [n_slice, n_slice]) of the dequant pairs of a stage, in issue order."""
                return [(m_slice, [stage * 2, stage * 2 + 1]) for m_slice in range(held_packets)]

            last_pair = stage_pairs(3)[-1]

            def run_k(parity, k_block):
                # The last block re-fetches clamped blocks so it stays inside the loop body.
                k_next = (k_block + 1 < KB).select(k_block + 1, KB - 1)
                previous = last_pair
                for stage in range_constexpr(4):
                    if const_expr(stage == 0):
                        write_s(3)
                    else:
                        write_s(stage - 1)
                        if const_expr(stage - 1 < held_packets):
                            write_h(stage - 1, h_stage[None, (stage - 1) % 2].load())
                        if const_expr(stage == 3 and held_packets == 4):
                            write_h(3, h_stage[None, 1].load())
                    rocdl.sched_barrier(0)
                    if const_expr(stage == 0):
                        read_held()
                    # Streamed B slices 2s, 2s+1 of this stage.
                    streamed = [
                        [Vec(read_b_slice(stage * 2 + i, h)).bitcast(fx.Int64) for h in range_constexpr(2)]
                        for i in range_constexpr(2)
                    ]
                    rocdl.sched_barrier(0)
                    if const_expr(stage == 0):
                        for next_j in range_constexpr(held_packets):
                            scale_a[None, (1 - parity) * 4 + next_j].store(load_scale_a(k_next, next_j))
                    load_s(k_next, stage)
                    if const_expr(stage < min(3, held_packets)):
                        load_h(k_next, stage, stage % 2)
                    if const_expr(stage == 2 and held_packets == 4):
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
                                a_value = held_word(m_slice, half_k, word)
                                b_value = streamed[inner][half_k][word]
                                accumulator = Vec.filled(4, 0.0, fx.Float32) if k_step == 0 else Vec(partial[None, base + inner].load())
                                partial[None, base + inner].store(
                                    Vec(rocdl.mfma_f32_16x16x32_fp8_fp8(T.f32x4, [b_value, a_value, accumulator, 0, 0, 0]))
                                )
                                # Keep MFMA -> FMA order: the FMA's partial then has two MFMAs of XDL latency cover.
                                rocdl.sched_barrier(0)
                                if const_expr(mfma_index == 0):
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
            for pair_begin in range_constexpr(0, held_packets, 2):
                for pro_packet in range_constexpr(pair_begin, pair_begin + 2):
                    load_h(0, pro_packet, pro_packet % 2)
                for pro_packet in range_constexpr(pair_begin, pair_begin + 2):
                    write_h(pro_packet, h_stage[None, pro_packet % 2].load())
                rocdl.sched_barrier(0)
            for pro_slice in range_constexpr(held_packets):
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
            token_offsets = []
            for fused in out_ids:
                out_token, out_valid = decode(fused)
                row_offset = out_token * (H * 2)
                if const_expr(single_store):
                    # Bit 0 marks the token's only local route: a plain store, no other route adds to it.
                    routes = load_i32(
                        _rsrc_of(pRouteCounts, num_tokens * 4), out_valid.select(out_token * 4, fx.Int32(OOB_OFFSET))
                    )
                    row_offset = row_offset | (routes == 1).select(fx.Int32(1), fx.Int32(0))
                token_offsets.append(out_valid.select(row_offset, fx.Int32(-1)))
            out_base = fx.Int64(fx.ptrtoint(pOut)).ir_value()
            transpose = wave * (16 * TRANSPOSE_PITCH)
            write_base = transpose + lane % 16 * TRANSPOSE_PITCH + lane // 16 * 2
            read_base = transpose + lane // 32 * TRANSPOSE_PITCH + lane % 32
            column_base = n_tile * (TILE_N * 2) + wave_n * wave_bytes + lane % 32 * 4
            finish_k_loop()
            # LLVM does not count the asm atomics: retire every load first so no later vmcnt wait covers them.
            rocdl.s_waitcnt(vmcnt=0)
            for j in range_constexpr(held_packets):
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
                        if const_expr(single_store):
                            address = ((row_offset & -2) + column_base + c * 128).ir_value()
                            if (row_offset >= 0) & ((row_offset & 1) == 1):
                                llvm.inline_asm(
                                    None, [address, value.ir_value(), out_base], "global_store_dword $0, $1, $2", "v,v,s",
                                    has_side_effects=True,
                                )
                            if (row_offset >= 0) & ((row_offset & 1) == 0):
                                llvm.inline_asm(
                                    None, [address, value.ir_value(), out_base], "global_atomic_pk_add_bf16 $0, $1, $2", "v,v,s",
                                    has_side_effects=True,
                                )
                        else:
                            if row_offset >= 0:
                                llvm.inline_asm(
                                    None,
                                    [(row_offset + column_base + c * 128).ir_value(), value.ir_value(), out_base],
                                    "global_atomic_pk_add_bf16 $0, $1, $2",
                                    "v,v,s",
                                    has_side_effects=True,
                                )

        def small_ctx():
            return SimpleNamespace(
                pA=pA, pB=pB, pOut=pOut, pScaleOut=pScaleA, pScaleA=pScaleA, pScaleB=pScaleB, pSortedIds=pSortedIds,
                pSortedWeights=pSortedWeights, pExpertIds=pExpertIds, num_tokens=num_tokens, num_blocks=num_blocks, tid=tid,
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
                bottom_id = uniform(load_i32(_rsrc_of(pSortedIds, num_blocks * (SORT_BLOCK_M * 4)), ((task // n_tiles) * SORT_BLOCK_M + SMALL_M_ROWS) * 4))
                if ((bottom_id & TOKEN_MASK) < num_tokens) & (((bottom_id >> 24) & 255) < topk):
                    big_body()
                    # Distinct last instructions: SimplifyCFG must not sink the two pipelines' common tail into one
                    # block (phis of the atomics' SGPR base miscompile).
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
        pOut: fx.Pointer,
        pScaleA: fx.Pointer,
        pScaleB: fx.Pointer,
        pSortedIds: fx.Pointer,
        pSortedWeights: fx.Pointer,
        pExpertIds: fx.Pointer,
        pNumValid: fx.Pointer,
        pRouteCounts: fx.Pointer,
        num_tokens: fx.Int32,
        num_blocks: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        if const_expr(merged_key is not None):
            grid_x = merged.grid(num_blocks)
        else:
            grid_x = (num_blocks * n_tiles + 7) // 8 * 8
        down_kernel(
            pA, pB, pOut, pScaleA, pScaleB, pSortedIds, pSortedWeights, pExpertIds, pNumValid, pRouteCounts,
            num_tokens, num_blocks,
            value_attrs={"llvm.passthrough": [["target-features", "-packed-fp32-ops"]]},
        ).launch(
            grid=(grid_x, 1, 1),
            block=(512, 1, 1),
            stream=stream,
        )

    launch.compile_hints["target_features"] = "-packed-fp32-ops"
    return launch


@functools.cache
def compile_moe_route_counts(topk):
    """counts[token] += 1 for every valid sorted ID (counts zeroed by the caller): local routes per token."""

    @flyc.kernel(known_block_size=[SORT_BLOCK_M, 1, 1])
    def count_kernel(pSortedIds: fx.Pointer, pNumValid: fx.Pointer, pCounts: fx.Pointer, num_tokens: fx.Int32, num_rows: fx.Int32):
        zero = fx.Int32(0).ir_value()
        position = fx.Int32(fx.block_idx.x) * SORT_BLOCK_M + fx.thread_idx.x
        valid_rows = fx.Int32(rocdl.raw_ptr_buffer_load(T.i32, _rsrc_of(pNumValid, 4), zero, zero, zero))
        fused = fx.Int32(
            rocdl.raw_ptr_buffer_load(T.i32, _rsrc_of(pSortedIds, num_rows * 4), (position * 4).ir_value(), zero, zero)
        )
        token = fused & TOKEN_MASK
        slot = (fused >> 24) & 255
        if (token < num_tokens) & (slot < topk) & (position < valid_rows):
            llvm.inline_asm(
                None,
                [(token * 4).ir_value(), fx.Int32(1).ir_value(), fx.Int64(fx.ptrtoint(pCounts)).ir_value()],
                "global_atomic_add $0, $1, $2",
                "v,v,s",
                has_side_effects=True,
            )

    @flyc.jit
    def launch(
        pSortedIds: fx.Pointer,
        pNumValid: fx.Pointer,
        pCounts: fx.Pointer,
        num_tokens: fx.Int32,
        num_rows: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        count_kernel(pSortedIds, pNumValid, pCounts, num_tokens, num_rows).launch(
            grid=((num_rows + SORT_BLOCK_M - 1) // SORT_BLOCK_M, 1, 1), block=(SORT_BLOCK_M, 1, 1), stream=stream
        )

    return launch


def moe_route_counts(sorted_ids, num_valid_ids, expert_ids, counts, topk, block_m=SORT_BLOCK_M):
    """counts [tokens] int32 = number of local routes of each token (refilled); block_m: rows per sorted block."""
    counts.zero_()
    run_compiled(
        compile_moe_route_counts(topk),
        flyc.from_c_void_p(fx.Int32, sorted_ids.data_ptr()),
        flyc.from_c_void_p(fx.Int32, num_valid_ids.data_ptr()),
        flyc.from_c_void_p(fx.Int32, counts.data_ptr()),
        counts.numel(),
        expert_ids.numel() * block_m,
        torch.cuda.current_stream(),
    )
    return counts


@functools.cache
def compile_moe_zero_rows(hidden_dim, bounded=False):
    """out[token] = 0 unless counts[token] == 1 (those rows are stored by the single_store Down kernel).

    bounded: only tokens below num_rows[0] (device int32, e.g. an EP receive count) are cleared.
    """
    H = hidden_dim
    chunks = H // 8
    # 4 tokens per wave, 16 per workgroup: one workgroup per token is dispatch bound (~23 us at 32K tokens).
    tokens_per_wave = 4

    @flyc.kernel(known_block_size=[256, 1, 1])
    def zero_kernel(pCounts: fx.Pointer, pOut: fx.Pointer, pNumRows: fx.Pointer, num_tokens: fx.Int32):
        wave = fx.Int32(rocdl.readfirstlane(T.i32, (fx.thread_idx.x // 64).ir_value()))
        lane = fx.thread_idx.x % 64
        counts = fx.get_iter(
            fx.rocdl.make_buffer_tensor(fx.make_view(pCounts, fx.make_layout(1, 1)), max_size=False, num_records_bytes=num_tokens * 4)
        )
        limit = num_tokens
        if const_expr(bounded):
            num_rows = fx.get_iter(
                fx.rocdl.make_buffer_tensor(fx.make_view(pNumRows, fx.make_layout(1, 1)), max_size=False, num_records_bytes=4)
            )
            limit = fx.Int32(rocdl.s_buffer_load(num_rows, fx.Int32(0)))
        out_rsrc = _rsrc_of(pOut, num_tokens * (H * 2))
        for slot in range_constexpr(tokens_per_wave):
            token = (fx.Int32(fx.block_idx.x) * 4 + wave) * tokens_per_wave + slot
            # Past num_tokens the count reads 0 and the stores fall out of the descriptor.
            if (fx.Int32(rocdl.s_buffer_load(counts, token)) != 1) & (token < limit):
                for part in range_constexpr((chunks + 63) // 64):
                    chunk = lane + part * 64
                    rocdl.raw_ptr_buffer_store(
                        Vec.filled(4, 0, fx.Int32).ir_value(),
                        out_rsrc,
                        (chunk < chunks).select(chunk * 16, fx.Int32(OOB_OFFSET)).ir_value(),
                        (token * (H * 2)).ir_value(),
                        aux=ir.IntegerAttr.get(T.i32, 0),
                    )

    @flyc.jit
    def launch(
        pCounts: fx.Pointer, pOut: fx.Pointer, pNumRows: fx.Pointer, num_tokens: fx.Int32, stream: fx.Stream = fx.Stream(None)
    ):
        zero_kernel(pCounts, pOut, pNumRows, num_tokens).launch(
            grid=((num_tokens + 4 * tokens_per_wave - 1) // (4 * tokens_per_wave), 1, 1), block=(256, 1, 1), stream=stream
        )

    return launch


def moe_zero_rows(out, counts, num_rows=None):
    """Zero the rows of out [tokens, H] whose token does not have exactly one local route.

    num_rows: optional device int32 tensor; rows from num_rows[0] on are left untouched.
    """
    run_compiled(
        compile_moe_zero_rows(out.shape[1], num_rows is not None),
        flyc.from_c_void_p(fx.Int32, counts.data_ptr()),
        flyc.from_c_void_p(fx.Int32, out.data_ptr()),
        flyc.from_c_void_p(fx.Int32, (counts if num_rows is None else num_rows).data_ptr()),
        out.shape[0],
        torch.cuda.current_stream(),
    )
    return out


def moe_down_a8w8_blockscale_gfx942_lds(
    a2, w2, a2_scale_t, w2_scale, sorted_ids, sorted_weights, expert_ids, num_valid_ids, out, topk,
    zero_out=True, single_store=False, route_counts=None, bm128=False,
):
    """out [tokens, H] BF16 += routing weight * (a2 row @ w2[expert]^T) for every route, with BF16 atomics.

    a2 [blocks * 256, I] FP8 sorted-row (moe_gateup_a8w8_blockscale_gfx942_lds output) with a2_scale_t
    [I/128, blocks * 256]; w2 shuffle_weight(16,16) of [E, H, I] FP8, w2_scale [E, H/128, I/128], sorted_weights
    [blocks * 256] FP32 alongside sorted_ids. zero_out=False keeps out's contents. single_store=True stores the rows
    of single-route tokens instead (route_counts: precomputed moe_route_counts, else computed here) and zero_out
    then clears only the other rows. bm128: blocks with at most 128 routes run the 128-row pipeline.
    """
    tokens = out.shape[0]
    if single_store and route_counts is None:
        route_counts = torch.empty(tokens, device=out.device, dtype=torch.int32)
        moe_route_counts(sorted_ids, num_valid_ids, expert_ids, route_counts, topk)
    if zero_out:
        if single_store:
            moe_zero_rows(out, route_counts)
        else:
            out.zero_()
    inter_dim = a2.shape[1]
    num_experts, hidden_dim, _ = w2.shape
    num_blocks = expert_ids.numel()
    capacity = num_blocks * SORT_BLOCK_M
    validate_parameters(tokens, hidden_dim, inter_dim, topk, num_experts, capacity)
    assert sorted_ids.numel() >= capacity and sorted_weights.numel() >= capacity and a2.shape[0] >= capacity
    assert not single_store or route_counts.shape == (tokens,) and route_counts.dtype == torch.int32
    launch = compile_moe_down_a8w8_blockscale_gfx942_lds(
        hidden_dim, inter_dim, topk, num_experts, bool(single_store), bm128=bool(bm128)
    )
    run_compiled(
        launch,
        flyc.from_c_void_p(fx.Int32, a2.data_ptr()),
        flyc.from_c_void_p(fx.Int32, w2.data_ptr()),
        flyc.from_c_void_p(fx.Int32, out.data_ptr()),
        flyc.from_c_void_p(fx.Float32, a2_scale_t.data_ptr()),
        flyc.from_c_void_p(fx.Float32, w2_scale.data_ptr()),
        flyc.from_c_void_p(fx.Int32, sorted_ids.data_ptr()),
        flyc.from_c_void_p(fx.Float32, sorted_weights.data_ptr()),
        flyc.from_c_void_p(fx.Int32, expert_ids.data_ptr()),
        flyc.from_c_void_p(fx.Int32, num_valid_ids.data_ptr()),
        flyc.from_c_void_p(fx.Int32, (route_counts if single_store else out).data_ptr()),
        tokens,
        num_blocks,
        torch.cuda.current_stream(),
    )
    return out
