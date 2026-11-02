# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx942 FP8 block-scale MoE GateUp (BM256) with BF16 routed SiLU(gate) * up output.

Built on the 8x1 GEMM core (gemm_a8w8_blockscale_gfx942_8x1.py). One workgroup takes one
256-row sorted block and 128 output columns; 8 waves stack along M. A rows are gathered by
sorted ID straight into VGPRs. B is aiter shuffle_weight(16,16) of [E, 2I, K] (gate rows, then
up rows) and streams through a four-slot N64xK128 LDS ring as packets gate0, gate1, up0, up1,
so each lane holds matching gate/up columns. Every per-lane offset is fixed per workgroup: the
K loop only advances scalar soffsets. Scales are FP32: A [K/128, tokens], B [E, 2I/128, K/128].
Sorted IDs are (slot << 24) | token; padding rows use an invalid token or slot. Blocks past
ceil(num_valid_ids[0] / 256) are never read, so sort buffers may keep spare capacity.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir import ir
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from .gemm_a8w8_blockscale_gfx942_8x1 import (
    A_STAGES,
    MFMA_ORDER,
    PACKET_DWORDS,
    PACKET_N,
    TILE_K,
    _lds_dwords,
)
from .gemm_a8w8_blockscale_gfx942_common import pack_bf16x2, s_buffer_load_f32
from .helpers import BufferTensor

SORT_BLOCK_M = 256
TILE_N = 128
TOKEN_MASK = 0xFFFFFF
# Past every descriptor used here; stays out of range after adding a tile-local byte offset.
OOB_OFFSET = 0x7FFF0000
LOG2E = 1.4426950408889634


# Next-block A parts issued per packet: the 8x1 spread, or all within the first two packets.
A_SCHEDULES = {"spread": A_STAGES, "early": [[0, "s0", 1], [2, "s1", 3], [], []]}


def validate_parameters(tokens, inter_dim, hidden_dim, topk, num_experts):
    if not 0 < tokens <= TOKEN_MASK:
        raise ValueError(f"tokens must be in [1, {TOKEN_MASK}]")
    if inter_dim <= 0 or inter_dim % TILE_N:
        raise ValueError("inter_dim must be a positive multiple of 128")
    if hidden_dim < TILE_K or hidden_dim % TILE_K:
        raise ValueError("hidden_dim must be a positive multiple of 128")
    if not 0 < topk < 256 or num_experts <= 0:
        raise ValueError("require 0 < topk < 256 and num_experts > 0")
    # Descriptor byte counts and every voffset must stay below OOB_OFFSET.
    extents = {
        "a": tokens * hidden_dim,
        "a_scale": tokens * (hidden_dim // TILE_K) * 4,
        "weight tile": (inter_dim + TILE_N) * hidden_dim,
        "output": tokens * topk * inter_dim * 2,
    }
    over = [f"{name}={size:,}" for name, size in extents.items() if size >= OOB_OFFSET]
    if over:
        raise ValueError("buffer byte extents too large: " + ", ".join(over))


@functools.cache
def compile_moe_gateup_a8w8_blockscale_gfx942(
    inter_dim, hidden_dim, topk, num_experts, xcd_swizzle=True, b_stages=2, a_schedule="spread", b_aux=0, fixed_k=False
):
    """fixed_k is benchmark-only: every K block reloads block 0 (cache-resident upper bound, not a GEMM)."""
    validate_parameters(1, inter_dim, hidden_dim, topk, num_experts)
    assert b_stages in (2, 4), "B VGPR staging slots must divide the 4 packets of a K block"
    a_stages = A_SCHEDULES[a_schedule]
    I, K = inter_dim, hidden_dim
    KB = K // TILE_K
    n_tiles = I // TILE_N
    b_k_bytes = TILE_K * 16
    scale_rows_b = 2 * I // 128
    # B rows of the four packets relative to this tile's first gate row.
    packet_rows = [0, PACKET_N, I, I + PACKET_N]

    @fx.struct
    class SharedTiles:
        b: fx.Array[fx.Int32, 4 * PACKET_DWORDS, 16]

    @flyc.kernel(known_block_size=[512, 1, 1])
    def gateup_kernel(
        pA: fx.Pointer,
        pB: fx.Pointer,
        pC: fx.Pointer,
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
        pid = fx.Int32(fx.block_idx.x)

        def rsrc_of(ptr, num_records_bytes):
            tensor = fx.rocdl.make_buffer_tensor(
                fx.make_view(ptr, fx.make_layout(1, 1)), max_size=False, num_records_bytes=num_records_bytes
            )
            return fx.rocdl.get_buffer_rsrc(fx.get_iter(tensor))

        def load_i32(rsrc, voffset):
            return fx.Int32(
                rocdl.raw_ptr_buffer_load(
                    T.i32, rsrc, fx.Int32(voffset).ir_value(), fx.Int32(0).ir_value(), fx.Int32(0).ir_value()
                )
            )

        def uniform(value):
            return fx.Int32(rocdl.readfirstlane(T.i32, value.ir_value()))

        valid_rows = uniform(load_i32(rsrc_of(pNumValid, 4), 0))
        valid_blocks = (valid_rows + (SORT_BLOCK_M - 1)) // SORT_BLOCK_M
        valid_blocks = (valid_blocks < num_blocks).select(valid_blocks, num_blocks)
        num_tasks = valid_blocks * n_tiles
        if const_expr(xcd_swizzle):
            # Hardware deals workgroups round-robin over 8 XCDs: give each XCD a contiguous task
            # range so the N tiles of a sorted block (same A rows) share one L2.
            chunk = (num_tasks + 7) >> 3
            xcd_slot = pid >> 3
            task = (pid & 7) * chunk + xcd_slot
            active = (xcd_slot < chunk) & (task < num_tasks)
        else:
            task = pid
            active = task < num_tasks

        if active:
            block = task // n_tiles
            n_tile = task - block * n_tiles
            expert = uniform(load_i32(rsrc_of(pExpertIds, num_blocks * 4), block * 4))

            local_row = wave * 16 + lane % 16
            sorted_rsrc = rsrc_of(pSortedIds, num_blocks * (SORT_BLOCK_M * 4))
            sorted_ids = [
                load_i32(sorted_rsrc, (block * SORT_BLOCK_M + local_row + m_slice * 128) * 4)
                for m_slice in range_constexpr(2)
            ]

            a_tensor = fx.rocdl.make_buffer_tensor(
                fx.make_view(pA, fx.make_layout(1, 1)), max_size=False, num_records_bytes=num_tokens * K
            )
            b_tensor = fx.rocdl.make_buffer_tensor(
                fx.make_view(
                    pB + (fx.Int64(expert) * (2 * I * K // 4) + fx.Int64(n_tile) * (TILE_N * K // 4)),
                    fx.make_layout(1, 1),
                ),
                max_size=False,
                num_records_bytes=(I + TILE_N) * K,
            )
            a_packet = BufferTensor(fx.make_view(fx.get_iter(a_tensor), fx.make_layout(4, 1)))
            b_packet = BufferTensor(fx.make_view(fx.get_iter(b_tensor), fx.make_layout(4, 1)))
            scale_a_rsrc = rsrc_of(pScaleA, num_tokens * (KB * 4))
            scale_b_rsrc = rsrc_of(pScaleB, num_experts * scale_rows_b * (KB * 4))
            c_rsrc = rsrc_of(pC, num_tokens * (topk * I * 2))

            # Gather offsets are fixed per workgroup; invalid rows read zeros and skip stores.
            a_voffsets, scale_a_voffsets, out_offsets = [], [], []
            for m_slice in range_constexpr(2):
                fused = sorted_ids[m_slice]
                token = fused & TOKEN_MASK
                slot = (fused >> 24) & 255
                valid = (token < num_tokens) & (slot < topk)
                a_voffsets.append(
                    [
                        valid.select(token * K + (lane // 16 + half_k * 4) * 16, fx.Int32(OOB_OFFSET))
                        for half_k in range_constexpr(2)
                    ]
                )
                scale_a_voffsets.append(valid.select(token * 4, fx.Int32(OOB_OFFSET)))
                out_offsets.append(
                    valid.select(((token * topk + slot) * I + n_tile * TILE_N) * 2, fx.Int32(OOB_OFFSET))
                )
            lane_b = wave // 2 * (16 * K) + wave % 2 * 1024 + lane * 16
            b_voffsets = [lane_b + packet_rows[packet] * K for packet in range_constexpr(4)]
            b_write = _lds_dwords(wave // 2 * 16 + lane % 16, wave % 2 * 4 + lane // 16)
            b_read = [_lds_dwords(lane % 16, lane // 16 + half_k * 4) for half_k in range_constexpr(2)]
            gate_scale_row = expert * scale_rows_b + n_tile
            scale_b_offsets = [gate_scale_row * (KB * 4), (gate_scale_row + I // 128) * (KB * 4)]

            lds = fx.SharedAllocator().allocate(SharedTiles).peek()

            def lds_view(dwords):
                return fx.make_view(lds.b.ptr + dwords, fx.make_layout(4, 1))

            def write_b(slot, value):
                lds_view(b_write + slot * PACKET_DWORDS).store(value)

            def read_b(slot, rep_n, half_k):
                view = lds_view(b_read[half_k] + (slot * PACKET_DWORDS + rep_n * 16 * (TILE_K // 4)))
                return Vec(view.load()).bitcast(fx.Int64)

            def k_offset(k_block):
                return 0 if const_expr(fixed_k) else k_block

            def load_a(k_block, part):
                m_slice, half_k = divmod(part, 2)
                return a_packet.load(voffset_bytes=a_voffsets[m_slice][half_k], soffset_bytes=k_offset(k_block) * TILE_K)

            def load_b(k_block, packet):
                return b_packet.load(voffset_bytes=b_voffsets[packet], soffset_bytes=k_offset(k_block) * b_k_bytes, aux=b_aux)

            def load_scale_a(k_block, m_slice):
                value = rocdl.raw_ptr_buffer_load(
                    T.f32,
                    scale_a_rsrc,
                    fx.Int32(scale_a_voffsets[m_slice]).ir_value(),
                    fx.Int32(k_offset(k_block) * num_tokens * 4).ir_value(),
                    fx.Int32(0).ir_value(),
                )
                return Vec.from_elements([fx.Float32(value)], fx.Float32)

            def load_scale_b(k_block, column):
                return Vec.from_elements(
                    [s_buffer_load_f32(scale_b_rsrc, scale_b_offsets[column] + k_offset(k_block) * 4)], fx.Float32
                )

            def barrier():
                rocdl.sched_barrier(0)
                rocdl.s_barrier()
                rocdl.sched_barrier(0)

            # Same register plan as the 8x1 GEMM; accum column m_slice * 16 + packet * 4 + rep_n.
            # Global packet x waits in B staging slot (x - 1) % b_stages until its LDS write.
            a_ops = fx.make_rmem_tensor(fx.make_layout((4, 8), (1, 4)), fx.Int32)
            b_stage = fx.make_rmem_tensor(fx.make_layout((4, b_stages), (1, 4)), fx.Int32)
            scale_a = fx.make_rmem_tensor(fx.make_layout((1, 4), (1, 1)), fx.Float32)
            scale_b = fx.make_rmem_tensor(fx.make_layout((1, 4), (1, 1)), fx.Float32)
            accum = fx.make_rmem_tensor(fx.make_layout((4, 32), (1, 4)), fx.Float32)
            partial = fx.make_rmem_tensor(fx.make_layout((4, 4), (1, 4)), fx.Float32)
            partial_scale = fx.make_rmem_tensor(fx.make_layout((1, 2), (1, 1)), fx.Float32)

            def load_a_part(slot, k_block, part):
                if const_expr(isinstance(part, str)):
                    m_slice = int(part[1])
                    scale_a[None, slot * 2 + m_slice].store(load_scale_a(k_block, m_slice))
                else:
                    a_ops[None, slot * 4 + part].store(load_a(k_block, part))

            def run_k(slot, k_block):
                k_next = (k_block + 1 < KB).select(k_block + 1, KB - 1)
                k_after = (k_block + 2 < KB).select(k_block + 2, KB - 1)
                for packet in range_constexpr(4):
                    write_b((packet + 1) % 4, b_stage[None, packet % b_stages].load())
                    rocdl.sched_barrier(0)
                    for part in a_stages[packet]:
                        load_a_part(1 - slot, k_next, part)
                    # Refill the freed staging slot b_stages + 1 packets ahead (clamped past the last block).
                    ahead = packet + b_stages + 1
                    b_block = [k_block, k_next, k_after][ahead // 4]
                    b_stage[None, packet % b_stages].store(load_b(b_block, ahead % 4))
                    b_ops = [[read_b(packet, rep_n, half_k) for half_k in range_constexpr(2)] for rep_n in range_constexpr(4)]
                    rocdl.sched_barrier(0)
                    rocdl.s_waitcnt(lgkmcnt=8)
                    barrier()
                    rocdl.s_setprio(3)
                    rocdl.s_waitcnt(lgkmcnt=0)
                    rocdl.sched_barrier(0)
                    if const_expr(packet == 1):
                        for column in range_constexpr(2):
                            scale_b[None, (1 - slot) * 2 + column].store(load_scale_b(k_next, column))
                        rocdl.sched_barrier(0)
                    for m_slice in range_constexpr(2):
                        previous_packet = (packet - 1) % 4 if m_slice == 0 else packet
                        for mfma_index in range_constexpr(16):
                            rep_n, k_step = MFMA_ORDER[mfma_index]
                            if const_expr(m_slice == 1 and mfma_index == 15):
                                barrier()
                            half_k, word = divmod(k_step, 2)
                            accumulator = Vec.filled(4, 0.0, fx.Float32) if k_step == 0 else Vec(partial[None, rep_n].load())
                            partial[None, rep_n].store(
                                Vec(
                                    rocdl.mfma_f32_16x16x32_fp8_fp8(
                                        T.f32x4,
                                        [
                                            b_ops[rep_n][half_k][word],
                                            Vec(a_ops[None, slot * 4 + m_slice * 2 + half_k].load()).bitcast(fx.Int64)[word],
                                            accumulator,
                                            0,
                                            0,
                                            0,
                                        ],
                                    )
                                )
                            )
                            rocdl.sched_barrier(0)
                            if const_expr(mfma_index == 0):
                                scale = Vec(scale_b[None, slot * 2 + packet // 2].load())[0] * Vec(scale_a[None, slot * 2 + m_slice].load())[0]
                                partial_scale[None, m_slice].store(Vec.from_elements([scale], fx.Float32))
                            if const_expr(mfma_index < 8):
                                dq_slice, dq_packet, dq_rep = 1 - m_slice, previous_packet, 2 + mfma_index // 4
                            else:
                                dq_slice, dq_packet, dq_rep = m_slice, packet, (mfma_index - 8) // 4
                            target = accum[None, dq_slice * 16 + dq_packet * 4 + dq_rep]
                            element = mfma_index % 4
                            target[element] = fx.fma(
                                Vec(partial[None, dq_rep].load())[element],
                                Vec(partial_scale[None, dq_slice].load())[0],
                                target[element],
                            )
                            rocdl.sched_barrier(0)
                    rocdl.s_setprio(0)

            for column in range_constexpr(2):
                scale_b[None, column].store(load_scale_b(0, column))
            accum.fill(0)
            # The first packet retires these zeros as the "previous" slice.
            for rep_n in range_constexpr(4):
                partial[None, rep_n].store(Vec.filled(4, 0.0, fx.Float32))
            for m_slice in range_constexpr(2):
                partial_scale[None, m_slice].store(Vec.filled(1, 0.0, fx.Float32))
            # Replay the steady-state issue order of packets g < 0 (block 0's A goes to slot 0).
            # Loop-entry load ages then match the back edge; otherwise the merged vmcnt at the
            # loop header tightens every iteration's first wait.
            for g in range_constexpr(-max(4, b_stages + 1), 0):
                if const_expr(g == -1):
                    write_b(0, b_stage[None, g % b_stages].load())
                if const_expr(g >= -4):
                    for part in a_stages[g + 4]:
                        load_a_part(0, 0, part)
                staged = g + b_stages + 1
                if const_expr(staged >= 0):
                    b_stage[None, g % b_stages].store(load_b(min(staged // 4, KB - 1), staged % 4))
                rocdl.sched_barrier(0)
            rocdl.s_waitcnt(lgkmcnt=0)
            barrier()
            # Waves 4-7 trail by one barrier so their memory phase overlaps waves 0-3 compute.
            if group == 1:
                barrier()

            if const_expr(KB % 2):
                run_k(0, fx.Int32(0))
            for pair in range(0, KB // 2):
                k_block = fx.Int32(pair) * 2 + KB % 2
                run_k(KB % 2, k_block)
                run_k(1 - KB % 2, k_block + 1)

            # The last slice's partials 2-3 have no later MFMA shadow.
            final_scale = Vec(partial_scale[None, 1].load())[0]
            for rep_n in range_constexpr(2, 4):
                target = accum[None, 16 + 12 + rep_n]
                values = Vec(partial[None, rep_n].load())
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

            lane_group = lane // 16
            odd_group = lane_group % 2 == 1
            low_rows = lane % 16 < 8
            exchange_lane = fx.Int32((lane ^ 16) * 4)
            column_bytes = (lane_group % 2 * 2 + lane_group // 2) * 16 + low_rows.select(fx.Int32(0), fx.Int32(64))

            def row_swap(value):
                # DPP row_ror:8 within 16 lanes = lane ^ 8: the same columns of row r ^ 8.
                return fx.Int32(rocdl.update_dpp(T.i32, value.ir_value(), value.ir_value(), 0x128, 0xF, 0xF, True))

            def column_pair(m_slice, column):
                """16 B per lane: 8 consecutive BF16 columns of subtile column (even lane groups) or column + 1."""
                halves = [
                    pack_bf16x2(
                        silu_mul(
                            Vec(accum[None, m_slice * 16 + column + side].load()),
                            Vec(accum[None, m_slice * 16 + 8 + column + side].load()),
                        )
                    )
                    for side in range_constexpr(2)
                ]
                left, right = halves
                # Even 16-lane groups keep column and take the partner's half; odd groups do column + 1.
                exchanged = [
                    fx.Int32(rocdl.ds_bpermute(T.i32, exchange_lane.ir_value(), odd_group.select(left[pair], right[pair]).ir_value()))
                    for pair in range_constexpr(2)
                ]
                return [odd_group.select(exchanged[pair], left[pair]) for pair in range_constexpr(2)] + [
                    odd_group.select(right[pair], exchanged[pair]) for pair in range_constexpr(2)
                ]

            for m_slice in range_constexpr(2):
                # Rows r < 8 and r ^ 8 trade halves of two column pairs, so each store covers 8 rows x 128 B.
                partner = row_swap(out_offsets[m_slice])
                row_offsets = [low_rows.select(out_offsets[m_slice], partner), low_rows.select(partner, out_offsets[m_slice])]
                for column in range_constexpr(0, 8, 4):
                    first, second = column_pair(m_slice, column), column_pair(m_slice, column + 2)
                    halves = [
                        [low_rows.select(first[i], row_swap(second[i])) for i in range_constexpr(4)],
                        [low_rows.select(row_swap(first[i]), second[i]) for i in range_constexpr(4)],
                    ]
                    for half in range_constexpr(2):
                        rocdl.raw_ptr_buffer_store(
                            Vec.from_elements(halves[half], fx.Int32).ir_value(),
                            c_rsrc,
                            (row_offsets[half] + column_bytes + column * 32).ir_value(),
                            fx.Int32(0).ir_value(),
                            aux=ir.IntegerAttr.get(T.i32, 0),
                        )

    @flyc.jit
    def launch(
        pA: fx.Pointer,
        pB: fx.Pointer,
        pC: fx.Pointer,
        pScaleA: fx.Pointer,
        pScaleB: fx.Pointer,
        pSortedIds: fx.Pointer,
        pExpertIds: fx.Pointer,
        pNumValid: fx.Pointer,
        num_tokens: fx.Int32,
        num_blocks: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        gateup_kernel(
            pA, pB, pC, pScaleA, pScaleB, pSortedIds, pExpertIds, pNumValid, num_tokens, num_blocks,
            value_attrs={"llvm.passthrough": [["target-features", "-packed-fp32-ops"]]},
        ).launch(
            grid=((num_blocks * n_tiles + 7) // 8 * 8, 1, 1),
            block=(512, 1, 1),
            stream=stream,
        )

    launch.compile_hints["target_features"] = "-packed-fp32-ops"
    return launch


def moe_gateup_a8w8_blockscale_gfx942(
    a, w1, a_scale_t, w1_scale, sorted_ids, expert_ids, num_valid_ids, out, xcd_swizzle=True, b_stages=2, a_schedule="spread", b_aux=0, fixed_k=False
):
    """a[T,K] fp8, w1[E,2I,K] fp8 shuffle_weight(16,16), a_scale_t[K/128,T], w1_scale[E,2I/128,K/128] fp32.

    sorted_ids[blocks*256] / expert_ids[blocks] / num_valid_ids[>=1] int32 as from aiter moe_sorting(256);
    out[T,topk,I] bf16 receives SiLU(gate) * up for every valid route.
    """
    tokens, hidden_dim = a.shape
    num_experts, two_inter, _ = w1.shape
    inter_dim = two_inter // 2
    topk = out.shape[1]
    num_blocks = expert_ids.numel()
    validate_parameters(tokens, inter_dim, hidden_dim, topk, num_experts)
    assert sorted_ids.numel() >= num_blocks * SORT_BLOCK_M
    launch = compile_moe_gateup_a8w8_blockscale_gfx942(
        inter_dim, hidden_dim, topk, num_experts, bool(xcd_swizzle), int(b_stages), a_schedule, int(b_aux), bool(fixed_k)
    )
    launch(
        flyc.from_c_void_p(fx.Int32, a.data_ptr()),
        flyc.from_c_void_p(fx.Int32, w1.data_ptr()),
        flyc.from_c_void_p(fx.Int32, out.data_ptr()),
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
