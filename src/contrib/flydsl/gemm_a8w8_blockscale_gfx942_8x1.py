# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL port of the PyHIP gfx942 8x1-wave FP8 block-scale GEMM (contrib/gemm_fp8_gfx942.py, v32).

C[M,N] = A[M,K] @ B[N,K]^T with ScaleA[K/128,M] and ScaleB[ceil(N/128),K/128], BF16 output.
Each 256x256 WG tile uses 8 waves stacked along M; a wave owns two 16-row slices 128 rows apart.
A goes straight into double-buffered VGPRs; B goes through two VGPR staging slots into a
four-slot N64xK128 LDS ring. Tile order, MFMA/dequant order, barriers and BF16 packing follow
the PyHIP kernel, so both produce bitwise identical outputs.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir import ir
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from .gemm_a8w8_blockscale_gfx942_common import pack_bf16x2, s_buffer_load_f32, tb_swizzle
from .helpers import BufferTensor

TILE = 256
TILE_K = 128
PACKET_N = 64
PACKET_DWORDS = PACKET_N * TILE_K // 4
# (rep_n, k_step): accumulations of one partial are one MFMA apart; a finished partial
# retires one element under each later MFMA.
MFMA_ORDER = [(n, k) for pair in (0, 2) for k in range(4) for n in (pair, pair + 1)]
# Next-block A parts (m_slice * 2 + half_k) and A scales ("s<m_slice>") issued per packet.
A_STAGES = [[0, "s0"], [2, "s1"], [1], [3]]
A_PROLOGUE = [0, 1, "s0", 2, 3, "s1"]


def _lds_dwords(row, column):
    # XOR swizzle of the 16B column keeps MFMA-pattern ds_read_b128 conflict free.
    return row * (TILE_K // 4) + ((row ^ column) & 7) * 4


@functools.cache
def compile_gemm_a8w8_blockscale_gfx942_8x1(N, K, preshuffle_b):
    assert N % 16 == 0 and K >= TILE_K and K % TILE_K == 0
    KB = K // TILE_K
    scale_rows_b = (N + 127) // 128
    b_k_bytes = TILE_K * 16 if preshuffle_b else TILE_K
    # v32: rotate the four N64 B packets per 4-WG group on the preshuffled N=4096 path.
    phased = preshuffle_b and N == 4096

    @fx.struct
    class SharedTiles:
        b: fx.Array[fx.Int32, 4 * PACKET_DWORDS, 16]

    @flyc.kernel(known_block_size=[512, 1, 1])
    def gemm_kernel(
        pA: fx.Pointer,
        pB: fx.Pointer,
        pC: fx.Pointer,
        pScaleA: fx.Pointer,
        pScaleB: fx.Pointer,
        M: fx.Int32,
    ):
        tid = fx.thread_idx.x
        lane = tid % 64
        wave = tid // 64
        group = fx.Int32(rocdl.readfirstlane(T.i32, (tid // 256).ir_value()))
        pid = fx.block_idx.x
        block_m, block_n = tb_swizzle(pid, M, N, TILE, TILE)
        packet_phase = pid // 4 & 3 if const_expr(phased) else 0
        row_base = block_m * TILE
        col_base = block_n * TILE
        rows_left = M - row_base
        rows = (rows_left < TILE).select(rows_left, TILE)
        cols_left = N - col_base
        cols = (cols_left < TILE).select(cols_left, TILE)

        a_tensor = fx.rocdl.make_buffer_tensor(
            fx.make_view(pA + fx.Int64(row_base) * (K // 4), fx.make_layout(TILE * K // 4, 1)),
            max_size=False,
            num_records_bytes=rows * K,
        )
        b_tensor = fx.rocdl.make_buffer_tensor(
            fx.make_view(pB + fx.Int64(col_base) * (K // 4), fx.make_layout(TILE * K // 4, 1)),
            max_size=False,
            num_records_bytes=cols * K,
        )
        c_tensor = fx.rocdl.make_buffer_tensor(
            fx.make_view(pC + fx.Int64(row_base) * (N // 2), fx.make_layout(TILE * N // 2, 1)),
            max_size=False,
            num_records_bytes=rows * (N * 2),
        )
        scale_a_tensor = fx.rocdl.make_buffer_tensor(
            fx.make_view(pScaleA, fx.make_layout(M * KB, 1)),
            max_size=False,
            num_records_bytes=M * (KB * 4),
        )
        scale_b_tensor = fx.rocdl.make_buffer_tensor(
            fx.make_view(pScaleB, fx.make_layout(scale_rows_b * KB, 1)),
            max_size=False,
            num_records_bytes=scale_rows_b * KB * 4,
        )
        a_packet = BufferTensor(fx.make_view(fx.get_iter(a_tensor), fx.make_layout(4, 1)))
        b_packet = BufferTensor(fx.make_view(fx.get_iter(b_tensor), fx.make_layout(4, 1)))
        scale_a_rsrc = fx.rocdl.get_buffer_rsrc(fx.get_iter(scale_a_tensor))
        scale_b_rsrc = fx.rocdl.get_buffer_rsrc(fx.get_iter(scale_b_tensor))
        c_rsrc = fx.rocdl.get_buffer_rsrc(fx.get_iter(c_tensor))

        # Row-dependent offsets stay in voffset: gfx9 raw buffers do not range-check soffset.
        local_row = wave * 16 + lane % 16
        a_voffsets = [
            [(local_row + m_slice * 128) * K + (lane // 16 + half_k * 4) * 16 for half_k in range_constexpr(2)]
            for m_slice in range_constexpr(2)
        ]
        if const_expr(preshuffle_b):
            lane_b = wave // 2 * (16 * K) + wave % 2 * 1024 + lane * 16
            b_write = _lds_dwords(wave // 2 * 16 + lane % 16, wave % 2 * 4 + lane // 16)
        else:
            lane_b = tid // 8 * K + tid % 8 * 16
            b_write = _lds_dwords(tid // 8, tid % 8)
        b_voffsets = [lane_b + (packet_phase ^ packet) * (PACKET_N * K) for packet in range_constexpr(4)]
        b_read = [_lds_dwords(lane % 16, lane // 16 + half_k * 4) for half_k in range_constexpr(2)]
        last_row = M - 1
        scale_a_voffsets = []
        for m_slice in range_constexpr(2):
            row = row_base + local_row + m_slice * 128
            # Rows past M are never stored; clamping keeps the K-block soffset in bounds.
            scale_a_voffsets.append((row < last_row).select(row, last_row) * 4)
        scale_b_block = col_base // 128
        if const_expr(phased):
            scale_b_offsets = [
                (scale_b_block + packet_phase // 2) * (KB * 4),
                (scale_b_block + (packet_phase // 2 ^ 1)) * (KB * 4),
            ]
        else:
            scale_b_offsets = [scale_b_block * (KB * 4), (scale_b_block + 1) * (KB * 4)]
        # The last N tile may have no second 128-column scale row.
        last_scale_b = (scale_rows_b - 1) * (KB * 4)
        scale_b_offsets[1] = (scale_b_offsets[1] < last_scale_b).select(scale_b_offsets[1], last_scale_b)

        lds = fx.SharedAllocator().allocate(SharedTiles).peek()

        def lds_view(dwords):
            return fx.make_view(lds.b.ptr + dwords, fx.make_layout(4, 1))

        def write_b(slot, value):
            lds_view(b_write + slot * PACKET_DWORDS).store(value)

        def read_b(slot, rep_n, half_k):
            view = lds_view(b_read[half_k] + (slot * PACKET_DWORDS + rep_n * 16 * (TILE_K // 4)))
            return Vec(view.load()).bitcast(fx.Int64)

        def load_a(k_block, part):
            m_slice, half_k = divmod(part, 2)
            return a_packet.load(voffset_bytes=a_voffsets[m_slice][half_k], soffset_bytes=k_block * TILE_K)

        def load_b(k_block, packet):
            return b_packet.load(voffset_bytes=b_voffsets[packet % 4], soffset_bytes=(k_block + packet // 4) * b_k_bytes)

        def load_scale_a(k_block, m_slice):
            value = rocdl.raw_ptr_buffer_load(
                T.f32,
                scale_a_rsrc,
                fx.Int32(scale_a_voffsets[m_slice]).ir_value(),
                fx.Int32(k_block * M * 4).ir_value(),
                fx.Int32(0).ir_value(),
            )
            return Vec.from_elements([fx.Float32(value)], fx.Float32)

        def load_scale_b(k_block, column):
            return Vec.from_elements([s_buffer_load_f32(scale_b_rsrc, scale_b_offsets[column] + k_block * 4)], fx.Float32)

        def barrier():
            rocdl.sched_barrier(0)
            rocdl.s_barrier()
            rocdl.sched_barrier(0)

        # Columns: a_ops slot * 4 + m_slice * 2 + half_k; scale_a/scale_b slot * 2 + (m_slice | N128 half).
        a_ops = fx.make_rmem_tensor(fx.make_layout((4, 8), (1, 4)), fx.Int32)
        b_stage = fx.make_rmem_tensor(fx.make_layout((4, 2), (1, 4)), fx.Int32)
        scale_a = fx.make_rmem_tensor(fx.make_layout((1, 4), (1, 1)), fx.Float32)
        scale_b = fx.make_rmem_tensor(fx.make_layout((1, 4), (1, 1)), fx.Float32)
        # accum column m_slice * 16 + packet * 4 + rep_n; partial column rep_n.
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
            # The last block re-fetches clamped blocks so it stays inside the loop body;
            # outside the loop LLVM sinks its dequant FMAs past the epilogue barrier and spills.
            k_next = (k_block + 1 < KB).select(k_block + 1, KB - 1)
            for packet in range_constexpr(4):
                write_b((packet + 1) % 4, b_stage[None, packet % 2].load())
                rocdl.sched_barrier(0)
                for part in A_STAGES[packet]:
                    load_a_part(1 - slot, k_next, part)
                b_block = k_block if packet == 0 else (k_block + 1 < KB).select(k_block + 1, KB - 1)
                b_stage[None, packet % 2].store(load_b(b_block, (packet + 3) % 4))
                b_ops = [[read_b(packet, rep_n, half_k) for half_k in range_constexpr(2)] for rep_n in range_constexpr(4)]
                rocdl.sched_barrier(0)
                rocdl.s_waitcnt(lgkmcnt=8)
                barrier()
                rocdl.s_setprio(3)
                rocdl.s_waitcnt(lgkmcnt=0)
                rocdl.sched_barrier(0)
                # Issued after the wait so SMEM latency overlaps MFMA; consumed one K block later.
                if const_expr(packet == 1):
                    for column in range_constexpr(2):
                        scale_b[None, (1 - slot) * 2 + column].store(load_scale_b(k_next, column))
                    rocdl.sched_barrier(0)
                for m_slice in range_constexpr(2):
                    previous_packet = (packet - 1) % 4 if m_slice == 0 else packet
                    for index in range_constexpr(16):
                        rep_n, k_step = MFMA_ORDER[index]
                        if const_expr(preshuffle_b and m_slice == 1 and index == 15):
                            barrier()
                        half_k, word = divmod(k_step, 2)
                        accumulator = Vec.filled(4, 0.0, fx.Float32) if k_step == 0 else Vec(partial[None, rep_n].load())
                        partial[None, rep_n].store(
                            Vec(
                                rocdl.mfma_f32_16x16x32_fp8_fp8(
                                    T.f32x4,
                                    [b_ops[rep_n][half_k][word], Vec(a_ops[None, slot * 4 + m_slice * 2 + half_k].load()).bitcast(fx.Int64)[word], accumulator, 0, 0, 0],
                                )
                            )
                        )
                        # Keep MFMA -> FMA order: the FMA's partial then has two MFMAs of XDL latency cover.
                        rocdl.sched_barrier(0)
                        if const_expr(index == 0):
                            scale = Vec(scale_b[None, slot * 2 + packet // 2].load())[0] * Vec(scale_a[None, slot * 2 + m_slice].load())[0]
                            partial_scale[None, m_slice].store(Vec.from_elements([scale], fx.Float32))
                        if const_expr(index < 8):
                            dq_slice, dq_packet, dq_rep = 1 - m_slice, previous_packet, 2 + index // 4
                        else:
                            dq_slice, dq_packet, dq_rep = m_slice, packet, (index - 8) // 4
                        target = accum[None, dq_slice * 16 + dq_packet * 4 + dq_rep]
                        element = index % 4
                        target[element] = fx.fma(
                            Vec(partial[None, dq_rep].load())[element],
                            Vec(partial_scale[None, dq_slice].load())[0],
                            target[element],
                        )
                        rocdl.sched_barrier(0)
                rocdl.s_setprio(0)
                if const_expr(not preshuffle_b):
                    barrier()

        b_stage[None, 0].store(load_b(0, 0))
        for part in A_PROLOGUE:
            load_a_part(0, 0, part)
        for column in range_constexpr(2):
            scale_b[None, column].store(load_scale_b(0, column))
        accum.fill(0)
        # The first packet retires these zeros as the "previous" slice.
        for rep_n in range_constexpr(4):
            partial[None, rep_n].store(Vec.filled(4, 0.0, fx.Float32))
        for m_slice in range_constexpr(2):
            partial_scale[None, m_slice].store(Vec.filled(1, 0.0, fx.Float32))
        write_b(0, b_stage[None, 0].load())
        rocdl.sched_barrier(0)
        b_stage[None, 0].store(load_b(0, 1))
        b_stage[None, 1].store(load_b(0, 2))
        rocdl.s_waitcnt(lgkmcnt=0)
        barrier()
        # Waves 4-7 trail by one barrier so their memory phase overlaps waves 0-3 compute.
        if group == 1:
            barrier()

        # Two blocks per iteration alternate the A slots; odd KB peels block 0 so the last block stays in the loop.
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

        lane_group = lane // 16
        odd_group = lane_group % 2 == 1
        exchange_lane = fx.Int32((lane ^ 16) * 4)
        output_col = col_base + (lane_group % 2 * 2 + lane_group // 2) * 8
        for m_slice in range_constexpr(2):
            row = local_row + m_slice * 128
            for rep_n in range_constexpr(0, 16, 2):
                column = output_col + (packet_phase * 4 ^ rep_n) * 16
                offset = (column < N).select(row * (N * 2) + column * 2, fx.Int32(0x7FFFFFFF))
                left = pack_bf16x2(Vec(accum[None, m_slice * 16 + rep_n].load()))
                right = pack_bf16x2(Vec(accum[None, m_slice * 16 + rep_n + 1].load()))
                # Even 16-lane groups keep rep_n and take the partner's half; odd groups do rep_n+1.
                exchanged = [
                    fx.Int32(rocdl.ds_bpermute(T.i32, exchange_lane.ir_value(), odd_group.select(left[pair], right[pair]).ir_value()))
                    for pair in range_constexpr(2)
                ]
                packed = Vec.from_elements(
                    [odd_group.select(exchanged[pair], left[pair]) for pair in range_constexpr(2)]
                    + [odd_group.select(right[pair], exchanged[pair]) for pair in range_constexpr(2)],
                    fx.Int32,
                )
                rocdl.raw_ptr_buffer_store(
                    packed.ir_value(),
                    c_rsrc,
                    offset.ir_value(),
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
        M: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        gemm_kernel(
            pA, pB, pC, pScaleA, pScaleB, M,
            value_attrs={"llvm.passthrough": [["target-features", "-packed-fp32-ops"]]},
        ).launch(
            grid=((M + TILE - 1) // TILE * ((N + TILE - 1) // TILE), 1, 1),
            block=(512, 1, 1),
            stream=stream,
        )

    launch.compile_hints["target_features"] = "-packed-fp32-ops"
    return launch


def gemm_a8w8_blockscale_gfx942_8x1(x, weight, out, x_scale_t, w_scale, preshuffle_b):
    """x[M,K], weight[N,K] (optionally preshuffled), x_scale_t[K/128,M], w_scale[ceil(N/128),K/128]."""
    M, K = x.shape
    N = weight.shape[0]
    launch = compile_gemm_a8w8_blockscale_gfx942_8x1(N, K, bool(preshuffle_b))
    launch(
        flyc.from_c_void_p(fx.Int32, x.data_ptr()),
        flyc.from_c_void_p(fx.Int32, weight.data_ptr()),
        flyc.from_c_void_p(fx.Int32, out.data_ptr()),
        flyc.from_c_void_p(fx.Float32, x_scale_t.data_ptr()),
        flyc.from_c_void_p(fx.Float32, w_scale.data_ptr()),
        M,
        torch.cuda.current_stream(),
    )
