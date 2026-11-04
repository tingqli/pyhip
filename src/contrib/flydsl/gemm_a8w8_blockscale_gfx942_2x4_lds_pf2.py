# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx942 FP8 block-scale GEMM on a 2x4 wave grid, A and B staged through LDS with a 2-stage load lead (pf2).

C[M,N] = A[M,K] @ B[N,K]^T with ScaleA[K/128, M] and ScaleB[ceil(N/128), K/128],
BF16 output. B is either plain [N,K] or aiter shuffle_weight(B, layout=(16, 16)).
Each 256x256 WG tile uses 8 waves: wave_m = wave // 4 owns 128 rows, wave_n = wave % 4
owns 64 columns. LDS holds one K128 tile of A and B (32 KiB each) as four 64-row packets;
packets are refilled from VGPR staging as soon as they are free, two stages after their global
loads are issued, so some loads run two K blocks ahead (the pf1 kernel: one stage, one K block).
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir import ir
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from .helpers import BufferTensor

TILE_M = TILE_N = 256
TILE_K = 128
PACKET_ROWS = 64
PACKET_DWORDS = PACKET_ROWS * TILE_K // 4


def _lds_dwords(row, column):
    # XOR swizzle of the 16B column keeps MFMA-pattern ds_read_b128 conflict free.
    return row * (TILE_K // 4) + ((row ^ column) & 7) * 4


@functools.cache
def compile_gemm_a8w8_blockscale_gfx942_2x4_lds_pf2(N, K, preshuffle_b):
    assert N % 16 == 0 and K >= TILE_K and K % TILE_K == 0
    KB = K // TILE_K
    n_tiles = (N + TILE_N - 1) // TILE_N
    scale_b_rows = (N + 127) // 128
    b_k_bytes = TILE_K * 16 if preshuffle_b else TILE_K

    @fx.struct
    class SharedTiles:
        a: fx.Array[fx.Int32, 4 * PACKET_DWORDS, 16]
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
        wave_m = fx.Int32(rocdl.readfirstlane(T.i32, (tid // 256).ir_value()))
        wave_n = fx.Int32(rocdl.readfirstlane(T.i32, (wave % 4).ir_value()))

        # Eight M tiles per group share each B column tile.
        num_m = (M + TILE_M - 1) // TILE_M
        group_tasks = 8 * n_tiles
        pid = fx.block_idx.x
        first_m = pid // group_tasks * 8
        remaining_m = num_m - first_m
        group_m = (remaining_m < 8).select(remaining_m, 8)
        block_m = first_m + pid % group_tasks % group_m
        block_n = pid % group_tasks // group_m
        row_base = block_m * TILE_M
        col_base = block_n * TILE_N
        rows_left = M - row_base
        rows = (rows_left < TILE_M).select(rows_left, TILE_M)
        cols_left = N - col_base
        cols = (cols_left < TILE_N).select(cols_left, TILE_N)

        a_tensor = fx.rocdl.make_buffer_tensor(
            fx.make_view(pA + fx.Int64(row_base) * (K // 4), fx.make_layout(TILE_M * K // 4, 1)),
            max_size=False,
            num_records_bytes=rows * K,
        )
        b_tensor = fx.rocdl.make_buffer_tensor(
            fx.make_view(pB + fx.Int64(col_base) * (K // 4), fx.make_layout(TILE_N * K // 4, 1)),
            max_size=False,
            num_records_bytes=cols * K,
        )
        c_tensor = fx.rocdl.make_buffer_tensor(
            fx.make_view(pC + fx.Int64(row_base) * (N // 2), fx.make_layout(TILE_M * N // 2, 1)),
            max_size=False,
            num_records_bytes=rows * (N * 2),
        )
        scale_a_tensor = fx.rocdl.make_buffer_tensor(
            fx.make_view(pScaleA, fx.make_layout(M * KB, 1)),
            max_size=False,
            num_records_bytes=M * (KB * 4),
        )
        scale_b_tensor = fx.rocdl.make_buffer_tensor(
            fx.make_view(pScaleB, fx.make_layout(scale_b_rows * KB, 1)),
            max_size=False,
            num_records_bytes=scale_b_rows * KB * 4,
        )
        a_packet = BufferTensor(fx.make_view(fx.get_iter(a_tensor), fx.make_layout(4, 1)))
        b_packet = BufferTensor(fx.make_view(fx.get_iter(b_tensor), fx.make_layout(4, 1)))
        scale_a_rsrc = fx.rocdl.get_buffer_rsrc(fx.get_iter(scale_a_tensor))
        scale_b_rsrc = fx.rocdl.get_buffer_rsrc(fx.get_iter(scale_b_tensor))
        c_rsrc = fx.rocdl.get_buffer_rsrc(fx.get_iter(c_tensor))

        def load_f32(rsrc, voffset):
            zero = fx.Int32(0).ir_value()
            value = rocdl.raw_ptr_buffer_load(T.f32, rsrc, fx.Int32(voffset).ir_value(), zero, zero)
            return Vec.from_elements([fx.Float32(value)], fx.Float32)

        # Row offsets stay in voffset so raw-buffer range checks zero rows/columns past M/N.
        copy_row = tid // 8
        copy_col = tid % 8
        a_voffsets = [(copy_row + p * PACKET_ROWS) * K + copy_col * 16 for p in range_constexpr(4)]
        a_write = _lds_dwords(copy_row, copy_col)
        if const_expr(preshuffle_b):
            b_voffsets = [
                (wave // 2 + p * 4) * (16 * K) + wave % 2 * 1024 + lane * 16
                for p in range_constexpr(4)
            ]
            b_write = _lds_dwords(wave // 2 * 16 + lane % 16, wave % 2 * 4 + lane // 16)
        else:
            b_voffsets = a_voffsets
            b_write = a_write
        a_read = [_lds_dwords(wave_m * 16 + lane % 16, lane // 16 + half_k * 4) for half_k in range_constexpr(2)]
        b_read = [_lds_dwords(wave_n * 16 + lane % 16, lane // 16 + half_k * 4) for half_k in range_constexpr(2)]
        scale_a_voffset = (row_base + wave_m * 16 + lane % 16) * 4
        scale_b_block = col_base // 128

        lds = fx.SharedAllocator().allocate(SharedTiles).peek()

        def lds_view(base, dwords):
            return fx.make_view(base + dwords, fx.make_layout(4, 1))

        def write_a(packet, value):
            lds_view(lds.a.ptr, a_write + packet * PACKET_DWORDS).store(value)

        def write_b(packet, value):
            lds_view(lds.b.ptr, b_write + packet * PACKET_DWORDS).store(value)

        def read_a(packet, rep_m, half_k):
            view = lds_view(lds.a.ptr, a_read[half_k] + (packet * PACKET_DWORDS + rep_m * 32 * 32))
            return Vec(view.load()).bitcast(fx.Int64)

        def read_b(side, rep_n, half_k):
            view = lds_view(lds.b.ptr, b_read[half_k] + (side * 2 + rep_n) * PACKET_DWORDS)
            return Vec(view.load()).bitcast(fx.Int64)

        def load_a(k_block, packet):
            return a_packet.load(voffset_bytes=a_voffsets[packet], soffset_bytes=k_block * TILE_K)

        def load_b(k_block, packet):
            return b_packet.load(voffset_bytes=b_voffsets[packet], soffset_bytes=k_block * b_k_bytes)

        def load_scale_a(k_block, packet, rep_m):
            offset = scale_a_voffset + (k_block * M + (packet * PACKET_ROWS + rep_m * 32)) * 4
            return load_f32(scale_a_rsrc, offset)

        def load_scale_b(k_block, side):
            row = scale_b_block + side
            row = (row < scale_b_rows).select(row, scale_b_rows - 1)
            return load_f32(scale_b_rsrc, (row * KB + k_block) * 4)

        def barrier():
            rocdl.sched_barrier(0)
            rocdl.s_barrier()
            rocdl.sched_barrier(0)

        # a_stage: [A(k) packet 3, A(k+1) packet 0]; b_stage: B(k+1) packets 0 and 1.
        a_stage = fx.make_rmem_tensor(fx.make_layout((4, 2), (1, 4)), fx.Int32)
        b_stage = fx.make_rmem_tensor(fx.make_layout((4, 2), (1, 4)), fx.Int32)
        scale_a = fx.make_rmem_tensor(fx.make_layout((1, 8), (1, 1)), fx.Float32)
        scale_b = fx.make_rmem_tensor(fx.make_layout((1, 2), (1, 1)), fx.Float32)
        scale_b_next = fx.make_rmem_tensor(fx.make_layout((1, 2), (1, 1)), fx.Float32)
        c_acc = fx.make_rmem_tensor(fx.make_layout((4, 32), (1, 4)), fx.Float32)
        # The last pair of each stage retires under the next pair's MFMAs, across stage and block edges.
        pending = fx.make_rmem_tensor(fx.make_layout((4, 2), (1, 4)), fx.Float32)
        pending_scale = fx.make_rmem_tensor(fx.make_layout((1, 2), (1, 1)), fx.Float32)
        last_pair = [30, 31]

        next_block = min(1, KB - 1)
        a_first = [load_a(0, packet) for packet in range_constexpr(3)]
        a_stage[None, 0].store(load_a(0, 3))
        b_first = [load_b(0, packet) for packet in range_constexpr(4)]
        a_stage[None, 1].store(load_a(next_block, 0))
        b_stage[None, 0].store(load_b(next_block, 0))
        b_stage[None, 1].store(load_b(next_block, 1))
        for packet in range_constexpr(3):
            for rep_m in range_constexpr(2):
                scale_a[None, packet * 2 + rep_m].store(load_scale_a(0, packet, rep_m))
        for side in range_constexpr(2):
            scale_b[None, side].store(load_scale_b(0, side))
        for packet in range_constexpr(3):
            write_a(packet, a_first[packet])
        for packet in range_constexpr(4):
            write_b(packet, b_first[packet])
        c_acc.fill(0)
        for rep_n in range_constexpr(2):
            pending[None, rep_n].store(Vec.filled(4, 0.0, fx.Float32))
        pending_scale[None, 0].store(Vec.filled(1, 0.0, fx.Float32))
        rocdl.s_waitcnt(lgkmcnt=0)
        barrier()
        # Waves 4-7 trail by one barrier so their memory phase overlaps waves 0-3 compute.
        if wave_m == 1:
            barrier()

        def run_block(k_block):
            k1 = (k_block + 1 < KB).select(k_block + 1, KB - 1)
            k2 = (k_block + 2 < KB).select(k_block + 2, KB - 1)
            b_ops = a_next1 = a_next2 = b_next2 = b_next3 = None
            pending_index = last_pair
            for stage in range_constexpr(4):
                # A packet p is free after stage p; all B packets are free after stage 0.
                if const_expr(stage == 0):
                    write_a(3, a_stage[None, 0].load())
                elif const_expr(stage == 1):
                    write_a(0, a_stage[None, 1].load())
                    write_b(0, b_stage[None, 0].load())
                elif const_expr(stage == 2):
                    write_a(1, a_next1)
                    write_b(1, b_stage[None, 1].load())
                    write_b(2, b_next2)
                else:
                    write_a(2, a_next2)
                    write_b(3, b_next3)

                if const_expr(stage == 0):
                    b_ops = [
                        [[read_b(0, rep_n, half_k) for half_k in range_constexpr(2)] for rep_n in range_constexpr(2)],
                        None,
                    ]
                a_ops = [[read_a(stage, rep_m, half_k) for half_k in range_constexpr(2)] for rep_m in range_constexpr(2)]

                if const_expr(stage == 0):
                    a_next1 = load_a(k1, 1)
                    b_next2 = load_b(k1, 2)
                    for side in range_constexpr(2):
                        scale_b_next[None, side].store(load_scale_b(k1, side))
                    for rep_m in range_constexpr(2):
                        scale_a[None, 6 + rep_m].store(load_scale_a(k_block, 3, rep_m))
                elif const_expr(stage == 1):
                    a_next2 = load_a(k1, 2)
                    b_next3 = load_b(k1, 3)
                elif const_expr(stage == 2):
                    a_stage[None, 0].store(load_a(k1, 3))
                    b_stage[None, 0].store(load_b(k2, 0))
                else:
                    a_stage[None, 1].store(load_a(k2, 0))
                    b_stage[None, 1].store(load_b(k2, 1))
                if const_expr(stage > 0):
                    for rep_m in range_constexpr(2):
                        scale_a[None, (stage - 1) * 2 + rep_m].store(load_scale_a(k1, stage - 1, rep_m))

                rocdl.s_waitcnt(lgkmcnt=0)
                barrier()
                if const_expr(stage == 0):
                    b_ops[1] = [
                        [read_b(1, rep_n, half_k) for half_k in range_constexpr(2)]
                        for rep_n in range_constexpr(2)
                    ]
                for side in range_constexpr(2):
                    side_scale = Vec(scale_b[None, side].load())[0]
                    for pair in range_constexpr(2):
                        rep_m = pair
                        pair_scale = Vec(scale_a[None, stage * 2 + rep_m].load())[0] * side_scale
                        partials = [Vec.filled(4, 0.0, fx.Float32) for _ in range_constexpr(2)]
                        for k_step in range_constexpr(4):
                            half_k, word = divmod(k_step, 2)
                            for rep_n in range_constexpr(2):
                                slot = k_step * 2 + rep_n
                                partials[rep_n] = Vec(
                                    rocdl.mfma_f32_16x16x32_fp8_fp8(
                                        T.f32x4,
                                        [b_ops[side][rep_n][half_k][word], a_ops[rep_m][half_k][word], partials[rep_n], 0, 0, 0],
                                    )
                                )
                                # Retire one element of the previous pair under each MFMA.
                                old_n, element = divmod(slot, 4)
                                old_values = Vec(pending[None, old_n].load())
                                old_scale = Vec(pending_scale[None, 0].load())[0]
                                target = c_acc[None, pending_index[old_n]]
                                target[element] = fx.fma(old_values[element], old_scale, target[element])
                                rocdl.sched_barrier(0)
                        for rep_n in range_constexpr(2):
                            pending[None, rep_n].store(partials[rep_n])
                        pending_scale[None, 0].store(Vec.from_elements([pair_scale], fx.Float32))
                        pending_index = [
                            ((stage * 2 + side) * 2 + rep_m) * 2 + rep_n for rep_n in range_constexpr(2)
                        ]
                barrier()
            for side in range_constexpr(2):
                scale_b[None, side].store(scale_b_next[None, side].load())

        for k_block in range(0, KB):
            run_block(fx.Int32(k_block))

        final_scale = Vec.filled(4, Vec(pending_scale[None, 0].load())[0], fx.Float32)
        for rep_n in range_constexpr(2):
            index = last_pair[rep_n]
            c_acc[None, index].store(fx.fma(Vec(pending[None, rep_n].load()), final_scale, Vec(c_acc[None, index].load())))

        if wave_m == 0:
            barrier()
        row_lane = wave_m * 16 + lane % 16
        col_lane = col_base + wave_n * 16 + lane // 16 * 4
        for index in range_constexpr(32):
            stage, side, rep_m, rep_n = index // 8, index // 4 % 2, index // 2 % 2, index % 2
            row = row_lane + (stage * PACKET_ROWS + rep_m * 32)
            col = col_lane + (side * 128 + rep_n * 64)
            offset = (col < N).select(row * (N * 2) + col * 2, fx.Int32(0x7FFFFFFF))
            packed = Vec(c_acc[None, index].load()).to(fx.BFloat16).bitcast(fx.Int32)
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
            grid=((M + TILE_M - 1) // TILE_M * n_tiles, 1, 1),
            block=(512, 1, 1),
            stream=stream,
        )

    launch.compile_hints["target_features"] = "-packed-fp32-ops"
    return launch


def gemm_a8w8_blockscale_gfx942_2x4_lds_pf2(x, weight, out, x_scale_t, w_scale, preshuffle_b):
    """x[M,K], weight[N,K] (optionally preshuffled), x_scale_t[K/128,M], w_scale[ceil(N/128),K/128]."""
    M, K = x.shape
    N = weight.shape[0]
    launch = compile_gemm_a8w8_blockscale_gfx942_2x4_lds_pf2(N, K, bool(preshuffle_b))
    launch(
        flyc.from_c_void_p(fx.Int32, x.data_ptr()),
        flyc.from_c_void_p(fx.Int32, weight.data_ptr()),
        flyc.from_c_void_p(fx.Int32, out.data_ptr()),
        flyc.from_c_void_p(fx.Float32, x_scale_t.data_ptr()),
        flyc.from_c_void_p(fx.Float32, w_scale.data_ptr()),
        M,
        torch.cuda.current_stream(),
    )
