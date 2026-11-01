# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL port of the PyHIP gfx942 4x2-LDS FP8 block-scale GEMM (contrib/gemm_fp8_gfx942_4x2_lds.py).

C[M,N] = A[M,K] @ B[N,K]^T with ScaleA[K/128,M] and ScaleB[ceil(N/128),K/128], BF16 output.
The gfx950 (N, M) = (4, 2) quadrant mapping: wave_m = wave // 4 and wave_n = wave % 4. A and B
are both copied cooperatively into 32 KiB LDS each (four 64-row packets). B operands are read
once per K block; A packet p is read in stage p. Tile order, MFMA/dequant order, barriers and
BF16 packing follow the PyHIP kernel, so both produce bitwise identical outputs.
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
PACKET_ROWS = 64
PACKET_DWORDS = PACKET_ROWS * TILE_K // 4


def _lds_dwords(row, column):
    # XOR swizzle of the 16B column keeps MFMA-pattern ds_read_b128 conflict free.
    return row * (TILE_K // 4) + ((row ^ column) & 7) * 4


@functools.cache
def compile_gemm_a8w8_blockscale_gfx942_4x2_lds(N, K, preshuffle_b):
    assert N % 16 == 0 and K >= TILE_K and K % TILE_K == 0
    KB = K // TILE_K
    scale_rows_b = (N + 127) // 128
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
        block_m, block_n = tb_swizzle(fx.block_idx.x, M, N, TILE, TILE)
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

        # Row offsets stay in voffset so raw-buffer range checks zero rows/columns past M/N.
        copy_row = tid // 8
        copy_col = tid % 8
        a_voffsets = [(copy_row + packet * PACKET_ROWS) * K + copy_col * 16 for packet in range_constexpr(4)]
        a_write = _lds_dwords(copy_row, copy_col)
        if const_expr(preshuffle_b):
            b_voffsets = [(wave // 2 + packet * 4) * (16 * K) + wave % 2 * 1024 + lane * 16 for packet in range_constexpr(4)]
            b_write = _lds_dwords(wave // 2 * 16 + lane % 16, wave % 2 * 4 + lane // 16)
        else:
            b_voffsets = a_voffsets
            b_write = a_write
        local_row = wave_m * 16 + lane % 16
        a_read = [
            [_lds_dwords(local_row + rep_m * 32, lane // 16 + half_k * 4) for half_k in range_constexpr(2)]
            for rep_m in range_constexpr(2)
        ]
        b_read = [
            [_lds_dwords(wave_n * 16 + rep_n * 64 + lane % 16, lane // 16 + half_k * 4) for half_k in range_constexpr(2)]
            for rep_n in range_constexpr(2)
        ]
        last_row = M - 1
        scale_a_voffsets = []
        for packet in range_constexpr(4):
            for rep_m in range_constexpr(2):
                row = row_base + local_row + (packet * PACKET_ROWS + rep_m * 32)
                # Rows past M are never stored; clamping keeps the K-block soffset in bounds.
                scale_a_voffsets.append((row < last_row).select(row, last_row) * 4)
        scale_b_offsets = [col_base // 128 * (KB * 4), (col_base // 128 + 1) * (KB * 4)]
        last_scale_b = (N - 1) // 128 * (KB * 4)
        scale_b_offsets[1] = (scale_b_offsets[1] < last_scale_b).select(scale_b_offsets[1], last_scale_b)

        lds = fx.SharedAllocator().allocate(SharedTiles).peek()

        def lds_view(base, dwords):
            return fx.make_view(base + dwords, fx.make_layout(4, 1))

        def write_a(packet):
            lds_view(lds.a.ptr, a_write + packet * PACKET_DWORDS).store(a_stage[None, packet].load())

        def write_b(packet):
            lds_view(lds.b.ptr, b_write + packet * PACKET_DWORDS).store(b_stage[None, packet % 2].load())

        def read_a(packet, rep_m, half_k):
            return lds_view(lds.a.ptr, a_read[rep_m][half_k] + packet * PACKET_DWORDS).load()

        def read_b(side, rep_n, half_k):
            return lds_view(lds.b.ptr, b_read[rep_n][half_k] + side * 2 * PACKET_DWORDS).load()

        def load_scale_a(k_block, packet, rep_m):
            value = rocdl.raw_ptr_buffer_load(
                T.f32,
                scale_a_rsrc,
                fx.Int32(scale_a_voffsets[packet * 2 + rep_m]).ir_value(),
                fx.Int32(k_block * M * 4).ir_value(),
                fx.Int32(0).ir_value(),
            )
            return Vec.from_elements([fx.Float32(value)], fx.Float32)

        def barrier():
            rocdl.sched_barrier(0)
            rocdl.s_barrier()
            rocdl.sched_barrier(0)

        # a_stage: four A packets; b_stage: two B slots. b_ops column side * 4 + rep_n * 2 + half_k.
        a_stage = fx.make_rmem_tensor(fx.make_layout((4, 4), (1, 4)), fx.Int32)
        b_stage = fx.make_rmem_tensor(fx.make_layout((4, 2), (1, 4)), fx.Int32)
        b_ops = fx.make_rmem_tensor(fx.make_layout((4, 8), (1, 4)), fx.Int32)
        # scale_a column slot * 2 + rep_m (slot = packet % 2); scale_b column slot * 2 + side (slot = block % 2).
        scale_a = fx.make_rmem_tensor(fx.make_layout((1, 4), (1, 1)), fx.Float32)
        scale_b = fx.make_rmem_tensor(fx.make_layout((1, 4), (1, 1)), fx.Float32)
        # accum column (packet * 2 + side) * 4 + rep_m * 2 + rep_n; partial column rep_m * 2 + rep_n.
        accum = fx.make_rmem_tensor(fx.make_layout((4, 32), (1, 4)), fx.Float32)
        partial = fx.make_rmem_tensor(fx.make_layout((4, 4), (1, 4)), fx.Float32)
        partial_scale = fx.make_rmem_tensor(fx.make_layout((1, 2), (1, 1)), fx.Float32)

        def load_scale_b(slot, k_block):
            for side in range_constexpr(2):
                scale_b[None, slot * 2 + side].store(
                    Vec.from_elements([s_buffer_load_f32(scale_b_rsrc, scale_b_offsets[side] + k_block * 4)], fx.Float32)
                )

        def run_k(slot, k_block):
            # The last block re-fetches clamped blocks so it stays inside the loop body.
            k_next = (k_block + 1 < KB).select(k_block + 1, KB - 1)
            for packet in range_constexpr(4):
                if const_expr(packet == 0):
                    write_a(3)
                else:
                    write_a(packet - 1)
                    write_b(packet - 1)
                    if const_expr(packet == 3):
                        write_b(3)
                rocdl.sched_barrier(0)
                if const_expr(packet == 0):
                    for side in range_constexpr(2):
                        for rep_n in range_constexpr(2):
                            for half_k in range_constexpr(2):
                                b_ops[None, side * 4 + rep_n * 2 + half_k].store(read_b(side, rep_n, half_k))
                a_ops = [[Vec(read_a(packet, rep_m, half_k)).bitcast(fx.Int64) for half_k in range_constexpr(2)] for rep_m in range_constexpr(2)]
                rocdl.sched_barrier(0)
                for rep_m in range_constexpr(2):
                    if const_expr(packet < 3):
                        scale_a[None, (packet + 1) % 2 * 2 + rep_m].store(load_scale_a(k_block, packet + 1, rep_m))
                    else:
                        scale_a[None, rep_m].store(load_scale_a(k_next, 0, rep_m))
                a_stage[None, packet].store(a_packet.load(voffset_bytes=a_voffsets[packet], soffset_bytes=k_next * TILE_K))
                if const_expr(packet < 3):
                    b_stage[None, packet % 2].store(b_packet.load(voffset_bytes=b_voffsets[packet], soffset_bytes=k_next * b_k_bytes))
                if const_expr(packet == 2):
                    b_stage[None, 1].store(b_packet.load(voffset_bytes=b_voffsets[3], soffset_bytes=k_next * b_k_bytes))
                rocdl.sched_barrier(0)
                rocdl.s_waitcnt(lgkmcnt=0)
                barrier()
                rocdl.s_setprio(3)
                if const_expr(packet == 0):
                    # Next block's B scales; consumed after several lgkmcnt(0) waits.
                    load_scale_b(1 - slot, k_next)
                    rocdl.sched_barrier(0)
                for side in range_constexpr(2):
                    previous_packet = (packet - 1) % 4 if side == 0 else packet
                    for pair_begin in range_constexpr(0, 4, 2):
                        rep_m = pair_begin // 2
                        for k_step in range_constexpr(4):
                            half_k, word = divmod(k_step, 2)
                            for rep_n in range_constexpr(2):
                                partial_index = pair_begin + rep_n
                                index = pair_begin * 4 + k_step * 2 + rep_n
                                b_value = Vec(b_ops[None, side * 4 + rep_n * 2 + half_k].load()).bitcast(fx.Int64)[word]
                                accumulator = Vec.filled(4, 0.0, fx.Float32) if k_step == 0 else Vec(partial[None, partial_index].load())
                                partial[None, partial_index].store(
                                    Vec(
                                        rocdl.mfma_f32_16x16x32_fp8_fp8(
                                            T.f32x4, [b_value, a_ops[rep_m][half_k][word], accumulator, 0, 0, 0]
                                        )
                                    )
                                )
                                # Keep MFMA -> FMA order: the FMA's partial then has two MFMAs of XDL latency cover.
                                rocdl.sched_barrier(0)
                                if const_expr(index == 0 or index == 8):
                                    scale = Vec(scale_a[None, packet % 2 * 2 + index // 8].load())[0] * Vec(scale_b[None, slot * 2 + side].load())[0]
                                    partial_scale[None, index // 8].store(Vec.from_elements([scale], fx.Float32))
                                if const_expr(index < 8):
                                    source, source_scale = 2 + index // 4, 1
                                    column = (previous_packet * 2 + 1 - side) * 4 + source
                                else:
                                    source, source_scale = (index - 8) // 4, 0
                                    column = (packet * 2 + side) * 4 + source
                                target = accum[None, column]
                                element = index % 4
                                target[element] = fx.fma(
                                    Vec(partial[None, source].load())[element],
                                    Vec(partial_scale[None, source_scale].load())[0],
                                    target[element],
                                )
                                rocdl.sched_barrier(0)
                rocdl.s_setprio(0)
                barrier()

        for packet in range_constexpr(4):
            a_stage[None, packet].store(a_packet.load(voffset_bytes=a_voffsets[packet], soffset_bytes=0))
        for pair_begin in range_constexpr(0, 4, 2):
            for packet in range_constexpr(pair_begin, pair_begin + 2):
                b_stage[None, packet % 2].store(b_packet.load(voffset_bytes=b_voffsets[packet], soffset_bytes=0))
            for packet in range_constexpr(pair_begin, pair_begin + 2):
                write_b(packet)
            rocdl.sched_barrier(0)
        for rep_m in range_constexpr(2):
            scale_a[None, rep_m].store(load_scale_a(0, 0, rep_m))
        load_scale_b(0, 0)
        for packet in range_constexpr(4):
            write_a(packet)
        accum.fill(0)
        # The first stage retires these zeros as the "previous" side.
        for index in range_constexpr(4):
            partial[None, index].store(Vec.filled(4, 0.0, fx.Float32))
        for index in range_constexpr(2):
            partial_scale[None, index].store(Vec.filled(1, 0.0, fx.Float32))
        rocdl.s_waitcnt(lgkmcnt=0)
        barrier()
        # Waves 4-7 trail by one barrier so their memory phase overlaps waves 0-3 compute.
        if wave_m == 1:
            barrier()

        # Two blocks per iteration alternate the B-scale slots; odd KB peels block 0.
        if const_expr(KB % 2):
            run_k(0, fx.Int32(0))
        for pair in range(0, KB // 2):
            k_block = fx.Int32(pair) * 2 + KB % 2
            run_k(KB % 2, k_block)
            run_k(1 - KB % 2, k_block + 1)

        # The final side's partials 2-3 have no later MFMA shadow.
        final_scale = Vec(partial_scale[None, 1].load())[0]
        for source in range_constexpr(2, 4):
            target = accum[None, (3 * 2 + 1) * 4 + source]
            values = Vec(partial[None, source].load())
            for element in range_constexpr(4):
                target[element] = fx.fma(values[element], final_scale, target[element])
        if wave_m == 0:
            barrier()

        output_col = col_base + wave_n * 16 + lane // 16 * 4
        for packet in range_constexpr(4):
            for side in range_constexpr(2):
                for rep_m in range_constexpr(2):
                    for rep_n in range_constexpr(2):
                        column = output_col + (side * 128 + rep_n * 64)
                        row = local_row + (packet * PACKET_ROWS + rep_m * 32)
                        offset = (column < N).select(row * (N * 2) + column * 2, fx.Int32(0x7FFFFFFF))
                        packed = Vec.from_elements(pack_bf16x2(Vec(accum[None, (packet * 2 + side) * 4 + rep_m * 2 + rep_n].load())), fx.Int32)
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


def gemm_a8w8_blockscale_gfx942_4x2_lds(x, weight, out, x_scale_t, w_scale, preshuffle_b):
    """x[M,K], weight[N,K] (optionally preshuffled), x_scale_t[K/128,M], w_scale[ceil(N/128),K/128]."""
    M, K = x.shape
    N = weight.shape[0]
    launch = compile_gemm_a8w8_blockscale_gfx942_4x2_lds(N, K, bool(preshuffle_b))
    launch(
        flyc.from_c_void_p(fx.Int32, x.data_ptr()),
        flyc.from_c_void_p(fx.Int32, weight.data_ptr()),
        flyc.from_c_void_p(fx.Int32, out.data_ptr()),
        flyc.from_c_void_p(fx.Float32, x_scale_t.data_ptr()),
        flyc.from_c_void_p(fx.Float32, w_scale.data_ptr()),
        M,
        torch.cuda.current_stream(),
    )
