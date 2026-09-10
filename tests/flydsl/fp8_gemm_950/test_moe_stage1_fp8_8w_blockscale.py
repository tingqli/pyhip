# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
#
# fp8 MoE stage1 gate/up GEMM (C = B_expert * A_token, output bf16).
# 抽象风格编写（flat_divide / make_tiled_copy / make_tiled_mma / make_fragment / fx.copy /
# fx.gemm），不做手动 byte-offset DMA。算法对标 test_gemm.py::compile_gemm_950 的
# gemm_8wave_950（fp8）：2x2 quadrant、8 wave（tiled_mma wave grid 4x2）、双缓冲 LDS、
# 每 region compute-phase(s_setprio + s_barrier) 调度。
#   - BLOCK_M=BLOCK_N=BLOCK_K=128, TILE_M=TILE_N=256, block=512(8 wave)
#   - MFMA V_MFMA_SCALE_F32_16X16X128_F8F6F4（scale=0）
#   - A/B LDS dual-padding（[[1024,16],[2048,32]]）消 bank conflict；tile-based fx.copy g2s。
#   - 约定：A 走 make_fragment_B，B 走 make_fragment_A；fx.gemm(mma, C, frag_B, frag_A)。
#
# 运行：cd /mywork/FlyDSL/tests/kernels && HIP_VISIBLE_DEVICES=4 python ./test_gemm_v9_fp8_8wave.py

import os

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr.typing import BFloat16, Float8E4M3FN, Float32, Int32, T, Vector
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl, vector, arith
from flydsl.expr.arith import CmpIPredicate
from flydsl.expr.typing import Vector as Vec
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as _llvm
from flydsl._mlir.dialects import fly as _fly_dialect
from flydsl.compiler.ast_rewriter import ASTRewriter


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


def div_up(x, y):
    return (x + y - 1) // y


def encode_waitcnt_950(vmcnt=63, expcnt=7, lgkmcnt=63):
    vm_lo = vmcnt & 0xF
    vm_hi = (vmcnt >> 4) & 0x3
    return vm_lo | (expcnt << 4) | (lgkmcnt << 8) | (vm_hi << 14)


SORT_BLOCK_M = 256
TOKEN_MASK = 0xFFFFFF


def compile_moe_stage1_fp8_8wave(
    TILE_M,
    TILE_N,
    TILE_K,
    N,
    K,
    TOPK,
    NUM_EXPERTS,
    pid_swizzle=True,
    permlane_epilogue=True,
    preshuffle_b=False,
    useTileDMA=False,
    gate_up=False,
):
    assert TILE_M == SORT_BLOCK_M
    assert preshuffle_b == False, "preshuffle B is not supported"
    assert useTileDMA == False, "MoE A gather requires the raw DMA path"
    BLOCK_M = TILE_M // 2
    BLOCK_N = TILE_N // 2
    BLOCK_K = TILE_K
    assert N % 8 == 0
    assert not gate_up or N % (2 * BLOCK_N) == 0
    assert not gate_up or permlane_epilogue
    output_N = N // 2 if gate_up else N
    element_type = fx.Float8E4M3FN
    elements_per_128b = 16  # 128bit / fp8(8bit)
    scaleA_groups = K // 128
    scaleB_rows = TILE_N // 128
    scaleB_elems = scaleB_rows * scaleA_groups
    with_scale = True

    def _get_pids_950(pid, GRID_MN, NUM_XCDS):
        num_pid_n = div_up(N, TILE_N)
        num_cus = 256
        grouped_blocks = GRID_MN - GRID_MN % num_cus
        if pid < grouped_blocks:
            block_base = pid // num_cus * num_cus
            cu_id = pid % num_cus
            xcd_id = cu_id % NUM_XCDS
            xcd_cu = cu_id // NUM_XCDS
            pid = block_base + xcd_id * (num_cus // NUM_XCDS) + xcd_cu
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
        return pid_m, pid_n

    get_pids_950 = ASTRewriter.transform(_get_pids_950)

    # A/B LDS dual padding（对标 gemm_4wave_950 fp8：[[1024,16],[2048,32]]）
    A_GROUP = 8 * BLOCK_K + 16
    a_lds_elems = 2 * A_GROUP + 32  # 每 2 组再 pad 32
    a_lds_elems = (
        BLOCK_M // 16
    ) * a_lds_elems  # 8 * (2*(8*128+16)+32) = 8*2112 = 16896

    @fx.struct
    class LDS:
        a_t0: fx.Array[Float8E4M3FN, a_lds_elems, 16]
        a_b0: fx.Array[Float8E4M3FN, a_lds_elems, 16]
        a_t1: fx.Array[Float8E4M3FN, a_lds_elems, 16]
        a_b1: fx.Array[Float8E4M3FN, a_lds_elems, 16]
        b_l0: fx.Array[Float8E4M3FN, a_lds_elems, 16]
        b_l1: fx.Array[Float8E4M3FN, a_lds_elems, 16]
        b_r0: fx.Array[Float8E4M3FN, a_lds_elems, 16]
        b_r1: fx.Array[Float8E4M3FN, a_lds_elems, 16]
        # scale a ping-pong LDS
        scale_a0: fx.Array[Float32, 512, 4]
        scale_a1: fx.Array[Float32, 512, 4]
        scale_b: fx.Array[Float32, (TILE_N // 128) * (K // 128), 4]
        route: fx.Array[Int32, SORT_BLOCK_M, 4]
        a_row_offset: fx.Array[Int32, SORT_BLOCK_M, 4]

    @flyc.kernel(known_block_size=[512, 1, 1])
    def moe_stage1_kernel(
        argA: fx.Tensor,
        argB: fx.Tensor,
        argC: fx.Tensor,
        argScaleA: fx.Tensor,
        argScaleB: fx.Tensor,
        argSortedIds: fx.Tensor,
        argExpertIds: fx.Tensor,
        argNumValidIds: fx.Tensor,
        numTokens: int,
        numExpertBlocks: int,
    ):
        tid = fx.thread_idx.x
        wave_id = tid // 64
        num_pid_n = div_up(N, TILE_N)
        num_expert_blocks_i32 = fx.Int32(numExpertBlocks)
        if const_expr(pid_swizzle):
            bid_x, bid_y = get_pids_950(
                fx.block_idx.x,
                fx.grid_dim.x,
                8,
            )
        else:
            bid_x = fx.block_idx.x // num_pid_n
            bid_y = fx.block_idx.x % num_pid_n

        bid_x_i32 = fx.Int32(bid_x)
        bid_y_i32 = fx.Int32(bid_y)
        num_tokens_i32 = fx.Int32(numTokens)
        expert_rsrc = fx.buffer_ops.create_buffer_resource(
            argExpertIds,
            num_records_bytes=arith._to_raw(num_expert_blocks_i32 * fx.Int32(4)),
        )
        expert_id = fx.Int32(
            fx.buffer_ops.buffer_load(expert_rsrc, bid_x_i32, vec_width=1, dtype=T.i32)
        )
        expert_id = fx.Int32(rocdl.readfirstlane(T.i32, arith._to_raw(expert_id)))
        sorted_rsrc = fx.buffer_ops.create_buffer_resource(
            argSortedIds,
            num_records_bytes=arith._to_raw(
                num_expert_blocks_i32 * fx.Int32(TILE_M * 4)
            ),
        )
        valid_rsrc = fx.buffer_ops.create_buffer_resource(
            argNumValidIds, num_records_bytes=arith._to_raw(fx.Int32(4))
        )
        num_valid_ids = fx.Int32(
            fx.buffer_ops.buffer_load(valid_rsrc, fx.Int32(0), vec_width=1, dtype=T.i32)
        )
        num_valid_ids = fx.Int32(
            rocdl.readfirstlane(T.i32, arith._to_raw(num_valid_ids))
        )

        a_iter = fx.recast_iter(element_type, fx.get_iter(argA))
        expert_b_elems = N * K
        b_iter = fx.add_offset(
            fx.recast_iter(element_type, fx.get_iter(argB)),
            expert_id * expert_b_elems,
        )
        A_2d = fx.Tensor(fx.make_view(a_iter, fx.make_layout((numTokens, K), (K, 1))))
        B_2d = fx.Tensor(fx.make_view(b_iter, fx.make_layout((N, K), (K, 1))))
        C_2d = fx.Tensor(
            fx.make_view(
                fx.get_iter(argC),
                fx.make_layout((numTokens * TOPK, output_N), (output_N, 1)),
            )
        )

        A = fx.rocdl.make_buffer_tensor(A_2d, max_size=False)
        B = fx.rocdl.make_buffer_tensor(B_2d, max_size=False)
        C = fx.rocdl.make_buffer_tensor(C_2d, max_size=False)
        a_dma_rsrc = fx.buffer_ops.create_buffer_resource(
            argA, num_records_bytes=arith._to_raw(num_tokens_i32 * fx.Int32(K))
        )
        expert_b_byte_offset = arith.index_cast(T.index, expert_id) * arith.constant(
            expert_b_elems, index=True
        )
        b_dma_rsrc = fx.buffer_ops.create_buffer_resource(
            argB,
            num_records_bytes=expert_b_elems,
            base_byte_offset=expert_b_byte_offset,
        )

        # subA/subB,  一个WG 被分成两个slice
        # bA flat_divide output :[BM, BK, REP_BM, REP_BK]
        # bA slice:[BM, BK, K//BK]
        # bB slice:[BN, BK, K//BK]
        bA_t = fx.flat_divide(A, (BLOCK_M, BLOCK_K))[None, None, 0, None]
        bA_b = fx.flat_divide(A, (BLOCK_M, BLOCK_K))[None, None, 0, None]
        if const_expr(gate_up):
            bB_l = fx.flat_divide(B, (BLOCK_N, BLOCK_K))[
                None, None, bid_y, None
            ]
            bB_r = fx.flat_divide(B, (BLOCK_N, BLOCK_K))[
                None, None, N // (2 * BLOCK_N) + bid_y, None
            ]
        else:
            bB_l = fx.flat_divide(B, (BLOCK_N, BLOCK_K))[
                None, None, bid_y * 2 + 0, None
            ]
            bB_r = fx.flat_divide(B, (BLOCK_N, BLOCK_K))[
                None, None, bid_y * 2 + 1, None
            ]

        # ======================================== global: A, B global memroy read layout/tensor  ===============================================
        # bA slice natural layout, 分成16个group,groups 的每行 intreleaved， group 内部行不连续
        # (BM, BK, K//BK), (K, 1, BK) -> ((groups, BM//groups), BK, K//BK), ((K, BM//groups*K), 1, BK)
        # permute subM , group 访问的layout: (( BM//groups, groups), BK, K//BK), ((BM//groups*K, K), 1, BK)
        a_grouped = fx.make_layout(
            ((8, BLOCK_M // 8), BLOCK_K, K // BLOCK_K),
            ((BLOCK_M // 8 * K, K), 1, BLOCK_K),
        )
        bA_t = fx.Tensor(fx.make_view(fx.get_iter(bA_t), a_grouped))
        bA_b = fx.Tensor(fx.make_view(fx.get_iter(bA_b), a_grouped))
        b_grouped = fx.make_layout(
            ((8, BLOCK_N // 8), BLOCK_K, K // BLOCK_K),
            ((BLOCK_N // 8 * K, K), 1, BLOCK_K),
        )
        bB_l = fx.Tensor(fx.make_view(fx.get_iter(bB_l), b_grouped))
        bB_r = fx.Tensor(fx.make_view(fx.get_iter(bB_r), b_grouped))
        # preshuffle B：host 端 shuffle_weight(B, layout=(16,64))，kernel 用 subB 再视图
        if const_expr(preshuffle_b):
            _subB = fx.make_layout(
                ((16, BLOCK_N // 16), (16, BLOCK_K // 16), K // BLOCK_K),
                ((16, 16 * K), (1, 256), 2048),
            )
            bB_l = fx.Tensor(fx.make_view(fx.get_iter(bB_l), _subB))
            bB_r = fx.Tensor(fx.make_view(fx.get_iter(bB_r), _subB))

        # ===========================   copy atom   ===========================
        async_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopyLDS128b(), 128)
        buffer_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), element_type)
        lds_copy_atom = fx.make_copy_atom(fx.UniversalCopy128b(), element_type)

        lds = fx.SharedAllocator().allocate(LDS).peek()

        if tid < SORT_BLOCK_M:
            route_row = bid_x_i32 * SORT_BLOCK_M + fx.Int32(tid)
            route_value = fx.buffer_ops.buffer_load(
                sorted_rsrc, route_row, vec_width=1, dtype=T.i32
            )
            fx.add_offset(lds.route.ptr, tid).store(fx.Int32(route_value))
            route_token = arith.andi(
                fx.Int32(route_value), arith.constant(TOKEN_MASK, type=T.i32)
            )
            fx.add_offset(lds.a_row_offset.ptr, tid).store(
                fx.Int32(route_token) * fx.Int32(K)
            )
        rocdl.s_waitcnt(encode_waitcnt_950(vmcnt=0, lgkmcnt=0))
        rocdl.s_barrier()

        def _load_route(row_local):
            return fx.Int32(fx.add_offset(lds.route.ptr, row_local).load())

        def _load_a_row_offset(row_local):
            return fx.Int32(fx.add_offset(lds.a_row_offset.ptr, row_local).load())

        # Stage the complete ScaleB tile before constructing any tiled-MMA or
        # accumulator fragments. The wait/barrier closes this register lifetime,
        # allowing the loader's address VGPRs to be reused by the MFMA pipeline.
        if const_expr(with_scale):
            sB_rsrc = fx.buffer_ops.create_buffer_resource(
                argScaleB,
                num_records_bytes=div_up(N, 128) * scaleA_groups * 4,
                base_byte_offset=arith.index_cast(T.index, expert_id)
                * arith.constant(div_up(N, 128) * scaleA_groups * 4, index=True),
            )
            scale_b_lds = fx.make_view(lds.scale_b.ptr, fx.make_layout(scaleB_elems, 1))
            scale_b_root_ptr = _fly_dialect.extract_aligned_pointer_as_index(
                ir.Type.parse("!llvm.ptr<3>"), arith._to_raw(scale_b_lds)
            )
            total_lanes = 512
            elems_per_128b_scale = 4
            elems_per_round_128b = total_lanes * elems_per_128b_scale
            rounds_128b = scaleB_elems // elems_per_round_128b
            loaded_128b = rounds_128b * elems_per_round_128b
            remaining_elems = scaleB_elems - loaded_128b
            rounds_32b = remaining_elems // total_lanes
            loaded_32b = rounds_32b * total_lanes
            tail_elems = remaining_elems - loaded_32b
            scale_b_global_base = fx.Int32(bid_y * scaleB_elems * 4)

            if const_expr(gate_up):
                scale_b_global_base = fx.Int32(bid_y * scaleA_groups * 4)
                lane_byte_offset_32b = fx.Int32(tid * 4)
                wave_offset_32b = rocdl.readfirstlane(
                    T.i32, arith._to_raw(fx.Int32(wave_id * 64 * 4))
                )
                for gate_up_half in range_constexpr(2):
                    if tid < scaleA_groups:
                        scale_b_dst = fx.buffer_ops.get_element_ptr(
                            scale_b_root_ptr,
                            byte_offset=wave_offset_32b
                            + gate_up_half * scaleA_groups * 4,
                            elem_type=T.i8,
                        )
                        scale_b_half_offset = gate_up_half * (N // 256) * scaleA_groups * 4
                        rocdl.raw_ptr_buffer_load_lds(
                            sB_rsrc,
                            scale_b_dst,
                            fx.Int32(4),
                            lane_byte_offset_32b,
                            fx.Int32(scale_b_global_base + scale_b_half_offset),
                            fx.Int32(0),
                            fx.Int32(0),
                        )

            if const_expr(not gate_up and rounds_128b > 0):
                lane_byte_offset_128b = fx.Int32(tid * 16)
                wave_offset_128b = rocdl.readfirstlane(
                    T.i32, arith._to_raw(fx.Int32(wave_id * 64 * 16))
                )
                for copy_round in range_constexpr(rounds_128b):
                    round_elem_offset = copy_round * elems_per_round_128b
                    scale_b_dst = fx.buffer_ops.get_element_ptr(
                        scale_b_root_ptr,
                        byte_offset=wave_offset_128b + round_elem_offset * 4,
                        elem_type=T.i8,
                    )
                    rocdl.raw_ptr_buffer_load_lds(
                        sB_rsrc,
                        scale_b_dst,
                        fx.Int32(16),
                        lane_byte_offset_128b,
                        fx.Int32(scale_b_global_base + round_elem_offset * 4),
                        fx.Int32(0),
                        fx.Int32(0),
                    )

            if const_expr(not gate_up and remaining_elems > 0):
                lane_byte_offset_32b = fx.Int32(tid * 4)
                wave_offset_32b = rocdl.readfirstlane(
                    T.i32, arith._to_raw(fx.Int32(wave_id * 64 * 4))
                )
                for copy_round in range_constexpr(rounds_32b):
                    round_elem_offset = loaded_128b + copy_round * total_lanes
                    scale_b_dst = fx.buffer_ops.get_element_ptr(
                        scale_b_root_ptr,
                        byte_offset=wave_offset_32b + round_elem_offset * 4,
                        elem_type=T.i8,
                    )
                    rocdl.raw_ptr_buffer_load_lds(
                        sB_rsrc,
                        scale_b_dst,
                        fx.Int32(4),
                        lane_byte_offset_32b,
                        fx.Int32(scale_b_global_base + round_elem_offset * 4),
                        fx.Int32(0),
                        fx.Int32(0),
                    )

                if const_expr(tail_elems > 0):
                    if tid < tail_elems:
                        tail_elem_offset = loaded_128b + loaded_32b
                        scale_b_dst = fx.buffer_ops.get_element_ptr(
                            scale_b_root_ptr,
                            byte_offset=wave_offset_32b + tail_elem_offset * 4,
                            elem_type=T.i8,
                        )
                        rocdl.raw_ptr_buffer_load_lds(
                            sB_rsrc,
                            scale_b_dst,
                            fx.Int32(4),
                            lane_byte_offset_32b,
                            fx.Int32(scale_b_global_base + tail_elem_offset * 4),
                            fx.Int32(0),
                            fx.Int32(0),
                        )

            rocdl.s_waitcnt(encode_waitcnt_950(vmcnt=0))
            rocdl.s_barrier()
            rocdl.sched_barrier(0)

        # A/B LDS dual padding write/read layout（参考 gemm_4wave_950 fp8 gluon）
        # fp8采用的是双padding, 每8行padding 16个元素， 每16行额外再padding 32个元素
        # bf16采用的是每8行 padding 16 个元素，
        # bf16和fp8 padding方式不同的主要原因是? 单padding 32个元素应该也可以？
        # bf16 , MFMA16x16x32, 一次DWORDx4 读取的32个， bf16每条lane读128bit, BK分成左右两部分读， 先读32个K， 再读另外的一半，
        # fp8,  MFMA16x16x128, 也是DWORDX4读取，与上面一样
        # todo: try fp8 32 但 padding.

        # WR, RD的layout主要是sub M 的mode transpose,
        # write的 128个M 是 ->(8, 16 groups) , 每个tile写8行， 根据双 padding 又分为-> (8, 2g, 8G)
        # read 的 128个M 是 ->(16 groups, 8),每个tile读16 groups, 双padding 有分为 -> (2g, 8G, 8)

        # =========================== lds : LDS read/write  layout & tensor ===========================
        _wr = fx.make_layout(
            ((8, 2, BLOCK_M // 16), BLOCK_K),
            ((BLOCK_K, 8 * BLOCK_K + 16, 2 * (8 * BLOCK_K + 16) + 32), 1),
        )
        _rd = fx.make_layout(
            ((2, BLOCK_M // 16, 8), (32, BLOCK_K // 32)),
            ((8 * BLOCK_K + 16, 2 * (8 * BLOCK_K + 16) + 32, BLOCK_K), (1, 32)),
        )
        ## global to LDS tile copy:
        # 8 wave g2s DMA tv：512 线程，tile(64, BLOCK_K)，每线程 2 次 128-bit load。
        _a_dma_tv = fx.make_layout(
            ((8, 8, 8), elements_per_128b),
            ((elements_per_128b * 64, 1, 8), 64),
        )
        dma = fx.make_tiled_copy(
            buffer_copy_atom, _a_dma_tv, fx.make_tile(64, BLOCK_K)
        ).get_slice(tid)
        # B LDS wr/rd：preshuffle 时用与 shuffle 一致的无 bank-conflict 布局（wr==rd），
        # 否则沿用与 A 相同的 dual-padding _wr/_rd。
        _wr_b = _wr
        _rd_b = _rd
        if const_expr(preshuffle_b):
            _b_lds = fx.make_layout(
                ((16, BLOCK_N // 16), (16, BLOCK_K // 16)), ((16, 2048), (1, 256))
            )
            _wr_b = _b_lds
            _rd_b = _b_lds
            # B 专属 g2s DMA（512 线程，tile(64,BLOCK_K)）：对标 4-wave 的 ((16,8,2),16),((1,512,16),32)
            # tile(32)，8-wave 行数翻倍 => (16,8,4),(1,1024,16),64 tile(64)。
            _b_g2s_tv = fx.make_layout(
                ((16, 8, 4), elements_per_128b), ((1, 1024, 16), 64)
            )
            dma_b = fx.make_tiled_copy(
                buffer_copy_atom, _b_g2s_tv, fx.make_tile(64, BLOCK_K)
            ).get_slice(tid)
        else:
            dma_b = dma
        # LDS A , read write tensor.
        sA_t_wr = [fx.make_view(lds.a_t0.ptr, _wr), fx.make_view(lds.a_t1.ptr, _wr)]
        sA_b_wr = [fx.make_view(lds.a_b0.ptr, _wr), fx.make_view(lds.a_b1.ptr, _wr)]
        sA_t_rd = [fx.make_view(lds.a_t0.ptr, _rd), fx.make_view(lds.a_t1.ptr, _rd)]
        sA_b_rd = [fx.make_view(lds.a_b0.ptr, _rd), fx.make_view(lds.a_b1.ptr, _rd)]
        # LDS B, read write tensor.
        sB_l_wr = [fx.make_view(lds.b_l0.ptr, _wr_b), fx.make_view(lds.b_l1.ptr, _wr_b)]
        sB_r_wr = [fx.make_view(lds.b_r0.ptr, _wr_b), fx.make_view(lds.b_r1.ptr, _wr_b)]
        sB_l_rd = [fx.make_view(lds.b_l0.ptr, _rd_b), fx.make_view(lds.b_l1.ptr, _rd_b)]
        sB_r_rd = [fx.make_view(lds.b_r0.ptr, _rd_b), fx.make_view(lds.b_r1.ptr, _rd_b)]

        # ============================= g2s: partition global memory and wr LDS ===============================
        aT_g = dma.partition_S(bA_t)
        aB_g = dma.partition_S(bA_b)
        bL_g = dma_b.partition_S(bB_l)
        bR_g = dma_b.partition_S(bB_r)
        aT_s = [dma.partition_D(sA_t_wr[0]), dma.partition_D(sA_t_wr[1])]
        aB_s = [dma.partition_D(sA_b_wr[0]), dma.partition_D(sA_b_wr[1])]
        bL_s = [dma_b.partition_D(sB_l_wr[0]), dma_b.partition_D(sB_l_wr[1])]
        bR_s = [dma_b.partition_D(sB_r_wr[0]), dma_b.partition_D(sB_r_wr[1])]

        # ===================================  s2r: tiled MMA  ============================================
        # ---- tiled MMA: 8 wave (wave grid 4x2)，4M wave , 2N wave , but A, B transposed
        # 所以实际的A, B 时 4 wave on N, 2 waves on M. 最终的实际的 nrM = 4, nrN = 2.
        # MFMA 16x16x128
        # 有4个地方设计A， B tranpose的变化：
        # MMA wave layout on MN, make_tiled_copyA/B, fx.gemm, C tranpose layout.
        mma_atom = fx.make_mma_atom(
            fx.rocdl.cdna4.MFMA_Scale(16, 16, 128, element_type)
        )
        mma_atom = fx.atom_set_value(mma_atom, "scale_a", fx.Int32(0))
        mma_atom = fx.atom_set_value(mma_atom, "scale_b", fx.Int32(0))
        k_perm = fx.make_layout((32, 4), (1, 32))
        tiled_mma = fx.make_tiled_mma(
            mma_atom, fx.make_layout((4, 2, 1), (1, 4, 0)), (None, None, k_perm)
        )

        # ==============================  s2r:MMA tiled partition LDS read tensor  ===========================
        # copy_a use MMA B config to partition bA tensor
        copy_a = fx.make_tiled_copy_B(lds_copy_atom, tiled_mma).get_slice(tid)
        # copy_b use MMA A config to partition bB tensor
        copy_b = fx.make_tiled_copy_A(lds_copy_atom, tiled_mma).get_slice(tid)
        s2r_src0_A_t = copy_a.partition_S(sA_t_rd[0])
        s2r_src0_A_b = copy_a.partition_S(sA_b_rd[0])
        s2r_src0_B_l = copy_b.partition_S(sB_l_rd[0])
        s2r_src0_B_r = copy_b.partition_S(sB_r_rd[0])
        s2r_src1_A_t = copy_a.partition_S(sA_t_rd[1])
        s2r_src1_A_b = copy_a.partition_S(sA_b_rd[1])
        s2r_src1_B_l = copy_b.partition_S(sB_l_rd[1])
        s2r_src1_B_r = copy_b.partition_S(sB_r_rd[1])

        # ==============================  s2r: A, B, C register fragment and retile  ===========================
        thr_mma = tiled_mma.thr_slice(tid)
        frag_A_t = thr_mma.make_fragment_B(sA_t_rd[0])
        frag_B_l = thr_mma.make_fragment_A(sB_l_rd[0])
        frag_B_r = thr_mma.make_fragment_A(sB_r_rd[0])

        dest_frag_A_t = copy_a.retile(frag_A_t)
        dest_frag_B_l = copy_b.retile(frag_B_l)
        dest_frag_B_r = copy_b.retile(frag_B_r)

        # ---- C fragments：转置 tile + make_fragment_C（对标 gemm_8wave_950，无 select）----
        output_block_n = bid_y if const_expr(gate_up) else bid_y * 2
        bC_tl = fx.flat_divide(C, (BLOCK_M, BLOCK_N))[
            None, None, bid_x * 2 + 0, output_block_n
        ]
        bC_tl = fx.composition(
            bC_tl, fx.make_ordered_layout((BLOCK_N, BLOCK_M), (1, 0))
        )

        frag_C_tl = thr_mma.make_fragment_C(bC_tl)
        frag_C_tr = thr_mma.make_fragment_C(bC_tl)
        frag_C_bl = thr_mma.make_fragment_C(bC_tl)
        frag_C_br = thr_mma.make_fragment_C(bC_tl)
        frag_P = thr_mma.make_fragment_C(bC_tl)  # 单级 FIFO partial

        # ==== A/B block-scale 设置：A per-token group-128，B per-128 rows/group-128 ====
        # C[m,n] = sum_kb scaleA[m,kb] * scaleB[n//128,kb] * partial[kb]。
        # scaleA [KB, M] 以 f32 写入 ping-pong LDS，计算 phase 按当前 MFMA 行读回。
        # C fragment 布局 [val=N, n0(N_REP), m0(M_REP)]；M 行 = quadrant_m*128 + m0*32
        #   + wave_m*16 + lane%16（wave_m=wave_id//4）=> scaleA 随 m0/lane 变化，广播 val/n0。
        M_REP = TILE_M // 64
        N_REP = TILE_N // 128
        if const_expr(with_scale):
            sA_rsrc = fx.buffer_ops.create_buffer_resource(
                argScaleA,
                num_records_bytes=arith._to_raw(
                    num_tokens_i32 * fx.Int32(scaleA_groups * 4)
                ),
            )
            lane_id = tid % 64
            wave_m = wave_id // 4
            scale_a_lds = [
                fx.make_view(lds.scale_a0.ptr, fx.make_layout(512, 1)),
                fx.make_view(lds.scale_a1.ptr, fx.make_layout(512, 1)),
            ]
            scale_lds_copy_atom = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32)
            scale_wave_dst_offset = rocdl.readfirstlane(
                T.i32, arith._to_raw(fx.Int32(tid * 4))
            )

            def _scale_dst_ptr(root_view, byte_offset):
                ptr_type = ir.Type.parse("!llvm.ptr<3>")
                root_ptr = _fly_dialect.extract_aligned_pointer_as_index(
                    ptr_type, arith._to_raw(root_view)
                )
                return fx.buffer_ops.get_element_ptr(
                    root_ptr, byte_offset=byte_offset, elem_type=T.i8
                )

            scale_row = tid % TILE_M
            route_mask = arith.constant(TOKEN_MASK, type=T.i32)

            def _ac_scale_a(buf, kb):
                scale_fused_id = _load_route(scale_row)
                scale_token_id = arith.andi(scale_fused_id, route_mask)
                scale_lane_src_offset = fx.Int32(scale_token_id * 4)
                scale_src_base = fx.Int32(kb * numTokens * 4)
                scale_dst = _scale_dst_ptr(scale_a_lds[buf], scale_wave_dst_offset)
                rocdl.raw_ptr_buffer_load_lds(
                    sA_rsrc,
                    scale_dst,
                    fx.Int32(4),
                    scale_lane_src_offset,
                    scale_src_base,
                    fx.Int32(0),
                    fx.Int32(0),
                )

            def _scale_b_addr(kb):
                return fx.Int32(fx.ptrtoint(lds.scale_b.ptr)) + kb * 4

            def _rd_scale_b(addr):
                result_type = ir.Type.parse("!llvm.struct<(f32, f32)>")
                result = _llvm.inline_asm(
                    result_type,
                    [arith._to_raw(addr)],
                    "ds_read_b32 $0, $2\n"
                    f"ds_read_b32 $1, $2 offset:{scaleA_groups * 4}",
                    "=&v,=&v,v,~{memory}",
                    has_side_effects=True,
                )
                return Vec.from_elements(
                    [
                        fx.Float32(_llvm.extractvalue(T.f32, result, [0])),
                        fx.Float32(_llvm.extractvalue(T.f32, result, [1])),
                    ],
                    fx.Float32,
                )

            def _scalarize_scale_b(scales):
                result_type = ir.Type.parse("!llvm.struct<(f32, f32)>")
                result = _llvm.inline_asm(
                    result_type,
                    [arith._to_raw(scales[0]), arith._to_raw(scales[1])],
                    "v_readfirstlane_b32 $0, $2\n" "v_readfirstlane_b32 $1, $3",
                    "=&s,=&s,v,v",
                    has_side_effects=True,
                )
                return Vec.from_elements(
                    [
                        fx.Float32(_llvm.extractvalue(T.f32, result, [0])),
                        fx.Float32(_llvm.extractvalue(T.f32, result, [1])),
                    ],
                    fx.Float32,
                )

            def _rd_scale_a(buf, bottom):
                half_offset = bottom * BLOCK_M
                wave_copy_offset = wave_m * TILE_M
                scales = []
                for m0 in range_constexpr(M_REP):
                    scale_offset = (
                        wave_copy_offset
                        + half_offset
                        + wave_m * 16
                        + lane_id % 16
                        + m0 * 32
                    )
                    scale_src = fx.make_view(
                        fx.add_offset(
                            lds.scale_a0.ptr if buf == 0 else lds.scale_a1.ptr,
                            scale_offset,
                        ),
                        fx.make_layout(1, 1),
                    )
                    scale_frag = fx.make_fragment_like(scale_src)
                    fx.copy(scale_lds_copy_atom, scale_src, scale_frag)
                    scales.append(Vec(scale_frag.load())[0])
                return Vec.from_elements(scales, fx.Float32)

        def do_gemm(frag_C, frag_B, frag_A, prev_scale_a=None, prev_scale_b=None):
            if const_expr(with_scale):
                # 单条 side-effect inline asm 固定 4x scalar FMA -> 1x MFMA，
                # scaled path 不再依赖 LLVM sched_group_barrier 的重排结果。
                result_type = ir.Type.parse(
                    "!llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, "
                    "vector<4xf32>, vector<4xf32>)>"
                )
                for m0 in range_constexpr(M_REP):
                    scale = Vec(prev_scale_a)[m0] * prev_scale_b
                    cs0 = frag_C[None, 0, m0]
                    cs1 = frag_C[None, 1, m0]
                    partial0 = Vec(frag_P[None, 0, m0].load())
                    partial1 = Vec(frag_P[None, 1, m0].load())
                    accum0 = Vec(cs0.load())
                    accum1 = Vec(cs1.load())
                    operand_a0 = vector.bitcast(
                        T.vec(8, T.i32), frag_B[None, 0, 0].load()
                    )
                    operand_a1 = vector.bitcast(
                        T.vec(8, T.i32), frag_B[None, 1, 0].load()
                    )
                    operand_b = vector.bitcast(
                        T.vec(8, T.i32), frag_A[None, m0, 0].load()
                    )
                    result = _llvm.inline_asm(
                        result_type,
                        [
                            arith._to_raw(partial0[0]),
                            arith._to_raw(partial0[1]),
                            arith._to_raw(partial0[2]),
                            arith._to_raw(partial0[3]),
                            arith._to_raw(partial1[0]),
                            arith._to_raw(partial1[1]),
                            arith._to_raw(partial1[2]),
                            arith._to_raw(partial1[3]),
                            arith._to_raw(scale),
                            arith._to_raw(accum0[0]),
                            arith._to_raw(accum0[1]),
                            arith._to_raw(accum0[2]),
                            arith._to_raw(accum0[3]),
                            arith._to_raw(accum1[0]),
                            arith._to_raw(accum1[1]),
                            arith._to_raw(accum1[2]),
                            arith._to_raw(accum1[3]),
                            arith._to_raw(operand_a0),
                            arith._to_raw(operand_a1),
                            arith._to_raw(operand_b),
                        ],
                        "v_fmac_f32 $0, $10, $18\n"
                        "v_fmac_f32 $1, $11, $18\n"
                        "v_fmac_f32 $2, $12, $18\n"
                        "v_fmac_f32 $3, $13, $18\n"
                        "v_mfma_f32_16x16x128_f8f6f4 $8, $27, $29, 0\n"
                        "v_fmac_f32 $4, $14, $18\n"
                        "v_fmac_f32 $5, $15, $18\n"
                        "v_fmac_f32 $6, $16, $18\n"
                        "v_fmac_f32 $7, $17, $18\n"
                        "v_mfma_f32_16x16x128_f8f6f4 $9, $28, $29, 0",
                        "=&v,=&v,=&v,=&v,=&v,=&v,=&v,=&v,=&v,=&v,"
                        "v,v,v,v,v,v,v,v,v,0,1,2,3,4,5,6,7,v,v,v",
                        has_side_effects=True,
                    )
                    cs0.store(
                        Vec.from_elements(
                            [
                                fx.Float32(_llvm.extractvalue(T.f32, result, [0])),
                                fx.Float32(_llvm.extractvalue(T.f32, result, [1])),
                                fx.Float32(_llvm.extractvalue(T.f32, result, [2])),
                                fx.Float32(_llvm.extractvalue(T.f32, result, [3])),
                            ],
                            fx.Float32,
                        )
                    )
                    cs1.store(
                        Vec.from_elements(
                            [
                                fx.Float32(_llvm.extractvalue(T.f32, result, [4])),
                                fx.Float32(_llvm.extractvalue(T.f32, result, [5])),
                                fx.Float32(_llvm.extractvalue(T.f32, result, [6])),
                                fx.Float32(_llvm.extractvalue(T.f32, result, [7])),
                            ],
                            fx.Float32,
                        )
                    )
                    frag_P[None, 0, m0].store(
                        _llvm.extractvalue(T.vec(4, T.f32), result, [8])
                    )
                    frag_P[None, 1, m0].store(
                        _llvm.extractvalue(T.vec(4, T.f32), result, [9])
                    )
            else:
                # # 单级 FIFO：先消费上一 phase 的 partial，再用当前 MFMA 覆盖 FIFO。
                # for m0 in range_constexpr(M_REP):
                #     for n0 in range_constexpr(N_REP):
                #         cs = frag_C[None, n0, m0]
                #         cs.store(cs.load() + frag_P[None, n0, m0].load())
                # frag_P.fill(0)
                # fx.gemm(mma_atom, frag_P, frag_B, frag_A, frag_P)

                fx.gemm(mma_atom, frag_C, frag_B, frag_A, frag_C)

        num_tiles = K // BLOCK_K
        assert num_tiles % 2 == 0
        a_dsrd = frag_A_t.load().numel * element_type.width // 8 // 16
        b_dsrd = frag_B_l.load().numel * element_type.width // 8 // 16
        a_vmem = (BLOCK_M * BLOCK_K * element_type.width // 8) // (512 * 16)
        b_vmem = (BLOCK_N * BLOCK_K * element_type.width // 8) // (512 * 16)

        def begin_compute_phase():
            rocdl.sched_barrier(0)
            rocdl.s_barrier()
            rocdl.sched_barrier(0)
            rocdl.s_waitcnt(encode_waitcnt_950(lgkmcnt=0))
            rocdl.s_setprio(1)
            rocdl.sched_barrier(0)

        def end_compute_phase():
            rocdl.sched_barrier(0)
            rocdl.s_setprio(0)
            rocdl.sched_barrier(0)
            rocdl.s_barrier()
            rocdl.sched_barrier(0)

        def wait_vmem_barrier(vmcnt):
            rocdl.s_waitcnt(encode_waitcnt_950(vmcnt=vmcnt, lgkmcnt=0))
            rocdl.s_barrier()
            rocdl.sched_barrier(0)

        # ---- prologue：预取 tile0/tile1 到 LDS buf0/buf1，再 s2r buf0 的 A_t/B_l ----
        def do_g2s(kk, buf):
            ki = fx.Int32(kk)
            fx.copy(async_copy_atom, bL_g[None, None, None, ki], bL_s[buf])
            rocdl.sched_barrier(0)
            fx.copy(async_copy_atom, aT_g[None, None, None, ki], aT_s[buf])
            rocdl.sched_barrier(0)
            fx.copy(async_copy_atom, aB_g[None, None, None, ki], aB_s[buf])
            rocdl.sched_barrier(0)

            fx.copy(async_copy_atom, bR_g[None, None, None, ki], bR_s[buf])
            rocdl.sched_barrier(0)

        # ---- 非 scale 版：对标 pyhip gemm_8wave 的 4-phase 精确流水 ----
        # 每 tile 分 4 个 compute-phase（TL/TR/BL/BR，各一条 MFMA），phase 间穿插一次
        # ds_read + 一条 g2s 预取（读后即刷 LDS，barrier 保证全 wave 读完再覆盖）。
        # vmcnt 用精确值（a_vmem + 2*b_vmem）而非全 drain，让 g2s 与 MFMA 重叠。
        NS_VMCNT = a_vmem + 2 * b_vmem
        _lgkm0 = encode_waitcnt_950(lgkmcnt=0)
        _s2r_At = [s2r_src0_A_t, s2r_src1_A_t]
        _s2r_Ab = [s2r_src0_A_b, s2r_src1_A_b]
        _s2r_Bl = [s2r_src0_B_l, s2r_src1_B_l]
        _s2r_Br = [s2r_src0_B_r, s2r_src1_B_r]

        def _rd_At(b):
            fx.copy(lds_copy_atom, _s2r_At[b], dest_frag_A_t, pred=None)

        def _rd_Ab(b):
            fx.copy(lds_copy_atom, _s2r_Ab[b], dest_frag_A_t, pred=None)

        def _rd_Bl(b):
            fx.copy(lds_copy_atom, _s2r_Bl[b], dest_frag_B_l, pred=None)

        def _rd_Br(b):
            fx.copy(lds_copy_atom, _s2r_Br[b], dest_frag_B_r, pred=None)

        # ---- 非 scale 版 raw scalar-pointer LDS DMA（对标 pyhip gemm_8wave_950 fp8 raw 路径）----
        # 把每条 g2s 的 LDS 目的地址从 vector(v_readfirstlane->m0) 改为每 wave 只算一次的
        # scalar 基址 + 编译期 static chunk 偏移，消除 v130-v147 这批地址 VGPR。
        # 前提：本 kernel 的 _wr/_rd 与 pyhip a_lds_write/read_layout 逐字节一致，且非
        # preshuffle 的 B 与 A 对称（grouped-row 全局 + dual-padding LDS），故 A 的 raw
        # 路径可直接复用到 B（仅 base row 换成 bid_y*TILE_N）。
        if const_expr(not useTileDMA):
            _elem_bytes = element_type.width // 8  # fp8 = 1

            def _dma_dst_ptr(root_view, byte_offset):
                _pt = ir.Type.parse("!llvm.ptr<3>")
                _rp = _fly_dialect.extract_aligned_pointer_as_index(
                    _pt, arith._to_raw(root_view)
                )
                return fx.buffer_ops.get_element_ptr(
                    _rp, byte_offset=byte_offset, elem_type=T.i8
                )

            # pyhip 的 g2s tv（512 线程 = 64 行 × 8 k-组，每线程 1×16 fp8）
            _g2s_tile, _g2s_tv = fx.make_layout_tv(
                fx.make_layout((8 * 8, 8), (8, 1)),
                fx.make_layout((1, elements_per_128b), (1, 1)),
            )
            _copy_g2s = fx.make_tiled_copy(
                buffer_copy_atom, _g2s_tv, _g2s_tile
            ).get_slice(tid)
            _dst_stride = _copy_g2s.partition_D(sA_t_wr[0]).stride[1].to_py_value()

            # 每 wave 的 LDS 基址（dual-padding group），readfirstlane 一次
            _a_wave_off_elems = wave_id % 2 * (8 * BLOCK_K + 16) + wave_id // 2 * (
                2 * (8 * BLOCK_K + 16) + 32
            )
            _a_wave_off_bytes = rocdl.readfirstlane(
                T.i32, arith._to_raw(fx.Int32(_a_wave_off_elems * _elem_bytes))
            )
            _b_wave_off_bytes = _a_wave_off_bytes  # 非 preshuffle B 与 A 同布局

            _aT_dst = [
                _dma_dst_ptr(sA_t_wr[0], _a_wave_off_bytes),
                _dma_dst_ptr(sA_t_wr[1], _a_wave_off_bytes),
            ]
            _aB_dst = [
                _dma_dst_ptr(sA_b_wr[0], _a_wave_off_bytes),
                _dma_dst_ptr(sA_b_wr[1], _a_wave_off_bytes),
            ]
            _bL_dst = [
                _dma_dst_ptr(sB_l_wr[0], _b_wave_off_bytes),
                _dma_dst_ptr(sB_l_wr[1], _b_wave_off_bytes),
            ]
            _bR_dst = [
                _dma_dst_ptr(sB_r_wr[0], _b_wave_off_bytes),
                _dma_dst_ptr(sB_r_wr[1], _b_wave_off_bytes),
            ]

            # 每 thread 的 (row, k) 源映射（对标 pyhip a_lane_row = tid//8）
            _a_lane_row = tid // 8
            _a_lane_k = tid % 8 * elements_per_128b
            _a_local_row = _a_lane_row % 8 * (BLOCK_M // 8) + _a_lane_row // 8
            _lane_k_byte_offset = fx.Int32(_a_lane_k * _elem_bytes)
            if const_expr(gate_up):
                _bL_src_wave_base = fx.Int32(bid_y * BLOCK_N * K * _elem_bytes)
                _bR_src_wave_base = fx.Int32(
                    (N // 2 + bid_y * BLOCK_N) * K * _elem_bytes
                )
            else:
                _bL_src_wave_base = fx.Int32(bid_y * TILE_N * K * _elem_bytes)
                _bR_src_wave_base = _bL_src_wave_base + BLOCK_N * K * _elem_bytes

            def _raw_a_g2s(dst_base, row_half, ki):
                for chunk in range_constexpr(BLOCK_M // 64):
                    row_local = row_half * BLOCK_M + _a_local_row + chunk * 8
                    _dp = fx.buffer_ops.get_element_ptr(
                        dst_base,
                        static_byte_offset=chunk * _dst_stride * _elem_bytes,
                        elem_type=T.i8,
                    )
                    _vo = _load_a_row_offset(row_local) + _lane_k_byte_offset
                    _so = fx.Int32(ki * BLOCK_K * _elem_bytes)
                    rocdl.raw_ptr_buffer_load_lds(
                        a_dma_rsrc,
                        _dp,
                        fx.Int32(16),
                        _vo,
                        _so,
                        fx.Int32(0),
                        fx.Int32(0),
                    )

            def _raw_b_g2s(dst_base, src_wave_base, ki):
                for chunk in range_constexpr(BLOCK_M // 64):
                    _dp = fx.buffer_ops.get_element_ptr(
                        dst_base,
                        static_byte_offset=chunk * _dst_stride * _elem_bytes,
                        elem_type=T.i8,
                    )
                    _so = src_wave_base + fx.Int32(
                        ki * BLOCK_K * _elem_bytes + chunk * 8 * K * _elem_bytes
                    )
                    rocdl.raw_ptr_buffer_load_lds(
                        b_dma_rsrc,
                        _dp,
                        fx.Int32(16),
                        fx.Int32((_a_local_row * K + _a_lane_k) * _elem_bytes),
                        _so,
                        fx.Int32(0),
                        fx.Int32(0),
                    )

        def _ac_At(b, ki):
            if const_expr(not useTileDMA):
                _raw_a_g2s(_aT_dst[b], 0, ki)
            else:
                fx.copy(async_copy_atom, aT_g[None, None, None, ki], aT_s[b])

        def _ac_Ab(b, ki):
            if const_expr(not useTileDMA):
                _raw_a_g2s(_aB_dst[b], 1, ki)
            else:
                fx.copy(async_copy_atom, aB_g[None, None, None, ki], aB_s[b])

        def _ac_Bl(b, ki):
            if const_expr(not useTileDMA):
                _raw_b_g2s(_bL_dst[b], _bL_src_wave_base, ki)
            else:
                fx.copy(async_copy_atom, bL_g[None, None, None, ki], bL_s[b])

        def _ac_Br(b, ki):
            if const_expr(not useTileDMA):
                _raw_b_g2s(_bR_dst[b], _bR_src_wave_base, ki)
            else:
                fx.copy(async_copy_atom, bR_g[None, None, None, ki], bR_s[b])

        rocdl.sched_barrier(0)
        if const_expr(with_scale):
            _ac_scale_a(0, fx.Int32(0))
            rocdl.sched_barrier(0)
        _ac_Bl(0, 0)
        rocdl.sched_barrier(0)
        _ac_At(0, 0)
        rocdl.sched_barrier(0)
        _ac_Br(0, 0)
        rocdl.sched_barrier(0)
        _ac_Ab(0, 0)
        rocdl.sched_barrier(0)
        if wave_id >= 4:
            rocdl.s_barrier()
        frag_C_tl.fill(0)
        frag_C_tr.fill(0)
        frag_C_bl.fill(0)
        frag_C_br.fill(0)

        vm_load_cnt_a = 2
        vm_load_cnt_b = 2
        vm_load_cnt_scale_a = 1 if const_expr(with_scale) else 0

        rocdl.sched_barrier(0)
        vmcnt = vm_load_cnt_a + vm_load_cnt_b
        rocdl.s_waitcnt(encode_waitcnt_950(vmcnt=vmcnt))
        rocdl.s_barrier()
        rocdl.sched_barrier(0)

        rocdl.sched_barrier(0)
        if const_expr(with_scale):
            _ac_scale_a(1, fx.Int32(1))
            rocdl.sched_barrier(0)
        _ac_At(1, 1)
        rocdl.sched_barrier(0)
        _ac_Bl(1, 1)
        rocdl.sched_barrier(0)
        _ac_Br(1, 1)
        rocdl.sched_barrier(0)

        vmcnt = vm_load_cnt_a + vm_load_cnt_b * 2 + vm_load_cnt_scale_a
        rocdl.s_waitcnt(encode_waitcnt_950(vmcnt=vmcnt))
        rocdl.s_barrier()
        rocdl.sched_barrier(0)

        if const_expr(with_scale):
            frag_P.fill(0)
            acc_init = [
                frag_C_tl.load(),
                frag_C_tr.load(),
                frag_C_bl.load(),
                frag_C_br.load(),
                frag_P.load(),
                Vec.from_elements(
                    [fx.Float32(0), fx.Float32(0), fx.Float32(0), fx.Float32(0)],
                    fx.Float32,
                ),
                fx.Float32(0),
            ]
        else:
            acc_init = [
                frag_C_tl.load(),
                frag_C_tr.load(),
                frag_C_bl.load(),
                frag_C_br.load(),
            ]

        for kidx, states in range(0, num_tiles, 2, init=acc_init):
            frag_C_tl.store(states[0])
            frag_C_tr.store(states[1])
            frag_C_bl.store(states[2])
            frag_C_br.store(states[3])
            if const_expr(with_scale):
                frag_P.store(states[4])
                fifo_scale_a_0 = Vec.from_elements(
                    [fx.Float32(0), fx.Float32(0), fx.Float32(0), fx.Float32(0)],
                    fx.Float32,
                )
                fifo_scale_b_0 = fx.Float32(0)
                fifo_scale_a_1 = Vec(states[5])
                fifo_scale_b_1 = fx.Float32(states[6])
            kiter = fx.Int32(kidx)
            if const_expr(with_scale):
                scale_b_addr_0 = _scale_b_addr(kiter)
                scale_b_addr_1 = _scale_b_addr(kiter + 1)

            if const_expr(True):
                tick = 0
                tock = 1
                _rd_Bl(tick)
                _rd_At(tick)
                if const_expr(with_scale):
                    mfma_scaleA = _rd_scale_a(tick, 0)
                    mfma_scaleB = _rd_scale_b(scale_b_addr_0)
                _ac_Ab(tock, kiter + 1)
                rocdl.sched_barrier(0)
                rocdl.s_waitcnt(encode_waitcnt_950(lgkmcnt=0))
                rocdl.sched_barrier(0)
                if const_expr(with_scale):
                    mfma_scaleB = _scalarize_scale_b(mfma_scaleB)

                begin_compute_phase()
                if const_expr(with_scale):
                    fifo_scale_a_0, fifo_scale_b_0 = mfma_scaleA, mfma_scaleB[0]
                    do_gemm(
                        frag_C_br, frag_B_l, frag_A_t, fifo_scale_a_1, fifo_scale_b_1
                    )
                else:
                    do_gemm(frag_C_tl, frag_B_l, frag_A_t)
                end_compute_phase()

                _rd_Br(tick)
                _ac_At(tick, kiter + 2)

                begin_compute_phase()
                if const_expr(with_scale):
                    fifo_scale_a_1, fifo_scale_b_1 = mfma_scaleA, mfma_scaleB[1]
                    do_gemm(
                        frag_C_tl, frag_B_r, frag_A_t, fifo_scale_a_0, fifo_scale_b_0
                    )
                else:
                    do_gemm(frag_C_tr, frag_B_r, frag_A_t)
                end_compute_phase()

                _rd_Ab(tick)
                if const_expr(with_scale):
                    mfma_scaleA = _rd_scale_a(tick, 1)
                _ac_Bl(tick, kiter + 2)

                begin_compute_phase()
                if const_expr(with_scale):
                    fifo_scale_a_0, fifo_scale_b_0 = mfma_scaleA, mfma_scaleB[0]
                    do_gemm(
                        frag_C_tr, frag_B_l, frag_A_t, fifo_scale_a_1, fifo_scale_b_1
                    )
                else:
                    do_gemm(frag_C_bl, frag_B_l, frag_A_t)
                end_compute_phase()

                _ac_Br(tick, kiter + 2)
                if const_expr(with_scale):
                    _ac_scale_a(tick, kiter + 2)
                rocdl.s_waitcnt(
                    encode_waitcnt_950(
                        vmcnt=vm_load_cnt_a + vm_load_cnt_b * 2 + vm_load_cnt_scale_a
                    )
                )

                begin_compute_phase()
                if const_expr(with_scale):
                    fifo_scale_a_1, fifo_scale_b_1 = mfma_scaleA, mfma_scaleB[1]
                    do_gemm(
                        frag_C_bl, frag_B_r, frag_A_t, fifo_scale_a_0, fifo_scale_b_0
                    )
                else:
                    do_gemm(frag_C_br, frag_B_r, frag_A_t)
                end_compute_phase()

                tick = 1
                tock = 0

                _rd_Bl(tick)
                _rd_At(tick)
                if const_expr(with_scale):
                    mfma_scaleA = _rd_scale_a(tick, 0)
                    mfma_scaleB = _rd_scale_b(scale_b_addr_1)
                _ac_Ab(tock, kiter + 2)
                rocdl.sched_barrier(0)
                rocdl.s_waitcnt(encode_waitcnt_950(lgkmcnt=0))
                rocdl.sched_barrier(0)
                if const_expr(with_scale):
                    mfma_scaleB = _scalarize_scale_b(mfma_scaleB)

                begin_compute_phase()
                if const_expr(with_scale):
                    fifo_scale_a_0, fifo_scale_b_0 = mfma_scaleA, mfma_scaleB[0]
                    do_gemm(
                        frag_C_br, frag_B_l, frag_A_t, fifo_scale_a_1, fifo_scale_b_1
                    )
                else:
                    do_gemm(frag_C_tl, frag_B_l, frag_A_t)
                end_compute_phase()

                _rd_Br(tick)
                _ac_At(tick, kiter + 3)

                begin_compute_phase()
                if const_expr(with_scale):
                    fifo_scale_a_1, fifo_scale_b_1 = mfma_scaleA, mfma_scaleB[1]
                    do_gemm(
                        frag_C_tl, frag_B_r, frag_A_t, fifo_scale_a_0, fifo_scale_b_0
                    )
                else:
                    do_gemm(frag_C_tr, frag_B_r, frag_A_t)
                end_compute_phase()

                _rd_Ab(tick)
                if const_expr(with_scale):
                    mfma_scaleA = _rd_scale_a(tick, 1)
                _ac_Bl(tick, kiter + 3)

                begin_compute_phase()
                if const_expr(with_scale):
                    fifo_scale_a_0, fifo_scale_b_0 = mfma_scaleA, mfma_scaleB[0]
                    do_gemm(
                        frag_C_tr, frag_B_l, frag_A_t, fifo_scale_a_1, fifo_scale_b_1
                    )
                else:
                    do_gemm(frag_C_bl, frag_B_l, frag_A_t)
                end_compute_phase()

                _ac_Br(tick, kiter + 3)
                if const_expr(with_scale):
                    _ac_scale_a(tick, kiter + 3)
                rocdl.s_waitcnt(
                    encode_waitcnt_950(
                        vmcnt=vm_load_cnt_a + vm_load_cnt_b * 2 + vm_load_cnt_scale_a
                    )
                )

                begin_compute_phase()
                if const_expr(with_scale):
                    fifo_scale_a_1, fifo_scale_b_1 = mfma_scaleA, mfma_scaleB[1]
                    do_gemm(
                        frag_C_bl, frag_B_r, frag_A_t, fifo_scale_a_0, fifo_scale_b_0
                    )
                else:
                    do_gemm(frag_C_br, frag_B_r, frag_A_t)
                end_compute_phase()
            if const_expr(with_scale):
                yield_values = [
                    frag_C_tl.load(),
                    frag_C_tr.load(),
                    frag_C_bl.load(),
                    frag_C_br.load(),
                    frag_P.load(),
                    fifo_scale_a_1,
                    fifo_scale_b_1,
                ]
            else:
                yield_values = [
                    frag_C_tl.load(),
                    frag_C_tr.load(),
                    frag_C_bl.load(),
                    frag_C_br.load(),
                ]
            results = yield yield_values

        frag_C_tl.store(results[0])
        frag_C_tr.store(results[1])
        frag_C_bl.store(results[2])
        frag_C_br.store(results[3])

        c_store_rsrc = fx.buffer_ops.create_buffer_resource(
            argC,
            num_records_bytes=arith._to_raw(
                num_tokens_i32 * fx.Int32(TOPK * output_N * 2)
            ),
        )
        bC_tr = fx.flat_divide(C, (BLOCK_M, BLOCK_N))[
            None, None, bid_x * 2 + 0, bid_y * 2 + 1
        ]
        bC_bl = fx.flat_divide(C, (BLOCK_M, BLOCK_N))[
            None, None, bid_x * 2 + 1, bid_y * 2 + 0
        ]
        bC_br = fx.flat_divide(C, (BLOCK_M, BLOCK_N))[
            None, None, bid_x * 2 + 1, bid_y * 2 + 1
        ]
        transposed_c_layout = fx.make_ordered_layout((BLOCK_N, BLOCK_M), (1, 0))
        bC_tr = fx.composition(bC_tr, transposed_c_layout)
        bC_bl = fx.composition(bC_bl, transposed_c_layout)
        bC_br = fx.composition(bC_br, transposed_c_layout)

        if const_expr(with_scale):
            frag_P.store(results[4])
            fifo_scale_a_1 = Vec(results[5])
            fifo_scale_b_1 = fx.Float32(results[6])
            for m0 in range_constexpr(M_REP):
                for n0 in range_constexpr(N_REP):
                    cs = frag_C_br[None, n0, m0]
                    scale = Vec(fifo_scale_a_1)[m0] * fifo_scale_b_1
                    scale_vec = Vec.from_elements(
                        [scale, scale, scale, scale], fx.Float32
                    )
                    cs.store(fx.fma(frag_P[None, n0, m0].load(), scale_vec, cs.load()))
        if wave_id < 4:
            rocdl.s_barrier()

        # ---- epilogue store ----
        N_tail = N % TILE_N != 0
        if const_expr(gate_up):
            pair_type = ir.Type.parse("!llvm.struct<(i32, i32)>")
            lane_id = tid % 64
            wave_m = wave_id // 4
            wave_n = wave_id % 4
            lane_group = lane_id // 16

            def silu_mul(gate, up):
                exponent = rocdl.exp2(
                    T.f32,
                    arith._to_raw(gate * fx.Float32(-1.4426950408889634)),
                )
                sigmoid = rocdl.rcp(T.f32, fx.Float32(1.0) + exponent)
                return gate * sigmoid * up

            def store_gateup(gate_frag, up_frag, row_half):
                for row_repeat in range_constexpr(M_REP):
                    for col_repeat in range_constexpr(0, N_REP, 2):
                        gate_a = Vec(gate_frag[None, col_repeat, row_repeat].load())
                        gate_b = Vec(
                            gate_frag[None, col_repeat + 1, row_repeat].load()
                        )
                        up_a = Vec(up_frag[None, col_repeat, row_repeat].load())
                        up_b = Vec(up_frag[None, col_repeat + 1, row_repeat].load())
                        out_a = Vec.from_elements(
                            [
                                silu_mul(gate_a[i], up_a[i])
                                for i in range_constexpr(4)
                            ],
                            fx.Float32,
                        )
                        out_b = Vec.from_elements(
                            [
                                silu_mul(gate_b[i], up_b[i])
                                for i in range_constexpr(4)
                            ],
                            fx.Float32,
                        )
                        d0_a = rocdl.cvt_pk_bf16_f32(out_a[0], out_a[1])
                        d1_a = rocdl.cvt_pk_bf16_f32(out_a[2], out_a[3])
                        d0_b = rocdl.cvt_pk_bf16_f32(out_b[0], out_b[1])
                        d1_b = rocdl.cvt_pk_bf16_f32(out_b[2], out_b[3])
                        swap0 = rocdl.permlane16_swap(
                            pair_type,
                            arith._to_raw(d0_a),
                            arith._to_raw(d0_b),
                            False,
                            False,
                        )
                        swap1 = rocdl.permlane16_swap(
                            pair_type,
                            arith._to_raw(d1_a),
                            arith._to_raw(d1_b),
                            False,
                            False,
                        )
                        packed = Vec.from_elements(
                            [
                                fx.Int32(_llvm.extractvalue(T.i32, swap0, [0])),
                                fx.Int32(_llvm.extractvalue(T.i32, swap1, [0])),
                                fx.Int32(_llvm.extractvalue(T.i32, swap0, [1])),
                                fx.Int32(_llvm.extractvalue(T.i32, swap1, [1])),
                            ],
                            fx.Int32,
                        )
                        row_local = (
                            row_half * BLOCK_M
                            + row_repeat * 32
                            + wave_m * 16
                            + lane_id % 16
                        )
                        sorted_row = bid_x_i32 * TILE_M + fx.Int32(row_local)
                        fused_id = _load_route(row_local)
                        token_id = arith.andi(
                            fused_id, arith.constant(TOKEN_MASK, type=T.i32)
                        )
                        slot_id = fused_id >> 24
                        route_valid = arith.andi(
                            arith.cmpi(CmpIPredicate.ult, sorted_row, num_valid_ids),
                            arith.andi(
                                arith.cmpi(
                                    CmpIPredicate.ult, token_id, num_tokens_i32
                                ),
                                arith.cmpi(
                                    CmpIPredicate.ult,
                                    slot_id,
                                    arith.constant(TOPK, type=T.i32),
                                ),
                            ),
                        )
                        col = (
                            bid_y * BLOCK_N
                            + col_repeat * 64
                            + lane_group % 2 * 64
                            + wave_n * 16
                            + lane_group // 2 * 8
                        )
                        output_element = (
                            fx.Int32(token_id) * TOPK + fx.Int32(slot_id)
                        ) * output_N + fx.Int32(col)
                        fx.buffer_ops.buffer_store(
                            packed,
                            c_store_rsrc,
                            output_element * 2,
                            offset_is_bytes=True,
                            mask=route_valid,
                        )

            store_gateup(frag_C_tl, frag_C_tr, 0)
            store_gateup(frag_C_bl, frag_C_br, 1)
        elif const_expr((permlane_epilogue or N_tail) and TILE_N % 256 == 0):
            pair_type = ir.Type.parse("!llvm.struct<(i32, i32)>")
            lane_id = tid % 64
            wave_m = wave_id // 4
            wave_n = wave_id % 4
            lane_group = lane_id // 16
            fragment_mode_0_repeat = TILE_N // 128
            fragment_mode_1_repeat = TILE_M // 64

            def store_c_quadrant(c_frag, quadrant_m, quadrant_n):
                for row_repeat in range_constexpr(fragment_mode_1_repeat):
                    for col_repeat in range_constexpr(0, fragment_mode_0_repeat, 2):
                        acc_a = Vec(c_frag[None, col_repeat, row_repeat].load())
                        acc_b = Vec(c_frag[None, col_repeat + 1, row_repeat].load())
                        d0_a = rocdl.cvt_pk_bf16_f32(acc_a[0], acc_a[1])
                        d1_a = rocdl.cvt_pk_bf16_f32(acc_a[2], acc_a[3])
                        d0_b = rocdl.cvt_pk_bf16_f32(acc_b[0], acc_b[1])
                        d1_b = rocdl.cvt_pk_bf16_f32(acc_b[2], acc_b[3])
                        swap0 = rocdl.permlane16_swap(
                            pair_type,
                            arith._to_raw(d0_a),
                            arith._to_raw(d0_b),
                            False,
                            False,
                        )
                        swap1 = rocdl.permlane16_swap(
                            pair_type,
                            arith._to_raw(d1_a),
                            arith._to_raw(d1_b),
                            False,
                            False,
                        )
                        packed = Vec.from_elements(
                            [
                                fx.Int32(_llvm.extractvalue(T.i32, swap0, [0])),
                                fx.Int32(_llvm.extractvalue(T.i32, swap1, [0])),
                                fx.Int32(_llvm.extractvalue(T.i32, swap0, [1])),
                                fx.Int32(_llvm.extractvalue(T.i32, swap1, [1])),
                            ],
                            fx.Int32,
                        )
                        row_local = (
                            quadrant_m * (TILE_M // 2)
                            + row_repeat * 32
                            + wave_m * 16
                            + lane_id % 16
                        )
                        sorted_row = bid_x_i32 * TILE_M + fx.Int32(row_local)
                        fused_id = _load_route(row_local)
                        token_id = arith.andi(
                            fused_id, arith.constant(TOKEN_MASK, type=T.i32)
                        )
                        slot_id = fused_id >> 24
                        token_valid = arith.cmpi(
                            CmpIPredicate.ult, token_id, num_tokens_i32
                        )
                        slot_valid = arith.cmpi(
                            CmpIPredicate.ult,
                            slot_id,
                            arith.constant(TOPK, type=T.i32),
                        )
                        sorted_valid = arith.cmpi(
                            CmpIPredicate.ult, sorted_row, num_valid_ids
                        )
                        route_valid = arith.andi(
                            sorted_valid, arith.andi(token_valid, slot_valid)
                        )
                        col = (
                            bid_y * TILE_N
                            + quadrant_n * (TILE_N // 2)
                            + col_repeat * 64
                            + lane_group % 2 * 64
                            + wave_n * 16
                            + lane_group // 2 * 8
                        )
                        output_element = (
                            fx.Int32(token_id) * TOPK + fx.Int32(slot_id)
                        ) * N + fx.Int32(col)
                        if const_expr(N_tail):
                            route_valid = arith.andi(route_valid, col < N)
                        fx.buffer_ops.buffer_store(
                            packed,
                            c_store_rsrc,
                            output_element * 2,
                            offset_is_bytes=True,
                            mask=route_valid,
                        )

            store_c_quadrant(frag_C_tl, 0, 0)
            store_c_quadrant(frag_C_tr, 0, 1)
            store_c_quadrant(frag_C_bl, 1, 0)
            store_c_quadrant(frag_C_br, 1, 1)
        else:
            assert (
                N % TILE_N == 0
            ), "N must be a multiple of TILE_N for permlane_epilogue=False, not supported for now"
            c_frag_bf16 = fx.make_fragment_like(frag_C_tl, dtype=fx.BFloat16)
            store_atom = fx.make_copy_atom(fx.rocdl.BufferCopy64b(), fx.BFloat16)
            store_thr = fx.make_tiled_copy_C(store_atom, tiled_mma).get_slice(tid)

            def store_c_quadrant(c_frag, bC):
                c_frag_bf16.store(c_frag.load().to(fx.BFloat16))
                fx.copy(
                    store_atom, store_thr.retile(c_frag_bf16), store_thr.partition_D(bC)
                )

            store_c_quadrant(frag_C_tl, bC_tl)
            store_c_quadrant(frag_C_tr, bC_tr)
            store_c_quadrant(frag_C_bl, bC_bl)
            store_c_quadrant(frag_C_br, bC_br)

    @flyc.jit
    def launch_moe_stage1(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
        scaleA: fx.Tensor,
        scaleB: fx.Tensor,
        sortedIds: fx.Tensor,
        expertIds: fx.Tensor,
        numValidIds: fx.Tensor,
        numTokens: int,
        numExpertBlocks: int,
        stream: fx.Stream = fx.Stream(None),
    ):
        moe_stage1_kernel(
            A,
            B,
            C,
            scaleA,
            scaleB,
            sortedIds,
            expertIds,
            numValidIds,
            numTokens,
            numExpertBlocks,
        ).launch(
            grid=(div_up(N, TILE_N) * numExpertBlocks, 1, 1),
            block=(512, 1, 1),
            stream=stream,
        )

    return launch_moe_stage1


# =========================== test / perf ===========================
TILE_M = 256
TILE_N = 256
TILE_K = 128
PERMLANE_EPILOGUE = _env_flag("PERMLANE", "1")

import pyhip


def permute_scale_a(scale_a):
    return scale_a.transpose(0, 1).contiguous()


def run_accuracy_case(
    tokens=128, intermediate_size=128, hidden_size=256, topk=2, num_experts=2
):
    from test_moe_mxfp8_mxfp4_gateup_4w import make_balanced_routing

    N = 2 * intermediate_size
    K = hidden_size
    KB = K // 128
    if N % TILE_N != 0 or K % 256 != 0:
        raise ValueError(
            "2*intermediate_size must divide TILE_N and hidden_size must be a multiple of 256"
        )

    topk_ids, _, sorted_ids, _, expert_ids, num_valid_ids = make_balanced_routing(
        tokens, topk, num_experts, hidden_size, device="cuda"
    )
    A = torch.randint(-2, 3, (tokens, K), device="cuda", dtype=torch.int8).to(
        torch.float8_e4m3fn
    )
    B = torch.randint(-2, 3, (num_experts, N, K), device="cuda", dtype=torch.int8).to(
        torch.float8_e4m3fn
    )
    scaleA = torch.rand((tokens, KB), device="cuda", dtype=torch.float32)
    scaleB = torch.rand(
        (num_experts, div_up(N, 128), KB), device="cuda", dtype=torch.float32
    )
    output = torch.full(
        (tokens, topk, N), float("nan"), device="cuda", dtype=torch.bfloat16
    )

    A_dequant = (A.float().view(tokens, KB, 128) * scaleA[:, :, None]).view(tokens, K)
    B_dequant = (
        B.float().view(num_experts, N, KB, 128)
        * scaleB.repeat_interleave(128, dim=1)[:, :N, :, None]
    ).view(num_experts, N, K)
    reference = torch.empty_like(output)
    for slot in range(topk):
        routed_weight = B_dequant[topk_ids[:, slot].to(torch.int64)]
        reference[:, slot] = torch.einsum("tk,tnk->tn", A_dequant, routed_weight).to(
            torch.bfloat16
        )

    stream = torch.cuda.current_stream()
    permuted_scaleA = permute_scale_a(scaleA)
    args = (
        A.view(torch.int8).view(-1),
        B.view(torch.int8).view(-1),
        output.view(-1),
        permuted_scaleA.view(-1),
        scaleB.view(-1),
        sorted_ids,
        expert_ids,
        num_valid_ids,
        tokens,
        expert_ids.numel(),
        stream,
    )
    launcher = compile_moe_stage1_fp8_8wave(
        TILE_M,
        TILE_N,
        TILE_K,
        N,
        K,
        topk,
        num_experts,
        permlane_epilogue=PERMLANE_EPILOGUE,
    )
    kernel = flyc.compile[{"opt_level": 2}](launcher, *args)
    kernel(*args)
    torch.cuda.synchronize()

    diff = pyhip.calc_diff(output.float(), reference.float())
    finite = bool(torch.isfinite(output).all())
    is_correct = finite and diff < 0.01
    print(
        f"tokens={tokens} topk={topk} experts={num_experts} N={N} K={K} "
        f"finite={finite} calc_diff={diff:.6f} correct={is_correct}"
    )
    return is_correct


def compare_with_gemm(
    tokens=16384,
    intermediate_size=1792,
    hidden_size=6144,
    warmup=10,
    rounds=11,
    launches_per_round=5,
):
    from test_gemm_fp8_8w_blockscale import compile_gemm_fp8_8wave
    from test_moe_mxfp8_mxfp4_gateup_4w import make_balanced_routing

    N = 2 * intermediate_size
    K = hidden_size
    KB = K // 128
    topk = 1
    num_experts = 1
    _, _, sorted_ids, _, expert_ids, num_valid_ids = make_balanced_routing(
        tokens, topk, num_experts, K, device="cuda"
    )
    A = torch.randint(-2, 3, (tokens, K), device="cuda", dtype=torch.int8).to(
        torch.float8_e4m3fn
    )
    B = torch.randint(-2, 3, (N, K), device="cuda", dtype=torch.int8).to(
        torch.float8_e4m3fn
    )
    scaleA = torch.rand((tokens, KB), device="cuda", dtype=torch.float32)
    permuted_scaleA = permute_scale_a(scaleA)
    scaleB = torch.rand((div_up(N, 128), KB), device="cuda", dtype=torch.float32)
    gemm_output = torch.empty((tokens, N), device="cuda", dtype=torch.bfloat16)
    moe_output = torch.empty((tokens, topk, N), device="cuda", dtype=torch.bfloat16)
    stream = torch.cuda.current_stream()

    gemm_args = (
        A.view(torch.int8).view(-1),
        B.view(torch.int8).view(-1),
        gemm_output.view(-1),
        permuted_scaleA.view(-1),
        scaleB.view(-1),
        tokens,
        stream,
    )
    moe_args = (
        A.view(torch.int8).view(-1),
        B.view(torch.int8).view(-1),
        moe_output.view(-1),
        permuted_scaleA.view(-1),
        scaleB.view(-1),
        sorted_ids,
        expert_ids,
        num_valid_ids,
        tokens,
        expert_ids.numel(),
        stream,
    )
    gemm_launcher = compile_gemm_fp8_8wave(
        TILE_M,
        TILE_N,
        TILE_K,
        N,
        K,
        permlane_epilogue=PERMLANE_EPILOGUE,
        with_scale=True,
    )
    moe_launcher = compile_moe_stage1_fp8_8wave(
        TILE_M,
        TILE_N,
        TILE_K,
        N,
        K,
        topk,
        num_experts,
        permlane_epilogue=PERMLANE_EPILOGUE,
    )
    gemm_kernel = flyc.compile[{"opt_level": 2}](gemm_launcher, *gemm_args)
    moe_kernel = flyc.compile[{"opt_level": 2}](moe_launcher, *moe_args)

    for _ in range(warmup):
        gemm_kernel(*gemm_args)
        moe_kernel(*moe_args)
    torch.cuda.synchronize()

    def measure(kernel, args):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(stream)
        for _ in range(launches_per_round):
            kernel(*args)
        end.record(stream)
        end.synchronize()
        return start.elapsed_time(end) / launches_per_round

    gemm_times = []
    moe_times = []
    for round_index in range(rounds):
        if round_index % 2 == 0:
            gemm_times.append(measure(gemm_kernel, gemm_args))
            moe_times.append(measure(moe_kernel, moe_args))
        else:
            moe_times.append(measure(moe_kernel, moe_args))
            gemm_times.append(measure(gemm_kernel, gemm_args))

    gemm_times.sort()
    moe_times.sort()
    gemm_median = gemm_times[len(gemm_times) // 2]
    moe_median = moe_times[len(moe_times) // 2]
    gap = (moe_median / gemm_median - 1.0) * 100.0
    flops = 2 * tokens * N * K
    output_diff = pyhip.calc_diff(moe_output[:, 0].float(), gemm_output.float())
    print(
        f"TOPK=1 expert=1 M={tokens} N={N} K={K} "
        f"GEMM median={gemm_median * 1e3:.1f}us best={gemm_times[0] * 1e3:.1f}us "
        f"MOE median={moe_median * 1e3:.1f}us best={moe_times[0] * 1e3:.1f}us "
        f"gap={gap:+.2f}% GEMM={flops / gemm_median / 1e9:.2f}TFLOPS "
        f"MOE={flops / moe_median / 1e9:.2f}TFLOPS diff={output_diff:.6f}"
    )
    return gap, output_diff


if __name__ == "__main__":
    props = torch.cuda.get_device_properties()
    assert "950" in props.gcnArchName, "fp8 MFMA_Scale 需要 gfx950"
    torch.manual_seed(0)
    if not run_accuracy_case():
        raise AssertionError("MoE stage1 FP8 blockscale accuracy check failed")
