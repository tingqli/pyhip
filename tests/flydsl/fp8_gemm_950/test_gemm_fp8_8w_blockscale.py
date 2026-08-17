# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
#
# fp8 GEMM (C = B * A, 输出 bf16) —— 8-wave 版本，按 test_gemm_v9_fp8.py 的 tile + layout
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


def compile_gemm_fp8_8wave(
    TILE_M,
    TILE_N,
    TILE_K,
    N,
    K,
    pid_swizzle=True,
    permlane_epilogue=True,
    preshuffle_b=False,
    with_scale=False,
    useTileDMA=False,
):
    assert preshuffle_b == False, f'preshuffle B not verified on non-scale path, scale not supported'
    BLOCK_M = TILE_M // 2
    BLOCK_N = TILE_N // 2
    BLOCK_K = TILE_K
    element_type = fx.Float8E4M3FN
    elements_per_128b = 16  # 128bit / fp8(8bit)
    scaleA_stride = K // 128
    scaleB_rows = TILE_N // 128
    scaleB_elems = scaleB_rows * scaleA_stride

    def _get_pids_950(pid, M, GRID_MN, NUM_XCDS, GROUP_SIZE_M):
        num_pid_m = (M + TILE_M - 1) // TILE_M
        num_pid_n = div_up(N, TILE_N)
        if const_expr(NUM_XCDS != 1):
            pids_per_xcd = (GRID_MN + NUM_XCDS - 1) // NUM_XCDS
            tall_xcds = GRID_MN % NUM_XCDS
            tall_xcds = (tall_xcds == 0).select(NUM_XCDS, tall_xcds)
            xcd = pid % NUM_XCDS
            local_pid = pid // NUM_XCDS
            if xcd < tall_xcds:
                pid = xcd * pids_per_xcd + local_pid
            else:
                pid = tall_xcds * pids_per_xcd + (xcd - tall_xcds) * (pids_per_xcd - 1) + local_pid
        if const_expr(GROUP_SIZE_M == 1):
            pid_m = pid // num_pid_n
            pid_n = pid % num_pid_n
        else:
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            group_id = pid // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            remaining_pid_m = num_pid_m - first_pid_m
            group_size_m = (remaining_pid_m < GROUP_SIZE_M).select(remaining_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
            pid_n = (pid % num_pid_in_group) // group_size_m
        return pid_m, pid_n

    get_pids_950 = ASTRewriter.transform(_get_pids_950)

    # A/B LDS dual padding（对标 gemm_4wave_950 fp8：[[1024,16],[2048,32]]）
    A_GROUP = 8 * BLOCK_K + 16
    a_lds_elems = 2 * A_GROUP + 32  # 每 2 组再 pad 32
    a_lds_elems = (BLOCK_M // 16) * a_lds_elems  # 8 * (2*(8*128+16)+32) = 8*2112 = 16896

    @fx.struct
    class LDS:
        a_t0: fx.Array[Float8E4M3FN, 16896, 16]
        a_b0: fx.Array[Float8E4M3FN, 16896, 16]
        a_t1: fx.Array[Float8E4M3FN, 16896, 16]
        a_b1: fx.Array[Float8E4M3FN, 16896, 16]
        b_l0: fx.Array[Float8E4M3FN, 16896, 16]
        b_l1: fx.Array[Float8E4M3FN, 16896, 16]
        b_r0: fx.Array[Float8E4M3FN, 16896, 16]
        b_r1: fx.Array[Float8E4M3FN, 16896, 16]
        #scale a ping-pong LDS        
        scale_a0: fx.Array[Float32, 512, 4]
        scale_a1: fx.Array[Float32, 512, 4]
        scale_b: fx.Array[Float32, (TILE_N // 128) * (K // 128), 4]

    @flyc.kernel(known_block_size=[512, 1, 1])
    def gemm_kernel(argA: fx.Tensor, argB: fx.Tensor, argC: fx.Tensor,
                    argScaleA: fx.Tensor, argScaleB: fx.Tensor, M: int):
        tid = fx.thread_idx.x
        wave_id = tid // 64
        num_pid_n = div_up(N, TILE_N)
        if const_expr(pid_swizzle):
            bid_x, bid_y = get_pids_950(fx.block_idx.x, M, fx.grid_dim.x, 8, 4)
        else:
            bid_x = fx.block_idx.x // num_pid_n
            bid_y = fx.block_idx.x % num_pid_n

        a_iter = fx.recast_iter(element_type, fx.get_iter(argA))
        b_iter = fx.recast_iter(element_type, fx.get_iter(argB))
        A_2d = fx.Tensor(fx.make_view(a_iter, fx.make_layout((M, K), (K, 1))))
        B_2d = fx.Tensor(fx.make_view(b_iter, fx.make_layout((N, K), (K, 1))))
        C_2d = fx.Tensor(fx.make_view(fx.get_iter(argC), fx.make_layout((M, N), (N, 1))))

        A = fx.rocdl.make_buffer_tensor(A_2d, max_size=False)
        B = fx.rocdl.make_buffer_tensor(B_2d, max_size=False)
        C = fx.rocdl.make_buffer_tensor(C_2d, max_size=False)
        a_dma_rsrc = fx.buffer_ops.create_buffer_resource(
            argA, num_records_bytes=arith._to_raw(fx.Int32(M * K))
        )
        b_dma_rsrc = fx.buffer_ops.create_buffer_resource(argB, num_records_bytes=N * K)

        #subA/subB,  一个WG 被分成两个slice
        #bA flat_divide output :[BM, BK, REP_BM, REP_BK]
        #bA slice:[BM, BK, K//BK]
        #bB slice:[BN, BK, K//BK]
        bA_t = fx.flat_divide(A, (BLOCK_M, BLOCK_K))[None, None, bid_x * 2 + 0, None]
        bA_b = fx.flat_divide(A, (BLOCK_M, BLOCK_K))[None, None, bid_x * 2 + 1, None]
        bB_l = fx.flat_divide(B, (BLOCK_N, BLOCK_K))[None, None, bid_y * 2 + 0, None]
        bB_r = fx.flat_divide(B, (BLOCK_N, BLOCK_K))[None, None, bid_y * 2 + 1, None]

        #======================================== global: A, B global memroy read layout/tensor  ===============================================
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

        # Stage the complete ScaleB tile before constructing any tiled-MMA or
        # accumulator fragments. The wait/barrier closes this register lifetime,
        # allowing the loader's address VGPRs to be reused by the MFMA pipeline.
        if const_expr(with_scale):
            sB_rsrc = fx.buffer_ops.create_buffer_resource(
                argScaleB,
                num_records_bytes=arith._to_raw(
                    fx.Int32(div_up(N, 128) * scaleA_stride * 4)
                ),
            )
            scale_b_lds = fx.make_view(
                lds.scale_b.ptr, fx.make_layout(scaleB_elems, 1)
            )
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

            if const_expr(rounds_128b > 0):
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
                        sB_rsrc, scale_b_dst, fx.Int32(16), lane_byte_offset_128b,
                        fx.Int32(scale_b_global_base + round_elem_offset * 4),
                        fx.Int32(0), fx.Int32(0),
                    )

            if const_expr(remaining_elems > 0):
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
                        sB_rsrc, scale_b_dst, fx.Int32(4), lane_byte_offset_32b,
                        fx.Int32(scale_b_global_base + round_elem_offset * 4),
                        fx.Int32(0), fx.Int32(0),
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
                            sB_rsrc, scale_b_dst, fx.Int32(4), lane_byte_offset_32b,
                            fx.Int32(scale_b_global_base + tail_elem_offset * 4),
                            fx.Int32(0), fx.Int32(0),
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
        dma = fx.make_tiled_copy(buffer_copy_atom, _a_dma_tv, fx.make_tile(64, BLOCK_K)).get_slice(tid)
        # B LDS wr/rd：preshuffle 时用与 shuffle 一致的无 bank-conflict 布局（wr==rd），
        # 否则沿用与 A 相同的 dual-padding _wr/_rd。
        _wr_b = _wr
        _rd_b = _rd
        if const_expr(preshuffle_b):
            _b_lds = fx.make_layout(((16, BLOCK_N // 16), (16, BLOCK_K // 16)), ((16, 2048), (1, 256)))
            _wr_b = _b_lds
            _rd_b = _b_lds
            # B 专属 g2s DMA（512 线程，tile(64,BLOCK_K)）：对标 4-wave 的 ((16,8,2),16),((1,512,16),32)
            # tile(32)，8-wave 行数翻倍 => (16,8,4),(1,1024,16),64 tile(64)。
            _b_g2s_tv = fx.make_layout(((16, 8, 4), elements_per_128b), ((1, 1024, 16), 64))
            dma_b = fx.make_tiled_copy(buffer_copy_atom, _b_g2s_tv, fx.make_tile(64, BLOCK_K)).get_slice(tid)
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
        mma_atom = fx.make_mma_atom(fx.rocdl.cdna4.MFMA_Scale(16, 16, 128, element_type))
        mma_atom = fx.atom_set_value(mma_atom, "scale_a", fx.Int32(0))
        mma_atom = fx.atom_set_value(mma_atom, "scale_b", fx.Int32(0))
        k_perm = fx.make_layout((32, 4), (1, 32))
        tiled_mma = fx.make_tiled_mma(mma_atom, fx.make_layout((4, 2, 1), (1, 4, 0)), (None, None, k_perm))
        
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
        bC_tl = fx.flat_divide(C, (BLOCK_M, BLOCK_N))[None, None, bid_x * 2 + 0, bid_y * 2 + 0]
        bC_tl = fx.composition(bC_tl, fx.make_ordered_layout((BLOCK_N, BLOCK_M), (1, 0)))

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
                argScaleA, num_records_bytes=arith._to_raw(fx.Int32(M * scaleA_stride * 4))
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
            scale_lane_src_offset = fx.Int32(scale_row * 4)
            scale_src_tile_base = fx.Int32(bid_x * TILE_M * 4)

            def _ac_scale_a(buf, kb):
                scale_dst = _scale_dst_ptr(scale_a_lds[buf], scale_wave_dst_offset)
                rocdl.raw_ptr_buffer_load_lds(
                    sA_rsrc, scale_dst, fx.Int32(4),
                    scale_lane_src_offset,
                    fx.Int32(scale_src_tile_base + kb * M * 4), fx.Int32(0), fx.Int32(0),
                )

            def _scale_b_addr(kb):
                return fx.Int32(fx.ptrtoint(lds.scale_b.ptr)) + kb * 4

            def _rd_scale_b(addr):
                result_type = ir.Type.parse("!llvm.struct<(f32, f32)>")
                result = _llvm.inline_asm(
                    result_type,
                    [arith._to_raw(addr)],
                    "ds_read_b32 $0, $2\n"
                    f"ds_read_b32 $1, $2 offset:{scaleA_stride * 4}",
                    "=&v,=&v,v,~{memory}",
                    has_side_effects=True,
                )
                return Vec.from_elements([
                    fx.Float32(_llvm.extractvalue(T.f32, result, [0])),
                    fx.Float32(_llvm.extractvalue(T.f32, result, [1])),
                ], fx.Float32)

            def _scalarize_scale_b(scales):
                result_type = ir.Type.parse("!llvm.struct<(f32, f32)>")
                result = _llvm.inline_asm(
                    result_type,
                    [arith._to_raw(scales[0]), arith._to_raw(scales[1])],
                    "v_readfirstlane_b32 $0, $2\n"
                    "v_readfirstlane_b32 $1, $3",
                    "=&s,=&s,v,v",
                    has_side_effects=True,
                )
                return Vec.from_elements([
                    fx.Float32(_llvm.extractvalue(T.f32, result, [0])),
                    fx.Float32(_llvm.extractvalue(T.f32, result, [1])),
                ], fx.Float32)

            def _rd_scale_a(buf, bottom):
                half_offset = bottom * BLOCK_M
                wave_copy_offset = wave_m * TILE_M
                scales = []
                for m0 in range_constexpr(M_REP):
                    scale_offset = (
                        wave_copy_offset + half_offset + wave_m * 16
                        + lane_id % 16 + m0 * 32
                    )
                    scale_src = fx.make_view(
                        fx.add_offset(lds.scale_a0.ptr if buf == 0 else lds.scale_a1.ptr, scale_offset),
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
                            arith._to_raw(partial0[0]), arith._to_raw(partial0[1]),
                            arith._to_raw(partial0[2]), arith._to_raw(partial0[3]),
                            arith._to_raw(partial1[0]), arith._to_raw(partial1[1]),
                            arith._to_raw(partial1[2]), arith._to_raw(partial1[3]),
                            arith._to_raw(scale),
                            arith._to_raw(accum0[0]), arith._to_raw(accum0[1]),
                            arith._to_raw(accum0[2]), arith._to_raw(accum0[3]),
                            arith._to_raw(accum1[0]), arith._to_raw(accum1[1]),
                            arith._to_raw(accum1[2]), arith._to_raw(accum1[3]),
                            arith._to_raw(operand_a0), arith._to_raw(operand_a1),
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
                    cs0.store(Vec.from_elements([
                        fx.Float32(_llvm.extractvalue(T.f32, result, [0])),
                        fx.Float32(_llvm.extractvalue(T.f32, result, [1])),
                        fx.Float32(_llvm.extractvalue(T.f32, result, [2])),
                        fx.Float32(_llvm.extractvalue(T.f32, result, [3])),
                    ], fx.Float32))
                    cs1.store(Vec.from_elements([
                        fx.Float32(_llvm.extractvalue(T.f32, result, [4])),
                        fx.Float32(_llvm.extractvalue(T.f32, result, [5])),
                        fx.Float32(_llvm.extractvalue(T.f32, result, [6])),
                        fx.Float32(_llvm.extractvalue(T.f32, result, [7])),
                    ], fx.Float32))
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
        assert num_tiles >= 4 and num_tiles % 2 == 0
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
                _rp = _fly_dialect.extract_aligned_pointer_as_index(_pt, arith._to_raw(root_view))
                return fx.buffer_ops.get_element_ptr(_rp, byte_offset=byte_offset, elem_type=T.i8)

            # pyhip 的 g2s tv（512 线程 = 64 行 × 8 k-组，每线程 1×16 fp8）
            _g2s_tile, _g2s_tv = fx.make_layout_tv(
                fx.make_layout((8 * 8, 8), (8, 1)),
                fx.make_layout((1, elements_per_128b), (1, 1)),
            )
            _copy_g2s = fx.make_tiled_copy(buffer_copy_atom, _g2s_tv, _g2s_tile).get_slice(tid)
            _dst_stride = _copy_g2s.partition_D(sA_t_wr[0]).stride[1].to_py_value()

            # 每 wave 的 LDS 基址（dual-padding group），readfirstlane 一次
            _a_wave_off_elems = (
                wave_id % 2 * (8 * BLOCK_K + 16)
                + wave_id // 2 * (2 * (8 * BLOCK_K + 16) + 32)
            )
            _a_wave_off_bytes = rocdl.readfirstlane(
                T.i32, arith._to_raw(fx.Int32(_a_wave_off_elems * _elem_bytes))
            )
            _b_wave_off_bytes = _a_wave_off_bytes  # 非 preshuffle B 与 A 同布局

            _aT_dst = [_dma_dst_ptr(sA_t_wr[0], _a_wave_off_bytes), _dma_dst_ptr(sA_t_wr[1], _a_wave_off_bytes)]
            _aB_dst = [_dma_dst_ptr(sA_b_wr[0], _a_wave_off_bytes), _dma_dst_ptr(sA_b_wr[1], _a_wave_off_bytes)]
            _bL_dst = [_dma_dst_ptr(sB_l_wr[0], _b_wave_off_bytes), _dma_dst_ptr(sB_l_wr[1], _b_wave_off_bytes)]
            _bR_dst = [_dma_dst_ptr(sB_r_wr[0], _b_wave_off_bytes), _dma_dst_ptr(sB_r_wr[1], _b_wave_off_bytes)]

            # 每 thread 的 (row, k) 源映射（对标 pyhip a_lane_row = tid//8）
            _a_lane_row = tid // 8
            _a_lane_k = tid % 8 * elements_per_128b
            _a_local_row = _a_lane_row % 8 * (BLOCK_M // 8) + _a_lane_row // 8
            _lane_src_offset = fx.Int32((_a_local_row * K + _a_lane_k) * _elem_bytes)
            _aT_src_wave_base = fx.Int32(bid_x * TILE_M * K * _elem_bytes)
            _bL_src_wave_base = fx.Int32(bid_y * TILE_N * K * _elem_bytes)

            def _raw_g2s(rsrc, dst_base, src_wave_base, ki):
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
                        rsrc, _dp, fx.Int32(16), _lane_src_offset, _so, fx.Int32(0), fx.Int32(0)
                    )

        def _ac_At(b, ki):
            if const_expr(not useTileDMA):
                _raw_g2s(a_dma_rsrc, _aT_dst[b], _aT_src_wave_base, ki)
            else:
                fx.copy(async_copy_atom, aT_g[None, None, None, ki], aT_s[b])

        def _ac_Ab(b, ki):
            if const_expr(not useTileDMA):
                _raw_g2s(a_dma_rsrc, _aB_dst[b], _aT_src_wave_base + BLOCK_M * K * _elem_bytes, ki)
            else:
                fx.copy(async_copy_atom, aB_g[None, None, None, ki], aB_s[b])

        def _ac_Bl(b, ki):
            if const_expr(not useTileDMA):
                _raw_g2s(b_dma_rsrc, _bL_dst[b], _bL_src_wave_base, ki)
            else:
                fx.copy(async_copy_atom, bL_g[None, None, None, ki], bL_s[b])

        def _ac_Br(b, ki):
            if const_expr(not useTileDMA):
                _raw_g2s(b_dma_rsrc, _bR_dst[b], _bL_src_wave_base + BLOCK_N * K * _elem_bytes, ki)
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
        
        vmcnt = vm_load_cnt_a + vm_load_cnt_b*2 + vm_load_cnt_scale_a
        rocdl.s_waitcnt(encode_waitcnt_950(vmcnt=vmcnt))
        rocdl.s_barrier()
        rocdl.sched_barrier(0)

        if const_expr(with_scale):
            frag_P.fill(0)
            acc_init = [
                frag_C_tl.load(), frag_C_tr.load(), frag_C_bl.load(), frag_C_br.load(),
                frag_P.load(),
                Vec.from_elements([fx.Float32(0), fx.Float32(0), fx.Float32(0), fx.Float32(0)], fx.Float32),
                fx.Float32(0),
            ]
        else:
            acc_init = [
                frag_C_tl.load(), frag_C_tr.load(), frag_C_bl.load(), frag_C_br.load(),
            ]

        
        for kidx, states in range(0, num_tiles, 2, init=acc_init):
            frag_C_tl.store(states[0])
            frag_C_tr.store(states[1])
            frag_C_bl.store(states[2])
            frag_C_br.store(states[3])
            if const_expr(with_scale):
                frag_P.store(states[4])
                fifo_scale_a_0 = Vec.from_elements(
                    [fx.Float32(0), fx.Float32(0), fx.Float32(0), fx.Float32(0)], fx.Float32
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
                _ac_Ab(tock, kiter+1)
                rocdl.sched_barrier(0)
                rocdl.s_waitcnt(encode_waitcnt_950(lgkmcnt=0))
                rocdl.sched_barrier(0)
                if const_expr(with_scale):
                    mfma_scaleB = _scalarize_scale_b(mfma_scaleB)
        
                begin_compute_phase()
                if const_expr(with_scale):
                    fifo_scale_a_0, fifo_scale_b_0 = mfma_scaleA, mfma_scaleB[0]
                    do_gemm(frag_C_br, frag_B_l, frag_A_t, fifo_scale_a_1, fifo_scale_b_1)
                else:
                    do_gemm(frag_C_tl, frag_B_l, frag_A_t)
                end_compute_phase()
                
                _rd_Br(tick)
                _ac_At(tick, kiter+2)

                begin_compute_phase()
                if const_expr(with_scale):
                    fifo_scale_a_1, fifo_scale_b_1 = mfma_scaleA, mfma_scaleB[1]
                    do_gemm(frag_C_tl, frag_B_r, frag_A_t, fifo_scale_a_0, fifo_scale_b_0)
                else:
                    do_gemm(frag_C_tr, frag_B_r, frag_A_t)
                end_compute_phase()

                _rd_Ab(tick)
                if const_expr(with_scale):
                    mfma_scaleA = _rd_scale_a(tick, 1)
                _ac_Bl(tick, kiter+2)

                begin_compute_phase()
                if const_expr(with_scale):
                    fifo_scale_a_0, fifo_scale_b_0 = mfma_scaleA, mfma_scaleB[0]
                    do_gemm(frag_C_tr, frag_B_l, frag_A_t, fifo_scale_a_1, fifo_scale_b_1)
                else:
                    do_gemm(frag_C_bl, frag_B_l, frag_A_t)
                end_compute_phase()

                _ac_Br(tick, kiter+2)
                if const_expr(with_scale):
                    _ac_scale_a(tick, kiter + 2)
                rocdl.s_waitcnt(encode_waitcnt_950(
                    vmcnt=vm_load_cnt_a + vm_load_cnt_b*2 + vm_load_cnt_scale_a
                ))
                
                begin_compute_phase()
                if const_expr(with_scale):
                    fifo_scale_a_1, fifo_scale_b_1 = mfma_scaleA, mfma_scaleB[1]
                    do_gemm(frag_C_bl, frag_B_r, frag_A_t, fifo_scale_a_0, fifo_scale_b_0)
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
                _ac_Ab(tock, kiter+2)
                rocdl.sched_barrier(0)
                rocdl.s_waitcnt(encode_waitcnt_950(lgkmcnt=0))
                rocdl.sched_barrier(0)
                if const_expr(with_scale):
                    mfma_scaleB = _scalarize_scale_b(mfma_scaleB)
        
                begin_compute_phase()
                if const_expr(with_scale):
                    fifo_scale_a_0, fifo_scale_b_0 = mfma_scaleA, mfma_scaleB[0]
                    do_gemm(frag_C_br, frag_B_l, frag_A_t, fifo_scale_a_1, fifo_scale_b_1)
                else:
                    do_gemm(frag_C_tl, frag_B_l, frag_A_t)
                end_compute_phase()
                
                _rd_Br(tick)
                _ac_At(tick, kiter+3)

                begin_compute_phase()
                if const_expr(with_scale):
                    fifo_scale_a_1, fifo_scale_b_1 = mfma_scaleA, mfma_scaleB[1]
                    do_gemm(frag_C_tl, frag_B_r, frag_A_t, fifo_scale_a_0, fifo_scale_b_0)
                else:
                    do_gemm(frag_C_tr, frag_B_r, frag_A_t)
                end_compute_phase()

                _rd_Ab(tick)
                if const_expr(with_scale):
                    mfma_scaleA = _rd_scale_a(tick, 1)
                _ac_Bl(tick, kiter+3)

                begin_compute_phase()
                if const_expr(with_scale):
                    fifo_scale_a_0, fifo_scale_b_0 = mfma_scaleA, mfma_scaleB[0]
                    do_gemm(frag_C_tr, frag_B_l, frag_A_t, fifo_scale_a_1, fifo_scale_b_1)
                else:
                    do_gemm(frag_C_bl, frag_B_l, frag_A_t)
                end_compute_phase()

                _ac_Br(tick, kiter+3)
                if const_expr(with_scale):
                    _ac_scale_a(tick, kiter+3)
                rocdl.s_waitcnt(encode_waitcnt_950(
                    vmcnt=vm_load_cnt_a + vm_load_cnt_b*2 + vm_load_cnt_scale_a
                ))
                
                begin_compute_phase()
                if const_expr(with_scale):
                    fifo_scale_a_1, fifo_scale_b_1 = mfma_scaleA, mfma_scaleB[1]
                    do_gemm(frag_C_bl, frag_B_r, frag_A_t, fifo_scale_a_0, fifo_scale_b_0)
                else:
                    do_gemm(frag_C_br, frag_B_r, frag_A_t)
                end_compute_phase()
            if const_expr(with_scale):
                yield_values = [
                    frag_C_tl.load(), frag_C_tr.load(), frag_C_bl.load(), frag_C_br.load(),
                    frag_P.load(), fifo_scale_a_1, fifo_scale_b_1,
                ]
            else:
                yield_values = [
                    frag_C_tl.load(), frag_C_tr.load(), frag_C_bl.load(), frag_C_br.load(),
                ]
            results = yield yield_values

        frag_C_tl.store(results[0])
        frag_C_tr.store(results[1])
        frag_C_bl.store(results[2])
        frag_C_br.store(results[3])
        
        c_store_rsrc = fx.buffer_ops.create_buffer_resource(argC, num_records_bytes=arith._to_raw(fx.Int32(M * N * 2)))
        bC_tr = fx.flat_divide(C, (BLOCK_M, BLOCK_N))[None, None, bid_x * 2 + 0, bid_y * 2 + 1]
        bC_bl = fx.flat_divide(C, (BLOCK_M, BLOCK_N))[None, None, bid_x * 2 + 1, bid_y * 2 + 0]
        bC_br = fx.flat_divide(C, (BLOCK_M, BLOCK_N))[None, None, bid_x * 2 + 1, bid_y * 2 + 1]
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
                    cs.store(fx.fma(
                        frag_P[None, n0, m0].load(), scale_vec, cs.load()
                    ))
        if wave_id < 4:
            rocdl.s_barrier()

        # ---- epilogue store ----
        if const_expr(permlane_epilogue and TILE_N % 256 == 0):
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
                        swap0 = rocdl.permlane16_swap(pair_type, arith._to_raw(d0_a), arith._to_raw(d0_b), False, False)
                        swap1 = rocdl.permlane16_swap(pair_type, arith._to_raw(d1_a), arith._to_raw(d1_b), False, False)
                        packed = Vec.from_elements(
                            [
                                fx.Int32(_llvm.extractvalue(T.i32, swap0, [0])),
                                fx.Int32(_llvm.extractvalue(T.i32, swap1, [0])),
                                fx.Int32(_llvm.extractvalue(T.i32, swap0, [1])),
                                fx.Int32(_llvm.extractvalue(T.i32, swap1, [1])),
                            ],
                            fx.Int32,
                        )
                        row = (
                            bid_x * TILE_M
                            + quadrant_m * (TILE_M // 2)
                            + row_repeat * 32
                            + wave_m * 16
                            + lane_id % 16
                        )
                        col = (
                            bid_y * TILE_N
                            + quadrant_n * (TILE_N // 2)
                            + col_repeat * 64
                            + lane_group % 2 * 64
                            + wave_n * 16
                            + lane_group // 2 * 8
                        )
                        byte_offset = (row * N + col) * 2
                        fx.buffer_ops.buffer_store(packed, c_store_rsrc, byte_offset, offset_is_bytes=True)

            store_c_quadrant(frag_C_tl, 0, 0)
            store_c_quadrant(frag_C_tr, 0, 1)
            store_c_quadrant(frag_C_bl, 1, 0)
            store_c_quadrant(frag_C_br, 1, 1)
        else:
            c_frag_bf16 = fx.make_fragment_like(frag_C_tl, dtype=fx.BFloat16)
            store_atom = fx.make_copy_atom(fx.rocdl.BufferCopy64b(), fx.BFloat16)
            store_thr = fx.make_tiled_copy_C(store_atom, tiled_mma).get_slice(tid)

            def store_c_quadrant(c_frag, bC):
                c_frag_bf16.store(c_frag.load().to(fx.BFloat16))
                fx.copy(store_atom, store_thr.retile(c_frag_bf16), store_thr.partition_D(bC))

            store_c_quadrant(frag_C_tl, bC_tl)
            store_c_quadrant(frag_C_tr, bC_tr)
            store_c_quadrant(frag_C_bl, bC_bl)
            store_c_quadrant(frag_C_br, bC_br)

    @flyc.jit
    def launch_gemm(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor, scaleA: fx.Tensor, scaleB: fx.Tensor,
                    M: int, stream: fx.Stream = fx.Stream(None)):
        gemm_kernel(A, B, C, scaleA, scaleB, M).launch(
            grid=(div_up(M, TILE_M) * div_up(N, TILE_N), 1, 1), block=(512, 1, 1), stream=stream
        )

    return launch_gemm


# =========================== test / perf ===========================
TILE_M = 256
TILE_N = 256
TILE_K = 128
M = int(os.environ.get("GEMM_M", 8192))
N = int(os.environ.get("GEMM_N", 8192))
K = int(os.environ.get("GEMM_K", 8192))
PERMLANE_EPILOGUE = _env_flag("PERMLANE", "1")

import pyhip


def _load_shuffle_weight():
    import sys as _sys, os.path as _osp
    _root = _osp.abspath(_osp.join(_osp.dirname(__file__), "..", ".."))
    if _root not in _sys.path:
        _sys.path.insert(0, _root)
    from tests.utils import shuffle_weight
    return shuffle_weight


def run_test(M, N, K, perf=False, permlane_output=True, preshuffle_b=False, with_scale=False,
             run_count=50, data_clones=32, useTiledDMA=False):
    shuffle_weight = _load_shuffle_weight() if preshuffle_b else None

    def _shuffle_b(x):
        return shuffle_weight(x, layout=(16, 64)) if preshuffle_b else x

    KB = K // 128
    empty = torch.empty(0, device="cuda", dtype=torch.float32)

    def _gen_scales():
        if not with_scale:
            return empty, empty
        sA = torch.rand((M, KB), device="cuda", dtype=torch.float32)
        sB = torch.rand((N // 128, KB), device="cuda", dtype=torch.float32)
        return sA, sB

    def _ref(a, b, sA, sB):
        if not with_scale:
            return a.float() @ b.float().t()
        a_deq = (a.float().view(M, KB, 128) * sA.view(M, KB, 1)).view(M, K)
        b_deq = (
            b.float().view(N // 128, 128, KB, 128)
            * sB.view(N // 128, 1, KB, 1)
        ).view(N, K)
        return a_deq @ b_deq.t()

    a = (torch.rand(M, K, device="cuda") / 10.0).to(torch.float8_e4m3fn)
    b = (torch.rand(N, K, device="cuda") / 10.0).to(torch.float8_e4m3fn)
    sA, sB = _gen_scales()
    ref = _ref(a, b, sA, sB)
    sA_kernel = sA.transpose(0, 1).contiguous() if with_scale else sA
    out = torch.zeros((M, N), device="cuda", dtype=torch.bfloat16)
    weight = _shuffle_b(b)
    stream = torch.cuda.current_stream()
    args = (a.view(torch.int8).view(-1), weight.view(torch.int8).view(-1), out.view(-1),
            sA_kernel.view(-1), sB.view(-1), M, stream)

    launcher = compile_gemm_fp8_8wave(TILE_M, TILE_N, TILE_K, N, K, permlane_epilogue=permlane_output,
                                      preshuffle_b=preshuffle_b, with_scale=with_scale, useTileDMA=useTiledDMA)
    kernel = flyc.compile[{"opt_level": 2}](launcher, *args)
    kernel(*args)
    torch.cuda.synchronize()

    out_f32 = out.float()
    abs_err = (out_f32 - ref).abs()
    check_rtol = 1.6e-2
    check_atol = 1e-5
    close_mask = torch.isclose(out_f32, ref, rtol=check_rtol, atol=check_atol)
    close_count = close_mask.count_nonzero().item()
    total_count = close_mask.numel()
    bf16_ref = ref.to(torch.bfloat16)
    bf16_exact_count = (out == bf16_ref).count_nonzero().item()
    top_count = min(100, total_count)
    top_errors, top_indices = torch.topk(abs_err.reshape(-1), top_count)
    top_refs = ref.reshape(-1)[top_indices]
    top_outputs = out_f32.reshape(-1)[top_indices]
    top_rel_errors = top_errors / top_refs.abs().clamp_min(torch.finfo(torch.float32).tiny)
    top_close = close_mask.reshape(-1)[top_indices]

    print(
        f"torch.isclose(rtol={check_rtol}, atol={check_atol}): "
        f"{close_count}/{total_count} ({close_count / total_count:.6%}), "
        f"not_close={total_count - close_count}"
    )
    print(
        f"exact vs bf16-rounded ref: {bf16_exact_count}/{total_count} "
        f"({bf16_exact_count / total_count:.6%}), mismatched={total_count - bf16_exact_count}"
    )
    print("top100: rank (row,col) ref output abs_error rel_error isclose")
    for rank in range(top_count):
        flat_index = top_indices[rank].item()
        row, col = divmod(flat_index, N)
        print(
            f"{rank + 1:3d} ({row:5d},{col:5d}) "
            f"ref={top_refs[rank].item(): .9e} "
            f"output={top_outputs[rank].item(): .9e} "
            f"abs_error={top_errors[rank].item(): .9e} "
            f"rel_error={top_rel_errors[rank].item(): .9e} "
            f"isclose={bool(top_close[rank].item())}"
        )
    # fp8×fp8→f32 累加对整数输入是精确的：与 f32 ref 的 diff 只来自输出转 bf16 的舍入。
    # 与「bf16 舍入后的 ref」比较应 ≈0（非 scale 时用来验证计算零误差）。
    diff = pyhip.calc_diff(out_f32, ref)
    diff_bf16ref = pyhip.calc_diff(out_f32, bf16_ref.float())
    is_correct = diff < 0.01
    print(f"####M={M} N={N} K={K} 8wave preshuffle_b={preshuffle_b} with_scale={with_scale}, useTiledDMA={useTiledDMA} "
          f"is_correct={is_correct} calc_diff(vs f32 ref)={diff:.6f} "
          f"calc_diff(vs bf16 ref)={diff_bf16ref:.6f} max_abs={abs_err.max().item():.3f}")

    if not perf:
        return is_correct

    As = [torch.randint(-2, 3, (M, K), device="cuda", dtype=torch.int8).to(torch.float8_e4m3fn) for _ in range(data_clones)]
    Bs = [_shuffle_b(torch.randint(-2, 3, (N, K), device="cuda", dtype=torch.int8).to(torch.float8_e4m3fn)) for _ in range(data_clones)]
    SAs = [(_gen_scales()[0] if with_scale else empty) for _ in range(data_clones)]
    SAs_kernel = [sa.transpose(0, 1).contiguous() if with_scale else sa for sa in SAs]
    SBs = [(_gen_scales()[1] if with_scale else empty) for _ in range(data_clones)]
    Cs = [torch.zeros((M, N), device="cuda", dtype=torch.bfloat16) for _ in range(data_clones)]
    arg_sets = [
        (As[i].view(torch.int8).view(-1), Bs[i].view(torch.int8).view(-1), Cs[i].view(-1),
         SAs_kernel[i].view(-1), SBs[i].view(-1), M, stream)
        for i in range(data_clones)
    ]
    flops = 2 * M * N * K
    mem_bytes = (M * K + N * K) * 1 + M * N * 2
    for i in range(data_clones):
        kernel(*arg_sets[i])
    torch.cuda.synchronize()
    di = 0
    latencies = []
    for _ in range(run_count):
        di = (di + 1) % data_clones
        with pyhip.cudaPerf(flops, mem_bytes, name=f"gemm_{di}") as p:
            kernel(*arg_sets[di])
        latencies.append(p.dt_ms)
    latencies.sort()
    best_ms = latencies[0]
    print(f"\n=== perf 8wave M={M} N={N} K={K} with_scale={with_scale} ===")
    print(f"gemm:  {best_ms*1e3:.1f} us  {flops/(best_ms*1e-3)/1e12:.2f} TFLOPS  {mem_bytes/(best_ms*1e-3)/1e9:.1f} GB/s")
    return is_correct


if __name__ == "__main__":
    props = torch.cuda.get_device_properties()
    assert "950" in props.gcnArchName, "fp8 MFMA_Scale 需要 gfx950"
    torch.manual_seed(0)
    
    K = 6144
    run_test(M=8192, N=8192, K=K, perf=True, permlane_output=PERMLANE_EPILOGUE, with_scale=True)
    run_test(M=16384, N=3584, K=K, perf=True, permlane_output=PERMLANE_EPILOGUE, with_scale=True)
