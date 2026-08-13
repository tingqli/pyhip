# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
#
# fp8 GEMM (C = B * A, 输出 bf16)，按 test_gemm_v9.py 的方式用 tile + layout 抽象编写
# （flat_divide / make_tiled_copy / make_tiled_mma / make_fragment / fx.copy / fx.gemm），
# 不做手动 byte-offset 计算。
#   - BLOCK_M=BLOCK_N=BLOCK_K=128, TILE_M=TILE_N=256, 4-wave, 2x2 quadrant
#   - MFMA 指令 V_MFMA_SCALE_F32_16X16X128_F8F6F4（scale=0 => 不含 scale）
#   - A/B 均普通输入 + LDS bank-conflict 消解：padding（默认，[[1024,32]] 单 padding，对标 bf16 v9）
#     或 swizzle（lds_swizzle=True, MBase=4）。LDS ping-pong 双缓冲 + 寄存器软件流水。
#   - 约定：A 走 make_fragment_B，B 走 make_fragment_A；fx.gemm(mma, C, frag_B, frag_A)。
#   - 用 SWIZZLE=1 环境变量切到 swizzle 版本。
#
# 运行：cd /mywork/FlyDSL/tests/kernels && HIP_VISIBLE_DEVICES=4 python ./test_gemm_v9_fp8.py

import os
import math

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr.typing import BFloat16, Float8E4M3FN, Float32, Int8, Int32, T, Vector
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl, vector, arith
from flydsl.expr.typing import Vector as Vec
from flydsl._mlir import ir
from flydsl._mlir.dialects import fly as _fly
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.compiler.ast_rewriter import ASTRewriter

__all__ = ["compile_gemm_fp8"]


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


def div_up(x, y):
    return (x + y - 1) // y


def encode_waitcnt_950(vmcnt=63, expcnt=7, lgkmcnt=63):
    vm_lo = vmcnt & 0xF
    vm_hi = (vmcnt >> 4) & 0x3
    return vm_lo | (expcnt << 4) | (lgkmcnt << 8) | (vm_hi << 14)


def waitvmcnt_barrier(vmcnt):
    # 对标 test_gemm_v9.py：s_waitcnt vmcnt(n) + s_waitcnt lgkmcnt(0) + s_barrier，
    # 一次完成 vmem/lds 等待与全 block 同步（内含 s_barrier，无需再单独 gpu.barrier）。
    rocdl.s_waitcnt(encode_waitcnt_950(vmcnt=vmcnt))
    rocdl.s_waitcnt(encode_waitcnt_950(lgkmcnt=0))
    rocdl.s_barrier()


def hot_loop_scheduler_mainloop(group_id, vmem_ops, dsrd_ops):
    total_mfmas = 16
    prev_dsrd = 0
    prev_vmem = 0
    for i in range_constexpr(total_mfmas):
        cur_dsrd = ((i + 3) * dsrd_ops + total_mfmas - 1) // total_mfmas
        cur_dsrd = min(cur_dsrd, dsrd_ops)
        if const_expr(cur_dsrd > prev_dsrd):
            rocdl.sched_group_barrier(rocdl.mask_dsrd, cur_dsrd - prev_dsrd, group_id)
        rocdl.sched_group_barrier(rocdl.mask_mfma, 1, group_id)
        cur_vmem = ((i + 1) * vmem_ops + total_mfmas - 1) // total_mfmas
        if const_expr(cur_vmem > prev_vmem):
            rocdl.sched_group_barrier(rocdl.mask_vmem_rd, cur_vmem - prev_vmem, group_id)
        prev_dsrd = cur_dsrd
        prev_vmem = cur_vmem


# VMEM_WRITE / VALU 掩码（flydsl 未导出 vmem_wr 常量，直接用 bit 值）。
_MASK_VALU = 0x002
_MASK_VMEM_WR = 0x040


def scheduler_store_overlap(group_id):
    # MFMA 领先：每 2 条 MFMA 穿插 store 的 VALU(cvt/permlane) 与 buffer_store(vmem_wr)，
    # 用 MFMA 计算掩盖 store 的写延迟（fp8 每象限 16 条 MFMA、8 次 buffer_store）。
    # MFMA 必须领先，否则 store 会挡住计算流水。
    for _ in range_constexpr(8):
        rocdl.sched_group_barrier(rocdl.mask_mfma, 2, group_id)
        rocdl.sched_group_barrier(_MASK_VALU, 6, group_id)
        rocdl.sched_group_barrier(_MASK_VMEM_WR, 1, group_id)


def compile_gemm_fp8(
    TILE_M,
    TILE_N,
    TILE_K,
    N,
    K,
    pid_swizzle=True,
    lds_swizzle=False,
    preshuffle_b=False,
    permlane_epilogue=True,
    store_overlap=False,
    with_scale=False,
):
    BLOCK_M = TILE_M // 2
    BLOCK_N = TILE_N // 2
    BLOCK_K = TILE_K
    assert BLOCK_K == 128
    assert K % 256 == 0
    element_type = fx.Float8E4M3FN
    elements_per_128b = 16  # 128bit / fp8(8bit)
    # sched_group_barrier 精确调度（对标 bf16 hot_loop_scheduler_mainloop）：
    # fp8 每象限一条 MFMA(16x16x128) 抵 bf16 两条(16x16x32)，故 MFMA 计数减半（32->16）；
    # ds_read/vmem 计数不变（数据量一致）。仅在完整流水迭代（同时有 s2r+g2s）时应用。
    _USE_SCHED = _env_flag("SCHED", "1")

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

    # A padding（对标 bf16 v9 / gluon a8w8 kWidth=16 单 padding [[1024,32]]：每 8 行 pad 32 fp8=32B）
    A_PAD = 32
    A_GROUP = 8 * BLOCK_K + A_PAD  # 1056
    a_lds_elems = (BLOCK_M // 8) * A_GROUP  # 16*1056 = 16896
    b_lds_elems = (BLOCK_N // 8) * A_GROUP  # 16896
    scale_lds_bytes = 128 * 8

    if with_scale:
        @fx.struct
        class LDS:
            a_t0: fx.Array[Float8E4M3FN, a_lds_elems, 16]
            a_t1: fx.Array[Float8E4M3FN, a_lds_elems, 16]
            a_b0: fx.Array[Float8E4M3FN, a_lds_elems, 16]
            a_b1: fx.Array[Float8E4M3FN, a_lds_elems, 16]
            b_l0: fx.Array[Float8E4M3FN, b_lds_elems, 16]
            b_l1: fx.Array[Float8E4M3FN, b_lds_elems, 16]
            b_r0: fx.Array[Float8E4M3FN, b_lds_elems, 16]
            b_r1: fx.Array[Float8E4M3FN, b_lds_elems, 16]
            scale_a_t0: fx.Array[Int8, scale_lds_bytes, 16]
            scale_a_t1: fx.Array[Int8, scale_lds_bytes, 16]
            scale_a_b0: fx.Array[Int8, scale_lds_bytes, 16]
            scale_a_b1: fx.Array[Int8, scale_lds_bytes, 16]
            scale_b_l0: fx.Array[Int8, scale_lds_bytes, 16]
            scale_b_l1: fx.Array[Int8, scale_lds_bytes, 16]
            scale_b_r0: fx.Array[Int8, scale_lds_bytes, 16]
            scale_b_r1: fx.Array[Int8, scale_lds_bytes, 16]
    else:
        @fx.struct
        class LDS:
            a_t0: fx.Array[Float8E4M3FN, a_lds_elems, 16]
            a_t1: fx.Array[Float8E4M3FN, a_lds_elems, 16]
            a_b0: fx.Array[Float8E4M3FN, a_lds_elems, 16]
            a_b1: fx.Array[Float8E4M3FN, a_lds_elems, 16]
            b_l0: fx.Array[Float8E4M3FN, b_lds_elems, 16]
            b_l1: fx.Array[Float8E4M3FN, b_lds_elems, 16]
            b_r0: fx.Array[Float8E4M3FN, b_lds_elems, 16]
            b_r1: fx.Array[Float8E4M3FN, b_lds_elems, 16]

    @flyc.kernel(known_block_size=[256, 1, 1])
    def gemm_kernel(
        argA: fx.Tensor,
        argB: fx.Tensor,
        argScaleA: fx.Tensor,
        argScaleB: fx.Tensor,
        argC: fx.Tensor,
        M: int,
    ):
        tid = fx.thread_idx.x
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

        if const_expr(with_scale):
            # Host: [rows, G] -> [rows/128, 4, 32, G] -> permute(3, 0, 2, 1).
            # View as i32 words (4 packed E8M0 = one uint32) so the LDS DMA copies one
            # dword/lane (value=1); an i8 value=4 copy misplaces the source by +3 bytes.
            scale_a_iter = fx.recast_iter(Int32, fx.get_iter(argScaleA))
            scale_b_iter = fx.recast_iter(Int32, fx.get_iter(argScaleB))
            scale_a_layout_int32 = fx.make_layout(
                ((32, 8), (M // 128, K // 128)),
                ((1, M // 4), (32, M)),
            )
            scale_b_layout_int32 = fx.make_layout(
                ((32, 8), (N // 128, K // 128)),
                ((1, N // 4), (32, N)),
            )
            ScaleA = fx.rocdl.make_buffer_tensor(
                fx.Tensor(fx.make_view(scale_a_iter, scale_a_layout_int32)), max_size=False
            )
            ScaleB = fx.rocdl.make_buffer_tensor(
                fx.Tensor(fx.make_view(scale_b_iter, scale_b_layout_int32)), max_size=False
            )
        c_store_rsrc = fx.buffer_ops.create_buffer_resource(argC, max_size=False)

        bA_t = fx.flat_divide(A, (BLOCK_M, BLOCK_K))[None, None, bid_x * 2 + 0, None]
        bA_b = fx.flat_divide(A, (BLOCK_M, BLOCK_K))[None, None, bid_x * 2 + 1, None]
        bB_l = fx.flat_divide(B, (BLOCK_N, BLOCK_K))[None, None, bid_y * 2 + 0, None]
        bB_r = fx.flat_divide(B, (BLOCK_N, BLOCK_K))[None, None, bid_y * 2 + 1, None]
        # A/B 全局 tile 视图：swizzle 版（全局 swizzle）或 padding 版（分组）。
        if const_expr(lds_swizzle):
            _nb = 4  # fp8 128-bit = 16 elem = 2^4
            _swg = fx.static(fx.SwizzleType.get(3, _nb, K.bit_length() - 1 - _nb))
            bA_t = fx.Tensor(fx.make_view(fx.get_iter(bA_t), fx.make_composed_layout(_swg, fx.get_layout(bA_t))))
            bA_b = fx.Tensor(fx.make_view(fx.get_iter(bA_b), fx.make_composed_layout(_swg, fx.get_layout(bA_b))))
            bB_l = fx.Tensor(fx.make_view(fx.get_iter(bB_l), fx.make_composed_layout(_swg, fx.get_layout(bB_l))))
            bB_r = fx.Tensor(fx.make_view(fx.get_iter(bB_r), fx.make_composed_layout(_swg, fx.get_layout(bB_r))))
        else:
            # A: 分组全局视图（每 8 行为一组，映射到 padding LDS 的行组），对标 bf16 v9 bA_layout。
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

        # preshuffle B：host 端已 shuffle_weight(B, layout=(16,64))，kernel 用 subB 再视图
        # 从 flat_divide 的 tile base 恢复 logical (n,k) -> shuffled 物理偏移（与 swizzle/padding 无关）。
        # subB 形状 ((ni16, nb=BLOCK_N//16), (k0=16, k1=BLOCK_K//16), k_tile)，
        # 步长 ((16, 16*K), (1, 256), 2048) 对应 shuffle 存储顺序 (nb,kb,k1,ni,k0)。
        if const_expr(preshuffle_b):
            _subB = fx.make_layout(
                ((16, BLOCK_N // 16), (16, BLOCK_K // 16), K // BLOCK_K),
                ((16, 16 * K), (1, 256), 2048),
            )
            bB_l = fx.Tensor(fx.make_view(fx.get_iter(bB_l), _subB))
            bB_r = fx.Tensor(fx.make_view(fx.get_iter(bB_r), _subB))

        bC_tl = fx.flat_divide(C, (BLOCK_M, BLOCK_N))[None, None, bid_x * 2 + 0, bid_y * 2 + 0]
        bC_tr = fx.flat_divide(C, (BLOCK_M, BLOCK_N))[None, None, bid_x * 2 + 0, bid_y * 2 + 1]
        bC_bl = fx.flat_divide(C, (BLOCK_M, BLOCK_N))[None, None, bid_x * 2 + 1, bid_y * 2 + 0]
        bC_br = fx.flat_divide(C, (BLOCK_M, BLOCK_N))[None, None, bid_x * 2 + 1, bid_y * 2 + 1]

        # ---- tiled MMA: MFMA_Scale 16x16x128 f8f6f4 ----
        mma_atom = fx.make_mma_atom(fx.rocdl.cdna4.MFMA_Scale(16, 16, 128, element_type))
        mma_atom = fx.atom_set_value(mma_atom, "scale_a", fx.Int32(0))
        mma_atom = fx.atom_set_value(mma_atom, "scale_b", fx.Int32(0))
        if const_expr(with_scale):
            # Logical B occupies MFMA operand A and logical A occupies operand B.
            scale_atoms = {
                (n0, m0): fx.make_mma_atom(
                    fx.rocdl.cdna4.MFMA_Scale(
                        16,
                        16,
                        128,
                        element_type,
                        opsel_a=n0,
                        opsel_b=m0,
                    )
                )
                for n0 in range_constexpr(4)
                for m0 in range_constexpr(4)
            }
            k_perm = fx.make_layout(((16, 2), 4), ((1, 64), 16))
        else:
            k_perm = fx.make_layout((32, 4), (1, 32))
        tiled_mma = fx.make_tiled_mma(mma_atom, fx.make_layout((2, 2, 1), (1, 2, 0)), (None, None, k_perm))
        thr_mma = tiled_mma.thr_slice(tid)

        # ---- copy atoms ----
        async_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopyLDS128b(), 128)
        buffer_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), element_type)
        lds_copy_atom = fx.make_copy_atom(fx.UniversalCopy128b(), element_type)

        # ---- LDS 分配 ----
        lds = fx.SharedAllocator().allocate(LDS).peek()

        scale_a_t_frag = None
        scale_a_b_frag = None
        scale_b_l_frag = None
        scale_b_r_frag = None
        if const_expr(with_scale):
            # Only the first 512 B of each 1 KB entry are useful. Recast those bytes
            # as 128 packed E8M0 dwords arranged as (32 rows, 4 row-repetitions).
            scale_lds_copy_atom = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Int32)
            scale_rd_layout = fx.make_layout((32, 4), (1, 32))
            scale_a_tv = fx.make_layout(((16, 4, 2, 2), 1), ((1, 32, 0, 16), 1))
            scale_b_tv = fx.make_layout(((16, 4, 2, 2), 1), ((1, 32, 16, 0), 1))
            scale_a_copy = fx.make_tiled_copy(
                scale_lds_copy_atom, scale_a_tv, fx.make_tile(32, 4)
            ).get_slice(tid)
            scale_b_copy = fx.make_tiled_copy(
                scale_lds_copy_atom, scale_b_tv, fx.make_tile(32, 4)
            ).get_slice(tid)

            def scale_rd_view(ptr):
                return fx.make_view(fx.recast_iter(Int32, ptr), scale_rd_layout)

            scale_a_t_rd = [
                scale_rd_view(lds.scale_a_t0.ptr),
                scale_rd_view(lds.scale_a_t1.ptr),
            ]
            scale_a_b_rd = [
                scale_rd_view(lds.scale_a_b0.ptr),
                scale_rd_view(lds.scale_a_b1.ptr),
            ]
            scale_b_l_rd = [
                scale_rd_view(lds.scale_b_l0.ptr),
                scale_rd_view(lds.scale_b_l1.ptr),
            ]
            scale_b_r_rd = [
                scale_rd_view(lds.scale_b_r0.ptr),
                scale_rd_view(lds.scale_b_r1.ptr),
            ]
            scale_a_t_src = [scale_a_copy.partition_S(view) for view in scale_a_t_rd]
            scale_a_b_src = [scale_a_copy.partition_S(view) for view in scale_a_b_rd]
            scale_b_l_src = [scale_b_copy.partition_S(view) for view in scale_b_l_rd]
            scale_b_r_src = [scale_b_copy.partition_S(view) for view in scale_b_r_rd]
            scale_a_t_frag = fx.make_fragment_like(scale_a_t_src[0])
            scale_a_b_frag = fx.make_fragment_like(scale_a_b_src[0])
            scale_b_l_frag = fx.make_fragment_like(scale_b_l_src[0])
            scale_b_r_frag = fx.make_fragment_like(scale_b_r_src[0])

            scale_a_dma_rsrc = fx.buffer_ops.create_buffer_resource(
                argScaleA, num_records_bytes=arith._to_raw(fx.Int32(M * K // 32))
            )
            scale_b_dma_rsrc = fx.buffer_ops.create_buffer_resource(
                argScaleB, num_records_bytes=arith._to_raw(fx.Int32(N * K // 32))
            )
            scale_lane_id = tid % 64
            scale_wave_id = tid // 64
            scale_wave_id_uni = fx.Int32(rocdl.readfirstlane(T.i32, arith._to_raw(scale_wave_id)))

            def make_scale_dma_ptr(ptr):
                view = fx.make_view(ptr, fx.make_layout(1, 1))
                root = _fly.extract_aligned_pointer_as_index(
                    ir.Type.parse("!llvm.ptr<3>"), arith._to_raw(view)
                )
                return fx.buffer_ops.get_element_ptr(
                    root, byte_offset=scale_wave_id_uni * 64 * 4, elem_type=T.i8
                )

            scale_a_t_dma_ptrs = [make_scale_dma_ptr(ptr) for ptr in (lds.scale_a_t0.ptr, lds.scale_a_t1.ptr)]
            scale_a_b_dma_ptrs = [make_scale_dma_ptr(ptr) for ptr in (lds.scale_a_b0.ptr, lds.scale_a_b1.ptr)]
            scale_b_l_dma_ptrs = [make_scale_dma_ptr(ptr) for ptr in (lds.scale_b_l0.ptr, lds.scale_b_l1.ptr)]
            scale_b_r_dma_ptrs = [make_scale_dma_ptr(ptr) for ptr in (lds.scale_b_r0.ptr, lds.scale_b_r1.ptr)]

            def raw_scale_g2s(rsrc, kk, ptr, row_tile, rows):
                packed_row = scale_wave_id_uni * 2 + scale_lane_id // 32
                tile_voffset = (scale_lane_id % 32) * 4 + packed_row * rows + row_tile * 32 * 4
                rocdl.raw_ptr_buffer_load_lds(
                    rsrc, ptr, fx.Int32(4), fx.Int32(tile_voffset), fx.Int32(kk * rows * 4),
                    fx.Int32(0), fx.Int32(0),
                )

        # wr/rd LDS 布局 + DMA tiled copy：swizzle 版（ordered wr + swizzle rd + make_layout_tv DMA）
        # 或 padding 版（分组 wr/rd + 专用 a_dma）。A/B 共用同一 dma。
        if const_expr(lds_swizzle):
            _swl = fx.static(fx.SwizzleType.get(3, 4, BLOCK_K.bit_length() - 1 - 4))
            _wr = fx.make_ordered_layout((BLOCK_M, BLOCK_K), (1, 0))
            _rd = fx.make_composed_layout(_swl, _wr)
            _g2s_tile, _g2s_tv = fx.make_layout_tv(
                fx.make_layout((8 * 4, 8), (8, 1)),
                fx.make_layout((1, elements_per_128b), (1, 1)),
            )
            dma = fx.make_tiled_copy(buffer_copy_atom, _g2s_tv, _g2s_tile).get_slice(tid)
        else:
            # A: fp8 kWidth=32 双层 padding（对标 gemm_4wave_950：[[1024,16],[2048,32]]），
            # 消除 MFMA_Scale A operand 读的 LDS bank conflict（单层 [[1024,32]] 无法清零）。
            _wr = fx.make_layout(
                ((8, 2, BLOCK_M // 16), BLOCK_K),
                ((BLOCK_K, 8 * BLOCK_K + 16, 2 * (8 * BLOCK_K + 16) + 32), 1),
            )
            _rd = fx.make_layout(
                ((2, BLOCK_M // 16, 8), (32, BLOCK_K // 32)),
                ((8 * BLOCK_K + 16, 2 * (8 * BLOCK_K + 16) + 32, BLOCK_K), (1, 32)),
            )
            _a_dma_tv = fx.make_layout(
                ((8, 8, 4), elements_per_128b),
                ((elements_per_128b * 32, 1, 8), 32),
            )
            dma = fx.make_tiled_copy(buffer_copy_atom, _a_dma_tv, fx.make_tile(32, BLOCK_K)).get_slice(tid)

        # B 的 LDS wr/rd：preshuffle 时用与 shuffle 一致的无 bank-conflict 布局（wr==rd），
        # 否则沿用与 A 相同的 _wr/_rd。仅 CORRECTNESS 需 subB 正确 + LDS 为任意双射；DMA tv 只影响性能。
        _wr_b = _wr
        _rd_b = _rd
        if const_expr(preshuffle_b):
            _b_lds = fx.make_layout(((16, BLOCK_N // 16), (16, BLOCK_K // 16)), ((16, 2048), (1, 256)))
            _wr_b = _b_lds
            _rd_b = _b_lds

        # B 专属 DMA：preshuffle 时 subB 的嵌套形状与 padding dma 不匹配，需独立 tiled_copy。
        # 手工 tv：每线程沿 k0(=16 连续 fp8=128b) 合并 load，n=ni(16)+16*half(2)，k=16*kblk(8)+kv(16)。
        # 对标 bf16 的 ((16,8,2),8),((1,256,16),32) tile(32,64)；fp8 value 16、kblk 步 512、tile(32,128)。
        if const_expr(preshuffle_b):
            _b_g2s_tv = fx.make_layout(((16, 8, 2), elements_per_128b), ((1, 512, 16), 32))
            dma_b = fx.make_tiled_copy(buffer_copy_atom, _b_g2s_tv, fx.make_tile(32, BLOCK_K)).get_slice(tid)
        else:
            dma_b = dma

        sA_t_wr = [fx.make_view(lds.a_t0.ptr, _wr), fx.make_view(lds.a_t1.ptr, _wr)]
        sA_b_wr = [fx.make_view(lds.a_b0.ptr, _wr), fx.make_view(lds.a_b1.ptr, _wr)]
        sA_t_rd = [fx.make_view(lds.a_t0.ptr, _rd), fx.make_view(lds.a_t1.ptr, _rd)]
        sA_b_rd = [fx.make_view(lds.a_b0.ptr, _rd), fx.make_view(lds.a_b1.ptr, _rd)]
        sB_l_wr = [fx.make_view(lds.b_l0.ptr, _wr_b), fx.make_view(lds.b_l1.ptr, _wr_b)]
        sB_r_wr = [fx.make_view(lds.b_r0.ptr, _wr_b), fx.make_view(lds.b_r1.ptr, _wr_b)]
        sB_l_rd = [fx.make_view(lds.b_l0.ptr, _rd_b), fx.make_view(lds.b_l1.ptr, _rd_b)]
        sB_r_rd = [fx.make_view(lds.b_r0.ptr, _rd_b), fx.make_view(lds.b_r1.ptr, _rd_b)]

        aT_g = dma.partition_S(bA_t)
        aB_g = dma.partition_S(bA_b)
        bL_g = dma_b.partition_S(bB_l)
        bR_g = dma_b.partition_S(bB_r)
        aT_s = [dma.partition_D(sA_t_wr[0]), dma.partition_D(sA_t_wr[1])]
        aB_s = [dma.partition_D(sA_b_wr[0]), dma.partition_D(sA_b_wr[1])]
        bL_s = [dma_b.partition_D(sB_l_wr[0]), dma_b.partition_D(sB_l_wr[1])]
        bR_s = [dma_b.partition_D(sB_r_wr[0]), dma_b.partition_D(sB_r_wr[1])]

        if const_expr(not lds_swizzle and not preshuffle_b):
            a_dma_rsrc = fx.buffer_ops.create_buffer_resource(
                argA, num_records_bytes=arith._to_raw(fx.Int32(M * K))
            )
            b_dma_rsrc = fx.buffer_ops.create_buffer_resource(
                argB, num_records_bytes=arith._to_raw(fx.Int32(N * K))
            )
            lane_id = tid % 64
            wave_id = tid // 64
            wave_id_uni = fx.Int32(rocdl.readfirstlane(T.i32, arith._to_raw(wave_id)))
            lane_voffset = fx.Int32((lane_id // 8) * 16 * K + (lane_id % 8) * 16)
            wave_lds_base = (wave_id_uni % 2) * (8 * BLOCK_K + 16) + (wave_id_uni // 2) * A_GROUP * 2

            def make_dma_ptr(ptr, copy_round):
                view = fx.make_view(ptr, fx.make_layout(1, 1))
                root = _fly.extract_aligned_pointer_as_index(
                    ir.Type.parse("!llvm.ptr<3>"), arith._to_raw(view)
                )
                return fx.buffer_ops.get_element_ptr(
                    root,
                    byte_offset=wave_lds_base + copy_round * 4 * A_GROUP,
                    elem_type=T.i8,
                )

            a_t_dma_ptrs = [[make_dma_ptr(ptr, r) for r in range_constexpr(4)] for ptr in (lds.a_t0.ptr, lds.a_t1.ptr)]
            a_b_dma_ptrs = [[make_dma_ptr(ptr, r) for r in range_constexpr(4)] for ptr in (lds.a_b0.ptr, lds.a_b1.ptr)]
            b_l_dma_ptrs = [[make_dma_ptr(ptr, r) for r in range_constexpr(4)] for ptr in (lds.b_l0.ptr, lds.b_l1.ptr)]
            b_r_dma_ptrs = [[make_dma_ptr(ptr, r) for r in range_constexpr(4)] for ptr in (lds.b_r0.ptr, lds.b_r1.ptr)]

            def raw_g2s(rsrc, kk, ptrs, row_tile):
                tile_soffset = fx.Int32(kk * BLOCK_K)
                tile_voffset = lane_voffset + fx.Int32((row_tile * BLOCK_M + wave_id_uni) * K)
                for copy_round in range_constexpr(4):
                    rocdl.raw_ptr_buffer_load_lds(
                        rsrc, ptrs[copy_round], fx.Int32(16), tile_voffset + copy_round * 4 * K,
                        tile_soffset,
                        fx.Int32(0), fx.Int32(0),
                    )

        # ---- LDS -> reg（对标 gemm_v9：A 走 B-operand，B 走 A-operand；均 padding rd）----
        # 每个 slice 只有一份寄存器 fragment（无寄存器双缓冲），双缓冲仅在 LDS 层（buf0/buf1）。
        copy_a = fx.make_tiled_copy_B(lds_copy_atom, tiled_mma).get_slice(tid)
        copy_b = fx.make_tiled_copy_A(lds_copy_atom, tiled_mma).get_slice(tid)
        # s2r 源：LDS buf0 / buf1（对标 gemm_v9 的 s2r_src0_* / s2r_src1_*）
        s2r_src0_A_t = copy_a.partition_S(sA_t_rd[0])
        s2r_src0_A_b = copy_a.partition_S(sA_b_rd[0])
        s2r_src0_B_l = copy_b.partition_S(sB_l_rd[0])
        s2r_src0_B_r = copy_b.partition_S(sB_r_rd[0])
        s2r_src1_A_t = copy_a.partition_S(sA_t_rd[1])
        s2r_src1_A_b = copy_a.partition_S(sA_b_rd[1])
        s2r_src1_B_l = copy_b.partition_S(sB_l_rd[1])
        s2r_src1_B_r = copy_b.partition_S(sB_r_rd[1])

        # 单份寄存器 fragment（A -> make_fragment_B, B -> make_fragment_A）
        frag_A_t = thr_mma.make_fragment_B(sA_t_rd[0])
        frag_A_b = thr_mma.make_fragment_B(sA_b_rd[0])
        frag_B_l = thr_mma.make_fragment_A(sB_l_rd[0])
        frag_B_r = thr_mma.make_fragment_A(sB_r_rd[0])
        dest_frag_A_t = copy_a.retile(frag_A_t)
        dest_frag_A_b = copy_a.retile(frag_A_b)
        dest_frag_B_l = copy_b.retile(frag_B_l)
        dest_frag_B_r = copy_b.retile(frag_B_r)

        # ---- C fragments（对标 test_gemm fp8: make_fragment_C 后 select[0,2,1]）----
        frag_C_tl = fx.select(thr_mma.make_fragment_C(bC_tl), [0, 2, 1])
        frag_C_tr = fx.select(thr_mma.make_fragment_C(bC_tr), [0, 2, 1])
        frag_C_bl = fx.select(thr_mma.make_fragment_C(bC_bl), [0, 2, 1])
        frag_C_br = fx.select(thr_mma.make_fragment_C(bC_br), [0, 2, 1])

        if const_expr(with_scale):
            def do_gemm(c_frag, b_frag, a_frag, scale_a_frag, scale_b_frag):
                c_value = c_frag.load().ir_value()
                b_value = vector.bitcast(T.vec(128, T.i8), b_frag.load().ir_value())
                a_value = vector.bitcast(T.vec(128, T.i8), a_frag.load().ir_value())
                scale_a = Vec(scale_a_frag.load())[0]
                scale_b = Vec(scale_b_frag.load())[0]
                for n0 in range_constexpr(4):
                    for m0 in range_constexpr(4):
                        c_offset = (m0 * 4 + n0) * 4
                        c_sub = vector.extract_strided_slice(
                            T.vec(4, T.f32), c_value, offsets=[c_offset], sizes=[4], strides=[1]
                        )
                        b_sub = vector.extract_strided_slice(
                            T.vec(32, T.i8), b_value, offsets=[n0 * 32], sizes=[32], strides=[1]
                        )
                        a_sub = vector.extract_strided_slice(
                            T.vec(32, T.i8), a_value, offsets=[m0 * 32], sizes=[32], strides=[1]
                        )
                        scaled_atom = fx.atom_set_value(scale_atoms[(n0, m0)], "scale_a", scale_b)
                        scaled_atom = fx.atom_set_value(scaled_atom, "scale_b", scale_a)
                        c_sub = _fly.mma_atom_call_ssa(
                            [T.vec(4, T.f32)], scaled_atom, b_sub, a_sub, c_sub
                        )
                        c_value = vector.insert_strided_slice(c_sub, c_value, [c_offset], [1])
                c_frag.store(c_value)
        else:
            def do_gemm(c_frag, b_frag, a_frag, scale_a_frag=None, scale_b_frag=None):
                fx.gemm(mma_atom, c_frag, b_frag, a_frag, c_frag)

        num_tiles = K // BLOCK_K
        assert num_tiles >= 4

        # ---- prologue：预取 tile0/tile1 到 LDS buf0/buf1，再把 buf0 的 B_l/A_t s2r 到寄存器 ----
        # 对标 gemm_v9：8 条 async g2s（2 tile × 4 array），waitvmcnt_barrier(24)，再 s2r B_l/A_t。
        def do_g2s(kk, buf):
            ki = fx.Int32(kk)
            if const_expr(not lds_swizzle and not preshuffle_b):
                raw_g2s(b_dma_rsrc, ki, b_l_dma_ptrs[buf], bid_y * 2)
            else:
                fx.copy(async_copy_atom, bL_g[None, None, None, ki], bL_s[buf])
            rocdl.sched_barrier(0)
            if const_expr(with_scale):
                raw_scale_g2s(scale_b_dma_rsrc, ki, scale_b_l_dma_ptrs[buf], bid_y * 2, N)
                rocdl.sched_barrier(0)

            if const_expr(not lds_swizzle and not preshuffle_b):
                raw_g2s(a_dma_rsrc, ki, a_t_dma_ptrs[buf], bid_x * 2)
            else:
                fx.copy(async_copy_atom, aT_g[None, None, None, ki], aT_s[buf])
            rocdl.sched_barrier(0)
            if const_expr(with_scale):
                raw_scale_g2s(scale_a_dma_rsrc, ki, scale_a_t_dma_ptrs[buf], bid_x * 2, M)
                rocdl.sched_barrier(0)

            if const_expr(not lds_swizzle and not preshuffle_b):
                raw_g2s(a_dma_rsrc, ki, a_b_dma_ptrs[buf], bid_x * 2 + 1)
            else:
                fx.copy(async_copy_atom, aB_g[None, None, None, ki], aB_s[buf])
            rocdl.sched_barrier(0)
            if const_expr(with_scale):
                raw_scale_g2s(scale_a_dma_rsrc, ki, scale_a_b_dma_ptrs[buf], bid_x * 2 + 1, M)
                rocdl.sched_barrier(0)

            if const_expr(not lds_swizzle and not preshuffle_b):
                raw_g2s(b_dma_rsrc, ki, b_r_dma_ptrs[buf], bid_y * 2 + 1)
            else:
                fx.copy(async_copy_atom, bR_g[None, None, None, ki], bR_s[buf])
            rocdl.sched_barrier(0)
            if const_expr(with_scale):
                raw_scale_g2s(scale_b_dma_rsrc, ki, scale_b_r_dma_ptrs[buf], bid_y * 2 + 1, N)
                rocdl.sched_barrier(0)

        vmcnt_per_phase = 5 if with_scale else 4

        do_g2s(0, 0)
        do_g2s(1, 1)
        waitvmcnt_barrier(vmcnt_per_phase * 6)
        fx.copy(lds_copy_atom, s2r_src0_B_l, dest_frag_B_l, pred=None)
        fx.copy(lds_copy_atom, s2r_src0_A_t, dest_frag_A_t, pred=None)
        if const_expr(with_scale):
            fx.copy(scale_lds_copy_atom, scale_b_l_src[0], scale_b_l_frag)
            fx.copy(scale_lds_copy_atom, scale_a_t_src[0], scale_a_t_frag)
        rocdl.sched_barrier(0)

        frag_C_tl.fill(0)
        frag_C_tr.fill(0)
        frag_C_bl.fill(0)
        frag_C_br.fill(0)
        rocdl.sched_barrier(0)
        acc_init = [frag_C_tl.load(), frag_C_tr.load(), frag_C_bl.load(), frag_C_br.load()]

        # 每 region 的 ds_read / vmem 计数（对标 gemm_4wave_950）：A operand 与 B operand
        # 的 fragment 大小不同，ds_read_b128 数量也不同；读 A 的 region 用 a_dsrd，读 B 的用
        # b_dsrd。之前对所有 region 统一传 dsrd=8，导致读 B 的 region 未被完整调度、访存交织
        # 退化并增加 wait cycles。
        a_dsrd = frag_A_t.load().numel * element_type.width // 8 // 16
        b_dsrd = frag_B_l.load().numel * element_type.width // 8 // 16
        a_vmem = (BLOCK_M * BLOCK_K * element_type.width // 8) // (256 * 16)
        b_vmem = (BLOCK_N * BLOCK_K * element_type.width // 8) // (256 * 16)

        # 每个 region：1 个象限 fx.gemm(C=B*A) + 下一 operand 的 s2r + 再下一块的 g2s，
        # 用 s2r_src0_*/s2r_src1_* 在 LDS buf0/buf1 之间 ping-pong；每个 slice 顺序与 gemm_v9 一致。
        # k-tile 内 4 个象限的顺序固定为：tl(A_t·B_l) -> bl(A_b·B_l) -> tr(A_t·B_r) -> br(A_b·B_r)。
        # 运行时循环（range + init/yield 累加器透传），不做常量展开。
        
        for kidx, states in range(0, num_tiles - 2, 2, init=acc_init):
        # for kiter in const_expr.range(0, K // BLOCK_K - 2, 2):
            frag_C_tl.store(states[0])
            frag_C_tr.store(states[1])
            frag_C_bl.store(states[2])
            frag_C_br.store(states[3])
            kiter = fx.Int32(kidx)

            # ---- k-tile = buf0：4 象限 ----
            do_gemm(frag_C_tl, frag_B_l, frag_A_t, scale_a_t_frag, scale_b_l_frag)
            waitvmcnt_barrier(vmcnt_per_phase * 5)
            fx.copy(lds_copy_atom, s2r_src0_A_b, dest_frag_A_b, pred=None)
            if const_expr(with_scale):
                fx.copy(scale_lds_copy_atom, scale_a_b_src[0], scale_a_b_frag)
            if const_expr(not lds_swizzle and not preshuffle_b):
                raw_g2s(b_dma_rsrc, kiter + 2, b_l_dma_ptrs[0], bid_y * 2)
            else:
                fx.copy(async_copy_atom, bL_g[None, None, None, kiter + 2], bL_s[0])
            if const_expr(with_scale):
                raw_scale_g2s(scale_b_dma_rsrc, kiter + 2, scale_b_l_dma_ptrs[0], bid_y * 2, N)
            hot_loop_scheduler_mainloop(0, b_vmem + int(with_scale), a_dsrd + int(with_scale))
            rocdl.sched_barrier(0)

            do_gemm(frag_C_bl, frag_B_l, frag_A_b, scale_a_b_frag, scale_b_l_frag)
            waitvmcnt_barrier(vmcnt_per_phase * 5)
            fx.copy(lds_copy_atom, s2r_src0_B_r, dest_frag_B_r, pred=None)
            if const_expr(with_scale):
                fx.copy(scale_lds_copy_atom, scale_b_r_src[0], scale_b_r_frag)
            if const_expr(not lds_swizzle and not preshuffle_b):
                raw_g2s(a_dma_rsrc, kiter + 2, a_t_dma_ptrs[0], bid_x * 2)
            else:
                fx.copy(async_copy_atom, aT_g[None, None, None, kiter + 2], aT_s[0])
            if const_expr(with_scale):
                raw_scale_g2s(scale_a_dma_rsrc, kiter + 2, scale_a_t_dma_ptrs[0], bid_x * 2, M)
            hot_loop_scheduler_mainloop(1, a_vmem + int(with_scale), b_dsrd + int(with_scale))
            rocdl.sched_barrier(0)

            do_gemm(frag_C_tr, frag_B_r, frag_A_t, scale_a_t_frag, scale_b_r_frag)
            waitvmcnt_barrier(vmcnt_per_phase * 5)
            fx.copy(lds_copy_atom, s2r_src1_B_l, dest_frag_B_l, pred=None)
            if const_expr(with_scale):
                fx.copy(scale_lds_copy_atom, scale_b_l_src[1], scale_b_l_frag)
            if const_expr(not lds_swizzle and not preshuffle_b):
                raw_g2s(a_dma_rsrc, kiter + 2, a_b_dma_ptrs[0], bid_x * 2 + 1)
            else:
                fx.copy(async_copy_atom, aB_g[None, None, None, kiter + 2], aB_s[0])
            if const_expr(with_scale):
                raw_scale_g2s(scale_a_dma_rsrc, kiter + 2, scale_a_b_dma_ptrs[0], bid_x * 2 + 1, M)
            hot_loop_scheduler_mainloop(2, a_vmem + int(with_scale), b_dsrd + int(with_scale))
            rocdl.sched_barrier(0)

            do_gemm(frag_C_br, frag_B_r, frag_A_b, scale_a_b_frag, scale_b_r_frag)
            waitvmcnt_barrier(vmcnt_per_phase * 5)
            fx.copy(lds_copy_atom, s2r_src1_A_t, dest_frag_A_t, pred=None)
            if const_expr(with_scale):
                fx.copy(scale_lds_copy_atom, scale_a_t_src[1], scale_a_t_frag)
            if const_expr(not lds_swizzle and not preshuffle_b):
                raw_g2s(b_dma_rsrc, kiter + 2, b_r_dma_ptrs[0], bid_y * 2 + 1)
            else:
                fx.copy(async_copy_atom, bR_g[None, None, None, kiter + 2], bR_s[0])
            if const_expr(with_scale):
                raw_scale_g2s(scale_b_dma_rsrc, kiter + 2, scale_b_r_dma_ptrs[0], bid_y * 2 + 1, N)
            hot_loop_scheduler_mainloop(3, b_vmem + int(with_scale), a_dsrd + int(with_scale))
            rocdl.sched_barrier(0)

            # ---- k-tile = buf1：4 象限 ----
            do_gemm(frag_C_tl, frag_B_l, frag_A_t, scale_a_t_frag, scale_b_l_frag)
            waitvmcnt_barrier(vmcnt_per_phase * 5)
            fx.copy(lds_copy_atom, s2r_src1_A_b, dest_frag_A_b, pred=None)
            if const_expr(with_scale):
                fx.copy(scale_lds_copy_atom, scale_a_b_src[1], scale_a_b_frag)
            if const_expr(not lds_swizzle and not preshuffle_b):
                raw_g2s(b_dma_rsrc, kiter + 3, b_l_dma_ptrs[1], bid_y * 2)
            else:
                fx.copy(async_copy_atom, bL_g[None, None, None, kiter + 3], bL_s[1])
            if const_expr(with_scale):
                raw_scale_g2s(scale_b_dma_rsrc, kiter + 3, scale_b_l_dma_ptrs[1], bid_y * 2, N)
            hot_loop_scheduler_mainloop(4, b_vmem + int(with_scale), a_dsrd + int(with_scale))
            rocdl.sched_barrier(0)

            do_gemm(frag_C_bl, frag_B_l, frag_A_b, scale_a_b_frag, scale_b_l_frag)
            waitvmcnt_barrier(vmcnt_per_phase * 5)
            fx.copy(lds_copy_atom, s2r_src1_B_r, dest_frag_B_r, pred=None)
            if const_expr(with_scale):
                fx.copy(scale_lds_copy_atom, scale_b_r_src[1], scale_b_r_frag)
            if const_expr(not lds_swizzle and not preshuffle_b):
                raw_g2s(a_dma_rsrc, kiter + 3, a_t_dma_ptrs[1], bid_x * 2)
            else:
                fx.copy(async_copy_atom, aT_g[None, None, None, kiter + 3], aT_s[1])
            if const_expr(with_scale):
                raw_scale_g2s(scale_a_dma_rsrc, kiter + 3, scale_a_t_dma_ptrs[1], bid_x * 2, M)
            hot_loop_scheduler_mainloop(5, a_vmem + int(with_scale), b_dsrd + int(with_scale))
            rocdl.sched_barrier(0)

            do_gemm(frag_C_tr, frag_B_r, frag_A_t, scale_a_t_frag, scale_b_r_frag)
            waitvmcnt_barrier(vmcnt_per_phase * 5)
            fx.copy(lds_copy_atom, s2r_src0_B_l, dest_frag_B_l, pred=None)
            if const_expr(with_scale):
                fx.copy(scale_lds_copy_atom, scale_b_l_src[0], scale_b_l_frag)
            if const_expr(not lds_swizzle and not preshuffle_b):
                raw_g2s(a_dma_rsrc, kiter + 3, a_b_dma_ptrs[1], bid_x * 2 + 1)
            else:
                fx.copy(async_copy_atom, aB_g[None, None, None, kiter + 3], aB_s[1])
            if const_expr(with_scale):
                raw_scale_g2s(scale_a_dma_rsrc, kiter + 3, scale_a_b_dma_ptrs[1], bid_x * 2 + 1, M)
            hot_loop_scheduler_mainloop(6, a_vmem + int(with_scale), b_dsrd + int(with_scale))
            rocdl.sched_barrier(0)

            do_gemm(frag_C_br, frag_B_r, frag_A_b, scale_a_b_frag, scale_b_r_frag)
            waitvmcnt_barrier(vmcnt_per_phase * 5)
            fx.copy(lds_copy_atom, s2r_src0_A_t, dest_frag_A_t, pred=None)
            if const_expr(with_scale):
                fx.copy(scale_lds_copy_atom, scale_a_t_src[0], scale_a_t_frag)
            if const_expr(not lds_swizzle and not preshuffle_b):
                raw_g2s(b_dma_rsrc, kiter + 3, b_r_dma_ptrs[1], bid_y * 2 + 1)
            else:
                fx.copy(async_copy_atom, bR_g[None, None, None, kiter + 3], bR_s[1])
            if const_expr(with_scale):
                raw_scale_g2s(scale_b_dma_rsrc, kiter + 3, scale_b_r_dma_ptrs[1], bid_y * 2 + 1, N)
            hot_loop_scheduler_mainloop(7, b_vmem + int(with_scale), a_dsrd + int(with_scale))
            rocdl.sched_barrier(0)

            results = yield [frag_C_tl.load(), frag_C_tr.load(), frag_C_bl.load(), frag_C_br.load()]
        frag_C_tl.store(results[0])
        frag_C_tr.store(results[1])
        frag_C_bl.store(results[2])
        frag_C_br.store(results[3])

        # ---- epilogue：最后 2 个 k-tile（buf0 / buf1），无 g2s，只做 s2r + gemm ----
        # buf0 的 4 象限
        waitvmcnt_barrier(vmcnt_per_phase * 5)
        do_gemm(frag_C_tl, frag_B_l, frag_A_t, scale_a_t_frag, scale_b_l_frag)
        fx.copy(lds_copy_atom, s2r_src0_A_b, dest_frag_A_b, pred=None)
        if const_expr(with_scale):
            fx.copy(scale_lds_copy_atom, scale_a_b_src[0], scale_a_b_frag)
        hot_loop_scheduler_mainloop(0, 0, 8 + int(with_scale))
        rocdl.sched_barrier(0)

        waitvmcnt_barrier(vmcnt_per_phase * 4)
        do_gemm(frag_C_bl, frag_B_l, frag_A_b, scale_a_b_frag, scale_b_l_frag)
        fx.copy(lds_copy_atom, s2r_src0_B_r, dest_frag_B_r, pred=None)
        if const_expr(with_scale):
            fx.copy(scale_lds_copy_atom, scale_b_r_src[0], scale_b_r_frag)
        hot_loop_scheduler_mainloop(1, 0, 8 + int(with_scale))
        rocdl.sched_barrier(0)

        waitvmcnt_barrier(vmcnt_per_phase * 3)
        do_gemm(frag_C_tr, frag_B_r, frag_A_t, scale_a_t_frag, scale_b_r_frag)
        fx.copy(lds_copy_atom, s2r_src1_B_l, dest_frag_B_l, pred=None)
        if const_expr(with_scale):
            fx.copy(scale_lds_copy_atom, scale_b_l_src[1], scale_b_l_frag)
        hot_loop_scheduler_mainloop(2, 0, 8 + int(with_scale))
        rocdl.sched_barrier(0)

        waitvmcnt_barrier(vmcnt_per_phase * 2)
        do_gemm(frag_C_br, frag_B_r, frag_A_b, scale_a_b_frag, scale_b_r_frag)
        fx.copy(lds_copy_atom, s2r_src1_A_t, dest_frag_A_t, pred=None)
        if const_expr(with_scale):
            fx.copy(scale_lds_copy_atom, scale_a_t_src[1], scale_a_t_frag)
        hot_loop_scheduler_mainloop(3, 0, 8 + int(with_scale))
        rocdl.sched_barrier(0)

        # ---- store_quadrant 定义提前（放到 buf1 尾部之前），供 store 与最后的 MFMA 交织 ----
        if const_expr(permlane_epilogue):
            # permlane：相邻两个 16x16 tile 经 permlane16_swap 重排后，每 lane 一次写 8 个连续 bf16
            # （128-bit 合并写）。fp8 op1=B 走 fragment_A 槽 => C 的 wave 朝向相对 bf16 转置，
            # 故 wave_m/wave_n 相对 bf16 permlane 互换（wave_m=wave_id//2, wave_n=wave_id%2）。
            pair_type = ir.Type.parse("!llvm.struct<(i32, i32)>")
            lane_id = tid % 64
            wave_id = tid // 64
            wave_m = wave_id // 2
            wave_n = wave_id % 2
            lane_group = lane_id // 16
            fragment_mode_0_repeat = TILE_N // 64
            fragment_mode_1_repeat = TILE_M // 64

            def store_quadrant(c_frag, bC, quadrant_m, quadrant_n):
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
                        row = bid_x * TILE_M + quadrant_m * (TILE_M // 2) + row_repeat * 32 + wave_m * 16 + lane_id % 16
                        col = (
                            bid_y * TILE_N
                            + quadrant_n * (TILE_N // 2)
                            + col_repeat * 32
                            + lane_group % 2 * 32
                            + wave_n * 16
                            + lane_group // 2 * 8
                        )
                        byte_offset = (row * N + col) * 2
                        fx.buffer_ops.buffer_store(packed, c_store_rsrc, byte_offset, offset_is_bytes=True)
        else:
            # 注意：fp8 op1=B 走 make_fragment_A 槽，C 的 wave 朝向相对 bf16 转置，
            # 故 c_tv 的两个 wave 维 stride 需交换为 (512, 16)。
            store_atom_bf16 = fx.make_copy_atom(fx.rocdl.BufferCopy64b(), fx.BFloat16)
            c_layout_w = fx.make_tiled_copy(
                store_atom_bf16,
                fx.make_layout(((16, 4, 2, 2), 4), ((1, 128, 512, 16), 32)),
                fx.make_tile(32, 32),
            )
            store_thr = c_layout_w.get_slice(tid)

            def store_quadrant(c_frag, bC, quadrant_m=0, quadrant_n=0):
                c_sel = fx.select(c_frag, [0, 2, 1])
                c_bf16 = fx.make_fragment_like(c_sel, dtype=fx.BFloat16)
                c_bf16.store(c_sel.load().to(fx.BFloat16))
                fx.copy(store_atom_bf16, store_thr.retile(c_bf16), store_thr.partition_D(bC))

        # buf1 的 4 象限。store_overlap 时把每象限的 store 与后一象限的 MFMA 交织，
        # 用 MFMA 计算掩盖 buffer_store 的写延迟（对标 bf16 v9 scheduler_store_overlap）；
        # 否则先算完 4 象限，再统一 store（不交织，用于对照）。
        if const_expr(store_overlap):
            waitvmcnt_barrier(vmcnt_per_phase * 1)
            do_gemm(frag_C_tl, frag_B_l, frag_A_t, scale_a_t_frag, scale_b_l_frag)
            fx.copy(lds_copy_atom, s2r_src1_A_b, dest_frag_A_b, pred=None)
            if const_expr(with_scale):
                fx.copy(scale_lds_copy_atom, scale_a_b_src[1], scale_a_b_frag)
            hot_loop_scheduler_mainloop(4, 0, 8 + int(with_scale))
            rocdl.sched_barrier(0)

            waitvmcnt_barrier(0)
            # bl 的 MFMA 与 tl 的 store 互相掩盖
            do_gemm(frag_C_bl, frag_B_l, frag_A_b, scale_a_b_frag, scale_b_l_frag)
            fx.copy(lds_copy_atom, s2r_src1_B_r, dest_frag_B_r, pred=None)
            if const_expr(with_scale):
                fx.copy(scale_lds_copy_atom, scale_b_r_src[1], scale_b_r_frag)
            store_quadrant(frag_C_tl, bC_tl, 0, 0)
            scheduler_store_overlap(5)
            rocdl.sched_barrier(0)

            # tr 的 MFMA 掩盖 bl 的 store
            do_gemm(frag_C_tr, frag_B_r, frag_A_t, scale_a_t_frag, scale_b_r_frag)
            store_quadrant(frag_C_bl, bC_bl, 1, 0)
            scheduler_store_overlap(6)
            rocdl.sched_barrier(0)

            # br 的 MFMA 掩盖 tr 的 store
            do_gemm(frag_C_br, frag_B_r, frag_A_b, scale_a_b_frag, scale_b_r_frag)
            store_quadrant(frag_C_tr, bC_tr, 0, 1)
            scheduler_store_overlap(7)
            rocdl.sched_barrier(0)

            # 最后 br 单独 store
            store_quadrant(frag_C_br, bC_br, 1, 1)
        else:
            waitvmcnt_barrier(vmcnt_per_phase)
            do_gemm(frag_C_tl, frag_B_l, frag_A_t, scale_a_t_frag, scale_b_l_frag)
            fx.copy(lds_copy_atom, s2r_src1_A_b, dest_frag_A_b, pred=None)
            if const_expr(with_scale):
                fx.copy(scale_lds_copy_atom, scale_a_b_src[1], scale_a_b_frag)
            hot_loop_scheduler_mainloop(4, 0, 8 + int(with_scale))
            rocdl.sched_barrier(0)

            waitvmcnt_barrier(0)
            do_gemm(frag_C_bl, frag_B_l, frag_A_b, scale_a_b_frag, scale_b_l_frag)
            fx.copy(lds_copy_atom, s2r_src1_B_r, dest_frag_B_r, pred=None)
            if const_expr(with_scale):
                fx.copy(scale_lds_copy_atom, scale_b_r_src[1], scale_b_r_frag)
            hot_loop_scheduler_mainloop(5, 0, 8 + int(with_scale))
            rocdl.sched_barrier(0)

            do_gemm(frag_C_tr, frag_B_r, frag_A_t, scale_a_t_frag, scale_b_r_frag)
            hot_loop_scheduler_mainloop(6, 0, 0)
            rocdl.sched_barrier(0)
            do_gemm(frag_C_br, frag_B_r, frag_A_b, scale_a_b_frag, scale_b_r_frag)
            hot_loop_scheduler_mainloop(7, 0, 0)
            rocdl.sched_barrier(0)

            store_quadrant(frag_C_tl, bC_tl, 0, 0)
            store_quadrant(frag_C_tr, bC_tr, 0, 1)
            store_quadrant(frag_C_bl, bC_bl, 1, 0)
            store_quadrant(frag_C_br, bC_br, 1, 1)

    @flyc.jit
    def launch_gemm(
        A: fx.Tensor,
        B: fx.Tensor,
        ScaleA: fx.Tensor,
        ScaleB: fx.Tensor,
        C: fx.Tensor,
        M: int,
        stream: fx.Stream = fx.Stream(None),
    ):
        # 累加器钉到 AGPR（force-agpr）+ mfma-vgpr-form=False：避免 C 累加器 VGPR/AGPR 混放导致的
        # v_accvgpr 拷贝与 VGPR 压力（对标 test_gemm_v9.py）。
        value_attrs = {
            "rocdl.waves_per_eu": 1,
            "passthrough": [["amdgpu-agpr-alloc", "256,256"]],
        }
        gemm_kernel(A, B, ScaleA, ScaleB, C, M, value_attrs=value_attrs).launch(
            grid=(div_up(M, TILE_M) * div_up(N, TILE_N), 1, 1), block=(256, 1, 1), stream=stream
        )

    launch_gemm.compile_hints["llvm_options"] = {"amdgpu-mfma-vgpr-form": False}
    return launch_gemm


# =========================== test / perf ===========================
TILE_M = 256
TILE_N = 256
TILE_K = 128
M = int(os.environ.get("GEMM_M", 8192))
N = int(os.environ.get("GEMM_N", 8192))
K = int(os.environ.get("GEMM_K", 8192))

# permlane 存储 / store 与 MFMA 交织：可用环境变量覆盖默认值（对标 bf16 v9 run_test 结构）。
PERMLANE_EPILOGUE = _env_flag("PERMLANE", "1")
STORE_OVERLAP = _env_flag("STORE_OVERLAP")

import pyhip


def _load_shuffle_weight():
    # host 端 shuffle_weight 在 tests.utils，延迟加载避免非 preshuffle 路径依赖它。
    # fp8 kWidth=16、BLOCK_K=128 => shuffle_weight layout=(16, 64)（BK=IK*2=128, K=16//1B）。
    import sys as _sys, os.path as _osp
    _flydsl_root = _osp.abspath(_osp.join(_osp.dirname(__file__), "..", ".."))
    if _flydsl_root not in _sys.path:
        _sys.path.insert(0, _flydsl_root)
    from tests.utils import shuffle_weight
    return shuffle_weight


def _load_mxfp8_quant():
    from aiter import dtypes
    from aiter.ops.quant import per_1x32_mx_quant_hip
    from aiter.utility.fp4_utils import e8m0_to_f32
    return per_1x32_mx_quant_hip, dtypes, e8m0_to_f32


def run_test(M, N, K, USE_SWIZZLE=False, PRESHUFFLE_B=False, perf=False,
             TILEM=256, TILEN=256, TILEK=128, permlane_output=True, store_overlap=False,
             with_scale=False, run_count=50, data_clones=32):
    shuffle_weight = _load_shuffle_weight() if PRESHUFFLE_B else None
    mxfp8_quant = _load_mxfp8_quant() if with_scale else None

    def _shuffle_b(x):
        return shuffle_weight(x, layout=(16, 64)) if PRESHUFFLE_B else x

    def _permute_scale(scale):
        scale = scale.view(torch.uint8)
        rows, groups = scale.shape
        permuted = scale.view(rows // 128, 4, 32, groups).permute(3, 0, 2, 1).contiguous().view(-1)
        # BufferCopyLDS32b reads an 8-group window although each BK128 consumes 4.
        # Keep the final workaround overread inside the allocation.
        padding = torch.full((rows * 4,), 127, device=scale.device, dtype=torch.uint8)
        return torch.cat((permuted, padding)).view(torch.int32)

    def _random_fp8(shape):
        return torch.randn(shape, device="cuda", dtype=torch.float32).to(torch.float8_e4m3fn)

    def _quant_mxfp8(x):
        per_1x32_mx_quant_hip, dtypes, _ = mxfp8_quant
        return per_1x32_mx_quant_hip(
            x.to(torch.bfloat16),
            quant_dtype=dtypes.fp8,
            scale_type=dtypes.fp8_e8m0,
            shuffle=False,
        )

    def _dequant_mxfp8(x, scale):
        _, _, e8m0_to_f32 = mxfp8_quant
        scale_f32 = e8m0_to_f32(scale).repeat_interleave(32, dim=1)
        return x.float() * scale_f32

    if with_scale:
        a, scale_a_raw = _quant_mxfp8(torch.randn((M, K), device="cuda") * 0.75)
        b, scale_b_raw = _quant_mxfp8(torch.randn((N, K), device="cuda") * 3.0)
        ref = _dequant_mxfp8(a, scale_a_raw) @ _dequant_mxfp8(b, scale_b_raw).t()
    else:
        a = _random_fp8((M, K))
        b = _random_fp8((N, K))
        scale_a_raw = scale_b_raw = None
        ref = a.float() @ b.float().t()
    out = torch.zeros((M, N), device="cuda", dtype=torch.bfloat16)
    weight = _shuffle_b(b)  # preshuffle 时喂 shuffle 后的 B；ref 仍用原始 b
    scale_a = (
        _permute_scale(scale_a_raw)
        if with_scale else torch.empty(1, device="cuda", dtype=torch.uint8)
    )
    scale_b = (
        _permute_scale(scale_b_raw)
        if with_scale else torch.empty(1, device="cuda", dtype=torch.uint8)
    )
    stream = torch.cuda.current_stream()
    args = (a.view(torch.int8).view(-1), weight.view(torch.int8).view(-1), scale_a, scale_b, out.view(-1), M, stream)

    launcher = compile_gemm_fp8(
        TILEM, TILEN, TILEK, N, K,
        lds_swizzle=USE_SWIZZLE,
        preshuffle_b=PRESHUFFLE_B,
        permlane_epilogue=permlane_output,
        store_overlap=store_overlap,
        with_scale=with_scale,
    )
    kernel = flyc.compile[{"opt_level": 2}](launcher, *args)
    kernel(*args)
    torch.cuda.synchronize()

    ref_bf16 = ref.to(torch.bfloat16)
    diff = pyhip.calc_diff(out.float(), ref_bf16, diff_thr=0.00001)
    is_correct = diff <= 0.00001
    print(f"####M={M} N={N} K={K} {USE_SWIZZLE=} {PRESHUFFLE_B=} {is_correct=} {diff=}")
    if not torch.allclose(out, ref_bf16, rtol=0.02, atol=0.01):
        abs_err = (out.float() - ref_bf16.float()).abs()
        tolerance = 0.01 + 0.02 * ref_bf16.float().abs()
        max_index = abs_err.argmax().item()
        max_row, max_col = divmod(max_index, N)
        print(
            f"  strict_allclose=False  max_abs_err={abs_err.max().item():.3f}  "
            f"outside_tolerance={(abs_err > tolerance).sum().item()}/{M*N}"
        )
        print(
            f"  max_error_at=({max_row}, {max_col})  "
            f"ref_fp32={ref[max_row, max_col].item()}  "
            f"ref_bf16={ref_bf16[max_row, max_col].item()}  "
            f"result={out[max_row, max_col].item()}  "
            f"abs_err={abs_err[max_row, max_col].item()}"
        )

    if not perf:
        return is_correct

    # ---- perf（多份数据轮转，排除 L2 cache 影响）----
    # 单份 A+B 只有几十 MB，反复喂同一份会常驻 L2 -> 高估 TFLOPS；轮转多份（远大于 L2）确保 cold data。
    As = [_random_fp8((M, K)) for _ in range(data_clones)]
    Bs = [_shuffle_b(_random_fp8((N, K))) for _ in range(data_clones)]
    Cs = [torch.zeros((M, N), device="cuda", dtype=torch.bfloat16) for _ in range(data_clones)]
    ScaleAs = [
        _permute_scale(torch.zeros((M, K // 32), device="cuda", dtype=torch.uint8))
        if with_scale else scale_a
        for _ in range(data_clones)
    ]
    ScaleBs = [
        _permute_scale(torch.zeros((N, K // 32), device="cuda", dtype=torch.uint8))
        if with_scale else scale_b
        for _ in range(data_clones)
    ]
    arg_sets = [
        (As[i].view(torch.int8).view(-1), Bs[i].view(torch.int8).view(-1), ScaleAs[i], ScaleBs[i], Cs[i].view(-1), M, stream)
        for i in range(data_clones)
    ]

    flops = 2 * M * N * K
    mem_bytes = (M * K + N * K) * 1 + M * N * 2  # fp8 A+B (1B) + bf16 C (2B)

    # warmup（轮转，把所有 clone 都碰一遍）
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
    tflops = flops / (best_ms * 1e-3) / 1e12
    bw_gbs = mem_bytes / (best_ms * 1e-3) / 1e9
    print(f"\n=== perf  M={M} N={N} K={K} USE_SWIZZLE={USE_SWIZZLE} PRESHUFFLE_B={PRESHUFFLE_B} {with_scale=} ===")
    print(f"gemm:  {best_ms*1e3:.1f} us  {tflops:.2f} TFLOPS  {bw_gbs:.1f} GB/s")
    return is_correct


if __name__ == "__main__":
    props = torch.cuda.get_device_properties()
    assert "950" in props.gcnArchName, "fp8 MFMA_Scale 需要 gfx950"
    torch.manual_seed(0)
    # run_test(M=M, N=N, K=K, USE_SWIZZLE=0, PRESHUFFLE_B=0, perf=0, TILEK=TILE_K, permlane_output=PERMLANE_EPILOGUE, store_overlap=STORE_OVERLAP, with_scale = False)
    run_test(M=M, N=N, K=K, USE_SWIZZLE=0, PRESHUFFLE_B=0, perf=0, TILEK=TILE_K, permlane_output=PERMLANE_EPILOGUE, store_overlap=STORE_OVERLAP, with_scale = True)
    # run_test(M=M, N=N, K=K, USE_SWIZZLE=1, PRESHUFFLE_B=0, perf=1, TILEK=TILE_K, permlane_output=PERMLANE_EPILOGUE, store_overlap=STORE_OVERLAP)
    # run_test(M=M, N=N, K=K, USE_SWIZZLE=0, PRESHUFFLE_B=1, perf=1, TILEK=TILE_K, permlane_output=PERMLANE_EPILOGUE, store_overlap=STORE_OVERLAP)
    # run_test(M=M, N=N, K=K, USE_SWIZZLE=1, PRESHUFFLE_B=1, perf=1, TILEK=TILE_K, permlane_output=PERMLANE_EPILOGUE, store_overlap=STORE_OVERLAP)
