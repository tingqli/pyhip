# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
#
# fp8 GEMM (C = B * A, 输出 bf16)，按 test_gemm_v9.py 的方式用 tile + layout 抽象编写
# （flat_divide / make_tiled_copy / make_tiled_mma / make_fragment / fx.copy / fx.gemm），
# LDS 读取保留 layout 抽象；G2S 使用预计算 byte offset 的 raw DMA。
#   - BLOCK_M=BLOCK_N=BLOCK_K=128, TILE_M=TILE_N=256, 4-wave, 2x2 quadrant
#   - MFMA 指令 V_MFMA_SCALE_F32_16X16X128_F8F6F4（scale=0 => 不含 scale）
#   - A/B 均普通输入 + LDS bank-conflict 消解：padding（默认，[[1024,32]] 单 padding，对标 bf16 v9）
#     或 swizzle（lds_swizzle=True, MBase=4）。LDS ping-pong 双缓冲 + 寄存器软件流水。
#   - 约定：A 走 make_fragment_B，B 走 make_fragment_A；fx.gemm(mma, C, frag_B, frag_A)。
#   - 通过 lds_swizzle 参数切到 swizzle 版本。
#
# CDNA4 (gfx950) only: FP8/MXFP8 GEMM with optional MXFP4 weights.

from functools import cache

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr.typing import BFloat16, Float8E4M3FN, Float32, Int8, Int32, T, Vector
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl, arith
from flydsl.expr.typing import Vector as Vec
from flydsl._mlir import ir
from flydsl._mlir.dialects import fly as _fly
from flydsl._mlir.dialects import llvm as _llvm

# Raw vector SSA is retained at the existing SSA-returning MFMA boundary.
from flydsl._mlir.dialects import vector
from flydsl.compiler.ast_rewriter import ASTRewriter

from .common import require_cdna4

__all__ = ["compile_gemm_fp8"]


def div_up(x, y):
    return (x + y - 1) // y


def _buffer_resource(tensor, num_records_bytes):
    buffer = rocdl.make_buffer_tensor(tensor, num_records_bytes=num_records_bytes)
    return rocdl.get_buffer_rsrc(fx.get_iter(buffer))


def _lds_byte_ptr(ptr, byte_offset):
    return fx.to_llvm_ptr(
        fx.add_offset(fx.recast_iter(fx.Uint8, ptr), fx.make_int_tuple(byte_offset))
    )


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
    # Compiler fence only: keep the compute prefix after this phase's waits.
    rocdl.sched_barrier(0)


def _schedule_compute(group_id, dsrd_ops, vmem_ops):
    """Compute prefix, sequential V1/M2/D2/M1 bundles and a compute tail.

    Counts already include scales. Skip exhausted memory groups; never emit
    zero-count hints or pad the actual 16 MFMAs. Scaled FP8 x FP8 (V5, D9)
    uses an M1 prefix so the final D1 also has an M1 followup:
    1 + 5 * 2 + 5 * 1 = 16. All other modes, including V0 drain, keep M2.
    """
    assert vmem_ops >= 0 and dsrd_ops >= 0
    mfma_prefix = 1 if dsrd_ops == 9 and vmem_ops == 5 else 2
    dsrd_groups = (dsrd_ops + 1) // 2
    dsrd_mfmas = min(dsrd_groups, 16 - mfma_prefix - 2 * vmem_ops)
    assert dsrd_mfmas >= 0
    mfma_tail = 16 - mfma_prefix - 2 * vmem_ops - dsrd_mfmas
    rocdl.sched_group_barrier(rocdl.mask_mfma, mfma_prefix, group_id)
    for i in range_constexpr(max(vmem_ops, dsrd_groups)):
        if const_expr(i < vmem_ops):
            rocdl.sched_group_barrier(rocdl.mask_vmem_rd, 1, group_id)
            rocdl.sched_group_barrier(rocdl.mask_mfma, 2, group_id)
        if const_expr(i < dsrd_groups):
            rocdl.sched_group_barrier(
                rocdl.mask_dsrd, min(2, dsrd_ops - 2 * i), group_id
            )
            if const_expr(i < dsrd_mfmas):
                rocdl.sched_group_barrier(rocdl.mask_mfma, 1, group_id)
    if const_expr(mfma_tail > 0):
        rocdl.sched_group_barrier(rocdl.mask_mfma, mfma_tail, group_id)


# VMEM_WRITE / VALU 掩码（flydsl 未导出 vmem_wr 常量，直接用 bit 值）。
_MASK_VALU = 0x002
_MASK_VMEM_WR = 0x040


def scheduler_store_overlap(group_id, dsrd_ops=0):
    # MFMA 领先：每 2 条 MFMA 穿插 store 的 VALU(cvt/permlane) 与 buffer_store(vmem_wr)，
    # 用 MFMA 计算掩盖 store 的写延迟（fp8 每象限 16 条 MFMA、8 次 buffer_store）。
    # MFMA 必须领先，否则 store 会挡住计算流水。
    # Phase 5 also reads the last B operand/scale. Pair those reads with the
    # existing M2 groups, without adding MFMAs or changing the eight stores.
    for i in range_constexpr(8):
        rocdl.sched_group_barrier(rocdl.mask_mfma, 2, group_id)
        if const_expr(2 * i < dsrd_ops):
            rocdl.sched_group_barrier(
                rocdl.mask_dsrd, min(2, dsrd_ops - 2 * i), group_id
            )
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
    b_lds_swizzle=None,
    preshuffle_b=False,
    permlane_epilogue=True,
    store_overlap=False,
    with_scale=False,
    b_mxfp4=False,
):
    """Return the cached launcher for this CDNA4 target and static configuration."""
    return _compile_gemm_fp8_cached(
        require_cdna4(),
        TILE_M,
        TILE_N,
        TILE_K,
        N,
        K,
        pid_swizzle,
        lds_swizzle,
        b_lds_swizzle,
        preshuffle_b,
        permlane_epilogue,
        store_overlap,
        with_scale,
        b_mxfp4,
    )


@cache
def _compile_gemm_fp8_cached(
    target,
    TILE_M,
    TILE_N,
    TILE_K,
    N,
    K,
    pid_swizzle,
    lds_swizzle,
    b_lds_swizzle,
    preshuffle_b,
    permlane_epilogue,
    store_overlap,
    with_scale,
    b_mxfp4,
):
    del target  # JIT launchers capture their target; keep it in the cache key.
    BLOCK_M = TILE_M // 2
    BLOCK_N = TILE_N // 2
    BLOCK_K = TILE_K
    assert BLOCK_K == 128
    assert K % 256 == 0
    assert N % 8 == 0
    # 目前不支持mxfp4 weight preshuffle.
    # preshuffle只支持mxfp8 for now.
    assert not b_mxfp4 or not preshuffle_b
    if b_lds_swizzle is None:
        b_lds_swizzle = True if b_mxfp4 else lds_swizzle
    if not b_mxfp4 and b_lds_swizzle != lds_swizzle:
        raise ValueError("independent B LDS swizzle is only supported for MXFP4")
    element_type = fx.Float8E4M3FN
    b_element_type = fx.Float4E2M1FN if b_mxfp4 else element_type
    # These specs describe LDS BK128 tiles. Raw DMA inverts the tile-local
    # swizzle before applying the global row stride K; no K-dependent swizzle.
    #     // A generic Swizzle functor
    # /* 0bxxxxxxxxxxxxxxxYYYxxxxxxxZZZxxxx
    #  *                               ^--^ MBase is the number of least-sig bits to keep constant
    #  *                  ^-^       ^-^     BBits is the number of bits in the mask
    #  *                    ^---------^     SShift is the distance to shift the YYY mask
    #  *                                       (pos shifts YYY to the right, neg shifts YYY to the left)
    #  *
    #  * e.g. Given
    #  * 0bxxxxxxxxxxxxxxxxYYxxxxxxxxxZZxxx
    #  * the result is
    #  * 0bxxxxxxxxxxxxxxxxYYxxxxxxxxxAAxxx where AA = ZZ xor YY
    #  */
    if b_mxfp4:
        swizzle_a_specs = ((3, 4, 3),)
        swizzle_b_specs = ((1, 5, 2),)
    else:
        swizzle_a_specs = ((3, 4, 4),)
        swizzle_b_specs = swizzle_a_specs
    # Retain the existing preshuffle/swizzle mode restriction. Raw DMA below
    # uses the same tile-local swizzle map for power-of-two and other K sizes.
    assert not (lds_swizzle and (K & (K - 1)) != 0) or not preshuffle_b

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
                pid = (
                    tall_xcds * pids_per_xcd
                    + (xcd - tall_xcds) * (pids_per_xcd - 1)
                    + local_pid
                )
        if const_expr(GROUP_SIZE_M == 1):
            pid_m = pid // num_pid_n
            pid_n = pid % num_pid_n
        else:
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            group_id = pid // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            remaining_pid_m = num_pid_m - first_pid_m
            group_size_m = (remaining_pid_m < GROUP_SIZE_M).select(
                remaining_pid_m, GROUP_SIZE_M
            )
            pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
            pid_n = (pid % num_pid_in_group) // group_size_m
        return pid_m, pid_n

    get_pids_950 = ASTRewriter.transform(_get_pids_950)

    # A 的 16-row footprint 固定为 2112 B。A8W8 使用 [[1024,16],[2048,32]]；
    # Hybrid 的 MFMA K permutation 改变 lane-to-K 映射，使用等容量的 [[1024,32]]。

    # A8w8 without scale:        1024+16, 2048+32, 双padding 所以2048个元素总共padding 16*2+32 = 64
    # A8W8 with scale:           1024+32, 单pading, 所以2048个元素总共padding 32*2 = 64
    # A8w4 with/without scale:   same with A8W8 with scale
    A_PAD = 32
    A_GROUP = 8 * BLOCK_K + A_PAD  # 1056
    a_group8 = 8 * BLOCK_K + (A_PAD if with_scale or b_mxfp4 else 16)

    # 无论什么方案， 每2048个A padding 64. 所以a_group16是一样的。
    a_group16 = 2 * A_GROUP
    a_lds_elems = (BLOCK_M // 8) * A_GROUP  # 16*1056 = 16896
    if b_mxfp4 and not b_lds_swizzle:
        # Keep each 2048-element (16-row) block contiguous for one full-wave DMA.
        # 64/128/256-element padding have the same conflict count; 64 was fastest
        # at M=N=K=8192 and has the smallest LDS footprint.
        b_group16 = 16 * BLOCK_K + 64
        b_lds_elems = (BLOCK_N // 16) * b_group16
    else:
        b_lds_elems = BLOCK_N * BLOCK_K if b_mxfp4 else (BLOCK_N // 8) * A_GROUP
    scale_lds_bytes = 128 * 8

    if with_scale:

        @fx.struct
        class LDS:
            a_t0: fx.Array[Float8E4M3FN, a_lds_elems, 16]
            a_t1: fx.Array[Float8E4M3FN, a_lds_elems, 16]
            a_b0: fx.Array[Float8E4M3FN, a_lds_elems, 16]
            a_b1: fx.Array[Float8E4M3FN, a_lds_elems, 16]
            b_l0: fx.Array[b_element_type, b_lds_elems, 16]
            b_l1: fx.Array[b_element_type, b_lds_elems, 16]
            b_r0: fx.Array[b_element_type, b_lds_elems, 16]
            b_r1: fx.Array[b_element_type, b_lds_elems, 16]
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
            b_l0: fx.Array[b_element_type, b_lds_elems, 16]
            b_l1: fx.Array[b_element_type, b_lds_elems, 16]
            b_r0: fx.Array[b_element_type, b_lds_elems, 16]
            b_r1: fx.Array[b_element_type, b_lds_elems, 16]

    @flyc.kernel(known_block_size=[256, 1, 1])
    def gemm_kernel(
        argA: fx.Tensor,
        argB: fx.Tensor,
        argScaleA: fx.Tensor,
        argScaleB: fx.Tensor,
        argC: fx.Tensor,
        M: fx.Int32,
    ):
        tid = fx.thread_idx.x
        num_pid_n = div_up(N, TILE_N)
        # M is launch-uniform, but its dynamic ABI value may be classified as
        # divergent. Scalarize once: a divergent scale soffset would introduce
        # an EXEC waterfall around every A-scale DMA and split the pipeline.
        scale_m_rows = fx.Int32(
            rocdl.readfirstlane(T.i32, arith._to_raw(fx.Int32(div_up(M, 256) * 256)))
        )
        scale_n_rows = div_up(N, 256) * 256
        if const_expr(pid_swizzle):
            bid_x, bid_y = get_pids_950(fx.block_idx.x, M, fx.grid_dim.x, 8, 4)
        else:
            bid_x = fx.block_idx.x // num_pid_n
            bid_y = fx.block_idx.x % num_pid_n

        C_2d = fx.Tensor(
            fx.make_view(fx.get_iter(argC), fx.make_layout((M, N), (N, 1)))
        )
        # All operand DMA resources use byte bounds, including packed FP4.
        a_dma_rsrc = _buffer_resource(argA, num_records_bytes=fx.Int64(M) * K)
        b_dma_rsrc = _buffer_resource(
            argB,
            num_records_bytes=N * K // (2 if b_mxfp4 else 1),
        )
        C = fx.rocdl.make_buffer_tensor(C_2d, max_size=False)
        c_store_rsrc = _buffer_resource(argC, num_records_bytes=fx.Int64(M) * N * 2)

        # Swizzle only the LDS read layout; global DMA maps are tile-local.
        def apply_swizzles(layout, specs):
            for mask, base, shift in specs:
                swizzle = fx.static(fx.SwizzleType.get(mask, base, shift))
                layout = fx.make_composed_layout(swizzle, layout)
            return layout

        bC_tl = fx.flat_divide(C, (BLOCK_M, BLOCK_N))[
            None, None, bid_x * 2 + 0, bid_y * 2 + 0
        ]
        bC_tr = fx.flat_divide(C, (BLOCK_M, BLOCK_N))[
            None, None, bid_x * 2 + 0, bid_y * 2 + 1
        ]
        bC_bl = fx.flat_divide(C, (BLOCK_M, BLOCK_N))[
            None, None, bid_x * 2 + 1, bid_y * 2 + 0
        ]
        bC_br = fx.flat_divide(C, (BLOCK_M, BLOCK_N))[
            None, None, bid_x * 2 + 1, bid_y * 2 + 1
        ]

        # ---- tiled MMA: MFMA_Scale 16x16x128 f8f6f4 ----
        # 这个MMA atom更多用于获得 A， B相关的寄存器。
        # scale 相关的寄存器无法通过这个方式获得。
        mma_atom = fx.make_mma_atom(
            fx.rocdl.cdna4.MFMA_Scale(16, 16, 128, b_element_type, element_type)
        )
        mma_atom = fx.atom_set_value(mma_atom, "scale_a", fx.Int32(0))
        mma_atom = fx.atom_set_value(mma_atom, "scale_b", fx.Int32(0))
        # 真正用于做gemm的 mma atom
        if const_expr(with_scale or b_mxfp4):
            # Logical B occupies MFMA operand A and logical A occupies operand B.
            scale_atoms = {
                (n0, m0): fx.make_mma_atom(
                    fx.rocdl.cdna4.MFMA_Scale(
                        16,
                        16,
                        128,
                        b_element_type,
                        element_type,
                        opsel_a=n0,
                        opsel_b=m0,
                    )
                )
                for n0 in range_constexpr(4)
                for m0 in range_constexpr(4)
            }
            if const_expr(not with_scale):
                scale_atoms = {
                    key: fx.atom_set_value(
                        fx.atom_set_value(atom, "scale_a", fx.Int32(0)),
                        "scale_b",
                        fx.Int32(0),
                    )
                    for key, atom in scale_atoms.items()
                }
            # as spec described in the MFMA_Scale documentation, the k dimension is divided into 4 sub-tiles of size 32 each.
            k_perm = fx.make_layout(((16, 2), 4), ((1, 64), 16))
        else:
            # each lane column would hold 32 elements of the k dimension, which not follow the spec. Some  thing like A/B both
            # permute the K dimension but not affect final accumulation  result.
            k_perm = fx.make_layout((32, 4), (1, 32))
        tiled_mma = fx.make_tiled_mma(
            mma_atom, fx.make_layout((2, 2, 1), (1, 2, 0)), (None, None, k_perm)
        )
        thr_mma = tiled_mma.thr_slice(tid)

        # ---- copy atoms ----
        lds_copy_atom = fx.make_copy_atom(fx.UniversalCopy128b(), element_type)
        lds_copy_atom_b = fx.make_copy_atom(fx.UniversalCopy128b(), b_element_type)

        # ---- LDS 分配 ----
        lds = fx.SharedAllocator().allocate(LDS).peek()
        lane_id = tid % 64
        wave_id = tid // 64
        wave_id_uniform = fx.Int32(rocdl.readfirstlane(T.i32, arith._to_raw(wave_id)))

        scale_a_t_frag = None
        scale_a_b_frag = None
        scale_b_l_frag = None
        scale_b_r_frag = None
        if const_expr(with_scale):
            # Each wave owns 64 consecutive scale dwords. Replicating the 32x4
            # logical scale tile per consuming wave uses the existing 1 KB entry
            # while giving every lane a distinct 64-bank LDS address.
            scale_lds_copy_atom = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Int32)
            scale_rd_layout = fx.make_layout(256, 1)
            scale_tv = fx.make_layout((256, 1), (1, 1))
            scale_copy = fx.make_tiled_copy(
                scale_lds_copy_atom, scale_tv, fx.make_tile(256)
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
            scale_a_t_src = [scale_copy.partition_S(view) for view in scale_a_t_rd]
            scale_a_b_src = [scale_copy.partition_S(view) for view in scale_a_b_rd]
            scale_b_l_src = [scale_copy.partition_S(view) for view in scale_b_l_rd]
            scale_b_r_src = [scale_copy.partition_S(view) for view in scale_b_r_rd]
            scale_a_t_frag = fx.make_fragment_like(scale_a_t_src[0])
            scale_a_b_frag = fx.make_fragment_like(scale_a_b_src[0])
            scale_b_l_frag = fx.make_fragment_like(scale_b_l_src[0])
            scale_b_r_frag = fx.make_fragment_like(scale_b_r_src[0])

            scale_a_dma_rsrc = _buffer_resource(
                argScaleA,
                num_records_bytes=fx.Int64(scale_m_rows) * (K // 32),
            )
            scale_b_dma_rsrc = _buffer_resource(
                argScaleB,
                num_records_bytes=scale_n_rows * K // 32,
            )

            def make_scale_dma_ptr(ptr):
                return _lds_byte_ptr(ptr, wave_id_uniform * 64 * 4)

            scale_a_t_dma_ptrs = [
                make_scale_dma_ptr(ptr)
                for ptr in (lds.scale_a_t0.ptr, lds.scale_a_t1.ptr)
            ]
            scale_a_b_dma_ptrs = [
                make_scale_dma_ptr(ptr)
                for ptr in (lds.scale_a_b0.ptr, lds.scale_a_b1.ptr)
            ]
            scale_b_l_dma_ptrs = [
                make_scale_dma_ptr(ptr)
                for ptr in (lds.scale_b_l0.ptr, lds.scale_b_l1.ptr)
            ]
            scale_b_r_dma_ptrs = [
                make_scale_dma_ptr(ptr)
                for ptr in (lds.scale_b_r0.ptr, lds.scale_b_r1.ptr)
            ]

            def make_scale_voffset(row_tile, rows, is_a):
                wave_half = wave_id_uniform // 2 if is_a else wave_id_uniform % 2
                scale_row = lane_id % 16 + wave_half * 16
                scale_group = lane_id // 16
                return (
                    fx.Int32(scale_row) * 4
                    + fx.Int32(scale_group) * rows
                    + fx.Int32(row_tile) * 32 * 4
                )

            scale_a_t_voffset = make_scale_voffset(bid_x * 2, scale_m_rows, True)
            scale_a_b_voffset = make_scale_voffset(bid_x * 2 + 1, scale_m_rows, True)
            scale_b_l_voffset = make_scale_voffset(bid_y * 2, scale_n_rows, False)
            scale_b_r_voffset = make_scale_voffset(bid_y * 2 + 1, scale_n_rows, False)

            # Host layout packs four E8M0 groups per BK128. Only the K step
            # changes in the loop, in soffset; every lane copies one dword.
            def raw_scale_g2s(rsrc, kk, ptr, voffset, rows):
                rocdl.raw_ptr_buffer_load_lds(
                    rsrc,
                    ptr,
                    fx.Int32(4),
                    voffset,
                    fx.Int32(kk * rows * 4),
                    fx.Int32(0),
                    fx.Int32(0),
                )

        # ---- LDS read layouts (unchanged by the raw G2S address factories) ----
        if const_expr(lds_swizzle):
            _wr = fx.make_ordered_layout((BLOCK_M, BLOCK_K), (1, 0))
            _rd = apply_swizzles(_wr, swizzle_a_specs)
        else:
            # A8W8 与 Hybrid 的 MFMA K permutation 不同，分别需要 1040 B 和
            # 1056 B 的 8-row stride；两者的 16-row stride 均为 2112 B。
            _wr = fx.make_layout(
                ((8, 2, BLOCK_M // 16), BLOCK_K),
                ((BLOCK_K, a_group8, a_group16), 1),
            )
            _rd = fx.make_layout(
                ((2, BLOCK_M // 16, 8), (32, BLOCK_K // 32)),
                ((a_group8, a_group16, BLOCK_K), (1, 32)),
            )

        # B's LDS read layout: preshuffle uses the host shuffle's physical
        # ordering; otherwise retain the original padding/swizzle layout.
        _wr_b = _wr
        _rd_b = _rd
        if const_expr(b_lds_swizzle):
            _wr_b = fx.make_ordered_layout((BLOCK_N, BLOCK_K), (1, 0))
            _rd_b = apply_swizzles(_wr_b, swizzle_b_specs)
        if const_expr(b_mxfp4):
            if const_expr(not b_lds_swizzle):
                _wr_b = fx.make_layout(
                    ((16, BLOCK_N // 16), BLOCK_K),
                    ((BLOCK_K, b_group16), 1),
                )
                _rd_b = _wr_b
        if const_expr(preshuffle_b):
            _b_lds = fx.make_layout(
                ((16, BLOCK_N // 16), (16, BLOCK_K // 16)), ((16, 2048), (1, 256))
            )
            _wr_b = _b_lds
            _rd_b = _b_lds

        sA_t_rd = [fx.make_view(lds.a_t0.ptr, _rd), fx.make_view(lds.a_t1.ptr, _rd)]
        sA_b_rd = [fx.make_view(lds.a_b0.ptr, _rd), fx.make_view(lds.a_b1.ptr, _rd)]
        sB_l_rd = [fx.make_view(lds.b_l0.ptr, _rd_b), fx.make_view(lds.b_l1.ptr, _rd_b)]
        sB_r_rd = [fx.make_view(lds.b_r0.ptr, _rd_b), fx.make_view(lds.b_r1.ptr, _rd_b)]

        # The instruction adds lane_id * 16 to each wave-uniform LDS base.
        # Factories run once per kernel, never inside the K loop or raw_g2s.
        def fp8_copy_slots(copy_round, swizzled, specs):
            if const_expr(swizzled):
                mask, _, shift = specs[0]
                physical_slot = tid + copy_round * 256
                logical_slot = physical_slot ^ (
                    (physical_slot >> shift) & ((1 << mask) - 1)
                )
                row = logical_slot // 8
                col_byte = (logical_slot % 8) * 16
                lds_byte = (wave_id_uniform * 64 + copy_round * 256) * 16
            else:
                row = wave_id_uniform + (lane_id // 8) * 16 + copy_round * 4
                col_byte = (lane_id % 8) * 16
                lds_byte = (
                    (wave_id_uniform % 2) * a_group8
                    + (wave_id_uniform // 2) * a_group16
                    + copy_round * 4 * A_GROUP
                )
            return row, col_byte, lds_byte

        def make_fp8_voffsets(row_tile, block_rows, swizzled, specs):
            voffsets = []
            for copy_round in range_constexpr(4):
                row, col_byte, _ = fp8_copy_slots(copy_round, swizzled, specs)
                voffsets.append(
                    (fx.Int32(row_tile) * block_rows + fx.Int32(row)) * K
                    + fx.Int32(col_byte)
                )
            return voffsets

        def make_fp8_dma_ptrs(ptr, swizzled, specs):
            return [
                _lds_byte_ptr(ptr, fp8_copy_slots(r, swizzled, specs)[2])
                for r in range_constexpr(4)
            ]

        def b_copy_slots(copy_round):
            if const_expr(preshuffle_b):
                physical_slot = tid + copy_round * 256
                nb = physical_slot // 128
                in_nb = physical_slot % 128
                # Existing _subB: ((16, BLOCK_N//16), (16, BLOCK_K//16), K//128)
                # strides ((16, 16*K), (1, 256), 2048). A contiguous LDS slot
                # has ni=in_nb%16, k1=in_nb//16, so ni*16+k1*256=in_nb*16.
                # Only nb's stride changes from LDS 2048 to global 16*K.
                src_byte = nb * 16 * K + in_nb * 16
                lds_byte = (wave_id_uniform * 64 + copy_round * 256) * 16
            elif const_expr(b_mxfp4):
                if const_expr(b_lds_swizzle):
                    physical_slot = tid + copy_round * 256
                    logical_slot = physical_slot ^ ((physical_slot >> 3) & 1)
                    row = (logical_slot // 32) * 8 + logical_slot % 8
                    col_byte = ((logical_slot % 32) // 8) * 16
                    lds_byte = (wave_id_uniform * 64 + copy_round * 256) * 16
                else:
                    chunk = wave_id_uniform + copy_round * 4
                    row = chunk * 16 + lane_id // 4
                    col_byte = (lane_id % 4) * 16
                    lds_byte = chunk * (b_group16 // 2)
                src_byte = fx.Int32(row) * (K // 2) + fx.Int32(col_byte)
            else:
                row, col_byte, lds_byte = fp8_copy_slots(
                    copy_round, b_lds_swizzle, swizzle_b_specs
                )
                src_byte = fx.Int32(row) * K + fx.Int32(col_byte)
            return fx.Int32(src_byte), lds_byte

        b_copy_rounds = 2 if b_mxfp4 else 4
        b_k_stride = 16 * BLOCK_K if preshuffle_b else BLOCK_K // (2 if b_mxfp4 else 1)

        def make_b_voffsets(row_tile):
            tile_byte = fx.Int32(row_tile) * BLOCK_N * (K // (2 if b_mxfp4 else 1))
            return [
                tile_byte + b_copy_slots(r)[0] for r in range_constexpr(b_copy_rounds)
            ]

        def make_b_dma_ptrs(ptr):
            return [
                _lds_byte_ptr(ptr, b_copy_slots(r)[1])
                for r in range_constexpr(b_copy_rounds)
            ]

        a_t_voffsets = make_fp8_voffsets(
            bid_x * 2, BLOCK_M, lds_swizzle, swizzle_a_specs
        )
        a_b_voffsets = make_fp8_voffsets(
            bid_x * 2 + 1, BLOCK_M, lds_swizzle, swizzle_a_specs
        )
        b_l_voffsets = make_b_voffsets(bid_y * 2)
        b_r_voffsets = make_b_voffsets(bid_y * 2 + 1)
        a_t_dma_ptrs = [
            make_fp8_dma_ptrs(ptr, lds_swizzle, swizzle_a_specs)
            for ptr in (lds.a_t0.ptr, lds.a_t1.ptr)
        ]
        a_b_dma_ptrs = [
            make_fp8_dma_ptrs(ptr, lds_swizzle, swizzle_a_specs)
            for ptr in (lds.a_b0.ptr, lds.a_b1.ptr)
        ]
        b_l_dma_ptrs = [make_b_dma_ptrs(ptr) for ptr in (lds.b_l0.ptr, lds.b_l1.ptr)]
        b_r_dma_ptrs = [make_b_dma_ptrs(ptr) for ptr in (lds.b_r0.ptr, lds.b_r1.ptr)]

        def raw_g2s(rsrc, kk, ptrs, voffsets, k_stride):
            tile_soffset = fx.Int32(kk * k_stride)
            for copy_round in range_constexpr(len(voffsets)):
                rocdl.raw_ptr_buffer_load_lds(
                    rsrc,
                    ptrs[copy_round],
                    fx.Int32(16),
                    voffsets[copy_round],
                    tile_soffset,
                    fx.Int32(0),
                    fx.Int32(0),
                )

        # ---- LDS -> reg（对标 gemm_v9：A 走 B-operand，B 走 A-operand；均 padding rd）----
        # 每个 slice 只有一份寄存器 fragment（无寄存器双缓冲），双缓冲仅在 LDS 层（buf0/buf1）。
        copy_a = fx.make_tiled_copy_B(lds_copy_atom, tiled_mma).get_slice(tid)
        copy_b = fx.make_tiled_copy_A(lds_copy_atom_b, tiled_mma).get_slice(tid)
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
        if const_expr(b_mxfp4):
            frag_B_l = fx.make_rmem_tensor(16, fx.Int32)
            frag_B_r = fx.make_rmem_tensor(16, fx.Int32)
        else:
            frag_B_l = thr_mma.make_fragment_A(sB_l_rd[0])
            frag_B_r = thr_mma.make_fragment_A(sB_r_rd[0])
        dest_frag_A_t = copy_a.retile(frag_A_t)
        dest_frag_A_b = copy_a.retile(frag_A_b)
        dest_frag_B_l = None
        dest_frag_B_r = None
        if const_expr(not b_mxfp4):
            dest_frag_B_l = copy_b.retile(frag_B_l)
            dest_frag_B_r = copy_b.retile(frag_B_r)

        # copy A from LDS to reg. tiled API. very simple.
        def load_a(src_partition, dst_partition):
            fx.copy(lds_copy_atom, src_partition, dst_partition, pred=None)

        # copy A from LDS to reg. tiled API only used for non-mxfp4 case. mxfp4 case is handled by raw API function.
        # todo: mxfp4 padding方案存在bank conflict, swizzle可以work,是不是可以考虑使用 标准的 tiled copy API.
        def load_b(src, src_partition, dst, dst_partition):
            if const_expr(b_mxfp4):
                lane_id = tid % 64
                wave_n = (tid // 64) % 2
                values = []
                for n0 in range_constexpr(4):
                    row = (n0 * 2 + wave_n) * 16 + lane_id % 16
                    col_byte = (lane_id // 16) * 16
                    if const_expr(b_lds_swizzle):
                        lds_byte = (
                            (row // 8) * 512 + (row % 8) * 16 + (col_byte // 16) * 128
                        )
                        lds_byte = lds_byte ^ (((lds_byte >> 7) & 1) << 4)
                    else:
                        lds_byte = (
                            (row // 16) * (b_group16 // 2)
                            + (row % 16) * (BLOCK_K // 2)
                            + col_byte
                        )
                    ptr = fx.add_offset(
                        fx.recast_iter(fx.Uint8, fx.get_iter(src)),
                        fx.make_int_tuple(lds_byte),
                    )
                    packed = (
                        fx.make_view(ptr, fx.make_layout(16, 1))
                        .load()
                        .bitcast(fx.Int32)
                    )
                    for word in range_constexpr(4):
                        values.append(packed[word])
                dst.store(Vec.from_elements(values, fx.Int32))
            else:
                fx.copy(lds_copy_atom, src_partition, dst_partition, pred=None)

        c_layout_tile = fx.make_rmem_tensor(
            fx.make_ordered_layout((BLOCK_N, BLOCK_M), (1, 0)), fx.Float32
        )
        frag_C_tl = thr_mma.make_fragment_C(c_layout_tile)
        frag_C_tr = thr_mma.make_fragment_C(c_layout_tile)
        frag_C_bl = thr_mma.make_fragment_C(c_layout_tile)
        frag_C_br = thr_mma.make_fragment_C(c_layout_tile)

        if const_expr(with_scale or b_mxfp4):

            def do_gemm(c_frag, b_frag, a_frag, scale_a_frag, scale_b_frag):
                c_value = c_frag.load().ir_value()
                b_value = vector.bitcast(
                    T.vec(64 if b_mxfp4 else 128, T.i8), b_frag.load().ir_value()
                )
                a_value = vector.bitcast(T.vec(128, T.i8), a_frag.load().ir_value())
                if const_expr(with_scale):
                    scale_a = Vec(scale_a_frag.load())[0]
                    scale_b = Vec(scale_b_frag.load())[0]
                for n0 in range_constexpr(4):
                    for m0 in range_constexpr(4):
                        c_offset = (m0 * 4 + n0) * 4
                        c_sub = vector.extract_strided_slice(
                            T.vec(4, T.f32),
                            c_value,
                            offsets=[c_offset],
                            sizes=[4],
                            strides=[1],
                        )
                        b_sub_bytes = 16 if b_mxfp4 else 32
                        b_sub = vector.extract_strided_slice(
                            T.vec(b_sub_bytes, T.i8),
                            b_value,
                            offsets=[n0 * b_sub_bytes],
                            sizes=[b_sub_bytes],
                            strides=[1],
                        )
                        if const_expr(b_mxfp4):
                            b_sub = vector.bitcast(T.vec(4, T.i32), b_sub)
                        a_sub = vector.extract_strided_slice(
                            T.vec(32, T.i8),
                            a_value,
                            offsets=[m0 * 32],
                            sizes=[32],
                            strides=[1],
                        )
                        if const_expr(with_scale):
                            scaled_atom = fx.atom_set_value(
                                scale_atoms[(n0, m0)], "scale_a", scale_b
                            )
                            scaled_atom = fx.atom_set_value(
                                scaled_atom, "scale_b", scale_a
                            )
                        else:
                            scaled_atom = scale_atoms[(n0, m0)]
                        c_sub = _fly.mma_atom_call_ssa(
                            [T.vec(4, T.f32)], scaled_atom, b_sub, a_sub, c_sub
                        )
                        c_value = vector.insert_strided_slice(
                            c_sub, c_value, [c_offset], [1]
                        )
                c_frag.store(c_value)

        else:

            def do_gemm(c_frag, b_frag, a_frag, scale_a_frag=None, scale_b_frag=None):
                fx.gemm(mma_atom, c_frag, b_frag, a_frag, c_frag)

            def do_gemm_mainloop(
                c_frag, b_frag, a_frag, scale_a_frag=None, scale_b_frag=None
            ):
                c_value = c_frag.load().ir_value()
                b_value = vector.bitcast(T.vec(128, T.i8), b_frag.load().ir_value())
                a_value = vector.bitcast(T.vec(128, T.i8), a_frag.load().ir_value())
                c_results = []
                for n0 in range_constexpr(4):
                    for m0 in range_constexpr(4):
                        c_offset = (m0 * 4 + n0) * 4
                        c_sub = vector.extract_strided_slice(
                            T.vec(4, T.f32),
                            c_value,
                            offsets=[c_offset],
                            sizes=[4],
                            strides=[1],
                        )
                        b_sub = vector.extract_strided_slice(
                            T.vec(32, T.i8),
                            b_value,
                            offsets=[n0 * 32],
                            sizes=[32],
                            strides=[1],
                        )
                        a_sub = vector.extract_strided_slice(
                            T.vec(32, T.i8),
                            a_value,
                            offsets=[m0 * 32],
                            sizes=[32],
                            strides=[1],
                        )
                        c_results.append(
                            _fly.mma_atom_call_ssa(
                                [T.vec(4, T.f32)], mma_atom, b_sub, a_sub, c_sub
                            )
                        )
                c_elements = []
                for m0 in range_constexpr(4):
                    for n0 in range_constexpr(4):
                        c_result = Vec(c_results[n0 * 4 + m0])
                        for elem in range_constexpr(4):
                            c_elements.append(c_result[elem])
                c_frag.store(Vec.from_elements(c_elements, fx.Float32))

        num_tiles = K // BLOCK_K
        assert num_tiles >= 4

        # Small prefetch helpers share the same hoisted addresses in prologue
        # and all eight phases. Future scales precede their data, as in MoE.
        def prefetch_b_l(kk, buf):
            if const_expr(with_scale):
                raw_scale_g2s(
                    scale_b_dma_rsrc,
                    kk,
                    scale_b_l_dma_ptrs[buf],
                    scale_b_l_voffset,
                    scale_n_rows,
                )
            raw_g2s(b_dma_rsrc, kk, b_l_dma_ptrs[buf], b_l_voffsets, b_k_stride)

        def prefetch_a_t(kk, buf):
            if const_expr(with_scale):
                raw_scale_g2s(
                    scale_a_dma_rsrc,
                    kk,
                    scale_a_t_dma_ptrs[buf],
                    scale_a_t_voffset,
                    scale_m_rows,
                )
            raw_g2s(a_dma_rsrc, kk, a_t_dma_ptrs[buf], a_t_voffsets, BLOCK_K)

        def prefetch_a_b(kk, buf):
            if const_expr(with_scale):
                raw_scale_g2s(
                    scale_a_dma_rsrc,
                    kk,
                    scale_a_b_dma_ptrs[buf],
                    scale_a_b_voffset,
                    scale_m_rows,
                )
            raw_g2s(a_dma_rsrc, kk, a_b_dma_ptrs[buf], a_b_voffsets, BLOCK_K)

        def prefetch_b_r(kk, buf):
            if const_expr(with_scale):
                raw_scale_g2s(
                    scale_b_dma_rsrc,
                    kk,
                    scale_b_r_dma_ptrs[buf],
                    scale_b_r_voffset,
                    scale_n_rows,
                )
            raw_g2s(b_dma_rsrc, kk, b_r_dma_ptrs[buf], b_r_voffsets, b_k_stride)

        # Preserve the two-tile prologue's B_l, A_t, A_b, B_r batch order.
        def do_g2s(kk, buf):
            ki = fx.Int32(kk)
            prefetch_b_l(ki, buf)
            rocdl.sched_barrier(0)
            prefetch_a_t(ki, buf)
            rocdl.sched_barrier(0)
            prefetch_a_b(ki, buf)
            rocdl.sched_barrier(0)
            prefetch_b_r(ki, buf)
            rocdl.sched_barrier(0)

        # A uses four full-wave VMEM instructions per operand. MXFP4 B now uses
        # two after replacing four subgroup loads with one full-wave load per
        # 2048-element block; each scale operand contributes one more VMEM.
        a_vmem = (BLOCK_M * BLOCK_K * element_type.width // 8) // (256 * 16)
        b_vmem = (BLOCK_N * BLOCK_K * b_element_type.width // 8) // (256 * 16)
        a_phase_vmem = a_vmem + int(with_scale)
        b_phase_vmem = b_vmem + int(with_scale)
        # (wait_ab, wait_ba): scaled FP4 (19,21), scaled FP8 (25,25),
        # unscaled FP8 (20,20), unscaled FP4 (14,16). Scales are counted once.
        wait_ab = 2 * a_phase_vmem + 3 * b_phase_vmem
        wait_ba = 3 * a_phase_vmem + 2 * b_phase_vmem

        do_g2s(0, 0)
        do_g2s(1, 1)
        waitvmcnt_barrier(3 * (a_phase_vmem + b_phase_vmem))
        load_b(sB_l_rd[0], s2r_src0_B_l, frag_B_l, dest_frag_B_l)
        load_a(s2r_src0_A_t, dest_frag_A_t)
        if const_expr(with_scale):
            fx.copy(scale_lds_copy_atom, scale_b_l_src[0], scale_b_l_frag)
            fx.copy(scale_lds_copy_atom, scale_a_t_src[0], scale_a_t_frag)
        rocdl.sched_barrier(0)

        frag_C_tl.fill(0)
        frag_C_tr.fill(0)
        frag_C_bl.fill(0)
        frag_C_br.fill(0)
        rocdl.sched_barrier(0)
        acc_init = [
            frag_C_tl.load(),
            frag_C_tr.load(),
            frag_C_bl.load(),
            frag_C_br.load(),
        ]

        # 每 region 的 ds_read / vmem 计数（对标 gemm_4wave_950）：A operand 与 B operand
        # 的 fragment 大小不同，ds_read_b128 数量也不同；读 A 的 region 用 a_dsrd，读 B 的用
        # b_dsrd。之前对所有 region 统一传 dsrd=8，导致读 B 的 region 未被完整调度、访存交织
        # 退化并增加 wait cycles。
        a_dsrd = frag_A_t.load().numel * element_type.width // 8 // 16
        b_dsrd = (
            4 if b_mxfp4 else frag_B_l.load().numel * b_element_type.width // 8 // 16
        )

        a_phase_dsrd = a_dsrd + int(with_scale)
        b_phase_dsrd = b_dsrd + int(with_scale)

        # 每个 region：wait/fence -> independent next s2r -> GEMM -> future G2S -> schedule，
        # 用 s2r_src0_*/s2r_src1_* 在 LDS buf0/buf1 之间 ping-pong；每个 slice 顺序与 gemm_v9 一致。
        # k-tile 内 4 个象限的顺序固定为：tl(A_t·B_l) -> bl(A_b·B_l) -> tr(A_t·B_r) -> br(A_b·B_r)。
        # 运行时循环（range + init/yield 累加器透传），不做常量展开。

        for kidx, states in range(0, num_tiles - 2, 2, init=acc_init):
            frag_C_tl.store(states[0])
            frag_C_tr.store(states[1])
            frag_C_bl.store(states[2])
            frag_C_br.store(states[3])
            kiter = fx.Int32(kidx)

            # ---- k-tile = buf0 ----
            # ----------------------buf0:part0----------------------
            waitvmcnt_barrier(wait_ab)
            load_a(s2r_src0_A_b, dest_frag_A_b)
            if const_expr(with_scale):
                fx.copy(scale_lds_copy_atom, scale_a_b_src[0], scale_a_b_frag)
            if const_expr(not with_scale and not b_mxfp4):
                do_gemm_mainloop(frag_C_tl, frag_B_l, frag_A_t)
            else:
                do_gemm(frag_C_tl, frag_B_l, frag_A_t, scale_a_t_frag, scale_b_l_frag)
            prefetch_b_l(kiter + 2, 0)
            _schedule_compute(0, a_phase_dsrd, b_phase_vmem)
            rocdl.sched_barrier(0)

            # ----------------------buf0:part1----------------------
            waitvmcnt_barrier(wait_ab)
            load_b(sB_r_rd[0], s2r_src0_B_r, frag_B_r, dest_frag_B_r)
            if const_expr(with_scale):
                fx.copy(scale_lds_copy_atom, scale_b_r_src[0], scale_b_r_frag)
            do_gemm(frag_C_bl, frag_B_l, frag_A_b, scale_a_b_frag, scale_b_l_frag)
            prefetch_a_t(kiter + 2, 0)
            _schedule_compute(1, b_phase_dsrd, a_phase_vmem)
            rocdl.sched_barrier(0)

            # ----------------------buf0:part2----------------------
            waitvmcnt_barrier(wait_ba)
            load_b(sB_l_rd[1], s2r_src1_B_l, frag_B_l, dest_frag_B_l)
            if const_expr(with_scale):
                fx.copy(scale_lds_copy_atom, scale_b_l_src[1], scale_b_l_frag)
            do_gemm(frag_C_tr, frag_B_r, frag_A_t, scale_a_t_frag, scale_b_r_frag)
            prefetch_a_b(kiter + 2, 0)
            _schedule_compute(2, b_phase_dsrd, a_phase_vmem)
            rocdl.sched_barrier(0)

            # ----------------------buf0:part3----------------------
            waitvmcnt_barrier(wait_ba)
            load_a(s2r_src1_A_t, dest_frag_A_t)
            if const_expr(with_scale):
                fx.copy(scale_lds_copy_atom, scale_a_t_src[1], scale_a_t_frag)
            do_gemm(frag_C_br, frag_B_r, frag_A_b, scale_a_b_frag, scale_b_r_frag)
            prefetch_b_r(kiter + 2, 0)
            _schedule_compute(3, a_phase_dsrd, b_phase_vmem)
            rocdl.sched_barrier(0)

            # ---- k-tile = buf1：4 象限 ----
            # ----------------------buf1:part0----------------------
            waitvmcnt_barrier(wait_ab)
            load_a(s2r_src1_A_b, dest_frag_A_b)
            if const_expr(with_scale):
                fx.copy(scale_lds_copy_atom, scale_a_b_src[1], scale_a_b_frag)
            do_gemm(frag_C_tl, frag_B_l, frag_A_t, scale_a_t_frag, scale_b_l_frag)
            prefetch_b_l(kiter + 3, 1)
            _schedule_compute(4, a_phase_dsrd, b_phase_vmem)
            rocdl.sched_barrier(0)

            # ----------------------buf1:part1----------------------
            waitvmcnt_barrier(wait_ab)
            load_b(sB_r_rd[1], s2r_src1_B_r, frag_B_r, dest_frag_B_r)
            if const_expr(with_scale):
                fx.copy(scale_lds_copy_atom, scale_b_r_src[1], scale_b_r_frag)
            do_gemm(frag_C_bl, frag_B_l, frag_A_b, scale_a_b_frag, scale_b_l_frag)
            prefetch_a_t(kiter + 3, 1)
            _schedule_compute(5, b_phase_dsrd, a_phase_vmem)
            rocdl.sched_barrier(0)

            # ----------------------buf1:part2----------------------
            waitvmcnt_barrier(wait_ba)
            load_b(sB_l_rd[0], s2r_src0_B_l, frag_B_l, dest_frag_B_l)
            if const_expr(with_scale):
                fx.copy(scale_lds_copy_atom, scale_b_l_src[0], scale_b_l_frag)
            do_gemm(frag_C_tr, frag_B_r, frag_A_t, scale_a_t_frag, scale_b_r_frag)
            prefetch_a_b(kiter + 3, 1)
            _schedule_compute(6, b_phase_dsrd, a_phase_vmem)
            rocdl.sched_barrier(0)

            # ----------------------buf1:part3----------------------
            waitvmcnt_barrier(wait_ba)
            load_a(s2r_src0_A_t, dest_frag_A_t)
            if const_expr(with_scale):
                fx.copy(scale_lds_copy_atom, scale_a_t_src[0], scale_a_t_frag)
            do_gemm(frag_C_br, frag_B_r, frag_A_b, scale_a_b_frag, scale_b_r_frag)
            prefetch_b_r(kiter + 3, 1)
            _schedule_compute(7, a_phase_dsrd, b_phase_vmem)
            rocdl.sched_barrier(0)

            results = yield [
                frag_C_tl.load(),
                frag_C_tr.load(),
                frag_C_bl.load(),
                frag_C_br.load(),
            ]
        frag_C_tl.store(results[0])
        frag_C_tr.store(results[1])
        frag_C_bl.store(results[2])
        frag_C_br.store(results[3])

        # ---- epilogue：最后 2 个 k-tile（buf0 / buf1），无 g2s，只做 s2r + gemm ----
        # Six hardware barriers total; phases 6/7 have no next LDS operand.
        # As in the mainloop, next reads are disjoint from the current GEMM.
        # buf0 的 4 象限
        waitvmcnt_barrier(wait_ab)
        load_a(s2r_src0_A_b, dest_frag_A_b)
        if const_expr(with_scale):
            fx.copy(scale_lds_copy_atom, scale_a_b_src[0], scale_a_b_frag)
        do_gemm(frag_C_tl, frag_B_l, frag_A_t, scale_a_t_frag, scale_b_l_frag)
        _schedule_compute(0, a_phase_dsrd, 0)
        rocdl.sched_barrier(0)

        waitvmcnt_barrier(2 * (a_phase_vmem + b_phase_vmem))
        load_b(sB_r_rd[0], s2r_src0_B_r, frag_B_r, dest_frag_B_r)
        if const_expr(with_scale):
            fx.copy(scale_lds_copy_atom, scale_b_r_src[0], scale_b_r_frag)
        do_gemm(frag_C_bl, frag_B_l, frag_A_b, scale_a_b_frag, scale_b_l_frag)
        _schedule_compute(1, b_phase_dsrd, 0)
        rocdl.sched_barrier(0)

        waitvmcnt_barrier(2 * a_phase_vmem + b_phase_vmem)
        load_b(sB_l_rd[1], s2r_src1_B_l, frag_B_l, dest_frag_B_l)
        if const_expr(with_scale):
            fx.copy(scale_lds_copy_atom, scale_b_l_src[1], scale_b_l_frag)
        do_gemm(frag_C_tr, frag_B_r, frag_A_t, scale_a_t_frag, scale_b_r_frag)
        _schedule_compute(2, b_phase_dsrd, 0)
        rocdl.sched_barrier(0)

        waitvmcnt_barrier(a_phase_vmem + b_phase_vmem)
        load_a(s2r_src1_A_t, dest_frag_A_t)
        if const_expr(with_scale):
            fx.copy(scale_lds_copy_atom, scale_a_t_src[1], scale_a_t_frag)
        do_gemm(frag_C_br, frag_B_r, frag_A_b, scale_a_b_frag, scale_b_r_frag)
        _schedule_compute(3, a_phase_dsrd, 0)
        rocdl.sched_barrier(0)
        N_tail = N % TILE_N != 0
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
                            + col_repeat * 32
                            + lane_group % 2 * 32
                            + wave_n * 16
                            + lane_group // 2 * 8
                        )
                        byte_offset = fx.Int32((row * N + col) * 2)
                        if const_expr(N_tail):
                            masked_offset = (col < N).select(
                                byte_offset, fx.Int32(0x7FFFFFFF)
                            )
                            rocdl.raw_ptr_buffer_store(
                                packed.ir_value(),
                                c_store_rsrc,
                                masked_offset.ir_value(),
                                fx.Int32(0).ir_value(),
                                aux=ir.IntegerAttr.get(T.i32, 0),
                            )
                        else:
                            rocdl.raw_ptr_buffer_store(
                                packed.ir_value(),
                                c_store_rsrc,
                                byte_offset.ir_value(),
                                fx.Int32(0).ir_value(),
                                aux=ir.IntegerAttr.get(T.i32, 0),
                            )

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
                store_src = store_thr.retile(c_bf16)
                store_dst = store_thr.partition_D(bC)
                if const_expr(N_tail):
                    col_coords = fx.Tensor(
                        fx.make_view(
                            fx.make_int_tuple(
                                bid_y * TILE_N + quadrant_n * (TILE_N // 2)
                            ),
                            fx.make_layout((BLOCK_M, BLOCK_N), (0, 1)),
                        )
                    )
                    store_cols = store_thr.partition_D(col_coords)
                    store_cols_per_atom = store_cols[0, None, None]
                    store_pred = fx.make_fragment_like(
                        store_cols_per_atom, dtype=fx.Boolean
                    )
                    store_pred.store(store_cols_per_atom.load() < N)
                    fx.copy(store_atom_bf16, store_src, store_dst, pred=store_pred)
                else:
                    fx.copy(store_atom_bf16, store_src, store_dst)

        # buf1 的 4 象限。store_overlap 时把每象限的 store 与后一象限的 MFMA 交织，
        # 用 MFMA 计算掩盖 buffer_store 的写延迟（对标 bf16 v9 scheduler_store_overlap）；
        # 否则先算完 4 象限，再统一 store（不交织，用于对照）。
        waitvmcnt_barrier(b_phase_vmem)
        load_a(s2r_src1_A_b, dest_frag_A_b)
        if const_expr(with_scale):
            fx.copy(scale_lds_copy_atom, scale_a_b_src[1], scale_a_b_frag)
        do_gemm(frag_C_tl, frag_B_l, frag_A_t, scale_a_t_frag, scale_b_l_frag)
        _schedule_compute(4, a_phase_dsrd, 0)
        rocdl.sched_barrier(0)

        waitvmcnt_barrier(0)
        load_b(sB_r_rd[1], s2r_src1_B_r, frag_B_r, dest_frag_B_r)
        if const_expr(with_scale):
            fx.copy(scale_lds_copy_atom, scale_b_r_src[1], scale_b_r_frag)
        do_gemm(frag_C_bl, frag_B_l, frag_A_b, scale_a_b_frag, scale_b_l_frag)
        if const_expr(store_overlap):
            # bl 的 MFMA 与 tl 的 store 互相掩盖
            store_quadrant(frag_C_tl, bC_tl, 0, 0)
            scheduler_store_overlap(5, b_phase_dsrd)
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
            _schedule_compute(5, b_phase_dsrd, 0)
            rocdl.sched_barrier(0)

            do_gemm(frag_C_tr, frag_B_r, frag_A_t, scale_a_t_frag, scale_b_r_frag)
            _schedule_compute(6, 0, 0)
            rocdl.sched_barrier(0)
            do_gemm(frag_C_br, frag_B_r, frag_A_b, scale_a_b_frag, scale_b_r_frag)
            _schedule_compute(7, 0, 0)
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
        M: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        # 累加器钉到 AGPR（force-agpr）+ mfma-vgpr-form=False：避免 C 累加器 VGPR/AGPR 混放导致的
        # v_accvgpr 拷贝与 VGPR 压力（对标 test_gemm_v9.py）。
        value_attrs = {
            "rocdl.waves_per_eu": 1,
            "passthrough": [["amdgpu-agpr-alloc", "256,256"]],
        }
        gemm_kernel(A, B, ScaleA, ScaleB, C, M, value_attrs=value_attrs).launch(
            grid=(div_up(M, TILE_M) * div_up(N, TILE_N), 1, 1),
            block=(256, 1, 1),
            stream=stream,
        )

    launch_gemm.compile_hints["llvm_options"] = {"amdgpu-mfma-vgpr-form": False}
    return launch_gemm


compile_gemm_fp8.cache_clear = _compile_gemm_fp8_cached.cache_clear
compile_gemm_fp8.cache_info = _compile_gemm_fp8_cached.cache_info