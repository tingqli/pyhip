import math
import os
os.environ.setdefault("FLYDSL_RUNTIME_ENABLE_CACHE", "0")

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.runtime.device import get_rocm_arch  # noqa: E402
from flydsl.compiler.ast_rewriter import ASTRewriter
from aiter.ops.shuffle import shuffle_weight

import pyhip

import pyhip.contrib.flydsl as fxu

# fxu.enable_dump_ir(True)

_, stream = pyhip.set_device()

# copy from https://github.com/ROCm/gfx9-gluon-tutorials/blob/main/kernels/gemm/a16w16/v8_beyond_hotloop/matmul_kernel.py
def _get_pids(
    pid,
    M,
    N,
    BM,
    BN,
    GRID_MN,
    NUM_XCDS,
    GROUP_SIZE_M,
):
    num_pid_m = (M + BM - 1) // BM
    num_pid_n = (N + BN - 1) // BN

    if fx.const_expr(NUM_XCDS != 1):
        ## pid remapping on xcds
        # Number of pids per XCD in the new arrangement
        pids_per_xcd = (GRID_MN + NUM_XCDS - 1) // NUM_XCDS
        # When GRID_MN cannot divide NUM_XCDS, some xcds will have
        # pids_per_xcd pids, the other will have pids_per_xcd - 1 pids.
        # We calculate the number of xcds that have pids_per_xcd pids as
        # tall_xcds
        tall_xcds = GRID_MN % NUM_XCDS
        # TODO
        # tall_xcds = NUM_XCDS if tall_xcds == 0 else tall_xcds
        tall_xcds = (tall_xcds == 0).select(NUM_XCDS, tall_xcds)
        # Compute current XCD and local pid within the XCD
        xcd = pid % NUM_XCDS
        local_pid = pid // NUM_XCDS
        # Calculate new pid based on the new grouping
        # Note that we need to consider the following two cases:
        # 1. the current pid is on a tall xcd
        # 2. the current pid is on a short xcd
        if xcd < tall_xcds:
            pid = xcd * pids_per_xcd + local_pid
        else:
            pid = tall_xcds * pids_per_xcd + (xcd - tall_xcds) * (pids_per_xcd - 1) + local_pid

    if fx.const_expr(GROUP_SIZE_M == 1):
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
    else:
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        # TODO
        #group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        group_size_m = (num_pid_m - first_pid_m < GROUP_SIZE_M).select(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

    return pid_m, pid_n

"""
复制了LuoCheng test_gemm.py 里面的实现
参考 
 https://github.com/ROCm/gfx950-gluon-tutorials/tree/main/kernels/gemm/a16w16/v7_sliceN
 https://github.com/ROCm/gfx950-gluon-tutorials/tree/main/kernels/gemm/a16w16/v8_sliceMN

ping-pong buffer LDS, split long MN dimension:
"""
def compile_v4(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, b_preshuffle):
    num_threads = 256

    TILE_M = BLOCK_M
    TILE_N = BLOCK_N
    TILE_K = BLOCK_K

    def _encode_waitcnt(vmcnt=63, expcnt=7, lgkmcnt=63):
        """Encode s_waitcnt bitfield for CDNA3 (gfx94x)."""
        vm_lo = vmcnt & 0xF
        vm_hi = (vmcnt >> 4) & 0x3
        return vm_lo | (expcnt << 4) | (lgkmcnt << 8) | (vm_hi << 14)


    get_pids = ASTRewriter.transform(_get_pids)

    @flyc.kernel
    def gemm_kernel_v4(A: fx.Tensor,B: fx.Tensor,C: fx.Tensor,):
        gpu_arch = get_rocm_arch()
        is_gfx942 = str(gpu_arch).startswith("gfx942")

        @fx.struct
        class SharedStorage:
            at_lds: fx.Array[A.dtype, BLOCK_M//2*BLOCK_K, 16]
            ab_lds: fx.Array[A.dtype, BLOCK_M//2*BLOCK_K, 16]
            bl_lds: fx.Array[B.dtype, BLOCK_N//2*BLOCK_K, 16]
            br_lds: fx.Array[B.dtype, BLOCK_N//2*BLOCK_K, 16]

        tid = fx.thread_idx.x
        bid_m, bid_n = fx.block_idx.x, fx.block_idx.y
        # 308 has 4 xcds
        #bid_m, bid_n = get_pids(fx.block_idx.x, M, N, TILE_M, TILE_N, fx.grid_dim.x, 4, 4)

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        swz = fx.SwizzleType.get(3, 3, 3)
        at_lds = fx.make_view(lds.at_lds.ptr, fx.make_composed_layout(fx.static(swz), fx.make_ordered_layout((BLOCK_M//2, BLOCK_K), (1, 0))))
        ab_lds = fx.make_view(lds.ab_lds.ptr, fx.make_composed_layout(fx.static(swz), fx.make_ordered_layout((BLOCK_M//2, BLOCK_K), (1, 0))))
        bl_lds = fx.make_view(lds.bl_lds.ptr, fx.make_composed_layout(fx.static(swz), fx.make_ordered_layout((BLOCK_N//2, BLOCK_K), (1, 0))))
        br_lds = fx.make_view(lds.br_lds.ptr, fx.make_composed_layout(fx.static(swz), fx.make_ordered_layout((BLOCK_N//2, BLOCK_K), (1, 0))))

        a_tensor = fx.rocdl.make_buffer_tensor(A, False)
        b_tensor = fx.rocdl.make_buffer_tensor(B, False)
        c_tensor = fx.rocdl.make_buffer_tensor(C, False)

        bA = fx.slice(a_tensor, (None, None, bid_m, None))   # BLOCK_M, BLOCK_K, (idx_block_m), num_blocks_k
        bB = fx.slice(b_tensor, (None, None, bid_n, None))   # BLOCK_N, BLOCK_K, (idx_block_n), num_blocks_k
        bC = fx.slice(c_tensor, (None, None, bid_m, bid_n))  # BLOCK_M, BLOCK_N, (idx_block_m), (idx_block_n)    

        bA = fx.logical_divide(bA, (BLOCK_M//2, None, None))  # ((BLOCK_M//2, 2), BLOCK_K, num_blocks_k)
        bB = fx.logical_divide(bB, (BLOCK_N//2, None, None))  # ((BLOCK_N//2, 2), BLOCK_K, num_blocks_k)
        bC = fx.flat_divide(bC, (BLOCK_M//2, BLOCK_N//2))  # (BLOCK_M//2, BLOCK_N//2, 2, 2)

        at_tile = bA[(None, 0), None, None]  # shape: [BLOCK_M//2, BLOCK_K, num_blocks_k]
        ab_tile = bA[(None, 1), None, None]  # shape: [BLOCK_M//2, BLOCK_K, num_blocks_k]

        bl_tile = bB[(None, 0), None, None]  # shape: [BLOCK_N//2, BLOCK_K, num_blocks_k]
        br_tile = bB[(None, 1), None, None]  # shape: [BLOCK_N//2, BLOCK_K, num_blocks_k]

        c_tl_tile = bC[None, None, 0, 0]  # shape: [BLOCK_M//2, BLOCK_N//2]
        c_tr_tile = bC[None, None, 0, 1]  # shape: [BLOCK_M//2, BLOCK_N//2]
        c_bl_tile = bC[None, None, 1, 0]  # shape: [BLOCK_M//2, BLOCK_N//2]
        c_br_tile = bC[None, None, 1, 1]  # shape: [BLOCK_M//2, BLOCK_N//2]

        # memory->lds layout
        buf_cp_atom_r = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), A.dtype)   # buffer_load/store_dwordx4
        # mi308 32bit写入延迟为8cycle，可以被16x16mfma的16cycle隐藏；128bit写入延迟为20cycle
        uni_cp_atom_r = fx.make_copy_atom(fx.UniversalCopy128b(), A.dtype) # ds_read/write_b128
        uni_cp_atom_w = fx.make_copy_atom(fx.UniversalCopy32b(), A.dtype)  # ds_read/write_b32
        VECT_WIDTH = 128//A.dtype.width
        tv_layout, tv_tilemn = fxu.make_mem_coalescing_2d_tv_layout(
                                            num_threads, VECT_WIDTH, BLOCK_K)
        at_cp_frag = fxu.Fragment.from_tvlayout(A.dtype, BLOCK_M//2, BLOCK_K, tv_layout, tv_tilemn, (buf_cp_atom_r, uni_cp_atom_w))
        ab_cp_frag = at_cp_frag.selfclone()
        bl_cp_frag = fxu.Fragment.from_tvlayout(B.dtype, BLOCK_N//2, BLOCK_K, tv_layout, tv_tilemn, (buf_cp_atom_r, uni_cp_atom_w))
        br_cp_frag = bl_cp_frag.selfclone()

        num_blocks_k = fx.get_scalar(fx.size(bA.layout[2]))
        print("num_blocks_k:", num_blocks_k)

        """
        为了使用 DW4/128b buffer_load/store, 需要保证 mma-atom 在K维度交织起来
        使得每个线程 FrgV 至少填满 128bit
        """ 
        mfma_K = {fx.Float32:4, fx.BFloat16:16, fx.Float16:16}[A.dtype]
        mfma_thrK = 4 # for MFMA-AB layout, always 4 threads along K dimension
        mfma_FrgV = mfma_K // mfma_thrK # per-thread value count
        mfma_FrgV_bits = mfma_FrgV * A.dtype.width
        mfma_cntK = 128 // mfma_FrgV_bits
        mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, mfma_K, A.dtype))
        thr_layout_mnk = fx.make_layout((2, 2, 1), (1, 2, 0))
        # k_perm = fx.make_layout((4, 4, 2), (1, 8, 4))
        k_perm = fx.make_layout((mfma_FrgV, mfma_thrK, mfma_cntK), (1, mfma_cntK*mfma_FrgV, mfma_FrgV))

        permutation_mnk = (None, None, k_perm)
        tiled_mma = fx.make_tiled_mma(mma_atom, thr_layout_mnk, permutation_mnk)

        """
        这里有一点 tricky, a矩阵被当作 TileMMA 的B矩阵，b矩阵被当作 TileMMA 的A矩阵
        因为调用 fx.gemm 的时候我们也会把 a_frag 当作 B 矩阵，b_frag 当作 A 矩阵传入，
        因此得到的 c_frag 将会是原来结果，按照 (BM//2, BN//2) 块单位的转置，此时我们
        进一步转置 c mem tensor 的layout，就能使得本来分布在 M 方向非物理连续的 4 个值
        变为 N 方向上物理连续的 4 个值，从而可以被 buffer_store_dwordx4 一次性写入
        """
        at_frag = fxu.Fragment.from_tiledmma(tiled_mma, BLOCK_M//2, BLOCK_K, "B", uni_cp_atom_r)
        ab_frag = at_frag.selfclone()
        bl_frag = fxu.Fragment.from_tiledmma(tiled_mma, BLOCK_N//2, BLOCK_K, "A", uni_cp_atom_r)
        br_frag = bl_frag.selfclone()

        buf_cp_atom_w = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), C.dtype)
        c_tl_frag = fxu.Fragment.from_tiledmma(tiled_mma, BLOCK_N//2, BLOCK_M//2, "C", [buf_cp_atom_w])
        c_tr_frag = c_tl_frag.selfclone()
        c_bl_frag = c_tl_frag.selfclone()
        c_br_frag = c_tl_frag.selfclone()

        c_tl_frag.fill(0)
        c_tr_frag.fill(0)
        c_bl_frag.fill(0)
        c_br_frag.fill(0)

        """ 存出转置的 Ct 时使用转置的 c mem layout, 实际上mem中得到的仍然是 C 而非 Ct """
        trans_layout = fx.make_ordered_layout((BLOCK_N//2, BLOCK_M//2), (1, 0))
        c_tl_tile = fx.composition(c_tl_tile, trans_layout)
        c_tr_tile = fx.composition(c_tr_tile, trans_layout)
        c_bl_tile = fx.composition(c_bl_tile, trans_layout)
        c_br_tile = fx.composition(c_br_tile, trans_layout)

        at_mem_tensor_thr = at_cp_frag.partition_S(at_tile, buf_cp_atom_r)
        ab_mem_tensor_thr = ab_cp_frag.partition_S(ab_tile, buf_cp_atom_r)
        bl_mem_tensor_thr = bl_cp_frag.partition_S(bl_tile, buf_cp_atom_r)
        br_mem_tensor_thr = br_cp_frag.partition_S(br_tile, buf_cp_atom_r)

        at_lds_tensor_thr_w = at_cp_frag.partition_D(at_lds, uni_cp_atom_w)
        ab_lds_tensor_thr_w = ab_cp_frag.partition_D(ab_lds, uni_cp_atom_w)
        bl_lds_tensor_thr_w = bl_cp_frag.partition_D(bl_lds, uni_cp_atom_w)
        br_lds_tensor_thr_w = br_cp_frag.partition_D(br_lds, uni_cp_atom_w)

        at_lds_tensor_thr_r = at_frag.partition_S(at_lds)
        ab_lds_tensor_thr_r = ab_frag.partition_S(ab_lds)
        bl_lds_tensor_thr_r = bl_frag.partition_S(bl_lds)
        br_lds_tensor_thr_r = br_frag.partition_S(br_lds)

        # prefetch
        # gr0: all
        bl_cp_frag.copy_from(bl_mem_tensor_thr[None, None, None, 0], buf_cp_atom_r)
        at_cp_frag.copy_from(at_mem_tensor_thr[None, None, None, 0], buf_cp_atom_r)
        ab_cp_frag.copy_from(ab_mem_tensor_thr[None, None, None, 0], buf_cp_atom_r)
        br_cp_frag.copy_from(br_mem_tensor_thr[None, None, None, 0], buf_cp_atom_r)
        # lw0: all
        bl_cp_frag.copy_to(bl_lds_tensor_thr_w, uni_cp_atom_w)
        at_cp_frag.copy_to(at_lds_tensor_thr_w, uni_cp_atom_w)
        ab_cp_frag.copy_to(ab_lds_tensor_thr_w, uni_cp_atom_w)
        br_cp_frag.copy_to(br_lds_tensor_thr_w, uni_cp_atom_w)

        rocdl = fx.rocdl
        gpu = fx.gpu
        const_expr = fx.const_expr
        range_constexpr = fx.range_constexpr

        # gr1: all
        bl_cp_frag.copy_from(bl_mem_tensor_thr[None, None, None, 1], buf_cp_atom_r)
        rocdl.sched_barrier(0)
        at_cp_frag.copy_from(at_mem_tensor_thr[None, None, None, 1], buf_cp_atom_r)
        rocdl.sched_barrier(0)
        ab_cp_frag.copy_from(ab_mem_tensor_thr[None, None, None, 1], buf_cp_atom_r)
        rocdl.sched_barrier(0)
        br_cp_frag.copy_from(br_mem_tensor_thr[None, None, None, 1], buf_cp_atom_r)
        rocdl.sched_barrier(0)
        
        gpu.barrier()

        # lr: bl0, at0
        bl_frag.copy_from(bl_lds_tensor_thr_r)
        at_frag.copy_from(at_lds_tensor_thr_r)
        rocdl.s_waitcnt(_encode_waitcnt(lgkmcnt=0))

        mem_b_half_cnt = bl_cp_frag.load().numel * A.dtype.width // 8 // 16
        mem_a_half_cnt = at_cp_frag.load().numel * A.dtype.width // 8 // 16
        lds_b_half_cnt = bl_frag.load().numel * A.dtype.width // 8 // 16
        lds_a_half_cnt = at_frag.load().numel * A.dtype.width // 8 // 16

        def hot_loop_scheduler(vmem_cnt, dsrd_cnt):
            if const_expr(is_gfx942):
                mfma_cnt = (TILE_M // 2 // 2 // 16) * (TILE_M // 2 // 2 // 16) * (TILE_K // 16)
                dswr_cnt = vmem_cnt
                if const_expr(TILE_M == 256 and TILE_N == 256):
                    rocdl.sched_mfma(2)
                    # ds write按照32bit计算
                    for _ in range_constexpr(dswr_cnt * 4):
                        rocdl.sched_dswr(1)
                        rocdl.sched_mfma(2)
                    for _ in range_constexpr(vmem_cnt):
                        rocdl.sched_vmem(1)
                        rocdl.sched_mfma(4)
                    for _ in range_constexpr(dsrd_cnt):
                        rocdl.sched_dsrd(1)
                        rocdl.sched_mfma(2)
                    rocdl.sched_mfma(mfma_cnt - 2 - dsrd_cnt * 2 - vmem_cnt * 4 - dswr_cnt * 8)
                else:
                    for _ in range_constexpr(dswr_cnt):
                        rocdl.sched_dswr(1)
                        rocdl.sched_mfma(2)
                        rocdl.sched_vmem(1)
                        rocdl.sched_mfma(2)
                    # for _ in range_constexpr(vmem_cnt):
                    #     rocdl.sched_mfma(4)
                    for _ in range_constexpr(dsrd_cnt):
                        rocdl.sched_dsrd(1)
                        rocdl.sched_mfma(1)
                    rocdl.sched_mfma(mfma_cnt - dsrd_cnt - vmem_cnt * 4 - dswr_cnt * 0)        
        assert K // TILE_K >= 2, "this kernel requires at least 2 iterations"
        for k, state in range(0, K // TILE_K - 0, 1, init=[]):

            # bl0 @ at0
            fx.gemm(mma_atom, c_tl_frag, bl_frag, at_frag, c_tl_frag)
            # lw: bl1
            rocdl.s_waitcnt(_encode_waitcnt(vmcnt=mem_a_half_cnt + mem_a_half_cnt + mem_b_half_cnt))
            bl_cp_frag.copy_to(bl_lds_tensor_thr_w, uni_cp_atom_w)
            # gr: bl2
            bl_cp_frag.copy_from(bl_mem_tensor_thr[None, None, None, k + 2], buf_cp_atom_r)
            # lr: ab0
            gpu.barrier()
            ab_frag.copy_from(ab_lds_tensor_thr_r, uni_cp_atom_r)
            hot_loop_scheduler(mem_b_half_cnt, lds_a_half_cnt)
            rocdl.sched_barrier(0)

            # bl0 @ ab0
            fx.gemm(mma_atom, c_bl_frag, bl_frag, ab_frag, c_bl_frag)
            # lw: at1
            rocdl.s_waitcnt(_encode_waitcnt(vmcnt=mem_a_half_cnt + mem_b_half_cnt + mem_b_half_cnt))
            at_cp_frag.copy_to(at_lds_tensor_thr_w, uni_cp_atom_w)
            # gr: at2
            at_cp_frag.copy_from(at_mem_tensor_thr[None, None, None, k + 2], buf_cp_atom_r)
            # lr: br0
            gpu.barrier()
            br_frag.copy_from(br_lds_tensor_thr_r, uni_cp_atom_r)
            hot_loop_scheduler(mem_a_half_cnt, lds_b_half_cnt)
            rocdl.sched_barrier(1)

            # br0 @ at0
            fx.gemm(mma_atom, c_tr_frag, br_frag, at_frag, c_tr_frag)
            # lw: ab1
            rocdl.s_waitcnt(_encode_waitcnt(vmcnt=mem_b_half_cnt + mem_b_half_cnt + mem_a_half_cnt))
            ab_cp_frag.copy_to(ab_lds_tensor_thr_w, uni_cp_atom_w)
            # gr: ab2
            ab_cp_frag.copy_from(ab_mem_tensor_thr[None, None, None, k + 2], buf_cp_atom_r)
            # lr: bl1
            gpu.barrier()
            bl_frag.copy_from(bl_lds_tensor_thr_r, uni_cp_atom_r)
            hot_loop_scheduler(mem_a_half_cnt, lds_b_half_cnt)
            rocdl.sched_barrier(2)

            # br0 @ ab0
            fx.gemm(mma_atom, c_br_frag, br_frag, ab_frag, c_br_frag)
            # lw: br1
            rocdl.s_waitcnt(_encode_waitcnt(vmcnt=mem_b_half_cnt + mem_a_half_cnt + mem_a_half_cnt))
            br_cp_frag.copy_to(br_lds_tensor_thr_w, uni_cp_atom_w)
            # gr: br2
            br_cp_frag.copy_from(br_mem_tensor_thr[None, None, None, k + 2], buf_cp_atom_r)
            # lr: at1
            gpu.barrier()
            at_frag.copy_from(at_lds_tensor_thr_r, uni_cp_atom_r)
            hot_loop_scheduler(mem_b_half_cnt, lds_a_half_cnt)
            rocdl.sched_barrier(3)

        if fx.const_expr(C.dtype == c_tl_frag.dtype):
            c_tl_frag.copy_to(c_tl_tile)
            c_tr_frag.copy_to(c_tr_tile)
            c_bl_frag.copy_to(c_bl_tile)
            c_br_frag.copy_to(c_br_tile)
        else:
            def f32_to_dtype(x):
                if fx.const_expr(C.dtype == fx.BFloat16):
                    round_bit = fx.Uint32(0x8000).ir_value().bitcast(fx.Float32.ir_type)
                    return ((x + round_bit).bitcast(fx.Uint32) >> 16).to(fx.Uint16).bitcast(fx.BFloat16)
                if fx.const_expr(C.dtype == fx.Float16):
                    return x.to(fx.Float16)
            buf_cp_atom_w = fx.make_copy_atom(fx.rocdl.BufferCopy64b(), C.dtype)
            c_frag_bf16 = fxu.Fragment.from_tiledmma(tiled_mma, BLOCK_N//2, BLOCK_M//2, "C", buf_cp_atom_w, dtype=C.dtype)
            c_frag_bf16.store(f32_to_dtype(c_tl_frag.load()))
            c_frag_bf16.copy_to(c_tl_tile)
            c_frag_bf16.store(f32_to_dtype(c_tr_frag.load()))
            c_frag_bf16.copy_to(c_tr_tile)
            c_frag_bf16.store(f32_to_dtype(c_bl_frag.load()))
            c_frag_bf16.copy_to(c_bl_tile)
            c_frag_bf16.store(f32_to_dtype(c_br_frag.load()))
            c_frag_bf16.copy_to(c_br_tile)

    @flyc.jit
    def launcher(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor,stream: fx.Stream):
        # recover static layout
        A = fxu.view_as(A, fx.make_layout((M, K), (K, 1)))
        if fx.const_expr(b_preshuffle):
            B = fx.Tensor(fx.make_view(
                fx.get_iter(B),
                # layout分成两部分：第一部分(m, n)描述一个wave内部划分为16行，(8列每组)x4组；第二部分为重复第一部分的次数
                # shape: (16, (8, 4)),   (N//16, K//32)
                # stride:(8,  (1, 128)), (K*16, 512))
                # 重新排列为第0维、第1维
                fx.make_layout(((16, N // 16), (8, 4, K // 32)), ((8, 16 * K), (1, 128, 512)))
            ))
        else:
            B = fxu.view_as(B, fx.make_layout((N, K), (K, 1)))

        C = fxu.view_as(C, fx.make_layout((M, N), (N, 1)))
        A = fx.flat_divide(A, (BLOCK_M, BLOCK_K))  # (BLOCK_M, num_blocks_m, num_blocks_k)
        B = fx.flat_divide(B, (BLOCK_N, BLOCK_K))  # (BLOCK_N, num_blocks_n, num_blocks_k)
        C = fx.flat_divide(C, (BLOCK_M, BLOCK_N)) # (BLOCK_M, BLOCK_N, num_blocks_m,num_blocks_n)
        grid_m = fx.get_scalar(C.shape[2])
        grid_n = fx.get_scalar(C.shape[3])
        # print("grid_m:", grid_m, "grid_n:", grid_n)

        value_attrs = None
        if fx.const_expr((TILE_M >= 128 and TILE_N > 128) or (TILE_M > 128 and TILE_N >= 128)):
            value_attrs = {"rocdl.waves_per_eu": 1,
                            "passthrough": [["amdgpu-agpr-alloc", "256,256"],]
                            }

        gemm_kernel_v4(A, B, C, value_attrs=value_attrs
                       ).launch(grid=(grid_m, grid_n, 1), block=(num_threads, 1, 1), stream=stream)
    return launcher

def test_gemm(compile, in_dtype, out_dtype, b_preshuffle, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K):
    A = torch.randn(M, K, dtype=in_dtype).cuda() / math.sqrt(K)
    B = torch.randn(N, K, dtype=in_dtype).cuda() / math.sqrt(K)
    C = torch.zeros(M, N, dtype=out_dtype).cuda()
    launcher = compile(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, b_preshuffle)

    if b_preshuffle:
        B_in = shuffle_weight(B)
    else:
        B_in = B

    hints = {
        #"maxnreg": 256,
        "opt_level": 2,
        #"llvm_options": ""
    }
    hints['llvm_options'] = {
        "amdgpu-mfma-vgpr-form": False,
    }
    args = (A, B_in, C, stream)
    kernel = flyc.compile[hints](launcher, *args)

    pyhip.run_perftest(kernel, *args,
                       num_verbose=0, num_flops=2*M*N*K,
                       num_name=f"gemm_{in_dtype}_{out_dtype}_{M}_{N}_{K}_{BLOCK_M}_{BLOCK_N}_{BLOCK_K}",)

    expected = A.to(out_dtype) @ B.to(out_dtype).T
    is_correct = torch.allclose(C, expected, atol=1e-5, rtol=1e-5)
    print(f"Result correct: {is_correct} Max diff: {(C - expected).abs().max().item()}")

if __name__ == "__main__":
    M, N, K = 256*8*2, 256*10*2, 1024*8
    #test_gemm(compile_v1, M, N, K, BLOCK_M = 64, BLOCK_N = 64, BLOCK_K = 32)
    #test_gemm(compile_v1, M, N, K, BLOCK_M = 128, BLOCK_N = 128, BLOCK_K = 32)
    #test_gemm(compile_v1, M, N, K, BLOCK_M = 256, BLOCK_N = 256, BLOCK_K = 32)
    
    for b_preshuffle in [False, True]:
        for in_dtype in [torch.bfloat16, torch.float16]:
            for out_dtype in [torch.bfloat16, torch.float16, torch.float32]:
                test_gemm(compile_v4, in_dtype, out_dtype, b_preshuffle,
                          M, N, K, BLOCK_M = 256, BLOCK_N = 256, BLOCK_K = 64)

    # torch.float32 also works, but the performance is not good
    test_gemm(compile_v4, torch.float32, torch.float32, False, M, N, K, BLOCK_M = 256, BLOCK_N = 256, BLOCK_K = 32)
