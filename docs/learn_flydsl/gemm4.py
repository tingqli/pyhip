# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
import math
import os
os.environ.setdefault("FLYDSL_RUNTIME_ENABLE_CACHE", "0")

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.runtime.device import get_rocm_arch  # noqa: E402

import pyhip

import pyhip.contrib.flydsl as fxu

#fxu.enable_dump_ir(True)

_, stream = pyhip.set_device()

"""
参考
 https://github.com/ROCm/gfx950-gluon-tutorials/tree/main/kernels/gemm/a16w16/v7_sliceN
 https://github.com/ROCm/gfx950-gluon-tutorials/tree/main/kernels/gemm/a16w16/v8_sliceMN

ping-pong buffer LDS, split long MN dimension:
"""
def compile_v4(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K):
    num_threads = 256

    TILE_M = BLOCK_M
    TILE_N = BLOCK_N
    TILE_K = BLOCK_K

    def _encode_waitcnt(vmcnt=63, expcnt=7, lgkmcnt=63):
        """Encode s_waitcnt bitfield for CDNA3 (gfx94x)."""
        vm_lo = vmcnt & 0xF
        vm_hi = (vmcnt >> 4) & 0x3
        return vm_lo | (expcnt << 4) | (lgkmcnt << 8) | (vm_hi << 14)

    @fx.struct
    class SharedStorage:
        at_lds: fx.Array[fx.BFloat16, BLOCK_M//2*BLOCK_K, 16]
        ab_lds: fx.Array[fx.BFloat16, BLOCK_M//2*BLOCK_K, 16]
        bl_lds: fx.Array[fx.BFloat16, BLOCK_N//2*BLOCK_K, 16]
        br_lds: fx.Array[fx.BFloat16, BLOCK_N//2*BLOCK_K, 16]

    @flyc.kernel
    def gemm_kernel_v4(A: fx.Tensor,B: fx.Tensor,C: fx.Tensor,):
        gpu_arch = get_rocm_arch()
        is_gfx942 = str(gpu_arch).startswith("gfx942")

        tid = fx.thread_idx.x
        bid_m, bid_n = fx.block_idx.x, fx.block_idx.y

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
        uni_cp_atom_r = fx.make_copy_atom(fx.UniversalCopy128b(), fx.BFloat16) # ds_read/write_b128
        uni_cp_atom_w = fx.make_copy_atom(fx.UniversalCopy32b(), fx.BFloat16)  # ds_read/write_b32
        VECT_WIDTH = 128//A.dtype.width
        tv_layout, tv_tilemn = fxu.make_mem_coalescing_2d_tv_layout(
                                            num_threads, VECT_WIDTH, BLOCK_K)
        at_cp_frag = fxu.Fragment.from_tvlayout(A.dtype, BLOCK_M//2, BLOCK_K, tv_layout, tv_tilemn, (buf_cp_atom_r, uni_cp_atom_w))
        ab_cp_frag = fxu.Fragment.from_tvlayout(A.dtype, BLOCK_M//2, BLOCK_K, tv_layout, tv_tilemn, (buf_cp_atom_r, uni_cp_atom_w))
        bl_cp_frag = fxu.Fragment.from_tvlayout(B.dtype, BLOCK_N//2, BLOCK_K, tv_layout, tv_tilemn, (buf_cp_atom_r, uni_cp_atom_w))
        br_cp_frag = fxu.Fragment.from_tvlayout(B.dtype, BLOCK_N//2, BLOCK_K, tv_layout, tv_tilemn, (buf_cp_atom_r, uni_cp_atom_w))

        num_blocks_k = fx.get_scalar(fx.size(bA.layout[2]))
        print("num_blocks_k:", num_blocks_k)

        mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, fx.BFloat16))
        thr_layout_mnk = fx.make_layout((2, 2, 1), (1, 2, 0))
        k_perm = fx.make_layout((4, 4, 2), (1, 8, 4))
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
        ab_frag = fxu.Fragment.from_tiledmma(tiled_mma, BLOCK_M//2, BLOCK_K, "B", uni_cp_atom_r)
        bl_frag = fxu.Fragment.from_tiledmma(tiled_mma, BLOCK_N//2, BLOCK_K, "A", uni_cp_atom_r)
        br_frag = fxu.Fragment.from_tiledmma(tiled_mma, BLOCK_N//2, BLOCK_K, "A", uni_cp_atom_r)

        buf_cp_atom_w = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), C.dtype) 
        c_tl_frag = fxu.Fragment.from_tiledmma(tiled_mma, BLOCK_N//2, BLOCK_M//2, "C", buf_cp_atom_w)
        c_tr_frag = fxu.Fragment.from_tiledmma(tiled_mma, BLOCK_N//2, BLOCK_M//2, "C", buf_cp_atom_w)
        c_bl_frag = fxu.Fragment.from_tiledmma(tiled_mma, BLOCK_N//2, BLOCK_M//2, "C", buf_cp_atom_w)
        c_br_frag = fxu.Fragment.from_tiledmma(tiled_mma, BLOCK_N//2, BLOCK_M//2, "C", buf_cp_atom_w)

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

        mem_b_half_cnt = bl_cp_frag.load().numel * fx.BFloat16.width // 8 // 16
        mem_a_half_cnt = at_cp_frag.load().numel * fx.BFloat16.width // 8 // 16
        lds_b_half_cnt = bl_frag.load().numel * fx.BFloat16.width // 8 // 16
        lds_a_half_cnt = at_frag.load().numel * fx.BFloat16.width // 8 // 16
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

        c_tl_frag.copy_to(c_tl_tile)
        c_tr_frag.copy_to(c_tr_tile)
        c_bl_frag.copy_to(c_bl_tile)
        c_br_frag.copy_to(c_br_tile)

    @flyc.jit
    def launcher(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor,stream: fx.Stream):
        # recover static layout
        A = fxu.view_as(A, fx.make_layout((M, K), (K, 1)))
        B = fxu.view_as(B, fx.make_layout((N, K), (K, 1)))
        C = fxu.view_as(C, fx.make_layout((M, N), (N, 1)))
        A = fx.flat_divide(A, (BLOCK_M, BLOCK_K))  # (BLOCK_M, num_blocks_m, num_blocks_k)
        B = fx.flat_divide(B, (BLOCK_N, BLOCK_K))  # (BLOCK_N, num_blocks_n, num_blocks_k)
        C = fx.flat_divide(C, (BLOCK_M, BLOCK_N)) # (BLOCK_M, BLOCK_N, num_blocks_m,num_blocks_n)
        grid_m = fx.get_scalar(C.shape[2])
        grid_n = fx.get_scalar(C.shape[3])
        print("grid_m:", grid_m, "grid_n:", grid_n)
        gemm_kernel_v4(A, B, C).launch(grid=(grid_m, grid_n, 1), block=(num_threads, 1, 1), stream=stream)
    return launcher

def test_gemm(compile, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K):
    in_dtype = torch.bfloat16
    out_dtype = torch.float32
    A = torch.randn(M, K, dtype=in_dtype).cuda() / math.sqrt(K)
    B = torch.randn(N, K, dtype=in_dtype).cuda() / math.sqrt(K)
    C = torch.zeros(M, N, dtype=out_dtype).cuda()
    launcher = compile(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K)

    hints = {
        #"maxnreg": 256,
        "opt_level": 2,
        #"llvm_options": ""
    }
    hints['llvm_options'] = {
        "amdgpu-mfma-vgpr-form": False,
    }
    args = (A, B, C, stream)
    kernel = flyc.compile[hints](launcher, *args)

    pyhip.run_perftest(kernel, *args,
                       num_verbose=0, num_flops=2*M*N*K,
                       num_name=f"gemm_{M}_{N}_{K}_{BLOCK_M}_{BLOCK_N}_{BLOCK_K}",)

    expected = A.to(out_dtype) @ B.to(out_dtype).T
    is_correct = torch.allclose(C, expected, atol=1e-5, rtol=1e-5)
    print(f"Result correct: {is_correct} Max diff: {(C - expected).abs().max().item()}")

if __name__ == "__main__":
    M, N, K = 256*8*2, 256*10*2, 1024*8
    #test_gemm(compile_v1, M, N, K, BLOCK_M = 64, BLOCK_N = 64, BLOCK_K = 32)
    #test_gemm(compile_v1, M, N, K, BLOCK_M = 128, BLOCK_N = 128, BLOCK_K = 32)
    #test_gemm(compile_v1, M, N, K, BLOCK_M = 256, BLOCK_N = 256, BLOCK_K = 32)
    test_gemm(compile_v4, M, N, K, BLOCK_M = 256, BLOCK_N = 256, BLOCK_K = 64)
