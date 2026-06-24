# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
import math
import os
os.environ.setdefault("FLYDSL_RUNTIME_ENABLE_CACHE", "0")

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
import pyhip

import pyhip.contrib.flydsl as fxu

# fxu.enable_dump_ir(True)

_, stream = pyhip.set_device()

"""
参考 https://github.com/ROCm/gfx950-gluon-tutorials/tree/main/kernels/gemm/a16w16/v3_lds
引入LDS，但是不排流水，另外封装了TiledCopy到 Fragment 类型中

"""
def compile_v2(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K):
    num_threads = 256

    @fx.struct
    class SharedStorage:
        a0: fx.Array[fx.BFloat16, BLOCK_M*BLOCK_K, 16]
        b0: fx.Array[fx.BFloat16, BLOCK_N*BLOCK_K, 16]
    
    @flyc.kernel
    def gemm_kernel_v2(A: fx.Tensor,B: fx.Tensor,C: fx.Tensor,):
        tid = fx.thread_idx.x
        bid_m, bid_n = fx.block_idx.x, fx.block_idx.y

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        swz = fx.SwizzleType.get(3, 3, 3)
        sA = fx.make_view(lds.a0.ptr, fx.make_composed_layout(fx.static(swz), fx.make_ordered_layout((BLOCK_M, BLOCK_K), (1, 0))))
        sB = fx.make_view(lds.b0.ptr, fx.make_composed_layout(fx.static(swz), fx.make_ordered_layout((BLOCK_N, BLOCK_K), (1, 0))))

        A = fx.rocdl.make_buffer_tensor(A)
        B = fx.rocdl.make_buffer_tensor(B)
        C = fx.rocdl.make_buffer_tensor(C)

        bA = fx.slice(A, (None, None, bid_m, None))   # BLOCK_M, BLOCK_K, (idx_block_m), num_blocks_k
        bB = fx.slice(B, (None, None, bid_n, None))   # BLOCK_N, BLOCK_K, (idx_block_n), num_blocks_k
        bC = fx.slice(C, (None, None, bid_m, bid_n))  # BLOCK_M, BLOCK_N, (idx_block_m), (idx_block_n)    

        num_blocks_k = fx.get_scalar(fx.size(bA.layout[2]))
        print("num_blocks_k:", num_blocks_k)

        mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, fx.BFloat16))
        thr_layout_mnk = fx.make_layout((2, 2, 1), (1, 2, 0))
        k_perm = fx.make_layout((4, 4, 2), (1, 8, 4))
        permutation_mnk = (None, None, k_perm)
        tiled_mma = fx.make_tiled_mma(mma_atom, thr_layout_mnk, permutation_mnk)

        copy_buff128b = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), A.dtype)   # buffer_load/store_dwordx4
        copy_ds128b = fx.make_copy_atom(fx.UniversalCopy128b(), A.dtype)        # ds_read/write_b128
        copy_buff32b = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), C.dtype)
        
        frag_A = fxu.Fragment.from_tiledmma(tiled_mma, BLOCK_M, BLOCK_K, "A", copy_ds128b)
        frag_B = fxu.Fragment.from_tiledmma(tiled_mma, BLOCK_N, BLOCK_K, "B", copy_ds128b)
        frag_C = fxu.Fragment.from_tiledmma(tiled_mma, BLOCK_M, BLOCK_N, "C", copy_buff32b)
        frag_C.fill(0)

        VECT_WIDTH = 128//A.dtype.width
        tv_layout, tv_tilemn = fxu.make_mem_coalescing_2d_tv_layout(
                                            num_threads, VECT_WIDTH, BLOCK_K)
        frag_copy_A = fxu.Fragment.from_tvlayout(A.dtype, BLOCK_M, BLOCK_K, tv_layout, tv_tilemn, (copy_buff128b, copy_ds128b))
        frag_copy_B = fxu.Fragment.from_tvlayout(B.dtype, BLOCK_N, BLOCK_K, tv_layout, tv_tilemn, (copy_buff128b, copy_ds128b))

        for k, state in range(fx.Index(0), fx.Index(num_blocks_k), fx.Index(1), init=[]):

            frag_copy_A.copy_from(bA[None, None, k], copy_buff128b)
            frag_copy_B.copy_from(bB[None, None, k], copy_buff128b)

            frag_copy_A.copy_to(sA, copy_ds128b)
            frag_copy_B.copy_to(sB, copy_ds128b)

            # wait all threads finish their part of copy before load them using another TV-layout
            fx.gpu.barrier()

            frag_A.copy_from(sA)
            frag_B.copy_from(sB)

            fx.gemm(tiled_mma, frag_C, frag_A, frag_B, frag_C)
        frag_C.copy_to(bC)

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
        gemm_kernel_v2(A, B, C).launch(grid=(grid_m, grid_n, 1), block=(num_threads, 1, 1), stream=stream)
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
    test_gemm(compile_v2, M, N, K, BLOCK_M = 256, BLOCK_N = 256, BLOCK_K = 64)
