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

#fxu.enable_dump_ir(True)

_, stream = pyhip.set_device()

"""
对TiledMMA进行了一定简化和封装，性能不变，隐藏了一些复杂的细节，用户可以更专注于计算部分，而不需要关心copy的部分。
"""
def compile_v1(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K):
    @flyc.kernel
    def gemm_kernel_v1(A: fx.Tensor,B: fx.Tensor,C: fx.Tensor,):
        tid = fx.thread_idx.x
        bid_m, bid_n = fx.block_idx.x, fx.block_idx.y

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

        copy_atomAB = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), A.dtype)
        copy_atomC = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), C.dtype)

        """
         this make_fragment_X() API clearly shows that the fragment is a per-thread-view
         of an original block of size (BLOCK_M, BLOCK_K).
         with given copy_atom, many preparation work is done inside, including preparing
         retiled view of the fragment inside.

         this set of API is trying to hide some complexity of the copying process, thus
         the user can focus on the computation part, and not worry about the copying part.
        """
        frag_A = fxu.Fragment.from_tiledmma(tiled_mma, BLOCK_M, BLOCK_K, "A", copy_atomAB)
        frag_B = fxu.Fragment.from_tiledmma(tiled_mma, BLOCK_N, BLOCK_K, "B", copy_atomAB)
        frag_C = fxu.Fragment.from_tiledmma(tiled_mma, BLOCK_M, BLOCK_N, "C", copy_atomC)
        frag_C.fill(0)

        """
          after partition, it becomes a per-thread-view with its first 2 modes
          being replaced by 3 modes, thus the slicing pattern
            copyA[None, None, None, k] references same data as bA[None, None, k]
        """
        copyA = frag_A.partition_S(bA)
        copyB = frag_B.partition_S(bB)

        for k, state in range(fx.Index(0), fx.Index(num_blocks_k), fx.Index(1), init=[]):
            
            """
            不做预先的partition, 直接指定 mem Tensor 输入也是可以的，而且更加简单:
                frag_A.copy_from(bA[None, None, k])
                frag_B.copy_from(bB[None, None, k])
            
            但是会在loop中引入些许额外的 partition 计算 overhead， 因此我们允许两种不同的方式，
            copy_from/copy_to 内部编译期会check实际传入的tensor是per-thread-view还是per-block-view,
            并做相应的处理。
            """

            frag_A.copy_from(copyA[None, None, None, k])
            frag_B.copy_from(copyB[None, None, None, k])
            
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
        gemm_kernel_v1(A, B, C).launch(grid=(grid_m, grid_n, 1), block=(256, 1, 1), stream=stream)
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
    test_gemm(compile_v1, M, N, K, BLOCK_M = 256, BLOCK_N = 256, BLOCK_K = 64)
