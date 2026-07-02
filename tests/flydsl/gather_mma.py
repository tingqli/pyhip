import math
import os
os.environ.setdefault("FLYDSL_RUNTIME_ENABLE_CACHE", "0")

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
import pyhip

import pyhip.contrib.flydsl as fxu

fxu.enable_dump_ir(True)

_, stream = pyhip.set_device()


def compile_v1(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K):
    @flyc.kernel
    def gemm_kernel_v1(A: fx.Tensor, B: fx.Tensor, Aidx: fx.Tensor, C: fx.Tensor,):
        tid = fx.thread_idx.x
        bid_m, bid_n = fx.block_idx.x, fx.block_idx.y

        A = fx.rocdl.make_buffer_tensor(A)
        B = fx.rocdl.make_buffer_tensor(B)
        C = fx.rocdl.make_buffer_tensor(C)
        Aidx = fx.rocdl.make_buffer_tensor(Aidx)

        bAidx = fx.slice(Aidx, (None, bid_m))       # BLOCK_M, (idx_block_m)
        bB = fx.slice(B, (None, None, bid_n, None)) # BLOCK_N, BLOCK_K, (idx_block_n), num_blocks_k
        bC = fx.slice(C, (None, None, bid_m, bid_n))# BLOCK_M, BLOCK_N, (idx_block_m), (idx_block_n)    

        num_blocks_k = fx.get_scalar(fx.size(bB.layout[2]))
        print("num_blocks_k:", num_blocks_k)

        mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, fx.BFloat16))
        thr_layout_mnk = fx.make_layout((2, 2, 1), (1, 2, 0))
        k_perm = fx.make_layout((4, 4, 2), (1, 8, 4))
        permutation_mnk = (None, None, k_perm)
        tiled_mma = fx.make_tiled_mma(mma_atom, thr_layout_mnk, permutation_mnk)

        copy_atomAB = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), A.dtype)
        copy_atomC = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), C.dtype)

        # 有没有办法获取当前线程的一个 frag_idx, 里面存放着当前线程要访问的每一行的真实坐标
        # 例如当前线程要访问第 16, 32 两行，并且按照 16,32,16,32 访问，而Aidx映射表 16映射到 123
        # 32 映射到 456, 那么 frag_idx 里面就有两项 123, 456，并且通过broadcast方式排列为跟 16,32,16,32
        # 一样的次序 123,456,123,456 ?

        frag_A = fxu.Fragment.from_tiledmma(tiled_mma, BLOCK_M, BLOCK_K, "A", copy_atomAB)
        
        # get column tensor
        col_tensor = fx.make_view(fx.make_int_tuple(0), fx.make_layout((BLOCK_M, BLOCK_K, num_blocks_k), (0, 1, BLOCK_K)))
        col_tview = frag_A.partition_S(col_tensor)
        print(" col_tview: ", col_tview)

        frag_rows = frag_A.load_gather_rows(bAidx)

        frag_B = fxu.Fragment.from_tiledmma(tiled_mma, BLOCK_N, BLOCK_K, "B", copy_atomAB)
        frag_C = fxu.Fragment.from_tiledmma(tiled_mma, BLOCK_M, BLOCK_N, "C", copy_atomC)
        frag_C.fill(0)
        copyB = frag_B.partition_S(bB)

        for k, state in range(fx.Index(0), fx.Index(num_blocks_k), fx.Index(1), init=[]):
            
            frag_A.copy_from(A, 
                             rows = frag_rows,
                             cols = col_tview[None, None, None, k])
            # gather_copy(frag_dst, bAidx_frag, col_tview[None, None, None, k])
            frag_B.copy_from(copyB[None, None, None, k])
            
            fx.gemm(tiled_mma, frag_C, frag_A, frag_B, frag_C)

        frag_C.copy_to(bC)

    @flyc.jit
    def launcher(A: fx.Tensor, B: fx.Tensor, Aidx: fx.Tensor, C: fx.Tensor,stream: fx.Stream):
        # recover static layout
        A = fxu.view_as(A, fx.make_layout((M, K), (K, 1)))
        B = fxu.view_as(B, fx.make_layout((N, K), (K, 1)))
        Aidx = fxu.view_as(Aidx, fx.make_layout(M, 1))
        C = fxu.view_as(C, fx.make_layout((M, N), (N, 1)))

        Aidx = fx.flat_divide(Aidx, (BLOCK_M, ))  # (BLOCK_M, num_blocks_m)
        B = fx.flat_divide(B, (BLOCK_N, BLOCK_K))  # (BLOCK_N, num_blocks_n, num_blocks_k)
        C = fx.flat_divide(C, (BLOCK_M, BLOCK_N)) # (BLOCK_M, BLOCK_N, num_blocks_m,num_blocks_n)
        grid_m = fx.get_scalar(C.shape[2])
        grid_n = fx.get_scalar(C.shape[3])
        print("grid_m:", grid_m, "grid_n:", grid_n)
        gemm_kernel_v1(A, B, Aidx, C).launch(grid=(grid_m, grid_n, 1), block=(256, 1, 1), stream=stream)
    return launcher

def test_gemm(compile, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K):
    in_dtype = torch.bfloat16
    out_dtype = torch.float32
    A = torch.randn(M, K, dtype=in_dtype).cuda() / math.sqrt(K)
    B = torch.randn(N, K, dtype=in_dtype).cuda() / math.sqrt(K)
    Aidx = torch.arange(0, M, 1, dtype=torch.int32).cuda()
    Aidx = torch.randint(0, M, (M,), dtype=torch.int32).cuda()
    C = torch.zeros(M, N, dtype=out_dtype).cuda()
    launcher = compile(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K)

    print(Aidx[:256].view(-1,16))
    hints = {
        #"maxnreg": 256,
        "opt_level": 2,
        #"llvm_options": ""
    }
    hints['llvm_options'] = {
        "amdgpu-mfma-vgpr-form": False,
    }
    args = (A, B, Aidx, C, stream)
    kernel = flyc.compile[hints](launcher, *args)

    pyhip.run_perftest(kernel, *args,
                       num_verbose=0, num_flops=2*M*N*K,
                       num_name=f"gemm_{M}_{N}_{K}_{BLOCK_M}_{BLOCK_N}_{BLOCK_K}",)

    expected = A[Aidx, :].to(out_dtype) @ B.to(out_dtype).T
    is_correct = torch.allclose(C, expected, atol=1e-5, rtol=1e-5)
    print(f"Result correct: {is_correct} Max diff: {(C - expected).abs().max().item()}")

if __name__ == "__main__":
    M, N, K = 256*8*2, 256*10*2, 1024*8
    #test_gemm(compile_v1, M, N, K, BLOCK_M = 64, BLOCK_N = 64, BLOCK_K = 32)
    #test_gemm(compile_v1, M, N, K, BLOCK_M = 128, BLOCK_N = 128, BLOCK_K = 32)
    #test_gemm(compile_v1, M, N, K, BLOCK_M = 256, BLOCK_N = 256, BLOCK_K = 32)
    test_gemm(compile_v1, M, N, K, BLOCK_M = 256, BLOCK_N = 256, BLOCK_K = 64)
