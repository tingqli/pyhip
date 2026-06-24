import math
import os
os.environ.setdefault("FLYDSL_RUNTIME_ENABLE_CACHE", "0")

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
import pyhip

import pyhip.contrib.flydsl as fxu

# fxu.enable_dump_ir(True)

if 0:
    make_tiled_copy = fxu.make_tiled_copy
    make_tiled_mma = fxu.make_tiled_mma
    make_tiled_copy_A = fxu.make_tiled_copy_A
    make_tiled_copy_B = fxu.make_tiled_copy_B
    make_tiled_copy_C = fxu.make_tiled_copy_C
else:
    make_tiled_copy = fx.make_tiled_copy
    make_tiled_mma = fx.make_tiled_mma
    make_tiled_copy_A = fx.make_tiled_copy_A
    make_tiled_copy_B = fx.make_tiled_copy_B
    make_tiled_copy_C = fx.make_tiled_copy_C    

#fxu.enable_dump_ir(True)
_, stream = pyhip.set_device()


def compile_v0(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K):
    @flyc.kernel
    def gemm_kernel_v0(A: fx.Tensor,B: fx.Tensor,C: fx.Tensor,):
        tid = fx.thread_idx.x
        bid_m, bid_n = fx.block_idx.x, fx.block_idx.y

        A = fx.rocdl.make_buffer_tensor(A)
        B = fx.rocdl.make_buffer_tensor(B)
        C = fx.rocdl.make_buffer_tensor(C)

        bA = fx.slice(A, (None, None, bid_m, None))   # BLOCK_M, BLOCK_K, (idx_block_m), num_blocks_k
        bB = fx.slice(B, (None, None, bid_n, None))   # BLOCK_M, BLOCK_K, (idx_block_n), num_blocks_k
        bC = fx.slice(C, (None, None, bid_m, bid_n))  # BLOCK_M, BLOCK_N, (idx_block_m), (idx_block_n)    

        num_blocks_k = fx.get_scalar(fx.size(bA.layout[2]))
        print("num_blocks_k:", num_blocks_k)

        mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, fx.BFloat16))
        thr_layout_mnk = fx.make_layout((2, 2, 1), (1, 2, 0))
        k_perm = fx.make_layout((4, 4, 2), (1, 8, 4))
        permutation_mnk = (None, None, k_perm)
        tiled_mma = make_tiled_mma(mma_atom, thr_layout_mnk, permutation_mnk)

        # tiled_mma的标准用法就是表达一个CU上全部可并行 mma_atom 的一次执行
        # 但是考虑到硬件超线程的存在，这不意味着一个CU上只能有4个wave并行，可以是work-group允许的更多
        # 也就是 tiled_mma 表达的是 mma_atom per wave 而不是 per SIMD硬件单元。
        # 并且跟之前手写汇编喜欢按照wave切分之后再循环不同，这里采用另一种布局，就是
        # 把所有可并行 mma-atom 看作一个整体交织起来组成一个 tile，然后重复这样的 tile 多次来覆盖
        # 一个gemm问题的 MNK 计算空间。也就是线程空间相邻的 mma-atom 在 MNK 空间也是相邻的，
        # 这样可以更好利用硬件的cache和寄存器。
        fxu.inspect(tiled_mma)

        copy_atomAB = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), A.dtype)
        copy_atomC = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), C.dtype)

        tiled_copy_A = make_tiled_copy_A(copy_atomAB, tiled_mma)
        tiled_copy_B = make_tiled_copy_B(copy_atomAB, tiled_mma)
        tiled_copy_C = make_tiled_copy_C(copy_atomC, tiled_mma)

        thr_copy_A = tiled_copy_A.get_slice(tid)
        thr_copy_B = tiled_copy_B.get_slice(tid)
        thr_copy_C = tiled_copy_C.get_slice(tid)

        thr_mma = tiled_mma.thr_slice(tid)

        # partition_S()
        #          input bA:  (BLOCK_M, BLOCK_K, num_blocks_k)
        #
        # 令  num_mma_tiles_BM =  BLOCK_M//tile_size_m
        # 令  num_mma_tiles_BN =  BLOCK_N//tile_size_n
        # 令  num_mma_tiles_BK =  BLOCK_K//tile_size_k
        #
        # copy_src_A:  ((trg_val, rest_val), num_mma_tiles_BM, num_mma_tiles_BK, num_blocks_k)
        # copy_src_B:  ((trg_val, rest_val), num_mma_tiles_BN, num_mma_tiles_BK, num_blocks_k)
        # copy_src_C:  ((trg_val, rest_val), num_mma_tiles_BM, num_mma_tiles_BN)
        #
        #  - trg_val: 单次copy_atom指令每个线程所读入的数据量
        #  - rest_val: 重复多少次copy_atom可以填满 tv-layout_A/B_tiled 的数据量 (总大小是 tile_size_mnk)
        #  - RestM, RestN, RestK .... 分别在M/N/K方向上重复多次 tv-layout
        #    直到达到 bA/bB (BLOCK_M/N, BLOCK_K, num_blocks_k) 的大小

        copy_src_A = thr_copy_A.partition_S(bA)
        copy_src_B = thr_copy_B.partition_S(bB)
        copy_dst_C = thr_copy_C.partition_S(bC)

        print("        bA:", bA)
        print("copy_src_A:", copy_src_A)
        print("        bC:", bC)
        print("copy_dst_C:", copy_dst_C)

        # 这些 tensor 类似 C++中循环外的数组定义，循环中读写它们的数据并不会修改他们的定义，
        # 因此没有SSA值的问题。可以在循环外定义这些tensor，循环中直接使用
        # 另外寄存器资源有限，只分配 BLOCK_M/N/K 这么多。
        #
        # frag_A: (FrgV, num_mma_tiles_BM, num_mma_tiles_BK) 所有线程一起总大小是 BLOCK_M * BLOCK_K 个元素
        # frag_B: (FrgV, num_mma_tiles_BN, num_mma_tiles_BK) 所有线程一起总大小是 BLOCK_N * BLOCK_K 个元素
        # frag_C: (FrgV, num_mma_tiles_BM, num_mma_tiles_BN) 所有线程一起总大小是 BLOCK_M * BLOCK_N 个元素
        #
        frag_A = thr_mma.make_fragment_A(bA[None, None, 0])
        frag_B = thr_mma.make_fragment_B(bB[None, None, 0])
        frag_C = thr_mma.make_fragment_C(bC)
        frag_C.fill(0)

        # copy_frag_A 需要把 frag_A 的数据按照 tv-layout_A_tiled 的要求重新排列
        # 主要就是要凑出连续的 copy_atom.layout_ref_tv里面要求的那么多个元素，
        # 才能被 copy_atom 指令搬运，使用 128bit copy_atom时，需要从 frag_A 的第三个维度中
        # num_mma_tiles_BK 中提取 stride最小的部分补入 FrgV 维度中直到满足 copy_atom.layout_ref_tv 的原子frag数据大小
        #
        # 假设 d = CopyAtom_FrgV // TiledMma_FrgV, 则
        #
        # copy_frag_A : (CopyAtom_FrgV, num_mma_tiles_BM, num_mma_tiles_BK//d)
        # copy_frag_B : (CopyAtom_FrgV, num_mma_tiles_BN, num_mma_tiles_BK//d)
        #
        copy_frag_A = thr_copy_A.retile(frag_A) 
        copy_frag_B = thr_copy_B.retile(frag_B)
        copy_frag_C = thr_copy_C.retile(frag_C)

        print(bA[None, None, 0])
        print(frag_A, copy_frag_A)
        print(frag_B, copy_frag_B)
        print(frag_C, copy_frag_C)

        # 动态loop
        for k, state in range(fx.Index(0), fx.Index(num_blocks_k), fx.Index(1), init=[]):

            # 此处发起多个copy-atom指令，从外存读入如下大小的数据到每个线程的寄存器中：
            # 因为 copy_src_A 已经按照 thread_id 切片了，所以从 copy_src_A/B 的布局看不到
            #
            #  令 num_mma_tiles_BM =  BLOCK_M//tile_size_m
            #  令 num_mma_tiles_BN =  BLOCK_N//tile_size_n
            #  令 num_mma_tiles_BK =  BLOCK_K//tile_size_k
            #
            # copy_src_A/copy_frag_A: ((trg_val, rest_val), num_mma_tiles_BM, num_mma_tiles_BK, num_blocks_k)
            # copy_src_B/copy_frag_B: ((trg_val, rest_val), num_mma_tiles_BN, num_mma_tiles_BK, num_blocks_k)
            fx.copy(copy_atomAB, copy_src_A[None, None, None, k], copy_frag_A, pred=None)
            fx.copy(copy_atomAB, copy_src_B[None, None, None, k], copy_frag_B, pred=None)

            # 此处发起多个mma-atom指令，计算每个线程的寄存器中的数据：
            #  - frag_A: (FrgV, num_mma_tiles_BM, num_mma_tiles_BK)
            #  - frag_B: (FrgV, num_mma_tiles_BN, num_mma_tiles_BK)
            #  - frag_C: (FrgV, num_mma_tiles_BM, num_mma_tiles_BN)
            #
            #   for m in range(num_mma_tiles_BM):
            #     for n in range(num_mma_tiles_BN):
            #       for k in range(num_mma_tiles_BK):
            #          mma_atom(frag_D[:, m, n], frag_A[:, m, k], frag_B[:, n, k], frag_C[:, m, n])
            fx.gemm(mma_atom, frag_C, frag_A, frag_B, frag_C)

        # 反思一下上面的编程模型，其实 tiled_mma 的单位有两层，一个是所有线程的 mma_atom 按照 make_tiled_mma 的 第二个参数
        # atom_layout(thr_layout_mnk)，以及第三个参数 permutation_mnk, 在 MNK 计算空间中布局组成所谓的 tiled_mma tile，
        # 这是最小层级，代表每个 mma 单元执行一条（当 permutation_mnk 不为空时可能是多条 ）指令所完成的计算。另一个层级
        # 是 BLOCK_M/N/K 代表的一个 block 的计算空间， tiled_mma 的API可以处理这个更大的空间，对其进行partition，并且
        # 以这个更大的空间为单位进行循环，直到完成整个 gemm 的计算。
        #
        # 这种方式下，从C矩阵角度看，4个wave，重复4次 tile的组合方式如下:
        #
        #  +-----+-----+
        #  |w0 w2|w0 w2|
        #  |w1 w3|w1 w3|
        #  +-----+-----+
        #  |w0 w2|w0 w2|
        #  |w1 w3|w1 w3|
        #  +-----+-----+
        #
        # 如果想使用下面的组合方式，则需要设定 permutation_mnk 为非空的值
        # 
        #  +-----+-----+
        #  |w0 w0|w2 w2|
        #  |w0 w0|w2 w2|
        #  +-----+-----+
        #  |w1 w1|w3 w3|
        #  |w1 w1|w3 w3|
        #  +-----+-----+
        #

        fx.copy(copy_atomC, copy_frag_C, copy_dst_C, pred=None)

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
        gemm_kernel_v0(A, B, C).launch(grid=(grid_m, grid_n, 1), block=(256, 1, 1), stream=stream)
    return launcher







def test_gemm(compile, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K):
    in_dtype = torch.bfloat16
    out_dtype = torch.float32
    A = torch.randn(M, K, dtype=in_dtype).cuda() / math.sqrt(K)
    B = torch.randn(N, K, dtype=in_dtype).cuda() / math.sqrt(K)
    C = torch.zeros(M, N, dtype=out_dtype).cuda()
    launcher = compile(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K)

    hints = {
        "maxnreg": 256,
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
    #test_gemm(compile_v0, M, N, K, BLOCK_M = 64, BLOCK_N = 64, BLOCK_K = 32)
    #test_gemm(compile_v0, M, N, K, BLOCK_M = 128, BLOCK_N = 128, BLOCK_K = 32)
    #test_gemm(compile_v0, M, N, K, BLOCK_M = 256, BLOCK_N = 256, BLOCK_K = 32)
    test_gemm(compile_v0, M, N, K, BLOCK_M = 256, BLOCK_N = 256, BLOCK_K = 64)
