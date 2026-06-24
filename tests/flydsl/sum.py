import math
import os
os.environ.setdefault("FLYDSL_RUNTIME_ENABLE_CACHE", "0")

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.ast_rewriter import ASTRewriter

import pyhip

import pyhip.contrib.flydsl as fxu

_, stream = pyhip.set_device()

# sum along an axis
# a.sum(dim=1) [B, M, N] -> [B, K]

def compile_sum_dim(L, M, N):
    num_threads = 64
    @flyc.kernel(known_block_size=[num_threads, 1, 1]) # known_block_size at compile time
    def kernel(A: fx.Tensor, B: fx.Tensor):
        batch = fx.block_idx.x
        tid = fx.thread_idx.x
        A = A[batch, None, None] # [M, N]
        B = B[batch, None]       # [N]
        # B = fx.raked_product(B, fx.make_layout(1,0)) # Unsqueeze an extra size-1 dimension
        copy_bits = 128
        copy_atom = fx.make_copy_atom(fx.UniversalCopy(copy_bits), A.dtype)
        VECT_WIDTH = fxu.div_e(copy_bits, A.dtype.width)
        tv_layout, tv_tilemn = fxu.make_mem_coalescing_2d_tv_layout(num_threads, VECT_WIDTH)
        assert tv_tilemn[0] == 1, "sum along an axis requires tileM=1"
        print(f"tv_layout = {tv_layout} tv_tilemn={tv_tilemn}")
        tiled_copyA = fx.make_tiled_copy(copy_atom, tv_layout, tv_tilemn)
        tiled_copyB = fx.make_tiled_copy(copy_atom, tv_layout, (tv_tilemn[1],))
        part_A = tiled_copyA.get_slice(tid).partition_S(A)
        part_B = tiled_copyB.get_slice(tid).partition_D(B)
        print("part_A: ", part_A)
        print("part_B: ", part_B)
        num_iters_m = fx.size(part_A.layout[1]).get_static_leaf_int
        num_iters_n = fx.size(part_A.layout[2]).get_static_leaf_int
        for n in fx.range_constexpr(num_iters_n):
            frag = fx.make_fragment_like(part_A[None, None, n])
            fx.copy(copy_atom, part_A[None, None, n], frag)

            vec_sum = frag[None, 0].load().to(fx.Float32)
            for m in fx.range_constexpr(1, num_iters_m):
                vec = frag[None, m].load().to(fx.Float32)
                vec_sum += vec

            # store out
            vec_sum = vec_sum.to(part_B.dtype)
            frag = fx.make_fragment_like(part_B[None, n])
            frag.store(vec_sum)
            fx.copy(copy_atom, frag, part_B[None, n])

    @flyc.jit
    def sum(A: fx.Tensor, B: fx.Tensor, stream):
        assert A.dtype == B.dtype
        assert A.leading_dim == 2, "kernel assumes 2nd mode is contiguous"
        batch_size = fx.get_scalar(A.shape[0])
        fake_static_bs = (1<<31)-1
        A = fx.Tensor(fx.make_view(fx.get_iter(A), fx.make_ordered_layout((fake_static_bs, M, N), (2,1,0))))
        B = fx.Tensor(fx.make_view(fx.get_iter(B), fx.make_ordered_layout((fake_static_bs, N), (1,0))))
        print(A)
        print(B)
        kernel(A, B).launch(
            grid=(batch_size, 1, 1),
            block=(num_threads, 1, 1), stream=stream
        )
    return sum

def test_sum_dim(L, M, N, dtype):
    sum = compile_sum_dim(L, M, N)

    A = torch.randn(L, M, N, dtype=dtype, device="cuda")
    B = torch.zeros(L, N, dtype=dtype, device="cuda")
    _, us = pyhip.run_perftest(sum, A, B, stream, num_verbose=1, num_iters=10, num_warmup=2, num_name="flydsl",
                               num_bytes=A.numel()*A.element_size() + B.numel()*B.element_size())
    ref = A.sum(1)
    ret = B
    assert torch.allclose(ref, ret, atol=1e-3), f"A.sum(1)={ref}  B = {ret}"

    # compare with pyhip-jit
    from pyhip.contrib.moe_gemm_mxfp4 import moe_gemm_final_reduce_bf16

    num_tokens_total = L
    num_CU = 80
    num_WG = num_CU * 4
    num_tokens_wg = num_tokens_total // num_WG
    num_extra_tokens = num_tokens_total % num_WG
    A = torch.randn(num_tokens_total, M, N, dtype=torch.bfloat16, device="cuda")
    B = torch.zeros(num_tokens_total, N, dtype=torch.bfloat16, device="cuda")
    _, us = pyhip.run_perftest(moe_gemm_final_reduce_bf16,[num_WG], [64], M, N,
                                A,
                                B,
                                num_tokens_wg, num_extra_tokens, num_tokens_total,
                                num_verbose=1, num_iters=10, num_warmup=2, num_name="pyhip-jit",
                                num_bytes=A.numel()*A.element_size() + B.numel()*B.element_size())

test_sum_dim(8192, 8, 4096, torch.bfloat16)
test_sum_dim(8192*4, 8, 4096, torch.bfloat16)
