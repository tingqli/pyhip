import os
import flydsl
import pytest
import math
import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import fly

import pyhip
import pyhip.contrib.flydsl.utils as fxu

# fxu.enable_dump_ir(True)

from flydsl._mlir.dialects import llvm

_, stream = pyhip.set_device()

def test_sum_1d():
    num_waves = 1
    num_threads = num_waves * 64

    @flyc.kernel(known_block_size=[num_threads, 1, 1]) # known_block_size at compile time
    def sum_kernel(A: fx.Tensor, B: fx.Tensor):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x
        A = A[None, bid]
        

        copy_bits = 128
        copy_atom = fx.make_copy_atom(fx.UniversalCopy(copy_bits), A.dtype)
        VECT_WIDTH = fxu.div_e(copy_bits, A.dtype.width)

        tv_layout, tv_tilemn = fxu.make_mem_coalescing_2d_tv_layout(num_threads, VECT_WIDTH)

        tiled_copy = fx.make_tiled_copy(copy_atom, tv_layout, (tv_tilemn,))
        part_A = tiled_copy.get_slice(tid).partition_S(A)
        print("====", part_A)

        #frag = fx.make_fragment_like(part_A[None, m, n])
        num_iters = fx.size(part_A.layout[1]).get_static_leaf_int

        if fx.const_expr(0):
            vec_sum = fx.full(part_A[None, 0].shape.to_py_value(), 0.0, part_A.dtype)
            # unroll load & store, AMDGPU backend will interleaving instructions to hide latency for us
            """
            # this simple version also allows AMDGPU backend to interleaving instructions
            # thus performance is not bad
            for m in fx.range_constexpr(num_iters):
                frag = fx.make_fragment_like(part_A[None, m])
                fx.copy(copy_atom, part_A[None, m], frag)
                vec_sum += frag.load()
            """
            frags = []
            for m in fx.range_constexpr(num_iters):
                frags.append(fx.make_fragment_like(part_A[None, m]))
                fx.copy(copy_atom, part_A[None, m], frags[-1])
            frag_iter = iter(frags)
            for m in fx.range_constexpr(num_iters):
                vec = next(frag_iter).load()
                #print(frag, vec, type(vec), vec.shape)
                vec_sum += vec
        else:
            # loop version
            #shape = part_A[None, 0].shape.to_py_value()
            #vec_zeros = fx.full(shape, 0.0, part_A.dtype)
            
            frag = fx.make_fragment_like(part_A[None, 0])
            fx.copy(copy_atom, part_A[None, 0], frag)
            vec_sum = frag.load()

            print("vec_zeros: ", vec_sum)
            for m in range(1, num_iters):
                frag = fx.make_fragment_like(part_A[None, m])
                fx.copy(copy_atom, part_A[None, m], frag)
                vec = frag.load()
                #print("frag: ", frag, "  vec:", vec, "  state:", state[0])
                vec_sum += vec.reshape(vec_sum.shape)
                #vec_sum += vec
                #print(vec, state[0], sum)

        fm_fast = fx.arith.FastMathFlags.fast
        sum = vec_sum.reduce(fx.vector.ReductionOp.ADD, fastmath=fm_fast)
        WARP_SIZE = 64
        def wave_reduce_add(x):
            w = x
            for _sh_exp in fx.range_constexpr(int(math.log2(WARP_SIZE))):
                off = WARP_SIZE // (2 << _sh_exp)
                peer = w.shuffle_xor(off, WARP_SIZE)
                w = w.addf(peer, fastmath=fm_fast)
            return w

        sum = wave_reduce_add(sum)
        copy_atom = fx.make_copy_atom(fx.UniversalAtomicAdd(B.dtype), B.dtype)
        if tid % WARP_SIZE == 0:
            tensor_sum = fx.make_rmem_tensor(1, sum.dtype)
            tensor_sum[0] = sum
            fx.copy_atom_call(copy_atom, tensor_sum, B)
        """
        out_ptr = fx.get_iter(B)
        addr = fx.ptrtoint(out_ptr)
        llvm_out_ptr = fx.buffer_ops.create_llvm_ptr(addr)

        if tid % WARP_SIZE == 0:
            llvm.atomicrmw(
                llvm.AtomicBinOp.fadd,
                llvm_out_ptr,
                sum.ir_value(),
                llvm.AtomicOrdering.monotonic,
                syncscope="agent",
                alignment=(B.dtype.width//8)
            )
        """

    @flyc.jit
    def sum(A: fx.Tensor, B: fx.Tensor, stream):
        assert A.dtype == B.dtype
        assert A.leading_dim == 0 and A.layout.rank == 1, "kernel assumes 1st mode is contiguous"

        # unlike copy, sum has extra computational overhead which needs bigger loop body
        # Amortize the overhead outside the loop by increasing the loop count (or trip count).

        tile_size = (64*4)*256
        A = fx.zipped_divide(A, fx.make_tile(tile_size))
        #grid_n = fx.get_scalar(A.shape[1][1])
        num_blocks = fx.get_scalar(A.shape[1])
        sum_kernel(A, B).launch(
            grid=(num_blocks, 1, 1),
            block=(num_threads, 1, 1), stream=stream
        )
    num_blocks = 80*4
    count_per_wave = 1024*256
    count = num_blocks * count_per_wave

    _, stream = pyhip.set_device()
    A = torch.randn(count, dtype=torch.float32, device="cuda")
    B = torch.zeros(1, dtype=torch.float32, device="cuda")
    _, us = pyhip.run_perftest(sum, A, B, stream,
                               num_verbose=1, num_iters=10, num_warmup=4, num_bytes=A.numel()*A.element_size())
    ref = A.sum()
    ret = B[0]
    assert torch.allclose(ref, ret, atol=1e-1), f"A.sum()={ref}  B = {ret}"

# test_sum_1d()


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
