import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import fly

import pyhip
import pyhip.contrib.flydsl.helpers as fxh

#fxh.dump_ir(True)

_, stream = pyhip.set_device()

# sum along an axis
# a.sum(dim=1) [B, M, N] -> [B, K]

def compile_sum_dim(TOPK, N):
    num_threads = 64
    @flyc.kernel(known_block_size=[num_threads, 1, 1]) # known_block_size at compile time
    def kernel(A: fx.Tensor, B: fx.Tensor):
        batch = fx.block_idx.x
        tid = fx.thread_idx.x
        copy_bits = 128
        copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy(copy_bits), A.dtype)
      
        A = fx.make_view(fx.get_iter(A) + fx.Int64(batch) * (TOPK * N), 
                         fx.make_layout((N, TOPK), (1, N)))
        B = fx.make_view(fx.get_iter(B) + fx.Int64(batch) * (N), 
                         fx.make_layout(N, 1))

        A = fx.rocdl.make_buffer_tensor(
            A,
            max_size=False,
            num_records_bytes=fx.Int64(N) * (TOPK * A.dtype.width // 8),
        )
        B = fx.rocdl.make_buffer_tensor(
            B,
            max_size=False,
            num_records_bytes=fx.Int64(N) * (B.dtype.width // 8),
        )

        # all_copy_atoms only partions the first mode, extra modes are considered as batch/broadcast dimension
        for dst, src in fxh.all_copy_atoms(B, A, atom_bits=copy_bits, num_threads=num_threads):
            frag = fx.make_fragment_like(src)
            fx.copy(copy_atom, src, frag)

            vec_sum = frag[None, 0].load().to(fx.Float32)
            for m in fx.range_constexpr(1, TOPK):
                vec = frag[None, m].load().to(fx.Float32)
                vec_sum += vec

            # store out
            vec_sum = vec_sum.to(dst.dtype)
            frag = fx.make_fragment_like(dst)
            frag.store(vec_sum)
            fx.copy(copy_atom, frag, dst)


    @flyc.jit
    def sum(A: fx.Tensor, B: fx.Tensor, stream):
        assert A.dtype == B.dtype
        assert A.leading_dim == 2, "kernel assumes 2nd mode is contiguous"
        batch_size = fx.get_scalar(A.shape[0])
        kernel(A, B).launch(
            grid=(batch_size, 1, 1),
            block=(num_threads, 1, 1), stream=stream
        )
    return sum

def test_sum_dim(L, M, N, dtype):
    sum = compile_sum_dim(M, N)

    A = torch.randn(L, M, N, dtype=dtype, device="cuda")
    B = torch.zeros(L, N, dtype=dtype, device="cuda")
    _, us = pyhip.run_perftest(sum, A, B, stream, num_verbose=1, num_iters=10, num_warmup=2, num_name="flydsl",
                               num_bytes=A.numel()*A.element_size() + B.numel()*B.element_size())
    ref = A.sum(1)
    ret = B
    assert pyhip.allclose(ref, ret, atol=1e-1)
    # assert torch.allclose(ref, ret, atol=1e-3), f"A.sum(1)={ref}\nB = {ret}"

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

#test_sum_dim(8192, 8, 4096, torch.bfloat16)
#test_sum_dim(8192*4, 8, 4096, torch.bfloat16)
test_sum_dim(131072, 8, 4096, torch.bfloat16)
