import pyhip
from pyhip.contrib.gemm_a4w4 import *
from pyhip import calc_diff

import pytest
import torch
import aiter
from aiter import dtypes
from aiter.utility import fp4_utils
from aiter.ops.shuffle import shuffle_weight

torch.set_printoptions(linewidth=3000, sci_mode=False, edgeitems=8, )
torch.set_default_device('cuda')
torch.manual_seed(0)

# https://github.com/ROCm/aiter/tree/main/csrc/ck_gemm_a4w4_blockscale
# AITER_REBUILD=1
wg_M = 256
wg_N = 256

def _run_gemm_a4w4(
                x: "Tensor",  # A:[M, K] bf16
                weight: "Tensor",  # B:[N, K/2] f4x2
                weight_scale: "Tensor",  # B_scale:[N, K/32] e8m0 paded
                use_jit
            ):
    M = x.shape[0]
    N = weight.shape[0]
    K = weight.shape[1]
    # use hip quant kernel for performance
    x_q, x_s = aiter.per_1x32_f4_quant_hip(x, shuffle=True)

    #print(f"{x.shape}{x.dtype},  {x_q.shape}{x_q.dtype},    {x_s.shape}{x_s.dtype} ")
    #print(f"{weight.shape}{weight.dtype},   {weight_scale.shape}{weight_scale.dtype}")
    #assert 0
    # 32 alignment is enough for dim0 padding of output for
    # gemm_a4w4 kernel
    y = torch.empty(
        (M + 31) // 32 * 32,
        weight.shape[0],
        device=x_q.device,
        dtype=x.dtype,
    )
    if use_jit:
        blk_cnt = ((M + wg_M - 1) // wg_M) * ((N + wg_N - 1) // wg_N)
        gemm_a4w4_kernel(
            [blk_cnt], [256],
            wg_M, wg_N, N, K, False, True,
            x_q.data_ptr(), x_s.data_ptr(),
            weight.data_ptr(), weight_scale.data_ptr(),
            y.data_ptr(), M)
    else:
        aiter.gemm_a4w4(x_q, weight, x_s, weight_scale.view(x_s.dtype), y, bpreshuffle=True)
    return y[:M]

def generate_mxfp4_weight(N,K,BUF_COPY=None):
    weight_type = torch.float4_e2m1fn_x2
    w_ = torch.randint(-2, 3, [N, K], dtype=torch.bfloat16)
    w_qt, w_qt_scale_ = aiter.get_torch_quant(aiter.QuantType.per_1x32)(w_, quant_dtype=weight_type)
    w_f32 = fp4_utils.mxfp4_to_f32(w_qt).to(dtype=torch.bfloat16).reshape(N, K // 32, 32)
    w_scale_f32 = fp4_utils.e8m0_to_f32(w_qt_scale_).to(dtype=torch.bfloat16).reshape(N, K // 32, 1)
    w_ref = (w_f32 * w_scale_f32).reshape(N, K)

    assert K % 256 == 0, f'e8m0_shuffle assume there will be 8 groups of 32 elements in K dim, current K={K} is not supported'
    w_qt_scale = fp4_utils.e8m0_shuffle(w_qt_scale_)
    w_qt = shuffle_weight(w_qt)
    if BUF_COPY is not None:
        w = [w_qt.clone() for _ in range(BUF_COPY)]
        w_scale = [w_qt_scale.clone() for _ in range(BUF_COPY)]
    else:
        w = w_qt
        w_scale = w_qt_scale
    return w, w_scale, w_ref

@pytest.mark.parametrize("M", [32, 256, 2400])
@pytest.mark.parametrize("N", [256, 256*6])
@pytest.mark.parametrize("K", [256])
def test_accuracy(M, N, K):
    w, w_scale, w_ref = generate_mxfp4_weight(N, K)
    A = torch.randn([M, K], dtype=torch.bfloat16)
    ref_out = A @ w_ref.t()
    jit_out = _run_gemm_a4w4(A, weight=w, weight_scale=w_scale, use_jit=True)
    diff = calc_diff(ref_out, jit_out)
    if diff > 0.01:
        print(diff)
        print(ref_out)
        print(jit_out)
        assert 0

def compare_perf(M,N,K):
    BUF_COPY = 32
    w, w_scale, w_ref = generate_mxfp4_weight(N, K, BUF_COPY)
    
    A = torch.randn([BUF_COPY, M, K], dtype=torch.bfloat16)
    ref_out = A[0] @ w_ref.t()

    aiter_out = _run_gemm_a4w4(A[0], weight=w[0], weight_scale=w_scale[0], use_jit=False)
    jit_out = _run_gemm_a4w4(A[0], weight=w[0], weight_scale=w_scale[0], use_jit=True)
    acc = []
    acc.append(f"{calc_diff(aiter_out, jit_out)=:.2f}")

    print(f"{A[0].shape} {A[0].dtype} x  {w[0].shape}  {w[0].dtype} , {w_scale[0].shape} {w_scale[0].dtype} =>  {jit_out.shape} {jit_out.dtype}")

    acc.append(f"{calc_diff(ref_out, aiter_out)=:.2f}")
    acc.append(f"{calc_diff(ref_out, jit_out)=:.2f}")

    if calc_diff(ref_out, jit_out) > 0.01:
        print(ref_out)
        print(y)

    flops = 2 * M * N * K
    mem_size = M * K * 2 + N * K * 0.5
    with pyhip.torchPerf():
        di = 0
        for i in range(10):
            with pyhip.cudaPerf(flops, mem_size, name="aiter"):
                _run_gemm_a4w4(A[di], weight=w[di], weight_scale=w_scale[di], use_jit=False)
                di += 1
        for i in range(10):
            with pyhip.cudaPerf(flops, mem_size, name="fp4gemm"):
                _run_gemm_a4w4(A[di], weight=w[di], weight_scale=w_scale[di], use_jit=True)
                di += 1

    print("\n".join(acc))

if __name__ == "__main__":
    #M,N,K = 24000,4096,8192
    #M,N,K = 24000,3072,4096
    #M,N,K = 24000,4096,1536
    test_accuracy(256, 256*4, 256*6)
    test_accuracy(24000, 4096, 8192)
    compare_perf(M=24000,N=4096,K=8192)