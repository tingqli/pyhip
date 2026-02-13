import pyhip
from pyhip.contrib.gemm_cdna4 import *
import pytest
import functools
import torch


torch.set_printoptions(linewidth=3000, sci_mode=False, edgeitems=8, )
torch.set_default_device('cuda')
torch.manual_seed(0)

@pytest.mark.parametrize("M", [32, 256, 2400])
@pytest.mark.parametrize("N", [256, 256*6])
@pytest.mark.parametrize("K", [256])
def test_accuracy(M, N, K, use_pre_shuffle = 0):
    wg_M = 256
    wg_N = 256
    blk_cnt = pyhip.div_up(M, wg_M) * pyhip.div_up(N, wg_N)

    A0 = torch.randn(M, K).to(dtype=torch.bfloat16)
    B0 = torch.randn(N, K).to(dtype=torch.bfloat16)
    ref_out = A0 @ B0.t()

    if use_pre_shuffle:
        A = pre_shuffle(A0, 16)
        B = pre_shuffle(B0, 16)
    else:
        A = A0
        B = B0

    cur_out = torch.ones(M, N, dtype=torch.bfloat16)

    gemm_kernel([blk_cnt],[4*64], wg_M, wg_N, N, K, use_pre_shuffle,
                A.data_ptr(), B.data_ptr(), cur_out.data_ptr(), M)
    
    if not torch.allclose(ref_out, cur_out, rtol=0.01, atol=0.01):
        print(ref_out)
        print(cur_out)
        print(cur_out[0].tolist())
        assert 0


def compare_perf(M, N, K, use_pre_shuffle = 0):
    wg_M = 256
    wg_N = 256

    blk_cnt = (M // wg_M) * (N // wg_N)

    A0 = torch.randn(M, K).to(dtype=torch.bfloat16)
    B0 = torch.randn(N, K).to(dtype=torch.bfloat16)
    C0 = A0 @ B0.t()

    if use_pre_shuffle:
        A = pre_shuffle(A0, 16)
        B = pre_shuffle(B0, 16)
    else:
        A = A0
        B = B0

    C = torch.zeros(M, N, dtype=torch.bfloat16)

    gemm_kernel([blk_cnt],[4*64], wg_M, wg_N, N, K, use_pre_shuffle, A.data_ptr(), B.data_ptr(), C.data_ptr(), M)


    ref_out = C0
    cur_out = C
    acc = "pass"
    if not torch.allclose(ref_out, cur_out, rtol=0.01, atol=0.01):
        print(f"================= ref_out : {ref_out.shape} ")
        print(ref_out)
        print(f"================= cur_out : {cur_out.shape} ")
        print(cur_out)
        idx = torch.where(torch.abs(ref_out - cur_out) > 0.03)
        if len(idx[0]):
            print(f'idx = {idx}\nref={ref_out[idx]}\ncur={cur_out[idx]}\n{len(idx[0])}')
        #assert 0
        acc = "failed"

    DATA_CLONES = 40
    As = [torch.clone(A) for _ in range(DATA_CLONES)]
    Bs = [torch.clone(B) for _ in range(DATA_CLONES)]
    Cs = [torch.clone(C) for _ in range(DATA_CLONES)]

    A0s = [torch.clone(A0) for _ in range(DATA_CLONES)]
    B0s = [torch.clone(B0) for _ in range(DATA_CLONES)]

    di = 0
    for i in range(10):
        di = (di + 1)%DATA_CLONES
        with pyhip.cudaPerf(M*N*K*2, (M*K*2+K*N*2), name=f"torch_{di}") as p0:
            ref = torch.nn.functional.linear(A0s[di], B0s[di])

    for i in range(10):
        di = (di + 1)%DATA_CLONES
        with pyhip.cudaPerf(M*N*K*2, (M*K*2+K*N*2), name=f"gemm_{di}") as p0:
            gemm_kernel([blk_cnt],[4*64], wg_M, wg_N, N, K, use_pre_shuffle,
                        As[di].data_ptr(),
                        Bs[di].data_ptr(),
                        Cs[di].data_ptr(), M)
    print(f"{acc=}")

if __name__ == "__main__":
    test_accuracy(2400, 256*4, 256*6)
    #test_accuracy(256, 256, 256)
    #compare_perf(M = 256*94, N = 256*16, K=8192)