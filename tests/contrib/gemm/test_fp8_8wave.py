import torch
import torch.nn.functional as F
import pyhip
import pytest
from pyhip.contrib.gemm_fp8 import *


torch.set_printoptions(linewidth=3000, sci_mode=False, edgeitems=8, )
torch.set_default_device('cuda')
torch.manual_seed(0)

@pytest.mark.parametrize("M", [32, 256, 2400])
@pytest.mark.parametrize("N", [256, 256*6])
@pytest.mark.parametrize("K", [256])
@pytest.mark.parametrize("bpreshuffle", [True, False])
def test(M, N, K, bpreshuffle):
    out_dtype = torch.bfloat16
    x = (torch.randn((M, K))).to(torch.float8_e4m3fn)
    w = (torch.randn((N, K))).to(torch.float8_e4m3fn)

    if 0:
        print("================= x")
        #print(x.to(out_dtype)[0:16, :64])
        #print(x.to(out_dtype)[0:16, 64:128])
        #print(x.to(out_dtype)[128+0:128+16, :64])
        #print(x.to(out_dtype)[128+0:128+16, 64:128])
        print("================= w 0")
        print(w.to(out_dtype)[:16, :64])
        print(w.to(out_dtype)[:16, 64:64+64])
        print("================= w 1")
        print(w.to(out_dtype)[128+0:128+16, :64])
        print(w.to(out_dtype)[128+0:128+16, 64:64+64])

    y0 = F.linear(x.to(out_dtype), w.to(out_dtype))
    y1 = torch.empty((M, N), dtype = out_dtype)
    wg_M, wg_N = 256, 256
    num_block_M = pyhip.div_up(M, wg_M)
    num_block_N = pyhip.div_up(N, wg_N)

    if bpreshuffle:
        w = pyhip.pre_shuffle(w, mfma_MN=16)
        # w = shuffle_weight(w, layout=(16, 16))

    gemm_fp8_8wave([num_block_N*num_block_M],[64*8], bpreshuffle, False,
                   wg_M, wg_N, N, K, x.data_ptr(), w.data_ptr(), y1.data_ptr(),
                   None, None, M)

    torch.cuda.synchronize()

    diff_all = pyhip.calc_diff(y0, y1)
    if diff_all > 0.01:
        print(f"{diff_all=}")
        diff_map = torch.empty([M//16, N//16], dtype=torch.float)
        for y in range(0,M,16):
            for x in range(0,N,16):
                diff = pyhip.calc_diff(y0[y:y+16,x:x+16], y1[y:y+16,x:x+16])
                diff_map[y//16,x//16] = diff

        print("====================== diff_map")
        print(diff_map)
        print("====================== y0 vs y1")
        print(y0[:16,128+0:128+16])
        print(y1[:16,128+0:128+16])
        for i in range(M):
            diff = pyhip.calc_diff(y0[i], y1[i])
            if diff > 0.01:
                print("======================", i, diff)
                print(y0[i].view(-1,8))
                print(y1[i].view(-1,8))
                assert 0

    BUF_COPY = 64
    As = [x.clone() for _ in range(BUF_COPY)]
    Bs = [w.clone() for _ in range(BUF_COPY)]
    Cs = [torch.empty((M, N), dtype = out_dtype) for _ in range(BUF_COPY)]
    di = 0
    for i in range(32):
        with pyhip.cudaPerf(M*N*K*2, name=f"gemm_fp8_8wave-{M}_{N}_{K}_{bpreshuffle=}"):
            gemm_fp8_8wave([num_block_N*num_block_M],[64*8], bpreshuffle, False,
                           wg_M, wg_N, N, K,
                           As[di].data_ptr(), Bs[di].data_ptr(), Cs[di].data_ptr(),
                           None, None, M)
        di = (di + 1)%BUF_COPY
    
    print(f"{diff_all=:.2f}")
'''
    HipKittens/kernels/gemm/fp8fp32/FP8_8wave# ./tk_kernel 
    Matrix dimensions: 8192x8192x8192, CUs: 256
    Warmup iterations: 500, Timing iterations: 100
    Optimized kernel (matmul_device):
    Kernel time (best): 0.418 ms,  TFLOPS: 2630.16
    Kernel time (avg ): 0.441 ms,  TFLOPS: 2494.26
'''

if __name__ == "__main__":
    M,N,K = 8192,8192,8192
    #M,N,K = 512,512,8192
    test(M=256,N=256,K=256, bpreshuffle = False)
    test(M=256,N=256,K=256, bpreshuffle = True)

    test(M=8192,N=8192,K=8192, bpreshuffle = False)
    test(M=8192,N=8192,K=8192, bpreshuffle = True)