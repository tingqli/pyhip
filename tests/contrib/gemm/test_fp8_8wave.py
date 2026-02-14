import torch
import torch.nn.functional as F
import pyhip
from pyhip.contrib.gemm_fp8 import *

torch.set_printoptions(linewidth=3000, sci_mode=False, edgeitems=8, )
torch.set_default_device('cuda')
torch.manual_seed(0)

def test(m, n, k):
    out_dtype = torch.bfloat16
    x = (torch.rand((m, k)) / 10).to(torch.float8_e4m3fn)
    w = (torch.rand((n, k)) / 10).to(torch.float8_e4m3fn)
    y0 = F.linear(x.to(out_dtype), w.to(out_dtype))
    y1 = torch.empy((m, n), dtype = out_dtype)
    wg_M, wg_N = 256, 256
    num_block_M = pyhip.div_up(m, wg_M)
    num_block_N = pyhip.div_up(n, wg_N)

    gemm_fp8_8wave([num_block_N, num_block_M],[64*8],
                   wg_M, wg_N, x.data_ptr(), w.data_ptr(), y1.data_ptr(), m)

    print(y0)
    print(y1)

'''
    HipKittens/kernels/gemm/fp8fp32/FP8_8wave# ./tk_kernel 
    Matrix dimensions: 8192x8192x8192, CUs: 256
    Warmup iterations: 500, Timing iterations: 100
    Optimized kernel (matmul_device):
    Kernel time (best): 0.418 ms,  TFLOPS: 2630.16
    Kernel time (avg ): 0.441 ms,  TFLOPS: 2494.26
'''

M,N,K = 8192,8192,8192
test(M,N,K)
