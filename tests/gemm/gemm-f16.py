import pyhip


hip = pyhip.module("gemm-f16.cpp")

import torch
torch.cuda.set_device(7)
torch.set_default_device('cuda')
torch.manual_seed(0)
cur_gpu_device =torch.cuda.get_device_name()
print(f"{torch.get_default_device()=}")

M = 32*4*2
N = 32*4*2
K = 8*4*256*16

A = torch.randn(M, K, dtype=torch.float16)
B = torch.randn(N, K, dtype=torch.float16)
out = torch.randn(M, N, dtype=torch.float)

gemm_tile_256x256x32 = hip.gemm_tile_256x256x32

num_CU = 80
for i in range(2):
    with pyhip.cudaPerf(M*N*K*2*num_CU):
        gemm_tile_256x256x32([num_CU],[256], A.data_ptr(), B.data_ptr(), K, out.data_ptr(), N, K//32)

if 0:
    ref = A.to(dtype=torch.float) @ B.transpose(1,0).to(dtype=torch.float)
    if torch.allclose(ref, out, atol=0.01, rtol=0.01):
        print("PASS")
    else:
        print(ref)
        print(out)