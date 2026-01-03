import pyhip
hip = pyhip.module("bench-mfma.cpp")

import torch
torch.cuda.set_device(2)
torch.set_default_device('cuda')
torch.manual_seed(0)
cur_gpu_device =torch.cuda.get_device_name()
print(f"{torch.get_default_device()=} {torch.cuda.device_count()}")

M = 32*4
N = 32*4
K = 8192

# peak 233 

A = torch.randn(M, 32, dtype=torch.float16)
B = torch.randn(N, 32, dtype=torch.float16)
out = torch.randn(M, N, dtype=torch.float)

def test_torch_fp16_perf(num_CU = 80, K = 8192):
    CU_rows = int(num_CU**0.5)
    CU_cols = num_CU//CU_rows
    M = 32 * 4 * CU_cols*2
    N = 32 * 4 * CU_rows*2
    A = torch.randn(M, K, dtype=torch.float16)
    B = torch.randn(N, K, dtype=torch.float16)
    for i in range(4):
        with pyhip.cudaPerf(M*N*K*2):
            out = torch.nn.functional.linear(A, B)

num_CU = 80
test_torch_fp16_perf(num_CU, K)

# if 1:
#     bench_mfma = hip.bench_mfma
#     for i in range(4):
#         with pyhip.cudaPerf(num_CU*4*M*N*K*2):
#             bench_mfma([num_CU],[64*4], A.data_ptr(), B.data_ptr(), 32, out.data_ptr(), N, K)
