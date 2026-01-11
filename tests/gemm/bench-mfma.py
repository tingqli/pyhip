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
K = 8192*4

# peak 233 

A = torch.randn(M, 32, dtype=torch.float16)
B = torch.randn(N, 32, dtype=torch.float16)
out = torch.randn(M, N, dtype=torch.float32)

def test_torch_fp16_perf(num_CU = 256, K = 8192):
    DATA_CLONES = 1
    CU_rows = int(num_CU**0.5)
    CU_cols = num_CU//CU_rows
    M = 32 * 4 * CU_cols*2
    N = 32 * 4 * CU_rows*2
    AA = torch.randn(M, K, dtype=torch.bfloat16)
    BB = torch.randn(N, K, dtype=torch.bfloat16)
    CC = torch.randn(M, N, dtype=torch.float32)
    
    As = [torch.clone(AA) for _ in range(DATA_CLONES)]
    Bs = [torch.clone(BB) for _ in range(DATA_CLONES)]
    Cs = [torch.clone(CC) for _ in range(DATA_CLONES)]
    for i in range(DATA_CLONES):
        with pyhip.cudaPerf(M*N*K*2):
            Cs[i] = torch.nn.functional.linear(As[i], Bs[i])

num_CU = 256*2
test_torch_fp16_perf(num_CU, K)

if 0:
    bench_mfma = hip.bench_mfma
    for i in range(100):
        with pyhip.cudaPerf(num_CU*4*M*N*K*2):
            bench_mfma([num_CU],[64*4], A.data_ptr(), B.data_ptr(), 32, out.data_ptr(), N, K)
