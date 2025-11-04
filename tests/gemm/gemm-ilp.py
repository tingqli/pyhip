import torch
import pyhip
import sys

import torch
torch.set_printoptions(linewidth=300)
torch.cuda.set_device(2)
torch.set_default_device('cuda')
torch.manual_seed(0)
cur_gpu_device =torch.cuda.get_device_name()
num_CU = torch.cuda.get_device_properties().multi_processor_count
NUM_XCD = 8 if num_CU > 80 else 4
print(f"{torch.get_default_device()=} with {num_CU=} {NUM_XCD=}")

hip = pyhip.module("gemm-ilp.cpp", f"-D {NUM_XCD=}")

# find the square-like gemm shape that use as much CUs as possible
min_gap = num_CU
best_rows = -1
for CU_rows in range(int(num_CU**0.5), 2, -1):
    CU_cols = num_CU//CU_rows
    gap = num_CU - CU_rows * CU_cols
    if gap < min_gap:
        min_gap = gap
        best_rows = CU_rows
        if gap == 0:
            break

BLK_M, BLK_N, BLK_K = 256, 256, 32
#BLK_M, BLK_N, BLK_K = 128, 128, 64

CU_rows = 8 #best_rows
CU_cols = 10 #num_CU//CU_rows
M = BLK_M * CU_rows
N = BLK_N * CU_cols
K = 8192

print(f" {M}x{N}x{K}  CUs: {CU_rows} x {CU_cols} = {CU_rows*CU_cols} ")

StrideK = 128*1024
# buffer_load_dwordx4's performance suddenly drops (by-half) when stride > 256KB
# and hipblas's performance is better when StrideK=K*8 / StrideK=K*4
StrideK = K*1

A = torch.randn(M, StrideK, dtype=torch.float16)
B = torch.randn(N, StrideK, dtype=torch.float16)
out = torch.randn(M, N, dtype=torch.float)

gemm = hip.gemm

if len(sys.argv) == 1:
    # https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.linear.html
    # torch.nn.functional.linear(input, weight, bias=None)
    #   input:  (*, input_featires)
    #   Weight: (output_features, input_featires)
    for i in range(4):
        with pyhip.cudaPerf(M*N*K*2, name="torch-linear"):
            ref = torch.nn.functional.linear(A[:,:K], B[:, :K])
    ref = ref.to(dtype=torch.float)

    gemm([(N//BLK_N), (M//BLK_M)],[256], A.data_ptr(), B.data_ptr(), StrideK, out.data_ptr(), N, K)

    pass_flag = torch.allclose(ref, out, atol=0.1, rtol=0.1)
    if not pass_flag:
        print(ref[:,0])
        print(out[:,0])

for i in range(4):
    with pyhip.cudaPerf(M*N*K*2, (M*K*2+K*N*2), name="gemm_tile"):
        gemm([(N//BLK_N), (M//BLK_M)],[256], A.data_ptr(), B.data_ptr(), StrideK, out.data_ptr(), N, K)

if len(sys.argv) == 1:
    print("PASS" if pass_flag else "FAILED")
