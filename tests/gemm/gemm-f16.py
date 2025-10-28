import torch
import pyhip
import sys

import torch
torch.cuda.set_device(7)
torch.set_default_device('cuda')
torch.manual_seed(0)
cur_gpu_device =torch.cuda.get_device_name()
num_CU = torch.cuda.get_device_properties().multi_processor_count
NUM_XCD = 8 if num_CU > 80 else 4
print(f"{torch.get_default_device()=} with {num_CU=} {NUM_XCD=}")
    
hip = pyhip.module("gemm-f16.cpp", f"-D {NUM_XCD=}")

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

CU_rows = 8 #best_rows
CU_cols = 10 #num_CU//CU_rows
M = 256 * CU_rows*1
N = 256 * CU_cols*1
K = 8192

print(f" {M}x{N}x{K}  CUs: {CU_rows} x {CU_cols} = {CU_rows*CU_cols} ")

StrideK = 128*1024
# buffer_load_dwordx4's performance suddenly drops (by-half) when stride > 256KB
# and hipblas's performance is better when StrideK=K*8 / StrideK=K*4
StrideK = K*1

A = torch.randn(M, StrideK, dtype=torch.float16)
B = torch.randn(N, StrideK, dtype=torch.float16)
out = torch.randn(M, N, dtype=torch.float)
blk_maps = torch.tensor([
0, 0, 0, 5, 4, 0, 4, 5, 
1, 0, 1, 5, 5, 0, 5, 5, 
2, 0, 2, 5, 6, 0, 6, 5, 
3, 0, 3, 5, 7, 0, 7, 5, 
0, 1, 0, 6, 4, 1, 4, 6, 
1, 1, 1, 6, 5, 1, 5, 6, 
2, 1, 2, 6, 6, 1, 6, 6, 
3, 1, 3, 6, 7, 1, 7, 6, 
0, 2, 0, 7, 4, 2, 4, 7, 
1, 2, 1, 7, 5, 2, 5, 7, 
2, 2, 2, 7, 6, 2, 6, 7, 
3, 2, 3, 7, 7, 2, 7, 7, 
0, 3, 0, 8, 4, 3, 4, 8, 
1, 3, 1, 8, 5, 3, 5, 8, 
2, 3, 2, 8, 6, 3, 6, 8, 
3, 3, 3, 8, 7, 3, 7, 8, 
0, 4, 0, 9, 4, 4, 4, 9, 
1, 4, 1, 9, 5, 4, 5, 9, 
2, 4, 2, 9, 6, 4, 6, 9, 
3, 4, 3, 9, 7, 4, 7, 9, 
], dtype=torch.int32)
gemm_tile_256x256x32 = hip.gemm_tile_256x256x32

if len(sys.argv) == 1:
    # https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.linear.html
    # torch.nn.functional.linear(input, weight, bias=None)
    #   input:  (*, input_featires)
    #   Weight: (output_features, input_featires)
    for i in range(8):
        with pyhip.cudaPerf(M*N*K*2, name="torch-linear"):
            ref = torch.nn.functional.linear(A[:,:K], B[:, :K])
    ref = ref.to(dtype=torch.float)

    gemm_tile_256x256x32([(N//256)*(M//256)],[256], A.data_ptr(), B.data_ptr(), StrideK, out.data_ptr(), N, K//32, blk_maps.data_ptr())

    pass_flag = torch.allclose(ref, out, atol=0.1, rtol=0.1)
    #if not pass_flag:
    #    print(ref)
    #    print(out)

for i in range(8):
    with pyhip.cudaPerf(M*N*K*2, (M*K*2+K*N*2), name="gemm_tile"):
        gemm_tile_256x256x32([(N//256)*(M//256)],[256], A.data_ptr(), B.data_ptr(), StrideK, out.data_ptr(), N, K//32, blk_maps.data_ptr())

if len(sys.argv) == 1:
    print("PASS" if pass_flag else "FAILED")
