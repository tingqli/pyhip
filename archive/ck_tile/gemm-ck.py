import torch
import pyhip
import sys

torch.cuda.set_device(1)
torch.set_default_device('cuda')
torch.manual_seed(0)
cur_gpu_device =torch.cuda.get_device_name()
print(f"{torch.get_default_device()=}")

hip = pyhip.module("gemm-ck.cpp")

num_CU = 80

CU_rows = int(num_CU**0.5)
CU_cols = num_CU//CU_rows
# CU_rows=10
# CU_cols=8
M = 256 * CU_rows*1
N = 256 * CU_cols*1
K = 8192

A = torch.randn(M, K, dtype=torch.float16)
B = torch.randn(N, K, dtype=torch.float16)
out = torch.randn(M, N, dtype=torch.float)

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

    gemm_tile_256x256x32([(N//256)*(M//256)],[256], A.data_ptr(), B.data_ptr(), out.data_ptr(), M, N, K)

    pass_flag = torch.allclose(ref, out, atol=0.1, rtol=0.1)
    if not pass_flag:
        print(ref)
        print(out)

for i in range(8):
    with pyhip.cudaPerf(M*N*K*2, (M*K*2+K*N*2), name="gemm_tile"):
        gemm_tile_256x256x32([(N//256)*(M//256)],[256], A.data_ptr(), B.data_ptr(), out.data_ptr(), M, N, K)

if len(sys.argv) == 1:
    print("PASS" if pass_flag else "FAILED")
