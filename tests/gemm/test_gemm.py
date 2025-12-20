import pyhip


import torch
torch.cuda.set_device(2)
torch.set_default_device('cuda')
torch.manual_seed(0)
torch.set_printoptions(linewidth=300)

cur_gpu_device =torch.cuda.get_device_name()
print(f"{torch.get_default_device()=} {torch.cuda.device_count()}")

#MFMA_F32_16x16x16_F16
MFMA_KL=4
#
REG_M=4
REG_N=2
#
WAVE_M = REG_M * 16
WAVE_N = REG_N *16
REG_K=2
BK = REG_K * 16
#
M_WAVES=2
N_WAVES=2
#
BM=M_WAVES*WAVE_M
BN=N_WAVES*WAVE_N

M_WGS=10
N_WGS=8
K_BLKS=1

M=BM*M_WGS
N=BN*N_WGS
K=BK*K_BLKS

MIN=-1
MAX=2

A = torch.randint(MIN,MAX,(M, K)).to(dtype=torch.float16)
B = torch.randint(MIN,MAX,(N, K)).to(dtype=torch.float16)
# A=torch.ones(M, K).to(torch.float16)
# B=torch.ones(N, K).to(torch.float16)
out=torch.zeros(M,N).to(torch.float32)
out_ref=torch.zeros(M,N).to(torch.float16)
B_tr=B.transpose(0,1)
print(f'/////////////////////////////////////////////////////////////////////////////////////////')
print(f'{M=}, {N=}, {K=}, {BM=},{BN=},{REG_M=}, {REG_N=}, GRID_DIM:[{M_WGS},{N_WGS}], BLOCK_DIM[{64*M_WAVES*N_WAVES}]')
print(f'////////////////////////////////////////////////////////////////////////////////////////')

hip = pyhip.module("gemm.cpp", f"-D{BM=} -D{BN=} -D{BK=} -D{WAVE_M=} -D{WAVE_N=}  -D{REG_M=} -D{REG_N=} -D{REG_K=} -D{N_WAVES=}")
run_mfma = hip.run_mfma
for i in range(2):
    with pyhip.cudaPerf(M*N*K*2):
        run_mfma([M_WGS, N_WGS],[64*M_WAVES*N_WAVES], A.data_ptr(), B.data_ptr(), out.data_ptr(), K, N)
out_ref=torch.matmul(A,B_tr)
# print(f'ref:{out_ref}')
# print(f'out:{out}')
assert torch.allclose(out_ref.to(torch.float32), out, atol=0.1, rtol=0.1)

