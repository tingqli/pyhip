import pyhip


import torch
torch.cuda.set_device(2)
torch.set_default_device('cuda')
torch.manual_seed(0)
torch.set_printoptions(linewidth=500)

cur_gpu_device =torch.cuda.get_device_name()
print(f"{torch.get_default_device()=} {torch.cuda.device_count()}")

#MFMA_F32_32x32x8_F16
MFMA_K=8
MFMA_KL=4
REG_KL=MFMA_KL//2
#REG_M, REG_N means how many M, N per lane for A, B.
#REG_KL and REG_K is a little tricky. 
#       REG_KL means one MFMA need how many registers for A, B per lane. For 32x32x8, KL=4. So for FP16 datatype, REG_KL=2
#       REG_K means the number of BK/MFMA_K
#       Each lane in A needs (REG_M * REG_K * REG_KL) VGPRs. Each lane in B needs (REG_N * REG_K * REG_K)
REG_M=4
REG_N=4
REG_KL=MFMA_KL//2
REG_K=4
assert REG_K % 2 == 0, f'REG_K % 2 == 0 to ensure 128 bit reading'


WAVE_M = REG_M*32
WAVE_N = REG_N*32
BK = REG_K*MFMA_K
#
M_WAVES=2
N_WAVES=2
#
BM=M_WAVES*WAVE_M
BN=N_WAVES*WAVE_N

M_WGS=1
N_WGS=1
K_BLKS=4

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

hip = pyhip.module("gemm.cpp", f"-D{BM=} -D{BN=} -D{BK=} -D{WAVE_M=} -D{WAVE_N=}  -D{REG_M=} -D{REG_N=} -D{REG_K=} -D{N_WAVES=}  -D{MFMA_KL=}")
run_mfma = hip.run_mfma
for i in range(2):
    with pyhip.cudaPerf(M*N*K*2):
        run_mfma([M_WGS, N_WGS],[64*M_WAVES*N_WAVES], A.data_ptr(), B.data_ptr(), out.data_ptr(), K, N)
out_ref=torch.matmul(A,B_tr)
# print(A[0:32])
# print("-------------------------")
# print(A[32:64])
# print(f'ref:{out_ref[32:48]}')
# print(f'out:{out[32:48]}')
assert torch.allclose(out_ref.to(torch.float32), out, atol=0.1, rtol=0.1)

