from itertools import accumulate
import pyhip
import pytest
import functools

# v_mfma_f32_32x32x8_bf16 when use950=False
# v_mfma_f32_32x32x16_bf16 when use950=True
# def test_mfma_bf16_tput(wgs_M, wgs_N, K, use950=False):
#     MFA_MN=32
#     MFA_K=16
#     regM=4
#     regN=4
#     regK=2
#     M_WAVES=2
#     N_WAVES=2
#     @pyhip.jit(with_debug_log=False, force_recompile=True)
#     def mfma_tput_kernel(J,  K):

#         ABRegSize = (MFA_MN*MFA_K*2//4)//64
#         CRegSize = (MFA_MN*MFA_MN)//64

#         bK = regK * MFA_K
#         cur_k = J.gpr("su32")
#         cur_k[0] = 0
#         k_loop_cnt = K//bK

#         mfma_A = J.gpr(regM, regK, ABRegSize, "vbf16x2")
#         mfma_B = J.gpr(regN, regK, ABRegSize, "vbf16x2")
#         mfma_C = J.gpr(regM, regN, CRegSize, "af32")
#         with J.While(cur_k[0] < k_loop_cnt):
#             for k in range(0,regK):
#                 for m in range(regM):
#                     for n in range(regN):
#                         if (use950):
#                             getattr(J, "v_mfma_f32_32x32x16_bf16")(mfma_C[m,n],
#                                                 mfma_B[n, k, 0:3],
#                                                 mfma_A[m, k, 0:3],
#                                                 mfma_C[m,n])
#                         else:
#                             getattr(J, "v_mfma_f32_32x32x8_bf16")(mfma_C[m,n],
#                                                 mfma_B[n, k, 0:1],
#                                                 mfma_A[m, k, 0:1],
#                                                 mfma_C[m,n])
#                             getattr(J, "v_mfma_f32_32x32x8_bf16")(mfma_C[m,n],
#                                     mfma_B[n, k, 2:3],
#                                     mfma_A[m, k, 2:3],
#                                     mfma_C[m,n])
#             cur_k[0] = cur_k[0] + 1           
#         # 4 Dwords, 8 bf16, MFA inner loop for 32x32x16
        
#     M = wgs_M * MFA_MN *regM * M_WAVES
#     N = wgs_N * MFA_MN *regN * N_WAVES
    
#     for i in range(10):
#         with pyhip.cudaPerf(M*N*K*2, (M*K*2+K*N*2), name=f"gemm_{M}x{N}x{K}_bf16") as p0:
#             mfma_tput_kernel([wgs_M*wgs_N],[64*M_WAVES*N_WAVES], K)
# # v_mfma_f32_32x32x16_bf8_bf8 when use950=False
# # v_mfma_f32_32x32x64_f8f6f4 when use950=True
# def test_950_mfma_fp8_tput(wgs_M, wgs_N, K, use950=False):
#     MFA_MN=32
#     MFA_K=64
#     regM=4
#     regN=4
#     regK=2
#     M_WAVES=2
#     N_WAVES=2
#     @pyhip.jit(with_debug_log=False, force_recompile=True)
#     def mfma_tput_kernel(J, K):

#         ABRegSize = (MFA_MN*MFA_K//4)//64
#         CRegSize = (MFA_MN*MFA_MN)//64

#         bK = regK * MFA_K
#         cur_k = J.gpr("su32")
#         cur_k[0] = 0
#         k_loop_cnt = K//bK

#         mfma_A = J.gpr(regM, regK, ABRegSize, "vbf16x2")
#         mfma_B = J.gpr(regN, regK, ABRegSize, "vbf16x2")
#         mfma_C = J.gpr(regM, regN, CRegSize, "af32")
#         with J.While(cur_k[0] < k_loop_cnt):
#             for k in range(0,regK):
#                 for m in range(regM):
#                     for n in range(regN):
#                         if use950:
#                             #v_mfma_f32_32x32x64_f8f6f4 v[0:15], v[4:11], v[4:11], v[0:15]
#                             getattr(J, "v_mfma_f32_32x32x64_f8f6f4")(mfma_C[m,n],
#                                                 mfma_B[n, k],
#                                                 mfma_A[m, k],
#                                                 mfma_C[m,n])
#                         else:              
#                             getattr(J, "v_mfma_f32_32x32x16_bf8_bf8")(mfma_C[m,n],
#                                                 mfma_B[n, k, 0:1],
#                                                 mfma_A[m, k, 0:1],
#                                                 mfma_C[m,n])
#                             getattr(J, "v_mfma_f32_32x32x16_bf8_bf8")(mfma_C[m,n],
#                                     mfma_B[n, k, 2:3],
#                                     mfma_A[m, k, 2:3],
#                                     mfma_C[m,n])
#                             getattr(J, "v_mfma_f32_32x32x16_bf8_bf8")(mfma_C[m,n],
#                                     mfma_B[n, k, 4:5],
#                                     mfma_A[m, k, 4:5],
#                                     mfma_C[m,n])
#                             getattr(J, "v_mfma_f32_32x32x16_bf8_bf8")(mfma_C[m,n],
#                                     mfma_B[n, k, 6:7],
#                                     mfma_A[m, k, 6:7],
#                                     mfma_C[m,n])
#             cur_k[0] = cur_k[0] + 1           
#         # 4 Dwords, 8 bf16, MFA inner loop for 32x32x16
        
#     M = wgs_M * MFA_MN *regM *M_WAVES
#     N = wgs_N * MFA_MN *regN *N_WAVES
    
#     for i in range(10):
#         with pyhip.cudaPerf(M*N*K*2, (M*K+K*N), name=f"950gemm_{M}x{N}x{K}_fp8") as p0:
#             mfma_tput_kernel([wgs_M*wgs_N],[64*M_WAVES*N_WAVES], K)

# test_mfma_bf16_tput(32, 32, 8192*4)
# test_mfma_bf16_tput(32, 32, 8192*4, use950=True)
# test_950_mfma_fp8_tput(32, 32, 8192)
# test_950_mfma_fp8_tput(32, 32, 8192, use950=True)
         
import torch
torch.set_printoptions(linewidth=3000, sci_mode=False, edgeitems=8, )
torch.cuda.set_device(2)
torch.set_default_device('cuda')
torch.manual_seed(0)


def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def align(x: int, y: int) -> int:
    return ceil_div(x, y) * y


def ceil_to_ue8m0(x: torch.Tensor):
    assert x.view(-1).amax().item() > 0
    return torch.pow(2.0, torch.ceil(torch.log2(x.abs())))


def per_token_cast_to_fp8(x: torch.Tensor, use_ue8m0: bool = True, block_size: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    m, n = x.shape
    padded_n = align(n, block_size)
    x_padded = torch.empty((m, padded_n), dtype=x.dtype).fill_(0)
    x_padded[:, :n] = x
    x_view = x_padded.view(m, -1, block_size)
    x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
    sf = x_amax / 448.0
    sf = ceil_to_ue8m0(sf) if use_ue8m0 else sf
    return (x_view * (1.0 / sf.unsqueeze(2))).to(torch.float8_e4m3fn).view(m, padded_n)[:, :n].contiguous(), sf

def generate_normal(m: int, n: int, k: int, out_dtype: torch.dtype,
                    use_ue8m0: bool = True):
    a = torch.randn((m, k), dtype=torch.float16)
    b = torch.randn((n, k), type=torch.float16)
    c = torch.randn((m, n), type=torch.float)
    ref_c = a.float() @ b.float().t()
    a_fp8 = per_token_cast_to_fp8(a, use_ue8m0=use_ue8m0)
    b_fp8 = per_token_cast_to_fp8(b, use_ue8m0=use_ue8m0)
    a_fp8 = a_fp8
    b_fp8 = b_fp8
    return a_fp8, b_fp8, c, ref_c


def test_fp8_32x32x64():
    M=32
    N=32
    K=64
    INT_MAX=3
    INT_MIN=-1
    sizeof_DWORDX4 = 16
    sizeof_fp32 = 4
    exponent_bias=127
    # scale 0.25, 0.5, 1
    A_scale_e8m0 = torch.randint(126, 130, (M,2), dtype=torch.uint8)
    B_scale_e8m0 = A_scale_e8m0[torch.randperm(A_scale_e8m0.size(0))]
    A_scale_fp16 = torch.pow(2.0,(A_scale_e8m0.to(dtype=torch.float16)-127.0)).to(dtype=torch.float16).repeat_interleave(32, dim=-1)
    B_scale_fp16 = torch.pow(2.0,(B_scale_e8m0.to(dtype=torch.float16)-127.0)).to(dtype=torch.float16).repeat_interleave(32, dim=-1)

    # print(f'scale_e8:{scale_e8m0}')
    # print(f'scale_fp16:{scale_fp16}')


    A_fp16 = torch.randint(INT_MIN, INT_MAX, (M, K)).to(dtype=torch.float16)
    B_fp16 = torch.randint(INT_MIN, INT_MAX, (N, K)).to(dtype=torch.float16)
    A=A_fp16.to(dtype=torch.float8_e4m3fn)
    B=B_fp16.to(dtype=torch.float8_e4m3fn)
    A_fp16 *= A_scale_fp16
    B_fp16 *= B_scale_fp16
    
    out = torch.randn(M, N, dtype=torch.float)

    @pyhip.jit(with_debug_log=True, force_recompile=True)
    def mfma_fp8_kernel(J, pA:"void*", pB:"void*", pC:"float*", pscaleA:"void*", pscaleB:"void*"):

        J.debug_setup(J.blockIdx.x[0] == 0)

        AB_voffset0 = J.gpr((J.threadIdx.x // 32) * sizeof_DWORDX4 + \
                                 (J.threadIdx.x % 32) * 64)
        AB_voffset1 = J.gpr((J.threadIdx.x // 32 + 2) * sizeof_DWORDX4 + \
                                (J.threadIdx.x % 32) * 64)
        
        scale_voffset = J.gpr((J.threadIdx.x % 32 *2 + J.threadIdx.x // 32))


        soffsetA = J.gpr("su32")
        soffsetB = J.gpr("su32")
        vdbg = J.gpr("vu32")
        soffsetA[0] = 0
        soffsetB[0] = 0    
        buff_a = J.Buffer(pA, M*K)
        buff_b = J.Buffer(pB, N*K)
        buff_c = J.Buffer(pC, M * N * sizeof_fp32)
        scale_a = J.Buffer(pscaleA, M*2)
        scale_b = J.Buffer(pscaleB, M*2)


        mfma_A = J.gpr(2, 4,"vbf16x2")
        mfma_B = J.gpr(2, 4, "vbf16x2")
        scale = J.gpr(2, "vf32")

        # scale = J.new_gpr("v", 1, dtype="f32", align=1)

        buff_a.load_dwordx4(mfma_A[0], AB_voffset0, soffsetA)
        buff_a.load_dwordx4(mfma_A[1], AB_voffset1, soffsetA)
    
        buff_b.load_dwordx4(mfma_B[0], AB_voffset0, soffsetB)
        buff_b.load_dwordx4(mfma_B[1], AB_voffset1, soffsetB)
        
        scale_a.load_ubyte(scale[0], scale_voffset, 0)
        scale_b.load_ubyte(scale[1], scale_voffset, 0)

        J.s_waitcnt(mod=f"vmcnt(0)")
        mfma_C = J.gpr(16, "af32")
        mfma_C[:] = 0
        
        getattr(J, "v_mfma_scale_f32_32x32x64_f8f6f4")(mfma_C,
                            mfma_B,
                            mfma_A,
                            mfma_C, scale[1], scale[0])
        # J.debug_log(mfma_C, torch.float, "4h.2h.32v.4h")
        C_voffset = J.gpr((J.threadIdx.x // 32) * sizeof_DWORDX4 + \
        (J.threadIdx.x % 32) * 128)
        soffsetC = J.gpr("su32")

        for i in range(4):
            soffsetC[0]=i*32
            buff_c.store_dwordx4(mfma_C[i*4:(i*4+3)], C_voffset, soffsetC)


    mfma_fp8_kernel([1],[64], A.data_ptr(), B.data_ptr(), out.data_ptr(), A_scale_e8m0.data_ptr(), B_scale_e8m0.data_ptr())
    torch.cuda.synchronize()
    ref_out = (A_fp16 @ B_fp16.t()).to(dtype=torch.float)
    if not torch.allclose(ref_out, out, rtol=0.02, atol=0.02):
        print("====================ref_out")
        print(ref_out)
        print("====================cur_out")
        print(out)
        idx = torch.where(torch.abs(ref_out - out) > 1.5)
        if len(idx[0]):
            print(f'idx = {idx}\nref={ref_out[idx]}\ncur={out[idx]}\n{len(idx[0])}')
        #assert 0
        acc_flag = False
    else:
        print("!!!!!!!!!!!!!!!!!!!!!PASS")




test_fp8_32x32x64()