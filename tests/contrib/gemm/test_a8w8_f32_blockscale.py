import torch
import torch.nn.functional as F
from einops import rearrange
from einops import repeat as eirp
from typing_extensions import List

import aiter
from aiter import dtypes
from aiter.ops.shuffle import shuffle_weight
from aiter.test_common import benchmark, checkAllclose, perftest
try:
    from aiter.ops.triton.gluon.gemm_a8w8_blockscale import (
        gemm_a8w8_blockscale as gluon_gemm_a8w8_blockscale,
    )
except:
    gluon_gemm_a8w8_blockscale = None

import pyhip
from pyhip.contrib.gemm_fp8 import *

torch.set_printoptions(linewidth=3000, sci_mode=False, edgeitems=8, )
torch.set_default_device('cuda')
torch.manual_seed(0)

block_shape = (128, 128)

@perftest()
def run_torch(x, weight, x_scale, w_scale, dtype=dtypes.bf16):
    block_shape_n, block_shape_k = block_shape
    m, k = x.shape
    n = weight.shape[0]
    scale_n = (n + block_shape_n - 1) // block_shape_n
    scale_k = (k + block_shape_k - 1) // block_shape_k
    x = x.to(x_scale.dtype).view(
        m, k // block_shape[1], block_shape[1]
    ) * x_scale.unsqueeze(-1)
    x = x.view(m, k)

    w_scale = rearrange(
        w_scale.view(-1, 1)
        .repeat(1, block_shape_n * block_shape_k)
        .view(scale_n, scale_k, block_shape_n, block_shape_k),
        "num_blk_n num_blk_k blk_n blk_k -> (num_blk_n blk_n) (num_blk_k blk_k)",
    )
    w_scale = w_scale[:n, :k]
    weight = weight.to(w_scale.dtype) * w_scale

    out = F.linear(x.to(dtypes.fp32), weight.to(dtypes.fp32))
    return out.to(dtype)

@perftest()
def run_gemm_ck(x, weight, x_scale, w_scale, dtype=dtypes.bf16):
    return aiter.gemm_a8w8_blockscale(x, weight, x_scale, w_scale, dtype)

@perftest()
def run_gemm_bpreshuffle_ck(x, weightshuffle, x_scale, w_scale, dtype=dtypes.bf16):
    return aiter.gemm_a8w8_blockscale_bpreshuffle(
        x, weightshuffle, x_scale, w_scale, dtype
    )

@perftest()
def run_asm(x, weight, x_scale, w_scale, dtype=dtypes.bf16, kernel_name=None):
    m, k = x.shape
    n, _ = weight.shape
    out = torch.empty((m, n), dtype=dtype, device=x.device)
    return aiter.gemm_a8w8_blockscale_bpreshuffle_asm(x, weight, out, x_scale, w_scale)

def test_gemm(dtype, m, n, k, ck_preshuffle=True):
    ret = {}
    dim = (m, n, k)
    block_shape_n, block_shape_k = block_shape
    scale_m = m
    scale_n = (n + block_shape_n - 1) // block_shape_n
    scale_k = (k + block_shape_k - 1) // block_shape_k
    x = (torch.rand((m, k), dtype=dtypes.fp32, device="cuda") / 10).to(dtypes.fp8)
    weight = (torch.rand((n, k), dtype=dtypes.fp32, device="cuda") / 10).to(dtypes.fp8)
    x_scale = torch.rand([scale_m, scale_k], dtype=dtypes.fp32, device="cuda")
    w_scale = torch.rand([scale_n, scale_k], dtype=dtypes.fp32, device="cuda")

    a, avg_a = run_torch(x, weight, x_scale, w_scale, dtype)

    x_scale_t = x_scale.transpose(0, 1).contiguous().view(*x_scale.shape)
    gemm_x_scale = x_scale_t if ck_preshuffle else x_scale
    gemm_weight = shuffle_weight(weight, layout=(16, 16)) if ck_preshuffle else weight
    run_func = run_gemm_bpreshuffle_ck if ck_preshuffle else run_gemm_ck
    b, avg_b = run_func(x, gemm_weight, gemm_x_scale, w_scale, dtype)

    err_ck = checkAllclose(a, b, msg="ck")
    ret["ck us"] = avg_b
    ret["ck TFLOPS"] = m * n * k * 2 / avg_b / 1e6
    ret["ck TB/s"] = (x.nbytes + weight.nbytes) / avg_b / 1e6
    ret["ck err"] = err_ck

    tag = "asm"
    weight_asm = shuffle_weight(weight, layout=(32, 16))
    # kernel_name = "_ZN5aiter43fp8gemm_bf16_blockscale_BpreShuffle_128x128E"
    # c, avg_c = run_asm(x, weight_asm, x_scale, w_scale, dtype, kernel_name=kernel_name)
    c, avg_c = run_asm(x, weight_asm, x_scale, w_scale, dtype)

    err_asm = checkAllclose(a, c, msg=f"{tag}")
    ret[f"{tag} us"] = avg_c
    ret[f"{tag} TFLOPS"] = m * n * k * 2 / avg_c / 1e6
    ret[f"{tag} TB/s"] = (x.nbytes + weight.nbytes) / avg_c / 1e6
    ret[f"{tag} err"] = err_asm
    ret["asm/ck"] = avg_c / avg_b

    for k,v in ret.items():
        print(f"\t{k}:{v}")
    return ret

def show_diff(y0, y1):
    diff_all = pyhip.calc_diff(y0, y1)
    if diff_all > 0.01:
        print(f"{diff_all=}")
        diff_map = torch.empty([M//16, N//16], dtype=torch.float)
        for y in range(0,M,16):
            for x in range(0,N,16):
                diff = pyhip.calc_diff(y0[y:y+16,x:x+16], y1[y:y+16,x:x+16])
                diff_map[y//16,x//16] = diff

        print("====================== diff_map")
        print(diff_map)
        print("====================== y0 vs y1")
        print(y0[:16,128+0:128+16])
        print(y1[:16,128+0:128+16])
        for i in range(M):
            diff = pyhip.calc_diff(y0[i], y1[i])
            if diff > 0.01:
                print("======================", i, diff)
                print(y0[i].view(-1,8))
                print(y1[i].view(-1,8))
                assert 0    

def compare_perf(m, n, k, ck_preshuffle=True):
    output_dtype = dtypes.bf16
    dim = (m, n, k)
    block_shape_n, block_shape_k = block_shape
    scale_m = m
    scale_n = (n + block_shape_n - 1) // block_shape_n
    scale_k = (k + block_shape_k - 1) // block_shape_k
    x = (torch.rand((m, k), dtype=dtypes.fp32, device="cuda") / 10).to(dtypes.fp8)
    weight = (torch.rand((n, k), dtype=dtypes.fp32, device="cuda") / 10).to(dtypes.fp8)
    x_scale = torch.rand([scale_m, scale_k], dtype=dtypes.fp32, device="cuda")
    w_scale = torch.rand([scale_n, scale_k], dtype=dtypes.fp32, device="cuda")
    #x_scale[...] = 1
    #w_scale[...] = 1
    print(w_scale.shape)

    out_torch, _ = run_torch(x, weight, x_scale, w_scale, output_dtype)


    x_scale_t = x_scale.transpose(0, 1).contiguous().view(*x_scale.shape)
    if ck_preshuffle:
        x_scale = x_scale_t

    BUF_COPY = 32
    As = [x.clone() for _ in range(BUF_COPY)]
    Ascales = [x_scale.clone() for _ in range(BUF_COPY)]
    Bs = [weight.clone() for _ in range(BUF_COPY)]
    Bscales = [w_scale.clone() for _ in range(BUF_COPY)]
    
    ck_kernel = aiter.gemm_a8w8_blockscale_bpreshuffle if ck_preshuffle else aiter.gemm_a8w8_blockscale
    di = 0
    for i in range(16):
        with pyhip.cudaPerf(m*n*k*2, (m*k+k*n), name=f"ck_kernel_{di}") as p0:
            out_ck = ck_kernel(As[di], Bs[di], Ascales[di], Bscales[di], output_dtype)
            di = (di + 1) % BUF_COPY

    if ck_preshuffle:
        out_asm = torch.empty((m, n), dtype=output_dtype, device=x.device)
        for i in range(16):
            with pyhip.cudaPerf(m*n*k*2, (m*k+k*n), name=f"asm_kernel_{di}") as p0:
                aiter.gemm_a8w8_blockscale_bpreshuffle_asm(As[di], Bs[di], out_asm, Ascales[di], Bscales[di])
            di = (di + 1) % BUF_COPY


    if gluon_gemm_a8w8_blockscale is not None:
        out_gluon = torch.empty((m, n), dtype=output_dtype, device=x.device)
        for i in range(16):
            with pyhip.cudaPerf(m*n*k*2, (m*k+k*n), name=f"gluon_kernel_{di}") as p0:
                gluon_gemm_a8w8_blockscale(As[di], Bs[di], Ascales[di], Bscales[di], output_dtype, out_gluon)
            di = (di + 1) % BUF_COPY

    if ck_preshuffle:
        wg_M, wg_N = 256, 256
        num_block_M = pyhip.div_up(m, wg_M)
        num_block_N = pyhip.div_up(n, wg_N)
        out_jit = torch.empty((m, n), dtype=output_dtype, device=x.device)
        for i in range(16):
            with pyhip.cudaPerf(m*n*k*2, (m*k+k*n), name=f"asmjit_kernel_{di}") as p0:
                gemm_fp8_8wave([num_block_N * num_block_M],[64*8], True, True,
                                wg_M, wg_N, N, K, As[di].data_ptr(), Bs[di].data_ptr(), out_jit.data_ptr(),
                                Ascales[di].data_ptr(), Bscales[di].data_ptr(), m)

            di = (di + 1) % BUF_COPY

    print(f"{pyhip.calc_diff(out_torch, out_ck)=:.2f}")
    print(f"{pyhip.calc_diff(out_torch, out_asm)=:.2f}")
    if gluon_gemm_a8w8_blockscale is not None:
        print(f"{pyhip.calc_diff(out_torch, out_gluon)=:.2f}")
    print(f"{pyhip.calc_diff(out_torch, out_jit)=:.2f}")
    #show_diff(out_torch, out_jit)

if __name__ == "__main__":
    '''
    MI350X: 
           M,N,K = 256*94, 256*16, 8192 
           ck_preshuffle=False: ck 1376.1 TFLOPS (跟相同shape的bf16的gemm性能相当)
           ck_preshuffle=True:  ck 1698.2 TFLOPS    asm 862.2 TFLOPS   gluon 527.7 TFLOPS  
    CK:  kernel_gemm_xdl_cshuffle_v3_multi_d_blockscale_b_preshuffle
            LDS_Block_Size 8192
            VGPR_Count 128
            Accum_VGPR_Count 0
            SGPR_Count 64
            workgroup_size 256
            grid_size 1540096

            v_mfma_f32_16x16x128_f8f6f4 v[248:251]
            v_mfma_f32_16x16x128_f8f6f4 v[244:247]
            v_mfma_f32_16x16x128_f8f6f4 v[240:243]
            buffer_load_dwordx4 没有使用LDS?
            v_pk_fma_f32
            v_fma_f32

    ASM: _ZN5aiter43fp8gemm_bf16_blockscale_BpreShuffle_128x128E.kd
    '''
    #
    print(type(dtypes.fp8), dtypes.fp8)

    #M,N,K = 256*94, 256*16, 8192 
    M,N,K=8192,4096,8192
    #M,N,K=256,256,128
    #test_gemm(dtypes.bf16, M, N, K, True)
    compare_perf(M,N,K, True)
    print(M,N,K)
"""
def run_torch(x, weight, x_scale, w_scale, bias=None, dtype=dtypes.bf16):
    x = x.to(dtypes.fp32) * x_scale
    weight = weight.to(dtypes.fp32) * w_scale
    out = F.linear(x, weight)
    if bias is not None:
        out = out.to(bias) + bias
    return out.to(dtype)

quantDtype=dtypes.fp8
dim = (m, n, k)
x = torch.randn((m, k), dtype=dtype, device="cuda")
weight = torch.randn((n, k), dtype=dtype, device="cuda")
x, x_scale = aiter.pertoken_quant(x, quant_dtype=quantDtype)
weight, w_scale = aiter.pertoken_quant(weight, quant_dtype=quantDtype)
weightshuffle = shuffle_weight(weight, layout=(16, 16))

a, avg_a = run_torch(x, weight, x_scale, w_scale, bias, dtype)

aiter.gemm_a8w8_CK(x, weight, x_scale, w_scale, bias, dtype)
aiter.gemm_a8w8_bpreshuffle(x, weight, x_scale, w_scale, None, dtype)
"""