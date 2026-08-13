import pytest
import os
import sys
import torch
import torch.nn.functional as F
from einops import rearrange
from einops import repeat as eirp
from typing_extensions import List

import aiter
from aiter import dtypes
from aiter.ops.quant import per_1x32_mx_quant_hip
from aiter.ops.shuffle import shuffle_weight
from aiter.test_common import benchmark, checkAllclose, perftest
from aiter.utility.fp4_utils import e8m0_to_f32
try:
    from aiter.ops.triton.gluon.gemm_a8w8_blockscale import (
        gemm_a8w8_blockscale as gluon_gemm_a8w8_blockscale,
    )
except:
    gluon_gemm_a8w8_blockscale = None
gluon_gemm_a8w8_blockscale = None

import pyhip
from pyhip.contrib.gemm_fp8 import *

FLYDSL_TEST_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "flydsl")
)
if FLYDSL_TEST_DIR not in sys.path:
    sys.path.insert(0, FLYDSL_TEST_DIR)
import flydsl.compiler as flyc
from test_gemm_fp8_8w_blockscale import compile_gemm_fp8_8wave
from test_mxfp8_gemm_4w import compile_gemm_fp8

torch.set_printoptions(linewidth=3000, sci_mode=False, edgeitems=8, )
torch.set_default_device('cuda')
torch.manual_seed(0)

block_shape = (128, 128)


def _dequant_blockscale_inputs(x, weight, x_scale, w_scale):
    m, k = x.shape
    n = weight.shape[0]
    scale_n = (n + block_shape[0] - 1) // block_shape[0]
    scale_k = (k + block_shape[1] - 1) // block_shape[1]
    x_fp32 = (
        x.float().view(m, scale_k, block_shape[1]) * x_scale.unsqueeze(-1)
    ).view(m, k)
    weight_scale = rearrange(
        w_scale.view(scale_n, scale_k, 1, 1).expand(
            scale_n, scale_k, block_shape[0], block_shape[1]
        ),
        "num_blk_n num_blk_k blk_n blk_k -> (num_blk_n blk_n) (num_blk_k blk_k)",
    )[:n, :k]
    return x_fp32, weight.float() * weight_scale


def _permute_mxfp8_scale(scale):
    scale = scale.view(torch.uint8)
    rows, groups = scale.shape
    permuted = (
        scale.view(rows // 128, 4, 32, groups)
        .permute(3, 0, 2, 1)
        .contiguous()
        .view(-1)
    )
    padding = torch.full((rows * 4,), 127, dtype=torch.uint8, device=scale.device)
    return torch.cat((permuted, padding)).view(torch.int32)


def run_flydsl4_mxfp8_accuracy(
    x, weight, x_scale, w_scale, output_dtype, num_repeats=0, data_clones=32
):
    m, k = x.shape
    n = weight.shape[0]
    x_source, weight_source = _dequant_blockscale_inputs(
        x, weight, x_scale, w_scale
    )
    mx_x, mx_x_scale = per_1x32_mx_quant_hip(
        x_source.to(torch.bfloat16),
        quant_dtype=dtypes.fp8,
        scale_type=dtypes.fp8_e8m0,
        shuffle=False,
    )
    mx_weight, mx_weight_scale = per_1x32_mx_quant_hip(
        weight_source.to(torch.bfloat16),
        quant_dtype=dtypes.fp8,
        scale_type=dtypes.fp8_e8m0,
        shuffle=False,
    )
    ref_x = mx_x.float() * e8m0_to_f32(mx_x_scale).repeat_interleave(32, dim=1)
    ref_weight = mx_weight.float() * e8m0_to_f32(mx_weight_scale).repeat_interleave(32, dim=1)
    ref = (ref_x @ ref_weight.t()).to(output_dtype)

    out = torch.empty((m, n), dtype=output_dtype, device=x.device)
    stream = torch.cuda.current_stream()
    mx_x_scale = _permute_mxfp8_scale(mx_x_scale)
    mx_weight_scale = _permute_mxfp8_scale(mx_weight_scale)
    args = (
        mx_x.view(torch.int8).view(-1),
        mx_weight.view(torch.int8).view(-1),
        mx_x_scale,
        mx_weight_scale,
        out.view(-1),
        m,
        stream,
    )
    launcher = compile_gemm_fp8(256, 256, 128, n, k, with_scale=True)
    kernel = flyc.compile[{"opt_level": 2}](launcher, *args)
    kernel(*args)
    torch.cuda.synchronize()
    diff = pyhip.calc_diff(ref, out, diff_thr=1e-5)
    print(f"flydsl4_mxfp8 accuracy: diff={diff:.6e}")

    if num_repeats:
        mx_x_clones = [mx_x.clone() for _ in range(data_clones)]
        mx_weight_clones = [mx_weight.clone() for _ in range(data_clones)]
        mx_x_scale_clones = [mx_x_scale.clone() for _ in range(data_clones)]
        mx_weight_scale_clones = [mx_weight_scale.clone() for _ in range(data_clones)]
        outputs = [torch.empty_like(out) for _ in range(data_clones)]
        arg_sets = [
            (
                mx_x_clones[i].view(torch.int8).view(-1),
                mx_weight_clones[i].view(torch.int8).view(-1),
                mx_x_scale_clones[i],
                mx_weight_scale_clones[i],
                outputs[i].view(-1),
                m,
                stream,
            )
            for i in range(data_clones)
        ]
        for clone_args in arg_sets:
            kernel(*clone_args)
        torch.cuda.synchronize()

        flops = 2 * m * n * k
        rw_bytes = (
            mx_x.numel() * mx_x.element_size()
            + mx_weight.numel() * mx_weight.element_size()
            + mx_x_scale.numel() * mx_x_scale.element_size()
            + mx_weight_scale.numel() * mx_weight_scale.element_size()
            + out.numel() * out.element_size()
        )
        latencies = []
        for i in range(num_repeats):
            clone_index = i % data_clones
            with pyhip.cudaPerf(
                flops, rw_bytes, name=f"flydsl4_mxfp8_kernel_{clone_index}"
            ) as perf:
                kernel(*arg_sets[clone_index])
            latencies.append(perf.dt_ms)
        best_ms = min(latencies)
        print(
            f"flydsl4_mxfp8 best: {best_ms * 1e3:.3f} us, "
            f"{flops / (best_ms * 1e-3) / 1e12:.1f} TFLOPS"
        )
    return out, ref

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

def txest_gemm(dtype, m, n, k, ck_preshuffle=True):
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


@pytest.mark.parametrize("k", [256])
@pytest.mark.parametrize("n", [256, 256*6])
@pytest.mark.parametrize("m", [32, 256, 2400])
def test_perf(m, n, k, num_repeats = 1, ck_preshuffle=True):
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

    if m % 256 == 0 and n % 256 == 0 and k >= 512 and k % 256 == 0:
        run_flydsl4_mxfp8_accuracy(
            x,
            weight,
            x_scale,
            w_scale,
            output_dtype,
            num_repeats=num_repeats,
        )
    else:
        print("skip flydsl4 mxfp8 accuracy: M/N must be multiples of 256; K must be a multiple of 256 and at least 512")

    x_scale_t = x_scale.transpose(0, 1).contiguous().view(*x_scale.shape)
    if ck_preshuffle:
        x_scale = x_scale_t

    BUF_COPY = 32
    As = [x.clone() for _ in range(BUF_COPY)]
    Ascales = [x_scale.clone() for _ in range(BUF_COPY)]
    ATscales = [x_scale_t.clone() for _ in range(BUF_COPY)]
    Bs = [weight.clone() for _ in range(BUF_COPY)]
    Bscales = [w_scale.clone() for _ in range(BUF_COPY)]
    
    rw_bytes = weight.numel() * weight.itemsize + \
               w_scale.numel() * w_scale.itemsize + \
               x.numel() * x.itemsize + \
               x_scale.numel() * x_scale.itemsize
    flops = m*n*k*2

    ck_kernel = aiter.gemm_a8w8_blockscale_bpreshuffle if ck_preshuffle else aiter.gemm_a8w8_blockscale
    di = 0
    for i in range(num_repeats):
        with pyhip.cudaPerf(flops, rw_bytes, name=f"ck_kernel_{di}") as p0:
            out_ck = ck_kernel(As[di], Bs[di], Ascales[di], Bscales[di], output_dtype)
            di = (di + 1) % BUF_COPY

    if ck_preshuffle:
        out_asm = torch.empty((m, n), dtype=output_dtype, device=x.device)
        for i in range(num_repeats):
            with pyhip.cudaPerf(flops, (m*k+k*n), name=f"asm_kernel_{di}") as p0:
                aiter.gemm_a8w8_blockscale_bpreshuffle_asm(As[di], Bs[di], out_asm, Ascales[di], Bscales[di])
            di = (di + 1) % BUF_COPY


    if gluon_gemm_a8w8_blockscale is not None:
        out_gluon = torch.empty((m, n), dtype=output_dtype, device=x.device)
        for i in range(num_repeats):
            with pyhip.cudaPerf(flops, rw_bytes, name=f"gluon_kernel_{di}") as p0:
                gluon_gemm_a8w8_blockscale(As[di], Bs[di], Ascales[di], Bscales[di], output_dtype, out_gluon)
            di = (di + 1) % BUF_COPY

    # gemm_8wave_fp8bf16fp16 requires  x_scale_t
    wg_M, wg_N = 256, 256
    num_block_M = pyhip.div_up(m, wg_M)
    num_block_N = pyhip.div_up(n, wg_N)
    out_jit = torch.empty((m, n), dtype=output_dtype, device=x.device)
    for i in range(num_repeats):
        with pyhip.cudaPerf(m*n*k*2, rw_bytes, name=f"asmjit_kernel_{di}") as p0:
            gemm_8wave_fp8bf16fp16([num_block_N * num_block_M],[64*8], "fp8", ck_preshuffle, True,
                            wg_M, wg_N, n, k, As[di].data_ptr(), Bs[di].data_ptr(), out_jit.data_ptr(),
                            ATscales[di].data_ptr(), Bscales[di].data_ptr(), m)

        di = (di + 1) % BUF_COPY

    out_flydsl = None
    if m % 256 == 0 and n % 256 == 0 and k >= 512 and k % 256 == 0:
        empty = torch.empty(0, dtype=dtypes.fp32, device=x.device)
        stream = torch.cuda.current_stream()
        flydsl_launcher = compile_gemm_fp8_8wave(
            256, 256, 128, n, k,
            permlane_epilogue=True,
            preshuffle_b=False,
            with_scale=True,
            useTileDMA=False,
        )
        flydsl_outputs = [
            torch.empty((m, n), dtype=output_dtype, device=x.device)
            for _ in range(BUF_COPY)
        ]
        flydsl_args = [
            (
                As[i].view(torch.int8).view(-1),
                Bs[i].view(torch.int8).view(-1),
                flydsl_outputs[i].view(-1),
                ATscales[i].view(-1),
                Bscales[i].view(-1),
                m,
                stream,
            )
            for i in range(BUF_COPY)
        ]
        flydsl_kernel = flyc.compile[{"opt_level": 2}](
            flydsl_launcher, *flydsl_args[0]
        )
        for i in range(BUF_COPY):
            flydsl_kernel(*flydsl_args[i])
        torch.cuda.synchronize()
        di = 0
        for i in range(num_repeats):
            with pyhip.cudaPerf(flops, rw_bytes, name=f"flydsl8_kernel_{di}"):
                flydsl_kernel(*flydsl_args[di])
            out_flydsl = flydsl_outputs[di]
            di = (di + 1) % BUF_COPY
    else:
        print("skip flydsl8: M and N must be multiples of 256; K must be a multiple of 256 and at least 512")

    print(f"{pyhip.calc_diff(out_torch, out_ck, diff_thr=0.01)=:.6f}")
    if ck_preshuffle:
        print(f"{pyhip.calc_diff(out_torch, out_asm, diff_thr=0.4)=:.6f}")
    if gluon_gemm_a8w8_blockscale is not None:
        print(f"{pyhip.calc_diff(out_torch, out_gluon, diff_thr=0.01)=:.2f}")
    print(f"{pyhip.calc_diff(out_torch, out_jit, diff_thr=0.04)=:.6f}")
    if out_flydsl is not None:
        print(f"{pyhip.calc_diff(out_torch, out_flydsl, diff_thr=0.04)=:.6f}")
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
    if 0:
        test_perf(256,256,128, num_repeats=1, ck_preshuffle=True)
        test_perf(256,256,256, num_repeats=1, ck_preshuffle=True)
        test_perf(8192,8192,4096, num_repeats=1, ck_preshuffle=True)

    #M,N,K = 256*94, 256*16, 8192 
    #M,N,K=8192,8192,8192
    #M,N,K=32768,9216,4096
    # pyhip_gemm_a8w8_blockscale:  torch.bfloat16 torch.float8_e4m3fn torch.Size([32, 4096]) torch.float8_e4m3fn torch.Size([1024, 4096]) [128, 128] True
    M,N,K=32,1024,4096 
    #M,N,K=256,256,128
    #test_gemm(dtypes.bf16, M, N, K, True)
    test_perf(M,N,K, num_repeats=16, ck_preshuffle=False)
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