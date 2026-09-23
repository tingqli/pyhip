import importlib.util
from functools import lru_cache, partial
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from einops import rearrange
from einops import repeat as eirp
from typing_extensions import List

import aiter
from aiter import dtypes
from aiter.jit.core import AITER_CONFIGS
from aiter.ops import gemm_op_a8w8
from aiter.ops.shuffle import shuffle_weight
from aiter.test_common import benchmark, checkAllclose, perftest

# Gluon is disabled for now.
gluon_gemm_a8w8_blockscale = None

PERF_VERBOSE = 0
import pyhip
from pyhip.contrib.gemm_fp8 import gemm_8wave_fp8bf16fp16

torch.set_printoptions(
    linewidth=3000,
    sci_mode=False,
    edgeitems=8,
)
torch.set_default_device("cuda")
torch.manual_seed(0)

block_shape = (128, 128)


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


def run_gemm_ck(x, weight, x_scale, w_scale, dtype=dtypes.bf16):
    return aiter.gemm_a8w8_blockscale(x, weight, x_scale, w_scale, dtype)


def run_gemm_bpreshuffle_ck(x, weightshuffle, x_scale, w_scale, dtype=dtypes.bf16):
    return aiter.gemm_a8w8_blockscale_bpreshuffle(
        x, weightshuffle, x_scale, w_scale, dtype
    )


def run_asm(x, weight, x_scale, w_scale, dtype=dtypes.bf16, kernel_name=None):
    m, k = x.shape
    n, _ = weight.shape
    out = torch.empty((m, n), dtype=dtype, device=x.device)
    return aiter.gemm_a8w8_blockscale_bpreshuffle_asm(x, weight, out, x_scale, w_scale)


@lru_cache(maxsize=1)
def _flydsl_module():
    # Reuse the existing kernel, without copying it or running its __main__ tests.
    path = (
        Path(__file__).resolve().parents[2]
        / "flydsl/fp8_gemm_950/test_gemm_fp8_8w_blockscale.py"
    )
    spec = importlib.util.spec_from_file_location("flydsl_blockscale_comparison", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare_flydsl(x, weight, out, x_scale_t, w_scale, split_m, preshuffle_b=False):
    m, k = x.shape
    n = weight.shape[0]
    fly = _flydsl_module()
    args = (
        x.view(torch.int8),
        weight.view(torch.int8),
        out.view(-1),
        x_scale_t.view(-1),
        w_scale.view(-1),
        m,
        torch.cuda.current_stream(),
    )
    launcher = fly.compile_gemm_fp8_8wave(
        256,
        256,
        128,
        n,
        k,
        with_scale=True,
        split_m=split_m,
        preshuffle_b=preshuffle_b,
    )
    return fly.flyc.compile[{"opt_level": 2}](launcher, *args), args


def _aiter_config(m, n, k, preshuffle, require_tuned):
    path = (
        AITER_CONFIGS.AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE_FILE
        if preshuffle
        else AITER_CONFIGS.AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_FILE
    )
    config = gemm_op_a8w8.get_CKGEMM_config(m, n, k, path)
    if require_tuned and config is None:
        raise RuntimeError(
            f"AIter has no tuned config for {(m, n, k)}, preshuffle={preshuffle}: {path}. "
            "Run csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale_tune.py "
            "with/without --preshuffle, and set AITER_CONFIG_GEMM_A8W8_BLOCKSCALE"
            "[_BPRESHUFFLE] to the resulting CSV before starting this process."
        )
    print(
        f"AIter preshuffle={preshuffle}: {path}\n  {config or 'UNTUNED fallback'}",
        flush=True,
    )
    return config


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

    a = run_torch(x, weight, x_scale, w_scale, dtype)

    x_scale_t = x_scale.transpose(0, 1).contiguous().view(*x_scale.shape)
    gemm_x_scale = x_scale_t if ck_preshuffle else x_scale
    gemm_weight = shuffle_weight(weight, layout=(16, 16)) if ck_preshuffle else weight
    run_func = run_gemm_bpreshuffle_ck if ck_preshuffle else run_gemm_ck
    b = run_func(x, gemm_weight, gemm_x_scale, w_scale, dtype)

    err_ck = checkAllclose(a, b, msg="ck", tol_err_ratio=0)
    assert err_ck == 0
    ret["ck err"] = err_ck
    ret["ck diff"] = pyhip.calc_diff(a, b, diff_thr=1e-5)

    if ck_preshuffle:
        tag = "asm"
        weight_asm = shuffle_weight(weight, layout=(16, 16))
        c = run_asm(x, weight_asm, x_scale_t, w_scale, dtype)
        err_asm = checkAllclose(a, c, msg=f"{tag}", tol_err_ratio=0)
        assert err_asm == 0
        ret[f"{tag} err"] = err_asm
        ret[f"{tag} diff"] = pyhip.calc_diff(a, c, diff_thr=1e-5)

    wg_M, wg_N = 256, 256
    num_block_M = pyhip.div_up(m, wg_M)
    num_block_N = pyhip.div_up(n, wg_N)
    out_jit = torch.empty((m, n), dtype=dtype, device=x.device)
    gemm_8wave_fp8bf16fp16(
        [num_block_N * num_block_M],
        [64 * 8],
        "fp8",
        ck_preshuffle,
        True,
        wg_M,
        wg_N,
        n,
        k,
        x.data_ptr(),
        gemm_weight.data_ptr(),
        out_jit.data_ptr(),
        x_scale_t.data_ptr(),
        w_scale.data_ptr(),
        m,
    )
    err_jit = checkAllclose(a, out_jit, msg="asmjit", tol_err_ratio=0)
    assert err_jit == 0
    ret["asmjit err"] = err_jit
    ret["asmjit diff"] = pyhip.calc_diff(a, out_jit, diff_thr=1e-5)

    for split_m in (False, True):
        tag = f"flydsl_split_m_{split_m}"
        out_fly = torch.empty((m, n), dtype=dtype, device=x.device)
        fly_kernel, args = prepare_flydsl(
            x,
            gemm_weight,
            out_fly,
            x_scale_t,
            w_scale,
            split_m,
            preshuffle_b=ck_preshuffle,
        )
        fly_kernel(*args)
        err_fly = checkAllclose(a, out_fly, msg=tag, tol_err_ratio=0)
        assert err_fly == 0
        ret[f"{tag} err"] = err_fly
        ret[f"{tag} diff"] = pyhip.calc_diff(a, out_fly, diff_thr=1e-5)

    for k, v in ret.items():
        print(f"\t{k}:{v}")
    return ret


@pytest.mark.parametrize("k", [16384])
@pytest.mark.parametrize("n", [4096])
@pytest.mark.parametrize("m", [4096])
@pytest.mark.parametrize("ck_preshuffle", [False, True])
def test_perf(m, n, k, num_repeats=1, ck_preshuffle=True, require_tuned=True):
    config = _aiter_config(m, n, k, ck_preshuffle, require_tuned)
    assert num_repeats > 0
    max_tflops = {}
    diffs = {}

    def record_perf(kernel_type, perf):
        max_tflops.setdefault(kernel_type, None)
        if perf.latencies:
            max_tflops[kernel_type] = max(max_tflops[kernel_type] or 0.0, perf.tflops())

    aiter_type = f"aiter_{config['libtype']}" if config else "aiter_untuned"
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
    # x_scale[...] = 1
    # w_scale[...] = 1
    print(w_scale.shape)

    out_torch = run_torch(x, weight, x_scale, w_scale, output_dtype)

    # PyHIP/FlyDSL always read column-major A scales. B scales remain [N/128,K/128].
    x_scale_t = x_scale.transpose(0, 1).contiguous().view(*x_scale.shape)
    if ck_preshuffle:
        x_scale = x_scale_t
        weight = shuffle_weight(weight, layout=(16, 16))

    BUF_COPY = 32
    As = [x.clone() for _ in range(BUF_COPY)]
    Ascales = [x_scale.clone() for _ in range(BUF_COPY)]
    ATscales = [x_scale_t.clone() for _ in range(BUF_COPY)]
    Bs = [weight.clone() for _ in range(BUF_COPY)]
    Bscales = [w_scale.clone() for _ in range(BUF_COPY)]

    rw_bytes = (
        weight.numel() * weight.itemsize
        + w_scale.numel() * w_scale.itemsize
        + x.numel() * x.itemsize
        + x_scale.numel() * x_scale.itemsize
    )
    flops = m * n * k * 2

    out_ck = torch.empty((m, n), dtype=output_dtype, device=x.device)
    if ck_preshuffle:
        ck_kernel = partial(
            aiter.gemm_a8w8_blockscale_bpreshuffle, dtype=output_dtype, out=out_ck
        )
    else:
        # Same tuned dispatch as the public API, but with allocation outside timing.
        lib = config["libtype"] if config else "ck"
        assert lib in ("ck", "cktile")
        ck_kernel = partial(
            getattr(aiter, f"gemm_a8w8_blockscale_{lib}"),
            Out=out_ck,
            splitK=int(config.get("splitK", 0)) if config else 0,
            kernelName=str(config.get("kernelName", "")) if config else "",
        )
    # Warm all copies and finish JIT before recording events; restart at copy 0.
    for di in range(BUF_COPY):
        ck_kernel(As[di], Bs[di], Ascales[di], Bscales[di])
    torch.cuda.synchronize()
    di = 0
    for i in range(num_repeats):
        with pyhip.cudaPerf(
            flops, rw_bytes, name=f"ck_kernel_{di}", verbose=PERF_VERBOSE
        ) as p0:
            ck_kernel(As[di], Bs[di], Ascales[di], Bscales[di])
        record_perf(aiter_type, p0)
        di = (di + 1) % BUF_COPY

    if ck_preshuffle:
        out_asm = torch.empty((m, n), dtype=output_dtype, device=x.device)
        for di in range(BUF_COPY):
            aiter.gemm_a8w8_blockscale_bpreshuffle_asm(
                As[di], Bs[di], out_asm, Ascales[di], Bscales[di]
            )
        torch.cuda.synchronize()
        di = 0
        for i in range(num_repeats):
            with pyhip.cudaPerf(
                flops, rw_bytes, name=f"asm_kernel_{di}", verbose=PERF_VERBOSE
            ) as p0:
                aiter.gemm_a8w8_blockscale_bpreshuffle_asm(
                    As[di], Bs[di], out_asm, Ascales[di], Bscales[di]
                )
            record_perf("asm", p0)
            di = (di + 1) % BUF_COPY

    if gluon_gemm_a8w8_blockscale is not None:
        out_gluon = torch.empty((m, n), dtype=output_dtype, device=x.device)
        for di in range(BUF_COPY):
            gluon_gemm_a8w8_blockscale(
                As[di], Bs[di], Ascales[di], Bscales[di], output_dtype, out_gluon
            )
        torch.cuda.synchronize()
        di = 0
        for i in range(num_repeats):
            with pyhip.cudaPerf(
                flops, rw_bytes, name=f"gluon_kernel_{di}", verbose=PERF_VERBOSE
            ) as p0:
                gluon_gemm_a8w8_blockscale(
                    As[di], Bs[di], Ascales[di], Bscales[di], output_dtype, out_gluon
                )
            record_perf("gluon", p0)
            di = (di + 1) % BUF_COPY

    # gemm_8wave_fp8bf16fp16 requires  x_scale_t
    wg_M, wg_N = 256, 256
    num_block_M = pyhip.div_up(m, wg_M)
    num_block_N = pyhip.div_up(n, wg_N)
    out_jit = torch.empty((m, n), dtype=output_dtype, device=x.device)
    for di in range(BUF_COPY):
        gemm_8wave_fp8bf16fp16(
            [num_block_N * num_block_M],
            [64 * 8],
            "fp8",
            ck_preshuffle,
            True,
            wg_M,
            wg_N,
            n,
            k,
            As[di].data_ptr(),
            Bs[di].data_ptr(),
            out_jit.data_ptr(),
            ATscales[di].data_ptr(),
            Bscales[di].data_ptr(),
            m,
        )
    torch.cuda.synchronize()
    di = 0
    for i in range(num_repeats):
        with pyhip.cudaPerf(
            m * n * k * 2, rw_bytes, name=f"asmjit_kernel_{di}", verbose=PERF_VERBOSE
        ) as p0:
            gemm_8wave_fp8bf16fp16(
                [num_block_N * num_block_M],
                [64 * 8],
                "fp8",
                ck_preshuffle,
                True,
                wg_M,
                wg_N,
                n,
                k,
                As[di].data_ptr(),
                Bs[di].data_ptr(),
                out_jit.data_ptr(),
                ATscales[di].data_ptr(),
                Bscales[di].data_ptr(),
                m,
            )

        record_perf("pyhip", p0)
        di = (di + 1) % BUF_COPY

    for split_m in (False, True):
        kernel_type = f"flydsl_split_m_{split_m}"
        out_fly = torch.empty((m, n), dtype=output_dtype, device=x.device)
        fly_kernel, args = prepare_flydsl(
            As[0],
            Bs[0],
            out_fly,
            ATscales[0],
            Bscales[0],
            split_m,
            preshuffle_b=ck_preshuffle,
        )
        fly_args = [
            (
                As[di].view(torch.int8),
                Bs[di].view(torch.int8),
                out_fly.view(-1),
                ATscales[di].view(-1),
                Bscales[di].view(-1),
                m,
                args[-1],
            )
            for di in range(BUF_COPY)
        ]
        for di in range(BUF_COPY):
            fly_kernel(*fly_args[di])
        torch.cuda.synchronize()
        di = 0
        for i in range(num_repeats):
            with pyhip.cudaPerf(
                flops,
                rw_bytes,
                name=f"flydsl_split_m_{split_m}_{di}",
                verbose=PERF_VERBOSE,
            ) as p0:
                fly_kernel(*fly_args[di])
            record_perf(kernel_type, p0)
            di = (di + 1) % BUF_COPY
        diffs[kernel_type] = pyhip.calc_diff(out_torch, out_fly, diff_thr=1e-5)

    diffs[aiter_type] = pyhip.calc_diff(out_torch, out_ck, diff_thr=1e-5)
    if ck_preshuffle:
        diffs["asm"] = pyhip.calc_diff(out_torch, out_asm, diff_thr=1e-5)
    if gluon_gemm_a8w8_blockscale is not None:
        diffs["gluon"] = pyhip.calc_diff(out_torch, out_gluon, diff_thr=1e-5)
    diffs["pyhip"] = pyhip.calc_diff(out_torch, out_jit, diff_thr=1e-5)

    print(f"\nSummary: M={m}, N={n}, K={k}, {ck_preshuffle=}, {num_repeats=}")
    rows = [
        (
            kernel_type,
            f"{diffs[kernel_type]:.6e}",
            f"{peak:.1f}" if peak is not None else "N/A (timing disabled)",
        )
        for kernel_type, peak in max_tflops.items()
    ]
    headers = ("Kernel type", "Diff", "Max TFLOPS")
    widths = [max(len(row[i]) for row in [headers, *rows]) for i in range(3)]
    row_format = f"| {{:<{widths[0]}}} | {{:>{widths[1]}}} | {{:>{widths[2]}}} |"
    print(row_format.format(*headers))
    print(
        f"| {'-' * widths[0]} | {'-' * (widths[1] - 1)}: | {'-' * (widths[2] - 1)}: |"
    )
    for row in rows:
        print(row_format.format(*row))


if __name__ == "__main__":
    """
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
    """
    #
    print(type(dtypes.fp8), dtypes.fp8)
    if 0:
        test_perf(256, 256, 128, num_repeats=1, ck_preshuffle=True)
        test_perf(256, 256, 256, num_repeats=1, ck_preshuffle=True)
        test_perf(8192, 8192, 4096, num_repeats=1, ck_preshuffle=True)

    # M,N,K = 256*94, 256*16, 8192
    # M,N,K=8192,8192,8192
    # M,N,K=32768,9216,4096
    # pyhip_gemm_a8w8_blockscale:  torch.bfloat16 torch.float8_e4m3fn torch.Size([32, 4096]) torch.float8_e4m3fn torch.Size([1024, 4096]) [128, 128] True
    M, N, K = 4096, 4096, 16384
    # M,N,K=256,256,128
    # txest_gemm(dtypes.bf16, M, N, K, False)
    test_perf(M, N, K, num_repeats=32, ck_preshuffle=False)
    test_perf(M, N, K, num_repeats=32, ck_preshuffle=True)

    M, N, K = 16384, 3584, 6144
    # txest_gemm(dtypes.bf16, M, N, K, False)
    test_perf(M, N, K, num_repeats=32, ck_preshuffle=False)
    test_perf(M, N, K, num_repeats=32, ck_preshuffle=True)

    M, N, K = 16384, 3392, 6144
    # M,N,K=256,256,128
    # txest_gemm(dtypes.bf16, M, N, K, False)
    test_perf(M, N, K, num_repeats=32, ck_preshuffle=False)
    test_perf(M, N, K, num_repeats=32, ck_preshuffle=True)

    print(M, N, K)
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
