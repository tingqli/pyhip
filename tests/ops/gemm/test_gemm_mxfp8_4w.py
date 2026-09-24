# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Accuracy and opt-in performance tests for the CDNA4 (gfx950) MXFP8 GEMM."""

import os
from types import SimpleNamespace

import pytest
import pyhip
from pyhip import div_up

torch = pytest.importorskip("torch")
flyc = pytest.importorskip("flydsl.compiler")


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


# =========================== test / perf ===========================
TILE_M = 256
TILE_N = 256
TILE_K = 128
M = int(os.environ.get("GEMM_M", 4096))
N = int(os.environ.get("GEMM_N", 4096))
K = int(os.environ.get("GEMM_K", 16384))

# permlane 存储 / store 与 MFMA 交织：可用环境变量覆盖默认值（对标 bf16 v9 run_test 结构）。
PERMLANE_EPILOGUE = _env_flag("PERMLANE", "1")
STORE_OVERLAP = _env_flag("STORE_OVERLAP")


def _load_shuffle_weight():
    # The FlyDSL wheel does not ship tests.utils. AIter's public helper uses
    # the same (16, 64) FP8 block permutation (BK=128, kWidth=16 bytes).
    from aiter.ops.shuffle import shuffle_weight

    return shuffle_weight


def _load_mxfp8_quant():
    from aiter import dtypes
    from aiter.ops.quant import per_1x32_mx_quant_hip
    from aiter.utility.fp4_utils import e8m0_to_f32, mxfp4_to_f32

    return per_1x32_mx_quant_hip, dtypes, e8m0_to_f32, mxfp4_to_f32


def run_test(
    M,
    N,
    K,
    USE_SWIZZLE=False,
    PRESHUFFLE_B=False,
    perf=False,
    TILEM=256,
    TILEN=256,
    TILEK=128,
    permlane_output=True,
    store_overlap=False,
    with_scale=False,
    B_MXFP4=False,
    B_LDS_SWIZZLE=None,
    run_count=50,
    data_clones=50,
):
    from pyhip.ops.gemm.flydsl.gemm_mxfp8_4w import compile_gemm_fp8

    assert N % (256 if PRESHUFFLE_B else 8) == 0
    assert not B_MXFP4 or not PRESHUFFLE_B
    if B_LDS_SWIZZLE is None:
        B_LDS_SWIZZLE = True if B_MXFP4 else USE_SWIZZLE
    shuffle_weight = _load_shuffle_weight() if PRESHUFFLE_B else None
    mxfp8_quant = _load_mxfp8_quant() if (with_scale or B_MXFP4) else None

    def _shuffle_b(x):
        return shuffle_weight(x, layout=(16, 64)) if PRESHUFFLE_B else x

    def _permute_scale(scale, padded_rows):
        scale = scale.view(torch.uint8)
        rows, groups = scale.shape
        if rows != padded_rows:
            scale = torch.cat(
                (
                    scale,
                    torch.full(
                        (padded_rows - rows, groups),
                        127,
                        device=scale.device,
                        dtype=torch.uint8,
                    ),
                ),
                dim=0,
            )
        rows = padded_rows
        permuted = (
            scale.view(rows // 128, 4, 32, groups)
            .permute(3, 0, 2, 1)
            .contiguous()
            .view(-1)
        )
        # Retain the legacy host allocation padding for run_test compatibility;
        # raw scale DMA consumes only four groups per BK128, without overread.
        padding = torch.full((rows * 4,), 127, device=scale.device, dtype=torch.uint8)
        return torch.cat((permuted, padding)).view(torch.int32)

    def _random_fp8(shape):
        return torch.randn(shape, device="cuda", dtype=torch.float32).to(
            torch.float8_e4m3fn
        )

    def _random_fp4(shape):
        rows, columns = shape
        return torch.randint(
            0, 256, (rows, columns // 2), device="cuda", dtype=torch.uint8
        ).view(torch.float4_e2m1fn_x2)

    def _quant_mxfp8(x):
        per_1x32_mx_quant_hip, dtypes, _, _ = mxfp8_quant
        return per_1x32_mx_quant_hip(
            x.to(torch.bfloat16),
            quant_dtype=dtypes.fp8,
            scale_type=dtypes.fp8_e8m0,
            shuffle=False,
        )

    def _dequant_mxfp8(x, scale):
        _, _, e8m0_to_f32, _ = mxfp8_quant
        scale_f32 = e8m0_to_f32(scale).repeat_interleave(32, dim=1)
        return x.float() * scale_f32

    def _quant_mxfp4(x):
        per_1x32_mx_quant_hip, dtypes, _, _ = mxfp8_quant
        return per_1x32_mx_quant_hip(
            x.to(torch.bfloat16),
            quant_dtype=dtypes.fp4x2,
            scale_type=dtypes.fp8_e8m0,
            shuffle=False,
        )

    def _dequant_mxfp4(x, scale):
        _, _, e8m0_to_f32, mxfp4_to_f32 = mxfp8_quant
        scale_f32 = e8m0_to_f32(scale).repeat_interleave(32, dim=1)
        return mxfp4_to_f32(x) * scale_f32

    if with_scale:
        a, scale_a_raw = _quant_mxfp8(torch.randn((M, K), device="cuda") * 0.75)
        b, scale_b_raw = (_quant_mxfp4 if B_MXFP4 else _quant_mxfp8)(
            torch.randn((N, K), device="cuda") * 3.0
        )
        b_ref = (_dequant_mxfp4 if B_MXFP4 else _dequant_mxfp8)(b, scale_b_raw)
        ref = _dequant_mxfp8(a, scale_a_raw) @ b_ref.t()
    else:
        a = _random_fp8((M, K))
        b = _random_fp4((N, K)) if B_MXFP4 else _random_fp8((N, K))
        scale_a_raw = scale_b_raw = None
        if B_MXFP4:
            _, _, _, mxfp4_to_f32 = mxfp8_quant
            ref = a.float() @ mxfp4_to_f32(b).t()
        else:
            ref = a.float() @ b.float().t()
    out = torch.zeros((M, N), device="cuda", dtype=torch.bfloat16)
    weight = _shuffle_b(b)  # preshuffle 时喂 shuffle 后的 B；ref 仍用原始 b
    scale_a = (
        _permute_scale(scale_a_raw, div_up(M, 256) * 256)
        if with_scale
        else torch.empty(1, device="cuda", dtype=torch.uint8)
    )
    scale_b = (
        _permute_scale(scale_b_raw, div_up(N, 256) * 256)
        if with_scale
        else torch.empty(1, device="cuda", dtype=torch.uint8)
    )
    stream = torch.cuda.current_stream()
    args = (
        # Keep dimensions below the signed int32 shape ABI limit at large M.
        # The kernel uses the same base pointers and explicit M/N/K bounds.
        a.view(torch.int8),
        weight.view(torch.int8),
        scale_a,
        scale_b,
        out.view(-1),
        M,
        stream,
    )

    launcher = compile_gemm_fp8(
        TILEM,
        TILEN,
        TILEK,
        N,
        K,
        lds_swizzle=USE_SWIZZLE,
        b_lds_swizzle=B_LDS_SWIZZLE,
        preshuffle_b=PRESHUFFLE_B,
        permlane_epilogue=permlane_output,
        store_overlap=store_overlap,
        with_scale=with_scale,
        b_mxfp4=B_MXFP4,
    )
    kernel = flyc.compile[{"opt_level": 2}](launcher, *args)
    kernel(*args)
    torch.cuda.synchronize()

    ref_bf16 = ref.to(torch.bfloat16)
    diff = (
        pyhip.calc_diff(out.float(), ref_bf16, diff_thr=0.00001)
        if torch.isfinite(out).all()
        else float("inf")
    )
    is_correct = diff <= 0.00001
    print(
        f"####M={M} N={N} K={K} {USE_SWIZZLE=} "
        f"{B_LDS_SWIZZLE=} {PRESHUFFLE_B=} {B_MXFP4=} "
        f"{is_correct=} {diff=}"
    )
    if not torch.allclose(out, ref_bf16, rtol=0.02, atol=0.01):
        abs_err = (out.float() - ref_bf16.float()).abs()
        tolerance = 0.01 + 0.02 * ref_bf16.float().abs()
        max_index = abs_err.argmax().item()
        max_row, max_col = divmod(max_index, N)
        print(
            f"  strict_allclose=False  max_abs_err={abs_err.max().item():.3f}  "
            f"outside_tolerance={(abs_err > tolerance).sum().item()}/{M*N}"
        )
        print(
            f"  max_error_at=({max_row}, {max_col})  "
            f"ref_fp32={ref[max_row, max_col].item()}  "
            f"ref_bf16={ref_bf16[max_row, max_col].item()}  "
            f"result={out[max_row, max_col].item()}  "
            f"abs_err={abs_err[max_row, max_col].item()}"
        )

    if not perf:
        return is_correct

    # ---- perf（多份数据轮转，排除 L2 cache 影响）----
    # 单份 A+B 只有几十 MB，反复喂同一份会常驻 L2 -> 高估 TFLOPS；轮转多份（远大于 L2）确保 cold data。
    if with_scale:
        quantized_as = [
            _quant_mxfp8(torch.randn((M, K), device="cuda", dtype=torch.float32) * 0.75)
            for _ in range(data_clones)
        ]
        quantized_bs = [
            (_quant_mxfp4 if B_MXFP4 else _quant_mxfp8)(
                torch.randn((N, K), device="cuda", dtype=torch.float32) * 3.0
            )
            for _ in range(data_clones)
        ]
        As = [quantized[0] for quantized in quantized_as]
        Bs = [_shuffle_b(quantized[0]) for quantized in quantized_bs]
        ScaleAs = [
            _permute_scale(quantized[1], div_up(M, 256) * 256)
            for quantized in quantized_as
        ]
        ScaleBs = [
            _permute_scale(quantized[1], div_up(N, 256) * 256)
            for quantized in quantized_bs
        ]
    else:
        As = [_random_fp8((M, K)) for _ in range(data_clones)]
        Bs = [
            _shuffle_b(_random_fp4((N, K)) if B_MXFP4 else _random_fp8((N, K)))
            for _ in range(data_clones)
        ]
        ScaleAs = [scale_a for _ in range(data_clones)]
        ScaleBs = [scale_b for _ in range(data_clones)]
    Cs = [
        torch.zeros((M, N), device="cuda", dtype=torch.bfloat16)
        for _ in range(data_clones)
    ]
    arg_sets = [
        (
            As[i].view(torch.int8),
            Bs[i].view(torch.int8),
            ScaleAs[i],
            ScaleBs[i],
            Cs[i].view(-1),
            M,
            stream,
        )
        for i in range(data_clones)
    ]

    flops = 2 * M * N * K
    mem_bytes = M * K + N * K * (0.5 if B_MXFP4 else 1) + M * N * 2
    if with_scale:
        mem_bytes += scale_a.numel() * scale_a.element_size()
        mem_bytes += scale_b.numel() * scale_b.element_size()

    # warmup（轮转，把所有 clone 都碰一遍）
    for i in range(data_clones):
        kernel(*arg_sets[i])
    torch.cuda.synchronize()

    di = 0
    latencies = []
    for _ in range(run_count):
        di = (di + 1) % data_clones
        with pyhip.cudaPerf(flops, mem_bytes, name=f"gemm_{di}") as p:
            kernel(*arg_sets[di])
        latencies.append(p.dt_ms)
    latencies.sort()
    best_ms = latencies[0]
    tflops = flops / (best_ms * 1e-3) / 1e12
    bw_gbs = mem_bytes / (best_ms * 1e-3) / 1e9
    print(
        f"\n=== perf  M={M} N={N} K={K} USE_SWIZZLE={USE_SWIZZLE} PRESHUFFLE_B={PRESHUFFLE_B} {with_scale=} {B_MXFP4=}==="
    )
    print(f"gemm:  {best_ms*1e3:.1f} us  {tflops:.2f} TFLOPS  {bw_gbs:.1f} GB/s")
    return is_correct


@pytest.mark.usefixtures("cdna4_device")
@pytest.mark.parametrize(
    "M,N,K",
    [
        (33, 64, 512),
        (62, 384, 512),
        (75, 448, 512),
        (111, 192, 512),
        (256, 256, 512),
        (257, 264, 768),
    ],
)
@pytest.mark.parametrize("with_scale", [False, True])
@pytest.mark.parametrize("B_MXFP4", [False, True])
def test_accuracy(M, N, K, with_scale, B_MXFP4):
    if with_scale or B_MXFP4:
        pytest.importorskip("aiter")
    assert run_test(
        M=M,
        N=N,
        K=K,
        USE_SWIZZLE=False,
        PRESHUFFLE_B=False,
        perf=False,
        TILEK=TILE_K,
        permlane_output=PERMLANE_EPILOGUE,
        store_overlap=STORE_OVERLAP,
        with_scale=with_scale,
        B_MXFP4=B_MXFP4,
    )


@pytest.mark.usefixtures("cdna4_device")
@pytest.mark.parametrize(
    "options",
    [
        pytest.param({"USE_SWIZZLE": True}, id="swizzle"),
        pytest.param({"PRESHUFFLE_B": True}, id="preshuffle"),
        pytest.param({"permlane_output": False}, id="no-permlane"),
        pytest.param({"store_overlap": True}, id="store-overlap"),
    ],
)
def test_layout_accuracy(options):
    if options.get("PRESHUFFLE_B", False):
        pytest.importorskip("aiter")
    assert run_test(
        M=256, N=256, K=512, with_scale=False, B_MXFP4=False, perf=False, **options
    )


@pytest.mark.perf
@pytest.mark.usefixtures("cdna4_device")
@pytest.mark.parametrize(
    "M,N,K",
    [(98304, 512, 6144), (196608, 512, 6144), (393216, 512, 6144)],
)
def test_perf(M, N, K):
    pytest.importorskip("aiter")
    assert run_test(
        M=M,
        N=N,
        K=K,
        USE_SWIZZLE=0,
        PRESHUFFLE_B=0,
        perf=True,
        TILEK=TILE_K,
        permlane_output=PERMLANE_EPILOGUE,
        store_overlap=STORE_OVERLAP,
        with_scale=True,
        B_MXFP4=True,
        B_LDS_SWIZZLE=True,
    )


@pytest.fixture
def gemm_factory(monkeypatch):
    from pyhip.ops.gemm.flydsl.gemm_mxfp8_4w import compile_gemm_fp8

    monkeypatch.setenv("FLYDSL_COMPILE_BACKEND", "rocm")
    monkeypatch.setenv("ARCH", "gfx950")
    monkeypatch.setenv("COMPILE_ONLY", "1")
    compile_gemm_fp8.cache_clear()
    yield compile_gemm_fp8
    compile_gemm_fp8.cache_clear()


@pytest.mark.parametrize(
    "backend,arch", [("rocm", "gfx942"), ("rocm", "gfx1100"), ("cuda", "sm_90")]
)
def test_requires_cdna4(monkeypatch, gemm_factory, backend, arch):
    from pyhip.ops.gemm.flydsl import common

    # A warmed cache must not bypass validation after switching targets.
    gemm_factory(TILE_M, TILE_N, TILE_K, 256, 512)
    cache_info = gemm_factory.cache_info()
    monkeypatch.setattr(
        common.flyc,
        "get_backend",
        lambda: SimpleNamespace(target=SimpleNamespace(backend=backend, arch=arch)),
    )
    with pytest.raises(RuntimeError, match="CDNA4"):
        gemm_factory(TILE_M, TILE_N, TILE_K, 256, 512)
    assert gemm_factory.cache_info() == cache_info


@pytest.mark.parametrize("arch", ["gfx950", "gfx950:sramecc+:xnack-"])
def test_accepts_cdna4_compile_target(monkeypatch, gemm_factory, arch):
    monkeypatch.setenv("ARCH", arch)
    assert callable(gemm_factory(TILE_M, TILE_N, TILE_K, 256, 512))


def test_launcher_cache(monkeypatch, gemm_factory):
    args = (TILE_M, TILE_N, TILE_K, 256, 512)
    launcher = gemm_factory(*args)
    assert gemm_factory(
        TILE_M=TILE_M, TILE_N=TILE_N, TILE_K=TILE_K, N=256, K=512,
        with_scale=False,
    ) is launcher
    assert gemm_factory(*args, with_scale=True) is not launcher
    assert gemm_factory(TILE_M, TILE_N, TILE_K, 384, 512) is not launcher
    assert gemm_factory.cache_info().hits == 1
    assert gemm_factory.cache_info().misses == 3

    monkeypatch.setenv("ARCH", "gfx950:sramecc+:xnack-")
    assert gemm_factory(*args) is not launcher
    monkeypatch.setenv("ARCH", "gfx950")
    assert gemm_factory(*args) is launcher
    assert gemm_factory.cache_info().currsize == 4

    gemm_factory.cache_clear()
    assert gemm_factory.cache_info().currsize == 0
    assert gemm_factory(*args) is not launcher


if __name__ == "__main__":
    if torch.version.hip is None or not torch.cuda.is_available():
        raise RuntimeError("MXFP8 GEMM requires a ROCm CDNA4 (gfx950) GPU")
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    if props.gcnArchName.split(":", 1)[0] != "gfx950":
        raise RuntimeError(
            f"MXFP8 GEMM requires CDNA4 (gfx950); got {props.gcnArchName}"
        )
    torch.manual_seed(0)
    # run_test(M=M, N=N, K=K, USE_SWIZZLE=0, PRESHUFFLE_B=0, perf=1, TILEK=TILE_K, permlane_output=PERMLANE_EPILOGUE, store_overlap=STORE_OVERLAP, with_scale = False, B_MXFP4=False)
    run_test(
        M=98304,
        N=512,
        K=6144,
        USE_SWIZZLE=0,
        PRESHUFFLE_B=0,
        perf=1,
        TILEK=TILE_K,
        permlane_output=PERMLANE_EPILOGUE,
        store_overlap=STORE_OVERLAP,
        with_scale=True,
        B_MXFP4=True,
        B_LDS_SWIZZLE=True,
    )

    run_test(
        M=196608,
        N=512,
        K=6144,
        USE_SWIZZLE=0,
        PRESHUFFLE_B=0,
        perf=1,
        TILEK=TILE_K,
        permlane_output=PERMLANE_EPILOGUE,
        store_overlap=STORE_OVERLAP,
        with_scale=True,
        B_MXFP4=True,
        B_LDS_SWIZZLE=True,
    )
    run_test(
        M=393216,
        N=512,
        K=6144,
        USE_SWIZZLE=0,
        PRESHUFFLE_B=0,
        perf=1,
        TILEK=TILE_K,
        permlane_output=PERMLANE_EPILOGUE,
        store_overlap=STORE_OVERLAP,
        with_scale=True,
        B_MXFP4=True,
        B_LDS_SWIZZLE=True,
    )
    # run_test(
    #     M=M,
    #     N=N,
    #     K=K,
    #     USE_SWIZZLE=0,
    #     PRESHUFFLE_B=0,
    #     perf=1,
    #     TILEK=TILE_K,
    #     permlane_output=PERMLANE_EPILOGUE,
    #     store_overlap=STORE_OVERLAP,
    #     with_scale=True,
    #     B_MXFP4=True,
    #     B_LDS_SWIZZLE=True,
    # )

    # run_test(M=M, N=N, K=K, USE_SWIZZLE=0, PRESHUFFLE_B=0, perf=1, TILEK=TILE_K, permlane_output=PERMLANE_EPILOGUE, store_overlap=STORE_OVERLAP, with_scale = False)
    # run_test(M=M, N=N, K=K, USE_SWIZZLE=0, PRESHUFFLE_B=0, perf=1, TILEK=TILE_K, permlane_output=PERMLANE_EPILOGUE, store_overlap=STORE_OVERLAP, with_scale = True)
    # run_test(M=M, N=N, K=K, USE_SWIZZLE=0, PRESHUFFLE_B=0, perf=1, TILEK=TILE_K, permlane_output=PERMLANE_EPILOGUE, store_overlap=STORE_OVERLAP, with_scale = False)
    # run_test(M=M, N=N, K=K, USE_SWIZZLE=1, PRESHUFFLE_B=0, perf=1, TILEK=TILE_K, permlane_output=PERMLANE_EPILOGUE, store_overlap=STORE_OVERLAP)
    # run_test(M=M, N=N, K=K, USE_SWIZZLE=0, PRESHUFFLE_B=1, perf=1, TILEK=TILE_K, permlane_output=PERMLANE_EPILOGUE, store_overlap=STORE_OVERLAP)
    # run_test(M=M, N=N, K=K, USE_SWIZZLE=1, PRESHUFFLE_B=1, perf=1, TILEK=TILE_K, permlane_output=PERMLANE_EPILOGUE, store_overlap=STORE_OVERLAP)