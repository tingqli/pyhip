# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""CDNA4 FP8 blockscale tests for baseline, half-M FIFO and preshuffled B."""

import os
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
flyc = pytest.importorskip("flydsl.compiler")

import pyhip
from pyhip import div_up


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


# =========================== test / perf ===========================
TILE_M = 256
TILE_N = 256
TILE_K = 128
M = int(os.environ.get("GEMM_M", 8192))
N = int(os.environ.get("GEMM_N", 8192))
K = int(os.environ.get("GEMM_K", 8192))
PERMLANE_EPILOGUE = _env_flag("PERMLANE", "1")
SPLIT_M = _env_flag("SPLIT_M", "1")
PRESHUFFLE_B = _env_flag("PRESHUFFLE_B", "0")


def run_test(
    M,
    N,
    K,
    perf=False,
    permlane_output=True,
    preshuffle_b=False,
    with_scale=False,
    run_count=50,
    data_clones=32,
    useTiledDMA=False,
    split_m=False,
):
    """Check a Torch reference and optionally time rotating input/output clones."""
    from pyhip.ops.gemm.flydsl.gemm_fp8_blockscale_8w import compile_gemm_fp8_8wave

    if preshuffle_b:
        from aiter.ops.shuffle import shuffle_weight

    KB = K // 128
    empty = torch.empty(0, device="cuda", dtype=torch.float32)

    def _gen_scales():
        if not with_scale:
            return empty, empty
        sA = torch.rand((M, KB), device="cuda", dtype=torch.float32)
        sB = torch.rand((div_up(N, 128), KB), device="cuda", dtype=torch.float32)
        return sA, sB

    def _ref(a, b, sA, sB):
        if not with_scale:
            return a.float() @ b.float().t()
        a_deq = (a.float().view(M, KB, 128) * sA.view(M, KB, 1)).view(M, K)
        b_deq = (
            b.float().view(N, KB, 128) * sB.repeat_interleave(128, dim=0)[:N, :, None]
        )
        b_deq = b_deq.view(N, K)
        return a_deq @ b_deq.t()

    a = (torch.rand(M, K, device="cuda") / 10.0).to(torch.float8_e4m3fn)
    b = (torch.rand(N, K, device="cuda") / 10.0).to(torch.float8_e4m3fn)
    sA, sB = _gen_scales()
    ref = _ref(a, b, sA, sB)
    b_kernel = shuffle_weight(b, layout=(16, 16)) if preshuffle_b else b
    sA_kernel = sA.transpose(0, 1).contiguous() if with_scale else sA
    out = torch.zeros((M, N), device="cuda", dtype=torch.bfloat16)
    stream = torch.cuda.current_stream()
    args = (
        a.view(torch.int8),
        b_kernel.view(torch.int8),
        out.view(-1),
        sA_kernel.view(-1),
        sB.view(-1),
        M,
        stream,
    )

    launcher = compile_gemm_fp8_8wave(
        TILE_M,
        TILE_N,
        TILE_K,
        N,
        K,
        permlane_epilogue=permlane_output,
        preshuffle_b=preshuffle_b,
        with_scale=with_scale,
        useTileDMA=useTiledDMA,
        split_m=split_m,
    )
    kernel = flyc.compile[{"opt_level": 2}](launcher, *args)
    kernel(*args)
    torch.cuda.synchronize()

    out_f32 = out.float()
    bf16_ref = ref.to(torch.bfloat16)

    # Report both the FP32 reference error and the BF16-rounded comparison.
    finite = bool(torch.isfinite(out_f32).all() & torch.isfinite(ref).all())
    diff_threshold = 0.00001
    diff = pyhip.calc_diff(out_f32, ref) if finite else float("inf")
    diff_bf16ref = (
        pyhip.calc_diff(out_f32, bf16_ref.float()) if finite else float("inf")
    )
    allclose = finite and torch.allclose(out_f32, ref, rtol=0.02, atol=0.01)
    diff_ok = finite and diff <= diff_threshold
    is_correct = diff_ok and allclose
    print(
        f"####M={M} N={N} K={K} 8wave preshuffle_b={preshuffle_b} with_scale={with_scale}, useTiledDMA={useTiledDMA} "
        f"split_m={split_m} "
        f"is_correct={is_correct} finite={finite} allclose={allclose} "
        f"calc_diff(vs f32 ref)={diff:.9g} diff_thr={diff_threshold:.9g} "
        f"diff_ok={diff_ok} calc_diff(vs bf16 ref)={diff_bf16ref:.9g}",
        flush=True,
    )
    if not finite:
        raise AssertionError("non-finite GEMM output or reference")
    pyhip.calc_diff(out_f32, ref, diff_thr=0.00001)
    torch.testing.assert_close(out_f32, ref, rtol=0.02, atol=0.01)

    if not perf:
        return is_correct

    As = [
        torch.randint(-2, 3, (M, K), device="cuda", dtype=torch.int8).to(
            torch.float8_e4m3fn
        )
        for _ in range(data_clones)
    ]
    Bs = [
        torch.randint(-2, 3, (N, K), device="cuda", dtype=torch.int8).to(
            torch.float8_e4m3fn
        )
        for _ in range(data_clones)
    ]
    if preshuffle_b:
        Bs = [shuffle_weight(b, layout=(16, 16)) for b in Bs]
    SAs = [(_gen_scales()[0] if with_scale else empty) for _ in range(data_clones)]
    SAs_kernel = [sa.transpose(0, 1).contiguous() if with_scale else sa for sa in SAs]
    SBs = [(_gen_scales()[1] if with_scale else empty) for _ in range(data_clones)]
    Cs = [
        torch.zeros((M, N), device="cuda", dtype=torch.bfloat16)
        for _ in range(data_clones)
    ]
    arg_sets = [
        (
            As[i].view(torch.int8),
            Bs[i].view(torch.int8),
            Cs[i].view(-1),
            SAs_kernel[i].view(-1),
            SBs[i].view(-1),
            M,
            stream,
        )
        for i in range(data_clones)
    ]
    flops = 2 * M * N * K
    mem_bytes = (M * K + N * K) * 1 + M * N * 2
    if with_scale:
        mem_bytes += (M + div_up(N, 128)) * KB * 4
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
    print(
        f"\n=== perf 8wave M={M} N={N} K={K} {with_scale=} {preshuffle_b=} {split_m=} ==="
    )
    print(
        f"gemm:  {best_ms*1e3:.1f} us  {flops/(best_ms*1e-3)/1e12:.2f} TFLOPS  {mem_bytes/(best_ms*1e-3)/1e9:.1f} GB/s"
    )
    return is_correct


@pytest.mark.usefixtures("cdna4_device")
@pytest.mark.parametrize(
    "M,N,K",
    [(33, 64, 512), (256, 256, 512), (257, 384, 768), (512, 3392, 6144)],
)
@pytest.mark.parametrize("with_scale,split_m", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("useTiledDMA", [False, True])
def test_accuracy(M, N, K, with_scale, split_m, useTiledDMA):
    assert run_test(
        M, N, K, perf=False, with_scale=with_scale,
        useTiledDMA=useTiledDMA, split_m=split_m,
    )


@pytest.mark.usefixtures("cdna4_device")
@pytest.mark.parametrize("M,N,K", [(33, 64, 512), (256, 256, 512), (257, 384, 768)])
@pytest.mark.parametrize("with_scale,split_m", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("useTiledDMA", [False, True])
def test_preshuffle_accuracy(M, N, K, with_scale, split_m, useTiledDMA):
    pytest.importorskip("aiter")
    assert run_test(
        M, N, K, perf=False, preshuffle_b=True,
        with_scale=with_scale, useTiledDMA=useTiledDMA, split_m=split_m,
    )


@pytest.mark.usefixtures("cdna4_device")
@pytest.mark.parametrize("permlane_output", [False, True])
@pytest.mark.parametrize("with_scale,split_m", [(False, False), (True, False), (True, True)])
def test_epilogue(permlane_output, with_scale, split_m):
    assert run_test(
        256, 256, 512, perf=False,
        permlane_output=permlane_output, with_scale=with_scale, split_m=split_m,
    )


@pytest.mark.perf
@pytest.mark.usefixtures("cdna4_device")
@pytest.mark.parametrize(
    "M,N,K",
    [(4096, 4096, 16384), (8192, 8192, 6144), (16384, 3584, 6144), (16384, 3392, 6144)],
)
@pytest.mark.parametrize("split_m", [False, True])
def test_perf(M, N, K, split_m):
    if PRESHUFFLE_B:
        pytest.importorskip("aiter")
    assert run_test(
        M, N, K, perf=True, permlane_output=PERMLANE_EPILOGUE,
        with_scale=True, preshuffle_b=PRESHUFFLE_B, split_m=split_m,
    )


@pytest.fixture
def gemm_factory(monkeypatch):
    from pyhip.ops.gemm.flydsl.gemm_fp8_blockscale_8w import compile_gemm_fp8_8wave

    monkeypatch.setenv("FLYDSL_COMPILE_BACKEND", "rocm")
    monkeypatch.setenv("ARCH", "gfx950")
    monkeypatch.setenv("COMPILE_ONLY", "1")
    compile_gemm_fp8_8wave.cache_clear()
    yield compile_gemm_fp8_8wave
    compile_gemm_fp8_8wave.cache_clear()


@pytest.mark.parametrize(
    "backend,arch", [("rocm", "gfx942"), ("rocm", "gfx1100"), ("cuda", "sm_90")],
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


def test_pipeline_cache(gemm_factory):
    args = (TILE_M, TILE_N, TILE_K, 256, 512)
    launchers = {
        (split_m, preshuffle_b): gemm_factory(
            *args, with_scale=True, split_m=split_m, preshuffle_b=preshuffle_b,
        )
        for split_m in (False, True)
        for preshuffle_b in (False, True)
    }
    assert len({id(launcher) for launcher in launchers.values()}) == 4
    for (split_m, preshuffle_b), launcher in launchers.items():
        assert gemm_factory(
            *args, with_scale=True, split_m=split_m, preshuffle_b=preshuffle_b,
        ) is launcher
    assert gemm_factory.cache_info().hits == 4
    assert gemm_factory.cache_info().currsize == 4


@pytest.mark.parametrize(
    "options",
    [
        pytest.param({"split_m": True}, id="split-without-scale"),
        pytest.param({"split_m": True, "with_scale": True, "TILE_M": 128}, id="split-tile"),
        pytest.param({"preshuffle_b": True, "N": 264}, id="preshuffle-n-alignment"),
        pytest.param({"preshuffle_b": True, "K": 384}, id="preshuffle-k-alignment"),
    ],
)
def test_invalid_pipeline_config(gemm_factory, options):
    config = dict(TILE_M=TILE_M, TILE_N=TILE_N, TILE_K=TILE_K, N=256, K=512)
    config.update(options)
    with pytest.raises(AssertionError):
        gemm_factory(**config)
    assert gemm_factory.cache_info().currsize == 0


if __name__ == "__main__":
    if torch.version.hip is None or not torch.cuda.is_available():
        raise RuntimeError("FP8 GEMM requires a ROCm CDNA4 (gfx950) GPU")
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    if props.gcnArchName.split(":", 1)[0] != "gfx950":
        raise RuntimeError(f"FP8 GEMM requires CDNA4 (gfx950); got {props.gcnArchName}")
    torch.manual_seed(0)
    
    run_test(
        M=4096, N=4096, K=16384, perf=True,
        permlane_output=PERMLANE_EPILOGUE, preshuffle_b=PRESHUFFLE_B,
        with_scale=True, split_m=SPLIT_M,
    )