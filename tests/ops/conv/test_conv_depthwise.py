"""Public API, numerical, stream and shared-kernel resource regression checks."""

import re
import subprocess
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

import pyhip
from pyhip.ops.conv import conv_depthwise as depthwise
from pyhip.runtime import hiptools

GPU = pytest.mark.skipif(
    torch.version.hip is None or not torch.cuda.is_available(),
    reason="requires a ROCm GPU",
)


def _case(dtype, shape=(1, 2, 5, 7, 32), bias=True, device="cuda"):
    torch.manual_seed(7)
    x = torch.randn(shape, device=device, dtype=dtype) / 8
    w = torch.randn((shape[1], 1, 3, 5, 5), device=device, dtype=dtype) / 8
    b = torch.randn((shape[1],), device=device, dtype=dtype) / 8 if bias else None
    return x, w, b


def _run(x, w, b=None, method="hip"):
    return depthwise.conv_depthwise_3d(
        x, w, b, padding=(0, 2, 2), groups=x.shape[1], method=method
    )


def _require_target():
    arch = hiptools.amdgpu_arch().split(":", 1)[0]
    if arch not in ("gfx942", "gfx950"):
        pytest.skip(f"unsupported target {arch}")
    return arch


def _assert_accuracy(actual, expected, dtype, *, atol=None):
    assert actual.shape == expected.shape and actual.dtype == dtype
    assert torch.isfinite(actual).all()
    assert pyhip.calc_diff(expected, actual) < 1e-4
    torch.testing.assert_close(
        actual,
        expected,
        atol=atol or (0.02 if dtype == torch.bfloat16 else 0.003),
        rtol=0.01 if dtype == torch.bfloat16 else 0.003,
    )


def test_cpu_default_and_torch_preserve_gradients():
    x, w, b = _case(torch.float32, device="cpu")
    x.requires_grad_()
    w.requires_grad_()
    expected = F.conv3d(x, w, b, padding=(0, 2, 2), groups=2)
    actual = _run(x, w, b, method=None)
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    assert x.grad is not None and w.grad is not None
    torch.testing.assert_close(_run(x, w, b, method="torch"), expected)
    with pytest.raises(ValueError, match="ROCm GPU"):
        _run(x, w, b)


@pytest.mark.parametrize("method", ["jit", "sgb", "packed", "auto"])
def test_removed_backend_controls(method):
    x, w, b = _case(torch.float32, device="cpu")
    with pytest.raises(ValueError, match="unsupported.*method"):
        _run(x, w, b, method=method)
    with pytest.raises(TypeError, match="hip_impl"):
        depthwise.conv_depthwise_3d(x, w, b, hip_impl="packed")


@pytest.mark.parametrize(
    "arch,dtype,tile,fma",
    [
        ("gfx942", torch.float16, 16, 0),
        ("gfx942", torch.bfloat16, 8, 1),
        ("gfx950:sramecc+:xnack-", torch.float16, 16, 0),
        ("gfx950", torch.bfloat16, 16, 0),
    ],
)
def test_compile_policy(arch, dtype, tile, fma):
    config = depthwise._kernel_config(arch, dtype)
    assert (config["OUTPUT_TILE"], config["BF16_FMA"]) == (tile, fma)


@GPU
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize(
    "shape", [(1, 2, 3, 1, 16), (2, 3, 6, 9, 48), (1, 2, 5, 45, 80), (1, 2, 5, 80, 80)]
)
def test_forced_hip_accuracy(dtype, bias, shape):
    _require_target()
    x, w, b = _case(dtype, shape, bias)
    expected = F.conv3d(x, w, b, padding=(0, 2, 2), groups=shape[1])
    _assert_accuracy(_run(x, w, b), expected, dtype)
    _assert_accuracy(_run(x, w, b, method=None), expected, dtype)


@GPU
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_unused_lane_nonfinite(dtype):
    _require_target()
    x = torch.ones((1, 1, 3, 3, 16), device="cuda", dtype=dtype)
    w = torch.zeros((1, 1, 3, 5, 5), device="cuda", dtype=dtype)
    w[0, 0, 0, 2, 2] = 1
    for value in (float("inf"), float("nan")):
        x[0, 0, 0, 0, 3] = value
        expected = F.conv3d(x, w, padding=(0, 2, 2))
        actual = _run(x, w)
        torch.testing.assert_close(actual, expected, equal_nan=True, atol=0, rtol=0)


@GPU
@pytest.mark.parametrize("forced_fma", [False, True])
@pytest.mark.parametrize("case", ["range", "cancellation", "nonfinite"])
def test_bf16_fma_and_native_against_reference(forced_fma, case):
    arch = _require_target()
    x, w, b = _case(torch.bfloat16, (2, 2, 5, 5, 32))
    if case == "range":
        x *= 1e10
        w *= 1e-10
    elif case == "cancellation":
        x.fill_(1)
        w.flatten()[::2] = 1
        w.flatten()[1::2] = -1
        b = None
    else:
        x.fill_(1)
        w.zero_()
        w[:, 0, 0, 2, 2] = 1
        x[0, 0, 0, 0, 3] = float("inf")
        b = None
    actual = _run(x, w, b)
    if forced_fma:
        cfg = depthwise._kernel_config(arch, x.dtype)
        cfg.update(BF16_FMA=1, OUTPUT_TILE=8)
        n, c, d, h, width = x.shape
        actual = torch.empty_like(actual)
        src = Path(depthwise.__file__).parent / depthwise._HIP_SOURCE
        pyhip.module(str(src), "-O2").conv_depthwise3d_hip(
            [n, c, d - 2],
            [256],
            x,
            actual,
            w,
            b if b is not None else 0,
            c,
            d,
            h,
            width,
            c,
            d - 2,
            h,
            width,
            BLOCK_H=h,
            BLOCK_W=width,
            PaddingD=0,
            PaddingH=2,
            PaddingW=2,
            KD=3,
            KH=5,
            KW=5,
            **cfg,
        )
    expected = F.conv3d(x, w, b, padding=(0, 2, 2), groups=2)
    if case == "nonfinite":
        torch.testing.assert_close(actual, expected, equal_nan=True, atol=0, rtol=0)
    else:
        _assert_accuracy(actual, expected, x.dtype, atol=0.08)
        reference = F.conv3d(
            x.float(),
            w.float(),
            b.float() if b is not None else None,
            padding=(0, 2, 2),
            groups=2,
        ).to(x.dtype)
        _assert_accuracy(actual, reference, x.dtype, atol=0.08)


@GPU
@pytest.mark.parametrize(
    "unsupported", ["width", "multiplier", "filter", "layout", "lds", "alignment"]
)
def test_fallback_and_forced_rejection(unsupported, monkeypatch):
    x, w, b = _case(torch.bfloat16)
    if unsupported == "width":
        x = x[..., :30].contiguous()
    elif unsupported == "multiplier":
        w, b = w.repeat(2, 1, 1, 1, 1), b.repeat(2)
    elif unsupported == "filter":
        w = w[:, :, :1].contiguous()
    elif unsupported == "layout":
        x = x.transpose(3, 4)
    elif unsupported == "lds":
        x = torch.ones((1, 2, 5, 160, 80), device="cuda", dtype=x.dtype)
    else:
        storage = torch.ones(x.numel() + 1, device="cuda", dtype=x.dtype)
        x = storage[1:].view_as(x)
    monkeypatch.setattr(
        pyhip,
        "module",
        lambda *_args, **_kw: pytest.fail("unexpected HIP fallback launch"),
    )
    expected = F.conv3d(x, w, b, padding=(0, 2, 2), groups=2)
    torch.testing.assert_close(_run(x, w, b, method=None), expected)
    with pytest.raises(ValueError, match="HIP depthwise Conv3D"):
        _run(x, w, b)


@GPU
def test_gradients_and_autocast_use_torch():
    x, w, b = _case(torch.float16)
    x.requires_grad_()
    actual = _run(x, w, b, method=None)
    actual.sum().backward()
    assert x.grad is not None
    with pytest.raises(ValueError, match="inference-only"):
        _run(x, w, b)
    with torch.no_grad():
        _assert_accuracy(
            _run(x, w, b), F.conv3d(x, w, b, padding=(0, 2, 2), groups=2), x.dtype
        )
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        torch.testing.assert_close(
            _run(x, w, b, method=None), F.conv3d(x, w, b, padding=(0, 2, 2), groups=2)
        )
        with pytest.raises(ValueError, match="autocast"):
            _run(x, w, b)


@GPU
def test_short_bias_rejected():
    x, w, b = _case(torch.float16)
    with pytest.raises(ValueError, match="one value per output channel"):
        _run(x, w, b[:1])


@GPU
def test_current_device_stream_and_graph():
    _require_target()
    original = torch.cuda.current_device()
    # Each worker uses one GPU; validate streams and replay on its selected device.
    target = original
    x, w, b = _case(torch.float16, device=f"cuda:{target}")
    stream = torch.cuda.Stream(device=target)
    stream.wait_stream(torch.cuda.current_stream(target))
    with torch.cuda.stream(stream):
        warm = _run(x, w, b)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured = _run(x, w, b)
        x.add_(0.25)
        graph.replay()
    stream.synchronize()
    assert torch.cuda.current_device() == original
    expected = F.conv3d(x, w, b, padding=(0, 2, 2), groups=2)
    _assert_accuracy(captured, expected, x.dtype)
    # Launch again on the worker's original stream after graph replay.
    _assert_accuracy(_run(x, w, b), expected, x.dtype)
    assert warm.device == x.device


def test_offline_arch_resources(tmp_path):
    if subprocess.run(
        ["bash", "-c", "command -v hipcc"], capture_output=True, check=False
    ).returncode:
        pytest.skip("requires HIP compiler")
    source = Path(depthwise.__file__).parent / depthwise._HIP_SOURCE
    for arch in ("gfx942", "gfx950"):
        for dtype in (torch.float16, torch.bfloat16):
            cfg = dict(
                BLOCK_H=45,
                BLOCK_W=80,
                PaddingD=0,
                PaddingH=2,
                PaddingW=2,
                KD=3,
                KH=5,
                KW=5,
                **depthwise._kernel_config(arch, dtype),
            )
            output = tmp_path / f"{arch}-{dtype}.s"
            run = subprocess.run(
                [
                    "hipcc",
                    "-x",
                    "hip",
                    "--offload-device-only",
                    f"--offload-arch={arch}",
                    "-std=c++20",
                    "-O2",
                    "-Rpass-analysis=kernel-resource-usage",
                    *[f"-D{k}={v}" for k, v in cfg.items()],
                    str(source),
                    "-S",
                    "-o",
                    str(output),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            assert run.returncode == 0, run.stderr
            assert re.search(r"ScratchSize \[bytes/lane\]: 0\b", run.stderr), run.stderr
            assert re.search(r"SGPRs Spill: 0\b", run.stderr), run.stderr
            assert re.search(r"VGPRs Spill: 0\b", run.stderr), run.stderr
            assembly = output.read_text()
            if arch == "gfx942" and dtype == torch.bfloat16:
                vgprs = int(re.search(r"VGPRs: (\d+)", run.stderr).group(1))
                assert vgprs <= 96, run.stderr
                assert "v_dot2" not in assembly and "v_fma_f32" in assembly
            else:
                assert "v_dot2" in assembly
