"""Correctness and dispatch checks for the packaged depthwise Conv3D kernel."""

import pytest
import torch
import torch.nn.functional as F

import pyhip
from pyhip.ops.conv import conv_depthwise as depthwise


pytestmark = pytest.mark.skipif(
    torch.version.hip is None or not torch.cuda.is_available(),
    reason="depthwise HIP tests require a ROCm GPU",
)


def _case(dtype, *, width=32, bias=True, out_multiplier=1):
    torch.manual_seed(7)
    x = torch.randn((1, 2, 5, 7, width), device="cuda", dtype=dtype) / 8
    w = torch.randn((2 * out_multiplier, 1, 3, 5, 5), device="cuda", dtype=dtype) / 8
    b = torch.randn((2 * out_multiplier,), device="cuda", dtype=dtype) / 8 if bias else None
    return x, w, b


def _run(x, w, b, hip_impl="auto"):
    return depthwise.conv_depthwise_3d(
        x, w, b, (1, 1, 1), (0, 2, 2), (1, 1, 1), 2,
        method="hip", hip_impl=hip_impl,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("with_bias", [False, True])
def test_packed_matches_torch(dtype, with_bias):
    x, w, b = _case(dtype, bias=with_bias)
    reason = depthwise._packed_unavailable_reason(
        x, w, b, (1, 1, 1), (0, 2, 2), (1, 1, 1), 2
    )
    if reason:
        pytest.skip(reason)

    expected = F.conv3d(x, w, b, padding=(0, 2, 2), groups=2)
    actual = _run(x, w, b, hip_impl="packed")
    assert actual.shape == expected.shape
    assert actual.dtype == dtype
    assert torch.isfinite(actual).all()
    assert pyhip.calc_diff(expected, actual) < 1e-4
    max_abs = (expected - actual).abs().max().item()
    assert max_abs < (0.02 if dtype == torch.bfloat16 else 0.003)


def test_packed_multiple_batches_and_wider_tile():
    torch.manual_seed(19)
    x = torch.randn((2, 3, 6, 9, 48), device="cuda", dtype=torch.float16) / 8
    w = torch.randn((3, 1, 3, 5, 5), device="cuda", dtype=torch.float16) / 8
    b = torch.randn((3,), device="cuda", dtype=torch.float16) / 8
    if depthwise._packed_unavailable_reason(
        x, w, b, (1, 1, 1), (0, 2, 2), (1, 1, 1), 3
    ):
        pytest.skip("packed FP16 is unavailable on this GPU")
    expected = F.conv3d(x, w, b, padding=(0, 2, 2), groups=3)
    actual = depthwise.conv_depthwise_3d(
        x, w, b, (1, 1, 1), (0, 2, 2), (1, 1, 1), 3, hip_impl="packed"
    )
    assert pyhip.calc_diff(expected, actual) < 1e-4
    assert (expected - actual).abs().max().item() < 0.003


def test_auto_uses_torch_for_unsupported_width(monkeypatch):
    x, w, b = _case(torch.float16, width=30)
    monkeypatch.setattr(pyhip, "module", lambda *_args: pytest.fail("packed kernel was selected"))
    expected = F.conv3d(x, w, b, padding=(0, 2, 2), groups=2)
    torch.testing.assert_close(_run(x, w, b), expected)


def test_auto_uses_torch_for_depthwise_multiplier(monkeypatch):
    x, w, b = _case(torch.float16, out_multiplier=2)
    monkeypatch.setattr(pyhip, "module", lambda *_args: pytest.fail("packed kernel was selected"))
    expected = F.conv3d(x, w, b, padding=(0, 2, 2), groups=2)
    torch.testing.assert_close(_run(x, w, b), expected)


def test_auto_uses_torch_for_bf16_on_gfx942(monkeypatch):
    x, w, b = _case(torch.bfloat16)
    monkeypatch.setattr(depthwise, "_device_arch", lambda _x: "gfx942")
    monkeypatch.setattr(pyhip, "module", lambda *_args: pytest.fail("packed kernel was selected"))
    expected = F.conv3d(x, w, b, padding=(0, 2, 2), groups=2)
    torch.testing.assert_close(_run(x, w, b), expected)


def test_forced_packed_reports_unsupported_shape():
    x, w, b = _case(torch.float16, width=30)
    with pytest.raises(ValueError, match="width divisible by 16"):
        _run(x, w, b, hip_impl="packed")


def test_forced_packed_rejects_short_bias():
    x, w, b = _case(torch.float16)
    with pytest.raises(ValueError, match="one value per output channel"):
        _run(x, w, b[:1], hip_impl="packed")


def test_assembly_jit_rejects_unsupported_layout():
    x, w, b = _case(torch.bfloat16)
    with pytest.raises(ValueError, match="BF16 depthwise layout"):
        depthwise.conv_depthwise_3d(
            x, w, b, (2, 1, 1), (0, 2, 2), (1, 1, 1), 2, method="jit"
        )


@pytest.mark.parametrize("removed_impl", ["original", "sgb"])
def test_removed_hip_implementations_are_rejected(removed_impl):
    x, w, b = _case(torch.float16)
    with pytest.raises(ValueError, match="old HIP kernels were removed"):
        _run(x, w, b, hip_impl=removed_impl)
