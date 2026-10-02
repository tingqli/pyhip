"""Shared HIP depthwise Conv3D with automatic or explicit backend selection."""

import pyhip

__all__ = ["conv_depthwise_3d"]

_HIP_SOURCE = "hip/conv_depthwise3d_hip.cpp"
_HIP_ARCHS = frozenset({"gfx942", "gfx950"})


def _triple(value):
    if isinstance(value, int):
        return (value, value, value)
    result = tuple(value)
    if len(result) != 3:
        raise ValueError("Conv3D stride, padding, and dilation must have three values")
    return result


def _kernel_config(arch, dtype):
    """Compile-time arithmetic and tile policy; also used by validation tools."""
    import torch

    base_arch = arch.split(":", 1)[0]
    if base_arch not in _HIP_ARCHS or dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"unsupported HIP Conv3D target {arch}, {dtype}")
    bf16_fma = base_arch == "gfx942" and dtype == torch.bfloat16
    return {
        "IO_DTYPE": "__half" if dtype == torch.float16 else "__hip_bfloat16",
        "OUTPUT_TILE": 8 if bf16_fma else 16,
        "BF16_FMA": int(bf16_fma),
    }


def _hip_unavailable_reason(input, weight, bias, stride, padding, dilation, groups):
    """Check the specialization without compiling, allocating, or reading tensors."""
    import torch

    if input.ndim != 5 or weight.ndim != 5:
        return "input and weight must be 5D"
    if input.device.type != "cuda" or torch.version.hip is None:
        return "a ROCm GPU tensor is required"
    if input.dtype not in (torch.float16, torch.bfloat16):
        return "only FP16 and BF16 are supported"
    if weight.device != input.device or weight.dtype != input.dtype:
        return "weight must have the same device and dtype as input"
    if bias is not None and (bias.device != input.device or bias.dtype != input.dtype):
        return "bias must have the same device and dtype as input"
    if (
        not input.is_contiguous()
        or not weight.is_contiguous()
        or (bias is not None and not bias.is_contiguous())
    ):
        return "input, weight, and bias must be contiguous NCDHW tensors"
    if input.data_ptr() % 4:
        return "input must have a dword-aligned base address"
    if torch.is_grad_enabled() and any(
        tensor is not None and tensor.requires_grad for tensor in (input, weight, bias)
    ):
        return "HIP Conv3D is inference-only; use no_grad or the automatic/Torch path"
    if torch.is_autocast_enabled("cuda"):
        return "HIP Conv3D requires autocast to be disabled"

    batch, channels, depth, height, width = input.shape
    out_channels, channels_per_group, kd, kh, kw = weight.shape
    if batch < 1 or channels < 1:
        return "batch and channel counts must be positive"
    if groups != channels or out_channels != channels or channels_per_group != 1:
        return "groups, input channels, and output channels must match"
    if bias is not None and (bias.ndim != 1 or bias.numel() != channels):
        return "bias must have one value per output channel"
    if (kd, kh, kw) != (3, 5, 5):
        return "the filter must be 3x5x5"
    if stride != (1, 1, 1) or dilation != (1, 1, 1) or padding != (0, 2, 2):
        return "stride=(1,1,1), dilation=(1,1,1), padding=(0,2,2) are required"
    if depth < 3 or height < 1 or width < 16 or width % 16:
        return "depth >= 3, height >= 1, and width divisible by 16 are required"
    if max(input.shape) > 2**31 - 1 or channels > 65535 or depth - 2 > 65535:
        return "tensor dimensions exceed the HIP kernel argument or grid limits"
    lds_bytes = (3 * (height + 4) * (width + 4) + 2 * kw) * input.element_size()
    if lds_bytes > 64 * 1024:
        return "the padded spatial tile exceeds 64 KiB of LDS"
    from pyhip.runtime.hiptools import amdgpu_arch

    arch = amdgpu_arch().split(":", 1)[0]
    if arch not in _HIP_ARCHS:
        return f"HIP depthwise Conv3D is unavailable on {arch}"
    return None


def conv_depthwise_3d(
    input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, *, method=None
):
    """Compute Conv3D with a shared HIP depthwise specialization.

    Automatic selection (``method=None``) uses HIP for contiguous FP16/BF16
    NCDHW inference on gfx942/gfx950 with filter 3x5x5, channel multiplier one,
    stride/dilation one, padding (0,2,2), width divisible by 16 and LDS <= 64 KiB.
    Other calls, including gradient-enabled inputs, use Torch. ``method='hip'``
    forces this specialization and raises on unsupported calls; ``'torch'``
    forces Torch. Compilation/loading must be warmed up before graph capture.
    Use one worker process per GPU on hosts with a single GPU architecture,
    following the existing HIP runtime's device and compilation model.
    The historical assembly JIT lives in experiments/conv.
    """
    import torch
    import torch.nn.functional as F

    if method not in (None, "hip", "torch"):
        raise ValueError(
            f"unsupported depthwise Conv3D method {method!r}; use None, 'hip', or 'torch'"
        )
    stride, padding, dilation = map(_triple, (stride, padding, dilation))
    if method == "torch":
        return F.conv3d(input, weight, bias, stride, padding, dilation, groups)
    reason = _hip_unavailable_reason(
        input, weight, bias, stride, padding, dilation, groups
    )
    if reason:
        if method == "hip":
            raise ValueError(f"HIP depthwise Conv3D: {reason}")
        return F.conv3d(input, weight, bias, stride, padding, dilation, groups)

    from pyhip.runtime.hiptools import amdgpu_arch

    arch = amdgpu_arch()
    config = _kernel_config(arch, input.dtype)
    batch, channels, depth, height, width = input.shape
    with torch.cuda.device(input.device):
        output = torch.empty(
            (batch, channels, depth - 2, height, width),
            dtype=input.dtype,
            device=input.device,
        )
        pyhip.module(_HIP_SOURCE, "-O2").conv_depthwise3d_hip(
            [batch, channels, depth - 2],
            [256],
            input.data_ptr(),
            output.data_ptr(),
            weight.data_ptr(),
            bias.data_ptr() if bias is not None else 0,
            channels,
            depth,
            height,
            width,
            channels,
            depth - 2,
            height,
            width,
            BLOCK_H=height,
            BLOCK_W=width,
            PaddingD=0,
            PaddingH=2,
            PaddingW=2,
            KD=3,
            KH=5,
            KW=5,
            **config,
        )
        return output
