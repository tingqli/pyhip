# SPDX-License-Identifier: MIT
"""MoE quantization policies using Torch references or HIP activations.

    quant = getattr(quantizer, "fp8_ptpc")
    aq, a_scale = quant.apply_a(a, backend="hip")  # runtime BF16/FP16 on ROCm
    wq, w_scale = quant.apply_w(w)       # w: [N, K] or [experts, N, K]
    a_ref = quant.dequant_a(aq, a_scale) # FP32, for untimed reference work

All results use natural row-major order, never sorting, padding or shuffling.
MXFP4 packs even/odd K elements into the low/high nibble; INT4 uses an UNPACKED
int8 container with values in [-7, 7]. Kernel-specific packing belongs to the
adapter. Scales multiply quantized values to reconstruct the input domain.

Default backend="torch" accepts finite FP16/BF16/FP32 on CPU/GPU; HIP requires
contiguous FP16/BF16 on ROCm. Per-tensor activation quantization always uses
Torch. apply_* checks metadata without reading values; validate_input() is an
optional untimed check. Each backend retains its rounding and zero/tiny
behavior without local repairs.
Importing this module does not import Aiter or initialize a GPU.
no_quant is the exact identity exception: it preserves the input object, dtype,
strides and autograd state; dequant_* still returns detached FP32 reference data.
"""

from __future__ import annotations

from functools import partial

import torch

__all__ = [
    "Quantizer", "get_quantizer", "validate_input",
    "no_quant", "bf16", "fp8_ptpc", "fp8_per_tensor", "fp8_blockscale",
    "a16w8_per_channel", "a16w8_per_tensor", "a16w8_blockscale",
    "fp8_per_token_per_tensor", "a16w4", "a8w4", "a4w4",
    "int8_ptpc", "int8_smoothquant", "fp8_int4_ptpc",
]

Quantized = tuple[torch.Tensor, torch.Tensor | None]
_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e4m3fnuz)


def _source(x: torch.Tensor) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        raise TypeError("expected a torch.Tensor")
    if x.layout != torch.strided or x.ndim < 2 or x.shape[-1] == 0:
        raise ValueError("expected a strided tensor of rank >= 2 with nonempty K")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("quantization input must be FP16, BF16 or FP32, not already quantized")
    return x.detach()


def _options(fp8_dtype, smooth_scale):
    if fp8_dtype is not None and fp8_dtype not in _FP8_DTYPES:
        raise ValueError("fp8_dtype must be E4M3FN or E4M3FNUZ")
    if smooth_scale is not None:
        raise ValueError("smooth_scale is only accepted by int8_smoothquant")


def _fp8_dtype(x, dtype):
    if dtype is not None:
        return dtype
    if x.device.type == "cpu" or (x.device.type == "cuda" and torch.version.hip is None):
        return torch.float8_e4m3fn
    if x.device.type == "cuda":
        arch = torch.cuda.get_device_properties(x.device).gcnArchName.split(":")[0]
        if arch == "gfx942":
            return torch.float8_e4m3fnuz
        if arch in ("gfx950", "gfx1250"):
            return torch.float8_e4m3fn
    raise NotImplementedError("unknown device FP8 format; pass fp8_dtype explicitly")


def _per_token(x, dtype, *, dtype_max=None, x_scale=None):
    from aiter.ops.quant import pertoken_quant

    q, scale = pertoken_quant(x, quant_dtype=dtype, dtypeMax=dtype_max, x_scale=x_scale)
    return q.contiguous(), scale.contiguous()


def _hip_input(x):
    x = _source(x)
    # These restrictions must precede C++: unsupported dtypes can abort there.
    if x.device.type != "cuda" or torch.version.hip is None:
        raise ValueError("HIP activation quantization requires a ROCm GPU")
    if x.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("HIP activation quantization requires FP16/BF16; use backend='torch' for FP32")
    if not x.is_contiguous() or x.shape[-1] % 32:
        raise ValueError("HIP activation quantization requires contiguous input and K divisible by 32")
    return x


def _hip_quant(x, *, mode, quant_dtype=None, fp8_dtype=None, smooth_scale=None, transpose_scale=False):
    _options(fp8_dtype, smooth_scale)
    x = _hip_input(x)
    from aiter import dtypes
    from aiter.ops.enum import QuantType
    from aiter.ops.quant import get_hip_quant

    dtype = _fp8_dtype(x, fp8_dtype) if quant_dtype is None else quant_dtype
    if dtype in _FP8_DTYPES and dtype != dtypes.fp8:
        raise ValueError("HIP quantization requires the native FP8 format; use backend='torch' for cross-format references")
    if transpose_scale and mode != "per_1x128":
        raise ValueError("transposed activation scales require fp8_blockscale")
    group = 128 if mode == "per_1x128" else 32 if mode == "per_1x32" else 1
    if x.shape[-1] % group:
        raise ValueError(f"HIP activation quantization requires K divisible by {group}")
    q_shape = (*x.shape[:-1], x.shape[-1] // (2 if dtype == torch.float4_e2m1fn_x2 else 1))
    scale_shape = (*x.shape[:-1], x.shape[-1] // group if group > 1 else 1)
    scale_dtype = torch.float8_e8m0fnu if mode == "per_1x32" else torch.float32
    if x.numel() == 0:
        return (torch.empty(q_shape, dtype=dtype, device=x.device),
                torch.zeros(scale_shape, dtype=scale_dtype, device=x.device))
    quant = get_hip_quant(getattr(QuantType, mode))
    if mode == "per_1x32":
        q, scale = quant(x.view(-1, x.shape[-1]), quant_dtype=dtype,
                         scale_type=scale_dtype, shuffle=False)
        return q.view(q_shape), scale.view(scale_shape)
    options = {"transpose_scale": transpose_scale} if mode == "per_1x128" else {}
    return quant(x, quant_dtype=dtype, **options)


def _bf16(x: torch.Tensor, *, fp8_dtype=None, smooth_scale=None) -> Quantized:
    _options(fp8_dtype, smooth_scale)
    return _source(x).to(torch.bfloat16).contiguous(), None


def _identity(x: torch.Tensor, *, fp8_dtype=None, smooth_scale=None) -> Quantized:
    _options(fp8_dtype, smooth_scale)
    _source(x)
    return x, None


def _fp8_rows(x: torch.Tensor, *, fp8_dtype=None, smooth_scale=None) -> Quantized:
    _options(fp8_dtype, smooth_scale)
    x = _source(x)
    return _per_token(x, _fp8_dtype(x, fp8_dtype))


def _fp8_tensor_a(x: torch.Tensor, *, fp8_dtype=None, smooth_scale=None) -> Quantized:
    _options(fp8_dtype, smooth_scale)
    x = _source(x)
    from aiter.ops.quant import per_tensor_quant

    # Aiter cannot reduce an empty tensor; supply its otherwise unused scale.
    scale = torch.zeros(1, dtype=torch.float32, device=x.device) if x.numel() == 0 else None
    q, scale = per_tensor_quant(x, scale=scale, quant_dtype=_fp8_dtype(x, fp8_dtype))
    return q.contiguous(), scale


def _fp8_tensor_w(x: torch.Tensor, *, fp8_dtype=None, smooth_scale=None) -> Quantized:
    _options(fp8_dtype, smooth_scale)
    x = _source(x)
    if x.shape[-2] == 0:
        raise ValueError("per-expert tensor quantization requires nonempty N")
    # Aiter's MoE weight preparation treats each expert matrix as one row.
    q, scale = _per_token(x.flatten(-2), _fp8_dtype(x, fp8_dtype))
    return q.reshape(x.shape), scale.unsqueeze(-1)


def _fp8_group_a(x: torch.Tensor, *, fp8_dtype=None, smooth_scale=None) -> Quantized:
    _options(fp8_dtype, smooth_scale)
    x = _source(x)
    if x.shape[-1] % 128:
        raise ValueError("FP8 blockscale activation requires K divisible by 128")
    grouped = x.reshape(*x.shape[:-1], x.shape[-1] // 128, 128)
    q, scale = _per_token(grouped, _fp8_dtype(x, fp8_dtype))
    return q.reshape(x.shape), scale.squeeze(-1)


def _weight_blocks(x):
    n, k = x.shape[-2:]
    if n == 0 or n % 128 or k % 128:
        raise ValueError("FP8 blockscale weight requires positive N/K divisible by 128")
    return x.reshape(*x.shape[:-2], n // 128, 128, k // 128, 128).transpose(-3, -2)


def _fp8_block_w(x: torch.Tensor, *, fp8_dtype=None, smooth_scale=None) -> Quantized:
    _options(fp8_dtype, smooth_scale)
    x = _source(x)
    blocks = _weight_blocks(x)
    q, scale = _per_token(blocks.flatten(-2), _fp8_dtype(x, fp8_dtype))
    q = q.reshape(blocks.shape).transpose(-3, -2).contiguous().reshape(x.shape)
    return q, scale.squeeze(-1)


def _mx_input(x, fp8_dtype, smooth_scale):
    _options(fp8_dtype, smooth_scale)
    x = _source(x).contiguous()
    if x.shape[-1] % 32:
        raise ValueError("MX quantization requires K divisible by 32; padding is the caller's responsibility")
    return x


def _mxfp4(x: torch.Tensor, *, fp8_dtype=None, smooth_scale=None) -> Quantized:
    x = _mx_input(x, fp8_dtype, smooth_scale)
    if x.numel() == 0:
        # Aiter's reference uses view(..., -1), ambiguous for empty outer dims.
        q = torch.empty((*x.shape[:-1], x.shape[-1] // 2), dtype=torch.float4_e2m1fn_x2, device=x.device)
        scale = torch.empty((*x.shape[:-1], x.shape[-1] // 32), dtype=torch.float8_e8m0fnu, device=x.device)
        return q, scale
    from aiter.ops.quant import per_1x32_f4_quant

    q, scale = per_1x32_f4_quant(x, shuffle=False)
    return q.contiguous(), scale.reshape(*x.shape[:-1], x.shape[-1] // 32).contiguous()


def _mxfp8(x: torch.Tensor, *, fp8_dtype=None, smooth_scale=None) -> Quantized:
    x = _mx_input(x, fp8_dtype, smooth_scale)
    dtype = _fp8_dtype(x, fp8_dtype)
    scale_shape = (*x.shape[:-1], x.shape[-1] // 32)
    if x.numel() == 0:
        return torch.empty_like(x, dtype=dtype), torch.empty(scale_shape, dtype=torch.float8_e8m0fnu, device=x.device)
    from aiter import dtypes
    from aiter.ops.quant import per_1x32_f8_scale_f8_quant

    if dtype == dtypes.fp8:
        q, scale = per_1x32_f8_scale_f8_quant(x, quant_dtype=dtype, scale_type=dtypes.fp8_e8m0, shuffle=False)
        return q.contiguous(), scale.reshape(scale_shape).contiguous()

    # The full Aiter wrapper accepts only the host's native FP8 format.
    # Explicit cross-format references use its dtype-aware scale helper.
    from aiter.utility.fp4_utils import e8m0_to_f32, f32_to_mx_e8m0_scale
    from aiter.utility.mx_types import MxDtypeInt

    mx_dtype = MxDtypeInt.FP8_E4M3_FNUZ if dtype == torch.float8_e4m3fnuz else MxDtypeInt.FP8_E4M3
    groups = x.float().reshape(*x.shape[:-1], x.shape[-1] // 32, 32)
    scale = f32_to_mx_e8m0_scale(groups.abs().amax(dim=-1), dtype=mx_dtype)
    q = (groups / e8m0_to_f32(scale).unsqueeze(-1)).to(dtype)
    return q.reshape(x.shape), scale.contiguous()


def _int8(x: torch.Tensor, *, fp8_dtype=None, smooth_scale=None) -> Quantized:
    _options(fp8_dtype, smooth_scale)
    return _per_token(_source(x), torch.int8)


def _smooth_scale(x, smooth_scale):
    if smooth_scale is None:
        raise ValueError("int8_smoothquant requires an explicit smooth_scale (ones for no smoothing)")
    if not isinstance(smooth_scale, torch.Tensor):
        raise TypeError("smooth_scale must be a torch.Tensor")
    if smooth_scale.layout != torch.strided or smooth_scale.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("smooth_scale must be a strided FP16, BF16 or FP32 tensor")
    if smooth_scale.device != x.device:
        raise ValueError("smooth_scale must be on the input device")
    try:
        shape = torch.broadcast_shapes(x.shape, smooth_scale.shape)
    except RuntimeError as error:
        raise ValueError("smooth_scale must broadcast to, not expand, the input shape") from error
    if shape != x.shape:
        raise ValueError("smooth_scale must not expand input dimensions; gather expert scales before quantizing")
    return smooth_scale.detach().float()


def _smooth_a(x: torch.Tensor, *, fp8_dtype=None, smooth_scale=None) -> Quantized:
    _options(fp8_dtype, None)
    x = _source(x)
    return _per_token(x, torch.int8, x_scale=_smooth_scale(x, smooth_scale))


def _smooth_a_hip(x, *, fp8_dtype=None, smooth_scale=None, transpose_scale=False):
    _options(fp8_dtype, None)
    x = _hip_input(x)
    smooth = _smooth_scale(x, smooth_scale)
    if transpose_scale:
        raise ValueError("transposed activation scales require fp8_blockscale")
    k = x.shape[-1]
    if k > 8192:
        raise ValueError("Aiter fused SmoothQuant requires K <= 8192")
    q = torch.empty_like(x, dtype=torch.int8)
    scale = torch.empty((*x.shape[:-1], 1), dtype=torch.float32, device=x.device)
    if x.numel():
        from aiter.ops.quant import smooth_per_token_scaled_quant

        mapping = None
        if smooth.numel() in (1, k):
            smooth = smooth.reshape(-1).expand(k).contiguous()
        else:
            smooth = smooth.expand(x.shape).contiguous().view(-1, k)
            mapping = torch.arange(x.numel() // k, dtype=torch.int32, device=x.device)
        smooth_per_token_scaled_quant(q.view(-1, k), x.view(-1, k), scale, smooth,
                                      smooth_scale_map=mapping)
    return q, scale


def _smooth_w(x: torch.Tensor, *, fp8_dtype=None, smooth_scale=None) -> Quantized:
    _options(fp8_dtype, None)
    x = _source(x).float()
    return _per_token(x / _smooth_scale(x, smooth_scale), torch.int8)


def _int8_output(x: torch.Tensor, *, fp8_dtype=None, smooth_scale=None, backend="torch") -> Quantized:
    _options(fp8_dtype, smooth_scale)
    x = _hip_input(x) if backend == "hip" else _source(x).contiguous()
    if x.shape[-1] % 32:
        raise ValueError("INT8 output quantization requires N divisible by 32")
    groups = x.reshape(*x.shape[:-1], x.shape[-1] // 32, 32)
    if backend == "hip":
        q, scale = _hip_quant(groups, mode="per_Token", quant_dtype=torch.int8)
    elif backend == "torch":
        q, scale = _per_token(groups, torch.int8)
    else:
        raise ValueError("backend must be 'torch' or 'hip'")
    return q.reshape(x.shape), scale.squeeze(-1)


def _int4_w(x: torch.Tensor, *, fp8_dtype=None, smooth_scale=None) -> Quantized:
    _options(fp8_dtype, smooth_scale)
    return _per_token(_source(x), torch.int8, dtype_max=7)


def _check_scale(q, scale, shape, *, mx=False):
    if not isinstance(q, torch.Tensor) or q.ndim < 2:
        raise ValueError("expected a rank >= 2 quantized tensor")
    if not isinstance(scale, torch.Tensor) or tuple(scale.shape) != tuple(shape):
        raise ValueError(f"expected scale shape {tuple(shape)}")
    dtype = torch.float8_e8m0fnu if mx else torch.float32
    if scale.dtype != dtype or scale.device != q.device:
        raise ValueError(f"expected {dtype} scale on the quantized tensor's device")


def _dequant_bf16(q, scale):
    if q.dtype != torch.bfloat16 or scale is not None:
        raise ValueError("unquantized policy expects BF16 and scale=None")
    return q.float()


def _dequant_identity(q, scale):
    if scale is not None:
        raise ValueError("no_quant expects scale=None")
    return _source(q).float()


def _dequant_rows(q, scale):
    _check_scale(q, scale, (*q.shape[:-1], 1))
    return q.float() * scale


def _dequant_tensor_a(q, scale):
    _check_scale(q, scale, (1,))
    return q.float() * scale


def _dequant_tensor_w(q, scale):
    _check_scale(q, scale, (*q.shape[:-2], 1, 1))
    return q.float() * scale


def _dequant_group(q, scale, group_size):
    if q.shape[-1] % group_size:
        raise ValueError("quantized K does not match group size")
    _check_scale(q, scale, (*q.shape[:-1], q.shape[-1] // group_size))
    groups = q.float().reshape(*q.shape[:-1], q.shape[-1] // group_size, group_size)
    return (groups * scale.unsqueeze(-1)).reshape(q.shape)


def _dequant_group128(q, scale):
    return _dequant_group(q, scale, 128)


def _dequant_block_w(q, scale):
    blocks = _weight_blocks(q.float())
    _check_scale(q, scale, (*q.shape[:-2], q.shape[-2] // 128, q.shape[-1] // 128))
    return (blocks * scale[..., None, None]).transpose(-3, -2).contiguous().reshape(q.shape)


def _dequant_mx(q, scale):
    from aiter.utility.fp4_utils import e8m0_to_f32, mxfp4_to_f32

    if q.dtype == torch.float4_e2m1fn_x2:
        values = mxfp4_to_f32(q)
    elif q.dtype in _FP8_DTYPES:
        values = q.float()
    else:
        raise ValueError("expected packed MXFP4 or MXFP8 storage")
    if values.shape[-1] % 32:
        raise ValueError("MX logical K must be divisible by 32")
    _check_scale(q, scale, (*values.shape[:-1], values.shape[-1] // 32), mx=True)
    values = values.reshape(*values.shape[:-1], values.shape[-1] // 32, 32)
    return (values * e8m0_to_f32(scale).unsqueeze(-1)).flatten(-2)


class Quantizer:
    """BF16 identity policy; subclasses replace executable methods, not tags.

    apply_a/apply_w return (q, scale). Optional fp8_dtype controls only sides
    producing FP8; smooth_scale is accepted only by int8_smoothquant. Leading
    dimensions are preserved. Identity may alias the input; no method mutates it.
    dequant_a/dequant_w return FP32 values in the quantized domain (after any
    smoothing); they do not undo smooth scales or restore routing/layouts.
    apply_a selects backend="torch" or "hip" where available; per-tensor A and
    apply_w always use Torch. Only HIP blockscale allows scale transpose.
    """

    _apply_a = staticmethod(_bf16)
    _apply_a_hip = None
    apply_w = staticmethod(_bf16)
    dequant_a = staticmethod(_dequant_bf16)
    dequant_w = staticmethod(_dequant_bf16)

    def apply_a(self, x, *, fp8_dtype=None, smooth_scale=None, backend="torch", transpose_scale=False) -> Quantized:
        if backend not in ("torch", "hip"):
            raise ValueError("backend must be 'torch' or 'hip'")
        if backend == "hip" and self._apply_a_hip is not None:
            return self._apply_a_hip(x, fp8_dtype=fp8_dtype, smooth_scale=smooth_scale,
                                     transpose_scale=transpose_scale)
        if transpose_scale:
            raise ValueError("transpose_scale requires HIP fp8_blockscale")
        return self._apply_a(x, fp8_dtype=fp8_dtype, smooth_scale=smooth_scale)


class _NoQuant(Quantizer):
    """Return the input itself and None: no cast, copy, or detach."""
    _apply_a = staticmethod(_identity)
    apply_w = staticmethod(_identity)
    dequant_a = staticmethod(_dequant_identity)
    dequant_w = staticmethod(_dequant_identity)


class _FP8PTPC(Quantizer):
    """A: one FP32 scale per row; W: one per expert/output channel."""
    _apply_a = staticmethod(_fp8_rows)
    _apply_a_hip = staticmethod(partial(_hip_quant, mode="per_Token"))
    apply_w = staticmethod(_fp8_rows)
    dequant_a = staticmethod(_dequant_rows)
    dequant_w = staticmethod(_dequant_rows)


class _FP8PerTensor(Quantizer):
    """Torch only. A: one scale per tensor; W: one per leading expert matrix."""
    _apply_a = staticmethod(_fp8_tensor_a)
    apply_w = staticmethod(_fp8_tensor_w)
    dequant_a = staticmethod(_dequant_tensor_a)
    dequant_w = staticmethod(_dequant_tensor_w)


class _FP8Blockscale(Quantizer):
    """A: FP32 scales [..., K/128]; W: [..., N/128, K/128]. No transpose."""
    _apply_a = staticmethod(_fp8_group_a)
    _apply_a_hip = staticmethod(partial(_hip_quant, mode="per_1x128"))
    apply_w = staticmethod(_fp8_block_w)
    dequant_a = staticmethod(_dequant_group128)
    dequant_w = staticmethod(_dequant_block_w)


class _A16W8PerChannel(Quantizer):
    apply_w = staticmethod(_fp8_rows)
    dequant_w = staticmethod(_dequant_rows)


class _A16W8PerTensor(Quantizer):
    apply_w = staticmethod(_fp8_tensor_w)
    dequant_w = staticmethod(_dequant_tensor_w)


class _A16W8Blockscale(Quantizer):
    apply_w = staticmethod(_fp8_block_w)
    dequant_w = staticmethod(_dequant_block_w)


class _FP8PerTokenPerTensor(_FP8PTPC):
    """Mixed policy: FP8 A per-token, FP8 W per-expert tensor."""
    apply_w = staticmethod(_fp8_tensor_w)
    dequant_w = staticmethod(_dequant_tensor_w)


class _A16W4(Quantizer):
    apply_w = staticmethod(_mxfp4)
    dequant_w = staticmethod(_dequant_mx)


class _A8W4(_A16W4):
    """MXFP8 A / MXFP4 W, both K-group32 E8M0 with Aiter's default scaling."""
    _apply_a = staticmethod(_mxfp8)
    _apply_a_hip = staticmethod(partial(_hip_quant, mode="per_1x32"))
    dequant_a = staticmethod(_dequant_mx)


class _A4W4(_A16W4):
    _apply_a = staticmethod(_mxfp4)
    _apply_a_hip = staticmethod(partial(_hip_quant, mode="per_1x32", quant_dtype=torch.float4_e2m1fn_x2))
    dequant_a = staticmethod(_dequant_mx)


class _INT8PTPC(Quantizer):
    """Aiter per-token INT8: divisor 127, truncation, zero scale replaced by 1."""
    _apply_a = staticmethod(_int8)
    _apply_a_hip = staticmethod(partial(_hip_quant, mode="per_Token", quant_dtype=torch.int8))
    apply_w = staticmethod(_int8)
    dequant_a = staticmethod(_dequant_rows)
    dequant_w = staticmethod(_dequant_rows)


class _INT8SmoothQuant(_INT8PTPC):
    """Quantize A*s and W/s; caller gathers per-expert A scales using routes.

    Shared scale: [K] or [1,K]. Expert W scale: [E,1,K]. Routed A scale:
    [tokens,topk,K]. Positive finite scales required; no routing is implicit.
    apply_output optionally quantizes already route-weighted Down values in
    N-groups of32. It is separate from A/W policy, never enabled implicitly.
    """
    _apply_a = staticmethod(_smooth_a)
    _apply_a_hip = staticmethod(_smooth_a_hip)
    apply_w = staticmethod(_smooth_w)
    apply_output = staticmethod(_int8_output)

    @staticmethod
    def dequant_output(q, scale):
        return _dequant_group(q, scale, 32)


class _FP8INT4PTPC(_FP8PTPC):
    """FP8 A per-token, signed INT4 W per-channel in int8 storage, RTZ."""
    apply_w = staticmethod(_int4_w)


no_quant = _NoQuant()
bf16 = Quantizer()
fp8_ptpc = _FP8PTPC()
fp8_per_tensor = _FP8PerTensor()
fp8_blockscale = _FP8Blockscale()
a16w8_per_channel = _A16W8PerChannel()
a16w8_per_tensor = _A16W8PerTensor()
a16w8_blockscale = _A16W8Blockscale()
fp8_per_token_per_tensor = _FP8PerTokenPerTensor()
a16w4 = _A16W4()
a8w4 = _A8W4()
a4w4 = _A4W4()
int8_ptpc = _INT8PTPC()
int8_smoothquant = _INT8SmoothQuant()
fp8_int4_ptpc = _FP8INT4PTPC()


def get_quantizer(name: str) -> Quantizer:
    """Resolve public names, accepting config spelling fp8-ptpc as fp8_ptpc."""
    if not isinstance(name, str):
        raise TypeError("quantizer name must be a string")
    key = name.replace("-", "_")
    quant = globals().get(key) if key in __all__ else None
    if not isinstance(quant, Quantizer):
        raise ValueError(f"unknown quantizer: {name!r}")
    return quant


def validate_input(x: torch.Tensor, *, smooth_scale: torch.Tensor | None = None) -> None:
    """Optional value validation OUTSIDE timing/capture; synchronizes on GPU."""
    _source(x)
    if not bool(torch.isfinite(x).all()):
        raise ValueError("input must contain only finite values")
    if smooth_scale is not None:
        _smooth_scale(x, smooth_scale)  # dtype/device/broadcast metadata
        if not bool((torch.isfinite(smooth_scale) & (smooth_scale > 0)).all()):
            raise ValueError("smooth_scale must contain only positive finite values")