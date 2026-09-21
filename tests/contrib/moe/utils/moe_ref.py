# SPDX-License-Identifier: MIT
"""Unsorted Torch MoE reference, with executable quantizer policies.

``get(...)`` binds model/numerical options and returns
``op(hidden_states, weight1, weight2, topk_weight, topk_ids, ...)``.
Weights are local TP shards in natural [expert, output, input] order. Pass
ordinary floating weights to quantize them, or quantizer-produced weights and
their w1_scale/w2_scale to dequantize without quantizing again.

Both GEMMs, activation, route weighting and TOPK reduction use FP32. Only the
selected quantizer and explicit intermediate/route/output dtypes round values.
This is an untimed, inference-only oracle, not a graph-capturable GPU kernel.
Import/factory construction does not load Aiter or initialize a GPU.
"""

from __future__ import annotations

from contextlib import contextmanager
import math

import torch
import torch.nn.functional as F

if __package__:
    from . import quantizer
else:
    import quantizer

__all__ = ["get", "calc_diff"]

_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e4m3fnuz)
_QUANT_DTYPES = (*_FP8_DTYPES, torch.int8, torch.float4_e2m1fn_x2)


def calc_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    """PyHIP's 1 - 2*sum(x*y)/sum(x*x+y*y), reduced in FP64; zero/zero = 0.

    Unlike the legacy diagnostic printer, reject nonfinite results explicitly.
    This is neither a maximum relative error nor a fraction of bad elements.
    """
    if x.shape != y.shape or x.device != y.device:
        raise ValueError("calc_diff inputs must have the same shape and device")
    if not x.is_floating_point() or not y.is_floating_point():
        raise TypeError("calc_diff expects floating-point results")
    x, y = x.detach().double(), y.detach().double()
    if not bool(torch.isfinite(x).all() & torch.isfinite(y).all()):
        raise ValueError("calc_diff inputs must be finite")
    denominator = (x * x + y * y).sum()
    if denominator.item() == 0:
        return 0.0
    return (1 - 2 * (x * y).sum() / denominator).item()


@contextmanager
def _fp32_compute(device_type):
    # Restore caller state even on failure. Torch's matmul-precision setting is
    # process-global: this untimed reference must not run beside timed threads.
    precision = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("highest")
        with torch.autocast(device_type=device_type, enabled=False):
            yield
    finally:
        torch.set_float32_matmul_precision(precision)


def _tensor(name, value, shape, device, dtypes):
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.layout != torch.strided or tuple(value.shape) != tuple(shape):
        raise ValueError(f"{name} must have strided shape {tuple(shape)}")
    if value.dtype not in dtypes:
        raise TypeError(f"{name} has unsupported dtype {value.dtype}")
    if value.device != device:
        raise ValueError(f"{name} must be on {device}")


def _weight_metadata(name, value, scale, shape, device, quantize_weights):
    packed = isinstance(value, torch.Tensor) and value.dtype == torch.float4_e2m1fn_x2
    storage_shape = (*shape[:-1], shape[-1] // 2) if packed else shape
    if packed and shape[-1] % 2:
        raise ValueError(f"{name} has an odd logical K for packed MXFP4")
    _tensor(name, value, storage_shape, device, _FLOAT_DTYPES if scale is None else _QUANT_DTYPES)
    if scale is not None:
        if not quantize_weights:
            raise ValueError("quantize_weights=False needs original, uncompressed weights without scales")
        if not isinstance(scale, torch.Tensor) or scale.ndim < 1 or scale.shape[0] != shape[0]:
            raise ValueError(f"{name} scale must retain the leading expert dimension")
        if scale.device != device or scale.layout != torch.strided:
            raise ValueError(f"{name} scale must be strided and on {device}")
        # The caller supplies natural quantizer output. dequant_w validates
        # group/scale shapes; do not run a second quantizer just to infer them.


def _smooth_scale(name, value, experts, k, device):
    if value is None:
        raise ValueError(f"int8_smoothquant requires {name}; pass ones to disable smoothing")
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    shapes = ((k,), (1, k), (experts, k), (1, 1, k), (experts, 1, k))
    if tuple(value.shape) not in shapes:
        raise ValueError(f"{name} must be shared [K]/[1,K]/[1,1,K] or per-expert [E,K]/[E,1,K]")
    _tensor(name, value, value.shape, device, _FLOAT_DTYPES)
    if not bool((torch.isfinite(value) & (value > 0)).all()):
        raise ValueError(f"{name} must be positive and finite")
    return value.detach().float().reshape(-1, 1, k).expand(experts, 1, k)


def _local_routes(ids, expert_mask, experts, topk, device):
    domain = experts
    if expert_mask is not None:
        if not isinstance(expert_mask, torch.Tensor) or expert_mask.ndim != 1:
            raise ValueError("expert_mask must be a one-dimensional global-expert mask")
        domain = expert_mask.numel()
        _tensor("expert_mask", expert_mask, (domain,), device, (torch.bool, torch.int32, torch.int64))
        if not bool(((expert_mask == 0) | (expert_mask == 1)).all()):
            raise ValueError("expert_mask entries must be 0 or 1")
        if expert_mask.sum().item() != experts:
            raise ValueError("expert_mask must select exactly the local weight expert count")
    if topk > domain:
        raise ValueError("topk must not exceed the global routing expert count")
    if not bool(((ids >= -1) & (ids < domain)).all()):
        raise ValueError("topk_ids must be in the routing domain, or -1 for an inactive route")
    ids = ids.detach().long()
    if expert_mask is None:
        return ids
    local = expert_mask.long().cumsum(0) - 1
    safe_ids = ids.clamp_min(0)
    return torch.where((ids >= 0) & (expert_mask[safe_ids] != 0), local[safe_ids], -1)


def _make_activation(name, gate_mode, swiglu_limit, beta, linear_beta):
    if not isinstance(name, str):
        raise TypeError("activation must be a string")
    name = name.lower().replace("-", "_")
    name = {"gelutanh": "gelu_tanh", "identity": "no", "none": "no", "situ": "situv2"}.get(name, name)
    if name not in ("no", "silu", "gelu", "gelu_tanh", "swiglu", "situv2"):
        raise ValueError(f"unsupported activation: {name!r}")
    if gate_mode == "none" and name in ("swiglu", "situv2"):
        raise ValueError(f"{name} needs both gate and up")
    if gate_mode == "none" and swiglu_limit is not None:
        raise ValueError("swiglu_limit needs a gated layout")
    if swiglu_limit is not None:
        if not math.isfinite(swiglu_limit) or swiglu_limit < 0:
            raise ValueError("swiglu_limit must be finite and nonnegative")
    if not all(math.isfinite(v) and v > 0 for v in (beta, linear_beta)):
        raise ValueError("beta and linear_beta must be positive and finite")
    limit = 7.0 if name == "swiglu" and swiglu_limit is None else swiglu_limit

    def unary(x):
        if name == "silu":
            return F.silu(x)
        if name in ("gelu", "gelu_tanh"):
            return F.gelu(x, approximate="tanh" if name == "gelu_tanh" else "none")
        return x

    def activate(projected):
        if gate_mode == "none":
            return unary(projected)
        if gate_mode == "interleave":
            gate, up = projected[..., 0::2], projected[..., 1::2]
        else:
            gate, up = projected.chunk(2, dim=-1)
        if limit is not None:
            gate = gate.clamp(max=limit)
            up = up.clamp(min=-limit, max=limit)
        if name == "swiglu":
            return gate * torch.sigmoid(1.702 * gate) * (up + 1.0)
        if name == "situv2":
            return (beta * torch.tanh(gate / beta) * torch.sigmoid(gate)) * (
                linear_beta * torch.tanh(up / linear_beta)
            )
        return unary(gate) * up

    return activate


def get(
    TP: int,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    quant_scheme_str: str = "no_quant",
    activation: str = "silu",
    *,
    gate_mode: str = "separated",
    swiglu_limit: float | None = None,
    beta: float = 1.0,
    linear_beta: float = 1.0,
    fp8_dtype: torch.dtype | None = None,
    quantize_weights: bool = True,
    doweight_stage1: bool = False,
    intermediate_dtype: torch.dtype | None = None,
    route_dtype: torch.dtype | None = None,
    output_dtype: torch.dtype = torch.float32,
    quantize_output: bool = False,
    token_chunk_size: int = 256,
):
    """Bind the numerical problem; return an ordinary callable, with no JIT.

    inter_dim is BEFORE TP division. Runtime W1/W2 contain one local shard with
    I = inter_dim // TP; no implicit padding, slicing, all-reduce or EP traffic.
    gate_mode: separated [gate;up], interleave [g0,u0,...], none (G1U0).
    mock_gate_only has the same numerical meaning as separated, not G1U0.

    quantize_weights=False keeps original weights in FP32 (W/s for SmoothQuant)
    while retaining the same activation quantizer: useful for loss measurement.
    no_quant instead disables both A/W quantization. Prequantized weights must
    have their natural quantizer-produced scales, not backend-shuffled buffers.

    doweight_stage1=True weights GEMM1 BEFORE bias1 and activation, and disables
    stage2 weighting. Otherwise weight (GEMM2 + bias2) once, before route_dtype.
    intermediate_dtype rounds AFTER activation and BEFORE stage2 quantization.
    quantize_output uses SmoothQuant's explicit N-group32 output compression.

    token_chunk_size bounds GEMM rows and the stage2 [chunk,topk,H] workspace;
    it never changes the whole-tensor activation quantizer's scale domain.
    """
    for name, value in (("TP", TP), ("model_dim", model_dim), ("inter_dim", inter_dim),
                        ("experts", experts), ("topk", topk), ("token_chunk_size", token_chunk_size)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if inter_dim % TP:
        raise ValueError("inter_dim must be divisible by TP")
    local_inter = inter_dim // TP
    if gate_mode not in ("separated", "interleave", "none", "mock_gate_only"):
        raise ValueError("gate_mode must be separated, interleave, none or mock_gate_only")
    for name, flag in (("quantize_weights", quantize_weights), ("doweight_stage1", doweight_stage1),
                       ("quantize_output", quantize_output)):
        if not isinstance(flag, bool):
            raise TypeError(f"{name} must be bool")
    for name, dtype in (("intermediate_dtype", intermediate_dtype), ("route_dtype", route_dtype)):
        if dtype is not None and dtype not in _FLOAT_DTYPES:
            raise ValueError(f"{name} must be None, FP16, BF16 or FP32")
    if output_dtype not in _FLOAT_DTYPES:
        raise ValueError("output_dtype must be FP16, BF16 or FP32")
    if fp8_dtype is not None and fp8_dtype not in _FP8_DTYPES:
        raise ValueError("fp8_dtype must be E4M3FN or E4M3FNUZ")
    quant = quantizer.get_quantizer(quant_scheme_str)
    smooth = quant is quantizer.int8_smoothquant
    if quantize_output and (not smooth or model_dim % 32):
        raise ValueError("quantize_output requires int8_smoothquant and model_dim divisible by 32")
    activate = _make_activation(activation, gate_mode, swiglu_limit, beta, linear_beta)
    gate_rows = local_inter if gate_mode == "none" else 2 * local_inter

    def activation_ref(value, scale=None):
        return quant.dequant_a(*quant.apply_a(value, fp8_dtype=fp8_dtype, smooth_scale=scale))

    def weight_ref(weight, scale, e, smoothing):
        value = weight[e]
        if scale is not None:
            result = quant.dequant_w(value, scale[e])
        elif quantize_weights:
            result = quant.dequant_w(*quant.apply_w(value, fp8_dtype=fp8_dtype, smooth_scale=smoothing))
        else:
            result = value.float()
            if smoothing is not None:
                result = result / smoothing
        return result

    @torch.no_grad()
    def ref_op(
        hidden_states: torch.Tensor,
        weight1: torch.Tensor,
        weight2: torch.Tensor,
        topk_weight: torch.Tensor,
        topk_ids: torch.Tensor, 
        *,
        w1_scale: torch.Tensor | None = None,
        w2_scale: torch.Tensor | None = None,
        a1_smooth_scale: torch.Tensor | None = None,
        a2_smooth_scale: torch.Tensor | None = None,
        bias1: torch.Tensor | None = None,
        bias2: torch.Tensor | None = None,
        expert_mask: torch.Tensor | None = None,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute using given routes (never router softmax, sorting or shuffle).

        hidden_states [M,H]; W1 [E,2I,H] (or [E,I,H]); W2 [E,H,I]; routes
        [M,topk]. Duplicate expert selections are independent route slots; -1
        is inactive. Routing weights are used as given, without normalization.
        expert_mask optionally maps global IDs to compact local weights in
        ascending global-expert order; unselected experts contribute zero.
        Smooth scales are shared [K]/[1,K]/[1,1,K] or expert [E,K]/[E,1,K].
        Biases [E,W1_rows], [E,H] and optional contiguous output [M,H] are natural
        tensors. Inputs are never mutated; output must not share their storage.
        """
        if not isinstance(hidden_states, torch.Tensor) or hidden_states.ndim != 2:
            raise ValueError("hidden_states must have shape [tokens, model_dim]")
        device = hidden_states.device
        tokens = hidden_states.shape[0]
        _tensor("hidden_states", hidden_states, (tokens, model_dim), device, _FLOAT_DTYPES)
        _tensor("topk_weight", topk_weight, (tokens, topk), device, _FLOAT_DTYPES)
        _tensor("topk_ids", topk_ids, (tokens, topk), device, (torch.int32, torch.int64))
        if not bool(torch.isfinite(topk_weight).all()):
            raise ValueError("topk_weight must be finite")
        _weight_metadata("weight1", weight1, w1_scale, (experts, gate_rows, model_dim), device,
                         quantize_weights)
        _weight_metadata("weight2", weight2, w2_scale, (experts, model_dim, local_inter), device,
                         quantize_weights)
        for name, bias, n in (("bias1", bias1, gate_rows), ("bias2", bias2, model_dim)):
            if bias is not None:
                _tensor(name, bias, (experts, n), device, _FLOAT_DTYPES)
        if output is not None:
            _tensor("output", output, (tokens, model_dim), device, (output_dtype,))
            if not output.is_contiguous() or output.requires_grad:
                raise ValueError("output must be contiguous and not require gradients")
            inputs = (hidden_states, weight1, weight2, topk_weight, topk_ids, w1_scale, w2_scale,
                      a1_smooth_scale, a2_smooth_scale, bias1, bias2, expert_mask)
            if output.numel() and any(
                isinstance(x, torch.Tensor) and x.numel() and x.device == device
                and x.untyped_storage().data_ptr() == output.untyped_storage().data_ptr()
                for x in inputs
            ):
                raise ValueError("output must not share storage with inputs")
        routes = _local_routes(topk_ids, expert_mask, experts, topk, device)
        s1 = s2 = None
        if smooth:
            s1 = _smooth_scale("a1_smooth_scale", a1_smooth_scale, experts, model_dim, device)
            s2 = _smooth_scale("a2_smooth_scale", a2_smooth_scale, experts, local_inter, device)
        elif a1_smooth_scale is not None or a2_smooth_scale is not None:
            raise ValueError("smooth scales are only accepted by int8_smoothquant")

        with _fp32_compute(device.type):
            # Do not quantize per-tensor A inside an expert/chunk loop. Stage1
            # sees all original tokens; stage2 sees all natural route slots.
            a1 = None if smooth else activation_ref(hidden_states)
            routed = torch.zeros((tokens, topk, local_inter), dtype=torch.float32, device=device)
            for e in range(experts):
                token_ids, slots = torch.where(routes == e)
                if token_ids.numel() == 0:
                    continue
                w1 = weight_ref(weight1, w1_scale, e, s1[e] if s1 is not None else None)
                for start in range(0, token_ids.numel(), token_chunk_size):
                    t, r = token_ids[start:start + token_chunk_size], slots[start:start + token_chunk_size]
                    a = activation_ref(hidden_states[t], s1[e]) if s1 is not None else a1[t]
                    projected = a @ w1.t()
                    if doweight_stage1:
                        projected = projected * topk_weight[t, r, None].float()
                    if bias1 is not None:
                        projected = projected + bias1[e].float()
                    value = activate(projected)
                    if intermediate_dtype is not None:
                        value = value.to(intermediate_dtype).float()
                    routed[t, r] = value
                del w1
            del a1
            a2 = routed if smooth else activation_ref(routed)
            del routed

            result = torch.empty((tokens, model_dim), dtype=torch.float32, device=device)
            for first in range(0, tokens, token_chunk_size):
                last = min(first + token_chunk_size, tokens)
                # Preserve natural TOPK order without atomic index_add and
                # without allocating a full [tokens,topk,model_dim] output.
                block = torch.zeros((last - first, topk, model_dim), dtype=torch.float32, device=device)
                for e in range(experts):
                    token_ids, slots = torch.where(routes[first:last] == e)
                    if token_ids.numel() == 0:
                        continue
                    w2 = weight_ref(weight2, w2_scale, e, s2[e] if s2 is not None else None)
                    for start in range(0, token_ids.numel(), token_chunk_size):
                        t, r = token_ids[start:start + token_chunk_size], slots[start:start + token_chunk_size]
                        a = a2[first + t, r]
                        if s2 is not None:
                            a = activation_ref(a, s2[e])
                        value = a @ w2.t()
                        if bias2 is not None:
                            value = value + bias2[e].float()
                        if not doweight_stage1:
                            value = value * topk_weight[first + t, r, None].float()
                        if quantize_output:
                            output_quant = quantizer.int8_smoothquant
                            value = output_quant.dequant_output(*output_quant.apply_output(value))
                        if route_dtype is not None:
                            value = value.to(route_dtype).float()
                        block[t, r] = value
                    del w2
                result[first:last] = block.sum(dim=1, dtype=torch.float32)
            result = result.to(output_dtype)
        if output is not None:
            output.copy_(result)
            return output
        return result

    return ref_op