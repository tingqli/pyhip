"""MoE 测试与 benchmark 共用的输入、独立参考、固定配置执行和计时。

导入本模块不初始化 GPU。调用方设置默认设备；所有 tensor 保持顶层参数，
让 run_perftest 正确复制输入、权重、scale 和输出，不在这里建立 buffer 池。
"""

import inspect
import math


def moe_types(model, *, dtype="fp8", quant="model", activation="silu", gate_mode="separated"):
    import aiter
    import torch

    kind = (model["quant_type"] if quant == "model" else quant) if dtype == "fp8" else dtype
    arch = torch.cuda.get_device_properties().gcnArchName
    fp8 = torch.float8_e4m3fn if "gfx950" in arch else torch.float8_e4m3fnuz
    weight_dtype = {"bf16": torch.bfloat16, "fp8": fp8, "mxfp4": torch.float4_e2m1fn_x2}[dtype]
    quant_type = {"bf16": aiter.QuantType.No, "ptpc": aiter.QuantType.per_Token,
                  "per_tensor": aiter.QuantType.per_Tensor, "block": aiter.QuantType.per_128x128,
                  "mxfp4": aiter.QuantType.per_1x32}[kind]
    if activation == "gelu" and (kind != "bf16" or gate_mode != "separated"):
        raise ValueError("GELU uses non-gated BF16 weights with separated gate mode")
    act = {"silu": aiter.ActivationType.Silu, "gelu": aiter.ActivationType.Gelu,
           "swiglu": aiter.ActivationType.Swiglu, "situv2": aiter.ActivationType.Situv2}[activation]
    return kind, weight_dtype, quant_type, act


def _make_weight(experts, rows, cols, kind, quant_dtype, generator, shuffled, interleave, gate,
                 structured_scales=False):
    import aiter
    import torch
    from aiter.ops.shuffle import shuffle_scale, shuffle_weight

    packed = kind == "mxfp4"
    weight = torch.empty((experts, rows, cols // 2 if packed else cols), dtype=quant_dtype)
    scale_shape = {"bf16": None, "ptpc": (experts, rows, 1), "per_tensor": (experts,),
                   "block": (experts, rows // 128, cols // 128),
                   "mxfp4": (experts * rows, cols // 32)}[kind]
    scales = (None if scale_shape is None else torch.empty(
        scale_shape, dtype=torch.uint8 if packed else torch.float32))
    for expert in range(experts):
        value = torch.randn((rows, cols), dtype=torch.bfloat16, generator=generator)
        if packed and structured_scales:
            # 沿用旧 MXFP4 测试的 expert/row/K-group 幅度变化，避免 scale 全部相同。
            value.mul_(2.0 ** (expert % 3 - 1))
            value.mul_(torch.pow(2.0, torch.arange(rows)[:, None] % 3 - 1).bfloat16())
            value.view(rows, cols // 32, 32).mul_(
                torch.pow(2.0, torch.arange(cols // 32)[None, :, None] % 5 - 2).bfloat16())
        if kind == "bf16":
            q, scale = value, None
        elif kind == "block":
            blocks = value.view(rows // 128, 128, cols // 128, 128).permute(0, 2, 1, 3)
            q, scale = aiter.get_torch_quant(aiter.QuantType.per_Token)(
                blocks.reshape(-1, 128 * 128), quant_dtype=quant_dtype)
            q = q.view(rows // 128, cols // 128, 128, 128).permute(0, 2, 1, 3).reshape(rows, cols)
            scale = scale.reshape(rows // 128, cols // 128)
        else:
            qtype = {"ptpc": aiter.QuantType.per_Token, "per_tensor": aiter.QuantType.per_Tensor,
                     "mxfp4": aiter.QuantType.per_1x32}[kind]
            q, scale = aiter.get_torch_quant(qtype)(value, quant_dtype=quant_dtype)
        if packed:
            weight[expert].view(torch.uint8).copy_(q.view(torch.uint8))
            scales[expert * rows:(expert + 1) * rows].copy_(scale.view(torch.uint8))
        else:
            weight[expert].copy_(q)
            if scales is not None:
                scales[expert].copy_(scale.reshape(scales[expert].shape))
    if packed:
        scales = scales.view(torch.float8_e8m0fnu)
    if shuffled:
        weight = shuffle_weight(weight, is_guinterleave=interleave, gate_up=gate)
        if packed:
            scales = shuffle_scale(scales, experts, is_guinterleave=interleave, gate_up=gate)
    else:
        weight.is_shuffled = False
    return weight, scales


def prepare_moe(model, tokens, *, dtype="fp8", quant="model", activation="silu",
                gate_mode="separated", preshuffle="on", routing="balanced", seed=0,
                beta=None, linear_beta=None, swiglu_limit=None, structured_scales=False):
    """生成一份公共 API 参数；I 使用 TP 切分后的实际维度，不做后端专属 padding。"""
    import torch

    h, i, e, k = model["HIDDEN_SIZE"], model["INTER_SIZE"] // model["TP"], model["E"], model["TOPK"]
    kind, weight_dtype, quant_type, act = moe_types(
        model, dtype=dtype, quant=quant, activation=activation, gate_mode=gate_mode)
    if kind == "block" and (h % 128 or i % 128):
        raise ValueError("block scales require H and I_tp divisible by 128; no implicit padding is allowed")
    if kind == "mxfp4" and "gfx950" not in torch.cuda.get_device_properties().gcnArchName:
        raise ValueError("these MXFP4 kernels require gfx950")
    generator = torch.Generator(device="cuda").manual_seed(seed)
    x = (torch.randn((tokens, h), dtype=torch.bfloat16, generator=generator) + 1) * .001
    interleave, gated = gate_mode == "interleave", activation != "gelu"
    w1, s1 = _make_weight(e, (2 if gated else 1) * i, h, kind, weight_dtype, generator,
                          preshuffle == "on", interleave, gated, structured_scales)
    w2, s2 = _make_weight(e, h, i, kind, weight_dtype, generator,
                          preshuffle == "on", interleave, False, structured_scales)
    if routing == "balanced":
        permutation = torch.randperm(e, dtype=torch.int32, generator=generator)
        ids = permutation.repeat((tokens * k + e - 1) // e)[:tokens * k].reshape(tokens, k).contiguous()
    elif routing == "random":
        scores = torch.randn((tokens, e), generator=generator)
        ids = scores.topk(k, dim=-1).indices.to(torch.int32).contiguous()
    else:
        raise ValueError(f"unknown routing: {routing}")
    weights = torch.randn((tokens, k), generator=generator)
    return dict(hidden_states=x, w1=w1, w2=w2, topk_weight=weights, topk_ids=ids,
                w1_scale=s1, w2_scale=s2, quant_type=quant_type, activation=act,
                gate_mode=gate_mode, beta=beta, linear_beta=linear_beta,
                swiglu_limit=swiglu_limit, output=torch.empty_like(x)), kind


def _reference_weight(weight, scale, expert, kind, interleave, gate):
    import torch

    value = weight[expert]
    n, k = value.shape
    shuffled = bool(getattr(weight, "is_shuffled", False))
    if kind == "fp4":
        value = value.view(torch.uint8)
    if shuffled and interleave and gate:
        value = value.view(n // 32, 2, k // 64, 4, 16, 16)
        value = value.permute(1, 0, 4, 2, 3, 5).reshape(n, k)
    elif shuffled:
        pack = 16 // value.element_size()
        value = value.view(n // 16, k // pack, 16, pack)
        value = value.permute(0, 2, 1, 3).reshape(n, k)
    if kind == "bf16":
        return value
    if kind == "fp4":
        from aiter.utility import fp4_utils

        cols = k * 2 // 32
        if shuffled:
            padded = (cols + 7) // 8 * 8
            scales = scale.view(torch.uint8).reshape(-1)[expert * n * padded:(expert + 1) * n * padded]
            scales = scales.view(n // 32, padded // 8, 4, 16, 2, 2)
            order = (5, 0, 3, 1, 4, 2) if interleave and gate else (0, 5, 3, 1, 4, 2)
            scales = scales.permute(order).reshape(n, padded)[:, :cols]
        else:
            scales = scale.view(torch.uint8).reshape(weight.shape[0], n, cols)[expert]
        value = fp4_utils.mxfp4_to_f32(value.view(torch.float4_e2m1fn_x2)).view(n, cols, 32)
        return (value * fp4_utils.e8m0_to_f32(scales).unsqueeze(-1)).reshape(n, k * 2).to(torch.bfloat16)
    value = value.float()
    if kind == "ptpc":
        value = value * scale.reshape(weight.shape[0], n, 1)[expert]
    elif kind == "per_tensor":
        value = value * scale.reshape(-1)[expert]
    else:
        value = (value.view(n // 128, 128, k // 128, 128)
                 * scale.reshape(weight.shape[0], n // 128, k // 128)[expert, :, None, :, None])
    value = value.reshape(n, k)
    return value if kind == "block" else value.to(torch.bfloat16)


def _reference_activation(value, kind, quant_dtype):
    import torch

    if kind == "a4w4":
        import aiter
        from aiter.utility import fp4_utils

        quant, scales = aiter.get_torch_quant(aiter.QuantType.per_1x32)(value, quant_dtype=quant_dtype)
        dequant = (fp4_utils.mxfp4_to_f32(quant).view(-1, 32)
                   * fp4_utils.e8m0_to_f32(scales).view(-1, 1))
        return dequant.reshape(value.shape).to(value.dtype)
    if kind not in ("ptpc", "per_tensor", "block"):
        return value
    shape = value.shape
    data = value.float().reshape(-1, 128 if kind == "block" else shape[-1])
    amax = data.abs().amax() if kind == "per_tensor" else data.abs().amax(dim=-1, keepdim=True)
    scale = amax / torch.finfo(quant_dtype).max
    scale = torch.where(scale == 0, 1.0, scale)
    result = ((data / scale).to(quant_dtype).float() * scale).reshape(shape)
    return result if kind == "block" else result.to(value.dtype)


def torch_reference(call, *, mxfp4_activations=False):
    """独立 Torch MoE 参考；沿用公共 API 默认值、原量化语义和数值顺序。

    只在调用时导入算子以补参数和验证输入，不执行任何被测 kernel。
    FP8 对输入和中间结果量化；MXFP4 使用 BF16 激活；GELU 为非 gated。
    固定 A4W4 kernel 测试可显式模拟两次 MXFP4 激活量化；API 调优仍用默认参考。
    """
    import aiter
    import torch
    from aiter.ops.flydsl.moe_common import GateMode
    from pyhip.ops.moe.tuned_moe import _native_kind, fused_moe

    bound = inspect.signature(fused_moe).bind(**call)
    bound.apply_defaults()
    call = bound.arguments
    x, w1, w2, ids, weights = (call[name] for name in
                              ("hidden_states", "w1", "w2", "topk_ids", "topk_weight"))
    kind = _native_kind(call)
    if kind is None:
        raise ValueError("independent MoE validation is unavailable for these inputs")
    if mxfp4_activations and kind != "fp4":
        raise ValueError("MXFP4 activation reference requires MXFP4 weights")
    act_kind = "a4w4" if mxfp4_activations else kind
    if not ((ids >= 0) & (ids < w1.shape[0])).all().item():
        raise ValueError("topk_ids contains an out-of-range expert")
    b, h = x.shape
    gelu = call["activation"] == aiter.ActivationType.Gelu
    i, topk = w1.shape[1] // (1 if gelu else 2), ids.shape[1]
    interleave = call["gate_mode"] == GateMode.INTERLEAVE
    dtype = x.dtype
    x = _reference_activation(x, act_kind, w1.dtype)
    mid = torch.empty((b, topk, i), dtype=dtype, device=x.device)
    for expert in range(w1.shape[0]):
        row, slot = torch.where(ids == expert)
        if row.numel() == 0:
            continue
        weight = _reference_weight(w1, call["w1_scale"], expert, kind, interleave, True)
        projection = x[row].float() @ weight.float().T
        act = call["activation"]
        if gelu:
            value = torch.nn.functional.gelu(projection)
        else:
            gate, up = projection.chunk(2, dim=-1)
            if act == aiter.ActivationType.Swiglu:
                limit = 7.0 if call["swiglu_limit"] is None else float(call["swiglu_limit"])
                gate, up = gate.clamp(max=limit), up.clamp(-limit, limit)
                value = gate * torch.sigmoid(1.702 * gate) * (up + 1.0)
            elif act == aiter.ActivationType.Situv2:
                beta = 1.0 if call["beta"] is None else float(call["beta"])
                linear_beta = 1.0 if call["linear_beta"] is None else float(call["linear_beta"])
                value = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
                value = value * (linear_beta * torch.tanh(up / linear_beta))
            else:
                value = torch.nn.functional.silu(gate) * up
        mid[row, slot] = value.to(dtype)
    mid = _reference_activation(mid, act_kind, w1.dtype)
    result = torch.zeros((b, h), dtype=torch.float32, device=x.device)
    for expert in range(w2.shape[0]):
        row, slot = torch.where(ids == expert)
        if row.numel() == 0:
            continue
        weight = _reference_weight(w2, call["w2_scale"], expert, kind, interleave, False)
        value = (mid[row, slot].float() @ weight.float().T) * weights[row, slot, None]
        result.index_add_(0, row, value)
    return result.to(dtype)


def make_moe_runner(config):
    """固定配置的完整 MoE 调用，不搜索、不查 winner、不 fallback。

    闭包只保存静态配置和 API 默认值，tensor 始终从每次调用取得。
    测试显式给出 dtype/布局/配置，不以 autotune 候选列表决定测试覆盖。
    """
    import torch
    from pyhip.ops.moe import tuned_moe as tm

    config = dict(config)
    if "_impl" not in config:
        raise ValueError("fixed MoE config requires _impl")
    defaults = {name: p.default for name, p in inspect.signature(tm.fused_moe).parameters.items()
                if p.default is not inspect.Parameter.empty}

    def run(hidden_states, w1, w2, topk_weight, topk_ids, **options):
        options = defaults | options
        with torch.cuda.device(hidden_states.device), torch.no_grad():
            return tm._fmoe_wrapper(hidden_states, w1, w2, topk_weight, topk_ids,
                                    options=options, batch_bucket=0, model_key="", **config)

    return run


def check_output(result, output, reference):
    """0.02 是能量归一化误差门槛，不是逐元素 2% 误差。"""
    import torch
    from pyhip import calc_diff

    if result is not output or result.shape != reference.shape or result.dtype != reference.dtype:
        return dict(status="ERROR", reason="output identity/shape/dtype mismatch")
    if not torch.isfinite(result).all().item():
        return dict(status="INCORRECT", reason="output contains NaN/Inf")
    diff = calc_diff(reference, result)
    if not math.isfinite(diff):
        return dict(status="INCORRECT", reason="non-finite calc_diff")
    return dict(status="PASS" if diff <= .02 else "INCORRECT", diff=diff)


def measure_moe(op, call, reference, *, iters=10, warmup=2, copies=0, allow_incorrect=False):
    """复用 run_perftest；检查所有实际输出副本，不在计时内计算参考。"""
    from pyhip import run_perftest

    outputs = {}

    def invoke(**buffers):
        result = op(**buffers)
        output = buffers["output"]
        outputs[output.data_ptr()] = (result, output)
        return result

    call["output"].fill_(float("nan"))
    stats = {}
    _, mean_us = run_perftest(invoke, **call, num_iters=iters, num_warmup=warmup,
                             num_copies=copies, num_stats=stats)
    worst = dict(status="PASS", diff=0.0)
    for result, output in outputs.values():
        check = check_output(result, output, reference)
        if check["status"] != "PASS" and not (allow_incorrect and check["status"] == "INCORRECT"):
            raise RuntimeError(f"timed output failed validation: {check}")
        if check.get("diff", math.inf) >= worst.get("diff", math.inf):
            worst = check
    stats["correctness"] = worst
    stats["mean_us"] = mean_us
    return stats