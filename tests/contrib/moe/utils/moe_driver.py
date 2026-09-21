# SPDX-License-Identifier: MIT
"""Function-based MoE drivers: factory(config) -> (prepare, run).

    prepare, run = registry["jit_splitk_16_64_True"](config)
    weights = prepare(w1_bf16, w2_bf16)
    run(x_bf16, weights, ids_i32, scores_f32, output)

Source W1=[E,2I,H] is always natural [gate; up], W2=[E,H,I]. I is local.
Factories are CPU-only; compilation is lazy. Prepared dictionaries belong to
the matching kernel layout. All drivers use stage2 route weighting and native
FP8. No SmoothQuant, implicit padding, TP slicing or ownership framework.
"""

from __future__ import annotations

import functools
import importlib
import json
import os
from pathlib import Path
from typing import NamedTuple

import torch

if __package__:
    from . import moe_ref, moe_tuned, quantizer
else:
    import moe_ref
    import moe_tuned
    import quantizer

__all__ = [
    "MOEconfig", "register", "registry", "validate_routes", "ref", "aiter", "tuned",
    "jit_splitk", "jit_blockscale", "jit_batch1", "jit_batch", "jit_fused", "jit_loopn",
    "jit_mxfp4", "fly_splitk", "fly_decode", "fly_prefill", "prefill_bf16", "prefill_fp8",
]


class MOEconfig(NamedTuple):
    model_dim: int
    inter_dim_tp: int
    experts: int
    topk: int
    quant_scheme: str
    activation: str
    preshuffle: bool
    swiglu_limit: float | None = None
    beta: float = 1.0
    linear_beta: float = 1.0
    output_dtype: torch.dtype = torch.bfloat16


registry = {}


def register(factory):
    """Register a config-only candidate under its function name."""
    if factory.__name__ in registry:
        raise ValueError(f"duplicate MoE driver: {factory.__name__}")
    registry[factory.__name__] = factory
    return factory


def _require(condition, message):
    if not condition:
        raise NotImplementedError(message)


def _positive(name, value):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _tensor(name, value, shape, device, dtype):
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tuple(value.shape) != tuple(shape) or value.layout != torch.strided:
        raise ValueError(f"{name} must have strided shape {tuple(shape)}")
    if value.device != device or value.dtype != dtype:
        raise ValueError(f"{name} must be {dtype} on {device}")
    if not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _copy(value):
    if value is None:
        return None
    # Byte copying also works for Torch packed/shell dtypes.
    copied = value.detach().contiguous().view(torch.uint8).clone().view(value.dtype).reshape(value.shape)
    if hasattr(value, "is_shuffled"):
        copied.is_shuffled = value.is_shuffled
    return copied


def _check_config(config, *, gpu=True, local=False):
    for name in ("model_dim", "inter_dim_tp", "experts", "topk"):
        _positive(name, getattr(config, name))
    if config.topk > min(config.experts, 255):
        raise ValueError("require topk <= experts and topk <= 255")
    if not isinstance(config.preshuffle, bool):
        raise TypeError("preshuffle must be bool")
    if config.output_dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("output_dtype must be FP16, BF16 or FP32")
    _require(config.quant_scheme != "int8_smoothquant", "MoE drivers do not support SmoothQuant")
    quantizer.get_quantizer(config.quant_scheme)
    moe_ref._make_activation(config.activation, "separated", config.swiglu_limit, config.beta, config.linear_beta)
    if gpu:
        _require(config.output_dtype == torch.bfloat16, "optimized drivers currently produce BF16 output")
    if local:
        _require(config.preshuffle, "this kernel requires preshuffle=True")
        _require(config.model_dim % 32 == config.inter_dim_tp % 32 == 0, "H and local I must be divisible by 32")


def _device(config, device):
    _require(device.type == "cuda" and torch.version.hip is not None, "requires an AMD ROCm GPU")
    gfx = torch.cuda.get_device_properties(device).gcnArchName.split(":")[0]
    _require(gfx in ("gfx942", "gfx950"), f"unsupported architecture {gfx}")
    if config.quant_scheme in ("a16w4", "a8w4", "a4w4"):
        _require(gfx == "gfx950", "MXFP4 drivers require gfx950")
    return gfx


def _tokens(config, x, output, block_m=16, *, max_tokens=None, local=True):
    tokens = x.shape[0]
    if tokens >= 2**24:
        raise ValueError(f"unsupported tokens={tokens}")
    _require(max_tokens is None or tokens <= max_tokens, f"unsupported tokens={tokens}; max_tokens={max_tokens}")
    if local:
        max_rows = tokens * config.topk + config.experts * block_m
        if max_rows * max(config.model_dim, config.inter_dim_tp) * 2 >= 2**32:
            raise ValueError("runtime workspaces exceed the local kernels' 32-bit buffer addressing")
    if output is None:
        raise ValueError("an output buffer is required")
    return tokens


def validate_routes(config, topk_ids, topk_weights, *, sorted_routes=True):
    """Optional untimed value check; runtime trusts the caller's tensors."""
    if not isinstance(topk_ids, torch.Tensor) or topk_ids.ndim != 2:
        raise ValueError("topk_ids must have shape [tokens, topk]")
    shape = (topk_ids.shape[0], config.topk)
    _tensor("topk_ids", topk_ids, shape, topk_ids.device, torch.int32)
    _tensor("topk_weights", topk_weights, shape, topk_ids.device, torch.float32)
    if not bool(((topk_ids >= 0) & (topk_ids < config.experts)).all()):
        raise ValueError(f"topk_ids must be in [0, {config.experts})")
    if not bool(torch.isfinite(topk_weights).all()):
        raise ValueError("topk_weights must be finite")
    if sorted_routes and config.topk > 1:
        ordered = topk_ids.sort(dim=-1).values
        if bool((ordered[:, 1:] == ordered[:, :-1]).any()):
            raise ValueError("Aiter sorting requires distinct experts within each token's TOPK")


@torch.no_grad()
def _prepare_weights(config, weight1, weight2, *, gpu=True, quantize=True):
    """Natural quantized copies. Packing remains the candidate's responsibility."""
    H, I, E = config.model_dim, config.inter_dim_tp, config.experts
    if isinstance(weight1, (tuple, list)):
        if len(weight1) != 2:
            raise ValueError("weight1 must be a (gate, up) pair")
        gate, up = weight1
        for label, tensor in (("gate", gate), ("up", up)):
            _tensor(label, tensor, (E, I, H), gate.device, torch.bfloat16)
            if getattr(tensor, "is_shuffled", False):
                raise ValueError("source weights must be unshuffled")
        weight1 = torch.cat((gate, up), dim=1)
    if not isinstance(weight1, torch.Tensor):
        raise TypeError("weight1 must be a tensor or (gate, up) pair")
    _tensor("weight1", weight1, (E, 2 * I, H), weight1.device, torch.bfloat16)
    _tensor("weight2", weight2, (E, H, I), weight1.device, torch.bfloat16)
    if any(getattr(w, "is_shuffled", False) for w in (weight1, weight2)):
        raise ValueError("source weights must be unshuffled")
    if gpu:
        _device(config, weight1.device)
    quant = quantizer.get_quantizer(config.quant_scheme)
    w1, s1 = quant.apply_w(weight1) if quantize else (weight1, None)
    w2, s2 = quant.apply_w(weight2) if quantize else (weight2, None)
    return dict(zip(("w1", "w2", "w1s", "w2s"), map(_copy, (w1, w2, s1, s2))))


def _shuffle_weights(config, weights, *, gate_mode="separated", down_interleave=False):
    from aiter.ops.shuffle import shuffle_scale, shuffle_weight

    w1, w2, s1, s2 = (weights[key] for key in ("w1", "w2", "w1s", "w2s"))
    if w1.dtype == torch.float4_e2m1fn_x2:
        interleave = gate_mode == "interleave"
        s1 = shuffle_scale(s1.flatten(0, 1), config.experts, is_guinterleave=interleave, gate_up=interleave)
        s2 = shuffle_scale(s2.flatten(0, 1), config.experts, is_guinterleave=down_interleave)
        if config.preshuffle:
            w1 = shuffle_weight(w1, is_guinterleave=interleave, gate_up=interleave)
            w2 = shuffle_weight(w2, is_guinterleave=down_interleave)
    elif config.preshuffle:
        w1, w2 = shuffle_weight(w1), shuffle_weight(w2)
    return dict(w1=w1, w2=w2, w1s=s1, w2s=s2)


def ref(config: MOEconfig, *, quantize_weights=True, intermediate_dtype=None,
        route_dtype=None, quantize_output=False, token_chunk_size=256):
    """Unsorted oracle, natural weights; no bias/mask/SmoothQuant driver extras."""
    _check_config(config, gpu=False)
    _require(not config.preshuffle, "reference requires preshuffle=False")
    reference = moe_ref.get(
        1, config.model_dim, config.inter_dim_tp, config.experts, config.topk,
        config.quant_scheme, config.activation, gate_mode="separated", doweight_stage1=False,
        swiglu_limit=config.swiglu_limit, beta=config.beta, linear_beta=config.linear_beta,
        quantize_weights=quantize_weights, intermediate_dtype=intermediate_dtype,
        route_dtype=route_dtype, output_dtype=config.output_dtype,
        quantize_output=quantize_output, token_chunk_size=token_chunk_size,
    )

    def prepare(weight1, weight2):
        return _prepare_weights(config, weight1, weight2, gpu=False, quantize=quantize_weights)

    def run(hidden_states, weights, topk_ids, topk_weights, output):
        _tokens(config, hidden_states, output, local=False)
        return reference(hidden_states, weights["w1"], weights["w2"], topk_weights, topk_ids,
                         w1_scale=weights["w1s"], w2_scale=weights["w2s"], output=output)

    return prepare, run


# Serial Aiter switching only; no concurrent/direct Aiter interleaving guarantee.
_ACTIVE_AITER = None
_AITER_CONFIG_ENV = None


def _kernel_name(call, stage):
    options = {}
    while isinstance(call, functools.partial):
        options = {**call.keywords, **options}
        call = call.func
    return options.get(f"kernelName{stage}", options.get("kernelName", ""))


def aiter(config: MOEconfig, *, tuned_config=None, gate_mode=None):
    """Public fused_moe; tuned_config is a caller-selected official CSV row."""
    _check_config(config)
    H, I, E, K = config.model_dim, config.inter_dim_tp, config.experts, config.topk
    q, activation = config.quant_scheme, config.activation
    gate_mode = gate_mode or ("interleave" if q == "a8w4" else "separated")
    _require(q in ("no_quant", "bf16", "fp8_ptpc", "fp8_per_tensor", "fp8_blockscale", "a16w4", "a8w4", "a4w4"),
             "Aiter has no INT8/INT4 MoE path")
    mx = q in ("a16w4", "a8w4", "a4w4")
    alignment = 256 if mx else (128 if q == "fp8_blockscale" else 32)
    _require(H % alignment == I % alignment == 0, f"H and local I must be divisible by {alignment}; no padding")
    if mx:
        _require(activation in ("silu", "swiglu", "situv2"), "unsupported MX activation")
        _require(gate_mode == ("interleave" if q == "a8w4" else "separated"), "unsupported MX gate layout")
        _require(q != "a16w4" or activation == "situv2", "A16W4 Aiter supports SiTUv2 only")
        _require(q != "a8w4" or activation in ("swiglu", "situv2"), "A8W4 requires SwiGLU/SiTUv2")
    else:
        _require(gate_mode == "separated" and activation in ("silu", "gelu", "gelu_tanh"),
                 "BF16/FP8 requires separated SiLU/GELU/GELU-tanh")
    _require(config.swiglu_limit is None or (activation == "swiglu" and config.swiglu_limit == 7),
             "standard Aiter supports only the default SwiGLU clamp")
    if tuned_config is not None and activation == "situv2":
        _require((config.beta, config.linear_beta) == (4.0, 25.0), "tuned SiTUv2 requires beta=4, linear_beta=25")
    if not config.preshuffle:
        _require(q == "a4w4" and activation == "silu" and gate_mode == "separated",
                 "only A4W4 separated SiLU is verified with preshuffle=False")
        _require(tuned_config is None, "tuned rows describe shuffled weights, not natural-weight CK")
    tuned_config = dict(tuned_config) if tuned_config is not None else None
    module = call = None
    metadata, verified = {}, set()

    def prepare(weight1, weight2):
        return _shuffle_weights(config, _prepare_weights(config, weight1, weight2),
                                gate_mode=gate_mode, down_interleave=gate_mode == "interleave")

    def activate(device):
        nonlocal module, call
        global _ACTIVE_AITER, _AITER_CONFIG_ENV
        module = importlib.import_module("aiter.fused_moe")
        package = importlib.import_module("aiter")
        if _ACTIVE_AITER is None:
            _AITER_CONFIG_ENV = os.environ.get("AITER_CONFIG_FMOE")
        if tuned_config is not None:
            from aiter.jit.core import AITER_CONFIG_FMOE
            os.environ["AITER_CONFIG_FMOE"] = AITER_CONFIG_FMOE
        elif _AITER_CONFIG_ENV is None:
            os.environ.pop("AITER_CONFIG_FMOE", None)
        else:
            os.environ["AITER_CONFIG_FMOE"] = _AITER_CONFIG_ENV
        qtype = {"no_quant": "No", "bf16": "No", "fp8_ptpc": "per_Token",
                 "fp8_per_tensor": "per_Tensor", "fp8_blockscale": "per_1x128"}.get(q, "per_1x32")
        act = {"silu": "Silu", "gelu": "Gelu", "gelu_tanh": "GeluTanh", "swiglu": "Swiglu", "situv2": "Situv2"}[activation]
        os.environ.update(AITER_ONLINE_TUNE="0", AITER_BYPASS_TUNE_CONFIG="0", AITER_KSPLIT="0", AITER_XBFLOAT16="0",
                          AITER_SITUV2_A8W4=str(int(q == "a8w4")), AITER_SITUV2_A4W4=str(int(q == "a4w4")),
                          AITER_BF16_FP8_MOE_BOUND="0", AITER_FLYDSL_STAGE2_FP8="0", AITER_MOE_A8W4_BYPASS_QUANT="0",
                          AITER_MXFP4_INTERMEDIATE="0", AITER_FLYDSL_FORCE_REDUCE="0", AITER_FORCE_A8W4="0")
        module._MOE_A8W4_BYPASS_QUANT, module._SWIGLU_MXFP4_BF16_BOUND = False, 0
        module.AITER_CONFIGS.get_config_file.cache_clear()
        module.get_2stage_cfgs.cache_clear()
        module.cfg_2stages_by_file.clear()
        module.cfg_2stages = ({}, {}) if tuned_config is not None or not config.preshuffle else None
        call = functools.partial(module.fused_moe, activation=getattr(package.ActivationType, act),
                                 quant_type=getattr(package.QuantType, qtype), gate_mode=gate_mode,
                                 dtype=config.output_dtype, doweight_stage1=False,
                                 swiglu_limit=config.swiglu_limit, beta=config.beta, linear_beta=config.linear_beta)
        verified.clear()
        metadata.clear()
        _ACTIVE_AITER = (run, device)

    def first_call(x, w1, w2, s1, s2, tw, ti, out):
        token = module.get_padded_M(x.shape[0])
        p = torch.cuda.get_device_properties(x.device)
        qa = torch.bfloat16 if q in ("no_quant", "bf16", "a16w4") else (
            torch.float4_e2m1fn_x2 if q == "a4w4" else torch.float8_e4m3fn if q == "a8w4" else w1.dtype)
        expected = dict(gfx=p.gcnArchName.split(":")[0], cu_num=p.multi_processor_count,
                        token=token, model_dim=H, inter_dim=I, expert=E, topk=K,
                        act_type=str(call.keywords["activation"]), dtype=str(config.output_dtype),
                        q_dtype_a=str(qa), q_dtype_w=str(w1.dtype), q_type=str(call.keywords["quant_type"]),
                        use_g1u1=True, doweight_stage1=False)
        row = tuned_config
        if row is not None:
            for name, value in expected.items():
                actual = row[name]
                if isinstance(value, bool):
                    actual = str(actual).lower() in ("true", "1", "1.0")
                elif isinstance(value, int):
                    actual = int(float(actual))
                if actual != value:
                    raise ValueError(f"tuned {name}={actual!r} does not match {value!r}")
            row = dict(row)
            for name in ("block_m", "ksplit"):
                row[name] = int(float(row[name]))
            for name in ("run_1stage", "flat", "xbf16"):
                row[name] = str(row.get(name, "0")).lower() in ("true", "1", "1.0")
            module.cfg_2stages[0][tuple(expected.values())] = row
        resolve, lookups, calls = module.get_2stage_cfgs, [], []

        def observe(*args, **kwargs):
            if (args[0], str(args[6]), str(args[7]), str(args[8])) != (token, str(qa), str(w1.dtype), expected["q_type"]):
                raise RuntimeError("Aiter changed the requested activation/weight quantization")
            selected = resolve(*args, **kwargs)
            if row is not None:
                for name in ("block_m", "ksplit", "run_1stage", "flat"):
                    if getattr(selected, name) != row[name]:
                        raise RuntimeError(f"Aiter ignored tuned {name}")
                for stage in (1,) if selected.run_1stage else (1, 2):
                    if _kernel_name(getattr(selected, f"stage{stage}"), stage) != row[f"kernelName{stage}"]:
                        raise RuntimeError(f"Aiter replaced tuned stage{stage}")
                if selected.run_1stage and bool(selected.stage1.keywords.get("xbf16", False)) != row["xbf16"]:
                    raise RuntimeError("Aiter ignored tuned xbf16")
            if selected.run_1stage and q.startswith("fp8") and selected.stage1.keywords.get("xbf16", False):
                raise NotImplementedError("xbf16 one-stage is not the requested FP8 activation policy")
            if not config.preshuffle:
                _require(not selected.run_1stage and selected.stage1.func is module.ck_moe_stage1
                         and selected.stage2.func is module.aiter.ck_moe_stage2_fwd,
                         "natural A4W4 requires the layout-aware CK two-stage path")
            lookups.append(selected)
            return selected

        module.get_2stage_cfgs, module.kernel_bench_callable = observe, calls
        try:
            result = call(x, w1, w2, tw, ti, w1_scale=s1, w2_scale=s2, output=out)
        finally:
            module.get_2stage_cfgs, module.kernel_bench_callable = resolve, None
        selected = lookups[-1]
        if [stage for stage, _ in calls] != (["stage1"] if selected.run_1stage else ["stage1", "stage2"]):
            raise NotImplementedError("unverified Aiter execution path")
        for stage, stage_call in calls:
            if _kernel_name(stage_call, int(stage[-1])) != _kernel_name(getattr(selected, stage), int(stage[-1])):
                raise RuntimeError(f"actual Aiter {stage} differs from metadata")
            if not selected.run_1stage and stage_call.args[0].dtype != qa:
                raise NotImplementedError(f"Aiter {stage} uses {stage_call.args[0].dtype}, not {qa}")
        metadata[(x.device, token)] = selected
        verified.add(x.shape[0])
        return result

    @torch.no_grad()
    def run(hidden_states, weights, topk_ids, topk_weights, output):
        if not _tokens(config, hidden_states, output, local=False):
            return output
        with torch.cuda.device(hidden_states.device):
            if _ACTIVE_AITER != (run, hidden_states.device):
                activate(hidden_states.device)
            w1, w2, s1, s2 = (weights[key] for key in ("w1", "w2", "w1s", "w2s"))
            if hidden_states.shape[0] not in verified:
                return first_call(hidden_states, w1, w2, s1, s2, topk_weights, topk_ids, output)
            return call(hidden_states, w1, w2, topk_weights, topk_ids, w1_scale=s1, w2_scale=s2, output=output)

    run.metadata = metadata  # Aiter-only diagnostics, not another driver hierarchy.
    return prepare, run


def tuned(config: MOEconfig, *, tuned_csv="./tuned_aiter/moe_tuned.csv", tokens=None):
    """Exact-M dispatch; prepare distinct winning layouts, never repack at runtime."""
    _check_config(config)
    path = Path(tuned_csv).expanduser().resolve()
    rows = moe_tuned.read_rows(path)
    requested = None if tokens is None else ([tokens] if isinstance(tokens, int) else list(tokens))

    @torch.no_grad()
    def prepare(weight1, weight2):
        device = weight2.device
        gfx = _device(config, device)
        key = moe_tuned.model_key(config._asdict(), dict(gfx=gfx, cu_num=torch.cuda.get_device_properties(device).multi_processor_count))
        matches = {int(row["tokens"]): row for row in rows if all(row[name] == value for name, value in key.items())}
        if not matches:
            raise ValueError(f"no tuned model/device match in {path}")
        counts = sorted(matches) if requested is None else requested
        if not counts:
            raise ValueError("prepare requires at least one tuned token count")
        cache, prepared = {}, {}
        for count in counts:
            if count not in matches:
                raise ValueError(f"no exact tuned tokens={count} in {path}; available: {sorted(matches)}")
            row = matches[count]
            name, params = row["driver"], json.loads(row["params"])
            identity = name, row["params"]
            if identity not in cache:
                if name == "aiter":
                    if not params.get("tuned_config"):
                        raise ValueError("tuned Aiter winner requires its measured config")
                    prepare_one, run_one = aiter(config, tuned_config=params["tuned_config"])
                else:
                    if name not in registry:
                        raise ValueError(f"unknown tuned driver {name!r}; regenerate {path}")
                    prepare_one, run_one = registry[name](config)
                cache[identity] = run_one, prepare_one(weight1, weight2)
            prepared[count] = cache[identity]
        return dict(by_tokens=prepared, device=device)

    def run(hidden_states, weights, topk_ids, topk_weights, output):
        count, prepared = hidden_states.shape[0], weights["by_tokens"]
        if count not in prepared:
            raise ValueError(f"tokens={count} was not prepared; available: {sorted(prepared)}")
        if hidden_states.device != weights["device"]:
            raise ValueError("tuned activations and prepared weights must be on the same device")
        run_one, prepared_weights = prepared[count]
        return run_one(hidden_states, prepared_weights, topk_ids, topk_weights, output)

    return prepare, run


def _ptr(value):
    return 0 if value is None else value.data_ptr()


def _fly_ptr(value):
    import flydsl.compiler as flyc
    import flydsl.expr as fx

    if value is None:
        return flyc.from_c_void_p(fx.Float32, 0)
    dtype = {torch.bfloat16: fx.BFloat16, torch.float32: fx.Float32, torch.int32: fx.Int32}.get(value.dtype, fx.Uint8)
    return flyc.from_c_void_p(dtype, value.data_ptr())


def _sort(config, block_m, tw, ti, out):
    from aiter.fused_moe import moe_sorting

    # Sorting also clears the destination for BF16 atomic accumulation.
    return moe_sorting(ti, tw, config.experts, config.model_dim, config.output_dtype, block_m, output=out)


def _quant_a(quant, x, *, transpose_scale=False):
    backend = "torch" if quant is quantizer.fp8_per_tensor else "hip"
    return quant.apply_a(x, backend=backend, transpose_scale=transpose_scale)


def _bf16_a(config, x):
    if config.quant_scheme.startswith("fp8_"):
        # Legacy kernels consume BF16 A: explicit dynamic QDQ, not native A8W8.
        quant = quantizer.get_quantizer(config.quant_scheme)
        return quant.dequant_a(*_quant_a(quant, x)).to(torch.bfloat16)
    return x


def _jit_config(config, block_m, block_n, down_bn, gate_mode):
    _check_config(config, local=True)
    H, I, q = config.model_dim, config.inter_dim_tp, config.quant_scheme
    _require(gate_mode == "separated", "legacy JIT kernels require separated gate/up packing")
    _require(config.activation == "silu" and config.swiglu_limit is None, "only unclamped SiLU is implemented")
    _require(block_m in (16, 32, 64), "JIT supports block_m=16/32/64")
    _require(block_n in (32, 64, 128) and down_bn in (32, 64, 128), "N tiles must be 32/64/128")
    _require(2 * I % block_n == H % down_bn == 0, "both GEMMs need full N tiles")
    _require(q not in ("a16w8_blockscale", "fp8_blockscale"),
             "legacy split-K reads the wrong first K-group scale; select jit_blockscale")
    _require(q in ("no_quant", "bf16", "a16w8_per_channel", "a16w8_per_tensor", "a16w4",
                   "fp8_ptpc", "fp8_per_tensor", "fp8_per_token_per_tensor"), "unsupported JIT quantizer")
    if q not in ("no_quant", "bf16"):
        _require(block_m * max(block_n, down_bn) <= 4096, "quantized JIT tiles exceed the register allocator's M*N=4096 limit")
    if q in ("no_quant", "bf16"):
        _require(H % 128 == 0, "BF16 split-K needs H divisible by 128")
    elif q == "a16w4":
        _require(H % 1024 == I % 256 == 0 and block_n % 64 == 0,
                 "MXFP4 split-K needs H%1024=0, I%256=0 and gate BN multiple of 64")
        _require(down_bn in (64, 128), "MXFP4 requires LDS down epilogue BN64/128")
    else:
        _require(H % 64 == I % 64 == 0, "FP8 row/tensor scales need H/I divisible by 64")


def _jit_prepare(config, weight1, weight2):
    weights = _prepare_weights(config, weight1, weight2)
    if config.quant_scheme in ("a16w8_per_tensor", "fp8_per_tensor", "fp8_per_token_per_tensor"):
        weights["w1s"] = weights["w1s"].expand(config.experts, 2 * config.inter_dim_tp, 1).contiguous()
        weights["w2s"] = weights["w2s"].expand(config.experts, config.model_dim, 1).contiguous()
    return _shuffle_weights(config, weights)


def jit_splitk(config: MOEconfig, block_m=16, block_n=128, down_bn=None, gate_mode="separated"):
    """Sorted split-K gateup/down; FP8 A is explicit dynamic QDQ/BF16."""
    down_bn = block_n if down_bn is None else down_bn
    _jit_config(config, block_m, block_n, down_bn, gate_mode)
    H, I, K = config.model_dim, config.inter_dim_tp, config.topk
    row_scale = config.quant_scheme != "a16w4"

    def prepare(weight1, weight2):
        return _jit_prepare(config, weight1, weight2)

    @torch.no_grad()
    def run(x, weights, topk_ids, topk_weights, output):
        M = _tokens(config, x, output, block_m)
        if not M:
            return output
        from pyhip.contrib.moe import moe_2stage_splitk

        with torch.cuda.device(x.device):
            w1, w2, s1, s2 = (weights[key] for key in ("w1", "w2", "w1s", "w2s"))
            ids, scores, experts, valid, result = _sort(config, block_m, topk_weights, topk_ids, output)
            metadata, grid = tuple(map(_ptr, (ids, scores, experts, valid))), experts.numel()
            mid = torch.empty((M, K, I), dtype=x.dtype, device=x.device)
            a1 = _bf16_a(config, x)
            moe_2stage_splitk([2 * I // block_n, grid], [256], w1.dtype, K, H, 2 * I,
                             True, block_m, block_n, _ptr(a1), _ptr(w1), _ptr(mid), *metadata, _ptr(s1), M, row_scale)
            a2 = _bf16_a(config, mid)
            moe_2stage_splitk([H // down_bn, grid], [64], w2.dtype, K, I, H,
                             False, block_m, down_bn, _ptr(a2), _ptr(w2), _ptr(result), *metadata, _ptr(s2), M, row_scale)
            return result

    run.activation_path = "fp8_qdq_bf16" if config.quant_scheme.startswith("fp8_") else "bf16"
    return prepare, run


def jit_blockscale(config: MOEconfig, block_m=256, block_n=256, down_path="persistent",
                   down_bn=None, num_oc_splits=None, persistent_workers=None):
    """Native FP8 8-wave pipeline; queue lifecycle remains inside each call."""
    _check_config(config, local=True)
    H, I, E, K = config.model_dim, config.inter_dim_tp, config.experts, config.topk
    _require(config.quant_scheme == "fp8_blockscale", "8-wave path implements fp8_blockscale only")
    _require(config.activation == "silu" and config.swiglu_limit is None, "only unclamped SiLU is implemented")
    _require(block_m in (128, 256) and block_n == 256, "8-wave gateup requires BM128/256 and BN256")
    _require(H % 128 == I % 128 == 0, "H and local I must be divisible by 128; no padding")
    _require(down_path in ("persistent", "tiled"), "down_path must be persistent or tiled")
    down_bn = _positive("down_bn", down_bn if down_bn is not None else (64 if down_path == "persistent" else 256))
    if down_path == "persistent":
        num_oc_splits = _positive("num_oc_splits", num_oc_splits if num_oc_splits is not None else 2)
        persistent_workers = _positive("persistent_workers", persistent_workers if persistent_workers is not None else 256)
        _require(down_bn == 64 and I == 256,
                 "persistent down requires BN64, I256; I128 has an uninitialized MFMA queue tail, use tiled")
        _require(H % (num_oc_splits * 128) == 0 and H // num_oc_splits >= 256,
                 "each OC split needs complete N128 scale blocks and at least four N64 tiles")
        _require(E * H * I < 2**32, "persistent down weight offsets must fit 32 bits")
        lds_down = 4 * down_bn * I + 8 * block_m + 4 * (H // num_oc_splits // 128) * (I // 128)
    else:
        _require(num_oc_splits is None and persistent_workers is None, "queue options only apply to persistent down")
        _require(down_bn == 256 and H % 256 == 0, "tiled down requires BN256 and H%256=0")
        lds_down = 256 * (block_m + down_bn) + 8 * block_m + 4096 + 8 * (I // 128)
    lds_gate = 256 * (block_m + block_n) + 8 * block_m + 4096 + 8 * (H // 128)
    _require(max(lds_gate, lds_down) <= 160 * 1024, "8-wave workgroup exceeds gfx950 LDS capacity")
    quant = quantizer.fp8_blockscale

    def prepare(weight1, weight2):
        _require(_device(config, weight2.device) == "gfx950", "8-wave FP8 MFMA/LDS layout requires gfx950")
        return _shuffle_weights(config, _prepare_weights(config, weight1, weight2))

    @torch.no_grad()
    def run(x, weights, topk_ids, topk_weights, output):
        M = _tokens(config, x, output, block_m)
        if not M:
            return output
        from pyhip.contrib.moe_gemm_8wave import moe_gemm_8wave_g1u1, moe_gemm_8wave_down

        with torch.cuda.device(x.device):
            w1, w2, s1, s2 = (weights[key] for key in ("w1", "w2", "w1s", "w2s"))
            ids, scores, experts, valid, result = _sort(config, block_m, topk_weights, topk_ids, output)
            grid, metadata = experts.numel(), tuple(map(_ptr, (ids, scores, experts, valid)))
            mid = torch.empty((M, K, I), dtype=x.dtype, device=x.device)
            routed = torch.empty((M, K, H), dtype=x.dtype, device=x.device)
            a1, as1 = _quant_a(quant, x, transpose_scale=True)
            tasks = (2 * I // block_n) * grid
            moe_gemm_8wave_g1u1([tasks], [512], False, "fp8", block_m, block_n,
                               E, 2 * I, H, True, True, K,
                               *metadata, _ptr(w1), _ptr(s1), _ptr(a1), _ptr(as1), _ptr(mid), M, tasks)
            a2, as2 = _quant_a(quant, mid, transpose_scale=True)
            if down_path == "persistent":
                # No self-reset: allocate a fresh zeroed queue counter for every call.
                head = torch.zeros(1, dtype=torch.int32, device=x.device)
                moe_gemm_8wave_down([persistent_workers], [512], False, "fp8", block_m, down_bn,
                                   E, H, I, num_oc_splits, False, True, K,
                                   *metadata, _ptr(w2), _ptr(s2), _ptr(a2), _ptr(as2), _ptr(routed), M, _ptr(head))
            else:
                tasks = (H // down_bn) * grid
                moe_gemm_8wave_g1u1([tasks], [512], False, "fp8", block_m, down_bn,
                                   E, H, I, False, True, K,
                                   *metadata, _ptr(w2), _ptr(s2), _ptr(a2), _ptr(as2), _ptr(routed), M, tasks)
            torch.sum(routed, dim=1, out=result)
            return result

    run.activation_path = "native_fp8_blockscale"
    return prepare, run


def _batch_config(config, down_bn):
    _jit_config(config, 16, 32, down_bn, "separated")
    _require(config.quant_scheme != "a16w4", "legacy batch gateup only supports BF16/row-scaled FP8")
    step = 128 if config.quant_scheme in ("no_quant", "bf16") else 256
    _require(config.model_dim >= 2 * step and config.model_dim % step == 0,
             f"runtime-K gateup needs H >= {2 * step}, H%{step}=0")


@register
def jit_batch1(config: MOEconfig):
    """Direct-route M=1 only; no sorting, output cleared on every call."""
    _batch_config(config, 32)
    H, I, K = config.model_dim, config.inter_dim_tp, config.topk
    _require(I >= (64 if config.quant_scheme in ("no_quant", "bf16") else 128),
             "runtime-K down requires at least two K blocks")

    def prepare(weight1, weight2):
        return _jit_prepare(config, weight1, weight2)

    @torch.no_grad()
    def run(x, weights, topk_ids, topk_weights, output):
        M = _tokens(config, x, output, max_tokens=1)
        if not M:
            return output
        from pyhip.contrib.moe import moe_gemm_batch1

        with torch.cuda.device(x.device):
            w1, w2, s1, s2 = (weights[key] for key in ("w1", "w2", "w1s", "w2s"))
            output.zero_()
            mid = torch.empty((M, K, I), dtype=x.dtype, device=x.device)
            a1 = _bf16_a(config, x)
            moe_gemm_batch1([2 * I // 32, K], [256], w1.dtype, True,
                            _ptr(a1), _ptr(w1), _ptr(mid), _ptr(topk_ids), _ptr(topk_weights), _ptr(s1), 1, 2 * I, H)
            a2 = _bf16_a(config, mid)
            moe_gemm_batch1([H // 32, K], [64], w2.dtype, False,
                            _ptr(a2), _ptr(w2), _ptr(output), _ptr(topk_ids), _ptr(topk_weights), _ptr(s2), 1, H, I)
            return output

    run.activation_path = "fp8_qdq_bf16" if config.quant_scheme.startswith("fp8_") else "bf16"
    return prepare, run


def jit_batch(config: MOEconfig, down_bn=64):
    """Legacy M16/N32 gateup followed by split-K down."""
    _batch_config(config, down_bn)
    H, I, K = config.model_dim, config.inter_dim_tp, config.topk

    def prepare(weight1, weight2):
        return _jit_prepare(config, weight1, weight2)

    @torch.no_grad()
    def run(x, weights, topk_ids, topk_weights, output):
        M = _tokens(config, x, output)
        if not M:
            return output
        from pyhip.contrib.moe import moe_gemm_batch, moe_2stage_splitk

        with torch.cuda.device(x.device):
            w1, w2, s1, s2 = (weights[key] for key in ("w1", "w2", "w1s", "w2s"))
            ids, scores, experts, valid, result = _sort(config, 16, topk_weights, topk_ids, output)
            grid, metadata = experts.numel(), tuple(map(_ptr, (ids, scores, experts, valid)))
            mid = torch.empty((M, K, I), dtype=x.dtype, device=x.device)
            a1 = _bf16_a(config, x)
            moe_gemm_batch([2 * I // 32, grid], [256], w1.dtype, True,
                           _ptr(a1), _ptr(w1), _ptr(mid), *metadata, _ptr(s1), M, 2 * I, H, K)
            a2 = _bf16_a(config, mid)
            moe_2stage_splitk([H // down_bn, grid], [64], w2.dtype, K, I, H,
                             False, 16, down_bn, _ptr(a2), _ptr(w2), _ptr(result), *metadata, _ptr(s2), M, True)
            return result

    run.activation_path = "fp8_qdq_bf16" if config.quant_scheme.startswith("fp8_") else "bf16"
    return prepare, run


def jit_fused(config: MOEconfig, block_m=16, block_n=128):
    """Register-resident one-stage BF16 path; this kernel still needs sorting."""
    _jit_config(config, block_m, block_n, block_n, "separated")
    H, I, K = config.model_dim, config.inter_dim_tp, config.topk
    _require(config.quant_scheme in ("no_quant", "bf16"), "one-stage down supports BF16 weights only")
    _require(block_m in (16, 32) and block_n in (64, 128) and I <= 384,
             "register-resident range is BM16/32, BN64/128, local I<=384")

    def prepare(weight1, weight2):
        return _jit_prepare(config, weight1, weight2)

    @torch.no_grad()
    def run(x, weights, topk_ids, topk_weights, output):
        M = _tokens(config, x, output, block_m)
        if not M:
            return output
        from pyhip.contrib.moe import moe_1stage_splitk

        with torch.cuda.device(x.device):
            w1, w2, s1, s2 = (weights[key] for key in ("w1", "w2", "w1s", "w2s"))
            ids, scores, experts, valid, result = _sort(config, block_m, topk_weights, topk_ids, output)
            metadata = tuple(map(_ptr, (ids, scores, experts, valid)))
            moe_1stage_splitk([1, experts.numel()], [256], w1.dtype, K, H, 2 * I, H,
                             block_m, block_n, _ptr(x), _ptr(w1), _ptr(s1), _ptr(w2), _ptr(s2), _ptr(result), *metadata, M)
            return result

    run.activation_path = "bf16"
    return prepare, run


def jit_loopn(config: MOEconfig, atomic_write=True):
    """Legacy batch gateup and fixed loop-N down, optionally routed reduction."""
    _batch_config(config, 64)
    H, I, K = config.model_dim, config.inter_dim_tp, config.topk
    _require(config.quant_scheme in ("a16w8_per_channel", "a16w8_per_tensor", "fp8_ptpc", "fp8_per_tensor", "fp8_per_token_per_tensor"),
             "loop-N down implements FP8 row-scaled weights only")
    _require(H % 1024 == 0 and I <= 512, "loop-N requires H%1024=0, I<=512")
    if not isinstance(atomic_write, bool):
        raise TypeError("atomic_write must be bool")

    def prepare(weight1, weight2):
        return _jit_prepare(config, weight1, weight2)

    @torch.no_grad()
    def run(x, weights, topk_ids, topk_weights, output):
        M = _tokens(config, x, output)
        if not M:
            return output
        from pyhip.contrib.moe import moe_gemm_batch, moe_2stage_down_loopn

        with torch.cuda.device(x.device):
            w1, w2, s1, s2 = (weights[key] for key in ("w1", "w2", "w1s", "w2s"))
            ids, scores, experts, valid, result = _sort(config, 16, topk_weights, topk_ids, output)
            grid, metadata = experts.numel(), tuple(map(_ptr, (ids, scores, experts, valid)))
            mid = torch.empty((M, K, I), dtype=x.dtype, device=x.device)
            a1 = _bf16_a(config, x)
            moe_gemm_batch([2 * I // 32, grid], [256], w1.dtype, True,
                           _ptr(a1), _ptr(w1), _ptr(mid), *metadata, _ptr(s1), M, 2 * I, H, K)
            a2 = _bf16_a(config, mid)
            routed = result if atomic_write else torch.empty((M, K, H), dtype=x.dtype, device=x.device)
            moe_2stage_down_loopn([H // 1024, grid], [256], w2.dtype, K, I, H,
                                 16, 16, _ptr(a2), _ptr(w2), _ptr(routed), *metadata, _ptr(s2), M, True, 1024, atomic_write, 3)
            if not atomic_write:
                torch.sum(routed, dim=1, out=result)
            return result

    run.activation_path = "fp8_qdq_bf16" if config.quant_scheme.startswith("fp8_") else "bf16"
    return prepare, run


def jit_mxfp4(config: MOEconfig, block_m=64, block_n=128):
    """Native A4W4 GEMMs with dynamic A quant/sort and TOPK reduction."""
    _check_config(config, local=True)
    H, I, E, K = config.model_dim, config.inter_dim_tp, config.experts, config.topk
    _require(config.quant_scheme == "a4w4", "native MXFP4 requires a4w4")
    _require(config.activation == "silu" and config.swiglu_limit is None, "only unclamped SiLU is implemented")
    _require(block_m in (64, 128) and block_n == 128, "native MXFP4 tiles are BM64/128, BN128")
    _require(H % 1024 == I % 256 == 0 and K <= 16, "native MXFP4/reducer needs H%1024=0, I%256=0, TOPK<=16")

    def prepare(weight1, weight2):
        return _shuffle_weights(config, _prepare_weights(config, weight1, weight2))

    @torch.no_grad()
    def run(x, weights, topk_ids, topk_weights, output):
        M = _tokens(config, x, output, block_m)
        if not M:
            return output
        from aiter.utility.fp4_utils import moe_mxfp4_sort
        from pyhip.contrib.moe_gemm_mxfp4 import moe_gemm_final_reduce_bf16, moe_gemm_mxfp4

        with torch.cuda.device(x.device):
            w1, w2, s1, s2 = (weights[key] for key in ("w1", "w2", "w1s", "w2s"))
            ids, scores, experts, valid, result = _sort(config, block_m, topk_weights, topk_ids, output)
            grid, metadata = experts.numel(), tuple(map(_ptr, (ids, scores, experts, valid)))
            mid = torch.empty((M, K, I), dtype=x.dtype, device=x.device)
            routed = torch.empty((M, K, H), dtype=x.dtype, device=x.device)

            def stage(a, w, s, dst, gateup):
                aq, scale = quantizer.a4w4.apply_a(a, backend="hip")
                scale = moe_mxfp4_sort(scale, sorted_ids=ids, num_valid_ids=valid, token_num=M, block_size=block_m)
                moe_gemm_mxfp4([w.shape[1] // 128, grid], [256], block_m, 128,
                              E, w.shape[1], w.shape[2], gateup, K,
                              *metadata, _ptr(w), _ptr(s), _ptr(aq), _ptr(scale), _ptr(dst), M)

            stage(x, w1, s1, mid, True)
            stage(mid, w2, s2, routed, False)
            moe_gemm_final_reduce_bf16([512], [64], K, H, _ptr(routed), _ptr(result), M // 512, M % 512, M)
            return result

    run.activation_path = "native_mxfp4"
    return prepare, run


def _fly_config(config, gate_mode):
    _check_config(config, local=True)
    _require(gate_mode in ("separated", "interleave"), "gate_mode is separated/interleave packing, not an activation mode")
    _require(gate_mode == "separated" or config.quant_scheme == "a16w4", "only MXFP4 Fly kernels support interleaved packing")
    _require(config.activation in ("silu", "swiglu", "situv2"), "Fly kernels implement SiLU/SwiGLU/SiTUv2 only")
    if config.activation == "silu":
        _require(config.swiglu_limit is None, "SiLU kernel cannot apply a clamp")
    elif config.activation == "situv2":
        _require(config.swiglu_limit is not None and config.swiglu_limit > 0,
                 "SiTUv2 kernel always clamps: supply a positive swiglu_limit")
    else:
        _require(config.swiglu_limit is None or config.swiglu_limit > 0, "SwiGLU kernel does not implement clamp=0")


def _fly_prepare(config, weight1, weight2, *, gate_mode):
    return _shuffle_weights(config, _prepare_weights(config, weight1, weight2),
                            gate_mode=gate_mode, down_interleave=config.quant_scheme == "a16w4")


def _compile_fly(config, *, alg, block_m, gate_m, gate_n, down_n, gate_mode,
                 gate_k=None, down_k=None, down_path="default", padding=None):
    """Called inside run on its current device, never during host preflight."""
    from pyhip.contrib.flydsl.moe_gemm_splitk import compile_gemm

    q = config.quant_scheme
    wdtype = "bf16" if q in ("no_quant", "bf16") else "fp4" if q == "a16w4" else "fp8"
    wquant = ("no" if wdtype == "bf16" else "mxfp4" if wdtype == "fp4" else
              "ptpc" if q in ("fp8_ptpc", "a16w8_per_channel") else "per_tensor")
    aquant = ("per_tensor" if q == "fp8_per_tensor" else "ptpc") if alg == "prefill_1x4" and wdtype == "fp8" else "no"
    common = dict(weight_dtype=wdtype, weight_quant_type=wquant, act_quant_type=aquant,
                  TOPK=config.topk, E=config.experts, alg=alg, METADATA_TILE_SIZE_M=block_m)
    gate = compile_gemm(N=2 * config.inter_dim_tp, K=config.model_dim, BLOCK_TILE_SIZE_M=gate_m,
                        BLOCK_TILE_SIZE_N=gate_n, stage="gateup", activation=config.activation,
                        swiglu_limit=config.swiglu_limit, situ_beta=config.beta, situ_linear_beta=config.linear_beta,
                        mxfp4_gate_up_interleaved=gate_mode == "interleave", tile_k=gate_k, **common)
    down = compile_gemm(N=config.model_dim, K=config.inter_dim_tp, BLOCK_TILE_SIZE_M=block_m,
                        BLOCK_TILE_SIZE_N=down_n, stage="down", USE_ATOMIC_WRITE=alg != "prefill_1x4",
                        down_path=down_path, down_output_padding_bytes=padding, tile_k=down_k, activation="silu", **common)
    return gate, down


def _fly_split_config(config, block_m, gate_n, down_n, gate_mode, *, direct=False):
    _fly_config(config, gate_mode)
    H, I, q = config.model_dim, config.inter_dim_tp, config.quant_scheme
    _require(q in ("no_quant", "bf16", "a16w8_per_channel", "a16w8_per_tensor", "a16w4",
                   "fp8_ptpc", "fp8_per_tensor", "fp8_per_token_per_tensor"), "unsupported split-K/decode quantizer")
    _require(block_m in (16, 32, 64, 128, 256), "metadata BM must be 16/32/64/128/256")
    _require(not direct or block_m == 16, "direct-route decode requires BM16")
    _require(gate_n in ((32, 64, 128) if direct else (64, 128)) and down_n in (32, 64, 128),
             "unsupported split-K/decode N tile")
    _require(2 * I % gate_n == H % down_n == 0, "both GEMMs need full N tiles")
    _require(H % 128 == 0, "split-K gate H must be divisible by 128")
    if q == "a16w4":
        _require(H % 512 == I % 128 == 0, "A16W4 needs H%512=0 and I%128=0")
    elif q not in ("no_quant", "bf16"):
        _require(I % 64 == 0, "FP8 down needs I%64=0")


def fly_splitk(config: MOEconfig, block_m=64, g1u1_block_n=128, down_bn=64, gate_mode=None):
    """Sorted two-stage split-K with atomic down."""
    gate_mode = gate_mode or ("interleave" if config.quant_scheme == "a16w4" else "separated")
    _fly_split_config(config, block_m, g1u1_block_n, down_bn, gate_mode)
    I, K = config.inter_dim_tp, config.topk
    kernels = {}

    def prepare(weight1, weight2):
        return _fly_prepare(config, weight1, weight2, gate_mode=gate_mode)

    @torch.no_grad()
    def run(x, weights, topk_ids, topk_weights, output):
        M = _tokens(config, x, output, block_m)
        if not M:
            return output
        from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

        with torch.cuda.device(x.device):
            if x.device not in kernels:
                kernels[x.device] = _compile_fly(config, alg="splitk", block_m=block_m, gate_m=block_m,
                                                 gate_n=g1u1_block_n, down_n=down_bn, gate_mode=gate_mode)
            gate, down = kernels[x.device]
            stream = torch.cuda.current_stream(x.device)
            w1, w2, s1, s2 = (weights[key] for key in ("w1", "w2", "w1s", "w2s"))
            ids, scores, experts, valid, result = _sort(config, block_m, topk_weights, topk_ids, output)
            grid, metadata = experts.numel(), tuple(map(_fly_ptr, (ids, scores, experts, valid)))
            mid = torch.empty((M, K, I), dtype=x.dtype, device=x.device)
            a1 = _bf16_a(config, x)
            _run_compiled(gate, *map(_fly_ptr, (a1, w1, mid)), *metadata, _fly_ptr(s1), M, grid, stream)
            a2 = _bf16_a(config, mid)
            _run_compiled(down, *map(_fly_ptr, (a2, w2, result)), *metadata, _fly_ptr(s2), M, grid, stream)
            return result

    run.activation_path = "fp8_qdq_bf16" if config.quant_scheme.startswith("fp8_") else "bf16"
    return prepare, run


def fly_decode(config: MOEconfig, block_m=16, g1u1_block_n=32, down_bn=64, gate_mode=None):
    """Direct routing; grid.z handles multiple tokens, no sorting."""
    gate_mode = gate_mode or ("interleave" if config.quant_scheme == "a16w4" else "separated")
    _fly_split_config(config, block_m, g1u1_block_n, down_bn, gate_mode, direct=True)
    I, K = config.inter_dim_tp, config.topk
    kernels = {}

    def prepare(weight1, weight2):
        return _fly_prepare(config, weight1, weight2, gate_mode=gate_mode)

    @torch.no_grad()
    def run(x, weights, topk_ids, topk_weights, output):
        M = _tokens(config, x, output, block_m)
        if not M:
            return output
        from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

        with torch.cuda.device(x.device):
            if x.device not in kernels:
                kernels[x.device] = _compile_fly(config, alg="batch1", block_m=block_m, gate_m=block_m,
                                                 gate_n=g1u1_block_n, down_n=down_bn, gate_mode=gate_mode)
            gate, down = kernels[x.device]
            stream = torch.cuda.current_stream(x.device)
            w1, w2, s1, s2 = (weights[key] for key in ("w1", "w2", "w1s", "w2s"))
            output.zero_()
            mid = torch.empty((M, K, I), dtype=x.dtype, device=x.device)
            a1 = _bf16_a(config, x)
            _run_compiled(gate, *map(_fly_ptr, (a1, w1, mid, topk_ids, topk_weights, s1)), M, stream)
            a2 = _bf16_a(config, mid)
            _run_compiled(down, *map(_fly_ptr, (a2, w2, output, topk_ids, topk_weights, s2)), M, stream)
            return output

    run.activation_path = "fp8_qdq_bf16" if config.quant_scheme.startswith("fp8_") else "bf16"
    return prepare, run


def fly_prefill(config: MOEconfig, sort_block_m=64, stage1_blockn=128, stage2_blockn=64,
                atomic=False, gate_mode="separated", *, stage1_blockm=None, stage1_tile_k=None,
                stage2_tile_k=None, down_path="default", down_output_padding_bytes=None):
    """BF16/native FP8 prefill with routed output and sorted TOPK reduction."""
    _fly_config(config, gate_mode)
    _require(atomic is False, "these prefill down kernels require routed output plus reduction")
    q, H, I, E, K = config.quant_scheme, config.model_dim, config.inter_dim_tp, config.experts, config.topk
    _require(q in ("no_quant", "bf16", "fp8_ptpc", "fp8_per_tensor", "fp8_per_token_per_tensor"), "unsupported prefill quantizer")
    fp8 = q not in ("no_quant", "bf16")
    bm, gn, dn = sort_block_m, stage1_blockn, stage2_blockn
    gm = stage1_blockm if stage1_blockm is not None else bm
    gk = stage1_tile_k if stage1_tile_k is not None else (128 if fp8 else 64)
    dk = stage2_tile_k if stage2_tile_k is not None else 128
    padding = down_output_padding_bytes
    for name, value in (("sort_block_m", bm), ("stage1_blockm", gm), ("stage1_blockn", gn), ("stage2_blockn", dn),
                        ("stage1_tile_k", gk), ("stage2_tile_k", dk)):
        _positive(name, value)
    _require(bm in (16, 32, 64, 128, 256), "metadata BM must be 16/32/64/128/256")
    _require(gm in (32, 64, 128, 256) and gm <= bm and bm % gm == 0, "gate BM must be 32/64/128/256 and divide metadata BM")
    _require(gn in (128, 256), "prefill gate BN must be 128/256")
    _require(2 * I % gn == H % dn == 0, "both GEMMs need full N tiles")
    _require(gk in ((128, 256) if fp8 else (64, 128)) and H % (2 * gk) == 0, "gate prefill needs an even count of valid K tiles")
    elem_bytes = 1 if fp8 else 2
    _require(gm * gn <= 2 * gm * gk * elem_bytes <= 65536, "gate CShuffle/A ping-pong exceeds LDS layout/capacity")
    if down_path in ("default", "1x4_64x256"):
        _require(stage2_tile_k is None, "down kernel has fixed microtile; stage2_tile_k only customizes 8x1")
    if padding is not None and (isinstance(padding, bool) or not isinstance(padding, int)):
        raise TypeError("down_output_padding_bytes must be an integer or None")
    if down_path == "default":
        _require(dn == 64 and padding is None, "default down is BN64 without padding")
        _require(H % 128 == 0 and I % (64 // elem_bytes) == 0, "default down needs complete H128/K microtiles")
        _require(bm * I * elem_bytes % 4096 == 0, "default down needs BM*I*element_bytes divisible by 4096")
        _require(bm * I * elem_bytes <= 65536, "default down caches all A in <=64KiB LDS")
    else:
        _require(fp8, "dedicated down paths are FP8 only")
        _require(padding in (0, 32, 64, 128), "dedicated down needs explicit 0/32/64/128 byte row padding")
        if down_path == "1x4_64x256":
            _require(bm == 64 and dn == 256 and I % 64 == 0, "1x4_64x256 needs BM64/BN256, I%64=0")
            _require(64 * I + (1024 if q == "fp8_ptpc" else 0) + 8192 <= 65536, "1x4 down exceeds 64KiB LDS")
        elif down_path in ("8x1", "8x1_compact"):
            _require(bm == (256 if down_path == "8x1" else 64) and dn == 128, "8x1 needs BM256/BN128; compact BM64/BN128")
            _require(I in (192, 256, 320, 384, 512, 640), "8x1 I must be 192/256/320/384/512/640")
            _require(dk == 128 or (dk == 192 and I in (192, 320)), "invalid 8x1 tile K")
            _require(down_path != "8x1_compact" or E <= 2048, "compact task builder supports E<=2048")
        else:
            raise ValueError(f"unknown down_path: {down_path!r}")
    quant, kernels = quantizer.get_quantizer(q), {}

    def prepare(weight1, weight2):
        return _fly_prepare(config, weight1, weight2, gate_mode=gate_mode)

    @torch.no_grad()
    def run(x, weights, topk_ids, topk_weights, output):
        M = _tokens(config, x, output, bm)
        if not M:
            return output
        from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled
        from pyhip.contrib.flydsl.moe_gemm_splitk import invert_sorted_ids, sorted_sum

        with torch.cuda.device(x.device):
            if x.device not in kernels:
                kernels[x.device] = _compile_fly(config, alg="prefill_1x4", block_m=bm, gate_m=gm, gate_n=gn, down_n=dn,
                                                 gate_mode=gate_mode, gate_k=gk, down_k=dk, down_path=down_path, padding=padding)
            gate, down = kernels[x.device]
            stream = torch.cuda.current_stream(x.device)
            w1, w2, s1, s2 = (weights[key] for key in ("w1", "w2", "w1s", "w2s"))
            ids, scores, experts, valid, result = _sort(config, bm, topk_weights, topk_ids, output)
            grid, metadata = experts.numel(), tuple(map(_fly_ptr, (ids, scores, experts, valid)))
            mid = torch.empty((M, K, I), dtype=x.dtype, device=x.device)
            a1, as1 = _quant_a(quant, x) if fp8 else (x, None)
            _run_compiled(gate, *map(_fly_ptr, (a1, w1, mid)), *metadata, *map(_fly_ptr, (s1, as1)), M, grid, stream)
            a2, as2 = _quant_a(quant, mid) if fp8 else (mid, None)
            routed = torch.empty((grid * bm, H + (padding or 0) // 2), dtype=x.dtype, device=x.device)
            compact = ()
            if down_path == "8x1_compact":
                from pyhip.contrib.flydsl.moe_gemm_2stage.gemm2_8x1_compact import allocate_task_buffers
                tasks = allocate_task_buffers(experts, E)
                compact = (*map(_fly_ptr, tasks), tasks[0].shape[0], tasks[1].shape[0])
            _run_compiled(down, *map(_fly_ptr, (a2, w2, routed)), *metadata, *map(_fly_ptr, (s2, as2)), M, grid, *compact, stream)
            loc = torch.empty((M, K), dtype=torch.int32, device=x.device)
            invert_sorted_ids(K)(ids, loc, valid, ids.numel(), M)
            sorted_sum(K, H, padding)(loc, routed, result, M)
            return result

    run.activation_path = "native_fp8" if fp8 else "bf16"
    return prepare, run


def prefill_bf16(config: MOEconfig, **tiles):
    _require(config.quant_scheme in ("no_quant", "bf16"), "prefill_bf16 requires BF16 weights/activations")
    return fly_prefill(config, **tiles)


def prefill_fp8(config: MOEconfig, **tiles):
    _require(config.quant_scheme in ("fp8_ptpc", "fp8_per_tensor", "fp8_per_token_per_tensor"), "prefill_fp8 requires FP8 weights/activations")
    return fly_prefill(config, **tiles)


@register
def aiter_a4w4_unshuffled(config: MOEconfig):
    _require(config.quant_scheme == "a4w4", "natural-weight Aiter variant requires A4W4")
    return aiter(config._replace(preshuffle=False))


@register
def prefill_fp8_1x4_64x256(config: MOEconfig):
    return prefill_fp8(config, sort_block_m=64, stage1_blockn=128, stage2_blockn=256,
                       down_path="1x4_64x256", down_output_padding_bytes=0)


@register
def prefill_fp8_8x1(config: MOEconfig):
    return prefill_fp8(config, sort_block_m=256, stage1_blockm=64, stage1_blockn=128, stage2_blockn=128,
                       down_path="8x1", down_output_padding_bytes=0)


@register
def prefill_fp8_8x1_compact(config: MOEconfig):
    return prefill_fp8(config, sort_block_m=64, stage1_blockn=128, stage2_blockn=128,
                       down_path="8x1_compact", down_output_padding_bytes=0)


def _register_fixed(name, factory, **options):
    # Small tile grids do not need dozens of handwritten wrapper functions.
    def candidate(config):
        return factory(config, **options)
    candidate.__name__ = name
    register(candidate)


for _factory in (jit_splitk, jit_blockscale, jit_batch, jit_fused, jit_loopn, jit_mxfp4, fly_splitk, fly_decode, prefill_bf16, prefill_fp8):
    _register_fixed(_factory.__name__, _factory)
for _bm in (16, 32, 64):
    for _bn in (64, 128):
        _register_fixed(f"jit_splitk_{_bm}_{_bn}_True", jit_splitk, block_m=_bm, block_n=_bn)
        _register_fixed(f"fly_splitk_{_bm}_{_bn}", fly_splitk, block_m=_bm, g1u1_block_n=_bn, down_bn=64)
for _factory in (prefill_bf16, prefill_fp8):
    for _bm in (32, 64):
        for _bn in (128, 256):
            _register_fixed(f"{_factory.__name__}_{_bm}_{_bn}_64_True", _factory,
                            sort_block_m=_bm, stage1_blockn=_bn, stage2_blockn=64)
for _bm in (128, 256):
    for _split in (1, 2, 4):
        _register_fixed(f"jit_blockscale_{_bm}_256_persistent_{_split}_True", jit_blockscale,
                        block_m=_bm, down_path="persistent", num_oc_splits=_split)
    _register_fixed(f"jit_blockscale_{_bm}_256_tiled_True", jit_blockscale, block_m=_bm, down_path="tiled")
del _factory, _bm, _bn, _split