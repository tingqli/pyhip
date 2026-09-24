"""自动调优完整 MoE 流程，候选包括 Aiter、PyHIP ASM 和 FlyDSL。

接口与 Aiter 保持一致。调用方负责权重和 scale 的 shuffle，并分别保留 w1/w2
上的 ``is_shuffled`` 属性。正常执行时不转换或缓存权重；校验时才逐个 expert
还原权重，用独立的 Torch 参考实现检查所有候选，Aiter 也不例外。

首次未命中缓存时执行校验和计时，后续直接复用选中的配置。
FLYDSL_AUTOTUNE=1 可强制重新调优，graph capture 前需关闭该开关。
FLYDSL_AUTOTUNE_CONFIG_DIR 用于指定离线配置的导出目录。
计时包含排序、激活量化、两次 GEMM 和 reduce，不包含参考计算和精度检查。
精度阈值沿用 test_moe 的 ``calc_diff <= 0.02``，并非逐元素 2% 相对误差。
"""

import json
import math
import os

import aiter
import torch
from aiter.fused_moe import fused_moe as _aiter_fused_moe
from aiter.fused_moe import moe_sorting
from aiter.ops.flydsl.kernels.mega_moe_gfx1250.types import Stage2ScatterContext
from aiter.ops.flydsl.moe_common import GateMode
from flydsl.autotune import Config, autotune

from pyhip import calc_diff
from pyhip.testing.moe import torch_reference as _torch_reference

__all__ = ["fused_moe", "record_dispatch", "last_dispatch"]

# 串行诊断用：成功返回后读取 last_dispatch，不保存 tensor；graph replay 不更新。
record_dispatch = False
last_dispatch = None


def ceil_pow2(n: int) -> int:
    """将 batch size 向上取整到 2 的幂，让同一 bucket 内的调用复用配置。"""
    return 1 << (max(1, n) - 1).bit_length()


def _native_kind(call):
    """根据 shape、dtype 和布局判断是否支持本地调优，其余情况交给 Aiter。"""
    x, w1, w2 = (call[name] for name in ("hidden_states", "w1", "w2"))
    ids, weights = call["topk_ids"], call["topk_weight"]
    if not x.is_cuda or x.ndim != 2 or w1.ndim != 3 or w2.ndim != 3:
        return None
    if any(t.requires_grad for t in (x, w1, w2)):
        return None  # 这些 kernel 只支持推理，未实现 backward。
    if x.dtype != torch.bfloat16 or call["dtype"] not in (None, torch.bfloat16):
        return None
    if any(getattr(w, attr, None) is not None for w in (w1, w2)
           for attr in ("inter_real", "aiter_original_k", "aiter_padded_k")):
        return None  # 这些 kernel 不支持 Aiter 的特殊 padding 布局。
    if w1.dtype != w2.dtype:
        return None
    if any(call[name] is not None for name in (
        "expert_mask", "num_local_tokens", "a1_scale", "a2_scale", "bias1", "bias2",
        "shared_w1", "shared_w2", "shared_w1_scale", "shared_w2_scale", "stage2_scatter",
    )):
        return None
    if (call["doweight_stage1"] or call["hidden_pad"] or call["intermediate_pad"]
            or call["splitk"] or call["shared_expert_id"] != -1
            or call["moe_sorting_dispatch_policy"] != 0):
        return None
    if call["activation"] not in (
        aiter.ActivationType.Silu, aiter.ActivationType.Swiglu, aiter.ActivationType.Situv2,
        aiter.ActivationType.Gelu,
    ):
        return None
    if call["activation"] == aiter.ActivationType.Silu and call["swiglu_limit"]:
        return None  # 现有 SiLU epilogue 未实现 Aiter 的 clamp 参数。
    if call["activation"] == aiter.ActivationType.Swiglu and call["swiglu_limit"] == 0:
        return None  # 现有 SwiGLU 会把 0 当作默认上限，与 Aiter 的含义不同。
    if ids.ndim != 2 or weights.shape != ids.shape or ids.shape[0] != x.shape[0]:
        return None
    if ids.dtype != torch.int32 or weights.dtype != torch.float32:
        return None
    if not (0 < x.shape[0] < 2**24 and 0 < ids.shape[1] < 256):
        return None
    if not all(t.device == x.device and t.is_contiguous() for t in (x, w1, w2, ids, weights)):
        return None
    arch = torch.cuda.get_device_properties(x.device).gcnArchName.split(":")[0]
    if arch not in ("gfx942", "gfx950"):
        return None

    quant = call["quant_type"]
    if w1.dtype == torch.bfloat16 and quant == aiter.QuantType.No:
        kind = "bf16"
    elif w1.dtype == (torch.float8_e4m3fn if arch == "gfx950" else torch.float8_e4m3fnuz):
        kind = {
            aiter.QuantType.per_Token: "ptpc",
            aiter.QuantType.per_Tensor: "per_tensor",
            aiter.QuantType.per_128x128: "block",
            aiter.QuantType.per_1x128: "block",
        }.get(quant)
    elif (arch == "gfx950" and w1.dtype == torch.float4_e2m1fn_x2
            and quant == aiter.QuantType.per_1x32):
        kind = "fp4"
    else:
        return None
    if kind is None:
        return None
    gelu = call["activation"] == aiter.ActivationType.Gelu
    if gelu and (kind != "bf16" or any(call[name] is not None for name in
                                      ("swiglu_limit", "beta", "linear_beta"))):
        return None  # 这里只接入非 gated、无量化的 GELU kernel。
    mode = call["gate_mode"]
    if mode != GateMode.SEPARATED and not (kind == "fp4" and mode == GateMode.INTERLEAVE):
        return None
    if mode == GateMode.INTERLEAVE and not all(getattr(w, "is_shuffled", False) for w in (w1, w2)):
        return None

    # FP4 每个字节存两个元素，w2 的最后一维乘 2 才是实际中间维度。
    e, h, i = w2.shape
    i *= 2 if kind == "fp4" else 1
    if e <= 0 or h <= 0 or i <= 0 or x.shape[1] != h:
        return None
    if tuple(w1.shape) != (e, i if gelu else 2 * i, h // (2 if kind == "fp4" else 1)):
        return None
    if h % 128 or i % 64:
        return None
    if kind == "bf16":
        return kind if call["w1_scale"] is None and call["w2_scale"] is None else None
    for name, weight, rows, cols in (("w1_scale", w1, 2 * i, h), ("w2_scale", w2, h, i)):
        scale = call[name]
        if scale is None or scale.device != x.device or not scale.is_contiguous():
            return None
        if kind == "fp4":
            # E8M0 shuffle 将 scale 列数 K/32 补齐到 8 的倍数，检查容量时要算上 padding。
            if scale.dtype not in (torch.uint8, torch.float8_e8m0fnu):
                return None
            if getattr(weight, "is_shuffled", False):
                if scale.numel() < e * rows * ((cols + 255) // 256 * 8):
                    return None
            elif scale.numel() != e * rows * (cols // 32):
                return None
        else:
            count = e * rows if kind == "ptpc" else e
            if kind == "block":
                if rows % 128 or cols % 128:
                    return None
                count = e * (rows // 128) * (cols // 128)
            if scale.dtype != torch.float32 or scale.numel() != count:
                return None
    return kind


def _configs(hidden_states, w1, w2, topk_weight, topk_ids, *, options,
             batch_bucket, model_key):
    """根据当前输入生成候选实现和 tile 配置，再由 prune 检查精度。"""
    call = dict(options, hidden_states=hidden_states, w1=w1, w2=w2,
                topk_weight=topk_weight, topk_ids=topk_ids)
    kind = _native_kind(call)
    configs = [Config(_impl="aiter")]
    if kind is None:
        return configs
    if options["activation"] == aiter.ActivationType.Gelu:
        h = hidden_states.shape[1]
        i, topk = w1.shape[1], topk_ids.shape[1]
        # 固定 M/N=256，LDS 130 KiB；按整个 bucket 检查 32-bit offset，含 padding/prefetch。
        if (torch.cuda.get_device_properties(hidden_states.device).gcnArchName.startswith("gfx950")
                and h % 256 == 0 and i % 256 == 0
                and options["block_size_M"] in (None, 0, -1, 256)
                and max((batch_bucket + 2) * topk, 256) * max(h, i) * 2 < 2**32):
            configs.append(Config(_impl="jit_gelu", tile_m_gate=256, tile_m_down=256,
                                  tile_n_gate=256, tile_n_down=256))
        return configs  # 其它 ASM/FlyDSL 候选实现的是 gated 激活，不能混用。
    h, i = hidden_states.shape[1], w1.shape[1] // 2
    separated = options["gate_mode"] == GateMode.SEPARATED
    silu = options["activation"] == aiter.ActivationType.Silu
    block_m = options["block_size_M"]

    def add(impl, gm=16, dm=16, gn=64, dn=64, **kwargs):
        """保留 caller 的 M 设置；FlyDSL 的 tile/资源限制交给编译检查。"""
        if block_m not in (None, 0, -1, dm):
            return
        # ASM wrapper 用整除计算 launch grid，不能丢掉最后一个 N tile。
        if impl.startswith("jit_") and ((2 * i) % gn or h % dn):
            return
        params = dict(_impl=impl, tile_m_gate=gm, tile_m_down=dm,
                      tile_n_gate=gn, tile_n_down=dn, **kwargs)
        if impl == "fly_prefill":
            for k in ((64, 128) if kind == "bf16" else (128, 256)):
                configs.append(Config(**params, tile_k_gate=k))
        else:
            configs.append(Config(**params))

    if kind in ("bf16", "block") and silu and separated:
        arch = torch.cuda.get_device_properties(hidden_states.device).gcnArchName.split(":")[0]
        # 这些 8-wave tile 的 LDS 用量要求 gfx950，BF16 不做激活量化。
        # 普通路径分别支持 raw/shuffled 权重，不替调用方转换布局。
        if arch == "gfx950":
            impl = "jit_8wave" if kind == "bf16" else "jit_blockscale"
            for m in (128, 256):
                for n in (128, 256):
                    add(impl, m, m, 256, n)
            # 沿用已有 persistent Down 的 M256 / 小 K 范围，counter 每次调用清零。
            if i <= 256 and getattr(w2, "is_shuffled", False):
                for splits in (1, 2, 4):
                    if h % (splits * 128) == 0 and h // splits >= 256:
                        add(impl, 256, 256, 256, 64,
                            down_path="persistent", num_oc_splits=splits)

    # 其余本地 kernel 要求两份权重都已 shuffle。
    if not all(getattr(w, "is_shuffled", False) for w in (w1, w2)):
        return configs
    if silu and separated:
        if kind in ("bf16", "ptpc"):
            if batch_bucket == 1:
                add("jit_batch1", gn=32, dn=32)
            add("jit_batch", gn=32)
            if kind == "ptpc" and h % 1024 == 0:
                add("jit_batch", gn=32, block_n=1024)
        if kind in ("bf16", "ptpc", "block", "fp4"):
            if (kind != "block" or h % 512 == 0) and (kind != "fp4" or h % 1024 == 0):
                for m in (16, 32, 64):
                    for n in (64, 128):
                        add("jit_splitk", m, m, n, n)
        # 一阶段融合 kernel 的 Down 只支持 BF16 权重，不能传入 FP8/FP4。
        if kind == "bf16" and i <= 512:
            add("jit_1stage")
        if kind == "fp4" and h % 256 == 0 and i % 256 == 0:
            add("jit_mxfp4", 128, 128, 128, 128)
            for m in (128, 256):
                for n in (128, 256):
                    add("jit_mxfp4_4wave", m, m, n, 128)

    if kind == "block":
        return configs  # 当前 FlyDSL kernel 不支持 128×128 block scale。
    if batch_bucket <= 32:
        down_ns = (32,) if kind == "fp4" else (64,)
        if kind == "fp4" and batch_bucket == 1:
            down_ns += (64,)
        for gn in (32, 64):
            for dn in down_ns:
                add("fly_decode", gn=gn, dn=dn, decode_alg="batch1")
    for m in (16, 32, 64):
        for n in (64, 128):
            add("fly_decode", m, m, n, n)
    if kind == "fp4" or batch_bucket < 64:
        return configs
    for m in (32, 64, 128):
        for n in (128, 256):
            add("fly_prefill", m, m, n, 128)
    if kind != "bf16":
        for path, dm, dn in (("1x4_64x256", 64, 256), ("8x1", 256, 128), ("8x1_compact", 64, 128)):
            for gn in (128, 256):
                for padding in (0, 128):
                    add("fly_prefill", 64, dm, gn, dn, down_path=path, padding=padding)
    return configs


def _run_jit(x, w1, w2, ids, weights, options, out, impl, m, gn, dn, block_n):
    """按各 ASM kernel 的接口组织调用，结果统一写入调用方提供的 out。"""
    from .asm.moe import (
        moe_1stage_splitk, moe_2stage_down_loopn, moe_2stage_splitk,
        moe_gemm_batch, moe_gemm_batch1,
    )

    b, h = x.shape
    e, n1, _ = w1.shape
    i, topk = n1 // 2, ids.shape[1]
    s1, s2 = options["w1_scale"], options["w2_scale"]
    p1, p2 = (s.data_ptr() if s is not None else 0 for s in (s1, s2))
    mid = torch.empty((b, topk, i), device=x.device, dtype=x.dtype)
    if impl == "jit_batch1":
        out.zero_()
        moe_gemm_batch1([n1 // 32, topk], [256], w1.dtype, True,
                        x.data_ptr(), w1.data_ptr(), mid.data_ptr(), ids.data_ptr(),
                        weights.data_ptr(), p1, b, n1, h)
        moe_gemm_batch1([h // 32, topk], [64], w2.dtype, False,
                        mid.data_ptr(), w2.data_ptr(), out.data_ptr(), ids.data_ptr(),
                        weights.data_ptr(), p2, b, h, i)
        return out
    # 排序时会将 out 清零，确保每次 Down 的 atomic add 都从零开始。
    si, sw, se, valid, _ = moe_sorting(ids, weights, e, h, x.dtype, m, output=out)
    grid = min(se.numel(), b * topk)
    routing = [t.data_ptr() for t in (si, sw, se, valid)]
    if impl == "jit_1stage":
        moe_1stage_splitk([1, grid], [256], w1.dtype, topk, h, n1, h, m, gn,
                         x.data_ptr(), w1.data_ptr(), p1, w2.data_ptr(), p2,
                         out.data_ptr(), *routing, b)
        return out
    if impl in ("jit_mxfp4", "jit_mxfp4_4wave"):
        from .asm.moe_gemm_mxfp4 import (
            moe_gemm_final_reduce_bf16, moe_gemm_mxfp4, moe_gemm_mxfp4_gateup_4wave,
        )
        from aiter.utility.fp4_utils import moe_mxfp4_sort

        if max(x.nbytes, w1.nbytes, w2.nbytes, mid.nbytes, b * topk * h * x.element_size()) >= 2**32:
            raise ValueError("MXFP4 ASM kernels require tensors smaller than 4 GiB")
        # A4W4 还需要量化激活，并按排序后的 token 顺序重排激活 scale。
        quant = aiter.get_hip_quant(aiter.QuantType.per_1x32)
        xq, xs = quant(x, quant_dtype=w1.dtype)
        xs = moe_mxfp4_sort(xs, sorted_ids=si, num_valid_ids=valid, token_num=b, block_size=m)
        if impl == "jit_mxfp4_4wave":
            moe_gemm_mxfp4_gateup_4wave([n1 // gn * grid], [256], m, gn, e, n1, h // 2, True, topk,
                                      *routing, w1.data_ptr(), p1, xq.data_ptr(), xs.data_ptr(), mid.data_ptr(), b)
        else:
            moe_gemm_mxfp4([n1 // gn, grid], [256], m, gn, e, n1, h // 2, True, topk,
                          *routing, w1.data_ptr(), p1, xq.data_ptr(), xs.data_ptr(), mid.data_ptr(), b)
        aq, scales = quant(mid.view(b * topk, i), quant_dtype=w1.dtype)
        scales = moe_mxfp4_sort(scales[:b * topk].view(b, topk, -1), sorted_ids=si,
                               num_valid_ids=valid, token_num=b, block_size=m)
        routes = torch.empty((b, topk, h), dtype=x.dtype, device=x.device)
        moe_gemm_mxfp4([h // dn, grid], [256], m, dn, e, h, i // 2, False, topk,
                      *routing, w2.data_ptr(), p2, aq.data_ptr(), scales.data_ptr(), routes.data_ptr(), b)
        # JIT reduce 双缓冲每轮处理两个 512 元素分片；不满足时仅替换最后的 sum。
        if h % 1024 == 0:
            moe_gemm_final_reduce_bf16([512], [64], topk, h, routes.data_ptr(), out.data_ptr(),
                                      b // 512, b % 512, b)
        else:
            torch.sum(routes, dim=1, out=out)
        return out
    ptpc = w1.dtype == torch.bfloat16 or options["quant_type"] == aiter.QuantType.per_Token
    if impl == "jit_batch":
        moe_gemm_batch([n1 // 32, grid], [256], w1.dtype, True, x.data_ptr(),
                       w1.data_ptr(), mid.data_ptr(), *routing, p1, b, n1, h, topk)
    else:
        moe_2stage_splitk([n1 // gn, grid], [256], w1.dtype, topk, h, n1, True, m, gn,
                         x.data_ptr(), w1.data_ptr(), mid.data_ptr(), *routing, p1, b, ptpc)
    if block_n:
        # 小 batch 直接 atomic add；较大 batch 先保存每条 route 的结果，再对 TOPK 做 reduce。
        atomic = b < 8
        routes = out if atomic else torch.empty((b, topk, h), device=x.device, dtype=x.dtype)
        moe_2stage_down_loopn([h // block_n, grid], [256], w2.dtype, topk, i, h, m, 16,
                             mid.data_ptr(), w2.data_ptr(), routes.data_ptr(), *routing,
                             p2, b, True, block_n, atomic, 3)
        if not atomic:
            torch.sum(routes, dim=1, out=out)
    else:
        moe_2stage_splitk([h // dn, grid], [64], w2.dtype, topk, i, h, False, m, dn,
                         mid.data_ptr(), w2.data_ptr(), out.data_ptr(), *routing, p2, b, ptpc)
    return out


def _run_gelu(x, w1, w2, ids, weights, out):
    """复用非 gated GELU 8-wave kernel，直接写入 caller output，不走 Torch fallback。"""
    from .asm.moe_gemm_8wave_gelu import moe_gemm_8wave_gelu

    b, h = x.shape
    e, i, _ = w1.shape
    topk = ids.shape[1]
    si, sw, se, valid, _ = moe_sorting(ids, weights, e, h, x.dtype, 256, output=out)
    routing = [t.data_ptr() for t in (si, sw, se, valid)]
    grid = min(se.numel(), b * topk)
    mid = torch.empty((b, topk, i), dtype=x.dtype, device=x.device)
    routes = torch.empty((b, topk, h), dtype=x.dtype, device=x.device)
    for source, weight, target, up in ((x, w1, mid, True), (mid, w2, routes, False)):
        n, k = weight.shape[1:]
        tasks = n // 256 * grid
        moe_gemm_8wave_gelu(
            [tasks], [512], not up, False, False, "bf16", 256, 256, n, k,
            up, bool(getattr(weight, "is_shuffled", False)), topk, *routing,
            weight.data_ptr(), 0, source.data_ptr(), 0, target.data_ptr(), 0, b, tasks,
        )
    torch.sum(routes, dim=1, out=out)
    return out


def _run_8wave(x, w1, w2, ids, weights, options, out, m, gn, dn,
               down_path, num_oc_splits):
    """BF16 或 W128×128 / A1×128 FP8 SiLU，共用 8-wave 两阶段流程。"""
    from .asm.moe_gemm_8wave import moe_gemm_8wave_down, moe_gemm_8wave_g1u1

    def ptr(t):
        return t.data_ptr() if t is not None else 0

    b, h = x.shape
    e, n1, _ = w1.shape
    i, topk = n1 // 2, ids.shape[1]
    si, sw, se, valid, _ = moe_sorting(ids, weights, e, h, x.dtype, m, output=out)
    routing = [t.data_ptr() for t in (si, sw, se, valid)]
    grid = min(se.numel(), b * topk)
    bf16 = w1.dtype == torch.bfloat16
    ab_dtype = "bf16" if bf16 else "fp8"
    quantize = aiter.get_hip_quant(aiter.QuantType.No if bf16 else aiter.QuantType.per_1x128)
    # kernel 按 [K/128, M] 读取激活 scale；不要再将它转为 row-major。
    aq, a_scale = quantize(x, quant_dtype=w1.dtype, transpose_scale=True)
    mid = torch.empty((b, topk, i), dtype=x.dtype, device=x.device)
    tasks = n1 // gn * grid
    moe_gemm_8wave_g1u1(
        [tasks], [512], aq.nbytes > (1 << 32), ab_dtype, m, gn, e, n1, h,
        True, bool(getattr(w1, "is_shuffled", False)), topk, *routing,
        w1.data_ptr(), ptr(options["w1_scale"]), aq.data_ptr(), ptr(a_scale),
        mid.data_ptr(), b, tasks,
    )
    dq, d_scale = quantize(mid, quant_dtype=w2.dtype, num_rows_factor=topk, transpose_scale=True)
    routes = torch.empty((b, topk, h), dtype=x.dtype, device=x.device)
    if down_path == "persistent":
        # 私有 counter 不跨调用/stream 共享，graph replay 也会执行这次清零。
        counter = torch.zeros(1, dtype=torch.int32, device=x.device)
        workers = torch.cuda.get_device_properties(x.device).multi_processor_count
        moe_gemm_8wave_down(
            [workers], [512], routes.nbytes > (1 << 32), ab_dtype, m, dn,
            e, h, i, num_oc_splits, False, True, topk, *routing,
            w2.data_ptr(), ptr(options["w2_scale"]), dq.data_ptr(), ptr(d_scale),
            routes.data_ptr(), b, counter.data_ptr(),
        )
    else:
        tasks = h // dn * grid
        moe_gemm_8wave_g1u1(
            [tasks], [512], dq.nbytes > (1 << 32), ab_dtype, m, dn, e, h, i,
            False, bool(getattr(w2, "is_shuffled", False)), topk, *routing,
            w2.data_ptr(), ptr(options["w2_scale"]), dq.data_ptr(), ptr(d_scale),
            routes.data_ptr(), b, tasks,
        )
    torch.sum(routes, dim=1, out=out)
    return out


def _run_fly(x, w1, w2, ids, weights, options, out, impl, gm, dm, gn, dn,
             decode_alg, down_path, padding, tile_k_gate=0):
    """执行 FlyDSL 的两阶段 MoE，每次调用都重新获取当前 stream。"""
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled
    from .flydsl.moe_gemm_splitk import compile_gemm, invert_sorted_ids, sorted_sum

    def ptr(t):
        """将 tensor 转为 FlyDSL 指针；FP8/FP4/E8M0 用字节指针传入。"""
        elem = {torch.bfloat16: fx.BFloat16, torch.float32: fx.Float32,
                torch.int32: fx.Int32}.get(t.dtype, fx.Uint8)
        return flyc.from_c_void_p(elem, t.data_ptr())

    def launch(params, tensors, *args):
        # 首次 compile 会同时执行 kernel，不能再调用一次，否则 atomic add 会重复累加。
        _run_compiled(compile_gemm(**params), *(ptr(t) for t in tensors), *args, stream)

    def quantize(t):
        """为 native FP8 prefill 生成量化激活和 per-token/per-tensor scale。"""
        qtype = (aiter.QuantType.per_Token if quant == "ptpc" else aiter.QuantType.per_Tensor)
        q, scale = aiter.get_hip_quant(qtype)(t.view(-1, t.shape[-1]), quant_dtype=w1.dtype)
        return q, scale.to(torch.float32).contiguous()

    b, h = x.shape
    e, n1, _ = w1.shape
    i, topk = n1 // 2, ids.shape[1]
    stream = torch.cuda.current_stream(x.device)
    kind = "bf16" if w1.dtype == torch.bfloat16 else "fp4" if w1.dtype == torch.float4_e2m1fn_x2 else "fp8"
    quant = {aiter.QuantType.No: "no", aiter.QuantType.per_Token: "ptpc",
             aiter.QuantType.per_Tensor: "per_tensor", aiter.QuantType.per_1x32: "mxfp4"}[options["quant_type"]]
    dummy = torch.empty(1, device=x.device, dtype=torch.float32)
    s1 = options["w1_scale"] if options["w1_scale"] is not None else dummy
    s2 = options["w2_scale"] if options["w2_scale"] is not None else dummy
    mid = torch.empty((b, topk, i), device=x.device, dtype=x.dtype)
    common = dict(weight_dtype=kind, weight_quant_type=quant, TOPK=topk, E=e,
                  act_quant_type="no" if kind == "fp4" else quant)
    activation = {aiter.ActivationType.Silu: "silu", aiter.ActivationType.Swiglu: "swiglu",
                  aiter.ActivationType.Situv2: "situv2"}[options["activation"]]
    g = dict(common, N=n1, K=h, stage="gateup", BLOCK_TILE_SIZE_M=gm,
             BLOCK_TILE_SIZE_N=gn, activation=activation,
             # Aiter 的 SiTUv2 不做 clamp；这里传入 inf，让现有 kernel 保持相同行为。
             swiglu_limit=float("inf") if activation == "situv2" else options["swiglu_limit"],
             situ_beta=1.0 if options["beta"] is None else float(options["beta"]),
             situ_linear_beta=1.0 if options["linear_beta"] is None else float(options["linear_beta"]),
             mxfp4_gate_up_interleaved=options["gate_mode"] == GateMode.INTERLEAVE)
    d = dict(common, N=h, K=i, stage="down", BLOCK_TILE_SIZE_M=dm, BLOCK_TILE_SIZE_N=dn)
    if impl == "fly_decode" and decode_alg == "batch1":
        # Gate/Up 清零整个输出，后续同 stream 的 Down 再 atomic add。
        launch(dict(g, alg="batch1", fused_down_clear=True),
               (x, w1, mid, ids, out, s1), b)
        launch(dict(d, alg="batch1"), (mid, w2, out, ids, weights, s2), b)
        return out

    si, sw, se, valid, _ = moe_sorting(ids, weights, e, h, x.dtype, dm, output=out)
    routing = (si, sw, se, valid)
    grid = se.numel()
    if impl == "fly_decode":
        launch(dict(g, alg="splitk", METADATA_TILE_SIZE_M=dm), (x, w1, mid, *routing, s1), b, grid)
        launch(dict(d, alg="splitk"), (mid, w2, out, *routing, s2), b, grid)
        return out

    gate_in, a_scale = quantize(x) if kind == "fp8" else (x, dummy)
    launch(dict(g, alg="prefill_1x4", METADATA_TILE_SIZE_M=dm,
                 tile_k=tile_k_gate or (128 if kind == "fp8" else 64)),
           (gate_in, w1, mid, *routing, s1, a_scale), b, grid)
    down_in, a_scale = quantize(mid) if kind == "fp8" else (mid, dummy)
    routes = torch.empty((grid * dm, h + (padding or 0) // 2), device=x.device, dtype=x.dtype)
    extra_tensors, extra_args = (), ()
    if down_path == "8x1_compact":
        from .flydsl.moe_gemm_2stage.gemm2_8x1_compact import allocate_task_buffers
        # 这里只分配 workspace；full/tail 任务表会在每次 launch 时由 GPU 重新生成。
        full, tail, counts = allocate_task_buffers(se, e)
        extra_tensors, extra_args = (full, tail, counts), (full.shape[0], tail.shape[0])
    params = dict(d, alg="prefill_1x4", USE_ATOMIC_WRITE=False,
                  down_path=down_path, down_output_padding_bytes=padding)
    # compact 接口中，workspace 指针必须跟在 b 和 grid 两个运行时参数之后。
    _run_compiled(compile_gemm(**params), *(ptr(t) for t in (down_in, w2, routes, *routing, s2, a_scale)),
                  b, grid, *(ptr(t) for t in extra_tensors), *extra_args, stream)
    loc_ids = torch.empty((b, topk), device=x.device, dtype=torch.int32)
    # 只为排序 buffer 中的有效条目建立反向索引，避免读取未初始化的尾部。
    invert_sorted_ids(topk)(si, loc_ids, valid, si.numel(), b)
    sorted_sum(topk, h, padding)(loc_ids, routes, out, b)
    return out


def _fmoe_wrapper(hidden_states, w1, w2, topk_weight, topk_ids, *, options,
                  batch_bucket, model_key, _impl="aiter", tile_m_gate=16,
                  tile_m_down=16, tile_n_gate=64, tile_n_down=64, block_n=0,
                  decode_alg="splitk", down_path="default", padding=None, num_oc_splits=1,
                  tile_k_gate=0):
    """供 autotune 调用的 Python 入口；Config 提供实现名称和 tile 参数。"""
    global last_dispatch
    if record_dispatch:
        last_dispatch = dict(_impl=_impl)
        if _impl != "aiter":
            last_dispatch.update(tile_m_gate=tile_m_gate, tile_m_down=tile_m_down,
                                 tile_n_gate=tile_n_gate, tile_n_down=tile_n_down, block_n=block_n,
                                 decode_alg=decode_alg, down_path=down_path, padding=padding,
                                 num_oc_splits=num_oc_splits)
            if _impl == "fly_prefill":
                last_dispatch["tile_k_gate"] = tile_k_gate
    if _impl == "aiter":
        return _aiter_fused_moe(hidden_states, w1, w2, topk_weight, topk_ids, **options)
    out = options["output"]
    if _impl == "jit_gelu":
        return _run_gelu(hidden_states, w1, w2, topk_ids, topk_weight, out)
    if _impl in ("jit_batch1", "jit_batch", "jit_splitk", "jit_1stage", "jit_mxfp4", "jit_mxfp4_4wave"):
        return _run_jit(hidden_states, w1, w2, topk_ids, topk_weight, options, out,
                        _impl, tile_m_down, tile_n_gate, tile_n_down, block_n)
    if _impl in ("jit_8wave", "jit_blockscale"):
        return _run_8wave(hidden_states, w1, w2, topk_ids, topk_weight, options, out,
                          tile_m_down, tile_n_gate, tile_n_down, down_path, num_oc_splits)
    if _impl in ("fly_decode", "fly_prefill"):
        return _run_fly(hidden_states, w1, w2, topk_ids, topk_weight, options, out,
                        _impl, tile_m_gate, tile_m_down, tile_n_gate, tile_n_down,
                        decode_alg, down_path, padding, tile_k_gate)
    raise ValueError(f"unknown MoE implementation: {_impl}")


def _prune_invalid_configs(configs, sig_args):
    """计时前完成候选的编译和精度检查，失败的配置不参与性能比较。"""
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("MoE autotuning must run before graph capture; unset FLYDSL_AUTOTUNE afterwards")
    call = dict(sig_args["options"], **{name: sig_args[name] for name in
                ("hidden_states", "w1", "w2", "topk_weight", "topk_ids")})
    reference = _torch_reference(call)
    if not torch.isfinite(reference).all().item():
        raise RuntimeError("Torch reference contains non-finite values; refusing to select a config")
    out = sig_args["options"]["output"]
    valid = []
    for config in configs:
        out.fill_(float("nan"))
        try:
            result = _fmoe_wrapper(**sig_args, **config.all_kwargs())
        except Exception as error:
            # 同步时若发现非法访存等致命 GPU 错误，直接抛出，停止后续调优。
            torch.cuda.synchronize(out.device)
            print(f"[moe validate] {config} rejected: {type(error).__name__}: {error}")
            continue
        torch.cuda.synchronize(out.device)
        if result is not out or not torch.isfinite(out).all().item():
            print(f"[moe validate] {config} rejected: output identity or finite check failed")
            continue
        diff = calc_diff(reference, out)
        if not math.isfinite(diff) or diff > 0.02:
            print(f"[moe validate] {config} rejected: calc_diff={diff:.6g}")
            continue
        valid.append(config)
    if not valid:
        raise RuntimeError(
            "No MoE config passed validation; check each weight's is_shuffled, "
            "gate_mode and scale layout (see rejected candidates above)"
        )
    return valid


# 不设 default：首次未命中缓存时必须先调优，不能直接使用未经校验的配置。
_autotuned_fmoe = autotune(
    configs=_configs,
    key=["batch_bucket", "model_key"],
    prune_configs_by=_prune_invalid_configs,
    artifact_name="pyhip_fused_moe_v6",
)(_fmoe_wrapper)


def _model_key(call):
    """把影响配置选择的信息加入 cache key；缓存只保存配置，不保存张量数据。"""
    # FlyDSL 不会读取 tensor 的自定义属性或展开 options 中的张量，需要手动补充。
    values = {"version": 6}
    for name, value in call.items():
        if isinstance(value, torch.Tensor):
            shape = tuple(value.shape)
            if name in ("hidden_states", "topk_ids", "topk_weight", "output"):
                shape = shape[1:]
            value = (shape, str(value.dtype), tuple(value.stride()), bool(getattr(value, "is_shuffled", False)))
        elif hasattr(value, "value"):
            value = value.value
        elif isinstance(value, torch.dtype):
            value = str(value)
        values[name] = value
    props = torch.cuda.get_device_properties(call["hidden_states"].device)
    values["device"] = (props.name, props.gcnArchName, props.multi_processor_count)
    values["environment"] = sorted((name, value) for name, value in os.environ.items()
                                    if name.startswith(("AITER_", "MOE_", "PYHIP_")))
    if call["w1"].dtype == torch.float4_e2m1fn_x2:
        # Aiter 根据实际 M 切换 FP4 激活精度，这些分界点不一定落在 bucket 边界上。
        m = call["hidden_states"].shape[0]
        values["fp4_precision_tier"] = (
            m < int(os.getenv("AITER_BF16_FP8_MOE_BOUND", "256")),
            m < int(os.getenv("GPTOSS_SWIGLU_MXFP4_BF16_BOUND", "256")),
        )
    return json.dumps(values, sort_keys=True, separators=(",", ":"))


def _make_tune_args(call):
    """将已补齐默认值和 output 的 API 参数打包给 autotuner，不修改原字典。"""
    tensor_names = ("hidden_states", "w1", "w2", "topk_weight", "topk_ids")
    return {name: call[name] for name in tensor_names} | dict(
        options={name: value for name, value in call.items() if name not in tensor_names},
        batch_bucket=ceil_pow2(call["hidden_states"].shape[0]), model_key=_model_key(call))


def fused_moe(
    hidden_states,
    w1,
    w2,
    topk_weight,
    topk_ids,
    expert_mask: torch.Tensor | None = None,
    activation=aiter.ActivationType.Silu,
    quant_type=aiter.QuantType.No,
    doweight_stage1=False,
    w1_scale: torch.Tensor | None = None,
    w2_scale: torch.Tensor | None = None,
    a1_scale: torch.Tensor | None = None,
    a2_scale: torch.Tensor | None = None,
    block_size_M=None,
    num_local_tokens: torch.Tensor | None = None,
    moe_sorting_dispatch_policy=0,
    dtype=None,
    hidden_pad=0,
    intermediate_pad=0,
    bias1=None,
    bias2=None,
    splitk=0,
    swiglu_limit=None,
    beta=None,
    linear_beta=None,
    gate_mode: str | None = GateMode.SEPARATED.value,
    shared_w1: torch.Tensor | None = None,
    shared_w2: torch.Tensor | None = None,
    shared_w1_scale: torch.Tensor | None = None,
    shared_w2_scale: torch.Tensor | None = None,
    shared_expert_id: int = -1,
    stage2_scatter: Stage2ScatterContext | None = None,
    output: torch.Tensor | None = None,
):
    """兼容 Aiter 的 MoE 推理接口；传入 output 时，会写入并返回该 tensor。

    对 BF16 输入、无 bias/EP 的组合调优：SiLU/SwiGLU/SiTUv2 使用 gated
    结构，GELU 使用非 gated BF16 权重。gfx950 的 BF16 / FP8 block-scale
    SiLU 8-wave 和 BF16 GELU 路径支持两份权重各自的 raw/shuffled 布局；其它组合的
    原始布局或混合布局只能尝试 Aiter。候选都需通过 Torch 参考检查。
    其他 API 功能直接转交 Aiter，不在这里额外校验。若 is_shuffled 属性丢失
    （例如重新包装成 Parameter），就按原始布局处理，不猜测实际存储方式。
    缓存命中后仍使用当前输入和 stream；graph capture 前应完成调优和预热。
    """
    global last_dispatch
    call = locals().copy()
    if _native_kind(call) is None:
        if record_dispatch:
            last_dispatch = dict(_impl="aiter")
        return _aiter_fused_moe(**call)
    with torch.cuda.device(hidden_states.device), torch.no_grad():
        if output is None:
            output = torch.empty_like(hidden_states)
        elif (output.shape != hidden_states.shape or output.dtype != hidden_states.dtype
              or output.device != hidden_states.device or not output.is_contiguous()):
            raise RuntimeError("output must match hidden_states' shape/dtype/device and be contiguous")
        for name, value in call.items():
            if name != "output" and isinstance(value, torch.Tensor):
                # out 会先被清零，所以不能与输入共享同一段内存。
                if (output.data_ptr() < value.data_ptr() + value.nbytes
                        and value.data_ptr() < output.data_ptr() + output.nbytes):
                    raise RuntimeError("output must not overlap MoE inputs")
        call["output"] = output
        return _autotuned_fmoe(**_make_tune_args(call))