# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
import os
import flydsl.compiler as flyc
import flydsl.expr as fx
import torch

from flydsl._mlir.dialects import llvm
from flydsl.expr import range_constexpr
from flydsl.expr.typing import T, as_ir_value
from flydsl.expr.typing import Vector as Vec

# 默认使用保留 NaN 的 RTE；RTA 必须显式开启，简化位运算不保留所有 NaN。
_SIMPLIFIED_BF16_RTA = os.environ.get(
    "AITER_FLYDSL_MOE_BF16_RTA_SIMPLIFIED", "1"
).lower() in ("1", "true")
_SIMPLIFIED_BF16_RTE = os.environ.get(
    "AITER_FLYDSL_MOE_BF16_RTE_SIMPLIFIED", "0"
).lower() in ("1", "true")

def _f32_to_bf16_rta(value):
    # 有限值及 Inf 与标准 RTA 一致；正负 halfway 均向远离零方向舍入。
    return (
        ((value.bitcast(fx.Uint32) + fx.Uint32(0x8000)) >> 16)
        .to(fx.Uint16)
        .bitcast(fx.BFloat16)
    )


def _f32_to_bf16_rte(value):
    if _SIMPLIFIED_BF16_RTE:
        # 与 main 一致的可选简化 RTE，不保证 NaN 保留。
        bits = value.bitcast(fx.Uint32)
        rounded = bits + fx.Uint32(0x7FFF) + ((bits >> 16) & fx.Uint32(1))
        return (rounded >> 16).to(fx.Uint16).bitcast(fx.BFloat16)
    # ck_tile/float_to_bf16_rtn_asm：默认将 NaN 规范化为 0x7FFF。
    rounded = llvm.inline_asm(
        T.i32,
        [
            as_ir_value(value),
            as_ir_value(fx.Uint32(0x7FFF)),
            as_ir_value(fx.Uint32(0x7FFF0000)),
        ],
        "v_cmp_u_f32 vcc, $1, $1\n\t"
        "v_bfe_u32 $0, $1, 16, 1\n\t"
        "v_add3_u32 $0, $1, $0, $2\n\t"
        "v_cndmask_b32 $0, $0, $3, vcc",
        "=&v,v,v,v,~{vcc}",
        has_side_effects=False,
    )
    return (fx.Uint32(rounded) >> 16).to(fx.Uint16).bitcast(fx.BFloat16)


def _f32_to_bf16(value):
    if _SIMPLIFIED_BF16_RTA:
        return _f32_to_bf16_rta(value)
    if isinstance(value, Vec):
        return Vec.from_elements(
            [_f32_to_bf16_rte(value[i]) for i in range_constexpr(value.numel)],
            fx.BFloat16,
        )
    return _f32_to_bf16_rte(value)

# ==================== 设备/Host基础 ====================

_TORCH_TO_FX = {
    torch.bfloat16: fx.BFloat16,
    torch.float32: fx.Float32,
    torch.float64: fx.Float64,
    torch.int32: fx.Int32,
    torch.float8_e4m3fnuz: fx.Uint8,
    torch.float8_e4m3fn: fx.Uint8,
}


def down_device_config_from_name(device_name):
    is_mi308 = "MI308" in device_name.upper()
    return is_mi308, 4 if is_mi308 else 8


def get_down_device_config():
    if not torch.cuda.is_available():
        return False, 8
    return down_device_config_from_name(
        torch.cuda.get_device_name(torch.cuda.current_device())
    )


def get_device_cache_key():
    if not torch.cuda.is_available():
        return None
    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    return (
        device,
        properties.name,
        properties.gcnArchName,
        properties.multi_processor_count,
    )


def torch_tensor_to_pointer(tensor):
    return flyc.from_c_void_p(_TORCH_TO_FX[tensor.dtype], tensor.data_ptr())


# ==================== 布局/Tensor reexport ====================

from pyhip.codegen.flydsl.helpers import (
    BufferTensor,
    FlyObjCache,
    LdsTensor,
    _as_ptr,
    all_copy_atoms,
    all_elements,
    asm_mark,
    atom_tensor,
    atomic_add_bf16,
    div_up,
    eltwise_op,
    make_1d_coord_tensor,
    split_works,
    torch_layout,
    view_as_torch_tensor,
)

__all__ = [
    "BufferTensor",
    "FlyObjCache",
    "LdsTensor",
    "_as_ptr",
    "all_copy_atoms",
    "all_elements",
    "asm_mark",
    "atom_tensor",
    "atomic_add_bf16",
    "div_up",
    "eltwise_op",
    "make_1d_coord_tensor",
    "split_works",
    "torch_layout",
    "view_as_torch_tensor",
]
