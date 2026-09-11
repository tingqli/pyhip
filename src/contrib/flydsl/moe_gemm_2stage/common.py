# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch


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

from ..helpers import (
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
