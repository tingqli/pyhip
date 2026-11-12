# Package copy of doc-mimo/kernels/a8w4_row_quant.py (test harness removed, env knobs frozen).
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Per-row FP8 (FNUZ) activation quant used by the A8W4 MoE kernels in this directory.

Copy of ``aiter.ops.flydsl.kernels.moe_2stage_a16wmix.a8w4.quantize_rows`` with the
row count moved from a compile-time constant to a launch argument. The original
``_compile_row_quant(cols, rows, dynamic_rows)`` bakes ``rows`` into the kernel (row
bound) and into the launch grid, so every new token count JIT-compiles another
kernel. Here the binary depends only on ``cols`` (and on whether the row bound is
read from the device); ``rows`` and the grid are passed per call. Same math, same
output bits.
"""

import functools
from functools import cache
import os

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir.dialects import arith, llvm
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from .._vendor import buffer_ops
from .._vendor import buffer_ops as bo
from .._vendor.act import gate_up_act, silu_mul_batch, situ_params
from .._vendor.a8w4_ops import _pack_activation, _pointer, channel_exponents
from .._vendor.moe_utils import _buffer_i32_scalar_read, _global_i32_at, _udiv, _umod
from .._vendor.mxfp4_gemm_common import (
    _gep1,
    _global_base_ptr1,
    _global_i32_buffer_tiles,
    _global_i32_buffer_view,
    _lds_ptr3,
    global_typed_ptr,
    lds_typed_ptr,
    lds_vec_load,
)
from .._vendor.tensor_shim import _run_compiled
from .._vendor.tensor_shim import _to_raw as _raw


@cache
def _compile_row_quant(cols, dynamic_rows):
    if cols % 256:
        raise ValueError("activation K must be a multiple of 256")
    chunks = cols // 256

    @flyc.kernel(known_block_size=(256, 1, 1))
    def kernel(source: fx.Int64, output: fx.Int64, scale: fx.Int64, valid: fx.Int64,
               i32_rows: fx.Int32):
        tid = fx.Int32(gpu.thread_id("x"))
        row = fx.Int32(gpu.block_id("x")) * 4 + tid // 64  # one wave per row
        lane = tid % 64
        count = i32_rows
        if const_expr(dynamic_rows):
            count = fx.ptr_load(_pointer(valid, fx.Int32, 4))
        if row < count:
            values = []
            amax = fx.Float32(0.0)
            for j in range_constexpr(chunks):
                offset = fx.Int64(row) * cols + lane * 4 + j * 256
                v = fx.ptr_load(fx.add_offset(_pointer(source, fx.BFloat16, 8), offset),
                                result_type=fx.Vector.make_type(4, fx.BFloat16)).to(fx.Float32)
                values.append(v)
                for k in range_constexpr(4):
                    amax = fx.max(amax, fx.max(v[k], -v[k]))
            for bit in range_constexpr(6):
                amax = fx.max(amax, amax.shuffle_xor(1 << bit, 64))
            s = (amax == 0.0).select(fx.Float32(1.0), amax / 240.0)
            inv = 1.0 / s
            if lane == 0:
                fx.ptr_store(s, fx.add_offset(_pointer(scale, fx.Float32, 4), row))
            for j in range_constexpr(chunks):
                v = values[j] * inv
                packed = _pack_activation(v[0], v[1], v[2], v[3])
                offset = fx.Int64(row) * (cols // 4) + lane + j * 64
                fx.ptr_store(packed, fx.add_offset(_pointer(output, fx.Int32, 4), offset))

    @flyc.jit
    def launch(source: fx.Int64, output: fx.Int64, scale: fx.Int64, valid: fx.Int64,
               i32_rows: fx.Int32, i32_grid: fx.Int32, stream: fx.Stream):
        kernel(source, output, scale, valid, i32_rows).launch(
            grid=(fx.Int64(i32_grid), 1, 1), block=(256, 1, 1), stream=stream)

    return launch


def quantize_rows(x, valid=None):
    """bf16 ``x`` [rows, cols] -> (fp8 FNUZ [rows, cols], f32 [rows]): one scale per
    row (amax / 240). ``valid``: int32 device scalar (e.g. ``num_valid_ids``); then
    only rows < valid[0] are written."""
    rows, cols = x.shape
    q = torch.empty_like(x, dtype=torch.float8_e4m3fnuz)
    scales = torch.empty(rows, device=x.device, dtype=torch.float32)
    _run_compiled(_compile_row_quant(cols, valid is not None), x.data_ptr(), q.data_ptr(),
                  scales.data_ptr(), valid.data_ptr() if valid is not None else 0,
                  int(rows), (rows + 3) // 4, torch.cuda.current_stream())
    return q, scales
