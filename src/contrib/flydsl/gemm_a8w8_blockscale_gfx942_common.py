# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Helpers shared by the FlyDSL ports of the PyHIP gfx942 FP8 block-scale GEMM kernels."""

import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec


def tb_swizzle(block_id, M, N, tile_m, tile_n, m01=8, group_num=8):
    """WG id -> (block_m, block_n), identical to pyhip.contrib.common.loaders.tb_swizzle."""
    m0 = (M + tile_m - 1) // tile_m
    n0 = (N + tile_n - 1) // tile_n
    group_size = (m0 * n0 + group_num - 1) // group_num
    big_group_num = group_num - (group_size * group_num - m0 * n0)
    group_id_y = block_id // group_num
    group_id_x = block_id - group_id_y * group_num
    remap = group_id_x * group_size + group_id_y
    remap = (group_id_x > big_group_num).select(remap + (big_group_num - group_id_x), remap)
    idx_m0 = remap // n0
    idx_n0 = remap - idx_m0 * n0
    m0_mod_m01 = m0 - m0 // m01 * m01
    m01_adapt = (idx_m0 < m0 - m0_mod_m01).select(fx.Int32(m01), m0_mod_m01)
    idx_m00 = idx_m0 // m01
    local = idx_n0 + (idx_m0 - idx_m00 * m01) * n0
    block_n = local // m01_adapt
    return local - block_n * m01_adapt + idx_m00 * m01, block_n


def s_buffer_load_f32(rsrc, offset_bytes):
    """Scalar s_buffer_load_dword hidden from LLVM; callers must wait lgkmcnt(0) before use."""
    # The tracked SMEM intrinsic lacks a memory operand in this LLVM and gets conservative waits.
    value = llvm.inline_asm(
        T.f32,
        [rsrc, fx.Int32(offset_bytes).ir_value()],
        "s_buffer_load_dword $0, $1, $2",
        "=s,s,s",
        has_side_effects=True,
    )
    return fx.Float32(value)


def pack_bf16x2(values):
    """Four FP32 -> two BF16x2 dwords, bitwise identical to PyHIP uni_cvt_pk_bf16_f32 on gfx942."""
    # PyHIP adds the 0x8000 bit pattern as an FP32 denormal, then v_perm keeps the high halves.
    bias = fx.Int32(0x8000).bitcast(fx.Float32)
    bits = [(fx.Float32(values[index]) + bias).bitcast(fx.Int32) for index in range(4)]
    return [bits[pair * 2] >> 16 & 0xFFFF | bits[pair * 2 + 1] & -65536 for pair in range(2)]
