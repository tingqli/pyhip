# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""MFMA operands: fp8 fragments in registers, one MFMA(16,16,32, fp8 FNUZ) step."""

import flydsl.expr as fx


def _fp8_frag8(v8):
    t = fx.make_rmem_tensor(fx.make_layout(8, 1), fx.Float8E4M3FNUZ)
    t.store(v8)
    return t


def _mma_fp8(mma_atom, acc, a8, b8):
    """acc += A(16x32) * B(32x16), one MFMA(16,16,32,fp8 FNUZ)."""
    fx.gemm(mma_atom, acc, _fp8_frag8(a8), _fp8_frag8(b8), acc)
