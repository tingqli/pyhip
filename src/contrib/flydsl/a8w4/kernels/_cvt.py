# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
# _pack_activation is from aiter 7cd70aca8 aiter/ops/flydsl/kernels/moe_2stage_a16wmix/a8w4.py.
"""Number format conversions on registers."""

import flydsl.expr as fx
from flydsl.expr import rocdl
from flydsl.expr.typing import T


def _pack_activation(a, b, c, d):
    """Four f32 -> four fp8 (FNUZ) in one i32: saturate to +-240, and map a rounded-to-zero
    magnitude to +0 (FNUZ has no -0: 0x80 is NaN)."""
    values = [fx.max(fx.min(v, 240.0), -240.0) for v in (a, b, c, d)]
    lo = rocdl.cvt_pk_fp8_f32(T.i32, values[0], values[1], fx.Int32(0), False)
    packed = fx.Uint32(rocdl.cvt_pk_fp8_f32(T.i32, values[2], values[3], lo, True))
    mag = packed & fx.Uint32(0x7F7F7F7F)
    zero = (~(mag + fx.Uint32(0x7F7F7F7F))) & fx.Uint32(0x80808080)
    return (packed & ~zero).bitcast(fx.Int32)
