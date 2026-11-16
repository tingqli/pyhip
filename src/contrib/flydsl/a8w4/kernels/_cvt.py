# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
# _pack_activation is from aiter 7cd70aca8 aiter/ops/flydsl/kernels/moe_2stage_a16wmix/a8w4.py.
"""Number format conversions on registers.

f32 -> fp8 (FNUZ) / bf16 packing for the epilogues, and the fd2 decode of the mxfp4
weights into fp8 MFMA operands with its per-channel scale factor.
"""

import flydsl.expr as fx
from flydsl.expr import rocdl
from flydsl.expr.typing import T

from ._mem import _raw


def _pack_activation(a, b, c, d):
    """Four f32 -> four fp8 (FNUZ) in one i32: saturate to +-240, and map a rounded-to-zero
    magnitude to +0 (FNUZ has no -0: 0x80 is NaN)."""
    values = [fx.max(fx.min(v, 240.0), -240.0) for v in (a, b, c, d)]
    lo = rocdl.cvt_pk_fp8_f32(T.i32, values[0], values[1], fx.Int32(0), False)
    packed = fx.Uint32(rocdl.cvt_pk_fp8_f32(T.i32, values[2], values[3], lo, True))
    mag = packed & fx.Uint32(0x7F7F7F7F)
    zero = (~(mag + fx.Uint32(0x7F7F7F7F))) & fx.Uint32(0x80808080)
    return (packed & ~zero).bitcast(fx.Int32)


def _fp8x4_pack(a, b, c, d):
    """4 finite f32 -> 4 fp8 FNUZ in one i32 (same result as ``_pack_activation``:
    clamp to +-240, RNE, no negative zero), with NaN-free min/max."""
    v = [fx.maxnumf(fx.Float32(fx.arith.minnumf(_raw(fx.Float32(x)), _raw(fx.Float32(240.0)))),
                    fx.Float32(-240.0))
         for x in (a, b, c, d)]
    lo = rocdl.cvt_pk_fp8_f32(T.i32, v[0], v[1], fx.Int32(0), False)
    packed = fx.Uint32(rocdl.cvt_pk_fp8_f32(T.i32, v[2], v[3], lo, True))
    mag = packed & fx.Uint32(0x7F7F7F7F)
    zero = (~(mag + fx.Uint32(0x7F7F7F7F))) & fx.Uint32(0x80808080)
    return (packed & ~zero).bitcast(fx.Int32)


def _bf16x2_rne(a, b):
    """2 finite f32 -> one i32 of 2 bf16, round-to-nearest-even (a in the low half;
    no NaN handling: epilogue values are finite)."""
    r = []
    for x in (a, b):
        u = fx.Float32(x).bitcast(fx.Uint32)
        r.append(u + fx.Uint32(0x7FFF) + ((u >> fx.Uint32(16)) & fx.Uint32(1)))
    return _perm_u32(r[1], r[0], 0x07060302).bitcast(fx.Int32)


def _rnu_hi(x):
    """f32 -> i32 whose high half is bf16(x), rounded to nearest, ties away from
    zero (one add; differs from RNE only on exact ties)."""
    return fx.Float32(x).bitcast(fx.Int32) + fx.Int32(0x8000)


# ---- fp4 -> fp8 decode of the fd2 weights -------------------------------------------
# W stays E2M1/E8M0. Output channel n is normalized by 2**(max_exp[n]-127-5)
# (``_channel_factor``, applied in the epilogue); per 32-K group the remaining
# 2**(5-delta), delta = max_exp - group_exp, is folded into an 8-entry
# E2M1-magnitude -> FP8-FNUZ byte table (``_fp8_lut_predelta``) so each nibble
# decodes with one v_perm.
#
# W and scales are pre-transformed by ``prep_w1_fast`` (``tp_gemm1``): each dword holds
# (k_j, k_{j+4}) in byte j (low, high nibble) and no -0 nibble, so even nibbles decode
# straight to K 0..3 and odd nibbles to K 4..7 (no interleave v_perm), and the sign is
# bit 3 of the nibble moved to bit 7 (no sign-table v_perm). The scale byte already
# holds min(delta, 11) * 8; groups below 2**-11 of the channel max (delta > 11) are
# zeroed in W, so they decode to 0.


def _perm_u32(hi, lo, sel):
    return fx.Uint32(
        rocdl.perm_b32(
            fx.Uint32(hi).ir_value(), fx.Uint32(lo).ir_value(), fx.Uint32(sel).ir_value()
        )
    )


def _fp4x8_to_fp8x8_fast(raw, lut):
    lo, hi, smask = lut
    raw = fx.Uint32(raw)
    even = ((raw << 4) & smask) | _perm_u32(hi, lo, raw & 0x07070707)
    odd = (raw & smask) | _perm_u32(hi, lo, (raw >> 4) & 0x07070707)
    return fx.Vector.from_elements(
        [even.bitcast(fx.Int32), odd.bitcast(fx.Int32)], fx.Int32
    ).bitcast(fx.Float8E4M3FNUZ)


def _fp8_lut_predelta(delta):
    """LUT of a 32-K group from its fd2 scale byte (min(max-code, 11)*8, broadcast
    to all 4 bytes)."""
    delta = fx.Uint32(delta)
    lo = (fx.Uint32(0x6C686080) - delta) & fx.Uint32(0xFFFFFF00)
    hi = fx.Uint32(0x7C787470) - delta
    return lo, hi, fx.Uint32(0x80808080)


def _channel_factor(exponent):
    """2**(exponent-127-5) as f32 (subnormal-safe)."""
    normal = ((exponent - 5).to(fx.Uint32) << 23).bitcast(fx.Float32)
    subnormal = (fx.Uint32(1) << (exponent + 17).to(fx.Uint32)).bitcast(fx.Float32)
    return (exponent >= 6).select(normal, subnormal)
