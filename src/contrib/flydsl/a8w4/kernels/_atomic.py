# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Atomics between workgroups, and the coherent load that polls what they wrote.

Global atomics are monotonic at agent scope (coherent across the XCD L2s), LDS atomics at
workgroup scope.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import arith, llvm
from flydsl.expr import range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from ._hw import CM_SYS
from ._mem import (
    _global_base_ptr1,
    _lds_ptr3,
    _raw,
    buffer_load,
    create_buffer_resource_from_addr,
    get_element_ptr,
)


def _gatomic_add(addr_i64, idx, val):
    """old = atomic add, agent scope (device-coherent across the XCD L2s)."""
    return fx.Int32(llvm.AtomicRMWOp(
        llvm.AtomicBinOp.add, get_element_ptr(_global_base_ptr1(addr_i64), idx * fx.Int32(4)),
        _raw(fx.Int32(val)), llvm.AtomicOrdering.monotonic, syncscope="agent",
        alignment=4).result)


def _gatomic_xchg(addr_i64, idx, val):
    llvm.AtomicRMWOp(
        llvm.AtomicBinOp.xchg, get_element_ptr(_global_base_ptr1(addr_i64), idx * fx.Int32(4)),
        _raw(fx.Int32(val)), llvm.AtomicOrdering.monotonic, syncscope="agent", alignment=4)


def _gload_sys(addr_i64, idx):
    rs = create_buffer_resource_from_addr(_raw(fx.Int64(addr_i64)))
    return fx.Int32(buffer_load(rs, idx, vec_width=1, dtype=T.i32,
                                cache_modifier=CM_SYS))


def _latomic_add(lds_i32, idx, val):
    return fx.Int32(llvm.AtomicRMWOp(
        llvm.AtomicBinOp.add, _lds_ptr3(lds_i32, idx * fx.Int32(4)), _raw(fx.Int32(val)),
        llvm.AtomicOrdering.monotonic, syncscope="workgroup", alignment=4).result)


@flyc.jit
def _atomic_bf16x4(arg_out, base, v4, ok):
    """out[base .. base+3] += 4 f32 values rounded to bf16 (2 bf16x2 agent atomics)."""
    if ok:
        for h in range_constexpr(2):
            pk = Vec.from_elements([v4[2 * h], v4[2 * h + 1]], fx.Float32).to(fx.BFloat16)
            llvm.AtomicRMWOp(
                llvm.AtomicBinOp.fadd,
                get_element_ptr(_global_base_ptr1(arg_out), (base + fx.Int32(2 * h)) * fx.Int32(2)),
                _raw(pk), llvm.AtomicOrdering.monotonic, syncscope="agent", alignment=4)


def _atomics4_saddr(datas, offs, out_i64):
    """bf16x2 atomic adds of ``datas[k]`` at ``out + offs[k]`` bytes (offs >= 0,
    per-tile i32, so the zext stays next to the atomic -> SGPR-base form)."""
    base = _global_base_ptr1(out_i64)
    for k in range_constexpr(len(offs)):
        off64 = arith.ExtUIOp(T.i64, offs[k]).result
        pk = Vec.from_elements([fx.Int32(datas[k])], fx.Int32).bitcast(fx.BFloat16)
        llvm.AtomicRMWOp(llvm.AtomicBinOp.fadd, get_element_ptr(base, byte_offset=off64),
                         _raw(pk), llvm.AtomicOrdering.monotonic, syncscope="agent",
                         alignment=4)


@flyc.jit
def _atomics_mi(datas, offs, col, out_i64, full, part):
    """One 16-row block's atomics at ``out + offs[k] + col``. Valid rows are a prefix
    of the expert block, so ``full`` (all 16 rows valid) and ``part`` (some valid)
    are wave-uniform: full blocks need no per-atomic exec masking; padding rows have
    offs < 0."""
    if full:
        _atomics4_saddr(datas, [_raw(fx.Int32(o) + col) for o in offs], out_i64)
    else:
        if part:
            base = _global_base_ptr1(out_i64)
            for k in range_constexpr(len(offs)):
                off = fx.Int32(offs[k])
                if off >= fx.Int32(0):
                    off64 = arith.ExtUIOp(T.i64, _raw(off + col)).result
                    pk = Vec.from_elements([fx.Int32(datas[k])], fx.Int32).bitcast(fx.BFloat16)
                    llvm.AtomicRMWOp(llvm.AtomicBinOp.fadd,
                                     get_element_ptr(base, byte_offset=off64), _raw(pk),
                                     llvm.AtomicOrdering.monotonic, syncscope="agent",
                                     alignment=4)
