# SPDX-License-Identifier: MIT
"""Shared GRRead data movement, synchronization, and numerical helpers without experimental branches."""

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import rocdl
from pyhip.contrib.flydsl.helpers import cvt_f32_to_bf16


def _weight_view(pointer, n, k):
    return fx.make_view(pointer, fx.make_layout(
        ((16, n // 16), (8, 4, k // 32)), ((8, 16 * k), (1, 128, 512))))


def _sigmoid_down(value):
    exponent = fx.Float32(fx.rocdl.exp2(fx.T.f32, fx.arith.unwrap(-value * 1.4426950408889634)))
    return fx.Float32(fx.rocdl.rcp(fx.T.f32, fx.arith.unwrap(1.0 + exponent)))


def _pin_v(value):
    return fx.Int32(llvm.inline_asm(fx.Int32.ir_type, [value.ir_value()], "", "=v,0", has_side_effects=True))


def _schedule_boundary():
    # 删除原调试文本，保留side-effect与两侧compiler barrier，避免改变已验收的机器调度。
    rocdl.sched_barrier(0)
    llvm.inline_asm(ir.Type.parse("!llvm.void"), [], "", "", has_side_effects=True)
    rocdl.sched_barrier(0)


def _priority(value):
    rocdl.sched_barrier(0)
    rocdl.s_setprio(value)
    rocdl.sched_barrier(0)


def _barrier():
    rocdl.sched_barrier(0)
    rocdl.s_barrier()
    rocdl.sched_barrier(0)


def _load(resource, address, scalar_offset=0, words=4):
    value = rocdl.RawPtrBufferLoadOp(
        fx.Int32.ir_type if words == 1 else ir.VectorType.get([words], fx.Int32.ir_type), resource,
        address.ir_value(), fx.Int32(scalar_offset).ir_value(),
        aux=ir.IntegerAttr.get(fx.Int32.ir_type, 0)).result
    return fx.Int32(value) if words == 1 else fx.Vector(value)


def _store(resource, address, values, scalar_offset=0):
    rocdl.RawPtrBufferStoreOp(values.ir_value(), resource, address.ir_value(),
        fx.Int32(scalar_offset).ir_value(), aux=ir.IntegerAttr.get(fx.Int32.ir_type, 0))


def _ds_read(address, offset):
    value = llvm.inline_asm(ir.VectorType.get([4], fx.Int32.ir_type),
        [address.ir_value()], f"ds_read_b128 $0, $1 offset:{offset}",
        "=v,v,~{memory}", has_side_effects=True)
    return fx.Vector(value)


def _ds_write(address, values, offset):
    llvm.inline_asm(ir.Type.parse("!llvm.void"), [address.ir_value(), values.ir_value()],
        f"ds_write_b128 $0, $1 offset:{offset}", "v,v,~{memory}", has_side_effects=True)


def _mfma(a, b, c):
    return fx.Vector(rocdl.mfma_f32_16x16x16bf16_1k(
        ir.VectorType.get([4], fx.Float32.ir_type),
        [a.bitcast(fx.Int16).ir_value(), b.bitcast(fx.Int16).ir_value(), c.ir_value(), 0, 0, 0]))


def _sigmoid(value):
    exponent = fx.Float32(rocdl.exp2(fx.Float32.ir_type, (-value * 1.4426950408889634).ir_value()))
    return fx.Float32(rocdl.rcp(fx.Float32.ir_type, (1.0 + exponent).ir_value()))


def _pack_y_mean(v0, v1):
    """Round the four-stream mean with the integer helper and pack two BF16 values into one DWORD."""
    values = fx.Vector.from_elements([v0 * 0.25, v1 * 0.25], fx.Float32)
    fragment = fx.make_rmem_tensor(fx.make_layout(2, 1), fx.Float32)
    fragment.store(values)
    return fx.Vector(cvt_f32_to_bf16(fragment).load()).bitcast(fx.Int32)[0]