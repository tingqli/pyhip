# SPDX-License-Identifier: MIT
"""Shared GR write buffer access, lane exchange and BF16 conversion helpers."""

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import rocdl
from pyhip.codegen.flydsl.helpers import rocdl_aux

NT = 'nt'
HIGH_HALF = 0xFFFF0000
# DPP controls: quad_perm [1,0,3,2], quad_perm [2,3,0,1], row_ror:4, row_ror:8.
DPP_XOR1, DPP_XOR2, DPP_ROR4, DPP_ROR8 = 0xB1, 0x4E, 0x124, 0x128
# ds_swizzle bit mode: and_mask 0x1F, xor_mask 0x10 (lane ^ 16 inside 32-lane halves).
SWIZZLE_XOR16 = 0x401F


def scalar(value):
    return fx.Int32(rocdl.readfirstlane(fx.Int32.ir_type, value.ir_value()))


def resource(view, num_bytes=None):
    tensor = fx.rocdl.make_buffer_tensor(view, max_size=False, num_records_bytes=num_bytes) \
        if num_bytes is not None else fx.rocdl.make_buffer_tensor(view, max_size=False)
    return fx.rocdl.get_buffer_rsrc(fx.get_iter(tensor))


def load(rsrc, offset, words=4, aux=0):
    result_type = fx.Int32.ir_type if words == 1 else ir.VectorType.get([words], fx.Int32.ir_type)
    value = rocdl.RawPtrBufferLoadOp(result_type, rsrc, fx.Int32(offset).ir_value(), fx.Int32(0).ir_value(),
                                     aux=rocdl_aux(aux)).result
    return fx.Int32(value) if words == 1 else fx.Vector(value)


def store(rsrc, offset, values, aux=0):
    rocdl.RawPtrBufferStoreOp(values.ir_value(), rsrc, fx.Int32(offset).ir_value(), fx.Int32(0).ir_value(),
                              aux=rocdl_aux(aux))


def dpp(value, ctrl):
    """Full-mask DPP move of an FP32 value (undefined old value lets LLVM fold it into the consumer)."""
    i32 = fx.Int32.ir_type
    bound_ctrl = fx.arith.unwrap(fx.arith.constant(True, type=ir.IntegerType.get_signless(1)))
    moved = llvm.call_intrinsic(i32, 'llvm.amdgcn.update.dpp.i32', [
        llvm.mlir_poison(i32), value.bitcast(fx.Int32).ir_value(), fx.Int32(ctrl).ir_value(),
        fx.Int32(0xF).ir_value(), fx.Int32(0xF).ir_value(), bound_ctrl], [], [])
    return fx.Int32(moved).bitcast(fx.Float32)


def lane_reduce_tail(value, lane):
    """Finish a row-local sum: add lanes (l + 4k) mod 16, then lane ^ 16 and lane ^ 32."""
    value = value + dpp(value, DPP_ROR4)
    value = value + dpp(value, DPP_ROR8)
    swapped = rocdl.ds_swizzle(fx.Int32.ir_type, value.bitcast(fx.Int32).ir_value(),
                               fx.Int32(SWIZZLE_XOR16).ir_value())
    value = value + fx.Int32(swapped).bitcast(fx.Float32)
    permuted = rocdl.ds_bpermute(fx.Int32.ir_type, ((lane ^ 32) * 4).ir_value(),
                                 value.bitcast(fx.Int32).ir_value())
    return value + fx.Int32(permuted).bitcast(fx.Float32)


def pair(lo, hi):
    return fx.Vector.from_elements([lo, hi], fx.Float32)


def pk_fma(a, b, c):
    result_type = ir.VectorType.get([2], fx.Float32.ir_type)
    return fx.Vector(llvm.call_intrinsic(result_type, 'llvm.fma.v2f32',
                                         [a.ir_value(), b.ir_value(), c.ir_value()], [], []))


def bf16_pair(word):
    """Unpack one DWORD holding two BF16 values into an FP32 pair (low element first)."""
    bits = word.bitcast(fx.Uint32)
    return pair((bits << 16).bitcast(fx.Float32), (bits & fx.Uint32(HIGH_HALF)).bitcast(fx.Float32))


def rne_bits(value):
    """FP32 bits with the BF16 round-to-nearest-even bias added; the high half is the BF16 result."""
    bits = value.bitcast(fx.Uint32)
    return bits + 0x7FFF + ((bits >> 16) & 1)


def pack_high(hi_bits, lo_bits):
    """Pack the high halves of two DWORDs: (hi_bits[31:16] << 16) | lo_bits[31:16]."""
    return fx.Uint32(llvm.call_intrinsic(fx.Uint32.ir_type, 'llvm.amdgcn.perm',
                                         [hi_bits.ir_value(), lo_bits.ir_value(),
                                          fx.Uint32(0x07060302).ir_value()], [], []))


def make_claim(counter_address):
    """Return a single-lane in-order claim: the previous value of atomic_add(counter, 1).

    The address gets a VGPR-sourced zero offset: with a uniform address the AMDGPU
    atomic optimizer rewrites the claim into a wave scan that waits on vmcnt(0),
    which (vmcnt is in-order) would also wait for every prefetch in flight. For the
    same reason the caller must not touch the result until it is really needed.
    """
    def claim():
        zero = fx.Int32(llvm.inline_asm(fx.Int32.ir_type, [], 'v_mov_b32 $0, 0', '=v', has_side_effects=False))
        pointer = llvm.inttoptr(ir.Type.parse('!llvm.ptr<1>'), (counter_address + fx.Int64(zero)).ir_value())
        return fx.Int32(llvm.AtomicRMWOp(llvm.AtomicBinOp.add, pointer, fx.Int32(1).ir_value(),
                                         llvm.AtomicOrdering.monotonic, syncscope='agent').res)

    return claim
