# Vendored from aiter/ops/flydsl/kernels (aiter 7cd70aca8) so the a8w4 package
# does not depend on aiter's private FlyDSL helpers. Imports made relative.
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors

import flydsl.expr as fx
from flydsl.expr.typing import T

from . import buffer_ops


def _lds_ptr3(base_i32, byte_off_i32):
    ptr_ty = fx.PointerType.get(T.i8, fx.AddressSpace.Shared)
    return fx.to_llvm_ptr(fx.inttoptr(ptr_ty, fx.Int64(base_i32 + byte_off_i32)))


def _global_base_ptr1(addr_i64):
    ptr_ty = fx.PointerType.get(T.i8, fx.AddressSpace.Global)
    return fx.to_llvm_ptr(fx.inttoptr(ptr_ty, fx.Int64(addr_i64)))


def _gep1(base_ptr, byte_off_i32):
    return buffer_ops.get_element_ptr(base_ptr, byte_offset=byte_off_i32)


def global_typed_ptr(arg, elem_ty, align=4, *, byte_offset=None):
    """Typed global fx.Pointer over a raw i64 device address; index in ELEMENTS (ptr[i]), not bytes."""
    ptr_ty = fx.PointerType.get(elem_ty, fx.AddressSpace.Global, align)
    if byte_offset is not None:
        byte_ptr_ty = fx.PointerType.get(T.i8, fx.AddressSpace.Global, align)
        base = fx.inttoptr(byte_ptr_ty, fx.Int64(arg))
        return fx.recast_iter(ptr_ty, fx.add_offset(base, byte_offset))
    return fx.inttoptr(ptr_ty, fx.Int64(arg))


def _global_i32_buffer_view(addr_i64, num_bytes):
    # BufferCopy soffset is in elements; make_layout's dynamic shape must be
    # i32/i64 rather than Index. Keep the hardware OOB bound in bytes.
    num_bytes_i64 = fx.Int64(num_bytes)
    view = fx.Tensor(
        fx.make_view(
            global_typed_ptr(addr_i64, T.i32),
            fx.make_layout(num_bytes_i64 // fx.Int64(4), 1),
        )
    )
    return fx.rocdl.make_buffer_tensor(
        view, max_size=False, num_records_bytes=num_bytes_i64
    )


def _global_i32_buffer_tiles(addr_i64, num_bytes, tile_elems):
    return fx.logical_divide(
        _global_i32_buffer_view(addr_i64, num_bytes), fx.make_layout(tile_elems, 1)
    )


def lds_typed_ptr(base_i32, elem_ty, align=4, *, byte_offset=None):
    """Typed LDS (Shared) fx.Pointer over an i32 LDS base; index in ELEMENTS (ptr[i]), not bytes."""
    ptr_ty = fx.PointerType.get(elem_ty, fx.AddressSpace.Shared, align)
    if byte_offset is not None:
        byte_ptr_ty = fx.PointerType.get(T.i8, fx.AddressSpace.Shared, align)
        base = fx.inttoptr(byte_ptr_ty, fx.Int32(base_i32))
        return fx.recast_iter(ptr_ty, fx.add_offset(base, byte_offset))
    return fx.inttoptr(ptr_ty, fx.Int32(base_i32))


def lds_vec_load(base_i32, byte_off_i32, result_type, elem_ty, align=4):
    """Typed LDS ds-read at a BYTE offset from the i32 LDS base; mirrors raw llvm.load (vector or scalar)."""
    elem_ir_ty = elem_ty.ir_type if hasattr(elem_ty, "ir_type") else elem_ty
    ptr = lds_typed_ptr(base_i32, elem_ir_ty, align=align, byte_offset=byte_off_i32)
    return fx.ptr_load(ptr, result_type=result_type)


