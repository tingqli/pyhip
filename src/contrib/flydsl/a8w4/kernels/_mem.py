# SPDX-License-Identifier: Apache-2.0 AND MIT
# Derived from aiter 7cd70aca8 (aiter/ops/flydsl/kernels): buffer_ops.py and
# mxfp4_gemm_common.py (Apache-2.0, Copyright (c) 2025-2026 FlyDSL Project Contributors),
# tensor_shim.py and moe_2stage_a16wmix/{a8w4,utils}.py (MIT, Copyright (C) 2024-2026,
# Advanced Micro Devices, Inc.). Modifications Copyright (C) 2026 Advanced Micro Devices, Inc.
"""Device-side memory helpers of the a8w4 kernels: IR value glue, index math, global / LDS
pointers, buffer (V#) loads and stores, i32 buffer views.

Only what the kernels call, with the gfx942 + FlyDSL 0.3.2 branches frozen. Import these by
name (``from ._mem import buffer_load``), not through a module object: FlyDSL 0.3.2 puts the
source of the plain helpers a kernel reaches by name, in the kernel's own directory, into the
kernel's cache key. The scan does not enter nested functions, so a helper called only inside
one is still missed: after editing a helper, run with a fresh ``FLYDSL_RUNTIME_CACHE_DIR``.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm, rocdl
from flydsl.expr import arith
from flydsl.expr.typing import T

# --------------------------------------------------------------------------------------
# IR values, index math
# --------------------------------------------------------------------------------------


def _raw(v):
    """Convert ArithValue / Numeric (Int32, Boolean, …) to raw ir.Value."""
    if isinstance(v, ir.Value):
        return v
    if hasattr(v, "ir_value"):
        return _raw(v.ir_value())
    return ir.Value._CAPICreate(v._CAPIPtr)


def _unwrap_value(value):
    """Like ``_raw``, but passes Python ints (and anything else not wrapped) through."""
    if hasattr(value, "ir_value") and not isinstance(value, ir.Value):
        return value.ir_value()
    depth = 0
    while depth < 10 and not isinstance(value, ir.Value):
        if hasattr(value, "_value"):
            value = value._value
        elif hasattr(value, "value"):
            value = value.value
        else:
            break
        depth += 1
    return value


def _udiv(a, c):
    cc = fx.Int32(c) if isinstance(c, int) else c
    return fx.Int32(arith.divui(_raw(a), _raw(cc)))


def _umod(a, c):
    cc = fx.Int32(c) if isinstance(c, int) else c
    return fx.Int32(arith.remui(_raw(a), _raw(cc)))


# --------------------------------------------------------------------------------------
# Pointers
# --------------------------------------------------------------------------------------


def _pointer(address, dtype, alignment=16):
    """Global fx.Pointer to ``dtype`` (an fx class) over a raw i64 device address."""
    return fx.inttoptr(
        fx.PointerType.get(dtype.ir_type, fx.AddressSpace.Global, alignment),
        fx.Int64(address),
    )


def global_typed_ptr(arg, elem_ty, align=4, *, byte_offset=None):
    """Typed global fx.Pointer over a raw i64 device address; index in ELEMENTS (ptr[i]), not bytes."""
    ptr_ty = fx.PointerType.get(elem_ty, fx.AddressSpace.Global, align)
    if byte_offset is not None:
        byte_ptr_ty = fx.PointerType.get(T.i8, fx.AddressSpace.Global, align)
        base = fx.inttoptr(byte_ptr_ty, fx.Int64(arg))
        return fx.recast_iter(ptr_ty, fx.add_offset(base, byte_offset))
    return fx.inttoptr(ptr_ty, fx.Int64(arg))


def _global_i32_at(addr_i64, idx):
    return global_typed_ptr(addr_i64, T.i32)[idx]


def _global_base_ptr1(addr_i64):
    """Raw llvm i8 global pointer over an i64 device address (for ``get_element_ptr``)."""
    ptr_ty = fx.PointerType.get(T.i8, fx.AddressSpace.Global)
    return fx.to_llvm_ptr(fx.inttoptr(ptr_ty, fx.Int64(addr_i64)))


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


def _lds_ptr3(base_i32, byte_off_i32):
    """Raw llvm i8 LDS pointer at ``base + byte_off``."""
    ptr_ty = fx.PointerType.get(T.i8, fx.AddressSpace.Shared)
    return fx.to_llvm_ptr(fx.inttoptr(ptr_ty, fx.Int64(base_i32 + byte_off_i32)))


def get_element_ptr(base_ptr, byte_offset):
    """i8 GEP on a raw llvm pointer: ``base_ptr + byte_offset`` (an integer or index IR value)."""
    base_ptr = _unwrap_value(base_ptr)
    offset_val = _unwrap_value(byte_offset)
    if isinstance(offset_val.type, ir.IndexType):
        offset_val = fx.Int64(offset_val).ir_value()
    return llvm.GEPOp(base_ptr.type, base_ptr, [offset_val], [-(2**31)], T.i8, None).result


def _store(ptr, idx, value):
    """ptr[idx] = value (a call: inside a dynamic if, the AST rewriter would take a
    subscript store to a closure variable for an assignment of it)."""
    ptr[idx] = value


@flyc.jit
def _lds_f32_store_masked(lds_f32, idx, val, ok):
    if ok:
        lds_f32[idx] = val


# --------------------------------------------------------------------------------------
# Buffer (V#) loads and stores: a resource descriptor (base, size, flags) in SGPRs plus a
# 32-bit per-lane byte offset; out-of-range accesses read 0 / are dropped.
# --------------------------------------------------------------------------------------

# V# word 3 on CDNA (gfx9xx): DATA_FORMAT = 7 (bits 14:12), NUM_FORMAT = 4 (bits 18:15),
# everything else 0 (RDNA would also need bit 24 and OOB_SELECT).
_BUFFER_FLAGS = (7 << 12) | (4 << 15)


def create_buffer_resource_from_addr(addr_i64, *, num_records_bytes=None):
    """Buffer resource descriptor (!llvm.ptr<8>) over a raw i64 device address.

    ``num_records_bytes``: size in bytes for the hardware OOB check (None: 0xFFFFFFFF).
    """
    addr_i64 = _unwrap_value(addr_i64)
    base_ptr = llvm.IntToPtrOp(ir.Type.parse("!llvm.ptr"), addr_i64).result
    flags = fx.Int32(_BUFFER_FLAGS).ir_value()
    stride = fx.Int16(0).ir_value()
    if num_records_bytes is None:
        num_records = fx.Int64(0xFFFFFFFF).ir_value()
    elif isinstance(num_records_bytes, int):
        nbytes = max(0, min(int(num_records_bytes), 0xFFFFFFFF))
        num_records = fx.Int64(nbytes).ir_value()
    else:
        num_records = fx.Int64(_unwrap_value(num_records_bytes)).ir_value()
    rsrc_type = ir.Type.parse("!llvm.ptr<8>")
    return rocdl.MakeBufferRsrcOp(rsrc_type, base_ptr, stride, num_records, flags).result


def _aux(cache_modifier):
    # FlyDSL 0.3.2: the cache policy is an i32 attribute (older versions: an operand)
    return ir.IntegerAttr.get(ir.IntegerType.get_signless(32), cache_modifier)


def buffer_load(rsrc, offset, vec_width=4, dtype=None, mask=None, cache_modifier=0):
    """``vec_width`` elements of ``dtype`` (default f32; an ir.Type or fx class) at
    element ``offset``; lanes with ``mask`` false read 0 (offset pushed out of range).
    ``cache_modifier``: bits ``CM_*`` of ``_hw`` (0 cached, 2 nt, 17 sc0|sc1)."""
    if dtype is None:
        dtype = T.f32
    elif hasattr(dtype, "ir_type"):
        dtype = dtype.ir_type

    # Buffer offsets truncate wider integers and sign-extend narrower ones.
    offset = fx.Int32(_unwrap_value(offset))
    # the instruction offset is in BYTES
    offset = offset * (dtype.width // 8)
    if mask is not None:
        offset = fx.Boolean(_unwrap_value(mask)).select(offset, 0x7FFFFFFF)

    result_type = dtype if vec_width == 1 else ir.VectorType.get([vec_width], dtype)
    soffset = fx.Int32(0).ir_value()
    return rocdl.RawPtrBufferLoadOp(
        result_type, rsrc, offset.ir_value(), soffset, aux=_aux(cache_modifier)
    ).result


def buffer_store(data, rsrc, offset, mask=None, cache_modifier=0):
    """Store ``data`` (scalar or vector) at element ``offset`` (in units of its element
    type); lanes with ``mask`` false store nothing. ``cache_modifier``: as ``buffer_load``."""
    data = _unwrap_value(data)
    rsrc = _unwrap_value(rsrc)
    offset = fx.Int32(_unwrap_value(offset))

    # the instruction offset is in BYTES
    element_type = getattr(data.type, "element_type", data.type)
    offset = offset * (element_type.width // 8)
    if mask is not None:
        offset = fx.Boolean(_unwrap_value(mask)).select(offset, 0x7FFFFFFF)

    soffset = fx.Int32(0).ir_value()
    rocdl.RawPtrBufferStoreOp(data, rsrc, offset.ir_value(), soffset, aux=_aux(cache_modifier))


def _rsrc(addr, nbytes=None):
    """Buffer resource over the i64 address ``addr``; ``nbytes``: bound of the hardware range
    check (None: 4 GB)."""
    if nbytes is None:
        return create_buffer_resource_from_addr(_raw(fx.Int64(addr)))
    return create_buffer_resource_from_addr(
        _raw(fx.Int64(addr)), num_records_bytes=_raw(fx.Int64(nbytes)))


def _ld(rsrc, idx, width=1, mask=None):
    """i32 (width 1) / i32 x width at i32 element ``idx`` (masked: 0)."""
    return buffer_load(rsrc, idx, vec_width=width, dtype=T.i32, mask=mask)


# --------------------------------------------------------------------------------------
# i32 buffer views (layout API: BufferCopy atoms)
# --------------------------------------------------------------------------------------


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
    return fx.rocdl.make_buffer_tensor(view, max_size=False, num_records_bytes=num_bytes_i64)


def _global_i32_buffer_tiles(addr_i64, num_bytes, tile_elems):
    return fx.logical_divide(
        _global_i32_buffer_view(addr_i64, num_bytes), fx.make_layout(tile_elems, 1)
    )


def _buffer_i32_scalar_read(tiles1, idx, atom):
    """Read one i32 dword at element ``idx`` from a ``_global_i32_buffer_tiles(..., 1)``
    view via the layout-API BufferCopy atom (buffer_load_dword; OOB-clamped by the
    buffer resource). ``tiles1`` is 1-dword tiles so the tile index == ``idx``.
    """
    r = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Int32)
    fx.copy(atom, fx.slice(tiles1, (None, idx)), r)
    return fx.Int32(fx.Vector(fx.memref_load_vec(r))[0])
