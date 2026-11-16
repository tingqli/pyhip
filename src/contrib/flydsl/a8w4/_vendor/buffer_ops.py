# Vendored from aiter/ops/flydsl/kernels (aiter 7cd70aca8) so the a8w4 package
# does not depend on aiter's private FlyDSL helpers. Imports made relative.
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Modifications Copyright (C) 2026 Advanced Micro Devices, Inc.

"""AMD buffer load/store operations (the subset a8w4 uses).

Buffer instructions take a resource descriptor (V#: base, size, flags) in SGPRs plus a
32-bit per-lane byte offset; out-of-range accesses read 0 / are dropped.

Upstream: FlyDSL ``kernels/common/buffer_ops.py`` @ ROCm/FlyDSL#880, via aiter. Kept:
``create_buffer_resource_from_addr``, ``buffer_load``, ``buffer_store``,
``get_element_ptr``, minus the options nobody here passes; the FlyDSL-version and
RDNA branches are frozen to gfx942 + FlyDSL 0.3.2.
"""

from __future__ import annotations

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm, rocdl
from flydsl._mlir.extras import types as T
from flydsl.expr.meta import dsl_loc_tracing

# V# word 3 on CDNA (gfx9xx): DATA_FORMAT = 7 (bits 14:12), NUM_FORMAT = 4 (bits 18:15),
# everything else 0 (RDNA would also need bit 24 and OOB_SELECT).
_BUFFER_FLAGS = (7 << 12) | (4 << 15)


def _unwrap_value(value):
    """Recursively unwrap ArithValue or similar wrappers to get the actual MLIR value.

    Handles:
    - FlyDSL ArithValue (has ._value)
    - flyc DSL Numeric like fx.Int32 (has .ir_value() method)
    - flyc ArithValue (is already ir.Value subclass)
    """
    # DSL Numeric (Int32, Float32, etc.) — use ir_value() to materialize
    if hasattr(value, "ir_value") and not isinstance(value, ir.Value):
        return value.ir_value()
    max_depth = 10  # Safety limit
    depth = 0
    while depth < max_depth and not isinstance(value, ir.Value):
        if hasattr(value, "_value"):
            value = value._value
        elif hasattr(value, "value"):
            value = value.value
        else:
            break
        depth += 1
    return value


@dsl_loc_tracing
def get_element_ptr(base_ptr, byte_offset) -> ir.Value:
    """i8 GEP: ``base_ptr + byte_offset`` (an integer or index IR value)."""
    base_ptr = _unwrap_value(base_ptr)
    offset_val = _unwrap_value(byte_offset)
    if isinstance(offset_val.type, ir.IndexType):
        offset_val = fx.Int64(offset_val).ir_value()
    return llvm.GEPOp(base_ptr.type, base_ptr, [offset_val], [-(2**31)], T.i8(), None).result


@dsl_loc_tracing
def create_buffer_resource_from_addr(
    addr_i64: ir.Value,
    *,
    num_records_bytes: int | ir.Value | None = None,
) -> ir.Value:
    """Buffer resource descriptor (!llvm.ptr<8>) over a raw i64 device address.

    ``num_records_bytes``: size in bytes for the hardware OOB check (None: 0xFFFFFFFF).
    """
    addr_i64 = _unwrap_value(addr_i64)
    ptr_type = ir.Type.parse("!llvm.ptr")
    base_ptr = llvm.IntToPtrOp(ptr_type, addr_i64).result
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
    return rocdl.MakeBufferRsrcOp(
        rsrc_type, base_ptr, stride, num_records, flags
    ).result


def _aux(cache_modifier):
    # FlyDSL 0.3.2: the cache policy is an i32 attribute (older versions: an operand)
    return ir.IntegerAttr.get(ir.IntegerType.get_signless(32), cache_modifier)


@dsl_loc_tracing
def buffer_load(
    rsrc: ir.Value,
    offset: ir.Value,
    vec_width: int = 4,
    dtype=None,
    mask: ir.Value | None = None,
    cache_modifier: int = 0,
) -> ir.Value:
    """``vec_width`` elements of ``dtype`` (default f32; an ir.Type or fx class) at
    element ``offset``; lanes with ``mask`` false read 0 (offset pushed out of range).
    ``cache_modifier``: 0 cached, 2 nt, 3 sc0|sc1."""
    if dtype is None:
        dtype = T.f32()
    elif hasattr(dtype, "ir_type"):
        dtype = dtype.ir_type

    # Buffer offsets truncate wider integers and sign-extend narrower ones.
    offset = fx.Int32(_unwrap_value(offset))
    # the instruction offset is in BYTES
    element_bytes = dtype.width // 8
    offset = offset * element_bytes
    if mask is not None:
        offset = fx.Boolean(_unwrap_value(mask)).select(offset, 0x7FFFFFFF)

    if vec_width == 1:
        result_type = dtype
    else:
        result_type = ir.VectorType.get([vec_width], dtype)

    soffset = fx.Int32(0).ir_value()
    load_op = rocdl.RawPtrBufferLoadOp(
        result_type,
        rsrc,
        offset.ir_value(),
        soffset,
        aux=_aux(cache_modifier),
    )
    return load_op.result


@dsl_loc_tracing
def buffer_store(
    data: ir.Value,
    rsrc: ir.Value,
    offset: ir.Value,
    mask: ir.Value | None = None,
    cache_modifier: int = 0,
):
    """Store ``data`` (scalar or vector) at element ``offset`` (in units of its element
    type); lanes with ``mask`` false store nothing."""
    data = _unwrap_value(data)
    rsrc = _unwrap_value(rsrc)
    offset = fx.Int32(_unwrap_value(offset))

    # the instruction offset is in BYTES
    data_type = data.type
    if hasattr(data_type, "element_type"):  # Vector type
        element_type = data_type.element_type
    else:  # Scalar type
        element_type = data_type
    element_bytes = element_type.width // 8
    offset = offset * element_bytes
    if mask is not None:
        offset = fx.Boolean(_unwrap_value(mask)).select(offset, 0x7FFFFFFF)

    soffset = fx.Int32(0).ir_value()
    rocdl.RawPtrBufferStoreOp(
        data,
        rsrc,
        offset.ir_value(),
        soffset,
        aux=_aux(cache_modifier),
    )
