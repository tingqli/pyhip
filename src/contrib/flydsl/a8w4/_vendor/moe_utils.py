# Vendored from aiter/ops/flydsl/kernels (aiter 7cd70aca8) so the a8w4 package
# does not depend on aiter's private FlyDSL helpers. Imports made relative.
"""Index / pointer helpers from moe_2stage_a16wmix/utils.py."""

import flydsl.expr as fx
from flydsl.expr import arith
from flydsl.expr.typing import T

from .mxfp4_gemm_common import global_typed_ptr
from .tensor_shim import _to_raw as _raw


def _udiv(a, c):
    cc = fx.Int32(c) if isinstance(c, int) else c
    return fx.Int32(arith.divui(_raw(a), _raw(cc)))


def _umod(a, c):
    cc = fx.Int32(c) if isinstance(c, int) else c
    return fx.Int32(arith.remui(_raw(a), _raw(cc)))


def _global_i32_at(addr_i64, idx):
    return global_typed_ptr(addr_i64, T.i32)[idx]


def _buffer_i32_scalar_read(tiles1, idx, atom):
    """Read one i32 dword at element ``idx`` from a ``_global_i32_buffer_tiles(..., 1)``
    view via the layout-API BufferCopy atom (buffer_load_dword; OOB-clamped by the
    buffer resource). ``tiles1`` is 1-dword tiles so the tile index == ``idx``.
    """
    r = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Int32)
    fx.copy(atom, fx.slice(tiles1, (None, idx)), r)
    return fx.Int32(fx.Vector(fx.memref_load_vec(r))[0])
