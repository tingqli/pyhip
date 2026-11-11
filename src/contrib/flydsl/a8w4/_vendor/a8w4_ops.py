# Vendored from aiter/ops/flydsl/kernels (aiter 7cd70aca8) so the a8w4 package
# does not depend on aiter's private FlyDSL helpers. Imports made relative.
"""fp8 packing and the W channel-exponent scan from moe_2stage_a16wmix/a8w4.py."""

from functools import cache

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from .tensor_shim import _run_compiled


def _pointer(address, dtype, alignment=16):
    return fx.inttoptr(
        fx.PointerType.get(dtype.ir_type, fx.AddressSpace.Global, alignment),
        fx.Int64(address),
    )


def _scale_index(row, group, groups):
    # shuffle_scale: [rows/32, groups/8, 4, 16, 2, 2].
    return (
        (row // 32) * (groups * 32)
        + (group // 8) * 256
        + (group % 4) * 64
        + (row % 16) * 4
        + ((group // 4) % 2) * 2
        + (row % 32) // 16
    )


def _pack_activation(a, b, c, d):
    values = [fx.max(fx.min(v, 240.0), -240.0) for v in (a, b, c, d)]
    lo = rocdl.cvt_pk_fp8_f32(T.i32, values[0], values[1], fx.Int32(0), False)
    packed = fx.Uint32(rocdl.cvt_pk_fp8_f32(T.i32, values[2], values[3], lo, True))
    mag = packed & fx.Uint32(0x7F7F7F7F)
    zero = (~(mag + fx.Uint32(0x7F7F7F7F))) & fx.Uint32(0x80808080)
    return (packed & ~zero).bitcast(fx.Int32)


@cache
def _compile_channel_exponents(rows, cols):
    groups = cols // 32
    rows_per_wave = 4 if cols >= 4096 else 16
    partitions = 64 // rows_per_wave
    if groups % partitions:
        raise ValueError("unsupported scale group geometry")
    rounds = groups // partitions
    rows_per_block = 4 * rows_per_wave

    @flyc.kernel(known_block_size=(256, 1, 1))
    def kernel(scales: fx.Int64, output: fx.Int64):
        tid = fx.Int32(gpu.thread_id("x"))
        row = fx.Int32(gpu.block_id("x")) * rows_per_block + (tid // 64) * rows_per_wave + tid % rows_per_wave
        part = (tid % 64) // rows_per_wave
        if row < rows:
            largest = fx.Int32(0)
            for j in range(rounds):
                group = part + j * partitions
                code = fx.ptr_load(fx.add_offset(_pointer(scales, fx.Uint8, 1),
                                                 fx.Int64(_scale_index(row, group, groups))))
                largest = fx.max(largest, code.to(fx.Int32) & 255)
            for bit in range_constexpr(partitions.bit_length() - 1):
                largest = fx.max(largest, largest.shuffle_xor(rows_per_wave << bit, 64))
            if part == 0:
                fx.ptr_store(largest, fx.add_offset(_pointer(output, fx.Int32, 4), row))

    @flyc.jit
    def launch(scales: fx.Int64, output: fx.Int64, stream: fx.Stream):
        kernel(scales, output).launch(grid=((rows + rows_per_block - 1) // rows_per_block,), block=(256,), stream=stream)

    return launch


def channel_exponents(packed, scales):
    experts, n, kp = packed.shape
    result = torch.empty((experts, n), device=packed.device, dtype=torch.int32)
    _run_compiled(_compile_channel_exponents(experts * n, kp * 2), scales.data_ptr(), result.data_ptr(),
         torch.cuda.current_stream())
    return result
