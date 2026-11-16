# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
# From aiter 7cd70aca8 aiter/ops/flydsl/kernels/moe_2stage_a16wmix/a8w4.py.
"""W channel exponents: per output channel (row of W), the largest e8m0 scale code over K,
read from the preshuffled scales. ``prepare_weights`` (``weights.py``) returns them as
``w*_exp`` and stores the fd2 group scales relative to them."""

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import gpu, range_constexpr

from pyhip.contrib.flydsl.host_launch import flydsl_kernel

from ._mem import _pointer


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


@flydsl_kernel
def channel_exp_kernel(*, rows, cols):
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
    channel_exp_kernel(scales, result, rows=experts * n, cols=kp * 2)
    return result
