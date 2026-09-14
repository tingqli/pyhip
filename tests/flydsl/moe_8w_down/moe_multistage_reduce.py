# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Selected packed BF16 TOPK sum: 256 threads, 2048 columns, NT reads."""

from functools import cache

import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl.expr import range_constexpr, rocdl
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

from pyhip.contrib.flydsl.moe_gemm_2stage.common import torch_tensor_to_pointer as _ptr
from moe_multistage_down import _scalar


@cache
def make_moe_sum(*, n, topk, sort_block_m=256):
    """Gather packed down routes through inverse; missing entries contribute0."""
    assert n > 0 and n % 512 == 0 and 0 < topk <= 255
    assert sort_block_m in (128, 256)
    assert sort_block_m * n * 2 < (1 << 32), "one packed M block must fit a buffer descriptor"
    column_blocks = (n + 2047) // 2048

    @flyc.kernel(known_block_size=[256, 1, 1])
    def moe_down_sum_kernel(output: fx.Pointer, source: fx.Pointer, inverse: fx.Pointer,
                            tokens: fx.Int32, source_rows: fx.Int32):
        tid = fx.Int32(fx.thread_idx.x)
        token = fx.Int32(fx.block_idx.x) // column_blocks
        block_col = fx.Int32(fx.block_idx.x) % column_blocks
        target_buffer = fx.rocdl.make_buffer_tensor(fx.make_view(output, fx.make_layout(tokens * n, 1)), False)
        drsrc = fx.rocdl.get_buffer_rsrc(fx.get_iter(target_buffer))
        zero = ir.IntegerAttr.get(fx.Int32.ir_type, 0)
        zero_offset = fx.Int32(0).ir_value()
        vector_type = ir.VectorType.get([4], fx.Int32.ir_type)
        read_aux = ir.IntegerAttr.get(fx.Int32.ir_type, 2)
        locations = [_scalar(inverse[token * topk + route]) for route in range_constexpr(topk)]

        def read_route(location, column):
            valid_row = (location >= 0) & (location < source_rows)
            safe_location = valid_row.select(location, fx.Int32(0))
            bm, row = safe_location // sort_block_m, safe_location % sort_block_m
            # A route's M block may start above 4 GiB; only the block-local
            # byte offset is 32-bit. Invalid inverse entries use a safe base.
            block = fx.make_view(source + fx.Int64(bm) * (sort_block_m * n),
                                 fx.make_layout(sort_block_m * n, 1))
            source_buffer = fx.rocdl.make_buffer_tensor(block, False)
            srsrc = fx.rocdl.get_buffer_rsrc(fx.get_iter(source_buffer))
            # OC partitions columns; it does not change the global N64 layout.
            element = column // 64 * (sort_block_m * 64) + row * 64 + column % 64
            valid = valid_row & (column < n)
            offset = valid.select(element * 2, fx.Int32(-1))
            return fx.Vector(rocdl.raw_ptr_buffer_load(
                vector_type, srsrc, offset.ir_value(), zero_offset, aux=read_aux,
            )).bitcast(fx.BFloat16)

        column = block_col * 2048 + tid * 8
        fragments = [read_route(locations[route], column) for route in range_constexpr(topk)]
        accum = fragments[0].to(fx.Float32)
        for route in range_constexpr(1, topk):
            accum = accum + fragments[route].to(fx.Float32)
        offset = (column < n).select((token * n + column) * 2, fx.Int32(-1))
        rocdl.raw_ptr_buffer_store(accum.to(fx.BFloat16).bitcast(fx.Int32).ir_value(),
                                  drsrc, offset.ir_value(), zero_offset, aux=zero)

    @flyc.jit
    def launch(output: fx.Pointer, source: fx.Pointer, inverse: fx.Pointer,
               tokens: fx.Int32, source_rows: fx.Int32, stream: fx.Stream):
        moe_down_sum_kernel(output, source, inverse, tokens, source_rows).launch(
            grid=(tokens * column_blocks, 1, 1), block=(256, 1, 1), stream=stream,
        )

    def reduce(output, source, inverse):
        assert output.ndim == 2 and output.shape[1] == n
        tokens = output.shape[0]
        assert source.ndim == 2 and source.shape[1] == n and source.shape[0] % sort_block_m == 0
        assert inverse.shape == (tokens, topk)
        assert source.dtype == output.dtype == torch.bfloat16 and inverse.dtype == torch.int32
        assert all(t.is_cuda and t.is_contiguous() and t.device == output.device for t in (output, source, inverse))
        assert source.shape[0] < (1 << 31) and output.numel() * 2 < (1 << 32)
        stream = torch.cuda.current_stream(output.device)
        compiled = getattr(launch, "_cf", None)
        if compiled is None:
            _run_compiled(launch, _ptr(output), _ptr(source), _ptr(inverse), fx.Int32(tokens),
                          fx.Int32(source.shape[0]), fx.Stream(stream.cuda_stream))
        else:
            compiled(output.data_ptr(), source.data_ptr(), inverse.data_ptr(), tokens, source.shape[0], stream.cuda_stream)
        return output

    reduce.config = {"output_layout": "packed", "sort_block_m": sort_block_m, "num_threads": 256, "block_cols": 2048,
                     "read_policy": 2, "write_policy": 0}
    return reduce