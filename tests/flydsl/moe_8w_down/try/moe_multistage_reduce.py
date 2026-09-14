# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Gather routed BF16 down results and sum TOPK without restoring the tensor.

Like moe_gemm_2stage.moe_reduce.sorted_sum, each block owns a token and a
contiguous column range, loads all route locations, accumulates in FP32,
and rounds once to BF16. Address decoding also supports the experimental
packed/linear output formats. Missing inverse entries (-1) contribute zero.
"""

from functools import cache

import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl.expr import const_expr, range_constexpr, rocdl
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

from pyhip.contrib.flydsl.moe_gemm_2stage.common import torch_tensor_to_pointer as _ptr
from moe_multistage_down import _scalar


@cache
def make_moe_sum(*, n, topk, num_oc_splits=4, output_layout="sorted",
                 num_threads=64, block_cols=1024, preload=True,
                 read_policy=0, write_policy=0, tree_sum=False, packed_rows=256):
    assert n > 0 and num_oc_splits > 0 and n % (128 * num_oc_splits) == 0
    assert 0 < topk <= 255
    assert output_layout in ("routed", "sorted", "packed", "linear", "linear_raw", "linear16", "packed_nmajor", "packed128")
    assert num_threads in (64, 128, 256)
    assert block_cols > 0 and block_cols % (num_threads * 8) == 0
    assert read_policy in (0, 2, 16, 18) and write_policy in (0, 2)
    assert packed_rows >= 256 and (packed_rows == 256 or output_layout == "packed")
    n_split = n // num_oc_splits
    column_blocks = (n + block_cols - 1) // block_cols

    @flyc.kernel(known_block_size=[num_threads, 1, 1])
    def moe_down_sum_kernel(output: fx.Pointer, source: fx.Pointer, inverse: fx.Pointer,
                            tokens: fx.Int32, source_rows: fx.Int32):
        tid = fx.Int32(fx.thread_idx.x)
        token = fx.Int32(fx.block_idx.x) // column_blocks
        block_col = fx.Int32(fx.block_idx.x) % column_blocks
        source_buffer = fx.rocdl.make_buffer_tensor(
            fx.make_view(source, fx.make_layout(source_rows * n, 1)), False,
        )
        target_buffer = fx.rocdl.make_buffer_tensor(
            fx.make_view(output, fx.make_layout(tokens * n, 1)), False,
        )
        srsrc = fx.rocdl.get_buffer_rsrc(fx.get_iter(source_buffer))
        drsrc = fx.rocdl.get_buffer_rsrc(fx.get_iter(target_buffer))
        zero = fx.Int32(0).ir_value()
        vector_type = ir.VectorType.get([4], fx.Int32.ir_type)
        pair_type = ir.VectorType.get([2], fx.Int32.ir_type)
        read_aux = ir.IntegerAttr.get(fx.Int32.ir_type, read_policy)
        write_aux = ir.IntegerAttr.get(fx.Int32.ir_type, write_policy)
        locations = [token * topk + route if output_layout == "routed" else _scalar(inverse[token * topk + route])
                     for route in range_constexpr(topk)]

        def read_route(location, column):
            bm, row = location // 256, location % 256
            oc, q = column // n_split, column % n_split // 64
            local_column = column % 64
            local_row, wave = row % 32, row // 32
            record = local_row % 2 * 2 + local_row // 16
            row_lane = local_row % 16 // 2 * 2 + local_column // 32 + local_column % 32 // 16 * 16
            lane = row_lane + local_column % 16 // 8 * 32
            element = (location * n + column if output_layout in ("routed", "sorted") else
                       bm * (packed_rows * n) + oc * (packed_rows * n_split) + q * (packed_rows * 64) + row * 64 + local_column if output_layout == "packed" else
                       bm * (256 * n) + oc * (256 * n_split) + q * (256 * 64) + wave * 2048 + (local_column // 16) * 512 + local_row * 8 + (local_column % 16 // 8) * 4 if output_layout == "linear_raw" else
                       bm * (256 * n) + oc * (256 * n_split) + q * (256 * 64) + wave * 2048 + record * 512 + lane * 8)
            if const_expr(output_layout == "linear16"):
                packet = local_row // 16 * 2 + local_column // 32
                raw_lane = (local_column % 16 // 4) * 16 + local_row % 16
                element = bm * (256 * n) + oc * (256 * n_split) + q * (256 * 64) + wave * 2048 + packet * 512 + raw_lane * 8 + local_column % 32 // 16 * 4
            if const_expr(output_layout == "packed_nmajor"):
                element = (column // 64 * source_rows + location) * 64 + local_column
            if const_expr(output_layout == "packed128"):
                element = bm * (256 * n) + column // 128 * (256 * 128) + row * 128 + column % 128
            logical_rows = source_rows if packed_rows == 256 else source_rows // packed_rows * 256
            valid = (location >= 0) & (location < logical_rows) & (column < n)
            offset = valid.select(element * 2, fx.Int32(-1))
            if const_expr(output_layout in ("linear_raw", "linear16")):
                left = fx.Vector(rocdl.raw_ptr_buffer_load(pair_type, srsrc, offset.ir_value(), zero, aux=read_aux))
                right_offset = valid.select((element + (128 if output_layout == "linear16" else 256)) * 2, fx.Int32(-1))
                right = fx.Vector(rocdl.raw_ptr_buffer_load(pair_type, srsrc, right_offset.ir_value(), zero, aux=read_aux))
                result = fx.Vector.from_elements([left[0], left[1], right[0], right[1]], fx.Int32)
            else:
                result = fx.Vector(rocdl.raw_ptr_buffer_load(vector_type, srsrc, offset.ir_value(), zero, aux=read_aux))
            return result.bitcast(fx.BFloat16)

        for turn in range_constexpr(block_cols // (num_threads * 8)):
            column = block_col * block_cols + tid * 8 + turn * num_threads * 8
            if const_expr(tree_sum):
                values = [read_route(locations[route], column).to(fx.Float32) for route in range_constexpr(topk)]
                while const_expr(len(values) > 1):
                    values = [values[index] + values[index + 1] if index + 1 < len(values) else values[index]
                              for index in range_constexpr(0, len(values), 2)]
                accum = values[0]
            elif const_expr(preload):
                fragments = [read_route(locations[route], column) for route in range_constexpr(topk)]
                accum = fragments[0].to(fx.Float32)
                for route in range_constexpr(1, topk):
                    accum = accum + fragments[route].to(fx.Float32)
            else:
                accum = read_route(locations[0], column).to(fx.Float32)
                for route in range_constexpr(1, topk):
                    accum = accum + read_route(locations[route], column).to(fx.Float32)
            offset = (column < n).select((token * n + column) * 2, fx.Int32(-1))
            rocdl.raw_ptr_buffer_store(accum.to(fx.BFloat16).bitcast(fx.Int32).ir_value(),
                                      drsrc, offset.ir_value(), zero, aux=write_aux)

    @flyc.jit
    def launch(output: fx.Pointer, source: fx.Pointer, inverse: fx.Pointer,
               tokens: fx.Int32, source_rows: fx.Int32, stream: fx.Stream):
        moe_down_sum_kernel(output, source, inverse, tokens, source_rows).launch(
            grid=(tokens * column_blocks, 1, 1), block=(num_threads, 1, 1), stream=stream,
        )

    def reduce(output: torch.Tensor, source: torch.Tensor, inverse: torch.Tensor):
        assert output.ndim == 2 and output.shape[1] == n
        tokens = output.shape[0]
        assert source.numel() % n == 0 and inverse.numel() == tokens * topk
        assert inverse.shape == (tokens, topk)
        if output_layout == "routed":
            assert source.shape == (tokens, topk, n)
        else:
            assert source.ndim == 2 and source.shape[1] == n and source.shape[0] % packed_rows == 0
        assert source.dtype == output.dtype == torch.bfloat16 and inverse.dtype == torch.int32
        assert all(t.is_cuda and t.is_contiguous() and t.device == output.device for t in (output, source, inverse))
        assert source.numel() * 2 < (1 << 32) and output.numel() * 2 < (1 << 32)
        stream = torch.cuda.current_stream(output.device)
        compiled = getattr(launch, "_cf", None)
        if compiled is None:
            _run_compiled(launch, _ptr(output), _ptr(source), _ptr(inverse),
                          fx.Int32(tokens), fx.Int32(source.numel() // n), fx.Stream(stream.cuda_stream))
        else:
            compiled(output.data_ptr(), source.data_ptr(), inverse.data_ptr(), tokens,
                     source.numel() // n, stream.cuda_stream)
        return output

    reduce.config = {"output_layout": output_layout, "num_threads": num_threads,
                     "block_cols": block_cols, "preload": preload,
                     "read_policy": read_policy, "write_policy": write_policy, "tree_sum": tree_sum}
    return reduce