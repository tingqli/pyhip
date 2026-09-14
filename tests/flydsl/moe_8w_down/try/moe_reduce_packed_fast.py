# SPDX-License-Identifier: MIT
"""Exact BF16 packed TOPK reduction with unsigned, OC-independent addressing."""

from functools import cache
import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import range_constexpr, rocdl
from flydsl._mlir import ir
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled
from pyhip.contrib.flydsl.moe_gemm_2stage.common import torch_tensor_to_pointer as _ptr
from moe_multistage_down import _scalar


@cache
def make_fast_sum(*, n, topk, num_threads=256, block_cols=2048, read_policy=2, **unused):
    assert n > 0 and n % 64 == 0 and topk > 0
    assert block_cols % (num_threads * 8) == 0

    @flyc.kernel(known_block_size=[num_threads, 1, 1])
    def packed_sum_fast_kernel(output: fx.Pointer, source: fx.Pointer, inverse: fx.Pointer,
                               tokens: fx.Int32, source_rows: fx.Int32):
        tid = fx.Uint32(fx.thread_idx.x)
        token = fx.Uint32(fx.block_idx.x)
        column_block = fx.Uint32(fx.block_idx.y)
        sb = fx.rocdl.make_buffer_tensor(fx.make_view(source, fx.make_layout(source_rows * n, 1)), False)
        db = fx.rocdl.make_buffer_tensor(fx.make_view(output, fx.make_layout(tokens * n, 1)), False)
        sr = fx.rocdl.get_buffer_rsrc(fx.get_iter(sb))
        dr = fx.rocdl.get_buffer_rsrc(fx.get_iter(db))
        zero = fx.Int32(0).ir_value()
        vec4 = ir.VectorType.get([4], fx.Int32.ir_type)
        locations = [fx.Uint32(_scalar(inverse[token * topk + i])) for i in range_constexpr(topk)]
        bases = [((loc >> 8) * (256 * n) + (loc & 255) * 64) * 2 for loc in locations]
        for turn in range_constexpr(block_cols // (num_threads * 8)):
            col = column_block * block_cols + tid * 8 + turn * num_threads * 8
            col_offset = ((col >> 6) * 16384 + (col & 63)) * 2
            values = []
            for i in range_constexpr(topk):
                good = (locations[i] < fx.Uint32(source_rows)) & (col < n)
                off = good.select(bases[i] + col_offset, fx.Uint32(0xFFFFFFFF))
                values.append(fx.Vector(rocdl.raw_ptr_buffer_load(
                    vec4, sr, off.ir_value(), zero, aux=ir.IntegerAttr.get(fx.Int32.ir_type, read_policy),
                )).bitcast(fx.BFloat16))
            accum = values[0].to(fx.Float32)
            for i in range_constexpr(1, topk):
                accum = accum + values[i].to(fx.Float32)
            off = (col < n).select((token * n + col) * 2, fx.Uint32(0xFFFFFFFF))
            rocdl.raw_ptr_buffer_store(accum.to(fx.BFloat16).bitcast(fx.Int32).ir_value(), dr, off.ir_value(), zero)

    @flyc.jit
    def launch(output: fx.Pointer, source: fx.Pointer, inverse: fx.Pointer,
               tokens: fx.Int32, source_rows: fx.Int32, stream: fx.Stream):
        packed_sum_fast_kernel(output, source, inverse, tokens, source_rows).launch(
            grid=(tokens, (n + block_cols - 1) // block_cols, 1), block=(num_threads, 1, 1), stream=stream)

    def reduce(output, source, inverse):
        assert output.shape == (inverse.shape[0], n) and inverse.shape[1] == topk
        assert source.ndim == 2 and source.shape[1] == n and source.shape[0] % 256 == 0
        assert output.dtype == source.dtype == torch.bfloat16 and inverse.dtype == torch.int32
        assert all(t.is_cuda and t.is_contiguous() and t.device == output.device for t in (output, source, inverse))
        assert source.numel() * 2 < 2**32 and output.numel() * 2 < 2**32
        stream = torch.cuda.current_stream(output.device)
        compiled = getattr(launch, "_cf", None)
        if compiled is None:
            _run_compiled(launch, _ptr(output), _ptr(source), _ptr(inverse), fx.Int32(output.shape[0]), fx.Int32(source.shape[0]), fx.Stream(stream.cuda_stream))
        else:
            compiled(output.data_ptr(), source.data_ptr(), inverse.data_ptr(), output.shape[0], source.shape[0], stream.cuda_stream)
    return reduce