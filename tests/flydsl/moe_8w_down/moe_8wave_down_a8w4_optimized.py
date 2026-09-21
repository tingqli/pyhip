# SPDX-License-Identifier: MIT
"""M256 eight-wave A8W4: routed DPP, cache-policy stores and early packing.

Keep the original four-slot / staggered 4+4-wave protocol and 256-worker
global queue. Reuse the four-wave Tensor DMA and bit-exact DPP helpers, not
its kernel or paired publication. Inputs use native sort256, GUI FP4/E8M0;
output stays BF16 [tokens, topk, N]. Counter is caller-owned and reset once.
"""

from functools import cache

import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import range_constexpr, rocdl
from flydsl.expr.arith import _to_raw
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled, ptr_arg
from pyhip.contrib.flydsl.moe_gemm_2stage.common import torch_tensor_to_pointer as _ptr
from moe_4wave_down_a8w4 import copy_async_lds, _tensor, _claim, _coalesce


@cache
def flydsl_moe_gemm_8wave_down_a8w4_optimized(*, n, k, topk, num_experts,
        block_n=128, num_oc_splits=4, output_cache_policy=18,
        coalesce_output=True, compute_overlap=True):
    """Explicit options retain one-factor controls; original dispatch is untouched.

    BN64 requires compute_overlap=False. Per-accumulator K order, scale opsel,
    route multiplication and BF16 rounding never change. No packed ABI,
    self-reset, sharding, wave-count conversion or compile-time source rewriting.
    """
    if k not in (128, 256, 384, 512) or block_n not in (64, 128):
        raise ValueError('require K128/256/384/512 and BN64/128')
    if num_oc_splits not in (1, 2, 4, 8) or n <= 0 or n % num_oc_splits:
        raise ValueError('N must divide into OC1/2/4/8')
    n_split = n // num_oc_splits
    packets = n_split // block_n
    if n_split % 128 or n_split % block_n or packets < 3:
        raise ValueError('each OC split must be 128-aligned with at least three packets')
    if compute_overlap and block_n != 128:
        raise ValueError('N64 early packing requires BN128')
    if not 0 < topk <= min(num_experts, 127) or output_cache_policy not in (0, 18):
        raise ValueError('require 0<TOPK<=min(E,127), output aux0/18')
    block_m, threads, workers, mm = 256, 512, 256, 2
    nn, kb_count = block_n // 16, k // 128
    kp_count = (kb_count + 1) // 2
    scale_cols, scale_stride = kp_count * 8, kp_count * 64
    tile_words = block_n * k // 8
    scale_words = block_n // 32 * scale_stride
    lds_bytes = tile_words * 16 + scale_words * 16 + block_m * 8
    if lds_bytes > 160 * 1024:
        raise ValueError('LDS budget exceeded')

    @fx.struct
    class Storage:
        b: fx.Array[fx.Int32, tile_words * 4, 16]
        scales: fx.Array[fx.Int32, scale_words * 4, 16]
        routes: fx.Array[fx.Float32, block_m, 16]
        ids: fx.Array[fx.Int32, block_m, 16]

    @flyc.kernel(known_block_size=[threads, 1, 1])
    def moe_8wave_down_a8w4_optimized(output: fx.Pointer, input_q: fx.Pointer, weight: fx.Pointer,
            input_scales: fx.Pointer, weight_scales: fx.Pointer, ids_arg: fx.Pointer,
            routes_arg: fx.Pointer, experts_arg: fx.Pointer, valid_arg: fx.Pointer,
            counter_arg: fx.Pointer, tokens: fx.Int32, capacity: fx.Int32):
        tid = fx.thread_idx.x
        wave, lane = tid // 64, tid % 64
        lr, lk = lane % 16, lane // 16
        lds = fx.SharedAllocator().allocate(Storage).peek()
        # Queue publication reuses retired B storage, as in the original kernel.
        ticket = lds.b.ptr
        ids = fx.make_view(lds.ids.ptr, fx.make_layout(block_m, 1))
        routes = fx.make_view(lds.routes.ptr, fx.make_layout(block_m, 1))
        b_ring = fx.make_view(lds.b.ptr, fx.make_ordered_layout((4, 16, 4, kb_count, nn, 4), 0))
        bs_ring = fx.make_view(lds.scales.ptr, fx.make_ordered_layout((64, kp_count, nn // 2, 4), 0))
        rows = tokens * topk
        a_data = _tensor(input_q, (16, k // 16, rows), fx.Int32)
        c_data = rocdl.make_buffer_tensor(_tensor(output, (8, n // 8, rows), fx.Int32), False)
        id_data = _tensor(ids_arg, (capacity * block_m,), fx.Int32)
        route_data = _tensor(routes_arg, (capacity * block_m,), fx.Float32)
        expert_data = _tensor(experts_arg, (capacity,), fx.Int32)
        valid_data = _tensor(valid_arg, (1,), fx.Int32)
        counter = _tensor(counter_arg, (1,), fx.Int32)
        b_data = _tensor(weight, (num_experts * n * k // 2,), fx.Int32)
        bs_data = _tensor(weight_scales, (num_experts * n * scale_cols,), fx.Int32)
        as_storage = _tensor(input_scales, (capacity * block_m * scale_cols,), fx.Int32)
        as_data = fx.make_view(fx.get_iter(as_storage), fx.make_layout(
            (64, kp_count, block_m // 32, capacity), (1, 64, scale_stride, block_m // 32 * scale_stride)))
        a = fx.make_rmem_tensor((8, mm, kb_count), fx.Int32)
        aw = fx.make_view(fx.get_iter(a), fx.make_ordered_layout((4, 2, mm, kb_count), 0))
        b = fx.make_rmem_tensor((4, nn, kb_count), fx.Int32)
        sa = fx.make_rmem_tensor((mm, kp_count), fx.Int32)
        sb = fx.make_rmem_tensor((nn // 2, kp_count), fx.Int32)
        acc = fx.make_rmem_tensor((4, mm, nn), fx.Float32)
        packed = fx.make_rmem_tensor((8, mm, nn // 2), fx.BFloat16)
        pair_type = ir.Type.parse('!llvm.struct<(i32, i32)>')
        limit = valid_data[0]
        running = fx.Boolean(True)
        while running:
            if tid == 0:
                ticket[0] = _claim(counter)
            fx.barrier()
            task = ticket[0]
            bm, oc = task // num_oc_splits, task % num_oc_splits
            running = bm * block_m < limit
            if running:
                expert = expert_data[bm]
                b_buffer, bs_buffer = rocdl.make_buffer_tensor(b_data, False), rocdl.make_buffer_tensor(bs_data, False)
                isrc = fx.make_view(fx.get_iter(id_data) + bm * block_m, fx.make_layout(block_m, 1))
                rsrc = fx.make_view(fx.get_iter(route_data) + bm * block_m, fx.make_layout(block_m, 1))
                copy_async_lds(rocdl.make_buffer_tensor(isrc, False), ids, block_m, threads)
                copy_async_lds(rocdl.make_buffer_tensor(rsrc, False), routes, block_m, threads)
                rocdl.asyncmark()
                rocdl.wait_asyncmark(0)
                fx.barrier()
                input_rows, live, weights, output_rows, coalesced_rows = {}, {}, {}, {}, {}
                for mi in range_constexpr(mm):
                    row = wave * 32 + mi * 16 + lr
                    encoded = ids[row]
                    token, slot = encoded & 0xFFFFFF, encoded >> 24
                    input_rows[mi] = token * topk + slot
                    live[mi] = (token < tokens) & (slot < topk) & (slot >= 0)
                    weights[mi] = routes[row]
                    output_rows[mi] = live[mi].select(input_rows[mi], rows)
                    if fx.const_expr(coalesce_output):
                        for parity in range_constexpr(2):
                            coalesced_rows[mi, parity] = fx.Int32(llvm.call_intrinsic(
                                fx.Int32.ir_type, 'llvm.amdgcn.mov.dpp.i32', [output_rows[mi].ir_value(),
                                fx.Int32(0xa0 + parity * 0x55).ir_value(), fx.Int32(15).ir_value(),
                                fx.Int32(15).ir_value(), fx.Boolean(False).ir_value()], [], []))
                a.fill(0)
                for mi in range_constexpr(mm):
                    if live[mi]:
                        for kb in range_constexpr(kb_count):
                            for part in range_constexpr(2):
                                aw[None, part, mi, kb] = a_data[None, lk + kb * 8 + part * 4, input_rows[mi]].load()
                for mi in range_constexpr(mm):
                    m16 = wave * mm + mi
                    for kp in range_constexpr(kp_count):
                        sa[mi, kp] = as_data[lane, kp, m16 // 2, bm] >> ((m16 % 2) * 8)

                def load_b(q):
                    offset = expert * (n * k // 8) + oc * (n_split * k // 8) + q * tile_words
                    src = fx.make_view(fx.get_iter(b_buffer) + offset, fx.make_layout(tile_words, 1))
                    copy_async_lds(src, b_ring[None, None, None, None, None, q % 4], tile_words, threads)
                    offset = (expert * (n // 32) + (oc * n_split + q * block_n) // 32) * scale_stride
                    src = fx.make_view(fx.get_iter(bs_buffer) + offset, fx.make_layout(scale_words, 1))
                    copy_async_lds(src, bs_ring[None, None, None, q % 4], scale_words, threads)

                def read_b(q):
                    for kb in range_constexpr(kb_count):
                        for ni in range_constexpr(nn):
                            b[None, ni, kb] = b_ring[None, lr, lk, kb, ni, q % 4].load()
                    for kp in range_constexpr(kp_count):
                        for np in range_constexpr(nn // 2):
                            sb[np, kp] = bs_ring[lane, kp, np, q % 4]

                def mfma(mi, ni, kb):
                    zero = fx.Int32(0)
                    b4 = b[None, ni, kb].load()
                    bv = fx.Vector.from_elements([b4[0], b4[1], b4[2], b4[3], zero, zero, zero, zero], fx.Int32)
                    value = rocdl.mfma_scale_f32_16x16x128_f8f6f4(ir.VectorType.get([4], fx.Float32.ir_type), [
                        bv.ir_value(), a[None, mi, kb].load().ir_value(), acc[None, mi, ni].load().ir_value(), 4, 0,
                        (kb % 2) * 2 + ni % 2, sb[ni // 2, kb // 2].ir_value(),
                        (kb % 2) * 2, sa[mi, kb // 2].ir_value()])
                    acc[None, mi, ni].store(value)
                    rocdl.sched_barrier(0)

                def pack_pair(mi, ni):
                    c0 = acc[None, mi, ni].load() * weights[mi]
                    c1 = acc[None, mi, ni + 1].load() * weights[mi]
                    d0, d1 = rocdl.cvt_pk_bf16_f32(c0[0], c0[1]), rocdl.cvt_pk_bf16_f32(c0[2], c0[3])
                    d2, d3 = rocdl.cvt_pk_bf16_f32(c1[0], c1[1]), rocdl.cvt_pk_bf16_f32(c1[2], c1[3])
                    lo = rocdl.permlane16_swap(pair_type, _to_raw(d0), _to_raw(d2), False, False)
                    hi = rocdl.permlane16_swap(pair_type, _to_raw(d1), _to_raw(d3), False, False)
                    values = fx.Vector.from_elements([
                        fx.Int32(llvm.extractvalue(fx.Int32.ir_type, lo, [0])), fx.Int32(llvm.extractvalue(fx.Int32.ir_type, hi, [0])),
                        fx.Int32(llvm.extractvalue(fx.Int32.ir_type, lo, [1])), fx.Int32(llvm.extractvalue(fx.Int32.ir_type, hi, [1]))], fx.Int32)
                    packed[None, mi, ni // 2].store(values.bitcast(fx.BFloat16))

                def compute():
                    acc.fill(0)
                    if fx.const_expr(compute_overlap):
                        for kb in range_constexpr(kb_count):
                            for mi in range_constexpr(mm):
                                for ni in range_constexpr(4):
                                    mfma(mi, ni, kb)
                        for kb in range_constexpr(kb_count):
                            for mi in range_constexpr(mm):
                                for ni in range_constexpr(4, 8):
                                    mfma(mi, ni, kb)
                                    if fx.const_expr(kb == 0 and ni % 2 == 1):
                                        pack_pair(mi, ni - 5)
                        for mi in range_constexpr(mm):
                            for ni in range_constexpr(4, 8, 2):
                                pack_pair(mi, ni)
                    else:
                        for kb in range_constexpr(kb_count):
                            for mi in range_constexpr(mm):
                                for ni in range_constexpr(nn):
                                    mfma(mi, ni, kb)
                        for mi in range_constexpr(mm):
                            for ni in range_constexpr(0, nn, 2):
                                pack_pair(mi, ni)
                    if fx.const_expr(coalesce_output):
                        for mi in range_constexpr(mm):
                            for pair in range_constexpr(0, nn // 2, 2):
                                parts = _coalesce(packed[None, mi, pair].load().bitcast(fx.Int32),
                                                  packed[None, mi, pair + 1].load().bitcast(fx.Int32))
                                packed[None, mi, pair].store(parts[0].bitcast(fx.BFloat16))
                                packed[None, mi, pair + 1].store(parts[1].bitcast(fx.BFloat16))

                def store(q):
                    n_base = oc * n_split + q * block_n
                    swap = (lk & 1) * 2 + (lk >> 1)
                    for mi in range_constexpr(mm):
                        for ni in range_constexpr(0, nn, 2):
                            row = output_rows[mi]
                            col = n_base // 8 + ni * 2 + swap
                            if fx.const_expr(coalesce_output):
                                row = coalesced_rows[mi, (ni // 2) % 2]
                                col = n_base // 8 + (ni // 4) * 8 + swap + (lr % 2) * 4
                            fragment = fx.make_rmem_tensor(4, fx.Int32)
                            fragment.store(packed[None, mi, ni // 2].load().bitcast(fx.Int32))
                            atom = fx.make_copy_atom(rocdl.BufferCopy128b(cache_modifier=output_cache_policy), fx.Int32)
                            fx.copy(atom, fragment, c_data[None, col, row])

                load_b(fx.Int32(0))
                rocdl.asyncmark()
                load_b(fx.Int32(1))
                rocdl.asyncmark()
                rocdl.wait_asyncmark(1)
                # Preserve both ends of the original staggered barrier protocol.
                if wave >= 4:
                    fx.barrier()
                fx.barrier()
                read_b(fx.Int32(0))
                load_b(fx.Int32(2))
                rocdl.asyncmark()
                rocdl.wait_asyncmark(1)
                rocdl.s_waitcnt(lgkmcnt=0)
                fx.barrier()
                compute()
                for q in range(fx.Int32(0), fx.Int32(packets - 3), fx.Int32(1)):
                    fx.barrier()
                    rocdl.s_setprio(1)
                    read_b(q + 1)
                    store(q)
                    load_b(q + 3)
                    rocdl.asyncmark()
                    rocdl.wait_asyncmark(1)
                    rocdl.s_setprio(0)
                    rocdl.s_waitcnt(lgkmcnt=0)
                    fx.barrier()
                    compute()
                fx.barrier()
                read_b(packets - 2)
                store(packets - 3)
                rocdl.wait_asyncmark(0)
                fx.barrier()
                compute()
                fx.barrier()
                read_b(packets - 1)
                store(packets - 2)
                compute()
                store(packets - 1)
                if wave < 4:
                    fx.barrier()

    @flyc.jit
    def launch(output: fx.Pointer, input_q: fx.Pointer, weight: fx.Pointer, sa: fx.Pointer, sb: fx.Pointer,
               ids: fx.Pointer, routes: fx.Pointer, experts: fx.Pointer, valid: fx.Pointer, counter: fx.Pointer,
               tokens: fx.Int32, capacity: fx.Int32, stream: fx.Stream):
        moe_8wave_down_a8w4_optimized(output, input_q, weight, sa, sb, ids, routes, experts, valid, counter, tokens, capacity,
            value_attrs={'rocdl.waves_per_eu': 2, 'rocdl.flat_work_group_size': '512,512',
                         'passthrough': [['target-features', '-packed-fp32-ops']]},
        ).launch(grid=(workers, 1, 1), block=(threads, 1, 1), stream=stream)

    def down(output, input_q, weight, sa, sb, ids, routes, experts, valid, counter):
        tokens = input_q.shape[0]
        assert input_q.shape == (tokens, topk, k) and output.shape == (tokens, topk, n)
        assert 0 < tokens < (1 << 24) and output.dtype == torch.bfloat16
        assert input_q.dtype == torch.float8_e4m3fn and weight.shape == (num_experts, n, k // 2)
        assert weight.element_size() == sa.element_size() == sb.element_size() == 1
        assert ids.shape == routes.shape and (ids.numel() + block_m - 1) // block_m == experts.numel()
        # mxfp4_moe_sort_fwd pads allocation rows to 32, not the sorting BM.
        # Only complete valid M256 blocks are consumed.
        assert sa.numel() == ((ids.numel() + 31) // 32 * 32) * scale_cols
        assert sb.numel() == num_experts * n * scale_cols
        assert ids.dtype == experts.dtype == valid.dtype == counter.dtype == torch.int32
        assert routes.dtype == torch.float32 and valid.numel() >= 1 and counter.numel() == 1
        for tensor in (output, input_q, weight, sa, sb, ids, routes, experts, valid, counter):
            assert tensor.is_cuda and tensor.device == output.device and tensor.is_contiguous()
            assert tensor.numel() * tensor.element_size() < (1 << 32), '32-bit buffer range exceeded'
        assert torch.cuda.get_device_properties(output.device).gcnArchName.startswith('gfx950')
        counter.zero_()
        _run_compiled(launch, _ptr(output), _ptr(input_q), ptr_arg(weight), ptr_arg(sa), ptr_arg(sb),
                      _ptr(ids), _ptr(routes), _ptr(experts), _ptr(valid), _ptr(counter),
                      fx.Int32(tokens), fx.Int32(experts.numel()), fx.Stream(torch.cuda.current_stream(output.device).cuda_stream))
        return output

    down.launch = launch
    down.config = dict(block_m=block_m, sort_block_m=block_m, block_n=block_n, num_waves=8,
        num_oc_splits=num_oc_splits, workers=workers, counter_elements=1, sharded=False,
        output_layout='routed', row_store_bytes=128 if coalesce_output else 64,
        output_cache_policy=output_cache_policy, compute_overlap=compute_overlap,
        coalesce_output=coalesce_output, lds_bytes=lds_bytes, publication='staggered4+4')
    return down