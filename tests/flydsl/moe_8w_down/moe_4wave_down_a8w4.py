# SPDX-License-Identifier: MIT
"""gfx950 A8W4 MoE down: M128, four waves, OC4, paired B ring, routed DPP.

A is FP8 [tokens, topk, K]; B is Aiter GUI-shuffled FP4 [E, N, K/2].
Both use shuffled per-1x32 E8M0 scales. Metadata must be natively sorted
with block_m=128. Output is weighted BF16 [tokens, topk, N], not packed.
The caller owns the counter and the final TOPK reduction. Each launch resets
the counter on the current stream; concurrent launches need separate buffers.
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


@flyc.jit
def copy_async_lds(src: fx.Tensor, dst: fx.Tensor, num_dwords: int, num_threads: int):
    """Copy contiguous 32-bit Tensor storage; caller owns asyncmark/wait/barrier.

    src retains its buffer descriptor through subviews. The LDS operand is a
    wave base, NOT a per-lane address: hardware adds lane * atom_bytes, even
    under partial EXEC. Low-lane masks handle the final vector and dword tails.
    """
    assert src.dtype.width == dst.dtype.width == 32
    assert num_dwords > 0 and num_threads > 0 and num_threads % 64 == 0
    tid = fx.thread_idx.x
    wave_base = fx.Int32(rocdl.readfirstlane(fx.Int32.ir_type, ((tid // 64) * 256).ir_value()))
    atom128 = fx.make_copy_atom(rocdl.cdna4.BufferLoadAsyncLDS128b(), 128)
    vectors, tail = num_dwords // 4, num_dwords % 4
    rounds, lanes = vectors // num_threads, vectors % num_threads
    for i in range_constexpr(rounds):
        source = fx.make_view(fx.get_iter(src) + i * num_threads * 4 + tid * 4, fx.make_layout(4, 1))
        target = fx.make_view(fx.get_iter(dst) + i * num_threads * 4 + wave_base, fx.make_layout(4, 1))
        fx.copy(atom128, source, target)
    if fx.const_expr(lanes != 0):
        if tid < lanes:
            source = fx.make_view(fx.get_iter(src) + rounds * num_threads * 4 + tid * 4, fx.make_layout(4, 1))
            target = fx.make_view(fx.get_iter(dst) + rounds * num_threads * 4 + wave_base, fx.make_layout(4, 1))
            fx.copy(atom128, source, target)
    if fx.const_expr(tail != 0):
        if tid < tail:
            atom32 = fx.make_copy_atom(rocdl.cdna4.BufferLoadAsyncLDS32b(), 32)
            source = fx.make_view(fx.get_iter(src) + vectors * 4 + tid, fx.make_layout(1, 1))
            target = fx.make_view(fx.get_iter(dst) + vectors * 4, fx.make_layout(1, 1))
            fx.copy(atom32, source, target)


def select_config(*, tokens, n, k, topk):
    """Small measured policy; unknown shapes keep cached stores/global queue.

    BN64 is chosen when BN128 exceeds half the 160 KiB LDS budget or cannot
    form paired packets. Cache/queue thresholds are limited to the measured
    N6144/TOPK8/K256,384 family, not asserted universal hardware thresholds.
    """
    if tokens <= 0 or n <= 0 or n % 512 or k not in (128, 256, 384, 512) or not 0 < topk < 128:
        raise ValueError("require tokens>0, N%512=0, K in 128/256/384/512, 0<TOPK<128")
    kp = (k // 128 + 1) // 2
    lds128 = 128 * k * 2 + 4 * kp * 64 * 16 + 128 * 8 + 4
    bn = 128 if lds128 <= 80 * 1024 and n % 1024 == 0 and n >= 2048 else 64
    if n // (4 * bn) < 4:
        raise ValueError("paired OC4 requires at least four packets per task")
    measured = n == 6144 and topk == 8 and k in (256, 384)
    return dict(block_n=bn, output_cache_policy=18 if measured and tokens >= 8192 else 0,
                sharded=measured and k == 256 and tokens >= 32768)


def _tensor(ptr, shape, dtype):
    # shape is in the incoming pointer's elements (E8M0 pointers are bytes).
    view = fx.make_view(ptr, fx.make_ordered_layout(shape, 0))
    target = fx.PointerType.get(dtype.ir_type, view.memspace, max(1, dtype.width // 8))
    return fx.make_view(fx.recast_iter(target, fx.get_iter(view)),
                        fx.recast_layout(view.layout, view.dtype.width, dtype.width))


def _claim(counter):
    return fx.Int32(llvm.AtomicRMWOp(llvm.AtomicBinOp.add, fx.to_llvm_ptr(fx.get_iter(counter)),
        fx.Int32(1).ir_value(), llvm.AtomicOrdering.monotonic, syncscope="agent", alignment=4).result)


def _coalesce(first, second):
    """Preserve the validated DPP schedule: exchange row.bit0 / BF16 col.bit5."""
    assembly = ["s_mov_b64 vcc, $16"]
    assembly.extend(f"v_cndmask_b32_dpp ${i}, ${12+i}, ${8+i}, vcc quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf"
                    for i in range(4))
    assembly.append("s_mov_b64 vcc, $17")
    assembly.extend(f"v_cndmask_b32_dpp ${4+i}, ${8+i}, ${12+i}, vcc quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf"
                    for i in range(4))
    result = llvm.inline_asm(
        ir.Type.parse("!llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32)>"),
        [first[i].ir_value() for i in range(4)] + [second[i].ir_value() for i in range(4)]
        + [fx.Uint64(0x5555555555555555).ir_value(), fx.Uint64(0xAAAAAAAAAAAAAAAA).ir_value()],
        "\n".join(assembly), ",".join(["=&v"] * 8 + ["v"] * 8 + ["s", "s", "~{vcc}"]),
        has_side_effects=True)
    parts = []
    for side in range(2):
        value = fx.Vector.from_elements([
            fx.Int32(llvm.extractvalue(fx.Int32.ir_type, result, [side * 4 + i])) for i in range(4)], fx.Int32)
        raw = value.ir_value()
        parts.append(fx.Vector(llvm.inline_asm(raw.type, [raw], "", "=v,0", has_side_effects=True)))
    return parts


@cache
def flydsl_moe_gemm_4wave_down_a8w4(*, n, k, topk, num_experts,
                                   block_n=128, output_cache_policy=0, sharded=False):
    """Return down(output, A, B, A_scales, B_scales, ids, routes, experts, valid, counter).

    Fixed M128 / 512 workers. Only BN64/128, aux0/18 and global/8-shard queues
    are retained. Counter: contiguous int32[1] or int32[256] (heads every32).
    No allocation, sorting, reduction, self-reset, task swizzle or tuning at launch.
    Routed/buffer allocations must stay below 4 GiB; rejected rather than wrapped.
    """
    if k not in (128, 256, 384, 512) or block_n not in (64, 128):
        raise ValueError("require K in 128/256/384/512 and BN64/128")
    if n <= 0 or n % (8 * block_n) or n // (4 * block_n) < 4:
        raise ValueError("paired OC4 requires an even number of packets, at least four")
    if not 0 < topk <= min(num_experts, 127) or output_cache_policy not in (0, 18):
        raise ValueError("require 0<TOPK<=min(E,127) and cache policy 0 or 18")
    block_m, threads, workers, mm = 128, 256, 512, 2
    nn, kb_count, packets = block_n // 16, k // 128, n // (4 * block_n)
    kp_count = (kb_count + 1) // 2
    scale_cols, scale_stride = kp_count * 8, kp_count * 64
    tile_bytes = block_n * k // 2
    scale_words = block_n // 32 * scale_stride
    lds_bytes = tile_bytes * 4 + scale_words * 16 + block_m * 8 + 4
    if lds_bytes > 160 * 1024:
        raise ValueError("tile exceeds gfx950 LDS budget")

    @fx.struct
    class Storage:
        b: fx.Array[fx.Int32, tile_bytes, 16]
        scales: fx.Array[fx.Int32, scale_words * 4, 16]
        ids: fx.Array[fx.Int32, block_m, 16]
        routes: fx.Array[fx.Float32, block_m, 16]
        ticket: fx.Array[fx.Int32, 1, 4]

    @flyc.kernel(known_block_size=[threads, 1, 1])
    def moe_4wave_down_a8w4(output: fx.Pointer, input_q: fx.Pointer, weight: fx.Pointer,
                           input_scales: fx.Pointer, weight_scales: fx.Pointer,
                           ids_arg: fx.Pointer, routes_arg: fx.Pointer, experts_arg: fx.Pointer,
                           valid_arg: fx.Pointer, counter_arg: fx.Pointer,
                           tokens: fx.Int32, capacity: fx.Int32):
        tid = fx.thread_idx.x
        wave, lane = tid // 64, tid % 64
        home = fx.Int32(fx.block_idx.x) % 8
        lr, lk = lane % 16, lane // 16
        lds = fx.SharedAllocator().allocate(Storage).peek()
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
        counter = _tensor(counter_arg, (256 if sharded else 1,), fx.Int32)
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
        limit = valid_data[0]
        running = fx.Boolean(True)
        while running:
            if tid == 0:
                head = counter
                if fx.const_expr(sharded):
                    head = fx.make_view(fx.get_iter(counter) + home * 32, fx.make_layout(1, 1))
                lds.ticket.ptr[0] = _claim(head)
            fx.barrier()
            task = lds.ticket.ptr[0]
            if fx.const_expr(sharded):
                task = task * 8 + home
            bm, oc = task // 4, task % 4
            running = bm * block_m < limit
            if running:
                expert = expert_data[bm]
                b_buffer = rocdl.make_buffer_tensor(b_data, False)
                bs_buffer = rocdl.make_buffer_tensor(bs_data, False)
                isrc = fx.make_view(fx.get_iter(id_data) + bm * block_m, fx.make_layout(block_m, 1))
                rsrc = fx.make_view(fx.get_iter(route_data) + bm * block_m, fx.make_layout(block_m, 1))
                copy_async_lds(rocdl.make_buffer_tensor(isrc, False), ids, block_m, threads)
                copy_async_lds(rocdl.make_buffer_tensor(rsrc, False), routes, block_m, threads)
                rocdl.asyncmark()
                rocdl.wait_asyncmark(0)
                fx.barrier()
                input_rows, live, weights, output_rows = {}, {}, {}, {}
                for mi in range_constexpr(mm):
                    row = wave * 32 + mi * 16 + lr
                    encoded = ids[row]
                    token, slot = encoded & 0xFFFFFF, encoded >> 24
                    input_rows[mi] = token * topk + slot
                    live[mi] = (token < tokens) & (slot < topk) & (slot >= 0)
                    weights[mi] = routes[row]
                    target = live[mi].select(input_rows[mi], rows)
                    for parity in range_constexpr(2):
                        output_rows[mi, parity] = fx.Int32(llvm.call_intrinsic(
                            fx.Int32.ir_type, "llvm.amdgcn.mov.dpp.i32", [target.ir_value(),
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
                    offset = expert * (n * k // 8) + oc * (n // 4 * k // 8) + q * (tile_bytes // 4)
                    src = fx.make_view(fx.get_iter(b_buffer) + offset, fx.make_layout(tile_bytes // 4, 1))
                    copy_async_lds(src, b_ring[None, None, None, None, None, q % 4], tile_bytes // 4, threads)
                    offset = (expert * (n // 32) + (oc * (n // 4) + q * block_n) // 32) * scale_stride
                    src = fx.make_view(fx.get_iter(bs_buffer) + offset, fx.make_layout(scale_words, 1))
                    copy_async_lds(src, bs_ring[None, None, None, q % 4], scale_words, threads)

                def read_b(q):
                    for kb in range_constexpr(kb_count):
                        for ni in range_constexpr(nn):
                            b[None, ni, kb] = b_ring[None, lr, lk, kb, ni, q % 4].load()
                    for kp in range_constexpr(kp_count):
                        for np in range_constexpr(nn // 2):
                            sb[np, kp] = bs_ring[lane, kp, np, q % 4]

                def compute():
                    acc.fill(0)
                    for kb in range_constexpr(kb_count):
                        for mi in range_constexpr(mm):
                            for ni in range_constexpr(nn):
                                b4 = b[None, ni, kb].load()
                                zero = fx.Int32(0)
                                bv = fx.Vector.from_elements([b4[0], b4[1], b4[2], b4[3], zero, zero, zero, zero], fx.Int32)
                                # Preserve native FP4/FP8 scaled-MFMA and scale opsel.
                                result = rocdl.mfma_scale_f32_16x16x128_f8f6f4(ir.VectorType.get([4], fx.Float32.ir_type), [
                                    bv.ir_value(), a[None, mi, kb].load().ir_value(),
                                    acc[None, mi, ni].load().ir_value(), 4, 0,
                                    (kb % 2) * 2 + ni % 2, sb[ni // 2, kb // 2].ir_value(),
                                    (kb % 2) * 2, sa[mi, kb // 2].ir_value()])
                                acc[None, mi, ni].store(result)
                                rocdl.sched_barrier(0)
                    for mi in range_constexpr(mm):
                        for ni in range_constexpr(0, nn, 2):
                            c0 = acc[None, mi, ni].load() * weights[mi]
                            c1 = acc[None, mi, ni + 1].load() * weights[mi]
                            d0, d1 = rocdl.cvt_pk_bf16_f32(c0[0], c0[1]), rocdl.cvt_pk_bf16_f32(c0[2], c0[3])
                            d2, d3 = rocdl.cvt_pk_bf16_f32(c1[0], c1[1]), rocdl.cvt_pk_bf16_f32(c1[2], c1[3])
                            ty = ir.Type.parse("!llvm.struct<(i32, i32)>")
                            # cvt_pk still returns a legacy scalar in FlyDSL 0.3.2.
                            lo = rocdl.permlane16_swap(ty, _to_raw(d0), _to_raw(d2), False, False)
                            hi = rocdl.permlane16_swap(ty, _to_raw(d1), _to_raw(d3), False, False)
                            values = fx.Vector.from_elements([
                                fx.Int32(llvm.extractvalue(fx.Int32.ir_type, lo, [0])), fx.Int32(llvm.extractvalue(fx.Int32.ir_type, hi, [0])),
                                fx.Int32(llvm.extractvalue(fx.Int32.ir_type, lo, [1])), fx.Int32(llvm.extractvalue(fx.Int32.ir_type, hi, [1]))], fx.Int32)
                            packed[None, mi, ni // 2].store(values.bitcast(fx.BFloat16))
                    for mi in range_constexpr(mm):
                        for pair in range_constexpr(0, nn // 2, 2):
                            parts = _coalesce(packed[None, mi, pair].load().bitcast(fx.Int32),
                                              packed[None, mi, pair + 1].load().bitcast(fx.Int32))
                            packed[None, mi, pair].store(parts[0].bitcast(fx.BFloat16))
                            packed[None, mi, pair + 1].store(parts[1].bitcast(fx.BFloat16))

                def store(q):
                    n_base = oc * (n // 4) + q * block_n
                    swap = (lk & 1) * 2 + (lk >> 1)
                    for mi in range_constexpr(mm):
                        for ni in range_constexpr(0, nn, 2):
                            row = output_rows[mi, (ni // 2) % 2]
                            col = n_base // 8 + (ni // 4) * 8 + swap + (lr % 2) * 4
                            fragment = fx.make_rmem_tensor(4, fx.Int32)
                            fragment.store(packed[None, mi, ni // 2].load().bitcast(fx.Int32))
                            atom = fx.make_copy_atom(rocdl.BufferCopy128b(cache_modifier=output_cache_policy), fx.Int32)
                            fx.copy(atom, fragment, c_data[None, col, row])

                for q in range_constexpr(4):
                    load_b(fx.Int32(q))
                    rocdl.asyncmark()

                def pair(q, first=False):
                    rocdl.s_setprio(1)
                    rocdl.wait_asyncmark(2 if first else 0)
                    fx.barrier()  # publish this pair and retire previous readers
                    read_b(q)
                    if fx.const_expr(not first):
                        store(q - 1)
                    if q + 2 < packets:
                        if fx.const_expr(not first):
                            load_b(q + 2)
                            load_b(q + 3)
                            rocdl.asyncmark()
                    rocdl.s_waitcnt(lgkmcnt=0)
                    rocdl.s_setprio(0)
                    compute()
                    rocdl.s_setprio(1)
                    read_b(q + 1)
                    store(q)
                    rocdl.s_waitcnt(lgkmcnt=0)
                    rocdl.s_setprio(0)
                    compute()

                pair(fx.Int32(0), first=True)
                for q in range(fx.Int32(2), fx.Int32(packets), fx.Int32(2)):
                    pair(q)
                store(packets - 1)
                rocdl.wait_asyncmark(0)
                rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0)
                fx.barrier()  # metadata and ring may now be reused by the next task

    @flyc.jit
    def launch(output: fx.Pointer, input_q: fx.Pointer, weight: fx.Pointer, sa: fx.Pointer, sb: fx.Pointer,
               ids: fx.Pointer, routes: fx.Pointer, experts: fx.Pointer, valid: fx.Pointer, counter: fx.Pointer,
               tokens: fx.Int32, capacity: fx.Int32, stream: fx.Stream):
        moe_4wave_down_a8w4(output, input_q, weight, sa, sb, ids, routes, experts, valid, counter, tokens, capacity,
            value_attrs={"rocdl.waves_per_eu": 2, "rocdl.flat_work_group_size": "256,256",
                         "passthrough": [["target-features", "-packed-fp32-ops"]]},
        ).launch(grid=(workers, 1, 1), block=(threads, 1, 1), stream=stream)

    def down(output, input_q, weight, sa, sb, ids, routes, experts, valid, counter):
        tokens = input_q.shape[0]
        assert input_q.shape == (tokens, topk, k) and output.shape == (tokens, topk, n)
        assert 0 < tokens < (1 << 24) and output.dtype == torch.bfloat16
        assert input_q.dtype == torch.float8_e4m3fn and weight.shape == (num_experts, n, k // 2)
        assert weight.element_size() == sa.element_size() == sb.element_size() == 1
        # Sorting may allocate a partial capacity tail; only complete valid
        # blocks are consumed, while shuffled scales round capacity up to M128.
        assert ids.shape == routes.shape and (ids.numel() + block_m - 1) // block_m == experts.numel()
        assert sa.numel() == experts.numel() * block_m * scale_cols and sb.numel() == num_experts * n * scale_cols
        assert ids.dtype == experts.dtype == valid.dtype == counter.dtype == torch.int32
        assert routes.dtype == torch.float32 and valid.numel() >= 1
        assert counter.numel() == (256 if sharded else 1)
        for tensor in (output, input_q, weight, sa, sb, ids, routes, experts, valid, counter):
            assert tensor.is_cuda and tensor.device == output.device and tensor.is_contiguous()
            assert tensor.numel() * tensor.element_size() < (1 << 32), "32-bit buffer range exceeded"
        assert torch.cuda.get_device_properties(output.device).gcnArchName.startswith("gfx950")
        counter.zero_()
        _run_compiled(launch, _ptr(output), _ptr(input_q), ptr_arg(weight), ptr_arg(sa), ptr_arg(sb),
                      _ptr(ids), _ptr(routes), _ptr(experts), _ptr(valid), _ptr(counter),
                      fx.Int32(tokens), fx.Int32(experts.numel()), fx.Stream(torch.cuda.current_stream(output.device).cuda_stream))
        return output

    down.launch = launch
    down.config = dict(block_m=block_m, sort_block_m=block_m, block_n=block_n, num_waves=4,
        num_oc_splits=4, persistent_workgroups=workers, workers=workers, publication="paired",
        output_layout="routed", row_store_bytes=128, output_cache_policy=output_cache_policy,
        sharded=sharded, counter_elements=256 if sharded else 1, lds_bytes=lds_bytes)
    return down