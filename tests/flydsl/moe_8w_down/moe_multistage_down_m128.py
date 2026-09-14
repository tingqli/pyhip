# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Fixed M128/4-wave/OC8 down: native sort128, ring4, exact FP32 scales.

Derived from the winning archived implementation; no imports from try.
Independent PF3 or paired persistent B publication; C stays in registers.
"""

from functools import cache

import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, range_constexpr, rocdl
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled
from pyhip.contrib.flydsl import helpers as fxh
from pyhip.contrib.flydsl.moe_gemm_2stage.common import torch_tensor_to_pointer as _ptr
from moe_multistage_down import _coalesce_output_pairs, _mark, _pack_pair, _pin_address, _pin_packet, _scalar, _stage_end


# Fixed case metadata, not factory tuning switches. Transpose widths describe
# task order, not physical XCD binding.
M128_CONTINUOUS_VMEM_CONFIG = {
    "block_m": 128, "block_n": 128, "sort_block_m": 128,
    "num_waves": 4, "num_oc_splits": 8, "xcd_count": 2,
    "weight_cache_policy": 16, "output_cache_policy": 18,
    "output_layout": "packed", "packed_rows": 128,
}


@cache
def make_m128_down(*, n, k=256, topk, num_experts, persistent=False):
    """Fixed OC8 packed ABI: independent width2 or width4/512-worker queues.

    Caller counter is preserved. Persistent heads are zero-initialized once;
    each shard's last exiting worker restores zero before kernel completion.
    This reset is part of Down, including graph replay; warm before capture.
    The cached callable owns queue state and must not run concurrently across
    streams. Its private heads must not be modified by callers. All tuning
    experiments live in analysis snapshots.
    """
    if k != 256 or n <= 0 or n % 1024 or not 0 < topk <= min(num_experts, 255):
        raise ValueError("M128 requires K256, N a positive multiple of1024, and valid TOPK")
    width = 4 if persistent else 2
    n_split, packets = n // 8, n // 512
    scale_count = n_split // 128 * 2
    scale_words = (scale_count + 63 + 255) // 256 * 256
    ids_offset = scale_words + 4 * 4096
    rows_offset = ids_offset + 128
    alive_offset = rows_offset + 128 * 3
    task_offset = alive_offset + 2
    arena_words = task_offset + int(persistent)
    attrs = {"rocdl.waves_per_eu": 2, "rocdl.flat_work_group_size": "256,256",
             "llvm.passthrough": [["target-features", "-packed-fp32-ops"], ["amdgpu-agpr-alloc", "0,0"]]}

    @fx.struct
    class Storage:
        arena: fx.Array[fx.Int32, arena_words, 16]

    @flyc.kernel(known_block_size=[256, 1, 1])
    def moe_down_m128_kernel(output: fx.Pointer, input_q: fx.Pointer, weight: fx.Pointer,
                             input_scales: fx.Pointer, weight_scales: fx.Pointer,
                             sorted_ids: fx.Pointer, sorted_weights: fx.Pointer, expert_ids: fx.Pointer,
                             valid_ids: fx.Pointer, counter: fx.Pointer, tokens: fx.Int32,
                             capacity_blocks: fx.Int32):
        tid = fx.Int32(fx.thread_idx.x)
        wave = _scalar(tid // 64)
        worker = fx.Int32(fx.block_idx.x)
        virtual = worker
        active_tasks = (_scalar(valid_ids[0]) // 128) * 8
        lds = fx.SharedAllocator().allocate(Storage).peek()
        task_slot = fx.make_view(lds.arena.ptr + task_offset, fx.make_layout(1, 1))

        def take_task(slot):
            # One claim site in the while header: two sites (prologue/latch)
            # were merged into divergent control flow by the installed LLVM.
            if tid == 0:
                # Logical shard=worker%8 guarantees coverage even if hardware
                # placement changes. Measured bid%8=XCC supplies the affinity;
                # it is not required for correctness and no stealing is needed.
                head = counter + worker % 8 * 32
                ptr = llvm.inttoptr(ir.Type.parse("!llvm.ptr<1>"), fx.ptrtoint(head).ir_value())
                slot[0] = fx.Int32(llvm.AtomicRMWOp(llvm.AtomicBinOp.add, ptr,
                    fx.Int32(1).ir_value(), llvm.AtomicOrdering.monotonic, syncscope="agent").res)
            fx.barrier()
            rank = _scalar(slot[0])
            return rank * 8 + worker % 8

        while (take_task(task_slot) if const_expr(persistent) else virtual) < active_tasks:
            task = _scalar(task_slot[0]) * 8 + worker % 8 if const_expr(persistent) else virtual
            chunk = active_tasks // width
            mapped = task % width * chunk + task // width
            task = (task < chunk * width).select(mapped, task)
            block_m, oc = task // 8, task % 8
            row_begin = block_m * 128
            expert = _scalar(expert_ids[block_m])
            rows = tokens * topk
            lane = tid % 64
            lr, lk = lane % 16, lane // 16
            bptr = lds.arena.ptr + scale_words
            fptr = fx.recast_iter(fx.PointerType.get(fx.Float32.ir_type, lds.arena.ptr.memspace, 16), lds.arena.ptr)
            ids = fx.make_view(lds.arena.ptr + ids_offset, fx.make_layout(128, 1))
            row_scales = fx.make_view(fptr + rows_offset, fx.make_layout(384, 1))
            bscales = fx.make_view(fptr, fx.make_layout(scale_count, 1))
            alive = fx.make_view(lds.arena.ptr + alive_offset, fx.make_layout(2, 1))
            asbuf = fx.rocdl.make_buffer_tensor(fx.make_view(input_scales, fx.make_layout(rows * 2, 1)), False)
            asrsrc = fx.rocdl.get_buffer_rsrc(fx.get_iter(asbuf))
            zero = fx.Int32(0).ir_value()
            if tid < 128:
                encoded = sorted_ids[row_begin + tid].bitcast(fx.Uint32)
                token, slot = encoded & 0xFFFFFF, encoded >> 24
                valid = (token < fx.Uint32(tokens)) & (slot < topk)
                ids[tid] = encoded.bitcast(fx.Int32)
                row_scales[256 + tid] = sorted_weights[row_begin + tid]
                source = fx.Int32(token) * topk + fx.Int32(slot)
                for kb in range_constexpr(2):
                    offset = valid.select((kb * rows + source) * 4, fx.Int32(-1))
                    row_scales[kb * 128 + tid] = fx.Float32(rocdl.raw_ptr_buffer_load(
                        fx.Float32.ir_type, asrsrc, offset.ir_value(), zero))
                if const_expr(not persistent):
                    mask = fx.Uint64(rocdl.ballot(fx.Uint64.ir_type, valid.ir_value()))
                    if lane == 0:
                        alive[wave] = (mask != fx.Uint64(0)).select(fx.Int32(1), fx.Int32(0))
            for turn in range_constexpr((scale_count + 255) // 256):
                index = tid + turn * 256
                if index < scale_count:
                    bscales[index] = weight_scales[expert * (n // 64) + oc * scale_count + index]
            fx.barrier()
            has_rows = fx.Boolean(True) if const_expr(persistent) else _scalar(alive[0] | alive[1]) != 0
            if has_rows:
                wview = fx.make_view(weight + fx.Int64(expert) * (n * 256) + fx.Int64(oc) * (n_split * 256),
                                     fx.make_layout(n_split * 256, 1))
                wbuf = fx.rocdl.make_buffer_tensor(wview, False)
                wrsrc = fx.rocdl.get_buffer_rsrc(fx.get_iter(wbuf))
                dma_base = _scalar(fx.Int32(fx.ptrtoint(bptr))) + wave * 1024
                dma_offset = _pin_address(tid * 16)
                phase_worker = worker
                if const_expr(persistent):
                    # Re-materialize invariant worker phase per task instead
                    # of keeping all derived N offsets live across the queue.
                    phase_worker = fx.Int32(llvm.inline_asm(fx.Int32.ir_type, [worker.ir_value()],
                        "", "=s,0", has_side_effects=True))

                def packet_index(q):
                    # Resident-worker phases retain B locality without making
                    # all consumers restart each task at the same N phase.
                    rotation = phase_worker // 8
                    return ((fx.Int32(q) // 2 + rotation) % (packets // 2)) * 2 + fx.Int32(q) % 2

                def dma_turn(q, phase, turn):
                    dst = llvm.inttoptr(ir.Type.parse("!llvm.ptr<3>"),
                        (dma_base + phase * 16384 + turn * 4096).ir_value())
                    rocdl.raw_ptr_buffer_load_async_lds(wrsrc, dst, fx.Int32(16).ir_value(),
                        dma_offset.ir_value(), (packet_index(q) * 16384 + turn * 4096).ir_value(), zero,
                        aux=ir.IntegerAttr.get(fx.Int32.ir_type, 16))

                for q in range_constexpr(min(4 if persistent else 3, packets)):
                    for turn in range_constexpr(4):
                        dma_turn(q, q, turn)
                    rocdl.asyncmark()
                blds = fxh.LdsTensor(fx.make_view(bptr, fx.make_layout(4, 1)))
                rlds = fxh.LdsTensor(fx.make_view(fptr + rows_offset, fx.make_layout(1, 1)))
                scale_base = _scalar(fx.Int32(fx.ptrtoint(fptr)))
                aptr = fx.recast_iter(fx.PointerType.get(fx.Int32.ir_type, input_q.memspace, 16), input_q)
                abuf = fx.rocdl.make_buffer_tensor(fx.make_view(aptr, fx.make_layout(rows * 64, 1)), False)
                apacket = fxh.BufferTensor(fx.make_view(fx.get_iter(abuf), fx.make_layout(4, 1)))
                out_rows = capacity_blocks * 128
                obuf = fx.rocdl.make_buffer_tensor(fx.make_view(output, fx.make_layout(out_rows * n, 1)), False)
                orsrc = fx.rocdl.get_buffer_rsrc(fx.get_iter(obuf))
                a = fx.make_rmem_tensor([8, 2, 2], fx.Int32)
                awords = fx.make_view(fx.get_iter(a), fx.make_ordered_layout([4, 2, 2, 2], 0))
                cached_rows = fx.make_rmem_tensor([2, 3], fx.Float32)
                out_addresses = []
                for mi in range_constexpr(2):
                    row = wave * 32 + mi * 16 + lr
                    encoded = ids[row].bitcast(fx.Uint32)
                    token, slot = encoded & 0xFFFFFF, encoded >> 24
                    valid = (token < fx.Uint32(tokens)) & (slot < topk)
                    source = fx.Int32(token) * topk + fx.Int32(slot)
                    for kb in range_constexpr(2):
                        for part in range_constexpr(2):
                            offset = valid.select(source * 256 + lk * 16 + kb * 128 + part * 64, fx.Int32(-1))
                            awords[None, part, mi, kb].store(apacket.load(voffset_bytes=offset))
                    for kb in range_constexpr(3):
                        address = fx.Int32(fx.ptrtoint(fptr + rows_offset)) + (kb * 128 + row) * 4
                        cached_rows[mi, kb] = rlds.load(address_bytes=address)[0]
                    if const_expr(persistent):
                        row_valid = valid.select(fx.Int32(1), fx.Int32(0))
                    for parity in range_constexpr(2):
                        target = wave * 32 + mi * 16 + lr // 2 * 2 + parity
                        if const_expr(persistent):
                            # Packed stores exchange a row bit; derive the
                            # exact target validity without rereading IDs.
                            mask = fx.Int32(llvm.call_intrinsic(fx.Int32.ir_type,
                                "llvm.amdgcn.mov.dpp.i32", [row_valid.ir_value(),
                                fx.Int32(0xa0 + parity * 0x55).ir_value(), fx.Int32(15).ir_value(),
                                fx.Int32(15).ir_value(), fx.Boolean(False).ir_value()], [], []))
                            valid = mask != 0
                        else:
                            encoded = ids[target].bitcast(fx.Uint32)
                            token, slot = encoded & 0xFFFFFF, encoded >> 24
                            valid = (token < fx.Uint32(tokens)) & (slot < topk)
                        column = ((lk & 1) * 2 + (lk >> 1)) * 8 + lr % 2 * 32
                        offset = block_m * (128 * n * 2) + oc * (128 * n_split * 2) + target * 128 + column * 2
                        out_addresses.append(_pin_address(valid.select(offset, out_rows * n * 2)))
                b_addresses = [_pin_address(fx.Int32(fx.ptrtoint(bptr)) + lr * 16 + lk * 256 + phase * 16384)
                               for phase in range_constexpr(4)]
                atom = fx.make_mma_atom(rocdl.cdna4.MFMA_Scale(16, 16, 128, fx.Float8E4M3FN))

                def store_one(packed, q, record):
                    rocdl.raw_ptr_buffer_store(packed[record].ir_value(), orsrc, out_addresses[record].ir_value(),
                        (packet_index(q) * 16384).ir_value(), aux=ir.IntegerAttr.get(fx.Int32.ir_type, 18))

                def store(packed, q):
                    for record in range_constexpr(4):
                        store_one(packed, q, record)

                pair_scales = fx.make_rmem_tensor(2, fx.Int32)

                def memory(q, phase, tail_left=4):
                    refill_pair = persistent and phase % 2 == 0 and tail_left > 2 and (not isinstance(q, int) or q != 0)
                    rocdl.s_setprio(3)
                    if const_expr(not persistent or phase % 2 == 0):
                        # Persistent publishes a pair and retires prior-pair
                        # readers before reusing their slots; PF3 publishes one.
                        wait = ((min(2, packets - 2) if isinstance(q, int) and q == 0 else 0)
                                if persistent else min(2, tail_left - 1))
                        rocdl.wait_asyncmark(wait)
                        _stage_end()
                    _mark(f"MOE128_MEMORY_BEGIN_{phase}")
                    b = fx.make_rmem_tensor([8, 4, 2], fx.Int32)
                    words = fx.make_view(fx.get_iter(b), fx.make_ordered_layout([4, 2, 4, 2], 0))
                    for kb in range_constexpr(2):
                        for ni in range_constexpr(4):
                            for part in range_constexpr(2):
                                if const_expr(refill_pair and part == 0):
                                    turn_id = kb * 4 + ni
                                    dma_turn(q + 2 + turn_id // 4, (phase + 2 + turn_id // 4) % 4, turn_id % 4)
                                elif const_expr(not persistent and tail_left > 3 and (kb * 8 + ni * 2 + part) % 4 == 0):
                                    dma_turn(q + 3, (phase + 3) % 4, (kb * 8 + ni * 2 + part) // 4)
                                words[None, part, ni, kb].store(blds.load(address_bytes=b_addresses[phase],
                                    offset_bytes=ni * 4096 + kb * 2048 + part * 1024))
                    if const_expr(refill_pair or (not persistent and tail_left > 3)):
                        rocdl.asyncmark()
                    for kb in range_constexpr(2):
                        if const_expr(not persistent or phase % 2 == 0):
                            address = scale_base + (packet_index(q) // 2 * 2 + kb) * 4
                            raw = llvm.inline_asm(ir.Type.parse("!llvm.struct<(i32, i32)>"), [address.ir_value()],
                                "s_mov_b32 $1, m0\ns_mov_b32 m0, $2\ns_nop 1\nds_read_addtid_b32 $0\ns_mov_b32 m0, $1",
                                "=&v,=&s,s,~{memory}", has_side_effects=True)
                            pair_scales[kb] = fx.Int32(llvm.extractvalue(fx.Int32.ir_type, raw, [0]))
                    rocdl.s_waitcnt(lgkmcnt=0)
                    for kb in range_constexpr(2):
                        for ni in range_constexpr(4):
                            for part in range_constexpr(2):
                                view = words[None, part, ni, kb]
                                view.store(_pin_packet(view.load()))
                    _mark(f"MOE128_MEMORY_END_{phase}")
                    return b, [pair_scales[kb] for kb in range_constexpr(2)]

                def compute(b, scales, previous, phase, q, old_words, first=False):
                    rocdl.s_setprio(0)
                    _mark(f"MOE128_COMPUTE_BEGIN_{phase}")
                    factors = [[cached_rows[mi, kb] * _scalar(scales[kb]).bitcast(fx.Float32) for mi in range_constexpr(2)]
                               for kb in range_constexpr(2)]
                    accum = fx.make_rmem_tensor([4, 2, 4], fx.Float32)
                    scaled, words, swaps = [[] for _ in range_constexpr(4)], [[] for _ in range_constexpr(4)], [[] for _ in range_constexpr(4)]
                    packed = [[], []]
                    pending, dequant, pack_index = [], [], 0

                    def pack_op(index):
                        pair_id, op = index // 14, index % 14
                        mi, pair = pair_id // 2, pair_id % 2
                        if const_expr(op < 8):
                            scaled[pair_id].append(previous[mi * 4 + pair * 2 + op // 4][op % 4] * cached_rows[mi, 2])
                        elif const_expr(op < 12):
                            wi = op - 8
                            words[pair_id].append(fx.Vector.from_elements(scaled[pair_id][wi * 2:wi * 2 + 2], fx.Float32).to(fx.BFloat16).bitcast(fx.Int32)[0])
                        else:
                            wi = op - 12
                            raw = rocdl.permlane16_swap(ir.Type.parse("!llvm.struct<(i32, i32)>"),
                                words[pair_id][wi].ir_value(), words[pair_id][wi + 2].ir_value(), False, False)
                            swaps[pair_id].append(raw)
                            if const_expr(op == 13):
                                lo, hi = swaps[pair_id]
                                packed[mi].append(_pin_packet(fx.Vector.from_elements([
                                    fx.Int32(llvm.extractvalue(fx.Int32.ir_type, lo, [0])),
                                    fx.Int32(llvm.extractvalue(fx.Int32.ir_type, hi, [0])),
                                    fx.Int32(llvm.extractvalue(fx.Int32.ir_type, lo, [1])),
                                    fx.Int32(llvm.extractvalue(fx.Int32.ir_type, hi, [1])),
                                ], fx.Int32)))

                    def retire(item):
                        partial, mi, ni, kb, element = item
                        value = partial.load()[element]
                        if const_expr(kb == 0):
                            accum[element, mi, ni] = value * factors[kb][mi]
                        else:
                            accum[element, mi, ni] = fx.Float32(fxh.eltwise_op(
                                "llvm.fma.f32", value, factors[kb][mi], accum[element, mi, ni]))

                    for index in range_constexpr(16):
                        if const_expr(index % 4 == 2 and (not isinstance(q, int) or q >= 2)):
                            store_one(old_words, q - 2, index // 4)
                        kb, local = index // 8, index % 8
                        mi, ni = local // 4, local % 4
                        partial = fx.make_rmem_tensor(4, fx.Float32)
                        partial.fill(0)
                        fx.gemm(atom, partial, b[None, ni, kb], a[None, mi, kb], partial)
                        rocdl.sched_barrier(0)
                        pending.append((partial, mi, ni, kb))
                        if const_expr(len(pending) > 2):
                            old = pending.pop(0)
                            dequant.extend([(*old, element) for element in range_constexpr(4)])
                        remaining = 7
                        for _ in range_constexpr(min(4, len(dequant))):
                            retire(dequant.pop(0))
                            remaining -= 1
                        while const_expr(not first and pack_index < 56 and remaining > 0):
                            pack_op(pack_index)
                            pack_index += 1
                            remaining -= 1
                        while const_expr(len(dequant) > 0 and remaining > 0):
                            retire(dequant.pop(0))
                            remaining -= 1
                        rocdl.sched_barrier(0)
                    for old in pending:
                        dequant.extend([(*old, element) for element in range_constexpr(4)])
                    for item in dequant:
                        retire(item)
                    while const_expr(not first and pack_index < 56):
                        pack_op(pack_index)
                        pack_index += 1
                    result = [_pin_packet(accum[None, mi, ni].load()) for mi in range_constexpr(2) for ni in range_constexpr(4)]
                    output_words = []
                    if const_expr(not first):
                        for mi in range_constexpr(2):
                            output_words.extend(_coalesce_output_pairs(packed[mi][0], packed[mi][1]))
                    _mark(f"MOE128_COMPUTE_END_{phase}")
                    return result, output_words

                rocdl.sched_barrier(0)
                current = [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range_constexpr(8)]
                packed = [fx.Vector.filled(4, 0, fx.Int32) for _ in range_constexpr(4)]
                prologue = min(4, packets)
                for q in range_constexpr(prologue):
                    b, scales = memory(q, q % 4, tail_left=packets - q)
                    current, packed = compute(b, scales, current, q % 4, q, packed, first=q == 0)
                loop_end = max(prologue, (packets - 3) // 4 * 4)
                if const_expr(packets > prologue):
                    for q, state in range(fx.Int32(prologue), fx.Int32(loop_end), fx.Int32(4), init=[*current, *packed]):
                        current = [fx.Vector(state[i]) for i in range_constexpr(8)]
                        packed = [fx.Vector(state[8 + i]) for i in range_constexpr(4)]
                        for phase in range_constexpr(4):
                            b, scales = memory(q + phase, phase)
                            current, packed = compute(b, scales, current, phase, q + phase, packed)
                        final = yield [*current, *packed]
                    current = [fx.Vector(final[i]) for i in range_constexpr(8)]
                    packed = [fx.Vector(final[8 + i]) for i in range_constexpr(4)]
                    for q in range_constexpr(loop_end, packets):
                        b, scales = memory(q, q % 4, tail_left=packets - q)
                        current, packed = compute(b, scales, current, q % 4, q, packed)
                if const_expr(persistent):
                    rocdl.s_setprio(3)
                store(packed, packets - 2)
                final_packed = []
                for mi in range_constexpr(2):
                    pieces = [_pack_pair(current[mi * 4 + pair * 2], current[mi * 4 + pair * 2 + 1], cached_rows[mi, 2])
                              for pair in range_constexpr(2)]
                    final_packed.extend(_coalesce_output_pairs(pieces[0], pieces[1]))
                store(final_packed, packets - 1)
                rocdl.wait_asyncmark(0)
                rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0)
            if const_expr(persistent):
                # All DMA/LDS readers retire before the next task reuses LDS.
                fx.barrier()
            else:
                virtual = active_tasks

        if const_expr(persistent):
            if tid == 0:
                # 64 workers/shard each make one terminal claim. This last
                # rank proves no worker can claim again; no grid-wide spin.
                if task_slot[0] == active_tasks // 8 + 63:
                    counter[worker % 8 * 32] = fx.Int32(0)

    @flyc.jit
    def launch(output: fx.Pointer, input_q: fx.Pointer, weight: fx.Pointer,
               input_scales: fx.Pointer, weight_scales: fx.Pointer, sorted_ids: fx.Pointer,
               sorted_weights: fx.Pointer, expert_ids: fx.Pointer, valid_ids: fx.Pointer,
               counter: fx.Pointer, tokens: fx.Int32, capacity_blocks: fx.Int32, stream: fx.Stream):
        moe_down_m128_kernel(output, input_q, weight, input_scales, weight_scales, sorted_ids,
            sorted_weights, expert_ids, valid_ids, counter, tokens, capacity_blocks, value_attrs=attrs).launch(
                grid=(512 if const_expr(persistent) else capacity_blocks * 8, 1, 1),
                block=(256, 1, 1), stream=stream)

    workspace = {}

    def down(output, input_q, weight, input_scales, weight_scales,
             sorted_ids, sorted_weights, expert_ids, valid_ids, counter):
        tokens, actual_topk, actual_k = input_q.shape
        tensors = (output, input_q, weight, input_scales, weight_scales, sorted_ids, sorted_weights, expert_ids, valid_ids, counter)
        assert 0 < tokens < (1 << 24) and (actual_topk, actual_k) == (topk, k)
        assert weight.shape == (num_experts, n, k) and output.shape == (expert_ids.numel() * 128, n)
        assert input_q.dtype == weight.dtype == torch.float8_e4m3fn and output.dtype == torch.bfloat16
        assert input_scales.dtype == weight_scales.dtype == sorted_weights.dtype == torch.float32
        assert sorted_ids.dtype == expert_ids.dtype == valid_ids.dtype == counter.dtype == torch.int32
        assert input_scales.numel() == tokens * topk * 2 and weight_scales.shape == (num_experts, n // 128, 2)
        assert sorted_ids.ndim == expert_ids.ndim == 1 and sorted_ids.shape == sorted_weights.shape
        assert expert_ids.numel() == (sorted_ids.numel() + 127) // 128, "native sort128 required"
        assert counter.numel() == 1 and valid_ids.numel() >= 1
        assert all(t.is_cuda and t.is_contiguous() and t.device == output.device for t in tensors)
        assert (output.numel() + n) * 2 < 1 << 32 and input_q.numel() < 1 << 32
        assert input_scales.numel() * 4 < 1 << 32 and n_split * 256 < 1 << 32
        assert torch.cuda.get_device_properties(output.device).gcnArchName.startswith("gfx950")
        if persistent:
            if "heads" not in workspace or workspace["heads"].device != output.device:
                workspace["heads"] = torch.zeros(256, dtype=torch.int32, device=output.device)
            tensors = (*tensors[:-1], workspace["heads"])
        stream = torch.cuda.current_stream(output.device)
        compiled = getattr(launch, "_cf", None)
        if compiled is None:
            _run_compiled(launch, *[_ptr(t) for t in tensors], fx.Int32(tokens), fx.Int32(expert_ids.numel()), fx.Stream(stream.cuda_stream))
        else:
            compiled(*(t.data_ptr() for t in tensors), tokens, expert_ids.numel(), stream.cuda_stream)
        return output

    down.config = {**M128_CONTINUOUS_VMEM_CONFIG, "persistent": persistent,
                   "scheduler": "sharded" if persistent else "independent", "persistent_workgroups": 512 if persistent else 0,
                   "xcd_count": width, "lds_bytes": arena_words * 4}
    if persistent:
        down.config["queue_self_reset"] = True
    down.workspace = workspace
    return down