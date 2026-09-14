# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Opt-in gfx950 MFMA32 MoE down experiment, not the default tuned kernel.

M256/N128/K256, eight waves, native 32x32x64 FP8 MFMA and (32,16) shuffled
weights. Two K64 MFMAs form each K128 partial before the original block
scales are applied. With fold_routing=True, FP32 routing multiplication is
reassociated into the scale factors (not bitwise-equivalent). defer_k1
carries the second partial into the next Compute. Without folding, its FMA
still precedes a separate routing multiply; the historical fourteen-VALU
schedule is not guaranteed. No AGPR, C LDS or scratch is intended.

output_layout defaults to routed [tokens,topk,N]. Experimental sorted,
packed and linear layouts use a caller-owned [expert_blocks*256,N] BF16
buffer and require a separate restoration step. linear_raw omits output
lane permutations, so its reduced VALU count is not a fourteen-gap claim.
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
from moe_multistage_down import (
    _coalesce_output_pairs, _mark, _pin_address, _pin_packet, _scalar,
    _stage_end, _task_thread_id,
)


MFMA32_CONFIG = {
    "block_n": 128, "num_oc_splits": 4, "persistent_workgroups": 256,
    "prefetch_distance": 3, "output_cache_policy": 2, "weight_cache_policy": 16,
    "fold_routing": True, "defer_k1": True,
}
MFMA32_EXPERIMENT_CONFIG = {
    **MFMA32_CONFIG, "skip_empty_waves": True,
    "cache_row_scales": True, "task_m_group": 2,
}


def _swap_words(first, second, width):
    operation = rocdl.permlane32_swap if width == 32 else rocdl.permlane16_swap
    result = operation(ir.Type.parse("!llvm.struct<(i32, i32)>"),
                       first.ir_value(), second.ir_value(), False, False)
    return [fx.Int32(llvm.extractvalue(fx.Int32.ir_type, result, [i])) for i in range(2)]


def _coalesce_word(first, second, odd):
    mask = fx.Uint64(0xAAAAAAAAAAAAAAAA if odd else 0x5555555555555555)
    left, right = (first, second) if odd else (second, first)
    return fx.Int32(llvm.inline_asm(
        fx.Int32.ir_type, [left.ir_value(), right.ir_value(), mask.ir_value()],
        "s_mov_b64 vcc, $3\nv_cndmask_b32_dpp $0, $1, $2, vcc quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf",
        "=&v,v,v,s,~{vcc}", has_side_effects=True,
    ))


@cache
def flydsl_moe_gemm_8wave_down_mfma32(
    *, n, k, topk, num_experts, block_m=256, block_n=128, num_oc_splits=4,
    persistent_workgroups=256, prefetch_distance=3, output_cache_policy=2,
    weight_cache_policy=16, fold_routing=True, defer_k1=True,
    skip_empty_waves=False, cache_row_scales=False, task_m_group=1,
    output_layout="routed", persistent=True, xcd_swizzle=False,
    valu_per_mfma=14, xcd_count=8, consumer_wait=False,
):
    assert block_m in (128, 256) and block_n == 128 and k == 256
    assert block_m == 256 or (not persistent and output_layout == "packed" and task_m_group == 1)
    assert xcd_count in (4, 8)
    assert not consumer_wait or (block_m == 128 and not persistent)
    threads = block_m * 2
    m_parts = 256 // block_m
    assert n > 0 and num_oc_splits > 0 and n % (128 * num_oc_splits) == 0
    assert 0 < topk <= min(num_experts, 255)
    assert persistent_workgroups > 0 and persistent_workgroups % 8 == 0
    assert prefetch_distance in (1, 2, 3)
    assert output_cache_policy in (0, 1, 2, 3, 16, 17, 18, 19)
    assert weight_cache_policy in (0, 1, 2, 3, 16, 17, 18, 19)
    assert not xcd_swizzle or not persistent, "XCD swizzle maps nonpersistent workgroup IDs"
    assert not xcd_swizzle or task_m_group == 1, "do not compose two task-order experiments"
    assert valu_per_mfma in (14, 16)
    assert task_m_group in (1, 2)
    assert output_layout in ("routed", "sorted", "packed", "linear", "linear_raw")
    sorted_output = output_layout != "routed"
    packed_output = output_layout in ("packed", "linear", "linear_raw")
    raw_output = output_layout == "linear_raw"
    n_split = n // num_oc_splits
    steps = n_split // 64
    slot_bytes = 16384
    scale_count = n_split // 128 * 2
    scale_words = (scale_count + 63 + 255) // 256 * 256
    ids_offset = scale_words + 4 * slot_bytes // 4
    rows_offset = ids_offset + block_m
    task_offset = rows_offset + block_m * 3
    assert (task_offset + 1) * 4 <= 160 * 1024
    exact_deferred = defer_k1 and not fold_routing
    pack_pair_ops = ((40 if raw_output else 44) if exact_deferred else
                     (24 if raw_output else 28) if not fold_routing or defer_k1 else
                     (8 if raw_output else 12))
    base_pack_ops = 2 * pack_pair_ops
    pack_count = base_pack_ops + (0 if raw_output else 24)
    state_c_count = 5 if defer_k1 else 2
    attrs = {
        "rocdl.waves_per_eu": 2,
        "rocdl.flat_work_group_size": f"{threads},{threads}",
        "llvm.passthrough": [["target-features", "-packed-fp32-ops"], ["amdgpu-agpr-alloc", "0,0"]],
    }

    @fx.struct
    class Storage:
        arena: fx.Array[fx.Int32, task_offset + 1]

    @flyc.kernel
    def moe_down_mfma32_kernel(
        output: fx.Pointer, input_q: fx.Pointer, weight: fx.Pointer,
        input_scales: fx.Pointer, weight_scales: fx.Pointer,
        sorted_ids: fx.Pointer, sorted_weights: fx.Pointer, expert_ids: fx.Pointer,
        valid_ids: fx.Pointer, counter: fx.Pointer,
        tokens: fx.Int32, expert_blocks: fx.Int32,
    ):
        tid = fx.Int32(fx.thread_idx.x)
        scalar_wave = _scalar(tid // 64)
        group = _scalar(tid // 256)
        rows = tokens * topk
        lds = fx.SharedAllocator().allocate(Storage).peek()
        bptr = lds.arena.ptr + scale_words
        ids = fx.make_view(lds.arena.ptr + ids_offset, fx.make_layout(block_m, 1))
        fptr = fx.recast_iter(fx.PointerType.get(fx.Float32.ir_type, lds.arena.ptr.memspace, 16), lds.arena.ptr)
        scales = fx.make_view(fptr, fx.make_layout(scale_count, 1))
        row_scales = fx.make_view(fptr + rows_offset, fx.make_layout(block_m * 3, 1))
        tasks = fx.make_view(lds.arena.ptr + task_offset, fx.make_layout(1, 1))
        blds = fxh.LdsTensor(fx.make_view(bptr, fx.make_layout(4, 1)))
        rlds = fxh.LdsTensor(fx.make_view(fptr + rows_offset, fx.make_layout(1, 1)))
        dma_base = _scalar(fx.Int32(fx.ptrtoint(bptr))) + scalar_wave * 1024
        scale_base = _scalar(fx.Int32(fx.ptrtoint(fptr)))
        aptr = fx.recast_iter(fx.PointerType.get(fx.Int32.ir_type, input_q.memspace, 16), input_q)
        abuf = fx.rocdl.make_buffer_tensor(fx.make_view(aptr, fx.make_layout(rows * 64, 1)), False)
        apacket = fxh.BufferTensor(fx.make_view(fx.get_iter(abuf), fx.make_layout(4, 1)))
        asbuf = fx.rocdl.make_buffer_tensor(fx.make_view(input_scales, fx.make_layout(rows * 2, 1)), False)
        asrsrc = fx.rocdl.get_buffer_rsrc(fx.get_iter(asbuf))
        output_rows = expert_blocks * 256 if const_expr(sorted_output) else rows
        obuf = fx.rocdl.make_buffer_tensor(fx.make_view(output, fx.make_layout(output_rows * n, 1)), False)
        orsrc = fx.rocdl.get_buffer_rsrc(fx.get_iter(obuf))
        limit = _scalar(valid_ids[0])
        atom = fx.make_mma_atom(rocdl.cdna4.MFMA_Scale(32, 32, 64, fx.Float8E4M3FN))
        zero = fx.Int32(0).ir_value()
        active = fx.Boolean(True)

        while active:
            t = fx.Uint32(_task_thread_id(scalar_wave))
            task = fx.Int32(0)
            if const_expr(persistent):
                if t == 0:
                    p = llvm.inttoptr(ir.Type.parse("!llvm.ptr<1>"), fx.ptrtoint(counter).ir_value())
                    next_task = fx.Int32(llvm.AtomicRMWOp(
                        llvm.AtomicBinOp.add, p, fx.Int32(1).ir_value(),
                        llvm.AtomicOrdering.monotonic, syncscope="agent",
                    ).res)
                    tasks[0] = next_task
                fx.barrier()
                task = _scalar(tasks[0])
            else:
                task = fx.Int32(fx.block_idx.x)
                if const_expr(xcd_swizzle):
                    active_tasks = (limit // 256) * num_oc_splits * m_parts
                    chunk = active_tasks // xcd_count
                    mapped = (task % xcd_count) * chunk + task // xcd_count
                    task = (task < chunk * xcd_count).select(mapped, task)
            bm, oc = task // num_oc_splits, task % num_oc_splits
            active = task < (limit // 256) * num_oc_splits * m_parts
            if active:
                if const_expr(task_m_group == 2):
                    total_blocks = limit // 256
                    group_index = task // (2 * num_oc_splits)
                    remaining = total_blocks - group_index * 2
                    width = (remaining < 2).select(remaining, fx.Int32(2))
                    local = task - group_index * 2 * num_oc_splits
                    bm, oc = group_index * 2 + local % width, local // width
                row_lane = t % 32
                k_lane = t % 64 // 32
                wave = t // 64
                parent, m_part = bm // m_parts, bm % m_parts
                row_begin = parent * 256 + m_part * block_m
                expert = _scalar(expert_ids[parent])
                if t < block_m:
                    ids[t] = sorted_ids[row_begin + t]
                    row_scales[block_m * 2 + t] = sorted_weights[row_begin + t]
                for turn in range_constexpr((scale_count + threads - 1) // threads):
                    index = t + turn * threads
                    if index < scale_count:
                        scales[index] = weight_scales[expert * (n // 128 * 2) + oc * scale_count + index]
                fx.barrier()
                if t < block_m:
                    encoded = ids[t].bitcast(fx.Uint32)
                    token, slot = encoded & 0xFFFFFF, encoded >> 24
                    valid = (token < fx.Uint32(tokens)) & (slot < topk)
                    source_row = fx.Int32(token) * topk + fx.Int32(slot)
                    for kb in range_constexpr(2):
                        offset = valid.select((kb * rows + source_row) * 4, fx.Int32(-1))
                        row_scales[kb * block_m + t] = fx.Float32(rocdl.raw_ptr_buffer_load(
                            fx.Float32.ir_type, asrsrc, offset.ir_value(), zero,
                        ))
                fx.barrier()
                wview = fx.make_view(weight + fx.Int64(expert) * (n * k) + fx.Int64(oc) * (n_split * k),
                                     fx.make_layout(n_split * k, 1))
                wbuf = fx.rocdl.make_buffer_tensor(wview, False)
                wrsrc = fx.rocdl.get_buffer_rsrc(fx.get_iter(wbuf))
                dma_offset = _pin_address(t * 16)
                a = fx.make_rmem_tensor([8, 4], fx.Int32)
                awords = fx.make_view(fx.get_iter(a), fx.make_ordered_layout([4, 2, 4], 0))
                row = wave * 32 + row_lane
                encoded = ids[row].bitcast(fx.Uint32)
                token, slot = encoded & 0xFFFFFF, encoded >> 24
                valid = (token < fx.Uint32(tokens)) & (slot < topk)
                wave_has_rows = fx.Int32(0)
                if const_expr(skip_empty_waves):
                    active_lanes = fx.Uint64(rocdl.ballot(fx.Uint64.ir_type, valid.ir_value()))
                    wave_has_rows = fx.Int32(llvm.inline_asm(
                        fx.Int32.ir_type, [(active_lanes != fx.Uint64(0)).select(fx.Int32(1), fx.Int32(0)).ir_value()],
                        "", "=s,0", has_side_effects=True,
                    ))
                source_row = fx.Int32(token) * topk + fx.Int32(slot)
                for kb in range_constexpr(4):
                    for part in range_constexpr(2):
                        offset = valid.select(source_row * 256 + k_lane * 32 + kb * 64 + part * 16, fx.Int32(-1))
                        awords[None, part, kb].store(apacket.load(voffset_bytes=offset))
                b_addresses = [_pin_address(fx.Int32(fx.ptrtoint(bptr)) + row_lane * 16 + k_lane * 1024 + i * slot_bytes)
                               for i in range_constexpr(4)]
                row_addresses = [_pin_address(fx.Int32(fx.ptrtoint(fptr + rows_offset)) + (kb * block_m + row) * 4)
                                 for kb in range_constexpr(3)]
                output_addresses = []
                for record in range_constexpr(4):
                    target_row = wave * 32 + row_lane if const_expr(raw_output) else wave * 32 + (row_lane % 16 // 2) * 2 + record // 2 + record % 2 * 16
                    route_id = ids[target_row].bitcast(fx.Uint32)
                    token, slot = route_id & 0xFFFFFF, route_id >> 24
                    valid = (token < fx.Uint32(tokens)) & (slot < topk)
                    column = k_lane * 8 + row_lane // 16 * 16 + row_lane % 2 * 32
                    output_row = row_begin + target_row if const_expr(sorted_output) else fx.Int32(token) * topk + fx.Int32(slot)
                    offset = (parent * (256 * n * 2) + oc * (256 * n_split * 2) + (m_part * block_m + target_row) * 128 + column * 2) if const_expr(packed_output) else output_row * (n * 2) + oc * (n_split * 2) + column * 2
                    if const_expr(output_layout in ("linear", "linear_raw")):
                        offset = bm * (256 * n * 2) + oc * (256 * n_split * 2) + wave * 4096 + record * 1024 + (t % 64) * 16
                    output_addresses.append(_pin_address(valid.select(offset, output_rows * (n * 2))))
                cached_rows = [fx.Float32(0) for _ in range_constexpr(3)]
                if const_expr(cache_row_scales):
                    cached_rows = [rlds.load(address_bytes=row_addresses[kb])[0] for kb in range_constexpr(3)]
                    rocdl.s_waitcnt(lgkmcnt=0)

                def dma(q, slot_index):
                    for turn in range_constexpr(slot_bytes // (threads * 16)):
                        address = dma_base + slot_index * slot_bytes + turn * threads * 16
                        dst = llvm.inttoptr(ir.Type.parse("!llvm.ptr<3>"), address.ir_value())
                        rocdl.raw_ptr_buffer_load_async_lds(
                            wrsrc, dst, fx.Int32(16).ir_value(), dma_offset.ir_value(),
                            fx.Int32(q * slot_bytes + turn * threads * 16).ir_value(), zero,
                            aux=ir.IntegerAttr.get(fx.Int32.ir_type, weight_cache_policy),
                        )
                    rocdl.asyncmark()

                def store_output(value, record, q):
                    rocdl.raw_ptr_buffer_store(value.ir_value(), orsrc, output_addresses[record].ir_value(),
                                              fx.Int32(q * (32768 if packed_output else 128)).ir_value(),
                                              aux=ir.IntegerAttr.get(fx.Int32.ir_type, output_cache_policy))

                def memory(q, slot_index, old, first=False, tail_left=100):
                    if const_expr(consumer_wait):
                        rocdl.wait_asyncmark(min(prefetch_distance - 1, tail_left - 1))
                        _stage_end()
                    _mark(f"MOE32_MEMORY_BEGIN_{slot_index % 2}")
                    b = fx.make_rmem_tensor([8, 2, 4], fx.Int32)
                    words = fx.make_view(fx.get_iter(b), fx.make_ordered_layout([4, 2, 2, 4], 0))
                    for kb in range_constexpr(4):
                        for ni in range_constexpr(2):
                            for part in range_constexpr(2):
                                words[None, part, ni, kb].store(blds.load(
                                    address_bytes=b_addresses[slot_index], offset_bytes=ni * 8192 + kb * 2048 + part * 512,
                                ))
                    asc = [cached_rows[kb] if cache_row_scales else rlds.load(address_bytes=row_addresses[kb])[0]
                           for kb in range_constexpr(2)]
                    route = cached_rows[2] if const_expr(cache_row_scales) else rlds.load(address_bytes=row_addresses[2])[0]
                    address = scale_base + fx.Int32(q // 2 * 8)
                    raw_scales = llvm.inline_asm(
                        ir.Type.parse("!llvm.struct<(i32, i32)>"), [address.ir_value()],
                        "s_mov_b32 $1, m0\ns_mov_b32 m0, $2\ns_nop 1\nds_read_addtid_b32 $0\ns_mov_b32 m0, $1",
                        "=&v,=&s,s,~{memory}", has_side_effects=True,
                    )
                    scale = fx.Int32(llvm.extractvalue(fx.Int32.ir_type, raw_scales, [0]))
                    rocdl.sched_barrier(0)
                    if const_expr(not first):
                        for record in range_constexpr(4):
                            store_output(old[record], record, q - 2)
                    if const_expr(tail_left > prefetch_distance):
                        dma(q + prefetch_distance, (slot_index + prefetch_distance) % 4)
                    if const_expr(not consumer_wait and tail_left > 1):
                        rocdl.wait_asyncmark(min(prefetch_distance - 1, tail_left - 2))
                    rocdl.s_waitcnt(lgkmcnt=0)
                    for kb in range_constexpr(4):
                        for ni in range_constexpr(2):
                            for part in range_constexpr(2):
                                view = words[None, part, ni, kb]
                                view.store(_pin_packet(view.load()))
                    _mark(f"MOE32_MEMORY_END_{slot_index % 2}")
                    if const_expr(not consumer_wait):
                        _stage_end()
                    return b, scale, asc, route

                def compute(b, scale, asc, route, previous, phase, first=False):
                    if const_expr(skip_empty_waves):
                        # All-invalid waves retain every VMEM event/barrier.
                        # Their C values cannot be observed: every output
                        # offset is the descriptor-end sentinel. An SSA if
                        # created spills that invalidate asyncmark accounting.
                        rocdl.sched_barrier(0)
                        llvm.inline_asm(ir.Type.parse("!llvm.void"), [wave_has_rows.ir_value()],
                                        "s_cmp_eq_u32 $0, 0\ns_cbranch_scc1 1f", "s,~{scc},~{memory}", has_side_effects=True)
                        rocdl.sched_barrier(0)
                    _mark(f"MOE32_COMPUTE_BEGIN_{phase % 2}")
                    coefficients = [fx.Int32(rocdl.readlane(fx.Int32.ir_type, scale.ir_value(), fx.Int32(kb).ir_value())).bitcast(fx.Float32)
                                    for kb in range_constexpr(2)]
                    factors = [(asc[kb] * coefficients[kb]) * route if fold_routing else asc[kb] * coefficients[kb]
                               for kb in range_constexpr(2)]
                    rocdl.sched_barrier(0)
                    accum = fx.make_rmem_tensor([16, 2], fx.Float32)
                    partial = [fx.make_rmem_tensor([16, 2], fx.Float32) for _ in range_constexpr(2)]
                    deferred_sums = [[], []]
                    scaled = [[], []]
                    words = [[], []]
                    swaps = [[], []]
                    packed_words = []
                    coalesced = [[], [], [], []]
                    dequant, pending = [], []
                    pack_index = 0

                    def pack_op(index):
                        ni, local_op = index // pack_pair_ops, index % pack_pair_ops
                        op = local_op - (16 if exact_deferred else 0) + (16 if fold_routing and not defer_k1 else 0)
                        if const_expr(index >= base_pack_ops + 8):
                            local = index - base_pack_ops - 8
                            pair, odd, word = local // 8, local % 8 // 4, local % 4
                            coalesced[odd * 2 + pair].append(_coalesce_word(packed_words[pair][word], packed_words[pair + 2][word], odd))
                        elif const_expr(index >= base_pack_ops):
                            ni, word = (index - base_pack_ops) // 4, (index - base_pack_ops) % 4
                            pair = _swap_words(packed_words[ni * 2][word], packed_words[ni * 2 + 1][word], 16)
                            packed_words[ni * 2][word], packed_words[ni * 2 + 1][word] = pair
                        elif const_expr(exact_deferred and local_op < 16):
                            deferred_sums[ni].append(fx.Float32(fxh.eltwise_op(
                                "llvm.fma.f32", previous[ni + 2][local_op], previous[4], previous[ni][local_op],
                            )))
                        elif const_expr(op < 16):
                            value = (deferred_sums[ni][op] * route if const_expr(exact_deferred) else
                                     fx.Float32(fxh.eltwise_op("llvm.fma.f32", previous[ni + 2][op], previous[4], previous[ni][op]))
                                     if const_expr(defer_k1) else previous[ni][op] * route)
                            scaled[ni].append(value)
                        elif const_expr(op < 24):
                            wi = op - 16
                            pair = [previous[ni][wi * 2], previous[ni][wi * 2 + 1]] if fold_routing and not defer_k1 else [scaled[ni][wi * 2], scaled[ni][wi * 2 + 1]]
                            words[ni].append(fx.Vector.from_elements(pair, fx.Float32).to(fx.BFloat16).bitcast(fx.Int32)[0])
                            if const_expr(raw_output and wi % 4 == 3):
                                packed_words.append(words[ni][wi - 3:wi + 1])
                        else:
                            wi = op - 24
                            group_index, word_index = wi // 2, wi % 2
                            swaps[ni].append(_swap_words(words[ni][group_index * 4 + word_index], words[ni][group_index * 4 + word_index + 2], 32))
                            if const_expr(wi % 2 == 1):
                                lo, hi = swaps[ni][-2:]
                                packed_words.append([lo[0], hi[0], lo[1], hi[1]])

                    def retire(item):
                        kb, ni, element = item
                        value = partial[kb][None, ni].load()[element]
                        if const_expr(kb == 0):
                            accum[element, ni] = value * factors[kb]
                        else:
                            accum[element, ni] = fx.Float32(fxh.eltwise_op("llvm.fma.f32", value, factors[kb], accum[element, ni]))

                    for index in range_constexpr(8):
                        kb, sub, ni = index // 4, index % 4 // 2, index % 2
                        frag = partial[kb][None, ni]
                        if const_expr(sub == 0):
                            frag.fill(0)
                        fx.gemm(atom, frag, b[None, ni, kb * 2 + sub], a[None, kb * 2 + sub], frag)
                        rocdl.sched_barrier(0)
                        if const_expr(sub == 1 and (not defer_k1 or kb == 0)):
                            pending.append((index + 2, kb, ni))
                        while const_expr(len(pending) > 0 and pending[0][0] <= index):
                            _, ready_kb, ready_ni = pending.pop(0)
                            dequant.extend([(ready_kb, ready_ni, element) for element in range_constexpr(16)])
                        remaining = valu_per_mfma
                        while const_expr(len(dequant) > 0 and remaining > 0):
                            retire(dequant.pop(0))
                            remaining -= 1
                        while const_expr(not first and pack_index < pack_count and remaining > 0):
                            pack_op(pack_index)
                            pack_index += 1
                            remaining -= 1
                        rocdl.sched_barrier(0)
                    for _, kb, ni in pending:
                        dequant.extend([(kb, ni, element) for element in range_constexpr(16)])
                    for item in dequant:
                        retire(item)
                    while const_expr(not first and pack_index < pack_count):
                        pack_op(pack_index)
                        pack_index += 1
                    packed = [_pin_packet(fx.Vector.from_elements(record, fx.Int32)) for record in (packed_words if raw_output else coalesced)] if const_expr(not first) else []
                    # Keep deferred producers inside this Compute/skip guard;
                    # otherwise LLVM can sink the penultimate tile past Memory.
                    result = [_pin_packet(accum[None, ni].load()) if exact_deferred else accum[None, ni].load()
                              for ni in range_constexpr(2)]
                    if const_expr(defer_k1):
                        result.extend([_pin_packet(partial[1][None, ni].load()) if exact_deferred else partial[1][None, ni].load()
                                       for ni in range_constexpr(2)])
                        result.append(_pin_address(factors[1].bitcast(fx.Int32)).bitcast(fx.Float32)
                                      if exact_deferred else factors[1])
                    _mark(f"MOE32_COMPUTE_END_{phase % 2}")
                    if const_expr(skip_empty_waves):
                        rocdl.sched_barrier(0)
                        llvm.inline_asm(ir.Type.parse("!llvm.void"), [], "1:", "~{memory}", has_side_effects=True)
                        rocdl.sched_barrier(0)
                    if const_expr(not consumer_wait):
                        _stage_end()
                    return result, packed

                rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0)
                for q in range_constexpr(min(prefetch_distance, steps)):
                    dma(q, q)
                if const_expr(not consumer_wait):
                    rocdl.wait_asyncmark(min(prefetch_distance, steps) - 1)
                    _stage_end()
                if const_expr(block_m == 256):
                    if group == 1:
                        _stage_end()
                initial_packed = [fx.Vector.filled(4, 0, fx.Int32) for _ in range_constexpr(4)]
                initial_c = [fx.Vector.filled(16, 0.0, fx.Float32) for _ in range_constexpr(4 if defer_k1 else 2)]
                if const_expr(defer_k1):
                    initial_c.append(fx.Float32(0))
                b, scale, asc, route = memory(0, 0, initial_packed, first=True, tail_left=steps)
                current, packed = compute(b, scale, asc, route, initial_c, 0, first=True)
                for q in range_constexpr(1, min(4, steps)):
                    b, scale, asc, route = memory(q, q, packed, first=q == 1, tail_left=steps - q)
                    current, packed = compute(b, scale, asc, route, current, q)
                loop_end = max(4, (steps - prefetch_distance) // 4 * 4)
                if const_expr(steps > 4):
                    state0 = [*current, *packed]
                    for q, state in range(fx.Int32(4), fx.Int32(loop_end), fx.Int32(4), init=state0):
                        current = [fx.Vector(state[i]) for i in range_constexpr(4 if defer_k1 else 2)]
                        if const_expr(defer_k1):
                            current.append(fx.Float32(state[4]))
                        packed = [fx.Vector(state[i + state_c_count]) for i in range_constexpr(4)]
                        for phase in range_constexpr(4):
                            b, scale, asc, route = memory(q + phase, phase, packed)
                            current, packed = compute(b, scale, asc, route, current, phase)
                        final = yield [*current, *packed]
                    current = [fx.Vector(final[i]) for i in range_constexpr(4 if defer_k1 else 2)]
                    if const_expr(defer_k1):
                        current.append(fx.Float32(final[4]))
                    packed = [fx.Vector(final[i + state_c_count]) for i in range_constexpr(4)]
                    for q in range_constexpr(loop_end, steps):
                        b, scale, asc, route = memory(q, q % 4, packed, tail_left=steps - q)
                        current, packed = compute(b, scale, asc, route, current, q)
                for record in range_constexpr(4):
                    store_output(packed[record], record, steps - 2)
                final_words = []
                for ni in range_constexpr(2):
                    completed = (fx.Vector(fxh.eltwise_op("llvm.fma.f32", current[ni + 2], current[4], current[ni]))
                                 if defer_k1 else current[ni])
                    final_value = completed if fold_routing else completed * route
                    words = fx.Vector(final_value).to(fx.BFloat16).bitcast(fx.Int32)
                    for pair in range_constexpr(2):
                        if const_expr(raw_output):
                            final_words.append(_pin_packet(fx.Vector.from_elements([words[pair * 4 + i] for i in range_constexpr(4)], fx.Int32)))
                        else:
                            lo = _swap_words(words[pair * 4], words[pair * 4 + 2], 32)
                            hi = _swap_words(words[pair * 4 + 1], words[pair * 4 + 3], 32)
                            final_words.append(_pin_packet(fx.Vector.from_elements([lo[0], hi[0], lo[1], hi[1]], fx.Int32)))
                if const_expr(not raw_output):
                    for ni in range_constexpr(2):
                        pairs = [_swap_words(final_words[ni * 2][word], final_words[ni * 2 + 1][word], 16) for word in range_constexpr(4)]
                        final_words[ni * 2] = _pin_packet(fx.Vector.from_elements([pair[0] for pair in pairs], fx.Int32))
                        final_words[ni * 2 + 1] = _pin_packet(fx.Vector.from_elements([pair[1] for pair in pairs], fx.Int32))
                    pair0 = _coalesce_output_pairs(final_words[0], final_words[2])
                    pair1 = _coalesce_output_pairs(final_words[1], final_words[3])
                    final_words = [pair0[0], pair1[0], pair0[1], pair1[1]]
                for record in range_constexpr(4):
                    store_output(final_words[record], record, steps - 1)
                rocdl.wait_asyncmark(0)
                rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0)
                _stage_end()
                if const_expr(block_m == 256):
                    if group == 0:
                        _stage_end()
            fx.barrier()
            if const_expr(not persistent):
                active = fx.Boolean(False)

    @flyc.jit
    def launch(
        output: fx.Pointer, input_q: fx.Pointer, weight: fx.Pointer,
        input_scales: fx.Pointer, weight_scales: fx.Pointer,
        sorted_ids: fx.Pointer, sorted_weights: fx.Pointer, expert_ids: fx.Pointer,
        valid_ids: fx.Pointer, counter: fx.Pointer,
        tokens: fx.Int32, expert_blocks: fx.Int32, stream: fx.Stream,
    ):
        moe_down_mfma32_kernel(output, input_q, weight, input_scales, weight_scales, sorted_ids,
                             sorted_weights, expert_ids, valid_ids, counter, tokens, expert_blocks,
                             value_attrs=attrs).launch(
                                 grid=(persistent_workgroups if const_expr(persistent) else expert_blocks * m_parts * num_oc_splits, 1, 1),
                                 block=(threads, 1, 1), stream=stream,
                             )

    def callable(
        output: torch.Tensor, input_q: torch.Tensor, weight: torch.Tensor,
        input_scales: torch.Tensor, weight_scales: torch.Tensor,
        sorted_ids: torch.Tensor, sorted_weights: torch.Tensor, expert_ids: torch.Tensor,
        valid_ids: torch.Tensor, counter: torch.Tensor,
    ):
        tokens, actual_topk, actual_k = input_q.shape
        tensors = (output, input_q, weight, input_scales, weight_scales,
                   sorted_ids, sorted_weights, expert_ids, valid_ids, counter)
        assert (actual_topk, actual_k) == (topk, k)
        assert weight.shape == (num_experts, n, k)
        expected_shape = ((expert_ids.numel() * 256, n) if sorted_output else (tokens, topk, n))
        assert output.shape == expected_shape, f"{output_layout} output requires {expected_shape}, got {tuple(output.shape)}"
        assert input_q.dtype == weight.dtype == torch.float8_e4m3fn
        assert output.dtype == torch.bfloat16
        assert input_scales.dtype == weight_scales.dtype == sorted_weights.dtype == torch.float32
        assert sorted_ids.dtype == expert_ids.dtype == valid_ids.dtype == counter.dtype == torch.int32
        assert input_scales.numel() == tokens * topk * 2 and weight_scales.shape == (num_experts, n // 128, 2)
        assert sorted_ids.shape == sorted_weights.shape and counter.numel() == 1 and valid_ids.numel() >= 1
        assert all(t.is_cuda and t.is_contiguous() and t.device == output.device for t in tensors)
        assert 0 < tokens < (1 << 24) and (output.numel() + n) * 2 < (1 << 32)
        assert input_q.numel() < (1 << 32) and input_scales.numel() * 4 < (1 << 32) and n_split * k < (1 << 32)
        assert torch.cuda.get_device_properties(output.device).gcnArchName.startswith("gfx950")
        if persistent:
            counter.zero_()
        stream = torch.cuda.current_stream(output.device)
        compiled = getattr(launch, "_cf", None)
        if compiled is None:
            _run_compiled(launch, *[_ptr(t) for t in tensors], fx.Int32(tokens), fx.Int32(expert_ids.numel()), fx.Stream(stream.cuda_stream))
        else:
            compiled(*(t.data_ptr() for t in tensors), tokens, expert_ids.numel(), stream.cuda_stream)
        return output

    callable.config = {
        "block_m": block_m, "block_n": 128, "num_oc_splits": num_oc_splits, "num_waves": threads // 64,
        "persistent_workgroups": persistent_workgroups, "stages": 4,
        "prefetch_distance": prefetch_distance, "output_cache_policy": output_cache_policy,
        "weight_cache_policy": weight_cache_policy, "coalesced_output": not raw_output,
        "weight_layout": (32, 16), "mfma": (32, 32, 64),
        "fold_routing": fold_routing, "defer_k1": defer_k1,
        "skip_empty_waves": skip_empty_waves, "cache_row_scales": cache_row_scales,
        "task_m_group": task_m_group,
        "output_layout": output_layout,
        "persistent": persistent, "xcd_swizzle": xcd_swizzle, "xcd_count": xcd_count,
        "valu_per_mfma": valu_per_mfma,
        "consumer_wait": consumer_wait,
    }
    return callable