# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Winning gfx950 FP8 MoE down: M256/N128/K256, OC4, PF3, packed BF16.

Eight waves per CTA form two staggered four-wave groups. Each N128 tile
uses two Memory/Compute pairs: direct-global-to-LDS N64/K256 packets,
sixteen MFMA16x16x128 instructions per Compute, and register-only packing
of the completed other half. The next tile's Memory stages retire the
previous output equally. A, scales and routing are cached per task.

Only persistent256 or independent width8 scheduling is supported, with
SC1 weights/NT output. Layout is [expert_block256, global_N64, row256, col64],
stored in a caller-owned 2-D allocation; consume it with the packed reducer.
"""

from functools import cache

import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, range_constexpr, rocdl

from pyhip.contrib.flydsl import helpers as fxh
from pyhip.contrib.flydsl.moe_gemm_2stage.common import torch_tensor_to_pointer as _ptr


ATT_TUNED_BN128_CONFIG = {
    "block_m": 256, "block_n": 128, "sort_block_m": 256, "num_waves": 8, "num_oc_splits": 4,
    "prefetch_distance": 3, "persistent_workgroups": 256,
    "output_cache_policy": 2, "weight_cache_policy": 16,
    "output_layout": "packed", "persistent": True,
}


def _scalar(value):
    return fx.Int32(rocdl.readfirstlane(fx.Int32.ir_type, value.ir_value()))


def _pin_address(value):
    return fx.Int32(llvm.inline_asm(
        fx.Int32.ir_type, [value.ir_value()], "", "=v,0", has_side_effects=True,
    ))


def _task_thread_id(wave):
    # Rebuild lane ID per task instead of spilling the entry workitem VGPR.
    lane = fx.Int32(llvm.inline_asm(
        fx.Int32.ir_type, [],
        "v_mbcnt_lo_u32_b32 $0, -1, 0\nv_mbcnt_hi_u32_b32 $0, -1, $0",
        "=v", has_side_effects=True,
    ))
    return wave * 64 + lane


def _pin_packet(value):
    raw = value.ir_value()
    return fx.Vector(llvm.inline_asm(raw.type, [raw], "", "=v,0", has_side_effects=True))


def _coalesce_output_pairs(first, second):
    """Exchange a row bit with a column bit for contiguous 128-byte stores."""
    mask, even = fx.Uint64(0xAAAAAAAAAAAAAAAA), fx.Uint64(0x5555555555555555)
    assembly = ["s_mov_b64 vcc, $16"]
    assembly.extend(f"v_cndmask_b32_dpp ${i}, ${12+i}, ${8+i}, vcc quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf"
                    for i in range(4))
    assembly.append("s_mov_b64 vcc, $17")
    assembly.extend(f"v_cndmask_b32_dpp ${4+i}, ${8+i}, ${12+i}, vcc quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf"
                    for i in range(4))
    result = llvm.inline_asm(
        ir.Type.parse("!llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32)>"),
        [first[i].ir_value() for i in range(4)] + [second[i].ir_value() for i in range(4)]
        + [even.ir_value(), mask.ir_value()],
        "\n".join(assembly), ",".join(["=&v"] * 8 + ["v"] * 8 + ["s", "s", "~{vcc}"]),
        has_side_effects=True,
    )
    return [_pin_packet(fx.Vector.from_elements([
        fx.Int32(llvm.extractvalue(fx.Int32.ir_type, result, [side * 4 + i])) for i in range(4)
    ], fx.Int32)) for side in range(2)]


def _stage_end():
    rocdl.sched_barrier(0)
    rocdl.s_barrier()
    rocdl.sched_barrier(0)


def _priority(value):
    rocdl.sched_barrier(0)
    rocdl.s_setprio(value)
    rocdl.sched_barrier(0)


def _mark(text):
    rocdl.sched_barrier(0)
    llvm.inline_asm(ir.Type.parse("!llvm.void"), [], f"; {text}", "", has_side_effects=True)
    rocdl.sched_barrier(0)


def _pack_pair(c0, c1, route):
    words0 = (c0 * route).to(fx.BFloat16).bitcast(fx.Int32)
    words1 = (c1 * route).to(fx.BFloat16).bitcast(fx.Int32)
    pair_type = ir.Type.parse("!llvm.struct<(i32, i32)>")
    lo = rocdl.permlane16_swap(pair_type, words0[0].ir_value(), words1[0].ir_value(), False, False)
    hi = rocdl.permlane16_swap(pair_type, words0[1].ir_value(), words1[1].ir_value(), False, False)
    return _pin_packet(fx.Vector.from_elements([
        fx.Int32(llvm.extractvalue(fx.Int32.ir_type, lo, [0])),
        fx.Int32(llvm.extractvalue(fx.Int32.ir_type, hi, [0])),
        fx.Int32(llvm.extractvalue(fx.Int32.ir_type, lo, [1])),
        fx.Int32(llvm.extractvalue(fx.Int32.ir_type, hi, [1])),
    ], fx.Int32))


@cache
def flydsl_moe_gemm_8wave_down(*, n, k=256, topk, num_experts, persistent=True):
    """Build the fixed winning packed down; return the ten-tensor callable.

    A: OCP E4M3FN [tokens,topk,256], A scales: physical K-major FP32.
    B: OCP E4M3FN [experts,N,256], shuffle_weight(layout=(16,16)).
    B scales: [experts,N/128,2]. Sorting uses M256 padded expert runs.
    Output: BF16 [expert_capacity*256,N], physically packed, NOT row-major.
    Persistent calls reset the counter; independent calls preserve it.
    Independent width8 swizzle changes task order, not the eight-wave pipeline.
    """
    assert k == 256 and n > 0 and n % 512 == 0
    assert 0 < topk <= min(num_experts, 255)
    n_split, ks, nt = n // 4, 2, n // 512
    slot_bytes, ring_slots = 16384, 4
    scale_count = n_split // 128 * ks
    scale_words = (scale_count + 63 + 255) // 256 * 256
    ids_offset = scale_words + ring_slots * slot_bytes // 4
    row_offset = ids_offset + 256
    task_offset = row_offset + 256 * (ks + 1)
    assert (task_offset + 1) * 4 <= 160 * 1024
    loop_end = 2 + max(0, (nt - 4) // 2) * 2
    kernel_attrs = {
        "rocdl.waves_per_eu": 2,
        "rocdl.flat_work_group_size": "512,512",
        "llvm.passthrough": [["target-features", "-packed-fp32-ops"], ["amdgpu-agpr-alloc", "0,0"]],
    }

    @fx.struct
    class SharedStorage:
        # Final63 scale words make the last ds_read_addtid window safe.
        arena: fx.Array[fx.Int32, task_offset + 1, 16]

    @flyc.kernel(known_block_size=[512, 1, 1])
    def moe_down_8stage_kernel(
        output: fx.Pointer, input_q: fx.Pointer, weight_shuffled: fx.Pointer,
        input_scales: fx.Pointer, weight_scales: fx.Pointer,
        sorted_ids: fx.Pointer, sorted_weights: fx.Pointer, sorted_expert_ids: fx.Pointer,
        num_valid_ids: fx.Pointer, task_counter: fx.Pointer,
        num_tokens: fx.Int32, output_capacity_rows: fx.Int32,
    ):
        tid = fx.Int32(fx.thread_idx.x)
        group, scalar_wave = _scalar(tid // 256), _scalar(tid // 64)
        rows = num_tokens * topk
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        b_pointer = lds.arena.ptr + scale_words
        ids_lds = fx.make_view(lds.arena.ptr + ids_offset, fx.make_layout(256, 1))
        scale_pointer = fx.recast_iter(
            fx.PointerType.get(fx.Float32.ir_type, lds.arena.ptr.memspace, 16), lds.arena.ptr,
        )
        row_scale_pointer = scale_pointer + row_offset
        scales_lds = fx.make_view(scale_pointer, fx.make_layout(scale_count, 1))
        a_scales_lds = fx.make_view(row_scale_pointer, fx.make_layout(256 * ks, 1))
        routes_lds = fx.make_view(row_scale_pointer + 256 * ks, fx.make_layout(256, 1))
        task_lds = fx.make_view(lds.arena.ptr + task_offset, fx.make_layout(1, 1))
        b_lds = fxh.LdsTensor(fx.make_view(b_pointer, fx.make_layout(4, 1)))
        row_scale_lds = fxh.LdsTensor(fx.make_view(row_scale_pointer, fx.make_layout(1, 1)))
        dma_base = _scalar(fx.Int32(fx.ptrtoint(b_pointer))) + scalar_wave * 1024
        a_pointer = fx.recast_iter(fx.PointerType.get(fx.Int32.ir_type, input_q.memspace, 16), input_q)
        a_buffer = fx.rocdl.make_buffer_tensor(fx.make_view(a_pointer, fx.make_layout(rows * (k // 4), 1)), False)
        a_packet = fxh.BufferTensor(fx.make_view(fx.get_iter(a_buffer), fx.make_layout(4, 1)))
        as_buffer = fx.rocdl.make_buffer_tensor(fx.make_view(input_scales, fx.make_layout(rows * ks, 1)), False)
        as_rsrc = fx.rocdl.get_buffer_rsrc(fx.get_iter(as_buffer))
        out_buffer = fx.rocdl.make_buffer_tensor(fx.make_view(output, fx.make_layout(output_capacity_rows * n, 1)), False)
        out_rsrc = fx.rocdl.get_buffer_rsrc(fx.get_iter(out_buffer))
        max_id = _scalar(num_valid_ids[0])
        atom = fx.make_mma_atom(rocdl.cdna4.MFMA_Scale(16, 16, 128, fx.Float8E4M3FN))
        zero, dma_size = fx.Int32(0).ir_value(), fx.Int32(16).ir_value()
        lds_ptr_type = ir.Type.parse("!llvm.ptr<3>")

        linear_task = fx.Int32(fx.block_idx.x)
        task = fx.Int32(0)
        running = fx.Boolean(True)
        while running:
            task_tid = fx.Uint32(_task_thread_id(scalar_wave))
            if const_expr(persistent):
                if task_tid == 0:
                    counter_ptr = llvm.inttoptr(ir.Type.parse("!llvm.ptr<1>"), fx.ptrtoint(task_counter).ir_value())
                    next_task = fx.Int32(llvm.AtomicRMWOp(
                        llvm.AtomicBinOp.add, counter_ptr, fx.Int32(1).ir_value(),
                        llvm.AtomicOrdering.monotonic, syncscope="agent",
                    ).res)
                    task_lds[0] = next_task
                fx.barrier()
                task = _scalar(task_lds[0])
            else:
                task = linear_task
                # Transpose only the valid prefix; preserve a bijective tail.
                chunk = (max_id // 256 * 4) // 8
                mapped = (task % 8) * chunk + task // 8
                task = (task < chunk * 8).select(mapped, task)
            blk_m, blk_oc = task // 4, task % 4
            running = blk_m * 256 < max_id
            if running:
                task_lane_row, task_lane_k, task_wave = task_tid % 16, (task_tid % 64) // 16, task_tid // 64
                dma_offset = _pin_address(task_tid * 16)
                row_begin = blk_m * 256
                expert = _scalar(sorted_expert_ids[blk_m])
                if task_tid < 256:
                    ids_lds[task_tid] = sorted_ids[row_begin + task_tid]
                    routes_lds[task_tid] = sorted_weights[row_begin + task_tid]
                for copy_round in range_constexpr((scale_count + 511) // 512):
                    index = task_tid + copy_round * 512
                    if index < scale_count:
                        scales_lds[index] = weight_scales[expert * (n // 128 * ks) + blk_oc * scale_count + index]
                fx.barrier()
                if task_tid < 256:
                    encoded = ids_lds[task_tid].bitcast(fx.Uint32)
                    token, slot = encoded & 0xFFFFFF, encoded >> 24
                    valid = (token < fx.Uint32(num_tokens)) & (slot < topk)
                    input_row = fx.Int32(token) * topk + fx.Int32(slot)
                    for kb in range_constexpr(ks):
                        scale_offset = valid.select((kb * rows + input_row) * 4, fx.Int32(-1))
                        a_scales_lds[kb * 256 + task_tid] = fx.Float32(rocdl.raw_ptr_buffer_load(
                            fx.Float32.ir_type, as_rsrc, scale_offset.ir_value(), zero,
                        ))
                fx.barrier()
                weight_view = fx.make_view(
                    weight_shuffled + fx.Int64(expert) * (n * k) + fx.Int64(blk_oc) * (n_split * k),
                    fx.make_layout(n_split * k, 1),
                )
                weight_buffer = fx.rocdl.make_buffer_tensor(weight_view, False)
                weight_rsrc = fx.rocdl.get_buffer_rsrc(fx.get_iter(weight_buffer))
                a = fx.make_rmem_tensor([8, 2, ks], fx.Int32)
                a_words = fx.make_view(fx.get_iter(a), fx.make_ordered_layout([4, 2, 2, ks], 0))
                stage_a_scales = fx.make_rmem_tensor([2, ks], fx.Float32)
                stage_routes = fx.make_rmem_tensor(2, fx.Float32)
                c = fx.make_rmem_tensor([4, 2, 8], fx.Float32)
                swap_col = (task_lane_k & 1) * 2 + (task_lane_k >> 1)
                for mi in range_constexpr(2):
                    row = task_wave * 32 + mi * 16 + task_lane_row
                    encoded = ids_lds[row].bitcast(fx.Uint32)
                    token, slot = encoded & 0xFFFFFF, encoded >> 24
                    valid = (token < fx.Uint32(num_tokens)) & (slot < topk)
                    input_row = fx.Int32(token) * topk + fx.Int32(slot)
                    for kb in range_constexpr(ks):
                        for part in range_constexpr(2):
                            offset = valid.select(input_row * k + task_lane_k * 16 + kb * 128 + part * 64, fx.Int32(-1))
                            a_words[None, part, mi, kb].store(a_packet.load(voffset_bytes=offset))
                coalesced_addresses = []
                for mi in range_constexpr(2):
                    addresses = []
                    for parity in range_constexpr(2):
                        row = task_wave * 32 + mi * 16 + (task_lane_row // 2) * 2 + parity
                        encoded = ids_lds[row].bitcast(fx.Uint32)
                        token, slot = encoded & 0xFFFFFF, encoded >> 24
                        valid = (token < fx.Uint32(num_tokens)) & (slot < topk)
                        offset = blk_m * (256 * n * 2) + blk_oc * (256 * n_split * 2) + row * 128
                        offset += swap_col * 16 + (task_lane_row % 2) * 64
                        addresses.append(_pin_address(valid.select(offset, output_capacity_rows * (n * 2))))
                    coalesced_addresses.append(addresses)

                # All VGPR addresses are pinned before the first Memory stage.
                row = task_wave * 32 + task_lane_row
                b_lane = task_lane_row * 16 + task_lane_k * 256
                b_addresses = [_pin_address(fx.Int32(fx.ptrtoint(b_pointer)) + b_lane + slot * slot_bytes)
                               for slot in range_constexpr(ring_slots)]
                row_addresses = [_pin_address(fx.Int32(fx.ptrtoint(row_scale_pointer)) + (kb * 256 + row) * 4)
                                 for kb in range_constexpr(ks + 1)]
                scalar_scale_base = _scalar(fx.Int32(fx.ptrtoint(scale_pointer)))

                def read_scale(index):
                    address = scalar_scale_base + fx.Int32(index) * 4
                    # M0 requires two wait states; preserve it for direct-LDS DMA.
                    result = llvm.inline_asm(
                        ir.Type.parse("!llvm.struct<(i32, i32)>"), [address.ir_value()],
                        "s_mov_b32 $1, m0\ns_mov_b32 m0, $2\ns_nop 1\nds_read_addtid_b32 $0\ns_mov_b32 m0, $1",
                        "=&v,=&s,s,~{memory}", has_side_effects=True,
                    )
                    return fx.Int32(llvm.extractvalue(fx.Int32.ir_type, result, [0]))

                def dma_b(n_tile, step):
                    target_n, half = n_tile + step // 2, step % 2
                    slot = (target_n * 2 + half) % ring_slots
                    soffset = fx.Int32(target_n * (128 * k) + half * (64 * k))
                    for copy_round in range_constexpr(2):
                        address = dma_base + slot * slot_bytes + copy_round * 8192
                        dst = llvm.inttoptr(lds_ptr_type, address.ir_value())
                        rocdl.raw_ptr_buffer_load_async_lds(
                            weight_rsrc, dst, dma_size, dma_offset.ir_value(),
                            (soffset + copy_round * 8192).ir_value(), zero,
                            aux=ir.IntegerAttr.get(fx.Int32.ir_type, 16),
                        )
                    rocdl.asyncmark()

                def read_b(n_tile, step, ring_phase):
                    b = fx.make_rmem_tensor([8, 4, ks], fx.Int32)
                    words = fx.make_view(fx.get_iter(b), fx.make_ordered_layout([4, 2, 4, ks], 0))
                    scale = fx.make_rmem_tensor(ks, fx.Int32)
                    for kb in range_constexpr(ks):
                        for ni in range_constexpr(4):
                            for part in range_constexpr(2):
                                words[None, part, ni, kb].store(b_lds.load(
                                    address_bytes=b_addresses[(ring_phase * 2 + step) % ring_slots],
                                    offset_bytes=ni * (16 * k) + kb * 2048 + part * 1024,
                                ))
                        scale[kb] = read_scale(n_tile * ks + kb)
                        for mi in range_constexpr(2):
                            stage_a_scales[mi, kb] = row_scale_lds.load(address_bytes=row_addresses[kb], offset_bytes=mi * 64)[0]
                    for mi in range_constexpr(2):
                        stage_routes[mi] = row_scale_lds.load(address_bytes=row_addresses[ks], offset_bytes=mi * 64)[0]
                    return b, scale

                def place_b(b):
                    words = fx.make_view(fx.get_iter(b), fx.make_ordered_layout([4, 16], 0))
                    for packet in range_constexpr(16):
                        words[None, packet].store(_pin_packet(words[None, packet].load()))

                def pack_record(record):
                    return [_pack_pair(c[None, mi, record * 2].load(), c[None, mi, record * 2 + 1].load(), stage_routes[mi])
                            for mi in range_constexpr(2)]

                def store_quarter(n_tile, packed, quarter):
                    mi, half = quarter % 2, quarter // 2
                    for local_record in range_constexpr(2):
                        record = half * 2 + local_record
                        rocdl.raw_ptr_buffer_store(
                            packed[record][mi].ir_value(), out_rsrc,
                            coalesced_addresses[mi][local_record].ir_value(),
                            fx.Int32((n_tile * 2 + half) * 32768).ir_value(),
                            aux=ir.IntegerAttr.get(fx.Int32.ir_type, 2),
                        )

                def compute_stage(b, scale, step, has_pack):
                    pack_id = 1 - step
                    coefficients = [_scalar(scale[kb]).bitcast(fx.Float32) for kb in range_constexpr(ks)]
                    factors = [[stage_a_scales[mi, kb] * coefficients[kb] for mi in range_constexpr(2)]
                               for kb in range_constexpr(ks)]
                    pack_scaled = [[] for _ in range_constexpr(4)]
                    pack_words = [[] for _ in range_constexpr(4)]
                    pack_swaps = [[] for _ in range_constexpr(4)]
                    packed, pending, dequant = [[], []], [], []
                    pack_index = 0

                    def pack_op(index):
                        pair_index, op = index // 14, index % 14
                        mi, pair = pair_index // 2, pair_index % 2
                        if const_expr(op < 8):
                            ni = pack_id * 4 + pair * 2 + op // 4
                            pack_scaled[pair_index].append(c[None, mi, ni].load()[op % 4] * stage_routes[mi])
                        elif const_expr(op < 12):
                            wi = op - 8
                            values = fx.Vector.from_elements([pack_scaled[pair_index][wi * 2], pack_scaled[pair_index][wi * 2 + 1]], fx.Float32)
                            pack_words[pair_index].append(values.to(fx.BFloat16).bitcast(fx.Int32)[0])
                        else:
                            wi, words = op - 12, pack_words[pair_index]
                            pack_swaps[pair_index].append(rocdl.permlane16_swap(
                                ir.Type.parse("!llvm.struct<(i32, i32)>"), words[wi].ir_value(), words[wi + 2].ir_value(), False, False,
                            ))
                            if const_expr(op == 13):
                                lo, hi = pack_swaps[pair_index]
                                packed[mi].append(_pin_packet(fx.Vector.from_elements([
                                    fx.Int32(llvm.extractvalue(fx.Int32.ir_type, lo, [0])),
                                    fx.Int32(llvm.extractvalue(fx.Int32.ir_type, hi, [0])),
                                    fx.Int32(llvm.extractvalue(fx.Int32.ir_type, lo, [1])),
                                    fx.Int32(llvm.extractvalue(fx.Int32.ir_type, hi, [1])),
                                ], fx.Int32)))

                    def retire_one(item):
                        old, mi, ni, kb, element = item
                        value, factor = old.load()[element], factors[kb][mi]
                        if const_expr(kb == 0):
                            c[element, mi, ni] = value * factor
                        else:
                            c[element, mi, ni] = fx.Float32(fxh.eltwise_op("llvm.fma.f32", value, factor, c[element, mi, ni]))

                    rocdl.sched_barrier(0)
                    for index in range_constexpr(16):
                        kb, local = index // 8, index % 8
                        mi, ni = local // 4, local % 4
                        c_ni = step * 4 + ni
                        partial = fx.make_rmem_tensor(4, fx.Float32)
                        partial.fill(0)
                        fx.gemm(atom, partial, b[None, ni, kb], a[None, mi, kb], partial)
                        rocdl.sched_barrier(0)
                        remaining = 7
                        pending.append((partial, mi, c_ni, kb))
                        if const_expr(len(pending) > 2):
                            old, old_mi, old_ni, old_kb = pending.pop(0)
                            dequant.extend([(old, old_mi, old_ni, old_kb, element) for element in range_constexpr(4)])
                        for _ in range_constexpr(min(4, remaining, len(dequant))):
                            retire_one(dequant.pop(0))
                            remaining -= 1
                        while const_expr(has_pack and pack_index < 56 and remaining > 0):
                            pack_op(pack_index)
                            remaining -= 1
                            pack_index += 1
                        while const_expr(remaining > 0 and len(dequant) > 0):
                            retire_one(dequant.pop(0))
                            remaining -= 1
                        rocdl.sched_barrier(0)
                    for old, old_mi, old_ni, old_kb in pending:
                        dequant.extend([(old, old_mi, old_ni, old_kb, element) for element in range_constexpr(4)])
                    for item in dequant:
                        retire_one(item)
                    while const_expr(has_pack and pack_index < 56):
                        pack_op(pack_index)
                        pack_index += 1
                    if const_expr(has_pack):
                        for mi in range_constexpr(2):
                            packed[mi] = _coalesce_output_pairs(packed[mi][0], packed[mi][1])
                    return packed

                def run_tile(n_tile, previous, first=False, ring_phase=0):
                    packed = []
                    tiles_left = nt - n_tile if const_expr(isinstance(n_tile, int)) else 3
                    ring_phase = n_tile if const_expr(isinstance(n_tile, int)) else ring_phase
                    for step in range_constexpr(2):
                        _mark(f"MOE8_MEMORY_BEGIN_{step}")
                        b, scale = read_b(n_tile, step, ring_phase)
                        rocdl.sched_barrier(0)
                        if const_expr(not first):
                            for mi in range_constexpr(2):
                                store_quarter(n_tile - 1, previous, step * 2 + mi)
                            rocdl.sched_barrier(0)
                        if const_expr(step + 3 < 2 * tiles_left):
                            dma_b(n_tile, step + 3)
                        if const_expr(step + 1 < 2 * tiles_left):
                            rocdl.wait_asyncmark(min(2, 2 * tiles_left - step - 2))
                        rocdl.s_waitcnt(lgkmcnt=0)
                        rocdl.sched_barrier(0)
                        place_b(b)
                        _mark(f"MOE8_MEMORY_END_{step}")
                        _stage_end()
                        _mark(f"MOE8_COMPUTE_BEGIN_{step}")
                        record = compute_stage(b, scale, step, not first or step == 1)
                        if const_expr(not first or step == 1):
                            parts = [[record[mi][pair] for mi in range_constexpr(2)] for pair in range_constexpr(2)]
                            if const_expr(step == 1):
                                packed.extend(parts)
                            else:
                                previous.extend(parts)
                        _mark(f"MOE8_COMPUTE_END_{step}")
                        _stage_end()
                    return packed

                def save_state(packed):
                    state = [c[None, mi, ni].load() for mi in range_constexpr(2) for ni in range_constexpr(4, 8)]
                    state.extend([packed[record][mi] for record in range_constexpr(2) for mi in range_constexpr(2)])
                    return state

                def restore_state(state):
                    index = 0
                    for mi in range_constexpr(2):
                        for ni in range_constexpr(4, 8):
                            c[None, mi, ni].store(state[index])
                            index += 1
                    packed = []
                    for _ in range_constexpr(2):
                        record_data = []
                        for mi in range_constexpr(2):
                            record_data.append(fx.Vector(state[index]))
                            index += 1
                        packed.append(record_data)
                    return packed

                rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0)
                for step in range_constexpr(min(3, 2 * nt)):
                    dma_b(0, step)
                rocdl.wait_asyncmark(min(3, 2 * nt) - 1)
                _stage_end()
                if group == 1:
                    _stage_end()
                packed = run_tile(0, [], first=True)
                if const_expr(nt > 1):
                    packed = run_tile(1, packed)
                if const_expr(nt >= 3):
                    initial = save_state(packed)
                    for n_tile, state in range(fx.Int32(2), fx.Int32(loop_end), fx.Int32(2), init=initial):
                        previous = restore_state(state)
                        for phase in range_constexpr(2):
                            packed = run_tile(fx.Int32(n_tile) + phase, previous, ring_phase=2 + phase)
                            previous = packed
                        results = yield save_state(packed)
                    previous = restore_state(results)
                    for n_tile in range_constexpr(loop_end, nt):
                        packed = run_tile(n_tile, previous)
                        previous = packed
                _priority(3)
                for record in range_constexpr(2, 4):
                    packed.append(pack_record(record))
                coalesced = [_coalesce_output_pairs(packed[2][mi], packed[3][mi]) for mi in range_constexpr(2)]
                packed[2] = [coalesced[mi][0] for mi in range_constexpr(2)]
                packed[3] = [coalesced[mi][1] for mi in range_constexpr(2)]
                _priority(0)
                for quarter in range_constexpr(4):
                    store_quarter(nt - 1, packed, quarter)
                rocdl.wait_asyncmark(0)
                rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0)
                _stage_end()
                if group == 0:
                    _stage_end()
            fx.barrier()
            if const_expr(not persistent):
                running = fx.Boolean(False)

    @flyc.jit
    def launch(output: fx.Pointer, input_q: fx.Pointer, weight_shuffled: fx.Pointer,
               input_scales: fx.Pointer, weight_scales: fx.Pointer,
               sorted_ids: fx.Pointer, sorted_weights: fx.Pointer, sorted_expert_ids: fx.Pointer,
               num_valid_ids: fx.Pointer, task_counter: fx.Pointer,
               num_tokens: fx.Int32, output_capacity_rows: fx.Int32, stream: fx.Stream):
        moe_down_8stage_kernel(
            output, input_q, weight_shuffled, input_scales, weight_scales,
            sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, task_counter,
            num_tokens, output_capacity_rows, value_attrs=kernel_attrs,
        ).launch(grid=(256 if const_expr(persistent) else output_capacity_rows // 256 * 4, 1, 1),
             block=(512, 1, 1), stream=stream)

    def down(output, input_q, weight_shuffled, input_scales, weight_scales,
             sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, task_counter):
        assert input_q.ndim == 3
        tokens, input_topk, input_k = input_q.shape
        assert 0 < tokens < (1 << 24) and (input_topk, input_k) == (topk, k)
        assert weight_shuffled.shape == (num_experts, n, k)
        assert sorted_expert_ids.ndim == sorted_ids.ndim == 1
        assert output.ndim == 2 and output.shape[1] == n and output.shape[0] % 256 == 0
        assert output.shape[0] >= (sorted_ids.numel() + 255) // 256 * 256
        assert input_scales.numel() == tokens * topk * ks
        assert weight_scales.shape == (num_experts, n // 128, ks)
        assert sorted_weights.shape == sorted_ids.shape
        assert input_q.dtype == weight_shuffled.dtype == torch.float8_e4m3fn and output.dtype == torch.bfloat16
        assert input_scales.dtype == weight_scales.dtype == sorted_weights.dtype == torch.float32
        assert sorted_ids.dtype == sorted_expert_ids.dtype == num_valid_ids.dtype == task_counter.dtype == torch.int32
        assert num_valid_ids.numel() >= 1 and task_counter.numel() == 1
        tensors = (output, input_q, weight_shuffled, input_scales, weight_scales,
                   sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, task_counter)
        assert all(t.is_cuda and t.device == output.device and t.is_contiguous() for t in tensors)
        assert input_q.numel() < (1 << 32) and input_scales.numel() * 4 < (1 << 32) and n_split * k < (1 << 32)
        assert (output.numel() + n) * 2 < (1 << 32), "32-bit offsets include the invalid-row sentinel"
        assert torch.cuda.get_device_properties(output.device).gcnArchName.startswith("gfx950")
        if persistent:
            task_counter.zero_()
        stream = torch.cuda.current_stream(output.device)
        compiled = getattr(launch, "_cf", None)
        if compiled is None:
            _run_compiled(launch, *[_ptr(t) for t in tensors], fx.Int32(tokens),
                          fx.Int32(output.shape[0]), fx.Stream(stream.cuda_stream))
        else:
            compiled(*(t.data_ptr() for t in tensors), tokens, output.shape[0], stream.cuda_stream)
        return output

    down.config = {**ATT_TUNED_BN128_CONFIG, "stages": 4,
                   "persistent": persistent, "persistent_workgroups": 256 if persistent else 0,
                   "xcd_count": 0 if persistent else 8}
    return down