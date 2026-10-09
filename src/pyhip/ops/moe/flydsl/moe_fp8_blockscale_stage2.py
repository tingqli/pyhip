# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""gfx950 M128/4-wave FP8 Down with configurable K and OC splits.

Consume native sort128 or sort256 and write packed BF16 M128/N64 tiles.
K256 retains its PF3/paired pipeline; larger positive multiples of 256 stream
K256 chunks through a fixed double buffer. Persistent queues are caller-owned.
"""

from functools import cache

import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, range_constexpr, rocdl
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

from pyhip.codegen.flydsl import helpers as fxh
from .moe_gemm_2stage.common import get_device_cache_key, torch_tensor_to_pointer as _ptr


M128_QUEUE_SIZE = 256  # Eight heads, spaced by 32 int32 elements; not OC splits.
M128_CONTINUOUS_VMEM_CONFIG = {
    "block_m": 128, "block_n": 128, "sort_block_m": 128,
    "num_waves": 4, "num_oc_splits": 8, "xcd_count": 2,
    "weight_cache_policy": 16, "output_cache_policy": 18,
    "output_layout": "packed", "packed_rows": 128,
}


def _scalar(value):
    return fx.Int32(rocdl.readfirstlane(fx.Int32.ir_type, value.ir_value()))


def _pin_address(value):
    return fx.Int32(llvm.inline_asm(
        fx.Int32.ir_type, [value.ir_value()], "", "=v,0", has_side_effects=True,
    ))


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
def _compile_m128_down(device_key, n, topk, num_experts, persistent,
                       is_sorted_bm_256, num_oc_splits):
    if device_key is None or device_key[2].split(":")[0] != "gfx950":
        raise ValueError("M128 blockscale Down requires gfx950")
    m_blocks_per_sort = 2 if is_sorted_bm_256 else 1
    width = 4 if persistent else 2
    n_split = n // num_oc_splits
    packets = n_split // 64
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
        active_tasks = (_scalar(valid_ids[0]) // 128) * num_oc_splits
        lds = fx.SharedAllocator().allocate(Storage).peek()
        task_slot = fx.make_view(lds.arena.ptr + task_offset, fx.make_layout(1, 1))

        def take_task(slot):
            # Keep one claim site in the while header. Shards are independent
            # of the number of output-channel splits.
            if tid == 0:
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
            block_m, oc = task // num_oc_splits, task % num_oc_splits
            row_begin = block_m * 128
            tile_has_rows = fx.Boolean(True)
            if const_expr(is_sorted_bm_256):
                # Native sorting has a valid prefix followed only by padding.
                # All waves read the same first ID; test before metadata/DMA.
                first = _scalar(sorted_ids[row_begin]).bitcast(fx.Uint32)
                tile_has_rows = ((first & 0xFFFFFF) < fx.Uint32(tokens)) & ((first >> 24) < topk)
            if tile_has_rows:
                expert = _scalar(expert_ids[block_m // m_blocks_per_sort])
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
                            fx.Float32.ir_type, asrsrc, offset.ir_value(), zero, aux=fxh.rocdl_aux(0)))
                    if const_expr(not persistent and not is_sorted_bm_256):
                        mask = fx.Uint64(rocdl.ballot(fx.Uint64.ir_type, valid.ir_value()))
                        if lane == 0:
                            alive[wave] = (mask != fx.Uint64(0)).select(fx.Int32(1), fx.Int32(0))
                for turn in range_constexpr((scale_count + 255) // 256):
                    index = tid + turn * 256
                    if index < scale_count:
                        bscales[index] = weight_scales[expert * (n // 64) + oc * scale_count + index]
                fx.barrier()
                has_rows = fx.Boolean(True) if const_expr(persistent or is_sorted_bm_256) else _scalar(alive[0] | alive[1]) != 0
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
                            aux=fxh.rocdl_aux(16))

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
                            (packet_index(q) * 16384).ir_value(), aux=fxh.rocdl_aux(18))

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
                fx.barrier()
            else:
                virtual = active_tasks

        if const_expr(persistent):
            if tid == 0:
                # Each shard has 64 workers, but its task count can differ by
                # one when OC splits do not make active_tasks divisible by 8.
                shard_tasks = (active_tasks + 7 - worker % 8) // 8
                if task_slot[0] == shard_tasks + 63:
                    counter[worker % 8 * 32] = fx.Int32(0)

    @flyc.jit
    def launch(output: fx.Pointer, input_q: fx.Pointer, weight: fx.Pointer,
               input_scales: fx.Pointer, weight_scales: fx.Pointer, sorted_ids: fx.Pointer,
               sorted_weights: fx.Pointer, expert_ids: fx.Pointer, valid_ids: fx.Pointer,
               counter: fx.Pointer, tokens: fx.Int32, capacity_blocks: fx.Int32, stream: fx.Stream):
        moe_down_m128_kernel(output, input_q, weight, input_scales, weight_scales, sorted_ids,
            sorted_weights, expert_ids, valid_ids, counter, tokens, capacity_blocks, value_attrs=attrs).launch(
                grid=(512 if const_expr(persistent) else capacity_blocks * num_oc_splits, 1, 1),
                block=(256, 1, 1), stream=stream)

    return launch


@cache
def _compile_m128_streaming_down(device_key, n, k, topk, num_experts, persistent,
                                 is_sorted_bm_256, num_oc_splits):
    """Specialize K256 to the original N pipeline; stream larger K in two slots.

    A single K iteration cannot recover the original A/scale reuse, N lookahead
    or interleaved packing. Reuse the exact launcher, not the generic loop body.
    """
    if device_key is None or device_key[2].split(":")[0] != "gfx950":
        raise ValueError("M128 blockscale Down requires gfx950")
    if k == 256:
        return _compile_m128_down(device_key, n, topk, num_experts, persistent,
                                  is_sorted_bm_256, num_oc_splits)
    m_blocks_per_sort = 2 if is_sorted_bm_256 else 1
    width = 4 if persistent else 2
    n_split = n // num_oc_splits
    packets, k_chunks, k_blocks = n_split // 64, k // 256, k // 128
    ids_offset = 2 * 4096
    routes_offset = ids_offset + 128
    alive_offset = routes_offset + 128
    task_offset = alive_offset + 2
    arena_words = task_offset + int(persistent)
    attrs = {"rocdl.waves_per_eu": 2, "rocdl.flat_work_group_size": "256,256",
             "llvm.passthrough": [["target-features", "-packed-fp32-ops"], ["amdgpu-agpr-alloc", "0,0"]]}

    @fx.struct
    class Storage:
        arena: fx.Array[fx.Int32, arena_words, 16]

    @flyc.kernel(known_block_size=[256, 1, 1])
    def moe_down_m128_streaming_kernel(output: fx.Pointer, input_q: fx.Pointer, weight: fx.Pointer,
                                       input_scales: fx.Pointer, weight_scales: fx.Pointer,
                                       sorted_ids: fx.Pointer, sorted_weights: fx.Pointer, expert_ids: fx.Pointer,
                                       valid_ids: fx.Pointer, counter: fx.Pointer, tokens: fx.Int32,
                                       capacity_blocks: fx.Int32):
        tid = fx.Int32(fx.thread_idx.x)
        wave = _scalar(tid // 64)
        worker = fx.Int32(fx.block_idx.x)
        virtual = worker
        active_tasks = (_scalar(valid_ids[0]) // 128) * num_oc_splits
        lds = fx.SharedAllocator().allocate(Storage).peek()
        task_slot = fx.make_view(lds.arena.ptr + task_offset, fx.make_layout(1, 1))

        def take_task(slot):
            if tid == 0:
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
            block_m, oc = task // num_oc_splits, task % num_oc_splits
            row_begin = block_m * 128
            tile_has_rows = fx.Boolean(True)
            if const_expr(is_sorted_bm_256):
                first = _scalar(sorted_ids[row_begin]).bitcast(fx.Uint32)
                tile_has_rows = ((first & 0xFFFFFF) < fx.Uint32(tokens)) & ((first >> 24) < topk)
            if tile_has_rows:
                expert = _scalar(expert_ids[block_m // m_blocks_per_sort])
                rows = tokens * topk
                lane = tid % 64
                lr, lk = lane % 16, lane // 16
                ids = fx.make_view(lds.arena.ptr + ids_offset, fx.make_layout(128, 1))
                routes = fx.make_view(lds.arena.ptr + routes_offset, fx.make_layout(128, 1))
                alive = fx.make_view(lds.arena.ptr + alive_offset, fx.make_layout(2, 1))
                if tid < 128:
                    encoded = sorted_ids[row_begin + tid].bitcast(fx.Uint32)
                    valid = ((encoded & 0xFFFFFF) < fx.Uint32(tokens)) & ((encoded >> 24) < topk)
                    ids[tid] = encoded.bitcast(fx.Int32)
                    routes[tid] = sorted_weights[row_begin + tid].bitcast(fx.Int32)
                    if const_expr(not persistent and not is_sorted_bm_256):
                        mask = fx.Uint64(rocdl.ballot(fx.Uint64.ir_type, valid.ir_value()))
                        if lane == 0:
                            alive[wave] = (mask != fx.Uint64(0)).select(fx.Int32(1), fx.Int32(0))
                fx.barrier()
                has_rows = fx.Boolean(True) if const_expr(persistent or is_sorted_bm_256) else _scalar(alive[0] | alive[1]) != 0
                if has_rows:
                    wview = fx.make_view(weight + fx.Int64(expert) * (n * k) + fx.Int64(oc) * (n_split * k),
                                         fx.make_layout(n_split * k, 1))
                    wrsrc = fx.rocdl.get_buffer_rsrc(fx.get_iter(fx.rocdl.make_buffer_tensor(wview, False)))
                    aptr = fx.recast_iter(fx.PointerType.get(fx.Int32.ir_type, input_q.memspace, 16), input_q)
                    abuf = fx.rocdl.make_buffer_tensor(fx.make_view(aptr, fx.make_layout(rows * (k // 4), 1)), False)
                    apacket = fxh.BufferTensor(fx.make_view(fx.get_iter(abuf), fx.make_layout(4, 1)))
                    asbuf = fx.rocdl.make_buffer_tensor(fx.make_view(input_scales, fx.make_layout(rows * k_blocks, 1)), False)
                    asrsrc = fx.rocdl.get_buffer_rsrc(fx.get_iter(asbuf))
                    out_rows = capacity_blocks * 128
                    obuf = fx.rocdl.make_buffer_tensor(fx.make_view(output, fx.make_layout(out_rows * n, 1)), False)
                    orsrc = fx.rocdl.get_buffer_rsrc(fx.get_iter(obuf))
                    blds = fxh.LdsTensor(fx.make_view(lds.arena.ptr, fx.make_layout(4, 1)))
                    dma_base = _scalar(fx.Int32(fx.ptrtoint(lds.arena.ptr))) + wave * 1024
                    dma_offset = _pin_address(tid * 16)
                    zero = fx.Int32(0).ir_value()
                    sources, valid_rows, routing, out_addresses = [], [], [], []
                    for mi in range_constexpr(2):
                        row = wave * 32 + mi * 16 + lr
                        encoded = ids[row].bitcast(fx.Uint32)
                        token, slot = encoded & 0xFFFFFF, encoded >> 24
                        valid_rows.append((token < fx.Uint32(tokens)) & (slot < topk))
                        sources.append(fx.Int32(token) * topk + fx.Int32(slot))
                        routing.append(routes[row].bitcast(fx.Float32))
                        for parity in range_constexpr(2):
                            target = wave * 32 + mi * 16 + lr // 2 * 2 + parity
                            encoded = ids[target].bitcast(fx.Uint32)
                            valid = ((encoded & 0xFFFFFF) < fx.Uint32(tokens)) & ((encoded >> 24) < topk)
                            column = ((lk & 1) * 2 + (lk >> 1)) * 8 + lr % 2 * 32
                            offset = block_m * (128 * n * 2) + oc * (128 * n_split * 2) + target * 128 + column * 2
                            out_addresses.append(_pin_address(valid.select(offset, out_rows * n * 2)))

                    def dma_chunk(packet, ki, phase):
                        # shuffle(16,16): each N16/K256 chunk is contiguous,
                        # but adjacent N16 groups are 16*K bytes apart.
                        for ni in range_constexpr(4):
                            dst = llvm.inttoptr(ir.Type.parse("!llvm.ptr<3>"),
                                (dma_base + phase * 16384 + ni * 4096).ir_value())
                            offset = packet * (64 * k) + ni * (16 * k) + ki * 4096
                            rocdl.raw_ptr_buffer_load_async_lds(wrsrc, dst, fx.Int32(16).ir_value(),
                                dma_offset.ir_value(), offset.ir_value(), zero, aux=fxh.rocdl_aux(16))

                    for q in range(fx.Int32(0), fx.Int32(packets), fx.Int32(1)):
                        q = fx.Int32(q)
                        packet = ((q // 2 + worker // 8) % (packets // 2)) * 2 + q % 2
                        dma_chunk(packet, fx.Int32(0), fx.Int32(0))
                        if const_expr(k_chunks > 1):
                            dma_chunk(packet, fx.Int32(1), fx.Int32(1))
                        current = [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range_constexpr(8)]
                        for ki, state in range(fx.Int32(0), fx.Int32(k_chunks), fx.Int32(1), init=current):
                            ki = fx.Int32(ki)
                            rocdl.s_setprio(3)
                            rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0)
                            _stage_end()
                            phase = ki % 2
                            b = fx.make_rmem_tensor([8, 4, 2], fx.Int32)
                            bwords = fx.make_view(fx.get_iter(b), fx.make_ordered_layout([4, 2, 4, 2], 0))
                            address = _pin_address(fx.Int32(fx.ptrtoint(lds.arena.ptr)) + lr * 16 + lk * 256 + phase * 16384)
                            for kb in range_constexpr(2):
                                for ni in range_constexpr(4):
                                    for part in range_constexpr(2):
                                        bwords[None, part, ni, kb].store(_pin_packet(blds.load(
                                            address_bytes=address, offset_bytes=ni * 4096 + kb * 2048 + part * 1024)))
                            rocdl.s_waitcnt(lgkmcnt=0)
                            fx.barrier()  # All readers retire before this slot's refill.
                            if ki + 2 < k_chunks:
                                dma_chunk(packet, ki + 2, phase)

                            a = fx.make_rmem_tensor([8, 2, 2], fx.Int32)
                            awords = fx.make_view(fx.get_iter(a), fx.make_ordered_layout([4, 2, 2, 2], 0))
                            factors = [[], []]
                            for kb in range_constexpr(2):
                                scale_index = (fx.Int64(expert) * (n // 128 * k_blocks)
                                               + fx.Int64(oc) * (n_split // 128 * k_blocks)
                                               + fx.Int64(packet // 2) * k_blocks + ki * 2 + kb)
                                scale_b = _scalar(weight_scales[scale_index].bitcast(fx.Int32)).bitcast(fx.Float32)
                                for mi in range_constexpr(2):
                                    for part in range_constexpr(2):
                                        a_offset = valid_rows[mi].select(sources[mi] * k + ki * 256 + lk * 16
                                                                        + kb * 128 + part * 64, fx.Int32(-1))
                                        awords[None, part, mi, kb].store(apacket.load(voffset_bytes=a_offset))
                                    scale_offset = valid_rows[mi].select(((ki * 2 + kb) * rows + sources[mi]) * 4, fx.Int32(-1))
                                    scale_a = fx.Float32(rocdl.raw_ptr_buffer_load(fx.Float32.ir_type,
                                        asrsrc, scale_offset.ir_value(), zero, aux=fxh.rocdl_aux(0)))
                                    factors[kb].append(scale_a * scale_b)

                            rocdl.s_setprio(0)
                            accum = fx.make_rmem_tensor([4, 2, 4], fx.Float32)
                            for mi in range_constexpr(2):
                                for ni in range_constexpr(4):
                                    accum[None, mi, ni].store(fx.Vector(state[mi * 4 + ni]))
                            atom = fx.make_mma_atom(rocdl.cdna4.MFMA_Scale(16, 16, 128, fx.Float8E4M3FN))
                            pending, dequant = [], []

                            def retire(item):
                                partial, mi, ni, kb, element = item
                                value = partial.load()[element]
                                result = fx.Float32(fxh.eltwise_op("llvm.fma.f32", value,
                                                                   factors[kb][mi], accum[element, mi, ni]))
                                if const_expr(kb == 0):
                                    result = (ki == 0).select(value * factors[kb][mi], result)
                                accum[element, mi, ni] = result

                            for index in range_constexpr(16):
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
                                for _ in range_constexpr(min(4, len(dequant))):
                                    retire(dequant.pop(0))
                                rocdl.sched_barrier(0)
                            for old in pending:
                                dequant.extend([(*old, element) for element in range_constexpr(4)])
                            for item in dequant:
                                retire(item)
                            final = yield [accum[None, mi, ni].load() for mi in range_constexpr(2) for ni in range_constexpr(4)]

                        # Only the complete K sum is multiplied by routing and
                        # converted to BF16; no BF16 partial sums or split-K.
                        current = [fx.Vector(final[i]) for i in range_constexpr(8)]
                        for mi in range_constexpr(2):
                            pieces = [_pack_pair(current[mi * 4 + pair * 2], current[mi * 4 + pair * 2 + 1], routing[mi])
                                      for pair in range_constexpr(2)]
                            packed = _coalesce_output_pairs(pieces[0], pieces[1])
                            for record in range_constexpr(2):
                                rocdl.raw_ptr_buffer_store(packed[record].ir_value(), orsrc,
                                    out_addresses[mi * 2 + record].ir_value(), (packet * 16384).ir_value(), aux=fxh.rocdl_aux(18))
                        rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0)
                        fx.barrier()
            if const_expr(persistent):
                fx.barrier()
            else:
                virtual = active_tasks

        if const_expr(persistent):
            if tid == 0:
                shard_tasks = (active_tasks + 7 - worker % 8) // 8
                if task_slot[0] == shard_tasks + 63:
                    counter[worker % 8 * 32] = fx.Int32(0)

    @flyc.jit
    def launch(output: fx.Pointer, input_q: fx.Pointer, weight: fx.Pointer,
               input_scales: fx.Pointer, weight_scales: fx.Pointer, sorted_ids: fx.Pointer,
               sorted_weights: fx.Pointer, expert_ids: fx.Pointer, valid_ids: fx.Pointer,
               counter: fx.Pointer, tokens: fx.Int32, capacity_blocks: fx.Int32, stream: fx.Stream):
        moe_down_m128_streaming_kernel(output, input_q, weight, input_scales, weight_scales, sorted_ids,
            sorted_weights, expert_ids, valid_ids, counter, tokens, capacity_blocks, value_attrs=attrs).launch(
                grid=(512 if const_expr(persistent) else capacity_blocks * num_oc_splits, 1, 1),
                block=(256, 1, 1), stream=stream)

    return launch


@cache
def make_m128_down(*, n, k=256, topk, num_experts, persistent=False,
                   is_sorted_bm_256=False, num_oc_splits=8):
    """Build a stateless ten-tensor Down callable; cache code, not queue heads.

    num_oc_splits partitions N, not K. Each split must contain whole N128
    scale blocks (at least two N64 packets). Output is always packed by M128.
    K is any positive multiple of 256 within the buffer-address and LDS bounds. K>256
    streams fixed K256 chunks; scales are A[K/128,rows] and B[E,N/128,K/128].
    With sort256, expert_ids[m//2] serves both halves and a first-ID check skips
    empty tiles; arbitrary holes in the native valid prefix are not supported.

    Persistent callers pass 256 zero-initialized int32 words as counter. The
    eight heads self-reset before completion. Never share this storage across
    concurrent launches; graph capture must retain its own queue allocation.
    Independent calls accept an unused one-element counter and preserve it.
    """
    if not isinstance(num_oc_splits, int) or num_oc_splits <= 0:
        raise ValueError("num_oc_splits must be a positive integer")
    if not isinstance(k, int) or k <= 0 or k % 256:
        raise ValueError("M128 requires K to be a positive integer multiple of256")
    if n <= 0 or n % (num_oc_splits * 128) or not 0 < topk <= min(num_experts, 255):
        raise ValueError("M128 requires N divisible by num_oc_splits*128 and valid TOPK")
    sort_block_m = 256 if is_sorted_bm_256 else 128
    n_split = n // num_oc_splits
    scale_words = (n_split // 128 * 2 + 63 + 255) // 256 * 256
    arena_words = ((scale_words + 4 * 4096 + 128 + 128 * 3 + 2) if k == 256
                   else (2 * 4096 + 128 * 2 + 2)) + int(persistent)
    if arena_words * 4 > 160 * 1024:
        raise ValueError(f"M128 Down needs {arena_words * 4} LDS bytes, exceeding gfx950's 163840")

    def down(output, input_q, weight, input_scales, weight_scales,
             sorted_ids, sorted_weights, expert_ids, valid_ids, counter):
        tokens, actual_topk, actual_k = input_q.shape
        tensors = (output, input_q, weight, input_scales, weight_scales, sorted_ids, sorted_weights, expert_ids, valid_ids, counter)
        assert 0 < tokens < (1 << 24) and (actual_topk, actual_k) == (topk, k)
        assert weight.shape == (num_experts, n, k) and output.shape == (expert_ids.numel() * sort_block_m, n)
        assert input_q.dtype == weight.dtype == torch.float8_e4m3fn and output.dtype == torch.bfloat16
        assert input_scales.dtype == weight_scales.dtype == sorted_weights.dtype == torch.float32
        assert sorted_ids.dtype == expert_ids.dtype == valid_ids.dtype == counter.dtype == torch.int32
        assert input_scales.numel() == tokens * topk * (k // 128) and weight_scales.shape == (num_experts, n // 128, k // 128)
        assert sorted_ids.ndim == expert_ids.ndim == 1 and sorted_ids.shape == sorted_weights.shape
        assert expert_ids.numel() == (sorted_ids.numel() + sort_block_m - 1) // sort_block_m, f"native sort{sort_block_m} required"
        assert counter.numel() == (M128_QUEUE_SIZE if persistent else 1) and valid_ids.numel() >= 1
        assert all(t.is_cuda and t.is_contiguous() and t.device == output.device for t in tensors)
        # The invalid-row voffset plus the last packet's soffset must not wrap.
        assert (output.numel() + 128 * n_split) * 2 < 1 << 32 and input_q.numel() < 1 << 32
        assert input_scales.numel() * 4 < 1 << 32 and n_split * k < 1 << 32
        capacity_blocks = expert_ids.numel() * (sort_block_m // 128)
        with torch.cuda.device(output.device):
            launcher = _compile_m128_streaming_down(get_device_cache_key(), n, k, topk, num_experts,
                                                    persistent, is_sorted_bm_256, num_oc_splits)
            stream = torch.cuda.current_stream(output.device)
            compiled = getattr(launcher, "_cf", None)
            if compiled is None:
                _run_compiled(launcher, *[_ptr(t) for t in tensors], fx.Int32(tokens),
                              fx.Int32(capacity_blocks), fx.Stream(stream.cuda_stream))
            else:
                compiled(*(t.data_ptr() for t in tensors), tokens, capacity_blocks, stream.cuda_stream)
        return output

    down.config = {**M128_CONTINUOUS_VMEM_CONFIG, "persistent": persistent, "k": k,
                   "num_oc_splits": num_oc_splits, "sort_block_m": sort_block_m,
                   "is_sorted_bm_256": is_sorted_bm_256,
                   "scheduler": "sharded" if persistent else "independent",
                   "persistent_workgroups": 512 if persistent else 0,
                   "xcd_count": 4 if persistent else 2, "lds_bytes": arena_words * 4}
    if persistent:
        down.config["queue_self_reset"] = True
    return down