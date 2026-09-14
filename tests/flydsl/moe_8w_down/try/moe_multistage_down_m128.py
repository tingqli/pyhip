# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Experimental independent CTAs, rolling B DMA and exact block-scale math.

Historical callers default to M256 sorting/packing. Native M128/M64 callers
must pass sort_block_m=block_m; expert IDs and packed strides use that size.
Alternate shapes, layouts and full-output atomic reduction remain explicit
experiments, not production dispatch. CTA barriers publish B and retire LDS
readers; they never synchronize independent CTAs. No routing reassociation.
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


# width4 is a logical task transpose, not a four-XCD launch (gfx950 uses eight).
# All eight OC slices consume the same A_M; grouping four OC consumers per XCD
# gives cache reuse opportunities instead of sending A_M to eight XCDs as in
# identity. Native sort128 placement: 1154/1157 M blocks touch two XCDs; three
# boundary blocks touch four. Loads/MFMA counts and CTA-private LDS do not change.
# With historical sort256, width4 also breaks the padding/SE phase alias
# {384,384,384,5}. Native sort128 has no such imbalance, so that explanation no
# longer applies. Selected empirically: native identity/width4/width8 Down was
# 504.997/479.890/488.238 us in the same two-round comparison. This does NOT prove
# width4 has fewer wave stalls or SE admission stalls than width8; its remaining
# advantage mixes access order, cache working set and concurrency (not isolated).
# Evidence and persistent/locality design: ../../attn_4wave/tools/se-dispatch.md
M128_CONTINUOUS_VMEM_CONFIG = {
    "block_m": 128, "rows_per_wave": 32, "num_oc_splits": 8,
    "prefetch_distance": 3, "ring_slots": 4, "stores_in_compute": True,
    "xcd_count": 4, "rotate_n": 3, "memory_priority": 3,
    "active_b_prefetch": 3, "spread_refill": True,
}


def _coalesce_n128(first, second):
    masks = [fx.Uint64(0x3333333333333333), fx.Uint64(0xCCCCCCCCCCCCCCCC)]
    assembly = ["s_mov_b64 vcc, $16"]
    assembly.extend(f"v_cndmask_b32_dpp ${i}, ${12+i}, ${8+i}, vcc quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf" for i in range(4))
    assembly.append("s_mov_b64 vcc, $17")
    assembly.extend(f"v_cndmask_b32_dpp ${4+i}, ${8+i}, ${12+i}, vcc quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf" for i in range(4))
    raw = llvm.inline_asm(ir.Type.parse("!llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32)>"),
        [first[i].ir_value() for i in range(4)] + [second[i].ir_value() for i in range(4)] + [mask.ir_value() for mask in masks],
        "\n".join(assembly), ",".join(["=&v"] * 8 + ["v"] * 8 + ["s", "s", "~{vcc}"]), has_side_effects=True)
    return [_pin_packet(fx.Vector.from_elements([fx.Int32(llvm.extractvalue(fx.Int32.ir_type, raw, [side * 4 + i]))
                                                for i in range(4)], fx.Int32)) for side in range(2)]


@cache
def make_prepare_rows(topk):
    @flyc.kernel(known_block_size=[256, 1, 1])
    def prepare_rows_kernel(dest: fx.Pointer, scale_dest: fx.Pointer, a: fx.Pointer, scales: fx.Pointer,
                            ids: fx.Pointer, routes: fx.Pointer, valid_ids: fx.Pointer,
                            tokens: fx.Int32, capacity_rows: fx.Int32):
        tid = fx.Int32(fx.thread_idx.x)
        row = fx.Int32(fx.block_idx.x) * 16 + tid // 16
        column = tid % 16
        if row < _scalar(valid_ids[0]):
            encoded = ids[row].bitcast(fx.Uint32)
            token, slot = encoded & 0xFFFFFF, encoded >> 24
            valid = (token < fx.Uint32(tokens)) & (slot < topk)
            source = fx.Int32(token) * topk + fx.Int32(slot)
            aptr = fx.recast_iter(fx.PointerType.get(fx.Int32.ir_type, a.memspace, 16), a)
            abuf = fx.rocdl.make_buffer_tensor(fx.make_view(aptr, fx.make_layout(tokens * topk * 64, 1)), False)
            dbuf = fx.rocdl.make_buffer_tensor(fx.make_view(dest, fx.make_layout(capacity_rows * 256, 1)), False)
            asrc = fx.rocdl.get_buffer_rsrc(fx.get_iter(abuf))
            dsrc = fx.rocdl.get_buffer_rsrc(fx.get_iter(dbuf))
            offset = valid.select(source * 256 + column * 16, fx.Int32(-1))
            zero = fx.Int32(0).ir_value()
            words = rocdl.raw_ptr_buffer_load(ir.VectorType.get([4], fx.Int32.ir_type), asrc, offset.ir_value(), zero)
            rocdl.raw_ptr_buffer_store(words, dsrc, (row * 256 + column * 16).ir_value(), zero)
            if column == 0:
                sbuf = fx.rocdl.make_buffer_tensor(fx.make_view(scales, fx.make_layout(tokens * topk * 2, 1)), False)
                ssrc = fx.rocdl.get_buffer_rsrc(fx.get_iter(sbuf))
                for kb in range_constexpr(2):
                    so = valid.select((kb * tokens * topk + source) * 4, fx.Int32(-1))
                    scale_dest[row * 3 + kb] = fx.Float32(rocdl.raw_ptr_buffer_load(fx.Float32.ir_type, ssrc, so.ir_value(), zero))
                scale_dest[row * 3 + 2] = routes[row]

    @flyc.jit
    def launch(dest: fx.Pointer, scale_dest: fx.Pointer, a: fx.Pointer, scales: fx.Pointer,
               ids: fx.Pointer, routes: fx.Pointer, valid_ids: fx.Pointer,
               tokens: fx.Int32, capacity_rows: fx.Int32, stream: fx.Stream):
        prepare_rows_kernel(dest, scale_dest, a, scales, ids, routes, valid_ids, tokens, capacity_rows).launch(
            grid=((capacity_rows + 15) // 16, 1, 1), block=(256, 1, 1), stream=stream)

    def prepare(dest, scale_dest, a, scales, ids, routes, valid_ids):
        tensors = (dest, scale_dest, a, scales, ids, routes, valid_ids)
        stream = torch.cuda.current_stream(a.device)
        compiled = getattr(launch, "_cf", None)
        if compiled is None:
            _run_compiled(launch, *[_ptr(t) for t in tensors], fx.Int32(a.shape[0]), fx.Int32(dest.shape[0]), fx.Stream(stream.cuda_stream))
        else:
            compiled(*(t.data_ptr() for t in tensors), a.shape[0], dest.shape[0], stream.cuda_stream)
    return prepare


@cache
def make_m128_down(*, n, k=256, topk, num_experts, block_m=128, block_n=128,
                   num_oc_splits=4, prefetch_distance=3, ring_slots=4,
                   output_cache_policy=2, weight_cache_policy=16,
                   output_layout="packed", xcd_swizzle=True,
                   skip_empty_tasks=True, refill_first=True, waves_per_eu=2,
                   coalesced_output=True, packed_fp32=False, global_b=False, rows_per_wave=32,
                   dma_in_compute=False, stores_in_compute=False, pack_immediate=False, half_k=False,
                   raw_output=False, presorted_a=False, xcd_count=8, task_m_group=1, n_major=False,
                   output_nmajor=False, atomic_sum=False, half_n=False, stream_b=False,
                   direct_metadata=False, unsigned_tasks=False, early_task_guard=False, wide_output=False,
                   packed_rows=None, valu_budget=7, retire_distance=2, relaxed_compute=False, rotate_n=0,
                   memory_priority=0, compute_priority=0, static_ctas=0,
                   prologue_b_prefetch=0, startup_refill_in_compute=False,
                   preload_metadata=False, refill_before_wait=False, eager_first_store=False,
                   active_b_prefetch=0, spread_refill=False, uniform_b_scales=False,
                   first_ready_store=False, record_placement=False, wave_tile_2x2=False,
                   register_metadata=False, dma_bytes=16, packet_rotation=False, pair_refill=False,
                   double_n_packet=False, sort_block_m=256,
                   queue_ctas=0, queue_batch=4, queue_record=False,
                   global_ctas=0, global_record=False):
    assert block_m in (64, 128, 256) and block_n == 128 and k == 256
    if sort_block_m not in (64, 128, 256) or sort_block_m % block_m:
        raise ValueError("sort_block_m must be 64/128/256 and a multiple of block_m")
    packed_rows = sort_block_m if packed_rows is None else packed_rows
    if queue_ctas:
        if not (0 < queue_ctas <= 4096 and queue_batch in (1, 2, 4, 8, 16)):
            raise ValueError("invalid XCD queue worker count or bundle size")
        if not (block_m == sort_block_m == 128 and rows_per_wave == 32 and num_oc_splits == 8
                and xcd_swizzle and not static_ctas and not record_placement and not early_task_guard
                and not unsigned_tasks and not presorted_a and not atomic_sum and not global_b
                and not half_k and not half_n and not wide_output and not raw_output and not output_nmajor
                and not wave_tile_2x2 and not double_n_packet and not stream_b
                and task_m_group == 1 and not n_major and packed_rows == 128):
            raise ValueError("XCD queues require the native M128/OC8 standard packed path")
    elif queue_record:
        raise ValueError("queue_record requires queue_ctas")
    if global_ctas:
        if not (0 < global_ctas <= 4096 and block_m == sort_block_m == 128 and rows_per_wave == 32
                and num_oc_splits in (4, 8) and not xcd_swizzle and not queue_ctas and not static_ctas
                and not record_placement and not early_task_guard and not unsigned_tasks
                and not presorted_a and not atomic_sum and not global_b and not half_k and not half_n
                and not wide_output and not raw_output and not output_nmajor and not wave_tile_2x2
                and not double_n_packet and not stream_b and task_m_group == 1 and not n_major
                and packed_rows == 128):
            raise ValueError("global persistent requires native M128/OC4 or OC8, without swizzle or another scheduler")
    elif global_record:
        raise ValueError("global_record requires global_ctas")
    if sort_block_m != 256 and (packed_rows != sort_block_m or raw_output or wide_output
                               or double_n_packet or wave_tile_2x2):
        raise ValueError("native sorting requires matching standard packed rows")
    assert output_layout == "packed"
    assert n > 0 and num_oc_splits > 0 and n % (128 * num_oc_splits) == 0
    assert 0 < topk <= min(num_experts, 255)
    assert 1 <= prefetch_distance <= ring_slots <= 6
    assert output_cache_policy in (0, 1, 2, 3, 16, 17, 18, 19) and weight_cache_policy in (0, 1, 2, 3, 16, 17, 18, 19)
    assert waves_per_eu in (1, 2, 3, 4)
    assert not (dma_in_compute or stores_in_compute) or (
        ((block_m == 128 and rows_per_wave in (16, 32)) or (block_m == 64 and rows_per_wave in (16, 32))) and not global_b)
    assert not pack_immediate or not stores_in_compute
    assert not half_k or (pack_immediate and ring_slots % 2 == 0 and not global_b and not dma_in_compute and not half_n)
    assert not raw_output or (rows_per_wave == 32 and pack_immediate and not half_k)
    assert xcd_count in (1, 2, 4, 8, 16, 32, 64) and task_m_group in (1, 2, 4, 8)
    assert not n_major or task_m_group == 1
    assert not output_nmajor or not raw_output
    assert not atomic_sum or (not raw_output and not output_nmajor)
    assert not half_n or (pack_immediate and ring_slots % 2 == 0 and not half_k and not raw_output
                         and not global_b and not dma_in_compute and not stores_in_compute and coalesced_output)
    assert not stream_b or (not half_k and not half_n and not global_b and ring_slots > prefetch_distance)
    assert not direct_metadata or (not presorted_a and n // num_oc_splits // 128 * 2 <= 64)
    assert not early_task_guard or (block_m in (64, 128) and direct_metadata and task_m_group == 1 and not n_major and skip_empty_tasks)
    assert not wide_output or (pack_immediate and not raw_output and not output_nmajor and not atomic_sum
                               and not half_k and not half_n and coalesced_output and ring_slots % 2 == 0)
    assert packed_rows == sort_block_m or (sort_block_m == 256 and packed_rows in (264, 272, 288, 320))
    assert packed_rows == sort_block_m or not (raw_output or output_nmajor or atomic_sum or wide_output)
    assert valu_budget in (4, 5, 6, 7, 8, 10, 12) and retire_distance in (2, 3, 4, 6)
    assert rotate_n in (0, 1, 2, 3)
    assert not rotate_n or not (half_k or half_n or wide_output or global_b)
    assert memory_priority in (0, 1, 2, 3) and compute_priority in (0, 1, 2, 3)
    assert static_ctas in (0, 256, 512, 768)
    assert not static_ctas or (not unsigned_tasks and not early_task_guard and not presorted_a and not atomic_sum)
    assert 0 <= prologue_b_prefetch <= prefetch_distance
    assert 0 <= active_b_prefetch <= prefetch_distance and not (prologue_b_prefetch and active_b_prefetch)
    assert not (startup_refill_in_compute or refill_before_wait
                or spread_refill or uniform_b_scales) or (
        (block_m, rows_per_wave) in ((128, 32), (64, 16)) and not global_b and not static_ctas
        and not half_k and not half_n and not wide_output
        and ring_slots > prefetch_distance)
    assert not (prologue_b_prefetch or preload_metadata) or (
        (block_m, rows_per_wave) in ((128, 32), (64, 16)) and not global_b and not static_ctas
        and not half_k and not half_n and not wide_output and not early_task_guard)
    assert not early_task_guard or not (prologue_b_prefetch or preload_metadata or uniform_b_scales)
    assert not active_b_prefetch or (
        (block_m, rows_per_wave) in ((128, 32), (64, 16)) and not global_b and not static_ctas
        and not half_k and not half_n and not wide_output)
    assert not preload_metadata or (prologue_b_prefetch > 0 and not direct_metadata and not presorted_a)
    assert not refill_before_wait or (not dma_in_compute and not startup_refill_in_compute and refill_first)
    assert not eager_first_store or (block_m == 128 and rows_per_wave == 32 and stores_in_compute
                                     and not pack_immediate and not half_k and not half_n
                                     and not wide_output and not packed_fp32 and not raw_output
                                     and not atomic_sum and not output_nmajor)
    assert not spread_refill or (refill_first and not refill_before_wait and not dma_in_compute and not stream_b)
    assert not uniform_b_scales or not direct_metadata
    assert not first_ready_store or (block_m == 128 and rows_per_wave == 32 and stores_in_compute
                                    and not eager_first_store and not pack_immediate and not packed_fp32
                                    and not half_k and not half_n and not wide_output and not raw_output
                                    and not atomic_sum and not output_nmajor and valu_budget == 7 and retire_distance == 2)
    assert not record_placement or not static_ctas
    assert not double_n_packet or (block_m == 64 and rows_per_wave == 32 and not wave_tile_2x2
        and not half_k and not half_n and not global_b and not stream_b and not wide_output and not raw_output
        and not pack_immediate and not eager_first_store and not first_ready_store and not atomic_sum
        and not output_nmajor and packed_rows == 256 and not active_b_prefetch and not prologue_b_prefetch
        and not spread_refill and not dma_in_compute and not startup_refill_in_compute and not packet_rotation)
    if double_n_packet and (waves_per_eu > 2 or ring_slots < 2):
        raise ValueError("double-N single-slot or forced-high-residency configuration failed correctness; use two slots and waves_per_eu=2")
    assert not register_metadata or (block_m == 64 and rows_per_wave == 16 and early_task_guard and direct_metadata)
    assert dma_bytes in (16, 32)
    if dma_bytes == 32:
        raise ValueError("gfx950 load-to-LDS intrinsic supports at most 16B per lane; see tests_regmeta_dma32.log")
    assert not packet_rotation or (block_m == 64 and rows_per_wave == 16 and not half_k and not half_n
        and not wide_output and not global_b and rotate_n == 0 and not active_b_prefetch and not prologue_b_prefetch)
    assert not pair_refill or (block_m == 64 and rows_per_wave == 16 and prefetch_distance == ring_slots == 2
        and not half_k and not half_n and not global_b and not stream_b and not spread_refill
        and not dma_in_compute and not refill_before_wait and not startup_refill_in_compute)
    assert dma_bytes == 16 or (block_m == 64 and rows_per_wave == 16 and not half_k and not half_n
        and not global_b and not dma_in_compute and not spread_refill and not startup_refill_in_compute)
    assert not wave_tile_2x2 or (block_m == 64 and rows_per_wave == 32 and not half_k and not half_n
        and not global_b and not stream_b and not wide_output and not raw_output and not pack_immediate
        and not eager_first_store and not first_ready_store and not packed_fp32 and not atomic_sum
        and not output_nmajor and coalesced_output and packed_rows == 256 and not dma_in_compute
        and not spread_refill and not active_b_prefetch and not prologue_b_prefetch)
    n_split = n // num_oc_splits
    assert rows_per_wave in (16, 32, 64) and block_m % rows_per_wave == 0
    threads = block_m // rows_per_wave * 64 * (2 if wave_tile_2x2 or double_n_packet else 1)
    m_fragments = rows_per_wave // 16
    n_fragments = 2 if half_n or wave_tile_2x2 else 4
    accum_vectors = m_fragments * n_fragments
    packed_vectors = m_fragments * (1 if wave_tile_2x2 else 4 if wide_output else 2)
    pack_operations = m_fragments * (14 if wave_tile_2x2 else 28)
    m_splits = sort_block_m // block_m
    k_packets = 2 if half_k else 1
    packets = n_split // (128 if double_n_packet else 32 if half_n else 64) * k_packets
    slot_bytes = 32768 if double_n_packet else 16384 // (2 if half_n else k_packets)
    dma_turns = slot_bytes // (threads * 16)
    compute_mfmas = accum_vectors * (1 if half_k else 2)
    dma_mfma_stride = max(1, compute_mfmas // max(1, dma_turns))
    scale_count = n_split // 128 * 2
    scale_words = (scale_count + 63 + 255) // 256 * 256
    ids_offset = scale_words + (0 if global_b else ring_slots * slot_bytes // 4)
    rows_offset = ids_offset + block_m
    alive_offset = rows_offset + block_m * 3
    alive_count = block_m // 64
    queue_offset = alive_offset + alive_count
    arena_words = queue_offset + (4 if queue_ctas else 1 if global_ctas else 0)
    assert arena_words * 4 <= 160 * 1024
    attrs = {"rocdl.waves_per_eu": waves_per_eu, "rocdl.flat_work_group_size": f"{threads},{threads}",
             "llvm.passthrough": [["target-features", "+packed-fp32-ops" if packed_fp32 else "-packed-fp32-ops"],
                                  ["amdgpu-agpr-alloc", "0,0"]]}

    @fx.struct
    class Storage:
        arena: fx.Array[fx.Int32, arena_words, 16]

    @flyc.kernel(known_block_size=[threads, 1, 1])
    def moe_down_m128_kernel(output: fx.Pointer, input_q: fx.Pointer, weight: fx.Pointer,
                             input_scales: fx.Pointer, weight_scales: fx.Pointer,
                             sorted_ids: fx.Pointer, sorted_weights: fx.Pointer, expert_ids: fx.Pointer,
                             valid_ids: fx.Pointer, counter: fx.Pointer, prepared_a: fx.Pointer, prepared_scales: fx.Pointer, tokens: fx.Int32,
                             capacity_blocks: fx.Int32):
        tid = fx.Int32(fx.thread_idx.x)
        wave = _scalar(tid // 64)
        wave_m = wave // 2 if const_expr(wave_tile_2x2 or double_n_packet) else wave
        lane = tid % 64
        lr, lk = lane % 16, lane // 16
        task = fx.Uint32(fx.block_idx.x) if const_expr(unsigned_tasks) else fx.Int32(fx.block_idx.x)
        linear_task = task
        valid_rows = fx.Uint32(_scalar(valid_ids[0])) if const_expr(unsigned_tasks) else _scalar(valid_ids[0])
        active_tasks = (valid_rows // sort_block_m) * m_splits * num_oc_splits
        if const_expr(global_ctas):
            # Direct transplant of the main M256 persistent scheduler: one
            # monotonic global atomicAdd(1), one LDS broadcast, no bundle state.
            global_lds = fx.SharedAllocator().allocate(Storage).peek()
            global_slot = fx.make_view(global_lds.arena.ptr + queue_offset, fx.make_layout(1, 1))

            def take_global_task(slot_view):
                # One claim site in the while header, like the main M256 loop.
                # Duplicating this before/after the loop let LLVM merge the
                # leader diamonds into a multi-entry divergent region: the
                # ordinary build entered compute with the leader masked off.
                if tid == 0:
                    head_ptr = llvm.inttoptr(ir.Type.parse("!llvm.ptr<1>"), fx.ptrtoint(counter).ir_value())
                    next_id = fx.Int32(llvm.AtomicRMWOp(llvm.AtomicBinOp.add,
                        head_ptr, fx.Int32(1).ir_value(), llvm.AtomicOrdering.monotonic, syncscope="agent").res)
                    slot_view[0] = next_id
                fx.barrier()
                return _scalar(slot_view[0])

            if const_expr(global_record):
                global_worker = _scalar(fx.Int32(fx.block_idx.x))
                global_xcc = fx.Int32(llvm.inline_asm(fx.Int32.ir_type, [],
                    "s_getreg_b32 $0, hwreg(HW_REG_XCC_ID, 0, 4)", "=s", has_side_effects=True))
                global_hw = fx.Int32(llvm.inline_asm(fx.Int32.ir_type, [],
                    "s_getreg_b32 $0, hwreg(HW_REG_HW_ID)", "=s", has_side_effects=True))
                if tid == 0:
                    counter[1 + global_worker * 4] = global_xcc
                    counter[1 + global_worker * 4 + 1] = global_hw
        if const_expr(queue_ctas):
            # The physical XCC, not block_id%8, selects the home queue. All
            # resident workers eventually scan each shard, including shards
            # with no workers: no coverage assumption and no global barrier.
            home_xcc = fx.Int32(llvm.inline_asm(fx.Int32.ir_type, [],
                "s_getreg_b32 $0, hwreg(HW_REG_XCC_ID, 0, 4)", "=s", has_side_effects=True))
            worker_id = _scalar(fx.Int32(fx.block_idx.x))
            queue_lds = fx.SharedAllocator().allocate(Storage).peek()
            queue_slots = fx.make_view(queue_lds.arena.ptr + queue_offset, fx.make_layout(4, 1))
            if tid == 0:
                queue_slots[0], queue_slots[1], queue_slots[2] = fx.Int32(0), fx.Int32(0), fx.Int32(0)
                queue_slots[3] = active_tasks
            fx.barrier()
            if const_expr(queue_record):
                worker_hw = fx.Int32(llvm.inline_asm(fx.Int32.ir_type, [],
                    "s_getreg_b32 $0, hwreg(HW_REG_HW_ID)", "=s", has_side_effects=True))
                if tid == 0:
                    counter[256 + worker_id * 4] = home_xcc
                    counter[256 + worker_id * 4 + 1] = worker_hw

            def take_xcd_task(slot_view):
                if tid == 0:
                    queue_next = slot_view[0]
                    queue_end = slot_view[1]
                    queue_visit = slot_view[2]
                    while (queue_next >= queue_end) & (queue_visit < 8):
                        queue_shard = (home_xcc + queue_visit) % 8
                        queue_ptr = llvm.inttoptr(ir.Type.parse("!llvm.ptr<1>"),
                            fx.ptrtoint(counter + queue_shard * 32).ir_value())
                        queue_rank = fx.Int32(llvm.AtomicRMWOp(llvm.AtomicBinOp.add,
                            queue_ptr, fx.Int32(queue_batch).ir_value(),
                            llvm.AtomicOrdering.monotonic, syncscope="agent").res)
                        queue_next = queue_rank * 8 + queue_shard
                        queue_end = queue_next + queue_batch * 8
                        queue_end = (queue_end < active_tasks).select(queue_end, active_tasks)
                        if queue_next >= active_tasks:
                            queue_visit = queue_visit + 1
                            queue_next, queue_end = fx.Int32(0), fx.Int32(0)
                    slot_view[3] = (queue_visit < 8).select(queue_next, active_tasks)
                    slot_view[0] = queue_next + 8
                    slot_view[1], slot_view[2] = queue_end, queue_visit
                fx.barrier()
                return _scalar(slot_view[3])

            linear_task = take_xcd_task(queue_slots)
            task = linear_task
        if const_expr(xcd_swizzle):
            # Transpose only the divisible prefix; keep the tail bijective.
            # task is a logical (M,OC) identity, not a physical XCD binding.
            # The locality argument uses measured placement, not an API promise.
            chunk = active_tasks // xcd_count
            mapped = (task % xcd_count) * chunk + task // xcd_count
            task = (task < chunk * xcd_count).select(mapped, task)
        if const_expr(record_placement):
            if tid == 0:
                # Diagnostic-only: observe actual placement, never select work
                # from an assumed block_id -> physical XCD correspondence.
                xcc = fx.Int32(llvm.inline_asm(fx.Int32.ir_type, [],
                    "s_getreg_b32 $0, hwreg(HW_REG_XCC_ID, 0, 4)", "=s", has_side_effects=True))
                hw_id = fx.Int32(llvm.inline_asm(fx.Int32.ir_type, [],
                    "s_getreg_b32 $0, hwreg(HW_REG_HW_ID)", "=s", has_side_effects=True))
                counter[linear_task * 4] = xcc
                counter[linear_task * 4 + 1] = hw_id
                counter[linear_task * 4 + 2] = fx.Int32(task)
                counter[linear_task * 4 + 3] = 0
        task_has_rows = fx.Boolean(True)
        guard_ids = fx.Int32(0)
        if const_expr(early_task_guard):
            if task < active_tasks:
                task_row = task // num_oc_splits * block_m
                live_rows = fx.Boolean(False)
                for part in range_constexpr(block_m // 64):
                    encoded = sorted_ids[task_row + lane + part * 64].bitcast(fx.Uint32)
                    if const_expr(register_metadata):
                        guard_ids = encoded.bitcast(fx.Int32)
                    token, slot = encoded & 0xFFFFFF, encoded >> 24
                    live_rows = live_rows | ((token < fx.Uint32(tokens)) & (slot < topk))
                task_has_rows = fx.Uint64(rocdl.ballot(fx.Uint64.ir_type, live_rows.ir_value())) != fx.Uint64(0)
        while ((take_global_task(global_slot) if const_expr(global_ctas) else task) < active_tasks) & task_has_rows:
            if const_expr(global_ctas):
                task = _scalar(global_slot[0])
            if const_expr(global_record):
                if tid == 0:
                    record_base = 1 + global_ctas * 4 + task * 4
                    visit_ptr = llvm.inttoptr(ir.Type.parse("!llvm.ptr<1>"), fx.ptrtoint(counter + record_base).ir_value())
                    llvm.AtomicRMWOp(llvm.AtomicBinOp.add, visit_ptr, fx.Int32(1).ir_value(),
                                    llvm.AtomicOrdering.monotonic, syncscope="agent")
                    counter[record_base + 1], counter[record_base + 2] = global_xcc, global_hw
                    counter[record_base + 3] = global_worker
                    counter[1 + global_worker * 4 + 2] = counter[1 + global_worker * 4 + 2] + 1
            if const_expr(queue_record):
                if tid == 0:
                    record = 256 + queue_ctas * 4 + linear_task * 8
                    record_ptr = llvm.inttoptr(ir.Type.parse("!llvm.ptr<1>"), fx.ptrtoint(counter + record).ir_value())
                    llvm.AtomicRMWOp(llvm.AtomicBinOp.add, record_ptr, fx.Int32(1).ir_value(),
                                    llvm.AtomicOrdering.monotonic, syncscope="agent")
                    ordinal = counter[256 + worker_id * 4 + 2] + 1
                    counter[256 + worker_id * 4 + 2] = ordinal
                    counter[256 + worker_id * 4 + 3] = counter[256 + worker_id * 4 + 3] + (home_xcc != linear_task % 8).select(fx.Int32(1), fx.Int32(0))
                    counter[record + 1], counter[record + 2] = home_xcc, worker_hw
                    counter[record + 3], counter[record + 4] = worker_id, task
                    counter[record + 5], counter[record + 7] = linear_task % 8, ordinal
            half_block, oc = task // num_oc_splits, task % num_oc_splits
            if const_expr(n_major):
                half_block, oc = task % (active_tasks // num_oc_splits), task // (active_tasks // num_oc_splits)
            if const_expr(task_m_group > 1):
                group_index = task // (task_m_group * num_oc_splits)
                group_remaining = active_tasks // num_oc_splits - group_index * task_m_group
                group_width = (group_remaining < task_m_group).select(group_remaining, fx.Int32(task_m_group))
                group_local = task - group_index * task_m_group * num_oc_splits
                half_block, oc = group_index * task_m_group + group_local % group_width, group_local // group_width
            parent, half = half_block // m_splits, half_block % m_splits
            row_begin = parent * sort_block_m + half * block_m
            expert = _scalar(expert_ids[parent])
            rows = tokens * topk
            lds = queue_lds if const_expr(queue_ctas) else global_lds if const_expr(global_ctas) else fx.SharedAllocator().allocate(Storage).peek()
            bptr = lds.arena.ptr + scale_words
            fptr = fx.recast_iter(fx.PointerType.get(fx.Float32.ir_type, lds.arena.ptr.memspace, 16), lds.arena.ptr)
            ids = fx.make_view(lds.arena.ptr + ids_offset, fx.make_layout(block_m, 1))
            row_scales = fx.make_view(fptr + rows_offset, fx.make_layout(block_m * 3, 1))
            bscales = fx.make_view(fptr, fx.make_layout(scale_count, 1))
            alive = fx.make_view(lds.arena.ptr + alive_offset, fx.make_layout(alive_count, 1))
            asbuf = fx.rocdl.make_buffer_tensor(fx.make_view(input_scales, fx.make_layout(rows * 2, 1)), False)
            asrsrc = fx.rocdl.get_buffer_rsrc(fx.get_iter(asbuf))
            zero = fx.Int32(0).ir_value()
            initial_id, initial_route = fx.Int32(0), fx.Float32(0.0)
            if const_expr(preload_metadata):
                id_buffer = fx.rocdl.make_buffer_tensor(fx.make_view(sorted_ids, fx.make_layout(capacity_blocks * sort_block_m, 1)), False)
                route_buffer = fx.rocdl.make_buffer_tensor(fx.make_view(sorted_weights, fx.make_layout(capacity_blocks * sort_block_m, 1)), False)
                metadata_offset = (tid < block_m).select((row_begin + tid) * 4, fx.Int32(-1))
                initial_id = fx.Int32(rocdl.raw_ptr_buffer_load(fx.Int32.ir_type,
                    fx.rocdl.get_buffer_rsrc(fx.get_iter(id_buffer)), metadata_offset.ir_value(), zero))
                initial_route = fx.Float32(rocdl.raw_ptr_buffer_load(fx.Float32.ir_type,
                    fx.rocdl.get_buffer_rsrc(fx.get_iter(route_buffer)), metadata_offset.ir_value(), zero))
                rocdl.sched_barrier(0)
            def seed_b_packets(count):
                # B has no dependency on sorted IDs/A scales. Fill unused LDS
                # slots before those dependent metadata loads can stall peers.
                seed_view = fx.make_view(weight + fx.Int64(expert) * (n * k) + fx.Int64(oc) * (n_split * k),
                                         fx.make_layout(n_split * k, 1))
                seed_buffer = fx.rocdl.make_buffer_tensor(seed_view, False)
                seed_resource = fx.rocdl.get_buffer_rsrc(fx.get_iter(seed_buffer))
                seed_base = _scalar(fx.Int32(fx.ptrtoint(bptr))) + wave * (64 * dma_bytes)
                seed_offset = _pin_address(tid * dma_bytes)
                for seed_q in range_constexpr(min(count, packets)):
                    seed_rotation = half_block if rotate_n == 1 else parent if rotate_n == 2 else task // 4
                    seed_index = ((seed_q // 2 + seed_rotation) % (packets // 2)) * 2 + seed_q % 2 if rotate_n else fx.Int32(seed_q)
                    for seed_turn in range_constexpr(slot_bytes // (threads * dma_bytes)):
                        seed_dest = llvm.inttoptr(ir.Type.parse("!llvm.ptr<3>"),
                            (seed_base + seed_q * slot_bytes + seed_turn * threads * dma_bytes).ir_value())
                        rocdl.raw_ptr_buffer_load_async_lds(
                            seed_resource, seed_dest, fx.Int32(dma_bytes).ir_value(), seed_offset.ir_value(),
                            (seed_index * slot_bytes + seed_turn * threads * dma_bytes).ir_value(), zero,
                            aux=ir.IntegerAttr.get(fx.Int32.ir_type, weight_cache_policy))
                    rocdl.asyncmark()
            if const_expr(prologue_b_prefetch):
                seed_b_packets(prologue_b_prefetch)
            if (tid < block_m) & fx.Boolean(not early_task_guard):
                encoded = (initial_id if const_expr(preload_metadata) else sorted_ids[row_begin + tid]).bitcast(fx.Uint32)
                token, slot = encoded & 0xFFFFFF, encoded >> 24
                valid = (token < fx.Uint32(tokens)) & (slot < topk)
                if const_expr(not direct_metadata):
                    ids[tid] = encoded.bitcast(fx.Int32)
                if const_expr(not presorted_a and not direct_metadata):
                    row_scales[block_m * 2 + tid] = initial_route if const_expr(preload_metadata) else sorted_weights[row_begin + tid]
                source_row = fx.Int32(token) * topk + fx.Int32(slot)
                if const_expr(not presorted_a and not direct_metadata):
                    for kb in range_constexpr(2):
                        offset = valid.select((kb * rows + source_row) * 4, fx.Int32(-1))
                        row_scales[kb * block_m + tid] = fx.Float32(rocdl.raw_ptr_buffer_load(
                            fx.Float32.ir_type, asrsrc, offset.ir_value(), zero,
                        ))
                mask = fx.Uint64(rocdl.ballot(fx.Uint64.ir_type, valid.ir_value()))
                if lane == 0:
                    alive[wave] = (mask != fx.Uint64(0)).select(fx.Int32(1), fx.Int32(0))
            if const_expr(not direct_metadata and not uniform_b_scales):
                for turn in range_constexpr((scale_count + threads - 1) // threads):
                    index = tid + turn * threads
                    if index < scale_count:
                        bscales[index] = weight_scales[expert * (n // 128 * 2) + oc * scale_count + index]
            if const_expr(not early_task_guard):
                fx.barrier()
            any_rows = fx.Int32(0)
            if const_expr(skip_empty_tasks and not early_task_guard):
                for i in range_constexpr(alive_count):
                    any_rows = any_rows | alive[i]
            has_rows = _scalar(any_rows) != 0 if const_expr(skip_empty_tasks and not early_task_guard) else fx.Boolean(True)
            if has_rows:
                if const_expr(queue_record):
                    if tid == 0:
                        counter[256 + queue_ctas * 4 + linear_task * 8 + 6] = 1
                if const_expr(record_placement):
                    if tid == 0:
                        counter[linear_task * 4 + 3] = 1
                if const_expr(active_b_prefetch):
                    # The task is live. Overlap B with independent A loads and
                    # output-address preparation, without padding-task traffic.
                    seed_b_packets(active_b_prefetch)
                blds = fxh.LdsTensor(fx.make_view(bptr, fx.make_layout(4, 1)))
                rlds = fxh.LdsTensor(fx.make_view(fptr + rows_offset, fx.make_layout(1, 1)))
                dma_base = _scalar(fx.Int32(fx.ptrtoint(bptr))) + wave * (64 * dma_bytes)
                dma_offset = _pin_address((tid // 128) * 4096 + (tid % 128) * 16 if half_k else tid * dma_bytes)
                scale_base = _scalar(fx.Int32(fx.ptrtoint(fptr)))
                scale_packet = fx.Int32(0)
                if const_expr(direct_metadata):
                    sbuf = fx.rocdl.make_buffer_tensor(fx.make_view(weight_scales, fx.make_layout(num_experts * n // 128 * 2, 1)), False)
                    ssrc = fx.rocdl.get_buffer_rsrc(fx.get_iter(sbuf))
                    soff = (lane < scale_count).select((expert * (n // 128 * 2) + oc * scale_count + lane) * 4, fx.Int32(-1))
                    scale_packet = fx.Int32(rocdl.raw_ptr_buffer_load(fx.Int32.ir_type, ssrc, soff.ir_value(), zero))
                wview = fx.make_view(weight + fx.Int64(expert) * (n * k) + fx.Int64(oc) * (n_split * k),
                                     fx.make_layout(n_split * k, 1))
                wbuf = fx.rocdl.make_buffer_tensor(wview, False)
                wrsrc = fx.rocdl.get_buffer_rsrc(fx.get_iter(wbuf))
                wbptr = fx.recast_iter(fx.PointerType.get(fx.Int32.ir_type, weight.memspace, 16), fx.get_iter(wview))
                wb32 = fx.rocdl.make_buffer_tensor(fx.make_view(wbptr, fx.make_layout(n_split * k // 4, 1)), False)
                wpacket = fxh.BufferTensor(fx.make_view(fx.get_iter(wb32), fx.make_layout(4, 1)))
                direct_offset = _pin_address(lr * 16 + lk * 256)
                asource = prepared_a if const_expr(presorted_a) else input_q
                arows = capacity_blocks * sort_block_m if const_expr(presorted_a) else rows
                aptr = fx.recast_iter(fx.PointerType.get(fx.Int32.ir_type, asource.memspace, 16), asource)
                abuf = fx.rocdl.make_buffer_tensor(fx.make_view(aptr, fx.make_layout(arows * 64, 1)), False)
                apacket = fxh.BufferTensor(fx.make_view(fx.get_iter(abuf), fx.make_layout(4, 1)))
                out_rows = capacity_blocks * packed_rows
                obuf = fx.rocdl.make_buffer_tensor(fx.make_view(output, fx.make_layout((tokens if atomic_sum else out_rows) * n, 1)), False)
                orsrc = fx.rocdl.get_buffer_rsrc(fx.get_iter(obuf))
                a = fx.make_rmem_tensor([8, m_fragments, 2], fx.Int32)
                awords = fx.make_view(fx.get_iter(a), fx.make_ordered_layout([4, 2, m_fragments, 2], 0))
                cached_rows = fx.make_rmem_tensor([m_fragments, 3], fx.Float32)
                out_addresses = []

                def row_id(row):
                    return (fx.Int32(fx.gpu.shuffle_idx(guard_ids, row, 64)) if register_metadata else
                            sorted_ids[row_begin + row] if direct_metadata else ids[row])

                for mi in range_constexpr(m_fragments):
                    row = wave_m * rows_per_wave + mi * 16 + lr
                    encoded = row_id(row).bitcast(fx.Uint32)
                    token, slot = encoded & 0xFFFFFF, encoded >> 24
                    valid = (token < fx.Uint32(tokens)) & (slot < topk)
                    source_row = row_begin + row if const_expr(presorted_a) else fx.Int32(token) * topk + fx.Int32(slot)
                    for kb in range_constexpr(2):
                        for part in range_constexpr(2):
                            offset = valid.select(source_row * 256 + lk * 16 + kb * 128 + part * 64, fx.Int32(-1))
                            awords[None, part, mi, kb].store(apacket.load(voffset_bytes=offset))
                    for kb in range_constexpr(3):
                        if const_expr(presorted_a):
                            cached_rows[mi, kb] = prepared_scales[(row_begin + row) * 3 + kb]
                        elif const_expr(direct_metadata):
                            if const_expr(kb < 2):
                                soff = valid.select((kb * rows + source_row) * 4, fx.Int32(-1))
                                cached_rows[mi, kb] = fx.Float32(rocdl.raw_ptr_buffer_load(fx.Float32.ir_type, asrsrc, soff.ir_value(), zero))
                            else:
                                cached_rows[mi, kb] = sorted_weights[row_begin + row]
                        else:
                            address = fx.Int32(fx.ptrtoint(fptr + rows_offset)) + (kb * block_m + row) * 4
                            cached_rows[mi, kb] = rlds.load(address_bytes=address)[0]
                    for parity in range_constexpr(1 if wave_tile_2x2 else 2):
                        target = (wave_m * rows_per_wave + mi * 16 + lr if const_expr(wave_tile_2x2) else
                                  wave_m * rows_per_wave + mi * 16 + (lr // 2) * 2 + parity if const_expr(coalesced_output and not raw_output)
                                  else wave_m * rows_per_wave + mi * 16 + lr)
                        encoded = row_id(target).bitcast(fx.Uint32)
                        token, slot = encoded & 0xFFFFFF, encoded >> 24
                        valid = (token < fx.Uint32(tokens)) & (slot < topk)
                        column = (wave % 2) * 32 + ((lk & 1) * 2 + (lk >> 1)) * 8 if const_expr(wave_tile_2x2) else (
                            ((lk & 1) * 2 + (lk >> 1)) * 8 + ((lr % 2) * 32 if const_expr(coalesced_output) else parity * 32))
                        offset = parent * (packed_rows * n * 2) + oc * (packed_rows * n_split * 2) + (half * block_m + target) * 128 + column * 2
                        if const_expr(double_n_packet):
                            offset = offset + (wave % 2) * 32768
                        if const_expr(output_nmajor):
                            offset = oc * (n_split // 64) * out_rows * 128 + (row_begin + target) * 128 + column * 2
                        if const_expr(atomic_sum):
                            offset = (fx.Int32(token) * n + oc * n_split + column) * 4
                        if const_expr(raw_output):
                            offset = parent * (256 * n * 2) + oc * (256 * n_split * 2) + (half * block_m // 32 + wave) * 4096 + (mi * 2 + parity) * 1024 + lane * 16
                        out_addresses.append(_pin_address(valid.select(offset, tokens * n * 4 if atomic_sum else out_rows * n * 2)))
                if const_expr(wide_output):
                    out_addresses = []
                    for mi in range_constexpr(m_fragments):
                        for parity in range_constexpr(2):
                            for side in range_constexpr(2):
                                target = wave * rows_per_wave + mi * 16 + (lr // 4) * 4 + side * 2 + parity
                                encoded = (sorted_ids[row_begin + target] if const_expr(direct_metadata) else ids[target]).bitcast(fx.Uint32)
                                token, slot = encoded & 0xFFFFFF, encoded >> 24
                                good = (token < fx.Uint32(tokens)) & (slot < topk)
                                column = ((lk & 1) * 2 + (lk >> 1)) * 8 + (lr % 2) * 32 + (lr // 2 % 2) * 64
                                offset = parent * (256 * n * 2) + oc * (256 * n_split * 2) + (half * block_m + target) * 256 + column * 2
                                out_addresses.append(_pin_address(good.select(offset, out_rows * n * 2)))
                b_addresses = [_pin_address(fx.Int32(fx.ptrtoint(bptr)) + lr * 16 + lk * 256 + slot * slot_bytes
                                            + ((wave % 2) * 16384 if double_n_packet else (wave % 2) * 8192 if wave_tile_2x2 else 0))
                               for slot in range_constexpr(ring_slots)]
                atom = fx.make_mma_atom(rocdl.cdna4.MFMA_Scale(16, 16, 128, fx.Float8E4M3FN))

                def packet_index(q):
                    rotation = half_block if rotate_n == 1 else parent if rotate_n == 2 else task // 4
                    return ((fx.Int32(q) + task) % packets if packet_rotation else
                            (fx.Int32(q) + fx.Int32(rotation)) % packets if double_n_packet and rotate_n else
                            ((fx.Int32(q) // 2 + fx.Int32(rotation)) % (packets // 2)) * 2 + fx.Int32(q) % 2 if rotate_n else fx.Int32(q))

                def dma_turn(q, slot, turn):
                    dst = llvm.inttoptr(ir.Type.parse("!llvm.ptr<3>"), (dma_base + slot * slot_bytes + turn * threads * dma_bytes).ir_value())
                    rocdl.raw_ptr_buffer_load_async_lds(
                        wrsrc, dst, fx.Int32(dma_bytes).ir_value(), dma_offset.ir_value(),
                        fx.Int32(q // 2 * 16384 + q % 2 * 2048 + turn * threads * 32
                                 if half_k else packet_index(q) * slot_bytes + turn * threads * dma_bytes).ir_value(), zero,
                        aux=ir.IntegerAttr.get(fx.Int32.ir_type, weight_cache_policy),
                    )

                def dma(q, slot):
                    # Every CTA cooperatively covers exactly one 16KiB packet.
                    for turn in range_constexpr(slot_bytes // (threads * dma_bytes)):
                        dma_turn(q, slot, turn)
                    rocdl.asyncmark()

                def store_one(packed, q, record):
                    if const_expr(atomic_sum):
                        values = packed[record].bitcast(fx.BFloat16).to(fx.Float32)
                        for value_id in range_constexpr(8):
                            rocdl.raw_ptr_buffer_atomic_fadd(
                                values[value_id].ir_value(), orsrc, out_addresses[record].ir_value(),
                                (fx.Int32(q) * 256 + value_id * 4).ir_value(),
                                aux=ir.IntegerAttr.get(fx.Int32.ir_type, 0),
                            )
                    else:
                        rocdl.raw_ptr_buffer_store(packed[record].ir_value(), orsrc, out_addresses[record].ir_value(),
                                                  (packet_index(q) * (out_rows * 128 if output_nmajor else 65536 if wide_output or double_n_packet else packed_rows * 128)).ir_value(),
                                                  aux=ir.IntegerAttr.get(fx.Int32.ir_type, output_cache_policy))

                def store(packed, q):
                    for record in range_constexpr(packed_vectors):
                        store_one(packed, q, record)

                def store_pair(first_pair, second_pair, q, mi):
                    combined = (_coalesce_output_pairs(first_pair, second_pair) if coalesced_output
                                else [first_pair, second_pair])
                    for side in range_constexpr(2):
                        rocdl.raw_ptr_buffer_store(combined[side].ir_value(), orsrc,
                            out_addresses[mi * 2 + side].ir_value(), (packet_index(q) * packed_rows * 128).ir_value(),
                            aux=ir.IntegerAttr.get(fx.Int32.ir_type, output_cache_policy))

                def memory(q, phase, packed, first=False, tail_left=100):
                    scatter_dma = dma_in_compute or (startup_refill_in_compute and isinstance(q, int) and q < 2)
                    early_refill = refill_before_wait and tail_left > prefetch_distance
                    spread_dma = spread_refill and not scatter_dma and tail_left > prefetch_distance
                    if const_expr(memory_priority != compute_priority):
                        rocdl.s_setprio(memory_priority)
                    # Completion waits belong at the next consumer, not before
                    # the previous Compute. This CTA's peers publish B here.
                    if const_expr(not global_b):
                        if const_expr(early_refill):
                            if const_expr(not isinstance(q, int) or q > 0):
                                # Retire all prior LDS readers before reusing
                                # their slot, independently of next B readiness.
                                _stage_end()
                            dma(q + prefetch_distance, (phase + prefetch_distance) % ring_slots)
                        rocdl.wait_asyncmark(min(prefetch_distance - 1 + int(early_refill), tail_left - 1))
                        _stage_end()
                    _mark(f"MOE128_MEMORY_BEGIN_{phase}")
                    if const_expr(not scatter_dma and not early_refill and not spread_dma and not global_b and refill_first and ring_slots > prefetch_distance and tail_left > prefetch_distance):
                        dma(q + prefetch_distance, (phase + prefetch_distance) % ring_slots)
                    b = fx.make_rmem_tensor([8, n_fragments, 2], fx.Int32)
                    words = fx.make_view(fx.get_iter(b), fx.make_ordered_layout([4, 2, n_fragments, 2], 0))
                    for kb in range_constexpr(2):
                        for ni in range_constexpr(n_fragments):
                            for part in range_constexpr(2):
                                if const_expr(spread_dma and (kb * n_fragments * 2 + ni * 2 + part) % 4 == 0):
                                    # Same four B turns, spaced by four LDS
                                    # reads; the slot was retired above.
                                    dma_turn(q + prefetch_distance, (phase + prefetch_distance) % ring_slots,
                                             (kb * n_fragments * 2 + ni * 2 + part) // 4)
                                if const_expr((not half_k or kb == phase % 2) and (not stream_b or kb == 0)):
                                    if const_expr(global_b):
                                        words[None, part, ni, kb].store(wpacket.load(
                                            voffset_bytes=direct_offset,
                                            soffset_bytes=fx.Int32(q * slot_bytes + ni * 4096 + kb * 2048 + part * 1024),
                                            aux=weight_cache_policy,
                                        ))
                                    else:
                                        words[None, part, ni, kb].store(blds.load(
                                            address_bytes=b_addresses[phase],
                                            offset_bytes=ni * 2048 + part * 1024 if half_k else ni * 4096 + kb * 2048 + part * 1024,
                                        ))
                    if const_expr(spread_dma):
                        rocdl.asyncmark()
                    scales = []
                    for kb in range_constexpr(2):
                        address = scale_base + (packet_index(q) // (1 if double_n_packet else 4 if half_n else 2 * k_packets) * 2 + kb) * 4
                        if const_expr(uniform_b_scales):
                            index = expert * (n // 128 * 2) + oc * scale_count + packet_index(q) // 2 * 2 + kb
                            scales.append(weight_scales[index].bitcast(fx.Int32))
                        elif const_expr(direct_metadata):
                            index = packet_index(q) // (1 if double_n_packet else 4 if half_n else 2 * k_packets) * 2 + kb
                            scales.append(fx.Int32(rocdl.readlane(fx.Int32.ir_type, scale_packet.ir_value(), index.ir_value())))
                        else:
                            raw = llvm.inline_asm(ir.Type.parse("!llvm.struct<(i32, i32)>"), [address.ir_value()],
                                "s_mov_b32 $1, m0\ns_mov_b32 m0, $2\ns_nop 1\nds_read_addtid_b32 $0\ns_mov_b32 m0, $1",
                                "=&v,=&s,s,~{memory}", has_side_effects=True)
                            scales.append(fx.Int32(llvm.extractvalue(fx.Int32.ir_type, raw, [0])))
                    if const_expr(not stores_in_compute and not first):
                        if const_expr(not (half_k or half_n or wide_output) or phase % 2 == 0):
                            store(packed, q // 2 - 1 if half_k or half_n or wide_output else q - (1 if pack_immediate else 2))
                    if const_expr(global_b):
                        rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0)
                    else:
                        rocdl.s_waitcnt(lgkmcnt=0)
                    if const_expr(not global_b and ring_slots == prefetch_distance):
                        # All waves must finish reading the current slot before
                        # any producer overwrites it. Independent CTAs proceed.
                        _stage_end()
                    if const_expr(not scatter_dma and not global_b and not pair_refill and (not refill_first or ring_slots == prefetch_distance) and tail_left > prefetch_distance):
                        dma(q + prefetch_distance, (phase + prefetch_distance) % ring_slots)
                    for kb in range_constexpr(2):
                        for ni in range_constexpr(n_fragments):
                            for part in range_constexpr(2):
                                if const_expr((not half_k or kb == phase % 2) and (not stream_b or kb == 0)):
                                    view = words[None, part, ni, kb]
                                    view.store(_pin_packet(view.load()))
                    _mark(f"MOE128_MEMORY_END_{phase}")
                    return b, scales

                def compute(b, scales, previous, phase, q, old_words, first=False, tail_left=100):
                    scatter_dma = dma_in_compute or (startup_refill_in_compute and isinstance(q, int) and q < 2)
                    eager_output = eager_first_store and isinstance(q, int) and q == 0
                    ready_output = first_ready_store and isinstance(q, int) and q == 1
                    skip_previous_pack = first or (eager_first_store and isinstance(q, int) and q == 1)
                    if const_expr(memory_priority != compute_priority):
                        rocdl.s_setprio(compute_priority)
                    _mark(f"MOE128_COMPUTE_BEGIN_{phase}")
                    if const_expr(pair_refill and tail_left > prefetch_distance):
                        # Memory has retired every LDS reader. Send two turns
                        # now and two halfway through useful MFMA work.
                        dma_turn(q + prefetch_distance, phase, 0)
                        dma_turn(q + prefetch_distance, phase, 1)
                    factors = [[cached_rows[mi, kb] * _scalar(scales[kb]).bitcast(fx.Float32) for mi in range_constexpr(m_fragments)]
                               for kb in range_constexpr(2)]
                    accum = fx.make_rmem_tensor([4, m_fragments, n_fragments], fx.Float32)
                    scaled = [[] for _ in range_constexpr(packed_vectors)]
                    words = [[] for _ in range_constexpr(packed_vectors)]
                    swaps = [[] for _ in range_constexpr(packed_vectors)]
                    packed = [[] for _ in range_constexpr(m_fragments)]
                    pending, dequant, pack_index = [], [], 0

                    def retire_vector(partial, mi, ni, kb):
                        value = partial.load()
                        factor = fx.Vector.filled(4, factors[kb][mi], fx.Float32)
                        if const_expr(kb == 0):
                            accum[None, mi, ni].store(value * factor)
                        else:
                            accum[None, mi, ni].store(fx.Vector(fxh.eltwise_op(
                                "llvm.fma.f32", value, factor, previous[mi * 4 + ni] if half_k else accum[None, mi, ni].load(),
                            )))

                    def pack_op(index):
                        pair_id, op = index // 14, index % 14
                        mi, pair = (pair_id, 0) if wave_tile_2x2 else (pair_id // 2, pair_id % 2)
                        if const_expr(op < 8):
                            scaled[pair_id].append(previous[mi * n_fragments + pair * 2 + op // 4][op % 4] * cached_rows[mi, 2])
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
                            accum[element, mi, ni] = fx.Float32(fxh.eltwise_op("llvm.fma.f32", value, factors[kb][mi],
                                                                           previous[mi * 4 + ni][element] if half_k else accum[element, mi, ni]))

                    for index in range_constexpr(accum_vectors * (1 if half_k else 2)):
                        if const_expr(pair_refill and index == 4 and tail_left > prefetch_distance):
                            dma_turn(q + prefetch_distance, phase, 2)
                            dma_turn(q + prefetch_distance, phase, 3)
                            rocdl.asyncmark()
                        if const_expr(stream_b and index == accum_vectors):
                            for ni in range_constexpr(n_fragments):
                                for part in range_constexpr(2):
                                    fragment = fxh.LdsTensor(fx.make_view(bptr, fx.make_layout(4, 1))).load(
                                        address_bytes=b_addresses[phase], offset_bytes=ni * 4096 + 2048 + part * 1024,
                                    )
                                    bwords = fx.make_view(fx.get_iter(b), fx.make_ordered_layout([4, 2, n_fragments, 2], 0))
                                    bwords[None, part, ni, 1].store(fragment)
                            rocdl.s_waitcnt(lgkmcnt=0)
                            for ni in range_constexpr(n_fragments):
                                for part in range_constexpr(2):
                                    view = bwords[None, part, ni, 1]
                                    view.store(_pin_packet(view.load()))
                        if const_expr(scatter_dma and index % dma_mfma_stride == 0 and tail_left > prefetch_distance):
                            dma_turn(q + prefetch_distance, (phase + prefetch_distance) % ring_slots, index // dma_mfma_stride)
                        if const_expr(stores_in_compute and index % 4 == 2):
                            if const_expr(not isinstance(q, int) or q >= (3 if eager_first_store or first_ready_store else 2)):
                                store_one(old_words, q - 2, index // 4)
                        if const_expr(scatter_dma and index == accum_vectors * 2 - 1 and tail_left > prefetch_distance):
                            rocdl.asyncmark()
                        kb, local = phase % 2 if half_k else index // accum_vectors, index % accum_vectors
                        mi, ni = local // n_fragments, local % n_fragments
                        partial = fx.make_rmem_tensor(4, fx.Float32)
                        partial.fill(0)
                        fx.gemm(atom, partial, b[None, ni, kb], a[None, mi, kb], partial)
                        if const_expr(not relaxed_compute):
                            rocdl.sched_barrier(0)
                        pending.append((partial, mi, ni, kb))
                        if const_expr(len(pending) > retire_distance):
                            old = pending.pop(0)
                            if const_expr(packed_fp32):
                                retire_vector(*old)
                            else:
                                dequant.extend([(*old, element) for element in range_constexpr(4)])
                        remaining = valu_budget
                        for _ in range_constexpr(min(4, len(dequant))):
                            retire(dequant.pop(0))
                            remaining -= 1
                        while const_expr(not pack_immediate and not packed_fp32 and not skip_previous_pack and pack_index < pack_operations and remaining > 0):
                            pack_op(pack_index)
                            pack_index += 1
                            remaining -= 1
                        while const_expr(len(dequant) > 0 and remaining > 0):
                            retire(dequant.pop(0))
                            remaining -= 1
                        if const_expr(not pack_immediate and packed_fp32 and not skip_previous_pack and index < packed_vectors):
                            pack_mi, pair = index // 2, index % 2
                            packed[pack_mi].append(_pack_pair(previous[pack_mi * 4 + pair * 2],
                                                            previous[pack_mi * 4 + pair * 2 + 1], cached_rows[pack_mi, 2]))
                        if const_expr(ready_output and index == 6):
                            # The first C0 row-fragment pair is already packed;
                            # publish it now instead of deferring to Compute2.
                            store_pair(packed[0][0], packed[0][1], 0, 0)
                        if const_expr(not relaxed_compute):
                            rocdl.sched_barrier(0)
                    for old in pending:
                        if const_expr(packed_fp32):
                            retire_vector(*old)
                        else:
                            dequant.extend([(*old, element) for element in range_constexpr(4)])
                    for item in dequant:
                        retire(item)
                    while const_expr(not pack_immediate and not packed_fp32 and not skip_previous_pack and pack_index < pack_operations):
                        pack_op(pack_index)
                        pack_index += 1
                    result = ([fx.Vector.filled(4, 0.0, fx.Float32) for _ in range_constexpr(accum_vectors)] if pack_immediate and (not half_k or phase % 2 == 1) else
                              [_pin_packet(accum[None, mi, ni].load()) for mi in range_constexpr(m_fragments) for ni in range_constexpr(n_fragments)])
                    output_words = []
                    if const_expr(pack_immediate and (not half_k or phase % 2 == 1)):
                        for mi in range_constexpr(m_fragments):
                            for pair in range_constexpr(n_fragments // 2):
                                if const_expr(raw_output):
                                    first = (accum[None, mi, pair * 2].load() * cached_rows[mi, 2]).to(fx.BFloat16).bitcast(fx.Int32)
                                    second = (accum[None, mi, pair * 2 + 1].load() * cached_rows[mi, 2]).to(fx.BFloat16).bitcast(fx.Int32)
                                    packed[mi].append(_pin_packet(fx.Vector.from_elements([first[0], first[1], second[0], second[1]], fx.Int32)))
                                else:
                                    packed[mi].append(_pack_pair(accum[None, mi, pair * 2].load(), accum[None, mi, pair * 2 + 1].load(), cached_rows[mi, 2]))
                    if const_expr(not ready_output and (pack_immediate or not skip_previous_pack) and (not half_k or phase % 2 == 1)):
                        for mi in range_constexpr(m_fragments):
                            if const_expr(half_n):
                                if const_expr(phase % 2 == 0):
                                    output_words.extend([packed[mi][0], fx.Vector.filled(4, 0, fx.Int32)])
                                else:
                                    output_words.extend(_coalesce_output_pairs(old_words[mi * 2], packed[mi][0]))
                            else:
                                output_words.extend(_coalesce_output_pairs(packed[mi][0], packed[mi][1]) if coalesced_output and not raw_output and not wave_tile_2x2 else packed[mi])
                    if const_expr(wide_output):
                        paired_words = []
                        for record in range_constexpr(m_fragments * 2):
                            if const_expr(phase % 2 == 0):
                                paired_words.extend([output_words[record], fx.Vector.filled(4, 0, fx.Int32)])
                            else:
                                paired_words.extend(_coalesce_n128(old_words[record * 2], output_words[record]))
                        output_words = paired_words
                    if const_expr(eager_output):
                        # There is no previous C to interleave in Compute0.
                        # Retire its real output now, before the B1 wait, and
                        # suppress the normal duplicate pack/store at q1/q2.
                        for mi in range_constexpr(m_fragments):
                            pair0 = _pack_pair(result[mi * 4], result[mi * 4 + 1], cached_rows[mi, 2])
                            pair1 = _pack_pair(result[mi * 4 + 2], result[mi * 4 + 3], cached_rows[mi, 2])
                            store_pair(pair0, pair1, 0, mi)
                    if const_expr(ready_output):
                        store_pair(packed[1][0], packed[1][1], 0, 1)
                    _mark(f"MOE128_COMPUTE_END_{phase}")
                    return result, (output_words if not half_k or phase % 2 == 1 else
                                    [fx.Vector.filled(4, 0, fx.Int32) for _ in range_constexpr(packed_vectors)])

                rocdl.sched_barrier(0)
                if const_expr(not global_b):
                    for q in range_constexpr(min(prologue_b_prefetch + active_b_prefetch, packets), min(prefetch_distance, packets)):
                        dma(q, q)
                current = [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range_constexpr(accum_vectors)]
                packed = [fx.Vector.filled(4, 0, fx.Int32) for _ in range_constexpr(packed_vectors)]
                prologue_steps = min(max(ring_slots, 2), packets)
                for q in range_constexpr(prologue_steps):
                    b, scales = memory(q, q % ring_slots, packed, first=q < (1 if pack_immediate else 2), tail_left=packets - q)
                    current, packed = compute(b, scales, current, q % ring_slots, q, packed, first=q == 0, tail_left=packets - q)
                loop_end = max(prologue_steps, (packets - prefetch_distance) // ring_slots * ring_slots)
                if const_expr(packets > prologue_steps):
                    for q, state in range(fx.Int32(prologue_steps), fx.Int32(loop_end), fx.Int32(ring_slots), init=[*current, *packed]):
                        current = [fx.Vector(state[i]) for i in range_constexpr(accum_vectors)]
                        packed = [fx.Vector(state[accum_vectors + i]) for i in range_constexpr(packed_vectors)]
                        for phase in range_constexpr(ring_slots):
                            b, scales = memory(q + phase, phase, packed)
                            current, packed = compute(b, scales, current, phase, q + phase, packed)
                        final = yield [*current, *packed]
                    current = [fx.Vector(final[i]) for i in range_constexpr(accum_vectors)]
                    packed = [fx.Vector(final[accum_vectors + i]) for i in range_constexpr(packed_vectors)]
                    for q in range_constexpr(loop_end, packets):
                        b, scales = memory(q, q % ring_slots, packed, tail_left=packets - q)
                        current, packed = compute(b, scales, current, q % ring_slots, q, packed, tail_left=packets - q)
                if const_expr((not (eager_first_store or first_ready_store) or packets > 2) and packets >= 2):
                    store(packed, packets // (2 if half_n or wide_output else k_packets) - (1 if pack_immediate else 2))
                final_packed = []
                if const_expr(not pack_immediate):
                    for mi in range_constexpr(m_fragments):
                        pieces = []
                        for pair in range_constexpr(n_fragments // 2):
                            pieces.append(_pack_pair(current[mi * n_fragments + pair * 2], current[mi * n_fragments + pair * 2 + 1], cached_rows[mi, 2]))
                        final_packed.extend(_coalesce_output_pairs(pieces[0], pieces[1]) if coalesced_output and not wave_tile_2x2 else pieces)
                    store(final_packed, packets - 1)
                if const_expr(not global_b):
                    rocdl.wait_asyncmark(0)
                rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0)
            if const_expr(prologue_b_prefetch):
                # Even an all-padding CTA must drain speculative LDS writes
                # before its allocation can be recycled for another CTA.
                rocdl.wait_asyncmark(0)
                rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0)
            if const_expr(static_ctas):
                fx.barrier()
                # Static persistent workers retain the virtual-grid transpose.
                # For P=static_ctas divisible by8, (bid+k*P)%8==bid%8; measured
                # bid%8 placement can preserve XCD affinity, but not time balance.
                linear_task = linear_task + static_ctas
                task = linear_task
                if const_expr(xcd_swizzle):
                    chunk = active_tasks // xcd_count
                    mapped = (task % xcd_count) * chunk + task // xcd_count
                    task = (task < chunk * xcd_count).select(mapped, task)
            elif const_expr(global_ctas):
                # The task epilogue above already drains B DMA, VMEM and LDS.
                # All readers must retire before any wave starts the next task;
                # the loop-header claim adds the task-ID publication barrier.
                fx.barrier()
            elif const_expr(queue_ctas):
                # Drain every wave's B DMA, VMEM and LDS readers before the
                # next logical task overwrites the same CTA-private arena.
                rocdl.wait_asyncmark(0)
                rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0)
                fx.barrier()
                linear_task = take_xcd_task(queue_slots)
                task = linear_task
                chunk = active_tasks // xcd_count
                mapped = (task % xcd_count) * chunk + task // xcd_count
                task = (task < chunk * xcd_count).select(mapped, task)
            else:
                task = active_tasks

    @flyc.jit
    def launch(output: fx.Pointer, input_q: fx.Pointer, weight: fx.Pointer,
               input_scales: fx.Pointer, weight_scales: fx.Pointer,
               sorted_ids: fx.Pointer, sorted_weights: fx.Pointer, expert_ids: fx.Pointer,
               valid_ids: fx.Pointer, counter: fx.Pointer, prepared_a: fx.Pointer, prepared_scales: fx.Pointer, tokens: fx.Int32,
               capacity_blocks: fx.Int32, stream: fx.Stream):
        moe_down_m128_kernel(output, input_q, weight, input_scales, weight_scales, sorted_ids,
                             sorted_weights, expert_ids, valid_ids, counter, prepared_a, prepared_scales, tokens, capacity_blocks,
                             value_attrs=attrs).launch(grid=(global_ctas if const_expr(global_ctas) else queue_ctas if const_expr(queue_ctas) else static_ctas if const_expr(static_ctas) else capacity_blocks * m_splits * num_oc_splits, 1, 1), block=(threads, 1, 1), stream=stream)

    workspace = {}
    prepare_rows = make_prepare_rows(topk) if presorted_a else None

    def down(output, input_q, weight, input_scales, weight_scales, sorted_ids, sorted_weights, expert_ids, valid_ids, counter):
        tokens, actual_topk, actual_k = input_q.shape
        tensors = (output, input_q, weight, input_scales, weight_scales, sorted_ids, sorted_weights, expert_ids, valid_ids, counter)
        assert (actual_topk, actual_k) == (topk, k) and 0 < tokens < (1 << 24)
        assert weight.shape == (num_experts, n, k) and output.shape == ((tokens, n) if atomic_sum else (expert_ids.numel() * packed_rows, n))
        assert input_q.dtype == weight.dtype == torch.float8_e4m3fn and output.dtype == (torch.float32 if atomic_sum else torch.bfloat16)
        assert input_scales.dtype == weight_scales.dtype == sorted_weights.dtype == torch.float32
        assert sorted_ids.dtype == expert_ids.dtype == valid_ids.dtype == counter.dtype == torch.int32
        assert input_scales.numel() == tokens * topk * 2 and weight_scales.shape == (num_experts, n // 128, 2)
        assert sorted_ids.shape == sorted_weights.shape and expert_ids.ndim == sorted_ids.ndim == 1
        assert expert_ids.numel() == (sorted_ids.numel() + sort_block_m - 1) // sort_block_m, "sorting block size does not match kernel"
        assert counter.numel() == ((1 + global_ctas * 4 + expert_ids.numel() * num_oc_splits * 4 if global_record else 1)
                      if global_ctas else (256 + queue_ctas * 4 + expert_ids.numel() * 8 * 8 if queue_record else 256)
                      if queue_ctas else expert_ids.numel() * m_splits * num_oc_splits * 4 if record_placement else 1)
        assert valid_ids.numel() >= 1
        assert all(t.is_cuda and t.is_contiguous() and t.device == output.device for t in tensors)
        assert (output.numel() + n) * output.element_size() < (1 << 32) and input_q.numel() < (1 << 32)
        assert input_scales.numel() * input_scales.element_size() < (1 << 32) and n_split * k < (1 << 32)
        assert torch.cuda.get_device_properties(output.device).gcnArchName.startswith("gfx950")
        if global_ctas:
            counter.zero_()
        pa, ps = input_q, input_scales
        if presorted_a:
            shape = (expert_ids.numel() * sort_block_m, 256)
            if "a" not in workspace or workspace["a"].shape != shape or workspace["a"].device != output.device:
                workspace["a"] = torch.empty(shape, dtype=input_q.dtype, device=output.device)
                workspace["s"] = torch.empty((shape[0], 3), dtype=torch.float32, device=output.device)
            pa, ps = workspace["a"], workspace["s"]
            prepare_rows(pa, ps, input_q, input_scales, sorted_ids, sorted_weights, valid_ids)
        launch_tensors = (*tensors, pa, ps)
        stream = torch.cuda.current_stream(output.device)
        compiled = getattr(launch, "_cf", None)
        if compiled is None:
            _run_compiled(launch, *[_ptr(t) for t in launch_tensors], fx.Int32(tokens), fx.Int32(expert_ids.numel()), fx.Stream(stream.cuda_stream))
        else:
            compiled(*(t.data_ptr() for t in launch_tensors), tokens, expert_ids.numel(), stream.cuda_stream)
        return output

    down.config = {"block_m": block_m, "sort_block_m": sort_block_m, "block_n": 128,
                   "num_waves": threads // 64, "num_oc_splits": num_oc_splits,
                   "prefetch_distance": prefetch_distance, "ring_slots": ring_slots, "consumer_wait": True,
                   "output_cache_policy": output_cache_policy, "weight_cache_policy": weight_cache_policy,
                   "persistent": bool(static_ctas or queue_ctas or global_ctas), "static_ctas": static_ctas,
                   "global_ctas": global_ctas, "global_record": global_record,
                   "queue_ctas": queue_ctas, "queue_batch": queue_batch, "queue_record": queue_record,
                   "queue_shards": 8 if queue_ctas else 0, "tail_steal": bool(queue_ctas),
                   "xcd_swizzle": xcd_swizzle, "xcd_count": xcd_count,
                   "task_m_group": task_m_group, "n_major": n_major,
                   "output_nmajor": output_nmajor,
                   "skip_empty_tasks": skip_empty_tasks, "refill_first": refill_first, "waves_per_eu": waves_per_eu,
                   "coalesced_output": coalesced_output, "packed_fp32": packed_fp32,
                   "global_b": global_b,
                   "rows_per_wave": rows_per_wave,
                   "dma_in_compute": dma_in_compute, "stores_in_compute": stores_in_compute,
                   "pack_immediate": pack_immediate,
                   "half_k": half_k,
                   "raw_output": raw_output,
                   "presorted_a": presorted_a,
                   "atomic_sum": atomic_sum,
                   "half_n": half_n,
                   "stream_b": stream_b,
                   "direct_metadata": direct_metadata, "unsigned_tasks": unsigned_tasks,
                   "early_task_guard": early_task_guard,
                   "wide_output": wide_output,
                   "packed_rows": packed_rows,
                   "valu_budget": valu_budget, "retire_distance": retire_distance, "relaxed_compute": relaxed_compute,
                   "rotate_n": rotate_n,
                   "memory_priority": memory_priority, "compute_priority": compute_priority,
                   "prologue_b_prefetch": prologue_b_prefetch, "startup_refill_in_compute": startup_refill_in_compute,
                   "preload_metadata": preload_metadata, "refill_before_wait": refill_before_wait,
                   "eager_first_store": eager_first_store,
                   "active_b_prefetch": active_b_prefetch,
                   "spread_refill": spread_refill, "uniform_b_scales": uniform_b_scales,
                   "first_ready_store": first_ready_store,
                   "record_placement": record_placement,
                   "wave_tile_2x2": wave_tile_2x2,
                   "register_metadata": register_metadata, "dma_bytes": dma_bytes,
                   "packet_rotation": packet_rotation,
                   "pair_refill": pair_refill,
                   "double_n_packet": double_n_packet,
                   "output_layout": "packed", "weight_layout": (16, 16), "lds_bytes": arena_words * 4}
    return down