# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx950 block-scaled MoE down: staggered waves, direct LDS, cached defaults.

BN128/K256 uses two Memory/Compute pairs (four stages).  Each Compute
consumes N64 x K256, issues sixteen MFMA16x16x128, and packs the completed
other N64 half.  The next tile's two Memory stages each write half of the
previous output.  Q[2*n+s] = B[n, half=s, full K256].

BN256/K256 keeps four Memory/Compute pairs and Q[4*n+s] = B[n,s//2,s%2].
Both use a four-slot direct-LDS ring, seed Q0/Q1 when present, and prefetch
Q[q+2].  Wave0..3 are one stage ahead of wave4..7.  LDS reads and the next
packet's DMA complete before the staggered barrier; no C data uses LDS.

BN256 packs old C2/C3 in Compute0/1 and current C0/C1 in Compute2/3, so the
BF16 work is independent of each stage's MFMA.  BN128 with K>256 retains its
BK128 schedule.  All paths default to cached memory access and use VGPR-only
code generation, with register-only BF16 conversion/permlane16_swap.
BN128 four-stage and BN256 eight-stage K256 paths optionally use fused DPP
coalescing for 128B-contiguous output, with non-temporal stores selected by
output_cache_policy.  The factory and returned ten-tensor callable retain
the original down-kernel interface.

The opt-in BN256 sixteen-stage mode splits each Memory/Compute pair in two:
N64 x K128 packets, eight MFMAs per Compute, two delayed stores and one DMA
per steady Memory.  steady_vmcnt=6/9 leaves two/three newer VMEM groups.

BN64/K256 uses two N32 x K256 packets and eight MFMAs per Compute.  Its
two Memory stages each retire half the preceding output (two stores) and
issue one direct-LDS packet.  prefetch_distance=1/2/3 means Memory q issues
Q[q+d]; its end waits for Q[q+1], leaving 3*(d-1) newer steady VMEM events.
"""

from functools import cache
from math import gcd

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import arith, const_expr, range_constexpr, rocdl

from pyhip.contrib.flydsl import helpers as fxh
from pyhip.contrib.flydsl.moe_gemm_2stage.common import torch_tensor_to_pointer as _ptr


# Explicit opt-in preset, measured on MI350X at N6144/K256/topk8/E384.
# It does not change the generic factory's defaults or the floating-point order.
ATT_TUNED_BN128_CONFIG = {
    "block_n": 128,
    "num_oc_splits": 4,
    "prefetch_distance": 3,
    "output_cache_policy": 2,  # gfx950: NT, not SC0/SC1
    "weight_cache_policy": 16,  # gfx950: SC1
    "persistent_workgroups": 256,
    "lds_read_first": True,
    "coalesced_output": True,
    "stage_priority": False,
}


def _scalar(value):
    return fx.Int32(rocdl.readfirstlane(fx.Int32.ir_type, arith._to_raw(value)))


def _pin_address(value):
    # Materialize invariant lane-dependent addresses in task preparation.
    return fx.Int32(llvm.inline_asm(
        fx.Int32.ir_type, [arith._to_raw(value)], "", "=v,0", has_side_effects=True,
    ))


def _task_thread_id(wave):
    # Reconstruct lane ID at each task boundary rather than spilling the
    # entry workitem-id VGPR across the entire persistent GEMM.
    lane = fx.Int32(llvm.inline_asm(
        fx.Int32.ir_type, [],
        "v_mbcnt_lo_u32_b32 $0, -1, 0\nv_mbcnt_hi_u32_b32 $0, -1, $0",
        "=v", has_side_effects=True,
    ))
    return wave * 64 + lane


def _pin_packet(value):
    # Allocate the contiguous four-VGPR store operand before entering Memory.
    raw = arith._to_raw(value)
    return fx.Vector(llvm.inline_asm(
        raw.type, [raw], "", "=v,0", has_side_effects=True,
    ))


def _coalesce_output_pairs(first, second):
    """Exchange one row bit with a contiguous output-column bit."""
    mask = fx.Uint64(0xAAAAAAAAAAAAAAAA)
    permutation = "[1,0,3,2]"
    even = fx.Uint64(0x5555555555555555)
    assembly = ["s_mov_b64 vcc, $16"]
    assembly.extend(f"v_cndmask_b32_dpp ${i}, ${12+i}, ${8+i}, vcc quad_perm:{permutation} row_mask:0xf bank_mask:0xf"
                    for i in range(4))
    assembly.append("s_mov_b64 vcc, $17")
    assembly.extend(f"v_cndmask_b32_dpp ${4+i}, ${8+i}, ${12+i}, vcc quad_perm:{permutation} row_mask:0xf bank_mask:0xf"
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
    # Assembly comments only: useful for checking the real scheduled ISA.
    rocdl.sched_barrier(0)
    llvm.inline_asm(ir.Type.parse("!llvm.void"), [], f"; {text}", "", has_side_effects=True)
    rocdl.sched_barrier(0)


def _pack_pair(c0, c1, route):
    """Same BF16 rounding and register-only permutation as moe_8wave_down."""
    c0 = c0 * route
    c1 = c1 * route
    # Native fptrunc lowers to the same cvt_pk_bf16_f32 but, unlike inline
    # assembly, participates in the MFMA/VALU scheduler and hazard tracking.
    words0 = c0.to(fx.BFloat16).bitcast(fx.Int32)
    words1 = c1.to(fx.BFloat16).bitcast(fx.Int32)
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
def flydsl_moe_gemm_8wave_down(
    *, n, k, topk, num_experts, block_m=256, block_n=256, num_oc_splits=1,
    bn256_stages=8, steady_vmcnt=6, prefetch_distance=None,
    output_cache_policy=0, weight_cache_policy=0, persistent_workgroups=256,
    lds_read_first=False, coalesced_output=False, stage_priority=True,
    output_layout="routed",
    persistent=True, xcd_swizzle=False, task_table=False,
    overlap_prologue=False, fused_metadata=False, batch_routing_ids=False,
    early_b_prefetch=False,
):
    """Return the same ten-tensor callable as moe_8wave_down's factory.

    A is contiguous OCP FP8 [tokens, topk, K], with physical K-major 1x128
    scales.  B uses shuffle_weight(layout=(16, 16)), with 128x128 scales.
    Output defaults to contiguous BF16 [tokens, topk, N], including routing
    weights. BN128/K256 with coalesced_output can explicitly use sorted or
    packed output in a caller-owned [expert_blocks*256,N] allocation. Packed
    physical order is [expert_block,OC_split,N64,row256,col64]; consume it
    with the matching down+reduce pipeline, not torch.sum on its shape.
    The caller owns the int32 task counter; it is reset on the current stream.
    prefetch_distance selects Q[q+d] (default2); steady_vmcnt only selects
    the separate BN256 sixteen-stage experiment.  ATT_TUNED_BN128_CONFIG
    is an opt-in K256 configuration; its four splits require N divisible512.
    persistent=False launches one workgroup per task-capacity slot, with
    device valid-count guards; xcd_swizzle optionally transposes eight XCDs.
    task_table interprets expert IDs as [physical_row_begin, expert] pairs
    and num_valid_ids[0] as full-task-count*256. Full rows must be M256 aligned.
    """
    assert block_m == 256
    assert not xcd_swizzle or not persistent, "XCD swizzle maps nonpersistent workgroup IDs"
    assert not (overlap_prologue or fused_metadata or batch_routing_ids or early_b_prefetch) or (
        block_n == 128 and k == 256 and coalesced_output and not persistent and not task_table
    ), "prologue experiments require the nonpersistent coalesced BN128/K256 path"
    assert not early_b_prefetch or not overlap_prologue, "select one A/B startup ordering"
    assert output_layout in ("routed", "sorted", "packed")
    assert output_layout == "routed" or (block_n == 128 and k == 256 and coalesced_output)
    assert output_cache_policy in (0, 1, 2, 3, 16, 17, 18, 19)
    assert weight_cache_policy in (0, 1, 2, 3, 16, 17, 18, 19)
    assert not coalesced_output or (block_n in (128, 256) and k == 256 and bn256_stages == 8)
    assert persistent_workgroups > 0 and persistent_workgroups % 8 == 0
    assert block_n in (64, 128, 256)
    assert k in (256, 384, 512, 640), "this is the BK128-multiple k128n path"
    assert block_n == 128 or k == 256, "BN64/BN256 require K256; use BN128 for larger K"
    small_n64 = block_n == 64
    assert bn256_stages in (8, 16)
    split_memory = bn256_stages == 16
    assert not split_memory or (block_n == 256 and k == 256), "16-stage requires BN256/K256"
    assert steady_vmcnt in (6, 9)
    assert split_memory or steady_vmcnt == 6, "steady_vmcnt requires BN256 sixteen-stage; use prefetch_distance for BN64"
    assert prefetch_distance is None or prefetch_distance in (1, 2, 3, 4)
    assert n > 0 and num_oc_splits > 0 and n % num_oc_splits == 0
    assert 0 < topk <= min(num_experts, 255)
    n_split = n // num_oc_splits
    assert n_split % block_n == 0
    ks = k // 128
    nt = n_split // block_n
    full_k_narrow = block_n in (64, 128) and k == 256
    half_n = block_n // 2
    record_count = 2 if small_n64 else 4
    record_n = block_n // record_count
    record_atoms = record_n // 16
    record_pairs = record_n // 32
    steps = 8 if split_memory else 2 if full_k_narrow else 2 * ks
    # Two stores followed by one DMA per microstage: wait for Q[q+1]
    # leaving (prefetch-1)*3 VMEM events in steady state.  Startup/tail
    # counts are reduced by wait_asyncmark, not a fixed raw vmcnt override.
    prefetch = prefetch_distance if prefetch_distance is not None else steady_vmcnt // 3 + 1 if split_memory else 2
    ring_slots = 8 if split_memory or small_n64 or prefetch >= 4 else 4
    packet_n = 64 if split_memory else half_n
    first_unpacked = 1 if small_n64 else 2 if block_n == 256 or full_k_narrow else 3
    slot_k = k if full_k_narrow else 128
    slot_bytes = packet_n * slot_k
    dma_rounds = slot_bytes // (512 * 16)
    scale_count = ((n_split + 127) // 128) * ks
    scale_words = (scale_count + 63 + 255) // 256 * 256
    ids_offset = scale_words + ring_slots * slot_bytes // 4
    row_offset = ids_offset + 256
    task_offset = row_offset + 256 * (ks + 1)
    assert (task_offset + 1) * 4 <= 160 * 1024, "per-task cache exceeds gfx950 LDS capacity"
    tile_stride = ring_slots // gcd(steps, ring_slots)
    tail_tiles = (prefetch + steps - 1) // steps
    loop_end = 2 + max(0, (nt - tail_tiles - 2) // tile_stride) * tile_stride
    pack_ops_per_pair = 14
    pack_ops = pack_ops_per_pair * (2 if split_memory or small_n64 else 4)
    mfma_count = 8 if split_memory or small_n64 else 16
    kernel_attrs = {
        "rocdl.waves_per_eu": 2,
        "rocdl.flat_work_group_size": "512,512",
        # waves_per_eu controls occupancy, not whether AGPRs can be allocated.
        # llvm.passthrough survives gpu.func -> llvm.func lowering.
        "llvm.passthrough": [
            ["target-features", "-packed-fp32-ops"],
            ["amdgpu-agpr-alloc", "0,0"],
        ],
    }

    @fx.struct
    class SharedStorage:
        # Fix the scale window and ring alignment in one allocation.  The
        # final 63 scale words cover the unused lanes of the last addtid read.
        arena: fx.Array[fx.Int32, task_offset + 1, 16]

    @flyc.kernel(known_block_size=[512, 1, 1])
    def moe_down_8stage_kernel(
        output: fx.Pointer,
        input_q: fx.Pointer,
        weight_shuffled: fx.Pointer,
        input_scales: fx.Pointer,
        weight_scales: fx.Pointer,
        sorted_ids: fx.Pointer,
        sorted_weights: fx.Pointer,
        sorted_expert_ids: fx.Pointer,
        num_valid_ids: fx.Pointer,
        task_counter: fx.Pointer,
        num_tokens: fx.Int32,
        num_expert_blocks: fx.Int32,
        output_capacity_rows: fx.Int32,
    ):
        tid = fx.Int32(fx.thread_idx.x)
        wave = tid // 64
        group = _scalar(tid // 256)
        scalar_wave = _scalar(wave)
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

        # Separate invariant VGPR offsets from wave-uniform SGPR offsets.
        dma_base = _scalar(fx.Int32(fx.ptrtoint(b_pointer))) + scalar_wave * 1024
        dma_wave_offset = (scalar_wave // 2) * (16 * k) + (scalar_wave % 2) * 1024

        a_pointer = fx.recast_iter(fx.PointerType.get(fx.Int32.ir_type, input_q.memspace, 16), input_q)
        a_view = fx.make_view(a_pointer, fx.make_layout(rows * (k // 4), 1))
        a_buffer = fx.rocdl.make_buffer_tensor(a_view, False)
        a_packet = fxh.BufferTensor(fx.make_view(fx.get_iter(a_buffer), fx.make_layout(4, 1)))
        as_view = fx.make_view(input_scales, fx.make_layout(rows * ks, 1))
        as_buffer = fx.rocdl.make_buffer_tensor(as_view, False)
        as_rsrc = fx.rocdl.get_buffer_rsrc(fx.get_iter(as_buffer))
        output_rows = output_capacity_rows if const_expr(output_layout != "routed") else rows
        out_view = fx.make_view(output, fx.make_layout(output_rows * n, 1))
        out_buffer = fx.rocdl.make_buffer_tensor(out_view, False)
        out_rsrc = fx.rocdl.get_buffer_rsrc(fx.get_iter(out_buffer))
        max_id = _scalar(num_valid_ids[0])
        atom = fx.make_mma_atom(rocdl.cdna4.MFMA_Scale(16, 16, 128, fx.Float8E4M3FN))
        zero = arith._to_raw(fx.Int32(0))
        dma_size = arith._to_raw(fx.Int32(16))
        lds_ptr_type = ir.Type.parse("!llvm.ptr<3>")

        running = fx.Boolean(True)
        while running:
            task_tid = fx.Uint32(_task_thread_id(scalar_wave))
            task = fx.Int32(0)
            if const_expr(persistent):
                if task_tid == 0:
                    counter_ptr = llvm.inttoptr(ir.Type.parse("!llvm.ptr<1>"), arith._to_raw(fx.ptrtoint(task_counter)))
                    next_task = fx.Int32(llvm.AtomicRMWOp(
                        llvm.AtomicBinOp.add, counter_ptr, arith._to_raw(fx.Int32(1)),
                        llvm.AtomicOrdering.monotonic, syncscope="agent",
                    ).res)
                    task_lds[0] = next_task
                fx.barrier()
                task = _scalar(task_lds[0])
            else:
                task = fx.Int32(fx.block_idx.x)
                if const_expr(xcd_swizzle):
                    # gfx950/MI350X: eight XCDs. Transpose only the divisible
                    # active prefix; remainder and overlaunch remain identity.
                    active_tasks = (max_id // 256) * num_oc_splits
                    chunk = active_tasks // 8
                    mapped = (task % 8) * chunk + task // 8
                    task = fx.Int32(arith.select(task < chunk * 8, mapped, task))
            blk_m = task // num_oc_splits
            blk_oc = task % num_oc_splits
            running = blk_m * 256 < max_id
            if running:
                # These coordinates are only used by this task.  The opaque
                # identity prevents loop-invariant address hoisting that kept
                # routing/LDS addresses live through the whole persistent GEMM.
                task_lane_row = task_tid % 16
                task_lane_k = (task_tid % 64) // 16
                task_wave = task_tid // 64
                dma_offset = _pin_address(task_tid * 16 if const_expr(full_k_narrow)
                                          else task_lane_row * 16 + task_lane_k * 256)
                row_begin = _scalar(sorted_expert_ids[2 * blk_m]) if const_expr(task_table) else blk_m * 256
                expert = _scalar(sorted_expert_ids[2 * blk_m + 1]) if const_expr(task_table) else _scalar(sorted_expert_ids[blk_m])

                def gather_row_scales(encoded, row_index):
                    token = encoded & 0xFFFFFF
                    slot = encoded >> 24
                    valid = (token < fx.Uint32(num_tokens)) & (slot < topk)
                    input_row = fx.Int32(token) * topk + fx.Int32(slot)
                    for kb in range_constexpr(ks):
                        scale_offset = arith.select(valid, (kb * rows + input_row) * 4, fx.Int32(-1))
                        row_scale = fx.Float32(rocdl.raw_ptr_buffer_load(
                            fx.Float32.ir_type, as_rsrc, arith._to_raw(scale_offset), zero,
                        ))
                        a_scales_lds[kb * 256 + row_index] = row_scale

                if task_tid < 256:
                    encoded = sorted_ids[row_begin + task_tid].bitcast(fx.Uint32)
                    ids_lds[task_tid] = encoded.bitcast(fx.Int32)
                    routes_lds[task_tid] = sorted_weights[row_begin + task_tid]
                    if const_expr(fused_metadata):
                        # The ID producer can gather its own A scales directly;
                        # all shared metadata is published by one barrier below.
                        gather_row_scales(encoded, task_tid)
                for copy_round in range_constexpr((scale_count + 511) // 512):
                    index = task_tid + copy_round * 512
                    if index < scale_count:
                        scales_lds[index] = weight_scales[
                            expert * (((n + 127) // 128) * ks) + (blk_oc * n_split // 128) * ks + index
                        ]
                if const_expr(not fused_metadata):
                    fx.barrier()
                    if task_tid < 256:
                        gather_row_scales(ids_lds[task_tid].bitcast(fx.Uint32), task_tid)
                fx.barrier()

                weight_view = fx.make_view(
                    weight_shuffled + fx.Int64(expert) * (n * k) + fx.Int64(blk_oc) * (n_split * k),
                    fx.make_layout(n_split * k, 1),
                )
                weight_buffer = fx.rocdl.make_buffer_tensor(weight_view, False)
                weight_rsrc = fx.rocdl.get_buffer_rsrc(fx.get_iter(weight_buffer))

                def dma_b(n_tile, step, voffset):
                    # step is constexpr even for the dynamic N loop.  Only
                    # scalar tile/slot arithmetic is emitted here.
                    target_n = n_tile + step // steps
                    target_step = step % steps
                    kb = (target_step // 2) % ks if const_expr(split_memory) else 0 if const_expr(full_k_narrow) else target_step % ks
                    half = (target_step // 4) * 2 + target_step % 2 if const_expr(split_memory) else target_step if const_expr(full_k_narrow) else target_step // ks
                    slot = (target_n * steps + target_step) % ring_slots
                    soffset = fx.Int32(target_n * (block_n * k) + half * (packet_n * k) + kb * 2048)
                    for copy_round in range_constexpr(dma_rounds):
                        address = dma_base + slot * slot_bytes + copy_round * (512 * 16)
                        dst = llvm.inttoptr(lds_ptr_type, arith._to_raw(address))
                        rocdl.raw_ptr_buffer_load_async_lds(
                            weight_rsrc, dst, dma_size, arith._to_raw(voffset),
                            arith._to_raw(soffset + copy_round * 8192 if const_expr(full_k_narrow)
                                          else soffset + dma_wave_offset + copy_round * (64 * k)), zero,
                            aux=ir.IntegerAttr.get(fx.Int32.ir_type, weight_cache_policy),
                        )
                    rocdl.asyncmark()

                if const_expr(early_b_prefetch):
                    # Metadata is published; overlap B with the independent A
                    # gather/address work. The final full wait also retires A.
                    for step in range_constexpr(min(prefetch, steps * nt)):
                        dma_b(0, step, dma_offset)
                    rocdl.sched_barrier(0)
                a = fx.make_rmem_tensor([8, 2, ks], fx.Int32)
                a_words = fx.make_view(fx.get_iter(a), fx.make_ordered_layout([4, 2, 2, ks], 0))
                stage_a_scales = fx.make_rmem_tensor([2, ks] if full_k_narrow else 2, fx.Float32)
                stage_routes = fx.make_rmem_tensor(2, fx.Float32)
                c = fx.make_rmem_tensor([4, 2, block_n // 16], fx.Float32)
                out_addresses = []
                swap_col = (task_lane_k & 1) * 2 + (task_lane_k >> 1)
                cached_ids = fx.make_rmem_tensor([2, 2], fx.Int32)
                if const_expr(batch_routing_ids):
                    id_pairs = fxh.LdsTensor(fx.make_view(fx.get_iter(ids_lds), fx.make_layout(2, 1)))
                    for mi in range_constexpr(2):
                        row_pair = task_wave * 32 + mi * 16 + (task_lane_row // 2) * 2
                        address = fx.Int32(fx.ptrtoint(fx.get_iter(ids_lds))) + row_pair * 4
                        cached_ids[None, mi].store(id_pairs.load(address_bytes=address))
                    rocdl.sched_barrier(0)
                    rocdl.s_waitcnt(lgkmcnt=0)
                    for mi in range_constexpr(2):
                        cached_ids[None, mi].store(_pin_packet(cached_ids[None, mi].load()))

                def input_id(mi):
                    if const_expr(batch_routing_ids):
                        return (task_lane_row % 2 == 0).select(cached_ids[0, mi], cached_ids[1, mi]).bitcast(fx.Uint32)
                    return ids_lds[task_wave * 32 + mi * 16 + task_lane_row].bitcast(fx.Uint32)

                for mi in range_constexpr(2):
                    row = task_wave * 32 + mi * 16 + task_lane_row
                    encoded = input_id(mi)
                    token = encoded & 0xFFFFFF
                    slot = encoded >> 24
                    valid = (token < fx.Uint32(num_tokens)) & (slot < topk)
                    output_row = fx.Int32(token) * topk + fx.Int32(slot)
                    # An invalid row starts at the descriptor's end, without
                    # wrapping when a positive N soffset is added later.
                    out_addresses.append(_pin_address(arith.select(
                        valid, output_row * (n * 2) + blk_oc * (n_split * 2) + swap_col * 16,
                        rows * (n * 2),
                    )))
                    for kb in range_constexpr(ks):
                        for part in range_constexpr(2):
                            offset = arith.select(
                                valid, output_row * k + task_lane_k * 16 + kb * 128 + part * 64,
                                fx.Int32(-1),
                            )
                            a_words[None, part, mi, kb].store(a_packet.load(voffset_bytes=offset))

                coalesced_addresses = []
                if const_expr(coalesced_output):
                    for mi in range_constexpr(2):
                        addresses = []
                        for parity in range_constexpr(2):
                            row = task_wave * 32 + mi * 16 + (task_lane_row // 2) * 2 + parity
                            encoded = (cached_ids[parity, mi].bitcast(fx.Uint32) if const_expr(batch_routing_ids)
                                       else ids_lds[row].bitcast(fx.Uint32))
                            token, slot = encoded & 0xFFFFFF, encoded >> 24
                            valid = (token < fx.Uint32(num_tokens)) & (slot < topk)
                            output_row = row_begin + row if const_expr(output_layout != "routed") else fx.Int32(token) * topk + fx.Int32(slot)
                            # Full task-table rows are M256 aligned; retain the
                            # original per-lane address arithmetic for baseline.
                            packed_block = row_begin // 256 if const_expr(task_table) else blk_m
                            offset = (packed_block * (256 * n * 2) + blk_oc * (256 * n_split * 2) + row * 128) if const_expr(output_layout == "packed") else output_row * (n * 2) + blk_oc * (n_split * 2)
                            offset += swap_col * 16 + (task_lane_row % 2) * 64
                            addresses.append(_pin_address(arith.select(valid, offset, output_rows * (n * 2))))
                        coalesced_addresses.append(addresses)

                # All VGPR addresses are invariant throughout this task.
                # Unroll a full ring period to make every slot constexpr.
                row = task_wave * 32 + task_lane_row
                b_lane = task_lane_row * 16 + task_lane_k * 256
                b_addresses = [
                    _pin_address(fx.Int32(fx.ptrtoint(b_pointer)) + b_lane + slot * slot_bytes)
                    for slot in range_constexpr(ring_slots)
                ]
                row_addresses = [
                    _pin_address(fx.Int32(fx.ptrtoint(row_scale_pointer)) + (kb * 256 + row) * 4)
                    for kb in range_constexpr(ks + 1)
                ]
                scalar_scale_base = _scalar(fx.Int32(fx.ptrtoint(scale_pointer)))

                def read_scale(index):
                    # ds_read_addtid uses M0 + lane*4, so dynamic N indexing
                    # stays entirely in SALU.  The explicit Memory wait below
                    # retires this opaque LDS read before its value is used.
                    address = scalar_scale_base + fx.Int32(index) * 4
                    # gfx950 requires two wait states after writing M0.
                    # Inline ASM is opaque to LLVM's M0 hazard recognizer.
                    # Restore M0 so LLVM-managed direct-LDS DMA is unaffected.
                    assembly = (
                        "s_mov_b32 $1, m0\ns_mov_b32 m0, $2\ns_nop 1\n"
                        "ds_read_addtid_b32 $0\ns_mov_b32 m0, $1"
                    )
                    result = llvm.inline_asm(
                        ir.Type.parse("!llvm.struct<(i32, i32)>"), [address.ir_value()], assembly,
                        "=&v,=&s,s,~{memory}", has_side_effects=True,
                    )
                    return fx.Int32(llvm.extractvalue(fx.Int32.ir_type, result, [0]))

                def broadcast_scale(value):
                    # A coefficient broadcast, not an address calculation.
                    return _scalar(value).bitcast(fx.Float32)

                def read_b(n_tile, step, ring_phase):
                    b = fx.make_rmem_tensor([8, packet_n // 16, ks] if full_k_narrow else [8, packet_n // 16], fx.Int32)
                    words = fx.make_view(fx.get_iter(b), fx.make_ordered_layout(
                        [4, 2, packet_n // 16, ks] if full_k_narrow else [4, 2, packet_n // 16], 0,
                    ))
                    if const_expr(full_k_narrow):
                        scale = fx.make_rmem_tensor(ks, fx.Int32)
                        for kb in range_constexpr(ks):
                            for ni in range_constexpr(half_n // 16):
                                for part in range_constexpr(2):
                                    words[None, part, ni, kb].store(b_lds.load(
                                        address_bytes=b_addresses[(ring_phase * steps + step) % ring_slots],
                                        offset_bytes=ni * (16 * k) + kb * 2048 + part * 1024,
                                    ))
                            # Two adjacent BN64 tiles share each N128 scale.
                            # Odd-sized splits can start at column64 of a block.
                            scale_n = ((blk_oc * n_split) % 128 + n_tile * block_n) // 128
                            scale[kb] = read_scale(scale_n * ks + kb)
                            for mi in range_constexpr(2):
                                stage_a_scales[mi, kb] = row_scale_lds.load(
                                    address_bytes=row_addresses[kb], offset_bytes=mi * 16 * 4,
                                )[0]
                    else:
                        for ni in range_constexpr(packet_n // 16):
                            for part in range_constexpr(2):
                                words[None, part, ni].store(b_lds.load(
                                    address_bytes=b_addresses[(ring_phase * steps + step) % ring_slots],
                                    offset_bytes=ni * 2048 + part * 1024,
                                ))
                        kb = (step // 2) % ks if const_expr(split_memory) else step % ks
                        half = (step // 4) * 2 + step % 2 if const_expr(split_memory) else step // ks
                        scale = read_scale(n_tile * (block_n // 128) * ks + (half * packet_n // 128) * ks + kb)
                        for mi in range_constexpr(2):
                            stage_a_scales[mi] = row_scale_lds.load(
                                address_bytes=row_addresses[kb], offset_bytes=mi * 16 * 4,
                            )[0]
                    for mi in range_constexpr(2):
                        stage_routes[mi] = row_scale_lds.load(
                            address_bytes=row_addresses[ks], offset_bytes=mi * 16 * 4,
                        )[0]
                    return b, scale

                def place_b(b):
                    if const_expr(block_n == 256 or full_k_narrow):
                        # Hardware copy operands are four registers wide.
                        # Constrain them AFTER the batch's one explicit wait;
                        # a 64-register inline-ASM tuple is not allocatable.
                        # Flatten the constexpr K/N fragment dimensions; each
                        # packet is still exactly one native ds_read_b128.
                        words = fx.make_view(fx.get_iter(b), fx.make_ordered_layout(
                            [4, 2 * (packet_n // 16) * (ks if full_k_narrow else 1)], 0,
                        ))
                        for packet in range_constexpr(2 * (packet_n // 16) * (ks if full_k_narrow else 1)):
                            words[None, packet].store(_pin_packet(words[None, packet].load()))

                def pack_record(record):
                    packed = []
                    for mi in range_constexpr(2):
                        row = []
                        for pair in range_constexpr(record_pairs):
                            ni = record * record_atoms + pair * 2
                            row.append(_pack_pair(c[None, mi, ni].load(), c[None, mi, ni + 1].load(), stage_routes[mi]))
                        if const_expr(coalesced_output and block_n == 256):
                            row = _coalesce_output_pairs(row[0], row[1])
                        packed.append(row)
                    return packed

                def pin_accumulators(step):
                    # Preserve whole four-FP32 vectors across Memory.  Without
                    # this, register coalescing can relocate individual C
                    # elements into a just-retired store tuple during Memory.
                    if const_expr(block_n == 256):
                        half = step // ks
                        first_record = (step // 4) * 2 + step % 2 if const_expr(split_memory) else 2 * half
                        for record in range_constexpr(first_record, first_record + (1 if split_memory else 2)):
                            for mi in range_constexpr(2):
                                for ni in range_constexpr(record_atoms):
                                    frag = c[None, mi, record * record_atoms + ni]
                                    frag.store(_pin_packet(frag.load()))

                def store_quarter(n_tile, packed, quarter, address, local_record_begin=0, local_record_end=2):
                    mi, half = quarter % 2, quarter // 2
                    for local_record in range_constexpr(local_record_begin, local_record_end):
                        record = half * 2 + local_record
                        for pair in range_constexpr(record_pairs):
                            store_record = half * 2 if const_expr(coalesced_output and block_n == 128) else record
                            soffset = fx.Int32((n_tile * block_n + store_record * record_n + (0 if coalesced_output else pair * 32)) * 2)
                            if const_expr(output_layout == "packed"):
                                soffset = fx.Int32((n_tile * 2 + half) * 32768)
                            store_row_index = local_record if const_expr(coalesced_output and block_n == 128) else pair
                            target = coalesced_addresses[mi][store_row_index] if const_expr(coalesced_output) else address
                            rocdl.raw_ptr_buffer_store(
                                arith._to_raw(packed[record][mi][pair]), out_rsrc,
                                arith._to_raw(target), arith._to_raw(soffset),
                                aux=ir.IntegerAttr.get(fx.Int32.ir_type, output_cache_policy),
                            )

                def mma_record(b, scale, kb, record):
                    # Retire a partial after three intervening independent MFMAs.
                    # Keep native IR arithmetic so LLVM tracks MFMA/AGPR
                    # hazards; llvm.passthrough disables packed-FP32 combining.
                    scale = broadcast_scale(scale)
                    factors = [stage_a_scales[mi] * scale
                               for mi in range_constexpr(2)]
                    pending = []
                    for mi in range_constexpr(2):
                        for ni in range_constexpr(record_atoms):
                            partial = fx.make_rmem_tensor(4, fx.Float32)
                            partial.fill(0)
                            fx.gemm(atom, partial, b[None, (record % 2) * record_atoms + ni], a[None, mi, kb], partial)
                            pending.append((partial, mi, ni))
                            rocdl.sched_barrier(0)
                            if const_expr(len(pending) > 3):
                                old, old_mi, old_ni = pending.pop(0)
                                dest = c[None, old_mi, record * record_atoms + old_ni]
                                if const_expr(kb == 0):
                                    dest.store(old.load() * factors[old_mi])
                                else:
                                    dest.store(fxh.eltwise_op("llvm.fma.f32", old.load(), factors[old_mi], dest.load()))
                            rocdl.sched_barrier(0)
                    for old, old_mi, old_ni in pending:
                        dest = c[None, old_mi, record * record_atoms + old_ni]
                        if const_expr(kb == 0):
                            dest.store(old.load() * factors[old_mi])
                        else:
                            dest.store(fxh.eltwise_op("llvm.fma.f32", old.load(), factors[old_mi], dest.load()))

                def compute_stage(b, scale, step, has_pack):
                    """Eight or sixteen MFMA slots with independent VALUs.

                    Pack an already completed N64 record, not the current one.
                    BN128: previous high half in step0, current low in step1.
                    BN64 uses the same full-K schedule at half the N width.
                    BN256: old C2/C3 in steps0/1, current C0/C1 in steps2/3.
                    Sixteen-stage mode packs one row of that record at a time.
                    Each pair needs 8 route muls + 4 cvts + 2 swaps.  All
                    operands/output stay in VGPRs; no artificial register
                    copies are added to fill the seven-VALU issue budget.
                    """
                    half = (step // 4) * 2 + step % 2 if const_expr(split_memory) else step if const_expr(full_k_narrow) else step // 2
                    pack_id = (step // 2 + 2) % 4 if const_expr(split_memory) else 1 - step if const_expr(full_k_narrow) else (step + 2) % 4
                    coefficients = [broadcast_scale(scale[kb]) for kb in range_constexpr(ks)] if const_expr(full_k_narrow) else []
                    coefficient = broadcast_scale(scale) if const_expr(not full_k_narrow) else fx.Float32(0)
                    factors = [[stage_a_scales[mi, kb] * coefficients[kb]
                                for mi in range_constexpr(1 if small_n64 and kb == 1 else 2)]
                               for kb in range_constexpr(ks)] if const_expr(full_k_narrow) else [
                                   stage_a_scales[mi] * coefficient
                                   for mi in range_constexpr(1 if split_memory else 2)
                               ]
                    pack_scaled = [[] for _ in range_constexpr(4)]
                    pack_words = [[] for _ in range_constexpr(4)]
                    pack_swaps = [[] for _ in range_constexpr(4)]
                    packed = [[], []]
                    pack_index = 0
                    pending = []
                    dequant = []

                    def pack_op(index):
                        pair_index, op = index // pack_ops_per_pair, index % pack_ops_per_pair
                        mi = step % 2 if const_expr(split_memory) else pair_index // (1 if small_n64 else 2)
                        pair = pair_index % (1 if small_n64 else 2)
                        if const_expr(op < 8):
                            ni = pack_id * (2 if small_n64 else 4) + pair * 2 + op // 4
                            pack_scaled[pair_index].append(c[None, mi, ni].load()[op % 4] * stage_routes[mi])
                        elif const_expr(op < 12):
                            wi = op - 8
                            values = fx.Vector.from_elements(
                                [pack_scaled[pair_index][wi * 2], pack_scaled[pair_index][wi * 2 + 1]],
                                fx.Float32,
                            )
                            pack_words[pair_index].append(values.to(fx.BFloat16).bitcast(fx.Int32)[0])
                        else:
                            wi = op - 12
                            words = pack_words[pair_index]
                            pack_swaps[pair_index].append(rocdl.permlane16_swap(
                                ir.Type.parse("!llvm.struct<(i32, i32)>"),
                                words[wi].ir_value(), words[wi + 2].ir_value(), False, False,
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
                        old, old_mi, old_ni, kb, element = item
                        value = old.load()[element]
                        factor = factors[kb][old_mi] if const_expr(full_k_narrow) else factors[old_mi]
                        if const_expr(kb == 0):
                            c[element, old_mi, old_ni] = value * factor
                        else:
                            c[element, old_mi, old_ni] = fx.Float32(fxh.eltwise_op(
                                "llvm.fma.f32", value, factor, c[element, old_mi, old_ni],
                            ))

                    rocdl.sched_barrier(0)
                    for index in range_constexpr(mfma_count):
                        atoms_per_row = 2 if const_expr(small_n64) else 4
                        packet, local = index // (atoms_per_row * 2), index % (atoms_per_row * 2)
                        mi, ni = local // atoms_per_row, local % atoms_per_row
                        kb = (step // 2) % 2 if const_expr(split_memory) else packet if const_expr(full_k_narrow) else step % 2
                        c_ni = half * atoms_per_row + ni if const_expr(full_k_narrow or split_memory) else half * 8 + packet * 4 + ni
                        partial = fx.make_rmem_tensor(4, fx.Float32)
                        partial.fill(0)
                        weight = b[None, ni, kb] if const_expr(full_k_narrow) else b[None, packet * 4 + ni]
                        fx.gemm(atom, partial, weight, a[None, mi, kb], partial)
                        rocdl.sched_barrier(0)
                        remaining = 7
                        if const_expr(small_n64 and index == 6):
                            # The K1/row1 factor is only needed during drain;
                            # schedule its useful multiply in the seventh gap.
                            factors[1].append(stage_a_scales[1, 1] * coefficients[1])
                            remaining -= 1
                        elif const_expr(split_memory and index == 6):
                            # Row1's factor is first consumed here.  Its real
                            # multiply fills the last pre-drain seven-VALU slot.
                            factors.append(stage_a_scales[1] * coefficient)
                            remaining -= 1
                        pending.append((partial, mi, c_ni, kb))
                        if const_expr(len(pending) > 2):
                            old, old_mi, old_ni, old_kb = pending.pop(0)
                            dequant.extend([(old, old_mi, old_ni, old_kb, element) for element in range_constexpr(4)])
                        for _ in range_constexpr(min(4, remaining, len(dequant))):
                            retire_one(dequant.pop(0))
                            remaining -= 1
                        while const_expr(has_pack and pack_index < pack_ops and remaining > 0):
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
                    while const_expr(has_pack and pack_index < pack_ops):
                        pack_op(pack_index)
                        pack_index += 1
                    if const_expr(coalesced_output and has_pack):
                        for mi in range_constexpr(2):
                            packed[mi] = _coalesce_output_pairs(packed[mi][0], packed[mi][1])
                    return packed

                def run_tile(n_tile, previous, first=False, last=False, ring_phase=0):
                    packed = []
                    tiles_left = nt - n_tile if const_expr(isinstance(n_tile, int)) else 3
                    ring_phase = n_tile if const_expr(isinstance(n_tile, int)) else ring_phase
                    for step in range_constexpr(steps):
                        kb, half = step % ks, step // ks
                        if const_expr(stage_priority):
                            _priority(0)
                        _mark(f"MOE8_MEMORY_BEGIN_{step}")
                        if const_expr(lds_read_first):
                            b, scale = read_b(n_tile, step, ring_phase)
                            rocdl.sched_barrier(0)
                        if const_expr(split_memory and not first):
                            quarter, local_record = step // 2, step % 2
                            store_quarter(n_tile - 1, previous, quarter, out_addresses[quarter % 2],
                                          local_record, local_record + 1)
                            rocdl.sched_barrier(0)
                        elif const_expr(full_k_narrow and not first):
                            # Each Memory writes one N half of the previous
                            # tile; BN64 uses one N32 record, BN128 uses two.
                            for mi in range_constexpr(2):
                                if const_expr(small_n64):
                                    store_quarter(n_tile - 1, previous, mi, out_addresses[mi], step, step + 1)
                                else:
                                    store_quarter(n_tile - 1, previous, step * 2 + mi, out_addresses[mi])
                            rocdl.sched_barrier(0)
                        elif const_expr(not first and step < 4):
                            store_quarter(n_tile - 1, previous, step, out_addresses[step % 2])
                            rocdl.sched_barrier(0)
                        if const_expr(not lds_read_first):
                            b, scale = read_b(n_tile, step, ring_phase)
                        if const_expr(step + prefetch < steps * tiles_left if full_k_narrow else not last or step + prefetch < steps):
                            dma_b(n_tile, step + prefetch, dma_offset)
                        if const_expr(step + 1 < steps * tiles_left if full_k_narrow else not last or step + 1 < steps):
                            # Wait for the next consumer.  Each narrow steady
                            # group is 2 stores + 1 DMA: BN64 d=1/2/3 gives
                            # vmcnt0/3/6; BN256 sixteen-stage uses vmcnt6/9.
                            # Startup and tail count only real requests.
                            rocdl.wait_asyncmark(min(prefetch - 1, steps * tiles_left - step - 2) if full_k_narrow
                                                else min(prefetch - 1, steps - step - 2) if last else prefetch - 1)
                        rocdl.s_waitcnt(lgkmcnt=0)
                        rocdl.sched_barrier(0)
                        place_b(b)
                        _mark(f"MOE8_MEMORY_END_{step}")
                        _stage_end()

                        if const_expr(stage_priority):
                            _priority(3)
                        _mark(f"MOE8_COMPUTE_BEGIN_{step}")
                        if const_expr(split_memory):
                            record = compute_stage(b, scale, step, not first or step >= 4)
                            if const_expr(not first or step >= 4):
                                if const_expr(step % 2 == 0):
                                    pack_rows = [record[0]]
                                else:
                                    pack_rows.append(record[1])
                                    if const_expr(step >= 4):
                                        packed.append(pack_rows)
                                    else:
                                        previous.append(pack_rows)
                        elif const_expr(full_k_narrow):
                            record = compute_stage(b, scale, step, not first or step == 1)
                            if const_expr(not first or step == 1):
                                # Split each N half into N32 output records.
                                parts = [[[record[mi][pair]] for mi in range_constexpr(2)]
                                         for pair in range_constexpr(1 if small_n64 else 2)]
                                if const_expr(step == 1):
                                    packed.extend(parts)
                                else:
                                    previous.extend(parts)
                        elif const_expr(block_n == 256):
                            record = compute_stage(b, scale, step, not first or step >= 2)
                            if const_expr(step >= 2):
                                packed.append(record)
                            elif const_expr(not first):
                                previous.append(record)
                        else:
                            for packet in range_constexpr(2):
                                record = half * 2 + packet
                                mma_record(b, scale, kb, record)
                                if const_expr(not first and step == 0 and packet == 0):
                                    previous.append(pack_record(3))
                                if const_expr(step == ks - 1 and packet == 1):
                                    packed.append(pack_record(0))
                                if const_expr(step == ks and packet == 0):
                                    packed.append(pack_record(1))
                                if const_expr(step == steps - 1 and packet == 1):
                                    packed.append(pack_record(2))
                        pin_accumulators(step)
                        _mark(f"MOE8_COMPUTE_END_{step}")
                        if const_expr(stage_priority):
                            _priority(0)
                        _stage_end()
                    return packed

                def save_state(packed):
                    state = [c[None, mi, ni].load() for mi in range_constexpr(2)
                             for ni in range_constexpr(first_unpacked * record_atoms, record_count * record_atoms)]
                    state.extend([packed[record][mi][pair] for record in range_constexpr(first_unpacked)
                                  for mi in range_constexpr(2) for pair in range_constexpr(record_pairs)])
                    return state

                def restore_state(state):
                    index = 0
                    for mi in range_constexpr(2):
                        for ni in range_constexpr(first_unpacked * record_atoms, record_count * record_atoms):
                            c[None, mi, ni].store(state[index])
                            index += 1
                    packed = []
                    for record in range_constexpr(first_unpacked):
                        record_data = []
                        for mi in range_constexpr(2):
                            row = []
                            for pair in range_constexpr(record_pairs):
                                row.append(fx.Vector(state[index]))
                                index += 1
                            record_data.append(row)
                        packed.append(record_data)
                    return packed

                # Preparation -> prologue -> stagger -> N0/N1 -> 1N loop -> tail.
                if const_expr(overlap_prologue):
                    # A loads precede B0's marker. Waiting for B0 below also
                    # retires A, without serializing A completion before B issue.
                    rocdl.sched_barrier(0)
                    rocdl.s_waitcnt(lgkmcnt=0)
                else:
                    rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0)
                if const_expr(not early_b_prefetch):
                    for step in range_constexpr(min(prefetch, steps * nt)):
                        dma_b(0, step, dma_offset)
                rocdl.wait_asyncmark(min(prefetch, steps * nt) - 1)
                _stage_end()
                if group == 1:
                    _stage_end()
                packed = run_tile(0, [], first=True, last=nt == 1)
                if const_expr(nt > 1):
                    packed = run_tile(1, packed, last=nt == 2)
                if const_expr(nt >= 3):
                    initial = save_state(packed)
                    for n_tile, state in range(fx.Int32(2), fx.Int32(loop_end), fx.Int32(tile_stride), init=initial):
                        previous = restore_state(state)
                        if const_expr(small_n64):
                            _mark("MOE8_STEADY_BEGIN")
                        for phase in range_constexpr(tile_stride):
                            packed = run_tile(fx.Int32(n_tile) + phase, previous, ring_phase=2 + phase)
                            previous = packed
                        if const_expr(small_n64):
                            _mark("MOE8_STEADY_END")
                        results = yield save_state(packed)
                    previous = restore_state(results)
                    # Static tail retains only DMA requests with consumers.
                    for n_tile in range_constexpr(loop_end, nt):
                        packed = run_tile(n_tile, previous, last=n_tile == nt - 1)
                        previous = packed

                # No speculative DMA survives the last consumer/task boundary.
                _priority(3)
                for record in range_constexpr(first_unpacked, record_count):
                    packed.append(pack_record(record))
                if const_expr(coalesced_output and block_n == 128):
                    rows_coalesced = [_coalesce_output_pairs(packed[2][mi][0], packed[3][mi][0])
                                      for mi in range_constexpr(2)]
                    packed[2] = [[rows_coalesced[mi][0]] for mi in range_constexpr(2)]
                    packed[3] = [[rows_coalesced[mi][1]] for mi in range_constexpr(2)]
                _priority(0)
                for quarter in range_constexpr(2 if small_n64 else 4):
                    store_quarter(nt - 1, packed, quarter, out_addresses[quarter % 2])
                rocdl.wait_asyncmark(0)
                rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0)
                _stage_end()
                if group == 0:
                    _stage_end()
            fx.barrier()
            if const_expr(not persistent):
                running = fx.Boolean(False)

    @flyc.jit
    def launch(
        output: fx.Pointer, input_q: fx.Pointer, weight_shuffled: fx.Pointer,
        input_scales: fx.Pointer, weight_scales: fx.Pointer,
        sorted_ids: fx.Pointer, sorted_weights: fx.Pointer, sorted_expert_ids: fx.Pointer,
        num_valid_ids: fx.Pointer, task_counter: fx.Pointer,
        num_tokens: fx.Int32, num_expert_blocks: fx.Int32, output_capacity_rows: fx.Int32, stream: fx.Stream,
    ):
        moe_down_8stage_kernel(
            output, input_q, weight_shuffled, input_scales, weight_scales,
            sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, task_counter,
            num_tokens, num_expert_blocks, output_capacity_rows,
            value_attrs=kernel_attrs,
        ).launch(grid=(persistent_workgroups if const_expr(persistent) else num_expert_blocks * num_oc_splits, 1, 1),
                 block=(512, 1, 1), stream=stream)

    def callable(
        output: torch.Tensor,
        input_q: torch.Tensor,
        weight_shuffled: torch.Tensor,
        input_scales: torch.Tensor,
        weight_scales: torch.Tensor,
        sorted_ids: torch.Tensor,
        sorted_weights: torch.Tensor,
        sorted_expert_ids: torch.Tensor,
        num_valid_ids: torch.Tensor,
        task_counter: torch.Tensor,
    ):
        num_tokens, input_topk, input_k = input_q.shape
        assert input_topk == topk and input_k == k
        assert weight_shuffled.shape == (num_experts, n, k)
        if task_table:
            assert sorted_expert_ids.ndim == 2 and sorted_expert_ids.shape[1] == 2
        else:
            assert sorted_expert_ids.ndim == 1
        task_capacity = sorted_expert_ids.shape[0]
        if output_layout == "routed":
            assert output.shape == (num_tokens, topk, n)
        else:
            assert output.ndim == 2 and output.shape[1] == n and output.shape[0] % 256 == 0
            assert output.shape[0] >= (sorted_ids.numel() + 255) // 256 * 256
        output_capacity_rows = output.numel() // n
        assert input_scales.numel() == num_tokens * topk * ks
        assert weight_scales.shape == (num_experts, (n + 127) // 128, ks)
        assert sorted_weights.shape == sorted_ids.shape
        assert 0 < num_tokens < (1 << 24)
        assert input_q.dtype == weight_shuffled.dtype == torch.float8_e4m3fn
        assert output.dtype == torch.bfloat16
        assert input_scales.dtype == weight_scales.dtype == sorted_weights.dtype == torch.float32
        assert sorted_ids.dtype == sorted_expert_ids.dtype == num_valid_ids.dtype == task_counter.dtype == torch.int32
        assert num_valid_ids.numel() >= 1 and task_counter.numel() == 1
        tensors = (output, input_q, weight_shuffled, input_scales, weight_scales,
                   sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, task_counter)
        assert all(t.is_cuda and t.device == output.device and t.is_contiguous() for t in tensors)
        assert input_q.numel() < (1 << 32)
        assert input_scales.numel() * 4 < (1 << 32) and n_split * k < (1 << 32)
        assert (output.numel() + n) * 2 < (1 << 32), "32-bit buffer offsets including the padding sentinel"
        assert torch.cuda.get_device_properties(output.device).gcnArchName.startswith("gfx950")
        if persistent:
            task_counter.zero_()
        stream = torch.cuda.current_stream(output.device)
        compiled = getattr(launch, "_cf", None)
        if compiled is None:
            _run_compiled(
                launch, *[_ptr(tensor) for tensor in tensors],
                fx.Int32(num_tokens), fx.Int32(task_capacity), fx.Int32(output_capacity_rows), fx.Stream(stream.cuda_stream),
            )
        else:
            # The compiled ABI accepts bare pointer/size/stream integers.
            # Read fresh addresses every call; no stale tensor or stream cache.
            compiled(
                *(tensor.data_ptr() for tensor in tensors), num_tokens,
                task_capacity, output_capacity_rows, stream.cuda_stream,
            )
        return output

    callable.config = {
        "block_m": block_m, "block_n": block_n, "num_oc_splits": num_oc_splits,
        "stages": 2 * steps, "prefetch_distance": prefetch,
        "output_cache_policy": output_cache_policy, "weight_cache_policy": weight_cache_policy,
        "persistent_workgroups": persistent_workgroups,
        "lds_read_first": lds_read_first, "coalesced_output": coalesced_output,
        "stage_priority": stage_priority,
        "output_layout": output_layout,
        "persistent": persistent, "xcd_swizzle": xcd_swizzle, "xcd_count": 8,
        "task_table": task_table,
        "overlap_prologue": overlap_prologue, "fused_metadata": fused_metadata,
        "batch_routing_ids": batch_routing_ids,
        "early_b_prefetch": early_b_prefetch,
    }
    return callable