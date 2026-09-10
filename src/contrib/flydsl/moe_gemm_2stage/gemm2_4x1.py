# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MoE stage2 4x1 down-projection kernel builder."""

import os

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.typing import as_ir_value
from flydsl.expr.utils.arith import _to_raw as _raw

from . import layout_helpers as fxh
from .common import get_down_device_config


def _build_moe_gemm2_4x1(
    N,
    K,
    weight_dtype,
    weight_quant_type,
    TOPK,
    BLOCK_TILE_SIZE_M,
    BLOCK_TILE_SIZE_N,
    stage="down",
    alg="splitk",
    E=None,
    USE_ATOMIC_WRITE=True,
    act_quant_type=None,
    tile_k=None,
    activation="silu",
    swiglu_limit=None,
    down_path="default",
    down_output_padding_bytes=None,
    METADATA_TILE_SIZE_M=None,
):
    del E, activation, swiglu_limit
    assert stage == "down"
    assert alg == "prefill_1x4"
    assert down_path == "4x1"
    assert weight_dtype == "fp8"
    assert weight_quant_type in ("ptpc", "per_tensor")
    if act_quant_type is None:
        act_quant_type = weight_quant_type
    assert (weight_quant_type == "ptpc" and act_quant_type == "ptpc") or (
        weight_quant_type == "per_tensor" and act_quant_type in ("ptpc", "per_tensor")
    )
    assert not USE_ATOMIC_WRITE
    assert BLOCK_TILE_SIZE_M in (128, 256)
    assert BLOCK_TILE_SIZE_N == 64
    if METADATA_TILE_SIZE_M is None:
        METADATA_TILE_SIZE_M = BLOCK_TILE_SIZE_M
    assert METADATA_TILE_SIZE_M == BLOCK_TILE_SIZE_M
    if tile_k is None and os.environ.get("MOE_PREFILL_TILE_K"):
        tile_k = int(os.environ["MOE_PREFILL_TILE_K"])
    if tile_k is None:
        tile_k = 192 if K % 192 == 0 and K % 128 != 0 else 128
    assert tile_k in (128, 192)
    assert K % tile_k == 0
    assert N % BLOCK_TILE_SIZE_N == 0
    assert down_output_padding_bytes in (0, 32, 64, 128)

    BLOCK_M = BLOCK_TILE_SIZE_M
    BLOCK_N = BLOCK_TILE_SIZE_N
    BLOCK_K = tile_k
    WAVE_M = BLOCK_M // 4
    K_STAGES = K // BLOCK_K
    DEDICATED_K256_CSHUFFLE = (
        N == 2048
        and K == 256
        and BLOCK_M == 256
        and BLOCK_K == 128
        and weight_quant_type == "ptpc"
    )
    PIPELINE_K256_WEIGHT_SCALE = DEDICATED_K256_CSHUFFLE
    PIPELINE_K256_CSHUFFLE_PACK = DEDICATED_K256_CSHUFFLE
    USE_SINGLE_WEIGHT_SLOT = N == 2048 and K == 512 and BLOCK_M == 128
    FULL_N_TILES_PER_WG = N // BLOCK_N
    SHORT_N_TILES_PER_WG = 8 if BLOCK_M == 128 else 16
    USE_SHORT_N = K_STAGES > 2 and FULL_N_TILES_PER_WG > SHORT_N_TILES_PER_WG
    FULL_N_MIN_TASKS = 768
    if USE_SHORT_N:
        assert N % (BLOCK_N * SHORT_N_TILES_PER_WG) == 0
    output_row_stride = N + down_output_padding_bytes // 2
    full_down_ops = fxh.FlyObjCache()
    short_down_ops = fxh.FlyObjCache()
    topology_enabled, xcc_count = get_down_device_config()
    se_per_xcc = 4
    cu_per_se = 5
    se_count = xcc_count * se_per_xcc

    def get_down_ops(n_tiles_per_wg):
        if n_tiles_per_wg == SHORT_N_TILES_PER_WG:
            return short_down_ops
        return full_down_ops

    def _encode_waitcnt(vmcnt=63, expcnt=7, lgkmcnt=63):
        vm_lo = vmcnt & 0xF
        vm_hi = (vmcnt >> 4) & 0x3
        return vm_lo | (expcnt << 4) | (lgkmcnt << 8) | (vm_hi << 14)

    def _pack_scaled_bf16_pairs(values, scales):
        fma_bias = as_ir_value(fx.Uint32(0x8000)).bitcast(fx.Float32.ir_type)
        scaled = fxh.eltwise_op("llvm.fma.f32", values, scales, fma_bias)
        selector = fx.Uint32(0x07060302)
        packed = []
        for index in range_constexpr(0, scaled.numel, 2):
            packed.append(
                llvm.inline_asm(
                    ir.IntegerType.get_signless(32),
                    [
                        _raw(scaled[index + 1]),
                        _raw(scaled[index]),
                        _raw(selector),
                    ],
                    "v_perm_b32 $0, $1, $2, $3",
                    "=v,v,v,s",
                    has_side_effects=True,
                )
            )
        return packed

    def _lds_barrier():
        rocdl.sched_barrier(0)
        rocdl.s_barrier()
        rocdl.sched_barrier(0)

    def _enter_read_write_stage():
        rocdl.sched_barrier(0)
        rocdl.s_setprio(0)
        rocdl.sched_barrier(0)

    def _enter_compute_stage():
        rocdl.sched_barrier(0)
        rocdl.s_setprio(3)
        rocdl.sched_barrier(0)

    @flyc.kernel(known_block_size=[256, 1, 1])
    def moe_2stage_down_prefill_4x1(
        p_input: fx.Pointer,
        p_weight: fx.Pointer,
        p_output: fx.Pointer,
        p_sorted_ids: fx.Pointer,
        p_sorted_weights: fx.Pointer,
        p_sorted_expert_ids: fx.Pointer,
        p_num_valid_ids: fx.Pointer,
        p_w_scale: fx.Pointer,
        p_a_scale: fx.Pointer,
        M: fx.Int32,
        n_tiles_per_wg: fx.Constexpr[int],
    ):
        tid = fx.Int32(gpu.thread_idx.x)
        lane_id = tid % 64
        wave_id = tid // 64
        max_valid_id = fxh.view_as_torch_tensor(p_num_valid_ids, (1,), fx.Int32)[0]
        workgroup_idx = fx.Int32(gpu.block_idx.y)
        e_idx = workgroup_idx
        if const_expr(topology_enabled):
            valid_tasks = fxh.div_up(fx.Uint32(max_valid_id), BLOCK_M)
            tasks_per_se = valid_tasks // se_count
            mapped_tasks = tasks_per_se * se_count
            tasks_per_xcc = tasks_per_se * se_per_xcc
            workgroup_idx_u32 = fx.Uint32(workgroup_idx)
            xcc_id = workgroup_idx_u32 & (xcc_count - 1)
            xcc_local_idx = workgroup_idx_u32 >> 2
            se_slot = xcc_local_idx & (se_per_xcc - 1)
            within_se = xcc_local_idx >> 2
            cu_slot = within_se % cu_per_se
            cu_round = within_se // cu_per_se
            short_cu_tasks = tasks_per_se // cu_per_se
            long_cu_count = tasks_per_se % cu_per_se
            cu_prefix_extra = arith.select(
                cu_slot < long_cu_count,
                cu_slot,
                long_cu_count,
            )
            se_local_rank = cu_slot * short_cu_tasks + cu_prefix_extra + cu_round
            logical_xcc = (xcc_id + 2) & (xcc_count - 1)
            mapped_e_idx = logical_xcc * tasks_per_xcc + se_slot * tasks_per_se + se_local_rank
            e_idx = fx.Int32(
                arith.select(
                    workgroup_idx_u32 < mapped_tasks,
                    mapped_e_idx,
                    workgroup_idx_u32,
                )
            )

        if e_idx * BLOCK_M < max_valid_id:
            input_tensor = fxh.view_as_torch_tensor(p_input, (M, TOPK, K), fx.Float8E4M3FNUZ)
            input_tensor = fx.rocdl.make_buffer_tensor(
                input_tensor,
                max_size=False,
                num_records_bytes=fx.Int64(M) * TOPK * K,
            )
            sorted_ids = fxh.view_as_torch_tensor(
                fxh._as_ptr(p_sorted_ids) + fx.Int64(e_idx) * BLOCK_M,
                (BLOCK_M,),
                fx.Int32,
            )
            sorted_ids_buffer = fx.rocdl.make_buffer_tensor(sorted_ids, max_size=False, num_records_bytes=BLOCK_M * 4)
            sorted_weights = fxh.view_as_torch_tensor(
                fxh._as_ptr(p_sorted_weights) + fx.Int64(e_idx) * BLOCK_M,
                (BLOCK_M,),
                fx.Float32,
            )
            expert_id = fxh.view_as_torch_tensor(p_sorted_expert_ids, (1,), fx.Int32)[e_idx]

            shared_allocator = fx.SharedAllocator()
            sorted_storage = shared_allocator.allocate(fx.Array[fx.Int32, BLOCK_M, 16])
            weight_storage = shared_allocator.allocate(fx.Array[fx.Float8E4M3FNUZ, 2 * BLOCK_N * BLOCK_K, 16])
            cshuffle_storage = None
            if const_expr(DEDICATED_K256_CSHUFFLE):
                cshuffle_storage = shared_allocator.allocate(
                    fx.Array[fx.BFloat16, 4 * 16 * BLOCK_N, 16]
                )
            sorted_lds = sorted_storage.peek().view(fx.make_layout(BLOCK_M, 1))
            if tid < BLOCK_M:
                sorted_lds[tid] = sorted_ids_buffer[tid]
            gpu.barrier()

            mm = get_down_ops(n_tiles_per_wg).create_thr_mma(fx.Float8E4M3FNUZ, (1, 4, 1))
            mma_atom = fx.make_mma_atom(
                fx.rocdl.MFMA(16, 16, 32, fx.Float8E4M3FNUZ)
            )
            c_fake = fx.make_view(
                fx.get_iter(input_tensor),
                fx.make_ordered_layout((BLOCK_N, BLOCK_M), (0, 1)),
            )
            frag_c = mm.make_fragment_C(c_fake)

            row_tensor = fx.make_view(
                fx.get_iter(sorted_weights),
                fx.make_layout((BLOCK_N, BLOCK_M), (0, 1)),
            )
            frag_row_scale = get_down_ops(n_tiles_per_wg).load_tiled_mma_fragC(
                mm, row_tensor, copy_atom_bits=32
            )
            coord_tensor = fx.make_view(
                fx.get_iter(sorted_lds),
                fx.make_layout((BLOCK_N, BLOCK_M), (0, 1)),
            )
            frag_coord = get_down_ops(n_tiles_per_wg).load_tiled_mma_fragC(
                mm, coord_tensor, copy_atom_bits=32
            )

            if const_expr(act_quant_type == "ptpc"):
                a_scale_tensor = fx.rocdl.make_buffer_tensor(
                    fxh.view_as_torch_tensor(p_a_scale, (M, TOPK), fx.Float32),
                    max_size=False,
                    num_records_bytes=fx.Int64(M) * TOPK * 4,
                )
                a_scale_copy = get_down_ops(n_tiles_per_wg).get_buffer_copy_atom(
                    fx.Float32, 32
                )
                frag_a_scale = mm.make_fragment_C(coord_tensor)
                frag_a_scale_retile = get_down_ops(
                    n_tiles_per_wg
                ).get_tiled_mma_retile(
                    mm, frag_a_scale, "C", copy_atom=a_scale_copy
                )
                for dst, coord in fxh.all_elements(
                    frag_a_scale_retile, frag_coord
                ):
                    sorted_id = coord[0].bitcast(fx.Uint32)
                    src = fxh.atom_tensor(
                        a_scale_tensor,
                        (sorted_id & 0xFFFFFF, sorted_id >> 24),
                        32,
                    )
                    fx.copy(a_scale_copy, src, dst)
                frag_row_scale.store(
                    frag_row_scale.load() * frag_a_scale.load()
                )
            else:
                a_scale = fx.make_view(
                    fxh._as_ptr(p_a_scale, fx.Float32), fx.make_layout(1, 1)
                )[0]
                frag_row_scale.store(frag_row_scale.load() * a_scale)

            input_copy = fx.make_copy_atom(
                (
                    fx.rocdl.BufferCopy128b()
                    if PIPELINE_K256_CSHUFFLE_PACK
                    else fx.rocdl.BufferCopy64b()
                ),
                fx.Float8E4M3FNUZ,
            )
            a_fragments = []
            for k_stage in range_constexpr(K_STAGES):
                a_fake = fx.make_view(
                    fx.get_iter(input_tensor),
                    fx.make_layout((BLOCK_M, BLOCK_K), (1, BLOCK_M)),
                )
                frag_a = mm.make_fragment_B(a_fake)
                for m_rep in range_constexpr(WAVE_M // 16):
                    local_row = wave_id * 16 + m_rep * 64 + lane_id % 16
                    sorted_id = sorted_lds[local_row].bitcast(fx.Uint32)
                    for k64 in range_constexpr(BLOCK_K // 64):
                        if const_expr(PIPELINE_K256_CSHUFFLE_PACK):
                            k_offset = (
                                k_stage * BLOCK_K
                                + k64 * 64
                                + (lane_id // 16) * 16
                            )
                            source = fxh.atom_tensor(
                                input_tensor,
                                (
                                    sorted_id & 0xFFFFFF,
                                    sorted_id >> 24,
                                    k_offset,
                                ),
                                128,
                            )
                            packed_input = fx.make_rmem_tensor(
                                fx.make_layout(16, 1), fx.Float8E4M3FNUZ
                            )
                            fx.copy(input_copy, source, packed_input)
                            packed_values = Vec(packed_input.load())
                            for k8 in range_constexpr(2):
                                frag_a[None, m_rep, (k8, k64)].store(
                                    packed_values.shuffle(
                                        packed_values,
                                        list(range(k8 * 8, k8 * 8 + 8)),
                                    )
                                )
                        else:
                            for k8 in range_constexpr(2):
                                k_offset = (
                                    k_stage * BLOCK_K
                                    + k64 * 64
                                    + (lane_id // 16) * 16
                                    + k8 * 8
                                )
                                source = fxh.atom_tensor(
                                    input_tensor,
                                    (
                                        sorted_id & 0xFFFFFF,
                                        sorted_id >> 24,
                                        k_offset,
                                    ),
                                    64,
                                )
                                fx.copy(
                                    input_copy,
                                    source,
                                    frag_a[None, m_rep, (k8, k64)],
                                )
                a_fragments.append(frag_a)
            fx.rocdl.s_waitcnt(_encode_waitcnt(vmcnt=0))

            weight_base = (
                fx.recast_iter(
                    fx.Float8E4M3FNUZ,
                    fxh._as_ptr(p_weight),
                )
                + fx.Int64(expert_id) * N * K
            )
            weight_flat = fx.rocdl.make_buffer_tensor(
                fx.make_view(weight_base, fx.make_layout(N * K, 1)),
                max_size=False,
                num_records_bytes=N * K,
            )
            weight_load_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Float8E4M3FNUZ)
            weight_store_atom = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Float8E4M3FNUZ)
            weight_copy_rounds = BLOCK_N * BLOCK_K // (16 * 256)
            weight_load_fragments = [
                [
                    fx.make_rmem_tensor(
                        fx.make_layout(16, 1), fx.Float8E4M3FNUZ
                    )
                    for _ in range_constexpr(weight_copy_rounds)
                ]
                for _ in range_constexpr(1 if USE_SINGLE_WEIGHT_SLOT else 2)
            ]

            def weight_lds_view(slot):
                ptr = weight_storage.peek().ptr + slot * (BLOCK_N * BLOCK_K)
                return fx.make_view(
                    ptr,
                    fx.make_layout(
                        ((16, BLOCK_N // 16), (16, BLOCK_K // 16)),
                        ((16, 16 * BLOCK_K), (1, 256)),
                    ),
                )

            lds_weights = [weight_lds_view(0), weight_lds_view(1)]

            def issue_weight_load(block_n, k_stage, fragment_slot):
                for copy_round in range_constexpr(weight_copy_rounds):
                    atom_index = tid + copy_round * 256
                    n_group = atom_index // BLOCK_K
                    within_group = atom_index % BLOCK_K
                    k_group = within_group // 16
                    n_inner = within_group % 16
                    global_offset = (
                        fx.Int32(block_n) * (BLOCK_N * K)
                        + n_group * (16 * K)
                        + (k_stage * (BLOCK_K // 16) + k_group) * 256
                        + n_inner * 16
                    )
                    source = fx.make_view(
                        fx.get_iter(weight_flat) + global_offset,
                        fx.make_layout(16, 1),
                    )
                    fx.copy(
                        weight_load_atom,
                        source,
                        weight_load_fragments[fragment_slot][copy_round],
                    )

            def commit_weight_lds(slot, fragment_slot):
                for copy_round in range_constexpr(weight_copy_rounds):
                    atom_index = tid + copy_round * 256
                    n_group = atom_index // BLOCK_K
                    within_group = atom_index % BLOCK_K
                    k_group = within_group // 16
                    n_inner = within_group % 16
                    lds_offset = slot * (BLOCK_N * BLOCK_K) + n_group * (16 * BLOCK_K) + k_group * 256 + n_inner * 16
                    destination = fx.make_view(
                        weight_storage.peek().ptr + lds_offset,
                        fx.make_layout(16, 1),
                    )
                    fx.copy(
                        weight_store_atom,
                        weight_load_fragments[fragment_slot][copy_round],
                        destination,
                    )

            output_base = fxh._as_ptr(p_output, fx.BFloat16) + fx.Int64(e_idx) * BLOCK_M * output_row_stride
            output_tensor = fx.make_view(
                output_base,
                fx.make_layout((N, BLOCK_M), (1, output_row_stride)),
            )
            output_tensor = fx.rocdl.make_buffer_tensor(
                output_tensor,
                max_size=False,
                num_records_bytes=BLOCK_M * output_row_stride * 2,
            )
            if const_expr(DEDICATED_K256_CSHUFFLE):
                cshuffle_lds = cshuffle_storage.peek().view(
                    fx.make_layout(4 * 16 * BLOCK_N, 1)
                )
            else:
                cshuffle_ptr = fx.recast_iter(
                    fx.BFloat16, weight_storage.peek().ptr
                )
                cshuffle_lds = fx.make_view(
                    cshuffle_ptr, fx.make_layout(4 * 16 * BLOCK_N, 1)
                )
            cshuffle_write_atom = get_down_ops(n_tiles_per_wg).get_universal_copy_atom(fx.BFloat16, 128)
            cshuffle_read_atom = get_down_ops(n_tiles_per_wg).get_universal_copy_atom(fx.BFloat16, 64)
            output_store_atom = fx.make_copy_atom(
                fx.rocdl.BufferCopy128b(
                    cache_modifier=2 if PIPELINE_K256_CSHUFFLE_PACK else 0
                ),
                fx.BFloat16,
            )

            weight_scale_base = None
            weight_scale_fragments = None
            if const_expr(
                PIPELINE_K256_WEIGHT_SCALE and weight_quant_type == "ptpc"
            ):
                weight_scale_base = fx.make_view(
                    fxh._as_ptr(p_w_scale, fx.Float32)
                    + fx.Int64(expert_id) * N,
                    fx.make_layout((BLOCK_N, BLOCK_M), (1, 0)),
                )
                first_weight_scale = get_down_ops(
                    n_tiles_per_wg
                ).load_tiled_mma_fragC(
                    mm, weight_scale_base, copy_atom_bits=32
                )
                weight_scale_fragments = [
                    first_weight_scale,
                    fx.make_fragment_like(first_weight_scale),
                ]

            lane_group = lane_id // 16
            lane_row = lane_id % 16
            row_in_8 = lane_row % 8
            row_half = lane_row // 8
            wave_lds_base = wave_id * (16 * BLOCK_N)

            def pack_cshuffle_row_pair(output, row_scales, weight_scales, row_pair):
                packed_records = []
                for n_pair in range_constexpr(2):
                    packed_chunks = []
                    for n_group in range_constexpr(2 * n_pair, 2 * n_pair + 2):
                        values = Vec(output[None, n_group, row_pair].load())
                        row_scale_values = Vec(
                            row_scales[None, n_group, row_pair].load()
                        )
                        if const_expr(weight_quant_type == "ptpc"):
                            weight_scale_values = Vec(
                                weight_scales[None, n_group, row_pair].load()
                            )
                        else:
                            weight_scale_values = weight_scales
                        weighted_values = fxh.eltwise_op(
                            "v_fma_f32",
                            values,
                            weight_scale_values,
                            fx.Float32(0.0),
                        )
                        packed_chunks.extend(
                            _pack_scaled_bf16_pairs(
                                weighted_values, row_scale_values
                            )
                        )
                    packed_records.append(
                        Vec.from_elements(packed_chunks, fx.Uint32).bitcast(
                            fx.BFloat16
                        )
                    )
                return packed_records

            def write_cshuffle_row_pair(packed_records):
                for n_pair in range_constexpr(2):
                    logical_record = n_pair * 4 + lane_group
                    physical_record = logical_record ^ row_in_8
                    lds_offset = (
                        wave_lds_base
                        + ((row_half * 8 + row_in_8) * 8 + physical_record) * 8
                    )
                    lds_dst = fx.make_view(
                        fx.get_iter(cshuffle_lds) + lds_offset,
                        fx.make_layout(8, 1),
                    )
                    lds_frag = fx.make_fragment_like(lds_dst)
                    lds_frag.store(packed_records[n_pair])
                    fx.copy(cshuffle_write_atom, lds_frag, lds_dst)

            def issue_read_cshuffle_row_pair(block_n, row_pair):
                out_frag_pairs = []
                destinations = []
                if const_expr(not PIPELINE_K256_CSHUFFLE_PACK):
                    fx.rocdl.s_waitcnt(_encode_waitcnt(lgkmcnt=0))
                for output_row_half in range_constexpr(2):
                    output_atom = lane_id % 8
                    n_group = output_atom // 2
                    n_pair = n_group // 2
                    chunk_half = n_group % 2
                    lane_group_begin = (output_atom % 2) * 2
                    out_frag_pair = []
                    for source_group in range_constexpr(2):
                        logical_record = (
                            n_pair * 4 + lane_group_begin + source_group
                        )
                        physical_record = logical_record ^ (lane_id // 8)
                        lds_offset = (
                            wave_lds_base
                            + (
                                (output_row_half * 8 + lane_id // 8) * 8
                                + physical_record
                            )
                            * 8
                            + chunk_half * 4
                        )
                        lds_src = fx.make_view(
                            fx.get_iter(cshuffle_lds) + lds_offset,
                            fx.make_layout(4, 1),
                        )
                        out_frag = fx.make_fragment_like(lds_src)
                        fx.copy(cshuffle_read_atom, lds_src, out_frag)
                        out_frag_pair.append(out_frag)
                    out_frag_pairs.append(out_frag_pair)
                    output_row = (
                        wave_id * 16
                        + row_pair * 64
                        + output_row_half * 8
                        + lane_id // 8
                    )
                    output_column = block_n * BLOCK_N + output_atom * 8
                    destinations.append(
                        fx.make_view(
                            fx.get_iter(output_tensor)
                            + output_tensor.layout(output_column, output_row),
                            fx.make_layout(8, 1),
                        )
                    )

                return out_frag_pairs, destinations

            def store_cshuffle_read_results(out_frag_pairs, destinations):
                fx.rocdl.s_waitcnt(_encode_waitcnt(lgkmcnt=2))
                first_output = fx.make_rmem_tensor(
                    fx.make_layout(8, 1), fx.BFloat16
                )
                first_output.store(
                    Vec(out_frag_pairs[0][0].load()).shuffle(
                        Vec(out_frag_pairs[0][1].load()), list(range(8))
                    )
                )
                fx.copy(output_store_atom, first_output, destinations[0])
                if const_expr(not PIPELINE_K256_CSHUFFLE_PACK):
                    fx.rocdl.s_waitcnt(_encode_waitcnt(lgkmcnt=0))
                second_output = fx.make_rmem_tensor(
                    fx.make_layout(8, 1), fx.BFloat16
                )
                second_output.store(
                    Vec(out_frag_pairs[1][0].load()).shuffle(
                        Vec(out_frag_pairs[1][1].load()), list(range(8))
                    )
                )
                fx.copy(output_store_atom, second_output, destinations[1])

            def read_store_cshuffle_row_pair(block_n, row_pair):
                out_frag_pairs, destinations = issue_read_cshuffle_row_pair(
                    block_n, row_pair
                )
                store_cshuffle_read_results(out_frag_pairs, destinations)

            def store_cshuffle(output, row_scales, weight_scales, block_n):
                if const_expr(PIPELINE_K256_CSHUFFLE_PACK):
                    packed_row_pair = pack_cshuffle_row_pair(
                        output, row_scales, weight_scales, 0
                    )
                    for row_pair in range_constexpr(WAVE_M // 16):
                        write_cshuffle_row_pair(packed_row_pair)
                        if const_expr(row_pair + 1 < WAVE_M // 16):
                            next_packed_row_pair = pack_cshuffle_row_pair(
                                output,
                                row_scales,
                                weight_scales,
                                row_pair + 1,
                            )
                        read_store_cshuffle_row_pair(block_n, row_pair)
                        if const_expr(row_pair + 1 < WAVE_M // 16):
                            packed_row_pair = next_packed_row_pair
                else:
                    legacy_lane_group = lane_id // 16
                    legacy_lane_row = lane_id % 16
                    legacy_row_in_8 = legacy_lane_row % 8
                    legacy_row_half = legacy_lane_row // 8
                    legacy_wave_lds_base = wave_id * (16 * BLOCK_N)
                    for row_pair in range_constexpr(WAVE_M // 16):
                        for n_pair in range_constexpr(2):
                            packed_chunks = []
                            for n_group in range_constexpr(
                                2 * n_pair, 2 * n_pair + 2
                            ):
                                values = Vec(
                                    output[None, n_group, row_pair].load()
                                )
                                row_scale_values = Vec(
                                    row_scales[
                                        None, n_group, row_pair
                                    ].load()
                                )
                                if const_expr(weight_quant_type == "ptpc"):
                                    weight_scale_values = Vec(
                                        weight_scales[
                                            None, n_group, row_pair
                                        ].load()
                                    )
                                else:
                                    weight_scale_values = weight_scales
                                weighted_values = fxh.eltwise_op(
                                    "v_fma_f32",
                                    values,
                                    weight_scale_values,
                                    fx.Float32(0.0),
                                )
                                packed_chunks.extend(
                                    _pack_scaled_bf16_pairs(
                                        weighted_values, row_scale_values
                                    )
                                )
                            packed_bf16 = Vec.from_elements(
                                packed_chunks, fx.Uint32
                            ).bitcast(fx.BFloat16)
                            logical_record = (
                                n_pair * 4 + legacy_lane_group
                            )
                            physical_record = (
                                logical_record ^ legacy_row_in_8
                            )
                            lds_offset = (
                                legacy_wave_lds_base
                                + (
                                    (
                                        legacy_row_half * 8
                                        + legacy_row_in_8
                                    )
                                    * 8
                                    + physical_record
                                )
                                * 8
                            )
                            lds_dst = fx.make_view(
                                fx.get_iter(cshuffle_lds) + lds_offset,
                                fx.make_layout(8, 1),
                            )
                            lds_frag = fx.make_fragment_like(lds_dst)
                            lds_frag.store(packed_bf16)
                            fx.copy(
                                cshuffle_write_atom, lds_frag, lds_dst
                            )

                        out_frag_pairs = []
                        destinations = []
                        fx.rocdl.s_waitcnt(
                            _encode_waitcnt(lgkmcnt=0)
                        )
                        for output_row_half in range_constexpr(2):
                            output_atom = lane_id % 8
                            n_group = output_atom // 2
                            n_pair = n_group // 2
                            chunk_half = n_group % 2
                            lane_group_begin = (output_atom % 2) * 2
                            out_frag_pair = []
                            for source_group in range_constexpr(2):
                                logical_record = (
                                    n_pair * 4
                                    + lane_group_begin
                                    + source_group
                                )
                                physical_record = (
                                    logical_record ^ (lane_id // 8)
                                )
                                lds_offset = (
                                    legacy_wave_lds_base
                                    + (
                                        (
                                            output_row_half * 8
                                            + lane_id // 8
                                        )
                                        * 8
                                        + physical_record
                                    )
                                    * 8
                                    + chunk_half * 4
                                )
                                lds_src = fx.make_view(
                                    fx.get_iter(cshuffle_lds) + lds_offset,
                                    fx.make_layout(4, 1),
                                )
                                out_frag = fx.make_fragment_like(lds_src)
                                fx.copy(
                                    cshuffle_read_atom, lds_src, out_frag
                                )
                                out_frag_pair.append(out_frag)
                            out_frag_pairs.append(out_frag_pair)
                            output_row = (
                                wave_id * 16
                                + row_pair * 64
                                + output_row_half * 8
                                + lane_id // 8
                            )
                            output_column = (
                                block_n * BLOCK_N + output_atom * 8
                            )
                            destinations.append(
                                fx.make_view(
                                    fx.get_iter(output_tensor)
                                    + output_tensor.layout(
                                        output_column, output_row
                                    ),
                                    fx.make_layout(8, 1),
                                )
                            )

                        fx.rocdl.s_waitcnt(
                            _encode_waitcnt(lgkmcnt=2)
                        )
                        first_output = fx.make_rmem_tensor(
                            fx.make_layout(8, 1), fx.BFloat16
                        )
                        first_output.store(
                            Vec(out_frag_pairs[0][0].load()).shuffle(
                                Vec(out_frag_pairs[0][1].load()),
                                list(range(8)),
                            )
                        )
                        fx.copy(
                            output_store_atom,
                            first_output,
                            destinations[0],
                        )
                        fx.rocdl.s_waitcnt(
                            _encode_waitcnt(lgkmcnt=0)
                        )
                        second_output = fx.make_rmem_tensor(
                            fx.make_layout(8, 1), fx.BFloat16
                        )
                        second_output.store(
                            Vec(out_frag_pairs[1][0].load()).shuffle(
                                Vec(out_frag_pairs[1][1].load()),
                                list(range(8)),
                            )
                        )
                        fx.copy(
                            output_store_atom,
                            second_output,
                            destinations[1],
                        )

            n_block_begin = fx.Int32(gpu.block_idx.x) * n_tiles_per_wg
            issue_weight_load(n_block_begin, 0, 0)
            fx.rocdl.s_waitcnt(_encode_waitcnt(vmcnt=0))
            commit_weight_lds(0, 0)
            _lds_barrier()
            for block_n_local in range_constexpr(n_tiles_per_wg):
                block_n = n_block_begin + block_n_local
                frag_c.fill(0)
                frag_weight_scale = None
                if const_expr(weight_quant_type == "ptpc"):
                    weight_scale = fx.make_view(
                        fxh._as_ptr(p_w_scale, fx.Float32) + fx.Int64(expert_id) * N + block_n * BLOCK_N,
                        fx.make_layout((BLOCK_N, BLOCK_M), (1, 0)),
                    )
                    if const_expr(PIPELINE_K256_WEIGHT_SCALE):
                        frag_weight_scale = weight_scale_fragments[
                            block_n_local & 1
                        ]
                    elif const_expr(K_STAGES <= 2):
                        frag_weight_scale = get_down_ops(n_tiles_per_wg).load_tiled_mma_fragC(
                            mm, weight_scale, copy_atom_bits=32
                        )
                for k_stage in range_constexpr(K_STAGES):
                    slot = k_stage & 1
                    if const_expr(not USE_SINGLE_WEIGHT_SLOT):
                        if const_expr(k_stage == 0 and K_STAGES > 1):
                            issue_weight_load(block_n, 1, 0)
                        if const_expr(k_stage + 1 < K_STAGES):
                            lookahead_fragment_slot = (k_stage & 1) ^ 1
                            if const_expr(k_stage + 2 < K_STAGES):
                                issue_weight_load(
                                    block_n,
                                    k_stage + 2,
                                    lookahead_fragment_slot,
                                )
                            else:
                                issue_weight_load(
                                    block_n + 1,
                                    0,
                                    lookahead_fragment_slot,
                                )
                        elif const_expr(K_STAGES == 1):
                            issue_weight_load(block_n + 1, 0, 0)
                    elif const_expr(k_stage + 1 < K_STAGES):
                        issue_weight_load(block_n, k_stage + 1, 0)
                    frag_weight = get_down_ops(n_tiles_per_wg).load_tiled_mma_fragA(
                        mm, lds_weights[slot], copy_atom_bits=128
                    )
                    if const_expr(
                        PIPELINE_K256_CSHUFFLE_PACK
                    ):
                        fx.rocdl.s_waitcnt(_encode_waitcnt(lgkmcnt=4))
                    else:
                        fx.rocdl.s_waitcnt(_encode_waitcnt(lgkmcnt=0))
                    _enter_compute_stage()
                    if const_expr(
                        PIPELINE_K256_CSHUFFLE_PACK and k_stage == 0
                    ):
                        first_weight = fx.make_view(
                            fx.get_iter(
                                frag_weight[None, None, (0, 0)]
                            ),
                            fx.make_layout((8, 2), (1, 32)),
                        )
                        first_output = fx.make_view(
                            fx.get_iter(frag_c),
                            fx.make_layout(
                                ((4, 1), 2, 4), ((1, 0), 4, 16)
                            ),
                        )
                        fx.gemm(
                            mm,
                            first_output,
                            first_weight,
                            a_fragments[k_stage][None, None, (0, 0)],
                            first_output,
                        )
                        fx.rocdl.sched_barrier(0)
                        fx.rocdl.s_waitcnt(_encode_waitcnt(lgkmcnt=0))
                        second_weight = fx.make_view(
                            fx.get_iter(
                                frag_weight[None, None, (0, 0)]
                            )
                            + 64,
                            fx.make_layout((8, 2), (1, 32)),
                        )
                        second_output = fx.make_view(
                            fx.get_iter(frag_c) + 8,
                            fx.make_layout(
                                ((4, 1), 2, 4), ((1, 0), 4, 16)
                            ),
                        )
                        fx.gemm(
                            mm,
                            second_output,
                            second_weight,
                            a_fragments[k_stage][None, None, (0, 0)],
                            second_output,
                        )
                        for k_iter in range_constexpr(BLOCK_K // 64):
                            for k_atom in range_constexpr(2):
                                if const_expr(
                                    not (k_iter == 0 and k_atom == 0)
                                ):
                                    fx.gemm(
                                        mm,
                                        frag_c,
                                        frag_weight[
                                            None,
                                            None,
                                            (k_atom, k_iter),
                                        ],
                                        a_fragments[k_stage][
                                            None,
                                            None,
                                            (k_atom, k_iter),
                                        ],
                                        frag_c,
                                    )
                        _enter_read_write_stage()
                    elif const_expr(
                        PIPELINE_K256_CSHUFFLE_PACK
                        and k_stage + 1 == K_STAGES
                    ):
                        def run_row_pair_mfma(
                            row_pair, n_group_begin=0, n_group_end=4
                        ):
                            for k_iter in range_constexpr(BLOCK_K // 64):
                                for k_atom in range_constexpr(2):
                                    for n_group in range_constexpr(
                                        n_group_begin, n_group_end
                                    ):
                                        fx.mma_atom_call(
                                            mma_atom,
                                            frag_c[None, n_group, row_pair],
                                            frag_weight[
                                                None,
                                                n_group,
                                                (k_atom, k_iter),
                                            ],
                                            a_fragments[k_stage][
                                                None,
                                                row_pair,
                                                (k_atom, k_iter),
                                            ],
                                            frag_c[None, n_group, row_pair],
                                        )

                        for row_pair in range_constexpr(WAVE_M // 16):
                            if const_expr(row_pair == 0):
                                run_row_pair_mfma(row_pair, 0, 2)
                                fx.rocdl.sched_barrier(0)
                                fx.rocdl.s_waitcnt(
                                    _encode_waitcnt(lgkmcnt=0)
                                )
                                run_row_pair_mfma(row_pair, 2, 4)
                                _enter_read_write_stage()
                                packed_row_pair = pack_cshuffle_row_pair(
                                    frag_c,
                                    frag_row_scale,
                                    frag_weight_scale,
                                    row_pair,
                                )
                                write_cshuffle_row_pair(packed_row_pair)
                                _enter_compute_stage()
                            elif const_expr(row_pair + 1 == WAVE_M // 16):
                                for k_iter in range_constexpr(1):
                                    for k_atom in range_constexpr(2):
                                        for n_group in range_constexpr(4):
                                            fx.mma_atom_call(
                                                mma_atom,
                                                frag_c[None, n_group, row_pair],
                                                frag_weight[
                                                    None,
                                                    n_group,
                                                    (k_atom, k_iter),
                                                ],
                                                a_fragments[k_stage][
                                                    None,
                                                    row_pair,
                                                    (k_atom, k_iter),
                                                ],
                                                frag_c[None, n_group, row_pair],
                                            )
                                _enter_read_write_stage()
                                out_frag_pairs, destinations = (
                                    issue_read_cshuffle_row_pair(
                                        block_n, row_pair - 1
                                    )
                                )
                                _enter_compute_stage()
                                for k_iter in range_constexpr(
                                    1, BLOCK_K // 64
                                ):
                                    for k_atom in range_constexpr(2):
                                        for n_group in range_constexpr(4):
                                            fx.mma_atom_call(
                                                mma_atom,
                                                frag_c[None, n_group, row_pair],
                                                frag_weight[
                                                    None,
                                                    n_group,
                                                    (k_atom, k_iter),
                                                ],
                                                a_fragments[k_stage][
                                                    None,
                                                    row_pair,
                                                    (k_atom, k_iter),
                                                ],
                                                frag_c[None, n_group, row_pair],
                                            )
                                _enter_read_write_stage()
                                store_cshuffle_read_results(
                                    out_frag_pairs, destinations
                                )
                                packed_row_pair = pack_cshuffle_row_pair(
                                    frag_c,
                                    frag_row_scale,
                                    frag_weight_scale,
                                    row_pair,
                                )
                                write_cshuffle_row_pair(packed_row_pair)
                            else:
                                run_row_pair_mfma(row_pair)
                                _enter_read_write_stage()
                                if const_expr(row_pair > 0):
                                    out_frag_pairs, destinations = (
                                        issue_read_cshuffle_row_pair(
                                            block_n, row_pair - 1
                                        )
                                    )
                                packed_row_pair = pack_cshuffle_row_pair(
                                    frag_c,
                                    frag_row_scale,
                                    frag_weight_scale,
                                    row_pair,
                                )
                                if const_expr(row_pair > 0):
                                    store_cshuffle_read_results(
                                        out_frag_pairs, destinations
                                    )
                                write_cshuffle_row_pair(packed_row_pair)
                                _enter_compute_stage()
                        read_store_cshuffle_row_pair(
                            block_n, WAVE_M // 16 - 1
                        )
                    else:
                        fx.gemm(
                            mm,
                            frag_c,
                            frag_weight,
                            a_fragments[k_stage],
                            frag_c,
                        )
                        _enter_read_write_stage()
                    if const_expr(k_stage + 1 < K_STAGES):
                        if const_expr(k_stage > 0):
                            _lds_barrier()
                        if const_expr(not USE_SINGLE_WEIGHT_SLOT):
                            fx.rocdl.s_waitcnt(
                                _encode_waitcnt(vmcnt=weight_copy_rounds)
                            )
                            commit_weight_lds(slot ^ 1, k_stage & 1)
                        else:
                            fx.rocdl.s_waitcnt(_encode_waitcnt(vmcnt=0))
                            commit_weight_lds(slot ^ 1, 0)
                        _lds_barrier()
                        if const_expr(
                            PIPELINE_K256_WEIGHT_SCALE
                            and k_stage == 0
                            and weight_quant_type == "ptpc"
                        ):
                            next_weight_scale = fx.make_view(
                                fx.get_iter(weight_scale_base)
                                + (block_n + 1) * BLOCK_N,
                                weight_scale_base.layout,
                            )
                            get_down_ops(n_tiles_per_wg).load_tiled_mma_fragC(
                                mm,
                                next_weight_scale,
                                dst=weight_scale_fragments[
                                    (block_n_local & 1) ^ 1
                                ],
                                copy_atom_bits=32,
                            )

                if const_expr(
                    USE_SINGLE_WEIGHT_SLOT
                    and block_n_local + 1 < n_tiles_per_wg
                ):
                    issue_weight_load(block_n + 1, 0, 0)

                if const_expr(weight_quant_type == "ptpc"):
                    if const_expr(K_STAGES > 2):
                        frag_weight_scale = get_down_ops(n_tiles_per_wg).load_tiled_mma_fragC(
                            mm, weight_scale, copy_atom_bits=32
                        )
                else:
                    weight_scale = fx.make_view(
                        fxh._as_ptr(p_w_scale, fx.Float32) + expert_id,
                        fx.make_layout(1, 1),
                    )[0]
                if const_expr(not DEDICATED_K256_CSHUFFLE):
                    _lds_barrier()
                if const_expr(not PIPELINE_K256_CSHUFFLE_PACK):
                    store_cshuffle(
                        frag_c,
                        frag_row_scale,
                        frag_weight_scale if weight_quant_type == "ptpc" else weight_scale,
                        block_n,
                    )
                if const_expr(block_n_local + 1 < n_tiles_per_wg):
                    if const_expr(not DEDICATED_K256_CSHUFFLE):
                        _lds_barrier()
                    fx.rocdl.s_waitcnt(_encode_waitcnt(vmcnt=WAVE_M // 8))
                    commit_weight_lds(
                        0, 0 if USE_SINGLE_WEIGHT_SLOT else (K_STAGES - 1) & 1
                    )
                    _lds_barrier()

    @flyc.jit
    def launch_prefill_4x1(
        p_input: fx.Pointer,
        p_weight: fx.Pointer,
        p_output: fx.Pointer,
        p_sorted_ids: fx.Pointer,
        p_sorted_weights: fx.Pointer,
        p_sorted_expert_ids: fx.Pointer,
        p_num_valid_ids: fx.Pointer,
        p_w_scale: fx.Pointer,
        p_a_scale: fx.Pointer,
        M: fx.Int32,
        task_num: fx.Int32,
        stream: fx.Stream,
    ):
        CompilationContext.get_current()
        full_down_ops.clear_all()
        full_kernel = moe_2stage_down_prefill_4x1(
            p_input,
            p_weight,
            p_output,
            p_sorted_ids,
            p_sorted_weights,
            p_sorted_expert_ids,
            p_num_valid_ids,
            p_w_scale,
            p_a_scale,
            M,
            FULL_N_TILES_PER_WG,
            value_attrs={"passthrough": [["target-features", "-packed-fp32-ops"]]},
        )
        if const_expr(USE_SHORT_N):
            short_down_ops.clear_all()
            short_kernel = moe_2stage_down_prefill_4x1(
                p_input,
                p_weight,
                p_output,
                p_sorted_ids,
                p_sorted_weights,
                p_sorted_expert_ids,
                p_num_valid_ids,
                p_w_scale,
                p_a_scale,
                M,
                SHORT_N_TILES_PER_WG,
                value_attrs={"passthrough": [["target-features", "-packed-fp32-ops"]]},
            )

            def launch_4x1(kernel, grid_x):
                kernel.launch(
                    grid=(grid_x, task_num, 1),
                    block=(256, 1, 1),
                    stream=stream,
                )

            if const_expr(BLOCK_M == 256):
                launch_4x1(
                    short_kernel,
                    N // (BLOCK_N * SHORT_N_TILES_PER_WG),
                )
            else:
                if task_num >= FULL_N_MIN_TASKS:
                    launch_4x1(full_kernel, 1)
                else:
                    launch_4x1(
                        short_kernel,
                        N // (BLOCK_N * SHORT_N_TILES_PER_WG),
                    )
        else:
            full_kernel.launch(
                grid=(1, task_num, 1),
                block=(256, 1, 1),
                stream=stream,
            )

    launch_prefill_4x1.compile_hints["target_features"] = "-packed-fp32-ops"
    return launch_prefill_4x1
