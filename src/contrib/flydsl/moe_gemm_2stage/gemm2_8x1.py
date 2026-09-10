# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""8x1入口与BK128原语绑定；N级流程由独立emitter编排，K192/K320各自特化。"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import fly_rocdl, llvm, rocdl as rocdl_dialect
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.typing import as_ir_value
from flydsl.expr.utils.arith import _to_raw as _raw

from . import layout_helpers as fxh
from .common import get_down_device_config


def _build_moe_gemm2_8x1(
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
    _task_table=False,
    _n_loop=1,
    _store_cache=2,
):
    # 1. Host配置与分支选择；不在此重新设计流水事件。
    del E, activation, swiglu_limit
    assert stage == "down"
    assert alg == "prefill_1x4"
    assert down_path == "8x1"
    assert weight_dtype == "fp8"
    assert weight_quant_type in ("ptpc", "per_tensor")
    if act_quant_type is None:
        act_quant_type = weight_quant_type
    assert (weight_quant_type == "ptpc" and act_quant_type == "ptpc") or (
        weight_quant_type == "per_tensor" and act_quant_type in ("ptpc", "per_tensor")
    ), f"unsupported 8x1 quant combo (weight={weight_quant_type}, act={act_quant_type})"
    assert not USE_ATOMIC_WRITE
    assert BLOCK_TILE_SIZE_M == 256
    assert BLOCK_TILE_SIZE_N == 128
    if METADATA_TILE_SIZE_M is None:
        METADATA_TILE_SIZE_M = BLOCK_TILE_SIZE_M
    assert METADATA_TILE_SIZE_M == BLOCK_TILE_SIZE_M
    assert K in (192, 256, 320, 384, 512, 640), "8x1仅支持K=192/256/320/384/512/640"
    if tile_k is None:
        tile_k = 192 if K in (192, 320) else 128
    assert tile_k == 128 or (K in (192, 320) and tile_k == 192), "8x1仅保留K192整块192、K320的128+192，其余K使用BK128"
    # 兼容统一传入BK128的现有调用；两种参数均只生成新实现，不再保留旧分支。
    if K in (192, 320):
        tile_k = 192
    assert N % BLOCK_TILE_SIZE_N == 0
    assert down_output_padding_bytes in (0, 32, 64, 128)
    assert _n_loop in (0, 1, 2)
    assert _store_cache in (0, 2), "store aux: 0=普通，2=SLC"

    if K == 192:
        from .gemm2_8x1_k192 import _build_moe_gemm2_8x1_k192

        return _build_moe_gemm2_8x1_k192(
            N, TOPK, down_output_padding_bytes,
            weight_quant_type=weight_quant_type, act_quant_type=act_quant_type,
            _task_table=_task_table, _n_loop=_n_loop,
            _store_cache=_store_cache,
        )

    if K == 320:
        from .gemm2_8x1_k320 import _build_moe_gemm2_8x1_k320

        return _build_moe_gemm2_8x1_k320(
            N, TOPK, down_output_padding_bytes,
            weight_quant_type=weight_quant_type, act_quant_type=act_quant_type,
            _task_table=_task_table, _n_loop=_n_loop,
            _store_cache=_store_cache,
        )

    assert K % tile_k == 0

    BLOCK_M = BLOCK_TILE_SIZE_M
    BLOCK_N = BLOCK_TILE_SIZE_N
    BLOCK_K = tile_k
    NUM_WAVES = 8
    WAVE_M = BLOCK_M // NUM_WAVES
    K_STAGES = K // BLOCK_K
    N_TILES = N // BLOCK_N
    TOTAL_CORES = N_TILES * K_STAGES
    WEIGHT_QUARTER_ATOMS = BLOCK_N * BLOCK_K // (16 * 4)
    WEIGHT_PREFETCH_SLOTS = 2
    OUTPUT_STORES_PER_WAVE = WAVE_M * BLOCK_N * 2 // (64 * 16)
    CSHUFFLE_N_PAIRS = BLOCK_N // 32
    output_row_stride = N + down_output_padding_bytes // 2
    use_n_loop = bool(_n_loop and N_TILES >= 3)
    from .gemm2_8x1_schedule import packing_events, vmem_wait_schedule
    vmem_budgets = vmem_wait_schedule(K, N_TILES, weight_quant_type == "ptpc")
    CSHUFFLE_WAVES = 4
    down_ops = fxh.FlyObjCache()
    topology_enabled, xcc_count = get_down_device_config()
    se_per_xcc = 4
    cu_per_se = 5
    se_count = xcc_count * se_per_xcc

    assert WEIGHT_QUARTER_ATOMS == 256
    assert OUTPUT_STORES_PER_WAVE == 8

    def _encode_waitcnt(vmcnt=63, expcnt=7, lgkmcnt=63):
        vm_lo = vmcnt & 0xF
        vm_hi = (vmcnt >> 4) & 0x3
        return vm_lo | (expcnt << 4) | (lgkmcnt << 8) | (vm_hi << 14)

    def _stage_end():
        rocdl.sched_barrier(0)
        rocdl.s_barrier()
        rocdl.sched_barrier(0)

    def _enter_memory_stage():
        rocdl.sched_barrier(0)
        rocdl.s_setprio(0)
        rocdl.sched_barrier(0)

    def _enter_compute_stage():
        rocdl.sched_barrier(0)
        rocdl.s_setprio(3)
        rocdl.sched_barrier(0)

    @flyc.kernel(known_block_size=[512, 1, 1])
    def moe_2stage_down_prefill_8x1(
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
    ):
        # 2. 任务/布局与驻留状态：所有早退均在两组错相开始之前。
        tid = fx.Int32(gpu.thread_idx.x)
        lane_id = tid % 64
        wave_id = tid // 64
        wave_group = fx.Int32(
            rocdl.readfirstlane(
                ir.IntegerType.get_signless(32),
                _raw(tid // 256),
            )
        )
        group_tid = tid % 256
        max_valid_id = fxh.view_as_torch_tensor(
            p_num_valid_ids, (1,), fx.Int32
        )[0]
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
            se_local_rank = (
                cu_slot * short_cu_tasks + cu_prefix_extra + cu_round
            )
            logical_xcc = (xcc_id + 2) & (xcc_count - 1)
            mapped_e_idx = (
                logical_xcc * tasks_per_xcc
                + se_slot * tasks_per_se
                + se_local_rank
            )
            e_idx = fx.Int32(
                arith.select(
                    workgroup_idx_u32 < mapped_tasks,
                    mapped_e_idx,
                    workgroup_idx_u32,
                )
            )

        if e_idx * BLOCK_M < max_valid_id:
            if const_expr(_task_table):
                row_begin = p_sorted_expert_ids[2 * e_idx]
            input_tensor = fx.rocdl.make_buffer_tensor(
                fxh.view_as_torch_tensor(
                    p_input, (M, TOPK, K), fx.Float8E4M3FNUZ
                ),
                max_size=False,
                num_records_bytes=fx.Int64(M) * TOPK * K,
            )
            sorted_ids = fx.rocdl.make_buffer_tensor(
                fxh.view_as_torch_tensor(
                    fxh._as_ptr(p_sorted_ids) + (fx.Int64(row_begin) if const_expr(_task_table) else fx.Int64(e_idx) * BLOCK_M),
                    (BLOCK_M,),
                    fx.Int32,
                ),
                max_size=False,
                num_records_bytes=BLOCK_M * 4,
            )
            sorted_weights = fxh.view_as_torch_tensor(
                fxh._as_ptr(p_sorted_weights) + (fx.Int64(row_begin) if const_expr(_task_table) else fx.Int64(e_idx) * BLOCK_M),
                (BLOCK_M,),
                fx.Float32,
            )
            if const_expr(_task_table):
                expert_id = p_sorted_expert_ids[2 * e_idx + 1]
            else:
                expert_id = fxh.view_as_torch_tensor(
                    p_sorted_expert_ids, (1,), fx.Int32
                )[e_idx]

            shared_allocator = fx.SharedAllocator()
            if const_expr(use_n_loop):
                weight_slots = shared_allocator.allocate(
                    fx.Array[fx.Float8E4M3FNUZ, 2 * BLOCK_N * BLOCK_K, 16]
                )
                weight_storage_ptrs = [weight_slots.peek().ptr, weight_slots.peek().ptr + BLOCK_N * BLOCK_K]
            else:
                weight_ping_storage = shared_allocator.allocate(
                    fx.Array[fx.Float8E4M3FNUZ, BLOCK_N * BLOCK_K, 16]
                )
                weight_pong_storage = shared_allocator.allocate(
                    fx.Array[fx.Float8E4M3FNUZ, BLOCK_N * BLOCK_K, 16]
                )
                weight_storage_ptrs = [weight_ping_storage.peek().ptr, weight_pong_storage.peek().ptr]
            cshuffle_storage = shared_allocator.allocate(
                fx.Array[fx.BFloat16, CSHUFFLE_WAVES * 16 * BLOCK_N, 16]
            )
            sorted_lds = fx.make_view(
                fx.recast_iter(fx.Int32, cshuffle_storage.peek().ptr),
                fx.make_layout(BLOCK_M, 1),
            )
            if tid < BLOCK_M:
                sorted_lds[tid] = sorted_ids[tid]
            gpu.barrier()

            mm = down_ops.create_thr_mma(
                fx.Float8E4M3FNUZ, (1, NUM_WAVES, 1)
            )
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
            frag_row_scale = down_ops.load_tiled_mma_fragC(
                mm, row_tensor, copy_atom_bits=32
            )
            if const_expr(act_quant_type == "ptpc"):
                coord_tensor = fx.make_view(
                    fx.get_iter(sorted_lds),
                    fx.make_layout((BLOCK_N, BLOCK_M), (0, 1)),
                )
                frag_coord = down_ops.load_tiled_mma_fragC(mm, coord_tensor, copy_atom_bits=32)
                a_scale_tensor = fx.rocdl.make_buffer_tensor(
                    fxh.view_as_torch_tensor(p_a_scale, (M, TOPK), fx.Float32),
                    max_size=False, num_records_bytes=fx.Int64(M) * TOPK * 4,
                )
                a_scale_copy = down_ops.get_buffer_copy_atom(fx.Float32, 32)
                frag_a_scale = mm.make_fragment_C(coord_tensor)
                frag_a_scale_retile = down_ops.get_tiled_mma_retile(
                    mm, frag_a_scale, "C", copy_atom=a_scale_copy
                )
                for dst, coord in fxh.all_elements(frag_a_scale_retile, frag_coord):
                    sorted_id = coord[0].bitcast(fx.Uint32)
                    source = fxh.atom_tensor(
                        a_scale_tensor, (sorted_id & 0xFFFFFF, sorted_id >> 24), 32,
                    )
                    fx.copy(a_scale_copy, source, dst)
                frag_row_scale.store(frag_row_scale.load() * frag_a_scale.load())
                if const_expr(weight_quant_type == "per_tensor"):
                    scalar_weight_scale = fx.make_view(
                        fxh._as_ptr(p_w_scale, fx.Float32) + expert_id, fx.make_layout(1, 1),
                    )[0]
                    frag_row_scale.store(frag_row_scale.load() * scalar_weight_scale)
            else:
                # 与1x8相同：每专家一个weight scale、全局一个activation scale，提前融合。
                scalar_a_scale = fx.make_view(
                    fxh._as_ptr(p_a_scale, fx.Float32), fx.make_layout(1, 1)
                )[0]
                scalar_w_scale = fx.make_view(
                    fxh._as_ptr(p_w_scale, fx.Float32) + expert_id, fx.make_layout(1, 1),
                )[0]
                frag_row_scale.store(frag_row_scale.load() * (scalar_a_scale * scalar_w_scale))

            weight_base = (
                fx.recast_iter(fx.Float8E4M3FNUZ, fxh._as_ptr(p_weight))
                + fx.Int64(expert_id) * N * K
            )
            weight_view = fx.make_view(
                weight_base, fx.make_layout(N * K, 1)
            )
            weight_flat = fx.rocdl.make_buffer_tensor(
                weight_view,
                max_size=False,
                num_records_bytes=N * K,
            )
            weight_rsrc = fly_rocdl.get_buffer_rsrc(
                _raw(fx.get_iter(weight_flat)),
                results=[ir.Type.parse("!llvm.ptr<8>")],
            )
            weight_staging = [
                fx.make_rmem_tensor(
                    fx.make_layout(16, 1), fx.Float8E4M3FNUZ
                )
                for _ in range_constexpr(
                    WEIGHT_PREFETCH_SLOTS
                )
            ]
            weight_store_atom = fx.make_copy_atom(
                fx.UniversalCopy128b(), fx.Float8E4M3FNUZ
            )

            def weight_lds_half_view(pointer, n_half):
                return fx.make_view(
                    pointer
                    + n_half * (BLOCK_N // 2) * BLOCK_K,
                    fx.make_layout(
                        ((16, BLOCK_N // 32), (16, BLOCK_K // 16)),
                        ((16, 16 * BLOCK_K), (1, 256)),
                    ),
                )

            def weight_lds_quarter_view(pointer, n_quarter):
                return fx.make_view(
                    pointer
                    + n_quarter * (BLOCK_N // 4) * BLOCK_K,
                    fx.make_layout(
                        ((16, BLOCK_N // 64), (16, BLOCK_K // 16)),
                        ((16, 16 * BLOCK_K), (1, 256)),
                    ),
                )

            lds_weight_halves = [
                [
                    weight_lds_half_view(storage, n_half)
                    for n_half in range_constexpr(2)
                ]
                for storage in weight_storage_ptrs
            ]
            lds_weight_quarters = [
                [
                    weight_lds_quarter_view(storage, n_quarter)
                    for n_quarter in range_constexpr(4)
                ]
                for storage in weight_storage_ptrs
            ]

            quarter_atom_index = (
                wave_group * WEIGHT_QUARTER_ATOMS
                + group_tid
            )
            quarter_n_group = quarter_atom_index // BLOCK_K
            quarter_within_group = quarter_atom_index % BLOCK_K
            quarter_k_group = quarter_within_group // 16
            quarter_n_inner = quarter_within_group % 16
            weight_quarter_lane_offset_bytes = (
                quarter_n_group * (16 * K)
                + quarter_k_group * 256
                + quarter_n_inner * 16
            )

            def issue_weight_quarter_load(
                block_n, k_stage, n_half, staging_index=0
            ):
                core_base_bytes = (
                    block_n * (BLOCK_N * K)
                    + k_stage * (BLOCK_K // 16) * 256
                    + n_half * (BLOCK_N // 2) * K
                )
                loaded = Vec(
                    rocdl_dialect.RawPtrBufferLoadOp(
                        ir.VectorType.get(
                            [4], ir.IntegerType.get_signless(32)
                        ),
                        weight_rsrc,
                        _raw(
                            fx.Int32(
                                weight_quarter_lane_offset_bytes
                            )
                        ),
                        _raw(fx.Int32(core_base_bytes)),
                        aux=ir.IntegerAttr.get(
                            ir.IntegerType.get_signless(32), 0
                        ),
                    ).result
                ).bitcast(fx.Float8E4M3FNUZ)
                weight_staging[staging_index].store(loaded)

            def commit_weight_quarter(
                slot, n_half, staging_index=0
            ):
                n_group = quarter_atom_index // BLOCK_K
                within_group = quarter_atom_index % BLOCK_K
                k_group = within_group // 16
                n_inner = within_group % 16
                lds_offset = (
                    n_half * (BLOCK_N // 2) * BLOCK_K
                    + n_group * (16 * BLOCK_K)
                    + k_group * 256
                    + n_inner * 16
                )
                destination = fx.make_view(
                    (weight_storage_ptrs[0] + slot * BLOCK_N * BLOCK_K if const_expr(use_n_loop)
                     else weight_storage_ptrs[slot]) + lds_offset,
                    fx.make_layout(16, 1),
                )
                fx.copy(
                    weight_store_atom,
                    weight_staging[staging_index],
                    destination,
                )

            # 3. Prologue第一段：首B必须早于A gather发出，不能合并到后面的初始化。
            # 只预填Q0=L/K0；其余请求统一按消费者顺序推进。
            # 仍在A gather前发出，以年轻的A/scale请求覆盖启动延迟。
            issue_weight_quarter_load(0, 0, 0, 0)
            fx.rocdl.sched_barrier(0)

            input_copy = fx.make_copy_atom(
                fx.rocdl.BufferCopy128b(), fx.Float8E4M3FNUZ
            )
            a_fragments = []
            for k_stage in range_constexpr(K_STAGES):
                a_fake = fx.make_view(
                    fx.get_iter(input_tensor),
                    fx.make_layout((BLOCK_M, BLOCK_K), (1, BLOCK_M)),
                )
                frag_a = mm.make_fragment_B(a_fake)
                for m_rep in range_constexpr(WAVE_M // 16):
                    local_row = (
                        wave_id * 16
                        + m_rep * (NUM_WAVES * 16)
                        + lane_id % 16
                    )
                    sorted_id = sorted_lds[local_row].bitcast(fx.Uint32)
                    for k64 in range_constexpr(BLOCK_K // 64):
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
                a_fragments.append(frag_a)

            output_base = (
                fxh._as_ptr(p_output, fx.BFloat16)
                + (fx.Int64(row_begin) * output_row_stride if const_expr(_task_table) else fx.Int64(e_idx) * BLOCK_M * output_row_stride)
            )
            output_tensor = fx.rocdl.make_buffer_tensor(
                fx.make_view(
                    output_base,
                    fx.make_layout(
                        (N, BLOCK_M), (1, output_row_stride)
                    ),
                ),
                max_size=False,
                num_records_bytes=BLOCK_M * output_row_stride * 2,
            )
            cshuffle_lds = cshuffle_storage.peek().view(
                fx.make_layout(CSHUFFLE_WAVES * 16 * BLOCK_N, 1)
            )
            cshuffle_write_atom = down_ops.get_universal_copy_atom(
                fx.BFloat16, 128
            )
            cshuffle_read_atom = down_ops.get_universal_copy_atom(
                fx.BFloat16, 64
            )
            output_store_atom = fx.make_copy_atom(
                fx.rocdl.BufferCopy128b(cache_modifier=_store_cache), fx.BFloat16
            )
            lane_group = lane_id // 16
            lane_row = lane_id % 16
            local_wave = wave_id % CSHUFFLE_WAVES
            wave_lds_base = local_wave * (16 * BLOCK_N)
            output_destination_offsets = []
            for row_pair in range_constexpr(WAVE_M // 16):
                row_pair_offsets = []
                for n_half in range_constexpr(2):
                    for output_row_half in range_constexpr(2):
                        output_atom = n_half * 8 + lane_id % 8
                        output_row = (
                            wave_id * 16
                            + row_pair * (NUM_WAVES * 16)
                            + output_row_half * 8
                            + lane_id // 8
                        )
                        row_pair_offsets.append(
                            output_tensor.layout(
                                output_atom * 8, output_row
                            )
                        )
                output_destination_offsets.append(row_pair_offsets)

            def pack_cshuffle_super_record(
                output, row_scales, weight_scales, n_pair
            ):
                weighted_fragments = []
                row_scale_fragments = []
                for row_pair in range_constexpr(WAVE_M // 16):
                    for n_group in range_constexpr(
                        2 * n_pair, 2 * n_pair + 2
                    ):
                        values = Vec(
                            output[None, n_group, row_pair].load()
                        )
                        if const_expr(weight_quant_type == "ptpc"):
                            weighted_fragments.append(
                                fxh.eltwise_op(
                                    "v_fma_f32", values,
                                    Vec(weight_scales[None, n_group % (BLOCK_N // 64), row_pair].load()),
                                    fx.Float32(0.0),
                                )
                            )
                        else:
                            weighted_fragments.append(values)
                        row_scale_fragments.append(
                            Vec(
                                row_scales[
                                    None, n_group, row_pair
                                ].load()
                            )
                        )

                fma_bias = as_ir_value(
                    fx.Uint32(0x8000)
                ).bitcast(fx.Float32.ir_type)
                scaled_fragments = [
                    fxh.eltwise_op(
                        "llvm.fma.f32",
                        weighted_fragments[fragment_index],
                        row_scale_fragments[fragment_index],
                        fma_bias,
                    )
                    for fragment_index in range_constexpr(
                        len(weighted_fragments)
                    )
                ]

                selector = fx.Uint32(0x07060302)
                packed_records = [
                    []
                    for _ in range_constexpr(WAVE_M // 16)
                ]
                for fragment_index in range_constexpr(
                    len(scaled_fragments)
                ):
                    scaled = scaled_fragments[fragment_index]
                    row_pair = fragment_index // 2
                    for index in range_constexpr(0, scaled.numel, 2):
                        packed_records[row_pair].append(
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
                return [
                    Vec.from_elements(chunks, fx.Uint32).bitcast(
                        fx.BFloat16
                    )
                    for chunks in packed_records
                ]

            def cshuffle_plane_offset(row, group, pair):
                # BF16元素偏移；source-group低位移到2KiB plane，保持128bit写。
                return ((row & 1) * 8 + ((group & 2) ^ (row & 2)) * 8
                        + ((pair & 1) ^ ((row >> 2) & 1)) * 32 + (pair >> 1) * 64
                        + (row >> 3) * 128 + (row & 6) * 128 + (group & 1) * 1024)

            # C读地址不随N/row_pair改变；完整地址固定，防止常量行偏移沉入memory。
            cshuffle_read_pointers = []
            for n_half in range_constexpr(2):
                for output_row_half in range_constexpr(2):
                    output_atom = n_half * 8 + lane_id % 8
                    n_group = output_atom // 2
                    offset = (wave_lds_base + cshuffle_plane_offset(
                        output_row_half * 8 + lane_id // 8, (output_atom % 2) * 2, n_group // 2)
                        + (n_group % 2) * 4)
                    pointer = fx.get_iter(cshuffle_lds) + offset
                    address = fx.Int32(llvm.inline_asm(ir.IntegerType.get_signless(32),
                        [_raw(fx.Int32(fx.ptrtoint(pointer)))], "", "=v,0", has_side_effects=True))
                    cshuffle_read_pointers.append(fx.inttoptr(pointer.type, address))

            def write_cshuffle_quarter(packed_records, n_half):
                for local_n_pair in range_constexpr(
                    CSHUFFLE_N_PAIRS // 2
                ):
                    n_pair = (
                        n_half * (CSHUFFLE_N_PAIRS // 2)
                        + local_n_pair
                    )
                    lds_offset = wave_lds_base + cshuffle_plane_offset(lane_row, lane_group, n_pair)
                    destination = fx.make_view(
                        fx.get_iter(cshuffle_lds) + lds_offset,
                        fx.make_layout(8, 1),
                    )
                    fragment = fx.make_fragment_like(destination)
                    fragment.store(packed_records[local_n_pair])
                    fx.copy(
                        cshuffle_write_atom, fragment, destination
                    )

            def issue_read_cshuffle_quarter(block_n, row_pair, n_half):
                output_fragments = []
                destinations = []
                for output_row_half in range_constexpr(2):
                    fragment_pair = []
                    for source_group in range_constexpr(2):
                        source = fx.make_view(cshuffle_read_pointers[2 * n_half + output_row_half] + source_group * 1024,
                                              fx.make_layout(4, 1))
                        fragment = fx.make_fragment_like(source)
                        fx.copy(cshuffle_read_atom, source, fragment)
                        fragment_pair.append(fragment)
                    # 只合并同一输出行的两段8B，避免跨行配对产生额外搬运。
                    fx.rocdl.sched_barrier(0)
                    output_fragments.append(fragment_pair)
                    destination_index = (
                        n_half * 2 + output_row_half
                    )
                    destinations.append((block_n, fx.make_view(
                        fx.get_iter(output_tensor) + output_destination_offsets[row_pair][destination_index],
                        fx.make_layout(8, 1))))
                return output_fragments, destinations

            def store_cshuffle_read_results(
                output_fragments, destinations, lgkmcnt=0
            ):
                fx.rocdl.s_waitcnt(
                    _encode_waitcnt(lgkmcnt=lgkmcnt)
                )
                for output_index in range_constexpr(
                    len(output_fragments)
                ):
                    first = Vec(
                        output_fragments[output_index][0].load()
                    )
                    second = Vec(
                        output_fragments[output_index][1].load()
                    )
                    output_fragment = fx.make_rmem_tensor(
                        fx.make_layout(8, 1), fx.BFloat16
                    )
                    output_fragment.store(
                        first.shuffle(second, list(range(8)))
                    )
                    output_n, destination = destinations[output_index]
                    fx.copy(output_store_atom, output_fragment, destination, soffset=fx.Int32(output_n * BLOCK_N))

            if const_expr(weight_quant_type == "ptpc"):
                weight_scale_buffer = fx.rocdl.make_buffer_tensor(
                    fxh.view_as_torch_tensor(fxh._as_ptr(p_w_scale, fx.Float32) + fx.Int64(expert_id) * N, (N,), fx.Float32),
                    max_size=False, num_records_bytes=N * 4,
                )

            def load_weight_scale(block_n, first_column, columns):
                weight_scale = fx.make_view(fx.get_iter(weight_scale_buffer) + first_column,
                                            fx.make_layout((columns, BLOCK_M), (1, 0)))
                # BufferCopy的soffset为元素；128bit保持每quarter的两条dwordx4。
                copy_atom = down_ops.get_buffer_copy_atom(fx.Float32, 128)
                fragment = mm.make_fragment_C(weight_scale)
                fx.copy(copy_atom, down_ops.get_tiled_mma_partition_S(mm, weight_scale, "C", copy_atom_bits=128),
                        down_ops.get_tiled_mma_retile(mm, fragment, "C", copy_atom=copy_atom),
                        soffset=fx.Int32(block_n * BLOCK_N))
                return fragment

            def issue_packed_output_quarter(
                block_n,
                packed_super_records,
                row_pair,
                n_half,
            ):
                packed_records = [
                    packed_super_records[
                        n_half * (CSHUFFLE_N_PAIRS // 2)
                        + local_n_pair
                    ][row_pair]
                    for local_n_pair in range_constexpr(
                        CSHUFFLE_N_PAIRS // 2
                    )
                ]
                write_cshuffle_quarter(packed_records, n_half)
                return issue_read_cshuffle_quarter(
                    block_n, row_pair, n_half
                )

            def store_packed_output_quarter(
                block_n,
                packed_super_records,
                row_pair,
                n_half,
                lgkmcnt=0,
            ):
                output_fragments, destinations = (
                    issue_packed_output_quarter(
                        block_n,
                        packed_super_records,
                        row_pair,
                        n_half,
                    )
                )
                store_cshuffle_read_results(
                    output_fragments,
                    destinations,
                    lgkmcnt=lgkmcnt,
                )

            def run_super_record_mfma(
                frag_weight, k_stage, n_pair, weight_n_group_begin
            ):
                local_n_pair = n_pair % (CSHUFFLE_N_PAIRS // 2)
                for k_iter in range_constexpr(BLOCK_K // 64):
                    for k_atom in range_constexpr(2):
                        for row_pair in range_constexpr(WAVE_M // 16):
                            for quarter_n_group in range_constexpr(2):
                                local_n_group = (
                                    2 * local_n_pair + quarter_n_group
                                )
                                n_group = (
                                    (n_pair // 2)
                                    * (BLOCK_N // 32)
                                    + local_n_group
                                )
                                fx.mma_atom_call(
                                    mma_atom,
                                    frag_c[
                                        None, n_group, row_pair
                                    ],
                                    frag_weight[
                                        None,
                                        weight_n_group_begin
                                        + quarter_n_group,
                                        (k_atom, k_iter),
                                    ],
                                    a_fragments[k_stage][
                                        None,
                                        row_pair,
                                        (k_atom, k_iter),
                                    ],
                                    frag_c[
                                        None, n_group, row_pair
                                    ],
                                )

            def clear_super_record(n_pair):
                for row_pair in range_constexpr(WAVE_M // 16):
                    for n_group in range_constexpr(
                        2 * n_pair, 2 * n_pair + 2
                    ):
                        frag_c[
                            None, n_group, row_pair
                        ].fill(0)

            def load_weight_scale_quarter(block_n, n_pair):
                if const_expr(weight_quant_type == "ptpc"):
                    return load_weight_scale(block_n, n_pair * (BLOCK_N // CSHUFFLE_N_PAIRS), BLOCK_N // CSHUFFLE_N_PAIRS)
                else:
                    # 保留编译期退休列表形状，不发送逐N的scale VMEM。
                    return fx.Float32(1.0)

            def constrain_mfma_valu_packet():
                for slot in range_constexpr(16):
                    fx.rocdl.sched_group_barrier(0x8, 1, 0)
                    if const_expr(weight_quant_type == "ptpc"):
                        if const_expr(slot < 13):
                            fx.rocdl.sched_group_barrier(0x2, 3, 0)
                        elif const_expr(slot == 13):
                            fx.rocdl.sched_group_barrier(0x2, 1, 0)
                    else:
                        # 融合标量scale后只有16 FMA+8 perm，共24条VALU。
                        fx.rocdl.sched_group_barrier(0x2, 2 if slot < 8 else 1, 0)
                fx.rocdl.sched_barrier(0)

            def load_rolling_scales(block_n, stage_in_tile, scale_quarters):
                # 通用rolling路径在前四stage加载scale。
                if const_expr(stage_in_tile < CSHUFFLE_N_PAIRS):
                    scale_quarters.append(
                        load_weight_scale_quarter(block_n, stage_in_tile)
                    )

            def pending_weight_position(half_core):
                # Q[q]是当前消费者；提交Q[q+1]，两拍前发起其VMEM。
                target = half_core + 1
                target_n = target // (2 * K_STAGES)
                target_k = target % K_STAGES
                return (
                    target_n, target_k, (target % (2 * K_STAGES)) // K_STAGES,
                    target < 2 * TOTAL_CORES, (target_n * K_STAGES + target_k) & 1,
                )

            def seed_weight_pipeline():
                # 首B此前已发出；这里只落LDS并建立Q1/Q2种子，不重排A gather。
                fx.rocdl.s_waitcnt(_encode_waitcnt(vmcnt=4))
                commit_weight_quarter(0, 0, 0)
                fx.rocdl.sched_barrier(0)
                prefetch0 = pending_weight_position(0)
                prefetch1 = pending_weight_position(1)
                if const_expr(prefetch0[3]):
                    issue_weight_quarter_load(
                        prefetch0[0], prefetch0[1], prefetch0[2], 0
                    )
                if const_expr(prefetch1[3]):
                    issue_weight_quarter_load(
                        prefetch1[0], prefetch1[1], prefetch1[2], 1
                    )
                fx.rocdl.s_waitcnt(
                    _encode_waitcnt(vmcnt=1 if prefetch1[3] else 0)
                )
                _stage_end()
                return prefetch0

            def emit_first_stage(prefetch0):
                # 首M0/C0有独立wait；结束后group1多一次barrier建立错相。
                _enter_memory_stage()
                scale_quarters = []
                load_rolling_scales(0, 0, scale_quarters)
                frag_weight = down_ops.load_tiled_mma_fragA(
                    mm,
                    lds_weight_halves[0][0],
                    copy_atom_bits=128,
                )
                fx.rocdl.s_waitcnt(_encode_waitcnt(vmcnt=2 if weight_quant_type == "ptpc" else 0))
                if const_expr(prefetch0[3]):
                    commit_weight_quarter(prefetch0[4], prefetch0[2], 0)
                prefetch2 = pending_weight_position(2)
                if const_expr(prefetch2[3]):
                    fx.rocdl.sched_barrier(0)
                    issue_weight_quarter_load(prefetch2[0], prefetch2[1], prefetch2[2], 0)
                fx.rocdl.s_waitcnt(_encode_waitcnt(lgkmcnt=0))
                _stage_end()
                _enter_compute_stage()
                for n_pair in range_constexpr(2):
                    run_super_record_mfma(frag_weight, 0, n_pair, 2 * n_pair)
                _enter_memory_stage()
                _stage_end()
                if wave_group == 1:
                    _stage_end()
                return scale_quarters, frag_weight

            def drain_output(pending_packed_super_records, pending_scale_quarters):
                # 最后N只补未pack的C3；旧BF16不可提前覆盖。
                fx.rocdl.s_waitcnt(_encode_waitcnt(vmcnt=0))
                pending_packed_super_records.append(
                    pack_cshuffle_super_record(
                        frag_c, frag_row_scale, pending_scale_quarters[-1], CSHUFFLE_N_PAIRS - 1,
                    )
                )
                for output_n_half in range_constexpr(2):
                    for row_pair in range_constexpr(WAVE_M // 16):
                        store_packed_output_quarter(
                            N_TILES - 1, pending_packed_super_records, row_pair, output_n_half,
                        )
                _stage_end()
                if wave_group == 0:
                    _stage_end()

            # 4–5. 执行目录：建立种子 → 首拍/错相 → N循环 → 末N排空。
            prefetch0 = seed_weight_pipeline()
            frag_c.fill(0)
            pending_packed_super_records = None
            pending_scale_quarters = None
            scale_quarters, frag_weight = emit_first_stage(prefetch0)
            packed_super_records = []
            if const_expr(use_n_loop):
                # 6a. 原语适配：地址生成与fragment布局不进入N级调度器。
                from .gemm2_8x1_nloop import emit_nloop

                if const_expr(K in (384, 640)):
                    # 奇数K段使B槽随N翻转；先建立真实lane partition，不能只前置裸槽指针。
                    b_template = weight_lds_quarter_view(weight_storage_ptrs[0], 0)
                    b_copy = down_ops.get_universal_copy_atom(fx.Float8E4M3FNUZ, 128)
                    b_partition = down_ops.get_tiled_mma_partition_S(mm, b_template, "A", copy_atom_bits=128)
                    b_read_pointer = fx.get_iter(b_partition)
                    b_write_pointer = (weight_storage_ptrs[0]
                        + (quarter_atom_index // BLOCK_K) * (16 * BLOCK_K)
                        + ((quarter_atom_index % BLOCK_K) // 16) * 256
                        + (quarter_atom_index % 16) * 16)

                def prepare_b_addresses(n):
                    # 两种相对槽的读/写地址在上一N的compute尾部准备，并跨回边携带。
                    parity = (n * K_STAGES) & 1
                    read_base = fx.Int32(fx.ptrtoint(b_read_pointer))
                    write_base = fx.Int32(fx.ptrtoint(b_write_pointer))
                    values = [base + fx.Int32(((parity + relative_slot) & 1) * BLOCK_N * BLOCK_K)
                              for base in (read_base, write_base) for relative_slot in range_constexpr(2)]
                    return [fx.Int32(llvm.inline_asm(ir.IntegerType.get_signless(32), [_raw(value)],
                                "", "=v,0", has_side_effects=True)) for value in values]

                def loop_issue(n, ks, half, entry):
                    issue_weight_quarter_load(n, ks, half, entry)
                    return weight_staging[entry]

                def loop_commit(slot, ks, half, entry, fragment, address=None):
                    weight_staging[entry].store(fragment.load())
                    if const_expr(address is not None):
                        base = fx.inttoptr(b_write_pointer.type, address)
                        destination = fx.make_view(base + half * (BLOCK_N // 2) * BLOCK_K, fx.make_layout(16, 1))
                        fx.copy(weight_store_atom, weight_staging[entry], destination)
                    else:
                        commit_weight_quarter(slot, half, entry)

                def loop_read(slot, half, ks, quarter, address=None):
                    if const_expr(address is not None):
                        base = fx.inttoptr(b_read_pointer.type, address)
                        source = fx.make_view(base + (2 * half + quarter) * (BLOCK_N // 4) * BLOCK_K, b_partition.layout)
                        fragment = mm.make_fragment_A(b_template)
                        fx.copy(b_copy, source, down_ops.get_tiled_mma_retile(mm, fragment, "A", copy_atom=b_copy))
                        return fragment
                    else:
                        view = weight_lds_quarter_view(
                            weight_storage_ptrs[0] + slot * BLOCK_N * BLOCK_K, 2 * half + quarter,
                        )
                        return down_ops.load_tiled_mma_fragA(mm, view, copy_atom_bits=128)

                def loop_pack(pair, scale):
                    return pack_cshuffle_super_record(frag_c, frag_row_scale, scale, pair)

                def loop_mma(weight, ks, pair):
                    run_super_record_mfma(weight, ks, pair, 0)

                def loop_wait(vmcnt=63, lgkmcnt=63):
                    fx.rocdl.s_waitcnt(_encode_waitcnt(vmcnt=vmcnt, lgkmcnt=lgkmcnt))

                pending_packed_super_records, pending_scale_quarters = emit_nloop(
                    k=K, n_tiles=N_TILES, unroll_n=_n_loop, ptpc=weight_quant_type == "ptpc",
                    c=frag_c, prefetched=weight_staging, first_scales=scale_quarters, ops=down_ops,
                    issue_b=loop_issue, commit_b=loop_commit, read_b=loop_read,
                    load_scale=load_weight_scale_quarter, pack=loop_pack,
                    issue_output=issue_packed_output_quarter, store_output=store_cshuffle_read_results,
                    mma=loop_mma, clear=clear_super_record, schedule_pack=constrain_mfma_valu_packet,
                    enter_memory=_enter_memory_stage, enter_compute=_enter_compute_stage,
                    stage_end=_stage_end, wait=loop_wait,
                    prepare_b_addresses=prepare_b_addresses if const_expr(K in (384, 640)) else None,
                )
            # 6b. 短N/_n_loop=0的实际展开实现，不是可删除的参考路径。
            total_half_cores = 1 if use_n_loop else TOTAL_CORES * 2
            for half_core in range_constexpr(1, total_half_cores):
                block_n = half_core // (K_STAGES * 2)
                stage_in_tile = half_core % (K_STAGES * 2)
                k_stage = stage_in_tile % K_STAGES
                n_half = stage_in_tile // K_STAGES
                current_core = block_n * K_STAGES + k_stage
                slot = current_core & 1
                n_pair_begin = n_half * (CSHUFFLE_N_PAIRS // 2)
                pending = pending_weight_position(half_core)
                pending_n_half = pending[2]
                has_pending = pending[3]

                future_pending = pending_weight_position(half_core + 2)
                has_future_pending = (
                    half_core + 2 < total_half_cores
                    and future_pending[3]
                )
                staging_index = half_core & 1

                if const_expr(stage_in_tile == 0):
                    scale_quarters = []
                    packed_super_records = []

                _enter_memory_stage()
                load_rolling_scales(block_n, stage_in_tile, scale_quarters)

                output_fragments = None
                output_destinations = None
                has_rolling_output = (
                    block_n > 0 and stage_in_tile < CSHUFFLE_N_PAIRS
                )
                if const_expr(has_rolling_output):
                    output_n_half = stage_in_tile // 2
                    output_row_pair = stage_in_tile % 2
                    (
                        output_fragments,
                        output_destinations,
                    ) = issue_packed_output_quarter(
                        block_n - 1,
                        pending_packed_super_records,
                        output_row_pair,
                        output_n_half,
                    )

                frag_weight_quarters = []
                frag_weight_quarters.append(
                    down_ops.load_tiled_mma_fragA(
                        mm,
                        lds_weight_quarters[slot][2 * n_half],
                        copy_atom_bits=128,
                    )
                )

                if const_expr(has_rolling_output):
                    store_cshuffle_read_results(
                        output_fragments,
                        output_destinations,
                        lgkmcnt=4,
                    )
                    fx.rocdl.sched_barrier(0)

                frag_weight_quarters.append(
                    down_ops.load_tiled_mma_fragA(
                        mm,
                        lds_weight_quarters[slot][2 * n_half + 1],
                        copy_atom_bits=128,
                    )
                )

                if const_expr(block_n == 0):
                    frag_weight = down_ops.load_tiled_mma_fragA(
                        mm,
                        lds_weight_halves[slot][n_half],
                        copy_atom_bits=128,
                    )

                if const_expr(has_pending):
                    fx.rocdl.s_waitcnt(
                        _encode_waitcnt(vmcnt=vmem_budgets[half_core])
                    )
                    commit_weight_quarter(
                        pending[4],
                        pending_n_half,
                        staging_index,
                    )

                if const_expr(has_future_pending):
                    fx.rocdl.sched_barrier(0)
                    issue_weight_quarter_load(
                        future_pending[0],
                        future_pending[1],
                        future_pending[2],
                        staging_index,
                    )

                _stage_end()

                _enter_compute_stage()
                for local_n_pair in range_constexpr(
                    CSHUFFLE_N_PAIRS // 2
                ):
                    n_pair = n_pair_begin + local_n_pair
                    if const_expr(k_stage == 0):
                        clear_super_record(n_pair)
                    fx.rocdl.sched_barrier(0)
                    if const_expr(block_n > 0):
                        run_super_record_mfma(
                            frag_weight_quarters[local_n_pair],
                            k_stage,
                            n_pair,
                            0,
                        )
                    else:
                        run_super_record_mfma(
                            frag_weight,
                            k_stage,
                            n_pair,
                            2 * local_n_pair,
                        )
                    # 复用原事件表：跨N仍为3份BF16＋1份FP32，不推迟pack。
                    for packet, previous, record in packing_events(K, stage_in_tile, block_n == 0):
                        if const_expr(local_n_pair == packet):
                            if const_expr(previous):
                                pending_packed_super_records.append(
                                    pack_cshuffle_super_record(
                                        frag_c, frag_row_scale, pending_scale_quarters[record], record,
                                    )
                                )
                            else:
                                packed_super_records.append(
                                    pack_cshuffle_super_record(
                                        frag_c, frag_row_scale, scale_quarters[record], record,
                                    )
                                )
                            constrain_mfma_valu_packet()

                if const_expr(
                    stage_in_tile + 1 == K_STAGES * 2
                ):
                    pending_packed_super_records = (
                        packed_super_records
                    )
                    pending_scale_quarters = scale_quarters

                fx.rocdl.s_waitcnt(_encode_waitcnt(lgkmcnt=0))
                _enter_memory_stage()
                _stage_end()

            # 7. 最后N排空与退出barrier补偿。
            drain_output(pending_packed_super_records, pending_scale_quarters)

    @flyc.jit
    def launch_prefill_8x1(
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
        down_ops.clear_all()
        kernel = moe_2stage_down_prefill_8x1(
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
            value_attrs={
                "passthrough": [["target-features", "-packed-fp32-ops"]]
            },
        )
        kernel.launch(
            grid=(1, task_num, 1),
            block=(512, 1, 1),
            stream=stream,
        )

    launch_prefill_8x1.compile_hints["target_features"] = "-packed-fp32-ops"
    return launch_prefill_8x1
