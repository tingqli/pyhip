# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""K128专用：八个wave按4M×2N分工，两个M32阶段复用N64的B寄存器。"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import fly_rocdl, llvm, rocdl as rocdl_dialect
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import Vector as Vec, as_ir_value
from flydsl.expr.utils.arith import _to_raw as _raw

from . import layout_helpers as fxh
from .common import get_down_device_config


def _build_moe_gemm2_8x1_k128(
    N, TOPK, down_output_padding_bytes, *, weight_quant_type="ptpc", act_quant_type=None,
    _task_table=False,
    _n_loop=1, _store_cache=2, _relax_vmcnt=True,
):
    K, BM, BN, WN = 128, 256, 128, 64
    if act_quant_type is None:
        act_quant_type = weight_quant_type
    assert (weight_quant_type == "ptpc" and act_quant_type == "ptpc") or (
        weight_quant_type == "per_tensor" and act_quant_type in ("ptpc", "per_tensor")
    )
    NT = N // BN
    use_n_loop = bool(_n_loop and NT >= 4)
    from .gemm2_8x1_schedule import k128_vmem_wait_schedule
    vmem_budgets = k128_vmem_wait_schedule(NT, weight_quant_type == "ptpc", _relax_vmcnt)
    STRIDE = N + down_output_padding_bytes // 2
    # B已常驻寄存器，单16KiB LDS槽足够；释放的空间用于两组独立CShuffle。
    RECORDS_PER_ROW = 17
    SCRATCH_PER_WAVE = 16 * RECORDS_PER_ROW * 8
    ops = fxh.FlyObjCache()
    topology, xcc_count = get_down_device_config()
    se_count = xcc_count * 4

    def wait(vmcnt=63, lgkmcnt=63):
        value = (vmcnt & 15) | (7 << 4) | (lgkmcnt << 8) | ((vmcnt >> 4) << 14)
        rocdl.s_waitcnt(value)

    def stage_end():
        rocdl.sched_barrier(0)
        rocdl.s_barrier()
        rocdl.sched_barrier(0)

    def priority(value):
        rocdl.sched_barrier(0)
        rocdl.s_setprio(value)
        rocdl.sched_barrier(0)

    def schedule_packet():
        for index in range_constexpr(16):
            rocdl.sched_group_barrier(0x8, 1, 0)
            if const_expr(weight_quant_type == "ptpc"):
                if const_expr(index < 13):
                    rocdl.sched_group_barrier(0x2, 3, 0)
                elif const_expr(index == 13):
                    rocdl.sched_group_barrier(0x2, 1, 0)
            else:
                rocdl.sched_group_barrier(0x2, 2 if index < 8 else 1, 0)
        rocdl.sched_barrier(0)

    @flyc.kernel(known_block_size=[512, 1, 1])
    def moe_2stage_down_prefill_8x1(
        p_input: fx.Pointer, p_weight: fx.Pointer, p_output: fx.Pointer,
        p_sorted_ids: fx.Pointer, p_sorted_weights: fx.Pointer,
        p_sorted_expert_ids: fx.Pointer, p_num_valid_ids: fx.Pointer,
        p_w_scale: fx.Pointer, p_a_scale: fx.Pointer, M: fx.Int32,
    ):
        tid = fx.Int32(gpu.thread_idx.x)
        lane, group_tid = tid % 64, tid % 256
        wave_m = group_tid // 64
        group = fx.Int32(rocdl.readfirstlane(ir.IntegerType.get_signless(32), _raw(tid // 256)))
        valid = fxh.view_as_torch_tensor(p_num_valid_ids, (1,), fx.Int32)[0]
        wg = fx.Int32(gpu.block_idx.y)
        e_idx = wg
        if const_expr(topology):
            tasks = fxh.div_up(fx.Uint32(valid), BM)
            per_se = tasks // se_count
            mapped = per_se * se_count
            u = fx.Uint32(wg)
            xcc, local = u & (xcc_count - 1), u >> 2
            se, within = local & 3, local >> 2
            cu, round_id = within % 5, within // 5
            short, long = per_se // 5, per_se % 5
            rank = cu * short + arith.select(cu < long, cu, long) + round_id
            logical = ((xcc + 2) & (xcc_count - 1)) * (per_se * 4) + se * per_se + rank
            e_idx = fx.Int32(arith.select(u < mapped, logical, u))

        if e_idx * BM < valid:
            if const_expr(_task_table):
                row_begin = p_sorted_expert_ids[2 * e_idx]
                task_expert = p_sorted_expert_ids[2 * e_idx + 1]
            allocator = fx.SharedAllocator()
            b0 = allocator.allocate(fx.Array[fx.Float8E4M3FNUZ, BN * K, 16])
            scratch = allocator.allocate(fx.Array[fx.BFloat16, 8 * SCRATCH_PER_WAVE, 16])
            scratch_view = scratch.peek().view(fx.make_layout(8 * SCRATCH_PER_WAVE, 1))
            ids_lds = fx.make_view(fx.recast_iter(fx.Int32, scratch.peek().ptr), fx.make_layout(BM, 1))
            ids = fx.rocdl.make_buffer_tensor(
                fxh.view_as_torch_tensor(fxh._as_ptr(p_sorted_ids) + (fx.Int64(row_begin) if const_expr(_task_table) else fx.Int64(e_idx) * BM), (BM,), fx.Int32),
                max_size=False, num_records_bytes=BM * 4,
            )
            if tid < BM:
                ids_lds[tid] = ids[tid]
            gpu.barrier()

            a = fx.rocdl.make_buffer_tensor(
                fxh.view_as_torch_tensor(p_input, (M, TOPK, K), fx.Float8E4M3FNUZ),
                max_size=False, num_records_bytes=fx.Int64(M) * TOPK * K,
            )
            weights = fx.rocdl.make_buffer_tensor(
                fx.make_view(
                    fx.recast_iter(fx.Float8E4M3FNUZ, fxh._as_ptr(p_weight))
                    + fx.Int64(task_expert if const_expr(_task_table) else fxh.view_as_torch_tensor(p_sorted_expert_ids, (1,), fx.Int32)[e_idx]) * N * K,
                    fx.make_layout(N * K, 1),
                ), max_size=False, num_records_bytes=N * K,
            )
            expert = task_expert if const_expr(_task_table) else fxh.view_as_torch_tensor(p_sorted_expert_ids, (1,), fx.Int32)[e_idx]
            weight_rsrc = fly_rocdl.get_buffer_rsrc(_raw(fx.get_iter(weights)), results=[ir.Type.parse("!llvm.ptr<8>")])
            staging = [fx.make_rmem_tensor(fx.make_layout(16, 1), fx.Float8E4M3FNUZ) for _ in range_constexpr(2)]
            bptr = b0.peek().ptr
            bcopy = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Float8E4M3FNUZ)
            lane_offset = group * (WN * K) + group_tid * 16

            def issue_b(n, quarter):
                loaded = Vec(rocdl_dialect.RawPtrBufferLoadOp(
                    ir.VectorType.get([4], ir.IntegerType.get_signless(32)), weight_rsrc,
                    _raw(fx.Int32(lane_offset)), _raw(fx.Int32(n * BN * K + quarter * 32 * K)),
                    aux=ir.IntegerAttr.get(ir.IntegerType.get_signless(32), 0),
                ).result).bitcast(fx.Float8E4M3FNUZ)
                staging[quarter].store(loaded)

            def commit_b(n, quarter):
                destination = fx.make_view(
                    bptr + lane_offset + quarter * 32 * K,
                    fx.make_layout(16, 1),
                )
                fx.copy(bcopy, staging[quarter], destination)

            # 四个M wave共享自己的N64；另一组计算相同M、另一半N。
            mm = ops.create_thr_mma(fx.Float8E4M3FNUZ, (1, 4, 1), tid=group_tid)
            atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, fx.Float8E4M3FNUZ))
            c = mm.make_fragment_C(fx.make_view(fx.get_iter(a), fx.make_ordered_layout((WN, BM), (0, 1))))
            row_weights = fxh.view_as_torch_tensor(fxh._as_ptr(p_sorted_weights) + (fx.Int64(row_begin) if const_expr(_task_table) else fx.Int64(e_idx) * BM), (BM,), fx.Float32)
            row_tensor = fx.make_view(fx.get_iter(row_weights), fx.make_layout((WN, BM), (0, 1)))
            row_scale = ops.load_tiled_mma_fragC(mm, row_tensor, copy_atom_bits=32)
            if const_expr(act_quant_type == "ptpc"):
                coords = ops.load_tiled_mma_fragC(mm, fx.make_view(fx.get_iter(ids_lds), fx.make_layout((WN, BM), (0, 1))), copy_atom_bits=32)
                als = fx.rocdl.make_buffer_tensor(
                    fxh.view_as_torch_tensor(p_a_scale, (M, TOPK), fx.Float32),
                    max_size=False, num_records_bytes=fx.Int64(M) * TOPK * 4,
                )
                scale_copy = ops.get_buffer_copy_atom(fx.Float32, 32)
                ascale = mm.make_fragment_C(row_tensor)
                retile = ops.get_tiled_mma_retile(mm, ascale, "C", copy_atom=scale_copy)
                for dst, coord in fxh.all_elements(retile, coords):
                    encoded = coord[0].bitcast(fx.Uint32)
                    fx.copy(scale_copy, fxh.atom_tensor(als, (encoded & 0xFFFFFF, encoded >> 24), 32), dst)
                row_scale.store(row_scale.load() * ascale.load())
                if const_expr(weight_quant_type == "per_tensor"):
                    ws = fx.make_view(fxh._as_ptr(p_w_scale, fx.Float32) + expert, fx.make_layout(1, 1))[0]
                    row_scale.store(row_scale.load() * ws)
            else:
                scalar_a = fx.make_view(fxh._as_ptr(p_a_scale, fx.Float32), fx.make_layout(1, 1))[0]
                scalar_w = fx.make_view(fxh._as_ptr(p_w_scale, fx.Float32) + expert, fx.make_layout(1, 1))[0]
                row_scale.store(row_scale.load() * (scalar_a * scalar_w))

            issue_b(0, 0)
            issue_b(0, 1)
            acopy = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Float8E4M3FNUZ)
            afrag = mm.make_fragment_B(fx.make_view(fx.get_iter(a), fx.make_layout((BM, K), (1, BM))))
            for rep in range_constexpr(4):
                encoded = ids_lds[wave_m * 16 + rep * 64 + lane % 16].bitcast(fx.Uint32)
                for k64 in range_constexpr(2):
                    source = fxh.atom_tensor(a, (encoded & 0xFFFFFF, encoded >> 24, k64 * 64 + (lane // 16) * 16), 128)
                    packed = fx.make_rmem_tensor(fx.make_layout(16, 1), fx.Float8E4M3FNUZ)
                    fx.copy(acopy, source, packed)
                    values = Vec(packed.load())
                    for ka in range_constexpr(2):
                        afrag[None, rep, (ka, k64)].store(values.shuffle(values, list(range(ka * 8, ka * 8 + 8))))
            wait(vmcnt=0)
            commit_b(0, 0)
            commit_b(0, 1)
            wait(lgkmcnt=0)
            stage_end()

            def read_b(n, quarter):
                view = fx.make_view(
                    bptr + group * (WN * K) + quarter * 32 * K,
                    fx.make_layout(((16, 2), (16, 8)), ((16, 16 * K), (1, 256))),
                )
                return ops.load_tiled_mma_fragA(mm, view, copy_atom_bits=128)

            def load_scales(n):
                if const_expr(weight_quant_type == "ptpc"):
                    view = fx.make_view(
                        fxh._as_ptr(p_w_scale, fx.Float32) + fx.Int64(expert) * N + n * BN + group * WN,
                        fx.make_layout((WN, BM), (1, 0)),
                    )
                    return ops.load_tiled_mma_fragC(mm, view, copy_atom_bits=32)
                else:
                    return fx.Float32(1.0)

            output = fx.rocdl.make_buffer_tensor(
                fx.make_view(fxh._as_ptr(p_output, fx.BFloat16) + (fx.Int64(row_begin) * STRIDE if const_expr(_task_table) else fx.Int64(e_idx) * BM * STRIDE),
                             fx.make_layout((N, BM), (1, STRIDE))),
                max_size=False, num_records_bytes=BM * STRIDE * 2,
            )
            store_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(cache_modifier=_store_cache), fx.BFloat16)
            write_atom = fx.make_copy_atom(fx.UniversalCopy128b(), fx.BFloat16)
            read_atom = fx.make_copy_atom(fx.UniversalCopy64b(), fx.BFloat16)
            scratch_base = (tid // 64) * SCRATCH_PER_WAVE
            output_offsets = []
            for rep in range_constexpr(4):
                output_offsets.append([
                    output.layout(group * WN + (lane % 8) * 8, wave_m * 16 + rep * 64 + half * 8 + lane // 8)
                    for half in range_constexpr(2)
                ])

            def pack_pair(row_begin, n_pair, wscale):
                weighted, rscale = [], []
                for rp in range_constexpr(2):
                    for ng in range_constexpr(2):
                        values = Vec(c[None, 2 * n_pair + ng, row_begin + rp].load())
                        if const_expr(weight_quant_type == "ptpc"):
                            weighted.append(fxh.eltwise_op("v_fma_f32", values, Vec(wscale[None, 2 * n_pair + ng, row_begin + rp].load()), fx.Float32(0.0)))
                        else:
                            weighted.append(values)
                        rscale.append(Vec(row_scale[None, 2 * n_pair + ng, row_begin + rp].load()))
                bias = as_ir_value(fx.Uint32(0x8000)).bitcast(fx.Float32.ir_type)
                scaled = [fxh.eltwise_op("llvm.fma.f32", weighted[i], rscale[i], bias) for i in range_constexpr(4)]
                packed = [[], []]
                for i in range_constexpr(4):
                    for j in range_constexpr(0, 4, 2):
                        packed[i // 2].append(llvm.inline_asm(
                            ir.IntegerType.get_signless(32), [_raw(scaled[i][j + 1]), _raw(scaled[i][j]), _raw(fx.Uint32(0x07060302))],
                            "v_perm_b32 $0, $1, $2, $3", "=v,v,v,s", has_side_effects=True,
                        ))
                return [Vec.from_elements(values, fx.Uint32).bitcast(fx.BFloat16) for values in packed]

            def retire(n, micro, packed_pairs):
                for rp in range_constexpr(2):
                    for pair in range_constexpr(2):
                        record = (4 * (2 * rp + pair) + lane // 16) ^ (2 * (lane % 8))
                        offset = scratch_base + (RECORDS_PER_ROW * (lane % 16) + record) * 8
                        dst = fx.make_view(fx.get_iter(scratch_view) + offset, fx.make_layout(8, 1))
                        fragment = fx.make_fragment_like(dst)
                        fragment.store(packed_pairs[pair][rp])
                        fx.copy(write_atom, fragment, dst)
                fragments, destinations = [], []
                for rp in range_constexpr(2):
                    for half in range_constexpr(2):
                        row = half * 8 + lane // 8
                        atom_n = lane % 8
                        pair = atom_n // 4
                        component = (atom_n // 2) % 2
                        record = (4 * (2 * rp + pair) + (atom_n % 2) * 2) ^ (2 * (row % 8))
                        offset = scratch_base + (RECORDS_PER_ROW * row + record) * 8 + component * 4
                        joined = fx.make_rmem_tensor(fx.make_layout(8, 1), fx.BFloat16)
                        for side in range_constexpr(2):
                            src = fx.make_view(fx.get_iter(scratch_view) + offset + side * 8, fx.make_layout(4, 1))
                            dst = fx.make_view(fx.get_iter(joined) + side * 4, fx.make_layout(4, 1))
                            fx.copy(read_atom, src, dst)
                        fragments.append(joined)
                        destinations.append(fx.make_view(fx.get_iter(output) + output_offsets[2 * micro + rp][half] + n * BN, fx.make_layout(8, 1)))
                return fragments, destinations

            def compute(micro, bfrags, old_scale, q):
                packed_pairs = []
                for pair in range_constexpr(2):
                    for rp in range_constexpr(2):
                        for ng in range_constexpr(2):
                            c[None, 2 * pair + ng, 2 * micro + rp].fill(0)
                    rocdl.sched_barrier(0)
                    for k64 in range_constexpr(2):
                        for ka in range_constexpr(2):
                            for rp in range_constexpr(2):
                                for ng in range_constexpr(2):
                                    dst = c[None, 2 * pair + ng, 2 * micro + rp]
                                    fx.mma_atom_call(atom, dst, bfrags[pair][None, ng, (ka, k64)], afrag[None, 2 * micro + rp, (ka, k64)], dst)
                    if const_expr(q > 0):
                        packed_pairs.append(pack_pair(2 * (1 - micro), pair, old_scale))
                        schedule_packet()
                return packed_pairs

            current_b = [read_b(0, quarter) for quarter in range_constexpr(2)]
            current_scale = load_scales(0)
            wait(vmcnt=0, lgkmcnt=0)
            if const_expr(NT > 1):
                issue_b(1, 0)
                issue_b(1, 1)
            c.fill(0)
            future_b = []
            pending_packed = []
            pending_output = None
            if const_expr(use_n_loop):
                from .gemm2_8x1_k128_nloop import emit_k128_nloop

                def loop_issue(n, quarter):
                    # 静态首尾裁剪；动态中间循环保证n<NT。
                    if const_expr(not isinstance(n, int) or n < NT):
                        issue_b(n, quarter)

                def loop_destinations(n, micro):
                    result = []
                    for rp in range_constexpr(2):
                        for half in range_constexpr(2):
                            result.append(fx.make_view(fx.get_iter(output) + output_offsets[2 * micro + rp][half] + n * BN, fx.make_layout(8, 1)))
                    return result

                def loop_store(fragments, destinations):
                    for index in range_constexpr(4):
                        fx.copy(store_atom, fragments[index], destinations[index])

                def first_stagger():
                    if group == 1:
                        stage_end()

                pending_packed, current_scale, pending_output = emit_k128_nloop(
                    NT, _n_loop, weight_quant_type == "ptpc", _relax_vmcnt,
                    c, staging, current_b, current_scale, ops, loop_issue, commit_b, read_b,
                    load_scales, retire, loop_destinations, loop_store, compute,
                    wait, priority, stage_end, first_stagger,
                )
            for q in range_constexpr(0 if use_n_loop else NT * 2):
                n, micro = q // 2, q % 2
                if const_expr(q > 0 and micro == 0):
                    current_b = future_b
                    future_b = []
                    previous_scale = current_scale
                else:
                    previous_scale = current_scale
                priority(0)
                if const_expr(n + 1 < NT):
                    # 两entry FIFO：另一个quarter请求更新，保留至多一个未完成VMEM。
                    wait(vmcnt=vmem_budgets[q])
                    commit_b(n + 1, micro)
                    rocdl.sched_barrier(0)
                if const_expr(n + 2 < NT):
                    issue_b(n + 2, micro)
                    rocdl.sched_barrier(0)
                if const_expr(q > 0 and micro == 0):
                    current_scale = load_scales(n)
                if const_expr(q >= 3):
                    # 上一memory发起的C读已经跨过32-MFMA compute，结果不需要MOV。
                    for index in range_constexpr(4):
                        fx.copy(store_atom, pending_output[0][index], pending_output[1][index])
                if const_expr(q >= 2):
                    fragments, destinations = retire(n - 1, micro, pending_packed)
                    pending_output = (fragments, destinations)
                # B commit早于4C write+4C read；只保证组内B可见，C读跨compute退休。
                wait(lgkmcnt=8 if q >= 2 else 0)
                stage_end()
                priority(3)
                if const_expr(n + 1 < NT):
                    future_b.append(read_b(n + 1, micro))
                pending_packed = compute(micro, current_b, previous_scale, q)
                wait(lgkmcnt=0)
                priority(0)
                stage_end()
                if const_expr(q == 0):
                    if group == 1:
                        stage_end()

            # 两个最后的M32输出：第一个已经packed，第二个仍在C寄存器。
            if const_expr(NT * 2 >= 3):
                for index in range_constexpr(4):
                    fx.copy(store_atom, pending_output[0][index], pending_output[1][index])
            fragments, destinations = retire(NT - 1, 0, pending_packed)
            wait(lgkmcnt=0)
            for index in range_constexpr(4):
                fx.copy(store_atom, fragments[index], destinations[index])
            wait(vmcnt=0)
            last_packed = [pack_pair(2, pair, current_scale) for pair in range_constexpr(2)]
            fragments, destinations = retire(NT - 1, 1, last_packed)
            wait(lgkmcnt=0)
            for index in range_constexpr(4):
                fx.copy(store_atom, fragments[index], destinations[index])
            stage_end()
            if group == 0:
                stage_end()

    @flyc.jit
    def launch_prefill_8x1(
        p_input: fx.Pointer, p_weight: fx.Pointer, p_output: fx.Pointer,
        p_sorted_ids: fx.Pointer, p_sorted_weights: fx.Pointer,
        p_sorted_expert_ids: fx.Pointer, p_num_valid_ids: fx.Pointer,
        p_w_scale: fx.Pointer, p_a_scale: fx.Pointer,
        M: fx.Int32, task_num: fx.Int32, stream: fx.Stream,
    ):
        CompilationContext.get_current()
        ops.clear_all()
        kernel = moe_2stage_down_prefill_8x1(
            p_input, p_weight, p_output, p_sorted_ids, p_sorted_weights,
            p_sorted_expert_ids, p_num_valid_ids, p_w_scale, p_a_scale, M,
            value_attrs={"passthrough": [["target-features", "-packed-fp32-ops"]]},
        )
        kernel.launch(grid=(1, task_num, 1), block=(512, 1, 1), stream=stream)

    launch_prefill_8x1.compile_hints["target_features"] = "-packed-fp32-ops"
    return launch_prefill_8x1