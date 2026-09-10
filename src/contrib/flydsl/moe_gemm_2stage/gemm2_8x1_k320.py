# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""K320：128+192四阶段；不padding或增加MFMA。"""

import os

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


def _build_moe_gemm2_8x1_k320(
    N, TOPK, padding, *, weight_quant_type="ptpc", act_quant_type=None,
    _task_table=False,
    _n_loop=1, _store_cache=2, _relax_vmcnt=True,
    _block_k=192,
):
    assert _block_k == 192
    K, BM, BN = 320, 256, 128
    K_WIDTHS = (128, 192)
    K_OFFSETS = (0, 128)
    if act_quant_type is None:
        act_quant_type = weight_quant_type
    assert (weight_quant_type == "ptpc" and act_quant_type == "ptpc") or (
        weight_quant_type == "per_tensor" and act_quant_type in ("ptpc", "per_tensor")
    )
    assert N > 0 and N % BN == 0
    assert padding in (0, 32, 64, 128)
    KS, NT = len(K_WIDTHS), N // BN
    use_n_loop = bool(_n_loop and NT >= 3)
    STRIDE = N + padding // 2
    ROLLING = os.environ.get("MOE_8X1_ROLLING_EPILOGUE", "1") != "0"
    from .gemm2_8x1_schedule import vmem_wait_schedule
    vmem_budgets = vmem_wait_schedule(
        K, NT, weight_quant_type == "ptpc", ROLLING, _relax_vmcnt,
        K_WIDTHS,
    )
    ops = fxh.FlyObjCache()
    topology, xcc_count = get_down_device_config()
    se_count = xcc_count * 4

    def wait(vmcnt=63, lgkmcnt=63):
        rocdl.s_waitcnt((vmcnt & 15) | (7 << 4) | (lgkmcnt << 8) | ((vmcnt >> 4) << 14))

    def stage_end():
        rocdl.sched_barrier(0)
        rocdl.s_barrier()
        rocdl.sched_barrier(0)

    def priority(value):
        rocdl.sched_barrier(0)
        rocdl.s_setprio(value)
        rocdl.sched_barrier(0)

    def schedule_pack():
        # PTPC为40条VALU，标量scale融合后为24条；均只放进BK128长packet。
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
        lane, wave, group_tid = tid % 64, tid // 64, tid % 256
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
            allocator = fx.SharedAllocator()
            # KS=2时slot=(n*2+ks)&1=ks，固定16/24KiB非对称槽，总LDS56KiB。
            b0 = allocator.allocate(fx.Array[fx.Float8E4M3FNUZ, BN * K_WIDTHS[0], 16])
            b1 = allocator.allocate(fx.Array[fx.Float8E4M3FNUZ, BN * K_WIDTHS[1], 16])
            bptrs = [b0.peek().ptr, b1.peek().ptr]
            scratch = allocator.allocate(fx.Array[fx.BFloat16, 4 * 16 * BN, 16])
            scratch_view = scratch.peek().view(fx.make_layout(4 * 16 * BN, 1))
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
            expert = p_sorted_expert_ids[2 * e_idx + 1] if const_expr(_task_table) else fxh.view_as_torch_tensor(p_sorted_expert_ids, (1,), fx.Int32)[e_idx]
            weights = fx.rocdl.make_buffer_tensor(
                fx.make_view(
                    fx.recast_iter(fx.Float8E4M3FNUZ, fxh._as_ptr(p_weight)) + fx.Int64(expert) * N * K,
                    fx.make_layout(N * K, 1),
                ), max_size=False, num_records_bytes=N * K,
            )
            rsrc = fly_rocdl.get_buffer_rsrc(_raw(fx.get_iter(weights)), results=[ir.Type.parse("!llvm.ptr<8>")])
            mm = ops.create_thr_mma(fx.Float8E4M3FNUZ, (1, 8, 1))
            atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, fx.Float8E4M3FNUZ))
            c = mm.make_fragment_C(fx.make_view(fx.get_iter(a), fx.make_ordered_layout((BN, BM), (0, 1))))
            routing_scale = fxh.view_as_torch_tensor(fxh._as_ptr(p_sorted_weights) + (fx.Int64(row_begin) if const_expr(_task_table) else fx.Int64(e_idx) * BM), (BM,), fx.Float32)
            row_tensor = fx.make_view(fx.get_iter(routing_scale), fx.make_layout((BN, BM), (0, 1)))
            row_scale = ops.load_tiled_mma_fragC(mm, row_tensor, copy_atom_bits=32)
            if const_expr(act_quant_type == "ptpc"):
                coords = ops.load_tiled_mma_fragC(
                    mm, fx.make_view(fx.get_iter(ids_lds), fx.make_layout((BN, BM), (0, 1))), copy_atom_bits=32,
                )
                als = fx.rocdl.make_buffer_tensor(
                    fxh.view_as_torch_tensor(p_a_scale, (M, TOPK), fx.Float32),
                    max_size=False, num_records_bytes=fx.Int64(M) * TOPK * 4,
                )
                scale_copy = ops.get_buffer_copy_atom(fx.Float32, 32)
                ascale = mm.make_fragment_C(row_tensor)
                for dst, coord in fxh.all_elements(ops.get_tiled_mma_retile(mm, ascale, "C", copy_atom=scale_copy), coords):
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

            def raw_b_load(byte_offset, scalar_offset, words):
                values = Vec(rocdl_dialect.RawPtrBufferLoadOp(
                    ir.VectorType.get([words], ir.IntegerType.get_signless(32)), rsrc,
                    _raw(fx.Int32(byte_offset)), _raw(fx.Int32(scalar_offset)),
                    aux=ir.IntegerAttr.get(ir.IntegerType.get_signless(32), 0),
                ).result).bitcast(fx.Uint32)
                fragment = fx.make_rmem_tensor(fx.make_layout(words, 1), fx.Uint32)
                fragment.store(values)
                return fragment

            # 仅预填Q0=L/K128；每线程16B，后续用同一消费FIFO。
            first_index = group * 256 + group_tid
            first_offset = ((first_index // K_WIDTHS[0]) * (16 * K)
                            + ((first_index % K_WIDTHS[0]) // 16) * 256 + (first_index % 16) * 16)
            first_b = raw_b_load(first_offset, 0, 4)
            rocdl.sched_barrier(0)

            # A严格按K_WIDTHS gather，128/192的总量为320。
            afragments = []
            acopy = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Float8E4M3FNUZ)
            for ks in range_constexpr(KS):
                width = K_WIDTHS[ks]
                af = mm.make_fragment_B(fx.make_view(fx.get_iter(a), fx.make_layout((BM, width), (1, BM))))
                for row in range_constexpr(2):
                    encoded = ids_lds[wave * 16 + row * 128 + lane % 16].bitcast(fx.Uint32)
                    for k64 in range_constexpr(width // 64):
                        offset = K_OFFSETS[ks] + k64 * 64 + (lane // 16) * 16
                        packed = fx.make_rmem_tensor(fx.make_layout(16, 1), fx.Float8E4M3FNUZ)
                        fx.copy(acopy, fxh.atom_tensor(a, (encoded & 0xFFFFFF, encoded >> 24, offset), 128), packed)
                        values = Vec(packed.load())
                        for k8 in range_constexpr(2):
                            part = values.shuffle(values, list(range(k8 * 8, k8 * 8 + 8)))
                            af[None, row, (k8, k64)].store(part)
                afragments.append(af)

            def b_position(q):
                target = q + 1
                n, ks = target // (2 * KS), target % KS
                return n, ks, (target % (2 * KS)) // KS, target < NT * KS * 2, (n * KS + ks) & 1

            def b_offsets(ks):
                width = K_WIDTHS[ks]
                index = group * (BN * width // 64) + group_tid
                extra = 0
                global_offset = (index // width) * (16 * K) + ((index % width) // 16) * 256 + (index % 16) * 16 + extra
                local_offset = (index // width) * (16 * width) + ((index % width) // 16) * 256 + (index % 16) * 16 + extra
                return global_offset, local_offset

            def tail192_b_offsets(part):
                # 连续LDS字节映回全K320 preshuffle；每16行只取K[128:320]。
                local_offset = tid * 16 if part == 0 else 8192 + tid * 8
                global_offset = (local_offset // (16 * 192)) * (16 * K) + local_offset % (16 * 192)
                return global_offset, local_offset

            def issue_b(position):
                n, ks, half = position[0], position[1], position[2]
                if const_expr(K_WIDTHS[ks] == 192):
                    scalar = n * BN * K + K_OFFSETS[ks] * 16 + half * (BN // 2) * K
                    pieces = [raw_b_load(tail192_b_offsets(part)[0], scalar, 4 if part == 0 else 2)
                              for part in range_constexpr(2)]
                    # 6个真实u32作为单一carry，重用通用Nloop；拼接不产生额外K加载/MFMA。
                    fragment = fx.make_rmem_tensor(fx.make_layout(6, 1), fx.Uint32)
                    fragment.store(Vec.from_elements(
                        [Vec(pieces[0].load())[i] for i in range_constexpr(4)]
                        + [Vec(pieces[1].load())[i] for i in range_constexpr(2)], fx.Uint32,
                    ))
                    return fragment
                # 每组256线程：BK128各16B；没有条件exec或冗余全宽加载。
                return raw_b_load(b_offsets(ks)[0], n * BN * K + K_OFFSETS[ks] * 16 + half * (BN // 2) * K, K_WIDTHS[ks] // 32)

            def commit_b(position, fragment):
                ks, half, slot = position[1], position[2], position[4]
                width = K_WIDTHS[ks]
                base = fx.recast_iter(fx.Uint32, bptrs[ks]) + half * (BN // 2) * (width // 4)
                if const_expr(width == 192):
                    values = Vec(fragment.load())
                    for part in range_constexpr(2):
                        words, start = (4, 0) if part == 0 else (2, 4)
                        piece = fx.make_rmem_tensor(fx.make_layout(words, 1), fx.Uint32)
                        piece.store(values.shuffle(values, list(range(start, start + words))))
                        destination = fx.make_view(base + tail192_b_offsets(part)[1] // 4, fx.make_layout(words, 1))
                        fx.copy(ops.get_universal_copy_atom(fx.Uint32, words * 32), piece, destination)
                else:
                    destination = fx.make_view(base + b_offsets(ks)[1] // 4, fx.make_layout(4, 1))
                    fx.copy(ops.get_universal_copy_atom(fx.Uint32, 128), fragment, destination)

            def read_b(slot, half, ks, quarter):
                width = K_WIDTHS[ks]
                # 固定槽分别是BK128/BK192，half间距分别8192/12288B。
                view = fx.make_view(
                    bptrs[ks] + half * (BN // 2) * width + quarter * (BN // 4) * width,
                    fx.make_layout(((16, BN // 64), (16, width // 16)), ((16, 16 * width), (1, 256))),
                )
                return ops.load_tiled_mma_fragA(mm, view, copy_atom_bits=128)

            out = fx.rocdl.make_buffer_tensor(
                fx.make_view(fxh._as_ptr(p_output, fx.BFloat16) + (fx.Int64(row_begin) * STRIDE if const_expr(_task_table) else fx.Int64(e_idx) * BM * STRIDE),
                             fx.make_layout((N, BM), (1, STRIDE))),
                max_size=False, num_records_bytes=BM * STRIDE * 2,
            )
            store_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(cache_modifier=_store_cache), fx.BFloat16)
            scratch_write = ops.get_universal_copy_atom(fx.BFloat16, 128)
            scratch_read = ops.get_universal_copy_atom(fx.BFloat16, 64)
            scratch_base = (wave % 4) * 16 * BN
            lane_group, row8, row_half = lane // 16, (lane % 16) % 8, (lane % 16) // 8

            if const_expr(weight_quant_type == "ptpc"):
                scale_buffer = fx.rocdl.make_buffer_tensor(
                    fxh.view_as_torch_tensor(fxh._as_ptr(p_w_scale, fx.Float32) + fx.Int64(expert) * N, (N,), fx.Float32),
                    max_size=False, num_records_bytes=N * 4,
                )

            def load_scale(n, pair):
                if const_expr(weight_quant_type == "ptpc"):
                    # fx.copy的soffset按元素计量；uniform N偏移由SGPR提供。
                    tensor = fx.make_view(fx.get_iter(scale_buffer) + pair * 32, fx.make_layout((32, BM), (1, 0)))
                    # 每pair仍两条dwordx4，保持原VMEM请求数和等待账本。
                    copy_atom = ops.get_buffer_copy_atom(fx.Float32, 128)
                    fragment = mm.make_fragment_C(tensor)
                    fx.copy(copy_atom, ops.get_tiled_mma_partition_S(mm, tensor, "C", copy_atom_bits=128),
                            ops.get_tiled_mma_retile(mm, fragment, "C", copy_atom=copy_atom),
                            soffset=fx.Int32(n * BN))
                    return fragment
                else:
                    return fx.Float32(1.0)

            def pack(pair, scale):
                weighted, rows = [], []
                for row in range_constexpr(2):
                    for ng in range_constexpr(2 * pair, 2 * pair + 2):
                        if const_expr(weight_quant_type == "ptpc"):
                            weighted.append(fxh.eltwise_op("v_fma_f32", Vec(c[None, ng, row].load()),
                                                          Vec(scale[None, ng % 2, row].load()), fx.Float32(0.0)))
                        else:
                            weighted.append(Vec(c[None, ng, row].load()))
                        rows.append(Vec(row_scale[None, ng, row].load()))
                bias = as_ir_value(fx.Uint32(0x8000)).bitcast(fx.Float32.ir_type)
                scaled = [fxh.eltwise_op("llvm.fma.f32", weighted[index], rows[index], bias) for index in range_constexpr(4)]
                selector = fx.Uint32(0x07060302)
                records = [[], []]
                for index in range_constexpr(4):
                    for element in range_constexpr(0, scaled[index].numel, 2):
                        records[index // 2].append(llvm.inline_asm(
                            ir.IntegerType.get_signless(32),
                            [_raw(scaled[index][element + 1]), _raw(scaled[index][element]), _raw(selector)],
                            "v_perm_b32 $0, $1, $2, $3", "=v,v,v,s", has_side_effects=True,
                        ))
                return [Vec.from_elements(record, fx.Uint32).bitcast(fx.BFloat16) for record in records]

            def issue_output(n, packed, row, half):
                def plane_offset(r, g, p):
                    # BF16元素偏移：两段8B分到相距2KiB的plane，producer仍128bit写。
                    return ((r & 1) * 8 + ((g & 2) ^ (r & 2)) * 8
                            + ((p & 1) ^ ((r >> 2) & 1)) * 32 + (p >> 1) * 64
                            + (r >> 3) * 128 + (r & 6) * 128 + (g & 1) * 1024)
                for local_pair in range_constexpr(2):
                    pair = half * 2 + local_pair
                    offset = scratch_base + plane_offset(lane % 16, lane_group, pair)
                    destination = fx.make_view(fx.get_iter(scratch_view) + offset, fx.make_layout(8, 1))
                    fragment = fx.make_fragment_like(destination)
                    fragment.store(packed[pair][row])
                    fx.copy(scratch_write, fragment, destination)
                fragments, destinations = [], []
                for oh in range_constexpr(2):
                    atom_index = half * 8 + lane % 8
                    ng = atom_index // 2
                    offset = scratch_base + plane_offset(oh * 8 + lane // 8, 2 * (atom_index % 2), ng // 2) + (ng % 2) * 4
                    pieces = []
                    for source_group in range_constexpr(2):
                        source = fx.make_view(fx.get_iter(scratch_view) + offset + source_group * 1024, fx.make_layout(4, 1))
                        fragment = fx.make_fragment_like(source)
                        fx.copy(scratch_read, source, fragment)
                        pieces.append(fragment)
                    # 合并同一输出的两次64bit读，禁止跨oh配对产生额外搬运。
                    fx.rocdl.sched_barrier(0)
                    fragments.append(pieces)
                    out_row = wave * 16 + row * 128 + oh * 8 + lane // 8
                    destinations.append((n, fx.make_view(fx.get_iter(out) + out.layout(atom_index * 8, out_row),
                                                        fx.make_layout(8, 1))))
                return fragments, destinations

            def store_output(fragments, destinations, lgkmcnt=0):
                wait(lgkmcnt=lgkmcnt)
                for index in range_constexpr(len(fragments)):
                    first, second = Vec(fragments[index][0].load()), Vec(fragments[index][1].load())
                    result = fx.make_rmem_tensor(fx.make_layout(8, 1), fx.BFloat16)
                    result.store(first.shuffle(second, list(range(8))))
                    output_n, destination = destinations[index]
                    fx.copy(store_atom, result, destination, soffset=fx.Int32(output_n * BN))

            def retire(n, packed):
                for half in range_constexpr(2):
                    for row in range_constexpr(2):
                        fragments, destinations = issue_output(n, packed, row, half)
                        store_output(fragments, destinations)

            def clear(pair):
                for row in range_constexpr(2):
                    for ng in range_constexpr(pair * 2, pair * 2 + 2):
                        c[None, ng, row].fill(0)

            def mma(weight, ks, pair):
                for ki in range_constexpr(K_WIDTHS[ks] // 64):
                    for ka in range_constexpr(2):
                        for row in range_constexpr(2):
                            for ng in range_constexpr(2):
                                weight_piece = weight[None, ng, (ka, ki)]
                                activation = afragments[ks][None, row, (ka, ki)]
                                fx.mma_atom_call(atom, c[None, pair * 2 + ng, row], weight_piece, activation, c[None, pair * 2 + ng, row])

            wait(vmcnt=4)
            commit_b((0, 0, 0, True, 0), first_b)
            rocdl.sched_barrier(0)
            prefetched = [issue_b(b_position(0)), None]
            if const_expr(b_position(1)[3]):
                prefetched[1] = issue_b(b_position(1))
            wait(vmcnt=1 if b_position(1)[3] else 0)
            stage_end()
            c.fill(0)

            # q0剥离，额外一拍barrier令两组4-wave交错memory/compute。
            priority(0)
            scales = [load_scale(0, 0)] if ROLLING else []
            first_b = [read_b(0, 0, 0, quarter) for quarter in range_constexpr(2)]
            wait(vmcnt=vmem_budgets[0])
            commit_b(b_position(0), prefetched[0])
            if const_expr(b_position(2)[3]):
                rocdl.sched_barrier(0)
                prefetched[0] = issue_b(b_position(2))
            wait(lgkmcnt=0)
            stage_end()
            priority(3)
            for pair in range_constexpr(2):
                rocdl.sched_barrier(0)
                mma(first_b[pair], 0, pair)
            priority(0)
            stage_end()
            if group == 1:
                stage_end()

            packed_current, packed_previous, scales_previous = [], [], []
            if const_expr(use_n_loop):
                from .gemm2_8x1_nloop import emit_nloop

                def loop_issue(n, ks, half, entry):
                    return issue_b((n, ks, half))

                def loop_commit(slot, ks, half, entry, fragment):
                    commit_b((0, ks, half, True, slot), fragment)

                def memory():
                    priority(0)

                def compute():
                    priority(3)

                packed_previous, scales_previous = emit_nloop(
                    K, NT, _n_loop, weight_quant_type == "ptpc", ROLLING, _relax_vmcnt,
                    c, prefetched, scales, ops, loop_issue, loop_commit, read_b, load_scale, pack,
                    issue_output, store_output, mma, clear, schedule_pack, memory, compute, stage_end, wait,
                    k_widths=K_WIDTHS,
                )
            for q in range_constexpr(1, 1 if use_n_loop else NT * KS * 2):
                n, stage = q // (2 * KS), q % (2 * KS)
                ks, half = stage % KS, stage // KS
                slot, entry = (n * KS + ks) & 1, q & 1
                pending, future = b_position(q), b_position(q + 2)
                has_future = q + 2 < NT * KS * 2 and future[3]
                if const_expr(stage == 0):
                    scales, packed_current = [], []
                priority(0)
                if const_expr(ROLLING):
                    scales.append(load_scale(n, stage))
                has_output = ROLLING and n > 0
                if const_expr(has_output):
                    output_quarter = stage
                    output_fragments, output_destinations = issue_output(n - 1, packed_previous, output_quarter % 2, output_quarter // 2)
                elif const_expr(not ROLLING and n > 0 and stage == 0):
                    retire(n - 1, packed_previous)

                bf = [read_b(slot, half, ks, 0)]
                if const_expr(has_output):
                    store_output(output_fragments, output_destinations, lgkmcnt=K_WIDTHS[ks] // 32)
                    rocdl.sched_barrier(0)
                bf.append(read_b(slot, half, ks, 1))
                if const_expr(pending[3]):
                    wait(vmcnt=vmem_budgets[q])
                    commit_b(pending, prefetched[entry])
                elif const_expr(ROLLING and weight_quant_type == "ptpc" and ks == 0):
                    # 末N即便没有下一B提交，跨SR打包仍必须保护scale消费者。
                    wait(vmcnt=vmem_budgets[q])
                if const_expr(has_future):
                    rocdl.sched_barrier(0)
                    prefetched[entry] = issue_b(future)
                if const_expr(q == 1):
                    # 与共享循环相同：首次H/K128提交后再放行领先wave组。
                    wait(lgkmcnt=0)
                stage_end()

                priority(3)
                for local_pair in range_constexpr(2):
                    pair = half * 2 + local_pair
                    if const_expr(ks == 0):
                        clear(pair)
                    rocdl.sched_barrier(0)
                    mma(bf[local_pair], ks, pair)
                    if const_expr(ROLLING and ks == 0):
                        if const_expr(n > 0 and half == 0):
                            retired_pair = local_pair + 2
                            packed_previous.append(pack(retired_pair, scales_previous[retired_pair]))
                            schedule_pack()
                        elif const_expr(half == 1):
                            packed_current.append(pack(local_pair, scales[local_pair]))
                            schedule_pack()
                if const_expr(stage == 2 * KS - 1):
                    packed_previous, scales_previous = packed_current, scales
                wait(lgkmcnt=0)
                priority(0)
                stage_end()
                if const_expr(not ROLLING and stage == 2 * KS - 1):
                    # pure分支在阶段边界之后打包，避免inline-asm紧邻末MFMA。
                    scales = [load_scale(n, pair) for pair in range_constexpr(4)]
                    wait(vmcnt=0)
                    packed_previous = [pack(pair, scales[pair]) for pair in range_constexpr(4)]
                    scales_previous = scales

            if const_expr(ROLLING):
                wait(vmcnt=0)
                for pair in range_constexpr(2, 4):
                    packed_previous.append(pack(pair, scales_previous[pair]))
            retire(NT - 1, packed_previous)
            stage_end()
            if group == 0:
                stage_end()

    @flyc.jit
    def launch_prefill_8x1(
        p_input: fx.Pointer, p_weight: fx.Pointer, p_output: fx.Pointer,
        p_sorted_ids: fx.Pointer, p_sorted_weights: fx.Pointer,
        p_sorted_expert_ids: fx.Pointer, p_num_valid_ids: fx.Pointer,
        p_w_scale: fx.Pointer, p_a_scale: fx.Pointer, M: fx.Int32, task_num: fx.Int32, stream: fx.Stream,
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