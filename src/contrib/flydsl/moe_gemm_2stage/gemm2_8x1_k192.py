# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""K192：双槽整块BK192，两拍48-MFMA/wave。"""

from functools import cache
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


def _build_moe_gemm2_8x1_k192(
    N, TOPK, padding, *, weight_quant_type="ptpc", act_quant_type=None,
    _task_table=False,
    _n_loop=1, _store_cache=2, _relax_vmcnt=True,
    _block_k=192,
):
    assert _block_k == 192
    K, BM, BN, MAX_BK = 192, 256, 128, _block_k
    K_WIDTHS = (192,)
    K_OFFSETS = (0,)
    if act_quant_type is None:
        act_quant_type = weight_quant_type
    assert (weight_quant_type == "ptpc" and act_quant_type == "ptpc") or (
        weight_quant_type == "per_tensor" and act_quant_type in ("ptpc", "per_tensor")
    )
    assert N > 0 and N % BN == 0
    assert padding in (0, 32, 64, 128)
    KS, NT = len(K_WIDTHS), N // BN
    STRIDE = N + padding // 2
    ROLLING = os.environ.get("MOE_8X1_ROLLING_EPILOGUE", "1") != "0"
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
        # PTPC40条VALU、per-tensor24条；BK192每packet有24MFMA。
        for index in range_constexpr(24):
            rocdl.sched_group_barrier(0x8, 1, 0)
            rocdl.sched_group_barrier(0x2, 2 if weight_quant_type == "ptpc" and index < 16 else 1, 0)
        rocdl.sched_barrier(0)

    @flyc.kernel(known_block_size=[512, 1, 1])
    def moe_2stage_down_prefill_8x1(
        p_input: fx.Pointer, p_weight: fx.Pointer, p_output: fx.Pointer,
        p_sorted_ids: fx.Pointer, p_sorted_weights: fx.Pointer,
        p_sorted_expert_ids: fx.Pointer, p_num_valid_ids: fx.Pointer,
        p_w_scale: fx.Pointer, p_a_scale: fx.Pointer, M: fx.Int32,
    ):
        tid = fx.Int32(gpu.thread_idx.x)
        # 保留原线程索引表达式的生成顺序，避免清理时引入SSA漂移。
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
            bslots = allocator.allocate(fx.Array[fx.Float8E4M3FNUZ, 2 * BN * MAX_BK, 16])
            bptrs = [bslots.peek().ptr, bslots.peek().ptr + BN * MAX_BK]
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
                # scalar activation乘每专家scalar weight，与1x8的量化契约一致。
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

            def issue_bk192(n, half):
                base = n * BN * K + half * (BN // 2) * K
                return [raw_b_load(tid * 16, base, 4), raw_b_load(8192 + tid * 8, base, 2)]

            # Q0仅包含L/BK192；每线程16B+8B，H半区由Q1单独提交。
            first_b = issue_bk192(0, 0)
            rocdl.sched_barrier(0)

            # A严格gather完整192，不加载或计算不存在的第四个K64。
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

            # partition只建立一次；完整lane地址在上一compute尾部准备，跨N携带。
            b_template = fx.make_view(bptrs[0],
                fx.make_layout(((16, BN // 64), (16, K // 16)), ((16, 16 * K), (1, 256))))
            b_copy = ops.get_universal_copy_atom(fx.Float8E4M3FNUZ, 128)
            b_partition = ops.get_tiled_mma_partition_S(mm, b_template, "A", copy_atom_bits=128)
            b_read_type = fx.get_iter(b_partition).type
            b_write_ptr = fx.recast_iter(fx.Uint32, bptrs[0])
            b_write_types = [b_write_ptr.type, (b_write_ptr + 2).type]

            def prepare_b_addresses(n):
                read_address = fx.Int32(fx.ptrtoint(fx.get_iter(b_partition))) + fx.Int32((n & 1) * BN * K)
                # 读L时写同N的H；读H时写下一N的L。两份完整lane指针在此准备。
                write_base = fx.Int32(fx.ptrtoint(bptrs[0]))
                write_h = write_base + fx.Int32((n & 1) * BN * K + (BN // 2) * K)
                write_l = write_base + fx.Int32(((n + 1) & 1) * BN * K)
                values = [read_address, write_h + tid * 16, write_h + 8192 + tid * 8,
                          write_l + tid * 16, write_l + 8192 + tid * 8]
                # 空tied asm不增加ISA指令，但强制地址在compute尾部sched屏障前就绪。
                return [fx.Int32(llvm.inline_asm(ir.IntegerType.get_signless(32), [_raw(value)],
                    "", "=v,0", has_side_effects=True)) for value in values]

            def commit_bk192(n, half, fragments, addresses):
                for part in range_constexpr(2):
                    words = 4 if part == 0 else 2
                    base = fx.inttoptr(b_write_types[part], addresses[1 + (1 - half) * 2 + part])
                    destination = fx.make_view(base, fx.make_layout(words, 1))
                    fx.copy(ops.get_universal_copy_atom(fx.Uint32, words * 32), fragments[part], destination)

            def read_b(slot, half, ks, quarter, addresses):
                base = fx.inttoptr(b_read_type, addresses[0])
                source = fx.make_view(base + half * (BN // 2) * K + quarter * (BN // 4) * K, b_partition.layout)
                fragment = mm.make_fragment_A(b_template)
                fx.copy(b_copy, source, ops.get_tiled_mma_retile(mm, fragment, "A", copy_atom=b_copy))
                return fragment

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
                    # fx.copy的soffset单位为元素，lowering再乘4转为原生字节偏移。
                    tensor = fx.make_view(fx.get_iter(scale_buffer) + pair * 32, fx.make_layout((32, BM), (1, 0)))
                    # 保持每pair两条dwordx4；copy32不会自动合并，会破坏VMEM等待账本。
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
                    # BF16元素偏移：source-group低位移到2KiB plane，保留128bit写。
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
                    # 让同一输出的两段8B合并为read2st64(offset1:4)，禁止跨oh配对。
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
                    # 线程内输出地址不含N循环变量；统一N偏移由SGPR soffset提供。
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

            wait(vmcnt=0)
            for part in range_constexpr(2):
                words = 4 if part == 0 else 2
                offset = tid * 16 if part == 0 else 8192 + tid * 8
                destination = fx.make_view(
                    fx.recast_iter(fx.Uint32, bptrs[0]) + offset // 4, fx.make_layout(words, 1),
                )
                fx.copy(ops.get_universal_copy_atom(fx.Uint32, words * 32), first_b[part], destination)
            wait(lgkmcnt=0)
            stage_end()
            c.fill(0)
            prefetched = [issue_bk192(0, 1), None]  # P0=Q1=当前H。
            if const_expr(NT > 1):
                prefetched[1] = issue_bk192(1, 0)  # P1=Q2=下一N的L。

            def first_stagger():
                if group == 1:
                    stage_end()

            packed_previous, scales_previous = emit_bk192_nloop(
                NT, _n_loop, weight_quant_type == "ptpc", ROLLING, _relax_vmcnt, c, prefetched, ops,
                issue_bk192, commit_bk192, read_b, load_scale, pack, issue_output, store_output,
                mma, clear, schedule_pack, priority, stage_end, wait, first_stagger, prepare_b_addresses,
            )

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


@cache
def bk192_wait_schedule(n_tiles, ptpc, rolling):
    """消费FIFO的每half-B为16B+8B两条VMEM；同时保护B及跨半区pack的scale。"""
    sequence, requests, scales, budgets = 0, {}, {}, []

    def issue(q):
        nonlocal sequence
        if q < 2 * n_tiles:
            sequence += 2
            requests[q] = sequence

    issue(1)
    issue(2)
    for n in range(n_tiles):
        for half in range(2):
            q = 2 * n + half
            if ptpc:
                for pair in range(2 * half, 2 * half + 2):
                    sequence += 2
                    scales[n, pair] = sequence
            if n > 0:
                sequence += 4
            required = [requests[q + 1]] if q + 1 < 2 * n_tiles else []
            if rolling and ptpc:
                if half == 0 and n > 0:
                    required.extend(scales[n - 1, pair] for pair in (2, 3))
                elif half == 1:
                    required.extend(scales[n, pair] for pair in (0, 1))
            budgets.append(min((sequence - event for event in required), default=63))
            issue(q + 3)
    return tuple(budgets)


@flyc.jit
def emit_bk192_nloop(
    n_tiles, unroll_n, ptpc, rolling, relax_vmcnt, c, prefetched, ops,
    issue_b, commit_b, read_b, load_scale, pack, issue_output, store_output,
    mma, clear, schedule_pack, priority, stage_end, wait, first_stagger, prepare_b_addresses,
):
    budgets = bk192_wait_schedule(n_tiles, ptpc, rolling)
    body_budgets = tuple(min((budgets[2 * n + half] for n in range(2, n_tiles - 2)), default=63)
                         for half in range(2))

    def run_tile(n, carries, previous_packed, previous_scales, first, has_next, has_future, addresses):
        packed, scales = [], []
        for half in range_constexpr(2):
            priority(0)
            for pair in range_constexpr(2 * half, 2 * half + 2):
                scales.append(load_scale(n, pair))
            bf = []
            for quarter in range_constexpr(2):
                if const_expr(not first):
                    fragments, destinations = issue_output(n - 1, previous_packed, quarter, half)
                bf.append(read_b(n & 1, half, 0, quarter, addresses))
                if const_expr(not first):
                    # 本次C读比随后6条B ds_read更早；不等待整个B片段再发store。
                    store_output(fragments, destinations, lgkmcnt=6)
                    fx.rocdl.sched_barrier(0)
            budget = budgets[2 * n + half] if isinstance(n, int) else body_budgets[half]
            if const_expr(budget != 63 or not relax_vmcnt):
                wait(vmcnt=budget if relax_vmcnt else 0)
            if const_expr(half == 0 or has_next):
                commit_b(n + half, 1 - half, carries[half], addresses)
            if const_expr((half == 0 and has_next) or (half == 1 and has_future)):
                fx.rocdl.sched_barrier(0)
                carries[half] = issue_b(n + 1 + half, 1 - half)
            stage_end()
            priority(3)
            for packet in range_constexpr(2):
                pair = half * 2 + packet
                clear(pair)
                fx.rocdl.sched_barrier(0)
                mma(bf[packet], 0, pair)
                if const_expr(rolling):
                    if const_expr(half == 0 and not first):
                        previous_packed.append(pack(2 + packet, previous_scales[packet]))
                        schedule_pack()
                    elif const_expr(half == 1):
                        packed.append(pack(packet, scales[packet]))
                        schedule_pack()
            if const_expr(half == 1 and has_next):
                addresses = prepare_b_addresses(n + 1)
            wait(lgkmcnt=0)
            priority(0)
            stage_end()
            if const_expr(first and half == 0):
                first_stagger()
        if const_expr(not rolling):
            # pure在compute阶段边界之后打包，避免inline-asm FMA过早读取MFMA结果。
            wait(vmcnt=0)
            packed = [pack(pair, scales[pair]) for pair in range_constexpr(4)]
        return carries, packed, scales, addresses

    carries, pending_packed, pending_scales, addresses = run_tile(
        0, prefetched, [], [], True, n_tiles > 1, n_tiles > 2, prepare_b_addresses(0),
    )
    if const_expr(n_tiles > 1):
        # n=1的首个输出过渡只执行一次；其后回边统一使用真正稳态的预算。
        carries, pending_packed, pending_scales, addresses = run_tile(
            1, carries, pending_packed, pending_scales[2:], False, n_tiles > 2, n_tiles > 3, addresses,
        )
    if const_expr(unroll_n == 0 or n_tiles < 4):
        for n in range_constexpr(2, n_tiles):
            carries, pending_packed, pending_scales, addresses = run_tile(
                n, carries, pending_packed, pending_scales[2:], False, n + 1 < n_tiles, n + 2 < n_tiles, addresses,
            )
    else:
        b_carriers = [[fx.make_fragment_like(part) for part in carry] for carry in carries]
        if const_expr(rolling and ptpc):
            scale_carriers = [fx.make_fragment_like(pending_scales[pair]) for pair in range_constexpr(2, 4)]

        def save_state(carries, packed, scales, addresses):
            state = [part.load() for carry in carries for part in carry]
            if const_expr(rolling):
                for row in range_constexpr(2):
                    for group in range_constexpr(4, 8):
                        state.append(c[None, group, row].load())
                if const_expr(ptpc):
                    state.extend(scales[pair].load() for pair in range_constexpr(2, 4))
            for pair in range_constexpr(2 if rolling else 4):
                state.extend(packed[pair][row] for row in range_constexpr(2))
            state.extend(addresses)
            return state

        def restore_state(state):
            for half in range_constexpr(2):
                for part in range_constexpr(2):
                    b_carriers[half][part].store(state[half * 2 + part])
            offset, scales = 4, []
            if const_expr(rolling):
                for row in range_constexpr(2):
                    for group in range_constexpr(4, 8):
                        c[None, group, row].store(state[offset])
                        offset += 1
                for pair in range_constexpr(2):
                    if const_expr(ptpc):
                        scale_carriers[pair].store(state[offset])
                        scales.append(scale_carriers[pair])
                        offset += 1
                    else:
                        scales.append(fx.Float32(1.0))
            packed = []
            for pair in range_constexpr(2 if rolling else 4):
                packed.append([Vec(state[offset]), Vec(state[offset + 1])])
                offset += 2
            return [list(carry) for carry in b_carriers], packed, scales, [fx.Int32(value) for value in state[-5:]]

        initial = save_state(carries, pending_packed, pending_scales, addresses)
        # 剥离最后两N，所有动态迭代的n+2均有效，不发出未消费的越界B请求。
        stop = 2 + ((n_tiles - 4) // unroll_n) * unroll_n
        ops.clear_all()
        for block_start, state in range(2, stop, unroll_n, init=initial):
            carries, previous_packed, previous_scales, addresses = restore_state(state)
            for offset in range_constexpr(unroll_n):
                carries, packed, scales, addresses = run_tile(
                    fx.Int64(block_start) + offset, carries, previous_packed, previous_scales, False, True, True, addresses,
                )
                previous_packed, previous_scales = packed, scales[2:]
            results = yield save_state(carries, packed, scales, addresses)
        ops.clear_all()
        carries, pending_packed, previous_scales, addresses = restore_state(results)
        for n in range_constexpr(stop, n_tiles):
            carries, pending_packed, pending_scales, addresses = run_tile(
                n, carries, pending_packed, previous_scales, False, n + 1 < n_tiles, n + 2 < n_tiles, addresses,
            )
            previous_scales = pending_scales[2:]
    return pending_packed, pending_scales