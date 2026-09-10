# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""共享8x1流水：编译期计划 → 单拍Memory/Compute → SSA状态 → N级编排。

K256/320/384/512/640共用事件顺序，K192保留独立两拍循环。
原语由builder绑定；本模块只决定何时发出，不计算lane地址或实现MFMA/pack。
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import Vector as Vec

from .gemm2_8x1_schedule import output_quarter, packing_events, vmem_wait_schedule


@flyc.jit
def emit_nloop(
    k, n_tiles, unroll_n, ptpc, c, prefetched,
    first_scales, ops, issue_b, commit_b, read_b, load_scale, pack,
    issue_output, store_output, mma, clear, schedule_pack,
    enter_memory, enter_compute, stage_end, wait,
    k_widths=None, prepare_b_addresses=None,
):
    # 1. 编译期计划。不能把剥离N的Python整数统一转成动态SSA值。
    assert k != 192, "K192 uses emit_bk192_nloop"
    assert k in (256, 320, 384, 512, 640), "shared 8x1 Nloop仅支持K=256/320/384/512/640"
    assert k != 320 or k_widths == (128, 192)
    assert prepare_b_addresses is None or k in (384, 640)
    widths = k_widths or tuple(min(128, k - 128 * index) for index in range((k + 127) // 128))
    ks = len(widths)
    budgets = vmem_wait_schedule(k, n_tiles, ptpc, k_widths)
    first_unpacked = 2 if k == 320 else 3

    def position(n, stage, delta=0):
        # 所有K按消费者顺序提交Q[q+1]；delta=2预取Q[q+3]。
        target = stage + delta + 1
        target_n, target_k = n + target // (2 * ks), target % ks
        return target_n, target_k, (target % (2 * ks)) // ks, (target_n * ks + target_k) & 1

    def valid(stage, delta, last):
        return not last or stage + delta + 1 < 2 * ks

    # 2. 单拍执行顺序。B消费Q[q]、提交Q[q+1]、预取Q[q+3]；pack不是store。
    def memory_stage(n, stage, carries, previous_packed, scales, first, last, addresses):
        k_stage, half, entry = stage % ks, stage // ks, stage & 1
        slot = (n * ks + k_stage) & 1
        output = output_quarter(k, stage, k_widths)
        has_output = not first and output is not None
        enter_memory()
        if const_expr(stage < 4):
            scales.append(load_scale(n, stage))
        if const_expr(has_output):
            fragments, destinations = issue_output(n - 1, previous_packed, output % 2, output // 2)
        if const_expr(prepare_b_addresses is not None):
            bf = [read_b(slot, half, k_stage, 0, addresses[k_stage & 1])]
        else:
            bf = [read_b(slot, half, k_stage, 0)]
        if const_expr(has_output):
            store_output(fragments, destinations, lgkmcnt=widths[k_stage] // 32)
            fx.rocdl.sched_barrier(0)
        if const_expr(prepare_b_addresses is not None):
            bf.append(read_b(slot, half, k_stage, 1, addresses[k_stage & 1]))
        else:
            bf.append(read_b(slot, half, k_stage, 1))
        # n=1已剥离，回边使用n>=2预算；同时保护B及本拍pack的scale。
        budget_n = 0 if first else n_tiles - 1 if last else n if isinstance(n, int) else 2
        if const_expr(valid(stage, 0, last) or (ptpc and packing_events(k, stage, first, k_widths))):
            wait(vmcnt=budgets[budget_n * 2 * ks + stage])
        if const_expr(valid(stage, 0, last)):
            target_n, target_k, target_half, target_slot = position(n, stage)
            if const_expr(prepare_b_addresses is not None):
                # L末K切换到同N的H/K0；奇数KS不能沿用“当前slot+1”。
                relative_slot = (target_k + ((stage + 1) // (2 * ks)) * ks) & 1
                commit_b(target_slot, target_k, target_half, entry, carries[entry], addresses[2 + relative_slot])
            else:
                commit_b(target_slot, target_k, target_half, entry, carries[entry])
        if const_expr(valid(stage, 2, last)):
            future_n, future_k, future_half, _ = position(n, stage, 2)
            fx.rocdl.sched_barrier(0)
            carries[entry] = issue_b(future_n, future_k, future_half, entry)
        if const_expr(k == 320 and first and stage == 1):
            # 首次H/K128交接必须在barrier前完成本组LDS写。
            wait(lgkmcnt=0)
        stage_end()
        return bf

    def compute_stage(n, stage, bf, previous_packed, previous_scales, packed, scales, first, last, addresses):
        k_stage, half = stage % ks, stage // ks
        enter_compute()
        for packet in range_constexpr(2):
            pair = 2 * half + packet
            if const_expr(k_stage == 0):
                clear(pair)
            fx.rocdl.sched_barrier(0)
            mma(bf[packet], k_stage, pair)
            for packed_packet, previous, record in packing_events(k, stage, first, k_widths):
                if const_expr(packet == packed_packet):
                    if const_expr(previous):
                        previous_packed.append(pack(record, previous_scales[record - first_unpacked]))
                    else:
                        packed.append(pack(record, scales[record]))
                    schedule_pack()
        if const_expr(prepare_b_addresses is not None and stage == 2 * ks - 1 and not last):
            addresses = prepare_b_addresses(n + 1)
        wait(lgkmcnt=0)
        enter_memory()
        stage_end()
        return addresses

    def run_tile(n, carries, previous_packed, previous_scales, scales, first, last, addresses):
        packed = []
        for stage in range_constexpr(1 if first else 0, 2 * ks):
            bf = memory_stage(n, stage, carries, previous_packed, scales, first, last, addresses)
            addresses = compute_stage(n, stage, bf, previous_packed, previous_scales, packed, scales, first, last, addresses)
        return carries, packed, scales, addresses

    # 3. 回边状态编解码。严格保留扁平SSA顺序，不携带当前bf或完整scale/C副本。
    # B搬运片段 → 未pack的FP32 C → 待用scale → 已pack的BF16 → 地址。
    def save_state(carries, packed, scales, addresses):
        state = [carries[index].load() for index in range_constexpr(2)]
        for row in range_constexpr(2):
            for group in range_constexpr(2 * first_unpacked, 8):
                state.append(c[None, group, row].load())
        if const_expr(ptpc):
            for pair in range_constexpr(first_unpacked, 4):
                state.append(scales[pair].load())
        for pair in range_constexpr(first_unpacked):
            for row in range_constexpr(2):
                state.append(packed[pair][row])
        if const_expr(prepare_b_addresses is not None):
            state.extend(addresses)
        return state

    def restore_state(state):
        for index in range_constexpr(2):
            b_carriers[index].store(state[index])
        offset = 2
        scales = []
        for row in range_constexpr(2):
            for group in range_constexpr(2 * first_unpacked, 8):
                c[None, group, row].store(state[offset])
                offset += 1
        for pair in range_constexpr(first_unpacked, 4):
            if const_expr(ptpc):
                scale_carriers[pair - first_unpacked].store(state[offset])
                offset += 1
                scales.append(scale_carriers[pair - first_unpacked])
            else:
                scales.append(fx.Float32(1.0))
        packed = []
        for pair in range_constexpr(first_unpacked):
            packed.append([Vec(state[offset]), Vec(state[offset + 1])])
            offset += 2
        addresses = [fx.Int32(value) for value in state[-4:]] if const_expr(prepare_b_addresses is not None) else []
        return list(b_carriers), packed, scales, addresses

    # 4. N级编排：N0余拍 → N1首次回写 → 稳态回边 → 展开余数 → 最后N。
    addresses = prepare_b_addresses(0) if const_expr(prepare_b_addresses is not None) else []
    carries, pending_packed, pending_scales, addresses = run_tile(0, prefetched, [], [], first_scales, True, False, addresses)
    carries, pending_packed, pending_scales, addresses = run_tile(
        1, carries, pending_packed, pending_scales[first_unpacked:], [], False, False, addresses,
    )
    b_carriers = [fx.make_fragment_like(carries[index]) for index in range_constexpr(2)]
    if const_expr(ptpc):
        scale_carriers = [fx.make_fragment_like(pending_scales[pair])
                          for pair in range_constexpr(first_unpacked, 4)]

    initial = save_state(carries, pending_packed, pending_scales, addresses)
    # Q[q+3]最多进入下一N，所有共享K都只需独立处理最后N。
    loop_start = 2
    stop = loop_start + (max(0, n_tiles - loop_start - 1) // unroll_n) * unroll_n
    ops.clear_all()
    for block_start, state in range(loop_start, stop, unroll_n, init=initial):
        carries, previous_packed, previous_scales, addresses = restore_state(state)
        for offset in range_constexpr(unroll_n):
            carries, packed, scales, addresses = run_tile(fx.Int64(block_start) + offset, carries, previous_packed, previous_scales, [], False, False, addresses)
            previous_packed, previous_scales = packed, scales[first_unpacked:]
        results = yield save_state(carries, packed, scales, addresses)
    ops.clear_all()
    carries, pending_packed, previous_scales, addresses = restore_state(results)
    for n in range_constexpr(stop, n_tiles - 1):
        carries, pending_packed, pending_scales, addresses = run_tile(
            n, carries, pending_packed, previous_scales, [], False, False, addresses,
        )
        previous_scales = pending_scales[first_unpacked:]
    _, pending_packed, pending_scales, _ = run_tile(n_tiles - 1, carries, pending_packed, previous_scales, [], False, True, addresses)
    return pending_packed, pending_scales