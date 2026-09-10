# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""K128的4M×2N布局：1N=两个M32微阶段，保留B/未退休C/输出读结果。"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import Vector as Vec

from .gemm2_8x1_schedule import k128_vmem_wait_schedule


@flyc.jit
def emit_k128_nloop(
    n_tiles, unroll_n, ptpc, relax, c, staging, current_b, current_scale,
    ops, issue_b, commit_b, read_b, load_scale, retire, destinations,
    store, compute, wait, priority, stage_end, first_stagger,
):
    budgets = k128_vmem_wait_schedule(n_tiles, ptpc, relax)

    def run_tile(n, bfrags, scale, packed, output, first, second, last):
        future = []
        for micro in range_constexpr(2):
            previous_scale = scale
            priority(0)
            budget_n = n if isinstance(n, int) else 2
            wait(vmcnt=budgets[2 * budget_n + micro])
            if const_expr(not last):
                commit_b(n + 1, micro)
                fx.rocdl.sched_barrier(0)
            # 中间循环只到倒数第三N，最后两N剥离，避免越界预取。
            if const_expr(not last):
                issue_b(n + 2, micro)
                fx.rocdl.sched_barrier(0)
            if const_expr(not first and micro == 0):
                scale = load_scale(n)
            if const_expr(not first and (not second or micro == 1)):
                store(output, destinations(n - 2 if micro == 0 else n - 1, 1 - micro))
            if const_expr(not first):
                output, _ = retire(n - 1, micro, packed)
            wait(lgkmcnt=0 if first else 8)
            stage_end()
            priority(3)
            if const_expr(not last):
                future.append(read_b(n + 1, micro))
            packed = compute(micro, bfrags, previous_scale, 0 if first and micro == 0 else 1)
            wait(lgkmcnt=0)
            priority(0)
            stage_end()
            if const_expr(first and micro == 0):
                first_stagger()
        return future, scale, packed, output

    # issue_b对倒数第二N的未来请求用编译期tail包装裁剪，循环主体没有动态分支。
    bfrags, scale, packed, output = run_tile(0, current_b, current_scale, [], None, True, False, False)
    bfrags, scale, packed, output = run_tile(1, bfrags, scale, packed, output, False, True, False)
    b_carriers = [fx.make_fragment_like(value) for value in bfrags]
    output_carriers = [fx.make_fragment_like(value) for value in output]
    if const_expr(ptpc):
        scale_carrier = fx.make_fragment_like(scale)

    def save(b, scale, packed, out):
        state = [value.load() for value in staging] + [value.load() for value in b]
        for row in range_constexpr(2, 4):
            for group in range_constexpr(4):
                state.append(c[None, group, row].load())
        if const_expr(ptpc):
            state.append(scale.load())
        for pair in range_constexpr(2):
            for row in range_constexpr(2):
                state.append(packed[pair][row])
        state.extend(value.load() for value in out)
        return state

    def restore(state):
        for index in range_constexpr(2):
            staging[index].store(state[index])
            b_carriers[index].store(state[index + 2])
        offset = 4
        for row in range_constexpr(2, 4):
            for group in range_constexpr(4):
                c[None, group, row].store(state[offset])
                offset += 1
        if const_expr(ptpc):
            scale_carrier.store(state[offset])
            offset += 1
            scale = scale_carrier
        else:
            scale = fx.Float32(1.0)
        packed = []
        for pair in range_constexpr(2):
            packed.append([Vec(state[offset]), Vec(state[offset + 1])])
            offset += 2
        for index in range_constexpr(4):
            output_carriers[index].store(state[offset + index])
        return list(b_carriers), scale, packed, list(output_carriers)

    initial = save(bfrags, scale, packed, output)
    stop = 2 + ((n_tiles - 4) // unroll_n) * unroll_n
    ops.clear_all()
    for block_start, state in range(2, stop, unroll_n, init=initial):
        bfrags, scale, packed, output = restore(state)
        for offset in range_constexpr(unroll_n):
            bfrags, scale, packed, output = run_tile(fx.Int64(block_start) + offset, bfrags, scale, packed, output, False, False, False)
        results = yield save(bfrags, scale, packed, output)
    ops.clear_all()
    bfrags, scale, packed, output = restore(results)
    for n in range_constexpr(stop, n_tiles - 1):
        bfrags, scale, packed, output = run_tile(n, bfrags, scale, packed, output, False, False, False)
    _, scale, packed, output = run_tile(n_tiles - 1, bfrags, scale, packed, output, False, False, True)
    return packed, scale, (output, destinations(n_tiles - 2, 1))