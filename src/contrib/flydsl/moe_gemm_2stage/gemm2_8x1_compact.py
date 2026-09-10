# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""显式8x1 compact路径及GPU任务表：M64 metadata，M256满块＋M64尾块。

每条任务为两个Int32：[physical_row_begin, expert_id]。两种任务共享原physical
row编号；counts保存full_count*256、tail_count*64供现有kernel的uniform guard使用。
要求每expert只出现一个连续run；run顺序不要求按expert编号排序。
满块末轮不足指定CU比例时，把全局满块表的后缀拆为M64，物理行编号不变。
"""

import functools
from fractions import Fraction

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr
import torch

from .common import get_device_cache_key, torch_tensor_to_pointer
from .gemm2_1x4 import _build_moe_gemm2_1x4
from .gemm2_8x1 import _build_moe_gemm2_8x1
from .gemm2_default import _build_moe_gemm2_default


def _build_moe_gemm2_8x1_compact(
    N, K, weight_dtype, weight_quant_type, TOPK,
    BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N, stage="down", alg="splitk", E=None,
    USE_ATOMIC_WRITE=True, act_quant_type=None, tile_k=None,
    activation="silu", swiglu_limit=None, down_path="default",
    down_output_padding_bytes=None, METADATA_TILE_SIZE_M=None,
    _n_loop=1, _store_cache=2,
    _min_tail_utilization=0.6,
):
    assert stage == "down" and alg == "prefill_1x4" and down_path == "8x1_compact"
    assert BLOCK_TILE_SIZE_M == 64 and BLOCK_TILE_SIZE_N == 128
    assert METADATA_TILE_SIZE_M in (None, 64)
    assert E is not None and 0 < E <= 2048
    assert K in (192, 256, 320, 384, 512, 640), "8x1_compact仅支持K=192/256/320/384/512/640"
    assert N > 0 and N % 128 == 0
    assert down_output_padding_bytes in (0, 32, 64, 128)
    common = dict(
        N=N, K=K, weight_dtype=weight_dtype, weight_quant_type=weight_quant_type,
        act_quant_type=act_quant_type, TOPK=TOPK, stage=stage, alg=alg,
        E=E, USE_ATOMIC_WRITE=USE_ATOMIC_WRITE, tile_k=tile_k,
        activation=activation, swiglu_limit=swiglu_limit,
        down_output_padding_bytes=down_output_padding_bytes, _task_table=True,
    )
    full = _build_moe_gemm2_8x1(
        **common, BLOCK_TILE_SIZE_M=256, BLOCK_TILE_SIZE_N=128, down_path="8x1",
        _n_loop=_n_loop, _store_cache=_store_cache,
    )
    # K192/K320的末块BK192只改变8x1满块；M64尾kernel保持其既有BK128算法。
    tail_common = {**common, "tile_k": 128 if K in (192, 320) and tile_k == 192 else tile_k}
    if N % 256 == 0:
        tail = _build_moe_gemm2_1x4(
            **tail_common, BLOCK_TILE_SIZE_M=64, BLOCK_TILE_SIZE_N=256, down_path="1x4_64x256",
            _store_cache=_store_cache,
        )
    else:
        # N128正确性/小shape回退；不能让N256尾kernel写越界。
        tail = _build_moe_gemm2_default(
            **tail_common, BLOCK_TILE_SIZE_M=64, BLOCK_TILE_SIZE_N=128, down_path="default",
        )
    build = build_task_table(E, min_tail_utilization=_min_tail_utilization)

    @flyc.jit
    def launch_compact(
        p_input: fx.Pointer, p_weight: fx.Pointer, p_output: fx.Pointer,
        p_sorted_ids: fx.Pointer, p_sorted_weights: fx.Pointer,
        p_sorted_expert_ids: fx.Pointer, p_num_valid_ids: fx.Pointer,
        p_w_scale: fx.Pointer, p_a_scale: fx.Pointer, M: fx.Int32, task_num: fx.Int32,
        p_full_tasks: fx.Pointer, p_tail_tasks: fx.Pointer, p_counts: fx.Pointer,
        full_capacity: fx.Int32, tail_capacity: fx.Int32, stream: fx.Stream,
    ):
        # 三个kernel全在同一stream，count只由device guard消费；无CPU读回。
        build(p_sorted_expert_ids, p_num_valid_ids, p_full_tasks, p_tail_tasks, p_counts, stream)
        full(
            p_input, p_weight, p_output, p_sorted_ids, p_sorted_weights,
            p_full_tasks, p_counts, p_w_scale, p_a_scale, M, full_capacity, stream,
        )
        tail(
            p_input, p_weight, p_output, p_sorted_ids, p_sorted_weights,
            p_tail_tasks, p_counts + 1, p_w_scale, p_a_scale, M, tail_capacity, stream,
        )

    launch_compact.compile_hints["target_features"] = "-packed-fp32-ops"
    return launch_compact


def device_cu_count(device=None):
    """ROCm的multi_processor_count为当前可用CU数，不按型号/XCC数猜测。"""
    if device is None:
        device = torch.cuda.current_device()
    count = int(torch.cuda.get_device_properties(device).multi_processor_count)
    assert count > 0
    return count


def _balance_parameters(cu_count, min_tail_utilization):
    assert 0 <= min_tail_utilization <= 1, "末轮利用率阈值必须位于[0,1]"
    threshold = Fraction(str(min_tail_utilization)).limit_denominator(1_000_000)
    if cu_count is None:
        cu_count = device_cu_count()
    assert isinstance(cu_count, int) and not isinstance(cu_count, bool) and cu_count > 0
    numerator, denominator = threshold.numerator, threshold.denominator
    max_split = max(0, (cu_count * numerator - 1) // denominator)
    return cu_count, numerator, denominator, max_split


def task_capacities(metadata_capacity, experts, *, cu_count=None, min_tail_utilization=0.6):
    assert metadata_capacity >= 0 and 0 < experts <= 2048
    _, _, _, max_split = _balance_parameters(cu_count, min_tail_utilization)
    # 原每expert最多3个tail；每个被拆满块再增加4个tail，总量不超过metadata块数。
    max_split = min(metadata_capacity // 4, max_split)
    return max(1, (metadata_capacity + 3) // 4), max(1, min(metadata_capacity, 3 * experts + 4 * max_split))


def allocate_task_buffers(sorted_expert_ids, experts, *, cu_count=None, min_tail_utilization=0.6):
    """只读host端tensor形状，不读取GPU数据，不同步。可在计时外预分配复用。"""
    if cu_count is None:
        cu_count = device_cu_count(sorted_expert_ids.device)
    full, tail = task_capacities(sorted_expert_ids.numel(), experts, cu_count=cu_count,
                                 min_tail_utilization=min_tail_utilization)
    return (
        torch.empty((full, 2), dtype=torch.int32, device=sorted_expert_ids.device),
        torch.empty((tail, 2), dtype=torch.int32, device=sorted_expert_ids.device),
        torch.empty(2, dtype=torch.int32, device=sorted_expert_ids.device),
    )


def build_task_table(experts, *, cu_count=None, min_tail_utilization=0.6):
    """先解析当前设备，再按CU/阈值缓存，避免仅按E缓存跨GPU误复用。"""
    assert 0 < experts <= 2048
    cu_count, numerator, denominator, _ = _balance_parameters(cu_count, min_tail_utilization)
    return _build_task_table_cached(get_device_cache_key(), experts, cu_count, numerator, denominator)


@functools.cache
def _build_task_table_cached(device_cache_key, experts, cu_count, threshold_numerator, threshold_denominator):
    del device_cache_key
    assert 0 < experts <= 2048
    width = max(256, 1 << (experts - 1).bit_length())
    threads = 256
    rounds = width // threads

    @flyc.kernel(known_block_size=[threads, 1, 1])
    def compact_m64_tasks(
        sorted_experts: fx.Pointer, valid_rows: fx.Pointer,
        full_tasks: fx.Pointer, tail_tasks: fx.Pointer, counts: fx.Pointer,
    ):
        tid = fx.Int32(gpu.thread_idx.x)
        storage = fx.SharedAllocator().allocate(fx.Array[fx.Int32, width * 4, 16])
        shared = storage.peek().view(fx.make_layout(width * 4, 1))
        for part in range_constexpr(rounds):
            idx = tid + part * threads
            shared[idx] = fx.Int32(0)
            shared[width + idx] = fx.Int32(0)
        gpu.barrier()
        blocks = valid_rows[0] // 64
        # 单WG遍历metadata：写run首尾到LDS，然后统一scan，避免计数初始化竞争。
        for base in range(fx.Int32(0), blocks, threads):
            idx = base + tid
            if idx < blocks:
                expert = sorted_experts[idx]
                if (expert >= 0) & (expert < experts):
                    prev_idx = fx.arith.select(idx > 0, idx - 1, idx)
                    next_idx = fx.arith.select(idx + 1 < blocks, idx + 1, idx)
                    if (idx == 0) | (sorted_experts[prev_idx] != expert):
                        shared[expert] = idx
                    if (idx + 1 == blocks) | (sorted_experts[next_idx] != expert):
                        shared[width + expert] = idx + 1
        gpu.barrier()
        for part in range_constexpr(rounds):
            idx = tid + part * threads
            size = shared[width + idx] - shared[idx]
            shared[2 * width + idx] = size // 4
            shared[3 * width + idx] = size % 4
        gpu.barrier()
        # 两个inclusive prefix共用每轮barrier，稳定按expert编号分配任务位置。
        for shift in range_constexpr(width.bit_length() - 1):
            distance = 1 << shift
            next_full, next_tail = [], []
            for part in range_constexpr(rounds):
                idx = tid + part * threads
                peer = fx.arith.select(idx >= distance, idx - distance, idx)
                add_full = fx.arith.select(idx >= distance, shared[2 * width + peer], fx.Int32(0))
                add_tail = fx.arith.select(idx >= distance, shared[3 * width + peer], fx.Int32(0))
                next_full.append(shared[2 * width + idx] + add_full)
                next_tail.append(shared[3 * width + idx] + add_tail)
            gpu.barrier()
            for part in range_constexpr(rounds):
                idx = tid + part * threads
                shared[2 * width + idx] = next_full[part]
                shared[3 * width + idx] = next_tail[part]
            gpu.barrier()
        original_full = shared[3 * width - 1]
        original_tail = shared[4 * width - 1]
        if const_expr(threshold_numerator > 0):
            remainder = original_full % cu_count
            # 严格小于：默认5*r < 3*CU；r==0不拆，CU数来自API。
            split_full = fx.arith.select(
                (remainder > 0) & (remainder * threshold_denominator < cu_count * threshold_numerator),
                remainder, fx.Int32(0),
            )
            keep_full = original_full - split_full
        else:
            split_full = fx.Int32(0)
            keep_full = original_full
        if tid == 0:
            counts[0] = keep_full * 256
            counts[1] = (original_tail + 4 * split_full) * 64
        for part in range_constexpr(rounds):
            expert = tid + part * threads
            if expert < experts:
                start = shared[expert]
                size = shared[width + expert] - start
                full_count, tail_count = size // 4, size % 4
                full_begin = shared[2 * width + expert] - full_count
                tail_begin = shared[3 * width + expert] - tail_count
                if const_expr(threshold_numerator > 0):
                    # 拆分的是全局满块表后缀，可跨expert边界；每expert内仍按物理行排序。
                    split_before = fx.arith.select(full_begin > keep_full, full_begin - keep_full, fx.Int32(0))
                    split_through = fx.arith.select(
                        full_begin + full_count > keep_full,
                        full_begin + full_count - keep_full, fx.Int32(0),
                    )
                    split_here = split_through - split_before
                    tail_begin += 4 * split_before
                for index in range(fx.Int32(0), full_count, 1):
                    slot = full_begin + index
                    if const_expr(threshold_numerator > 0):
                        if slot < keep_full:
                            full_tasks[2 * slot] = (start + 4 * index) * 64
                            full_tasks[2 * slot + 1] = expert
                        else:
                            converted_index = slot - keep_full - split_before
                            for part64 in range_constexpr(4):
                                tail_slot = tail_begin + 4 * converted_index + part64
                                tail_tasks[2 * tail_slot] = (start + 4 * index + part64) * 64
                                tail_tasks[2 * tail_slot + 1] = expert
                    else:
                        full_tasks[2 * slot] = (start + 4 * index) * 64
                        full_tasks[2 * slot + 1] = expert
                for index in range_constexpr(3):
                    if index < tail_count:
                        slot = tail_begin + (4 * split_here if const_expr(threshold_numerator > 0) else 0) + index
                        tail_tasks[2 * slot] = (start + 4 * full_count + index) * 64
                        tail_tasks[2 * slot + 1] = expert

    @flyc.jit
    def launch(
        sorted_experts: fx.Pointer, valid_rows: fx.Pointer,
        full_tasks: fx.Pointer, tail_tasks: fx.Pointer, counts: fx.Pointer,
        stream: fx.Stream,
    ):
        compact_m64_tasks(sorted_experts, valid_rows, full_tasks, tail_tasks, counts).launch(
            grid=(1, 1, 1), block=(threads, 1, 1), stream=stream,
        )

    return launch


build_task_table.cache_clear = _build_task_table_cached.cache_clear
build_task_table.cache_info = _build_task_table_cached.cache_info


def prepare_tasks(sorted_experts, valid_rows, experts, buffers=None, *, cu_count=None, min_tail_utilization=0.6):
    """Tensor便利入口；任务计数始终保留在device，不用于CPU分支或grid决策。"""
    assert sorted_experts.device.type == "cuda" and valid_rows.device == sorted_experts.device
    if cu_count is None:
        cu_count = device_cu_count(sorted_experts.device)
    if buffers is None:
        buffers = allocate_task_buffers(sorted_experts, experts, cu_count=cu_count,
                                        min_tail_utilization=min_tail_utilization)
    required_full, required_tail = task_capacities(sorted_experts.numel(), experts, cu_count=cu_count,
                                                  min_tail_utilization=min_tail_utilization)
    assert buffers[0].ndim == buffers[1].ndim == 2 and buffers[0].shape[1] == buffers[1].shape[1] == 2
    assert buffers[0].shape[0] >= required_full and buffers[1].shape[0] >= required_tail, "任务workspace容量不足，需按新的CU拆分上限分配"
    assert buffers[2].numel() >= 2 and all(t.dtype == torch.int32 and t.device == sorted_experts.device for t in buffers)
    with torch.cuda.device(sorted_experts.device):
        build_task_table(experts, cu_count=cu_count, min_tail_utilization=min_tail_utilization)(
            torch_tensor_to_pointer(sorted_experts), torch_tensor_to_pointer(valid_rows),
            *(torch_tensor_to_pointer(value) for value in buffers), torch.cuda.current_stream(),
        )
    return buffers