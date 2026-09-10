# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""显式8x1 compact路径：M64 metadata，M256满块＋M64尾块，输出编号不变。"""

import flydsl.compiler as flyc
import flydsl.expr as fx

from .compact_tasks import build_task_table
from .gemm2_1x4 import _build_moe_gemm2_1x4
from .gemm2_8x1 import _build_moe_gemm2_8x1
from .gemm2_default import _build_moe_gemm2_default


def _build_moe_gemm2_8x1_compact(
    N, K, weight_dtype, weight_quant_type, TOPK,
    BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N, stage="down", alg="splitk", E=None,
    USE_ATOMIC_WRITE=True, act_quant_type=None, tile_k=None,
    activation="silu", swiglu_limit=None, down_path="default",
    down_output_padding_bytes=None, METADATA_TILE_SIZE_M=None,
    _n_loop=1, _store_cache=2, _relax_vmcnt=True,
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
        _n_loop=_n_loop, _store_cache=_store_cache, _relax_vmcnt=_relax_vmcnt,
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