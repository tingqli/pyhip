# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Compatibility dispatcher for the stage-specific MoE GEMM builders."""

import functools

from .common import get_device_cache_key
from .gemm1 import compile_moe_gemm1
from .gemm2 import compile_moe_gemm2


@functools.cache
def _compile_gemm_cached(
    device_cache_key,
    N,
    K,
    weight_dtype,
    weight_quant_type,
    TOPK,
    BLOCK_TILE_SIZE_M,
    BLOCK_TILE_SIZE_N,
    stage="gateup",
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
    del device_cache_key
    if stage == "gateup":
        return compile_moe_gemm1(
            N=N,
            K=K,
            weight_dtype=weight_dtype,
            weight_quant_type=weight_quant_type,
            TOPK=TOPK,
            BLOCK_TILE_SIZE_M=BLOCK_TILE_SIZE_M,
            BLOCK_TILE_SIZE_N=BLOCK_TILE_SIZE_N,
            alg=alg,
            E=E,
            act_quant_type=act_quant_type,
            tile_k=tile_k,
            activation=activation,
            swiglu_limit=swiglu_limit,
            METADATA_TILE_SIZE_M=METADATA_TILE_SIZE_M,
        )
    return compile_moe_gemm2(
        N=N,
        K=K,
        weight_dtype=weight_dtype,
        weight_quant_type=weight_quant_type,
        TOPK=TOPK,
        BLOCK_TILE_SIZE_M=BLOCK_TILE_SIZE_M,
        BLOCK_TILE_SIZE_N=BLOCK_TILE_SIZE_N,
        alg=alg,
        E=E,
        USE_ATOMIC_WRITE=USE_ATOMIC_WRITE,
        act_quant_type=act_quant_type,
        tile_k=tile_k,
        activation=activation,
        swiglu_limit=swiglu_limit,
        down_path=down_path,
        down_output_padding_bytes=down_output_padding_bytes,
        METADATA_TILE_SIZE_M=METADATA_TILE_SIZE_M,
    )


def compile_gemm(
    N,
    K,
    weight_dtype,
    weight_quant_type,
    TOPK,
    BLOCK_TILE_SIZE_M,
    BLOCK_TILE_SIZE_N,
    stage="gateup",
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
    return _compile_gemm_cached(
        get_device_cache_key(),
        N,
        K,
        weight_dtype,
        weight_quant_type,
        TOPK,
        BLOCK_TILE_SIZE_M,
        BLOCK_TILE_SIZE_N,
        stage,
        alg,
        E,
        USE_ATOMIC_WRITE,
        act_quant_type,
        tile_k,
        activation,
        swiglu_limit,
        down_path,
        down_output_padding_bytes,
        METADATA_TILE_SIZE_M,
    )


compile_gemm.cache_clear = _compile_gemm_cached.cache_clear
compile_gemm.cache_info = _compile_gemm_cached.cache_info
