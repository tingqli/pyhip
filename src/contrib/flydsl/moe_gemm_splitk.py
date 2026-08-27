# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Compatibility facade for the refactored two-stage MoE kernels."""

from .moe_gemm_2stage.gemm import compile_gemm
from .moe_gemm_2stage.moe_reduce import invert_sorted_ids, sorted_sum
from .moe_gemm_2stage.quant import flydsl_absmax, flydsl_quant_per_tensor

__all__ = [
    "compile_gemm",
    "flydsl_absmax",
    "flydsl_quant_per_tensor",
    "invert_sorted_ids",
    "sorted_sum",
]
