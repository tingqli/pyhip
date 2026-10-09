# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Experiment entry for the packaged packed BF16 TOPK reducer."""

from pyhip.ops.moe.flydsl.moe_packed_reduce import make_moe_sum

__all__ = ["make_moe_sum"]