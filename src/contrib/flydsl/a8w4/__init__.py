# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""A8W4 (fp8 activation x mxfp4 weight) MoE for gfx942 (MI300X / MI325X), FlyDSL.

    from a8w4 import prepare_weights, a8w4_moe
    W = prepare_weights(w1, w2, w1_scale, w2_scale)
    out = a8w4_moe(x, W, topk_weights, topk_ids, expert_offset=..., num_experts=...)
    # EP on an all-to-all receive buffer (valid row count on device, e.g. MORI):
    out = a8w4_moe_dyn(x, W, topk_weights, topk_ids, m_valid, x_scale=..., ...)

``ops.moe`` / ``ops.moe_dyn``: the ops (``kernelkit``) behind them: ``ops.moe(...)``
runs the config the tuned table picks (``ops.moe.pick(problem)``), ``ops.moe.run(cfg, ...)``
a given one; tests, tune and bench use them. See README.md.
"""

from ._ops import DEFAULT_CSV, ops  # noqa: F401
from .moe import a8w4_moe, a8w4_moe_dyn  # noqa: F401
from .weights import prepare_weights, shuffle_mxfp4, weight_shape  # noqa: F401
