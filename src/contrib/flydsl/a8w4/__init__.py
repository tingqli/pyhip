# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""A8W4 (fp8 activation x mxfp4 weight) MoE for gfx942 (MI300X / MI325X), FlyDSL.

    from a8w4 import prepare_weights, a8w4_moe
    W = prepare_weights(w1, w2, w1_scale, w2_scale)
    out = a8w4_moe(x, W, topk_weights, topk_ids, expert_offset=..., num_experts=...)
    # EP on an all-to-all receive buffer (valid row count on device, e.g. MORI):
    out = a8w4_moe_dyn(x, W, topk_weights, topk_ids, m_valid, x_scale=..., ...)

See README.md.
"""

from .moe import (  # noqa: F401
    DEFAULT_CSV,
    a8w4_moe,
    a8w4_moe_dyn,
    config_name,
    default_config,
    dyn_config,
    lookup_config,
    parse_config,
    run_config,
    supported,
)
from .weights import prepare_weights, shuffle_mxfp4, weight_shape  # noqa: F401
