# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""The ops of the a8w4 package (``pyhip.kit``): ``ops.moe`` and ``ops.moe_dyn``.

``ref.py`` registers their problem side (``moe_ref`` / ``moe_test`` / ``moe_check`` /
``moe_dims``, ``moe_dyn_*``), ``moe.py`` the kernel families (``moe_<kind>``), each with
``@op_register`` as its outermost decorator.
"""

from pyhip.kit import Registry

ops = Registry()
op_register = ops.register
