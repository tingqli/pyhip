# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""The ops of the a8w4 package (``pyhip.kit``): ``ops.moe`` and ``ops.moe_dyn``.

``ref.py`` registers their problem side (``moe_ref`` / ``moe_problem`` / ``moe_check`` /
``moe_dims``, ``moe_dyn_*``), ``moe.py`` the solvers (``moe_<solver>``), each
with ``@op_register`` as its outermost decorator.

Dispatch (``ops.moe(...)``, ``ops.moe.pick(problem)``): the tuned table ``DEFAULT_CSV``
(written by ``pyhip.kit.tune``): keyed by the shape, routing and act, bucketed by
``num_tokens`` (its column ``~num_tokens``: the first tuned point >= the call's token
count). A key without rows raises ValueError: tune it first. How the table is tuned:
``tune.TUNING``.
"""

import os

from pyhip.kit import Registry

DEFAULT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs",
                           "a8w4_moe_tuned.csv")
ops = Registry("a8w4", table=DEFAULT_CSV)
op_register = ops.register
