# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""The ops of the a8w4 package (``kernelkit``): ``ops.moe`` and ``ops.moe_dyn``.

``ref.py`` registers their problem side (``moe_ref`` / ``moe_problem`` / ``moe_check`` /
``moe_dims``, ``moe_dyn_*``), ``moe.py`` the solvers (``moe_<solver>``), each
with ``@op_register`` as its outermost decorator.

Dispatch (``ops.moe(...)``, ``ops.moe.pick(problem)``): the tuned table ``DEFAULT_CSV``
(written by ``kernelkit.tune``): keyed by the shape, routing and act, bucketed by
``num_tokens`` (its column ``~num_tokens``: the first tuned point >= the call's token
count). A key without rows raises ValueError: tune it first. How the table is tuned:
``tune.TUNING``.

``ep_ops.moe_ep`` (``moe_ep.py``): the expert-parallel MoE layer (dispatch, experts,
combine), one call on each rank's GPU (``@multi_gpu("world")``; solvers ``mega`` and the
MORI-based ``mori`` / ``mori_eager``), with its own table ``EP_CSV`` keyed by the shape,
experts and ranks and bucketed by the batch over all ranks (``~tokens``); tuned by
``tune.EP_TUNING``.
"""

import os

from kernelkit import Registry

DEFAULT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs",
                           "a8w4_moe_tuned.csv")
ops = Registry("a8w4", table=DEFAULT_CSV)
op_register = ops.register

EP_CSV = os.path.join(os.path.dirname(DEFAULT_CSV), "a8w4_moe_ep_tuned.csv")
ep_ops = Registry("a8w4-moe-ep", table=EP_CSV)
ep_register = ep_ops.register
