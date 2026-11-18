# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""The ops of the a8w4 package (``pyhip.kit``): ``ops.moe`` and ``ops.moe_dyn``.

``ref.py`` registers their problem side (``moe_ref`` / ``moe_test`` / ``moe_check`` /
``moe_dims``, ``moe_dyn_*``), ``moe.py`` the kernel families (``moe_<kind>``), each
with ``@op_register`` as its outermost decorator.

Dispatch (``ops.moe(...)``, ``ops.moe.pick(dims)``): the tuned table ``DEFAULT_CSV``
(written by ``pyhip.kit.tune``), bucketed by ``num_tokens`` (the first tuned point >= the
call's token count); an activation without rows of its own uses the SiLU rows (the configs
rank alike, README 7.8). A key without rows raises ValueError: tune it first. Tuning:
the ``TOKENS`` points; ``expert_offset`` is not a table key; the timed calls
cycle through copies of the weights ``W``.
"""

import os

from pyhip.kit import Registry

DEFAULT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs",
                           "a8w4_moe_tuned.csv")
TOKENS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 320, 384, 448, 512,
          640, 768, 1024, 1280, 1536, 2048, 2560, 3072, 4096, 6144, 8192, 12288, 16384, 24576,
          32768)
ops = Registry("a8w4", table=DEFAULT_CSV, bucket="num_tokens", relax={"act": "silu"},
               points=TOKENS, unkeyed=("expert_offset",), rotate=("W",))
op_register = ops.register
