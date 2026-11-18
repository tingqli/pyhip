# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""The tuned configs of every problem (what the dispatch can pick) vs the reference, on
the problems below (``ref.moe_problem`` / ``moe_dyn_problem`` turn them into inputs), generated
by ``pyhip.kit``:

    pytest a8w4/tests/test_a8w4_ops.py -q -k "ep8 and ep3k"   # from the directory holding a8w4/
"""

import os
import sys

import pytest
import torch

from pyhip.kit import dim
from pyhip.kit.testing import op_tests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from a8w4 import ops  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or "gfx942" not in torch.cuda.get_device_properties(0).gcnArchName,
    reason="a8w4 kernels need gfx942 (fp8 FNUZ MFMA)")

MOE, MOE_DYN = ops.moe, ops.moe_dyn

# every shape x activation at 1, 77 and 600 tokens
test_moe = op_tests(MOE, MOE.problems("shape", dim("num_tokens", (1, 77, 600), id="t{}"), "act"))
# an EP all-to-all receive buffer of capacity rows, num_tokens of them valid
test_moe_dyn = op_tests(MOE_DYN, MOE_DYN.problems(
    "shape", dim("capacity, num_tokens", {"v0": (512, 0), "v1": (512, 1), "v37": (512, 37),
                                         "v300": (512, 300), "v1500": (2048, 1500)}),
    "fp8", "act"))
