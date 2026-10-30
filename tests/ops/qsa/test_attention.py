"""Pytest entry for sparse_prefill_op_test.py: functional and graph cases, groups 12/6/3."""

import pytest
import torch

from tests.ops.qsa import sparse_prefill_op_test as op

# The op test makes CUDA the default device on import; keep it local to its cases.
torch.set_default_device(None)

GROUPS = (12, 6, 3)


def _run(test, case, group, layout):
    if not torch.cuda.is_available() or op.get_gfx() not in op.SUPPORTED_GFX:
        pytest.skip("requires gfx942")
    with torch.device("cuda"):
        test(case, group * op.CASES[layout][2])


@pytest.mark.parametrize("group", GROUPS)
@pytest.mark.parametrize("case", [*op.FUNCTIONAL_CASES, *op.CONSTANT_CASES])
def test_sparse_prefill(case, group):
    _run(op.test_qsa_sparse_attention, case, group, case)


@pytest.mark.parametrize("group", GROUPS)
@pytest.mark.parametrize("case", list(op.GRAPH_CASES))
def test_graph_replay(case, group):
    _run(op.test_qsa_graph_replay, case, group, op.GRAPH_CASES[case][0])
