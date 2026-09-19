# SPDX-License-Identifier: MIT
"""GRRead回归测试共用的原BF16参考和容差；不参与正式运行。"""

import torch
import torch.nn.functional as F

from pyhip.contrib.flydsl.gr_read.common import C, H, R

CHECK_ROWS = 1024
DOWN_TOLERANCE = dict(rtol=2 * torch.finfo(torch.bfloat16).eps, atol=2e-5)
OUTPUT_TOLERANCE = dict(rtol=1e-2, atol=5e-3)


@torch.compile(fullgraph=True)
def _mix_reference(x, w_down, w_up):
    p = F.silu(F.linear(x, w_down) / C)
    logits = F.linear(p, w_up)
    gates = torch.sigmoid(logits).unflatten(-1, (C, H))
    return p, (gates * x.unflatten(-1, (C, H))).mean(dim=-2)


def reference_bf16(x, w_down, w_up):
    activation = torch.empty((x.shape[0], R), device=x.device, dtype=torch.bfloat16)
    output = torch.empty((x.shape[0], H), device=x.device, dtype=torch.bfloat16)
    with torch.autocast("cuda", enabled=False):
        for begin in range(0, x.shape[0], CHECK_ROWS):
            end = min(begin + CHECK_ROWS, x.shape[0])
            p, y = _mix_reference(x[begin:end], w_down, w_up)
            assert p.dtype == y.dtype == torch.bfloat16
            activation[begin:end], output[begin:end] = p, y
    return activation, output


def check_close(actual, expected, tolerance):
    assert actual.shape == expected.shape
    error_squared = expected_squared = 0.0
    for begin in range(0, actual.shape[0], CHECK_ROWS):
        a, e = actual[begin:begin + CHECK_ROWS].double(), expected[begin:begin + CHECK_ROWS].double()
        torch.testing.assert_close(a, e, **tolerance)
        error_squared += (a - e).square().sum().item()
        expected_squared += e.square().sum().item()
    return (error_squared / expected_squared if expected_squared else error_squared) ** 0.5