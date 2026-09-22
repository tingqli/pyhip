# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Shared M128/M256 packed down + inverse rebuild + TOPK reduce."""

import torch

if __package__:
    from .moe_multistage_reduce import make_moe_sum
else:
    from moe_multistage_reduce import make_moe_sum
from pyhip.ops.moe.flydsl.moe_gemm_2stage.moe_reduce import invert_sorted_ids


class DownReduceWorkspace:
    """Reusable storage; one instance per stream/in-flight invocation."""

    def __init__(self):
        self.data = None
        self.inverse = None

    def prepare(self, output, input_q, expert_ids, *, sort_block_m=256):
        assert output.ndim == 2 and input_q.ndim == 3
        assert sort_block_m in (128, 256)
        tokens, topk, _ = input_q.shape
        rows = max(tokens * topk, expert_ids.numel() * sort_block_m)
        shape = (rows, output.shape[1])
        # Retain the larger allocation when comparing different sorting sizes.
        if (self.data is None or self.data.shape[0] < rows or self.data.shape[1] != shape[1]
                or self.data.device != output.device):
            self.data = torch.empty(shape, dtype=torch.bfloat16, device=output.device)
        if self.inverse is None or self.inverse.shape != (tokens, topk) or self.inverse.device != output.device:
            self.inverse = torch.empty((tokens, topk), dtype=torch.int32, device=output.device)
        return self.data, self.inverse


def compile_packed_down_reduce(down, *, n, topk, workspace=None):
    """Compose a selected down callable, writing final BF16 [tokens,N].

    Every invocation rebuilds inverse, even for identical tensor pointers.
    Warm once before graph capture; sorting, quantization and weight shuffle
    are caller responsibilities. The down configuration determines the
    native sorting/packing stride; the ten-tensor launch ABI is unchanged.
    """
    workspace = DownReduceWorkspace() if workspace is None else workspace
    sort_block_m = down.config["block_m"]
    assert sort_block_m in (128, 256) and down.config["output_layout"] == "packed"
    inverse_kernel = invert_sorted_ids(topk)
    reduce = make_moe_sum(n=n, topk=topk, sort_block_m=sort_block_m)

    def buffers(args):
        output, input_q, expert_ids = args[0], args[1], args[7]
        assert output.shape == (input_q.shape[0], n) and output.dtype == torch.bfloat16
        assert input_q.shape[1:] == (topk, 256)
        assert output.is_cuda and output.is_contiguous() and output.device == input_q.device
        storage, inverse = workspace.prepare(output, input_q, expert_ids, sort_block_m=sort_block_m)
        return storage[:expert_ids.numel() * sort_block_m], inverse

    def invert(args, inverse):
        inverse.fill_(-1)
        inverse_kernel(args[5], inverse, args[8], args[5].numel(), args[1].shape[0])

    def launch(*args):
        middle, inverse = buffers(args)
        down(middle, *args[1:])
        invert(args, inverse)
        return reduce(args[0], middle, inverse)

    def components(*args):
        middle, inverse = buffers(args)
        return {"gemm": lambda: down(middle, *args[1:]),
                "inverse": lambda: invert(args, inverse),
                "reduce": lambda: reduce(args[0], middle, inverse)}

    def poison(*args):
        middle, inverse = buffers(args)
        middle.fill_(torch.nan)
        inverse.fill_(0x123456)

    launch.benchmark_components = components
    launch.poison_workspace = poison
    launch.workspace = workspace
    launch.config = {**down.config, "reduction": "custom",
                     "includes_inverse": True, "includes_reduce": True}
    return launch