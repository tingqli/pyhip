# SPDX-License-Identifier: MIT
"""Public GRRead API with batch-dependent Down/Up splits and two launches for nonempty inputs."""

import flydsl.compiler as flyc
import torch

from .common import (
    BLOCK_M, C, H, K, R, preshuffle_weight, select_down_n_splits,
    select_n_splits, validate_rows,
)
from .prefill_down import make_down
from .prefill_up import make_up


class CombinedPaddedGRRead:
    """BF16 X[T,10240] -> Y[T,2560] for any allocatable nonnegative integer T on gfx942.

    Prepare weights, workspace, and JIT at construction; select Down N1/N2 and Up N2/N4/N8 by batch and CU count.
    Rows are runtime arguments; batches with the same N splits share artifacts while each instance has fixed workspace shapes.
    Kernels rebase X/P/Y to the current CTA tile for local buffer access; the host does not split rows.
    Every nonempty batch launches Down and Up exactly once each, preserving full P/Y layouts.
    No experimental decode/prefill, tile, prefetch, or cache switches are exposed.
    Down preserves the BF16 GEMM/SiLU boundary; Up uses full K320 FP32 logits and integer BF16 output rounding.
    Return reusable internal output; callers must copy results that need to outlive subsequent calls.
    Call an instance sequentially on one stream; concurrent streams require separate instances.
    """

    def __init__(self, rows, w_down, w_up):
        validate_rows(rows)
        if w_down.shape != (R, K) or w_up.shape != (K, R):
            raise ValueError("expected W_down[320,10240] and W_up[10240,320]")
        if w_down.dtype != torch.bfloat16 or w_up.dtype != torch.bfloat16:
            raise ValueError("GRRead weights must be BF16")
        if w_down.device != w_up.device or not w_down.is_cuda or torch.version.hip is None:
            raise ValueError("weights must be on the same ROCm device")
        props = torch.cuda.get_device_properties(w_down.device)
        if props.gcnArchName.split(":", 1)[0] != "gfx942":
            raise ValueError("GRRead currently targets gfx942")
        self.rows = rows
        self.dtype, self.device = w_down.dtype, w_down.device
        self.down_n_splits = select_down_n_splits(rows, props.multi_processor_count)
        self.n_splits = select_n_splits(rows, props.multi_processor_count)
        with torch.cuda.device(self.device):
            # H64内按stream遍历两个H32；三个N分片共用同一权重物理布局。
            up_interleaved = w_up.detach().reshape(C, H // 64, 2, 4, 2, 4, R).permute(1, 0, 2, 4, 3, 5, 6).contiguous().reshape(K, R)
            self.w_down = preshuffle_weight(w_down)
            self.w_up = preshuffle_weight(up_interleaved)
            self.partial = torch.empty((rows + BLOCK_M - 1) // BLOCK_M * BLOCK_M * R,
                                       dtype=self.dtype, device=self.device)
            self.output = torch.empty((rows, H), dtype=self.dtype, device=self.device)
            self.down = self.up = None
            if rows:
                # flyc.compile会执行一次launcher，必须提供足量X；二维shape避免扁平numel超i32。
                x = torch.empty((rows, K), dtype=self.dtype, device=self.device)
                partial = self.partial.view(-1, R)
                stream = torch.cuda.current_stream(self.device)
                # Up禁用SLP的编译选项不传播到Down；维持分别编译的既有合同。
                ld = make_down(n_splits=self.down_n_splits)
                lu = make_up(n_splits=self.n_splits)
                self.down = flyc.compile(ld, x, self.w_down, partial, rows, stream)
                self.up = flyc.compile(lu, x, self.w_up, partial, self.output, rows, stream)

    def _check_input(self, x):
        if x.shape != (self.rows, K) or x.dtype != self.dtype or x.device != self.device:
            raise ValueError("input must match the prepared rows, dtype and device")
        if not x.is_contiguous():
            raise ValueError("input must be contiguous")

    def run_down(self, x):
        self._check_input(x)
        if self.rows:
            assert self.down is not None
            with torch.cuda.device(self.device):
                self.down(x, self.w_down, self.partial.view(-1, R), self.rows, torch.cuda.current_stream(self.device))
        return self.partial

    def run_up(self, x):
        self._check_input(x)
        if self.rows:
            assert self.up is not None
            with torch.cuda.device(self.device):
                self.up(x, self.w_up, self.partial.view(-1, R), self.output, self.rows, torch.cuda.current_stream(self.device))
        return self.output

    def __call__(self, x):
        self._check_input(x)
        if self.rows:
            assert self.down is not None and self.up is not None
            with torch.cuda.device(self.device):
                stream = torch.cuda.current_stream(self.device)
                partial = self.partial.view(-1, R)
                self.down(x, self.w_down, partial, self.rows, stream)
                self.up(x, self.w_up, partial, self.output, self.rows, stream)
        return self.output