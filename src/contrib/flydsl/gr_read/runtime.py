# SPDX-License-Identifier: MIT
"""GRRead正式入口：构造时按batch选择Up，热路径固定Down＋Up两次launch。"""

import flydsl.compiler as flyc
import torch

from .common import BLOCK_M, C, H, K, R, preshuffle_weight, select_n_splits, validate_rows
from .down import make_down
from .up_n2 import make_up_n2
from .up_n4_n8 import make_up_n4_n8


class CombinedPaddedGRRead:
    """BF16 X[T,10240] -> Y[T,2560]，支持gfx942上的T=0..65536。

    构造时准备权重、workspace和JIT；N2/N4/N8由batch及设备CU数自动选择。
    不保留decode/prefill、tile、预取或cache的实验开关。
    Down保留BF16 GEMM/SiLU边界；Up用完整K320 FP32 logits和整数BF16输出舍入。
    返回内部可复用output；需保留多次结果时由调用者复制。
    同一实例只能在同一stream顺序调用；并发stream须使用独立实例。
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
        self.n_splits = select_n_splits(rows, props.multi_processor_count)
        self.padded_rows = (rows + BLOCK_M - 1) // BLOCK_M * BLOCK_M
        with torch.cuda.device(self.device):
            # H64内按stream遍历两个H32；物理布局与已确认的三个Up核完全一致。
            up_interleaved = w_up.detach().reshape(C, H // 64, 2, 4, 2, 4, R).permute(1, 0, 2, 4, 3, 5, 6).contiguous().reshape(K, R)
            self.w_down = preshuffle_weight(w_down)
            self.w_up = preshuffle_weight(up_interleaved)
            self.partial = torch.empty(self.padded_rows * R, dtype=self.dtype, device=self.device)
            self.output = torch.empty((rows, H), dtype=self.dtype, device=self.device)
            self.down = self.up = None
            if rows:
                x = torch.empty(rows * K, dtype=self.dtype, device=self.device)
                ld = make_down(rows, self.padded_rows)
                lu = make_up_n2(rows, self.padded_rows) if self.n_splits == 2 else make_up_n4_n8(
                    rows, self.padded_rows, n_splits=self.n_splits)
                stream = torch.cuda.current_stream(self.device)
                # Up禁用SLP的编译选项不传播到Down；维持分别编译的既有合同。
                self.down = flyc.compile(ld, x, self.w_down, self.partial, stream)
                self.up = flyc.compile(lu, x, self.w_up, self.partial, self.output.view(-1), stream)

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
                self.down(x.view(-1), self.w_down, self.partial, torch.cuda.current_stream(self.device))
        return self.partial

    def run_up(self, x):
        self._check_input(x)
        if self.rows:
            assert self.up is not None
            with torch.cuda.device(self.device):
                self.up(x.view(-1), self.w_up, self.partial, self.output.view(-1), torch.cuda.current_stream(self.device))
        return self.output

    def __call__(self, x):
        self._check_input(x)
        if self.rows:
            assert self.down is not None and self.up is not None
            with torch.cuda.device(self.device):
                stream = torch.cuda.current_stream(self.device)
                flat = x.view(-1)
                self.down(flat, self.w_down, self.partial, stream)
                self.up(flat, self.w_up, self.partial, self.output.view(-1), stream)
        return self.output