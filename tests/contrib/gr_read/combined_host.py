# SPDX-License-Identifier: MIT
"""CombinedPaddedGRRead：准备权重和 workspace，热路径只有两个 GPU launch。"""

from dataclasses import replace

import flydsl.compiler as flyc
import torch

if __package__:
    from .kernel import C, H, K, R, default_config, make_launchers, preshuffle_weight
else:
    from kernel import C, H, K, R, default_config, make_launchers, preshuffle_weight


class CombinedPaddedGRRead:
    """BF16 X[T,10240] -> Y[T,2560]，T=0..65536，当前针对 gfx942。

    Y = mean_stream(sigmoid(silu(X @ W_down.T / 4) @ W_up.T) * X)。
    构造时完成 shuffle/JIT/分配；调用始终复用 partial/output，支持 graph replay。
    默认 decode 保留 BM16；implementation="prefill" 只使用胜出配置。
    implementation="tuned" 是 prefill 的兼容名称，不再提供旧实验配置。
    prefill的P为BF16，Up原始FP32 logits直接sigmoid，Y均值使用整数helper舍入。
    up_block_m选择128完整H32拍或256两个H16子阶段；decode仍保留旧FP32工作区。
    一个实例仅用于同一 stream 的顺序调用；并发 stream 必须使用独立实例。
    """

    def __init__(self, rows, w_down, w_up, hidden_pad=4, *, implementation="decode", up_block_m=128):
        config = default_config(rows, hidden_pad)
        if implementation not in ("decode", "prefill", "tuned"):
            raise ValueError("implementation must be decode, prefill or tuned (prefill alias)")
        prefill = implementation != "decode"
        if up_block_m not in (128, 256) or (not prefill and up_block_m != 128):
            raise ValueError("up_block_m must be 128 or 256 for prefill; decode uses the default")
        if prefill:
            config = replace(config, block_m=64, down_n=320, block_k=64,
                             split_k=1, down_mode="prefill_1x4", compensate_hidden=False)
        if w_down.shape != (R, K) or w_up.shape != (K, R):
            raise ValueError("expected W_down[320,10240] and W_up[10240,320]")
        if w_down.dtype != torch.bfloat16 or w_up.dtype != torch.bfloat16:
            raise ValueError("GR read weights must be BF16")
        if w_down.device != w_up.device or not w_down.is_cuda or torch.version.hip is None:
            raise ValueError("weights must be on the same ROCm device")
        arch = torch.cuda.get_device_properties(w_down.device).gcnArchName.split(":", 1)[0]
        if arch != "gfx942":
            raise ValueError("GR read currently targets gfx942")
        self.rows, self.config = rows, config
        self.implementation = implementation
        self.dtype, self.device = w_down.dtype, w_down.device
        if prefill:
            # H64内按stream遍历两个H32包；交换H16子块与lane组，使每lane8H连续。
            up_interleaved = w_up.detach().reshape(C, H // 64, 2, 4, 2, 4, R).permute(1, 0, 2, 4, 3, 5, 6).contiguous().reshape(K, R)
        else:
            # decoding仍使用原H/C交错布局。
            up_interleaved = w_up.detach().reshape(C, H, R).permute(1, 0, 2).contiguous().reshape(K, R)
        self.w_down = preshuffle_weight(w_down)
        self.w_up = preshuffle_weight(up_interleaved)
        self.up_block_m = up_block_m if prefill else 16
        alignment = self.up_block_m if prefill else config.block_m
        padded = (rows + alignment - 1) // alignment * alignment
        self.padded_rows = padded
        splits = config.split_k if config.down_mode == "partial" else 1
        partial_dtype = torch.bfloat16 if prefill else torch.float32
        self.partial = torch.empty(splits * padded * R, dtype=partial_dtype, device=self.device)
        self.output = torch.empty((rows, H), dtype=self.dtype, device=self.device)
        self.down = self.up = self.dispatch = None
        if rows:
            x = torch.empty(rows * K, dtype=self.dtype, device=self.device)
            if prefill:
                if __package__:
                    from .prefil_down_1x4 import make_down_1x4
                    from .prefil_up_8x1 import make_up_8x1
                else:
                    from prefil_down_1x4 import make_down_1x4
                    from prefil_up_8x1 import make_up_8x1
                ld = make_down_1x4(rows, padded)
                lu = make_up_8x1(rows, padded, block_m=self.up_block_m)
            else:
                ld, lu, combined = make_launchers(rows, config)

            stream = torch.cuda.current_stream(self.device)
            self.down = flyc.compile(ld, x, self.w_down, self.partial, stream)
            self.up = flyc.compile(lu, x, self.w_up, self.partial, self.output.view(-1), stream)
            if prefill:
                # 嵌套jit不会独立应用Up的LLVM选项；分别编译/顺序launch，避免SLP开关污染Down。
                self.dispatch = self._launch_staged
            else:
                self.dispatch = flyc.compile(combined, x, self.w_down, self.w_up, self.partial,
                                             self.output.view(-1), stream)

    def _launch_staged(self, x, w_down, w_up, partial, output, stream):
        assert self.down is not None and self.up is not None
        self.down(x, w_down, partial, stream)
        self.up(x, w_up, partial, output, stream)

    def _check_input(self, x):
        if x.shape != (self.rows, K) or x.dtype != self.dtype or x.device != self.device:
            raise ValueError("input must match the prepared rows, dtype and device")
        if not x.is_contiguous():
            raise ValueError("input must be contiguous")

    def run_down(self, x):
        self._check_input(x)
        if self.rows:
            assert self.down is not None
            self.down(x.view(-1), self.w_down, self.partial, torch.cuda.current_stream(self.device))
        return self.partial

    def run_up(self, x):
        self._check_input(x)
        if self.rows:
            assert self.up is not None
            self.up(x.view(-1), self.w_up, self.partial, self.output.view(-1), torch.cuda.current_stream(self.device))
        return self.output

    def __call__(self, x):
        self._check_input(x)
        if self.rows:
            assert self.dispatch is not None
            self.dispatch(x.view(-1), self.w_down, self.w_up, self.partial, self.output.view(-1),
                          torch.cuda.current_stream(self.device))
        return self.output