# SPDX-License-Identifier: MIT
"""Preparation and two-launch prefill calls; no test or model dependencies."""
from functools import cache

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext
import torch

from .common import H, K, R, select_prefill_config, validate_rows
from .prefill_down import make_down
from .prefill_up import make_up


@cache
def _launchers(config):
    dm, dw, dn, dk, um, un, swizzle = config
    return (
        make_down(n_splits=dn, block_m=dm, num_waves=dw, block_k=dk, swizzle_shift=swizzle),
        make_up(n_splits=un, block_m=um),
    )


@cache
def _pair_launcher(config):
    down, up = _launchers(config)

    @flyc.jit
    def launch(X: fx.Tensor, WD: fx.Tensor, WU: fx.Tensor, P: fx.Tensor,
               Y: fx.Tensor, rows: fx.Int64, stream: fx.Stream):
        down(X, WD, P, rows, stream)
        up(X, WU, P, Y, rows, stream)

    launch.compile_hints['llvm_options'] = {'vectorize-slp': False}
    return launch


def _check_buffer(tensor, shape, dtype, device, name):
    if tensor.shape != shape or tensor.dtype != dtype or tensor.device != device or not tensor.is_contiguous():
        raise ValueError(f'{name} must be contiguous {dtype} {shape} on {device}')
    if tensor.numel() and tensor.data_ptr() % 16:
        raise ValueError(f'{name} must have a 16-byte-aligned base')


class GRReadPrefill:
    """Prepared BF16 GR read prefill on gfx942.

    Pass weights from prepare_weights(), once per original weight pair. Instances
    for different T share those tensors and own separate P/Y workspaces. Prepare
    before hot execution; each nonempty call launches Down and Up on the current
    stream. The next call overwrites output. Concurrent streams need separate
    instances. Optional P/Y buffers must be independent of X and the weights.
    """

    def __init__(self, rows, packed_down, packed_up, *, partial=None, output=None):
        validate_rows(rows)
        if not packed_down.is_cuda or torch.version.hip is None:
            raise ValueError('GRReadPrefill requires packed weights on a ROCm device')
        self.rows, self.device = rows, packed_down.device
        for name, weight in (('packed_down', packed_down), ('packed_up', packed_up)):
            _check_buffer(weight, (K * R,), torch.bfloat16, self.device, name)
        props = torch.cuda.get_device_properties(self.device)
        if props.gcnArchName.split(':', 1)[0] != 'gfx942':
            raise ValueError('GRReadPrefill currently targets gfx942')
        self.w_down, self.w_up = packed_down, packed_up
        self.config = select_prefill_config(rows, props.multi_processor_count)
        self.down_config = self.config[:4]
        self.up_config = self.config[4:6]
        self.partial = partial if partial is not None else torch.empty(
            (rows, R), dtype=torch.bfloat16, device=self.device)
        self.output = output if output is not None else torch.empty(
            (rows, H), dtype=torch.bfloat16, device=self.device)
        _check_buffer(self.partial, (rows, R), torch.bfloat16, self.device, 'partial')
        _check_buffer(self.output, (rows, H), torch.bfloat16, self.device, 'output')
        self.down = self.up = self.dispatch = None
        if rows:
            # FlyDSL 0.3.2 artifacts own device-specific HIP module handles.
            # Hints participate in its cache key; these tags do not change codegen.
            device_hints = {'gr_read_device': self.device.index, 'gr_read_arch': props.gcnArchName}
            with torch.cuda.device(self.device), CompilationContext.compile_hints(device_hints):
                # compile() executes once: all example tensors have their full size.
                x = torch.zeros((rows, K), dtype=torch.bfloat16, device=self.device)
                stream = torch.cuda.current_stream(self.device)
                down, up = _launchers(self.config)
                self.down = flyc.compile(down, x, self.w_down, self.partial, rows, stream)
                self.up = flyc.compile(up, x, self.w_up, self.partial, self.output, rows, stream)
                if props.multi_processor_count == 80 and 48 <= rows <= 512:
                    self.dispatch = flyc.compile(_pair_launcher(self.config), x, self.w_down, self.w_up,
                                                 self.partial, self.output, rows, stream)

    def _check_input(self, x):
        if (x.shape != (self.rows, K) or x.dtype != torch.bfloat16 or x.device != self.device
                or not x.is_contiguous() or x.data_ptr() % 16):
            raise ValueError('input must match prepared rows/device, contiguous aligned BF16 [T,10240]')

    def run_down(self, x):
        """Run Down only, for stage validation/timing."""
        self._check_input(x)
        if self.rows and torch.cuda.current_device() != self.device.index:
            with torch.cuda.device(self.device):
                return self.run_down(x)
        if self.rows:
            self.down(x, self.w_down, self.partial, self.rows, torch.cuda.current_stream(self.device))
        return self.partial

    def run_up(self, x):
        """Run Up using the P already produced for this input."""
        self._check_input(x)
        if self.rows and torch.cuda.current_device() != self.device.index:
            with torch.cuda.device(self.device):
                return self.run_up(x)
        if self.rows:
            self.up(x, self.w_up, self.partial, self.output, self.rows, torch.cuda.current_stream(self.device))
        return self.output

    def __call__(self, x):
        self._check_input(x)
        if self.rows and torch.cuda.current_device() != self.device.index:
            with torch.cuda.device(self.device):
                return self(x)
        if self.rows:
            stream = torch.cuda.current_stream(self.device)
            if self.dispatch is not None:
                self.dispatch(x, self.w_down, self.w_up, self.partial, self.output, self.rows, stream)
            else:
                self.down(x, self.w_down, self.partial, self.rows, stream)
                self.up(x, self.w_up, self.partial, self.output, self.rows, stream)
        return self.output
