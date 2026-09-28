# SPDX-License-Identifier: MIT
"""Preparation and two-launch decode/prefill calls; no test or model dependencies."""
from functools import cache, lru_cache
from threading import Lock
import warnings

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext
import torch

from .common import H, H as HS, K, R, select_prefill_config, validate_rows
from .down import make_down, make_decode_down
from .up import make_up, make_decode_up


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
    if (not isinstance(tensor, torch.Tensor) or tensor.shape != shape or tensor.dtype != dtype
            or tensor.device != device or not tensor.is_contiguous()):
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
            warnings.warn(f"GR read prefill was tuned on gfx942; running on {props.gcnArchName}",
                          RuntimeWarning, stacklevel=2)
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
                if props.multi_processor_count == 80 and 33 <= rows <= 512:
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


@cache
def _decode_pair_launcher(rows):
    down, up = make_decode_down(rows), make_decode_up(rows)

    @flyc.jit
    def launch(X: fx.Tensor, WD: fx.Tensor, WU: fx.Tensor, P: fx.Tensor, Y: fx.Tensor, stream: fx.Stream):
        if fx.const_expr(rows > 0):
            down(X, WD, P, stream)
            up(X, WU, P, Y, stream)

    return launch


class GRReadDecode:
    """Prepared T1..32 decode with compact FP32 P[4,T,320]; weights are shared with prefill."""

    def __init__(self, rows, packed_down, packed_up):
        if not isinstance(rows, int) or isinstance(rows, bool) or not 1 <= rows <= 32:
            raise ValueError("decode supports T=1..32")
        for weight in (packed_down, packed_up):
            if weight.shape != (K * R,) or weight.dtype != torch.bfloat16 or not weight.is_contiguous():
                raise ValueError("expected flat contiguous BF16 packed weights")
            if weight.data_ptr() % 16:
                raise ValueError("packed weight base must be 16-byte aligned")
        if not packed_down.is_cuda or packed_down.device != packed_up.device or torch.version.hip is None:
            raise ValueError("packed weights must share a ROCm device")
        self.rows, self.device = rows, packed_down.device
        props = torch.cuda.get_device_properties(self.device)
        if props.gcnArchName.split(":")[0] != "gfx942":
            warnings.warn(f"GR read decode was tuned on gfx942; running on {props.gcnArchName}",
                          RuntimeWarning, stacklevel=2)
        self.w_down, self.w_up = packed_down, packed_up
        # rows 是输入 tensor / 捕获图的固定行数，不是 replay 时变化的 live token 数。
        self.partial = torch.empty(4 * rows * R, dtype=torch.float32, device=self.device)
        self.output = torch.empty((rows, HS), dtype=torch.bfloat16, device=self.device)
        # HIP module handles in FlyDSL 0.3.2 are device-specific.
        device_hints = {'gr_read_device': self.device.index, 'gr_read_arch': props.gcnArchName}
        with torch.cuda.device(self.device), CompilationContext.compile_hints(device_hints):
            # flyc.compile executes once: all example buffers must be valid.
            x = torch.zeros(rows * K, dtype=torch.bfloat16, device=self.device)
            self.dispatch = flyc.compile(
                _decode_pair_launcher(rows), x, self.w_down, self.w_up, self.partial,
                self.output.view(-1), torch.cuda.current_stream(self.device),
            )

    def __call__(self, x):
        if x.shape != (self.rows, K) or x.dtype != torch.bfloat16 or x.device != self.device:
            raise ValueError("input must match the prepared rows, dtype and device")
        if not x.is_contiguous():
            raise ValueError("input must be contiguous")
        if torch.cuda.current_device() != self.device.index:
            with torch.cuda.device(self.device):
                return self(x)
        self.dispatch(x.view(-1), self.w_down, self.w_up, self.partial, self.output.view(-1),
                      torch.cuda.current_stream(self.device))
        return self.output


# Only compiled launch code is cached. Call tensors/workspaces never belong here.
_compiled_calls = {}
_compile_lock = Lock()


@cache
def _device_properties(device):
    props = torch.cuda.get_device_properties(device)
    if props.gcnArchName.split(':', 1)[0] != 'gfx942':
        warnings.warn(f'GR read was tuned on gfx942; running on {props.gcnArchName}',
                      RuntimeWarning, stacklevel=3)
    return props


@lru_cache(maxsize=128)
def _prefill_config(rows, compute_units):
    """Cache small host-side metadata, without specializing GPU code for each T."""
    return select_prefill_config(rows, compute_units)


def _compile_call(rows, config, paired, x, packed_down, packed_up, partial, output, stream):
    # compile() executes the call once. Use the full, initialized caller inputs
    # and return that first result without launching a second time.
    if rows <= 32:
        return (flyc.compile(_decode_pair_launcher(rows), x.view(-1), packed_down,
                             packed_up, partial, output.view(-1), stream),)
    if paired:
        return (flyc.compile(_pair_launcher(config), x, packed_down, packed_up,
                             partial, output, rows, stream),)
    down, up = _launchers(config)
    return (flyc.compile(down, x, packed_down, partial, rows, stream),
            flyc.compile(up, x, packed_up, partial, output, rows, stream))


def gr_read(x, packed_down, packed_up, *, output=None):
    """Run BF16 GR read with shared packed weights and per-call intermediate storage.

    X is contiguous, 16-byte-aligned [T,10240] on a ROCm device; weights come from
    prepare_weights(). T1..32 uses exact-row decode. Prefill starts at T33, with
    actual-row configuration selection and runtime rows. Configurations share
    compiled code across T. No input rows are padded or copied.
    The result is [T,2560]; if output is supplied, it is written and returned.

    Warm the needed shapes/configurations on each device before Graph capture.
    Only code is cached: each call owns its P and, by default, its Y. Graph replay
    reuses captured buffers; live-token counts do not change their row strides.
    The caller manages normal Torch stream dependencies for inputs and outputs.
    """
    if not isinstance(x, torch.Tensor) or x.ndim != 2 or x.shape[1] != K:
        raise ValueError('input must be a BF16 tensor with shape [T,10240]')
    rows, device = x.shape[0], x.device
    if not x.is_cuda or torch.version.hip is None:
        raise ValueError('gr_read requires inputs on a ROCm device')
    _check_buffer(x, (rows, K), torch.bfloat16, device, 'input')
    for name, weight in (('packed_down', packed_down), ('packed_up', packed_up)):
        _check_buffer(weight, (K * R,), torch.bfloat16, device, name)
    if output is not None:
        _check_buffer(output, (rows, H), torch.bfloat16, device, 'output')
        if rows:
            # Shapes, BF16 dtype and contiguity are already checked. Read each
            # pointer once instead of repeatedly querying tensor sizes/strides.
            begin = output.data_ptr()
            end = begin + rows * H * 2
            for tensor, elements in ((x, rows * K), (packed_down, K * R), (packed_up, K * R)):
                address = tensor.data_ptr()
                if begin < address + elements * 2 and address < end:
                    raise ValueError('output must not overlap the input or packed weights')
    if torch.is_grad_enabled() and any(t.requires_grad for t in
            (x, packed_down, packed_up) + (() if output is None else (output,))):
        raise ValueError('gr_read is inference-only; use torch.no_grad() or inference_mode()')
    if rows == 0:
        return output if output is not None else torch.empty((0, H), dtype=x.dtype, device=device)
    if torch.cuda.current_device() != device.index:
        with torch.cuda.device(device):
            return gr_read(x, packed_down, packed_up, output=output)

    props = _device_properties(device)
    config = rows if rows <= 32 else _prefill_config(rows, props.multi_processor_count)
    paired = rows <= 32 or (props.multi_processor_count == 80 and rows <= 512)
    key = (device.index, props.gcnArchName, config, paired)
    kernels = _compiled_calls.get(key)
    if kernels is None and torch.cuda.is_current_stream_capturing():
        raise RuntimeError('gr_read must be warmed on this device before Graph capture')

    if output is None:
        output = torch.empty((rows, H), dtype=x.dtype, device=device)
    partial = torch.empty((4 * rows * R,) if rows <= 32 else (rows, R),
                          dtype=torch.float32 if rows <= 32 else torch.bfloat16, device=device)
    stream = torch.cuda.current_stream(device)
    if kernels is None:
        with _compile_lock:
            kernels = _compiled_calls.get(key)
            if kernels is None:
                hints = {'gr_read_device': device.index, 'gr_read_arch': props.gcnArchName}
                with CompilationContext.compile_hints(hints):
                    kernels = _compile_call(rows, config, paired, x, packed_down, packed_up,
                                            partial, output, stream)
                _compiled_calls[key] = kernels
                return output
    if rows <= 32:
        kernels[0](x.view(-1), packed_down, packed_up, partial, output.view(-1), stream)
    elif paired:
        kernels[0](x, packed_down, packed_up, partial, output, rows, stream)
    else:
        down, up = kernels
        down(x, packed_down, partial, rows, stream)
        up(x, packed_up, partial, output, rows, stream)
    return output
