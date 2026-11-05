# SPDX-License-Identifier: MIT
"""Functional GR write prefill entry: one compiled host call launches gate then apply."""
from functools import cache
from threading import Lock
import warnings

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext
import torch

from .apply import make_apply
from .common import C, EPS, H, K, launch_grids, workspace_words
from .gate import make_gate


@cache
def _launcher(eps):
    gate, apply = make_gate(), make_apply(eps)

    @flyc.jit
    def launch(Y: fx.Tensor, R: fx.Tensor, N: fx.Tensor, WP: fx.Tensor, G: fx.Tensor, A: fx.Tensor,
               OUT: fx.Tensor, NORMED: fx.Tensor, rows: fx.Int32, gate_grid: fx.Int32, apply_grid: fx.Int32,
               stream: fx.Stream):
        gate(N, WP, A, rows, gate_grid, stream)
        apply(Y, R, A, G, OUT, NORMED, rows, apply_grid, stream)

    launch.compile_hints['llvm_options'] = {'vectorize-slp': False}
    return launch


def _check_buffer(tensor, shape, dtype, device, name):
    if (not isinstance(tensor, torch.Tensor) or tensor.shape != shape or tensor.dtype != dtype
            or tensor.device != device or not tensor.is_contiguous()):
        raise ValueError(f'{name} must be contiguous {dtype} {tuple(shape)} on {device}')
    if tensor.numel() and tensor.data_ptr() % 16:
        raise ValueError(f'{name} must have a 16-byte-aligned base')


def _span(tensor):
    begin = tensor.data_ptr()
    return begin, begin + tensor.numel() * tensor.element_size()


# Only compiled launch code is cached; tensors and workspaces never belong here.
_compiled_calls = {}
_compile_lock = Lock()


@cache
def _device_properties(device):
    props = torch.cuda.get_device_properties(device)
    if props.gcnArchName.split(':', 1)[0] != 'gfx942':
        warnings.warn(f'GR write was tuned on gfx942; running on {props.gcnArchName}',
                      RuntimeWarning, stacklevel=3)
    return props


def gr_write(block_output, residual, normed_residual, packed_inject, packed_norm, *, eps=EPS,
             out=None, normed=None):
    """Fused GR write (hc_combine) plus the next per-branch Gemma RMSNorm, prefill path.

    block_output y: BF16 [T, 2560]; residual r and normed_residual n: BF16 [T, 10240];
    packed_inject / packed_norm come from prepare_weights(). For every row m and branch c:

        a[m, c] = 2 * sigmoid(dot(n[m], W[c]) / 4)        (FP32 accumulation)
        out[m, c*2560:(c+1)*2560] = bf16(r + a * y)        (FP32 FMA, round to nearest even)
        normed = bf16((out * rsqrt(mean(out^2) + eps)) * (1 + w))   per 2560-wide branch

    Returns (out, normed), both BF16 [T, 10240]; supplied out/normed are written and returned.
    Inputs must be contiguous and 16-byte aligned; outputs must not overlap inputs, weights or
    each other. Every nonempty call launches the gate then the apply kernel through one compiled
    host entry on the current stream, with a per-call FP32 workspace ([T, 4] gate values plus the
    apply row counter that the gate kernel zeroes). Warm up on each device before Graph capture.
    """
    if not isinstance(residual, torch.Tensor) or residual.ndim != 2 or residual.shape[1] != K:
        raise ValueError('residual must be a BF16 tensor with shape [T, 10240]')
    rows, device = residual.shape[0], residual.device
    if not residual.is_cuda or torch.version.hip is None:
        raise ValueError('gr_write requires inputs on a ROCm device')
    _check_buffer(block_output, (rows, H), torch.bfloat16, device, 'block_output')
    _check_buffer(residual, (rows, K), torch.bfloat16, device, 'residual')
    _check_buffer(normed_residual, (rows, K), torch.bfloat16, device, 'normed_residual')
    _check_buffer(packed_inject, (C * K,), torch.bfloat16, device, 'packed_inject')
    _check_buffer(packed_norm, (K,), torch.float32, device, 'packed_norm')
    eps = float(eps)
    if not eps > 0.0:
        raise ValueError('eps must be positive')
    inputs = (block_output, residual, normed_residual, packed_inject, packed_norm)
    outputs = []
    for name, tensor in (('out', out), ('normed', normed)):
        if tensor is not None:
            _check_buffer(tensor, (rows, K), torch.bfloat16, device, name)
            outputs.append((name, tensor))
    if rows and outputs:
        spans = [_span(t) for t in inputs]
        for name, tensor in outputs:
            begin, end = _span(tensor)
            if any(begin < other_end and other_begin < end for other_begin, other_end in spans):
                raise ValueError(f'{name} must not overlap the inputs or prepared weights')
        if len(outputs) == 2:
            (begin, end), (other_begin, other_end) = _span(out), _span(normed)
            if begin < other_end and other_begin < end:
                raise ValueError('out and normed must not overlap')
    if torch.is_grad_enabled() and any(t.requires_grad for t in inputs + tuple(t for _, t in outputs)):
        raise ValueError('gr_write is inference-only; use torch.no_grad() or inference_mode()')
    if out is None:
        out = torch.empty((rows, K), dtype=torch.bfloat16, device=device)
    if normed is None:
        normed = torch.empty((rows, K), dtype=torch.bfloat16, device=device)
    if rows == 0:
        return out, normed
    if torch.cuda.current_device() != device.index:
        with torch.cuda.device(device):
            return gr_write(block_output, residual, normed_residual, packed_inject, packed_norm, eps=eps,
                            out=out, normed=normed)

    props = _device_properties(device)
    key = (device.index, props.gcnArchName, eps)
    dispatch = _compiled_calls.get(key)
    if dispatch is None and torch.cuda.is_current_stream_capturing():
        raise RuntimeError('gr_write must be warmed on this device before Graph capture')
    gate_grid, apply_grid = launch_grids(rows, props.multi_processor_count)
    workspace = torch.empty(workspace_words(rows), dtype=torch.float32, device=device)
    stream = torch.cuda.current_stream(device)
    args = (block_output, residual, normed_residual, packed_inject, packed_norm, workspace, out, normed,
            rows, gate_grid, apply_grid, stream)
    if dispatch is None:
        with _compile_lock:
            dispatch = _compiled_calls.get(key)
            if dispatch is None:
                hints = {'gr_write_device': device.index, 'gr_write_arch': props.gcnArchName}
                with CompilationContext.compile_hints(hints):
                    # compile() executes the call once with these full, valid buffers.
                    _compiled_calls[key] = flyc.compile(_launcher(eps), *args)
                return out, normed
    dispatch(*args)
    return out, normed
