"""
Ref from python/sglang/jit_kernel/fused_sigmoid_mul_add_gluon.py 
"""

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from functools import cache
# CDNA wave=64 fixed. Layout config: BLOCK_SIZE = SIZE_PER_THREAD * WARP_SIZE * WARPS_PER_CTA.
WARP_SIZE = 64
SIZE_PER_THREAD = 2
WARPS_PER_CTA = 2
BLOCK_SIZE = SIZE_PER_THREAD * WARP_SIZE * WARPS_PER_CTA  # 4096

# WARP_SIZE = 64
# SIZE_PER_THREAD = 4
# WARPS_PER_CTA = 1
# BLOCK_SIZE = SIZE_PER_THREAD * WARP_SIZE * WARPS_PER_CTA  # 4096

_layout = gl.BlockedLayout(
    size_per_thread=[SIZE_PER_THREAD],
    threads_per_warp=[WARP_SIZE],
    warps_per_cta=[WARPS_PER_CTA],
    order=[0],
)

"""
@gluon.jit
def _fused_sigmoid_mul_add_gluon_kernel_cdna3(
    gate_ptr,
    shared_ptr,
    out_ptr,
    hidden_size,
    shared_stride_row,
    out_stride_row,
    BLOCK_SIZE: gl.constexpr,
):
    row = gl.program_id(0)
    col_block = gl.program_id(1)

    col_offsets = col_block * BLOCK_SIZE + gl.arange(0, BLOCK_SIZE, layout=_col_layout)
    mask = col_offsets < hidden_size

    zeros_offsets = gl.zeros([BLOCK_SIZE], dtype=gl.int32, layout=_col_layout)
    gate_val = gl.amd.cdna3.buffer_load(gate_ptr + row, zeros_offsets).to(gl.float32)
    sig = 1.0 / (1.0 + gl.exp(-gate_val))

    shared_offsets = row * shared_stride_row + col_offsets
    out_offsets = row * out_stride_row + col_offsets

    shared_val = gl.amd.cdna3.buffer_load(shared_ptr, shared_offsets, mask=mask).to(gl.float32)
    out_val = gl.amd.cdna3.buffer_load(out_ptr, out_offsets, mask=mask).to(gl.float32)

    result = (out_val + sig * shared_val).to(gl.bfloat16)
    gl.amd.cdna3.buffer_store(result, out_ptr, out_offsets, mask=mask)
"""

@gluon.jit
def _fused_sigmoid_mul_add_gluon_kernel(
    gate_ptr,
    shared_ptr,
    out_ptr,
    hidden_size,
    shared_stride_row,
    out_stride_row,
    BLOCK_SIZE: gl.constexpr,
):
    row = gl.program_id(0)
    col_block = gl.program_id(1)

    col_offsets = col_block * BLOCK_SIZE + gl.arange(0, BLOCK_SIZE, layout=_layout)
    mask = col_offsets < hidden_size

    #gate_val = gl.load(gate_ptr + row).to(gl.float32)
    #zeros_offsets = gl.zeros([BLOCK_SIZE], dtype=gl.int32, layout=layout) cdan3 version
    temp = gl.arange(0, BLOCK_SIZE, layout=_layout)
    zeros_offsets = (temp * 0).to(gl.int32)
    gate_val = gl.amd.cdna4.buffer_load(gate_ptr+ row,zeros_offsets).to(gl.float32)
    sig = 1.0 / (1.0 + gl.exp(-gate_val))

    shared_offsets = row * shared_stride_row + col_offsets
    out_offsets = row * out_stride_row + col_offsets

    shared_val =  gl.amd.cdna4.buffer_load(shared_ptr, shared_offsets, mask=mask).to(gl.float32)
    out_val = gl.amd.cdna4.buffer_load(out_ptr, out_offsets, mask=mask).to(gl.float32)

    result = (out_val + sig * shared_val).to(gl.bfloat16)
    #gl.store(out_ptr + out_offsets, result, mask=mask)
    gl.amd.cdna4.buffer_store(result, out_ptr, out_offsets, mask=mask)

def fused_sigmoid_mul_add_gluon(
    gate: torch.Tensor,
    shared_output: torch.Tensor,
    final_hidden_states: torch.Tensor,
) -> None:
    """Fused sigmoid-mul-add: final_hidden_states += sigmoid(gate) * shared_output.

    Args:
        gate: [num_tokens, 1] (or flattenable to 1D).
        shared_output: [num_tokens, hidden_size].
        final_hidden_states: [num_tokens, hidden_size], modified in-place.
    """
   # num_tokens, hidden_size = shared_output.shape
    num_tokens,hidden_size = 4, 1024
    gate_flat = gate.view(-1)

    num_col_blocks = triton.cdiv(hidden_size, BLOCK_SIZE)
    grid = (num_tokens, num_col_blocks)

    _fused_sigmoid_mul_add_gluon_kernel[grid](
        gate_flat,
        shared_output,
        final_hidden_states,
        hidden_size,
        shared_output.stride(0),
        final_hidden_states.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=WARPS_PER_CTA,
    )
