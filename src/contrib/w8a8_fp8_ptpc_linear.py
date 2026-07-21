from typing import TYPE_CHECKING, Callable, List, Optional, Tuple

import pyhip

import torch
import aiter
from aiter import get_hip_quant
#aiter ck tile
from aiter import gemm_a8w8_CK
#aiter preshuffle flydsl
from aiter import gemm_a8w8_bpreshuffle

from .gemm_fp8 import gemm_8wave_fp8bf16fp16
from pyhip import div_up
from aiter import dtypes
# flydsl 8 wave import
import sys, os, flydsl
import flydsl.compiler.jit_function
repo_root = os.path.dirname(os.path.dirname(os.path.dirname(flydsl.__file__)))  # .../FlyDSL
sys.path.insert(0, repo_root)
from kernels.gemm.fp8_gemm_8wave import compile_fp8_gemm_8w
import flydsl.compiler as flyc
__all__ = ["w8a8_ptpc_fp8_linear", "flyc_w8a8_ptpc_fp8"]

# Cache compiled FlyDSL kernels keyed by (M, N, K, b_preshuffle) so run_perftest's
# repeated invocations reuse one compile instead of recompiling every call.
_FLY_KERNEL_CACHE = {}

def flyc_w8a8_ptpc_fp8(M, N, K, tile_m, tile_n, b_preshuffled, x_q, wei_q, x_scale, wei_scale):
    c_out = torch.empty((M, N), dtype = torch.bfloat16)
    launch_fn = compile_fp8_gemm_8w(
    K=K,
    BLOCK_M=tile_m,
    BLOCK_N=tile_n,
    b_preshuffled=b_preshuffled,)

    def _as_i8(t: torch.Tensor) -> torch.Tensor:
        return t.view(torch.int8) if "float8" in str(t.dtype) else t
    def _args(c, a, b, sa, sb):
        b_flat = _as_i8(b).contiguous().view(-1)
        sa_flat = sa.contiguous().view(-1)
        sb_flat = sb.contiguous().view(-1)
        b_flat = flyc.from_torch_tensor(b_flat)
        sa_flat = flyc.from_torch_tensor(sa_flat)
        sb_flat = flyc.from_torch_tensor(sb_flat)
        return (
            _as_i8(a).contiguous().view(-1),
            b_flat,
            c.contiguous().view(-1),
            sa_flat,
            sb_flat,
            M,
            N,
            torch.cuda.current_stream(),
        )
    compiled = flyc.compile(launch_fn, *_args(c_out, x_q, wei_q, x_scale, wei_scale))
    print(type(compiled))
    return compiled
    
# sglang/python/sglang/srt/layers/quantization/fp8_utils.py
def w8a8_ptpc_fp8_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    b_preshuffle = False,
    method = "auto",
    out_dtype = torch.bfloat16,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    assert input_scale is not None
    assert input_scale is not None
    assert bias is None
    assert b_preshuffle is False
    # [M, K]
    input_2d = input.view(-1, input.shape[-1])
    # [M , N]
    output_shape = [*input.shape[:-1], weight.shape[0]]
    assert out_dtype == torch.bfloat16
    K = input.shape[-1]
    M = input.numel() // K
    N = weight.shape[0]
    
    if method == "aiter":
        # bias = None, dtype = aiter.dtypes.bf16, splitK = None
        output = aiter.gemm_a8w8_CK(input_2d, weight, input_scale, weight_scale)

        return output.to(
            dtype=torch.bfloat16
        ).view(*output_shape)

    elif method == "fly":
        c_out = torch.empty((M, N), dtype = out_dtype)
        def _as_i8(t: torch.Tensor) -> torch.Tensor:
            return t.view(torch.int8) if "float8" in str(t.dtype) else t
        def _args(c, a, b, sa, sb):
            b_flat = _as_i8(b).contiguous().view(-1)
            sa_flat = sa.contiguous().view(-1)
            sb_flat = sb.contiguous().view(-1)
            b_flat = flyc.from_torch_tensor(b_flat)
            sa_flat = flyc.from_torch_tensor(sa_flat)
            sb_flat = flyc.from_torch_tensor(sb_flat)
            return (
                _as_i8(a).contiguous().view(-1),
                b_flat,
                c.contiguous().view(-1),
                sa_flat,
                sb_flat,
                M,
                N,
                torch.cuda.current_stream(),
            )
        tileM = 256
        tileN = 256
        key = (M, N, K, bool(b_preshuffle), tileM, tileN)
        fly_kernel = _FLY_KERNEL_CACHE.get(key)
        if fly_kernel is None:
            launch_fn = compile_fp8_gemm_8w(
                K=K, BLOCK_M=256, BLOCK_N=256, b_preshuffled=b_preshuffle
            )
            fly_kernel = flyc.compile(
                launch_fn, *_args(c_out, input_2d, weight, input_scale, weight_scale)
            )
            _FLY_KERNEL_CACHE[key] = fly_kernel
        fly_kernel(*_args(c_out, input_2d, weight, input_scale, weight_scale))
        return c_out

    wg_M, wg_N = 256, 256
    num_block_M = pyhip.div_up(M, wg_M)
    num_block_N = pyhip.div_up(N, wg_N)
    y1 = torch.empty((M, N), dtype = out_dtype)

    if 0:
        gemm_8wave_fp8bf16fp16([num_block_N*num_block_M],[64*8], "fp8", b_preshuffle, False, False,
                    wg_M, wg_N, N, K, input.data_ptr(), weight.data_ptr(), y1.data_ptr(),
                    None, None, M)
    else:
    
        gemm_8wave_fp8bf16fp16([num_block_N*num_block_M],[64*8], "fp8", b_preshuffle, False, True,
                    wg_M, wg_N, N, K, input.data_ptr(), weight.data_ptr(), y1.data_ptr(),
                    input_scale.data_ptr(), weight_scale.data_ptr(), M)
    return y1

