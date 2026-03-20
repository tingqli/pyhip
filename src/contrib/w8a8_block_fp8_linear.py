from typing import TYPE_CHECKING, Callable, List, Optional, Tuple

import pyhip

import torch
import aiter
from aiter import gemm_a8w8_bpreshuffle, get_hip_quant
from aiter.ops.triton.gemm_a8w8_blockscale import gemm_a8w8_blockscale as aiter_gemm_a8w8_blockscale

aiter_per1x128_quant = get_hip_quant(aiter.QuantType.per_1x128)

from .gemm_fp8 import gemm_8wave_fp8bf16fp16
from .gluon.gemm_splitk import gemm_splitk
from pyhip import div_up

__all__ = ["w8a8_block_fp8_linear"]

# sglang/python/sglang/srt/layers/quantization/fp8_utils.py
def w8a8_block_fp8_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    block_size: List[int],
    weight_scale: torch.Tensor,
    input_scale: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    b_preshuffle = True,
    method = "auto"
) -> torch.Tensor:
    if method == "aiter":
        assert input_scale is None
        # assert input_scale is None
        input_2d = input.view(-1, input.shape[-1])
        output_shape = [*input.shape[:-1], weight.shape[0]]
        # if input_scale not None, input is quanted
        if input_scale is not None:
            q_input = input_2d
            x_scale = input_scale

        else:
            q_input, x_scale = aiter_per1x128_quant(input_2d, quant_dtype=aiter.dtypes.fp8)

        #print("aiter_gemm_a8w8_blockscale: ", input.dtype, q_input.dtype, q_input.shape, weight.dtype, weight.shape, block_size, input_scale is None)
        output = aiter_gemm_a8w8_blockscale(
            q_input,
            weight,
            x_scale,
            weight_scale,
            dtype=torch.bfloat16 if input_scale is not None else input.dtype,
        )
        if bias is not None:
            output += bias

        return output.to(
            dtype=torch.bfloat16 if input_scale is not None else input_2d.dtype
        ).view(*output_shape)


    assert input_scale is None
    assert block_size == [128, 128] or block_size == (128, 128), block_size
    assert input.dtype == torch.bfloat16
    K = input.shape[-1]
    M = input.numel() // K
    N = weight.shape[0]
    output = torch.empty([*input.shape[:-1], N], dtype = input.dtype, device = input.device)

    if (M <= 512 and method != "jit") or method == "gluon":
        #if input.device.index == 0:
        #    print("=========================input ", input.shape, input.dtype, input.device)
        #    print("=========================weight ", weight.shape, weight.dtype, weight.device)
        #    print("=========================output ", output.shape, output.dtype, output.device)
        #    print("=========================weight_scale ", weight_scale.shape, weight_scale.dtype, weight_scale.device, b_preshuffle)

        if gemm_splitk(input, weight, output, weight_scale, b_preshuffle):
            if bias is not None:
                output += bias
            return output

    q_input, x_scale = aiter_per1x128_quant(input.view(M, K), quant_dtype=aiter.dtypes.fp8, transpose_scale=True)

    #print(M,K,N, input.shape, q_input.shape, x_scale.shape)
    #print("pyhip_gemm_a8w8_blockscale: ", input.dtype, q_input.dtype, q_input.shape, weight.dtype, weight.shape, block_size, input_scale is None)
    use_f32_blockscales_128 = True
    wg_M, wg_N = 256, 256
    num_block_M = pyhip.div_up(M, wg_M)
    num_block_N = pyhip.div_up(N, wg_N)
    gemm_8wave_fp8bf16fp16([num_block_M * num_block_N],[64*8],
                    "fp8", b_preshuffle, use_f32_blockscales_128,
                    wg_M, wg_N, N, K,
                    q_input.data_ptr(),
                    weight.data_ptr(),
                    output.data_ptr(),
                    x_scale.data_ptr(), weight_scale.data_ptr(), M)
    if bias is not None:
        output += bias

    return output

