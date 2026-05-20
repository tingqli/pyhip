import os
os.environ['PYHIP_JIT_LOG'] = '0'

import torch
import pytest
from typing import Optional

from pyhip import cudaPerf, torchPerf, calc_diff, div_up
from pyhip.contrib.moe_gemm_mxfp4 import *
from pyhip.contrib.moe import *

import aiter
from aiter.utility import fp4_utils

USE_FP4_SHUFFLE_WEIGHT = 1

DDD = int(os.getenv("DDD", "0"))

def is_arch_type(arch):
    props = torch.cuda.get_device_properties()
    return arch in props.gcnArchName

def get_fp8type():
    return torch.float8_e4m3fn if is_arch_type('950') else torch.float8_e4m3fnuz

def get_fp4type_if_valid():
    return torch.float4_e2m1fn_x2 if is_arch_type('950') else None

prec_bf16 = (torch.bfloat16, aiter.QuantType.No)
prec_fp8_ptpc = (get_fp8type(), aiter.QuantType.per_Token)
prec_fp8_b = (get_fp8type(), aiter.QuantType.per_1x128)
prec_fp8_t = (get_fp8type(), aiter.QuantType.per_Tensor)
prec_mxfp4 = (get_fp4type_if_valid(), aiter.QuantType.per_1x32)

quant2str_dict = {
    aiter.QuantType.per_Token: 'per_Token',
    aiter.QuantType.per_1x128: 'per_1x128',
    aiter.QuantType.per_Tensor: 'per_Tensor',
    aiter.QuantType.per_1x32: 'per_1x32',
}

def _run_aiter(hidden_states,
            w1,  # [expert(local_expert:EP), inter_dim*2, dim] N,K
            w2,  # [expert(local_expert:EP), dim, inter_dim]
            topk_weight,
            topk_ids,
            w1_scale: Optional[torch.tensor] = None,  # [expert(local_expert:EP), inter_dim, 1]
            w2_scale: Optional[torch.tensor] = None,  # [expert(local_expert:EP), model_dim, 1]
            quant_type = aiter.QuantType.No):
    from aiter.fused_moe import fused_moe
    """
    from aiter import QuantType
    if w1.dtype == torch.float4_e2m1fn_x2:
        quant_type = QuantType.per_1x32
    elif w1.dtype == torch.bfloat16:
        quant_type = QuantType.No
    elif not fp8_ptpc and wei_is_fp8(w1.dtype):
        quant_type = QuantType.per_128x128
    else:
        quant_type = QuantType.per_Token
    """
    return fused_moe(
        hidden_states,
        w1,
        w2,
        topk_weight,
        topk_ids,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        quant_type=quant_type
    )

# https://github.com/huggingface/transformers/blob/1fed6166c00b800330fcda8494f78cbcad8e4e3b/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py#L235-L263
def get_torch_ref(hidden_states, w1, w2, topk_weight, topk_ids):
    batch_size, hidden_dim = hidden_states.shape
    E, N1, K1 = w1.shape
    INTER_SIZE = N1 // 2
    final_hidden_states = torch.zeros(
        (batch_size, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
    )

    # One hot encode the selected experts to create an expert mask
    # this will be used to easily index which expert is going to be sollicitated
    expert_mask = torch.nn.functional.one_hot(topk_ids.to(dtype=torch.long), num_classes=E).permute(2, 1, 0)

    # Loop over all available experts in the model and perform the computation on each expert
    for expert_idx in range(E):
        def expert_forward(n, x):
            gate_proj = w1[n, 0 : INTER_SIZE].t()
            up_proj = w1[n, INTER_SIZE :,].t()
            down_proj = w2[n].t()
            return (torch.nn.functional.silu(x @ gate_proj) * (x @ up_proj)) @ down_proj
        idx, top_x = torch.where(expert_mask[expert_idx])

        # Index the correct hidden states and compute the expert hidden state for
        # the current expert. We need to make sure to multiply the output hidden
        # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
        current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
        current_hidden_states = expert_forward(expert_idx, current_state) * topk_weight[top_x, idx, None]

        # However `index_add_` only support torch tensors for indexing so we'll use
        # the `top_x` tensor here.
        final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
    return final_hidden_states

def wei_is_fp8(weight_type):
    return weight_type == torch.float8_e4m3fn or weight_type == torch.float8_e4m3fnuz

from dataclasses import dataclass
import aiter
from aiter.ops.shuffle import shuffle_weight
from aiter.fused_moe import moe_sorting
from aiter.utility import fp4_utils
from aiter.ops.quant import pertoken_quant

ext_topk_ids = None

def quant_expert_weights(w1, quant_type, dtype):
    if quant_type == aiter.QuantType.per_Token:
        torch_quant = aiter.get_torch_quant(aiter.QuantType.per_Token)
        w1_qt, w1s = torch_quant(w1, quant_dtype=dtype)
        w1_ref = (w1_qt.to(dtype=w1.dtype) * w1s).to(dtype=w1.dtype)
        return w1_qt, w1s, w1_ref

    if quant_type == aiter.QuantType.per_Tensor:
        fmax = torch.finfo(dtype).max
        w1s = w1.float().abs().amax(dim=(1,2)) / fmax
        w1_qt = (
            (w1.float() / w1s.view(E, 1, 1)).clamp(-fmax, fmax).to(dtype)
        )
        w1_ref = (w1_qt.to(dtype=w1.dtype) * w1s.view(E, 1, 1)).to(dtype=w1.dtype)
        return w1_qt, w1s, w1_ref
    assert 0, quant_type

@dataclass
class TestCase:
    # all MOE problem specification is data member
    TILE_M:int
    TILE_N:int
    HIDDEN_SIZE:int
    INTER_SIZE_TP:int
    E:int
    TOPK:int
    run_count:int = 10
    INTER_SIZE_TP_ADJ:int = 0

    perf:list = None

    def __call__(self, kernel_type, weight_type, quant_type, B=1, run_count=0):
        run_count = self.run_count if run_count <= 0 else run_count
        fp8_quant_type = quant_type
        TILE_M=self.TILE_M
        TILE_N=self.TILE_N
        HIDDEN_SIZE = self.HIDDEN_SIZE
        INTER_SIZE_TP = self.INTER_SIZE_TP
        
        # adjust INTER_SIZE_TP
        if kernel_type == 'aiter' and weight_type == torch.float4_e2m1fn_x2 and INTER_SIZE_TP % 128 != 0:
            INTER_SIZE_TP = div_up(INTER_SIZE_TP, 128) * 128

        if (weight_type == torch.float8_e4m3fn or weight_type == torch.float8_e4m3fnuz) and INTER_SIZE_TP % 128 != 0:
            #INTER_SIZE_TP = div_up(INTER_SIZE_TP, 128) * 128
            pass

        self.INTER_SIZE_TP_ADJ = INTER_SIZE_TP

        E=self.E
        TOPK=self.TOPK
        global ext_topk_ids
        BUF_COPY = 32
        hidden_states = (torch.randn([BUF_COPY, B, HIDDEN_SIZE], dtype=torch.bfloat16) + 1)*0.001
        if weight_type == torch.bfloat16:
            w_ = torch.randn([E, INTER_SIZE_TP * 2, HIDDEN_SIZE], dtype=weight_type)
            w1_ref = w_
            w1 = [w_.clone() for _ in range(BUF_COPY)]
            w_ = torch.randn([E, HIDDEN_SIZE, INTER_SIZE_TP], dtype=weight_type)
            w2_ref = w_
            w2 = [w_.clone() for _ in range(BUF_COPY)]
            w1_scale = [None] * BUF_COPY
            w2_scale = [None] * BUF_COPY
        elif weight_type == torch.float4_e2m1fn_x2:
            # w_ = torch.randn([E, INTER_SIZE_TP * 2, HIDDEN_SIZE], dtype=torch.bfloat16)
            w_ = torch.randn([E, INTER_SIZE_TP * 2, HIDDEN_SIZE], dtype=torch.bfloat16)
            w1_qt, w1_qt_scale_ = aiter.get_torch_quant(aiter.QuantType.per_1x32)(w_, quant_dtype=weight_type)

            #w1_qt_scale_[...] = 1.0
            w1_f32 = fp4_utils.mxfp4_to_f32(w1_qt).to(dtype=torch.bfloat16).reshape(E, INTER_SIZE_TP * 2, HIDDEN_SIZE // 32, 32)
            w1_scale_f32 = fp4_utils.e8m0_to_f32(w1_qt_scale_).to(dtype=torch.bfloat16).reshape(E, INTER_SIZE_TP * 2, HIDDEN_SIZE // 32, 1)
            w1_ref = (w1_f32 * w1_scale_f32).reshape(E, INTER_SIZE_TP * 2, HIDDEN_SIZE)
            w1_qt_scale = fp4_utils.e8m0_shuffle(w1_qt_scale_)
            if USE_FP4_SHUFFLE_WEIGHT:
                w1 = [shuffle_weight(w1_qt) for _ in range(BUF_COPY)]
            else:
                w1 = [w1_qt.clone() for _ in range(BUF_COPY)]
            w1_scale = [w1_qt_scale.clone() for _ in range(BUF_COPY)]
            # w_ = torch.randn([E, HIDDEN_SIZE, INTER_SIZE_TP], dtype=torch.bfloat16)
            w_ = torch.randn([E, HIDDEN_SIZE, INTER_SIZE_TP], dtype=torch.bfloat16)
            w2_qt, w2_qt_scale_ = aiter.get_torch_quant(aiter.QuantType.per_1x32)(w_, quant_dtype=weight_type)
            #w2_qt_scale_[...] = 1.0

            w2_f32 = fp4_utils.mxfp4_to_f32(w2_qt).to(dtype=torch.bfloat16).reshape(E, HIDDEN_SIZE, INTER_SIZE_TP // 32, 32)
            w2_scale_f32 = fp4_utils.e8m0_to_f32(w2_qt_scale_).to(dtype=torch.bfloat16).reshape(E, HIDDEN_SIZE, INTER_SIZE_TP // 32, 1)
            w2_ref = (w2_f32 * w2_scale_f32).reshape(E, HIDDEN_SIZE, INTER_SIZE_TP)
            # pad scale
            w2_qt_scale_pad = torch.zeros(w2_qt_scale_.shape[0], div_up(w2_qt_scale_.shape[1], 8) * 8, dtype=w2_qt_scale_.dtype)
            w2_qt_scale_pad[:, :w2_qt_scale_.shape[1]] = w2_qt_scale_
            w2_qt_scale = fp4_utils.e8m0_shuffle(w2_qt_scale_pad)
            if USE_FP4_SHUFFLE_WEIGHT:
                w2 = [shuffle_weight(w2_qt) for _ in range(BUF_COPY)]
            else:
                w2 = [w2_qt.clone() for _ in range(BUF_COPY)]
            w2_scale = [w2_qt_scale.clone() for _ in range(BUF_COPY)]
        elif (fp8_quant_type == aiter.QuantType.per_Token or fp8_quant_type == aiter.QuantType.per_Tensor) and wei_is_fp8(weight_type):
            w_ = torch.randn([E, INTER_SIZE_TP * 2, HIDDEN_SIZE], dtype=torch.bfloat16)
            w1_qt, w1_qt_scale, w1_ref = quant_expert_weights(w_, fp8_quant_type, weight_type)
            w1 = [w1_qt.clone() for _ in range(BUF_COPY)]
            for e in w1: e.is_shuffled = True
            w1_scale = [w1_qt_scale.clone() for _ in range(BUF_COPY)]
            w_ = torch.randn([E, HIDDEN_SIZE, INTER_SIZE_TP], dtype=torch.bfloat16)
            w2_qt, w2_qt_scale, w2_ref = quant_expert_weights(w_, fp8_quant_type, weight_type)
            w2 = [w2_qt.clone() for _ in range(BUF_COPY)]
            for e in w2: e.is_shuffled = True
            w2_scale = [w2_qt_scale.clone() for _ in range(BUF_COPY)]
        elif wei_is_fp8(weight_type):
            def weight_per_128x128_quant(weight, quant_dtype):
                E, dim1, dim2 = weight.shape
                assert dim1 % 128 == 0 and dim2 % 128 == 0, f"weight shape {weight.shape} is not aligned to 128 for per 128x128 quantization"

                weight_blocks = weight.view(
                    E, dim1 // 128, 128, dim2 // 128, 128
                )  # [E, num_blocks_dim1, 128, num_blocks_dim2, 128]
                weight_blocks = weight_blocks.permute(
                    0, 1, 3, 2, 4
                ).contiguous()  # [E, num_blocks_dim1, num_blocks_dim2, 128, 128]
                weight_blocks = weight_blocks.view(
                    E, -1, 128 * 128
                )  # [E, num_blocks, 128*128]
                weight_qt, weight_scale = pertoken_quant(
                    weight_blocks, quant_dtype=quant_dtype
                )
                weight_qt = weight_qt.view(
                    E, dim1 // 128, dim2 // 128, 128, 128
                )  # [E, num_blocks_dim1, num_blocks_dim2, 128, 128]
                weight_qt = weight_qt.permute(
                    0, 1, 3, 2, 4
                ).contiguous()  # [E, num_blocks_dim1, 128, num_blocks_dim2, 128]
                weight_qt = weight_qt.view(E, dim1, dim2)  # [E, dim1, dim2]
                weight_scale = weight_scale.view(
                    E, dim1 // 128, dim2 // 128
                )  # [E, num_blocks_dim1, num_blocks_dim2]
                return weight_qt, weight_scale

            QUAN_BLOCK_SZ=128
            assert  HIDDEN_SIZE%QUAN_BLOCK_SZ == 0 and INTER_SIZE_TP%QUAN_BLOCK_SZ==0, f"HIDDEN_SIZE and INTER_SIZE/TP must be multiples of {QUAN_BLOCK_SZ}  for per block quantization"
            assert  QUAN_BLOCK_SZ%TILE_N == 0, f"{QUAN_BLOCK_SZ=}  must be multiples of {TILE_N=}  for per block quantization"

            w_ = torch.randn([E*INTER_SIZE_TP * 2 * HIDDEN_SIZE // 128, 128], dtype=torch.bfloat16) / 2.0

            w1_qt, w1_qt_scale = weight_per_128x128_quant(w_.view(E, INTER_SIZE_TP * 2, HIDDEN_SIZE), quant_dtype=weight_type)
            # w1_qt_scale[...] = 1.0
            # print(w1_qt_scale)

            # print(f'==========={w1_qt.shape=}, {w1_qt_scale.shape=}')
            w1_ref = (w1_qt.to(dtype=torch.bfloat16).view(E, INTER_SIZE_TP * 2//128, 128, HIDDEN_SIZE//128,  128) * w1_qt_scale.view((E, INTER_SIZE_TP * 2//128, 1, HIDDEN_SIZE//128,  1))).to(dtype=torch.bfloat16)
            #w1_ref = w_
            w1_ref = w1_ref.view(E, INTER_SIZE_TP * 2, HIDDEN_SIZE)
            w1_qt = w1_qt.view(E, INTER_SIZE_TP * 2, HIDDEN_SIZE)
            w1_qt_scale = w1_qt_scale.view(E, INTER_SIZE_TP * 2//128, HIDDEN_SIZE//128)
            w1 = [w1_qt.clone() for _ in range(BUF_COPY)]
            w1_scale = [w1_qt_scale.clone() for _ in range(BUF_COPY)]
        
            w_ = torch.randn([E*HIDDEN_SIZE, INTER_SIZE_TP], dtype=torch.bfloat16) / 2.0
            w2_qt, w2_qt_scale = weight_per_128x128_quant(w_.view(E, HIDDEN_SIZE, INTER_SIZE_TP), quant_dtype=weight_type)
            # w2_qt_scale[...] = 1.0
            # print(f'==========={w2_qt.shape=}, {w2_qt_scale.shape=}')
            w2_ref = (w2_qt.to(dtype=torch.bfloat16).view(E, HIDDEN_SIZE//128, 128, INTER_SIZE_TP//128,  128) * w2_qt_scale.view(E, HIDDEN_SIZE//128, 1, INTER_SIZE_TP//128,  1)).to(dtype=torch.bfloat16)
            w2_ref = w2_ref.view(E, HIDDEN_SIZE, INTER_SIZE_TP)
            w2_qt = w2_qt.view(E, HIDDEN_SIZE, INTER_SIZE_TP)
            w2_qt_scale = w2_qt_scale.view(E, HIDDEN_SIZE//128, INTER_SIZE_TP//128)
            w2 = [w2_qt.clone() for _ in range(BUF_COPY)]
            w2_scale = [w2_qt_scale.clone() for _ in range(BUF_COPY)]
        else:
            assert 0, f'not support weight type "{weight_type}"'

        topk_weight = torch.randn([BUF_COPY, B, TOPK], dtype=torch.float32)
        topk_ids = torch.ones([BUF_COPY, B, TOPK], dtype=torch.int32)
        if ext_topk_ids is not None:
            topk_ids[...] = ext_topk_ids[None, :, :]
        # make a B*TOPK seq, which contains 0...E-1 0...E-1
        rep_e = div_up(B * TOPK, E)
        topk_ids_1d = torch.ones([rep_e, E], dtype=torch.int32)
        topk_ids_1d[:, ] = torch.randperm(E, dtype=torch.int32)
        topk_ids[:, ] = topk_ids_1d.reshape(-1)[ : B * TOPK].reshape(B, TOPK)
        access_expert = torch.unique(topk_ids[0])
        access_expert = access_expert.shape[0]

        flops = 2 * B * TOPK * (HIDDEN_SIZE * INTER_SIZE_TP * 2 + HIDDEN_SIZE * INTER_SIZE_TP)
        if weight_type == torch.bfloat16:
            ele_size = 2
        elif weight_type == torch.float4_e2m1fn_x2:
            ele_size = 0.5
        else:
            ele_size = 1
        mem_size = B * HIDDEN_SIZE * 2 + (HIDDEN_SIZE * INTER_SIZE_TP * 2 + HIDDEN_SIZE * INTER_SIZE_TP) * access_expert * ele_size

        def run(hidden_states, w1, w2, topk_weight, topk_ids, w1_scale, w2_scale, fp8_quant_type):
            fp8_ptpc = (fp8_quant_type == aiter.QuantType.per_Token)
            quant_type_str = quant2str_dict.get(fp8_quant_type, 'no')
            B = hidden_states.shape[0]
            E, N1, K1 = w1.shape
            N2, K2 = w2.shape[1], w2.shape[2]
            gemm1_out = torch.empty([B, TOPK, N1 // 2], dtype=hidden_states.dtype, device=hidden_states.device)
            #print(topk_weight.shape, topk_weight.dtype)
            #assert 0
            if kernel_type == '16x32_2s_b1':
                # test moe_gemm_batch: 2 stages, BLOCK_TILE_M=16, BLOCK_TILE_N=32, batch == 1
                cur_out = torch.zeros([1, N2], dtype=hidden_states.dtype, device=hidden_states.device)
                moe_gemm_batch1([N1 // 32, TOPK],[256], w1.dtype, True, hidden_states.data_ptr(), w1.data_ptr(), gemm1_out.data_ptr(), topk_ids.data_ptr(), topk_weight.data_ptr(), w1_scale.data_ptr() if w1_scale is not None else 0, 1, N1, K1, quant_type_str)
                moe_gemm_batch1([N2 // 32, TOPK],[64], w1.dtype, False, gemm1_out.data_ptr(), w2.data_ptr(), cur_out.data_ptr(), topk_ids.data_ptr(), topk_weight.data_ptr(), w2_scale.data_ptr() if w2_scale is not None else 0, 1, N2, K2, quant_type_str)
            elif kernel_type == '16x32_2s_b':
                # test moe_gemm_batch: 2 stages, BLOCK_TILE_M=16, BLOCK_TILE_N=32
                sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, cur_out = moe_sorting(
                    topk_ids,
                    topk_weight,
                    E,
                    K1,     # reduce dim is same with output dim
                    hidden_states.dtype,
                    16,
                    None,
                    None,
                    0,
                )
                grid = sorted_expert_ids.shape[0]
                if B * TOPK <= E:
                    grid = B * TOPK
                moe_gemm_batch([N1 // 32, grid], [256],
                                w1.dtype, True,
                                hidden_states.data_ptr(), w1.data_ptr(), gemm1_out.data_ptr(), sorted_ids.data_ptr(), sorted_weights.data_ptr(), sorted_expert_ids.data_ptr(), num_valid_ids.data_ptr(), w1_scale.data_ptr() if w1_scale is not None else 0, B, N1, K1, TOPK, quant_type_str)
                # moe_gemm_batch([N2 // 32, grid], [64],
                #                 w1.dtype, False,
                #                 gemm1_out.data_ptr(), w2.data_ptr(), cur_out.data_ptr(), sorted_ids.data_ptr(), sorted_weights.data_ptr(), sorted_expert_ids.data_ptr(), num_valid_ids.data_ptr(), w2_scale.data_ptr() if w2_scale is not None else 0, B, N2, K2, TOPK)
                num_CU = torch.cuda.get_device_properties().multi_processor_count
                BLOCK_N = 1024
                if (w1.dtype == torch.float8_e4m3fn or w1.dtype == torch.float8_e4m3fnuz) and fp8_ptpc and N2 // BLOCK_N * grid >= num_CU:
                    BLOCK_TILE_SIZE_M = 16
                    BLOCK_TILE_SIZE_N = 16
                    assert N2 % BLOCK_N == 0
                    use_atomic_write = B < 8
                    NUM_STAGES = 3
                    gemm2_out = cur_out
                    if not use_atomic_write:
                        gemm2_out = torch.empty([B, TOPK, HIDDEN_SIZE], dtype=torch.bfloat16)
                    moe_2stage_down_loopn([N2 // BLOCK_N, grid], [256],
                                    w1.dtype, TOPK, K2, N2, BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N,
                                    gemm1_out.data_ptr(), w2.data_ptr(), gemm2_out.data_ptr(), sorted_ids.data_ptr(), sorted_weights.data_ptr(), sorted_expert_ids.data_ptr(), num_valid_ids.data_ptr(),
                                    w2_scale.data_ptr() if w2_scale is not None else 0, B, fp8_ptpc, BLOCK_N, use_atomic_write, NUM_STAGES)
                    if not use_atomic_write:
                        cur_out = torch.sum(gemm2_out, dim=1)

                else:
                    BLOCK_TILE_SIZE_M = 16
                    BLOCK_TILE_SIZE_N = 64
                    moe_2stage_splitk([N2 // BLOCK_TILE_SIZE_N, grid], [64],
                                    w1.dtype, TOPK, K2, N2, False, BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N,
                                    gemm1_out.data_ptr(), w2.data_ptr(), cur_out.data_ptr(), sorted_ids.data_ptr(), sorted_weights.data_ptr(), sorted_expert_ids.data_ptr(), num_valid_ids.data_ptr(), w2_scale.data_ptr() if w2_scale is not None else 0, B, quant_type_str)
            elif kernel_type == 'mxn_splitk_2s':
                # test moe_gemm_batch_vmn: 2 stages, m/n can be set
                if weight_type == torch.float4_e2m1fn_x2:
                    K1 *= 2
                    K2 *= 2
                sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, cur_out = moe_sorting(
                    topk_ids,
                    topk_weight,
                    E,
                    K1,     # reduce dim is same with output dim
                    hidden_states.dtype,
                    TILE_M,
                    None,
                    None,
                    0,
                )
                BLOCK_TILE_SIZE_M = TILE_M
                BLOCK_TILE_SIZE_N = TILE_N
                grid = sorted_expert_ids.shape[0]
                if B * TOPK <= E:
                    grid = B * TOPK
                moe_2stage_splitk([N1 // BLOCK_TILE_SIZE_N, grid], [256],
                                w1.dtype, TOPK, K1, N1, True, BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N,
                                hidden_states.data_ptr(), w1.data_ptr(), gemm1_out.data_ptr(), sorted_ids.data_ptr(), sorted_weights.data_ptr(), sorted_expert_ids.data_ptr(), num_valid_ids.data_ptr(), w1_scale.data_ptr() if w1_scale is not None else 0, B, fp8_ptpc)
                moe_2stage_splitk([N2 // BLOCK_TILE_SIZE_N, grid], [64],
                                w1.dtype, TOPK, K2, N2, False, BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N,
                                gemm1_out.data_ptr(), w2.data_ptr(), cur_out.data_ptr(), sorted_ids.data_ptr(), sorted_weights.data_ptr(), sorted_expert_ids.data_ptr(), num_valid_ids.data_ptr(), w2_scale.data_ptr() if w2_scale is not None else 0, B, fp8_ptpc)
            elif kernel_type == 'mxn_splitk_1s':
                # test moe_gemm_stage1
                sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, cur_out = moe_sorting(
                    topk_ids,
                    topk_weight,
                    E,
                    K1,     # reduce dim is same with output dim
                    hidden_states.dtype,
                    TILE_M,
                    None,
                    None,
                    0,
                )
                BLOCK_TILE_SIZE_M = TILE_M
                BLOCK_TILE_SIZE_N = TILE_N
                moe_1stage_splitk([1, sorted_expert_ids.shape[0]], [256], 
                                w1.dtype, TOPK, K1, N1, N2, BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N,
                                hidden_states.data_ptr(), w1.data_ptr(), w1_scale.data_ptr() if w1_scale is not None else 0, w2.data_ptr(), w2_scale.data_ptr() if w2_scale is not None else 0,
                                cur_out.data_ptr(), 
                                sorted_ids.data_ptr(), sorted_weights.data_ptr(), sorted_expert_ids.data_ptr(), num_valid_ids.data_ptr(), B)
            elif kernel_type == 'mxn_2s':
                #assert weight_type == torch.bfloat16, f'mxn_2s only support bfloat16, but got {weight_type}'
                # test moe_gemm_batch_vmn: 2 stages, m/n can be set
                sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, cur_out = moe_sorting(
                    topk_ids,
                    topk_weight,
                    E,
                    N2,     # reduce dim is same with output dim
                    hidden_states.dtype,
                    TILE_M,
                    None,
                    None,
                    0,
                )
                #print(f"================ {hidden_states.shape=} {hidden_states.dtype} {topk_ids.shape} {topk_weight.shape} {E} {K1}-{N2} {TILE_M} {cur_out.shape=}")
                if weight_type == torch.float4_e2m1fn_x2:
                    # if B <= 1024:
                    #     a1, a1_scale = fused_dynamic_mxfp4_quant_moe_sort(
                    #         hidden_states,
                    #         sorted_ids=sorted_ids,
                    #         num_valid_ids=num_valid_ids,
                    #         token_num=token_num,
                    #         topk=1,
                    #         block_size=block_size_M,
                    #     )
                    # else:
                    from aiter.utility.fp4_utils import moe_mxfp4_sort
                    quant_func = aiter.get_hip_quant(aiter.QuantType.per_1x32)
                    hidden_states_q, hidden_states_scale = quant_func(
                        hidden_states,
                        scale=None,
                        quant_dtype=torch.float4_e2m1fn_x2,
                        num_rows=None,
                    )
                    # TODO: it seems assume using 8(x32)blocks
                    hidden_states_scale = moe_mxfp4_sort(
                        hidden_states_scale,
                        sorted_ids=sorted_ids,
                        num_valid_ids=num_valid_ids,
                        token_num=B,
                        block_size=TILE_M,
                    )
                    
                    # TODO: call kernel
                    # gemm1_out : torch.empty([B, TOPK, N1 // 2], dtype=hidden_states.dtype, device=hidden_states.device)
                    if 0:
                        if 0:
                            torch.save((sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, w1, w1_scale,
                                        hidden_states_q, hidden_states_scale, gemm1_out), 'tensors_tuple2.pt')
                            assert 0
                        moe_gemm_ref(TILE_M, TILE_N, True, sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids,
                                    w1, w1_scale,
                                    #hidden_states, None,
                                    hidden_states_q, hidden_states_scale,
                                    gemm1_out)
                        gemm1_out_q, gemm1_out_scale = quant_func(
                            gemm1_out.view(B*TOPK, -1),
                            scale=None,
                            quant_dtype=torch.float4_e2m1fn_x2,
                            num_rows=None,
                        )
                        gemm1_out_scale = moe_mxfp4_sort(
                            gemm1_out_scale[: B * TOPK, :].view(B, TOPK, -1),
                            sorted_ids=sorted_ids,
                            num_valid_ids=num_valid_ids,
                            token_num=B,
                            block_size=TILE_M,
                        )
                        if 0:
                            torch.save((sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, w2, w2_scale,
                                        gemm1_out_q, gemm1_out_scale, cur_out), 'tensors_tuple.pt')
                            assert 0
                        moe_gemm_ref(TILE_M, TILE_N, False, sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids,
                                    w2, w2_scale, 
                                    #gemm1_out, None,
                                    gemm1_out_q.view(B, TOPK, -1), gemm1_out_scale,
                                    cur_out)                    
                    else:
                        gateup_OC = w1.shape[1]
                        assert gateup_OC % TILE_N == 0
                        num_oc_blocks = gateup_OC // TILE_N
                        num_e_blocks = sorted_expert_ids.shape[0]                    
                        moe_gemm_mxfp4([num_oc_blocks, num_e_blocks],[256],
                            TILE_M, TILE_N,
                            w1.shape[0], w1.shape[1], w1.shape[2], 
                            True, TOPK, # gate_up,
                            sorted_ids.data_ptr(),
                            sorted_weights.data_ptr(),
                            sorted_expert_ids.data_ptr(),
                            num_valid_ids.data_ptr(),
                            w1.data_ptr(), w1_scale.data_ptr(),
                            hidden_states_q.data_ptr(), hidden_states_scale.data_ptr(),
                            gemm1_out.data_ptr(), B)

                        gemm1_out_q, gemm1_out_scale = quant_func(
                            gemm1_out.view(B*TOPK, -1),
                            scale=None,
                            quant_dtype=torch.float4_e2m1fn_x2,
                            num_rows=None,
                        )
                        gemm1_out_scale = moe_mxfp4_sort(
                            gemm1_out_scale[: B * TOPK, :].view(B, TOPK, -1),
                            sorted_ids=sorted_ids,
                            num_valid_ids=num_valid_ids,
                            token_num=B,
                            block_size=TILE_M,
                        )

                        down_OC = w2.shape[1]
                        assert down_OC % TILE_N == 0
                        num_oc_blocks = down_OC // TILE_N
                        num_e_blocks = sorted_expert_ids.shape[0]
                        gemm2_out = torch.empty(B, TOPK, N2, dtype=torch.bfloat16)
                        moe_gemm_mxfp4([num_oc_blocks, num_e_blocks],[256],
                            TILE_M, TILE_N,
                            w2.shape[0], w2.shape[1], w2.shape[2], 
                            False, TOPK, # gate_up,
                            sorted_ids.data_ptr(),
                            sorted_weights.data_ptr(),
                            sorted_expert_ids.data_ptr(),
                            num_valid_ids.data_ptr(),
                            w2.data_ptr(), w2_scale.data_ptr(),
                            gemm1_out_q.data_ptr(), gemm1_out_scale.data_ptr(),
                            #cur_out.data_ptr(),
                            gemm2_out.data_ptr(),
                            B)
                        if 1:
                            num_WG = 256 * 2
                            num_tokens_wg = B // num_WG
                            num_extra_tokens = B % num_WG
                            moe_gemm_final_reduce_bf16([num_WG], [64], TOPK, N2,
                                                    gemm2_out.data_ptr(),
                                                    cur_out.data_ptr(),
                                                    num_tokens_wg, num_extra_tokens, B)
                        '''
                        for i_tok in range(B):
                            cur_out[i_tok,:] = 0
                            for topk in range(TOPK):
                                cur_out[i_tok,:] += gemm2_out[i_tok, topk] * topk_weight[i_tok, topk]
                        '''
                elif weight_type == torch.bfloat16:
                    # moe_gemm_ref(TILE_M, TILE_N, True, sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids,
                    #              w1, w1_scale, hidden_states, None, gemm1_out)
                    # moe_gemm_ref(TILE_M, TILE_N, False, sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids,
                    #              w2, w2_scale, gemm1_out, None, cur_out)

                    BLOCK_TILE_SIZE_M = TILE_M
                    BLOCK_TILE_SIZE_N = TILE_N
                    id_buf = torch.zeros(64, dtype=torch.int32)
                    with cudaPerf(2 * B * TOPK * HIDDEN_SIZE * INTER_SIZE_TP * 2, HIDDEN_SIZE * INTER_SIZE_TP * access_expert * ele_size * 2, name=f"up") as p:
                        moe_2stage_gateup([80], [256],
                                    w1.dtype, TOPK, K1, N1, BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N, str(fp8_quant_type),
                                    id_buf.data_ptr(), hidden_states.data_ptr(), w1.data_ptr(), gemm1_out.data_ptr(), sorted_ids.data_ptr(), sorted_expert_ids.data_ptr(), num_valid_ids.data_ptr(),
                                    None, w1_scale, B, N1 // BLOCK_TILE_SIZE_N * sorted_expert_ids.shape[0])
                    gemm2_out = torch.empty(B, TOPK, N2, dtype=torch.bfloat16, device=hidden_states.device)
                    moe_2stage_down([1, sorted_expert_ids.shape[0]], [256],
                                w1.dtype, TOPK, K2, N2, False, BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N,
                                gemm1_out, w2, gemm2_out, sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, None, w2_scale, B, sorted_expert_ids.shape[0])
                    num_WG = 80 * 4
                    num_tokens_wg = B // num_WG
                    num_extra_tokens = B % num_WG
                    moe_gemm_final_reduce_bf16([num_WG], [64], TOPK, N2,
                                            gemm2_out.data_ptr(),
                                            cur_out.data_ptr(),
                                            num_tokens_wg, num_extra_tokens, B)
                elif fp8_quant_type == aiter.QuantType.per_Token or fp8_quant_type == aiter.QuantType.per_Tensor:
                    BLOCK_TILE_SIZE_M = TILE_M
                    BLOCK_TILE_SIZE_N = TILE_N
                    # always quantize activation with aiter.QuantType.per_Token
                    # fp8_quant_type is only for weights
                    quant_func = aiter.get_hip_quant(aiter.QuantType.per_Token)

                    with cudaPerf(0, 0, name=f"quant_up") as p:
                        hidden_states_q, hidden_states_scale = quant_func(
                            hidden_states,
                            scale=None,
                            quant_dtype=weight_type,
                            num_rows=None,
                        )
                    if 0:
                        hs_s = hidden_states_scale
                        w1_s = w1_scale
                        if fp8_quant_type == aiter.QuantType.per_Tensor:
                            w1_s = w1_s[:,None,None].expand(-1,N1,1)
                        moe_2stage_gateup_ref(hidden_states_q, hs_s,
                                            w1, w1_s,
                                            gemm1_out,
                                            TILE_M, 
                                            sorted_ids, sorted_expert_ids, sorted_weights, num_valid_ids, TOPK)
                    else:
                        id_buf = torch.zeros(64, dtype=torch.int32)
                        with cudaPerf(2 * B * TOPK * HIDDEN_SIZE * INTER_SIZE_TP * 2, HIDDEN_SIZE * INTER_SIZE_TP * access_expert * ele_size * 2, name=f"up") as p:
                            moe_2stage_gateup([80], [256],
                                        w1.dtype, TOPK, K1, N1, BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N,  str(fp8_quant_type),
                                        id_buf,
                                        hidden_states_q, w1, 
                                        gemm1_out, 
                                        sorted_ids, 
                                        sorted_expert_ids, 
                                        num_valid_ids, 
                                        hidden_states_scale,
                                        w1_scale, B, N1 // BLOCK_TILE_SIZE_N * sorted_expert_ids.shape[0])
                    # down
                    with cudaPerf(0, 0, name=f"quant_down") as p:
                        gemm1_out_q, gemm1_out_scale = quant_func(
                            gemm1_out.view(B * TOPK, -1),
                            scale=None,
                            quant_dtype=w2.dtype,
                            num_rows=None,
                        )

                    if DDD:
                        print(sorted_expert_ids)
                        total_wasted = 0
                        for k in range(0, sorted_expert_ids.numel()):
                            if k*BLOCK_TILE_SIZE_M > num_valid_ids[0]: break
                            n_valid = ((sorted_ids[k*BLOCK_TILE_SIZE_M:(k+1)*BLOCK_TILE_SIZE_M] >> 24) < TOPK).sum().item()
                            if n_valid < BLOCK_TILE_SIZE_M:
                                print(f" expert {sorted_expert_ids[k]}:   {n_valid}/{BLOCK_TILE_SIZE_M}")
                                total_wasted += BLOCK_TILE_SIZE_M - n_valid
                            else:
                                print(f" expert {sorted_expert_ids[k]}:   ______")
                        print(f"Total wasted: {total_wasted/sorted_ids.numel()*100:.2f} %")
                        assert 0

                    if 0:
                        g1_s = gemm1_out_scale
                        w2_s = w2_scale
                        if fp8_quant_type == aiter.QuantType.per_Tensor:
                            w2_s = w2_s[:,None,None].expand(-1,N2,-1)
                        moe_2stage_down_ref(gemm1_out_q.view(B, TOPK, -1), g1_s,
                                            w2, w2_s,
                                            cur_out,
                                            TILE_M, 
                                            sorted_ids, sorted_expert_ids, sorted_weights, num_valid_ids)
                    else:
                        gemm2_out = torch.empty(B, TOPK, N2, dtype=torch.bfloat16, device=gemm1_out_q.device)
                        down_mem_size = HIDDEN_SIZE * INTER_SIZE_TP * access_expert * ele_size + B * TOPK * INTER_SIZE_TP * 2 + B * TOPK * HIDDEN_SIZE * 2
                        with cudaPerf(2 * B * TOPK * HIDDEN_SIZE * INTER_SIZE_TP, down_mem_size, name=f"down") as p:
                            moe_2stage_down([1, sorted_expert_ids.shape[0]], [256],
                                        w2.dtype, TOPK, K2, N2, False, BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N, str(fp8_quant_type),
                                        gemm1_out_q, w2, 
                                        gemm2_out, #cur_out,
                                        sorted_ids,
                                        sorted_weights,
                                        sorted_expert_ids,
                                        num_valid_ids,
                                        gemm1_out_scale,
                                        w2_scale,
                                        B,
                                        sorted_expert_ids.shape[0])
                        with cudaPerf(rw_bytes=B*(TOPK+1)*N2*2, name="reduce") as p:
                            if 0:
                                cur_out = gemm2_out.sum(dim=1)
                            else:
                                num_WG = 80 * 4
                                num_tokens_wg = B // num_WG
                                num_extra_tokens = B % num_WG
                                moe_gemm_final_reduce_bf16([num_WG], [64], TOPK, N2,
                                                        gemm2_out.data_ptr(),
                                                        cur_out.data_ptr(),
                                                        num_tokens_wg, num_extra_tokens, B)

            else:
                assert 0, f'not support kernel type "{kernel_type}"'
            return cur_out

        tflops_res = []
        latencies = []
        bw = []
        if kernel_type == 'aiter':
            # aiter preshuffle seems doesn't help much for fp8
            # if wei_is_fp8(w1[0].dtype):
            #     j = 0
            #     for _ in range(run_count):
            #         w1[j] = shuffle_weight(w1[j], layout=(16, 16))
            #         w2[j] = shuffle_weight(w2[j], layout=(16, 16))
            #         j = (j + 1) % BUF_COPY
            i = 0
            for _ in range(run_count):
                with cudaPerf(flops, mem_size, name=f"{kernel_type}[{B=},{str(weight_type).split('.')[1]}]") as p:
                    _run_aiter(hidden_states=hidden_states[i], w1=w1[i], w2=w2[i], topk_weight=topk_weight[i], topk_ids=topk_ids[i], w1_scale=w1_scale[i], w2_scale=w2_scale[i], quant_type=quant_type)
                i = (i + 1) % BUF_COPY
                tflops_res.append(p.tflops())
                latencies.append(p.dt())
                bw.append(p.bw())
            diff = 0
        else:
            if weight_type == torch.float4_e2m1fn_x2:
                # fp4 no shuffle
                w1_qt_aiter = w1[0]
                w2_qt_aiter = w2[0]
            else:
                w1_qt_aiter = shuffle_weight(w1[0], layout=(16, 16))
                w2_qt_aiter = shuffle_weight(w2[0], layout=(16, 16))
            ref_out = get_torch_ref(hidden_states=hidden_states[0], w1=w1_ref, w2=w2_ref, topk_weight=topk_weight[0], topk_ids=topk_ids[0])
            # aiter_out = _run_aiter(hidden_states=hidden_states[0], w1=w1[0], w2=w2[0], topk_weight=topk_weight[0], topk_ids=topk_ids[0], w1_scale=w1_scale[0], w2_scale=w2_scale[0])
            cur_out = run(hidden_states=hidden_states[0], w1=w1_qt_aiter, w2=w2_qt_aiter, topk_weight=topk_weight[0], topk_ids=topk_ids[0], w1_scale=w1_scale[0], w2_scale=w2_scale[0], fp8_quant_type=fp8_quant_type)
            #print(f">>>>>>>>>>>>>>> {calc_diff(aiter_out, ref_out)=} ")
            #print(f">>>>>>>>>>>>>>> {calc_diff(cur_out, ref_out)=} ")
            #print(f">>>>>>>>>>>>>>> {calc_diff(aiter_out, cur_out)=} ")

            i = 0
            for _ in range(run_count):
                with cudaPerf(flops, mem_size, name=f"{kernel_type}[{B=},{str(weight_type).split('.')[1]}]") as p:
                    run(hidden_states=hidden_states[i], w1=w1[i], w2=w2[i], topk_weight=topk_weight[i], topk_ids=topk_ids[i], w1_scale=w1_scale[i], w2_scale=w2_scale[i], fp8_quant_type=fp8_quant_type)
                i = (i + 1) % BUF_COPY
                tflops_res.append(p.tflops())
                latencies.append(p.dt())
                bw.append(p.bw())

            #print(f">>>>>>>>>>>>>>> {calc_diff(aiter_out, ref_out)=} ")
            #print(f">>>>>>>>>>>>>>> {calc_diff(cur_out, ref_out)=} ")
            #print(f">>>>>>>>>>>>>>> {calc_diff(aiter_out, cur_out)=} ")

            diff = calc_diff(ref_out, cur_out)#, diff_thr=0.01)
            if 1 and diff > 0.02:
            #if not torch.allclose(ref_out, cur_out, rtol=0.02, atol=0.02):
                print(ref_out)
                print(cur_out)
                idx = torch.where(torch.abs(ref_out - cur_out) > 0.01)
                if len(idx[0]):
                    print(f'idx = {idx}\nref={ref_out[idx]}\ncur={cur_out[idx]}\n{len(idx[0])}')
                assert 0, f"{kernel_type=}, {B=}, {weight_type=}, {TILE_M=}, {TILE_N=}, {run_count=}"
            else:
                quantype=""
                if wei_is_fp8(weight_type):
                    if fp8_quant_type == aiter.QuantType.per_Token:
                        quantype = "@PTPC"
                    elif fp8_quant_type == aiter.QuantType.per_Tensor:
                        quantype = "@Tensor"
                    else:
                        quantype = "@blockwise"
                print(f"{kernel_type}[{B=} {weight_type=}{quantype}] acc OK err {diff=:.6f}")
        if run_count > 0:
            return {'flops': sum(tflops_res[1:])/len(tflops_res[1:]),              # tflops
                    'latency': sum(latencies[1:])/len(latencies[1:]) * 1e6,        # us
                    'bw': sum(bw[1:]) / len(bw[1:]),
                    "diff" : diff}                               # GB/s

    # special path for batch1 
    def entry_b1(self, prec):
        if self.perf is None:
            self.perf = []
        kernel_type = '16x32_2s_b1'
        perf = {}
        perf[kernel_type] = {}
        perf_prec = {}
        for weight_type, quant_type in prec:
            if weight_type is None: continue
            perf_prec[1] = self(kernel_type, weight_type, quant_type, B=1)
            perf[kernel_type][str(weight_type)] = perf_prec
        self.perf.append(perf)

    def entry_common(self, kernel_type, batch, prec):
        if self.perf is None:
            self.perf = []
        perf = {}
        perf[kernel_type] = {}
        for weight_type, quant_type in prec:
            if weight_type is None: continue
            perf_prec = {}
            for i in batch:
                ret = self(kernel_type, weight_type, quant_type, i)
                key = f'{i}'
                if self.INTER_SIZE_TP_ADJ != self.INTER_SIZE_TP:
                    key += f" (adjusted INTER_SIZE_TP={self.INTER_SIZE_TP}=>{self.INTER_SIZE_TP_ADJ})"
                perf_prec[key] = ret

            perf[kernel_type][str(weight_type)+"@"+str(quant_type)] = perf_prec
        self.perf.append(perf)

    def show_perf(self):
        print('\nsummary:')
        for perf in self.perf:
            for kernel, vals in perf.items():
                for prec, vals_ in vals.items():
                    for b, data in vals_.items():
                        print(f'{kernel}[{prec:<4} B={b:<4}]: {data["latency"]:5.0f} us, {data["bw"]:6.1f} GB/s, {data["flops"]:4.1f} tflops,  diff : {data["diff"]:.6f}')


def init_env():
    torch.set_printoptions(linewidth=3000, sci_mode=False, edgeitems=8, )
    torch.set_default_device('cuda')
    torch.manual_seed(0)
    #pyhip.set_device()

def test_acc(test):
    init_env()
    #entry_common('aiter', batch=[8192], prec=[torch.float4_e2m1fn_x2], TILE_M=128, TILE_N=128, run_count=2, HIDDEN_SIZE=4096, INTER_SIZE=2048, TP=8)
    # entry_common('mxn_splitk_2s', batch=[16], prec=[torch.float4_e2m1fn_x2], TILE_M=TILE_M, TILE_N=TILE_N, HIDDEN_SIZE=HIDDEN_SIZE, INTER_SIZE=INTER_SIZE, TP=TP, run_count=0)

    #entry_common('mxn_2s', batch=[8192], test_fp8=False, TILE_M=128, TILE_N=128, run_count=0)
    #assert 0,"========================"

    batch = list(range(2, 64))
    # fix TILE_M=16, TILE_N=32
    test.entry_b1(run_count=0, prec=[prec_bf16, prec_fp8_ptpc])           # batch 1
    test.entry_common('16x32_2s_b', batch=batch, prec=[prec_bf16, prec_fp8_ptpc])
    batch += list(range(128, 256))
    batch += [i * 256 for i in range(1, 4)]
    batch += [i * 2048 for i in range(1, 5)]
    batch += list(range(2048 * 3, 2048 * 3 + 256))
    test.entry_common('mxn_splitk_2s', batch=batch, prec=[prec_bf16, prec_mxfp4])
    # TILE_M/N is configurable
    test.entry_common('mxn_splitk_2s', batch=batch, prec=[prec_fp8_ptpc])
    test.entry_common('mxn_splitk_2s', batch=batch, prec=[prec_fp8_b])

    # TODO: support fp8
    test.entry_common('mxn_splitk_1s', batch=batch, prec=[prec_bf16])
    # entry_common('mxn_2s', batch=batch, prec=[torch.bfloat16], TILE_M=128, TILE_N=128, HIDDEN_SIZE=HIDDEN_SIZE, INTER_SIZE=INTER_SIZE, TP=TP, run_count=0)


@pytest.mark.parametrize("batch", [[1, 2, 4, 8, 12, 16, 32, 64]])
def test_small_batch_perf(batch, TILE_M, TILE_N, HIDDEN_SIZE, INTER_SIZE_TP, E, TOPK):
    init_env()
    test = TestCase(TILE_M, TILE_N, HIDDEN_SIZE, INTER_SIZE_TP, E, TOPK)
    if is_arch_type('942'):
        test.entry_common('aiter', batch, prec=[prec_bf16, prec_fp8_ptpc])
    else:
        test.entry_common('aiter', batch, prec=[prec_bf16])
    # fix TILE_M=16, TILE_N=32
    test.entry_b1(prec=[prec_bf16, prec_fp8_ptpc])
    test.entry_common('16x32_2s_b', batch=batch, prec=[prec_fp8_ptpc])
    test.show_perf()

@pytest.mark.parametrize("batch", [[16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]])
def test_perf(batch, TILE_M=32, TILE_N=64, HIDDEN_SIZE=4096, INTER_SIZE_TP=2048, E=512, TOPK=10, test_sets=['aiter', 'mxn_2s', 'mxn_splitk_2s']):
    init_env()
    test = TestCase(TILE_M, TILE_N, HIDDEN_SIZE, INTER_SIZE_TP, E, TOPK)
    if 'aiter' in test_sets:
        test.entry_common('aiter', batch, prec=[prec_bf16, prec_fp8_ptpc, prec_mxfp4])
        # perf.append(entry_common('aiter', batch, prec=[torch.bfloat16, get_fp8type()], HIDDEN_SIZE=HIDDEN_SIZE, INTER_SIZE=INTER_SIZE, TP=TP, TILE_M=TILE_M, TILE_N=TILE_N))
        # perf.append(entry_common('aiter', batch, prec=[get_fp8type()], HIDDEN_SIZE=HIDDEN_SIZE, INTER_SIZE=INTER_SIZE, TP=TP, TILE_M=TILE_M, TILE_N=TILE_N, TOPK=10, E=512, fp8_ptpc=False))

    # TODO: support fp8
    # perf.append(entry_common('mxn_splitk_1s', batch=batch, prec=[torch.bfloat16], TILE_M=TILE_M, TILE_N=TILE_N, HIDDEN_SIZE=HIDDEN_SIZE, INTER_SIZE=INTER_SIZE, TP=TP))
    if 'mxn_2s' in test_sets:
        test.entry_common('mxn_2s', batch=batch, prec=[ prec_fp8_ptpc])
    if 'mxn_splitk_2s' in test_sets:
        # TILE_M/N is configurable
        test.entry_common('mxn_splitk_2s', batch=batch, prec=[torch.bfloat16, prec_mxfp4])
        test.entry_common('mxn_splitk_2s', batch=batch, prec=[prec_fp8_ptpc])
        test.entry_common('mxn_splitk_2s', batch=batch, prec=[prec_fp8_b])
    test.show_perf()

if __name__ == '__main__':
    TILE_M = 16
    TILE_N = 64
    HIDDEN_SIZE, INTER_SIZE_TP, E, TOPK = 4096, 128, 512, 10
    HIDDEN_SIZE, INTER_SIZE_TP, E, TOPK = 4096, 192, 192, 8

    init_env()
    
    # entry_common('mxn_splitk_2s', batch=batch, prec=[get_fp8type()], TILE_M=TILE_M, TILE_N=TILE_N, HIDDEN_SIZE=HIDDEN_SIZE, INTER_SIZE=INTER_SIZE, TP=TP, run_count=0, fp8_ptpc=False)
    #entry_common('mxn_2s', batch=batch, prec=[torch.bfloat16], TILE_M=128, TILE_N=128, HIDDEN_SIZE=HIDDEN_SIZE, INTER_SIZE=INTER_SIZE, TP=TP, run_count=0)
    #entry_common('mxn_2s', batch=batch, prec=[torch.float4_e2m1fn_x2], TILE_M=128, TILE_N=128, HIDDEN_SIZE=HIDDEN_SIZE, INTER_SIZE=INTER_SIZE, TP=TP, run_count=0)
    #entry_common('aiter', batch=batch, prec=[torch.float4_e2m1fn_x2], TILE_M=128, TILE_N=128, HIDDEN_SIZE=HIDDEN_SIZE, INTER_SIZE=INTER_SIZE, TP=TP)
    #entry_common('mxn_2s', batch=batch, prec=[torch.float4_e2m1fn_x2], TILE_M=128, TILE_N=128, HIDDEN_SIZE=HIDDEN_SIZE, INTER_SIZE=INTER_SIZE, TP=TP)
    #with torchPerf():
    #    entry_common('aiter', batch, prec=[get_fp4type_if_valid()], HIDDEN_SIZE=HIDDEN_SIZE, INTER_SIZE=INTER_SIZE, TP=TP, TILE_M=TILE_M, TILE_N=TILE_N)
    if 0:
        test_acc(TILE_M=TILE_M, TILE_N=TILE_N, HIDDEN_SIZE=HIDDEN_SIZE, INTER_SIZE_TP=INTER_SIZE_TP)
        batch = [1, 2, 4, 8, 12, 16, 32]
        #batch = [1, 2, 4]
        test_small_batch_perf(batch, HIDDEN_SIZE=HIDDEN_SIZE, INTER_SIZE_TP=INTER_SIZE_TP)
        # batch = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
        # batch = [1, 2, 4, 8, 16, 32,64,128, 256]
        # test_perf(batch, TILE_M=TILE_M, TILE_N=TILE_N, HIDDEN_SIZE=HIDDEN_SIZE, INTER_SIZE=INTER_SIZE, TP=TP)
    else:
        batch = [512, 1024, 2048]
        #batch = [8192,]
        if 0:
            ext_topk_ids = torch.load("/root/tingqli/topk_ids/topk_ids_79.pt")
            print(ext_topk_ids.shape)
            batch[0] = ext_topk_ids.shape[0]

        HIDDEN_SIZE, INTER_SIZE_TP, E, TOPK = 4096, 192, 192, 8
        #decoding
        TILE_M, TILE_N = 16, 64
        batch = [2, 4, 8, 16, 32, 64, 128, 256]
        test_dec = TestCase(TILE_M, TILE_N, HIDDEN_SIZE, INTER_SIZE_TP, E, TOPK)
        test_dec.entry_common('aiter', [1] + batch, prec=[prec_fp8_t])
        test_dec.entry_common('16x32_2s_b1', [1], prec=[prec_fp8_t])
        test_dec.entry_common('16x32_2s_b', batch, prec=[prec_fp8_t])

        # prefill per tensor
        TILE_M, TILE_N = 128, 128
        batch = [512,1024,2048,4096,8192, 16384, 32768, 65536, 131072]
        test_prefill = TestCase(TILE_M, TILE_N, HIDDEN_SIZE, INTER_SIZE_TP, E, TOPK)
        test_prefill.entry_common('aiter', batch, prec=[prec_fp8_t])
        test_prefill.entry_common('mxn_2s', batch, prec=[prec_fp8_t])

        # prefill ptpc
        TILE_M, TILE_N = 128, 256
        HIDDEN_SIZE, INTER_SIZE_TP, E, TOPK = 4096, 128, 512, 10
        batch = [i * 1024 for i in range(1, 9)] + [16 * 1024, 32 * 1024, 64 * 1024]
        test_ptpc = TestCase(TILE_M, TILE_N, HIDDEN_SIZE, INTER_SIZE_TP, E, TOPK)
        test_ptpc.entry_common('aiter', batch, prec=[prec_fp8_ptpc])
        test_ptpc.entry_common('mxn_2s', batch, prec=[prec_fp8_ptpc])
        
        test_dec.show_perf()
        test_prefill.show_perf()
        test_ptpc.show_perf()
        #entry_common('mxn_2s', batch=batch, prec=[get_fp8type()], E=E, TOPK=TOPK, TILE_M=128, TILE_N=128, HIDDEN_SIZE=HIDDEN_SIZE, INTER_SIZE=INTER_SIZE, TP=TP, run_count=10)
        #test_perf(batch, TILE_M=128, TILE_N=256, HIDDEN_SIZE=HIDDEN_SIZE, INTER_SIZE_TP=INTER_SIZE_TP, E=E, TOPK=TOPK, test_sets=['aiter', 'mxn_2s'])


