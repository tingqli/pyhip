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


def _run_aiter(hidden_states,
            w1,  # [expert(local_expert:EP), inter_dim*2, dim] N,K
            w2,  # [expert(local_expert:EP), dim, inter_dim]
            topk_weight,
            topk_ids,
            w1_scale: Optional[torch.tensor] = None,  # [expert(local_expert:EP), inter_dim, 1]
            w2_scale: Optional[torch.tensor] = None,  # [expert(local_expert:EP), model_dim, 1]
            ):
    from aiter.fused_moe import fused_moe
    from aiter import QuantType
    if w1.dtype == torch.float4_e2m1fn_x2:
        quant_type = QuantType.per_1x32
    elif w1.dtype == torch.bfloat16:
        quant_type = QuantType.No
    else:
        quant_type = QuantType.per_Token
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

def _run_batch(kernel_type, B=1, weight_type=torch.bfloat16, TILE_M=16, TILE_N=32, run_count=10, HIDDEN_SIZE=2048, INTER_SIZE=1024, TOPK=8, E=128, TP=8):
    INTER_SIZE_TP = INTER_SIZE // TP
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
        import aiter
        from aiter.utility import fp4_utils
        from aiter.ops.shuffle import shuffle_weight
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
    else:
        import aiter
        torch_quant = aiter.get_torch_quant(aiter.QuantType.per_Token)
        w_ = torch.randn([E, INTER_SIZE_TP * 2, HIDDEN_SIZE], dtype=torch.bfloat16)
        w1_qt, w1_qt_scale = torch_quant(w_, quant_dtype=weight_type)
        w1_ref = (w1_qt.to(dtype=torch.bfloat16) * w1_qt_scale).to(dtype=torch.bfloat16)
        w1 = [w1_qt.clone() for _ in range(BUF_COPY)]
        w1_scale = [w1_qt_scale.clone() for _ in range(BUF_COPY)]
        w_ = torch.randn([E, HIDDEN_SIZE, INTER_SIZE_TP], dtype=torch.bfloat16)
        w2_qt, w2_qt_scale = torch_quant(w_, quant_dtype=weight_type)
        w2_ref = (w2_qt.to(dtype=torch.bfloat16) * w2_qt_scale).to(dtype=torch.bfloat16)
        w2 = [w2_qt.clone() for _ in range(BUF_COPY)]
        w2_scale = [w2_qt_scale.clone() for _ in range(BUF_COPY)]

    topk_weight = torch.randn([BUF_COPY, B, TOPK], dtype=torch.float32)
    topk_ids = torch.ones([BUF_COPY, B, TOPK], dtype=torch.int32)
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

    import aiter
    from aiter.ops.shuffle import shuffle_weight
    from aiter.fused_moe import moe_sorting
    from aiter.utility import fp4_utils


    def run(hidden_states, w1, w2, topk_weight, topk_ids, w1_scale, w2_scale):
        B = hidden_states.shape[0]
        E, N1, K1 = w1.shape
        N2, K2 = w2.shape[1], w2.shape[2]
        gemm1_out = torch.empty([B, TOPK, N1 // 2], dtype=hidden_states.dtype, device=hidden_states.device)
        #print(topk_weight.shape, topk_weight.dtype)
        #assert 0
        if kernel_type == '16x32_2s_b1':
            # test moe_gemm_batch: 2 stages, BLOCK_TILE_M=16, BLOCK_TILE_N=32, batch == 1
            cur_out = torch.zeros([1, N2], dtype=hidden_states.dtype, device=hidden_states.device)
            moe_gemm_batch1([N1 // 32, TOPK],[256], w1.dtype, True, hidden_states.data_ptr(), w1.data_ptr(), gemm1_out.data_ptr(), topk_ids.data_ptr(), topk_weight.data_ptr(), w1_scale.data_ptr() if w1_scale is not None else 0, 1, N1, K1)
            moe_gemm_batch1([N2 // 32, TOPK],[64], w1.dtype, False, gemm1_out.data_ptr(), w2.data_ptr(), cur_out.data_ptr(), topk_ids.data_ptr(), topk_weight.data_ptr(), w2_scale.data_ptr() if w2_scale is not None else 0, 1, N2, K2)
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
            moe_gemm_batch([N1 // 32, sorted_expert_ids.shape[0]], [256],
                            w1.dtype, True,
                            hidden_states.data_ptr(), w1.data_ptr(), gemm1_out.data_ptr(), sorted_ids.data_ptr(), sorted_weights.data_ptr(), sorted_expert_ids.data_ptr(), num_valid_ids.data_ptr(), w1_scale.data_ptr() if w1_scale is not None else 0, B, N1, K1, TOPK)
            moe_gemm_batch([N2 // 32, sorted_expert_ids.shape[0]], [64],
                            w1.dtype, False,
                            gemm1_out.data_ptr(), w2.data_ptr(), cur_out.data_ptr(), sorted_ids.data_ptr(), sorted_weights.data_ptr(), sorted_expert_ids.data_ptr(), num_valid_ids.data_ptr(), w2_scale.data_ptr() if w2_scale is not None else 0, B, N2, K2, TOPK)
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
                               hidden_states.data_ptr(), w1.data_ptr(), gemm1_out.data_ptr(), sorted_ids.data_ptr(), sorted_weights.data_ptr(), sorted_expert_ids.data_ptr(), num_valid_ids.data_ptr(), w1_scale.data_ptr() if w1_scale is not None else 0, B)
            moe_2stage_splitk([N2 // BLOCK_TILE_SIZE_N, grid], [64],
                               w1.dtype, TOPK, K2, N2, False, BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N,
                               gemm1_out.data_ptr(), w2.data_ptr(), cur_out.data_ptr(), sorted_ids.data_ptr(), sorted_weights.data_ptr(), sorted_expert_ids.data_ptr(), num_valid_ids.data_ptr(), w2_scale.data_ptr() if w2_scale is not None else 0, B)
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
            else:
                moe_gemm_ref(TILE_M, TILE_N, True, sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids,
                             w1, w1_scale, hidden_states, None, gemm1_out)
                moe_gemm_ref(TILE_M, TILE_N, False, sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids,
                             w2, w2_scale, gemm1_out, None, cur_out)

                BLOCK_TILE_SIZE_M = TILE_M
                BLOCK_TILE_SIZE_N = TILE_N
                #moe_2stage_gateup([N1 // BLOCK_TILE_SIZE_N, sorted_expert_ids.shape[0]], [256],
                #                w1.dtype, TOPK, K1, N1, BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N,
                #                hidden_states.data_ptr(), w1.data_ptr(), gemm1_out.data_ptr(), sorted_ids.data_ptr(), sorted_expert_ids.data_ptr(), num_valid_ids.data_ptr(), w1_scale.data_ptr() if w1_scale is not None else 0, B)
                #moe_2stage_down([1, sorted_expert_ids.shape[0]], [256],
                #                w1.dtype, TOPK, K2, N2, False, BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N,
                #                gemm1_out.data_ptr(), w2.data_ptr(), cur_out.data_ptr(), sorted_ids.data_ptr(), sorted_weights.data_ptr(), sorted_expert_ids.data_ptr(), num_valid_ids.data_ptr(), w2_scale.data_ptr() if w2_scale is not None else 0, B)

            # BLOCK_TILE_SIZE_N = 64
            # moe_2stage_splitk([N2 // BLOCK_TILE_SIZE_N, sorted_expert_ids.shape[0]], [64],
            #                    w1.dtype, TOPK, K2, N2, False, BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N,
            #                    gemm1_out.data_ptr(), w2.data_ptr(), cur_out.data_ptr(), sorted_ids.data_ptr(), sorted_weights.data_ptr(), sorted_expert_ids.data_ptr(), num_valid_ids.data_ptr(), w2_scale.data_ptr() if w2_scale is not None else 0, B)
        else:
            assert 0, f'not support kernel type "{kernel_type}"'
        return cur_out

    tflops_res = []
    latencies = []
    bw = []
    if kernel_type == 'aiter':
        # aiter needs preshuffle weights
        i = 0
        for _ in range(run_count):
            with cudaPerf(flops, mem_size, name=f"{kernel_type}[{B=},{str(weight_type).split('.')[1]}]") as p:
                _run_aiter(hidden_states=hidden_states[i], w1=w1[i], w2=w2[i], topk_weight=topk_weight[i], topk_ids=topk_ids[i], w1_scale=w1_scale[i], w2_scale=w2_scale[i])
            i = (i + 1) % BUF_COPY
            tflops_res.append(p.tflops())
            latencies.append(p.dt())
            bw.append(p.bw())
    else:
        if weight_type == torch.float4_e2m1fn_x2:
            # fp4 no shuffle
            w1_qt_aiter = w1[0]
            w2_qt_aiter = w2[0]
        else:
            w1_qt_aiter = shuffle_weight(w1[0], layout=(16, 16))
            w2_qt_aiter = shuffle_weight(w2[0], layout=(16, 16))
        ref_out = get_torch_ref(hidden_states=hidden_states[0], w1=w1_ref, w2=w2_ref, topk_weight=topk_weight[0], topk_ids=topk_ids[0])
        aiter_out = _run_aiter(hidden_states=hidden_states[0], w1=w1[0], w2=w2[0], topk_weight=topk_weight[0], topk_ids=topk_ids[0], w1_scale=w1_scale[0], w2_scale=w2_scale[0])
        cur_out = run(hidden_states=hidden_states[0], w1=w1_qt_aiter, w2=w2_qt_aiter, topk_weight=topk_weight[0], topk_ids=topk_ids[0], w1_scale=w1_scale[0], w2_scale=w2_scale[0])
        #print(f">>>>>>>>>>>>>>> {calc_diff(aiter_out, ref_out)=} ")
        #print(f">>>>>>>>>>>>>>> {calc_diff(cur_out, ref_out)=} ")
        #print(f">>>>>>>>>>>>>>> {calc_diff(aiter_out, cur_out)=} ")

        i = 0
        for _ in range(run_count):
            with cudaPerf(flops, mem_size, name=f"{kernel_type}[{B=},{str(weight_type).split('.')[1]}]") as p:
                run(hidden_states=hidden_states[i], w1=w1[i], w2=w2[i], topk_weight=topk_weight[i], topk_ids=topk_ids[i], w1_scale=w1_scale[i], w2_scale=w2_scale[i])
            i = (i + 1) % BUF_COPY
            tflops_res.append(p.tflops())
            latencies.append(p.dt())
            bw.append(p.bw())

        #print(f">>>>>>>>>>>>>>> {calc_diff(aiter_out, ref_out)=} ")
        #print(f">>>>>>>>>>>>>>> {calc_diff(cur_out, ref_out)=} ")
        #print(f">>>>>>>>>>>>>>> {calc_diff(aiter_out, cur_out)=} ")

        if weight_type == torch.float4_e2m1fn_x2:
            diff = calc_diff(aiter_out, cur_out)
        else:
            diff = calc_diff(ref_out, cur_out)
        if diff > 0.02:
            #if not torch.allclose(ref_out, cur_out, rtol=0.1, atol=0.03):
            print(ref_out)
            print(cur_out)
            idx = torch.where(torch.abs(ref_out - cur_out) > 0.03)
            if len(idx[0]):
                print(f'idx = {idx}\nref={ref_out[idx]}\ncur={cur_out[idx]}\n{len(idx[0])}')
            assert 0, f"{kernel_type=}, {B=}, {weight_type=}, {TILE_M=}, {TILE_N=}, {run_count=}"
        else:
            print(f"{kernel_type}[{B=} {weight_type=}] acc OK")
    if run_count > 0:
        return {'flops': sum(tflops_res[1:])/len(tflops_res[1:]),              # tflops
                'latency': sum(latencies[1:])/len(latencies[1:]) * 1e6,        # us
                'bw': sum(bw[1:]) / len(bw[1:])}                               # GB/s

def is_arch_type(arch):
    props = torch.cuda.get_device_properties()
    return arch in props.gcnArchName

def get_fp8type():
    return torch.float8_e4m3fn if is_arch_type('950') else torch.float8_e4m3fnuz

def get_fp4type_if_valid():
    return torch.float4_e2m1fn_x2 if is_arch_type('950') else None

# special path for batch1 
def entry_b1(prec=[torch.bfloat16], HIDDEN_SIZE=2048, INTER_SIZE=1024, TOPK=8, E=128, TP=8, run_count=10):
    kernel_type = '16x32_2s_b1'
    perf = {}
    perf[kernel_type] = {}
    perf_prec = {}

    for weight_type in prec:
        if weight_type is None: continue
        perf_prec[1] = _run_batch(kernel_type, B=1, weight_type=weight_type, run_count=run_count, HIDDEN_SIZE=HIDDEN_SIZE, INTER_SIZE=INTER_SIZE, TOPK=TOPK, E=E, TP=TP)
        perf[kernel_type][str(weight_type)] = perf_prec
    return perf

def entry_common(kernel_type, batch, prec=[torch.bfloat16], TILE_M=32, TILE_N=64, HIDDEN_SIZE=2048, INTER_SIZE=1024, TOPK=8, E=128, TP=8, run_count=10):
    perf = {}
    perf[kernel_type] = {}
    for weight_type in prec:
        if weight_type is None: continue
        perf_prec = {}
        org_INTER_SIZE = INTER_SIZE

        if (weight_type == torch.float8_e4m3fn or weight_type == torch.float8_e4m3fnuz) and INTER_SIZE // TP % 128 != 0:
            INTER_SIZE = div_up(INTER_SIZE // TP, 128) * 128 * TP
        if kernel_type == 'aiter' and weight_type == torch.float4_e2m1fn_x2 and INTER_SIZE // TP % 128 != 0:
            INTER_SIZE = div_up(INTER_SIZE // TP, 128) * 128 * TP
        for i in batch:
            if org_INTER_SIZE != INTER_SIZE:
                key = f'{i} (adjusted INTER_SIZE={INTER_SIZE})'
            else:
                key = f'{i}'
            perf_prec[key] = _run_batch(kernel_type, B=i, weight_type=weight_type, TILE_M=TILE_M, TILE_N=TILE_N, run_count=run_count, HIDDEN_SIZE=HIDDEN_SIZE, INTER_SIZE=INTER_SIZE, TOPK=TOPK, E=E, TP=TP)
        perf[kernel_type][str(weight_type)] = perf_prec
        INTER_SIZE = org_INTER_SIZE
    
    return perf

def init_env():
    torch.set_printoptions(linewidth=3000, sci_mode=False, edgeitems=8, )
    torch.set_default_device('cuda')
    torch.manual_seed(0)

def test_acc(TILE_M=32, TILE_N=64, HIDDEN_SIZE=4096, INTER_SIZE=2048, TP=8):
    init_env()
    #entry_common('aiter', batch=[8192], prec=[torch.float4_e2m1fn_x2], TILE_M=128, TILE_N=128, run_count=2, HIDDEN_SIZE=4096, INTER_SIZE=2048, TP=8)
    #entry_common('mxn_splitk_2s', batch=[16], prec=[torch.float4_e2m1fn_x2], TILE_M=TILE_M, TILE_N=TILE_N, HIDDEN_SIZE=HIDDEN_SIZE, INTER_SIZE=INTER_SIZE, TP=TP, run_count=0)

    #entry_common('mxn_2s', batch=[8192], test_fp8=False, TILE_M=128, TILE_N=128, run_count=0)
    #assert 0,"========================"
    batch = list(range(2, 64))
    # fix TILE_M=16, TILE_N=32
    entry_b1(run_count=0, prec=[torch.bfloat16, get_fp8type()], HIDDEN_SIZE=HIDDEN_SIZE, INTER_SIZE=INTER_SIZE, TP=TP)           # batch 1
    entry_common('16x32_2s_b', batch=batch, prec=[torch.bfloat16, get_fp8type()], TILE_M=16, TILE_N=32, HIDDEN_SIZE=HIDDEN_SIZE, INTER_SIZE=INTER_SIZE, TP=TP, run_count=0)
    batch += list(range(128, 256))
    batch += [i * 256 for i in range(1, 4)]
    batch += [i * 2048 for i in range(1, 5)]
    batch += list(range(2048 * 3, 2048 * 3 + 256))
    # TILE_M/N is configurable
    entry_common('mxn_splitk_2s', batch=batch, prec=[torch.bfloat16, get_fp8type(), get_fp4type_if_valid()], TILE_M=TILE_M, TILE_N=TILE_N, HIDDEN_SIZE=HIDDEN_SIZE, INTER_SIZE=INTER_SIZE, TP=TP, run_count=0)
    # TODO: support fp8
    entry_common('mxn_splitk_1s', batch=batch, prec=[torch.bfloat16], TILE_M=TILE_M, TILE_N=TILE_N, HIDDEN_SIZE=HIDDEN_SIZE, INTER_SIZE=INTER_SIZE, TP=TP, run_count=0)
    entry_common('mxn_2s', batch=batch, prec=[torch.bfloat16], TILE_M=128, TILE_N=128, HIDDEN_SIZE=HIDDEN_SIZE, INTER_SIZE=INTER_SIZE, TP=TP, run_count=0)

def show_perf(perf):
    print('\nsummary:')
    for kernel, vals in perf.items():
        for prec, vals_ in vals.items():
            for b, data in vals_.items():
                print(f'{kernel}[{prec:<4} B={b:<4}]: {data["latency"]:5.0f} us, {data["bw"]:6.1f} GB/s, {data["flops"]:4.1f} tflops')

@pytest.mark.parametrize("batch", [[1, 2, 4, 8, 12, 16, 32, 64]])
def test_small_batch_perf(batch, HIDDEN_SIZE=4096, INTER_SIZE=2048, TP=8):
    init_env()
    perf = {}
    if is_arch_type('942'):
        perf.update(entry_common('aiter', batch, prec=[torch.bfloat16, get_fp8type()], HIDDEN_SIZE=HIDDEN_SIZE, INTER_SIZE=INTER_SIZE, TP=TP))
    else:
        perf.update(entry_common('aiter', batch, prec=[torch.bfloat16], HIDDEN_SIZE=HIDDEN_SIZE, INTER_SIZE=INTER_SIZE, TP=TP))
    # fix TILE_M=16, TILE_N=32
    perf.update(entry_b1(prec=[torch.bfloat16, get_fp8type()], HIDDEN_SIZE=HIDDEN_SIZE, INTER_SIZE=INTER_SIZE, TP=TP))           # batch 1
    perf.update(entry_common('16x32_2s_b', batch=batch, prec=[torch.bfloat16, get_fp8type()], TILE_M=16, TILE_N=32, HIDDEN_SIZE=HIDDEN_SIZE, INTER_SIZE=INTER_SIZE, TP=TP))
    show_perf(perf)

@pytest.mark.parametrize("batch", [[16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]])
def test_perf(batch, TILE_M=32, TILE_N=64, HIDDEN_SIZE=4096, INTER_SIZE=2048, TP=8):
    init_env()
    perf = {}
    perf.update(entry_common('aiter', batch, prec=[torch.bfloat16, get_fp8type(), get_fp4type_if_valid()], HIDDEN_SIZE=HIDDEN_SIZE, INTER_SIZE=INTER_SIZE, TP=TP, TILE_M=TILE_M, TILE_N=TILE_N))
    # TODO: support fp8
    perf.update(entry_common('mxn_splitk_1s', batch=batch, prec=[torch.bfloat16], TILE_M=TILE_M, TILE_N=TILE_N, HIDDEN_SIZE=HIDDEN_SIZE, INTER_SIZE=INTER_SIZE, TP=TP))
    perf.update(entry_common('mxn_2s', batch=batch, prec=[torch.bfloat16], TILE_M=128, TILE_N=128, HIDDEN_SIZE=HIDDEN_SIZE, INTER_SIZE=INTER_SIZE, TP=TP))
    # TILE_M/N is configurable
    perf.update(entry_common('mxn_splitk_2s', batch=batch, prec=[torch.bfloat16, get_fp8type(), get_fp4type_if_valid()], TILE_M=TILE_M, TILE_N=TILE_N, HIDDEN_SIZE=HIDDEN_SIZE, INTER_SIZE=INTER_SIZE, TP=TP))
    show_perf(perf)

if __name__ == '__main__':
    TILE_M = 128
    TILE_N = 128
    HIDDEN_SIZE = 4096
    INTER_SIZE = 1536
    TP = 4
    batch =  [256*2+253]
    batch =  [94*256]
    batch =  [24000]
    init_env()
    #entry_common('mxn_2s', batch=batch, prec=[torch.bfloat16], TILE_M=128, TILE_N=128, HIDDEN_SIZE=HIDDEN_SIZE, INTER_SIZE=INTER_SIZE, TP=TP, run_count=0)
    #entry_common('mxn_2s', batch=batch, prec=[torch.float4_e2m1fn_x2], TILE_M=128, TILE_N=128, HIDDEN_SIZE=HIDDEN_SIZE, INTER_SIZE=INTER_SIZE, TP=TP, run_count=0)
    #entry_common('aiter', batch=batch, prec=[torch.float4_e2m1fn_x2], TILE_M=128, TILE_N=128, HIDDEN_SIZE=HIDDEN_SIZE, INTER_SIZE=INTER_SIZE, TP=TP)
    #entry_common('mxn_2s', batch=batch, prec=[torch.float4_e2m1fn_x2], TILE_M=128, TILE_N=128, HIDDEN_SIZE=HIDDEN_SIZE, INTER_SIZE=INTER_SIZE, TP=TP)
    #with torchPerf():
    #    entry_common('aiter', batch, prec=[get_fp4type_if_valid()], HIDDEN_SIZE=HIDDEN_SIZE, INTER_SIZE=INTER_SIZE, TP=TP, TILE_M=TILE_M, TILE_N=TILE_N)
    if 1:
        test_acc(TILE_M=TILE_M, TILE_N=TILE_N, HIDDEN_SIZE=HIDDEN_SIZE, INTER_SIZE=INTER_SIZE, TP=TP)
        batch = [1, 2, 4, 8, 12, 16, 32, 64]
        test_small_batch_perf(batch, HIDDEN_SIZE=HIDDEN_SIZE, INTER_SIZE=INTER_SIZE, TP=TP)
        batch = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
        test_perf(batch, TILE_M=TILE_M, TILE_N=TILE_N, HIDDEN_SIZE=HIDDEN_SIZE, INTER_SIZE=INTER_SIZE, TP=TP)
