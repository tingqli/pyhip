import pyhip
import torch
import aiter
from aiter import dtypes

from pyhip.contrib.moe_gemm_8wave import moe_gemm_final_reduce_bf16, moe_gemm_8wave

__all__ = [
    "fused_moe_asmjit"
]

"""
act   : fp8 per-token block-scale    1 x 128 
weight: fp8 block-scale            128 x 128 

use 8-wave methods for both fp8_blockscale (bf16/fp16) MOE ?

stage1: gate-up
stage2: down

MXFP requires special scheduling

Here we focus on EP, which means each GPU handles a full gate/up/down MLP.

num_local_tokens:
    actual token count in hidden_states is this one instead of shape[0], to adapt to CUDA graph

从最简化的方式逐渐逼近目标，例如先实现一个直接调用gemm组合实现目标的版本
必须先有一个参考版本, 逐步替换为优化版本，在更小的修改粒度上保持正确性
先实现bf16-8wave版本，降低复杂度，保证一些基本逻辑正确性
"""
def de_shuffle_weight(weight, mfma_MN = 16):
    M, K = weight.shape
    K_bytes = K * weight.itemsize
    sizeof_DW4 = 16
    mfma_K_lanes = 64 // mfma_MN
    mfma_K_L = sizeof_DW4//weight.itemsize
    mfma_K = mfma_K_lanes * mfma_K_L 

    assert M % mfma_MN == 0
    mfma_K_bytes = mfma_K_lanes * sizeof_DW4
    assert K_bytes % mfma_K_bytes == 0
    #x = x.reshape(M//mfma_MN, mfma_MN, K//mfma_K, mfma_K_lanes, mfma_K_L)
    #x = x.permute(0,2,3,1,4)

    assert K % mfma_K == 0
    weight = weight.reshape(M//mfma_MN, K//mfma_K, mfma_K_lanes, mfma_MN, mfma_K_L)
    weight = weight.permute(0,3,1,2,4)
    weight = weight.reshape(M, K).contiguous()
    return weight

def moe_gemm_ref(topk, block_m, sorted_ids, sorted_expert_ids, sorted_weights, num_valid_ids,
                 input, input_scale, weight, w_scale, output, is_gateup):
    #assert weight.dtype == torch.bfloat16
    if is_gateup:
        NUM_TOKENS, NUM_STATES = input.shape
    else:
        NUM_TOKENS, TOPK, NUM_STATES = input.shape
    
    NUM_EXPERTS, OC, IC = weight.shape
    NUM_BLOCKS = sorted_expert_ids.shape[0]

    weight_de_shuffle = torch.empty_like(weight)
    for i in range(NUM_EXPERTS):
        weight_de_shuffle[i,...] = de_shuffle_weight(weight[i, ...])

    if w_scale is not None:
        # dequantize
        weight_de_shuffle = weight_de_shuffle.view(NUM_EXPERTS, OC//128, 128, IC//128, 128).to(w_scale.dtype) * w_scale.view(NUM_EXPERTS, OC//128, 1, IC//128, 1)
        weight_de_shuffle = weight_de_shuffle.view(NUM_EXPERTS, OC, IC)

    if input_scale is not None:
        # dequantize
        # print(input_scale.dtype, input_scale.shape, NUM_TOKENS, NUM_STATES)
        if is_gateup:
            input = input.view(NUM_TOKENS, NUM_STATES//128, 128).to(input_scale.dtype) * input_scale.view(-1, NUM_TOKENS).t().contiguous().view(NUM_TOKENS, NUM_STATES//128, 1)
            input = input.view(NUM_TOKENS, NUM_STATES)
        else:
            input = input.view(NUM_TOKENS*TOPK, NUM_STATES//128, 128).to(input_scale.dtype) * input_scale.view(NUM_STATES//128, NUM_TOKENS*TOPK).t().contiguous().view(NUM_TOKENS*TOPK, NUM_STATES//128, 1)
            input = input.view(NUM_TOKENS, TOPK, NUM_STATES)

    for e_idx in range(NUM_BLOCKS):
        max_id = num_valid_ids[0]
        if e_idx * block_m >= max_id:
            break

        s_e_id = sorted_expert_ids[e_idx]
        i0 = e_idx*block_m
        i1 = i0 + block_m

        ids = sorted_ids[i0:i1].clone()
        tok_ids = ids & 0xFFFFFF
        tok_topk = ids >> 24

        valid_mask = tok_topk < torch.tensor(topk)

        expert_w = weight_de_shuffle[s_e_id, ...]
        if is_gateup:
            src = input[tok_ids[valid_mask],...]
        else:
            src = input[tok_ids[valid_mask], tok_topk[valid_mask], ...]

        act = src.to(torch.float) @ expert_w.t().to(torch.float)

        if is_gateup:
            act_gate = act[:, :(OC//2)]
            act_up = act[:,(OC//2):]
            act = torch.nn.functional.silu(act_gate) * act_up
            output[tok_ids[valid_mask], tok_topk[valid_mask], :] = act.to(output.dtype)
        else:
            tok_w = sorted_weights[i0:i1]
            cur_out = act[...].to(output.dtype) * tok_w[valid_mask, None]
            #output[tok_ids[valid_mask], ...] += cur_out
            output[tok_ids[valid_mask], tok_topk[valid_mask], :] = cur_out.to(output.dtype)

moe_gemm = moe_gemm_ref

def fused_moe_asmjit(
    hidden_states,
    w1,  # [expert(local_expert:EP), inter_dim*2, dim] N,K
    w2,  # [expert(local_expert:EP), dim, inter_dim]
    topk_weight,
    topk_ids,
    expert_mask = None,  # EP
    activation = aiter.ActivationType.Silu,
    quant_type = aiter.QuantType.per_128x128,
    doweight_stage1 = False,
    # following for quant
    w1_scale = None,  # [expert(local_expert:EP), inter_dim, 1]
    w2_scale = None,  # [expert(local_expert:EP), model_dim, 1]
    a1_scale = None,  # [expert(local_expert:EP), 1, model_dim]
    a2_scale = None,  # [expert(local_expert:EP), 1, inter_dim]
    # following for tuning
    block_size_M=None,
    num_local_tokens = None,
    moe_sorting_dispatch_policy=0,
    dtype=None,
    # following for cktile support
    hidden_pad=0,
    intermediate_pad=0,
    bias1=None,
    bias2=None,
    splitk=0,
):
    device = hidden_states.device
    def get_inter_dim(w1_shape, w2_shape):
        E, _, model_dim = w1_shape
        E, model_dim, inter_dim = w2_shape

        int4_war = model_dim // w1_shape[-1]
        inter_dim *= int4_war
        return E, model_dim, inter_dim

    M, topk = topk_ids.shape
    E, model_dim, inter_dim = get_inter_dim(w1.shape, w2.shape)
    assert w1.shape[1] == inter_dim * 2

    isG1U1 = inter_dim != w1.shape[1]
    isShuffled = getattr(w1, "is_shuffled", False)
    global_E = E
    if expert_mask is not None:
        global_E = expert_mask.numel()

    dtype = hidden_states.dtype if dtype is None else dtype
    assert dtype == aiter.dtypes.bf16
    assert w1.dtype == w2.dtype

    q_dtype_w = w1.dtype
    q_dtype_a = dtype if quant_type == aiter.QuantType.No else w1.dtype

    verbose = 0

    if verbose:
        print(f"============================================= {device=} ")
        print(f" quant_type : {quant_type}")
        print(f" dtype      : {dtype}")
        print(f" w1.dtype   : {w1.dtype}")
        print(f" w2.dtype   : {w2.dtype}")
        print(f" q_dtype_a  : {q_dtype_a}")
        print(f" {M=} {topk=} {global_E=} {E=} {model_dim=} {inter_dim=}")
        print(f" {isG1U1=} {isShuffled=} ")

        print("\thidden_states:", list(hidden_states.shape), hidden_states.dtype)   # hidden_states: [M, model_dim] torch.bfloat16
        print("\tw1:", list(w1.shape), w1.dtype)                                    # w1           : [E, inter_dim*2, model_dim]  (isG1U1 is True)
        print("\tw2:", list(w2.shape), w2.dtype)                                    # w2           : [E, model_dim, inter_dim]
        print("\ttopk_weight:", list(topk_weight.shape), topk_weight.dtype)         # topk_weight  : [M, topk]      torch.float32
        print("\ttopk_ids:", list(topk_ids.shape), topk_ids.dtype)                  # topk_ids     : [M, topk]      torch.int32
        if expert_mask is not None:
            print("\texpert_mask:", list(expert_mask.shape), expert_mask.dtype)
        if w1_scale is not None:
            print("\tw1_scale:", list(w1_scale.shape), w1_scale.dtype)              # w1_scale     : [E, inter_dim*2//128,  model_dim//128]  torch.float32
        if w2_scale is not None:
            print("\tw2_scale:", list(w2_scale.shape), w2_scale.dtype)              # w2_scale     : [E, model_dim//128, inter_dim//128]     torch.float32

    assert activation == aiter.ActivationType.Silu
    assert quant_type == aiter.QuantType.per_128x128 or quant_type == aiter.QuantType.No

    assert doweight_stage1 == False

    assert a1_scale is None
    assert a2_scale is None

    assert hidden_pad == 0
    assert intermediate_pad == 0
    assert bias1 is None
    assert bias2 is None
    assert splitk == 0

    assert isG1U1 == True
    assert isShuffled == True

    # determine block_size_M
    block_size_M = 256

    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_out = aiter.fused_moe.moe_sorting(
        topk_ids, topk_weight, global_E, model_dim,  dtype,
        block_size_M,
        expert_mask, num_local_tokens, moe_sorting_dispatch_policy,
    )
    if verbose:
        print("sorted_ids        :", list(sorted_ids.shape), sorted_ids.dtype)
        print("\t", sorted_ids[:block_size_M])
        print("\t", sorted_ids[block_size_M:block_size_M*2])
        print("\t", sorted_ids[-block_size_M:])
        print("sorted_weights    :", list(sorted_weights.shape), sorted_weights.dtype)
        print("sorted_expert_ids :", list(sorted_expert_ids.shape), sorted_expert_ids.dtype, sorted_expert_ids.tolist())
        print("num_valid_ids     :", list(num_valid_ids.shape), num_valid_ids.dtype, num_valid_ids.tolist())
        print("moe_out           :", list(moe_out.shape), moe_out.dtype)

    def dummy_quant_func(input, scale, quant_dtype=None, num_rows = 0, num_rows_factor = 0, transpose_scale = False):
        return input, None

    if quant_type == aiter.QuantType.per_128x128:
        act_quant_func = aiter.get_hip_quant(aiter.QuantType.per_1x128)
    else:
        assert quant_type == aiter.QuantType.No
        act_quant_func = dummy_quant_func

    token_num, _ = hidden_states.shape

    if num_local_tokens is None:
        num_local_tokens = torch.tensor(token_num, dtype=torch.int)

    a1, a1_scale = act_quant_func(
        hidden_states,
        scale=a1_scale,
        quant_dtype=q_dtype_a,
        num_rows=num_local_tokens,
        transpose_scale=True
    )

    if verbose:
        print("\ta1        :", list(a1.shape), a1.dtype)
        print("\ta1_scale  :", list(a1_scale.shape), a1_scale.dtype)
        assert 0

    """
    ratio = a1_scale.element_size() // a1.element_size()
    a2 = torch.empty(
        (token_num + (token_num * ratio + 127) // 128, topk, inter_dim),
        dtype=q_dtype_a,
        device=device,
    )
    """
    a2 = torch.empty(
        (token_num, topk, inter_dim),
        dtype=torch.bfloat16,
        device=device,
    )
    w1_scale = w1_scale.view(dtypes.fp8_e8m0) if w1.dtype == dtypes.fp4x2 else w1_scale
    moe_gemm(topk, block_size_M, sorted_ids, sorted_expert_ids, sorted_weights, num_valid_ids,
             a1, a1_scale,
             w1, w1_scale, a2, is_gateup=True)

    a2, a2_scale = act_quant_func(
        a2, #a2.view(token_num*topk, -1),
        scale=a2_scale,
        quant_dtype=q_dtype_a,
        num_rows=num_local_tokens,
        num_rows_factor=topk,
        transpose_scale=True
    )
    a2 = a2.view(token_num, topk, inter_dim)

    w2_scale = w2_scale.view(dtypes.fp8_e8m0) if w2.dtype == dtypes.fp4x2 else w2_scale

    stage2_out = torch.empty(
        (token_num, topk, model_dim),
        dtype=torch.bfloat16,
        device=device,
    )

    if 0:
        moe_gemm(topk, block_size_M, sorted_ids, sorted_expert_ids, sorted_weights, num_valid_ids,
                a2, a2_scale,
                w2, w2_scale, stage2_out, is_gateup=False)
    else:
        if q_dtype_a == torch.bfloat16:
            AB_dtype = "bf16"
        else:
            assert 0
        wg_M, wg_N = 256, 256
        num_oc_blocks = model_dim//256
        num_e_blocks = sorted_expert_ids.shape[0]
        #sorted_expert_ids[...] = 0
        valid_e_blocks = num_valid_ids[0].item()//wg_M
        with pyhip.cudaPerf(num_oc_blocks*valid_e_blocks*wg_M*wg_N*inter_dim*2, name="moe_gemm_8wave_down"):
            moe_gemm_8wave([num_oc_blocks, num_e_blocks], [8*64],
                    AB_dtype, wg_M, wg_N,
                    E, model_dim, inter_dim, 
                    False, topk,
                    sorted_ids.data_ptr(),
                    sorted_weights.data_ptr(),
                    sorted_expert_ids.data_ptr(),
                    num_valid_ids.data_ptr(),
                    w2.data_ptr(), None if w2_scale is None else w2_scale.data_ptr(),
                    a2.data_ptr(), None if a2_scale is None else a2_scale.data_ptr(),
                    stage2_out.data_ptr(),
                    token_num) # num_local_tokens.data_ptr() ?

    if 1:
        num_WG = 256 * 2
        num_tokens_wg = num_local_tokens // num_WG
        num_extra_tokens = num_local_tokens % num_WG
        moe_gemm_final_reduce_bf16([num_WG], [64], topk, model_dim,
                                stage2_out.data_ptr(),
                                moe_out.data_ptr(),
                                num_tokens_wg, num_extra_tokens, num_local_tokens)

    return moe_out