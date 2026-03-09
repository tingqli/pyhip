import pyhip
import torch
import aiter
from aiter import dtypes

from .moe_gemm_8wave import moe_gemm_final_reduce_bf16, moe_gemm_8wave
from .moe import moe_2stage_splitk, moe_2stage_gateup, moe_2stage_down

__all__ = [
    "fused_moe"
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
                 input, input_scale, weight, w_scale, output, is_gateup, is_shuffled):
    #assert weight.dtype == torch.bfloat16
    if is_gateup:
        NUM_TOKENS, NUM_STATES = input.shape
    else:
        NUM_TOKENS, TOPK, NUM_STATES = input.shape
    
    NUM_EXPERTS, OC, IC = weight.shape
    NUM_BLOCKS = sorted_expert_ids.shape[0]

    weight_de_shuffle = weight.clone()
    if is_shuffled:
        for i in range(NUM_EXPERTS):
            weight_de_shuffle[i, ...] = de_shuffle_weight(weight[i, ...])

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
            output[tok_ids[valid_mask], ...] += cur_out
            #output[tok_ids[valid_mask], tok_topk[valid_mask], :] = cur_out.to(output.dtype)


def moe_sorting_ref(topk_ids,       # [num_tokens, topk]
                    topk_weights,   # [num_tokens, topk]
                    num_experts,    # number of global experts
                    model_dim,      # for returning output buffer [num_tokens, model_dim] moebuf_dtype
                    moebuf_dtype,   # 
                    block_size,     # 
                    expert_mask=None,       # 
                    num_local_tokens=None,  # 
                    dispatch_policy=0):
    assert expert_mask is None
    assert num_local_tokens is None
    num_tokens, topk = topk_ids.shape

    # 
    def div_up(x, y):
        return (x + y - 1) // y

    num_e_blocks = div_up(num_tokens * topk + num_experts * (block_size - 1), block_size)

    sorted_ids = torch.empty([num_e_blocks * block_size], dtype=torch.uint32)
    sorted_weights = torch.empty([num_e_blocks * block_size], dtype=torch.float)
    sorted_expert_ids = torch.empty([num_e_blocks], dtype=torch.uint32)
    num_valid_ids = torch.empty([2], dtype=torch.uint32)
    moe_out = torch.empty([num_tokens, model_dim], dtype=moebuf_dtype)

    index = 0
    for expert_id in range(num_experts):
        for i in range(num_tokens):
            for t in range(topk):
                if topk_ids[i, t] == expert_id:
                    if index % block_size == 0:
                        sorted_expert_ids[index//block_size] = expert_id
                    sorted_ids[index] = (t << 24)|(i)
                    sorted_weights[index] = topk_weights[i, t]
                    index += 1
        i0 = index % 256
        if i0:
            for i in range(i0, 256):
                sorted_ids[index] = (topk << 24)|(num_tokens)
                sorted_weights[index] = 0
                index += 1

    num_valid_ids[0] = index

    return sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_out


def fused_moe(
    hidden_states,
    w1,  # [expert(local_expert:EP), inter_dim*2, dim] N,K
    w2,  # [expert(local_expert:EP), dim, inter_dim]
    topk_weight,
    topk_ids,
    expert_mask = None,  # EP
    activation = aiter.ActivationType.Silu,
    quant_type = aiter.QuantType.No,
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
    assert expert_mask is None

    """
    /sgl-workspace/sglang/python/sglang/srt/layers/quantization/unquant.py
        : after wrapping shuffled tensor with torch.nn.Parameter(), the "is_shuffled" attributes is missing
    layer.w13_weight = torch.nn.Parameter(
        shuffle_weight(layer.w13_weight.data, (16, 16)),
        requires_grad=False,
    ) 
    """
    if isinstance(w1, torch.nn.parameter.Parameter):
        w1_is_shuffled = True
    else:
        w1_is_shuffled = getattr(w1, "is_shuffled", False)

    if isinstance(w2, torch.nn.parameter.Parameter):
        w2_is_shuffled = True
    else:
        w2_is_shuffled = getattr(w2, "is_shuffled", False)

    isG1U1 = inter_dim != w1.shape[1]

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
        print(f" {M=} {topk=} {global_E=} {E=} {model_dim=} {inter_dim=} {isG1U1=}")

        print("\thidden_states:", list(hidden_states.shape), hidden_states.dtype)   # hidden_states: [M, model_dim] torch.bfloat16
        print("\tw1:", list(w1.shape), w1.dtype)                                    # w1           : [E, inter_dim*2, model_dim]  (isG1U1 is True)  torch.float8_e4m3fn/torch.bfloat16
        print("\tw2:", list(w2.shape), w2.dtype)                                    # w2           : [E, model_dim, inter_dim]                      torch.float8_e4m3fn/torch.bfloat16
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

    token_num, _ = hidden_states.shape

    a2 = torch.empty(
        (token_num, topk, inter_dim),
        dtype=torch.bfloat16,
        device=device,
    )


    #moe_sorting = moe_sorting_ref
    moe_sorting = aiter.fused_moe.moe_sorting

    estimated_tokens_per_expert = token_num * topk / E
    if estimated_tokens_per_expert < 32 and quant_type == aiter.QuantType.No:
        # mem-bound case : very small num_tokens
        block_size_M = 32
        block_size_N = 128
        sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_out = moe_sorting(
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

        grid = sorted_expert_ids.shape[0]
        if token_num * topk <= E:
            grid = token_num * topk

        #with pyhip.cudaPerf(rw_bytes=num_valid_ids[0]/block_size_M*(model_dim * inter_dim * 2 * 2), name="moe_2stage_splitk(12)"):
        if 1:
            moe_2stage_splitk([inter_dim*2 // block_size_N, grid], [256],
                                w1.dtype, topk, model_dim, inter_dim*2, True, block_size_M, block_size_N,
                                hidden_states.data_ptr(), w1.data_ptr(), a2.data_ptr(),
                                sorted_ids.data_ptr(), sorted_weights.data_ptr(), sorted_expert_ids.data_ptr(), num_valid_ids.data_ptr(), w1_scale.data_ptr() if w1_scale is not None else 0, token_num)
        #with pyhip.cudaPerf(rw_bytes=num_valid_ids[0]/block_size_M*(model_dim * inter_dim * 1 * 2), name=f"moe_2stage_splitk(3)-{block_size_M}-{num_valid_ids[0].item()//block_size_M}"):
        if 1:
            if 1:
                moe_2stage_splitk([model_dim // block_size_N, grid], [64],
                            w1.dtype, topk, inter_dim, model_dim, False, block_size_M, block_size_N,
                            a2.data_ptr(), w2.data_ptr(), moe_out.data_ptr(),
                            sorted_ids.data_ptr(), sorted_weights.data_ptr(), sorted_expert_ids.data_ptr(), num_valid_ids.data_ptr(), w2_scale.data_ptr() if w2_scale is not None else 0, token_num)
            else:
                moe_2stage_down([1, sorted_expert_ids.shape[0]], [256],
                            w1.dtype, topk, inter_dim, model_dim, False, block_size_M, block_size_N,
                            a2.data_ptr(), w2.data_ptr(), moe_out.data_ptr(),
                            sorted_ids.data_ptr(), sorted_weights.data_ptr(), sorted_expert_ids.data_ptr(), num_valid_ids.data_ptr(), w2_scale.data_ptr() if w2_scale is not None else 0, token_num)


        return moe_out


    # prefill phase kernels, estimated num-tokens per expert is big enough
    # determine block_size_M:
    #   set to 128 if num_local_tokens is too small to use all available CUs
    block_size_M = 256
    block_size_N = 256

    while estimated_tokens_per_expert < 128:
        estimated_oc_blocks = min(inter_dim*2, model_dim)//block_size_N
        estimated_num_e_blocks = ((token_num * topk) // block_size_M)
        estimated_work_groups = estimated_num_e_blocks * estimated_oc_blocks

        cu_usage = estimated_work_groups / torch.cuda.get_device_properties().multi_processor_count
        if cu_usage != int(cu_usage):
            wasted_usage = (1 - (cu_usage - int(cu_usage))) / cu_usage
        else:
            wasted_usage = 0
        #print(f"{estimated_work_groups=} {torch.cuda.get_device_properties().multi_processor_count=} {cu_usage=} {wasted_usage=}")

        if cu_usage < 1 or wasted_usage > 0.05:
            if block_size_M > 128:
                block_size_M = 128
            else:
                # quant fp8 has some limitations in block size
                if quant_type == aiter.QuantType.No:
                    block_size_N = 128
                break
        else:
            break
    
    # keep block_size_N at 256 in most-case, only fallback to 128 on extream cases?
    block_size_N = 256
    
    assert block_size_M in [128,256]
    assert block_size_N in [128,256]

    #print(f"{block_size_M=} {block_size_N=}")

    #with pyhip.cudaPerf(0, name="moe_sorting"):
    if 1:
        sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_out = moe_sorting(
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

    a1, a1_scale = act_quant_func(
        hidden_states,
        scale=a1_scale,
        quant_dtype=q_dtype_a,
        num_rows=num_local_tokens,
        transpose_scale=True
    )

    if verbose:
        print("\ta1        :", list(a1.shape), a1.dtype)                # [token_num, model_dim]        torch.float8_e4m3fn
        if a1_scale is not None:
            print("\ta1_scale  :", list(a1_scale.shape), a1_scale.dtype)    # [token_num, model_dim//128]   torch.float32

    """
    ratio = a1_scale.element_size() // a1.element_size()
    a2 = torch.empty(
        (token_num + (token_num * ratio + 127) // 128, topk, inter_dim),
        dtype=q_dtype_a,
        device=device,
    )
    """

    w1_scale = w1_scale.view(dtypes.fp8_e8m0) if w1.dtype == dtypes.fp4x2 else w1_scale

    wg_M, wg_N = block_size_M, block_size_N
    assert inter_dim*2 % wg_N == 0
    num_oc_blocks = inter_dim*2//wg_N
    num_e_blocks = sorted_expert_ids.shape[0]
    #valid_e_blocks = num_valid_ids[0].item()//wg_M
    if 0:
        moe_gemm_ref(topk, block_size_M, sorted_ids, sorted_expert_ids, sorted_weights, num_valid_ids,
                a1, a1_scale,
                w1, w1_scale, a2, True, w1_is_shuffled)
    else:
        if a1.dtype == torch.bfloat16:
            AB_dtype = "bf16"
        elif a1.dtype == torch.float8_e4m3fn and w1.dtype == torch.float8_e4m3fn:
            AB_dtype = "fp8"
        else:
            assert 0, f"{a1.dtype=} {w1.dtype=}"
        #print(sorted_ids)
        #print(sorted_expert_ids)
        #print(f"{num_oc_blocks=} {num_valid_ids[0].item()=} {valid_e_blocks=} / {num_e_blocks=} {inter_dim=}")
        #with pyhip.cudaPerf(num_oc_blocks*valid_e_blocks*wg_M*wg_N*model_dim*2, name="moe_gemm_8wave_gateup"):
        if 1:
            moe_gemm_8wave([num_oc_blocks, num_e_blocks], [8*64],
                    AB_dtype, wg_M, wg_N,
                    E, inter_dim*2, model_dim, 
                    True, w1_is_shuffled, topk,
                    sorted_ids.data_ptr(),
                    sorted_weights.data_ptr(),
                    sorted_expert_ids.data_ptr(),
                    num_valid_ids.data_ptr(),
                    w1.data_ptr(), None if w1_scale is None else w1_scale.data_ptr(),
                    a1.data_ptr(), None if a1_scale is None else a1_scale.data_ptr(),
                    a2.data_ptr(),
                    token_num) # num_local_tokens.data_ptr() ?        

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

    if verbose:
        print("\ta2        :", list(a2.shape), a2.dtype)                # [token_num, topk, inter_dim]        torch.float8_e4m3fn
        print("\ta2_scale  :", list(a2_scale.shape), a2_scale.dtype)    # [token_num, topk, inter_dim//128]   torch.float32

    assert model_dim % wg_N == 0
    num_oc_blocks = model_dim//wg_N
    num_e_blocks = sorted_expert_ids.shape[0]

    #with pyhip.torchPerf("./xxx"):
    if 0:
        moe_gemm_ref(topk, block_size_M, sorted_ids, sorted_expert_ids, sorted_weights, num_valid_ids,
                a2, a2_scale,
                w2, w2_scale, moe_out, False, w2_is_shuffled)
    else:
        #print(f"{num_oc_blocks=} {valid_e_blocks=} {inter_dim=}")
        #with pyhip.cudaPerf(num_oc_blocks*valid_e_blocks*wg_M*wg_N*inter_dim*2, name="moe_gemm_8wave_down"):
        if 1:
            if a2.dtype == torch.bfloat16:
                AB_dtype = "bf16"
            elif a2.dtype == torch.float8_e4m3fn and w2.dtype == torch.float8_e4m3fn:
                AB_dtype = "fp8"
            else:
                assert 0, f"{a2.dtype=} {w2.dtype=}"
            #sorted_expert_ids[...] = 0
            moe_gemm_8wave([num_oc_blocks, num_e_blocks], [8*64],
                    AB_dtype, wg_M, wg_N,
                    E, model_dim, inter_dim, 
                    False, w2_is_shuffled, topk,
                    sorted_ids.data_ptr(),
                    sorted_weights.data_ptr(),
                    sorted_expert_ids.data_ptr(),
                    num_valid_ids.data_ptr(),
                    w2.data_ptr(), None if w2_scale is None else w2_scale.data_ptr(),
                    a2.data_ptr(), None if a2_scale is None else a2_scale.data_ptr(),
                    stage2_out.data_ptr(),
                    token_num) # num_local_tokens.data_ptr() ?

        if 0:
            moe_out = stage2_out[:,0,:]
            for i in range(1, topk):
                moe_out += stage2_out[:,i,:]
        else:
            num_WG = 256 * 2
            num_tokens_wg = token_num // num_WG
            num_extra_tokens = token_num % num_WG
            moe_gemm_final_reduce_bf16([num_WG], [64], topk, model_dim,
                                    stage2_out.data_ptr(),
                                    moe_out.data_ptr(),
                                    num_tokens_wg, num_extra_tokens, token_num)

    return moe_out