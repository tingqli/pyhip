import pyhip
import torch
import aiter
from aiter import dtypes

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

"""

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

    verbose = 1

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

    assert w1_scale is not None
    assert w2_scale is not None

    assert isG1U1 == True
    assert isShuffled == True


    # determine block_size_M
    block_size_M = 256

    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_out = aiter.fused_moe.moe_sorting(
        topk_ids,
        topk_weight,
        global_E,
        model_dim,
        dtype,
        block_size_M,
        expert_mask,
        num_local_tokens,
        moe_sorting_dispatch_policy,
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

    quant_func = aiter.get_hip_quant(aiter.QuantType.per_1x128)
    a1, a1_scale = quant_func(
        hidden_states,
        scale=a1_scale,
        quant_dtype=q_dtype_a,
        num_rows=num_local_tokens,
        transpose_scale=True
    )

    token_num, _ = hidden_states.shape

    ratio = a1_scale.element_size() // a1.element_size()
    a2 = torch.empty(
        (token_num + (token_num * ratio + 127) // 128, topk, inter_dim),
        dtype=q_dtype_a,
        device=device,
    )

    def moe_stage1(a1, w1, w2, sorted_ids, sorted_expert_ids, num_valid_ids, a2, topk, block_m, a1_scale, w1_scale, sorted_weights, **kwargs):
        a2_bf16 = torch.empty(
            (token_num, topk, inter_dim),
            dtype=torch.bfloat16,
            device=device,
        )
        return a2_bf16
    def moe_stage2(a2, w1, w2, sorted_ids, sorted_expert_ids, num_valid_ids, moe_out, topk, w2_scale, a2_scale, block_m, sorted_weights, **kwargs):
        return moe_out

    extra_stage1_args = {}
    extra_stage2_args = {}

    a2 = moe_stage1(
        a1,
        w1,
        w2,
        sorted_ids,
        sorted_expert_ids,
        num_valid_ids,
        a2,
        topk,
        block_m=block_size_M,
        a1_scale=a1_scale,
        w1_scale=(
            w1_scale.view(dtypes.fp8_e8m0) if w1.dtype == dtypes.fp4x2 else w1_scale
        ),
        sorted_weights=sorted_weights if doweight_stage1 else None,
        **extra_stage1_args,
    )

    if quant_type != aiter.QuantType.No:
        a2, a2_scale = quant_func(
            a2,
            scale=a2_scale,
            quant_dtype=q_dtype_a,
            num_rows=num_local_tokens,
            num_rows_factor=topk,
        )
        a2 = a2.view(token_num, topk, inter_dim)

    moe_stage2(
        a2,
        w1,
        w2,
        sorted_ids,
        sorted_expert_ids,
        num_valid_ids,
        moe_out,
        topk,
        w2_scale=(
            w2_scale.view(dtypes.fp8_e8m0) if w2.dtype == dtypes.fp4x2 else w2_scale
        ),
        a2_scale=a2_scale,
        block_m=block_size_M,
        sorted_weights=sorted_weights if not doweight_stage1 else None,
        **extra_stage2_args,
    )

    return moe_out