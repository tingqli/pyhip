import pyhip
import torch
import aiter
from aiter import dtypes
from aiter.utility import fp4_utils
import os
import contextlib

from .moe_gemm_8wave_gelu import moe_gemm_8wave_gelu

VERBOSE = int(os.getenv("VERBOSE", "0"))

__all__ = [
    "fused_moe_gelu"
]
from .moe_gemm_ref import moe_gemm_ref

def debug_verbose_moe_sorting(sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_out, token_num, topk, block_size_M, estimated_tokens_per_expert):
    if VERBOSE:
        num_expert_blocks = num_valid_ids[0].item()//block_size_M
        print("estimated_tokens_per_expert        :", estimated_tokens_per_expert)
        print("num_expert_blocks :", num_expert_blocks)
        print("hit_experts       :", len(set(sorted_expert_ids.tolist()[:num_expert_blocks])))
        print("sorted_ids        :", list(sorted_ids.shape), sorted_ids.dtype)
        for n in [0,1, num_expert_blocks-1]:
            n0 = n*block_size_M
            n1 = n0 + block_size_M
            print("\t", (sorted_ids[n0:n1] & 0xFFFFFF).tolist())

        hit_tokens = len(set(sorted_ids[:num_expert_blocks*block_size_M]))
        total_tokens = token_num * topk
        print(f"hit tokens        :{hit_tokens} / {total_tokens} = {hit_tokens/total_tokens*100:.1f} %", )

        print("sorted_weights    :", list(sorted_weights.shape), sorted_weights.dtype)
        print("sorted_expert_ids :", list(sorted_expert_ids.shape), sorted_expert_ids.dtype, sorted_expert_ids.tolist()[:num_expert_blocks])
        print("num_valid_ids     :", list(num_valid_ids.shape), num_valid_ids.dtype, num_valid_ids.tolist())
        print("moe_out           :", list(moe_out.shape), moe_out.dtype)

def wei_is_fp8(weight_type):
    return weight_type == dtypes.fp8

_call_times = 0

def fused_moe_gelu(
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
    method="auto", # jit, gluon, auto
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
    assert w1.shape[1] == inter_dim

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
    assert not isG1U1
    assert activation == aiter.ActivationType.Gelu

    global_E = E
    if expert_mask is not None:
        global_E = expert_mask.numel()

    dtype = hidden_states.dtype if dtype is None else dtype
    assert dtype == aiter.dtypes.bf16
    assert w1.dtype == w2.dtype
    q_dtype_w = w1.dtype
    q_dtype_a = dtype if quant_type == aiter.QuantType.No else w1.dtype

    if quant_type == aiter.QuantType.per_1x32:
        assert q_dtype_w == torch.float4_e2m1fn_x2
    elif quant_type == aiter.QuantType.No:
        assert q_dtype_w == torch.bfloat16
    else:
        assert 0, f"{quant_type=}"

    assert doweight_stage1 == False
    assert a1_scale is None
    assert a2_scale is None
    assert hidden_pad == 0
    assert intermediate_pad == 0
    assert bias1 is None
    assert bias2 is None
    assert splitk == 0

    if VERBOSE:
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
            print("\t", expert_mask)
        if w1_scale is not None:
            print("\tw1_scale:", list(w1_scale.shape), w1_scale.dtype)              # w1_scale     : [E, inter_dim*2//128,  model_dim//128]  torch.float32
        if w2_scale is not None:
            print("\tw2_scale:", list(w2_scale.shape), w2_scale.dtype)              # w2_scale     : [E, model_dim//128, inter_dim//128]     torch.float32

    token_num, _ = hidden_states.shape

    a2 = torch.empty(
        (token_num, topk, inter_dim),
        dtype=torch.bfloat16,
        device=device,
    )

    act_quant_func = aiter.get_hip_quant(quant_type)

    #moe_sorting = moe_sorting_ref
    moe_sorting = aiter.fused_moe.moe_sorting
    block_size_M = 256
    block_size_N = 256

    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_out = moe_sorting(
        topk_ids, topk_weight, global_E, model_dim,  dtype,
        block_size_M,
        expert_mask, num_local_tokens, moe_sorting_dispatch_policy,
    )
    debug_verbose_moe_sorting(sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_out, token_num, topk, block_size_M, 0)

    # 128x128 FP8 needs transposed-scale
    # 1x32 MXFP4 don't need it
    if quant_type == aiter.QuantType.per_1x32:
        a1, a1_scale = act_quant_func(
            hidden_states,
            scale=a1_scale,
            quant_dtype=q_dtype_a,
            num_rows=num_local_tokens
        )
        a1_scale = fp4_utils.moe_mxfp4_sort(a1_scale, sorted_ids=sorted_ids, num_valid_ids=num_valid_ids, token_num=token_num, block_size=block_size_M)
    else:
        a1, a1_scale = act_quant_func(
            hidden_states,
            scale=a1_scale,
            quant_dtype=q_dtype_a,
            num_rows=num_local_tokens,
            transpose_scale=True
        )

    do_perf = 0

    wg_M, wg_N = block_size_M, block_size_N
    assert inter_dim % wg_N == 0
    num_oc_blocks = inter_dim//wg_N
    num_e_blocks = sorted_expert_ids.shape[0]
    valid_e_blocks = num_valid_ids[0].item()//wg_M if do_perf else 0
    AB_dtype = "bf16"

    if quant_type == aiter.QuantType.per_1x32:
        moe_gemm_ref(activation, quant_type, topk, block_size_M, sorted_ids, sorted_expert_ids, sorted_weights, num_valid_ids,
                    a1, a1_scale, w1, w1_scale, a2, True, w1_is_shuffled)
    else:
        flops = num_oc_blocks*valid_e_blocks*wg_M*wg_N*model_dim*2
        #flops = token_num * topk * inter_dim * model_dim * 2
        with contextlib.nullcontext() if not do_perf else pyhip.cudaPerf(flops, name=f"moe_gemm_8wave_gelu[up  ]"):
            moe_gemm_8wave_gelu([num_oc_blocks, num_e_blocks], [8*64],
                    a1.element_size() * a1.numel() > (1<<32),
                    AB_dtype, wg_M, wg_N,
                    E, inter_dim, model_dim, 
                    True, w1_is_shuffled, topk,
                    sorted_ids.data_ptr(),
                    sorted_weights.data_ptr(),
                    sorted_expert_ids.data_ptr(),
                    num_valid_ids.data_ptr(),
                    w1.data_ptr(), None if w1_scale is None else w1_scale.data_ptr(),
                    a1.data_ptr(), None if a1_scale is None else a1_scale.data_ptr(),
                    a2.data_ptr(),
                    token_num) # num_local_tokens.data_ptr() ?

    if quant_type == aiter.QuantType.per_1x32:
        a2, a2_scale = act_quant_func(
            a2.view(token_num*topk, -1),
            scale=a2_scale,
            quant_dtype=q_dtype_a,
            num_rows=num_local_tokens,
        )
    else:
        a2, a2_scale = act_quant_func(
            a2, #a2.view(token_num*topk, -1),
            scale=a2_scale,
            quant_dtype=q_dtype_a,
            num_rows=num_local_tokens,
            num_rows_factor=topk,
            transpose_scale=True
        )
    a2 = a2.view(token_num, topk, -1)
    w2_scale = w2_scale.view(dtypes.fp8_e8m0) if w2.dtype == dtypes.fp4x2 else w2_scale

    if quant_type == aiter.QuantType.per_1x32:
        a2_scale = fp4_utils.moe_mxfp4_sort(a2_scale[: token_num * topk, :].view(token_num, topk, -1), sorted_ids=sorted_ids, num_valid_ids=num_valid_ids, token_num=token_num,block_size=block_size_M,)

    if quant_type == aiter.QuantType.per_1x32:
        moe_gemm_ref(activation, quant_type, topk, block_size_M, sorted_ids, sorted_expert_ids, sorted_weights, num_valid_ids,
                    a2, a2_scale, w2, w2_scale, moe_out, False, w2_is_shuffled)
    else:
        num_oc_blocks = model_dim//wg_N
        num_e_blocks = sorted_expert_ids.shape[0]
        stage2_out = torch.empty((token_num, topk, model_dim), dtype=torch.bfloat16, device=device)
        flops = num_oc_blocks*valid_e_blocks*wg_M*wg_N*inter_dim*2
        # flops = token_num * topk * inter_dim * model_dim * 2
        with contextlib.nullcontext() if not do_perf else pyhip.cudaPerf(flops, name=f"moe_gemm_8wave_gelu[down]"):
            moe_gemm_8wave_gelu([num_oc_blocks, num_e_blocks], [8*64],
                            a2.element_size() * a2.numel() > (1<<32),
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
        moe_out = stage2_out.sum(dim=1)

    # moe_out = stage2_out.sum(dim=1)
    return moe_out
