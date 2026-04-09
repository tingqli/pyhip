import pyhip
import torch
import aiter
from aiter import dtypes
import os
import contextlib

from .moe_gemm_8wave import moe_gemm_final_reduce_bf16, moe_gemm_8wave_g1u1
from .moe import moe_2stage_splitk, moe_2stage_gateup, moe_2stage_down
try:
    SPLITK = int(os.getenv("USE_GLUON_SPLITK", "1"))
    if SPLITK:
        from .gluon.moe_gemm_splitk import moe_2stage_splitk_gateup, moe_2stage_splitk_down
    else:
        from .gluon.moe_gemm import moe_2stage_gateup as moe_2stage_splitk_gateup, moe_2stage_down as moe_2stage_splitk_down
    from .gluon.moe_gemm_4wave import moe_2stage_gateup as moe_2stage_gateup_4w, moe_2stage_down as moe_2stage_down_4w
except:
    moe_2stage_splitk_gateup = None
    moe_2stage_splitk_down = None

from .moe_gemm_down_tp import *

from .fused_moe_gelu import fused_moe_gelu

from .moe_gemm_ref import moe_gemm_ref

USE_GLUON = int(os.getenv("USE_GLUON", "1"))
USE_GLUON2 = int(os.getenv("USE_GLUON2", "0"))
VERBOSE = int(os.getenv("VERBOSE", "0"))

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
    method="auto", # jit, gluon, auto
):
    if activation == aiter.ActivationType.Gelu:
        return fused_moe_gelu(hidden_states, w1, w2, topk_weight, topk_ids, 
                                expert_mask = expert_mask,
                                activation = activation,
                                quant_type = quant_type,
                                doweight_stage1 = doweight_stage1,
                                w1_scale = w1_scale,
                                w2_scale = w2_scale,
                                a1_scale = a1_scale,
                                a2_scale = a2_scale,
                                block_size_M = block_size_M,
                                num_local_tokens = num_local_tokens,
                                moe_sorting_dispatch_policy = moe_sorting_dispatch_policy,
                                dtype = dtype,
                                hidden_pad = hidden_pad,
                                intermediate_pad = intermediate_pad,
                                bias1 = bias1,
                                bias2 = bias2,
                                splitk = splitk,
                                method = method)

    device = hidden_states.device
    def get_inter_dim(w1_shape, w2_shape):
        E, _, model_dim = w1_shape
        E, model_dim, inter_dim = w2_shape

        int4_war = model_dim // w1_shape[-1]
        inter_dim *= int4_war
        return E, model_dim, inter_dim

    M, topk = topk_ids.shape
    E, model_dim, inter_dim = get_inter_dim(w1.shape, w2.shape)
    assert w1.shape[1] == inter_dim * 2 or w1.shape[1] == inter_dim

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

    if hidden_states.device.index == 0:
        global _call_times
        if hidden_states.shape[0] == 16384 and _call_times == 0:
            # torch.save(topk_ids, "topk_ids.pt")
            # torch.load("/root/sglang/topk_ids.pt")
            _call_times += 1
        #print("==== ", list(hidden_states.shape), list(w1.shape), w1.dtype, list(w2.shape), w2.dtype)
        

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

    if quant_type == aiter.QuantType.per_128x128:
        # 128x128 block-scale quant on weights, 1x128 per-token quant on act
        act_quant_func = aiter.get_hip_quant(aiter.QuantType.per_1x128)
    else:
        act_quant_func = aiter.get_hip_quant(quant_type)

    #moe_sorting = moe_sorting_ref
    moe_sorting = aiter.fused_moe.moe_sorting

    estimated_tokens_per_expert = token_num * topk / global_E

    if method in ["gluon", "auto"] and \
        estimated_tokens_per_expert < 32 and (
        quant_type == aiter.QuantType.No or
        # (wei_is_fp8(w1.dtype) and quant_type == aiter.QuantType.per_Token) or
        (wei_is_fp8(w1.dtype) and quant_type == aiter.QuantType.per_128x128)
        ):
        # mem-bound case : very small num_tokens
        def blk_sz_heuristics():
            #base on Qwen 3.5 TP8 case tested result
            if model_dim == 4096 and inter_dim == 128 and wei_is_fp8(w1.dtype):
                return 16, 64
            else:
                return 32, 128
        block_size_M, block_size_N = blk_sz_heuristics()
        sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_out = moe_sorting(
            topk_ids, topk_weight, global_E, model_dim,  dtype,
            block_size_M,
            expert_mask, num_local_tokens, moe_sorting_dispatch_policy,
        )
        debug_verbose_moe_sorting(sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_out, token_num, topk, block_size_M, estimated_tokens_per_expert)

        grid = sorted_expert_ids.shape[0]
        if token_num * topk <= E:
            grid = token_num * topk

        fp8_is_ptpc =  (quant_type == aiter.QuantType.per_Token and wei_is_fp8(w1.dtype))
        moe_2stage_splitk_gateup
        moe_2stage_splitk_down

        if USE_GLUON:
            BLOCK_TILE_SIZE_M = block_size_M
            BLOCK_TILE_SIZE_N = block_size_N
            grid = sorted_expert_ids.shape[0]
            B = token_num
            N1, K1 = inter_dim * 2, model_dim
            N2, K2 = model_dim, inter_dim
            if B * topk <= E:
                grid = B * topk
            moe_2stage_splitk_gateup[(N1 // BLOCK_TILE_SIZE_N, grid)](
                                hidden_states, w1, a2, sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, w1_scale[0] if w1_scale is not None else None, B,
                                w1.dtype, topk, K1, N1, BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N, 4)
            moe_2stage_splitk_down[(N2 // BLOCK_TILE_SIZE_N, grid)](
                                a2, w2, moe_out, sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, w2_scale[0] if w2_scale is not None else None, B,
                                w1.dtype, topk, K2, N2, BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N, num_warps=1 if SPLITK else 4)
        else:
            #with pyhip.cudaPerf(rw_bytes=num_valid_ids[0]/block_size_M*(model_dim * inter_dim * 2 * 2), name="moe_2stage_splitk(12)"):
            if 1:
                moe_2stage_splitk([inter_dim*2 // block_size_N, grid], [256],
                                    w1.dtype, topk, model_dim, inter_dim*2, True, block_size_M, block_size_N,
                                    hidden_states.data_ptr(), w1.data_ptr(), a2.data_ptr(),
                                    sorted_ids.data_ptr(), sorted_weights.data_ptr(), sorted_expert_ids.data_ptr(), num_valid_ids.data_ptr(), w1_scale.data_ptr() if w1_scale is not None else 0, token_num, fp8_is_ptpc)
            #with pyhip.cudaPerf(rw_bytes=num_valid_ids[0]/block_size_M*(model_dim * inter_dim * 1 * 2), name=f"moe_2stage_splitk(3)-{block_size_M}-{num_valid_ids[0].item()//block_size_M}"):
            if 1:
                if 1:
                    moe_2stage_splitk([model_dim // block_size_N, grid], [64],
                                w1.dtype, topk, inter_dim, model_dim, False, block_size_M, block_size_N,
                                a2.data_ptr(), w2.data_ptr(), moe_out.data_ptr(),
                                sorted_ids.data_ptr(), sorted_weights.data_ptr(), sorted_expert_ids.data_ptr(), num_valid_ids.data_ptr(), w2_scale.data_ptr() if w2_scale is not None else 0, token_num, fp8_is_ptpc)
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
    debug_verbose_moe_sorting(sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_out, token_num, topk, block_size_M, estimated_tokens_per_expert)

    a1, a1_scale = act_quant_func(
        hidden_states,
        scale=a1_scale,
        quant_dtype=q_dtype_a,
        num_rows=num_local_tokens,
        transpose_scale=True
    )

    if VERBOSE:
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
    do_perf = False
    if do_perf:
        valid_e_blocks = num_valid_ids[0].item()//wg_M
    if 0:
        moe_gemm_ref(activation, quant_type, topk, block_size_M, sorted_ids, sorted_expert_ids, sorted_weights, num_valid_ids,
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
        with contextlib.nullcontext() if not do_perf else pyhip.cudaPerf(num_oc_blocks*valid_e_blocks*wg_M*wg_N*model_dim*2, name=f"moe_gemm_8wave_gateup"):
            if USE_GLUON2:
                BLOCK_TILE_SIZE_M = block_size_M
                BLOCK_TILE_SIZE_N = block_size_N
                grid = sorted_expert_ids.shape[0]
                B = token_num
                N1, K1 = inter_dim * 2, model_dim
                N2, K2 = model_dim, inter_dim
                if B * topk <= E:
                    grid = B * topk
                moe_2stage_gateup_4w[(N1 // BLOCK_TILE_SIZE_N, grid)](
                                    hidden_states, w1, a2, sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, w1_scale[0] if w1_scale is not None else None, B,
                                    w1.dtype, topk, K1, N1, BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N, 4)
            else:
                moe_gemm_8wave_g1u1([num_oc_blocks, num_e_blocks], [8*64],
                        a1.element_size() * a1.numel() > (1<<32),
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

    if VERBOSE:
        print("\ta2        :", list(a2.shape), a2.dtype)                # [token_num, topk, inter_dim]        torch.float8_e4m3fn
        if a2_scale is not None:
            print("\ta2_scale  :", list(a2_scale.shape), a2_scale.dtype)    # [token_num, topk, inter_dim//128]   torch.float32

    assert model_dim % wg_N == 0
    num_oc_blocks = model_dim//wg_N
    num_e_blocks = sorted_expert_ids.shape[0]

    #with pyhip.torchPerf("./xxx"):
    if 0:
        moe_gemm_ref(activation, quant_type, topk, block_size_M, sorted_ids, sorted_expert_ids, sorted_weights, num_valid_ids,
                a2, a2_scale,
                w2, w2_scale, moe_out, False, w2_is_shuffled)
    else:
        #print(f"{num_oc_blocks=} {valid_e_blocks=} {inter_dim=}")
        #print(w2.dtype, w2.shape, a2.dtype, a2.shape)
        if do_perf:
            rw_bytes = w2.numel() * w2.element_size() + a2.numel() * a2.element_size() + stage2_out.numel()*stage2_out.element_size()
            flops = num_oc_blocks*valid_e_blocks*wg_M*wg_N*inter_dim*2
        with contextlib.nullcontext() if not do_perf else pyhip.cudaPerf(flops=flops, rw_bytes=rw_bytes, name="moe_gemm_down         "):
            if a2.dtype == torch.bfloat16:
                AB_dtype = "bf16"
            elif a2.dtype == torch.float8_e4m3fn and w2.dtype == torch.float8_e4m3fn:
                AB_dtype = "fp8"
            else:
                assert 0, f"{a2.dtype=} {w2.dtype=}"
            #sorted_expert_ids[...] = 0
            if inter_dim <= 256 and w2_is_shuffled:
                moe_gemm_down_tp([1, num_e_blocks], [4*64],
                                stage2_out.element_size() * stage2_out.numel() > (1<<32),
                                AB_dtype, wg_M, 64,
                                E, model_dim, inter_dim, 
                                False, w2_is_shuffled, topk,
                                sorted_ids.data_ptr(),
                                sorted_weights.data_ptr(),
                                sorted_expert_ids.data_ptr(),
                                num_valid_ids.data_ptr(),
                                w2.data_ptr(), None if w2_scale is None else w2_scale.data_ptr(),
                                a2.data_ptr(), None if a2_scale is None else a2_scale.data_ptr(),
                                stage2_out.data_ptr(),
                                token_num)
            else:
                if USE_GLUON2:
                    BLOCK_TILE_SIZE_M = block_size_M
                    BLOCK_TILE_SIZE_N = block_size_N
                    grid = sorted_expert_ids.shape[0]
                    B = token_num
                    N1, K1 = inter_dim * 2, model_dim
                    N2, K2 = model_dim, inter_dim
                    if B * topk <= E:
                        grid = B * topk
                    moe_2stage_down_4w[(N2 // BLOCK_TILE_SIZE_N, grid)](
                                        a2, w2, stage2_out, sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, w2_scale[0] if w2_scale is not None else None, B,
                                        w1.dtype, topk, K2, N2, BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N, 4)
                else:
                    moe_gemm_8wave_g1u1([num_oc_blocks, num_e_blocks], [8*64],
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

        if 1:
            moe_out = stage2_out.sum(dim=1)
        else:
            num_WG = 256 * 2
            num_tokens_wg = token_num // num_WG
            num_extra_tokens = token_num % num_WG
            moe_gemm_final_reduce_bf16([num_WG], [64], topk, model_dim,
                                    stage2_out.data_ptr(),
                                    moe_out.data_ptr(),
                                    num_tokens_wg, num_extra_tokens, token_num)

    return moe_out