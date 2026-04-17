import torch
import aiter

from aiter.fused_moe_bf16_asm import asm_moe, torch_moe
from aiter.fused_moe import moe_sorting, fused_moe
from aiter import ActivationType
from aiter.jit.utils.chip_info import get_gfx
from aiter import QuantType
from aiter.utility import fp4_utils

import pyhip
from pyhip.contrib.moe_gemm_8wave_gelu import moe_gemm_8wave_gelu

def div_up(x, y):
    return (x + y - 1) // y

def reduce_i8(x, scale, block_quant=256):
    kwargs = {"BLOCK_SIZE_M":256,
              "TOPK":x.shape[1],
              "ROW_PER_BLOCK":4,
              "ROW_PER_BLOCK2":4,
              "ROW_PER_BLOCK1":1,
              "BLOCK_M2":8,
              "QUANT1_K":x.shape[2],
              "QUANT2_K":x.shape[2],
              "REDUCE_K":x.shape[2],
              "BLOCK_QUANT":block_quant,}
    hip = pyhip.module(f"quant-i8.cpp")
    out = torch.empty((x.shape[0], x.shape[2]), dtype=torch.bfloat16, device=x.device)
    hip.reduce_i8([x.shape[0]], [256], x, scale, out, x.shape[0], **kwargs)
    return out

def quant_act(x, topk, M, model_dim, smooth_scale, sorted_ids, sorted_expert_ids, num_valid_ids, is_gemm1=True, topk_ids=None):
    hip = pyhip.module(f"quant-i8.cpp")
    kwargs = {"BLOCK_SIZE_M":256,
              "TOPK":topk,
              "ROW_PER_BLOCK":4,
              "ROW_PER_BLOCK2":4,
              "ROW_PER_BLOCK1":1,
              "BLOCK_M2":8,
              "QUANT1_K":model_dim,
              "QUANT2_K":model_dim,
              "REDUCE_K":model_dim,
              "BLOCK_QUANT":256,}
    device = x.device
    DEBUG = False
    if DEBUG:
        x_quant = torch.ones((M, topk, model_dim), dtype=torch.int8, device=device)
        x_quant_scale = torch.ones([sorted_ids.shape[0]], dtype=torch.float32, device=device)
    else:
        x_quant = torch.empty((M, topk, model_dim), dtype=torch.int8, device=device)
        x_quant_scale = torch.empty([sorted_ids.shape[0]], dtype=torch.float32, device=device)
    if is_gemm1:
        if smooth_scale.shape[0] == 1:
            hip.quant1_notopk([M], [64], 
                x, smooth_scale, x_quant, x_quant_scale, 
                topk_ids, M, **kwargs
                )
            x_quant_scale.is_sorted = True
        else:
            hip.quant1([2*div_up(M, kwargs["ROW_PER_BLOCK1"])], [64], 
                x, smooth_scale, x_quant, x_quant_scale, 
                topk_ids, M, **kwargs)
            x_quant_scale.is_sorted = False
    else:
        hip.quant2([sorted_expert_ids.shape[0], kwargs["BLOCK_SIZE_M"] // kwargs["BLOCK_M2"]], [256], 
            x, smooth_scale, x_quant, x_quant_scale, 
            sorted_ids, sorted_expert_ids, num_valid_ids,
            M, **kwargs)
        x_quant_scale.is_sorted = True
    return x_quant, x_quant_scale

def smooth_quant_w_i8(weight,         # [num_experts, OC, IC]
                     smooth_scale,   # [num_experts, IC]
                     ):
    scaled_weight = weight / smooth_scale
    per_oc_scale = ((scaled_weight.abs().max(dim=2, keepdim=True)[0])/128) # [num_experts, OC, 1]
    quanted_weight = (scaled_weight / per_oc_scale).round().clamp(-128, 127).to(torch.int8)
    return quanted_weight, per_oc_scale # [num_experts, OC, IC], [num_experts, OC, 1]

def smooth_quant_w_mxfp4(weight,         # [num_experts, OC, IC]
                     smooth_scale,       # [num_experts, IC]
                     ):
    num_experts, OC, IC = weight.shape
    quant_1x32 = aiter.get_torch_quant(aiter.QuantType.per_1x32)
    scaled_weight = weight / smooth_scale
    qa, qs = quant_1x32(scaled_weight, quant_dtype=aiter.dtypes.fp4x2, shuffle=False)
    return qa, qs.view(num_experts, OC, IC//32) # [num_experts, OC, IC//2] torch.float4_e2m1fn_x2, [num_experts, OC, IC//32] torch.float8_e8m0fnu

def moe_smoothquant_gelu_ref(
        hidden_states,  # [num_tokens, model_dim] torch.bfloat16
        w1,             # [expert, inter_dim, model_dim] torch.int8
        w2,             # [expert, model_dim, inter_dim] torch.int8
        topk_weight,    # [num_tokens, topk] torch.float32
        topk_ids,       # [num_tokens, topk] torch.int32
        w1_scale, # [expert, inter_dim, 1] torch.float32
        w2_scale, # [expert, model_dim, 1] torch.float32
        a1_scale, # [expert, 1, model_dim] torch.float32
        a2_scale, # [expert, 1, inter_dim] torch.float32
        ):

    num_experts, inter_dim, model_dim = w1.shape

    a1_scale = a1_scale.expand(num_experts, -1, -1)
    a2_scale = a2_scale.expand(num_experts, -1, -1)
    def int8_ptpc_mm(a, b, b_pc_scale):
        #return (a.to(torch.float32) @ b.to(torch.float32)) * b_pc_scale.reshape(1, -1)
        pt_scale_a = ((a.abs().max(dim=1)[0])/128)[:, None]
        # input after gelu may become zero, avoid div zero
        pt_scale_a = torch.maximum(pt_scale_a, torch.tensor(0.00001))
        a_i8 = (a / pt_scale_a).round().clamp(-128, 127).to(torch.float32)
        c = a_i8 @ b.to(torch.float32)
        return c * pt_scale_a * b_pc_scale.reshape(1, -1)

    final_hidden_states = torch.zeros_like(hidden_states)
    with torch.no_grad():
        expert_mask = torch.nn.functional.one_hot(topk_ids.long(), num_classes=num_experts)
        expert_mask = expert_mask.permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

    for expert_idx in expert_hit:
        expert_idx = expert_idx[0]
        if expert_idx == num_experts:
            continue
        top_k_pos, token_idx = torch.where(expert_mask[expert_idx])

        current_state = hidden_states[token_idx]

        current_state *= a1_scale[expert_idx]

        up = int8_ptpc_mm(current_state, w1[expert_idx].t(), w1_scale[expert_idx])

        current_hidden_states = torch.nn.functional.gelu(up)

        # after gelu, some activation becomes zeros, per-token quant may results 0
        current_hidden_states = int8_ptpc_mm(current_hidden_states * a2_scale[expert_idx],
                                            w2[expert_idx].t(), w2_scale[expert_idx])

        current_hidden_states = current_hidden_states * topk_weight[token_idx, top_k_pos, None]
        final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

    return final_hidden_states


def smoothquant_i8_per_tok(x, x_smooth_scale, topk, sorted_ids, sorted_expert_ids, num_valid_ids, block_size_M):
    num_tokens = x.shape[0]
    dims = x.shape[-1]

    if x_smooth_scale.shape[0] == 1:
        # shared smooth-scale, just quant each token
        pt_scales = torch.maximum(x.abs().max(dim=(x.ndim-1), keepdim=True)[0] / 128, torch.tensor(0.0000001))
        quant_x = (x * x_smooth_scale[0,...] / pt_scales).round().clamp(-128, 127).to(torch.int8)

        pt_scales = pt_scales.to(torch.float32)
        pt_scales.is_sorted = False
        return quant_x, pt_scales

    # per-expert smooth-scale
    quant_x = torch.empty([num_tokens, topk, dims], dtype=torch.int8)
    pt_scales = torch.zeros([len(sorted_ids)], dtype=torch.float32)
    for eblock_i, expert_id in enumerate(sorted_expert_ids):
        i0 = eblock_i * block_size_M
        i1 = i0 + block_size_M

        if i0 >= num_valid_ids[0]: break

        ids = sorted_ids[i0:i1]
        ids = ids[((ids >> 24) & 0xFF) < topk]

        cur_num_toks = len(ids)

        tok_id = ids & 0xFFFFFF
        topk_id = (ids >> 24 & 0xFF)

        if x.ndim == 3:
            tok_states = x[tok_id, topk_id, :]
        else:
            tok_states = x[tok_id, :]

        # [cur_num_toks, dims] * [expert, 1, dims]
        tok_states = tok_states * x_smooth_scale[expert_id, 0:1, :] 
        pts = torch.maximum(tok_states.abs().max(dim=1, keepdim=True)[0] / 128, torch.tensor(0.0000001))
        quant_x[tok_id, topk_id, :] = (tok_states / pts).round().clamp(-128, 127).to(torch.int8)

        pt_scales[i0:i0+cur_num_toks] = pts.view(-1)

    pt_scales.is_sorted = True
    return quant_x, pt_scales


def fused_moe_gelu_sqi8(
        hidden_states,  # [num_tokens, model_dim] torch.bfloat16
        w1,             # [expert, inter_dim, model_dim] torch.int8
        w2,             # [expert, model_dim, inter_dim] torch.int8
        topk_weight,    # [num_tokens, topk] torch.float32
        topk_ids,       # [num_tokens, topk] torch.int32
        w1_scale, # [expert, inter_dim, 1] torch.float32
        w2_scale, # [expert, model_dim, 1] torch.float32
        a1_smooth_scale, # [expert/1, 1, model_dim] torch.float32
        a2_smooth_scale, # [expert/1, 1, inter_dim] torch.float32):
    ):
    device = hidden_states.device
    dtype = hidden_states.dtype
    def get_inter_dim(w1_shape, w2_shape):
        E, _, model_dim = w1_shape
        E, model_dim, inter_dim = w2_shape

        int4_war = model_dim // w1_shape[-1]
        inter_dim *= int4_war
        return E, model_dim, inter_dim
    M, topk = topk_ids.shape
    E, model_dim, inter_dim = get_inter_dim(w1.shape, w2.shape)
    if isinstance(w1, torch.nn.parameter.Parameter):
        w1_is_shuffled = True
    else:
        w1_is_shuffled = getattr(w1, "is_shuffled", False)

    if isinstance(w2, torch.nn.parameter.Parameter):
        w2_is_shuffled = True
    else:
        w2_is_shuffled = getattr(w2, "is_shuffled", False)
    assert w1.shape[1] == inter_dim

    block_size_M = 256
    block_size_N = 256

    expert_mask = None
    num_local_tokens = None
    moe_sorting_dispatch_policy = 0

    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_out = moe_sorting(
        topk_ids, topk_weight, E, model_dim,  dtype,
        block_size_M,
        expert_mask, num_local_tokens, moe_sorting_dispatch_policy,
    )

    # quantize hidden_states in sorted_ids [tok_topk]
    # print(sorted_ids.dtype, sorted_ids.shape)

    def moe_gemm_ref(input, pt_scale, weight, pc_scale, stage_index, oquant_block_size):
        num_tokens, topk, dims = input.shape
        num_experts, odims, idims = weight.shape
        if oquant_block_size:
            assert odims % oquant_block_size == 0
            output = torch.empty((num_tokens, topk, odims), dtype=torch.int8, device=device)
            o_scales = torch.empty((num_tokens, topk, odims//oquant_block_size), dtype=torch.float32, device=device)
        else:
            output = torch.empty((num_tokens, topk, odims), dtype=torch.bfloat16, device=device)
            o_scales = None

        assert input.ndim == 3
        for eblock_i, expert_id in enumerate(sorted_expert_ids):
            i0 = eblock_i * block_size_M
            i1 = i0 + block_size_M
            ids = sorted_ids[i0:i1]
            if i0 >= num_valid_ids[0]: break
            if expert_id >= E or expert_id < 0: break

            ids = ids[((ids >> 24) & 0xFF) < topk]
            cur_num_toks = len(ids)

            tok_id = ids & 0xFFFFFF
            topk_id = (ids >> 24 & 0xFF)

            cur_input = input[tok_id, topk_id, :]
            cur_weight = weight[expert_id,...]

            cur_out = cur_input.to(torch.float32) @ cur_weight.to(torch.float32).t()

            if 0:
                print("==============")
                print(cur_out.shape)
                print(pt_scale[i0:(i0+cur_num_toks), None].shape)
                print(pc_scale[expert_id, None, :, 0].shape)
                assert 0

            if getattr(pt_scale, "is_sorted", True):
                cur_pt_scale = pt_scale[i0:(i0+cur_num_toks), None]
            else:
                cur_pt_scale = pt_scale.view(-1, topk)[tok_id, topk_id, None]
            cur_out = cur_out * cur_pt_scale * pc_scale[expert_id, None, :, 0]

            if stage_index == 1:
                cur_out = torch.nn.functional.gelu(cur_out)
            else:
                cur_out = cur_out * sorted_weights[i0:i0+cur_num_toks, None]

            if oquant_block_size:
                scales = torch.maximum((cur_out.view(-1, oquant_block_size).abs().max(dim=1, keepdim=True)[0]), torch.tensor(0.0000001))/127
                output[tok_id, topk_id, :] = (cur_out.view(-1, oquant_block_size) / scales).round().clamp(-128, 127).to(torch.int8).view(-1, odims)
                o_scales[tok_id, topk_id, :] = scales.view(-1, odims//oquant_block_size)
            else:
                output[tok_id, topk_id, :] = cur_out.to(output.dtype)
        return output, o_scales

    def moe_gemm_jit(input, pt_scale, weight, pc_scale, stage_index, oquant_block_size):
        if input.ndim == 3:
            num_tokens, _, dims = input.shape
            is_in_3d = 1
        else:
            num_tokens, dims = input.shape
            is_in_3d = 0
        num_experts, odims, idims = weight.shape
        if oquant_block_size:
            assert odims % oquant_block_size == 0
            output = torch.empty((num_tokens, topk, odims), dtype=torch.int8, device=device)
            o_scales = torch.empty((num_tokens, topk, odims//oquant_block_size), dtype=torch.float32, device=device)
        else:
            output = torch.empty((num_tokens, topk, odims), dtype=torch.bfloat16, device=device)
            o_scales = None
        AB_dtype = "s8"
        is_over_4GB = input.numel() * input.element_size() > (1<<32)
        is_pts_sorted = getattr(pt_scale, "is_sorted", False)
        wg_M, wg_N = 256, 256
        is_gate_up = (stage_index == 1)
        bpreshuffle = False
        num_tokens = input.shape[0]
        num_oc_blocks = output.shape[-1] // wg_N
        num_e_blocks = sorted_expert_ids.shape[0]

        moe_gemm_8wave_gelu([num_oc_blocks * num_e_blocks],[8*64],
                        is_in_3d,
                        is_over_4GB,
                        is_pts_sorted,
                        AB_dtype, wg_M, wg_N,
                        output.shape[-1], input.shape[-1], 
                        is_gate_up, bpreshuffle,
                        topk,
                        sorted_ids,
                        sorted_weights,
                        sorted_expert_ids,
                        num_valid_ids,
                        weight, pc_scale,
                        input, pt_scale,
                        output,
                        o_scales,
                        num_tokens, num_oc_blocks * num_e_blocks)
        return output, o_scales

    def fake_quant(x, block_size = 256):
        RANG = 127
        scale = torch.maximum(x.view(-1, block_size).abs().max(dim=1, keepdim=True)[0]/(RANG-1), torch.tensor(0.0000001))
        x_i8 = (x.view(-1, block_size) / scale).round().clamp(-RANG, RANG-1)
        return (x_i8.to(x.dtype) * scale).view(x.shape)
    num_tokens, _ = hidden_states.shape

    if (a1_smooth_scale is None) or (a2_smooth_scale is None):
        pt_quant_func = aiter.get_hip_quant(aiter.QuantType.per_Token)

    with pyhip.cudaPerf(name=f"quant_act_up"):
        if a1_smooth_scale is None:
            a1, a1_scale = pt_quant_func(
                hidden_states,
                scale=None,
                quant_dtype=torch.int8,
            )
            a1_scale.is_sorted = False
        elif 0:
            a1, a1_scale = quant_act(hidden_states, topk, hidden_states.shape[0], hidden_states.shape[1], a1_smooth_scale, sorted_ids, sorted_expert_ids, num_valid_ids, True, topk_ids)
        else:
            a1, a1_scale = smoothquant_i8_per_tok(hidden_states, a1_smooth_scale, topk, sorted_ids, sorted_expert_ids, num_valid_ids, block_size_M)

    moe_gemm = moe_gemm_jit
    with pyhip.cudaPerf(num_tokens * topk * model_dim * inter_dim * 2, name=f"{moe_gemm.__name__}_up"):
        a2_v, a2_s = moe_gemm(a1, a1_scale, w1, w1_scale, 1, 0)

    #a2_bf16 = fake_quant(a2_bf16)

    with pyhip.cudaPerf(name=f"quant_act_down"):
        if a2_smooth_scale is None:
            a2, a2_scale = pt_quant_func(
                a2_v,
                scale=None,
                quant_dtype=torch.int8,
            )
            a2_scale.is_sorted = False
        elif 1:
            a2, a2_scale = quant_act(a2_v, topk, a2_v.shape[0], a2_v.shape[2], a2_smooth_scale, sorted_ids, sorted_expert_ids, num_valid_ids, False, topk_ids)
        else:
            a2, a2_scale = smoothquant_i8_per_tok(a2_v, a2_smooth_scale, topk, sorted_ids, sorted_expert_ids, num_valid_ids, block_size_M)

    # quantize output to int8 in unit of 1x256
    oquant_block_size = 0

    # only ref supports this so-far
    moe_gemm = moe_gemm_jit if oquant_block_size > 0 else moe_gemm_jit
    with pyhip.cudaPerf(num_tokens * topk * model_dim * inter_dim * 2, name=f"{moe_gemm.__name__}_down"):
        stage2_out, stage2_out_scale = moe_gemm(a2, a2_scale, w2, w2_scale, 2, oquant_block_size)

    with pyhip.cudaPerf(name=f"reduce"):
        if oquant_block_size:
            if 0:
                # dequantize & sum
                stage2_out = (stage2_out.view(-1, oquant_block_size).to(torch.bfloat16) * stage2_out_scale.to(torch.bfloat16).view(-1, 1))
                moe_out = stage2_out.view(-1, topk, model_dim).sum(dim=1)
            else:
                moe_out = reduce_i8(stage2_out.view(num_tokens, topk, model_dim), stage2_out_scale.view(num_tokens, topk, model_dim//oquant_block_size), oquant_block_size)
        else:
            moe_out = stage2_out.sum(dim=1)

    return moe_out


def fused_moe_gelu_sq_mxfp4(
        hidden_states,  # [num_tokens, model_dim] torch.bfloat16
        w1,             # [expert, inter_dim, model_dim] torch.int8
        w2,             # [expert, model_dim, inter_dim] torch.int8
        topk_weight,    # [num_tokens, topk] torch.float32
        topk_ids,       # [num_tokens, topk] torch.int32
        w1_scale, # [expert, inter_dim, 1] torch.float32
        w2_scale, # [expert, model_dim, 1] torch.float32
        a1_smooth_scale, # [expert, 1, model_dim] torch.float32
        a2_smooth_scale, # [expert, 1, inter_dim] torch.float32):
    ):
    device = hidden_states.device
    dtype = hidden_states.dtype
    def get_inter_dim(w1_shape, w2_shape):
        E, _, model_dim = w1_shape
        E, model_dim, inter_dim = w2_shape

        int4_war = model_dim // w1_shape[-1]
        inter_dim *= int4_war
        return E, model_dim, inter_dim
    M, topk = topk_ids.shape
    E, model_dim, inter_dim = get_inter_dim(w1.shape, w2.shape)
    w1_is_shuffled = getattr(w1, "is_shuffled", False)
    w2_is_shuffled = getattr(w2, "is_shuffled", False)

    assert not w1_is_shuffled
    assert not w2_is_shuffled

    assert w1.shape[1] == inter_dim

    block_size_M = 256
    block_size_N = 256

    expert_mask = None
    num_local_tokens = None
    moe_sorting_dispatch_policy = 0

    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_out = moe_sorting(
        topk_ids, topk_weight, E, model_dim,  dtype,
        block_size_M,
        expert_mask, num_local_tokens, moe_sorting_dispatch_policy,
    )

    # quantize hidden_states in sorted_ids [tok_topk]
    # print(sorted_ids.dtype, sorted_ids.shape)

    def smoothquant_mxfp4(x, x_smooth_scale, topk):
        # per_1x32_f4_quant(x, scale=None, quant_dtype=dtypes.fp4x2, shuffle=False):
        quant_1x32 = aiter.get_torch_quant(aiter.QuantType.per_1x32)
        # print(x_smooth_scale.shape, num_valid_ids)
        num_tokens = x.shape[0]
        dims = x.shape[-1]
        quant_x = torch.empty([num_tokens, topk, dims//2], dtype=torch.int8)
        quant_s = torch.zeros([num_tokens, topk, dims//32], dtype=torch.int8)
        for eblock_i, expert_id in enumerate(sorted_expert_ids):
            i0 = eblock_i * block_size_M
            i1 = i0 + block_size_M

            if i0 >= num_valid_ids[0]: break
            if expert_id >= E or expert_id < 0: break

            ids = sorted_ids[i0:i1]
            ids = ids[((ids >> 24) & 0xFF) < topk]

            cur_num_toks = len(ids)

            tok_id = ids & 0xFFFFFF
            topk_id = (ids >> 24 & 0xFF)

            if x.ndim == 3:
                tok_states = x[tok_id, topk_id, :]
            else:
                tok_states = x[tok_id, :]

            # [cur_num_toks, dims] * [expert, 1, dims]
            tok_states = tok_states * x_smooth_scale[expert_id, 0:1, :]

            # mxfp4 quantize
            qa, qs = quant_1x32(tok_states, quant_dtype=aiter.dtypes.fp4x2, shuffle=False)
            # print(qa.shape, qa.dtype) # [num_tokens, dims//2] torch.float4_e2m1fn_x2
            # print(qs.shape, qs.dtype) # [num_tokens, dims//32] torch.float8_e8m0fnu
            quant_x[tok_id, topk_id, :] = qa.view(torch.int8)
            quant_s[tok_id, topk_id, :] = qs.view(torch.int8)

        quant_s.is_sorted = False
        return quant_x.view(torch.float4_e2m1fn_x2), quant_s.view(torch.float8_e8m0fnu)


    def moe_gemm_ref(output, input, i_scale, weight, w_scale, stage_index):
        assert input.ndim == 3
        # int8 support slicing
        i_scale = fp4_utils.e8m0_to_f32(i_scale)
        w_scale = fp4_utils.e8m0_to_f32(w_scale)
        input = fp4_utils.mxfp4_to_f32(input)
        weight = fp4_utils.mxfp4_to_f32(weight)
        dims = input.shape[-1]
        for eblock_i, expert_id in enumerate(sorted_expert_ids):
            i0 = eblock_i * block_size_M
            i1 = i0 + block_size_M
            ids = sorted_ids[i0:i1]
            if i0 >= num_valid_ids[0]: break
            if expert_id >= E or expert_id < 0: break

            ids = ids[((ids >> 24) & 0xFF) < topk]
            cur_num_toks = len(ids)

            tok_id = ids & 0xFFFFFF
            topk_id = (ids >> 24 & 0xFF)

            cur_input = input[tok_id, topk_id, :]
            cur_weight = weight[expert_id,...]

            #print(cur_input.view(-1, dims//32, 32).shape, i_scale[tok_id, topk_id, :, None].shape)
            #print(cur_weight.view(-1, dims//32, 32).shape, w_scale.shape, (w_scale[expert_id, :].view(-1,dims//32,1)).shape)
            #assert 0
            cur_input = (cur_input.view(-1, dims//32, 32) * i_scale[tok_id, topk_id, :, None]).view(-1, dims)
            cur_weight = (cur_weight.view(-1, dims//32, 32) * w_scale[expert_id, :].view(-1,dims//32,1)).view(-1, dims)

            cur_out = cur_input @ cur_weight.t()

            if 0:
                print("==============")
                print(cur_out.shape)
                #print(pt_scale[i0:(i0+cur_num_toks), None].shape)
                #print(pc_scale[expert_id, None, :, 0].shape)
                assert 0

            if stage_index == 1:
                cur_out = torch.nn.functional.gelu(cur_out)
            else:
                cur_out = cur_out * sorted_weights[i0:i0+cur_num_toks, None]

            output[tok_id, topk_id, :] = cur_out.to(output.dtype)

    moe_gemm = moe_gemm_ref

    num_tokens, _ = hidden_states.shape
    with pyhip.cudaPerf(name=f"quant_act_up"):
        a1, a1_scale = smoothquant_mxfp4(hidden_states, a1_smooth_scale, topk)
        #       a1: [num_tokens, topk, dims//2] torch.float4_e2m1fn_x2
        # a1_scale: [num_tokens, topk, dims//32] torch.float8_e8m0fnu

    a2_bf16 = torch.empty((num_tokens, topk, inter_dim), dtype=torch.bfloat16, device=device,)

    with pyhip.cudaPerf(num_tokens * topk * model_dim * inter_dim * 2, name=f"{moe_gemm.__name__}[  up]"):
        moe_gemm(a2_bf16, a1, a1_scale, w1, w1_scale, 1)

    with pyhip.cudaPerf(name=f"quant_act_down"):
        a2, a2_scale = smoothquant_mxfp4(a2_bf16, a2_smooth_scale, topk)

    stage2_out = torch.empty((num_tokens, topk, model_dim), dtype=torch.bfloat16, device=device)

    with pyhip.cudaPerf(num_tokens * topk * model_dim * inter_dim * 2, name=f"{moe_gemm.__name__}[down]"):
        moe_gemm(stage2_out, a2, a2_scale, w2, w2_scale, 2)

    moe_out = stage2_out.sum(dim=1)

    return moe_out

def test_fmoe_sqi8(num_tokens, model_dim, inter_dim, num_experts, topk, use_smoothquant, shared_smoothquant_up):
    device = "cuda"

    x0 = torch.randn(num_tokens, model_dim, dtype=torch.bfloat16, device=device)

    w1_f32 = torch.randn(num_experts, inter_dim, model_dim, dtype=torch.float32)
    w2_f32 = torch.randn(num_experts, model_dim, inter_dim, dtype=torch.float32)

    if use_smoothquant:
        if shared_smoothquant_up:
            smooth_dim = 1
        else:
            smooth_dim = num_experts
        fc1_smooth_scale = 1.0/torch.randint(low=1, high=5, size=[smooth_dim, 1, model_dim], dtype=torch.float32, device=device)
        fc2_smooth_scale = 1.0/torch.randint(low=1, high=5, size=[num_experts, 1, inter_dim], dtype=torch.float32, device=device)
    else:
        fc1_smooth_scale = torch.ones([num_experts, 1, model_dim], dtype=torch.float32, device=device)
        fc2_smooth_scale = torch.ones([num_experts, 1, inter_dim], dtype=torch.float32, device=device)


    w1,fc1_scale = smooth_quant_w_i8(w1_f32, fc1_smooth_scale)
    w2,fc2_scale = smooth_quant_w_i8(w2_f32, fc2_smooth_scale)

    w1_mxfp4,fc1_scale_mxfp4 = smooth_quant_w_mxfp4(w1_f32, fc1_smooth_scale)
    w2_mxfp4,fc2_scale_mxfp4 = smooth_quant_w_mxfp4(w2_f32, fc2_smooth_scale)

    router_weights = torch.randn(num_tokens, num_experts, dtype=torch.float32)
    ret_topk = torch.topk(router_weights, topk)
    x1 = ret_topk.values.to(torch.float32)
    x2 = ret_topk.indices.to(torch.int32)
    
    #x1 = torch.randn(num_tokens, topk, dtype=torch.float32, device=device)
    # x2 = torch.load("/data02/leifanding.lfd/seedance-v2-amd/tests/inference/topk_ids.pt", weights_only=False) # torch.Size([seq_len, 20]), torch.int32
    #x2 = torch.topk(x1, low=0, high=num_experts, size=(num_tokens, topk), dtype=torch.int32) # torch.Size([seq_len, 20]), torch.int32

    # fused_moe(
    #     # hidden_states=x0.to(torch.int8),
    #     hidden_states=x0,
    #     w1=w1,
    #     w2=w2,
    #     topk_weight=x1,
    #     topk_ids=x2,
    #     expert_mask=None,
    #     activation=ActivationType.Gelu,
    #     quant_type=QuantType.per_Token,
    #     doweight_stage1=False,
    #     w1_scale=fc1_scale,
    #     w2_scale=fc2_scale,
    #     # a1_scale=a1_scale,
    #     # a2_scale=a2_scale,
    #     dtype=torch.bfloat16,
    # )

    ref0 = torch_moe(x0, w1_f32, w2_f32, x1, x2, None, None, None, None, activation=ActivationType.Gelu)
    ref1 = torch_moe(x0, w1, w2, x1, x2, fc1_scale, fc2_scale,
                     fc1_smooth_scale.expand(num_experts,-1,-1),
                     fc2_smooth_scale.expand(num_experts,-1,-1),
                     activation=ActivationType.Gelu)
    # ref2 = moe_smoothquant_gelu_ref(x0, w1, w2, x1, x2, fc1_scale, fc2_scale, fc1_smooth_scale.expand(num_experts, -1, -1), fc2_smooth_scale)

    num_flops = num_tokens * topk * model_dim * inter_dim * 2 * 2
    ret_i8, dt_i8 = pyhip.run_perftest(
            fused_moe_gelu_sqi8,
            x0,
            w1,
            w2,
            x1,
            x2,
            fc1_scale,
            fc2_scale,
            fc1_smooth_scale if use_smoothquant else None,
            fc2_smooth_scale if use_smoothquant else None,
            num_verbose = 1,
            num_flops = num_flops,
        )

    if 1:
        ret_fp4, dt_fp4 = None, 0
    else:
        ret_fp4, dt_fp4 = pyhip.run_perftest(
                fused_moe_gelu_sq_mxfp4,
                x0,
                w1_mxfp4,
                w2_mxfp4,
                x1,
                x2,
                fc1_scale_mxfp4,
                fc2_scale_mxfp4,
                fc1_smooth_scale.expand(num_experts, -1, -1),
                fc2_smooth_scale,
                num_verbose = 1,
                num_warmup = 0,
                num_iters = 1
            )
    if 1:
        ret_asm, dt_asm = None, 0
    else:
        ret_asm, dt_asm = pyhip.run_perftest(
                asm_moe,
                x0,
                w1,
                w2,
                x1,
                x2,
                fc1_scale,
                fc2_scale,
                fc1_smooth_scale.expand(num_experts, -1, -1),
                fc2_smooth_scale,
                get_gfx() != "gfx942",
                #False,
                None,
                None,
                None,
                ActivationType.Gelu,
                num_verbose = 0
            )

    print(f"{num_tokens=}, {model_dim=}, {inter_dim=}, {num_experts=}, {topk=}, {use_smoothquant=}, {shared_smoothquant_up=}")
    print(f"\t{pyhip.calc_diff(ref0, ref1)=:.6f}")
    if ret_asm is not None:
        print(f"\t{pyhip.calc_diff(ref0, ret_asm)=:.6f}  {dt_asm:.0f} us")
    print(f"\t{pyhip.calc_diff(ref0, ret_i8)=:.6f}  {dt_i8:.0f} us")
    print(f"\t{pyhip.calc_diff(ref1, ret_i8, -0.01)=:.6f}  {dt_i8:.0f} us")
    if ret_fp4 is not None:
        print(f"\t{pyhip.calc_diff(ref0, ret_fp4)=:.6f}  {dt_fp4:.0f} us")
        print(f"\t{pyhip.calc_diff(ref1, ret_fp4, -0.01)=:.6f}  {dt_fp4:.0f} us")


if __name__ == "__main__":
    pyhip.set_device()

    #test_fmoe_sqi8(num_tokens = 40960, model_dim = 4096, inter_dim = 1536, num_experts = 400, topk = 20)
    #test_fmoe_sqi8(num_tokens = 40960, model_dim = 4096, inter_dim = 1536, num_experts = 800, topk = 25)
    #test_fmoe_sqi8(num_tokens = 19147, model_dim = 4096, inter_dim = 1536, num_experts = 400, topk = 20,  use_smoothquant=1, shared_smoothquant_up=0)
    #test_fmoe_sqi8(num_tokens = 19147, model_dim = 4096, inter_dim = 1536, num_experts = 400, topk = 20,  use_smoothquant=0, shared_smoothquant_up=0)
    test_fmoe_sqi8(num_tokens = 19147, model_dim = 4096, inter_dim = 1536, num_experts = 400, topk = 20,  use_smoothquant=1, shared_smoothquant_up=1)
