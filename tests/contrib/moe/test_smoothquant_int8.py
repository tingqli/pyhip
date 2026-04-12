import torch
import aiter
from aiter.fused_moe_bf16_asm import asm_moe, torch_moe
from aiter.fused_moe import fused_moe
from aiter import ActivationType
from aiter.jit.utils.chip_info import get_gfx
from aiter import QuantType

from pyhip.contrib.moe_gemm_8wave_gelu import moe_gemm_8wave_gelu

import pyhip
import os

pyhip.set_device()

device = "cuda"

model_dim = 4096
inter_dim = 1536
num_tokens = 19147
num_experts = 400
topk = 20

BLOCK_SIZE_M = 256
ROW_PER_BLOCK = 4       # rows for quant
ROW_PER_BLOCK1 = 2      # rows for quant1
ROW_PER_BLOCK2 = 4      # rows for quant2, threads: 4(x16)
BLOCK_M2 = 8            # rows for quant2, workgroup size in M
ROW_PER_BLOCK2_SEQ = 1  # rows for quant2_seq
DEBUG = True
QUANT1_K = model_dim
QUANT2_K = inter_dim
TOPK = topk

current_dir = os.path.dirname(os.path.abspath(__file__))
hip = pyhip.module(f"{current_dir}/quant-i8.cpp", f"-D{BLOCK_SIZE_M=} -D{TOPK=} -D{ROW_PER_BLOCK=} -D{ROW_PER_BLOCK2=} -D{ROW_PER_BLOCK1=} -D{BLOCK_M2=} -D{QUANT1_K=} -D{QUANT2_K=} -D{ROW_PER_BLOCK2_SEQ=}")
quant = hip.quant

def quant_act(x, topk, M, model_dim, smooth_scale, sorted_ids, sorted_expert_ids, num_valid_ids, is_gemm1=True):
    device = x.device
    DEBUG = False
    if DEBUG:
        x_quant = torch.ones((M, topk, model_dim), dtype=torch.int8, device=device)
        x_quant_scale = torch.ones([sorted_ids.shape[0]], dtype=torch.float32, device=device)
    else:
        x_quant = torch.empty((M, topk, model_dim), dtype=torch.int8, device=device)
        x_quant_scale = torch.empty([sorted_ids.shape[0]], dtype=torch.float32, device=device)
    if is_gemm1:
        quant([sorted_expert_ids.shape[0], BLOCK_SIZE_M // ROW_PER_BLOCK], [64], 
            x.data_ptr(), smooth_scale.data_ptr(), x_quant.data_ptr(), x_quant_scale.data_ptr(), 
            sorted_ids.data_ptr(), sorted_expert_ids.data_ptr(), num_valid_ids.data_ptr(),
            M, model_dim, 1
            )
    else:
        hip.quant2([sorted_expert_ids.shape[0], BLOCK_SIZE_M // BLOCK_M2], [256], 
            x.data_ptr(), smooth_scale.data_ptr(), x_quant.data_ptr(), x_quant_scale.data_ptr(), 
            sorted_ids.data_ptr(), sorted_expert_ids.data_ptr(), num_valid_ids.data_ptr(),
            M)
    return x_quant, x_quant_scale

def smooth_quant_w(weight,         # [num_experts, OC, IC]
                   smooth_scale,   # [num_experts, IC]
                   ):
    scaled_weight = weight / smooth_scale
    per_oc_scale = ((scaled_weight.abs().max(dim=2, keepdim=True)[0])/128) # [num_experts, OC, 1]
    quanted_weight = (scaled_weight / per_oc_scale).round().clamp(-128, 127).to(torch.int8)
    return quanted_weight, per_oc_scale # [num_experts, OC, IC], [num_experts, OC, 1]

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


def fused_moe_gelu_sqi8(
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
    if isinstance(w1, torch.nn.parameter.Parameter):
        w1_is_shuffled = True
    else:
        w1_is_shuffled = getattr(w1, "is_shuffled", False)

    if isinstance(w2, torch.nn.parameter.Parameter):
        w2_is_shuffled = True
    else:
        w2_is_shuffled = getattr(w2, "is_shuffled", False)
    assert w1.shape[1] == inter_dim

    moe_sorting = aiter.fused_moe.moe_sorting
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

    def smoothquant_per_tok(x, x_smooth_scale, topk):
        # print(x_smooth_scale.shape, num_valid_ids)
        num_tokens = x.shape[0]
        dims = x.shape[-1]
        quant_x = torch.empty([num_tokens, topk, dims], dtype=torch.int8)
        pt_scales = torch.zeros([len(sorted_ids)], dtype=torch.float32)
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
            pts = torch.maximum(tok_states.abs().max(dim=1, keepdim=True)[0] / 128, torch.tensor(0.0000001))
            quant_x[tok_id, topk_id, :] = (tok_states / pts).round().clamp(-128, 127).to(torch.int8)

            pt_scales[i0:i0+cur_num_toks] = pts.view(-1)

        return quant_x, pt_scales


    def moe_gemm_ref(output, input, pt_scale, weight, pc_scale, stage_index):
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

            cur_out = cur_out * pt_scale[i0:(i0+cur_num_toks), None] * pc_scale[expert_id, None, :, 0]

            if stage_index == 1:
                cur_out = torch.nn.functional.gelu(cur_out)
            else:
                cur_out = cur_out * sorted_weights[i0:i0+cur_num_toks, None]

            output[tok_id, topk_id, :] = cur_out.to(output.dtype)

    def moe_gemm_jit(output, input, pt_scale, weight, pc_scale, stage_index):
        AB_dtype = "s8"
        is_input_over_4GB = input.numel() * input.element_size() > (1<<32)
        wg_M, wg_N = 256, 256
        is_gate_up = (stage_index == 1)
        bpreshuffle = False
        num_tokens = input.shape[0]
        num_oc_blocks = output.shape[-1] // wg_N
        num_e_blocks = sorted_expert_ids.shape[0]
        moe_gemm_8wave_gelu([num_oc_blocks * num_e_blocks],[8*64], is_input_over_4GB,
                        AB_dtype, wg_M, wg_N,
                        E, output.shape[-1], input.shape[-1], 
                        is_gate_up, bpreshuffle, topk,
                        sorted_ids.data_ptr(),
                        sorted_weights.data_ptr(),
                        sorted_expert_ids.data_ptr(),
                        num_valid_ids.data_ptr(),
                        weight.data_ptr(), pc_scale.data_ptr(),
                        input.data_ptr(), pt_scale.data_ptr(),
                        output.data_ptr(),
                        num_tokens, num_oc_blocks * num_e_blocks)

    moe_gemm = moe_gemm_jit

    num_tokens, _ = hidden_states.shape
    with pyhip.cudaPerf(name=f"quant_act_up"):
        if 1:
            a1, a1_scale = quant_act(hidden_states, topk, hidden_states.shape[0], hidden_states.shape[1], a1_smooth_scale, sorted_ids, sorted_expert_ids, num_valid_ids, True)
        else:
            a1, a1_scale = smoothquant_per_tok(hidden_states, a1_smooth_scale, topk)
    a2_bf16 = torch.empty((num_tokens, topk, inter_dim), dtype=torch.bfloat16, device=device,)

    with pyhip.cudaPerf(name=f"{moe_gemm.__name__}[  up]"):
        moe_gemm(a2_bf16, a1, a1_scale, w1, w1_scale, 1)

    with pyhip.cudaPerf(name=f"quant_act_down"):
        if 1:
            a2, a2_scale = quant_act(a2_bf16, topk, a2_bf16.shape[0], a2_bf16.shape[2], a2_smooth_scale, sorted_ids, sorted_expert_ids, num_valid_ids, False)
        else:
            a2, a2_scale = smoothquant_per_tok(a2_bf16, a2_smooth_scale, topk)

    stage2_out = torch.empty((num_tokens, topk, model_dim), dtype=torch.bfloat16, device=device)

    with pyhip.cudaPerf(name=f"{moe_gemm.__name__}[down]"):
        moe_gemm(stage2_out, a2, a2_scale, w2, w2_scale, 2)

    moe_out = stage2_out.sum(dim=1)

    return moe_out


def test_fmoe_sqi8(num_tokens, model_dim, inter_dim, num_experts, topk):
    x0 = torch.randn(num_tokens, model_dim, dtype=torch.bfloat16, device=device)

    fc1_smooth_scale = 1.0/torch.randint(low=1, high=5, size=[num_experts, 1, model_dim], dtype=torch.float32, device=device)
    fc2_smooth_scale = 1.0/torch.randint(low=1, high=5, size=[num_experts, 1, inter_dim], dtype=torch.float32, device=device)

    w1_f32 = torch.randn(num_experts, inter_dim, model_dim, dtype=torch.float32).abs()
    w2_f32 = torch.randn(num_experts, model_dim, inter_dim, dtype=torch.float32).abs()

    w1,fc1_scale = smooth_quant_w(w1_f32, fc1_smooth_scale)
    w2,fc2_scale = smooth_quant_w(w2_f32, fc2_smooth_scale)

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
    ref1 = torch_moe(x0, w1, w2, x1, x2, fc1_scale, fc2_scale, fc1_smooth_scale, fc2_smooth_scale, activation=ActivationType.Gelu)
    ref2 = moe_smoothquant_gelu_ref(x0, w1, w2, x1, x2, fc1_scale, fc2_scale, fc1_smooth_scale, fc2_smooth_scale)

    ret_jit, dt_jit = None, 0
    if 1:
        ret_jit, dt_jit = pyhip.run_perftest(
                fused_moe_gelu_sqi8,
                x0,
                w1,
                w2,
                x1,
                x2,
                fc1_scale,
                fc2_scale,
                fc1_smooth_scale,
                fc2_smooth_scale,
                num_verbose = 1,
                num_iters = 4
            )

    ret_asm, dt_asm = pyhip.run_perftest(
            asm_moe,
            x0,
            w1,
            w2,
            x1,
            x2,
            fc1_scale,
            fc2_scale,
            fc1_smooth_scale,
            fc2_smooth_scale,
            get_gfx() != "gfx942",
            #False,
            None,
            None,
            None,
            ActivationType.Gelu,
            num_verbose = 1
        )

    print(f"{pyhip.calc_diff(ref0, ref1)=:.6f}")
    print(f"{pyhip.calc_diff(ref0, ref2)=:.6f}")
    print(f"{pyhip.calc_diff(ref0, ret_asm)=:.6f}")
    if ret_jit is not None:
        print(f"{pyhip.calc_diff(ref0, ret_jit)=:.6f}")
        print(f"{pyhip.calc_diff(ref2, ret_jit, 0.01)=:.6f}")


if __name__ == "__main__":
    test_fmoe_sqi8(num_tokens = num_tokens, model_dim = model_dim, inter_dim = inter_dim, num_experts = num_experts, topk = topk)