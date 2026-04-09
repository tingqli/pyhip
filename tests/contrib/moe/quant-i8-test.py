import torch
from aiter.fused_moe_bf16_asm import asm_moe, torch_moe
from aiter.fused_moe import fused_moe
from aiter import ActivationType
from aiter.jit.utils.chip_info import get_gfx
from aiter import QuantType
import aiter

import pyhip
import os

torch.set_default_device('cuda')
torch.manual_seed(0)

BLOCK_SIZE_M = 256
ROW_PER_BLOCK = 4
TOPK = 20
NUM_THREADS = 64
DEBUG = True
num_tokens = 19149
# num_tokens = 1
model_dim, inter_dim = 4096, 1536
num_experts, topk = 400, TOPK

current_dir = os.path.dirname(os.path.abspath(__file__))
hip = pyhip.module(f"{current_dir}/quant-i8.cpp", f"-D{BLOCK_SIZE_M=} -D{TOPK=} -D{ROW_PER_BLOCK=} -D{NUM_THREADS=}")
quant = hip.quant

def int8_ptpc(a):
    #return (a.to(torch.float32) @ b.to(torch.float32)) * b_pc_scale.reshape(1, -1)
    pt_scale_a = ((a.abs().max(dim=1)[0]) / 128)[:, None]
    # input after gelu may become zero, avoid div zero
    pt_scale_a = torch.maximum(pt_scale_a, torch.tensor(0.00001))
    a_i8 = (a / pt_scale_a).round().clamp(-128, 127).to(torch.int8)
    return a_i8, pt_scale_a.squeeze(1)

def quant_act(x, topk, M, model_dim, smooth_scale, sorted_ids, sorted_expert_ids, num_valid_ids, is_gemm1=True):
    device = x.device
    if DEBUG:
        x_quant = torch.ones((M, topk, model_dim), dtype=torch.int8, device=device)
        x_quant_scale = torch.ones([sorted_expert_ids.shape[0], BLOCK_SIZE_M], dtype=torch.float32, device=device)
    else:
        x_quant = torch.empty((M, topk, model_dim), dtype=torch.int8, device=device)
        x_quant_scale = torch.empty([sorted_expert_ids.shape[0], BLOCK_SIZE_M], dtype=torch.float32, device=device)
    quant([sorted_expert_ids.shape[0], BLOCK_SIZE_M // ROW_PER_BLOCK], [64], 
           x.data_ptr(), smooth_scale.data_ptr(), x_quant.data_ptr(), x_quant_scale.data_ptr(), 
           sorted_ids.data_ptr(), sorted_expert_ids.data_ptr(), num_valid_ids.data_ptr(),
           M, model_dim, 1 if is_gemm1 else 0
           )
    return x_quant, x_quant_scale

def torch_ref(x, topk, M, model_dim, smooth_scale, sorted_ids, sorted_expert_ids, num_valid_ids):
    device = x.device
    if DEBUG:
        x_quant = torch.ones((M, topk, model_dim), dtype=torch.int8, device=device)
        x_quant_scale = torch.ones([sorted_expert_ids.shape[0], BLOCK_SIZE_M], dtype=torch.float32, device=device)
    else:
        x_quant = torch.empty((M, topk, model_dim), dtype=torch.int8, device=device)
        x_quant_scale = torch.empty([sorted_expert_ids.shape[0], BLOCK_SIZE_M], dtype=torch.float32, device=x.device)
    for i in range(sorted_expert_ids.shape[0]):
        if i * BLOCK_SIZE_M >= num_valid_ids[0].item():
            break
        raw_sorted_id = sorted_ids[i * BLOCK_SIZE_M:(i + 1) * BLOCK_SIZE_M]
        sorted_id = raw_sorted_id & 0xFFFFFF
        topk_id = raw_sorted_id >> 24
        sorted_id = sorted_id[sorted_id < M]
        topk_id = topk_id[topk_id < topk]
        x_s = x[sorted_id] * smooth_scale[sorted_expert_ids[i]]
        x_q, x_qs = int8_ptpc(x_s)
        x_quant[sorted_id, topk_id] = x_q
        x_quant_scale[i, :x_qs.shape[0]] = x_qs

    return x_quant, x_quant_scale


#############################################################
def moe_sorting_ck(
    topk_ids,
    topk_weights,
    num_experts,
    model_dim,
    moebuf_dtype,
    block_size=BLOCK_SIZE_M,
    expert_mask=None,
):
    device = topk_ids.device
    M, topk = topk_ids.shape
    topk = topk_ids.shape[1]
    max_num_tokens_padded = topk_ids.numel() + num_experts * block_size - topk
    max_num_m_blocks = int((max_num_tokens_padded + block_size - 1) // block_size)
    sorted_ids = torch.empty((max_num_tokens_padded,), dtype=torch.int32, device=device)
    sorted_weights = torch.empty(
        (max_num_tokens_padded,), dtype=torch.float32, device=device
    )
    sorted_expert_ids = torch.empty(
        (max_num_m_blocks,), dtype=torch.int32, device=device
    )
    num_valid_ids = torch.empty((2), dtype=torch.int32, device=device)
    moe_buf = torch.empty((M, model_dim), dtype=moebuf_dtype, device=device)

    aiter.moe_sorting_fwd(
        topk_ids,
        topk_weights,
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        moe_buf,
        num_experts,
        block_size,
        expert_mask,
    )
    return sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf


x = torch.randn(num_tokens, model_dim, dtype=torch.bfloat16)
fc1_smooth_scale = 1.0/torch.randint(low=1, high=5, size=[num_experts, model_dim], dtype=torch.float32)
fc1_smooth_scale[1,:] = 0
M = num_tokens
topk_weight = torch.randn([M, topk], dtype=torch.float32)
topk_ids = torch.ones([M, topk], dtype=torch.int32)
# make a M*TOPK seq, which contains 0...E-1 0...E-1
def div_up(x, y):
    return (x + y - 1) // y

rep_e = div_up(M * topk, num_experts)
topk_ids_1d = torch.ones([rep_e, num_experts], dtype=torch.int32)
topk_ids_1d[:, ] = torch.randperm(num_experts, dtype=torch.int32)
topk_ids[:, ] = topk_ids_1d.reshape(-1)[ : M * topk].reshape(M, topk)

sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = moe_sorting_ck(topk_ids, topk_weight, num_experts, model_dim, torch.float32, BLOCK_SIZE_M, None)
cur_act, cur_scale = quant_act(x, topk, M, model_dim, fc1_smooth_scale, sorted_ids, sorted_expert_ids, num_valid_ids)
ref_act, ref_scale = torch_ref(x, topk, M, model_dim, fc1_smooth_scale, sorted_ids, sorted_expert_ids, num_valid_ids)
torch.testing.assert_close(cur_scale, ref_scale)
torch.testing.assert_close(cur_act, ref_act, atol=1.1, rtol=200)
print('done.')
