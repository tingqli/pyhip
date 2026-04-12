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
ROW_PER_BLOCK = 4       # rows for quant
ROW_PER_BLOCK1 = 1      # rows for quant1
ROW_PER_BLOCK2 = 4      # rows for quant2, threads: 4(x16)
BLOCK_M2 = 8            # rows for quant2, workgroup size in M
ROW_PER_BLOCK2_SEQ = 1  # rows for quant2_seq
TOPK = 20
DEBUG = True
num_tokens = 19149
# num_tokens = 1
model_dim, inter_dim = 4096, 1536
num_experts, topk = 400, TOPK
QUANT1_K = model_dim
QUANT2_K = inter_dim

current_dir = os.path.dirname(os.path.abspath(__file__))
hip = pyhip.module(f"{current_dir}/quant-i8.cpp", f"-D{BLOCK_SIZE_M=} -D{TOPK=} -D{ROW_PER_BLOCK=} -D{ROW_PER_BLOCK2=} -D{ROW_PER_BLOCK1=} -D{BLOCK_M2=} -D{QUANT1_K=} -D{QUANT2_K=} -D{ROW_PER_BLOCK2_SEQ=}")
quant = hip.quant

def int8_ptpc(a):
    #return (a.to(torch.float32) @ b.to(torch.float32)) * b_pc_scale.reshape(1, -1)
    pt_scale_a = ((a.abs().max(dim=1)[0]) / 128)[:, None]
    # input after gelu may become zero, avoid div zero
    pt_scale_a = torch.maximum(pt_scale_a, torch.tensor(0.00001))
    a_i8 = (a / pt_scale_a).round().clamp(-128, 127).to(torch.int8)
    return a_i8, pt_scale_a.squeeze(1)

def div_up(x, y):
    return (x + y - 1) // y

def quant_act(x, topk, M, model_dim, smooth_scale, sorted_ids, sorted_expert_ids, num_valid_ids, is_gemm1=True, topk_ids=None):
    device = x.device
    if DEBUG:
        x_quant = torch.ones((M, topk, model_dim), dtype=torch.int8, device=device)
        x_quant_scale = torch.ones([sorted_expert_ids.shape[0], BLOCK_SIZE_M], dtype=torch.float32, device=device)
    else:
        x_quant = torch.empty((M, topk, model_dim), dtype=torch.int8, device=device)
        x_quant_scale = torch.empty([sorted_expert_ids.shape[0], BLOCK_SIZE_M], dtype=torch.float32, device=device)
    if is_gemm1:
        if 0:
            quant([sorted_expert_ids.shape[0], BLOCK_SIZE_M // ROW_PER_BLOCK], [64], 
                x.data_ptr(), smooth_scale.data_ptr(), x_quant.data_ptr(), x_quant_scale.data_ptr(), 
                sorted_ids.data_ptr(), sorted_expert_ids.data_ptr(), num_valid_ids.data_ptr(),
                M, model_dim, 1
                )
        else:
            hip.quant1([div_up(M, ROW_PER_BLOCK1)], [64], 
                x.data_ptr(), smooth_scale.data_ptr(), x_quant.data_ptr(), x_quant_scale.data_ptr(), 
                topk_ids.data_ptr(),
                M
                )
    else:
        if 1:
            assert inter_dim * TOPK * num_tokens * 2 < 2**32-1, "quant2 kernel use buffer load, not greater than 4GB"
            hip.quant2([sorted_expert_ids.shape[0], BLOCK_SIZE_M // BLOCK_M2], [256], 
                x.data_ptr(), smooth_scale.data_ptr(), x_quant.data_ptr(), x_quant_scale.data_ptr(), 
                sorted_ids.data_ptr(), sorted_expert_ids.data_ptr(), num_valid_ids.data_ptr(),
                M
                )
        else:
            hip.quant2_seq([div_up(M, ROW_PER_BLOCK2_SEQ)], [64], 
                x.data_ptr(), smooth_scale.data_ptr(), x_quant.data_ptr(), x_quant_scale.data_ptr(), 
                topk_ids.data_ptr(),
                M
                )

    return x_quant, x_quant_scale

def torch_ref(x, topk, M, model_dim, smooth_scale, sorted_ids, sorted_expert_ids, num_valid_ids):
    device = x.device
    if DEBUG:
        x_quant = torch.ones((M, topk, model_dim), dtype=torch.int8, device=device)
        x_quant_scale = torch.ones([sorted_expert_ids.shape[0], BLOCK_SIZE_M], dtype=torch.float32, device=device)
    else:
        x_quant = torch.empty((M, topk, model_dim), dtype=torch.int8, device=device)
        x_quant_scale = torch.empty([sorted_expert_ids.shape[0], BLOCK_SIZE_M], dtype=torch.float32, device=device)

    def smoothquant_per_tok(x, x_smooth_scale, topk):
        # print(x_smooth_scale.shape, num_valid_ids)
        num_tokens = x.shape[0]
        dims = x.shape[-1]
        quant_x = torch.ones([num_tokens, topk, dims], dtype=torch.int8)
        pt_scales = torch.ones([sorted_expert_ids.shape[0], BLOCK_SIZE_M], dtype=torch.float32)
        for eblock_i, expert_id in enumerate(sorted_expert_ids):
            i0 = eblock_i * BLOCK_SIZE_M
            i1 = i0 + BLOCK_SIZE_M

            if i0 >= num_valid_ids[0]: break
            if expert_id >= num_experts or expert_id < 0: break

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
            tok_states = tok_states * x_smooth_scale[expert_id, :] 
            pts = torch.maximum(tok_states.abs().max(dim=1, keepdim=True)[0] / 128, torch.tensor(0.0000001))
            quant_x[tok_id, topk_id, :] = (tok_states / pts).round().clamp(-128, 127).to(torch.int8)

            pt_scales[eblock_i,:cur_num_toks] = pts.view(-1)

        return quant_x, pt_scales

    return smoothquant_per_tok(x, smooth_scale, topk)

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
fc2_smooth_scale = 1.0/torch.randint(low=1, high=5, size=[num_experts, inter_dim], dtype=torch.float32)
M = num_tokens
topk_weight = torch.randn([M, topk], dtype=torch.float32)
topk_ids = torch.ones([M, topk], dtype=torch.int32)
# make a M*TOPK seq, which contains 0...E-1 0...E-1

rep_e = div_up(M * topk, num_experts)
topk_ids_1d = torch.ones([rep_e, num_experts], dtype=torch.int32)
topk_ids_1d[:, ] = torch.randperm(num_experts, dtype=torch.int32)
topk_ids[:, ] = topk_ids_1d.reshape(-1)[ : M * topk].reshape(M, topk)

sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = moe_sorting_ck(topk_ids, topk_weight, num_experts, model_dim, torch.float32, BLOCK_SIZE_M, None)
cur_act, cur_scale = quant_act(x, topk, M, model_dim, fc1_smooth_scale, sorted_ids, sorted_expert_ids, num_valid_ids, True, topk_ids)
ref_act, ref_scale = torch_ref(x, topk, M, model_dim, fc1_smooth_scale, sorted_ids, sorted_expert_ids, num_valid_ids)
# torch.testing.assert_close(cur_scale, ref_scale)
torch.testing.assert_close(cur_act, ref_act, atol=1.1, rtol=2.1)
print('done.')

mem_size = x.numel() * x.element_size() + fc1_smooth_scale.numel() * fc1_smooth_scale.element_size() + sorted_ids.numel() * sorted_ids.element_size() + sorted_expert_ids.numel() * sorted_expert_ids.element_size() + num_valid_ids.numel() * num_valid_ids.element_size() + cur_act.numel() * cur_act.element_size() + cur_scale.numel() * cur_scale.element_size()
print(f'gemm1 quantization, mem size {mem_size/1000/1000:.2f} MB')
for _ in range(10):
    with pyhip.cudaPerf(0, mem_size, name=f"quant1") as p:
        quant_act(x, topk, M, model_dim, fc1_smooth_scale, sorted_ids, sorted_expert_ids, num_valid_ids, True, topk_ids)

x1 = torch.randn(num_tokens, topk, inter_dim, dtype=torch.bfloat16)

cur_act, cur_scale = quant_act(x1, topk, M, inter_dim, fc2_smooth_scale, sorted_ids, sorted_expert_ids, num_valid_ids, False, topk_ids)
ref_act, ref_scale = torch_ref(x1, topk, M, inter_dim, fc2_smooth_scale, sorted_ids, sorted_expert_ids, num_valid_ids)
# torch.testing.assert_close(cur_scale, ref_scale)
torch.testing.assert_close(cur_act, ref_act, atol=1.1, rtol=200)

mem_size = x1.numel() * x1.element_size() + fc2_smooth_scale.numel() * fc2_smooth_scale.element_size() + sorted_ids.numel() * sorted_ids.element_size() + sorted_expert_ids.numel() * sorted_expert_ids.element_size() + num_valid_ids.numel() * num_valid_ids.element_size() + cur_act.numel() * cur_act.element_size() + cur_scale.numel() * cur_scale.element_size()
print(f"gemm2 input size {mem_size/1000/1000:.2f} MB")
for _ in range(10):
    with pyhip.cudaPerf(0, mem_size, name=f"quant2") as p:
        quant_act(x1, topk, M, inter_dim, fc2_smooth_scale, sorted_ids, sorted_expert_ids, num_valid_ids, False, topk_ids)