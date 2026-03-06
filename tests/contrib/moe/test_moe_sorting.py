import pyhip
import torch
import aiter

from pyhip.contrib.moe_sorting import moe_sorting

def div_up(x, y):
    return (x + y - 1) // y

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



def test_sorting(num_experts, topk, num_tokens, block_size):
    """
    def moe_sorting(J,num_experts,       # number of global experts
                 num_workgroups,    # number of work-groups
                 topk,
                 block_size,
                 tmp_table:"int*",       # high 24bit : number of tokens, low 8 bit : CU updates
                 topk_ids:"int*",       # [num_tokens, topk]  int
                 topk_weights:"float*", # [num_tokens, topk]                  
                 num_tokens:"int",
                 num_local_tokens:"int*",
                 sorted_ids:"int*",       # 
                 sorted_weights:"float*",
                 sorted_expert_ids:"int*",
                 num_valid_ids:"int*"
                 ):
    """
    torch.set_default_device("cuda")
    torch.manual_seed(0)

    num_workgroups = torch.cuda.get_device_properties().multi_processor_count

    tmp_table = torch.zeros(1 + div_up(num_experts, 64)*64, dtype=torch.int32)

    router_weights = torch.randn(num_tokens, num_experts, dtype=torch.float)
    ret_topk = torch.topk(router_weights, topk)
    topk_ids = ret_topk.indices.to(torch.int32)
    topk_weights = ret_topk.values

    num_e_blocks = div_up(num_tokens * topk + num_experts * (block_size - 1), block_size)
    sorted_ids = torch.empty([num_e_blocks * block_size], dtype=torch.uint32)
    sorted_weights = torch.empty([num_e_blocks * block_size], dtype=torch.float)
    sorted_expert_ids = torch.empty([num_e_blocks], dtype=torch.uint32)
    num_valid_ids = torch.empty([2], dtype=torch.uint32)

    for k in range(16):
        with pyhip.cudaPerf(0, name="moe_sorting"):
            moe_sorting([num_workgroups], [64], 
                        num_experts,
                        num_workgroups,
                        topk,
                        block_size,
                        tmp_table.data_ptr(),
                        topk_ids.data_ptr(),
                        topk_weights.data_ptr(),
                        num_tokens,
                        None,
                        sorted_ids.data_ptr(),
                        sorted_weights.data_ptr(),
                        sorted_expert_ids.data_ptr(),
                        num_valid_ids.data_ptr(),
                        )
        if k < 9:
            # check correctness
            topk_ids2 = topk_ids.clone().cpu()
            total_ids_count = num_valid_ids[0].item()
            assert (total_ids_count % block_size) == 0
            for i in range(0, total_ids_count, block_size):
                eid = sorted_expert_ids[i//block_size].item()
                for j in range(i, i+block_size):
                    sid = sorted_ids[j].item()
                    tok_topk = sid >> 24
                    tok_id = sid & 0xFFFFFF
                    ref_eid = topk_ids2[tok_id, tok_topk].item()
                    assert topk_ids2[tok_id, tok_topk] == eid, f"{k=} {tok_id=} {tok_topk=} {ref_eid=} {eid=}"
                    assert topk_weights[tok_id, tok_topk] == sorted_weights[j]
                    topk_ids2[tok_id, tok_topk] = -1
            assert torch.all(topk_ids2 == -1)
            print("pass")

    print(topk_ids.shape, topk_ids.dtype)
    print(topk_ids)
    print(topk_weights.shape, topk_weights.dtype)
    print(topk_weights)
    
    print("============= sorted_ids\n", sorted_ids)
    print("============= sorted_weights\n", sorted_weights)
    print("============= sorted_expert_ids\n", sorted_expert_ids)
    print("============= num_valid_ids\n", num_valid_ids)

    for i in range(num_experts):
        print(f"{i} : {tmp_table[i+1].item()}")
    print(f"sync-flag : {tmp_table[0].item()}")    

test_sorting(8, 8, 512, 256)
