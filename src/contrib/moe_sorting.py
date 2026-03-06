from pyhip import jit

__all__ = [
    "moe_sorting",
]

@jit(with_debug_log=False)
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
    topk_ids [num_tokens, topk] 均匀分配一下到各个CU, CU把分配到的部分进行局部
    分组（在LDS里面），分组完毕之后，用 atomic 争抢任务来决定每个expert的最终摆入位置
    这些位置记录到外存之后，更新一个标记，每个CU会读取这些位置以决定自己已经分组好的数据该写入何处
        tmp [num_experts, 2] 

        global_atomic_add(offset, tmp[0], length_of_expert)
    
        tmp[0]的低8 bit 用来记录已经更新过的CU的个数，如果该个数已经到达总CU数，说明大家都已经更新过，此时tmp[0]的高位
        就是属于该expert的总的token个数，否则就要等到该数字到达cu总数
    
    均分 num_tokens
    通过 global atomic_add 为每个表项 分配局部offset，该局部offset连同 token_id(tid+topk)/weight 记录到LDS中
    直到分配到的token全部处理完毕。

    再次使用 global atomic_add 获取tmp_table里面全部专家，看个数是否都已经被所有CU更新好了，是的话，就增加总offset

    """
    with J.If(num_local_tokens[0] != 0):
        J.s_load_dword(num_tokens, num_local_tokens, 0)
        J.s_waitcnt(mod=f"lgkmcnt(0)")

    # partition num_tokens evenly among all work-groups
    num_tokens_wg = J.gpr("su32", num_tokens // num_workgroups)
    num_extra_tokens = J.gpr("su32", num_tokens % num_workgroups)

    wg_id = J.blockIdx.x

    tok0 = J.gpr("su32")
    tok1 = J.gpr("su32")
    with J.If(wg_id[0] < num_extra_tokens[0]) as If:
        tok0[0] = wg_id[0] * (1 + num_tokens_wg) # need to do 1 more 
        tok1[0] = tok0 + (1 + num_tokens_wg)

        If.Else()
        tok_base = num_extra_tokens * (1 + num_tokens_wg)
        tok0[0] = tok_base + (wg_id - num_extra_tokens) * num_tokens_wg
        tok1[0] = tok0 + num_tokens_wg

    J.s_min_u32(tok1, tok1[0], num_tokens[0])

    # load all token-topk & weights into LDS, token-by-token
    MAX_TOKENS_WG = 8*1024
    lds_topk_ids = J.alloc_lds(MAX_TOKENS_WG * J.sizeof_u32)
    lds_topk_weights = J.alloc_lds(MAX_TOKENS_WG * J.sizeof_f32)
    lds_offsets = J.alloc_lds(MAX_TOKENS_WG * J.sizeof_u32)
    lds_total_cnt = J.alloc_lds(J.div_up(num_experts, 64*4) * 64*4 * J.sizeof_u32)

    vaddr = J.gpr("vu32", J.threadIdx.x[0] * J.sizeof_u32)

    with J.ExecMask(J.threadIdx.x[0] < topk):
        J.s_mov_b32("m0", lds_topk_ids)
        k = J.gpr("su32", tok0[0])
        with J.While(k[0] < tok1[0]):
            J.global_load_lds_dword(vaddr + k[0] * (topk*J.sizeof_u32), topk_ids)
            J.s_addk_i32("m0", topk*J.sizeof_u32)
            k[0] += 1

        J.s_mov_b32("m0", lds_topk_weights)
        k = J.gpr("su32", tok0[0])
        with J.While(k[0] < tok1[0]):
            J.global_load_lds_dword(vaddr + k[0] * (topk*J.sizeof_f32), topk_weights)
            J.s_addk_i32("m0", topk*J.sizeof_f32)
            k[0] += 1

    J.s_waitcnt(mod=f"vmcnt(0)")
    
    # allocate offsets globally using global_atomic_add + tmp_table
    vaddr_ds_tok = J.gpr("vu32", J.threadIdx.x[0] * J.sizeof_u32 + lds_topk_ids)
    vaddr_ds_off = J.gpr("vu32", J.threadIdx.x[0] * J.sizeof_u32 + lds_offsets)
    vexpert_ids = J.gpr("vu32")
    voffset = J.gpr("vu32")
    vone = J.gpr("vu32", 1)
    vzero = J.gpr("vu32", 0)
    k = J.gpr("su32", tok0[0])
    with J.ExecMask(J.threadIdx.x[0] < topk):
        with J.While(k[0] < tok1[0]):
            J.ds_read_b32(vexpert_ids, vaddr_ds_tok)
            J.s_waitcnt(mod=f"lgkmcnt(0)")

            # vdata : expert index
            J.global_atomic_add(voffset, vexpert_ids * J.sizeof_u32, vone, tmp_table, mod=f"offset:4 sc0")
            J.s_waitcnt(mod=f"vmcnt(0)")

            # vinfo has the allocated offset for each experts
            J.ds_write_b32(vaddr_ds_off, voffset)

            vaddr_ds_off[0] += topk*J.sizeof_u32
            vaddr_ds_tok[0] += topk*J.sizeof_u32
            k[0] += 1

    J.s_waitcnt(mod=f"lgkmcnt(0)")

    # tmp_table[0] as a global sync flag
    vaddr_vmem = J.gpr("vu32", 0)
    with J.ExecMask(J.threadIdx.x[0] == 0):
        # increase sync flag
        J.global_atomic_add(vaddr_vmem, vone, tmp_table)

        # wait all other WG to reach here
        s_sync_flag = J.gpr("su32", 1)

        with J.While((s_sync_flag[0] % num_workgroups) != 0):
            J.global_atomic_add(voffset, vaddr_vmem, vzero, tmp_table, mod="sc0")
            J.s_waitcnt(mod=f"vmcnt(0)")
            J.v_readfirstlane_b32(s_sync_flag, voffset)
            J.s_nop(8)

    # scan each token-topk, one-token by another, stores (topk<<24 | token_id) 
    # if it hits current expert
    assert topk <= 64

    vdata_tok = J.gpr("vu32")
    vdata_wei = J.gpr("vu32")
    vdata_off = J.gpr("vu32")
    cur_num_tokens = J.gpr("su32", 0)
    s_e_idx = J.gpr("vu32", 0)

    # load lds_total_cnt back
    vaddr = J.gpr("vu32", J.threadIdx.x[0] * J.sizeof_DW4)
    J.s_mov_b32("m0", lds_total_cnt)
    for expert_id in range(0, 1 + num_experts, 64*4):
        J.global_load_lds_dwordx4(vaddr, tmp_table)
        vaddr[0] += 64*J.sizeof_DW4
        J.s_addk_i32("m0", 64*J.sizeof_DW4)
    J.s_waitcnt(mod=f"vmcnt(0)")

    for expert_id in range(num_experts):
        vaddr_vmem[0] = expert_id * J.sizeof_u32

        vaddr_ds_tok = J.gpr("vu32", J.threadIdx.x[0] * J.sizeof_u32 + lds_topk_ids)
        vaddr_ds_wei = J.gpr("vu32", J.threadIdx.x[0] * J.sizeof_u32 + lds_topk_weights)
        vaddr_ds_off = J.gpr("vu32", J.threadIdx.x[0] * J.sizeof_u32 + lds_offsets)
        with J.ExecMask(J.threadIdx.x[0] < topk):
            k = J.gpr("su32", tok0[0])
            with J.While(k[0] < tok1[0]):
                J.ds_read_b32(vdata_tok, vaddr_ds_tok)
                J.ds_read_b32(vdata_wei, vaddr_ds_wei)
                J.ds_read_b32(vdata_off, vaddr_ds_off)
                J.s_waitcnt(mod=f"lgkmcnt(0)")
                # only 1 tok topk at-most may pass below condition
                with J.ExecMask(vdata_tok[0] == expert_id):
                    topk_id = J.threadIdx.x[0]
                    J.global_store_dword((vdata_off[0] + cur_num_tokens)*J.sizeof_u32,  (topk_id << 24) | k[0], sorted_ids)
                    J.global_store_dword((vdata_off[0] + cur_num_tokens)*J.sizeof_u32,  vdata_wei, sorted_weights)
                k[0] += 1
                vaddr_ds_tok[0] += topk * J.sizeof_u32
                vaddr_ds_wei[0] += topk * J.sizeof_u32
                vaddr_ds_off[0] += topk * J.sizeof_u32

        vaddr_ds_cnt = J.gpr("vu32", expert_id * J.sizeof_u32 + lds_total_cnt + J.sizeof_u32)
        v_cnt = J.gpr("vu32")
        s_cnt = J.gpr("su32")
        with J.ExecMask(J.threadIdx.x[0] == 0):
            J.ds_read_b32(v_cnt, vaddr_ds_cnt)#  lds_total_cnt)
            J.s_waitcnt(mod=f"lgkmcnt(0)")
            # J.global_atomic_add(v_cnt, vaddr_vmem, vzero, tmp_table, mod="offset:4 sc0")
            #J.s_waitcnt(mod=f"vmcnt(0)")
            J.v_readfirstlane_b32(s_cnt, v_cnt)
            J.s_nop(8)

        # align s_cnt up-to block_size
        s_cnt[0] = ((s_cnt + block_size - 1) // block_size) * block_size

        # update sorted_expert_ids (only 1 wg is enough)
        vdata_expert_id = J.gpr("vu32", expert_id)
        with J.If(wg_id[0] == 0):
            with J.ExecMask(J.threadIdx.x[0] == 0):
                k[0] = 0
                with J.While(k[0] < s_cnt[0]):
                    J.global_store_dword(s_e_idx[0], vdata_expert_id, sorted_expert_ids)
                    s_e_idx[0] += J.sizeof_u32
                    k[0] += block_size

        cur_num_tokens[0] += s_cnt[0]

    #J.s_endpgm()

    # num_valid_ids
    vdata_num_tokens = J.gpr("vu32", cur_num_tokens[0])
    vaddr_vmem[0] = J.threadIdx.x[0] * J.sizeof_u32
    with J.If(wg_id[0] == 0):
        with J.ExecMask(J.threadIdx.x[0] == 0):
            J.global_store_dword(vaddr_vmem[0], vdata_num_tokens, num_valid_ids)
            # clear sync flag for next run
            J.global_store_dword(vaddr_vmem[0], vzero, tmp_table)

        # clear all expert local count
        for expert_id in range(0, num_experts, 64):
            J.global_store_dword(vaddr_vmem[0], vzero, tmp_table, mod="offset:4")
            vaddr_vmem[0] += 64*J.sizeof_u32
