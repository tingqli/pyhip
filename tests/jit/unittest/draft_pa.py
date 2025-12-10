import pyhip
import torch
import math
torch.cuda.set_device(6)
torch.set_default_device('cuda')
torch.manual_seed(0)
torch.set_printoptions(linewidth=3000, sci_mode=False, edgeitems=8, )

from functools import cache

"""
仿照公共表达式提取优化
    这些被cache的函数被首次调用时（cache-miss）会生成代码，
    后继任何位置需要使用此处代码生成的vgpr时都只会直接引用
    生成的目标vgpr,并且最终生成的代码不会一直占用物理VGPR，
    而是在最后一次引用之后释放

    因此需要注意，这些函数只能在循环外层调用一次，并且如果
    参数中包含其他vgpr或者sgpr，则如果这些参数的值改变了的话
    cache无法察觉，因此会复用之前的计算结果,为了避免这个缺陷
    尽量使用纯值参数

    每当我们发现某个表达式在一个展开的循环中似乎会被重复计算
    就可以使用下面的模式，既能够保证在第一次需要的时候生成代码
    计算结果，也能保证第二次在后面展开的代码中使用该结果时避免
    重复计算。
"""
@cache
def get_lane_id(J):
    vgpr_lane_id = J.gpr(J.threadIdx.x[0] & 63)
    return vgpr_lane_id

@cache
def get_lane_id_div(J, divisor):
    assert isinstance(divisor, int)
    return J.gpr(get_lane_id(J) // divisor)

@cache
def get_lane_id_mod(J, divisor):
    assert isinstance(divisor, int)
    return J.gpr(get_lane_id(J) % divisor)

@cache
def get_lane_voffset(J, lane_rows_cols, lane_bytes, stride_bytes):
    lane_rows, lane_cols = lane_rows_cols
    assert isinstance(lane_rows, int)
    assert isinstance(lane_cols, int)
    assert isinstance(lane_bytes, int)
    assert isinstance(stride_bytes, int)
    if lane_cols == 1:
        voffset = J.gpr(get_lane_id_mod(J, lane_rows)*stride_bytes + get_lane_id_div(J, lane_rows)*(lane_bytes))
    else:
        voffset = J.gpr(get_lane_id_div(J, lane_cols)*stride_bytes + get_lane_id_mod(J, lane_cols)*(lane_bytes))
    return voffset

# compile-time constants
def get_jit_kernel(HQ, # query heads
                   HK, # key heads
                   S,  # head size
                   BLOCK_SIZE,   # block size
                   KV_PART_SIZE, # kv-lengths each warp handles
                   acc_scale,
                   num_parts,    # how many steps each warp goes
                   debug_loc,    # debug
                   ):
    @pyhip.jit("-g")
    def pa_jit(J,
                query:"__bf16*",      #[B, HQ, S]
                key_cache:"__bf16*",    # [BLOCK, BLOCK_SIZE, HK, S]
                value_cache:"__bf16*",  # 
                kv_indptr:"uint*",      # [B + 1]
                kv_page_indices:"uint*", # [B * KV_LEN + 1]
                out_seg:"__bf16*",      # [B, HQ, max_num_parts, S]
                max_out:"float*",       # [B, HQ, max_num_parts, 1]
                sum_out:"float*",       # [B, HQ, max_num_parts, 1]
                # max_num_parts (max number of workgroups for one-item/one-head-group in batch)
                max_num_parts:"uint",
                log_ptr:"uint*",
                ):

        '''
        偏移之后，找到本WG需要处理的数据：
            
        '''
        b = J.blockIdx.x
        hk = J.blockIdx.y
        kv_part = J.blockIdx.z
        sizeof_bf16 = 2

        # 这个读取latency非常大，需要提前发起hidding,
        kv_cum_len = J.gpr("su32x2")
        J.s_load_dwordx2(kv_cum_len, kv_indptr, b[0]<<2)

        assert (HQ % HK) == 0
        hq = J.gpr(hk * (HQ // HK))

        # query  [B, HQ, S] => [1, (HQ//HK), S]
        query[:] = query + (b*(HQ*S*2) + hq*(S*2))
        # key_cache/value_cache : [BLOCK, BLOCK_SIZE, HK, S] => [BLOCK, BLOCK_SIZE, 1(HK), S]
        key_cache[:] = key_cache[:] + hk[0]*(S*2)
        value_cache[:] = value_cache[:] + hk[0]*(S*2)
        # out_seg : [B, HQ, PART, S]          => [1, (HQ//HK), max_num_parts, S]
        # max_out : [B, HQ, max_num_parts, 1] => [1, (HQ//HK), max_num_parts, 1]
        # sum_out : [B, HQ, max_num_parts, 1] => [1, (HQ//HK), max_num_parts, 1]
        output_offet = J.gpr(b * max_num_parts * (HQ) + (hq*max_num_parts + kv_part))
        out_seg[:] = out_seg[:] + output_offet*(S*2)
        max_out[:] = max_out[:] + output_offet*(4)
        sum_out[:] = sum_out[:] + output_offet*(4)

        lane_id = get_lane_id(J)
        lane_mod_16 = get_lane_id_mod(J, 16)
        lane_div_16 = get_lane_id_div(J, 16)
        warp_id = J.gpr("su32")
        J.v_readfirstlane_b32(warp_id, J.threadIdx.x[0] // 64)

        # 先顺序执行把正确性调对
        # 如何验证正确性？调试阶段可以使用debug_log，优化阶段需要更加自动化的检查正确性机制，就是需要把运算结果写回内存，由host侧代码检查正确性
        # 但是直接传回acc的话，数据量过大，可以采用仅仅传回某个特定位置的acc的方法，这个特定位置由外部参数指定，外部多次运行传入不同位置参数
        # 就能拿出不同位置计算结果并检查正确性，并且这个也用debug_log完成，无需额外申请内存
        if debug_loc is not None:
            J.debug_setup(log_ptr, (b[0] == debug_loc["b"])&(hk[0] == debug_loc["hk"])&(kv_part[0]==debug_loc["kv_part"])&(warp_id[0]==debug_loc["warp_id"]))
        #J.debug_setup(log_ptr, (b[0] == (save_acc_loc&0xFF))&(hk[0] == (save_acc_loc>>8)&0xFF)&(kv_part[0]==(save_acc_loc>>16)&0xFF)&(warp_id[0]==(save_acc_loc>>16)&0xFF))

        # Python 没有类似C++那种使用{}限制变量作用域，提高代码可读性
        # 的方法，类似效果的就是使用closure
        def issue_load_query():
            q_buf = J.Buffer(query, (HQ//HK)*S*2)
            assert (HQ//HK) <= 16, f"use mfma16 requires M <= 16"
            q_cur = J.gpr(f"au32x{4*S//32}") # bfloat16x8
            for i in range(4*S//32):
                J.v_accvgpr_write_b32(q_cur[i], 0)

            #voffset = J.gpr((S*sizeof_bf16)*lane_mod_16[0] + lane_div_16[0]*(4*4))
            voffset = get_lane_voffset(J, (16, 1), 16, S*sizeof_bf16)
            with J.ExecMask(lane_mod_16 < (HQ//HK)):
                # vdst, voffset, soffset, offset12=0
                for k in range(S//32):
                    q_buf.load_dwordx4(q_cur[4*k+0:4*k+3], voffset, 0, offset12=k*32*sizeof_bf16)
            return q_cur

        # 为了利用dowrdx4高效加载，每个lane里面有8个bf16
        # 16 x 32 的query子矩阵被加载到交织的两个16x16里面
        # 总大小是 [(HQ//HK), (S)] bf16

        # [B, HK, div_up(KV_LEN, KV_PART_SIZE)], [256]

        q_cur = issue_load_query()

        J.s_waitcnt(mod=f"lgkmcnt({0})")

        # kv-length for current batch
        kv_len = J.gpr(kv_cum_len[1] - kv_cum_len[0])
        # early return
        J.Jump("continue_following", kv_part * KV_PART_SIZE < kv_len)
        J.s_endpgm()
        J.Label("continue_following")

        kv_len_start = J.gpr("su32")
        kv_len_end  = J.gpr("su32")

        J.s_min_u32(kv_len_start, kv_part * KV_PART_SIZE, kv_len)
        J.s_min_u32(kv_len_end,   kv_len_start + KV_PART_SIZE, kv_len)
        kv_page_indices[:] = kv_page_indices[:] + (kv_cum_len[0] + kv_len_start)*4

        kv_part_len = J.gpr(kv_len_end[0] - kv_len_start[0])
        kv_page_ids_buff = J.Buffer(kv_page_indices, kv_part_len * 4)

        # kv_page_indices 是连续摆放的，可以连续读取
        # 全部预取到LDS比较占用空间，每轮预取下一轮需要的即可，4个wave各取各的
        assert BLOCK_SIZE == 1
        # 加载kv_page_indices的初始延迟比较大，等待之前一次性多读入全部要用到的可以提高效率
        # kv_page_ids_buff会对越界的索引返回0，因此cur_kv_ids中越界的部分被0填充不会产生越界
        kv_ids = J.gpr(f'vu32x{num_parts}')
        for part_idx in range(num_parts):
            offset12 = part_idx*(KV_PART_SIZE//num_parts)*4
            assert offset12.bit_length() <= 12            
            kv_page_ids_buff.load_dword(kv_ids[part_idx], J.threadIdx.x[0]*4, 0, offset12=offset12)

        # 每次kv-len维度上步进256，自己完成其中的64个计算，步进4次，4个warps一共完成1024个计算
        # 每个warp根据warp_id，每次取 lds_kv_ids 中属于自己的那64个state
        #   (warp_id*64) + part_idx*256
        key_sum = J.gpr("vf32")
        key_sum[0] = 0

        """
        LDS有限，因此要以尽量小的单位使用
        寄存器比较大，可以用来做cache, key_reg/value_reg 
        """
        # 外存读入每次DWORDx4对应 4 x S(128) 的数据, 64xS一共需要读取64//4次
        key_reg = J.gpr(f"au32x{4*(64//4)}")
        value_reg = J.gpr(f"au32x{4*(64//4)}")

        # 初始化64位基地址
        vKbase64bits = J.gpr(f"vu32x2")
        vVbase64bits = J.gpr(f"vu32x2")
        J.v_mov_b64(vKbase64bits, key_cache)
        J.v_mov_b64(vVbase64bits, value_cache)
        J.v_add_co_u32(vKbase64bits[0], "vcc", lane_mod_16[0]*16, vKbase64bits[0])
        J.v_addc_co_u32(vKbase64bits[1], "vcc", 0, vKbase64bits[1], "vcc")
        J.v_add_co_u32(vVbase64bits[0], "vcc", lane_mod_16[0]*16, vVbase64bits[0])
        J.v_addc_co_u32(vVbase64bits[1], "vcc", 0, vVbase64bits[1], "vcc")

        # 64x128 bf16
        lds_key = J.alloc_lds(64 * S * sizeof_bf16*4)

        mtime_start = J.gpr("su32x2")
        #J.s_memtime(mtime_start)

        # 预取第一part的key/value
        J.s_waitcnt(mod=f"vmcnt(0)")

        # 形成 offsets
        kv_offsets = J.gpr("vu32x16")
        cur_bperm = J.gpr(lane_div_16[0]*4)
        #for i in range(16):
        #    J.ds_bpermute_b32(kv_offsets[i], cur_bperm, cur_kv_ids, mod=f"offset:{i*4*4}")
        #J.s_waitcnt(mod=f"lgkmcnt({0})")

        # 预取第一轮需要的数据
        kv_off = J.gpr("vu32x2")
        hks_bits = J.shift_bits(HK*S*2)
        vKaddr64bits = J.gpr(f"vu32x2")
        vVaddr64bits = J.gpr(f"vu32x2")
        cur_bperm = J.gpr(lane_div_16[0]*4)

        for i in range(16):
            J.ds_bpermute_b32(kv_offsets[i], cur_bperm, kv_ids[0], mod=f"offset:{i*4*4}")
            J.s_waitcnt(mod=f"lgkmcnt({0})")
            kv_off[0], kv_off[1] = kv_offsets[i], 0
            J.v_lshlrev_b64(kv_off, hks_bits, kv_off) # 从这个移位开始就可能产生超过4GB 32位的值
            vKaddr64bits[:] = vKbase64bits[:] + kv_off[:]
            J.global_load_dwordx4(key_reg[4*i+0:4*i+3], vKaddr64bits, "off")

        vm_cnt_preload_value = 0
        for i in range(16):
            kv_off[0], kv_off[1] = kv_offsets[i], 0
            J.v_lshlrev_b64(kv_off, hks_bits, kv_off) # 从这个移位开始就可能产生超过4GB 32位的值
            vVaddr64bits[:] = vVbase64bits[:] + kv_off[:]
            #J.debug_log(kv_offsets[i], torch.int32)
            # kv_offsets[i]使用完毕，加载part1需要的索引
            J.ds_bpermute_b32(kv_offsets[i], cur_bperm, kv_ids[1], mod=f"offset:{i*4*4}")
            # 这个load很大概率会引起issue stall,因此可以在它前面加入可以遮盖的计算
            J.global_load_dwordx4(value_reg[4*i+0:4*i+3], vVaddr64bits, "off")
            vm_cnt_preload_value += 1
        J.s_waitcnt(mod=f"lgkmcnt({0})")

        out = J.gpr(f"vf32x{16*S//64}")
        for i in range(out.count):
            out[i] = 0
        prev_max = J.gpr("vf32")
        cur_sum = J.gpr("vf32")
        prev_max[0] = torch.finfo(torch.float).min
        cur_sum[0] = 0
        for part_idx in range(num_parts):
            vm_cnt_preload_key = 0 # 统计从这个点开始，又发起了多少个vmem指令
            vaddr_warp_base = J.gpr("vu32")
            vaddr_warp_base[0] = warp_id * (16*S*sizeof_bf16)
            acc = J.gpr(f"vf32x{4*4}", align=4)
            for n in range(4):
                J.s_waitcnt(mod=f"vmcnt({vm_cnt_preload_value+16-(n+1)*4})")
                vKaddr64bits = J.gpr(f"vu32x2")
                # 可以发起一次Q*K计算的最小数据单位：16行的key 已经就位，写入LDS
                # 这个写入比较快并且issue stall的代价小，因此没有跟任何计算交织
                for i in range(4):
                    i0 = 4*i
                    row = J.gpr(lane_div_16[0] + i0)
                    col = lane_mod_16[0]
                    vaddr = J.gpr(row*(S*sizeof_bf16) + (col^(row))*16 + vaddr_warp_base)
                    J.ds_write_b128(vaddr, key_reg[16*n+i0:16*n+i0+3], mod=f"offset:{lds_key}")
                    # J.debug_log(key_reg[16*n+i0:16*n+i0+3], torch.bfloat16, "4v.16h.8h")

                temp_key = [J.gpr("au32x4"),J.gpr("au32x4")]
                acc_out = acc[4*n:4*n+3]

                row = lane_mod_16[0]
                col = lane_div_16[0]
                vaddr = J.gpr(row*(S*sizeof_bf16) + (col^(row))*16 + vaddr_warp_base)

                # 因为写入和读取layout不同，必须等待写入完全结束才能发起读取
                J.s_waitcnt(mod=f"lgkmcnt({0})")
                # 因为每次ds读取可以产生2次MFMA计算，因此pipeline这两步可以并行
                J.ds_read_b128(temp_key[0], vaddr, mod=f"offset:{lds_key}")
                next_round_load_key = 0
                for k in range(S//32):
                    if (k + 1) < (S//32):
                        row = lane_mod_16[0]
                        col = J.gpr(lane_div_16[0] + (k+1)*(32//8))
                        vaddr = J.gpr(row*(S*sizeof_bf16) + (col^(row))*16 + vaddr_warp_base)
                        J.ds_read_b128(temp_key[(1+k)&1], vaddr, mod=f"offset:{lds_key}")
                        J.s_waitcnt(mod=f"lgkmcnt({1})")
                    else:
                        J.s_waitcnt(mod=f"lgkmcnt({0})")

                    #J.debug_log(temp_key[k&1], torch.bfloat16, "4h.16v.8h")
                    acc_in = 0 if k==0 else acc_out
                    # 7.5. Dependency Resolution: Required Independent Instructions
                    J.v_mfma_f32_16x16x16_bf16(acc_out, temp_key[k&1][0:1], q_cur[4*k+0:4*k+1], acc_in)
                    J.s_nop(15)
                    J.v_mfma_f32_16x16x16_bf16(acc_out, temp_key[k&1][2:3], q_cur[4*k+2:4*k+3], acc_out)
                    # 发起下一轮的vmem预取，有很大概率会引起issue stall
                    # 因此我们跟上面耗时的LDS+MFMA指令交织起来，降低指令密度，降低stall概率
                    if (part_idx + 1) < num_parts:
                        if next_round_load_key < 4:
                            i0 = 4*next_round_load_key
                            kv_off[0], kv_off[1] = kv_offsets[n*4 + next_round_load_key], 0
                            J.v_lshlrev_b64(kv_off, hks_bits, kv_off) # 从这个移位开始就可能产生超过4GB 32位的值
                            vKaddr64bits[:] = vKbase64bits[:] + kv_off[:]
                            J.global_load_dwordx4(key_reg[16*n+i0:16*n+i0+3], vKaddr64bits, "off")
                            vm_cnt_preload_key += 1
                            next_round_load_key += 1
                if (part_idx + 1) < num_parts:
                    assert next_round_load_key == 4

            # online-softmax 计算开始，此时可以交织完成value数据的准备以避免issue stall
            if debug_loc is not None:
                J.debug_log(acc, torch.float, "4h.4h.16v.4h")
                #J.debug_log(q_cur, torch.bfloat16, "4h.4h.16v.8h")
                #J.s_waitcnt(mod=f"vmcnt(0) lgkmcnt(0)");J.s_endpgm()

            for n in range(4):
                acc_out = acc[4*n:4*n+3]
                for i in range(4):
                    acc_out[i] = acc_out[i] * acc_scale

            """
            prev_max = torch.full([qlen,1], torch.finfo(torch.float).min, dtype=torch.float)
            cur_sum = torch.full([qlen,1], 0, dtype=torch.float)
            O = torch.full([qlen, 128], 0, dtype=torch.float)
            for min_part in range(num_parts):
                S = acc[:,min_part*64:(min_part+1)*64] # 8x64
                V = matV[min_part*64:(min_part+1)*64,:].float()
                rowmax = S.max(dim=1, keepdim = True).values
                cur_max = torch.maximum(rowmax, prev_max)
                P = (S - cur_max).exp()
                max_fixup = (prev_max-cur_max).exp()
                cur_sum = max_fixup * cur_sum + P.sum(dim=1, keepdim = True)
                prev_max = cur_max
                O = max_fixup * O + (P @ V)
                fa2_outs.append(O)
            fa2_out = (1.0/cur_sum)*O
            """

            ############# rowmax = S.max(dim=1, keepdim = True).values ##########
            ############# cur_max = torch.maximum(rowmax, prev_max)
            # rowmax, 避免超出kv-len的计算引入错误的max，把这部分acc值先置为 fmin
            # acc 的layout是 4h.4h.16v.4h, 每个lane内部对应元素的k_pos要根据这个layout
            # 4个float被组织在一个lane里面，但是每个float的k_pos不同，因此下面的k_pos
            # 只是lane中第一个元素的k_pos
            k_pos = J.gpr("vu32")
            k_pos[0] = part_idx * (KV_PART_SIZE//num_parts) + warp_id * (KV_PART_SIZE//num_parts//4) + (lane_div_16[0] * 4)
            fmin = J.gpr("vf32")
            fmin[0] = torch.finfo(torch.float).min
            for n in range(4):
                acc_out = acc[4*n:4*n+3]
                for i in range(4):
                    J.SetMask("vcc", k_pos + i < kv_part_len)
                    J.v_cndmask_b32_e32(acc_out[i], fmin, acc_out[i], "vcc")
                k_pos[0] = k_pos[0] + 16

            # cur_max: cur_max = torch.maximum(rowmax, prev_max)
            cur_max = J.gpr("vf32")
            cur_max[0] = prev_max[0]
            for n in range(4):
                acc_out = acc[4*n:4*n+3]
                for i in range(4):
                    J.v_max_f32(cur_max, cur_max, acc_out[i])

             # cur_max: cross-lane
            vtemp = J.gpr("vf32")
            for mask in [32, 16]:
                J.ds_bpermute_b32(vtemp, (lane_id ^ mask)*4, cur_max)
                J.s_waitcnt(mod=f"lgkmcnt(0)")
                J.v_max_f32(cur_max, cur_max, vtemp)

            # acc现在保存P P = (S - cur_max).exp()
            import math
            alpha = math.log2(math.e)
            for n in range(4):
                acc_out = acc[4*n:4*n+3]
                for i in range(4):
                    J.v_exp_f32(acc_out[i], (acc_out[i] - cur_max) * alpha)
            
            # max_fixup = (prev_max-cur_max).exp()
            max_fixup = J.gpr(prev_max - cur_max)
            J.v_exp_f32(max_fixup, max_fixup * alpha)

            # cur_sum = max_fixup * cur_sum + P.sum(dim=1, keepdim = True)
            psum = J.gpr("vf32")
            psum[0] = 0
            for n in range(4):
                acc_out = acc[4*n:4*n+3]
                for i in range(4):
                    psum[0] = psum[0] + acc_out[i]

            vtemp = J.gpr("vf32")
            for mask in [32, 16]:
                J.ds_bpermute_b32(vtemp, (lane_id ^ mask)*4, psum)
                J.s_waitcnt(mod=f"lgkmcnt(0)")
                psum[0] = psum + vtemp

            cur_sum[0] = cur_sum * max_fixup + psum

            # prev_max = cur_max
            prev_max[0] = cur_max

            # O = max_fixup * O + (P @ V)
            if debug_loc is not None:
                J.debug_log(cur_sum, torch.float, "4h.16v.1h")
                J.debug_log(max_fixup, torch.float, "4h.16v.1h")


            # out : max_fixup是per-row的 
            # O = max_fixup*O + (P @ V)
            for i in range(out.count):
                out[i] = out[i] * max_fixup

            # 需要用到value了，等待前一轮value就位
            '''
            for n in range(4):
                J.s_waitcnt(mod=f"vmcnt({vm_cnt_preload_key + 16 - (n+1)*4})")
                # 16行的 value 已经就位，写入LDS
                for i in range(4):
                    i0 = 4*i
                    row = J.gpr(lane_div_16[0] + i0)
                    col = lane_mod_16[0]
                    vaddr = J.gpr(row*(S*sizeof_bf16) + (col^(row))*16 + vaddr_warp_base)
                    J.ds_write_b128(vaddr, value_reg[16*n+i0:16*n+i0+3], mod=f"offset:{lds_key}")
            '''

            vm_cnt_preload_value = 0
            if (part_idx + 1) < num_parts:
                vVaddr64bits = J.gpr(f"vu32x2")
                for i in range(16):
                    kv_off[0], kv_off[1] = kv_offsets[i], 0
                    J.v_lshlrev_b64(kv_off, hks_bits, kv_off) # 从这个移位开始就可能产生超过4GB 32位的值
                    vVaddr64bits[:] = vVbase64bits[:] + kv_off[:]
                    # 计算下轮预取需要用到的offset
                    if (part_idx + 2) < num_parts:
                        J.ds_bpermute_b32(kv_offsets[i], cur_bperm, kv_ids[part_idx + 2], mod=f"offset:{i*4*4}")
                    J.global_load_dwordx4(value_reg[4*i+0:4*i+3], vVaddr64bits, "off")
                    vm_cnt_preload_value += 1
            if (part_idx + 2) < num_parts:
                J.s_waitcnt(mod=f"lgkmcnt({0})")

        #a=J.alloc_lds((8-4)*1024)
        J.s_waitcnt(mod=f"vmcnt({0})")

        mtime_stop = J.gpr("su32x2")
        J.s_memtime(mtime_stop)

        J.s_waitcnt(mod=f"lgkmcnt({0})")

        J.s_sub_u32(mtime_stop[0], mtime_stop[0], mtime_start[0])
        J.s_subb_u32(mtime_stop[1], mtime_stop[1], mtime_start[1])

        if debug_loc is not None:
            vdata = J.gpr("vu32")
            vdata[0] = mtime_stop[0]
            #J.global_atomic_umax(J.threadIdx.x[0] << 2, vdata, debug_out)
            J.debug_log(mtime_stop[0], torch.uint32)
            J.debug_log(mtime_stop[1], torch.uint32)

        # scalar-memoryreads can return out-of-order:
        # following hack prevent kernel-arg loading to reuse same sgprs for different args
        return

    return pa_jit



B = 1
HQ = 32
HK = 4
S = 128

# KV_LEN 如何切分为CU个数的整数倍,切分不均匀是很大的原因，
#
# B*HK=4, KV_LEN需要是20的整数倍才能保证均匀分给80个CU 45694/20=2284.7
# 因此每个block需要完成2284~2285个token，但是每个batch的kv-len数目不同
# 每个block可以遍历 kv_indptr 来确定自己需要负责的kv-len区间，因为 kv_indptr
# 相对较小，遍历很快，
KV_LEN = 40*1024
#KV_LEN = 45694

#KV_LEN = 512
DT = torch.bfloat16
BLOCK_SIZE = 1
BLOCK_NUM = B * KV_LEN + 1000
FAKE_Q = 0
FAKE_K_IDX = 0
OUTPUT_QK = 0
BUF_COPY = 1
BUF_COPY = 32
USE_REDUCE_JIT = True

KV_MIN_PART_SIZE = 256
KV_PART_SIZE = 256 * 4
KV_PART_SIZE_WARP = (KV_MIN_PART_SIZE // 4)

query = torch.randint(-2, 3, [B, HQ, S], dtype=DT)
key_caches = []
value_caches = []
kv_indptrs = []
kv_page_indices_ = []
kv_last_page_lens_ = []
# [N, BLOCK_SIZE, HK, S]
for _ in range(BUF_COPY):
    key_cache = torch.randint(-2, 3, [BLOCK_NUM, BLOCK_SIZE, HK, S], dtype=DT)
    value_cache = torch.randint(-2, 3, [BLOCK_NUM, BLOCK_SIZE, HK, S], dtype=DT)
    batch_start = [0] * (B + 1)
    for b in range(B):
        batch_start[b + 1] = (b + 1) * KV_LEN
    kv_indptr = torch.tensor(batch_start, dtype=torch.int32)
    kv_page_indices = torch.linspace(1, KV_LEN * B, KV_LEN * B, dtype=torch.int32)
    kv_last_page_lens = torch.ones([KV_LEN], dtype=torch.int32)
    key_caches.append(key_cache)
    value_caches.append(value_cache)
    kv_indptrs.append(kv_indptr)
    kv_page_indices_.append(kv_page_indices)
    kv_last_page_lens_.append(kv_last_page_lens)
scale = 1 / (S**0.5)

num_parts = KV_PART_SIZE//KV_MIN_PART_SIZE


# pa hip
def div_up(x, y):
    return (x + y - 1) // y

max_num_parts = div_up(KV_LEN, KV_PART_SIZE)
# -g -ggdb -O1
my_out_seg = torch.ones([B, HQ, max_num_parts, S], dtype=DT) * 3
my_out = torch.empty([B, HQ, S], dtype=DT)
my_max = torch.empty([B, HQ, max_num_parts, 1], dtype=torch.float32) * 5
my_sum = torch.empty([B, HQ, max_num_parts, 1], dtype=torch.float32) * 6
qk_out = torch.ones([B, HK, HQ // HK, max_num_parts * KV_PART_SIZE], dtype=torch.float32)
i = 0

print(query[0,:8,:])
print(key_cache[:16,0,0,:])
accuracy = {}

# 参考flashattention2的python实现
def FA2_ref(debug_loc):
    hk = debug_loc["hk"]
    qlen = HQ//HK
    hq = hk*qlen
    kv_part = debug_loc["kv_part"]
    warp_id = debug_loc["warp_id"]
    matQ = query[debug_loc["b"], hq:hq+qlen,:]
    # 只取 debug_warp_id 负责的部分数据
    kv_pos = []
    for min_part in range(num_parts):
        for i in range(64):
            kv_pos.append(kv_part*KV_PART_SIZE + min_part*KV_MIN_PART_SIZE + warp_id*64 + i)
    kv_idxs = kv_page_indices_[0][kv_pos]
    matK = key_caches[0][kv_idxs, 0, hk, :]
    matV = value_caches[0][kv_idxs, 0, hk, :]
    # 一次性计算参考答案
    def normal_attn():
        acc = torch.nn.functional.linear(matQ.float(), matK.float())
        acc = acc*scale
        rowmax = acc.max(dim=1, keepdim = True).values
        acc_exp = (acc - rowmax).exp()
        rowsum = acc_exp.sum(dim=1, keepdim = True)
        acc_softmax = acc_exp/rowsum
        output = acc_softmax @ matV.float()
        return output
    
    fa2_logs = {"Q":matQ, "K":matK, "kv_idxs":kv_idxs}
    fa2_logs["O"] = []
    fa2_logs["S"] = []
    fa2_logs["cur_max"] = []
    fa2_logs["max_fixup"] = []
    fa2_logs["cur_sum"] = []
    def FA2_online_softmax():
        # 按照16x64的间隔，做4次online softmax
        prev_max = torch.full([qlen,1], torch.finfo(torch.float).min, dtype=torch.float)
        cur_sum = torch.full([qlen,1], 0, dtype=torch.float)
        O = torch.full([qlen, 128], 0, dtype=torch.float)
        Q = matQ.float()
        for min_part in range(num_parts):
            K = matK[min_part*64:(min_part+1)*64,:].float()
            V = matV[min_part*64:(min_part+1)*64,:].float()
            S = Q @ K.t()
            fa2_logs["S"].append(S)
            S = S*scale
            rowmax = S.max(dim=1, keepdim = True).values
            cur_max = torch.maximum(rowmax, prev_max)
            fa2_logs["cur_max"].append(cur_max)
            P = (S - cur_max).exp()
            max_fixup = (prev_max-cur_max).exp()
            fa2_logs["max_fixup"].append(max_fixup)
            cur_sum = max_fixup * cur_sum + P.sum(dim=1, keepdim = True)
            fa2_logs["cur_sum"].append(cur_sum)
            prev_max = cur_max
            O = max_fixup * O
            fa2_logs["O"].append(O)
            O += (P.bfloat16().float() @ V)
        fa2_out = (1.0/cur_sum)*O
        return fa2_out

    ref_out = normal_attn()
    fa2_out = FA2_online_softmax()
    if not torch.allclose(ref_out, fa2_out, atol=0.01, rtol=0.01):
        print(ref_out)
        print(fa2_out)
        assert 0
    #print(S.shape, rowmax.shape, rowsum.shape)
    return fa2_logs


if 1:
    print("======================= verify correctness ==============================")
    debug_loc = {
        "b": 0,
        "hk": 2,
        "kv_part": 12,
        "warp_id": 3
    }
    pa_jit = get_jit_kernel(HQ, HK, S, BLOCK_SIZE, KV_PART_SIZE, scale, num_parts, debug_loc)
    debug_log_ptr = pa_jit.log_ptr()
    pa_jit([B, HK, max_num_parts], [256],
            query.data_ptr(),           # [B, HQ, S]
            key_caches[0].data_ptr(),   # [BLOCK_NUM, BLOCK_SIZE, HK, S]
            value_caches[0].data_ptr(), # [BLOCK_NUM, BLOCK_SIZE, HK, S]
            kv_indptrs[0].data_ptr(),
            kv_page_indices_[0].data_ptr(),
            my_out_seg.data_ptr(),
            my_max.data_ptr(), 
            my_sum.data_ptr(),
            max_num_parts,
            debug_log_ptr)
    logs = pa_jit.get_logs()
    fa2_logs = FA2_ref(debug_loc)
    print(fa2_logs["kv_idxs"])
    print("=============Q")
    print(fa2_logs["Q"])

    if "q_cur" in logs:
        print("=============q_cur")
        print(logs["q_cur"][0])

    print("=============K", fa2_logs["K"].shape)
    print(fa2_logs["K"])
    print("=============temp_key")
    if "temp_key" in logs:
        for k in logs["temp_key"]:
            print(":", k.shape)
            print(k)
    if "acc" in logs:
        print(logs["acc"][0].shape, fa2_logs["S"][0].shape)
        accuracy["acc"] = "pass"
        for part in range(num_parts):
            res = logs["acc"][part][:(HQ//HK),:].cuda()
            ref = fa2_logs["S"][part]
            if not torch.allclose(res, ref):
                accuracy["acc"] = f"failed at {part}/{num_parts}"
                print("=============ref", ref.device)
                print(ref)
                print("=============res", res.device)
                print(res)
                break
    
    if "out" in logs:
        accuracy["out"] = "pass"
        for part in range(num_parts):
            res = logs["out"][part][:(HQ//HK),:].cuda()
            ref = fa2_logs["O"][part]
            print("out", res.shape, ref.shape)
            if not torch.allclose(res, ref):
                accuracy["out"] = f"failed at {part}/{num_parts}"
                print("=============ref", ref.device)
                print(ref)
                print("=============res", res.device)
                print(res)
                break
    
    if "max_fixup" in logs:
        accuracy["max_fixup"] = "pass"
        for part in range(num_parts):
            res = logs["max_fixup"][part].cuda()[:(HQ//HK),0:1]
            ref = fa2_logs["max_fixup"][part]
            print("max_fixup", res.shape, ref.shape)
            if not torch.allclose(res, ref):
                accuracy["max_fixup"] = f"failed at {part}/{num_parts}"
                print("=============ref", ref.device)
                print(ref)
                print("=============res", res.device)
                print(logs["max_fixup"][part].cuda())
                break

    if "cur_max" in logs:
        accuracy["cur_max"] = "pass"
        for part in range(num_parts):
            res = logs["cur_max"][part].cuda()[:(HQ//HK),0:1]
            ref = fa2_logs["cur_max"][part]
            print("cur_max", res.shape, ref.shape)
            if not torch.allclose(res, ref):
                accuracy["cur_max"] = f"failed at {part}/{num_parts}"
                print("=============ref", ref.device)
                print(ref)
                print("=============res", res.device)
                print(logs["cur_max"][part].cuda())
                break
    if "cur_sum" in logs:
        accuracy["cur_sum"] = "pass"
        for part in range(num_parts):
            res = logs["cur_sum"][part].cuda()[:(HQ//HK),0:1]
            ref = fa2_logs["cur_sum"][part]
            print("cur_sum", res.shape, ref.shape)
            if not torch.allclose(res, ref):
                accuracy["cur_sum"] = f"failed at {part}/{num_parts}"
                print("=============ref", ref.device)
                print(ref)
                print("=============res", res.device)
                print(logs["cur_sum"][part].cuda())
                break        
    print(accuracy)
    #assert 0

print("======================= test performance ==============================")
pa_jit = get_jit_kernel(HQ, HK, S, BLOCK_SIZE, KV_PART_SIZE, scale, num_parts, None)
debug_log_ptr = pa_jit.log_ptr()
for round in range(10):
    with pyhip.cudaPerf(B * HQ // HK * KV_LEN * S * 2 * 2, B * (HK * KV_LEN * S * 2 * 2), name="pa_jit"):
        pa_jit([B, HK, max_num_parts], [256],
               query.data_ptr(), 
               key_caches[i].data_ptr(), 
               value_caches[i].data_ptr(), 
               kv_indptrs[i].data_ptr(), 
               kv_page_indices_[i].data_ptr(), 
               my_out_seg.data_ptr(),
               my_max.data_ptr(), 
               my_sum.data_ptr(),
               max_num_parts,
               debug_log_ptr)
    i = (i + 1) % BUF_COPY
print(f"[{B}, {HK}, {div_up(KV_LEN, KV_PART_SIZE)}]")
print(f"{accuracy=}")
