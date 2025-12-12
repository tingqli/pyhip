import pyhip
import torch
import math

from pyhip.asmjit import Addr2D
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
    GQA = HQ // HK
    def reduce(J:pyhip.JIT,
               warp_id,
               p_out_seg,
               p_max_out,
               p_sum_out,
               lds_base,
               cur_sum,
               cur_max,
               vout,
               ):
        lane_id = get_lane_id(J)
        lane_mod_16 = get_lane_id_mod(J, 16)
        lane_div_16 = get_lane_id_div(J, 16)

        J.s_barrier()
        with J.ExecMask(lane_div_16[0] == 0):
            # ds_write2st64_b32 v34, v109, v94 offset1:1
            addr = Addr2D(J, lds_base, warp_id, lane_mod_16 * 4, 16 * 4)
            J.ds_write2_b32(addr.get_addr(), cur_max, cur_sum, mod=f'offset1:{4 * 16}')
        J.s_barrier()

        maxs = J.gpr(GQA, 'vf32')
        sums = J.gpr(GQA, 'vf32')
        gqa4 = div_up(GQA, 4)
        real_max = J.gpr(gqa4, 'vf32')
        real_sum = J.gpr(gqa4, 'vf32')
        for i in range(gqa4):
            m_token_id = gqa4 * warp_id + i
            # TODO: precompute offset
            addr = Addr2D(J, lds_base, 0, m_token_id * 4, 16 * 4)
            J.ds_read2_b32(maxs[4 * i + 0 : 4 * i + 1], addr.get_addr(), mod=f'offset1:{16}')
            J.ds_read2_b32(maxs[4 * i + 2 : 4 * i + 3], addr.get_addr(), mod=f'offset0:{2 * 16} offset1:{3 * 16}')
            J.ds_read2_b32(sums[4 * i + 0 : 4 * i + 1], addr.get_addr(), mod=f'offset0:{4 * 16} offset1:{5 * 16}')
            J.ds_read2_b32(sums[4 * i + 2 : 4 * i + 3], addr.get_addr(), mod=f'offset0:{6 * 16} offset1:{7 * 16}')
            J.s_waitcnt(mod='lgkmcnt(0)')
            J.v_max_f32_e32(real_max[i], maxs[4 * i + 0], maxs[4 * i + 1])
            J.v_max3_f32(real_max[i], real_max[i], maxs[4 * i + 2], maxs[4 * i + 3])
            tmp = J.gpr(4, 'vf32')
            alpha = math.log2(math.exp(1))
            J.v_exp_f32_e32(tmp[0], (maxs[4 * i + 0] - real_max[i]) * alpha)
            J.v_exp_f32_e32(tmp[1], (maxs[4 * i + 1] - real_max[i]) * alpha)
            J.v_exp_f32_e32(tmp[2], (maxs[4 * i + 2] - real_max[i]) * alpha)
            J.v_exp_f32_e32(tmp[3], (maxs[4 * i + 3] - real_max[i]) * alpha)
            J.v_pk_mul_f32(tmp[0:1], tmp[0:1], sums[4 * i + 0 : 4 * i + 1])
            J.v_pk_mul_f32(tmp[2:3], tmp[2:3], sums[4 * i + 2 : 4 * i + 3])
            J.v_add_f32_e32(tmp[0], tmp[0], tmp[1])
            J.v_add_f32_e32(tmp[2], tmp[2], tmp[3])
            J.v_add_f32_e32(real_sum[i], tmp[0], tmp[2])

        vout_low = J.gpr(S // 64 * 4 * 2, 'vf32')
        for k in range(S // 64):
            vout_low_4 = vout_low[k * 8 : k * 8 + 7]
            for i in range(4):
                # NOTE: the following will be error:
                # vout_4 = vout[k * 4 : k * 4 + 3]
                # vout_low_4[2 * i + 1] = (vout_4[2][i] >> 16) | (vout_4[3][i] & 0xffff0000)
                vout_low_4[2 * i + 0] = (vout[k * 4 + 0, i] >> 16) | (vout[k * 4 + 1, i] & 0xffff0000)
                vout_low_4[2 * i + 1] = (vout[k * 4 + 2, i] >> 16) | (vout[k * 4 + 3, i] & 0xffff0000)

        J.s_barrier()

        addr = Addr2D(J, lds_base, warp_id, S * 2 * lane_mod_16, 16 * S * 2)
        lane_div_16_4 = lane_div_16 * 4
        for k in range(S // 64):
            J.ds_write_b64(addr.get_addr() + ((lane_div_16_4    ) ^ lane_mod_16) * 8, vout_low[8 * k + 0 : 8 * k + 1], mod=f'offset:{k * 64 * 2}')
            J.ds_write_b64(addr.get_addr() + ((lane_div_16_4 + 1) ^ lane_mod_16) * 8, vout_low[8 * k + 2 : 8 * k + 3], mod=f'offset:{k * 64 * 2}')
            J.ds_write_b64(addr.get_addr() + ((lane_div_16_4 + 2) ^ lane_mod_16) * 8, vout_low[8 * k + 4 : 8 * k + 5], mod=f'offset:{k * 64 * 2}')
            J.ds_write_b64(addr.get_addr() + ((lane_div_16_4 + 3) ^ lane_mod_16) * 8, vout_low[8 * k + 6 : 8 * k + 7], mod=f'offset:{k * 64 * 2}')
        J.s_barrier()

        offset_out = max_num_parts * (S * 2)
        offset_max_sum = max_num_parts * 4
        for i in range(gqa4):
            m_token_id = gqa4 * warp_id + i
            with J.ExecMask(m_token_id < GQA):
                addr = Addr2D(J, lds_base, m_token_id, ((m_token_id ^ (lane_id >> 1)) * 2 + (lane_id & 1)) * 4, S * 2)
                tmp = J.gpr(4, 'vf32')
                J.ds_read_b32(tmp[0], addr.get_addr(), mod=f'offset:{16 * S * 2 * 0}')
                J.ds_read_b32(tmp[1], addr.get_addr(), mod=f'offset:{16 * S * 2 * 1}')
                J.ds_read_b32(tmp[2], addr.get_addr(), mod=f'offset:{16 * S * 2 * 2}')
                J.ds_read_b32(tmp[3], addr.get_addr(), mod=f'offset:{16 * S * 2 * 3}')
                J.s_waitcnt(mod='lgkmcnt(0)')
                out_v = J.gpr(2, 'vf32')
                out_v[0] = tmp[0] << 16
                out_v[1] = tmp[0] & 0xffff0000

                exp_d_max = J.gpr("vf32")
                J.v_exp_f32_e32(exp_d_max, (maxs[4 * i + 0] - real_max[i]) * alpha)
                for j in range(2):
                    out_v[j] = exp_d_max * out_v[j]
                for k in range(1, 4):
                    tmp[0] = tmp[k] << 16
                    tmp[1] = tmp[k] & 0xffff0000
                    exp_d_max = J.gpr("vf32")
                    J.v_exp_f32_e32(exp_d_max, (maxs[4 * i + k] - real_max[i]) * alpha)
                    for j in range(2):
                        tmp[j] = exp_d_max * tmp[j]
                        out_v[j] += tmp[j]
                inv_sum_scale = J.gpr('vf32')
                J.v_rcp_f32(inv_sum_scale, real_sum[i] + 1e-6)
                J.v_mul_f32(out_v[0], out_v[0], inv_sum_scale)
                J.v_mul_f32(out_v[1], out_v[1], inv_sum_scale)
                bf16x2 = J.gpr((out_v[1] & 0xffff0000) | (out_v[0] >> 16))

                J.global_store_dword(lane_id << 2, bf16x2, p_out_seg)
                tmp = J.gpr("su32")
                J.v_readfirstlane_b32(tmp, real_max[i])
                J.s_store_dword(tmp, p_max_out, 0, mod="glc")
                J.v_readfirstlane_b32(tmp, real_sum[i])
                J.s_store_dword(tmp, p_sum_out, 0, mod="glc")
                p_out_seg[0] += offset_out
                p_max_out[0] += offset_max_sum
                p_sum_out[0] += offset_max_sum
                # J.s_waitcnt(mod=f"vmcnt(0)")

    @pyhip.jit("-g")
    def pa_jit(J:pyhip.JIT,
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
        kv_cum_len = J.gpr(2, "su32")
        J.s_load_dwordx2(kv_cum_len, kv_indptr, b[0]<<2)

        assert (HQ % HK) == 0
        hq = J.gpr(hk * (HQ // HK))

        # query  [B, HQ, S] => [1, (HQ//HK), S]
        query[:] = query + (b*(HQ*S*2) + hq*(S*2))
        # key_cache/value_cache : [BLOCK, BLOCK_SIZE, HK, S] => [BLOCK, BLOCK_SIZE, 1(HK), S]
        key_cache[:] = key_cache[:] + hk[0]*(S*2)
        value_cache[:] = value_cache[:] + hk[0]*(S*2)
        warp_id = J.gpr("su32")
        J.v_readfirstlane_b32(warp_id, J.threadIdx.x[0] // 64)
        # out_seg : [B, HQ, PART, S]          => [1, (HQ//HK), max_num_parts, S]
        # max_out : [B, HQ, max_num_parts, 1] => [1, (HQ//HK), max_num_parts, 1]
        # sum_out : [B, HQ, max_num_parts, 1] => [1, (HQ//HK), max_num_parts, 1]
        gqa4 = div_up(GQA, 4)
        output_offet = J.gpr(b * max_num_parts * (HQ) + ((hq+gqa4*warp_id)*max_num_parts + kv_part))
        output_offet4 = output_offet*(4)
        out_seg[:] = out_seg[:] + output_offet*(S*2)
        max_out[:] = max_out[:] + output_offet4
        sum_out[:] = sum_out[:] + output_offet4

        lane_id = get_lane_id(J)
        lane_mod_16 = get_lane_id_mod(J, 16)
        lane_div_16 = get_lane_id_div(J, 16)

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
            q_cur = J.gpr(4*S//32, f"au32") # bfloat16x8
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
        kv_ids = J.gpr(num_parts, 'vu32')
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
        key_reg = J.gpr(64//4, 4, "au32")
        value_reg = J.gpr(64//4, 4, "au32")

        # 初始化64位基地址
        vKbase64bits = J.gpr(2, "vu32")
        vVbase64bits = J.gpr(2, "vu32")
        J.v_mov_b64(vKbase64bits, key_cache)
        J.v_mov_b64(vVbase64bits, value_cache)
        J.v_add_co_u32(vKbase64bits[0], "vcc", lane_mod_16[0]*16, vKbase64bits[0])
        J.v_addc_co_u32(vKbase64bits[1], "vcc", 0, vKbase64bits[1], "vcc")
        J.v_add_co_u32(vVbase64bits[0], "vcc", lane_mod_16[0]*16, vVbase64bits[0])
        J.v_addc_co_u32(vVbase64bits[1], "vcc", 0, vVbase64bits[1], "vcc")

        # 64x128 bf16
        lds_key = J.alloc_lds(64 * S * sizeof_bf16)

        mtime_start = J.gpr(2, "su32")
        #J.s_memtime(mtime_start)

        # 预取第一part的key/value
        J.s_waitcnt(mod=f"vmcnt(0)")

        # 形成 offsets
        kv_offsets = J.gpr(16, "vu32")
        cur_bperm = J.gpr(lane_div_16[0]*4)
        #for i in range(16):
        #    J.ds_bpermute_b32(kv_offsets[i], cur_bperm, cur_kv_ids, mod=f"offset:{i*4*4}")
        #J.s_waitcnt(mod=f"lgkmcnt({0})")

        # 预取第一轮需要的数据
        kv_off = J.gpr(2, "vu32")
        hks_bits = J.shift_bits(HK*S*2)
        vKaddr64bits = J.gpr(2, "vu32")
        vVaddr64bits = J.gpr(2, "vu32")
        cur_bperm = J.gpr(lane_div_16[0]*4)

        for i in range(16):
            J.ds_bpermute_b32(kv_offsets[i], cur_bperm, kv_ids[0], mod=f"offset:{i*4*4}")
            J.s_waitcnt(mod=f"lgkmcnt({0})")
            kv_off[0], kv_off[1] = kv_offsets[i], 0
            J.v_lshlrev_b64(kv_off, hks_bits, kv_off) # 从这个移位开始就可能产生超过4GB 32位的值
            vKaddr64bits[:] = vKbase64bits[:] + kv_off[:]
            J.global_load_dwordx4(key_reg[i], vKaddr64bits, "off")

        vm_cnt_preload_value = 0
        for i in range(16):
            kv_off[0], kv_off[1] = kv_offsets[i], 0
            J.v_lshlrev_b64(kv_off, hks_bits, kv_off) # 从这个移位开始就可能产生超过4GB 32位的值
            vVaddr64bits[:] = vVbase64bits[:] + kv_off[:]
            #J.debug_log(kv_offsets[i], torch.int32)
            # kv_offsets[i]使用完毕，加载part1需要的索引
            J.ds_bpermute_b32(kv_offsets[i], cur_bperm, kv_ids[1], mod=f"offset:{i*4*4}")
            # 这个load很大概率会引起issue stall,因此可以在它前面加入可以遮盖的计算
            J.global_load_dwordx4(value_reg[i], vVaddr64bits, "off")
            vm_cnt_preload_value += 1
        J.s_waitcnt(mod=f"lgkmcnt({0})")

        vout = J.gpr(S//16, 4, "vf32")  #  some 16x16 f32 
        for i in range(S//16):
            vout[i,0] = 0
            vout[i,1] = 0
            vout[i,2] = 0
            vout[i,3] = 0
        prev_max = J.gpr("vf32")
        cur_sum = J.gpr("vf32")
        prev_max[0] = torch.finfo(torch.float).min
        cur_sum[0] = 0
        for part_idx in range(num_parts):
            vm_cnt_preload_key = 0 # 统计从这个点开始，又发起了多少个vmem指令
            vaddr_warp_base = J.gpr("vu32")
            vaddr_warp_base[0] = warp_id * (16*S*sizeof_bf16)
            acc = J.gpr(4, 4, "vf32", align=4)
            for n in range(4):
                J.s_waitcnt(mod=f"vmcnt({vm_cnt_preload_value+16-(n+1)*4})")
                vKaddr64bits = J.gpr(2, "vu32")
                # 可以发起一次Q*K计算的最小数据单位：16行的key 已经就位，写入LDS
                # 这个写入比较快并且issue stall的代价小，因此没有跟任何计算交织
                for i in range(4):
                    i0 = 4*i
                    row = J.gpr(lane_div_16[0] + i0)
                    col = lane_mod_16[0]
                    vaddr = J.gpr(row*(S*sizeof_bf16) + (col^(row))*16 + vaddr_warp_base)
                    J.ds_write_b128(vaddr, key_reg[4*n+i], mod=f"offset:{lds_key}")
                    # J.debug_log(key_reg[16*n+i0:16*n+i0+3], torch.bfloat16, "4v.16h.8h")

                temp_key = J.gpr(2, 4, "au32") # double buffer of 16x32xbf16

                row = lane_mod_16[0]
                col = lane_div_16[0]
                vaddr = J.gpr(row*(S*sizeof_bf16) + (col^(row))*16 + vaddr_warp_base)

                # 因为写入和读取layout不同，必须等待写入完全结束才能发起读取
                #J.s_waitcnt(mod=f"lgkmcnt({0})")
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
                    acc_in = 0 if k==0 else acc[n]
                    # 7.5. Dependency Resolution: Required Independent Instructions
                    J.v_mfma_f32_16x16x16_bf16(acc[n], temp_key[k&1, 0:1], q_cur[4*k+0:4*k+1], acc_in)
                    #J.s_nop(15)
                    J.v_mfma_f32_16x16x16_bf16(acc[n], temp_key[k&1, 2:3], q_cur[4*k+2:4*k+3], acc[n])
                    # 发起下一轮的vmem预取，有很大概率会引起issue stall
                    # 因此我们跟上面耗时的LDS+MFMA指令交织起来，降低指令密度，降低stall概率
                    if (part_idx + 1) < num_parts:
                        if next_round_load_key < 4:
                            i0 = 4*next_round_load_key
                            kv_off[0], kv_off[1] = kv_offsets[n*4 + next_round_load_key], 0
                            J.v_lshlrev_b64(kv_off, hks_bits, kv_off) # 从这个移位开始就可能产生超过4GB 32位的值
                            vKaddr64bits[:] = vKbase64bits[:] + kv_off[:]
                            J.global_load_dwordx4(key_reg[4*n + next_round_load_key], vKaddr64bits, "off")
                            vm_cnt_preload_key += 1
                            next_round_load_key += 1
                if (part_idx + 1) < num_parts:
                    assert next_round_load_key == 4

            # online-softmax 计算开始，此时可以交织完成value数据的准备以避免issue stall
            if debug_loc is not None:
                J.debug_log(acc[...], torch.float, "4h.4h.16v.4h")
                #J.debug_log(q_cur, torch.bfloat16, "4h.4h.16v.8h")
                #J.s_waitcnt(mod=f"vmcnt(0) lgkmcnt(0)");J.s_endpgm()

            for n in range(4):
                for i in range(4):
                    acc[n,i] = acc[n,i] * acc_scale

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
                for i in range(4):
                    J.SetMask("vcc", k_pos + i < kv_part_len)
                    J.v_cndmask_b32_e32(acc[n,i], fmin, acc[n,i], "vcc")
                k_pos[0] = k_pos[0] + 16

            # cur_max: cur_max = torch.maximum(rowmax, prev_max)
            cur_max = J.gpr("vf32")
            cur_max[0] = prev_max[0]
            for n in range(4):
                for i in range(4):
                    J.v_max_f32(cur_max, cur_max, acc[n, i])

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
                for i in range(4):
                    J.v_exp_f32(acc[n,i], (acc[n,i] - cur_max) * alpha)
            
            # max_fixup = (prev_max-cur_max).exp()
            max_fixup = J.gpr(prev_max - cur_max)
            J.v_exp_f32(max_fixup, max_fixup * alpha)

            # cur_sum = max_fixup * cur_sum + P.sum(dim=1, keepdim = True)
            psum = J.gpr(2, "vf32")
            psum[0] = 0
            psum[1] = 0
            for n in range(4):
                J.v_pk_add_f32(psum, psum, acc[n,0:1])
                J.v_pk_add_f32(psum, psum, acc[n,2:3])
                #for i in range(4): psum[0] = psum[0] + acc[n,i]
            psum[0] = psum[0] + psum[1]

            vtemp = J.gpr("vf32")
            for mask in [32, 16]:
                J.ds_bpermute_b32(vtemp, (lane_id ^ mask)*4, psum[0])
                J.s_waitcnt(mod=f"lgkmcnt(0)")
                psum[0] = psum[0] + vtemp

            cur_sum[0] = cur_sum * max_fixup + psum[0]

            # prev_max = cur_max
            prev_max[0] = cur_max

            # O = max_fixup * O + (P @ V)
            if debug_loc is not None:
                J.debug_log(cur_sum, torch.float, "4h.16v.1h")
                J.debug_log(max_fixup, torch.float, "4h.16v.1h")

            # out : max_fixup是per-row的, vout列交织并不影响
            # O = max_fixup*O + (P @ V)
            for i in range(S//16):
                vout[i,0] = vout[i,0] * max_fixup
                vout[i,1] = vout[i,1] * max_fixup
                vout[i,2] = vout[i,2] * max_fixup
                vout[i,3] = vout[i,3] * max_fixup

            # P 转换为 bf16
            acc_low = J.gpr(4, 2, "vbf16x2", align=4) # 4 x bfloat16x4
            #acc_low = acc
            for n in range(4):
                acc_low[n,0] = (acc[n,0]>>16)|(acc[n,1]&0xFFFF0000)
                acc_low[n,1] = (acc[n,2]>>16)|(acc[n,3]&0xFFFF0000)

            # O += (P@V) 需要用到value了，等待前一轮value就位
            
            cur_v_write_lds = J.gpr(lane_div_16 * (S * 2) + lane_mod_16 * (8 * 2) + vaddr_warp_base)

            vm_cnt_preload_value = 0
            for n in range(4):
                # 等待value到达，发起下一轮preload之前发起计算
                J.s_waitcnt(mod=f"vmcnt({vm_cnt_preload_key + 16 - (n+1)*4})")

                # write 4x128 elements to lds for each call
                J.ds_write_b128(cur_v_write_lds, value_reg[n*4 + 0], mod='offset:0')
                J.ds_write_b128(cur_v_write_lds, value_reg[n*4 + 1], mod='offset:1024')
                J.ds_write_b128(cur_v_write_lds, value_reg[n*4 + 2], mod='offset:2048')
                J.ds_write_b128(cur_v_write_lds, value_reg[n*4 + 3], mod='offset:3072')

                v_curs = J.gpr(4, 2, 'vbf16x2')     # 4 16x16 bf16
                v_curs_tr = J.gpr(4, 2, 'vbf16x2')  # 4 16x16 bf16
                # read 64 elements from lds for each iter
                for j in range(S // 64):
                    cur_v_read_lds = J.gpr(vaddr_warp_base + lane_div_16 * (4 * S * 2) + (j * 64 * 2) + lane_mod_16 * (4 * 2))
                    J.ds_read_b64(v_curs[0], cur_v_read_lds, mod=f'offset:0')
                    J.ds_read_b64(v_curs[1], cur_v_read_lds, mod=f'offset:{S * 2}')
                    J.ds_read_b64(v_curs[2], cur_v_read_lds, mod=f'offset:{S * 4}')
                    J.ds_read_b64(v_curs[3], cur_v_read_lds, mod=f'offset:{S * 6}')
                    J.s_waitcnt(mod='lgkmcnt(0)')
                    J.transpose_per_lane(4, 4, 2, v_curs[...], v_curs_tr[...])

                    #J.debug_log(v_curs_tr, torch.bfloat16, "4v1.4h.16v4.4h")

                    J.v_mfma_f32_16x16x16_bf16(vout[j*4 + 0], v_curs_tr[0], acc_low[n], vout[j*4 + 0])
                    J.v_mfma_f32_16x16x16_bf16(vout[j*4 + 1], v_curs_tr[1], acc_low[n], vout[j*4 + 1])
                    J.v_mfma_f32_16x16x16_bf16(vout[j*4 + 2], v_curs_tr[2], acc_low[n], vout[j*4 + 2])
                    J.v_mfma_f32_16x16x16_bf16(vout[j*4 + 3], v_curs_tr[3], acc_low[n], vout[j*4 + 3])

                if (part_idx + 1) < num_parts:
                    vVaddr64bits = J.gpr(2, "vu32")
                    for i0 in range(4):
                        i = n*4 + i0
                        kv_off[0], kv_off[1] = kv_offsets[i], 0
                        J.v_lshlrev_b64(kv_off, hks_bits, kv_off) # 从这个移位开始就可能产生超过4GB 32位的值
                        vVaddr64bits[:] = vVbase64bits[:] + kv_off[:]
                        # 计算下轮预取需要用到的offset
                        if (part_idx + 2) < num_parts:
                            J.ds_bpermute_b32(kv_offsets[i], cur_bperm, kv_ids[part_idx + 2], mod=f"offset:{i*4*4}")
                        J.global_load_dwordx4(value_reg[i], vVaddr64bits, "off")
                        vm_cnt_preload_value += 1

            if debug_loc is not None:
                # vout: 是列交织的
                #J.debug_log(vout, torch.float, "4v1.4h.16v4.4h")
                pass

        reduce(J, warp_id, out_seg, max_out, sum_out, 
               lds_base=lds_key, cur_sum=cur_sum, cur_max=cur_max, vout=vout)
        vout_tr = J.gpr(S//16, 4, 'vf32')
        J.transpose_per_lane(4, 4, 4, vout[0:3], vout_tr[0:3])
        J.transpose_per_lane(4, 4, 4, vout[4:7], vout_tr[4:7])

        if debug_loc is not None:
            def dump_vout(debug_log_ptr, gprs):
                for i in range(2):
                    #vdata = vout_tr[i * 16 : i * 16 + 15]
                    voffset = J.gpr(i * 64 * 4 + lane_mod_16 * (S * 4) + lane_div_16 * (16 * 4))
                    J.global_store_dwordx4(voffset, vout_tr[i*4 + 0], debug_log_ptr, mod=f"offset:{0}")
                    J.global_store_dwordx4(voffset, vout_tr[i*4 + 1], debug_log_ptr, mod=f"offset:{16}")
                    J.global_store_dwordx4(voffset, vout_tr[i*4 + 2], debug_log_ptr, mod=f"offset:{32}")
                    J.global_store_dwordx4(voffset, vout_tr[i*4 + 3], debug_log_ptr, mod=f"offset:{48}")
                return (16, S)
            J.debug_log(vout_tr[...], torch.float, dump_vout)

        J.s_waitcnt(mod=f"lgkmcnt({0})")

        #a=J.alloc_lds((8-4)*1024)
        J.s_waitcnt(mod=f"vmcnt({0})")

        mtime_stop = J.gpr(2, "su32")
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

    @pyhip.jit()
    def pa_reduce_jit(J, kv_indptr:"uint*",
                        out_seg:"__bf16*",
                        max_out:"float*",
                        sum_out:"float*",
                        out:"__bf16*",
                        max_part:"int",
                        checks:"float*"):
        # asm volatile(";xxx  %0  %1  %2"::"s"(blockIdx.x),"s"(blockIdx.y),"s"(blockIdx.z));
        # 上面的hip代码诱导编译器告诉我们blockIdx存放位置是s2/3/4
        b = J.blockIdx.x
        hq = J.blockIdx.y
        lane_id = J.gpr(J.threadIdx.x[0] % 64)
        warp_id = J.gpr(J.threadIdx.x[0] // 64)
        s_warp_id = J.gpr("su32")
        J.v_readfirstlane_b32(s_warp_id, warp_id)

        # 每个WG处理一个batch的一个head，
        offset1 = J.gpr(b * HQ * max_part + hq * max_part)
        offset4 = J.gpr(offset1 * 4)

        # 支持2xsgpr和sgpr的运算？
        # 支持指令参数表达式？可以节省代码函数和手工分配临时变量寄存器？
        J.s_add_u32(max_out[0], max_out[0], offset4)
        J.s_addc_u32(max_out[1], max_out[1], 0)

        kv_inds = J.gpr(2, "si32",align=2)
        J.s_load_dwordx2(kv_inds, kv_indptr, 0)
        J.s_waitcnt(mod=f"lgkmcnt(0)") # 这类的wait应该可以自动生成，在第一次使用load指令结果的地方？
        kv_len = J.gpr(kv_inds[1] - kv_inds[0])
        part_num = J.gpr((kv_len[0] + (KV_PART_SIZE - 1))//KV_PART_SIZE)

        # 每个wave都独立的把最大值求出来，这一步其实可以WG内的wave协作分工来求
        # 最后来一次跨wave的reduce
        real_max = J.gpr("vf32")
        real_max[0] = torch.finfo(torch.float).min
        vi = J.auto_gpr(lane_id[0])
        with J.While() as loop:
            with J.ExecMask(vi < part_num):
                vdst = J.gpr("vf32")
                J.global_load_dword(vdst, vi[0] << 2, max_out)
                J.s_waitcnt(mod=f"vmcnt(0)")
                J.v_max_f32(real_max, real_max, vdst)
            J.s_cbranch_scc0(mod=loop["end"])# 没有非零的exec-mask，退出loop
            vi[0] = vi + 64

        # per-wave cross-lane reduce
        real_max = J.reduce("v_max_f32", real_max)

        cur_val = J.gpr(2, "vf32")
        cur_val[0] = 0
        cur_val[1] = 0
        warp_sum = J.gpr("vf32")
        warp_sum[0] = 0

        # N个wave协作加权part_num这么多个token，但是part_num可能不能被N整除，
        # 因此循环次数每个wave的都不一样，但是一个wave内部的所有threads/lanes循环次数一样
        # 因此无需ExecMask

        J.s_add_u32(out_seg[0], out_seg[0], offset1*(S*2))
        J.s_addc_u32(out_seg[1], out_seg[1], 0)
        
        J.s_add_u32(sum_out[0], sum_out[0], offset4)
        J.s_addc_u32(sum_out[1], sum_out[1], 0)

        J.s_add_u32(out[0], out[0], (b * HQ * S + hq * S)*2)
        J.s_addc_u32(out[1], out[1], 0)

        J.s_waitcnt(mod=f"lgkmcnt(0)")    
        si = J.gpr("si32")
        J.v_readfirstlane_b32(si, warp_id)
        with J.While(si < part_num) as loop:
            cur_val_low = J.gpr(2, "vf32") # bf16x2 => f32x2
            vaddr = J.gpr("vi32")

            soff = J.gpr(si*(2*S))
            vaddr[0] = lane_id*4 + soff

            cur_max = J.gpr("sf32")
            cur_sum = J.gpr("sf32")
            soff = J.gpr(si << 2)
            J.s_load_dword(cur_max, max_out, soff)
            J.s_load_dword(cur_sum, sum_out, soff)
            assert S % 64 == 0
            assert S == 128
            J.global_load_dword(cur_val_low[0], vaddr, out_seg)
            J.s_waitcnt(mod=f"vmcnt(0) lgkmcnt(0)")

            # bf16x2 => f32x2
            cur_val_low[1] = cur_val_low[0] & 0xffff0000
            cur_val_low[0] = cur_val_low[0] << 16
            
            # SALU do not support float
            import math
            alpha = math.log2(math.exp(1))

            exp_d_max = J.gpr("vf32")
            J.v_exp_f32_e32(exp_d_max, (cur_max[0] - real_max[0])*alpha)
            exp_d_max[0] = cur_sum[0] * exp_d_max

            cur_val[0] = cur_val[0] + cur_val_low[0] * exp_d_max[0]
            cur_val[1] = cur_val[1] + cur_val_low[1] * exp_d_max[0]

            warp_sum[0] = warp_sum[0] + exp_d_max[0]

            si[0] = si + 4

        sum_lds = J.alloc_lds(4*4)   # __shared__ float sum_lds[4];
        out_lds = J.alloc_lds(4*4*S) # __shared__ float out_lds[4 * S];

        vaddr = J.gpr("vi32")
        vaddr[0] = sum_lds + (s_warp_id<<2)
        J.ds_write_b32(vaddr, warp_sum)
        vaddr[0] = (S*4)
        vaddr[0] = out_lds + (s_warp_id*vaddr[0] + lane_id*(2*4))
        J.ds_write_b64(vaddr, cur_val)
        J.s_waitcnt(mod=f"lgkmcnt(0)")

        J.s_barrier()

        J.Jump("exit_final", s_warp_id[0] != 0)

        other_sum = J.gpr(3, "vf32")
        vaddr[0] = sum_lds
        J.ds_read_b32(other_sum[0], vaddr, mod=f"offset:4")
        J.ds_read_b32(other_sum[1], vaddr, mod=f"offset:8")
        J.ds_read_b32(other_sum[2], vaddr, mod=f"offset:12")
        J.s_waitcnt(mod=f"lgkmcnt(0)")
        warp_sum[0] = warp_sum[0] + other_sum[0] + other_sum[1] + other_sum[2]

        other_val = J.gpr(2, "vf32")
        for i in range(1,4):
            vaddr[0] = out_lds + (i*S*4 + lane_id*(2*4))
            J.ds_read_b64(other_val, vaddr)
            J.s_waitcnt(mod=f"lgkmcnt(0)")
            J.v_pk_add_f32(cur_val, cur_val, other_val)

        inv_sum_scale = J.gpr("vf32")

        J.v_rcp_f32(inv_sum_scale, warp_sum[0] + 1e-6)
        J.v_mul_f32(cur_val[0], cur_val[0], inv_sum_scale)
        J.v_mul_f32(cur_val[1], cur_val[1], inv_sum_scale)

        bf16x2 = J.gpr((cur_val[1] & 0xffff0000) | (cur_val[0] >> 16))

        J.global_store_dword(lane_id<<2, bf16x2, out)
        J.s_waitcnt(mod=f"vmcnt(0)")

        if 0:
            J.Jump("skip", (b[0] == 0) & (hq[0] == 0) & (s_warp_id[0] == 0), reverse=True)
            checks = J.gpr(2, "su32",align=2)
            vaddr = J.gpr(lane_id[0]<<2)
            J.global_store_dword(vaddr, bf16x2, checks)
            J.s_waitcnt(mod=f"vmcnt(0)")
            J.Label("skip")
            J.s_nop(0)

        J.Label("exit_final")

    return pa_jit, pa_reduce_jit



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

def get_full_ref():
    ref_qk = query.reshape(B, HK, HQ // HK, -1) @ key_caches[0][1:KV_LEN*B +1].reshape(B, -1, HK, S).permute(0, 2, 3, 1)
    ref_qk = ref_qk.to(torch.float32)
    ref_qk = ref_qk * scale
    s = torch.softmax(ref_qk, dim=-1).to(value_cache.dtype)
    ref_out = s @ value_caches[0][1:KV_LEN*B+1].reshape(B, -1, HK, S).permute(0, 2, 1, 3)
    ref_out = ref_out.reshape(B, HQ, S)
    return ref_out.to(query.dtype)

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
        fa2_logs["O"].append(O)
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
    pa_jit, pa_reduce_jit = get_jit_kernel(HQ, HK, S, BLOCK_SIZE, KV_PART_SIZE, scale, num_parts, debug_loc)
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
    pa_reduce_jit([B, HQ], [256], 
                kv_indptrs[0].data_ptr(), my_out_seg.data_ptr(), my_max.data_ptr(), my_sum.data_ptr(), my_out.data_ptr(), max_num_parts,
                0)

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
    if "vout_tr[...]" in logs:
        accuracy["O"] = "pass"
        res = logs["vout_tr[...]"][-1][:(HQ//HK)].cuda()
        ref = fa2_logs["O"][-1]
        if not torch.allclose(res, ref, atol=0.05, rtol=0.05):
            accuracy["O"] = f"failed"
            print("============== ref =============")
            print(ref)
            print("============== res =============")
            print(res)
    print(accuracy)
    ref_out = get_full_ref()
    idx = torch.where(torch.abs(ref_out - my_out) > 0.05)
    if len(idx[0]):
        print(f'idx = {idx}\nref_out={ref_out[idx]}\ncur={my_out[idx]}')

    assert torch.allclose(ref_out, my_out, rtol=0.02, atol=0.02), "out is wrong"
    print('acc ok')

print("======================= test performance ==============================")
pa_jit, pa_reduce_jit = get_jit_kernel(HQ, HK, S, BLOCK_SIZE, KV_PART_SIZE, scale, num_parts, None)
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
    pa_reduce_jit([B, HQ], [256], 
                kv_indptrs[i].data_ptr(), my_out_seg.data_ptr(), my_max.data_ptr(), my_sum.data_ptr(), my_out.data_ptr(), max_num_parts,
                0)

    i = (i + 1) % BUF_COPY
print(f"[{B}, {HK}, {div_up(KV_LEN, KV_PART_SIZE)}]")
print(f"{accuracy=}")
