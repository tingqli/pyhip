import os
os.environ['PYHIP_JIT_LOG'] = '1'
import pyhip
import torch
import math

from pyhip.asmjit import Addr2D, float_to_ieee754_bits_little
torch.cuda.set_device(2)
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


def reduce(J:pyhip.JIT,
            GQA,
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
        J.v_exp_f32_e32(tmp[0], maxs[4 * i + 0] - real_max[i])
        J.v_exp_f32_e32(tmp[1], maxs[4 * i + 1] - real_max[i])
        J.v_exp_f32_e32(tmp[2], maxs[4 * i + 2] - real_max[i])
        J.v_exp_f32_e32(tmp[3], maxs[4 * i + 3] - real_max[i])
        J.v_pk_mul_f32(tmp[0:1], tmp[0:1], sums[4 * i + 0 : 4 * i + 1])
        J.v_pk_mul_f32(tmp[2:3], tmp[2:3], sums[4 * i + 2 : 4 * i + 3])
        J.v_add_f32_e32(tmp[0], tmp[0], tmp[1])
        J.v_add_f32_e32(tmp[2], tmp[2], tmp[3])
        J.v_add_f32_e32(real_sum[i], tmp[0], tmp[2])

    offset_max_sum = max_num_parts * 4
    for i in range(gqa4):
        m_token_id = gqa4 * warp_id + i
        with J.ExecMask(m_token_id < GQA):
            tmp = J.gpr("su32")
            J.v_readfirstlane_b32(tmp, real_max[i])
            J.s_store_dword(tmp, p_max_out, 0, mod="glc")
            J.v_readfirstlane_b32(tmp, real_sum[i])
            J.s_store_dword(tmp, p_sum_out, 0, mod="glc")
            p_max_out[0] += offset_max_sum
            p_sum_out[0] += offset_max_sum

    vout_low = J.gpr(S // 64 * 4 * 2, 'vf32')
    for k in range(S // 64):
        vout_low_4 = vout_low[k * 8 : k * 8 + 7]
        for i in range(4):
            # NOTE: the following will be error:
            # vout_4 = vout[k * 4 : k * 4 + 3]
            # vout_low_4[2 * i + 1] = (vout_4[2][i] >> 16) | (vout_4[3][i] & 0xffff0000)
            vout_low_4[2 * i + 0] = (vout[4 * k + i, 0] >> 16) | (vout[4 * k + i, 1] & 0xffff0000)
            vout_low_4[2 * i + 1] = (vout[4 * k + i, 2] >> 16) | (vout[4 * k + i, 3] & 0xffff0000)

    J.s_barrier()

    addr = Addr2D(J, lds_base, warp_id, S * 2 * lane_mod_16, 16 * S * 2)
    for k in range(S // 64):
        J.ds_write_b64(addr.get_addr() + ((lane_div_16    ) ^ lane_mod_16) * 8, vout_low[8 * k + 0 : 8 * k + 1], mod=f'offset:{k * 64 * 2}')
        J.ds_write_b64(addr.get_addr() + ((lane_div_16 + 4) ^ lane_mod_16) * 8, vout_low[8 * k + 2 : 8 * k + 3], mod=f'offset:{k * 64 * 2}')
        J.ds_write_b64(addr.get_addr() + ((lane_div_16 + 8) ^ lane_mod_16) * 8, vout_low[8 * k + 4 : 8 * k + 5], mod=f'offset:{k * 64 * 2}')
        J.ds_write_b64(addr.get_addr() + ((lane_div_16 +12) ^ lane_mod_16) * 8, vout_low[8 * k + 6 : 8 * k + 7], mod=f'offset:{k * 64 * 2}')
    J.s_barrier()

    offset_out = max_num_parts * (S * 2)
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
            J.v_exp_f32_e32(exp_d_max, maxs[4 * i + 0] - real_max[i])
            for j in range(2):
                out_v[j] = exp_d_max * out_v[j]
            for k in range(1, 4):
                tmp[0] = tmp[k] << 16
                tmp[1] = tmp[k] & 0xffff0000
                exp_d_max = J.gpr("vf32")
                J.v_exp_f32_e32(exp_d_max, maxs[4 * i + k] - real_max[i])
                for j in range(2):
                    tmp[j] = exp_d_max * tmp[j]
                    out_v[j] += tmp[j]
            inv_sum_scale = J.gpr('vf32')
            J.v_rcp_f32(inv_sum_scale, real_sum[i] + 1e-6)
            J.v_mul_f32(out_v[0], out_v[0], inv_sum_scale)
            J.v_mul_f32(out_v[1], out_v[1], inv_sum_scale)
            bf16x2 = J.gpr((out_v[1] & 0xffff0000) | (out_v[0] >> 16))

            J.global_store_dword(lane_id << 2, bf16x2, p_out_seg, mod='nt')
            p_out_seg[0] += offset_out
            # J.s_waitcnt(mod=f"vmcnt(0)")

@pyhip.jit("-g")
def pa_jit(J:pyhip.JIT,
            HQ,HK,S,BLOCK_SIZE,KV_PART_SIZE,acc_scale,num_parts,
            query:"__bf16*",        # [B, HQ, S]
            key_cache:"__bf16*",    # [block_num, HK, block_size // (16*2), 2, S // ITEMSIZE, 16, ITEMSIZE]
            value_cache:"__bf16*",  # [block_num, HK, block_size // (4*ITEMSIZE), S // 16, 4, 16, ITEMSIZE]
            block_table:"uint*",    # [B, max_num_blocks_per_seq]
            seq_len:"uint*",        # [B]
            out_seg:"__bf16*",      # [B, HQ, max_num_parts, S]
            max_out:"float*",       # [B, HQ, max_num_parts, 1]
            sum_out:"float*",       # [B, HQ, max_num_parts, 1]
            # max_num_parts (max number of workgroups for one-item/one-head-group in batch)
            max_num_parts:"uint",
            max_num_blocks_per_seq:"uint",
            ):
    acc_scale *= math.log2(math.exp(1))
    GQA = HQ // HK
    '''
    偏移之后，找到本WG需要处理的数据：
        
    '''
    b = J.blockIdx.x
    hk = J.blockIdx.y
    kv_part = J.blockIdx.z
    sizeof_bf16 = 2

    # 这个读取latency非常大，需要提前发起hidding,
    # kv-length for current batch
    kv_len = J.gpr(1, "su32")
    J.s_load_dword(kv_len, seq_len, b[0]<<2)

    s_acc_scale = J.gpr(2, 'sf32')
    s_acc_scale[0] = acc_scale
    s_acc_scale[1] = acc_scale

    assert (HQ % HK) == 0
    hq = J.gpr(hk * (HQ // HK))

    # query  [B, HQ, S] => [1, (HQ//HK), S]
    query[:] = query + (b*(HQ*S*2) + hq*(S*2))
    # key_cache : [block_num, HK, block_size // (16*2), 2, S // ITEMSIZE, 16, ITEMSIZE]
    key_cache[:] = key_cache[:] + hk[0]*(S*2*BLOCK_SIZE)
    # value_cache: [block_num, HK, block_size // (4*ITEMSIZE), S // 16, 4, 16, ITEMSIZE]
    value_cache[:] = value_cache[:] + hk[0]*(S*2*BLOCK_SIZE)
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

    # early return
    J.Jump("continue_following", kv_part * KV_PART_SIZE < kv_len)
    J.s_endpgm()
    J.Label("continue_following")

    kv_len_start = J.gpr("su32")
    kv_len_end  = J.gpr("su32")

    J.s_min_u32(kv_len_start, kv_part * KV_PART_SIZE, kv_len)
    J.s_min_u32(kv_len_end,   kv_len_start + KV_PART_SIZE, kv_len)
    block_table[:] = block_table[:] + (b[0] * max_num_blocks_per_seq + kv_len_start // BLOCK_SIZE) * 4

    kv_part_len = J.gpr(kv_len_end[0] - kv_len_start[0])
    block_table_buff = J.Buffer(block_table, (kv_part_len + (BLOCK_SIZE - 1)) // BLOCK_SIZE * 4)

    assert BLOCK_SIZE == 32 or BLOCK_SIZE == 64, 'block size must be 32 or 64'
    # 加载block_table的初始延迟比较大，等待之前一次性多读入全部要用到的可以提高效率
    # block_table会对越界的索引返回0，因此cur_kv_ids中越界的部分被0填充不会产生越界
    # the block number each wave will access
    kv_ids_num = KV_PART_SIZE // 4 // BLOCK_SIZE
    kv_ids = J.gpr(1, 'vu32')
    block_table_buff.load_dword(kv_ids[0], lane_mod_16 % kv_ids_num * 4, warp_id * (kv_ids_num * 4))
    assert 0 < kv_ids_num <= 16, f'kv id# per simd must be in (0, 16]'

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
    J.v_add_co_u32(vKbase64bits[0], "vcc", lane_id*16, vKbase64bits[0])
    J.v_addc_co_u32(vKbase64bits[1], "vcc", 0, vKbase64bits[1], "vcc")
    J.v_add_co_u32(vVbase64bits[0], "vcc", lane_id*16, vVbase64bits[0])
    J.v_addc_co_u32(vVbase64bits[1], "vcc", 0, vVbase64bits[1], "vcc")

    # 64x128 bf16
    lds_key = J.alloc_lds(64 * S * sizeof_bf16)

    mtime_start = J.gpr(2, "su32")
    #J.s_memtime(mtime_start)

    # 预取第一part的key/value
    J.s_waitcnt(mod=f"vmcnt(0)")

    # 形成 offsets
    kv_offsets = J.gpr(16, "vu32")

    # 预取第一轮需要的数据
    kv_off = J.gpr(2, "vu32")
    hks_bits = J.shift_bits(HK*S*2*BLOCK_SIZE)
    vKaddr64bits = J.gpr(2, "vu32")
    vVaddr64bits = J.gpr(2, "vu32")

    for i in range(kv_ids_num):
        J.v_mov_b32_dpp(kv_offsets[i], kv_ids[0], mod=f'row_newbcast:{i} row_mask:0xf bank_mask:0xf')
    # block number each part iter will access
    kv_ids_num_per_part = kv_ids_num // num_parts
    assert kv_ids_num_per_part > 0
    # shared row(=16tokens) num
    key_shared_idx_per_64 = 4 // kv_ids_num_per_part
    value_shared_idx_per_64 = 2 // kv_ids_num_per_part
    next_key_block_idx = 0
    s_4096 = J.gpr(2, 'su32')
    s_4096[0], s_4096[1] = 4096, 0
    # iter 4 times x 16 tokens 
    for n in range(4):
        if n % key_shared_idx_per_64 == 0:
            kv_off[0], kv_off[1] = kv_offsets[next_key_block_idx], 0
            J.v_lshlrev_b64(kv_off, hks_bits, kv_off) # 从这个移位开始就可能产生超过4GB 32位的值
            vKaddr64bits[:] = vKbase64bits[:] + kv_off[:]
            next_key_block_idx += 1
            offset = 0
        # load along [16, S], S = 4x[4x8(aka lane4)]
        for j in range(4):
            J.global_load_dwordx4(key_reg[4 * n + j], vKaddr64bits, mod=f",off, offset:{offset}")
            offset += 1024
            if offset >= 4096:
                offset = 0
                vKaddr64bits[:] = vKaddr64bits[:] + s_4096[:]

    vm_cnt_preload_value = 0
    next_value_block_idx = 0
    # iter 2 times x 32 tokens 
    for n in range(2):
        if n % value_shared_idx_per_64 == 0:
            kv_off[0], kv_off[1] = kv_offsets[next_value_block_idx], 0
            J.v_lshlrev_b64(kv_off, hks_bits, kv_off) # 从这个移位开始就可能产生超过4GB 32位的值
            vVaddr64bits[:] = vVbase64bits[:] + kv_off[:]
            next_value_block_idx += 1
            offset = 0
        # load along [S, 32], S = [16(aka lane16)x8]
        for j in range(8):
            J.global_load_dwordx4(value_reg[8 * n + j], vVaddr64bits, mod=f",off, offset:{offset}")
            vm_cnt_preload_value += 1
            offset += 1024
            if offset >= 4096:
                offset = 0
                vVaddr64bits[:] = vVaddr64bits[:] + s_4096[:]

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
    # for stat info
    J.Label('main')
    for part_idx in range(num_parts):
        vm_cnt_preload_key = 0 # 统计从这个点开始，又发起了多少个vmem指令
        acc = J.gpr(4, 4, "vf32", align=4)
        for n in range(4):
            if (part_idx + 1) < num_parts:
                if n % key_shared_idx_per_64 == 0:
                    assert next_key_block_idx < kv_ids_num
                    kv_off[0], kv_off[1] = kv_offsets[next_key_block_idx], 0
                    next_key_block_idx += 1
                    J.v_lshlrev_b64(kv_off, hks_bits, kv_off) # 从这个移位开始就可能产生超过4GB 32位的值
                    vKaddr64bits[:] = vKbase64bits[:] + kv_off[:]
                    offset = 0
            for k in range(S//32):
                if part_idx != num_parts - 1:
                    J.s_waitcnt(mod=f"vmcnt({vm_cnt_preload_value+16-1})")
                else:
                    J.s_waitcnt(mod=f"vmcnt({vm_cnt_preload_value+16-n*4-k-1})")

                #J.debug_log(temp_key[k&1], torch.bfloat16, "4h.16v.8h")
                acc_in = 0 if k==0 else acc[n]
                # 7.5. Dependency Resolution: Required Independent Instructions
                J.v_mfma_f32_16x16x16_bf16(acc[n], key_reg[4*n+k, 0:1], q_cur[4*k+0:4*k+1], acc_in)
                J.v_mfma_f32_16x16x16_bf16(acc[n], key_reg[4*n+k, 2:3], q_cur[4*k+2:4*k+3], acc[n])
                # 发起下一轮的vmem预取，有很大概率会引起issue stall
                # 因此我们跟上面耗时的LDS+MFMA指令交织起来，降低指令密度，降低stall概率
                if (part_idx + 1) < num_parts:
                    J.global_load_dwordx4(key_reg[4*n + k], vKaddr64bits, mod=f",off, offset:{offset}")
                    vm_cnt_preload_key += 1
                    offset += 1024
                    if offset >= 4096:
                        offset = 0
                        vKaddr64bits[:] = vKaddr64bits[:] + s_4096[:]

        # online-softmax 计算开始，此时可以交织完成value数据的准备以避免issue stall

        for n in range(4):
            J.v_pk_mul_f32(acc[n,0:1], s_acc_scale, acc[n,0:1])
            J.v_pk_mul_f32(acc[n,2:3], s_acc_scale, acc[n,2:3])

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
        first_idx = J.gpr(warp_id * (KV_PART_SIZE//4) + part_idx * (KV_PART_SIZE//num_parts//4))

        with J.If(first_idx + 64 > kv_part_len):
            k_pos = J.gpr("vu32")
            k_pos[0] = first_idx + (lane_div_16[0] * 8)
            fmin = J.gpr("vf32")
            fmin[0] = torch.finfo(torch.float).min
            for n in range(2):
                for i in range(8):
                    J.SetMask("vcc", k_pos + i < kv_part_len)
                    J.v_cndmask_b32_e32(acc[2*n+i//4,i%4], fmin, acc[2*n+i//4,i%4], "vcc")
                k_pos[0] = k_pos[0] + 32

        # cur_max: cur_max = torch.maximum(rowmax, prev_max)
        cur_max = J.gpr("vf32")
        cur_max[0] = prev_max[0]
        for n in range(4):
            J.vmax("f32", cur_max, [cur_max, acc[n, 0], acc[n, 1], acc[n, 2], acc[n, 3]])

            # cur_max: cross-lane
        vtemp = J.gpr("vf32")
        for mask in [32, 16]:
            J.ds_bpermute_b32(vtemp, (lane_id ^ mask)*4, cur_max)
            J.s_waitcnt(mod=f"lgkmcnt(0)")
            J.v_max_f32(cur_max, cur_max, vtemp)

        # acc现在保存P P = (S - cur_max).exp()
        for n in range(4):
            for i in range(4):
                J.v_exp_f32(acc[n,i], acc[n,i] - cur_max)
        
        # max_fixup = (prev_max-cur_max).exp()
        if part_idx:
            max_fixup = J.gpr(2, 'vf32')
            max_fixup[0] = J.gpr(prev_max - cur_max)
            J.v_exp_f32(max_fixup[0], max_fixup[0])

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

        if part_idx:
            cur_sum[0] = cur_sum * max_fixup[0] + psum[0]
        else:
            cur_sum[0] = psum[0]

        # prev_max = cur_max
        prev_max[0] = cur_max

        # O = max_fixup * O + (P @ V)

        if part_idx:
            # out : max_fixup是per-row的, vout列交织并不影响
            # O = max_fixup*O + (P @ V)
            max_fixup[1] = max_fixup[0]
            for i in range(S//16):
                J.v_pk_mul_f32(vout[i,0:1], max_fixup, vout[i,0:1])
                J.v_pk_mul_f32(vout[i,2:3], max_fixup, vout[i,2:3])

        # P 转换为 bf16
        acc_low = J.gpr(4, 2, "vbf16x2", align=4) # 4 x bfloat16x4
        #acc_low = acc
        for n in range(4):
            acc_low[n,0] = (acc[n,0]>>16)|(acc[n,1]&0xFFFF0000)
            acc_low[n,1] = (acc[n,2]>>16)|(acc[n,3]&0xFFFF0000)

        # O += (P@V) 需要用到value了，等待前一轮value就位

        vm_cnt_preload_value = 0
        # 2 groups, each group has 32 tokens of value
        for n in range(2):
            if (part_idx + 1) < num_parts:
                if n % value_shared_idx_per_64 == 0:
                    assert next_value_block_idx < kv_ids_num
                    kv_off[0], kv_off[1] = kv_offsets[next_value_block_idx], 0
                    next_value_block_idx += 1
                    J.v_lshlrev_b64(kv_off, hks_bits, kv_off) # 从这个移位开始就可能产生超过4GB 32位的值
                    vVaddr64bits[:] = vVbase64bits[:] + kv_off[:]
                    offset = 0

            # [S, 32 tokens] x [16 query, 4(lane4)*2group*(2*4)]'
            for j in range(S // 16):
                # 等待value到达，发起下一轮preload之前发起计算
                if part_idx != num_parts - 1:
                    J.s_waitcnt(mod=f"vmcnt({vm_cnt_preload_key + 16 - 1})")
                else:
                    J.s_waitcnt(mod=f"vmcnt({vm_cnt_preload_key + 16 - j - (n*8) - 1})")
                J.v_mfma_f32_16x16x16_bf16(vout[j], value_reg[n*8+j,0:1], acc_low[2*n+0, 0:1], vout[j])
                J.v_mfma_f32_16x16x16_bf16(vout[j], value_reg[n*8+j,2:3], acc_low[2*n+1, 0:1], vout[j])

                if (part_idx + 1) < num_parts:
                    J.global_load_dwordx4(value_reg[8*n + j], vVaddr64bits, mod=f",off, offset:{offset}")
                    vm_cnt_preload_value += 1
                    offset += 1024
                    if offset >= 4096:
                        vVaddr64bits[:] = vVaddr64bits[:] + s_4096[:]
                        offset = 0

    # for stat info
    J.Label('main_end')

    reduce(J, GQA, warp_id, out_seg, max_out, sum_out, 
            lds_base=lds_key, cur_sum=cur_sum, cur_max=cur_max, vout=vout)

    # J.s_waitcnt(mod=f"lgkmcnt({0})")

    #a=J.alloc_lds((8-4)*1024)
    # J.s_waitcnt(mod=f"vmcnt({0})")

    # scalar-memoryreads can return out-of-order:
    # following hack prevent kernel-arg loading to reuse same sgprs for different args
    return

@pyhip.jit()
def pa_reduce_jit(J, 
                  KV_PART_SIZE, HQ, S,
                  seq_len:"uint*",
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

    kv_len = J.gpr(1, "su32")
    J.s_load_dword(kv_len, seq_len, b[0]<<2)
    J.s_waitcnt(mod=f"lgkmcnt(0)") # 这类的wait应该可以自动生成，在第一次使用load指令结果的地方？
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
        exp_d_max = J.gpr("vf32")
        J.v_exp_f32_e32(exp_d_max, cur_max[0] - real_max[0])
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
KV_LEN = 45694

#KV_LEN = 512
DT = torch.bfloat16
BLOCK_SIZE16 = 16
# should be [32, 64]
BLOCK_SIZE = 32
BUF_COPY = 1
BUF_COPY = 32

KV_MIN_PART_SIZE = 256
# should be [256, 512, 1024]
KV_PART_SIZE = 256 * 4

query = torch.randint(-2, 3, [B, HQ, S], dtype=DT)
ITEMSIZE = 16 // query.itemsize
seq_lens = {}
key_caches = {}
value_caches = {}
block_tables = {}
max_num_blocks_per_seq = {}
block_num = {}
# [N, BLOCK_SIZE, HK, S]
for block_size in (BLOCK_SIZE16, BLOCK_SIZE):
    max_num_blocks_per_seq[block_size] = (KV_LEN + block_size - 1) // block_size
    block_num = B * max_num_blocks_per_seq[block_size]
    key_caches[block_size] = []
    value_caches[block_size] = []
    block_tables[block_size] = []
    seq_lens[block_size] = []
    for _ in range(BUF_COPY):
        if block_size == BLOCK_SIZE16:
            key_cache_shape = (block_num, HK, S // ITEMSIZE, block_size, ITEMSIZE)
        else:
            # [..., block_size // mfma M group, mfma M group, K // vec_size, mfma M, vec_size]
            # the highest 2 dimensions are for one 16x16x(4*8)mfma, `S // ITEMSIZE` is used for reduced dimension
            #   in order to match value_cache reduced dimension (4*8), the result of key*query should be also (4*8) which means each thread will have contious 8 elments.
            #   so the 5th `16` dimension and 3rd `2` come from:
            #      actual tokens 5th   3rd
            #      token 0- 3    0- 3  group0
            #      token 4- 7    0- 3  group1
            #      token 8-11    4- 7  group0
            #      token12-15    4- 7  group1
            #      token16-19    8-11  gourp0
            #      token20-23    8-11  gourp1
            #      token24-27   12-15  gourp0
            #      token28-31   12-15  gourp1
            key_cache_shape = (block_num, HK, block_size // (16*2), 2, S // ITEMSIZE, 16, ITEMSIZE)
        key_cache = torch.randint(-2, 3, key_cache_shape, dtype=DT)
        if block_size == BLOCK_SIZE16:
            value_cache_shape = (block_num, HK, block_size // ITEMSIZE, S, ITEMSIZE)
        else:
            # the highest 3 dimensions are for one 16x16x(4*8)mfma, `block_size // (4*ITEMSIZE)` is used for reduced dimension
            #   `S // 16` is for N dimension
            # [..., block_size // mfma K, M // 16, mfma K col, mfma M, vec_size]
            value_cache_shape = (block_num, HK, block_size // (4*ITEMSIZE), S // 16, 4, 16, ITEMSIZE)
        value_cache = torch.randint(-2, 3, value_cache_shape, dtype=DT)
        seq_len = torch.full(size=(B,), fill_value=KV_LEN, dtype=torch.int)
        block_table = torch.linspace(0, block_num - 1, block_num, dtype=torch.int32).reshape(B, max_num_blocks_per_seq[block_size])
        seq_lens[block_size].append(seq_len)
        key_caches[block_size].append(key_cache)
        value_caches[block_size].append(value_cache)
        block_tables[block_size].append(block_table)

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

def get_full_ref(block_size):
    # [block_num, HK, block_size // (16*2), 2, S // ITEMSIZE, 16, ITEMSIZE]
    #  ->[block_num, HK, block_size // (16*2), 2, 16, S // ITEMSIZE, ITEMSIZE]
    #  ->[block_num, HK, block_size // (16*2), 32, S]
    key_cache = key_caches[block_size][0].permute(0, 1, 2, 3, 5, 4, 6).reshape(-1, HK, block_size // 32, 32, S)
    #  ->[block_num, HK, block_size // (16*2), 8, 4, S]
    key_cache = key_cache.reshape(-1, HK, block_size // (16*2), 8, 4, S)
    # interleave 0, 1, ... 4, 5... to 0, 4, 1, 5 ...
    key_cache = key_cache[:,:,:,(0,4,1,5,2,6,3,7), :, :]
    #  ->[block_num, HK, block_size, S]
    key_cache = key_cache.reshape(-1, HK, block_size, S)
    #  ->[B, max_num_blocks_per_seq, HK, block_size, S] -> [B, HK, max_num_blocks_per_seq, block_size, S]
    key_cache = key_cache.reshape(B, max_num_blocks_per_seq[block_size], HK, block_size, S).permute(0, 2, 1, 3, 4)
    #  ->[B, HK, KV_LEN_pad, S]
    key_cache = key_cache.reshape(B, HK, -1, S)
    key_cache = key_cache[..., :KV_LEN, :].transpose(2, 3)
    ref_qk = query.reshape(B, HK, HQ // HK, -1) @ key_cache
    ref_qk = ref_qk.to(torch.float32)
    ref_qk = ref_qk * scale
    s = torch.softmax(ref_qk, dim=-1).to(key_cache.dtype)
    # [block_num, HK, block_size // (4*ITEMSIZE), S // 16, 4, 16, ITEMSIZE]
    #  ->[block_num, HK, block_size // (4*ITEMSIZE), 4, ITEMSIZE, S // 16, 16]
    #  ->[block_num, HK, block_size, S]
    value_cache = value_caches[block_size][0].permute(0, 1, 2, 4, 6, 3, 5).reshape(-1, HK, block_size, S)
    #  ->[B, max_num_blocks_per_seq, HK, block_size, S] -> [B, HK, max_num_blocks_per_seq, block_size, S]
    value_cache = value_cache.reshape(B, max_num_blocks_per_seq[block_size], HK, block_size, S).permute(0, 2, 1, 3, 4)
    #  ->[B, HK, KV_LEN_pad, S]
    value_cache = value_cache.reshape(B, HK, -1, S)
    value_cache = value_cache[..., :KV_LEN, :]
    ref_out = s @ value_cache
    ref_out = ref_out.reshape(B, HQ, S)
    return ref_out.to(query.dtype)

def run_aiter(query,
               key_cache,
               value_cache,
               block_tables,
               seq_lens,
               max_num_blocks_per_seq):
    return aiter.pa_fwd_asm(
        query,
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        max_num_blocks_per_seq,
    )

if 0:
    import aiter
    out_ref = None
    # if 0:
    #     torch.save({
    #         "q": query,
    #         "k": key_cache[1:45694+1],
    #         "v": value_cache[1:45694+1],
    #         "kv_page_indices": kv_page_indices,
    #         "out": out
    #     }, "/mywork/users/luocheng/sglang/mytest/pa.pt")
    if 0:
        import os.path
        data = torch.load(os.path.dirname(__file__) + '/pa.pt')
        query = data['q'].to(device=query.device)
        key_caches[-1][1:KV_LEN+1] = data['k']
        value_caches[-1][1:KV_LEN+1] = data['v']
        out_ref = data['out'].to(device=query.device)
    out = run_aiter(query=query, key_cache=key_caches[BLOCK_SIZE16][-1], value_cache=value_caches[BLOCK_SIZE16][-1],
                     block_tables=block_tables[BLOCK_SIZE16][-1], seq_lens=seq_lens[BLOCK_SIZE16][-1], max_num_blocks_per_seq=max_num_blocks_per_seq[BLOCK_SIZE16])

    if out_ref is not None:
        assert torch.allclose(out, out_ref), "aiter acc is wrong"
        print('aiter acc ok')
    i = 0
    for _ in range(10):
        with pyhip.cudaPerf(B * HQ // HK * KV_LEN * S * 2 * 2, B * (HK * KV_LEN * S * 2 * 2), name="aiter"):
            run_aiter(query=query, key_cache=key_caches[BLOCK_SIZE16][i], value_cache=value_caches[BLOCK_SIZE16][i],
                       block_tables=block_tables[BLOCK_SIZE16][i], seq_lens=seq_lens[BLOCK_SIZE16][i], max_num_blocks_per_seq=max_num_blocks_per_seq[BLOCK_SIZE16])
        i = (i + 1) % BUF_COPY

def test_acc():
    print("======================= verify correctness ==============================")
    pa_jit([B, HK, max_num_parts], [256],
           HQ, HK, S, BLOCK_SIZE, KV_PART_SIZE, scale, num_parts,
            query.data_ptr(),           # [B, HQ, S]
            key_caches[BLOCK_SIZE][0].data_ptr(),   # [BLOCK_NUM, HK, S // 8, BLOCK_SIZE, 8]
            value_caches[BLOCK_SIZE][0].data_ptr(), # [BLOCK_NUM, HK, S, BLOCK_SIZE // 8, 8]
            block_tables[BLOCK_SIZE][0].data_ptr(),
            seq_lens[BLOCK_SIZE][0].data_ptr(),
            my_out_seg.data_ptr(),
            my_max.data_ptr(), 
            my_sum.data_ptr(),
            max_num_parts,
            max_num_blocks_per_seq[BLOCK_SIZE])
    pa_reduce_jit([B, HQ], [256], KV_PART_SIZE, HQ, S,
                seq_lens[BLOCK_SIZE][0].data_ptr(), my_out_seg.data_ptr(), my_max.data_ptr(), my_sum.data_ptr(), my_out.data_ptr(), max_num_parts,
                0)

    ref_out = get_full_ref(BLOCK_SIZE)
    idx = torch.where(torch.abs(ref_out - my_out) > 0.05)
    if len(idx[0]):
        print(f'idx = {idx}\nref_out={ref_out[idx]}\ncur={my_out[idx]}')

    assert torch.allclose(ref_out, my_out, rtol=0.02, atol=0.02), "out is wrong"
    print('acc ok')

def test_perf():
    print("======================= test performance ==============================")

    i = 0
    for round in range(10):
        with pyhip.cudaPerf(B * HQ // HK * KV_LEN * S * 2 * 2, B * (HK * KV_LEN * S * 2 * 2), name="pa_jit"):
            pa_jit([B, HK, max_num_parts], [256],
                    HQ, HK, S, BLOCK_SIZE, KV_PART_SIZE, scale, num_parts,
                    query.data_ptr(),                       # [B, HQ, S]
                    key_caches[BLOCK_SIZE][i].data_ptr(),   # [BLOCK_NUM, HK, S // 8, BLOCK_SIZE, 8]
                    value_caches[BLOCK_SIZE][i].data_ptr(), # [BLOCK_NUM, HK, S, BLOCK_SIZE // 8, 8]
                    block_tables[BLOCK_SIZE][i].data_ptr(),
                    seq_lens[BLOCK_SIZE][i].data_ptr(),
                    my_out_seg.data_ptr(),
                    my_max.data_ptr(), 
                    my_sum.data_ptr(),
                    max_num_parts,
                    max_num_blocks_per_seq[BLOCK_SIZE])
            pa_reduce_jit([B, HQ], [256], KV_PART_SIZE, HQ, S,
                        seq_lens[BLOCK_SIZE][i].data_ptr(), my_out_seg.data_ptr(), my_max.data_ptr(), my_sum.data_ptr(), my_out.data_ptr(), max_num_parts,
                        0)

        i = (i + 1) % BUF_COPY
    print(f"[{B}, {HK}, {div_up(KV_LEN, KV_PART_SIZE)}]")

if __name__ == '__main__':
    test_acc()
    test_perf()