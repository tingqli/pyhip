import pyhip
import torch
import math

from pyhip.asmjit import Addr2D
torch.set_printoptions(linewidth=300)
torch.cuda.set_device(2)
torch.set_default_device('cuda')
torch.manual_seed(0)

def test_reduce():
    HQ = 16
    HK = 1
    S = 128
    GQA = HQ // HK
    # p_vout: f32[4, 16, 128], p_cur_maxes: f32[4, 16], p_cur_sums: f32[4, 16]
    # p_out_seg: bf16[16, 128], p_max_out: f32[16], p_sum_out: f32[16]
    @pyhip.jit("-ggdb")
    def kernel(J:pyhip.JIT, p_vout:"void*", p_cur_maxes:"void*", p_cur_sums:"void*",
                            p_out_seg:"void*", p_max_out:"void*", p_sum_out:"void*", max_part:"int"):
        warp_id = J.gpr("su32")
        v_warp_id = J.gpr(J.threadIdx.x[0] // 64)
        J.v_readfirstlane_b32(warp_id, v_warp_id)
        lane_id = J.gpr(J.threadIdx.x[0] % 64)

        # need to be merged beg
        lane4_id = J.gpr(lane_id >> 4)
        lane16_id = J.gpr(lane_id & 15)
        # need to be merged end

        tmp = J.gpr("su32")
        zero = J.gpr("su32")
        zero[0] = 0
        # p_vout += warp_id * (HQ * S * 4)
        tmp[0] = warp_id * (HQ * S * 4)
        J.s_add_u32(p_vout[0], p_vout[0], tmp)
        J.s_addc_u32(p_vout[1], p_vout[1], zero)
        # p_cur_maxes += warp_id * HQ
        tmp[0] = warp_id * (HQ * 4)
        J.s_add_u32(p_cur_maxes[0], p_cur_maxes[0], tmp)
        J.s_addc_u32(p_cur_maxes[1], p_cur_maxes[1], zero)
        # p_cur_sums += warp_id * HQ
        J.s_add_u32(p_cur_sums[0], p_cur_sums[0], tmp)
        J.s_addc_u32(p_cur_sums[1], p_cur_sums[1], zero)
        # max, sum
        cur_max = J.gpr("vf32")
        cur_sum = J.gpr("vf32")
        voff = J.gpr(lane16_id << 2)
        J.global_load_dword(cur_max, voff, p_cur_maxes)
        J.global_load_dword(cur_sum, voff, p_cur_sums)
        # vout
        vout_buf   = J.Buffer(p_vout, HQ * S * 4)
        vout = J.gpr(f'vf32x{S // 64 * 4 * 4}')
        for i in range(2):
            voff[0] = lane16_id * (S * 4) + i * 64 * 4 + lane4_id * (4 * 4 * 4)
            v_half = vout[i * 16 : i * 16 + 15]
            for j in range(4):
                vout_buf.load_dwordx4(v_half[j * 4 : j * 4 + 3], voff, 0, offset12=j*16)
        J.s_waitcnt(mod='vmcnt(0)')

        lds_base = J.alloc_lds(32 * 1024)
        def div_up(x, y): return (x + y - 1) // y
        def compute():
            b = J.blockIdx.x
            hk = J.blockIdx.y
            kv_part = J.blockIdx.z
            hq = hk * GQA
            offset1 = J.gpr(b * max_part * HQ + hq * max_part + kv_part)
            offset4 = J.gpr(offset1 * 4)
            J.s_add_u32(p_out_seg[0], p_out_seg[0], offset1 * (S * 2))
            J.s_addc_u32(p_out_seg[1], p_out_seg[1], 0)
            J.s_add_u32(p_max_out[0], p_max_out[0], offset4)
            J.s_addc_u32(p_max_out[1], p_max_out[1], 0)
            J.s_add_u32(p_sum_out[0], p_sum_out[0], offset4)
            J.s_addc_u32(p_sum_out[1], p_sum_out[1], 0)

            J.s_barrier()
            with J.ExecMask(lane4_id[0] == 0):
                # ds_write2st64_b32 v34, v109, v94 offset1:1
                addr = Addr2D(J, lds_base, warp_id, lane16_id * 4, 16 * 4)
                J.ds_write2_b32(addr.get_addr(), cur_max, cur_sum, mod=f'offset1:{4 * 16}')
            J.s_barrier()

            maxs = J.gpr(f'vf32x{GQA}')
            sums = J.gpr(f'vf32x{GQA}')
            gqa4 = div_up(GQA, 4)
            real_max = J.gpr(f'vf32x{gqa4}')
            real_sum = J.gpr(f'vf32x{gqa4}')
            for i in range(gqa4):
                m_token_id = gqa4 * v_warp_id + i
                # TODO: precompute offset
                addr = Addr2D(J, lds_base, 0, m_token_id * 4, 16 * 4)
                J.ds_read2_b32(maxs[4 * i + 0 : 4 * i + 1], addr.get_addr(), mod=f'offset1:{16}')
                J.ds_read2_b32(maxs[4 * i + 2 : 4 * i + 3], addr.get_addr(), mod=f'offset0:{2 * 16} offset1:{3 * 16}')
                J.ds_read2_b32(sums[4 * i + 0 : 4 * i + 1], addr.get_addr(), mod=f'offset0:{4 * 16} offset1:{5 * 16}')
                J.ds_read2_b32(sums[4 * i + 2 : 4 * i + 3], addr.get_addr(), mod=f'offset0:{6 * 16} offset1:{7 * 16}')
                J.s_waitcnt(mod='lgkmcnt(0)')
                J.v_max_f32_e32(real_max[i], maxs[4 * i + 0], maxs[4 * i + 1])
                J.v_max3_f32(real_max[i], real_max[i], maxs[4 * i + 2], maxs[4 * i + 3])
                tmp = J.gpr(f'vf32x4')
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

            vout_low = J.gpr(f'vf32x{S // 64 * 4 * 2}')
            for k in range(S // 64):
                vout_low_4 = vout_low[k * 8 : k * 8 + 7]
                vout_4 = vout[k * 16 : k * 16 + 15]
                for i in range(4):
                    vout_low_4_r = vout_low_4[2 * i : 2 * i + 1]
                    vout_4_r = vout_4[4 * i : 4 * i + 3]
                    # TODO: transpose
                    for j in range(2):
                        vout_low_4_r[j] = (vout_4_r[2 * j] >> 16) | (vout_4_r[2 * j + 1] & 0xffff0000)
            J.s_barrier()

            addr = Addr2D(J, lds_base, v_warp_id, S * 2 * lane16_id, 16 * S * 2)
            lane4_id_4 = lane4_id * 4
            for k in range(S // 64):
                J.ds_write_b64(addr.get_addr() + ((lane4_id_4    ) ^ lane16_id) * 8, vout_low[8 * k + 0 : 8 * k + 1], mod=f'offset:{k * 64 * 2}')
                J.ds_write_b64(addr.get_addr() + ((lane4_id_4 + 1) ^ lane16_id) * 8, vout_low[8 * k + 2 : 8 * k + 3], mod=f'offset:{k * 64 * 2}')
                J.ds_write_b64(addr.get_addr() + ((lane4_id_4 + 2) ^ lane16_id) * 8, vout_low[8 * k + 4 : 8 * k + 5], mod=f'offset:{k * 64 * 2}')
                J.ds_write_b64(addr.get_addr() + ((lane4_id_4 + 3) ^ lane16_id) * 8, vout_low[8 * k + 6 : 8 * k + 7], mod=f'offset:{k * 64 * 2}')
            J.s_barrier()

            s_m_token_id = J.gpr("su32")
            J.v_readfirstlane_b32(s_m_token_id, gqa4 * v_warp_id)
            offset1 = J.gpr(s_m_token_id * max_part)
            offset4 = J.gpr(offset1 * 4)
            J.s_add_u32(p_out_seg[0], p_out_seg[0], offset1 * (S * 2))
            J.s_addc_u32(p_out_seg[1], p_out_seg[1], 0)
            J.s_add_u32(p_max_out[0], p_max_out[0], offset4)
            J.s_addc_u32(p_max_out[1], p_max_out[1], 0)
            J.s_add_u32(p_sum_out[0], p_sum_out[0], offset4)
            J.s_addc_u32(p_sum_out[1], p_sum_out[1], 0)
            max_addr = J.gpr('vu32x2')
            sum_addr = J.gpr('vu32x2')
            J.v_lshl_add_u64(max_addr, p_max_out, 0, 0)
            J.v_lshl_add_u64(sum_addr, p_sum_out, 0, 0)

            for i in range(gqa4):
                m_token_id = gqa4 * v_warp_id + i
                s_m_token_id = J.gpr("su32")
                J.v_readfirstlane_b32(s_m_token_id, m_token_id)
                with J.ExecMask(m_token_id < GQA):
                    addr = Addr2D(J, lds_base, m_token_id, ((m_token_id ^ (lane_id >> 1)) * 2 + (lane_id & 1)) * 4, S * 2)
                    tmp = J.gpr('vf32x4')
                    J.ds_read_b32(tmp[0], addr.get_addr(), mod=f'offset:{16 * S * 2 * 0}')
                    J.ds_read_b32(tmp[1], addr.get_addr(), mod=f'offset:{16 * S * 2 * 1}')
                    J.ds_read_b32(tmp[2], addr.get_addr(), mod=f'offset:{16 * S * 2 * 2}')
                    J.ds_read_b32(tmp[3], addr.get_addr(), mod=f'offset:{16 * S * 2 * 3}')
                    J.s_waitcnt(mod='lgkmcnt(0)')
                    out_v = J.gpr('vf32x2')
                    out_v[0] = tmp[0] << 16
                    out_v[1] = tmp[0] & 0xffff0000

                    exp_d_max = J.gpr("vf32")
                    J.v_exp_f32_e32(exp_d_max, (maxs[4 * i + 0] - real_max[i]) * alpha)
                    for j in range(2):
                        out_v[j] = sums[4 * i + 0] * exp_d_max * out_v[j]
                    for k in range(1, 4):
                        tmp[0] = tmp[k] << 16
                        tmp[1] = tmp[k] & 0xffff0000
                        exp_d_max = J.gpr("vf32")
                        J.v_exp_f32_e32(exp_d_max, (maxs[4 * i + k] - real_max[i]) * alpha)
                        for j in range(2):
                            tmp[j] = sums[4 * i + k] * exp_d_max * tmp[j]
                            out_v[j] += tmp[j]
                    inv_sum_scale = J.gpr('vf32')
                    J.v_rcp_f32(inv_sum_scale, real_sum[i] + 1e-6)
                    J.v_mul_f32(out_v[0], out_v[0], inv_sum_scale)
                    J.v_mul_f32(out_v[1], out_v[1], inv_sum_scale)
                    bf16x2 = J.gpr((out_v[1] & 0xffff0000) | (out_v[0] >> 16))

                    J.global_store_dword(lane_id << 2, bf16x2, p_out_seg)
                    J.global_store_dword(max_addr, real_max[i], mod='off')
                    J.global_store_dword(sum_addr, real_sum[i], mod='off')
                    p_out_seg[0] += S * 2
                    max_addr[0] += 4
                    sum_addr[0] += 4
                    J.s_waitcnt(mod=f"vmcnt(0)")

        compute()

        return

    num_blocks = 1
    waves_per_block = 4
    # [4, 16, 128]
    vout = torch.randn([waves_per_block, HQ, S], dtype=torch.float32)
    # [4]
    cur_maxes = torch.randn([waves_per_block, HQ, 1], dtype=torch.float32)
    # [4]
    cur_sums  = torch.randint(-1, 2, [waves_per_block, HQ, 1], dtype=torch.float32)
    PART = 1
    B = 1
    # [B, HQ, PART, S]
    p_out_seg = torch.zeros([B, HQ, PART, S], dtype=torch.bfloat16)
    # [B, HQ, PART]
    p_max_out = torch.zeros([B, HQ, PART], dtype=torch.float32)
    # [B, HQ, PART]
    p_sum_out = torch.zeros([B, HQ, PART], dtype=torch.float32)

    kernel([num_blocks],[64*waves_per_block], vout.data_ptr(), cur_maxes.data_ptr(), cur_sums.data_ptr(),
            p_out_seg.data_ptr(), p_max_out.data_ptr(), p_sum_out.data_ptr(), PART)
    def get_ref():
        real_max = cur_maxes.max(dim=0, keepdim=True)[0]
        real_sum = torch.sum(torch.exp(cur_maxes - real_max) * cur_sums, dim=0)
        ref_out = vout * cur_sums * torch.exp(cur_maxes - real_max)
        ref_out = torch.sum(ref_out, dim=0) / real_sum
        return real_max, real_sum, ref_out.to(torch.bfloat16).reshape([B, HQ, PART, S])
    ref_max, ref_sum, ref_out = get_ref()
    if not torch.allclose(p_max_out, ref_max):
        print(f'{p_max_out=} {ref_max=}')
        assert 0
    if not torch.allclose(p_sum_out, ref_sum):
        print(f'{p_sum_out=} {ref_sum=}')
        assert 0
    if not torch.allclose(p_out_seg, ref_out, rtol=0.02, atol=0.02):
        idx = torch.where(torch.abs(ref_out - p_out_seg) > 0.02)
        if len(idx[0]):
            print(f'idx = {idx}\nref_out={ref_out[idx]}\ncur={p_out_seg[idx]}')
        assert 0
    print('done')

if __name__ == "__main__":
    test_reduce()

