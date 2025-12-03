import pyhip
import torch
import math
torch.set_printoptions(linewidth=300)
torch.cuda.set_device(2)
torch.set_default_device('cuda')
torch.manual_seed(0)

def test_sv():
    M = 16
    K = 64
    N = 128
    N_ITER = 64
    # pS: bf16[16, 64*1], pV: bf16[64*1, 128], pSV: f32[4, 16, 128]
    @pyhip.jit("-ggdb")
    def kernel(J:pyhip.JIT, pScore:"void*", pValue:"void*", pOut:"void*", debug_out:"int*"):
        warp_id = J.gpr("su32")
        J.v_readfirstlane_b32(warp_id, J.threadIdx.x[0] // 64)
        lane_id = J.gpr(J.threadIdx.x[0] % 64)

        # need to be merged beg
        key_load_row_id = J.gpr(lane_id >> 4)
        key_load_col_id = J.gpr(lane_id & 15)
        # need to be merged end

        score_buf = J.Buffer(pScore, M * K * 2)
        value_buf = J.Buffer(pValue, K * N * 2)
        tmp = J.gpr("su32")
        zero = J.gpr("su32")
        zero[0] = 0
        tmp[0] = warp_id * (M * N * 4)
        J.s_add_u32(pOut[0], pOut[0], tmp)
        J.s_addc_u32(pOut[1], pOut[1], zero)
        out_buf   = J.Buffer(pOut, M * N * 4)

        # init vcache
        v_reg_caches = J.gpr(f'vf32x{N * K * 2 // 4 // 64}')
        voffset = J.gpr(lane_id << 4)
        offset12 = 0
        for i in range(K * N * 2 // 1024):
            value_buf.load_dwordx4(v_reg_caches[i * 4 + 0 : i * 4 + 3], voffset, 0, offset12=offset12)
            offset12 += 64*16
            if offset12 >= (1<<12):
                voffset[0] = voffset[0] + offset12
                offset12 = 0

        # init score
        acc_low = J.gpr(f'vf32x{4 * K // 16 // 2}')
        
        k_log2 = int(math.log2(K * 2))
        assert K * 2 == 2**k_log2
        voffset[0] = (key_load_col_id[0] << k_log2) + (key_load_row_id << 3)
        for i in range(K // 16):
            score_buf.load_dwordx2(acc_low[i * 4 // 2 : i * 4 // 2 + 1], voffset, 0, offset12=i * 32)

        J.s_waitcnt(mod='vmcnt(0)')

        vout = J.gpr(f'vf32x{N // 64 * 4 * 4}')
        def compute(is_first_iter):
            lds_base = warp_id * (16 * N * 2) + J.alloc_lds(32 * 1024)
            cur_v_write_lds = J.gpr(key_load_row_id * (N * 2) + key_load_col_id * (8 * 2) + lds_base)

            trans_low = J.gpr("su32")
            trans_low[0] = 0x01_00_05_04
            trans_high = J.gpr("su32")
            trans_high[0] = 0x03_02_07_06
            def transpose4x4_b16(src, dst):
                def trans2x2(s0, s1, d0, d1):
                    J.v_perm_b32(d0, s0, s1, trans_low)
                    J.v_perm_b32(d1, s0, s1, trans_high)
                trans2x2(src[0], src[2], dst[0], dst[2])
                trans2x2(src[1], src[3], dst[4], dst[6])
                trans2x2(src[4], src[6], dst[1], dst[3])
                trans2x2(src[5], src[7], dst[5], dst[7])

            for i in range(4):
                # write 4x128 elements to lds for each call
                J.ds_write_b128(cur_v_write_lds, v_reg_caches[i * 16 + 0: i * 16 + 3], mod='offset:0')
                J.ds_write_b128(cur_v_write_lds, v_reg_caches[i * 16 + 4: i * 16 + 7], mod='offset:1024')
                J.ds_write_b128(cur_v_write_lds, v_reg_caches[i * 16 + 8: i * 16 +11], mod='offset:2048')
                J.ds_write_b128(cur_v_write_lds, v_reg_caches[i * 16 +12: i * 16 +15], mod='offset:3072')
                # read 64 elements from lds for each iter
                v_curs = J.gpr('vf32x8')
                v_curs_tr = J.gpr('vf32x8')
                for j in range(N // 64):
                    # ds_read2_b64 v[74:77], v193 offset0:64 offset1:96
                    cur_v_read_lds = J.gpr(lds_base + key_load_row_id * (4 * N * 2) + (j * 64 * 2) + key_load_col_id * (4 * 2))
                    J.ds_read_b64(v_curs[0:1], cur_v_read_lds, mod=f'offset:0')
                    J.ds_read_b64(v_curs[2:3], cur_v_read_lds, mod=f'offset:{N * 2}')
                    J.ds_read_b64(v_curs[4:5], cur_v_read_lds, mod=f'offset:{N * 4}')
                    J.ds_read_b64(v_curs[6:7], cur_v_read_lds, mod=f'offset:{N * 6}')
                    J.s_waitcnt(mod='lgkmcnt(0)')
                    transpose4x4_b16(v_curs, v_curs_tr)
                    if is_first_iter and i == 0:
                        J.v_mfma_f32_16x16x16_bf16(vout[j * 16 + 0: j * 16 + 3], v_curs_tr[0:1], acc_low[2 * i : 2 * i + 1], 0)
                        J.v_mfma_f32_16x16x16_bf16(vout[j * 16 + 4: j * 16 + 7], v_curs_tr[2:3], acc_low[2 * i : 2 * i + 1], 0)
                        J.v_mfma_f32_16x16x16_bf16(vout[j * 16 + 8: j * 16 +11], v_curs_tr[4:5], acc_low[2 * i : 2 * i + 1], 0)
                        J.v_mfma_f32_16x16x16_bf16(vout[j * 16 +12: j * 16 +15], v_curs_tr[6:7], acc_low[2 * i : 2 * i + 1], 0)
                    else:
                        J.v_mfma_f32_16x16x16_bf16(vout[j * 16 + 0: j * 16 + 3], v_curs_tr[0:1], acc_low[2 * i : 2 * i + 1], vout[j * 16 + 0: j * 16 + 3])
                        J.v_mfma_f32_16x16x16_bf16(vout[j * 16 + 4: j * 16 + 7], v_curs_tr[2:3], acc_low[2 * i : 2 * i + 1], vout[j * 16 + 4: j * 16 + 7])
                        J.v_mfma_f32_16x16x16_bf16(vout[j * 16 + 8: j * 16 +11], v_curs_tr[4:5], acc_low[2 * i : 2 * i + 1], vout[j * 16 + 8: j * 16 +11])
                        J.v_mfma_f32_16x16x16_bf16(vout[j * 16 +12: j * 16 +15], v_curs_tr[6:7], acc_low[2 * i : 2 * i + 1], vout[j * 16 +12: j * 16 +15])

        compute(True)

        def transpose4x4_b32(src, dst):
            for i in range(4):
                for j in range(4):
                    J.v_mov_b32_e32(dst[j * 4 + i], src[i * 4 + j])

        vout_tr = J.gpr(f'vf32x{N // 64 * 4 * 4}')
        transpose4x4_b32(vout[0:15], vout_tr[0:15])
        transpose4x4_b32(vout[16:31], vout_tr[16:31])

        for i in range(2):
            vdata = vout_tr[i * 16 : i * 16 + 15]
            voffset[0] = i * 64 * 4 + key_load_col_id * (N * 4) + key_load_row_id * (16 * 4)
            out_buf.store_dwordx4(vdata[0:3], voffset, 0, offset12=0)
            out_buf.store_dwordx4(vdata[4:7], voffset, 0, offset12=16)
            out_buf.store_dwordx4(vdata[8:11], voffset, 0, offset12=32)
            out_buf.store_dwordx4(vdata[12:15], voffset, 0, offset12=48)

        # prepare stage(will remove/merge):
        #   key_load_row_id, key_load_col_id   // merge
        #   dram -> v_reg_caches               // remove
        #   dram -> acc_low                    // remove
        # load ds
        #   v_reg_caches -> ds
        #   ds -> v_curs
        # transpose
        #   v_curs -> v_curs_tr
        # mfma
        #   acc_low @ v_reg_caches
        # save                                 // remove

        return

    num_blocks = 1
    waves_per_block = 4
    # [16, 64*1]
    score = torch.randn([M, K], dtype=torch.bfloat16)
    # [64*1, 128]
    value = torch.randn([K, N], dtype=torch.bfloat16)
    # [16, 128]
    cur_out = torch.zeros([waves_per_block, M, N], dtype=torch.float32)
    debug_out = torch.zeros([1024], dtype=torch.int32)
    kernel([num_blocks],[64*waves_per_block], score.data_ptr(), value.data_ptr(), cur_out.data_ptr(), debug_out.data_ptr())
    def get_ref():
        s = score.to(torch.float32)
        v = value.to(torch.float32)
        ref = s @ v
        ref_out = torch.zeros_like(cur_out)
        ref_out[:,] = ref
        return ref_out
    ref_out = get_ref()
    if not torch.allclose(cur_out, ref_out, rtol=0.01):
        idx = torch.where(torch.abs(ref_out - cur_out) > 0.5)
        if len(idx[0]):
            print(f'idx = {idx}\nref_out={ref_out[idx]}\ncur={cur_out[idx]}')
        assert 0
    print('done')

if __name__ == "__main__":
    test_sv()

