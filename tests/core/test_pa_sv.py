import pyhip
import torch
import math
torch.set_printoptions(linewidth=300)
torch.set_default_device('cuda')
torch.manual_seed(0)

def test_sv():
    M = 16
    K = 64
    N = 128
    N_ITER = 64
    # pS: bf16[16, 64*1], pV: bf16[64*1, 128], pSV: f32[4, 16, 128]
    @pyhip.jit()
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
        pOut[:] = pOut[:] + warp_id * (M * N * 4)
        out_buf   = J.Buffer(pOut, M * N * 4)

        # init vcache
        v_reg_caches = J.gpr(N * K * 2 // 64//4//4, 4, "bf16x2")

        voffset = J.gpr(lane_id * 16)
        offset12 = 0
        for i in range(N * K * 2 // 64//4//4):
            value_buf.load_dwordx4(v_reg_caches[i], voffset, 0, offset12=offset12)
            offset12 += 64*16
            if offset12 >= (1<<12):
                voffset[0] = voffset[0] + offset12
                offset12 = 0

        # init score
        acc_low = J.gpr(K//16, 2, "bf16x2")

        voffset[0] = (key_load_col_id * (2*K)) + (key_load_row_id * 8)
        for i in range(K//16):
            score_buf.load_dwordx2(acc_low[i], voffset, 0, offset12=i * 32)

        J.s_waitcnt(mod='vmcnt(0)')

        # vout = J.gpr(f'vf32x{N // 64 * 4 * 4}')
        vout = J.gpr(N//16, 4, "f32")

        def compute(is_first_iter):
            lds_base = warp_id * (16 * N * 2) + J.alloc_lds(32 * 1024)
            cur_v_write_lds = J.gpr(key_load_row_id * (N * 2) + key_load_col_id * (8 * 2) + lds_base)

            for i in range(4):
                # write 4x128 elements to lds for each call
                J.ds_write_b128(cur_v_write_lds, v_reg_caches[i*4+ 0], mod='offset:0')
                J.ds_write_b128(cur_v_write_lds, v_reg_caches[i*4+ 1], mod='offset:1024')
                J.ds_write_b128(cur_v_write_lds, v_reg_caches[i*4+ 2], mod='offset:2048')
                J.ds_write_b128(cur_v_write_lds, v_reg_caches[i*4+ 3], mod='offset:3072')
                # read 64 elements from lds for each iter
                v_curs = J.gpr(4, 2, "bf16x2")
                v_curs_tr = J.gpr(4, 2, "bf16x2")
                #v_curs_tr = J.gpr('vf32x8')
                for j in range(N // 64):
                    # ds_read2_b64 v[74:77], v193 offset0:64 offset1:96
                    cur_v_read_lds = J.gpr(lds_base + key_load_row_id * (4 * N * 2) + (j * 64 * 2) + key_load_col_id * (4 * 2))
                    J.ds_read_b64(v_curs[0], cur_v_read_lds, mod=f'offset:0')
                    J.ds_read_b64(v_curs[1], cur_v_read_lds, mod=f'offset:{N * 2}')
                    J.ds_read_b64(v_curs[2], cur_v_read_lds, mod=f'offset:{N * 4}')
                    J.ds_read_b64(v_curs[3], cur_v_read_lds, mod=f'offset:{N * 6}')
                    J.s_waitcnt(mod='lgkmcnt(0)')
                    J.transpose_per_lane(4, 4, 2, v_curs[...], v_curs_tr[...])
                    if is_first_iter and i == 0:
                        J.v_mfma_f32_16x16x16_bf16(vout[j*4 + 0], v_curs_tr[0], acc_low[i], 0)
                        J.v_mfma_f32_16x16x16_bf16(vout[j*4 + 1], v_curs_tr[1], acc_low[i], 0)
                        J.v_mfma_f32_16x16x16_bf16(vout[j*4 + 2], v_curs_tr[2], acc_low[i], 0)
                        J.v_mfma_f32_16x16x16_bf16(vout[j*4 + 3], v_curs_tr[3], acc_low[i], 0)
                    else:
                        J.v_mfma_f32_16x16x16_bf16(vout[j*4 + 0], v_curs_tr[0], acc_low[i], vout[j*4 + 0])
                        J.v_mfma_f32_16x16x16_bf16(vout[j*4 + 1], v_curs_tr[1], acc_low[i], vout[j*4 + 1])
                        J.v_mfma_f32_16x16x16_bf16(vout[j*4 + 2], v_curs_tr[2], acc_low[i], vout[j*4 + 2])
                        J.v_mfma_f32_16x16x16_bf16(vout[j*4 + 3], v_curs_tr[3], acc_low[i], vout[j*4 + 3])

        compute(True)

        # vout 
        # vout_tr = J.gpr(f'vf32x{N // 64 * 4 * 4}')
        vout_tr = J.gpr(N//16, 4, "fp32")
        J.transpose_per_lane(4, 4, 4, vout[0:3], vout_tr[0:3])
        J.transpose_per_lane(4, 4, 4, vout[4:7], vout_tr[4:7])

        for i in range(2):
            #vdata = vout_tr[i * 16 : i * 16 + 15]
            voffset[0] = i * 64 * 4 + key_load_col_id * (N * 4) + key_load_row_id * (16 * 4)
            out_buf.store_dwordx4(vout_tr[i*4 + 0], voffset, 0, offset12=0)
            out_buf.store_dwordx4(vout_tr[i*4 + 1], voffset, 0, offset12=16)
            out_buf.store_dwordx4(vout_tr[i*4 + 2], voffset, 0, offset12=32)
            out_buf.store_dwordx4(vout_tr[i*4 + 3], voffset, 0, offset12=48)

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

