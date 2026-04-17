import pyhip
import torch



@pyhip.jit(with_debug_log=False)
def quant_mfma_output(J,
                        mode,
                        input:"void*", 
                        output:"void*",
                        o_scales:"void*"):
    num_warps = 8
    WARPS_COL = 4
    WARPS_ROW = 2
    warp_m = J.gpr(J.warp_id[0] // WARPS_COL) # warp row: 0 to 1
    warp_n = J.gpr(J.warp_id[0] % WARPS_COL)  # warp col: 0 to 3

    lds = J.LDSTensor([2,2,16], base=1000)

    print(f"{lds[0,0]=}")
    print(f"{lds[0]=}")
    print(f"{lds[0,1]=}")
    print(f"{lds[1,0]=}")
    print(f"{lds[1]=}")
    print(f"{lds[1,1]=}")
    print(f"{lds[0,2]=}")
    assert 0

    wg_M = 256
    wg_N = 256
    nbN = J.div(wg_N, 16)
    nbM = J.div(wg_M, 16)
    nrM = J.div(nbM, WARPS_ROW, 2) # 4
    nrN = J.div(nbN, WARPS_COL, 2) # 2

    mfma_C = J.gpr(4, nrM, nrN, 4, "vf32")

    row = J.lane_id % 16
    col = J.lane_id // 16
    i_stride = wg_N * J.sizeof_f32
    o_stride = wg_N * J.sizeof_s8

    if mode >= 100:
        mfma_C[...] = 0

        for cindex in range(4):
            cm = cindex // 2
            cn = cindex % 2
            for m in range(nrM):
                for n in range(nrN):
                    vaddr = J.gpr("vu32", col * J.sizeof_DW4 + row * i_stride)
                    vaddr += warp_n * (32 * J.sizeof_f32)
                    vaddr += warp_m * (64 * i_stride)
                    vaddr += cm * 128 * i_stride
                    vaddr += cn * 128 * J.sizeof_f32
                    vaddr += m * 16 * i_stride
                    vaddr += n * 16 * J.sizeof_f32
                    J.global_load_dwordx4(mfma_C[cindex, m, n], vaddr, input)

        J.s_waitcnt(mod="vmcnt(0)")
        J.s_barrier()

    if (mode % 100) == 1:
        # allocate 2 rows of data 2*16*256*fp32 = 32k
        lds_stride = wg_N * J.sizeof_f32 + 16
        lds = J.alloc_lds(2*16*lds_stride)

        vflow = J.gpr("vf32", -128.0)
        vfhigh = J.gpr("vf32", 127.0)
        vnz_guard = J.gpr("vf32", 1e-6)

        for cm in range(2):
            for m in range(nrM):
                assert nrN == 2
                # write w0123 & w4567 into two rows
                vaddr = J.gpr("vu32", col * J.sizeof_DW4 + row * lds_stride + lds)
                vaddr += J.gpr(warp_m * (16 * lds_stride))
                vaddr += J.gpr(warp_n * (32 * J.sizeof_f32))
                J.ds_write_b128(vaddr, mfma_C[cm*2+0, m, 0], mod=f"offset:{0}")
                J.ds_write_b128(vaddr, mfma_C[cm*2+0, m, 1], mod=f"offset:{16*J.sizeof_f32}")

                J.ds_write_b128(vaddr, mfma_C[cm*2+1, m, 0], mod=f"offset:{128*J.sizeof_f32 + 0}")
                J.ds_write_b128(vaddr, mfma_C[cm*2+1, m, 1], mod=f"offset:{128*J.sizeof_f32 + 16*J.sizeof_f32}")

                # each warp process 4-rows, totally 32-rows
                # warp_m : 0 first 16-rows, 1 second 16-rows
                # warp_n : 0/1/2/3 * 4 rows
                # one row is 256-fp32s, quantized into 256-s8

                vmem_offset = J.gpr("vu32", J.lane_id * J.sizeof_DW)
                vmem_offset += cm*128*o_stride + m*16*o_stride
                vmem_offset += warp_m * (64*o_stride)
                vmem_offset += warp_n * (4*o_stride)

                J.s_waitcnt(mod="lgkmcnt(0)")
                J.s_barrier()

                vf32 = J.gpr(4, 4, "vf32")
                vaddr = J.gpr("vu32", J.lane_id * J.sizeof_DW4 + lds)
                vaddr += J.gpr(warp_n * (4*lds_stride))
                vaddr += J.gpr(warp_m * (16*lds_stride))
                for r in range(4):
                    J.ds_read_b128(vf32[r], vaddr, mod=f"offset:{r*lds_stride}")
                
                J.s_waitcnt(mod="lgkmcnt(0)")

                for r in range(4):
                    vmax = J.gpr("vf32")
                    smax = J.gpr("sf32")
                    J.v_max3_f32(vmax, abs(vf32[r,0]), abs(vf32[r,1]), abs(vf32[r,2]))
                    J.v_max3_f32(vmax, vmax, abs(vf32[r,3]), vnz_guard)
                    J.s_nop(mod="2")

                    J.v_max_f32_dpp(vmax, vmax, vmax, mod="row_shr:8 bound_ctrl:0")
                    J.s_nop(mod="2")
                    J.v_max_f32_dpp(vmax, vmax, vmax, mod="row_shr:4 bound_ctrl:0")
                    J.s_nop(mod="2")
                    J.v_max_f32_dpp(vmax, vmax, vmax, mod="row_shr:2 bound_ctrl:0")
                    J.s_nop(mod="2")
                    J.v_max_f32_dpp(vmax, vmax, vmax, mod="row_shr:1 bound_ctrl:0")
                    J.s_nop(mod="2")
                    J.v_max_f32_dpp(vmax, vmax, vmax, mod="row_bcast:15 bound_ctrl:0")
                    J.s_nop(mod="2")
                    J.v_max_f32_dpp(vmax, vmax, vmax, mod="row_bcast:31 bound_ctrl:0")
                    J.s_nop(2)
                    J.v_readlane_b32(smax, vmax, 63)
                    J.s_nop(2)
                    vmax[0] = smax

                    row_scale = J.gpr("vf32", vmax * (1/127))
                    inv_row_scale = J.gpr("vf32")
                    J.v_rcp_f32(inv_row_scale, row_scale)

                    if 1:
                        J.SetMask("exec", 1)
                        #J.ds_write_b32()
                        vaddr = J.gpr("vu32", warp_m*(64*J.sizeof_f32) + warp_n*(4*J.sizeof_f32))
                        #vdata = J.gpr("vf32", smax)
                        J.global_store_dword(vaddr, vf32[r,1], o_scales, mod=f"offset:{(cm*128+m*16+r)*J.sizeof_f32}")
                        J.SetMask("exec", -1)

                    # quantize current row
                    vf32[r, 0] *= inv_row_scale
                    vf32[r, 1] *= inv_row_scale
                    vf32[r, 2] *= inv_row_scale
                    vf32[r, 3] *= inv_row_scale
                    J.v_rndne_f32(vf32[r, 0], vf32[r, 0])
                    J.v_rndne_f32(vf32[r, 1], vf32[r, 1])
                    J.v_rndne_f32(vf32[r, 2], vf32[r, 2])
                    J.v_rndne_f32(vf32[r, 3], vf32[r, 3])
                    J.v_med3_f32(vf32[r, 0], vf32[r, 0], vflow, vfhigh)
                    J.v_med3_f32(vf32[r, 1], vf32[r, 1], vflow, vfhigh)
                    J.v_med3_f32(vf32[r, 2], vf32[r, 2], vflow, vfhigh)
                    J.v_med3_f32(vf32[r, 3], vf32[r, 3], vflow, vfhigh)
                    vs8x4 = J.gpr("vu32")
                    J.v_cvt_i32_f32_sdwa(vs8x4, vf32[r, 0], mod="dst_sel:BYTE_0 dst_unused:UNUSED_PRESERVE src0_sel:DWORD")
                    J.s_nop(1)
                    J.v_cvt_i32_f32_sdwa(vs8x4, vf32[r, 1], mod="dst_sel:BYTE_1 dst_unused:UNUSED_PRESERVE src0_sel:DWORD")
                    J.s_nop(1)
                    J.v_cvt_i32_f32_sdwa(vs8x4, vf32[r, 2], mod="dst_sel:BYTE_2 dst_unused:UNUSED_PRESERVE src0_sel:DWORD")
                    J.s_nop(1)
                    J.v_cvt_i32_f32_sdwa(vs8x4, vf32[r, 3], mod="dst_sel:BYTE_3 dst_unused:UNUSED_PRESERVE src0_sel:DWORD")
                    J.s_nop(1)
                    J.global_store_dword(vmem_offset, vs8x4, output, mod=f"offset:{r*o_stride}")

                J.s_barrier()
        return

    # 4 warp-n, each warp-n 256 row-max
    lds_base = J.alloc_lds(256*4*J.sizeof_f32)
    vflow = J.gpr("vf32", -128.0)
    vfhigh = J.gpr("vf32", 127.0)
    vnz_guard = J.gpr("vf32", 1e-6)

    for cm in range(2):
        for m in range(nrM):
            assert nrN == 2
            vmax0 = J.gpr("vf32")
            vmax1 = J.gpr("vf32")
            vmax2 = J.gpr("vf32")
            vmax3 = J.gpr("vf32")
            J.v_max3_f32(vmax0, abs(mfma_C[cm*2+0, m, 0, 0]), abs(mfma_C[cm*2+0, m, 0, 1]), abs(mfma_C[cm*2+0, m, 0, 2]))
            J.v_max3_f32(vmax1, abs(mfma_C[cm*2+0, m, 1, 0]), abs(mfma_C[cm*2+0, m, 1, 1]), abs(mfma_C[cm*2+0, m, 1, 2]))
            J.v_max3_f32(vmax2, abs(mfma_C[cm*2+1, m, 0, 0]), abs(mfma_C[cm*2+1, m, 0, 1]), abs(mfma_C[cm*2+1, m, 0, 2]))
            J.v_max3_f32(vmax3, abs(mfma_C[cm*2+1, m, 1, 0]), abs(mfma_C[cm*2+1, m, 1, 1]), abs(mfma_C[cm*2+1, m, 1, 2]))

            J.v_max3_f32(vmax0, vmax0, abs(mfma_C[cm*2+0, m, 0, 3]), abs(mfma_C[cm*2+0, m, 1, 3]))
            J.v_max3_f32(vmax1, vmax1, abs(mfma_C[cm*2+1, m, 0, 3]), abs(mfma_C[cm*2+1, m, 1, 3]))

            J.v_max3_f32(vmax0, vmax0, vmax1, vmax2)
            J.v_max3_f32(vmax0, vmax0, vmax3, vnz_guard)

            # vmax0 in MFMA-16x16-layout, now get per-row max
            # permlane16 (0 1 2 3), (0 1 2 3) => (0 0 2 2) (1 1 3 3)
            # permlane32 (m01 m01, m23 m23), (m01 m01, m23 m23) => (m01,m01,m01,m01) (m23,m23,m23,m23)
            vmax1[0] = vmax0
            J.s_nop(2)
            J.v_permlane16_swap_b32(vmax0, vmax1)
            J.v_max_f32(vmax0, vmax0, vmax1) # (m01 m01, m23 m23)
            vmax2[0] = vmax0
            J.s_nop(2)
            J.v_permlane32_swap_b32(vmax0, vmax2)
            J.v_max_f32(vmax0, vmax0, vmax2) # (m0123 m0123, m0123 m0123)

            # now we need to get max of warp rows: w0123, w4567
            # each warp store max value of 16-rows to ds
            ds_vaddr = J.gpr("vu32", lds_base + (cm*128 + m*16)*J.sizeof_f32 + J.lane_id * J.sizeof_f32)
            ds_vaddr += warp_m * (64*J.sizeof_f32)
            ds_vaddr += warp_n * (256*J.sizeof_f32)
            J.SetMask("exec", 0xFFFF) # only write 16-elements
            J.ds_write_b32(ds_vaddr, vmax0)
            J.SetMask("exec", -1)

    J.s_waitcnt(mod="lgkmcnt(0)")
    J.s_barrier()

    vmax0 = J.gpr(2,nrM,"vf32")
    vmax1 = J.gpr(2,nrM,"vf32")
    vmax2 = J.gpr(2,nrM,"vf32")
    vmax3 = J.gpr(2,nrM,"vf32")
    for cm in range(2):
        for m in range(nrM):
            ds_vaddr = J.gpr("vu32", lds_base + (cm*128 + m*16)*J.sizeof_f32 + (J.lane_id % 16) * J.sizeof_f32)
            ds_vaddr += warp_m * (64*J.sizeof_f32)
            J.ds_read_b32(vmax0[cm,m], ds_vaddr, mod=f"offset:{0*256*J.sizeof_f32}")
            J.ds_read_b32(vmax1[cm,m], ds_vaddr, mod=f"offset:{1*256*J.sizeof_f32}")
            J.ds_read_b32(vmax2[cm,m], ds_vaddr, mod=f"offset:{2*256*J.sizeof_f32}")
            J.ds_read_b32(vmax3[cm,m], ds_vaddr, mod=f"offset:{3*256*J.sizeof_f32}")
    J.s_waitcnt(mod="lgkmcnt(0)")

    for cm in range(2):
        for m in range(nrM):
            # read-back max values of other 4 warps in same row
            J.v_max3_f32(vmax0[cm,m], vmax0[cm,m], vmax1[cm,m], vmax2[cm,m])
            J.v_max_f32(vmax0[cm,m], vmax0[cm,m], vmax3[cm,m])

            # now quantize mfma_C[cm*2+01, m, 01, 0123]  4x2x2=16 regs
            row_scale = J.gpr("vf32", vmax0[cm,m] * (1/127))
            inv_row_scale = J.gpr("vf32")
            J.v_rcp_f32(inv_row_scale[0], row_scale)
            #inv_row_scale[1] = inv_row_scale[0]

            vmem_offset = J.gpr("vu32", (J.lane_id % 16)*o_stride + (J.lane_id // 16) * J.sizeof_DW2)
            vmem_offset += cm*128*o_stride + m*16*o_stride
            vmem_offset += warp_m * (64*o_stride)
            vmem_offset += warp_n * (32*J.sizeof_s8)

            for cn in range(2):
                c_index = cm*2 + cn
                for k in range(0,4):
                    #J.v_pk_mul_f32(mfma_C[c_index, m, 0, k:k+1], mfma_C[c_index, m, 0, k:k+1], inv_row_scale)
                    #J.v_pk_mul_f32(mfma_C[c_index, m, 1, k:k+1], mfma_C[c_index, m, 1, k:k+1], inv_row_scale)
                    mfma_C[c_index, m, 0, k] *= inv_row_scale[0]
                    mfma_C[c_index, m, 1, k] *= inv_row_scale[0]
                for k in range(4):
                    J.v_rndne_f32(mfma_C[c_index, m, 0, k], mfma_C[c_index, m, 0, k])
                    J.v_rndne_f32(mfma_C[c_index, m, 1, k], mfma_C[c_index, m, 1, k])
                for k in range(4):
                    J.v_med3_f32(mfma_C[c_index, m, 0, k], mfma_C[c_index, m, 0, k], vflow, vfhigh)
                    J.v_med3_f32(mfma_C[c_index, m, 1, k], mfma_C[c_index, m, 1, k], vflow, vfhigh)

                vs8x4 = J.gpr(2, "vu32")
                for k in range(4):
                    J.v_cvt_i32_f32_sdwa(vs8x4[0], mfma_C[c_index, m, 0, k], mod=f"dst_sel:BYTE_{k} dst_unused:UNUSED_PRESERVE src0_sel:DWORD")
                    J.v_cvt_i32_f32_sdwa(vs8x4[1], mfma_C[c_index, m, 1, k], mod=f"dst_sel:BYTE_{k} dst_unused:UNUSED_PRESERVE src0_sel:DWORD")
                J.s_nop(1)
                #J.global_store_dwordx2(vmem_offset, vs8x4, output, mod=f"offset:{cn*128*J.sizeof_s8}")












pyhip.set_device()

input = torch.randn([256, 256], dtype=torch.float32)
input[...] = 0
#input[...] = -2
for r in range(256):
    input[r,0] = 2
    input[r,1] = -2

output = torch.ones([256, 256], dtype=torch.int8)
o_scale = torch.ones([256], dtype=torch.float32)

method = 2
quant_mfma_output([1],[8*64], 100+method, input, output, o_scale)

print(input)
for r in range(256):
    print(r, output[r,:16].tolist())
print(o_scale)


pyhip.run_perftest(quant_mfma_output, [1896*16],[8*64], method, input, output, o_scale, num_verbose=1)
#pyhip.run_perftest(quant_mfma_output, [256],[8*64], method, input, output, o_scale, num_verbose=1)