import pyhip

__all__ = [
    "gemm_a4w4_kernel", "pre_shuffle"
]

def pre_shuffle(x, mfma_MN):
    M, K = x.shape
    K_bytes = K * x.itemsize
    sizeof_DW4 = 16
    mfma_K_lanes = 64 // mfma_MN
    mfma_K_L = sizeof_DW4//x.itemsize
    mfma_K = mfma_K_lanes * mfma_K_L 

    assert M % mfma_MN == 0
    mfma_K_bytes = mfma_K_lanes * sizeof_DW4
    assert K_bytes % mfma_K_bytes == 0

    x = x.reshape(M//mfma_MN, mfma_MN, K//mfma_K, mfma_K_lanes, mfma_K_L)
    x = x.permute(0,2,3,1,4)
    return x.contiguous()

@pyhip.jit()
def gemm_a4w4_kernel(J, wg_M, wg_N, N, K, a_preshuffle, b_preshuffle,
                     pA:"void*", pAscale:"void*",
                     pB:"void*", pBscale:"void*",
                     pC:"void*", M:"int"):
    # K in unit of fp4x2 128 bytes
    wg_K = J.div(128, J.sizeof_fp4x2)

    J.show_gemm_buf(mfma_MN = 16, n_mfma_K = 4, wave_CNT = [2,2], wave_Size = [128, 128])

    A_dtype = "fp4x2"
    B_dtype = "fp4x2"
    C_dtype = "bf16"
    M01 = 8
    GroupNum = 8

    stride_k = K * J.sizeof_fp4x2
    stride_C = N * J.sizeof(C_dtype)

    blk_m, blk_n = J.tb_swizzle(J.blockIdx.x, M, wg_M, wg_N, N, M01, GroupNum)
    pA[:] += blk_m * (wg_M * stride_k)
    pB[:] += blk_n * (wg_N * stride_k)
    pC[:] += blk_m * (wg_M * stride_C) # + blk_n * (wg_N * J.sizeof(C_dtype)))

    stride_scale = J.div(K, 128) * J.sizeof_DW * 64
    pAscale[:] += blk_m * J.div(wg_M, 32) * stride_scale
    pBscale[:] += blk_n * J.div(wg_N, 32) * stride_scale

    pAscale[:] += (J.warp_id[0] // 2) * (J.div(wg_M//2, 32) * stride_scale)
    pBscale[:] += (J.warp_id[0] % 2) * (J.div(wg_N//2, 32) * stride_scale)

    sbuff_a = J.Buffer(pAscale, J.div(wg_M//2, 32) * stride_scale)
    sbuff_b = J.Buffer(pBscale, J.div(wg_N//2, 32) * stride_scale)

    M0 = J.gpr("su32", blk_m * wg_M)
    M1 = J.gpr("su32")
    J.s_min_u32(M1, M0 + wg_M, M)

    assert N % wg_N == 0
    num_warps = 4
    nbN = J.div(wg_N, 16)
    nbM = J.div(wg_M, 16)
    nbK = 2 # 2 MFMA 16x16 
    buff_a = J.Buffer(pA, (M1 - M0) * stride_k)
    buff_b = J.Buffer(pB, wg_N * stride_k)
    buff_c = J.Buffer(pC, (M1 - M0) * stride_C)
    ldsA = [J.alloc_lds(nbM * nbK * 1024), J.alloc_lds(nbM * nbK * 1024)]
    ldsB = [J.alloc_lds(nbN * nbK * 1024), J.alloc_lds(nbN * nbK * 1024)]

    nrM = J.div(nbM, 2)
    nrN = J.div(nbN, 2)
    nrK = nbK

    warp_m = J.gpr((J.warp_id[0] // 2)*nrM)
    warp_n = J.gpr((J.warp_id[0] % 2)*nrN)

    #vm_load_a, vm_load_cnt_a, ds_read_a = get_loader(J, buff_a, a_preshuffle, nbM, nbK, stride_k, warp_m)
    #vm_load_b, vm_load_cnt_b, ds_read_b = get_loader(J, buff_b, b_preshuffle, nbN, nbK, stride_k, warp_n)

    num_warps = 4
    vm_load_a, vm_load_cnt_a, vm_offset_inc_a, ds_read_a = J.get_mfma_loader(use_pre_shuffle, num_warps, wg_M, 128, stride_k, warp_m*16)
    vm_load_b, vm_load_cnt_b, vm_offset_inc_b, ds_read_b = J.get_mfma_loader(use_pre_shuffle, num_warps, wg_N, 128, stride_k, warp_n*16)

    print(f"============={nbM=}, {nbN=}, {nbK=} {nrM=} {nrN=} {nrK=}")

    mfma_A = J.gpr(2, nrM, 4, "vbf16x2")
    mfma_B = J.gpr(2, nrN, 4, "vbf16x2")
    mfma_C = J.gpr(nrM, nrN, 4, "af32")
    
    mfma_Ascale = J.gpr(2, J.div(nrM, 2), "vu32") # 4
    mfma_Bscale = J.gpr(2, J.div(nrN, 2), "vu32") # 4

    def mfma(reg_id, lds_id):
        # lds_id : scale register is grouped by lds_id
        # src0: Matrix A scale {OP_SEL_HI [0], OP_SEL[0]} defines which part of scale is used by the Matrix A of MFMA instruction.
        # src1: Matrix B scale {OP_SEL_HI [1], OP_SEL[1]} defines which part of scale is used by the Matrix B of MFMA instruction.
        for m in range(nrM):
            for n in range(nrN):
                sel_scale_B = (n & 1) + (reg_id & 1)*2
                sel_scale_A = (m & 1) + (reg_id & 1)*2
                mod = f"op_sel:[{sel_scale_B & 1}, {sel_scale_A & 1},0] op_sel_hi:[{sel_scale_B//2}, {sel_scale_A//2}, 0] cbsz:4 blgp:4"

                # J.v_mfma_f32_16x16x32_bf16(mfma_C[m,n], mfma_B[reg_id, n], mfma_A[reg_id, m], mfma_C[m,n])
                J.v_mfma_scale_f32_16x16x128_f8f6f4(mfma_C[m,n], mfma_B[reg_id, n], mfma_A[reg_id, m], mfma_C[m,n],
                                                    mfma_Bscale[lds_id, n//2],
                                                    mfma_Ascale[lds_id, m//2],
                                                    mod=mod)
                yield 16

    koffset_a = J.gpr("su32", 0)
    koffset_b = J.gpr("su32", 0)
    J.emit(vm_load_a(ldsA[0], buff_a, koffset_a))
    J.emit(vm_load_b(ldsB[0], buff_b, koffset_b))
    koffset_a[0] += vm_offset_inc_a
    koffset_b[0] += vm_offset_inc_b

    # load scales
    vaddr_scale = J.gpr("vu32", J.lane_id[0] * J.sizeof_DW)

    def load_next_scales(index):
        vaddr = J.gpr("vu32", vaddr_scale[0])
        for ii in range(J.div(nrM, 2)):
            sbuff_a.load_dword(mfma_Ascale[index & 1, ii], vaddr, 0)
            vaddr[0] += stride_scale
            yield 1
        vaddr = J.gpr("vu32", vaddr_scale[0])
        for ii in range(J.div(nrN, 2)):
            sbuff_b.load_dword(mfma_Bscale[index & 1, ii], vaddr, 0)
            vaddr[0] += stride_scale
            yield 1
        vaddr_scale[0] += J.sizeof_DW * 64

    num_scale_loads = J.div(nrM, 2) + J.div(nrN, 2)

    J.emit(load_next_scales(0))

    J.emit(vm_load_a(ldsA[1], buff_a, koffset_a))
    J.emit(vm_load_b(ldsB[1], buff_b, koffset_b))
    koffset_a[0] += vm_offset_inc_a
    koffset_b[0] += vm_offset_inc_b

    mfma_C[...] = 0

    '''
    ab0: mfma ab0 | ds_read ab1; wait_lgkmcnt(0), barrier; vm-load a01 | load mfma_Ascale[1]
    ab1: mfma ab1 | vm-load b01; wait_vmcnt, barrier, ds_read ab2      | load mfma_Bscale[1]

    ab2: mfma ab2 | ds_read ab3; wait_lgkmcnt(0), barrier; vm-load  a23 | load mfma_Ascale[0]
    ab3: mfma ab3 | vm-load b23;  wait_vmcnt, barrier, ds_read ab0      | load mfma_Bscale[0]
    '''
    J.s_waitcnt(mod=f"vmcnt({vm_load_cnt_b + vm_load_cnt_a})")
    J.s_barrier()

    # ds_read ab0
    for m in range(nrM): ds_read_a(ldsA[0], mfma_A[0, m], m, 0)
    for n in range(nrN): ds_read_b(ldsB[0], mfma_B[0, n], n, 0)
    J.s_waitcnt(mod=f"lgkmcnt(0)")

    def loop_body(lds_id):
        # mfma ab0
        mfma_ab0 = mfma(0, lds_id)

        load_s = load_next_scales(lds_id + 1)
        # ds_read ab1
        for m in range(nrM):
            J.emit(mfma_ab0, 16)
            ds_read_a(ldsA[lds_id], mfma_A[1, m], m, 1)
            J.emit(load_s, 1)
        for n in range(nrN):
            J.emit(mfma_ab0, 16)
            ds_read_b(ldsB[lds_id], mfma_B[1, n], n, 1)
            J.emit(load_s, 1)

        for ii in range(4):
            J.emit(mfma_ab0, 16)
            J.emit(load_s, 1)

        J.emit(load_s)

        J.s_waitcnt(mod=f"lgkmcnt(0)")
        J.emit(mfma_ab0, 16)
        J.s_barrier()

        mfma_ab1 = mfma(1, lds_id)

        # vm-load a01
        vm_load = vm_load_a(ldsA[lds_id], buff_a, koffset_a)
        J.emit([mfma_ab0, mfma_ab1], 16)
        J.emit(vm_load, 1) # first emit produce preparing instructions
        J.emit([mfma_ab0, mfma_ab1], 16)
        for _ in range(vm_load_cnt_a):
            J.emit(vm_load, 1)
            J.emit([mfma_ab0, mfma_ab1], 96)
        J.emit(vm_load)

        # vm-load b01
        vm_load = vm_load_b(ldsB[lds_id], buff_b, koffset_b)
        J.emit([mfma_ab0, mfma_ab1], 16)
        J.emit(vm_load, 1) # first emit produce preparing instructions
        J.emit([mfma_ab0, mfma_ab1], 16)
        for _ in range(vm_load_cnt_b - 4):
            J.emit(vm_load, 1)
            J.emit([mfma_ab0, mfma_ab1], 96)

        J.emit(mfma_ab0) # emit all MFMA using AB register0 (since ds_read will override it)

        # wait vm-load a23/b23 to finish
        J.s_waitcnt(mod=f"vmcnt({vm_load_cnt_b + vm_load_cnt_a - 4})")
        J.s_barrier()

        J.emit(vm_load, 1)
        # ds_read ab2
        for m in range(nrM):
            J.emit(mfma_ab1, 16)
            ds_read_a(ldsA[(lds_id + 1)&1], mfma_A[0, m], m, 0)

        J.emit(vm_load, 1)
        for n in range(nrN):
            J.emit(mfma_ab1, 16)
            ds_read_b(ldsB[(lds_id + 1)&1], mfma_B[0, n], n, 0)

        J.emit(vm_load, 1)
        J.emit(mfma_ab1, 96)
        J.emit(vm_load, 1)
        J.emit(mfma_ab1)
        koffset_a[0] += vm_offset_inc_a
        koffset_b[0] += vm_offset_inc_b

        J.s_waitcnt(mod=f"lgkmcnt(0)")

    koff = J.gpr("su32", 0)
    loop_cnt = K // (2*wg_K)
    with J.While(koff[0] < loop_cnt):
        loop_body(0)
        loop_body(1)
        koff[0] += 1

    if K % (2*wg_K):
        loop_body(0)

    for lds in ldsA: J.free_lds(lds) 
    for lds in ldsB: J.free_lds(lds)

    stride_c = N * J.sizeof_bf16
    vbf16 = J.gpr(4, "vbf16x2")
    col = J.lane_id // 16
    swap_12_col = (col & 1) * 2 + (col >> 1)

    vaddr = J.gpr(((J.lane_id % 16) + warp_m * 16)*stride_c + swap_12_col * J.sizeof_DW4 + warp_n * 4 * J.sizeof_DW2 + \
            blk_n * (wg_N * J.sizeof(C_dtype)))

    for m in range(nrM):
        for n in range(0, nrN, 2):
            J.uni_cvt_pk_bf16_f32(vbf16[0], mfma_C[m,n,0], mfma_C[m,n,1]) 
            J.uni_cvt_pk_bf16_f32(vbf16[1], mfma_C[m,n,2], mfma_C[m,n,3])
            J.uni_cvt_pk_bf16_f32(vbf16[2], mfma_C[m,n+1,0], mfma_C[m,n+1,1])
            J.uni_cvt_pk_bf16_f32(vbf16[3], mfma_C[m,n+1,2], mfma_C[m,n+1,3])
            #    a0    a1   a2   a3   | 01 23
            #    b0    b1   b2   b3   | 45 67
            #  v_permlane16_swap_b32(a, b)
            #    a0    b0   a2   b2   |
            #    a1    b1   a3   b3   |
            #
            # swap of row 1 & 2 are done by swapping lane-address 
            J.v_permlane16_swap_b32(vbf16[0], vbf16[2])
            J.v_permlane16_swap_b32(vbf16[1], vbf16[3])
            buff_c.store_dwordx4(vbf16, vaddr, 0, offset12 = n*4*J.sizeof_DW2)
        vaddr[0] += 16*stride_c
