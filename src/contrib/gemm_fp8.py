import pyhip

__all__ = [
    "gemm_fp8_8wave",
]


@pyhip.jit()
def gemm_fp8_8wave(J, wg_M, wg_N, N, K,
                   pA:"void*", # [M, K]  torch.float8_e4m3fn   row-major
                   pB:"void*", # [N, K]  torch.float8_e4m3fn   row-major
                   pC:"void*", # [M, N]  torch.bfloat16        row-major
                   M:"int"):
    """
    https://github.com/HazyResearch/HipKittens/blob/.../kernels/gemm/fp8fp32/FP8_8wave/8_wave.cu
    """

    A_dtype = "fp8"
    B_dtype = "fp8"
    C_dtype = "bf16"
    M01 = 8
    GroupNum = 8

    assert A_dtype == B_dtype

    # loader always load 128bytes (8 x DW4-lanes) along K dimension
    wg_K = J.div(128, J.sizeof(A_dtype))

    stride_k = K * J.sizeof_fp8
    stride_C = N * J.sizeof(C_dtype)

    blk_m, blk_n = J.tb_swizzle(J.blockIdx.x, M, wg_M, wg_N, N, M01, GroupNum)
    pA[:] += blk_m * (wg_M * stride_k)
    pB[:] += blk_n * (wg_N * stride_k)
    pC[:] += blk_m * (wg_M * stride_C) # + blk_n * (wg_N * J.sizeof(C_dtype)))

    M0 = J.gpr("su32", blk_m * wg_M)
    M1 = J.gpr("su32")
    J.s_min_u32(M1, M0 + wg_M, M)

    assert N % wg_N == 0
    num_warps = 8
    nbN = J.div(wg_N, 16)
    nbM = J.div(wg_M, 16)
    nbK = 2 # 2 MFMA 16x16 
    buff_a = J.Buffer(pA, (M1 - M0) * stride_k)
    buff_b = J.Buffer(pB, wg_N * stride_k)
    buff_c = J.Buffer(pC, (M1 - M0) * stride_C)

    WARPS_COL = 4
    WARPS_ROW = 2
    BLOCK_SIZE_ROW = wg_M
    BLOCK_SIZE_COL = wg_N
    BLOCK_K = 128
    HALF_BLOCK_SIZE_ROW = BLOCK_SIZE_ROW // 2
    HALF_BLOCK_SIZE_COL = BLOCK_SIZE_COL // 2
    
    lds_base = J.alloc_lds(HALF_BLOCK_SIZE_ROW * BLOCK_K * 4 + HALF_BLOCK_SIZE_COL * BLOCK_K * 4)
    ldsA = {}
    ldsB = {}
    lds = lds_base

    ldsA[0,0] = lds; lds += HALF_BLOCK_SIZE_ROW * BLOCK_K
    ldsA[0,1] = lds; lds += HALF_BLOCK_SIZE_ROW * BLOCK_K
    ldsA[1,0] = lds; lds += HALF_BLOCK_SIZE_ROW * BLOCK_K
    ldsA[1,1] = lds; lds += HALF_BLOCK_SIZE_ROW * BLOCK_K

    ldsB[0,0] = lds; lds += HALF_BLOCK_SIZE_COL * BLOCK_K
    ldsB[0,1] = lds; lds += HALF_BLOCK_SIZE_COL * BLOCK_K
    ldsB[1,0] = lds; lds += HALF_BLOCK_SIZE_COL * BLOCK_K
    ldsB[1,1] = lds; lds += HALF_BLOCK_SIZE_COL * BLOCK_K

    nrM = J.div(nbM, WARPS_ROW, 2) # 4
    nrN = J.div(nbN, WARPS_COL, 2) # 2
    nrK = nbK

    warp_m = J.gpr(J.warp_id[0] // WARPS_COL) # warp row: 0 to 1
    warp_n = J.gpr(J.warp_id[0] % WARPS_COL)  # warp col: 0 to 3

    use_pre_shuffle = False
    vm_load_a, vm_load_cnt_a, vm_offset_inc_a, ds_read_a = J.get_mfma_loader(use_pre_shuffle, num_warps, HALF_BLOCK_SIZE_ROW, BLOCK_K, stride_k, warp_m*64)
    vm_load_b, vm_load_cnt_b, vm_offset_inc_b, ds_read_b = J.get_mfma_loader(use_pre_shuffle, num_warps, HALF_BLOCK_SIZE_COL, BLOCK_K, stride_k, warp_n*32)

    # v_mfma_f32_16x16x128_f8f6f4: 
    mfma_A = J.gpr(nrM, 2, 4, "vfp8x4")            # 4x[16,128]
    mfma_B = J.gpr(2, nrN, 2, 4, "vfp8x4")            # 2x[16,128]
    mfma_C = J.gpr(4, nrM, nrN, 4, "vf32")      # 4x[4,2]x[16,16]
    mfma_C[...] = 0

    def mfma(c_index):
        b_index = c_index % 1
        for m in range(nrM):
            for n in range(nrN):
                J.v_mfma_f32_16x16x128_f8f6f4(mfma_C[c_index, m, n], mfma_B[b_index, n], mfma_A[m], mfma_C[c_index, m, n])
                yield 16


    # 第一步确保基础设施正确，使用最低效简单的pipeline，8-wave一起读入LDS，一起读出到寄存器，计算
    loop_cnt = J.div(K, wg_K)
    assert HALF_BLOCK_SIZE_ROW == HALF_BLOCK_SIZE_COL

    koffsets = J.gpr(2, "su32", 0, stride_k * HALF_BLOCK_SIZE_ROW)

    def vm_loadA(m, k):
        assert m in [0, 1]
        assert k in [0, 1]
        return vm_load_a(ldsA[m,k], buff_a, koffsets[k])

    def vm_loadB(m, k):
        assert m in [0, 1]
        assert k in [0, 1]
        return vm_load_b(ldsB[m,k], buff_b, koffsets[k])

    for k in range(loop_cnt):
        J.emit(vm_loadB(0,0))
        J.emit(vm_loadA(0,0))
        J.s_waitcnt(mod="vmcnt(0)")

        for m in range(nrM):
            ds_read_a(ldsA[0,0], mfma_A[m, 0], m, 0)
            ds_read_a(ldsA[0,0], mfma_A[m, 1], m, 1)
        for n in range(nrN):
            ds_read_b(ldsB[0,0], mfma_B[0, n, 0], n, 0)
            ds_read_b(ldsB[0,0], mfma_B[0, n, 1], n, 1)

        J.s_waitcnt(mod="lgkmcnt(0)")
        J.emit(mfma(0))

        J.emit(vm_loadB(0,1))
        J.s_waitcnt(mod="vmcnt(0)")
        for n in range(nrN):
            ds_read_b(ldsB[0,1], mfma_B[1, n, 0], n, 0)
            ds_read_b(ldsB[0,1], mfma_B[1, n, 1], n, 1)
        J.s_waitcnt(mod="lgkmcnt(0)")
        J.emit(mfma(1))

        J.emit(vm_loadA(0,1))
        J.s_waitcnt(mod="vmcnt(0)")
        for m in range(nrM):
            ds_read_a(ldsA[0,1], mfma_A[m, 0], m, 0)
            ds_read_a(ldsA[0,1], mfma_A[m, 1], m, 1)
        J.s_waitcnt(mod="lgkmcnt(0)")
        J.emit(mfma(2))
        J.emit(mfma(3))

        koffsets[0] += BLOCK_K
        koffsets[1] += BLOCK_K

    stride_c = N * J.sizeof_bf16
    vbf16 = J.gpr(4, "vbf16x2")
    col = J.lane_id // 16
    swap_12_col = (col & 1) * 2 + (col >> 1)

    vaddr0 = J.gpr(((J.lane_id % 16) + warp_m * 64)*stride_c + swap_12_col * J.sizeof_DW4 + warp_n * 32 * J.sizeof_bf16 + \
            blk_n * (wg_N * J.sizeof(C_dtype)))

    for cindex in range(4):
        cm = cindex // 2
        cn = cindex % 2
        vaddr = J.gpr("vu32", vaddr0[0] + cm*HALF_BLOCK_SIZE_ROW*stride_c)
        for m in range(nrM):
            for n in range(0, nrN, 2):
                J.uni_cvt_pk_bf16_f32(vbf16[0], mfma_C[cindex, m,n,0], mfma_C[cindex, m,n,1]) 
                J.uni_cvt_pk_bf16_f32(vbf16[1], mfma_C[cindex, m,n,2], mfma_C[cindex, m,n,3])
                J.uni_cvt_pk_bf16_f32(vbf16[2], mfma_C[cindex, m,n+1,0], mfma_C[cindex, m,n+1,1])
                J.uni_cvt_pk_bf16_f32(vbf16[3], mfma_C[cindex, m,n+1,2], mfma_C[cindex, m,n+1,3])
                #    a0    a1   a2   a3   | 01 23
                #    b0    b1   b2   b3   | 45 67
                #  v_permlane16_swap_b32(a, b)
                #    a0    b0   a2   b2   |
                #    a1    b1   a3   b3   |
                #
                # swap of row 1 & 2 are done by swapping lane-address 
                J.v_permlane16_swap_b32(vbf16[0], vbf16[2])
                J.v_permlane16_swap_b32(vbf16[1], vbf16[3])
                buff_c.store_dwordx4(vbf16, vaddr, 0, offset12 = n*4*J.sizeof_DW2 + cn*HALF_BLOCK_SIZE_COL*J.sizeof_bf16)
            vaddr[0] += 16*stride_c

@pyhip.jit()
def gemm_fp8_blockscale(J, wg_M, wg_N, N, K,
                     scale_BM,
                     scale_BN,
                     scale_BK,
                     pA:"void*", pAscale:"float*",
                     pB:"void*", pBscale:"float*",
                     pC:"void*", M:"int"):
    """
    A/B scale is float, so it's not MXFP8
        pA      : [M, K]  torch.float8_e4m3fn   row-major
        pB      : [N, K]  torch.float8_e4m3fn   pre-shuffled
        pAscale : [div_up(M,scale_BM), div_up(K, scale_BK) ] x [scale_BM, scale_BK] float
        pBscale : [div_up(N,scale_BN), div_up(K, scale_BK) ] x [scale_BN, scale_BK] float
    
    to use v_mfma_f32_16x16x128_f8f6f4(32 cycles), each MFMA's output must be scaled before accumulation
    so MFMA's accumulation is not used.

    It also implies that accumulation register needs to be VGPR instead of AccVGPR.

    this means we cannot use 4-wave for 256 x 256 block size, HipKitten's 8-wave methods allows
    all GPRs to be VGPR, thus it's a better way.



    """


    assert scale_BM == 1
    assert scale_BN == 128
    assert scale_BK == 128

    pass