import pyhip

__all__ = [
    "gemm_fp8_8wave",
]


def get_loader_row_major(J, num_warps, M, K, vm_stride, warp_row0):
    if 0:
        # 通过下面的可视化得知swizzle可以解决读入数据使用mfma格式ds_read时潜在的bank-conflict问题
        mfma_MN = 16
        mfma_K = 64//mfma_MN
        num_mfmas = K // (mfma_K * J.sizeof_DW4)
        J.show_mfma_in_lds(mfma_MN=mfma_MN, num_mfmas=num_mfmas)
        J.show_mfma_in_lds(mfma_MN=mfma_MN, num_mfmas=num_mfmas, swizzle_1=1)
        J.show_mfma_in_lds(mfma_MN=mfma_MN, num_mfmas=num_mfmas, swizzle_2=1)
        J.show_mfma_in_lds(mfma_MN=mfma_MN, num_mfmas=num_mfmas, swizzle_2=2)
        print(f"{wg_M=} {wg_K=} {M=} {K=}")
        assert 0

    # each wave load 8x128 bytes , 8 waves loads 64x128 bytes
    lds_stride = K
    num_lanes_per_row = J.div(lds_stride, J.sizeof_DW4)
    num_rows_per_load = J.div(64, num_lanes_per_row)
    warp_m_off = J.warp_id[0] * num_rows_per_load

    def swizzle(row, col):
        return (col ^ row) % num_lanes_per_row

    row = J.threadIdx.x // num_lanes_per_row
    col = J.threadIdx.x % num_lanes_per_row
    swizzle_col = swizzle(row, col)
    vmem_voff = J.gpr(row * vm_stride + swizzle_col * J.sizeof_DW4)
    lds_warp_off = J.gpr("su32", warp_m_off * lds_stride)

    vm_load_cnt = len(range(0, M, num_rows_per_load * num_warps))

    # WG-level coorperative loader (closure & generator):
    #   loads [M x K] bytes from specified buff+offset into lds
    def vm_load(lds_offset, buff, vm_offset):
        J.s_mov_b32("m0", lds_warp_off + lds_offset)
        voff = J.gpr("vu32", vmem_voff[0] + vm_offset)
        for m in range(0, M, 8*num_warps):
            yield 1
            buff.load_dwordx4(None, voff, 0, offset12=0)
            J.s_addk_i32("m0", 64*num_warps*J.sizeof_DW4)
            voff[0] += (num_rows_per_load * num_warps) * vm_stride

    # the [M x K] bytes LDS buffer is accessed by following closure
    # this closure using ds_read_b128 to load a 16x64 bytes MFMA data
    # thus LDS buffer's layout is blocked as [M//16,K//64] x [16,64]
    # each warp has its own row offsets specified in warp_row0
    assert M % 16 == 0
    assert K % 64 == 0

    col = J.lane_id // 16
    row = J.lane_id % 16
    num_regs_K = J.div(K, 64)
    voff = J.gpr(num_regs_K, "vu32")
    voff2 = J.gpr(num_regs_K, "vu32")
    for k in range(num_regs_K):
        # each ds_read_b128 took 4 x DW-lanes
        voff[k] = (row + warp_row0) * lds_stride + swizzle(row + warp_row0, col + k*4) * J.sizeof_DW4
        # ds_read_b128's imm offset is limited to 16bits, this additional voffset handles
        # the overflow case
        voff2[k] = voff[k] + 64*1024

    def ds_read_16x64(lds_offset, vdst, m, k):
        offset = lds_offset + m*16*lds_stride
        if offset >= 64*1024:
            voffset = voff2[k]
            offset -= 64*1024
        else:
            voffset = voff[k]
        J.ds_read_b128(vdst, voffset, mod=f"offset:{offset}")
    return vm_load, vm_load_cnt, ds_read_16x64


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
    num_warps = 4
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
    
    lds_base = J.alloc_lds(HALF_BLOCK_SIZE_ROW * BLOCK_K * 4 * 2)
    ldsA = {}
    ldsB = {}
    lds = lds_base

    ldsA[0,0] = lds; lds += HALF_BLOCK_SIZE_ROW * BLOCK_K
    ldsA[0,1] = lds; lds += HALF_BLOCK_SIZE_ROW * BLOCK_K
    ldsA[1,0] = lds; lds += HALF_BLOCK_SIZE_ROW * BLOCK_K
    ldsA[1,1] = lds; lds += HALF_BLOCK_SIZE_ROW * BLOCK_K

    ldsB[0,0] = lds; lds += HALF_BLOCK_SIZE_ROW * BLOCK_K
    ldsB[0,1] = lds; lds += HALF_BLOCK_SIZE_ROW * BLOCK_K
    ldsB[1,0] = lds; lds += HALF_BLOCK_SIZE_ROW * BLOCK_K
    ldsB[1,1] = lds; lds += HALF_BLOCK_SIZE_ROW * BLOCK_K

    nrM = J.div(nbM, WARPS_ROW, 2) # 4
    nrN = J.div(nbN, WARPS_COL, 2) # 2
    nrK = nbK

    warp_m = J.gpr(J.warp_id[0] // WARPS_COL) # warp row: 0 to 1
    warp_n = J.gpr(J.warp_id[0] % WARPS_COL)  # warp col: 0 to 3

    vm_load_a, vm_load_cnt_a, ds_read_a = get_loader_row_major(J, buff_a, nbM, nbK, stride_k, warp_m)
    vm_load_b, vm_load_cnt_b, ds_read_b = get_loader_row_major(J, buff_b, nbN, nbK, stride_k, warp_n)

    # v_mfma_f32_16x16x128_f8f6f4: 
    mfma_A = J.gpr(nrM, 8, "vfp8x4")            # 4x16x128
    mfma_B = J.gpr(nrN, 8, "vfp8x4")            # 2x16x128
    mfma_C = J.gpr(4, nrM, nrN, 4, "vf32")      # 

    # 第一步确保基础设施正确，使用最低效简单的pipeline，8-wave一起读入LDS，一起读出到寄存器，计算
    loop_cnt = J.div(K, wg_K)
    for k in range(loop_cnt):
        J.emit(vm_load_a(ldsA[0]))
        J.emit(vm_load_b(ldsB[0]))

        for m in range(nrM): ds_read_a(ldsA[0], mfma_A[0, m], m, 0)
        for n in range(nrN): ds_read_b(ldsB[0], mfma_B[0, n], n, 0)



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