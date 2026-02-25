import pyhip
import torch

__all__ = [
    "gemm_fp8_8wave",
]

@pyhip.jit(with_debug_log = False)
def gemm_fp8_8wave(J, bpreshuffle,
                   use_f32_blockscales_128, # scale_BM,scale_BN,scale_BK = 1,128,128 
                   wg_M, wg_N, N, K, 
                   pA:"void*", # [M, K]  torch.float8_e4m3fn   row-major
                   pB:"void*", # [N, K]  torch.float8_e4m3fn   row-major
                   pC:"void*", # [M, N]  torch.bfloat16        row-major
                   pScaleA:"float*", #    [div_up(M,scale_BM), div_up(K, scale_BK)]
                                     # or [div_up(K, scale_BK), div_up(M,scale_BM)] if bpreshuffle
                   pScaleB:"float*", # [div_up(N,scale_BN), div_up(K, scale_BK) ]
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
    Mc = J.gpr("su32", M1 - M0)

    assert N % wg_N == 0
    num_warps = 8
    nbN = J.div(wg_N, 16)
    nbM = J.div(wg_M, 16)
    nbK = 2 # 2 MFMA 16x16 
    buff_a = J.Buffer(pA, Mc * stride_k)
    buff_b = J.Buffer(pB, wg_N * stride_k)
    buff_c = J.Buffer(pC, Mc * stride_C)

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
    vm_load_b, vm_load_cnt_b, vm_offset_inc_b, ds_read_b = J.get_mfma_loader(bpreshuffle, num_warps, HALF_BLOCK_SIZE_COL, BLOCK_K, stride_k, warp_n*32)

    if use_f32_blockscales_128:
        assert bpreshuffle == True, "exepct scaleA in [k,m] layout"
        scale_BM, scale_BN, scale_BK = 1,128,128 
        # tic-toc LDS buffer for 256 per-token per-k-128 scales
        # 1-warp is enough to load this buffer
        lds_scaleA = [J.alloc_lds(num_warps * 64 * J.sizeof_f32),
                      J.alloc_lds(num_warps * 64 * J.sizeof_f32)]
        # if pScaleA in [m,k] layout
        # pScaleA[:] += blk_m * (wg_M * J.div(K, scale_BK) * J.sizeof_f32)
        # buff_sa = J.Buffer(pScaleA, (M1 - M0) * J.div(K, scale_BK) * J.sizeof_f32)
        # scaleA : [div_up(K, scale_BK), div_up(M,scale_BM)]
        #   
        # pScaleA[:] += blk_m * (wg_M * J.sizeof_f32)
        buff_sa = J.Buffer(pScaleA, M[0] * J.div(K, scale_BK) * J.sizeof_f32)
        voffset_scaleA = J.gpr(J.threadIdx.x[0] * J.sizeof_f32 + blk_m * (wg_M * J.sizeof_f32))
        assert wg_M <= num_warps * 64
        # vm_load_scaleA(lds_scaleA[toc])
        # ds_read scaleA must be in MFMA_16x4 format
        # ds_read scaleB broad-cast in to 16x4 too
        def vm_load_scaleA(lds, bk):
            # bk: index of k block with size of 128
            # use execmask to ensure same impact on vmcnt for all warps
            J.s_mov_b32("m0", lds + J.warp_id[0]*(64*J.sizeof_f32))
            voff = J.gpr("vu32", voffset_scaleA[0] + J.gpr("su32", M[0]*(bk*J.sizeof_f32)))
            #with J.ExecMask(J.threadIdx.x[0] < wg_M, early_skip=False):
            buff_sa.load_dword(None, voff, 0)

        # scale of B(weights) are very small, can be all loaded into LDS
        lds_scaleB = J.alloc_lds(J.div(K, scale_BK) * J.div(wg_N, scale_BN) * J.sizeof_f32)
        pScaleB[:] += blk_n * (J.div(wg_N, scale_BN) * J.div(K, scale_BK) * J.sizeof_f32)

        J.wg_load_lds(lds_scaleB, pScaleB, J.div(wg_N, scale_BN) * J.div(K, scale_BK) * J.sizeof_f32,
                      num_warps, wait_barrier = True)

        num_scaleB = J.div(wg_N, scale_BN)
        mfma_scaleA = J.gpr(nrM, "vf32")
        mfma_scaleB = J.gpr(num_scaleB, 2, "vf32")
        vaddr_scaleA = J.gpr("vu32", (J.lane_id[0] % 16)*J.sizeof_f32 + warp_m * (16*nrM * J.sizeof_f32))
        def ds_read_scaleA(lds, m0):
            assert m0 in [0, 1]
            vaddr = J.gpr("vu32", vaddr_scaleA[0] + lds)
            for m in range(nrM):
                off = (m0*HALF_BLOCK_SIZE_ROW + m*16)*J.sizeof_f32
                J.ds_read_b32(mfma_scaleA[m], vaddr, mod=f"offset:{off}")

        vaddr_scaleB = J.gpr(num_scaleB, "vu32")
        for i in range(num_scaleB):
            vaddr_scaleB[i] = lds_scaleB + i*J.div(K, scale_BK)*J.sizeof_f32
        def ds_read_scaleB(bk):
            # k0: in unit of scale_BK
            # n0: in unit of scale_BN
            # all warps share the same scaleB
            assert scale_BN >= nrN * 16 * 4
            if isinstance(bk, int):
                off = bk * J.sizeof_f32
                for i in range(num_scaleB):
                    J.ds_read_b32(mfma_scaleB[i,0], vaddr_scaleB[i], mod=f"offset:{off}")
                    #J.ds_read_b32(mfma_scaleB[i,1], vaddr_scaleB[i], mod=f"offset:{off}")
            else:
                for i in range(num_scaleB):
                    J.ds_read_b32(mfma_scaleB[i,0], vaddr_scaleB[i] + bk * J.sizeof_f32)


    # v_mfma_f32_16x16x128_f8f6f4: 
    mfma_A = J.gpr(nrM, 2, 4, "vfp8x4")            # 4x[16,128]
    mfma_B = J.gpr(2, nrN, 2, 4, "vfp8x4")            # 2x[16,128]
    mfma_C = J.gpr(4, nrM, nrN, 4, "vf32")      # 4x[4,2]x[16,16]

    if use_f32_blockscales_128:
        MFMA_FIFO_CNT = nrM * nrN
        # circular fifo buffer for post-processing
        # prepare scales for next round
        mfma_fifo_scale = J.gpr(2, nrM, "vf32")
        mfma_fifo = J.gpr(MFMA_FIFO_CNT, 4, "vf32")
        mfma_fifo_c_index = None
        mfma_fifo_read_id = 0
        mfma_fifo_write_id = 0

        def mfma(c_index):
            nonlocal mfma_fifo_scale, mfma_fifo, mfma_fifo_c_index, mfma_fifo_read_id, mfma_fifo_write_id
            b_index = c_index % 2

            mfma_fifo_next_read_id = mfma_fifo_write_id
            fifo_read_id = mfma_fifo_read_id
            for m in range(nrM):
                for n in range(nrN):
                    if n == 0:
                        mfma_fifo_scale[c_index%2, m] = mfma_scaleA[m] * mfma_scaleB[b_index,0]
                    if mfma_fifo_c_index is not None:
                        J.v_fmac_f32(mfma_C[mfma_fifo_c_index, m, n, 0], mfma_fifo[fifo_read_id % MFMA_FIFO_CNT, 0], mfma_fifo_scale[mfma_fifo_c_index % 2,m])
                        J.v_fmac_f32(mfma_C[mfma_fifo_c_index, m, n, 1], mfma_fifo[fifo_read_id % MFMA_FIFO_CNT, 1], mfma_fifo_scale[mfma_fifo_c_index % 2,m])
                        J.v_fmac_f32(mfma_C[mfma_fifo_c_index, m, n, 2], mfma_fifo[fifo_read_id % MFMA_FIFO_CNT, 2], mfma_fifo_scale[mfma_fifo_c_index % 2,m])
                        J.v_fmac_f32(mfma_C[mfma_fifo_c_index, m, n, 3], mfma_fifo[fifo_read_id % MFMA_FIFO_CNT, 3], mfma_fifo_scale[mfma_fifo_c_index % 2,m])
                        fifo_read_id += 1

                    J.v_mfma_f32_16x16x128_f8f6f4(mfma_fifo[mfma_fifo_write_id % MFMA_FIFO_CNT], mfma_B[b_index, n], mfma_A[m], 0)
                    mfma_fifo_write_id += 1
                    yield 16
            mfma_fifo_read_id = mfma_fifo_next_read_id
            mfma_fifo_c_index = c_index
        
        def mfma_tail():
            fifo_read_id = mfma_fifo_read_id
            for m in range(nrM):
                for n in range(nrN):
                    if mfma_fifo_c_index is not None:
                        J.v_fmac_f32(mfma_C[mfma_fifo_c_index, m, n, 0], mfma_fifo[fifo_read_id % MFMA_FIFO_CNT, 0], mfma_fifo_scale[mfma_fifo_c_index % 2,m])
                        J.v_fmac_f32(mfma_C[mfma_fifo_c_index, m, n, 1], mfma_fifo[fifo_read_id % MFMA_FIFO_CNT, 1], mfma_fifo_scale[mfma_fifo_c_index % 2,m])
                        J.v_fmac_f32(mfma_C[mfma_fifo_c_index, m, n, 2], mfma_fifo[fifo_read_id % MFMA_FIFO_CNT, 2], mfma_fifo_scale[mfma_fifo_c_index % 2,m])
                        J.v_fmac_f32(mfma_C[mfma_fifo_c_index, m, n, 3], mfma_fifo[fifo_read_id % MFMA_FIFO_CNT, 3], mfma_fifo_scale[mfma_fifo_c_index % 2,m])
                        fifo_read_id += 1

    else:
        def mfma(c_index):
            b_index = c_index % 2
            for m in range(nrM):
                for n in range(nrN):
                    J.v_mfma_f32_16x16x128_f8f6f4(mfma_C[c_index, m, n], mfma_B[b_index, n], mfma_A[m], mfma_C[c_index, m, n])
                    yield 16
        def mfma_tail():
            pass

    loop_cnt = J.div(K, wg_K)
    assert HALF_BLOCK_SIZE_ROW == HALF_BLOCK_SIZE_COL

    a_moffsets = J.gpr(2, "su32", 0, stride_k * HALF_BLOCK_SIZE_ROW)
    if bpreshuffle:
        b_moffsets = J.gpr(2, "su32", 0, stride_k * HALF_BLOCK_SIZE_ROW)

    def step_k():
        a_moffsets[0] += vm_offset_inc_a
        a_moffsets[1] += vm_offset_inc_a
        if bpreshuffle:
            b_moffsets[0] += vm_offset_inc_b
            b_moffsets[1] += vm_offset_inc_b

    def vm_loadA(k, m):
        assert m in [0, 1]
        assert k in [0, 1]
        return vm_load_a(ldsA[k,m], buff_a, a_moffsets[m])

    def vm_loadB(k, m):
        assert m in [0, 1]
        assert k in [0, 1]
        if bpreshuffle:
            return vm_load_b(ldsB[k,m], buff_b, b_moffsets[m])
        else:
            return vm_load_b(ldsB[k,m], buff_b, a_moffsets[m])

    def ds_readA(k, m):
        for i in range(nrM):
            ds_read_a(ldsA[k,m], mfma_A[i, 0], i, 0)
            ds_read_a(ldsA[k,m], mfma_A[i, 1], i, 1)

    def ds_readB(k, m):
        for i in range(nrN):
            ds_read_b(ldsB[k,m], mfma_B[m, i, 0], i, 0)
            ds_read_b(ldsB[k,m], mfma_B[m, i, 1], i, 1)

    #print(nrM, nrN); assert 0
    if 1:
        # 8-wave pipeline invented by HipKittens
        tic = 0
        toc = 1
        if use_f32_blockscales_128: vm_load_scaleA(lds_scaleA[tic], 0)
        J.emit(vm_loadB(tic,0))
        J.emit(vm_loadA(tic,0))
        J.emit(vm_loadB(tic,1))
        J.emit(vm_loadA(tic,1))

        with J.If(warp_m[0] == 1):
            J.s_barrier()

        mfma_C[...] = 0

        J.s_waitcnt(mod=f"vmcnt({vm_load_cnt_a + vm_load_cnt_b})"); J.s_barrier()

        step_k()

        if use_f32_blockscales_128:
            vm_load_scaleA(lds_scaleA[toc], 1)
            vm_load_cnt_scaleA = 1
        else:
            vm_load_cnt_scaleA = 0
        J.emit(vm_loadA(toc,0))
        J.emit(vm_loadB(toc,0))
        J.emit(vm_loadB(toc,1))

        J.s_waitcnt(mod=f"vmcnt({vm_load_cnt_a + vm_load_cnt_b*2 + vm_load_cnt_scaleA})"); J.s_barrier()

        def loop_body(k, loop_cnt):
            nonlocal tic, toc
            ds_readB(tic, 0)    # lgkmcnt += nrN*2 (2*2)
            ds_readA(tic, 0)    # lgkmcnt += nrM*2 (4*2)

            if use_f32_blockscales_128:
                ds_read_scaleA(lds_scaleA[tic], 0)
                ds_read_scaleB(k)

            J.emit(vm_loadA(toc,1))
            step_k()
            J.s_waitcnt(mod=f"lgkmcnt(0)"); J.s_barrier()

            J.s_waitcnt(mod="lgkmcnt(0)"); J.s_setprio(1)
            J.emit(mfma(0))
            J.s_setprio(0); J.s_barrier()
            #===============================================================
            # after this s_barrier, lgkmcnt(8) ensures all 8-waves has finished
            # accessing B[tic,0], so next vm_load can overwrite A[toc,0],B[toc,0],B[toc,1],A[toc,1]

            ds_readB(tic, 1)
            J.emit(vm_loadA(tic,0))                         # vm_load_cnt_a
            J.s_barrier()

            J.s_waitcnt(mod="lgkmcnt(0)"); J.s_setprio(1)
            J.emit(mfma(1))
            J.s_setprio(0); J.s_barrier()

            ds_readA(tic, 1)
            if use_f32_blockscales_128:
                ds_read_scaleA(lds_scaleA[tic], 1)
            J.emit(vm_loadB(tic,0))                         # vm_load_cnt_b
            J.s_barrier()

            J.s_waitcnt(mod="lgkmcnt(0)"); J.s_setprio(1)
            J.emit(mfma(2))
            J.s_setprio(0); J.s_barrier()

            J.emit(vm_loadB(tic,1))                         # vm_load_cnt_b
            if use_f32_blockscales_128:
                vm_load_scaleA(lds_scaleA[tic], k+2)
            J.s_waitcnt(mod=f"vmcnt({vm_load_cnt_a + vm_load_cnt_b*2 + vm_load_cnt_scaleA})"); J.s_barrier()

            J.s_setprio(1)
            J.emit(mfma(3))
            J.s_setprio(0); J.s_barrier()
            #===============================================================
            # after this s_barrier, we have all A[toc] & B[toc] loaded in LDS
            # so in next iteration, we can ds_read A[tic] & B[tic] w/o waitting for any vmcnt

            tic ^= 1
            toc ^= 1

        if 1:
            for k in range(loop_cnt):
                loop_body(k, loop_cnt)
        else:
            k = J.gpr("su32", 0)
            with J.While(k[0] < loop_cnt):
                loop_body(k, loop_cnt)
                k[0] += 1
                loop_body(k, loop_cnt)
                k[0] += 1
            if loop_cnt % 2:
                loop_body(k, loop_cnt)

        mfma_tail()
        J.s_waitcnt(mod="vmcnt(0)")

        with J.If(warp_m[0] == 0):
            J.s_barrier()

    else:
        # 第一步确保基础设施正确，使用最低效简单的pipeline，8-wave一起读入LDS，一起读出到寄存器，计算
        # naive pipeline, for debugging basic building blocks
        mfma_C[...] = 0

        J.debug_setup((J.warp_id[0] == 0) & (J.blockIdx.x[0] == 0))
        for k in range(loop_cnt):
            J.emit(vm_loadB(0,0))
            J.emit(vm_loadA(0,0))
            if use_f32_blockscales_128: vm_load_scaleA(lds_scaleA[0], k)
            J.s_waitcnt(mod="vmcnt(0)"); J.s_barrier()

            ds_readA(0,0)
            ds_readB(0,0)
            if use_f32_blockscales_128:
                ds_read_scaleA(lds_scaleA[0], 0)
                ds_read_scaleB(k)
            J.s_waitcnt(mod="lgkmcnt(0)"); J.s_barrier()
            J.emit(mfma(0))

            #J.debug_log(mfma_A[0,0], torch.float8_e4m3fn, "4h.16v.16h")
            #J.debug_log(mfma_A[0,1], torch.float8_e4m3fn, "4h.16v.16h")
            #J.s_endpgm()

            J.emit(vm_loadB(0,1))
            J.s_waitcnt(mod="vmcnt(0)"); J.s_barrier()

            ds_readB(0,1)
            J.s_waitcnt(mod="lgkmcnt(0)"); J.s_barrier()
            J.emit(mfma(1))

            #J.debug_log(mfma_B[1,0,0], torch.float8_e4m3fn, "4h.16v.16h")
            #J.debug_log(mfma_B[1,0,1], torch.float8_e4m3fn, "4h.16v.16h")
            #J.s_endpgm()

            J.emit(vm_loadA(0,1))
            J.s_waitcnt(mod="vmcnt(0)"); J.s_barrier()

            ds_readA(0,1)
            if use_f32_blockscales_128:
                ds_read_scaleA(lds_scaleA[0], 1)
            J.s_waitcnt(mod="lgkmcnt(0)"); J.s_barrier()

            #J.debug_log(mfma_A[0,0], torch.float8_e4m3fn, "4h.16v.16h")
            #J.debug_log(mfma_A[0,1], torch.float8_e4m3fn, "4h.16v.16h")
            #J.s_endpgm()

            J.emit(mfma(2))
            J.emit(mfma(3))

            step_k()

    #J.debug_log(mfma_C[1,0,0], torch.float, "4h.16v.4h")
    #J.s_endpgm()

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
