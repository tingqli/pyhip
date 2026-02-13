import pyhip

__all__ = [
   "pre_shuffle", "gemm_kernel"
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

def get_loader(J, buff, use_pre_shuffle, nbM, nbK, stride_b, ibM0):
    num_warps = 4
    #stride_b = 0
    if use_pre_shuffle:
        stride_1kb = J.div(16*stride_b, 1024)
        warp_k = J.warp_id[0] % nbK
        warp_m = J.warp_id[0] // nbK
        vmem_warp_off = warp_m * (stride_1kb * 1024) + warp_k * 1024
        vmem_voff = J.gpr(J.lane_id[0] * J.sizeof_DW4 + vmem_warp_off)
        lds_warp_off = J.gpr("su32", warp_m * (nbK * 1024) + warp_k * 1024)

        voff = J.gpr(J.lane_id[0] * J.sizeof_DW4 + ibM0 * (nbK * 1024))
        voff2 = J.gpr("vu32", voff[0] + 64*1024)
        def ds_read_1kb(lds, vdst, m, k):
            offset = lds + m*(nbK * 1024) + k*1024
            if offset >= 64*1024:
                voffset = voff2
                offset -= 64*1024
            else:
                voffset = voff
            J.ds_read_b128(vdst, voffset, mod=f"offset:{offset}")

        vm_load_cnt = len(range(J.div(nbM, num_warps//nbK)))
        # return a loader constructor which can emit
        def vm_load(lds_offset):
            J.s_mov_b32("m0", lds_warp_off + lds_offset)
            voff = J.gpr("vu32", vmem_voff[0])

            for m in range(J.div(nbM, num_warps//nbK)):
                yield 1
                buff.load_dwordx4(None, voff, 0, offset12=0)
                J.s_addk_i32("m0", 256*J.sizeof_DW4)
                voff[0] += (num_warps//nbK)*(stride_1kb)*1024

            vmem_voff[0] += nbK * 1024
    else:
        if 0:
            # 通过下面的可视化得知swizzle可以解决读入数据使用mfma格式ds_read时潜在的bank-conflict问题
            J.show_mfma_in_lds(mfma_MN=16, num_mfmas=2)
            J.show_mfma_in_lds(mfma_MN=16, num_mfmas=2, swizzle_1=1)
            J.show_mfma_in_lds(mfma_MN=16, num_mfmas=2, swizzle_2=1)
            J.show_mfma_in_lds(mfma_MN=16, num_mfmas=2, swizzle_2=2)
            print(f"{wg_M=} {wg_K=} {nbM=} {nbK=}")
            assert 0
        # each wave load 8x128 bytes , 4 waves loads 32x128 bytes
        lds_stride_b = nbK * 4 * J.sizeof_DW4
        warp_m_off = J.warp_id[0] * 8

        def swizzle(row, col):
            #return col
            return (col ^ row) % 8
        col = J.threadIdx.x % 8
        row = J.threadIdx.x // 8
        swizzle_col = swizzle(row, col)
        vmem_voff = J.gpr(row * stride_b + swizzle_col * J.sizeof_DW4)
        lds_warp_off = J.gpr("su32", warp_m_off * lds_stride_b)

        vm_load_cnt = len(range(0, nbM * 16, 8*num_warps))

        def vm_load(lds_offset):
            J.s_mov_b32("m0", lds_warp_off + lds_offset)
            voff = J.gpr("vu32", vmem_voff[0])
            for m in range(0, nbM * 16, 8*num_warps):
                yield 1
                buff.load_dwordx4(None, voff, 0, offset12=0)
                J.s_addk_i32("m0", 256*J.sizeof_DW4)
                voff[0] += (8*num_warps) * stride_b

            vmem_voff[0] += nbK * 4 * J.sizeof_DW4

        col = J.lane_id // 16
        row = J.lane_id % 16
        swizzle_col = swizzle(row, col)
        voff = J.gpr(2, "vu32",
                     (row + ibM0*16) * lds_stride_b + swizzle(row, col) * J.sizeof_DW4,
                     (row + ibM0*16) * lds_stride_b + swizzle(row, col + 4) * J.sizeof_DW4)
        # ds_read_b128's offset is just 16bits 
        voff2 = J.gpr(2, "vu32", voff[0] + 64*1024, voff[1] + 64*1024)
        def ds_read_1kb(lds, vdst, m, k):
            offset = lds + m*16*lds_stride_b
            if offset >= 64*1024:
                voffset = voff2[k]
                offset -= 64*1024
            else:
                voffset = voff[k]
            J.ds_read_b128(vdst, voffset, mod=f"offset:{offset}")

    return vm_load, vm_load_cnt, ds_read_1kb


@pyhip.jit()
def gemm_kernel(J, wg_M, wg_N, N, K, use_pre_shuffle, pA:"void*", pB:"void*", pC:"void*", M:"int"):
    # 128 bytes
    wg_K = J.div(128, J.sizeof_bf16)

    A_dtype = "bf16"
    B_dtype = "bf16"
    C_dtype = "bf16"
    M01 = 8
    GroupNum = 8

    stride_k = K * J.sizeof_bf16
    stride_c = N * J.sizeof(C_dtype)

    blk_m, blk_n = J.tb_swizzle(J.blockIdx.x, M, wg_M, wg_N, N, M01, GroupNum)
    pA[:] += blk_m * (wg_M * K * J.sizeof(A_dtype))
    pB[:] += blk_n * (wg_N * K * J.sizeof(B_dtype))
    pC[:] += blk_m * (wg_M * stride_c)# + blk_n * (wg_N * J.sizeof(C_dtype)))

    M0 = J.gpr("su32", blk_m * wg_M)
    M1 = J.gpr("su32")
    J.s_min_u32(M1, M0 + wg_M, M)
    assert N % wg_N == 0
    assert K % wg_K == 0
    num_warps = 4
    nbN = J.div(wg_N, 16)
    nbM = J.div(wg_M, 16)
    nbK = 2 # 2 MFMA 16x16 
    buff_a = J.Buffer(pA, (M1 - M0) * stride_k)
    buff_b = J.Buffer(pB, wg_N * stride_k)
    buff_c = J.Buffer(pC, (M1 - M0) * stride_c)
    ldsA = [J.alloc_lds(nbM * nbK * 1024), J.alloc_lds(nbM * nbK * 1024)]
    ldsB = [J.alloc_lds(nbN * nbK * 1024), J.alloc_lds(nbN * nbK * 1024)]

    nrM = J.div(nbM, 2)
    nrN = J.div(nbN, 2)
    nrK = nbK

    warp_m = J.gpr((J.warp_id[0] // 2)*nrM)
    warp_n = J.gpr((J.warp_id[0] % 2)*nrN)

    vm_load_a, vm_load_cnt_a, ds_read_a = get_loader(J, buff_a, use_pre_shuffle, nbM, nbK, stride_k, warp_m)
    vm_load_b, vm_load_cnt_b, ds_read_b = get_loader(J, buff_b, use_pre_shuffle, nbN, nbK, stride_k, warp_n)

    print(f"============={nbM=}, {nbN=}, {nbK=} {nrM=} {nrN=} {nrK=}")

    mfma_A = J.gpr(2, nrM, 4, "vbf16x2")
    mfma_B = J.gpr(2, nrN, 4, "vbf16x2")
    mfma_C = J.gpr(nrM, nrN, 4, "af32")

    def mfma(reg_id):
        for m in range(nrM):
            for n in range(nrN):
                J.v_mfma_f32_16x16x32_bf16(mfma_C[m,n], mfma_B[reg_id, n], mfma_A[reg_id, m], mfma_C[m,n])
                yield 16

    J.emit(vm_load_a(ldsA[0]))
    J.emit(vm_load_b(ldsB[0]))

    J.emit(vm_load_a(ldsA[1]))
    J.emit(vm_load_b(ldsB[1]))
    mfma_C[...] = 0

    '''
    ab0: mfma ab0 | ds_read ab1; wait_lgkmcnt(0), barrier; vm-load a01
    ab1: mfma ab1 | vm-load b01; wait_vmcnt, barrier, ds_read ab2

    ab2: mfma ab2 | ds_read ab3; wait_lgkmcnt(0), barrier; vm-load  a23
    ab3: mfma ab3 | vm-load b23;  wait_vmcnt, barrier, ds_read ab0
    '''
    J.s_waitcnt(mod=f"vmcnt({vm_load_cnt_b + vm_load_cnt_a})")
    J.s_barrier()

    # ds_read ab0
    for m in range(nrM): ds_read_a(ldsA[0], mfma_A[0, m], m, 0)
    for n in range(nrN): ds_read_b(ldsB[0], mfma_B[0, n], n, 0)
    J.s_waitcnt(mod=f"lgkmcnt(0)")

    def loop_body(lds_id):
        # mfma ab0
        mfma_ab0 = mfma(0)

        # ds_read ab1
        for m in range(nrM):
            J.emit(mfma_ab0, 16)
            ds_read_a(ldsA[lds_id], mfma_A[1, m], m, 1)
        for n in range(nrN):
            J.emit(mfma_ab0, 16)
            ds_read_b(ldsB[lds_id], mfma_B[1, n], n, 1)

        J.emit(mfma_ab0, 64)

        J.s_waitcnt(mod=f"lgkmcnt(0)")
        J.emit(mfma_ab0, 16)
        J.s_barrier()

        mfma_ab1 = mfma(1)

        # vm-load a01
        vm_load = vm_load_a(ldsA[lds_id])
        J.emit([mfma_ab0, mfma_ab1], 16)
        J.emit(vm_load, 1) # first emit produce preparing instructions
        J.emit([mfma_ab0, mfma_ab1], 16)
        for _ in range(vm_load_cnt_a):
            J.emit(vm_load, 1)
            J.emit([mfma_ab0, mfma_ab1], 96)
        J.emit(vm_load)

        # vm-load b01
        vm_load = vm_load_b(ldsB[lds_id])
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
        J.s_waitcnt(mod=f"lgkmcnt(0)")

    if 0:
        for koff in range(J.div(K, wg_K)):
            loop_body(koff & 1)
    else:
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

    vdata = J.gpr(8, "vbf16x2")
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
            buff_c.store_dwordx4(vbf16, vaddr, 0, offset12=n*4*J.sizeof_DW2)
        vaddr[0] += 16*stride_c
