import pyhip

from .common.loaders import get_mfma_loader, tb_swizzle

__all__ = [
   "gemm_kernel"
]

@pyhip.jit()
def gemm_kernel(J, wg_M, wg_N, N, K, use_pre_shuffle, pA:"void*", pB:"void*", pC:"void*", M:"int"):
    # 128 bytes
    # wg_K = J.div(128, J.sizeof_bf16)
    wg_K = 64

    A_dtype = "bf16"
    B_dtype = "bf16"
    C_dtype = "bf16"
    M01 = 8
    GroupNum = 8

    stride_k = K * J.sizeof_bf16
    stride_c = N * J.sizeof(C_dtype)

    blk_m, blk_n = tb_swizzle(J, J.blockIdx.x, M, wg_M, wg_N, N, M01, GroupNum)
    pA[:] += blk_m * (wg_M * K * J.sizeof(A_dtype))
    pB[:] += blk_n * (wg_N * K * J.sizeof(B_dtype))
    pC[:] += blk_m * (wg_M * stride_c)# + blk_n * (wg_N * J.sizeof(C_dtype)))

    M0 = J.gpr("su32", blk_m * wg_M)
    M1 = J.gpr("su32")
    J.s_min_u32(M1, M0 + wg_M, M)
    assert N % wg_N == 0
    assert K % wg_K == 0
    num_warps = 4
    nbN = J.div(wg_N, 16)# the N tile number per WG, each N tile 16n
    nbM = J.div(wg_M, 16)# the M tile number per WG, each M tile 16m
    nbK = 2 # 2 MFMA 16x16 # the k tile number per WG, each K tile 32k
    buff_a = J.Buffer(pA, (M1 - M0) * stride_k)
    buff_b = J.Buffer(pB, wg_N * stride_k)
    buff_c = J.Buffer(pC, (M1 - M0) * stride_c)
    # ldsA = [J.alloc_lds(nbM * nbK * 1024), J.alloc_lds(nbM * nbK * 1024)]
    # ldsB = [J.alloc_lds(nbN * nbK * 1024), J.alloc_lds(nbN * nbK * 1024)]
    
    ldsA0 = [J.alloc_lds(nbM//2 * nbK * 1024), J.alloc_lds(nbM//2 * nbK * 1024)]
    ldsB0 = [J.alloc_lds(nbN//2 * nbK * 1024), J.alloc_lds(nbN//2 * nbK * 1024)]
    ldsA1 = [J.alloc_lds(nbM//2 * nbK * 1024), J.alloc_lds(nbM//2 * nbK * 1024)]
    ldsB1 = [J.alloc_lds(nbN//2 * nbK * 1024), J.alloc_lds(nbN//2 * nbK * 1024)]
    
    nrM = J.div(nbM, 2) # the N tile number per warp, each N tile 16n
    nrN = J.div(nbN, 2) # the N tile number per warp, each M tile 16n
    nrK = nbK           # the k tile number per warp, each K tile 32k

    warp_m = J.gpr((J.warp_id[0] // 2)*nrM) # M start tile idx of the warp inside the one WG
    warp_n = J.gpr((J.warp_id[0] % 2)*nrN)  # N start tile idx of the warp inside the one WG

    num_warps = 4
    vm_load_a_idx, vm_load_cnt_a, vm_offset_inc_a, ds_read_a = get_mfma_loader(J, use_pre_shuffle, num_warps, wg_M, 128, stride_k, warp_m//2*16)
    vm_load_b_idx, vm_load_cnt_b, vm_offset_inc_b, ds_read_b = get_mfma_loader(J, use_pre_shuffle, num_warps, wg_N, 128, stride_k, warp_n//2*16)

    print(f"============={nbM=}, {nbN=}, {nbK=} {nrM=} {nrN=} {nrK=}")

    #[2, 2] is for [a/b_idx, bk_idx]
    mfma_A = J.gpr(2, 2, nrM//2, 4, "vbf16x2") 
    mfma_B = J.gpr(2, 2, nrN//2, 4, "vbf16x2")
    #[2, 2] is for [a_idx, b_idx]
    mfma_C = J.gpr(2, 2, nrM//2, nrN//2, 4, "af32")

    def mfma(a_idx, b_idx, reg_id):
        for m in range(nrM//2):
            for n in range(nrN//2):
                J.v_mfma_f32_16x16x32_bf16(mfma_C[a_idx, b_idx, m,n], mfma_B[b_idx, reg_id, n], mfma_A[a_idx, reg_id, m], mfma_C[a_idx, b_idx, m,n])
                yield 16
    if 1:
        koffset_a = J.gpr("su32", 0)
        koffset_b = J.gpr("su32", 0)
        mfma_C[...] = 0
        ####################################prelog:
        # ping prefetch
        #AC B0
        # J.emit(vm_load_b_idx(ldsB0[0], buff_b, koffset_b, 0))
        # #AC A0
        # J.emit(vm_load_a_idx(ldsA0[0], buff_a, koffset_a, 0))
        # #AC A1
        # J.emit(vm_load_a_idx(ldsA1[0], buff_a, koffset_a, 1))
        # #AC B1
        # J.emit(vm_load_b_idx(ldsB1[0], buff_b, koffset_b, 1))
        
        
        #AC A0
        J.emit(vm_load_a_idx(ldsA0[0], buff_a, koffset_a, 0))
        #AC B0
        J.emit(vm_load_b_idx(ldsB0[0], buff_b, koffset_b, 0))
        #AC A1
        J.emit(vm_load_a_idx(ldsA1[0], buff_a, koffset_a, 1))
        #AC B1
        J.emit(vm_load_b_idx(ldsB1[0], buff_b, koffset_b, 1))
        koffset_a[0] += vm_offset_inc_a
        koffset_b[0] += vm_offset_inc_b
    
        # pong prefetch
        #AC B0
        # J.emit(vm_load_b_idx(ldsB0[1], buff_b, koffset_b, 0))
        # #AC A0
        # J.emit(vm_load_a_idx(ldsA0[1], buff_a, koffset_a, 0))
        # #AC A1
        # J.emit(vm_load_a_idx(ldsA1[1], buff_a, koffset_a, 1))
        # #AC B1
        # J.emit(vm_load_b_idx(ldsB1[1], buff_b, koffset_b, 1))
        
        J.emit(vm_load_a_idx(ldsA0[1], buff_a, koffset_a, 0))
        #AC B0
        J.emit(vm_load_b_idx(ldsB0[1], buff_b, koffset_b, 0))
        #AC A1
        J.emit(vm_load_a_idx(ldsA1[1], buff_a, koffset_a, 1))
        #AC B1
        J.emit(vm_load_b_idx(ldsB1[1], buff_b, koffset_b, 1))
        koffset_a[0] += vm_offset_inc_a
        koffset_b[0] += vm_offset_inc_b
        
        J.s_waitcnt(mod=f"vmcnt({0})")
        # J.s_waitcnt(mod=f"lgkmcnt(0)")
        J.s_barrier()
        
        for n in range(nrN//2): ds_read_b(ldsB0[0], mfma_B[0, 0, n], n, 0)
        for n in range(nrN//2): ds_read_b(ldsB0[0], mfma_B[0, 1, n], n, 1)


        for m in range(nrM//2): ds_read_a(ldsA0[0], mfma_A[0, 0, m], m, 0)
        for m in range(nrM//2): ds_read_a(ldsA0[0], mfma_A[0, 1, m], m, 1)

        mfma_a0b0_k0=mfma(0, 0, 0)
        mfma_a0b1_k0=mfma(0, 1, 0)
        mfma_a1b0_k0=mfma(1, 0, 0)
        mfma_a1b1_k0=mfma(1, 1, 0)
    
        mfma_a0b0_k1=mfma(0, 0, 1)
        mfma_a0b1_k1=mfma(0, 1, 1)
        mfma_a1b0_k1=mfma(1, 0, 1)
        mfma_a1b1_k1=mfma(1, 1, 1)
        ####################################main loop:
        # def loop_body(idx):
        #     cur_lds = idx % 2
        #     next_lds = (idx + 1) % 2
        #     # part 0:
        #     J.s_waitcnt(mod=f"vmcnt({0})")
        #     J.s_barrier()
        #     J.s_waitcnt(mod=f"lgkmcnt(0)")

        #     J.emit(mfma_a0b0_k0)
        #     J.emit(mfma_a0b0_k1)
        #     for m in range(0, nrM//2):
        #         ds_read_a(ldsA1[cur_lds], mfma_A[1, 0, m], m, 0)
        #         ds_read_a(ldsA1[cur_lds], mfma_A[1, 1, m], m, 1)
        #     J.emit(vm_load_b_idx(ldsB0[cur_lds], buff_b, koffset_b, 0))

        #     # part 1:
        #     J.s_waitcnt(mod=f"vmcnt({0})")
        #     J.s_barrier()
        #     J.s_waitcnt(mod=f"lgkmcnt(0)")
        #     J.emit(mfma_a1b0_k0)
        #     J.emit(mfma_a1b0_k1)
        #     for n in range(0, nrN//2):
        #         ds_read_a(ldsB1[cur_lds], mfma_B[1, 0, n], n, 0)
        #         ds_read_a(ldsB1[cur_lds], mfma_B[1, 1, n], n, 1)
        #     J.emit(vm_load_a_idx(ldsA0[cur_lds], buff_a, koffset_a, 0))
            
        #     # part 2:
        #     J.s_waitcnt(mod=f"vmcnt({0})")
        #     J.s_barrier()
        #     J.s_waitcnt(mod=f"lgkmcnt(0)")
        #     J.emit(mfma_a0b1_k0)
        #     J.emit(mfma_a0b1_k1)
        #     for n in range(0, nrN//2): 
        #         ds_read_b(ldsB0[next_lds], mfma_B[0, 0, n], n, 0)
        #         ds_read_b(ldsB0[next_lds], mfma_B[0, 1, n], n, 1)          
        #     J.emit(vm_load_a_idx(ldsA1[cur_lds], buff_a, koffset_a, 1))

        #     # part 3:
        #     J.s_waitcnt(mod=f"vmcnt({0})")
        #     J.s_barrier()
        #     J.s_waitcnt(mod=f"lgkmcnt(0)")
        #     J.emit(mfma_a1b1_k0)
        #     J.emit(mfma_a1b1_k1)
        #     for m in range(0, nrM//2):
        #         ds_read_a(ldsA0[next_lds], mfma_A[0, 0, m], m, 0)
        #         ds_read_a(ldsA0[next_lds], mfma_A[0, 1, m], m, 1)
        #     J.emit(vm_load_b_idx(ldsB1[cur_lds], buff_b, koffset_b, 1))
    
        #     koffset_a[0] += vm_offset_inc_a
        #     koffset_b[0] += vm_offset_inc_b
            
        # for k_idx in range((K//wg_K) - 2):
        #     loop_body(k_idx)
        ####################################epologue 0:

        # cur_lds = ((K//wg_K) - 2) % 2
        # next_lds = (((K//wg_K) - 1)) % 2
        cur_lds = 0
        next_lds = 1
        #part 0
        J.s_waitcnt(mod=f"vmcnt({0})")
        J.s_barrier()
        J.s_waitcnt(mod=f"lgkmcnt(0)")

        J.emit(mfma_a0b0_k0)
        J.emit(mfma_a0b0_k1)
        for m in range(nrM//2): ds_read_a(ldsA1[0], mfma_A[1, 0, m], m, 0)
        for m in range(nrM//2): ds_read_a(ldsA1[0], mfma_A[1, 1, m], m, 1)


        # part 1:
        J.s_waitcnt(mod=f"lgkmcnt(0)")
        J.emit(mfma_a1b0_k0)
        J.emit(mfma_a1b0_k1)
        for n in range(nrN//2): ds_read_b(ldsB1[0], mfma_B[1, 0, n], n, 0)
        for n in range(nrN//2): ds_read_b(ldsB1[0], mfma_B[1, 1, n], n, 1)   

        
        # part 2:
        J.s_waitcnt(mod=f"lgkmcnt(0)")
        J.emit(mfma_a0b1_k0)
        J.emit(mfma_a0b1_k1)
        for n in range(nrN//2): ds_read_b(ldsB0[1], mfma_B[0, 0, n], n, 0)
        for n in range(nrN//2): ds_read_b(ldsB0[1], mfma_B[0, 1, n], n, 1)        

        # part 3:
        J.s_waitcnt(mod=f"lgkmcnt(0)")
        J.emit(mfma_a1b1_k0)
        J.emit(mfma_a1b1_k1)

        for m in range(nrM//2): ds_read_a(ldsA0[1], mfma_A[0, 0, m], m, 0)
        for m in range(nrM//2): ds_read_a(ldsA0[1], mfma_A[0, 1, m], m, 1)
        ####################################epologue 1:
        # part 0:

        cur_lds = next_lds
        J.s_waitcnt(mod=f"lgkmcnt(0)")
        J.emit(mfma_a0b0_k0)
        J.emit(mfma_a0b0_k1)
        for m in range(nrM//2): ds_read_a(ldsA1[1], mfma_A[1, 0, m], m, 0)
        for m in range(nrM//2): ds_read_a(ldsA1[1], mfma_A[1, 1, m], m, 1)

        # part 1:
        J.s_waitcnt(mod=f"lgkmcnt(0)")
        J.emit(mfma_a1b0_k0)
        J.emit(mfma_a1b0_k1)
        for n in range(nrN//2): ds_read_b(ldsB1[1], mfma_B[1, 0, n], n, 0)
        for n in range(nrN//2): ds_read_b(ldsB1[1], mfma_B[1, 1, n], n, 1)   

        
        # part 2:
        J.s_waitcnt(mod=f"lgkmcnt(0)")
        J.emit(mfma_a0b1_k0)
        J.emit(mfma_a0b1_k1)
        
        # part 3:
        J.s_waitcnt(mod=f"lgkmcnt(0)")
        J.emit(mfma_a1b1_k0)
        J.emit(mfma_a1b1_k1)

    for lds in ldsA0: J.free_lds(lds) 
    for lds in ldsB0: J.free_lds(lds)
    for lds in ldsA1: J.free_lds(lds) 
    for lds in ldsB1: J.free_lds(lds)
    
    vdata = J.gpr(8, "vbf16x2")
    vbf16 = J.gpr(4, "vbf16x2")
    col = J.lane_id // 16
    swap_12_col = (col & 1) * 2 + (col >> 1)
    vaddr_org = J.gpr(((J.lane_id % 16) + warp_m//2 * 16)*stride_c + swap_12_col * J.sizeof_DW4 + warp_n//2 * 4 * J.sizeof_DW2 + \
            blk_n * (wg_N * J.sizeof(C_dtype)))
    vaddr = J.gpr("vu32", 0)
    vaddr[0] = vaddr_org[0]
    for m in range(nrM//2):
        for n in range(0, nrN//2, 2):
            J.uni_cvt_pk_bf16_f32(vbf16[0], mfma_C[0,0, m , n, 0], mfma_C[0,0, m , n, 1])
            J.uni_cvt_pk_bf16_f32(vbf16[1], mfma_C[0,0, m , n,2], mfma_C[0,0, m , n, 3])
            J.uni_cvt_pk_bf16_f32(vbf16[2], mfma_C[0,0, m , n+1,0], mfma_C[0,0, m , n+1,1])
            J.uni_cvt_pk_bf16_f32(vbf16[3], mfma_C[0,0, m , n+1,2], mfma_C[0,0, m , n+1,3])
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
        
    vaddr[0] = vaddr_org[0] + 256
    for m in range(nrM//2):
        for n in range(0, nrN//2, 2):
            J.uni_cvt_pk_bf16_f32(vbf16[0], mfma_C[0,1, m , n, 0], mfma_C[0,1, m , n, 1])
            J.uni_cvt_pk_bf16_f32(vbf16[1], mfma_C[0,1, m , n,2], mfma_C[0,1, m , n, 3])
            J.uni_cvt_pk_bf16_f32(vbf16[2], mfma_C[0,1, m , n+1,0], mfma_C[0,1, m , n+1,1])
            J.uni_cvt_pk_bf16_f32(vbf16[3], mfma_C[0,1, m , n+1,2], mfma_C[0,1, m , n+1,3])
            #    a0    a1   a2   a3   | 01 23
            #    b0    b1   b2   b3   | 45 67 
            #  v_permlane16_swap_b32(a, b)S
            #    a0    b0   a2   b2   |
            #    a1    b1   a3   b3   |
            #
            # swap of row 1 & 2 are done by swapping lane-address 
            J.v_permlane16_swap_b32(vbf16[0], vbf16[2])
            J.v_permlane16_swap_b32(vbf16[1], vbf16[3])
            buff_c.store_dwordx4(vbf16, vaddr, 0, offset12=n*4*J.sizeof_DW2)
        vaddr[0] += 16*stride_c
        
    vaddr[0] = vaddr_org[0] + stride_c *128
    for m in range(nrM//2):
        for n in range(0, nrN//2, 2):
            J.uni_cvt_pk_bf16_f32(vbf16[0], mfma_C[1, 0, m , n, 0], mfma_C[1, 0, m , n, 1])
            J.uni_cvt_pk_bf16_f32(vbf16[1], mfma_C[1, 0, m , n,2], mfma_C[1, 0, m , n, 3])
            J.uni_cvt_pk_bf16_f32(vbf16[2], mfma_C[1, 0, m , n+1,0], mfma_C[1, 0, m , n+1,1])
            J.uni_cvt_pk_bf16_f32(vbf16[3], mfma_C[1, 0, m , n+1,2], mfma_C[1, 0, m , n+1,3])
            #    a0    a1   a2   a3   | 01 23
            #    b0    b1   b2   b3   | 45 67 
            #  v_permlane16_swap_b32(a, b)S
            #    a0    b0   a2   b2   |
            #    a1    b1   a3   b3   |
            #
            # swap of row 1 & 2 are done by swapping lane-address 
            J.v_permlane16_swap_b32(vbf16[0], vbf16[2])
            J.v_permlane16_swap_b32(vbf16[1], vbf16[3])
            buff_c.store_dwordx4(vbf16, vaddr, 0, offset12=n*4*J.sizeof_DW2)
        vaddr[0] += 16*stride_c
        
            
    vaddr[0] = vaddr_org[0] + stride_c *128 + 256
    for m in range(nrM//2):
        for n in range(0, nrN//2, 2):
            J.uni_cvt_pk_bf16_f32(vbf16[0], mfma_C[1,1, m , n, 0], mfma_C[1,1, m , n, 1])
            J.uni_cvt_pk_bf16_f32(vbf16[1], mfma_C[1,1, m , n,2], mfma_C[1,1, m , n, 3])
            J.uni_cvt_pk_bf16_f32(vbf16[2], mfma_C[1,1, m , n+1,0], mfma_C[1,1, m , n+1,1])
            J.uni_cvt_pk_bf16_f32(vbf16[3], mfma_C[1,1, m , n+1,2], mfma_C[1,1, m , n+1,3])
            #    a0    a1   a2   a3   | 01 23
            #    b0    b1   b2   b3   | 45 67 
            #  v_permlane16_swap_b32(a, b)S
            #    a0    b0   a2   b2   |
            #    a1    b1   a3   b3   |
            #
            # swap of row 1 & 2 are done by swapping lane-address 
            J.v_permlane16_swap_b32(vbf16[0], vbf16[2])
            J.v_permlane16_swap_b32(vbf16[1], vbf16[3])
            buff_c.store_dwordx4(vbf16, vaddr, 0, offset12=n*4*J.sizeof_DW2)
        vaddr[0] += 16*stride_c