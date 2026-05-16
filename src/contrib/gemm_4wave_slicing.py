import pyhip

from .common.loaders import  tb_swizzle

__all__ = [
   "gemm_kernel_slicing"
]

@pyhip.jit()
def gemm_kernel_slicing(J, wg_M, wg_N, N, K, use_pre_shuffle, pA:"void*", pB:"void*", pC:"void*", M:"int"):
    # 128 bytes
    # wg_K = J.div(128, J.sizeof_bf16)
    wg_K = 64

    A_dtype = "bf16"
    B_dtype = "bf16"
    C_dtype = "bf16"
    M01 = 4
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

    nrM = J.div(nbM, 2) # the N tile number per warp, each N tile 16n
    nrN = J.div(nbN, 2) # the N tile number per warp, each M tile 16n
    nrK = nbK           # the k tile number per warp, each K tile 32k

    slice_nrM = J.div(nrM, 2)
    slice_nrN = J.div(nrM, 2)
    
    assert (slice_nrM == slice_nrN)
    

    # Half-M global load layout
    # gLoadLayoutA: gl.constexpr = gl.DistributedLinearLayout(
    #     reg_bases=[[0, 1], [0, 2], [0, 4], [4, 0], [8, 0]],
    #     lane_bases=[[0, 8], [0, 16], [0, 32], [16, 0], [32, 0], [64, 0]],
    #     warp_bases=[[1, 0], [2, 0]],
    #     block_bases=[],
    #     shape=[BLOCK_M // 2, BLOCK_K],
    # )
    
    # # Half-M padded shared layout
    # sharedLayoutA: gl.constexpr = gl.PaddedSharedLayout(
    #     [[512, 16]],
    #     [
    #         [0, 1],
    #         [0, 2],
    #         [0, 4],
    #         [0, 8],
    #         [0, 16],
    #         [0, 32],
    #         [16, 0],
    #         [32, 0],
    #         [64, 0],
    #         [1, 0],
    #         [2, 0],
    #         [4, 0],
    #         [8, 0],
    #     ],
    #     [],
    #     [BLOCK_M // 2, BLOCK_K],
    # )
    
    # padding in bytes. 
    padding_sz = 32
    
    #|       LDS read by lane row0     |        padding     |     LDS read by lane row1          |        padding    |  .....LDS read by lane row16     
    #[m0,m16,m32,m48,m64,m80,m96,m112] | paddding 16 bf16   |  [m1,m17,m33,m49,m65,m81,m97,m113] | paddding 16 bf16  |  .....[m16,m32,m48,m64,m80,m96,m112,m128]
    # A LDS layout on m dimension:
    #[m0,m16,m32,m48,m64,m80,m96,m112]  the length is M//num_rows_per_load= 128 // (64 // 8) = 8, is also num_warps*vm_load_cnt
    #[m1,m17,m33,m49,m65,m81,m97,m113],
    #[m2,m18,m34,m50,m66,m82,m98,m114],
    #[m3,m19,m35,m51,m67,m83,m99,m115],
    #[m16,m32,m48,m64,m80,m96,m112,m128],

    # The LDS M dimension is determined by sharedLayoutA
    # sharedLyaout A is particially restricted by "when buffer_load into LDS, 64 lane data would be stored into LDS contineously"
    lds_padding_total = (16-1)*padding_sz
    ldsA0 = [J.alloc_lds(nbM//2*nbK*1024 + lds_padding_total), J.alloc_lds(nbM//2*nbK*1024 + lds_padding_total)]
    ldsB0 = [J.alloc_lds(nbN//2*nbK*1024 + lds_padding_total), J.alloc_lds(nbN//2*nbK*1024 + lds_padding_total)]
    ldsA1 = [J.alloc_lds(nbM//2*nbK*1024 + lds_padding_total), J.alloc_lds(nbM//2*nbK*1024 + lds_padding_total)]
    ldsB1 = [J.alloc_lds(nbN//2*nbK*1024 + lds_padding_total), J.alloc_lds(nbN//2*nbK*1024 + lds_padding_total)]

    warp_m = J.gpr((J.warp_id[0] // 2)*nrM) # M start tile idx of the warp inside the one WG
    warp_n = J.gpr((J.warp_id[0] % 2)*nrN)  # N start tile idx of the warp inside the one WG

    m_warpid = J.gpr(J.warp_id[0] // 2) 
    n_warpid = J.gpr(J.warp_id[0] % 2)

    # get_mfma_loader(J, use_pre_shuffle, num_warps, wg_M, 128, stride_k, warp_m//2*16)
    def get_mfma_loader_padding(J, num_warps, M, K, vm_stride, warpid_m, padding_sz=32):
        """
        return loaders for loading a [M, K] u8-tile data from VMEM (with stride of vm_stride)
        into [M, K] u8-LDS-tile and from LDS into VGPRs

        when loading from VMEM into LDS, all warps are loading data cooperatively
        with coalescing in mind. all threads are loading contigously in row-major layout

        when loading from LDS into VGPRs, ds_read_b128() is used thus each load feed
        16x64 bytes into VGPR, which is suitable for working with MFMA_16x16x? instrutions.
        
        this function returns a few python-closure (also maybe a generator) loader functions
        the external VMEM buffer and offsets are specified in these loader function

        Args:
            num_warps (int)      : how many warps are used for cooperatively loading data from VMEM into LDS
            M, K      (int)      : dimension of 2D tiles loaded, M rows and K columns of uint8/bytes
            vm_stride (int/sgpr) : the stride (in bytes) of external 2D VMEM tensor to be loaded
            warpid_m             : warpid on m,n dimension
            padding_sz           : padding elment size in byte.
        
        Returns:
            vm_load(lds_offset, buff, vm_offset) : [M, K] u8-VMEM-tile to [M, K] u8-LDS-tile loader generator function
                            lds_offset      (int) : target u8-LDS-tile offset
                            buff         (Buffer) : VMEM buffer object
                            vm_offset  (int/sgpr) : offset relative to buff base

            vm_load_cnt                          : number of vm load instructions issued by each vm_load()

            vm_offset_inc                        : increamental offsets after each vm_load (which is K)

            ds_read_16x64(lds_offset, vdst, m, k) : load a [16, 64] u8-LDS-tile into VGPRs
                lds_offset   (int) : source u8-LDS-tile offset
                vdst       (vgprs) : dest VGPRs
                m            (int) : m*16 is row offset of [16,64] tile inside [M, K] u8-LDS-tile
                k            (int) : k*64 is col offset of [16,64] tile inside [M, K] u8-LDS-tile
        """
        #1K bytes is 8 rows in LDS. 1024/2/54 = 8 rows. 64 thread lds write would be 8 rows.
        # each wave load 8x128 bytes , 8 waves loads 64x128 bytes
        lds_stride = K
        num_lanes_per_row = J.div(lds_stride, J.sizeof_DW4)
        num_rows_per_load = J.div(64, num_lanes_per_row)
        warp_m_off = J.warp_id[0] * num_rows_per_load
        vm_load_cnt = len(range(0, M, num_rows_per_load * num_warps))


        lane_row = J.lane_id // num_lanes_per_row
        lane_col = J.lane_id % num_lanes_per_row
        lane_row_stride = vm_stride*num_warps*vm_load_cnt
        vmem_voff = J.gpr(lane_row * lane_row_stride + J.warp_id[0] * vm_stride + lane_col * J.sizeof_DW4)
        lds_warp_off = J.gpr("su32", warp_m_off * lds_stride + J.warp_id[0] * padding_sz)

        def vm_load_idx(lds_offset, buff, vm_offset, emitter=None):
            temp = J.gpr("su32", lds_warp_off[0] + lds_offset)
            if emitter is not None:
                J.emit(emitter, 16)
            J.s_mov_b32("m0", temp[0])
            if emitter is not None:
                J.emit(emitter, 16)
            voff = J.gpr("vu32", vmem_voff[0] + vm_offset)
            load_stride = (num_rows_per_load * num_warps) * vm_stride
            
            for m in range(0, M//(8*num_warps)):
                buff.load_dwordx4(None, voff, 0, offset12=0)
                yield 1
                J.s_addk_i32("m0", 64*num_warps*J.sizeof_DW4+padding_sz*num_warps)
                yield 1
                voff[0] += (num_warps) * vm_stride
                yield 1



        # the [M x K] bytes LDS buffer is accessed by following closure
        # this closure using ds_read_b128 to load a 16x64 bytes MFMA data
        # thus LDS buffer's layout is blocked as [M//16,K//64] x [16,64]
        # each warp has its own row offsets specified in warp_row0
        assert M % 16 == 0
        assert K % 64 == 0

        m_number_warps = num_warps // 2
        col = J.lane_id // 16
        row = J.lane_id % 16
        num_regs_K = K // 64
        # LDS load count = 2*vm_load_cnt, but one row is divided into 2 lds read. So row is same 
        lds_rlane_stride = lds_stride * m_number_warps * vm_load_cnt

        voff = J.gpr(num_regs_K, "vu32")
        voff2 = J.gpr(num_regs_K, "vu32")

        for k in range(num_regs_K):
            # each ds_read_b128 took 4 x DW-lanes
            # voff[k] = (row + warp_row0) * lds_stride + swizzle((row + warp_row0), col + k*4) * J.sizeof_DW4
            voff[k] = row * (lds_rlane_stride+padding_sz) + warpid_m*lds_stride*vm_load_cnt + (col + k*4) * J.sizeof_DW4
            voff2[k] = voff[k] + 64*1024

        def ds_read_16x64_idx(lds_offset, vdst, m, k):
            offset = lds_offset + lds_stride*m
            if offset >= 64*1024:
                voffset = voff2[k]
                offset -= 64*1024
            else:
                voffset = voff[k]
            J.ds_read_b128(vdst, voffset, mod=f"offset:{offset}")
        vm_offset_inc = K
        # if 0:
        #     return vm_load, vm_load_cnt, vm_offset_inc, ds_read_16x64
        return vm_load_idx, vm_load_cnt, vm_offset_inc, ds_read_16x64_idx

    vm_load_a_idx, vm_load_cnt_a, vm_offset_inc_a, ds_read_a = get_mfma_loader_padding(J, num_warps, wg_M//2, 128, stride_k, m_warpid, padding_sz)
    vm_load_b_idx, vm_load_cnt_b, vm_offset_inc_b, ds_read_b = get_mfma_loader_padding(J, num_warps, wg_N//2, 128, stride_k, n_warpid, padding_sz)

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
    koffset_a0 = J.gpr("su32", 0)
    koffset_b0 = J.gpr("su32", 0)
    koffset_a1 = J.gpr("su32", stride_k *wg_M//2)
    koffset_b1 = J.gpr("su32", stride_k *wg_N//2)

    mfma_C[...] = 0
    ####################################prelog:
    # ping prefetch
    #AC B0 to ping
    J.emit(vm_load_b_idx(ldsB0[0], buff_b, koffset_b0))
    #AC A0 to ping
    J.emit(vm_load_a_idx(ldsA0[0], buff_a, koffset_a0))
    #AC A1 to ping
    J.emit(vm_load_a_idx(ldsA1[0], buff_a, koffset_a1))
    #AC B1 to ping
    J.emit(vm_load_b_idx(ldsB1[0], buff_b, koffset_b1))
    
    koffset_a0[0] += vm_offset_inc_a
    koffset_b0[0] += vm_offset_inc_b
    koffset_a1[0] += vm_offset_inc_a
    koffset_b1[0] += vm_offset_inc_b

    # pong prefetch
    #AC B0 to pong
    J.emit(vm_load_b_idx(ldsB0[1], buff_b, koffset_b0))
    #AC A0 to pong
    J.emit(vm_load_a_idx(ldsA0[1], buff_a, koffset_a0))
    #AC A1 to pong
    J.emit(vm_load_a_idx(ldsA1[1], buff_a, koffset_a1))
    #AC B1 to pong
    J.emit(vm_load_b_idx(ldsB1[1], buff_b, koffset_b1))
    
    koffset_a0[0] += vm_offset_inc_a
    koffset_b0[0] += vm_offset_inc_b
    koffset_a1[0] += vm_offset_inc_a
    koffset_b1[0] += vm_offset_inc_b
    
    # readiness: AC B0 and A0 to LDS0
    J.s_waitcnt(mod=f"vmcnt({24})")
    J.s_barrier() 
    
    # B0 in LDS0 into Reg. 
    for n in range(nrN//2): ds_read_b(ldsB0[0], mfma_B[0, 0, n], n, 0)
    for n in range(nrN//2): ds_read_b(ldsB0[0], mfma_B[0, 1, n], n, 1)
    # A0 in LDS0 into Reg. 
    for m in range(nrM//2): ds_read_a(ldsA0[0], mfma_A[0, 0, m], m, 0)
    for m in range(nrM//2): ds_read_a(ldsA0[0], mfma_A[0, 1, m], m, 1)

    # part 0: Matmul(0)(A0xB0), LR(cur) A[1] form LDS[cur%2], prefetch AC(cur+2) B[0] to LDS[cur%2]
    # readiness: AC A1[cur] to LDS[cur%2],      readiness: B0 and A0[cur] into reg 

    J.s_waitcnt(mod=f"vmcnt({20}) lgkmcnt(0)")
    J.s_barrier()
    ###################################main loop:
    def loop_body(idx):
        cur_lds = idx % 2
        next_lds = (idx + 1) % 2
        # initiliaze generator
        mfma_a0b0_k0=mfma(0, 0, 0)
        mfma_a0b1_k0=mfma(0, 1, 0)
        mfma_a1b0_k0=mfma(1, 0, 0)
        mfma_a1b1_k0=mfma(1, 1, 0)
    
        mfma_a0b0_k1=mfma(0, 0, 1)
        mfma_a0b1_k1=mfma(0, 1, 1)
        mfma_a1b0_k1=mfma(1, 0, 1)
        mfma_a1b1_k1=mfma(1, 1, 1)

        J.emit([mfma_a0b0_k0, mfma_a0b0_k1], 16)
        vm_load = vm_load_b_idx(ldsB0[cur_lds], buff_b, koffset_b0, mfma_a0b0_k1)
        J.emit([mfma_a0b0_k0, mfma_a0b0_k1], 48)
        for m in range(nrM//2):
            J.emit([mfma_a0b0_k0, mfma_a0b0_k1], 16)
            ds_read_a(ldsA1[cur_lds], mfma_A[1, 0, m], m, 0)
            J.emit([mfma_a0b0_k0, mfma_a0b0_k1], 16)
            ds_read_a(ldsA1[cur_lds], mfma_A[1, 1, m], m, 1)
        for _ in range(vm_load_cnt_b):
            J.emit([mfma_a0b0_k0, mfma_a0b0_k1], 16)
            J.emit(vm_load, 1)                
            J.emit([mfma_a0b0_k0, mfma_a0b0_k1], 16)
            J.emit(vm_load, 1)                
            J.emit([mfma_a0b0_k0, mfma_a0b0_k1], 16)
            J.emit(vm_load, 1)                
            J.emit([mfma_a0b0_k0, mfma_a0b0_k1], 16)

        # part 1:
        J.emit([mfma_a0b0_k0, mfma_a0b0_k1], 16)
        J.s_waitcnt(mod=f"vmcnt({20}) lgkmcnt(0)")
        J.emit([mfma_a0b0_k0, mfma_a0b0_k1], 16)
        J.s_barrier()

        J.emit([mfma_a1b0_k0, mfma_a1b0_k1],16)
        vm_load = vm_load_a_idx(ldsA0[cur_lds], buff_a, koffset_a0, mfma_a1b0_k0)
        J.emit([mfma_a1b0_k0, mfma_a1b0_k1],48)
        for n in range(nrN//2):
            J.emit([mfma_a1b0_k0, mfma_a1b0_k1], 16)
            ds_read_b(ldsB1[cur_lds], mfma_B[1, 0, n], n, 0)
            J.emit([mfma_a1b0_k0, mfma_a1b0_k1], 16)
            ds_read_b(ldsB1[cur_lds], mfma_B[1, 1, n], n, 1) 
        for _ in range(vm_load_cnt_a):
            J.emit([mfma_a1b0_k0, mfma_a1b0_k1], 16)
            J.emit(vm_load, 1)
            J.emit([mfma_a1b0_k0, mfma_a1b0_k1], 16)
            J.emit(vm_load, 1)
            J.emit([mfma_a1b0_k0, mfma_a1b0_k1], 16)
            J.emit(vm_load, 1)
            J.emit([mfma_a1b0_k0, mfma_a1b0_k1], 16)

        # J.emit(mfma_a1b0_k0)
        # J.emit(mfma_a1b0_k1)

        # part 2:
        J.emit([mfma_a1b0_k0, mfma_a1b0_k1], 16)
        J.s_waitcnt(mod=f"vmcnt({20}) lgkmcnt(0)")
        J.emit([mfma_a1b0_k0, mfma_a1b0_k1], 16)
        J.s_barrier()
        J.emit([mfma_a0b1_k0, mfma_a0b1_k1], 16)
        vm_load = vm_load_a_idx(ldsA1[cur_lds], buff_a, koffset_a1, mfma_a0b1_k0)
        J.emit([mfma_a0b1_k0, mfma_a0b1_k1], 48)
        for n in range(nrN//2):
            J.emit([mfma_a0b1_k0, mfma_a0b1_k1], 16)
            ds_read_b(ldsB0[next_lds], mfma_B[0, 0, n], n, 0)
            J.emit([mfma_a0b1_k0, mfma_a0b1_k1], 16)
            ds_read_b(ldsB0[next_lds], mfma_B[0, 1, n], n, 1)
        for _ in range(vm_load_cnt_a):
            J.emit([mfma_a0b1_k0, mfma_a0b1_k1], 16)
            J.emit(vm_load, 1)
            J.emit([mfma_a0b1_k0, mfma_a0b1_k1], 16)
            J.emit(vm_load, 1)
            J.emit([mfma_a0b1_k0, mfma_a0b1_k1], 16)
            J.emit(vm_load, 1)
            J.emit([mfma_a0b1_k0, mfma_a0b1_k1], 16)


        # J.emit(mfma_a0b1_k0)
        # J.emit(mfma_a0b1_k1)
        # part 3:
        J.emit([mfma_a0b1_k0, mfma_a0b1_k1], 16)
        J.s_waitcnt(mod=f"vmcnt({20})")
        J.emit([mfma_a0b1_k0, mfma_a0b1_k1], 16)
        J.s_barrier()
        J.emit([mfma_a1b1_k0, mfma_a1b1_k1], 16)
        vm_load = vm_load_b_idx(ldsB1[cur_lds], buff_b, koffset_b1, mfma_a1b1_k0)
        J.emit([mfma_a1b1_k0, mfma_a1b1_k1], 16)
        for m in range(nrM//2):
            J.emit([mfma_a1b1_k0, mfma_a1b1_k1], 16)
            ds_read_a(ldsA0[next_lds], mfma_A[0, 0, m], m, 0)
            J.emit([mfma_a1b1_k0, mfma_a1b1_k1], 16)
            ds_read_a(ldsA0[next_lds], mfma_A[0, 1, m], m, 1)
        for _ in range(vm_load_cnt_b):
            J.emit([mfma_a1b1_k0, mfma_a1b1_k1], 16)
            J.emit(vm_load, 1)
            J.emit([mfma_a1b1_k0, mfma_a1b1_k1], 16)
            J.emit(vm_load, 1)
            J.emit([mfma_a1b1_k0, mfma_a1b1_k1], 16)
            J.emit(vm_load, 1)
            J.emit([mfma_a1b1_k0, mfma_a1b1_k1], 16)

        # J.emit(mfma_a1b1_k0)
        # J.emit(mfma_a1b1_k1)
        # readiness: AC A1[cur] to LDS[cur%2],      readiness: B0 and A0[cur] into reg 
        J.emit([mfma_a1b1_k0, mfma_a1b1_k1], 16)
        J.s_waitcnt(mod=f"vmcnt({20}) lgkmcnt(0)")
        J.emit([mfma_a1b1_k0, mfma_a1b1_k1], 16)
        J.s_barrier()
        J.emit([mfma_a1b1_k0, mfma_a1b1_k1], 16)
        koffset_a0[0] += vm_offset_inc_a
        koffset_a1[0] += vm_offset_inc_a

        J.emit([mfma_a1b1_k0, mfma_a1b1_k1], 16)
        koffset_b0[0] += vm_offset_inc_b
        koffset_b1[0] += vm_offset_inc_b
        J.emit([mfma_a1b1_k0, mfma_a1b1_k1], 16)
    if 0:
        for k_idx in range((K//wg_K) - 2):
            loop_body(k_idx)
    else:
        koff = J.gpr("su32", 0)
        loop_cnt = ((K//wg_K) - 2)//2
        kidx = 0
        with J.While(koff[0] < loop_cnt):
            loop_body(0)
            loop_body(1)
            koff[0] += 1
    ####################################epologue 0:

    mfma_a0b0_k0=mfma(0, 0, 0)
    mfma_a0b1_k0=mfma(0, 1, 0)
    mfma_a1b0_k0=mfma(1, 0, 0)
    mfma_a1b1_k0=mfma(1, 1, 0)
    mfma_a0b0_k1=mfma(0, 0, 1)
    mfma_a0b1_k1=mfma(0, 1, 1)
    mfma_a1b0_k1=mfma(1, 0, 1)
    mfma_a1b1_k1=mfma(1, 1, 1)

    cur_lds = ((K//wg_K) - 2) % 2
    next_lds = (cur_lds+1) % 2
    #part 0
    J.s_waitcnt(mod=f"vmcnt({20}) lgkmcnt(0)")
    J.emit([mfma_a0b0_k0, mfma_a0b0_k1], 16)
    J.s_barrier()

    for m in range(nrM//2):
        J.emit([mfma_a0b0_k0, mfma_a0b0_k1], 16)
        ds_read_a(ldsA1[cur_lds], mfma_A[1, 0, m], m, 0)
    for m in range(nrM//2):
        J.emit([mfma_a0b0_k0, mfma_a0b0_k1], 16)
        ds_read_a(ldsA1[cur_lds], mfma_A[1, 1, m], m, 1)
    J.emit(mfma_a0b0_k0)
    J.emit(mfma_a0b0_k1)
    # part 1:
    J.s_waitcnt(mod=f"vmcnt({16}) lgkmcnt(0)")
    J.emit([mfma_a1b0_k0, mfma_a1b0_k1], 16)
    J.s_barrier()

    for n in range(nrN//2):
        J.emit([mfma_a1b0_k0, mfma_a1b0_k1], 16)
        ds_read_b(ldsB1[cur_lds], mfma_B[1, 0, n], n, 0)
    for n in range(nrN//2):
        J.emit([mfma_a1b0_k0, mfma_a1b0_k1], 16)
        ds_read_b(ldsB1[cur_lds], mfma_B[1, 1, n], n, 1)   
    J.emit(mfma_a1b0_k0)
    J.emit(mfma_a1b0_k1)
    
    # part 2:
    J.s_waitcnt(mod=f"vmcnt({12}) lgkmcnt(0)")
    J.emit([mfma_a0b1_k0, mfma_a0b1_k1], 16)
    J.s_barrier()

    for n in range(nrN//2):
        J.emit([mfma_a0b1_k0, mfma_a0b1_k1], 16)
        ds_read_b(ldsB0[next_lds], mfma_B[0, 0, n], n, 0)
    for n in range(nrN//2):
        J.emit([mfma_a0b1_k0, mfma_a0b1_k1], 16)
        ds_read_b(ldsB0[next_lds], mfma_B[0, 1, n], n, 1)        
    J.emit(mfma_a0b1_k0)
    J.emit(mfma_a0b1_k1)
    # part 3:
    J.s_waitcnt(mod=f"vmcnt({8})")
    J.emit([mfma_a1b1_k0, mfma_a1b1_k1], 16)
    J.s_barrier()

    for m in range(nrM//2):
        J.emit([mfma_a1b1_k0, mfma_a1b1_k1], 16)
        ds_read_a(ldsA0[next_lds], mfma_A[0, 0, m], m, 0)
    for m in range(nrM//2):
        J.emit([mfma_a1b1_k0, mfma_a1b1_k1], 16)
        ds_read_a(ldsA0[next_lds], mfma_A[0, 1, m], m, 1)
    J.emit(mfma_a1b1_k0)
    J.emit(mfma_a1b1_k1)
    
    ####################################epologue 1:
    # part 0:

    mfma_a0b0_k0=mfma(0, 0, 0)
    mfma_a0b1_k0=mfma(0, 1, 0)
    mfma_a1b0_k0=mfma(1, 0, 0)
    mfma_a1b1_k0=mfma(1, 1, 0)
    mfma_a0b0_k1=mfma(0, 0, 1)
    mfma_a0b1_k1=mfma(0, 1, 1)
    mfma_a1b0_k1=mfma(1, 0, 1)
    mfma_a1b1_k1=mfma(1, 1, 1)

    cur_lds = next_lds
    J.s_waitcnt(mod=f"vmcnt({4}) lgkmcnt(0)")
    J.emit([mfma_a0b0_k0, mfma_a0b0_k1], 16)
    J.s_barrier()

    for m in range(nrM//2):
        J.emit([mfma_a0b0_k0, mfma_a0b0_k1], 16)
        ds_read_a(ldsA1[cur_lds], mfma_A[1, 0, m], m, 0)
    for m in range(nrM//2):
        J.emit([mfma_a0b0_k0, mfma_a0b0_k1], 16)
        ds_read_a(ldsA1[cur_lds], mfma_A[1, 1, m], m, 1)
    J.emit(mfma_a0b0_k0)
    J.emit(mfma_a0b0_k1)
    # part 1:
    J.s_waitcnt(mod=f"vmcnt({0}) lgkmcnt(0)")
    J.emit([mfma_a1b0_k0, mfma_a1b0_k1], 16)
    J.s_barrier()
    J.emit([mfma_a1b0_k0, mfma_a1b0_k1], 16)

    vdata = J.gpr(8, "vbf16x2")
    vbf16 = J.gpr(4, "vbf16x2")
    col = J.lane_id // 16
    swap_12_col = (col & 1) * 2 + (col >> 1)
    vaddr_org = J.gpr(((J.lane_id % 16) + warp_m//2 * 16)*stride_c + swap_12_col * J.sizeof_DW4 + warp_n//2 * 4 * J.sizeof_DW2 + \
            blk_n * (wg_N * J.sizeof(C_dtype)))
    vaddr = J.gpr("vu32", 0)
    vaddr[0] = vaddr_org[0]

    for n in range(nrN//2):
        J.emit([mfma_a1b0_k0, mfma_a1b0_k1], 16)
        ds_read_b(ldsB1[cur_lds], mfma_B[1, 0, n], n, 0)
    for n in range(nrN//2):
        J.emit([mfma_a1b0_k0, mfma_a1b0_k1], 16)
        ds_read_b(ldsB1[cur_lds], mfma_B[1, 1, n], n, 1)
    for m in range(nrM//2):
        for n in range(0, nrN//2, 2):
            J.emit([mfma_a1b0_k0, mfma_a1b0_k1], 16)
            J.uni_cvt_pk_bf16_f32(vbf16[0], mfma_C[0,0, m , n, 0], mfma_C[0,0, m , n, 1])
            J.uni_cvt_pk_bf16_f32(vbf16[1], mfma_C[0,0, m , n,2], mfma_C[0,0, m , n, 3])
            J.uni_cvt_pk_bf16_f32(vbf16[2], mfma_C[0,0, m , n+1,0], mfma_C[0,0, m , n+1,1])
            J.emit([mfma_a1b0_k0, mfma_a1b0_k1], 16)
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
            J.emit([mfma_a1b0_k0, mfma_a1b0_k1], 16)
            buff_c.store_dwordx4(vbf16, vaddr, 0, offset12=n*4*J.sizeof_DW2)
        vaddr[0] += 16*stride_c
    
    J.emit(mfma_a1b0_k0)
    J.emit(mfma_a1b0_k1)
    # part 2:
    J.s_waitcnt(mod=f"lgkmcnt(0)")
    J.emit([mfma_a0b1_k0, mfma_a0b1_k1], 16)
    vaddr[0] = vaddr_org[0] + stride_c *128
    for m in range(nrM//2):
        for n in range(0, nrN//2, 2):
            J.uni_cvt_pk_bf16_f32(vbf16[0], mfma_C[1, 0, m , n, 0], mfma_C[1, 0, m , n, 1])
            J.emit([mfma_a0b1_k0, mfma_a0b1_k1], 16)
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
            J.emit([mfma_a0b1_k0, mfma_a0b1_k1], 16)

            J.v_permlane16_swap_b32(vbf16[0], vbf16[2])
            J.v_permlane16_swap_b32(vbf16[1], vbf16[3])
            J.emit([mfma_a0b1_k0, mfma_a0b1_k1], 16)
            buff_c.store_dwordx4(vbf16, vaddr, 0, offset12=n*4*J.sizeof_DW2)
        vaddr[0] += 16*stride_c
    J.emit(mfma_a0b1_k0)
    J.emit(mfma_a0b1_k1)
    # part 3:
    vaddr[0] = vaddr_org[0] + 256
    J.emit([mfma_a1b1_k0, mfma_a1b1_k1], 16)
    for m in range(nrM//2):
        for n in range(0, nrN//2, 2):
            J.uni_cvt_pk_bf16_f32(vbf16[0], mfma_C[0,1, m , n, 0], mfma_C[0,1, m , n, 1])
            J.uni_cvt_pk_bf16_f32(vbf16[1], mfma_C[0,1, m , n,2], mfma_C[0,1, m , n, 3])
            J.emit([mfma_a1b1_k0, mfma_a1b1_k1], 16)
            J.uni_cvt_pk_bf16_f32(vbf16[2], mfma_C[0,1, m , n+1,0], mfma_C[0,1, m , n+1,1])
            J.uni_cvt_pk_bf16_f32(vbf16[3], mfma_C[0,1, m , n+1,2], mfma_C[0,1, m , n+1,3])
            #    a0    a1   a2   a3   | 01 23
            #    b0    b1   b2   b3   | 45 67 
            #  v_permlane16_swap_b32(a, b)S
            #    a0    b0   a2   b2   |
            #    a1    b1   a3   b3   |
            #
            # swap of row 1 & 2 are done by swapping lane-address
            J.emit([mfma_a1b1_k0, mfma_a1b1_k1], 16)
            J.v_permlane16_swap_b32(vbf16[0], vbf16[2])
            J.v_permlane16_swap_b32(vbf16[1], vbf16[3])
            J.emit([mfma_a1b1_k0, mfma_a1b1_k1], 16)
            buff_c.store_dwordx4(vbf16, vaddr, 0, offset12=n*4*J.sizeof_DW2)
        vaddr[0] += 16*stride_c
        
    J.emit(mfma_a1b1_k0)
    J.emit(mfma_a1b1_k1)

    for lds in ldsA0: J.free_lds(lds) 
    for lds in ldsB0: J.free_lds(lds)
    for lds in ldsA1: J.free_lds(lds) 
    for lds in ldsB1: J.free_lds(lds)
            
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