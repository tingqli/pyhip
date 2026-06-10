import pyhip

from .common.loaders import tb_swizzle

__all__ = ["gemm_kernel_slicing"]

USE_GLUON_SWIZZLE = 1


# def get_pids(
#     M,
#     N,
#     BM: gl.constexpr,
#     BN: gl.constexpr,
#     GRID_MN: gl.constexpr,
#     NUM_XCDS: gl.constexpr,
#     GROUP_SIZE_M: gl.constexpr,
# ):
#     pid = gl.program_id(axis=0)
#     num_pid_m = gl.cdiv(M, BM)
#     num_pid_n = gl.cdiv(N, BN)

#     if NUM_XCDS != 1:
#         ## pid remapping on xcds
#         pids_per_xcd = (GRID_MN + NUM_XCDS - 1) // NUM_XCDS
#         tall_xcds = GRID_MN % NUM_XCDS
#         tall_xcds = NUM_XCDS if tall_xcds == 0 else tall_xcds
#         xcd = pid % NUM_XCDS
#         local_pid = pid // NUM_XCDS
#         if xcd < tall_xcds:
#             pid = xcd * pids_per_xcd + local_pid
#         else:
#             pid = tall_xcds * pids_per_xcd + (xcd - tall_xcds) * (pids_per_xcd - 1) + local_pid

#     if GROUP_SIZE_M == 1:
#         pid_m = pid // num_pid_n
#         pid_n = pid % num_pid_n
#     else:
#         num_pid_in_group = GROUP_SIZE_M * num_pid_n
#         group_id = pid // num_pid_in_group
#         first_pid_m = group_id * GROUP_SIZE_M
#         group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
#         pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
#         pid_n = (pid % num_pid_in_group) // group_size_m


#     return pid_m, pid_n
def get_pids(
    J,
    pid: "sgpr",
    M: "sgpr",
    N: int,
    BM: int,
    BN: int,
    NUM_XCDS: int,
    GROUP_SIZE_M: int,
    GRID_MN: int,
):

    num_pid_m = J.gpr(J.div_up(M[0], BM))
    num_pid_n = J.div_up(N, BN)

    if NUM_XCDS != 1:
        ## pid remapping on xcds
        pids_per_xcd = (GRID_MN + NUM_XCDS - 1) // NUM_XCDS
        tall_xcds = GRID_MN % NUM_XCDS
        tall_xcds = NUM_XCDS if tall_xcds == 0 else tall_xcds
        xcd = J.gpr(pid[0] % NUM_XCDS)
        local_pid = J.gpr(pid[0] // NUM_XCDS)

        pid[0] = (
            tall_xcds * pids_per_xcd
            + (xcd[0] - tall_xcds) * (pids_per_xcd - 1)
            + local_pid[0]
        )
        with J.If(xcd[0] < tall_xcds):
            pid[0] = xcd[0] * pids_per_xcd + local_pid[0]

    pid_m = J.gpr("su32")
    pid_n = J.gpr("su32")
    if GROUP_SIZE_M == 1:
        pid_m[0] = pid[0] // num_pid_n
        pid_n[0] = pid[0] % num_pid_n
    else:
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = J.gpr(pid[0] // num_pid_in_group)
        first_pid_m = J.gpr(group_id[0] * GROUP_SIZE_M)
        group_size_m = J.gpr("su32")
        group_size_m[0] = num_pid_m[0] - first_pid_m[0]
        with J.If(group_size_m[0] > GROUP_SIZE_M):
            group_size_m[0] = GROUP_SIZE_M
        reg0 = J.gpr(pid[0] % num_pid_in_group)
        pid_n[0] = reg0[0] // group_size_m[0]
        pid_m[0] = first_pid_m[0] + (reg0[0] - group_size_m[0] * pid_n[0])

    return pid_m, pid_n


@pyhip.jit()
def gemm_kernel_slicing(
    J,
    wg_M,
    wg_N,
    N,
    K,
    use_pre_shuffle,
    GRID_MN,
    pA: "void*",
    pB: "void*",
    pC: "void*",
    M: "int",
):
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
    if USE_GLUON_SWIZZLE:
        NUM_XCDS = 8
        GROUP_SIZE_M = 4
        blk_m, blk_n = get_pids(
            J,
            J.blockIdx.x,
            M,
            N,
            wg_M,
            wg_N,
            NUM_XCDS,
            GROUP_SIZE_M,
            GRID_MN,
        )
    else:
        blk_m, blk_n = tb_swizzle(J, J.blockIdx.x, M, wg_M, wg_N, N, M01, GroupNum)

    pA[:] += blk_m * (wg_M * K * J.sizeof(A_dtype))
    pB[:] += blk_n * (wg_N * K * J.sizeof(B_dtype))
    pC[:] += blk_m * (wg_M * stride_c)  # + blk_n * (wg_N * J.sizeof(C_dtype)))

    M0 = J.gpr("su32", blk_m * wg_M)
    M1 = J.gpr("su32")
    J.s_min_u32(M1, M0 + wg_M, M)
    assert N % wg_N == 0
    assert K % wg_K == 0
    num_warps = 4
    nbN = J.div(wg_N, 16)  # the N tile number per WG, each N tile 16n
    nbM = J.div(wg_M, 16)  # the M tile number per WG, each M tile 16m
    nbK = 2  # 2 MFMA 16x16 # the k tile number per WG, each K tile 32k
    buff_a = J.Buffer(pA, (M1 - M0) * stride_k)
    buff_b = J.Buffer(pB, wg_N * stride_k)
    buff_c = J.Buffer(pC, (M1 - M0) * stride_c)

    nrM = J.div(nbM, 2)  # the N tile number per warp, each N tile 16n
    nrN = J.div(nbN, 2)  # the N tile number per warp, each M tile 16n
    nrK = nbK  # the k tile number per warp, each K tile 32k

    slice_nrM = J.div(nrM, 2)
    slice_nrN = J.div(nrN, 2)
    slice_nbM = J.div(nbM, 2)
    slice_nbN = J.div(nbN, 2)
    assert slice_nrM == slice_nrN

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

    # Usually, lds read via MFMA format would cause the bank conflict between 16 rows and need to 15xpadding(16 MFMA rows -1) for one MFMA16x16x32. In one LDS entry,
    # we have 128 rows, it means (128-1)*padding_sz is needed if MFMA rows are contineously stored in LDS(m0,m1,m2....m127). One padding_sz only benefit one row in LDS,
    # which cause too much padding memory wasted in LDS memory.
    # In this kernel, we reordered the row layout in LDS. One padding_sz would be valid for {slice_nbM}(128/16=8) rows not just one rows. So total padding for one entry is
    # (16-1)*padding_sz. The reorder M layout is :

    # |       LDS read by lane row0     |        padding     |     LDS read by lane row1          |        padding    |  .....LDS read by lane row16
    # [m0,m16,m32,m48,m64,m80,m96,m112] | paddding 16 bf16   |  [m1,m17,m33,m49,m65,m81,m97,m113] | paddding 16 bf16  |  .....[m16,m32,m48,m64,m80,m96,m112,m128]
    # A LDS layout on m dimension:
    # [m0,m16,m32,m48,m64,m80,m96,m112]  the length is M//num_rows_per_load= 128 // (64 // 8) = 8, is also num_warps*vm_load_cnt
    # [m1,m17,m33,m49,m65,m81,m97,m113],
    # [m2,m18,m34,m50,m66,m82,m98,m114],
    # [m3,m19,m35,m51,m67,m83,m99,m115],
    # [m16,m32,m48,m64,m80,m96,m112,m128],

    # The LDS M dimension is determined by sharedLayoutA
    # sharedLyaout A is particially restricted by "when buffer_load into LDS, 64 lane data would be stored into LDS contineously"
    lds_padding_total = (16 - 1) * padding_sz

    # ldsA[256, 64] would be divided into 2 slices on M dimesion, ldsA0, ldsA1. each slice has 2 entries for ping-pong buffer. Each one is [128,64] elements + lds_padding_total.
    # ldsA[256, 64] would be divided into 2 slices on N dimesion, ldsA0, ldsA1. each slice has 2 entries for ping-pong buffer. Each one is [128,64] elemenst + lds_padding_total.
    ldsA0 = [
        J.alloc_lds(slice_nbM * nbK * 1024 + lds_padding_total),
        J.alloc_lds(slice_nbM * nbK * 1024 + lds_padding_total),
    ]
    ldsB0 = [
        J.alloc_lds(slice_nbN * nbK * 1024 + lds_padding_total),
        J.alloc_lds(slice_nbN * nbK * 1024 + lds_padding_total),
    ]
    ldsA1 = [
        J.alloc_lds(slice_nbM * nbK * 1024 + lds_padding_total),
        J.alloc_lds(slice_nbM * nbK * 1024 + lds_padding_total),
    ]
    ldsB1 = [
        J.alloc_lds(slice_nbN * nbK * 1024 + lds_padding_total),
        J.alloc_lds(slice_nbN * nbK * 1024 + lds_padding_total),
    ]

    # nrM, nrN offset for each warp in the WG.
    warp_nrM = J.gpr((J.warp_id[0] // 2) * slice_nrM)
    warp_nrN = J.gpr((J.warp_id[0] % 2) * slice_nrN)

    m_warpid = J.gpr(J.warp_id[0] // 2)
    n_warpid = J.gpr(J.warp_id[0] % 2)

    # [[ping,pong], [B0,A0,A1,B1], 4]
    lds_soff_precal = J.gpr(2, 4, 4, "su32")
    # 128 is LDS row in bytes. 8 rows per wap.
    lds_warp_offset = J.gpr("su32", J.warp_id[0] * 8 * 128 + J.warp_id[0] * padding_sz)
    store_stride = 64 * num_warps * J.sizeof_DW4 + padding_sz * num_warps
    for cnt in range(4):
        # LDS_0:
        # B0 offset
        lds_soff_precal[0, 0, cnt] = lds_warp_offset[0] + store_stride * cnt + ldsB0[0]
        # A0 offset
        lds_soff_precal[0, 1, cnt] = lds_soff_precal[0, 0, cnt] - ldsB0[0] + ldsA0[0]
        # A1 offset
        lds_soff_precal[0, 2, cnt] = lds_soff_precal[0, 0, cnt] - ldsB0[0] + ldsA1[0]
        # B1 offset
        lds_soff_precal[0, 3, cnt] = lds_soff_precal[0, 0, cnt] - ldsB0[0] + ldsB1[0]

        # LDS_1:
        # B0
        lds_soff_precal[1, 0, cnt] = lds_soff_precal[0, 0, cnt] - ldsB0[0] + ldsB0[1]
        # A0 offset
        lds_soff_precal[1, 1, cnt] = lds_soff_precal[0, 0, cnt] - ldsB0[0] + ldsA0[1]
        # A1 offset
        lds_soff_precal[1, 2, cnt] = lds_soff_precal[0, 0, cnt] - ldsB0[0] + ldsA1[1]
        # B1 offset
        lds_soff_precal[1, 3, cnt] = lds_soff_precal[0, 0, cnt] - ldsB0[0] + ldsB1[1]

    def get_mfma_loader_padding(J, num_warps, M, K, vm_stride, warpid_m, padding_sz=32):
        """
        return padding loaders for loading a [M, K] u8-tile data from VMEM (with stride of vm_stride)
        into [M, K] u8-LDS-tile and from LDS into VGPRs.

        when loading from VMEM into LDS, all warps are loading data cooperatively
        with coalescing in mind. here coaleascing would be only applied on lanes in one row(8 lanes per row in this case).
        m0, m1 in the physical vmem will NOT be stored into LDS contineously.
        To ensure [m0,m16,m32,m48,m64,m80,m96,m112], [m1,m17,m33,m49,m65,m81,m97,m113]... layout in LDS, we would make 64 lanes in one warp
        accesss vmem in groups. Each row in lane is a group. one group would load vmem data with coalescing but difference group would have
        a row stride when reading, such as 8 row lanes would load [m0,m16,m32,m48,m64,m80,m96,m112] NOT [m0, m1, m2, m3, m4, m5, m6, m7].
        The row stride is determined by the number of warps and vm_load count.

        when loading from LDS into VGPRs, ds_read_b128() is used thus each load feed
        16x64 bytes into VGPR, which is suitable for working with MFMA_16x16x? instrutions.

        this function returns a few python-closure (also maybe a generator) loader functions
        the external VMEM buffer and offsets are specified in these loader function

        Args:
            num_warps (int)      : how many warps are used for cooperatively loading data from VMEM into LDS
            M, K      (int)      : dimension of 2D tiles loaded, M rows and K columns of uint8/bytes
            vm_stride (int/sgpr) : the stride (in bytes) of external 2D VMEM tensor to be loaded
            warpid_m             : warpid on m,n dimension
            padding_sz           : padding element size in byte.

        Returns:

            vm_load(idx, lds_offset, buff, vm_offset, emitter=None) : [M, K] u8-VMEM-tile to [M, K] u8-LDS-tile loader generator function.
                                                                     would have vm_load_cnt*2 yield in the vm_load. buffer_load, and s_addk_i32
                                                                     both need to be interleaved with
                            idx             (int) : the slice index for current load. [256, 64] is divided into 2 slice.
                            lds_offset      (int) : pre-calculated LDS offset for current load. Used to mov into M0.
                            buff         (Buffer) : VMEM buffer object
                            vm_offset  (int/sgpr) : offset relative to buff base
                            emitter               : The emmitter ISAs would be inserted when having some scalar instructions. Any better way?

            vm_load_cnt                          : number of vm load instructions issued by each vm_load()

            vm_offset_inc                        : increamental offsets after each vm_load (which is K)

            ds_read_16x64(lds_offset, vdst, m, k) : load a [16, 64] u8-LDS-tile into VGPRs
                lds_offset   (int) : source u8-LDS-tile offset
                vdst       (vgprs) : dest VGPRs
                m            (int) : m*16 is row offset of [16,64] tile inside [M, K] u8-LDS-tile
                k            (int) : k*64 is col offset of [16,64] tile inside [M, K] u8-LDS-tile
        """
        # 1K bytes is 8 rows in LDS. 1024/2/64 = 8 rows. 64 thread lds write would be 8 rows.
        # each wave load 8x128 bytes , 4 waves loads 32x128 bytes. load_count M//(num_rows_per_load*wavecnt)
        lds_stride = K
        num_lanes_per_row = J.div(lds_stride, J.sizeof_DW4)
        num_rows_per_load = J.div(64, num_lanes_per_row)
        warp_m_off = J.warp_id[0] * num_rows_per_load
        vm_load_cnt = len(range(0, M, num_rows_per_load * num_warps))

        lane_row = J.lane_id // num_lanes_per_row
        lane_col = J.lane_id % num_lanes_per_row
        # The stride between 2 consecutive rows for vmem loading.
        lane_row_stride = vm_stride * num_warps * vm_load_cnt

        # Pre-calculate all the voffset.
        # [slice_parts, vm_load_cnt]
        vmem_voff = J.gpr(2, vm_load_cnt, "vu32")

        vmem_voff[0, 0] = (
            lane_row * lane_row_stride
            + J.warp_id[0] * vm_stride
            + lane_col * J.sizeof_DW4
        )
        vmem_voff[1, 0] = vmem_voff[0, 0] + M * vm_stride

        for cnt in range(1, vm_load_cnt):
            vmem_voff[0, cnt] = vmem_voff[0, 0] + (vm_stride * num_warps) * cnt
            vmem_voff[1, cnt] = vmem_voff[0, cnt] + M * vm_stride

        def vm_load(idx, lds_offset, buff, vm_offset):
            for m in range(0, vm_load_cnt):
                J.s_mov_b32("m0", lds_offset[m])
                yield 1
                buff.load_dwordx4(None, vmem_voff[idx, m], 0, offset12=0)
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
            # MFMA 1st row lane: [m0,m16,m32,m48,m64,m80,m96,m112] would be firsted read into regA, regB. Then to wave.
            #                    [m0,m16,m32,m48]  4 x lds_read_b128 by wave0 & wave 1
            #                    [m64,m80,m96,m112]  4 x lds_read_b128 by wave2 & wave 3

            voff[k] = (
                row * (lds_rlane_stride + padding_sz)
                + warpid_m * lds_stride * vm_load_cnt
                + (col + k * 4) * J.sizeof_DW4
            )
            voff2[k] = voff[k] + 64 * 1024

        # for m in range(slice_nrM):
        #     ds_read_a(ldsA1[cur_lds], mfma_A[1, 0, m], m, 0)
        #     J.emit([mfma_a0b0_k0, mfma_a0b0_k1], 16)
        #     ds_read_a(ldsA1[cur_lds], mfma_A[1, 1, m], m, 1)
        #     J.emit([mfma_a0b0_k0, mfma_a0b0_k1], 16)
        def ds_read_16x64_idx(lds_offset, vdst, m, k):
            # this offset would be calculated in the compiled time.
            offset = lds_offset + lds_stride * m
            if offset >= 64 * 1024:
                voffset = voff2[k]
                offset -= 64 * 1024
            else:
                voffset = voff[k]
            J.ds_read_b128(vdst, voffset, mod=f"offset:{offset}")

        vm_offset_inc = K
        return vm_load, vm_load_cnt, vm_offset_inc, ds_read_16x64_idx

    # In the 4 wave case:
    # load 128x64xbf16 by 4 waves: vm_load_a/vm_load_b would have 4 buffer_load_dwordx4 ISAs.
    # read 128x64xbf16 from one LDS entry into reg by 4 waves: would need 8 ds_read_b128 ISAs.

    vm_load_a, vm_load_cnt_a, vm_offset_inc_a, ds_read_a = get_mfma_loader_padding(
        J, num_warps, wg_M // 2, 128, stride_k, m_warpid, padding_sz
    )
    vm_load_b, vm_load_cnt_b, vm_offset_inc_b, ds_read_b = get_mfma_loader_padding(
        J, num_warps, wg_N // 2, 128, stride_k, n_warpid, padding_sz
    )

    print(f"============={nbM=}, {nbN=}, {nbK=} {nrM=} {nrN=} {nrK=}")

    s_offset_inc_a = J.gpr("su32", vm_offset_inc_a)
    s_offset_inc_b = J.gpr("su32", vm_offset_inc_b)
    # [2, 2] is for [a/b_slice_num, bk_slice_num]
    mfma_A = J.gpr(2, 2, slice_nrM, 4, "vbf16x2")
    mfma_B = J.gpr(2, 2, slice_nrN, 4, "vbf16x2")
    # [2, 2] is for [a_idx, b_idx]
    mfma_C = J.gpr(2, 2, slice_nrM, slice_nrN, 4, "af32")

    def mfma(a_idx, b_idx, reg_id):
        for m in range(slice_nrM):
            for n in range(slice_nrN):
                J.v_mfma_f32_16x16x32_bf16(
                    mfma_C[a_idx, b_idx, m, n],
                    mfma_B[b_idx, reg_id, n],
                    mfma_A[a_idx, reg_id, m],
                    mfma_C[a_idx, b_idx, m, n],
                )
                yield 16

    koffset_a = J.gpr("su32", 0)
    koffset_b = J.gpr("su32", 0)

    ####################################prelog:
    # [[ping,pong], [B0,A0,A1,B1], 4]
    # lds_soff_precal = J.gpr(2, 4, 4, "vu32")
    # ping prefetch
    # AC B0 to ping
    J.emit(vm_load_b(0, lds_soff_precal[0, 0], buff_b, koffset_b))
    # AC A0 to ping
    J.emit(vm_load_a(0, lds_soff_precal[0, 1], buff_a, koffset_a))
    # AC A1 to ping
    J.emit(vm_load_a(1, lds_soff_precal[0, 2], buff_a, koffset_a))
    # AC B1 to ping
    J.emit(vm_load_b(1, lds_soff_precal[0, 3], buff_b, koffset_b))

    # A, B advance BK
    buff_a.advance(s_offset_inc_a[0])
    buff_b.advance(s_offset_inc_b[0])

    # pong prefetch
    # AC B0 to pong
    J.emit(vm_load_b(0, lds_soff_precal[1, 0], buff_b, koffset_b))
    # AC A0 to pong
    J.emit(vm_load_a(0, lds_soff_precal[1, 1], buff_a, koffset_a))
    # AC A1 to pong
    J.emit(vm_load_a(1, lds_soff_precal[1, 2], buff_a, koffset_a))
    # AC B1 to pong
    J.emit(vm_load_b(1, lds_soff_precal[1, 3], buff_b, koffset_b))

    # A, B advance BK
    buff_a.advance(s_offset_inc_a[0])
    buff_b.advance(s_offset_inc_b[0])

    # readiness: B0 and A0 to LDS0
    J.s_waitcnt(mod=f"vmcnt({24})")
    J.s_barrier()

    # B0 first half K LDS0 into Reg.
    for n in range(slice_nrN):
        ds_read_b(ldsB0[0], mfma_B[0, 0, n], n, 0)
    # B0 second half K LDS0 into Reg.
    for n in range(slice_nrN):
        ds_read_b(ldsB0[0], mfma_B[0, 1, n], n, 1)
    # A0 first half K LDS0 into Reg.
    for m in range(slice_nrM):
        ds_read_a(ldsA0[0], mfma_A[0, 0, m], m, 0)
    # A0 second half K LDS0 into Reg.
    for m in range(slice_nrM):
        ds_read_a(ldsA0[0], mfma_A[0, 1, m], m, 1)
    mfma_C[...] = 0

    # readiness: vm:AC A1[cur] to LDS[cur%2], lds:B0 and A0[cur] into reg
    J.s_waitcnt(mod=f"vmcnt({20}) lgkmcnt(0)")
    J.s_barrier()

    ###################################main loop:
    # https://github.com/ROCm/gfx950-gluon-tutorials/tree/main/kernels/gemm/a16w16/v8_sliceMN
    def loop_body(idx):

        cur_lds = idx % 2
        next_lds = (idx + 1) % 2
        # initiliaze generator
        mfma_a0b0_k0 = mfma(0, 0, 0)
        mfma_a0b1_k0 = mfma(0, 1, 0)
        mfma_a1b0_k0 = mfma(1, 0, 0)
        mfma_a1b1_k0 = mfma(1, 1, 0)

        mfma_a0b0_k1 = mfma(0, 0, 1)
        mfma_a0b1_k1 = mfma(0, 1, 1)
        mfma_a1b0_k1 = mfma(1, 0, 1)
        mfma_a1b1_k1 = mfma(1, 1, 1)
        # The loop is divided into 4 parts to get Matmul(A0,B0), Matmul(A1,B0), Matmul(A0,B1), Matmul(A1,B1).
        # for  data in [cur],[cur+1], [cur+2]. LDS read and prefetch would follow the sequence of B0->A0->A1->B1. LDS load sequence would be divided into 2 iteration.
        #  Loop initial status:
        #                    [MFMA REG status]:  [cur]:A0 and B0 already in registers. Other regs loading not trigged.
        #                    [LDS status]     :  cur and cur+1 prefetch into LDS is triggered with sequence B0->A0->A1->B1. [cur]: A0 and B0  already in LDS and have been loaded into registers.
        #
        #  pipelining in one loop:
        #                     1. MFMA ISA         :  A0xB0[cur]  ->A1xB0[cur]   -> A0xB1[cur]      -> A1xB1[cur]
        #                     2. READ LDS ISA     :  A1[cur]     ->B1[cur]      -> B0[cur+1]       ->A0[cur+1], for cur+1, sequence still is B0->A0->A1->B1, but divided into 2 iterations.
        #                     3. VMEM prefetch ISA: B0[cur+2]    ->A0[cur+2]    ->A1[cur+2]        ->B1[cur+2]
        #
        #  The magic pipelineing:
        #  The pipelining can interleave MFMA with LDS read and VMEM prefetch because of the independence between them. The 'independence'
        #  is not just data dependency but also resource dependency(source and destination are shared by MFMA, VMEM and LDS ISAs).
        #  The slice gemm have 2 sets of LDS but only one set of reg A,B,C. So for each ISA source is available and destination is released.
        #  So [cur] and [cur+2] share same LDS entry.. [cur] and [cur+1] share same reg entry. Let us take the part 2 as example:
        #           MFMA ISA:     ensure A0, B1 cur in the reg by lgkcnt(0)
        #           READ LDS:     src: ensure B0[cur+1] prefetched into LDS by vmcnt(20), dest: ensure Reg B0 is released for[cur],
        #                         because B0 data would be overrided by [cur+1]. B0[cur] data is not needed.
        #           VMEM prefetch: dest: A1[cur+2] share same LDS entry with A1[cur]. Ensure A1[cur] data already read into registers.
        #
        # ISA interleaving:
        # ISAs number in each part of pipeline: MFMA ISAs:32, read_lds_b128:  8, vm_load: 4. others. The 32 MFMAs ISA would be distributed for each part:
        #  4MFMA + 8x(read_lds + MFMA) + (s_addk_i32 + 2MFMA) + 4x(3MFMA+buffer_load+s_addk_i32) + 4xMFMA + (2xMFMA+barrier+s_waitcnt)

        # mainloop part 0:
        J.emit([mfma_a0b0_k0, mfma_a0b0_k1], 16)
        vm_load = vm_load_b(0, lds_soff_precal[cur_lds, 0], buff_b, koffset_b)
        J.emit([mfma_a0b0_k0, mfma_a0b0_k1], 48)
        J.emit([mfma_a0b0_k0, mfma_a0b0_k1], 16)
        J.emit([mfma_a0b0_k0, mfma_a0b0_k1], 16)
        for m in range(slice_nrM):
            ds_read_a(ldsA1[cur_lds], mfma_A[1, 0, m], m, 0)
            J.emit([mfma_a0b0_k0, mfma_a0b0_k1], 16)
        for m in range(slice_nrM):
            ds_read_a(ldsA1[cur_lds], mfma_A[1, 1, m], m, 1)
            J.emit([mfma_a0b0_k0, mfma_a0b0_k1], 16)

        for _ in range(vm_load_cnt_b):
            J.emit(vm_load, 1)
            J.emit([mfma_a0b0_k0, mfma_a0b0_k1], 16)
            J.emit(vm_load, 1)
            J.emit([mfma_a0b0_k0, mfma_a0b0_k1], 16)
            J.emit([mfma_a0b0_k0, mfma_a0b0_k1], 16)
            J.emit([mfma_a0b0_k0, mfma_a0b0_k1], 16)

        J.emit([mfma_a0b0_k0, mfma_a0b0_k1], 16)
        J.s_waitcnt(mod=f"vmcnt({20}) lgkmcnt(0)")
        J.emit([mfma_a0b0_k0, mfma_a0b0_k1], 16)
        J.s_barrier()

        # mainloop part 1:
        J.emit([mfma_a1b0_k0, mfma_a1b0_k1], 16)
        vm_load = vm_load_a(0, lds_soff_precal[cur_lds, 1], buff_a, koffset_a)
        J.emit([mfma_a1b0_k0, mfma_a1b0_k1], 48)
        J.emit([mfma_a1b0_k0, mfma_a1b0_k1], 16)
        J.emit([mfma_a1b0_k0, mfma_a1b0_k1], 16)
        for n in range(slice_nrN):
            ds_read_b(ldsB1[cur_lds], mfma_B[1, 0, n], n, 0)
            J.emit([mfma_a1b0_k0, mfma_a1b0_k1], 16)
        for n in range(slice_nrN):
            ds_read_b(ldsB1[cur_lds], mfma_B[1, 1, n], n, 1)
            J.emit([mfma_a1b0_k0, mfma_a1b0_k1], 16)

        for _ in range(vm_load_cnt_a):
            J.emit(vm_load, 1)
            J.emit([mfma_a1b0_k0, mfma_a1b0_k1], 16)
            J.emit(vm_load, 1)
            J.emit([mfma_a1b0_k0, mfma_a1b0_k1], 16)
            J.emit([mfma_a1b0_k0, mfma_a1b0_k1], 16)
            J.emit([mfma_a1b0_k0, mfma_a1b0_k1], 16)
        J.emit([mfma_a1b0_k0, mfma_a1b0_k1], 16)
        J.s_waitcnt(mod=f"vmcnt({20}) lgkmcnt(0)")
        J.emit([mfma_a1b0_k0, mfma_a1b0_k1], 16)
        J.s_barrier()

        # mainloop part 2:
        J.emit([mfma_a0b1_k0, mfma_a0b1_k1], 16)
        vm_load = vm_load_a(1, lds_soff_precal[cur_lds, 2], buff_a, koffset_a)
        J.emit([mfma_a0b1_k0, mfma_a0b1_k1], 48)
        J.emit([mfma_a0b1_k0, mfma_a0b1_k1], 16)
        J.emit([mfma_a0b1_k0, mfma_a0b1_k1], 16)
        for n in range(slice_nrN):
            ds_read_b(ldsB0[next_lds], mfma_B[0, 0, n], n, 0)
            J.emit([mfma_a0b1_k0, mfma_a0b1_k1], 16)
        for n in range(slice_nrN):
            ds_read_b(ldsB0[next_lds], mfma_B[0, 1, n], n, 1)
            J.emit([mfma_a0b1_k0, mfma_a0b1_k1], 16)

        for _ in range(vm_load_cnt_a):
            J.emit(vm_load, 1)
            J.emit([mfma_a0b1_k0, mfma_a0b1_k1], 16)
            J.emit(vm_load, 1)
            J.emit([mfma_a0b1_k0, mfma_a0b1_k1], 16)
            J.emit([mfma_a0b1_k0, mfma_a0b1_k1], 16)
            J.emit([mfma_a0b1_k0, mfma_a0b1_k1], 16)

        J.emit([mfma_a0b1_k0, mfma_a0b1_k1], 16)
        J.s_waitcnt(mod=f"vmcnt({20})")
        J.emit([mfma_a0b1_k0, mfma_a0b1_k1], 16)
        J.s_barrier()

        # mainloop part 3:
        J.emit([mfma_a1b1_k0, mfma_a1b1_k1], 16)
        vm_load = vm_load_b(1, lds_soff_precal[cur_lds, 3], buff_b, koffset_b)
        J.emit([mfma_a1b1_k0, mfma_a1b1_k1], 16)
        J.emit([mfma_a1b1_k0, mfma_a1b1_k1], 16)
        J.emit([mfma_a1b1_k0, mfma_a1b1_k1], 16)
        for m in range(slice_nrM):
            ds_read_a(ldsA0[next_lds], mfma_A[0, 0, m], m, 0)
            J.emit([mfma_a1b1_k0, mfma_a1b1_k1], 16)
        for m in range(slice_nrM):
            ds_read_a(ldsA0[next_lds], mfma_A[0, 1, m], m, 1)
            J.emit([mfma_a1b1_k0, mfma_a1b1_k1], 16)

        for _ in range(vm_load_cnt_b):
            J.emit(vm_load, 1)
            J.emit([mfma_a1b1_k0, mfma_a1b1_k1], 16)
            J.emit(vm_load, 1)
            J.emit([mfma_a1b1_k0, mfma_a1b1_k1], 16)
            J.emit([mfma_a1b1_k0, mfma_a1b1_k1], 16)
            J.emit([mfma_a1b1_k0, mfma_a1b1_k1], 16)

        J.emit([mfma_a1b1_k0, mfma_a1b1_k1], 16)
        J.s_waitcnt(mod=f"vmcnt({20}) lgkmcnt(0)")
        J.emit([mfma_a1b1_k0, mfma_a1b1_k1], 16)
        J.s_barrier()
        J.emit([mfma_a1b1_k0, mfma_a1b1_k1], 16)
        buff_a.advance(s_offset_inc_a[0])
        J.emit([mfma_a1b1_k0, mfma_a1b1_k1], 16)
        buff_b.advance(s_offset_inc_b[0])
        J.emit([mfma_a1b1_k0, mfma_a1b1_k1], 16)
        # endof mainloop

    if 0:
        for k_idx in range((K // wg_K) - 2):
            loop_body(k_idx)
    else:
        koff = J.gpr("su32", 0)
        loop_cnt = ((K // wg_K) - 2) // 2
        kidx = 0
        with J.While(koff[0] < loop_cnt):
            # unroll the ping-pong
            loop_body(0)
            loop_body(1)
            koff[0] += 1
    ####################################epologue 0:

    mfma_a0b0_k0 = mfma(0, 0, 0)
    mfma_a0b1_k0 = mfma(0, 1, 0)
    mfma_a1b0_k0 = mfma(1, 0, 0)
    mfma_a1b1_k0 = mfma(1, 1, 0)
    mfma_a0b0_k1 = mfma(0, 0, 1)
    mfma_a0b1_k1 = mfma(0, 1, 1)
    mfma_a1b0_k1 = mfma(1, 0, 1)
    mfma_a1b1_k1 = mfma(1, 1, 1)

    cur_lds = ((K // wg_K) - 2) % 2
    next_lds = (cur_lds + 1) % 2
    # epologue0 part 0
    J.emit([mfma_a0b0_k0], 16)
    J.s_waitcnt(mod=f"vmcnt({20}) lgkmcnt(0)")
    J.emit([mfma_a0b0_k0], 16)
    J.s_barrier()
    J.emit(mfma_a0b0_k0, 64)
    for m in range(slice_nrM):
        J.emit([mfma_a0b0_k0], 16)
        ds_read_a(ldsA1[cur_lds], mfma_A[1, 0, m], m, 0)
    for m in range(slice_nrM):
        J.emit([mfma_a0b0_k0], 16)
        ds_read_a(ldsA1[cur_lds], mfma_A[1, 1, m], m, 1)
    J.emit(mfma_a0b0_k0)
    J.emit(mfma_a0b0_k1, 16 * (16 - 2))
    J.s_waitcnt(mod=f"vmcnt({16}) lgkmcnt(0)")
    J.emit(mfma_a0b0_k1, 16)
    J.s_barrier()
    J.emit(mfma_a0b0_k1, 16)

    # epologue0 part 1:
    J.emit([mfma_a1b0_k0], 64)
    for n in range(slice_nrN):
        J.emit([mfma_a1b0_k0], 16)
        ds_read_b(ldsB1[cur_lds], mfma_B[1, 0, n], n, 0)
    for n in range(slice_nrN):
        J.emit([mfma_a1b0_k0], 16)
        ds_read_b(ldsB1[cur_lds], mfma_B[1, 1, n], n, 1)
    J.emit(mfma_a1b0_k0)
    J.emit(mfma_a1b0_k1, 16 * (16 - 2))
    J.s_waitcnt(mod=f"vmcnt({12}) lgkmcnt(0)")
    J.emit(mfma_a1b0_k1, 16)
    J.s_barrier()
    J.emit(mfma_a1b0_k1, 16)

    # epologue0 part 2:
    J.emit([mfma_a0b1_k0], 64)
    for n in range(slice_nrN):
        J.emit([mfma_a0b1_k0], 16)
        ds_read_b(ldsB0[next_lds], mfma_B[0, 0, n], n, 0)
    for n in range(slice_nrN):
        J.emit([mfma_a0b1_k0], 16)
        ds_read_b(ldsB0[next_lds], mfma_B[0, 1, n], n, 1)
    J.emit(mfma_a0b1_k0)
    J.emit(mfma_a0b1_k1, 16 * (16 - 2))

    J.s_waitcnt(mod=f"vmcnt({8})")
    J.emit(mfma_a0b1_k1, 16)
    J.s_barrier()
    J.emit(mfma_a0b1_k1, 16)

    # epologue0 part 3:
    J.emit([mfma_a1b1_k0], 64)
    for m in range(slice_nrM):
        J.emit([mfma_a1b1_k0], 16)
        ds_read_a(ldsA0[next_lds], mfma_A[0, 0, m], m, 0)
    for m in range(slice_nrM):
        J.emit([mfma_a1b1_k0], 16)
        ds_read_a(ldsA0[next_lds], mfma_A[0, 1, m], m, 1)
    J.emit(mfma_a1b1_k0)
    J.emit(mfma_a1b1_k1, 16 * (16 - 2))
    J.s_waitcnt(mod=f"vmcnt({4}) lgkmcnt(0)")
    J.emit(mfma_a1b1_k1, 16)
    J.s_barrier()
    J.emit(mfma_a1b1_k1, 16)

    ####################################epologue 1:
    # epologue1 part 0:

    ####################################epologue 1:
    # part 0:
    mfma_a0b0_k0 = mfma(0, 0, 0)
    mfma_a0b1_k0 = mfma(0, 1, 0)
    mfma_a1b0_k0 = mfma(1, 0, 0)
    mfma_a1b1_k0 = mfma(1, 1, 0)
    mfma_a0b0_k1 = mfma(0, 0, 1)
    mfma_a0b1_k1 = mfma(0, 1, 1)
    mfma_a1b0_k1 = mfma(1, 0, 1)
    mfma_a1b1_k1 = mfma(1, 1, 1)

    cur_lds = next_lds
    J.s_waitcnt(mod=f"vmcnt({4}) lgkmcnt(0)")
    J.emit([mfma_a0b0_k0, mfma_a0b0_k1], 16)
    J.s_barrier()

    for m in range(slice_nrM):
        J.emit([mfma_a0b0_k0, mfma_a0b0_k1], 16)
        ds_read_a(ldsA1[cur_lds], mfma_A[1, 0, m], m, 0)
    for m in range(slice_nrM):
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
    vaddr_org = J.gpr(
        ((J.lane_id % 16) + warp_nrM * 16) * stride_c
        + swap_12_col * J.sizeof_DW4
        + warp_nrN * 4 * J.sizeof_DW2
        + blk_n * (wg_N * J.sizeof(C_dtype))
    )
    vaddr = J.gpr("vu32", 0)
    vaddr[0] = vaddr_org[0]

    for n in range(slice_nrN):
        J.emit([mfma_a1b0_k0, mfma_a1b0_k1], 16)
        ds_read_b(ldsB1[cur_lds], mfma_B[1, 0, n], n, 0)
    for n in range(slice_nrN):
        J.emit([mfma_a1b0_k0, mfma_a1b0_k1], 16)
        ds_read_b(ldsB1[cur_lds], mfma_B[1, 1, n], n, 1)
    for m in range(slice_nrM):
        for n in range(0, slice_nrN, 2):
            J.emit([mfma_a1b0_k0, mfma_a1b0_k1], 16)
            J.uni_cvt_pk_bf16_f32(
                vbf16[0], mfma_C[0, 0, m, n, 0], mfma_C[0, 0, m, n, 1]
            )
            J.uni_cvt_pk_bf16_f32(
                vbf16[1], mfma_C[0, 0, m, n, 2], mfma_C[0, 0, m, n, 3]
            )
            J.uni_cvt_pk_bf16_f32(
                vbf16[2], mfma_C[0, 0, m, n + 1, 0], mfma_C[0, 0, m, n + 1, 1]
            )
            J.emit([mfma_a1b0_k0, mfma_a1b0_k1], 16)
            J.uni_cvt_pk_bf16_f32(
                vbf16[3], mfma_C[0, 0, m, n + 1, 2], mfma_C[0, 0, m, n + 1, 3]
            )
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
            buff_c.store_dwordx4(vbf16, vaddr, 0, offset12=n * 4 * J.sizeof_DW2)
        vaddr[0] += 16 * stride_c

    J.emit(mfma_a1b0_k0)
    J.emit(mfma_a1b0_k1)
    # part 2:
    J.s_waitcnt(mod=f"lgkmcnt(0)")
    J.emit([mfma_a0b1_k0, mfma_a0b1_k1], 16)
    vaddr[0] = vaddr_org[0] + stride_c * 128
    for m in range(slice_nrM):
        for n in range(0, slice_nrN, 2):
            J.uni_cvt_pk_bf16_f32(
                vbf16[0], mfma_C[1, 0, m, n, 0], mfma_C[1, 0, m, n, 1]
            )
            J.emit([mfma_a0b1_k0, mfma_a0b1_k1], 16)
            J.uni_cvt_pk_bf16_f32(
                vbf16[1], mfma_C[1, 0, m, n, 2], mfma_C[1, 0, m, n, 3]
            )
            J.uni_cvt_pk_bf16_f32(
                vbf16[2], mfma_C[1, 0, m, n + 1, 0], mfma_C[1, 0, m, n + 1, 1]
            )
            J.uni_cvt_pk_bf16_f32(
                vbf16[3], mfma_C[1, 0, m, n + 1, 2], mfma_C[1, 0, m, n + 1, 3]
            )
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
            buff_c.store_dwordx4(vbf16, vaddr, 0, offset12=n * 4 * J.sizeof_DW2)
        vaddr[0] += 16 * stride_c
    J.emit(mfma_a0b1_k0)
    J.emit(mfma_a0b1_k1)
    # part 3:
    vaddr[0] = vaddr_org[0] + 256
    J.emit([mfma_a1b1_k0, mfma_a1b1_k1], 16)
    for m in range(slice_nrM):
        for n in range(0, slice_nrN, 2):
            J.uni_cvt_pk_bf16_f32(
                vbf16[0], mfma_C[0, 1, m, n, 0], mfma_C[0, 1, m, n, 1]
            )
            J.uni_cvt_pk_bf16_f32(
                vbf16[1], mfma_C[0, 1, m, n, 2], mfma_C[0, 1, m, n, 3]
            )
            J.emit([mfma_a1b1_k0, mfma_a1b1_k1], 16)
            J.uni_cvt_pk_bf16_f32(
                vbf16[2], mfma_C[0, 1, m, n + 1, 0], mfma_C[0, 1, m, n + 1, 1]
            )
            J.uni_cvt_pk_bf16_f32(
                vbf16[3], mfma_C[0, 1, m, n + 1, 2], mfma_C[0, 1, m, n + 1, 3]
            )
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
            buff_c.store_dwordx4(vbf16, vaddr, 0, offset12=n * 4 * J.sizeof_DW2)
        vaddr[0] += 16 * stride_c

    J.emit(mfma_a1b1_k0)
    J.emit(mfma_a1b1_k1)

    for lds in ldsA0:
        J.free_lds(lds)
    for lds in ldsB0:
        J.free_lds(lds)
    for lds in ldsA1:
        J.free_lds(lds)
    for lds in ldsB1:
        J.free_lds(lds)

    vaddr[0] = vaddr_org[0] + stride_c * 128 + 256
    for m in range(slice_nrM):
        for n in range(0, slice_nrN, 2):
            J.uni_cvt_pk_bf16_f32(
                vbf16[0], mfma_C[1, 1, m, n, 0], mfma_C[1, 1, m, n, 1]
            )
            J.uni_cvt_pk_bf16_f32(
                vbf16[1], mfma_C[1, 1, m, n, 2], mfma_C[1, 1, m, n, 3]
            )
            J.uni_cvt_pk_bf16_f32(
                vbf16[2], mfma_C[1, 1, m, n + 1, 0], mfma_C[1, 1, m, n + 1, 1]
            )
            J.uni_cvt_pk_bf16_f32(
                vbf16[3], mfma_C[1, 1, m, n + 1, 2], mfma_C[1, 1, m, n + 1, 3]
            )
            #    a0    a1   a2   a3   | 01 23
            #    b0    b1   b2   b3   | 45 67
            #  v_permlane16_swap_b32(a, b)S
            #    a0    b0   a2   b2   |
            #    a1    b1   a3   b3   |
            #
            # swap of row 1 & 2 are done by swapping lane-address
            J.v_permlane16_swap_b32(vbf16[0], vbf16[2])
            J.v_permlane16_swap_b32(vbf16[1], vbf16[3])
            buff_c.store_dwordx4(vbf16, vaddr, 0, offset12=n * 4 * J.sizeof_DW2)
        vaddr[0] += 16 * stride_c
