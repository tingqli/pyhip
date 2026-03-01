import pyhip

__all__ = [
    "get_mfma_loader_preshuffled",
    "get_mfma_loader_row_major",
    "get_mfma_loader",
    "get_mfma_loader_sorted_tok",
    "tb_swizzle"
]


def tb_swizzle(J, block_1d_id:"sgpr", M:"sgpr", wg_M:int, wg_N:int, N:int, M01:int, GroupNum:int):
    if GroupNum <= 1 and M01 <= 1:
        N0 = J.div_up(N, wg_N)
        blk_m = J.gpr(block_1d_id // N0)
        blk_n = J.gpr(block_1d_id - blk_m*N0)
        return blk_m, blk_n

    M0 = J.gpr(J.div_up(M, wg_M))
    N0 = J.div_up(N, wg_N)
    group_size    = J.div_up(M0 * N0, GroupNum)
    big_group_num = J.gpr(GroupNum - (group_size * GroupNum - M0 * N0))
    group_id_y    = J.gpr(block_1d_id // GroupNum)
    group_id_x    = J.gpr(block_1d_id - group_id_y * GroupNum) 

    remap_block_1d_id = J.gpr(group_id_x * group_size + group_id_y)

    with J.If(group_id_x > big_group_num):
        remap_block_1d_id[0] += (big_group_num - group_id_x)

    idx_M0 = J.gpr(remap_block_1d_id // N0)
    idx_N0 = J.gpr(remap_block_1d_id - idx_M0 * N0)

    M0_tmp     = J.gpr(M0 // M01)
    M0_mod_M01 = J.gpr(M0 - M0_tmp * M01)

    # M01_adapt = (idx_M0 < M0 - M0_mod_M01) ? M01 : M0_mod_M01;
    M01_adapt = J.gpr("su32")
    J.SetMask("scc", idx_M0 < M0 - M0_mod_M01)
    J.s_cselect_b32(M01_adapt, M01, M0_mod_M01)

    idx_M00          = J.gpr(idx_M0 // M01)
    idx_M01          = J.gpr(idx_M0 - idx_M00 * M01)
    idx_N0_M01_local = J.gpr(idx_N0 + idx_M01 * N0)

    N_out           = J.gpr(idx_N0_M01_local // M01_adapt)
    idx_loc_mod_M01 = J.gpr(idx_N0_M01_local - N_out * M01_adapt)

    M_out = J.gpr(idx_loc_mod_M01 + idx_M00 * M01)
    return M_out, N_out


def get_mfma_loader_row_major(J, num_warps, M, K, vm_stride, warp_row0):
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
        warp_row0            : per-warp row offset when ds_read_16x64() loads from LDS to VGPRs
    
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
    vm_offset_inc = K
    return vm_load, vm_load_cnt, vm_offset_inc, ds_read_16x64

def get_mfma_loader_preshuffled(J, num_warps, M, K, vm_stride, warp_row0):
    """
    return loaders for loading a [M, K] u8-tile data from VMEM (with stride of vm_stride)
    into [M, K] u8-LDS-tile and from LDS into VGPRs

    data in VMEM was pre-shuffled in input-format of MFMA_16x16x?
            x = x.reshape(M//16, 16, K//64, 4, 4).permute(0,2,3,1,4).contiguous()

    """
    # preshuffled data are in blocked layout [M//16,K//64] x [16,64]
    # both in vmem or LDS
    stride_1kb = J.div(16*vm_stride, 1024)
    nbK = J.div(K, 64) # number of 16x64 blocks along K dimension
    warp_k = J.warp_id[0] % nbK
    warp_m = J.warp_id[0] // nbK
    vmem_warp_off = warp_m * (stride_1kb * 1024) + warp_k * 1024
    vmem_voff = J.gpr(J.lane_id[0] * J.sizeof_DW4 + vmem_warp_off)
    lds_warp_off = J.gpr("su32", warp_m * (nbK * 1024) + warp_k * 1024)

    voff = J.gpr(J.lane_id[0] * J.sizeof_DW4 + (warp_row0 // 16) * (nbK * 1024))
    voff2 = J.gpr("vu32", voff[0] + 64*1024)
    def ds_read_1kb(lds, vdst, m, k):
        offset = lds + m*(nbK * 1024) + k*1024
        if offset >= 64*1024:
            voffset = voff2
            offset -= 64*1024
        else:
            voffset = voff
        J.ds_read_b128(vdst, voffset, mod=f"offset:{offset}")

    vm_load_cnt = J.div(J.div(M, 16), J.div(num_warps, nbK))

    def vm_load(lds_offset, buff, vm_offset):
        J.s_mov_b32("m0", lds_warp_off + lds_offset)
        voff = J.gpr("vu32", vmem_voff[0] + vm_offset)
        for m in range(vm_load_cnt):
            yield 1
            buff.load_dwordx4(None, voff, 0, offset12=0)
            J.s_addk_i32("m0", 64*num_warps*J.sizeof_DW4)
            voff[0] += (num_warps//nbK)*(stride_1kb)*1024
        #vmem_voff[0] += nbK * 1024

    vm_offset_inc = nbK * 1024
    return vm_load, vm_load_cnt, vm_offset_inc, ds_read_1kb

def get_mfma_loader(J, use_pre_shuffle, num_warps, M, K, vm_stride, warp_row0):
    if use_pre_shuffle:
        return get_mfma_loader_preshuffled(J, num_warps, M, K, vm_stride, warp_row0)
    else:
        return get_mfma_loader_row_major(J, num_warps, M, K, vm_stride, warp_row0)

def get_mfma_loader_sorted_tok(J, num_warps, M, K, vm_stride, warp_row0, lds_sorted_ids, TOPK, num_tokens):
    """
    Args:
        num_warps       (int) : how many warps are used for cooperatively loading data from VMEM into LDS
        M, K      (int)       : dimension of 2D tiles loaded, M rows and K columns of uint8/bytes
        vm_stride (int/sgpr)  : the stride (in bytes) of external 2D VMEM tensor to be loaded
        warp_row0             : per-warp row offset when ds_read_16x64() loads from LDS to VGPRs

        lds_sorted_ids  (int) : LDS offset where sorted_ids u32-vector is stored, each u32 item has token
                                row index stored in its low 24bits and topk index stored in its high 8bits.

        TOPK                  : > 0 : input layout [num_tokens, topk, dims]
                                <=0 : input layout [num_tokens, dims]
        num_tokens            : total number of valid tokens in external VMEM
    
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
    nbM = J.div(M, 16)
    nbK = J.div(K, 64)
    if 0:
        # check bank-conflict
        J.show_mfma_in_lds(mfma_MN=16, num_mfmas=2) 
        J.show_mfma_in_lds(mfma_MN=16, num_mfmas=2, swizzle_1=1)
        J.show_mfma_in_lds(mfma_MN=16, num_mfmas=2, swizzle_2=1)
        J.show_mfma_in_lds(mfma_MN=16, num_mfmas=2, swizzle_2=2)
        assert 0

    # each wave load 8x128 bytes , 4 waves loads 32x128 bytes
    lds_stride = K
    num_lanes_per_row = J.div(lds_stride, J.sizeof_DW4) # 8 when K==128
    num_rows_per_load = J.div(64, num_lanes_per_row)    # 8 when K==128
    warp_m_off = J.warp_id[0] * num_rows_per_load

    def swizzle(row, col):
        return (col ^ row) % num_lanes_per_row

    col = J.threadIdx.x % num_lanes_per_row
    row = J.threadIdx.x // num_lanes_per_row
    swizzle_col = swizzle(row, col)

    lds_warp_off = J.gpr("su32", warp_m_off * lds_stride)

    # each vm-load-dw4 can load 8 rows (since K=128bytes)
    # since tok-ids are discrete, we need a vmem_off for each load
    vm_load_cnt = len(range(0, M, 8*num_warps))

    vmem_voff = J.gpr(vm_load_cnt, "vu32")

    ds_vaddr = J.gpr(row * J.sizeof_DW + lds_sorted_ids)

    for m in range(vm_load_cnt):
        J.ds_read_b32(vmem_voff[m], ds_vaddr + m*num_warps*8*J.sizeof_DW)

    J.s_waitcnt(mod=f"lgkmcnt(0)")

    for m in range(vm_load_cnt):
        tokid = J.gpr(2, "vu32", vmem_voff[m] & 0xFFFFFF, vmem_voff[m] >> 24)
        if TOPK > 0:
            vmem_voff[m] = tokid[0]*(TOPK*vm_stride) + tokid[1]*vm_stride + swizzle_col * J.sizeof_DW4
        else:
            vmem_voff[m] = tokid[0]*vm_stride + swizzle_col * J.sizeof_DW4

        # maybe don't need following code, since Buffer size ensures no read overflow can happen
        #with J.ExecMask(tokid[0] >= num_tokens[0]):
        #    vmem_voff[m] = 0

    def vm_load(lds_offset, buff, vm_offset, half=None):
        J.s_mov_b32("m0", lds_warp_off + lds_offset)
        if half is None:
            m_range = range(vm_load_cnt)
        elif half == 0:
            m_range = range(0, J.div(vm_load_cnt,2))
        elif half == 1:
            m_range = range(J.div(vm_load_cnt,2), vm_load_cnt)
        else:
            assert 0
        for m in m_range:
            yield 1
            buff.load_dwordx4(None, vmem_voff[m] + vm_offset, 0, offset12=0)
            J.s_addk_i32("m0", num_warps*64*J.sizeof_DW4)
            # vmem_voff[m] += nbK * 4 * J.sizeof_DW4

    col = J.lane_id // 16
    row = J.lane_id % 16
    swizzle_col = swizzle(row, col)
    num_regs_K = J.div(K, 64)
    voff = J.gpr(num_regs_K, "vu32")
    voff2 = J.gpr(num_regs_K, "vu32")
    for k in range(num_regs_K):
        voff[k] = (row + warp_row0) * lds_stride + swizzle(row, col + k*4) * J.sizeof_DW4
        voff2[k] = voff[k] + 64*1024 # # ds_read_b128's offset is just 16bits

    def ds_read_16x64(lds, vdst, m, k):
        offset = lds + m*16*lds_stride
        if offset >= 64*1024:
            voffset = voff2[k]
            offset -= 64*1024
        else:
            voffset = voff[k]
        J.ds_read_b128(vdst, voffset, mod=f"offset:{offset}")
    vm_offset_inc = K

    return vm_load, vm_load_cnt, vm_offset_inc, ds_read_16x64
