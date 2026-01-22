import pyhip
import pytest
import functools
import torch


torch.set_printoptions(linewidth=3000, sci_mode=False, edgeitems=8, )
torch.set_default_device('cuda')
torch.manual_seed(0)

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

use_pre_shuffle = 0
wg_M = 256
wg_N = 256
M = wg_M * 16
N = wg_N * 16
K = 8192

blk_cnt = (M // wg_M) * (N // wg_N)

A0 = torch.randn(M, K).to(dtype=torch.bfloat16)
B0 = torch.randn(N, K).to(dtype=torch.bfloat16)
C0 = A0 @ B0.t()

if use_pre_shuffle:
    A = pre_shuffle(A0, 16)
    B = pre_shuffle(B0, 16)
else:
    A = A0
    B = B0

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

        wgload_cnt = len(range(J.div(nbM, num_warps//nbK)))
        # return a loader constructor which can emit
        def wg_load_next(lds_offset):
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

        wgload_cnt = len(range(0, nbM * 16, 8*num_warps))

        def wg_load_next(lds_offset):
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

    return wg_load_next, wgload_cnt, ds_read_1kb


@pyhip.jit()
def gemm_kernel(J, N, K, use_pre_shuffle, pA:"void*", pB:"void*", pC:"void*", M:"int"):
    wg_M = 256
    wg_N = 256
    # 128 bytes
    wg_K = J.div(128, J.sizeof_bf16)

    A_dtype = "bf16"
    B_dtype = "bf16"
    C_dtype = "bf16"
    M01 = 8
    GroupNum = 8

    stride_k = K * J.sizeof_bf16

    blk_m, blk_n = J.tb_swizzle(J.blockIdx.x, M, wg_M, wg_N, N, M01, GroupNum)
    pA[:] = pA[:] + blk_m * (wg_M * K * J.sizeof(A_dtype))
    pB[:] = pB[:] + blk_n * (wg_N * K * J.sizeof(B_dtype))
    pC[:] = pC[:] + (blk_m * (wg_M * N * J.sizeof(C_dtype)) + blk_n * (wg_N * J.sizeof(C_dtype)))

    assert N % wg_N == 0
    num_warps = 4
    nbN = J.div(wg_N, 16)
    nbM = J.div(wg_M, 16)
    nbK = 2 # 2 MFMA 16x16 
    buff_a = J.Buffer(pA, wg_M * stride_k)
    buff_b = J.Buffer(pB, wg_N * stride_k)
    ldsA = [J.alloc_lds(nbM * nbK * 1024), J.alloc_lds(nbM * nbK * 1024)]
    ldsB = [J.alloc_lds(nbN * nbK * 1024), J.alloc_lds(nbN * nbK * 1024)]

    nrM = J.div(nbM, 2)
    nrN = J.div(nbN, 2)
    nrK = nbK

    warp_m = J.gpr((J.warp_id[0] // 2)*nrM)
    warp_n = J.gpr((J.warp_id[0] % 2)*nrN)

    wg_load_a, wg_load_cnt_a, ds_read_a = get_loader(J, buff_a, use_pre_shuffle, nbM, nbK, stride_k, warp_m)
    wg_load_b, wg_load_cnt_b, ds_read_b = get_loader(J, buff_b, use_pre_shuffle, nbN, nbK, stride_k, warp_n)

    print(f"============={nbM=}, {nbN=}, {nbK=} {nrM=} {nrN=} {nrK=}")

    mfma_A = J.gpr(2, nrM, 4, "vbf16x2")
    mfma_B = J.gpr(2, nrN, 4, "vbf16x2")
    mfma_C = J.gpr(nrM, nrN, 4, "af32")
    mfma_C[...] = 0

    def mfma(reg_id):
        for m in range(nrM):
            for n in range(nrN):
                J.v_mfma_f32_16x16x32_bf16(mfma_C[m,n], mfma_B[reg_id, n], mfma_A[reg_id, m], mfma_C[m,n])
                yield 16

    J.emit(wg_load_a(ldsA[0])) # vmcnt:8
    J.emit(wg_load_b(ldsB[0])) # vmcnt:8

    J.emit(wg_load_a(ldsA[1])) # vmcnt:8
    J.emit(wg_load_b(ldsB[1])) # vmcnt:8

    '''
    ab0: mfma ab0 | ds_read ab1; wait_lgkmcnt(0), barrier; vm-load a01
    ab1: mfma ab1 | vm-load b01; wait_vmcnt, barrier, ds_read ab2

    ab2: mfma ab2 | ds_read ab3; wait_lgkmcnt(0), barrier; vm-load  a23
    ab3: mfma ab3 | vm-load b23;  wait_vmcnt, barrier, ds_read ab0
    '''
    J.s_waitcnt(mod="vmcnt(16)")
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
        wgload = wg_load_a(ldsA[lds_id])
        J.emit([mfma_ab0, mfma_ab1], 16)
        J.emit(wgload, 1)
        for _ in range(wg_load_cnt_a):
            J.emit(wgload, 1)
            J.emit([mfma_ab0, mfma_ab1], 96)
        J.emit(wgload)

        # mfma ab1

        # vm-load b01
        wgload = wg_load_b(ldsB[lds_id])
        J.emit([mfma_ab0, mfma_ab1], 16)
        J.emit(wgload, 1)
        for _ in range(wg_load_cnt_b - 3):
            J.emit(wgload, 1)
            J.emit([mfma_ab0, mfma_ab1], 96)

        J.emit(mfma_ab0)

        # wait vm-load a23/b23 to finish
        J.s_waitcnt(mod=f"vmcnt({16-3})")
        J.s_barrier()

        J.emit(wgload, 1)
        # ds_read ab2
        for m in range(nrM):
            J.emit(mfma_ab1, 16)
            ds_read_a(ldsA[(lds_id + 1)&1], mfma_A[0, m], m, 0)

        J.emit(wgload, 1)
        for n in range(nrN):
            J.emit(mfma_ab1, 16)
            ds_read_b(ldsB[(lds_id + 1)&1], mfma_B[0, n], n, 0)

        J.emit(wgload)
        J.emit(mfma_ab1)
        
        J.s_waitcnt(mod=f"lgkmcnt(0)")

    #for koff in range(J.div(K, wg_K)):
    #    loop_body(koff & 1)

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

    if 1:
        buff_c = J.Buffer(pC, wg_M * N * J.sizeof_bf16)
        def swizzle(row, col):
            return (row) ^ col
        mfma_MN = 16
        wave_nCM = nrM
        wave_nCN = nrN
        wave_size_M = wg_M // 2
        wave_size_N = wg_N // 2
        warp_id_m = J.warp_id // 2
        warp_id_n = J.warp_id % 2
        warp_offset_m = warp_id_m * wave_size_M
        warp_offset_n = warp_id_n * wave_size_N
        
        lds_out = J.alloc_lds(4 * mfma_MN * wave_size_N * J.sizeof_fp32)
        lds_warp_offset = J.warp_id * (mfma_MN * wave_size_N * J.sizeof_fp32)
        num_lanes_per_row = wave_size_N * J.sizeof_fp32 // J.sizeof_DW4
        rows_per_read = J.div(64, num_lanes_per_row)
        assert mfma_MN % rows_per_read == 0            

        vdata = J.gpr(4, "vu32")
        vbf16 = J.gpr(2, "vbf16x2")
        for m in range(wave_nCM):
            for n in range(wave_nCN):
                row = J.lane_id % mfma_MN
                col = J.lane_id // mfma_MN + n * (mfma_MN * J.sizeof_fp32 // J.sizeof_DW4)
                swizzle_col = swizzle(row, col) % (num_lanes_per_row)
                vaddr_w = J.gpr((row) * (wave_size_N * J.sizeof_fp32) + \
                                lds_warp_offset + \
                                (swizzle_col) * J.sizeof_DW4)
                J.ds_write_b128(vaddr_w, mfma_C[m,n], mod=f"offset:{lds_out}")


            voffset = J.gpr((J.lane_id % num_lanes_per_row) * J.sizeof_DW2 + \
                            (J.lane_id // num_lanes_per_row + m*mfma_MN + warp_offset_m) * (N*J.sizeof_bf16) + \
                            (warp_offset_n*J.sizeof_bf16))

            for r in range(0, mfma_MN, rows_per_read):
                row = J.lane_id // num_lanes_per_row + r
                col = J.lane_id % num_lanes_per_row
                swizzle_col = swizzle(row, col) % num_lanes_per_row
                vaddr_r = J.gpr((swizzle_col) * J.sizeof_DW4 + \
                                (row) * (wave_size_N * J.sizeof_fp32) + \
                                lds_warp_offset)
                J.ds_read_b128(vdata, vaddr_r, mod=f"offset:{lds_out}")
                J.s_waitcnt(mod=f"lgkmcnt(0)")
                J.uni_cvt_pk_bf16_f32(vbf16[0], vdata[0], vdata[1])
                J.uni_cvt_pk_bf16_f32(vbf16[1], vdata[2], vdata[3])
                buff_c.store_dwordx2(vbf16[0:1], voffset, 0)
                voffset[0] += rows_per_read * (N*J.sizeof_bf16)        
    else:
        stride_c = N * J.sizeof_fp32
        vaddr = J.gpr(((J.lane_id % 16) + warp_m * 16)*stride_c + ((J.lane_id // 16) + warp_n * 4) * J.sizeof_DW4)
        for m in range(nrM):
            for n in range(nrN):
                J.global_store_dwordx4(vaddr, mfma_C[m,n], pC, mod=f"offset:{n*4*J.sizeof_DW4}")
            vaddr[0] += 16*stride_c

C = torch.zeros(M, N, dtype=torch.bfloat16)

gemm_kernel([blk_cnt],[4*64], N, K, use_pre_shuffle, A.data_ptr(), B.data_ptr(), C.data_ptr(), M)


ref_out = C0
cur_out = C
acc = "pass"
if not torch.allclose(ref_out, cur_out, rtol=0.01, atol=0.01):
    print(f"================= ref_out : {ref_out.shape} ")
    print(ref_out)
    print(f"================= cur_out : {cur_out.shape} ")
    print(cur_out)
    idx = torch.where(torch.abs(ref_out - cur_out) > 0.03)
    if len(idx[0]):
        print(f'idx = {idx}\nref={ref_out[idx]}\ncur={cur_out[idx]}\n{len(idx[0])}')
    #assert 0
    acc = "failed"


DATA_CLONES = 40
As = [torch.clone(A) for _ in range(DATA_CLONES)]
Bs = [torch.clone(B) for _ in range(DATA_CLONES)]
Cs = [torch.clone(C) for _ in range(DATA_CLONES)]

A0s = [torch.clone(A0) for _ in range(DATA_CLONES)]
B0s = [torch.clone(B0) for _ in range(DATA_CLONES)]

di = 0
for i in range(10):
    di = (di + 1)%DATA_CLONES
    with pyhip.cudaPerf(M*N*K*2, (M*K*2+K*N*2), name=f"torch_{di}") as p0:
        ref = torch.nn.functional.linear(A0s[di], B0s[di])

for i in range(10):
    di = (di + 1)%DATA_CLONES
    with pyhip.cudaPerf(M*N*K*2, (M*K*2+K*N*2), name=f"gemm_{di}") as p0:
        gemm_kernel([blk_cnt],[4*64], N, K, use_pre_shuffle,
                    As[di].data_ptr(),
                    Bs[di].data_ptr(),
                    Cs[di].data_ptr(), M)
print(f"{acc=}")