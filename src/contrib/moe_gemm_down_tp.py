from pyhip import jit, JIT
import torch

from .common.loaders import get_mfma_loader, get_mfma_loader_sorted_tok

__all__ = [
    "moe_gemm_down_tp"
]

"""
moe gemm down-proj in Tensor-Parallel case is special : IC dimension is extreamly small, which turns the problem into mem-bound
pipeline hides MFMA into load-mfma_B-Matrix & store-mfma_C-matrix:
    - vm-load LDS1
    - ds_read B0 from LDS0
    - store C0
    - mfma mfma_A B0 C1
to save registers, ds_read & MFMA are serialized (combined into a single stage, since it's mem-bound anyway)

4-warps divide problem along M dimension

rules:
    1. experiment HW capabilities/characteristic to determine the projection & guide the optimization decisions
        forget the accuracy.
        just load-mfma_B & store-mfma_C continously.
        how fast it can be?

    2. step-by-step, using simplest pipeline to ensure accracy/functioning of basic building-blocks
        load-mfma_A directly from vmem into register
        load-mfma_B cooperatively
        MFMA-ABC
        store-mfma_C

"""

@jit(with_debug_log=False)
def moe_gemm_down_tp(J, AB_dtype, wg_M, wg_N,
                   NUM_EXPERTS, OC, IC, 
                   gate_up, bpreshuffle, TOPK,
                   sorted_ids:"uint*",
                   sorted_weights:"float*",
                   sorted_expert_ids:"uint*",
                   num_valid_ids:"uint*",
                   weight:"void*",pScaleB:"void*",
                   input:"void*", pScaleA:"void*",
                   output:"void*",
                   num_tokens:"uint"):
    C_dtype = "bf16"
    assert AB_dtype in ["fp8", "bf16"]
    assert C_dtype == "bf16"

    assert gate_up == False
    num_warps = 4
    stride_k = IC * J.sizeof(AB_dtype)

    # all 4 warps distributed in 4x1
    # there is no share of mfma_A matrix, each warp loads directly their own part from VMEM

    # load expert_id
    blk_n = J.blockIdx.x # split along OC
    blk_m = J.blockIdx.y #; blk_m[0] *= 0
    expert_id = J.gpr(1, 'su32')
    J.s_load_dword(expert_id, sorted_expert_ids, blk_m[0] * J.sizeof_u32)
    max_id = J.gpr(1, 'su32')
    J.s_load_dword(max_id, num_valid_ids, 0)

    warp_M = J.div(wg_M, num_warps)
    sorted_ids[:] += blk_m * (wg_M * J.sizeof_u32)
    sorted_weights[:] += blk_m * (wg_M * J.sizeof_u32)

    J.s_waitcnt(mod=f"lgkmcnt(0)")

    J.debug_setup((blk_m[0] == 0) & (blk_n[0] == 0) & (J.warp_id[0] == 0))
    with J.If(blk_m[0] * wg_M >= max_id[0]):
        J.s_endpgm()
    
    # prefetch sorted ids & weights into LDS
    lds_sorted_ids = J.alloc_lds(wg_M * J.sizeof_u32)
    lds_sorted_weights = J.alloc_lds(wg_M * J.sizeof_DW)
    J.wg_load_lds(lds_sorted_ids, sorted_ids, wg_M * J.sizeof_u32, num_warps = num_warps, wait_barrier = True)
    J.wg_load_lds(lds_sorted_weights, sorted_weights, wg_M * J.sizeof_f32, num_warps = num_warps, wait_barrier = True)

    nrM = J.div(warp_M, 16)     # 4 @ wg_M=256
    nrN = J.div(wg_N, 16)       # 4 @ wg_N=64
    nrK = J.div_up(IC*J.sizeof(AB_dtype), 64)     # always use 64-bytes in K dims (due to dwordx4/b128)
    mfma_A = J.gpr(nrM, nrK, 4, "abf16x2")          # 4 b32 regs in MFMA-16 layout : 16x16xfp32/16x32xbf16/16x64xfp8
    mfma_B = J.gpr(nrN, nrK, 4, "abf16x2")
    mfma_C = J.gpr(2, nrM, nrN, 4, "vf32")          # mfma_C is ping-pong buffered for storing output  

    if AB_dtype == "fp8":
        assert nrK >= 2 and (nrK % 2) == 0, f"v_mfma_f32_16x16x128_f8f6f4 needs K=128, but {nrK=}"

    # load whole mfma_A matrix into register & reuse them for different output tiles
    warp_offset_m = J.warp_id[0] * (warp_M * J.sizeof_u32)
    vrows = J.gpr(nrM, "vu32")
    vweights = J.gpr(nrM, "vu32")
    row = J.gpr("vu32", (J.lane_id % 16) * J.sizeof_u32 + warp_offset_m)
    for m in range(nrM):
        J.ds_read_b32(vrows[m], row, mod=f"offset:{lds_sorted_ids}")
        J.ds_read_b32(vweights[m], row, mod=f"offset:{lds_sorted_weights}")
        row[0] += 16*J.sizeof_u32

    J.s_waitcnt(mod=f"lgkmcnt(0)")

    buff_a = J.Buffer(input, num_tokens * TOPK * stride_k)
    for m in range(nrM):
        row_off = (vrows[m] & 0xFFFFFF) * (TOPK * stride_k) + (vrows[m] >> 24) * (stride_k)
        col_off = (J.lane_id // 16) * J.sizeof_DW4
        vaddr = J.gpr("vu32", row_off + col_off)
        for k in range(nrK):
            buff_a.load_dwordx4(mfma_A[m, k], vaddr, 0, offset12=k*64)
    J.s_waitcnt(mod=f"vmcnt(0)")

    # wait before first use
    #J.s_waitcnt(mod=f"vmcnt(0)")

    # vm_load_b, vm_load_cnt_b, vm_offset_inc_b, ds_read_b = get_mfma_loader(J, bpreshuffle, num_warps, wg_N, BLOCK_K, stride_k, 0)
    assert bpreshuffle
    # load mfma_B-tile [wg_N//16, ICb//64, 16, 64] into LDS : 4-warps in mem-coalescing way
    # since bpreshuffle is True, each DW4-vmload loads a 16x64xbytes tile, 
    num_16x64b_wg_N = J.div(wg_N, 16)                     # 4 @ wg_N=64
    num_16x64b_K = J.div_up(IC * J.sizeof(AB_dtype), 64)  # 4 @ IC=128xbf16
    num_bytes_B = num_16x64b_wg_N * num_16x64b_K * 16*64

    ldsB = [J.alloc_lds(num_bytes_B),
            J.alloc_lds(num_bytes_B)]

    num_vm_loads = J.div(num_16x64b_wg_N * num_16x64b_K, num_warps) # 4 @ num_warps=4
    vm_load_voff = J.gpr("vu32", J.threadIdx.x[0] * J.sizeof_DW4)
    lds_warp_off = J.gpr("su32", J.warp_id[0] * (64*J.sizeof_DW4))

    weight[:] += expert_id * (OC * stride_k)
    buff_b = J.Buffer(weight, OC * stride_k)
    
    def vm_load_B(lds, vm_offset):
        J.s_mov_b32("m0", lds_warp_off + lds)
        voff = J.gpr("vu32", vm_load_voff + vm_offset)
        for i in range(num_vm_loads):
            yield 1
            buff_b.load_dwordx4(None, voff, 0, offset12=0)
            J.s_addk_i32("m0", num_warps*64*J.sizeof_DW4)
            voff[0] += num_warps*64*J.sizeof_DW4

    voff = J.gpr(J.lane_id[0] * J.sizeof_DW4)
    voff2 = J.gpr("vu32", voff[0] + 64*1024)
    def ds_read_B(lds, n, k):
        assert k >=0 and k < num_16x64b_K
        assert n >=0 and n < num_16x64b_wg_N
        offset = lds + n*(num_16x64b_K * 1024) + k*1024
        if offset >= 64*1024:
            voffset = voff2
            offset -= 64*1024
        else:
            voffset = voff
        J.ds_read_b128(mfma_B[n, k], voffset, mod=f"offset:{offset}")

    if AB_dtype == "bf16":
        def mfma(c_index):
            for k in range(nrK):
                for m in range(nrM):
                    for n in range(nrN):
                        J.v_mfma_f32_16x16x32_bf16(mfma_C[c_index, m, n],
                                                mfma_B[n, k],
                                                mfma_A[m, k],
                                                0 if k == 0 else mfma_C[c_index, m, n])
                        yield 16
        def ds_read_scaleB(idx_wgN):
            pass
    else:
        assert AB_dtype == "fp8"
        scale_BM, scale_BN, scale_BK = 1,128,128 
        # due to mem-bound, we dequant directly after MFMA
        num_scales_K = J.div(IC, scale_BK)
        nrsN = J.div(wg_N, scale_BN) if wg_N >= scale_BN else 1
        mfma_scaleA = J.gpr(nrM, num_scales_K, "vf32")
        mfma_scaleB = J.gpr(nrsN, num_scales_K, "vf32")

        # since IC is small, load all scaleA into registers
        # expecting scaleA to be in [k,m] layout
        buff_sa = J.Buffer(pScaleA, num_scales_K * (num_tokens[0] * TOPK) * J.sizeof_f32)

        vridx = J.gpr(nrM, "vu32")
        for bm in range(nrM):
            _voff = J.gpr("vu32")
            _voff[0] = J.gpr("vu32", J.warp_id[0] * (warp_M * J.sizeof_u32) + \
                      (bm * 16 * J.sizeof_u32) + \
                      (J.lane_id[0] % 16) * J.sizeof_u32)
            J.ds_read_b32(vridx[bm], _voff, mod=f"offset:{lds_sorted_ids}")

        J.s_waitcnt(mod=f"lgkmcnt(0)")

        for bm in range(nrM):
            vridx[bm] = (vridx[bm] & 0xFFFFFF) * (TOPK * J.sizeof_f32) + \
                        (vridx[bm] >> 24) * J.sizeof_f32

        for bk in range(num_scales_K):
            for bm in range(nrM):
                buff_sa.load_dword(mfma_scaleA[bm, bk], vridx[bm], 0, offset12=0)
                vridx[bm] += num_tokens[0] * TOPK * J.sizeof_f32

        # rely on load-B-scales to do the vm_wait & sync
        J.s_waitcnt(mod=f"vmcnt({0})")
        J.s_barrier()

        # since IC is small, load all B scales into LDS: 
        sizeof_scaleB = J.div(OC, scale_BN) * J.div(IC, scale_BK) * J.sizeof_f32
        pScaleB[:] += expert_id * sizeof_scaleB
        lds_scaleB = J.alloc_lds(sizeof_scaleB)
        J.wg_load_lds(lds_scaleB, pScaleB, sizeof_scaleB, num_warps, wait_barrier = True)

        def ds_read_scaleB(idx_wgN):
            # each scale is broadcasted to all lanes
            assert wg_N <= scale_BN
            bn_wgN = (idx_wgN * wg_N // scale_BN)
            for bn in range(nrsN):
                for bk in range(num_scales_K):
                    vaddr = J.gpr("vu32", lds_scaleB + (bn + bn_wgN)*J.div(IC, scale_BK)*J.sizeof_f32 + bk * J.sizeof_f32)
                    J.ds_read_b32(mfma_scaleB[bn, bk], vaddr)

        def mfma(c_index):
            for m in range(nrM):
                for n in range(nrN):
                    mfma_C[c_index,m,n,0] = 0
                    mfma_C[c_index,m,n,1] = 0
                    mfma_C[c_index,m,n,2] = 0
                    mfma_C[c_index,m,n,3] = 0
            
            for k in range(0,nrK,2):
                for m in range(nrM):
                    for n in range(nrN):
                        mfma_scaleAB = J.gpr("vf32", mfma_scaleA[m, k*64//scale_BK] * mfma_scaleB[n*16//scale_BN, k*64//scale_BK])
                        temp = J.gpr(4, "vf32")
                        J.v_mfma_f32_16x16x128_f8f6f4(temp,
                                                      mfma_B[n, k:k+1],
                                                      mfma_A[m, k:k+1],
                                                      0)
                        J.s_nop(15)
                        J.v_fmac_f32(mfma_C[c_index, m, n, 0], temp[0], mfma_scaleAB)
                        J.v_fmac_f32(mfma_C[c_index, m, n, 1], temp[1], mfma_scaleAB)
                        J.v_fmac_f32(mfma_C[c_index, m, n, 2], temp[2], mfma_scaleAB)
                        J.v_fmac_f32(mfma_C[c_index, m, n, 3], temp[3], mfma_scaleAB)
                        yield 16
            # mfma_B = J.gpr(nrN, nrK, 4, "abf16x2")
            #J.debug_log(mfma_A[0], torch.float8_e4m3fn)
            #J.debug_log(mfma_B[0], torch.float8_e4m3fn)
            #J.debug_log(mfma_A[0, 0], torch.float8_e4m3fn)
            #J.debug_log(mfma_C[c_index, 0, 0], torch.float)
            #J.debug_log(mfma_scaleAB, torch.float)
            #J.s_endpgm()
            
    # prepare output offsets
    stride_c = OC * J.sizeof(C_dtype)
    buff_c = J.Buffer(output, num_tokens * TOPK * stride_c)
    for m in range(nrM):
        row_off = (vrows[m] & 0xFFFFFF) * (TOPK * stride_c) + (vrows[m] >> 24) * (stride_c)
        col = (J.lane_id // 16)
        swap_12_col = (col & 1) * 2 + (col >> 1)
        vrows[m] = row_off + swap_12_col * J.sizeof_DW4

    num_vm_stores = nrM * (nrN//2)
    def storeC(c_index, soffset):
        vbf16 = J.gpr(4, "vbf16x2")
        for m in range(nrM):
            for n in range(0, nrN, 2):
                for i in range(4):
                    J.v_mul_f32(mfma_C[c_index,m,n,i], mfma_C[c_index,m,n,i], vweights[m])
                yield 16
                for i in range(4):
                    J.v_mul_f32(mfma_C[c_index,m,n+1,i], mfma_C[c_index,m,n+1,i], vweights[m])
                yield 16
                J.uni_cvt_pk_bf16_f32(vbf16[0], mfma_C[c_index, m,n,0], mfma_C[c_index, m,n,1])
                J.uni_cvt_pk_bf16_f32(vbf16[1], mfma_C[c_index, m,n,2], mfma_C[c_index, m,n,3])
                J.uni_cvt_pk_bf16_f32(vbf16[2], mfma_C[c_index, m,n+1,0], mfma_C[c_index, m,n+1,1])
                J.uni_cvt_pk_bf16_f32(vbf16[3], mfma_C[c_index, m,n+1,2], mfma_C[c_index, m,n+1,3])
                yield 16
                J.v_permlane16_swap_b32(vbf16[0], vbf16[2])
                J.v_permlane16_swap_b32(vbf16[1], vbf16[3])
                buff_c.store_dwordx4(vbf16, vrows[m], soffset, offset12 = n*16*J.sizeof(C_dtype))
                yield 64

    # perf-experiment
    loop_cnt = J.div(OC, wg_N)
    if 1:
        # num_vm_loads = 4
        # num_vm_stores = 8
        # num_mfmas=64
        num_mfmas = nrK * nrM * nrN
        print(f"{num_vm_loads=} {num_vm_stores=} {num_mfmas=}")

        J.emit(vm_load_B(ldsB[1], 0*num_bytes_B))

        for loop_n in range(loop_cnt):
            tic = loop_n&1
            toc = tic ^ 1
            J.emit(vm_load_B(ldsB[tic], (loop_n + 1)*num_bytes_B))

            if loop_n > 1:
                J.s_waitcnt(mod=f"vmcnt({num_vm_loads + num_vm_stores})")
            else:
                J.s_waitcnt(mod=f"vmcnt({num_vm_loads})")
            J.s_barrier()

            for n in range(nrN):
                for k in range(nrK):
                    ds_read_B(ldsB[toc], n, k)
            ds_read_scaleB(loop_n)
            J.s_waitcnt(mod=f"lgkmcnt({0})")
            J.s_barrier()

            compute = mfma(toc)

            # store mfma_C interleaving with MFMA
            if loop_n > 0:
                soffset = J.gpr("su32", (loop_n-1)*wg_N*J.sizeof(C_dtype))
                storer_C = storeC(tic, soffset)
                while True:
                    e1 = J.emit(compute, 64)
                    e0 = J.emit(storer_C, 32)
                    if not e0:
                        break

            J.emit(compute)

        soffset = J.gpr("su32", (loop_cnt-1)*wg_N*J.sizeof(C_dtype))
        J.emit(storeC(toc, soffset))
    else:
        for loop_n in range(loop_cnt):
            J.emit(vm_load_B(ldsB[0], loop_n*num_bytes_B))
            J.s_waitcnt(mod=f"vmcnt({0})")
            J.s_barrier()

            # we only load B once, so with B-tile ready in LDS
            # we would like to load all A
            for n in range(nrN):
                for k in range(nrK):
                    ds_read_B(ldsB[0], n, k)
            ds_read_scaleB(loop_n)
            J.s_waitcnt(mod=f"lgkmcnt({0})")
            J.s_barrier()

            J.emit(mfma(0))

            #J.debug_log(mfma_C[0,0,0], torch.float, "4h.16v.4h")

            # store mfma_C
            soffset = J.gpr("su32", loop_n*wg_N*J.sizeof(C_dtype))
            J.emit(storeC(0, soffset))

