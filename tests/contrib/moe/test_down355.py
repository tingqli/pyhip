import pyhip
from pyhip import calc_diff
from pyhip.contrib.moe_gemm_down_tp import moe_gemm_down_tp
import torch

import contextlib

pyhip.set_device(0)

@pyhip.jit(with_debug_log=False)
def moe_gemm_down_8wave(J, is_output_over_4GB, AB_dtype, wg_M, wg_N,
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
    num_warps = 8
    stride_k = IC * J.sizeof(AB_dtype)

    num_token_topks = J.gpr(num_tokens * TOPK)
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
    mfma_A = J.gpr(nrM, nrK, 4, "vbf16x2")          # 4 b32 regs in MFMA-16 layout : 16x16xfp32/16x32xbf16/16x64xfp8
    mfma_B = J.gpr(nrN, nrK, 4, "vbf16x2")
    mfma_C = J.gpr(nrM, nrN, 4, "vf32")
    mfma_C_bf16 = J.gpr(nrM, nrN, 4, "vbf16x2")

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

    for m in range(nrM):
        vrows[m] = (vrows[m] & 0xFFFFFF) * TOPK + (vrows[m] >> 24)
        with J.ExecMask(vrows[m] < num_token_topks):
            vaddr = J.gpr(2, "vu32")
            vaddr[0] = (J.lane_id // 16) * J.sizeof_DW4 # col_off
            vaddr[1] = 0
            J.v_lshl_add_u64(vaddr, input, 0, vaddr)
            J.v_mad_u64_u32(vaddr, "vcc", vrows[m], J.gpr("su32", stride_k), vaddr)
            for k in range(nrK):
                J.global_load_dwordx4(mfma_A[m, k], vaddr, "off", mod=f"offset:{k*64}")

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

    ldsB = [J.alloc_lds(num_bytes_B) for _ in range(4)]

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
        def mfma():
            for k in range(nrK):
                for m in range(nrM):
                    for n in range(nrN):
                        J.v_mfma_f32_16x16x32_bf16(mfma_C[m, n],
                                                mfma_B[n, k],
                                                mfma_A[m, k],
                                                0 if k == 0 else mfma_C[m, n])
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
        buff_sa = J.Buffer(pScaleA, num_scales_K * num_token_topks * J.sizeof_f32)

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
                vridx[bm] += num_token_topks * J.sizeof_f32

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


        def mfma():
            mfma_C_initialized = {}
            for m in range(nrM):
                for n in range(nrN):
                    mfma_C_initialized[m,n] = 0

            # C & mfma_scaleAB & (m,n)
            dequant_queue = []
            for k in range(0,nrK,2):
                for m in range(nrM):
                    for n in range(nrN):
                        temp = J.gpr(4, "vf32")
                        mfma_scaleAB = J.gpr("vf32", mfma_scaleA[m, k*64//scale_BK] * mfma_scaleB[n*16//scale_BN, k*64//scale_BK])
                        J.v_mfma_f32_16x16x128_f8f6f4(temp,
                                                      mfma_B[n, k:k+1],
                                                      mfma_A[m, k:k+1],
                                                      0)
                        #J.s_nop(4)
                        dequant_queue.append([temp, mfma_scaleAB, (m,n)])
                        if len(dequant_queue) > 3:
                            tc, ts, (tm,tn) = dequant_queue.pop(0)
                            if mfma_C_initialized[tm, tn] == 0:
                                mfma_C_initialized[tm, tn] = 1
                                J.v_mul_f32(mfma_C[tm, tn, 0], tc[0], ts)
                                J.v_mul_f32(mfma_C[tm, tn, 1], tc[1], ts)
                                J.v_mul_f32(mfma_C[tm, tn, 2], tc[2], ts)
                                J.v_mul_f32(mfma_C[tm, tn, 3], tc[3], ts)
                            else:
                                J.v_fmac_f32(mfma_C[tm, tn, 0], tc[0], ts)
                                J.v_fmac_f32(mfma_C[tm, tn, 1], tc[1], ts)
                                J.v_fmac_f32(mfma_C[tm, tn, 2], tc[2], ts)
                                J.v_fmac_f32(mfma_C[tm, tn, 3], tc[3], ts)
                        yield 16

            while len(dequant_queue):
                tc, ts, (tm,tn) = dequant_queue.pop(0)
                J.v_fmac_f32(mfma_C[tm, tn, 0], tc[0], ts)
                J.v_fmac_f32(mfma_C[tm, tn, 1], tc[1], ts)
                J.v_fmac_f32(mfma_C[tm, tn, 2], tc[2], ts)
                J.v_fmac_f32(mfma_C[tm, tn, 3], tc[3], ts)

            for m in range(nrM):
                for n in range(0, nrN, 2):
                    for i in range(4):
                        J.v_mul_f32(mfma_C[m,n,i], mfma_C[m,n,i], vweights[m])
                    for i in range(4):
                        J.v_mul_f32(mfma_C[m,n+1,i], mfma_C[m,n+1,i], vweights[m])

                    J.uni_cvt_pk_bf16_f32(mfma_C_bf16[m,n,0], mfma_C[ m,n,0], mfma_C[ m,n,1])
                    J.uni_cvt_pk_bf16_f32(mfma_C_bf16[m,n,1], mfma_C[ m,n,2], mfma_C[ m,n,3])
                    J.uni_cvt_pk_bf16_f32(mfma_C_bf16[m,n,2], mfma_C[ m,n+1,0], mfma_C[ m,n+1,1])
                    J.uni_cvt_pk_bf16_f32(mfma_C_bf16[m,n,3], mfma_C[ m,n+1,2], mfma_C[ m,n+1,3])
                    J.v_permlane16_swap_b32(mfma_C_bf16[m,n,0], mfma_C_bf16[m,n,2])
                    J.v_permlane16_swap_b32(mfma_C_bf16[m,n,1], mfma_C_bf16[m,n,3])

    # prepare output offsets
    stride_c = OC * J.sizeof(C_dtype)
    buff_c = J.Buffer(output, num_token_topks * stride_c)
    vaddr_rows = J.gpr(nrM, 2, "vu32")
    for m in range(nrM):
        if is_output_over_4GB:
            J.v_mad_u64_u32(vaddr_rows[m], "vcc", vrows[m], J.gpr("su32", stride_c), 0)
            J.v_lshl_add_u64(vaddr_rows[m], output, 0, vaddr_rows[m])
            col = (J.lane_id // 16)
            swap_12_col = (col & 1) * 2 + (col >> 1)
            J.v_lshl_add_u64(vaddr_rows[m], J.gpr(2, "vu32", swap_12_col*J.sizeof_DW4, 0), 0, vaddr_rows[m])
        else:
            row_off = vrows[m] * stride_c
            col = (J.lane_id // 16)
            swap_12_col = (col & 1) * 2 + (col >> 1)
            vaddr_rows[m,0] = row_off + swap_12_col * J.sizeof_DW4

    num_vm_stores = nrM * (nrN//2)
    def storeC(block_n):
        soffset = J.gpr("su32", block_n*wg_N*J.sizeof(C_dtype))
        for m in range(nrM):
            with J.ExecMask(vrows[m] < num_token_topks, early_skip=False) if is_output_over_4GB else contextlib.nullcontext():
                for n in range(0, nrN, 2):
                    if is_output_over_4GB:
                        vaddr_64bits = J.gpr(2, "vu32", soffset[0], 0)
                        J.v_lshl_add_u64(vaddr_64bits, vaddr_64bits, 0, vaddr_rows[m])
                        J.global_store_dwordx4(vaddr_64bits, mfma_C_bf16[m,n], "off", mod=f"offset:{n*16*J.sizeof(C_dtype)}")
                    else:
                        buff_c.store_dwordx4(mfma_C_bf16[m,n], vaddr_rows[m,0], soffset, offset12 = n*16*J.sizeof(C_dtype))
                    yield 64

    # perf-experiment
    loop_cnt = J.div(OC, wg_N)

    num_mfmas = nrK * nrM * nrN
    # print(f"{num_vm_loads=} {num_vm_stores=} {num_mfmas=} {loop_cnt=}")
    """
    ----------------------------------------------------: conditional-displacement-barrier

    global_read     B0         |  global_read     B0
    global_read     B1         |  global_read     B1

    
    wave-0123                  |  wave-4567 (with extra initial barrirer)
                               |
    ----------------------------BBBBBBBBBBBBBBBBBBBBBBBB  wait B0 in LDS (BBBB... is extra initial barrirer)
    ds_read         B0         |
    global_read     B2         |
    ----------------------------------------------------  wait ds_read finished
    compute A,B0,Ctop0         | ds_read         B0
                               | global_read     B2

                          Loop body
    ----------------------------------------------------  wait B1 in LDS
    ds_read         B1         |
    global_store    Ctop0      | compute A,B0,Cbut0
    global_read     B3         |
    ----------------------------------------------------  wait ds_read finished
    compute A,B1,Ctop1         | ds_read         B1
                               | global_store    Cbut0
                               | global_read     B3
    ----------------------------------------------------  wait B2 in LDS
    ds_read         B2         | compute A,B1,Cbut1
    global_store    Ctop1      |
    global_read     B4         |
    ---------------------------------------------------- wait ds_read finished
    compute A,B2,Ctop2         | ds_read         B2
                               | global_store    Cbut1
                               | global_read     B4
    ---------------------------------------------------- wait B3 in LDS
    """
    def ds_readB(block_n):
        for n in range(nrN):
            for k in range(nrK):
                ds_read_B(ldsB[block_n%len(ldsB)], n, k)
        ds_read_scaleB(block_n)

    def global_readB(block_n):
        J.emit(vm_load_B(ldsB[block_n%len(ldsB)], block_n*num_bytes_B))

    def compute(block_n):
        J.emit(mfma())

    def global_store(block_n):
        J.emit(storeC(block_n))

    """
    # reference 
    for loop_n in range(loop_cnt):
        global_readB(loop_n)
        J.s_waitcnt(mod=f"vmcnt(0)")
        J.s_barrier()

        ds_readB(loop_n)
        J.s_waitcnt(mod=f"lgkmcnt(0)")
        J.s_barrier()
        compute(loop_n)

        global_store(loop_n)
        J.s_waitcnt(mod=f"vmcnt(0)")
        J.s_barrier()
    """
    global_readB(0)
    global_readB(1)
    J.s_waitcnt(mod=f"vmcnt({num_vm_loads})")

    # (BBBB... is extra initial barrirer for wave 4567)
    with J.If(J.warp_id[0] > 3): J.s_barrier()

    J.s_barrier()
    ds_readB(0)
    global_readB(2)

    J.s_waitcnt(mod=f"lgkmcnt(0) vmcnt({num_vm_loads})")
    J.s_barrier()
    compute(0)

    for loop_n in range(loop_cnt - 3):
        J.s_barrier()
        ds_readB(loop_n + 1)
        global_store(loop_n)
        global_readB(loop_n + 3)

        J.s_waitcnt(mod=f"lgkmcnt(0) vmcnt({num_vm_loads + num_vm_stores})")
        J.s_barrier()
        compute(loop_n + 1)

    J.s_barrier()
    ds_readB(loop_cnt - 2)
    global_store(loop_cnt - 3)

    J.s_waitcnt(mod=f"lgkmcnt(0) vmcnt({num_vm_stores})")
    J.s_barrier()
    compute(loop_cnt - 2)

    J.s_barrier()
    ds_readB(loop_cnt - 1)
    global_store(loop_cnt - 2)
    J.s_waitcnt(mod=f"lgkmcnt(0)")

    compute(loop_cnt - 1)
    global_store(loop_cnt - 1)

    # (BBBB... extra initial barrirer for wave 0123)
    with J.If(J.warp_id[0] < 4): J.s_barrier()


def check_diff(ret1, ret2):
    print(ret1.shape)
    print(ret2.shape)
    cnt = 0
    for b in range(ret1.shape[0]):
        for t in range(ret1.shape[1]):
            d = calc_diff(ret1[b,t], ret2[b,t])
            if d > 0:
                print(f"============ {b},{t} {d}")
                print(ret1[b,t,48:])
                print(ret2[b,t,48:])
                assert pyhip.allclose(ret1[b,t], ret2[b,t])
                cnt += 1
                assert cnt < 2

def test_down(target_file):

    args_dict = torch.load(target_file)
    num_e_blocks = args_dict["num_e_blocks"]
    is_output_over_4GB = args_dict["is_output_over_4GB"]
    AB_dtype = args_dict["AB_dtype"]
    wg_M = args_dict["wg_M"]
    E = args_dict["E"]
    model_dim = args_dict["model_dim"]
    inter_dim = args_dict["inter_dim"]
    w2_is_shuffled = args_dict["w2_is_shuffled"]
    topk = args_dict["topk"]
    sorted_ids = args_dict["sorted_ids"]
    sorted_weights = args_dict["sorted_weights"]
    sorted_expert_ids = args_dict["sorted_expert_ids"]
    num_valid_ids = args_dict["num_valid_ids"]
    w2 = args_dict["w2"]
    w2_scale = args_dict["w2_scale"]
    a2 = args_dict["a2"]
    a2_scale = args_dict["a2_scale"]
    stage2_out = args_dict["stage2_out"]
    token_num = args_dict["token_num"]

    ref_new = stage2_out
    if 0:
        ref_new = torch.zeros_like(stage2_out)
        #w2[...] = 0.125
        moe_gemm_down_tp([1, num_e_blocks], [4*64],
                        is_output_over_4GB,
                        AB_dtype, wg_M, 64,
                        E, model_dim, inter_dim, 
                        False, w2_is_shuffled, topk,
                        sorted_ids.data_ptr(),
                        sorted_weights.data_ptr(),
                        sorted_expert_ids.data_ptr(),
                        num_valid_ids.data_ptr(),
                        w2.data_ptr(), None if w2_scale is None else w2_scale.data_ptr(),
                        a2.data_ptr(), None if a2_scale is None else a2_scale.data_ptr(),
                        ref_new.data_ptr(),
                        token_num)

        #print("ref_new: ", calc_diff(ref, ref_new))
    if 0:
        ref_new[...] = 0
        moe_gemm_down_8wave([1, num_e_blocks], [8*64],
                        True,
                        is_output_over_4GB,
                        AB_dtype, wg_M, 64,
                        E, model_dim, inter_dim, 
                        False, w2_is_shuffled, topk,
                        sorted_ids.data_ptr(),
                        sorted_weights.data_ptr(),
                        sorted_expert_ids.data_ptr(),
                        num_valid_ids.data_ptr(),
                        w2.data_ptr(), None if w2_scale is None else w2_scale.data_ptr(),
                        a2.data_ptr(), None if a2_scale is None else a2_scale.data_ptr(),
                        ref_new.data_ptr(),
                        token_num)

    ret0 = ref_new
    for k in range(10):
        ret1 = torch.zeros_like(ref_new)
        with pyhip.cudaPerf():
            moe_gemm_down_8wave([1, num_e_blocks], [8*64],
                            is_output_over_4GB,
                            AB_dtype, wg_M, 64,
                            E, model_dim, inter_dim, 
                            False, w2_is_shuffled, topk,
                            sorted_ids.data_ptr(),
                            sorted_weights.data_ptr(),
                            sorted_expert_ids.data_ptr(),
                            num_valid_ids.data_ptr(),
                            w2.data_ptr(), None if w2_scale is None else w2_scale.data_ptr(),
                            a2.data_ptr(), None if a2_scale is None else a2_scale.data_ptr(),
                            ret1.data_ptr(),
                            token_num)
        print(k, calc_diff(ref_new, ret1), calc_diff(ret0, ret1))

        #check_diff(ref_new, ret1)
        #assert 0
        ret0 = ret1

        #print("ret2 vs  ref: ", calc_diff(ref, ret2))
        #print("ret1 vs ret2: ", calc_diff(ret1, ret2))

    pyhip.run_perftest(moe_gemm_down_8wave, [1, num_e_blocks], [8*64],
                            is_output_over_4GB,
                            AB_dtype, wg_M, 64,
                            E, model_dim, inter_dim, 
                            False, w2_is_shuffled, topk,
                            sorted_ids.data_ptr(),
                            sorted_weights.data_ptr(),
                            sorted_expert_ids.data_ptr(),
                            num_valid_ids.data_ptr(),
                            w2.data_ptr(), None if w2_scale is None else w2_scale.data_ptr(),
                            a2.data_ptr(), None if a2_scale is None else a2_scale.data_ptr(),
                            ret1.data_ptr(),
                            token_num,
                            num_verbose=1)

test_down("moe_gemm_down_16384_256_6144_256_True.pt")