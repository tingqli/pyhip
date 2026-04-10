from pyhip import jit, JIT
import torch

from .common.loaders import get_mfma_loader, get_mfma_loader_sorted_tok

__all__ = [
    "moe_gemm_8wave_gelu",
]

"""

While JIT code offers flexibility, it tends to become very messy when we mixing too many functionalities into the generator.

Special 8-wave version for GeLU, non-gate-up structure:

    up = gemm_up(x)
    act = GELU(up)
    down = gemm_down(act)

support bf16/int8_PTPC/mxfp4

"""

@jit(with_debug_log=False)
def moe_gemm_8wave_gelu(J, is_input_over_4GB,
                   AB_dtype, wg_M, wg_N,
                   NUM_EXPERTS, OC, IC, 
                   gate_up, bpreshuffle, TOPK,
                   sorted_ids:"uint*",
                   sorted_weights:"float*",
                   sorted_expert_ids:"uint*",
                   num_valid_ids:"uint*",
                   weight:"void*",pScaleB:"void*",
                   input:"void*", pScaleA:"void*",
                   output:"void*",
                   num_tokens:"uint",
                   num_blocks:"uint"):
    num_warps = 8

    assert not is_input_over_4GB

    assert AB_dtype in ["bf16", "s8"]
    C_dtype = "bf16"

    K = IC
    # loader always load 128bytes (8 x DW4-lanes) along K dimension
    wg_K = J.div(128, J.sizeof(AB_dtype))

    stride_k = IC * J.sizeof(AB_dtype)
    #stride_gate_up = J.div(J.div(OC, wg_N), 2) * wg_N * stride_k

    blk1d = J.blockIdx.x

    # map 1d index to e_block_id & oc_block_id
    # in unit of NUM_CU=32x8, the remainder part is evenly distributed
    """
       cu_id 0  1  2  3  4  5  6  7 | 8  9 10 11 12 13 14 15 |
      xcd_id 0  1  2  3  4  5  6  7 | 0  1  2  3  4  5  6  7 |
      xcd_cu 0                      | 1  1  1  1  1  1  1  1
             0 32 64      . . . . . | 1  33 65  .. .... ...| 

    """
    NUM_CU = 256
    num_oc_blocks = J.div(OC, wg_N)
    num_groupped_blocks = num_blocks - (num_blocks % NUM_CU)
    blk_m = J.gpr("su32")
    blk_n = J.gpr("su32")
    with J.If(blk1d < num_groupped_blocks) as If:
        blk_base = (blk1d // NUM_CU) * NUM_CU
        cu_id = blk1d % NUM_CU
        xcd_id = cu_id % 8
        xcd_cu = cu_id // 8
        task_id = xcd_id * 32 + xcd_cu
        new_blk1d = blk_base + task_id
        blk_m[0] = new_blk1d // num_oc_blocks
        blk_n[0] = new_blk1d - blk_m * num_oc_blocks

        If.Else()
        blk_m[0] = blk1d // num_oc_blocks
        blk_n[0] = blk1d - blk_m * num_oc_blocks

    #blk_n = J.blockIdx.x # split along OC
    #blk_m = J.blockIdx.y
    #blk_m[0] *= 0
    #blk_n[0] *= 0
    expert_id = J.gpr(1, 'su32')
    J.s_load_dword(expert_id, sorted_expert_ids, blk_m[0] * J.sizeof_u32)
    max_id = J.gpr(1, 'su32')
    J.s_load_dword(max_id, num_valid_ids, 0)
    J.s_waitcnt(mod=f"lgkmcnt(0)")

    J.debug_setup((blk_m[0] == 0) & (blk_n[0] == 0) & (J.warp_id[0] == 0))
    with J.If(blk_m[0] * wg_M >= max_id[0]):
        J.s_endpgm()

    sorted_ids[:] += blk_m * (wg_M * J.sizeof_u32)
    sorted_weights[:] += blk_m * (wg_M * J.sizeof_u32)

    #i_scale[:] += blk_m * (J.div(wg_M,32) * stride_scale32x256)
    #w_scale[:] += expert_id * (J.div(OC,32) * stride_scale32x256)
    #i_scale[:] += (J.warp_id[0] // 2) * (J.div(wg_M//2, 32) * stride_scale32x256)

    # basic configuration for 8-wave
    WARPS_COL = 4
    WARPS_ROW = 2
    BLOCK_SIZE_ROW = wg_M
    BLOCK_SIZE_COL = wg_N
    BLOCK_K = 128 # in bytes
    HALF_BLOCK_SIZE_ROW = J.div(BLOCK_SIZE_ROW, 2)
    HALF_BLOCK_SIZE_COL = J.div(BLOCK_SIZE_COL, 2)
    MINI_BLOCK_M = J.div(HALF_BLOCK_SIZE_ROW, WARPS_ROW) # 64
    MINI_BLOCK_N = J.div(HALF_BLOCK_SIZE_COL, WARPS_COL) # 32

    # expert_id[0] = 0

    offset_64bit = J.gpr(2,"su32")
    J.s_mul_hi_u32(offset_64bit[1], expert_id, (OC * stride_k))
    J.s_mul_i32(offset_64bit[0], expert_id, (OC * stride_k))
    weight[:] += offset_64bit

    if gate_up and AB_dtype != "s8":
        """
        weight[:] += expert_id * (OC * stride_k) + blk_n * (wg_N//2 * stride_k)
        w_scale[:] += blk_n * (J.div(wg_N//2, 32) * stride_scale32x256) + (J.warp_id[0] % 2) * (J.div(wg_N//4, 32) * stride_scale32x256)
        # gate-scale buff + up-scale buff
        sbuff_b = [None, None]
        sbuff_b[0] = J.Buffer(w_scale, J.div(wg_N//4, 32) * stride_scale32x256)
        w_scale[:] += J.div(OC//2, 32) * stride_scale32x256
        sbuff_b[1] = J.Buffer(w_scale, J.div(wg_N//4, 32) * stride_scale32x256)
        """

        # B matrix needs to be interleaved by HALF_BLOCK_SIZE_COL
        # vm_load_b(k, m=0) loads from gate-weight
        # vm_load_b(k, m=1) loads from up-weight
        # AB_dtype == "s8" means smooth-quant, input is also of shape [num_tokens, TOPK, dims]
        LOADER_TOPK = 0
        buff_a = J.Buffer(input, num_tokens * stride_k)
    else:
        LOADER_TOPK = TOPK
        buff_a = J.Buffer(input, num_tokens * TOPK * stride_k)

    weight[:] += blk_n * (wg_N * stride_k)
    # w_scale[:] += blk_n * (J.div(wg_N, 32) * stride_scale32x256) + (J.warp_id[0] % 2) * (J.div(wg_N//2, 32) * stride_scale32x256)
    # sbuff_b = J.Buffer(w_scale, J.div(wg_N//2, 32) * stride_scale32x256)
    buff_b = J.Buffer(weight, wg_N * stride_k)
    stride_n = OC * J.sizeof(C_dtype)

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

    nbN = J.div(wg_N, 16)
    nbM = J.div(wg_M, 16)
    nbK = 2 # 2 MFMA 16x16 
    nrM = J.div(nbM, WARPS_ROW, 2) # 4
    nrN = J.div(nbN, WARPS_COL, 2) # 2
    nrK = nbK

    warp_m = J.gpr(J.warp_id[0] // WARPS_COL) # warp row: 0 to 1
    warp_n = J.gpr(J.warp_id[0] % WARPS_COL)  # warp col: 0 to 3

    # prefetch sorted ids into LDS
    lds_sorted_ids = J.alloc_lds(wg_M * J.sizeof_u32)
    lds_sorted_weights = J.alloc_lds(wg_M * J.sizeof_DW)
    if AB_dtype == "s8":
        # load weight per-OC scales into LDS
        lds_pc_scales = J.alloc_lds(wg_N * J.sizeof_DW)
        pScaleB[:] += (expert_id * OC  + blk_n * wg_N) * J.sizeof_DW
        J.wg_load_lds(lds_pc_scales, pScaleB, wg_N * J.sizeof_DW, num_warps = num_warps, wait_barrier = False)

        lds_pt_scales = J.alloc_lds(wg_M * J.sizeof_DW)
        pScaleA[:] += blk_m * (wg_M * J.sizeof_DW)
        J.wg_load_lds(lds_pt_scales, pScaleA, wg_M * J.sizeof_DW, num_warps = num_warps, wait_barrier = False)

    if gate_up:
        J.wg_load_lds(lds_sorted_ids, sorted_ids, wg_M * J.sizeof_u32, num_warps = num_warps, wait_barrier = True)
    else:
        J.wg_load_lds(lds_sorted_ids, sorted_ids, wg_M * J.sizeof_u32, num_warps = num_warps, wait_barrier = False)
        J.wg_load_lds(lds_sorted_weights, sorted_weights, wg_M * J.sizeof_f32, num_warps = num_warps, wait_barrier = True)

    vm_load_a, vm_load_cnt_a, vm_offset_inc_a, ds_read_a = get_mfma_loader_sorted_tok(J, num_warps, BLOCK_SIZE_ROW, BLOCK_K, stride_k, warp_m*MINI_BLOCK_M, lds_sorted_ids, LOADER_TOPK, num_tokens, input, is_input_over_4GB)
    vm_load_b, vm_load_cnt_b, vm_offset_inc_b, ds_read_b = get_mfma_loader(J, bpreshuffle, num_warps, HALF_BLOCK_SIZE_COL, BLOCK_K, stride_k, warp_n*MINI_BLOCK_N)
    vm_load_cnt_a = vm_load_cnt_a // 2

    # use_f32_blockscales_128 = (AB_dtype == "fp8")

    mfma_A = J.gpr(nrM, 2, 4, "vfp8x4")            # 4x[16,128]
    mfma_B = J.gpr(2, nrN, 2, 4, "vfp8x4")            # 2x[16,128]
    mfma_C = J.gpr(4, nrM, nrN, 4, "vf32")      # 4x[4,2]x[16,16]

    if AB_dtype == "bf16":
        def mfma(c_index):
            b_index = c_index % 2
            for k in range(2):
                for m in range(nrM):
                    for n in range(nrN):
                        J.v_mfma_f32_16x16x32_bf16(mfma_C[c_index, m, n], mfma_B[b_index, n, k], mfma_A[m, k], mfma_C[c_index, m, n])
                        yield 16
    elif AB_dtype == "s8":
        def mfma(c_index):
            b_index = c_index % 2
            for k in range(2):
                for m in range(nrM):
                    for n in range(nrN):
                        J.v_mfma_i32_16x16x64_i8(mfma_C[c_index, m, n], mfma_B[b_index, n, k], mfma_A[m, k], mfma_C[c_index, m, n])
                        yield 16
    else:
        assert AB_dtype == "fp16" or AB_dtype == "f16" 
        def mfma(c_index):
            b_index = c_index % 2
            for k in range(2):
                for m in range(nrM):
                    for n in range(nrN):
                        J.v_mfma_f32_16x16x32_f16(mfma_C[c_index, m, n], mfma_B[b_index, n, k], mfma_A[m, k], mfma_C[c_index, m, n])
                        yield 16

    loop_cnt = J.div(K, wg_K)
    #assert HALF_BLOCK_SIZE_ROW == HALF_BLOCK_SIZE_COL

    a_moffset = J.gpr("su32", 0)
    b_moffsets = J.gpr(2, "su32", 0, stride_k * HALF_BLOCK_SIZE_COL)

    def step_k():
        a_moffset[0] += vm_offset_inc_a
        b_moffsets[0] += vm_offset_inc_b
        b_moffsets[1] += vm_offset_inc_b

    def vm_loadA(k, m):
        assert m in [0, 1]
        assert k in [0, 1]
        return vm_load_a(ldsA[k,m], buff_a, a_moffset, half=m)

    def vm_loadB(k, m):
        assert m in [0, 1]
        assert k in [0, 1]
        return vm_load_b(ldsB[k,m], buff_b, b_moffsets[m])

    def ds_readA(k, m):
        for i in range(nrM):
            ds_read_a(ldsA[k,m], mfma_A[i, 0], i, 0)
            ds_read_a(ldsA[k,m], mfma_A[i, 1], i, 1)

    def ds_readB(k, m):
        for i in range(nrN):
            ds_read_b(ldsB[k,m], mfma_B[m, i, 0], i, 0)
            ds_read_b(ldsB[k,m], mfma_B[m, i, 1], i, 1)

    if 1: 
        # 8-wave pipeline invented by HipKittens
        tic = 0
        toc = 1
        # if use_f32_blockscales_128: vm_load_scaleA(lds_scaleA[tic], 0)
        J.emit(vm_loadB(tic,0))
        J.emit(vm_loadA(tic,0))
        J.emit(vm_loadB(tic,1))
        J.emit(vm_loadA(tic,1))

        with J.If(warp_m[0] == 1):
            J.s_barrier()

        mfma_C[...] = 0

        J.s_waitcnt(mod=f"vmcnt({vm_load_cnt_a + vm_load_cnt_b})"); J.s_barrier()

        step_k()

        # if use_f32_blockscales_128:
        #     vm_load_scaleA(lds_scaleA[toc], 1)
        #     vm_load_cnt_scaleA = 1
        # else:
        #     vm_load_cnt_scaleA = 0
        vm_load_cnt_scaleA = 0
        J.emit(vm_loadA(toc,0))
        J.emit(vm_loadB(toc,0))
        J.emit(vm_loadB(toc,1))

        J.s_waitcnt(mod=f"vmcnt({vm_load_cnt_a + vm_load_cnt_b*2 + vm_load_cnt_scaleA})"); J.s_barrier()

        def loop_body(k, loop_cnt):
            nonlocal tic, toc
            ds_readB(tic, 0)    # lgkmcnt += nrN*2 (2*2)
            ds_readA(tic, 0)    # lgkmcnt += nrM*2 (4*2)

            #if use_f32_blockscales_128:
            #    ds_read_scaleA(lds_scaleA[tic], 0)
            #    ds_read_scaleB(k)

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
            #if use_f32_blockscales_128: ds_read_scaleA(lds_scaleA[tic], 1)
            J.emit(vm_loadB(tic,0))                         # vm_load_cnt_b
            J.s_barrier()

            J.s_waitcnt(mod="lgkmcnt(0)"); J.s_setprio(1)
            J.emit(mfma(2))
            J.s_setprio(0); J.s_barrier()

            J.emit(vm_loadB(tic,1))                         # vm_load_cnt_b
            #if use_f32_blockscales_128: vm_load_scaleA(lds_scaleA[tic], k+2)
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
            # assert not use_f32_blockscales_128, "there is an unknown accuracy issue for f32 blockscale-128 case"
            assert (loop_cnt % 2) == 0
            k = J.gpr("su32", 0)

            with J.While(k[0] < loop_cnt):
                loop_body(k, loop_cnt)
                k[0] += 1
                loop_body(k, loop_cnt)
                k[0] += 1

        J.s_waitcnt(mod="vmcnt(0)")
        #J.s_waitcnt(mod="lgkmcnt(0)")
        with J.If(warp_m[0] == 0):
            J.s_barrier()
    else:
        mfma_C[...] = 0
        for k in range(loop_cnt):
            J.emit(vm_loadB(0,0))
            J.emit(vm_loadA(0,0))
            # if use_f32_blockscales_128: vm_load_scaleA(lds_scaleA[0], k)
            J.s_waitcnt(mod="vmcnt(0)"); J.s_barrier()

            ds_readA(0,0)
            ds_readB(0,0)
            #if use_f32_blockscales_128:
            #    ds_read_scaleA(lds_scaleA[0], 0)
            #    ds_read_scaleB(k)
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
            #if use_f32_blockscales_128:
            #    ds_read_scaleA(lds_scaleA[0], 1)
            J.s_waitcnt(mod="lgkmcnt(0)"); J.s_barrier()

            #J.debug_log(mfma_A[0,0], torch.float8_e4m3fn, "4h.16v.16h")
            #J.debug_log(mfma_A[0,1], torch.float8_e4m3fn, "4h.16v.16h")
            #J.s_endpgm()

            J.emit(mfma(2))
            J.emit(mfma(3))

            step_k()

        J.s_waitcnt(mod="lgkmcnt(0)")
        J.s_waitcnt(mod="vmcnt(0)")


    if AB_dtype == "s8":
        # PTPC dequantize : mfma_C = J.gpr(4, nrM, nrN, 4, "vf32")
        # scales are loaded into LDS lds_pt_scales/lds_pc_scales, load them from LDS is fast & flexible
        for cindex in range(4):
            cm = cindex // 2
            cn = cindex % 2
            # issue load scales
            scale_pt = J.gpr(nrM, "vf32")
            scale_pc = J.gpr(nrN, 4, "vf32")

            ds_vaddr = (J.lane_id % 16 + cm * HALF_BLOCK_SIZE_ROW + warp_m * MINI_BLOCK_M) * J.sizeof_DW + lds_pt_scales
            for m in range(nrM):
                J.ds_read_b32(scale_pt[m], ds_vaddr, mod=f"offset:{m*16*J.sizeof_DW}")

            ds_vaddr = (J.lane_id // 16) * J.sizeof_DW4 + (cn * HALF_BLOCK_SIZE_COL + warp_n * MINI_BLOCK_N)* J.sizeof_DW + lds_pc_scales
            for n in range(nrN):
                J.ds_read_b128(scale_pc[n], ds_vaddr, mod=f"offset:{n*16*J.sizeof_DW}")

            for m in range(nrM):
                for n in range(nrN):
                    J.v_cvt_f32_i32(mfma_C[cindex,m,n,0], mfma_C[cindex,m,n,0])
                    J.v_cvt_f32_i32(mfma_C[cindex,m,n,1], mfma_C[cindex,m,n,1])
                    J.v_cvt_f32_i32(mfma_C[cindex,m,n,2], mfma_C[cindex,m,n,2])
                    J.v_cvt_f32_i32(mfma_C[cindex,m,n,3], mfma_C[cindex,m,n,3])

            J.s_waitcnt(mod="lgkmcnt(0)")

            # dequant
            scale_ptpc = J.gpr(nrN, 4, "vf32")
            for m in range(nrM):
                for n in range(nrN):
                    scale_ptpc[n, 0] = scale_pt[m] * scale_pc[n, 0]
                    scale_ptpc[n, 1] = scale_pt[m] * scale_pc[n, 1]
                    scale_ptpc[n, 2] = scale_pt[m] * scale_pc[n, 2]
                    scale_ptpc[n, 3] = scale_pt[m] * scale_pc[n, 3]

                for n in range(nrN):
                    mfma_C[cindex,m,n,0] *= scale_ptpc[n, 0]
                    mfma_C[cindex,m,n,1] *= scale_ptpc[n, 1]
                    mfma_C[cindex,m,n,2] *= scale_ptpc[n, 2]
                    mfma_C[cindex,m,n,3] *= scale_ptpc[n, 3]

    if gate_up:
        # gelu(c[0]) gelu(c[1])   64*32
        # gelu(c[2]) gelu(c[3])   64*32
        # convert to bfloat16 and
        # scatter output to : [num_tokens, topk, dims]
        vrows = J.gpr(2, nrM, "vu32")
        for cm in range(2):
            row = J.gpr("vu32", ((J.lane_id % 16) + (cm * HALF_BLOCK_SIZE_ROW) + (warp_m * MINI_BLOCK_M))  * J.sizeof_u32)
            for m in range(nrM):
                J.ds_read_b32(vrows[cm, m], row + lds_sorted_ids)
                row[0] += 16*J.sizeof_u32
        J.s_waitcnt(mod=f"lgkmcnt(0)")

        if nrN == 1:
            vbf16 = J.gpr(2, "vbf16x2") # DWORDx4
            col = J.lane_id // 16 
            vaddr0 = J.gpr("vu32", col * J.sizeof_DW2 + warp_n * 16 * J.sizeof_bf16 + blk_n * (wg_N * J.sizeof(C_dtype)))
        else:
            vbf16 = J.gpr(4, "vbf16x2") # DWORDx4
            col = J.lane_id // 16
            swap_12_col = (col & 1) * 2 + (col >> 1)
            vaddr0 = J.gpr("vu32", swap_12_col * J.sizeof_DW4 + warp_n * nrN * 16 * J.sizeof_bf16 + blk_n * (wg_N * J.sizeof(C_dtype)))

        for cindex in range(4):
            cm = cindex // 2
            cn = cindex % 2
            for m in range(nrM):
                vrows_topk = J.gpr(vrows[cm, m] >> 24)
                with J.ExecMask(vrows_topk < TOPK):
                    # to support (num_tokens * TOPK * stride_n) > 4GB, we can only use global_store_dword
                    vaddr = J.gpr(2, "vu32", output[0], output[1])
                    J.v_lshl_add_u64(vaddr, J.gpr(2, "vu32", vaddr0 + vrows_topk * (stride_n), 0), 0, vaddr)
                    J.v_mad_u64_u32(vaddr, "vcc", (vrows[cm, m] & 0xFFFFFF), J.gpr("vu32", TOPK * stride_n), vaddr)

                    assert nrN in [1, 2], f"{nrN=}"
                    if nrN == 1:
                        n = 0
                        mfma_C[cindex,m,n,0] = J.gelu(mfma_C[cindex,m,n,0])
                        mfma_C[cindex,m,n,1] = J.gelu(mfma_C[cindex,m,n,1])
                        mfma_C[cindex,m,n,2] = J.gelu(mfma_C[cindex,m,n,2])
                        mfma_C[cindex,m,n,3] = J.gelu(mfma_C[cindex,m,n,3])
                        J.uni_cvt_pk_bf16_f32(vbf16[0], mfma_C[cindex, m,n,0], mfma_C[cindex, m,n,1]) 
                        J.uni_cvt_pk_bf16_f32(vbf16[1], mfma_C[cindex, m,n,2], mfma_C[cindex, m,n,3])
                        # buff_c.store_dwordx2(vbf16[0:1], vaddr, 0, offset12 = n*16*J.sizeof(C_dtype) + cn*HALF_BLOCK_SIZE_COL*J.sizeof_bf16)
                        J.global_store_dwordx2(vaddr, vbf16[0:1], "off", mod=f"offset:{n*16*J.sizeof(C_dtype) + cn*HALF_BLOCK_SIZE_COL*J.sizeof_bf16}")
                    else:
                        for n in range(0, nrN, 2):
                            mfma_C[cindex,m,n,0] = J.gelu(mfma_C[cindex,m,n,0])
                            mfma_C[cindex,m,n,1] = J.gelu(mfma_C[cindex,m,n,1])
                            mfma_C[cindex,m,n,2] = J.gelu(mfma_C[cindex,m,n,2])
                            mfma_C[cindex,m,n,3] = J.gelu(mfma_C[cindex,m,n,3])

                            mfma_C[cindex,m,n+1,0] = J.gelu(mfma_C[cindex,m,n+1,0])
                            mfma_C[cindex,m,n+1,1] = J.gelu(mfma_C[cindex,m,n+1,1])
                            mfma_C[cindex,m,n+1,2] = J.gelu(mfma_C[cindex,m,n+1,2])
                            mfma_C[cindex,m,n+1,3] = J.gelu(mfma_C[cindex,m,n+1,3])

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
                            # buff_c.store_dwordx4(vbf16, vaddr, 0, offset12 = n*16*J.sizeof(C_dtype) + cn*HALF_BLOCK_SIZE_COL*J.sizeof_bf16)
                            J.global_store_dwordx4(vaddr, vbf16, "off", mod=f"offset:{0*16*J.sizeof(C_dtype) + cn*HALF_BLOCK_SIZE_COL*J.sizeof_bf16}")

    else:
        # scatter output to : [num_tokens, topk, dims]
        vrows = J.gpr(2, nrM, "vu32")
        vweights = J.gpr(2, nrM, "vf32")
        for cm in range(2): 
            row = J.gpr("vu32", ((J.lane_id % 16) + (cm * HALF_BLOCK_SIZE_ROW) + (warp_m * MINI_BLOCK_M))  * J.sizeof_u32)
            for m in range(nrM):
                J.ds_read_b32(vrows[cm, m], row + lds_sorted_ids)
                J.ds_read_b32(vweights[cm, m], row + lds_sorted_weights)
                row[0] += 16*J.sizeof_u32

        J.s_waitcnt(mod=f"lgkmcnt(0)")

        stride_c = OC * J.sizeof(C_dtype)
        if nrN == 1:
            vbf16 = J.gpr(2, "vbf16x2") # DWORDx4
            col = J.lane_id // 16
            vaddr0 = J.gpr("vu32", col * J.sizeof_DW2 + warp_n * 16 * J.sizeof_bf16 + blk_n * (wg_N * J.sizeof(C_dtype)))
        else:
            vbf16 = J.gpr(4, "vbf16x2") # DWORDx4
            col = J.lane_id // 16
            swap_12_col = (col & 1) * 2 + (col >> 1)
            vaddr0 = J.gpr("vu32", swap_12_col * J.sizeof_DW4 + warp_n * 32 * J.sizeof_bf16 + blk_n * (wg_N * J.sizeof(C_dtype)))

        for cindex in range(4):
            cm = cindex // 2
            cn = cindex % 2
            for m in range(nrM):
                vrows_topk = J.gpr(vrows[cm, m] >> 24)
                with J.ExecMask(vrows_topk < TOPK):
                    vaddr = J.gpr(2, "vu32", output[0], output[1])
                    J.v_lshl_add_u64(vaddr, J.gpr(2, "vu32", vaddr0 + vrows_topk * (stride_c), 0), 0, vaddr)
                    J.v_mad_u64_u32(vaddr, "vcc", (vrows[cm, m] & 0xFFFFFF), J.gpr("vu32", TOPK * stride_c), vaddr)                

                    assert nrN in [1, 2], f"{nrN=}"
                    if nrN == 1:
                        n = 0
                        J.v_mul_f32(mfma_C[cindex,m,n,0], mfma_C[cindex,m,n,0], vweights[cm, m])
                        J.v_mul_f32(mfma_C[cindex,m,n,1], mfma_C[cindex,m,n,1], vweights[cm, m])
                        J.v_mul_f32(mfma_C[cindex,m,n,2], mfma_C[cindex,m,n,2], vweights[cm, m])
                        J.v_mul_f32(mfma_C[cindex,m,n,3], mfma_C[cindex,m,n,3], vweights[cm, m])
                        J.uni_cvt_pk_bf16_f32(vbf16[0], mfma_C[cindex, m,n,0], mfma_C[cindex, m,n,1]) 
                        J.uni_cvt_pk_bf16_f32(vbf16[1], mfma_C[cindex, m,n,2], mfma_C[cindex, m,n,3])
                        # buff_c.store_dwordx2(vbf16[0:1], vaddr, 0, offset12 = n*16*J.sizeof(C_dtype) + cn*HALF_BLOCK_SIZE_COL*J.sizeof_bf16)
                        J.global_store_dwordx2(vaddr, vbf16[0:1], "off", mod=f"offset:{n*16*J.sizeof(C_dtype) + cn*HALF_BLOCK_SIZE_COL*J.sizeof_bf16}")
                    else:
                        for n in range(0, nrN, 2):
                            J.v_mul_f32(mfma_C[cindex,m,n,0], mfma_C[cindex,m,n,0], vweights[cm, m])
                            J.v_mul_f32(mfma_C[cindex,m,n,1], mfma_C[cindex,m,n,1], vweights[cm, m])
                            J.v_mul_f32(mfma_C[cindex,m,n,2], mfma_C[cindex,m,n,2], vweights[cm, m])
                            J.v_mul_f32(mfma_C[cindex,m,n,3], mfma_C[cindex,m,n,3], vweights[cm, m])

                            J.v_mul_f32(mfma_C[cindex,m,n+1,0], mfma_C[cindex,m,n+1,0], vweights[cm, m])
                            J.v_mul_f32(mfma_C[cindex,m,n+1,1], mfma_C[cindex,m,n+1,1], vweights[cm, m])
                            J.v_mul_f32(mfma_C[cindex,m,n+1,2], mfma_C[cindex,m,n+1,2], vweights[cm, m])
                            J.v_mul_f32(mfma_C[cindex,m,n+1,3], mfma_C[cindex,m,n+1,3], vweights[cm, m])

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
                            # buff_c.store_dwordx4(vbf16, vaddr, 0, offset12 = n*16*J.sizeof(C_dtype) + cn*HALF_BLOCK_SIZE_COL*J.sizeof_bf16)
                            J.global_store_dwordx4(vaddr, vbf16, "off", mod=f"offset:{0*16*J.sizeof(C_dtype) + cn*HALF_BLOCK_SIZE_COL*J.sizeof_bf16} sc1 nt")

    return