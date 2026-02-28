from pyhip import jit, JIT
import torch

from .common.loaders import get_mfma_loader, get_mfma_loader_sorted_tok

__all__ = [
    "moe_gemm_final_reduce_bf16",
    "moe_gemm_8wave",
]

@jit(with_debug_log=False)
def moe_gemm_final_reduce_bf16(J, TOPK, OC,
                               input:"void*",
                               output:"void*",
                               num_tokens_wg:"int",
                               num_big_wg:"int",
                               num_tokens_total:"int"):
    wg_id = J.blockIdx.x

    tok0 = J.gpr("su32")
    tok1 = J.gpr("su32")
    #tok0[0] = wg_id[0] * (num_tokens_wg) # need to do 1 more 
    #tok1[0] = tok0 + (num_tokens_wg) 

    with J.If(wg_id[0] < num_big_wg[0]) as If:
        tok0[0] = wg_id[0] * (1 + num_tokens_wg) # need to do 1 more 
        tok1[0] = tok0 + (1 + num_tokens_wg)

        If.Else()
        tok_base = num_big_wg * (1 + num_tokens_wg)
        tok0[0] = tok_base + (wg_id - num_big_wg) * num_tokens_wg
        tok1[0] = tok0 + num_tokens_wg

    J.s_min_u32(tok1, tok1[0], num_tokens_total[0])
    
    input[:] += tok0[0] * (TOPK * OC * J.sizeof_bf16)
    output[:] += tok0[0] * (OC * J.sizeof_bf16)

    buff = J.Buffer(input, (tok1[0] - tok0[0]) * (TOPK * OC * J.sizeof_bf16))
    buff_out = J.Buffer(output, (tok1[0] - tok0[0]) * (OC * J.sizeof_bf16))

    voffset_prefetch = J.gpr(J.threadIdx.x[0] * J.sizeof_DW4)
    voffset_output = J.gpr(J.threadIdx.x[0] * J.sizeof_DW4)
    num_threads = 64

    vinput = J.gpr(2, TOPK, 4, "vu32")

    part_size = num_threads * J.sizeof_DW4 // J.sizeof_bf16
    part_cnt = J.div(OC, part_size)
    assert part_cnt % 2  == 0

    index = 0
    voff = J.gpr("vu32", voffset_prefetch)
    for topk in range(TOPK):
        buff.load_dwordx4(vinput[index, topk], voff, 0, offset12=0)
        voff[0] += OC * J.sizeof_bf16
    voffset_prefetch[0] += num_threads * J.sizeof_DW4
    index = index ^ 1

    with J.While(tok0[0] < tok1[0]):
        assert index == 1

        for part_id in range(part_cnt):
            voff = J.gpr("vu32", voffset_prefetch)
            for topk in range(TOPK):
                buff.load_dwordx4(vinput[index, topk], voff, 0, offset12=0)
                voff[0] += OC * J.sizeof_bf16
            voffset_prefetch[0] += part_size * J.sizeof_bf16
            if part_id == (part_cnt - 2):
                voffset_prefetch[0] += (TOPK*OC - OC) * J.sizeof_bf16 # go to next token
            index = index ^ 1

            # wait for vinput[index,...] to be ready
            J.s_waitcnt(mod=f"vmcnt({TOPK})")

            voutput = J.gpr(8, "vf32")
            for topk in range(TOPK):
                # compute current 
                if topk == 0:
                    for i in range(4):
                        voutput[2*i+0] = vinput[index, topk, i] << 16
                        voutput[2*i+1] = vinput[index, topk, i] & 0xFFFF0000
                else:
                    for i in range(4):
                        vf32x2 = J.gpr(2, "vf32")
                        vf32x2[0] = vinput[index, topk, i] << 16
                        vf32x2[1] = vinput[index, topk, i] & 0xFFFF0000
                        J.v_pk_add_f32(voutput[2*i+0:2*i+1], voutput[2*i+0:2*i+1], vf32x2)

            vout = J.gpr(4, "vbf16x2")
            for i in range(4):
                J.uni_cvt_pk_bf16_f32(vout[i], voutput[2*i+0], voutput[2*i+1])
            buff_out.store_dwordx4(vout, voffset_output, 0, offset12=0)
            voffset_output[0] += part_size * J.sizeof_bf16

        assert index == 1

        tok0[0] += 1
    J.s_waitcnt(mod=f"vmcnt({0})")

"""
moe gemm 
"""


"""
vm_load, vm_load_cnt, vm_offset_inc, ds_read = J.get_mfma_loader(use_pre_shuffle, num_warps, BM, BK, stride_k, warp_m*64)
    stride_k是外存数据的stride.

    def get_loader_row_major(self, num_warps, BM, BK, vm_stride, warp_row0)
    def vm_load(lds_offset, buff, vm_offset)
        都会加载 2D tensor [BM, BK, uint8] 到 LDS 中

    def ds_read_16x64(lds_offset, vdst, m, k)
        从LDS [BM, BK, uint8] 中按照 mfma_16 的格式加载 16x64 字节大小的数据到vdst中 (因为这么大的数据正好是ds_read_b128可以一次性完成的)
        m,k就是偏移

num_warps 这么多个warp，协同发起加载指令
每次调用 vm_load 
每次调用 ds_read 都会从

loader函数如何单独调试正确性？可以从实现最简单的moe_gemm开始 (bf16类型的down_proj)

"""



@jit(with_debug_log=False)
def moe_gemm_8wave(J, AB_dtype, wg_M, wg_N,
                   NUM_EXPERTS, OC, IC, 
                   gate_up, TOPK,
                   sorted_ids:"uint*",
                   sorted_weights:"float*",
                   sorted_expert_ids:"uint*",
                   num_valid_ids:"uint*",
                   weight:"void*",w_scale:"void*",
                   input:"void*", i_scale:"void*",
                   output:"void*",
                   num_tokens:"uint"):
    num_warps = 8

    assert AB_dtype in ["fp8", "bf16", "fp16", "f16"]
    C_dtype = "bf16"

    K = IC
    # loader always load 128bytes (8 x DW4-lanes) along K dimension
    wg_K = J.div(128, J.sizeof(AB_dtype))

    stride_k = IC * J.sizeof(AB_dtype)
    stride_n = OC * J.sizeof(C_dtype)
    stride_gate_up = J.div(J.div(OC, wg_N), 2) * wg_N * stride_k

    blk_n = J.blockIdx.x # split along OC
    blk_m = J.blockIdx.y
    #blk_m[0] *= 0
    expert_id = J.gpr(1, 'su32')
    J.s_load_dword(expert_id, sorted_expert_ids, blk_m[0] * J.sizeof_u32)
    max_id = J.gpr(1, 'su32')
    J.s_load_dword(max_id, num_valid_ids, 0)
    J.s_waitcnt(mod=f"lgkmcnt(0)")

    J.debug_setup((blk_m[0] == 0) & (blk_n[0] == 0) & (J.warp_id[0] == 0))
    with J.If(blk_m[0] * wg_M >= max_id[0]):
        J.s_endpgm()
    
    #if gate_up:
    #    output[:] += blk_n * (wg_N//2 * J.sizeof(C_dtype))
    #else:
    #    output[:] += blk_n * (wg_N * J.sizeof(C_dtype))


    sorted_ids[:] += blk_m * (wg_M * J.sizeof_u32)
    sorted_weights[:] += blk_m * (wg_M * J.sizeof_u32)

    #i_scale[:] += blk_m * (J.div(wg_M,32) * stride_scale32x256)
    #w_scale[:] += expert_id * (J.div(OC,32) * stride_scale32x256)
    #i_scale[:] += (J.warp_id[0] // 2) * (J.div(wg_M//2, 32) * stride_scale32x256)

    if gate_up:
        """
        weight[:] += expert_id * (OC * stride_k) + blk_n * (wg_N//2 * stride_k)
        w_scale[:] += blk_n * (J.div(wg_N//2, 32) * stride_scale32x256) + (J.warp_id[0] % 2) * (J.div(wg_N//4, 32) * stride_scale32x256)
        # gate-scale buff + up-scale buff
        sbuff_b = [None, None]
        sbuff_b[0] = J.Buffer(w_scale, J.div(wg_N//4, 32) * stride_scale32x256)
        w_scale[:] += J.div(OC//2, 32) * stride_scale32x256
        sbuff_b[1] = J.Buffer(w_scale, J.div(wg_N//4, 32) * stride_scale32x256)
        """
        assert 0
        buff_a = J.Buffer(input, num_tokens * stride_k)
    else:
        weight[:] += expert_id * (OC * stride_k) + blk_n * (wg_N * stride_k)
        # w_scale[:] += blk_n * (J.div(wg_N, 32) * stride_scale32x256) + (J.warp_id[0] % 2) * (J.div(wg_N//2, 32) * stride_scale32x256)
        # sbuff_b = J.Buffer(w_scale, J.div(wg_N//2, 32) * stride_scale32x256)
        buff_a = J.Buffer(input, num_tokens * TOPK * stride_k)
        buff_b = J.Buffer(weight, wg_N * stride_k)
        buff_c = J.Buffer(output, num_tokens * TOPK * stride_n)

    # basic configuration for 8-wave
    WARPS_COL = 4
    WARPS_ROW = 2
    BLOCK_SIZE_ROW = wg_M
    BLOCK_SIZE_COL = wg_N
    BLOCK_K = 128
    HALF_BLOCK_SIZE_ROW = J.div(BLOCK_SIZE_ROW, 2)
    HALF_BLOCK_SIZE_COL = J.div(BLOCK_SIZE_COL, 2)
    MINI_BLOCK_M = J.div(HALF_BLOCK_SIZE_ROW, WARPS_ROW) # 64
    MINI_BLOCK_N = J.div(HALF_BLOCK_SIZE_COL, WARPS_COL) # 32

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
    J.wg_load_lds(lds_sorted_ids, sorted_ids, wg_M * J.sizeof_u32, num_warps = num_warps, wait_barrier = False)
    J.wg_load_lds(lds_sorted_weights, sorted_weights, wg_M * J.sizeof_f32, num_warps = num_warps, wait_barrier = True)

    bpreshuffle = True

    vm_load_a, vm_load_cnt_a, vm_offset_inc_a, ds_read_a = get_mfma_loader_sorted_tok(J, num_warps, HALF_BLOCK_SIZE_ROW, BLOCK_K, stride_k, warp_m*MINI_BLOCK_M, lds_sorted_ids, TOPK, num_tokens)
    vm_load_b, vm_load_cnt_b, vm_offset_inc_b, ds_read_b = get_mfma_loader(J, bpreshuffle, num_warps, HALF_BLOCK_SIZE_COL, BLOCK_K, stride_k, warp_n*MINI_BLOCK_N)

    mfma_A = J.gpr(nrM, 2, 4, "vfp8x4")            # 4x[16,128]
    mfma_B = J.gpr(2, nrN, 2, 4, "vfp8x4")            # 2x[16,128]
    mfma_C = J.gpr(4, nrM, nrN, 4, "vf32")      # 4x[4,2]x[16,16]

    def mfma(c_index):
        b_index = c_index % 2
        for k in range(2):
            for m in range(nrM):
                for n in range(nrN):
                    J.v_mfma_f32_16x16x32_bf16(mfma_C[c_index, m, n], mfma_B[b_index, n, k], mfma_A[m, k], mfma_C[c_index, m, n])
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

    mfma_C[...] = 0
    for k in range(loop_cnt):
        J.emit(vm_loadB(0,0))
        J.emit(vm_loadA(0,0))
        #if use_f32_blockscales_128: vm_load_scaleA(lds_scaleA[0], k)
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


    # scatter output to : [num_tokens, topk, dims]
    vrows = J.gpr(WARPS_ROW, nrM, "vu32")
    vweights = J.gpr(WARPS_ROW, nrM, "vf32")
    for m0 in range(2):
        for m in range(nrM):
            row = (J.lane_id % 16) + (m0 * HALF_BLOCK_SIZE_ROW) + (warp_m * MINI_BLOCK_M) + m*16
            J.ds_read_b32(vrows[m0, m], row * J.sizeof_u32 + lds_sorted_ids)
            J.ds_read_b32(vweights[m0, m], row * J.sizeof_f32 + lds_sorted_weights)

    J.s_waitcnt(mod=f"lgkmcnt(0)")

    stride_c = OC * J.sizeof(C_dtype)
    vbf16 = J.gpr(4, "vbf16x2") # DWORDx4
    col = J.lane_id // 16
    swap_12_col = (col & 1) * 2 + (col >> 1)

    vaddr0 = J.gpr("vu32", swap_12_col * J.sizeof_DW4 + warp_n * 32 * J.sizeof_bf16 + blk_n * (wg_N * J.sizeof(C_dtype)))

    for cindex in range(4):
        cm = cindex // 2
        cn = cindex % 2
        for m in range(nrM):
            vaddr = J.gpr("vu32",
                          vaddr0 + \
                          (vrows[cm, m] & 0xFFFFFF) * (TOPK * stride_c) + \
                          (vrows[cm, m] >> 24) * (stride_c))

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
                buff_c.store_dwordx4(vbf16, vaddr, 0, offset12 = n*16*J.sizeof(C_dtype) + cn*HALF_BLOCK_SIZE_COL*J.sizeof_bf16)
    return