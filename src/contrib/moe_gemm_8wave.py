from pyhip import jit, JIT
import torch
import contextlib

from .common.loaders import get_mfma_loader, get_mfma_loader_sorted_tok

__all__ = [
    "moe_gemm_final_reduce_bf16",
    "moe_gemm_8wave_g1u1",
    "moe_gemm_8wave_down"
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

    input[:] += J.s_mul_u32_u64(tok0, (TOPK * OC * J.sizeof_bf16))
    output[:] += J.s_mul_u32_u64(tok0, (OC * J.sizeof_bf16))

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
优化是循序渐进的过程，逐渐寻找和逼近到最佳性能设计，经常需要尝试各种不同的手段

"""



@jit(with_debug_log=False)
def moe_gemm_8wave_g1u1(J, is_input_over_4GB,
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
    """
    blockscale gemm的算力上限假设为2400T, 每个CU就是 2400/256=9.375 TFLOPS
    读取带宽假定是5TB/s, 每个CU就是 5e12/256 B/s
    M,N,K = 256,256,6144
      按照算力bound, 6144*256*256*2/9.375e12*1e6 = 85.89us
      按照带宽bound, 6144*256*2/(5e12/256)*1e6 = 161.06us
    实际cache能够起到一定的减少带宽压力的作用，但是总的来说仍然是带宽bound。

    按照161us估算带宽bound下算力可达到的上限几乎要减半了：
      6144*256*256*2/161e-6*256*1e-12 = 1280 TFLOPS

    """
    num_warps = 8

    assert AB_dtype in ["fp8", "bf16", "fp16", "f16"]
    C_dtype = "bf16"

    K = IC
    # loader always load 128bytes (8 x DW4-lanes) along K dimension
    wg_K = J.div(128, J.sizeof(AB_dtype))

    stride_k = IC * J.sizeof(AB_dtype)
    #stride_gate_up = J.div(J.div(OC, wg_N), 2) * wg_N * stride_k

    #blk_n = J.blockIdx.x # split along OC
    #blk_m = J.blockIdx.y

    blk1d = J.blockIdx.x
    NUM_CU = 256
    num_oc_blocks = J.div(OC, wg_N)
    num_groupped_blocks = num_blocks - (num_blocks % NUM_CU)
    blk_m = J.gpr("su32")
    blk_n = J.gpr("su32")
    use_xcd_swizzle = 1
    if use_xcd_swizzle:
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
    else:
        blk_m[0] = blk1d // num_oc_blocks
        blk_n[0] = blk1d - blk_m * num_oc_blocks


    #blk_m[0] *= 0
    expert_id = J.gpr(1, 'su32')
    J.s_load_dword(expert_id, sorted_expert_ids, blk_m[0] * J.sizeof_u32)
    max_id = J.gpr(1, 'su32')
    J.s_load_dword(max_id, num_valid_ids, 0)
    J.s_waitcnt(mod=f"lgkmcnt(0)")

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

    offset_64bit = J.gpr(2,"su32")
    J.s_mul_hi_u32(offset_64bit[1], expert_id, (OC * stride_k))
    J.s_mul_i32(offset_64bit[0], expert_id, (OC * stride_k))
    weight[:] += offset_64bit

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
        LOADER_TOPK = 0
        # B matrix needs to be interleaved by HALF_BLOCK_SIZE_COL
        # vm_load_b(k, m=0) loads from gate-weight
        # vm_load_b(k, m=1) loads from up-weight
        buff_a = J.Buffer(input, num_tokens * stride_k)
        buff_b = {}
        weight[:] += blk_n * (J.div(wg_N, 2) * stride_k) # gate-weight
        buff_b[0] = J.Buffer(weight, J.div(wg_N, 2) * stride_k)
        weight[:] += J.div(OC, 2) * stride_k # up-weights
        buff_b[1] = J.Buffer(weight, J.div(wg_N, 2) * stride_k)
        stride_n = J.div(OC,2) * J.sizeof(C_dtype)
        # assert bpreshuffle
    else:
        LOADER_TOPK = TOPK
        weight[:] += blk_n * (wg_N * stride_k)
        # w_scale[:] += blk_n * (J.div(wg_N, 32) * stride_scale32x256) + (J.warp_id[0] % 2) * (J.div(wg_N//2, 32) * stride_scale32x256)
        # sbuff_b = J.Buffer(w_scale, J.div(wg_N//2, 32) * stride_scale32x256)
        buff_a = J.Buffer(input, num_tokens * TOPK * stride_k)
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
    if gate_up:
        J.wg_load_lds(lds_sorted_ids, sorted_ids, wg_M * J.sizeof_u32, num_warps = num_warps, wait_barrier = True)
    else:
        J.wg_load_lds(lds_sorted_ids, sorted_ids, wg_M * J.sizeof_u32, num_warps = num_warps, wait_barrier = False)
        J.wg_load_lds(lds_sorted_weights, sorted_weights, wg_M * J.sizeof_f32, num_warps = num_warps, wait_barrier = True)

    vm_load_a, vm_load_cnt_a, vm_offset_inc_a, ds_read_a = get_mfma_loader_sorted_tok(J, num_warps, BLOCK_SIZE_ROW, BLOCK_K, stride_k, warp_m*MINI_BLOCK_M, lds_sorted_ids, LOADER_TOPK, num_tokens, input, is_input_over_4GB)
    vm_load_b, vm_load_cnt_b, vm_offset_inc_b, ds_read_b = get_mfma_loader(J, bpreshuffle, num_warps, HALF_BLOCK_SIZE_COL, BLOCK_K, stride_k, warp_n*MINI_BLOCK_N)
    vm_load_cnt_a = vm_load_cnt_a // 2

    use_f32_blockscales_128 = (AB_dtype == "fp8")

    if use_f32_blockscales_128:
        # "exepct scaleA in [k,m] layout"
        scale_BM, scale_BN, scale_BK = 1,128,128 
        # tic-toc LDS buffer for 256 per-token per-k-128 scales
        # 1-load per warp is enough to load this buffer
        lds_scaleA = [J.alloc_lds(num_warps * 64 * J.sizeof_f32),
                      J.alloc_lds(num_warps * 64 * J.sizeof_f32)]

        vrows = J.gpr("vu32")
        J.ds_read_b32(vrows, J.threadIdx.x[0] * J.sizeof_u32 + lds_sorted_ids)
        J.s_waitcnt(mod=f"lgkmcnt(0)")

        if gate_up:
            buff_sa = J.Buffer(pScaleA, num_tokens[0] * (J.div(K, scale_BK) * J.sizeof_u32))
            voffset_scaleA = J.gpr((vrows[0] & 0xFFFFFF) * J.sizeof_u32)
        else:
            buff_sa = J.Buffer(pScaleA, num_tokens[0] * (TOPK * J.div(K, scale_BK) * J.sizeof_u32))
            voffset_scaleA = J.gpr((vrows[0] & 0xFFFFFF) * (TOPK * J.sizeof_u32) + \
                                   (vrows[0] >> 24) * J.sizeof_u32)

        assert wg_M <= num_warps * 64
        # vm_load_scaleA(lds_scaleA[toc])
        # ds_read scaleA must be in MFMA_16x4 format
        # ds_read scaleB broad-cast in to 16x4 too
        def vm_load_scaleA(lds, bk):
            # bk: index of k block with size of 128
            # use execmask to ensure same impact on vmcnt for all warps
            J.s_mov_b32("m0", lds + J.warp_id[0]*(64*J.sizeof_f32))
            if gate_up:
                voff = J.gpr("vu32", voffset_scaleA[0] + J.gpr("su32", num_tokens[0]*(bk*J.sizeof_u32)))
            else:
                voff = J.gpr("vu32", voffset_scaleA[0] + J.gpr("su32", num_tokens[0]*(TOPK*bk*J.sizeof_u32)))
            #with J.ExecMask(J.threadIdx.x[0] < wg_M, early_skip=False):
            buff_sa.load_dword(None, voff, 0)

        # scale of B(weights) are very small, can be all loaded into LDS
        pScaleB[:] += expert_id * J.div(OC, scale_BN) * J.div(K, scale_BK) * J.sizeof_f32
        lds_scaleB = J.alloc_lds(J.div(K, scale_BK) * J.div(wg_N, scale_BN) * J.sizeof_f32)
        if gate_up:
            # first half from gate, second half from up
            assert wg_N >= 2 * scale_BN
            pScaleB[:] += blk_n * (J.div(wg_N, 2, scale_BN) * J.div(K, scale_BK) * J.sizeof_f32)
            J.wg_load_lds(lds_scaleB, pScaleB, J.div(wg_N, 2, scale_BN) * J.div(K, scale_BK) * J.sizeof_f32,
                        num_warps, wait_barrier = False)

            pScaleB[:] += J.div(OC, 2, scale_BN) * J.div(K, scale_BK) * J.sizeof_f32
            J.wg_load_lds(lds_scaleB + J.div(wg_N, 2, scale_BN) * J.div(K, scale_BK) * J.sizeof_f32,
                          pScaleB, J.div(wg_N, 2, scale_BN) * J.div(K, scale_BK) * J.sizeof_f32,
                          num_warps, wait_barrier = True)
        else:
            pScaleB[:] += blk_n * (J.div(wg_N, scale_BN) * J.div(K, scale_BK) * J.sizeof_f32)
            J.wg_load_lds(lds_scaleB, pScaleB, J.div(wg_N, scale_BN) * J.div(K, scale_BK) * J.sizeof_f32,
                        num_warps, wait_barrier = True)

        num_scaleB = J.div(wg_N, scale_BN)
        mfma_scaleA = J.gpr(nrM, "vf32")
        mfma_scaleB = J.gpr(num_scaleB, "vf32")
        vaddr_scaleA = J.gpr("vu32", (J.lane_id[0] % 16)*J.sizeof_f32 + warp_m * (16*nrM * J.sizeof_f32))
        def ds_read_scaleA(lds, m0):
            assert m0 in [0, 1]
            vaddr = J.gpr("vu32", vaddr_scaleA[0] + lds)
            for m in range(nrM):
                off = (m0*HALF_BLOCK_SIZE_ROW + m*16)*J.sizeof_f32
                J.ds_read_b32(mfma_scaleA[m], vaddr, mod=f"offset:{off}")

        vaddr_scaleB = J.gpr(num_scaleB, "vu32")
        for i in range(num_scaleB):
            vaddr_scaleB[i] = lds_scaleB + i*J.div(K, scale_BK)*J.sizeof_f32
        def ds_read_scaleB(bk):
            # k0: in unit of scale_BK
            # n0: in unit of scale_BN
            # all warps share the same scaleB
            assert scale_BN >= nrN * 16 * 4
            if isinstance(bk, int):
                off = bk * J.sizeof_f32
                for i in range(num_scaleB):
                    J.ds_read_b32(mfma_scaleB[i], vaddr_scaleB[i], mod=f"offset:{off}")
            else:
                for i in range(num_scaleB):
                    J.ds_read_b32(mfma_scaleB[i], vaddr_scaleB[i] + bk * J.sizeof_f32)


    mfma_A = J.gpr(nrM, 2, 4, "vfp8x4")            # 4x[16,128]
    mfma_B = J.gpr(2, nrN, 2, 4, "vfp8x4")            # 2x[16,128]
    mfma_C = J.gpr(4, nrM, nrN, 4, "vf32")      # 4x[4,2]x[16,16]

    if use_f32_blockscales_128:
        MFMA_FIFO_CNT = nrM * nrN
        # circular fifo buffer for post-processing
        # prepare scales for next round
        mfma_fifo_scale = J.gpr(2, nrM, "vf32")
        mfma_fifo = J.gpr(MFMA_FIFO_CNT, 4, "vf32")
        mfma_fifo_scale[...] = 0
        mfma_fifo[...] = 0
        mfma_fifo_c_index = 0

        def mfma(c_index):
            nonlocal mfma_fifo_scale, mfma_fifo, mfma_fifo_c_index
            b_index = c_index % 2

            fifo_read_id = 0
            fifo_write_id = 0
            for m in range(nrM):
                for n in range(nrN):
                    if n == 0:
                        mfma_fifo_scale[c_index%2, m] = mfma_scaleA[m] * mfma_scaleB[b_index]
                    J.v_fmac_f32(mfma_C[mfma_fifo_c_index, m, n, 0], mfma_fifo[fifo_read_id, 0], mfma_fifo_scale[mfma_fifo_c_index % 2,m])
                    J.v_fmac_f32(mfma_C[mfma_fifo_c_index, m, n, 1], mfma_fifo[fifo_read_id, 1], mfma_fifo_scale[mfma_fifo_c_index % 2,m])
                    J.v_fmac_f32(mfma_C[mfma_fifo_c_index, m, n, 2], mfma_fifo[fifo_read_id, 2], mfma_fifo_scale[mfma_fifo_c_index % 2,m])
                    J.v_fmac_f32(mfma_C[mfma_fifo_c_index, m, n, 3], mfma_fifo[fifo_read_id, 3], mfma_fifo_scale[mfma_fifo_c_index % 2,m])
                    fifo_read_id += 1

                    J.v_mfma_f32_16x16x128_f8f6f4(mfma_fifo[fifo_write_id % MFMA_FIFO_CNT], mfma_B[b_index, n], mfma_A[m], 0)
                    fifo_write_id += 1
                    yield 16
            mfma_fifo_c_index = c_index
        
        def mfma_tail():
            fifo_read_id = 0
            for m in range(nrM):
                for n in range(nrN):
                    if mfma_fifo_c_index is not None:
                        J.v_fmac_f32(mfma_C[mfma_fifo_c_index, m, n, 0], mfma_fifo[fifo_read_id, 0], mfma_fifo_scale[mfma_fifo_c_index % 2,m])
                        J.v_fmac_f32(mfma_C[mfma_fifo_c_index, m, n, 1], mfma_fifo[fifo_read_id, 1], mfma_fifo_scale[mfma_fifo_c_index % 2,m])
                        J.v_fmac_f32(mfma_C[mfma_fifo_c_index, m, n, 2], mfma_fifo[fifo_read_id, 2], mfma_fifo_scale[mfma_fifo_c_index % 2,m])
                        J.v_fmac_f32(mfma_C[mfma_fifo_c_index, m, n, 3], mfma_fifo[fifo_read_id, 3], mfma_fifo_scale[mfma_fifo_c_index % 2,m])
                        fifo_read_id += 1
    elif AB_dtype == "bf16":
        def mfma(c_index):
            b_index = c_index % 2
            for k in range(2):
                for m in range(nrM):
                    for n in range(nrN):
                        J.v_mfma_f32_16x16x32_bf16(mfma_C[c_index, m, n], mfma_B[b_index, n, k], mfma_A[m, k], mfma_C[c_index, m, n])
                        yield 16
        def mfma_tail():
            pass
    else:
        assert AB_dtype == "fp16" or AB_dtype == "f16" 
        def mfma(c_index):
            b_index = c_index % 2
            for k in range(2):
                for m in range(nrM):
                    for n in range(nrN):
                        J.v_mfma_f32_16x16x32_f16(mfma_C[c_index, m, n], mfma_B[b_index, n, k], mfma_A[m, k], mfma_C[c_index, m, n])
                        yield 16
        def mfma_tail():
            pass
    """
    def mfma(c_index):
        b_index = c_index % 2
        for k in range(2):
            for m in range(nrM):
                for n in range(nrN):
                    J.v_mfma_f32_16x16x32_bf16(mfma_C[c_index, m, n], mfma_B[b_index, n, k], mfma_A[m, k], mfma_C[c_index, m, n])
                    yield 16
    def mfma_tail():
        pass
    """

    loop_cnt = J.div(K, wg_K)
    #assert HALF_BLOCK_SIZE_ROW == HALF_BLOCK_SIZE_COL

    a_moffset = J.gpr("su32", 0)
    if gate_up:
        b_moffsets = J.gpr(2, "su32", 0, 0)
    else:
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
        if gate_up:
            return vm_load_b(ldsB[k,m], buff_b[m], b_moffsets[m])
        else:
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
        if use_f32_blockscales_128: vm_load_scaleA(lds_scaleA[tic], 0)
        J.emit(vm_loadB(tic,0))
        J.emit(vm_loadA(tic,0))
        J.emit(vm_loadB(tic,1))
        J.emit(vm_loadA(tic,1))

        with J.If(warp_m[0] == 1):
            J.s_barrier()

        mfma_C[...] = 0

        J.s_waitcnt(mod=f"vmcnt({vm_load_cnt_a + vm_load_cnt_b})"); J.s_barrier()

        step_k()

        if use_f32_blockscales_128:
            vm_load_scaleA(lds_scaleA[toc], 1)
            vm_load_cnt_scaleA = 1
        else:
            vm_load_cnt_scaleA = 0
        J.emit(vm_loadA(toc,0))
        J.emit(vm_loadB(toc,0))
        J.emit(vm_loadB(toc,1))

        J.s_waitcnt(mod=f"vmcnt({vm_load_cnt_a + vm_load_cnt_b*2 + vm_load_cnt_scaleA})"); J.s_barrier()

        def loop_body(k, loop_cnt):
            nonlocal tic, toc
            ds_readB(tic, 0)    # lgkmcnt += nrN*2 (2*2)
            ds_readA(tic, 0)    # lgkmcnt += nrM*2 (4*2)

            if use_f32_blockscales_128:
                ds_read_scaleA(lds_scaleA[tic], 0)
                ds_read_scaleB(k)

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
            if use_f32_blockscales_128: ds_read_scaleA(lds_scaleA[tic], 1)
            J.emit(vm_loadB(tic,0))                         # vm_load_cnt_b
            J.s_barrier()

            J.s_waitcnt(mod="lgkmcnt(0)"); J.s_setprio(1)
            J.emit(mfma(2))
            J.s_setprio(0); J.s_barrier()

            J.emit(vm_loadB(tic,1))                         # vm_load_cnt_b
            if use_f32_blockscales_128: vm_load_scaleA(lds_scaleA[tic], k+2)
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
            assert not use_f32_blockscales_128, "there is an unknown accuracy issue for f32 blockscale-128 case"
            assert (loop_cnt % 2) == 0
            k = J.gpr("su32", 0)

            with J.While(k[0] < loop_cnt):
                loop_body(k, loop_cnt)
                k[0] += 1
                loop_body(k, loop_cnt)
                k[0] += 1

        mfma_tail()
        J.s_waitcnt(mod="vmcnt(0)")
        #J.s_waitcnt(mod="lgkmcnt(0)")
        with J.If(warp_m[0] == 0):
            J.s_barrier()
    else:
        mfma_C[...] = 0
        for k in range(loop_cnt):
            J.emit(vm_loadB(0,0))
            J.emit(vm_loadA(0,0))
            if use_f32_blockscales_128: vm_load_scaleA(lds_scaleA[0], k)
            J.s_waitcnt(mod="vmcnt(0)"); J.s_barrier()

            ds_readA(0,0)
            ds_readB(0,0)
            if use_f32_blockscales_128:
                ds_read_scaleA(lds_scaleA[0], 0)
                ds_read_scaleB(k)
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
            if use_f32_blockscales_128:
                ds_read_scaleA(lds_scaleA[0], 1)
            J.s_waitcnt(mod="lgkmcnt(0)"); J.s_barrier()

            #J.debug_log(mfma_A[0,0], torch.float8_e4m3fn, "4h.16v.16h")
            #J.debug_log(mfma_A[0,1], torch.float8_e4m3fn, "4h.16v.16h")
            #J.s_endpgm()

            J.emit(mfma(2))
            J.emit(mfma(3))

            step_k()

        mfma_tail()
        J.s_waitcnt(mod="lgkmcnt(0)")
        J.s_waitcnt(mod="vmcnt(0)")

    if gate_up:
        # silu(c[0])*c[1]   64*32
        # silu(c[2])*c[3]   64*32
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
            vaddr0 = J.gpr("vu32", col * J.sizeof_DW2 + warp_n * 16 * J.sizeof_bf16 + blk_n * (J.div(wg_N,2) * J.sizeof(C_dtype)))
        else:
            vbf16 = J.gpr(4, "vbf16x2") # DWORDx4
            col = J.lane_id // 16
            swap_12_col = (col & 1) * 2 + (col >> 1)
            vaddr0 = J.gpr("vu32", swap_12_col * J.sizeof_DW4 + warp_n * nrN * 16 * J.sizeof_bf16 + blk_n * (J.div(wg_N,2) * J.sizeof(C_dtype)))

        for cm in range(2):
            igate = cm*2 + 0
            iup = cm*2 + 1
            for m in range(nrM):
                vrows_topk = J.gpr(vrows[cm, m] >> 24)
                with J.ExecMask(vrows_topk < TOPK):
                    # to support (num_tokens * TOPK * stride_n) > 4GB, we can only use global_store_dword
                    vaddr = J.gpr(2, "vu32", output[0], output[1])
                    J.v_lshl_add_u64(vaddr, J.gpr(2, "vu32", vaddr0 + vrows_topk * (stride_n), 0), 0, vaddr)
                    J.v_mad_u64_u32(vaddr, "vcc", (vrows[cm, m] & 0xFFFFFF), J.gpr("vu32", TOPK * stride_n), vaddr)
                    for n in range(nrN):
                        mfma_C[igate,m,n,0] = J.silu(mfma_C[igate,m,n,0]) * mfma_C[iup,m,n,0]
                        mfma_C[igate,m,n,1] = J.silu(mfma_C[igate,m,n,1]) * mfma_C[iup,m,n,1]
                        mfma_C[igate,m,n,2] = J.silu(mfma_C[igate,m,n,2]) * mfma_C[iup,m,n,2]
                        mfma_C[igate,m,n,3] = J.silu(mfma_C[igate,m,n,3]) * mfma_C[iup,m,n,3]
                    assert nrN in [1, 2], f"{nrN=}"
                    if nrN == 1:
                        J.uni_cvt_pk_bf16_f32(vbf16[0], mfma_C[igate, m,0,0], mfma_C[igate, m,0,1])
                        J.uni_cvt_pk_bf16_f32(vbf16[1], mfma_C[igate, m,0,2], mfma_C[igate, m,0,3])
                        # buff_c.store_dwordx2(vbf16[0:1], vaddr, 0, offset12 = 0*16*J.sizeof(C_dtype))
                        J.global_store_dwordx2(vaddr, vbf16[0:1], "off", mod=f"offset:{0*16*J.sizeof(C_dtype)}")
                    else:
                        J.uni_cvt_pk_bf16_f32(vbf16[0], mfma_C[igate, m,0,0], mfma_C[igate, m,0,1])
                        J.uni_cvt_pk_bf16_f32(vbf16[1], mfma_C[igate, m,0,2], mfma_C[igate, m,0,3])
                        J.uni_cvt_pk_bf16_f32(vbf16[2], mfma_C[igate, m,1,0], mfma_C[igate, m,1,1])
                        J.uni_cvt_pk_bf16_f32(vbf16[3], mfma_C[igate, m,1,2], mfma_C[igate, m,1,3])
                        J.v_permlane16_swap_b32(vbf16[0], vbf16[2])
                        J.v_permlane16_swap_b32(vbf16[1], vbf16[3])
                        # buff_c.store_dwordx4(vbf16, vaddr, 0, offset12 = 0*16*J.sizeof(C_dtype))
                        J.global_store_dwordx4(vaddr, vbf16, "off", mod=f"offset:{0*16*J.sizeof(C_dtype)}")
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
                            J.global_store_dwordx4(vaddr, vbf16, "off", mod=f"offset:{0*16*J.sizeof(C_dtype) + cn*HALF_BLOCK_SIZE_COL*J.sizeof_bf16}")

    return


@jit(with_debug_log=False)
def moe_gemm_8wave_down(J, is_output_over_4GB, AB_dtype, wg_M, wg_N,
                        NUM_EXPERTS, OC, IC, num_oc_splits,
                        gate_up, bpreshuffle, TOPK,
                        _sorted_ids:"uint*",
                        _sorted_weights:"float*",
                        _sorted_expert_ids:"uint*",
                        num_valid_ids:"uint*",
                        _weight:"void*",_pScaleB:"void*",
                        input:"void*",pScaleA:"void*",
                        output:"void*",
                        num_tokens:"uint",
                        blk_atomic_int:"void*"):
    C_dtype = "bf16"
    assert AB_dtype in ["fp8", "bf16"]
    assert C_dtype == "bf16"

    assert (OC % num_oc_splits) == 0, f"{OC=} {num_oc_splits=}"
    OC = OC // num_oc_splits

    assert gate_up == False
    num_warps = 8
    stride_k = IC * J.sizeof(AB_dtype)

    num_token_topks = J.gpr(num_tokens * TOPK)

    if 0:
        """ 没有找到 persistent kernel 动态分配任务如何使用 XCD swizzle 提高L2命中率的方法 
        动态任务分配性能不稳定，不知道是否跟分配随机性有关，因此有一个优化思路就是预先根据
        每个expert-block中token的数量尽量均匀分配到每个persistent kernel，这需要一个单独的
        kernel完成分配过程，这个分配过程可以尽量把相邻的expert分给相同的XCD, 直到无法继续均匀分配为止。
        
        """
        blk_oc = J.blockIdx.x # split along OC
        blk_m = J.blockIdx.y #; blk_m[0] *= 0
        # num_oc_splits*num_e_blocks
        blk_id = J.blockIdx.x
        # 根据硬件scheduler的设计还原当前block分配到device上的具体CU/SE/XCD
        NUM_XCD = 8
        SE_PER_XCD = 4
        CU_PER_SE = 8
        CU_PER_XCD = (SE_PER_XCD * CU_PER_SE) # 32
        NUM_CU = (NUM_XCD * CU_PER_XCD)
        xcd_id = J.gpr(blk_id % NUM_XCD)
        #se_id = J.gpr((blk_id // NUM_XCD) % SE_PER_XCD)
        #cu_id = J.gpr((blk_id // (NUM_XCD * SE_PER_XCD)) % CU_PER_SE)
        cu_id = ((blk_id // NUM_XCD) % CU_PER_XCD)
        global_cu_id = J.gpr(cu_id +  xcd_id * CU_PER_XCD)

        blk_id = global_cu_id

    max_id = J.gpr(1, 'su32')
    J.s_load_dword(max_id, num_valid_ids, 0)

    with J.While():
        blk_id = J.gpr("su32")
        lds_blk_id = J.alloc_lds(J.sizeof_u32)
        vnext_blk_id = J.gpr("vu32", 0)
        vone = J.gpr("vu32", 1)
        vaddr_zeros = J.gpr("vu32", 0)
        with J.ExecMask(J.threadIdx.x[0] == 0):
            J.global_atomic_add(vnext_blk_id, vaddr_zeros, vone, blk_atomic_int, mod="sc0")
            J.s_waitcnt(mod=f"vmcnt(0)")
            # broadcast vnext_blk_id to all waves in the warp by LDS
            J.ds_write_b32(vaddr_zeros, vnext_blk_id, mod=f"offset:{lds_blk_id}")
            J.s_waitcnt(mod=f"lgkmcnt(0)")
        J.s_barrier()

        J.ds_read_b32(vnext_blk_id, vaddr_zeros, mod=f"offset:{lds_blk_id}")
        J.s_waitcnt(mod=f"lgkmcnt(0)")

        J.v_readfirstlane_b32(blk_id, vnext_blk_id)
        J.s_nop(8)
        
        J.free_lds(lds_blk_id)

        blk_oc = J.gpr(blk_id % num_oc_splits)
        blk_m = J.gpr(blk_id // num_oc_splits)

        with J.If(blk_m[0] * wg_M >= max_id[0]):
            J.s_endpgm()

        warp_M = J.div(wg_M, num_warps)
        sorted_ids = J.gpr(2, "su32", _sorted_ids[0], _sorted_ids[1])
        sorted_ids[:] += blk_m[0] * (wg_M * J.sizeof_u32)
        sorted_weights = J.gpr(2, "su32", _sorted_weights[0], _sorted_weights[1])
        sorted_weights[:] += blk_m[0] * (wg_M * J.sizeof_u32)

        expert_id = J.gpr(1, 'su32')
        J.s_load_dword(expert_id, _sorted_expert_ids, blk_m[0] * J.sizeof_u32)
        J.s_waitcnt(mod=f"lgkmcnt(0)")

        # prefetch sorted ids & weights into LDS
        lds_sorted_ids = J.alloc_lds(wg_M * J.sizeof_u32)
        lds_sorted_weights = J.alloc_lds(wg_M * J.sizeof_DW)
        J.wg_load_lds(lds_sorted_ids, sorted_ids, wg_M * J.sizeof_u32, num_warps = num_warps, wait_barrier = False)
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

        # lazy wait before first use
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

        weight = J.gpr(2, "su32", _weight[0], _weight[1])
        weight[:] += expert_id * (num_oc_splits * OC * stride_k)
        if num_oc_splits > 1:
            weight[:] += blk_oc * OC * stride_k
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
            #J.s_waitcnt(mod=f"vmcnt({0})")
            #J.s_barrier()

            # since IC is small, load all B scales into LDS: 
            sizeof_scaleB = J.div(OC, scale_BN) * J.div(IC, scale_BK) * J.sizeof_f32
            pScaleB = J.gpr(2, "su32", _pScaleB[0], _pScaleB[1])
            pScaleB[:] += expert_id * num_oc_splits * sizeof_scaleB
            if num_oc_splits > 1:
                pScaleB[:] += blk_oc * sizeof_scaleB

            lds_scaleB = J.alloc_lds(sizeof_scaleB)
            J.wg_load_lds(lds_scaleB, pScaleB, sizeof_scaleB, num_warps, wait_barrier = False)

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
        stride_c = num_oc_splits * OC * J.sizeof(C_dtype)
        buff_c = J.Buffer(output, num_token_topks * stride_c)
        vaddr_rows = J.gpr(nrM, 2, "vu32")
        for m in range(nrM):
            if is_output_over_4GB:
                J.v_mad_u64_u32(vaddr_rows[m], "vcc", vrows[m], J.gpr("su32", stride_c), 0)
                J.v_lshl_add_u64(vaddr_rows[m], output, 0, vaddr_rows[m])
                col = (J.lane_id // 16)
                swap_12_col = (col & 1) * 2 + (col >> 1)
                J.v_lshl_add_u64(vaddr_rows[m], J.gpr(2, "vu32", swap_12_col*J.sizeof_DW4 + J.gpr("su32", blk_oc * OC * J.sizeof(C_dtype)), 0), 0, vaddr_rows[m])
            else:
                row_off = vrows[m] * stride_c
                col = (J.lane_id // 16)
                swap_12_col = (col & 1) * 2 + (col >> 1)
                vaddr_rows[m,0] = row_off + swap_12_col * J.sizeof_DW4
                vaddr_rows[m,0] += J.gpr("su32", blk_oc * OC * J.sizeof(C_dtype))

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
                            buff_c.store_dwordx4(mfma_C_bf16[m,n], vaddr_rows[m,0], soffset, offset12 = n*16*J.sizeof(C_dtype), ext_mod="")
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
