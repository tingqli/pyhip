import pyhip
import torch

def get_loader_b_preshuffled(J, weight, wg_N, nbM, nbK, stride_b, ibM0, gate_up, stride_gate_up):
    num_warps = 4
    stride_1kb = J.div(16*stride_b, 1024)
    assert nbK == 2
    warp_k = J.warp_id[0] % nbK
    warp_m = J.warp_id[0] // nbK
    # each vm load can load [num_warps] = [2,2] 1KB blocks
    vm_load_cnt = J.div(nbM, num_warps//nbK)

    if gate_up:
        # gate-up, select buff based on warp_m
        weight[:] += warp_m * stride_gate_up
        buff = J.Buffer(weight, wg_N//2 * stride_b)
        vmem_warp_off = warp_k * 1024
        # interleaving gate (load by warp-0/1) with up (load by warp-2/3)
        # the interleave unit is [32 x 2xBK] due to scale unit size
        #           0 0 2 2 | 0 0 2 2
        #           1 1 3 3 | 1 1 3 3
        # warp_m    0 0 1 1 | 0 0 1 1
        # 0
        lds_warp_stride0 = nbK * 1024
        lds_warp_stride1 = 2*lds_warp_stride0
        step_m01 = J.div(nbM//2, num_warps//nbK)
        lds_warp_off = J.gpr("su32", warp_m * (step_m01 * nbK * 1024) + warp_k * 1024)
    else:
        buff = J.Buffer(weight, wg_N * stride_b)
        vmem_warp_off = warp_m * (stride_1kb * 1024) + warp_k * 1024
        lds_warp_off = J.gpr("su32", warp_m * (nbK * 1024) + warp_k * 1024)

    vmem_voff = J.gpr(J.lane_id[0] * J.sizeof_DW4 + vmem_warp_off)
    

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

    # return a loader constructor which can emit
    def vm_load(lds_offset):
        J.s_mov_b32("m0", lds_warp_off + lds_offset)
        voff = J.gpr("vu32", vmem_voff[0])

        if gate_up:
            assert vm_load_cnt % 2 == 0
            for m in range(vm_load_cnt):
                yield 1
                buff.load_dwordx4(None, voff, 0, offset12=0)
                if m == (vm_load_cnt//2 - 1):
                    J.s_addk_i32("m0", lds_warp_stride0 + lds_warp_stride1)
                else:
                    J.s_addk_i32("m0", lds_warp_stride0)
                voff[0] += (num_warps//nbK//2)*(stride_1kb)*1024
        else:
            for m in range(vm_load_cnt):
                yield 1
                buff.load_dwordx4(None, voff, 0, offset12=0)
                J.s_addk_i32("m0", 256*J.sizeof_DW4)
                voff[0] += (num_warps//nbK)*(stride_1kb)*1024

        vmem_voff[0] += nbK * 1024
    return vm_load, vm_load_cnt, ds_read_1kb

def get_loader_sorted_tok(J, buff, lds_sorted_ids, nbM, nbK, stride_b, ibM0, gate_up, TOPK, num_tokens):
    num_warps = 4

    if 0:
        # check bank-conflict
        J.show_mfma_in_lds(mfma_MN=16, num_mfmas=2) 
        J.show_mfma_in_lds(mfma_MN=16, num_mfmas=2, swizzle_1=1)
        J.show_mfma_in_lds(mfma_MN=16, num_mfmas=2, swizzle_2=1)
        J.show_mfma_in_lds(mfma_MN=16, num_mfmas=2, swizzle_2=2)
        assert 0
    # each wave load 8x128 bytes , 4 waves loads 32x128 bytes
    lds_stride_b = nbK * 4 * J.sizeof_DW4
    warp_m_off = J.warp_id[0] * 8

    def swizzle(row, col):
        return (col ^ row) % 8
    col = J.threadIdx.x % 8
    row = J.threadIdx.x // 8
    swizzle_col = swizzle(row, col)
    # vmem_voff = J.gpr(row * stride_b + swizzle_col * J.sizeof_DW4)
    lds_warp_off = J.gpr("su32", warp_m_off * lds_stride_b)

    # each vm-load-dw4 can load 8 rows (since K=128bytes)
    # since tok-ids are discrete, we need a vmem_off for each load
    vm_load_cnt = len(range(0, nbM * 16, 8*num_warps))
    vmem_voff = J.gpr(vm_load_cnt, "vu32")

    ds_vaddr = J.gpr(row * J.sizeof_DW + lds_sorted_ids)

    for m in range(vm_load_cnt):
        J.ds_read_b32(vmem_voff[m], ds_vaddr + m*num_warps*8*J.sizeof_DW)

    J.s_waitcnt(mod=f"lgkmcnt(0)")

    for m in range(vm_load_cnt):
        tokid = J.gpr(2, "vu32", vmem_voff[m] & 0xFFFFFF, vmem_voff[m] >> 24)
        if not gate_up:
            vmem_voff[m] = tokid[0]*(TOPK*stride_b) + tokid[1]*stride_b + swizzle_col * J.sizeof_DW4
        else:
            vmem_voff[m] = tokid[0]*stride_b + swizzle_col * J.sizeof_DW4

        # don't need following code, since Buffer size ensures no read overflow can happen
        with J.ExecMask(tokid[0] >= num_tokens[0]):
            vmem_voff[m] = 0

    def vm_load(lds_offset):
        J.s_mov_b32("m0", lds_warp_off + lds_offset)
        for m in range(vm_load_cnt):
            yield 1
            buff.load_dwordx4(None, vmem_voff[m], 0, offset12=0)
            J.s_addk_i32("m0", 256*J.sizeof_DW4)
            vmem_voff[m] += nbK * 4 * J.sizeof_DW4

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
    return vm_load, vm_load_cnt, ds_read_1kb

import os
EIDX = int(os.getenv("EIDX","0"))

@pyhip.jit(with_debug_log=False)
def moe_gemm_down_mxfp4(J, wg_M, wg_N,
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
    # K dimension is too small
    # to reduce prelog/epilog overhead, we need to loop more
    # over N dimension too.
    assert not gate_up

@pyhip.jit(with_debug_log=False)
def moe_gemm_final_reduce_bf16(J, TOPK, OC, input:"void*", output:"void*", num_tokens_wg:"int", num_big_wg:"int", num_tokens_total:"int"):
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

def test_moe_gemm_final_reduce_bf16():
    torch.set_printoptions(linewidth=3000, sci_mode=False, edgeitems=8, )
    torch.set_default_device('cuda')
    torch.manual_seed(0)

    TOPK = 8
    OC = 4096
    num_tokens_total = 24000
    input = torch.randn(num_tokens_total, TOPK, OC, dtype=torch.bfloat16)
    output = torch.empty(num_tokens_total, OC, dtype=torch.bfloat16)
    num_CU = torch.cuda.get_device_properties().multi_processor_count
    num_WG = num_CU * 2
    
    num_tokens_wg = num_tokens_total // num_WG
    num_extra_tokens = num_tokens_total % num_WG
    '''
    num_big_wg = num_extra_tokens
    if wg_id < num_big_wg:
        tok0 = wg_id * (1 + num_tokens_wg) # need to do 1 more 
        tok1 = tok0 + (1 + num_tokens_wg)
    else:
        tok_base = num_big_wg * (1 + num_tokens_wg)
        tok0 = tok_base + (wg_id - num_big_wg) * num_tokens_wg
        tok1 = tok0 + num_tokens_wg
    '''

    print(num_WG, num_tokens_wg, num_extra_tokens, num_tokens_total)
    moe_gemm_final_reduce_bf16([num_WG], [64], TOPK, OC,
                               input.data_ptr(),
                               output.data_ptr(),
                               num_tokens_wg, num_extra_tokens, num_tokens_total)
    
    ref = torch.zeros(num_tokens_total, OC, dtype=torch.float)
    for i in range(num_tokens_total):
        for t in range(TOPK):
            ref[i] += input[i, t]

    ref = ref.to(torch.bfloat16)
    for i in range(num_tokens_total):
        if not torch.allclose(ref[i], output[i]):
            print(i)
            print(ref[i])
            print(output[i])
            assert 0
    
    for _ in range(10):
        with pyhip.cuPerf(name="moe_gemm_final_reduce_bf16"):
            moe_gemm_final_reduce_bf16([num_WG], [64], TOPK, OC,
                                    input.data_ptr(),
                                    output.data_ptr(),
                                    num_tokens_wg, num_extra_tokens, num_tokens_total)
    assert 0


# no_pass=["pass_cse", "pass_dce", "pass_dse"]
@pyhip.jit(with_debug_log=False)
def moe_gemm_mxfp4(J, wg_M, wg_N,
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

    assert OC % wg_N == 0
    num_warps = 4
    nbN = J.div(wg_N, 16)
    nbM = J.div(wg_M, 16)
    nbK = 2 # 2 MFMA 16x16 
    stride_k = IC * J.sizeof_fp4x2
    stride_c = OC * J.sizeof_bf16
    stride_gate_up = J.div(J.div(OC, wg_N), 2) * wg_N * stride_k
    nrM = J.div(nbM, 2)
    nrN = J.div(nbN, 2)
    nrK = nbK
    stride_scale32x256 = J.div(IC*2, 256) * J.warp_size * J.sizeof_DW

    # J.show_gemm_buf(mfma_MN = 16, n_mfma_K = 4, wave_CNT = [2,2], wave_Size = [64, 64])
    
    n_idx = J.blockIdx.x # split along OC
    e_idx = J.blockIdx.y 
    s_e_id = J.gpr(1, 'su32')
    J.s_load_dword(s_e_id, sorted_expert_ids, e_idx[0] * 4)
    max_id = J.gpr(1, 'su32')
    J.s_load_dword(max_id, num_valid_ids, 0)
    J.s_waitcnt(mod=f"lgkmcnt(0)")

    J.debug_setup((e_idx[0] == 0) & (n_idx[0] == 0) & (J.warp_id[0] == 0))
    with J.If(e_idx[0] * wg_M >= max_id[0]):
        J.s_endpgm()
    
    if gate_up:
        output[:] += n_idx * (wg_N//2 * J.sizeof_bf16)
    else:
        output[:] += n_idx * (wg_N * J.sizeof_bf16)
    sorted_ids[:] += e_idx * (wg_M * J.sizeof_DW)
    sorted_weights[:] += e_idx * (wg_M * J.sizeof_DW)

    i_scale[:] += e_idx * (J.div(wg_M,32) * stride_scale32x256)
    w_scale[:] += s_e_id * (J.div(OC,32) * stride_scale32x256)
    i_scale[:] += (J.warp_id[0] // 2) * (J.div(wg_M//2, 32) * stride_scale32x256)
    
    sbuff_a = J.Buffer(i_scale, J.div(wg_M//2, 32) * stride_scale32x256)
    if gate_up:
        weight[:] += s_e_id * (OC * stride_k) + n_idx * (wg_N//2 * stride_k)

        # B matrix scale is also interleaved
        # scale is blocked in unit of [32,256,fp4] 256 bytes, then layout in [OC, IC] style
        # so wg_N must also be in unit of 32.
        # to interleave gate/up in 2x2 waves, wg_N needs to be at least 32*4 = 128
        #
        #     warp0/2          warp1/3
        #   gatex32 upx32 | gatex32 upx32
        #    ...     ...  |  ...     ...
        #

        #   stride_scale32x256 = J.div(IC*2, 256) * J.warp_size * J.sizeof_DW
        #
        #
        w_scale[:] += n_idx * (J.div(wg_N//2, 32) * stride_scale32x256) + (J.warp_id[0] % 2) * (J.div(wg_N//4, 32) * stride_scale32x256)
        # gate-scale buff + up-scale buff
        sbuff_b = [None, None]
        sbuff_b[0] = J.Buffer(w_scale, J.div(wg_N//4, 32) * stride_scale32x256)
        w_scale[:] += J.div(OC//2, 32) * stride_scale32x256
        sbuff_b[1] = J.Buffer(w_scale, J.div(wg_N//4, 32) * stride_scale32x256)
    else:
        weight[:] += s_e_id * (OC * stride_k) + n_idx * (wg_N * stride_k)
        w_scale[:] += n_idx * (J.div(wg_N, 32) * stride_scale32x256) + (J.warp_id[0] % 2) * (J.div(wg_N//2, 32) * stride_scale32x256)
        sbuff_b = J.Buffer(w_scale, J.div(wg_N//2, 32) * stride_scale32x256)

    if gate_up:
        buff_a = J.Buffer(input, num_tokens * stride_k)
    else:
        buff_a = J.Buffer(input, num_tokens * TOPK * stride_k)

    ldsA = [J.alloc_lds(nbM * nbK * 1024), J.alloc_lds(nbM * nbK * 1024)]
    ldsB = [J.alloc_lds(nbN * nbK * 1024), J.alloc_lds(nbN * nbK * 1024)]

    warp_m = J.gpr((J.warp_id[0] // 2)*nrM)
    warp_n = J.gpr((J.warp_id[0] % 2)*nrN)

    # load sorted_ids into LDS
    lds_sorted_ids = J.alloc_lds(wg_M * J.sizeof_DW)
    lds_sorted_weights = J.alloc_lds(wg_M * J.sizeof_DW)
    J.wg_load_lds(lds_sorted_ids, sorted_ids, wg_M * J.sizeof_DW, wait_barrier = False)
    J.wg_load_lds(lds_sorted_weights, sorted_weights, wg_M * J.sizeof_DW, wait_barrier = True)

    vm_load_a, vm_load_cnt_a, ds_read_a = get_loader_sorted_tok(J, buff_a, lds_sorted_ids, nbM, nbK, stride_k, warp_m, gate_up, TOPK, num_tokens)
    vm_load_b, vm_load_cnt_b, ds_read_b = get_loader_b_preshuffled(J, weight, wg_N, nbN, nbK, stride_k, warp_n, gate_up, stride_gate_up)

    mfma_A = J.gpr(2, nrM, 4, "vbf16x2")
    mfma_B = J.gpr(2, nrN, 4, "vbf16x2")
    mfma_C = J.gpr(nrM, nrN, 4, "vf32")
    
    mfma_Ascale = J.gpr(2, J.div(nrM, 2), "vu32") # 4
    mfma_Bscale = J.gpr(2, J.div(nrN, 2), "vu32") # 4

    def mfma(reg_id, lds_id):
        # lds_id : scale register is grouped by lds_id
        # src0: Matrix A scale {OP_SEL_HI [0], OP_SEL[0]} defines which part of scale is used by the Matrix A of MFMA instruction.
        # src1: Matrix B scale {OP_SEL_HI [1], OP_SEL[1]} defines which part of scale is used by the Matrix B of MFMA instruction.
        for m in range(nrM):
            for n in range(nrN):
                sel_scale_B = (n & 1) + (reg_id & 1)*2
                sel_scale_A = (m & 1) + (reg_id & 1)*2
                mod = f"op_sel:[{sel_scale_B & 1}, {sel_scale_A & 1},0] op_sel_hi:[{sel_scale_B//2}, {sel_scale_A//2}, 0] cbsz:4 blgp:4"

                # J.v_mfma_f32_16x16x32_bf16(mfma_C[m,n], mfma_B[reg_id, n], mfma_A[reg_id, m], mfma_C[m,n])
                J.v_mfma_scale_f32_16x16x128_f8f6f4(mfma_C[m,n], mfma_B[reg_id, n], mfma_A[reg_id, m], mfma_C[m,n],
                                                    mfma_Bscale[lds_id, n//2],
                                                    mfma_Ascale[lds_id, m//2],
                                                    mod=mod)
                yield 16

    J.emit(vm_load_a(ldsA[0]))
    J.emit(vm_load_b(ldsB[0]))

    # load scales
    vaddr_scale = J.gpr("vu32", J.lane_id[0] * J.sizeof_DW)

    def load_next_scales(index):
        vaddr = J.gpr("vu32", vaddr_scale[0])
        for ii in range(J.div(nrM, 2)):
            sbuff_a.load_dword(mfma_Ascale[index & 1, ii], vaddr, 0)
            vaddr[0] += stride_scale32x256
            yield 1
        if gate_up:
            #
            vaddr = J.gpr("vu32", vaddr_scale[0])
            assert nrN == 4
            for ii in range(J.div(nrN, 2)//2):
                sbuff_b[0].load_dword(mfma_Bscale[index & 1, 2*ii + 0], vaddr, 0) # gate
                sbuff_b[1].load_dword(mfma_Bscale[index & 1, 2*ii + 1], vaddr, 0) # up
                vaddr[0] += stride_scale32x256
                yield 1
        else:
            vaddr = J.gpr("vu32", vaddr_scale[0])
            for ii in range(J.div(nrN, 2)):
                sbuff_b.load_dword(mfma_Bscale[index & 1, ii], vaddr, 0)
                vaddr[0] += stride_scale32x256
                yield 1
        vaddr_scale[0] += J.sizeof_DW * 64

    num_scale_loads = J.div(nrM, 2) + J.div(nrN, 2)

    J.emit(load_next_scales(0))

    J.emit(vm_load_a(ldsA[1]))
    J.emit(vm_load_b(ldsB[1]))
    mfma_C[...] = 0

    '''
    ab0: mfma ab0 | ds_read ab1; wait_lgkmcnt(0), barrier; vm-load a01 | load mfma_Ascale[1]
    ab1: mfma ab1 | vm-load b01; wait_vmcnt, barrier, ds_read ab2      | load mfma_Bscale[1]

    ab2: mfma ab2 | ds_read ab3; wait_lgkmcnt(0), barrier; vm-load  a23 | load mfma_Ascale[0]
    ab3: mfma ab3 | vm-load b23;  wait_vmcnt, barrier, ds_read ab0      | load mfma_Bscale[0]
    '''
    J.s_waitcnt(mod=f"vmcnt({vm_load_cnt_b + vm_load_cnt_a})")
    J.s_barrier()

    # ds_read ab0
    for m in range(nrM): ds_read_a(ldsA[0], mfma_A[0, m], m, 0)
    for n in range(nrN): ds_read_b(ldsB[0], mfma_B[0, n], n, 0)
    J.s_waitcnt(mod=f"lgkmcnt(0)")

    #for n in range(nrN): J.debug_log(mfma_B[0, n], torch.int32, "4h.16v.4h")
    #for m in range(nrM): J.debug_log(mfma_A[0, m], torch.int32, "4h.16v.4h")
    #J.s_endpgm()

    def loop_body(lds_id):
        # mfma ab0
        mfma_ab0 = mfma(0, lds_id)

        load_s = load_next_scales(lds_id + 1)

        # ds_read ab1
        for m in range(nrM):
            J.emit(mfma_ab0, 16)
            ds_read_a(ldsA[lds_id], mfma_A[1, m], m, 1)
            J.emit(load_s, 1)
        for n in range(nrN):
            J.emit(mfma_ab0, 16)
            ds_read_b(ldsB[lds_id], mfma_B[1, n], n, 1)
            J.emit(load_s, 1)

        for ii in range(4):
            J.emit(mfma_ab0, 16)
            J.emit(load_s, 1)

        J.emit(load_s)

        J.s_waitcnt(mod=f"lgkmcnt(0)")
        J.emit(mfma_ab0, 16)
        J.s_barrier()

        mfma_ab1 = mfma(1, lds_id)

        # vm-load a01
        vm_load = vm_load_a(ldsA[lds_id])
        J.emit([mfma_ab0, mfma_ab1], 16)
        J.emit(vm_load, 1) # first emit produce preparing instructions
        J.emit([mfma_ab0, mfma_ab1], 16)
        for _ in range(vm_load_cnt_a):
            J.emit(vm_load, 1)
            J.emit([mfma_ab0, mfma_ab1], 96)
        J.emit(vm_load)

        # vm-load b01
        vm_load = vm_load_b(ldsB[lds_id])
        J.emit([mfma_ab0, mfma_ab1], 16)
        J.emit(vm_load, 1) # first emit produce preparing instructions
        J.emit([mfma_ab0, mfma_ab1], 16)
        for _ in range(vm_load_cnt_b - 4):
            J.emit(vm_load, 1)
            J.emit([mfma_ab0, mfma_ab1], 96)

        J.emit(mfma_ab0) # emit all MFMA using AB register0 (since ds_read will override it)

        # wait vm-load a23/b23 to finish
        J.s_waitcnt(mod=f"vmcnt({vm_load_cnt_b + vm_load_cnt_a - 4})")
        #J.s_waitcnt(mod=f"vmcnt(0)")
        J.s_barrier()

        J.emit(vm_load, 1)
        # ds_read ab2
        for m in range(nrM):
            J.emit(mfma_ab1, 16)
            ds_read_a(ldsA[(lds_id + 1)&1], mfma_A[0, m], m, 0)

        J.emit(vm_load, 1)
        for n in range(nrN):
            J.emit(mfma_ab1, 16)
            ds_read_b(ldsB[(lds_id + 1)&1], mfma_B[0, n], n, 0)

        J.emit(vm_load, 1)
        J.emit(mfma_ab1, 96)
        J.emit(vm_load, 1)
        J.emit(mfma_ab1)
        J.s_waitcnt(mod=f"lgkmcnt(0)")

    # K is in unit of byte/fp4x2, each loop handles 64bytes
    wg_K = 128
    K = IC
    koff = J.gpr("su32", 0)
    loop_cnt = K // (2*wg_K)
    with J.While(koff[0] < loop_cnt):
        loop_body(0)
        loop_body(1)
        koff[0] += 1

    if K % (2*wg_K):
        loop_body(0)

    # for n in range(nrN): J.debug_log(mfma_C[0, n], torch.float, "4h.16v.4h")

    for lds in ldsA: J.free_lds(lds)
    for lds in ldsB: J.free_lds(lds)

    if gate_up:
        vrows = J.gpr(nrM, "vu32")
        vaddrs = J.gpr(nrM, "vu32")
        vweights = J.gpr(nrM, 2, "vf32")
        for m in range(nrM):
            row = (J.lane_id % 16) + warp_m*16 + m*16
            J.ds_read_b32(vrows[m], row * J.sizeof_DW + lds_sorted_ids)

        J.s_waitcnt(mod=f"lgkmcnt(0)")

        col = J.lane_id // 16
        swap_12_col = (col & 1) * 2 + (col >> 1)

        stride_c = OC//2 * J.sizeof_bf16
        for m in range(nrM):
            #J.s_waitcnt(mod=f"lgkmcnt({nrM-1-m})")
            topk = J.gpr(vrows[m] >> 24)
            vrows[m] = vrows[m] & 0xFFFFFF
            vaddrs[m] = vrows[m] * (TOPK * stride_c) +  topk*stride_c + swap_12_col * J.sizeof_DW4 + warp_n * ((16//2) * J.sizeof_bf16)

        # gate 0,1 up 2,3
        # 
        assert nrN == 4
        assert nrN % 2 == 0

        for m in range(nrM):
            with J.ExecMask(vrows[m] < num_tokens[0]):
                out0 = J.gpr(4, "vf32")
                out1 = J.gpr(4, "vf32")
                vbf16 = J.gpr(4, "vbf16x2")

                if mfma_C.rtype == "a":
                    vf32 = J.gpr(2, 4, "vf32")
                    for i in range(4):
                        J.v_accvgpr_read_b32(vf32[0,i], mfma_C[m, 0, i])
                        J.v_accvgpr_read_b32(vf32[1,i], mfma_C[m, 2, i])
                    for i in range(4):
                        out0[i] = vf32[1, i] * J.silu(vf32[0, i])

                    for i in range(4):
                        J.v_accvgpr_read_b32(vf32[0,i], mfma_C[m, 1, i])
                        J.v_accvgpr_read_b32(vf32[1,i], mfma_C[m, 3, i])

                    for i in range(4):
                        out1[i] = vf32[1, i] * J.silu(vf32[0, i])
                else:
                    for i in range(4):
                        out0[i] = mfma_C[m, 2, i] * J.silu(mfma_C[m, 0, i])
                        out1[i] = mfma_C[m, 3, i] * J.silu(mfma_C[m, 1, i])

                
                J.uni_cvt_pk_bf16_f32(vbf16[0], out0[0], out0[1])
                J.uni_cvt_pk_bf16_f32(vbf16[1], out0[2], out0[3])

                J.uni_cvt_pk_bf16_f32(vbf16[2], out1[0], out1[1])
                J.uni_cvt_pk_bf16_f32(vbf16[3], out1[2], out1[3])
                #    a0    a1   a2   a3   | 01 23
                #    b0    b1   b2   b3   | 45 67
                #  v_permlane16_swap_b32(a, b)
                #    a0    b0   a2   b2   |
                #    a1    b1   a3   b3   |
                #
                # swap of row 1 & 2 are done by swapping lane-address 
                J.v_permlane16_swap_b32(vbf16[0], vbf16[2])
                J.v_permlane16_swap_b32(vbf16[1], vbf16[3])
                J.global_store_dwordx4(vaddrs[m], vbf16, output, mod=f"offset:{0}")
    elif 1:
        vrows = J.gpr(nrM, "vu32")
        vaddrs = J.gpr(nrM, "vu32")
        vweights = J.gpr(nrM, 2, "vf32")
        for m in range(nrM):
            row = (J.lane_id % 16) + warp_m*16 + m*16
            J.ds_read_b32(vrows[m], row * J.sizeof_DW + lds_sorted_ids)
            J.ds_read2_b32(vweights[m], row * J.sizeof_DW + lds_sorted_weights)

        J.s_waitcnt(mod=f"lgkmcnt(0)")

        col = J.lane_id // 16
        swap_12_col = (col & 1) * 2 + (col >> 1)

        stride_c = OC * J.sizeof_bf16
        for m in range(nrM):
            #J.s_waitcnt(mod=f"lgkmcnt({nrM-1-m})")
            topk = J.gpr(vrows[m] >> 24)
            vrows[m] = vrows[m] & 0xFFFFFF
            vaddrs[m] = vrows[m] * (TOPK * stride_c) +  topk*stride_c + swap_12_col * J.sizeof_DW4 + warp_n * (4 * J.sizeof_DW2)

        # gate 0,1 up 2,3
        # 
        assert nrN == 4
        assert nrN % 2 == 0

        for m in range(nrM):
            with J.ExecMask(vrows[m] < num_tokens[0]):
                for n in range(0, nrN, 2):
                    vbf16 = J.gpr(4, "vbf16x2")
                    vf32x4 = J.gpr(4, "vf32")
                    J.v_pk_mul_f32(vf32x4[0:1], mfma_C[m,n,0:1], vweights[m])
                    J.v_pk_mul_f32(vf32x4[2:3], mfma_C[m,n,2:3], vweights[m])
                    J.uni_cvt_pk_bf16_f32(vbf16[0], vf32x4[0], vf32x4[1])
                    J.uni_cvt_pk_bf16_f32(vbf16[1], vf32x4[2], vf32x4[3])

                    J.v_pk_mul_f32(vf32x4[0:1], mfma_C[m,n+1,0:1], vweights[m])
                    J.v_pk_mul_f32(vf32x4[2:3], mfma_C[m,n+1,2:3], vweights[m])
                    J.uni_cvt_pk_bf16_f32(vbf16[2], vf32x4[0], vf32x4[1])
                    J.uni_cvt_pk_bf16_f32(vbf16[3], vf32x4[2], vf32x4[3])
                    J.v_permlane16_swap_b32(vbf16[0], vbf16[2])
                    J.v_permlane16_swap_b32(vbf16[1], vbf16[3])
                    J.global_store_dwordx4(vaddrs[m], vbf16, output, mod=f"offset:{n//2 * 16*4}")
    else:
        vweights = J.gpr(nrM, 2, "vf32")
        for m in range(nrM):
            row = (J.lane_id % 16) + warp_m*16 + m*16
            J.ds_read2_b32(vweights[m], row * J.sizeof_DW + lds_sorted_weights)

        n_rows_per_loop = 256//64
        n_loops = J.div(wg_M, n_rows_per_loop)

        vrows = J.gpr(n_loops, "vu32")
        row = J.gpr((J.threadIdx.x[0] // 64) * J.sizeof_DW + lds_sorted_ids)
        for m in range(n_loops):
            J.ds_read_b32(vrows[m], row)
            row[0] += n_rows_per_loop * J.sizeof_DW

        J.s_waitcnt(mod=f"lgkmcnt(0)")

        layout2 = J.Layout([128, 128], "bf16", 0)
        lds = J.alloc_lds(layout2.total_size())

        row = J.lane_id[0] % 16
        col = J.lane_id[0] // 16

        # J.show_mfma_in_lds(mfma_MN=16, num_mfmas=8, swizzle_2=-2, lane_bytes=8);    assert 0
        voffset = []
        for n in range(nrN):
            swizzle_col = (warp_n*4 + n*4 + col) ^ (row * 2)
            voff, _ = layout2[warp_m*16 + row, swizzle_col*4]
            voffset.append(voff)

        for m in range(nrM):
            for n in range(nrN):
                vbf16x4 = J.gpr(2, "vbf16x2")
                vf32x4 = J.gpr(4, "vf32")
                J.v_pk_mul_f32(vf32x4[0:1], mfma_C[m,n,0:1], vweights[m])
                J.v_pk_mul_f32(vf32x4[2:3], mfma_C[m,n,2:3], vweights[m])
                J.uni_cvt_pk_bf16_f32(vbf16x4[0], vf32x4[0], vf32x4[1])
                J.uni_cvt_pk_bf16_f32(vbf16x4[1], vf32x4[2], vf32x4[3])
                J.ds_write_b64(voffset[n], vbf16x4, mod=f"offset:{lds}")
                voffset[n][0] += 16 * layout2.stride(0)

        J.s_waitcnt(mod=f"lgkmcnt(0)")
        J.s_barrier()

        layout_src = J.Layout([128, 128//2], "bf16x2", 0)

        row = J.gpr(J.threadIdx.x[0] // layout_src.size(1))
        col = J.gpr(J.threadIdx.x[0] % layout_src.size(1))

        vdata = J.gpr(n_loops, "vbf16x2")
        i_row = 0
        for m in range(n_loops):
            ds_row = J.gpr("vu32", row[0] + i_row)
            swizzle_col = (col[0]) ^ ((ds_row % 16) * 4)
            voffset, _ = layout_src[ds_row[0], swizzle_col]
            J.ds_read_b32(vdata[m], voffset, mod=f"offset:{lds}")
            i_row += n_rows_per_loop

        J.s_waitcnt(mod=f"lgkmcnt(0)")

        for m in range(n_loops):
            vmem_row = J.gpr(vrows[m] & 0xFFFFFF)
            with J.ExecMask(vmem_row < num_tokens):
                vaddr = J.gpr(vmem_row * stride_c + col * J.sizeof("bf16x2"))
                J.global_atomic_pk_add_bf16(vaddr, vdata[m], output)

def de_shuffle_weight(weight, mfma_MN = 16):
    M, K = weight.shape
    K_bytes = K * weight.itemsize
    sizeof_DW4 = 16
    mfma_K_lanes = 64 // mfma_MN
    mfma_K_L = sizeof_DW4//weight.itemsize
    mfma_K = mfma_K_lanes * mfma_K_L 

    assert M % mfma_MN == 0
    mfma_K_bytes = mfma_K_lanes * sizeof_DW4
    assert K_bytes % mfma_K_bytes == 0
    #x = x.reshape(M//mfma_MN, mfma_MN, K//mfma_K, mfma_K_lanes, mfma_K_L)
    #x = x.permute(0,2,3,1,4)

    assert K % mfma_K == 0
    weight = weight.reshape(M//mfma_MN, K//mfma_K, mfma_K_lanes, mfma_MN, mfma_K_L)
    weight = weight.permute(0,3,1,2,4)
    weight = weight.reshape(M, K).contiguous()
    return weight


import aiter
from aiter.utility import fp4_utils
from aiter.ops.shuffle import shuffle_weight

def mxfp4_dequant(quant, scale):
    _src = fp4_utils.mxfp4_to_f32(quant.view(torch.float4_e2m1fn_x2)).to(dtype=torch.bfloat16)
    rows, cols = _src.shape
    #return _src
    #print(rows, cols, _src.shape)
    _scale = fp4_utils.e8m0_to_f32(scale).to(dtype=torch.bfloat16)
    _scale = _scale.reshape(rows//32, cols//256, 4, 1, 16, 4).repeat(1,1,1,32,1,1).view(rows//32, cols//256, 128, 16, 4).permute(0,1,4,3,2)
    for r in range(0, rows, 32):
        for c in range(0, cols, 256):
            sss = _scale[r//32, c//256, :, :, :]
            _src[r:r+16, c:c+128] *= sss[0]
            _src[r+16:r+32, c:c+128] *= sss[1]
            _src[r:r+16, c+128:c+256] *= sss[2]
            _src[r+16:r+32, c+128:c+256] *= sss[3]

    return _src

def moe_gemm_ref(BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N, gate_up,
                    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids,
                    weight, w_scale, input, i_scale, output, debug_e_idx = -1, EIDX = None):
    M = input.shape[0]

    if "_x2" in str(weight.dtype):
        print(f"{BLOCK_TILE_SIZE_M=}")
        print(f"{BLOCK_TILE_SIZE_N=}")
        print(f"{input.shape=} {input.dtype=}")         # input.shape=[24064, 2048] input.dtype=torch.float4_e2m1fn_x2
        if i_scale is not None:
            print(f"  {i_scale.shape} {i_scale.dtype}")   # [208896, 128] torch.float8_e8m0fnu
        print(f"{output.shape=}")                       # output.shape = [24064, 8, 1536]
        print(f"{weight.shape=} {weight.dtype=}")       # weight.shape=[128, 3072, 2048] weight.dtype=torch.float4_e2m1fn_x2
        if w_scale is not None:
            print(f"  {w_scale.shape} {w_scale.dtype}")   # [393216 (128*3072), 128] torch.float8_e8m0fnu
        print(f"{sorted_ids.shape=}")                   # sorted_ids.shape=torch.Size([208888])
        print(f"{sorted_expert_ids.shape=}")            # sorted_expert_ids.shape=torch.Size([1632])
        print(f"{sorted_weights.shape=}")               # sorted_weights.shape=torch.Size([208888])
        print(f"{num_valid_ids=}")

    NUM_EXPERTS, OC, IC = weight.shape
    if "_x2" in str(weight.dtype): IC *= 2
    NUM_BLOCKS = sorted_expert_ids.shape[0]

    if i_scale is not None:
        input = input.view(torch.int8)
        print(f"===== i_scale {i_scale.shape} {i_scale.dtype}")
    if w_scale is not None:
        weight = weight.view(torch.int8)
        w_scale = w_scale.view(NUM_EXPERTS, OC, -1)
        print(f"===== w_scale {w_scale.shape} {w_scale.dtype}")

    num_sorted_ids = sorted_ids.shape[0]
    
    if EIDX is None:
        EIDX = sorted_expert_ids.shape[0]
    #for e_idx in range(sorted_expert_ids.shape[0]):
    for e_idx in range(EIDX):
        #for e_idx in range(1):
        #e_idx = EIDX
        s_e_id = sorted_expert_ids[e_idx]
        max_id = num_valid_ids[0]
        if e_idx * BLOCK_TILE_SIZE_M >= max_id: continue
        i0 = e_idx*BLOCK_TILE_SIZE_M
        i1 = (e_idx+1)*BLOCK_TILE_SIZE_M

        ids = sorted_ids[i0:i1].clone()
        valid_mask = (ids & 0xFFFFFF) < torch.tensor(M)
        ids[(ids & 0xFFFFFF) >= torch.tensor(M)] = 0
        tok_ids = ids & 0xFFFFFF
        top_k = ids >> 24
        tok_w = sorted_weights[i0:i1]
        expert_w = weight[s_e_id, ...]
        if 0:
            print("====================== tok_ids")
            print(tok_ids)
            print("====================== top_k")
            print(top_k)
            print("====================== s_e_id")
            print(s_e_id)
            print("====================== tok_w")
            print(tok_w)

            if e_idx == -11:
                print(tok_ids)

        if gate_up:
            src = input[tok_ids,...]
        else:
            src = input[tok_ids, top_k, ...]

        if 0:
            print("??????????? ", src.shape, src.dtype)
            print(src.view(torch.int32)[:16, :16])
            

        if i_scale is not None:
            src = mxfp4_dequant(src, i_scale[i0:i1,...])

        w = de_shuffle_weight(expert_w)
        if 0:
            print(w.shape, w.dtype, w.view(torch.int32).shape)
            wi32_gate = w.view(torch.int32)
            wi32_up = w[OC//2:,:].view(torch.int32)
            print(wi32_gate[:16, :16])
            print(wi32_up[:16, :16])
            print(wi32_gate[16:16*2, :16])
            print(wi32_up[16:16*2, :16])
            
        #assert 0

        if debug_e_idx == e_idx:
            print(w.view(torch.int32)[:16, :16])

        if w_scale is not None:
            w = mxfp4_dequant(w, w_scale[s_e_id,...])

        act = src @ w.t()

        if 0:
            m0,n0 = 0,0
            print(act[m0:m0+16, n0:n0+32])
            n0 += OC//2
            print(act[m0:m0+16, n0:n0+32])

        if gate_up:
            act = torch.nn.functional.silu(act[:, :(OC//2)]) * (act[:,(OC//2):])

        if debug_e_idx == e_idx:
            print(f"======== ===========")
            print(tok_ids)
            print(f"======== {debug_e_idx=}  {act.shape} {act.dtype}")
            m0 = 0*(BLOCK_TILE_SIZE_M//2)
            n0 = (0)*(BLOCK_TILE_SIZE_N//2)
            print(act[m0:m0+16, n0:n0+64])
            print(act[m0, n0:n0+64].tolist())

        if gate_up:
            output[tok_ids[valid_mask], top_k[valid_mask], :] = act[valid_mask, ...]
        else:
            output[tok_ids[valid_mask], ...] += act[valid_mask, ...] * tok_w[valid_mask, None]

def calc_diff(x: torch.Tensor, y: torch.Tensor):
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    if denominator == 0:    # Which means that all elements in x and y are 0
        return 0.0
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim

def test_gateup():
    torch.set_printoptions(linewidth=3000, sci_mode=False, edgeitems=8, )
    torch.set_default_device('cuda')
    torch.manual_seed(0)

    E = 128
    TOPK = 8
    TILE_M = 128
    TILE_N = 128

    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, w1, w1_scale,hidden_states_q, hidden_states_scale, gemm1_out = torch.load('tensors_tuple2.pt')
    
    print(sorted_expert_ids[127])
    print(w1.shape, w1.dtype)
    print(w1_scale.shape, w1_scale.dtype)

    B = hidden_states_q.shape[0]

    EIDXs = None
    #EIDXs = sorted_expert_ids.shape[0]
    #EIDXs = 1
    #assert 0

    gemm1_out[...] = 0
    the_out = gemm1_out.clone()
    moe_gemm_ref(TILE_M, TILE_N, True, sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids,
                    w1, w1_scale,
                    #hidden_states, None,
                    hidden_states_q, hidden_states_scale,
                    gemm1_out, EIDX=EIDXs)
    print(w1.shape, w1.dtype)
    gateup_OC = w1.shape[1]
    assert gateup_OC % TILE_N == 0
    num_oc_blocks = gateup_OC // TILE_N
    num_e_blocks = sorted_expert_ids.shape[0]
    print(f"{num_oc_blocks=} {num_e_blocks=}")
    moe_gemm_mxfp4([num_oc_blocks, 1],[256],
        TILE_M, TILE_N,
        w1.shape[0], w1.shape[1], w1.shape[2], 
        True, TOPK, # gate_up,
        sorted_ids.data_ptr(),
        sorted_weights.data_ptr(),
        sorted_expert_ids.data_ptr(),
        num_valid_ids.data_ptr(),
        w1.data_ptr(), w1_scale.data_ptr(),
        hidden_states_q.data_ptr(), hidden_states_scale.data_ptr(),
        the_out.data_ptr(), B)
    if 1:
        print("===============", EIDXs, sorted_expert_ids.shape[0], num_valid_ids)
        for i in range(B):
            for t in range(TOPK):
                diff = calc_diff(gemm1_out[i,t,:], the_out[i,t,:])
                if diff > 0.02:
                    print(f"=============== {i},{t}   {diff:.3f}")
                    print(f"     gemm1_out ")
                    print(gemm1_out[i,t,:].view(-1,64))
                    print("      the_out ")
                    print(the_out[i,t,:].view(-1,64))
                    assert 0

def test_down():
    torch.set_printoptions(linewidth=3000, sci_mode=False, edgeitems=8, )
    torch.set_default_device('cuda')
    torch.manual_seed(0)

    E = 128
    TOPK = 8
    HIDDEN_SIZE = 4096
    INTER_SIZE_TP = 768
    TILE_M = 128
    TILE_N = 128
    # torch.save((sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, w2, w2_scale, gemm1_out_q, gemm1_out_scale, cur_out), 'tensors_tuple.pt')
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, w2, w2_scale, gemm1_out_q, gemm1_out_scale, cur_out = torch.load('tensors_tuple.pt')

    B = gemm1_out_q.shape[0] // TOPK
    
    gate_up = False 
    if 0:
        print(gemm1_out_q.shape)
        print(gemm1_out_scale.shape, gemm1_out_scale.dtype)
        print(w2_scale.shape, w2_scale.dtype)
        
        #assert 0
        #assert 0
        # sorted_ids[TILE_M + 32] = sorted_ids[TILE_M + 33]
        #swap = sorted_ids[TILE_M:TILE_M*2].clone()
        sorted_ids[TILE_M:TILE_M*2] = sorted_ids[:TILE_M]
        sorted_weights[TILE_M:TILE_M+TILE_M] = sorted_weights[:TILE_M]
        cnnt0, cnnt1 = 32, 64
        #cnnt0, cnnt1 = 0, 32
        print(">>>>>>>>>>>>>")
        print(gemm1_out_scale[cnnt0:cnnt1])
        print(">>>>>>>>>>>>>")
        print(gemm1_out_scale[TILE_M+cnnt0:TILE_M+cnnt1])

        gemm1_out_scale[31] = 0.3

        for k in range(gemm1_out_scale.shape[0]//TILE_M):
            if not torch.all(gemm1_out_scale[k*TILE_M+cnnt0:k*TILE_M+cnnt1] == 0).item():
                print(k, gemm1_out_scale[k*TILE_M+cnnt0:k*TILE_M+cnnt1])

        #assert 0
        gemm1_out_scale[TILE_M+cnnt0:TILE_M+cnnt1] = gemm1_out_scale[cnnt0:cnnt1]
        sorted_expert_ids[1] = sorted_expert_ids[0]

        #sorted_ids[:TILE_M] = swap[...]
        
        #sorted_ids[:TILE_M] = sorted_ids[TILE_M:TILE_M*2]

    the_out = cur_out.clone()
    moe_gemm_ref(TILE_M, TILE_N, False, sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids,
                    w2, w2_scale, 
                    #gemm1_out, None,
                    gemm1_out_q.view(B, TOPK, -1), gemm1_out_scale,
                    cur_out)

    print(">>>>>>>>>>>>>>>>>>>>>>")
    for iii in range(0,sorted_ids.shape[0],TILE_M):
        print((sorted_ids[iii:iii+TILE_M] & 0xFFFFFF).tolist())
    print(">>>>>>>>>>>>>>>>>>>>>>")

    down_OC = w2.shape[1]
    assert down_OC % TILE_N == 0
    num_oc_blocks = down_OC // TILE_N
    num_e_blocks = sorted_expert_ids.shape[0]
    print(f"{num_oc_blocks=} {num_e_blocks=}")
    moe_gemm_mxfp4([num_oc_blocks, num_e_blocks],[256],
        TILE_M, TILE_N,
        w2.shape[0], w2.shape[1], w2.shape[2], 
        False, TOPK, # gate_up,
        sorted_ids.data_ptr(),
        sorted_weights.data_ptr(),
        sorted_expert_ids.data_ptr(),
        num_valid_ids.data_ptr(),
        w2.data_ptr(), w2_scale.data_ptr(),
        gemm1_out_q.data_ptr(), gemm1_out_scale.data_ptr(),
        the_out.data_ptr(), B)
    print("=============== cur_out")
    print(cur_out[:16,:128])
    print("=============== the_out")
    print(the_out[:16,:128])
    print(f"=== {calc_diff(cur_out, the_out):.3f}")
    for ttt in range(B):
        diff = calc_diff(cur_out[ttt], the_out[ttt])
        if diff > 0.01:
            print(f"=============== {ttt} diff : {diff:.4f}")
            for oc_b in range(num_oc_blocks):
                co = cur_out[ttt, oc_b*TILE_N:oc_b*TILE_N + TILE_N]
                to = the_out[ttt, oc_b*TILE_N:oc_b*TILE_N + TILE_N]
                #print(f"        {co.view(-1,32)}")
                #print(f"        {to.view(-1,32)}")
                d = calc_diff(co, to)
                print(f"{d:.2f},", end="")
            print()
            assert 0

if __name__ == "__main__":

    test_moe_gemm_final_reduce_bf16()
    #test_down()
    #test_gateup()