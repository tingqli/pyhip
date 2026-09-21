from pyhip import jit, JIT
import torch

__all__ = [
    "moe_gemm_final_reduce_bf16",
    "moe_gemm_mxfp4_gateup_8wave",
    "moe_gemm_mxfp4_gateup_4wave",
    "moe_gemm_mxfp4"
]

def get_loader_b_preshuffled(J, weight, wg_N, nbM, nbK, stride_b, ibM0, gate_up, stride_gate_up, lds_woff=None):
    # 4 waprs load weight coorperatively into LDS
    #
    # weight:
    #    dtype:       int8/byte (type-less)
    #    shape:       [wg_N, ???]  
    #    stride:      stride_b
    #    preshuffled: [wg_N//16, ???//64] x [16,64] (1KB block)
    #
    # LDS:
    #    shape [wgN//16, nbK] x [16,64]
    # 
    num_warps = 4

    stride_1kb = J.div(16*stride_b, 1024)
    assert nbK == 2

    # warps coorperatively load weights in following configuration
    # but it doesn't mean they will compute with weights using the same configuration
    warp_id = J.gpr("su32", J.warp_id[0] % num_warps)   # 0,1, 2,3
    warp_k = warp_id % nbK                              # 0,1, 0,1
    warp_m = warp_id // nbK                             # 0,0, 1,1
    # each vm load can load [num_warps] = [2,2] 1KB blocks
    vm_load_cnt = J.div(nbM, num_warps//nbK)

    if gate_up:
        # gate-up, select buff based on warp_m
        weight[:] += warp_m * stride_gate_up
        buff = J.Buffer(weight, wg_N//2 * stride_b)
        vmem_warp_off = warp_k * 1024
        # interleaving gate (load by warp-0/1) with up (load by warp-2/3)
        # the interleave unit is [32 x 2xBK] due to scale unit size
        #
        # warp_id of each load_dwordx4 (which loads a 1KB block [16,64])
        # [wg_N//16, nbK] == [8, 2]
        #           0 0 2 2 | 0 0 2 2
        #           1 1 3 3 | 1 1 3 3
        # warp_m    0 0 1 1 | 0 0 1 1
        #
        # [wg_N//16, nbK] == [16, 2]
        #           0 0 0 0 2 2 2 2 | 0 0 0 0 2 2 2 2
        #           1 1 1 1 3 3 3 3 | 1 1 1 1 3 3 3 3
        # warp_m    0 0 0 0 1 1 1 1 | 0 0 0 0 1 1 1 1
        #
        lds_warp_stride0 = nbK * 1024
        lds_warp_stride1 = J.div(nbM, 4)*lds_warp_stride0
        step_m01 = J.div(nbM//2, num_warps//nbK)
        lds_warp_off = J.gpr("su32", warp_m * (step_m01 * nbK * 1024) + warp_k * 1024)
    else:
        buff = J.Buffer(weight, wg_N * stride_b)
        vmem_warp_off = warp_m * (stride_1kb * 1024) + warp_k * 1024
        lds_warp_off = J.gpr("su32", warp_m * (nbK * 1024) + warp_k * 1024)

    vmem_voff = J.gpr(J.lane_id[0] * J.sizeof_DW4 + vmem_warp_off)

    def vm_load(lds_offset):
        if lds_woff is not None:
            J.s_mov_b32("m0", lds_warp_off + lds_woff + lds_offset)
        else:
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

    # LDS layout
    #   B0 B2 B4 B6 ....
    #   B1 B3 B5 B7
    #
    # 1KB blocks are loaded into LDS in above order;
    # nbK is 2, which is height in 1KB blocks;
    # ibM0 is the warp-level offset along N/OC-dimension in 1KB regs;
    #
    voff = J.gpr(J.lane_id[0] * J.sizeof_DW4 + ibM0 * (nbK * 1024))
    voff2 = J.gpr("vu32", voff[0] + 64*1024)
    if lds_woff is not None:
        voff[0] += lds_woff
        voff2[0] += lds_woff

    def ds_read_1kb(lds, vdst, m, k):
        offset = lds + m*(nbK * 1024) + k*1024
        if offset >= 64*1024:
            voffset = voff2
            offset -= 64*1024
        else:
            voffset = voff
        J.ds_read_b128(vdst, voffset, mod=f"offset:{offset}")

    # return a loader constructor which can emit
    return vm_load, vm_load_cnt, ds_read_1kb

def get_loader_sorted_tok(J, buff, lds_sorted_ids, nbM, nbK, stride_b, ibM0, gate_up, TOPK, num_tokens, lds_woff=None):
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
    warp_m_off = (J.warp_id[0] % num_warps) * 8

    def swizzle(row, col):
        return (col ^ row) % 8
    
    threadIdx = J.gpr("vu32", J.threadIdx.x % (64*num_warps))

    col = threadIdx % 8
    row = threadIdx // 8
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
        if lds_woff is not None:
            J.s_mov_b32("m0", lds_warp_off + lds_woff + lds_offset)
        else:
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
    if lds_woff is not None:
        voff[0] += lds_woff
        voff2[0] += lds_woff

    def ds_read_1kb(lds, vdst, m, k):
        offset = lds + m*16*lds_stride_b
        if offset >= 64*1024:
            voffset = voff2[k]
            offset -= 64*1024
        else:
            voffset = voff[k]
        J.ds_read_b128(vdst, voffset, mod=f"offset:{offset}")
    return vm_load, vm_load_cnt, ds_read_1kb

@jit(with_debug_log=False)
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


'''
wg_M = 128, wg_N = 256
8-wave 才能占满整个CU，此时，如何显式调度8个wave遮盖内存开销？
HipKitten的方式可以简化问题，每4-wave一组，组内可以简单的设计两个阶段:
 - stage1: 读外存到LDS|ds_read到reg
 - stage2: MFMA计算

读外存到LDS进一步分为读A和读B，这两个操作都由8-wave协作完成，

CU共有寄存器：64*512*4*4 = 512 KB
累加寄存器：128*256*4 = 128 KB
A/B寄存器：128*128*2*2 = 64 KB

LDS缓冲最多可以到达3级：
    3缓冲 (128*128*3 + 256*128*3)/1024 ： 144 KB

4-wave: |  Load   | compute | Load    |
4-wave: | compute |  Load   | compute |

load由vmload和ds_read组成，vm_load占用一个LDS缓冲，ds_read占用另一个
load完成整个128x256的结果矩阵所需输入数据的LDS数据读入，

2组4-waves做完全无关的交织

prelog: vmload LDS1

 - vmload LDS0 | ds_read LDS1
 - compute r
 - vmload LDS1 | ds_read LDS0
 - compute r

首先实现上面的单4-wave的逻辑并保证正确
'''
@jit(with_debug_log=False)
def moe_gemm_mxfp4_gateup_8wave(J, wg_M, wg_N,
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
    assert gate_up == True
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

    warp_row = J.gpr("su32", J.warp_id[0] // 4)
    warp_col = J.gpr("su32", J.warp_id[0] % 4)

    # wave0123
    # wave4567
    #n_idx[0] *= 0
    n_idx[0] = n_idx[0] * 2 + warp_row[0]

    e_idx = J.blockIdx.y 
    #e_idx[0] *= 0
    s_e_id = J.gpr(1, 'su32')
    J.s_load_dword(s_e_id, sorted_expert_ids, e_idx[0] * 4)
    max_id = J.gpr(1, 'su32')
    J.s_load_dword(max_id, num_valid_ids, 0)
    J.s_waitcnt(mod=f"lgkmcnt(0)")

    J.debug_setup((e_idx[0] == 0) & (n_idx[0] == 0) & (J.warp_id[0] == 0))
    with J.If(e_idx[0] * wg_M >= max_id[0]):
        J.s_endpgm()
    
    output[:] += n_idx * (wg_N//2 * J.sizeof_bf16)
    sorted_ids[:] += e_idx * (wg_M * J.sizeof_DW)
    sorted_weights[:] += e_idx * (wg_M * J.sizeof_DW)

    i_scale[:] += e_idx * (J.div(wg_M,32) * stride_scale32x256)
    w_scale[:] += s_e_id * (J.div(OC,32) * stride_scale32x256)
    i_scale[:] += (warp_col[0] // 2) * (J.div(wg_M//2, 32) * stride_scale32x256)
    
    sbuff_a = J.Buffer(i_scale, J.div(wg_M//2, 32) * stride_scale32x256)

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
    w_scale[:] += n_idx * (J.div(wg_N//2, 32) * stride_scale32x256) + (warp_col[0] % 2) * (J.div(wg_N//4, 32) * stride_scale32x256)
    # gate-scale buff + up-scale buff
    sbuff_b = [None, None]
    sbuff_b[0] = J.Buffer(w_scale, J.div(wg_N//4, 32) * stride_scale32x256)
    w_scale[:] += J.div(OC//2, 32) * stride_scale32x256
    sbuff_b[1] = J.Buffer(w_scale, J.div(wg_N//4, 32) * stride_scale32x256)

    buff_a = J.Buffer(input, num_tokens * stride_k)

    LDSA_BUFF_SIZE = nbM * nbK * 1024
    LDSB_BUFF_SIZE = nbN * nbK * 1024
    lds_woff = J.gpr("su32", warp_row[0] * LDSA_BUFF_SIZE)
    ldsA = [J.alloc_lds(LDSA_BUFF_SIZE*2), J.alloc_lds(LDSA_BUFF_SIZE*2)]
    ldsB = [J.alloc_lds(LDSB_BUFF_SIZE*2), J.alloc_lds(LDSB_BUFF_SIZE*2)]

    warp_m = J.gpr((warp_col[0] // 2)*nrM)
    warp_n = J.gpr((warp_col[0] % 2)*nrN)

    # load sorted_ids into LDS
    lds_sorted_ids = J.alloc_lds(wg_M * J.sizeof_DW)
    #lds_sorted_weights = J.alloc_lds(wg_M * J.sizeof_DW)
    J.wg_load_lds(lds_sorted_ids, sorted_ids, wg_M * J.sizeof_DW, num_warps = 8, wait_barrier = True)
    #J.wg_load_lds(lds_sorted_weights, sorted_weights, wg_M * J.sizeof_DW, wait_barrier = True)

    vm_load_a, vm_load_cnt_a, ds_read_a = get_loader_sorted_tok(J, buff_a, lds_sorted_ids, nbM, nbK, stride_k, warp_m, gate_up, TOPK, num_tokens, lds_woff=lds_woff)
    vm_load_b, vm_load_cnt_b, ds_read_b = get_loader_b_preshuffled(J, weight, wg_N, nbN, nbK, stride_k, warp_n, gate_up, stride_gate_up, lds_woff=lds_woff)

    mfma_A = J.gpr(2, nrM, 4, "vbf16x2")
    mfma_B = J.gpr(2, nrN, 4, "vbf16x2")
    mfma_C = J.gpr(nrM, nrN, 4, "vf32")

    # load scales: ping-ponbg buffer register
    mfma_Ascale = J.gpr(2, J.div(nrM, 2), "vu32") # 4
    mfma_Bscale = J.gpr(2, J.div(nrN, 2), "vu32") # 4
    vaddr_scale = J.gpr("vu32", J.lane_id[0] * J.sizeof_DW)
    def load_next_scales(index):
        vaddr = J.gpr("vu32", vaddr_scale[0])
        for ii in range(J.div(nrM, 2)):
            sbuff_a.load_dword(mfma_Ascale[index & 1, ii], vaddr, 0)
            vaddr[0] += stride_scale32x256
            yield 1
        vaddr = J.gpr("vu32", vaddr_scale[0])
        assert nrN >= 4, f"{nrN=}"
        for ii in range(J.div(nrN, 2)//2):
            sbuff_b[0].load_dword(mfma_Bscale[index & 1, 2*ii + 0], vaddr, 0) # gate
            sbuff_b[1].load_dword(mfma_Bscale[index & 1, 2*ii + 1], vaddr, 0) # up
            vaddr[0] += stride_scale32x256
            yield 1
        vaddr_scale[0] += J.sizeof_DW * 64
    num_scale_loads = J.div(nrM, 2) + J.div(nrN, 2)

    #print(len(mfma_A) + len(mfma_B) + len(mfma_C) + len(mfma_Ascale) + len(mfma_Bscale))
    #assert 0
    def mfma_mn(reg_id, scale_reg_id, m, n):
        # lds_id : scale register is grouped by lds_id
        # src0: Matrix A scale {OP_SEL_HI [0], OP_SEL[0]} defines which part of scale is used by the Matrix A of MFMA instruction.
        # src1: Matrix B scale {OP_SEL_HI [1], OP_SEL[1]} defines which part of scale is used by the Matrix B of MFMA instruction.
        sel_scale_B = (n & 1) + (reg_id & 1)*2
        sel_scale_A = (m & 1) + (reg_id & 1)*2
        mod = f"op_sel:[{sel_scale_B & 1}, {sel_scale_A & 1},0] op_sel_hi:[{sel_scale_B//2}, {sel_scale_A//2}, 0] cbsz:4 blgp:4"

        # J.v_mfma_f32_16x16x32_bf16(mfma_C[m,n], mfma_B[reg_id, n], mfma_A[reg_id, m], mfma_C[m,n])
        J.v_mfma_scale_f32_16x16x128_f8f6f4(mfma_C[m,n], mfma_B[reg_id, n], mfma_A[reg_id, m], mfma_C[m,n],
                                            mfma_Bscale[scale_reg_id, n//2],
                                            mfma_Ascale[scale_reg_id, m//2],
                                            mod=mod)

    def mfma(reg_id, scale_reg_id):
        for m in range(nrM):
            for n in range(nrN):
                mfma_mn(reg_id, scale_reg_id, m, n)
                yield 16

    #for n in range(nrN): J.debug_log(mfma_B[0, n], torch.int32, "4h.16v.4h")
    #for m in range(nrM): J.debug_log(mfma_A[0, m], torch.int32, "4h.16v.4h")
    #J.s_endpgm()

    '''
    prelog: vmload LDS1

    - vmload LDS0 | ds_read LDS1
    - compute r
    - vmload LDS1 | ds_read LDS0
    - compute r
    '''
    with J.If(warp_row[0] == 1):
        J.s_barrier()

    J.emit(load_next_scales(0))
    J.emit(vm_load_a(ldsA[0]))
    J.emit(vm_load_b(ldsB[0]))
    mfma_C[...] = 0

    def loop_body(lds_id):
        def ds_readA(m, k):
            ds_read_a(ldsA[lds_id], mfma_A[k, m], m, k)
        def ds_readB(n, k):
            ds_read_b(ldsB[lds_id], mfma_B[k, n], n, k)
        def vMFMA(m,n,k):
            #return
            mfma_mn(k, lds_id, m, n)

        #J.s_setprio(1)
        # vmload LDS0 | ds_read LDS1
        #with J.view_code():
        J.emit(load_next_scales(lds_id^1))
        J.emit(vm_load_a(ldsA[lds_id^1]))
        J.emit(vm_load_b(ldsB[lds_id^1]))
        
        # wait vm-load to be finished, so DS read can start
        J.s_waitcnt(mod=f"vmcnt({vm_load_cnt_b + vm_load_cnt_a + num_scale_loads})")
        J.s_barrier()
        
        # interleave ds_read & MFMA
        
        # after first 4 MFMA emit 1 ds_read along with each MFMA
        assert nrM == 4
        assert nrN == 4
        ds_readA(0, 0); ds_readB(0, 0)
        ds_readA(1, 0); ds_readB(1, 0)
        J.s_waitcnt(mod=f"lgkmcnt({2})")
        vMFMA(0, 0, 0); ds_readA(2, 0); 
        J.s_waitcnt(mod=f"lgkmcnt({2})")
        vMFMA(1, 0, 0); ds_readB(2, 0)
        J.s_waitcnt(mod=f"lgkmcnt({2})")
        vMFMA(0, 1, 0); ds_readA(3, 0)
        vMFMA(1, 1, 0); ds_readB(3, 0)
        J.s_waitcnt(mod=f"lgkmcnt({2})")
        vMFMA(0, 2, 0); ds_readA(0, 1)
        vMFMA(1, 2, 0); 
        vMFMA(2, 0, 0); ds_readB(0, 1)
        vMFMA(2, 1, 0); 
        vMFMA(2, 2, 0); ds_readA(1, 1)
        J.s_waitcnt(mod=f"lgkmcnt({3})")
        vMFMA(0, 3, 0); ds_readB(1, 1)
        vMFMA(1, 3, 0); ds_readA(2, 1)
        vMFMA(2, 3, 0); ds_readA(3, 1)
        vMFMA(3, 0, 0); ds_readB(2, 1)
        vMFMA(3, 1, 0); ds_readB(3, 1)
        vMFMA(3, 2, 0); 
        vMFMA(3, 3, 0); 
        J.s_waitcnt(mod=f"lgkmcnt({4})")

        vMFMA(0, 0, 1)
        vMFMA(0, 1, 1)
        vMFMA(1, 0, 1)
        vMFMA(1, 1, 1)
        J.s_waitcnt(mod=f"lgkmcnt({0})")

        for m in range(4):
            for n in range(4):
                if m > 1 or n > 1:
                    vMFMA(m, n, 1)

        J.s_barrier()

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

    J.s_waitcnt(mod=f"vmcnt(0)")

    # for n in range(nrN): J.debug_log(mfma_C[0, n], torch.float, "4h.16v.4h")

    for lds in ldsA: J.free_lds(lds)
    for lds in ldsB: J.free_lds(lds)

    vrows = J.gpr(nrM, "vu32")
    vaddrs = J.gpr(nrM, "vu32")
    vweights = J.gpr(nrM, 2, "vf32")
    for m in range(nrM):
        row = (J.lane_id % 16) + warp_m*16 + m*16
        J.ds_read_b32(vrows[m], row * J.sizeof_DW + lds_sorted_ids)

    J.s_waitcnt(mod=f"lgkmcnt(0)")
    with J.If(warp_row[0] == 0):
        J.s_barrier()

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
    assert nrN >= 4
    assert nrN % 2 == 0

    for m in range(nrM):
        with J.ExecMask(vrows[m] < num_tokens[0]):
            out0 = J.gpr(nrN, "vf32")
            out1 = J.gpr(nrN, "vf32")
            vbf16 = J.gpr(nrN, "vbf16x2")

            if mfma_C.rtype == "a":
                vf32 = J.gpr(2, nrN, "vf32")
                for i in range(nrN):
                    J.v_accvgpr_read_b32(vf32[0,i], mfma_C[m, 0, i])
                    J.v_accvgpr_read_b32(vf32[1,i], mfma_C[m, 2, i])
                for i in range(nrN):
                    out0[i] = vf32[1, i] * J.silu(vf32[0, i])

                for i in range(nrN):
                    J.v_accvgpr_read_b32(vf32[0,i], mfma_C[m, 1, i])
                    J.v_accvgpr_read_b32(vf32[1,i], mfma_C[m, 3, i])

                for i in range(nrN):
                    out1[i] = vf32[1, i] * J.silu(vf32[0, i])
            else:
                for i in range(nrN):
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
            J.global_store_dwordx4(vaddrs[m], vbf16[0:3], output, mod=f"offset:{0}")


# no_pass=["pass_cse", "pass_dce", "pass_dse"]
# use 256x256 blocksize
@jit(with_debug_log=False)
def moe_gemm_mxfp4_gateup_4wave(J, wg_M, wg_N,
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
    assert gate_up == True
    assert OC % wg_N == 0
    num_oc_blocks = J.div(OC, wg_N)
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
    #with J.view_code():
    e_idx, n_idx = J.sgpr_div(J.blockIdx.x, num_oc_blocks)

    # 0+ 8*0, 8*1, .... 8*7
    # 1+ 8*0, 8*1, .... 8*7
    # ....
    

    #print(num_oc_blocks)
    #assert 0

    #n_idx = J.blockIdx.x % num_oc_blocks # split along OC
    #e_idx = J.blockIdx.y // num_oc_blocks
    
    # blockIdx is the physical CU location, map it to expert_id

    # e_idx[0] *= 0
    s_e_id = J.gpr(1, 'su32')
    J.s_load_dword(s_e_id, sorted_expert_ids, e_idx[0] * 4)
    max_id = J.gpr(1, 'su32')
    J.s_load_dword(max_id, num_valid_ids, 0)
    J.s_waitcnt(mod=f"lgkmcnt(0)")

    J.debug_setup((e_idx[0] == 0) & (n_idx[0] == 0) & (J.warp_id[0] == 0))
    with J.If(e_idx[0] * wg_M >= max_id[0]):
        J.s_endpgm()
    
    output[:] += n_idx * (wg_N//2 * J.sizeof_bf16)
    sorted_ids[:] += e_idx * (wg_M * J.sizeof_DW)
    sorted_weights[:] += e_idx * (wg_M * J.sizeof_DW)

    i_scale[:] += e_idx * (J.div(wg_M,32) * stride_scale32x256)
    w_scale[:] += s_e_id * (J.div(OC,32) * stride_scale32x256)
    i_scale[:] += (J.warp_id[0] // 2) * (J.div(wg_M//2, 32) * stride_scale32x256)
    
    sbuff_a = J.Buffer(i_scale, J.div(wg_M//2, 32) * stride_scale32x256)
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

    buff_a = J.Buffer(input, num_tokens * stride_k)

    ldsA = [J.alloc_lds(nbM * nbK * 1024), J.alloc_lds(nbM * nbK * 1024)]
    ldsB = [J.alloc_lds(nbN * nbK * 1024), J.alloc_lds(nbN * nbK * 1024)]

    warp_m = J.gpr((J.warp_id[0] // 2)*nrM)
    warp_n = J.gpr((J.warp_id[0] % 2)*nrN)

    # load sorted_ids into LDS
    lds_sorted_ids = J.alloc_lds(wg_M * J.sizeof_DW)
    #lds_sorted_weights = J.alloc_lds(wg_M * J.sizeof_DW)
    J.wg_load_lds(lds_sorted_ids, sorted_ids, wg_M * J.sizeof_DW, wait_barrier = True)
    #J.wg_load_lds(lds_sorted_weights, sorted_weights, wg_M * J.sizeof_DW, wait_barrier = True)

    vm_load_a, vm_load_cnt_a, ds_read_a = get_loader_sorted_tok(J, buff_a, lds_sorted_ids, nbM, nbK, stride_k, warp_m, gate_up, TOPK, num_tokens)
    vm_load_b, vm_load_cnt_b, ds_read_b = get_loader_b_preshuffled(J, weight, wg_N, nbN, nbK, stride_k, warp_n, gate_up, stride_gate_up)

    mfma_A = J.gpr(2, nrM, 4, "vbf16x2")
    mfma_B = J.gpr(2, nrN, 4, "vbf16x2")
    mfma_C = J.gpr(nrM, nrN, 4, "af32")
    
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
        vaddr = J.gpr("vu32", vaddr_scale[0])
        assert nrN >= 4, f"{nrN=}"
        nrScaleBd2 = J.div(nrN, 2, 2)
        for ii in range(nrScaleBd2):
            sbuff_b[0].load_dword(mfma_Bscale[index & 1, ii + 0], vaddr, 0) # gate
            sbuff_b[1].load_dword(mfma_Bscale[index & 1, ii + nrScaleBd2], vaddr, 0) # up
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

    #J.debug_log(mfma_B[0, 0], torch.int32, "4h.16v.4h")
    #J.debug_log(mfma_B[0, nrN//2], torch.int32, "4h.16v.4h")
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
        J.emit(vm_load)
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

    #for n in range(nrN): J.debug_log(mfma_C[0, n], torch.float, "4h.16v.4h")
    J.debug_log(mfma_C[0, 0], torch.float, "4h.16v.4h")
    J.debug_log(mfma_C[0, nrN//2], torch.float, "4h.16v.4h")

    J.s_waitcnt(mod=f"vmcnt(0)")

    for lds in ldsA: J.free_lds(lds)
    for lds in ldsB: J.free_lds(lds)

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
        vaddrs[m] = vrows[m] * (TOPK * stride_c) +  topk*stride_c + swap_12_col * J.sizeof_DW4 + \
                    warp_n * ((16//2) * J.sizeof_bf16)

    # gate 0,1 up 2,3
    # 
    assert nrN >= 4
    assert nrN % 2 == 0

    for m in range(nrM):
        with J.ExecMask(vrows[m] < num_tokens[0]):
            for n in range(0, J.div(nrN, 2), 2):
                i_gate = n
                i_up = i_gate + J.div(nrN, 2)

                out0 = J.gpr(4, "vf32")
                out1 = J.gpr(4, "vf32")
                vbf16 = J.gpr(4, "vbf16x2")

                for i in range(4):
                    out0[i] = mfma_C[m, i_up, i] * J.silu(mfma_C[m, i_gate, i])
                    out1[i] = mfma_C[m, i_up+1, i] * J.silu(mfma_C[m, i_gate+1, i])

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
                J.global_store_dwordx4(vaddrs[m], vbf16[0:3], output, mod=f"offset:{n*16*4//2}")



# no_pass=["pass_cse", "pass_dce", "pass_dse"]
@jit(with_debug_log=False)
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
    # e_idx[0] *= 0
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
            assert nrN >= 4, f"{nrN=}"
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
        assert nrN >= 4
        assert nrN % 2 == 0

        for m in range(nrM):
            with J.ExecMask(vrows[m] < num_tokens[0]):
                out0 = J.gpr(nrN, "vf32")
                out1 = J.gpr(nrN, "vf32")
                vbf16 = J.gpr(nrN, "vbf16x2")

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
                J.global_store_dwordx4(vaddrs[m], vbf16[0:3], output, mod=f"offset:{0}")
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


