import os
import random
from typing import Optional
os.environ['PYHIP_JIT_LOG'] = '0'
from pyhip import cudaPerf, jit, JIT
import torch

from functools import cache
#####################################################################
# kernel define
@cache
def get_lane_id(J):
    vgpr_lane_id = J.gpr(J.threadIdx.x[0] & 63)
    return vgpr_lane_id

@cache
def get_lane_id_div(J, divisor):
    assert isinstance(divisor, int)
    return J.gpr(get_lane_id(J) // divisor)

@cache
def get_lane_id_mod(J, divisor):
    assert isinstance(divisor, int)
    return J.gpr(get_lane_id(J) % divisor)

def div_up(x, y):
    return (x + y - 1) // y

"""
def shuffle_weight(x: torch.Tensor, layout=(16, 16), use_int4=False) -> torch.Tensor:
    # Hardcode BLOCK_K and BLOCK_N
    IN, IK = layout
    BK = IK * 2
    K = 16 // x.element_size() if not use_int4 else 32
    BN = IN
    assert x.shape[-2] % BN == 0, f"{x.shape[-2]} % {BN} == {x.shape[-2] % BN }"
    assert x.shape[-1] % BK == 0, f"{x.shape[-1]} % {BK} == {x.shape[-1] % BK }"

    x_ = x
    x_ = x_.view(-1, x.shape[-2] // BN, BN, x.shape[-1] // BK, BK // K, K)
    x_ = x_.permute(0, 1, 3, 4, 2, 5)
    x_ = x_.contiguous()
    x_ = x_.view(*x.shape)
    return x_

    [-1, N//16, 16n, K//32, 4k, 8k]
    [-1, N//16, K//32, 4k, 16n, 8k]
"""

@jit()
def moe_gemm_batch1(J:JIT,
                    weight_dtype,
                    K,            # compile-time args
                    N,            # compile-time args
                    with_silu,    # compile-time args
                    p_input:"void*",
                    p_weight:"void*",
                    p_output:"void*",
                    p_topk_ids:"void*",
                    p_topk_weight:"float*", 
                    p_w_scale:"float*",
                    M:"int"):

    BLOCK_SIZE_M = 16  # nBM * 16
    BLOCK_SIZE_N = 32  # nBN * 32
    BLOCK_SIZE_K = 32
    assert K % BLOCK_SIZE_K == 0
    assert BLOCK_SIZE_M % 16 == 0
    assert BLOCK_SIZE_N % 32 == 0

    nBM = BLOCK_SIZE_M // 16
    nBN = BLOCK_SIZE_N // 32
    use_split_K = with_silu #K > 4*32
    num_split_K = 4 if use_split_K else 1

    sizeof_bf16 = 2
    sizeof_f32 = 4
    sizeof_w = sizeof_bf16 if weight_dtype == torch.bfloat16 else 1
    stride_A = K*sizeof_bf16
    stride_B = K*sizeof_w

    Creg = J.gpr(2, 4, "vf32")
    # there is 16 elements per weight read if fp8, so A should be double read
    A_rep = 1 if weight_dtype == torch.bfloat16 else 2
    Areg = J.gpr(2, A_rep, 2, 2, "vbf16x2") # 8-bf16 == DWORDx4
    Breg = J.gpr(2, 2, 2, 2, "vbf16x2") # 8-bf16 == DWORDx4

    Creg[:] = 0

    # expert index(not id)
    e_idx = J.blockIdx.y
    s_e_id = J.gpr(1, 'su32')
    J.s_load_dword(s_e_id, p_topk_ids, e_idx[0] * 4)

    # hide following initialization into s_waitcnt
    pp_reg_id = 0
    s_weight = J.gpr(1, 'su32')

    if True:
        # one WG per CU,  4 waves split on K, 
        lane_mod_16 = get_lane_id_mod(J, 16)
        lane_div_16 = get_lane_id_div(J, 16)
        warp_id = J.gpr("su32")
        J.v_readfirstlane_b32(warp_id, J.threadIdx.x[0] // 64)
        assert K % (num_split_K*32) == 0
        p_weight[:] = p_weight[:] + (16 if with_silu else 32)*J.blockIdx.x*stride_B
        p_output[:] = p_output[:] + (16 if with_silu else 32)*J.blockIdx.x*sizeof_bf16
        J.debug_setup((J.blockIdx.x[0] == 0) & (e_idx[0] == 0) & (warp_id[0] == 0))

        p_weight1 = J.gpr(2,"su32")
        if with_silu:
            J.s_add_u32(p_weight1[0], p_weight[0], (N//2)*K*sizeof_w)
            J.s_addc_u32(p_weight1[1], p_weight[1], 0)
        else:
            J.s_add_u32(p_weight1[0], p_weight[0], 16*K*sizeof_w)
            J.s_addc_u32(p_weight1[1], p_weight[1], 0)
        # 16 elements for a if fp8
        a_element_num_per_thread = 8 if weight_dtype == torch.bfloat16 else 16
        voffset_a = J.gpr((J.threadIdx.x % 16)*stride_A + (J.threadIdx.x//16) * (a_element_num_per_thread*sizeof_bf16))
        voffset_b = J.gpr(J.threadIdx.x * 16)

        soffset_kb = J.gpr("su32")
        soffset_ka = J.gpr("su32")
        soffset_kb[0] = 0
        soffset_ka[0] = 0

        if with_silu:
            # [E, M, N//2]
            p_output[:] = p_output[:] + e_idx * (N // 2 * sizeof_bf16)
        else:
            # [M, N]
            p_input[:] = p_input[:] + e_idx * (K * sizeof_bf16)
            J.s_load_dword(s_weight, p_topk_weight, e_idx[0] * 4)

        buff_a = J.Buffer(p_input, M*K*sizeof_bf16)

        J.s_waitcnt(mod=f"lgkmcnt(0)")
        if weight_dtype != torch.bfloat16:
            v_w_scale = J.gpr(2, 2, 'vf32')
            # scale layout: [E, N, 1]
            offset0 = J.gpr(1, 'vu32')
            offset0[0] = J.gpr(s_e_id * (N * sizeof_f32) + (16 if with_silu else 32) * J.blockIdx.x * sizeof_f32) + lane_mod_16 * sizeof_f32
            offset1 = offset0 + (N // 2 * sizeof_f32 if with_silu else 16 * sizeof_f32)
            J.global_load_dword(v_w_scale[0, 0], offset0, p_w_scale)
            J.global_load_dword(v_w_scale[1, 0], offset1, p_w_scale)

        buff_a.load_dwordx4(Areg[pp_reg_id, 0], voffset_a, soffset_ka)
        if weight_dtype != torch.bfloat16:
            buff_a.load_dwordx4(Areg[pp_reg_id, 1], voffset_a, soffset_ka + 16)

    p_weight[:] = p_weight[:] + s_e_id * (N * K * sizeof_w)
    p_weight1[:] = p_weight1[:] + s_e_id * (N * K * sizeof_w)

    buff_b0 = J.Buffer(p_weight, 16*K*sizeof_w)
    buff_b1 = J.Buffer(p_weight1, 16*K*sizeof_w)

    # ping pong register buffer id
    buff_b0.load_dwordx4(Breg[pp_reg_id, 0], voffset_b, soffset_kb)
    buff_b1.load_dwordx4(Breg[pp_reg_id, 1], voffset_b, soffset_kb)
    soffset_kb[0] = soffset_kb[0] + 16*64*num_split_K
    soffset_ka[0] = soffset_ka[0] + a_element_num_per_thread*4*num_split_K*sizeof_bf16
    pp_reg_id = pp_reg_id ^ 1

    k_step_wg = num_split_K*32 if weight_dtype == torch.bfloat16 else num_split_K*64
    if weight_dtype == torch.bfloat16:
        assert K % (num_split_K*32) == 0, f'K must be multiple of {num_split_K*32}'
    else:
        # a wave needs at least 64 elements
        assert K % 64 == 0, 'K must be multiple of 64'

    # (A0.B0.C0.D0.A1.B1.C1.D1)[3, 2, 7, 6] = (A1.B1.A0.B0)
    pattern_cvt_bf16 = J.gpr("su32")
    pattern_cvt_bf16[0] = 0x03_02_07_06
    k_max = div_up(K, k_step_wg)
    for k in range(k_max):
        # [16,32] * [2,16,32] => [2,16,16]
        if k + 1 < k_max:
            buff_a.load_dwordx4(Areg[pp_reg_id, 0], voffset_a, soffset_ka)
            if weight_dtype != torch.bfloat16:
                buff_a.load_dwordx4(Areg[pp_reg_id, 1], voffset_a, soffset_ka + 16)
            buff_b0.load_dwordx4(Breg[pp_reg_id, 0], voffset_b, soffset_kb)
            buff_b1.load_dwordx4(Breg[pp_reg_id, 1], voffset_b, soffset_kb)
            pp_reg_id = pp_reg_id ^ 1
            soffset_kb[0] = soffset_kb[0] + 16*64*num_split_K
            soffset_ka[0] = soffset_ka[0] + a_element_num_per_thread*4*num_split_K*sizeof_bf16
            if weight_dtype == torch.bfloat16:
                J.s_waitcnt(mod=f"vmcnt(3)")
            else:
                J.s_waitcnt(mod=f"vmcnt(4)")
        else:
            pp_reg_id = pp_reg_id ^ 1
            J.s_waitcnt(mod=f"vmcnt(0)")

        if weight_dtype != torch.bfloat16:
            if k == 0:
                J.v_mov_b32(v_w_scale[0, 1], v_w_scale[0, 0])
                J.v_mov_b32(v_w_scale[1, 1], v_w_scale[1, 0])

            # decompress
            v_w_f32 = J.gpr(2, 2, 'vf32', align=4)
            v_w_bf16 = J.gpr(2, 2, 'vf32', align=4)
            # Breg[2, 2, 2, 2]: pin, block_id, #no of 8 items, 2 dwords/4bf16
            for i in range(2):
                for j in range(2):
                    for block_id in range(2):
                        J.v_cvt_pk_f32_fp8(v_w_f32[0], Breg[pp_reg_id, block_id, i, j])
                        J.v_cvt_pk_f32_fp8_sdwa(v_w_f32[1], Breg[pp_reg_id, block_id, i, j], mod='src0_sel:WORD_1')
                        J.v_pk_mul_f32(v_w_f32[0], v_w_f32[0], v_w_scale[block_id])
                        J.v_pk_mul_f32(v_w_f32[1], v_w_f32[1], v_w_scale[block_id])
                        J.v_perm_b32(v_w_bf16[block_id, 0], v_w_f32[0, 0], v_w_f32[0, 1], pattern_cvt_bf16)
                        J.v_perm_b32(v_w_bf16[block_id, 1], v_w_f32[1, 0], v_w_f32[1, 1], pattern_cvt_bf16)
                    # 2, A_rep, 2, 2
                    for block_id in range(2):
                        J.v_mfma_f32_16x16x16_bf16(Creg[block_id], v_w_bf16[block_id], Areg[pp_reg_id, i, j], Creg[block_id])
        else:
            J.v_mfma_f32_16x16x16_bf16(Creg[0], Breg[pp_reg_id,0,0], Areg[pp_reg_id,0,0], Creg[0])
            J.v_mfma_f32_16x16x16_bf16(Creg[1], Breg[pp_reg_id,1,0], Areg[pp_reg_id,0,0], Creg[1])

            J.v_mfma_f32_16x16x16_bf16(Creg[0], Breg[pp_reg_id,0,1], Areg[pp_reg_id,0,1], Creg[0])
            J.v_mfma_f32_16x16x16_bf16(Creg[1], Breg[pp_reg_id,1,1], Areg[pp_reg_id,0,1], Creg[1])

    if use_split_K:
        assert with_silu
        # split K
        lds_buff = J.LDSTensor([4*16, 32], torch.float)

        vaddr = J.gpr("vu32")
        lds_buff.write("b128", Creg[0], lane_mod_16 + warp_id*16, lane_div_16*4)
        lds_buff.write("b128", Creg[1], lane_mod_16 + warp_id*16, lane_div_16*4 + 16)

        J.s_waitcnt(mod=f"lgkmcnt(0)")
        J.s_barrier()
        # each wave reduce 4 rows
        vrow = J.gpr(lane_div_16 + warp_id*4)
        gate_up = J.gpr(4, 2, "vf32")
        # interleave 
        lds_buff.read("b32", gate_up[0], vrow + 0*16, (lane_mod_16), offset1 = 16)
        lds_buff.read("b32", gate_up[1], vrow + 1*16, (lane_mod_16), offset1 = 16)
        lds_buff.read("b32", gate_up[2], vrow + 2*16, (lane_mod_16), offset1 = 16)
        lds_buff.read("b32", gate_up[3], vrow + 3*16, (lane_mod_16), offset1 = 16)

        J.s_waitcnt(mod=f"lgkmcnt(2)")
        J.v_pk_add_f32(gate_up[0], gate_up[0], gate_up[1])
        J.s_waitcnt(mod=f"lgkmcnt(0)")
        J.v_pk_add_f32(gate_up[2], gate_up[2], gate_up[3])
        J.v_pk_add_f32(gate_up[0], gate_up[0], gate_up[2])

        if with_silu:
            out = J.gpr(gate_up[0, 1] * J.silu(gate_up[0, 0]))
            vaddr = J.gpr(vrow * (N//2*sizeof_bf16) + lane_mod_16*(sizeof_bf16))
            with J.ExecMask(vrow < M[0]):
                J.global_store_short_d16_hi(vaddr, out[0], p_output)
        else:
            # convert to bf16
            vaddr = J.gpr(vrow * (N*sizeof_bf16) + lane_mod_16*(2*sizeof_bf16))
            with J.ExecMask(vrow < M[0]):
                out = J.gpr("vf32")
                out[0] = (gate_up[0,0]>>16)|(gate_up[0,1]&0xFFFF0000)
                J.global_store_dword(vaddr, out[0], p_output)
    else:
        # split N
        assert not with_silu
        vaddr = J.gpr(lane_mod_16 * (N*sizeof_bf16) + lane_div_16*(4*sizeof_bf16))
        creg_low = J.gpr(2, 2, "vbf16x2", align=4)
        J.s_waitcnt(mod=f"lgkmcnt(0)")
        for i in range(2):
            for j in range(4):
                Creg[i, j] = Creg[i, j] * s_weight
        for i in range(2):
            creg_low[i,0] = (Creg[i,0]>>16)|(Creg[i,1]&0xFFFF0000)
            creg_low[i,1] = (Creg[i,2]>>16)|(Creg[i,3]&0xFFFF0000)

        with J.ExecMask(lane_mod_16 < M[0]):
            J.global_atomic_pk_add_bf16(vaddr, creg_low[0,0], p_output)
            J.global_atomic_pk_add_bf16(vaddr+4, creg_low[0,1], p_output)
            J.global_atomic_pk_add_bf16(vaddr+16*sizeof_bf16, creg_low[1,0], p_output)
            J.global_atomic_pk_add_bf16(vaddr+16*sizeof_bf16+4, creg_low[1,1], p_output)

@jit()
def moe_gemm_batch(J:JIT,
                   TOPK,
                   K,            # compile-time args
                   N,            # compile-time args
                   with_silu,    # compile-time args
                   p_input:"void*",
                   p_weight:"void*",
                   p_output:"void*",
                   # sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids
                   p_sorted_ids:"void*",
                   p_sorted_weights:"float*",
                   p_sorted_expert_ids:"void*",
                   p_num_valid_ids:"void*",
                   M:"int"):
    BLOCK_SIZE_M = 16  # nBM * 16
    BLOCK_SIZE_N = 32  # nBN * 32
    BLOCK_SIZE_K = 32
    assert K % BLOCK_SIZE_K == 0
    assert BLOCK_SIZE_M % 16 == 0
    assert BLOCK_SIZE_N % 32 == 0

    nBM = BLOCK_SIZE_M // 16
    nBN = BLOCK_SIZE_N // 32
    use_split_K = with_silu #K > 4*32
    num_split_K = 4 if use_split_K else 1

    sizeof_bf16 = 2
    sizeof_f32 = 4
    stride_AB = K*sizeof_bf16

    Creg = J.gpr(2, 4, "vf32")
    Areg = J.gpr(2, 2, 2, "vbf16x2") # 8-bf16 == DWORDx4
    Breg = J.gpr(2, 2, 2, 2, "vbf16x2") # 8-bf16 == DWORDx4

    Creg[:] = 0

    # grid: N // 32, sorted_expert_ids.shape[0]
    # expert index in p_sorted_expert_ids
    e_idx = J.blockIdx.y
    s_e_id = J.gpr(1, 'su32')
    J.s_load_dword(s_e_id, p_sorted_expert_ids, e_idx[0] * 4)
    max_id = J.gpr(1, 'su32')
    J.s_load_dword(max_id, p_num_valid_ids, 0)
    J.s_waitcnt(mod=f"lgkmcnt(0)")
    # invalid padding section
    J.Jump("continue_following", e_idx * BLOCK_SIZE_M < max_id)
    J.s_endpgm()
    J.Label("continue_following")

    # hide following initialization into s_waitcnt
    pp_reg_id = 0
    s_weight = J.gpr(1, 'su32')

    if True:
        p_sorted_ids[:] += e_idx * (BLOCK_SIZE_M * 4)
        # one WG per CU,  4 waves split on K, 
        lane_mod_16 = get_lane_id_mod(J, 16)
        lane_div_16 = get_lane_id_div(J, 16)
        warp_id = J.gpr("su32")
        J.v_readfirstlane_b32(warp_id, J.threadIdx.x[0] // 64)
        assert K % (num_split_K*32) == 0
        J.debug_setup((J.blockIdx.x[0] == 3) & (J.blockIdx.y[0] == 0) & (warp_id[0] == 3))
        # J.debug_log(p_sorted_expert_ids, torch_dtype=torch.uint32)
        # J.debug_log(p_sorted_weights, torch_dtype=torch.uint32)

        v_sorted_id = J.gpr('vu32')
        # assume only BLOCK_SIZE_M==16
        assert BLOCK_SIZE_M == 16
        J.global_load_dword(v_sorted_id, lane_mod_16 << 2, p_sorted_ids)
        if not with_silu:
            p_sorted_weights[:] += e_idx * (BLOCK_SIZE_M * 4)
            v_weight = J.gpr('vf32')
            J.global_load_dword(v_weight, lane_mod_16 << 2, p_sorted_weights)

        p_weight[:] = p_weight[:] + (16 if with_silu else 32)*J.blockIdx.x*stride_AB
        p_output[:] = p_output[:] + (16 if with_silu else 32)*J.blockIdx.x*sizeof_bf16

        p_weight1 = J.gpr(2,"su32")
        if with_silu:
            J.s_add_u32(p_weight1[0], p_weight[0], (N//2)*K*sizeof_bf16)
            J.s_addc_u32(p_weight1[1], p_weight[1], 0)
        else:
            J.s_add_u32(p_weight1[0], p_weight[0], 16*K*sizeof_bf16)
            J.s_addc_u32(p_weight1[1], p_weight[1], 0)
        # wait for v_sorted_id
        J.s_waitcnt(mod=f"vmcnt(0)")
        v_token_id = v_sorted_id & 0xffffff
        if with_silu:
            voffset_a = J.gpr(v_token_id*stride_AB + (J.threadIdx.x//16) * 16)
        else:
            v_topk_id = v_sorted_id >> 24
            # input layout: [B, TOPK, INTER_MEDIA]
            voffset_a = J.gpr(v_token_id*(TOPK*K*sizeof_bf16) + v_topk_id*(K*sizeof_bf16)+ (J.threadIdx.x//16) * 16)

        voffset_b = J.gpr(J.threadIdx.x * 16)

        soffset_kb = J.gpr("su32")
        soffset_ka = J.gpr("su32")
        soffset_kb[0] = 0
        soffset_ka[0] = 0

        buff_a = J.Buffer(p_input, M*(K*sizeof_bf16) if with_silu else M*(TOPK*K*sizeof_bf16))
        buff_a.load_dwordx4(Areg[pp_reg_id], voffset_a, soffset_ka)

    # wait for s_e_id
    J.s_waitcnt(mod=f"lgkmcnt(0)")
    p_weight[:] = p_weight[:] + s_e_id * (N * K * sizeof_bf16)
    p_weight1[:] = p_weight1[:] + s_e_id * (N * K * sizeof_bf16)

    buff_b0 = J.Buffer(p_weight, 16*K*sizeof_bf16)
    buff_b1 = J.Buffer(p_weight1, 16*K*sizeof_bf16)

    # ping pong register buffer id
    buff_b0.load_dwordx4(Breg[pp_reg_id, 0], voffset_b, soffset_kb)
    buff_b1.load_dwordx4(Breg[pp_reg_id, 1], voffset_b, soffset_kb)
    soffset_kb[0] = soffset_kb[0] + 16*(32*num_split_K)*sizeof_bf16
    soffset_ka[0] = soffset_ka[0] + 32*num_split_K*sizeof_bf16
    pp_reg_id = pp_reg_id ^ 1

    k_max = div_up(K, num_split_K*32)
    for k in range(k_max):
        # [16,32] * [2,16,32] => [2,16,16]
        if k + 1 < k_max:
            buff_a.load_dwordx4(Areg[pp_reg_id], voffset_a, soffset_ka)
            buff_b0.load_dwordx4(Breg[pp_reg_id, 0], voffset_b, soffset_kb)
            buff_b1.load_dwordx4(Breg[pp_reg_id, 1], voffset_b, soffset_kb)
            pp_reg_id = pp_reg_id ^ 1
            soffset_kb[0] = soffset_kb[0] + 16*(32*num_split_K)*sizeof_bf16
            soffset_ka[0] = soffset_ka[0] + 32*num_split_K*sizeof_bf16
            J.s_waitcnt(mod=f"vmcnt(3)")
        else:
            pp_reg_id = pp_reg_id ^ 1
            J.s_waitcnt(mod=f"vmcnt(0)")

        J.v_mfma_f32_16x16x16_bf16(Creg[0], Breg[pp_reg_id,0,0], Areg[pp_reg_id,0], Creg[0])
        J.v_mfma_f32_16x16x16_bf16(Creg[1], Breg[pp_reg_id,1,0], Areg[pp_reg_id,0], Creg[1])

        J.v_mfma_f32_16x16x16_bf16(Creg[0], Breg[pp_reg_id,0,1], Areg[pp_reg_id,1], Creg[0])
        J.v_mfma_f32_16x16x16_bf16(Creg[1], Breg[pp_reg_id,1,1], Areg[pp_reg_id,1], Creg[1])

    if use_split_K:
        assert with_silu
        # split K
        lds_buff = J.LDSTensor([4*16, 32], torch.float)

        vaddr = J.gpr("vu32")
        lds_buff.write("b128", Creg[0], lane_mod_16 + warp_id*16, lane_div_16*4)
        lds_buff.write("b128", Creg[1], lane_mod_16 + warp_id*16, lane_div_16*4 + 16)

        J.s_waitcnt(mod=f"lgkmcnt(0)")
        J.s_barrier()
        v_sorted_id_permute = J.gpr("vu32")

        # each wave reduce 4 rows
        vrow = J.gpr(lane_div_16 + warp_id*4)
        J.ds_bpermute_b32(v_sorted_id_permute, vrow * 4, v_sorted_id)
        gate_up = J.gpr(4, 2, "vf32")
        # interleave 
        lds_buff.read("b32", gate_up[0], vrow + 0*16, (lane_mod_16), offset1 = 16)
        lds_buff.read("b32", gate_up[1], vrow + 1*16, (lane_mod_16), offset1 = 16)
        lds_buff.read("b32", gate_up[2], vrow + 2*16, (lane_mod_16), offset1 = 16)
        lds_buff.read("b32", gate_up[3], vrow + 3*16, (lane_mod_16), offset1 = 16)

        J.s_waitcnt(mod=f"lgkmcnt(2)")
        J.v_pk_add_f32(gate_up[0], gate_up[0], gate_up[1])
        J.s_waitcnt(mod=f"lgkmcnt(0)")
        J.v_pk_add_f32(gate_up[2], gate_up[2], gate_up[3])
        J.v_pk_add_f32(gate_up[0], gate_up[0], gate_up[2])

        if with_silu:
            out = J.gpr(gate_up[0, 1] * J.silu(gate_up[0, 0]))
            # output: [B, TOPK, N]
            v_token_id = J.gpr(v_sorted_id_permute[0] & 0xffffff)
            v_topk_id = J.gpr(v_sorted_id_permute[0] >> 24)
            vaddr = J.gpr(v_token_id * (N//2*sizeof_bf16*TOPK) + v_topk_id * (N//2*sizeof_bf16) + lane_mod_16*(sizeof_bf16))
            with J.ExecMask(v_token_id < M[0]):
                J.global_store_short_d16_hi(vaddr, out[0], p_output)
        else:
            # convert to bf16
            vaddr = J.gpr(vrow * (N*sizeof_bf16) + lane_mod_16*(2*sizeof_bf16))
            with J.ExecMask(vrow < M[0]):
                out = J.gpr("vf32")
                out[0] = (gate_up[0,0]>>16)|(gate_up[0,1]&0xFFFF0000)
                J.global_store_dword(vaddr, out[0], p_output)
    else:
        # split N
        assert not with_silu
        # output layout: [B, N]
        vaddr = J.gpr(v_token_id * (N*sizeof_bf16) + lane_div_16*(4*sizeof_bf16))
        creg_low = J.gpr(2, 2, "vbf16x2", align=4)
        J.s_waitcnt(mod=f"lgkmcnt(0)")
        for i in range(2):
            for j in range(4):
                Creg[i, j] = Creg[i, j] * v_weight
        for i in range(2):
            creg_low[i,0] = (Creg[i,0]>>16)|(Creg[i,1]&0xFFFF0000)
            creg_low[i,1] = (Creg[i,2]>>16)|(Creg[i,3]&0xFFFF0000)
        with J.ExecMask(v_token_id < M[0]):
            J.global_atomic_pk_add_bf16(vaddr, creg_low[0,0], p_output)
            J.global_atomic_pk_add_bf16(vaddr+4, creg_low[0,1], p_output)
            J.global_atomic_pk_add_bf16(vaddr+16*sizeof_bf16, creg_low[1,0], p_output)
            J.global_atomic_pk_add_bf16(vaddr+16*sizeof_bf16+4, creg_low[1,1], p_output)

#####################################################################
def run_aiter(hidden_states,
            w1,  # [expert(local_expert:EP), inter_dim*2, dim] N,K
            w2,  # [expert(local_expert:EP), dim, inter_dim]
            topk_weight,
            topk_ids,
            w1_scale: Optional[torch.tensor] = None,  # [expert(local_expert:EP), inter_dim, 1]
            w2_scale: Optional[torch.tensor] = None,  # [expert(local_expert:EP), model_dim, 1]
            ):
    from aiter.fused_moe import fused_moe
    from aiter import QuantType
    return fused_moe(
        hidden_states,
        w1,
        w2,
        topk_weight,
        topk_ids,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        quant_type=QuantType.No if w1.dtype == torch.bfloat16 else QuantType.per_Token
    )

def run_batch(B=1, weight_type=torch.bfloat16):
    HIDDEN_SIZE = 2048
    TP = 4
    INTER_SIZE = 768
    E = 128
    TOPK = 8
    INTER_SIZE_TP = INTER_SIZE // TP
    BUF_COPY = 32
    hidden_states = (torch.randn([BUF_COPY, B, HIDDEN_SIZE], dtype=torch.bfloat16) + 1)*0.001
    if weight_type == torch.bfloat16:
        w_ = torch.randn([E, INTER_SIZE_TP * 2, HIDDEN_SIZE], dtype=weight_type)
        w1 = [w_.clone() for _ in range(BUF_COPY)]
        w_ = torch.randn([E, HIDDEN_SIZE, INTER_SIZE_TP], dtype=weight_type)
        w2 = [w_.clone() for _ in range(BUF_COPY)]
        w1_scale = [None] * BUF_COPY
        w2_scale = [None] * BUF_COPY
    else:
        import aiter
        torch_quant = aiter.get_torch_quant(aiter.QuantType.per_Token)
        w_ = torch.randn([E, INTER_SIZE_TP * 2, HIDDEN_SIZE], dtype=torch.bfloat16)
        w1_qt, w1_qt_scale = torch_quant(w_, quant_dtype=weight_type)
        w1 = [w1_qt.clone() for _ in range(BUF_COPY)]
        w1_scale = [w1_qt_scale.clone() for _ in range(BUF_COPY)]
        w_ = torch.randn([E, HIDDEN_SIZE, INTER_SIZE_TP], dtype=torch.bfloat16)
        w2_qt, w2_qt_scale = torch_quant(w_, quant_dtype=weight_type)
        w2 = [w2_qt.clone() for _ in range(BUF_COPY)]
        w2_scale = [w2_qt_scale.clone() for _ in range(BUF_COPY)]

    topk_weight = torch.randn([BUF_COPY, B, TOPK], dtype=torch.float32)
    topk_ids = torch.ones([BUF_COPY, B, TOPK], dtype=torch.int32)
    topk_ids_base = torch.randperm(E, dtype=torch.int32)

    if 1:
        import aiter

        topk_ids[0,:,] = topk_ids_base[: TOPK]
        from aiter.ops.shuffle import shuffle_weight
        # aiter needs preshuffle weights
        w1_qt_aiter = shuffle_weight(w1[0], layout=(16, 16))
        w2_qt_aiter = shuffle_weight(w2[0], layout=(16, 16))
        ref_out = run_aiter(hidden_states=hidden_states[0], w1=w1_qt_aiter, w2=w2_qt_aiter, topk_weight=topk_weight[0], topk_ids=topk_ids[0], w1_scale=w1_scale[0], w2_scale=w2_scale[0])
        i = 0
        flops = 2 * B * (HIDDEN_SIZE * INTER_SIZE_TP * 2 + B * TOPK * HIDDEN_SIZE * INTER_SIZE_TP)
        mem_size = (HIDDEN_SIZE * INTER_SIZE_TP * 2 + HIDDEN_SIZE * INTER_SIZE_TP) * TOPK * 2
        for _ in range(10):
            idx_start = random.randint(0, E - TOPK)
            topk_ids[i,:,] = topk_ids_base[idx_start : idx_start + TOPK]
            with cudaPerf(flops, mem_size, name=f"aiter[{B=}]"):
                run_aiter(hidden_states=hidden_states[i], w1=w1[i], w2=w2[i], topk_weight=topk_weight[i], topk_ids=topk_ids[i], w1_scale=w1_scale[i], w2_scale=w2_scale[i])
            i = (i + 1) % BUF_COPY

    if 1:
        import aiter
        from aiter import ActivationType, QuantType, dtypes
        from aiter.fused_moe import moe_sorting

        org_fused_moe = None
        def my_fused_moe(
            hidden_states,
            w1,  # [expert(local_expert:EP), inter_dim*2, dim] N,K
            w2,  # [expert(local_expert:EP), dim, inter_dim]
            topk_weight,
            topk_ids,
            expert_mask: Optional[torch.tensor] = None,  # EP
            activation=ActivationType.Silu,
            quant_type=QuantType.No,
            doweight_stage1=False,
            # following for quant
            w1_scale: Optional[torch.tensor] = None,  # [expert(local_expert:EP), inter_dim, 1]
            w2_scale: Optional[torch.tensor] = None,  # [expert(local_expert:EP), model_dim, 1]
            a1_scale: Optional[torch.tensor] = None,  # [expert(local_expert:EP), 1, model_dim]
            a2_scale: Optional[torch.tensor] = None,  # [expert(local_expert:EP), 1, inter_dim]
            # following for tuning
            block_size_M=None,
            num_local_tokens: Optional[torch.tensor] = None,
            moe_sorting_dispatch_policy=0,
            dtype=None,
            # following for cktile support
            hidden_pad=0,
            intermediate_pad=0,
            bias1=None,
            bias2=None,
        ):
            # the following should be added in aiter.fused_moe.fused_moe
            if hidden_states.shape[0] <= 16 and hidden_states.dtype == torch.bfloat16 and expert_mask is None and activation == ActivationType.Silu and \
                (quant_type == QuantType.No and w1.dtype == torch.bfloat16) or (quant_type == QuantType.per_Token and w1.dtype == torch.float8_e4m3fnuz):
                B = hidden_states.shape[0]
                E, N1, K1 = w1.shape
                N2, K2 = w2.shape[1], w2.shape[2]
                assert N1 == 2 * K2
                gemm1_out = torch.empty([B, TOPK, N1 // 2], dtype=hidden_states.dtype, device=hidden_states.device)
                if B == 1:
                    assert N1 == 2 * K2
                    gemm2_out = torch.zeros([1, N2], dtype=hidden_states.dtype, device=hidden_states.device)
                    moe_gemm_batch1([N1//32, TOPK],[256], w1.dtype, K1, N1, True, hidden_states.data_ptr(), w1.data_ptr(), gemm1_out.data_ptr(), topk_ids.data_ptr(), topk_weight.data_ptr(), w1_scale.data_ptr() if w1_scale is not None else 0, 1)
                    moe_gemm_batch1([N2//32, TOPK],[64], w1.dtype, K2, N2, False, gemm1_out.data_ptr(), w2.data_ptr(), gemm2_out.data_ptr(), topk_ids.data_ptr(), topk_weight.data_ptr(), w2_scale.data_ptr() if w2_scale is not None else 0, 1)
                    return gemm2_out
                else:
                    BLOCK_M = 16
                    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = moe_sorting(
                        topk_ids,
                        topk_weight,
                        E,
                        K1,     # reduce dim is same with output dim
                        hidden_states.dtype,
                        BLOCK_M,
                        expert_mask,
                        num_local_tokens,
                        moe_sorting_dispatch_policy,
                    )

                    moe_gemm_batch([N1 // 32, sorted_expert_ids.shape[0]], [256],
                                    TOPK, K1, N1, True,
                                    hidden_states.data_ptr(), w1.data_ptr(), gemm1_out.data_ptr(), sorted_ids.data_ptr(), sorted_weights.data_ptr(), sorted_expert_ids.data_ptr(), num_valid_ids.data_ptr(), B)
                    moe_gemm_batch([N2 // 32, sorted_expert_ids.shape[0]], [64],
                                    TOPK, K2, N2, False,
                                    gemm1_out.data_ptr(), w2.data_ptr(), moe_buf.data_ptr(), sorted_ids.data_ptr(), sorted_weights.data_ptr(), sorted_expert_ids.data_ptr(), num_valid_ids.data_ptr(), B)

                    return moe_buf
            else:
                return org_fused_moe(hidden_states,
                    w1,  # [expert(local_expert:EP), inter_dim*2, dim] N,K
                    w2,  # [expert(local_expert:EP), dim, inter_dim]
                    topk_weight,
                    topk_ids,
                    expert_mask,  # EP
                    activation,
                    quant_type,
                    doweight_stage1,
                    # following for quant
                    w1_scale,  # [expert(local_expert:EP), inter_dim, 1]
                    w2_scale,  # [expert(local_expert:EP), model_dim, 1]
                    a1_scale,  # [expert(local_expert:EP), 1, model_dim]
                    a2_scale,  # [expert(local_expert:EP), 1, inter_dim]
                    # following for tuning
                    block_size_M,
                    num_local_tokens,
                    moe_sorting_dispatch_policy,
                    dtype,
                    # following for cktile support
                    hidden_pad,
                    intermediate_pad,
                    bias1,
                    bias2)
        org_fused_moe = aiter.fused_moe.fused_moe
        aiter.fused_moe.fused_moe = my_fused_moe

        topk_ids[0,:,] = topk_ids_base[: TOPK]
        w1_qt_aiter = shuffle_weight(w1[0], layout=(16, 16))
        w2_qt_aiter = shuffle_weight(w2[0], layout=(16, 16))
        cur_out = run_aiter(hidden_states=hidden_states[0], w1=w1_qt_aiter, w2=w2_qt_aiter, topk_weight=topk_weight[0], topk_ids=topk_ids[0], w1_scale=w1_scale[0], w2_scale=w2_scale[0])
        i = 0
        for _ in range(10):
            idx_start = random.randint(0, E - TOPK)
            topk_ids[i,:,] = topk_ids_base[idx_start : idx_start + TOPK]
            with cudaPerf(flops, mem_size, name=f"cur[{B=}]"):
                run_aiter(hidden_states=hidden_states[i], w1=w1[i], w2=w2[i], topk_weight=topk_weight[i], topk_ids=topk_ids[i], w1_scale=w1_scale[i], w2_scale=w2_scale[i])
            i = (i + 1) % BUF_COPY

        print(ref_out.shape, cur_out.shape)
        aiter.fused_moe.fused_moe = org_fused_moe
        if not torch.allclose(ref_out, cur_out, rtol=0.02, atol=0.02):
            print(cur_out)
            idx = torch.where(torch.abs(ref_out - cur_out) > 0.02)
            if len(idx[0]):
                print(f'idx = {idx}\nref={ref_out[idx]}\ncur={cur_out[idx]}\n{len(idx[0])}')
            assert 0
        else:
            print("acc OK")

def test():
    torch.cuda.set_device(2)
    torch.set_default_device('cuda')
    torch.manual_seed(0)
    torch.set_printoptions(linewidth=3000, sci_mode=False, edgeitems=8, )

    for i in range(16):
        run_batch(B=i+1, weight_type=torch.bfloat16)

    run_batch(1, weight_type=torch.float8_e4m3fnuz)
    # for i in range(16):
    #     run_batch(B=i+1)

if __name__ == '__main__':
    test()