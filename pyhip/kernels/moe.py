import os
import random
from typing import Optional

import pytest
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

def moe_gemm_loop(J:JIT,
                  weight_dtype,
                  K,
                  N,
                  num_split_k,
                  buff_a,
                  buff_b,
                  p_w_scale,
                  voffset_a,
                  voffset_b,
                  voffset_scale,
                  C_reg,
                  soffset_a = 0,
                  soffset_b = 0,
                  BLOCK_TILE_SIZE_N = 32,
                  BLOCK_TILE_SIZE_M = 16,
                  ):
    assert BLOCK_TILE_SIZE_M % 16 == 0, f'BLOCK_TILE_SIZE_M must be multiple of 16, current {BLOCK_TILE_SIZE_M=}'
    assert BLOCK_TILE_SIZE_N % 32 == 0, f'BLOCK_TILE_SIZE_N must be multiple of 32, current {BLOCK_TILE_SIZE_N=}'
    lane_mod_16 = get_lane_id_mod(J, 16)
    lane_div_16 = get_lane_id_div(J, 16)
    sizeof_f32 = 4
    sizeof_bf16 = 2
    sizeof_w = sizeof_bf16 if weight_dtype == torch.bfloat16 else 1
    # 16 elements for a if fp8
    a_element_num_per_thread = 8 if weight_dtype == torch.bfloat16 else 16

    soffset_kb = J.gpr("su32")
    soffset_ka = J.gpr("su32")
    soffset_kb[0] = soffset_b
    soffset_ka[0] = soffset_a

    # num block in A vert direction
    A_vert = BLOCK_TILE_SIZE_M // 16
    # num block in B horz direction
    B_horz = BLOCK_TILE_SIZE_N // 16
    # there is 16 elements per weight read if fp8, so A should be double read
    A_rep = 1 if weight_dtype == torch.bfloat16 else 2
    # A_reg layout:
    # pinpong  index for vert direction                 index for different mem read(x16bytes)   index for different mfma   minimal for one mfma
    # pinpong  dword4x[?]                               dword4[?]                                dword2[?]                  dword[?](for mfma)
    A_reg = J.gpr(2, A_vert, A_rep, 2, 2, "abf16x2") # 8-bf16 == DWORDx4
    # B_reg layout:
    # bf16: pinpong  n(diff N)/index for different mem read(x16bytes)  index for different mfma  minimal for one mfma
    # fp8:  ..       ..                                                ..                        index for different mfma
    B_reg = J.gpr(2, B_horz, 2, 2, "vbf16x2") # 8-bf16 == DWORDx4

    if weight_dtype != torch.bfloat16:
        v_w_scale = J.gpr(B_horz, 2, 'vf32')
        for n in range(B_horz):
            J.global_load_dword(v_w_scale[n, 0], voffset_scale[n], p_w_scale)

    # ping pong register buffer id
    pp_reg_id = 0
    k_step_wg = num_split_k * 32 if weight_dtype == torch.bfloat16 else num_split_k * 64

    # (A0.B0.C0.D0.A1.B1.C1.D1)[3, 2, 7, 6] = (A1.B1.A0.B0)
    pattern_cvt_bf16 = J.gpr("su32")
    pattern_cvt_bf16[0] = 0x03_02_07_06
    k_max = div_up(K, k_step_wg)
    s_cvt_bf16_bias = J.gpr(1, "su32")
    s_cvt_bf16_bias[0] = 0x00008000

    def load_gen(pp_reg_id):
        for m in range(A_vert):
            buff_a.load_dwordx4(A_reg[pp_reg_id, m, 0], voffset_a[m], soffset_ka)
            if weight_dtype != torch.bfloat16:
                yield buff_a.load_dwordx4(A_reg[pp_reg_id, m, 1], voffset_a[m], soffset_ka + 16)
        for n in range(B_horz):
            yield buff_b.load_dwordx4(B_reg[pp_reg_id, n], voffset_b[n], soffset_kb)

        soffset_kb[0] = soffset_kb[0] + 16 * 64 * num_split_k
        soffset_ka[0] = soffset_ka[0] + a_element_num_per_thread * 4 * num_split_k * sizeof_bf16
        if weight_dtype == torch.bfloat16:
            J.s_waitcnt(mod=f"vmcnt({B_horz + A_vert})")
        else:
            J.s_waitcnt(mod=f"vmcnt({B_horz + A_vert * 2})")
    
    def mfma_gen(pp_reg_id):
        if weight_dtype != torch.bfloat16:
            # decompress
            v_w_f32 = J.gpr(2, 2, 'vf32', align=4)
            v_w_bf16 = J.gpr(B_horz, 2, 'vf32', align=4)
            for i in range(2):
                for j in range(2):
                    for n in range(B_horz):
                        J.v_cvt_pk_f32_fp8(v_w_f32[0], B_reg[pp_reg_id, n, i, j])
                        J.v_cvt_pk_f32_fp8_sdwa(v_w_f32[1], B_reg[pp_reg_id, n, i, j], mod='src0_sel:WORD_1')
                        J.v_pk_mul_f32(v_w_f32[0], v_w_f32[0], v_w_scale[n])
                        J.v_pk_mul_f32(v_w_f32[1], v_w_f32[1], v_w_scale[n])
                        J.v_add_u32(v_w_f32[0, 0], v_w_f32[0, 0], s_cvt_bf16_bias)
                        J.v_add_u32(v_w_f32[0, 1], v_w_f32[0, 1], s_cvt_bf16_bias)
                        J.v_add_u32(v_w_f32[1, 0], v_w_f32[1, 0], s_cvt_bf16_bias)
                        J.v_add_u32(v_w_f32[1, 1], v_w_f32[1, 1], s_cvt_bf16_bias)
                        J.v_perm_b32(v_w_bf16[n, 0], v_w_f32[0, 0], v_w_f32[0, 1], pattern_cvt_bf16)
                        J.v_perm_b32(v_w_bf16[n, 1], v_w_f32[1, 0], v_w_f32[1, 1], pattern_cvt_bf16)
                    # 2, A_rep, 2, 2
                    for n in range(B_horz):
                        for m in range(A_vert):
                            yield J.v_mfma_f32_16x16x16_bf16(C_reg[n, m], v_w_bf16[n], A_reg[pp_reg_id, m, i, j], C_reg[n, m])
        else:
            for m in range(A_vert):
                for n in range(B_horz):
                    yield J.v_mfma_f32_16x16x16_bf16(C_reg[n, m], B_reg[pp_reg_id, n, 0], A_reg[pp_reg_id, m, 0, 0], C_reg[n, m])
                    yield J.v_mfma_f32_16x16x16_bf16(C_reg[n, m], B_reg[pp_reg_id, n, 1], A_reg[pp_reg_id, m, 0, 1], C_reg[n, m])

    def loop(pp_reg_id):
        loader = load_gen(pp_reg_id)
        mfma = mfma_gen(1 - pp_reg_id)

        J.emitter()([loader])

        J.emitter()([mfma])

    # prolog
    loader = load_gen(0)
    J.emitter()([loader])
    if weight_dtype != torch.bfloat16:
        for n in range(B_horz):
            J.v_mov_b32(v_w_scale[n, 1], v_w_scale[n, 0])
    pp_reg_id = 1

    def tail(pp_reg_id):
        J.s_waitcnt(mod=f"vmcnt(0)")
        mfma = mfma_gen(pp_reg_id)
        J.emitter()([mfma])

    if isinstance(K, int):
        if weight_dtype == torch.bfloat16:
            assert K % (num_split_k * 32) == 0, f'K must be multiple of {num_split_k * 32}'
        else:
            # a wave needs at least 64 elements
            assert K % 64 == 0, 'K must be multiple of 64'
    
        for k in range(0, k_max - 1):
            loop(pp_reg_id)
            pp_reg_id ^= 1

        # tail
        tail(1 - pp_reg_id)

    else:
        cur_k = J.gpr("si32")
        cur_k[0] = 0
        # at least 2 blocks, TODO: add check?
        k_loop_cnt = K // k_step_wg - 2

        # unroll 2 times for pin/pong buffer, align #loop to even
        with J.While(cur_k[0] < k_loop_cnt):
            loop(1)
            loop(0)
            cur_k[0] += 2
        J.Jump("odd_k_block", cur_k[0] == k_loop_cnt + 1)
        # there are 2 blocks left
        loop(1)
        # tail
        tail(1)
        J.Jump("k_block_end")
        # 1 block + tail
        J.Label("odd_k_block")
        # tail
        tail(0)
        J.Label("k_block_end")

@jit()
def moe_gemm_batch1(J:JIT,
                    weight_dtype,
                    #K,            # compile-time args
                    #N,            # compile-time args
                    with_silu,    # compile-time args
                    p_input:"void*",
                    p_weight:"void*",
                    p_output:"void*",
                    p_topk_ids:"void*",
                    p_topk_weight:"float*", 
                    p_w_scale:"float*",
                    M:"int",
                    N:"int",
                    K:"int",
                    ):

    BLOCK_TILE_SIZE_M = 16  # nBM * 16
    BLOCK_TILE_SIZE_N = 32  # nBN * 32
    BLOCK_SIZE_K = 32
    #assert K % BLOCK_SIZE_K == 0
    assert BLOCK_TILE_SIZE_M % 16 == 0
    assert BLOCK_TILE_SIZE_N % 32 == 0

    use_split_k = with_silu
    num_split_k = 4 if use_split_k else 1

    sizeof_bf16 = 2
    sizeof_f32 = 4
    sizeof_w = sizeof_bf16 if weight_dtype == torch.bfloat16 else 1
    stride_A = K * sizeof_bf16
    stride_B = K * sizeof_w

    C_reg = J.gpr(2, 1, 4, "vf32")

    C_reg[:] = 0

    # expert index(not id)
    e_idx = J.blockIdx.y
    s_e_id = J.gpr(1, 'su32')
    J.s_load_dword(s_e_id, p_topk_ids, e_idx[0] * 4)
    if not with_silu:
        s_weight = J.gpr(1, 'su32')
        J.s_load_dword(s_weight, p_topk_weight, e_idx[0] * 4)

    # hide following initialization into s_waitcnt
    # one WG per CU,  4 waves split on K
    lane_mod_16 = get_lane_id_mod(J, 16)
    lane_div_16 = get_lane_id_div(J, 16)
    warp_id = J.gpr("su32")
    J.v_readfirstlane_b32(warp_id, J.threadIdx.x[0] // 64)

    output_offset = (16 if with_silu else 32) * J.blockIdx.x * sizeof_bf16
    voffset_b = J.gpr(2, 'vu32')
    if with_silu:
        voffset_b[0] = 16 * J.blockIdx.x * stride_B + J.threadIdx.x * 16
        # [E, M, N//2]
        output_offset = e_idx * (N // 2 * sizeof_bf16) + output_offset
        voffset_b[1] = voffset_b[0] + (N // 2) * K * sizeof_w
    else:
        voffset_b[0] = 32 * J.blockIdx.x * stride_B + J.threadIdx.x * 16
        # [M, N]
        p_input[:] = p_input[:] + e_idx * (K * sizeof_bf16)
        voffset_b[1] = voffset_b[0] + K * (16 * sizeof_w)
    p_output[:] = p_output[:] + output_offset
    if weight_dtype != torch.bfloat16:
        p_w_scale[:] += (16 if with_silu else 32) * sizeof_f32 * J.blockIdx.x

    # 16 elements for a if fp8
    a_element_num_per_thread = 8 if weight_dtype == torch.bfloat16 else 16
    voffset_a = J.gpr((J.threadIdx.x % 16) * stride_A + (J.threadIdx.x // 16) * (a_element_num_per_thread * sizeof_bf16))
    buff_a = J.Buffer(p_input, M * K * sizeof_bf16)

    # wait for s_e_id
    J.s_waitcnt(mod=f"lgkmcnt(0)")

    p_weight[:] = p_weight[:] + s_e_id * (N * K * sizeof_w)
    buff_b = J.Buffer(p_weight, K * N * sizeof_w)

    voffset_scale = J.gpr(2, 'vu32')
    if weight_dtype != torch.bfloat16:
        voffset_scale[0] = J.gpr(s_e_id * (N * sizeof_f32)) + lane_mod_16 * sizeof_f32
        voffset_scale[1] = voffset_scale[0] + (N // 2 * sizeof_f32 if with_silu else 16 * sizeof_f32)

    moe_gemm_loop(J, weight_dtype, K, N, num_split_k,
                  buff_a, buff_b, p_w_scale,
                  voffset_a, voffset_b, voffset_scale, C_reg)

    s_cvt_bf16_bias = J.gpr(1, "su32")
    s_cvt_bf16_bias[0] = 0x00008000
    if use_split_k:
        assert with_silu
        # split K
        lds_buff = J.LDSTensor([4 * 16, 32], torch.float)

        vaddr = J.gpr("vu32")
        lds_buff.write("b128", C_reg[0, 0], lane_mod_16 + warp_id * 16, lane_div_16 * 4)
        lds_buff.write("b128", C_reg[1, 0], lane_mod_16 + warp_id * 16, lane_div_16 * 4 + 16)

        J.s_waitcnt(mod=f"lgkmcnt(0)")
        J.s_barrier()
        # each wave reduce 4 rows
        vrow = J.gpr(lane_div_16 + warp_id * 4)
        gate_up = J.gpr(4, 2, "vf32")
        # interleave 
        lds_buff.read("b32", gate_up[0], vrow + 0 * 16, (lane_mod_16), offset1 = 16)
        lds_buff.read("b32", gate_up[1], vrow + 1 * 16, (lane_mod_16), offset1 = 16)
        lds_buff.read("b32", gate_up[2], vrow + 2 * 16, (lane_mod_16), offset1 = 16)
        lds_buff.read("b32", gate_up[3], vrow + 3 * 16, (lane_mod_16), offset1 = 16)

        J.s_waitcnt(mod=f"lgkmcnt(2)")
        J.v_pk_add_f32(gate_up[0], gate_up[0], gate_up[1])
        J.s_waitcnt(mod=f"lgkmcnt(0)")
        J.v_pk_add_f32(gate_up[2], gate_up[2], gate_up[3])
        J.v_pk_add_f32(gate_up[0], gate_up[0], gate_up[2])

        if with_silu:
            out = J.gpr(gate_up[0, 1] * J.silu(gate_up[0, 0]))
            J.v_add_u32(out[0], out[0], s_cvt_bf16_bias)
            vaddr = J.gpr(vrow * (N // 2 * sizeof_bf16) + lane_mod_16 * (sizeof_bf16))
            with J.ExecMask(vrow < M[0]):
                J.global_store_short_d16_hi(vaddr, out[0], p_output)
        else:
            # convert to bf16
            vaddr = J.gpr(vrow * (N * sizeof_bf16) + lane_mod_16 * (2 * sizeof_bf16))
            with J.ExecMask(vrow < M[0]):
                out = J.gpr("vf32")
                out[0] = (gate_up[0, 0] >> 16) | (gate_up[0, 1] & 0xFFFF0000)
                J.global_store_dword(vaddr, out[0], p_output)
    else:
        # split N
        assert not with_silu
        vaddr = J.gpr(lane_mod_16 * (N * sizeof_bf16) + lane_div_16 * (4 * sizeof_bf16))
        creg_low = J.gpr(2, 2, "vbf16x2", align=4)
        J.s_waitcnt(mod=f"lgkmcnt(0)")
        for i in range(2):
            for j in range(4):
                C_reg[i, 0, j] = C_reg[i, 0, j] * s_weight
                J.v_add_u32(C_reg[i, 0, j], C_reg[i, 0, j], s_cvt_bf16_bias)
        for i in range(2):
            creg_low[i,0] = (C_reg[i, 0, 0] >> 16) |(C_reg[i, 0, 1] & 0xFFFF0000)
            creg_low[i,1] = (C_reg[i, 0, 2] >> 16) |(C_reg[i, 0, 3] & 0xFFFF0000)

        with J.ExecMask(lane_mod_16 < M[0]):
            J.global_atomic_pk_add_bf16(vaddr                       , creg_low[0, 0], p_output)
            J.global_atomic_pk_add_bf16(vaddr                    + 4, creg_low[0, 1], p_output)
            J.global_atomic_pk_add_bf16(vaddr + 16 * sizeof_bf16    , creg_low[1, 0], p_output)
            J.global_atomic_pk_add_bf16(vaddr + 16 * sizeof_bf16 + 4, creg_low[1, 1], p_output)

@jit()
def moe_gemm_batch(J:JIT,
                   weight_dtype,
                   #TOPK,
                   #K,            # compile-time args
                   #N,            # compile-time args
                   with_silu,    # compile-time args
                   p_input:"void*",
                   p_weight:"void*",
                   p_output:"void*",
                   # sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids
                   p_sorted_ids:"void*",
                   p_sorted_weights:"float*",
                   p_sorted_expert_ids:"void*",
                   p_num_valid_ids:"void*",
                   p_w_scale:"float*",
                   M:"int",
                   N:"int",
                   K:"int",
                   TOPK:"int"):
    BLOCK_TILE_SIZE_M = 16  # nBM * 16
    BLOCK_TILE_SIZE_N = 32  # nBN * 32
    BLOCK_SIZE_K = 32
    #assert K % BLOCK_SIZE_K == 0
    assert BLOCK_TILE_SIZE_M % 16 == 0
    assert BLOCK_TILE_SIZE_N % 32 == 0

    use_split_k = with_silu
    num_split_k = 4 if use_split_k else 1

    sizeof_bf16 = 2
    sizeof_f32 = 4
    sizeof_w = sizeof_bf16 if weight_dtype == torch.bfloat16 else 1
    stride_A = K * sizeof_bf16
    stride_B = K * sizeof_w

    C_reg = J.gpr(2, 1, 4, "vf32")

    C_reg[:] = 0

    # grid: N // 32, sorted_expert_ids.shape[0]
    # expert index in p_sorted_expert_ids
    e_idx = J.blockIdx.y
    s_e_id = J.gpr(1, 'su32')
    J.s_load_dword(s_e_id, p_sorted_expert_ids, e_idx[0] * 4)
    max_id = J.gpr(1, 'su32')
    J.s_load_dword(max_id, p_num_valid_ids, 0)
    J.s_waitcnt(mod=f"lgkmcnt(0)")
    # invalid padding section
    J.Jump("continue_following", e_idx * BLOCK_TILE_SIZE_M < max_id)
    J.s_endpgm()
    J.Label("continue_following")

    # hide following initialization into s_waitcnt
    p_sorted_ids[:] += e_idx * (BLOCK_TILE_SIZE_M * 4)
    # one WG per CU,  4 waves split on K
    lane_mod_16 = get_lane_id_mod(J, 16)
    lane_div_16 = get_lane_id_div(J, 16)
    warp_id = J.gpr("su32")
    J.v_readfirstlane_b32(warp_id, J.threadIdx.x[0] // 64)

    v_sorted_id = J.gpr('vu32')
    # assume only BLOCK_TILE_SIZE_M==16
    assert BLOCK_TILE_SIZE_M == 16
    J.global_load_dword(v_sorted_id, lane_mod_16 << 2, p_sorted_ids)
    voffset_b = J.gpr(2, 'vu32')
    if with_silu:
        voffset_b[0] = 16 * J.blockIdx.x * stride_B + J.threadIdx.x * 16
        voffset_b[1] = voffset_b[0] + (N // 2) * K * sizeof_w
    else:
        p_sorted_weights[:] += e_idx * (BLOCK_TILE_SIZE_M * 4)
        v_weight = J.gpr('vf32')
        voffset_b[0] = 32 * J.blockIdx.x * stride_B + J.threadIdx.x * 16
        J.global_load_dword(v_weight, lane_mod_16 << 2, p_sorted_weights)
        voffset_b[1] = voffset_b[0] + K * (16 * sizeof_w)

    p_output[:] = p_output[:] + (16 if with_silu else 32) * J.blockIdx.x * sizeof_bf16

    buff_a = J.Buffer(p_input, M * (K * sizeof_bf16) if with_silu else M * (TOPK * K * sizeof_bf16))

    # wait for v_sorted_id
    J.s_waitcnt(mod=f"vmcnt(0)")
    p_weight[:] = p_weight[:] + s_e_id * (N * K * sizeof_w)
    buff_b = J.Buffer(p_weight, K * N * sizeof_w)

    v_token_id = v_sorted_id & 0xffffff
    # 16 elements for a if fp8
    a_element_num_per_thread = 8 if weight_dtype == torch.bfloat16 else 16
    if with_silu:
        voffset_a = J.gpr(v_token_id * stride_A + (J.threadIdx.x // 16) * (a_element_num_per_thread * sizeof_bf16))
    else:
        v_topk_id = v_sorted_id >> 24
        # input layout: [B, TOPK, INTER_MEDIA]
        voffset_a = J.gpr(v_token_id * TOPK * K * sizeof_bf16 + v_topk_id * (K * sizeof_bf16) + (J.threadIdx.x // 16) * (a_element_num_per_thread * sizeof_bf16))

    voffset_scale = J.gpr(2, 'vu32')
    if weight_dtype != torch.bfloat16:
        p_w_scale[:] += (16 if with_silu else 32) * sizeof_f32 * J.blockIdx.x
        voffset_scale[0] = J.gpr(s_e_id * (N * sizeof_f32)) + lane_mod_16 * sizeof_f32
        voffset_scale[1] = voffset_scale[0] + (N // 2 * sizeof_f32 if with_silu else 16 * sizeof_f32)

    moe_gemm_loop(J, weight_dtype, K, N, num_split_k,
                  buff_a, buff_b, p_w_scale,
                  voffset_a, voffset_b, voffset_scale, C_reg)

    s_cvt_bf16_bias = J.gpr(1, "su32")
    s_cvt_bf16_bias[0] = 0x00008000
    if use_split_k:
        assert with_silu
        # split K
        lds_buff = J.LDSTensor([4 * 16, 32], torch.float)

        vaddr = J.gpr("vu32")
        lds_buff.write("b128", C_reg[0, 0], lane_mod_16 + warp_id * 16, lane_div_16 * 4)
        lds_buff.write("b128", C_reg[1, 0], lane_mod_16 + warp_id * 16, lane_div_16 * 4 + 16)

        J.s_waitcnt(mod=f"lgkmcnt(0)")
        J.s_barrier()
        v_sorted_id_permute = J.gpr("vu32")

        # each wave reduce 4 rows
        vrow = J.gpr(lane_div_16 + warp_id * 4)
        J.ds_bpermute_b32(v_sorted_id_permute, vrow * 4, v_sorted_id)
        gate_up = J.gpr(4, 2, "vf32")
        # interleave 
        lds_buff.read("b32", gate_up[0], vrow + 0 * 16, (lane_mod_16), offset1 = 16)
        lds_buff.read("b32", gate_up[1], vrow + 1 * 16, (lane_mod_16), offset1 = 16)
        lds_buff.read("b32", gate_up[2], vrow + 2 * 16, (lane_mod_16), offset1 = 16)
        lds_buff.read("b32", gate_up[3], vrow + 3 * 16, (lane_mod_16), offset1 = 16)

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
            J.v_add_u32(out[0], out[0], s_cvt_bf16_bias)
            vaddr = J.gpr(v_token_id * (N // 2 * sizeof_bf16 * TOPK) + v_topk_id * (N // 2 * sizeof_bf16) + lane_mod_16 * sizeof_bf16)
            with J.ExecMask(v_token_id < M[0]):
                J.global_store_short_d16_hi(vaddr, out[0], p_output)
        else:
            # convert to bf16
            vaddr = J.gpr(vrow * (N * sizeof_bf16) + lane_mod_16 * (2 * sizeof_bf16))
            with J.ExecMask(vrow < M[0]):
                out = J.gpr("vf32")
                out[0] = (gate_up[0, 0] >> 16) | (gate_up[0, 1] & 0xFFFF0000)
                J.global_store_dword(vaddr, out[0], p_output)
    else:
        # split N
        assert not with_silu
        # output layout: [B, N]
        vaddr = J.gpr(v_token_id * (N * sizeof_bf16) + lane_div_16 * (4 * sizeof_bf16))
        creg_low = J.gpr(2, 2, "vbf16x2", align=4)
        J.s_waitcnt(mod=f"lgkmcnt(0)")
        for i in range(2):
            for j in range(4):
                C_reg[i, 0, j] = C_reg[i, 0, j] * v_weight
                J.v_add_u32(C_reg[i, 0, j], C_reg[i, 0, j], s_cvt_bf16_bias)
        for i in range(2):
            creg_low[i, 0] = (C_reg[i, 0, 0] >> 16) | (C_reg[i, 0, 1] & 0xFFFF0000)
            creg_low[i, 1] = (C_reg[i, 0, 2] >> 16) | (C_reg[i, 0, 3] & 0xFFFF0000)
        with J.ExecMask(v_token_id < M[0]):
            J.global_atomic_pk_add_bf16(vaddr                       , creg_low[0, 0], p_output)
            J.global_atomic_pk_add_bf16(vaddr                    + 4, creg_low[0, 1], p_output)
            J.global_atomic_pk_add_bf16(vaddr + 16 * sizeof_bf16    , creg_low[1, 0], p_output)
            J.global_atomic_pk_add_bf16(vaddr + 16 * sizeof_bf16 + 4, creg_low[1, 1], p_output)

# BLOCK_TILE_SIZE_N/M is configurable
@jit()
def moe_gemm_batch_vmn(J:JIT,
                       weight_dtype,
                       TOPK,
                       K,            # compile-time args
                       N,            # compile-time args
                       with_silu,    # compile-time args
                       BLOCK_TILE_SIZE_M,
                       BLOCK_TILE_SIZE_N,
                       p_input:"void*",
                       p_weight:"void*",
                       p_output:"void*",
                       # sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids
                       p_sorted_ids:"void*",
                       p_sorted_weights:"float*",
                       p_sorted_expert_ids:"void*",
                       p_num_valid_ids:"void*",
                       p_w_scale:"float*",
                       M:"int",):
    #assert K % BLOCK_SIZE_K == 0
    assert BLOCK_TILE_SIZE_M % 16 == 0
    assert BLOCK_TILE_SIZE_N % 32 == 0
    assert N % BLOCK_TILE_SIZE_N == 0

    use_split_k = with_silu
    num_split_k = 4 if use_split_k else 1
    BLOCK_TILE_SIZE_N_HALF = BLOCK_TILE_SIZE_N // 2

    sizeof_bf16 = 2
    sizeof_f32 = 4
    sizeof_w = sizeof_bf16 if weight_dtype == torch.bfloat16 else 1
    stride_A = K * sizeof_bf16
    stride_B = K * sizeof_w

    A_vert = BLOCK_TILE_SIZE_M // 16
    B_horz = BLOCK_TILE_SIZE_N // 16

    C_reg = J.gpr(B_horz, A_vert, 4, "vf32")

    C_reg[:] = 0

    # grid: N // 32, sorted_expert_ids.shape[0]
    # expert index in p_sorted_expert_ids
    e_idx = J.blockIdx.y
    s_e_id = J.gpr(1, 'su32')
    J.s_load_dword(s_e_id, p_sorted_expert_ids, e_idx[0] * 4)
    max_id = J.gpr(1, 'su32')
    J.s_load_dword(max_id, p_num_valid_ids, 0)
    J.s_waitcnt(mod=f"lgkmcnt(0)")
    # invalid padding section
    J.Jump("continue_following", e_idx * BLOCK_TILE_SIZE_M < max_id)
    J.s_endpgm()
    J.Label("continue_following")

    # hide following initialization into s_waitcnt
    p_sorted_ids[:] += e_idx * (BLOCK_TILE_SIZE_M * 4)
    # one WG per CU,  4 waves split on K
    lane_mod_16 = get_lane_id_mod(J, 16)
    lane_div_16 = get_lane_id_div(J, 16)
    warp_id = J.gpr("su32")
    J.v_readfirstlane_b32(warp_id, J.threadIdx.x[0] // 64)

    v_sorted_id = J.gpr(A_vert, 'vu32')
    J.global_load_dword(v_sorted_id[0], lane_mod_16 << 2, p_sorted_ids)
    for n in range(1, A_vert):
        p_sorted_ids[:] += 16 * 4
        J.global_load_dword(v_sorted_id[n], lane_mod_16 << 2, p_sorted_ids)
    voffset_b = J.gpr(B_horz, 'vu32')
    if with_silu:
        voffset_b[0] = BLOCK_TILE_SIZE_N_HALF * J.blockIdx.x * stride_B + J.threadIdx.x * 16
        voffset_b[B_horz // 2] = voffset_b[0] + (N // 2) * K * sizeof_w
        for m in range(1, B_horz // 2):
            voffset_b[m] = voffset_b[0] + K * (16 * sizeof_w * m)
            voffset_b[B_horz // 2 + m] = voffset_b[B_horz // 2] + K * (16 * sizeof_w * m)
    else:
        p_sorted_weights[:] += e_idx * (BLOCK_TILE_SIZE_M * 4)
        v_weight = J.gpr(A_vert, 'vf32')
        voffset_b[0] = BLOCK_TILE_SIZE_N * J.blockIdx.x * stride_B + J.threadIdx.x * 16
        J.global_load_dword(v_weight[0], lane_mod_16 << 2, p_sorted_weights)
        for m in range(1, B_horz):
            voffset_b[m] = voffset_b[0] + K * (16 * sizeof_w * m)
        for m in range(1, A_vert):
            p_sorted_weights[:] += 16 * 4
            J.global_load_dword(v_weight[m], lane_mod_16 << 2, p_sorted_weights)

    p_output[:] = p_output[:] + (BLOCK_TILE_SIZE_N_HALF if with_silu else BLOCK_TILE_SIZE_N) * J.blockIdx.x * sizeof_bf16

    buff_a = J.Buffer(p_input, M * (K * sizeof_bf16) if with_silu else M * (TOPK * K * sizeof_bf16))

    # wait for v_sorted_id
    J.s_waitcnt(mod=f"vmcnt(0)")
    voffset_a = J.gpr(A_vert, 'vu32')
    v_topk_id = J.gpr(A_vert, 'vu32')
    v_token_id = J.gpr(A_vert, 'vu32')
    # 16 elements for a if fp8
    a_element_num_per_thread = 8 if weight_dtype == torch.bfloat16 else 16
    for m in range(A_vert):
        v_token_id[m] = v_sorted_id[m] & 0xffffff
        if with_silu:
            voffset_a[m] = J.gpr(v_token_id[m] * stride_A + (J.threadIdx.x // 16) * (a_element_num_per_thread * sizeof_bf16))
        else:
            v_topk_id[m] = v_sorted_id[m] >> 24
            # input layout: [B, TOPK, INTER_MEDIA]
            voffset_a[m] = J.gpr(v_token_id[m] * TOPK * K * sizeof_bf16 + v_topk_id[m] * (K * sizeof_bf16) + (J.threadIdx.x // 16) * (a_element_num_per_thread * sizeof_bf16))

    voffset_scale = J.gpr(B_horz, 'vu32')
    if weight_dtype != torch.bfloat16:
        p_w_scale[:] += (BLOCK_TILE_SIZE_N_HALF if with_silu else BLOCK_TILE_SIZE_N) * sizeof_f32 * J.blockIdx.x
        voffset_scale[0] = J.gpr(s_e_id * (N * sizeof_f32)) + lane_mod_16 * sizeof_f32
        if with_silu:
            voffset_scale[B_horz // 2] = voffset_scale[0] + N // 2 * sizeof_f32
            for m in range(1, B_horz // 2):
                voffset_scale[m] = voffset_scale[0] + 16 * sizeof_f32 * m
                voffset_scale[B_horz // 2 + m] = voffset_scale[B_horz // 2] + 16 * sizeof_f32 * m
        else:
            for m in range(1, B_horz):
                voffset_scale[m] = voffset_scale[0] + 16 * sizeof_f32 * m

    p_weight[:] = p_weight[:] + s_e_id * (N * K * sizeof_w)
    buff_b = J.Buffer(p_weight, K * N * sizeof_w)

    moe_gemm_loop(J, weight_dtype, K, N, num_split_k,
                  buff_a, buff_b, p_w_scale,
                  voffset_a, voffset_b, voffset_scale, C_reg, BLOCK_TILE_SIZE_N=BLOCK_TILE_SIZE_N, BLOCK_TILE_SIZE_M=BLOCK_TILE_SIZE_M)

    s_cvt_bf16_bias = J.gpr(1, "su32")
    s_cvt_bf16_bias[0] = 0x00008000
    if use_split_k:
        # split K
        lds_buff = J.LDSTensor([4 * BLOCK_TILE_SIZE_M, 32], torch.float)
        # each 32 N as a group to compute gate*up
        n_groups = BLOCK_TILE_SIZE_N // 32
        n_half0 = 0
        n_half1 = B_horz // 2
        for n in range(n_groups):
            vaddr = J.gpr("vu32")
            for m in range(A_vert):
                lds_buff.write("b128", C_reg[n + n_half0, m], lane_mod_16 + warp_id * 16 + m * 64, lane_div_16 * 4)
                lds_buff.write("b128", C_reg[n + n_half1, m], lane_mod_16 + warp_id * 16 + m * 64, lane_div_16 * 4 + 16)

            J.s_waitcnt(mod=f"lgkmcnt(0)")
            J.s_barrier()
            v_sorted_id_permute = J.gpr("vu32")
            # each wave reduce 4 rows once
            vrow = J.gpr(lane_div_16 + warp_id * 4)
            for m in range(A_vert):
                J.ds_bpermute_b32(v_sorted_id_permute, vrow * 4, v_sorted_id[m])
                gate_up = J.gpr(4, 2, "vf32")
                # interleave 
                lds_buff.read("b32", gate_up[0], vrow + (0 * 16 + m * 64), lane_mod_16, offset1 = 16)
                lds_buff.read("b32", gate_up[1], vrow + (1 * 16 + m * 64), lane_mod_16, offset1 = 16)
                lds_buff.read("b32", gate_up[2], vrow + (2 * 16 + m * 64), lane_mod_16, offset1 = 16)
                lds_buff.read("b32", gate_up[3], vrow + (3 * 16 + m * 64), lane_mod_16, offset1 = 16)

                J.s_waitcnt(mod=f"lgkmcnt(2)")
                J.v_pk_add_f32(gate_up[0], gate_up[0], gate_up[1])
                J.s_waitcnt(mod=f"lgkmcnt(0)")
                J.v_pk_add_f32(gate_up[2], gate_up[2], gate_up[3])
                J.v_pk_add_f32(gate_up[0], gate_up[0], gate_up[2])

                out = J.gpr(gate_up[0, 1] * J.silu(gate_up[0, 0]))
                # output: [B, TOPK, N]
                v_token_id = J.gpr(v_sorted_id_permute[0] & 0xffffff)
                v_topk_id = J.gpr(v_sorted_id_permute[0] >> 24)
                J.v_add_u32(out[0], out[0], s_cvt_bf16_bias)
                vaddr = J.gpr(v_token_id * (N // 2 * sizeof_bf16 * TOPK) + v_topk_id * (N // 2 * sizeof_bf16) + lane_mod_16 * sizeof_bf16 + n * 16 * sizeof_bf16)
                with J.ExecMask(v_token_id < M[0]):
                    J.global_store_short_d16_hi(vaddr, out[0], p_output)
            J.s_barrier()
    else:
        for n in range(B_horz):
            # split N
            assert not with_silu
            # output layout: [B, N]
            for m in range(A_vert):
                vaddr = J.gpr(v_token_id[m] * (N * sizeof_bf16) + lane_div_16 * (4 * sizeof_bf16) + n * 4 * (4 * sizeof_bf16))
                creg_low = J.gpr(2, "vbf16x2", align=2)
                J.s_waitcnt(mod=f"lgkmcnt(0)")
                for j in range(4):
                    C_reg[n, m, j] = C_reg[n, m, j] * v_weight[m]
                    J.v_add_u32(C_reg[n, m, j], C_reg[n, m, j], s_cvt_bf16_bias)
                creg_low[0] = (C_reg[n, m, 0] >> 16) | (C_reg[n, m, 1] & 0xFFFF0000)
                creg_low[1] = (C_reg[n, m, 2] >> 16) | (C_reg[n, m, 3] & 0xFFFF0000)
                with J.ExecMask(v_token_id[m] < M[0]):
                    J.global_atomic_pk_add_bf16(vaddr    , creg_low[0], p_output)
                    J.global_atomic_pk_add_bf16(vaddr + 4, creg_low[1], p_output)

def down_kernel(J, mfma_MN, num_mfma_n, BM, N, K,
                A, # A = J.gpr(num_mfma_m, num_mfma_k, 4, "abf16x2")
                lds_token_ids,  # 256 int32 token_id
                lds_weights,    # 256 fp32 weights 
                pB:"void*",
                pC:"void*",
                M):
    sizeof_DW = 4
    sizeof_DW2 = sizeof_DW * 2
    sizeof_DW4 = sizeof_DW * 4
    sizeof_bf16 = 2

    #pA[:] = pA[:] + J.blockIdx.x * (M*K*sizeof_bf16)
    #pB[:] = pB[:] + J.blockIdx.x * (N*K*sizeof_bf16)
    #pC[:] = pC[:] + J.blockIdx.x * (M*N*sizeof_bf16)

    # given DW4 lane size, how many bf16 items along K direction
    mfma_K = (64//mfma_MN) * (sizeof_DW4//sizeof_bf16) # mfma_K = 32

    # load A [BM x K] bf16 into AccGPRs 
    num_mfma_m = J.div(BM, mfma_MN)
    num_mfma_k = J.div(K, mfma_K)  # K=96, num_mfma_k=3

    """
    A = J.gpr(num_mfma_m, num_mfma_k, 4, "abf16x2")
    buff_a = J.Buffer(pA, BM * K * sizeof_bf16)
    row = J.lane_id % mfma_MN
    col = J.lane_id // mfma_MN
    vaddr = J.gpr(row * (K*sizeof_bf16) + col * sizeof_DW4)
    soff = J.gpr("su32")
    soff[0] = 0
    for m in range(num_mfma_m):
        for k in range(num_mfma_k):
            buff_a.load_dwordx4(A[m,k], vaddr, soff, offset12=k*mfma_K*sizeof_bf16)
        soff[0] = soff[0] + mfma_MN * K * sizeof_bf16
    """

    # 4 warps work in parallel along N dimension
    buff_b = J.Buffer(pB, N * K * sizeof_bf16)
    # ping-pong buffer
    B = J.gpr(2, num_mfma_n, num_mfma_k, 4, "abf16x2")
    C = J.gpr(2, num_mfma_m, num_mfma_n, 4, "vf32")

    # prelog0, load Bn0
    # prelog1, load Bn1, compute Cn0
    # loop:    load Bn2, compute Cn1, store Cn0 to LDS & load Cn0 & store to HBM
    voff_b = J.gpr(J.lane_id * sizeof_DW4 + J.gpr(J.warp_id * (mfma_MN * K * sizeof_bf16)))
    soff_b = J.gpr("su32")
    soff_b[0] = 0
    def loadB_generator(index):
        for n in range(num_mfma_n):
            for k in range(num_mfma_k):
                yield buff_b.load_dwordx4(B[index,n,k], voff_b, soff_b)
                soff_b[0] = soff_b[0] + mfma_MN * mfma_K * sizeof_bf16
            soff_b[0] = soff_b[0] + (3*num_mfma_k * mfma_MN * mfma_K * sizeof_bf16)

    def mfma_generator(index):
        for k in range(num_mfma_k):
            for m in range(num_mfma_m):
                for n in range(num_mfma_n):
                    Ci = 0 if k == 0 else C[index,m,n]
                    yield J.v_mfma_f32_16x16x16_bf16(C[index,m,n], B[index,n,k,0:1], A[m,k,0:1], Ci)
        for k in range(num_mfma_k):
            for m in range(num_mfma_m):
                for n in range(num_mfma_n):
                    yield J.v_mfma_f32_16x16x16_bf16(C[index,m,n], B[index,n,k,2:3], A[m,k,2:3], C[index,m,n])

    mfma_cycles = 16 if mfma_MN == 16 else 32
    emit_mfma = J.emitter(mfma_cycles)
    emit_bload = J.emitter(1)

    # prelog0, load Bn0
    emit_bload([loadB_generator(0)])

    # prelog1, load Bn1, compute Cn0
    emit_bload([loadB_generator(1)])
    J.s_waitcnt(mod=f"vmcnt({num_mfma_n*num_mfma_k})")
    J.s_waitcnt(mod=f"lgkmcnt(0)")

    emit_mfma([mfma_generator(0)])

    # loop:    load Bn2, compute Cn1, store Cn0 to LDS & load Cn0 & store to HBM
    s_cvt_bf16_bias = J.get_sgpr_const(0x00008000)

    vmem_lane_size = sizeof_DW

    lds_padding = (4 if vmem_lane_size == sizeof_DW else 8) * sizeof_bf16 # to avoid bank-conflict
    lds_width = num_mfma_n * 4 * mfma_MN * sizeof_bf16
    lds_stride = lds_width + lds_padding

    # WG level write C into LDS
    row = J.threadIdx.x % mfma_MN
    col = J.threadIdx.x // mfma_MN
    voff_c_lds_w = J.gpr(row * lds_stride + col * (4 * sizeof_bf16))

    # WG level load C from LDS
    num_lanes_ldsr = J.div(lds_width, vmem_lane_size)
    assert num_lanes_ldsr <= 64, num_lanes_ldsr
    col = J.threadIdx.x % num_lanes_ldsr
    row = J.threadIdx.x // num_lanes_ldsr
    num_rows_per_load = J.div(256, num_lanes_ldsr)
    num_loads = J.div(num_mfma_m * mfma_MN, num_rows_per_load)
    voff_c_lds_r = J.gpr(row * lds_stride + col * (vmem_lane_size))
    vmem_stride = N * sizeof_bf16
    v_weights = J.gpr(num_mfma_m, 2, "vf32") # pkmul
    voff_vmem = J.gpr(num_loads, "vu32")
    for m in range(num_mfma_m):
        J.ds_read_b32(v_weights[m,0], (m*mfma_MN + (J.lane_id % mfma_MN))*4, mod=f"offset:{lds_weights}")

    voff_vmem_row = J.gpr(num_loads, "vu32")
    for i in range(num_loads):
        J.ds_read_b32(voff_vmem_row[i], row * 4, mod=f"offset:{lds_token_ids + i * num_rows_per_load * 4}")

    J.s_waitcnt(mod="lgkmcnt(0)")

    lds = J.alloc_lds((num_mfma_m * mfma_MN) * (lds_stride))

    for m in range(num_mfma_m):
        v_weights[m,1] = v_weights[m,0]

    for i in range(num_loads):
        voff_vmem[i] = voff_vmem_row[i] * vmem_stride + col * (vmem_lane_size)

    temp_c = J.gpr(num_loads, vmem_lane_size//sizeof_DW, "vbf16x2")

    def loop_body(ni):
        J.s_waitcnt(mod=f"vmcnt({num_loads})")

        mfma1 = mfma_generator((ni+1)&1)
        B_loader = loadB_generator(ni&1)

        #cvt_f32_to_pk_bf16(n&1)
        index = ni&1
        for m in range(num_mfma_m):
            emit_bload([B_loader], 1)
            for n in range(num_mfma_n):
                J.v_pk_mul_f32(C[index,m,n,0:1], C[index,m,n,0:1], v_weights[m])
                J.v_pk_mul_f32(C[index,m,n,2:3], C[index,m,n,2:3], v_weights[m])
                J.v_add_u32(C[index,m,n,0], C[index,m,n,0], s_cvt_bf16_bias)
                J.v_add_u32(C[index,m,n,1], C[index,m,n,1], s_cvt_bf16_bias)
                J.v_add_u32(C[index,m,n,2], C[index,m,n,2], s_cvt_bf16_bias)
                J.v_add_u32(C[index,m,n,3], C[index,m,n,3], s_cvt_bf16_bias)
                J.pk_f32_to_bf16(C[index,m,n,0], C[index,m,n,0], C[index,m,n,1])
                J.pk_f32_to_bf16(C[index,m,n,1], C[index,m,n,2], C[index,m,n,3])
            #emit_mfma([mfma1], 16)

        # ds_write_C(ni&1)
        index = ni & 1
        for m in range(num_mfma_m):
            emit_bload([B_loader], 1)
            for n in range(num_mfma_n):
                offset = lds + m*mfma_MN*lds_stride + n*(4*mfma_MN*sizeof_bf16)
                J.ds_write_b64(voff_c_lds_w, C[index, m, n, 0:1], mod=f"offset:{offset}")
                emit_mfma([mfma1], 16)
        emit_mfma([mfma1], 32)
        J.s_waitcnt(mod=f"lgkmcnt(0)")
        J.s_barrier()

        #ds_load_C()
        for i in range(num_loads):
            offset = lds + i * num_rows_per_load * lds_stride
            if vmem_lane_size == sizeof_DW4:
                J.ds_read_b128(temp_c[i], voff_c_lds_r, mod=f"offset:{offset}")
            else:
                assert vmem_lane_size == sizeof_DW
                J.ds_read_b32(temp_c[i], voff_c_lds_r, mod=f"offset:{offset}")
            emit_mfma([mfma1], 32)

        emit_bload([B_loader])

        #atomic_pk_add_bf16()
        for i in range(num_loads):
            J.s_waitcnt(mod=f"lgkmcnt({min(15,num_loads - i - 1)})")
            if vmem_lane_size == sizeof_DW4:
                J.global_store_dwordx4(voff_vmem[i], temp_c[i], pC)      # this is fast:  (48us)
            else:
                assert vmem_lane_size == sizeof_DW
                # the bigger the M is, the bigger the perf-diff is
                with J.ExecMask(voff_vmem_row[i] < M[0], early_skip=False):
                    J.global_atomic_pk_add_bf16(voff_vmem[i], temp_c[i], pC) # this is much slower than directly store      (60us)
            emit_mfma([mfma1], 32)

        emit_mfma([mfma1])
        pC[:] += (4 * num_mfma_n * mfma_MN * sizeof_bf16) 

    loop_i = J.gpr("su32")
    loop_i[0] = 0
    loop_cnt = J.div(N, 4 * num_mfma_n * mfma_MN)
    J.s_waitcnt(mod=f"vmcnt(0)")
    with J.While(loop_i[0] < (loop_cnt//2)):
        loop_body(0)
        loop_body(1)
        loop_i[0] = loop_i[0] + 1
    if loop_cnt % 2:
        loop_body(0)

@jit(with_debug_log=False)
def moe_gemm_stage1(J:JIT,
                    weight_dtype,
                    TOPK,
                    K1,            # gate/up
                    N1,            # N for gate/up, K for down
                    N2,            # down
                    BLOCK_TILE_SIZE_M,
                    BLOCK_TILE_SIZE_N,
                    p_input:"void*",
                    p_w_up:"void*",  # up+gate
                    p_w_up_scale:"float*",
                    p_w_down:"void*",
                    p_w_down_scale:"float*",
                    p_output:"void*",
                    # sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids
                    p_sorted_ids:"void*",
                    p_sorted_weights:"float*",
                    p_sorted_expert_ids:"void*",
                    p_num_valid_ids:"void*",
                    M:"int",
                    ):
    #assert K % BLOCK_SIZE_K == 0
    assert BLOCK_TILE_SIZE_M % 16 == 0
    # to get 4*8bf16(aka dword16) for gemm2, the gemm1 output must be multiple of 64, after gate*up the width will be 32
    assert BLOCK_TILE_SIZE_N % 64 == 0
    K2 = N1//2

    sizeof_bf16 = 2
    sizeof_f32 = 4
    sizeof_w = sizeof_bf16 if weight_dtype == torch.bfloat16 else 1
    stride_A1 = K1 * sizeof_bf16

    A_vert = BLOCK_TILE_SIZE_M // 16
    B_horz = BLOCK_TILE_SIZE_N // 16

    C_reg = J.gpr(B_horz, A_vert, 4, "vf32")

    C_reg[:] = 0

    # grid: N // 32, sorted_expert_ids.shape[0]
    # expert index in p_sorted_expert_ids
    e_idx = J.blockIdx.y
    s_e_id = J.gpr(1, 'su32')
    J.s_load_dword(s_e_id, p_sorted_expert_ids, e_idx[0] * 4)
    max_id = J.gpr(1, 'su32')
    J.s_load_dword(max_id, p_num_valid_ids, 0)
    J.s_waitcnt(mod=f"lgkmcnt(0)")
    # invalid padding section
    J.Jump("continue_following", e_idx * BLOCK_TILE_SIZE_M < max_id)
    J.s_endpgm()
    J.Label("continue_following")

    # hide following initialization into s_waitcnt
    p_sorted_ids[:] += e_idx * (BLOCK_TILE_SIZE_M * 4)
    p_sorted_weights[:] += e_idx * (BLOCK_TILE_SIZE_M * 4)

    # one WG per CU,  4 waves split on K
    lane_mod_16 = get_lane_id_mod(J, 16)
    lane_div_16 = get_lane_id_div(J, 16)
    warp_id = J.gpr("su32")
    J.v_readfirstlane_b32(warp_id, J.threadIdx.x[0] // 64)

    lds_buff = J.LDSTensor([4 * BLOCK_TILE_SIZE_M, 64], torch.float32)
    lds_buff_out = lds_buff.view_as([BLOCK_TILE_SIZE_M, 32 // 2], torch.float32)

    v_sorted_id = J.gpr(A_vert, 'vu32')
    J.global_load_dword(v_sorted_id[0], lane_mod_16 << 2, p_sorted_ids)
    for n in range(1, A_vert):
        p_sorted_ids[:] += 16 * 4
        J.global_load_dword(v_sorted_id[n], lane_mod_16 << 2, p_sorted_ids)

    # load 256 weights
    v_sorted_weights = J.gpr('vf32')
    assert BLOCK_TILE_SIZE_M <= 256
    with J.ExecMask(J.threadIdx.x < BLOCK_TILE_SIZE_M):
        J.global_load_dword(v_sorted_weights, J.threadIdx.x * 4, p_sorted_weights)

    voffset_b = J.gpr(B_horz, 'vu32')
    voffset_b[0] = J.threadIdx.x * 16
    voffset_b[B_horz // 2] = voffset_b[0] + (N1 // 2 * sizeof_w) * K1
    for m in range(1, B_horz // 2):
        voffset_b[m] = voffset_b[0] + K1 * (16 * sizeof_w * m)
        voffset_b[B_horz // 2 + m] = voffset_b[B_horz // 2] + K1 * (16 * sizeof_w * m)

    buff_a = J.Buffer(p_input, M * (K1 * sizeof_bf16))

    # wait for v_sorted_id
    J.s_waitcnt(mod=f"vmcnt(0)")

    voffset_a1 = J.gpr(A_vert, 'vu32')
    v_topk_id = J.gpr(A_vert, 'vu32')
    v_token_id = J.gpr(A_vert, 'vu32')
    # 16 elements for a if fp8
    a_element_num_per_thread = 8 if weight_dtype == torch.bfloat16 else 16
    for m in range(A_vert):
        v_token_id[m] = v_sorted_id[m] & 0xffffff
        voffset_a1[m] = J.gpr(v_token_id[m] * stride_A1 + (J.threadIdx.x // 16) * (a_element_num_per_thread * sizeof_bf16))
        v_topk_id[m] = v_sorted_id[m] >> 24
        # # input layout: [B, TOPK, INTER_MEDIA]
        #voffset_a2[m] = J.gpr(v_token_id[m] * TOPK * K * sizeof_bf16 + v_topk_id[m] * (K * sizeof_bf16) + (J.threadIdx.x // 16) * (a_element_num_per_thread * sizeof_bf16))
    voffset_scale = J.gpr(B_horz, 'vu32')
    if weight_dtype != torch.bfloat16:
        voffset_scale[0] = J.gpr(s_e_id * (N1 * sizeof_f32)) + lane_mod_16 * sizeof_f32
        voffset_scale[B_horz // 2] = voffset_scale[0] + N1 // 2 * sizeof_f32
        for m in range(1, B_horz // 2):
            voffset_scale[m] = voffset_scale[0] + 16 * sizeof_f32 * m
            voffset_scale[B_horz // 2 + m] = voffset_scale[B_horz // 2] + 16 * sizeof_f32 * m

    p_w_up[:] += s_e_id * (N1 * K1 * sizeof_w)
    p_w_down[:] += s_e_id * (N2 * K2 * sizeof_w)

    buff_b = J.Buffer(p_w_up, K1 * N1 * sizeof_w)

    s_cvt_bf16_bias = J.gpr(1, "su32")
    s_cvt_bf16_bias[0] = 0x00008000
    # split K
    # each 32 N as a group to compute gate*up
    n_groups = BLOCK_TILE_SIZE_N // 32
    n_half0 = 0
    n_half1 = B_horz // 2
    # each wave reduce 4 rows once
    vrow = J.gpr(lane_div_16 + warp_id * 4)

    A_reg_gemm2 = J.gpr(A_vert, (N1 // BLOCK_TILE_SIZE_N) * (n_groups // 2), 4, "abf16x2")
    for block_n in range(N1 // BLOCK_TILE_SIZE_N):
        if block_n != 0:
            C_reg[:] = 0
        moe_gemm_loop(J, weight_dtype, K1, N1, 4,
                    buff_a, buff_b, p_w_up_scale,
                    voffset_a1, voffset_b, voffset_scale, C_reg, soffset_a=0, soffset_b=block_n * (BLOCK_TILE_SIZE_N // 2 * K1 * sizeof_w), BLOCK_TILE_SIZE_N=BLOCK_TILE_SIZE_N, BLOCK_TILE_SIZE_M=BLOCK_TILE_SIZE_M)
        # each wave holds [4, 16, A_vert] floats
        vouts = J.gpr(A_vert, "vf32")
        for n in range(n_groups // 2):
            vaddr = J.gpr("vu32")
            for m in range(A_vert):
                lds_buff.write("b128", C_reg[n * 2 + 0 + n_half0, m], lane_mod_16 + warp_id * 16 + m * 64, lane_div_16 * 4)
                lds_buff.write("b128", C_reg[n * 2 + 1 + n_half0, m], lane_mod_16 + warp_id * 16 + m * 64, lane_div_16 * 4 + 16)
                lds_buff.write("b128", C_reg[n * 2 + 0 + n_half1, m], lane_mod_16 + warp_id * 16 + m * 64, lane_div_16 * 4 + 32)
                lds_buff.write("b128", C_reg[n * 2 + 1 + n_half1, m], lane_mod_16 + warp_id * 16 + m * 64, lane_div_16 * 4 + 48)

            J.s_waitcnt(mod=f"lgkmcnt(0)")
            J.s_barrier()
            v_sorted_id_permute = J.gpr("vu32")
            for m in range(A_vert):
                #J.ds_bpermute_b32(v_sorted_id_permute, vrow * 4, v_sorted_id[m]) 
                gate_up = J.gpr(4, 2, 2, "vf32")
                # interleave 
                lds_buff.read("b64", gate_up[0], vrow + (0 * 16 + m * 64), lane_mod_16 * 2, offset1 = 16)
                lds_buff.read("b64", gate_up[1], vrow + (1 * 16 + m * 64), lane_mod_16 * 2, offset1 = 16)
                lds_buff.read("b64", gate_up[2], vrow + (2 * 16 + m * 64), lane_mod_16 * 2, offset1 = 16)
                lds_buff.read("b64", gate_up[3], vrow + (3 * 16 + m * 64), lane_mod_16 * 2, offset1 = 16)

                J.s_waitcnt(mod=f"lgkmcnt(2)")
                J.v_pk_add_f32(gate_up[0, 0], gate_up[0, 0], gate_up[1, 0])
                J.v_pk_add_f32(gate_up[0, 1], gate_up[0, 1], gate_up[1, 1])
                J.s_waitcnt(mod=f"lgkmcnt(0)")
                J.v_pk_add_f32(gate_up[2, 0], gate_up[2, 0], gate_up[3, 0])
                J.v_pk_add_f32(gate_up[2, 1], gate_up[2, 1], gate_up[3, 1])
                J.v_pk_add_f32(gate_up[0, 0], gate_up[0, 0], gate_up[2, 0])
                J.v_pk_add_f32(gate_up[0, 1], gate_up[0, 1], gate_up[2, 1])

                out0 = J.gpr(gate_up[0, 1, 0] * J.silu(gate_up[0, 0, 0]))
                out1 = J.gpr(gate_up[0, 1, 1] * J.silu(gate_up[0, 0, 1]))
                J.v_add_u32(out0[0], out0[0], s_cvt_bf16_bias)
                J.v_add_u32(out1[0], out1[0], s_cvt_bf16_bias)
                J.pk_f32_to_bf16(vouts[m], out0[0], out1[0])
                # to remove, test only
                # output: [B, TOPK, N]
                if 0:
                    v_token_id = J.gpr(v_sorted_id_permute[0] & 0xffffff)
                    v_topk_id = J.gpr(v_sorted_id_permute[0] >> 24)
                    vaddr = J.gpr(v_token_id * (N1 // 2 * sizeof_bf16 * TOPK) + v_topk_id * (N1 // 2 * sizeof_bf16) + lane_mod_16 * sizeof_f32 + n * 16 * sizeof_f32 + block_n * (BLOCK_TILE_SIZE_N // 2) * sizeof_bf16)
                    with J.ExecMask(v_token_id < M[0]):
                        J.global_store_dword(vaddr, vouts[m], p_output)

            # be sure read done
            #    lds_buff_out: [BLOCK_TILE_SIZE_M, 32 // 2], torch.float32)
            J.s_barrier()
            for m in range(A_vert):
                lds_buff_out.write("b32", vouts[m],
                                   lane_div_16 + warp_id * 4 + m * 16, lane_mod_16)

            J.s_waitcnt(mod=f"lgkmcnt(0)")
            J.s_barrier()

            for m in range(A_vert):
                lds_buff_out.read("b128", A_reg_gemm2[m, block_n * (n_groups // 2) + n],
                                   lane_mod_16 + m * 16, lane_div_16 * 4)

            J.s_waitcnt(mod=f"lgkmcnt(0)")
            J.s_barrier()

        if weight_dtype != torch.bfloat16:
            p_w_up_scale[:] += BLOCK_TILE_SIZE_N // 2 * sizeof_f32

    # be sure A_reg_gemm2 ready
    J.s_waitcnt(mod=f"lgkmcnt(0)")

    # release LDS buffer 
    lds_buff.free()
    lds_weights = J.alloc_lds(256*4)
    lds_token_ids = J.alloc_lds(256*4)

    with J.ExecMask(J.threadIdx.x < BLOCK_TILE_SIZE_M):
        J.ds_write_b32(J.threadIdx.x * 4, v_sorted_weights, mod=f"offset:{lds_weights}")

    with J.ExecMask(J.threadIdx.x < 16):
        for m in range(A_vert):
            J.ds_write_b32(J.lane_id * 4, v_sorted_id[m] & 0xffffff, mod=f"offset:{lds_token_ids + m*16*4}")
    J.s_waitcnt(mod=f"lgkmcnt(0)")
    J.s_barrier()
    # gemm2
    num_mfma_n = 2
    down_kernel(J, 16,
                num_mfma_n, BLOCK_TILE_SIZE_M, N2, K2,
                A_reg_gemm2, # A = J.gpr(num_mfma_m, num_mfma_k, 4, "abf16x2")
                lds_token_ids,  # 256 int32 token_id
                lds_weights,    # 256 fp32 weights 
                p_w_down,
                p_output,
                M)


#####################################################################
def _run_aiter(hidden_states,
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

def _run_batch(kernel_type, B=1, weight_type=torch.bfloat16, TILE_M=16, TILE_N=32, run_count=10):
    HIDDEN_SIZE = 2048
    TP = 4
    INTER_SIZE = 768
    TP = 8
    INTER_SIZE = 1024
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
    # make a B*TOPK seq, which contains 0...E-1 0...E-1
    rep_e = div_up(B * TOPK, E)
    topk_ids_1d = torch.ones([rep_e, E], dtype=torch.int32)
    topk_ids_1d[:, ] = torch.randperm(E, dtype=torch.int32)
    topk_ids[:, ] = topk_ids_1d.reshape(-1)[ : B * TOPK].reshape(B, TOPK)
    access_expert = torch.unique(topk_ids[0])
    access_expert = access_expert.shape[0]

    flops = 2 * B * TOPK * (HIDDEN_SIZE * INTER_SIZE_TP * 2 + HIDDEN_SIZE * INTER_SIZE_TP)
    mem_size = B * HIDDEN_SIZE * 2 + (HIDDEN_SIZE * INTER_SIZE_TP * 2 + HIDDEN_SIZE * INTER_SIZE_TP) * access_expert * (2 if weight_type == torch.bfloat16 else 1)

    import aiter
    from aiter.ops.shuffle import shuffle_weight
    from aiter.fused_moe import moe_sorting

    def run(hidden_states, w1, w2, topk_weight, topk_ids, w1_scale, w2_scale):
        B = hidden_states.shape[0]
        E, N1, K1 = w1.shape
        N2, K2 = w2.shape[1], w2.shape[2]
        gemm1_out = torch.empty([B, TOPK, N1 // 2], dtype=hidden_states.dtype, device=hidden_states.device)
        if kernel_type == '16x32_2s_b1':
            # test moe_gemm_batch: 2 stages, BLOCK_TILE_M=16, BLOCK_TILE_N=32, batch == 1
            cur_out = torch.zeros([1, N2], dtype=hidden_states.dtype, device=hidden_states.device)
            moe_gemm_batch1([N1 // 32, TOPK],[256], w1.dtype, True, hidden_states.data_ptr(), w1.data_ptr(), gemm1_out.data_ptr(), topk_ids.data_ptr(), topk_weight.data_ptr(), w1_scale.data_ptr() if w1_scale is not None else 0, 1, N1, K1)
            moe_gemm_batch1([N2 // 32, TOPK],[64], w1.dtype, False, gemm1_out.data_ptr(), w2.data_ptr(), cur_out.data_ptr(), topk_ids.data_ptr(), topk_weight.data_ptr(), w2_scale.data_ptr() if w2_scale is not None else 0, 1, N2, K2)
        elif kernel_type == '16x32_2s_b':
            # test moe_gemm_batch: 2 stages, BLOCK_TILE_M=16, BLOCK_TILE_N=32
            sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, cur_out = moe_sorting(
                topk_ids,
                topk_weight,
                E,
                K1,     # reduce dim is same with output dim
                hidden_states.dtype,
                16,
                None,
                None,
                0,
            )
            moe_gemm_batch([N1 // 32, sorted_expert_ids.shape[0]], [256],
                            w1.dtype, True,
                            hidden_states.data_ptr(), w1.data_ptr(), gemm1_out.data_ptr(), sorted_ids.data_ptr(), sorted_weights.data_ptr(), sorted_expert_ids.data_ptr(), num_valid_ids.data_ptr(), w1_scale.data_ptr() if w1_scale is not None else 0, B, N1, K1, TOPK)
            moe_gemm_batch([N2 // 32, sorted_expert_ids.shape[0]], [64],
                            w1.dtype, False,
                            gemm1_out.data_ptr(), w2.data_ptr(), cur_out.data_ptr(), sorted_ids.data_ptr(), sorted_weights.data_ptr(), sorted_expert_ids.data_ptr(), num_valid_ids.data_ptr(), w2_scale.data_ptr() if w2_scale is not None else 0, B, N2, K2, TOPK)
        elif kernel_type == 'mxn_splitk_2s':
            # test moe_gemm_batch_vmn: 2 stages, m/n can be set
            sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, cur_out = moe_sorting(
                topk_ids,
                topk_weight,
                E,
                K1,     # reduce dim is same with output dim
                hidden_states.dtype,
                TILE_M,
                None,
                None,
                0,
            )
            BLOCK_TILE_SIZE_M = TILE_M
            BLOCK_TILE_SIZE_N = TILE_N
            moe_gemm_batch_vmn([N1 // BLOCK_TILE_SIZE_N, sorted_expert_ids.shape[0]], [256],
                                w1.dtype, TOPK, K1, N1, True, BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N,
                                hidden_states.data_ptr(), w1.data_ptr(), gemm1_out.data_ptr(), sorted_ids.data_ptr(), sorted_weights.data_ptr(), sorted_expert_ids.data_ptr(), num_valid_ids.data_ptr(), w1_scale.data_ptr() if w1_scale is not None else 0, B)
            moe_gemm_batch_vmn([N2 // BLOCK_TILE_SIZE_N, sorted_expert_ids.shape[0]], [64],
                                w1.dtype, TOPK, K2, N2, False, BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N,
                                gemm1_out.data_ptr(), w2.data_ptr(), cur_out.data_ptr(), sorted_ids.data_ptr(), sorted_weights.data_ptr(), sorted_expert_ids.data_ptr(), num_valid_ids.data_ptr(), w2_scale.data_ptr() if w2_scale is not None else 0, B)
        elif kernel_type == 'mxn_splitk_1s':
            # test moe_gemm_stage1
            sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, cur_out = moe_sorting(
                topk_ids,
                topk_weight,
                E,
                K1,     # reduce dim is same with output dim
                hidden_states.dtype,
                TILE_M,
                None,
                None,
                0,
            )
            BLOCK_TILE_SIZE_M = TILE_M
            BLOCK_TILE_SIZE_N = TILE_N
            moe_gemm_stage1([1, sorted_expert_ids.shape[0]], [256], 
                            w1.dtype, TOPK, K1, N1, N2, BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N,
                            hidden_states.data_ptr(), w1.data_ptr(), w1_scale.data_ptr() if w1_scale is not None else 0, w2.data_ptr(), w2_scale.data_ptr() if w2_scale is not None else 0,
                            cur_out.data_ptr(), 
                            sorted_ids.data_ptr(), sorted_weights.data_ptr(), sorted_expert_ids.data_ptr(), num_valid_ids.data_ptr(), B)
        else:
            assert 0, f'not support kernel type "{kernel_type}"'
        return cur_out

    tflops_res = []
    latencies = []
    bw = []
    if kernel_type == 'aiter':
        # aiter needs preshuffle weights
        i = 0
        for _ in range(run_count):
            with cudaPerf(flops, mem_size, name=f"{kernel_type}[{B=},{str(weight_type).split('.')[1]}]") as p:
                _run_aiter(hidden_states=hidden_states[i], w1=w1[i], w2=w2[i], topk_weight=topk_weight[i], topk_ids=topk_ids[i], w1_scale=w1_scale[i], w2_scale=w2_scale[i])
            i = (i + 1) % BUF_COPY
            tflops_res.append(p.tflops())
            latencies.append(p.dt())
            bw.append(p.bw())
    else:
        w1_qt_aiter = shuffle_weight(w1[0], layout=(16, 16))
        w2_qt_aiter = shuffle_weight(w2[0], layout=(16, 16))
        ref_out = _run_aiter(hidden_states=hidden_states[0], w1=w1_qt_aiter, w2=w2_qt_aiter, topk_weight=topk_weight[0], topk_ids=topk_ids[0], w1_scale=w1_scale[0], w2_scale=w2_scale[0])
        cur_out = run(hidden_states=hidden_states[0], w1=w1_qt_aiter, w2=w2_qt_aiter, topk_weight=topk_weight[0], topk_ids=topk_ids[0], w1_scale=w1_scale[0], w2_scale=w2_scale[0])
        i = 0
        for _ in range(run_count):
            with cudaPerf(flops, mem_size, name=f"{kernel_type}[{B=},{str(weight_type).split('.')[1]}]") as p:
                run(hidden_states=hidden_states[i], w1=w1[i], w2=w2[i], topk_weight=topk_weight[i], topk_ids=topk_ids[i], w1_scale=w1_scale[i], w2_scale=w2_scale[i])
            i = (i + 1) % BUF_COPY
            tflops_res.append(p.tflops())
            latencies.append(p.dt())
            bw.append(p.bw())

        if not torch.allclose(ref_out, cur_out, rtol=0.1, atol=0.03):
            print(cur_out)
            idx = torch.where(torch.abs(ref_out - cur_out) > 0.03)
            if len(idx[0]):
                print(f'idx = {idx}\nref={ref_out[idx]}\ncur={cur_out[idx]}\n{len(idx[0])}')
            assert 0
        else:
            print(f"{kernel_type}[{B=} {weight_type=}] acc OK")
    if run_count > 0:
        return {'flops': sum(tflops_res[1:])/len(tflops_res[1:]),              # tflops
                'latency': sum(latencies[1:])/len(latencies[1:]) * 1e6,        # us
                'bw': sum(bw[1:]) / len(bw[1:])}                               # GB/s

# special path for batch1 
def entry_b1(test_fp8=True, run_count=10):
    kernel_type = '16x32_2s_b1'
    perf = {}
    perf[kernel_type] = {}
    perf_bf16 = {}

    perf_bf16[1] = _run_batch(kernel_type, B=1, weight_type=torch.bfloat16, run_count=run_count)
    perf[kernel_type]['bf16'] = perf_bf16
    if test_fp8:
        perf_fp8 = {}
        perf_fp8[1] = _run_batch(kernel_type, B=1, weight_type=torch.float8_e4m3fnuz, run_count=run_count)
        perf[kernel_type]['fp8'] = perf_fp8
    return perf

def entry_common(kernel_type, batch, test_fp8=True, TILE_M=None, TILE_N=None, run_count=10):
    perf = {}
    perf[kernel_type] = {}
    perf_bf16 = {}
    for i in batch:
        perf_bf16[i] = _run_batch(kernel_type, B=i, weight_type=torch.bfloat16, TILE_M=TILE_M, TILE_N=TILE_N, run_count=run_count)
    perf[kernel_type]['bf16'] = perf_bf16

    if test_fp8:
        perf_fp8 = {}
        for i in batch: 
            perf_fp8[i] = _run_batch(kernel_type, B=i, weight_type=torch.float8_e4m3fnuz, TILE_M=TILE_M, TILE_N=TILE_N, run_count=run_count)
        perf[kernel_type]['fp8'] = perf_fp8
    
    return perf

def init_env():
    torch.set_printoptions(linewidth=3000, sci_mode=False, edgeitems=8, )
    torch.set_default_device('cuda')
    torch.manual_seed(0)

def test_acc():
    init_env()
    batch = list(range(2, 64))
    # fix TILE_M=16, TILE_N=32
    entry_b1(run_count=0)
    entry_common('16x32_2s_b', batch=batch, test_fp8=True, run_count=0)
    batch += list(range(128, 256))
    batch += [i * 256 for i in range(1, 4)]
    batch += [i * 2048 for i in range(1, 5)]
    # TODO: mxn_splitk_1s may fail on the case:
    #batch += list(range(2048 * 3, 2048 * 3 + 256))
    # TILE_M/N is configurable
    entry_common('mxn_splitk_2s', batch=batch, test_fp8=True, TILE_M=32, TILE_N=64, run_count=0)
    # TODO: support fp8
    entry_common('mxn_splitk_1s', batch=batch, test_fp8=False, TILE_M=32, TILE_N=128, run_count=0)

def show_perf(perf):
    print('\nsummary:')
    for kernel, vals in perf.items():
        for prec, vals_ in vals.items():
            for b, data in vals_.items():
                print(f'{kernel}[{prec:<4} B={b:<4}]: {data["latency"]:5.0f} us, {data["bw"]:6.1f} GB/s, {data["flops"]:4.1f} tflops')

@pytest.mark.parametrize("batch", [[1, 2, 4, 8, 12, 16, 32, 64]])
def test_small_batch_perf(batch):
    init_env()
    perf = {}
    perf.update(entry_common('aiter', batch))
    # fix TILE_M=16, TILE_N=32
    perf.update(entry_b1())           # batch 1
    perf.update(entry_common('16x32_2s_b', batch=batch, test_fp8=True))
    show_perf(perf)

@pytest.mark.parametrize("batch", [[16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]])
def test_perf(batch):
    init_env()
    perf = {}
    test_fp8 = False
    perf.update(entry_common('aiter', batch, test_fp8=test_fp8))
    # TODO: support fp8
    perf.update(entry_common('mxn_splitk_1s', batch=batch, test_fp8=False, TILE_M=32, TILE_N=128))
    # TILE_M/N is configurable
    perf.update(entry_common('mxn_splitk_2s', batch=batch, test_fp8=test_fp8, TILE_M=32, TILE_N=64))
    show_perf(perf)

if __name__ == '__main__':
    test_acc()
    batch = [1, 2, 4, 8, 12, 16, 32, 64]
    test_small_batch_perf(batch)
    batch = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    test_perf(batch)