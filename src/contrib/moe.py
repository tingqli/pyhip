import os
import random
from pyhip import jit, JIT
import torch

from functools import cache
from .common.gemm import UGEMM
from .common.gemm_splitk import gemm_splitk

USE_FP4_SHUFFLE_WEIGHT = 1

__all__ = [
    "moe_gemm_batch1",
    "moe_gemm_batch",
    "moe_2stage_splitk",
    "moe_2stage_gateup",
    "moe_2stage_down",
    "moe_1stage_splitk",
]


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

    gemm_splitk(J, weight_dtype, K, N, num_split_k,
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

    gemm_splitk(J, weight_dtype, K, N, num_split_k,
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

@jit()
def moe_2stage_splitk(J:JIT,
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
    assert weight_dtype == torch.bfloat16 or weight_dtype == torch.float4_e2m1fn_x2 or weight_dtype == torch.float8_e4m3fn or weight_dtype == torch.float8_e4m3fnuz
    if weight_dtype == torch.float4_e2m1fn_x2:
        if with_silu:
            assert BLOCK_TILE_SIZE_N % 64 == 0, f'due to scale is packed with [2*16, 8*32], gate/up each needs 2*16 rows; current BLOCK_TILE_SIZE_N={BLOCK_TILE_SIZE_N} is not supported'
            assert K % 1024 == 0, f'will read (16*4)*4(wave) bytes once in main loop, aka 256*2=512 elements; to use packed scale in K dimension, will double read; current K={K} is not supported'
        else:
            assert BLOCK_TILE_SIZE_N % 32 == 0, f'due to scale is packed with [2*16, 8*32], current BLOCK_TILE_SIZE_N={BLOCK_TILE_SIZE_N} is not supported'
            assert K % 32 == 0, f'will read 16*4 bytes once in main loop, aka 64*2=128 elements; tails support K%32==0; current K={K} is not supported'
    elif weight_dtype == torch.float8_e4m3fn or weight_dtype == torch.float8_e4m3fnuz:
        assert K % 64 == 0, f'will read 16*4 bytes once in main loop, aka 64 elements; current K={K} is not supported'
    else:
        assert K % 32 == 0, f'will read 16*4 bytes once in main loop, aka 64/2=32 elements; current K={K} is not supported'
    assert BLOCK_TILE_SIZE_M % 16 == 0
    assert BLOCK_TILE_SIZE_N % 32 == 0
    assert N % BLOCK_TILE_SIZE_N == 0

    use_split_k = with_silu
    num_split_k = 4 if use_split_k else 1
    BLOCK_TILE_SIZE_N_HALF = BLOCK_TILE_SIZE_N // 2

    sizeof_bf16 = 2
    sizeof_f32 = 4
    def get_k_bytes(k_in_elements):
        if weight_dtype == torch.bfloat16:
            return k_in_elements * sizeof_bf16
        elif weight_dtype == torch.float4_e2m1fn_x2:
            return k_in_elements // 2
        elif weight_dtype == torch.float8_e4m3fn or weight_dtype == torch.float8_e4m3fnuz:
            return k_in_elements
    stride_A = K * sizeof_bf16
    stride_B = get_k_bytes(K)

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
        if weight_dtype == torch.float4_e2m1fn_x2:
            if USE_FP4_SHUFFLE_WEIGHT:
                # shffled
                voffset_b[0] = (BLOCK_TILE_SIZE_N_HALF * J.blockIdx.x) * stride_B + J.lane_id * 16 + J.warp_id * (16 * get_k_bytes(K) // 4)
            else:
                # not shuffled
                voffset_b[0] = (BLOCK_TILE_SIZE_N_HALF * J.blockIdx.x + lane_mod_16) * stride_B + lane_div_16 * 16 + J.warp_id * get_k_bytes(K // 4)
        else:
            # shffled
            voffset_b[0] = BLOCK_TILE_SIZE_N_HALF * J.blockIdx.x * stride_B + J.threadIdx.x * 16
        voffset_b[B_horz // 2] = voffset_b[0] + (N // 2) * get_k_bytes(K)
        for m in range(1, B_horz // 2):
            voffset_b[m] = voffset_b[0] + get_k_bytes(K) * 16 * m
            voffset_b[B_horz // 2 + m] = voffset_b[B_horz // 2] + get_k_bytes(K) * 16 * m
    else:
        p_sorted_weights[:] += e_idx * (BLOCK_TILE_SIZE_M * 4)
        v_weight = J.gpr(A_vert, 'vf32')
        if weight_dtype == torch.float4_e2m1fn_x2:
            if USE_FP4_SHUFFLE_WEIGHT:
                # shffled
                voffset_b[0] = (BLOCK_TILE_SIZE_N * J.blockIdx.x) * stride_B + J.lane_id * 16
            else:
                # not shuffled
                voffset_b[0] = (BLOCK_TILE_SIZE_N * J.blockIdx.x + lane_mod_16) * stride_B + lane_div_16 * 16
        else:
            # shffled
            voffset_b[0] = BLOCK_TILE_SIZE_N * J.blockIdx.x * stride_B + J.threadIdx.x * 16
        J.global_load_dword(v_weight[0], lane_mod_16 << 2, p_sorted_weights)
        for m in range(1, B_horz):
            voffset_b[m] = voffset_b[0] + get_k_bytes(K) * 16 * m
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
    # a elements number:
    # bf16: 8 elements
    # fp8: 16 elements
    # fp4: 32 elements
    if weight_dtype == torch.bfloat16:
        a_element_num_per_thread = 8
    elif weight_dtype == torch.float4_e2m1fn_x2:
        a_element_num_per_thread = 32
    elif weight_dtype == torch.float8_e4m3fn or weight_dtype == torch.float8_e4m3fnuz:
        a_element_num_per_thread = 16
    else:
        raise ValueError(f"Unsupported weight dtype: {weight_dtype}")
    for m in range(A_vert):
        v_token_id[m] = v_sorted_id[m] & 0xffffff
        if with_silu:
            if weight_dtype != torch.float4_e2m1fn_x2:
                voffset_a[m] = J.gpr(v_token_id[m] * stride_A + (J.threadIdx.x // 16) * (a_element_num_per_thread * sizeof_bf16))
            else:
                voffset_a[m] = J.gpr(v_token_id[m] * stride_A + (lane_div_16) * (a_element_num_per_thread * sizeof_bf16) + J.warp_id * (K // 4 * sizeof_bf16))
        else:
            v_topk_id[m] = v_sorted_id[m] >> 24
            # input layout: [B, TOPK, INTER_MEDIA]
            voffset_a[m] = J.gpr(v_token_id[m] * TOPK * K * sizeof_bf16 + v_topk_id[m] * (K * sizeof_bf16) + (J.threadIdx.x // 16) * (a_element_num_per_thread * sizeof_bf16))

    voffset_scale = None
    if weight_dtype == torch.float4_e2m1fn_x2:
        # 32 rows shared in a uint32
        voffset_scale = J.gpr(B_horz // 2, 'vu32')
        # the group of fp4 is 32 elements, and scale will align to 8 groups
        k_scale_stride = div_up(div_up(K, 32), 8) * 8
        p_w_scale[:] += (BLOCK_TILE_SIZE_N_HALF if with_silu else BLOCK_TILE_SIZE_N) * k_scale_stride * J.blockIdx.x
        if with_silu:
            voffset_scale[0] = J.gpr(s_e_id * (N * k_scale_stride)) + J.lane_id * sizeof_f32 + J.warp_id * (k_scale_stride // 8 // 4 * 64 * sizeof_f32)
            voffset_scale[B_horz // 4] = voffset_scale[0] + N // 2 * k_scale_stride
            for m in range(1, B_horz // 4):
                voffset_scale[m] = voffset_scale[0] + 32 * k_scale_stride * m
                voffset_scale[B_horz // 4 + m] = voffset_scale[B_horz // 4] + 32 * k_scale_stride * m
        else:
            voffset_scale[0] = J.gpr(s_e_id * (N * k_scale_stride)) + J.threadIdx.x * sizeof_f32
            for m in range(1, B_horz // 2):
                voffset_scale[m] = voffset_scale[0] + 32 * k_scale_stride * m
    elif weight_dtype == torch.float8_e4m3fn or weight_dtype == torch.float8_e4m3fnuz:
        voffset_scale = J.gpr(B_horz, 'vu32')
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

    p_weight[:] = p_weight[:] + s_e_id * (N * get_k_bytes(K))
    buff_b = J.Buffer(p_weight, N * get_k_bytes(K))

    gemm_splitk(J, weight_dtype, K, N, num_split_k,
                buff_a, buff_b, p_w_scale,
                voffset_a, voffset_b, voffset_scale, C_reg, BLOCK_TILE_SIZE_N=BLOCK_TILE_SIZE_N, BLOCK_TILE_SIZE_M=BLOCK_TILE_SIZE_M, USE_FP4_SHUFFLE_WEIGHT=USE_FP4_SHUFFLE_WEIGHT)

    s_cvt_bf16_bias = J.gpr(1, "su32")
    s_cvt_bf16_bias[0] = 0x00008000

    ###############################################
    if with_silu and BLOCK_TILE_SIZE_N % 64 == 0 and"gfx950" in J.arch:
        # use more lds
        lds_buff = J.LDSTensor([4 * BLOCK_TILE_SIZE_M, 64], torch.float32)
        # each 32 N as a group to compute gate*up
        n_groups = BLOCK_TILE_SIZE_N // 32
        n_half0 = 0
        n_half1 = B_horz // 2
        # each wave holds [4, 16, A_vert] floats
        vouts = J.gpr(A_vert, "vf32")
        vrow = J.gpr(lane_div_16 + J.warp_id * 4)
        for n in range(n_groups // 2):
            vaddr = J.gpr("vu32")
            for m in range(A_vert):
                lds_buff.write("b128", C_reg[n * 2 + 0 + n_half0, m], lane_mod_16 + J.warp_id * 16 + m * 64, lane_div_16 * 4)
                lds_buff.write("b128", C_reg[n * 2 + 1 + n_half0, m], lane_mod_16 + J.warp_id * 16 + m * 64, lane_div_16 * 4 + 16)
                lds_buff.write("b128", C_reg[n * 2 + 0 + n_half1, m], lane_mod_16 + J.warp_id * 16 + m * 64, lane_div_16 * 4 + 32)
                lds_buff.write("b128", C_reg[n * 2 + 1 + n_half1, m], lane_mod_16 + J.warp_id * 16 + m * 64, lane_div_16 * 4 + 48)

            J.s_waitcnt(mod=f"lgkmcnt(0)")
            J.s_barrier()
            v_sorted_id_permute = J.gpr("vu32")
            for m in range(A_vert):
                J.ds_bpermute_b32(v_sorted_id_permute, vrow * 4, v_sorted_id[m]) 
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
                # output: [B, TOPK, N]
                v_token_id = J.gpr(v_sorted_id_permute[0] & 0xffffff)
                v_topk_id = J.gpr(v_sorted_id_permute[0] >> 24)
                vaddr = J.gpr(v_token_id * (N // 2 * J.sizeof_bf16 * TOPK) + v_topk_id * (N // 2 * J.sizeof_bf16) + lane_mod_16 * sizeof_f32 + n * 16 * sizeof_f32)
                with J.ExecMask(v_token_id < M[0], early_skip=False):
                    J.global_store_dword(vaddr, vouts[m], p_output)

            J.s_barrier()

        return
    ###############################################
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
                with J.ExecMask(v_token_id < M[0], early_skip=False):
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
                for j in range(4):
                    C_reg[n, m, j] = C_reg[n, m, j] * v_weight[m]
                    J.v_add_u32(C_reg[n, m, j], C_reg[n, m, j], s_cvt_bf16_bias)
                creg_low[0] = (C_reg[n, m, 0] >> 16) | (C_reg[n, m, 1] & 0xFFFF0000)
                creg_low[1] = (C_reg[n, m, 2] >> 16) | (C_reg[n, m, 3] & 0xFFFF0000)
                with J.ExecMask(v_token_id[m] < M[0], early_skip=False):
                    J.global_atomic_pk_add_bf16(vaddr    , creg_low[0], p_output)
                    J.global_atomic_pk_add_bf16(vaddr + 4, creg_low[1], p_output)

@jit()
def moe_2stage_gateup(J:JIT,
                      weight_dtype,
                      TOPK,
                      K,            # compile-time args
                      N,            # compile-time args
                      BLOCK_TILE_SIZE_M,
                      BLOCK_TILE_SIZE_N,
                      p_input:"void*",
                      p_weight:"void*",
                      p_output:"void*",
                      # sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids
                      p_sorted_ids:"void*",
                      p_sorted_expert_ids:"void*",
                      p_num_valid_ids:"void*",
                      p_w_scale:"float*",
                      M:"int",):
    class MFMA_DW4Loader:
        def __init__(self, J:JIT, ptr, buff_size, row_index,
                    wg_M:int, row_bytes:int, stride_bytes:int,
                    wave_cnt:int, swizzle_row_div:int,
                    skip_load:bool = False):
            self.buff = J.Buffer(ptr, buff_size) # wg_M*stride_bytes)
            self.wave_cnt = wave_cnt
            self.wg_M = wg_M
            self.J = J
            sizeof_DWORDX4 = 16
            assert (row_bytes) % sizeof_DWORDX4 == 0 # each lane prefetch DWORDx4 which is 8xhalf
            num_lanes_per_row = row_bytes // sizeof_DWORDX4
            assert 64 % num_lanes_per_row == 0
            dw4_prefetch_MN = (self.wave_cnt*64//num_lanes_per_row)
            assert dw4_prefetch_MN >= 1
            assert self.wg_M % dw4_prefetch_MN == 0
            # print(f"{self.wg_M=} {num_lanes_per_row=} {dw4_prefetch_MN=}")
            num_prefetch_M = self.wg_M // dw4_prefetch_MN
            self.prefetch_reg = J.gpr(num_prefetch_M, 4, "vu32")
            self.num_lanes_per_row = num_lanes_per_row
            self.prefetch_voffset = J.gpr(num_prefetch_M, "vu32")
            for i in range(num_prefetch_M):
                self.prefetch_voffset[i] = (J.threadIdx.x % num_lanes_per_row) * sizeof_DWORDX4 + row_index[i] * stride_bytes
            self.prefetch_soffset = J.gpr("su32")
            self.prefetch_step_size = (dw4_prefetch_MN)*stride_bytes
            self.num_prefetch_M = num_prefetch_M
            self.stride_bytes = stride_bytes
            self.swizzle_row_div = swizzle_row_div
            if skip_load:
                self.prefetch_step_size = 0

            # precompute ds_write vaddr
            self.ds_write_b128_vaddr = [] 
            for index in range(num_prefetch_M):
                vaddr = J.gpr("vu32")
                col = J.threadIdx.x % num_lanes_per_row
                row = (J.threadIdx.x // num_lanes_per_row) + index*dw4_prefetch_MN
                swizzle_col = ((row//swizzle_row_div) ^ col) % (num_lanes_per_row)
                vaddr[0] = J.gpr((row * row_bytes) + swizzle_col*(sizeof_DWORDX4))
                self.ds_write_b128_vaddr.append(vaddr) 

        def __len__(self):
            return self.num_prefetch_M

        def reset_offset(self, koff):
            self.prefetch_soffset[0] = koff[0]

        def prefetch(self, index):
            assert index < self.num_prefetch_M
            self.buff.load_dwordx4(self.prefetch_reg[index], self.prefetch_voffset[index], self.prefetch_soffset)

        def ds_write(self, index, lds_base):
            assert index < self.num_prefetch_M
            self.J.ds_write_b128(self.ds_write_b128_vaddr[index], self.prefetch_reg[index], mod=f"offset:{lds_base}") #  vaddr, vdata offset gds

    class MFMA_DW4Loader_preshuffled:
        def __init__(self, J:JIT, ptr, buff_size, mfma_MN,
                    wg_M:int, row_bytes:int, stride_bytes:int,
                    wave_cnt:int, swizzle_row_div:int,
                    skip_load:bool = False, N=0):
            self.buff = J.Buffer(ptr, buff_size)
            sizeof_DWORDX4 = 16
            assert row_bytes % sizeof_DWORDX4 == 0
            row_lanes = row_bytes // sizeof_DWORDX4
            mfma_K_lanes = 64 // mfma_MN
            assert row_lanes % mfma_K_lanes == 0, f"{row_lanes=} {mfma_K_lanes=}"
            assert wg_M % mfma_MN == 0
            num_prefetch_m = wg_M // mfma_MN
            num_prefetch_n = row_lanes // mfma_K_lanes
            assert (num_prefetch_m * num_prefetch_n) % wave_cnt == 0
            wave_prefetches = num_prefetch_m * num_prefetch_n // wave_cnt
            # big_waves = wave_cnt - (wave_prefetches * wave_cnt - num_prefetch_m * num_prefetch_n)
            num_lanes_per_row = row_bytes // sizeof_DWORDX4

            warp_id = J.warp_id
            lane_id = J.lane_id
            # at compile time, precompute offsets for waves and generate conditional assign
            self.ds_write_b128_vaddr = [J.gpr("vu32") for _ in range(wave_prefetches)]
            self.prefetch_reg = J.gpr(wave_prefetches, 4, "vu32")
            self.prefetch_sbase = J.gpr("su32")
            self.prefetch_offsets = J.gpr(wave_prefetches, "su32")
            self.prefetch_voffset = lane_id * sizeof_DWORDX4 + BLOCK_TILE_SIZE_N // 2 * stride_bytes * J.blockIdx.x
            prefetch_id = 0
            for wave in range(wave_cnt):
                with J.If(warp_id[0] == wave):
                    # pre-shuffled data unit size: [mfma_MN, mfma_K_lanes * sizeof_DWORDX4]
                    for i in range(wave_prefetches):
                        if wave < wave_cnt // 2:
                            # first half
                            prefetch_n = prefetch_id % num_prefetch_n
                            prefetch_m = prefetch_id // num_prefetch_n
                            offset = prefetch_m * (mfma_MN * stride_bytes) + prefetch_n * (mfma_MN * mfma_K_lanes * sizeof_DWORDX4)
                        else:
                            # second half
                            prefetch_n = prefetch_id % num_prefetch_n
                            prefetch_m = prefetch_id // num_prefetch_n - num_prefetch_m // 2
                            offset = N // 2 * stride_bytes + prefetch_m * (mfma_MN * stride_bytes) + prefetch_n * (mfma_MN * mfma_K_lanes * sizeof_DWORDX4)
                        assert prefetch_m < num_prefetch_m
                        prefetch_id += 1
                        self.prefetch_offsets[i] = offset

                        col = (lane_id // mfma_MN) + prefetch_n * mfma_K_lanes
                        # the expected reading layout for 2x2 wave with 128 rows: 
                        # wave0: [ 0:16] [N//2+ 0:N//2+16]
                        # wave1: [16:32] [N//2+16:N//2+32]
                        # wave2: [32:48] [N//2+32:N//2+48]
                        # wave3: [48:64] [N//2+48:N//2+64]
                        if wave < wave_cnt // 2:
                            # write to even position
                            row = (lane_id % mfma_MN) + prefetch_m * 2 * mfma_MN
                        else:
                            # write to odd position
                            row = (lane_id % mfma_MN) + (prefetch_m * 2 + 1) * mfma_MN
                        swizzle_col = ((row//swizzle_row_div) ^ col) % (num_lanes_per_row)
                        self.ds_write_b128_vaddr[i][0] = (row * row_bytes) + swizzle_col*(sizeof_DWORDX4)

            self.wave_prefetches = wave_prefetches
            self.mfma_MN = mfma_MN
            self.J = J

        def __len__(self):
            return self.wave_prefetches

        def reset_offset(self, koff_bytes):
            self.prefetch_sbase[0] = koff_bytes[0] * self.mfma_MN

        def prefetch(self, index):
            assert index < self.wave_prefetches
            self.buff.load_dwordx4(self.prefetch_reg[index], self.prefetch_voffset, self.prefetch_offsets[index] + self.prefetch_sbase[0])

        def ds_write(self, index, lds_base):
            assert index < self.wave_prefetches
            self.J.ds_write_b128(self.ds_write_b128_vaddr[index], self.prefetch_reg[index], mod=f"offset:{lds_base}") #  vaddr, vdata offset gds

    assert BLOCK_TILE_SIZE_M % 16 == 0
    assert BLOCK_TILE_SIZE_N % 32 == 0
    assert N % BLOCK_TILE_SIZE_N == 0

    BLOCK_TILE_SIZE_N_HALF = BLOCK_TILE_SIZE_N // 2

    sizeof_bf16 = 2
    sizeof_f32 = 4
    sizeof_DWORDX4 = 16
    sizeof_w = sizeof_bf16 if weight_dtype == torch.bfloat16 else 1
    stride_A = K * sizeof_bf16
    stride_B = K * sizeof_w
    # 4(mfma columns) * 16(dwordx4) * 2(unroll 2 times)
    BLOCK_TILE_SIZE_K = 4 * 16 * 2 / sizeof_w
    read_col_lanes_per_wave = 4 * 16 * 2 // 16
    read_row_lanes_per_wave = 64 // read_col_lanes_per_wave
    assert BLOCK_TILE_SIZE_M % (4 * read_row_lanes_per_wave) == 0
    A_vert = BLOCK_TILE_SIZE_M // 4 // read_row_lanes_per_wave

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
    J.debug_setup((J.warp_id[0] == 0) & (J.blockIdx.x[0] == 0) & (e_idx[0] == 0))

    # hide following initialization into s_waitcnt
    p_sorted_ids[:] += e_idx * (BLOCK_TILE_SIZE_M * 4)
    # one WG per CU,  4 waves split on K
    lane_mod_16 = get_lane_id_mod(J, 16)
    lane_div_16 = get_lane_id_div(J, 16)
    warp_id = J.gpr("su32")
    J.v_readfirstlane_b32(warp_id, J.threadIdx.x[0] // 64)

    v_sorted_id = J.gpr(A_vert, 'vu32')
    v_sorted_id_off = J.gpr((J.lane_id // read_col_lanes_per_wave + J.warp_id * read_row_lanes_per_wave) * 4)
    J.global_load_dword(v_sorted_id[0], v_sorted_id_off, p_sorted_ids)
    p_sorted_ids_tmp = J.gpr(2, 'su32')
    p_sorted_ids_tmp[0] = p_sorted_ids[0]
    p_sorted_ids_tmp[1] = p_sorted_ids[1]
    for n in range(1, A_vert):
        p_sorted_ids_tmp[:] += read_row_lanes_per_wave * 4 * sizeof_f32
        J.global_load_dword(v_sorted_id[n], v_sorted_id_off, p_sorted_ids_tmp)
    # for write, thread layout [64 // 4(=thread lanes of N), 4(=TILE_N // 2wave // 2(gate+up) * sizeof_bf16) // 16]
    write_col_lanes_per_wave = BLOCK_TILE_SIZE_N // 2 // 2 * sizeof_bf16 // 16
    write_row_lanes_per_wave = 64 // write_col_lanes_per_wave
    assert BLOCK_TILE_SIZE_M % (4 * write_row_lanes_per_wave) == 0
    A_vert_write = BLOCK_TILE_SIZE_M // 2 // write_row_lanes_per_wave
    v_sorted_id_write = J.gpr(A_vert_write, 'vu32')
    v_sorted_id_write_off = J.gpr((J.lane_id // write_col_lanes_per_wave + J.warp_id // 2 * (A_vert_write * write_row_lanes_per_wave)) * 4)
    J.global_load_dword(v_sorted_id_write[0], v_sorted_id_write_off, p_sorted_ids)
    for n in range(1, A_vert_write):
        p_sorted_ids[:] += write_row_lanes_per_wave * sizeof_f32
        J.global_load_dword(v_sorted_id_write[n], v_sorted_id_write_off, p_sorted_ids)

    p_output[:] = p_output[:] + BLOCK_TILE_SIZE_N_HALF * sizeof_bf16 * J.blockIdx.x

    # wait for v_sorted_id
    J.s_waitcnt(mod=f"vmcnt(0)")
    v_token_id = J.gpr(A_vert, 'vu32')
    for m in range(A_vert):
        v_token_id[m] = v_sorted_id[m] & 0xffffff

    p_weight[:] = p_weight[:] + s_e_id * (N * K * sizeof_w)

    gemm = UGEMM(J, 16, [BLOCK_TILE_SIZE_M // 2, BLOCK_TILE_SIZE_N // 2], [2, 2], K, N)
    swizzle_row_div = 1
    total_wave_cnt = 4
    loaderA = MFMA_DW4Loader(J, p_input, M * (K * sizeof_bf16), v_token_id,
                             gemm.wg_M, sizeof_bf16*gemm.wg_K, stride_A,
                             total_wave_cnt, swizzle_row_div, False)
    loaderB = MFMA_DW4Loader_preshuffled(J, p_weight, N * K * sizeof_bf16, 16,
                                         gemm.wg_N, sizeof_bf16*gemm.wg_K, stride_B,
                                         total_wave_cnt, swizzle_row_div, False, N)
    # C_reg: [wave_nCM, wave_nCN]
    C_reg = gemm.run(loaderA, loaderB, None, M, 0, 0)

    s_cvt_bf16_bias = J.gpr(1, "su32")
    s_cvt_bf16_bias[0] = 0x00008000

    # swizzle-LDS to form better memory-coelascing VMEM
    # each warp has its own [mfma_M x wave_size_N] output buffer
    wave_size_N = BLOCK_TILE_SIZE_N // 2 // 2
    lds_out = J.alloc_lds(4 * 16 * wave_size_N * sizeof_bf16)
    lds_warp_offset = warp_id * (16 * wave_size_N * sizeof_bf16)
    num_lanes_per_row = write_col_lanes_per_wave
    assert 64 % num_lanes_per_row == 0
    rows_per_read = 64 // num_lanes_per_row
    assert rows_per_read > 0
    assert 16 % rows_per_read == 0

    vdata = J.gpr(4, "vu32", align=4)

    assert A_vert_write % gemm.wave_nCM == 0
    write_times16 = A_vert_write // gemm.wave_nCM
    for m in range(gemm.wave_nCM):
        for n in range(gemm.wave_nCN // 2):
            for i in range(4):
                tmp = J.gpr(2, "vf32", align=2)
                J.v_accvgpr_read_b32(tmp[0], C_reg[m, 2 * n + 0, i])
                J.v_accvgpr_read_b32(tmp[1], C_reg[m, 2 * n + 1, i])
                tmp[0] = J.gpr(tmp[1] * J.silu(tmp[0]))
                J.v_add_u32(vdata[i], tmp[0], s_cvt_bf16_bias)
            vouts = J.gpr(2, "vf32")
            J.pk_f32_to_bf16(vouts[0], vdata[0], vdata[1])
            J.pk_f32_to_bf16(vouts[1], vdata[2], vdata[3])

            row = lane_mod_16
            col = lane_div_16 + n * (16 * sizeof_bf16 // 8)
            # writing unit is 8 bytes while reading unit is 16 bytes
            swizzle_col = (col // 2 ^ row) % (num_lanes_per_row)
            vaddr_w = J.gpr((row) * (wave_size_N * sizeof_bf16) + \
                            lds_warp_offset + \
                            (swizzle_col * 16 + (col & 1) * 8))
            J.ds_write_b64(vaddr_w, vouts, mod=f"offset:{lds_out}")

        for r in range(0, 16, rows_per_read):
            cur_m = v_sorted_id_write[m * write_times16 + r // rows_per_read] & 0xffffff
            voffset = J.gpr(cur_m * (N // 2 * sizeof_bf16 * TOPK) +
                            (v_sorted_id_write[m * write_times16 + r // rows_per_read] >> 24) * (N // 2 * sizeof_bf16) + 
                            (J.lane_id % write_col_lanes_per_wave) * sizeof_DWORDX4 + J.warp_id % 2 * (write_col_lanes_per_wave * sizeof_DWORDX4))
            row = J.lane_id // num_lanes_per_row + r
            col = J.lane_id % num_lanes_per_row
            swizzle_col = (row ^ col) % num_lanes_per_row
            vaddr_r = J.gpr((swizzle_col) * sizeof_DWORDX4 + \
                            (row) * (wave_size_N * sizeof_bf16) + \
                            lds_warp_offset)
            J.ds_read_b128(vdata, vaddr_r, mod=f"offset:{lds_out}")
            J.s_waitcnt(mod=f"lgkmcnt(0)")
            with J.ExecMask(cur_m < M[0]):
                J.global_store_dwordx4(voffset, vdata, p_output)


@jit()
def moe_2stage_down(J:JIT,
                    weight_dtype,
                    TOPK, K, N,          # 8,  128, 2048
                    with_silu,           #
                    BLOCK_TILE_SIZE_M,   # 32
                    BLOCK_TILE_SIZE_N,   # 64
                    p_input:"void*",     # [8192, 8, 128]
                    p_weight:"void*",    # [128, 2048, 128]
                    p_output:"void*",    # [8192, 2048]
                    p_sorted_ids:"void*",        # [69624]
                    p_sorted_weights:"float*",   # [69624]
                    p_sorted_expert_ids:"void*", # [2176]
                    p_num_valid_ids:"void*",     # [2]  value: [65536,  8192]
                    p_w_scale:"float*",
                    M:"int",):
    sizeof_w = J.sizeof(weight_dtype)
    dtype_A = "bf16"

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

    p_sorted_ids[:] += e_idx * (BLOCK_TILE_SIZE_M * 4)
    p_sorted_weights[:] += e_idx * (BLOCK_TILE_SIZE_M * 4)
    p_weight[:] += s_e_id * (N * K * sizeof_w)

    mfma_MN = 16
    mfma_K = (64//mfma_MN) * (J.sizeof_DW4//J.sizeof(dtype_A))
    num_mfma_m = J.div(BLOCK_TILE_SIZE_M, mfma_MN)
    num_mfma_k = J.div(K, mfma_K)

    A = J.gpr(num_mfma_m, num_mfma_k, 4, "abf16x2")

    # collect token id
    row = J.lane_id % mfma_MN
    col = J.lane_id // mfma_MN
    v_sorted_id = J.gpr(num_mfma_m, 'vu32')
    for m in range(num_mfma_m):
        J.global_load_dword(v_sorted_id[m], row*J.sizeof_DW + m*mfma_MN*J.sizeof_DW, p_sorted_ids)

    J.s_waitcnt(mod=f"vmcnt(0)")

    vaddr = J.gpr(num_mfma_m, "vu32")
    for m in range(num_mfma_m):
        vaddr[m] = (v_sorted_id[m] & 0xFFFFFF) * (TOPK * K * sizeof_w) + (v_sorted_id[m]>>24) *(K * sizeof_w) + col * J.sizeof_DW4

    buff_a = J.Buffer(p_input, M * TOPK * K * J.sizeof(dtype_A))
    for m in range(num_mfma_m):
        for k in range(num_mfma_k):
            buff_a.load_dwordx4(A[m,k], vaddr[m], 0, offset12=k*mfma_K*J.sizeof(dtype_A))
    
    v_sorted_weights = J.gpr('vf32')
    assert BLOCK_TILE_SIZE_M <= 256
    with J.ExecMask(J.threadIdx.x < BLOCK_TILE_SIZE_M):
        J.global_load_dword(v_sorted_weights, J.threadIdx.x * 4, p_sorted_weights)

    J.s_waitcnt(mod=f"vmcnt(0)")

    lds_weights = J.alloc_lds(256*4)
    lds_token_ids = J.alloc_lds(256*4)

    with J.ExecMask(J.threadIdx.x < BLOCK_TILE_SIZE_M):
        J.ds_write_b32(J.threadIdx.x * 4, v_sorted_weights, mod=f"offset:{lds_weights}")
    with J.ExecMask(J.threadIdx.x < 16):
        for m in range(num_mfma_m):
            J.ds_write_b32(J.lane_id * 4, v_sorted_id[m] & 0xffffff, mod=f"offset:{lds_token_ids + m*16*4}")

    J.s_waitcnt(mod=f"lgkmcnt(0)")
    J.s_barrier()

    num_mfma_n = 1 if BLOCK_TILE_SIZE_M > 64 else 2
    
    down_kernel(J, mfma_MN, num_mfma_n, BLOCK_TILE_SIZE_M, N, K,
                A, lds_token_ids, lds_weights,
                p_weight, p_output, M)

def down_kernel(J, mfma_MN, num_mfma_n, BM, N, K,
                A, # A = J.gpr(num_mfma_m, num_mfma_k, 4, "abf16x2")
                lds_token_ids,  # 256 int32 token_id
                lds_weights,    # 256 fp32 weights 
                pB:"void*",
                pC:"void*",
                M):
    # given DW4 lane size, how many bf16 items along K direction
    mfma_K = (64//mfma_MN) * (J.sizeof_DW4//J.sizeof_bf16) # mfma_K = 32

    # load A [BM x K] bf16 into AccGPRs 
    num_mfma_m = J.div(BM, mfma_MN)
    num_mfma_k = J.div(K, mfma_K)  # K=96, num_mfma_k=3

    # 4 warps work in parallel along N dimension
    buff_b = J.Buffer(pB, N * K * J.sizeof_bf16)
    # ping-pong buffer
    B = J.gpr(2, num_mfma_n, num_mfma_k, 4, "vbf16x2")
    C = J.gpr(2, num_mfma_m, num_mfma_n, 4, "vf32")

    # prelog0, load Bn0
    # prelog1, load Bn1, compute Cn0
    # loop:    load Bn2, compute Cn1, store Cn0 to LDS & load Cn0 & store to HBM
    voff_b = J.gpr(J.lane_id * J.sizeof_DW4 + J.gpr(J.warp_id * (mfma_MN * K * J.sizeof_bf16)))
    soff_b = J.gpr("su32")
    soff_b[0] = 0
    def loadB_generator(index):
        for n in range(num_mfma_n):
            for k in range(num_mfma_k):
                yield 1
                buff_b.load_dwordx4(B[index,n,k], voff_b, soff_b)
                soff_b[0] = soff_b[0] + mfma_MN * mfma_K * J.sizeof_bf16
            soff_b[0] = soff_b[0] + (3*num_mfma_k * mfma_MN * mfma_K * J.sizeof_bf16)

    def mfma_generator(index):
        for k in range(num_mfma_k):
            for m in range(num_mfma_m):
                for n in range(num_mfma_n):
                    Ci = 0 if k == 0 else C[index,m,n]
                    yield 16
                    J.v_mfma_f32_16x16x16_bf16(C[index,m,n], B[index,n,k,0:1], A[m,k,0:1], Ci)
        for k in range(num_mfma_k):
            for m in range(num_mfma_m):
                for n in range(num_mfma_n):
                    yield 16
                    J.v_mfma_f32_16x16x16_bf16(C[index,m,n], B[index,n,k,2:3], A[m,k,2:3], C[index,m,n])

    # prelog0, load Bn0
    J.emit(loadB_generator(0))

    # prelog1, load Bn1, compute Cn0
    J.emit(loadB_generator(1))
    J.s_waitcnt(mod=f"vmcnt({num_mfma_n*num_mfma_k})")
    J.s_waitcnt(mod=f"lgkmcnt(0)")

    J.emit(mfma_generator(0))

    # loop:    load Bn2, compute Cn1, store Cn0 to LDS & load Cn0 & store to HBM
    s_cvt_bf16_bias = J.get_sgpr_const(0x00008000)

    vmem_lane_size = J.sizeof_DW

    lds_padding = (4 if vmem_lane_size == J.sizeof_DW else 8) * J.sizeof_bf16 # to avoid bank-conflict
    lds_width = num_mfma_n * 4 * mfma_MN * J.sizeof_bf16
    lds_stride = lds_width + lds_padding

    # WG level write C into LDS
    row = J.threadIdx.x % mfma_MN
    col = J.threadIdx.x // mfma_MN
    voff_c_lds_w = J.gpr(row * lds_stride + col * (4 * J.sizeof_bf16))

    # WG level load C from LDS
    num_lanes_ldsr = J.div(lds_width, vmem_lane_size)
    assert num_lanes_ldsr <= 64, num_lanes_ldsr
    col = J.threadIdx.x % num_lanes_ldsr
    row = J.threadIdx.x // num_lanes_ldsr
    num_rows_per_load = J.div(256, num_lanes_ldsr)
    num_loads = J.div(num_mfma_m * mfma_MN, num_rows_per_load)
    voff_c_lds_r = J.gpr(row * lds_stride + col * (vmem_lane_size))
    vmem_stride = N * J.sizeof_bf16
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

    temp_c = J.gpr(num_loads, vmem_lane_size//J.sizeof_DW, "vbf16x2")

    def loop_body(ni):
        J.s_waitcnt(mod=f"vmcnt({num_loads})")

        mfma1 = mfma_generator((ni+1)&1)
        B_loader = loadB_generator(ni&1)

        #cvt_f32_to_pk_bf16(n&1)
        index = ni&1
        for m in range(num_mfma_m):
            J.emit(B_loader, 1)
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
            J.emit(B_loader, 1)
            for n in range(num_mfma_n):
                offset = lds + m*mfma_MN*lds_stride + n*(4*mfma_MN*J.sizeof_bf16)
                J.ds_write_b64(voff_c_lds_w, C[index, m, n, 0:1], mod=f"offset:{offset}")
                J.emit(mfma1, 16)
        J.emit(mfma1, 32)
        J.s_waitcnt(mod=f"lgkmcnt(0)")
        J.s_barrier()

        #ds_load_C()
        for i in range(num_loads):
            offset = lds + i * num_rows_per_load * lds_stride
            if vmem_lane_size == J.sizeof_DW4:
                J.ds_read_b128(temp_c[i], voff_c_lds_r, mod=f"offset:{offset}")
            else:
                assert vmem_lane_size == J.sizeof_DW
                J.ds_read_b32(temp_c[i], voff_c_lds_r, mod=f"offset:{offset}")
            J.emit(mfma1, 32)

        J.emit(B_loader)

        #atomic_pk_add_bf16()
        for i in range(num_loads):
            J.s_waitcnt(mod=f"lgkmcnt({min(15,num_loads - i - 1)})")
            if vmem_lane_size == J.sizeof_DW4:
                J.global_store_dwordx4(voff_vmem[i], temp_c[i], pC)      # this is fast:  (48us)
            else:
                assert vmem_lane_size == J.sizeof_DW
                # the bigger the M is, the bigger the perf-diff is
                with J.ExecMask(voff_vmem_row[i] < M[0], early_skip=False):
                    J.global_atomic_pk_add_bf16(voff_vmem[i], temp_c[i], pC) # this is much slower than directly store      (60us)
            J.emit(mfma1, 32)

        J.emit(mfma1)
        pC[:] += (4 * num_mfma_n * mfma_MN * J.sizeof_bf16) 

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
def moe_1stage_splitk(J:JIT,
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
    # to get 4*8bf16(aka dword16) for gemm2, the gemm1 output must be multiple of 64, after gate*up the width will be 32
    assert BLOCK_TILE_SIZE_N % 64 == 0
    K2 = N1//2
    sizeof_w = J.sizeof(weight_dtype)
    stride_A1 = K1 * J.sizeof_bf16

    A_vert = J.div(BLOCK_TILE_SIZE_M, 16)
    B_horz = J.div(BLOCK_TILE_SIZE_N, 16)
    C_reg = J.gpr(B_horz, A_vert, 4, "vf32", 0)

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

    lds_buff = J.LDSTensor([4 * BLOCK_TILE_SIZE_M, 64], torch.float32)
    lds_buff_out = lds_buff.view_as([BLOCK_TILE_SIZE_M, 32 // 2], torch.float32)

    v_sorted_id = J.gpr(A_vert, 'vu32')
    for n in range(A_vert):
        J.global_load_dword(v_sorted_id[n], lane_mod_16*J.sizeof_DW + n*16*J.sizeof_DW, p_sorted_ids)

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

    buff_a = J.Buffer(p_input, M * (K1 * J.sizeof_bf16))

    # wait for v_sorted_id
    J.s_waitcnt(mod=f"vmcnt(0)")

    voffset_a1 = J.gpr(A_vert, 'vu32')
    v_topk_id = J.gpr(A_vert, 'vu32')
    v_token_id = J.gpr(A_vert, 'vu32')
    # 16 elements for a if fp8
    a_element_num_per_thread = 8 if weight_dtype == torch.bfloat16 else 16
    for m in range(A_vert):
        v_token_id[m] = v_sorted_id[m] & 0xffffff
        voffset_a1[m] = J.gpr(v_token_id[m] * stride_A1 + (J.threadIdx.x // 16) * (a_element_num_per_thread * J.sizeof_bf16))
        v_topk_id[m] = v_sorted_id[m] >> 24
        # # input layout: [B, TOPK, INTER_MEDIA]
        #voffset_a2[m] = J.gpr(v_token_id[m] * TOPK * K * sizeof_bf16 + v_topk_id[m] * (K * sizeof_bf16) + (J.threadIdx.x // 16) * (a_element_num_per_thread * sizeof_bf16))
    voffset_scale = J.gpr(B_horz, 'vu32')
    if weight_dtype != torch.bfloat16:
        voffset_scale[0] = J.gpr(s_e_id * (N1 * J.sizeof_f32)) + lane_mod_16 * J.sizeof_f32
        voffset_scale[B_horz // 2] = voffset_scale[0] + N1 // 2 * J.sizeof_f32
        for m in range(1, B_horz // 2):
            voffset_scale[m] = voffset_scale[0] + 16 * J.sizeof_f32 * m
            voffset_scale[B_horz // 2 + m] = voffset_scale[B_horz // 2] + 16 * J.sizeof_f32 * m

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
    vrow = J.gpr(lane_div_16 + J.warp_id * 4)

    A_reg_gemm2 = J.gpr(A_vert, (N1 // BLOCK_TILE_SIZE_N) * (n_groups // 2), 4, "abf16x2")
    for block_n in range(N1 // BLOCK_TILE_SIZE_N):
        if block_n != 0:
            C_reg[:] = 0
        gemm_splitk(J, weight_dtype, K1, N1, 4,
                    buff_a, buff_b, p_w_up_scale,
                    voffset_a1, voffset_b, voffset_scale, C_reg,
                    soffset_a=0, soffset_b=block_n * (BLOCK_TILE_SIZE_N // 2 * K1 * sizeof_w),
                    BLOCK_TILE_SIZE_N=BLOCK_TILE_SIZE_N, BLOCK_TILE_SIZE_M=BLOCK_TILE_SIZE_M)
        # each wave holds [4, 16, A_vert] floats
        vouts = J.gpr(A_vert, "vf32")
        for n in range(n_groups // 2):
            vaddr = J.gpr("vu32")
            for m in range(A_vert):
                lds_buff.write("b128", C_reg[n * 2 + 0 + n_half0, m], lane_mod_16 + J.warp_id * 16 + m * 64, lane_div_16 * 4)
                lds_buff.write("b128", C_reg[n * 2 + 1 + n_half0, m], lane_mod_16 + J.warp_id * 16 + m * 64, lane_div_16 * 4 + 16)
                lds_buff.write("b128", C_reg[n * 2 + 0 + n_half1, m], lane_mod_16 + J.warp_id * 16 + m * 64, lane_div_16 * 4 + 32)
                lds_buff.write("b128", C_reg[n * 2 + 1 + n_half1, m], lane_mod_16 + J.warp_id * 16 + m * 64, lane_div_16 * 4 + 48)

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
                    vaddr = J.gpr(v_token_id * (N1 // 2 * J.sizeof_bf16 * TOPK) + v_topk_id * (N1 // 2 * J.sizeof_bf16) + lane_mod_16 * sizeof_f32 + n * 16 * sizeof_f32 + block_n * (BLOCK_TILE_SIZE_N // 2) * J.sizeof_bf16)
                    with J.ExecMask(v_token_id < M[0]):
                        J.global_store_dword(vaddr, vouts[m], p_output)

            # be sure read done
            #    lds_buff_out: [BLOCK_TILE_SIZE_M, 32 // 2], torch.float32)
            J.s_barrier()
            for m in range(A_vert):
                lds_buff_out.write("b32", vouts[m],
                                   lane_div_16 + J.warp_id * 4 + m * 16, lane_mod_16)

            J.s_waitcnt(mod=f"lgkmcnt(0)")
            J.s_barrier()

            for m in range(A_vert):
                lds_buff_out.read("b128", A_reg_gemm2[m, block_n * (n_groups // 2) + n],
                                   lane_mod_16 + m * 16, lane_div_16 * 4)

            J.s_waitcnt(mod=f"lgkmcnt(0)")
            J.s_barrier()

        if weight_dtype != torch.bfloat16:
            p_w_up_scale[:] += BLOCK_TILE_SIZE_N // 2 * J.sizeof_f32

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
