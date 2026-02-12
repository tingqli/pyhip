import os
import random
from typing import Optional

import pytest
os.environ['PYHIP_JIT_LOG'] = '0'
from pyhip import cudaPerf, jit, JIT
import torch

from functools import cache
try:
    # work as package
    from .common.gemm import UGEMM
    from .common.gemm_splitk import gemm_splitk
except:
    from common.gemm import UGEMM
    from common.gemm_splitk import gemm_splitk

USE_FP4_SHUFFLE_WEIGHT = 1
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
                      p_w_scale:"void*",
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
        assert K % 128 == 0, f'will read 16*4 bytes once in main loop, aka 64 elements; current K={K} is not supported'
    else:
        assert K % 128 == 0, f'will read 16*4 bytes once in main loop, aka 64/2=32 elements; current K={K} is not supported'
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
    if with_silu:
        C_reg = J.gpr(B_horz, A_vert, 4, "af32")
    else:
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
    # elif weight_dtype == torch.float8_e4m3fn or weight_dtype == torch.float8_e4m3fnuz:
    #     voffset_scale = J.gpr(B_horz, 'vu32')
    #     p_w_scale[:] += (BLOCK_TILE_SIZE_N_HALF if with_silu else BLOCK_TILE_SIZE_N) * sizeof_f32 * J.blockIdx.x
    #     voffset_scale[0] = J.gpr(s_e_id * (N * sizeof_f32)) + lane_mod_16 * sizeof_f32
    #     if with_silu:
    #         voffset_scale[B_horz // 2] = voffset_scale[0] + N // 2 * sizeof_f32
    #         for m in range(1, B_horz // 2):
    #             voffset_scale[m] = voffset_scale[0] + 16 * sizeof_f32 * m
    #             voffset_scale[B_horz // 2 + m] = voffset_scale[B_horz // 2] + 16 * sizeof_f32 * m
    #     else:
    #         for m in range(1, B_horz):
    #             voffset_scale[m] = voffset_scale[0] + 16 * sizeof_f32 * m

    elif weight_dtype == torch.float8_e4m3fn or weight_dtype == torch.float8_e4m3fnuz:
        # the scale layout is [E, N, K//128]
        k_scale_stride = K // 128 * sizeof_f32
        voffset_scale = J.gpr(B_horz, 'vu32')
        # N_wg scale offset
        p_w_scale[:] += (BLOCK_TILE_SIZE_N_HALF if with_silu else BLOCK_TILE_SIZE_N) * J.blockIdx.x * k_scale_stride
        if with_silu:
            #[E, INTER_SIZE_TP*2, HIDDEN_SIZE//128]
            # Expert wg offset + 64 lane offset + warp offset
            voffset_scale[0] = J.gpr(s_e_id * (N * k_scale_stride)) + lane_mod_16 * k_scale_stride + J.warp_id * (k_scale_stride // 4)
            # N tile offset within wave offset
            voffset_scale[B_horz // 2] = voffset_scale[0] + (N // 2 * k_scale_stride)
            for m in range(1, B_horz // 2):
                voffset_scale[m] = voffset_scale[0] + 16 * m * k_scale_stride
                voffset_scale[B_horz // 2 + m] = voffset_scale[B_horz // 2] + 16 * m * k_scale_stride
        else:
            # Expert wg offset + 64 lane offset 
            voffset_scale[0] = J.gpr(s_e_id * (N * k_scale_stride)) + lane_mod_16 * k_scale_stride 
            for m in range(1, B_horz):
                voffset_scale[m] = voffset_scale[0] + 16 * m * k_scale_stride
    
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
                col = lane_div_16
                row = lane_mod_16
                col = (row ^ col) % 8
                lds_buff.write("b128", C_reg[n * 2 + 0 + n_half0, m], lane_mod_16 + J.warp_id * 16 + m * 64, col * 4)
                col = lane_div_16 + 4
                row = lane_mod_16
                col = (row ^ col) % 8
                lds_buff.write("b128", C_reg[n * 2 + 1 + n_half0, m], lane_mod_16 + J.warp_id * 16 + m * 64, col * 4)
                col = lane_div_16
                row = lane_mod_16
                col = (row ^ col) % 8
                lds_buff.write("b128", C_reg[n * 2 + 0 + n_half1, m], lane_mod_16 + J.warp_id * 16 + m * 64, col * 4 + 32)
                col = lane_div_16 + 4
                row = lane_mod_16
                col = (row ^ col) % 8
                lds_buff.write("b128", C_reg[n * 2 + 1 + n_half1, m], lane_mod_16 + J.warp_id * 16 + m * 64, col * 4 + 32)

            J.s_waitcnt(mod=f"lgkmcnt(0)")
            J.s_barrier()
            v_sorted_id_permute = J.gpr("vu32")
            for m in range(A_vert):
                J.ds_bpermute_b32(v_sorted_id_permute, vrow * 4, v_sorted_id[m]) 
                gate_up = J.gpr(4, 2, 2, "vf32")
                # interleave 
                col = lane_mod_16 // 2
                row = vrow
                col = (row ^ col) % 8
                lds_buff.read("b64", gate_up[0], vrow + (0 * 16 + m * 64), col * 4 + (lane_mod_16 & 1) * 2, offset1 = 16)
                lds_buff.read("b64", gate_up[1], vrow + (1 * 16 + m * 64), col * 4 + (lane_mod_16 & 1) * 2, offset1 = 16)
                lds_buff.read("b64", gate_up[2], vrow + (2 * 16 + m * 64), col * 4 + (lane_mod_16 & 1) * 2, offset1 = 16)
                lds_buff.read("b64", gate_up[3], vrow + (3 * 16 + m * 64), col * 4 + (lane_mod_16 & 1) * 2, offset1 = 16)

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
    if w1.dtype == torch.float4_e2m1fn_x2:
        quant_type = QuantType.per_1x32
    elif w1.dtype == torch.bfloat16:
        quant_type = QuantType.No
    else:
        quant_type = QuantType.per_Token
    return fused_moe(
        hidden_states,
        w1,
        w2,
        topk_weight,
        topk_ids,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        quant_type=quant_type
    )

# https://github.com/huggingface/transformers/blob/1fed6166c00b800330fcda8494f78cbcad8e4e3b/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py#L235-L263
def get_torch_ref(hidden_states, w1, w2, topk_weight, topk_ids):
    batch_size, hidden_dim = hidden_states.shape
    E, N1, K1 = w1.shape
    INTER_SIZE = N1 // 2
    final_hidden_states = torch.zeros(
        (batch_size, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
    )

    # One hot encode the selected experts to create an expert mask
    # this will be used to easily index which expert is going to be sollicitated
    expert_mask = torch.nn.functional.one_hot(topk_ids.to(dtype=torch.long), num_classes=E).permute(2, 1, 0)

    # Loop over all available experts in the model and perform the computation on each expert
    for expert_idx in range(E):
        def expert_forward(n, x):
            gate_proj = w1[n, 0 : INTER_SIZE].t()
            up_proj = w1[n, INTER_SIZE :,].t()
            down_proj = w2[n].t()
            return (torch.nn.functional.silu(x @ gate_proj) * (x @ up_proj)) @ down_proj
        idx, top_x = torch.where(expert_mask[expert_idx])

        # Index the correct hidden states and compute the expert hidden state for
        # the current expert. We need to make sure to multiply the output hidden
        # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
        current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
        current_hidden_states = expert_forward(expert_idx, current_state) * topk_weight[top_x, idx, None]

        # However `index_add_` only support torch tensors for indexing so we'll use
        # the `top_x` tensor here.
        final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
    return final_hidden_states

def _run_batch(kernel_type, B=1, weight_type=torch.bfloat16, TILE_M=16, TILE_N=32, run_count=10, HIDDEN_SIZE=2048, INTER_SIZE=1024, TOPK=8, E=128, TP=8):
    fake_scale = False
    if 'FAKE_SCALE' in os.environ:
        fake_scale = True
        INTMAX=3
        INTMIN=-2
        DIVISOR=100
    INTER_SIZE_TP = INTER_SIZE // TP
    BUF_COPY = 32
    hidden_states = (torch.randn([BUF_COPY, B, HIDDEN_SIZE], dtype=torch.bfloat16) + 1)*0.001
    if weight_type == torch.bfloat16:
        w_ = torch.randn([E, INTER_SIZE_TP * 2, HIDDEN_SIZE], dtype=weight_type)
        w1_ref = w_
        w1 = [w_.clone() for _ in range(BUF_COPY)]
        w_ = torch.randn([E, HIDDEN_SIZE, INTER_SIZE_TP], dtype=weight_type)
        w2_ref = w_
        w2 = [w_.clone() for _ in range(BUF_COPY)]
        w1_scale = [None] * BUF_COPY
        w2_scale = [None] * BUF_COPY
    elif weight_type == torch.float4_e2m1fn_x2:
        import aiter
        from aiter.utility import fp4_utils
        from aiter.ops.shuffle import shuffle_weight
        # w_ = torch.randn([E, INTER_SIZE_TP * 2, HIDDEN_SIZE], dtype=torch.bfloat16)
        w_ = torch.randint(-2, 3, [E, INTER_SIZE_TP * 2, HIDDEN_SIZE], dtype=torch.bfloat16) / 2
        w1_qt, w1_qt_scale_ = aiter.get_torch_quant(aiter.QuantType.per_1x32)(w_, quant_dtype=weight_type)
        w1_f32 = fp4_utils.mxfp4_to_f32(w1_qt).to(dtype=torch.bfloat16).reshape(E, INTER_SIZE_TP * 2, HIDDEN_SIZE // 32, 32)
        w1_scale_f32 = fp4_utils.e8m0_to_f32(w1_qt_scale_).to(dtype=torch.bfloat16).reshape(E, INTER_SIZE_TP * 2, HIDDEN_SIZE // 32, 1)
        w1_ref = (w1_f32 * w1_scale_f32).reshape(E, INTER_SIZE_TP * 2, HIDDEN_SIZE)
        w1_qt_scale = fp4_utils.e8m0_shuffle(w1_qt_scale_)
        if USE_FP4_SHUFFLE_WEIGHT:
            w1 = [shuffle_weight(w1_qt) for _ in range(BUF_COPY)]
        else:
            w1 = [w1_qt.clone() for _ in range(BUF_COPY)]
        w1_scale = [w1_qt_scale.clone() for _ in range(BUF_COPY)]
        # w_ = torch.randn([E, HIDDEN_SIZE, INTER_SIZE_TP], dtype=torch.bfloat16)
        w_ = torch.randint(-1, 3, [E, HIDDEN_SIZE, INTER_SIZE_TP], dtype=torch.bfloat16) / 2
        w2_qt, w2_qt_scale_ = aiter.get_torch_quant(aiter.QuantType.per_1x32)(w_, quant_dtype=weight_type)
        w2_f32 = fp4_utils.mxfp4_to_f32(w2_qt).to(dtype=torch.bfloat16).reshape(E, HIDDEN_SIZE, INTER_SIZE_TP // 32, 32)
        w2_scale_f32 = fp4_utils.e8m0_to_f32(w2_qt_scale_).to(dtype=torch.bfloat16).reshape(E, HIDDEN_SIZE, INTER_SIZE_TP // 32, 1)
        w2_ref = (w2_f32 * w2_scale_f32).reshape(E, HIDDEN_SIZE, INTER_SIZE_TP)
        # pad scale
        w2_qt_scale_pad = torch.zeros(w2_qt_scale_.shape[0], div_up(w2_qt_scale_.shape[1], 8) * 8, dtype=w2_qt_scale_.dtype)
        w2_qt_scale_pad[:, :w2_qt_scale_.shape[1]] = w2_qt_scale_
        w2_qt_scale = fp4_utils.e8m0_shuffle(w2_qt_scale_pad)
        if USE_FP4_SHUFFLE_WEIGHT:
            w2 = [shuffle_weight(w2_qt) for _ in range(BUF_COPY)]
        else:
            w2 = [w2_qt.clone() for _ in range(BUF_COPY)]
        w2_scale = [w2_qt_scale.clone() for _ in range(BUF_COPY)]
    # elif weight_type == torch.float8_e4m3fn or weight_type == torch.float8_e4m3fnuz: 
    #     import aiter
    #     torch_quant = aiter.get_torch_quant(aiter.QuantType.per_Token)
    #     w_ = torch.randn([E, INTER_SIZE_TP * 2, HIDDEN_SIZE], dtype=torch.bfloat16)
    #     w1_qt, w1_qt_scale = torch_quant(w_, quant_dtype=weight_type)
    #     w1_ref = (w1_qt.to(dtype=torch.bfloat16) * w1_qt_scale).to(dtype=torch.bfloat16)
    #     w1 = [w1_qt.clone() for _ in range(BUF_COPY)]
    #     w1_scale = [w1_qt_scale.clone() for _ in range(BUF_COPY)]
    #     w_ = torch.randn([E, HIDDEN_SIZE, INTER_SIZE_TP], dtype=torch.bfloat16)
    #     w2_qt, w2_qt_scale = torch_quant(w_, quant_dtype=weight_type)
    #     w2_ref = (w2_qt.to(dtype=torch.bfloat16) * w2_qt_scale).to(dtype=torch.bfloat16)
    #     w2 = [w2_qt.clone() for _ in range(BUF_COPY)]
    #     w2_scale = [w2_qt_scale.clone() for _ in range(BUF_COPY)]
    elif weight_type == torch.float8_e4m3fn or weight_type == torch.float8_e4m3fnuz: 
        assert  HIDDEN_SIZE//(128) and INTER_SIZE//(128*TP), "HIDDEN_SIZE and INTER_SIZE//TP must be multiples of 128  for per block quantization"
        import aiter
        QUAN_BLOCK_SZ = 128
        assert HIDDEN_SIZE % QUAN_BLOCK_SZ == 0 and QUAN_BLOCK_SZ % 128 == 0, "HIDDEN_SIZE must be divisible by QUAN_BLOCK_SZ"
        torch_quant = aiter.get_torch_quant(aiter.QuantType.per_Token)
        if 0:
            w_ = torch.randn([E*INTER_SIZE_TP * 2 * HIDDEN_SIZE // QUAN_BLOCK_SZ, QUAN_BLOCK_SZ], dtype=torch.bfloat16)
        else:
            w_ = torch.randint(-3, 5,[E*INTER_SIZE_TP * 2 * HIDDEN_SIZE // QUAN_BLOCK_SZ, QUAN_BLOCK_SZ]).to(dtype=torch.bfloat16) / 100.0
            # ensure every 3 adjacent rows(each row is QUAN_BLOCK_SZ) would have different 3 quantization scales.
            w_[1:-1:3, :] *= 2
            w_[2:-1:3, :] *= 3
            #random change data in QUAN_BLOCK_SZ dimension.
            w_[:, 0:-1:3] *= 2
            
        w1_qt, w1_qt_scale = torch_quant(w_, quant_dtype=weight_type)
        w1_qt_scale.view(-1, 1)
        w1_qt_scale = w1_qt_scale.repeat(1, QUAN_BLOCK_SZ//128)
        w1_qt = w1_qt.view(E, INTER_SIZE_TP * 2, HIDDEN_SIZE)
        w1_qt_scale = w1_qt_scale.view(E, INTER_SIZE_TP * 2, HIDDEN_SIZE//128)
        # if fake_scale:
        #     tmp_k_pos = 63
        #     hidden_states[:,:,:] = 0
        #     hidden_states[:,0,tmp_k_pos] = 1.0
        #     # E, INTER_SIZE_TP * 2, HIDDEN_SIZE, fp8
        #     w1_qt[:,:,:] = 0.0
        #     w1_qt[0,0,tmp_k_pos] = 4.0
        #     w1_qt[0,INTER_SIZE_TP,tmp_k_pos] = 4.0
        #     # w1_qt[0,:,tmp_k_pos] = 4.0

        #     #E, INTER_SIZE_TP * 2, HIDDEN_SIZE//128, float32
        #     w1_qt_scale[:,:,:] = 0.0
        #     w1_qt_scale[0,0, tmp_k_pos//128] = 2.0
        #     w1_qt_scale[0,INTER_SIZE_TP, tmp_k_pos//128] = 2.0
        #     # w1_qt_scale[0,:,tmp_k_pos//128] = 2.0

            # [BUF_COPY, B, HIDDEN_SIZE],
        print(f'==========={w1_qt.shape=}, {w1_qt_scale.shape=}')
        w1_ref = (w1_qt.to(dtype=torch.bfloat16).view(-1, 128) * w1_qt_scale.view(-1, 1)).to(dtype=torch.bfloat16)
        w1_ref = w1_ref.view(E, INTER_SIZE_TP * 2, HIDDEN_SIZE)
        w1 = [w1_qt.clone() for _ in range(BUF_COPY)]
        w1_scale = [w1_qt_scale.clone() for _ in range(BUF_COPY)]
        if 0:
            w_ = torch.randn([E*HIDDEN_SIZE, INTER_SIZE_TP], dtype=torch.bfloat16)
        else:
            w_ = torch.randint(-3, 5,[E*HIDDEN_SIZE, INTER_SIZE_TP]).to(dtype=torch.bfloat16) / 100.0
            w_[:, 0:-1:3] += 0.04
        torch_quant = aiter.get_torch_quant(aiter.QuantType.per_1x128)

        w2_qt, w2_qt_scale = torch_quant(w_, quant_dtype=weight_type)
        w2_qt = w2_qt.view(E, HIDDEN_SIZE, INTER_SIZE_TP)
        w2_qt_scale = w2_qt_scale.view(E, HIDDEN_SIZE, INTER_SIZE_TP//128)

        print(f'==========={w2_qt.shape=}, {w2_qt_scale.shape=}')
        
        w2_ref = (w2_qt.to(dtype=torch.bfloat16).view(-1, 128) * w2_qt_scale.view(-1, 1)).to(dtype=torch.bfloat16)
        w2_ref = w2_ref.view(E, HIDDEN_SIZE, INTER_SIZE_TP)

        w2 = [w2_qt.clone() for _ in range(BUF_COPY)]
        w2_scale = [w2_qt_scale.clone() for _ in range(BUF_COPY)]


    else:
        assert 0, f'not support weight type "{weight_type}"'

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
    if weight_type == torch.bfloat16:
        ele_size = 2
    elif weight_type == torch.float4_e2m1fn_x2:
        ele_size = 0.5
    else:
        ele_size = 1
    mem_size = B * HIDDEN_SIZE * 2 + (HIDDEN_SIZE * INTER_SIZE_TP * 2 + HIDDEN_SIZE * INTER_SIZE_TP) * access_expert * ele_size

    import aiter
    from aiter.ops.shuffle import shuffle_weight
    from aiter.fused_moe import moe_sorting

    def run(hidden_states, w1, w2, topk_weight, topk_ids, w1_scale, w2_scale):
        B = hidden_states.shape[0]
        E, N1, K1 = w1.shape
        N2, K2 = w2.shape[1], w2.shape[2]
        gemm1_out = torch.empty([B, TOPK, N1 // 2], dtype=hidden_states.dtype, device=hidden_states.device)
        if kernel_type == 'mxn_splitk_2s':
            # test moe_gemm_batch_vmn: 2 stages, m/n can be set
            if weight_type == torch.float4_e2m1fn_x2:
                K1 *= 2
                K2 *= 2
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
            grid = sorted_expert_ids.shape[0]
            if B * TOPK <= E:
                grid = B * TOPK
            moe_2stage_splitk([N1 // BLOCK_TILE_SIZE_N, grid], [256],
                               w1.dtype, TOPK, K1, N1, True, BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N,
                               hidden_states.data_ptr(), w1.data_ptr(), gemm1_out.data_ptr(), sorted_ids.data_ptr(), sorted_weights.data_ptr(), sorted_expert_ids.data_ptr(), num_valid_ids.data_ptr(), w1_scale.data_ptr() if w1_scale is not None else 0, B)
            moe_2stage_splitk([N2 // BLOCK_TILE_SIZE_N, grid], [64],
                               w1.dtype, TOPK, K2, N2, False, BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N,
                               gemm1_out.data_ptr(), w2.data_ptr(), cur_out.data_ptr(), sorted_ids.data_ptr(), sorted_weights.data_ptr(), sorted_expert_ids.data_ptr(), num_valid_ids.data_ptr(), w2_scale.data_ptr() if w2_scale is not None else 0, B)
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
        if weight_type == torch.float4_e2m1fn_x2:
            # fp4 no shuffle
            w1_qt_aiter = w1[0]
            w2_qt_aiter = w2[0]
        else:
            #shuffle to [E, n/16, k/32, 2k, 16n, 16k]
            w1_qt_aiter = shuffle_weight(w1[0], layout=(16, 16))
            w2_qt_aiter = shuffle_weight(w2[0], layout=(16, 16))
        ref_out = get_torch_ref(hidden_states=hidden_states[0], w1=w1_ref, w2=w2_ref, topk_weight=topk_weight[0], topk_ids=topk_ids[0])
        cur_out = run(hidden_states=hidden_states[0], w1=w1_qt_aiter, w2=w2_qt_aiter, topk_weight=topk_weight[0], topk_ids=topk_ids[0], w1_scale=w1_scale[0], w2_scale=w2_scale[0])
        i = 0
        for _ in range(run_count):
            with cudaPerf(flops, mem_size, name=f"{kernel_type}[{B=},{str(weight_type).split('.')[1]}]") as p:
                run(hidden_states=hidden_states[i], w1=w1[i], w2=w2[i], topk_weight=topk_weight[i], topk_ids=topk_ids[i], w1_scale=w1_scale[i], w2_scale=w2_scale[i])
            i = (i + 1) % BUF_COPY
            tflops_res.append(p.tflops())
            latencies.append(p.dt())
            bw.append(p.bw())

        print(ref_out[0,0:40])
        print(cur_out[0,0:40])
        rtol = 0.05
        atoi = 0.01
        if not torch.allclose(cur_out, ref_out, rtol=rtol, atol=atoi):
            # print(cur_out)
            idx = torch.where(torch.abs(ref_out - cur_out) >= (torch.abs(ref_out)*rtol + atoi))
            if len(idx[0]):
                print(f'idx = {idx}\nref={ref_out[idx]}\ncur={cur_out[idx]}\n{len(idx[0])}')
            assert 0, f"{kernel_type=}, {B=}, {weight_type=}, {TILE_M=}, {TILE_N=}, {run_count=}"
        else:
            print(f"{kernel_type}[{B=} {weight_type=}] acc OK")
    if run_count > 0:
        return {'flops': sum(tflops_res[1:])/len(tflops_res[1:]),              # tflops
                'latency': sum(latencies[1:])/len(latencies[1:]) * 1e6,        # us
                'bw': sum(bw[1:]) / len(bw[1:])}                               # GB/s

def is_arch_type(arch):
    props = torch.cuda.get_device_properties()
    return arch in props.gcnArchName

def get_fp8type():
    return torch.float8_e4m3fn if is_arch_type('950') else torch.float8_e4m3fnuz

def get_fp4type_if_valid():
    return torch.float4_e2m1fn_x2 if is_arch_type('950') else None
def entry_common(kernel_type, batch, prec=[torch.bfloat16], TILE_M=32, TILE_N=64, HIDDEN_SIZE=2048, INTER_SIZE=1024, TOPK=8, E=160, TP=8, run_count=10):
    perf = {}
    perf[kernel_type] = {}
    for weight_type in prec:
        if weight_type is None: continue
        perf_prec = {}
        org_INTER_SIZE = INTER_SIZE

        if (weight_type == torch.float8_e4m3fn or weight_type == torch.float8_e4m3fnuz) and INTER_SIZE // TP % 128 != 0:
            INTER_SIZE = div_up(INTER_SIZE // TP, 128) * 128 * TP
        if kernel_type == 'aiter' and weight_type == torch.float4_e2m1fn_x2 and INTER_SIZE // TP % 128 != 0:
            INTER_SIZE = div_up(INTER_SIZE // TP, 128) * 128 * TP
        for i in batch:
            if org_INTER_SIZE != INTER_SIZE:
                key = f'{i} (adjusted INTER_SIZE={INTER_SIZE})'
            else:
                key = f'{i}'
            perf_prec[key] = _run_batch(kernel_type, B=i, weight_type=weight_type, TILE_M=TILE_M, TILE_N=TILE_N, run_count=run_count, HIDDEN_SIZE=HIDDEN_SIZE, INTER_SIZE=INTER_SIZE, TOPK=TOPK, E=E, TP=TP)
        perf[kernel_type][str(weight_type)] = perf_prec
        INTER_SIZE = org_INTER_SIZE
    
    return perf

def init_env():
    torch.set_printoptions(linewidth=3000, sci_mode=False, edgeitems=8, )
    torch.set_default_device('cuda')
    torch.manual_seed(0)

def test_acc(TILE_M=32, TILE_N=64, HIDDEN_SIZE=4096, INTER_SIZE=2048, TP=8):
    init_env()
    #entry_common('aiter', batch=[8192], prec=[torch.float4_e2m1fn_x2], TILE_M=128, TILE_N=128, run_count=2, HIDDEN_SIZE=4096, INTER_SIZE=2048, TP=8)
    #entry_common('mxn_splitk_2s', batch=[16], prec=[torch.float4_e2m1fn_x2], TILE_M=TILE_M, TILE_N=TILE_N, HIDDEN_SIZE=HIDDEN_SIZE, INTER_SIZE=INTER_SIZE, TP=TP, run_count=0)

    #entry_common('mxn_2s', batch=[8192], test_fp8=False, TILE_M=128, TILE_N=128, run_count=0)
    #assert 0,"========================"
    batch = [16,32,64,128]
    batch += list(range(128, 256))
    # TILE_M/N is configurable
    entry_common('mxn_splitk_2s', batch=batch, prec=[get_fp8type()], TILE_M=TILE_M, TILE_N=TILE_N, HIDDEN_SIZE=HIDDEN_SIZE, INTER_SIZE=INTER_SIZE, TP=TP, run_count=0)

def show_perf(perf):
    print('\nsummary:')
    for kernel, vals in perf.items():
        for prec, vals_ in vals.items():
            for b, data in vals_.items():
                print(f'{kernel}[{prec:<4} B={b:<4}]: {data["latency"]:5.0f} us, {data["bw"]:6.1f} GB/s, {data["flops"]:4.1f} tflops')

@pytest.mark.parametrize("batch", [[16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]])
def test_perf(batch, TILE_M=32, TILE_N=64, HIDDEN_SIZE=4096, INTER_SIZE=2048, TP=8):
    init_env()
    perf = {}
    # perf.update(entry_common('aiter', batch, prec=[torch.bfloat16, get_fp8type(), get_fp4type_if_valid()], HIDDEN_SIZE=HIDDEN_SIZE, INTER_SIZE=INTER_SIZE, TP=TP, TILE_M=TILE_M, TILE_N=TILE_N))
    # TODO: support fp8
    # TILE_M/N is configurable
    perf.update(entry_common('mxn_splitk_2s', batch=batch, prec=[get_fp8type()], TILE_M=TILE_M, TILE_N=TILE_N, HIDDEN_SIZE=HIDDEN_SIZE, INTER_SIZE=INTER_SIZE, TP=TP))
    show_perf(perf)

if __name__ == '__main__':
    TILE_M = 16
    TILE_N = 128
    HIDDEN_SIZE = 6144
    INTER_SIZE = 2560
    TP = 4
    test_acc(TILE_M=TILE_M, TILE_N=TILE_N, HIDDEN_SIZE=HIDDEN_SIZE, INTER_SIZE=INTER_SIZE, TP=TP)
    # batch = [16]
    batch = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    test_perf(batch, TILE_M=TILE_M, TILE_N=TILE_N, HIDDEN_SIZE=HIDDEN_SIZE, INTER_SIZE=INTER_SIZE, TP=TP)
