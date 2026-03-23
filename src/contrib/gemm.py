import os
import random
from typing import Optional
from torch import Tensor

import pytest
os.environ['PYHIP_JIT_LOG'] = '0'
from pyhip import jit, JIT
import torch

from functools import cache
# work as package
# from .common.gemm import UGEMM
from .common.gemm_splitk import gemm_splitk


from .common.gemm import UGEMM
from .common.gemm_splitk import gemm_splitk

__all__ = [
    "gemm_splitk_jit",
    "gemm_splitk_wd",
]


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

@jit(with_debug_log=False)
def gemm_splitk_wd(J:JIT,
                   weight_dtype,
                   K,            # compile-time args
                   N,            # compile-time args
                   BLOCK_TILE_SIZE_M,
                   BLOCK_TILE_SIZE_N,
                   p_input:"void*",
                   p_weight:"void*",
                   p_output:"void*",
                   p_w_scale:"float*",
                   M:"int",):
    assert weight_dtype == torch.float4_e2m1fn_x2 or weight_dtype == torch.float8_e4m3fn or weight_dtype == torch.float8_e4m3fnuz
    num_split_k = 4
    if weight_dtype == torch.float4_e2m1fn_x2:
        assert BLOCK_TILE_SIZE_N % 32 == 0, f'due to scale is packed with [2*16, 8*32], current BLOCK_TILE_SIZE_N={BLOCK_TILE_SIZE_N} is not supported'
        assert K % 1024 == 0, f'will read (16*4)*4(wave) bytes once in main loop, aka 256*2=512 elements; to use packed scale in K dimension, will double read; current K={K} is not supported'
    elif weight_dtype == torch.float8_e4m3fn or weight_dtype == torch.float8_e4m3fnuz:
        # fp8 blockwise: K % (num_split_k * 128) == 0
        assert K % (num_split_k * 128) == 0, f'fp8 blockwise K{K} should be multiple of {num_split_k * 128}'
        assert BLOCK_TILE_SIZE_N % 32 == 0, f'current BLOCK_TILE_SIZE_N={BLOCK_TILE_SIZE_N} is not supported'
    assert BLOCK_TILE_SIZE_M % 16 == 0
    assert N % BLOCK_TILE_SIZE_N == 0

    def get_k_bytes(k_in_elements):
        if weight_dtype == torch.bfloat16:
            return k_in_elements * J.sizeof_bf16
        elif weight_dtype == torch.float4_e2m1fn_x2:
            return k_in_elements // 2
        elif weight_dtype == torch.float8_e4m3fn or weight_dtype == torch.float8_e4m3fnuz:
            return k_in_elements
    stride_A = K * J.sizeof_bf16
    stride_B = get_k_bytes(K)
    stride_C = N * J.sizeof_bf16

    A_vert = BLOCK_TILE_SIZE_M // 16
    B_horz = BLOCK_TILE_SIZE_N // 16

    C_reg = J.gpr(B_horz, A_vert, 4, "vf32")

    C_reg[:] = 0

    # one WG per CU,  4 waves split on K
    lane_mod_16 = get_lane_id_mod(J, 16)
    lane_div_16 = get_lane_id_div(J, 16)
    warp_id = J.warp_id
    J.debug_setup((J.blockIdx.x[0] == 0) & (J.blockIdx.y[0] == 0) & (warp_id[0] == 0))

    voffset_b = J.gpr(B_horz, 'vu32')
    if weight_dtype == torch.float4_e2m1fn_x2:
        # shffled
        voffset_b[0] = J.blockIdx.x * (BLOCK_TILE_SIZE_N * stride_B) + J.lane_id * 16 + J.warp_id * (16 * get_k_bytes(K) // 4)
        for m in range(1, B_horz):
            voffset_b[m] = voffset_b[0] + get_k_bytes(K) * 16 * m
    elif weight_dtype == torch.float8_e4m3fn or weight_dtype == torch.float8_e4m3fnuz:
        # shffled
        voffset_b[0] = BLOCK_TILE_SIZE_N * J.blockIdx.x * stride_B + J.threadIdx.x * 16
        for m in range(1, B_horz):
            voffset_b[m] = voffset_b[0] + get_k_bytes(K) * 16 * m
    else:
        assert 0

    buff_a = J.Buffer(p_input, M * (K * J.sizeof_bf16))
    buff_c = J.Buffer(p_output, M * (N * J.sizeof_bf16))

    voffset_a = J.gpr(A_vert, 'vu32')
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
    voffset_a[0] = J.gpr((lane_mod_16 + J.blockIdx.y * BLOCK_TILE_SIZE_M) * stride_A) + lane_div_16 * (a_element_num_per_thread * J.sizeof_bf16) + J.warp_id * (K // 4 * J.sizeof_bf16)
    for m in range(1, A_vert):
        voffset_a[m] = voffset_a[0] + 16 * stride_A * m

    voffset_scale = None
    if weight_dtype == torch.float4_e2m1fn_x2:
        # 32 rows shared in a uint32
        voffset_scale = J.gpr(B_horz // 2, 'vu32')
        # the group of fp4 is 32 elements, and scale will align to 8 groups
        k_scale_stride = div_up(div_up(K, 32), 8) * 8
        p_w_scale[:] += BLOCK_TILE_SIZE_N * k_scale_stride * J.blockIdx.x
        voffset_scale[0] = J.lane_id * J.sizeof_fp32 + J.warp_id * (k_scale_stride // 8 // 4 * 64 * J.sizeof_fp32)
        for m in range(1, B_horz // 2):
            voffset_scale[m] = voffset_scale[0] + 32 * k_scale_stride * m
    elif weight_dtype == torch.float8_e4m3fn or weight_dtype == torch.float8_e4m3fnuz:
        # fp8 blockwise: scale layout is [N//128, K//128]
        k_scale_stride = (K // 128) * J.sizeof_fp32
        p_w_scale[:] += (BLOCK_TILE_SIZE_N * J.blockIdx.x // 128) * k_scale_stride
        voffset_scale = J.gpr(2, 'vu32')
        voffset_scale[0] = J.threadIdx.x //128 * 4
        voffset_scale[1] = voffset_scale[0] 

    buff_b = J.Buffer(p_weight, N * get_k_bytes(K))

    # gemm_splitk(J, weight_dtype, K, N, num_split_k,
    #             buff_a, buff_b, p_w_scale,
    #             voffset_a, voffset_b, voffset_scale, C_reg, BLOCK_TILE_SIZE_N=BLOCK_TILE_SIZE_N, BLOCK_TILE_SIZE_M=BLOCK_TILE_SIZE_M, USE_FP4_SHUFFLE_WEIGHT=USE_FP4_SHUFFLE_WEIGHT, fp8_ptpc=fp8_ptpc)
    
    gemm_splitk(J, weight_dtype, K, N, num_split_k,
                buff_a, buff_b, p_w_scale,
                voffset_a, voffset_b, voffset_scale, C_reg, BLOCK_TILE_SIZE_N=BLOCK_TILE_SIZE_N, BLOCK_TILE_SIZE_M=BLOCK_TILE_SIZE_M, USE_FP4_SHUFFLE_WEIGHT=True, fp8_ptpc=False)

    # split K
    lds_buff = J.LDSTensor([4 * BLOCK_TILE_SIZE_M, 32], torch.float)
    # each 32 N as a group
    n_groups = BLOCK_TILE_SIZE_N // 32
    vrow = J.gpr(lane_div_16 + warp_id * 4)
    c_offset = J.gpr(1, 'vu32')
    m_block_stride = J.gpr(1, 'vu32')
    m_block_stride[0] =BLOCK_TILE_SIZE_M * stride_C
    c_offset[0] = J.threadIdx.x[0] // 16 * stride_C + lane_mod_16 * 4 + J.blockIdx.y * m_block_stride + J.blockIdx.x * (BLOCK_TILE_SIZE_N * J.sizeof_bf16)
    s_c_offset = J.gpr(1, 'su32')
    s_c_offset[0] = 0
    for n in range(n_groups):
        for m in range(A_vert):
            col = lane_div_16
            row = lane_mod_16
            col = (row ^ col) % 8
            lds_buff.write("b128", C_reg[2 * n + 0, m], lane_mod_16 + warp_id * 16 + m * 64, col * 4)
            col = lane_div_16 + 4
            col = (row ^ col) % 8
            lds_buff.write("b128", C_reg[2 * n + 1, m], lane_mod_16 + warp_id * 16 + m * 64, col * 4)

        J.s_waitcnt(mod=f"lgkmcnt(0)")
        J.s_barrier()
        # each wave reduce 4 rows once
        for m in range(A_vert):
            c_in_wave = J.gpr(4, 2, "vf32", align=2)
            # interleave
            col = lane_mod_16 // 2
            row = vrow
            col = (row ^ col) % 8
            lds_buff.read("b64", c_in_wave[0], vrow + (0 * 16 + m * 64), col * 4 + (lane_mod_16 & 1) * 2)
            lds_buff.read("b64", c_in_wave[1], vrow + (1 * 16 + m * 64), col * 4 + (lane_mod_16 & 1) * 2)
            lds_buff.read("b64", c_in_wave[2], vrow + (2 * 16 + m * 64), col * 4 + (lane_mod_16 & 1) * 2)
            lds_buff.read("b64", c_in_wave[3], vrow + (3 * 16 + m * 64), col * 4 + (lane_mod_16 & 1) * 2)

            J.s_waitcnt(mod=f"lgkmcnt(2)")
            J.v_pk_add_f32(c_in_wave[0], c_in_wave[0], c_in_wave[1])
            J.s_waitcnt(mod=f"lgkmcnt(0)")
            J.v_pk_add_f32(c_in_wave[2], c_in_wave[2], c_in_wave[3])
            J.v_pk_add_f32(c_in_wave[0], c_in_wave[0], c_in_wave[2])

            tmp = J.gpr(1, 'vu32')
            tmp[0] = c_offset[0] + m * 16 * stride_C + n * 32 * J.sizeof_bf16
            J.v_cvt_pk_bf16_f32(c_in_wave[0, 0], c_in_wave[0, 0], c_in_wave[0, 1])
            J.debug_log(c_in_wave[0, 0], torch_dtype=torch.bfloat16, message=f'{m=}{n=}')
            buff_c.store_dword(c_in_wave[0, 0], tmp, s_c_offset)
        J.s_barrier()


def gemm_splitk_jit(input: torch.Tensor,
                weight: torch.Tensor,
                output: torch.Tensor,
                weight_scale: torch.Tensor,
                b_preshuffle = True):
        if not b_preshuffle:
            # so far only support preshuffled weights
            return False
        M, K = input.shape
        N = weight.shape[0]
        
        def get_tile_mn(M):
            num_CU = torch.cuda.get_device_properties().multi_processor_count
            solutions = []
            for tile_m in [16, 32, 64]:
                for tile_n in [32, 64, 128]:
                    works = div_up(M, tile_m) * div_up(N, tile_n)
                    if works >= num_CU:
                        round = works // num_CU
                        reminder = works % num_CU
                        solutions.append((round, reminder, tile_m, tile_n))
                    else:
                        reminder = num_CU - works % num_CU
                        solutions.append((100000, reminder, tile_m, tile_n))
        # prefer less rounds; then less reminder
            TILE_M, TILE_N = sorted(solutions)[0][2:]
            return TILE_M, TILE_N
    
        BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N = get_tile_mn(M)
        gemm_splitk_wd([div_up(N, BLOCK_TILE_SIZE_N), div_up(M, BLOCK_TILE_SIZE_M)], [256],
                            weight.dtype, K, N, BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N,
                            input.data_ptr(), weight.data_ptr(), output.data_ptr(), weight_scale.data_ptr() if weight_scale is not None else 0, M)
        return True