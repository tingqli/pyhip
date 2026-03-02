import torch
import triton
import triton.language as tl
from functools import cache

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from triton.experimental.gluon.language.amd.cdna3 import (
        sched_barrier as _amd_iglp_sched_barrier,
    )

from triton.experimental.gluon.language.amd.cdna3 import (
    sched_group_barrier as _amd_iglp_sched_group_barrier,
)

@cache
def moe_2stage_splitk_gateup(BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N):
    num_warps = 4
    BLOCK_TILE_SIZE_K: gl.constexpr = gl.constexpr(4 * 8 * num_warps)

    assert BLOCK_TILE_SIZE_M.bit_count() == 1, "BLOCK_TILE_SIZE_M must be power of 2"
    assert BLOCK_TILE_SIZE_N.bit_count() == 1, "BLOCK_TILE_SIZE_N must be power of 2"
    assert BLOCK_TILE_SIZE_M % 16 == 0, "BLOCK_TILE_SIZE_M must be multiple of 16"
    assert BLOCK_TILE_SIZE_N % 64 == 0, "BLOCK_TILE_SIZE_N must be multiple of 64"

    # input a layout
    # 2**3=8 elememts per thread(aka 8*sizeof(bf16)=16 bytes)
    reg_bases = [[0, 1], [0, 2], [0, 4]]
    m = BLOCK_TILE_SIZE_M // 16
    bit_pos = 16
    while m // 2:
        reg_bases.append([bit_pos, 0])
        bit_pos *= 2
        m = m // 2
    # thread:[1, 8(bf16)], lane: [16, 4], wave: [1, 4], rep thread: [M/16, 1]
    mem_a_layout: gl.constexpr = gl.DistributedLinearLayout(
                                reg_bases=reg_bases,
                                lane_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [0, 8], [0, 16]],
                                warp_bases=[[0, 32], [0, 64]],
                                block_bases=[],
                                shape=[BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_K])
    # lds layout
    # process 16x32 tile each
    lds_layout_write: gl.constexpr = gl.SwizzledSharedLayout(vec=4, per_phase=1, max_phase=16, order=[3, 2, 1, 0])
    reg_bases=[[0, 0, 0, 1], [1, 0, 0, 0], [2, 0, 0, 0]]
    n = BLOCK_TILE_SIZE_N // 2 // 32
    bit_pos = 1
    while n // 2:
        reg_bases.append([0, bit_pos, 0, 0])
        bit_pos *= 2
        n = n // 2
    m = BLOCK_TILE_SIZE_M // 16
    bit_pos = 16
    while m // 2:
        reg_bases.append([0, 0, bit_pos, 0])
        bit_pos *= 2
        m = m // 2
    # thread: [4wave, 2], lane: [4, 16], wave[4, 1], rep thread: [N/64, M/2]
    lds_layout_read: gl.constexpr = gl.DistributedLinearLayout(
                                reg_bases=reg_bases,
                                lane_bases=[[0, 0, 0, 2], [0, 0, 0, 4], [0, 0, 0, 8], [0, 0, 0, 16], [0, 0, 1, 0], [0, 0, 2, 0]],
                                warp_bases=[[0, 0, 4, 0], [0, 0, 8, 0]],
                                block_bases=[],
                                shape=[num_warps, BLOCK_TILE_SIZE_N // 2 // 32, BLOCK_TILE_SIZE_M, 32])

    @gluon.jit
    def moe_up(p_input,            # bf16 [M, K]
               p_weight,           # bf16 [K/8 * 16 * 8, N/16]
               p_output,           # bf16 [M, N]
               # sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids
               p_sorted_ids,
               p_sorted_weights,
               p_sorted_expert_ids,
               p_num_valid_ids,
               p_w_scale,
               M,
               weight_dtype: gl.constexpr,
               TOPK: gl.constexpr,
               K: gl.constexpr,
               N: gl.constexpr,
               BLOCK_TILE_SIZE_M: gl.constexpr,
               BLOCK_TILE_SIZE_N: gl.constexpr,
               num_warps: gl.constexpr = 4,
               ):
        tile_n = gl.program_id(0)
        e_idx = gl.program_id(1)
        sorted_id_layout: gl.constexpr = gl.BlockedLayout(size_per_thread=[1, 1],
                                                        threads_per_warp=[16, 4],
                                                        warps_per_cta=[1, num_warps],
                                                        order=[0, 1])
        sorted_id_offsets = e_idx * BLOCK_TILE_SIZE_M + gl.arange(0, BLOCK_TILE_SIZE_M, layout=gl.SliceLayout(1, sorted_id_layout))[:, None]
        sorted_ids = gl.amd.cdna3.buffer_load(p_sorted_ids, sorted_id_offsets)
        num_valid_ids = gl.load(p_num_valid_ids)
        sorted_expert_id = gl.load(p_sorted_expert_ids + e_idx)
        if e_idx * BLOCK_TILE_SIZE_M >= num_valid_ids:
            return
        p_weight = p_weight + sorted_expert_id * K * N
        if weight_dtype == torch.bfloat16:
            mem_b_layout: gl.constexpr = gl.BlockedLayout(size_per_thread=[8, 1],
                                        threads_per_warp=[64, 1],
                                        warps_per_cta=[num_warps, 1],
                                        order=[0, 1])
        sorted_ids_org = sorted_ids
        sorted_ids = gl.convert_layout(sorted_ids.reshape(BLOCK_TILE_SIZE_M), gl.SliceLayout(1, mem_a_layout), assert_trivial=True)
        sorted_topk = sorted_ids >> 24
        sorted_ids = sorted_ids & 0xffffff
        sorted_topk = gl.where(sorted_topk < TOPK, sorted_topk, 0)
        sorted_ids = gl.where(sorted_ids < M, sorted_ids, 0)
        mem_a_offsets = (sorted_ids * K)[:, None] + \
                        gl.arange(0, BLOCK_TILE_SIZE_K, layout=gl.SliceLayout(0, mem_a_layout))[None, :]
        mem_b_offsets_top = tile_n * BLOCK_TILE_SIZE_N // 2 * K + gl.arange(0, BLOCK_TILE_SIZE_N // 2 // 16, layout=gl.SliceLayout(0, mem_b_layout))[None, :] * (K // 8 * 16 * 8) + \
                        gl.arange(0, BLOCK_TILE_SIZE_K * 16, layout=gl.SliceLayout(1, mem_b_layout))[:, None]
        mem_b_offsets_bot = N // 2 * K + tile_n * BLOCK_TILE_SIZE_N // 2 * K + gl.arange(0, BLOCK_TILE_SIZE_N // 2 // 16, layout=gl.SliceLayout(0, mem_b_layout))[None, :] * (K // 8 * 16 * 8) + \
                        gl.arange(0, BLOCK_TILE_SIZE_K * 16, layout=gl.SliceLayout(1, mem_b_layout))[:, None]
        c_layout: gl.constexpr = gl.amd.AMDMFMALayout(version=4,
                                                    instr_shape=[16, 16, 32],
                                                    transposed=True,
                                                    tiles_per_warp=[1, BLOCK_TILE_SIZE_M // 16, BLOCK_TILE_SIZE_N // 16 // 2],
                                                    warps_per_cta=[num_warps, 1, 1])
        a_fma_layout: gl.constexpr = gl.DotOperandLayout(0, c_layout, k_width=8)
        b_fma_layout: gl.constexpr = gl.DotOperandLayout(1, c_layout, k_width=8)
        acc_top = gl.zeros((num_warps, BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N // 2), dtype=gl.float32, layout=c_layout)
        acc_bot = gl.zeros((num_warps, BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N // 2), dtype=gl.float32, layout=c_layout)
        for k_start in gl.static_range(0, K, BLOCK_TILE_SIZE_K):
            a_offsets = mem_a_offsets + k_start
            a = gl.amd.cdna3.buffer_load(p_input, a_offsets)
            b_offsets_top = mem_b_offsets_top + (k_start // 8) * 16 * 8
            b_offsets_bot = mem_b_offsets_bot + (k_start // 8) * 16 * 8
            if weight_dtype == torch.bfloat16:
                b_top = gl.amd.cdna3.buffer_load(p_weight, b_offsets_top)
                b_bot = gl.amd.cdna3.buffer_load(p_weight, b_offsets_bot)
            else:
                assert 0, "only bf16 weight is supported in this kernel"
            a = a.reshape(BLOCK_TILE_SIZE_M, num_warps, 4, 8).permute(1, 0, 2, 3).reshape(num_warps, BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_K // num_warps)
            a_fma = gl.convert_layout(a, a_fma_layout, assert_trivial=True)
            b_top = b_top.reshape(num_warps, 4, 16, 8, BLOCK_TILE_SIZE_N // 16 // 2).permute(0, 1, 3, 4, 2).reshape(num_warps, BLOCK_TILE_SIZE_K // num_warps, BLOCK_TILE_SIZE_N // 2)
            b_fma_top = gl.convert_layout(b_top, b_fma_layout, assert_trivial=True)
            b_bot = b_bot.reshape(num_warps, 4, 16, 8, BLOCK_TILE_SIZE_N // 16 // 2).permute(0, 1, 3, 4, 2).reshape(num_warps, BLOCK_TILE_SIZE_K // num_warps, BLOCK_TILE_SIZE_N // 2)
            b_fma_bot = gl.convert_layout(b_bot, b_fma_layout, assert_trivial=True)
            acc_top = gl.amd.cdna4.mfma(a_fma, b_fma_top, acc_top)
            acc_bot = gl.amd.cdna4.mfma(a_fma, b_fma_bot, acc_bot)

        # (num_warps, BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N // 2) --> (BLOCK_TILE_SIZE_M, num_warps, BLOCK_TILE_SIZE_N // 2 // 32, 32) -->
        #   (num_warps, BLOCK_TILE_SIZE_N // 2 // 32, BLOCK_TILE_SIZE_M, 32)
        acc_top = acc_top.reshape(num_warps, BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N // 2 // 32, 32).permute(0, 2, 1, 3)
        acc_bot = acc_bot.reshape(num_warps, BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N // 2 // 32, 32).permute(0, 2, 1, 3)
        lds_c = gl.allocate_shared_memory(gl.float32, [num_warps, BLOCK_TILE_SIZE_N // 2 // 32, BLOCK_TILE_SIZE_M, 32], layout=lds_layout_write)
        # rocprof-compute report no bank conflict, TODO: verify
        #gl.static_print('bank conflict =', gl.bank_conflicts(acc_top.type, lds_c.type))
        lds_c.store(acc_top)
        cur_acc_top = lds_c.load(layout=lds_layout_read).permute(0, 2, 1, 3).reshape(num_warps, BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N // 2)
        lds_c.store(acc_bot)
        cur_acc_bot = lds_c.load(layout=lds_layout_read).permute(0, 2, 1, 3).reshape(num_warps, BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N // 2)
        # -> [BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N]
        cur_acc_top = cur_acc_top.sum(axis=0)
        cur_acc_bot = cur_acc_bot.sum(axis=0)
        log2_exp1: gl.constexpr = -1.4426950408889634
        acc = (cur_acc_top * (1.0 / (1.0 + gl.exp2(cur_acc_top * log2_exp1)))) * cur_acc_bot
        acc:gl.tensor = acc.to(gl.bfloat16)

        mem_c_layout: gl.constexpr = gl.BlockedLayout(
            size_per_thread=[1, 2],
            threads_per_warp=[4, 16],
            warps_per_cta=[num_warps, 1],
            order=[1, 0],
        )
        sorted_ids_org = gl.convert_layout(sorted_ids_org.reshape(BLOCK_TILE_SIZE_M), gl.SliceLayout(1, mem_c_layout), assert_trivial=False)
        sorted_ids = sorted_ids_org & 0xffffff
        sorted_topk = sorted_ids_org >> 24
        out_offsets_n = (tile_n * BLOCK_TILE_SIZE_N // 2 + gl.arange(0, BLOCK_TILE_SIZE_N // 2, layout=gl.SliceLayout(0, mem_c_layout)))
        out_offsets = sorted_ids[:, None] * N // 2 * TOPK + sorted_topk[:, None] * N // 2 + out_offsets_n[None, :]
        mask = (sorted_ids < M)[:, None]
        gl.amd.cdna3.buffer_store(gl.convert_layout(acc, mem_c_layout, assert_trivial=True), p_output, out_offsets, mask=mask)
    
    return moe_up

@cache
def moe_2stage_splitk_down(
                           BLOCK_TILE_SIZE_M,
                           BLOCK_TILE_SIZE_N,
                          ):
    num_warps = 1
    BLOCK_TILE_SIZE_K: gl.constexpr = gl.constexpr(4 * 8 * num_warps)
    # input a layout
    # 2**3=8 elememts per thread(aka 8*sizeof(bf16)=16 bytes)
    reg_bases = [[0, 1], [0, 2], [0, 4]]
    assert BLOCK_TILE_SIZE_M.bit_count() == 1, "BLOCK_TILE_SIZE_M must be power of 2"
    assert BLOCK_TILE_SIZE_N.bit_count() == 1, "BLOCK_TILE_SIZE_N must be power of 2"
    assert BLOCK_TILE_SIZE_M % 16 == 0, "BLOCK_TILE_SIZE_M must be multiple of 16"
    assert BLOCK_TILE_SIZE_N % 32 == 0, "BLOCK_TILE_SIZE_N must be multiple of 32"
    m = BLOCK_TILE_SIZE_M // 16
    bit_pos = 16
    while m // 2:
        reg_bases.append([bit_pos, 0])
        bit_pos *= 2
        m = m // 2
    # thread:[1, 8(bf16)], lane: [16, 4], wave: [1, 1], rep thread: [M/16, 1]
    mem_a_layout: gl.constexpr = gl.DistributedLinearLayout(
                                reg_bases=reg_bases,
                                lane_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [0, 8], [0, 16]],
                                warp_bases=[],
                                block_bases=[],
                                shape=[BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_K])
    # output layout
    reg_bases: gl.constexpr = [[0, 1], [0, 2], [0, 16]]
    n = BLOCK_TILE_SIZE_N // 32
    bit_pos = 32
    while n // 2:
        reg_bases.append([0, bit_pos])
        bit_pos *= 2
        n = n // 2
    m = BLOCK_TILE_SIZE_M // 16
    bit_pos = 16
    while m // 2:
        reg_bases.append([bit_pos, 0])
        bit_pos *= 2
        m = m // 2
    # thread:[1, 4(bf16)], lane: [16, 4], wave: [1, 1], rep thread: [M/16, N/32]
    mem_c_layout: gl.constexpr = gl.DistributedLinearLayout(
                                reg_bases=reg_bases,
                                lane_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]],
                                warp_bases=[],
                                block_bases=[],
                                shape=[BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N],)

    @gluon.jit
    def moe_down(p_input,            # bf16 [M, K]
                 p_weight,           # bf16 [K/8 * 16 * 8, N/16]
                 p_output,           # bf16 [M, N]
                 # sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids
                 p_sorted_ids,
                 p_sorted_weights,
                 p_sorted_expert_ids,
                 p_num_valid_ids,
                 p_w_scale,
                 M,
                 weight_dtype: gl.constexpr,
                 TOPK: gl.constexpr,
                 K: gl.constexpr,
                 N: gl.constexpr,
                 BLOCK_TILE_SIZE_M: gl.constexpr,
                 BLOCK_TILE_SIZE_N: gl.constexpr,
                 num_warps: gl.constexpr = 1,
                 waves_per_eu: gl.constexpr = 4,):
        tile_n = gl.program_id(0)
        e_idx = gl.program_id(1)
        sorted_id_layout: gl.constexpr = gl.BlockedLayout(size_per_thread=[1, 1],
                                                        threads_per_warp=[16, 4],
                                                        warps_per_cta=[num_warps, 1],
                                                        order=[0, 1])
        sorted_id_offsets = e_idx * BLOCK_TILE_SIZE_M + gl.arange(0, BLOCK_TILE_SIZE_M, layout=gl.SliceLayout(1, sorted_id_layout))[:, None]
        sorted_ids = gl.amd.cdna3.buffer_load(p_sorted_ids, sorted_id_offsets)
        num_valid_ids = gl.load(p_num_valid_ids)
        sorted_expert_id = gl.load(p_sorted_expert_ids + e_idx)
        if e_idx * BLOCK_TILE_SIZE_M >= num_valid_ids:
            return
        p_weight = p_weight + sorted_expert_id * K * N
        sorted_weights_offsets = sorted_id_offsets
        sorted_weights = gl.amd.cdna3.buffer_load(p_sorted_weights, sorted_weights_offsets)
        if weight_dtype == torch.bfloat16:
            mem_b_layout: gl.constexpr = gl.BlockedLayout(size_per_thread=[8, 1],
                                        threads_per_warp=[64, 1],
                                        warps_per_cta=[num_warps, 1],
                                        order=[0, 1])
        sorted_ids_org = sorted_ids
        sorted_ids = gl.convert_layout(sorted_ids.reshape(BLOCK_TILE_SIZE_M), gl.SliceLayout(1, mem_a_layout), assert_trivial=True)
        sorted_topk = sorted_ids >> 24
        sorted_ids = sorted_ids & 0xffffff
        sorted_topk = gl.where(sorted_topk < TOPK, sorted_topk, 0)
        sorted_ids = gl.where(sorted_ids < M, sorted_ids, 0)
        mem_a_offsets = (sorted_ids * K * TOPK + sorted_topk * K)[:, None] + \
                        gl.arange(0, BLOCK_TILE_SIZE_K, layout=gl.SliceLayout(0, mem_a_layout))[None, :]
        mem_b_offsets = tile_n * BLOCK_TILE_SIZE_N * K + gl.arange(0, BLOCK_TILE_SIZE_N // 16, layout=gl.SliceLayout(0, mem_b_layout))[None, :] * (K // 8 * 16 * 8) + \
                        gl.arange(0, BLOCK_TILE_SIZE_K * 16, layout=gl.SliceLayout(1, mem_b_layout))[:, None]
        c_layout: gl.constexpr = gl.amd.AMDMFMALayout(version=4, 
                                                    instr_shape=[16, 16, 32], 
                                                    transposed=True,
                                                    tiles_per_warp=[1, BLOCK_TILE_SIZE_M // 16, BLOCK_TILE_SIZE_N // 16],
                                                    warps_per_cta=[num_warps, 1, 1])
        a_fma_layout: gl.constexpr = gl.DotOperandLayout(0, c_layout, k_width=8)
        b_fma_layout: gl.constexpr = gl.DotOperandLayout(1, c_layout, k_width=8)
        acc = gl.zeros((num_warps, BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N), dtype=gl.float32, layout=c_layout)
        # prefetch first tile
        a = gl.amd.cdna3.buffer_load(p_input, mem_a_offsets)
        if weight_dtype == torch.bfloat16:
            b = gl.amd.cdna3.buffer_load(p_weight, mem_b_offsets)
        # https://github.com/llvm/llvm-project/blob/84ce7b1f1d2997d1880f9b24ee6e6a75d4045e0a/llvm/include/llvm/IR/IntrinsicsAMDGPU.td#L350-L362
        s_vemm: gl.constexpr = 0x0010
        s_vmem_read: gl.constexpr = 0x0020
        s_vmem_write: gl.constexpr = 0x0040
        s_mfma: gl.constexpr = 0x0008
        s_ds_read: gl.constexpr = 0x0100
        s_ds_write: gl.constexpr = 0x0200
        s_valu: gl.constexpr = 0x0002
        s_salu: gl.constexpr = 0x0004
        gl.allocate_shared_memory(gl.float32, [10, 1024], layout=gl.SwizzledSharedLayout(4, 1, 8, [0, 1]))

        for k_start in gl.static_range(0, K - BLOCK_TILE_SIZE_K, BLOCK_TILE_SIZE_K):
            _amd_iglp_sched_barrier(0)
            a_offsets = mem_a_offsets + k_start + BLOCK_TILE_SIZE_K
            next_a = gl.amd.cdna3.buffer_load(p_input, a_offsets)
            b_offsets = mem_b_offsets + (k_start // 8) * 16 * 8 + BLOCK_TILE_SIZE_K // 8 * 16 * 8
            if weight_dtype == torch.bfloat16:
                next_b = gl.amd.cdna3.buffer_load(p_weight, b_offsets)
            else:
                assert 0, "only bf16 weight is supported in this kernel"
            a = a.reshape(BLOCK_TILE_SIZE_M, num_warps, 4, 8).permute(1, 0, 2, 3).reshape(num_warps, BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_K // num_warps)
            a_fma = gl.convert_layout(a, a_fma_layout, assert_trivial=True)
            b = b.reshape(num_warps, 4, 16, 8, BLOCK_TILE_SIZE_N // 16).permute(0, 1, 3, 4, 2).reshape(num_warps, BLOCK_TILE_SIZE_K // num_warps, BLOCK_TILE_SIZE_N)
            b_fma = gl.convert_layout(b, b_fma_layout, assert_trivial=True)
            acc = gl.amd.cdna4.mfma(a_fma, b_fma, acc)
            a = next_a
            b = next_b
            _amd_iglp_sched_group_barrier(s_vmem_read, BLOCK_TILE_SIZE_M // 16 + BLOCK_TILE_SIZE_N // 16, 0)
            _amd_iglp_sched_group_barrier(s_mfma, BLOCK_TILE_SIZE_M // 16 * BLOCK_TILE_SIZE_N // 16, 0)

        _amd_iglp_sched_barrier(0)
        # tail
        a = a.reshape(BLOCK_TILE_SIZE_M, num_warps, 4, 8).permute(1, 0, 2, 3).reshape(num_warps, BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_K // num_warps)
        a_fma = gl.convert_layout(a, a_fma_layout, assert_trivial=True)
        b = b.reshape(num_warps, 4, 16, 8, BLOCK_TILE_SIZE_N // 16).permute(0, 1, 3, 4, 2).reshape(num_warps, BLOCK_TILE_SIZE_K // num_warps, BLOCK_TILE_SIZE_N)
        b_fma = gl.convert_layout(b, b_fma_layout, assert_trivial=True)
        acc = gl.amd.cdna4.mfma(a_fma, b_fma, acc)

        sorted_ids_org = gl.convert_layout(sorted_ids_org.reshape(BLOCK_TILE_SIZE_M), gl.SliceLayout(1, mem_c_layout), assert_trivial=False)
        sorted_ids = sorted_ids_org & 0xffffff
        out_offsets_m = sorted_ids
        out_offsets_n = (tile_n * BLOCK_TILE_SIZE_N + gl.arange(0, BLOCK_TILE_SIZE_N, layout=gl.SliceLayout(0, mem_c_layout)))
        out_offsets = out_offsets_m[:, None] * N + out_offsets_n[None, :]
        acc = acc.reshape(BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N)
        acc = gl.convert_layout(acc, mem_c_layout, assert_trivial=True)
        sorted_weights = gl.convert_layout(sorted_weights.reshape(BLOCK_TILE_SIZE_M), gl.SliceLayout(1, mem_c_layout), assert_trivial=True)[:, None]
        acc = (acc * sorted_weights).to(gl.bfloat16)
        mask = (out_offsets_m < M)[:, None]
        gl.atomic_add(p_output + out_offsets, acc, sem='relaxed', mask=mask)
        # gl.amd.cdna4.buffer_atomic_add(p_output, out_offsets, acc, sem='relaxed', mask=mask)
        for _ in gl.static_range(BLOCK_TILE_SIZE_M * BLOCK_TILE_SIZE_N // 64 // 2):
            _amd_iglp_sched_group_barrier(s_vemm, 1, 0)
            _amd_iglp_sched_group_barrier(s_valu, 4, 0)

    return moe_down

#####################################################################
import pytest
from typing import Optional
from pyhip import cudaPerf, jit, JIT, torchPerf

def div_up(a, b):
    return (a + b - 1) // b

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

import aiter
from aiter.utility import fp4_utils

def _run_batch(kernel_type, B=1, weight_type=torch.bfloat16, TILE_M=16, TILE_N=32, run_count=10, HIDDEN_SIZE=2048, INTER_SIZE=1024, TOPK=8, E=128, TP=8):
    from aiter.ops.shuffle import shuffle_weight
    INTER_SIZE_TP = INTER_SIZE // TP
    BUF_COPY = 32
    hidden_states = (torch.randn([BUF_COPY, B, HIDDEN_SIZE], dtype=torch.bfloat16) + 1)*0.001
    if weight_type == torch.bfloat16:
        w_ = torch.randn([E, INTER_SIZE_TP * 2, HIDDEN_SIZE], dtype=weight_type)
        w1_ref = w_
        w_shuffled = shuffle_weight(w_).reshape(E, INTER_SIZE_TP * 2 // 16, -1)
        w1 = [w_shuffled.clone() for _ in range(BUF_COPY)]
        w_ = torch.randn([E, HIDDEN_SIZE, INTER_SIZE_TP], dtype=weight_type)
        w2_ref = w_
        w_shuffled = shuffle_weight(w_).reshape(E, HIDDEN_SIZE // 16, -1)
        w2 = [w_shuffled.clone() for _ in range(BUF_COPY)]
        w1_scale = [None] * BUF_COPY
        w2_scale = [None] * BUF_COPY
    elif weight_type == torch.float4_e2m1fn_x2:
        import aiter
        from aiter.utility import fp4_utils
        from aiter.ops.shuffle import shuffle_weight
        # w_ = torch.randn([E, INTER_SIZE_TP * 2, HIDDEN_SIZE], dtype=torch.bfloat16)
        w_ = torch.randn([E, INTER_SIZE_TP * 2, HIDDEN_SIZE], dtype=torch.bfloat16)
        w1_qt, w1_qt_scale_ = aiter.get_torch_quant(aiter.QuantType.per_1x32)(w_, quant_dtype=weight_type)

        #w1_qt_scale_[...] = 1.0
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
        w_ = torch.randn([E, HIDDEN_SIZE, INTER_SIZE_TP], dtype=torch.bfloat16)
        w2_qt, w2_qt_scale_ = aiter.get_torch_quant(aiter.QuantType.per_1x32)(w_, quant_dtype=weight_type)
        #w2_qt_scale_[...] = 1.0

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
    else:
        import aiter
        torch_quant = aiter.get_torch_quant(aiter.QuantType.per_Token)
        w_ = torch.randn([E, INTER_SIZE_TP * 2, HIDDEN_SIZE], dtype=torch.bfloat16)
        w1_qt, w1_qt_scale = torch_quant(w_, quant_dtype=weight_type)
        w1_ref = (w1_qt.to(dtype=torch.bfloat16) * w1_qt_scale).to(dtype=torch.bfloat16)
        w1 = [w1_qt.clone() for _ in range(BUF_COPY)]
        w1_scale = [w1_qt_scale.clone() for _ in range(BUF_COPY)]
        w_ = torch.randn([E, HIDDEN_SIZE, INTER_SIZE_TP], dtype=torch.bfloat16)
        w2_qt, w2_qt_scale = torch_quant(w_, quant_dtype=weight_type)
        w2_ref = (w2_qt.to(dtype=torch.bfloat16) * w2_qt_scale).to(dtype=torch.bfloat16)
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
    if weight_type == torch.bfloat16:
        ele_size = 2
    elif weight_type == torch.float4_e2m1fn_x2:
        ele_size = 0.5
    else:
        ele_size = 1
    mem_size = B * HIDDEN_SIZE * 2 + (HIDDEN_SIZE * INTER_SIZE_TP * 2 + HIDDEN_SIZE * INTER_SIZE_TP) * access_expert * ele_size

    import aiter
    from aiter.fused_moe import moe_sorting
    from aiter.utility import fp4_utils


    def run(hidden_states, w1, w2, topk_weight, topk_ids, w1_scale, w2_scale):
        B = hidden_states.shape[0]
        N1, K1 = INTER_SIZE_TP * 2, HIDDEN_SIZE
        N2, K2 = HIDDEN_SIZE, INTER_SIZE_TP
        gemm1_out = torch.empty([B, TOPK, N1 // 2], dtype=hidden_states.dtype, device=hidden_states.device)
        #print(topk_weight.shape, topk_weight.dtype)
        #assert 0
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
            x=moe_2stage_splitk_gateup(BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N)[(N1 // BLOCK_TILE_SIZE_N, grid)](
                                hidden_states, w1, gemm1_out, sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, w1_scale[0] if w1_scale is not None else None, B,
                                w1.dtype, TOPK, K1, N1, BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N, 4)
            # print(x.asm['amdgcn'])
            # assert 0
            x=moe_2stage_splitk_down(BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N)[(N2 // BLOCK_TILE_SIZE_N, grid)](
                                gemm1_out, w2, cur_out, sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, w2_scale[0] if w2_scale is not None else None, B,
                                w1.dtype, TOPK, K2, N2, BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N, num_warps=1)
            # print(x.asm['amdgcn'])
            # assert 0
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
        w1_qt_aiter = w1[0]
        w2_qt_aiter = w2[0]
        ref_out = get_torch_ref(hidden_states=hidden_states[0], w1=w1_ref, w2=w2_ref, topk_weight=topk_weight[0], topk_ids=topk_ids[0])
        # aiter_out = _run_aiter(hidden_states=hidden_states[0], w1=w1[0], w2=w2[0], topk_weight=topk_weight[0], topk_ids=topk_ids[0], w1_scale=w1_scale[0], w2_scale=w2_scale[0])
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
            print(ref_out)
            print(cur_out)
            idx = torch.where(torch.abs(ref_out - cur_out) > 0.03)
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

def entry_common(kernel_type, batch, prec=[torch.bfloat16], TILE_M=32, TILE_N=64, HIDDEN_SIZE=2048, INTER_SIZE=1024, TOPK=8, E=128, TP=8, run_count=10):
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
    batch = list(range(2, 64))
    # fix TILE_M=16, TILE_N=32
    batch += list(range(128, 256))
    batch += [i * 256 for i in range(1, 4)]
    batch += [i * 2048 for i in range(1, 5)]
    batch += list(range(2048 * 3, 2048 * 3 + 256))
    # TILE_M/N is configurable
    entry_common('mxn_splitk_2s', batch=batch, prec=[torch.bfloat16, get_fp8type(), get_fp4type_if_valid()], TILE_M=TILE_M, TILE_N=TILE_N, HIDDEN_SIZE=HIDDEN_SIZE, INTER_SIZE=INTER_SIZE, TP=TP, run_count=0)

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
    # TILE_M/N is configurable
    perf.update(entry_common('mxn_splitk_2s', batch=batch, prec=[torch.bfloat16], TILE_M=TILE_M, TILE_N=TILE_N, HIDDEN_SIZE=HIDDEN_SIZE, INTER_SIZE=INTER_SIZE, TP=TP))
    show_perf(perf)

if __name__ == '__main__':
    TILE_M = 16
    TILE_N = 128
    HIDDEN_SIZE = 4096
    INTER_SIZE = 1536
    TP = 1
    init_env()
    batch = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    batch = [16]
    test_perf(batch, TILE_M=TILE_M, TILE_N=TILE_N, HIDDEN_SIZE=HIDDEN_SIZE, INTER_SIZE=INTER_SIZE, TP=TP)
