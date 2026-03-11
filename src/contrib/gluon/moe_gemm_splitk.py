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

__all__ = [
    'moe_2stage_splitk_gateup', 'moe_2stage_splitk_down'
]

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
        sorted_ids = gl.convert_layout(sorted_ids, mem_a_layout, assert_trivial=True)
        sorted_topk = sorted_ids >> 24
        sorted_ids = sorted_ids & 0xffffff
        sorted_topk = gl.where(sorted_topk < TOPK, sorted_topk, 0)
        sorted_ids = gl.where(sorted_ids < M, sorted_ids, 0)
        mem_a_offsets = (sorted_ids * K) + \
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
        sorted_ids_org = gl.convert_layout(sorted_ids_org, mem_c_layout, assert_trivial=False)
        sorted_ids = sorted_ids_org & 0xffffff
        sorted_topk = sorted_ids_org >> 24
        out_offsets_n = (tile_n * BLOCK_TILE_SIZE_N // 2 + gl.arange(0, BLOCK_TILE_SIZE_N // 2, layout=gl.SliceLayout(0, mem_c_layout)))
        out_offsets = sorted_ids * N // 2 * TOPK + sorted_topk * N // 2 + out_offsets_n[None, :]
        mask = (sorted_ids < M)
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
    if 0:
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
    else:
        assert BLOCK_TILE_SIZE_N % 64 == 0, f'64 elements in N dimenstion will be better for write out'
        if BLOCK_TILE_SIZE_N == 64:
            mem_c_layout: gl.constexpr = gl.BlockedLayout(size_per_thread=[1, 2],
                                                        threads_per_warp=[2, 32],
                                                        warps_per_cta=[1, 1],
                                                        order=[1, 0])
        else:
            mem_c_layout: gl.constexpr = gl.BlockedLayout(size_per_thread=[1, 2],
                                                        threads_per_warp=[1, 64],
                                                        warps_per_cta=[1, 1],
                                                        order=[1, 0])

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
        sorted_ids = gl.convert_layout(sorted_ids, mem_a_layout, assert_trivial=True)
        sorted_topk = sorted_ids >> 24
        sorted_ids = sorted_ids & 0xffffff
        sorted_topk = gl.where(sorted_topk < TOPK, sorted_topk, 0)
        sorted_ids = gl.where(sorted_ids < M, sorted_ids, 0)
        mem_a_offsets = (sorted_ids * K * TOPK + sorted_topk * K) + \
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

        sorted_ids_org = gl.convert_layout(sorted_ids_org, mem_c_layout, assert_trivial=False)
        sorted_ids = sorted_ids_org & 0xffffff
        out_offsets_m = sorted_ids
        out_offsets_n = (tile_n * BLOCK_TILE_SIZE_N + gl.arange(0, BLOCK_TILE_SIZE_N, layout=gl.SliceLayout(0, mem_c_layout)))
        out_offsets = out_offsets_m * N + out_offsets_n[None, :]
        acc = acc.reshape(BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N)
        # `broadcasting in Triton is special: the operands for broadcast must come from a reduction of the same tensor.` from:
        #   https://github.com/triton-lang/triton/issues/8580
        sorted_weights = gl.convert_layout(sorted_weights, acc.type.layout, assert_trivial=True)
        acc = (acc * sorted_weights).to(gl.bfloat16)
        acc = gl.convert_layout(acc, mem_c_layout, assert_trivial=False)
        # the following will trigger lds to shuffle sorted_weights
        # sorted_weights = gl.convert_layout(sorted_weights, mem_c_layout, assert_trivial=False)
        # acc = (acc * sorted_weights).to(gl.bfloat16)
        mask = (out_offsets_m < M)
        #gl.atomic_add(p_output + out_offsets, acc, sem='relaxed', mask=mask)
        gl.amd.cdna4.buffer_atomic_add(p_output, out_offsets, acc, sem='relaxed', mask=mask)
        for _ in gl.static_range(BLOCK_TILE_SIZE_M * BLOCK_TILE_SIZE_N // 64 // 2):
            _amd_iglp_sched_group_barrier(s_vemm, 1, 0)
            _amd_iglp_sched_group_barrier(s_valu, 4, 0)

    # interleave 'load->mfma->write'
    @gluon.jit
    def moe_down_interleave(p_input,            # bf16 [M, K]
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
        # each step will compute 16 elements
        BLOCK_TILE_SIZE_N_SUB: gl.constexpr = 16
        mem_b_offsets = tile_n * BLOCK_TILE_SIZE_N * K + gl.arange(0, BLOCK_TILE_SIZE_N_SUB // 16, layout=gl.SliceLayout(0, mem_b_layout))[None, :] * (K // 8 * 16 * 8) + \
                        gl.arange(0, BLOCK_TILE_SIZE_K * 16, layout=gl.SliceLayout(1, mem_b_layout))[:, None]
        c_layout: gl.constexpr = gl.amd.AMDMFMALayout(version=4, 
                                                    instr_shape=[16, 16, 32], 
                                                    transposed=True,
                                                    tiles_per_warp=[1, BLOCK_TILE_SIZE_M // 16, BLOCK_TILE_SIZE_N_SUB // 16],
                                                    warps_per_cta=[num_warps, 1, 1])
        a_fma_layout: gl.constexpr = gl.DotOperandLayout(0, c_layout, k_width=8)
        b_fma_layout: gl.constexpr = gl.DotOperandLayout(1, c_layout, k_width=8)
        # https://github.com/llvm/llvm-project/blob/84ce7b1f1d2997d1880f9b24ee6e6a75d4045e0a/llvm/include/llvm/IR/IntrinsicsAMDGPU.td#L350-L362
        s_vemm: gl.constexpr = 0x0010
        s_vmem_read: gl.constexpr = 0x0020
        s_vmem_write: gl.constexpr = 0x0040
        s_mfma: gl.constexpr = 0x0008
        s_ds_read: gl.constexpr = 0x0100
        s_ds_write: gl.constexpr = 0x0200
        s_valu: gl.constexpr = 0x0002
        s_salu: gl.constexpr = 0x0004
        out_offsets_n = (tile_n * BLOCK_TILE_SIZE_N + gl.arange(0, BLOCK_TILE_SIZE_N_SUB, layout=gl.SliceLayout(0, mem_c_layout)))

        # prefetch first tile
        a = gl.amd.cdna3.buffer_load(p_input, mem_a_offsets)
        if weight_dtype == torch.bfloat16:
            mem_b_offsets_sub = mem_b_offsets
            b = gl.amd.cdna3.buffer_load(p_weight, mem_b_offsets_sub)

        _amd_iglp_sched_barrier(0)
        for n in gl.static_range(BLOCK_TILE_SIZE_N // BLOCK_TILE_SIZE_N_SUB):
            acc = gl.zeros((num_warps, BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N_SUB), dtype=gl.float32, layout=c_layout)

            a_offsets = mem_a_offsets + BLOCK_TILE_SIZE_K
            b_offsets = mem_b_offsets_sub + BLOCK_TILE_SIZE_K // 8 * 16 * 8
            for k_start in gl.static_range(0, K - BLOCK_TILE_SIZE_K * 2, BLOCK_TILE_SIZE_K * 2):
                #_amd_iglp_sched_barrier(0)
                next_a = gl.amd.cdna3.buffer_load(p_input, a_offsets)
                if weight_dtype == torch.bfloat16:
                    next_b = gl.amd.cdna3.buffer_load(p_weight, b_offsets)
                else:
                    assert 0, "only bf16 weight is supported in this kernel"
                a = a.reshape(BLOCK_TILE_SIZE_M, num_warps, 4, 8).permute(1, 0, 2, 3).reshape(num_warps, BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_K // num_warps)
                a_fma = gl.convert_layout(a, a_fma_layout, assert_trivial=True)
                b = b.reshape(num_warps, 4, 16, 8, BLOCK_TILE_SIZE_N_SUB // 16).permute(0, 1, 3, 4, 2).reshape(num_warps, BLOCK_TILE_SIZE_K // num_warps, BLOCK_TILE_SIZE_N_SUB)
                b_fma = gl.convert_layout(b, b_fma_layout, assert_trivial=True)
                acc = gl.amd.cdna4.mfma(a_fma, b_fma, acc)
                #_amd_iglp_sched_barrier(0)
                #_amd_iglp_sched_group_barrier(s_vmem_read, BLOCK_TILE_SIZE_M // 16 + BLOCK_TILE_SIZE_N_SUB // 16, 0)
                #_amd_iglp_sched_group_barrier(s_mfma, BLOCK_TILE_SIZE_M // 16 * BLOCK_TILE_SIZE_N_SUB // 16, 0)
                a_offsets += BLOCK_TILE_SIZE_K
                b_offsets += BLOCK_TILE_SIZE_K // 8 * 16 * 8

                #_amd_iglp_sched_barrier(0)
                a = gl.amd.cdna3.buffer_load(p_input, a_offsets)
                if weight_dtype == torch.bfloat16:
                    b = gl.amd.cdna3.buffer_load(p_weight, b_offsets)
                else:
                    assert 0, "only bf16 weight is supported in this kernel"
                next_a = next_a.reshape(BLOCK_TILE_SIZE_M, num_warps, 4, 8).permute(1, 0, 2, 3).reshape(num_warps, BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_K // num_warps)
                a_fma = gl.convert_layout(next_a, a_fma_layout, assert_trivial=True)
                next_b = next_b.reshape(num_warps, 4, 16, 8, BLOCK_TILE_SIZE_N_SUB // 16).permute(0, 1, 3, 4, 2).reshape(num_warps, BLOCK_TILE_SIZE_K // num_warps, BLOCK_TILE_SIZE_N_SUB)
                b_fma = gl.convert_layout(next_b, b_fma_layout, assert_trivial=True)
                acc = gl.amd.cdna4.mfma(a_fma, b_fma, acc)
                #_amd_iglp_sched_barrier(0)
                #_amd_iglp_sched_group_barrier(s_vmem_read, BLOCK_TILE_SIZE_M // 16 + BLOCK_TILE_SIZE_N_SUB // 16, 0)
                #_amd_iglp_sched_group_barrier(s_mfma, BLOCK_TILE_SIZE_M // 16 * BLOCK_TILE_SIZE_N_SUB // 16, 0)
                a_offsets += BLOCK_TILE_SIZE_K
                b_offsets += BLOCK_TILE_SIZE_K // 8 * 16 * 8

            #_amd_iglp_sched_barrier(0)
            # tail
            next_a = gl.amd.cdna3.buffer_load(p_input, a_offsets)
            if weight_dtype == torch.bfloat16:
                next_b = gl.amd.cdna3.buffer_load(p_weight, b_offsets)
            #_amd_iglp_sched_barrier(0)
            a = a.reshape(BLOCK_TILE_SIZE_M, num_warps, 4, 8).permute(1, 0, 2, 3).reshape(num_warps, BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_K // num_warps)
            a_fma = gl.convert_layout(a, a_fma_layout, assert_trivial=True)
            b = b.reshape(num_warps, 4, 16, 8, BLOCK_TILE_SIZE_N_SUB // 16).permute(0, 1, 3, 4, 2).reshape(num_warps, BLOCK_TILE_SIZE_K // num_warps, BLOCK_TILE_SIZE_N_SUB)
            b_fma = gl.convert_layout(b, b_fma_layout, assert_trivial=True)
            acc = gl.amd.cdna4.mfma(a_fma, b_fma, acc)

            if n != BLOCK_TILE_SIZE_N // BLOCK_TILE_SIZE_N_SUB - 1:
                a = gl.amd.cdna3.buffer_load(p_input, mem_a_offsets)
                if weight_dtype == torch.bfloat16:
                    mem_b_offsets_sub += BLOCK_TILE_SIZE_N_SUB * K
                    b = gl.amd.cdna3.buffer_load(p_weight, mem_b_offsets_sub)
            #_amd_iglp_sched_barrier(0)
            next_a = next_a.reshape(BLOCK_TILE_SIZE_M, num_warps, 4, 8).permute(1, 0, 2, 3).reshape(num_warps, BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_K // num_warps)
            a_fma = gl.convert_layout(next_a, a_fma_layout, assert_trivial=True)
            next_b = next_b.reshape(num_warps, 4, 16, 8, BLOCK_TILE_SIZE_N_SUB // 16).permute(0, 1, 3, 4, 2).reshape(num_warps, BLOCK_TILE_SIZE_K // num_warps, BLOCK_TILE_SIZE_N_SUB)
            b_fma = gl.convert_layout(next_b, b_fma_layout, assert_trivial=True)
            acc = gl.amd.cdna4.mfma(a_fma, b_fma, acc)

            sorted_ids_org = gl.convert_layout(sorted_ids_org.reshape(BLOCK_TILE_SIZE_M), gl.SliceLayout(1, mem_c_layout), assert_trivial=False)
            sorted_ids = sorted_ids_org & 0xffffff
            out_offsets_m = sorted_ids
            out_offsets = out_offsets_m[:, None] * N + out_offsets_n[None, :]
            acc = acc.reshape(BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N_SUB)
            acc = gl.convert_layout(acc, mem_c_layout, assert_trivial=True)
            sorted_weights = gl.convert_layout(sorted_weights.reshape(BLOCK_TILE_SIZE_M), gl.SliceLayout(1, mem_c_layout), assert_trivial=True)[:, None]
            acc = (acc * sorted_weights).to(gl.bfloat16)
            mask = (out_offsets_m < M)[:, None]
            #gl.atomic_add(p_output + out_offsets, acc, sem='relaxed', mask=mask)
            out_offsets_n += BLOCK_TILE_SIZE_N_SUB
            gl.amd.cdna4.buffer_atomic_add(p_output, out_offsets, acc, sem='relaxed', mask=mask)
            #gl.amd.cdna3.buffer_store(acc, p_output, out_offsets, mask)
            # for _ in gl.static_range(BLOCK_TILE_SIZE_M * BLOCK_TILE_SIZE_N // 64 // 2):
            #     _amd_iglp_sched_group_barrier(s_vemm, 1, 0)
            #     _amd_iglp_sched_group_barrier(s_valu, 4, 0)

    return moe_down
