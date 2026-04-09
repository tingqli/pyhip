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
    'moe_2stage_gateup', 'moe_2stage_down'
]

@triton.language.core._aggregate
class GateupArgs:
    BLOCK_TILE_SIZE_M: gl.constexpr
    BLOCK_TILE_SIZE_N: gl.constexpr
    weight_dtype: gl.constexpr
    num_warps: gl.constexpr
    BLOCK_TILE_SIZE_K: gl.constexpr
    SHUFFLE: gl.constexpr

    @gluon.constexpr_function
    def __init__(self, BLOCK_TILE_SIZE_M: gl.constexpr, BLOCK_TILE_SIZE_N: gl.constexpr, weight_dtype: gl.constexpr):
        assert BLOCK_TILE_SIZE_M.bit_count() == 1, "BLOCK_TILE_SIZE_M must be power of 2"
        assert BLOCK_TILE_SIZE_N.bit_count() == 1, "BLOCK_TILE_SIZE_N must be power of 2"
        assert BLOCK_TILE_SIZE_M % 128 == 0, "BLOCK_TILE_SIZE_M must be multiple of 128"
        assert BLOCK_TILE_SIZE_N % 64 == 0, "BLOCK_TILE_SIZE_N must be multiple of 64"

        self.num_warps = gl.constexpr(4)
        self.BLOCK_TILE_SIZE_K = gl.constexpr(64)
        self.SHUFFLE = gl.constexpr(1)

        self.BLOCK_TILE_SIZE_M = gl.constexpr(BLOCK_TILE_SIZE_M)
        self.BLOCK_TILE_SIZE_N = gl.constexpr(BLOCK_TILE_SIZE_N)
        self.weight_dtype = gl.constexpr(weight_dtype)

    @gluon.constexpr_function
    def get_a_layout(self):
        reg_bases = [[0, 1], [0, 2], [0, 4], [4, 0], [8, 0]]
        m = self.BLOCK_TILE_SIZE_M // 128
        bit_pos = 128
        while m // 2:
            reg_bases.append([bit_pos, 0])
            bit_pos *= 2
            m = m // 2
        mem_a_layout = gl.DistributedLinearLayout(reg_bases=reg_bases,
                                                lane_bases=[[0, 8], [0, 16], [0, 32], [16, 0], [32, 0], [64, 0]],
                                                warp_bases=[[1, 0], [2, 0]],
                                                block_bases=[],
                                                shape=[self.BLOCK_TILE_SIZE_M, self.BLOCK_TILE_SIZE_K])
        return mem_a_layout

    @gluon.constexpr_function
    def get_b_layout(self):
        mem_b_layout = gl.BlockedLayout(size_per_thread=[8, 1],
                                    threads_per_warp=[64, 1],
                                    warps_per_cta=[1, 4],
                                    order=[0, 1])
        return mem_b_layout

    @gluon.constexpr_function
    def get_lds_layout_a(self):
        offset_bases = []
        m = self.BLOCK_TILE_SIZE_M // 128
        bit_pos = 128
        while m // 2:
            offset_bases.append([bit_pos, 0])
            bit_pos *= 2
            m = m // 2
        lds_a_layout = gl.PaddedSharedLayout(
            [[512, 16]],
            [
                [0, 1],
                [0, 2],
                [0, 4],
                [0, 8],
                [0, 16],
                [0, 32],
                [16, 0],
                [32, 0],
                [64, 0],
                [1, 0],
                [2, 0],
                [4, 0],
                [8, 0],
            ] + offset_bases,
            [],
            [self.BLOCK_TILE_SIZE_M, self.BLOCK_TILE_SIZE_K]
        )
        return lds_a_layout

    @gluon.constexpr_function
    def get_lds_layout_b(self):
        return gl.SwizzledSharedLayout(1, 1, 1, [0, 1])

    @gluon.constexpr_function
    def get_lds_layout_b_read(self):
        BLOCK_N = self.BLOCK_TILE_SIZE_N // 2
        reg_bases=[[1, 0], [2, 0], [4, 0], [512, 0]]
        n = BLOCK_N // 32
        bit_pos = 2
        while n // 2:
            reg_bases.append([0, bit_pos])
            bit_pos *= 2
            n = n // 2
        lds_b_read_layout = gl.DistributedLinearLayout(reg_bases=reg_bases,
                                                        lane_bases=[[8, 0], [16, 0], [32, 0], [64, 0], [128, 0], [256, 0]],
                                                        warp_bases=[[0, 1], [0, 0]],
                                                        block_bases=[],
                                                        shape=[self.BLOCK_TILE_SIZE_K * 16, BLOCK_N // 16])
        return lds_b_read_layout

    @gluon.constexpr_function
    def get_lds_layout_sort_ids_read(self):
        reg_bases=[[4, 0], [8, 0]]
        m = self.BLOCK_TILE_SIZE_M // 128
        bit_pos = 128
        while m // 2:
            reg_bases.append([bit_pos, 0])
            bit_pos *= 2
            m = m // 2
        sorted_ids_lds_layout_read = gl.DistributedLinearLayout(reg_bases=reg_bases,
                                                                lane_bases=[[0, 0], [0, 0], [0, 0], [16, 0], [32, 0], [64, 0]],
                                                                warp_bases=[[1, 0], [2, 0]],
                                                                block_bases=[],
                                                                shape=[self.BLOCK_TILE_SIZE_M, 1])
        return sorted_ids_lds_layout_read


@gluon.jit
def moe_2stage_gateup(p_input,            # bf16 [M, K]
                      p_weight,           # bf16 [K/8 * 16 * 8, N/16]
                      p_output,           # bf16 [M, N]
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
    sorted_id_layout: gl.constexpr = gl.BlockedLayout(size_per_thread=[1],
                                                    threads_per_warp=[64],
                                                    warps_per_cta=[num_warps],
                                                    order=[0])
    sorted_id_offsets = e_idx * BLOCK_TILE_SIZE_M + gl.arange(0, BLOCK_TILE_SIZE_M, layout=sorted_id_layout)

    num_valid_ids = gl.load(p_num_valid_ids)
    sorted_expert_id = gl.load(p_sorted_expert_ids + e_idx).to(gl.int64)
    if e_idx * BLOCK_TILE_SIZE_M >= num_valid_ids:
        return
    args = GateupArgs(BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N, weight_dtype)
    BLOCK_TILE_SIZE_K: gl.constexpr = args.BLOCK_TILE_SIZE_K
    mem_a_layout: gl.constexpr = args.get_a_layout()
    mem_b_layout: gl.constexpr = args.get_b_layout()
    sorted_ids_lds_layout_read: gl.constexpr = args.get_lds_layout_sort_ids_read()
    lds_a_layout: gl.constexpr = args.get_lds_layout_a()
    lds_b_layout: gl.constexpr = args.get_lds_layout_b()
    lds_b_read_layout: gl.constexpr = args.get_lds_layout_b_read()

    sorted_ids_mem = gl.amd.cdna3.buffer_load(p_sorted_ids, sorted_id_offsets)
    sorted_ids_lds_layout: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, order=[0, 1])
    sorted_ids_lds = gl.allocate_shared_memory(gl.int32, [BLOCK_TILE_SIZE_M, 1], layout=sorted_ids_lds_layout)
    sorted_ids_lds.store(sorted_ids_mem.reshape([BLOCK_TILE_SIZE_M, 1]))
    sorted_ids = gl.amd.cdna4.async_copy.load_shared_relaxed(sorted_ids_lds, layout=sorted_ids_lds_layout_read)

    p_weight = p_weight + sorted_expert_id * K * N

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
                                                warps_per_cta=[2, 2])
    a_fma_layout: gl.constexpr = gl.DotOperandLayout(0, c_layout, k_width=8)
    b_fma_layout: gl.constexpr = gl.DotOperandLayout(1, c_layout, k_width=8)
    lds_a = gl.allocate_shared_memory(gl.bfloat16, [2, BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_K], lds_a_layout)
    lds_b_top = gl.allocate_shared_memory(gl.bfloat16, [2, BLOCK_TILE_SIZE_K * 16, BLOCK_TILE_SIZE_N // 2 // 16], lds_b_layout)
    lds_b_bot = gl.allocate_shared_memory(gl.bfloat16, [2, BLOCK_TILE_SIZE_K * 16, BLOCK_TILE_SIZE_N // 2 // 16], lds_b_layout)
    acc_top = gl.zeros((BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N // 2), dtype=gl.float32, layout=c_layout)
    acc_bot = gl.zeros((BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N // 2), dtype=gl.float32, layout=c_layout)

    s_vmem_read: gl.constexpr = 0x0020
    s_mfma: gl.constexpr = 0x0008
    s_ds_read: gl.constexpr = 0x0100
    s_ds_write: gl.constexpr = 0x0200

    # global prefetch 0, 1
    for i in gl.static_range(2):
        gl.amd.cdna4.async_copy.buffer_load_to_shared(lds_a.index(i), p_input, mem_a_offsets)
        gl.amd.cdna4.async_copy.buffer_load_to_shared(lds_b_top.index(i), p_weight, mem_b_offsets_top)
        gl.amd.cdna4.async_copy.commit_group()

        gl.amd.cdna4.async_copy.buffer_load_to_shared(lds_b_bot.index(i), p_weight, mem_b_offsets_bot)
        gl.amd.cdna4.async_copy.commit_group()
        mem_a_offsets += BLOCK_TILE_SIZE_K
        mem_b_offsets_top += BLOCK_TILE_SIZE_K * 16
        mem_b_offsets_bot += BLOCK_TILE_SIZE_K * 16

    gl.amd.cdna4.async_copy.wait_group(3)
    # local prefetch 0
    a = gl.amd.cdna4.async_copy.load_shared_relaxed(lds_a.index(0), a_fma_layout)
    b_top = gl.amd.cdna4.async_copy.load_shared_relaxed(lds_b_top.index(0), lds_b_read_layout)
    b_top = b_top.reshape(BLOCK_TILE_SIZE_K // 8, 16, 8, BLOCK_TILE_SIZE_N // 2 // 16).permute(0, 2, 3, 1).reshape(BLOCK_TILE_SIZE_K, BLOCK_TILE_SIZE_N // 2)
    b_top = gl.convert_layout(b_top, b_fma_layout, assert_trivial=True)

    _amd_iglp_sched_barrier(0)
    for k_start in range(0, (K - BLOCK_TILE_SIZE_K * 2) // (2 * BLOCK_TILE_SIZE_K)):
        _amd_iglp_sched_barrier(0)
        ##### c0t->w0b->l0b->p2t
        acc_top = gl.amd.cdna4.mfma(a, b_top, acc_top)
        gl.amd.cdna4.async_copy.wait_group(2)
        b_bot = gl.amd.cdna4.async_copy.load_shared_relaxed(lds_b_bot.index(0), lds_b_read_layout)
        b_bot = b_bot.reshape(BLOCK_TILE_SIZE_K // 8, 16, 8, BLOCK_TILE_SIZE_N // 2 // 16).permute(0, 2, 3, 1).reshape(BLOCK_TILE_SIZE_K, BLOCK_TILE_SIZE_N // 2)
        b_bot = gl.convert_layout(b_bot, b_fma_layout, assert_trivial=True)
        gl.amd.cdna4.async_copy.buffer_load_to_shared(lds_a.index(0), p_input, mem_a_offsets)
        gl.amd.cdna4.async_copy.buffer_load_to_shared(lds_b_top.index(0), p_weight, mem_b_offsets_top)
        gl.amd.cdna4.async_copy.commit_group()
        #### c0b->w1t->l1t->p2b
        gl.inline_asm_elementwise(asm=';',
                                  constraints='=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r',
                                  args=[b_top],
                                  dtype=gl.bfloat16,
                                  is_pure=False,
                                  pack=32)
        for _ in gl.static_range(8):
            _amd_iglp_sched_group_barrier(s_mfma, 1, 0)
            _amd_iglp_sched_group_barrier(s_ds_read, 1, 0)
        for _ in gl.static_range(12):
            _amd_iglp_sched_group_barrier(s_mfma, 4, 0)
            _amd_iglp_sched_group_barrier(s_vmem_read, 1, 0)
        _amd_iglp_sched_group_barrier(s_mfma, 64 - 8 - 12*4, 0)
        _amd_iglp_sched_barrier(0)

        acc_bot = gl.amd.cdna4.mfma(a, b_bot, acc_bot)
        gl.amd.cdna4.async_copy.wait_group(2)
        a_next = gl.amd.cdna4.async_copy.load_shared_relaxed(lds_a.index(1), a_fma_layout)
        b_top = gl.amd.cdna4.async_copy.load_shared_relaxed(lds_b_top.index(1), lds_b_read_layout)
        b_top = b_top.reshape(BLOCK_TILE_SIZE_K // 8, 16, 8, BLOCK_TILE_SIZE_N // 2 // 16).permute(0, 2, 3, 1).reshape(BLOCK_TILE_SIZE_K, BLOCK_TILE_SIZE_N // 2)
        b_top = gl.convert_layout(b_top, b_fma_layout, assert_trivial=True)
        gl.amd.cdna4.async_copy.buffer_load_to_shared(lds_b_bot.index(0), p_weight, mem_b_offsets_bot)
        gl.amd.cdna4.async_copy.commit_group()
        mem_a_offsets += BLOCK_TILE_SIZE_K
        mem_b_offsets_top += BLOCK_TILE_SIZE_K * 16
        mem_b_offsets_bot += BLOCK_TILE_SIZE_K * 16
        gl.inline_asm_elementwise(asm=';', constraints='=r,=r,=r,=r,r,r,r,r,r,r,r,r', args=[b_top, b_bot],
                                dtype=gl.bfloat16, is_pure=False, pack=8)
        gl.inline_asm_elementwise(asm=';', constraints='=r,=r,=r,=r,r,r,r,r,r,r,r,r', args=[a, a_next],
                                dtype=gl.bfloat16, is_pure=False, pack=8)
        for _ in gl.static_range(24):
            _amd_iglp_sched_group_barrier(s_mfma, 1, 1)
            _amd_iglp_sched_group_barrier(s_ds_read, 1, 1)
        for _ in gl.static_range(4):
            _amd_iglp_sched_group_barrier(s_mfma, 4, 1)
            _amd_iglp_sched_group_barrier(s_vmem_read, 1, 1)
        _amd_iglp_sched_group_barrier(s_mfma, 64 - 24 - 4*4, 1)
        _amd_iglp_sched_barrier(0)

        ##### c1t->w1b->l1b->p3t
        acc_top = gl.amd.cdna4.mfma(a_next, b_top, acc_top)
        gl.amd.cdna4.async_copy.wait_group(2)
        b_bot = gl.amd.cdna4.async_copy.load_shared_relaxed(lds_b_bot.index(1), lds_b_read_layout)
        b_bot = b_bot.reshape(BLOCK_TILE_SIZE_K // 8, 16, 8, BLOCK_TILE_SIZE_N // 2 // 16).permute(0, 2, 3, 1).reshape(BLOCK_TILE_SIZE_K, BLOCK_TILE_SIZE_N // 2)
        b_bot = gl.convert_layout(b_bot, b_fma_layout, assert_trivial=True)
        gl.amd.cdna4.async_copy.buffer_load_to_shared(lds_a.index(1), p_input, mem_a_offsets)
        gl.amd.cdna4.async_copy.buffer_load_to_shared(lds_b_top.index(1), p_weight, mem_b_offsets_top)
        gl.amd.cdna4.async_copy.commit_group()
        #### c1b->w2t->l2t->p3b
        gl.inline_asm_elementwise(asm=';', constraints='=r,=r,=r,=r,r,r,r,r,r,r,r,r', args=[b_top, b_bot],
                                dtype=gl.bfloat16, is_pure=False, pack=8)
        gl.inline_asm_elementwise(asm=';', constraints='=r,=r,=r,=r,r,r,r,r,r,r,r,r', args=[a, a_next],
                                dtype=gl.bfloat16, is_pure=False, pack=8)
        for _ in gl.static_range(8):
            _amd_iglp_sched_group_barrier(s_mfma, 1, 2)
            _amd_iglp_sched_group_barrier(s_ds_read, 1, 2)
        for _ in gl.static_range(12):
            _amd_iglp_sched_group_barrier(s_mfma, 4, 2)
            _amd_iglp_sched_group_barrier(s_vmem_read, 1, 2)
        _amd_iglp_sched_group_barrier(s_mfma, 64 - 8 - 12*4, 2)
        _amd_iglp_sched_barrier(0)

        acc_bot = gl.amd.cdna4.mfma(a_next, b_bot, acc_bot)
        gl.amd.cdna4.async_copy.wait_group(2)
        a = gl.amd.cdna4.async_copy.load_shared_relaxed(lds_a.index(0), a_fma_layout)
        b_top = gl.amd.cdna4.async_copy.load_shared_relaxed(lds_b_top.index(0), lds_b_read_layout)
        b_top = b_top.reshape(BLOCK_TILE_SIZE_K // 8, 16, 8, BLOCK_TILE_SIZE_N // 2 // 16).permute(0, 2, 3, 1).reshape(BLOCK_TILE_SIZE_K, BLOCK_TILE_SIZE_N // 2)
        b_top = gl.convert_layout(b_top, b_fma_layout, assert_trivial=True)
        gl.amd.cdna4.async_copy.buffer_load_to_shared(lds_b_bot.index(1), p_weight, mem_b_offsets_bot)
        gl.amd.cdna4.async_copy.commit_group()
        mem_a_offsets += BLOCK_TILE_SIZE_K
        mem_b_offsets_top += BLOCK_TILE_SIZE_K * 16
        mem_b_offsets_bot += BLOCK_TILE_SIZE_K * 16

        gl.inline_asm_elementwise(asm=';', constraints='=r,=r,=r,=r,r,r,r,r,r,r,r,r', args=[b_top, b_bot],
                                dtype=gl.bfloat16, is_pure=False, pack=8)
        gl.inline_asm_elementwise(asm=';', constraints='=r,=r,=r,=r,r,r,r,r,r,r,r,r', args=[a, a_next],
                                dtype=gl.bfloat16, is_pure=False, pack=8)
        for _ in gl.static_range(24):
            _amd_iglp_sched_group_barrier(s_mfma, 1, 3)
            _amd_iglp_sched_group_barrier(s_ds_read, 1, 3)
        for _ in gl.static_range(4):
            _amd_iglp_sched_group_barrier(s_mfma, 4, 3)
            _amd_iglp_sched_group_barrier(s_vmem_read, 1, 3)
        _amd_iglp_sched_group_barrier(s_mfma, 64 - 24 - 4*4, 3)
        _amd_iglp_sched_barrier(0)

    _amd_iglp_sched_barrier(0)
    # tail
    ##### c0t->w0b->l0b->
    acc_top = gl.amd.cdna4.mfma(a, b_top, acc_top)
    gl.amd.cdna4.async_copy.wait_group(2)
    b_bot = gl.amd.cdna4.async_copy.load_shared_relaxed(lds_b_bot.index(0), lds_b_read_layout)
    b_bot = b_bot.reshape(BLOCK_TILE_SIZE_K // 8, 16, 8, BLOCK_TILE_SIZE_N // 2 // 16).permute(0, 2, 3, 1).reshape(BLOCK_TILE_SIZE_K, BLOCK_TILE_SIZE_N // 2)
    b_bot = gl.convert_layout(b_bot, b_fma_layout, assert_trivial=True)
    #### c0b->w1t->l1t->
    acc_bot = gl.amd.cdna4.mfma(a, b_bot, acc_bot)
    gl.amd.cdna4.async_copy.wait_group(1)
    a_next = gl.amd.cdna4.async_copy.load_shared_relaxed(lds_a.index(1), a_fma_layout)
    b_top = gl.amd.cdna4.async_copy.load_shared_relaxed(lds_b_top.index(1), lds_b_read_layout)
    b_top = b_top.reshape(BLOCK_TILE_SIZE_K // 8, 16, 8, BLOCK_TILE_SIZE_N // 2 // 16).permute(0, 2, 3, 1).reshape(BLOCK_TILE_SIZE_K, BLOCK_TILE_SIZE_N // 2)
    b_top = gl.convert_layout(b_top, b_fma_layout, assert_trivial=True)
    ##### c1t->w1b->l1b->
    acc_top = gl.amd.cdna4.mfma(a_next, b_top, acc_top)
    gl.amd.cdna4.async_copy.wait_group(0)
    b_bot = gl.amd.cdna4.async_copy.load_shared_relaxed(lds_b_bot.index(1), lds_b_read_layout)
    b_bot = b_bot.reshape(BLOCK_TILE_SIZE_K // 8, 16, 8, BLOCK_TILE_SIZE_N // 2 // 16).permute(0, 2, 3, 1).reshape(BLOCK_TILE_SIZE_K, BLOCK_TILE_SIZE_N // 2)
    b_bot = gl.convert_layout(b_bot, b_fma_layout, assert_trivial=True)
    #### c1b->
    acc_bot = gl.amd.cdna4.mfma(a_next, b_bot, acc_bot)
    _amd_iglp_sched_barrier(0)

    log2_exp1: gl.constexpr = -1.4426950408889634
    acc = (acc_top * (1.0 / (1.0 + gl.exp2(acc_top * log2_exp1)))) * acc_bot
    acc = acc.to(gl.bfloat16)

    mem_c_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8],
        threads_per_warp=[4, 16],
        warps_per_cta=[num_warps, 1],
        order=[1, 0],
    )
    sorted_ids_org = gl.amd.cdna4.async_copy.load_shared_relaxed(sorted_ids_lds, mem_c_layout)
    sorted_ids_org = gl.convert_layout(sorted_ids_org, mem_c_layout, assert_trivial=False)
    sorted_ids = sorted_ids_org & 0xffffff
    sorted_topk = sorted_ids_org >> 24
    out_offsets_n = (tile_n * BLOCK_TILE_SIZE_N // 2 + gl.arange(0, BLOCK_TILE_SIZE_N // 2, layout=gl.SliceLayout(0, mem_c_layout)))
    out_offsets = sorted_ids * N // 2 * TOPK + sorted_topk * N // 2 + out_offsets_n[None, :]
    mask = (sorted_ids < M)
    gl.amd.cdna3.buffer_store(gl.convert_layout(acc, mem_c_layout, assert_trivial=False), p_output, out_offsets, mask=mask)


@triton.language.core._aggregate
class GatedownArgs:
    BLOCK_TILE_SIZE_M: gl.constexpr
    BLOCK_TILE_SIZE_N: gl.constexpr
    weight_dtype: gl.constexpr
    num_warps: gl.constexpr
    BLOCK_TILE_SIZE_K: gl.constexpr
    SHUFFLE: gl.constexpr

    @gluon.constexpr_function
    def __init__(self, BLOCK_TILE_SIZE_M: gl.constexpr, BLOCK_TILE_SIZE_N: gl.constexpr, weight_dtype: gl.constexpr):
        assert BLOCK_TILE_SIZE_M.bit_count() == 1, "BLOCK_TILE_SIZE_M must be power of 2"
        assert BLOCK_TILE_SIZE_N.bit_count() == 1, "BLOCK_TILE_SIZE_N must be power of 2"
        assert BLOCK_TILE_SIZE_M % 128 == 0, "BLOCK_TILE_SIZE_M must be multiple of 128"
        assert BLOCK_TILE_SIZE_N % 64 == 0, "BLOCK_TILE_SIZE_N must be multiple of 64"

        self.num_warps = gl.constexpr(4)
        self.BLOCK_TILE_SIZE_K = gl.constexpr(64)
        self.SHUFFLE = gl.constexpr(1)

        self.BLOCK_TILE_SIZE_M = gl.constexpr(BLOCK_TILE_SIZE_M)
        self.BLOCK_TILE_SIZE_N = gl.constexpr(BLOCK_TILE_SIZE_N)
        self.weight_dtype = gl.constexpr(weight_dtype)

    @gluon.constexpr_function
    def get_a_layout(self):
        reg_bases = [[0, 1], [0, 2], [0, 4], [4, 0], [8, 0]]
        m = self.BLOCK_TILE_SIZE_M // 128
        bit_pos = 128
        while m // 2:
            reg_bases.append([bit_pos, 0])
            bit_pos *= 2
            m = m // 2
        mem_a_layout = gl.DistributedLinearLayout(reg_bases=reg_bases,
                                                lane_bases=[[0, 8], [0, 16], [0, 32], [16, 0], [32, 0], [64, 0]],
                                                warp_bases=[[1, 0], [2, 0]],
                                                block_bases=[],
                                                shape=[self.BLOCK_TILE_SIZE_M, self.BLOCK_TILE_SIZE_K])
        return mem_a_layout

    @gluon.constexpr_function
    def get_b_layout(self):
        mem_b_layout = gl.BlockedLayout(size_per_thread=[8, 1],
                                    threads_per_warp=[64, 1],
                                    warps_per_cta=[1, 4],
                                    order=[0, 1])
        return mem_b_layout

    @gluon.constexpr_function
    def get_c_layout(self):
        # base: [32, 64]
        assert self.BLOCK_TILE_SIZE_N % 64 == 0, f'64 elements in N dimenstion will be better for write out'
        if self.BLOCK_TILE_SIZE_N == 64:
            mem_c_layout: gl.constexpr = gl.BlockedLayout(size_per_thread=[1, 2],
                                                        threads_per_warp=[2, 32],
                                                        warps_per_cta=[2, 2],
                                                        order=[1, 0])
        else:
            mem_c_layout: gl.constexpr = gl.BlockedLayout(size_per_thread=[1, 2],
                                                        threads_per_warp=[1, 64],
                                                        warps_per_cta=[2, 2],
                                                        order=[1, 0])
        return mem_c_layout

    @gluon.constexpr_function
    def get_lds_layout_a(self):
        offset_bases = []
        m = self.BLOCK_TILE_SIZE_M // 128
        bit_pos = 128
        while m // 2:
            offset_bases.append([bit_pos, 0])
            bit_pos *= 2
            m = m // 2
        lds_a_layout = gl.PaddedSharedLayout(
            [[512, 16]],
            [
                [0, 1],
                [0, 2],
                [0, 4],
                [0, 8],
                [0, 16],
                [0, 32],
                [16, 0],
                [32, 0],
                [64, 0],
                [1, 0],
                [2, 0],
                [4, 0],
                [8, 0],
            ] + offset_bases,
            [],
            [self.BLOCK_TILE_SIZE_M, self.BLOCK_TILE_SIZE_K]
        )
        return lds_a_layout

    @gluon.constexpr_function
    def get_lds_layout_b(self):
        return gl.SwizzledSharedLayout(1, 1, 1, [0, 1])

    @gluon.constexpr_function
    def get_lds_layout_b_read(self):
        BLOCK_N = self.BLOCK_TILE_SIZE_N // 2
        reg_bases=[[1, 0], [2, 0], [4, 0], [512, 0]]
        n = BLOCK_N // 32
        bit_pos = 2
        while n // 2:
            reg_bases.append([0, bit_pos])
            bit_pos *= 2
            n = n // 2
        lds_b_read_layout = gl.DistributedLinearLayout(reg_bases=reg_bases,
                                                        lane_bases=[[8, 0], [16, 0], [32, 0], [64, 0], [128, 0], [256, 0]],
                                                        warp_bases=[[0, 1], [0, 0]],
                                                        block_bases=[],
                                                        shape=[self.BLOCK_TILE_SIZE_K * 16, BLOCK_N // 16])
        return lds_b_read_layout

    @gluon.constexpr_function
    def get_lds_layout_sort_ids_read(self):
        reg_bases=[[4, 0], [8, 0]]
        m = self.BLOCK_TILE_SIZE_M // 128
        bit_pos = 128
        while m // 2:
            reg_bases.append([bit_pos, 0])
            bit_pos *= 2
            m = m // 2
        sorted_ids_lds_layout_read = gl.DistributedLinearLayout(reg_bases=reg_bases,
                                                                lane_bases=[[0, 0], [0, 0], [0, 0], [16, 0], [32, 0], [64, 0]],
                                                                warp_bases=[[1, 0], [2, 0]],
                                                                block_bases=[],
                                                                shape=[self.BLOCK_TILE_SIZE_M, 1])
        return sorted_ids_lds_layout_read

    @gluon.constexpr_function
    def get_layout_sort_ids_weight(self):
        reg_bases=[[32, 0], [64, 0]]
        m = self.BLOCK_TILE_SIZE_M // 128
        bit_pos = 128
        while m // 2:
            reg_bases.append([bit_pos, 0])
            bit_pos *= 2
            m = m // 2
        sorted_weights_layout = gl.DistributedLinearLayout(reg_bases=reg_bases,
                                                            lane_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [0, 0], [0, 0]],
                                                            warp_bases=[[0, 0], [16, 0]],
                                                            block_bases=[],
                                                            shape=[self.BLOCK_TILE_SIZE_M, 1])
        return sorted_weights_layout


@gluon.jit
def moe_2stage_down(p_input,            # bf16 [M, K]
                    p_weight,           # bf16 [K/8 * 16 * 8, N/16]
                    p_output,           # bf16 [M, N]
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
                    num_warps: gl.constexpr = 4):
    tile_n = gl.program_id(0)
    e_idx = gl.program_id(1)

    sorted_id_layout: gl.constexpr = gl.BlockedLayout(size_per_thread=[1],
                                                    threads_per_warp=[64],
                                                    warps_per_cta=[num_warps],
                                                    order=[0])
    sorted_id_offsets = e_idx * BLOCK_TILE_SIZE_M + gl.arange(0, BLOCK_TILE_SIZE_M, layout=sorted_id_layout)

    num_valid_ids = gl.load(p_num_valid_ids)
    sorted_expert_id = gl.load(p_sorted_expert_ids + e_idx).to(gl.int64)
    if e_idx * BLOCK_TILE_SIZE_M >= num_valid_ids:
        return
    args = GatedownArgs(BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N, weight_dtype)
    BLOCK_TILE_SIZE_K: gl.constexpr = args.BLOCK_TILE_SIZE_K
    mem_a_layout: gl.constexpr = args.get_a_layout()
    mem_b_layout: gl.constexpr = args.get_b_layout()
    sorted_ids_lds_layout_read: gl.constexpr = args.get_lds_layout_sort_ids_read()
    lds_a_layout: gl.constexpr = args.get_lds_layout_a()
    lds_b_layout: gl.constexpr = args.get_lds_layout_b()
    lds_b_read_layout: gl.constexpr = args.get_lds_layout_b_read()
    sorted_weights_layout: gl.constexpr = args.get_layout_sort_ids_weight()

    sorted_ids_mem = gl.amd.cdna3.buffer_load(p_sorted_ids, sorted_id_offsets)
    sorted_ids_lds_layout: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, order=[0, 1])
    sorted_ids_lds = gl.allocate_shared_memory(gl.int32, [BLOCK_TILE_SIZE_M, 1], layout=sorted_ids_lds_layout)
    sorted_ids_lds.store(sorted_ids_mem.reshape([BLOCK_TILE_SIZE_M, 1]))
    sorted_ids = gl.amd.cdna4.async_copy.load_shared_relaxed(sorted_ids_lds, layout=sorted_ids_lds_layout_read)

    p_weight = p_weight + sorted_expert_id * K * N
    sorted_weights_offsets = e_idx * BLOCK_TILE_SIZE_M + gl.arange(0, BLOCK_TILE_SIZE_M, layout=gl.SliceLayout(1, sorted_weights_layout))[:, None]

    sorted_ids = gl.convert_layout(sorted_ids, mem_a_layout, assert_trivial=True)
    sorted_topk = sorted_ids >> 24
    sorted_ids = sorted_ids & 0xffffff
    sorted_topk = gl.where(sorted_topk < TOPK, sorted_topk, 0)
    sorted_ids = gl.where(sorted_ids < M, sorted_ids, 0)
    mem_a_offsets = (sorted_ids * K * TOPK + sorted_topk * K) + \
                    gl.arange(0, BLOCK_TILE_SIZE_K, layout=gl.SliceLayout(0, mem_a_layout))[None, :]
    mem_b_offsets_top = tile_n * BLOCK_TILE_SIZE_N * K + gl.arange(0, BLOCK_TILE_SIZE_N // 2 // 16, layout=gl.SliceLayout(0, mem_b_layout))[None, :] * (K // 8 * 16 * 8) + \
                    gl.arange(0, BLOCK_TILE_SIZE_K * 16, layout=gl.SliceLayout(1, mem_b_layout))[:, None]
    mem_b_offsets_bot = BLOCK_TILE_SIZE_N // 2 * K + tile_n * BLOCK_TILE_SIZE_N * K + gl.arange(0, BLOCK_TILE_SIZE_N // 2 // 16, layout=gl.SliceLayout(0, mem_b_layout))[None, :] * (K // 8 * 16 * 8) + \
                    gl.arange(0, BLOCK_TILE_SIZE_K * 16, layout=gl.SliceLayout(1, mem_b_layout))[:, None]
    c_layout: gl.constexpr = gl.amd.AMDMFMALayout(version=4,
                                                instr_shape=[16, 16, 32],
                                                transposed=True,
                                                warps_per_cta=[2, 2])
    a_fma_layout: gl.constexpr = gl.DotOperandLayout(0, c_layout, k_width=8)
    b_fma_layout: gl.constexpr = gl.DotOperandLayout(1, c_layout, k_width=8)
    lds_a = gl.allocate_shared_memory(gl.bfloat16, [2, BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_K], lds_a_layout)
    lds_b_top = gl.allocate_shared_memory(gl.bfloat16, [2, BLOCK_TILE_SIZE_K * 16, BLOCK_TILE_SIZE_N // 2 // 16], lds_b_layout)
    lds_b_bot = gl.allocate_shared_memory(gl.bfloat16, [2, BLOCK_TILE_SIZE_K * 16, BLOCK_TILE_SIZE_N // 2 // 16], lds_b_layout)
    acc_top = gl.zeros((BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N // 2), dtype=gl.float32, layout=c_layout)
    acc_bot = gl.zeros((BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N // 2), dtype=gl.float32, layout=c_layout)

    s_vemm: gl.constexpr = 0x0010
    s_vmem_read: gl.constexpr = 0x0020
    s_vmem_write: gl.constexpr = 0x0040
    s_mfma: gl.constexpr = 0x0008
    s_ds_read: gl.constexpr = 0x0100
    s_ds_write: gl.constexpr = 0x0200
    s_valu: gl.constexpr = 0x0002
    s_salu: gl.constexpr = 0x0004

    # global prefetch 0, 1
    for i in gl.static_range(2):
        gl.amd.cdna4.async_copy.buffer_load_to_shared(lds_a.index(i), p_input, mem_a_offsets)
        gl.amd.cdna4.async_copy.buffer_load_to_shared(lds_b_top.index(i), p_weight, mem_b_offsets_top)
        gl.amd.cdna4.async_copy.commit_group()

        gl.amd.cdna4.async_copy.buffer_load_to_shared(lds_b_bot.index(i), p_weight, mem_b_offsets_bot)
        gl.amd.cdna4.async_copy.commit_group()
        mem_a_offsets += BLOCK_TILE_SIZE_K
        mem_b_offsets_top += BLOCK_TILE_SIZE_K * 16
        mem_b_offsets_bot += BLOCK_TILE_SIZE_K * 16

    gl.amd.cdna4.async_copy.wait_group(3)
    # local prefetch 0
    a = gl.amd.cdna4.async_copy.load_shared_relaxed(lds_a.index(0), a_fma_layout)
    b_top = gl.amd.cdna4.async_copy.load_shared_relaxed(lds_b_top.index(0), lds_b_read_layout)
    b_top = b_top.reshape(BLOCK_TILE_SIZE_K // 8, 16, 8, BLOCK_TILE_SIZE_N // 2 // 16).permute(0, 2, 3, 1).reshape(BLOCK_TILE_SIZE_K, BLOCK_TILE_SIZE_N // 2)
    b_top = gl.convert_layout(b_top, b_fma_layout, assert_trivial=True)

    _amd_iglp_sched_barrier(0)
    for k_start in range(0, (K - BLOCK_TILE_SIZE_K * 2) // (2 * BLOCK_TILE_SIZE_K)):
        _amd_iglp_sched_barrier(0)
        ##### c0t->w0b->l0b->p2t
        acc_top = gl.amd.cdna4.mfma(a, b_top, acc_top)
        gl.amd.cdna4.async_copy.wait_group(2)
        b_bot = gl.amd.cdna4.async_copy.load_shared_relaxed(lds_b_bot.index(0), lds_b_read_layout)
        b_bot = b_bot.reshape(BLOCK_TILE_SIZE_K // 8, 16, 8, BLOCK_TILE_SIZE_N // 2 // 16).permute(0, 2, 3, 1).reshape(BLOCK_TILE_SIZE_K, BLOCK_TILE_SIZE_N // 2)
        b_bot = gl.convert_layout(b_bot, b_fma_layout, assert_trivial=True)
        gl.amd.cdna4.async_copy.buffer_load_to_shared(lds_a.index(0), p_input, mem_a_offsets)
        gl.amd.cdna4.async_copy.buffer_load_to_shared(lds_b_top.index(0), p_weight, mem_b_offsets_top)
        gl.amd.cdna4.async_copy.commit_group()
        #### c0b->w1t->l1t->p2b
        gl.inline_asm_elementwise(asm=';',
                                  constraints='=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r',
                                  args=[b_top],
                                  dtype=gl.bfloat16,
                                  is_pure=False,
                                  pack=32)
        for _ in gl.static_range(8):
            _amd_iglp_sched_group_barrier(s_mfma, 1, 0)
            _amd_iglp_sched_group_barrier(s_ds_read, 1, 0)
        for _ in gl.static_range(12):
            _amd_iglp_sched_group_barrier(s_mfma, 4, 0)
            _amd_iglp_sched_group_barrier(s_vmem_read, 1, 0)
        _amd_iglp_sched_group_barrier(s_mfma, 64 - 8 - 12*4, 0)
        _amd_iglp_sched_barrier(0)

        acc_bot = gl.amd.cdna4.mfma(a, b_bot, acc_bot)
        gl.amd.cdna4.async_copy.wait_group(2)
        a_next = gl.amd.cdna4.async_copy.load_shared_relaxed(lds_a.index(1), a_fma_layout)
        b_top = gl.amd.cdna4.async_copy.load_shared_relaxed(lds_b_top.index(1), lds_b_read_layout)
        b_top = b_top.reshape(BLOCK_TILE_SIZE_K // 8, 16, 8, BLOCK_TILE_SIZE_N // 2 // 16).permute(0, 2, 3, 1).reshape(BLOCK_TILE_SIZE_K, BLOCK_TILE_SIZE_N // 2)
        b_top = gl.convert_layout(b_top, b_fma_layout, assert_trivial=True)
        gl.amd.cdna4.async_copy.buffer_load_to_shared(lds_b_bot.index(0), p_weight, mem_b_offsets_bot)
        gl.amd.cdna4.async_copy.commit_group()
        mem_a_offsets += BLOCK_TILE_SIZE_K
        mem_b_offsets_top += BLOCK_TILE_SIZE_K * 16
        mem_b_offsets_bot += BLOCK_TILE_SIZE_K * 16
        gl.inline_asm_elementwise(asm=';', constraints='=r,=r,=r,=r,r,r,r,r,r,r,r,r', args=[b_top, b_bot],
                                dtype=gl.bfloat16, is_pure=False, pack=8)
        gl.inline_asm_elementwise(asm=';', constraints='=r,=r,=r,=r,r,r,r,r,r,r,r,r', args=[a, a_next],
                                dtype=gl.bfloat16, is_pure=False, pack=8)
        for _ in gl.static_range(24):
            _amd_iglp_sched_group_barrier(s_mfma, 1, 1)
            _amd_iglp_sched_group_barrier(s_ds_read, 1, 1)
        for _ in gl.static_range(4):
            _amd_iglp_sched_group_barrier(s_mfma, 4, 1)
            _amd_iglp_sched_group_barrier(s_vmem_read, 1, 1)
        _amd_iglp_sched_group_barrier(s_mfma, 64 - 24 - 4*4, 1)
        _amd_iglp_sched_barrier(0)

        ##### c1t->w1b->l1b->p3t
        acc_top = gl.amd.cdna4.mfma(a_next, b_top, acc_top)
        gl.amd.cdna4.async_copy.wait_group(2)
        b_bot = gl.amd.cdna4.async_copy.load_shared_relaxed(lds_b_bot.index(1), lds_b_read_layout)
        b_bot = b_bot.reshape(BLOCK_TILE_SIZE_K // 8, 16, 8, BLOCK_TILE_SIZE_N // 2 // 16).permute(0, 2, 3, 1).reshape(BLOCK_TILE_SIZE_K, BLOCK_TILE_SIZE_N // 2)
        b_bot = gl.convert_layout(b_bot, b_fma_layout, assert_trivial=True)
        gl.amd.cdna4.async_copy.buffer_load_to_shared(lds_a.index(1), p_input, mem_a_offsets)
        gl.amd.cdna4.async_copy.buffer_load_to_shared(lds_b_top.index(1), p_weight, mem_b_offsets_top)
        gl.amd.cdna4.async_copy.commit_group()
        #### c1b->w2t->l2t->p3b
        gl.inline_asm_elementwise(asm=';', constraints='=r,=r,=r,=r,r,r,r,r,r,r,r,r', args=[b_top, b_bot],
                                dtype=gl.bfloat16, is_pure=False, pack=8)
        gl.inline_asm_elementwise(asm=';', constraints='=r,=r,=r,=r,r,r,r,r,r,r,r,r', args=[a, a_next],
                                dtype=gl.bfloat16, is_pure=False, pack=8)
        for _ in gl.static_range(8):
            _amd_iglp_sched_group_barrier(s_mfma, 1, 2)
            _amd_iglp_sched_group_barrier(s_ds_read, 1, 2)
        for _ in gl.static_range(12):
            _amd_iglp_sched_group_barrier(s_mfma, 4, 2)
            _amd_iglp_sched_group_barrier(s_vmem_read, 1, 2)
        _amd_iglp_sched_group_barrier(s_mfma, 64 - 8 - 12*4, 2)
        _amd_iglp_sched_barrier(0)

        acc_bot = gl.amd.cdna4.mfma(a_next, b_bot, acc_bot)
        gl.amd.cdna4.async_copy.wait_group(2)
        a = gl.amd.cdna4.async_copy.load_shared_relaxed(lds_a.index(0), a_fma_layout)
        b_top = gl.amd.cdna4.async_copy.load_shared_relaxed(lds_b_top.index(0), lds_b_read_layout)
        b_top = b_top.reshape(BLOCK_TILE_SIZE_K // 8, 16, 8, BLOCK_TILE_SIZE_N // 2 // 16).permute(0, 2, 3, 1).reshape(BLOCK_TILE_SIZE_K, BLOCK_TILE_SIZE_N // 2)
        b_top = gl.convert_layout(b_top, b_fma_layout, assert_trivial=True)
        gl.amd.cdna4.async_copy.buffer_load_to_shared(lds_b_bot.index(1), p_weight, mem_b_offsets_bot)
        gl.amd.cdna4.async_copy.commit_group()
        mem_a_offsets += BLOCK_TILE_SIZE_K
        mem_b_offsets_top += BLOCK_TILE_SIZE_K * 16
        mem_b_offsets_bot += BLOCK_TILE_SIZE_K * 16

        gl.inline_asm_elementwise(asm=';', constraints='=r,=r,=r,=r,r,r,r,r,r,r,r,r', args=[b_top, b_bot],
                                dtype=gl.bfloat16, is_pure=False, pack=8)
        gl.inline_asm_elementwise(asm=';', constraints='=r,=r,=r,=r,r,r,r,r,r,r,r,r', args=[a, a_next],
                                dtype=gl.bfloat16, is_pure=False, pack=8)
        for _ in gl.static_range(24):
            _amd_iglp_sched_group_barrier(s_mfma, 1, 3)
            _amd_iglp_sched_group_barrier(s_ds_read, 1, 3)
        for _ in gl.static_range(4):
            _amd_iglp_sched_group_barrier(s_mfma, 4, 3)
            _amd_iglp_sched_group_barrier(s_vmem_read, 1, 3)
        _amd_iglp_sched_group_barrier(s_mfma, 64 - 24 - 4*4, 3)
        _amd_iglp_sched_barrier(0)

    _amd_iglp_sched_barrier(0)
    # tail
    ##### c0t->w0b->l0b->
    acc_top = gl.amd.cdna4.mfma(a, b_top, acc_top)
    gl.amd.cdna4.async_copy.wait_group(2)
    b_bot = gl.amd.cdna4.async_copy.load_shared_relaxed(lds_b_bot.index(0), lds_b_read_layout)
    b_bot = b_bot.reshape(BLOCK_TILE_SIZE_K // 8, 16, 8, BLOCK_TILE_SIZE_N // 2 // 16).permute(0, 2, 3, 1).reshape(BLOCK_TILE_SIZE_K, BLOCK_TILE_SIZE_N // 2)
    b_bot = gl.convert_layout(b_bot, b_fma_layout, assert_trivial=True)
    gl.inline_asm_elementwise(asm=';',
                                constraints='=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r',
                                args=[b_top],
                                dtype=gl.bfloat16,
                                is_pure=False,
                                pack=32)
    for _ in gl.static_range(8):
        _amd_iglp_sched_group_barrier(s_mfma, 1, 4)
        _amd_iglp_sched_group_barrier(s_ds_read, 1, 4)
    _amd_iglp_sched_group_barrier(s_mfma, 64 - 8 - 12*0, 4)
    _amd_iglp_sched_barrier(0)

    #### c0b->w1t->l1t->
    acc_bot = gl.amd.cdna4.mfma(a, b_bot, acc_bot)
    gl.amd.cdna4.async_copy.wait_group(1)
    a_next = gl.amd.cdna4.async_copy.load_shared_relaxed(lds_a.index(1), a_fma_layout)
    b_top = gl.amd.cdna4.async_copy.load_shared_relaxed(lds_b_top.index(1), lds_b_read_layout)
    b_top = b_top.reshape(BLOCK_TILE_SIZE_K // 8, 16, 8, BLOCK_TILE_SIZE_N // 2 // 16).permute(0, 2, 3, 1).reshape(BLOCK_TILE_SIZE_K, BLOCK_TILE_SIZE_N // 2)
    b_top = gl.convert_layout(b_top, b_fma_layout, assert_trivial=True)
    gl.inline_asm_elementwise(asm=';', constraints='=r,=r,=r,=r,r,r,r,r,r,r,r,r', args=[b_top, b_bot],
                            dtype=gl.bfloat16, is_pure=False, pack=8)
    gl.inline_asm_elementwise(asm=';', constraints='=r,=r,=r,=r,r,r,r,r,r,r,r,r', args=[a, a_next],
                            dtype=gl.bfloat16, is_pure=False, pack=8)
    for _ in gl.static_range(24):
        _amd_iglp_sched_group_barrier(s_mfma, 1, 5)
        _amd_iglp_sched_group_barrier(s_ds_read, 1, 5)
    _amd_iglp_sched_group_barrier(s_mfma, 64 - 24 - 4*0, 5)
    _amd_iglp_sched_barrier(0)

    ##### c1t->w1b->l1b->
    acc_top = gl.amd.cdna4.mfma(a_next, b_top, acc_top)
    gl.amd.cdna4.async_copy.wait_group(0)
    b_bot = gl.amd.cdna4.async_copy.load_shared_relaxed(lds_b_bot.index(1), lds_b_read_layout)
    b_bot = b_bot.reshape(BLOCK_TILE_SIZE_K // 8, 16, 8, BLOCK_TILE_SIZE_N // 2 // 16).permute(0, 2, 3, 1).reshape(BLOCK_TILE_SIZE_K, BLOCK_TILE_SIZE_N // 2)
    b_bot = gl.convert_layout(b_bot, b_fma_layout, assert_trivial=True)
    gl.inline_asm_elementwise(asm=';',
                                constraints='=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r',
                                args=[b_top],
                                dtype=gl.bfloat16,
                                is_pure=False,
                                pack=32)
    for _ in gl.static_range(8):
        _amd_iglp_sched_group_barrier(s_mfma, 1, 6)
        _amd_iglp_sched_group_barrier(s_ds_read, 1, 6)
    _amd_iglp_sched_group_barrier(s_mfma, 64 - 8 - 12*0, 6)

    sorted_weights = gl.amd.cdna3.buffer_load(p_sorted_weights, sorted_weights_offsets)
    #### c1b->
    acc_bot = gl.amd.cdna4.mfma(a_next, b_bot, acc_bot)

    if 0:
        mem_c_layout: gl.constexpr = args.get_c_layout()
        sorted_ids_org = gl.amd.cdna4.async_copy.load_shared_relaxed(sorted_ids_lds, mem_c_layout)
        sorted_ids_org = gl.convert_layout(sorted_ids_org, mem_c_layout, assert_trivial=False)
        sorted_ids = sorted_ids_org & 0xffffff
        out_offsets_m = sorted_ids
        out_offsets_n = (tile_n * BLOCK_TILE_SIZE_N + gl.arange(0, BLOCK_TILE_SIZE_N // 2, layout=gl.SliceLayout(0, mem_c_layout)))
        out_offsets = out_offsets_m * N + out_offsets_n[None, :]
        sorted_weights_top = gl.convert_layout(sorted_weights, acc_top.type.layout, assert_trivial=True)
        acc_top = (acc_top * sorted_weights_top).to(gl.bfloat16)
        acc_top = gl.convert_layout(acc_top, mem_c_layout, assert_trivial=False)
        mask = (out_offsets_m < M)
        gl.amd.cdna4.buffer_atomic_add(p_output, out_offsets, acc_top, sem='relaxed', mask=mask)
        sorted_weights_bot = gl.convert_layout(sorted_weights, acc_bot.type.layout, assert_trivial=True)
        acc_bot = (acc_bot * sorted_weights_bot).to(gl.bfloat16)
        acc_bot = gl.convert_layout(acc_bot, mem_c_layout, assert_trivial=False)
        out_offsets_bot = out_offsets + BLOCK_TILE_SIZE_N // 2
        gl.amd.cdna4.buffer_atomic_add(p_output, out_offsets_bot, acc_bot, sem='relaxed', mask=mask)
    else:
        mem_c_layout: gl.constexpr = gl.BlockedLayout(
            size_per_thread=[1, 8],
            threads_per_warp=[4, 16],
            warps_per_cta=[num_warps, 1],
            order=[1, 0],
        )
        sorted_ids_org = gl.amd.cdna4.async_copy.load_shared_relaxed(sorted_ids_lds, mem_c_layout)
        # sorted_ids_org = gl.convert_layout(sorted_ids_org, mem_c_layout, assert_trivial=False)
        sorted_ids = (sorted_ids_org & 0xffffff)
        mask = (sorted_ids < M)
        sorted_ids = sorted_ids #.to(gl.int64)
        sorted_topk = sorted_ids_org >> 24
        out_offsets_n = (tile_n * BLOCK_TILE_SIZE_N + gl.arange(0, BLOCK_TILE_SIZE_N // 2, layout=gl.SliceLayout(0, mem_c_layout)))
        out_offsets = sorted_ids * N * TOPK + sorted_topk * N + out_offsets_n[None, :]
        sorted_weights_top = gl.convert_layout(sorted_weights, acc_top.type.layout, assert_trivial=True)
        acc_top = (acc_top * sorted_weights_top).to(gl.bfloat16)
        p_out_top = p_output + out_offsets
        gl.store(p_out_top, gl.convert_layout(acc_top, mem_c_layout, assert_trivial=False), mask)
        sorted_weights_bot = gl.convert_layout(sorted_weights, acc_bot.type.layout, assert_trivial=True)
        acc_bot = (acc_bot * sorted_weights_bot).to(gl.bfloat16)
        p_out_bot = p_output + out_offsets + BLOCK_TILE_SIZE_N // 2
        gl.store(p_out_bot, gl.convert_layout(acc_bot, mem_c_layout, assert_trivial=False), mask)
