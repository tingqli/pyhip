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

from pyhip.contrib.gluon.utils import read_cycle, read_realtime, get_cu_id
from common.utils import gen_timing

SHUFFLE = 1

@triton.language.core._aggregate
class Args:
    BLOCK_TILE_SIZE_M: gl.constexpr
    BLOCK_TILE_SIZE_N: gl.constexpr
    weight_dtype: gl.constexpr
    num_warps: gl.constexpr
    BLOCK_TILE_SIZE_K: gl.constexpr
    SHUFFLE: gl.constexpr

    @gluon.constexpr_function
    def __init__(self, BLOCK_TILE_SIZE_M: gl.constexpr, BLOCK_TILE_SIZE_N: gl.constexpr, weight_dtype: gl.constexpr, num_warps: gl.constexpr, SHUFFLE: gl.constexpr):
        assert BLOCK_TILE_SIZE_M.bit_count() == 1, "BLOCK_TILE_SIZE_M must be power of 2"
        assert BLOCK_TILE_SIZE_N.bit_count() == 1, "BLOCK_TILE_SIZE_N must be power of 2"
        assert BLOCK_TILE_SIZE_M % 32 == 0, "BLOCK_TILE_SIZE_M must be multiple of 32"
        assert BLOCK_TILE_SIZE_N % 32 == 0, "BLOCK_TILE_SIZE_N must be multiple of 32"
        assert num_warps == 4 or num_warps == 8, "only support 4 or 8 warps"
        assert BLOCK_TILE_SIZE_M % 128 == 0, "BLOCK_TILE_SIZE_M must be >= 128"

        if not SHUFFLE:
            assert weight_dtype == torch.bfloat16, 'only bf16 support no shuffle'

        self.BLOCK_TILE_SIZE_M = gl.constexpr(BLOCK_TILE_SIZE_M)
        self.BLOCK_TILE_SIZE_N = gl.constexpr(BLOCK_TILE_SIZE_N)
        self.weight_dtype = gl.constexpr(weight_dtype)
        self.num_warps = gl.constexpr(num_warps)
        if weight_dtype == torch.bfloat16:
            self.BLOCK_TILE_SIZE_K = gl.constexpr(64)
        else:
            self.BLOCK_TILE_SIZE_K = gl.constexpr(128)
        self.SHUFFLE = gl.constexpr(SHUFFLE)

    @gluon.constexpr_function
    def get_mem_a_layout(self):
        if self.num_warps == 4:
            # base: [128, 64]
            reg_bases = [[0, 1], [0, 2], [0, 4], [4, 0], [8, 0]]
            lane_bases = [[0, 8], [0, 16], [0, 32], [16, 0], [32, 0], [64, 0]]
            warp_bases = [[1, 0], [2, 0]]
        else:
            reg_bases = [[0, 1], [0, 2], [0, 4], [8, 0]]
            lane_bases = [[0, 8], [0, 16], [0, 32], [16, 0], [32, 0], [64, 0]]
            warp_bases = [[1, 0], [2, 0], [4, 0]]
        m = self.BLOCK_TILE_SIZE_M // 128
        bit_pos = 128
        while m // 2:
            reg_bases.append([bit_pos, 0])
            bit_pos *= 2
            m = m // 2
        mem_a_layout = gl.DistributedLinearLayout(reg_bases=reg_bases,
                                                lane_bases=lane_bases,
                                                warp_bases=warp_bases,
                                                block_bases=[],
                                                shape=[self.BLOCK_TILE_SIZE_M, self.BLOCK_TILE_SIZE_K])
        return mem_a_layout

    @gluon.constexpr_function
    def get_mem_b_layout(self):
        # B: preshuffle layout
        if self.weight_dtype == torch.bfloat16:
            if self.SHUFFLE:
                mem_b_layout = gl.BlockedLayout(size_per_thread=[8, 1],
                                            threads_per_warp=[64, 1],
                                            warps_per_cta=[1, self.num_warps],
                                            order=[0, 1])
            else:
                # [64, 32]
                reg_bases = [[1, 0], [2, 0], [4, 0]]
                n = self.BLOCK_TILE_SIZE_N // 32
                bit_pos = 32
                while n // 2:
                    reg_bases.append([0, bit_pos])
                    bit_pos *= 2
                    n = n // 2
                mem_b_layout = gl.DistributedLinearLayout(reg_bases=reg_bases,
                                                        lane_bases=[[8, 0], [16, 0], [32, 0], [0, 4], [0, 8], [0, 16]],
                                                        warp_bases=[[0, 1], [0, 2]],
                                                        block_bases=[],
                                                        shape=[self.BLOCK_TILE_SIZE_K, self.BLOCK_TILE_SIZE_N])
        else:
            ...
        return mem_b_layout

    @gluon.constexpr_function
    def get_mem_c_layout(self):
        mem_c_layout: gl.constexpr = gl.BlockedLayout(
            size_per_thread=[1, 8],
            threads_per_warp=[4, 16],
            warps_per_cta=[self.num_warps, 1],
            order=[1, 0],
        )
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
        if self.SHUFFLE:
            lds_b_layout = gl.SwizzledSharedLayout(1, 1, 1, [0, 1])
        else:
            offset_bases = []
            n = self.BLOCK_TILE_SIZE_N // 32
            bit_pos = 32
            while n // 2:
                offset_bases.append([0, bit_pos])
                bit_pos *= 2
                n = n // 2
            lds_b_layout = gl.PaddedSharedLayout(
                [[512, 16]],
                [
                    [1, 0],
                    [2, 0],
                    [4, 0],
                    [8, 0],
                    [16, 0],
                    [32, 0],
                    [0, 4],
                    [0, 8],
                    [0, 16],
                    [0, 1],
                    [0, 2],
                ] + offset_bases,
                [],
                [self.BLOCK_TILE_SIZE_K, self.BLOCK_TILE_SIZE_N]
            )
        return lds_b_layout

    @gluon.constexpr_function
    def get_lds_layout_b_read(self):
        BLOCK_N = self.BLOCK_TILE_SIZE_N // 2
        warp_bases=[[0, 1], [0, 0]] if self.num_warps == 4 else [[0, 1], [0, 0], [0, 0]]
        # [64, 32]
        reg_bases=[[1, 0], [2, 0], [4, 0], [512, 0]]
        n = BLOCK_N // 32
        bit_pos = 2
        while n // 2:
            reg_bases.append([0, bit_pos])
            bit_pos *= 2
            n = n // 2
        lds_b_read_layout:gl.constexpr = gl.DistributedLinearLayout(reg_bases=reg_bases,
                                                                    lane_bases=[[8, 0], [16, 0], [32, 0], [64, 0], [128, 0], [256, 0]],
                                                                    warp_bases=warp_bases,
                                                                    block_bases=[],
                                                                    shape=[self.BLOCK_TILE_SIZE_K * 16, BLOCK_N // 16])
        return lds_b_read_layout

# copy from https://github.com/ROCm/gfx9-gluon-tutorials/blob/main/kernels/gemm/a16w16/v8_beyond_hotloop/matmul_kernel.py
@gluon.jit
def get_pids(
    M,
    N: gl.constexpr,
    BM: gl.constexpr,
    BN: gl.constexpr,
    GRID_MN,
    NUM_XCDS: gl.constexpr,
    GROUP_SIZE_M: gl.constexpr,
):
    pid = gl.program_id(axis=0)
    num_pid_m = gl.cdiv(M, BM)
    num_pid_n = gl.cdiv(N, BN)

    if NUM_XCDS != 1:
        ## pid remapping on xcds
        # Number of pids per XCD in the new arrangement
        pids_per_xcd = (GRID_MN + NUM_XCDS - 1) // NUM_XCDS
        # When GRID_MN cannot divide NUM_XCDS, some xcds will have
        # pids_per_xcd pids, the other will have pids_per_xcd - 1 pids.
        # We calculate the number of xcds that have pids_per_xcd pids as
        # tall_xcds
        tall_xcds = GRID_MN % NUM_XCDS
        tall_xcds = NUM_XCDS if tall_xcds == 0 else tall_xcds
        # Compute current XCD and local pid within the XCD
        xcd = pid % NUM_XCDS
        local_pid = pid // NUM_XCDS
        # Calculate new pid based on the new grouping
        # Note that we need to consider the following two cases:
        # 1. the current pid is on a tall xcd
        # 2. the current pid is on a short xcd
        if xcd < tall_xcds:
            pid = xcd * pids_per_xcd + local_pid
        else:
            pid = tall_xcds * pids_per_xcd + (xcd - tall_xcds) * (pids_per_xcd - 1) + local_pid

    if GROUP_SIZE_M == 1:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
    else:
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

    return pid_m, pid_n

# p: prefetch, w: wait, t: top, b: bottom, c: compute, l: local prefetch
# pro :  p0->p1->w0t->l0t
# loop:  [c0t->w0b->l0b->p2t == c0b->w1t->l1t->p2b] ->
#        [c1t->w1b->l1b->p3t == c1b->w2t->l2t->p3b] -> iter 0 end(unroll 2 times)
#        [c2t->w2b->l2b->p4t == c2b->w3t->l3t->p4b] ->
#        [c3t->w3b->l3b->p5t == c3b->w4t->l4t->p5b] -> iter 1 end(unroll 2 times)
#        ...
# epi:   [c(n-1)t->w(n-1)b->l(n-1)b-> == c(n-1)b->w(n)t->l(n)t->] ->
#        [c(n  )t->w(n  )b->l(n  )b-> == c(n  )b]   -> end
# stage1: c, stage2: p, stage3: l
@gluon.jit
def bf16_3stage_2(
    p_input,            # bf16 [M, K]
    p_weight,           # bf16 [K/8 * 16 * 8, N/16]
    p_output,           # bf16 [M, N]
    p_weight_scale,     # f32 [N/128, K/128] or None
    M,
    weight_dtype: gl.constexpr,
    K: gl.constexpr,
    N: gl.constexpr,
    BLOCK_TILE_SIZE_M: gl.constexpr,
    BLOCK_TILE_SIZE_N: gl.constexpr,
    p_debug_buf,
    SHUFFLE: gl.constexpr = 1,
    enable_debug: gl.constexpr = False,
    num_warps: gl.constexpr = 4,
):
    if enable_debug:
        _amd_iglp_sched_barrier(0)
        start = read_realtime()
        start_cycle = read_cycle()
        _amd_iglp_sched_barrier(0)

    args = Args(BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N, weight_dtype, num_warps, SHUFFLE)
    BLOCK_TILE_SIZE_K: gl.constexpr = args.BLOCK_TILE_SIZE_K
    mem_a_layout: gl.constexpr = args.get_mem_a_layout()
    mem_b_layout: gl.constexpr = args.get_mem_b_layout()
    mem_c_layout: gl.constexpr = args.get_mem_c_layout()
    lds_a_layout: gl.constexpr = args.get_lds_layout_a()
    lds_b_layout: gl.constexpr = args.get_lds_layout_b()
    if SHUFFLE:
        lds_b_read_layout: gl.constexpr = args.get_lds_layout_b_read()

    pid = gl.program_id(0)
    if 1:
        max_tile_n: gl.constexpr = N // BLOCK_TILE_SIZE_N
        tile_m = pid // max_tile_n
        tile_n = pid % max_tile_n
    else:
        num_pid = gl.num_programs(0)
        tile_m, tile_n = get_pids(M, N, BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N, num_pid,
                                8, 4)
    mem_a_offset_m = (tile_m * BLOCK_TILE_SIZE_M + gl.arange(0, BLOCK_TILE_SIZE_M, layout=gl.SliceLayout(1, mem_a_layout))) % M
    mem_a_offsets = mem_a_offset_m[:, None] * K + gl.arange(0, BLOCK_TILE_SIZE_K, layout=gl.SliceLayout(0, mem_a_layout))[None, :]
    gl.static_assert(SHUFFLE == 1, "not support SHUFFLE == 0")
    if SHUFFLE:
        mem_b_offsets = tile_n * BLOCK_TILE_SIZE_N * K + gl.arange(0, BLOCK_TILE_SIZE_N // 2 // 16, layout=gl.SliceLayout(0, mem_b_layout))[None, :] * K * 16 + \
                        gl.arange(0, BLOCK_TILE_SIZE_K * 16, layout=gl.SliceLayout(1, mem_b_layout))[:, None]
    else:
        mem_b_offsets = tile_n * BLOCK_TILE_SIZE_N * K + gl.arange(0, BLOCK_TILE_SIZE_N, layout=gl.SliceLayout(0, mem_b_layout))[None, :] * K + \
                        gl.arange(0, BLOCK_TILE_SIZE_K, layout=gl.SliceLayout(1, mem_b_layout))[:, None]
    if weight_dtype == torch.bfloat16:
        c_layout: gl.constexpr = gl.amd.AMDMFMALayout(version=4,
                                                    instr_shape=[16, 16, 32],
                                                    transposed=True,
                                                    warps_per_cta=[4, 2])
        a_fma_layout: gl.constexpr = gl.DotOperandLayout(0, c_layout, k_width=8)
        b_fma_layout: gl.constexpr = gl.DotOperandLayout(1, c_layout, k_width=8)
    else:
        c_layout: gl.constexpr = gl.amd.AMDMFMALayout(version=4,
                                                    instr_shape=[16, 16, 32],
                                                    transposed=True,
                                                    tiles_per_warp=[1, BLOCK_TILE_SIZE_M // 16, BLOCK_TILE_SIZE_N // 16],
                                                    warps_per_cta=[num_warps, 1, 1])
        a_fma_layout: gl.constexpr = gl.DotOperandLayout(0, c_layout, k_width=16)
        b_fma_layout: gl.constexpr = gl.DotOperandLayout(1, c_layout, k_width=16)
        mem_scale_offsets = (tile_n * BLOCK_TILE_SIZE_N // 128) * (K // 128) + gl.amd.cdna3.warp_id()
        weight_scale = gl.load(p_weight_scale + mem_scale_offsets)
    lds_a = gl.allocate_shared_memory(gl.bfloat16, [2, BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_K], lds_a_layout)
    lds_b_top = gl.allocate_shared_memory(gl.bfloat16, [2, BLOCK_TILE_SIZE_K * 16, BLOCK_TILE_SIZE_N // 2 // 16], lds_b_layout)
    lds_b_bot = gl.allocate_shared_memory(gl.bfloat16, [2, BLOCK_TILE_SIZE_K * 16, BLOCK_TILE_SIZE_N // 2 // 16], lds_b_layout)
    p_weight_top = p_weight
    p_weight_bot = p_weight + (BLOCK_TILE_SIZE_N // 2) * K
    # global prefetch 0, 1
    for i in gl.static_range(2):
        gl.amd.cdna4.async_copy.buffer_load_to_shared(lds_a.index(i), p_input, mem_a_offsets)
        gl.amd.cdna4.async_copy.buffer_load_to_shared(lds_b_top.index(i), p_weight_top, mem_b_offsets)
        gl.amd.cdna4.async_copy.commit_group()

        gl.amd.cdna4.async_copy.buffer_load_to_shared(lds_b_bot.index(i), p_weight_bot, mem_b_offsets)
        gl.amd.cdna4.async_copy.commit_group()
        mem_a_offsets += BLOCK_TILE_SIZE_K
        if SHUFFLE:
            mem_b_offsets += BLOCK_TILE_SIZE_K * 16
        else:
            mem_b_offsets += BLOCK_TILE_SIZE_K
    acc_top = gl.zeros((BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N // 2), dtype=gl.float32, layout=c_layout)
    acc_bot = gl.zeros((BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N // 2), dtype=gl.float32, layout=c_layout)
    s_vmem_read: gl.constexpr = 0x0020
    s_mfma: gl.constexpr = 0x0008
    s_ds_read: gl.constexpr = 0x0100
    s_ds_write: gl.constexpr = 0x0200
    if enable_debug:
        _amd_iglp_sched_barrier(0)
        loop_start = read_realtime()
        loop_start_cycle = read_cycle()
        _amd_iglp_sched_barrier(0)

    gl.amd.cdna4.async_copy.wait_group(3)
    # local prefetch 0
    a = gl.amd.cdna4.async_copy.load_shared_relaxed(lds_a.index(0), a_fma_layout)
    b_top = gl.amd.cdna4.async_copy.load_shared_relaxed(lds_b_top.index(0), lds_b_read_layout)
    b_top = b_top.reshape(BLOCK_TILE_SIZE_K // 8, 16, 8, BLOCK_TILE_SIZE_N // 2 // 16).permute(0, 2, 3, 1).reshape(BLOCK_TILE_SIZE_K, BLOCK_TILE_SIZE_N // 2)
    b_top = gl.convert_layout(b_top, b_fma_layout, assert_trivial=True)

    _amd_iglp_sched_barrier(0)
    for k_start in range(0, (K - BLOCK_TILE_SIZE_K * 2) // (2 * BLOCK_TILE_SIZE_K)):
        # _amd_iglp_sched_barrier(0)
        ##### c0t->w0b->l0b->p2t
        # compute 0t
        acc_top = gl.amd.cdna4.mfma(a, b_top, acc_top)
        # local prefetch 0b
        gl.amd.cdna4.async_copy.wait_group(2)
        b_bot = gl.amd.cdna4.async_copy.load_shared_relaxed(lds_b_bot.index(0), lds_b_read_layout)
        b_bot = b_bot.reshape(BLOCK_TILE_SIZE_K // 8, 16, 8, BLOCK_TILE_SIZE_N // 2 // 16).permute(0, 2, 3, 1).reshape(BLOCK_TILE_SIZE_K, BLOCK_TILE_SIZE_N // 2)
        b_bot = gl.convert_layout(b_bot, b_fma_layout, assert_trivial=True)
        # prefetch 2t
        gl.amd.cdna4.async_copy.buffer_load_to_shared(lds_a.index(0), p_input, mem_a_offsets)
        gl.amd.cdna4.async_copy.buffer_load_to_shared(lds_b_top.index(0), p_weight_top, mem_b_offsets)
        gl.amd.cdna4.async_copy.commit_group()
        #### c0b->w1t->l1t->p2b
        # compute 0b
        acc_bot = gl.amd.cdna4.mfma(a, b_bot, acc_bot)
        # local prefetch 1t
        gl.amd.cdna4.async_copy.wait_group(2)
        a_next = gl.amd.cdna4.async_copy.load_shared_relaxed(lds_a.index(1), a_fma_layout)
        b_top = gl.amd.cdna4.async_copy.load_shared_relaxed(lds_b_top.index(1), lds_b_read_layout)
        b_top = b_top.reshape(BLOCK_TILE_SIZE_K // 8, 16, 8, BLOCK_TILE_SIZE_N // 2 // 16).permute(0, 2, 3, 1).reshape(BLOCK_TILE_SIZE_K, BLOCK_TILE_SIZE_N // 2)
        b_top = gl.convert_layout(b_top, b_fma_layout, assert_trivial=True)
        # prefetch 2b
        gl.amd.cdna4.async_copy.buffer_load_to_shared(lds_b_bot.index(0), p_weight_bot, mem_b_offsets)
        gl.amd.cdna4.async_copy.commit_group()
        mem_a_offsets += BLOCK_TILE_SIZE_K
        if SHUFFLE:
            mem_b_offsets += BLOCK_TILE_SIZE_K * 16
        else:
            mem_b_offsets += BLOCK_TILE_SIZE_K

        ############## unroll ######################
        ##### c1t->w1b->l1b->p3t
        # compute 1t
        acc_top = gl.amd.cdna4.mfma(a_next, b_top, acc_top)
        # local prefetch 1b
        gl.amd.cdna4.async_copy.wait_group(2)
        b_bot = gl.amd.cdna4.async_copy.load_shared_relaxed(lds_b_bot.index(1), lds_b_read_layout)
        b_bot = b_bot.reshape(BLOCK_TILE_SIZE_K // 8, 16, 8, BLOCK_TILE_SIZE_N // 2 // 16).permute(0, 2, 3, 1).reshape(BLOCK_TILE_SIZE_K, BLOCK_TILE_SIZE_N // 2)
        b_bot = gl.convert_layout(b_bot, b_fma_layout, assert_trivial=True)
        # prefetch 3t
        gl.amd.cdna4.async_copy.buffer_load_to_shared(lds_a.index(1), p_input, mem_a_offsets)
        gl.amd.cdna4.async_copy.buffer_load_to_shared(lds_b_top.index(1), p_weight_top, mem_b_offsets)
        gl.amd.cdna4.async_copy.commit_group()
        #### c1b->w2t->l2t->p3b
        # compute 1b
        acc_bot = gl.amd.cdna4.mfma(a_next, b_bot, acc_bot)
        # local prefetch 2t
        gl.amd.cdna4.async_copy.wait_group(2)
        a = gl.amd.cdna4.async_copy.load_shared_relaxed(lds_a.index(0), a_fma_layout)
        b_top = gl.amd.cdna4.async_copy.load_shared_relaxed(lds_b_top.index(0), lds_b_read_layout)
        b_top = b_top.reshape(BLOCK_TILE_SIZE_K // 8, 16, 8, BLOCK_TILE_SIZE_N // 2 // 16).permute(0, 2, 3, 1).reshape(BLOCK_TILE_SIZE_K, BLOCK_TILE_SIZE_N // 2)
        b_top = gl.convert_layout(b_top, b_fma_layout, assert_trivial=True)
        # prefetch 3b
        gl.amd.cdna4.async_copy.buffer_load_to_shared(lds_b_bot.index(1), p_weight_bot, mem_b_offsets)
        gl.amd.cdna4.async_copy.commit_group()
        mem_a_offsets += BLOCK_TILE_SIZE_K
        if SHUFFLE:
            mem_b_offsets += BLOCK_TILE_SIZE_K * 16
        else:
            mem_b_offsets += BLOCK_TILE_SIZE_K

        _amd_iglp_sched_barrier(0)
        for _ in gl.static_range(BLOCK_TILE_SIZE_M // 32 + BLOCK_TILE_SIZE_N // 32):
            _amd_iglp_sched_group_barrier(s_vmem_read, 1, 0)
            _amd_iglp_sched_group_barrier(s_mfma, 1, 0)
        for _ in gl.static_range(BLOCK_TILE_SIZE_M // 16 + BLOCK_TILE_SIZE_N // 16):
            _amd_iglp_sched_group_barrier(s_ds_read, 1, 0)
            _amd_iglp_sched_group_barrier(s_mfma, 1, 0)

    _amd_iglp_sched_barrier(0)
    if enable_debug:
        loop_end = read_realtime()
        loop_end_cycle = read_cycle()
        _amd_iglp_sched_barrier(0)
    # tail
    if weight_dtype == torch.float8_e4m3fn:...
    else:
        ##### c0t->w0b->l0b->
        # compute 0t
        acc_top = gl.amd.cdna4.mfma(a, b_top, acc_top)
        # local prefetch 0b
        gl.amd.cdna4.async_copy.wait_group(2)
        b_bot = gl.amd.cdna4.async_copy.load_shared_relaxed(lds_b_bot.index(0), lds_b_read_layout)
        b_bot = b_bot.reshape(BLOCK_TILE_SIZE_K // 8, 16, 8, BLOCK_TILE_SIZE_N // 2 // 16).permute(0, 2, 3, 1).reshape(BLOCK_TILE_SIZE_K, BLOCK_TILE_SIZE_N // 2)
        b_bot = gl.convert_layout(b_bot, b_fma_layout, assert_trivial=True)
        #### c0b->w1t->l1t->
        # compute 0b
        acc_bot = gl.amd.cdna4.mfma(a, b_bot, acc_bot)
        # local prefetch 1t
        gl.amd.cdna4.async_copy.wait_group(1)
        a_next = gl.amd.cdna4.async_copy.load_shared_relaxed(lds_a.index(1), a_fma_layout)
        b_top = gl.amd.cdna4.async_copy.load_shared_relaxed(lds_b_top.index(1), lds_b_read_layout)
        b_top = b_top.reshape(BLOCK_TILE_SIZE_K // 8, 16, 8, BLOCK_TILE_SIZE_N // 2 // 16).permute(0, 2, 3, 1).reshape(BLOCK_TILE_SIZE_K, BLOCK_TILE_SIZE_N // 2)
        b_top = gl.convert_layout(b_top, b_fma_layout, assert_trivial=True)

        ############## unroll ######################
        ##### c1t->w1b->l1b->
        # compute 1t
        acc_top = gl.amd.cdna4.mfma(a_next, b_top, acc_top)
        # local prefetch 1b
        gl.amd.cdna4.async_copy.wait_group(0)
        b_bot = gl.amd.cdna4.async_copy.load_shared_relaxed(lds_b_bot.index(1), lds_b_read_layout)
        b_bot = b_bot.reshape(BLOCK_TILE_SIZE_K // 8, 16, 8, BLOCK_TILE_SIZE_N // 2 // 16).permute(0, 2, 3, 1).reshape(BLOCK_TILE_SIZE_K, BLOCK_TILE_SIZE_N // 2)
        b_bot = gl.convert_layout(b_bot, b_fma_layout, assert_trivial=True)
        #### c1b->
        # compute 1b
        acc_bot = gl.amd.cdna4.mfma(a_next, b_bot, acc_bot)

        out_offsets_m = (tile_m * BLOCK_TILE_SIZE_M + gl.arange(0, BLOCK_TILE_SIZE_M, layout=gl.SliceLayout(1, mem_c_layout))) % M
        out_offsets_n = (tile_n * BLOCK_TILE_SIZE_N + gl.arange(0, BLOCK_TILE_SIZE_N // 2, layout=gl.SliceLayout(0, mem_c_layout)))# % N
        out_offsets = out_offsets_m[:, None] * N + out_offsets_n[None, :]
        acc_top = acc_top.to(gl.bfloat16)
        gl.amd.cdna3.buffer_store(gl.convert_layout(acc_top, mem_c_layout, assert_trivial=False), p_output, out_offsets)
        acc_bot = acc_bot.to(gl.bfloat16)
        out_offsets += BLOCK_TILE_SIZE_N // 2
        gl.amd.cdna3.buffer_store(gl.convert_layout(acc_bot, mem_c_layout, assert_trivial=False), p_output, out_offsets)
    if enable_debug:
        _amd_iglp_sched_barrier(0)
        end = read_realtime()
        end_cycle = read_cycle()
        cu_id, se_id, xcc_id, slot_id = get_cu_id()
        gl.store(p_debug_buf + pid * 9 + 0, start)
        gl.store(p_debug_buf + pid * 9 + 1, start_cycle)
        gl.store(p_debug_buf + pid * 9 + 2, loop_start)
        gl.store(p_debug_buf + pid * 9 + 3, loop_start_cycle)
        gl.store(p_debug_buf + pid * 9 + 4, loop_end)
        gl.store(p_debug_buf + pid * 9 + 5, loop_end_cycle)
        gl.store(p_debug_buf + pid * 9 + 6, end)
        gl.store(p_debug_buf + pid * 9 + 7, end_cycle)
        gl.store(p_debug_buf + pid * 9 + 8,  (slot_id & 0xff) | ((cu_id & 0xff) << 8) | ((se_id & 0xff) << 16) | ((xcc_id & 0xff) << 24))

gemm = bf16_3stage_2 # bf16_2stage_1 bf16_2stage bf16_3stage bf16_3stage_2
#####################################################################
from pyhip import cudaPerf
from torch import Tensor
import pytest

def div_up(x, y):
    return (x + y - 1) // y
def _run_aiter(
                x: Tensor,  # A:[M, K] bf16
                weight: Tensor,  # B:[N, K/2] f4x2
                weight_scale: Tensor,  # B_scale:[N, K/32] e8m0 paded
            ):
    from aiter import gemm_a4w4, per_1x32_f4_quant_hip, gemm_a16w16
    M = x.shape[0]
    N = weight.shape[0]
    K = weight.shape[1]
    if x.dtype == torch.bfloat16:
        out = torch.empty([M, N], dtype=torch.bfloat16, device=x.device)
        gemm_a16w16(x, weight, out, splitK=4, bpreshuffle=True)
        return out

    # use hip quant kernel for performance
    x_q, x_s = per_1x32_f4_quant_hip(x, shuffle=True)

    # 32 alignment is enough for dim0 padding of output for
    # gemm_a4w4 kernel
    y = torch.empty(
        (M + 31) // 32 * 32,
        weight.shape[0],
        device=x_q.device,
        dtype=x.dtype,
    )

    gemm_a4w4(
        x_q, weight, x_s, weight_scale.view(x_s.dtype), y, bpreshuffle=True
    )

    return y[:M]

def _run_batch(kernel_type, M=1, weight_type=torch.bfloat16, TILE_M=16, TILE_N=32, run_count=10, N=4096, K=4096):
    BUF_COPY = 32
    A = torch.randn([BUF_COPY, M, K], dtype=torch.bfloat16)
    from aiter.ops.shuffle import shuffle_weight
    import aiter
    if weight_type == torch.float4_e2m1fn_x2:
        from aiter.utility import fp4_utils
        w_ = torch.randint(-2, 3, [N, K], dtype=torch.bfloat16) / 2
        w_qt, w_qt_scale_ = aiter.get_torch_quant(aiter.QuantType.per_1x32)(w_, quant_dtype=weight_type)
        w_f32 = fp4_utils.mxfp4_to_f32(w_qt).to(dtype=torch.bfloat16).reshape(N, K // 32, 32)
        w_scale_f32 = fp4_utils.e8m0_to_f32(w_qt_scale_).to(dtype=torch.bfloat16).reshape(N, K // 32, 1)
        w_ref = (w_f32 * w_scale_f32).reshape(N, K)
        assert K % 256 == 0, f'e8m0_shuffle assume there will be 8 groups of 32 elements in K dim, current K={K} is not supported'
        w_qt_scale = fp4_utils.e8m0_shuffle(w_qt_scale_)
        w = [shuffle_weight(w_qt) for _ in range(BUF_COPY)]
        w_scale = [w_qt_scale.clone() for _ in range(BUF_COPY)]
    elif weight_type == torch.bfloat16:
        w_ = torch.randn([N, K], dtype=torch.bfloat16)
        if SHUFFLE:
            w_shuffled = shuffle_weight(w_).reshape(N // 16, -1)
            w = [w_shuffled.clone() for _ in range(BUF_COPY)]
            w_org = [x.reshape([N, K]) for x in w]
        else:
            w = [w_ for _ in range(BUF_COPY)]
        w_scale = [None] * BUF_COPY
        w_ref = w_
    elif weight_type == torch.float8_e4m3fn:
        w_qt = torch.randint(-2, 3, [N, K], dtype=torch.float32).to(weight_type)
        w_qt_scale = torch.randint(-2, 3, [N // 128, K // 128], dtype=torch.float32)
        # # TODO
        # w_qt_scale[:] = 1
        w_f32 = w_qt.to(dtype=torch.float32).reshape(N // 128, 128, K // 128, 128)
        w_scale_f32 = w_qt_scale.reshape(N // 128, 1, K // 128, 1)
        w_ref = (w_f32 * w_scale_f32).reshape(N, K).to(dtype=torch.bfloat16)
        w = [shuffle_weight(w_qt) for _ in range(BUF_COPY)]
        w_scale = [w_qt_scale.clone() for _ in range(BUF_COPY)]
    else:
        assert 0, f'Only fp4 weight is supported in this test, current weight_type={weight_type}'

    flops = 2 * M * N * K
    if weight_type == torch.bfloat16:
        ele_size = 2
    elif weight_type == torch.float4_e2m1fn_x2:
        ele_size = 0.5
    else:
        ele_size = 1
    mem_size = M * K * 2 + N * K * ele_size

    def run(A, weight, weight_scale, p_debug_buf=None):
        M, K = A.shape
        N = w_ref.shape[0]
        gemm_out = torch.empty([M, N], dtype=A.dtype, device=A.device)
        num_warps = 8
        if kernel_type == 'mxn_2s':
            BLOCK_TILE_SIZE_M = TILE_M
            BLOCK_TILE_SIZE_N = TILE_N
            grids = div_up(N, BLOCK_TILE_SIZE_N) * div_up(M, BLOCK_TILE_SIZE_M)

            x = gemm[(grids,)](
                A,                # bf16 [M, K]
                weight.T,         # bf16 [K, N]
                gemm_out,         # bf16 [M, N]
                weight_scale,
                M,
                weight_type,
                K,
                N,
                BLOCK_TILE_SIZE_M,
                BLOCK_TILE_SIZE_N,
                p_debug_buf,
                SHUFFLE,
                enable_debug=p_debug_buf is not None,
                num_warps=num_warps
                )
            # print(x.asm['amdgcn'])
            # assert 0
        else:
            assert 0, f'not support kernel type "{kernel_type}"'
        return gemm_out

    tflops_res = []
    latencies = []
    bw = []
    if kernel_type == 'torch':
        i = 0
        for _ in range(run_count):
            with cudaPerf(flops, mem_size, name=f"{kernel_type}[{M=},{str(weight_type).split('.')[1]}]") as p:
                ref = torch.nn.functional.linear(A[i], w_org[i])
            tflops_res.append(p.tflops())
            latencies.append(p.dt())
            bw.append(p.bw())
    elif kernel_type == 'aiter':
        # aiter needs preshuffle weights
        i = 0
        for _ in range(run_count):
            with cudaPerf(flops, mem_size, name=f"{kernel_type}[{M=},{str(weight_type).split('.')[1]}]") as p:
                _run_aiter(x=A[i], weight=w_org[i], weight_scale=w_scale[i])
            i = (i + 1) % BUF_COPY
            tflops_res.append(p.tflops())
            latencies.append(p.dt())
            bw.append(p.bw())
    else:
        ref_out = A[0] @ w_ref.t()
        cur_out = run(A=A[0], weight=w[0], weight_scale=w_scale[0])
        i = 0
        for _ in range(run_count):
            with cudaPerf(flops, mem_size, name=f"{kernel_type}[{M=},{str(weight_type).split('.')[1]}]") as p:
                run(A=A[i], weight=w[i], weight_scale=w_scale[i])
            i = (i + 1) % BUF_COPY
            tflops_res.append(p.tflops())
            latencies.append(p.dt())
            bw.append(p.bw())

        grids = div_up(N, TILE_N) * div_up(M, TILE_M)
        p_debug_buf = torch.zeros([grids * 9], dtype=torch.int64)
        for _ in range(2):
            with cudaPerf(flops, mem_size, name=f"D{kernel_type}[{M=},{str(weight_type).split('.')[1]}]") as p:
                run(A=A[i], weight=w[i], weight_scale=w_scale[i], p_debug_buf=p_debug_buf)
            i = (i + 1) % BUF_COPY
        gen_timing(p_debug_buf, TILE_N * K * 2, TILE_N * K * TILE_M * 2)
        if not torch.allclose(ref_out, cur_out, rtol=0.1, atol=0.03):
            print(cur_out)
            idx = torch.where(torch.abs(ref_out - cur_out) > 0.03)
            if len(idx[0]):
                print(f'idx = {idx}\nref={ref_out[idx]}\ncur={cur_out[idx]}\n{len(idx[0])}')
            assert 0, f"{kernel_type=}, {M=}, {weight_type=}, {TILE_M=}, {TILE_N=}, {run_count=}"
        else:
            print(f"{kernel_type}[{M=} {weight_type=}] acc OK")
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

def entry_common(kernel_type, M, prec=[torch.bfloat16], TILE_M=32, TILE_N=64, N=4096, K=4096, run_count=10):
    perf = {}
    perf[kernel_type] = {}
    for weight_type in prec:
        if weight_type is None: continue
        perf_prec = {}
        for i in M:
            perf_prec[i] = _run_batch(kernel_type, M=i, weight_type=weight_type, TILE_M=TILE_M, TILE_N=TILE_N, run_count=run_count, N=N, K=K)
        perf[kernel_type][str(weight_type)] = perf_prec
    
    return perf

def init_env():
    torch.set_printoptions(linewidth=3000, sci_mode=False, edgeitems=8, )
    torch.set_default_device('cuda')
    torch.manual_seed(0)

def test_acc(TILE_M=32, TILE_N=64, N=4096, K=4096):
    init_env()
    M = list(range(2, 64))
    # fix TILE_M=16, TILE_N=32
    M += list(range(128, 256))
    M += [i * 256 for i in range(1, 4)]
    M += [i * 2048 for i in range(1, 5)]
    M += list(range(2048 * 3, 2048 * 3 + 256))
    # TILE_M/N is configurable
    entry_common('mxn_2s', M=M, prec=[torch.bfloat16], TILE_M=TILE_M, TILE_N=TILE_N, N=N, K=K, run_count=0)

def show_perf(perf, dict_tile_mn):
    print('\nsummary:')
    for kernel, vals in perf.items():
        for prec, vals_ in vals.items():
            for b, data in vals_.items():
                if kernel != 'aiter':
                    TILE_M, TILE_N = dict_tile_mn[f'{b}']
                    print(f'{kernel}[{prec:<4} B={b:<4}({TILE_M}x{TILE_N})]: {data["latency"]:5.0f} us, {data["bw"]:6.1f} GB/s, {data["flops"]:4.1f} tflops')
                else:
                    print(f'{kernel}[{prec:<4} B={b:<4}]: {data["latency"]:5.0f} us, {data["bw"]:6.1f} GB/s, {data["flops"]:4.1f} tflops')

@pytest.mark.parametrize("M", [[1, 2, 4, 8, 12, 16, 32, 64]])
def test_perf(M, TILE_M=32, TILE_N=64, N=4096, K=4096):
    init_env()
    perf = {}
    perf.update(entry_common('torch', M, prec=[torch.bfloat16], N=N, K=K, TILE_M=TILE_M, TILE_N=TILE_N))
    # TILE_M/N is configurable
    # perf.update(entry_common('mxn_2s', M=M, prec=[torch.bfloat16, get_fp8type()], TILE_M=TILE_M, TILE_N=TILE_N, N=N, K=K))
    # perf.update(entry_common('mxn_2s', M=M, prec=[get_fp8type()], TILE_M=TILE_M, TILE_N=TILE_N, N=N, K=K))
    perf.update(entry_common('mxn_2s', M=M, prec=[torch.bfloat16], TILE_M=TILE_M, TILE_N=TILE_N, N=N, K=K))
    return perf

def merge(a: dict, b: dict, path=[]):
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                merge(a[key], b[key], path + [str(key)])
            elif a[key] != b[key]:
                raise Exception('Conflict at ' + '.'.join(path + [str(key)]))
        else:
            a[key] = b[key]
    return a

if __name__ == '__main__':
    #TILE_M = 16
    #TILE_N = 128
    N, K = 4096*1, 1024*16 # 4096*8*2 /128 = 512
    #N, K = 64*1024*1, 1024*16
    Ms = [16, 32, 64, 128, 256]
    Ms = [4096,] # 64, 128]
    perf = {}
    dict_tile_mn = {}

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

    TILE_M, TILE_N = get_tile_mn(64)
    for M in Ms:
        TILE_M, TILE_N = get_tile_mn(M)
        # if N == 9216 and K == 4096:
        #     if M in [16]:
        #         TILE_M = 16
        #         TILE_N = 64
        #     elif M in [32]:
        #         TILE_M = 32
        #         TILE_N = 64
        #     elif M in [64, 128]:
        #         TILE_M = 32
        #         TILE_N = 128

        #TILE_M, TILE_N = 16, 64
        TILE_M, TILE_N = 256, 256
        dict_tile_mn[f'{M}'] = (TILE_M, TILE_N)
        print(f'final selected TILE_M={TILE_M}, TILE_N={TILE_N}')
        #test_acc(TILE_M=TILE_M, TILE_N=TILE_N, N=N, K=K)
        perf = merge(perf, test_perf([M], TILE_M=TILE_M, TILE_N=TILE_N, N=N, K=K))
    show_perf(perf, dict_tile_mn)