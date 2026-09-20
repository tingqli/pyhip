# SPDX-License-Identifier: MIT
"""GRRead Down: M64/N320 or N160/K64, FP32 accumulation, a BF16 GEMM boundary, and SiLU output."""

from functools import cache

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr, rocdl

from .common import K, R
from .helpers import _sigmoid_down as _sigmoid, _weight_view

BM, BK = 64, 64
GROUP_STEPS = 8
GROUPS = K // BK // GROUP_STEPS


@cache
def make_down(*, n_splits=1):
    """Specialize only by N splits; the launcher receives the actual rows at runtime.

    N1 covers 320 channels; N2 CTAs cover disjoint 160-channel slices without split-K or reduction launches.
    Callers must provide positive rows and P[rows, R]; tail stores are bounded by the actual row count.
    """
    if n_splits not in (1, 2):
        raise ValueError("n_splits must be 1 or 2")
    BN = R // n_splits
    N_WAVES, M_WAVES = 4 // n_splits, n_splits

    @fx.struct
    class Shared:
        a0: fx.Array[fx.BFloat16, BM * BK, 16]
        a1: fx.Array[fx.BFloat16, BM * BK, 16]

    @flyc.kernel(known_block_size=[256, 1, 1])
    def gr_read_down(X: fx.Tensor, W: fx.Tensor, P: fx.Tensor, rows: fx.Int64):
        tid = fx.Int32(fx.thread_idx.x)
        im, jn, _ = fx.block_idx
        # 先以64位把全局指针移到本M64 tile；buffer内仅保留局部32位偏移。
        row_begin = fx.Int64(im) * BM
        remaining = rows - row_begin
        valid_rows = (remaining > 0).select(remaining, fx.Int64(0))
        valid_rows = (valid_rows < BM).select(valid_rows, fx.Int64(BM))
        # 尾块X只读取有效行；越界buffer读返回0，P的尾行store由descriptor屏蔽。
        x_row_begin = (row_begin < rows).select(row_begin, rows)
        x = fx.rocdl.make_buffer_tensor(fx.make_view(fx.get_iter(X) + x_row_begin * K,
                    fx.make_layout((BM, K), (K, 1))), max_size=False, num_records_bytes=valid_rows * K * 2)
        a_tiles = fx.flat_divide(x, fx.make_tile(BM, BK))[None, None, 0, None]
        if const_expr(n_splits == 1):
            weights = fx.rocdl.make_buffer_tensor(_weight_view(fx.get_iter(W), R, K), max_size=False)
            output = fx.rocdl.make_buffer_tensor(fx.make_view(fx.get_iter(P) + row_begin * R,
                        fx.make_layout((R, BM), (1, R))), max_size=False, num_records_bytes=valid_rows * R * 2)
            b_tiles = fx.flat_divide(weights, fx.make_tile(BN, BK))[None, None, jn, None]
            c_tile = fx.flat_divide(output, fx.make_tile(BN, BM))[None, None, jn, 0]
        else:
            # preshuffle的每N16块连续；N160正好10块，从局部N0开始避免N64取整。
            n_begin = fx.Int64(jn) * BN
            weights = fx.rocdl.make_buffer_tensor(
                _weight_view(fx.get_iter(W) + n_begin * K, BN, K), max_size=False)
            p_bytes = (valid_rows > 0).select(((valid_rows - 1) * R + BN) * 2, fx.Int64(0))
            output = fx.rocdl.make_buffer_tensor(fx.make_view(fx.get_iter(P) + row_begin * R + n_begin,
                        fx.make_layout((BN, BM), (1, R))), max_size=False,
                        num_records_bytes=p_bytes)
            b_tiles = fx.flat_divide(weights, fx.make_tile(BN, BK))[None, None, 0, None]
            c_tile = fx.flat_divide(output, fx.make_tile(BN, BM))[None, None, 0, 0]
        atom = fx.make_mma_atom(rocdl.MFMA(16, 16, 16, fx.BFloat16))
        if const_expr(n_splits == 1):
            tiled = fx.make_tiled_mma(atom, fx.make_layout((4, 1, 1), (1, 0, 0)),
                                     (None, None, fx.make_layout((4, 4, 2), (1, 8, 4))))
        else:
            tiled = fx.make_tiled_mma(atom, fx.make_layout((2, 2, 1), (1, 2, 0)),
                                     (None, None, fx.make_layout((4, 4, 2), (1, 8, 4))))
        mma_thread = tiled.thr_slice(tid)
        copy_g = fx.make_copy_atom(rocdl.BufferCopy128b(), fx.BFloat16)
        copy_s = fx.make_copy_atom(fx.UniversalCopy128b(), fx.BFloat16)
        threads_k, threads_m = BK // 8, 256 // (BK // 8)
        tv = fx.make_layout(((threads_k, threads_m), (1, 8)), ((threads_m * 8, 1), (1, threads_m)))
        a_gcopy = fx.make_tiled_copy(copy_g, tv, fx.make_tile(threads_m, BK)).get_slice(tid)
        a_source = a_gcopy.partition_S(a_tiles)
        a_transfer = fx.make_fragment_like(a_source[None, None, None, 0])
        shared = fx.SharedAllocator().allocate(Shared).peek()
        swizzle = fx.make_composed_layout(fx.static(fx.SwizzleType.get(3, 3, 3)),
                                           fx.make_ordered_layout((BM, BK), (1, 0)))
        a_lds = [fx.make_view(pointer, swizzle) for pointer in (shared.a0.ptr, shared.a1.ptr)]
        a_scopy = fx.make_tiled_copy(copy_s, tv, fx.make_tile(threads_m, BK)).get_slice(tid)
        a_destinations = [a_scopy.partition_D(a) for a in a_lds]
        a_transfer_s = a_scopy.retile(a_transfer)
        a_mcopy = fx.make_tiled_copy_B(copy_s, tiled).get_slice(tid)
        a_reads = [a_mcopy.partition_S(a) for a in a_lds]
        a_fragment = mma_thread.make_fragment_B(a_lds[0])
        a_registers = a_mcopy.retile(a_fragment)
        b_copy = fx.make_tiled_copy_A(copy_g, tiled).get_slice(tid)
        b_source = b_copy.partition_S(b_tiles)
        b_fragments = [mma_thread.make_fragment_A(b_tiles[None, None, 0]) for _ in range_constexpr(2)]
        b_registers = [b_copy.retile(b) for b in b_fragments]
        c_fragment = mma_thread.make_fragment_C(c_tile)
        c_fragment.fill(0)

        def read_g2r(q, slot):
            fx.copy(copy_g, a_source[None, None, None, q], a_transfer)
            fx.copy(copy_g, b_source[None, None, None, q], b_registers[slot])

        def stage(slot, next_k, *, read_next=True):
            if const_expr(read_next):
                read_g2r(next_k, slot ^ 1)
            for ki in range_constexpr(BK // 32):
                fx.copy(copy_s, a_reads[slot][None, None, ki], a_registers[None, None, ki])
                for token_tile in range_constexpr(BM // (16 * M_WAVES)):
                    for channel_tile in range_constexpr(BN // (16 * N_WAVES)):
                        for k16 in range_constexpr(2):
                            fx.mma_atom_call(atom, c_fragment[None, channel_tile, token_tile],
                                b_fragments[slot][None, channel_tile, (k16, ki)], a_fragment[None, token_tile, (k16, ki)],
                                c_fragment[None, channel_tile, token_tile])
            if const_expr(read_next):
                fx.copy(copy_s, a_transfer_s, a_destinations[slot ^ 1])
                # A2+B10条VMEM；N2的B跨M wave复用地址，但仍由各wave独立读取。
                # N1: 12*6+4+2*2=80 MFMA；N2: 12*3+2*2=40 MFMA。
                for request in range_constexpr(12):
                    if const_expr(request < 8 // M_WAVES):
                        rocdl.sched_dsrd(1)
                    rocdl.sched_vmem(1)
                    rocdl.sched_mfma(6 // M_WAVES)
                if const_expr(n_splits == 1):
                    rocdl.sched_mfma(4)
                for _ in range_constexpr(2):
                    rocdl.sched_dswr(1)
                    rocdl.sched_mfma(2)
            rocdl.s_waitcnt(lgkmcnt=0)
            rocdl.sched_barrier(0)
            fx.gpu.barrier()
            rocdl.sched_barrier(0)

        read_g2r(fx.Int32(0), 0)
        fx.copy(copy_s, a_transfer_s, a_destinations[0])
        rocdl.s_waitcnt(lgkmcnt=0)
        fx.gpu.barrier()
        for group, state in range(fx.Index(0), fx.Index(GROUPS - 1), fx.Index(1), init=[c_fragment.load()]):
            c_fragment.store(state[0])
            for step in range_constexpr(GROUP_STEPS):
                stage(step & 1, fx.Int32(group) * GROUP_STEPS + step + 1)
            result = yield [c_fragment.load()]
        # FlyDSL单个yield结果是SSA值，不是单元素列表。
        c_fragment.store(result)
        for step in range_constexpr(GROUP_STEPS):
            stage(step & 1, fx.Int32((GROUPS - 1) * GROUP_STEPS + step + 1), read_next=step < GROUP_STEPS - 1)

        # 对齐原torch.compile：GEMM输出先舍入BF16，再融合FP32缩放/SiLU。
        acc = c_fragment.load() + 0.0
        z = fx.Vector(acc).to(fx.BFloat16).to(fx.Float32) * 0.25
        activated = fx.Vector.from_elements([z[i] * _sigmoid(z[i]) for i in range_constexpr(z.numel)], fx.Float32)
        p_bf16 = fx.make_fragment_like(c_fragment, dtype=fx.BFloat16)
        p_bf16.store(activated.to(fx.BFloat16))
        copy_c = fx.make_copy_atom(rocdl.BufferCopy64b(), fx.BFloat16)
        c_copy = fx.make_tiled_copy_C(copy_c, tiled).get_slice(tid)
        fx.copy(copy_c, c_copy.retile(p_bf16), c_copy.partition_D(c_tile))

    @flyc.jit
    def launch_down(X: fx.Tensor, W: fx.Tensor, P: fx.Tensor,
                    rows: fx.Int64, stream: fx.Stream):
        m_tiles = (rows + BM - 1) // BM
        gr_read_down(X, W, P, rows).launch(grid=(m_tiles, R // BN, 1), block=(256, 1, 1), stream=stream)

    return launch_down