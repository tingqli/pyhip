# SPDX-License-Identifier: MIT
"""GRRead Down with full K reduction, optional small-M tiles, a BF16 GEMM boundary, and SiLU."""

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
def make_down(*, n_splits=1, block_m=64, num_waves=4, block_k=64, swizzle_shift=3):
    """Build a complete-K Down with tunable M/N tiles, waves, BK and LDS swizzle.

    N1/N2/N4 retain their original M64/W4/BK64 configurations.
    N5/N10/N20 allow smaller M tiles and one, two or four waves.
    BK64..1024 choices are bounded by thread-copy coverage and 64 KiB LDS.
    K groups and scheduling budgets follow the actual tile/request counts.
    The original FP32 accumulation -> BF16 -> FP32 SiLU -> BF16 order is preserved.
    Runtime rows bound all tail loads/stores; weight packing is unchanged.
    """
    if n_splits not in (1, 2, 4, 5, 10, 20):
        raise ValueError('unsupported N split')
    if block_m not in (16, 32, 64) or num_waves not in (1, 2, 4):
        raise ValueError('unsupported M/waves')
    if block_k not in (64, 128, 256, 512, 1024):
        raise ValueError('unsupported BK')
    BM, BK = block_m, block_k
    assert BM * BK * 4 <= 65536 and BK // 8 <= num_waves * 64
    GROUP_STEPS = 8
    while (K // BK) % GROUP_STEPS:
        GROUP_STEPS //= 2
    GROUPS = K // BK // GROUP_STEPS
    SWIZZLE_SHIFT = (BK.bit_length() - 4) if swizzle_shift is None else swizzle_shift
    THREADS = num_waves * 64
    BN = R // n_splits
    if n_splits in (1, 2, 4):
        assert (BM, num_waves, BK) == (64, 4, 64)
        N_WAVES, M_WAVES = {1: (4, 1), 2: (2, 2), 4: (1, 4)}[n_splits]
    else:
        N_WAVES = min(num_waves, BN // 16)
        M_WAVES = num_waves // N_WAVES
    assert BM % (16 * M_WAVES) == 0 and BN % (16 * N_WAVES) == 0
    STORE_MFMA = 2 if n_splits <= 5 else 1
    A_REQUESTS = BM * BK // (THREADS * 8)
    A_LDS_READS = BM // M_WAVES // 8 * (BK // 64)
    VMEM_REQUESTS = A_REQUESTS + BN // N_WAVES // 8 * (BK // 64)
    STAGE_MFMA = BM // (16 * M_WAVES) * (BN // (16 * N_WAVES)) * (BK // 16)

    @fx.struct
    class Shared:
        a0: fx.Array[fx.BFloat16, BM * BK, 16]
        a1: fx.Array[fx.BFloat16, BM * BK, 16]

    @flyc.kernel(known_block_size=[THREADS, 1, 1])
    def gr_read_down(X: fx.Tensor, W: fx.Tensor, P: fx.Tensor, rows: fx.Int64):
        tid = fx.Int32(fx.thread_idx.x)
        im, jn, _ = fx.block_idx
        # 先以64位把全局指针移到本M tile；buffer内仅保留局部32位偏移。
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
            # preshuffle的每N16块连续；各分片从局部N0开始，避免N64取整。
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
        elif const_expr(n_splits == 2):
            tiled = fx.make_tiled_mma(atom, fx.make_layout((2, 2, 1), (1, 2, 0)),
                                     (None, None, fx.make_layout((4, 4, 2), (1, 8, 4))))
        else:
            tiled = fx.make_tiled_mma(atom, fx.make_layout((N_WAVES, M_WAVES, 1), (1, N_WAVES, 0)),
                                     (None, None, fx.make_layout((4, 4, 2), (1, 8, 4))))
        mma_thread = tiled.thr_slice(tid)
        copy_g = fx.make_copy_atom(rocdl.BufferCopy128b(), fx.BFloat16)
        copy_s = fx.make_copy_atom(fx.UniversalCopy128b(), fx.BFloat16)
        threads_k, threads_m = BK // 8, THREADS // (BK // 8)
        tv = fx.make_layout(((threads_k, threads_m), (1, 8)), ((threads_m * 8, 1), (1, threads_m)))
        a_gcopy = fx.make_tiled_copy(copy_g, tv, fx.make_tile(threads_m, BK)).get_slice(tid)
        a_source = a_gcopy.partition_S(a_tiles)
        a_transfer = fx.make_fragment_like(a_source[None, None, None, 0])
        shared = fx.SharedAllocator().allocate(Shared).peek()
        swizzle = fx.make_composed_layout(fx.static(fx.SwizzleType.get(3, 3, SWIZZLE_SHIFT)),
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
                if const_expr(n_splits in (1, 2)):
                    # 保留已验收路径：A2+B10请求，N1/N2分别80/40条MFMA。
                    for request in range_constexpr(12):
                        if const_expr(request < 8 // M_WAVES):
                            rocdl.sched_dsrd(1)
                        rocdl.sched_vmem(1)
                        rocdl.sched_mfma(6 // M_WAVES)
                    if const_expr(n_splits == 1):
                        rocdl.sched_mfma(4)
                else:
                    # M64的N4/N5保持原配额；小M按实际A搬运条数调整，K顺序不变。
                    # 按N分片为每条A LDS store留1或2条MFMA，其余按真实VMEM请求数分配。
                    for request in range_constexpr(VMEM_REQUESTS):
                        ds_reads = A_LDS_READS // VMEM_REQUESTS + (request < A_LDS_READS % VMEM_REQUESTS)
                        if const_expr(ds_reads):
                            rocdl.sched_dsrd(ds_reads)
                        rocdl.sched_vmem(1)
                        count = ((request + 1) * (STAGE_MFMA - STORE_MFMA * A_REQUESTS) // VMEM_REQUESTS
                                 - request * (STAGE_MFMA - STORE_MFMA * A_REQUESTS) // VMEM_REQUESTS)
                        if const_expr(count > 0):
                            rocdl.sched_mfma(count)
                for _ in range_constexpr(A_REQUESTS):
                    rocdl.sched_dswr(1)
                    rocdl.sched_mfma(STORE_MFMA)
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
        gr_read_down(X, W, P, rows).launch(grid=(m_tiles, R // BN, 1), block=(THREADS, 1, 1), stream=stream)

    return launch_down


# Decode T1..32: original global split-K=4 pipeline.
@cache
def make_decode_down(rows):
    (bm, bk, waves) = (16 if rows <= 16 else 32, 128, 4)
    (split, dn) = (4, 16)
    prefetch_unroll = 2 if rows <= 25 else 1
    iterations = K // waves // bk // split
    padded_rows = (rows + bm - 1) // bm * bm

    @fx.struct
    class DownShared:
        partials: fx.Array[fx.Float32, bm * dn * waves, 16]

    @flyc.kernel
    def down_wave_splitk_pipeline(X: fx.Tensor, W: fx.Tensor, A: fx.Tensor):
        tid = fx.thread_idx.x
        (lane, wave) = (tid % 64, tid // 64)
        (im, jn, sk) = fx.block_idx
        x = fx.rocdl.make_buffer_tensor(fx.make_view(fx.get_iter(X), fx.make_layout((rows, K), (K, 1))), max_size=False)
        w_layout = fx.make_layout(((16, R // 16), (8, 4, K // 32)), ((8, 16 * K), (1, 128, 512)))
        w = fx.rocdl.make_buffer_tensor(fx.make_view(fx.get_iter(W), w_layout), max_size=False)
        shared = fx.SharedAllocator().allocate(DownShared).peek()
        partials = shared.partials.view(fx.make_layout((bm, dn, waves), (dn, 1, bm * dn)))
        a_tile = fx.flat_divide(w, fx.make_tile(dn, bk))[None, None, jn, None]
        b_tile = fx.flat_divide(x, fx.make_tile(bm, bk))[None, None, im, None]
        c_tile = fx.select(partials[None, None, wave], [1, 0])
        mma = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, fx.BFloat16))
        tiled = fx.make_tiled_mma(mma, fx.make_layout((1, 1, 1), (0, 0, 0)), (None, None, fx.make_layout((4, 4, 2), (1, 8, 4))))
        thr = tiled.thr_slice(lane)
        copy_ab = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16)
        copy_c = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32)
        ca = fx.make_tiled_copy_A(copy_ab, tiled).get_slice(lane)
        cb = fx.make_tiled_copy_B(copy_ab, tiled).get_slice(lane)
        cc = fx.make_tiled_copy_C(copy_c, tiled).get_slice(lane)
        fa = thr.make_fragment_A(a_tile[None, None, 0])
        fb = thr.make_fragment_B(b_tile[None, None, 0])
        fc = thr.make_fragment_C(c_tile)
        (ga, gb) = (ca.partition_S(a_tile), cb.partition_S(b_tile))
        (ra, rb) = (ca.retile(fa), cb.retile(fb))
        fc.fill(0)
        next_a = thr.make_fragment_A(a_tile[None, None, 0])
        next_b = thr.make_fragment_B(b_tile[None, None, 0])
        (next_ra, next_rb) = (ca.retile(next_a), cb.retile(next_b))
        first = sk * (K // split // bk) + wave
        fx.copy(copy_ab, ga[None, None, None, first], ra)
        fx.copy(copy_ab, gb[None, None, None, first], rb)
        for (ki, state) in range(fx.Index(1), fx.Index(iterations), fx.Index(prefetch_unroll), init=[fa.load(), fb.load(), fc.load()]):
            fa.store(state[0])
            fb.store(state[1])
            fc.store(state[2])
            kt = sk * (K // split // bk) + fx.Int32(ki) * waves + wave
            fx.copy(copy_ab, ga[None, None, None, kt], next_ra)
            fx.copy(copy_ab, gb[None, None, None, kt], next_rb)
            fx.gemm(mma, fc, fa, fb, fc)
            if fx.const_expr(prefetch_unroll == 2):
                second = kt + waves
                fx.copy(copy_ab, ga[None, None, None, second], ra)
                fx.copy(copy_ab, gb[None, None, None, second], rb)
                fx.gemm(mma, fc, next_a, next_b, fc)
                (carried_a, carried_b) = (fa.load(), fb.load())
            else:
                (carried_a, carried_b) = (next_a.load(), next_b.load())
            result = (yield [carried_a, carried_b, fc.load()])
        fa.store(result[0])
        fb.store(result[1])
        fc.store(result[2])
        fx.gemm(mma, fc, fa, fb, fc)
        fx.copy(copy_c, cc.retile(fc), cc.partition_D(c_tile))
        fx.gpu.barrier()
        out = fx.make_view(fx.get_iter(A), fx.make_layout((padded_rows, R, split), (R, 1, padded_rows * R)))
        for i in range_constexpr((bm * dn + waves * 64 - 1) // (waves * 64)):
            index = tid + i * waves * 64
            if index < bm * dn:
                (row, col) = (index // dn, index % dn)
                total = fx.Float32(0.0)
                for s in range_constexpr(waves):
                    total = total + fx.memref_load(partials, (row, col, s))
                fx.memref_store(total, out, (im * bm + row, jn * dn + col, sk))

    @flyc.jit
    def launch(X: fx.Tensor, W: fx.Tensor, A: fx.Tensor, stream: fx.Stream):
        down_wave_splitk_pipeline(X, W, A).launch(grid=(padded_rows // bm, R // dn, split), block=(waves * 64, 1, 1), stream=stream)
    return launch
