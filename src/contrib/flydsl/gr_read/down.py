# SPDX-License-Identifier: MIT
"""GR read Down：M64/N320/K64，FP32累加，BF16 GEMM边界与SiLU输出。"""

from functools import cache

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr, rocdl

from .common import K, R, validate_launch_rows
from .helpers import _sigmoid_down as _sigmoid, _weight_view

BM, BN, BK = 64, 320, 64
GROUP_STEPS = 8
GROUPS = K // BK // GROUP_STEPS


@cache
def make_down(rows, padded_rows):
    """构造正式Down；每个CTA计算64行和全部320个隐藏通道。"""
    validate_launch_rows(rows, padded_rows, BM)

    @fx.struct
    class Shared:
        a0: fx.Array[fx.BFloat16, BM * BK, 16]
        a1: fx.Array[fx.BFloat16, BM * BK, 16]

    @flyc.kernel(known_block_size=[256, 1, 1])
    def gr_read_down(X: fx.Tensor, W: fx.Tensor, P: fx.Tensor):
        tid = fx.Int32(fx.thread_idx.x)
        im, jn, _ = fx.block_idx
        # 先以64位把全局指针移到本M64 tile；buffer内仅保留局部32位偏移。
        row_begin = fx.Int64(im) * BM
        if const_expr(rows % BM == 0):
            valid_rows = (row_begin < rows).select(fx.Int64(BM), fx.Int64(0))
        else:
            remaining = fx.Int64(rows) - row_begin
            valid_rows = (remaining > 0).select(remaining, fx.Int64(0))
            valid_rows = (valid_rows < BM).select(valid_rows, fx.Int64(BM))
        # 全padding CTA的X范围为0，所有buffer读返回0；P仍写完整64行零。
        x_row_begin = (row_begin < rows).select(row_begin, fx.Int64(rows))
        x = fx.rocdl.make_buffer_tensor(fx.make_view(fx.get_iter(X) + x_row_begin * K,
                    fx.make_layout((BM, K), (K, 1))), max_size=False, num_records_bytes=valid_rows * K * 2)
        weights = fx.rocdl.make_buffer_tensor(_weight_view(fx.get_iter(W), R, K), max_size=False)
        output = fx.rocdl.make_buffer_tensor(fx.make_view(fx.get_iter(P) + row_begin * R,
                    fx.make_layout((R, BM), (1, R))), max_size=False, num_records_bytes=BM * R * 2)
        a_tiles = fx.flat_divide(x, fx.make_tile(BM, BK))[None, None, 0, None]
        b_tiles = fx.flat_divide(weights, fx.make_tile(BN, BK))[None, None, jn, None]
        c_tile = fx.flat_divide(output, fx.make_tile(BN, BM))[None, None, jn, 0]
        atom = fx.make_mma_atom(rocdl.MFMA(16, 16, 16, fx.BFloat16))
        tiled = fx.make_tiled_mma(atom, fx.make_layout((4, 1, 1), (1, 0, 0)),
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
                for token_tile in range_constexpr(BM // 16):
                    for channel_tile in range_constexpr(BN // 64):
                        for k16 in range_constexpr(2):
                            fx.mma_atom_call(atom, c_fragment[None, channel_tile, token_tile],
                                b_fragments[slot][None, channel_tile, (k16, ki)], a_fragment[None, token_tile, (k16, ki)],
                                c_fragment[None, channel_tile, token_tile])
            if const_expr(read_next):
                fx.copy(copy_s, a_transfer_s, a_destinations[slot ^ 1])
                # A2+B10个VMEM请求，每次间隔6条MFMA；8次DS读，2次A写。
                # 每拍共12*6 + 4 + 2*2 = 80条MFMA，保持既定的交错顺序。
                for request in range_constexpr(12):
                    if const_expr(request < 8):
                        rocdl.sched_dsrd(1)
                    rocdl.sched_vmem(1)
                    rocdl.sched_mfma(6)
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
    def launch_down(X: fx.Tensor, W: fx.Tensor, P: fx.Tensor, stream: fx.Stream):
        gr_read_down(X, W, P).launch(grid=(padded_rows // BM, R // BN, 1), block=(256, 1, 1), stream=stream)

    return launch_down