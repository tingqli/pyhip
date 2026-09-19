# SPDX-License-Identifier: MIT
"""GR read 的 BF16 MFMA kernel；K128 B 双缓冲与 FP32 累加。

流水参考 src/contrib/flydsl/moe_gemm_2stage/gemm2_8x1.py 的 k128n 分支，
按小 M 改为四个 wave，不引入 MoE 路由、FP8 量化或跨 CTA 原子归约。
接口和数值顺序参考 apinge/pyhip 的 gr_read_qwen/CombinedPaddedGRRead。
"""

from dataclasses import dataclass
from functools import cache

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import range_constexpr

C, H, R = 4, 2560, 320
K = C * H
# 64K 压测仍沿用 decoding 的 BM16 流水；X 最大 1.25 GiB，地址保持在 i32 范围。
MAX_ROWS = 65536
LOG2E = 1.4426950408889634


@dataclass(frozen=True)
class Config:
    block_m: int = 16
    down_n: int = 64
    up_n: int = 128
    block_k: int = 128
    split_k: int = 16
    waves: int = 4
    down_waves: int = 4
    down_mode: str = "partial"
    hidden_pad: int = 4
    preshuffle: bool = True
    compensate_hidden: bool = True
    prefetch_low: bool = False


def default_config(rows, hidden_pad=4):
    if not isinstance(rows, int) or isinstance(rows, bool) or not 0 <= rows <= MAX_ROWS:
        raise ValueError(f"GR read supports 0..{MAX_ROWS} rows")
    if hidden_pad not in (0, 4, 8, 16, 32):
        raise ValueError("hidden_pad must be 0, 4, 8, 16 or 32 (64-bit alignment)")
    return Config(down_n=16 if rows > 16 else 64,
                  down_mode="wave_splitk" if rows > 16 else "partial",
                  hidden_pad=hidden_pad)


def preshuffle_weight(weight):
    """BF16 [N,K] -> [N/16,K/32,4,16,8]，仅在准备阶段执行。"""
    n, k = weight.shape
    return weight.detach().reshape(n // 16, 16, k // 32, 4, 8).permute(0, 2, 3, 1, 4).contiguous().view(-1)


def _weight_view(pointer, n, k):
    return fx.make_view(pointer, fx.make_layout(
        ((16, n // 16), (8, 4, k // 32)), ((8, 16 * k), (1, 128, 512))))


def _tiled_mma(waves):
    atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, fx.BFloat16))
    tiled = fx.make_tiled_mma(atom, fx.make_layout((1, waves, 1), (0, 1, 0)),
                            (None, None, fx.make_layout((4, 4, 2), (1, 8, 4))))
    return atom, tiled


def _read_b_g2r(weights, n_begin, k_begin, full_k, bn, bk, copy_tid, threads):
    """每 lane 读取若干 16B packet；偏移单位均为 BF16 元素。"""
    turns = bn * bk // (threads * 8)
    fragment = fx.make_rmem_tensor(fx.make_layout((8, turns), (1, 8)), fx.BFloat16)
    copy = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16)
    for turn in range_constexpr(turns):
        packet = copy_tid + turn * threads
        n_group, within = packet // (2 * bk), packet % (2 * bk)
        offset = (n_begin + n_group * 16) * full_k + k_begin * 16 + within * 8
        source = fx.make_view(fx.get_iter(weights) + offset, fx.make_layout(8, 1))
        fx.copy(copy, source, fragment[None, turn])
    return fragment


def _store_b_r2s(fragment, pointer, bn, bk, copy_tid, threads):
    copy = fx.make_copy_atom(fx.UniversalCopy128b(), fx.BFloat16)
    for turn in range_constexpr(bn * bk // (threads * 8)):
        destination = fx.make_view(pointer + (copy_tid + turn * threads) * 8, fx.make_layout(8, 1))
        fx.copy(copy, fragment[None, turn], destination)


def _read_b_s2r(tiled, mma_tid, pointer, bn, bk):
    source = _weight_view(pointer, bn, bk)
    copy = fx.make_copy_atom(fx.UniversalCopy128b(), fx.BFloat16)
    partition = fx.make_tiled_copy_B(copy, tiled).get_slice(mma_tid)
    fragment = tiled.thr_slice(mma_tid).make_fragment_B(source)
    fx.copy(copy, partition.partition_S(source), partition.retile(fragment))
    return fragment


def _stage_end():
    # 发布下一槽，并确保所有 wave 已读完当前槽，之后才允许复用。
    fx.rocdl.s_waitcnt(lgkmcnt=0)
    fx.rocdl.sched_barrier(0)
    fx.gpu.barrier()
    fx.rocdl.sched_barrier(0)


def _sigmoid(value):
    exponent = fx.Float32(fx.rocdl.exp2(fx.T.f32, fx.arith.unwrap(-value * LOG2E)))
    return fx.Float32(fx.rocdl.rcp(fx.T.f32, fx.arith.unwrap(1.0 + exponent)))


@cache
def make_launchers(rows, config):
    # 编译缓存显式捕获标量，不能只依赖 Config 对象的字段哈希。
    bm, dn, un, bk = config.block_m, config.down_n, config.up_n, config.block_k
    split, waves, down_waves = config.split_k, config.waves, config.down_waves
    hidden_pad = config.hidden_pad
    hidden_stride = R + hidden_pad
    wave_split = config.down_mode == "wave_splitk"
    padded_rows = (rows + bm - 1) // bm * bm
    threads = waves * 64
    b_packet_n = 64
    up_widths, up_offsets = (128, 128, 64), (0, 128, 256)
    assert 2 * b_packet_n * bk * 2 + 2 * bm * hidden_stride * 2 + bm * un * 4 <= 65536

    @fx.struct
    class DownShared:
        b: fx.Array[fx.BFloat16, 2 * dn * bk * (down_waves if wave_split else 1), 16]
        partials: fx.Array[fx.Float32, bm * dn * down_waves if wave_split else 1, 16]

    @fx.struct
    class UpShared:
        hidden: fx.Array[fx.BFloat16, bm * hidden_stride, 16]
        hidden_low: fx.Array[fx.BFloat16, bm * hidden_stride, 16]
        b: fx.Array[fx.BFloat16, 2 * b_packet_n * bk, 16]
        logits: fx.Array[fx.Float32, bm * un, 16]

    @flyc.kernel(known_block_size=[256, 1, 1])
    def gr_read_down(X: fx.Tensor, W: fx.Tensor, P: fx.Tensor):
        tid = fx.Int32(fx.thread_idx.x)
        lane, wave = tid % 64, tid // 64
        im, jn, sk = fx.block_idx
        x = fx.rocdl.make_buffer_tensor(fx.make_view(fx.get_iter(X), fx.make_layout((rows, K), (K, 1))),
                                       max_size=False, num_records_bytes=rows * K * 2)
        weights = fx.rocdl.make_buffer_tensor(W, max_size=False)
        shared = fx.SharedAllocator().allocate(DownShared).peek()
        if fx.const_expr(wave_split):
            mma_tid, mma_waves, copy_tid, copy_threads = lane, 1, lane, 64
            steps = K // (down_waves * bk)
            b_base = shared.b.ptr + wave * 2 * dn * bk
            partials = shared.partials.view(fx.make_layout((bm, dn, down_waves), (dn, 1, bm * dn)))
            c_tile = partials[None, None, wave]
            copy_c = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32)
        else:
            mma_tid, mma_waves, copy_tid, copy_threads = tid, waves, tid, threads
            steps = K // (split * bk)
            b_base = shared.b.ptr
            output = fx.rocdl.make_buffer_tensor(fx.make_view(fx.get_iter(P),
                fx.make_layout((padded_rows, R, split), (R, 1, padded_rows * R))), max_size=False)
            c_tile = fx.flat_divide(output[None, None, sk], fx.make_tile(bm, dn))[None, None, im, jn]
            copy_c = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
        atom, tiled = _tiled_mma(mma_waves)
        mma_thread = tiled.thr_slice(mma_tid)
        copy_a = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16)
        a_copy = fx.make_tiled_copy_A(copy_a, tiled).get_slice(mma_tid)
        a_tiles = fx.flat_divide(x, fx.make_tile(bm, bk))[None, None, im, None]
        a_fragment = mma_thread.make_fragment_A(a_tiles[None, None, 0])
        a_source, a_registers = a_copy.partition_S(a_tiles), a_copy.retile(a_fragment)
        c_fragment = mma_thread.make_fragment_C(c_tile)
        c_fragment.fill(0)

        def k_begin(q):
            if fx.const_expr(wave_split):
                return (q * down_waves + wave) * bk
            return (sk * steps + q) * bk

        def read_b(q):
            return _read_b_g2r(weights, jn * dn, k_begin(q), K, dn, bk, copy_tid, copy_threads)

        def run_kblock(q, b_prefetched, *, store_next=True, read_next=True):
            # Memory：当前 B s→r；下一 B r→s；下下 B g→r。
            b_fragment = _read_b_s2r(tiled, mma_tid, b_base + (q % 2) * dn * bk, dn, bk)
            fx.copy(copy_a, a_source[None, None, None, k_begin(q) // bk], a_registers)
            if fx.const_expr(store_next):
                _store_b_r2s(b_prefetched, b_base + ((q + 1) % 2) * dn * bk, dn, bk, copy_tid, copy_threads)
            if fx.const_expr(read_next):
                b_next = read_b(q + 2)
            else:
                b_next = b_prefetched
            _stage_end()
            # Compute：纯 BF16 dot，绝不套用 FP8 的 scale/舍入语义。
            fx.gemm(atom, c_fragment, a_fragment, b_fragment, c_fragment)
            return b_next

        b_first = read_b(0)
        _store_b_r2s(b_first, b_base, dn, bk, copy_tid, copy_threads)
        _stage_end()
        b_prefetched = read_b(1)
        b_carrier = fx.make_fragment_like(b_prefetched)
        # 两个尾拍单独排空，绝不读取下一个 split 或权重末尾之外的 packet。
        for q, state in range(fx.Index(0), fx.Index(steps - 2), fx.Index(1),
                              init=[c_fragment.load(), b_prefetched.load()]):
            c_fragment.store(state[0])
            b_carrier.store(state[1])
            b_next = run_kblock(fx.Int32(q), b_carrier)
            result = yield [c_fragment.load(), b_next.load()]
        c_fragment.store(result[0])
        b_carrier.store(result[1])
        run_kblock(steps - 2, b_carrier, read_next=False)
        run_kblock(steps - 1, b_carrier, store_next=False, read_next=False)
        c_copy = fx.make_tiled_copy_C(copy_c, tiled).get_slice(mma_tid)
        fx.copy(copy_c, c_copy.retile(c_fragment), c_copy.partition_D(c_tile))

        if fx.const_expr(wave_split):
            _stage_end()
            activation = fx.make_view(fx.get_iter(P), fx.make_layout((padded_rows, R), (R, 1)))
            for turn in range_constexpr(bm * dn // threads):
                index = tid + turn * threads
                row, col = index // dn, index % dn
                total = fx.Float32(0.0)
                for s in range_constexpr(down_waves):
                    total = total + fx.memref_load(partials, (row, col, s))
                z = total * (1.0 / C)
                activated = z * _sigmoid(z)
                fx.memref_store(activated, activation, (im * bm + row, jn * dn + col))

    @flyc.kernel(known_block_size=[256, 1, 1])
    def gr_read_up_gate(X: fx.Tensor, W: fx.Tensor, P: fx.Tensor, Y: fx.Tensor):
        tid = fx.Int32(fx.thread_idx.x)
        im, jn, _ = fx.block_idx
        shared = fx.SharedAllocator().allocate(UpShared).peek()
        hidden = shared.hidden.view(fx.make_layout((bm, R), (hidden_stride, 1)))
        hidden_low = shared.hidden_low.view(fx.make_layout((bm, R), (hidden_stride, 1)))
        h4 = shared.hidden.view(fx.make_layout((4, (R // 4, bm)), (1, (4, hidden_stride))))
        h_low4 = shared.hidden_low.view(fx.make_layout((4, (R // 4, bm)), (1, (4, hidden_stride))))
        p4 = fx.flat_divide(fx.rocdl.make_buffer_tensor(P, max_size=False), fx.make_tile(4))
        p_fragment = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Float32)
        h_fragment = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.BFloat16)
        copy_p = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Float32)
        copy_h = fx.make_copy_atom(fx.UniversalCopy64b(), fx.BFloat16)
        for turn in range_constexpr(bm * R // (threads * 4)):
            index = tid + turn * threads
            if fx.const_expr(wave_split):
                fx.copy(copy_p, p4[None, im * bm * (R // 4) + index], p_fragment)
                activated = p_fragment.load()
            else:
                total = fx.Vector.filled(4, 0.0, fx.Float32)
                for s in range_constexpr(split):
                    offset = (s * padded_rows + im * bm) * (R // 4) + index
                    fx.copy(copy_p, p4[None, offset], p_fragment)
                    total = total + p_fragment.load()
                z = total * (1.0 / C)
                activated = fx.Vector.from_elements([z[j] * _sigmoid(z[j]) for j in range_constexpr(4)], fx.Float32)
            high = activated.to(fx.BFloat16)
            h_fragment.store(high)
            fx.copy(copy_h, h_fragment, h4[None, index])
            h_fragment.store((activated - high.to(fx.Float32)).to(fx.BFloat16))
            fx.copy(copy_h, h_fragment, h_low4[None, index])
        _stage_end()

        atom, tiled = _tiled_mma(waves)
        mma_thread = tiled.thr_slice(tid)
        a_copy = fx.make_tiled_copy_A(copy_h, tiled).get_slice(tid)
        a_high, a_low_views = [], []
        # 与 k128n 相同，A high 驻留寄存器；low 不预取，按需读取以缩短生存期。
        for ks in range_constexpr(3):
            hi = fx.make_view(fx.get_iter(hidden) + up_offsets[ks],
                              fx.make_layout((bm, up_widths[ks]), (hidden_stride, 1)))
            lo = fx.make_view(fx.get_iter(hidden_low) + up_offsets[ks], hi.layout)
            hi_fragment = mma_thread.make_fragment_A(hi)
            fx.copy(copy_h, a_copy.partition_S(hi), a_copy.retile(hi_fragment))
            a_high.append(hi_fragment)
            a_low_views.append(lo)

        weights = fx.rocdl.make_buffer_tensor(W, max_size=False)
        logits = shared.logits.view(fx.make_layout((bm, un), (un, 1)))
        c_tiles, c_fragments = [], []
        for half in range_constexpr(2):
            c_tile = fx.make_view(fx.get_iter(logits) + half * b_packet_n,
                                 fx.make_layout((bm, b_packet_n), (un, 1)))
            fragment = mma_thread.make_fragment_C(c_tile)
            fragment.fill(0)
            c_tiles.append(c_tile)
            c_fragments.append(fragment)

        def read_b(q):
            return _read_b_g2r(weights, jn * un + (q // 3) * b_packet_n, up_offsets[q % 3],
                               R, b_packet_n, up_widths[q % 3], tid, threads)

        b_first = read_b(0)
        _store_b_r2s(b_first, shared.b.ptr, b_packet_n, up_widths[0], tid, threads)
        _stage_end()
        b_prefetched = read_b(1)
        for q in range_constexpr(6):
            ks, half = q % 3, q // 3
            b_fragment = _read_b_s2r(tiled, tid, shared.b.ptr + (q % 2) * b_packet_n * bk,
                                     b_packet_n, up_widths[ks])
            if fx.const_expr(q + 1 < 6):
                _store_b_r2s(b_prefetched, shared.b.ptr + ((q + 1) % 2) * b_packet_n * bk,
                             b_packet_n, up_widths[(q + 1) % 3], tid, threads)
            if fx.const_expr(q + 2 < 6):
                b_prefetched = read_b(q + 2)
            _stage_end()
            fx.gemm(atom, c_fragments[half], a_high[ks], b_fragment, c_fragments[half])
            a_low = mma_thread.make_fragment_A(a_low_views[ks])
            fx.copy(copy_h, a_copy.partition_S(a_low_views[ks]), a_copy.retile(a_low))
            fx.gemm(atom, c_fragments[half], a_low, b_fragment, c_fragments[half])
        copy_c = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32)
        c_copy = fx.make_tiled_copy_C(copy_c, tiled).get_slice(tid)
        for half in range_constexpr(2):
            fx.copy(copy_c, c_copy.retile(c_fragments[half]), c_copy.partition_D(c_tiles[half]))
        _stage_end()

        x = fx.make_view(fx.get_iter(X), fx.make_layout((rows, K), (K, 1)))
        y = fx.make_view(fx.get_iter(Y), fx.make_layout((rows, H), (H, 1)))
        for turn in range_constexpr(bm * (un // C) // threads):
            index = tid + turn * threads
            row_local, col_local = index // (un // C), index % (un // C)
            row, col = im * bm + row_local, jn * (un // C) + col_local
            if row < rows:
                total = fx.Float32(0.0)
                for stream_id in range_constexpr(C):
                    gate = _sigmoid(fx.memref_load(logits, (row_local, col_local * C + stream_id)))
                    value = fx.memref_load(x, (row, stream_id * H + col)).to(fx.Float32)
                    total = total + gate * value
                fx.memref_store((total * (1.0 / C)).to(fx.BFloat16), y, (row, col))

    @flyc.jit
    def launch_down(X: fx.Tensor, W: fx.Tensor, P: fx.Tensor, stream: fx.Stream):
        gr_read_down(X, W, P).launch(grid=(padded_rows // bm, R // dn, 1 if wave_split else split),
                                    block=(threads, 1, 1), stream=stream)

    @flyc.jit
    def launch_up(X: fx.Tensor, W: fx.Tensor, P: fx.Tensor, Y: fx.Tensor, stream: fx.Stream):
        gr_read_up_gate(X, W, P, Y).launch(grid=(padded_rows // bm, K // un, 1),
                                         block=(threads, 1, 1), stream=stream)

    @flyc.jit
    def launch_combined(X: fx.Tensor, WD: fx.Tensor, WU: fx.Tensor, P: fx.Tensor, Y: fx.Tensor, stream: fx.Stream):
        if fx.const_expr(rows > 0 and hidden_pad >= 0):
            launch_down(X, WD, P, stream)
            launch_up(X, WU, P, Y, stream)

    return launch_down, launch_up, launch_combined