# SPDX-License-Identifier: MIT
"""GRRead Up N2/N4/N8/N10/N20/N40 with cooperative X128 loads and two-stage W prefetch.

X/P loads and Y stores use NT; W uses the default policy. P is BF16 with full K320 and FP32 logits.
The four streams for each H64 use the original FMA order and the integer BF16 output helper.
Two 20 KiB B LDS slots plus 3 KiB of X LDS per wave use 64 KiB total with 512 threads.
Each substage loads future X while P loads overlap startup X/B LDS transfers.
N splits are an internal compile-time parameter selected by batch, with no experimental switches.
"""
from functools import cache

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, range_constexpr, rocdl
from pyhip.contrib.flydsl.helpers import cvt_f32_to_bf16

from .common import H, K, R
from .helpers import (
    _barrier,
    _ds_read,
    _ds_write,
    _load,
    _schedule_boundary,
    _mfma,
    _pack_y_mean,
    _pin_v,
    _priority,
    _sigmoid,
)

BM = 256
B_SLOT_BYTES = 32 * R * 2
GROUP_STEPS = 8
SUB_WAITS = (
    ((1, 3), (3, 4), (2, 3), (3, 4), (2, 3), (3, 4), (2, 3), (3, 4)),
    ((2, 3), (4, 5), (3, 4), (4, 5), (3, 4), (3, 4), (2, 3), (3, 4)),
    ((2, 3), (4, 5), (3, 4), (4, 5), (3, 4), (3, 4), (2, 0), (0, 0)),
    # N40单组：没有旧Y写出，前六拍沿用FIRST；末两拍停止W/X预取并排空。
    ((1, 3), (3, 4), (2, 3), (3, 4), (2, 3), (3, 4), (2, 0), (0, 0)),
)


def _load_nt(resource, address, scalar_offset=0, words=4):
    value = rocdl.RawPtrBufferLoadOp(
        fx.Int32.ir_type if words == 1 else ir.VectorType.get([words], fx.Int32.ir_type), resource,
        address.ir_value(), fx.Int32(scalar_offset).ir_value(),
        aux=ir.IntegerAttr.get(fx.Int32.ir_type, 2)).result
    return fx.Int32(value) if words == 1 else fx.Vector(value)


def _store_nt(resource, address, values, scalar_offset=0):
    rocdl.RawPtrBufferStoreOp(values.ir_value(), resource, address.ir_value(),
        fx.Int32(scalar_offset).ir_value(), aux=ir.IntegerAttr.get(fx.Int32.ir_type, 2))


def _pack_y_from_mean(v0, v1):
    fragment = fx.make_rmem_tensor(fx.make_layout(2, 1), fx.Float32)
    fragment.store(fx.Vector.from_elements([v0, v1], fx.Float32))
    return fx.Vector(cvt_f32_to_bf16(fragment).load()).bitcast(fx.Int32)[0]


@cache
def make_up(*, n_splits):
    """Build the unified Up launcher without queue state, fallbacks, or alternate cache policies.

    Each CTA processes 40/n_splits whole H64 groups, with eight H32 packets per group.
    Runtime rows share one artifact across full tiles and tails; the grid covers actual rows only.
    Callers must provide positive rows and X/P/Y storage covering all actual rows.
    Automatic dispatch includes N10/N20 for calibrated small batches and retains N2/N4/N8 elsewhere.
    N40 is selected for calibrated batches up to 512 rows on 80-CU devices.
    """
    if n_splits not in (2, 4, 8, 10, 20, 40):
        raise ValueError('n_splits must be 2, 4, 8, 10, 20 or 40')
    N_SPLITS = n_splits
    PACKETS = K // 32 // N_SPLITS
    GROUPS = PACKETS // GROUP_STEPS

    @fx.struct
    class Shared:
        b: fx.Array[fx.Int32, (2 * B_SLOT_BYTES + 24576) // 4, 16]

    @flyc.kernel(known_block_size=[512, 1, 1])
    def gr_read_up(X: fx.Tensor, W: fx.Tensor, P: fx.Tensor, Y: fx.Tensor, rows: fx.Int64):
        tid = fx.Int32(fx.thread_idx.x)
        lane, wave = tid % 64, tid // 64
        group = fx.Int32(rocdl.readfirstlane(fx.Int32.ir_type, (tid // 256).ir_value()))
        # 完整M-major逻辑覆盖；正确性不依赖block编号到物理XCC的映射。
        worker = fx.Int32(fx.block_idx.x)
        block_m = (worker.to(fx.Uint32) // N_SPLITS).to(fx.Int32)
        n_task = (worker.to(fx.Uint32) % N_SPLITS).to(fx.Int32)
        shared = fx.SharedAllocator().allocate(Shared).peek()
        b_base = fx.Int32(fx.ptrtoint(shared.b.ptr))
        _priority(1)
        _schedule_boundary()
        task_tid = fx.Uint32(llvm.inline_asm(fx.Uint32.ir_type, [tid.ir_value()], '', '=v,0', has_side_effects=True))
        lane, wave = fx.Int32(task_tid & 63), fx.Int32(task_tid >> 6)
        n_base, n_phase = n_task * PACKETS, fx.Int32(0)
        # X/P/Y在CTA入口用64位元素偏移重设基址；循环内地址只相对本M256 tile。
        row_begin = block_m.to(fx.Int64) * BM
        remaining = rows - row_begin
        valid_rows = (remaining > 0).select(remaining, fx.Int64(0))
        valid_rows = (valid_rows < BM).select(valid_rows, fx.Int64(BM))
        tile_x = fx.make_view(fx.get_iter(X) + row_begin * K, fx.make_layout(BM * K, 1))
        tile_p = fx.make_view(fx.get_iter(P) + row_begin * R, fx.make_layout(BM * R, 1))
        tile_y = fx.make_view(fx.get_iter(Y) + row_begin * H, fx.make_layout(BM * H, 1))
        buffers = [fx.rocdl.make_buffer_tensor(W, max_size=False),
                   fx.rocdl.make_buffer_tensor(tile_p, max_size=False, num_records_bytes=valid_rows * R * 2),
                   fx.rocdl.make_buffer_tensor(tile_x, max_size=False, num_records_bytes=valid_rows * K * 2),
                   fx.rocdl.make_buffer_tensor(tile_y, max_size=False, num_records_bytes=valid_rows * H * 2)]
        wr, pr, xr, yr = [fx.rocdl.get_buffer_rsrc(fx.get_iter(t)) for t in buffers]
        row = wave * 32 + lane % 16
        p_address, x_address, y_address = [], [], []
        for mi in range_constexpr(2):
            mr = row + mi * 16
            p_address.append(_pin_v(mr * R * 2 + lane // 16 * 16))
            y_address.append(_pin_v(mr * H * 2 + lane // 16 * 16))
        w16, w8 = _pin_v(tid * 16), _pin_v(tid * 8)
        b_read = _pin_v(b_base + lane % 16 * 16 + lane // 16 * 256)
        b_write16, b_write8 = _pin_v(b_base + tid * 16), _pin_v(b_base + tid * 8)

        # 每row八个lane，各16B；四条load覆盖本wave的M32/H64。
        # 行偏移统一放入vector address，让同一buffer extent正确屏蔽任意尾行。
        x_pair_address = []
        x_row8 = wave * 32 + lane % 8
        for part in range_constexpr(4):
            mr = x_row8 + part * 8
            x_pair_address.append(_pin_v(mr * K * 2 + lane // 8 * 16))
        # 每wave3KiB：低半M16的1KiB分时复用，高半M32保留2KiB。
        x_lds_base = b_base + 2 * B_SLOT_BYTES + wave * 3072
        x_write = [_pin_v(x_lds_base + lane % 8 * 16 + lane // 8 % 4 * 256 + lane // 32 * (1024 + mi * 1024))
                   for mi in range_constexpr(2)]
        x_read = _pin_v(x_lds_base + lane * 16)

        def packet(q):
            shifted = q + n_phase
            return (n_base + (shifted >= PACKETS).select(shifted - PACKETS, shifted)).to(fx.Uint32)

        def read_b_g2r(q, part):
            return _load(wr, w8 if part == 2 else w16, packet(q) * B_SLOT_BYTES + part * 8192,
                         words=2 if part == 2 else 4)

        def store_b_r2s(slot, part, value):
            if const_expr(part == 2):
                llvm.inline_asm(ir.Type.parse('!llvm.void'), [b_write8.ir_value(), value.ir_value()],
                    f'ds_write_b64 $0,$1 offset:{slot * B_SLOT_BYTES + 16384}', 'v,v,~{memory}', has_side_effects=True)
            else:
                _ds_write(b_write16, value, slot * B_SLOT_BYTES + part * 8192)

        def load_x_pair_g2r(q, step, part):
            return _load_nt(xr, x_pair_address[part], packet(q) // 8 * 128 + step // 2 * H * 2)

        def x_pair_r2s_s2r(first8, second8, mi, defer_low=False):
            _ds_write(x_write[mi], first8, 0)
            _ds_write(x_write[mi], second8, 128)
            rocdl.s_waitcnt(lgkmcnt=0)
            low = _ds_read(x_read, 0)
            if const_expr(not defer_low):
                rocdl.s_waitcnt(lgkmcnt=0)
            return low

        def store_y_r2g(q, y):
            for mi in range_constexpr(2):
                _store_nt(yr, y_address[mi], fx.Vector.from_elements(y[mi * 4:mi * 4 + 4], fx.Int32),
                       packet(q) // 8 * 128 + packet(q) % 2 * 64)

        def post(mi, h16, m, stream, half, c, x, totals):
            value = (x[mi][h16 * 2 + m // 2].bitcast(fx.Uint32) >> (m % 2 * 16)).to(fx.Uint16).bitcast(fx.BFloat16).to(fx.Float32)
            gate = _sigmoid(c[mi * 2 + h16][m])
            index = mi * 16 + half * 8 + h16 * 4 + m
            if const_expr(stream == 0):
                totals[index] = fx.Float32(0.0)
            totals[index] = fx.Float32(llvm.call_intrinsic(fx.Float32.ir_type, 'llvm.fma.f32',
                [gate.ir_value(), value.ir_value(), totals[index].ir_value()], [], []))

        def pack_pair(mi, h16, pair, half, totals):
            index = mi * 16 + half * 8 + h16 * 4 + pair * 2
            return _pin_v(_pack_y_mean(totals[index], totals[index + 1]))

        def run_group(g, b_g2r, c_previous, x_previous, x_pair_g2r, totals, y, *, first=False, last=False, x_initial=None):
            x_pair_r = x_initial if const_expr(first) else [fx.Vector.filled(4, 0, fx.Int32) for _ in range_constexpr(2)]
            x_drain = [fx.Vector.filled(4, 0, fx.Int32) for _ in range_constexpr(2)]
            first_packet = packet(g * GROUP_STEPS)
            next_packet = packet((g + 1) * GROUP_STEPS)
            if const_expr(not first):
                y_group_offset = packet((g - 1) * GROUP_STEPS) // 8 * 128
            for step in range_constexpr(GROUP_STEPS):
                q = g * GROUP_STEPS + step
                if const_expr(step == 0):
                    y_low_r2g = y[:]
                for h16 in range_constexpr(2):
                    if const_expr(first and step == 0 and h16 == 0):
                        _priority(0)
                    _schedule_boundary()
                    rocdl.s_waitcnt(vmcnt=SUB_WAITS[3 if first and last else 0 if first else 2 if last else 1][step][h16])
                    if const_expr(step % 2 == 0 and (not first or step > 0)):
                        if const_expr(h16 == 0):
                            x_previous = [_ds_read(x_read, 1024 + mi * 1024) for mi in range_constexpr(2)]
                            rocdl.s_waitcnt(lgkmcnt=0)
                        x_pair_r[h16] = x_pair_r2s_s2r(x_pair_g2r[h16 * 2], x_pair_g2r[h16 * 2 + 1], h16, defer_low=True)
                    b_s2r = [_ds_read(b_read, step % 2 * B_SLOT_BYTES + h16 * 10240 + ki * 1024)
                             for ki in range_constexpr(10)]
                    for part in range_constexpr(3):
                        if const_expr(part // 2 == h16):
                            if const_expr(not last or step < 7):
                                store_b_r2s((step + 1) % 2, part, b_g2r[part])
                            if const_expr(not last or step < 6):
                                w_packet = (next_packet if const_expr(step >= 6) else first_packet) + (step + 2) % 8
                                b_g2r[part] = _load(wr, w8 if part == 2 else w16,
                                    w_packet * B_SLOT_BYTES + part * 8192, words=2 if part == 2 else 4)
                    if const_expr(not last or step < 6):
                        part = h16 * 2 + step % 2
                        x_pair_q = q + (2 if const_expr(step % 2 == 0) else 1)
                        x_pair_g2r[part] = load_x_pair_g2r(x_pair_q, ((step // 2 + 1) * 2) % 8, part)
                    if const_expr(h16 == 1 and step < 4 and not first):
                        old_q = g * GROUP_STEPS - 2 + step // 2
                        mi = step % 2
                        words = y_low_r2g if const_expr(step < 2) else y
                        _store_nt(yr, y_address[mi], fx.Vector.from_elements(words[mi * 4:mi * 4 + 4], fx.Int32),
                               y_group_offset + step // 2 * 64)
                    if const_expr(last and step == 7 and h16 == 1):
                        x_drain = [_ds_read(x_read, 1024 + mi * 1024) for mi in range_constexpr(2)]
                    rocdl.s_waitcnt(lgkmcnt=0)
                    _schedule_boundary()
                    _barrier()
                    _priority(3)
                    _schedule_boundary()
                    c_f32 = [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range_constexpr(2)]
                    scaled, exponent, x_f32, denominator, gates, y_mean = [[fx.Float32(0.0) for _ in range_constexpr(8)] for _ in range_constexpr(6)]
                    for ordinal in range_constexpr(40):
                        mi, ki = ordinal % 2, ordinal // 2
                        b = b_s2r[ki // 2].bitcast(fx.BFloat16)
                        b = b.shuffle(b, list(range(ki % 2 * 4, ki % 2 * 4 + 4)))
                        c_f32[mi] = _mfma(b, p_bf16[mi][ki], c_f32[mi])
                        rocdl.sched_barrier(0)
                        # 每个MFMA间隔只一条exp/rcp，或至多三条普通VALU。
                        pair, turn = ordinal // 10, ordinal % 10
                        if const_expr((not first or step > 0) and turn == 0):
                            scaled[pair * 2] = -c_previous[pair // 2 * 2 + h16][pair % 2 * 2] * 1.4426950408889634
                            scaled[pair * 2 + 1] = -c_previous[pair // 2 * 2 + h16][pair % 2 * 2 + 1] * 1.4426950408889634
                            value = x_previous[pair // 2][h16 * 2 + pair % 2]
                            x_f32[pair * 2] = value.bitcast(fx.Uint32).to(fx.Uint16).bitcast(fx.BFloat16).to(fx.Float32)
                        if const_expr((not first or step > 0) and turn in (1, 2)):
                            post_index = pair * 2 + turn - 1
                            exponent[post_index] = fx.Float32(rocdl.exp2(fx.Float32.ir_type, scaled[post_index].ir_value()))
                        if const_expr((not first or step > 0) and turn == 3):
                            denominator[pair * 2] = 1.0 + exponent[pair * 2]
                            denominator[pair * 2 + 1] = 1.0 + exponent[pair * 2 + 1]
                            value = x_previous[pair // 2][h16 * 2 + pair % 2]
                            x_f32[pair * 2 + 1] = (value.bitcast(fx.Uint32) >> 16).to(fx.Uint16).bitcast(fx.BFloat16).to(fx.Float32)
                        if const_expr((not first or step > 0) and turn in (4, 5)):
                            post_index = pair * 2 + turn - 4
                            gates[post_index] = fx.Float32(rocdl.rcp(fx.Float32.ir_type, denominator[post_index].ir_value()))
                        if const_expr((not first or step > 0) and turn in (6, 7)):
                            post_index = pair * 2 + turn - 6
                            index = post_index // 4 * 16 + (step - 1) % 2 * 8 + h16 * 4 + post_index % 4
                            if const_expr((step - 1) % 8 // 2 == 0):
                                totals[index] = fx.Float32(0.0)
                            totals[index] = fx.Float32(llvm.call_intrinsic(fx.Float32.ir_type, 'llvm.fma.f32',
                                [gates[post_index].ir_value(), x_f32[post_index].ir_value(), totals[index].ir_value()], [], []))
                            if const_expr(last):
                                totals[index] = _pin_v(totals[index].bitcast(fx.Int32)).bitcast(fx.Float32)
                        if const_expr(((step == 0 and not first) or step == 7) and turn in (7, 8)):
                            post_index = pair * 2 + turn - 7
                            index = post_index // 4 * 16 + (step - 1) % 2 * 8 + h16 * 4 + post_index % 4
                            y_mean[post_index] = totals[index] * 0.25
                        if const_expr(((step == 0 and not first) or step == 7) and turn == 9):
                            y[pair // 2 * 4 + h16 * 2 + pair % 2] = _pin_v(_pack_y_from_mean(y_mean[pair * 2], y_mean[pair * 2 + 1]))
                        rocdl.sched_barrier(0)
                    for mi in range_constexpr(2):
                        c_previous[mi * 2 + h16] = c_f32[mi]
                    _schedule_boundary()
                    _priority(0)
                    _barrier()
                if const_expr(step % 2 == 0):
                    x_previous = x_pair_r[:]
            return b_g2r, c_previous, x_drain, x_pair_g2r, totals, y

        b0 = [read_b_g2r(fx.Int32(0), part) for part in range_constexpr(3)]
        x_pair_g2r = [load_x_pair_g2r(fx.Int32(0), 0, part) for part in range_constexpr(4)]
        p_bf16 = []
        for mi in range_constexpr(2):
            fragments = []
            for kb in range_constexpr(5):
                for k32 in range_constexpr(2):
                    both = _load_nt(pr, p_address[mi], kb * 128 + k32 * 64, words=4)
                    for k4 in range_constexpr(2):
                        fragments.append(both.shuffle(both, [k4 * 2, k4 * 2 + 1]).bitcast(fx.BFloat16))
            p_bf16.append(fragments)

        # P到FIRST0的MFMA才消费，允许它与X/B LDS启动重叠。
        # 源序预算vmcnt20之外，LLVM可能为重排后的真实消费者补更严等待。
        rocdl.s_waitcnt(vmcnt=20)
        x_initial = [fx.Vector.filled(4, 0, fx.Int32) for _ in range_constexpr(2)]
        _ds_write(x_write[0], x_pair_g2r[0], 0)
        _ds_write(x_write[0], x_pair_g2r[1], 128)
        for part in range_constexpr(3):
            store_b_r2s(0, part, b0[part])
        rocdl.s_waitcnt(lgkmcnt=0)
        x_initial[0] = _ds_read(x_read, 0)
        b_g2r = [read_b_g2r(fx.Int32(1), part) for part in range_constexpr(3)]
        rocdl.s_waitcnt(lgkmcnt=0)
        _ds_write(x_write[1], x_pair_g2r[2], 0)
        _ds_write(x_write[1], x_pair_g2r[3], 128)
        rocdl.s_waitcnt(lgkmcnt=0)
        x_initial[1] = _ds_read(x_read, 0)
        rocdl.s_waitcnt(lgkmcnt=0)
        _barrier()
        if group == 1:
            _barrier()
        _schedule_boundary()
        state = run_group(fx.Int32(0), b_g2r,
            [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range_constexpr(4)],
            [fx.Vector.filled(4, 0, fx.Int32) for _ in range_constexpr(2)], x_pair_g2r,
            [fx.Float32(0.0) for _ in range_constexpr(32)],
            [fx.Int32(0) for _ in range_constexpr(8)], first=True, last=GROUPS == 1, x_initial=x_initial)

        def save(state):
            b, c, x, xn, totals, y = state
            return b + c + x + xn + totals + y

        def restore(values):
            return ([fx.Vector(values[i]) for i in range_constexpr(3)],
                    [fx.Vector(values[i]) for i in range_constexpr(3, 7)],
                    [fx.Vector(values[i]) for i in range_constexpr(7, 9)],
                    [fx.Vector(values[i]) for i in range_constexpr(9, 13)],
                    [fx.Float32(values[i]) for i in range_constexpr(13, 45)],
                    [fx.Int32(values[i]) for i in range_constexpr(45, 53)])

        if const_expr(GROUPS == 1):
            # N40的同一组只执行一次，直接将最后packet的状态交给公共drain。
            _, c_previous, x_previous, _, totals, y = state
        else:
            if const_expr(GROUPS > 2):
                for g, values in range(fx.Index(1), fx.Index(GROUPS - 1), fx.Index(1), init=save(state)):
                    state = run_group(fx.Int32(g), *restore(values))
                    result = yield save(state)
            else:
                # N20仅有FIRST和LAST两组；直接传递FIRST状态，不生成空稳态循环。
                result = save(state)
            _, c_previous, x_previous, _, totals, y = run_group(fx.Int32(GROUPS - 1), *restore(result), last=True)
        if group == 0:
            _barrier()
        _priority(0)
        _schedule_boundary()
        store_y_r2g(fx.Int32(PACKETS - 2), y)
        _schedule_boundary()
        _priority(3)
        _schedule_boundary()
        for mi in range_constexpr(2):
            for h16 in range_constexpr(2):
                for m in range_constexpr(4):
                    post(mi, h16, m, 3, 1, c_previous, x_previous, totals)
        y = [pack_pair(mi, h16, pair, 1, totals) for mi in range_constexpr(2)
             for h16 in range_constexpr(2) for pair in range_constexpr(2)]
        _schedule_boundary()
        _priority(0)
        _schedule_boundary()
        store_y_r2g(fx.Int32(PACKETS - 1), y)
        _schedule_boundary()
        _priority(1)
        _schedule_boundary()
        rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0)
        _barrier()
        _schedule_boundary()

    @flyc.jit
    def launch_up(X: fx.Tensor, W: fx.Tensor, P: fx.Tensor, Y: fx.Tensor,
                  rows: fx.Int64, stream: fx.Stream):
        gr_read_up(X, W, P, Y, rows).launch(grid=(((rows + BM - 1) // BM) * N_SPLITS, 1, 1), block=(512, 1, 1), stream=stream)

    launch_up.compile_hints['llvm_options'] = {'vectorize-slp': False}
    return launch_up