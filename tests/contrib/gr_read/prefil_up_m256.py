# SPDX-License-Identifier: MIT
"""GR read Up M256：完整K320、两个H16子阶段，FP32 logits与整数helper Y打包。

每wave两组M16共享B；P整K常驻，B双20KiB LDS。4+4 wave错相，真实Memory零VALU。
X0在prologue读取，u0/u1各读取X(q+1)的一组M16；无末包预取和额外全局预排。
任务width8前缀转置、N四相位；每8包复用无符号地址，P相邻片段合并128-bit启动加载。
旧Y在step0..3的第二子阶段各写一份M16，放在未来B/X读取之后。
MFMA计算q，VALU处理q-1；每个MFMA间隔仅一条exp/rcp，或最多三条普通VALU。
"""
from functools import cache

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, range_constexpr, rocdl
from pyhip.contrib.flydsl.helpers import cvt_f32_to_bf16

if __package__:
    from .kernel import H, K, R
    from .prefil_up_8x1 import _barrier, _ds_read, _ds_write, _load, _mark, _mfma, _pack_y_mean, _pin_v, _priority, _sigmoid, _store
else:
    from kernel import H, K, R
    from prefil_up_8x1 import _barrier, _ds_read, _ds_write, _load, _mark, _mfma, _pack_y_mean, _pin_v, _priority, _sigmoid, _store

BM, N_SPLITS = 256, 2
B_SLOT_BYTES = 32 * R * 2
PACKETS = K // 32 // N_SPLITS
GROUP_STEPS = 8
GROUPS = PACKETS // GROUP_STEPS
# 各H32包的两个子阶段：B的3次搬运分成2+1；u0/u1各读下一X的mi0/mi1。
# step0..3的第二段在B/X读取之后各写一次旧Y；wait仍保护B(q+1)及X(q-1)。
SUB_WAITS = (
    ((4, 6), (6, 8), (5, 8), (5, 8), (5, 8), (5, 8), (5, 8), (5, 8)),
    ((5, 8), (6, 9), (7, 10), (7, 10), (7, 10), (6, 9), (5, 8), (5, 8)),
    ((5, 8), (6, 9), (7, 10), (7, 10), (7, 10), (6, 7), (2, 3), (2, 0)),
)


def _pack_y_from_mean(v0, v1):
    """均值已在前一个MFMA间隔算完，此处只调用真实整数BF16 helper。"""
    fragment = fx.make_rmem_tensor(fx.make_layout(2, 1), fx.Float32)
    fragment.store(fx.Vector.from_elements([v0, v1], fx.Float32))
    return fx.Vector(cvt_f32_to_bf16(fragment).load()).bitcast(fx.Int32)[0]


@cache
def make_up_m256(rows, padded_rows):
    assert 0 < rows <= padded_rows and padded_rows % BM == 0

    @fx.struct
    class Shared:
        b: fx.Array[fx.Int32, 2 * B_SLOT_BYTES // 4, 16]

    @flyc.kernel(known_block_size=[512, 1, 1])
    def gr_read_up_m256_n16(X: fx.Tensor, W: fx.Tensor, P: fx.Tensor, Y: fx.Tensor):
        tid = fx.Int32(fx.thread_idx.x)
        lane, wave = tid % 64, tid // 64
        group = fx.Int32(rocdl.readfirstlane(fx.Int32.ir_type, (tid // 256).ir_value()))
        m_tiles = (rows + BM - 1) // BM
        worker = fx.Int32(fx.block_idx.x) + fx.Int32(fx.block_idx.y) * m_tiles
        chunk = m_tiles * N_SPLITS // 8
        task = (worker < chunk * 8).select(worker % 8 * chunk + worker // 8, worker)
        first_m = task // N_SPLITS
        width = (m_tiles - first_m < 1).select(m_tiles - first_m, fx.Int32(1))
        within = task % N_SPLITS
        block_m, n_base = first_m + within % width, within // width * PACKETS
        n_phase = block_m % 4 * 40
        row = block_m * BM + wave * 32 + lane % 16
        shared = fx.SharedAllocator().allocate(Shared).peek()
        b_base = fx.Int32(fx.ptrtoint(shared.b.ptr))
        buffers = [fx.rocdl.make_buffer_tensor(t, max_size=False) for t in (W, P, X, Y)]
        wr, pr, xr, yr = [fx.rocdl.get_buffer_rsrc(fx.get_iter(t)) for t in buffers]
        p_address, x_address, y_address = [], [], []
        for mi in range_constexpr(2):
            mr = row + mi * 16
            p_address.append(_pin_v((mr < rows).select(mr * R * 2 + lane // 16 * 16, fx.Int32(0x7fffffff))))
            x_address.append(_pin_v((mr < rows).select(mr * K * 2 + lane // 16 * 16, fx.Int32(0x7fffffff))))
            y_address.append(_pin_v((mr < rows).select(mr * H * 2 + lane // 16 * 16, fx.Int32(0x7fffffff))))
        w16, w8 = _pin_v(tid * 16), _pin_v(tid * 8)
        b_read = _pin_v(b_base + lane % 16 * 16 + lane // 16 * 256)
        b_write16, b_write8 = _pin_v(b_base + tid * 16), _pin_v(b_base + tid * 8)

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

        def load_x_g2r(q, step, mi):
            return _load(xr, x_address[mi], packet(q) // 8 * 128 + step // 2 * H * 2 + step % 2 * 64)

        def store_y_r2g(q, y):
            for mi in range_constexpr(2):
                _store(yr, y_address[mi], fx.Vector.from_elements(y[mi * 4:mi * 4 + 4], fx.Int32),
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

        def run_group(g, b_g2r, c_previous, x_previous, x_g2r, totals, y, *, first=False, last=False):
            first_packet = packet(g * GROUP_STEPS)
            next_packet = packet((g + 1) * GROUP_STEPS)
            if const_expr(not first):
                y_group_offset = packet((g - 1) * GROUP_STEPS) // 8 * 128
            for step in range_constexpr(GROUP_STEPS):
                q = g * GROUP_STEPS + step
                x_current = x_g2r[:]  # 分片更新下一X，保留当前包的SSA快照。
                if const_expr(step == 0):
                    y_low_r2g = y[:]
                for h16 in range_constexpr(2):
                    if const_expr(first and step == 0 and h16 == 0):
                        _priority(0)
                    _mark(f"MEMORY_{'FIRST' if first else 'LAST' if last else 'LOOP'}_{step}_{h16}_BEGIN")
                    b_s2r = [_ds_read(b_read, step % 2 * B_SLOT_BYTES + h16 * 10240 + ki * 1024)
                             for ki in range_constexpr(10)]
                    rocdl.s_waitcnt(vmcnt=SUB_WAITS[0 if first else 2 if last else 1][step][h16])
                    for part in range_constexpr(3):
                        if const_expr(part // 2 == h16):
                            if const_expr(not last or step < 7):
                                store_b_r2s((step + 1) % 2, part, b_g2r[(step % 2) * 3 + part])
                            if const_expr(not last or step < 5):
                                w_packet = (next_packet if const_expr(step >= 5) else first_packet) + (step + 3) % 8
                                b_g2r[(step % 2) * 3 + part] = _load(wr, w8 if part == 2 else w16,
                                    w_packet * B_SLOT_BYTES + part * 8192, words=2 if part == 2 else 4)
                    if const_expr(not last or step < 7):
                        # mi0/mi1各一条128-bit；每条仍同时包含两个H16的X。
                        x_packet = next_packet if const_expr(step == 7) else first_packet
                        x_g2r[h16] = _load(xr, x_address[h16], x_packet // 8 * 128
                            + (step + 1) % 8 // 2 * H * 2 + (step + 1) % 2 * 64)
                    if const_expr(h16 == 1 and step < 4 and not first):
                        # 四拍各写一份M16/H32，低半Y快照防止step0新pack覆盖旧结果。
                        old_q = g * GROUP_STEPS - 2 + step // 2
                        mi = step % 2
                        words = y_low_r2g if const_expr(step < 2) else y
                        _store(yr, y_address[mi], fx.Vector.from_elements(words[mi * 4:mi * 4 + 4], fx.Int32),
                               y_group_offset + step // 2 * 64)
                    rocdl.s_waitcnt(lgkmcnt=0)
                    _mark(f"MEMORY_{'FIRST' if first else 'LAST' if last else 'LOOP'}_{step}_{h16}_END")
                    _barrier()
                    _priority(3)
                    _mark(f'COMPUTE_{step}_{h16}_BEGIN')
                    c_f32 = [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range_constexpr(2)]
                    scaled, exponent, x_f32, denominator, gates, y_mean = [[fx.Float32(0.0) for _ in range_constexpr(8)] for _ in range_constexpr(6)]
                    for ordinal in range_constexpr(40):
                        mi, ki = ordinal % 2, ordinal // 2
                        b = b_s2r[ki // 2].bitcast(fx.BFloat16)
                        b = b.shuffle(b, list(range(ki % 2 * 4, ki % 2 * 4 + 4)))
                        c_f32[mi] = _mfma(b, p_bf16[mi][ki], c_f32[mi])
                        rocdl.sched_barrier(0)
                        # 每10条MFMA处理一对旧元素；turn1/2/4/5仅exp/rcp。
                        # 其余turn最多3条普通VALU；不pin中间值，不用额外MFMA填空。
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
                    _mark(f'COMPUTE_{step}_{h16}_END')
                    _priority(0)
                    _barrier()
                x_previous = x_current
            return b_g2r, c_previous, x_previous, x_g2r, totals, y

        b0 = [read_b_g2r(fx.Int32(0), part) for part in range_constexpr(3)]
        x_g2r = [load_x_g2r(fx.Int32(0), 0, mi) for mi in range_constexpr(2)]
        p_bf16 = []
        for mi in range_constexpr(2):
            fragments = []
            for kb in range_constexpr(5):
                for k32 in range_constexpr(2):
                    both = _load(pr, p_address[mi], kb * 128 + k32 * 64, words=4)
                    for k4 in range_constexpr(2):
                        fragments.append(both.shuffle(both, [k4 * 2, k4 * 2 + 1]).bitcast(fx.BFloat16))
            p_bf16.append(fragments)

        rocdl.s_waitcnt(vmcnt=0)
        for part in range_constexpr(3):
            store_b_r2s(0, part, b0[part])
        rocdl.s_waitcnt(lgkmcnt=0)
        _barrier()
        b_g2r = [read_b_g2r(fx.Int32(q), part) for q in range_constexpr(1, 3) for part in range_constexpr(3)]
        if group == 1:
            _barrier()
        state = run_group(fx.Int32(0), b_g2r,
            [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range_constexpr(4)],
            [fx.Vector.filled(4, 0, fx.Int32) for _ in range_constexpr(2)], x_g2r,
            [fx.Float32(0.0) for _ in range_constexpr(32)],
            [fx.Int32(0) for _ in range_constexpr(8)], first=True)

        def save(state):
            b, c, x, xn, totals, y = state
            return b + c + x + xn + totals + y

        def restore(values):
            return ([fx.Vector(values[i]) for i in range_constexpr(6)],
                    [fx.Vector(values[i]) for i in range_constexpr(6, 10)],
                    [fx.Vector(values[i]) for i in range_constexpr(10, 12)],
                    [fx.Vector(values[i]) for i in range_constexpr(12, 14)],
                    [fx.Float32(values[i]) for i in range_constexpr(14, 46)],
                    [fx.Int32(values[i]) for i in range_constexpr(46, 54)])

        for g, values in range(fx.Index(1), fx.Index(GROUPS - 1), fx.Index(1), init=save(state)):
            state = run_group(fx.Int32(g), *restore(values))
            result = yield save(state)
        _, c_previous, x_previous, _, totals, y = run_group(fx.Int32(GROUPS - 1), *restore(result), last=True)
        if group == 0:
            _barrier()
        _priority(0)
        _mark('MEMORY_PRE_DRAIN_BEGIN')
        store_y_r2g(fx.Int32(PACKETS - 2), y)
        _mark('MEMORY_PRE_DRAIN_END')
        _priority(3)
        _mark('COMPUTE_DRAIN_BEGIN')
        for mi in range_constexpr(2):
            for h16 in range_constexpr(2):
                for m in range_constexpr(4):
                    post(mi, h16, m, 3, 1, c_previous, x_previous, totals)
        y = [pack_pair(mi, h16, pair, 1, totals) for mi in range_constexpr(2)
             for h16 in range_constexpr(2) for pair in range_constexpr(2)]
        _mark('COMPUTE_DRAIN_END')
        _priority(0)
        _mark('MEMORY_DRAIN_BEGIN')
        store_y_r2g(fx.Int32(PACKETS - 1), y)
        _mark('MEMORY_DRAIN_END')

    @flyc.jit
    def launch_up(X: fx.Tensor, W: fx.Tensor, P: fx.Tensor, Y: fx.Tensor, stream: fx.Stream):
        gr_read_up_m256_n16(X, W, P, Y).launch(grid=((rows + BM - 1) // BM, N_SPLITS, 1), block=(512, 1, 1), stream=stream)

    launch_up.compile_hints['llvm_options'] = {'vectorize-slp': False}
    return launch_up