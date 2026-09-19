# SPDX-License-Identifier: MIT
"""BF16 GR read Up：M128/H32/K320、8 waves、两个N分片。

P为BF16；两个H16依次完成完整K320，原始FP32 logits直接做sigmoid/FMA/mean。
Y调用cvt_f32_to_bf16的整数位加偏置打包，不使用浮点FMAAK舍入。
W静态换列使每lane的8个H连续，X/Y均直接128-bit搬运，不需要动态预排或DPP。
4+4 wave错相，两组都执行Memory/Compute；Memory包括真实priority窗口必须零VALU。
g=VMEM，r=寄存器，s=LDS。B使用两个20KiB槽；每8包组成H64的四stream归约组。
"""

from functools import cache

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, range_constexpr, rocdl
from pyhip.contrib.flydsl.helpers import cvt_f32_to_bf16

if __package__:
    from .kernel import C, H, K, R
else:
    from kernel import C, H, K, R

BM, WAVES, N_SPLITS = 128, 8, 2
B_SLOT_BYTES = 32 * R * 2
PACKETS = K // 32 // N_SPLITS
GROUP_STEPS = 8
GROUPS = PACKETS // GROUP_STEPS
# 每包3次B读取、1次X读取；每H64组的前两包各写一次旧Y。
# 等待保护下一B q+1、上一包X；末包同时等待当前X供drain。
VMEM_WAITS = (
    (4, 4, 4, 4, 4, 4, 4, 4),
    (5, 6, 5, 4, 4, 4, 4, 4),
    (5, 6, 5, 4, 4, 4, 1, 0),
)


def _pin_v(value):
    return fx.Int32(llvm.inline_asm(fx.Int32.ir_type, [value.ir_value()], "", "=v,0", has_side_effects=True))


def _mark(text):
    rocdl.sched_barrier(0)
    llvm.inline_asm(ir.Type.parse("!llvm.void"), [], f"; GR_{text}", "", has_side_effects=True)
    rocdl.sched_barrier(0)


def _priority(value):
    rocdl.sched_barrier(0)
    rocdl.s_setprio(value)
    rocdl.sched_barrier(0)


def _barrier():
    rocdl.sched_barrier(0)
    rocdl.s_barrier()
    rocdl.sched_barrier(0)


def _load(resource, address, scalar_offset=0, words=4):
    value = rocdl.RawPtrBufferLoadOp(
        fx.Int32.ir_type if words == 1 else ir.VectorType.get([words], fx.Int32.ir_type), resource,
        address.ir_value(), fx.Int32(scalar_offset).ir_value(),
        aux=ir.IntegerAttr.get(fx.Int32.ir_type, 0)).result
    return fx.Int32(value) if words == 1 else fx.Vector(value)


def _store(resource, address, values, scalar_offset=0):
    rocdl.RawPtrBufferStoreOp(values.ir_value(), resource, address.ir_value(),
        fx.Int32(scalar_offset).ir_value(), aux=ir.IntegerAttr.get(fx.Int32.ir_type, 0))


def _ds_read(address, offset):
    value = llvm.inline_asm(ir.VectorType.get([4], fx.Int32.ir_type),
        [address.ir_value()], f"ds_read_b128 $0, $1 offset:{offset}",
        "=v,v,~{memory}", has_side_effects=True)
    return fx.Vector(value)


def _ds_write(address, values, offset):
    llvm.inline_asm(ir.Type.parse("!llvm.void"), [address.ir_value(), values.ir_value()],
        f"ds_write_b128 $0, $1 offset:{offset}", "v,v,~{memory}", has_side_effects=True)


def _mfma(a, b, c):
    return fx.Vector(rocdl.mfma_f32_16x16x16bf16_1k(
        ir.VectorType.get([4], fx.Float32.ir_type),
        [a.bitcast(fx.Int16).ir_value(), b.bitcast(fx.Int16).ir_value(), c.ir_value(), 0, 0, 0]))


def _sigmoid(value):
    exponent = fx.Float32(rocdl.exp2(fx.Float32.ir_type, (-value * 1.4426950408889634).ir_value()))
    return fx.Float32(rocdl.rcp(fx.Float32.ir_type, (1.0 + exponent).ir_value()))


def _pack_y_mean(v0, v1):
    """四路均值后用原helper整数舍入；两个BF16留在一个DWORD。"""
    values = fx.Vector.from_elements([v0 * 0.25, v1 * 0.25], fx.Float32)
    fragment = fx.make_rmem_tensor(fx.make_layout(2, 1), fx.Float32)
    fragment.store(values)
    return fx.Vector(cvt_f32_to_bf16(fragment).load()).bitcast(fx.Int32)[0]


@cache
def make_up_8x1(rows, padded_rows, *, block_m=128):
    """M128完整H32拍或M256两个H16子阶段，共用FP32 logits与Y helper。"""
    if block_m == 256:
        if __package__:
            from .prefil_up_m256 import make_up_m256
        else:
            from prefil_up_m256 import make_up_m256
        return make_up_m256(rows, padded_rows)
    if block_m != 128:
        raise ValueError("Up block_m must be 128 or 256")
    assert 0 < rows <= padded_rows and padded_rows % BM == 0

    @fx.struct
    class Shared:
        b: fx.Array[fx.Int32, 2 * B_SLOT_BYTES // 4, 16]

    @flyc.kernel(known_block_size=[512, 1, 1])
    def gr_read_up_fullk_n16(X: fx.Tensor, W: fx.Tensor, P: fx.Tensor, Y: fx.Tensor):
        tid = fx.Int32(fx.thread_idx.x)
        lane, wave = tid % 64, tid // 64
        group = fx.Int32(rocdl.readfirstlane(fx.Int32.ir_type, (tid // 256).ir_value()))
        m_tiles = (rows + BM - 1) // BM
        worker = fx.Int32(fx.block_idx.x) + fx.Int32(fx.block_idx.y) * m_tiles
        # width2有效前缀转置；保留已计时版本的任务解码顺序。
        chunk = m_tiles * N_SPLITS // 2
        task = (worker < chunk * 2).select(worker % 2 * chunk + worker // 2, worker)
        first_m = task // N_SPLITS
        width = (m_tiles - first_m < 1).select(m_tiles - first_m, fx.Int32(1))
        within = task % N_SPLITS
        block_m, n_base = first_m + within % width, within // width * PACKETS
        n_phase = block_m % 2 * (PACKETS // 2)
        row = block_m * BM + wave * 16 + lane % 16
        shared = fx.SharedAllocator().allocate(Shared).peek()
        b_base = fx.Int32(fx.ptrtoint(shared.b.ptr))
        buffers = [fx.rocdl.make_buffer_tensor(t, max_size=False) for t in (W, P, X, Y)]
        wr, pr, xr, yr = [fx.rocdl.get_buffer_rsrc(fx.get_iter(t)) for t in buffers]
        p_address = _pin_v((row < rows).select(row * R * 2 + lane // 16 * 16, fx.Int32(0x7fffffff)))
        x_address = _pin_v((row < rows).select(row * K * 2 + lane // 16 * 16, fx.Int32(0x7fffffff)))
        y_address = _pin_v((row < rows).select(row * H * 2 + lane // 16 * 16, fx.Int32(0x7fffffff)))
        w16, w8 = _pin_v(tid * 16), _pin_v(tid * 8)
        b_read = _pin_v(b_base + lane % 16 * 16 + lane // 16 * 256)
        b_write16, b_write8 = _pin_v(b_base + tid * 16), _pin_v(b_base + tid * 8)
        # 每wave16行P，各lane40DWORD；20个K16片段全程复用。
        p_bf16 = [_load(pr, p_address, kb * 128 + k32 * 64 + k4 * 8, words=2).bitcast(fx.BFloat16)
                  for kb in range_constexpr(5) for k32 in range_constexpr(2) for k4 in range_constexpr(2)]

        def packet(q):
            shifted = q + n_phase
            return n_base + (shifted >= PACKETS).select(shifted - PACKETS, shifted)

        def read_b_g2r(q):
            parts = [_load(wr, w16, packet(q) * B_SLOT_BYTES + i * 8192) for i in range_constexpr(2)]
            tail = _load(wr, w8, packet(q) * B_SLOT_BYTES + 16384, words=2)
            return fx.Vector.from_elements([v[i] for v in parts for i in range_constexpr(4)]
                                           + [tail[i] for i in range_constexpr(2)], fx.Int32)

        def store_b_r2s(q, b):
            for part in range_constexpr(2):
                _ds_write(b_write16, b.shuffle(b, list(range(part * 4, part * 4 + 4))),
                          q % 2 * B_SLOT_BYTES + part * 8192)
            tail = b.shuffle(b, [8, 9])
            llvm.inline_asm(ir.Type.parse('!llvm.void'), [b_write8.ir_value(), tail.ir_value()],
                f'ds_write_b64 $0,$1 offset:{16384 + q % 2 * B_SLOT_BYTES}', 'v,v,~{memory}', has_side_effects=True)

        def store_y_r2g(q, y):
            _store(yr, y_address, fx.Vector.from_elements(y, fx.Int32), packet(q) // 8 * 128 + packet(q) % 2 * 64)

        def post(m, stream, h_sub, c, x, totals):
            value = (x[h_sub % 2 * 2 + m // 2].bitcast(fx.Uint32) >> (m % 2 * 16)).to(fx.Uint16).bitcast(fx.BFloat16).to(fx.Float32)
            gate = _sigmoid(c[h_sub % 2][m])
            if const_expr(stream == 0):
                totals[h_sub * 4 + m] = fx.Float32(0.0)
            totals[h_sub * 4 + m] = fx.Float32(llvm.call_intrinsic(fx.Float32.ir_type, 'llvm.fma.f32',
                [gate.ir_value(), value.ir_value(), totals[h_sub * 4 + m].ir_value()], [], []))

        def pack(totals, h_sub):
            y = []
            for part in range_constexpr(2):
                index = h_sub * 4 + part * 2
                y.append(_pin_v(_pack_y_mean(totals[index], totals[index + 1])))
            return y

        def run_group(g, b_g2r, c_previous, x_previous, totals, y, *, first=False, last=False):
            for step in range_constexpr(GROUP_STEPS):
                q = g * GROUP_STEPS + step
                _priority(0)
                _mark(f"MEMORY_{'FIRST' if first else 'LAST' if last else 'LOOP'}_{step}_BEGIN")
                b_s2r = [[], []]
                for block in range_constexpr(4):
                    for part in range_constexpr(5):
                        h16, k32 = (block * 5 + part) // 10, (block * 5 + part) % 10
                        b_s2r[h16].append(_ds_read(b_read, step % 2 * B_SLOT_BYTES + h16 * 10240 + k32 * 1024))
                    if const_expr(block == 1):
                        x_g2r = _load(xr, x_address, packet(q) // 8 * 128 + step // 2 * H * 2 + step % 2 * 64)
                if const_expr(step < 2 and not first):
                    store_y_r2g(q - 2, y)
                rocdl.s_waitcnt(vmcnt=VMEM_WAITS[0 if first else 2 if last else 1][step])
                if const_expr(not last or step + 1 < GROUP_STEPS):
                    store_b_r2s(step + 1, b_g2r[step % 2])
                if const_expr(not last or step + 3 < GROUP_STEPS):
                    b_g2r[step % 2] = read_b_g2r(q + 3)
                # 当前B所有wave读完后才允许后续覆写；不能省略opaque DS的显式等待。
                rocdl.s_waitcnt(lgkmcnt=0)
                _mark(f"MEMORY_{'FIRST' if first else 'LAST' if last else 'LOOP'}_{step}_END")
                _barrier()
                _priority(3)
                _mark(f'COMPUTE_{step}_BEGIN')
                if const_expr((step == 0 and not first) or step == 7):
                    y = []
                c_f32 = [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range_constexpr(2)]
                for ordinal in range_constexpr(40):
                    k16, h16 = ordinal % 20, ordinal // 20
                    b = b_s2r[h16][k16 // 2].bitcast(fx.BFloat16)
                    b = b.shuffle(b, list(range(k16 % 2 * 4, k16 % 2 * 4 + 4)))
                    c_f32[h16] = _mfma(b, p_bf16[k16], c_f32[h16])
                    if const_expr(not first or step > 0):
                        if const_expr(ordinal % 5 == 2):
                            m, sub = ordinal // 5 % 4, ordinal // 20
                            post(m, (step - 1) % 8 // 2, (step - 1) % 2 * 2 + sub, c_previous, x_previous, totals)
                            if const_expr(last):
                                # 防止最后H64的部分stream被优化器下沉到priority0的drain前。
                                index = (step - 1) % 2 * 8 + sub * 4 + m
                                totals[index] = _pin_v(totals[index].bitcast(fx.Int32)).bitcast(fx.Float32)
                            rocdl.sched_barrier(0)
                    if const_expr(((step == 0 and not first) or step == 7) and ordinal in (7, 17, 27, 37)):
                        part = (ordinal - 7) // 10
                        sub = (step - 1) % 2 * 2 + part // 2
                        index = sub * 4 + part % 2 * 2
                        y.append(_pin_v(_pack_y_mean(totals[index], totals[index + 1])))
                c_previous, x_previous = c_f32, x_g2r
                _mark(f'COMPUTE_{step}_END')
                _priority(0)
                _barrier()
            return b_g2r, c_previous, x_previous, totals, y

        b0 = read_b_g2r(fx.Int32(0))
        rocdl.s_waitcnt(vmcnt=0)
        store_b_r2s(0, b0)
        rocdl.s_waitcnt(lgkmcnt=0)
        _barrier()
        b_g2r = [read_b_g2r(fx.Int32(q)) for q in range_constexpr(1, 3)]
        if group == 1:
            _barrier()
        state = run_group(fx.Int32(0), b_g2r, [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range_constexpr(2)],
                          fx.Vector.filled(4, 0, fx.Int32), [fx.Float32(0.0) for _ in range_constexpr(16)],
                          [fx.Int32(0) for _ in range_constexpr(4)], first=True)

        def save(state):
            b, c, x, totals, y = state
            return b + c + [x] + totals + y

        def restore(values):
            return ([fx.Vector(values[i]) for i in range_constexpr(2)],
                    [fx.Vector(values[i]) for i in range_constexpr(2, 4)], fx.Vector(values[4]),
                    [fx.Float32(values[i]) for i in range_constexpr(5, 21)],
                    [fx.Int32(values[i]) for i in range_constexpr(21, 25)])

        for g, values in range(fx.Index(1), fx.Index(GROUPS - 1), fx.Index(1), init=save(state)):
            state = run_group(fx.Int32(g), *restore(values))
            result = yield save(state)
        _, c_previous, x_previous, totals, y = run_group(fx.Int32(GROUPS - 1), *restore(result), last=True)
        if group == 0:
            _barrier()
        _priority(0)
        _mark('MEMORY_PRE_DRAIN_BEGIN')
        store_y_r2g(fx.Int32(PACKETS - 2), y)
        _mark('MEMORY_PRE_DRAIN_END')
        _priority(3)
        _mark('COMPUTE_DRAIN_BEGIN')
        for sub in range_constexpr(2):
            for m in range_constexpr(4):
                post(m, 3, 2 + sub, c_previous, x_previous, totals)
        y = pack(totals, 2) + pack(totals, 3)
        _mark('COMPUTE_DRAIN_END')
        _priority(0)
        _mark('MEMORY_DRAIN_BEGIN')
        store_y_r2g(fx.Int32(PACKETS - 1), y)
        _mark('MEMORY_DRAIN_END')

    @flyc.jit
    def launch_up(X: fx.Tensor, W: fx.Tensor, P: fx.Tensor, Y: fx.Tensor, stream: fx.Stream):
        gr_read_up_fullk_n16(X, W, P, Y).launch(grid=((rows + BM - 1) // BM, N_SPLITS, 1), block=(512, 1, 1), stream=stream)

    launch_up.compile_hints['llvm_options'] = {'vectorize-slp': False}
    return launch_up