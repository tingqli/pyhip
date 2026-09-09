"""Single-wave gfx942/gfx950 BF16 SWA: 64 threads, native 16x16 MFMA.

Each CTA owns 16 or 32 query rows and visits only their visible KV-tile union.
K/V are read directly from page64 SHUFFLE-5D caches into registers: no gather,
LDS allocation, cross-wave barrier, persistent queue, or multi-wave fallback.
gfx950 retains its K32 MFMA path; gfx942 splits each K32 into two native K16s.
"""

import functools
import math

import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import as_ir_value, gpu, rocdl
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm

try:
    from ._dsl import select, rmem, resource, wait
except ImportError:
    from _dsl import select, rmem, resource, wait


BM, DV, PAGE, THREADS = 16, 128, 64, 64
LOG2E = math.log2(math.e)


def _uniform(value):
    return fx.Int32(rocdl.readfirstlane(fx.Int32.ir_type, fx.Int32(value).ir_value()))


def _min(a, b):
    return (a < b).select(a, b)


def _max(a, b):
    return (a > b).select(a, b)


def _page(table, index):
    address = fx.Int64(fx.ptrtoint(fx.get_iter(table) + fx.Int64(index)))
    value = fx.Int32(llvm.inline_asm(fx.Int32.ir_type, [address.ir_value()],
                                    "s_load_dword $0, $1, 0", "=s,s,~{memory}", has_side_effects=True))
    return fx.Int32(llvm.inline_asm(fx.Int32.ir_type, [value.ir_value()],
                                   "s_waitcnt lgkmcnt(0)", "=s,0", has_side_effects=True))


def _exp(value):
    return fx.Float32(llvm.call_intrinsic(fx.Float32.ir_type, "llvm.amdgcn.exp2.f32",
                                         [fx.Float32(value).ir_value()], [], []))


def _maxf(a, b):
    return fx.Float32(a).maximumf(fx.Float32(b))


def _local_reduce(values, maximum=False):
    partials = [values[i] for i in range(values.numel)]
    while len(partials) > 1:
        partials = [(_maxf(partials[i], partials[i + 1]) if maximum else partials[i] + partials[i + 1])
                    for i in range(0, len(partials), 2)]
    return partials[0]


def _row_reduce(value, maximum=False, arch=950):
    if arch == 942:
        lane = fx.Int32(gpu.thread_id("x"))
        for offset in (16, 32):
            other = value.shuffle_xor(offset, 64)
            # permlane*_swap(value, value) returns the lower row first in
            # both lanes. Preserve that operand order as well as the tree.
            lower = (lane & offset) == 0
            lo, hi = lower.select(value, other), lower.select(other, value)
            value = _maxf(lo, hi) if maximum else lo + hi
        return value
    pair_type = ir.Type.parse("!llvm.struct<(i32, i32)>")
    for swap in (rocdl.permlane16_swap, rocdl.permlane32_swap):
        bits = value.bitcast(fx.Int32).ir_value()
        pair = swap(pair_type, bits, bits, False, True)
        lo = fx.Int32(llvm.extractvalue(fx.Int32.ir_type, pair, [0])).bitcast(fx.Float32)
        hi = fx.Int32(llvm.extractvalue(fx.Int32.ir_type, pair, [1])).bitcast(fx.Float32)
        value = _maxf(lo, hi) if maximum else lo + hi
    return value


def _atom_k(k, arch):
    return 16 if arch == 942 else k


def _mma(k, arch):
    # CDNA3's native BF16 K16 atom lowers to 16x16x16bf16_1k. A loaded
    # K32 fragment has k = 8 * (lane >> 4) + value, value in [0, 8).
    # Split its eight registers as (4, 2): the first K16 covers k%8 < 4,
    # the second k%8 >= 4. Q/K and P/V use the SAME K permutation, so their
    # dot products need no cross-lane repack. Explicit fragment shapes below
    # double the K-iteration count and preserve the original register order.
    return fx.make_tiled_mma(fx.make_mma_atom(rocdl.MFMA(16, 16, _atom_k(k, arch), fx.BFloat16)),
                             fx.make_layout((1, 1, 1), (1, 1, 1)))


def _load(resource, voffset, soffset, words=4):
    return fx.Vector(rocdl.raw_ptr_buffer_load(ir.VectorType.get([words], fx.Int32.ir_type), as_ir_value(resource),
                                              fx.Int32(voffset).ir_value(), fx.Int32(soffset).ir_value(), fx.Int32(0).ir_value()))


def _resource(tensor, size):
    return resource(tensor, size)


def _q_load(resource, lane, dq, qrow, arch, row_base=0):
    parts = [_load(resource, (((lane & 15) + row_base) * qrow + (lane >> 4) * 8 + k * 32) * 2, 0)
             for k in range(dq // 32)]
    words = fx.Vector.from_elements([p[i] for p in parts for i in range(4)], fx.Int32)
    frag = _mma(32, arch).make_fragment_B(rmem((BM, dq), fx.BFloat16))
    frag.store(words.bitcast(fx.BFloat16))
    return frag


def _k_load(resource, lane, tile, physical, dq, hk, bn):
    parts = []
    base = physical * (hk * PAGE * dq * 2) + (tile & 63) * 16
    for n in range(bn // 16):
        row = lane & 15 if bn == 16 else (lane & 3) + ((lane & 12) << 1) + (n & 1) * 4 + (n // 2) * 32
        for k in range(dq // 32):
            offset = (((lane >> 4) + k * 4) * PAGE + row) * 16
            parts.append(_load(resource, offset, base))
    words = fx.Vector.from_elements([p[i] for p in parts for i in range(4)], fx.Int32)
    return words


def _v_load(resource, lane, tile, physical, hk, bn):
    atom_k = 16 if bn == 16 else 32
    parts = []
    base = physical * (hk * PAGE * DV * 2) + (tile & 63) * (DV * 2)
    for n in range(DV // 16):
        for k in range(bn // atom_k):
            token = (lane >> 4) * (atom_k // 4) + k * atom_k
            offset = (token // 8 * DV + n * 16 + (lane & 15)) * 16 + (token & 7) * 2
            parts.append(_load(resource, offset, base, atom_k // 8))
    words = fx.Vector.from_elements([p[i] for p in parts for i in range(atom_k // 8)], fx.Int32)
    return words


def _qk_pair(q0, q1, words, dq, bn, arch):
    atom_k = _atom_k(32, arch)
    mma = _mma(32, arch)
    k = rmem((atom_k // 4, dq // atom_k, bn // 16), fx.BFloat16)
    k.store(fx.Vector(words).bitcast(fx.BFloat16))
    qs = rmem((atom_k // 4, dq // atom_k, 2), fx.BFloat16)
    a, b = q0.load(), q1.load()
    qs.store(fx.Vector.from_elements([a[i] for i in range(a.numel)] + [b[i] for i in range(b.numel)], fx.BFloat16))
    score = mma.make_fragment_C(rmem((bn, 32), fx.Float32))
    score.fill(0.0)
    fx.gemm(mma, score, select(k, [0, 2, 1]), select(qs, [0, 2, 1]), score, traversal_order="mnk")
    values = score.load()
    return (fx.Vector.from_elements([values[i] for i in range(bn // 4)], fx.Float32),
            fx.Vector.from_elements([values[i + bn // 4] for i in range(bn // 4)], fx.Float32))


@flyc.jit
def _softmax_tile(scores, output, maximum, total, scale, row, tile, kv_len, q_len,
                  WINDOW: fx.Constexpr[int], BN: fx.Constexpr[int], ARCH: fx.Constexpr[int]):
    lane = fx.Int32(gpu.thread_id("x"))
    values = fx.Vector(scores)
    diagonal = kv_len - q_len + row
    values = values * scale
    diagonal0 = _uniform(diagonal - (lane & 15))
    if (tile < diagonal0 + 15 - WINDOW) | (tile + BN > diagonal0 + 1):
        masked = []
        for i in fx.range_constexpr(BN // 4):
            col = tile + ((lane >> 4) * 4 + i if BN == 16 else (lane >> 4) * 8 + (i // 8) * 32 + i % 8)
            # Live query rows imply diagonal < kv_len; unsigned distance
            # combines causal and left-window bounds into one comparison.
            valid = fx.Uint32(diagonal - col) <= fx.Uint32(WINDOW)
            masked.append(valid.select(values[i], fx.Float32(float("-inf"))))
        values = fx.Vector.from_elements(masked, fx.Float32)
    candidate = _row_reduce(_local_reduce(values, True), True, ARCH)
    maximum, total, output = fx.Float32(maximum), fx.Float32(total), fx.Vector(output)
    advance = candidate - maximum > 8.0
    new_max = advance.select(_maxf(maximum, candidate), maximum)
    probs = fx.Vector.from_elements([_exp(values[i] - new_max) for i in range(BN // 4)], fx.Float32)
    ballot = fx.Int64(rocdl.ballot(fx.Int64.ir_type, advance.ir_value()))
    if ballot != fx.Int64(0):
        correction = _exp(maximum - new_max)
        total = total * correction
        output = output * correction
    total = total + _local_reduce(probs)
    return probs.to(fx.BFloat16), output, new_max, total


def _pv_pair(p0, p1, vwords, o0, o1, bn, arch):
    atom_k = _atom_k(16 if bn == 16 else 32, arch)
    mma = _mma(atom_k, arch)
    ps = rmem((atom_k // 4, bn // atom_k, 2), fx.BFloat16)
    ps.store(fx.Vector.from_elements([p0[i] for i in range(bn // 4)] + [p1[i] for i in range(bn // 4)], fx.BFloat16))
    vs = rmem((atom_k // 4, bn // atom_k, 8), fx.BFloat16)
    vs.store(fx.Vector(vwords).bitcast(fx.BFloat16))
    acc = mma.make_fragment_C(rmem((128, 32), fx.Float32))
    acc.store(fx.Vector.from_elements([o0[i] for i in range(32)] + [o1[i] for i in range(32)], fx.Float32))
    fx.gemm(mma, acc, select(vs, [0, 2, 1]), select(ps, [0, 2, 1]), acc, traversal_order="mnk")
    values = acc.load()
    return (fx.Vector.from_elements([values[i] for i in range(32)], fx.Float32),
            fx.Vector.from_elements([values[i + 32] for i in range(32)], fx.Float32))


@flyc.jit
def _mask_v(words, lane, tile, kv_len, BN: fx.Constexpr[int]):
    values = fx.Vector(words)
    if kv_len - tile < BN:
        atom_k = 16 if BN == 16 else 32
        per_lane = atom_k // 4
        masks = []
        for k in fx.range_constexpr(BN // atom_k):
            for word in fx.range_constexpr(per_lane // 2):
                token = tile + (lane >> 4) * per_lane + k * atom_k + word * 2
                mask = (token < kv_len).select(fx.Int32(0xFFFF), fx.Int32(0))
                mask = mask | (token + 1 < kv_len).select(fx.Int32(-65536), fx.Int32(0))
                masks.append(mask)
        values = fx.Vector.from_elements([values[i] & masks[i % len(masks)] for i in range(values.numel)], fx.Int32)
    return values


@flyc.jit
def _compute(q, kwords, vwords, output, maximum, total, scale, row, tile, kv_len, q_len,
             WINDOW: fx.Constexpr[int], BN: fx.Constexpr[int], DQ: fx.Constexpr[int], ARCH: fx.Constexpr[int]):
    lane = fx.Int32(gpu.thread_id("x"))
    qk_atom_k = _atom_k(32, ARCH)
    k_storage = rmem((qk_atom_k // 4, DQ // qk_atom_k, BN // 16), fx.BFloat16)
    k_storage.store(fx.Vector(kwords).bitcast(fx.BFloat16))
    k = select(k_storage, [0, 2, 1])
    scores = _mma(32, ARCH).make_fragment_C(rmem((BN, BM), fx.Float32))
    scores.fill(0.0)
    fx.gemm(_mma(32, ARCH), scores, k, q, scores, traversal_order="kmn")
    probs, output, new_max, total = _softmax_tile(scores.load(), output, maximum, total, scale,
                                                row, tile, kv_len, q_len, WINDOW, BN, ARCH)
    atom_k = _atom_k(16 if BN == 16 else 32, ARCH)
    pmma = _mma(atom_k, ARCH)
    p = pmma.make_fragment_B(rmem((BM, BN), fx.BFloat16))
    p.store(probs)
    if fx.const_expr(ARCH == 942):
        # Keep packed V words live across the probability conversion and
        # expose their SSA lifetime before assigning the PV operand view.
        words = fx.Vector(vwords)
        parts = []
        for begin in fx.range_constexpr(0, words.numel, 8):
            chunk = fx.Vector.from_elements([words[i] for i in range(begin, min(begin + 8, words.numel))], fx.Int32)
            pinned = fx.Vector(llvm.inline_asm(chunk.ir_value().type, [chunk.ir_value()], "", "=v,0", has_side_effects=True))
            parts.extend(pinned[i] for i in range(pinned.numel))
        vwords = fx.Vector.from_elements(parts, fx.Int32)
    v_storage = rmem((atom_k // 4, BN // atom_k, DV // 16), fx.BFloat16)
    v_storage.store(fx.Vector(vwords).bitcast(fx.BFloat16))
    v = select(v_storage, [0, 2, 1])
    acc = pmma.make_fragment_C(rmem((DV, BM), fx.Float32))
    acc.store(output)
    fx.gemm(pmma, acc, v, p, acc, traversal_order="mnk")
    return acc.load(), new_max, total


@flyc.kernel(known_block_size=[THREADS, 1, 1])
def _swa_kernel(Q: fx.Tensor, K: fx.Tensor, V: fx.Tensor, O: fx.Tensor, LSE: fx.Tensor,
    CQ: fx.Tensor, KI: fx.Tensor, PAGES: fx.Tensor, LAST: fx.Tensor,
    QS: fx.Tensor, KS: fx.Tensor, VS: fx.Tensor, SINK: fx.Tensor,
    H: fx.Constexpr[int], HK: fx.Constexpr[int], NP: fx.Constexpr[int], DQ: fx.Constexpr[int],
    QROW: fx.Constexpr[int], QHEAD: fx.Constexpr[int], OROW: fx.Constexpr[int], OHEAD: fx.Constexpr[int],
    WINDOW: fx.Constexpr[int], HAS_SINK: fx.Constexpr[bool], PER_TOKEN: fx.Constexpr[bool],
    WITH_LSE: fx.Constexpr[bool], SCALE: fx.Constexpr[float], BN: fx.Constexpr[int], ARCH: fx.Constexpr[int]):
    head, batch, qb = fx.Int32(gpu.block_id("x")), fx.Int32(gpu.block_id("y")), fx.Int32(gpu.block_id("z"))
    lane = fx.Int32(gpu.thread_id("x"))
    q0 = _uniform(CQ[batch])
    q_len = _uniform(CQ[batch + 1]) - q0
    start = _uniform(KI[batch])
    pages = _uniform(KI[batch + 1]) - start
    kv_len = (pages > 0).select((pages - 1) * PAGE + _uniform(LAST[batch]), fx.Int32(0))
    qstart = qb * BM
    if qstart < q_len:
        valid_q = _min(fx.Int32(BM), q_len - qstart)
        row = qstart + (lane & 15)
        hkv = head // (H // HK)
        table = fx.make_view(fx.get_iter(PAGES) + fx.Int64(start), fx.make_layout(pages, 1))
        gq = _resource(fx.make_view(fx.get_iter(Q) + ((fx.Int64(q0) + fx.Int64(qstart)) * QROW + fx.Int64(head) * QHEAD),
                                     fx.make_layout(BM * QROW, 1)), valid_q * QROW * 2)
        gk = _resource(fx.make_view(fx.get_iter(K) + (fx.Int64(hkv) * PAGE * DQ),
                                     fx.make_layout((NP * HK - hkv) * PAGE * DQ, 1)), (NP * HK - hkv) * PAGE * DQ * 2)
        gv = _resource(fx.make_view(fx.get_iter(V) + (fx.Int64(hkv) * PAGE * DV),
                                     fx.make_layout((NP * HK - hkv) * PAGE * DV, 1)), (NP * HK - hkv) * PAGE * DV * 2)
        q = _q_load(gq, lane, DQ, QROW, ARCH)
        scale = fx.Float32(KS[0]) * fx.Float32(SCALE * LOG2E)
        if fx.const_expr(PER_TOKEN):
            scale = scale * rocdl.make_buffer_tensor(QS, max_size=False)[(q0 + row) * H + head]
        else:
            scale = scale * QS[0]
        first = _max(qstart + kv_len - q_len - WINDOW, fx.Int32(0)) & -BN
        end = _max(_min(qstart + valid_q + kv_len - q_len, kv_len), fx.Int32(0))
        maximum, total = fx.Float32(-1.0e30), fx.Float32(0.0)
        output = fx.Vector.filled(DV // 4, 0.0, fx.Float32)
        if fx.const_expr(HAS_SINK):
            sink = fx.Float32(SINK[head]) * fx.Float32(LOG2E)
            maximum = _maxf(maximum, sink)
            total = _exp(sink - maximum) * fx.Float32(0.25)
        wait(vmcnt=0)
        if first < end:
            page = _page(table, first >> 6)
            k = _k_load(gk, lane, first, page, DQ, HK, BN)
            vw = _v_load(gv, lane, first, page, HK, BN)
            for tile in range(first, end, fx.Int32(BN)):
                wait(vmcnt=0)
                vw = _mask_v(vw, lane, tile, kv_len, BN)
                output, maximum, total = _compute(q, k, vw, output, maximum, total, scale,
                                                  row, tile, kv_len, q_len, WINDOW, BN, DQ, ARCH)
                if tile + BN < end:
                    page = _page(table, (tile + BN) >> 6)
                    k = _k_load(gk, lane, tile + BN, page, DQ, HK, BN)
                    vw = _v_load(gv, lane, tile + BN, page, HK, BN)
        outbuf = rocdl.make_buffer_tensor(fx.make_view(
            fx.get_iter(O) + ((fx.Int64(q0) + fx.Int64(qstart)) * OROW + fx.Int64(head) * OHEAD),
            fx.make_layout(BM * OROW, 1)), num_records_bytes=valid_q * OROW * 2)
        total = _row_reduce(total, arch=ARCH)
        inv = (total > 0.0).select(fx.Float32(1.0) / total, fx.Float32(0.0)) * VS[0]
        atom = fx.make_copy_atom(rocdl.BufferCopy64b(), fx.BFloat16)
        for n in fx.range_constexpr(DV // 16):
            values = fx.Vector.from_elements([output[n * 4 + j] * inv for j in range(4)], fx.Float32)
            src = rmem(4, fx.BFloat16)
            src.store(values.to(fx.BFloat16))
            offset = (lane & 15) * OROW + (lane >> 4) * 4 + n * 16
            fx.copy(atom, src, fx.make_view(fx.get_iter(outbuf) + offset, fx.make_layout(4, 1)))
        if fx.const_expr(WITH_LSE):
            if (lane < 16) & (lane < valid_q):
                log_l = fx.Float32(llvm.call_intrinsic(fx.Float32.ir_type, "llvm.log2.f32", [total.ir_value()], [], []))
                LSE[(q0 + row) * H + head] = (total > 0.0).select(
                    (maximum + log_l) * fx.Float32(math.log(2.0)), fx.Float32(float("-inf")))


@flyc.jit
def _launch(Q: fx.Tensor, K: fx.Tensor, V: fx.Tensor, O: fx.Tensor, LSE: fx.Tensor,
    CQ: fx.Tensor, KI: fx.Tensor, PAGES: fx.Tensor, LAST: fx.Tensor,
    QS: fx.Tensor, KS: fx.Tensor, VS: fx.Tensor, SINK: fx.Tensor,
    H: fx.Constexpr[int], HK: fx.Constexpr[int], NP: fx.Constexpr[int], B: fx.Constexpr[int], MAX_Q: fx.Constexpr[int],
    DQ: fx.Constexpr[int], QROW: fx.Constexpr[int], QHEAD: fx.Constexpr[int], OROW: fx.Constexpr[int], OHEAD: fx.Constexpr[int],
    WINDOW: fx.Constexpr[int], HAS_SINK: fx.Constexpr[bool], PER_TOKEN: fx.Constexpr[bool], WITH_LSE: fx.Constexpr[bool],
    SCALE: fx.Constexpr[float], BN: fx.Constexpr[int], QUERY_TILE: fx.Constexpr[int], ARCH: fx.Constexpr[int], stream: fx.Stream):
    if fx.const_expr(QUERY_TILE == 32):
        _swa32_kernel(Q, K, V, O, LSE, CQ, KI, PAGES, LAST, QS, KS, VS, SINK,
            H, HK, NP, DQ, QROW, QHEAD, OROW, OHEAD, WINDOW, HAS_SINK, PER_TOKEN, WITH_LSE, SCALE, BN, ARCH,
        ).launch(grid=(H, B, (MAX_Q + 31) // 32), block=(THREADS, 1, 1), stream=stream)
    else:
        _swa_kernel(Q, K, V, O, LSE, CQ, KI, PAGES, LAST, QS, KS, VS, SINK,
            H, HK, NP, DQ, QROW, QHEAD, OROW, OHEAD, WINDOW, HAS_SINK, PER_TOKEN, WITH_LSE, SCALE, BN, ARCH,
        ).launch(grid=(H, B, (MAX_Q + BM - 1) // BM), block=(THREADS, 1, 1), stream=stream)


@flyc.kernel(known_block_size=[THREADS, 1, 1])
def _swa32_kernel(Q: fx.Tensor, K: fx.Tensor, V: fx.Tensor, O: fx.Tensor, LSE: fx.Tensor,
    CQ: fx.Tensor, KI: fx.Tensor, PAGES: fx.Tensor, LAST: fx.Tensor,
    QS: fx.Tensor, KS: fx.Tensor, VS: fx.Tensor, SINK: fx.Tensor,
    H: fx.Constexpr[int], HK: fx.Constexpr[int], NP: fx.Constexpr[int], DQ: fx.Constexpr[int],
    QROW: fx.Constexpr[int], QHEAD: fx.Constexpr[int], OROW: fx.Constexpr[int], OHEAD: fx.Constexpr[int],
    WINDOW: fx.Constexpr[int], HAS_SINK: fx.Constexpr[bool], PER_TOKEN: fx.Constexpr[bool],
    WITH_LSE: fx.Constexpr[bool], SCALE: fx.Constexpr[float], BN: fx.Constexpr[int], ARCH: fx.Constexpr[int]):
    head, batch, qb = fx.Int32(gpu.block_id("x")), fx.Int32(gpu.block_id("y")), fx.Int32(gpu.block_id("z"))
    lane = fx.Int32(gpu.thread_id("x"))
    q0 = _uniform(CQ[batch])
    q_len = _uniform(CQ[batch + 1]) - q0
    start = _uniform(KI[batch])
    pages = _uniform(KI[batch + 1]) - start
    kv_len = (pages > 0).select((pages - 1) * PAGE + _uniform(LAST[batch]), fx.Int32(0))
    qstart = qb * 32
    if qstart < q_len:
        valid_q = _min(fx.Int32(32), q_len - qstart)
        row = qstart + (lane & 15)
        hkv = head // (H // HK)
        table = fx.make_view(fx.get_iter(PAGES) + fx.Int64(start), fx.make_layout(pages, 1))
        gq = _resource(fx.make_view(fx.get_iter(Q) + ((fx.Int64(q0) + fx.Int64(qstart)) * QROW + fx.Int64(head) * QHEAD),
                                     fx.make_layout(32 * QROW, 1)), valid_q * QROW * 2)
        gk = _resource(fx.make_view(fx.get_iter(K) + (fx.Int64(hkv) * PAGE * DQ),
                                     fx.make_layout((NP * HK - hkv) * PAGE * DQ, 1)), (NP * HK - hkv) * PAGE * DQ * 2)
        gv = _resource(fx.make_view(fx.get_iter(V) + (fx.Int64(hkv) * PAGE * DV),
                                     fx.make_layout((NP * HK - hkv) * PAGE * DV, 1)), (NP * HK - hkv) * PAGE * DV * 2)
        q_lo, q_hi = _q_load(gq, lane, DQ, QROW, ARCH), _q_load(gq, lane, DQ, QROW, ARCH, 16)
        scale0 = fx.Float32(KS[0]) * fx.Float32(SCALE * LOG2E)
        scale1 = scale0
        if fx.const_expr(PER_TOKEN):
            qs = rocdl.make_buffer_tensor(QS, max_size=False)
            scale0 = scale0 * qs[(q0 + row) * H + head]
            scale1 = scale1 * qs[(q0 + row + 16) * H + head]
        else:
            scale0, scale1 = scale0 * QS[0], scale1 * QS[0]
        first = _max(qstart + kv_len - q_len - WINDOW, fx.Int32(0)) & -BN
        end = _max(_min(qstart + valid_q + kv_len - q_len, kv_len), fx.Int32(0))
        maximum0, total0 = fx.Float32(-1.0e30), fx.Float32(0.0)
        if fx.const_expr(HAS_SINK):
            sink = fx.Float32(SINK[head]) * fx.Float32(LOG2E)
            maximum0 = _maxf(maximum0, sink)
            total0 = _exp(sink - maximum0) * fx.Float32(0.25)
        maximum1, total1 = maximum0, total0
        output0, output1 = fx.Vector.filled(32, 0.0, fx.Float32), fx.Vector.filled(32, 0.0, fx.Float32)
        wait(vmcnt=0)
        for tile in range(first, end, fx.Int32(BN)):
            page = _page(table, tile >> 6)
            kw = _k_load(gk, lane, tile, page, DQ, HK, BN)
            wait(vmcnt=0)
            rocdl.sched_barrier(0)
            vw = _v_load(gv, lane, tile, page, HK, BN)
            score0, score1 = _qk_pair(q_lo, q_hi, kw, DQ, BN, ARCH)
            p0, output0, maximum0, total0 = _softmax_tile(score0, output0, maximum0, total0, scale0,
                                                        row, tile, kv_len, q_len, WINDOW, BN, ARCH)
            p1, output1, maximum1, total1 = _softmax_tile(score1, output1, maximum1, total1, scale1,
                                                        row + 16, tile, kv_len, q_len, WINDOW, BN, ARCH)
            wait(vmcnt=0)
            rocdl.sched_barrier(0)
            vw = _mask_v(vw, lane, tile, kv_len, BN)
            output0, output1 = _pv_pair(p0, p1, vw, output0, output1, BN, ARCH)
        outbuf = rocdl.make_buffer_tensor(fx.make_view(fx.get_iter(O) + ((fx.Int64(q0) + fx.Int64(qstart)) * OROW + fx.Int64(head) * OHEAD),
                                           fx.make_layout(32 * OROW, 1)), num_records_bytes=valid_q * OROW * 2)
        atom = fx.make_copy_atom(rocdl.BufferCopy64b(), fx.BFloat16)
        for m in fx.range_constexpr(2):
            values = output0 if m == 0 else output1
            total = _row_reduce(total0 if m == 0 else total1, arch=ARCH)
            maximum = maximum0 if m == 0 else maximum1
            inv = (total > 0.0).select(fx.Float32(1.0) / total, fx.Float32(0.0)) * VS[0]
            for n in fx.range_constexpr(8):
                src = rmem(4, fx.BFloat16)
                src.store(fx.Vector.from_elements([values[n * 4 + j] * inv for j in range(4)], fx.Float32).to(fx.BFloat16))
                offset = ((lane & 15) + m * 16) * OROW + (lane >> 4) * 4 + n * 16
                fx.copy(atom, src, fx.make_view(fx.get_iter(outbuf) + offset, fx.make_layout(4, 1)))
            if fx.const_expr(WITH_LSE):
                if (lane < 16) & (lane + m * 16 < valid_q):
                    log_l = fx.Float32(llvm.call_intrinsic(fx.Float32.ir_type, "llvm.log2.f32", [total.ir_value()], [], []))
                    LSE[(q0 + row + m * 16) * H + head] = (total > 0.0).select(
                        (maximum + log_l) * fx.Float32(math.log(2.0)), fx.Float32(float("-inf")))


def _flat(tensor):
    if tensor.numel() == 0:
        return tensor.reshape(-1)
    extent = 1 + sum((n - 1) * s for n, s in zip(tensor.shape, tensor.stride()))
    return tensor.as_strided((extent,), (1,))


class _SWA:
    def __init__(self, heads, kv_heads, dq, window, sink, block_n, query_tile):
        self.heads, self.kv_heads, self.dq = heads, kv_heads, dq
        self.window, self.has_sink = window, sink
        self.block_n, self.query_tile = block_n, query_tile
        self._compiled = {}

    def __call__(self, Q, K, V, cu_seqlens_q, cu_seqlens_k, kv_indptr, kv_page_indices,
                 max_seqlen_q, max_seqlen_k, causal, q_descale, k_descale, v_descale,
                 kv_last_page_lens, out=None, sink_ptr=None, stream=None, *, return_lse=False, lse=None, softmax_scale=None):
        if not Q.is_cuda:
            raise NotImplementedError("single-wave SWA requires gfx942 or gfx950")
        arch = torch.cuda.get_device_properties(Q.device).gcnArchName.split(":", 1)[0]
        if arch not in ("gfx942", "gfx950"):
            raise NotImplementedError("single-wave SWA requires gfx942 or gfx950")
        if Q.dtype != torch.bfloat16 or K.dtype != Q.dtype or V.dtype != Q.dtype:
            raise NotImplementedError("BF16 Q/K/V only")
        if not causal:
            raise ValueError("SWA requires bottom-right causal attention")
        if Q.ndim != 3 or Q.shape[1:] != (self.heads, self.dq) or Q.stride(-1) != 1:
            raise ValueError("Q must be [tokens, heads, Dqk] with contiguous head dimension")
        if K.ndim != 5 or V.ndim != 5 or K.shape != (V.shape[0], self.kv_heads, self.dq // 8, PAGE, 8) or V.shape[1:] != (self.kv_heads, 8, DV, 8):
            raise ValueError("K/V require page64 SHUFFLE-5D layout")
        if K.device != Q.device or V.device != Q.device or not K.is_contiguous() or not V.is_contiguous():
            raise ValueError("K/V must be contiguous on the input GPU")
        if max(K.numel(), V.numel()) * 2 >= 2**31:
            raise NotImplementedError("physical KV cache must fit signed int32 byte offsets")
        if self.has_sink:
            if sink_ptr is None or sink_ptr.shape != (self.heads,) or sink_ptr.dtype != torch.float32 or sink_ptr.device != Q.device or not sink_ptr.is_contiguous():
                raise ValueError("sink must be contiguous FP32 [heads] on the input GPU")
        elif sink_ptr is not None:
            raise ValueError("sink_ptr requires has_sink=True")
        batch = cu_seqlens_q.numel() - 1
        if batch < 1 or kv_indptr.numel() != batch + 1 or kv_last_page_lens.numel() != batch:
            raise ValueError("inconsistent batch metadata")
        metadata = (cu_seqlens_q, kv_indptr, kv_page_indices, kv_last_page_lens)
        if cu_seqlens_k is not None:
            if cu_seqlens_k.numel() != batch + 1:
                raise ValueError("inconsistent KV prefix length")
            metadata += (cu_seqlens_k,)
        if any(t.ndim != 1 or t.dtype != torch.int32 or t.device != Q.device or not t.is_contiguous() for t in metadata):
            raise ValueError("metadata must be contiguous device int32")
        if min(max_seqlen_q, max_seqlen_k) < 0:
            raise ValueError("maximum lengths must be nonnegative")
        if any(t.dtype != torch.float32 or t.device != Q.device or not t.is_contiguous() for t in (q_descale, k_descale, v_descale)):
            raise ValueError("descales must be contiguous device FP32")
        if k_descale.numel() != 1 or v_descale.numel() != 1 or q_descale.numel() not in (1, Q.shape[0] * self.heads):
            raise ValueError("expected scalar K/V and scalar or per-token/head Q descales")
        scale = self.dq**-0.5 if softmax_scale is None else float(softmax_scale)
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("softmax_scale must be finite and positive")
        stream = torch.cuda.current_stream(Q.device) if stream is None else stream
        if stream.device != Q.device:
            raise ValueError("stream must belong to the input GPU")
        with torch.cuda.stream(stream):
            if out is None:
                out = torch.empty(Q.shape[0], self.heads, DV, device=Q.device, dtype=torch.bfloat16)
            if return_lse and lse is None:
                lse = torch.empty(Q.shape[0], self.heads, device=Q.device, dtype=torch.float32)
        if out.shape != (Q.shape[0], self.heads, DV) or out.dtype != torch.bfloat16 or out.device != Q.device or out.stride(-1) != 1:
            raise ValueError("invalid output buffer")
        if lse is not None and (lse.shape != Q.shape[:2] or lse.dtype != torch.float32 or lse.device != Q.device or not lse.is_contiguous()):
            raise ValueError("LSE must be contiguous FP32 [tokens, heads]")
        if Q.shape[0] and max_seqlen_q:
            args = (_flat(Q), _flat(K), _flat(V), _flat(out), _flat(lse) if lse is not None else k_descale,
                    cu_seqlens_q, kv_indptr, kv_page_indices, kv_last_page_lens,
                    _flat(q_descale), k_descale.view(-1), v_descale.view(-1), sink_ptr if self.has_sink else k_descale,
                    self.heads, self.kv_heads, K.shape[0], batch, max_seqlen_q, self.dq,
                    Q.stride(0), Q.stride(1), out.stride(0), out.stride(1), self.window, self.has_sink,
                        q_descale.numel() != 1, lse is not None, scale, self.block_n, self.query_tile, int(arch[3:]), stream)
            signature = tuple((a.dtype, tuple(a.shape)) if isinstance(a, torch.Tensor) else ("stream",) if hasattr(a, "cuda_stream") else a for a in args)
            key = (Q.device, signature)
            compiled = self._compiled.get(key)
            with torch.cuda.device(Q.device):
                if compiled is None:
                    self._compiled[key] = flyc.compile(_launch, *args)
                else:
                    compiled(*args)
        return (out, lse) if return_lse else out


@functools.cache
def PagedAttention(num_qo_heads, num_kv_heads, head_dim_qk, head_dim_v, page_size,
                   is_causal=True, quant_query_mode="per-token", key_layout="vectorized",
                   window_left=128, has_sink=False, *, block_n=None, query_tile=None):
    """Return a native gfx942/gfx950 direct-paged, single-wave SWA callable.

    The default is BM16/BN16 for window_left<=16, BM32/BN32 otherwise.
    Explicit query_tile={16,32} and block_n={16,32,64} allow measured tuning.
    Caller-owned device metadata must describe valid pages/lengths, with actual
    query lengths bounded by max_seqlen_q. Sink logits are unscaled natural
    logits (finite or -inf), and contribute no value to the numerator.
    """
    if head_dim_qk not in (128, 192) or (head_dim_v, page_size, key_layout) != (DV, PAGE, "vectorized"):
        raise NotImplementedError("SWA supports BF16 D128/D192 V128 page64 SHUFFLE-5D only")
    if not is_causal or not isinstance(window_left, int) or not 0 <= window_left < 2**31:
        raise ValueError("SWA requires causal=True and a nonnegative signed-int32 window")
    if num_qo_heads <= 0 or num_kv_heads <= 0 or num_qo_heads % num_kv_heads:
        raise ValueError("query heads must be a positive multiple of KV heads")
    block_n = (16 if window_left <= 16 else 32) if block_n is None else block_n
    query_tile = (16 if window_left <= 16 else 32) if query_tile is None else query_tile
    if quant_query_mode not in ("per-token", "per-tensor") or block_n not in (16, 32, 64):
        raise ValueError("invalid scale mode or block_n")
    if query_tile not in (16, 32):
        raise ValueError("query_tile must be 16 or 32")
    return _SWA(num_qo_heads, num_kv_heads, head_dim_qk, window_left, has_sink, block_n, query_tile)