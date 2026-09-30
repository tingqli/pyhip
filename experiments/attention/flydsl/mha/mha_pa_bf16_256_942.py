"""gfx942 DQ=DV=256 specialization of the BF16 paged-attention backend.

BM128/BN64, eight staggered waves, native 16x16x16 BF16 MFMA. One
32-KiB K slot and one 32-KiB V slot fit in 64 KiB LDS. Each wave owns
16 query rows, so the complete FP32 output occupies 64 registers/lane.
The public wrapper, SHUFFLE-5D cache ABI and numerical contract are shared
with mha_pa_bf16_942; this module contains only the wide device pipeline.
"""

import math

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu, rocdl
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm

if __package__:
    from .mha_pa_bf16_942 import (
        _uniform, _min, _pin_i32, _pin_s64, _pin, _join, _exp, _maximum,
        _pack_bf16, _stage_end, _wait, _schedule, _prefetch_page,
        _page_ready, _buffer, _buffer_words, _read_address, _rescale,
        _advance_max,
    )
else:
    from mha_pa_bf16_942 import (
        _uniform, _min, _pin_i32, _pin_s64, _pin, _join, _exp, _maximum,
        _pack_bf16, _stage_end, _wait, _schedule, _prefetch_page,
        _page_ready, _buffer, _buffer_words, _read_address, _rescale,
        _advance_max,
    )


BM, BN, D, THREADS = 128, 64, 256, 512
K_PITCH, K_BYTES, V_BYTES = 1024, 32768, 32768
LDS_BYTES = K_BYTES + V_BYTES
LOG2E = math.log2(math.e)


def _mma():
    return fx.make_tiled_mma(fx.make_mma_atom(rocdl.MFMA(16, 16, 16, fx.BFloat16)),
                             fx.make_layout((1, 8, 1), (1, 1, 0)))


def _q_fragment(resource, offset, lane):
    parts = [_buffer_words(resource, offset + (lane >> 4) * 16 + k * 64) for k in range(8)]
    words = fx.Vector.from_elements([p[i] for p in parts for i in range(4)], fx.Int32)
    frag = _mma().make_fragment_B(fx.make_rmem_tensor(fx.make_layout((BM, D), (1, BM)), fx.BFloat16))
    frag.store(words.bitcast(fx.BFloat16))
    return frag


def _load_k(resource, lane_offset, page, tile, hk, page_size, second_page):
    offset = page * (hk * page_size * D * 2) + (tile * BN % page_size) * 16
    if page_size == 32:
        lane = fx.Int32(gpu.thread_id("x")) & 63
        address = _pin_i32(lane_offset + (lane < 32).select(page, second_page) * (hk * page_size * D * 2))
        parts = [_buffer_words(resource, address, i * page_size * 128) for i in range(4)]
    else:
        parts = [_buffer_words(resource, lane_offset, offset + i * page_size * 128) for i in range(4)]
    return fx.Vector.from_elements([p[j] for p in parts for j in range(4)], fx.Int32)


def _load_v(base, lane_offset, page, tile, hk, page_size, second_page):
    address = _pin_s64(base + fx.Int64(page) * (hk * page_size * D * 2)
                      + fx.Int64(tile * BN % page_size) * (D * 2))
    second = _pin_s64(base + fx.Int64(second_page) * (hk * page_size * D * 2)) if page_size == 32 else address + 16384
    parts = [fx.Vector(llvm.inline_asm(ir.VectorType.get([4], fx.Int32.ir_type),
        [lane_offset.ir_value(), _pin_s64((address if i < 2 else second) + (i % 2) * 8192).ir_value()],
        "global_load_dwordx4 $0, $1, $2", "=v,v,s,~{memory}", has_side_effects=True)) for i in range(4)]
    return fx.Vector.from_elements([p[j] for p in parts for j in range(4)], fx.Int32)


def _write(address, words, stride):
    for i in range(4):
        part = fx.Vector.from_elements([words[i * 4 + j] for j in range(4)], fx.Int32)
        llvm.inline_asm(ir.Type.parse("!llvm.void"), [address.ir_value(), part.ir_value()],
                        f"ds_write_b128 $0, $1 offset:{i * stride}", "v,v,~{memory}", has_side_effects=True)


def _read_k(address, half, publish_address=None, pending=None):
    # Both operands use the same K32 permutation: lane-group*8 + value.
    # Native K16 consumes the lower/upper four values in two MFMA steps.
    parts = []
    for n in range(2):
        for k in range(8):
            parts.append(_read_address(address[n], half * 512 + k * (4 * K_PITCH)))
            if publish_address is not None and k % 4 == 3:
                packet = n * 2 + k // 4
                words = fx.Vector.from_elements([pending[packet * 4 + j] for j in range(4)], fx.Int32)
                llvm.inline_asm(ir.Type.parse("!llvm.void"), [publish_address.ir_value(), words.ir_value()],
                    f"ds_write_b128 $0, $1 offset:{packet * 8192}", "v,v,~{memory}", has_side_effects=True)
    words = fx.Vector.from_elements([p[i] for p in parts for i in range(4)], fx.Int32)
    storage = fx.make_rmem_tensor(fx.make_layout((4, 16, 2), (1, 4, 64)), fx.BFloat16)
    storage.store(words.bitcast(fx.BFloat16))
    return fx.make_view(fx.get_iter(storage), fx.make_layout((4, 2, 16), (1, 64, 4)))


def _read_v(address, half, publish_address=None, pending=None):
    parts = []
    for n in range(8):
        for k in range(2):
            parts.append(_read_address(address, half * 2048 + n * 256 + k * 16384))
        if publish_address is not None and n % 2 == 1:
            packet = n // 2
            words = fx.Vector.from_elements([pending[packet * 4 + j] for j in range(4)], fx.Int32)
            llvm.inline_asm(ir.Type.parse("!llvm.void"), [publish_address.ir_value(), words.ir_value()],
                f"ds_write_b128 $0, $1 offset:{packet * 8192}", "v,v,~{memory}", has_side_effects=True)
    words = fx.Vector.from_elements([p[i] for p in parts for i in range(4)], fx.Int32)
    frag = fx.make_rmem_tensor(fx.make_layout((4, 4, 8), (1, 4, 16)), fx.BFloat16)
    frag.store(words.bitcast(fx.BFloat16))
    return frag


def _qk(q, k):
    mma = _mma()
    acc = mma.make_fragment_C(fx.make_rmem_tensor(fx.make_layout((32, BM), (1, 32)), fx.Float32))
    acc.fill(0.0)
    fx.gemm(mma, acc, k, q, acc, traversal_order="kmn")
    return acc.load()


def _pv(probabilities, values, output):
    mma = _mma()
    p = mma.make_fragment_B(fx.make_rmem_tensor(fx.make_layout((BM, BN), (1, BM)), fx.BFloat16))
    p.store(probabilities)
    acc = mma.make_fragment_C(fx.make_rmem_tensor(fx.make_layout((128, BM), (1, 128)), fx.Float32))
    acc.store(output)
    operand = fx.make_view(fx.get_iter(values), fx.make_layout((4, 8, 4), (1, 16, 4)))
    fx.gemm(mma, acc, operand, p, acc, traversal_order="mnk")
    return acc.load()


def _sum(values):
    partials = [values[i] + values[i + 1] for i in range(0, 16, 2)]
    for width in (4, 2, 1):
        partials = [partials[2 * i] + partials[2 * i + 1] for i in range(width)]
    return partials[0]


def _max(values):
    # Keep MFMA -> max dependencies visible to LLVM. The earlier opaque
    # max tree failed repeated-bit-exact grid checks; maxnum passed without
    # the NaN-propagating compare/select expansion of maximumf.
    def maximum(a, b):
        return fx.Float32(llvm.call_intrinsic(fx.Float32.ir_type, "llvm.maxnum.f32",
                                              [a.ir_value(), b.ir_value()], [], []))
    parts = [maximum(values[i], values[i + 1]) for i in range(0, 16, 2)]
    for width in (4, 2, 1):
        parts = [maximum(parts[2 * i], parts[2 * i + 1]) for i in range(width)]
    return parts[0]


def _cross(value, addresses):
    return tuple(fx.Float32(llvm.inline_asm(fx.Float32.ir_type, [address.ir_value(), value.ir_value()],
        "ds_bpermute_b32 $0, $1, $2", "=v,v,v,~{memory}", has_side_effects=True)) for address in addresses)


def _center(values, scale, maximum):
    return fx.Vector.from_elements([fx.Float32(llvm.inline_asm(
        fx.Float32.ir_type, [values[i].ir_value(), scale.ir_value(), maximum.ir_value()],
        "v_fma_f32 $0, $1, $2, -$3", "=v,v,v,v", has_side_effects=False)) for i in range(16)], fx.Float32)


def _exps(values):
    return fx.Vector.from_elements([_exp(values[i]) for i in range(16)], fx.Float32)


def _pack(values):
    bits = fx.Vector(values).bitcast(fx.Uint32) + fx.Uint32(0x8000)
    selector = fx.Int32(0x07060302)
    words = [fx.Int32(llvm.inline_asm(fx.Int32.ir_type,
        [bits[i + 1].ir_value(), bits[i].ir_value(), selector.ir_value()],
        "v_perm_b32 $0, $1, $2, $3", "=v,v,v,s", has_side_effects=True)) for i in range(0, 16, 2)]
    return fx.Vector.from_elements(words, fx.Int32).bitcast(fx.BFloat16)


@flyc.jit
def _mask(scores, tile, row, q_len, kv_len, CAUSAL: fx.Constexpr[bool]):
    scores = fx.Vector(scores)
    if fx.const_expr(CAUSAL):
        if row - (fx.Int32(gpu.thread_id("x")) & 15) + kv_len - q_len < (tile + 1) * BN:
            bound = _pin_i32(_min(kv_len - 1, kv_len - q_len + row) - tile * BN
                             - ((fx.Int32(gpu.thread_id("x")) >> 4) & 3) * 8)
            scores = fx.Vector.from_elements([(bound >= (i // 8) * 32 + i % 8).select(scores[i], fx.Float32(float("-inf")))
                                              for i in range(16)], fx.Float32)
    else:
        if kv_len - tile * BN < BN:
            bound = _pin_i32(kv_len - 1 - tile * BN - ((fx.Int32(gpu.thread_id("x")) >> 4) & 3) * 8)
            scores = fx.Vector.from_elements([(bound >= (i // 8) * 32 + i % 8).select(scores[i], fx.Float32(float("-inf")))
                                              for i in range(16)], fx.Float32)
    return scores


@flyc.jit
def _v_tail(values, remaining):
    words = fx.Vector(values).bitcast(fx.Int32)
    if remaining < BN:
        limit = _pin_i32(remaining - ((fx.Int32(gpu.thread_id("x")) >> 4) & 3) * 8)
        masks = []
        for i in fx.range_constexpr(8):
            token = (i // 4) * 32 + (i % 4) * 2
            mask = (limit > token).select(fx.Int32(0xFFFF), fx.Int32(0))
            mask = mask | (limit > token + 1).select(fx.Int32(-65536), fx.Int32(0))
            masks.append(mask)
        words = fx.Vector.from_elements([words[i] & masks[i % 8] for i in range(words.numel)], fx.Int32)
    return words.bitcast(fx.BFloat16)


def _pages(table, tile, last, kv_len, page_size):
    first = _min(tile, last) * BN // page_size
    page0 = _prefetch_page(table, first)
    page1 = _prefetch_page(table, _min(first + 1, (kv_len - 1) // page_size)) if page_size == 32 else page0
    return page0, page1


@flyc.jit
def _body(Q, K, V, O, LSE, QS, KS, VS, table, storage, q0, q_len, kv_len, head, qb,
          H: fx.Constexpr[int], HK: fx.Constexpr[int], NP: fx.Constexpr[int], PAGE: fx.Constexpr[int],
          CAUSAL: fx.Constexpr[bool], PER_TOKEN: fx.Constexpr[bool], WITH_LSE: fx.Constexpr[bool],
          SCALE: fx.Constexpr[float], STAGGER: fx.Constexpr[bool]):
    # Explicit closure dependencies keep native caching aware of nested-phase
    # edits after FlyDSL rewrites runtime branches into local functions.
    read_k, read_v, write = _read_k, _read_v, _write
    load_k, load_v, qk, pv = _load_k, _load_v, _qk, _pv
    local_sum, local_max, cross = _sum, _max, _cross
    center, exps, pack, mask = _center, _exps, _pack, _mask
    rescale, advance_max = _rescale, _advance_max
    pages_for_tile = _pages
    tid = fx.Int32(gpu.thread_id("x"))
    wave, lane = _uniform(tid >> 6), tid & 63
    q_start = qb * BM
    row = q_start + wave * 16 + (lane & 15)
    valid = _min(fx.Int32(BM), q_len - q_start)
    hkv = head // (H // HK)
    qptr = fx.get_iter(Q) + ((fx.Int64(q0) + fx.Int64(q_start)) * (H * D) + fx.Int64(head) * D)
    gq = _buffer(fx.make_view(qptr, fx.make_layout(BM * H * D, 1)), valid * H * D * 2)
    gk = _buffer(fx.make_view(fx.get_iter(K) + hkv * PAGE * D,
                            fx.make_layout((NP * HK - hkv) * PAGE * D, 1)), (NP * HK - hkv) * PAGE * D * 2)
    q = _q_fragment(gq, (wave * 16 + (lane & 15)) * (H * D * 2), lane)
    scale = fx.Float32(KS[0]) * fx.Float32(SCALE * LOG2E)
    if fx.const_expr(PER_TOKEN):
        scale = scale * rocdl.make_buffer_tensor(QS, max_size=False)[(q0 + row) * H + head]
    else:
        scale = scale * QS[0]
    scale = fx.Float32(scale)
    tiles = (kv_len + BN - 1) // BN
    if fx.const_expr(CAUSAL):
        end = (q_start + valid + kv_len - q_len + BN - 1) // BN
        tiles = _min(tiles, (end > 0).select(end, fx.Int32(1)))
    last = tiles - 1
    page0, page01 = pages_for_tile(table, fx.Int32(0), last, kv_len, PAGE)
    page1, page11 = pages_for_tile(table, fx.Int32(1), last, kv_len, PAGE)
    page2, page21 = pages_for_tile(table, fx.Int32(2), last, kv_len, PAGE)
    page0, page01 = _page_ready(page0), _page_ready(page01)
    page1, page11 = _page_ready(page1), _page_ready(page11)
    page2, page21 = _page_ready(page2), _page_ready(page21)
    shared = fx.Int32(fx.ptrtoint(fx.get_iter(storage)))
    krow = (lane & 3) + ((lane & 12) << 1)
    kr = tuple(_pin_i32(shared + (lane >> 4) * K_PITCH + (((krow + n * 4) * 16) ^ (((lane >> 4) & 1) * 64))) for n in range(2))
    kw = _pin_i32(shared + (tid >> 6) * K_PITCH + ((tid & 63) * 16 ^ (((tid >> 6) & 1) * 64)))
    vr = _pin_i32(shared + K_BYTES + (lane >> 4) * 4096 + (lane & 15) * 16)
    vw = _pin_i32(shared + K_BYTES + tid * 16)
    klane = _pin_i32((tid >> 6) * (PAGE * 16) + (tid & (min(PAGE, BN) - 1)) * 16)
    vlane = _pin_i32(tid * 16)
    vbase = _pin_s64(fx.Int64(fx.ptrtoint(fx.get_iter(V))) + fx.Int64(hkv) * (PAGE * D * 2))
    cross_addresses = tuple(_pin_i32((lane ^ offset) * 4) for offset in (16, 32, 48))
    k0 = load_k(gk, klane, page0, fx.Int32(0), HK, PAGE, page01)
    v0 = load_v(vbase, vlane, page0, fx.Int32(0), HK, PAGE, page01)
    _wait(vmcnt=0)
    write(kw, k0, 8 * K_PITCH)
    write(vw, v0, 8192)
    _wait(lgkmcnt=0)
    _stage_end()
    if fx.const_expr(STAGGER):
        _stage_end()
    k = read_k(kr, 0)
    _wait(lgkmcnt=0)
    _stage_end()
    lo = qk(q, k)
    o0 = _pin(fx.Vector.filled(32, 0.0, fx.Float32))
    o1 = _pin(fx.Vector.filled(32, 0.0, fx.Float32))
    _schedule(32, 3, 5)
    _stage_end()
    k = read_k(kr, 1)
    k1 = load_k(gk, klane, page1, _min(fx.Int32(1), last), HK, PAGE, page11)
    _wait(lgkmcnt=0)
    _stage_end()
    hi = qk(q, k)
    scores = mask(_join(lo, hi), fx.Int32(0), row, q_len, kv_len, CAUSAL)
    maximum = local_max(scores)
    maximum = _maximum(maximum, maximum.shuffle_xor(16, 64))
    maximum = _maximum(maximum, maximum.shuffle_xor(32, 64))
    maximum = _maximum(maximum * scale, fx.Float32(-1.0e30)) + 1.0
    scores = center(scores, scale, maximum)
    row_sum = fx.Float32(0.0)
    _stage_end()
    _wait(vmcnt=0)
    write(kw, k1, 8 * K_PITCH)
    _wait(lgkmcnt=0)
    _stage_end()
    _stage_end()

    @flyc.jit
    def phase(previous, maximum, row_sum, o0, o1, pending, current_page, current_page1, next_page, next_page1, t):
        previous, maximum, row_sum = fx.Vector(previous), fx.Float32(maximum), fx.Float32(row_sum)
        o0, o1, t = fx.Vector(o0), fx.Vector(o1), fx.Int32(t)
        # S0 publishes V(t-1) only after both preceding V-high readers.
        k = read_k(kr, 0, vw, fx.Vector(pending))
        vp = load_v(vbase, vlane, current_page, t, HK, PAGE, current_page1)
        request, request1 = pages_for_tile(table, t + 2, last, kv_len, PAGE)
        future_page, future_page1 = _page_ready(request), _page_ready(request1)
        _stage_end()
        # S1: QK-low overlaps the previous tile's exponential work.
        lo = qk(q, k)
        previous = exps(previous)
        _schedule(32, 1, 1, True)
        _stage_end()
        # S2: retire K-high LDS reads; V(t) stays in staging registers.
        k = read_k(kr, 1)
        _wait(lgkmcnt=0)
        _stage_end()
        # S3: QK-high overlaps local sum and BF16 probability packing.
        hi = qk(q, k)
        total = local_sum(previous)
        p = pack(previous)
        _schedule(32, 3, 2)
        _stage_end()
        # S4: three independent cross-row shuffles share the V-read wait.
        sums = cross(total, cross_addresses)
        v = read_v(vr, 0)
        kp = load_k(gk, klane, next_page, _min(t + 1, last), HK, PAGE, next_page1)
        _wait(lgkmcnt=0)
        _stage_end()
        # S5: PV-low, finish the row sum and find the next local maximum.
        o0 = pv(p, v, o0)
        row_sum = row_sum + ((total + sums[0]) + (sums[1] + sums[2]))
        current = mask(_join(lo, hi), t, row, q_len, kv_len, CAUSAL)
        candidate = local_max(current)
        _schedule(32, 2, 3)
        _stage_end()
        # S6: V-high and cross-row max; K can be overwritten only now,
        # after BOTH staggered wave groups have consumed K-high.
        maxima = cross(candidate, cross_addresses)
        _wait(vmcnt=0)
        v = read_v(vr, 1, kw, kp)
        _wait(lgkmcnt=0)
        _stage_end()
        # S7: PV-high, center the next scores, and preserve lazy rescaling.
        o1 = pv(p, v, o1)
        candidate = _maximum(_maximum(candidate, maxima[0]), _maximum(maxima[1], maxima[2])) * scale
        ballot = fx.Int64(rocdl.ballot(fx.Int64.ir_type, (candidate > maximum + 7.0).ir_value()))
        new_max = (candidate > maximum + 7.0).select(candidate + 1.0, maximum)
        current = center(current, scale, new_max)
        _schedule(32, 3, 4)
        o0, o1, row_sum = rescale(o0, o1, row_sum, maximum, new_max, ballot)
        new_max = advance_max(maximum, new_max)
        _stage_end()
        return current, new_max, row_sum, o0, o1, vp, next_page, next_page1, future_page, future_page1

    pending = v0
    for t in range(fx.Int32(1), tiles, fx.Int32(1)):
        scores, maximum, row_sum, o0, o1, pending, page1, page11, page2, page21 = phase(
            scores, maximum, row_sum, o0, o1, pending, page1, page11, page2, page21, t)

    write(vw, fx.Vector(pending), 8192)
    scores = exps(fx.Vector(scores))
    total = local_sum(scores)
    total = total + total.shuffle_xor(16, 64)
    total = total + total.shuffle_xor(32, 64)
    row_sum = fx.Float32(row_sum) + total
    p = pack(scores)
    _wait(vmcnt=0, lgkmcnt=0)
    _stage_end()
    _stage_end()
    v = read_v(vr, 0)
    _wait(lgkmcnt=0)
    rocdl.sched_barrier(0)
    v.store(_v_tail(v.load(), kv_len - last * BN))
    _stage_end()
    o0 = pv(p, v, o0)
    _stage_end()
    v = read_v(vr, 1)
    _wait(lgkmcnt=0)
    rocdl.sched_barrier(0)
    v.store(_v_tail(v.load(), kv_len - last * BN))
    _stage_end()
    o1 = pv(p, v, o1)
    _stage_end()
    if fx.const_expr(not STAGGER):
        _stage_end()

    inv = (row_sum > 0.0).select(fx.Float32(1.0) / row_sum, fx.Float32(0.0)) * VS[0]
    out_tid = _pin_i32(fx.Int32(gpu.thread_id("x")))
    out_row = (out_tid >> 6) * 16 + (out_tid & 15)
    optr = fx.get_iter(O) + ((fx.Int64(q0) + fx.Int64(q_start)) * (H * D) + fx.Int64(head) * D)
    obuf = rocdl.make_buffer_tensor(fx.make_view(optr, fx.make_layout(BM * H * D, 1)),
                                    num_records_bytes=valid * H * D * 2)
    outputs = _join(o0, o1)
    atom = fx.make_copy_atom(rocdl.BufferCopy128b(), fx.BFloat16)
    # Reuse 16 KiB for each 64-column C-shuffle, then coalesced 128-bit stores.
    for half in fx.range_constexpr(4):
        for n in fx.range_constexpr(4):
            values = fx.Vector.from_elements([outputs[half * 16 + n * 4 + i] * inv for i in range(4)], fx.Float32)
            words = _pack_bf16(values)
            col = n * 16 + ((out_tid >> 4) & 3) * 4
            element = (out_row * 64 + col) ^ ((out_row & 15) * 4)
            address = fx.Int32(fx.ptrtoint(fx.get_iter(storage) + element * 2))
            llvm.inline_asm(ir.Type.parse("!llvm.void"), [address.ir_value(), words.ir_value()],
                            "ds_write_b64 $0, $1", "v,v,~{memory}", has_side_effects=True)
        _wait(lgkmcnt=0)
        _stage_end()
        for i in fx.range_constexpr(2):
            element = out_tid * 8 + i * THREADS * 8
            read_row, read_col = element // 64, element % 64
            address = shared + (element ^ ((read_row & 14) * 4)) * 2
            words = _read_address(address)
            _wait(lgkmcnt=0)
            rocdl.sched_barrier(0)
            src = fx.make_rmem_tensor(8, fx.BFloat16)
            src.store(fx.Vector.from_elements([((read_row & 1) == 0).select(words[j], words[j ^ 2])
                                               for j in range(4)], fx.Int32).bitcast(fx.BFloat16))
            offset = read_row * (H * D) + half * 64 + read_col
            fx.copy(atom, src, fx.make_view(fx.get_iter(obuf) + offset, fx.make_layout(8, 1)))
        _stage_end()
    if fx.const_expr(WITH_LSE):
        if ((out_tid & 63) < 16) & (out_row < valid):
            log_l = fx.Float32(llvm.call_intrinsic(fx.Float32.ir_type, "llvm.log2.f32", [row_sum.ir_value()], [], []))
            LSE[(q0 + q_start + out_row) * H + head] = (row_sum > 0.0).select(
                (fx.Float32(maximum) + log_l) * fx.Float32(math.log(2.0)), fx.Float32(float("-inf")))


@flyc.jit
def _work(Q, K, V, O, LSE, CQ, KI, PAGES, LAST, QS, KS, VS, storage, head, batch, qb,
          H: fx.Constexpr[int], HK: fx.Constexpr[int], NP: fx.Constexpr[int], PAGE: fx.Constexpr[int],
          CAUSAL: fx.Constexpr[bool], PER_TOKEN: fx.Constexpr[bool], WITH_LSE: fx.Constexpr[bool], SCALE: fx.Constexpr[float]):
    body = _body
    q0 = _uniform(CQ[batch])
    q_len = _uniform(CQ[batch + 1]) - q0
    start = _uniform(KI[batch])
    pages = _uniform(KI[batch + 1]) - start
    kv_len = (pages - 1) * PAGE + _uniform(LAST[batch])
    table = fx.make_view(fx.get_iter(PAGES) + start, fx.make_layout(pages, 1))
    if qb * BM < q_len:
        group = _uniform(fx.Int32(gpu.thread_id("x")) >> 8)
        if group != 0:
            body(Q, K, V, O, LSE, QS, KS, VS, table, storage, q0, q_len, kv_len, head, qb,
                 H, HK, NP, PAGE, CAUSAL, PER_TOKEN, WITH_LSE, SCALE, True)
        else:
            body(Q, K, V, O, LSE, QS, KS, VS, table, storage, q0, q_len, kv_len, head, qb,
                 H, HK, NP, PAGE, CAUSAL, PER_TOKEN, WITH_LSE, SCALE, False)


@flyc.kernel(known_block_size=[THREADS, 1, 1])
def _attention_256_kernel_942(Q: fx.Tensor, K: fx.Tensor, V: fx.Tensor, O: fx.Tensor, LSE: fx.Tensor,
    CQ: fx.Tensor, KI: fx.Tensor, PAGES: fx.Tensor, LAST: fx.Tensor, QS: fx.Tensor, KS: fx.Tensor, VS: fx.Tensor,
    H: fx.Constexpr[int], HK: fx.Constexpr[int], NP: fx.Constexpr[int], B: fx.Constexpr[int],
    MAX_Q: fx.Constexpr[int], PAGE: fx.Constexpr[int], CAUSAL: fx.Constexpr[bool],
    PER_TOKEN: fx.Constexpr[bool], WITH_LSE: fx.Constexpr[bool], SCALE: fx.Constexpr[float],
    PERSISTENT: fx.Constexpr[bool], CUS: fx.Constexpr[int]):
    storage = fx.SharedAllocator().allocate(fx.Array[fx.Int8, LDS_BYTES, 16]).peek().view(fx.make_layout(LDS_BYTES, 1))
    if fx.const_expr(PERSISTENT):
        work = fx.Int32(gpu.block_id("x"))
        while work < H * B * ((MAX_Q + BM - 1) // BM):
            head, batch, qb = work % H, (work // H) % B, work // (H * B)
            _work(Q, K, V, O, LSE, CQ, KI, PAGES, LAST, QS, KS, VS, storage, head, batch, qb,
                  H, HK, NP, PAGE, CAUSAL, PER_TOKEN, WITH_LSE, SCALE)
            work = work + CUS
    else:
        head, batch, qb = fx.Int32(gpu.block_id("x")), fx.Int32(gpu.block_id("y")), fx.Int32(gpu.block_id("z"))
        _work(Q, K, V, O, LSE, CQ, KI, PAGES, LAST, QS, KS, VS, storage, head, batch, qb,
              H, HK, NP, PAGE, CAUSAL, PER_TOKEN, WITH_LSE, SCALE)


@flyc.jit
def _launch_attention_256(Q: fx.Tensor, K: fx.Tensor, V: fx.Tensor, O: fx.Tensor, LSE: fx.Tensor,
    CQ: fx.Tensor, KI: fx.Tensor, PAGES: fx.Tensor, LAST: fx.Tensor, QS: fx.Tensor, KS: fx.Tensor, VS: fx.Tensor,
    H: fx.Constexpr[int], HK: fx.Constexpr[int], NP: fx.Constexpr[int], B: fx.Constexpr[int],
    MAX_Q: fx.Constexpr[int], DQ: fx.Constexpr[int], DV: fx.Constexpr[int], PAGE: fx.Constexpr[int],
    CAUSAL: fx.Constexpr[bool], PER_TOKEN: fx.Constexpr[bool], WITH_LSE: fx.Constexpr[bool],
    SCALE: fx.Constexpr[float], PERSISTENT: fx.Constexpr[bool], CUS: fx.Constexpr[int], stream: fx.Stream):
    grid = (min(CUS, H * B * ((MAX_Q + BM - 1) // BM)), 1, 1) if PERSISTENT else (H, B, (MAX_Q + BM - 1) // BM)
    _attention_256_kernel_942(Q, K, V, O, LSE, CQ, KI, PAGES, LAST, QS, KS, VS,
        H, HK, NP, B, MAX_Q, PAGE, CAUSAL, PER_TOKEN, WITH_LSE, SCALE, PERSISTENT, CUS,
        value_attrs={"rocdl.waves_per_eu": 2, "passthrough": [["target-features", "-packed-fp32-ops"]]},
    ).launch(grid=grid, block=(THREADS, 1, 1), stream=stream)