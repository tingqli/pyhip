"""Direct-paged gfx950 BF16 D128/D192, V128 attention with SWA/sinks.

The attention kernel reads SHUFFLE-5D page64 K/V through the current page table.
No gather, linear-KV workspace, data cache or auxiliary GPU launch is used.
OPUS's Q/V LDS alias, compile-time wave groups and ping/pong pipeline are retained.
Only LDS mode is supported; persistent=None keeps the original single-launch
non-persistent default. Local FlyDSL adapters do not alter the native schedule.
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


BM, BN, DV, THREADS = 256, 64, 128, 512
V_TILE = 8 * 2 * 520
LOG2E = math.log2(math.e)


def _min(a, b):
    return (a < b).select(a, b)


def _max(a, b):
    return (a > b).select(a, b)


def _uniform(value):
    return fx.Int32(rocdl.readfirstlane(fx.Int32.ir_type, fx.Int32(value).ir_value()))


def _prefetch_page(table, index):
    # Page indices are wave-uniform. SMEM avoids draining the in-flight KV
    # DMA queue, unlike a global_load + vmcnt(0) + readfirstlane sequence.
    address = fx.Int64(fx.ptrtoint(fx.get_iter(table) + fx.Int64(index)))
    return fx.Int32(llvm.inline_asm(
        fx.Int32.ir_type, [address.ir_value()],
        "s_load_dword $0, $1, 0", "=s,s,~{memory}",
        has_side_effects=True,
    ))


def _page_ready(value, vmcnt=None):
    # The tied result cannot be used until the scalar/LDS queue is drained.
    wait = "s_waitcnt lgkmcnt(0)" if vmcnt is None else f"s_waitcnt vmcnt({vmcnt}) lgkmcnt(0)"
    return fx.Int32(llvm.inline_asm(
        fx.Int32.ir_type, [fx.Int32(value).ir_value()],
        wait, "=s,0", has_side_effects=True,
    ))


def _exp(value):
    value = fx.Float32(value)
    return fx.Float32(llvm.call_intrinsic(
        fx.Float32.ir_type, "llvm.amdgcn.exp2.f32", [value.ir_value()], [], []
    ))


def _scale_sub(value, scale, maximum):
    value, scale, maximum = fx.Float32(value), fx.Float32(scale), fx.Float32(maximum)
    # Keep scalar FMA: packed FP32 VALU cannot co-issue with this MFMA schedule.
    return fx.Float32(llvm.inline_asm(
        fx.Float32.ir_type, [value.ir_value(), scale.ir_value(), maximum.ir_value()],
        "v_fma_f32 $0, $1, $2, -$3", "=v,v,v,v", has_side_effects=False
    ))


def _pin_i32(value):
    return fx.Int32(llvm.inline_asm(
        fx.Int32.ir_type, [fx.Int32(value).ir_value()], "", "=v,0", has_side_effects=True
    ))


def _pin_f32(value):
    return fx.Float32(llvm.inline_asm(
        fx.Float32.ir_type, [fx.Float32(value).ir_value()], "", "=v,0", has_side_effects=True
    ))


def _slice(values, begin, count, dtype=fx.Float32):
    return fx.Vector.from_elements([values[begin + i] for i in range(count)], dtype)


def _join(left, right, count=16, dtype=fx.Float32):
    return fx.Vector.from_elements(
        [left[i] for i in range(count)] + [right[i] for i in range(count)], dtype
    )


def _pin(values, chunk=8):
    # Chunked tied operands avoid demanding one contiguous 32/64-VGPR block.
    values = fx.Vector(values)
    dtype = values.dtype
    if dtype.width < 32:
        values = values.bitcast(fx.Int32)
    chunks = []
    for start in range(0, values.numel, chunk):
        part = _slice(values, start, min(chunk, values.numel - start), values.dtype)
        part = fx.Vector(llvm.inline_asm(
            part.ir_value().type, [part.ir_value()], "", "=v,0", has_side_effects=True
        ))
        chunks.extend(part[i] for i in range(part.numel))
    return fx.Vector.from_elements(chunks, values.dtype).bitcast(dtype)


def _cross32(value, maximum=False):
    pair = rocdl.permlane32_swap(
        ir.Type.parse("!llvm.struct<(i32, i32)>"),
        value.bitcast(fx.Int32).ir_value(), value.bitcast(fx.Int32).ir_value(), False, True
    )
    a = fx.Int32(llvm.extractvalue(fx.Int32.ir_type, pair, [0])).bitcast(fx.Float32)
    b = fx.Int32(llvm.extractvalue(fx.Int32.ir_type, pair, [1])).bitcast(fx.Float32)
    return a.maximumf(b) if maximum else a + b


def _row_max(scores):
    scores = fx.Vector(scores)
    result = fx.Float32(-1.0e30)
    for i in range(32):
        result = result.maximumf(scores[i])
    return _cross32(result, True)


def _row_sum(scores):
    scores = fx.Vector(scores)
    result = fx.Float32(0.0)
    for i in range(32):
        result = result + scores[i]
    return _cross32(result)


def _exp_slice(scores, start, end):
    scores = fx.Vector(scores)
    return fx.Vector.from_elements(
        [_exp(scores[i]) if start <= i < end else scores[i] for i in range(32)], fx.Float32
    )


def _center_slice(scores, scale, maximum, start, end):
    scores = fx.Vector(scores)
    return _pin(fx.Vector.from_elements(
        [_pin_f32(_scale_sub(scores[i], scale, maximum)) if start <= i < end else scores[i]
         for i in range(32)], fx.Float32
    ), 32)


def _stage_end():
    rocdl.sched_barrier(0)
    rocdl.s_barrier()
    rocdl.sched_barrier(0)


def _schedule(pairs, valu, group, exp=False):
    for _ in range(pairs):
        rocdl.sched_group_barrier(rocdl.mask_mfma, 1, group)
        rocdl.sched_group_barrier(0x400 if exp else 0x002, valu, group)


def _dma(resource, storage, destination, voffset, soffset):
    rocdl.raw_ptr_buffer_load_lds(
        as_ir_value(resource), as_ir_value(fx.to_llvm_ptr(fx.get_iter(storage) + destination)),
        as_ir_value(fx.Int32(16)), as_ir_value(fx.Int32(voffset)), as_ir_value(fx.Int32(soffset)),
        as_ir_value(fx.Int32(0)), as_ir_value(fx.Int32(0))
    )


def _read_operand(storage, base, immediate=0):
    # The copy atoms currently model all direct-LDS DMA as one aliasing store:
    # they insert vmcnt(0) even when the phase reads a different ring buffer.
    # Match OPUS's explicit LDS-read boundary; the caller owns lgkmcnt(0) and
    # the cross-wave stage barrier. No loaded value may be consumed before it.
    # ds_read has a 16-bit byte immediate; Q's third D chunk exceeds it.
    high = (immediate * 2) & ~0xFFFF
    address = fx.Int32(fx.ptrtoint(fx.get_iter(storage) + (base + high // 2)))
    return fx.Vector(llvm.inline_asm(
        ir.VectorType.get([4], fx.Int32.ir_type), [as_ir_value(address)],
        f"ds_read_b128 $0, $1 offset:{(immediate * 2) & 0xFFFF}", "=v,v,~{memory}",
        has_side_effects=True,
    )).bitcast(fx.BFloat16)


def _read_qk(storage, base, dq, q=False):
    mma = _tiled_mma()
    tile = rmem((BM if q else BN // 2, dq), fx.BFloat16)
    result = mma.make_fragment_B(tile) if q else mma.make_fragment_A(tile)
    for k in range(dq // 16):
        offset = (k // 4) * 16448 + (k % 4) * 16 if q else k * 1040
        result[None, 0, k].store(_read_operand(storage, base, offset))
    return result


def _read_v(storage, base):
    # K permutes token bits 2/3 so consecutive P registers correspond to
    # eight consecutive tokens. V can then use native b128 reads, no transpose.
    # Assemble packed words before the BF16 store: repeated BF16 slice stores
    # followed by the tail-mask bitcast otherwise introduce redundant BFI moves.
    chunks = [_read_operand(storage, base, n * 256 + k * 2080).bitcast(fx.Int32)
              for n in range(2) for k in range(4)]
    words = fx.Vector.from_elements([part[i] for part in chunks for i in range(4)], fx.Int32)
    result = rmem(fx.make_layout((8, 4, 2), (1, 8, 32)), fx.BFloat16)
    result.store(words.bitcast(fx.BFloat16))
    return result


@flyc.jit
def _mask_v_tail(values, remaining):
    words = fx.Vector(values).bitcast(fx.Int32)
    if remaining < BN:
        # 0 * NaN is NaN: masking P alone cannot sanitize poisoned V padding.
        lane = _pin_i32(fx.Int32(gpu.thread_id("x")))
        limit = _pin_i32(remaining - ((lane >> 5) & 1) * 8)
        masks = []
        for i in fx.range_constexpr(16):
            token = (i // 4) * 16 + (i % 4) * 2
            bits = (limit > token).select(fx.Int32(0xFFFF), fx.Int32(0))
            bits = bits | (limit > token + 1).select(fx.Int32(-65536), fx.Int32(0))
            masks.append(_pin_i32(bits))
        words = fx.Vector.from_elements([words[i] & masks[i % 16] for i in range(32)], fx.Int32)
    return words.bitcast(fx.BFloat16)


def _tiled_mma():
    # Eight waves partition query rows; each pipeline superunit covers 32 K
    # columns (QK) or 64 output columns (PV), not the entire BN/DV tile.
    atom = fx.make_mma_atom(rocdl.MFMA(32, 32, 16, fx.BFloat16))
    return fx.make_tiled_mma(atom, fx.make_layout((1, 8, 1), (1, 1, 0)))


def _qk(q, k, active=None):
    mma = _tiled_mma()
    acc = mma.make_fragment_C(rmem((BN // 2, BM), fx.Float32))
    acc.fill(0.0)
    if active is None:
        fx.gemm(mma, acc, k, q, acc, traversal_order="kmn")
    else:
        @flyc.jit
        def compute_visible():
            if active:
                fx.gemm(mma, acc, k, q, acc, traversal_order="kmn")
        compute_visible()
    return acc.load()


def _pv(p, v, o0, o1, active=None):
    mma = _tiled_mma()
    probs = mma.make_fragment_B(rmem((BM, BN), fx.BFloat16))
    probs.store(p)
    acc = mma.make_fragment_C(rmem((DV // 2, BM), fx.Float32))
    acc.store(_join(o0, o1))
    # View packed V as (atom values, output columns, reduction). No register
    # transpose. Traversal is fastest-axis first: M,N,K gives an outer K loop
    # alternating the two independent M accumulators, as in the OPUS schedule.
    if active is None:
        fx.gemm(mma, acc, select(v, [0, 2, 1]), probs, acc, traversal_order="mnk")
    else:
        @flyc.jit
        def compute_visible():
            if active:
                fx.gemm(mma, acc, select(v, [0, 2, 1]), probs, acc, traversal_order="mnk")
        compute_visible()
    return acc[None, 0, 0].load(), acc[None, 1, 0].load()


@flyc.jit
def _mask(scores, tile, wave_row, kv_len, q_len, causal: fx.Constexpr[bool], window: fx.Constexpr[int]):
    scores, tile = fx.Vector(scores), fx.Int32(tile)
    lane_half = (fx.Int32(gpu.thread_id("x")) & 32) // 4
    if fx.const_expr(window >= 0):
        # The inclusive window is [q + KV - Q - window, q + KV - Q].
        # Unsigned distance tests both bounds, including negative diagonals.
        window_lane = _pin_i32(fx.Int32(gpu.thread_id("x")))
        diagonal = _pin_i32(kv_len - q_len + wave_row + (window_lane & 31)
                    - tile * BN - (window_lane & 32) // 4)
        scores = fx.Vector.from_elements([
            (fx.Uint32(diagonal - ((i // 8) * 16 + i % 8))
             <= fx.Uint32(window)).select(scores[i], fx.Float32(float("-inf")))
            for i in range(32)
        ], fx.Float32)
    elif fx.const_expr(causal):
        if wave_row + kv_len - q_len < (tile + 1) * BN:
            mask_lane = _pin_i32(fx.Int32(gpu.thread_id("x")))
            boundary = _pin_i32(kv_len - q_len + wave_row + (mask_lane & 31)
                                - tile * 64 - (mask_lane & 32) // 4)
            scores = fx.Vector.from_elements([
                (boundary >= (i // 8) * 16 + i % 8).select(
                    scores[i], fx.Float32(float("-inf")))
                for i in range(32)
            ], fx.Float32)
    else:
        # All callers pass a valid KV tile, so only its remaining length is
        # needed; avoid a separate modulo, last-tile division and conjunction.
        remaining = kv_len - tile * BN
        if remaining < BN:
            # Materialize inside the tail branch. Otherwise LICM keeps 32 wave
            # predicates (64 SGPRs) live throughout the entire main loop.
            border_limit = _pin_i32(remaining - 1 - lane_half)
            scores = fx.Vector.from_elements([
                (border_limit >= (i // 8) * 16 + i % 8).select(
                    scores[i], fx.Float32(float("-inf")))
                for i in range(32)
            ], fx.Float32)
    return scores


@flyc.jit
def _rescale(o0, o1, o2, o3, old_max, new_max, row_sum, mask):
    o0, o1, o2, o3 = fx.Vector(o0), fx.Vector(o1), fx.Vector(o2), fx.Vector(o3)
    old_max, new_max, row_sum = fx.Float32(old_max), fx.Float32(new_max), fx.Float32(row_sum)
    if mask != fx.Int64(0):
        correction = _exp(old_max - new_max)
        o0 = o0 * correction
        o1 = o1 * correction
        o2 = o2 * correction
        o3 = o3 * correction
        row_sum = row_sum * correction
    return o0, o1, o2, o3, row_sum


@flyc.jit
def _attention_body(
    Q, K, V, O, LSE, QS, KS, VS, SINK, TABLE, storage, q0, q_len, kv_len, batch, head, qb,
    H: fx.Constexpr[int], HK: fx.Constexpr[int], NP: fx.Constexpr[int],
    DQ: fx.Constexpr[int], WINDOW: fx.Constexpr[int], HAS_SINK: fx.Constexpr[bool],
    QROW: fx.Constexpr[int], QHEAD: fx.Constexpr[int],
    OROW: fx.Constexpr[int], OHEAD: fx.Constexpr[int],
    PER_TOKEN: fx.Constexpr[bool], CAUSAL: fx.Constexpr[bool],
    WITH_LSE: fx.Constexpr[bool], SCALE: fx.Constexpr[float], STAGGER: fx.Constexpr[bool],
    REVERSE: fx.Constexpr[bool] = False, MERGED: fx.Constexpr[bool] = False,
    PRUNE_SWA: fx.Constexpr[bool] = False,
):
    K_CHUNKS = DQ // 64
    K_TILE = 8 * K_CHUNKS * 520
    KEEP_VMCNT = K_CHUNKS + 2
    tid = fx.Int32(gpu.thread_id("x"))
    wave = _uniform(tid >> 6)
    lane = tid & 63
    if fx.const_expr(MERGED):
        lane = _pin_i32(lane)
    q_start = qb * BM
    wave_row = q_start + wave * 32
    row = wave_row + (lane & 31)
    q_valid = _min(fx.Int32(BM), q_len - q_start)
    hkv = head // (H // HK)
    q_ptr = fx.get_iter(Q) + ((fx.Int64(q0) + fx.Int64(q_start)) * QROW + fx.Int64(head) * QHEAD)
    q_view = fx.make_view(q_ptr, fx.make_layout((BM, DQ), (QROW, 1)))
    k_ptr = fx.get_iter(K) + (fx.Int64(hkv) * BN * DQ)
    v_ptr = fx.get_iter(V) + (fx.Int64(hkv) * BN * DV)
    gq = resource(q_view, q_valid * QROW * 2)
    gk = resource(
        fx.make_view(k_ptr, fx.make_layout(NP * HK * BN * DQ, 1)),
        (NP * HK - hkv) * BN * DQ * 2,
    )
    gv = resource(
        fx.make_view(v_ptr, fx.make_layout(NP * HK * BN * DV, 1)),
        (NP * HK - hkv) * BN * DV * 2,
    )
    if fx.const_expr(MERGED):
        # Keep the shared scalar scale out of VGPRs across both paired passes.
        scale_bits = _prefetch_page(KS, fx.Int32(0))
        scale = _page_ready(scale_bits).bitcast(fx.Float32) * fx.Float32(SCALE * LOG2E)
    else:
        scale = fx.Float32(KS[0]) * fx.Float32(SCALE * LOG2E)
    if fx.const_expr(PER_TOKEN):
        qs_buf = rocdl.make_buffer_tensor(QS, max_size=False)
        scale = scale * qs_buf[(q0 + row) * H + head]
    else:
        scale = scale * QS[0]
    scale = fx.Float32(llvm.inline_asm(
        fx.Float32.ir_type, [fx.Float32(scale).ir_value()], "", "=v,0", has_side_effects=True
    ))
    rocdl.sched_barrier(0)

    total_tiles = (kv_len + 63) // 64
    tiles = total_tiles
    if fx.const_expr(CAUSAL):
        tiles = _max(_min((q_start + q_valid + kv_len - q_len + 63) // 64, total_tiles), fx.Int32(1))
    first = fx.Int32(0)
    if fx.const_expr(WINDOW >= 0):
        first = _max((q_start + kv_len - q_len - WINDOW) // BN, fx.Int32(0))
        tiles = tiles - first
    last = tiles - 1
    q_lane = lane
    if fx.const_expr(MERGED):
        # Q-only coordinates must die after each prologue, not be hoisted
        # across both paired passes and spilled throughout the hot loop.
        q_lane = _pin_i32(lane)
    local_g = wave + (q_lane >> 3) * 8
    gq_off = (local_g * QROW + (q_lane & 7) * 8) * 2
    gk_off = (wave * 512 + lane * 8) * 2
    gv_off = ((wave >> 1) * 1024 + (wave & 1) * 512 + lane * 8) * 2
    k_row = (lane & 19) | ((lane & 4) << 1) | ((lane & 8) >> 1)
    rk = k_row * 8 + (lane >> 5) * 520
    rq = (q_lane & 7) * 2056 + wave * 256 + ((q_lane >> 3) & 3) * 64 + (q_lane >> 5) * 8
    rv = (lane & 31) * 8 + (lane >> 5) * 1040

    def data_tile(position):
        return first + (last - position if REVERSE else position)

    def visible_wave_tile(position):
        return ((data_tile(position) * BN < wave_row + kv_len - q_len + 32)
                & (data_tile(position) * BN + BN > wave_row + kv_len - q_len - WINDOW)) if PRUNE_SWA else None

    def load_k(physical, slot, begin=0, end=K_CHUNKS):
        offset = physical * (BN * HK * DQ * 2)
        for d in fx.range_constexpr(begin, end):
            _dma(gk, storage, slot * K_TILE + (d * 8 + wave) * 520,
                 gk_off + d * (64 * BN * 2), offset)

    def load_v(physical, slot):
        offset = physical * (BN * HK * DV * 2)
        for d in fx.range_constexpr(2):
            _dma(gv, storage, 2 * K_TILE + slot * V_TILE + (d * 8 + wave) * 520,
                  gv_off + d * (32 * DV * 2), offset)

    page0 = _prefetch_page(TABLE, data_tile(fx.Int32(0)))
    page1 = _prefetch_page(TABLE, data_tile(_min(fx.Int32(1), last)))
    page2 = _prefetch_page(TABLE, data_tile(_min(fx.Int32(2), last)))
    # Q aliases the entire V region until Q has been read into registers.
    for d in fx.range_constexpr(K_CHUNKS):
        for n in fx.range_constexpr(4):
            _dma(gq, storage, 2 * K_TILE + (d * 8 + wave) * 2056 + n * 512,
                 gq_off + d * 128 + n * 64 * QROW * 2, fx.Int32(0))
    page0 = _page_ready(page0)
    page1 = _page_ready(page1)
    page2 = _page_ready(page2)
    load_k(page0, 0)
    wait(vmcnt=K_CHUNKS)
    _stage_end()
    q_frag = _read_qk(storage, 2 * K_TILE + rq, DQ, True)
    load_k(page1, 1)
    wait(lgkmcnt=0, vmcnt=K_CHUNKS)
    _stage_end()
    if fx.const_expr(STAGGER):
        _stage_end()

    k_frag = _read_qk(storage, rk, DQ)
    load_v(page0, 0)
    wait(lgkmcnt=0)
    _stage_end()
    s0 = _qk(q_frag, k_frag, visible_wave_tile(fx.Int32(0)))
    o0 = fx.Vector.filled(16, 0.0, fx.Float32)
    o1 = fx.Vector.filled(16, 0.0, fx.Float32)
    o2 = fx.Vector.filled(16, 0.0, fx.Float32)
    o3 = fx.Vector.filled(16, 0.0, fx.Float32)
    _schedule(DQ // 16, 3 if DQ == 192 else 5, 5)
    o0, o1, o2, o3 = _pin(o0), _pin(o1), _pin(o2), _pin(o3)
    _stage_end()
    k_frag = _read_qk(storage, rk + 256, DQ)
    wait(lgkmcnt=0)
    _stage_end()
    s1 = _qk(q_frag, k_frag, visible_wave_tile(fx.Int32(0)))
    scores = _mask(_join(s0, s1), data_tile(fx.Int32(0)), wave_row, kv_len, q_len, CAUSAL, WINDOW)
    maximum = (_row_max(scores) * scale).maximumf(fx.Float32(-1.0e30))
    row_sum = fx.Float32(0.0)
    if fx.const_expr(HAS_SINK):
        # One zero-value virtual key per head. row_sum is already reduced
        # across lane +/-32, so the sink contributes once, not twice.
        sink_log2 = fx.Float32(SINK[head]) * fx.Float32(LOG2E)
        maximum = maximum.maximumf(sink_log2)
        row_sum = _exp(sink_log2 - maximum)
    scores = _center_slice(scores, scale, maximum, 0, 32)
    wait(vmcnt=2)
    _stage_end()
    load_k(page2, 0)
    _stage_end()

    @flyc.jit
    def phase(previous, maximum, row_sum, o0, o1, o2, o3, current_page, next_page, t, CUR: fx.Constexpr[int], PREV: fx.Constexpr[int]):
        previous = fx.Vector(previous)
        o0, o1, o2, o3 = fx.Vector(o0), fx.Vector(o1), fx.Vector(o2), fx.Vector(o3)
        maximum, row_sum, t = fx.Float32(maximum), fx.Float32(row_sum), fx.Int32(t)
        # S0/S1: memory of t, then QK(t) + exp(t-1).
        k = _read_qk(storage, CUR * K_TILE + rk, DQ)
        if fx.const_expr(STAGGER):
            load_v(current_page, CUR)
        wait(lgkmcnt=0)
        _stage_end()
        future_page = _prefetch_page(TABLE, data_tile(_min(t + 2, last)))
        lo = _qk(q_frag, k, visible_wave_tile(t))
        previous = _pin(_exp_slice(previous, 0, 24), 32)
        if fx.const_expr(DQ == 128):
            _schedule(8, 3, 1, True)
        elif fx.const_expr(STAGGER):
            _schedule(1, 3, 1, True)
            rocdl.sched_group_barrier(rocdl.mask_mfma, 3, 1)
            _schedule(8, 3, 1, True)
            rocdl.sched_group_barrier(rocdl.mask_mfma, 1, 1)
        else:
            rocdl.sched_group_barrier(rocdl.mask_mfma, 3, 1)
            _schedule(1, 2, 1, True)
            _schedule(8, 3, 1, True)
        _stage_end()
        # S2/S3: second QK half + remaining exp, sum, BF16 P.
        k = _read_qk(storage, CUR * K_TILE + rk + 256, DQ)
        if fx.const_expr(not STAGGER):
            load_v(current_page, CUR)
        # One instruction retains both the rolling DMA and page/LDS waits.
        future_page = _page_ready(future_page, KEEP_VMCNT)
        _stage_end()
        hi = _qk(q_frag, k, visible_wave_tile(t))
        previous = _exp_slice(previous, 24, 32)
        row_sum = _pin_f32(row_sum + _row_sum(previous))
        p = _pin(previous.to(fx.BFloat16), 16)
        _schedule(2, 3, 2, True)
        rocdl.sched_group_barrier(rocdl.mask_mfma, 1, 2)
        rocdl.sched_group_barrier(0x400, 2, 2)
        rocdl.sched_group_barrier(0x002, 2, 2)
        if fx.const_expr(DQ == 128):
            _schedule(5, 10, 2)
        else:
            _schedule(5, 6, 2)
            _schedule(1, 5, 2)
            _schedule(2, 6, 2)
            _schedule(1, 2, 2)
        _stage_end()
        current = _join(lo, hi)
        # S4/S5: V(t-1), K(t+2), then PV + current softmax head.
        v = _read_v(storage, 2 * K_TILE + PREV * V_TILE + rv)
        load_k(future_page, CUR, 0, 1 if STAGGER else K_CHUNKS)
        current = _mask(current, data_tile(t), wave_row, kv_len, q_len, CAUSAL, WINDOW)
        wait(lgkmcnt=0)
        if fx.const_expr(REVERSE):
            # Forward phases consume t-1 <= tiles-2: these V pages are full.
            # Reverse traversal can consume the partial final page at t=1.
            rocdl.sched_barrier(0)
            v.store(_mask_v_tail(v.load(), kv_len - data_tile(t - 1) * BN))
        _stage_end()
        o0, o1 = _pv(p, v, o0, o1, visible_wave_tile(t - 1))
        candidate = _row_max(current) * scale
        mask = fx.Int64(rocdl.ballot(fx.Int64.ir_type, (candidate - maximum > 8.0).ir_value()))
        new_max = _pin_f32((mask != fx.Int64(0)).select(maximum.maximumf(candidate), maximum))
        split = 12 if STAGGER else 6
        current = _center_slice(current, scale, new_max, 0, split)
        for count in (5, 5, 6, 4, 2, 4, 4, 4):
            _schedule(1, count, 3)
        _stage_end()
        # S6/S7: other V half, then finish PV and conditional rescale.
        v = _read_v(storage, 2 * K_TILE + PREV * V_TILE + 520 + rv)
        if fx.const_expr(STAGGER):
            load_k(future_page, CUR, 1, K_CHUNKS)
        wait(lgkmcnt=0, vmcnt=KEEP_VMCNT)
        if fx.const_expr(REVERSE):
            rocdl.sched_barrier(0)
            v.store(_mask_v_tail(v.load(), kv_len - data_tile(t - 1) * BN))
        _stage_end()
        o2, o3 = _pv(p, v, o2, o3, visible_wave_tile(t - 1))
        current = _center_slice(current, scale, new_max, split, 32)
        if fx.const_expr(STAGGER):
            for count in (3, 2, 2, 2, 3, 3, 4, 4):
                _schedule(1, count, 4)
        else:
            _schedule(2, 3, 4)
            _schedule(5, 4, 4)
        o0, o1, o2, o3, row_sum = _rescale(o0, o1, o2, o3, maximum, new_max, row_sum, mask)
        _stage_end()
        return current, new_max, row_sum, o0, o1, o2, o3, next_page, future_page

    for t in range(fx.Int32(1), tiles - 1, fx.Int32(2)):
        rocdl.sched_barrier(0)
        scores, maximum, row_sum, o0, o1, o2, o3, page1, page2 = phase(scores, maximum, row_sum, o0, o1, o2, o3, page1, page2, t, 1, 0)
        rocdl.sched_barrier(0)
        scores, maximum, row_sum, o0, o1, o2, o3, page1, page2 = phase(scores, maximum, row_sum, o0, o1, o2, o3, page1, page2, t + 1, 0, 1)
        rocdl.sched_barrier(0)
    rocdl.sched_barrier(0)
    if (tiles & 1) == 0:
        scores, maximum, row_sum, o0, o1, o2, o3, page1, page2 = phase(scores, maximum, row_sum, o0, o1, o2, o3, page1, page2, tiles - 1, 1, 0)

    maximum, row_sum = fx.Float32(maximum), fx.Float32(row_sum)
    wait(vmcnt=K_CHUNKS)
    _stage_end()
    scores = _exp_slice(scores, 0, 32)
    row_sum = row_sum + _row_sum(scores)
    p = _pin(scores.to(fx.BFloat16))
    _stage_end()
    last_slot = (tiles - 1) & 1
    v = _read_v(storage, 2 * K_TILE + last_slot * V_TILE + rv)
    wait(lgkmcnt=0)
    rocdl.sched_barrier(0)
    v.store(_mask_v_tail(v.load(), kv_len - data_tile(last) * BN))
    _stage_end()
    o0, o1 = _pv(p, v, o0, o1, visible_wave_tile(last))
    _stage_end()
    v = _read_v(storage, 2 * K_TILE + last_slot * V_TILE + 520 + rv)
    wait(lgkmcnt=0)
    rocdl.sched_barrier(0)
    v.store(_mask_v_tail(v.load(), kv_len - data_tile(last) * BN))
    _stage_end()
    o2, o3 = _pv(p, v, o2, o3, visible_wave_tile(last))
    rocdl.sched_barrier(0)
    if fx.const_expr(not STAGGER):
        rocdl.s_barrier()

    out_tid = fx.Int32(llvm.inline_asm(fx.Int32.ir_type, [tid.ir_value()], "", "=v,0", has_side_effects=True))
    out_row = (out_tid >> 6) * 32 + (out_tid & 31)
    o_ptr = fx.get_iter(O) + ((fx.Int64(q0) + fx.Int64(q_start)) * OROW + fx.Int64(head) * OHEAD)
    o_buf = rocdl.make_buffer_tensor(fx.make_view(o_ptr, fx.make_layout((BM, DV), (OROW, 1))), num_records_bytes=q_valid * OROW * 2)
    out_atom = fx.make_copy_atom(rocdl.BufferCopy128b(), fx.BFloat16)
    inv = (row_sum > 0.0).select(fx.Float32(1.0) / row_sum, fx.Float32(0.0)) * VS[0]
    outputs = (o0, o1, o2, o3)
    pair_type = ir.Type.parse("!llvm.struct<(i32, i32)>")
    for g in fx.range_constexpr(8):
        vals = outputs[g // 2]
        begin = (g & 1) * 8
        packed = [rocdl.cvt_pk_bf16_f32(vals[begin + i] * inv, vals[begin + i + 1] * inv) for i in range(0, 8, 2)]
        pairs = [rocdl.permlane32_swap(pair_type, packed[i], packed[i + 2], False, True) for i in range(2)]
        words = fx.Vector.from_elements([
            fx.Int32(llvm.extractvalue(fx.Int32.ir_type, pairs[i & 1], [i // 2])) for i in range(4)
        ], fx.Int32)
        src = rmem(fx.make_layout(8, 1), fx.BFloat16)
        src.store(words.bitcast(fx.BFloat16))
        offset = out_row * OROW + ((out_tid >> 5) & 1) * 8 + g * 16
        dst = fx.make_view(fx.get_iter(o_buf) + offset, fx.make_layout(8, 1))
        fx.copy(out_atom, src, dst)
    if fx.const_expr(WITH_LSE):
        if (out_tid & 63) < 32:
            if out_row < q_valid:
                log_l = fx.Float32(llvm.call_intrinsic(fx.Float32.ir_type, "llvm.log2.f32", [row_sum.ir_value()], [], []))
                lse = (row_sum > 0.0).select((maximum + log_l) * fx.Float32(math.log(2.0)), fx.Float32(float("-inf")))
                LSE[(q0 + q_start + out_row) * H + head] = lse


@flyc.jit
def _attention_sequence(
    Q, K, V, O, LSE, QS, KS, VS, SINK, TABLE, storage, q0, q_len, kv_len, batch, head, qb,
    H: fx.Constexpr[int], HK: fx.Constexpr[int], NP: fx.Constexpr[int],
    DQ: fx.Constexpr[int], WINDOW: fx.Constexpr[int], HAS_SINK: fx.Constexpr[bool],
    QROW: fx.Constexpr[int], QHEAD: fx.Constexpr[int], OROW: fx.Constexpr[int], OHEAD: fx.Constexpr[int],
    PER_TOKEN: fx.Constexpr[bool], CAUSAL: fx.Constexpr[bool], WITH_LSE: fx.Constexpr[bool],
    SCALE: fx.Constexpr[float], STAGGER: fx.Constexpr[bool], MERGE: fx.Constexpr[bool],
    PRUNE_SWA: fx.Constexpr[bool] = False,
):
    _attention_body(Q, K, V, O, LSE, QS, KS, VS, SINK, TABLE, storage, q0, q_len, kv_len, batch, head, qb,
                   H, HK, NP, DQ, WINDOW, HAS_SINK, QROW, QHEAD, OROW, OHEAD, PER_TOKEN, CAUSAL, WITH_LSE, SCALE, STAGGER,
                   False, MERGE, PRUNE_SWA)
    if fx.const_expr(MERGE):
        mirror = (q_len + BM - 1) // BM - 1 - qb
        if mirror > qb:
            # OPUS pairs the shortest and longest causal query blocks. The
            # mirror scans backward so both passes balance work and reuse L2.
            _attention_body(Q, K, V, O, LSE, QS, KS, VS, SINK, TABLE, storage, q0, q_len, kv_len, batch, head, mirror,
                           H, HK, NP, DQ, WINDOW, HAS_SINK, QROW, QHEAD, OROW, OHEAD, PER_TOKEN, CAUSAL, WITH_LSE, SCALE, STAGGER,
                           True, MERGE, PRUNE_SWA)


@flyc.jit
def _zero_tile(O, LSE, SINK, q0, q_len, head, qb,
               H: fx.Constexpr[int], OROW: fx.Constexpr[int], OHEAD: fx.Constexpr[int],
               WITH_LSE: fx.Constexpr[bool], HAS_SINK: fx.Constexpr[bool]):
    tid = fx.Int32(gpu.thread_id("x"))
    valid = _min(q_len - qb * BM, fx.Int32(BM))
    ptr = fx.get_iter(O) + ((fx.Int64(q0) + fx.Int64(qb) * BM) * OROW + fx.Int64(head) * OHEAD)
    buf = rocdl.make_buffer_tensor(
        fx.make_view(ptr, fx.make_layout((BM, DV), (OROW, 1))),
        num_records_bytes=valid * OROW * 2,
    )
    zeros = rmem(fx.make_layout(8, 1), fx.BFloat16)
    zeros.fill(0.0)
    atom = fx.make_copy_atom(rocdl.BufferCopy128b(), fx.BFloat16)
    for i in fx.range_constexpr(8):
        index = tid * 8 + i * THREADS * 8
        offset = (index // DV) * OROW + index % DV
        dst = fx.make_view(fx.get_iter(buf) + offset, fx.make_layout(8, 1))
        fx.copy(atom, zeros, dst)
    if fx.const_expr(WITH_LSE):
        if (tid < BM) & (qb * BM + tid < q_len):
            LSE[(q0 + qb * BM + tid) * H + head] = SINK[head] if HAS_SINK else fx.Float32(float("-inf"))


def _attention_storage(allocator, dq):
    elements = 2 * (8 * (dq // 64) * 520) + max(32 * (dq // 64) * 520, 2 * V_TILE)
    return allocator.allocate(fx.Array[fx.BFloat16, elements, 16]).peek().view(fx.make_layout(elements, 1))


@flyc.jit
def _attention_task(Q, K, V, O, LSE, CQ, KI, PAGES, LAST, QS, KS, VS, SINK, storage, head, batch, qb,
    H: fx.Constexpr[int], HK: fx.Constexpr[int], NP: fx.Constexpr[int],
    DQ: fx.Constexpr[int], WINDOW: fx.Constexpr[int], HAS_SINK: fx.Constexpr[bool],
    QROW: fx.Constexpr[int], QHEAD: fx.Constexpr[int], OROW: fx.Constexpr[int], OHEAD: fx.Constexpr[int],
    PER_TOKEN: fx.Constexpr[bool], CAUSAL: fx.Constexpr[bool], WITH_LSE: fx.Constexpr[bool],
    SCALE: fx.Constexpr[float], MERGE: fx.Constexpr[bool], PRUNE_SWA: fx.Constexpr[bool] = False):
    q0 = _uniform(CQ[batch])
    q_len = _uniform(CQ[batch + 1]) - q0
    page_start = _uniform(KI[batch])
    pages = _uniform(KI[batch + 1]) - page_start
    table = fx.make_view(fx.get_iter(PAGES) + fx.Int64(page_start), fx.make_layout(pages, 1))
    kv_len = (pages > 0).select((pages - 1) * 64 + _uniform(LAST[batch]), fx.Int32(0))
    q_blocks = (q_len + BM - 1) // BM
    work_blocks = (q_blocks + 1) // 2 if MERGE else q_blocks
    if qb < work_blocks:
        if kv_len > 0:
            group = _uniform(fx.Int32(gpu.thread_id("x")) >> 8)
            if group != 0:
                _attention_sequence(Q, K, V, O, LSE, QS, KS, VS, SINK, table, storage, q0, q_len, kv_len, batch, head, qb, H, HK, NP, DQ, WINDOW, HAS_SINK, QROW, QHEAD, OROW, OHEAD, PER_TOKEN, CAUSAL, WITH_LSE, SCALE, True, MERGE, PRUNE_SWA)
            else:
                _attention_sequence(Q, K, V, O, LSE, QS, KS, VS, SINK, table, storage, q0, q_len, kv_len, batch, head, qb, H, HK, NP, DQ, WINDOW, HAS_SINK, QROW, QHEAD, OROW, OHEAD, PER_TOKEN, CAUSAL, WITH_LSE, SCALE, False, MERGE, PRUNE_SWA)
        else:
            _zero_tile(O, LSE, SINK, q0, q_len, head, qb, H, OROW, OHEAD, WITH_LSE, HAS_SINK)
            if fx.const_expr(MERGE):
                mirror = q_blocks - 1 - qb
                if mirror > qb:
                    _zero_tile(O, LSE, SINK, q0, q_len, head, mirror, H, OROW, OHEAD, WITH_LSE, HAS_SINK)


@flyc.kernel(known_block_size=[THREADS, 1, 1])
def _attention_kernel(Q: fx.Tensor, K: fx.Tensor, V: fx.Tensor, O: fx.Tensor, LSE: fx.Tensor,
    CQ: fx.Tensor, KI: fx.Tensor, PAGES: fx.Tensor, LAST: fx.Tensor, QS: fx.Tensor, KS: fx.Tensor, VS: fx.Tensor, SINK: fx.Tensor,
    H: fx.Constexpr[int], HK: fx.Constexpr[int], NP: fx.Constexpr[int],
    DQ: fx.Constexpr[int], WINDOW: fx.Constexpr[int], HAS_SINK: fx.Constexpr[bool],
    QROW: fx.Constexpr[int], QHEAD: fx.Constexpr[int], OROW: fx.Constexpr[int], OHEAD: fx.Constexpr[int],
    PER_TOKEN: fx.Constexpr[bool], CAUSAL: fx.Constexpr[bool], WITH_LSE: fx.Constexpr[bool],
    SCALE: fx.Constexpr[float], MERGE: fx.Constexpr[bool], PRUNE_SWA: fx.Constexpr[bool]):
    storage = _attention_storage(fx.SharedAllocator(), DQ)
    _attention_task(Q, K, V, O, LSE, CQ, KI, PAGES, LAST, QS, KS, VS, SINK, storage,
        fx.Int32(gpu.block_id("x")), fx.Int32(gpu.block_id("y")), fx.Int32(gpu.block_id("z")),
        H, HK, NP, DQ, WINDOW, HAS_SINK, QROW, QHEAD, OROW, OHEAD, PER_TOKEN, CAUSAL, WITH_LSE, SCALE, MERGE, PRUNE_SWA)


def _atomic_increment(counter, index):
    # Agent-scoped atomic ticket; the stream orders complete kernel launches.
    pointer = fx.to_llvm_ptr(fx.get_iter(counter) + fx.Int64(index))
    return fx.Int32(llvm.AtomicRMWOp(
        llvm.AtomicBinOp.add, as_ir_value(pointer), fx.Int32(1).ir_value(),
        llvm.AtomicOrdering.monotonic, syncscope="agent", alignment=4,
    ).result)


@flyc.jit
def _next_ticket(counter, mailbox, tid):
    if tid == 0:
        mailbox[0] = _atomic_increment(counter, 0)
        wait(lgkmcnt=0)
    # The pipeline may leave speculative K DMA outstanding at its epilogue.
    # Drain every wave before a new task reuses the same Q/K/V LDS addresses.
    wait(vmcnt=0, lgkmcnt=0)
    # Broadcast the ticket AND close all waves' previous Q/V LDS lifetimes.
    _stage_end()
    ticket = mailbox[0]
    wait(lgkmcnt=0)
    return _uniform(ticket)


@flyc.jit
def _finish_scheduler(counter, tid):
    if tid == 0:
        finished = _atomic_increment(counter, 1)
        if finished == fx.Int32(gpu.grid_dim.x) - 1:
            counter[0] = fx.Int32(gpu.grid_dim.x)
            counter[1] = fx.Int32(0)


@flyc.kernel(known_block_size=[THREADS, 1, 1])
def _attention_persistent_kernel(Q: fx.Tensor, K: fx.Tensor, V: fx.Tensor, O: fx.Tensor, LSE: fx.Tensor,
    CQ: fx.Tensor, KI: fx.Tensor, PAGES: fx.Tensor, LAST: fx.Tensor, QS: fx.Tensor, KS: fx.Tensor, VS: fx.Tensor, SINK: fx.Tensor,
    COUNTER: fx.Tensor, H: fx.Constexpr[int], HK: fx.Constexpr[int], NP: fx.Constexpr[int], B: fx.Constexpr[int],
    DQ: fx.Constexpr[int], WINDOW: fx.Constexpr[int], HAS_SINK: fx.Constexpr[bool],
    QROW: fx.Constexpr[int], QHEAD: fx.Constexpr[int], OROW: fx.Constexpr[int], OHEAD: fx.Constexpr[int],
    PER_TOKEN: fx.Constexpr[bool], CAUSAL: fx.Constexpr[bool], WITH_LSE: fx.Constexpr[bool],
    SCALE: fx.Constexpr[float], MERGE: fx.Constexpr[bool], PRUNE_SWA: fx.Constexpr[bool]):
    allocator = fx.SharedAllocator()
    storage = _attention_storage(allocator, DQ)
    mailbox = allocator.allocate(fx.Array[fx.Int32, 1, 4]).peek()
    tid = fx.Int32(gpu.thread_id("x"))

    def batch_tasks(batch):
        q_len = _uniform(CQ[batch + 1]) - _uniform(CQ[batch])
        blocks = (q_len + BM - 1) // BM
        return ((blocks + 1) // 2 if MERGE else blocks) * H

    @flyc.jit
    def locate(ticket, batch, begin, end):
        # Tickets only increase per CTA. Skip complete/empty batches, not
        # individual heads or max-Q padding, and read current device metadata.
        while (batch < B) & (ticket >= end):
            begin = end
            batch += 1
            if batch < B:
                end += batch_tasks(batch)
        return batch, begin, end

    ticket = fx.Int32(gpu.block_id("x"))
    batch, begin, end = locate(ticket, fx.Int32(0), fx.Int32(0), batch_tasks(fx.Int32(0)))
    while batch < B:
        local = ticket - begin
        _attention_task(Q, K, V, O, LSE, CQ, KI, PAGES, LAST, QS, KS, VS, SINK, storage,
            local % H, batch, local // H,
            H, HK, NP, DQ, WINDOW, HAS_SINK, QROW, QHEAD, OROW, OHEAD, PER_TOKEN, CAUSAL, WITH_LSE, SCALE, MERGE, PRUNE_SWA)
        ticket = _next_ticket(COUNTER, mailbox, tid)
        batch, begin, end = locate(ticket, batch, begin, end)
    _finish_scheduler(COUNTER, tid)


@flyc.jit
def _launch_attention(Q: fx.Tensor, K: fx.Tensor, V: fx.Tensor, O: fx.Tensor, LSE: fx.Tensor,
    CQ: fx.Tensor, KI: fx.Tensor, PAGES: fx.Tensor, LAST: fx.Tensor, QS: fx.Tensor, KS: fx.Tensor, VS: fx.Tensor, SINK: fx.Tensor,
    H: fx.Constexpr[int], HK: fx.Constexpr[int], NP: fx.Constexpr[int], B: fx.Constexpr[int], MAX_Q: fx.Constexpr[int],
    DQ: fx.Constexpr[int], WINDOW: fx.Constexpr[int], HAS_SINK: fx.Constexpr[bool],
    QROW: fx.Constexpr[int], QHEAD: fx.Constexpr[int], OROW: fx.Constexpr[int], OHEAD: fx.Constexpr[int],
    PER_TOKEN: fx.Constexpr[bool], CAUSAL: fx.Constexpr[bool], WITH_LSE: fx.Constexpr[bool], SCALE: fx.Constexpr[float], stream: fx.Stream):
    query_blocks = (MAX_Q + BM - 1) // BM
    merge = CAUSAL and WINDOW < 0 and query_blocks * H * B >= 512
    # Gating regresses small grids / wide windows. Keep their original ISA;
    # enable only the large, narrow-window regimes validated by paired sweeps.
    prune = B == 1 and query_blocks * H >= 1024 and 0 <= WINDOW <= (128 if DQ == 192 else 64)
    _attention_kernel(Q, K, V, O, LSE, CQ, KI, PAGES, LAST, QS, KS, VS, SINK, H, HK, NP, DQ, WINDOW, HAS_SINK, QROW, QHEAD, OROW, OHEAD,
        PER_TOKEN, CAUSAL, WITH_LSE, SCALE, merge, prune,
        value_attrs={"rocdl.waves_per_eu": 2},
    ).launch(grid=(H, B, (query_blocks + 1) // 2 if merge else query_blocks), block=(THREADS, 1, 1), stream=stream)


@flyc.jit
def _launch_persistent(Q: fx.Tensor, K: fx.Tensor, V: fx.Tensor, O: fx.Tensor, LSE: fx.Tensor,
    CQ: fx.Tensor, KI: fx.Tensor, PAGES: fx.Tensor, LAST: fx.Tensor, QS: fx.Tensor, KS: fx.Tensor, VS: fx.Tensor, SINK: fx.Tensor,
    H: fx.Constexpr[int], HK: fx.Constexpr[int], NP: fx.Constexpr[int], B: fx.Constexpr[int], MAX_Q: fx.Constexpr[int],
    DQ: fx.Constexpr[int], WINDOW: fx.Constexpr[int], HAS_SINK: fx.Constexpr[bool],
    QROW: fx.Constexpr[int], QHEAD: fx.Constexpr[int], OROW: fx.Constexpr[int], OHEAD: fx.Constexpr[int],
    PER_TOKEN: fx.Constexpr[bool], CAUSAL: fx.Constexpr[bool], WITH_LSE: fx.Constexpr[bool], SCALE: fx.Constexpr[float],
    COUNTER: fx.Tensor, GRID: fx.Constexpr[int], stream: fx.Stream):
    query_blocks = (MAX_Q + BM - 1) // BM
    merge = CAUSAL and WINDOW < 0 and query_blocks * H * B >= 512
    prune = B == 1 and query_blocks * H >= 1024 and 0 <= WINDOW <= (128 if DQ == 192 else 64)
    _attention_persistent_kernel(Q, K, V, O, LSE, CQ, KI, PAGES, LAST, QS, KS, VS, SINK, COUNTER,
        H, HK, NP, B, DQ, WINDOW, HAS_SINK, QROW, QHEAD, OROW, OHEAD, PER_TOKEN, CAUSAL, WITH_LSE, SCALE, merge, prune,
        value_attrs={"rocdl.waves_per_eu": 2},
    ).launch(grid=(GRID, 1, 1), block=(THREADS, 1, 1), stream=stream)


def _flat(tensor):
    """View storage, preserving the real Q/O row and head strides separately."""
    if tensor.numel() == 0:
        return tensor.reshape(-1)
    extent = 1 + sum((n - 1) * s for n, s in zip(tensor.shape, tensor.stride()))
    return tensor.as_strided((extent,), (1,))


class _PagedAttention:
    bf16_backend = "native-8wave"

    def __init__(self, heads, kv_heads, dq, causal, quant_query_mode, window_left, has_sink, persistent):
        self.heads, self.kv_heads, self.causal = heads, kv_heads, causal
        self.dq, self.window_left, self.has_sink = dq, window_left, has_sink
        self.quant_query_mode = quant_query_mode
        self.persistent = persistent
        self._scheduler_counters = {}
        self._compiled = {}

    def _scheduler(self, device, stream, batch, max_q):
        blocks = (max_q + BM - 1) // BM
        merge = self.causal and self.window_left < 0 and blocks * self.heads * batch >= 512
        tasks = ((blocks + 1) // 2 if merge else blocks) * self.heads * batch
        grid = min(torch.cuda.get_device_properties(device).multi_processor_count, tasks)
        if tasks + grid >= 2**31:
            raise ValueError("persistent work tickets must fit signed int32")
        key = (device, stream.cuda_stream, grid)
        counter = self._scheduler_counters.get(key)
        if counter is None:
            with torch.cuda.stream(stream):
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError("warm persistent attention on this stream before graph capture")
                counter = torch.tensor([grid, 0], device=device, dtype=torch.int32)
            self._scheduler_counters[key] = counter
        return counter, grid

    def _validate_sink(self, sink_ptr, device):
        if self.has_sink:
            if (sink_ptr is None or sink_ptr.shape != (self.heads,) or sink_ptr.dtype != torch.float32
                    or sink_ptr.device != device or not sink_ptr.is_contiguous()):
                raise ValueError("sink_ptr must be contiguous FP32 [query heads] on the input GPU")
        elif sink_ptr is not None:
            raise ValueError("sink_ptr requires has_sink=True")

    def _run(self, fn, args):
        signature = tuple((x.dtype, tuple(x.shape), tuple(x.stride())) if isinstance(x, torch.Tensor)
                          else ("stream",) if hasattr(x, "cuda_stream") else x for x in args)
        key = (fn, args[0].device, signature)
        compiled = self._compiled.get(key)
        with torch.cuda.device(args[0].device):
            if compiled is None:
                # flyc.compile traces, compiles AND performs the first launch.
                self._compiled[key] = flyc.compile(fn, *args)
            else:
                compiled(*args)

    def __call__(self, Q, K, V, cu_seqlens_q, cu_seqlens_k, kv_indptr, kv_page_indices,
                 max_seqlen_q, max_seqlen_k, causal, q_descale, k_descale, v_descale,
                 kv_last_page_lens, out=None, sink_ptr=None, stream=None, *, return_lse=False, lse=None, softmax_scale=None):
        """Run paged attention; descales must be finite and strictly positive.

        A sink is an unscaled natural-logit, per query head, for a virtual key
        with value zero. Finite sink logits and -inf (disabled) are supported.
        GPU metadata values (page IDs, prefix sums, last-page lengths and the
        supplied maximum lengths) must be consistent. They are not copied to
        the CPU, so the hot path remains asynchronous and graph-capturable.
        Persistent mode caches an 8-byte scheduler header per device/stream/grid;
        warm each specialization on its capture stream before recording a graph.
        Graphs retain that header: overlapping replays must originate from
        independently warmed capture streams, not a single shared header.
        """
        if not Q.is_cuda or K.device != Q.device or V.device != Q.device:
            raise ValueError("Q/K/V must be on the same GPU")
        if "gfx950" not in torch.cuda.get_device_properties(Q.device).gcnArchName:
            raise NotImplementedError("this kernel requires gfx950")
        if Q.dtype != torch.bfloat16 or K.dtype != Q.dtype or V.dtype != Q.dtype:
            raise NotImplementedError("only BF16 Q/K/V are supported")
        if causal != self.causal:
            raise ValueError("causal must match the factory")
        self._validate_sink(sink_ptr, Q.device)
        if Q.ndim != 3 or Q.shape[1:] != (self.heads, self.dq) or Q.stride(-1) != 1:
            raise ValueError("Q must be [tokens, heads, head_dim_qk] with contiguous head dimension")
        if K.ndim != 5 or V.ndim != 5 or K.shape != (V.shape[0], self.kv_heads, self.dq // 8, 64, 8) or V.shape[1:] != (self.kv_heads, 8, 128, 8):
            raise ValueError("K/V must use page64 SHUFFLE-5D layouts")
        if not K.is_contiguous() or not V.is_contiguous():
            raise ValueError("paged K/V storage must be contiguous")
        if max_seqlen_q < 0 or max_seqlen_k < 0:
            raise ValueError("sequence-length bounds must be nonnegative")
        if K.numel() * K.element_size() >= 2**31:
            raise NotImplementedError("physical KV cache must fit the signed 32-bit buffer offset")
        if softmax_scale is not None and (not math.isfinite(softmax_scale) or softmax_scale <= 0):
            raise ValueError("softmax_scale must be finite and positive")
        batch = cu_seqlens_q.numel() - 1
        if batch < 1 or kv_indptr.numel() != batch + 1 or kv_last_page_lens.numel() != batch:
            raise ValueError("inconsistent batch metadata")
        metadata = (cu_seqlens_q, kv_indptr, kv_page_indices, kv_last_page_lens)
        if cu_seqlens_k is not None:
            if cu_seqlens_k.numel() != batch + 1:
                raise ValueError("inconsistent KV prefix length")
            metadata += (cu_seqlens_k,)
        if any(t.ndim != 1 or t.dtype != torch.int32 or t.device != Q.device or not t.is_contiguous() for t in metadata):
            raise ValueError("metadata must be contiguous device int32 tensors")
        if k_descale.numel() != 1 or v_descale.numel() != 1 or q_descale.numel() not in (1, Q.shape[0] * self.heads):
            raise ValueError("expected scalar K/V descales and scalar or per-token/head Q descales")
        if any(t.dtype != torch.float32 or t.device != Q.device or not t.is_contiguous() for t in (q_descale, k_descale, v_descale)):
            raise ValueError("descales must be contiguous device FP32 tensors")
        stream = torch.cuda.current_stream(Q.device) if stream is None else stream
        if stream.device != Q.device:
            raise ValueError("stream must belong to the input GPU")
        with torch.cuda.stream(stream):
            if out is None:
                out = torch.empty((Q.shape[0], self.heads, DV), dtype=torch.bfloat16, device=Q.device)
            if return_lse and lse is None:
                lse = torch.empty((Q.shape[0], self.heads), dtype=torch.float32, device=Q.device)
        if out.shape != (Q.shape[0], self.heads, DV) or out.dtype != torch.bfloat16 or out.device != Q.device or out.stride(-1) != 1:
            raise ValueError("invalid output buffer")
        if lse is not None and (lse.shape != (Q.shape[0], self.heads) or lse.dtype != torch.float32 or lse.device != Q.device or not lse.is_contiguous()):
            raise ValueError("LSE must be contiguous FP32 [tokens, heads]")
        if Q.shape[0] > 0 and max_seqlen_q > 0:
            args = (
                _flat(Q), _flat(K), _flat(V), _flat(out), _flat(lse) if lse is not None else k_descale.reshape(-1),
                cu_seqlens_q, kv_indptr, kv_page_indices, kv_last_page_lens,
                _flat(q_descale), k_descale.reshape(-1), v_descale.reshape(-1),
                sink_ptr if sink_ptr is not None else k_descale.reshape(-1),
                self.heads, self.kv_heads, K.shape[0], batch, max_seqlen_q,
                self.dq, self.window_left, self.has_sink,
                Q.stride(0), Q.stride(1), out.stride(0), out.stride(1),
                q_descale.numel() != 1, self.causal, lse is not None,
                float(1 / math.sqrt(self.dq) if softmax_scale is None else softmax_scale),
            )
            if self.persistent:
                counter, grid = self._scheduler(Q.device, stream, batch, max_seqlen_q)
                self._run(_launch_persistent, (*args, counter, grid, stream))
            else:
                self._run(_launch_attention, (*args, stream))
        return (out, lse) if return_lse else out


@functools.cache
def PagedAttention(num_qo_heads, num_kv_heads, head_dim_qk, head_dim_v, page_size,
                   is_causal, quant_query_mode="per-token", key_layout="vectorized",
                   window_left=-1, has_sink=False, *, memory_mode="lds", persistent=None):
    """Return native gfx950 BF16 attention, with optional SWA, sinks and LSE.

    Only memory_mode='lds' is supported. persistent=None selects False, so
    default calls allocate no scheduler state or auxiliary GPU dispatch.
    persistent=True retains the cached 8-byte per-stream scheduler header.
    """
    if memory_mode != "lds":
        raise NotImplementedError("gfx950 BF16 attention supports memory_mode='lds' only")
    if head_dim_qk not in (128, 192) or (head_dim_v, page_size, key_layout) != (DV, BN, "vectorized"):
        raise NotImplementedError("only gfx950 BF16 D128/D192, V128 page64 attention is supported")
    if not isinstance(window_left, int) or window_left < -1 or window_left >= 2**31:
        raise ValueError("window_left must be -1 or a nonnegative signed 32-bit integer")
    if window_left >= 0 and not is_causal:
        raise ValueError("SWA requires bottom-right causal attention")
    if num_kv_heads <= 0 or num_qo_heads <= 0 or num_qo_heads % num_kv_heads:
        raise ValueError("query heads must be a positive multiple of KV heads")
    if quant_query_mode not in ("per-token", "per-tensor"):
        raise ValueError("unsupported query scale mode")
    persistent = False if persistent is None else persistent
    if not isinstance(persistent, bool):
        raise ValueError("persistent must be a bool or None")
    return _PagedAttention(num_qo_heads, num_kv_heads, head_dim_qk, is_causal, quant_query_mode, window_left, has_sink, persistent)