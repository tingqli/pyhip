"""gfx942 FP8-FNUZ paged attention: BM256/BN64, eight staggered wave64s.

K/V use the page64 SHUFFLE-5D ABI. ``memory_mode='lds'`` stages cooperative
buffer loads through packed VGPRs into 41,600 bytes of LDS for D192
(33,280 for D128). Only LDS mode is supported; it uses no direct-LDS DMA,
LDS-transpose instructions, or native FP32-to-BF16 conversion.
"""

import functools
import math

import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu, rocdl
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm


BM, BN, DV, THREADS = 256, 64, 128, 512
PADDED_CHUNK = 1040
V_TILE = 8 * PADDED_CHUNK
LOG2E = math.log2(math.e)


def _uniform(value):
    return fx.Int32(rocdl.readfirstlane(fx.Int32.ir_type, fx.Int32(value).ir_value()))


def _min(a, b):
    return (a < b).select(a, b)


def _pin_i32(value):
    return fx.Int32(llvm.inline_asm(fx.Int32.ir_type, [fx.Int32(value).ir_value()],
                                  "", "=v,0", has_side_effects=True))


def _pin(values, chunk=8):
    values = fx.Vector(values)
    dtype = values.dtype
    words = values.bitcast(fx.Int32) if dtype.width < 32 else values
    result = []
    for start in range(0, words.numel, chunk):
        part = fx.Vector.from_elements([words[i] for i in range(start, min(start + chunk, words.numel))], words.dtype)
        tied = fx.Vector(llvm.inline_asm(part.ir_value().type, [part.ir_value()], "", "=v,0", has_side_effects=True))
        result.extend(tied[i] for i in range(tied.numel))
    return fx.Vector.from_elements(result, words.dtype).bitcast(dtype)


def _join(a, b):
    return fx.Vector.from_elements([a[i] for i in range(16)] + [b[i] for i in range(16)], fx.Float32)


def _exp(value):
    return fx.Float32(llvm.call_intrinsic(fx.Float32.ir_type, "llvm.amdgcn.exp2.f32",
                                         [fx.Float32(value).ir_value()], [], []))


def _maximum(a, b):
    return fx.Float32(llvm.inline_asm(fx.Float32.ir_type, [fx.Float32(a).ir_value(), fx.Float32(b).ir_value()],
                                    "v_max_f32 $0, $1, $2", "=v,v,v", has_side_effects=False))


def _max3(a, b, c):
    return fx.Float32(llvm.inline_asm(fx.Float32.ir_type,
        [a.ir_value(), b.ir_value(), c.ir_value()], "v_max3_f32 $0, $1, $2, $3", "=v,v,v,v", has_side_effects=False))


def _row_max(values):
    # Independent MAX3 chains avoid the 32-deep VALU dependency chain.
    p = [_max3(values[i], values[i + 1], values[i + 2]) for i in range(0, 30, 3)]
    a = _max3(_max3(p[0], p[1], p[2]), _max3(p[3], p[4], p[5]), _max3(p[6], p[7], p[8]))
    b = _max3(p[9], values[30], values[31])
    value = _max3(a, b, fx.Float32(-1.0e30))
    return _maximum(value, value.shuffle_xor(32, 64))


def _row_sum(values):
    partials = [values[i] + values[i + 1] for i in range(0, 32, 2)]
    for width in (8, 4, 2, 1):
        partials = [partials[2 * i] + partials[2 * i + 1] for i in range(width)]
    value = partials[0]
    return value + value.shuffle_xor(32, 64)


def _center(values, scale, maximum, begin, end):
    result = []
    for i in range(32):
        value = values[i]
        if begin <= i < end:
            value = fx.Float32(llvm.inline_asm(
                fx.Float32.ir_type, [value.ir_value(), scale.ir_value(), maximum.ir_value()],
                "v_fma_f32 $0, $1, $2, -$3", "=v,v,v,v", has_side_effects=False))
        result.append(value)
    return _pin(fx.Vector.from_elements(result, fx.Float32), 32)


def _exp_part(values, begin, end):
    return fx.Vector.from_elements([_exp(values[i]) if begin <= i < end else values[i] for i in range(32)], fx.Float32)


def _pack_probability(values, dtype):
    words = []
    for i in range(0, 32, 4):
        lo = rocdl.cvt_pk_fp8_f32(fx.Int32.ir_type, values[i], values[i + 1], fx.Int32(0), False)
        hi = rocdl.cvt_pk_fp8_f32(fx.Int32.ir_type, values[i + 2], values[i + 3], lo, True)
        words.append(fx.Int32(hi))
    return _pin(fx.Vector.from_elements(words, fx.Int32).bitcast(dtype))


def _pack_bf16(values):
    # Software round-to-nearest-even, following the old gfx942 bit-pack path
    # but including the retained-bit tie correction (not round-half-up).
    bits = values.bitcast(fx.Uint32)
    rounded = bits + fx.Uint32(0x7FFF) + ((bits >> 16) & fx.Uint32(1))
    words = [(rounded[i] >> 16) | (rounded[i + 1] & fx.Uint32(0xFFFF0000))
             for i in range(0, values.numel, 2)]
    return fx.Vector.from_elements(words, fx.Uint32)


def _stage_end():
    # Compiler scheduling fences + CTA rendezvous, NOT a memory-counter wait.
    # Explicit s_waitcnt at each producer/consumer boundary is still required.
    rocdl.sched_barrier(0)
    rocdl.s_barrier()
    rocdl.sched_barrier(0)


def _wait(*, vmcnt=63, expcnt=7, lgkmcnt=63):
    # gfx942 immediate encoding; also supported by FlyDSL 0.2.2, whose
    # s_waitcnt wrapper accepts only a bitfield rather than keyword counts.
    # Counts are per-wave outstanding operations, not cycles or byte counts:
    # vmcnt -> vector memory; lgkmcnt -> LDS/SMEM (including ds_bpermute).
    # A threshold N waits until the counter is <= N; omitted counters keep
    # their maximum/no-wait value. This neither waits for MFMA nor syncs a CTA.
    rocdl.s_waitcnt((vmcnt & 15) | (expcnt << 4) | (lgkmcnt << 8) | ((vmcnt >> 4) << 14))


def _schedule(pairs, count, group, exp=False):
    for _ in range(pairs):
        rocdl.sched_group_barrier(rocdl.mask_mfma, 1, group)
        rocdl.sched_group_barrier(0x400 if exp else 0x002, count, group)


def _prefetch_page(table, index):
    address = fx.Int64(fx.ptrtoint(fx.get_iter(table) + index))
    return fx.Int32(llvm.inline_asm(fx.Int32.ir_type, [address.ir_value()],
                                  "s_load_dword $0, $1, 0", "=s,s,~{memory}", has_side_effects=True))


def _page_ready(value):
    # s_waitcnt lgkmcnt(0): make s_load_dword's page ID usable. Also drains
    # this wave's earlier LDS operations; deliberately does NOT wait for VMEM.
    return fx.Int32(llvm.inline_asm(fx.Int32.ir_type, [value.ir_value()],
                                  "s_waitcnt lgkmcnt(0)", "=s,0", has_side_effects=True))


def _buffer(tensor, elements):
    # Build the same bounded raw descriptor on FlyDSL 0.2.2 and 0.3.x;
    # get_buffer_rsrc is not exported by the older native gfx942 environment.
    address = fx.Int64(fx.ptrtoint(fx.get_iter(tensor)))
    pointer = llvm.inttoptr(ir.Type.parse("!llvm.ptr"), address.ir_value())
    return rocdl.make_buffer_rsrc(ir.Type.parse("!llvm.ptr<8>"), pointer,
                                 fx.Int16(0).ir_value(), fx.Int64(elements).ir_value(),
                                 fx.Int32(0x27000).ir_value())


def _global_words(resource, voffset, soffset, count=4):
    return fx.Vector(rocdl.raw_ptr_buffer_load(ir.VectorType.get([count], fx.Int32.ir_type),
                                              resource, fx.Int32(voffset).ir_value(),
                                              fx.Int32(soffset).ir_value(), fx.Int32(0).ir_value()))


def _lds_words(storage, base, immediate=0):
    # Async inline LDS read: the caller must s_waitcnt lgkmcnt(0) before use.
    address = fx.Int32(fx.ptrtoint(fx.get_iter(storage) + base))
    return fx.Vector(llvm.inline_asm(ir.VectorType.get([4], fx.Int32.ir_type), [address.ir_value()],
        f"ds_read_b128 $0, $1 offset:{immediate}", "=v,v,~{memory}", has_side_effects=True))


def _cooperative_load(resource, tid, physical_offset, rounds):
    parts = [_global_words(resource, (tid + i * THREADS) * 8, physical_offset, 2) for i in range(rounds)]
    return fx.Vector.from_elements([part[j] for part in parts for j in range(2)], fx.Int32)


def _cooperative_store(storage, tid, words, slot_offset, rounds, key=False):
    atom = fx.make_copy_atom(fx.UniversalCopy64b(), fx.Int32)
    for i in range(rounds):
        byte = (tid + i * THREADS) * 8
        chunk, within = byte // 1024, byte % 1024
        # XOR row bit4 into bank bit2 for gfx942's b128 service groups
        # (lanes0..3+20..23, etc.). gfx950's grouping is different.
        if key:
            within = within ^ ((within & 256) >> 2)
        offset = slot_offset + chunk * PADDED_CHUNK + within
        pointer = fx.recast_iter(fx.PointerType.get(fx.Int32.ir_type, storage.memspace, 8),
                     fx.get_iter(storage) + offset)
        dst = fx.make_view(pointer, fx.make_layout(2, 1))
        src = fx.make_rmem_tensor(2, fx.Int32)
        src.store(fx.Vector.from_elements([words[2 * i], words[2 * i + 1]], fx.Int32))
        fx.copy(atom, src, dst)


def _mma(dtype):
    return fx.make_tiled_mma(fx.make_mma_atom(rocdl.MFMA(32, 32, 16, dtype)),
                             fx.make_layout((1, 8, 1), (1, 1, 0)))


def _q_fragment(resource, row_offset, lane, dq, dtype):
    parts = [_global_words(resource, row_offset + (lane >> 5) * 16 + d * 32, 0) for d in range(dq // 32)]
    words = fx.Vector.from_elements([p[i] for p in parts for i in range(4)], fx.Int32)
    frag = _mma(dtype).make_fragment_B(fx.make_rmem_tensor(fx.make_layout((BM, dq), (1, BM)), dtype))
    frag.store(words.bitcast(dtype))
    return frag


def _k_fragment(storage, base, slot, half, dq, dtype):
    k_tile = (dq // 16) * PADDED_CHUNK
    parts = [_lds_words(storage, slot * k_tile + base + half * 512, d * 2080) for d in range(dq // 32)]
    words = fx.Vector.from_elements([p[i] for p in parts for i in range(4)], fx.Int32)
    frag = _mma(dtype).make_fragment_A(fx.make_rmem_tensor(fx.make_layout((32, dq), (1, 32)), dtype))
    frag.store(words.bitcast(dtype))
    return frag


def _v_fragment(storage, base, slot, half, dq, dtype):
    k_tile = (dq // 16) * PADDED_CHUNK
    parts = [_lds_words(storage, 2 * k_tile + slot * V_TILE + base + half * PADDED_CHUNK,
                        n * 512 + k * 4160) for n in range(2) for k in range(2)]
    words = fx.Vector.from_elements([p[i] for p in parts for i in range(4)], fx.Int32)
    frag = fx.make_rmem_tensor(fx.make_layout((8, 4, 2), (1, 8, 32)), dtype)
    frag.store(words.bitcast(dtype))
    return frag


@flyc.jit
def _v_tail(values, remaining):
    words = fx.Vector(values).bitcast(fx.Int32)
    if remaining < BN:
        limit = _pin_i32(remaining - ((fx.Int32(gpu.thread_id("x")) >> 5) & 1) * 16)
        masks = []
        for i in fx.range_constexpr(8):
            token = (i // 4) * 32 + (i % 4) * 4
            mask = fx.Int32(0)
            for byte in fx.range_constexpr(4):
                mask = mask | (limit > token + byte).select(fx.Int32((255 << (8 * byte)) - (2**32 if byte == 3 else 0)), fx.Int32(0))
            masks.append(mask)
        words = fx.Vector.from_elements([words[i] & masks[i % 8] for i in range(16)], fx.Int32)
    return words.bitcast(fx.Vector(values).dtype)


def _qk(q, k, dtype):
    mma = _mma(dtype)
    acc = mma.make_fragment_C(fx.make_rmem_tensor(fx.make_layout((32, BM), (1, 32)), fx.Float32))
    acc.fill(0.0)
    fx.gemm(mma, acc, k, q, acc, traversal_order="kmn")
    return acc.load()


def _pv(p, v, o0, o1, dtype):
    mma = _mma(dtype)
    prob = mma.make_fragment_B(fx.make_rmem_tensor(fx.make_layout((BM, BN), (1, BM)), dtype))
    prob.store(p)
    acc = mma.make_fragment_C(fx.make_rmem_tensor(fx.make_layout((64, BM), (1, 64)), fx.Float32))
    acc.store(_join(o0, o1))
    operand = fx.make_view(fx.get_iter(v), fx.make_layout((8, 2, 4), (1, 32, 8)))
    fx.gemm(mma, acc, operand, prob, acc, traversal_order="mnk")
    return acc[None, 0, 0].load(), acc[None, 1, 0].load()


@flyc.jit
def _mask(scores, tile, row, q_len, kv_len, causal: fx.Constexpr[bool]):
    values = fx.Vector(scores)
    lane_half = (fx.Int32(gpu.thread_id("x")) >> 5) & 1
    if fx.const_expr(causal):
        bound = _min(kv_len - 1, kv_len - q_len + row) - tile * BN - lane_half * 16
        values = fx.Vector.from_elements([(bound >= (i // 16) * 32 + i % 16).select(values[i], fx.Float32(float("-inf")))
                                         for i in range(32)], fx.Float32)
    else:
        if kv_len - tile * BN < BN:
            tail_bound = _pin_i32(kv_len - 1 - tile * BN - lane_half * 16)
            values = fx.Vector.from_elements([(tail_bound >= (i // 16) * 32 + i % 16).select(values[i], fx.Float32(float("-inf")))
                                             for i in range(32)], fx.Float32)
    return values


@flyc.jit
def _rescale(o0, o1, o2, o3, row_sum, old_max, new_max, ballot):
    o0, o1, o2, o3 = fx.Vector(o0), fx.Vector(o1), fx.Vector(o2), fx.Vector(o3)
    row_sum = fx.Float32(row_sum)
    if ballot != fx.Int64(0):
        correction = _exp(fx.Float32(old_max) - fx.Float32(new_max))
        o0, o1, o2, o3 = o0 * correction, o1 * correction, o2 * correction, o3 * correction
        row_sum = row_sum * correction
    return o0, o1, o2, o3, row_sum


@flyc.jit
def _body(Q, K, V, O, LSE, QS, KS, VS, table, storage, q0, q_len, kv_len, head, qb,
          H: fx.Constexpr[int], HK: fx.Constexpr[int], NP: fx.Constexpr[int], DQ: fx.Constexpr[int],
          CAUSAL: fx.Constexpr[bool], PER_TOKEN: fx.Constexpr[bool], WITH_LSE: fx.Constexpr[bool],
          SCALE: fx.Constexpr[float], STAGGER: fx.Constexpr[bool]):
    dtype = Q.dtype
    tid = fx.Int32(gpu.thread_id("x"))
    wave, lane = _uniform(tid >> 6), tid & 63
    q_start = qb * BM
    row = q_start + wave * 32 + (lane & 31)
    valid = _min(fx.Int32(BM), q_len - q_start)
    hkv = head // (H // HK)
    q_ptr = fx.get_iter(Q) + ((fx.Int64(q0) + fx.Int64(q_start)) * (H * DQ) + fx.Int64(head) * DQ)
    gq = _buffer(fx.make_view(q_ptr, fx.make_layout(BM * H * DQ, 1)), valid * H * DQ)
    gk = _buffer(fx.make_view(fx.get_iter(K) + hkv * BN * DQ, fx.make_layout((NP * HK - hkv) * BN * DQ, 1)),
                 (NP * HK - hkv) * BN * DQ)
    gv = _buffer(fx.make_view(fx.get_iter(V) + hkv * BN * DV, fx.make_layout((NP * HK - hkv) * BN * DV, 1)),
                 (NP * HK - hkv) * BN * DV)
    q = _q_fragment(gq, (wave * 32 + (lane & 31)) * (H * DQ), lane, DQ, dtype)
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
    page0 = _prefetch_page(table, fx.Int32(0))
    page1 = _prefetch_page(table, _min(fx.Int32(1), last))
    page2 = _prefetch_page(table, _min(fx.Int32(2), last))
    # Three explicit s_waitcnt lgkmcnt(0) calls; the first already covers
    # all three issued page loads. No vector-memory wait is requested here.
    page0, page1, page2 = _page_ready(page0), _page_ready(page1), _page_ready(page2)
    k_row = (lane & 3) | ((lane & 4) << 2) | ((lane & 24) >> 1)
    k_base = (lane >> 5) * PADDED_CHUNK + ((k_row * 16) ^ ((k_row & 16) << 2))
    v_base = (lane >> 5) * (2 * PADDED_CHUNK) + (lane & 31) * 16
    k0 = _cooperative_load(gk, tid, page0 * (HK * BN * DQ), DQ // 64)
    k1 = _cooperative_load(gk, tid, page1 * (HK * BN * DQ), DQ // 64)
    v0 = _cooperative_load(gv, tid, page0 * (HK * BN * DV), 2)
    # s_waitcnt vmcnt(0): K(0)/K(1)/V(0) staging registers must be ready
    # before copying them into LDS; also drains earlier Q/scale VMEM.
    _wait(vmcnt=0)
    _cooperative_store(storage, tid, k0, 0, DQ // 64, True)
    _cooperative_store(storage, tid, k1, (DQ // 16) * PADDED_CHUNK, DQ // 64, True)
    _cooperative_store(storage, tid, v0, 2 * (DQ // 16) * PADDED_CHUNK, 2)
    # s_waitcnt vmcnt(0) lgkmcnt(0): finish prologue loads and LDS writes
    # before all waves rendezvous.
    _wait(vmcnt=0, lgkmcnt=0)
    _stage_end()
    if fx.const_expr(STAGGER):
        # Extra entry rendezvous offsets group1 by one stage; no new s_wait.
        _stage_end()

    k = _k_fragment(storage, k_base, 0, 0, DQ, dtype)
    # s_waitcnt vmcnt(0) lgkmcnt(0): K(0).lo LDS read ready for the first QK MFMA.
    _wait(vmcnt=0, lgkmcnt=0)
    _stage_end()
    lo = _qk(q, k, dtype)
    o0 = fx.Vector.filled(16, 0.0, fx.Float32)
    o1 = fx.Vector.filled(16, 0.0, fx.Float32)
    o2 = fx.Vector.filled(16, 0.0, fx.Float32)
    o3 = fx.Vector.filled(16, 0.0, fx.Float32)
    _schedule(DQ // 16, 3, 5)
    o0, o1, o2, o3 = _pin(o0), _pin(o1), _pin(o2), _pin(o3)
    _stage_end()
    k = _k_fragment(storage, k_base, 0, 1, DQ, dtype)
    # s_waitcnt vmcnt(0) lgkmcnt(0): K(0).hi ready before QK consumes it.
    _wait(vmcnt=0, lgkmcnt=0)
    _stage_end()
    hi = _qk(q, k, dtype)
    scores = _mask(_join(lo, hi), fx.Int32(0), row, q_len, kv_len, CAUSAL)
    maximum = _maximum(_row_max(scores) * scale, fx.Float32(-1.0e30))
    scores = _center(scores, scale, maximum, 0, 32)
    row_sum = fx.Float32(0.0)
    _stage_end()
    # Prime the extra page-ID lookahead: phase(1) will prefetch K(3).
    # Its address must be ready before S0 so K.lo need not wait midway.
    page3_request = _prefetch_page(table, _min(fx.Int32(3), last))
    k2 = _cooperative_load(gk, tid, page2 * (HK * BN * DQ), DQ // 64)
    # s_waitcnt vmcnt(0): prefetched K(2) ready for the retired K(0) slot.
    _wait(vmcnt=0)
    _cooperative_store(storage, tid, k2, 0, DQ // 64, True)
    # s_waitcnt lgkmcnt(0): complete K(2) publication AND page(3) SMEM.
    # The tied scalar result keeps the next phase's address use after it.
    page3 = _page_ready(page3_request)
    _stage_end()

    @flyc.jit
    def phase(previous, maximum, row_sum, o0, o1, o2, o3, prev_page, current_page, next_page, future_page, t,
              CUR: fx.Constexpr[int], PREV: fx.Constexpr[int]):
        previous, maximum, row_sum = fx.Vector(previous), fx.Float32(maximum), fx.Float32(row_sum)
        o0, o1, o2, o3 = fx.Vector(o0), fx.Vector(o1), fx.Vector(o2), fx.Vector(o3)
        t = fx.Int32(t)
        # S0: issue K.lo first and keep its LGKM wait at the very end.
        # future_page = page(t+2) was made ready in the previous phase (or
        # prologue); the new page(t+3) request is only used by the next phase.
        k = _k_fragment(storage, k_base, CUR, 0, DQ, dtype)
        # Compiler-only fence: do not hoist address setup or VMEM before
        # the six D192/four D128 LDS reads. It does not wait for the data.
        rocdl.sched_barrier(0)
        future_request = _prefetch_page(table, _min(t + 3, last))
        v_staging = _cooperative_load(gv, tid, current_page * (HK * BN * DV), 2)
        k_staging = _cooperative_load(gk, tid, future_page * (HK * BN * DQ), DQ // 64)
        # Keep all independent work above the single final LGKM wait.
        # V must still precede K in VMEM order for S2's rolling counter.
        rocdl.sched_barrier(0)
        # s_waitcnt lgkmcnt(0): K(t).lo AND next phase's page ID ready;
        # V(t)/K(t+2) VMEM remain free to overlap this wait and S1.
        future = _page_ready(future_request)
        _stage_end()
        # S1: QK low + 24 previous exps, with explicit VALU/MFMA co-issue hints.
        lo = _qk(q, k, dtype)
        previous = _pin(_exp_part(previous, 0, 24), 32)
        _schedule(DQ // 16, 2 if DQ == 192 else 3, 1, True)
        _stage_end()
        # S2: current K high and publish V(t), while the other group computes.
        k = _k_fragment(storage, k_base, CUR, 1, DQ, dtype)
        # s_waitcnt vmcnt(3) for D192, vmcnt(2) for D128: S0 issued
        # V's two b64 loads BEFORE K's DQ//64 loads. Wait for V only,
        # allowing those newer K(t+2) requests to remain outstanding.
        _wait(vmcnt=DQ // 64)
        _cooperative_store(storage, tid, v_staging, 2 * (DQ // 16) * PADDED_CHUNK + CUR * V_TILE, 2)
        # s_waitcnt lgkmcnt(0): K(t).hi read + V(t) LDS writes complete.
        _wait(lgkmcnt=0)
        _stage_end()
        # S3: QK high + remaining exps, sum and native packed FP8 conversion.
        hi = _qk(q, k, dtype)
        previous = _exp_part(previous, 24, 32)
        row_sum = row_sum + _row_sum(previous)
        p = _pack_probability(previous, dtype)
        _schedule(4, 2, 2, True)
        _schedule(DQ // 16 - 4, 8, 2)
        _stage_end()
        # S4: V(t-1) low and K(t+2) publication after BOTH groups read K(t).
        v = _v_fragment(storage, v_base, PREV, 0, DQ, dtype)
        # s_waitcnt vmcnt(0): retire K(t+2) global prefetch before store.
        # Safe slot reuse comes from the stage barriers, not this wait.
        _wait(vmcnt=0)
        _cooperative_store(storage, tid, k_staging, CUR * (DQ // 16) * PADDED_CHUNK, DQ // 64, True)
        # s_waitcnt lgkmcnt(0): V.lo ready + K(t+2) LDS publication complete;
        # also covers earlier pending lane shuffles.
        _wait(lgkmcnt=0)
        _stage_end()
        current = _mask(_join(lo, hi), t, row, q_len, kv_len, CAUSAL)
        # S5: PV low + current max/center. A six-bit lazy margin stays within
        # FP8-FNUZ finite range; gfx950 BF16's eight-bit margin is unsafe here.
        o0, o1 = _pv(p, v, o0, o1, dtype)
        candidate = _row_max(current) * scale
        ballot = fx.Int64(rocdl.ballot(fx.Int64.ir_type, (candidate - maximum > 6.0).ir_value()))
        new_max = (ballot != fx.Int64(0)).select(_maximum(maximum, candidate), maximum)
        split = 12 if STAGGER else 6
        current = _center(current, scale, new_max, 0, split)
        # Keep the local MAX3 tree interleaved with the first four PV
        # MFMAs. Then issue the xor32 ds_bpermute and two independent
        # PV MFMAs before its v_max consumer. These are compiler-only
        # hints; LLVM still inserts lgkmcnt(0) before using the result.
        # Main-loop D192 ISA has two MFMAs in the gap; tail/causal
        # blocks may schedule differently and must retain the wait.
        _schedule(3, 5, 3)
        rocdl.sched_group_barrier(rocdl.mask_mfma, 1, 3)
        rocdl.sched_group_barrier(0x002, 3, 3)
        rocdl.sched_group_barrier(rocdl.mask_dsrd, 1, 3)
        rocdl.sched_group_barrier(rocdl.mask_mfma, 2, 3)
        rocdl.sched_group_barrier(0x002, 10, 3)
        _schedule(2, 5, 3)
        _stage_end()
        # S6: V high; V(t-1) is a full page in every forward main phase.
        v = _v_fragment(storage, v_base, PREV, 1, DQ, dtype)
        # s_waitcnt vmcnt(0) lgkmcnt(0): V.hi LDS read ready for S7;
        # drain pending VMEM/LGKM too.
        _wait(vmcnt=0, lgkmcnt=0)
        _stage_end()
        # S7: PV high + remaining centered scores + lazy output rescale.
        o2, o3 = _pv(p, v, o2, o3, dtype)
        current = _center(current, scale, new_max, split, 32)
        _schedule(8, 4, 4)
        o0, o1, o2, o3, row_sum = _rescale(o0, o1, o2, o3, row_sum, maximum, new_max, ballot)
        _stage_end()
        return current, new_max, row_sum, o0, o1, o2, o3, current_page, next_page, future_page, future

    for t in range(fx.Int32(1), tiles - 1, fx.Int32(2)):
        scores, maximum, row_sum, o0, o1, o2, o3, page0, page1, page2, page3 = phase(scores, maximum, row_sum, o0, o1, o2, o3, page0, page1, page2, page3, t, 1, 0)
        scores, maximum, row_sum, o0, o1, o2, o3, page0, page1, page2, page3 = phase(scores, maximum, row_sum, o0, o1, o2, o3, page0, page1, page2, page3, t + 1, 0, 1)
    if (tiles & 1) == 0:
        scores, maximum, row_sum, o0, o1, o2, o3, page0, page1, page2, page3 = phase(scores, maximum, row_sum, o0, o1, o2, o3, page0, page1, page2, page3, last, 1, 0)

    scores = _exp_part(fx.Vector(scores), 0, 32)
    row_sum = fx.Float32(row_sum) + _row_sum(scores)
    p = _pack_probability(scores, dtype)
    # s_waitcnt vmcnt(0) lgkmcnt(0): drain outstanding prefetch/publication
    # and lane-shuffle work at the main-loop -> last-page boundary.
    _wait(vmcnt=0, lgkmcnt=0)
    _stage_end()
    v = _v_fragment(storage, v_base, last & 1, 0, DQ, dtype)
    # s_waitcnt vmcnt(0) lgkmcnt(0) + scheduler fence BEFORE _v_tail uses
    # asynchronous V.lo data. Zero invalid KV bytes to avoid 0 * NaN in PV.
    _wait(vmcnt=0, lgkmcnt=0)
    rocdl.sched_barrier(0)
    v.store(_v_tail(v.load(), kv_len - last * BN))
    _stage_end()
    o0, o1 = _pv(p, v, o0, o1, dtype)
    _stage_end()
    v = _v_fragment(storage, v_base, last & 1, 1, DQ, dtype)
    # Same wait/fence for V.hi (Dv columns64:128, NOT the next KV half).
    # Both V halves reduce over the same BN64 tokens and use the same tail.
    _wait(vmcnt=0, lgkmcnt=0)
    rocdl.sched_barrier(0)
    v.store(_v_tail(v.load(), kv_len - last * BN))
    _stage_end()
    o2, o3 = _pv(p, v, o2, o3, dtype)
    _stage_end()
    if fx.const_expr(not STAGGER):
        # Balance group1's extra entry barrier. Both groups have now drained
        # their last V reads before C-shuffle aliases the K/V LDS storage.
        _stage_end()
    inv = (row_sum > 0.0).select(fx.Float32(1.0) / row_sum, fx.Float32(0.0)) * VS[0]
    out_tid = _pin_i32(fx.Int32(gpu.thread_id("x")))
    out_row = (out_tid >> 6) * 32 + (out_tid & 31)
    optr = fx.get_iter(O) + ((fx.Int64(q0) + fx.Int64(q_start)) * (H * DV) + fx.Int64(head) * DV)
    obuf = rocdl.make_buffer_tensor(fx.make_view(optr, fx.make_layout(BM * H * DV, 1)), num_records_bytes=valid * H * DV * 2)
    # Both staggered groups have drained the K/V rings. Reuse the first
    # 32 KiB for two output halves: contiguous b128 VMEM stores instead
    # of one request per query row. XOR row[3:0] into column[5:2] makes
    # b64 LDS writes conflict-free; odd rows swap two packed-word pairs.
    atom = fx.make_copy_atom(rocdl.BufferCopy128b(), fx.BFloat16)
    outputs = (o0, o1, o2, o3)
    for half in fx.range_constexpr(2):
        for n in fx.range_constexpr(2):
            for group in fx.range_constexpr(4):
                val = fx.Vector.from_elements([outputs[half * 2 + n][group * 4 + i] * inv for i in range(4)], fx.Float32)
                words = _pack_bf16(val)
                col = n * 32 + group * 8 + ((out_tid >> 5) & 1) * 4
                element = (out_row * 64 + col) ^ ((out_row & 15) * 4)
                address = fx.Int32(fx.ptrtoint(fx.get_iter(storage) + element * 2))
                llvm.inline_asm(ir.Type.parse("!llvm.void"), [address.ir_value(), words.ir_value()],
                    "ds_write_b64 $0, $1", "v,v,~{memory}", has_side_effects=True)
        # s_waitcnt lgkmcnt(0): finish this wave's C-shuffle b64 writes;
        # the following CTA barrier makes all writers ready for readers.
        _wait(lgkmcnt=0)
        _stage_end()
        for i in fx.range_constexpr(4):
            element = out_tid * 8 + i * THREADS * 8
            read_row, read_col = element // 64, element % 64
            element = element ^ ((read_row & 14) * 4)
            words = _lds_words(storage, element * 2)
            # s_waitcnt lgkmcnt(0) + fence: read result must be ready
            # before word-pair permutation and the b128 global store.
            _wait(lgkmcnt=0)
            rocdl.sched_barrier(0)
            src = fx.make_rmem_tensor(8, fx.BFloat16)
            src.store(fx.Vector.from_elements([
                ((read_row & 1) == 0).select(words[j], words[j ^ 2]) for j in range(4)
            ], fx.Int32).bitcast(fx.BFloat16))
            offset = read_row * (H * DV) + half * 64 + read_col
            fx.copy(atom, src, fx.make_view(fx.get_iter(obuf) + offset, fx.make_layout(8, 1)))
        # Readers rendezvous before the next half overwrites LDS. No
        # explicit VMEM-store drain here; this barrier is not vmcnt(0).
        _stage_end()
    if fx.const_expr(WITH_LSE):
        if ((out_tid & 63) < 32) & (out_row < valid):
            log_l = fx.Float32(llvm.call_intrinsic(fx.Float32.ir_type, "llvm.log2.f32", [row_sum.ir_value()], [], []))
            LSE[(q0 + q_start + out_row) * H + head] = (row_sum > 0.0).select(
                (fx.Float32(maximum) + log_l) * fx.Float32(math.log(2.0)), fx.Float32(float("-inf")))


@flyc.kernel(known_block_size=[THREADS, 1, 1])
def _attention_kernel_942(Q: fx.Tensor, K: fx.Tensor, V: fx.Tensor, O: fx.Tensor, LSE: fx.Tensor,
    CQ: fx.Tensor, KI: fx.Tensor, PAGES: fx.Tensor, LAST: fx.Tensor, QS: fx.Tensor, KS: fx.Tensor, VS: fx.Tensor,
    H: fx.Constexpr[int], HK: fx.Constexpr[int], NP: fx.Constexpr[int], DQ: fx.Constexpr[int],
    CAUSAL: fx.Constexpr[bool], PER_TOKEN: fx.Constexpr[bool], WITH_LSE: fx.Constexpr[bool],
    SCALE: fx.Constexpr[float]):
    head, batch, qb = fx.Int32(gpu.block_id("x")), fx.Int32(gpu.block_id("y")), fx.Int32(gpu.block_id("z"))
    q0 = _uniform(CQ[batch])
    q_len = _uniform(CQ[batch + 1]) - q0
    start = _uniform(KI[batch])
    pages = _uniform(KI[batch + 1]) - start
    kv_len = (pages > 0).select((pages - 1) * BN + _uniform(LAST[batch]), fx.Int32(0))
    table = fx.make_view(fx.get_iter(PAGES) + start, fx.make_layout(pages, 1))
    size = 2 * (DQ // 16) * PADDED_CHUNK + 2 * V_TILE
    storage = fx.SharedAllocator().allocate(fx.Array[fx.Int8, size, 16]).peek().view(fx.make_layout(size, 1))
    if qb * BM < q_len:
        if kv_len > 0:
            group = _uniform(fx.Int32(gpu.thread_id("x")) >> 8)
            if group != 0:
                _body(Q, K, V, O, LSE, QS, KS, VS, table, storage, q0, q_len, kv_len, head, qb,
                        H, HK, NP, DQ, CAUSAL, PER_TOKEN, WITH_LSE, SCALE, True)
            else:
                _body(Q, K, V, O, LSE, QS, KS, VS, table, storage, q0, q_len, kv_len, head, qb,
                        H, HK, NP, DQ, CAUSAL, PER_TOKEN, WITH_LSE, SCALE, False)
        else:
            tid = fx.Int32(gpu.thread_id("x"))
            valid = _min(q_len - qb * BM, fx.Int32(BM))
            pointer = fx.get_iter(O) + ((fx.Int64(q0) + fx.Int64(qb * BM)) * (H * DV) + fx.Int64(head) * DV)
            out_buffer = rocdl.make_buffer_tensor(fx.make_view(pointer, fx.make_layout(BM * H * DV, 1)),
                                                 num_records_bytes=valid * H * DV * 2)
            zeros = fx.make_rmem_tensor(8, fx.BFloat16)
            zeros.fill(0.0)
            zero_copy = fx.make_copy_atom(rocdl.BufferCopy128b(), fx.BFloat16)
            for i in fx.range_constexpr(BM * DV // (THREADS * 8)):
                index = tid * 8 + i * THREADS * 8
                offset = (index // DV) * (H * DV) + index % DV
                fx.copy(zero_copy, zeros, fx.make_view(fx.get_iter(out_buffer) + offset, fx.make_layout(8, 1)))
            if fx.const_expr(WITH_LSE):
                if (tid < BM) & (qb * BM + tid < q_len):
                    LSE[(q0 + qb * BM + tid) * H + head] = fx.Float32(float("-inf"))


@flyc.jit
def _launch_attention(Q: fx.Tensor, K: fx.Tensor, V: fx.Tensor, O: fx.Tensor, LSE: fx.Tensor,
    CQ: fx.Tensor, KI: fx.Tensor, PAGES: fx.Tensor, LAST: fx.Tensor, QS: fx.Tensor, KS: fx.Tensor, VS: fx.Tensor,
    H: fx.Constexpr[int], HK: fx.Constexpr[int], NP: fx.Constexpr[int], B: fx.Constexpr[int], MAX_Q: fx.Constexpr[int],
    DQ: fx.Constexpr[int], CAUSAL: fx.Constexpr[bool], PER_TOKEN: fx.Constexpr[bool], WITH_LSE: fx.Constexpr[bool],
    SCALE: fx.Constexpr[float], stream: fx.Stream):
    _attention_kernel_942(Q, K, V, O, LSE, CQ, KI, PAGES, LAST, QS, KS, VS, H, HK, NP, DQ,
        CAUSAL, PER_TOKEN, WITH_LSE, SCALE, value_attrs={"rocdl.waves_per_eu": 2},
    ).launch(grid=(H, B, (MAX_Q + BM - 1) // BM), block=(THREADS, 1, 1), stream=stream)


class _PagedAttention:
    def __init__(self, heads, kv_heads, dq, causal):
        self.heads, self.kv_heads, self.dq, self.causal = heads, kv_heads, dq, causal
        self.memory_mode = "lds"
        self._compiled = {}

    def __call__(self, Q, K, V, cu_seqlens_q, cu_seqlens_k, kv_indptr, kv_page_indices,
                 max_seqlen_q, max_seqlen_k, causal, q_descale, k_descale, v_descale,
                 kv_last_page_lens, out=None, sink_ptr=None, stream=None, *, return_lse=False, lse=None, softmax_scale=None):
        """Run native FNUZ attention with caller-owned asynchronous metadata.

        Descales must be finite and strictly positive. Prefix sums/page IDs,
        last-page lengths and maximum lengths must describe the current input;
        they are never synchronized back to the host. Warm before graph capture.
        """
        if not Q.is_cuda or "gfx942" not in torch.cuda.get_device_properties(Q.device).gcnArchName:
            raise NotImplementedError("this public kernel requires gfx942; use the separate cross-compile validation harness elsewhere")
        if Q.dtype != torch.float8_e4m3fnuz or K.dtype != Q.dtype or V.dtype != Q.dtype:
            raise NotImplementedError("gfx942 native FP8 E4M3FNUZ Q/K/V only")
        if causal != self.causal or sink_ptr is not None:
            raise ValueError("causal must match the factory; sink is not implemented in this initial gfx942 pipeline")
        if Q.ndim != 3 or Q.shape[1:] != (self.heads, self.dq) or not Q.is_contiguous():
            raise ValueError("Q must be contiguous [tokens, heads, Dqk]")
        if K.ndim != 5 or V.ndim != 5 or K.shape != (V.shape[0], self.kv_heads, self.dq // 16, BN, 16) or V.shape[1:] != (self.kv_heads, 4, DV, 16):
            raise ValueError("K/V must be page64 SHUFFLE-5D layouts")
        if not K.is_contiguous() or not V.is_contiguous() or K.device != Q.device or V.device != Q.device:
            raise ValueError("K/V must be contiguous on the input GPU")
        if K.numel() >= 2**31 or V.numel() >= 2**31:
            raise NotImplementedError("physical KV buffer byte spans must fit signed int32")
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
        if max_seqlen_q < 0 or max_seqlen_k < 0:
            raise ValueError("length bounds must be nonnegative")
        scales = (q_descale, k_descale, v_descale)
        if any(t.dtype != torch.float32 or t.device != Q.device or not t.is_contiguous() for t in scales):
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
        if out.shape != (Q.shape[0], self.heads, DV) or out.dtype != torch.bfloat16 or out.device != Q.device or not out.is_contiguous():
            raise ValueError("output must be contiguous BF16 [tokens, heads, 128]")
        if lse is not None and (lse.shape != (Q.shape[0], self.heads) or lse.dtype != torch.float32 or lse.device != Q.device or not lse.is_contiguous()):
            raise ValueError("LSE must be contiguous FP32 [tokens, heads]")
        if Q.shape[0] and max_seqlen_q:
            args = (Q.view(-1), K.view(-1), V.view(-1), out.view(-1), lse.view(-1) if lse is not None else k_descale.view(-1),
                    cu_seqlens_q, kv_indptr, kv_page_indices, kv_last_page_lens, q_descale.view(-1), k_descale.view(-1), v_descale.view(-1),
                    self.heads, self.kv_heads, K.shape[0], batch, max_seqlen_q, self.dq, self.causal,
                    q_descale.numel() != 1, lse is not None, scale, stream)
            signature = tuple((a.dtype, tuple(a.shape)) if isinstance(a, torch.Tensor) else ("stream",) if hasattr(a, "cuda_stream") else a for a in args)
            key = (Q.device, signature)
            compiled = self._compiled.get(key)
            with torch.cuda.device(Q.device):
                if compiled is None:
                    self._compiled[key] = flyc.compile(_launch_attention, *args)
                else:
                    compiled(*args)
        return (out, lse) if return_lse else out


@functools.cache
def PagedAttention(num_qo_heads, num_kv_heads, head_dim_qk, head_dim_v, page_size,
                   is_causal, quant_query_mode="per-token", key_layout="vectorized",
                   window_left=-1, has_sink=False, *, memory_mode="lds", persistent=None):
    """Native FNUZ backend; memory_mode='lds' is retained for compatibility only."""
    if memory_mode != "lds":
        raise NotImplementedError("gfx942 FP8 supports memory_mode='lds' only; register mode has been removed")
    if persistent not in (None, False):
        raise NotImplementedError("gfx942 FP8 does not implement persistent scheduling")
    if head_dim_qk not in (128, 192) or (head_dim_v, page_size, key_layout) != (DV, BN, "vectorized"):
        raise NotImplementedError("gfx942 FP8 D128/D192 V128 page64 SHUFFLE-5D only")
    if window_left != -1 or has_sink:
        raise NotImplementedError("initial gfx942 pipeline supports full causal/noncausal; SWA/sink are not implemented")
    if num_qo_heads <= 0 or num_kv_heads <= 0 or num_qo_heads % num_kv_heads:
        raise ValueError("query heads must be a positive multiple of KV heads")
    if quant_query_mode not in ("per-token", "per-tensor"):
        raise ValueError("unsupported scale mode")
    return _PagedAttention(num_qo_heads, num_kv_heads, head_dim_qk, is_causal)