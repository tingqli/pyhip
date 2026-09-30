"""Experimental gfx942 M32 attention: QK key-split, P exchange, PV DV-split.

BM128/BN64, eight waves paired as (w, w+4). Each pair owns 32 query rows.
Both waves reduce the complete DQ256; each computes a different 32-key half,
then exchanges BF16 probabilities and FP32 row statistics through LDS. Each
wave accumulates only DV128 (64 FP32/lane), in two DV64 register fragments.

K occupies LDS [0,32768), V [32768,65536). After every K reader retires, K
aliases two P planes [0,16384), max [16384,18432), sum [18432,20480).
Five CTA rendezvous per BN enforce K->P, max publication, P/sum publication,
P/V retirement, and next-KV publication. No inter-CTA communication/counters.

Import this module's PagedAttention explicitly. The default v48 factory and
device code are not changed. Numerical/API validation is shared with it.
"""

import functools
import math

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu, rocdl
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm

if __package__:
    from .mha_pa_bf16_942 import (
        PagedAttention as _validate_factory, _PagedAttention, _uniform, _min,
        _pin_i32, _pin_s64, _buffer, _buffer_words, _read_address, _stage_end,
        _wait, _page_ready, _pack_bf16, _rescale, _advance_max,
    )
    from .mha_pa_bf16_256_942 import (
        _load_k, _load_v, _pages, _max as _max16, _sum as _sum16,
        _center, _exps, _pack,
    )
else:
    from mha_pa_bf16_942 import (
        PagedAttention as _validate_factory, _PagedAttention, _uniform, _min,
        _pin_i32, _pin_s64, _buffer, _buffer_words, _read_address, _stage_end,
        _wait, _page_ready, _pack_bf16, _rescale, _advance_max,
    )
    from mha_pa_bf16_256_942 import (
        _load_k, _load_v, _pages, _max as _max16, _sum as _sum16,
        _center, _exps, _pack,
    )


BM, BN, D, THREADS = 128, 64, 256, 512
K_BYTES, V_BYTES, LDS_BYTES = 32768, 32768, 65536
P_BYTES, MAX_BASE, SUM_BASE = 16384, 16384, 18432


def _k_write_offset(tid):
    return (tid >> 6) * 1024 + (((tid & 63) * 16) ^ ((tid & 16) << 2))


def _k_read_offset(lane, key_half):
    krow = (lane & 3) | ((lane & 4) << 1) | ((lane & 8) >> 1) | (lane & 16)
    return key_half * 512 + (lane >> 5) * 1024 + ((krow * 16) ^ ((krow & 16) << 2))


def _p_write_offset(tid):
    return tid * 16


def _p_peer_offset(tid):
    return (tid ^ 256) * 16


def _v_read_offset(lane, value_half, quarter):
    return K_BYTES + (lane >> 5) * 4096 + (lane & 31) * 16 + value_half * 2048 + quarter * 1024


def _mma():
    return fx.make_tiled_mma(
        fx.make_mma_atom(rocdl.MFMA(32, 32, 8, fx.BFloat16)),
        fx.make_layout((1, 4, 1), (1, 1, 0)),
        (None, None, fx.make_layout((4, 2, 2), (1, 8, 4))),
    )


def _q_fragment(resource, offset, lane):
    parts = [_buffer_words(resource, offset + (lane >> 5) * 16 + k * 32) for k in range(16)]
    return fx.Vector.from_elements([part[i] for part in parts for i in range(4)], fx.Int32)


def _read_k(address, chunk):
    parts = [_read_address(address, chunk * 16384 + k * 2048) for k in range(8)]
    return fx.Vector.from_elements([part[i] for part in parts for i in range(4)], fx.Int32)


def _read_v(address):
    parts = [_read_address(address, n * 512 + k * 8192) for n in range(2) for k in range(4)]
    return fx.Vector.from_elements([part[i] for part in parts for i in range(4)], fx.Int32)


def _read_p(address):
    parts = [_read_address(address, packet * 8192) for packet in range(2)]
    return fx.Vector.from_elements([part[i] for part in parts for i in range(4)], fx.Int32)


def _publish(address, words):
    for packet in range(words.numel // 4):
        part = fx.Vector.from_elements([words[packet * 4 + j] for j in range(4)], fx.Int32)
        llvm.inline_asm(ir.Type.parse("!llvm.void"), [address.ir_value(), part.ir_value()],
                        f"ds_write_b128 $0, $1 offset:{packet * 8192}", "v,v,~{memory}", has_side_effects=True)


def _write_scalar(address, value):
    llvm.inline_asm(ir.Type.parse("!llvm.void"), [address.ir_value(), value.ir_value()],
                    "ds_write_b32 $0, $1", "v,v,~{memory}", has_side_effects=True)


def _read_scalar(address):
    return fx.Float32(llvm.inline_asm(fx.Float32.ir_type, [address.ir_value()],
        "ds_read_b32 $0, $1", "=v,v,~{memory}", has_side_effects=True))


def _maximum(a, b):
    return fx.Float32(llvm.call_intrinsic(fx.Float32.ir_type, "llvm.maxnum.f32",
                                         [a.ir_value(), b.ir_value()], [], []))


def _qk(q_words, address):
    mma = _mma()
    s = mma.make_fragment_C(fx.make_rmem_tensor(fx.make_layout((32, BM), (1, 32)), fx.Float32))
    s.fill(0.0)
    # One K128 operand at a time: keep the K operand narrow without changing
    # the per-score DQ reduction order or carrying both halves together.
    for chunk in range(2):
        words = _read_k(address, chunk)
        _wait(lgkmcnt=0)
        rocdl.sched_barrier(0)
        q = mma.make_fragment_B(fx.make_rmem_tensor(fx.make_layout((BM, 128), (1, BM)), fx.BFloat16))
        q.store(fx.Vector.from_elements([q_words[chunk * 32 + i] for i in range(32)], fx.Int32).bitcast(fx.BFloat16))
        k = mma.make_fragment_A(fx.make_rmem_tensor(fx.make_layout((32, 128), (1, 32)), fx.BFloat16))
        k.store(words.bitcast(fx.BFloat16))
        fx.gemm(mma, s, k, q, s, traversal_order="kmn")
        rocdl.sched_barrier(0)
    return s.load()


def _merge_p(own, peer, half):
    low = [(half == 0).select(own[i], peer[i]) for i in range(8)]
    high = [(half == 0).select(peer[i], own[i]) for i in range(8)]
    return fx.Vector.from_elements(low + high, fx.Int32).bitcast(fx.BFloat16)


def _pv(p, words, output):
    mma = _mma()
    prob = mma.make_fragment_B(fx.make_rmem_tensor(fx.make_layout((BM, BN), (1, BM)), fx.BFloat16))
    prob.store(p)
    # Compact storage first, then the established N/K-transposed MFMA view.
    values = fx.make_rmem_tensor(fx.make_layout((8, 4, 2), (1, 8, 32)), fx.BFloat16)
    values.store(words.bitcast(fx.BFloat16))
    operand = fx.make_view(fx.get_iter(values), fx.make_layout((4, 2, (2, 4)), (1, 32, (4, 8))))
    acc = mma.make_fragment_C(fx.make_rmem_tensor(fx.make_layout((64, BM), (1, 64)), fx.Float32))
    acc.store(output)
    fx.gemm(mma, acc, operand, prob, acc, traversal_order="mnk")
    return acc.load()


@flyc.jit
def _mask(scores, tile, half, row, q_len, kv_len, CAUSAL: fx.Constexpr[bool]):
    scores = fx.Vector(scores)
    if fx.const_expr(CAUSAL):
        bound = _min(kv_len - 1, kv_len - q_len + row)
    else:
        bound = kv_len - 1
    limit = _pin_i32(bound - tile * BN - half * 32 - ((fx.Int32(gpu.thread_id("x")) >> 5) & 1) * 8)
    return fx.Vector.from_elements([
        (limit >= (i // 8) * 16 + i % 8).select(scores[i], fx.Float32(float("-inf")))
        for i in range(16)
    ], fx.Float32)


@flyc.jit
def _v_tail(words, remaining):
    words = fx.Vector(words)
    if remaining < BN:
        limit = _pin_i32(remaining - ((fx.Int32(gpu.thread_id("x")) >> 5) & 1) * 8)
        masks = []
        for i in fx.range_constexpr(16):
            token = (i // 4) * 16 + (i % 4) * 2
            mask = (limit > token).select(fx.Int32(0xFFFF), fx.Int32(0))
            mask = mask | (limit > token + 1).select(fx.Int32(-65536), fx.Int32(0))
            masks.append(mask)
        words = fx.Vector.from_elements([words[i] & masks[i % 16] for i in range(32)], fx.Int32)
    return words


@flyc.jit
def _body(Q, K, V, O, LSE, QS, KS, VS, table, storage, q0, q_len, kv_len, head, qb,
          H: fx.Constexpr[int], HK: fx.Constexpr[int], NP: fx.Constexpr[int], PAGE: fx.Constexpr[int],
          CAUSAL: fx.Constexpr[bool], PER_TOKEN: fx.Constexpr[bool], WITH_LSE: fx.Constexpr[bool],
          SCALE: fx.Constexpr[float]):
    # Explicit closures keep native FlyDSL cache dependencies visible after AST lifting.
    read_v, read_p, read_scalar = _read_v, _read_p, _read_scalar
    load_k, load_v, publish, write_scalar = _load_k, _load_v, _publish, _write_scalar
    qk, pv, mask, tail, merge_p = _qk, _pv, _mask, _v_tail, _merge_p
    local_max, local_sum, maximum2 = _max16, _sum16, _maximum
    exps, center, pack, rescale, advance_max = _exps, _center, _pack, _rescale, _advance_max
    pages, ready, wait, sync = _pages, _page_ready, _wait, _stage_end
    tid = _pin_i32(fx.Int32(gpu.thread_id("x")))
    lane, wave = tid & 63, _uniform(tid >> 6)
    half, pair = _uniform(tid >> 8), wave & 3
    q_start = qb * BM
    localrow = pair * 32 + (lane & 31)
    row = q_start + localrow
    valid = _min(fx.Int32(BM), q_len - q_start)
    hkv = head // (H // HK)
    qptr = fx.get_iter(Q) + ((fx.Int64(q0) + fx.Int64(q_start)) * (H * D) + fx.Int64(head) * D)
    gq = _buffer(fx.make_view(qptr, fx.make_layout(BM * H * D, 1)), valid * H * D * 2)
    gk = _buffer(fx.make_view(fx.get_iter(K) + fx.Int64(hkv) * PAGE * D,
                            fx.make_layout((NP * HK - hkv) * PAGE * D, 1)), (NP * HK - hkv) * PAGE * D * 2)
    q = _q_fragment(gq, localrow * H * D * 2, lane)
    scale = fx.Float32(KS[0]) * fx.Float32(SCALE * math.log2(math.e))
    if fx.const_expr(PER_TOKEN):
        qsptr = fx.get_iter(QS) + (fx.Int64(q0) + fx.Int64(q_start)) * H + fx.Int64(head)
        qs = rocdl.make_buffer_tensor(fx.make_view(qsptr, fx.make_layout(BM * H, 1)), num_records_bytes=valid * H * 4)
        scale = scale * qs[localrow * H]
    else:
        scale = scale * QS[0]
    scale = fx.Float32(scale)
    tiles = (kv_len + BN - 1) // BN
    if fx.const_expr(CAUSAL):
        end = (q_start + valid + kv_len - q_len + BN - 1) // BN
        tiles = _min(tiles, (end > 0).select(end, fx.Int32(1)))
    last = tiles - 1
    shared = fx.Int32(fx.ptrtoint(fx.get_iter(storage)))
    # The first two versions spilled long-lived address expressions. Recreate
    # LDS addresses at their consumers; the opaque tid pin prevents LICM from
    # extending these values across the persistent loop or all five phases.
    klane = _pin_i32((tid >> 6) * PAGE * 16 + (tid & (min(PAGE, BN) - 1)) * 16)
    vlane = _pin_i32(tid * 16)
    vbase = _pin_s64(fx.Int64(fx.ptrtoint(fx.get_iter(V))) + fx.Int64(hkv) * PAGE * D * 2)
    page0, page1 = pages(table, fx.Int32(0), last, kv_len, PAGE)
    page0, page1 = ready(page0), ready(page1)
    kp = load_k(gk, klane, page0, fx.Int32(0), HK, PAGE, page1)
    vp = load_v(vbase, vlane, page0, fx.Int32(0), HK, PAGE, page1)
    wait(vmcnt=0)
    rocdl.sched_barrier(0)
    publish(_pin_i32(shared + _k_write_offset(_pin_i32(tid))), kp)
    publish(_pin_i32(shared + K_BYTES + _pin_i32(tid) * 16), vp)
    wait(lgkmcnt=0)
    sync()
    maximum, total = fx.Float32(-1.0e30), fx.Float32(0.0)
    o0, o1 = fx.Vector.filled(32, 0.0, fx.Float32), fx.Vector.filled(32, 0.0, fx.Float32)
    for tile in range(fx.Int32(0), tiles, fx.Int32(1)):
        tile, maximum, total = fx.Int32(tile), fx.Float32(maximum), fx.Float32(total)
        o0, o1 = fx.Vector(o0), fx.Vector(o1)
        kr = _pin_i32(shared + _k_read_offset(_pin_i32(tid) & 63, half))
        scores = mask(qk(q, kr), tile, half, row, q_len, kv_len, CAUSAL)
        sync()  # 1: every K operand has been consumed before K aliases P/meta.
        candidate = local_max(scores)
        candidate = maximum2(candidate, candidate.shuffle_xor(32, 64))
        write_scalar(_pin_i32(shared + MAX_BASE + _pin_i32(tid) * 4), candidate)
        wait(lgkmcnt=0)
        sync()  # 2: both key halves have published row maxima.
        peer_max = read_scalar(_pin_i32(shared + MAX_BASE + (_pin_i32(tid) ^ 256) * 4))
        wait(lgkmcnt=0)
        rocdl.sched_barrier(0)
        candidate = maximum2(candidate, peer_max) * scale
        advance = candidate > maximum + 7.0
        new_max = advance.select(candidate + 1.0, maximum)
        probabilities = exps(center(scores, scale, new_max))
        subtotal = local_sum(probabilities)
        subtotal = subtotal + subtotal.shuffle_xor(32, 64)
        own_p = pack(probabilities).bitcast(fx.Int32)
        ballot = fx.Int64(rocdl.ballot(fx.Int64.ir_type, advance.ir_value()))
        o0, o1, total = rescale(o0, o1, total, maximum, new_max, ballot)
        maximum = advance_max(maximum, new_max)
        publish(_pin_i32(shared + _p_write_offset(_pin_i32(tid))), own_p)
        write_scalar(_pin_i32(shared + SUM_BASE + _pin_i32(tid) * 4), subtotal)
        wait(lgkmcnt=0)
        sync()  # 3: BF16 P and FP32 row sums are published, not independent softmaxes.
        peer_p = read_p(_pin_i32(shared + _p_peer_offset(_pin_i32(tid))))
        peer_sum = read_scalar(_pin_i32(shared + SUM_BASE + (_pin_i32(tid) ^ 256) * 4))
        v0 = read_v(_pin_i32(shared + _v_read_offset(_pin_i32(tid) & 63, half, 0)))
        wait(lgkmcnt=0)
        rocdl.sched_barrier(0)
        p = merge_p(own_p, peer_p, half)
        total = total + (subtotal + peer_sum)
        o0 = pv(p, tail(v0, kv_len - tile * BN), o0)
        v1 = read_v(_pin_i32(shared + _v_read_offset(_pin_i32(tid) & 63, half, 1)))
        # Prefetch after score/P retirement; all speculative page IDs are clamped.
        next_tile = _min(tile + 1, last)
        page0, page1 = pages(table, next_tile, last, kv_len, PAGE)
        page0, page1 = ready(page0), ready(page1)
        kp = load_k(gk, klane, page0, next_tile, HK, PAGE, page1)
        vp = load_v(vbase, vlane, page0, next_tile, HK, PAGE, page1)
        wait(lgkmcnt=0)
        sync()  # 4: all P, statistics and V readers have retired before overwrite.
        o1 = pv(p, tail(v1, kv_len - tile * BN), o1)
        wait(vmcnt=0)
        rocdl.sched_barrier(0)
        publish(_pin_i32(shared + _k_write_offset(_pin_i32(tid))), kp)
        publish(_pin_i32(shared + K_BYTES + _pin_i32(tid) * 16), vp)
        wait(lgkmcnt=0)
        sync()  # 5: next KV is visible to both halves of every pair.

    o0, o1, total, maximum = fx.Vector(o0), fx.Vector(o1), fx.Float32(total), fx.Float32(maximum)
    inv = (total > 0.0).select(fx.Float32(1.0) / total, fx.Float32(0.0)) * VS[0]
    optr = fx.get_iter(O) + ((fx.Int64(q0) + fx.Int64(q_start)) * H * D + fx.Int64(head) * D)
    obuf = rocdl.make_buffer_tensor(fx.make_view(optr, fx.make_layout(BM * H * D, 1)), num_records_bytes=valid * H * D * 2)
    atom = fx.make_copy_atom(rocdl.BufferCopy64b(), fx.BFloat16)
    for part in fx.range_constexpr(2):
        output = o0 if part == 0 else o1
        for n in fx.range_constexpr(8):
            values = fx.Vector.from_elements([output[n * 4 + i] * inv for i in range(4)], fx.Float32)
            src = fx.make_rmem_tensor(4, fx.BFloat16)
            src.store(_pack_bf16(values).bitcast(fx.BFloat16))
            col = half * 128 + part * 64 + (n // 4) * 32 + (n % 4) * 8 + (lane >> 5) * 4
            fx.copy(atom, src, fx.make_view(fx.get_iter(obuf) + localrow * H * D + col, fx.make_layout(4, 1)))
    if fx.const_expr(WITH_LSE):
        if (half == 0) & (lane < 32) & (row < q_len):
            log_l = fx.Float32(llvm.call_intrinsic(fx.Float32.ir_type, "llvm.log2.f32", [total.ir_value()], [], []))
            LSE[(q0 + row) * H + head] = (total > 0.0).select(
                (maximum + log_l) * fx.Float32(math.log(2.0)), fx.Float32(float("-inf")))
    wait(vmcnt=0, lgkmcnt=0)
    sync()  # Close output traffic before persistent task reuse.


@flyc.jit
def _work(Q, K, V, O, LSE, CQ, KI, PAGES, LAST, QS, KS, VS, storage, head, batch, qb,
          H: fx.Constexpr[int], HK: fx.Constexpr[int], NP: fx.Constexpr[int], PAGE: fx.Constexpr[int],
          CAUSAL: fx.Constexpr[bool], PER_TOKEN: fx.Constexpr[bool], WITH_LSE: fx.Constexpr[bool], SCALE: fx.Constexpr[float]):
    body = _body
    q0 = _uniform(CQ[batch])
    q_len = _uniform(CQ[batch + 1]) - q0
    start = _uniform(KI[batch])
    count = _uniform(KI[batch + 1]) - start
    kv_len = (count - 1) * PAGE + _uniform(LAST[batch])
    table = fx.make_view(fx.get_iter(PAGES) + start, fx.make_layout(count, 1))
    if qb * BM < q_len:
        body(Q, K, V, O, LSE, QS, KS, VS, table, storage, q0, q_len, kv_len, head, qb,
             H, HK, NP, PAGE, CAUSAL, PER_TOKEN, WITH_LSE, SCALE)


@flyc.kernel(known_block_size=[THREADS, 1, 1])
def _attention_256_pexchange_kernel_942(Q: fx.Tensor, K: fx.Tensor, V: fx.Tensor, O: fx.Tensor, LSE: fx.Tensor,
    CQ: fx.Tensor, KI: fx.Tensor, PAGES: fx.Tensor, LAST: fx.Tensor, QS: fx.Tensor, KS: fx.Tensor, VS: fx.Tensor,
    H: fx.Constexpr[int], HK: fx.Constexpr[int], NP: fx.Constexpr[int], B: fx.Constexpr[int],
    MAX_Q: fx.Constexpr[int], PAGE: fx.Constexpr[int], CAUSAL: fx.Constexpr[bool],
    PER_TOKEN: fx.Constexpr[bool], WITH_LSE: fx.Constexpr[bool], SCALE: fx.Constexpr[float],
    PERSISTENT: fx.Constexpr[bool], CUS: fx.Constexpr[int]):
    work_body = _work
    storage = fx.SharedAllocator().allocate(fx.Array[fx.Int8, LDS_BYTES, 16]).peek().view(fx.make_layout(LDS_BYTES, 1))
    if fx.const_expr(PERSISTENT):
        task = fx.Int32(gpu.block_id("x"))
        while task < H * B * ((MAX_Q + BM - 1) // BM):
            task = fx.Int32(task)
            head, batch, qb = task % H, (task // H) % B, task // (H * B)
            work_body(Q, K, V, O, LSE, CQ, KI, PAGES, LAST, QS, KS, VS, storage, head, batch, qb,
                      H, HK, NP, PAGE, CAUSAL, PER_TOKEN, WITH_LSE, SCALE)
            task = task + CUS
    else:
        work_body(Q, K, V, O, LSE, CQ, KI, PAGES, LAST, QS, KS, VS, storage,
                  fx.Int32(gpu.block_id("x")), fx.Int32(gpu.block_id("y")), fx.Int32(gpu.block_id("z")),
                  H, HK, NP, PAGE, CAUSAL, PER_TOKEN, WITH_LSE, SCALE)


@flyc.jit
def _launch_attention_256_pexchange(Q: fx.Tensor, K: fx.Tensor, V: fx.Tensor, O: fx.Tensor, LSE: fx.Tensor,
    CQ: fx.Tensor, KI: fx.Tensor, PAGES: fx.Tensor, LAST: fx.Tensor, QS: fx.Tensor, KS: fx.Tensor, VS: fx.Tensor,
    H: fx.Constexpr[int], HK: fx.Constexpr[int], NP: fx.Constexpr[int], B: fx.Constexpr[int],
    MAX_Q: fx.Constexpr[int], DQ: fx.Constexpr[int], DV: fx.Constexpr[int], PAGE: fx.Constexpr[int],
    CAUSAL: fx.Constexpr[bool], PER_TOKEN: fx.Constexpr[bool], WITH_LSE: fx.Constexpr[bool],
    SCALE: fx.Constexpr[float], PERSISTENT: fx.Constexpr[bool], CUS: fx.Constexpr[int], stream: fx.Stream):
    assert DQ == DV == D
    grid = (min(CUS, H * B * ((MAX_Q + BM - 1) // BM)), 1, 1) if PERSISTENT else (H, B, (MAX_Q + BM - 1) // BM)
    _attention_256_pexchange_kernel_942(Q, K, V, O, LSE, CQ, KI, PAGES, LAST, QS, KS, VS,
        H, HK, NP, B, MAX_Q, PAGE, CAUSAL, PER_TOKEN, WITH_LSE, SCALE, PERSISTENT, CUS,
        value_attrs={"rocdl.waves_per_eu": 2, "passthrough": [["target-features", "-packed-fp32-ops"]]},
    ).launch(grid=grid, block=(THREADS, 1, 1), stream=stream)


@functools.cache
def PagedAttention(num_qo_heads, num_kv_heads, head_dim_qk, head_dim_v, page_size,
                   is_causal, quant_query_mode="per-token", key_layout="vectorized",
                   window_left=-1, has_sink=False, *, memory_mode="lds", persistent=None):
    """Opt-in M32 paired-wave D256 implementation; same public BF16 contract."""
    validated = _validate_factory(num_qo_heads, num_kv_heads, head_dim_qk, head_dim_v, page_size,
        is_causal, quant_query_mode, key_layout, window_left, has_sink,
        memory_mode=memory_mode, persistent=persistent)
    if head_dim_qk != D or head_dim_v != D:
        raise NotImplementedError("P-exchange requires DQ=DV=256")
    # A fresh wrapper/cache: never change the object cached by the default factory.
    kernel = _PagedAttention(num_qo_heads, num_kv_heads, D, D, page_size,
                             is_causal, quant_query_mode, validated.persistent)
    kernel._launch = _launch_attention_256_pexchange
    kernel.bf16_backend = "native-m32-key-split-p-exchange"
    return kernel