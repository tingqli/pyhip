"""gfx942 FP8-FNUZ paged attention: BM256/BN64, eight staggered wave64s.

K/V use the page64 SHUFFLE-5D ABI. Cooperative buffer loads stage K/V through
packed VGPRs into two K and two V LDS slots: 41,600 bytes for D192 (33,280 for
D128). No direct-LDS DMA, LDS-transpose instructions or native FP32-to-BF16
conversion is used. ``_body`` documents the pipeline; see also readme.md.
"""

import functools
import math

import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu, rocdl
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from pyhip.codegen.flydsl.helpers import rocdl_aux

from ._common import (
    _MAX_FLOOR, _buffer, _center_part, _exp, _exp_part, _join, _max3, _maximum, _mfma_hazard_wait, _min,
    _pack_bf16, _pin, _pin_i32, _read_page_g2r, _reround_if, _row_max, _row_sum, _schedule, _softmax_scale,
    _stage_end, _uniform, _wait,
)

BM, BN, DV, THREADS = 256, 64, 128, 512
PADDED_CHUNK = 1040
V_TILE = 8 * PADDED_CHUNK
LOG2E = math.log2(math.e)


def _page_ready_tied(value):
    # s_waitcnt lgkmcnt(0): make s_load_dword's page ID usable. Also drains
    # this wave's earlier LDS operations; deliberately does NOT wait for VMEM.
    return fx.Int32(llvm.inline_asm(fx.Int32.ir_type, [value.ir_value()],
                                  "s_waitcnt lgkmcnt(0)", "=s,0", has_side_effects=True))


def _pack_step(values, k):
    """FP8 words 2k and 2k + 1 of P (values 8k .. 8k + 7): the PV B operand of K-step k."""
    words = []
    for i in (8 * k, 8 * k + 4):
        # The high-half conversion overwrites bits 16-31, so the low one needs no zeroed input.
        lo = rocdl.cvt_pk_fp8_f32(fx.Int32.ir_type, values[i], values[i + 1], llvm.mlir_poison(fx.Int32.ir_type), False)
        hi = rocdl.cvt_pk_fp8_f32(fx.Int32.ir_type, values[i + 2], values[i + 3], lo, True)
        words.append(fx.Int32(hi))
    return words


def _pack_probability(values, dtype):
    words = [word for k in range(4) for word in _pack_step(values, k)]
    return _pin(fx.Vector.from_elements(words, fx.Int32).bitcast(dtype))


def _mfma(a_words, b_words, acc):
    a = fx.Vector.from_elements(list(a_words), fx.Int32).bitcast(fx.Int64)[0]
    b = fx.Vector.from_elements(list(b_words), fx.Int32).bitcast(fx.Int64)[0]
    return fx.Vector(rocdl.mfma_f32_32x32x16_fp8_fp8_(ir.VectorType.get([16], fx.Float32.ir_type), a.ir_value(),
                                                     b.ir_value(), fx.Vector(acc).ir_value(), 0, 0,
                                                     rocdl._blgp_attr(0)))


def _add(a, b):
    # Opaque scalar add: SLP would form v_pk_add_f32, which waits for in-flight MFMAs.
    return fx.Float32(llvm.inline_asm(fx.Float32.ir_type, [fx.Float32(a).ir_value(), fx.Float32(b).ir_value()],
                                    "v_add_f32 $0, $1, $2", "=v,v,v", has_side_effects=False))


def _max_tree(values):
    """_row_max_local, one MAX3 per step; the last step yields the lane's row max."""
    p = []
    for i in range(0, 30, 3):
        p.append(_max3(values[i], values[i + 1], values[i + 2]))
        yield None
    q = []
    for i in range(0, 9, 3):
        q.append(_max3(p[i], p[i + 1], p[i + 2]))
        yield None
    b = _max3(p[9], values[30], values[31])
    yield None
    a = _max3(q[0], q[1], q[2])
    yield None
    yield _max3(a, b, fx.Float32(_MAX_FLOOR))


def _sum_tree(values):
    """_row_sum_local, one add per step; the last step yields the lane's row sum."""
    partials = []
    for i in range(0, 32, 2):
        partials.append(_add(values[i], values[i + 1]))
        yield None
    for width in (8, 4, 2):
        merged = []
        for i in range(width):
            merged.append(_add(partials[2 * i], partials[2 * i + 1]))
            yield None
        partials = merged
    yield _add(partials[0], partials[1])


def _center(value, scale, maximum):
    return fx.Float32(llvm.inline_asm(fx.Float32.ir_type, [value.ir_value(), scale.ir_value(), maximum.ir_value()],
                                    "v_fma_f32 $0, $1, $2, -$3", "=v,v,v,v", has_side_effects=False))


def _pv_lo_with_max(p_prev, v_words, o0, o1, s_cur):
    """S5 in fenced program order: PV MFMAs of dims 0-63 by K-step; the FP8 conversion of the
    next K-step rides behind the first MFMA and MAX3 of S(t) behind the second, so the tree and
    its xor32 ds_bpermute issue before the last K-step, which takes the first 16 adds of the
    row sum of P(t-1). Returns P words, O, the row max and the row-sum generator."""
    tree = _max_tree(fx.Vector(s_cur))
    sums = _sum_tree(p_prev)
    p_words = _pack_step(p_prev, 0)
    local = other = None
    for k in range(4):
        for m in range(2):
            rocdl.sched_barrier(0)
            i = 8 * m + 2 * k
            acc = _mfma((v_words[i], v_words[i + 1]), p_words[2 * k:2 * k + 2], o1 if m else o0)
            o0, o1 = (o0, acc) if m else (acc, o1)
            rocdl.sched_barrier(0)
            if k == 3:
                for _ in range(8):
                    next(sums)
            elif m == 0:
                p_words += _pack_step(p_prev, k + 1)
            else:
                for _ in range(6 if k == 2 else 5):
                    local = next(tree)
                if k == 2:
                    other = local.shuffle_xor(32, 64)
    row_max = _maximum(local, other)
    rocdl.sched_barrier(0)
    return p_words, o0, o1, row_max, sums


def _pv_hi_with_sum(p_words, v_words, o2, o3, sums, s_cur, scale, maximum):
    """S7 in fenced program order: six VALU after each PV MFMA (dims 64-127): the rest of the
    row sum, its xor32 ds_bpermute, then S'(t). Returns O, the two partial sums and S'(t)."""
    s_cur = fx.Vector(s_cur)
    out = {}

    def valu():
        for _ in range(15):
            local = next(sums)
            yield
        out["local"], out["other"] = local, local.shuffle_xor(32, 64)
        yield
        out["centered"] = []
        for i in range(32):
            out["centered"].append(_center(s_cur[i], scale, maximum))
            yield

    ops = valu()
    for k in range(4):
        for m in range(2):
            rocdl.sched_barrier(0)
            i = 8 * m + 2 * k
            acc = _mfma((v_words[i], v_words[i + 1]), p_words[2 * k:2 * k + 2], o3 if m else o2)
            o2, o3 = (o2, acc) if m else (acc, o3)
            rocdl.sched_barrier(0)
            for _ in range(6):
                next(ops)
    rocdl.sched_barrier(0)
    centered = _pin(fx.Vector.from_elements(out["centered"], fx.Float32), 32)
    return o2, o3, out["local"], out["other"], centered


def _schedule_qk_exp(dq, group):
    # Two exponentials (16 issue cycles each) per dependent QK MFMA link; D192 has 12 links for 16.
    links = dq // 16
    _schedule(16 - links, 2, group, True)
    _schedule(2 * links - 16, 1, group, True)


def _read_g2r(resource, voffset, soffset, count=4):
    return fx.Vector(rocdl.raw_ptr_buffer_load(ir.VectorType.get([count], fx.Int32.ir_type),
                                              resource, fx.Int32(voffset).ir_value(),
                                              fx.Int32(soffset).ir_value(), aux=rocdl_aux(0)))


def _read_s2r(storage, base, immediate=0):
    # Async inline LDS read: the caller must s_waitcnt lgkmcnt(0) before use.
    address = fx.Int32(fx.ptrtoint(fx.get_iter(storage) + base))
    return fx.Vector(llvm.inline_asm(ir.VectorType.get([4], fx.Int32.ir_type), [address.ir_value()],
        f"ds_read_b128 $0, $1 offset:{immediate}", "=v,v,~{memory}", has_side_effects=True))


def _read_kv_g2r(resource, tid, physical_offset, rounds):
    """Cooperative copy of one page-64 K or V tile into staging VGPRs (8 bytes per thread and round)."""
    parts = [_read_g2r(resource, (tid + i * THREADS) * 8, physical_offset, 2) for i in range(rounds)]
    return fx.Vector.from_elements([part[j] for part in parts for j in range(2)], fx.Int32)


def _store_kv_r2s(storage, tid, words, slot_offset, rounds, key=False):
    """Publish staged K/V words to an LDS slot; 1024-byte chunks are padded to 1040 bytes."""
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


def _read_q_g2r(resource, row_offset, lane, dq, dtype):
    parts = [_read_g2r(resource, row_offset + (lane >> 5) * 16 + d * 32, 0) for d in range(dq // 32)]
    words = fx.Vector.from_elements([p[i] for p in parts for i in range(4)], fx.Int32)
    frag = _mma(dtype).make_fragment_B(fx.make_rmem_tensor(fx.make_layout((BM, dq), (1, BM)), dtype))
    frag.store(words.bitcast(dtype))
    return frag


def _read_k_s2r(storage, base, slot, half, dq, dtype):
    """Keys 32 * half .. 32 * half + 31 of the K tile in LDS slot `slot`, as the QK A operand."""
    k_tile = (dq // 16) * PADDED_CHUNK
    parts = [_read_s2r(storage, slot * k_tile + base + half * 512, d * 2080) for d in range(dq // 32)]
    words = fx.Vector.from_elements([p[i] for p in parts for i in range(4)], fx.Int32)
    frag = _mma(dtype).make_fragment_A(fx.make_rmem_tensor(fx.make_layout((32, dq), (1, 32)), dtype))
    frag.store(words.bitcast(dtype))
    return frag


def _read_v_words(storage, base, slot, half, dq):
    """Dims 64 * half .. 64 * half + 63 of the V tile in LDS slot `slot`: 16 words, MFMA (k, m) uses 8m + 2k, +1."""
    k_tile = (dq // 16) * PADDED_CHUNK
    parts = [_read_s2r(storage, 2 * k_tile + slot * V_TILE + base + half * PADDED_CHUNK,
                       n * 512 + k * 4160) for n in range(2) for k in range(2)]
    return fx.Vector.from_elements([p[i] for p in parts for i in range(4)], fx.Int32)


def _read_v_s2r(storage, base, slot, half, dq, dtype):
    """_read_v_words as the PV A operand fragment."""
    frag = fx.make_rmem_tensor(fx.make_layout((8, 4, 2), (1, 8, 32)), dtype)
    frag.store(_read_v_words(storage, base, slot, half, dq).bitcast(dtype))
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


def _unlikely(condition):
    # Branch weights: LLVM lays the guarded block out of the fall-through path. One hinted
    # branch for rescale and exact rule measured +0.6% over the BF16 kernels' two branches.
    return fx.Boolean(llvm.intr_expect(fx.Boolean(condition).ir_value(), fx.Boolean(False).ir_value()))


@flyc.jit
def _rescale(o0, o1, o2, o3, row_sum, s_cur, old_max, new_max, ballot, exact):
    """Lazy rescale of O and l when m moved, plus the exact rule for S'(t).

    The FMA S * scale - m is up to ulp(m) / 2 off the rounded m. Once some |m| >= 2^22
    (checked when m moves, then sticky), S'(t) is also rounded to a multiple of ulp(m).
    """
    o0, o1, o2, o3 = fx.Vector(o0), fx.Vector(o1), fx.Vector(o2), fx.Vector(o3)
    row_sum, s_cur, ballot, exact = fx.Float32(row_sum), fx.Vector(s_cur), fx.Int64(ballot), fx.Int64(exact)
    if _unlikely((ballot | exact) != fx.Int64(0)):
        if ballot != fx.Int64(0):
            correction = _exp(fx.Float32(old_max) - fx.Float32(new_max))
            o0, o1, o2, o3 = o0 * correction, o1 * correction, o2 * correction, o3 * correction
            row_sum = row_sum * correction
            big = fx.absf(fx.Float32(new_max)) >= 2.0**22
            exact = exact | fx.Int64(rocdl.ballot(fx.Int64.ir_type, big.ir_value()))
        s_cur = _reround_if(s_cur, new_max, exact)
    return o0, o1, o2, o3, row_sum, s_cur, exact


def _store_o_tile_r2g(outputs, inv, storage, obuf, out_tid, out_row, heads):
    """O * inv to BF16, C-shuffled through LDS (r2s b64, s2r b128), then b128 row stores (r2g).

    Reuses the first 32 KiB of LDS for one 64-column half at a time. XOR row[3:0] into
    column[5:2] keeps the b64 writes conflict-free; odd rows swap two packed-word pairs.
    """
    atom = fx.make_copy_atom(rocdl.BufferCopy128b(), fx.BFloat16)
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
        # All writers' b64 stores complete before any wave reads this half back.
        _wait(lgkmcnt=0)
        _stage_end()
        for i in fx.range_constexpr(4):
            element = out_tid * 8 + i * THREADS * 8
            read_row, read_col = element // 64, element % 64
            element = element ^ ((read_row & 14) * 4)
            words = _read_s2r(storage, element * 2)
            _wait(lgkmcnt=0)
            rocdl.sched_barrier(0)
            src = fx.make_rmem_tensor(8, fx.BFloat16)
            src.store(fx.Vector.from_elements([
                ((read_row & 1) == 0).select(words[j], words[j ^ 2]) for j in range(4)
            ], fx.Int32).bitcast(fx.BFloat16))
            offset = read_row * (heads * DV) + half * 64 + read_col
            fx.copy(atom, src, fx.make_view(fx.get_iter(obuf) + offset, fx.make_layout(8, 1)))
        # Readers rendezvous before the next half overwrites LDS (not a VMEM-store drain).
        _stage_end()


@flyc.jit
def _body(Q, K, V, O, LSE, QS, KS, VS, table, storage, q0, q_len, kv_len, head, qb,
          H: fx.Constexpr[int], HK: fx.Constexpr[int], NP: fx.Constexpr[int], DQ: fx.Constexpr[int],
          CAUSAL: fx.Constexpr[bool], PER_TOKEN: fx.Constexpr[bool], WITH_LSE: fx.Constexpr[bool],
          SCALE: fx.Constexpr[float], STAGGER: fx.Constexpr[bool]):
    """One task: query rows 256 * qb .. 256 * qb + 255 of one head over KV tiles t of BN = 64 keys.

    LDS holds two K and two V slots; tile t uses slot t & 1. Page ids run three tiles
    ahead of QK, K two tiles ahead (g2r in S0, r2s in S4 of phase t - 2) and V one tile
    ahead (g2r in S0, r2s in S2 of phase t - 1). Phase t runs QK and the row max of
    tile t and, one tile behind, the exponentials, row sum and PV of tile t - 1, in
    eight stages: even stages are memory (VMEM/LDS), odd stages compute (MFMA/VALU).
    Waves 4-7 (STAGGER) run one stage behind waves 0-3, so one group's memory stage
    overlaps the other group's compute stage.

    stage  phase t; cur = t & 1, prev = (t - 1) & 1
    S0     s2r K(t) keys 0-31 [cur]; s_load page(t+3); g2r V(t), K(t+2)
    S1     QK: S(t) keys 0-31          softmax: P(t-1) = exp2(S'(t-1)), values 0-15
    S2     s2r K(t) keys 32-63 [cur]; V(t) lands; r2s V(t) -> [cur]
    S3     QK: S(t) keys 32-63         softmax: P(t-1) values 16-31
    S4     s2r V(t-1) dims 0-63 [prev]; K(t+2) lands; r2s K(t+2) -> [cur]; mask S(t)
    S5     PV: O dims 0-63 += P V      softmax: P(t-1) to FP8, row max of S(t), row-sum adds
    S6     s2r V(t-1) dims 64-127 [prev]; new m
    S7     PV: O dims 64-127 += P V    softmax: l += row sum, S'(t); rescale O and l

    Compute stages are bound by the VALU issue port (an exp2 takes 16 cycles, a
    conversion 8): S1/S3 fit two exponentials per link of their dependent QK MFMA
    chain, and S5/S7 issue the PV MFMAs in fenced program order with the rest of the
    softmax between them. Memory stages keep only short VALU work, since the other
    group's compute stage on the same SIMD starves it.

    Softmax in log2 units: S' = S * scale - m (one FMA), P = exp2(S'), l = sum(P) and
    O = sum(P V); the output is O / l * v_descale. m moves to a larger row max only when
    that exceeds m by more than 6 (FP8-FNUZ keeps P finite), rescaling O and l. The FMA
    is up to ulp(m) / 2 off the rounded m: once some |m| >= 2^22 (checked when m moves,
    then sticky), S7's lazy-rescale branch also rounds S'(t) to a multiple of ulp(m),
    which matches the D256 kernels' rounded products.
    """
    # FlyDSL keys its disk cache on helpers named in this function's own code, not in closures.
    _helpers = (_read_kv_g2r, _store_kv_r2s, _read_k_s2r, _read_v_s2r, _page_ready_tied, _qk, _pv, _mask,
                _v_tail, _join, _mfma_hazard_wait, _maximum, _row_max, _row_sum, _center_part, _exp_part,
                _pack_probability, _rescale, _pin, _schedule, _stage_end, _wait, _schedule_qk_exp, _read_v_words,
                _pack_step, _mfma, _add, _max3, _max_tree, _sum_tree, _center, _pv_lo_with_max, _pv_hi_with_sum,
                _reround_if)
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
    # Q stays in registers for every KV tile.
    q = _read_q_g2r(gq, (wave * 32 + (lane & 31)) * (H * DQ), lane, DQ, dtype)
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
    k_slot = (DQ // 16) * PADDED_CHUNK
    v_slot = (2 * k_slot, 2 * k_slot + V_TILE)
    # LDS read bases: K rows in 32x32-MFMA order with the bank XOR; V rows per lane.
    k_row = (lane & 3) | ((lane & 4) << 2) | ((lane & 24) >> 1)
    k_s2r_base = (lane >> 5) * PADDED_CHUNK + ((k_row * 16) ^ ((k_row & 16) << 2))
    v_s2r_base = (lane >> 5) * (2 * PADDED_CHUNK) + (lane & 31) * 16

    def read_k_g2r(page):
        return _read_kv_g2r(gk, tid, page * (HK * BN * DQ), DQ // 64)

    def read_v_g2r(page):
        return _read_kv_g2r(gv, tid, page * (HK * BN * DV), 2)

    def prologue():
        """Pages 0-3; K(0) -> [0], K(1) -> [1], V(0) -> [0]; QK S(0), m and S'(0); K(2) -> [0]."""
        page0 = _read_page_g2r(table, fx.Int32(0))
        page1 = _read_page_g2r(table, _min(fx.Int32(1), last))
        page2 = _read_page_g2r(table, _min(fx.Int32(2), last))
        # The first lgkmcnt(0) already covers all three requests; no VMEM wait.
        page0, page1, page2 = _page_ready_tied(page0), _page_ready_tied(page1), _page_ready_tied(page2)
        # Memory: g2r K(0), K(1), V(0); r2s into K slots 0/1 and V slot 0.
        k_0, k_1, v_0 = read_k_g2r(page0), read_k_g2r(page1), read_v_g2r(page0)
        _wait(vmcnt=0)
        _store_kv_r2s(storage, tid, k_0, 0, DQ // 64, True)
        _store_kv_r2s(storage, tid, k_1, k_slot, DQ // 64, True)
        _store_kv_r2s(storage, tid, v_0, v_slot[0], 2)
        _wait(vmcnt=0, lgkmcnt=0)
        _stage_end()
        if fx.const_expr(STAGGER):
            # Waves 4-7 fall one stage behind.
            _stage_end()
        # Memory: s2r K(0) keys 0-31.
        k_lo = _read_k_s2r(storage, k_s2r_base, 0, 0, DQ, dtype)
        _wait(vmcnt=0, lgkmcnt=0)
        _stage_end()
        # Compute: QK S(0) keys 0-31; O = 0.
        s_lo = _qk(q, k_lo, dtype)
        o0 = fx.Vector.filled(16, 0.0, fx.Float32)
        o1 = fx.Vector.filled(16, 0.0, fx.Float32)
        o2 = fx.Vector.filled(16, 0.0, fx.Float32)
        o3 = fx.Vector.filled(16, 0.0, fx.Float32)
        _schedule(DQ // 16, 3, 5)
        o0, o1, o2, o3 = _pin(o0), _pin(o1), _pin(o2), _pin(o3)
        _stage_end()
        # Memory: s2r K(0) keys 32-63.
        k_hi = _read_k_s2r(storage, k_s2r_base, 0, 1, DQ, dtype)
        _wait(vmcnt=0, lgkmcnt=0)
        _stage_end()
        # Compute: QK S(0) keys 32-63; softmax m and S'(0).
        s_hi = _qk(q, k_hi, dtype)
        s_0 = _mask(_join(s_lo, s_hi), fx.Int32(0), row, q_len, kv_len, CAUSAL)
        _mfma_hazard_wait()
        row_max = _maximum(_row_max(s_0) * scale, fx.Float32(_MAX_FLOOR))
        s_0 = _center_part(s_0, scale, row_max, 0, 32)
        # The exact rule of _rescale for S'(0); exact carries it through the phases.
        exact = fx.Int64(rocdl.ballot(fx.Int64.ir_type, (fx.absf(row_max) >= 2.0**22).ir_value()))
        s_0 = _reround_if(s_0, row_max, exact)
        row_sum = fx.Float32(0.0)
        _stage_end()
        # Memory: s_load page(3); K(2) g2r and r2s into the retired slot 0.
        page3_request = _read_page_g2r(table, _min(fx.Int32(3), last))
        k_2 = read_k_g2r(page2)
        _wait(vmcnt=0)
        _store_kv_r2s(storage, tid, k_2, 0, DQ // 64, True)
        # lgkmcnt(0) completes the K(2) writes and makes page(3) usable.
        page3 = _page_ready_tied(page3_request)
        _stage_end()
        return s_0, row_max, row_sum, o0, o1, o2, o3, page0, page1, page2, page3, exact

    @flyc.jit
    def phase(s_prev, row_max, row_sum, o0, o1, o2, o3, page_prev, page_cur, page_next, page_next2, exact, t,
              CUR: fx.Constexpr[int], PREV: fx.Constexpr[int]):
        """Phase t (table in _body): page_prev .. page_next2 are the page ids of tiles t-1 .. t+2.

        CUR = t & 1 is the slot of K(t), V(t) and K(t+2); PREV holds V(t-1). exact carries
        the exact rule of _rescale.
        """
        s_prev, row_max, row_sum = fx.Vector(s_prev), fx.Float32(row_max), fx.Float32(row_sum)
        o0, o1, o2, o3 = fx.Vector(o0), fx.Vector(o1), fx.Vector(o2), fx.Vector(o3)
        exact, t = fx.Int64(exact), fx.Int32(t)
        # S0 memory: s2r K(t) keys 0-31 first; its lgkmcnt wait comes last.
        k_lo = _read_k_s2r(storage, k_s2r_base, CUR, 0, DQ, dtype)
        # Compiler-only fence: keep address setup and VMEM after the LDS reads.
        rocdl.sched_barrier(0)
        page_next3_request = _read_page_g2r(table, _min(t + 3, last))
        v_cur = read_v_g2r(page_cur)
        k_next2 = read_k_g2r(page_next2)
        # V(t) must precede K(t+2) in VMEM order for the S2 rolling vmcnt.
        rocdl.sched_barrier(0)
        # lgkmcnt(0): K(t) keys 0-31 and page(t+3) ready; V(t)/K(t+2) stay in flight.
        page_next3 = _page_ready_tied(page_next3_request)
        _stage_end()
        # S1 compute: QK S(t) keys 0-31; P(t-1) values 0-15.
        s_lo = _qk(q, k_lo, dtype)
        p_prev = _pin(_exp_part(s_prev, 0, 16), 32)
        _schedule_qk_exp(DQ, 1)
        _stage_end()
        # S2 memory: s2r K(t) keys 32-63; r2s V(t) once it lands.
        k_hi = _read_k_s2r(storage, k_s2r_base, CUR, 1, DQ, dtype)
        # vmcnt(DQ // 64): V(t)'s two loads retired; the newer K(t+2) loads stay outstanding.
        _wait(vmcnt=DQ // 64)
        _store_kv_r2s(storage, tid, v_cur, v_slot[CUR], 2)
        _wait(lgkmcnt=0)
        _stage_end()
        # S3 compute: QK S(t) keys 32-63; P(t-1) values 16-31.
        s_hi = _qk(q, k_hi, dtype)
        p_prev = _pin(_exp_part(p_prev, 16, 32), 32)
        _schedule_qk_exp(DQ, 2)
        _stage_end()
        # S4 memory: s2r V(t-1) dims 0-63; r2s K(t+2) into slot CUR, which both groups have read; mask S(t).
        v_lo = _read_v_words(storage, v_s2r_base, PREV, 0, DQ)
        _wait(vmcnt=0)
        _store_kv_r2s(storage, tid, k_next2, CUR * k_slot, DQ // 64, True)
        # lgkmcnt(0): V(t-1) dims 0-63 ready, K(t+2) published, pending shuffles done.
        _wait(lgkmcnt=0)
        s_cur = _mask(_join(s_lo, s_hi), t, row, q_len, kv_len, CAUSAL)
        _stage_end()
        # S5 compute: FP8 P(t-1); PV O dims 0-63; row max of S(t); first adds of the row sum.
        p_words, o0, o1, local_max, sums = _pv_lo_with_max(p_prev, v_lo, o0, o1, s_cur)
        _stage_end()
        # S6 memory: s2r V(t-1) dims 64-127 (V(t-1) is a full page in every main phase); lazy new max.
        v_hi = _read_v_words(storage, v_s2r_base, PREV, 1, DQ)
        rocdl.sched_barrier(0)
        candidate = local_max * scale
        ballot = fx.Int64(rocdl.ballot(fx.Int64.ir_type, (candidate - row_max > 6.0).ir_value()))
        new_max = (ballot != fx.Int64(0)).select(_maximum(row_max, candidate), row_max)
        rocdl.sched_barrier(0)
        _wait(vmcnt=0, lgkmcnt=0)
        _stage_end()
        # S7 compute: PV O dims 64-127; l += row sum of P(t-1); S'(t); lazy rescale of O and l.
        o2, o3, local_sum, other_sum, s_cur = _pv_hi_with_sum(p_words, v_hi, o2, o3, sums, s_cur, scale, new_max)
        row_sum = row_sum + (local_sum + other_sum)
        o0, o1, o2, o3, row_sum, s_cur, exact = _rescale(o0, o1, o2, o3, row_sum, s_cur, row_max, new_max, ballot,
                                                         exact)
        _stage_end()
        return s_cur, new_max, row_sum, o0, o1, o2, o3, page_cur, page_next, page_next2, page_next3, exact

    def epilogue(state):
        """exp, row sum and PV of the last tile; V(last) already sits in slot last & 1."""
        s_last, row_max, row_sum, o0, o1, o2, o3 = state[:7]
        p_last = _exp_part(fx.Vector(s_last), 0, 32)
        row_sum = fx.Float32(row_sum) + _row_sum(p_last)
        p_fp8 = _pack_probability(p_last, dtype)
        # Drain outstanding prefetches and shuffles at the loop -> last-tile boundary.
        _wait(vmcnt=0, lgkmcnt=0)
        _stage_end()
        # Memory: s2r V(last) dims 0-63; zero rows past kv_len (0 * NaN is NaN).
        v_lo = _read_v_s2r(storage, v_s2r_base, last & 1, 0, DQ, dtype)
        _wait(vmcnt=0, lgkmcnt=0)
        rocdl.sched_barrier(0)
        v_lo.store(_v_tail(v_lo.load(), kv_len - last * BN))
        _stage_end()
        # Compute: PV O dims 0-63.
        o0, o1 = _pv(p_fp8, v_lo, o0, o1, dtype)
        _stage_end()
        # Memory: s2r V(last) dims 64-127 with the same tail.
        v_hi = _read_v_s2r(storage, v_s2r_base, last & 1, 1, DQ, dtype)
        _wait(vmcnt=0, lgkmcnt=0)
        rocdl.sched_barrier(0)
        v_hi.store(_v_tail(v_hi.load(), kv_len - last * BN))
        _stage_end()
        # Compute: PV O dims 64-127.
        o2, o3 = _pv(p_fp8, v_hi, o2, o3, dtype)
        _stage_end()
        if fx.const_expr(not STAGGER):
            # Waves 0-3 wait for waves 4-7 before the output shuffle reuses LDS.
            _stage_end()
        return row_max, row_sum, (o0, o1, o2, o3)

    # Pipeline: prologue stages K(0..2)/V(0) and runs QK(0); phase t = 1 .. tiles - 1
    # (two per trip, swapping the constant slot indices); epilogue finishes PV(last).
    state = prologue()
    for t in range(fx.Int32(1), tiles - 1, fx.Int32(2)):
        state = phase(*state, t, 1, 0)
        state = phase(*state, t + 1, 0, 1)
    if (tiles & 1) == 0:
        state = phase(*state, last, 1, 0)
    row_max, row_sum, outputs = epilogue(state)

    inv = (row_sum > 0.0).select(fx.Float32(1.0) / row_sum, fx.Float32(0.0)) * VS[0]
    out_tid = _pin_i32(fx.Int32(gpu.thread_id("x")))
    out_row = (out_tid >> 6) * 32 + (out_tid & 31)
    optr = fx.get_iter(O) + ((fx.Int64(q0) + fx.Int64(q_start)) * (H * DV) + fx.Int64(head) * DV)
    obuf = rocdl.make_buffer_tensor(fx.make_view(optr, fx.make_layout(BM * H * DV, 1)), num_records_bytes=valid * H * DV * 2)
    _store_o_tile_r2g(outputs, inv, storage, obuf, out_tid, out_row, H)
    if fx.const_expr(WITH_LSE):
        if ((out_tid & 63) < 32) & (out_row < valid):
            log_l = fx.Float32(llvm.call_intrinsic(fx.Float32.ir_type, "llvm.log2.f32", [row_sum.ir_value()], [], []))
            LSE[(q0 + q_start + out_row) * H + head] = (row_sum > 0.0).select(
                (fx.Float32(row_max) + log_l) * fx.Float32(math.log(2.0)), fx.Float32(float("-inf")))


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
        CAUSAL, PER_TOKEN, WITH_LSE, SCALE,
        value_attrs={"rocdl.waves_per_eu": 2,
                     "llvm.target_features": ir.Attribute.parse('#llvm.target_features<["-packed-fp32-ops"]>')},
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

        Descales must be finite and strictly positive, and softmax_scale * log2(e)
        * q_descale * k_descale a positive finite FP32. K/V within each KV length
        must be finite: rows of a causal tile share its V, so a non-finite value
        also reaches rows that mask its key. Prefix sums/page IDs, last-page
        lengths and maximum lengths must describe the current input; they are
        never synchronized back to the host. Warm before graph capture.
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
        scale = _softmax_scale(softmax_scale, self.dq**-0.5)
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
