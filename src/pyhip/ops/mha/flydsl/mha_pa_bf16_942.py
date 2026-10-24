"""gfx942 BF16 paged attention with alternating memory and compute stages.

Dq/Dv 128/192 use BM256/BN64 (32x32x8 MFMA), eight wave64s in two groups that are
staggered by one stage. Each KV tile runs eight stages: even stages issue VMEM/LDS,
odd stages MFMA/VALU, and the two groups interleave them. ``_body_v128`` is the
DV128/no-LSE hot path (LDS addresses prepared outside the loop, V through 64-bit
GLOBAL loads, one K slot and a two-slot V ring: 49,920/58,240 LDS bytes for
D128/D192). ``_body_wide`` keeps the checked LSE/V192 path with cooperative
copies. Dq = Dv = 256 dispatches to mha_pa_bf16_256_paged_942 (BM128/BN64, K/V
DMA). Persistent and ordinary grids share these bodies and the SHUFFLE-5D ABI.

``flash_attn_varlen_func`` is the D256 linear/page1/page4 API backed by
mha_pa_bf16_256_linear_942 (no KV conversion).
"""

import functools
import math
import os
import weakref
from numbers import Real

import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu, rocdl
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm

from ._common import (
    _MAX_FLOOR, _advance_max, _buffer, _buffer_words, _center_part, _exp, _exp_part, _join, _maximum,
    _mfma_hazard_wait, _min, _pack_bf16, _page_ready, _pin, _pin_i32, _pin_s64, _read_address, _read_page_g2r,
    _reround_if, _row_max, _row_max_local, _row_sum, _row_sum_local, _schedule, _softmax_scale, _stage_end,
    _uniform, _wait,
)

BM, BN, THREADS = 256, 64, 512
PADDED_CHUNK = 1040
V_SLOT = 16640  # one DV128 V tile: 16 padded 1040-byte chunks
LOG2E = math.log2(math.e)
# P to BF16: RNE like O, or ties away from zero ("1": one VALU less per value in QK-high). Read at import.
_P_BF16_RTA = os.environ.get("AITER_FLYDSL_MHA_P_BF16_RTA", "0").lower() in ("1", "true")
# m: lazy, or "1": the running row max (P of the max key is exactly 1 in BF16; a rescale per raised max).
_EXACT_MAX = os.environ.get("AITER_FLYDSL_MHA_EXACT_MAX", "0").lower() in ("1", "true")


def _cross32_async(value, address):
    # Issue in a memory stage; consume only after its explicit LGKM wait and
    # compiler/CTA boundary. This shuffle does not allocate LDS storage.
    return fx.Float32(llvm.inline_asm(fx.Float32.ir_type, [address.ir_value(), value.ir_value()],
        "ds_bpermute_b32 $0, $1, $2", "=v,v,v,~{memory}", has_side_effects=True))


def _pack_probability(values):
    if not _P_BF16_RTA:
        return _pack_bf16(values).bitcast(fx.BFloat16)
    bits = fx.Vector(values).bitcast(fx.Uint32) + fx.Uint32(0x8000)
    selector = fx.Int32(0x07060302)
    words = [fx.Int32(llvm.inline_asm(
        fx.Int32.ir_type, [bits[i + 1].ir_value(), bits[i].ir_value(), selector.ir_value()],
        "v_perm_b32 $0, $1, $2, $3", "=v,v,v,s", has_side_effects=True,
    )) for i in range(0, 32, 2)]
    return fx.Vector.from_elements(words, fx.Int32).bitcast(fx.BFloat16)


def _first_max(scores, scale):
    row_max = _maximum(_row_max(scores) * scale, fx.Float32(_MAX_FLOOR))
    return row_max if _EXACT_MAX else row_max + 1.0


def _next_max(candidate, row_max):
    """Ballot of the rows whose m moves, and the new m: lazily the row max + 1 once it exceeds m + 7."""
    if _EXACT_MAX:
        ballot = fx.Int64(rocdl.ballot(fx.Int64.ir_type, (candidate > row_max).ir_value()))
        return ballot, (candidate > row_max).select(candidate, row_max)
    ballot = fx.Int64(rocdl.ballot(fx.Int64.ir_type, (candidate > row_max + 7.0).ir_value()))
    return ballot, (candidate > row_max + 7.0).select(candidate + 1.0, row_max)


def _read_tile_pages_g2r(table, tile, last, last_page, page_size):
    """Page ids (first, second) of a KV tile: BN64 spans two page32 pages; page128 holds two tiles."""
    # Clamp every speculative SMEM request, including the second page.
    first = _min(tile, last) * BN // page_size
    page0 = _read_page_g2r(table, first)
    page1 = _read_page_g2r(table, _min(first + 1, last_page)) if page_size == 32 else page0
    return page0, page1


def _read_global_g2r(tensor, offset):
    # A genuine 64-bit GLOBAL pointer; never replace V with a raw descriptor.
    pointer = fx.get_iter(tensor) + fx.Int64(offset)
    source = fx.make_view(pointer, fx.make_layout(8, 1))
    fragment = fx.make_rmem_tensor(8, fx.BFloat16)
    fx.copy(fx.make_copy_atom(fx.UniversalCopy128b(), fx.BFloat16), source, fragment)
    return fragment.load().bitcast(fx.Int32)


def _read_s2r(storage, base, immediate=0):
    address = fx.Int32(fx.ptrtoint(fx.get_iter(storage) + base))
    return fx.Vector(llvm.inline_asm(ir.VectorType.get([4], fx.Int32.ir_type), [address.ir_value()],
        f"ds_read_b128 $0, $1 offset:{immediate}", "=v,v,~{memory}", has_side_effects=True))


def _store_r2s(address, words, rounds, immediate=0):
    """ds_write_b128 of `rounds` 16-byte words at a precomputed LDS address, 8320 bytes apart."""
    for i in range(rounds):
        part = fx.Vector.from_elements([words[i * 4 + j] for j in range(4)], fx.Int32)
        llvm.inline_asm(ir.Type.parse("!llvm.void"), [address.ir_value(), part.ir_value()],
            f"ds_write_b128 $0, $1 offset:{immediate + i * 8320}", "v,v,~{memory}", has_side_effects=True)


# ---- DV128 hot path: per-lane addresses are prepared once, stages only issue ----

def _k_g2r_offset(lane_offset, page0, page1, tile, half_page, hk, dq, page_size):
    page = half_page.select(page1, page0) if page_size == 32 else page0
    return _pin_i32(lane_offset + page * (hk * page_size * dq * 2) + (tile * BN % page_size) * 16)


def _read_k_g2r(resource, address, dq, page_size):
    """K tile into staging VGPRs; page32's divergent page sits in VOFFSET, never a scalar waterfall."""
    parts = [_buffer_words(resource, address, i * page_size * 128) for i in range(dq // 64)]
    return fx.Vector.from_elements([p[j] for p in parts for j in range(4)], fx.Int32)


def _v_g2r_bases(base, page0, page1, tile, hk, page_size):
    first = base + fx.Int64(page0) * (hk * page_size * 256) + fx.Int64(tile * BN % page_size) * 256
    second = base + fx.Int64(page1) * (hk * page_size * 256) if page_size == 32 else first + 8192
    return _pin_s64(first), _pin_s64(second)


def _read_v_g2r(first, second, lane_offset):
    # Full 64-bit SGPR address plus lane-relative VGPR byte offset. This is
    # GLOBAL, not BUFFER, and does not truncate the tensor's GPU base.
    parts = [fx.Vector(llvm.inline_asm(ir.VectorType.get([4], fx.Int32.ir_type),
        [lane_offset.ir_value(), address.ir_value()], "global_load_dwordx4 $0, $1, $2",
        "=v,v,s,~{memory}", has_side_effects=True)) for address in (first, second)]
    return fx.Vector.from_elements([p[j] for p in parts for j in range(4)], fx.Int32)


def _read_k_s2r(address, half, dq):
    """Keys 32 * half .. 32 * half + 31 as the QK A operand, from the pinned LDS read address."""
    parts = [_read_address(address, half * 512 + d * 2080) for d in range(dq // 16)]
    words = fx.Vector.from_elements([p[i] for p in parts for i in range(4)], fx.Int32)
    frag = _mma().make_fragment_A(fx.make_rmem_tensor(fx.make_layout((32, dq), (1, 32)), fx.BFloat16))
    frag.store(words.bitcast(fx.BFloat16))
    return frag


def _read_v_s2r(address, half, slot=0):
    """V dims 64 * half .. 64 * half + 63 of ring slot `slot` as the PV A operand."""
    parts = [_read_address(address, slot * V_SLOT + half * 1040 + n * 512 + k * 4160)
             for n in range(2) for k in range(4)]
    words = fx.Vector.from_elements([p[i] for p in parts for i in range(4)], fx.Int32)
    frag = fx.make_rmem_tensor(fx.make_layout((8, 4, 2), (1, 8, 32)), fx.BFloat16)
    frag.store(words.bitcast(fx.BFloat16))
    return frag


# ---- Cooperative tile copies (LSE/V192 path and the D128 prologue) ----

def _read_k_tile_g2r(resource, tid, page0, page1, tile, hk, dq, page_size):
    if dq == 192:
        tid = _pin_i32(fx.Int32(gpu.thread_id("x"))) & 511
    parts = []
    for i in range(dq // 64):
        atom = tid + i * THREADS
        token, column = atom % BN, atom // BN
        page = (token < 32).select(page0, page1) if page_size == 32 else page0
        within = token % page_size + (tile * BN) % page_size
        offset = column * page_size * 16 + within * 16
        if page_size == 32:
            # Different halves of a wave copy different pages. Their page
            # offset belongs in VOFFSET, not scalar SOFFSET (which waterfalls).
            parts.append(_buffer_words(resource, offset + page * (hk * page_size * dq * 2)))
        else:
            parts.append(_buffer_words(resource, offset, page * (hk * page_size * dq * 2)))
    return fx.Vector.from_elements([p[j] for p in parts for j in range(4)], fx.Int32)


def _read_v_tile_g2r(tensor, tid, page0, page1, tile, hkv, hk, dv, page_size, bounded=False):
    if bounded:
        tid = _pin_i32(fx.Int32(gpu.thread_id("x"))) & 511
    parts = []
    for i in range(dv // 64):
        atom = tid + i * THREADS
        token, column = (atom // dv) * 8, atom % dv
        page = (token < 32).select(page0, page1) if page_size == 32 else page0
        within = token % page_size + (tile * BN) % page_size
        offset = ((fx.Int64(page) * hk + fx.Int64(hkv)) * (page_size * dv)
                  + fx.Int64(within // 8) * (dv * 8) + fx.Int64(column) * 8)
        parts.append(_read_global_g2r(tensor, offset))
    return fx.Vector.from_elements([p[j] for p in parts for j in range(4)], fx.Int32)


def _store_tile_r2s(storage, tid, words, slot_offset, rounds, key=False, bounded=False):
    if bounded:
        tid = _pin_i32(fx.Int32(gpu.thread_id("x"))) & 511
    atom = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Int32)
    for i in range(rounds):
        byte = (tid + i * THREADS) * 16
        chunk, within = byte // 1024, byte % 1024
        if key:
            within = within ^ ((within & 256) >> 2)
        offset = slot_offset + chunk * PADDED_CHUNK + within
        pointer = fx.recast_iter(fx.PointerType.get(fx.Int32.ir_type, storage.memspace, 16),
                                 fx.get_iter(storage) + offset)
        src = fx.make_rmem_tensor(4, fx.Int32)
        src.store(fx.Vector.from_elements([words[4 * i + j] for j in range(4)], fx.Int32))
        fx.copy(atom, src, fx.make_view(pointer, fx.make_layout(4, 1)))


def _mma():
    # BF16 atom K8 consumes four values/lane. Pair its reduction steps for
    # b128 copies, rather than reinterpreting the FP8 atom's eight values.
    return fx.make_tiled_mma(fx.make_mma_atom(rocdl.MFMA(32, 32, 8, fx.BFloat16)),
                             fx.make_layout((1, 8, 1), (1, 1, 0)),
                             (None, None, fx.make_layout((4, 2, 2), (1, 8, 4))))


def _read_q_g2r(resource, row_offset, lane, dq):
    parts = [_buffer_words(resource, row_offset + (lane >> 5) * 16 + d * 32) for d in range(dq // 16)]
    words = fx.Vector.from_elements([p[i] for p in parts for i in range(4)], fx.Int32)
    frag = _mma().make_fragment_B(fx.make_rmem_tensor(fx.make_layout((BM, dq), (1, BM)), fx.BFloat16))
    frag.store(words.bitcast(fx.BFloat16))
    return frag


def _read_k_tile_s2r(storage, base, half, dq, dv, full=False):
    """Keys 32 * half .. + 31 from the K slot; D192/V192 without `full` read only the first K64 chunk."""
    width = 64 if (dq == 192 or dv == 192) and not full else dq
    parts = [_read_s2r(storage, base, half * 512 + d * 2080) for d in range(width // 16)]
    words = fx.Vector.from_elements([p[i] for p in parts for i in range(4)], fx.Int32)
    frag = _mma().make_fragment_A(fx.make_rmem_tensor(fx.make_layout((32, width), (1, 32)), fx.BFloat16))
    frag.store(words.bitcast(fx.BFloat16))
    return frag


def _read_v_tile_s2r(storage, base, slot, half, dq, dv):
    width = dv // 2
    parts = []
    for n in range(width // 32):
        for k in range(BN // 16):
            byte = (2 * k * dv + half * width + n * 32) * 16
            parts.append(_read_s2r(storage, ((dq + slot * dv) // 8) * PADDED_CHUNK + base,
                                   (byte // 1024) * PADDED_CHUNK + byte % 1024))
    words = fx.Vector.from_elements([p[i] for p in parts for i in range(4)], fx.Int32)
    frag = fx.make_rmem_tensor(fx.make_layout((8, 4, width // 32), (1, 8, 32)), fx.BFloat16)
    frag.store(words.bitcast(fx.BFloat16))
    return frag


@flyc.jit
def _v_tail(values, remaining):
    words = fx.Vector(values).bitcast(fx.Int32)
    if remaining < BN:
        limit = _pin_i32(remaining - ((fx.Int32(gpu.thread_id("x")) >> 5) & 1) * 8)
        masks = []
        for i in fx.range_constexpr(16):
            token = (i // 4) * 16 + (i % 4) * 2
            mask = (limit > token).select(fx.Int32(0xFFFF), fx.Int32(0))
            mask = mask | (limit > token + 1).select(fx.Int32(-65536), fx.Int32(0))
            masks.append(mask)
        words = fx.Vector.from_elements([words[i] & masks[i % 16] for i in range(words.numel)], fx.Int32)
    return words.bitcast(fx.BFloat16)


def _qk(q, k, storage, base, half, dq, dv, full=False):
    mma = _mma()
    acc = mma.make_fragment_C(fx.make_rmem_tensor(fx.make_layout((32, BM), (1, 32)), fx.Float32))
    acc.fill(0.0)
    if (dq == 192 or dv == 192) and not full:
        # Full D192 K + Q + previous P + O exceeds the two-wave VGPR
        # budget. Consume contiguous K64 chunks in the SAME reduction order;
        # each retired operand frees its registers before the next LDS read.
        for chunk in range(dq // 64):
            if chunk:
                rocdl.sched_barrier(0)
                k = _read_k_tile_s2r(storage, base + chunk * 8320, half, 64, dv)
                _wait(lgkmcnt=0)
                # Inline LDS reads are opaque to LLVM's wait insertion. A
                # scheduler fence prevents the following MFMAs from moving
                # above this explicit wait (verified in the failed K64 ISA).
                rocdl.sched_barrier(0)
            q_part = fx.make_view(fx.get_iter(q) + chunk * 32,
                                  fx.make_layout((4, 1, (2, 4)), (1, 0, (4, 8))))
            fx.gemm(mma, acc, k, q_part, acc, traversal_order="kmn")
    else:
        fx.gemm(mma, acc, k, q, acc, traversal_order="kmn")
    return acc.load()


def _pv(p, v, output, dv):
    mma = _mma()
    prob = mma.make_fragment_B(fx.make_rmem_tensor(fx.make_layout((BM, BN), (1, BM)), fx.BFloat16))
    prob.store(p)
    acc = mma.make_fragment_C(fx.make_rmem_tensor(fx.make_layout((dv // 2, BM), (1, dv // 2)), fx.Float32))
    acc.store(output)
    operand = fx.make_view(fx.get_iter(v), fx.make_layout((4, dv // 64, (2, 4)), (1, 32, (4, 8))))
    fx.gemm(mma, acc, operand, prob, acc, traversal_order="mnk")
    return acc.load()


@flyc.jit
def _mask(scores, tile, row, q_len, kv_len, causal: fx.Constexpr[bool]):
    values = fx.Vector(scores)
    lane_half = (fx.Int32(gpu.thread_id("x")) >> 5) & 1
    if fx.const_expr(causal):
        if row - (fx.Int32(gpu.thread_id("x")) & 31) + kv_len - q_len < (tile + 1) * BN:
            bound = _pin_i32(_min(kv_len - 1, kv_len - q_len + row) - tile * BN - lane_half * 8)
            values = fx.Vector.from_elements([
                (bound >= fx.Int32((i // 8) * 16 + i % 8)).select(values[i], fx.Float32(float("-inf")))
                for i in range(32)
            ], fx.Float32)
    else:
        if kv_len - tile * BN < BN:
            # Materialize the lane coordinate only in the rare tail branch.
            # Otherwise LICM carries a per-lane bound with a VALU decrement
            # at the next iteration's memory-stage entry.
            tail_lane = _pin_i32(fx.Int32(gpu.thread_id("x")))
            bound = _pin_i32(kv_len - 1 - tile * BN - ((tail_lane >> 5) & 1) * 8)
            values = fx.Vector.from_elements([
                (bound >= fx.Int32((i // 8) * 16 + i % 8)).select(values[i], fx.Float32(float("-inf")))
                for i in range(32)
            ], fx.Float32)
    return values


@flyc.jit
def _rescale(o0, o1, row_sum, s_cur, old_max, new_max, ballot, exact):
    """Rescale of O and l when m moved (_next_max), plus the exact rule for S'(t).

    The FMA S * scale - m is up to ulp(m) / 2 off the rounded m. Once some |m| >= 2^22
    (checked when m moves, then sticky), S'(t) is also rounded to a multiple of ulp(m).
    """
    o0, o1, row_sum, s_cur = fx.Vector(o0), fx.Vector(o1), fx.Float32(row_sum), fx.Vector(s_cur)
    ballot, exact = fx.Int64(ballot), fx.Int64(exact)
    if ballot != fx.Int64(0):
        correction = _exp(fx.Float32(old_max) - fx.Float32(new_max))
        o0, o1, row_sum = o0 * correction, o1 * correction, row_sum * correction
        big = fx.absf(fx.Float32(new_max)) >= 2.0**22
        exact = exact | fx.Int64(rocdl.ballot(fx.Int64.ir_type, big.ir_value()))
    # A separate branch: merged into the one above, V192 (spilling) lost 5%.
    s_cur = _reround_if(s_cur, new_max, exact)
    return o0, o1, row_sum, s_cur, exact


def _store_o_tile_r2g(o0, o1, inv, storage, obuf, out_tid, out_row, heads, dv):
    """O * inv to BF16, C-shuffled through LDS (r2s b64, s2r b128), then b128 row stores (r2g).

    Reuses 32 KiB of LDS for one 64-column slice at a time (V192 has a third slice).
    """
    atom = fx.make_copy_atom(rocdl.BufferCopy128b(), fx.BFloat16)
    outputs = _join(o0, o1)
    for half in fx.range_constexpr(dv // 64):
        for n in fx.range_constexpr(2):
            for group in fx.range_constexpr(4):
                val = fx.Vector.from_elements([outputs[half * 32 + n * 16 + group * 4 + i] * inv for i in range(4)], fx.Float32)
                words = _pack_bf16(val)
                col = n * 32 + group * 8 + ((out_tid >> 5) & 1) * 4
                element = (out_row * 64 + col) ^ ((out_row & 15) * 4)
                address = fx.Int32(fx.ptrtoint(fx.get_iter(storage) + element * 2))
                llvm.inline_asm(ir.Type.parse("!llvm.void"), [address.ir_value(), words.ir_value()],
                                "ds_write_b64 $0, $1", "v,v,~{memory}", has_side_effects=True)
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
            offset = read_row * (heads * dv) + half * 64 + read_col
            fx.copy(atom, src, fx.make_view(fx.get_iter(obuf) + offset, fx.make_layout(8, 1)))
        _stage_end()


@flyc.jit
def _body_v128(Q, K, V, O, LSE, QS, KS, VS, table, storage, q0, q_len, kv_len, head, qb,
               H: fx.Constexpr[int], HK: fx.Constexpr[int], NP: fx.Constexpr[int], DQ: fx.Constexpr[int],
               DV: fx.Constexpr[int], PAGE: fx.Constexpr[int], CAUSAL: fx.Constexpr[bool],
               PER_TOKEN: fx.Constexpr[bool], WITH_LSE: fx.Constexpr[bool],
               SCALE: fx.Constexpr[float], STAGGER: fx.Constexpr[bool]):
    """DV128 without LSE (Dq 128/192): query rows 256 * qb .. + 255 of one head over KV tiles t.

    LDS: one K slot of (Dq / 8) * 1040 bytes and a two-slot V ring (V(t) in slot t & 1).
    Page ids run two tiles ahead of QK, K one tile ahead (g2r in S4, r2s in S6 of phase
    t - 1, after both groups read K(t - 1)) and V one tile ahead (GLOBAL g2r in S0, r2s
    in S2). Phase t runs QK and the row max of tile t and, one tile behind, the
    exponentials, row sum and PV of tile t - 1. Even stages are memory (VMEM/LDS),
    odd stages compute (MFMA/VALU); waves 4-7 (STAGGER) run one stage behind waves 0-3.
    Row sums/maxima of the two lane halves meet through ds_bpermute in memory stages.

    stage  phase t; cur = t & 1, prev = (t - 1) & 1
    S0     s2r K(t) keys 0-31; s_load pages(t+2); g2r V(t)
    S1     QK: S(t) keys 0-31          softmax: P(t-1) = exp2(S'(t-1)), values 0-23
    S2     V(t) lands; r2s V(t) -> [cur]; s2r K(t) keys 32-63
    S3     QK: S(t) keys 32-63         softmax: P(t-1) values 24-31, lane row sum, P to BF16
    S4     exchange row sums; s2r V(t-1) dims 0-63 [prev]; g2r K(t+1)
    S5     PV: O dims 0-63 += P V      softmax: l += row sum; mask S(t), lane row max
    S6     exchange row maxima; s2r V(t-1) dims 64-127 [prev]; K(t+1) lands; r2s K(t+1)
    S7     PV: O dims 64-127 += P V    softmax: new m, S'(t); rescale O and l

    Softmax in log2 units: S' = S * scale - m (one FMA), P = exp2(S'), l = sum(P) and
    O = sum(P V); the output is O / l * v_descale. m starts at the first row max + 1 and
    moves to a later row max + 1 only when that exceeds m + 7, rescaling O and l
    (AITER_FLYDSL_MHA_EXACT_MAX=1: m is the running row max). The
    FMA is up to ulp(m) / 2 off the rounded m: once some |m| >= 2^22 (checked when m
    moves, then sticky), a cold branch of _rescale also rounds S'(t) to a multiple of
    ulp(m), which matches the D256 kernels' rounded products.
    """
    # FlyDSL keys its disk cache on helpers named in this function's own code, not in closures.
    _helpers = (_read_k_g2r, _k_g2r_offset, _v_g2r_bases, _read_v_g2r, _store_r2s, _read_k_s2r, _read_v_s2r,
                _read_k_tile_g2r, _read_k_tile_s2r, _read_v_tile_s2r, _qk, _pv, _mask, _v_tail, _join,
                _mfma_hazard_wait, _maximum, _row_max, _row_max_local, _row_sum, _row_sum_local, _center_part,
                _exp_part, _pack_probability, _cross32_async, _rescale, _advance_max, _pin, _schedule,
                _stage_end, _wait, _reround_if, _first_max, _next_max)
    tid = fx.Int32(gpu.thread_id("x"))
    wave, lane = _uniform(tid >> 6), tid & 63
    q_start = qb * BM
    row = q_start + wave * 32 + (lane & 31)
    valid = _min(fx.Int32(BM), q_len - q_start)
    hkv = head // (H // HK)
    q_ptr = fx.get_iter(Q) + ((fx.Int64(q0) + fx.Int64(q_start)) * (H * DQ) + fx.Int64(head) * DQ)
    gq = _buffer(fx.make_view(q_ptr, fx.make_layout(BM * H * DQ, 1)), valid * H * DQ * 2)
    gk = _buffer(fx.make_view(fx.get_iter(K) + hkv * PAGE * DQ,
                            fx.make_layout((NP * HK - hkv) * PAGE * DQ, 1)),
                 (NP * HK - hkv) * PAGE * DQ * 2)
    # Q stays in registers for every KV tile.
    q = _read_q_g2r(gq, (wave * 32 + (lane & 31)) * (H * DQ * 2), lane, DQ)
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
    last, last_page = tiles - 1, (kv_len - 1) // PAGE
    # Page ids of tiles 0-2; every later phase t requests pages(t+2).
    pages_0 = _read_tile_pages_g2r(table, fx.Int32(0), last, last_page, PAGE)
    pages_1 = _read_tile_pages_g2r(table, fx.Int32(1), last, last_page, PAGE)
    pages_2 = _read_tile_pages_g2r(table, fx.Int32(2), last, last_page, PAGE)
    pages_0 = _page_ready(pages_0[0]), _page_ready(pages_0[1])
    pages_1 = _page_ready(pages_1[0]), _page_ready(pages_1[1])
    pages_2 = _page_ready(pages_2[0]), _page_ready(pages_2[1])
    # BF16 P carries eight consecutive keys per b128, not FP8's sixteen.
    k_row = (lane & 3) | ((lane & 4) << 1) | ((lane & 8) >> 1) | (lane & 16)
    k_base = (lane >> 5) * PADDED_CHUNK + ((k_row * 16) ^ ((k_row & 16) << 2))
    v_base = (lane >> 5) * (DV // 64) * PADDED_CHUNK + (lane & 31) * 16
    k_bytes = (DQ // 8) * PADDED_CHUNK
    half_page = (lane & 32) != 0
    # Loop-invariant lane addresses: LDS read/write (r/w) bases, K/V g2r lane offsets.
    shared = fx.Int32(fx.ptrtoint(fx.get_iter(storage)))
    within = (tid & 63) * 16
    k_s2r = _pin_i32(shared + k_base)
    k_r2s = _pin_i32(shared + (tid >> 6) * PADDED_CHUNK + (within ^ ((within & 256) >> 2)))
    v_s2r = _pin_i32(shared + k_bytes + v_base)
    v_r2s = _pin_i32(shared + k_bytes + (tid >> 6) * PADDED_CHUNK + within)
    k_lane = _pin_i32((tid >> 6) * (PAGE * 16) + (tid & (min(PAGE, BN) - 1)) * 16)
    v_lane = _pin_i32(tid * 16)
    cross_address = _pin_i32((lane ^ 32) * 4)
    v_global_base = _pin_s64(fx.Int64(fx.ptrtoint(fx.get_iter(V))) + fx.Int64(hkv) * (PAGE * DV * 2))

    def read_k_g2r(pages, tile):
        return _read_k_g2r(gk, _k_g2r_offset(k_lane, pages[0], pages[1], tile, half_page, HK, DQ, PAGE), DQ, PAGE)

    def read_v_g2r(pages, tile):
        first, second = _v_g2r_bases(v_global_base, pages[0], pages[1], tile, HK, PAGE)
        return _read_v_g2r(first, second, v_lane)

    def prologue():
        """K(0), V(0) -> LDS; QK S(0), m and S'(0); K(1) -> LDS (D192 loads it during QK(0))."""
        # Memory: g2r K(0), V(0); r2s K(0) and V(0) -> [0].
        k_0 = read_k_g2r(pages_0, fx.Int32(0))
        v_0 = read_v_g2r(pages_0, fx.Int32(0))
        _wait(vmcnt=0)
        _store_r2s(k_r2s, k_0, DQ // 64)
        _store_r2s(v_r2s, v_0, 2)
        _wait(vmcnt=0, lgkmcnt=0)
        _stage_end()
        if fx.const_expr(STAGGER):
            # Waves 4-7 fall one stage behind.
            _stage_end()
        # Memory: s2r K(0) keys 0-31.
        if fx.const_expr(DQ == 192):
            k_lo = _read_k_s2r(k_s2r, 0, DQ)
        else:
            k_lo = _read_k_tile_s2r(storage, k_base, 0, DQ, DV, True)
        _wait(lgkmcnt=0)
        _stage_end()
        # Compute: QK S(0) keys 0-31; O = 0.
        s_lo = _qk(q, k_lo, storage, k_base, 0, DQ, DV, True)
        o0 = _pin(fx.Vector.filled(DV // 4, 0.0, fx.Float32))
        o1 = _pin(fx.Vector.filled(DV // 4, 0.0, fx.Float32))
        _schedule(DQ // 8, 3, 5)
        _stage_end()
        # Memory: s2r K(0) keys 32-63 (D192 also issues g2r K(1)).
        if fx.const_expr(DQ == 192):
            k_hi = _read_k_s2r(k_s2r, 1, DQ)
            k_1 = read_k_g2r(pages_1, _min(fx.Int32(1), last))
        else:
            k_hi = _read_k_tile_s2r(storage, k_base, 1, DQ, DV, True)
        _wait(lgkmcnt=0)
        _stage_end()
        # Compute: QK S(0) keys 32-63; softmax m and S'(0).
        s_hi = _qk(q, k_hi, storage, k_base, 1, DQ, DV, True)
        s_0 = _mask(_join(s_lo, s_hi), fx.Int32(0), row, q_len, kv_len, CAUSAL)
        _mfma_hazard_wait()
        row_max = _first_max(s_0, scale)
        s_0 = _center_part(s_0, scale, row_max, 0, 32)
        # The exact rule of _rescale for S'(0); exact carries it through the phases.
        exact = fx.Int64(rocdl.ballot(fx.Int64.ir_type, (fx.absf(row_max) >= 2.0**22).ir_value()))
        s_0 = _reround_if(s_0, row_max, exact)
        row_sum = fx.Float32(0.0)
        _stage_end()
        # Memory: both groups read K(0); g2r K(1) (D128) and r2s it into the single K slot.
        if fx.const_expr(DQ == 128):
            k_1 = _read_k_tile_g2r(gk, tid, pages_1[0], pages_1[1], _min(fx.Int32(1), last), HK, DQ, PAGE)
        _wait(vmcnt=0)
        _store_r2s(k_r2s, k_1, DQ // 64)
        _wait(lgkmcnt=0)
        _stage_end()
        # The leading group waits one more stage for the trailing group's K(1)
        # writes; an LGKM wait alone cannot cover another wave.
        _stage_end()
        return s_0, row_max, row_sum, o0, o1, *pages_1, *pages_2, exact

    @flyc.jit
    def phase(s_prev, row_max, row_sum, o0, o1, page_cur0, page_cur1, page_next0, page_next1, exact, t,
              CUR: fx.Constexpr[int], PREV: fx.Constexpr[int]):
        """Phase t (table in _body_v128); page_cur*/page_next* are the page ids of tiles t and t+1.

        D128 unrolls two phases so CUR/PREV are constants; D192 runs one phase per trip and
        derives the V slots from t. exact carries the exact rule of _rescale.
        """
        s_prev, row_max, row_sum = fx.Vector(s_prev), fx.Float32(row_max), fx.Float32(row_sum)
        o0, o1, exact, t = fx.Vector(o0), fx.Vector(o1), fx.Int64(exact), fx.Int32(t)
        slot_cur = (t & 1) if DQ == 192 else CUR
        slot_prev = ((t - 1) & 1) if DQ == 192 else PREV
        # S0 memory: s2r K(t) keys 0-31; s_load pages(t+2); g2r V(t) (SGPR bases only change).
        k_lo = _read_k_s2r(k_s2r, 0, DQ)
        rocdl.sched_barrier(0)
        requests_next2 = _read_tile_pages_g2r(table, t + 2, last, last_page, PAGE)
        v_cur = read_v_g2r((page_cur0, page_cur1), t)
        rocdl.sched_barrier(0)
        page_next2_0, page_next2_1 = _page_ready(requests_next2[0]), _page_ready(requests_next2[1])
        _stage_end()
        # S1 compute: QK S(t) keys 0-31; P(t-1) values 0-23. D192 ring addresses are made here.
        s_lo = _qk(q, k_lo, storage, k_base, 0, DQ, DV, True)
        v_read, v_write = v_s2r, v_r2s
        if fx.const_expr(DQ == 192):
            v_read = _pin_i32(v_s2r + slot_prev * V_SLOT)
            v_write = _pin_i32(v_r2s + slot_cur * V_SLOT)
        p_prev = _pin(_exp_part(s_prev, 0, 24), 32)
        _schedule(DQ // 8, 2, 1, True)
        _stage_end()
        # S2 memory: r2s V(t) -> [cur] once it lands; s2r K(t) keys 32-63.
        _wait(vmcnt=0)
        _store_r2s(v_write, v_cur, 2, CUR * V_SLOT if DQ == 128 else 0)
        k_hi = _read_k_s2r(k_s2r, 1, DQ)
        _wait(lgkmcnt=0)
        _stage_end()
        # S3 compute: QK S(t) keys 32-63; P(t-1) values 24-31, lane row sum, BF16 P; K(t+1) address.
        s_hi = _qk(q, k_hi, storage, k_base, 1, DQ, DV, True)
        p_prev = _exp_part(p_prev, 24, 32)
        lane_sum = _row_sum_local(p_prev)
        p_bf16 = _pack_probability(p_prev)
        _schedule(4, 2, 2, True)
        _schedule(DQ // 8 - 4, 8, 2)
        k_next_offset = _k_g2r_offset(k_lane, page_next0, page_next1, _min(t + 1, last), half_page, HK, DQ, PAGE)
        _stage_end()
        # S4 memory: exchange row sums; s2r V(t-1) dims 0-63; g2r K(t+1).
        other_sum = _cross32_async(lane_sum, cross_address)
        v_lo = _read_v_s2r(v_read, 0, PREV if DQ == 128 else 0)
        _wait(vmcnt=0)
        k_next = _read_k_g2r(gk, k_next_offset, DQ, PAGE)
        # The shuffle shares this LDS wait; it has no consumer before S5.
        _wait(lgkmcnt=0)
        _stage_end()
        # S5 compute: PV O dims 0-63; l += row sum; mask S(t) and its lane row max.
        s_cur = _mask(_join(s_lo, s_hi), t, row, q_len, kv_len, CAUSAL)
        o0 = _pv(p_bf16, v_lo, o0, DV)
        row_sum = row_sum + (lane_sum + other_sum)
        lane_max = _row_max_local(s_cur)
        _schedule(DV // 8, 2, 3)
        _stage_end()
        # S6 memory: exchange row maxima; s2r V(t-1) dims 64-127; r2s K(t+1) after both groups read K(t).
        other_max = _cross32_async(lane_max, cross_address)
        v_hi = _read_v_s2r(v_read, 1, PREV if DQ == 128 else 0)
        _wait(vmcnt=0)
        _store_r2s(k_r2s, k_next, DQ // 64)
        _wait(lgkmcnt=0)
        _stage_end()
        # S7 compute: PV O dims 64-127; new m and S'(t); lazy rescale of O and l.
        o1 = _pv(p_bf16, v_hi, o1, DV)
        candidate = _maximum(lane_max, other_max) * scale
        ballot, new_max = _next_max(candidate, row_max)
        s_cur = _center_part(s_cur, scale, new_max, 0, 32)
        _schedule(DV // 8, 4, 4)
        o0, o1, row_sum, s_cur, exact = _rescale(o0, o1, row_sum, s_cur, row_max, new_max, ballot, exact)
        new_max = _advance_max(row_max, new_max)
        _stage_end()
        return s_cur, new_max, row_sum, o0, o1, page_next0, page_next1, page_next2_0, page_next2_1, exact

    def epilogue(state):
        """exp, row sum and PV of the last tile; V(last) already sits in slot last & 1."""
        s_last, row_max, row_sum, o0, o1 = state[:5]
        p_last = _exp_part(fx.Vector(s_last), 0, 32)
        row_sum = fx.Float32(row_sum) + _row_sum(p_last)
        p_bf16 = _pack_probability(p_last)
        _wait(vmcnt=0, lgkmcnt=0)
        _stage_end()
        # Memory: s2r V(last) dims 0-63; zero rows past kv_len (0 * NaN is NaN).
        v_lo = _read_v_tile_s2r(storage, v_base, last & 1, 0, DQ, DV)
        _wait(lgkmcnt=0)
        rocdl.sched_barrier(0)
        v_lo.store(_v_tail(v_lo.load(), kv_len - last * BN))
        _stage_end()
        # Compute: PV O dims 0-63.
        o0 = _pv(p_bf16, v_lo, o0, DV)
        _stage_end()
        # Memory: s2r V(last) dims 64-127.
        v_hi = _read_v_tile_s2r(storage, v_base, last & 1, 1, DQ, DV)
        _wait(lgkmcnt=0)
        rocdl.sched_barrier(0)
        v_hi.store(_v_tail(v_hi.load(), kv_len - last * BN))
        _stage_end()
        # Compute: PV O dims 64-127.
        o1 = _pv(p_bf16, v_hi, o1, DV)
        _stage_end()
        if fx.const_expr(not STAGGER):
            # Waves 0-3 wait for waves 4-7 before the output shuffle reuses LDS.
            _stage_end()
        return row_max, row_sum, o0, o1

    # Pipeline: prologue stages K(0)/V(0), runs QK(0) and stages K(1); phases t = 1 .. tiles - 1;
    # epilogue finishes PV(last).
    state = prologue()
    if fx.const_expr(DQ == 128):
        # Two phases per trip keep the V ring slots compile-time constants.
        for t in range(fx.Int32(1), tiles - 1, fx.Int32(2)):
            state = phase(*state, t, 1, 0)
            state = phase(*state, t + 1, 0, 1)
        if (tiles & 1) == 0:
            state = phase(*state, last, 1, 0)
    else:
        # D192 derives the slots from t; a single phase avoids doubling register pressure.
        for t in range(fx.Int32(1), tiles, fx.Int32(1)):
            state = phase(*state, t, 0, 0)
    row_max, row_sum, o0, o1 = epilogue(state)

    inv = (row_sum > 0.0).select(fx.Float32(1.0) / row_sum, fx.Float32(0.0)) * VS[0]
    out_tid = _pin_i32(fx.Int32(gpu.thread_id("x")))
    out_row = (out_tid >> 6) * 32 + (out_tid & 31)
    optr = fx.get_iter(O) + ((fx.Int64(q0) + fx.Int64(q_start)) * (H * DV) + fx.Int64(head) * DV)
    obuf = rocdl.make_buffer_tensor(fx.make_view(optr, fx.make_layout(BM * H * DV, 1)),
                                    num_records_bytes=valid * H * DV * 2)
    _store_o_tile_r2g(o0, o1, inv, storage, obuf, out_tid, out_row, H, DV)


@flyc.jit
def _body_wide(Q, K, V, O, LSE, QS, KS, VS, table, storage, q0, q_len, kv_len, head, qb,
               H: fx.Constexpr[int], HK: fx.Constexpr[int], NP: fx.Constexpr[int], DQ: fx.Constexpr[int],
               DV: fx.Constexpr[int], PAGE: fx.Constexpr[int], CAUSAL: fx.Constexpr[bool],
               PER_TOKEN: fx.Constexpr[bool], WITH_LSE: fx.Constexpr[bool],
               SCALE: fx.Constexpr[float], STAGGER: fx.Constexpr[bool]):
    """LSE or DV192: query rows 256 * qb .. + 255 of one head over KV tiles t, cooperative copies.

    LDS: one K slot; V uses a two-slot ring for D128/V128 (V(t) in slot t & 1) and one
    slot otherwise. Page ids run two tiles ahead and K one tile ahead (g2r in S0; r2s
    into the single slot in S4 for D128/V128, in S6 when D192/V192 QK streams K64
    chunks through S3). V(t) is loaded in S0; the ring publishes it in S2, the single
    slot in S0 of phase t + 1. Same stage roles and softmax as _body_v128, except the
    row sum/max reductions use lane shuffles inside compute stages.

    stage  phase t; ring: cur = t & 1, prev = (t - 1) & 1
    S0     s2r K(t) keys 0-31; single slot: r2s V(t-1); s_load pages(t+2); g2r V(t), K(t+1)
    S1     QK: S(t) keys 0-31          softmax: P(t-1) values 0-23
    S2     s2r K(t) keys 32-63; ring: V(t) lands, r2s V(t) -> [cur]
    S3     QK: S(t) keys 32-63         softmax: P(t-1) values 24-31, l += row sum, P to BF16
    S4     s2r V(t-1) dims 0-63; K(t+1) lands; D128/V128: r2s K(t+1)
    S5     PV: O dims 0-63 += P V      softmax: mask S(t), new m, S'(t) values 0..split
    S6     s2r V(t-1) dims 64-DV; D192/V192: r2s K(t+1)
    S7     PV: rest of O += P V        softmax: S'(t) values split..31; rescale O and l
    """
    # FlyDSL keys its disk cache on helpers named in this function's own code, not in closures.
    _helpers = (_read_k_tile_g2r, _read_v_tile_g2r, _store_tile_r2s, _read_k_tile_s2r, _read_v_tile_s2r,
                _qk, _pv, _mask, _v_tail, _join, _mfma_hazard_wait, _maximum, _row_max, _row_sum,
                _center_part, _exp_part, _pack_probability, _rescale, _pin, _schedule, _stage_end, _wait,
                _reround_if, _first_max, _next_max)
    tid = fx.Int32(gpu.thread_id("x"))
    wave, lane = _uniform(tid >> 6), tid & 63
    q_start = qb * BM
    row = q_start + wave * 32 + (lane & 31)
    valid = _min(fx.Int32(BM), q_len - q_start)
    hkv = head // (H // HK)
    v_ring = DV == 128 and DQ == 128
    q_ptr = fx.get_iter(Q) + ((fx.Int64(q0) + fx.Int64(q_start)) * (H * DQ) + fx.Int64(head) * DQ)
    gq = _buffer(fx.make_view(q_ptr, fx.make_layout(BM * H * DQ, 1)), valid * H * DQ * 2)
    gk = _buffer(fx.make_view(fx.get_iter(K) + hkv * PAGE * DQ,
                            fx.make_layout((NP * HK - hkv) * PAGE * DQ, 1)),
                 (NP * HK - hkv) * PAGE * DQ * 2)
    # Q stays in registers for every KV tile.
    q = _read_q_g2r(gq, (wave * 32 + (lane & 31)) * (H * DQ * 2), lane, DQ)
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
    last, last_page = tiles - 1, (kv_len - 1) // PAGE
    pages_0 = _read_tile_pages_g2r(table, fx.Int32(0), last, last_page, PAGE)
    pages_1 = _read_tile_pages_g2r(table, fx.Int32(1), last, last_page, PAGE)
    pages_2 = _read_tile_pages_g2r(table, fx.Int32(2), last, last_page, PAGE)
    pages_0 = _page_ready(pages_0[0]), _page_ready(pages_0[1])
    pages_1 = _page_ready(pages_1[0]), _page_ready(pages_1[1])
    pages_2 = _page_ready(pages_2[0]), _page_ready(pages_2[1])
    k_row = (lane & 3) | ((lane & 4) << 1) | ((lane & 8) >> 1) | (lane & 16)
    k_base = (lane >> 5) * PADDED_CHUNK + ((k_row * 16) ^ ((k_row & 16) << 2))
    v_base = (lane >> 5) * (DV // 64) * PADDED_CHUNK + (lane & 31) * 16
    k_bytes = (DQ // 8) * PADDED_CHUNK

    def read_k_g2r(pages, tile):
        return _read_k_tile_g2r(gk, tid, pages[0], pages[1], tile, HK, DQ, PAGE)

    def read_v_g2r(pages, tile):
        return _read_v_tile_g2r(V, tid, pages[0], pages[1], tile, hkv, HK, DV, PAGE, True)

    def store_k_r2s(words):
        _store_tile_r2s(storage, tid, words, 0, DQ // 64, True, True)

    def store_v_r2s(words, slot=0):
        _store_tile_r2s(storage, tid, words, k_bytes + slot * (DV // 8) * PADDED_CHUNK, DV // 64, bounded=True)

    def prologue():
        """K(0), V(0) -> LDS; QK S(0), m and S'(0); K(1) -> LDS."""
        k_0 = read_k_g2r(pages_0, fx.Int32(0))
        v_0 = read_v_g2r(pages_0, fx.Int32(0))
        _wait(vmcnt=0)
        store_k_r2s(k_0)
        store_v_r2s(v_0)
        _wait(vmcnt=0, lgkmcnt=0)
        _stage_end()
        if fx.const_expr(STAGGER):
            # Waves 4-7 fall one stage behind.
            _stage_end()
        # Memory: s2r K(0) keys 0-31 (D192/V192: first K64 chunk).
        k_lo = _read_k_tile_s2r(storage, k_base, 0, DQ, DV)
        _wait(lgkmcnt=0)
        _stage_end()
        # Compute: QK S(0) keys 0-31; O = 0.
        s_lo = _qk(q, k_lo, storage, k_base, 0, DQ, DV)
        o0 = _pin(fx.Vector.filled(DV // 4, 0.0, fx.Float32))
        o1 = _pin(fx.Vector.filled(DV // 4, 0.0, fx.Float32))
        _schedule(DQ // 8, 3, 5)
        _stage_end()
        # Memory: s2r K(0) keys 32-63.
        k_hi = _read_k_tile_s2r(storage, k_base, 1, DQ, DV)
        _wait(lgkmcnt=0)
        _stage_end()
        # Compute: QK S(0) keys 32-63; softmax m and S'(0).
        s_hi = _qk(q, k_hi, storage, k_base, 1, DQ, DV)
        s_0 = _mask(_join(s_lo, s_hi), fx.Int32(0), row, q_len, kv_len, CAUSAL)
        _mfma_hazard_wait()
        row_max = _first_max(s_0, scale)
        s_0 = _center_part(s_0, scale, row_max, 0, 32)
        # The exact rule of _rescale for S'(0); exact carries it through the phases.
        exact = fx.Int64(rocdl.ballot(fx.Int64.ir_type, (fx.absf(row_max) >= 2.0**22).ir_value()))
        s_0 = _reround_if(s_0, row_max, exact)
        row_sum = fx.Float32(0.0)
        _stage_end()
        if fx.const_expr(DQ == 192 or DV == 192):
            # Streaming QK reads K through the end of QK-high: both groups must finish first.
            _stage_end()
        # Memory: g2r K(1) and r2s it into the single K slot.
        k_1 = read_k_g2r(pages_1, _min(fx.Int32(1), last))
        _wait(vmcnt=0)
        store_k_r2s(k_1)
        _wait(lgkmcnt=0)
        _stage_end()
        # The leading group waits one more stage for the trailing group's K(1) writes.
        _stage_end()
        return s_0, row_max, row_sum, o0, o1, v_0, *pages_1, *pages_2, exact

    @flyc.jit
    def phase(s_prev, row_max, row_sum, o0, o1, v_pending, page_cur0, page_cur1, page_next0, page_next1, exact, t,
              CUR: fx.Constexpr[int], PREV: fx.Constexpr[int]):
        """Phase t (table in _body_wide); v_pending is the staged V(t-1) of the single-slot path.

        exact carries the exact rule of _rescale.
        """
        s_prev, row_max, row_sum = fx.Vector(s_prev), fx.Float32(row_max), fx.Float32(row_sum)
        o0, o1, v_pending, t = fx.Vector(o0), fx.Vector(o1), fx.Vector(v_pending), fx.Int32(t)
        exact = fx.Int64(exact)
        # S0 memory: s2r K(t) keys 0-31; single slot publishes V(t-1); pages(t+2); g2r V(t), K(t+1).
        k_lo = _read_k_tile_s2r(storage, k_base, 0, DQ, DV)
        rocdl.sched_barrier(0)
        if fx.const_expr(not v_ring):
            store_v_r2s(v_pending)
        requests_next2 = _read_tile_pages_g2r(table, t + 2, last, last_page, PAGE)
        v_cur = read_v_g2r((page_cur0, page_cur1), t)
        k_next = read_k_g2r((page_next0, page_next1), _min(t + 1, last))
        rocdl.sched_barrier(0)
        page_next2_0, page_next2_1 = _page_ready(requests_next2[0]), _page_ready(requests_next2[1])
        _stage_end()
        # S1 compute: QK S(t) keys 0-31; P(t-1) values 0-23.
        s_lo = _qk(q, k_lo, storage, k_base, 0, DQ, DV)
        p_prev = _pin(_exp_part(s_prev, 0, 24), 32)
        _schedule(DQ // 8, 2, 1, True)
        _stage_end()
        # S2 memory: s2r K(t) keys 32-63; the ring publishes V(t) -> [cur] once V lands.
        k_hi = _read_k_tile_s2r(storage, k_base, 1, DQ, DV)
        if fx.const_expr(v_ring):
            _wait(vmcnt=DQ // 64)
            store_v_r2s(v_cur, CUR)
        _wait(lgkmcnt=0)
        _stage_end()
        # S3 compute: QK S(t) keys 32-63; P(t-1) values 24-31, l += row sum, BF16 P.
        s_hi = _qk(q, k_hi, storage, k_base, 1, DQ, DV)
        p_prev = _exp_part(p_prev, 24, 32)
        row_sum = row_sum + _row_sum(p_prev)
        p_bf16 = _pack_probability(p_prev)
        _schedule(4, 2, 2, True)
        _schedule(DQ // 8 - 4, 8, 2)
        _stage_end()
        # S4 memory: s2r V(t-1) dims 0-63; D128/V128 publishes K(t+1) (both groups read K(t)).
        v_lo = _read_v_tile_s2r(storage, v_base, PREV if v_ring else 0, 0, DQ, DV)
        _wait(vmcnt=0)
        if fx.const_expr(DQ == 128 and DV == 128):
            store_k_r2s(k_next)
        _wait(lgkmcnt=0)
        _stage_end()
        # S5 compute: PV O dims 0-63; mask S(t), lazy new m and S'(t) values 0..split.
        s_cur = _mask(_join(s_lo, s_hi), t, row, q_len, kv_len, CAUSAL)
        o0 = _pv(p_bf16, v_lo, o0, DV)
        candidate = _row_max(s_cur) * scale
        ballot, new_max = _next_max(candidate, row_max)
        split = 12 if STAGGER else 6
        s_cur = _center_part(s_cur, scale, new_max, 0, split)
        _schedule(3, 5, 3)
        rocdl.sched_group_barrier(rocdl.mask_mfma, 1, 3)
        rocdl.sched_group_barrier(0x002, 3, 3)
        rocdl.sched_group_barrier(rocdl.mask_dsrd, 1, 3)
        rocdl.sched_group_barrier(rocdl.mask_mfma, 2, 3)
        rocdl.sched_group_barrier(0x002, 10, 3)
        _schedule(DV // 8 - 6, 5, 3)
        _stage_end()
        # S6 memory: s2r V(t-1) dims 64-DV. D192/V192 publish K(t+1) only now: the partner
        # group's S3 streams K until the end of QK-high, two stages before the next K read.
        v_hi = _read_v_tile_s2r(storage, v_base, PREV if v_ring else 0, 1, DQ, DV)
        if fx.const_expr(DQ == 192 or DV == 192):
            store_k_r2s(k_next)
        _wait(lgkmcnt=0)
        _stage_end()
        # S7 compute: rest of PV; S'(t) values split..31; lazy rescale of O and l.
        o1 = _pv(p_bf16, v_hi, o1, DV)
        s_cur = _center_part(s_cur, scale, new_max, split, 32)
        _schedule(DV // 8, 4, 4)
        o0, o1, row_sum, s_cur, exact = _rescale(o0, o1, row_sum, s_cur, row_max, new_max, ballot, exact)
        _stage_end()
        v_pending = fx.Vector.filled(DV // 16, 0, fx.Int32) if v_ring else v_cur
        return (s_cur, new_max, row_sum, o0, o1, v_pending, page_next0, page_next1, page_next2_0, page_next2_1,
                exact)

    def epilogue(state):
        """Single slot: r2s V(last). exp, row sum and PV of the last tile."""
        s_last, row_max, row_sum, o0, o1, v_pending = state[:6]
        if fx.const_expr(not v_ring):
            # Publish V(last) after both groups' S6 readers of V(last - 1).
            store_v_r2s(v_pending)
        p_last = _exp_part(fx.Vector(s_last), 0, 32)
        row_sum = fx.Float32(row_sum) + _row_sum(p_last)
        p_bf16 = _pack_probability(p_last)
        _wait(vmcnt=0, lgkmcnt=0)
        _stage_end()
        if fx.const_expr(not v_ring):
            # Wait for the trailing group's V(last) publication.
            _stage_end()
        # Memory: s2r V(last) dims 0-63; zero rows past kv_len.
        v_lo = _read_v_tile_s2r(storage, v_base, (last & 1) if v_ring else 0, 0, DQ, DV)
        _wait(lgkmcnt=0)
        rocdl.sched_barrier(0)
        v_lo.store(_v_tail(v_lo.load(), kv_len - last * BN))
        _stage_end()
        o0 = _pv(p_bf16, v_lo, o0, DV)
        _stage_end()
        # Memory: s2r V(last) dims 64-DV.
        v_hi = _read_v_tile_s2r(storage, v_base, (last & 1) if v_ring else 0, 1, DQ, DV)
        _wait(lgkmcnt=0)
        rocdl.sched_barrier(0)
        v_hi.store(_v_tail(v_hi.load(), kv_len - last * BN))
        _stage_end()
        o1 = _pv(p_bf16, v_hi, o1, DV)
        _stage_end()
        if fx.const_expr(not STAGGER):
            _stage_end()
        return row_max, row_sum, o0, o1

    state = prologue()
    if fx.const_expr(v_ring):
        # Two phases per trip keep the V ring slots compile-time constants.
        for t in range(fx.Int32(1), tiles - 1, fx.Int32(2)):
            state = phase(*state, t, 1, 0)
            state = phase(*state, t + 1, 0, 1)
        if (tiles & 1) == 0:
            state = phase(*state, last, 1, 0)
    else:
        for t in range(fx.Int32(1), tiles, fx.Int32(1)):
            state = phase(*state, t, 0, 0)
    row_max, row_sum, o0, o1 = epilogue(state)

    inv = (row_sum > 0.0).select(fx.Float32(1.0) / row_sum, fx.Float32(0.0)) * VS[0]
    out_tid = _pin_i32(fx.Int32(gpu.thread_id("x")))
    out_row = (out_tid >> 6) * 32 + (out_tid & 31)
    optr = fx.get_iter(O) + ((fx.Int64(q0) + fx.Int64(q_start)) * (H * DV) + fx.Int64(head) * DV)
    obuf = rocdl.make_buffer_tensor(fx.make_view(optr, fx.make_layout(BM * H * DV, 1)),
                                    num_records_bytes=valid * H * DV * 2)
    _store_o_tile_r2g(o0, o1, inv, storage, obuf, out_tid, out_row, H, DV)
    if fx.const_expr(WITH_LSE):
        if ((out_tid & 63) < 32) & (out_row < valid):
            log_l = fx.Float32(llvm.call_intrinsic(fx.Float32.ir_type, "llvm.log2.f32", [row_sum.ir_value()], [], []))
            LSE[(q0 + q_start + out_row) * H + head] = (row_sum > 0.0).select(
                (fx.Float32(row_max) + log_l) * fx.Float32(math.log(2.0)), fx.Float32(float("-inf")))


@flyc.jit
def _work(Q, K, V, O, LSE, CQ, KI, PAGES, LAST, QS, KS, VS, storage, head, batch, qb,
          H: fx.Constexpr[int], HK: fx.Constexpr[int], NP: fx.Constexpr[int], DQ: fx.Constexpr[int],
          DV: fx.Constexpr[int], PAGE: fx.Constexpr[int], CAUSAL: fx.Constexpr[bool],
          PER_TOKEN: fx.Constexpr[bool], WITH_LSE: fx.Constexpr[bool], SCALE: fx.Constexpr[float]):
    # AST rewriting lifts branch bodies into local functions. Capture the
    # compute dependency before the branch so the native cache follows it.
    body = _body_wide
    if fx.const_expr(DV == 128 and not WITH_LSE):
        body = _body_v128
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
                  H, HK, NP, DQ, DV, PAGE, CAUSAL, PER_TOKEN, WITH_LSE, SCALE, True)
        else:
            body(Q, K, V, O, LSE, QS, KS, VS, table, storage, q0, q_len, kv_len, head, qb,
                  H, HK, NP, DQ, DV, PAGE, CAUSAL, PER_TOKEN, WITH_LSE, SCALE, False)


@flyc.kernel(known_block_size=[THREADS, 1, 1])
def _attention_kernel_942(Q: fx.Tensor, K: fx.Tensor, V: fx.Tensor, O: fx.Tensor, LSE: fx.Tensor,
    CQ: fx.Tensor, KI: fx.Tensor, PAGES: fx.Tensor, LAST: fx.Tensor, QS: fx.Tensor, KS: fx.Tensor, VS: fx.Tensor,
    H: fx.Constexpr[int], HK: fx.Constexpr[int], NP: fx.Constexpr[int], B: fx.Constexpr[int],
    MAX_Q: fx.Constexpr[int], DQ: fx.Constexpr[int], DV: fx.Constexpr[int], PAGE: fx.Constexpr[int],
    CAUSAL: fx.Constexpr[bool], PER_TOKEN: fx.Constexpr[bool], WITH_LSE: fx.Constexpr[bool],
    SCALE: fx.Constexpr[float], PERSISTENT: fx.Constexpr[bool], CUS: fx.Constexpr[int]):
    size = ((DQ + (2 if DV == 128 and (DQ == 128 or not WITH_LSE) else 1) * DV) // 8) * PADDED_CHUNK
    storage = fx.SharedAllocator().allocate(fx.Array[fx.Int8, size, 16]).peek().view(fx.make_layout(size, 1))
    if fx.const_expr(PERSISTENT):
        # A fixed resident grid revisits the SAME FP8-derived body. Grid-stride
        # work assignment needs no shared counter, allocation, or reset kernel;
        # independent streams and captured graphs have no mutable shared state.
        work = fx.Int32(gpu.block_id("x"))
        while work < H * B * ((MAX_Q + BM - 1) // BM):
            head, batch, qb = work % H, (work // H) % B, work // (H * B)
            _work(Q, K, V, O, LSE, CQ, KI, PAGES, LAST, QS, KS, VS, storage, head, batch, qb,
                  H, HK, NP, DQ, DV, PAGE, CAUSAL, PER_TOKEN, WITH_LSE, SCALE)
            work = work + CUS
    else:
        head, batch, qb = fx.Int32(gpu.block_id("x")), fx.Int32(gpu.block_id("y")), fx.Int32(gpu.block_id("z"))
        _work(Q, K, V, O, LSE, CQ, KI, PAGES, LAST, QS, KS, VS, storage, head, batch, qb,
              H, HK, NP, DQ, DV, PAGE, CAUSAL, PER_TOKEN, WITH_LSE, SCALE)


@flyc.jit
def _launch_attention(Q: fx.Tensor, K: fx.Tensor, V: fx.Tensor, O: fx.Tensor, LSE: fx.Tensor,
    CQ: fx.Tensor, KI: fx.Tensor, PAGES: fx.Tensor, LAST: fx.Tensor, QS: fx.Tensor, KS: fx.Tensor, VS: fx.Tensor,
    H: fx.Constexpr[int], HK: fx.Constexpr[int], NP: fx.Constexpr[int], B: fx.Constexpr[int],
    MAX_Q: fx.Constexpr[int], DQ: fx.Constexpr[int], DV: fx.Constexpr[int], PAGE: fx.Constexpr[int],
    CAUSAL: fx.Constexpr[bool], PER_TOKEN: fx.Constexpr[bool], WITH_LSE: fx.Constexpr[bool],
    SCALE: fx.Constexpr[float], PERSISTENT: fx.Constexpr[bool], CUS: fx.Constexpr[int], stream: fx.Stream):
    grid = (min(CUS, H * B * ((MAX_Q + BM - 1) // BM)), 1, 1) if PERSISTENT else (H, B, (MAX_Q + BM - 1) // BM)
    _attention_kernel_942(Q, K, V, O, LSE, CQ, KI, PAGES, LAST, QS, KS, VS, H, HK, NP, B,
        MAX_Q, DQ, DV, PAGE, CAUSAL, PER_TOKEN, WITH_LSE, SCALE, PERSISTENT, CUS,
        value_attrs={"rocdl.waves_per_eu": 2,
                     "llvm.target_features": ir.Attribute.parse('#llvm.target_features<["-packed-fp32-ops"]>')},
    ).launch(grid=grid, block=(THREADS, 1, 1), stream=stream)


class _PagedAttention:
    bf16_backend = "native-8wave-8stage"

    def __init__(self, heads, kv_heads, dq, dv, page, causal, mode, persistent):
        self.heads, self.kv_heads, self.dq, self.dv = heads, kv_heads, dq, dv
        self.page_size, self.causal, self.quant_query_mode = page, causal, mode
        self.memory_mode, self.persistent = "lds", persistent
        self._launch = _launch_attention
        if dq == dv == 256:
            from .mha_pa_bf16_256_paged_942 import _launch
            self._launch = _launch
            self.bf16_backend = "native-m16-dma4-q0-kearly-vearly-interleave1-1-pages1-lateS41"
        self._compiled = {}

    def __call__(self, Q, K, V, cu_seqlens_q, cu_seqlens_k, kv_indptr, kv_page_indices,
                 max_seqlen_q, max_seqlen_k, causal, q_descale, k_descale, v_descale,
                 kv_last_page_lens, out=None, sink_ptr=None, stream=None, *, return_lse=False, lse=None, softmax_scale=None):
        """Run with contiguous SHUFFLE-5D KV and asynchronous device metadata.

        Descales are finite positive FP32 and the scale product
        softmax_scale * log2(e) * q_descale * k_descale must stay a positive
        finite FP32. K/V within each KV length must be finite: rows of a causal
        tile share its V, so a non-finite value also reaches rows that mask its
        key. Metadata must be consistent/in-bounds; no GPU metadata is copied to
        the host. Active KV must be nonempty and causal KV >= Q. Caller output/LSE
        must not overlap inputs or each other. Empty Q returns without dispatch.
        Warm before graph capture.
        """
        if not isinstance(causal, bool) or causal != self.causal or not isinstance(return_lse, bool):
            raise ValueError("causal must match the factory; return_lse must be bool")
        if sink_ptr is not None:
            raise NotImplementedError("gfx942 BF16 full MHA does not support sinks")
        if not isinstance(Q, torch.Tensor) or not Q.is_cuda:
            raise ValueError("Q/K/V must be tensors on the same gfx942 GPU")
        device = Q.device
        properties = torch.cuda.get_device_properties(device)
        if getattr(properties, "gcnArchName", "").split(":", 1)[0] != "gfx942":
            raise NotImplementedError("this BF16 backend requires gfx942")
        for name, tensor in (("Q", Q), ("K", K), ("V", V)):
            if not isinstance(tensor, torch.Tensor) or tensor.device != device:
                raise ValueError(f"{name} must be a tensor on the input GPU")
            if tensor.dtype != torch.bfloat16:
                raise NotImplementedError("gfx942 BF16 attention requires BF16 Q/K/V")
            if tensor.layout != torch.strided or not tensor.is_contiguous():
                raise ValueError(f"{name} must be contiguous")
            if tensor.numel() * tensor.element_size() >= 2**31:
                raise NotImplementedError(f"{name} byte span must fit signed int32 addressing")
        if Q.ndim != 3 or Q.shape[1:] != (self.heads, self.dq):
            raise ValueError("Q must be [tokens, query heads, Dq]")
        if (K.ndim != 5 or V.ndim != 5
                or K.shape != (V.shape[0], self.kv_heads, self.dq // 8, self.page_size, 8)
                or V.shape[1:] != (self.kv_heads, self.page_size // 8, self.dv, 8)):
            raise ValueError("K/V must use the factory's BF16 SHUFFLE-5D layouts")
        tokens = Q.shape[0]
        for name, bound in (("max_seqlen_q", max_seqlen_q), ("max_seqlen_k", max_seqlen_k)):
            if not isinstance(bound, int) or isinstance(bound, bool) or not 0 <= bound < 2**31:
                raise ValueError(f"{name} must be a nonnegative signed int32 bound")
        if causal and max_seqlen_k < max_seqlen_q:
            raise ValueError("bottom-right causal attention requires KV >= Q")
        if tokens and max_seqlen_q == 0:
            raise ValueError("nonempty Q requires max_seqlen_q > 0")
        metadata = (("cu_seqlens_q", cu_seqlens_q), ("kv_indptr", kv_indptr),
                    ("kv_page_indices", kv_page_indices), ("kv_last_page_lens", kv_last_page_lens))
        if cu_seqlens_k is not None:
            metadata += (("cu_seqlens_k", cu_seqlens_k),)
        for name, tensor in metadata:
            if (not isinstance(tensor, torch.Tensor) or tensor.ndim != 1 or tensor.dtype != torch.int32
                    or tensor.device != device or tensor.layout != torch.strided or not tensor.is_contiguous()):
                raise ValueError(f"{name} must be contiguous device int32 metadata")
        batch = cu_seqlens_q.numel() - 1
        if (batch < 0 or tokens > 0 and batch == 0 or kv_indptr.shape != cu_seqlens_q.shape
                or kv_last_page_lens.numel() != batch
                or cu_seqlens_k is not None and cu_seqlens_k.shape != cu_seqlens_q.shape):
            raise ValueError("inconsistent batch metadata shapes")
        if tokens and (max_seqlen_k == 0 or K.shape[0] == 0 or kv_page_indices.numel() == 0):
            raise NotImplementedError("active sequences require nonempty KV")
        for name, tensor in (("q_descale", q_descale), ("k_descale", k_descale), ("v_descale", v_descale)):
            if (not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.float32
                    or tensor.device != device or tensor.layout != torch.strided or not tensor.is_contiguous()):
                raise ValueError(f"{name} must be contiguous FP32 on the input GPU")
        if k_descale.numel() != 1 or v_descale.numel() != 1:
            raise ValueError("K/V descales must each contain one value")
        if q_descale.numel() != 1 and (self.quant_query_mode != "per-token" or q_descale.numel() != tokens * self.heads):
            raise ValueError("Q descale must be scalar or one value per token/head")
        scale = _softmax_scale(softmax_scale, self.dq**-0.5)
        out_shape, lse_shape = (tokens, self.heads, self.dv), (tokens, self.heads)
        if tokens * self.heads * self.dv * 2 >= 2**31:
            raise NotImplementedError("output byte span must fit signed int32 addressing")
        for name, tensor, shape, dtype in (("out", out, out_shape, torch.bfloat16), ("lse", lse, lse_shape, torch.float32)):
            if tensor is not None and (not isinstance(tensor, torch.Tensor) or tensor.shape != shape or tensor.dtype != dtype
                    or tensor.device != device or tensor.layout != torch.strided or not tensor.is_contiguous()):
                raise ValueError(f"{name} must be contiguous {dtype} {shape} on the input GPU")
        stream = torch.cuda.current_stream(device) if stream is None else stream
        if getattr(stream, "device", None) != device or not hasattr(stream, "cuda_stream"):
            raise ValueError("stream must belong to the input GPU")
        with torch.cuda.device(device), torch.cuda.stream(stream):
            if out is None:
                out = torch.empty(out_shape, device=device, dtype=torch.bfloat16)
            if return_lse and lse is None:
                lse = torch.empty(lse_shape, device=device, dtype=torch.float32)
            if tokens:
                args = (Q.view(-1), K.view(-1), V.view(-1), out.view(-1),
                        lse.view(-1) if lse is not None else k_descale.view(-1),
                        cu_seqlens_q, kv_indptr, kv_page_indices, kv_last_page_lens,
                        q_descale.view(-1), k_descale.view(-1), v_descale.view(-1),
                        self.heads, self.kv_heads, K.shape[0], batch, max_seqlen_q, self.dq, self.dv,
                        self.page_size, self.causal, q_descale.numel() != 1, lse is not None, scale,
                        self.persistent, properties.multi_processor_count, stream)
                signature = tuple((a.dtype, tuple(a.shape)) if isinstance(a, torch.Tensor)
                                  else ("stream",) if hasattr(a, "cuda_stream") else a for a in args)
                key = (device, signature)
                compiled = self._compiled.get(key)
                if compiled is None:
                    # flyc.compile also launches this first call.
                    self._compiled[key] = flyc.compile(self._launch, *args)
                else:
                    compiled(*args)
        return (out, lse) if return_lse else out


@functools.cache
def PagedAttention(num_qo_heads, num_kv_heads, head_dim_qk, head_dim_v, page_size,
                   is_causal, quant_query_mode="per-token", key_layout="vectorized",
                   window_left=-1, has_sink=False, *, memory_mode="lds", persistent=None):
    """FP8-derived BF16 body; persistent=None/True uses a fixed resident grid.

    persistent=False selects FP8's ordinary head/batch/query-tile grid. Both
    paths use the same eight-stage kernel and require no scheduler workspace.
    Dq/Dv128/192 use BM256. Dq=Dv256 defaults to BM128 M16 K+V DMA,
    early page lookahead and the S4 completion wait after its barrier.
    """
    if memory_mode != "lds" or key_layout != "vectorized":
        raise NotImplementedError("gfx942 BF16 supports LDS and vectorized K only")
    if persistent is not None and not isinstance(persistent, bool):
        raise ValueError("persistent must be bool or None")
    if not isinstance(is_causal, bool) or not isinstance(has_sink, bool):
        raise ValueError("is_causal and has_sink must be bools")
    if window_left != -1 or has_sink:
        raise NotImplementedError("gfx942 BF16 full MHA does not support SWA or sinks")
    if any(not isinstance(x, int) or isinstance(x, bool) for x in
           (num_qo_heads, num_kv_heads, head_dim_qk, head_dim_v, page_size)):
        raise ValueError("head counts, dimensions and page_size must be integers")
    if num_qo_heads <= 0 or num_kv_heads <= 0 or num_qo_heads % num_kv_heads:
        raise ValueError("query heads must be a positive multiple of KV heads")
    supported_dims = (head_dim_qk in (128, 192) and head_dim_v in (128, 192)) or head_dim_qk == head_dim_v == 256
    if not supported_dims or page_size not in (32, 64, 128):
        raise NotImplementedError("gfx942 BF16 supports Dq/Dv128/192 or Dq=Dv256 and pages32/64/128")
    if quant_query_mode not in ("per-token", "per-tensor"):
        raise ValueError("query scale mode must be per-token or per-tensor")
    return _PagedAttention(num_qo_heads, num_kv_heads, head_dim_qk, head_dim_v, page_size,
                           is_causal, quant_query_mode, persistent is not False)


# ---- D256-only linear/page1/page4 public API; not SHUFFLE-5D KV ----

_VALIDATED = {}


def _linear_run():
    # Imported on first use: listing or collecting tests must not build the D256 kernel module.
    from .mha_pa_bf16_256_linear_942 import run

    return run


def _validate_metadata(tensors, shapes, limits):
    cq, ck = tensors[:2]
    table = tensors[2] if len(tensors) == 3 else None
    key = (tuple(map(id, tensors)), shapes, limits)
    try:
        version = tuple((t._version, t.device, tuple(t.shape), t.data_ptr()) for t in tensors)
    except RuntimeError as exc:
        raise ValueError("Metadata must have version counters; create it outside inference_mode().") from exc
    cached = _VALIDATED.get(key)
    if cached is not None and cached[1] == version and all(
        ref() is t for ref, t in zip(cached[0], tensors)
    ):
        return
    import torch

    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Unwarmed or changed metadata: warm this call outside graph capture first.")
    qb = cq.cpu().tolist()
    kb = qb if ck is cq else ck.cpu().tolist()
    nq, nk = shapes[0][0], shapes[1][0]
    max_q, max_k, causal, page = limits
    if (qb[0] != 0 or kb[0] != 0 or qb[-1] != nq
            or table is None and kb[-1] != nk):
        raise ValueError("Bounds must start at zero and end at Q/K token counts (K is logical when paged).")
    qlens = [end - start for start, end in zip(qb, qb[1:])]
    klens = [end - start for start, end in zip(kb, kb[1:])]
    if (any(not 0 <= n <= max_q for n in qlens)
            or any(not 0 <= n <= max_k for n in klens)):
        raise ValueError("Bounds must be monotonic and sequence lengths must not exceed their maxima.")
    if any(qn > 0 and (kn == 0 or causal and kn < qn) for qn, kn in zip(qlens, klens)):
        raise ValueError("Active Q requires nonempty KV; bottom-right causal attention requires KV >= Q.")
    if table is not None:
        if nk % page:
            raise ValueError("With block_table, physical K/V tokens must be a multiple of page_size.")
        for length, row in zip(klens, table.cpu().tolist()):
            count = (length + page - 1) // page
            if count > len(row) or any(p < 0 or p >= nk // page for p in row[:count]):
                raise ValueError("block_table has missing columns or invalid active physical page IDs.")
    _VALIDATED[key] = (
        tuple(weakref.ref(t, lambda _, key=key: _VALIDATED.pop(key, None)) for t in tensors),
        version,
    )


def _overlaps(a, b):
    na, nb = a.numel() * a.element_size(), b.numel() * b.element_size()
    # data_ptr includes view offsets; Python integers preserve full 64-bit pointers.
    return bool(na and nb and a.data_ptr() <= b.data_ptr() + nb - 1
                and b.data_ptr() <= a.data_ptr() + na - 1)


def flash_attn_varlen_func(
    q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
    min_seqlen_q=0, dropout_p=0.0, softmax_scale=None, logits_soft_cap=0.0,
    causal=False, window_size=(-1, -1, 0), bias=None, alibi_slopes=None,
    deterministic=False, return_lse=False, return_attn_probs=False,
    how_v3_bf16_cvt=1, block_table=None, out=None, cu_seqlens_q_padded=None,
    cu_seqlens_k_padded=None, sink_ptr=None, layout="linear", key_layout=None,
    num_waves=8, *, page_size=1, persistent=None, stream=None,
):
    """Return BF16 O[NQ,H,256], or (O, FP32 LSE[NQ,H]) with return_lse.

    Q/K/V are contiguous token-major BF16; H must be a positive multiple of HK.
    Pages are consecutive groups of physical K/V rows; table entries may permute
    them arbitrarily. Without a table, page_size imposes no padding requirement.
    Missing CK means self-attention with CQ boundaries and equal Q/K token counts.
    Causal masking is bottom-right. Q/K/V/O pointers must be 16-byte aligned.
    K/V rows within each KV length must be finite: rows of a causal tile share its
    V, so a non-finite value also reaches rows that mask its key.
    Warm metadata and the core specialization before capture; rewarm after edits.
    Metadata must track PyTorch versions. persistent=None selects True.
    """
    import torch

    unsupported = [name for name, value, default in (
        ("min_seqlen_q", min_seqlen_q, 0), ("dropout_p", dropout_p, 0.0),
        ("logits_soft_cap", logits_soft_cap, 0.0), ("deterministic", deterministic, False),
        ("return_attn_probs", return_attn_probs, False), ("how_v3_bf16_cvt", how_v3_bf16_cvt, 1),
    ) if not isinstance(value, Real) or value != default]
    unsupported += [name for name, value in (
        ("bias", bias), ("alibi_slopes", alibi_slopes), ("sink_ptr", sink_ptr),
        ("cu_seqlens_q_padded", cu_seqlens_q_padded), ("cu_seqlens_k_padded", cu_seqlens_k_padded),
    ) if value is not None]
    if (not isinstance(window_size, (tuple, list))
            or not all(isinstance(x, Real) for x in window_size)
            or tuple(window_size) not in ((-1, -1), (-1, -1, 0))):
        unsupported.append("window_size/sinks")
    if (not isinstance(layout, str) or layout.lower() != "linear"
            or key_layout is not None and (not isinstance(key_layout, str) or key_layout.lower() != "linear")):
        unsupported.append("nonlinear layout/key_layout")
    if unsupported:
        raise NotImplementedError("D256 varlen attention does not support: " + ", ".join(unsupported))
    if not isinstance(causal, bool) or not isinstance(return_lse, bool):
        raise ValueError("causal and return_lse must be bools")
    if persistent is not None and not isinstance(persistent, bool):
        raise ValueError("persistent must be bool or None")
    if type(num_waves) is not int or num_waves != 8:
        raise NotImplementedError("Only num_waves=8 is supported")
    if type(page_size) is not int or page_size not in (1, 4):
        raise NotImplementedError("Only page_size=1 or 4 is supported")
    for name, value in (("max_seqlen_q", max_seqlen_q), ("max_seqlen_k", max_seqlen_k)):
        if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value < 2**31:
            raise ValueError(f"{name} must be a nonnegative signed-int32 host bound")
    scale = _softmax_scale(softmax_scale, 1.0 / 16)
    if not isinstance(q, torch.Tensor) or not q.is_cuda:
        raise ValueError("Q must be a device tensor on gfx942")
    device = q.device

    def check(name, tensor, dtype):
        if (not isinstance(tensor, torch.Tensor) or tensor.device != device or tensor.dtype != dtype
                or tensor.layout != torch.strided or not tensor.is_contiguous()):
            raise ValueError(f"{name} must be contiguous {dtype} on the Q GPU")
        if tensor.numel() * tensor.element_size() >= 2**31:
            raise ValueError(f"{name} byte span must be < 2**31")
        if tensor.requires_grad:
            raise NotImplementedError("Backward/autograd is not supported")

    for name, tensor in (("Q", q), ("K", k), ("V", v)):
        check(name, tensor, torch.bfloat16)
        if tensor.ndim != 3 or tensor.shape[-1] != 256:
            raise NotImplementedError("Q/K/V must be linear [tokens, heads, 256]")
        if tensor.data_ptr() % 16:
            raise ValueError(f"{name} data_ptr must be 16-byte aligned")
    if k.shape != v.shape or q.shape[1] <= 0 or k.shape[1] <= 0 or q.shape[1] % k.shape[1]:
        raise ValueError("K/V shapes must match; Q heads must be a positive multiple of KV heads")
    if cu_seqlens_k is None:
        if block_table is not None or q.shape[0] != k.shape[0]:
            raise ValueError("cu_seqlens_k=None requires equal-token self-attention without block_table")
        cu_seqlens_k = cu_seqlens_q
    metadata = (cu_seqlens_q, cu_seqlens_k) + (() if block_table is None else (block_table,))
    for name, tensor in zip(("cu_seqlens_q", "cu_seqlens_k", "block_table"), metadata):
        check(name, tensor, torch.int32)
    if (cu_seqlens_q.ndim != 1 or cu_seqlens_q.numel() < 1
            or cu_seqlens_k.shape != cu_seqlens_q.shape):
        raise ValueError("CQ/CK must have equal 1D shapes [B+1]")
    if block_table is not None and (block_table.ndim != 2 or block_table.shape[0] != cu_seqlens_q.numel() - 1):
        raise ValueError("block_table must have shape [B, max_pages]")
    if getattr(torch.cuda.get_device_properties(device), "gcnArchName", "").split(":", 1)[0] != "gfx942":
        raise NotImplementedError("This BF16 D256 backend requires gfx942")
    stream = torch.cuda.current_stream(device) if stream is None else stream
    if not isinstance(stream, torch.cuda.Stream) or stream.device != device:
        raise ValueError("stream must be a torch CUDA/ROCm stream on the Q GPU")
    with torch.cuda.device(device), torch.cuda.stream(stream):
        _validate_metadata(metadata, (tuple(q.shape), tuple(k.shape)),
                           (max_seqlen_q, max_seqlen_k, causal, page_size))
        if out is None:
            out = torch.empty_like(q)
        check("out", out, torch.bfloat16)
        if out.shape != q.shape or out.data_ptr() % 16:
            raise ValueError("out must match Q's shape and have a 16-byte-aligned data_ptr")
        lse = torch.empty(q.shape[:2], dtype=torch.float32, device=device) if return_lse else None
        sources = (q, k, v) + metadata + (() if lse is None else (lse,))
        if any(_overlaps(out, tensor) for tensor in sources):
            raise ValueError("out must not overlap Q/K/V, metadata, or LSE")
        return _linear_run()(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                           out=out, lse=lse, page_size=page_size, causal=causal, softmax_scale=scale,
                           block_table=block_table, persistent=persistent is not False, stream=stream)


__all__ = ["PagedAttention", "flash_attn_varlen_func"]
