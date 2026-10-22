"""gfx942 BF16 Dq = Dv = 256 paged attention over SHUFFLE-5D K/V (page 32/64/128).

BM128/BN64 with 16x16x16 MFMA; eight wave64s in two groups staggered by one stage.
Q stays in registers. K and V tiles (32 KiB each) reach LDS through direct DMA
(``buffer_load ... lds``, one dword per lane and packet): V(t-1) packets are
interleaved with the K(t) LDS reads and K(t+1) packets with the V(t-1) reads.
Page ids are prefetched two tiles ahead with SMEM; the S4 LDS wait is placed
after its barrier, just before the first consumer. ``_body`` holds the
pipeline table; mha_pa_bf16_942.PagedAttention dispatches here for D256.
"""

import math

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import gpu, rocdl

from ._common import (
    BM, BN, D, K_BYTES, LDS_BYTES, LOG2E, THREADS,
    _advance_max, _buffer, _center, _cross, _exps, _join, _mask, _max, _maximum, _min, _pack,
    _pack_bf16, _page_ready, _pin, _pin_i32, _pin_s64, _pv, _qk, _read_address, _read_page_g2r,
    _read_q_g2r, _rescale, _schedule, _stage_end, _sum, _uniform, _v_tail, _wait,
)

K_PITCH = 1024  # LDS bytes between the K rows of one wave


def _read_tile_pages_g2r(table, tile, last, kv_len, page_size):
    """Page ids (first, second) of a KV tile: BN64 spans two page32 pages."""
    first = _min(tile, last) * BN // page_size
    page0 = _read_page_g2r(table, first)
    page1 = _read_page_g2r(table, _min(first + 1, (kv_len - 1) // page_size)) if page_size == 32 else page0
    return page0, page1


def _read_k_s2r(address, half):
    """Keys 32 * half .. 32 * half + 31 as the QK A operand."""
    parts = [_read_address(address[n], half * 512 + k * (4 * K_PITCH))
             for n in range(2) for k in range(8)]
    words = fx.Vector.from_elements([p[i] for p in parts for i in range(4)], fx.Int32)
    storage = fx.make_rmem_tensor(fx.make_layout((4, 16, 2), (1, 4, 64)), fx.BFloat16)
    storage.store(words.bitcast(fx.BFloat16))
    return fx.make_view(fx.get_iter(storage), fx.make_layout((4, 2, 16), (1, 64, 4)))


def _read_v_s2r(address, half):
    """V dims 128 * half .. 128 * half + 127 as the PV A operand."""
    parts = [_read_address(address, half * 2048 + n * 256 + k * 16384)
             for n in range(8) for k in range(2)]
    words = fx.Vector.from_elements([p[i] for p in parts for i in range(4)], fx.Int32)
    frag = fx.make_rmem_tensor(fx.make_layout((4, 4, 8), (1, 4, 16)), fx.BFloat16)
    frag.store(words.bitcast(fx.BFloat16))
    return frag


def _dma_dword_g2s(resource, storage, destination, voffset, soffset=0):
    # destination is wave-uniform m0; hardware writes DWORD at m0+4*lane.
    rocdl.raw_ptr_buffer_load_lds(
        resource, fx.to_llvm_ptr(fx.get_iter(storage) + destination),
        fx.Int32(4).ir_value(), fx.Int32(voffset).ir_value(), fx.Int32(soffset).ir_value(),
        fx.Int32(0).ir_value(), fx.Int32(0).ir_value(),
    )


def _k_g2s_voffset(tid, page_size):
    wave, lane = tid >> 6, tid & 63
    return wave * (page_size * 16) + ((lane >> 2) ^ ((wave & 1) * 4)) * 16 + (lane & 3) * 4


def _k_g2s_destination(wave, packet):
    return wave * K_PITCH + (packet // 4) * 8192 + (packet % 4) * 256


def _k_g2s_soffset(page, second_page, tile, packet, hk, page_size):
    physical = second_page if page_size == 32 and packet % 4 >= 2 else page
    token_group = (packet % 4) % (page_size // 16)
    return physical * (hk * page_size * D * 2) + (tile * BN % page_size) * 16 + token_group * 256 + (packet // 4) * page_size * 128


def _read_k_packet_g2s(resource, storage, wave, voffset, page, second_page, tile, hk, page_size, packet):
    _dma_dword_g2s(resource, storage, _k_g2s_destination(wave, packet), voffset,
                   _k_g2s_soffset(page, second_page, tile, packet, hk, page_size))


def _v_tile_byte_offset(page, tile, hk, page_size):
    return page * (hk * page_size * D * 2) + (tile * BN % page_size) * (D * 2)


def _v_g2s_voffset(tid):
    return tid * 4


def _v_g2s_soffset(packet, page_size):
    return (packet % 8 if page_size == 32 else packet) * 2048


def _v_tile_resources(base_address, page, second_page, tile, hk, page_size):
    """Descriptors of one V tile, rebuilt from the full 64-bit address of its page(s)."""
    def resource(address, extent):
        pointer = llvm.inttoptr(ir.Type.parse("!llvm.ptr"), _pin_s64(address).ir_value())
        return rocdl.make_buffer_rsrc(ir.Type.parse("!llvm.ptr<8>"), pointer,
            fx.Int16(0).ir_value(), fx.Int64(extent).ir_value(), fx.Int32(0x27000).ir_value())
    # Widen BEFORE page-stride multiplication; only tile-relative offsets
    # enter the buffer instruction.
    first = resource(base_address + _v_tile_byte_offset(fx.Int64(page), fx.Int64(tile), hk, page_size),
                     min(page_size, BN) * D * 2)
    second = resource(base_address + _v_tile_byte_offset(fx.Int64(second_page), fx.Int64(tile), hk, page_size),
                      page_size * D * 2) if page_size == 32 else first
    return first, second


def _read_v_packet_g2s(resources, storage, wave, voffset, page_size, packet):
    resource = resources[1] if page_size == 32 and packet >= 8 else resources[0]
    destination = fx.Int32(llvm.inline_asm(fx.Int32.ir_type,
        [fx.Int32(wave * 256).ir_value()],
        f"s_add_u32 $0, $1, {K_BYTES + packet * 2048}", "=s,s,~{scc}", has_side_effects=True))
    offset = fx.Int32(llvm.inline_asm(fx.Int32.ir_type, [],
        f"s_mov_b32 $0, {_v_g2s_soffset(packet, page_size)}", "=s", has_side_effects=True))
    _dma_dword_g2s(resource, storage, destination, voffset, offset)


def _read_k_s2r_v_g2s(address, half, resources, storage, wave, voffset, page_size):
    """s2r K keys 32 * half .. + 31, with the 16 V(t-1) DMA packets issued between the reads."""
    parts = []
    for n in range(2):
        for k in range(8):
            parts.append(_read_address(address[n], half * 512 + k * (4 * K_PITCH)))
            _read_v_packet_g2s(resources, storage, wave, voffset, page_size, n * 8 + k)
    words = fx.Vector.from_elements([p[i] for p in parts for i in range(4)], fx.Int32)
    fragment = fx.make_rmem_tensor(fx.make_layout((4, 16, 2), (1, 4, 64)), fx.BFloat16)
    fragment.store(words.bitcast(fx.BFloat16))
    return fx.make_view(fx.get_iter(fragment), fx.make_layout((4, 2, 16), (1, 64, 4)))


def _read_v_s2r_k_g2s(address, half, resource, storage, wave, voffset, page, second_page,
                      tile, hk, page_size):
    """s2r V dims 128 * half .. + 127, with the 16 K(t+1) DMA packets issued between the reads."""
    parts = []
    for n in range(8):
        for k in range(2):
            parts.append(_read_address(address, half * 2048 + n * 256 + k * 16384))
            _read_k_packet_g2s(resource, storage, wave, voffset, page, second_page, tile, hk, page_size, n * 2 + k)
    words = fx.Vector.from_elements([p[i] for p in parts for i in range(4)], fx.Int32)
    frag = fx.make_rmem_tensor(fx.make_layout((4, 4, 8), (1, 4, 16)), fx.BFloat16)
    frag.store(words.bitcast(fx.BFloat16))
    return frag


def _store_o_tile_r2g(o0, o1, inv, storage, obuf, out_tid, out_row, shared, heads):
    """O * inv to BF16, C-shuffled through LDS (r2s b64, s2r b128), then b128 row stores (r2g)."""
    outputs = _join(o0, o1)
    atom = fx.make_copy_atom(rocdl.BufferCopy128b(), fx.BFloat16)
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
            offset = read_row * (heads * D) + half * 64 + read_col
            fx.copy(atom, src, fx.make_view(fx.get_iter(obuf) + offset, fx.make_layout(8, 1)))
        _stage_end()


@flyc.jit
def _body(Q, K, V, O, LSE, QS, KS, VS, table, storage, q0, q_len, kv_len, head, qb,
          H: fx.Constexpr[int], HK: fx.Constexpr[int], NP: fx.Constexpr[int], PAGE: fx.Constexpr[int],
          CAUSAL: fx.Constexpr[bool], PER_TOKEN: fx.Constexpr[bool], WITH_LSE: fx.Constexpr[bool],
          SCALE: fx.Constexpr[float], STAGGER: fx.Constexpr[bool]):
    """One task: query rows 128 * qb .. + 127 of one head over its KV tiles t of BN = 64 keys.

    LDS holds one K tile (32 KiB) and one V tile (32 KiB). Page ids run two tiles ahead
    of QK and K one tile ahead; V(t-1) arrives during phase t. Phase t runs QK and the
    row max of tile t and, one tile behind, the exponentials, row sum and PV of tile
    t - 1. A phase has four steps, each a memory stage then a compute stage; waves 4-7
    (STAGGER) run one stage behind waves 0-3, so one group's memory stage overlaps the
    other group's compute stage.

    step  memory stage                                compute stage
    0     s_load pages(t+2); g2s V(t-1) with s2r      QK: S(t) keys 0-31
          K(t) keys 0-31 between its packets          softmax: P(t-1) = exp2(S'(t-1))
    1     s2r K(t) keys 32-63; V(t-1) lands           QK: S(t) keys 32-63
                                                      softmax: row sum of P(t-1), P to BF16
    2     exchange row sums; g2s K(t+1) with s2r      PV: O dims 0-127 += P(t-1) V(t-1)
          V(t-1) dims 0-127 between packets           softmax: l += row sum; mask S(t), max
    3     exchange row maxima; K(t+1) lands;          PV: O dims 128-255 += P(t-1) V(t-1)
          s2r V(t-1) dims 128-255                     softmax: m, S'(t); rescale O and l

    Softmax in log2 units: S' = S * scale - m, P = exp2(S'), l = sum(P) and O = sum(P V);
    the output is O / l * v_descale. m starts at the first row max + 1 and moves to a
    later row max + 1 only when that exceeds m + 7, rescaling O and l.
    """
    # FlyDSL keys its disk cache on helpers named in this function's own code, not in closures.
    _helpers = (_read_k_packet_g2s, _read_k_s2r, _read_v_s2r, _read_k_s2r_v_g2s, _read_v_s2r_k_g2s,
                _read_v_packet_g2s, _v_tile_resources, _qk, _pv, _sum, _max, _cross, _center, _exps, _pack,
                _mask, _v_tail, _rescale, _advance_max, _join, _maximum, _pin, _schedule, _stage_end, _wait)
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
    # Q stays in registers for every KV tile.
    q = _read_q_g2r(gq, (wave * 16 + (lane & 15)) * (H * D * 2), lane)
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
    # Page ids of tiles 0-2; every later phase t requests pages(t+2).
    pages_0 = _read_tile_pages_g2r(table, fx.Int32(0), last, kv_len, PAGE)
    pages_1 = _read_tile_pages_g2r(table, fx.Int32(1), last, kv_len, PAGE)
    pages_2 = _read_tile_pages_g2r(table, fx.Int32(2), last, kv_len, PAGE)
    pages_0 = _page_ready(pages_0[0]), _page_ready(pages_0[1])
    pages_1 = _page_ready(pages_1[0]), _page_ready(pages_1[1])
    pages_2 = _page_ready(pages_2[0]), _page_ready(pages_2[1])
    # Loop-invariant addresses: K/V LDS read bases (s2r), DMA lane offsets (g2s), lane exchanges.
    shared = fx.Int32(fx.ptrtoint(fx.get_iter(storage)))
    key_row = (lane & 3) + ((lane & 12) << 1)
    k_s2r = tuple(_pin_i32(shared + (lane >> 4) * K_PITCH + (((key_row + n * 4) * 16) ^ (((lane >> 4) & 1) * 64)))
                  for n in range(2))
    k_g2s = _pin_i32(_k_g2s_voffset(tid, PAGE))
    v_s2r = _pin_i32(shared + K_BYTES + (lane >> 4) * 4096 + (lane & 15) * 16)
    v_g2s = _pin_i32(_v_g2s_voffset(tid))
    v_base = _pin_s64(fx.Int64(fx.ptrtoint(fx.get_iter(V))) + fx.Int64(hkv) * (PAGE * D * 2))
    cross_addresses = tuple(_pin_i32((lane ^ offset) * 4) for offset in (16, 32, 48))

    def read_k_g2s(pages, tile):
        for packet in fx.range_constexpr(16):
            _read_k_packet_g2s(gk, storage, wave, k_g2s, pages[0], pages[1], tile, HK, PAGE, packet)

    def prologue():
        """K(0) g2s; QK S(0), m and S'(0); K(1) g2s."""
        # Memory: g2s K(0).
        read_k_g2s(pages_0, fx.Int32(0))
        _wait(vmcnt=0)
        _wait(lgkmcnt=0)
        _stage_end()
        if fx.const_expr(STAGGER):
            # Waves 4-7 fall one stage behind.
            _stage_end()
        # Memory: s2r K(0) keys 0-31.
        k_lo = _read_k_s2r(k_s2r, 0)
        _wait(lgkmcnt=0)
        _stage_end()
        # Compute: QK S(0) keys 0-31; O = 0.
        s_lo = _qk(q, k_lo)
        o0 = _pin(fx.Vector.filled(32, 0.0, fx.Float32))
        o1 = _pin(fx.Vector.filled(32, 0.0, fx.Float32))
        _schedule(32, 3, 5)
        _stage_end()
        # Memory: s2r K(0) keys 32-63.
        k_hi = _read_k_s2r(k_s2r, 1)
        _wait(lgkmcnt=0)
        _stage_end()
        # Compute: QK S(0) keys 32-63; softmax m and S'(0).
        s_hi = _qk(q, k_hi)
        s_0 = _mask(_join(s_lo, s_hi), fx.Int32(0), row, q_len, kv_len, CAUSAL)
        row_max = _max(s_0)
        row_max = _maximum(row_max, row_max.shuffle_xor(16, 64))
        row_max = _maximum(row_max, row_max.shuffle_xor(32, 64))
        row_max = _maximum(row_max * scale, fx.Float32(-1.0e30)) + 1.0
        s_0 = _center(s_0, scale, row_max)
        row_sum = fx.Float32(0.0)
        _stage_end()
        # Memory: g2s K(1); no compute.
        read_k_g2s(pages_1, _min(fx.Int32(1), last))
        _wait(vmcnt=0)
        _wait(lgkmcnt=0)
        _stage_end()
        _stage_end()
        return s_0, row_max, row_sum, o0, o1, *pages_0, *pages_1, *pages_2

    @flyc.jit
    def phase(s_prev, row_max, row_sum, o0, o1, page_prev0, page_prev1, page_cur0, page_cur1,
              page_next0, page_next1, t):
        """Phase t (table in _body); page_prev*, page_cur*, page_next* are the page ids of tiles t-1 .. t+1."""
        s_prev, row_max, row_sum = fx.Vector(s_prev), fx.Float32(row_max), fx.Float32(row_sum)
        o0, o1, t = fx.Vector(o0), fx.Vector(o1), fx.Int32(t)
        # Step 0 memory: s_load pages(t+2); g2s V(t-1) between the s2r of K(t) keys 0-31.
        requests_next2 = _read_tile_pages_g2r(table, t + 2, last, kv_len, PAGE)
        v_prev_resources = _v_tile_resources(v_base, page_prev0, page_prev1, t - 1, HK, PAGE)
        k_lo = _read_k_s2r_v_g2s(k_s2r, 0, v_prev_resources, storage, wave, v_g2s, PAGE)
        page_next2_0 = _page_ready(requests_next2[0])
        if fx.const_expr(PAGE != 32):
            page_next2_1 = page_next2_0
        else:
            page_next2_1 = _page_ready(requests_next2[1])
        _stage_end()
        # Step 0 compute: QK S(t) keys 0-31; P(t-1) = exp2(S'(t-1)).
        s_lo = _qk(q, k_lo)
        p_prev = _exps(s_prev)
        _schedule(32, 1, 1, True)
        _stage_end()
        # Step 1 memory: s2r K(t) keys 32-63; V(t-1) lands.
        k_hi = _read_k_s2r(k_s2r, 1)
        _wait(vmcnt=0)
        _wait(lgkmcnt=0)
        _stage_end()
        # Step 1 compute: QK S(t) keys 32-63; row sum of P(t-1), P to BF16.
        s_hi = _qk(q, k_hi)
        lane_sum = _sum(p_prev)
        p_bf16 = _pack(p_prev)
        _schedule(32, 3, 2)
        _stage_end()
        # Step 2 memory: exchange row sums; g2s K(t+1) between the s2r of V(t-1) dims 0-127.
        other_sums = _cross(lane_sum, cross_addresses)
        v_lo = _read_v_s2r_k_g2s(v_s2r, 0, gk, storage, wave, k_g2s, page_next0, page_next1,
                                 _min(t + 1, last), HK, PAGE)
        _stage_end()
        # Step 2 compute: the barrier does not cover the V reads; retire them before the first PV.
        _wait(lgkmcnt=0)
        rocdl.sched_barrier(0)
        o0 = _pv(p_bf16, v_lo, o0)
        row_sum = row_sum + ((lane_sum + other_sums[0]) + (other_sums[1] + other_sums[2]))
        s_cur = _mask(_join(s_lo, s_hi), t, row, q_len, kv_len, CAUSAL)
        lane_max = _max(s_cur)
        _schedule(32, 2, 3)
        _stage_end()
        # Step 3 memory: exchange row maxima; K(t+1) lands; s2r V(t-1) dims 128-255.
        other_maxima = _cross(lane_max, cross_addresses)
        _wait(vmcnt=0)
        v_hi = _read_v_s2r(v_s2r, 1)
        _wait(lgkmcnt=0)
        _stage_end()
        # Step 3 compute: PV O dims 128-255; new m and S'(t); lazy rescale of O and l.
        o1 = _pv(p_bf16, v_hi, o1)
        candidate = _maximum(_maximum(lane_max, other_maxima[0]), _maximum(other_maxima[1], other_maxima[2])) * scale
        ballot = fx.Int64(rocdl.ballot(fx.Int64.ir_type, (candidate > row_max + 7.0).ir_value()))
        new_max = (candidate > row_max + 7.0).select(candidate + 1.0, row_max)
        s_cur = _center(s_cur, scale, new_max)
        _schedule(32, 3, 4)
        o0, o1, row_sum = _rescale(o0, o1, row_sum, row_max, new_max, ballot)
        new_max = _advance_max(row_max, new_max)
        _stage_end()
        return (s_cur, new_max, row_sum, o0, o1, page_cur0, page_cur1, page_next0, page_next1,
                page_next2_0, page_next2_1)

    def epilogue(state):
        """g2s V(last) with zero rows past kv_len; exp, row sum and PV of the last tile."""
        s_last, row_max, row_sum, o0, o1, page_last0, page_last1 = state[:7]
        # Memory: g2s V(last), with P(last) and l computed while it lands.
        v_last_resources = _v_tile_resources(v_base, page_last0, page_last1, last, HK, PAGE)
        for packet in fx.range_constexpr(16):
            _read_v_packet_g2s(v_last_resources, storage, wave, v_g2s, PAGE, packet)
        p_last = _exps(fx.Vector(s_last))
        lane_sum = _sum(p_last)
        lane_sum = lane_sum + lane_sum.shuffle_xor(16, 64)
        lane_sum = lane_sum + lane_sum.shuffle_xor(32, 64)
        row_sum = fx.Float32(row_sum) + lane_sum
        p_bf16 = _pack(p_last)
        _wait(vmcnt=0, lgkmcnt=0)
        _stage_end()
        _stage_end()
        # Memory: s2r V(last) dims 0-127, zeroing rows past kv_len (0 * NaN is NaN).
        v_lo = _read_v_s2r(v_s2r, 0)
        _wait(lgkmcnt=0)
        rocdl.sched_barrier(0)
        v_lo.store(_v_tail(v_lo.load(), kv_len - last * BN))
        _stage_end()
        # Compute: PV O dims 0-127.
        o0 = _pv(p_bf16, v_lo, o0)
        _stage_end()
        # Memory: s2r V(last) dims 128-255.
        v_hi = _read_v_s2r(v_s2r, 1)
        _wait(lgkmcnt=0)
        rocdl.sched_barrier(0)
        v_hi.store(_v_tail(v_hi.load(), kv_len - last * BN))
        _stage_end()
        # Compute: PV O dims 128-255.
        o1 = _pv(p_bf16, v_hi, o1)
        _stage_end()
        if fx.const_expr(not STAGGER):
            # Waves 0-3 wait one stage for waves 4-7.
            _stage_end()
        return row_max, row_sum, o0, o1

    # Pipeline: pages two tiles ahead of QK, K one tile ahead; PV runs one tile behind.
    state = prologue()
    for t in range(fx.Int32(1), tiles, fx.Int32(1)):
        state = phase(*state, t)
    row_max, row_sum, o0, o1 = epilogue(state)

    inv = (row_sum > 0.0).select(fx.Float32(1.0) / row_sum, fx.Float32(0.0)) * VS[0]
    out_tid = _pin_i32(fx.Int32(gpu.thread_id("x")))
    out_row = (out_tid >> 6) * 16 + (out_tid & 15)
    optr = fx.get_iter(O) + ((fx.Int64(q0) + fx.Int64(q_start)) * (H * D) + fx.Int64(head) * D)
    obuf = rocdl.make_buffer_tensor(fx.make_view(optr, fx.make_layout(BM * H * D, 1)),
                                    num_records_bytes=valid * H * D * 2)
    _store_o_tile_r2g(o0, o1, inv, storage, obuf, out_tid, out_row, shared, H)
    if fx.const_expr(WITH_LSE):
        if ((out_tid & 63) < 16) & (out_row < valid):
            log_l = fx.Float32(llvm.call_intrinsic(fx.Float32.ir_type, "llvm.log2.f32", [row_sum.ir_value()], [], []))
            LSE[(q0 + q_start + out_row) * H + head] = (row_sum > 0.0).select(
                (fx.Float32(row_max) + log_l) * fx.Float32(math.log(2.0)), fx.Float32(float("-inf")))


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
def _attention_256_dma_kernel_942(Q: fx.Tensor, K: fx.Tensor, V: fx.Tensor, O: fx.Tensor, LSE: fx.Tensor,
    CQ: fx.Tensor, KI: fx.Tensor, PAGES: fx.Tensor, LAST: fx.Tensor, QS: fx.Tensor, KS: fx.Tensor, VS: fx.Tensor,
    H: fx.Constexpr[int], HK: fx.Constexpr[int], NP: fx.Constexpr[int], B: fx.Constexpr[int],
    MAX_Q: fx.Constexpr[int], PAGE: fx.Constexpr[int], CAUSAL: fx.Constexpr[bool],
    PER_TOKEN: fx.Constexpr[bool], WITH_LSE: fx.Constexpr[bool], SCALE: fx.Constexpr[float],
    PERSISTENT: fx.Constexpr[bool], CUS: fx.Constexpr[int]):
    work_body = _work
    storage = fx.SharedAllocator().allocate(fx.Array[fx.Int8, LDS_BYTES, 16]).peek().view(fx.make_layout(LDS_BYTES, 1))
    if fx.const_expr(PERSISTENT):
        work = fx.Int32(gpu.block_id("x"))
        while work < H * B * ((MAX_Q + BM - 1) // BM):
            head, batch, qb = work % H, (work // H) % B, work // (H * B)
            work_body(Q, K, V, O, LSE, CQ, KI, PAGES, LAST, QS, KS, VS, storage, head, batch, qb,
                      H, HK, NP, PAGE, CAUSAL, PER_TOKEN, WITH_LSE, SCALE)
            work = work + CUS
    else:
        work_body(Q, K, V, O, LSE, CQ, KI, PAGES, LAST, QS, KS, VS, storage,
                  fx.Int32(gpu.block_id("x")), fx.Int32(gpu.block_id("y")), fx.Int32(gpu.block_id("z")),
                  H, HK, NP, PAGE, CAUSAL, PER_TOKEN, WITH_LSE, SCALE)


@flyc.jit
def _launch(Q: fx.Tensor, K: fx.Tensor, V: fx.Tensor, O: fx.Tensor, LSE: fx.Tensor,
    CQ: fx.Tensor, KI: fx.Tensor, PAGES: fx.Tensor, LAST: fx.Tensor, QS: fx.Tensor, KS: fx.Tensor, VS: fx.Tensor,
    H: fx.Constexpr[int], HK: fx.Constexpr[int], NP: fx.Constexpr[int], B: fx.Constexpr[int],
    MAX_Q: fx.Constexpr[int], DQ: fx.Constexpr[int], DV: fx.Constexpr[int], PAGE: fx.Constexpr[int],
    CAUSAL: fx.Constexpr[bool], PER_TOKEN: fx.Constexpr[bool], WITH_LSE: fx.Constexpr[bool],
    SCALE: fx.Constexpr[float], PERSISTENT: fx.Constexpr[bool], CUS: fx.Constexpr[int], stream: fx.Stream):
    assert DQ == DV == D
    grid = (min(CUS, H * B * ((MAX_Q + BM - 1) // BM)), 1, 1) if PERSISTENT else (H, B, (MAX_Q + BM - 1) // BM)
    _attention_256_dma_kernel_942(Q, K, V, O, LSE, CQ, KI, PAGES, LAST, QS, KS, VS,
        H, HK, NP, B, MAX_Q, PAGE, CAUSAL, PER_TOKEN, WITH_LSE, SCALE, PERSISTENT, CUS,
        value_attrs={"rocdl.waves_per_eu": 2, "passthrough": [["target-features", "-packed-fp32-ops"]]},
    ).launch(grid=grid, block=(THREADS, 1, 1), stream=stream)
