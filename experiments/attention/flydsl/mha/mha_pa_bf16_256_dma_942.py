"""gfx942 D256 DMA pipeline and explicit scheduling experiments.

The public BF16 wrapper explicitly selects the validated v73 combination.
This module's own factory retains its original experimental defaults (V
GLOBAL). Optional V DMA uses a full-64-bit tile base; m0_offset compensates
that base before sharing M0 as the source offset and LDS destination.
The opt-in v98 pipeline adds late/progressive waits, even-D32-first K WAR
retirement, S7 PV/center overlap and persistent task order. M16 arithmetic,
K XOR layout, 64-KiB allocation and public tensor ABI are unchanged.
Importing or constructing an experiment does not mutate public defaults.
"""

import functools
import math

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu, rocdl
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm

if __package__:
    from . import mha_pa_bf16_256_942 as base
    from .mha_pa_bf16_942 import (
        PagedAttention as _validate_factory, _PagedAttention, _uniform, _min,
        _pin_i32, _pin_s64, _pin, _join, _maximum, _pack_bf16, _stage_end,
        _wait, _schedule, _page_ready, _buffer, _read_address, _rescale, _advance_max,
    )
else:
    import mha_pa_bf16_256_942 as base
    from mha_pa_bf16_942 import (
        PagedAttention as _validate_factory, _PagedAttention, _uniform, _min,
        _pin_i32, _pin_s64, _pin, _join, _maximum, _pack_bf16, _stage_end,
        _wait, _schedule, _page_ready, _buffer, _read_address, _rescale, _advance_max,
    )

BM, BN, D, THREADS = base.BM, base.BN, base.D, base.THREADS
K_PITCH, K_BYTES, V_BYTES, LDS_BYTES = base.K_PITCH, base.K_BYTES, base.V_BYTES, base.LDS_BYTES
LOG2E = base.LOG2E


def _dma4(resource, storage, destination, voffset, soffset=0):
    # destination is wave-uniform m0; hardware writes DWORD at m0+4*lane.
    rocdl.raw_ptr_buffer_load_lds(
        resource, fx.to_llvm_ptr(fx.get_iter(storage) + destination),
        fx.Int32(4).ir_value(), fx.Int32(voffset).ir_value(), fx.Int32(soffset).ir_value(),
        fx.Int32(0).ir_value(), fx.Int32(0).ir_value(),
    )


def _q_dma_offset(tid, heads):
    return ((tid >> 6) * 16 + ((tid & 63) >> 2)) * (heads * D * 2) + (tid & 3) * 4


def _q_dma_destination(wave, packet):
    return wave * 8192 + packet * 256


def _q_read_offset(tid):
    return (tid >> 6) * 8192 + ((tid & 63) >> 4) * 256 + (tid & 15) * 16


def _q_dma_fragment(resource, storage, tid, heads):
    # Keep prologue-only DMA address expressions out of the persistent loop's
    # live-in set. Otherwise LLVM hoists them and spills unrelated tail masks.
    tid = _pin_i32(tid)
    wave = _uniform(tid >> 6)
    voffset = _pin_i32(_q_dma_offset(tid, heads))
    address = _pin_i32(fx.Int32(fx.ptrtoint(fx.get_iter(storage))) + _q_read_offset(tid))
    # First 128 D values are ready before reading. Each retired 16-B Q
    # fragment is interleaved with four DWORD DMA for the other D128 half.
    for packet in range(16):
        _dma4(resource, storage, _q_dma_destination(wave, packet), voffset, packet * 16)
    _wait(vmcnt=0)
    parts = []
    for k in range(4):
        parts.append(_read_address(address, k * 1024))
        for j in range(4):
            packet = 16 + k * 4 + j
            _dma4(resource, storage, _q_dma_destination(wave, packet), voffset, packet * 16)
    _wait(vmcnt=0)
    for k in range(4, 8):
        parts.append(_read_address(address, k * 1024))
    _wait(lgkmcnt=0)
    # All eight waves retire Q reads before K/V reuse any of these bytes.
    _stage_end()
    words = fx.Vector.from_elements([part[i] for part in parts for i in range(4)], fx.Int32)
    frag = base._mma().make_fragment_B(fx.make_rmem_tensor(fx.make_layout((BM, D), (1, BM)), fx.BFloat16))
    frag.store(words.bitcast(fx.BFloat16))
    return frag


def _k_dma_offset(tid, page_size):
    wave, lane = tid >> 6, tid & 63
    # Invert the EXISTING K XOR permutation at the DMA's global source.
    # A 256-byte DMA wave writes 16 token packets, four DWORDs per token.
    return wave * (page_size * 16) + ((lane >> 2) ^ ((wave & 1) * 4)) * 16 + (lane & 3) * 4


def _k_dma_destination(wave, packet):
    return wave * K_PITCH + (packet // 4) * 8192 + (packet % 4) * 256


def _k_dma_soffset(page, second_page, tile, packet, hk, page_size):
    physical = second_page if page_size == 32 and packet % 4 >= 2 else page
    token_group = (packet % 4) % (page_size // 16)
    return physical * (hk * page_size * D * 2) + (tile * BN % page_size) * 16 + token_group * 256 + (packet // 4) * page_size * 128


def _dma_k(resource, storage, wave, voffset, page, second_page, tile, hk, page_size, packet):
    _dma4(resource, storage, _k_dma_destination(wave, packet), voffset,
          _k_dma_soffset(page, second_page, tile, packet, hk, page_size))


def _v_tile_byte_offset(page, tile, hk, page_size):
    return page * (hk * page_size * D * 2) + (tile * BN % page_size) * (D * 2)


def _v_dma_offset(tid):
    return tid * 4


def _v_dma_destination(wave, packet):
    return K_BYTES + wave * 256 + packet * 2048


def _v_dma_soffset(packet, page_size):
    return (packet % 8 if page_size == 32 else packet) * 2048


def _descriptor_words(address, extent):
    bits = fx.Uint64(address)
    return fx.Vector.from_elements([fx.Uint32(bits), fx.Uint32(bits >> 32) & fx.Uint32(0xFFFF),
                                   fx.Uint32(extent), fx.Uint32(0x27000)], fx.Uint32)


def _v_resources(base_address, page, second_page, tile, hk, page_size):
    def resource(address, extent):
        pointer = llvm.inttoptr(ir.Type.parse("!llvm.ptr"), _pin_s64(address).ir_value())
        return rocdl.make_buffer_rsrc(ir.Type.parse("!llvm.ptr<8>"), pointer,
            fx.Int16(0).ir_value(), fx.Int64(extent).ir_value(), fx.Int32(0x27000).ir_value())
    # Widen BEFORE page-stride multiplication; only tile-relative 0..32764B
    # offsets enter the buffer instruction. No pointer truncation to int32.
    first = resource(base_address + _v_tile_byte_offset(fx.Int64(page), fx.Int64(tile), hk, page_size),
                     min(page_size, BN) * D * 2)
    second = resource(base_address + _v_tile_byte_offset(fx.Int64(second_page), fx.Int64(tile), hk, page_size),
                      page_size * D * 2) if page_size == 32 else first
    return first, second


def _dma_v(resources, storage, wave, voffset, page_size, packet):
    resource = resources[1] if page_size == 32 and packet >= 8 else resources[0]
    # Keep the sixteen m0 destinations / large SOFFSET constants local to
    # their DMA. Hoisting both K and V sets caused eight SGPR spills to lanes.
    destination = fx.Int32(llvm.inline_asm(fx.Int32.ir_type,
        [fx.Int32(wave * 256).ir_value()],
        f"s_add_u32 $0, $1, {K_BYTES + packet * 2048}", "=s,s,~{scc}", has_side_effects=True))
    offset = fx.Int32(llvm.inline_asm(fx.Int32.ir_type, [],
        f"s_mov_b32 $0, {_v_dma_soffset(packet, page_size)}", "=s", has_side_effects=True))
    _dma4(resource, storage, destination, voffset, offset)


def _read_k_vdma(address, half, resources, storage, wave, voffset, page_size,
                  begin, count, interleave):
    # V writes and K reads occupy disjoint 32-KiB halves of the allocation.
    if not interleave:
        for packet in range(begin, begin + count):
            _dma_v(resources, storage, wave, voffset, page_size, packet)
    parts = []
    for n in range(2):
        for k in range(8):
            parts.append(_read_address(address[n], half * 512 + k * (4 * K_PITCH)))
            index = n * 8 + k
            if interleave and index % (16 // count) == (16 // count) - 1:
                _dma_v(resources, storage, wave, voffset, page_size, begin + index // (16 // count))
    words = fx.Vector.from_elements([part[i] for part in parts for i in range(4)], fx.Int32)
    fragment = fx.make_rmem_tensor(fx.make_layout((4, 16, 2), (1, 4, 64)), fx.BFloat16)
    fragment.store(words.bitcast(fx.BFloat16))
    return fx.make_view(fx.get_iter(fragment), fx.make_layout((4, 2, 16), (1, 64, 4)))


def _read_k_m0_vdma(address, resources, wave, page_size):
    """Share m0 as V SOFFSET; shift each bounded descriptor down by 32 KiB.

    With voffset=lane*4, base' + m0 + voffset is exactly the old V source.
    PAGE32's second descriptor is shifted by 48 KiB to compensate packet8.
    m0 still selects the unchanged LDS destination; no duplicate offset SALU.
    """
    # M0 is intentionally clobbered in this gfx942 leaf region (LLVM warns
    # because it is reserved). Every packet sets M0 before using it, with a
    # DS instruction providing hazard distance; subsequent intrinsic DMA
    # must set M0 anew. Actual emitted code and zero-spill gates are required.
    lane = _pin_i32((fx.Int32(gpu.thread_id("x")) & 63) * 4)
    parts = []
    for n in range(2):
        for k in range(8):
            packet = n * 8 + k
            resource = resources[1] if page_size == 32 and packet >= 8 else resources[0]
            parts.append(fx.Vector(llvm.inline_asm(ir.VectorType.get([4], fx.Int32.ir_type),
                [address[n].ir_value(), resource.ir_value(), lane.ir_value(), fx.Int32(wave * 256).ir_value()],
                f"s_add_u32 m0, $4, {K_BYTES + packet * 2048}\n"
                f"ds_read_b128 $0, $1 offset:{k * (4 * K_PITCH)}\n"
                "buffer_load_dword $3, $2, m0 offen lds",
                "=&v,v,s,v,s,~{m0},~{scc},~{memory}", has_side_effects=True)))
    words = fx.Vector.from_elements([part[i] for part in parts for i in range(4)], fx.Int32)
    frag = fx.make_rmem_tensor(fx.make_layout((4, 16, 2), (1, 4, 64)), fx.BFloat16)
    frag.store(words.bitcast(fx.BFloat16))
    return fx.make_view(fx.get_iter(frag), fx.make_layout((4, 2, 16), (1, 64, 4)))


def _m0_v_resources(base_address, page, second_page, tile, hk, page_size):
    first_address = base_address + _v_tile_byte_offset(fx.Int64(page), fx.Int64(tile), hk, page_size)
    first = _descriptor_words(_pin_s64(first_address - K_BYTES), K_BYTES + min(page_size, BN) * D * 2)
    if page_size == 32:
        second_address = base_address + _v_tile_byte_offset(fx.Int64(second_page), fx.Int64(tile), hk, page_size)
        second = _descriptor_words(_pin_s64(second_address - K_BYTES - 16384), LDS_BYTES)
    else:
        second = first
    return first, second


def _read_k_leading_first(address):
    # Leading DMA waves0..3 overwrite only even D32 planes. Retire both N16
    # reads of those planes before exposing the eight odd-plane reads.
    parts = {}
    for parity in range(2):
        for n in range(2):
            for k in range(parity, 8, 2):
                parts[n, k] = _read_address(address[n], 512 + k * (4 * K_PITCH))
    words = fx.Vector.from_elements([parts[n, k][i] for n in range(2) for k in range(8) for i in range(4)], fx.Int32)
    frag = fx.make_rmem_tensor(fx.make_layout((4, 16, 2), (1, 4, 64)), fx.BFloat16)
    frag.store(words.bitcast(fx.BFloat16))
    return fx.make_view(fx.get_iter(frag), fx.make_layout((4, 2, 16), (1, 64, 4)))


def _read_k_write_dma(address, publish_address, pending, resource, storage, wave,
                      voffset, page, second_page, tile, hk, page_size):
    # S2: K-low readers of BOTH groups have finished at the preceding barrier.
    # Read old K-high, publish V, DMA next K-low (disjoint token halves).
    parts = []
    for n in range(2):
        for k in range(8):
            parts.append(_read_address(address[n], 512 + k * (4 * K_PITCH)))
            if k % 4 == 3:
                packet = n * 2 + k // 4
                words = fx.Vector.from_elements([pending[packet * 4 + j] for j in range(4)], fx.Int32)
                llvm.inline_asm(ir.Type.parse("!llvm.void"), [publish_address.ir_value(), words.ir_value()],
                    f"ds_write_b128 $0, $1 offset:{packet * 8192}", "v,v,~{memory}", has_side_effects=True)
            if k % 2 == 1:
                ordinal = n * 4 + k // 2
                packet = (ordinal // 2) * 4 + ordinal % 2
                _dma_k(resource, storage, wave, voffset, page, second_page, tile, hk, page_size, packet)
    words = fx.Vector.from_elements([part[i] for part in parts for i in range(4)], fx.Int32)
    fragment = fx.make_rmem_tensor(fx.make_layout((4, 16, 2), (1, 4, 64)), fx.BFloat16)
    fragment.store(words.bitcast(fx.BFloat16))
    return fx.make_view(fx.get_iter(fragment), fx.make_layout((4, 2, 16), (1, 64, 4)))


def _read_v_dma(address, half, resource, storage, wave, voffset, page, second_page,
                tile, hk, page_size, begin, count, interleave):
    # Each selected DMA targets K; all DS loads below target disjoint V.
    if not interleave:
        for ordinal in range(count):
            packet = (ordinal // 2) * 4 + ordinal % 2 + 2 if begin == 16 else begin + ordinal
            _dma_k(resource, storage, wave, voffset, page, second_page, tile, hk, page_size, packet)
    parts = []
    for n in range(8):
        for k in range(2):
            parts.append(_read_address(address, half * 2048 + n * 256 + k * 16384))
            index = n * 2 + k
            if interleave and index % (16 // count) == (16 // count) - 1:
                ordinal = index // (16 // count)
                packet = (ordinal // 2) * 4 + ordinal % 2 + 2 if begin == 16 else begin + ordinal
                _dma_k(resource, storage, wave, voffset, page, second_page, tile, hk, page_size, packet)
    words = fx.Vector.from_elements([part[i] for part in parts for i in range(4)], fx.Int32)
    frag = fx.make_rmem_tensor(fx.make_layout((4, 4, 8), (1, 4, 16)), fx.BFloat16)
    frag.store(words.bitcast(fx.BFloat16))
    return frag


def _pv_progressive(probabilities, values, output, columns):
    """Consume oldest V packets without waiting for the entire LDS read train.

    There are sixteen ordered DS reads, two per N16 output. Only S4 or the
    leading S6 can enter here with outstanding reads; no SMEM is outstanding.
    Each group waits for ALL packets used by its MFMAs. The final group waits
    for zero before the scalar cross-row results or next rendezvous are used.
    """
    aa, bb = values.load(), fx.Vector(probabilities)
    acc = [fx.Vector.from_elements([output[n * 4 + i] for i in range(4)], fx.Float32)
           for n in range(8)]
    for first in range(0, 8, columns):
        _wait(lgkmcnt=16 - (first + columns) * 2)
        rocdl.sched_barrier(0)
        for step in range(4):
            b = fx.Vector.from_elements([bb[step * 4 + i] for i in range(4)], fx.BFloat16)
            for n in range(first, first + columns):
                a = fx.Vector.from_elements([aa[n * 16 + step * 4 + i] for i in range(4)], fx.BFloat16)
                acc[n] = fx.Vector(rocdl.mfma_f32_16x16x16bf16_1k(ir.VectorType.get([4], fx.Float32.ir_type),
                    [a.bitcast(fx.Int16).ir_value(), b.bitcast(fx.Int16).ir_value(), acc[n].ir_value(), 0, 0, 0]))
        rocdl.sched_barrier(0)
    return fx.Vector.from_elements([acc[n][i] for n in range(8) for i in range(4)], fx.Float32)


def _pv_step(probabilities, values, acc, first, columns):
    aa, bb = values.load(), fx.Vector(probabilities)
    for step in range(4):
        b = fx.Vector.from_elements([bb[step * 4 + i] for i in range(4)], fx.BFloat16)
        for n in range(first, first + columns):
            a = fx.Vector.from_elements([aa[n * 16 + step * 4 + i] for i in range(4)], fx.BFloat16)
            acc[n] = fx.Vector(rocdl.mfma_f32_16x16x16bf16_1k(ir.VectorType.get([4], fx.Float32.ir_type),
                [a.bitcast(fx.Int16).ir_value(), b.bitcast(fx.Int16).ir_value(), acc[n].ir_value(), 0, 0, 0]))


def _pv_s7(probabilities, values, output, current, candidate, maxima, scale, maximum):
    acc = [fx.Vector.from_elements([output[n * 4 + i] for i in range(4)], fx.Float32) for n in range(8)]
    centered = []
    for first in range(0, 8, 2):
        _wait(lgkmcnt=12 - first * 2)
        rocdl.sched_barrier(0)
        _pv_step(probabilities, values, acc, first, 2)
        if first == 0:
            candidate = _maximum(_maximum(candidate, maxima[0]), _maximum(maxima[1], maxima[2])) * scale
            predicate = candidate > maximum + 7.0
            ballot = fx.Int64(rocdl.ballot(fx.Int64.ir_type, predicate.ir_value()))
            new_max = predicate.select(candidate + 1.0, maximum)
        for i in range(first * 2, first * 2 + 4):
            centered.append(fx.Float32(llvm.inline_asm(fx.Float32.ir_type,
                [current[i].ir_value(), scale.ir_value(), new_max.ir_value()],
                "v_fma_f32 $0, $1, $2, -$3", "=v,v,v,v", has_side_effects=False)))
        _schedule(8, 2, 13 + first)
        rocdl.sched_barrier(0)
    return (fx.Vector.from_elements([acc[n][i] for n in range(8) for i in range(4)], fx.Float32),
            fx.Vector.from_elements(centered, fx.Float32), new_max, ballot)


@flyc.jit
def _body(Q, K, V, O, LSE, QS, KS, VS, table, storage, q0, q_len, kv_len, head, qb,
          H: fx.Constexpr[int], HK: fx.Constexpr[int], NP: fx.Constexpr[int], PAGE: fx.Constexpr[int],
          CAUSAL: fx.Constexpr[bool], PER_TOKEN: fx.Constexpr[bool], WITH_LSE: fx.Constexpr[bool],
          SCALE: fx.Constexpr[float], STAGGER: fx.Constexpr[bool],
          Q_DMA: fx.Constexpr[bool], K_DMA: fx.Constexpr[int], INTERLEAVE: fx.Constexpr[bool],
          V_DMA: fx.Constexpr[int], V_INTERLEAVE: fx.Constexpr[bool],
          EARLY_PAGES: fx.Constexpr[bool], LATE_S4_WAIT: fx.Constexpr[bool],
          LEAD_WAIT: fx.Constexpr[int], LATE_S0_WAIT: fx.Constexpr[bool],
          PV_COLUMNS: fx.Constexpr[int], M0_OFFSET: fx.Constexpr[bool],
          PARTIAL_K_WAIT: fx.Constexpr[bool], PV_S7_OVERLAP: fx.Constexpr[bool]):
    read_k, read_v, write = base._read_k, base._read_v, base._write
    load_k, load_v, qk, pv = base._load_k, base._load_v, base._qk, base._pv
    local_sum, local_max, cross = base._sum, base._max, base._cross
    center, exps, pack, mask = base._center, base._exps, base._pack, base._mask
    rescale, advance_max = _rescale, _advance_max
    pages_for_tile, dma_k, read_v_dma = base._pages, _dma_k, _read_v_dma
    read_k_write_dma = _read_k_write_dma
    v_resources, dma_v, read_k_vdma = _v_resources, _dma_v, _read_k_vdma
    pv_progressive = _pv_progressive
    m0_v_resources, read_k_m0_vdma = _m0_v_resources, _read_k_m0_vdma
    read_k_leading_first = _read_k_leading_first
    pv_s7 = _pv_s7
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
    if fx.const_expr(Q_DMA):
        q = _q_dma_fragment(gq, storage, tid, H)
    else:
        q = base._q_fragment(gq, (wave * 16 + (lane & 15)) * (H * D * 2), lane)
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
    if fx.const_expr(not K_DMA):
        kw = _pin_i32(shared + (tid >> 6) * K_PITCH + ((tid & 63) * 16 ^ (((tid >> 6) & 1) * 64)))
        klane = _pin_i32((tid >> 6) * (PAGE * 16) + (tid & (min(PAGE, BN) - 1)) * 16)
        kd = fx.Int32(0)
    else:
        kw, klane = fx.Int32(0), fx.Int32(0)
        kd = _pin_i32(_k_dma_offset(tid, PAGE))
    vr = _pin_i32(shared + K_BYTES + (lane >> 4) * 4096 + (lane & 15) * 16)
    if fx.const_expr(V_DMA):
        vd = _pin_i32(_v_dma_offset(tid))
        vw, vlane = fx.Int32(0), fx.Int32(0)
    else:
        vd = fx.Int32(0)
        vw = _pin_i32(shared + K_BYTES + tid * 16)
        vlane = _pin_i32(tid * 16)
    vbase = _pin_s64(fx.Int64(fx.ptrtoint(fx.get_iter(V))) + fx.Int64(hkv) * (PAGE * D * 2))
    cross_addresses = tuple(_pin_i32((lane ^ offset) * 4) for offset in (16, 32, 48))
    if fx.const_expr(K_DMA):
        for packet in fx.range_constexpr(16):
            dma_k(gk, storage, wave, kd, page0, page01, fx.Int32(0), HK, PAGE, packet)
    else:
        k0 = load_k(gk, klane, page0, fx.Int32(0), HK, PAGE, page01)
    if fx.const_expr(V_DMA):
        v0 = fx.Int32(0)
    else:
        v0 = load_v(vbase, vlane, page0, fx.Int32(0), HK, PAGE, page01)
    _wait(vmcnt=0)
    if fx.const_expr(not K_DMA):
        write(kw, k0, 8 * K_PITCH)
    if fx.const_expr(not V_DMA):
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
    if fx.const_expr(not K_DMA):
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
    if fx.const_expr(K_DMA):
        for packet in fx.range_constexpr(16):
            dma_k(gk, storage, wave, kd, page1, page11, _min(fx.Int32(1), last), HK, PAGE, packet)
        _wait(vmcnt=0)
    else:
        _wait(vmcnt=0)
        write(kw, k1, 8 * K_PITCH)
    _wait(lgkmcnt=0)
    _stage_end()
    _stage_end()

    @flyc.jit
    def phase(previous, maximum, row_sum, o0, o1, pending, previous_page, previous_page1,
              current_page, current_page1, next_page, next_page1, t):
        previous, maximum, row_sum = fx.Vector(previous), fx.Float32(maximum), fx.Float32(row_sum)
        o0, o1, t = fx.Vector(o0), fx.Vector(o1), fx.Int32(t)
        if fx.const_expr(EARLY_PAGES):
            # Issue page lookahead before the DS/DMA train, but do not read
            # or copy its asynchronous scalar result until the S0 tail.
            request, request1 = pages_for_tile(table, t + 2, last, kv_len, PAGE)
        if fx.const_expr(M0_OFFSET):
            resources = m0_v_resources(vbase, previous_page, previous_page1, t - 1, HK, PAGE)
            k = read_k_m0_vdma(kr, resources, wave, PAGE)
        elif fx.const_expr(V_DMA == 1 or V_DMA == 2):
            resources = v_resources(vbase, previous_page, previous_page1, t - 1, HK, PAGE)
            k = read_k_vdma(kr, 0, resources, storage, wave, vd, PAGE, 0,
                            16 if V_DMA == 1 else 8, V_INTERLEAVE)
        elif fx.const_expr(V_DMA == 3 or K_DMA == 4):
            k = read_k(kr, 0)
        else:
            k = read_k(kr, 0, vw, fx.Vector(pending))
        if fx.const_expr(V_DMA):
            vp = fx.Int32(0)
        else:
            vp = load_v(vbase, vlane, current_page, t, HK, PAGE, current_page1)
        if fx.const_expr(not EARLY_PAGES):
            request, request1 = pages_for_tile(table, t + 2, last, kv_len, PAGE)
        if fx.const_expr(LATE_S0_WAIT):
            # Neither K-low nor the future page is consumed by the partner
            # at this rendezvous; K overwrite remains protected by S2.
            _stage_end()
        future_page = _page_ready(request)
        if fx.const_expr(EARLY_PAGES and PAGE != 32):
            future_page1 = future_page
        else:
            future_page1 = _page_ready(request1)
        if fx.const_expr(LATE_S0_WAIT):
            rocdl.sched_barrier(0)
        else:
            _stage_end()
        lo = qk(q, k)
        previous = exps(previous)
        _schedule(32, 1, 1, True)
        _stage_end()
        if fx.const_expr(STAGGER and PARTIAL_K_WAIT):
            k = read_k_leading_first(kr)
        elif fx.const_expr(V_DMA == 2 or V_DMA == 3):
            resources = v_resources(vbase, previous_page, previous_page1, t - 1, HK, PAGE)
            k = read_k_vdma(kr, 1, resources, storage, wave, vd, PAGE,
                            8 if V_DMA == 2 else 0, 8 if V_DMA == 2 else 16, V_INTERLEAVE)
        elif fx.const_expr(K_DMA == 4):
            k = read_k_write_dma(kr, vw, fx.Vector(pending), gk, storage, wave, kd,
                                 next_page, next_page1, _min(t + 1, last), HK, PAGE)
        else:
            k = read_k(kr, 1)
        if fx.const_expr(V_DMA):
            # Retire this group's V DMA by S2. Leading S3 pairs with trailing
            # S2, so the leading S4 V consumer sees both groups' completed DMA.
            _wait(vmcnt=0)
        if fx.const_expr(STAGGER and PARTIAL_K_WAIT):
            _wait(lgkmcnt=8)
        elif fx.const_expr(STAGGER or not (LEAD_WAIT & 1)):
            _wait(lgkmcnt=0)
        _stage_end()
        if fx.const_expr((not STAGGER and (LEAD_WAIT & 1)) or (STAGGER and PARTIAL_K_WAIT)):
            # Leading S2 pairs with trailing S1, not an overwrite. The
            # trailing S2 pre-barrier wait protects leading S4 K DMA;
            # partial K waits retire even D32 planes there, then drain here.
            _wait(lgkmcnt=0)
            rocdl.sched_barrier(0)
        hi = qk(q, k)
        total = local_sum(previous)
        p = pack(previous)
        _schedule(32, 3, 2)
        _stage_end()
        # The leading group's S3 barrier pairs with the trailing group's
        # S2 K-high retirement (even D32 first for partial K waits).
        # Therefore S4 is safe for the leading group's K DMA.
        sums = cross(total, cross_addresses)
        if fx.const_expr(K_DMA == 1 or K_DMA == 2):
            v = read_v_dma(vr, 0, gk, storage, wave, kd, next_page, next_page1,
                          _min(t + 1, last), HK, PAGE, 0, 16 if K_DMA == 1 else 8, INTERLEAVE)
        elif fx.const_expr(K_DMA == 4):
            v = read_v_dma(vr, 0, gk, storage, wave, kd, next_page, next_page1,
                          _min(t + 1, last), HK, PAGE, 16, 8, INTERLEAVE)
        else:
            v = read_v(vr, 0)
            if fx.const_expr(not K_DMA):
                kp = load_k(gk, klane, next_page, _min(t + 1, last), HK, PAGE, next_page1)
        if fx.const_expr(not LATE_S4_WAIT):
            _wait(lgkmcnt=0)
        _stage_end()
        if fx.const_expr(LATE_S4_WAIT and not PV_COLUMNS):
            # This rendezvous does not release a V overwrite. S6 retirement
            # still protects the next S0 DMA; wait before any PV/sum consumer.
            _wait(lgkmcnt=0)
            rocdl.sched_barrier(0)
        if fx.const_expr(PV_COLUMNS):
            o0 = pv_progressive(p, v, o0, PV_COLUMNS)
        else:
            o0 = pv(p, v, o0)
        row_sum = row_sum + ((total + sums[0]) + (sums[1] + sums[2]))
        current = mask(_join(lo, hi), t, row, q_len, kv_len, CAUSAL)
        candidate = local_max(current)
        _schedule(32, 2, 3)
        _stage_end()
        maxima = cross(candidate, cross_addresses)
        _wait(vmcnt=0)
        if fx.const_expr(K_DMA == 2 or K_DMA == 3):
            v = read_v_dma(vr, 1, gk, storage, wave, kd, next_page, next_page1,
                          _min(t + 1, last), HK, PAGE, 8 if K_DMA == 2 else 0,
                          8 if K_DMA == 2 else 16, INTERLEAVE)
            _wait(vmcnt=0)
        elif fx.const_expr(K_DMA == 1 or K_DMA == 4):
            v = read_v(vr, 1)
        else:
            v = read_v(vr, 1, kw, kp)
        # Each group drains its DMA before the S6 rendezvous; the leading
        # next-S0 reader starts only after the trailing group's S6 has ended.
        if fx.const_expr(STAGGER or not (LEAD_WAIT & 2)):
            _wait(lgkmcnt=0)
        _stage_end()
        if fx.const_expr(not STAGGER and (LEAD_WAIT & 2) and not PV_COLUMNS):
            # Leading S6 pairs with trailing S5. The trailing S6 wait is
            # deliberately unchanged: it retires V reads before next S0 DMA.
            _wait(lgkmcnt=0)
            rocdl.sched_barrier(0)
        if fx.const_expr(not STAGGER and PV_S7_OVERLAP):
            o1, current, new_max, ballot = pv_s7(p, v, o1, current, candidate, maxima, scale, maximum)
        elif fx.const_expr(not STAGGER and PV_COLUMNS):
            o1 = pv_progressive(p, v, o1, PV_COLUMNS)
        else:
            o1 = pv(p, v, o1)
        if fx.const_expr(not (not STAGGER and PV_S7_OVERLAP)):
            candidate = _maximum(_maximum(candidate, maxima[0]), _maximum(maxima[1], maxima[2])) * scale
            ballot = fx.Int64(rocdl.ballot(fx.Int64.ir_type, (candidate > maximum + 7.0).ir_value()))
            new_max = (candidate > maximum + 7.0).select(candidate + 1.0, maximum)
            current = center(current, scale, new_max)
            _schedule(32, 3, 4)
        o0, o1, row_sum = rescale(o0, o1, row_sum, maximum, new_max, ballot)
        new_max = advance_max(maximum, new_max)
        _stage_end()
        return current, new_max, row_sum, o0, o1, vp, current_page, current_page1, next_page, next_page1, future_page, future_page1

    pending = v0
    previous_page, previous_page1 = page0, page01
    for t in range(fx.Int32(1), tiles, fx.Int32(1)):
        scores, maximum, row_sum, o0, o1, pending, previous_page, previous_page1, page1, page11, page2, page21 = phase(
            scores, maximum, row_sum, o0, o1, pending, previous_page, previous_page1, page1, page11, page2, page21, t)

    if fx.const_expr(V_DMA):
        resources = v_resources(vbase, previous_page, previous_page1, last, HK, PAGE)
        for packet in fx.range_constexpr(16):
            dma_v(resources, storage, wave, vd, PAGE, packet)
    else:
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
    v.store(base._v_tail(v.load(), kv_len - last * BN))
    _stage_end()
    o0 = pv(p, v, o0)
    _stage_end()
    v = read_v(vr, 1)
    _wait(lgkmcnt=0)
    rocdl.sched_barrier(0)
    v.store(base._v_tail(v.load(), kv_len - last * BN))
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
          CAUSAL: fx.Constexpr[bool], PER_TOKEN: fx.Constexpr[bool], WITH_LSE: fx.Constexpr[bool], SCALE: fx.Constexpr[float],
          Q_DMA: fx.Constexpr[bool], K_DMA: fx.Constexpr[int], INTERLEAVE: fx.Constexpr[bool],
          V_DMA: fx.Constexpr[int], V_INTERLEAVE: fx.Constexpr[bool],
          EARLY_PAGES: fx.Constexpr[bool], LATE_S4_WAIT: fx.Constexpr[bool],
          LEAD_WAIT: fx.Constexpr[int], LATE_S0_WAIT: fx.Constexpr[bool],
          PV_COLUMNS: fx.Constexpr[int], M0_OFFSET: fx.Constexpr[bool],
          PARTIAL_K_WAIT: fx.Constexpr[bool], PV_S7_OVERLAP: fx.Constexpr[bool]):
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
                 H, HK, NP, PAGE, CAUSAL, PER_TOKEN, WITH_LSE, SCALE, True, Q_DMA, K_DMA, INTERLEAVE,
                 V_DMA, V_INTERLEAVE, EARLY_PAGES, LATE_S4_WAIT,
                 LEAD_WAIT, LATE_S0_WAIT, PV_COLUMNS, M0_OFFSET, PARTIAL_K_WAIT, PV_S7_OVERLAP)
        else:
            body(Q, K, V, O, LSE, QS, KS, VS, table, storage, q0, q_len, kv_len, head, qb,
                 H, HK, NP, PAGE, CAUSAL, PER_TOKEN, WITH_LSE, SCALE, False, Q_DMA, K_DMA, INTERLEAVE,
                 V_DMA, V_INTERLEAVE, EARLY_PAGES, LATE_S4_WAIT,
                 LEAD_WAIT, LATE_S0_WAIT, PV_COLUMNS, M0_OFFSET, PARTIAL_K_WAIT, PV_S7_OVERLAP)


@flyc.kernel(known_block_size=[THREADS, 1, 1])
def _attention_256_dma_kernel_942(Q: fx.Tensor, K: fx.Tensor, V: fx.Tensor, O: fx.Tensor, LSE: fx.Tensor,
    CQ: fx.Tensor, KI: fx.Tensor, PAGES: fx.Tensor, LAST: fx.Tensor, QS: fx.Tensor, KS: fx.Tensor, VS: fx.Tensor,
    H: fx.Constexpr[int], HK: fx.Constexpr[int], NP: fx.Constexpr[int], B: fx.Constexpr[int],
    MAX_Q: fx.Constexpr[int], PAGE: fx.Constexpr[int], CAUSAL: fx.Constexpr[bool],
    PER_TOKEN: fx.Constexpr[bool], WITH_LSE: fx.Constexpr[bool], SCALE: fx.Constexpr[float],
    PERSISTENT: fx.Constexpr[bool], CUS: fx.Constexpr[int],
    Q_DMA: fx.Constexpr[bool], K_DMA: fx.Constexpr[int], INTERLEAVE: fx.Constexpr[bool],
    V_DMA: fx.Constexpr[int], V_INTERLEAVE: fx.Constexpr[bool],
    EARLY_PAGES: fx.Constexpr[bool], LATE_S4_WAIT: fx.Constexpr[bool],
    LEAD_WAIT: fx.Constexpr[int], LATE_S0_WAIT: fx.Constexpr[bool], PV_COLUMNS: fx.Constexpr[int],
    M0_OFFSET: fx.Constexpr[bool], TASK_ORDER: fx.Constexpr[int],
    PARTIAL_K_WAIT: fx.Constexpr[bool], PV_S7_OVERLAP: fx.Constexpr[bool]):
    work_body = _work
    storage = fx.SharedAllocator().allocate(fx.Array[fx.Int8, LDS_BYTES, 16]).peek().view(fx.make_layout(LDS_BYTES, 1))
    if fx.const_expr(PERSISTENT):
        work = fx.Int32(gpu.block_id("x"))
        while work < H * B * ((MAX_Q + BM - 1) // BM):
            if fx.const_expr(TASK_ORDER == 1):
                query_blocks = (MAX_Q + BM - 1) // BM
                head, batch, qb = work // (B * query_blocks), (work // query_blocks) % B, work % query_blocks
            elif fx.const_expr(TASK_ORDER > 1):
                total = H * B * ((MAX_Q + BM - 1) // BM)
                columns = total // TASK_ORDER
                mapped = (work < columns * TASK_ORDER).select((work % TASK_ORDER) * columns + work // TASK_ORDER, work)
                head, batch, qb = mapped % H, (mapped // H) % B, mapped // (H * B)
            else:
                head, batch, qb = work % H, (work // H) % B, work // (H * B)
            work_body(Q, K, V, O, LSE, CQ, KI, PAGES, LAST, QS, KS, VS, storage, head, batch, qb,
                      H, HK, NP, PAGE, CAUSAL, PER_TOKEN, WITH_LSE, SCALE, Q_DMA, K_DMA, INTERLEAVE,
                      V_DMA, V_INTERLEAVE, EARLY_PAGES, LATE_S4_WAIT,
                      LEAD_WAIT, LATE_S0_WAIT, PV_COLUMNS, M0_OFFSET, PARTIAL_K_WAIT, PV_S7_OVERLAP)
            work = work + CUS
    else:
        work_body(Q, K, V, O, LSE, CQ, KI, PAGES, LAST, QS, KS, VS, storage,
                  fx.Int32(gpu.block_id("x")), fx.Int32(gpu.block_id("y")), fx.Int32(gpu.block_id("z")),
                  H, HK, NP, PAGE, CAUSAL, PER_TOKEN, WITH_LSE, SCALE, Q_DMA, K_DMA, INTERLEAVE,
                  V_DMA, V_INTERLEAVE, EARLY_PAGES, LATE_S4_WAIT,
                  LEAD_WAIT, LATE_S0_WAIT, PV_COLUMNS, M0_OFFSET, PARTIAL_K_WAIT, PV_S7_OVERLAP)


@functools.cache
def _launcher(q_dma, k_dma, interleave, v_dma, v_interleave, early_pages, late_s4_wait,
              lead_wait=0, late_s0_wait=False, pv_columns=0, m0_offset=False,
              task_order=0, partial_k_wait=False, pv_s7_overlap=False):
    @flyc.jit
    def launch(Q: fx.Tensor, K: fx.Tensor, V: fx.Tensor, O: fx.Tensor, LSE: fx.Tensor,
        CQ: fx.Tensor, KI: fx.Tensor, PAGES: fx.Tensor, LAST: fx.Tensor, QS: fx.Tensor, KS: fx.Tensor, VS: fx.Tensor,
        H: fx.Constexpr[int], HK: fx.Constexpr[int], NP: fx.Constexpr[int], B: fx.Constexpr[int],
        MAX_Q: fx.Constexpr[int], DQ: fx.Constexpr[int], DV: fx.Constexpr[int], PAGE: fx.Constexpr[int],
        CAUSAL: fx.Constexpr[bool], PER_TOKEN: fx.Constexpr[bool], WITH_LSE: fx.Constexpr[bool],
        SCALE: fx.Constexpr[float], PERSISTENT: fx.Constexpr[bool], CUS: fx.Constexpr[int], stream: fx.Stream):
        assert DQ == DV == D
        grid = (min(CUS, H * B * ((MAX_Q + BM - 1) // BM)), 1, 1) if PERSISTENT else (H, B, (MAX_Q + BM - 1) // BM)
        _attention_256_dma_kernel_942(Q, K, V, O, LSE, CQ, KI, PAGES, LAST, QS, KS, VS,
            H, HK, NP, B, MAX_Q, PAGE, CAUSAL, PER_TOKEN, WITH_LSE, SCALE, PERSISTENT, CUS,
            q_dma, k_dma, interleave, v_dma, v_interleave, early_pages, late_s4_wait,
            lead_wait, late_s0_wait, pv_columns, m0_offset, task_order, partial_k_wait, pv_s7_overlap,
            value_attrs={"rocdl.waves_per_eu": 2, "passthrough": [["target-features", "-packed-fp32-ops"]]},
        ).launch(grid=grid, block=(THREADS, 1, 1), stream=stream)
    return launch


@functools.cache
def PagedAttention(num_qo_heads, num_kv_heads, head_dim_qk, head_dim_v, page_size,
                   is_causal, quant_query_mode="per-token", key_layout="vectorized",
                   window_left=-1, has_sink=False, *, memory_mode="lds", persistent=None,
                   dma_query=True, dma_key="early", interleave=True,
                   dma_value="off", value_interleave=True,
                   early_pages=False, late_s4_wait=False,
                   lead_wait=0, late_s0_wait=False, pv_columns=0, m0_offset=False,
                   task_order=0, partial_k_wait=False, pv_s7_overlap=False):
    validated = _validate_factory(num_qo_heads, num_kv_heads, head_dim_qk, head_dim_v, page_size,
        is_causal, quant_query_mode, key_layout, window_left, has_sink,
        memory_mode=memory_mode, persistent=persistent)
    if head_dim_qk != D or head_dim_v != D:
        raise NotImplementedError("direct-to-LDS experiment requires DQ=DV=256")
    if not isinstance(dma_query, bool) or not isinstance(interleave, bool):
        raise ValueError("dma_query and interleave must be bool")
    modes = {"off": 0, "early": 1, "split": 2, "late": 3, "write_mix": 4}
    if dma_key not in modes:
        raise ValueError("dma_key must be off, early, split, late or write_mix")
    v_modes = {"off": 0, "early": 1, "split": 2, "late": 3}
    if dma_value not in v_modes or not isinstance(value_interleave, bool):
        raise ValueError("dma_value must be off, early, split or late; value_interleave must be bool")
    if dma_value != "off" and dma_key == "write_mix":
        raise NotImplementedError("V DMA and the V-register write_mix diagnostic are separate experiments")
    if not isinstance(early_pages, bool) or not isinstance(late_s4_wait, bool):
        raise ValueError("early_pages and late_s4_wait must be bool")
    if type(lead_wait) is not int or lead_wait not in range(4) or not isinstance(late_s0_wait, bool):
        raise ValueError("lead_wait must be an integer in 0..3 and late_s0_wait must be bool")
    if (lead_wait or late_s0_wait) and not (early_pages and late_s4_wait):
        raise NotImplementedError("wait trials require the v73 page/S4 combination")
    if type(pv_columns) is not int or pv_columns not in (0, 1, 2, 4, 8):
        raise ValueError("pv_columns must be 0, 1, 2, 4 or 8")
    if pv_columns and not (early_pages and late_s4_wait and (lead_wait & 2)):
        raise NotImplementedError("progressive PV requires v73 and leading S6 late wait")
    if not isinstance(m0_offset, bool):
        raise ValueError("m0_offset must be bool")
    if m0_offset and not (early_pages and late_s4_wait):
        raise NotImplementedError("V m0 offset requires the v73 page/S4 combination")
    if type(task_order) is not int or task_order not in (0, 1, 2, 4, 8, 16):
        raise ValueError("task_order must be 0, 1, 2, 4, 8 or 16")
    if task_order and not validated.persistent:
        raise NotImplementedError("task_order requires persistent scheduling")
    if not isinstance(partial_k_wait, bool):
        raise ValueError("partial_k_wait must be bool")
    if partial_k_wait and not (early_pages and late_s4_wait and pv_columns and lead_wait == 3):
        raise NotImplementedError("partial K wait requires v73/PV progression and late leading waits")
    if not isinstance(pv_s7_overlap, bool):
        raise ValueError("pv_s7_overlap must be bool")
    if pv_s7_overlap and not (early_pages and late_s4_wait and late_s0_wait and pv_columns and lead_wait == 3):
        raise NotImplementedError("S7 PV overlap requires the selected progressive-PV combination")
    if (early_pages or late_s4_wait) and (dma_query or dma_key != "early" or dma_value != "early"):
        raise NotImplementedError("memory-stage diagnostics require Q DMA off and K/V DMA early")
    kernel = _PagedAttention(num_qo_heads, num_kv_heads, D, D, page_size,
                             is_causal, quant_query_mode, validated.persistent)
    kernel._launch = _launcher(dma_query, modes[dma_key], interleave, v_modes[dma_value], value_interleave,
                               early_pages, late_s4_wait)
    if lead_wait or late_s0_wait or pv_columns or m0_offset or task_order or partial_k_wait or pv_s7_overlap:
        kernel._launch = _launcher(dma_query, modes[dma_key], interleave, v_modes[dma_value], value_interleave,
                                   early_pages, late_s4_wait, lead_wait, late_s0_wait, pv_columns,
                                   m0_offset, task_order, partial_k_wait, pv_s7_overlap)
    kernel.bf16_backend = f"native-m16-dma4-q{int(dma_query)}-k{dma_key}-v{dma_value}-interleave{int(interleave)}-{int(value_interleave)}"
    if early_pages or late_s4_wait:
        kernel.bf16_backend += f"-pages{int(early_pages)}-lateS4{int(late_s4_wait)}"
    if lead_wait or late_s0_wait:
        kernel.bf16_backend += f"-leadwait{lead_wait}-lateS0{int(late_s0_wait)}"
    if pv_columns:
        kernel.bf16_backend += f"-pvcolumns{pv_columns}"
    if m0_offset:
        kernel.bf16_backend += f"-m0offset{int(m0_offset)}"
    if task_order:
        kernel.bf16_backend += f"-taskorder{task_order}"
    if partial_k_wait:
        kernel.bf16_backend += "-partialkwait"
    if pv_s7_overlap:
        kernel.bf16_backend += "-pvs7overlap"
    return kernel