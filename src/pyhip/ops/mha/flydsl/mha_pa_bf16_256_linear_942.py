"""Native gfx942 BF16 D256 attention over linear Q/K/V/O, without KV conversion.

BM128/BN64, eight staggered waves, 32-KiB K + 32-KiB row-major V in LDS.
K is XOR-swizzled at the DMA source. V is transposed after LDS reads with
the same 2x2 BF16 byte-permutation selectors used by CK transpose_vectors.
The PV N coordinate is permuted so each lane transposes contiguous 8x8
pieces; the output shuffle restores ordinary [T,H,D] storage. K/V reach LDS
through direct DMA (``buffer_load ... lds``); each phase interleaves the DMA
packets of one tile with the LDS reads of another. ``_body`` holds the
pipeline table. QSA's union kernel reuses the LDS layout and these helpers.
"""

import math

import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu, rocdl
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from pyhip.codegen.flydsl.helpers import rocdl_aux

from ._common import (
    _MAX_FLOOR, _advance_max, _buffer, _cross, _exps, _join, _mask, _max, _maxnum, _min,
    _pack_bf16, _pin, _pin_i32, _qk, _load_address, _read_q_g2r, _rescale, _scale, _schedule, _schedule_exps,
    _stage_end, _sum, _uniform, _wait,
)

BM, BN, D, THREADS = 128, 64, 256, 512
K_BYTES, LDS_BYTES = 32768, 65536


def _k_phase(token):
    return (token & 3) | ((token >> 3) & 1) * 4


def _k_lds_address(token, dimension):
    return token * 512 + ((dimension // 2) ^ (_k_phase(token) * 4)) * 4


def _v_lds_address(token, dimension):
    return K_BYTES + token * 512 + dimension * 2


def _copy_coordinates(wave, packet, page=0):
    """(token, first dimension) that a wave's DMA packet writes: 16 packets cover a 64x256 tile."""
    token = (wave & 3) * 16 + packet if page == 4 else (wave & 3) + packet * 4
    return token, (wave >> 2) * 128


def _read_kv_rows_g2r(table, tile, kv_start, kv_len, wave, page, paged, hk):
    """DMA source rows of a KV tile; paged tiles read their physical pages from the table (VMEM)."""
    row, channel = _copy_coordinates(wave, 0, page if paged else 0)
    if paged:
        # One vectorized page-table request per wave. The 16 DMA SOFFSETs
        # read the wave-uniform selected lane; no divergent scalar waterfall.
        lane = fx.Int32(gpu.thread_id("x")) & 63
        logical = fx.Uint32(_min(tile * BN + (lane & (3 if page == 4 else 15)) * 4 + row, kv_len - 1))
        index = logical >> (2 if page == 4 else 0)
        physical = table[fx.Int32(index)]
        return (physical * page + (row & (page - 1))) * (hk * D * 2) + channel * 2
    return kv_start + tile * BN + row


def _dma_offsets(rows, page):
    """Per-packet scalar offsets (SOFFSET) of a tile's DMA: one value without a page table."""
    if page:
        return fx.Vector.from_elements([fx.Int32(rocdl.readlane(fx.Int32.ir_type, rows.ir_value(), fx.Int32(packet).ir_value()))
                                        for packet in range(4 if page == 4 else 16)], fx.Int32)
    return fx.Vector.from_elements([rows], fx.Int32)


def _read_packet_g2s(resource, storage, wave, lane, rows, tile, kv_len, hk, packet, is_v, extent, page, tail=False,
                     read=None):
    """DMA packet `packet` of a K or V tile (one dword per lane); `read` = (LDS address, immediate)
    issues one ds_read_b128 in the same leaf and returns its (asynchronous) result."""
    token, channel = _copy_coordinates(wave, packet, page)
    if is_v:
        voffset = lane * 4
    else:
        voffset = (lane ^ (_k_phase(token) * 4)) * 4
    # Materialize only the current M0/SOFFSET derivation; hoisting all 16
    # constants across the persistent loop spilled SGPRs in the first trial.
    base_row, base_channel = _copy_coordinates(wave, 0, page)
    base_destination = fx.Int32(base_row * 512 + base_channel * 2)
    immediate = (K_BYTES if is_v else 0) + packet * (512 if page == 4 else 2048)
    # Keep both scalar address producers in the DS/DMA leaf. Independent asm
    # results otherwise add hazard nops between SOFFSET, M0 and the LDS read.
    if not page and read is not None and not tail:
        read_address, read_immediate = read
        origin = fx.Int32(rows[0] * (hk * D * 2) + channel * 2)
        result_type = ir.Type.parse("!llvm.struct<(vector<4xi32>, i32)>")
        result = llvm.inline_asm(result_type,
            [fx.Int32(read_address).ir_value(), resource, fx.Int32(voffset).ir_value(),
             base_destination.ir_value(), origin.ir_value()],
            f"s_add_u32 $1, $6, {packet * 4 * hk * D * 2}\n"
            f"s_add_u32 m0, $5, {immediate}\n"
            f"ds_read_b128 $0, $2 offset:{read_immediate}\n"
            "buffer_load_dword $4, $3, $1 offen lds",
            "=&v,=&s,v,s,v,s,s,~{m0},~{scc},~{memory}", has_side_effects=True)
        return fx.Vector(llvm.extractvalue(ir.VectorType.get([4], fx.Int32.ir_type), result, [0]))
    if page:
        scalar = rows[packet // 4] if page == 4 else rows[packet]
        if page == 4:
            voffset = voffset + (packet & 3) * (hk * D * 2)
    else:
        origin = rows[0] * (hk * D * 2) + channel * 2
        scalar = fx.Int32(llvm.inline_asm(fx.Int32.ir_type, [fx.Int32(origin).ir_value()],
            f"s_add_u32 $0, $1, {packet * 4 * hk * D * 2}", "=s,s,~{scc}", has_side_effects=True))
    # Steady-state V(t-1) is a complete tile. Invalid K rows are independently
    # masked before softmax, so their bounded loads need no per-packet predicate.
    # Only the drain V tile needs zero padding: zero P times a NaN V is NaN.
    offset = fx.Int32(voffset)
    if tail:
        valid = tile * BN + token < kv_len
        offset = valid.select(offset, fx.Int32(extent))
    if read is not None:
        read_address, read_immediate = read
        # Explicit M0 leaf region, like the validated paged D256 DMA path:
        # the independent DS read supplies the required M0->VMEM distance.
        # All asynchronous results are consumed only after explicit waits.
        return fx.Vector(llvm.inline_asm(ir.VectorType.get([4], fx.Int32.ir_type),
            [fx.Int32(read_address).ir_value(), resource, offset.ir_value(), base_destination.ir_value(), scalar.ir_value()],
            f"s_add_u32 m0, $4, {immediate}\n"
            f"ds_read_b128 $0, $1 offset:{read_immediate}\n"
            "buffer_load_dword $3, $2, $5 offen lds",
            "=&v,v,s,v,s,s,~{m0},~{scc},~{memory}", has_side_effects=True))
    destination = fx.Int32(llvm.inline_asm(fx.Int32.ir_type, [base_destination.ir_value()],
        f"s_add_u32 $0, $1, {immediate}", "=s,s,~{scc}", has_side_effects=True))
    rocdl.raw_ptr_buffer_load_lds(resource, fx.to_llvm_ptr(fx.get_iter(storage) + destination),
        fx.Int32(4).ir_value(), offset.ir_value(), fx.Int32(scalar).ir_value(),
        fx.Int32(0).ir_value(), aux=rocdl_aux(0))


def _k_reads(addresses, half, lower_first=False):
    """(slot, address, immediate) of the 16 LDS reads of K keys 32 * half .. + 31, in issue order."""
    order = ([(n, k + high * 4) for high in range(2) for n in range(2) for k in range(4)] if lower_first
             else [(n, k) for n in range(2) for k in range(8)])
    return [(n * 8 + k, addresses[n * 2 + k % 2], half * 16384 + (k // 2) * 128) for n, k in order]


def _v_reads(address, half, blocks=2):
    """(slot, address, immediate) of the LDS reads of V dims 128 * half .. + 127, in issue order."""
    return [(block * 8 + row, address, half * 256 + block * 16384 + row * 512)
            for block in range(blocks) for row in range(8)]


def _words(parts):
    return fx.Vector.from_elements([part[i] for part in parts for i in range(4)], fx.Int32)


def _k_operand(parts):
    """QK A operand from the 16 K reads of one 32-key half."""
    frag = fx.make_rmem_tensor(fx.make_layout((4, 16, 2), (1, 4, 64)), fx.BFloat16)
    frag.store(_words(parts).bitcast(fx.BFloat16))
    return fx.make_view(fx.get_iter(frag), fx.make_layout((4, 2, 16), (1, 64, 4)))


def _read_k_s2r(addresses, half, lower_first=False):
    parts = [None] * 16
    for slot, address, immediate in _k_reads(addresses, half, lower_first):
        parts[slot] = _load_address(address, immediate)
    return _k_operand(parts)


def _read_v_s2r(address, half, blocks=2):
    return _words([_load_address(at, immediate) for _, at, immediate in _v_reads(address, half, blocks)])


def _v_operands(values, step, progressive):
    raw = fx.Vector(values)
    if progressive:
        _wait(lgkmcnt=12 - step * 4)
        rocdl.sched_barrier(0)
    row = (step // 2) * 8 + (step % 2) * 4
    words = []
    for n in range(8):
        selector = 0x03020706 if n % 2 else 0x01000504
        words.extend(fx.Int32(rocdl.perm_b32(raw[(row + r) * 4 + n // 2].ir_value(),
            raw[(row + r + 1) * 4 + n // 2].ir_value(), fx.Int32(selector).ir_value())) for r in (0, 2))
    return fx.Vector.from_elements(words, fx.Int32)


def _pack_probabilities(values):
    # The RNE bits of _common._pack_bf16; exposing the permutes helps only the static specialization.
    bits = fx.Vector(values).bitcast(fx.Uint32)
    rounded = bits + fx.Uint32(0x7FFF) + ((bits >> 16) & fx.Uint32(1))
    words = [fx.Int32(rocdl.perm_b32(rounded[i + 1].ir_value(), rounded[i].ir_value(),
                                   fx.Int32(0x07060302).ir_value())) for i in range(0, 16, 2)]
    return fx.Vector.from_elements(words, fx.Int32).bitcast(fx.BFloat16)


def _pack(values):
    return _pack_bf16(values).bitcast(fx.BFloat16)


def _pv(probabilities, values, output, progressive=False, prepared=None, center_args=None, summary_args=None, steps=4):
    """O += P V over one V half (transposed in registers with v_perm), softmax VALU folded in.

    summary_args = (S, lane sum, other sums, l) also returns l + row sum and the lane max of S;
    center_args = (S, lane max, other maxima, scale, m) returns S * scale, m' and the ballot.
    """
    p = fx.Vector(probabilities)
    acc = [fx.Vector.from_elements([output[n * 4 + i] for i in range(4)], fx.Float32) for n in range(8)]
    prepared_steps = 0 if prepared is None else prepared.numel // 16
    operands = _v_operands(values, 0, progressive) if prepared is None else fx.Vector.from_elements([prepared[i] for i in range(16)], fx.Int32)
    scaled = []
    rocdl.sched_barrier(0)
    for step in range(steps):
        if step + 1 < prepared_steps:
            following = fx.Vector.from_elements([prepared[(step + 1) * 16 + i] for i in range(16)], fx.Int32)
        else:
            following = _v_operands(values, step + 1, progressive) if step + 1 < steps else operands
        b = fx.Vector.from_elements([p[step * 4 + i] for i in range(4)], fx.BFloat16)
        for n in range(8):
            a = fx.Vector.from_elements([operands[n * 2 + i] for i in range(2)], fx.Int32).bitcast(fx.Int16)
            acc[n] = fx.Vector(rocdl.mfma_f32_16x16x16bf16_1k(ir.VectorType.get([4], fx.Float32.ir_type),
                [a.ir_value(), b.bitcast(fx.Int16).ir_value(), acc[n].ir_value(), 0, 0, 0]))
        if center_args is not None:
            current, candidate, maxima, scale, maximum = center_args
            if step == 0:
                candidate = _maxnum(_maxnum(candidate, maxima[0]), _maxnum(maxima[1], maxima[2])) * scale
                predicate = candidate > maximum + 7.0
                ballot = fx.Int64(rocdl.ballot(fx.Int64.ir_type, predicate.ir_value()))
                new_max = predicate.select(candidate + 1.0, maximum)
            for i in range(step * 4, step * 4 + 4):
                scaled.append(fx.Float32(current[i]) * scale)
        if summary_args is not None and step == 3:
            current, total, sums, row_sum = summary_args
            row_sum = row_sum + ((total + sums[0]) + (sums[1] + sums[2]))
            candidate = _max(current)
        if step + 1 < steps and step + 1 >= prepared_steps:
            _schedule(8, 3 if center_args is not None else 2, 20 + step)
        elif summary_args is not None:
            _schedule(8, 3, 23)
        rocdl.sched_barrier(0)
        operands = following
    result = fx.Vector.from_elements([acc[n][i] for n in range(8) for i in range(4)], fx.Float32)
    if center_args is not None:
        # Pinned: unpinned, LLVM sinks the scaling into the next memory stage.
        return result, _pin(fx.Vector.from_elements(scaled, fx.Float32)), new_max, ballot
    if summary_args is not None:
        return result, row_sum, candidate
    return result


def _shuffle_o_r2s2r(o0, o1, inv, shared, tid):
    """O * inv as BF16 through LDS (b128 writes, XOR-swizzled rows); returns this thread's
    eight 16-byte pieces of contiguous O rows."""
    row = (tid >> 6) * 16 + (tid & 15)
    for half in range(2):
        output = o1 if half else o0
        for i in range(4):
            values = fx.Vector.from_elements([output[n * 4 + i] * inv for n in range(8)], fx.Float32)
            words = _pack_bf16(values)
            column = (((tid >> 4) & 3) * 4 + i) * 8 + half * 128
            address = fx.Int32(shared + ((row * 256 + column) ^ ((row & 7) * 8)) * 2)
            llvm.inline_asm(ir.Type.parse("!llvm.void"), [address.ir_value(), words.ir_value()],
                            "ds_write_b128 $0, $1", "v,v,~{memory}", has_side_effects=True)
    _wait(lgkmcnt=0)
    _stage_end()
    parts = []
    for part in range(8):
        element = tid * 8 + part * THREADS * 8
        read_row = element // 256
        parts.append(_load_address(shared + (element ^ ((read_row & 7) * 8)) * 2))
    _wait(lgkmcnt=0)
    rocdl.sched_barrier(0)
    return parts


def _store_o_r2g(parts, buffer, tid, heads):
    atom = fx.make_copy_atom(rocdl.BufferCopy128b(), fx.BFloat16)
    for part in range(8):
        element = tid * 8 + part * THREADS * 8
        read_row, column = element // 256, element % 256
        fragment = fx.make_rmem_tensor(8, fx.BFloat16)
        fragment.store(parts[part].bitcast(fx.BFloat16))
        offset = read_row * heads * D + column
        fx.copy(atom, fragment, fx.make_view(fx.get_iter(buffer) + offset, fx.make_layout(8, 1)))


@flyc.jit
def _body(Q, K, V, O, LSE, CQ, CK, TABLE, storage, head, batch, qb,
          H: fx.Constexpr[int], HK: fx.Constexpr[int], NK: fx.Constexpr[int], MAX_PAGES: fx.Constexpr[int],
          PAGE: fx.Constexpr[int], PAGED: fx.Constexpr[bool], CAUSAL: fx.Constexpr[bool],
          WITH_LSE: fx.Constexpr[bool], SCALE: fx.Constexpr[float], STAGGER: fx.Constexpr[bool],
          STATIC_Q: fx.Constexpr[int], STATIC_K: fx.Constexpr[int]):
    """One task: query rows 128 * qb .. + 127 of one head over its KV tiles t of BN = 64 keys.

    LDS holds one K tile and one V tile (32 KiB each). DMA rows (page-table lookups when
    paged) run two tiles ahead of QK and K one tile ahead; V(t-1) arrives during phase t.
    Phase t runs QK and the row max of tile t and, one tile behind, the exponentials, row
    sum and PV of tile t - 1. A phase has four steps, each a memory stage then a compute
    stage; waves 4-7 (STAGGER) run one stage behind waves 0-3, so one group's memory stage
    overlaps the other group's compute stage.

    step  memory stage                                compute stage
    0     g2r rows(t+2); g2s V(t-1) with s2r K(t)     QK: S(t) keys 0-31
          keys 0-31 between its packets               softmax: P(t-1) = exp2(S'(t-1) - m)
    1     s2r K(t) keys 32-63; V(t-1) lands           QK: S(t) keys 32-63
                                                      softmax: row sum of P(t-1), P to BF16
    2     exchange row sums; g2s K(t+1) with s2r      mask S(t); PV: O dims 0-127 += P V
          V(t-1) dims 0-127 between packets           softmax: l += row sum, lane max of S(t)
    3     exchange row maxima; K(t+1) lands;          PV: O dims 128-255 += P(t-1) V(t-1)
          s2r V(t-1) dims 128-255                     softmax: m, S'(t); rescale O and l

    A static B1 noncausal shape whose last tile has at most 32 keys runs that tile as a
    NARROW phase (no keys 32-63) and drains only the first 32-token V plane.

    Softmax in log2 units: S' = S * scale, P = exp2(S' - m), l = sum(P) and O = sum(P V);
    the output is O / l. S' is rounded before m is subtracted, like m itself. m starts at
    the first row max + 1 and moves to a later row max + 1 only when that exceeds m + 7,
    rescaling O and l.
    """
    # FlyDSL keys its disk cache on helpers named in this function's own code, not in closures.
    _helpers = (_read_kv_rows_g2r, _dma_offsets, _read_packet_g2s, _k_reads, _v_reads, _k_operand, _words,
                _read_k_s2r, _read_v_s2r, _load_address, _v_operands, _pv, _qk, _sum, _max, _cross, _exps, _pack,
                _pack_probabilities, _mask, _scale, _rescale, _advance_max, _join, _maxnum, _pin,
                _schedule, _schedule_exps, _stage_end, _wait)
    pack = _pack_probabilities if STATIC_Q > 0 else _pack
    if fx.const_expr(STATIC_Q > 0):
        q0, k0 = fx.Int32(0), fx.Int32(0)
        q_len, kv_len = fx.Int32(STATIC_Q), fx.Int32(STATIC_K)
    else:
        q0, k0 = _uniform(CQ[batch]), _uniform(CK[batch])
        q_len, kv_len = _uniform(CQ[batch + 1]) - q0, _uniform(CK[batch + 1]) - k0
    tid = fx.Int32(gpu.thread_id("x"))
    wave, lane = _uniform(tid >> 6), tid & 63
    q_start = qb * BM
    row = q_start + wave * 16 + (lane & 15)
    valid = _min(fx.Int32(BM), q_len - q_start)
    hkv = head // (H // HK)
    page = PAGE if PAGED else 0
    table = fx.make_view(fx.get_iter(TABLE) + batch * MAX_PAGES, fx.make_layout(MAX_PAGES, 1))
    gq = _buffer(fx.make_view(fx.get_iter(Q) + (fx.Int64(q0) + q_start) * (H * D) + fx.Int64(head) * D,
                              fx.make_layout(BM * H * D, 1)), valid * H * D * 2)
    extent = (NK * HK - hkv) * D * 2
    gk = _buffer(fx.make_view(fx.get_iter(K) + hkv * D, fx.make_layout((NK * HK - hkv) * D, 1)), extent)
    gv = _buffer(fx.make_view(fx.get_iter(V) + hkv * D, fx.make_layout((NK * HK - hkv) * D, 1)), extent)
    # Q stays in registers for every KV tile.
    q = _read_q_g2r(gq, (wave * 16 + (lane & 15)) * H * D * 2, lane)
    scale = fx.Float32(SCALE * math.log2(math.e))
    tiles = (kv_len + BN - 1) // BN
    if fx.const_expr(CAUSAL):
        end = (q_start + valid + kv_len - q_len + BN - 1) // BN
        tiles = _min(tiles, (end > 0).select(end, fx.Int32(1)))
    last = tiles - 1
    narrow_tail = STATIC_Q > 0 and 0 < STATIC_K % BN <= 32 and STATIC_K > BN
    # Loop-invariant LDS read addresses (s2r) and lane-exchange addresses.
    shared = fx.Int32(fx.ptrtoint(fx.get_iter(storage)))
    key_row = (lane & 3) + ((lane & 12) << 1)
    k_s2r = tuple(_pin_i32(shared + _k_lds_address(key_row + n * 4, (lane >> 4) * 8 + parity * 32))
                  for n in range(2) for parity in range(2))
    v_s2r = _pin_i32(shared + _v_lds_address((lane >> 4) * 8, (lane & 15) * 8))
    cross_addresses = tuple(_pin_i32((lane ^ offset) * 4) for offset in (16, 32, 48))

    # g = global, s = LDS, r = registers; reads only issue, the stages place the waits.
    def read_rows_g2r(tile):
        return _read_kv_rows_g2r(table, tile, k0, kv_len, wave, PAGE, PAGED, HK)

    def read_tile_g2s(resource, soffsets, is_v, s2r=None, tail_tile=None, packets=16):
        """DMA packets 0 .. packets-1 of a K/V tile; read i of the s2r plan is issued just before packet i."""
        parts = [None] * packets
        for packet in fx.range_constexpr(packets):
            read, slot = None, packet
            if fx.const_expr(s2r is not None):
                slot, address, immediate = s2r[packet]
                read = (address, immediate)
            parts[slot] = _read_packet_g2s(resource, storage, wave, lane, soffsets, tail_tile, kv_len, HK, packet,
                                           is_v, extent, page, tail_tile is not None, read)
        return parts

    def read_k_g2s(soffsets, s2r=None):
        return read_tile_g2s(gk, soffsets, False, s2r)

    def read_v_g2s(soffsets, s2r=None, tail_tile=None, packets=16):
        return read_tile_g2s(gv, soffsets, True, s2r, tail_tile, packets)

    def prologue():
        """Rows of tiles 0-2 (g2r); K(0) g2s; QK S(0), m and S'(0); K(1) g2s."""
        rows_0 = read_rows_g2r(fx.Int32(0))
        rows_1 = read_rows_g2r(_min(fx.Int32(1), last))
        rows_2 = read_rows_g2r(_min(fx.Int32(2), last))
        # Memory: g2s K(0).
        soffsets_0 = _dma_offsets(rows_0, page)
        read_k_g2s(soffsets_0)
        _wait(vmcnt=0, lgkmcnt=0)
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
        o0, o1 = _pin(fx.Vector.filled(32, 0.0, fx.Float32)), _pin(fx.Vector.filled(32, 0.0, fx.Float32))
        _schedule(32, 3, 5)
        _stage_end()
        # Memory: s2r K(0) keys 32-63.
        k_hi = _read_k_s2r(k_s2r, 1)
        _wait(lgkmcnt=0)
        _stage_end()
        # Compute: QK S(0) keys 32-63; softmax m and S'(0).
        s_hi = _qk(q, k_hi)
        s_0 = _scale(_mask(_join(s_lo, s_hi), fx.Int32(0), row, q_len, kv_len, CAUSAL), scale)
        row_max = _max(s_0)
        row_max = _maxnum(row_max, row_max.shuffle_xor(16, 64))
        row_max = _maxnum(row_max, row_max.shuffle_xor(32, 64))
        row_max = _maxnum(row_max, fx.Float32(_MAX_FLOOR)) + 1.0
        row_sum = fx.Float32(0.0)
        _stage_end()
        # Memory: g2s K(1); no compute.
        soffsets_1 = _dma_offsets(rows_1, page)
        read_k_g2s(soffsets_1)
        _wait(vmcnt=0, lgkmcnt=0)
        _stage_end()
        _stage_end()
        return s_0, row_max, row_sum, o0, o1, soffsets_0, soffsets_1, rows_2

    @flyc.jit
    def phase(s_prev, row_max, row_sum, o0, o1, soffsets_prev, soffsets_cur, rows_next, t,
              NARROW: fx.Constexpr[bool]):
        """Phase t (table in _body). soffsets_prev/soffsets_cur are the DMA SOFFSETs of tiles t-1
        and t; rows_next the rows of tile t+1. K(t) is in LDS; V(t-1) arrives in step 0."""
        s_prev, row_max, row_sum = fx.Vector(s_prev), fx.Float32(row_max), fx.Float32(row_sum)
        o0, o1, t = fx.Vector(o0), fx.Vector(o1), fx.Int32(t)
        soffsets_prev = fx.Vector(soffsets_prev)
        soffsets_cur, rows_next = fx.Vector(soffsets_cur), fx.Int32(rows_next)
        # Step 0 memory: g2r rows(t+2); g2s V(t-1) with s2r K(t) keys 0-31 between its packets.
        rows_next2 = read_rows_g2r(_min(t + 2, last))
        k_lo = _k_operand(read_v_g2s(soffsets_prev, s2r=_k_reads(k_s2r, 0)))
        _stage_end()
        # Step 0 compute: QK S(t) keys 0-31; P(t-1) = exp2(S'(t-1) - m).
        _wait(lgkmcnt=0)
        rocdl.sched_barrier(0)
        s_lo = _qk(q, k_lo)
        p_prev = _exps(s_prev, row_max)
        _schedule_exps(16, 1, 1, 2)
        _stage_end()
        # Step 1 memory: s2r K(t) keys 32-63 (none for a NARROW tile); V(t-1) lands.
        if fx.const_expr(not NARROW):
            k_hi = _read_k_s2r(k_s2r, 1, STAGGER)
        _wait(vmcnt=0)
        if fx.const_expr(STAGGER):
            # The leading DMA owns D[0:128]; retire those eight reads before it
            # can overwrite K. The remaining reads only touch D[128:256].
            _wait(lgkmcnt=8)
        _stage_end()
        # Step 1 compute: QK S(t) keys 32-63; row sum of P(t-1), P to BF16; SOFFSETs of K(t+1).
        _wait(lgkmcnt=0)
        rocdl.sched_barrier(0)
        # A NARROW tile has no live keys 32-63; the mask turns these placeholders into -inf.
        s_hi = fx.Vector.filled(8, 0.0, fx.Float32) if NARROW else _qk(q, k_hi)
        lane_sum, p_bf16 = _sum(p_prev), pack(p_prev)
        soffsets_next = _dma_offsets(rows_next, page)
        _schedule(32, 3, 2)
        _stage_end()
        # Step 2 memory: exchange row sums; g2s K(t+1) with s2r V(t-1) dims 0-127 between its packets.
        other_sums = _cross(lane_sum, cross_addresses)
        v_lo = _words(read_k_g2s(soffsets_next, s2r=_v_reads(v_s2r, 0)))
        v_ops = _v_operands(v_lo, 0, True)
        _stage_end()
        # Step 2 compute: mask S(t); PV O dims 0-127 with l += row sum and the lane max of S(t).
        s_cur = _mask(_join(s_lo, s_hi), t, row, q_len, kv_len, CAUSAL)
        o0, row_sum, lane_max = _pv(p_bf16, v_lo, o0, True, v_ops,
                                    summary_args=(s_cur, lane_sum, other_sums, row_sum))
        _schedule(32, 3, 3)
        _stage_end()
        # Step 3 memory: exchange row maxima; K(t+1) lands; s2r V(t-1) dims 128-255.
        other_maxima = _cross(lane_max, cross_addresses)
        _wait(vmcnt=0)
        v_hi = _read_v_s2r(v_s2r, 1)
        v_ops = _v_operands(v_hi, 0, True)
        # These reads touch only V's high D128 half. The leading group's next step-0 DMA
        # writes only the low half; the trailing DMA follows our step-3 retirement.
        _stage_end()
        # Step 3 compute: PV O dims 128-255 with new m and S'(t); lazy rescale of O and l.
        o1, s_cur, new_max, ballot = _pv(p_bf16, v_hi, o1, True, v_ops,
                                         (s_cur, lane_max, other_maxima, scale, row_max))
        _schedule(32, 3, 4)
        o0, o1, row_sum = _rescale(o0, o1, row_sum, row_max, new_max, ballot)
        new_max = _advance_max(row_max, new_max)
        _stage_end()
        return s_cur, new_max, row_sum, o0, o1, soffsets_cur, soffsets_next, rows_next2

    def epilogue(state):
        """g2s V(last) with rows past kv_len reading 0; exp, row sum and PV of the last tile."""
        s_last, row_max, row_sum, o0, o1, soffsets_last = state[:6]
        # Memory: g2s V(last); a narrow tail drains only the first 32-token plane.
        read_v_g2s(fx.Vector(soffsets_last), tail_tile=last, packets=8 if narrow_tail else 16)
        # P(last) and l are computed while V(last) lands.
        p_last = _exps(fx.Vector(s_last), fx.Float32(row_max))
        lane_sum = _sum(p_last)
        lane_sum = lane_sum + lane_sum.shuffle_xor(16, 64)
        lane_sum = lane_sum + lane_sum.shuffle_xor(32, 64)
        row_sum = fx.Float32(row_sum) + lane_sum
        p_bf16 = pack(p_last)
        _wait(vmcnt=0, lgkmcnt=0)
        _stage_end()
        _stage_end()
        # Memory: s2r V(last) dims 0-127.  Compute: PV O dims 0-127.
        v_lo = _read_v_s2r(v_s2r, 0, 1 if narrow_tail else 2)
        _wait(lgkmcnt=0)
        _stage_end()
        o0 = _pv(p_bf16, v_lo, o0, steps=2 if narrow_tail else 4)
        _stage_end()
        # Memory: s2r V(last) dims 128-255.  Compute: PV O dims 128-255.
        v_hi = _read_v_s2r(v_s2r, 1, 1 if narrow_tail else 2)
        _wait(lgkmcnt=0)
        _stage_end()
        o1 = _pv(p_bf16, v_hi, o1, steps=2 if narrow_tail else 4)
        _stage_end()
        if fx.const_expr(not STAGGER):
            # Waves 0-3 wait one stage for waves 4-7.
            _stage_end()
        return row_max, row_sum, o0, o1

    # Pipeline: rows run two tiles ahead of QK, K one tile ahead; PV runs one tile behind.
    state = prologue()
    if fx.const_expr(not CAUSAL and not WITH_LSE):
        # Four phases per trip amortize the scalar SOFFSET queue moves in the full path.
        unroll = 4
        loop_end = tiles - 1 if narrow_tail else tiles
        for t in range(fx.Int32(1), loop_end - (unroll - 1), fx.Int32(unroll)):
            for j in fx.range_constexpr(unroll):
                state = phase(*state, t + j, False)
        remainder = ((loop_end - 1) & -unroll) + 1
        for t in range(remainder, loop_end, fx.Int32(1)):
            state = phase(*state, t, False)
        if fx.const_expr(narrow_tail):
            state = phase(*state, last, True)
    else:
        # Causal bounds and LSE increase live state; two phases per trip avoid the
        # scalar spills of four-phase causal/LSE specializations.
        for t in range(fx.Int32(1), tiles - 1, fx.Int32(2)):
            state = phase(*state, t, False)
            state = phase(*state, t + 1, False)
        if (tiles & 1) == 0:
            state = phase(*state, last, False)
    row_max, row_sum, o0, o1 = epilogue(state)

    inv = (row_sum > 0.0).select(fx.Float32(1.0) / row_sum, fx.Float32(0.0))
    output = rocdl.make_buffer_tensor(fx.make_view(fx.get_iter(O) + (fx.Int64(q0) + q_start) * (H * D) + fx.Int64(head) * D,
                                                 fx.make_layout(BM * H * D, 1)), num_records_bytes=valid * H * D * 2)
    pinned_tid = _pin_i32(tid)
    o_parts = _shuffle_o_r2s2r(o0, o1, inv, shared, pinned_tid)
    _store_o_r2g(o_parts, output, pinned_tid, H)
    # The next task's K(0) DMA overwrites LDS.
    _stage_end()
    if fx.const_expr(WITH_LSE):
        if (lane < 16) & (row < q_len):
            log_sum = fx.Float32(llvm.call_intrinsic(fx.Float32.ir_type, "llvm.log2.f32", [row_sum.ir_value()], [], []))
            LSE[(q0 + row) * H + head] = (row_max + log_sum) * fx.Float32(math.log(2.0))


@flyc.jit
def _work(Q, K, V, O, LSE, CQ, CK, TABLE, storage, work,
          H: fx.Constexpr[int], HK: fx.Constexpr[int], NK: fx.Constexpr[int], B: fx.Constexpr[int],
          MAX_PAGES: fx.Constexpr[int], PAGE: fx.Constexpr[int], PAGED: fx.Constexpr[bool],
          CAUSAL: fx.Constexpr[bool], WITH_LSE: fx.Constexpr[bool], SCALE: fx.Constexpr[float],
          STATIC_Q: fx.Constexpr[int], STATIC_K: fx.Constexpr[int]):
    body = _body
    head, batch, qb = work % H, (work // H) % B, work // (H * B)
    q_len = fx.Int32(STATIC_Q) if STATIC_Q > 0 else _uniform(CQ[batch + 1]) - _uniform(CQ[batch])
    if qb * BM < q_len:
        group = _uniform(fx.Int32(gpu.thread_id("x")) >> 8)
        if group != 0:
            body(Q, K, V, O, LSE, CQ, CK, TABLE, storage, head, batch, qb,
                 H, HK, NK, MAX_PAGES, PAGE, PAGED, CAUSAL, WITH_LSE, SCALE, True, STATIC_Q, STATIC_K)
        else:
            body(Q, K, V, O, LSE, CQ, CK, TABLE, storage, head, batch, qb,
                 H, HK, NK, MAX_PAGES, PAGE, PAGED, CAUSAL, WITH_LSE, SCALE, False, STATIC_Q, STATIC_K)


@flyc.kernel(name="dense_mha_bf16_d256", known_block_size=[THREADS, 1, 1])
def _linear_256_kernel(Q: fx.Tensor, K: fx.Tensor, V: fx.Tensor, O: fx.Tensor, LSE: fx.Tensor,
    CQ: fx.Tensor, CK: fx.Tensor, TABLE: fx.Tensor,
    H: fx.Constexpr[int], HK: fx.Constexpr[int], NK: fx.Constexpr[int], B: fx.Constexpr[int],
    MAX_Q: fx.Constexpr[int], MAX_PAGES: fx.Constexpr[int], PAGE: fx.Constexpr[int], PAGED: fx.Constexpr[bool],
    CAUSAL: fx.Constexpr[bool], WITH_LSE: fx.Constexpr[bool], SCALE: fx.Constexpr[float],
    PERSISTENT: fx.Constexpr[bool], CUS: fx.Constexpr[int], STATIC_Q: fx.Constexpr[int], STATIC_K: fx.Constexpr[int]):
    work_body = _work
    storage = fx.SharedAllocator().allocate(fx.Array[fx.Int8, LDS_BYTES, 16]).peek().view(fx.make_layout(LDS_BYTES, 1))
    work = fx.Int32(gpu.block_id("x"))
    if fx.const_expr(PERSISTENT and CAUSAL and B == 1 and not PAGED):
        # A single request's causal cost grows with qb. Longest-first snake
        # rows balance persistent workers without changing a tile's arithmetic.
        query_blocks = (MAX_Q + BM - 1) // BM
        tasks = H * query_blocks
        while work < ((tasks + CUS - 1) // CUS) * CUS:
            row, column = work // CUS, work % CUS
            rank = row * CUS + ((row & 1) != 0).select(CUS - 1 - column, column)
            if rank < tasks:
                head, qb = rank % H, query_blocks - 1 - rank // H
                mapped = qb * H + head
                work_body(Q, K, V, O, LSE, CQ, CK, TABLE, storage, mapped,
                          H, HK, NK, B, MAX_PAGES, PAGE, PAGED, CAUSAL, WITH_LSE, SCALE, STATIC_Q, STATIC_K)
            work = work + CUS
    elif fx.const_expr(PERSISTENT):
        while work < H * B * ((MAX_Q + BM - 1) // BM):
            query_blocks = (MAX_Q + BM - 1) // BM
            head = work // (B * query_blocks)
            batch = (work // query_blocks) % B
            qb = work % query_blocks
            mapped = (qb * B + batch) * H + head
            work_body(Q, K, V, O, LSE, CQ, CK, TABLE, storage, mapped,
                      H, HK, NK, B, MAX_PAGES, PAGE, PAGED, CAUSAL, WITH_LSE, SCALE, STATIC_Q, STATIC_K)
            work = work + CUS
    else:
        work_body(Q, K, V, O, LSE, CQ, CK, TABLE, storage, work,
                  H, HK, NK, B, MAX_PAGES, PAGE, PAGED, CAUSAL, WITH_LSE, SCALE, STATIC_Q, STATIC_K)


@flyc.jit
def _launch(Q: fx.Tensor, K: fx.Tensor, V: fx.Tensor, O: fx.Tensor, LSE: fx.Tensor,
    CQ: fx.Tensor, CK: fx.Tensor, TABLE: fx.Tensor,
    H: fx.Constexpr[int], HK: fx.Constexpr[int], NK: fx.Constexpr[int], B: fx.Constexpr[int],
    MAX_Q: fx.Constexpr[int], MAX_PAGES: fx.Constexpr[int], PAGE: fx.Constexpr[int], PAGED: fx.Constexpr[bool],
    CAUSAL: fx.Constexpr[bool], WITH_LSE: fx.Constexpr[bool], SCALE: fx.Constexpr[float],
    PERSISTENT: fx.Constexpr[bool], CUS: fx.Constexpr[int], STATIC_Q: fx.Constexpr[int], STATIC_K: fx.Constexpr[int], stream: fx.Stream):
    tasks = H * B * ((MAX_Q + BM - 1) // BM)
    _linear_256_kernel(Q, K, V, O, LSE, CQ, CK, TABLE, H, HK, NK, B, MAX_Q, MAX_PAGES,
        PAGE, PAGED, CAUSAL, WITH_LSE, SCALE, PERSISTENT, CUS, STATIC_Q, STATIC_K,
        value_attrs={"rocdl.waves_per_eu": 2,
                     "llvm.target_features": ir.Attribute.parse('#llvm.target_features<["-packed-fp32-ops"]>')},
    ).launch(grid=(min(CUS, tasks) if PERSISTENT else tasks, 1, 1), block=(THREADS, 1, 1), stream=stream)


_COMPILED = {}


def run(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, *, out,
        lse=None, page_size=1, causal=False, softmax_scale=None, block_table=None, persistent=True, stream=None):
    """Internal launch; the central MHA varlen API validates bounds and aliases."""
    if q.numel() == 0:
        return (out, lse) if lse is not None else out
    stream = torch.cuda.current_stream(q.device) if stream is None else stream
    prop = torch.cuda.get_device_properties(q.device)
    table = cu_seqlens_k if block_table is None else block_table.reshape(-1)
    scale = D**-0.5 if softmax_scale is None else float(softmax_scale)
    # Validated B1/nonpaged offsets are [0, tensor.shape[0]]; maxima alone
    # cannot establish this for ragged batches or logical paged KV lengths.
    static = cu_seqlens_q.numel() == 2 and block_table is None and not causal and lse is None
    args = (q.view(-1), k.view(-1), v.view(-1), out.view(-1), q.view(-1) if lse is None else lse.view(-1),
            cu_seqlens_q, cu_seqlens_k, table, q.shape[1], k.shape[1], k.shape[0], cu_seqlens_q.numel() - 1,
            max_seqlen_q, 1 if block_table is None else block_table.shape[1], page_size, block_table is not None,
            causal, lse is not None, scale, persistent, prop.multi_processor_count,
            q.shape[0] if static else 0, k.shape[0] if static else 0, stream)
    signature = tuple((a.dtype, tuple(a.shape), tuple(a.stride())) if isinstance(a, torch.Tensor)
                      else ("stream",) if hasattr(a, "cuda_stream") else a for a in args)
    key = (q.device, signature)
    with torch.cuda.device(q.device), torch.cuda.stream(stream):
        compiled = _COMPILED.get(key)
        if compiled is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("warm this linear specialization before graph capture")
            # flyc.compile also launches this first call.
            _COMPILED[key] = flyc.compile(_launch, *args)
        else:
            compiled(*args)
    return (out, lse) if lse is not None else out


__all__ = ["run"]
