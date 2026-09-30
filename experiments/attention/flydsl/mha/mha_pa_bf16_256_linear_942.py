"""Native gfx942 BF16 D256 attention over linear Q/K/V/O, without KV conversion.

BM128/BN64, eight staggered waves, 32-KiB K + 32-KiB row-major V in LDS.
K is XOR-swizzled at the DMA source. V is transposed after LDS reads with
the same 2x2 BF16 byte-permutation selectors used by CK transpose_vectors.
The PV N coordinate is permuted so each lane transposes contiguous 8x8
pieces; the output shuffle restores ordinary [T,H,D] storage.
"""

import math

import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu, rocdl
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm

if __package__:
    from . import mha_pa_bf16_256_942 as base
    from .mha_pa_bf16_942 import (
        _uniform, _min, _pin_i32, _pin, _join, _maximum, _pack_bf16,
        _stage_end, _wait, _schedule, _buffer,
        _read_address, _rescale, _advance_max,
    )
else:
    import mha_pa_bf16_256_942 as base
    from mha_pa_bf16_942 import (
        _uniform, _min, _pin_i32, _pin, _join, _maximum, _pack_bf16,
        _stage_end, _wait, _schedule, _buffer,
        _read_address, _rescale, _advance_max,
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
    token = (wave & 3) * 16 + packet if page == 4 else (wave & 3) + packet * 4
    return token, (wave >> 2) * 128


def _source_rows(table, tile, kv_start, kv_len, wave, page, paged, hk):
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
    if page:
        return fx.Vector.from_elements([fx.Int32(rocdl.readlane(fx.Int32.ir_type, rows.ir_value(), fx.Int32(packet).ir_value()))
                                        for packet in range(4 if page == 4 else 16)], fx.Int32)
    return fx.Vector.from_elements([rows], fx.Int32)


def _dma(resource, storage, wave, lane, rows, tile, kv_len, hk, packet, is_v, extent, page, tail=False,
         read_address=None, read_immediate=0):
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
    if read_address is not None:
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
        fx.Int32(0).ir_value(), fx.Int32(0).ir_value())


def _read_k(addresses, half, lower_first=False):
    parts = [None] * 16
    order = [(n, k + high * 4) for high in range(2) for n in range(2) for k in range(4)] if lower_first else [(n, k) for n in range(2) for k in range(8)]
    for n, k in order:
        parts[n * 8 + k] = _read_address(addresses[n * 2 + k % 2], half * 16384 + (k // 2) * 128)
    words = fx.Vector.from_elements([part[i] for part in parts for i in range(4)], fx.Int32)
    frag = fx.make_rmem_tensor(fx.make_layout((4, 16, 2), (1, 4, 64)), fx.BFloat16)
    frag.store(words.bitcast(fx.BFloat16))
    return fx.make_view(fx.get_iter(frag), fx.make_layout((4, 2, 16), (1, 64, 4)))


def _read_v(address, half):
    parts = []
    for block in range(2):
        for row in range(8):
            parts.append(_read_address(address, half * 256 + block * 16384 + row * 512))
    return fx.Vector.from_elements([part[i] for part in parts for i in range(4)], fx.Int32)


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


def _pv(probabilities, values, output, progressive=False, prepared=None, center_args=None, summary_args=None):
    p = fx.Vector(probabilities)
    acc = [fx.Vector.from_elements([output[n * 4 + i] for i in range(4)], fx.Float32) for n in range(8)]
    prepared_steps = 0 if prepared is None else prepared.numel // 16
    operands = _v_operands(values, 0, progressive) if prepared is None else fx.Vector.from_elements([prepared[i] for i in range(16)], fx.Int32)
    centered = []
    rocdl.sched_barrier(0)
    for step in range(4):
        if step + 1 < prepared_steps:
            following = fx.Vector.from_elements([prepared[(step + 1) * 16 + i] for i in range(16)], fx.Int32)
        else:
            following = _v_operands(values, step + 1, progressive) if step < 3 else operands
        b = fx.Vector.from_elements([p[step * 4 + i] for i in range(4)], fx.BFloat16)
        for n in range(8):
            a = fx.Vector.from_elements([operands[n * 2 + i] for i in range(2)], fx.Int32).bitcast(fx.Int16)
            acc[n] = fx.Vector(rocdl.mfma_f32_16x16x16bf16_1k(ir.VectorType.get([4], fx.Float32.ir_type),
                [a.ir_value(), b.bitcast(fx.Int16).ir_value(), acc[n].ir_value(), 0, 0, 0]))
        if center_args is not None:
            current, candidate, maxima, scale, maximum = center_args
            if step == 0:
                candidate = _maximum(_maximum(candidate, maxima[0]), _maximum(maxima[1], maxima[2])) * scale
                predicate = candidate > maximum + 7.0
                ballot = fx.Int64(rocdl.ballot(fx.Int64.ir_type, predicate.ir_value()))
                new_max = predicate.select(candidate + 1.0, maximum)
            for i in range(step * 4, step * 4 + 4):
                centered.append(fx.Float32(llvm.inline_asm(fx.Float32.ir_type,
                    [current[i].ir_value(), scale.ir_value(), new_max.ir_value()],
                    "v_fma_f32 $0, $1, $2, -$3", "=v,v,v,v", has_side_effects=False)))
        if summary_args is not None and step == 3:
            current, total, sums, row_sum = summary_args
            row_sum = row_sum + ((total + sums[0]) + (sums[1] + sums[2]))
            candidate = base._max(current)
        if step < 3 and step + 1 >= prepared_steps:
            _schedule(8, 3 if center_args is not None else 2, 20 + step)
        elif summary_args is not None:
            _schedule(8, 3, 23)
        rocdl.sched_barrier(0)
        operands = following
    result = fx.Vector.from_elements([acc[n][i] for n in range(8) for i in range(4)], fx.Float32)
    if center_args is not None:
        return result, fx.Vector.from_elements(centered, fx.Float32), new_max, ballot
    if summary_args is not None:
        return result, row_sum, candidate
    return result


def _output(o0, o1, inv, buffer, storage, shared, tid, heads):
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
    atom = fx.make_copy_atom(rocdl.BufferCopy128b(), fx.BFloat16)
    parts = []
    for part in range(8):
        element = tid * 8 + part * THREADS * 8
        read_row, column = element // 256, element % 256
        parts.append(_read_address(shared + (element ^ ((read_row & 7) * 8)) * 2))
    _wait(lgkmcnt=0)
    rocdl.sched_barrier(0)
    for part in range(8):
        element = tid * 8 + part * THREADS * 8
        read_row, column = element // 256, element % 256
        fragment = fx.make_rmem_tensor(8, fx.BFloat16)
        fragment.store(parts[part].bitcast(fx.BFloat16))
        offset = read_row * heads * D + column
        fx.copy(atom, fragment, fx.make_view(fx.get_iter(buffer) + offset, fx.make_layout(8, 1)))
    _stage_end()


@flyc.jit
def _body(Q, K, V, O, LSE, CQ, CK, TABLE, storage, head, batch, qb,
          H: fx.Constexpr[int], HK: fx.Constexpr[int], NK: fx.Constexpr[int], MAX_PAGES: fx.Constexpr[int],
          PAGE: fx.Constexpr[int], PAGED: fx.Constexpr[bool], CAUSAL: fx.Constexpr[bool],
          WITH_LSE: fx.Constexpr[bool], SCALE: fx.Constexpr[float], STAGGER: fx.Constexpr[bool]):
    read_k, read_v, dma, source_rows, pv = _read_k, _read_v, _dma, _source_rows, _pv
    dma_offsets = _dma_offsets
    v_operands = _v_operands
    qk, local_sum, local_max, cross = base._qk, base._sum, base._max, base._cross
    exps, pack, mask, center = base._exps, base._pack, base._mask, base._center
    rescale, advance_max = _rescale, _advance_max
    q0, k0 = _uniform(CQ[batch]), _uniform(CK[batch])
    q_len, kv_len = _uniform(CQ[batch + 1]) - q0, _uniform(CK[batch + 1]) - k0
    tid = fx.Int32(gpu.thread_id("x"))
    wave, lane = _uniform(tid >> 6), tid & 63
    q_start = qb * BM
    row = q_start + wave * 16 + (lane & 15)
    valid = _min(fx.Int32(BM), q_len - q_start)
    hkv = head // (H // HK)
    table = fx.make_view(fx.get_iter(TABLE) + batch * MAX_PAGES, fx.make_layout(MAX_PAGES, 1))
    gq = _buffer(fx.make_view(fx.get_iter(Q) + (fx.Int64(q0) + q_start) * (H * D) + fx.Int64(head) * D,
                              fx.make_layout(BM * H * D, 1)), valid * H * D * 2)
    extent = (NK * HK - hkv) * D * 2
    gk = _buffer(fx.make_view(fx.get_iter(K) + hkv * D, fx.make_layout((NK * HK - hkv) * D, 1)), extent)
    gv = _buffer(fx.make_view(fx.get_iter(V) + hkv * D, fx.make_layout((NK * HK - hkv) * D, 1)), extent)
    q = base._q_fragment(gq, (wave * 16 + (lane & 15)) * H * D * 2, lane)
    scale = fx.Float32(SCALE * math.log2(math.e))
    tiles = (kv_len + BN - 1) // BN
    if fx.const_expr(CAUSAL):
        end = (q_start + valid + kv_len - q_len + BN - 1) // BN
        tiles = _min(tiles, (end > 0).select(end, fx.Int32(1)))
    last = tiles - 1
    shared = fx.Int32(fx.ptrtoint(fx.get_iter(storage)))
    key_row = (lane & 3) + ((lane & 12) << 1)
    kr = tuple(_pin_i32(shared + _k_lds_address(key_row + n * 4, (lane >> 4) * 8 + parity * 32))
               for n in range(2) for parity in range(2))
    vr = _pin_i32(shared + _v_lds_address((lane >> 4) * 8, (lane & 15) * 8))
    cross_addresses = tuple(_pin_i32((lane ^ offset) * 4) for offset in (16, 32, 48))
    rows0 = source_rows(table, fx.Int32(0), k0, kv_len, wave, PAGE, PAGED, HK)
    rows1 = source_rows(table, _min(fx.Int32(1), last), k0, kv_len, wave, PAGE, PAGED, HK)
    rows2 = source_rows(table, _min(fx.Int32(2), last), k0, kv_len, wave, PAGE, PAGED, HK)
    offsets = dma_offsets(rows0, PAGE if PAGED else 0)
    v_offsets = offsets
    for packet in fx.range_constexpr(16):
        dma(gk, storage, wave, lane, offsets, fx.Int32(0), kv_len, HK, packet, False, extent, PAGE if PAGED else 0)
    _wait(vmcnt=0, lgkmcnt=0)
    _stage_end()
    if fx.const_expr(STAGGER):
        _stage_end()
    k = read_k(kr, 0)
    _wait(lgkmcnt=0)
    _stage_end()
    lo = qk(q, k)
    o0, o1 = _pin(fx.Vector.filled(32, 0.0, fx.Float32)), _pin(fx.Vector.filled(32, 0.0, fx.Float32))
    _schedule(32, 3, 5)
    _stage_end()
    k = read_k(kr, 1)
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
    offsets = dma_offsets(rows1, PAGE if PAGED else 0)
    current_offsets = offsets
    for packet in fx.range_constexpr(16):
        dma(gk, storage, wave, lane, offsets, _min(fx.Int32(1), last), kv_len, HK, packet, False, extent, PAGE if PAGED else 0)
    _wait(vmcnt=0, lgkmcnt=0)
    _stage_end()
    _stage_end()

    @flyc.jit
    def phase(previous, maximum, row_sum, o0, o1, previous_offsets, current_offsets, next_rows, t):
        previous, maximum, row_sum = fx.Vector(previous), fx.Float32(maximum), fx.Float32(row_sum)
        o0, o1, t = fx.Vector(o0), fx.Vector(o1), fx.Int32(t)
        previous_offsets = fx.Vector(previous_offsets)
        current_offsets, next_rows = fx.Vector(current_offsets), fx.Int32(next_rows)
        future_rows = source_rows(table, _min(t + 2, last), k0, kv_len, wave, PAGE, PAGED, HK)
        parts = []
        for n in fx.range_constexpr(2):
            for step in fx.range_constexpr(8):
                parts.append(dma(gv, storage, wave, lane, previous_offsets, t - 1, kv_len, HK,
                    n * 8 + step, True, extent, PAGE if PAGED else 0,
                    read_address=kr[n * 2 + step % 2], read_immediate=(step // 2) * 128))
        words = fx.Vector.from_elements([part[i] for part in parts for i in range(4)], fx.Int32)
        fragment = fx.make_rmem_tensor(fx.make_layout((4, 16, 2), (1, 4, 64)), fx.BFloat16)
        fragment.store(words.bitcast(fx.BFloat16))
        k = fx.make_view(fx.get_iter(fragment), fx.make_layout((4, 2, 16), (1, 64, 4)))
        _stage_end()
        _wait(lgkmcnt=0)
        rocdl.sched_barrier(0)
        lo = qk(q, k)
        previous = exps(previous)
        _schedule(32, 1, 1, True)
        _stage_end()
        k = read_k(kr, 1, STAGGER)
        _wait(vmcnt=0)
        if fx.const_expr(STAGGER):
            # Leading DMA owns D[0:128]; retire those eight reads before it
            # can overwrite K. The remaining reads only touch D[128:256].
            _wait(lgkmcnt=8)
        _stage_end()
        _wait(lgkmcnt=0)
        rocdl.sched_barrier(0)
        hi = qk(q, k)
        total, probabilities = local_sum(previous), pack(previous)
        offsets = dma_offsets(next_rows, PAGE if PAGED else 0)
        _schedule(32, 3, 2)
        _stage_end()
        sums = cross(total, cross_addresses)
        parts = []
        for block in fx.range_constexpr(2):
            for r in fx.range_constexpr(8):
                parts.append(dma(gk, storage, wave, lane, offsets, _min(t + 1, last), kv_len, HK,
                    block * 8 + r, False, extent, PAGE if PAGED else 0,
                    read_address=vr, read_immediate=block * 16384 + r * 512))
        v = fx.Vector.from_elements([part[i] for part in parts for i in range(4)], fx.Int32)
        prepared = v_operands(v, 0, True)
        _stage_end()
        current = mask(_join(lo, hi), t, row, q_len, kv_len, CAUSAL)
        o0, row_sum, candidate = pv(probabilities, v, o0, True, prepared,
            summary_args=(current, total, sums, row_sum))
        _schedule(32, 3, 3)
        _stage_end()
        maxima = cross(candidate, cross_addresses)
        _wait(vmcnt=0)
        v = read_v(vr, 1)
        prepared = v_operands(v, 0, True)
        # These reads touch only V's high D128 half. Leading next-S0 DMA
        # writes only the low half; trailing DMA follows our S7 retirement.
        _stage_end()
        o1, current, new_max, ballot = pv(probabilities, v, o1, True, prepared,
            (current, candidate, maxima, scale, maximum))
        _schedule(32, 3, 4)
        o0, o1, row_sum = rescale(o0, o1, row_sum, maximum, new_max, ballot)
        new_max = advance_max(maximum, new_max)
        _stage_end()
        return current, new_max, row_sum, o0, o1, current_offsets, offsets, future_rows

    if fx.const_expr(not CAUSAL and not WITH_LSE):
        # Four phases amortize scalar offset queue moves in the full path.
        for t in range(fx.Int32(1), tiles - 3, fx.Int32(4)):
            for j in fx.range_constexpr(4):
                scores, maximum, row_sum, o0, o1, v_offsets, current_offsets, rows2 = phase(
                    scores, maximum, row_sum, o0, o1, v_offsets, current_offsets, rows2, t + j)
        remainder = ((tiles - 1) & -4) + 1
        for t in range(remainder, tiles, fx.Int32(1)):
            scores, maximum, row_sum, o0, o1, v_offsets, current_offsets, rows2 = phase(
                scores, maximum, row_sum, o0, o1, v_offsets, current_offsets, rows2, t)
    else:
        # Causal bounds and LSE increase live state; two phases avoid scalar
        # spills observed with four-phase causal/LSE specializations.
        for t in range(fx.Int32(1), tiles - 1, fx.Int32(2)):
            scores, maximum, row_sum, o0, o1, v_offsets, current_offsets, rows2 = phase(
                scores, maximum, row_sum, o0, o1, v_offsets, current_offsets, rows2, t)
            scores, maximum, row_sum, o0, o1, v_offsets, current_offsets, rows2 = phase(
                scores, maximum, row_sum, o0, o1, v_offsets, current_offsets, rows2, t + 1)
        if (tiles & 1) == 0:
            scores, maximum, row_sum, o0, o1, v_offsets, current_offsets, rows2 = phase(
                scores, maximum, row_sum, o0, o1, v_offsets, current_offsets, rows2, last)
    offsets = fx.Vector(v_offsets)
    for packet in fx.range_constexpr(16):
        dma(gv, storage, wave, lane, offsets, last, kv_len, HK, packet, True, extent, PAGE if PAGED else 0, True)
    scores = exps(fx.Vector(scores))
    total = local_sum(scores)
    total = total + total.shuffle_xor(16, 64)
    total = total + total.shuffle_xor(32, 64)
    row_sum = fx.Float32(row_sum) + total
    probabilities = pack(scores)
    _wait(vmcnt=0, lgkmcnt=0)
    _stage_end()
    _stage_end()
    v = read_v(vr, 0)
    _wait(lgkmcnt=0)
    _stage_end()
    o0 = pv(probabilities, v, o0)
    _stage_end()
    v = read_v(vr, 1)
    _wait(lgkmcnt=0)
    _stage_end()
    o1 = pv(probabilities, v, o1)
    _stage_end()
    if fx.const_expr(not STAGGER):
        _stage_end()
    inv = (row_sum > 0.0).select(fx.Float32(1.0) / row_sum, fx.Float32(0.0))
    output = rocdl.make_buffer_tensor(fx.make_view(fx.get_iter(O) + (fx.Int64(q0) + q_start) * (H * D) + fx.Int64(head) * D,
                                                 fx.make_layout(BM * H * D, 1)), num_records_bytes=valid * H * D * 2)
    _output(o0, o1, inv, output, storage, shared, _pin_i32(tid), H)
    if fx.const_expr(WITH_LSE):
        if (lane < 16) & (row < q_len):
            log_sum = fx.Float32(llvm.call_intrinsic(fx.Float32.ir_type, "llvm.log2.f32", [row_sum.ir_value()], [], []))
            LSE[(q0 + row) * H + head] = (maximum + log_sum) * fx.Float32(math.log(2.0))


@flyc.jit
def _work(Q, K, V, O, LSE, CQ, CK, TABLE, storage, work,
          H: fx.Constexpr[int], HK: fx.Constexpr[int], NK: fx.Constexpr[int], B: fx.Constexpr[int],
          MAX_PAGES: fx.Constexpr[int], PAGE: fx.Constexpr[int], PAGED: fx.Constexpr[bool],
          CAUSAL: fx.Constexpr[bool], WITH_LSE: fx.Constexpr[bool], SCALE: fx.Constexpr[float]):
    body = _body
    head, batch, qb = work % H, (work // H) % B, work // (H * B)
    q_len = _uniform(CQ[batch + 1]) - _uniform(CQ[batch])
    if qb * BM < q_len:
        group = _uniform(fx.Int32(gpu.thread_id("x")) >> 8)
        if group != 0:
            body(Q, K, V, O, LSE, CQ, CK, TABLE, storage, head, batch, qb,
                 H, HK, NK, MAX_PAGES, PAGE, PAGED, CAUSAL, WITH_LSE, SCALE, True)
        else:
            body(Q, K, V, O, LSE, CQ, CK, TABLE, storage, head, batch, qb,
                 H, HK, NK, MAX_PAGES, PAGE, PAGED, CAUSAL, WITH_LSE, SCALE, False)


@flyc.kernel(known_block_size=[THREADS, 1, 1])
def _linear_256_kernel(Q: fx.Tensor, K: fx.Tensor, V: fx.Tensor, O: fx.Tensor, LSE: fx.Tensor,
    CQ: fx.Tensor, CK: fx.Tensor, TABLE: fx.Tensor,
    H: fx.Constexpr[int], HK: fx.Constexpr[int], NK: fx.Constexpr[int], B: fx.Constexpr[int],
    MAX_Q: fx.Constexpr[int], MAX_PAGES: fx.Constexpr[int], PAGE: fx.Constexpr[int], PAGED: fx.Constexpr[bool],
    CAUSAL: fx.Constexpr[bool], WITH_LSE: fx.Constexpr[bool], SCALE: fx.Constexpr[float],
    PERSISTENT: fx.Constexpr[bool], CUS: fx.Constexpr[int]):
    work_body = _work
    storage = fx.SharedAllocator().allocate(fx.Array[fx.Int8, LDS_BYTES, 16]).peek().view(fx.make_layout(LDS_BYTES, 1))
    work = fx.Int32(gpu.block_id("x"))
    if fx.const_expr(PERSISTENT):
        while work < H * B * ((MAX_Q + BM - 1) // BM):
            query_blocks = (MAX_Q + BM - 1) // BM
            head = work // (B * query_blocks)
            batch = (work // query_blocks) % B
            qb = work % query_blocks
            mapped = (qb * B + batch) * H + head
            work_body(Q, K, V, O, LSE, CQ, CK, TABLE, storage, mapped,
                      H, HK, NK, B, MAX_PAGES, PAGE, PAGED, CAUSAL, WITH_LSE, SCALE)
            work = work + CUS
    else:
        work_body(Q, K, V, O, LSE, CQ, CK, TABLE, storage, work,
                  H, HK, NK, B, MAX_PAGES, PAGE, PAGED, CAUSAL, WITH_LSE, SCALE)


@flyc.jit
def _launch(Q: fx.Tensor, K: fx.Tensor, V: fx.Tensor, O: fx.Tensor, LSE: fx.Tensor,
    CQ: fx.Tensor, CK: fx.Tensor, TABLE: fx.Tensor,
    H: fx.Constexpr[int], HK: fx.Constexpr[int], NK: fx.Constexpr[int], B: fx.Constexpr[int],
    MAX_Q: fx.Constexpr[int], MAX_PAGES: fx.Constexpr[int], PAGE: fx.Constexpr[int], PAGED: fx.Constexpr[bool],
    CAUSAL: fx.Constexpr[bool], WITH_LSE: fx.Constexpr[bool], SCALE: fx.Constexpr[float],
    PERSISTENT: fx.Constexpr[bool], CUS: fx.Constexpr[int], stream: fx.Stream):
    tasks = H * B * ((MAX_Q + BM - 1) // BM)
    _linear_256_kernel(Q, K, V, O, LSE, CQ, CK, TABLE, H, HK, NK, B, MAX_Q, MAX_PAGES,
        PAGE, PAGED, CAUSAL, WITH_LSE, SCALE, PERSISTENT, CUS,
        value_attrs={"rocdl.waves_per_eu": 2, "passthrough": [["target-features", "-packed-fp32-ops"]]},
    ).launch(grid=(min(CUS, tasks) if PERSISTENT else tasks, 1, 1), block=(THREADS, 1, 1), stream=stream)


_COMPILED = {}


def run(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, *, out,
        lse=None, page_size=1, causal=False, softmax_scale=None, block_table=None, persistent=True, stream=None):
    """Internal validated launch; public validation lives in the varlen adapter."""
    if q.numel() == 0:
        return (out, lse) if lse is not None else out
    stream = torch.cuda.current_stream(q.device) if stream is None else stream
    prop = torch.cuda.get_device_properties(q.device)
    table = cu_seqlens_k if block_table is None else block_table.reshape(-1)
    scale = D**-0.5 if softmax_scale is None else float(softmax_scale)
    args = (q.view(-1), k.view(-1), v.view(-1), out.view(-1), q.view(-1) if lse is None else lse.view(-1),
            cu_seqlens_q, cu_seqlens_k, table, q.shape[1], k.shape[1], k.shape[0], cu_seqlens_q.numel() - 1,
            max_seqlen_q, 1 if block_table is None else block_table.shape[1], page_size, block_table is not None,
            causal, lse is not None, scale, persistent, prop.multi_processor_count, stream)
    signature = tuple((a.dtype, tuple(a.shape), tuple(a.stride())) if isinstance(a, torch.Tensor)
                      else ("stream",) if hasattr(a, "cuda_stream") else a for a in args)
    key = (q.device, signature)
    with torch.cuda.device(q.device), torch.cuda.stream(stream):
        compiled = _COMPILED.get(key)
        if compiled is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("warm this linear specialization before graph capture")
            _COMPILED[key] = flyc.compile(_launch, *args)
        else:
            compiled(*args)
    return (out, lse) if lse is not None else out


__all__ = ["run"]