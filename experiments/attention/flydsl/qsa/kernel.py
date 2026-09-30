"""gfx942 D256 multi-query/multi-head sparse attention with exact membership.

The LDS, QK/PV MFMA and staggered pipeline are derived from the adjacent
mha_pa_bf16_256_linear_942 implementation. Q/O rows and sparse masks differ.
"""

import math

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import gpu, rocdl

from ..mha import mha_pa_bf16_256_942 as base
from ..mha import mha_pa_bf16_256_linear_942 as linear
from ..mha.mha_pa_bf16_942 import (
    _advance_max,
    _buffer,
    _join,
    _maximum,
    _min,
    _pack_bf16,
    _pin,
    _pin_i32,
    _read_address,
    _rescale,
    _schedule,
    _stage_end,
    _uniform,
    _wait,
)

BM, BN, D, THREADS, LDS_BYTES = 128, 64, 256, 512, 65536


def _source_rows(table, tile, count, wave, hk):
    lane = fx.Int32(gpu.thread_id("x")) & 63
    index = _min(tile * 16 + (wave & 3) * 4 + (lane & 3), count - 1)
    block = fx.Int32(table[index])
    return block * (4 * hk * D * 2) + (wave >> 2) * 256


def _bounded_dma(
    resource,
    storage,
    wave,
    lane,
    rows,
    tile,
    kv_len,
    hk,
    packet,
    is_v,
    extent,
    page,
    tail=False,
    read_address=None,
    read_immediate=0,
):
    # Raw-buffer bounds exclude SOFFSET; partial physical blocks need full VOFFSET.
    token, _ = linear._copy_coordinates(wave, packet, 4)
    channel_offset = lane * 4 if is_v else (lane ^ (linear._k_phase(token) * 4)) * 4
    offset = rows[packet // 4] + (packet & 3) * (hk * D * 2) + channel_offset
    if tail:
        offset = (tile * BN + token < kv_len).select(offset, fx.Int32(extent))
    base_row, base_channel = linear._copy_coordinates(wave, 0, 4)
    base_destination = fx.Int32(base_row * 512 + base_channel * 2)
    immediate = (32768 if is_v else 0) + packet * 512
    if read_address is not None:
        return fx.Vector(
            llvm.inline_asm(
                ir.VectorType.get([4], fx.Int32.ir_type),
                [
                    fx.Int32(read_address).ir_value(),
                    resource,
                    fx.Int32(offset).ir_value(),
                    base_destination.ir_value(),
                ],
                f"s_add_u32 m0, $4, {immediate}\n"
                f"ds_read_b128 $0, $1 offset:{read_immediate}\n"
                "buffer_load_dword $3, $2, 0 offen lds",
                "=&v,v,s,v,s,~{m0},~{scc},~{memory}",
                has_side_effects=True,
            )
        )
    destination = fx.Int32(
        llvm.inline_asm(
            fx.Int32.ir_type,
            [base_destination.ir_value()],
            f"s_add_u32 $0, $1, {immediate}",
            "=s,s,~{scc}",
            has_side_effects=True,
        )
    )
    rocdl.raw_ptr_buffer_load_lds(
        resource,
        fx.to_llvm_ptr(fx.get_iter(storage) + destination),
        fx.Int32(4).ir_value(),
        fx.Int32(offset).ir_value(),
        fx.Int32(0).ir_value(),
        fx.Int32(0).ir_value(),
        fx.Int32(0).ir_value(),
    )


@flyc.jit
def _mask(
    scores,
    masks,
    tile,
    query,
    qvalid,
    common_tiles,
    G: fx.Constexpr[int],
    GP: fx.Constexpr[int],
    BQ: fx.Constexpr[int],
):
    values = fx.Vector(scores)
    if tile >= common_tiles:
        lane = fx.Int32(gpu.thread_id("x")) & 63
        offset = (tile * BQ + _min(query, fx.Int32(BQ - 1))) * 4 + (lane >> 4)
        bits = fx.Int32(masks[offset])
        result = []
        for i in fx.range_constexpr(16):
            allowed = ((bits >> i) & 1) != 0
            result.append(allowed.select(values[i], fx.Float32(float("-inf"))))
        values = fx.Vector.from_elements(result, fx.Float32)
    return values


def _output(
    o0, o1, inv, buffer, shared, tid, heads, g, gp, bq, qvalid, extent, row_offset
):
    row = (tid >> 6) * 16 + (tid & 15)
    for half in range(2):
        output = o1 if half else o0
        for i in range(4):
            values = fx.Vector.from_elements(
                [output[n * 4 + i] * inv for n in range(8)], fx.Float32
            )
            words = _pack_bf16(values)
            column = (((tid >> 4) & 3) * 4 + i) * 8 + half * 128
            address = fx.Int32(shared + ((row * 256 + column) ^ ((row & 7) * 8)) * 2)
            llvm.inline_asm(
                ir.Type.parse("!llvm.void"),
                [address.ir_value(), words.ir_value()],
                "ds_write_b128 $0, $1",
                "v,v,~{memory}",
                has_side_effects=True,
            )
    _wait(lgkmcnt=0)
    _stage_end()
    parts = []
    for part in range(8):
        element = tid * 8 + part * THREADS * 8
        read_row = element // 256
        parts.append(_read_address(shared + (element ^ ((read_row & 7) * 8)) * 2))
    _wait(lgkmcnt=0)
    rocdl.sched_barrier(0)
    for part in range(8):
        element = tid * 8 + part * THREADS * 8
        read_row, column = element // 256, element % 256
        query, head = (read_row + row_offset) // gp, (read_row + row_offset) % gp
        offset = (query * heads * D + head * D + column) * 2
        valid = (query < qvalid) & (query < bq) & (head < g)
        offset = valid.select(offset, fx.Int32(extent))
        fragment = fx.make_rmem_tensor(8, fx.BFloat16)
        fragment.store(parts[part].bitcast(fx.BFloat16))
        atom = fx.make_copy_atom(rocdl.BufferCopy128b(), fx.BFloat16)
        fx.copy(
            atom,
            fragment,
            fx.make_view(fx.get_iter(buffer) + offset // 2, fx.make_layout(8, 1)),
        )
    _stage_end()


@flyc.jit
def _body(
    Q,
    K,
    V,
    O,
    META,
    BLOCKS,
    MEMBERS,
    COUNTS,
    storage,
    work,
    H: fx.Constexpr[int],
    HK: fx.Constexpr[int],
    NQ: fx.Constexpr[int],
    NK: fx.Constexpr[int],
    CAP: fx.Constexpr[int],
    BQ: fx.Constexpr[int],
    GP: fx.Constexpr[int],
    SCALE: fx.Constexpr[float],
    TAIL_BOUNDS: fx.Constexpr[bool],
    STAGGER: fx.Constexpr[bool],
):
    read_k, read_v, dma, pv = linear._read_k, linear._read_v, linear._dma, linear._pv
    if fx.const_expr(TAIL_BOUNDS):
        dma = _bounded_dma
    dma_offsets, v_operands = linear._dma_offsets, linear._v_operands
    source_rows, mask, output_fn = _source_rows, _mask, _output
    qk, local_sum, local_max, cross = base._qk, base._sum, base._max, base._cross
    exps, pack, center = base._exps, base._pack, base._center
    rescale, advance_max = _rescale, _advance_max
    slices = (BQ * GP + 127) // 128
    tile_id, hkv = work // (HK * slices), work % HK
    row_offset = ((work // HK) % slices) * 128
    q0, qvalid = _uniform(META[tile_id * 5]), _uniform(META[tile_id * 5 + 1])
    k0, kv_len = _uniform(META[tile_id * 5 + 2]), _uniform(META[tile_id * 5 + 3])
    count = _uniform(COUNTS[tile_id * 2])
    common_tiles = _uniform(COUNTS[tile_id * 2 + 1])
    table = fx.make_view(fx.get_iter(BLOCKS) + tile_id * CAP, fx.make_layout(CAP, 1))
    masks = fx.make_view(
        fx.get_iter(MEMBERS) + tile_id * (CAP // 16) * BQ * 4,
        fx.make_layout((CAP // 16) * BQ * 4, 1),
    )
    tid = fx.Int32(gpu.thread_id("x"))
    wave, lane = _uniform(tid >> 6), tid & 63
    row = row_offset + wave * 16 + (lane & 15)
    query, head = row // GP, row % GP
    valid_row = (query < qvalid) & (query < BQ) & (head < (H // HK))
    qextent = (NQ - q0) * H * D * 2 - hkv * (H // HK) * D * 2
    qptr = fx.get_iter(Q) + fx.Int64(q0) * (H * D) + hkv * (H // HK) * D
    gq = _buffer(fx.make_view(qptr, fx.make_layout(NQ * H * D, 1)), qextent)
    qoffset = valid_row.select((query * H * D + head * D) * 2, fx.Int32(qextent))
    q = base._q_fragment(gq, qoffset, lane)
    extent = (kv_len * HK - hkv) * D * 2
    gk = _buffer(
        fx.make_view(
            fx.get_iter(K) + (fx.Int64(k0) * HK + hkv) * D,
            fx.make_layout(NK * HK * D, 1),
        ),
        extent,
    )
    gv = _buffer(
        fx.make_view(
            fx.get_iter(V) + (fx.Int64(k0) * HK + hkv) * D,
            fx.make_layout(NK * HK * D, 1),
        ),
        extent,
    )
    scale = fx.Float32(SCALE * math.log2(math.e))
    compact_len = count * 4
    tiles = (compact_len + BN - 1) // BN
    last = tiles - 1
    shared = fx.Int32(fx.ptrtoint(fx.get_iter(storage)))
    key_row = (lane & 3) + ((lane & 12) << 1)
    kr = tuple(
        _pin_i32(
            shared
            + linear._k_lds_address(key_row + n * 4, (lane >> 4) * 8 + parity * 32)
        )
        for n in range(2)
        for parity in range(2)
    )
    vr = _pin_i32(shared + linear._v_lds_address((lane >> 4) * 8, (lane & 15) * 8))
    cross_addresses = tuple(_pin_i32((lane ^ offset) * 4) for offset in (16, 32, 48))
    rows0 = source_rows(table, fx.Int32(0), count, wave, HK)
    rows1 = source_rows(table, _min(fx.Int32(1), last), count, wave, HK)
    rows2 = source_rows(table, _min(fx.Int32(2), last), count, wave, HK)
    offsets = dma_offsets(rows0, 4)
    v_offsets = offsets
    for packet in fx.range_constexpr(16):
        dma(
            gk,
            storage,
            wave,
            lane,
            offsets,
            fx.Int32(0),
            compact_len,
            HK,
            packet,
            False,
            extent,
            4,
        )
    _wait(vmcnt=0, lgkmcnt=0)
    _stage_end()
    if fx.const_expr(STAGGER):
        _stage_end()
    k = read_k(kr, 0)
    _wait(lgkmcnt=0)
    _stage_end()
    lo = qk(q, k)
    o0, o1 = _pin(fx.Vector.filled(32, 0.0, fx.Float32)), _pin(
        fx.Vector.filled(32, 0.0, fx.Float32)
    )
    _schedule(32, 3, 5)
    _stage_end()
    k = read_k(kr, 1)
    _wait(lgkmcnt=0)
    _stage_end()
    hi = qk(q, k)
    scores = mask(
        _join(lo, hi), masks, fx.Int32(0), query, qvalid, common_tiles, H // HK, GP, BQ
    )
    maximum = local_max(scores)
    maximum = _maximum(maximum, maximum.shuffle_xor(16, 64))
    maximum = _maximum(maximum, maximum.shuffle_xor(32, 64))
    maximum = _maximum(maximum * scale, fx.Float32(-1.0e30)) + 1.0
    scores = center(scores, scale, maximum)
    row_sum = fx.Float32(0.0)
    _stage_end()
    offsets = dma_offsets(rows1, 4)
    current_offsets = offsets
    for packet in fx.range_constexpr(16):
        dma(
            gk,
            storage,
            wave,
            lane,
            offsets,
            _min(fx.Int32(1), last),
            compact_len,
            HK,
            packet,
            False,
            extent,
            4,
        )
    _wait(vmcnt=0, lgkmcnt=0)
    _stage_end()
    _stage_end()

    @flyc.jit
    def phase(
        previous,
        maximum,
        row_sum,
        o0,
        o1,
        previous_offsets,
        current_offsets,
        next_rows,
        t,
        MASKED: fx.Constexpr[bool],
    ):
        previous, maximum, row_sum = (
            fx.Vector(previous),
            fx.Float32(maximum),
            fx.Float32(row_sum),
        )
        o0, o1, t = fx.Vector(o0), fx.Vector(o1), fx.Int32(t)
        previous_offsets = fx.Vector(previous_offsets)
        current_offsets, next_rows = fx.Vector(current_offsets), fx.Int32(next_rows)
        future_rows = source_rows(table, _min(t + 2, last), count, wave, HK)
        parts = []
        for n in fx.range_constexpr(2):
            for step in fx.range_constexpr(8):
                parts.append(
                    dma(
                        gv,
                        storage,
                        wave,
                        lane,
                        previous_offsets,
                        t - 1,
                        compact_len,
                        HK,
                        n * 8 + step,
                        True,
                        extent,
                        4,
                        read_address=kr[n * 2 + step % 2],
                        read_immediate=(step // 2) * 128,
                    )
                )
        words = fx.Vector.from_elements(
            [part[i] for part in parts for i in range(4)], fx.Int32
        )
        fragment = fx.make_rmem_tensor(
            fx.make_layout((4, 16, 2), (1, 4, 64)), fx.BFloat16
        )
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
            _wait(lgkmcnt=8)
        _stage_end()
        _wait(lgkmcnt=0)
        rocdl.sched_barrier(0)
        hi = qk(q, k)
        total, probabilities = local_sum(previous), pack(previous)
        offsets = dma_offsets(next_rows, 4)
        _schedule(32, 3, 2)
        _stage_end()
        sums = cross(total, cross_addresses)
        parts = []
        for block in fx.range_constexpr(2):
            for r in fx.range_constexpr(8):
                parts.append(
                    dma(
                        gk,
                        storage,
                        wave,
                        lane,
                        offsets,
                        _min(t + 1, last),
                        compact_len,
                        HK,
                        block * 8 + r,
                        False,
                        extent,
                        4,
                        read_address=vr,
                        read_immediate=block * 16384 + r * 512,
                    )
                )
        v = fx.Vector.from_elements(
            [part[i] for part in parts for i in range(4)], fx.Int32
        )
        prepared = v_operands(v, 0, True)
        _stage_end()
        current = _join(lo, hi)
        if fx.const_expr(MASKED):
            current = mask(
                current, masks, t, query, qvalid, common_tiles, H // HK, GP, BQ
            )
        o0, row_sum, candidate = pv(
            probabilities,
            v,
            o0,
            True,
            prepared,
            summary_args=(current, total, sums, row_sum),
        )
        _schedule(32, 3, 3)
        _stage_end()
        maxima = cross(candidate, cross_addresses)
        _wait(vmcnt=0)
        v = read_v(vr, 1)
        prepared = v_operands(v, 0, True)
        _stage_end()
        o1, current, new_max, ballot = pv(
            probabilities,
            v,
            o1,
            True,
            prepared,
            (current, candidate, maxima, scale, maximum),
        )
        _schedule(32, 3, 4)
        o0, o1, row_sum = rescale(o0, o1, row_sum, maximum, new_max, ballot)
        new_max = advance_max(maximum, new_max)
        _stage_end()
        return current, new_max, row_sum, o0, o1, current_offsets, offsets, future_rows

    common_end = (common_tiles > 1).select(common_tiles, fx.Int32(1))
    for t in range(fx.Int32(1), common_end - 3, fx.Int32(4)):
        for j in fx.range_constexpr(4):
            scores, maximum, row_sum, o0, o1, v_offsets, current_offsets, rows2 = phase(
                scores,
                maximum,
                row_sum,
                o0,
                o1,
                v_offsets,
                current_offsets,
                rows2,
                t + j,
                False,
            )
    remainder = ((common_end - 1) & -4) + 1
    for t in range(remainder, common_end, fx.Int32(1)):
        scores, maximum, row_sum, o0, o1, v_offsets, current_offsets, rows2 = phase(
            scores,
            maximum,
            row_sum,
            o0,
            o1,
            v_offsets,
            current_offsets,
            rows2,
            t,
            False,
        )
    for t in range(common_end, tiles, fx.Int32(1)):
        scores, maximum, row_sum, o0, o1, v_offsets, current_offsets, rows2 = phase(
            scores, maximum, row_sum, o0, o1, v_offsets, current_offsets, rows2, t, True
        )
    offsets = fx.Vector(v_offsets)
    for packet in fx.range_constexpr(16):
        dma(
            gv,
            storage,
            wave,
            lane,
            offsets,
            last,
            compact_len,
            HK,
            packet,
            True,
            extent,
            4,
            True,
        )
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
    output = rocdl.make_buffer_tensor(
        fx.make_view(
            fx.get_iter(O) + fx.Int64(q0) * (H * D) + hkv * (H // HK) * D,
            fx.make_layout(NQ * H * D, 1),
        ),
        num_records_bytes=qextent,
    )
    output_fn(
        o0,
        o1,
        inv,
        output,
        shared,
        _pin_i32(tid),
        H,
        H // HK,
        GP,
        BQ,
        qvalid,
        qextent,
        row_offset,
    )


@flyc.kernel(known_block_size=[THREADS, 1, 1])
def _kernel(
    Q: fx.Tensor,
    K: fx.Tensor,
    V: fx.Tensor,
    O: fx.Tensor,
    META: fx.Tensor,
    BLOCKS: fx.Tensor,
    MEMBERS: fx.Tensor,
    COUNTS: fx.Tensor,
    ACTIVE: fx.Tensor,
    H: fx.Constexpr[int],
    HK: fx.Constexpr[int],
    NQ: fx.Constexpr[int],
    NK: fx.Constexpr[int],
    CAP: fx.Constexpr[int],
    BQ: fx.Constexpr[int],
    GP: fx.Constexpr[int],
    TASKS: fx.Constexpr[int],
    GRID: fx.Constexpr[int],
    SCALE: fx.Constexpr[float],
    TAIL_BOUNDS: fx.Constexpr[bool],
):
    body = _body
    storage = (
        fx.SharedAllocator()
        .allocate(fx.Array[fx.Int8, LDS_BYTES, 16])
        .peek()
        .view(fx.make_layout(LDS_BYTES, 1))
    )
    work = fx.Int32(gpu.block_id("x"))
    while work < TASKS:
        tiles = TASKS // HK
        mapped = (work % tiles) * HK + work // tiles
        tile = mapped // (HK * ((BQ * GP + 127) // 128))
        if _uniform(ACTIVE[tile]) != 0:
            group = _uniform(fx.Int32(gpu.thread_id("x")) >> 8)
            if group != 0:
                body(
                    Q,
                    K,
                    V,
                    O,
                    META,
                    BLOCKS,
                    MEMBERS,
                    COUNTS,
                    storage,
                    mapped,
                    H,
                    HK,
                    NQ,
                    NK,
                    CAP,
                    BQ,
                    GP,
                    SCALE,
                    TAIL_BOUNDS,
                    True,
                )
            else:
                body(
                    Q,
                    K,
                    V,
                    O,
                    META,
                    BLOCKS,
                    MEMBERS,
                    COUNTS,
                    storage,
                    mapped,
                    H,
                    HK,
                    NQ,
                    NK,
                    CAP,
                    BQ,
                    GP,
                    SCALE,
                    TAIL_BOUNDS,
                    False,
                )
        work = work + GRID


@flyc.jit
def _launch(
    Q: fx.Tensor,
    K: fx.Tensor,
    V: fx.Tensor,
    O: fx.Tensor,
    META: fx.Tensor,
    BLOCKS: fx.Tensor,
    MEMBERS: fx.Tensor,
    COUNTS: fx.Tensor,
    ACTIVE: fx.Tensor,
    H: fx.Constexpr[int],
    HK: fx.Constexpr[int],
    NQ: fx.Constexpr[int],
    NK: fx.Constexpr[int],
    CAP: fx.Constexpr[int],
    BQ: fx.Constexpr[int],
    GP: fx.Constexpr[int],
    TASKS: fx.Constexpr[int],
    GRID: fx.Constexpr[int],
    SCALE: fx.Constexpr[float],
    TAIL_BOUNDS: fx.Constexpr[bool],
    stream: fx.Stream,
):
    _kernel(
        Q,
        K,
        V,
        O,
        META,
        BLOCKS,
        MEMBERS,
        COUNTS,
        ACTIVE,
        H,
        HK,
        NQ,
        NK,
        CAP,
        BQ,
        GP,
        TASKS,
        GRID,
        SCALE,
        TAIL_BOUNDS,
        value_attrs={
            "rocdl.waves_per_eu": 2,
            "passthrough": [["target-features", "-packed-fp32-ops"]],
        },
    ).launch(grid=(GRID, 1, 1), block=(THREADS, 1, 1), stream=stream)


_COMPILED = {}


def run(*, inputs, plan, out):
    if plan.num_tiles == 0:
        return
    stream = torch.cuda.current_stream(inputs.q.device)
    args = (
        inputs.q.view(-1),
        inputs.k.view(-1),
        inputs.v.view(-1),
        out.view(-1),
        plan.metadata.view(-1),
        plan.blocks.view(-1),
        plan.score_masks.view(-1),
        plan.counts.view(-1),
        plan.active,
        inputs.q.shape[1],
        inputs.k.shape[1],
        inputs.q.shape[0],
        inputs.k.shape[0],
        plan.block_capacity,
        plan.query_tile,
        plan.group_padded,
        plan.num_tiles
        * inputs.k.shape[1]
        * ((plan.query_tile * plan.group_padded + 127) // 128),
        plan.grid,
        inputs.scale,
        any(
            (q + p) % 4 for q, p in zip(inputs.spec.query_lens, inputs.spec.prefix_lens)
        ),
        stream,
    )
    key = (
        inputs.q.device,
        tuple(
            (
                (a.dtype, tuple(a.shape), tuple(a.stride()))
                if isinstance(a, torch.Tensor)
                else ("stream",) if isinstance(a, torch.cuda.Stream) else a
            )
            for a in args
        ),
    )
    with torch.cuda.device(inputs.q.device), torch.cuda.stream(stream):
        compiled = _COMPILED.get(key)
        if compiled is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Warm the QSA specialization before graph capture")
            _COMPILED[key] = flyc.compile(_launch, *args)
        else:
            compiled(*args)
