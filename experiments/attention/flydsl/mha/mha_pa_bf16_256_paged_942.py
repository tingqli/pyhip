"""Validated gfx942 D256 paged path: register Q, early K/V DMA and late S4 wait."""

import math

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import gpu, rocdl

if __package__:
    from . import _common as base
    from ._common import (
        _advance_max, _buffer, _join, _maximum, _min, _pack_bf16,
        _page_ready, _pin, _pin_i32, _pin_s64, _read_address,
        _rescale, _schedule, _stage_end, _uniform, _wait,
    )
else:
    import _common as base
    from _common import (
        _advance_max, _buffer, _join, _maximum, _min, _pack_bf16,
        _page_ready, _pin, _pin_i32, _pin_s64, _read_address,
        _rescale, _schedule, _stage_end, _uniform, _wait,
    )

BM, BN, D, THREADS = base.BM, base.BN, base.D, base.THREADS
K_PITCH, K_BYTES, LDS_BYTES = base.K_PITCH, base.K_BYTES, base.LDS_BYTES
LOG2E = base.LOG2E


def _dma4(resource, storage, destination, voffset, soffset=0):
    # destination is wave-uniform m0; hardware writes DWORD at m0+4*lane.
    rocdl.raw_ptr_buffer_load_lds(
        resource, fx.to_llvm_ptr(fx.get_iter(storage) + destination),
        fx.Int32(4).ir_value(), fx.Int32(voffset).ir_value(), fx.Int32(soffset).ir_value(),
        fx.Int32(0).ir_value(), fx.Int32(0).ir_value(),
    )


def _k_dma_offset(tid, page_size):
    wave, lane = tid >> 6, tid & 63
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


def _v_dma_soffset(packet, page_size):
    return (packet % 8 if page_size == 32 else packet) * 2048


def _v_resources(base_address, page, second_page, tile, hk, page_size):
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


def _dma_v(resources, storage, wave, voffset, page_size, packet):
    resource = resources[1] if page_size == 32 and packet >= 8 else resources[0]
    destination = fx.Int32(llvm.inline_asm(fx.Int32.ir_type,
        [fx.Int32(wave * 256).ir_value()],
        f"s_add_u32 $0, $1, {K_BYTES + packet * 2048}", "=s,s,~{scc}", has_side_effects=True))
    offset = fx.Int32(llvm.inline_asm(fx.Int32.ir_type, [],
        f"s_mov_b32 $0, {_v_dma_soffset(packet, page_size)}", "=s", has_side_effects=True))
    _dma4(resource, storage, destination, voffset, offset)


def _read_k_vdma(address, half, resources, storage, wave, voffset, page_size):
    parts = []
    for n in range(2):
        for k in range(8):
            parts.append(_read_address(address[n], half * 512 + k * (4 * K_PITCH)))
            index = n * 8 + k
            _dma_v(resources, storage, wave, voffset, page_size, index)
    words = fx.Vector.from_elements([p[i] for p in parts for i in range(4)], fx.Int32)
    fragment = fx.make_rmem_tensor(fx.make_layout((4, 16, 2), (1, 4, 64)), fx.BFloat16)
    fragment.store(words.bitcast(fx.BFloat16))
    return fx.make_view(fx.get_iter(fragment), fx.make_layout((4, 2, 16), (1, 64, 4)))


def _read_v_dma(address, half, resource, storage, wave, voffset, page, second_page,
                tile, hk, page_size):
    parts = []
    for n in range(8):
        for k in range(2):
            parts.append(_read_address(address, half * 2048 + n * 256 + k * 16384))
            index = n * 2 + k
            _dma_k(resource, storage, wave, voffset, page, second_page, tile, hk, page_size, index)
    words = fx.Vector.from_elements([p[i] for p in parts for i in range(4)], fx.Int32)
    frag = fx.make_rmem_tensor(fx.make_layout((4, 4, 8), (1, 4, 16)), fx.BFloat16)
    frag.store(words.bitcast(fx.BFloat16))
    return frag


@flyc.jit
def _body(Q, K, V, O, LSE, QS, KS, VS, table, storage, q0, q_len, kv_len, head, qb,
          H: fx.Constexpr[int], HK: fx.Constexpr[int], NP: fx.Constexpr[int], PAGE: fx.Constexpr[int],
          CAUSAL: fx.Constexpr[bool], PER_TOKEN: fx.Constexpr[bool], WITH_LSE: fx.Constexpr[bool],
          SCALE: fx.Constexpr[float], STAGGER: fx.Constexpr[bool]):
    read_k, read_v, qk, pv = base._read_k, base._read_v, base._qk, base._pv
    local_sum, local_max, cross = base._sum, base._max, base._cross
    center, exps, pack, mask = base._center, base._exps, base._pack, base._mask
    rescale, advance_max = _rescale, _advance_max
    pages_for_tile, dma_k, read_v_dma = base._pages, _dma_k, _read_v_dma
    v_resources, dma_v, read_k_vdma = _v_resources, _dma_v, _read_k_vdma
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
    kd = _pin_i32(_k_dma_offset(tid, PAGE))
    vr = _pin_i32(shared + K_BYTES + (lane >> 4) * 4096 + (lane & 15) * 16)
    vd = _pin_i32(_v_dma_offset(tid))
    vbase = _pin_s64(fx.Int64(fx.ptrtoint(fx.get_iter(V))) + fx.Int64(hkv) * (PAGE * D * 2))
    cross_addresses = tuple(_pin_i32((lane ^ offset) * 4) for offset in (16, 32, 48))
    for packet in fx.range_constexpr(16):
        dma_k(gk, storage, wave, kd, page0, page01, fx.Int32(0), HK, PAGE, packet)
    _wait(vmcnt=0)
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
    for packet in fx.range_constexpr(16):
        dma_k(gk, storage, wave, kd, page1, page11, _min(fx.Int32(1), last), HK, PAGE, packet)
    _wait(vmcnt=0)
    _wait(lgkmcnt=0)
    _stage_end()
    _stage_end()

    @flyc.jit
    def phase(previous, maximum, row_sum, o0, o1, previous_page, previous_page1,
              current_page, current_page1, next_page, next_page1, t):
        previous, maximum, row_sum = fx.Vector(previous), fx.Float32(maximum), fx.Float32(row_sum)
        o0, o1, t = fx.Vector(o0), fx.Vector(o1), fx.Int32(t)
        request, request1 = pages_for_tile(table, t + 2, last, kv_len, PAGE)
        resources = v_resources(vbase, previous_page, previous_page1, t - 1, HK, PAGE)
        k = read_k_vdma(kr, 0, resources, storage, wave, vd, PAGE)
        future_page = _page_ready(request)
        if fx.const_expr(PAGE != 32):
            future_page1 = future_page
        else:
            future_page1 = _page_ready(request1)
        _stage_end()
        lo = qk(q, k)
        previous = exps(previous)
        _schedule(32, 1, 1, True)
        _stage_end()
        k = read_k(kr, 1)
        _wait(vmcnt=0)
        _wait(lgkmcnt=0)
        _stage_end()
        hi = qk(q, k)
        total = local_sum(previous)
        p = pack(previous)
        _schedule(32, 3, 2)
        _stage_end()
        sums = cross(total, cross_addresses)
        v = read_v_dma(vr, 0, gk, storage, wave, kd, next_page, next_page1,
                      _min(t + 1, last), HK, PAGE)
        _stage_end()
        # The S4 rendezvous does not release a V overwrite; retire readers
        # immediately before their first PV/sum consumers.
        _wait(lgkmcnt=0)
        rocdl.sched_barrier(0)
        o0 = pv(p, v, o0)
        row_sum = row_sum + ((total + sums[0]) + (sums[1] + sums[2]))
        current = mask(_join(lo, hi), t, row, q_len, kv_len, CAUSAL)
        candidate = local_max(current)
        _schedule(32, 2, 3)
        _stage_end()
        maxima = cross(candidate, cross_addresses)
        _wait(vmcnt=0)
        v = read_v(vr, 1)
        _wait(lgkmcnt=0)
        _stage_end()
        o1 = pv(p, v, o1)
        candidate = _maximum(_maximum(candidate, maxima[0]), _maximum(maxima[1], maxima[2])) * scale
        ballot = fx.Int64(rocdl.ballot(fx.Int64.ir_type, (candidate > maximum + 7.0).ir_value()))
        new_max = (candidate > maximum + 7.0).select(candidate + 1.0, maximum)
        current = center(current, scale, new_max)
        _schedule(32, 3, 4)
        o0, o1, row_sum = rescale(o0, o1, row_sum, maximum, new_max, ballot)
        new_max = advance_max(maximum, new_max)
        _stage_end()
        return current, new_max, row_sum, o0, o1, current_page, current_page1, next_page, next_page1, future_page, future_page1

    previous_page, previous_page1 = page0, page01
    for t in range(fx.Int32(1), tiles, fx.Int32(1)):
        scores, maximum, row_sum, o0, o1, previous_page, previous_page1, page1, page11, page2, page21 = phase(
            scores, maximum, row_sum, o0, o1, previous_page, previous_page1, page1, page11, page2, page21, t)

    resources = v_resources(vbase, previous_page, previous_page1, last, HK, PAGE)
    for packet in fx.range_constexpr(16):
        dma_v(resources, storage, wave, vd, PAGE, packet)
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