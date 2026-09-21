"""gfx942 BF16 paged attention with alternating memory and compute stages.

BM256/BN64, eight wave64s staggered by one stage. The V128/no-LSE hot path
keeps VMEM/DS in S0/2/4/6 and MFMA/VALU in S1/3/5/7, following gfx950's
stage organization. Lane addresses are prepared before memory stages;
V uses a full 64-bit GLOBAL scalar base, never a buffer descriptor.
One K slot and two V slots use 49,920/58,240 bytes for D128/D192. Softmax
lags QK by one tile; cross32 reductions issue in memory stages and retire
with the operand reads. Optional LSE/V192 retain the checked wide path.
Ordinary and persistent grids share this body and public tensor layouts.
"""

import functools
import math

import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu, rocdl
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm


BM, BN, THREADS = 256, 64, 512
PADDED_CHUNK = 1040
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
    a, b = fx.Vector(a), fx.Vector(b)
    return fx.Vector.from_elements([a[i] for i in range(a.numel)] + [b[i] for i in range(b.numel)], fx.Float32)


def _exp(value):
    return fx.Float32(llvm.call_intrinsic(fx.Float32.ir_type, "llvm.amdgcn.exp2.f32",
                                         [fx.Float32(value).ir_value()], [], []))


def _maximum(a, b):
    return fx.Float32(llvm.inline_asm(fx.Float32.ir_type, [fx.Float32(a).ir_value(), fx.Float32(b).ir_value()],
                                    "v_max_f32 $0, $1, $2", "=v,v,v", has_side_effects=False))


def _max3(a, b, c):
    return fx.Float32(llvm.inline_asm(fx.Float32.ir_type, [a.ir_value(), b.ir_value(), c.ir_value()],
                                    "v_max3_f32 $0, $1, $2, $3", "=v,v,v,v", has_side_effects=False))


def _row_max_local(values):
    p = [_max3(values[i], values[i + 1], values[i + 2]) for i in range(0, 30, 3)]
    a = _max3(_max3(p[0], p[1], p[2]), _max3(p[3], p[4], p[5]), _max3(p[6], p[7], p[8]))
    b = _max3(p[9], values[30], values[31])
    return _max3(a, b, fx.Float32(-1.0e30))


def _row_max(values):
    value = _row_max_local(values)
    return _maximum(value, value.shuffle_xor(32, 64))


def _row_sum_local(values):
    partials = [values[i] + values[i + 1] for i in range(0, 32, 2)]
    for width in (8, 4, 2, 1):
        partials = [partials[2 * i] + partials[2 * i + 1] for i in range(width)]
    return partials[0]


def _row_sum(values):
    value = _row_sum_local(values)
    return value + value.shuffle_xor(32, 64)


def _cross32_async(value, address):
    # Issue in a memory stage; consume only after its explicit LGKM wait and
    # compiler/CTA boundary. This shuffle does not allocate LDS storage.
    return fx.Float32(llvm.inline_asm(fx.Float32.ir_type, [address.ir_value(), value.ir_value()],
        "ds_bpermute_b32 $0, $1, $2", "=v,v,v,~{memory}", has_side_effects=True))


def _pin_s64(value):
    return fx.Int64(llvm.inline_asm(fx.Int64.ir_type, [fx.Int64(value).ir_value()],
                                   "", "=s,0", has_side_effects=True))


def _advance_max(old, new):
    return fx.Float32(llvm.inline_asm(fx.Float32.ir_type, [old.ir_value(), new.ir_value()],
                                    "v_mov_b32 $0, $2", "=v,0,v", has_side_effects=True))


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


def _pack_bf16(values):
    # gfx942 has no native packed FP32 -> BF16 instruction. Software RNE,
    # as in the FP8 kernel's output path, is used for O.
    bits = fx.Vector(values).bitcast(fx.Uint32)
    rounded = bits + fx.Uint32(0x7FFF) + ((bits >> 16) & fx.Uint32(1))
    words = [(rounded[i] >> 16) | (rounded[i + 1] & fx.Uint32(0xFFFF0000))
             for i in range(0, values.numel, 2)]
    return fx.Vector.from_elements(words, fx.Uint32)


def _pack_probability(values):
    # P uses the established BF16 round-half-up contract. Packing each pair
    # directly avoids materializing 32 RNE tie bits during the QK-high stage.
    bits = fx.Vector(values).bitcast(fx.Uint32) + fx.Uint32(0x8000)
    selector = fx.Int32(0x07060302)
    words = [fx.Int32(llvm.inline_asm(
        fx.Int32.ir_type, [bits[i + 1].ir_value(), bits[i].ir_value(), selector.ir_value()],
        "v_perm_b32 $0, $1, $2, $3", "=v,v,v,s", has_side_effects=True,
    )) for i in range(0, 32, 2)]
    return fx.Vector.from_elements(words, fx.Int32).bitcast(fx.BFloat16)


def _stage_end():
    # Compiler fences and CTA rendezvous, not a memory-counter wait.
    rocdl.sched_barrier(0)
    rocdl.s_barrier()
    rocdl.sched_barrier(0)


def _wait(*, vmcnt=63, expcnt=7, lgkmcnt=63):
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
    # A tied '=s,0' result lets register allocation copy the still-pending
    # SMEM result BEFORE the wait when page32/page64 aliases do not coalesce.
    # Keep the first read of that register inside the assembly, AFTER waiting.
    return fx.Int32(llvm.inline_asm(fx.Int32.ir_type, [value.ir_value()],
                                  "s_waitcnt lgkmcnt(0)\ns_mov_b32 $0, $1",
                                  "=s,s", has_side_effects=True))


def _prefetch_tile(table, tile, last, last_page, page_size):
    # BN64 spans two physical pages at page32; page128 contains two tiles.
    # Clamp every speculative SMEM request, including the second page.
    first = _min(tile, last) * BN // page_size
    page0 = _prefetch_page(table, first)
    page1 = _prefetch_page(table, _min(first + 1, last_page)) if page_size == 32 else page0
    return page0, page1


def _buffer(tensor, size_bytes):
    address = fx.Int64(fx.ptrtoint(fx.get_iter(tensor)))
    pointer = llvm.inttoptr(ir.Type.parse("!llvm.ptr"), address.ir_value())
    return rocdl.make_buffer_rsrc(ir.Type.parse("!llvm.ptr<8>"), pointer,
                                 fx.Int16(0).ir_value(), fx.Int64(size_bytes).ir_value(),
                                 fx.Int32(0x27000).ir_value())


def _buffer_words(resource, voffset, soffset=0):
    return fx.Vector(rocdl.raw_ptr_buffer_load(ir.VectorType.get([4], fx.Int32.ir_type), resource,
                                              fx.Int32(voffset).ir_value(), fx.Int32(soffset).ir_value(),
                                              fx.Int32(0).ir_value()))


def _global_words(tensor, offset):
    # A genuine 64-bit GLOBAL pointer; never replace V with a raw descriptor.
    pointer = fx.get_iter(tensor) + fx.Int64(offset)
    source = fx.make_view(pointer, fx.make_layout(8, 1))
    fragment = fx.make_rmem_tensor(8, fx.BFloat16)
    fx.copy(fx.make_copy_atom(fx.UniversalCopy128b(), fx.BFloat16), source, fragment)
    return fragment.load().bitcast(fx.Int32)


def _lds_words(storage, base, immediate=0):
    address = fx.Int32(fx.ptrtoint(fx.get_iter(storage) + base))
    return fx.Vector(llvm.inline_asm(ir.VectorType.get([4], fx.Int32.ir_type), [address.ir_value()],
        f"ds_read_b128 $0, $1 offset:{immediate}", "=v,v,~{memory}", has_side_effects=True))


def _read_address(address, immediate=0):
    return fx.Vector(llvm.inline_asm(ir.VectorType.get([4], fx.Int32.ir_type), [address.ir_value()],
        f"ds_read_b128 $0, $1 offset:{immediate}", "=v,v,~{memory}", has_side_effects=True))


def _write_address(address, words, rounds, immediate=0):
    for i in range(rounds):
        part = fx.Vector.from_elements([words[i * 4 + j] for j in range(4)], fx.Int32)
        llvm.inline_asm(ir.Type.parse("!llvm.void"), [address.ir_value(), part.ir_value()],
            f"ds_write_b128 $0, $1 offset:{immediate + i * 8320}", "v,v,~{memory}", has_side_effects=True)


def _k_address(lane_offset, page0, page1, tile, half_page, hk, dq, page_size):
    page = half_page.select(page1, page0) if page_size == 32 else page0
    return _pin_i32(lane_offset + page * (hk * page_size * dq * 2) + (tile * BN % page_size) * 16)


def _load_k_address(resource, address, dq, page_size):
    # Per-chunk SOFFSET is scalar; even page32 keeps its divergent physical
    # page in the precomputed VOFFSET, never in a scalar waterfall.
    parts = [_buffer_words(resource, address, i * page_size * 128) for i in range(dq // 64)]
    return fx.Vector.from_elements([p[j] for p in parts for j in range(4)], fx.Int32)


def _v_addresses(base, page0, page1, tile, hk, page_size):
    first = base + fx.Int64(page0) * (hk * page_size * 256) + fx.Int64(tile * BN % page_size) * 256
    second = base + fx.Int64(page1) * (hk * page_size * 256) if page_size == 32 else first + 8192
    return _pin_s64(first), _pin_s64(second)


def _load_v_address(first, second, lane_offset):
    # Full 64-bit SGPR address plus lane-relative VGPR byte offset. This is
    # GLOBAL, not BUFFER, and does not truncate the tensor's GPU base.
    parts = [fx.Vector(llvm.inline_asm(ir.VectorType.get([4], fx.Int32.ir_type),
        [lane_offset.ir_value(), address.ir_value()], "global_load_dwordx4 $0, $1, $2",
        "=v,v,s,~{memory}", has_side_effects=True)) for address in (first, second)]
    return fx.Vector.from_elements([p[j] for p in parts for j in range(4)], fx.Int32)


def _k_fragment_address(address, half, dq):
    parts = [_read_address(address, half * 512 + d * 2080) for d in range(dq // 16)]
    words = fx.Vector.from_elements([p[i] for p in parts for i in range(4)], fx.Int32)
    frag = _mma().make_fragment_A(fx.make_rmem_tensor(fx.make_layout((32, dq), (1, 32)), fx.BFloat16))
    frag.store(words.bitcast(fx.BFloat16))
    return frag


def _v_fragment_address(address, half, slot=0):
    parts = [_read_address(address, slot * 16640 + half * 1040 + n * 512 + k * 4160)
             for n in range(2) for k in range(4)]
    words = fx.Vector.from_elements([p[i] for p in parts for i in range(4)], fx.Int32)
    frag = fx.make_rmem_tensor(fx.make_layout((8, 4, 2), (1, 8, 32)), fx.BFloat16)
    frag.store(words.bitcast(fx.BFloat16))
    return frag


def _load_k(resource, tid, page0, page1, tile, hk, dq, page_size):
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


def _load_v(tensor, tid, page0, page1, tile, hkv, hk, dv, page_size, bounded=False):
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
        parts.append(_global_words(tensor, offset))
    return fx.Vector.from_elements([p[j] for p in parts for j in range(4)], fx.Int32)


def _cooperative_store(storage, tid, words, slot_offset, rounds, key=False, bounded=False):
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


def _q_fragment(resource, row_offset, lane, dq):
    parts = [_buffer_words(resource, row_offset + (lane >> 5) * 16 + d * 32) for d in range(dq // 16)]
    words = fx.Vector.from_elements([p[i] for p in parts for i in range(4)], fx.Int32)
    frag = _mma().make_fragment_B(fx.make_rmem_tensor(fx.make_layout((BM, dq), (1, BM)), fx.BFloat16))
    frag.store(words.bitcast(fx.BFloat16))
    return frag


def _k_fragment(storage, base, half, dq, dv, full=False):
    width = 64 if (dq == 192 or dv == 192) and not full else dq
    parts = [_lds_words(storage, base, half * 512 + d * 2080) for d in range(width // 16)]
    words = fx.Vector.from_elements([p[i] for p in parts for i in range(4)], fx.Int32)
    frag = _mma().make_fragment_A(fx.make_rmem_tensor(fx.make_layout((32, width), (1, 32)), fx.BFloat16))
    frag.store(words.bitcast(fx.BFloat16))
    return frag


def _v_fragment(storage, base, slot, half, dq, dv):
    width = dv // 2
    parts = []
    for n in range(width // 32):
        for k in range(BN // 16):
            byte = (2 * k * dv + half * width + n * 32) * 16
            parts.append(_lds_words(storage, ((dq + slot * dv) // 8) * PADDED_CHUNK + base,
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
                k = _k_fragment(storage, base + chunk * 8320, half, 64, dv)
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
def _rescale(o0, o1, row_sum, old_max, new_max, ballot):
    o0, o1, row_sum = fx.Vector(o0), fx.Vector(o1), fx.Float32(row_sum)
    if ballot != fx.Int64(0):
        correction = _exp(fx.Float32(old_max) - fx.Float32(new_max))
        o0, o1, row_sum = o0 * correction, o1 * correction, row_sum * correction
    return o0, o1, row_sum


@flyc.jit
def _body(Q, K, V, O, LSE, QS, KS, VS, table, storage, q0, q_len, kv_len, head, qb,
          H: fx.Constexpr[int], HK: fx.Constexpr[int], NP: fx.Constexpr[int], DQ: fx.Constexpr[int],
          DV: fx.Constexpr[int], PAGE: fx.Constexpr[int], CAUSAL: fx.Constexpr[bool],
          PER_TOKEN: fx.Constexpr[bool], WITH_LSE: fx.Constexpr[bool],
          SCALE: fx.Constexpr[float], STAGGER: fx.Constexpr[bool]):
    # Explicit closure dependency for the helper used only by nested phase.
    rescale = _rescale
    bounded = DQ == 192 or DV == 192 or WITH_LSE
    v_ring = DV == 128 and (DQ == 128 or not WITH_LSE)
    full_k = DQ == 192 and DV == 128 and not WITH_LSE
    separate = DV == 128 and not WITH_LSE
    # Explicit dependencies for helpers used by the nested phase after AST
    # rewriting; keep the native cache sensitive to every scheduling edit.
    read_k, read_v, write_lds = _k_fragment_address, _v_fragment_address, _write_address
    k_address, load_k_address = _k_address, _load_k_address
    v_addresses, load_v_address = _v_addresses, _load_v_address
    sum_local, max_local, cross_async = _row_sum_local, _row_max_local, _cross32_async
    advance_max = _advance_max
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
    q = _q_fragment(gq, (wave * 32 + (lane & 31)) * (H * DQ * 2), lane, DQ)
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
    page00, page01 = _prefetch_tile(table, fx.Int32(0), last, last_page, PAGE)
    page10, page11 = _prefetch_tile(table, fx.Int32(1), last, last_page, PAGE)
    page20, page21 = _prefetch_tile(table, fx.Int32(2), last, last_page, PAGE)
    page00, page01 = _page_ready(page00), _page_ready(page01)
    page10, page11 = _page_ready(page10), _page_ready(page11)
    page20, page21 = _page_ready(page20), _page_ready(page21)
    # BF16 P carries eight consecutive keys per b128, not FP8's sixteen.
    k_row = (lane & 3) | ((lane & 4) << 1) | ((lane & 8) >> 1) | (lane & 16)
    k_base = (lane >> 5) * PADDED_CHUNK + ((k_row * 16) ^ ((k_row & 16) << 2))
    v_base = (lane >> 5) * (DV // 64) * PADDED_CHUNK + (lane & 31) * 16
    k_bytes = (DQ // 8) * PADDED_CHUNK
    kr_address = kw_address = vr_address = vw_address = k_lane = v_lane = cross_address = fx.Int32(0)
    v_global_base = fx.Int64(0)
    half_page = (lane & 32) != 0
    if fx.const_expr(separate):
        shared = fx.Int32(fx.ptrtoint(fx.get_iter(storage)))
        within = (tid & 63) * 16
        kr_address = _pin_i32(shared + k_base)
        kw_address = _pin_i32(shared + (tid >> 6) * PADDED_CHUNK + (within ^ ((within & 256) >> 2)))
        vr_address = _pin_i32(shared + k_bytes + v_base)
        vw_address = _pin_i32(shared + k_bytes + (tid >> 6) * PADDED_CHUNK + within)
        k_lane = _pin_i32((tid >> 6) * (PAGE * 16) + (tid & (min(PAGE, BN) - 1)) * 16)
        v_lane = _pin_i32(tid * 16)
        cross_address = _pin_i32((lane ^ 32) * 4)
        v_global_base = _pin_s64(fx.Int64(fx.ptrtoint(fx.get_iter(V))) + fx.Int64(hkv) * (PAGE * DV * 2))
    v_pending = fx.Vector.filled(DV // 16, 0, fx.Int32)
    if fx.const_expr(separate):
        k0_address = k_address(k_lane, page00, page01, fx.Int32(0), half_page, HK, DQ, PAGE)
        k0 = load_k_address(gk, k0_address, DQ, PAGE)
        v0, v1 = v_addresses(v_global_base, page00, page01, fx.Int32(0), HK, PAGE)
        v_pending = load_v_address(v0, v1, v_lane)
    else:
        k0 = _load_k(gk, tid, page00, page01, fx.Int32(0), HK, DQ, PAGE)
        v_pending = _load_v(V, tid, page00, page01, fx.Int32(0), hkv, HK, DV, PAGE, bounded)
    _wait(vmcnt=0)
    if fx.const_expr(separate):
        write_lds(kw_address, k0, DQ // 64)
        write_lds(vw_address, v_pending, 2)
    else:
        _cooperative_store(storage, tid, k0, 0, DQ // 64, True, bounded)
        _cooperative_store(storage, tid, v_pending, k_bytes, DV // 64, bounded=bounded)
    _wait(vmcnt=0, lgkmcnt=0)
    _stage_end()
    if fx.const_expr(STAGGER):
        _stage_end()

    if fx.const_expr(separate and DQ == 192):
        k = read_k(kr_address, 0, DQ)
    else:
        k = _k_fragment(storage, k_base, 0, DQ, DV, full_k)
    _wait(lgkmcnt=0)
    _stage_end()
    lo = _qk(q, k, storage, k_base, 0, DQ, DV, full_k)
    o0 = _pin(fx.Vector.filled(DV // 4, 0.0, fx.Float32))
    o1 = _pin(fx.Vector.filled(DV // 4, 0.0, fx.Float32))
    _schedule(DQ // 8, 3, 5)
    _stage_end()
    k1 = fx.Vector.filled(DQ // 16, 0, fx.Int32)
    if fx.const_expr(separate and DQ == 192):
        k = read_k(kr_address, 1, DQ)
        k1_address = k_address(k_lane, page10, page11, _min(fx.Int32(1), last), half_page, HK, DQ, PAGE)
        k1 = load_k_address(gk, k1_address, DQ, PAGE)
    else:
        k = _k_fragment(storage, k_base, 1, DQ, DV, full_k)
    _wait(lgkmcnt=0)
    _stage_end()
    hi = _qk(q, k, storage, k_base, 1, DQ, DV, full_k)
    scores = _mask(_join(lo, hi), fx.Int32(0), row, q_len, kv_len, CAUSAL)
    maximum = _maximum(_row_max(scores) * scale, fx.Float32(-1.0e30)) + 1.0
    scores = _center(scores, scale, maximum, 0, 32)
    row_sum = fx.Float32(0.0)
    _stage_end()
    if fx.const_expr((DQ == 192 or DV == 192) and not separate):
        # Retain the wide-path prologue rendezvous. Streaming variants read
        # K inside QK-high, so both groups must finish before replacing K0.
        _stage_end()
    # Both groups have read K0. The only K slot can now hold K1.
    if fx.const_expr(not separate or DQ == 128):
        k1 = _load_k(gk, tid, page10, page11, _min(fx.Int32(1), last), HK, DQ, PAGE)
    _wait(vmcnt=0)
    if fx.const_expr(separate):
        write_lds(kw_address, k1, DQ // 64)
    else:
        _cooperative_store(storage, tid, k1, 0, DQ // 64, True, bounded)
    _wait(lgkmcnt=0)
    _stage_end()
    # Single-slot BF16 differs from FP8's prepublished K1 ring. The leading
    # group must wait one additional stage for the trailing group's K1 writes
    # before entering S0; an LGKM wait alone cannot cover another wave.
    _stage_end()

    @flyc.jit
    def phase(previous, maximum, row_sum, o0, o1, v_pending, current0, current1, next0, next1, t,
              CUR: fx.Constexpr[int], PREV: fx.Constexpr[int]):
        previous, maximum, row_sum = fx.Vector(previous), fx.Float32(maximum), fx.Float32(row_sum)
        o0, o1, v_pending, t = fx.Vector(o0), fx.Vector(o1), fx.Vector(v_pending), fx.Int32(t)
        current_slot = (t & 1) if DQ == 192 and v_ring else CUR
        previous_slot = ((t - 1) & 1) if DQ == 192 and v_ring else PREV
        # S0 memory: K(t).lo from LDS + V(t) GLOBAL + scalar page lookahead.
        # Lane offsets are loop-invariant; changing V bases uses SGPRs only.
        if fx.const_expr(separate):
            k = read_k(kr_address, 0, DQ)
        else:
            k = _k_fragment(storage, k_base, 0, DQ, DV, full_k)
        rocdl.sched_barrier(0)
        if fx.const_expr(not v_ring):
            _cooperative_store(storage, tid, v_pending, k_bytes, DV // 64, bounded=bounded)
        request0, request1 = _prefetch_tile(table, t + 2, last, last_page, PAGE)
        v_staging = fx.Vector.filled(DV // 16, 0, fx.Int32)
        if fx.const_expr(not full_k and not separate):
            v_staging = _load_v(V, tid, current0, current1, t, hkv, HK, DV, PAGE, bounded)
        k_staging = fx.Vector.filled(DQ // 16, 0, fx.Int32)
        if fx.const_expr(separate):
            v_first, v_second = v_addresses(v_global_base, current0, current1, t, HK, PAGE)
            v_staging = load_v_address(v_first, v_second, v_lane)
        else:
            k_staging = _load_k(gk, tid, next0, next1, _min(t + 1, last), HK, DQ, PAGE)
        rocdl.sched_barrier(0)
        future0, future1 = _page_ready(request0), _page_ready(request1)
        _stage_end()
        # S1 compute: QK(t).lo + first 24 exps of softmax(t-1). Dynamic
        # D192 V-ring addresses are materialized here, not in S2/S4/S6.
        lo = _qk(q, k, storage, k_base, 0, DQ, DV, full_k)
        p = fx.Vector.filled(32, 0.0, fx.BFloat16)
        local_sum = fx.Float32(0.0)
        v_read, v_write = vr_address, vw_address
        if fx.const_expr(separate):
            if fx.const_expr(full_k):
                v_read = _pin_i32(vr_address + previous_slot * 16640)
                v_write = _pin_i32(vw_address + current_slot * 16640)
        previous = _pin(_exp_part(previous, 0, 24), 32)
        _schedule(DQ // 8, 2, 1, True)
        _stage_end()
        # S2 memory: publish V(t) in its alternate slot, then read K(t).hi.
        # Retire staging registers before allocating the full K fragment.
        other_sum = fx.Float32(0.0)
        if fx.const_expr(separate):
            _wait(vmcnt=0)
            write_lds(v_write, v_staging, 2, CUR * 16640 if DQ == 128 else 0)
            k = read_k(kr_address, 1, DQ)
        else:
            k = _k_fragment(storage, k_base, 1, DQ, DV, full_k)
        if fx.const_expr(v_ring and not full_k and not separate):
            _wait(vmcnt=DQ // 64)
            _cooperative_store(storage, tid, v_staging, k_bytes + current_slot * (DV // 8) * PADDED_CHUNK,
                               DV // 64, bounded=bounded)
        _wait(lgkmcnt=0)
        _stage_end()
        # S3 compute: QK(t).hi + remaining exp/local sum/P(t-1), and the
        # divergent page32 K address for the following memory stage.
        hi = _qk(q, k, storage, k_base, 1, DQ, DV, full_k)
        previous = _exp_part(previous, 24, 32)
        if fx.const_expr(separate):
            local_sum = sum_local(previous)
        else:
            row_sum = row_sum + _row_sum(previous)
        p = _pack_probability(previous)
        _schedule(4, 2, 2, True)
        _schedule(DQ // 8 - 4, 8, 2)
        k_fetch = fx.Int32(0)
        if fx.const_expr(separate):
            k_fetch = k_address(k_lane, next0, next1, _min(t + 1, last), half_page, HK, DQ, PAGE)
        _stage_end()
        # S4 memory: issue cross32(sum), read V(t-1).lo and prefetch K(t+1).
        # The shuffle shares the final LDS wait; no immediate dependent use.
        if fx.const_expr(separate):
            other_sum = cross_async(local_sum, cross_address)
            v = read_v(v_read, 0, PREV if DQ == 128 else 0)
        else:
            v = _v_fragment(storage, v_base, previous_slot if v_ring else 0, 0, DQ, DV)
        _wait(vmcnt=0)
        if fx.const_expr(separate):
            k_staging = load_k_address(gk, k_fetch, DQ, PAGE)
        if fx.const_expr(((DQ == 128 and DV == 128) or full_k) and not separate):
            _cooperative_store(storage, tid, k_staging, 0, DQ // 64, True, bounded)
        _wait(lgkmcnt=0)
        _stage_end()
        # S5 compute: PV(t-1).lo + mask/local max of QK(t), finish sum(t-1).
        current = _mask(_join(lo, hi), t, row, q_len, kv_len, CAUSAL)
        o0 = _pv(p, v, o0, DV)
        local_max = fx.Float32(0.0)
        ballot, new_max = fx.Int64(0), maximum
        split = 12 if STAGGER else 6
        if fx.const_expr(separate):
            row_sum = row_sum + (local_sum + other_sum)
            local_max = max_local(current)
            _schedule(DV // 8, 2, 3)
        else:
            candidate = _row_max(current) * scale
            ballot = fx.Int64(rocdl.ballot(fx.Int64.ir_type, (candidate > maximum + 7.0).ir_value()))
            new_max = (candidate > maximum + 7.0).select(candidate + 1.0, maximum)
            current = _center(current, scale, new_max, 0, split)
            _schedule(3, 5, 3)
            rocdl.sched_group_barrier(rocdl.mask_mfma, 1, 3)
            rocdl.sched_group_barrier(0x002, 3, 3)
            rocdl.sched_group_barrier(rocdl.mask_dsrd, 1, 3)
            rocdl.sched_group_barrier(rocdl.mask_mfma, 2, 3)
            rocdl.sched_group_barrier(0x002, 10, 3)
            _schedule(DV // 8 - 6, 5, 3)
        _stage_end()
        # S6 memory: cross32(max), V(t-1).hi and K(t+1) publication. Both
        # groups retired their current K reads before this single-slot write.
        other_max = fx.Float32(0.0)
        if fx.const_expr(separate):
            other_max = cross_async(local_max, cross_address)
            v = read_v(v_read, 1, PREV if DQ == 128 else 0)
            _wait(vmcnt=0)
            write_lds(kw_address, k_staging, DQ // 64)
        else:
            v = _v_fragment(storage, v_base, previous_slot if v_ring else 0, 1, DQ, DV)
        if fx.const_expr((DQ == 192 or DV == 192) and not full_k):
            # The partner's S3 streams K through the end of QK-high.
            # S4 would overwrite its last chunks. S6 follows BOTH readers
            # and still leaves two stages before the next K-low read.
            _cooperative_store(storage, tid, k_staging, 0, DQ // 64, True, bounded)
        _wait(lgkmcnt=0)
        _stage_end()
        # S7 compute: PV(t-1).hi + complete max/center(t) and lazy rescale.
        o1 = _pv(p, v, o1, DV)
        if fx.const_expr(separate):
            candidate = _maximum(local_max, other_max) * scale
            ballot = fx.Int64(rocdl.ballot(fx.Int64.ir_type, (candidate > maximum + 7.0).ir_value()))
            new_max = (candidate > maximum + 7.0).select(candidate + 1.0, maximum)
            current = _center(current, scale, new_max, 0, 32)
        else:
            current = _center(current, scale, new_max, split, 32)
        _schedule(DV // 8, 4, 4)
        o0, o1, row_sum = rescale(o0, o1, row_sum, maximum, new_max, ballot)
        if fx.const_expr(separate):
            new_max = advance_max(maximum, new_max)
        _stage_end()
        pending = fx.Vector.filled(DV // 16, 0, fx.Int32) if v_ring else v_staging
        return current, new_max, row_sum, o0, o1, pending, next0, next1, future0, future1

    if fx.const_expr(v_ring and DQ == 128):
        # FP8's two-phase unroll keeps V ring indices compile-time constants.
        for t in range(fx.Int32(1), tiles - 1, fx.Int32(2)):
            scores, maximum, row_sum, o0, o1, v_pending, page10, page11, page20, page21 = phase(
                scores, maximum, row_sum, o0, o1, v_pending, page10, page11, page20, page21, t, 1, 0)
            scores, maximum, row_sum, o0, o1, v_pending, page10, page11, page20, page21 = phase(
                scores, maximum, row_sum, o0, o1, v_pending, page10, page11, page20, page21, t + 1, 0, 1)
        if (tiles & 1) == 0:
            scores, maximum, row_sum, o0, o1, v_pending, page10, page11, page20, page21 = phase(
                scores, maximum, row_sum, o0, o1, v_pending, page10, page11, page20, page21, last, 1, 0)
    else:
        # D192 ring indices are scalar t&1 values. Keeping a single phase
        # avoids the large register-pressure increase of double unrolling.
        for t in range(fx.Int32(1), tiles, fx.Int32(1)):
            scores, maximum, row_sum, o0, o1, v_pending, page10, page11, page20, page21 = phase(
                scores, maximum, row_sum, o0, o1, v_pending, page10, page11, page20, page21, t, 0, 0)

    # Single-slot variants publish the last prefetched V here, after both
    # preceding S6 readers. Ring variants published it during S2/S4.
    if fx.const_expr(not v_ring):
        _cooperative_store(storage, tid, v_pending, k_bytes, DV // 64, bounded=bounded)
    scores = _exp_part(fx.Vector(scores), 0, 32)
    row_sum = fx.Float32(row_sum) + _row_sum(scores)
    p = _pack_probability(scores)
    _wait(vmcnt=0, lgkmcnt=0)
    _stage_end()
    if fx.const_expr(not v_ring):
        # Wait for the trailing group's final single-slot V publication.
        _stage_end()
    v = _v_fragment(storage, v_base, (last & 1) if v_ring else 0, 0, DQ, DV)
    _wait(lgkmcnt=0)
    rocdl.sched_barrier(0)
    v.store(_v_tail(v.load(), kv_len - last * BN))
    _stage_end()
    o0 = _pv(p, v, o0, DV)
    _stage_end()
    v = _v_fragment(storage, v_base, (last & 1) if v_ring else 0, 1, DQ, DV)
    _wait(lgkmcnt=0)
    rocdl.sched_barrier(0)
    v.store(_v_tail(v.load(), kv_len - last * BN))
    _stage_end()
    o1 = _pv(p, v, o1, DV)
    _stage_end()
    if fx.const_expr(not STAGGER):
        _stage_end()

    inv = (row_sum > 0.0).select(fx.Float32(1.0) / row_sum, fx.Float32(0.0)) * VS[0]
    out_tid = _pin_i32(fx.Int32(gpu.thread_id("x")))
    out_row = (out_tid >> 6) * 32 + (out_tid & 31)
    optr = fx.get_iter(O) + ((fx.Int64(q0) + fx.Int64(q_start)) * (H * DV) + fx.Int64(head) * DV)
    obuf = rocdl.make_buffer_tensor(fx.make_view(optr, fx.make_layout(BM * H * DV, 1)),
                                    num_records_bytes=valid * H * DV * 2)
    # FP8 C-shuffle: reuse 32 KiB, one 64-column slice at a time. The public
    # token-major output is unchanged; V192 simply has a third slice.
    atom = fx.make_copy_atom(rocdl.BufferCopy128b(), fx.BFloat16)
    outputs = _join(o0, o1)
    for half in fx.range_constexpr(DV // 64):
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
            words = _lds_words(storage, element * 2)
            _wait(lgkmcnt=0)
            rocdl.sched_barrier(0)
            src = fx.make_rmem_tensor(8, fx.BFloat16)
            src.store(fx.Vector.from_elements([
                ((read_row & 1) == 0).select(words[j], words[j ^ 2]) for j in range(4)
            ], fx.Int32).bitcast(fx.BFloat16))
            offset = read_row * (H * DV) + half * 64 + read_col
            fx.copy(atom, src, fx.make_view(fx.get_iter(obuf) + offset, fx.make_layout(8, 1)))
        _stage_end()
    if fx.const_expr(WITH_LSE):
        if ((out_tid & 63) < 32) & (out_row < valid):
            log_l = fx.Float32(llvm.call_intrinsic(fx.Float32.ir_type, "llvm.log2.f32", [row_sum.ir_value()], [], []))
            LSE[(q0 + q_start + out_row) * H + head] = (row_sum > 0.0).select(
                (fx.Float32(maximum) + log_l) * fx.Float32(math.log(2.0)), fx.Float32(float("-inf")))


@flyc.jit
def _work(Q, K, V, O, LSE, CQ, KI, PAGES, LAST, QS, KS, VS, storage, head, batch, qb,
          H: fx.Constexpr[int], HK: fx.Constexpr[int], NP: fx.Constexpr[int], DQ: fx.Constexpr[int],
          DV: fx.Constexpr[int], PAGE: fx.Constexpr[int], CAUSAL: fx.Constexpr[bool],
          PER_TOKEN: fx.Constexpr[bool], WITH_LSE: fx.Constexpr[bool], SCALE: fx.Constexpr[float]):
    # AST rewriting lifts branch bodies into local functions. Capture the
    # compute dependency before the branch so the native cache follows it.
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
        value_attrs={"rocdl.waves_per_eu": 2, "passthrough": [["target-features", "-packed-fp32-ops"]]},
    ).launch(grid=grid, block=(THREADS, 1, 1), stream=stream)


class _PagedAttention:
    bf16_backend = "native-8wave-8stage"

    def __init__(self, heads, kv_heads, dq, dv, page, causal, mode, persistent):
        self.heads, self.kv_heads, self.dq, self.dv = heads, kv_heads, dq, dv
        self.page_size, self.causal, self.quant_query_mode = page, causal, mode
        self.memory_mode, self.persistent = "lds", persistent
        self._compiled = {}

    def __call__(self, Q, K, V, cu_seqlens_q, cu_seqlens_k, kv_indptr, kv_page_indices,
                 max_seqlen_q, max_seqlen_k, causal, q_descale, k_descale, v_descale,
                 kv_last_page_lens, out=None, sink_ptr=None, stream=None, *, return_lse=False, lse=None, softmax_scale=None):
        """Run with contiguous SHUFFLE-5D KV and asynchronous device metadata.

        Descales are finite positive FP32. Metadata must be consistent/in-bounds;
        no GPU metadata is copied to the host. Active KV must be nonempty and
        causal KV >= Q. Caller output/LSE must not overlap inputs or each other.
        Empty Q returns without dispatch. Warm before graph capture.
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
        if isinstance(softmax_scale, (torch.Tensor, bool)):
            raise ValueError("softmax_scale must be a host scalar or None")
        scale = self.dq**-0.5 if softmax_scale is None else float(softmax_scale)
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("softmax_scale must be finite and positive")
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
                    self._compiled[key] = flyc.compile(_launch_attention, *args)
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
    if head_dim_qk not in (128, 192) or head_dim_v not in (128, 192) or page_size not in (32, 64, 128):
        raise NotImplementedError("gfx942 BF16 supports D128/D192, V128/V192 and pages32/64/128")
    if quant_query_mode not in ("per-token", "per-tensor"):
        raise ValueError("query scale mode must be per-token or per-tensor")
    return _PagedAttention(num_qo_heads, num_kv_heads, head_dim_qk, head_dim_v, page_size,
                           is_causal, quant_query_mode, persistent is not False)