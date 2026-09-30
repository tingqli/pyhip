"""Shared gfx942 BF16 arithmetic, raw-buffer operations and pipeline barriers."""

import math

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import gpu, rocdl
from pyhip.codegen.flydsl.helpers import rocdl_aux


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


def _pin_s64(value):
    return fx.Int64(llvm.inline_asm(fx.Int64.ir_type, [fx.Int64(value).ir_value()],
                                   "", "=s,0", has_side_effects=True))


def _advance_max(old, new):
    return fx.Float32(llvm.inline_asm(fx.Float32.ir_type, [old.ir_value(), new.ir_value()],
                                    "v_mov_b32 $0, $2", "=v,0,v", has_side_effects=True))


def _pack_bf16(values):
    # gfx942 has no native packed FP32 -> BF16 instruction. Software RNE,
    # as in the FP8 kernel's output path, is used for O.
    bits = fx.Vector(values).bitcast(fx.Uint32)
    rounded = bits + fx.Uint32(0x7FFF) + ((bits >> 16) & fx.Uint32(1))
    words = [(rounded[i] >> 16) | (rounded[i + 1] & fx.Uint32(0xFFFF0000))
             for i in range(0, values.numel, 2)]
    return fx.Vector.from_elements(words, fx.Uint32)


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


def _buffer(tensor, size_bytes):
    address = fx.Int64(fx.ptrtoint(fx.get_iter(tensor)))
    pointer = llvm.inttoptr(ir.Type.parse("!llvm.ptr"), address.ir_value())
    return rocdl.make_buffer_rsrc(ir.Type.parse("!llvm.ptr<8>"), pointer,
                                 fx.Int16(0).ir_value(), fx.Int64(size_bytes).ir_value(),
                                 fx.Int32(0x27000).ir_value())


def _buffer_words(resource, voffset, soffset=0):
    return fx.Vector(rocdl.raw_ptr_buffer_load(ir.VectorType.get([4], fx.Int32.ir_type), resource,
                                              fx.Int32(voffset).ir_value(), fx.Int32(soffset).ir_value(),
                                              aux=rocdl_aux(0)))


def _read_address(address, immediate=0):
    return fx.Vector(llvm.inline_asm(ir.VectorType.get([4], fx.Int32.ir_type), [address.ir_value()],
        f"ds_read_b128 $0, $1 offset:{immediate}", "=v,v,~{memory}", has_side_effects=True))


@flyc.jit
def _rescale(o0, o1, row_sum, old_max, new_max, ballot):
    o0, o1, row_sum = fx.Vector(o0), fx.Vector(o1), fx.Float32(row_sum)
    if ballot != fx.Int64(0):
        correction = _exp(fx.Float32(old_max) - fx.Float32(new_max))
        o0, o1, row_sum = o0 * correction, o1 * correction, row_sum * correction
    return o0, o1, row_sum


# ---- D256-only geometry, MFMA fragments, softmax and paged LDS readers ----
# The helpers above remain dimension-independent; these use BM128/BN64/D256.

BM, BN, D, THREADS = 128, 64, 256, 512
K_PITCH, K_BYTES, V_BYTES = 1024, 32768, 32768
LDS_BYTES = K_BYTES + V_BYTES
LOG2E = math.log2(math.e)


def _mma():
    return fx.make_tiled_mma(fx.make_mma_atom(rocdl.MFMA(16, 16, 16, fx.BFloat16)),
                             fx.make_layout((1, 8, 1), (1, 1, 0)))


def _q_fragment(resource, offset, lane):
    parts = [_buffer_words(resource, offset + (lane >> 4) * 16 + k * 64) for k in range(8)]
    words = fx.Vector.from_elements([p[i] for p in parts for i in range(4)], fx.Int32)
    frag = _mma().make_fragment_B(fx.make_rmem_tensor(fx.make_layout((BM, D), (1, BM)), fx.BFloat16))
    frag.store(words.bitcast(fx.BFloat16))
    return frag


def _read_k(address, half):
    parts = [_read_address(address[n], half * 512 + k * (4 * K_PITCH))
             for n in range(2) for k in range(8)]
    words = fx.Vector.from_elements([p[i] for p in parts for i in range(4)], fx.Int32)
    storage = fx.make_rmem_tensor(fx.make_layout((4, 16, 2), (1, 4, 64)), fx.BFloat16)
    storage.store(words.bitcast(fx.BFloat16))
    return fx.make_view(fx.get_iter(storage), fx.make_layout((4, 2, 16), (1, 64, 4)))


def _read_v(address, half):
    parts = [_read_address(address, half * 2048 + n * 256 + k * 16384)
             for n in range(8) for k in range(2)]
    words = fx.Vector.from_elements([p[i] for p in parts for i in range(4)], fx.Int32)
    frag = fx.make_rmem_tensor(fx.make_layout((4, 4, 8), (1, 4, 16)), fx.BFloat16)
    frag.store(words.bitcast(fx.BFloat16))
    return frag


def _qk(q, k):
    mma = _mma()
    acc = mma.make_fragment_C(fx.make_rmem_tensor(fx.make_layout((32, BM), (1, 32)), fx.Float32))
    acc.fill(0.0)
    fx.gemm(mma, acc, k, q, acc, traversal_order="kmn")
    return acc.load()


def _pv(probabilities, values, output):
    mma = _mma()
    p = mma.make_fragment_B(fx.make_rmem_tensor(fx.make_layout((BM, BN), (1, BM)), fx.BFloat16))
    p.store(probabilities)
    acc = mma.make_fragment_C(fx.make_rmem_tensor(fx.make_layout((128, BM), (1, 128)), fx.Float32))
    acc.store(output)
    operand = fx.make_view(fx.get_iter(values), fx.make_layout((4, 8, 4), (1, 16, 4)))
    fx.gemm(mma, acc, operand, p, acc, traversal_order="mnk")
    return acc.load()


def _sum(values):
    partials = [values[i] + values[i + 1] for i in range(0, 16, 2)]
    for width in (4, 2, 1):
        partials = [partials[2 * i] + partials[2 * i + 1] for i in range(width)]
    return partials[0]


def _max(values):
    # Keep MFMA -> max dependencies visible to LLVM. The earlier opaque
    # max tree failed repeated-bit-exact grid checks; maxnum passed without
    # the NaN-propagating compare/select expansion of maximumf.
    def maximum(a, b):
        return fx.Float32(llvm.call_intrinsic(fx.Float32.ir_type, "llvm.maxnum.f32",
                                              [a.ir_value(), b.ir_value()], [], []))
    parts = [maximum(values[i], values[i + 1]) for i in range(0, 16, 2)]
    for width in (4, 2, 1):
        parts = [maximum(parts[2 * i], parts[2 * i + 1]) for i in range(width)]
    return parts[0]


def _cross(value, addresses):
    return tuple(fx.Float32(llvm.inline_asm(fx.Float32.ir_type, [address.ir_value(), value.ir_value()],
        "ds_bpermute_b32 $0, $1, $2", "=v,v,v,~{memory}", has_side_effects=True)) for address in addresses)


def _center(values, scale, maximum):
    return fx.Vector.from_elements([fx.Float32(llvm.inline_asm(
        fx.Float32.ir_type, [values[i].ir_value(), scale.ir_value(), maximum.ir_value()],
        "v_fma_f32 $0, $1, $2, -$3", "=v,v,v,v", has_side_effects=False)) for i in range(16)], fx.Float32)


def _exps(values):
    return fx.Vector.from_elements([_exp(values[i]) for i in range(16)], fx.Float32)


def _pack(values):
    bits = fx.Vector(values).bitcast(fx.Uint32) + fx.Uint32(0x8000)
    selector = fx.Int32(0x07060302)
    words = [fx.Int32(llvm.inline_asm(fx.Int32.ir_type,
        [bits[i + 1].ir_value(), bits[i].ir_value(), selector.ir_value()],
        "v_perm_b32 $0, $1, $2, $3", "=v,v,v,s", has_side_effects=True)) for i in range(0, 16, 2)]
    return fx.Vector.from_elements(words, fx.Int32).bitcast(fx.BFloat16)


@flyc.jit
def _mask(scores, tile, row, q_len, kv_len, CAUSAL: fx.Constexpr[bool]):
    scores = fx.Vector(scores)
    if fx.const_expr(CAUSAL):
        if row - (fx.Int32(gpu.thread_id("x")) & 15) + kv_len - q_len < (tile + 1) * BN:
            bound = _pin_i32(_min(kv_len - 1, kv_len - q_len + row) - tile * BN
                             - ((fx.Int32(gpu.thread_id("x")) >> 4) & 3) * 8)
            scores = fx.Vector.from_elements([(bound >= (i // 8) * 32 + i % 8).select(scores[i], fx.Float32(float("-inf")))
                                              for i in range(16)], fx.Float32)
    else:
        if kv_len - tile * BN < BN:
            bound = _pin_i32(kv_len - 1 - tile * BN - ((fx.Int32(gpu.thread_id("x")) >> 4) & 3) * 8)
            scores = fx.Vector.from_elements([(bound >= (i // 8) * 32 + i % 8).select(scores[i], fx.Float32(float("-inf")))
                                              for i in range(16)], fx.Float32)
    return scores


@flyc.jit
def _v_tail(values, remaining):
    words = fx.Vector(values).bitcast(fx.Int32)
    if remaining < BN:
        limit = _pin_i32(remaining - ((fx.Int32(gpu.thread_id("x")) >> 4) & 3) * 8)
        masks = []
        for i in fx.range_constexpr(8):
            token = (i // 4) * 32 + (i % 4) * 2
            mask = (limit > token).select(fx.Int32(0xFFFF), fx.Int32(0))
            mask = mask | (limit > token + 1).select(fx.Int32(-65536), fx.Int32(0))
            masks.append(mask)
        words = fx.Vector.from_elements([words[i] & masks[i % 8] for i in range(words.numel)], fx.Int32)
    return words.bitcast(fx.BFloat16)


def _pages(table, tile, last, kv_len, page_size):
    first = _min(tile, last) * BN // page_size
    page0 = _prefetch_page(table, first)
    page1 = _prefetch_page(table, _min(first + 1, (kv_len - 1) // page_size)) if page_size == 32 else page0
    return page0, page1
