"""Shared FlyDSL helpers of the gfx942 MHA kernels (also used by QSA).

Data-movement names follow the MoE kernels: g = global memory, s = LDS
(shared), r = registers. ``read_x_g2r`` loads global data into registers,
``read_x_g2s`` copies it straight into LDS (``buffer_load ... lds``),
``store_x_r2s``/``read_x_s2r`` write/read LDS and ``store_x_r2g`` writes the
result. Read helpers only issue instructions; kernels place every wait.

1. Scalar/vector utilities, raw buffers, page ids and stage barriers.
2. BM256 32x32-MFMA softmax pieces of the D128/D192 BF16 and FP8 kernels.
3. BM128/BN64 D256 geometry, 16x16 MFMA fragments and softmax of the paged
   and linear D256 kernels.
"""

import math

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import gpu, rocdl
from pyhip.codegen.flydsl.helpers import rocdl_aux


# ---- 1. Utilities, raw buffers, page ids and stage barriers ----

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
    # gfx942 has no native packed FP32 -> BF16 instruction; software RNE for O.
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
    # gfx94x s_waitcnt encoding; an omitted counter is not waited on.
    rocdl.s_waitcnt((vmcnt & 15) | (expcnt << 4) | (lgkmcnt << 8) | ((vmcnt >> 4) << 14))


def _schedule(pairs, count, group, exp=False):
    # Interleave one MFMA with `count` VALU (0x002) or transcendental (0x400) ops.
    for _ in range(pairs):
        rocdl.sched_group_barrier(rocdl.mask_mfma, 1, group)
        rocdl.sched_group_barrier(0x400 if exp else 0x002, count, group)


def _read_page_g2r(table, index):
    """s_load_dword of one page id; _page_ready makes it usable."""
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
    """Raw buffer descriptor; loads at or past size_bytes return zero, stores are dropped."""
    address = fx.Int64(fx.ptrtoint(fx.get_iter(tensor)))
    pointer = llvm.inttoptr(ir.Type.parse("!llvm.ptr"), address.ir_value())
    return rocdl.make_buffer_rsrc(ir.Type.parse("!llvm.ptr<8>"), pointer,
                                 fx.Int16(0).ir_value(), fx.Int64(size_bytes).ir_value(),
                                 fx.Int32(0x27000).ir_value())


def _buffer_words(resource, voffset, soffset=0):
    """One 16-byte buffer_load (g2r) at byte offsets voffset + soffset."""
    return fx.Vector(rocdl.raw_ptr_buffer_load(ir.VectorType.get([4], fx.Int32.ir_type), resource,
                                              fx.Int32(voffset).ir_value(), fx.Int32(soffset).ir_value(),
                                              aux=rocdl_aux(0)))


def _read_address(address, immediate=0):
    """One ds_read_b128 (s2r) at a precomputed LDS byte address; asynchronous until lgkmcnt."""
    return fx.Vector(llvm.inline_asm(ir.VectorType.get([4], fx.Int32.ir_type), [address.ir_value()],
        f"ds_read_b128 $0, $1 offset:{immediate}", "=v,v,~{memory}", has_side_effects=True))


@flyc.jit
def _rescale(o0, o1, row_sum, old_max, new_max, ballot):
    o0, o1, row_sum = fx.Vector(o0), fx.Vector(o1), fx.Float32(row_sum)
    if ballot != fx.Int64(0):
        correction = _exp(fx.Float32(old_max) - fx.Float32(new_max))
        o0, o1, row_sum = o0 * correction, o1 * correction, row_sum * correction
    return o0, o1, row_sum


# ---- 2. BM256 32x32-MFMA softmax (one query row per lane pair, 32 scores per lane) ----

def _max3(a, b, c):
    return fx.Float32(llvm.inline_asm(fx.Float32.ir_type, [a.ir_value(), b.ir_value(), c.ir_value()],
                                    "v_max3_f32 $0, $1, $2, $3", "=v,v,v,v", has_side_effects=False))


def _mfma_hazard_wait():
    # Inline-asm readers of fresh MFMA accumulators get no wait states from LLVM's hazard recognizer.
    # 11 = NumPasses(8) + 3 for the 32x32 MFMAs of this kernel on gfx942.
    rocdl.sched_barrier(0)
    llvm.inline_asm(None, [], "s_nop 10", "", has_side_effects=True)
    rocdl.sched_barrier(0)


def _row_max_local(values):
    # Independent MAX3 chains avoid the 32-deep VALU dependency chain.
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


def _center_part(values, scale, maximum, begin, end):
    """S' = S * scale - m for scores begin..end-1; the rest pass through."""
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
    """P = exp2(S') for scores begin..end-1; the rest pass through."""
    return fx.Vector.from_elements([_exp(values[i]) if begin <= i < end else values[i] for i in range(32)], fx.Float32)


# ---- 3. D256 BM128/BN64: 16x16 MFMA fragments and softmax (16 scores per lane) ----

BM, BN, D, THREADS = 128, 64, 256, 512
K_BYTES, V_BYTES = 32768, 32768
LDS_BYTES = K_BYTES + V_BYTES
LOG2E = math.log2(math.e)


def _mma():
    return fx.make_tiled_mma(fx.make_mma_atom(rocdl.MFMA(16, 16, 16, fx.BFloat16)),
                             fx.make_layout((1, 8, 1), (1, 1, 0)))


def _read_q_g2r(resource, offset, lane):
    """Q rows of this lane as the QK B operand; Q stays in registers for every KV tile."""
    parts = [_buffer_words(resource, offset + (lane >> 4) * 16 + k * 64) for k in range(8)]
    words = fx.Vector.from_elements([p[i] for p in parts for i in range(4)], fx.Int32)
    frag = _mma().make_fragment_B(fx.make_rmem_tensor(fx.make_layout((BM, D), (1, BM)), fx.BFloat16))
    frag.store(words.bitcast(fx.BFloat16))
    return frag


def _qk(q, k):
    """S = K Q^T for 32 keys: 16 scores per lane (two 16x16 tiles of eight waves)."""
    mma = _mma()
    acc = mma.make_fragment_C(fx.make_rmem_tensor(fx.make_layout((32, BM), (1, 32)), fx.Float32))
    acc.fill(0.0)
    fx.gemm(mma, acc, k, q, acc, traversal_order="kmn")
    return acc.load()


def _pv(probabilities, values, output):
    """O += P V for one 128-dimension half of V."""
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
    """Exchange a row statistic with the three other lane groups of the row (ds_bpermute)."""
    return tuple(fx.Float32(llvm.inline_asm(fx.Float32.ir_type, [address.ir_value(), value.ir_value()],
        "ds_bpermute_b32 $0, $1, $2", "=v,v,v,~{memory}", has_side_effects=True)) for address in addresses)


def _center(values, scale, maximum):
    return fx.Vector.from_elements([fx.Float32(llvm.inline_asm(
        fx.Float32.ir_type, [values[i].ir_value(), scale.ir_value(), maximum.ir_value()],
        "v_fma_f32 $0, $1, $2, -$3", "=v,v,v,v", has_side_effects=False)) for i in range(16)], fx.Float32)


def _exps(values):
    return fx.Vector.from_elements([_exp(values[i]) for i in range(16)], fx.Float32)


def _pack(values):
    """P to BF16 (round half up) with one v_perm_b32 per pair."""
    bits = fx.Vector(values).bitcast(fx.Uint32) + fx.Uint32(0x8000)
    selector = fx.Int32(0x07060302)
    words = [fx.Int32(llvm.inline_asm(fx.Int32.ir_type,
        [bits[i + 1].ir_value(), bits[i].ir_value(), selector.ir_value()],
        "v_perm_b32 $0, $1, $2, $3", "=v,v,v,s", has_side_effects=True)) for i in range(0, 16, 2)]
    return fx.Vector.from_elements(words, fx.Int32).bitcast(fx.BFloat16)


@flyc.jit
def _mask(scores, tile, row, q_len, kv_len, CAUSAL: fx.Constexpr[bool]):
    """-inf for keys past the KV length or right of the bottom-right causal diagonal."""
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
    """Zero V rows past the KV length: P = 0 times a NaN pad would still be NaN."""
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
