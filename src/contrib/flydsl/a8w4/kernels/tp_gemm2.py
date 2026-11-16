# Package copy of doc-mimo/kernels/test_a8w4_gemm2.py (test harness removed, env knobs frozen).
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Standalone A8W4 MoE stage2 (down-proj GEMM + routed atomic scatter) FlyDSL kernel
for gfx942.

Extracted from the a16w4 port (``moe_2stage_a16wmix/gemm2.py``, ``a_dtype="fp8r"``)
with only the configuration the tuned A8W4 build actually runs
(``flydsl_moe2_afp8r_wfp4_bf16_t{BM}x{TN}x{TK}_atomic[_bnt*][_xcd*][_ftz]``):

  * A  : FP8-FNUZ row-quantized stage1 intermediate ``[sorted_size, inter_dim]``
         (by SORTED position) + f32 per-sorted-row scale
  * W2 : mxfp4 (E2M1 + per-1x32 E8M0) ``[E, model_dim, inter_dim/2]``, standard
         N-major preshuffle
  * per-output-channel max E8M0 exponent derived in-kernel from the W2 scales
  * MFMA(16,16,32,fp8) with fp4 -> fp8 decode through a per-32K-group byte LUT
    (+ a sign-mask LUT, so FNUZ -0 never needs a per-fragment fix-up)
  * K (= inter_dim) loop unrolled, XOR-swizzled A-LDS, 4 waves split TILE_N
  * ``dyn_rows``: per-WG runtime skip of all-padding 16-row sub-blocks
  * epilogue: acc * a_scale[row] * 2**(exp-132) * route_weight -> bf16 LDS tile
    (half the port's f32 staging) -> packed bf16x2 global atomic add into
    ``[tokens, model_dim]``
"""

import functools
from functools import cache
import os

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir.dialects import arith, llvm
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from .._vendor import buffer_ops
from .._vendor import buffer_ops as bo
from .._vendor.act import gate_up_act, silu_mul_batch, situ_params
from .._vendor.a8w4_ops import _pack_activation, _pointer, channel_exponents
from .._vendor.moe_utils import _buffer_i32_scalar_read, _global_i32_at, _udiv, _umod
from .._vendor.mxfp4_gemm_common import (
    _gep1,
    _global_base_ptr1,
    _global_i32_buffer_tiles,
    _global_i32_buffer_view,
    _lds_ptr3,
    global_typed_ptr,
    lds_typed_ptr,
    lds_vec_load,
)
from .._vendor.tensor_shim import _run_compiled
from .._vendor.tensor_shim import _to_raw as _raw


# =============================================================================
# Device helpers: fp4 -> fp8 decode, channel factor, MFMA (same as test_a8w4_gemm1)
# =============================================================================
#
# W stays E2M1/E8M0. Output channel n is normalized by 2**(max_exp[n]-127-5)
# (``_channel_factor``, applied in the epilogue); per 32-K group the remaining
# 2**(5-delta), delta = max_exp - group_exp, is folded into an 8-entry
# E2M1-magnitude -> FP8-FNUZ byte table (``_fp8_lut``) so each nibble decodes with
# one v_perm.


def _perm_u32(hi, lo, sel):
    return fx.Uint32(
        rocdl.perm_b32(
            fx.Uint32(hi).ir_value(), fx.Uint32(lo).ir_value(), fx.Uint32(sel).ir_value()
        )
    )


def _fp8_lut(group_code, max_code, flush_tiny):
    """8-entry ``|E2M1| * 2**(5-delta)`` -> FP8-FNUZ byte table.

    Returns ``(lo, hi, sign_lo, sign_hi)``: the magnitude table plus a same-shaped
    mask holding 0x80 for every non-zero entry (FNUZ has no -0; 0x80 is NaN).
    """
    difference = max_code - group_code
    delta = fx.min(difference, 11).to(fx.Uint32) * fx.Uint32(0x08080808)
    lo = (fx.Uint32(0x6C686080) - delta) & fx.Uint32(0xFFFFFF00)
    hi = fx.Uint32(0x7C787470) - delta
    if const_expr(flush_tiny):
        # Approximation: drop groups < 2**-11 of this channel's largest scale.
        lo = (difference > 11).select(fx.Uint32(0), lo)
        hi = (difference > 11).select(fx.Uint32(0), hi)
    else:
        for diff, low, high in (
            (12, 0x0C080400, 0x1C181410),
            (13, 0x06040200, 0x14100C08),
            (14, 0x03020100, 0x0C080604),
            (15, 0x02010000, 0x06040302),
            (16, 0x01000000, 0x03020201),
            (17, 0, 0x02010100),
            (18, 0, 0x01000000),
        ):
            lo = (difference == diff).select(fx.Uint32(low), lo)
            hi = (difference == diff).select(fx.Uint32(high), hi)
        lo = (difference >= 19).select(fx.Uint32(0), lo)
        hi = (difference >= 19).select(fx.Uint32(0), hi)
    # Entries are <= 0x7C, so +0x7F sets bit 7 exactly for non-zero bytes (no carry).
    sign_lo = (lo + fx.Uint32(0x7F7F7F7F)) & fx.Uint32(0x80808080)
    sign_hi = (hi + fx.Uint32(0x7F7F7F7F)) & fx.Uint32(0x80808080)
    return lo, hi, sign_lo, sign_hi


def _fp4x8_to_fp8x8(raw, lut):
    """Eight consecutive FP4 nibbles (one i32) -> v8 FP8-FNUZ MFMA operand."""
    raw = fx.Uint32(raw)
    lo, hi, sign_lo, sign_hi = lut
    sel_even = raw & 0x07070707  # magnitudes of nibbles 0,2,4,6
    sel_odd = (raw >> 4) & 0x07070707  # magnitudes of nibbles 1,3,5,7
    # byte = |value| | (nibble sign & 0x80-if-non-zero)
    even = (_perm_u32(sign_hi, sign_lo, sel_even) & (raw << 4)) | _perm_u32(
        hi, lo, sel_even
    )
    odd = (_perm_u32(sign_hi, sign_lo, sel_odd) & raw) | _perm_u32(hi, lo, sel_odd)
    lo32 = _perm_u32(odd, even, 0x05010400).bitcast(fx.Int32)
    hi32 = _perm_u32(odd, even, 0x07030602).bitcast(fx.Int32)
    return fx.Vector.from_elements([lo32, hi32], fx.Int32).bitcast(fx.Float8E4M3FNUZ)


# ---- fast decode (``fast_dec``, same scheme as test_a8w4_gemm1.py) ---------------
# W pre-transformed by ``prep_w2_fast``: byte j of a dword = (k_j, k_{j+4}) and no -0,
# so even/odd nibbles decode straight to K 0..3 / 4..7 and the sign is a bit move.
# fast_dec=2 also stores min(max-code, 11)*8 per group in the scale bytes (flushed
# groups zeroed in W) and the per-channel max codes (int32) right after the scales.


def _fp8_lut_fast(group_code, max_code):
    difference = max_code - group_code
    delta = fx.min(difference, 11).to(fx.Uint32) * fx.Uint32(0x08080808)
    lo = (fx.Uint32(0x6C686080) - delta) & fx.Uint32(0xFFFFFF00)
    hi = fx.Uint32(0x7C787470) - delta
    flushed = difference > 11
    lo = flushed.select(fx.Uint32(0), lo)
    hi = flushed.select(fx.Uint32(0), hi)
    smask = flushed.select(fx.Uint32(0), fx.Uint32(0x80808080))
    return lo, hi, smask


def _fp8_lut_predelta(delta):
    delta = fx.Uint32(delta)
    lo = (fx.Uint32(0x6C686080) - delta) & fx.Uint32(0xFFFFFF00)
    hi = fx.Uint32(0x7C787470) - delta
    return lo, hi, fx.Uint32(0x80808080)


def _fp4x8_to_fp8x8_fast(raw, lut):
    lo, hi, smask = lut
    raw = fx.Uint32(raw)
    even = ((raw << 4) & smask) | _perm_u32(hi, lo, raw & 0x07070707)
    odd = (raw & smask) | _perm_u32(hi, lo, (raw >> 4) & 0x07070707)
    return fx.Vector.from_elements(
        [even.bitcast(fx.Int32), odd.bitcast(fx.Int32)], fx.Int32
    ).bitcast(fx.Float8E4M3FNUZ)


def _channel_factor(exponent):
    """2**(exponent-127-5) as f32 (subnormal-safe)."""
    normal = ((exponent - 5).to(fx.Uint32) << 23).bitcast(fx.Float32)
    subnormal = (fx.Uint32(1) << (exponent + 17).to(fx.Uint32)).bitcast(fx.Float32)
    return (exponent >= 6).select(normal, subnormal)


def _fp8_frag8(v8):
    t = fx.make_rmem_tensor(fx.make_layout(8, 1), fx.Float8E4M3FNUZ)
    t.store(v8)
    return t


def _mma_fp8(mma_atom, acc, a8, b8):
    """acc += A(16x32) * B(32x16), one MFMA(16,16,32,fp8 FNUZ)."""
    fx.gemm(mma_atom, acc, _fp8_frag8(a8), _fp8_frag8(b8), acc)


# =============================================================================
# Epilogue: routing-weighted bf16x2 global atomic scatter (gfx942)
# =============================================================================


def _flat_buffer(arg, elem_ty, align):
    ptr = global_typed_ptr(arg, elem_ty, align=align)
    view = fx.Tensor(fx.make_view(ptr, fx.make_layout((1, 1), (1, 1))))
    return fx.rocdl.make_buffer_tensor(view, max_size=True)


def _atomic_add_bf16x2(out_base, elem_off, lo, hi):
    """``out[elem_off : elem_off+2] += (lo, hi)`` as one global_atomic_pk_add_bf16."""
    pk = Vec.from_elements([lo, hi], fx.Float32).to(fx.BFloat16)
    llvm.AtomicRMWOp(
        llvm.AtomicBinOp.fadd,
        _gep1(out_base, elem_off * fx.Int32(2)),
        _raw(pk),
        llvm.AtomicOrdering.monotonic,
        syncscope="agent",
        alignment=4,
    )


# @flyc.jit is LOAD-BEARING (the scatter epilogues): it AST-rewrites ``if token <
# i32_M`` into an scf.if. As a plain function the guard would run as a Python if
# (dropped at trace), so the atomic adds would fire on padding rows.
@flyc.jit
def _atomic_bf16_epilog(
    lds_base_i32, accm, arg_out, arg_stids, arg_sweights, m_row, n_block, wave, lane,
    i32_M, BM, N_OUT, BN,
):
    """Coalesced scatter through LDS: apply the routing weight in the MFMA layout,
    stage the tile as bf16 [BM, BN] (same bf16 rounding point as the port's f32
    staging, half the LDS), re-read it as (8 rows x 32 lanes x 2 cols) and
    ``global_atomic_pk_add_bf16`` into ``out[token, n]`` (128 B per row per op)."""
    # Row count comes from the accumulator list, not BM: a nested @flyc.jit call
    # does not re-specialize on Python int arguments (dyn_rows calls this with
    # several row variants in one kernel).
    M_CHUNKS = len(accm)
    M_REPS = M_CHUNKS * 2  # 256 threads = 8 rows x 32 lanes per pass
    n_per_wave = BN // 4
    num_acc_n = n_per_wave // 16
    s_count = BN // 64  # each pass covers 64 cols (32 lanes x 2)
    lane_div_16 = lane // fx.Int32(16)
    lane_mod_16 = lane % fx.Int32(16)
    lds_bf16 = lds_typed_ptr(lds_base_i32, T.bf16)

    tx = fx.Int32(gpu.thread_id("x"))
    m_lane = tx // fx.Int32(32)
    col_start = (tx % fx.Int32(32)) * fx.Int32(2)

    stids = _flat_buffer(arg_stids, T.i32, 4)
    sweights = _flat_buffer(arg_sweights, T.f32, 4)
    load_i32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Int32)
    load_f32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
    # gfx942 has no buffer_atomic_pk_add_bf16 -> global packed atomic.
    out_base = _global_base_ptr1(arg_out)

    def load_scalar(atom, src, index, elem_ty):
        frag = fx.make_rmem_tensor(1, elem_ty)
        fx.copy(atom, src[None, index], frag)
        return Vec(frag.load())[0]

    packed = []
    for mr in range_constexpr(M_REPS):
        sorted_pos = m_row + fx.Int32(mr * 8) + m_lane
        packed.append(load_scalar(load_i32, stids, sorted_pos, fx.Int32))

    # MFMA acc layout (lane: col L%16, rows (L//16)*4 + v) -> weighted bf16 in LDS.
    for i in range_constexpr(M_CHUNKS):
        row_base = fx.Int32(i * 16) + lane_div_16 * fx.Int32(4)
        w = [
            load_scalar(load_f32, sweights, m_row + row_base + fx.Int32(v), fx.Float32)
            for v in range_constexpr(4)
        ]
        for J in range_constexpr(num_acc_n):
            col = wave * fx.Int32(n_per_wave) + fx.Int32(J * 16) + lane_mod_16
            vec = Vec(accm[i][J])
            bf4 = Vec.from_elements(
                [vec[v] * w[v] for v in range_constexpr(4)], fx.Float32
            ).to(fx.BFloat16)
            for v in range_constexpr(4):
                lds_bf16[(row_base + fx.Int32(v)) * fx.Int32(BN) + col] = bf4[v]

    gpu.barrier()
    # Drain the (long-landed) sorted-id loads before the first atomic: gfx9 has
    # one in-order vmcnt for loads and stores, so a token id consumed after
    # atomics were issued makes LLVM wait for those atomics (ATT: ~0.57 us/wave
    # of mid-epilogue vmcnt stalls serializing the atomic groups).
    rocdl.s_waitcnt(vmcnt=0)

    for mr in range_constexpr(M_REPS):
        row_in_block = fx.Int32(mr * 8) + m_lane
        token_id = packed[mr] & fx.Int32(0x00FFFFFF)
        if token_id < i32_M:  # padding rows carry a sentinel token id
            row_base_addr = (
                token_id * fx.Int32(N_OUT) + n_block * fx.Int32(BN) + col_start
            )
            for s in range_constexpr(s_count):
                idx0 = row_in_block * fx.Int32(BN) + col_start + fx.Int32(s * 64)
                pk = lds_vec_load(
                    lds_base_i32,
                    idx0 * fx.Int32(2),
                    Vec.make_type(2, fx.BFloat16),
                    fx.BFloat16,
                    align=4,
                )
                llvm.AtomicRMWOp(
                    llvm.AtomicBinOp.fadd,
                    _gep1(out_base, (row_base_addr + fx.Int32(s * 64)) * fx.Int32(2)),
                    _raw(pk),
                    llvm.AtomicOrdering.monotonic,
                    syncscope="agent",
                    alignment=4,
                )


@flyc.jit
def _direct64_scatter_tile(accm, ids, arg_out, col_w, lane, i32_M, N_OUT):
    """LDS-free scatter with 64 B row segments (NTILE ``direct64`` epilogue).

    ``accm[mi][ni]``: the wave's two 16-col blocks (ni = 0, 1; already weighted),
    MFMA layout lane L = col L%16 of rows (L//16)*4 + v. Per row v one
    shuffle_xor(1): the even lane of a pair keeps block 0 (cols c, c+1), the odd
    lane block 1 (cols c-1, c), so the 16 lanes of a row group cover the wave's
    32 contiguous columns -> 16 x bf16x2 = 64 B per row per atomic instruction
    (32 B segments halve the atomic throughput, >= 64 B is full rate:
    experiments/atomic_bw). ``ids[mi][v]``: sorted id of row mi*16 +
    (L//16)*4 + v; ``col_w``: first column of the wave."""
    odd = lane & fx.Int32(1)
    is_odd = odd == fx.Int32(1)
    col = col_w + odd * fx.Int32(16) + (lane % fx.Int32(16)) - odd
    out_base = _global_base_ptr1(arg_out)
    for mi in range_constexpr(len(accm)):
        v0 = Vec(accm[mi][0])
        v1 = Vec(accm[mi][1])
        for k in range_constexpr(4):
            recv = is_odd.select(v0[k], v1[k]).shuffle_xor(fx.Int32(1), fx.Int32(64))
            lo = is_odd.select(recv, v0[k])
            hi = is_odd.select(v1[k], recv)
            tok = fx.Int32(ids[mi][k]) & fx.Int32(0x00FFFFFF)
            if tok < i32_M:
                _atomic_add_bf16x2(out_base, tok * fx.Int32(N_OUT) + col, lo, hi)


# =============================================================================
# Kernel
# =============================================================================


@functools.cache
def compile_a8w4_gemm2(
    *,
    D_HIDDEN,
    D_INTER,
    NE,
    BM=64,
    TILE_N=128,
    TILE_K=256,
    b_cache_mod=2,
    xcd_swizzle=0,
    waves_per_eu=None,
    flush_tiny=True,
    epilog="lds",
    dyn_rows=False,
    A_GROUP=0,
    fast_dec=0,
    NTILE=0,
    ROW_VGPR=False,
):
    """Build the A8W4 stage2 launcher (N = model_dim, K = inter_dim).

    ``NTILE`` (0 = the original one-tile-per-CTA body): each CTA owns one m-block
    and NTILE adjacent n-blocks (grid / NTILE). A is staged to LDS once and reused;
    the row-static data (ratio s0/s1 of the two A-scale groups, s1 * route weight,
    sorted id) is loaded once and kept in LDS (+12 B/row) or, with ``ROW_VGPR``,
    in VGPRs, so one accumulator set is rescaled at the group boundary instead of
    two; all A loads are issued before any LDS write; W / scales of n-tile t+1 are
    prefetched before tile t computes, so they land under tile t's MFMA +
    epilogue. NTILE uses the ``direct64`` epilogue (LDS = A tile).

    ``A_GROUP``: 0 = one A scale per sorted row (``a_scale`` f32 [sorted_size]);
    128 = one per (sorted row, 128-K group) (``a_scale`` f32 [sorted_size, K/128],
    as written by gemm1's fused-quant ``q8s`` epilogue). Each group accumulates
    into its own accumulators, combined at dequant as sum_g acc_g * s_g (a single
    accumulator rescaled by s_g/s_{g+1} at group boundaries needed more VGPRs:
    the compiler hoists the scale loads + divides; 89 vs 77 at bm32).

    ``epilog``: ``"lds"`` (default, NTILE=0) stages a weighted bf16 tile in LDS
    (BM*TILE_N*2, half the port's f32 staging) for 128 B contiguous atomics;
    ``"direct64"`` (NTILE only) scatters straight from the MFMA accumulator
    (``_direct64_scatter_tile``: no LDS staging, 64 B-contiguous atomics).

    ``dyn_rows``: each WG computes only the leading 16-row sub-blocks of its sorted
    block that hold valid rows (valid rows are contiguous from the block start),
    so A loads / MFMA / dequant / scatter skip all-padding sub-blocks. All row
    variants (16, 32, ..., BM) are compiled in; W traffic is unchanged.

    Launch args (raw device pointers): ``a`` fp8 [sorted_size, K], ``a_scale`` f32
    [sorted_size], ``w`` mxfp4 [E, N, K/2] (standard preshuffle), ``w_scale`` e8m0
    (shuffled), ``sorted_expert_ids``, ``num_valid_ids``, ``sorted_ids``,
    ``sorted_weights``, ``out`` bf16 [tokens, N] (pre-zeroed, atomically
    accumulated), then ``tokens``, four unused scalars (``dbg``, ``tbl``,
    ``rank``, ``epoch``: they only keep the kernarg layout, and so the ISA, of the
    tuned build; they go with the host-launch rewrite), ``grid`` and the stream.
    """
    K = D_INTER
    N_OUT = D_HIDDEN
    NUM_WAVES = 4
    assert K % TILE_K == 0 and TILE_K % 128 == 0, f"bad K={K} / TILE_K={TILE_K}"
    assert N_OUT % TILE_N == 0, f"model_dim must be a multiple of TILE_N={TILE_N}"
    assert TILE_N % (NUM_WAVES * 16) == 0, "TILE_N must be a multiple of 64"
    assert BM % 16 == 0, f"BM must be a multiple of 16, got {BM}"
    assert (BM * TILE_K) % (256 * 16) == 0, "A tile must be whole 16 B per thread"
    assert epilog == ("direct64" if NTILE else "lds"), f"bad epilog {epilog!r}"
    assert epilog != "direct64" or TILE_N // NUM_WAVES == 32, (
        "direct64: 32 columns per wave")
    assert fast_dec in (0, 1, 2) and (flush_tiny or not fast_dec), fast_dec
    assert A_GROUP in (0, 128), f"A_GROUP must be 0 or 128, got {A_GROUP}"
    A_GROUPS = K // 128 if A_GROUP else 1  # A scale groups (= accumulator sets)

    NUM_N_BLOCKS = N_OUT // TILE_N
    K_TILES = K // TILE_K
    N_REPEAT = TILE_N // NUM_WAVES // 16  # 16-col MFMA blocks per wave
    K_STEPS = TILE_K // 32  # MFMA K32 steps per tile
    K_GROUPS = TILE_K // 128  # 128-K groups: one W dwordx4 + one scale byte each
    # LDS epilogue: the A tile and the bf16 staging tile never coexist (barrier
    # between) -> max, not sum. direct64: A tile only.
    LDS_BYTES = (
        BM * TILE_K if epilog == "direct64" else max(BM * TILE_K, BM * TILE_N * 2)
    )
    if NTILE:
        assert NUM_N_BLOCKS % NTILE == 0, f"NTILE={NTILE} must divide {NUM_N_BLOCKS}"
        assert K_TILES == 1, "NTILE keeps the whole A (one K tile) in LDS"
    ROW_OFF = LDS_BYTES  # NTILE row data: f32 ratio[BM], f32 fin[BM], i32 id[BM]
    if NTILE:
        assert K // 128 <= 2, "NTILE rescales one accumulator across <= 2 A groups"
        if not ROW_VGPR:  # ROW_VGPR: row data per lane in VGPRs, no LDS
            LDS_BYTES += BM * 12
    NBG = NUM_N_BLOCKS // max(NTILE, 1)  # CTAs per m-block

    # W (mxfp4) N-major preshuffle in bytes: (N/16, K/128, klane=4, nlane=16, kpack=16).
    W_K0 = K // 128
    W_BYTES = NE * N_OUT * K // 2
    # E8M0 scale preshuffle (dwords): (N/32, Kpad/256, klane=4, nlane=16); each dword
    # packs [kpack=2] x [npack=2] bytes.
    SCALE_K_PAD = ((K + 255) // 256) * 256
    SC_STRIDE_N0 = (SCALE_K_PAD // 256) * 64
    WS_BYTES = NE * N_OUT * (SCALE_K_PAD // 32)

    ftz_tag = "_ftz" if flush_tiny else ""
    wpe_tag = f"_w{waves_per_eu}" if waves_per_eu else ""
    kname = (
        f"a8w4_gemm2_h{N_OUT}_i{K}_ne{NE}_bm{BM}_tn{TILE_N}_tk{TILE_K}"
        f"_bcm{b_cache_mod}_xcd{xcd_swizzle}{wpe_tag}{ftz_tag}_{epilog}"
        f"{f'_fd{fast_dec}' if fast_dec else ''}"
        f"{'_dyn' if dyn_rows else ''}{f'_ag{A_GROUP}' if A_GROUP else ''}"
        f"{f'_nt{NTILE}' if NTILE else ''}{'_rv' if NTILE and ROW_VGPR else ''}"
    )

    def _body(
        lds_ptr,
        arg_a,
        arg_a_scale,
        arg_w,
        arg_w_scale,
        arg_eids,
        arg_stids,
        arg_sweights,
        arg_out,
        i32_M,
        tile,
        lane,
        wave,
    ):
        lane_div_16 = lane // fx.Int32(16)
        lane_mod_16 = lane % fx.Int32(16)

        # ---- tile decode: m-block (one expert) x n-block (model_dim tile) -------
        m_block = tile // fx.Int32(NBG)
        n_block = tile % fx.Int32(NBG)  # NTILE: n-group, first n-block n_block*NTILE
        e = rocdl.readfirstlane(T.i32, _raw(_global_i32_at(arg_eids, m_block)))
        m_row = m_block * fx.Int32(BM)  # first sorted row of this m-block
        by_n = n_block * fx.Int32(TILE_N)

        # ---- W operand resources ----------------------------------------------
        expert_off = e * fx.Int32(N_OUT)
        w_tiles = _global_i32_buffer_tiles(arg_w, min(W_BYTES, 0xFFFFFFFF), 4)
        w_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(b_cache_mod), fx.Int32)
        ws_tiles = _global_i32_buffer_tiles(arg_w_scale, min(WS_BYTES, 0xFFFFFFFF), 1)
        ws_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(0), fx.Int32)

        # ---- A operand: BM sorted rows -> VGPRs -> LDS (single slot) ----------
        # Rows < cumsum0 always exist in the [sorted_size, K] buffer; the grid never
        # visits rows beyond it, so an unclamped resource is fine.
        x_tiles = fx.logical_divide(
            _global_i32_buffer_view(arg_a, fx.Int64(0xFFFFFFFF)), fx.make_layout(4, 1)
        )
        x_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(b_cache_mod), fx.Int32)
        lds_flat = fx.make_view(
            fx.recast_iter(fx.Int32, lds_ptr), fx.make_layout(LDS_BYTES // 4, 1)
        )
        lds_tiles16 = fx.logical_divide(lds_flat, fx.make_layout(4, 1))
        lds_tiles8 = fx.logical_divide(lds_flat, fx.make_layout(2, 1))
        lds_write_atom = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Int32)
        lds_read_atom = fx.make_copy_atom(fx.UniversalCopy64b(), fx.Int32)

        def _swizzle(row, col_bytes):
            # A-LDS bank-conflict XOR swizzle; applied to the GMEM source column on
            # the write (LDS destination linear) and to the LDS column on the read.
            return col_bytes ^ ((row & fx.Int32(TILE_K // 16 - 1)) * fx.Int32(16))

        tx = fx.Int32(gpu.thread_id("x"))

        # ---- W columns: wave owns TILE_N/4 output channels ----------------------
        # Addresses are K-invariant per lane; a K tile only adds a static offset.
        # Two 16-col blocks of one 32-row scale block share a scale dword (npack
        # halves) -> load it once.
        share_scale = (TILE_N // NUM_WAVES) % 32 == 0
        n_wave_base = wave * fx.Int32(TILE_N // NUM_WAVES)

        def w_cols(by_n):
            w_base, sc_base, sc_key, sc_npack = [], {}, [], []
            for ni in range_constexpr(N_REPEAT):
                n0 = expert_off + by_n + n_wave_base + fx.Int32(ni * 16)
                w_base.append(
                    (n0 // fx.Int32(16)) * fx.Int32(W_K0 * 64)
                    + lane_div_16 * fx.Int32(16)
                    + lane_mod_16
                )
                key = ni // 2 if share_scale else ni
                sc_key.append(key)
                sc_npack.append((n0 // fx.Int32(16)) % fx.Int32(2))
                if key not in sc_base:
                    sc_base[key] = (
                        (n0 // fx.Int32(32)) * fx.Int32(SC_STRIDE_N0)
                        + lane_div_16 * fx.Int32(16)
                        + lane_mod_16
                    )
            return w_base, sc_base, sc_key, sc_npack

        # ---- e8m0 codes for the whole K (short: K = inter_dim) ------------------
        # Lane L holds 32-K group 4*G + L//16 of every 128-K group G; the dword of
        # group G is (G//2) along K, byte (G%2)*2 + npack.
        def scale_issue(sc_base):
            return {
                (key, G2): _buffer_i32_scalar_read(
                    ws_tiles, sc_base[key] + fx.Int32(G2 * 64), ws_atom
                )
                for key in sc_base
                for G2 in range((K // 128 + 1) // 2)
            }

        def exp_issue(by_n, ni):
            # fast_dec=2: per-channel max codes stored after the scale bytes
            n_lane = expert_off + by_n + n_wave_base + fx.Int32(ni * 16) + lane_mod_16
            return fx.Int32(
                _global_i32_at(fx.Int64(arg_w_scale) + fx.Int64(WS_BYTES), n_lane)
            )

        def scale_codes(packed, sc_key, sc_npack, by_n, exp_pre=None):
            code = {}  # code[ni, G]: this lane's e8m0 byte of 128-K group G
            for ni in range_constexpr(N_REPEAT):
                for G in range_constexpr(K // 128):
                    if const_expr(fast_dec == 2):
                        # byte (G%2)*2 + npack = delta*8, broadcast by one v_perm
                        p = packed[sc_key[ni], G // 2]
                        sel = (fx.Int32((G % 2) * 2) + sc_npack[ni]) * fx.Int32(
                            0x01010101
                        )
                        code[ni, G] = _perm_u32(p, p, sel)
                        continue
                    shift = fx.Int32((G % 2) * 16) + sc_npack[ni] * fx.Int32(8)
                    code[ni, G] = packed[sc_key[ni], G // 2].shrui(shift) & fx.Int32(
                        0xFF
                    )
            # Per-output-channel max e8m0 code over K: lane max, then across the 4
            # lanes (L//16) that share the column (fast_dec=2: stored after the scales).
            exp_n = []
            for ni in range_constexpr(N_REPEAT):
                if const_expr(fast_dec == 2):
                    exp_n.append(exp_issue(by_n, ni) if exp_pre is None else exp_pre[ni])
                    continue
                mx = code[ni, 0]
                for G in range_constexpr(1, K // 128):
                    mx = (code[ni, G] > mx).select(code[ni, G], mx)
                for off in (16, 32):
                    peer = mx.shuffle_xor(fx.Int32(off), fx.Int32(64))
                    mx = (peer > mx).select(peer, mx)
                exp_n.append(mx)
            return code, exp_n

        if const_expr(not NTILE):
            w_base, sc_base, sc_key, sc_npack = w_cols(by_n)
            code, exp_n = scale_codes(scale_issue(sc_base), sc_key, sc_npack, by_n)

        mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, fx.Float8E4M3FNUZ))

        def run_rows(rows):
            """A load, K loop, dequant and scatter for the first ``rows`` rows."""
            M_REPEAT = rows // 16  # 16-row MFMA blocks
            A_LOADS = rows * TILE_K // (256 * 16)  # 16 B A copies per thread

            a_src_tile, a_dst_tile = [], []
            for i in range_constexpr(A_LOADS):
                idx_dw = tx * fx.Int32(4) + fx.Int32(i * 256 * 4)
                row = idx_dw // fx.Int32(TILE_K // 4)
                col_bytes = (idx_dw % fx.Int32(TILE_K // 4)) * fx.Int32(4)
                row_bytes = (m_row + row) * fx.Int32(K)  # A is indexed by sorted row
                a_src_tile.append((row_bytes + _swizzle(row, col_bytes)) // fx.Int32(16))
                a_dst_tile.append((row * fx.Int32(TILE_K) + col_bytes) // fx.Int32(16))

            def store_a_tile(base_k):
                k16 = fx.Int32(base_k // 16)
                for i in range_constexpr(A_LOADS):
                    r = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                    fx.copy(x_atom, fx.slice(x_tiles, (None, a_src_tile[i] + k16)), r)
                    fx.copy(lds_write_atom, r, fx.slice(lds_tiles16, (None, a_dst_tile[i])))

            def load_a(mi, ku):
                # Lane L of K-step ku: row L%16, K = (ku//4)*128 + (L//16)*32 + (ku%4)*8.
                row = lane_mod_16 + fx.Int32(mi * 16)
                col = lane_div_16 * fx.Int32(32) + fx.Int32((ku // 4) * 128 + (ku % 4) * 8)
                off = (row * fx.Int32(TILE_K) + _swizzle(row, col)) // fx.Int32(8)
                r = fx.make_rmem_tensor(fx.make_layout(2, 1), fx.Int32)
                fx.copy_atom_call(lds_read_atom, fx.slice(lds_tiles8, (None, off)), r)
                return fx.Vector(fx.memref_load_vec(r)).bitcast(fx.Float8E4M3FNUZ)

            # ---- accumulators (one set per A scale group) ---------------------------
            acc_layout = fx.make_layout(4, 1)
            accg = [
                [
                    [fx.make_rmem_tensor(acc_layout, fx.Float32) for _ in range(N_REPEAT)]
                    for _ in range(M_REPEAT)
                ]
                for _ in range(A_GROUPS)
            ]
            accm = accg[0]
            zero4 = fx.Vector.filled(4, 0.0, fx.Float32)
            for acc in accg:
                for mi in range_constexpr(M_REPEAT):
                    for ni in range_constexpr(N_REPEAT):
                        acc[mi][ni].store(zero4)

            as_tiles = _global_i32_buffer_tiles(arg_a_scale, fx.Int64(0xFFFFFFFF), 1)
            as_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Int32)
            # grouped: the A_GROUPS scales of a row are contiguous -> one load
            asg_tiles = _global_i32_buffer_tiles(
                arg_a_scale, fx.Int64(0xFFFFFFFF), A_GROUPS
            )
            asg_atom = (
                fx.make_copy_atom(fx.rocdl.BufferCopy64b(), fx.Int32)
                if A_GROUPS == 2
                else fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Int32)
                if A_GROUPS == 4
                else None
            )

            def a_scales_row(mi, ii):
                """[A_GROUPS] f32 scales of this lane's row ii of block mi."""
                row = m_row + fx.Int32(mi * 16 + ii) + lane_div_16 * fx.Int32(4)
                if const_expr(asg_atom is None):
                    return [
                        _buffer_i32_scalar_read(
                            as_tiles, row * fx.Int32(A_GROUPS) + fx.Int32(g), as_atom
                        ).bitcast(fx.Float32)
                        for g in range(A_GROUPS)
                    ]
                r = fx.make_rmem_tensor(fx.make_layout(A_GROUPS, 1), fx.Int32)
                fx.copy(asg_atom, fx.slice(asg_tiles, (None, row)), r)
                v = fx.Vector(fx.memref_load_vec(r))
                return [fx.Int32(v[g]).bitcast(fx.Float32) for g in range(A_GROUPS)]

            # ---- K loop (static: K = inter_dim is a few tiles) ----------------------
            for kt in range_constexpr(K_TILES):
                base_k = kt * TILE_K
                store_a_tile(base_k)
                raw = []  # raw[ni][g][j]: i32 with 8 fp4 for K-step g*4 + j
                for ni in range_constexpr(N_REPEAT):
                    per_g = []
                    for g in range_constexpr(K_GROUPS):
                        r = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                        idx = w_base[ni] + fx.Int32((base_k // 128 + g) * 64)
                        fx.copy(w_atom, fx.slice(w_tiles, (None, idx)), r)
                        v4 = fx.Vector(fx.memref_load_vec(r))
                        per_g.append([fx.Int32(v4[j]) for j in range(4)])
                    raw.append(per_g)
                rocdl.s_waitcnt(lgkmcnt=0)
                gpu.barrier()  # A(kt) visible in LDS
                for ni in range_constexpr(N_REPEAT):
                    luts = [
                        _fp8_lut_predelta(code[ni, base_k // 128 + g])
                        if fast_dec == 2
                        else _fp8_lut_fast(code[ni, base_k // 128 + g], exp_n[ni])
                        if fast_dec
                        else _fp8_lut(code[ni, base_k // 128 + g], exp_n[ni], flush_tiny)
                        for g in range_constexpr(K_GROUPS)
                    ]
                    dec = _fp4x8_to_fp8x8_fast if fast_dec else _fp4x8_to_fp8x8
                    for ku in range_constexpr(K_STEPS):
                        b8 = dec(raw[ni][ku // 4][ku % 4], luts[ku // 4])
                        gg = base_k // 128 + ku // 4  # 128-K group of this K step
                        acc = accg[gg if A_GROUP else 0]
                        for mi in range_constexpr(M_REPEAT):
                            _mma_fp8(mma_atom, acc[mi][ni], load_a(mi, ku), b8)
                gpu.barrier()  # A reads done before the slot (or the epilogue) reuses LDS

            # ---- dequant: sum_g acc_g * a_scale[row, g] * 2**(exp-132) --------------
            f_n = [_channel_factor(x) for x in exp_n]
            for mi in range_constexpr(M_REPEAT):
                per_row = [a_scales_row(mi, ii) for ii in range_constexpr(4)]
                rs = [
                    [per_row[ii][g] for ii in range_constexpr(4)]
                    for g in range_constexpr(A_GROUPS)
                ]
                for ni in range_constexpr(N_REPEAT):
                    if const_expr(A_GROUP):
                        vs = [Vec(accg[g][mi][ni].load()) for g in range(A_GROUPS)]
                        out4 = []
                        for ii in range_constexpr(4):
                            t = vs[0][ii] * rs[0][ii]
                            for g in range_constexpr(1, A_GROUPS):
                                t = t + vs[g][ii] * rs[g][ii]
                            out4.append(t * f_n[ni])
                    else:
                        v = Vec(accm[mi][ni].load())
                        out4 = [v[ii] * (rs[0][ii] * f_n[ni]) for ii in range_constexpr(4)]
                    accm[mi][ni].store(Vec.from_elements(out4, fx.Float32))

            # ---- epilogue: routing weight + bf16x2 atomic scatter -------------------
            accm_v = [
                [accm[mi][ni].load().ir_value() for ni in range(N_REPEAT)]
                for mi in range(M_REPEAT)
            ]
            _atomic_bf16_epilog(
                fx.Int32(fx.ptrtoint(lds_ptr)),
                accm_v,
                arg_out,
                arg_stids,
                arg_sweights,
                m_row,
                n_block,
                wave,
                lane,
                i32_M,
                BM,
                N_OUT,
                TILE_N,
            )


        # ---- NTILE body: one m-block x NTILE adjacent n-blocks ----------------------
        def _ld1(atom, src, index, elem_ty):
            frag = fx.make_rmem_tensor(1, elem_ty)
            fx.copy(atom, src[None, index], frag)
            return Vec(frag.load())[0]

        def run_rows_nt(rows):
            M_REPEAT = rows // 16
            A_LOADS = rows * TILE_K // (256 * 16)

            # A: every global load first (LLVM does not hoist a buffer load above an
            # earlier ds_write); the LDS writes go after the W / row loads below.
            a_regs = []
            for i in range_constexpr(A_LOADS):
                idx_dw = tx * fx.Int32(4) + fx.Int32(i * 256 * 4)
                row = idx_dw // fx.Int32(TILE_K // 4)
                col_bytes = (idx_dw % fx.Int32(TILE_K // 4)) * fx.Int32(4)
                src = ((m_row + row) * fx.Int32(K) + _swizzle(row, col_bytes)) // fx.Int32(16)
                r = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                fx.copy(x_atom, fx.slice(x_tiles, (None, src)), r)
                a_regs.append((r, (row * fx.Int32(TILE_K) + col_bytes) // fx.Int32(16)))

            def w_issue(t):
                """Loads of n-tile t: W (one K tile), scale dwords, max codes."""
                nb = n_block * fx.Int32(NTILE) + fx.Int32(t)
                by_n_t = nb * fx.Int32(TILE_N)
                wb, scb, sck, scn = w_cols(by_n_t)
                packed_t = scale_issue(scb)
                exp_t = (
                    [exp_issue(by_n_t, ni) for ni in range(N_REPEAT)]
                    if fast_dec == 2
                    else None
                )
                raw = []
                for ni in range_constexpr(N_REPEAT):
                    per_g = []
                    for g in range_constexpr(K_GROUPS):
                        r = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                        fx.copy(w_atom, fx.slice(w_tiles, (None, wb[ni] + fx.Int32(g * 64))), r)
                        v4 = fx.Vector(fx.memref_load_vec(r))
                        per_g.append([fx.Int32(v4[j]) for j in range(4)])
                    raw.append(per_g)
                return dict(nb=nb, by_n=by_n_t, packed=packed_t, sck=sck, scn=scn,
                            exp=exp_t, raw=raw)

            cur = w_issue(0)

            # Row-static data, once per CTA: per sorted row ratio = s0/s1 (A-scale
            # groups 0/1; the single accumulator is rescaled by it at the group
            # boundary), fin = s_last * route weight, and the sorted id.
            # ROW_VGPR: each lane loads its own rows into VGPRs (vector loads, no
            # LDS; LDS = A tile only). Else: cooperatively into LDS (+12 B/row),
            # 4 threads per row writing the same value.
            if const_expr(ROW_VGPR):
                # lane-distributed: lane L of every wave holds rows h*64 + L
                # (h < BM/64); a consumer fetches its row with one ds_bpermute
                # (LDS crossbar, no LDS space, ~3 VGPRs per 64 rows)
                rs_as = buffer_ops.create_buffer_resource_from_addr(_raw(fx.Int64(arg_a_scale)))
                rs_w = buffer_ops.create_buffer_resource_from_addr(_raw(fx.Int64(arg_sweights)))
                rs_id = buffer_ops.create_buffer_resource_from_addr(_raw(fx.Int64(arg_stids)))
                RH = -(-BM // 64)
                raw_s, raw_w, rid_h = [], [], []
                for h in range_constexpr(RH):
                    grow_h = m_row + fx.Int32(h * 64) + lane
                    if const_expr(A_GROUPS > 1):
                        v = Vec(buffer_ops.buffer_load(
                            rs_as, grow_h * fx.Int32(A_GROUPS), vec_width=A_GROUPS, dtype=T.i32))
                        raw_s.append([fx.Int32(v[g]) for g in range(A_GROUPS)])
                    else:
                        raw_s.append([fx.Int32(buffer_ops.buffer_load(
                            rs_as, grow_h, vec_width=1, dtype=T.i32))])
                    raw_w.append(fx.Int32(buffer_ops.buffer_load(rs_w, grow_h, vec_width=1, dtype=T.i32)))
                    rid_h.append(fx.Int32(buffer_ops.buffer_load(rs_id, grow_h, vec_width=1, dtype=T.i32)))
            else:
                rrow = tx % fx.Int32(BM)
                grow = m_row + rrow
                if const_expr(A_GROUP):
                    r = fx.make_rmem_tensor(fx.make_layout(A_GROUPS, 1), fx.Int32)
                    fx.copy(asg_atom_nt, fx.slice(asg_tiles_nt, (None, grow)), r)
                    v = fx.Vector(fx.memref_load_vec(r))
                    a_s = [fx.Int32(v[g]).bitcast(fx.Float32) for g in range(A_GROUPS)]
                else:
                    a_s = [_ld1(ld_f32, as_buf_nt, grow, fx.Float32)]
                r_w = _ld1(ld_f32, sweights_buf, grow, fx.Float32)
                r_id = _ld1(ld_i32, stids_buf, grow, fx.Int32)

            for r, dst in a_regs:
                fx.copy(lds_write_atom, r, fx.slice(lds_tiles16, (None, dst)))
            lds_i32 = fx.Int32(fx.ptrtoint(lds_ptr))
            if const_expr(not ROW_VGPR):
                row_f = lds_typed_ptr(lds_i32 + fx.Int32(ROW_OFF), T.f32)
                row_i = lds_typed_ptr(lds_i32 + fx.Int32(ROW_OFF), T.i32)
                s_last = a_s[-1]
                if const_expr(A_GROUP):
                    # an all-zero group quantizes to scale 0 (its acc part is 0 too)
                    s_last = (s_last == fx.Float32(0.0)).select(fx.Float32(1.0), s_last)
                    row_f[rrow] = a_s[0] / s_last
                row_f[fx.Int32(BM) + rrow] = s_last * r_w
                row_i[fx.Int32(2 * BM) + rrow] = r_id
            rocdl.s_waitcnt(lgkmcnt=0)
            gpu.barrier()  # A (+ LDS row data) visible (stay for every n-tile)

            if const_expr(ROW_VGPR):
                ratio_h, fin_h = [], []  # i32 bit patterns (ds_bpermute moves i32)
                for h in range_constexpr(RH):
                    w_h = raw_w[h].bitcast(fx.Float32)
                    s_l = raw_s[h][-1].bitcast(fx.Float32)
                    if const_expr(A_GROUP):
                        s_l = (s_l == fx.Float32(0.0)).select(fx.Float32(1.0), s_l)
                        ratio_h.append((raw_s[h][0].bitcast(fx.Float32) / s_l).bitcast(fx.Int32))
                    fin_h.append((s_l * w_h).bitcast(fx.Int32))

                def _fetch(q, row_static, row_lane):
                    """q[row // 64] at block row ``row_static + row_lane`` (row_lane <
                    64 - row_static % 64, so the half is static)."""
                    src = fx.Int32(row_static % 64) + row_lane
                    return fx.Int32(rocdl.ds_bpermute(T.i32, src * fx.Int32(4), q[row_static // 64]))

            def row4(slot, mi):
                """f32 row data of this lane's 4 acc rows of block mi."""
                if const_expr(ROW_VGPR):
                    q = ratio_h if slot == 0 else fin_h
                    return [
                        _fetch(q, mi * 16 + v, lane_div_16 * fx.Int32(4)).bitcast(fx.Float32)
                        for v in range(4)
                    ]
                byte = fx.Int32(ROW_OFF + slot * BM * 4 + mi * 64) + lane_div_16 * fx.Int32(16)
                return Vec(lds_vec_load(lds_i32, byte, Vec.make_type(4, fx.Float32), fx.Float32, align=16))

            def row_id(row):
                return lds_vec_load(
                    lds_i32, fx.Int32(ROW_OFF + 2 * BM * 4) + row * fx.Int32(4),
                    T.i32, fx.Int32, align=4,
                )

            def id_direct(mi, k):
                """direct64 scatter: sorted id of this lane's row k of block mi."""
                if const_expr(ROW_VGPR):
                    return _raw(_fetch(rid_h, mi * 16 + k, lane_div_16 * fx.Int32(4)))
                return _raw(row_id(fx.Int32(mi * 16 + k) + lane_div_16 * fx.Int32(4)))

            def load_a(mi, ku):
                row = lane_mod_16 + fx.Int32(mi * 16)
                col = lane_div_16 * fx.Int32(32) + fx.Int32((ku // 4) * 128 + (ku % 4) * 8)
                off = (row * fx.Int32(TILE_K) + _swizzle(row, col)) // fx.Int32(8)
                r = fx.make_rmem_tensor(fx.make_layout(2, 1), fx.Int32)
                fx.copy_atom_call(lds_read_atom, fx.slice(lds_tiles8, (None, off)), r)
                return fx.Vector(fx.memref_load_vec(r)).bitcast(fx.Float8E4M3FNUZ)

            zero4 = fx.Vector.filled(4, 0.0, fx.Float32)
            for t in range_constexpr(NTILE):
                # keep the scheduler from interleaving tile t with t+1 (without
                # the lds epilogue's barriers it hoists the next tile's work and
                # the VGPR count doubles: bm128 direct64 nt4 378 -> 1 wave/SIMD)
                rocdl.sched_barrier(0)
                nxt = w_issue(t + 1) if t + 1 < NTILE else None  # prefetch
                code_t, exp_t = scale_codes(
                    cur["packed"], cur["sck"], cur["scn"], cur["by_n"], cur["exp"]
                )
                acc = [
                    [fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Float32) for _ in range(N_REPEAT)]
                    for _ in range(M_REPEAT)
                ]
                for mi in range_constexpr(M_REPEAT):
                    for ni in range_constexpr(N_REPEAT):
                        acc[mi][ni].store(zero4)
                luts = [
                    [
                        _fp8_lut_predelta(code_t[ni, g])
                        if fast_dec == 2
                        else _fp8_lut_fast(code_t[ni, g], exp_t[ni])
                        if fast_dec
                        else _fp8_lut(code_t[ni, g], exp_t[ni], flush_tiny)
                        for g in range_constexpr(K_GROUPS)
                    ]
                    for ni in range_constexpr(N_REPEAT)
                ]
                dec = _fp4x8_to_fp8x8_fast if fast_dec else _fp4x8_to_fp8x8
                # ku outer: each A fragment is read from LDS once and feeds every
                # ni (half the A LDS traffic of the ni-outer order)
                for ku in range_constexpr(K_STEPS):
                    if const_expr(A_GROUP and ku == 4):
                        # group 0 -> group 1 boundary: acc = acc0 * s0 / s1
                        for mi in range_constexpr(M_REPEAT):
                            rt = row4(0, mi)
                            for ni in range_constexpr(N_REPEAT):
                                a4 = Vec(acc[mi][ni].load())
                                acc[mi][ni].store(
                                    Vec.from_elements([a4[v] * rt[v] for v in range(4)], fx.Float32)
                                )
                    b8s = [
                        dec(cur["raw"][ni][ku // 4][ku % 4], luts[ni][ku // 4])
                        for ni in range_constexpr(N_REPEAT)
                    ]
                    for mi in range_constexpr(M_REPEAT):
                        a8 = load_a(mi, ku)
                        for ni in range_constexpr(N_REPEAT):
                            _mma_fp8(mma_atom, acc[mi][ni], a8, b8s[ni])

                f_n = [_channel_factor(x) for x in exp_t]
                accm_v = []
                for mi in range_constexpr(M_REPEAT):
                    fin = row4(1, mi)
                    row_v = []
                    for ni in range_constexpr(N_REPEAT):
                        a4 = Vec(acc[mi][ni].load())
                        row_v.append(
                            Vec.from_elements(
                                [a4[v] * fin[v] * f_n[ni] for v in range(4)], fx.Float32
                            ).ir_value()
                        )
                    accm_v.append(row_v)

                _direct64_scatter_tile(
                    accm_v,
                    [[id_direct(mi, k) for k in range(4)] for mi in range(M_REPEAT)],
                    arg_out,
                    cur["by_n"] + n_wave_base,
                    lane,
                    i32_M,
                    N_OUT,
                )
                cur = nxt

        if const_expr(NTILE):
            stids_buf = _flat_buffer(arg_stids, T.i32, 4)
            sweights_buf = _flat_buffer(arg_sweights, T.f32, 4)
            ld_i32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Int32)
            ld_f32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
            as_buf_nt = _flat_buffer(arg_a_scale, T.f32, 4)
            asg_tiles_nt = _global_i32_buffer_tiles(
                arg_a_scale, fx.Int64(0xFFFFFFFF), A_GROUPS
            )
            asg_atom_nt = (
                fx.make_copy_atom(fx.rocdl.BufferCopy64b(), fx.Int32)
                if A_GROUPS == 2
                else fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Int32)
                if A_GROUPS == 4
                else None
            )
        run = run_rows_nt if NTILE else run_rows

        # ---- row-variant dispatch ------------------------------------------------
        if const_expr(dyn_rows):
            # Valid rows of a sorted block are contiguous from its start, so the
            # first 16-row sub-block whose first row is padding bounds the work.
            pad = {
                i: (
                    fx.Int32(
                        rocdl.readfirstlane(
                            T.i32,
                            _raw(_global_i32_at(arg_stids, m_row + fx.Int32(16 * i))),
                        )
                    )
                    & fx.Int32(0x00FFFFFF)
                )
                >= i32_M
                for i in range(1, BM // 16)
            }

            def _dispatch_from(i):
                # sub-blocks 0..i-1 hold valid rows; decide on sub-block i.
                if i == BM // 16:
                    run(BM)
                    return
                cond = pad[i]

                @flyc.jit
                def _dispatch():
                    if cond:
                        run(16 * i)
                    else:
                        _dispatch_from(i + 1)

                _dispatch()

            _dispatch_from(1)
        else:
            run(BM)

    @fx.struct
    class SharedStorage:
        raw: fx.Array[fx.Uint8, LDS_BYTES, 16]

    @flyc.jit
    def _entry(
        arg_a,
        arg_a_scale,
        arg_w,
        arg_w_scale,
        arg_eids,
        arg_cumsum,
        arg_stids,
        arg_sweights,
        arg_out,
        i32_M,
        arg_dbg,
        arg_tbl,
        i32_rank,
        i32_epoch,
    ):
        lds_ptr = fx.SharedAllocator().allocate(SharedStorage).peek().raw.ptr
        tx = fx.Int32(gpu.thread_id("x"))
        bx = fx.Int32(gpu.block_id("x"))
        lane = tx % fx.Int32(64)
        wave = rocdl.readfirstlane(T.i32, tx // fx.Int32(64))
        cumsum0 = _global_i32_at(arg_cumsum, fx.Int32(0))
        total_m_blocks = cumsum0 // fx.Int32(BM)
        bound = total_m_blocks * fx.Int32(NBG)

        # Bijective XCD round-robin over valid tiles (always on: stage2 is HBM-bound);
        # xcd_swizzle>0 also groups xcd_swizzle m-blocks per XCD for L2 locality.
        NXCD = 8
        xq = _udiv(bound, NXCD)
        xr = _umod(bound, NXCD)

        def _xcd(pid):
            xc = _umod(pid, NXCD)
            wgid = xc * xq + fx.min(xc, xr) + _udiv(pid, NXCD)
            if const_expr(xcd_swizzle <= 0):
                return wgid
            ng = fx.Int32(xcd_swizzle * NBG)
            first_m = (wgid // ng) * fx.Int32(xcd_swizzle)
            group_m = fx.min(total_m_blocks - first_m, fx.Int32(xcd_swizzle))
            wig = wgid % ng
            return (first_m + wig % group_m) * fx.Int32(NBG) + wig // group_m

        # `bx >= 0` is always true but dropping it reschedules the ISA; kept.
        if (bx >= fx.Int32(0)) & (bx < bound):
            _body(
                lds_ptr,
                arg_a,
                arg_a_scale,
                arg_w,
                arg_w_scale,
                arg_eids,
                arg_stids,
                arg_sweights,
                arg_out,
                i32_M,
                _xcd(bx),
                lane,
                wave,
            )

    @flyc.kernel(name=kname, known_block_size=[256, 1, 1])
    def a8w4_gemm2_kernel(
        arg_a: fx.Int64,
        arg_a_scale: fx.Int64,
        arg_w: fx.Int64,
        arg_w_scale: fx.Int64,
        arg_eids: fx.Int64,
        arg_cumsum: fx.Int64,
        arg_stids: fx.Int64,
        arg_sweights: fx.Int64,
        arg_out: fx.Int64,
        i32_M: fx.Int32,
        arg_dbg: fx.Int64,  # unused (kernarg layout of the tuned build)
        arg_tbl: fx.Int64,  # unused
        i32_rank: fx.Int32,  # unused
        i32_epoch: fx.Int32,  # unused
    ):
        _entry(
            arg_a,
            arg_a_scale,
            arg_w,
            arg_w_scale,
            arg_eids,
            arg_cumsum,
            arg_stids,
            arg_sweights,
            arg_out,
            i32_M,
            arg_dbg,
            arg_tbl,
            i32_rank,
            i32_epoch,
        )

    @flyc.jit
    def launch(
        arg_a: fx.Int64,
        arg_a_scale: fx.Int64,
        arg_w: fx.Int64,
        arg_w_scale: fx.Int64,
        arg_eids: fx.Int64,
        arg_cumsum: fx.Int64,
        arg_stids: fx.Int64,
        arg_sweights: fx.Int64,
        arg_out: fx.Int64,
        i32_M: fx.Int32,
        arg_dbg: fx.Int64,
        arg_tbl: fx.Int64,
        i32_rank: fx.Int32,
        i32_epoch: fx.Int32,
        i32_grid: fx.Int32,
        stream: fx.Stream,
    ):
        a8w4_gemm2_kernel(
            arg_a,
            arg_a_scale,
            arg_w,
            arg_w_scale,
            arg_eids,
            arg_cumsum,
            arg_stids,
            arg_sweights,
            arg_out,
            i32_M,
            arg_dbg,
            arg_tbl,
            i32_rank,
            i32_epoch,
            value_attrs=(
                {"rocdl.waves_per_eu": waves_per_eu} if waves_per_eu else None
            ),
        ).launch(grid=(fx.Int64(i32_grid), 1, 1), block=(256, 1, 1), stream=stream)

    return launch


def a8w4_gemm2(
    a_fp8,
    a_scale,
    w2,
    w2_scale,
    sorted_ids,
    sorted_weights,
    sorted_expert_ids,
    num_valid_ids,
    out,
    *,
    num_experts,
    tile_m=64,
    tile_n=128,
    tile_k=256,
    b_cache_mod=2,
    xcd_swizzle=0,
    flush_tiny=True,
    waves_per_eu=None,
    epilog="lds",
    dyn_rows=False,
    a_group=0,
    fast_dec=0,
    n_tiles=0,
    row_vgpr=False,
):
    """Host launcher. ``out`` is bf16 ``[tokens, model_dim]``; it is accumulated
    into (atomics), so the caller zeroes it (moe_sorting's moe_buf already is).
    ``a_group=128``: ``a_scale`` is f32 [sorted_size, inter_dim/128].
    ``fast_dec``: 1/2 need ``w2`` / ``w2_scale`` from ``prep_w2_fast``.
    ``n_tiles``: n-blocks per CTA (``NTILE`` of ``compile_a8w4_gemm2``; 0 = off;
    needs ``epilog="direct64"``)."""
    tokens, model_dim = out.shape
    inter_dim = a_fp8.shape[1]
    launch = compile_a8w4_gemm2(
        D_HIDDEN=model_dim,
        D_INTER=inter_dim,
        NE=num_experts,
        BM=tile_m,
        TILE_N=tile_n,
        TILE_K=tile_k,
        b_cache_mod=b_cache_mod,
        xcd_swizzle=xcd_swizzle,
        flush_tiny=flush_tiny,
        waves_per_eu=waves_per_eu,
        epilog=epilog,
        dyn_rows=dyn_rows,
        A_GROUP=a_group,
        fast_dec=fast_dec,
        NTILE=n_tiles,
        ROW_VGPR=row_vgpr,
    )
    grid = sorted_expert_ids.numel() * (model_dim // tile_n) // max(n_tiles, 1)
    _run_compiled(
        launch,
        a_fp8.data_ptr(),
        a_scale.data_ptr(),
        w2.data_ptr(),
        w2_scale.data_ptr(),
        sorted_expert_ids.data_ptr(),
        num_valid_ids.data_ptr(),
        sorted_ids.data_ptr(),
        sorted_weights.data_ptr(),
        out.data_ptr(),
        int(tokens),
        0,  # dbg, tbl, rank, epoch: unused kernel args
        0,
        0,
        0,
        int(grid),
        torch.cuda.current_stream(),
    )
    return out

