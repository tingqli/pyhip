# Package copy of doc-mimo/kernels/test_a8w4_gemm1.py (test harness removed, env knobs frozen).
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Standalone A8W4 MoE stage1 (gate+up GEMM + SiLU) FlyDSL kernel for gfx942.

Extracted from the a16w4 port (``moe_2stage_a16wmix/gemm1.py``, ``a_dtype="fp8r"``)
with only the configuration the tuned A8W4 build actually runs:

  * A  : FP8-FNUZ row-quantized activation ``[tokens, K]`` + f32 per-token scale
  * W1 : mxfp4 (E2M1 + per-1x32 E8M0), standard GGUU N-major preshuffle
  * MFMA(16,16,32,fp8) with fp4 -> fp8 decode through a per-32K-group byte LUT
    (+ a sign-mask LUT, so FNUZ -0 never needs a per-fragment fix-up)
  * runtime K loop, XOR-swizzled A-LDS, 4 waves split TILE_N; small BM prefetches
    tile kt+1 (A + W) during kt with double-buffered A-LDS (``pipeline``)
  * epilogue: acc * a_scale[token] * 2**(exp-132) -> SiLU(gate)*up -> bf16
    ``[sorted_size, inter_dim]`` written by SORTED position
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


# "hot" routing: 1/32 of the experts get this score bias, so they land in most
# tokens' top-k (a few experts own a large share of the routed rows).
# SiTUv2 test params: kimi-k3 production betas; the clamp bound sits inside the
# gate/up range of the random data (std ~64) so clamped and free values both occur.


# =============================================================================
# Device helpers: fp4 -> fp8 decode, channel factor, MFMA
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

    ``group_code`` is the 32-K group's E8M0 byte, ``max_code`` the channel max.
    Includes RNE subnormals for delta in 12..18; delta >= 19 rounds to zero.
    Returns ``(lo, hi, sign_lo, sign_hi)``: the magnitude table plus a same-shaped
    mask holding 0x80 for every non-zero entry, so the decode can attach the sign
    without ever producing 0x80 (FNUZ has no -0; 0x80 is NaN).
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


# ---- fast decode (``fast_dec``): W pre-transformed by ``prep_w1_fast_decode`` ----
# Each dword holds (k_j, k_{j+4}) in byte j (low, high nibble) and no -0 nibble, so
# even nibbles decode straight to K 0..3 and odd nibbles to K 4..7 (no interleave
# v_perm), and the sign is bit 3 of the nibble moved to bit 7 (no sign-table
# v_perm). flush_tiny only: then every non-zero magnitude maps to a non-zero byte
# unless the whole group is flushed, where the sign mask is zeroed instead.


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


def _fp4x8_to_fp8x8_fast(raw, lut):
    lo, hi, smask = lut
    raw = fx.Uint32(raw)
    even = ((raw << 4) & smask) | _perm_u32(hi, lo, raw & 0x07070707)
    odd = (raw & smask) | _perm_u32(hi, lo, (raw >> 4) & 0x07070707)
    return fx.Vector.from_elements(
        [even.bitcast(fx.Int32), odd.bitcast(fx.Int32)], fx.Int32
    ).bitcast(fx.Float8E4M3FNUZ)


def _fp8_lut_predelta(delta):
    """``fast_dec=2``: the scale byte already holds min(max-code, 11)*8 (flushed
    groups are zeroed in W by ``prep_w1_predelta``), broadcast to all 4 bytes."""
    delta = fx.Uint32(delta)
    lo = (fx.Uint32(0x6C686080) - delta) & fx.Uint32(0xFFFFFF00)
    hi = fx.Uint32(0x7C787470) - delta
    return lo, hi, fx.Uint32(0x80808080)


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
# Kernel
# =============================================================================


@functools.cache
def compile_a8w4_gemm1(
    *,
    D_HIDDEN,
    D_INTER,
    NE,
    BM=64,
    TILE_N=128,
    TILE_K=256,
    b_cache_mod=1,
    xcd_swizzle=0,
    waves_per_eu=None,
    flush_tiny=True,
    pipeline=None,
    dyn_m=False,
    epilog="bf16",
    TOPK=None,
    NPES=1,
    w_layout="std",
    fast_dec=False,
    act="silu",
):
    """Build the A8W4 stage1 launcher.

    ``act``: ``"silu"`` or ``"situv2"`` (as gemm1.py; SiTUv2 beta / linear_beta /
    clamp limit are runtime f32 launch args, nothing baked).

    ``pipeline`` (default: ``rows <= 32``): prefetch tile kt+1 (A + W into VGPRs)
    while kt computes, double-buffered A-LDS, one barrier per tile. Wins in the
    memory-bound small-M regime; costs ~1 tile of staging VGPRs, so larger-BM
    (compute-bound prefill) tiles keep the single-slot loop.

    ``dyn_m``: sorting stays at block ``BM`` but each WG decides at runtime to
    compute only the first ``BM/2`` rows when sorted row ``BM/2`` of its block is
    already padding (valid rows of a block are contiguous from its start). Cold
    experts then cost a BM/2 tile while hot experts keep full BM tiles (fewer
    re-reads of their weights). Both row variants are compiled into the kernel.

    Launch args (raw device pointers): ``x`` fp8 [tokens, K], ``x_scale`` f32
    [tokens], ``w`` mxfp4 [E, 2*inter, K/2] (standard preshuffle), ``w_scale`` e8m0
    (shuffled), ``w_exp`` int32 [E, 2*inter] (``channel_exponents``),
    ``sorted_expert_ids``, ``num_valid_ids``, ``sorted_ids``, ``out`` bf16
    [sorted_size, inter], then ``tokens``, ``tbl``, ``rank``, ``epoch``, ``rows``,
    the five SiTUv2 scalars (``situ_args``), ``grid`` and the stream.

    Nothing batch-size dependent is baked in: ``rows`` (rows of the q8 output
    plane, below) and the flag-table stride (= ``grid``, one flag per tile) are
    launch args, so one binary serves every token count.

    ``epilog`` (experiment for the N-split + all-gather TP idea, moe-new-tp.md):
      * ``"bf16"`` (default): bf16 ``[sorted_size, inter]`` by sorted position.
      * ``"q8"``: SiLU output staged as bf16 in LDS, quantized to fp8 FNUZ with one
        f32 scale per (row, TILE_N columns), written UNSORTED (row = token*TOPK +
        slot, ``rows`` = tokens*TOPK) into ``out``: ``[rows, inter]`` fp8 then
        ``[rows, inter/TILE_N]`` f32 scales.
      * ``"q8push"``: same tile, pushed (sc0|sc1) into plane ``rank`` of every TP
        peer's symmetric buffer (``tbl``: peer bases), double-buffered by epoch
        parity; then flag ``[rank][tile] = epoch`` on every peer. Slot layout:
        ``[NPES][rows][inter]`` fp8, ``[NPES][rows][inter/TILE_N]`` f32 (slot =
        that rounded up to 256 B); flags ``[NPES][grid]`` i32 after the two slots.
        The layout is a contract with the peers: keep ``rows`` fixed per buffer.
      * ``"q8pushs"``: as q8push but rows by SORTED position (``rows`` =
        sorted_size; the layout gemm2 already reads A in; padding rows not sent)
        and scales ``[NPES][inter/TILE_N][rows]`` f32, staged in LDS and pushed
        as one contiguous BM*4 B run per tile instead of 4 B scattered stores.
      * ``"q8s"``: as q8 (local) but rows by SORTED position (``rows`` =
        sorted_size): ``out`` = ``[sorted_size, inter]`` fp8 then
        ``[sorted_size, inter/TILE_N]`` f32, i.e. gemm2's A + per-128-K-group scale
        (``a8w4_gemm2(..., a_group=128)``); replaces the separate row-quant kernel.

    ``w_layout``: ``"std"`` = standard preshuffle; ``"tile"`` = W / scales retiled
    by ``retile_w1`` so each (expert, n-block, K tile) is one contiguous run
    (TILE_N*TILE_K bytes of W + TILE_N*TILE_K/16 bytes of scales).
    """
    assert epilog in ("bf16", "q8", "q8s", "q8push", "q8pushs"), epilog
    assert act in ("silu", "situv2"), act
    assert w_layout in ("std", "tile"), w_layout
    assert flush_tiny or not fast_dec, "fast_dec needs flush_tiny"
    TILE_W = w_layout == "tile"
    PUSH = epilog in ("q8push", "q8pushs")
    SORTED = epilog in ("q8s", "q8pushs")  # rows by sorted position
    SCALE_RUN = epilog == "q8pushs"  # scales via LDS as [NB][rows] runs
    K = D_HIDDEN
    INTER = D_INTER
    N_OUT = 2 * INTER  # gate | up
    NUM_WAVES = 4
    assert K % TILE_K == 0 and TILE_K % 256 == 0, f"bad K={K} / TILE_K={TILE_K}"
    assert N_OUT % 256 == 0, f"2*inter_dim must be a multiple of 256, got {N_OUT}"
    assert INTER % TILE_N == 0, f"inter_dim must be a multiple of TILE_N={TILE_N}"
    assert TILE_N % (NUM_WAVES * 16) == 0, f"TILE_N must be a multiple of 64"
    assert BM % 16 == 0, f"BM must be a multiple of 16, got {BM}"
    assert (BM * TILE_K) % (256 * 16) == 0, "A tile must be whole 16 B per thread"

    NUM_N_BLOCKS = INTER // TILE_N
    K_TILES = K // TILE_K
    N_REPEAT = TILE_N // NUM_WAVES // 16  # 16-col MFMA blocks per wave (gate and up)
    K_STEPS = TILE_K // 32  # MFMA K32 steps per tile
    K_GROUPS = TILE_K // 128  # 128-K groups: one W dwordx4 + one scale dword each
    # Row variants compiled into the kernel (rows per WG -> double-buffered or not).
    ROW_VARIANTS = [BM // 2, BM] if dyn_m else [BM]
    if dyn_m:
        assert (BM // 2) % 16 == 0 and (BM // 2 * TILE_K) % (256 * 16) == 0, (
            f"dyn_m needs BM/2 to be a valid tile, got BM={BM}"
        )

    def _pipe(rows):
        return bool(rows <= 32 if pipeline is None else pipeline) and K_TILES > 1

    LDS_BYTES = max(r * TILE_K * (2 if _pipe(r) else 1) for r in ROW_VARIANTS)
    if epilog != "bf16":
        LDS_BYTES = max(LDS_BYTES, BM * TILE_N * 2 + BM * 4)  # bf16 tile + scales
        assert TOPK, "q8 epilogues need TOPK"
    Q_TPR = TILE_N // 16  # threads per row in the quant pass (16 cols each)
    Q_RPP = 256 // Q_TPR  # rows per pass
    assert all(r % Q_RPP == 0 for r in ROW_VARIANTS) or epilog == "bf16"
    # ablation knob (experiment only): comma list of noscale, nopub, cm0, cm16,
    # one, selfonly. A tuple, not a set: FlyDSL keys its JIT cache on scalar/tuple
    # closure values only, a set would silently reuse another variant's binary.
    PUSH_VAR = ()  # frozen (was env-tunable in the experiment copy)

    # W (mxfp4) N-major preshuffle in bytes: (N/16, K/128, klane=4, nlane=16, kpack=16).
    W_K0 = K // 128
    W_BYTES = NE * N_OUT * K // 2
    # E8M0 scale preshuffle (dwords): (N/32, Kpad/256, klane=4, nlane=16); each dword
    # packs [kpack=2] x [npack=2] bytes.
    SCALE_K_PAD = ((K + 255) // 256) * 256
    SC_STRIDE_N0 = (SCALE_K_PAD // 256) * 64
    WS_BYTES = NE * N_OUT * (SCALE_K_PAD // 32)
    # "tile" layout (retile_w1): W 1 KB chunks as [E][NB][K/TILE_K][gu][TILE_N/16]
    # [TILE_K/128]; scale 256 B chunks as [E][NB][Kpad/256][gu][TILE_N/32].
    W_TILE_U16 = 2 * (TILE_N // 16) * K_GROUPS * 64  # one K tile, 16 B units
    SC_KB_DW = 2 * (TILE_N // 32) * 64  # one 256-K scale block, dwords
    if TILE_W:
        assert K % 256 == 0 and TILE_N % 32 == 0

    ftz_tag = "_ftz" if flush_tiny else ""
    wpe_tag = f"_w{waves_per_eu}" if waves_per_eu else ""
    pipe_tag = "".join("p" if _pipe(r) else "s" for r in ROW_VARIANTS)
    dyn_tag = "_dynm" if dyn_m else ""
    kname = (
        f"a8w4_gemm1_h{K}_i{INTER}_ne{NE}_bm{BM}_tn{TILE_N}_tk{TILE_K}"
        f"_bcm{b_cache_mod}_xcd{xcd_swizzle}{wpe_tag}{ftz_tag}{dyn_tag}_{pipe_tag}"
        + ("_wt" if TILE_W else "")
        + (f"_fd{int(fast_dec)}" if fast_dec else "")
        + ("" if act == "silu" else f"_{act}")
        + ("" if epilog == "bf16" else f"_{epilog}_k{TOPK}_w{NPES}")
        + "".join(f"_{v}" for v in sorted(PUSH_VAR))
    )

    def _body(
        lds_ptr,
        arg_x,
        arg_x_scale,
        arg_w,
        arg_w_scale,
        arg_w_exp,
        arg_eids,
        arg_cumsum,
        arg_mind,
        arg_out,
        i32_ntok,
        arg_tbl,
        i32_rank,
        i32_epoch,
        i32_rows,
        i32_tiles,
        situ,
        tile,
        lane,
        wave,
    ):
        lane_div_16 = lane // fx.Int32(16)
        lane_mod_16 = lane % fx.Int32(16)

        # ---- tile decode: m-block (one expert) x n-block (inter tile) ----------
        n_block = tile % fx.Int32(NUM_N_BLOCKS)
        m_block = tile // fx.Int32(NUM_N_BLOCKS)
        e = rocdl.readfirstlane(T.i32, _raw(_global_i32_at(arg_eids, m_block)))
        bx_m = m_block * fx.Int32(BM)  # first sorted row of this m-block
        by_n = n_block * fx.Int32(TILE_N)
        inter_i32 = fx.Int32(INTER)

        # ---- W operand resources ------------------------------------------------
        expert_off = e * fx.Int32(N_OUT)
        w_tiles = _global_i32_buffer_tiles(arg_w, min(W_BYTES, 0xFFFFFFFF), 4)
        w_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(b_cache_mod), fx.Int32)
        ws_tiles = _global_i32_buffer_tiles(arg_w_scale, min(WS_BYTES, 0xFFFFFFFF), 1)
        ws_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(0), fx.Int32)

        # ---- output resource: [cumsum0, INTER] bf16, OOB stores are dropped -----
        cumsum0 = _global_i32_at(arg_cumsum, fx.Int32(0))
        out_rsrc = buffer_ops.create_buffer_resource_from_addr(
            _raw(fx.Int64(arg_out)),
            num_records_bytes=_raw(fx.Int64(cumsum0) * fx.Int64(INTER * 2)),
        )

        # ---- A operand: gather BM sorted rows of x -> VGPRs -> LDS (2 slots) ----
        # Extent = the real [tokens, K] alloc so padding rows (token >= tokens) read 0.
        x_tiles = fx.logical_divide(
            _global_i32_buffer_view(arg_x, fx.Int64(i32_ntok) * fx.Int64(K)),
            fx.make_layout(4, 1),
        )
        x_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(2), fx.Int32)
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

        # ---- W columns: wave owns TILE_N/4 gate cols (+ the matching up cols) ---
        # gu: 0 = gate row n, 1 = up row n + INTER. Addresses are K-invariant per
        # lane; a K tile only adds a uniform offset. Two 16-col blocks of one
        # 32-row scale block share a scale dword (npack halves) -> load it once.
        share_scale = (TILE_N // NUM_WAVES) % 32 == 0
        n_wave_base = wave * fx.Int32(TILE_N // NUM_WAVES)
        out_cols = []
        w_base, sc_base, sc_key, sc_npack = {}, {}, {}, {}
        tile_id = e * fx.Int32(NUM_N_BLOCKS) + n_block  # "tile" layout run index
        for ni in range_constexpr(N_REPEAT):
            n_local = by_n + n_wave_base + fx.Int32(ni * 16)
            out_cols.append(n_local + lane_mod_16)
            j16 = (n_wave_base + fx.Int32(ni * 16)) // fx.Int32(16)  # n16 in tile
            for gu in range_constexpr(2):
                n0 = expert_off + n_local + (inter_i32 if gu else fx.Int32(0))
                if const_expr(TILE_W):
                    w_chunk = tile_id * fx.Int32(K_TILES * W_TILE_U16) + (
                        fx.Int32(gu * (TILE_N // 16)) + j16
                    ) * fx.Int32(K_GROUPS * 64)
                else:
                    w_chunk = (n0 // fx.Int32(16)) * fx.Int32(W_K0 * 64)
                w_base[gu, ni] = w_chunk + lane_div_16 * fx.Int32(16) + lane_mod_16
                key = (gu, ni // 2) if share_scale else (gu, ni)
                sc_key[gu, ni] = key
                sc_npack[gu, ni] = (n0 // fx.Int32(16)) % fx.Int32(2)
                if key not in sc_base:
                    if const_expr(TILE_W):
                        j32 = (n_wave_base + fx.Int32(ni * 16)) // fx.Int32(32)
                        s_chunk = tile_id * fx.Int32(
                            (SCALE_K_PAD // 256) * SC_KB_DW
                        ) + (fx.Int32(gu * (TILE_N // 32)) + j32) * fx.Int32(64)
                    else:
                        s_chunk = (n0 // fx.Int32(32)) * fx.Int32(SC_STRIDE_N0)
                    sc_base[key] = s_chunk + lane_div_16 * fx.Int32(16) + lane_mod_16

        # Register staging of one W tile (raw fp4 dwordx4 + packed e8m0 dwords); the
        # tile for kt+1 is loaded into them while kt computes from SSA copies.
        w_regs = {
            (gu, ni, g): fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
            for gu in range(2)
            for ni in range(N_REPEAT)
            for g in range(K_GROUPS)
        }
        s_regs = {
            (key, g): fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Int32)
            for key in sc_base
            for g in range(K_GROUPS)
        }

        def load_w_global(base_k):
            # e8m0 dword of 128-K group g sits in K-block (base_k + g*128) // 256.
            SKB = SC_KB_DW if TILE_W else 64  # dwords per 256-K scale block step
            s_off = (base_k // fx.Int32(256)) * fx.Int32(SKB)
            for (key, g), r in s_regs.items():
                idx = sc_base[key] + s_off + fx.Int32((g // 2) * SKB)
                fx.copy(ws_atom, fx.slice(ws_tiles, (None, idx)), r)
            if const_expr(TILE_W):
                w_off = (base_k // fx.Int32(TILE_K)) * fx.Int32(W_TILE_U16)
            else:
                w_off = (base_k // fx.Int32(128)) * fx.Int32(64)
            for (gu, ni, g), r in w_regs.items():
                idx = w_base[gu, ni] + w_off + fx.Int32(g * 64)
                fx.copy(w_atom, fx.slice(w_tiles, (None, idx)), r)

        def read_w():
            # raw[gu, ni, g][j]: i32 with 8 fp4 for K-step g*4 + j.
            # code[gu, ni, g]: e8m0 byte (kpack = g%2, npack = column parity).
            raw, code = {}, {}
            packed = {
                k: fx.Int32(fx.Vector(fx.memref_load_vec(r))[0])
                for k, r in s_regs.items()
            }
            for (gu, ni, g), r in w_regs.items():
                v4 = fx.Vector(fx.memref_load_vec(r))
                raw[gu, ni, g] = [fx.Int32(v4[j]) for j in range(4)]
                if const_expr(fast_dec >= 2):
                    # byte (g%2)*2 + npack of the dword, broadcast by one v_perm
                    p = packed[sc_key[gu, ni], g]
                    sel = (fx.Int32((g % 2) * 2) + sc_npack[gu, ni]) * fx.Int32(
                        0x01010101
                    )
                    code[gu, ni, g] = _perm_u32(p, p, sel)
                    continue
                shift = fx.Int32((g % 2) * 16) + sc_npack[gu, ni] * fx.Int32(8)
                code[gu, ni, g] = packed[sc_key[gu, ni], g].shrui(shift) & fx.Int32(
                    0xFF
                )
            return raw, code

        # Per-output-channel max e8m0 exponent of this lane's gate / up column.
        exp_gate = [
            fx.Int32(_global_i32_at(arg_w_exp, expert_off + c)) for c in out_cols
        ]
        exp_up = [
            fx.Int32(_global_i32_at(arg_w_exp, expert_off + inter_i32 + c))
            for c in out_cols
        ]

        mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, fx.Float8E4M3FNUZ))
        exps = (exp_gate, exp_up)
        xs_tiles = _global_i32_buffer_tiles(
            arg_x_scale, fx.Int64(i32_ntok) * fx.Int64(4), 1
        )
        xs_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Int32)

        def run_rows(rows):
            """K loop + epilogue for the first ``rows`` rows of this m-block."""
            M_REPEAT = rows // 16  # 16-row MFMA blocks
            A_LOADS = rows * TILE_K // (256 * 16)  # 16 B A copies per thread / tile
            SLOT_BYTES = rows * TILE_K  # one fp8 A tile
            PIPE = _pipe(rows)

            # Per-thread copy slots, K-invariant (in 16 B tile units): source =
            # gathered token row (+ XOR-swizzled column), destination = linear slot.
            a_src_tile, a_dst_tile = [], []
            for i in range_constexpr(A_LOADS):
                idx_dw = tx * fx.Int32(4) + fx.Int32(i * 256 * 4)
                row = idx_dw // fx.Int32(TILE_K // 4)
                col_bytes = (idx_dw % fx.Int32(TILE_K // 4)) * fx.Int32(4)
                # sorted_ids: token id in the low 24 bits, topk slot in the high 8.
                fused = fx.Int32(_global_i32_at(arg_mind, bx_m + row))
                row_bytes = (fused & fx.Int32(0x00FFFFFF)) * fx.Int32(K)
                a_src_tile.append(
                    (row_bytes + _swizzle(row, col_bytes)) // fx.Int32(16)
                )
                a_dst_tile.append((row * fx.Int32(TILE_K) + col_bytes) // fx.Int32(16))
            a_regs = [
                fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                for _ in range(A_LOADS)
            ]

            def load_a_global(base_k):
                k16 = base_k // fx.Int32(16)
                for i in range_constexpr(A_LOADS):
                    fx.copy(
                        x_atom, fx.slice(x_tiles, (None, a_src_tile[i] + k16)), a_regs[i]
                    )

            def store_a_lds(slot):
                s16 = fx.Int32(slot) * fx.Int32(SLOT_BYTES // 16)
                for i in range_constexpr(A_LOADS):
                    fx.copy(
                        lds_write_atom,
                        a_regs[i],
                        fx.slice(lds_tiles16, (None, a_dst_tile[i] + s16)),
                    )

            def load_a(mi, ku, slot):
                # Lane L of K-step ku: row L%16, K = (ku//4)*128 + (L//16)*32 + (ku%4)*8.
                row = lane_mod_16 + fx.Int32(mi * 16)
                col = lane_div_16 * fx.Int32(32) + fx.Int32(
                    (ku // 4) * 128 + (ku % 4) * 8
                )
                off = (row * fx.Int32(TILE_K) + _swizzle(row, col)) // fx.Int32(8)
                off = off + fx.Int32(slot) * fx.Int32(SLOT_BYTES // 8)
                r = fx.make_rmem_tensor(fx.make_layout(2, 1), fx.Int32)
                fx.copy_atom_call(lds_read_atom, fx.slice(lds_tiles8, (None, off)), r)
                return fx.Vector(fx.memref_load_vec(r)).bitcast(fx.Float8E4M3FNUZ)

            # ---- accumulators -------------------------------------------------
            acc_layout = fx.make_layout(4, 1)
            acc_gate = [
                [fx.make_rmem_tensor(acc_layout, fx.Float32) for _ in range(N_REPEAT)]
                for _ in range(M_REPEAT)
            ]
            acc_up = [
                [fx.make_rmem_tensor(acc_layout, fx.Float32) for _ in range(N_REPEAT)]
                for _ in range(M_REPEAT)
            ]
            zero4 = fx.Vector.filled(4, 0.0, fx.Float32)
            for mi in range_constexpr(M_REPEAT):
                for ni in range_constexpr(N_REPEAT):
                    acc_gate[mi][ni].store(zero4)
                    acc_up[mi][ni].store(zero4)
            accs = (acc_gate, acc_up)

            def compute_tile(w_tile, slot):
                raw, code = w_tile
                for ni in range_constexpr(N_REPEAT):
                    luts = [
                        [
                            _fp8_lut_predelta(code[gu, ni, g])
                            if fast_dec >= 2
                            else _fp8_lut_fast(code[gu, ni, g], exps[gu][ni])
                            if fast_dec
                            else _fp8_lut(code[gu, ni, g], exps[gu][ni], flush_tiny)
                            for g in range_constexpr(K_GROUPS)
                        ]
                        for gu in range_constexpr(2)
                    ]
                    dec = _fp4x8_to_fp8x8_fast if fast_dec else _fp4x8_to_fp8x8
                    for ku in range_constexpr(K_STEPS):
                        g, j = ku // 4, ku % 4
                        bs = [
                            dec(raw[gu, ni, g][j], luts[gu][g])
                            for gu in range_constexpr(2)
                        ]
                        # A is re-read from LDS per N block (cheap) instead of holding
                        # the whole BM x TILE_K tile in VGPRs.
                        for mi in range_constexpr(M_REPEAT):
                            a8 = load_a(mi, ku, slot)
                            for gu in range_constexpr(2):
                                _mma_fp8(mma_atom, accs[gu][mi][ni], a8, bs[gu])

            def stage(kt, slot, prefetch):
                """One K tile: A(kt) is in LDS ``slot``, W(kt) in the W registers.

                With ``prefetch``, A(kt+1)/W(kt+1) global loads are issued before the
                MFMA cluster (A first, so its ds_write only waits on A), and A(kt+1) is
                written to the other slot afterwards. The single barrier per tile
                orders both the A(kt) writes before the reads and the reads of a slot
                before its overwrite two tiles later.
                """
                rocdl.s_waitcnt(lgkmcnt=0)
                gpu.barrier()
                w_tile = read_w()
                if const_expr(prefetch):
                    next_k = (kt + fx.Int32(1)) * fx.Int32(TILE_K)
                    load_a_global(next_k)
                    load_w_global(next_k)
                    # Keep the prefetch issue ahead of the MFMA cluster (the machine
                    # scheduler otherwise sinks it to the end of the tile).
                    rocdl.sched_barrier(0)
                compute_tile(w_tile, slot)
                if const_expr(prefetch):
                    store_a_lds(fx.Int32(1) - slot)

            # ---- main K loop ------------------------------------------------------
            # range() lowers to scf.for only inside an AST-rewritten (@flyc.jit) function;
            # accumulators / staging registers are rmem tensors mutated in place.
            if const_expr(PIPE):
                # prologue (tile 0) + runtime scf.for (prefetching) + last tile
                load_a_global(fx.Int32(0))
                load_w_global(fx.Int32(0))
                store_a_lds(0)

                @flyc.jit
                def _k_loop_pipe(k_first):
                    for kt in range(K_TILES - 1):
                        kt_i = k_first + fx.Int32(kt)
                        stage(kt_i, kt_i % fx.Int32(2), True)

                _k_loop_pipe(fx.Int32(0))
                stage(fx.Int32(K_TILES - 1), (K_TILES - 1) % 2, False)
            else:
                # single A slot, no cross-tile prefetch: lowest VGPR / LDS footprint
                @flyc.jit
                def _k_loop(k_first):
                    for kt in range(K_TILES):
                        base_k = (k_first + fx.Int32(kt)) * fx.Int32(TILE_K)
                        load_a_global(base_k)
                        load_w_global(base_k)
                        store_a_lds(0)
                        rocdl.s_waitcnt(lgkmcnt=0)
                        gpu.barrier()  # A(kt) visible in LDS
                        compute_tile(read_w(), 0)
                        gpu.barrier()  # all reads done before the slot is overwritten

                _k_loop(fx.Int32(0))

            # ---- epilogue: dequant -> SiLU(gate)*up -> bf16 by sorted position ------
            f_gate = [_channel_factor(x) for x in exp_gate]
            f_up = [_channel_factor(x) for x in exp_up]
            if const_expr(epilog != "bf16"):
                _q8_epilogue(rows, M_REPEAT, acc_gate, acc_up, f_gate, f_up)
                return
            for mi in range_constexpr(M_REPEAT):
                for ii in range_constexpr(4):
                    row_in_tile = (
                        fx.Int32(mi * 16) + lane_div_16 * fx.Int32(4) + fx.Int32(ii)
                    )
                    sorted_row = bx_m + row_in_tile
                    fused = fx.Int32(_global_i32_at(arg_mind, sorted_row))
                    token = fused & fx.Int32(0x00FFFFFF)
                    valid = token < i32_ntok  # padding rows carry a sentinel token id
                    rs = _buffer_i32_scalar_read(xs_tiles, token, xs_atom).bitcast(
                        fx.Float32
                    )
                    for ni in range_constexpr(N_REPEAT):
                        g = fx.Float32(fx.Vector(fx.memref_load_vec(acc_gate[mi][ni]))[ii])
                        u = fx.Float32(fx.Vector(fx.memref_load_vec(acc_up[mi][ni]))[ii])
                        g = g * (rs * f_gate[ni])
                        u = u * (rs * f_up[ni])
                        y = gate_up_act(act, [g], [u], situ)[0]
                        out_idx = sorted_row * inter_i32 + out_cols[ni]
                        buffer_ops.buffer_store(
                            y.to(fx.BFloat16), _raw(out_rsrc), _raw(out_idx), mask=valid
                        )

        def _q8_epilogue(rows, M_REPEAT, acc_gate, acc_up, f_gate, f_up):
            """SiLU -> bf16 LDS tile -> per-(row, TILE_N) fp8 quant -> 16 B stores of
            unsorted rows to the local buffer (q8) or to every TP peer (q8push)."""
            lds_i32 = fx.Int32(fx.ptrtoint(lds_ptr))
            lds_bf16 = lds_typed_ptr(lds_i32, T.bf16)
            gpu.barrier()  # every wave finished reading the A tile from LDS
            for mi in range_constexpr(M_REPEAT):
                for ii in range_constexpr(4):
                    row_in_tile = (
                        fx.Int32(mi * 16) + lane_div_16 * fx.Int32(4) + fx.Int32(ii)
                    )
                    fused = fx.Int32(_global_i32_at(arg_mind, bx_m + row_in_tile))
                    token = fused & fx.Int32(0x00FFFFFF)
                    # padding rows: OOB scale read -> 0 -> y = 0 (masked at the store)
                    rs = _buffer_i32_scalar_read(xs_tiles, token, xs_atom).bitcast(
                        fx.Float32
                    )
                    for ni in range_constexpr(N_REPEAT):
                        g = fx.Float32(fx.Vector(fx.memref_load_vec(acc_gate[mi][ni]))[ii])
                        u = fx.Float32(fx.Vector(fx.memref_load_vec(acc_up[mi][ni]))[ii])
                        y = gate_up_act(
                            act, [g * (rs * f_gate[ni])], [u * (rs * f_up[ni])], situ
                        )[0]
                        col = n_wave_base + fx.Int32(ni * 16) + lane_mod_16
                        lds_bf16[row_in_tile * fx.Int32(TILE_N) + col] = y.to(fx.BFloat16)
            gpu.barrier()

            # q8 plane layout from the runtime row count (i32 dword indices as
            # before; byte extents in i64): data [NPES][rows][INTER] fp8, then the
            # f32 scales; a push slot is that rounded up to 256 B, two slots
            # (epoch parity), then the flags.
            rows_i64 = fx.Int64(i32_rows)
            data_b = rows_i64 * fx.Int64(NPES * INTER)
            slot_b = (
                data_b + rows_i64 * fx.Int64(NPES * NUM_N_BLOCKS * 4) + fx.Int64(255)
            ) & fx.Int64(-256)
            s_base = i32_rows * fx.Int32(NPES * INTER // 4)  # dword index of scales
            if const_expr(PUSH):
                tbl = global_typed_ptr(arg_tbl, T.i64, align=8)
                parity = fx.Int64(i32_epoch & fx.Int32(1)) * slot_b
                # store i of this WG goes to rank (rank + tile + i): spread over links
                dsts = [
                    buffer_ops.create_buffer_resource_from_addr(
                        _raw(
                            fx.Int64(
                                tbl[
                                    i32_rank
                                    if const_expr("selfonly" in PUSH_VAR)
                                    else (i32_rank + tile + fx.Int32(i)) % fx.Int32(NPES)
                                ]
                            )
                            + parity
                        ),
                        # bounded: masked lanes use offset 0x7FFFFFFF and must be
                        # dropped by the range check, not written at base + 2 GB
                        num_records_bytes=_raw(slot_b),
                    )
                    for i in range(1 if "one" in PUSH_VAR else NPES)
                ]
                plane = i32_rank * i32_rows
                cm = 17  # sc0|sc1 write-through (system coherent), as two-shot light
                if const_expr("cm0" in PUSH_VAR):
                    cm = 0
                if const_expr("cm16" in PUSH_VAR):
                    cm = 16
            else:
                dsts = [
                    buffer_ops.create_buffer_resource_from_addr(
                        _raw(fx.Int64(arg_out)),
                        num_records_bytes=_raw(
                            rows_i64 * fx.Int64(INTER + NUM_N_BLOCKS * 4)
                        ),
                    )
                ]
                plane = fx.Int32(0)
                cm = 0

            r_in = tx // fx.Int32(Q_TPR)
            sub = tx % fx.Int32(Q_TPR)
            for p in range_constexpr(rows // Q_RPP):
                row_in_tile = fx.Int32(p * Q_RPP) + r_in
                fused = fx.Int32(_global_i32_at(arg_mind, bx_m + row_in_tile))
                token = fused & fx.Int32(0x00FFFFFF)
                valid = token < i32_ntok
                if const_expr(SORTED):
                    dest_row = plane + bx_m + row_in_tile
                else:
                    dest_row = plane + token * fx.Int32(TOPK) + fused.shrui(fx.Int32(24))
                byte0 = (row_in_tile * fx.Int32(TILE_N) + sub * fx.Int32(16)) * fx.Int32(2)
                vals = []
                for h in range_constexpr(2):
                    v = Vec(
                        lds_vec_load(
                            lds_i32,
                            byte0 + fx.Int32(h * 16),
                            Vec.make_type(8, fx.BFloat16),
                            fx.BFloat16,
                            align=16,
                        )
                    ).to(fx.Float32)
                    vals.extend(v[k] for k in range(8))
                amax = fx.Float32(0.0)
                for k in range_constexpr(16):
                    amax = fx.max(amax, fx.max(vals[k], -vals[k]))
                for bit in range_constexpr(Q_TPR.bit_length() - 1):
                    amax = fx.max(amax, amax.shuffle_xor(1 << bit, 64))
                s = (amax == 0.0).select(fx.Float32(1.0), amax / 240.0)
                inv = 1.0 / s
                packed = Vec.from_elements(
                    [
                        _pack_activation(*[vals[4 * q + k] * inv for k in range(4)])
                        for q in range(4)
                    ],
                    fx.Int32,
                )
                d_off = (dest_row * fx.Int32(INTER) + by_n + sub * fx.Int32(16)) // fx.Int32(4)
                s_off = s_base + (dest_row * fx.Int32(NUM_N_BLOCKS) + n_block)
                s_valid = (sub == fx.Int32(0)).select(token, i32_ntok) < i32_ntok
                if const_expr(SCALE_RUN):
                    # all TPR lanes of the row hold the same s: identical LDS writes
                    lds_typed_ptr(lds_i32, T.f32)[
                        fx.Int32(BM * TILE_N * 2 // 4) + row_in_tile
                    ] = s
                for d in dsts:
                    buffer_ops.buffer_store(packed, d, d_off, mask=valid, cache_modifier=cm)
                    if const_expr("noscale" not in PUSH_VAR and not SCALE_RUN):
                        buffer_ops.buffer_store(s, d, s_off, mask=s_valid, cache_modifier=cm)

            if const_expr(SCALE_RUN):
                # scales of this tile's rows: [rank][n_block][bx_m : bx_m+rows] f32,
                # one contiguous rows*4 B run per peer (padding rows: s = 1, unused)
                gpu.barrier()
                sv = lds_vec_load(
                    lds_i32,
                    fx.Int32(BM * TILE_N * 2) + (tx % fx.Int32(BM // 4)) * fx.Int32(16),
                    Vec.make_type(4, fx.Float32),
                    fx.Float32,
                    align=16,
                )
                s_run = s_base + (
                    (i32_rank * fx.Int32(NUM_N_BLOCKS) + n_block) * i32_rows
                    + bx_m
                    + tx * fx.Int32(4)
                )
                for d in dsts:
                    buffer_ops.buffer_store(
                        sv, d, s_run, mask=tx < fx.Int32(rows // 4), cache_modifier=cm
                    )

            if const_expr(PUSH and "nopub" not in PUSH_VAR):
                rocdl.s_waitcnt(vmcnt=0)
                gpu.barrier()
                _publish(
                    arg_tbl,
                    i32_rank,
                    i32_epoch,
                    tx,
                    m_block * fx.Int32(NUM_N_BLOCKS) + n_block,
                    slot_b * fx.Int64(2),
                    i32_tiles,
                )

        # ---- row-variant dispatch ------------------------------------------------
        if const_expr(dyn_m):
            # Valid rows of a sorted block are contiguous from its start, so the
            # block fits BM/2 rows iff sorted row BM/2 already holds padding.
            probe = rocdl.readfirstlane(
                T.i32, _raw(_global_i32_at(arg_mind, bx_m + fx.Int32(BM // 2)))
            )
            half = (fx.Int32(probe) & fx.Int32(0x00FFFFFF)) >= i32_ntok

            @flyc.jit
            def _dispatch():
                if half:
                    run_rows(BM // 2)
                else:
                    run_rows(BM)

            _dispatch()
        else:
            run_rows(BM)

    @flyc.jit
    def _publish(arg_tbl, i32_rank, i32_epoch, tx, flag_idx, flag_off, i32_tiles):
        """Lane t < NPES sets flag[rank][tile] = epoch on rank (rank + t)."""
        if tx < fx.Int32(NPES):
            tbl = global_typed_ptr(arg_tbl, T.i64, align=8)
            peer = (i32_rank + tx) % fx.Int32(NPES)
            llvm.AtomicRMWOp(
                llvm.AtomicBinOp.xchg,
                _gep1(
                    _global_base_ptr1(fx.Int64(tbl[peer]) + fx.Int64(flag_off)),
                    (i32_rank * i32_tiles + flag_idx) * fx.Int32(4),
                ),
                _raw(i32_epoch),
                llvm.AtomicOrdering.monotonic,
                alignment=4,
            )

    @fx.struct
    class SharedStorage:
        raw: fx.Array[fx.Uint8, LDS_BYTES, 16]

    @flyc.jit
    def _entry(
        arg_x,
        arg_x_scale,
        arg_w,
        arg_w_scale,
        arg_w_exp,
        arg_eids,
        arg_cumsum,
        arg_mind,
        arg_out,
        i32_ntok,
        arg_tbl,
        i32_rank,
        i32_epoch,
        i32_rows,
        i32_tiles,
        f32_situ_beta,
        f32_situ_beta_rcp,
        f32_situ_linbeta,
        f32_situ_linbeta_rcp,
        f32_swiglu_limit,
    ):
        if const_expr(act == "situv2"):
            situ = situ_params(
                fx.Float32(f32_situ_beta),
                fx.Float32(f32_situ_beta_rcp),
                fx.Float32(f32_situ_linbeta),
                fx.Float32(f32_situ_linbeta_rcp),
                fx.Float32(f32_swiglu_limit),
            )
        else:
            situ = None
        lds_ptr = fx.SharedAllocator().allocate(SharedStorage).peek().raw.ptr
        tx = fx.Int32(gpu.thread_id("x"))
        bx = fx.Int32(gpu.block_id("x"))
        lane = tx % fx.Int32(64)
        wave = rocdl.readfirstlane(T.i32, tx // fx.Int32(64))
        cumsum0 = _global_i32_at(arg_cumsum, fx.Int32(0))
        total_m_blocks = cumsum0 // fx.Int32(BM)
        bound = total_m_blocks * fx.Int32(NUM_N_BLOCKS)

        # Bijective XCD round-robin over valid tiles; xcd_swizzle>0 also groups
        # xcd_swizzle m-blocks per XCD for L2 locality.
        NXCD = 8
        xq = _udiv(bound, NXCD)
        xr = _umod(bound, NXCD)

        def _xcd(pid):
            xc = _umod(pid, NXCD)
            wgid = xc * xq + fx.min(xc, xr) + _udiv(pid, NXCD)
            ng = fx.Int32(xcd_swizzle * NUM_N_BLOCKS)
            first_m = (wgid // ng) * fx.Int32(xcd_swizzle)
            group_m = fx.min(total_m_blocks - first_m, fx.Int32(xcd_swizzle))
            wig = wgid % ng
            return (first_m + wig % group_m) * fx.Int32(NUM_N_BLOCKS) + wig // group_m

        if bx < bound:
            if const_expr(xcd_swizzle > 0):
                tile = _xcd(bx)
            else:
                tile = bx
            _body(
                lds_ptr,
                arg_x,
                arg_x_scale,
                arg_w,
                arg_w_scale,
                arg_w_exp,
                arg_eids,
                arg_cumsum,
                arg_mind,
                arg_out,
                i32_ntok,
                arg_tbl,
                i32_rank,
                i32_epoch,
                i32_rows,
                i32_tiles,
                situ,
                tile,
                lane,
                wave,
            )

    @flyc.kernel(name=kname, known_block_size=[256, 1, 1])
    def a8w4_gemm1_kernel(
        arg_x: fx.Int64,
        arg_x_scale: fx.Int64,
        arg_w: fx.Int64,
        arg_w_scale: fx.Int64,
        arg_w_exp: fx.Int64,
        arg_eids: fx.Int64,
        arg_cumsum: fx.Int64,
        arg_mind: fx.Int64,
        arg_out: fx.Int64,
        i32_ntok: fx.Int32,
        arg_tbl: fx.Int64,
        i32_rank: fx.Int32,
        i32_epoch: fx.Int32,
        i32_rows: fx.Int32,
        i32_tiles: fx.Int32,
        f32_situ_beta: fx.Float32,
        f32_situ_beta_rcp: fx.Float32,
        f32_situ_linbeta: fx.Float32,
        f32_situ_linbeta_rcp: fx.Float32,
        f32_swiglu_limit: fx.Float32,
    ):
        _entry(
            arg_x,
            arg_x_scale,
            arg_w,
            arg_w_scale,
            arg_w_exp,
            arg_eids,
            arg_cumsum,
            arg_mind,
            arg_out,
            i32_ntok,
            arg_tbl,
            i32_rank,
            i32_epoch,
            i32_rows,
            i32_tiles,
            f32_situ_beta,
            f32_situ_beta_rcp,
            f32_situ_linbeta,
            f32_situ_linbeta_rcp,
            f32_swiglu_limit,
        )

    @flyc.jit
    def launch(
        arg_x: fx.Int64,
        arg_x_scale: fx.Int64,
        arg_w: fx.Int64,
        arg_w_scale: fx.Int64,
        arg_w_exp: fx.Int64,
        arg_eids: fx.Int64,
        arg_cumsum: fx.Int64,
        arg_mind: fx.Int64,
        arg_out: fx.Int64,
        i32_ntok: fx.Int32,
        arg_tbl: fx.Int64,
        i32_rank: fx.Int32,
        i32_epoch: fx.Int32,
        i32_rows: fx.Int32,
        f32_situ_beta: fx.Float32,
        f32_situ_beta_rcp: fx.Float32,
        f32_situ_linbeta: fx.Float32,
        f32_situ_linbeta_rcp: fx.Float32,
        f32_swiglu_limit: fx.Float32,
        i32_grid: fx.Int32,
        stream: fx.Stream,
    ):
        a8w4_gemm1_kernel(
            arg_x,
            arg_x_scale,
            arg_w,
            arg_w_scale,
            arg_w_exp,
            arg_eids,
            arg_cumsum,
            arg_mind,
            arg_out,
            i32_ntok,
            arg_tbl,
            i32_rank,
            i32_epoch,
            i32_rows,
            i32_grid,  # flag-table stride: one flag per launched tile
            f32_situ_beta,
            f32_situ_beta_rcp,
            f32_situ_linbeta,
            f32_situ_linbeta_rcp,
            f32_swiglu_limit,
            value_attrs=(
                {"rocdl.waves_per_eu": waves_per_eu} if waves_per_eu else None
            ),
        ).launch(grid=(fx.Int64(i32_grid), 1, 1), block=(256, 1, 1), stream=stream)

    return launch


def a8w4_gemm1(
    x_fp8,
    x_scale,
    w1,
    w1_scale,
    w1_exp,
    sorted_ids,
    sorted_expert_ids,
    num_valid_ids,
    out,
    *,
    num_experts,
    tile_m=64,
    tile_n=128,
    tile_k=256,
    b_cache_mod=1,
    xcd_swizzle=0,
    flush_tiny=True,
    pipeline=None,
    waves_per_eu=None,
    dyn_m=False,
    epilog="bf16",
    inter_dim=None,
    topk=None,
    npes=1,
    tbl=None,
    rank=0,
    epoch=0,
    w_layout="std",
    fast_dec=False,
    act="silu",
    situ_beta=1.0,
    situ_linear_beta=1.0,
    swiglu_limit=float("inf"),
):
    """Host launcher. ``out`` is bf16 ``[sorted_expert_ids.numel()*tile_m, inter]``
    (``epilog="bf16"``), the local q8 buffer (``"q8"``), or ignored (``"q8push"``:
    ``tbl`` = int64 peer-base table of the symmetric buffer)."""
    tokens, model_dim = x_fp8.shape
    if epilog == "bf16":
        inter_dim = out.shape[1]
    q8 = epilog != "bf16"
    launch = compile_a8w4_gemm1(
        D_HIDDEN=model_dim,
        D_INTER=inter_dim,
        NE=num_experts,
        BM=tile_m,
        TILE_N=tile_n,
        TILE_K=tile_k,
        b_cache_mod=b_cache_mod,
        xcd_swizzle=xcd_swizzle,
        flush_tiny=flush_tiny,
        pipeline=pipeline,
        waves_per_eu=waves_per_eu,
        dyn_m=dyn_m,
        epilog=epilog,
        TOPK=topk if q8 else None,
        NPES=npes,
        w_layout=w_layout,
        fast_dec=fast_dec,
        act=act,
    )
    grid = sorted_expert_ids.numel() * (inter_dim // tile_n)
    # q8 epilogues: rows of the [rows, inter] fp8 plane (its scales follow it). A
    # launch arg, not a compile constant: it changes with every batch size.
    rows = 0
    if q8:
        rows = (
            sorted_expert_ids.numel() * tile_m
            if epilog in ("q8s", "q8pushs")
            else tokens * topk
        )
    _run_compiled(
        launch,
        x_fp8.data_ptr(),
        x_scale.data_ptr(),
        w1.data_ptr(),
        w1_scale.data_ptr(),
        w1_exp.data_ptr(),
        sorted_expert_ids.data_ptr(),
        num_valid_ids.data_ptr(),
        sorted_ids.data_ptr(),
        out.data_ptr() if out is not None else 0,
        int(tokens),
        tbl.data_ptr() if tbl is not None else 0,
        int(rank),
        int(epoch),
        int(rows),
        *situ_args(situ_beta, situ_linear_beta, swiglu_limit),
        int(grid),
        torch.cuda.current_stream(),
    )
    return out


def situ_args(situ_beta=1.0, situ_linear_beta=1.0, swiglu_limit=float("inf")):
    """Runtime SiTUv2 launch args (beta, 1/beta, linear_beta, 1/linear_beta,
    clamp limit; +inf = no clamp), as flydsl_a16w4_gemm1. Ignored by silu."""
    beta, lbeta = float(situ_beta), float(situ_linear_beta)
    if beta <= 0.0 or lbeta <= 0.0:
        raise ValueError(f"situ_beta/situ_linear_beta must be > 0, got {beta}/{lbeta}")
    return beta, 1.0 / beta, lbeta, 1.0 / lbeta, float(swiglu_limit)


def prep_w1_fast_decode(w_u8):
    """W bytes (any of the layouts above) -> the ``fast_dec`` form, per dword:
    nibble order k0..k7 -> byte j = (k_j | k_{j+4} << 4), and -0 (0x8) -> +0."""
    b = w_u8.reshape(-1, 4)
    nib = torch.stack([b & 0xF, b >> 4], dim=-1).reshape(-1, 8)  # k0..k7
    nib = torch.where(nib == 8, torch.zeros_like(nib), nib)
    return (nib[:, :4] | (nib[:, 4:] << 4)).contiguous().view(-1)


def prep_w1_predelta(w1_u8, w1_scale_u8, w1_exp):
    """``fast_dec=2`` prep on the STANDARD-preshuffled W1 bytes / e8m0 scale bytes
    (before ``retile_w1`` / ``prep_w1_fast_decode``): scale byte ->
    min(max_code - code, 11) * 8, and W of flushed groups (max_code - code > 11,
    which the kernel decodes to 0) set to 0. ``w1_exp``: ``channel_exponents``.

    W bytes are (N/16, K/128, klane 4, nlane 16, 16 B): 16 B = the 32-K group
    kg = k128*4 + klane of column n = n16*16 + nlane. Scale bytes are (N/32, K/256,
    klane 4, nlane 16, kpack 2, npack 2): column n32*32 + npack*16 + nlane, group
    k256*8 + kpack*4 + klane."""
    NT = w1_exp.numel()  # all experts' output channels
    K = w1_u8.numel() * 2 // NT
    G = K // 32
    assert K % 256 == 0 and NT % 32 == 0 and w1_scale_u8.numel() == NT * G
    s6 = w1_scale_u8.reshape(NT // 32, K // 256, 4, 16, 2, 2)
    code = s6.permute(0, 5, 3, 1, 4, 2).reshape(NT, G).to(torch.int32)
    max_code = w1_exp.reshape(NT, 1).to(torch.int32)
    assert torch.equal(code.amax(-1, keepdim=True), max_code), "channel max mismatch"
    diff = max_code - code
    d8 = (diff.clamp(max=11) * 8).to(torch.uint8)
    s = d8.view(NT // 32, 2, 16, K // 256, 2, 4).permute(0, 3, 5, 2, 4, 1)
    flushed = (diff > 11).view(NT // 16, 16, K // 128, 4).permute(0, 2, 3, 1)
    w = w1_u8.reshape(NT // 16, K // 128, 4, 16, 16).clone()
    w[flushed] = 0
    return w.view(w1_u8.shape), s.contiguous().view(-1)


def prep_w1_fast(w1_u8, w1_scale_u8, w1_exp, fast_dec):
    """Standard-preshuffled W1 / scale bytes -> the ``fast_dec`` (1 or 2) form for
    the std ``w_layout``. ``w1_exp`` must come from the ORIGINAL scales."""
    assert fast_dec in (1, 2), fast_dec
    if fast_dec == 2:
        w1_u8, w1_scale_u8 = prep_w1_predelta(w1_u8, w1_scale_u8, w1_exp)
    return prep_w1_fast_decode(w1_u8).view(w1_u8.shape), w1_scale_u8


# =============================================================================
# Reference + test
# =============================================================================


