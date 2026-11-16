# Package copy of doc-mimo/kernels/test_a8w4_gemm1.py (test harness and experiment knobs removed).
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Standalone A8W4 MoE stage1 (gate+up GEMM + SiLU) FlyDSL kernel for gfx942.

Extracted from the a16w4 port (``moe_2stage_a16wmix/gemm1.py``, ``a_dtype="fp8r"``)
with only the configuration the tuned A8W4 build actually runs:

  * A  : FP8-FNUZ row-quantized activation ``[tokens, K]`` + f32 per-token scale
  * W1 : mxfp4 (E2M1 + per-1x32 E8M0), standard GGUU N-major preshuffle
  * MFMA(16,16,32,fp8) with fp4 -> fp8 decode of the fd2 weights (``prep_w1_fast``)
    through a per-32K-group byte LUT, one v_perm per 4 nibbles
  * runtime K loop, XOR-swizzled A-LDS, 4 waves split TILE_N; tiles of <= 32 rows
    prefetch tile kt+1 (A + W) during kt with double-buffered A-LDS
  * q8s epilogue: acc * a_scale[token] * 2**(exp-132) -> act(gate)*up -> bf16 in
    LDS -> fp8 + one f32 scale per (row, TILE_N columns), written by SORTED
    position: gemm2's A operand
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from ._act import gate_up_act, situ_params
from ._cvt import _pack_activation
from ._launch import _run_compiled
from ._mem import (
    _buffer_i32_scalar_read,
    _global_i32_at,
    _global_i32_buffer_tiles,
    _global_i32_buffer_view,
    _raw,
    _udiv,
    _umod,
    buffer_store,
    create_buffer_resource_from_addr,
    lds_typed_ptr,
    lds_vec_load,
)


# =============================================================================
# Device helpers: fp4 -> fp8 decode, channel factor, MFMA
# =============================================================================
#
# W stays E2M1/E8M0. Output channel n is normalized by 2**(max_exp[n]-127-5)
# (``_channel_factor``, applied in the epilogue); per 32-K group the remaining
# 2**(5-delta), delta = max_exp - group_exp, is folded into an 8-entry
# E2M1-magnitude -> FP8-FNUZ byte table (``_fp8_lut_predelta``) so each nibble
# decodes with one v_perm.


def _perm_u32(hi, lo, sel):
    return fx.Uint32(
        rocdl.perm_b32(
            fx.Uint32(hi).ir_value(), fx.Uint32(lo).ir_value(), fx.Uint32(sel).ir_value()
        )
    )


# ---- fd2 decode: W and scales pre-transformed by ``prep_w1_fast`` ----------------
# Each dword holds (k_j, k_{j+4}) in byte j (low, high nibble) and no -0 nibble, so
# even nibbles decode straight to K 0..3 and odd nibbles to K 4..7 (no interleave
# v_perm), and the sign is bit 3 of the nibble moved to bit 7 (no sign-table
# v_perm). The scale byte already holds min(delta, 11) * 8; groups below 2**-11 of
# the channel max (delta > 11) are zeroed in W, so they decode to 0.


def _fp4x8_to_fp8x8_fast(raw, lut):
    lo, hi, smask = lut
    raw = fx.Uint32(raw)
    even = ((raw << 4) & smask) | _perm_u32(hi, lo, raw & 0x07070707)
    odd = (raw & smask) | _perm_u32(hi, lo, (raw >> 4) & 0x07070707)
    return fx.Vector.from_elements(
        [even.bitcast(fx.Int32), odd.bitcast(fx.Int32)], fx.Int32
    ).bitcast(fx.Float8E4M3FNUZ)


def _fp8_lut_predelta(delta):
    """LUT of a 32-K group from its fd2 scale byte (min(max-code, 11)*8, broadcast
    to all 4 bytes)."""
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
    dyn_m=False,
    act="silu",
):
    """Build the A8W4 stage1 launcher (fd2 weights, q8s epilogue).

    ``act``: ``"silu"`` or ``"situv2"`` (as gemm1.py; SiTUv2 beta / linear_beta /
    clamp limit are runtime f32 launch args, nothing baked).

    Pipelining (row tiles of <= 32 rows): prefetch tile kt+1 (A + W into VGPRs)
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
    ``sorted_expert_ids``, ``num_valid_ids``, ``sorted_ids``, ``out`` (the q8
    plane, below), then ``tokens``, three unused scalars (``tbl``, ``rank``,
    ``epoch``), ``rows``, the five SiTUv2 scalars (``situ_args``), ``grid`` and the
    stream. The unused scalars (and the kernel's ``i32_tiles``) only keep the
    kernarg layout, and so the ISA, of the tuned build; they go with the host-launch
    rewrite.

    Nothing batch-size dependent is baked in: ``rows`` (rows of the q8 output
    plane) is a launch arg, so one binary serves every token count.

    q8s epilogue: the activation output is staged as bf16 in LDS, quantized to fp8
    FNUZ with one f32 scale per (row, TILE_N columns), rows by SORTED position
    (``rows`` = sorted_size): ``out`` = ``[sorted_size, inter]`` fp8 then
    ``[sorted_size, inter/TILE_N]`` f32, i.e. gemm2's A + per-128-K-group scale
    (``a8w4_gemm2``'s A layout); replaces a separate row-quant kernel.
    """
    assert act in ("silu", "situv2"), act
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
        return rows <= 32 and K_TILES > 1

    LDS_BYTES = max(r * TILE_K * (2 if _pipe(r) else 1) for r in ROW_VARIANTS)
    # bf16 epilogue tile; the BM*4 B tail is unused but keeps the tuned LDS size
    LDS_BYTES = max(LDS_BYTES, BM * TILE_N * 2 + BM * 4)
    Q_TPR = TILE_N // 16  # threads per row in the quant pass (16 cols each)
    Q_RPP = 256 // Q_TPR  # rows per pass
    assert all(r % Q_RPP == 0 for r in ROW_VARIANTS)

    # W (mxfp4) N-major preshuffle in bytes: (N/16, K/128, klane=4, nlane=16, kpack=16).
    W_K0 = K // 128
    W_BYTES = NE * N_OUT * K // 2
    # E8M0 scale preshuffle (dwords): (N/32, Kpad/256, klane=4, nlane=16); each dword
    # packs [kpack=2] x [npack=2] bytes.
    SCALE_K_PAD = ((K + 255) // 256) * 256
    SC_STRIDE_N0 = (SCALE_K_PAD // 256) * 64
    WS_BYTES = NE * N_OUT * (SCALE_K_PAD // 32)

    wpe_tag = f"_w{waves_per_eu}" if waves_per_eu else ""
    pipe_tag = "".join("p" if _pipe(r) else "s" for r in ROW_VARIANTS)
    dyn_tag = "_dynm" if dyn_m else ""
    # _ftz / _fd2 / _q8s: tags of options that are now always on (names unchanged)
    kname = (
        f"a8w4_gemm1_h{K}_i{INTER}_ne{NE}_bm{BM}_tn{TILE_N}_tk{TILE_K}"
        f"_bcm{b_cache_mod}_xcd{xcd_swizzle}{wpe_tag}_ftz{dyn_tag}_{pipe_tag}_fd2"
        + ("" if act == "silu" else f"_{act}")
        + "_q8s"
    )

    def _body(
        lds_ptr,
        arg_x,
        arg_x_scale,
        arg_w,
        arg_w_scale,
        arg_w_exp,
        arg_eids,
        arg_mind,
        arg_out,
        i32_ntok,
        i32_rows,
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
        for ni in range_constexpr(N_REPEAT):
            n_local = by_n + n_wave_base + fx.Int32(ni * 16)
            out_cols.append(n_local + lane_mod_16)
            for gu in range_constexpr(2):
                n0 = expert_off + n_local + (inter_i32 if gu else fx.Int32(0))
                w_chunk = (n0 // fx.Int32(16)) * fx.Int32(W_K0 * 64)
                w_base[gu, ni] = w_chunk + lane_div_16 * fx.Int32(16) + lane_mod_16
                key = (gu, ni // 2) if share_scale else (gu, ni)
                sc_key[gu, ni] = key
                sc_npack[gu, ni] = (n0 // fx.Int32(16)) % fx.Int32(2)
                if key not in sc_base:
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
            s_off = (base_k // fx.Int32(256)) * fx.Int32(64)
            for (key, g), r in s_regs.items():
                idx = sc_base[key] + s_off + fx.Int32((g // 2) * 64)
                fx.copy(ws_atom, fx.slice(ws_tiles, (None, idx)), r)
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
                # byte (g%2)*2 + npack of the dword, broadcast by one v_perm
                p = packed[sc_key[gu, ni], g]
                sel = (fx.Int32((g % 2) * 2) + sc_npack[gu, ni]) * fx.Int32(
                    0x01010101
                )
                code[gu, ni, g] = _perm_u32(p, p, sel)
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
                            for g in range_constexpr(K_GROUPS)
                        ]
                        for gu in range_constexpr(2)
                    ]
                    for ku in range_constexpr(K_STEPS):
                        g, j = ku // 4, ku % 4
                        bs = [
                            _fp4x8_to_fp8x8_fast(raw[gu, ni, g][j], luts[gu][g])
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

            # ---- epilogue -------------------------------------------------------
            f_gate = [_channel_factor(x) for x in exp_gate]
            f_up = [_channel_factor(x) for x in exp_up]
            _q8_epilogue(rows, M_REPEAT, acc_gate, acc_up, f_gate, f_up)

        def _q8_epilogue(rows, M_REPEAT, acc_gate, acc_up, f_gate, f_up):
            """SiLU -> bf16 LDS tile -> per-(row, TILE_N) fp8 quant -> 16 B stores of
            the rows by sorted position (+ one f32 scale per row and n-block)."""
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

            # q8 plane layout from the runtime row count: data [rows][INTER] fp8,
            # then [rows][NUM_N_BLOCKS] f32 scales (dword index ``s_base``).
            rows_i64 = fx.Int64(i32_rows)
            s_base = i32_rows * fx.Int32(INTER // 4)
            dst = create_buffer_resource_from_addr(
                _raw(fx.Int64(arg_out)),
                num_records_bytes=_raw(rows_i64 * fx.Int64(INTER + NUM_N_BLOCKS * 4)),
            )

            r_in = tx // fx.Int32(Q_TPR)
            sub = tx % fx.Int32(Q_TPR)
            for p in range_constexpr(rows // Q_RPP):
                row_in_tile = fx.Int32(p * Q_RPP) + r_in
                fused = fx.Int32(_global_i32_at(arg_mind, bx_m + row_in_tile))
                token = fused & fx.Int32(0x00FFFFFF)
                valid = token < i32_ntok
                dest_row = bx_m + row_in_tile
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
                buffer_store(packed, dst, d_off, mask=valid)
                buffer_store(s, dst, s_off, mask=s_valid)

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
                arg_mind,
                arg_out,
                i32_ntok,
                i32_rows,
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
        arg_tbl: fx.Int64,  # unused (kernarg layout of the tuned build)
        i32_rank: fx.Int32,  # unused
        i32_epoch: fx.Int32,  # unused
        i32_rows: fx.Int32,
        i32_tiles: fx.Int32,  # unused
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
            i32_grid,  # i32_tiles (unused)
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
    inter_dim,
    tile_m=64,
    tile_n=128,
    tile_k=256,
    b_cache_mod=1,
    xcd_swizzle=0,
    waves_per_eu=None,
    dyn_m=False,
    act="silu",
    situ_beta=1.0,
    situ_linear_beta=1.0,
    swiglu_limit=float("inf"),
):
    """Host launcher. ``w1`` / ``w1_scale`` / ``w1_exp``: fd2 weights
    (``prepare_weights``). ``out`` is the q8 buffer: fp8 [sorted_size, inter_dim]
    then f32 [sorted_size, inter_dim / tile_n], sorted_size =
    ``sorted_expert_ids.numel() * tile_m``."""
    tokens, model_dim = x_fp8.shape
    launch = compile_a8w4_gemm1(
        D_HIDDEN=model_dim,
        D_INTER=inter_dim,
        NE=num_experts,
        BM=tile_m,
        TILE_N=tile_n,
        TILE_K=tile_k,
        b_cache_mod=b_cache_mod,
        xcd_swizzle=xcd_swizzle,
        waves_per_eu=waves_per_eu,
        dyn_m=dyn_m,
        act=act,
    )
    grid = sorted_expert_ids.numel() * (inter_dim // tile_n)
    # rows of the q8 plane (its scales follow it). A launch arg, not a compile
    # constant: it changes with every batch size.
    rows = sorted_expert_ids.numel() * tile_m
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
        out.data_ptr(),
        int(tokens),
        0,  # tbl, rank, epoch: unused kernel args
        0,
        0,
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
    """Preshuffled W bytes -> the fd2 nibble order, per dword:
    nibble order k0..k7 -> byte j = (k_j | k_{j+4} << 4), and -0 (0x8) -> +0."""
    b = w_u8.reshape(-1, 4)
    nib = torch.stack([b & 0xF, b >> 4], dim=-1).reshape(-1, 8)  # k0..k7
    nib = torch.where(nib == 8, torch.zeros_like(nib), nib)
    return (nib[:, :4] | (nib[:, 4:] << 4)).contiguous().view(-1)


def prep_w1_predelta(w1_u8, w1_scale_u8, w1_exp):
    """fd2 scale prep on the STANDARD-preshuffled W1 bytes / e8m0 scale bytes
    (before ``prep_w1_fast_decode``): scale byte ->
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


def prep_w1_fast(w1_u8, w1_scale_u8, w1_exp):
    """Standard-preshuffled W / scale bytes -> the fd2 form (any MoE weight, W1 or
    W2). ``w1_exp`` must come from the ORIGINAL scales."""
    w1_u8, w1_scale_u8 = prep_w1_predelta(w1_u8, w1_scale_u8, w1_exp)
    return prep_w1_fast_decode(w1_u8).view(w1_u8.shape), w1_scale_u8

