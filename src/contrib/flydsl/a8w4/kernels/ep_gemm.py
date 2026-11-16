# Package copy of doc-mimo/kernels/ep8_moe.py (test harness removed, env knobs frozen).
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Mimo v2.6 flash MoE, **EP8** per-rank kernels (gfx942, FlyDSL).

EP8 shape per rank: model_dim H 4096, full inter_dim I 2048, 32 local experts (of
256 global, topk 8). Weights are mxfp4 (E2M1 + per-1x32 E8M0), prepared once by
``prepare_weights`` into the fd2 form (nibble-reordered W, per-group scale deltas,
per-channel max exponent). A = fp8 FNUZ (x per token row, the activation output per
(row, 128) group), MFMA 16x16x32 fp8; W decodes fp4 -> fp8 through the fd2 byte LUT.

The gate/up activation (gemm1 epilogue) is SiLU or SiTUv2 (``act``; SiTUv2 betas
and clamp are runtime launch args).

The W decode carries the per-32-K-group scale relative to the output channel's max
group scale; the channel factor 2**(max-132) is applied in the epilogue.

Kernel structure (both GEMMs, ``_make_core``): one WG (NW waves) per (m-block,
n-block) tile; the waves split the N columns (each wave owns all rows of its
columns, so a decoded W fragment feeds R/16 MFMAs and no W element is decoded twice
in a WG); A K-tiles go through double-buffered XOR-swizzled LDS; W goes straight
from global into VGPRs in the MFMA operand layout ``PF`` K-tiles ahead (the runtime
K loop is unrolled 2*PF times so every prefetch buffer alternates between two
static register sets: no copies of in-flight registers at the back-edge); A
fragments are read one MFMA step ahead. A WG runs the smallest compiled row variant
(ROWV) that covers its block's valid rows.

Tile -> WG: the plain order gives XCD x (= bx % 8) a contiguous range of m-major
tiles; with the prep's schedule (``sched``) XCD x runs its cost-balanced range of
m-blocks, longest tiles first, in WG (= hardware dispatch) order. ``ep_moe``
picks the config (``CFGS``) by token count (``AUTO``).
"""

import functools
import os

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from ._act import gate_up_act, situ_args, situ_params
from ._cvt import (
    _channel_factor,
    _fp4x8_to_fp8x8_fast,
    _fp8_lut_predelta,
    _fp8x4_pack,
    _pack_activation,
    _perm_u32,
)
from ._launch import _run_compiled
from ._mem import (
    _buffer_i32_scalar_read,
    _global_base_ptr1,
    _global_i32_at,
    _global_i32_buffer_tiles,
    _global_i32_buffer_view,
    _raw,
    _udiv,
    _umod,
    buffer_store,
    create_buffer_resource_from_addr,
    get_element_ptr,
    global_typed_ptr,
    lds_typed_ptr,
    lds_vec_load,
)
from . import ep_prep as PR  # noqa: E402


HERE = os.path.dirname(os.path.abspath(__file__))


# a8w4 intermediate: one f32 scale per (row, 128 columns). An fp8 MFMA K32 step
# takes K offsets klane*32 + 0..7 of a whole 128-K chunk, so 128 is the finest
# group gemm2 can rescale between MFMA steps.
QG1 = 128


# =============================================================================
# weights
# =============================================================================


# =============================================================================
# device helpers
# =============================================================================


def _frag(v_i32, elem, n):
    """MFMA operand rmem fragment of ``n`` ``elem`` from a vector of i32."""
    t = fx.make_rmem_tensor(fx.make_layout(n, 1), elem)
    t.store(Vec(v_i32).bitcast(elem))
    return t


def _dispatch_rows(rowv, pads, run_rows):
    """Run the smallest row variant whose next probe row is padding (nested ifs:
    a compile-time if followed by a runtime elif is unreliable in FlyDSL)."""
    def level(i):
        if const_expr(i == len(rowv) - 1):
            run_rows(rowv[-1])
            return

        @flyc.jit
        def _d():
            if pads[i]:
                run_rows(rowv[i])
            else:
                level(i + 1)

        _d()

    level(0)


@flyc.jit
def _masked_lds_store(atom, reg, dst, ok):
    if ok:
        fx.copy(atom, reg, dst)


@flyc.jit
def _lds_f32_store_masked(lds_f32, idx, val, ok):
    if ok:
        lds_f32[idx] = val


@flyc.jit
def _atomic_bf16x4(arg_out, base, v4, ok):
    """out[base .. base+3] += 4 f32 values rounded to bf16 (2 bf16x2 agent atomics)."""
    if ok:
        for h in range_constexpr(2):
            pk = Vec.from_elements([v4[2 * h], v4[2 * h + 1]], fx.Float32).to(fx.BFloat16)
            llvm.AtomicRMWOp(
                llvm.AtomicBinOp.fadd,
                get_element_ptr(_global_base_ptr1(arg_out), (base + fx.Int32(2 * h)) * fx.Int32(2)),
                _raw(pk), llvm.AtomicOrdering.monotonic, syncscope="agent", alignment=4)


@flyc.jit
def _zero_slice(arg_zero, i32_n16, bx, grid, tx, NT):
    """WG bx of ``grid`` zeroes its share of ``i32_n16`` 16 B units at arg_zero."""
    if i32_n16 > fx.Int32(0):
        rs = create_buffer_resource_from_addr(
            _raw(fx.Int64(arg_zero)), num_records_bytes=_raw(fx.Int64(i32_n16) * fx.Int64(16)))
        per = (i32_n16 + grid - fx.Int32(1)) // grid
        z = Vec.filled(4, 0, fx.Int32)
        for i in range((per + fx.Int32(NT - 1)) // fx.Int32(NT)):
            u = fx.Int32(i) * fx.Int32(NT) + tx
            # out-of-range units (u >= per, or past the end) are dropped by the bound
            off = (u < per).select(bx * per + u, i32_n16)
            buffer_store(z, rs, off * fx.Int32(4))


def _zero_rc_pre(arg_rc, i32_n16, i32_m, ROW16, bx, grid):
    """route-count-aware zeroing, part 1 (kernel start): this WG's share of the
    ``i32_n16`` 16 B units of out [M, ROW16 * 16 B] spans <= 2 rows (host asserts
    share <= ROW16); their local route counts are loaded now into SGPRs (nothing
    per lane stays live across the tile) and consumed after the tile."""
    rc = global_typed_ptr(arg_rc, T.i32)
    per = (i32_n16 + grid - fx.Int32(1)) // grid
    r0 = fx.min(bx * per // fx.Int32(ROW16), i32_m - fx.Int32(1))
    r1 = fx.min(r0 + fx.Int32(1), i32_m - fx.Int32(1))
    return (per, r0, fx.Int32(rocdl.readfirstlane(T.i32, _raw(fx.Int32(rc[r0])))),
            fx.Int32(rocdl.readfirstlane(T.i32, _raw(fx.Int32(rc[r1])))))


def _zero_rc_post(arg_zero, i32_n16, ROW16, UPT, pre, bx, tx, NT):
    """part 2 (after the tile): zero the units whose row has != 1 local route (a
    single-route row is stored whole by gemm2)."""
    per, r0, rc0, rc1 = pre
    rs = create_buffer_resource_from_addr(
        _raw(fx.Int64(arg_zero)), num_records_bytes=_raw(fx.Int64(i32_n16) * fx.Int64(16)))
    z = Vec.filled(4, 0, fx.Int32)
    for k in range_constexpr(UPT):
        u = fx.Int32(k * NT) + tx
        off = (u < per).select(bx * per + u, i32_n16)  # past the end: dropped
        rcv = (off // fx.Int32(ROW16) == r0).select(rc0, rc1)
        buffer_store(z, rs, (rcv != fx.Int32(1)).select(off, i32_n16) * fx.Int32(4))


@flyc.jit
def _st_or_add_bf16x2(arg_out, idx, pk, single):
    """out[idx .. idx+1] (bf16): plain store for a single-route row, else atomic add."""
    ptr = get_element_ptr(_global_base_ptr1(arg_out), idx * fx.Int32(2))
    if single:
        llvm.StoreOp(_raw(pk), ptr, alignment=4)
    else:
        llvm.AtomicRMWOp(llvm.AtomicBinOp.fadd, ptr, _raw(pk), llvm.AtomicOrdering.monotonic,
                         syncscope="agent", alignment=4)


def _xcd_tile(bx, bound):
    """Bijective XCD round-robin: XCD x (= bx % 8) owns a contiguous range of tiles."""
    xq = _udiv(bound, 8)
    xr = _umod(bound, 8)
    xc = _umod(bx, 8)
    return xc * xq + fx.min(xc, xr) + _udiv(bx, 8)


def _opaque_tx(tx):
    """(tx, lane, wave) re-derived from an asm copy of tx right before a tile body:
    LLVM cannot hoist the body's per-lane address math (out of the tile loop / to
    the kernel start), whose long live ranges spilled up to ~80 VGPRs (bf16); with
    it no config spills and the fp8 kernels need ~15 VGPRs less."""
    t = fx.Int32(llvm.inline_asm(T.i32, [_raw(tx)], "v_mov_b32 $0, $1", "=v,v",
                                 has_side_effects=True))
    return t, t % fx.Int32(64), fx.Int32(rocdl.readfirstlane(T.i32, _raw(t // fx.Int32(64))))


def _tile_of(tile, NB):
    """static schedule: tile -> (m_block, n_block), m-major."""
    return tile // fx.Int32(NB), tile % fx.Int32(NB)


def _sched_i32(arg_sched, idx):
    return fx.Int32(rocdl.readfirstlane(T.i32, _raw(_global_i32_at(arg_sched, idx))))


def _bucket_bounds(arg_sched, bx, grid, NB):
    """prep's tile order: XCD x (= bx % 8; the hardware deals WGs round robin to
    the XCDs and, inside one, to the CUs in WG order) owns the tiles [bst[x] NB,
    bst[x+1] NB) (m-major) and runs them with its nwg WGs: WG i (= bx // 8) takes
    tiles i, i + nwg, ... (one, unless the bucket outgrows the grid)."""
    xcd = bx % fx.Int32(8)
    lo = _sched_i32(arg_sched, xcd) * fx.Int32(NB)
    hi = _sched_i32(arg_sched, xcd + fx.Int32(1)) * fx.Int32(NB)
    return lo, hi, (grid - xcd + fx.Int32(7)) // fx.Int32(8)


def _sched_tile(arg_sched, t, NB):
    return (_sched_i32(arg_sched, fx.Int32(PR.SCHED_BORDER) + t // fx.Int32(NB)),
            t % fx.Int32(NB))


def _grid(tiles, sched):
    """one WG per (upper bound) tile; with the prep's schedule a multiple of 8 (8
    XCD buckets, dead WGs exit at once)."""
    return int(-(-tiles // 8) * 8 if sched is not None else tiles)


def _row_pads(arg_sids, bx_m, rowv, i32_m):
    """per row variant (but the last): is sorted row ``bx_m + r`` already padding?"""
    return [
        (fx.Int32(rocdl.readfirstlane(T.i32, _raw(_global_i32_at(arg_sids, bx_m + fx.Int32(r)))))
         & fx.Int32(0x00FFFFFF)) >= i32_m
        for r in rowv[:-1]
    ]


# =============================================================================
# shared K loop
# =============================================================================


def _make_core(*, K, W_ROWS, BM, PF, bcm, cols, lds_ptr, LDS_BYTES, lane, tx, NT,
               arg_w, arg_ws, arg_a, a_num_bytes, ARB=256, acm=2, SCH=1, ST_STEP=None,
               PRIO=0, TR=False, AQ8=False):
    """Trace-time builder of the streamed-W K loop.

    ``cols``: list of (key, n0, skey): the 16-col W blocks this wave owns (``n0`` =
    first W row over all experts, ``skey`` groups the blocks sharing a scale dword).
    Returns ``run(R, row_base16, hook) -> acc[key][mi]`` (rmem f32x4, MFMA C layout:
    lane L holds rows (L//16)*4 + v, column L%16; ``TR``: the W fragment is the
    MFMA A operand, so acc is transposed: lane L holds row L%16 of the 16-row block
    and the 4 consecutive columns (L//16)*4 + v). ``row_base16(row)``: 16 B-unit
    index of A row ``row`` (< R) of the block (indices past ``a_num_bytes`` read 0).
    ``hook(acc, kt, s)``: called before MFMA step s of every tile (group rescale).
    ``ARB``: A bytes per row per K tile (128 / 256).
    ``AQ8``: A source is bf16 (``row_base16`` in bf16 16 B units), quantized while
    staged to fp8 with one scale per (row, 128-K tile); the per-row ratio
    s[kt-1]/s[kt] (1 at kt 0) goes to LDS f32 [2][BM] at ``2*BM*ARB`` (slot kt%2,
    visible from tile kt's barrier) and the last tile's scale to f32 [BM] after it.
    """
    assert not AQ8 or ARB == 128, "AQ8: one 128-K group per fp8 tile"
    TK = ARB  # K per tile (1 B per fp8 A element)
    KG = TK // 128
    assert KG >= 1, "a K tile holds whole 128-K W chunks"
    KT = K // TK
    STEPS = TK // 32  # MFMA K steps per tile
    assert PF >= 1 and KT % (2 * PF) == 0 and KT // (2 * PF) >= 2, (KT, PF)
    W_STEP = 0
    ST_STEP = STEPS // 2 if ST_STEP is None else ST_STEP
    assert 0 <= ST_STEP < STEPS
    SLOT = BM * ARB
    W_K0 = K // 128
    W_BYTES = W_ROWS * K // 2
    WS_BYTES = W_ROWS * K // 32
    SC_STRIDE_N0 = (K // 256) * 64  # scale dwords per 32-col block
    CH = ARB // 16  # 16 B A chunks per row per tile
    lane_div_16 = lane // fx.Int32(16)
    lane_mod_16 = lane % fx.Int32(16)

    w_tiles = _global_i32_buffer_tiles(arg_w, W_BYTES, 4)
    w_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(bcm), fx.Int32)
    ws_tiles = _global_i32_buffer_tiles(arg_ws, WS_BYTES, 1)
    ws_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(0), fx.Int32)
    keys = [k for k, _, _ in cols]
    w_base, npack, s_base, skey_of = {}, {}, {}, {}
    for key, n0, skey in cols:
        w_base[key] = (n0 // fx.Int32(16)) * fx.Int32(W_K0 * 64) + lane
        npack[key] = (n0 // fx.Int32(16)) % fx.Int32(2)
        skey_of[key] = skey
        if skey not in s_base:
            s_base[skey] = (n0 // fx.Int32(32)) * fx.Int32(SC_STRIDE_N0) + lane

    def new_wbuf():
        return dict(
            w={(k, g): fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
               for k in keys for g in range(KG)},
            s={sk: fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Int32) for sk in s_base},
        )

    def load_w(buf, kt):
        k128 = kt * fx.Int32(KG)
        for (k, g), r in buf["w"].items():
            fx.copy(w_atom, fx.slice(w_tiles, (None, w_base[k] + (k128 + fx.Int32(g))
                                               * fx.Int32(64))), r)
        s_off = (k128 // fx.Int32(2)) * fx.Int32(64)  # 256-K scale block
        for sk, r in buf["s"].items():
            fx.copy(ws_atom, fx.slice(ws_tiles, (None, s_base[sk] + s_off)), r)

    def read_w(buf, kt):
        """-> raw[key, g] = 4 x i32 (8 fp4 each), lut[key, g]."""
        raw, lut = {}, {}
        packed = {sk: fx.Int32(Vec(fx.memref_load_vec(r))[0]) for sk, r in buf["s"].items()}
        for (k, g), r in buf["w"].items():
            v4 = Vec(fx.memref_load_vec(r))
            raw[k, g] = [fx.Int32(v4[j]) for j in range(4)]
            p = packed[skey_of[k]]
            kpack = (kt * fx.Int32(KG) + fx.Int32(g)) % fx.Int32(2)  # 128-K chunk parity
            sel = (kpack * fx.Int32(2) + npack[k]) * fx.Int32(0x01010101)
            lut[k, g] = _fp8_lut_predelta(_perm_u32(p, p, sel))
        return raw, lut

    a_tiles = fx.logical_divide(_global_i32_buffer_view(arg_a, a_num_bytes),
                                fx.make_layout(4, 1))
    a_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(acm), fx.Int32)
    lds_flat = fx.make_view(fx.recast_iter(fx.Int32, lds_ptr), fx.make_layout(LDS_BYTES // 4, 1))
    lds16 = fx.logical_divide(lds_flat, fx.make_layout(4, 1))
    lds8 = fx.logical_divide(lds_flat, fx.make_layout(2, 1))
    lds_w_atom = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Int32)
    lds_r_atom = fx.make_copy_atom(fx.UniversalCopy64b(), fx.Int32)
    lds_r2_atom = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Int32)
    mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, fx.Float8E4M3FNUZ))

    def swz(row, col16):
        return col16 ^ (row & fx.Int32(CH - 1))

    def run(R, row_base16, hook=None, pre_a=None):
        MR = R // 16
        A_LOADS = -(-(R * CH) // NT)  # 16 B A chunks per thread per tile
        AU = 2 if AQ8 else 1  # 16 B source units per LDS chunk
        a_src, a_dst, a_ok, a_row, a_c16 = [], [], [], [], []
        for i in range_constexpr(A_LOADS):
            ch = tx + fx.Int32(i * NT)
            row = ch // fx.Int32(CH)
            c16 = ch % fx.Int32(CH)
            ok = row < fx.Int32(R)
            rowc = ok.select(row, fx.Int32(0))
            a_dst.append(rowc * fx.Int32(CH) + swz(rowc, c16))
            a_ok.append(ok)
            a_row.append(rowc)
            a_c16.append(c16)

        def a_sources():  # after pre_a: row_base16 may read rows written by a producer
            for i in range_constexpr(A_LOADS):
                a_src.append(row_base16(a_row[i]) + a_c16[i] * fx.Int32(AU))
        a_regs = [[fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32) for _ in range(AU)]
                  for _ in range(A_LOADS)]
        if const_expr(AQ8):
            assert CH == 8 and NT % 8 == 0
            RQ = 2 * SLOT // 4  # f32 index of the ratio slots, fin after them
            lds_f = lds_typed_ptr(fx.Int32(fx.ptrtoint(lds_ptr)), T.f32)
            s_prev = [fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Float32)
                      for _ in range(A_LOADS)]

        def load_a(kt):
            k16 = kt * fx.Int32(CH * AU)
            for i in range_constexpr(A_LOADS):
                for h in range_constexpr(AU):
                    fx.copy(a_atom, fx.slice(a_tiles, (None, a_src[i] + k16 + fx.Int32(h))),
                            a_regs[i][h])

        def quant_chunk(i, slot, first, last):
            """16 bf16 -> 16 fp8 (row scale over the row's 8 chunk lanes) + ratio/fin."""
            vals = []
            for h in range_constexpr(2):
                v8 = Vec(fx.memref_load_vec(a_regs[i][h])).bitcast(fx.BFloat16).to(fx.Float32)
                vals.extend(fx.Float32(v8[j]) for j in range(8))
            am = fx.Float32(0.0)
            for v in vals:
                am = fx.maxnumf(am, fx.absf(v))
            for bit in range_constexpr(3):
                am = fx.maxnumf(am, am.shuffle_xor(1 << bit, 64))
            s = (am == 0.0).select(fx.Float32(1.0), am / 240.0)
            inv = 1.0 / s
            q = Vec.from_elements([_fp8x4_pack(*[vals[4 * j + k] * inv for k in range(4)])
                                   for j in range(4)], fx.Int32)
            if const_expr(first):
                ratio = fx.Float32(1.0)
            else:
                ratio = fx.Float32(Vec(fx.memref_load_vec(s_prev[i]))[0]) / s
            s_prev[i].store(Vec.from_elements([s], fx.Float32))
            head = a_ok[i] & (a_c16[i] == fx.Int32(0))
            _lds_f32_store_masked(lds_f, fx.Int32(RQ) + fx.Int32(slot) * fx.Int32(BM) + a_row[i],
                                  ratio, head)
            if const_expr(last):
                _lds_f32_store_masked(lds_f, fx.Int32(RQ + 2 * BM) + a_row[i], s, head)
            return q

        def store_a(slot, first=False, last=False):
            s16 = fx.Int32(slot) * fx.Int32(SLOT // 16)
            for i in range_constexpr(A_LOADS):
                dst = fx.slice(lds16, (None, a_dst[i] + s16))
                src = a_regs[i][0]
                if const_expr(AQ8):
                    src = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                    src.store(quant_chunk(i, slot, first, last))
                if const_expr((i + 1) * NT <= R * CH):
                    fx.copy(lds_w_atom, src, dst)
                else:
                    _masked_lds_store(lds_w_atom, src, dst, a_ok[i])

        def read_a(mi, s, slot):
            row = lane_mod_16 + fx.Int32(mi * 16)
            # K32 step: bytes (s//4)*128 + klane*32 + (s%4)*8
            byte = fx.Int32((s // 4) * 128 + (s % 4) * 8) + lane_div_16 * fx.Int32(32)
            c16 = swz(row, byte // fx.Int32(16))
            off8 = (row * fx.Int32(ARB) + c16 * fx.Int32(16) + byte % fx.Int32(16)) \
                // fx.Int32(8) + fx.Int32(slot) * fx.Int32(SLOT // 8)
            r = fx.make_rmem_tensor(fx.make_layout(2, 1), fx.Int32)
            fx.copy_atom_call(lds_r_atom, fx.slice(lds8, (None, off8)), r)
            return Vec(fx.memref_load_vec(r))

        def read_a2(mi, s, slot):
            """16 B = the A fragments of MFMA steps s, s+1 (s even): one
            ds_read_b128 (conflict-free with the 16 B XOR swizzle; two b64 reads of
            the same bytes are 2-way bank conflicted)."""
            row = lane_mod_16 + fx.Int32(mi * 16)
            byte = fx.Int32((s // 4) * 128 + (s % 4) * 8) + lane_div_16 * fx.Int32(32)
            off16 = row * fx.Int32(CH) + swz(row, byte // fx.Int32(16)) \
                + fx.Int32(slot) * fx.Int32(SLOT // 16)
            r = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
            fx.copy_atom_call(lds_r2_atom, fx.slice(lds16, (None, off16)), r)
            return Vec(fx.memref_load_vec(r))

        acc = {k: [fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Float32) for _ in range(MR)]
               for k in keys}
        zero4 = Vec.filled(4, 0.0, fx.Float32)
        for k in keys:
            for mi in range_constexpr(MR):
                acc[k][mi].store(zero4)

        def decode_b(raw, lut, s):
            """B operand words (2 x i32) of MFMA step s, per key."""
            bws = {}
            for k in keys:  # step s = 128-K chunk s//4, dword s%4
                b8 = _fp4x8_to_fp8x8_fast(raw[k, s // 4][s % 4], lut[k, s // 4])
                bws[k] = Vec(b8.bitcast(fx.Int32))
            return bws

        def a_frags(s, slot):
            return [read_a(mi, s, slot) for mi in range_constexpr(MR)]

        ELEM = fx.Float8E4M3FNUZ
        NEL = 8

        def mfmas(a_cur, bws):
            for k in keys:
                bf = _frag(bws[k], ELEM, NEL)
                for mi in range_constexpr(MR):
                    af = _frag(a_cur[mi], ELEM, NEL)
                    if const_expr(TR):
                        fx.gemm(mma_atom, acc[k][mi], bf, af, acc[k][mi])
                    else:
                        fx.gemm(mma_atom, acc[k][mi], af, bf, acc[k][mi])

        def compute(raw, lut, slot, kt):
            a_cur = a_frags(0, slot)
            for s in range_constexpr(STEPS):
                a_nxt = a_frags(s + 1, slot) if s + 1 < STEPS else None
                if const_expr(hook is not None):
                    hook(acc, kt, s)
                mfmas(a_cur, decode_b(raw, lut, s))
                a_cur = a_nxt

        # W prefetch buffers: wbufs[u][p] holds the tiles kt = u (mod PF) of parity
        # p = (kt // PF) % 2; the loop body covers 2*PF tiles.
        wbufs = [[new_wbuf() for _ in range(2)] for _ in range(PF)]
        for u in range_constexpr(PF):
            load_w(wbufs[u][0], fx.Int32(u))
        if const_expr(pre_a is not None):
            pre_a()  # e.g. wait for A's producer; the W prefetch is already in flight
        a_sources()
        load_a(fx.Int32(0))
        store_a(0, first=True, last=KT == 1)
        load_a(fx.Int32(1))

        def tile_step_v1(kt, u, p, more_w, store_next, load_next2, last_a=False):
            """K tile kt: W in wbufs[u][p], A in LDS slot kt%2, A(kt+1) in regs."""
            slot = kt % fx.Int32(2)
            rocdl.s_waitcnt(lgkmcnt=0)
            gpu.barrier()  # A(kt) visible; slot (kt+1)%2 free (read at kt-1)
            if const_expr(store_next):
                store_a(fx.Int32(1) - slot, last=last_a)
            if const_expr(load_next2):
                load_a(kt + fx.Int32(2))
            raw, lut = read_w(wbufs[u][p], kt)
            if const_expr(more_w):
                load_w(wbufs[u][1 - p], kt + fx.Int32(PF))
            rocdl.sched_barrier(0)
            compute(raw, lut, slot, kt)

        def tile_step_v2(kt, u, p, more_w, store_next, load_next2, last_a=False):
            """Same data flow as v1, but the W prefetch, the A(kt+1) LDS store and the
            A(kt+2) global load are issued between MFMA steps (step ``W_STEP`` /
            ``ST_STEP``), A fragments (read as step pairs) and B decodes run one step
            ahead, and every step is a separate scheduling region: after the barrier
            only step 0's A reads and B decode stand before the first MFMA."""
            slot = kt % fx.Int32(2)
            rocdl.s_waitcnt(lgkmcnt=0)
            gpu.barrier()  # A(kt) visible; slot (kt+1)%2 free (read at kt-1)
            raw, lut = read_w(wbufs[u][p], kt)
            a_pair = [read_a2(mi, 0, slot) for mi in range_constexpr(MR)]
            b_cur = decode_b(raw, lut, 0)
            rocdl.sched_barrier(0)
            for s in range_constexpr(STEPS):
                if const_expr(s == W_STEP and more_w):
                    load_w(wbufs[u][1 - p], kt + fx.Int32(PF))
                if const_expr(s == ST_STEP and store_next):
                    store_a(fx.Int32(1) - slot, last=last_a)
                    if const_expr(load_next2):
                        load_a(kt + fx.Int32(2))
                a_cur = [Vec.from_elements([a_pair[mi][2 * (s % 2)], a_pair[mi][2 * (s % 2) + 1]],
                                           fx.Int32) for mi in range_constexpr(MR)]
                if const_expr(s % 2 == 1 and s + 1 < STEPS):  # next pair, one step ahead
                    a_pair = [read_a2(mi, s + 1, slot) for mi in range_constexpr(MR)]
                if const_expr(hook is not None):
                    hook(acc, kt, s)
                b_nxt = decode_b(raw, lut, s + 1) if s + 1 < STEPS else None
                mfmas(a_cur, b_cur)
                rocdl.sched_barrier(0)
                b_cur = b_nxt

        # v3: the MFMAs of every tile's last step are issued after the next tile's
        # barrier (their operands wait in a_def / b_def), so the A reads of the new
        # tile overlap MFMA work instead of an idle pipe. Tile 0 "issues" a zero-A
        # deferred step (adds 0).
        a_def = [fx.make_rmem_tensor(fx.make_layout(2, 1), fx.Int32) for _ in range(MR)]
        b_def = {k: fx.make_rmem_tensor(fx.make_layout(2, 1), fx.Int32) for k in keys}
        if const_expr(SCH == 3):
            for mi in range_constexpr(MR):
                a_def[mi].store(Vec.filled(2, 0, fx.Int32))
            for k in keys:
                b_def[k].store(Vec.filled(2, 0, fx.Int32))

        def flush_deferred():
            mfma_block([Vec(fx.memref_load_vec(a_def[mi])) for mi in range_constexpr(MR)],
                       {k: Vec(fx.memref_load_vec(b_def[k])) for k in keys})

        def mfma_block(a_cur, bws):
            """PRIO: the step's MFMAs run at raised wave priority as one scheduling
            region, so the two waves of a SIMD ping-pong (one issues MFMAs, the
            other its decode / LDS / VMEM work) instead of drifting a whole tile
            apart under oldest-first arbitration."""
            if const_expr(PRIO):
                rocdl.sched_barrier(0)
                rocdl.s_setprio(1)
            mfmas(a_cur, bws)
            if const_expr(PRIO):
                rocdl.s_setprio(0)
                rocdl.sched_barrier(0)

        def tile_step_v3(kt, u, p, more_w, store_next, load_next2, last_a=False):
            slot = kt % fx.Int32(2)
            raw, lut = read_w(wbufs[u][p], kt)  # W regs only: before the barrier
            b_cur = decode_b(raw, lut, 0)
            rocdl.s_waitcnt(lgkmcnt=0)
            gpu.barrier()  # A(kt) visible; slot (kt+1)%2 free (read at kt-1)
            a_pair = [read_a2(mi, 0, slot) for mi in range_constexpr(MR)]
            flush_deferred()  # tile kt-1's last step
            rocdl.sched_barrier(0)
            for s in range_constexpr(STEPS):
                if const_expr(s == W_STEP and more_w):
                    load_w(wbufs[u][1 - p], kt + fx.Int32(PF))
                if const_expr(s == ST_STEP and store_next):
                    store_a(fx.Int32(1) - slot, last=last_a)
                    if const_expr(load_next2):
                        load_a(kt + fx.Int32(2))
                a_cur = [Vec.from_elements([a_pair[mi][2 * (s % 2)], a_pair[mi][2 * (s % 2) + 1]],
                                           fx.Int32) for mi in range_constexpr(MR)]
                if const_expr(s % 2 == 1 and s + 1 < STEPS):  # next pair, one step ahead
                    a_pair = [read_a2(mi, s + 1, slot) for mi in range_constexpr(MR)]
                if const_expr(hook is not None):
                    hook(acc, kt, s)
                if const_expr(s + 1 < STEPS):
                    b_nxt = decode_b(raw, lut, s + 1)
                    mfma_block(a_cur, b_cur)
                    rocdl.sched_barrier(0)
                    b_cur = b_nxt
                else:
                    for mi in range_constexpr(MR):
                        a_def[mi].store(a_cur[mi])
                    for k in keys:
                        b_def[k].store(b_cur[k])

        tile_step = {1: tile_step_v1, 2: tile_step_v2, 3: tile_step_v3}[SCH]

        @flyc.jit
        def _k_loop():
            for it in range(KT // (2 * PF) - 1):
                for q in range_constexpr(2 * PF):
                    kt = fx.Int32(it) * fx.Int32(2 * PF) + fx.Int32(q)
                    tile_step(kt, q % PF, q // PF, True, True, True)

        _k_loop()
        for q in range_constexpr(2 * PF):  # last 2*PF tiles
            kt = KT - 2 * PF + q
            tile_step(fx.Int32(kt), q % PF, q // PF, kt + PF < KT, kt + 1 < KT, kt + 2 < KT,
                      kt + 1 == KT - 1)
        if const_expr(SCH == 3):
            flush_deferred()
        rocdl.s_waitcnt(lgkmcnt=0)
        gpu.barrier()  # every wave is done reading A from LDS
        return acc

    return run


# =============================================================================
# gemm1: gate|up + activation (SiLU / SiTUv2)
# =============================================================================


@functools.cache
def compile_ep_gemm1(*, H, I, NE, BM, NI, ROWV, PF=2, bcm=2, wpe=None, NW=4, ARB=256,
                     acm=2, SCH=1, ST=None, PRIO=0, TR=False, ORDER=False, ZRC=False,
                     XG=False, DYN=False, ACT="silu"):
    """Stage 1: ``h = act(A W1g^T) * (A W1u^T)`` for the sorted rows of one expert
    block, NI inter columns per WG.

    ``ACT``: "silu" or "situv2" (as tp_gemm1: SiTUv2 beta / linear_beta / clamp
    limit are runtime f32 launch args, see ``situ_args``; silu ignores them).

    A = fp8 x + f32 row scale; out = fp8 [sorted, I] by sorted row + f32 scale per
    (row, 128 cols) group, ``[sorted, I/128]``.
    ``BM``: sorting block; ``ROWV``: row variants compiled in (ascending, last =
    BM): a WG runs the smallest one that covers its valid rows. ``PF``: W prefetch
    distance in K tiles; ``bcm``: W load cache modifier (2 = nt).
    ``XG``: x scales are per (token, 128-K group), ``[M, H/128]`` (e.g. a
    blockwise-quantized EP dispatch): one accumulator rescaled by s_{g-1}/s_g at
    every group start (as gemm2), times s_last in the epilogue.
    ``DYN``: the grid may be smaller than the tile bound (WGs stride over tiles), and
    instead of zeroing ``zero`` up front, rows < num_valid_ids[1] with != 1 local
    route (``arg_rc``) are zeroed after the tiles.
    """
    assert ACT in ("silu", "situv2"), ACT
    assert not (DYN and ZRC), "DYN does its own route-count-aware zeroing"
    K, N2 = H, 2 * I
    NB = I // NI
    NT = 64 * NW  # threads per WG
    NIW = NI // NW
    NC = NIW // 16  # 16-col blocks per wave, per gate/up
    assert NIW % 16 == 0 and I % NI == 0
    assert NI % QG1 == 0, "quant groups of 128 columns per WG"
    assert ROWV[-1] == BM and all(r % 16 == 0 for r in ROWV)
    assert not TR or (QG1 % NIW == 0 and NI % QG1 == 0), \
        "TR epilogue: every wave inside one 128-col quant group"
    LDS_BYTES = max(2 * BM * ARB, BM * NI * 2)
    # XG: ratio table [G][BM] f32 + fin [BM] after the A double buffer (the bf16
    # epilogue tile and the TR amax table stay below it)
    G = H // QG1
    KS = 32  # K per MFMA step
    TK = ARB
    RATIO_OFF = 2 * BM * ARB
    if XG:
        assert TK % QG1 == 0 and BM * NI * 2 <= RATIO_OFF and BM * NW * 4 <= RATIO_OFF
        LDS_BYTES = RATIO_OFF + (G + 1) * BM * 4
    assert LDS_BYTES <= 65536, LDS_BYTES
    kname = (
        f"ep_gemm1_fp8_h{H}_i{I}_e{NE}_bm{BM}_ni{NI}_r{'x'.join(map(str, ROWV))}"
        f"_pf{PF}_bcm{bcm}_nw{NW}_a{ARB}" + (f"_w{wpe}" if wpe else "")
        + (f"_acm{acm}" if acm != 2 else "") + (f"_s{SCH}" if SCH != 1 else "")
        + (f"_st{ST}" if ST is not None else "") + (f"_p{PRIO}" if PRIO else "")
        + ("_tr" if TR else "") + ("_o" if ORDER else "") + ("_zrc" if ZRC else "")
        + ("_xg" if XG else "") + ("_dyn" if DYN else "")
        + ("" if ACT == "silu" else f"_{ACT}")
    )
    # route-count-aware zeroing of gemm2's out [M, H] bf16: <= ZRC_UPT units / thread
    ZRC_ROW16 = H * 2 // 16
    ZRC_UPT = -(-ZRC_ROW16 // NT)

    def _body(lds_ptr, arg_x, arg_xs, arg_w, arg_ws, arg_wexp, arg_eids, arg_sids,
              arg_out, arg_outs, i32_m, m_block, n_block, lane, wave, tx, situ):
        lane_div_16 = lane // fx.Int32(16)
        lane_mod_16 = lane % fx.Int32(16)
        e = fx.Int32(rocdl.readfirstlane(T.i32, _raw(_global_i32_at(arg_eids, m_block))))
        bx_m = m_block * fx.Int32(BM)
        col0 = n_block * fx.Int32(NI) + wave * fx.Int32(NIW)  # first inter col of wave
        cols = []
        for gu in range_constexpr(2):
            for c in range_constexpr(NC):
                n0 = e * fx.Int32(N2) + fx.Int32(gu * I) + col0 + fx.Int32(c * 16)
                cols.append(((gu, c), n0, (gu, c // 2)))
        exps = {
            (gu, c): fx.Int32(_global_i32_at(
                arg_wexp, e * fx.Int32(N2) + fx.Int32(gu * I) + col0 + fx.Int32(c * 16)
                + lane_mod_16))
            for gu in range(2) for c in range(NC)
        } if not TR else {
            (gu, c, v): fx.Int32(_global_i32_at(
                arg_wexp, e * fx.Int32(N2) + fx.Int32(gu * I) + col0 + fx.Int32(c * 16)
                + lane_div_16 * fx.Int32(4) + fx.Int32(v)))
            for gu in range(2) for c in range(NC) for v in range(4)
        }
        core = _make_core(K=K, W_ROWS=NE * N2, BM=BM, PF=PF, bcm=bcm, cols=cols,
                          lds_ptr=lds_ptr, LDS_BYTES=LDS_BYTES, lane=lane, tx=tx, NT=NT,
                          arg_w=arg_w, arg_ws=arg_ws, arg_a=arg_x,
                          a_num_bytes=fx.Int64(i32_m) * fx.Int64(K), ARB=ARB, acm=acm,
                          SCH=SCH, ST_STEP=ST, PRIO=PRIO, TR=TR)
        lds_i32 = fx.Int32(fx.ptrtoint(lds_ptr))
        lds_f32 = lds_typed_ptr(lds_i32, T.f32)

        def row_base16(row):  # gather x rows by token (padding token = M -> OOB -> 0)
            tok = fx.Int32(_global_i32_at(arg_sids, bx_m + row)) & fx.Int32(0x00FFFFFF)
            return tok * fx.Int32(K // 16)

        def build_ratios(R):
            """XG, LDS: ratio[g][r] = s[tok_r][g-1] / s[tok_r][g] (ratio[0] = 1),
            fin[r] = s[tok_r][G-1]; a zero scale (all-zero group, padding row) acts
            as 1 (its fp8 values are 0)."""
            xs_tiles = _global_i32_buffer_tiles(
                arg_xs, fx.Int64(i32_m) * fx.Int64(G * 4), 1)
            xs_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Int32)
            one = fx.Float32(1.0)
            for p in range_constexpr(-(-(R * G) // NT)):
                idx = tx + fx.Int32(p * NT)
                r = idx // fx.Int32(G)
                g = idx % fx.Int32(G)
                ok = r < fx.Int32(R)
                rr = ok.select(r, fx.Int32(0))
                tok = fx.Int32(_global_i32_at(arg_sids, bx_m + rr)) & fx.Int32(0x00FFFFFF)
                srow = tok * fx.Int32(G)
                cur = _buffer_i32_scalar_read(xs_tiles, srow + g, xs_atom).bitcast(fx.Float32)
                prv = _buffer_i32_scalar_read(
                    xs_tiles, srow + fx.max(g - fx.Int32(1), fx.Int32(0)), xs_atom
                ).bitcast(fx.Float32)
                cur = (cur == fx.Float32(0.0)).select(one, cur)
                prv = (prv == fx.Float32(0.0)).select(one, prv)
                ratio = (g == fx.Int32(0)).select(one, prv / cur)
                _lds_f32_store_masked(lds_f32, fx.Int32(RATIO_OFF // 4) + g * fx.Int32(BM) + rr,
                                      ratio, ok)
                _lds_f32_store_masked(lds_f32, fx.Int32(RATIO_OFF // 4 + G * BM) + rr, cur,
                                      ok & (g == fx.Int32(G - 1)))

        def ratio4(base_f32, mi):
            """this lane's 4 rows (L//16)*4 + v of 16-row block mi from an f32 [BM] row."""
            off = (base_f32 + fx.Int32(mi * 16) + lane_div_16 * fx.Int32(4)) * fx.Int32(4)
            return Vec(lds_vec_load(lds_i32, off, Vec.make_type(4, fx.Float32), fx.Float32,
                                    align=16))

        def hook(acc, kt, s):
            kk = s * KS  # K offset of step s in the tile
            if const_expr(kk % QG1 != 0):
                return
            gidx = kt * fx.Int32(TK // QG1) + fx.Int32(kk // QG1)
            for key in acc:
                for mi in range_constexpr(len(acc[key])):
                    if const_expr(TR):  # lane: row mi*16 + L%16, 4 columns
                        rt = lds_f32[fx.Int32(RATIO_OFF // 4 + mi * 16) + gidx * fx.Int32(BM)
                                     + lane_mod_16]
                        a4 = Vec(acc[key][mi].load())
                        acc[key][mi].store(Vec.from_elements(
                            [a4[v] * rt for v in range(4)], fx.Float32))
                    else:
                        rt = ratio4(fx.Int32(RATIO_OFF // 4) + gidx * fx.Int32(BM), mi)
                        a4 = Vec(acc[key][mi].load())
                        acc[key][mi].store(Vec.from_elements(
                            [a4[v] * rt[v] for v in range(4)], fx.Float32))

        def fin_of(r):
            return fx.Float32(lds_f32[fx.Int32(RATIO_OFF // 4 + G * BM) + r])

        def run_rows(R):
            if const_expr(XG):
                build_ratios(R)  # visible to all waves after the K loop's first barrier
            acc = core(R, row_base16, hook if XG else None)
            if const_expr(TR):
                _epilogue_tr(R, acc)
            else:
                _epilogue(R, acc)

        def _epilogue_tr(R, acc):
            """Transposed acc (lane: row L%16 of each 16-row block, 4 consecutive
            columns): dequant -> act(gate)*up in registers, then the row amax over
            the lane's columns, the 4 lane groups (xor 16 / 32) and the NW waves
            (LDS [row][NW] f32) -> fp8 dword stores + one f32 scale per row."""
            MR = R // 16
            rows = [fx.Int32(mi * 16) + lane_mod_16 for mi in range(MR)]
            toks = [fx.Int32(_global_i32_at(arg_sids, bx_m + r)) & fx.Int32(0x00FFFFFF)
                    for r in rows]
            valid = [t < i32_m for t in toks]
            fac = {k: _channel_factor(v) for k, v in exps.items()}
            if const_expr(XG):
                rs = [fin_of(r) for r in rows]
            else:
                xs_tiles = _global_i32_buffer_tiles(arg_xs, fx.Int64(i32_m) * fx.Int64(4), 1)
                xs_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Int32)
                rs = [_buffer_i32_scalar_read(xs_tiles, t, xs_atom).bitcast(fx.Float32)
                      for t in toks]
            ys = {}
            for mi in range_constexpr(MR):
                for c in range_constexpr(NC):
                    g4 = Vec(fx.memref_load_vec(acc[0, c][mi]))
                    u4 = Vec(fx.memref_load_vec(acc[1, c][mi]))
                    gv = [fx.Float32(g4[v]) * (rs[mi] * fac[0, c, v]) for v in range(4)]
                    uv = [fx.Float32(u4[v]) * (rs[mi] * fac[1, c, v]) for v in range(4)]
                    ys[mi, c] = gate_up_act(ACT, gv, uv, situ)
            col = n_block * fx.Int32(NI) + wave * fx.Int32(NIW) + lane_div_16 * fx.Int32(4)
            for mi in range_constexpr(MR):
                am = fx.Float32(0.0)
                for c in range_constexpr(NC):
                    for v in range_constexpr(4):
                        am = fx.maxnumf(am, fx.absf(ys[mi, c][v]))
                am = fx.maxnumf(am, am.shuffle_xor(16, 64))
                am = fx.maxnumf(am, am.shuffle_xor(32, 64))
                lds_f32[rows[mi] * fx.Int32(NW) + wave] = am  # 4 lanes, same value
            gpu.barrier()
            out_rsrc = create_buffer_resource_from_addr(
                _raw(fx.Int64(arg_out)), num_records_bytes=_raw(
                    fx.Int64(bx_m + fx.Int32(BM)) * fx.Int64(I)))
            os_rsrc = create_buffer_resource_from_addr(
                _raw(fx.Int64(arg_outs)), num_records_bytes=_raw(
                    fx.Int64(bx_m + fx.Int32(BM)) * fx.Int64(I // QG1 * 4)))
            GPW = NI // QG1  # quant groups per WG row
            WPG = NW // GPW  # waves per group
            grp = wave // fx.Int32(WPG)
            for mi in range_constexpr(MR):
                am = fx.Float32(0.0)
                pv = Vec(lds_vec_load(lds_i32, (rows[mi] * fx.Int32(NW) + grp * fx.Int32(WPG))
                                      * fx.Int32(4), Vec.make_type(WPG, fx.Float32),
                                      fx.Float32, align=4 * WPG))
                for j in range_constexpr(WPG):
                    am = fx.maxnumf(am, fx.Float32(pv[j]))
                s = (am == 0.0).select(fx.Float32(1.0), am * fx.Float32(1.0 / 240.0))
                inv = fx.Float32(rocdl.rcp(T.f32, _raw(s)))
                srow = bx_m + rows[mi]
                for c in range_constexpr(NC):
                    q = _fp8x4_pack(*[ys[mi, c][v] * inv for v in range(4)])
                    buffer_store(
                        q, out_rsrc, (srow * fx.Int32(I) + col + fx.Int32(c * 16))
                        // fx.Int32(4), mask=valid[mi])
                buffer_store(
                    s, os_rsrc, srow * fx.Int32(I // QG1) + n_block * fx.Int32(GPW) + grp,
                    mask=valid[mi] & (lane_div_16 == fx.Int32(0))
                    & (wave % fx.Int32(WPG) == fx.Int32(0)))

        def _epilogue(R, acc):
            """dequant -> act(gate)*up -> bf16 LDS tile [R, NI] -> fp8 (row, 128)
            quant, 16 B stores by sorted row."""
            MR = R // 16
            lds_bf16 = lds_typed_ptr(lds_i32, T.bf16)
            fac = {k: _channel_factor(v) for k, v in exps.items()}
            if const_expr(not XG):
                xs_tiles = _global_i32_buffer_tiles(arg_xs, fx.Int64(i32_m) * fx.Int64(4), 1)
                xs_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Int32)
            for mi in range_constexpr(MR):
                for v in range_constexpr(4):
                    rit = fx.Int32(mi * 16 + v) + lane_div_16 * fx.Int32(4)
                    if const_expr(XG):
                        rs = fin_of(rit)
                    else:
                        tok = fx.Int32(_global_i32_at(arg_sids, bx_m + rit)) & fx.Int32(0x00FFFFFF)
                        rs = _buffer_i32_scalar_read(xs_tiles, tok, xs_atom).bitcast(fx.Float32)
                    for c in range_constexpr(NC):
                        gv = fx.Float32(Vec(fx.memref_load_vec(acc[0, c][mi]))[v])
                        uv = fx.Float32(Vec(fx.memref_load_vec(acc[1, c][mi]))[v])
                        gv = gv * (rs * fac[0, c])
                        uv = uv * (rs * fac[1, c])
                        y = gate_up_act(ACT, [gv], [uv], situ)[0]
                        col = wave * fx.Int32(NIW) + fx.Int32(c * 16) + lane_mod_16
                        lds_bf16[rit * fx.Int32(NI) + col] = y.to(fx.BFloat16)
            gpu.barrier()
            _quant_store(R)

        def _quant_store(R):
            # 8 threads x 16 cols per (row, 128-col group); 32 (row, group)s per pass
            TPR = QG1 // 16
            GPW = NI // QG1  # groups per WG row
            RPP = NT // TPR
            out_rsrc = create_buffer_resource_from_addr(
                _raw(fx.Int64(arg_out)), num_records_bytes=_raw(
                    fx.Int64(bx_m + fx.Int32(BM)) * fx.Int64(I)))
            os_rsrc = create_buffer_resource_from_addr(
                _raw(fx.Int64(arg_outs)), num_records_bytes=_raw(
                    fx.Int64(bx_m + fx.Int32(BM)) * fx.Int64(I // QG1 * 4)))
            rg_in = tx // fx.Int32(TPR)
            sub = tx % fx.Int32(TPR)
            for p in range_constexpr(-(-(R * GPW) // RPP)):
                rg = fx.Int32(p * RPP) + rg_in
                rit = rg // fx.Int32(GPW)
                grp = rg % fx.Int32(GPW)
                tok = fx.Int32(_global_i32_at(arg_sids, bx_m + rit)) & fx.Int32(0x00FFFFFF)
                valid = (tok < i32_m) & (rit < fx.Int32(R))
                ccol = grp * fx.Int32(QG1) + sub * fx.Int32(16)  # column in the WG tile
                byte0 = (rit * fx.Int32(NI) + ccol) * fx.Int32(2)
                vals = []
                for h in range_constexpr(2):
                    vv = Vec(lds_vec_load(lds_i32, byte0 + fx.Int32(h * 16),
                                          Vec.make_type(8, fx.BFloat16), fx.BFloat16,
                                          align=16)).to(fx.Float32)
                    vals.extend(vv[k] for k in range(8))
                amax = fx.Float32(0.0)
                for k in range_constexpr(16):
                    amax = fx.max(amax, fx.max(vals[k], -vals[k]))
                for bit in range_constexpr(TPR.bit_length() - 1):
                    amax = fx.max(amax, amax.shuffle_xor(1 << bit, 64))
                s = (amax == 0.0).select(fx.Float32(1.0), amax / 240.0)
                inv = 1.0 / s
                packed = Vec.from_elements(
                    [_pack_activation(*[vals[4 * q + k] * inv for k in range(4)])
                     for q in range(4)], fx.Int32)
                srow = bx_m + rit
                buffer_store(
                    packed, out_rsrc,
                    (srow * fx.Int32(I) + n_block * fx.Int32(NI) + ccol) // fx.Int32(4),
                    mask=valid)
                buffer_store(
                    s, os_rsrc, srow * fx.Int32(I // QG1) + n_block * fx.Int32(GPW) + grp,
                    mask=valid & (sub == fx.Int32(0)))

        if const_expr(len(ROWV) == 1):
            run_rows(BM)
        else:
            _dispatch_rows(ROWV, _row_pads(arg_sids, bx_m, ROWV, i32_m), run_rows)

    @fx.struct
    class SharedStorage:
        raw: fx.Array[fx.Uint8, LDS_BYTES, 16]

    @flyc.jit
    def _zero_dyn(arg_zero, arg_rc, arg_cumsum, bx, grid, tx):
        """DYN: rows r = bx, bx + grid, ... < num_valid_ids[1] of gemm2's out with
        != 1 local route are zeroed (single-route rows are stored whole by gemm2;
        rows past the valid count are never read). One wave-uniform count per row."""
        mv = fx.Int32(rocdl.readfirstlane(T.i32, _raw(fx.Int32(
            _global_i32_at(arg_cumsum, fx.Int32(1))))))
        rc = global_typed_ptr(arg_rc, T.i32)
        n16 = mv * fx.Int32(ZRC_ROW16)
        rs = create_buffer_resource_from_addr(
            _raw(fx.Int64(arg_zero)), num_records_bytes=_raw(fx.Int64(n16) * fx.Int64(16)))
        z = Vec.filled(4, 0, fx.Int32)
        r = bx
        while r < mv:
            rcv = fx.Int32(rocdl.readfirstlane(T.i32, _raw(fx.Int32(rc[r]))))
            for k in range_constexpr(ZRC_UPT):
                u = fx.Int32(k * NT) + tx
                ok = (u < fx.Int32(ZRC_ROW16)) & (rcv != fx.Int32(1))
                buffer_store(
                    z, rs, ok.select(r * fx.Int32(ZRC_ROW16) + u, n16) * fx.Int32(4))
            r = r + grid

    @flyc.jit
    def _entry(arg_x, arg_xs, arg_w, arg_ws, arg_wexp, arg_eids, arg_cumsum, arg_sids,
               arg_out, arg_outs, i32_m, arg_zero, i32_zero16, arg_sched, arg_rc,
               f32_sb, f32_sbr, f32_lb, f32_lbr, f32_lim):
        if const_expr(ACT == "situv2"):
            situ = situ_params(fx.Float32(f32_sb), fx.Float32(f32_sbr), fx.Float32(f32_lb),
                               fx.Float32(f32_lbr), fx.Float32(f32_lim))
        else:
            situ = None
        lds_ptr = fx.SharedAllocator().allocate(SharedStorage).peek().raw.ptr
        tx = fx.Int32(gpu.thread_id("x"))
        bx = fx.Int32(gpu.block_id("x"))
        lane = tx % fx.Int32(64)
        wave = fx.Int32(rocdl.readfirstlane(T.i32, _raw(tx // fx.Int32(64))))
        grid = fx.Int32(gpu.grid_dim.x)
        if const_expr(ZRC):
            zpre = _zero_rc_pre(arg_rc, i32_zero16, i32_m, ZRC_ROW16, bx, grid)
        else:
            if const_expr(not DYN):
                _zero_slice(arg_zero, i32_zero16, bx, grid, tx, NT)
        bound = _global_i32_at(arg_cumsum, fx.Int32(0)) // fx.Int32(BM) * fx.Int32(NB)
        if const_expr(ORDER):
            lo, hi, nwg = _bucket_bounds(arg_sched, bx, grid, NB)
            t = lo + bx // fx.Int32(8)
            while t < hi:
                m_block, n_block = _sched_tile(arg_sched, t, NB)
                txo, laneo, waveo = _opaque_tx(tx)
                _body(lds_ptr, arg_x, arg_xs, arg_w, arg_ws, arg_wexp, arg_eids, arg_sids,
                      arg_out, arg_outs, i32_m, m_block, n_block, laneo, waveo, txo, situ)
                t = t + nwg
                if t < hi:
                    gpu.barrier()
        else:
            if const_expr(DYN):
                t = bx
                while t < bound:
                    m_block, n_block = _tile_of(_xcd_tile(t, bound), NB)
                    txo, laneo, waveo = _opaque_tx(tx)
                    _body(lds_ptr, arg_x, arg_xs, arg_w, arg_ws, arg_wexp, arg_eids,
                          arg_sids, arg_out, arg_outs, i32_m, m_block, n_block, laneo,
                          waveo, txo, situ)
                    t = t + grid
                    if t < bound:
                        gpu.barrier()
            else:
                if bx < bound:
                    tile = _xcd_tile(bx, bound)
                    m_block, n_block = _tile_of(tile, NB)
                    txo, laneo, waveo = _opaque_tx(tx)
                    _body(lds_ptr, arg_x, arg_xs, arg_w, arg_ws, arg_wexp, arg_eids,
                          arg_sids, arg_out, arg_outs, i32_m, m_block, n_block, laneo,
                          waveo, txo, situ)
        if const_expr(ZRC):
            _zero_rc_post(arg_zero, i32_zero16, ZRC_ROW16, ZRC_UPT, zpre, bx, tx, NT)
        if const_expr(DYN):
            _zero_dyn(arg_zero, arg_rc, arg_cumsum, bx, grid, tx)

    @flyc.kernel(name=kname, known_block_size=[NT, 1, 1])
    def kernel(arg_x: fx.Int64, arg_xs: fx.Int64, arg_w: fx.Int64, arg_ws: fx.Int64,
               arg_wexp: fx.Int64, arg_eids: fx.Int64, arg_cumsum: fx.Int64,
               arg_sids: fx.Int64, arg_out: fx.Int64, arg_outs: fx.Int64,
               i32_m: fx.Int32, arg_zero: fx.Int64, i32_zero16: fx.Int32,
               arg_sched: fx.Int64, arg_rc: fx.Int64, f32_sb: fx.Float32,
               f32_sbr: fx.Float32, f32_lb: fx.Float32, f32_lbr: fx.Float32,
               f32_lim: fx.Float32):
        _entry(arg_x, arg_xs, arg_w, arg_ws, arg_wexp, arg_eids, arg_cumsum, arg_sids,
               arg_out, arg_outs, i32_m, arg_zero, i32_zero16, arg_sched, arg_rc,
               f32_sb, f32_sbr, f32_lb, f32_lbr, f32_lim)

    @flyc.jit
    def launch(arg_x: fx.Int64, arg_xs: fx.Int64, arg_w: fx.Int64, arg_ws: fx.Int64,
               arg_wexp: fx.Int64, arg_eids: fx.Int64, arg_cumsum: fx.Int64,
               arg_sids: fx.Int64, arg_out: fx.Int64, arg_outs: fx.Int64,
               i32_m: fx.Int32, arg_zero: fx.Int64, i32_zero16: fx.Int32,
               arg_sched: fx.Int64, arg_rc: fx.Int64, f32_sb: fx.Float32,
               f32_sbr: fx.Float32, f32_lb: fx.Float32, f32_lbr: fx.Float32,
               f32_lim: fx.Float32, i32_grid: fx.Int32, stream: fx.Stream):
        kernel(arg_x, arg_xs, arg_w, arg_ws, arg_wexp, arg_eids, arg_cumsum, arg_sids,
               arg_out, arg_outs, i32_m, arg_zero, i32_zero16, arg_sched, arg_rc,
               f32_sb, f32_sbr, f32_lb, f32_lbr, f32_lim,
               value_attrs=({"rocdl.waves_per_eu": wpe} if wpe else None),
               ).launch(grid=(fx.Int64(i32_grid), 1, 1), block=(NT, 1, 1), stream=stream)

    return launch


def ep_gemm1(x, x_scale, W, sorted_ids, sorted_expert_ids, num_valid_ids, out, out_scale,
             *, block_m, ni=128, rowv=None, pf=2, bcm=2, wpe=None, nw=4, arb=256,
             zero=None, acm=2, sch=1, st=None, prio=0, tr=False, sched=None, zero_rc=None,
             x_group=False, dyn=False, grid=None, act="silu", situ_beta=1.0,
             situ_linear_beta=1.0, swiglu_limit=float("inf")):
    """Host launcher. ``x``: fp8 [M, H] + ``x_scale`` f32 [M];
    ``out``: fp8 (u8) [sorted, I] + ``out_scale`` f32 [sorted, I/128]; ``W`` from ``prepare_weights``. ``zero``: optional tensor (e.g.
    gemm2's output) the WGs set to 0 on the way (nbytes % 16 == 0). ``sched``: the
    prep's tile schedule (XCD buckets, longest tiles first) instead of the plain
    m-major order. ``zero_rc``: prep's local route counts -> only rows with != 1
    route are zeroed (``zero`` = gemm2's out [M, H] run with ``single_store``).
    ``x_group``: ``x_scale`` is f32 [M, H/128] (per token and 128-K group).
    ``dyn``: prep ran with ``m_valid``: only rows < num_valid_ids[1] of ``zero`` with
    != 1 route (``zero_rc`` required) are zeroed; ``grid`` (a multiple of 8) may be
    below the tile bound (the WGs stride over the tiles).
    ``act``: "silu" or "situv2" with ``situ_beta`` / ``situ_linear_beta`` /
    ``swiglu_limit`` (+inf: no clamp), as ``tp_gemm1.a8w4_gemm1``."""
    M, H = x.shape
    NE, N2, _ = W["w1"].shape
    I = N2 // 2
    if grid is None:
        grid = _grid(sorted_expert_ids.numel() * (I // ni), sched)
    if dyn:
        assert zero is not None and zero_rc is not None and grid % 8 == 0
        zrc = False
    elif zero_rc is not None:
        assert zero is not None and zero.shape == (M, H) and zero.dtype == torch.bfloat16
        # a WG's share must span <= 2 rows (<= one row of units); else zero all rows
        zrc = -(-(M * H * 2 // 16) // grid) <= H * 2 // 16
    else:
        zrc = False
    launch = compile_ep_gemm1(H=H, I=I, NE=NE, BM=block_m, NI=ni,
                              ROWV=tuple(rowv or (block_m,)), PF=pf, bcm=bcm, wpe=wpe,
                              NW=nw, ARB=arb, acm=acm, SCH=sch, ST=st, PRIO=prio, TR=tr,
                              ORDER=sched is not None, ZRC=zrc, XG=x_group, DYN=dyn,
                              ACT=act)
    _run_compiled(
        launch, x.data_ptr(), x_scale.data_ptr(),
        W["w1"].data_ptr(), W["w1_scale"].data_ptr(), W["w1_exp"].data_ptr(),
        sorted_expert_ids.data_ptr(), num_valid_ids.data_ptr(), sorted_ids.data_ptr(),
        out.data_ptr(), out_scale.data_ptr(),
        int(M), zero.data_ptr() if zero is not None else 0,
        int(zero.numel() * zero.element_size() // 16) if zero is not None else 0,
        sched.data_ptr() if sched is not None else 0,
        zero_rc.data_ptr() if (zrc or dyn) else 0,
        *situ_args(situ_beta, situ_linear_beta, swiglu_limit),
        int(grid), torch.cuda.current_stream())
    return out


# =============================================================================
# gemm2: down-proj + routed bf16x2 atomic scatter
# =============================================================================


@functools.cache
def compile_ep_gemm2(*, H, I, NE, BM, NO, ROWV, PF=2, bcm=2, wpe=None, NW=4, ARB=256,
                     acm=2, SCH=1, ST=None, PRIO=0, TR=False, ORDER=False, SST=False,
                     DYN=False):
    """Stage 2: ``out[token] += w_route * (A W2^T)`` for the sorted rows of one
    expert block, NO output columns per WG (bf16x2 atomics, ``out`` pre-zeroed).

    A = gemm1's fp8 [sorted, I] with one f32 scale per (row, 128) group
    (``[sorted, I/128]``): one accumulator, rescaled by s_{g-1}/s_g at every group
    start, times s_last in the epilogue."""
    K, N = I, H
    TK = ARB
    NB = N // NO
    NT = 64 * NW  # threads per WG
    NOW = NO // NW
    RPS = NT // 32  # rows per atomic scatter pass (32 lanes x 2 cols per row)
    NC = NOW // 16
    assert NOW % 16 == 0 and N % NO == 0 and NO % 64 == 0
    assert ROWV[-1] == BM and all(r % 16 == 0 for r in ROWV)
    QG = QG1
    G = K // QG  # A scale groups
    assert TK % QG == 0, (TK, QG)
    KS = 32  # K per MFMA step
    RATIO_OFF = 2 * BM * ARB  # ratio table [G][BM] f32 + fin [BM] after A
    LDS_BYTES = max(RATIO_OFF + (G + 1) * BM * 4, BM * NO * 2)
    assert LDS_BYTES <= 65536, LDS_BYTES
    kname = (
        f"ep_gemm2_fp8_h{H}_i{I}_e{NE}_bm{BM}_no{NO}_r{'x'.join(map(str, ROWV))}"
        f"_pf{PF}_bcm{bcm}_nw{NW}_a{ARB}" + (f"_w{wpe}" if wpe else "")
        + (f"_acm{acm}" if acm != 2 else "") + (f"_s{SCH}" if SCH != 1 else "")
        + (f"_st{ST}" if ST is not None else "") + (f"_p{PRIO}" if PRIO else "")
        + ("_tr" if TR else "") + ("_o" if ORDER else "") + ("_sst" if SST else "")
        + ("_dyn" if DYN else "")
    )
    assert not (SST and TR), "single_store: LDS-staged epilogue only"

    def _body(lds_ptr, arg_a, arg_as, arg_w, arg_ws, arg_wexp, arg_eids, arg_sids, arg_sw,
              arg_out, i32_m, i32_sorted, m_block, n_block, lane, wave, tx):
        lane_div_16 = lane // fx.Int32(16)
        lane_mod_16 = lane % fx.Int32(16)
        e = fx.Int32(rocdl.readfirstlane(T.i32, _raw(_global_i32_at(arg_eids, m_block))))
        bx_m = m_block * fx.Int32(BM)
        col0 = n_block * fx.Int32(NO) + wave * fx.Int32(NOW)
        cols = [((0, c), e * fx.Int32(N) + col0 + fx.Int32(c * 16), (0, c // 2))
                for c in range(NC)]
        exps = [fx.Int32(_global_i32_at(arg_wexp, e * fx.Int32(N) + col0 + fx.Int32(c * 16)
                                        + lane_mod_16)) for c in range(NC)] if not TR else [
            [fx.Int32(_global_i32_at(arg_wexp, e * fx.Int32(N) + col0 + fx.Int32(c * 16)
                                     + lane_div_16 * fx.Int32(4) + fx.Int32(v)))
             for v in range(4)] for c in range(NC)]
        core = _make_core(K=K, W_ROWS=NE * N, BM=BM, PF=PF, bcm=bcm, cols=cols,
                          lds_ptr=lds_ptr, LDS_BYTES=LDS_BYTES, lane=lane, tx=tx, NT=NT,
                          arg_w=arg_w, arg_ws=arg_ws, arg_a=arg_a,
                          a_num_bytes=fx.Int64(i32_sorted) * fx.Int64(K), ARB=ARB,
                          acm=acm, SCH=SCH, ST_STEP=ST, PRIO=PRIO, TR=TR)
        lds_i32 = fx.Int32(fx.ptrtoint(lds_ptr))
        lds_f32 = lds_typed_ptr(lds_i32, T.f32)

        def row_base16(row):  # A rows by sorted position (contiguous per block)
            return (bx_m + row) * fx.Int32(K // 16)

        def build_ratios(R):
            """LDS: ratio[g][r] = s[r][g-1] / s[r][g] (ratio[0] = 1), fin[r] = s[r][G-1]."""
            as_tiles = _global_i32_buffer_tiles(
                arg_as, fx.Int64(i32_sorted) * fx.Int64(G * 4), 1)
            as_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Int32)
            for p in range_constexpr(-(-(R * G) // NT)):
                idx = tx + fx.Int32(p * NT)
                r = idx // fx.Int32(G)
                g = idx % fx.Int32(G)
                ok = r < fx.Int32(R)
                rr = ok.select(r, fx.Int32(0))
                srow = (bx_m + rr) * fx.Int32(G)
                cur = _buffer_i32_scalar_read(as_tiles, srow + g, as_atom).bitcast(fx.Float32)
                prv = _buffer_i32_scalar_read(
                    as_tiles, srow + fx.max(g - fx.Int32(1), fx.Int32(0)), as_atom
                ).bitcast(fx.Float32)
                cur_ok = (cur == fx.Float32(0.0)).select(fx.Float32(1.0), cur)
                ratio = (g == fx.Int32(0)).select(fx.Float32(1.0), prv / cur_ok)
                _lds_f32_store_masked(lds_f32, fx.Int32(RATIO_OFF // 4) + g * fx.Int32(BM) + rr,
                                      ratio, ok)
                _lds_f32_store_masked(lds_f32, fx.Int32(RATIO_OFF // 4 + G * BM) + rr, cur,
                                      ok & (g == fx.Int32(G - 1)))

        def ratio4(base_f32, mi):
            """this lane's 4 rows (L//16)*4 + v of 16-row block mi from an f32 [BM] row."""
            off = (base_f32 + fx.Int32(mi * 16) + lane_div_16 * fx.Int32(4)) * fx.Int32(4)
            return Vec(lds_vec_load(lds_i32, off, Vec.make_type(4, fx.Float32), fx.Float32,
                                    align=16))

        def hook(acc, kt, s):
            kk = s * KS  # K offset of step s in the tile
            if const_expr(kk % QG != 0):
                return
            gidx = kt * fx.Int32(TK // QG) + fx.Int32(kk // QG)
            for mi in range_constexpr(len(acc[0, 0])):
                if const_expr(TR):  # lane: row mi*16 + L%16, 4 columns
                    rt = lds_f32[fx.Int32(RATIO_OFF // 4 + mi * 16) + gidx * fx.Int32(BM)
                                 + lane_mod_16]
                    for c in range_constexpr(NC):
                        a4 = Vec(acc[0, c][mi].load())
                        acc[0, c][mi].store(Vec.from_elements(
                            [a4[v] * rt for v in range(4)], fx.Float32))
                else:
                    rt = ratio4(fx.Int32(RATIO_OFF // 4) + gidx * fx.Int32(BM), mi)
                    for c in range_constexpr(NC):
                        a4 = Vec(acc[0, c][mi].load())
                        acc[0, c][mi].store(Vec.from_elements(
                            [a4[v] * rt[v] for v in range(4)], fx.Float32))

        def run_rows(R):
            build_ratios(R)  # visible to all waves after the K loop's first barrier
            acc = core(R, row_base16, hook)
            if const_expr(TR):
                _epilogue_tr(R, acc)
            else:
                _epilogue(R, acc)

        def _epilogue_tr(R, acc):
            """Transposed acc: lane = row mi*16 + L%16 x 4 consecutive columns ->
            acc * fin * route weight * channel factor -> 2 bf16x2 atomics per (mi, c)
            straight from registers (no staging tile, no barrier)."""
            MR = R // 16
            fac = [[_channel_factor(x) for x in xs4] for xs4 in exps]
            sw_tiles = _global_i32_buffer_tiles(arg_sw, fx.Int64(i32_sorted) * fx.Int64(4), 1)
            sw_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Int32)
            for mi in range_constexpr(MR):
                r = fx.Int32(mi * 16) + lane_mod_16
                tok = fx.Int32(_global_i32_at(arg_sids, bx_m + r)) & fx.Int32(0x00FFFFFF)
                wt = _buffer_i32_scalar_read(sw_tiles, bx_m + r, sw_atom).bitcast(fx.Float32)
                wt = wt * lds_f32[fx.Int32(RATIO_OFF // 4 + G * BM) + r]
                for c in range_constexpr(NC):
                    a4 = Vec(acc[0, c][mi].load())
                    v4 = [a4[v] * (wt * fac[c][v]) for v in range(4)]
                    base = tok * fx.Int32(N) + col0 + fx.Int32(c * 16) + lane_div_16 * fx.Int32(4)
                    _atomic_bf16x4(arg_out, base, v4, tok < i32_m)

        def _epilogue(R, acc):
            """acc * fin * channel factor * route weight -> bf16 LDS tile [R, NO] ->
            bf16x2 atomics, 128 B per row (8 rows x 32 lanes x 2 cols per pass)."""
            MR = R // 16
            fac = [_channel_factor(x) for x in exps]
            sw_tiles = _global_i32_buffer_tiles(arg_sw, fx.Int64(i32_sorted) * fx.Int64(4), 1)
            sw_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Int32)
            lds_bf16 = lds_typed_ptr(lds_i32, T.bf16)
            vals = {}
            for mi in range_constexpr(MR):
                rows4 = fx.Int32(mi * 16) + lane_div_16 * fx.Int32(4)
                wts = [_buffer_i32_scalar_read(sw_tiles, bx_m + rows4 + fx.Int32(v), sw_atom)
                       .bitcast(fx.Float32) for v in range(4)]
                fin = ratio4(fx.Int32(RATIO_OFF // 4 + G * BM), mi)
                wts = [wts[v] * fin[v] for v in range(4)]
                for c in range_constexpr(NC):
                    a4 = Vec(acc[0, c][mi].load())
                    vals[mi, c] = [a4[v] * (wts[v] * fac[c]) for v in range(4)]
            gpu.barrier()  # ratio table / A region free for the staging tile
            for mi in range_constexpr(MR):
                rows4 = fx.Int32(mi * 16) + lane_div_16 * fx.Int32(4)
                for c in range_constexpr(NC):
                    col = wave * fx.Int32(NOW) + fx.Int32(c * 16) + lane_mod_16
                    bf4 = Vec.from_elements(vals[mi, c], fx.Float32).to(fx.BFloat16)
                    for v in range_constexpr(4):
                        lds_bf16[(rows4 + fx.Int32(v)) * fx.Int32(NO) + col] = bf4[v]
            gpu.barrier()
            _scatter(R)

        @flyc.jit
        def _scatter_row(rib, tok, single):
            if tok < i32_m:  # padding rows carry token id M
                cs = (tx % fx.Int32(32)) * fx.Int32(2)
                base = tok * fx.Int32(N) + n_block * fx.Int32(NO) + cs
                for s in range_constexpr(NO // 64):
                    pk = lds_vec_load(lds_i32, (rib * fx.Int32(NO) + cs + fx.Int32(s * 64))
                                      * fx.Int32(2), Vec.make_type(2, fx.BFloat16),
                                      fx.BFloat16, align=4)
                    if const_expr(SST):
                        _st_or_add_bf16x2(arg_out, base + fx.Int32(s * 64), pk, single)
                    else:
                        llvm.AtomicRMWOp(
                            llvm.AtomicBinOp.fadd,
                            get_element_ptr(_global_base_ptr1(arg_out), (base + fx.Int32(s * 64)) * fx.Int32(2)),
                            _raw(pk), llvm.AtomicOrdering.monotonic, syncscope="agent",
                            alignment=4)

        def _scatter(R):
            ids = []
            for p in range_constexpr(-(-R // RPS)):
                rib = fx.Int32(p * RPS) + tx // fx.Int32(32)
                sid = fx.Int32(_global_i32_at(arg_sids, bx_m + fx.min(rib, fx.Int32(R - 1))))
                tok = sid & fx.Int32(0x00FFFFFF)
                # single-route token (prep's SID_SINGLE, the sign bit): store, no atomic
                ids.append((rib, (rib < fx.Int32(R)).select(tok, i32_m), sid < fx.Int32(0)))
            rocdl.s_waitcnt(vmcnt=0)  # token ids landed before the first atomic
            for p in range_constexpr(-(-R // RPS)):
                _scatter_row(ids[p][0], ids[p][1], ids[p][2])

        if const_expr(len(ROWV) == 1):
            run_rows(BM)
        else:
            _dispatch_rows(ROWV, _row_pads(arg_sids, bx_m, ROWV, i32_m), run_rows)

    @fx.struct
    class SharedStorage:
        raw: fx.Array[fx.Uint8, LDS_BYTES, 16]

    @flyc.jit
    def _entry(arg_a, arg_as, arg_w, arg_ws, arg_wexp, arg_eids, arg_cumsum, arg_sids,
               arg_sw, arg_out, i32_m, i32_sorted, arg_sched):
        lds_ptr = fx.SharedAllocator().allocate(SharedStorage).peek().raw.ptr
        tx = fx.Int32(gpu.thread_id("x"))
        bx = fx.Int32(gpu.block_id("x"))
        lane = tx % fx.Int32(64)
        wave = fx.Int32(rocdl.readfirstlane(T.i32, _raw(tx // fx.Int32(64))))
        grid = fx.Int32(gpu.grid_dim.x)
        bound = _global_i32_at(arg_cumsum, fx.Int32(0)) // fx.Int32(BM) * fx.Int32(NB)
        if const_expr(ORDER):
            lo, hi, nwg = _bucket_bounds(arg_sched, bx, grid, NB)
            t = lo + bx // fx.Int32(8)
            while t < hi:
                m_block, n_block = _sched_tile(arg_sched, t, NB)
                txo, laneo, waveo = _opaque_tx(tx)
                _body(lds_ptr, arg_a, arg_as, arg_w, arg_ws, arg_wexp, arg_eids, arg_sids,
                      arg_sw, arg_out, i32_m, i32_sorted, m_block, n_block, laneo, waveo, txo)
                t = t + nwg
                if t < hi:
                    gpu.barrier()
        else:
            if const_expr(DYN):  # the grid may be below the tile bound: stride
                t = bx
                while t < bound:
                    m_block, n_block = _tile_of(_xcd_tile(t, bound), NB)
                    txo, laneo, waveo = _opaque_tx(tx)
                    _body(lds_ptr, arg_a, arg_as, arg_w, arg_ws, arg_wexp, arg_eids,
                          arg_sids, arg_sw, arg_out, i32_m, i32_sorted, m_block, n_block,
                          laneo, waveo, txo)
                    t = t + grid
                    if t < bound:
                        gpu.barrier()
            else:
                if bx < bound:
                    tile = _xcd_tile(bx, bound)
                    m_block, n_block = _tile_of(tile, NB)
                    txo, laneo, waveo = _opaque_tx(tx)
                    _body(lds_ptr, arg_a, arg_as, arg_w, arg_ws, arg_wexp, arg_eids,
                          arg_sids, arg_sw, arg_out, i32_m, i32_sorted, m_block, n_block,
                          laneo, waveo, txo)

    @flyc.kernel(name=kname, known_block_size=[NT, 1, 1])
    def kernel(arg_a: fx.Int64, arg_as: fx.Int64, arg_w: fx.Int64, arg_ws: fx.Int64,
               arg_wexp: fx.Int64, arg_eids: fx.Int64, arg_cumsum: fx.Int64,
               arg_sids: fx.Int64, arg_sw: fx.Int64, arg_out: fx.Int64, i32_m: fx.Int32,
               i32_sorted: fx.Int32, arg_sched: fx.Int64):
        _entry(arg_a, arg_as, arg_w, arg_ws, arg_wexp, arg_eids, arg_cumsum, arg_sids, arg_sw,
               arg_out, i32_m, i32_sorted, arg_sched)

    @flyc.jit
    def launch(arg_a: fx.Int64, arg_as: fx.Int64, arg_w: fx.Int64, arg_ws: fx.Int64,
               arg_wexp: fx.Int64, arg_eids: fx.Int64, arg_cumsum: fx.Int64,
               arg_sids: fx.Int64, arg_sw: fx.Int64, arg_out: fx.Int64, i32_m: fx.Int32,
               i32_sorted: fx.Int32, arg_sched: fx.Int64, i32_grid: fx.Int32,
               stream: fx.Stream):
        kernel(arg_a, arg_as, arg_w, arg_ws, arg_wexp, arg_eids, arg_cumsum, arg_sids, arg_sw,
               arg_out, i32_m, i32_sorted, arg_sched,
               value_attrs=({"rocdl.waves_per_eu": wpe} if wpe else None),
               ).launch(grid=(fx.Int64(i32_grid), 1, 1), block=(NT, 1, 1), stream=stream)

    return launch


def ep_gemm2(a, a_scale, W, sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids,
             out, *, block_m, no=128, rowv=None, pf=2, bcm=2, wpe=None, nw=4, arb=256,
             acm=2, sch=1, st=None, prio=0, tr=False, sched=None, single_store=False,
             dyn=False, grid=None):
    """Host launcher. ``a``: fp8 (u8) [sorted, I] + ``a_scale`` f32 [sorted, I/128];
    ``out`` bf16 [tokens, H], pre-zeroed, atomically accumulated.
    ``sched``: see ``ep_gemm1``. ``single_store``: rows of tokens with one local
    route (prep's SID_SINGLE flag) are stored instead of atomically added, so they
    need no zeroing (``ep_gemm1(zero_rc=...)``); ``out`` must be private to this
    call (a store overwrites, it does not accumulate)."""
    M, H = out.shape
    NE = W["w2"].shape[0]
    sorted_rows, I = a.shape
    launch = compile_ep_gemm2(H=H, I=I, NE=NE, BM=block_m, NO=no,
                              ROWV=tuple(rowv or (block_m,)), PF=pf, bcm=bcm, wpe=wpe,
                              NW=nw, ARB=arb, acm=acm, SCH=sch, ST=st, PRIO=prio, TR=tr,
                              ORDER=sched is not None, SST=single_store, DYN=dyn)
    if grid is None:
        grid = _grid(sorted_expert_ids.numel() * (H // no), sched)
    assert not dyn or grid % 8 == 0
    _run_compiled(
        launch, a.data_ptr(), a_scale.data_ptr(),
        W["w2"].data_ptr(), W["w2_scale"].data_ptr(), W["w2_exp"].data_ptr(),
        sorted_expert_ids.data_ptr(), num_valid_ids.data_ptr(), sorted_ids.data_ptr(),
        sorted_weights.data_ptr(), out.data_ptr(), int(M), int(sorted_rows),
        sched.data_ptr() if sched is not None else 0, int(grid), torch.cuda.current_stream())
    return out


# =============================================================================
# end to end
# =============================================================================

# (gemm1 kwargs, gemm2 kwargs) per name; block_m is shared by both stages.
# W loads: nt ("bcm" 2, the default) except in l128c: cached W (bcm 0) only pays off where
# the m-blocks of an expert re-read its W (>= ~1.5 blocks of rows per expert, 6k+ EP8
# tokens: 1-4% faster); below, it is up to 5% (l128) and 10% (t16 - s32) slower, timed
# on rotated weight copies. experiments/log_20261115.md
CFGS = {
    # <= 8 tokens: 1 - 8 active experts; narrower n-tiles put more CUs on the W
    # stream (gemm1 keeps 128-col quant groups)
    "t16": dict(bm=16, sched=False, g1=dict(ni=128, nw=8, sch=3), g2=dict(no=64, nw=4, sch=3)),
    # small blocks: the plain m-major tile order (the prep's order cuts whole
    # 16 / 32-row blocks per XCD: coarser than the tile split, no gain to win)
    "s16": dict(bm=16, sched=False, g1=dict(ni=128, nw=8, sch=3), g2=dict(no=256, nw=8, sch=3)),
    "s32": dict(bm=32, sched=False, g1=dict(ni=128, nw=8, sch=3), g2=dict(no=256, nw=8, sch=3)),
    "m64": dict(bm=64, rowv=(32, 64), g1=dict(ni=128, nw=8, arb=128, sch=3, tr=True),
                g2=dict(no=256, nw=8, arb=128, sch=3, pf=1)),
    "l128": dict(bm=128, rowv=(32, 64, 128), g1=dict(ni=128, nw=8, arb=128, sch=3, tr=True),
                 g2=dict(no=256, nw=8, arb=128, sch=3)),
    "l128c": dict(bm=128, rowv=(32, 64, 128),
                  g1=dict(ni=128, nw=8, arb=128, sch=3, tr=True, bcm=0),
                  g2=dict(no=256, nw=8, arb=128, sch=3, bcm=0)),
    # pre-tuning baseline (sch1, no TR, plain order), kept for A/B
    "l128_v1": dict(bm=128, rowv=(32, 64, 128), sched=False, g1=dict(ni=128, nw=8),
                    g2=dict(no=256, nw=8, arb=128)),
}

# token-count thresholds of the auto config (EP8 MI300X, uniform routing A/B; l128c from
# the 20261115 table: l128 wins at 4096 tokens, l128c from 6144)
AUTO = ((8, "t16"), (480, "s16"), (960, "s32"), (2176, "m64"), (5120, "l128"), (None, "l128c"))


def pick_cfg(tokens):
    """CFGS entry for ``tokens`` (rows of x) -- see AUTO."""
    for lim, name in AUTO:
        if lim is None or tokens <= lim:
            return CFGS[name]


def ep_moe(x, W, topk_weights, topk_ids, *, e0, cfg=None, act="silu", **situ):
    """EP8 local MoE: bf16 x [M, H] -> bf16 [M, H] = sum over the token's LOCAL
    experts of w * expert(x) (global ``topk_ids``; the rank owns the experts
    [e0, e0 + E), E = the local weight count, in that order).
    ``cfg``: an entry of ``CFGS`` (None: ``pick_cfg(M)``). ``act``: "silu" or "situv2"; ``situ``: situ_beta /
    situ_linear_beta / swiglu_limit (see ``ep_gemm1``).
    3 launches: prep (sort + x quant), gemm1 (+ zeroes the output), gemm2."""
    M, H = x.shape
    cfg = cfg if cfg is not None else pick_cfg(M)
    E = W["w2"].shape[0]
    bm = cfg["bm"]
    rowv = cfg.get("rowv", (bm,))
    I = W["w2"].shape[2] * 2
    sids, sw, seids, nv, x8, xs, sched, rc = PR.ep_prep(
        topk_ids, topk_weights, x, e0=e0, block_m=bm, quant=True, num_local=E)
    sorted_size = seids.numel() * bm
    sc = sched if cfg.get("sched", True) else None
    sst = cfg.get("sst", True)  # single-route tokens: store in gemm2, no zeroing
    g1 = dict(dict(rowv=rowv, sched=sc, zero_rc=rc if sst else None, act=act, **situ),
              **cfg["g1"])
    g2 = dict(dict(rowv=rowv, sched=sc, single_store=sst), **cfg["g2"])
    out = torch.empty((M, H), dtype=torch.bfloat16, device=x.device)
    a = torch.empty((sorted_size, I), dtype=torch.uint8, device=x.device)
    a_s = torch.empty((sorted_size, I // QG1), dtype=torch.float32, device=x.device)
    ep_gemm1(x8, xs, W, sids, seids, nv, a, a_s, block_m=bm, zero=out, **g1)
    ep_gemm2(a, a_s, W, sids, sw, seids, nv, out, block_m=bm, **g2)
    return out


def _dyn_grid(tiles):
    return max(8, -(-tiles // 8) * 8)


def ep_moe_dyn(x, x_scale, W, topk_weights, topk_ids, m_valid, *, e0, est_tokens=None,
               out=None, cfg=None, act="silu", **situ):
    """a8w4 EP MoE on an all-to-all receive buffer (e.g. MORI dispatch).

    ``x`` [M, H] is capacity sized: only rows [0, m_valid[0]) are valid (``m_valid``:
    int32 device tensor, read by the kernels, so no host sync / CUDA-graph safe);
    ``topk_ids`` (global ids, the rank owns [e0, e0 + E)) / ``topk_weights``
    [M, topk] likewise. ``x``: fp8 FNUZ with ``x_scale`` f32 [M, H/128] (blockwise
    dispatch quant) or [M] / [M, 1] (per token), or bf16 with ``x_scale`` None
    (quantized per token by the prep).
    ``est_tokens``: host estimate of the valid rows, only for the config and the
    GEMM grids (the WGs stride over the tiles, so any value is correct).
    ``act`` / ``situ``: as ``ep_moe``.
    Returns bf16 [M, H] (or ``out``); rows >= m_valid are left undefined."""
    M, H = x.shape
    est = M if est_tokens is None else max(1, min(M, int(est_tokens)))
    cfg = cfg if cfg is not None else pick_cfg(est)
    E = W["w2"].shape[0]
    K = topk_ids.shape[1]
    bm = cfg["bm"]
    rowv = cfg.get("rowv", (bm,))
    I = W["w2"].shape[2] * 2
    quant = x.dtype == torch.bfloat16
    sids, sw, seids, nv, x8, xs, sched, rc = PR.ep_prep(
        topk_ids, topk_weights, x, e0=e0, block_m=bm, quant=quant, num_local=E,
        m_valid=m_valid)
    x_group = False
    if not quant:
        x8 = x
        if x_scale.dim() == 2 and x_scale.shape[1] > 1:
            assert x_scale.shape == (M, H // QG1), x_scale.shape
            xs, x_group = x_scale.contiguous(), True
        else:
            xs = x_scale.reshape(M).contiguous()
    sorted_size = seids.numel() * bm
    sc = sched if cfg.get("sched", True) else None
    # grid: the m-blocks the estimate can produce (capped by the capacity)
    nblk = min(seids.numel(), est * K // bm + E)
    g1 = dict(dict(rowv=rowv, sched=sc, zero_rc=rc, act=act, **situ), **cfg["g1"])
    g2 = dict(dict(rowv=rowv, sched=sc, single_store=True), **cfg["g2"])
    if out is None:
        out = torch.empty((M, H), dtype=torch.bfloat16, device=x.device)
    a = torch.empty((sorted_size, I), dtype=torch.uint8, device=x.device)
    a_s = torch.empty((sorted_size, I // QG1), dtype=torch.float32, device=x.device)
    ep_gemm1(x8, xs, W, sids, seids, nv, a, a_s, block_m=bm, zero=out,
             x_group=x_group, dyn=True, grid=_dyn_grid(nblk * (I // g1["ni"])), **g1)
    ep_gemm2(a, a_s, W, sids, sw, seids, nv, out, block_m=bm, dyn=True,
             grid=_dyn_grid(nblk * (H // g2["no"])), **g2)
    return out
