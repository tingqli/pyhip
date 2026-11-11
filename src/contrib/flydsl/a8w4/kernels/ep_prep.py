# Package copy of doc-mimo/kernels/ep8_prep.py (test harness removed, env knobs frozen).
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""EP8 prep kernel (gfx942, FlyDSL): local-expert routing sort + x row quant in ONE
launch, replacing aiter ``moe_sorting`` (with ``expert_mask``) + ``quantize_rows``.

Outputs the moe_sorting contract restricted to the rank's EL local experts:
``sorted_ids`` (token | slot << 24, padding = M | topk << 24), ``sorted_weights``,
``sorted_expert_ids`` (local id per BM block), ``num_valid_ids`` = [padded rows, M];
capacity M*topk + EL*BM rows (any routing), so the GEMM grids stay small. With
``quant`` also fp8 x + f32 row scale for every token with >= 1 local expert, bit
identical to ``quantize_rows`` (other rows are not read by gemm1). WG 0 also writes
the GEMMs' tile schedule (8 cost-balanced XCD ranges of m-blocks, long tiles first;
see "GEMM tile order") after the block expert ids.

Sort, two compiled variants (host picks by M * topk):
  * single WG (``MULTI=False``, <= 256*EPT entries): WG 0 counts with LDS atomics,
    computes the padded offsets and scatters; no global atomics, no grid sync.
  * multi WG: each WG counts its EPT*256 entries in LDS and reserves its range per
    expert with one global atomic per expert; grid barrier (grid <= 2 WG/CU, all
    resident); offsets from the global counts, scatter, padding rows, block experts;
    the last WG resets the counters (graph-replay safe).
The row order inside an expert depends on atomic order (not deterministic).

x quant: one wave per row; each wave first loads the topk ids of all its rows (lane i
= row i) and ballots the rows with a local expert, then walks them with the next
row's x loads issued before the current row is reduced and packed.

``MDYN`` (EP all-to-all receive buffers, e.g. MORI): ``M`` is the buffer capacity
(padding token id, buffer layout) while only the first ``m_valid`` rows (an i32 read
on device, clamped to M) are sorted / quantized; ``num_valid_ids[1]`` = m_valid. A
MULTI-compiled kernel then picks the single-WG sort at run time when the valid
entries fit one WG.
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


NT = 256
# GEMM tile schedule (i32, right after the cap // BM block expert ids): [0, 9): XCD
# bucket starts (m-block slots); [SCHED_BORDER, + nblk): the m-block order.
SCHED_BORDER = 16
# sorted id flag: the token has exactly one local route (see _scatter)
SID_SINGLE = -(1 << 31)
NUM_CU = 304


def _gatomic_add(addr_i64, idx, val):
    """old = atomic add, agent scope (device-coherent across the XCD L2s)."""
    return fx.Int32(llvm.AtomicRMWOp(
        llvm.AtomicBinOp.add, _gep1(_global_base_ptr1(addr_i64), idx * fx.Int32(4)),
        _raw(fx.Int32(val)), llvm.AtomicOrdering.monotonic, syncscope="agent",
        alignment=4).result)


def _gatomic_xchg(addr_i64, idx, val):
    llvm.AtomicRMWOp(
        llvm.AtomicBinOp.xchg, _gep1(_global_base_ptr1(addr_i64), idx * fx.Int32(4)),
        _raw(fx.Int32(val)), llvm.AtomicOrdering.monotonic, syncscope="agent", alignment=4)


_CM_SYS = 17  # sc0|sc1: system-coherent load, bypasses the (non-coherent) XCD L2


def _gload_sys(addr_i64, idx):
    rs = buffer_ops.create_buffer_resource_from_addr(_raw(fx.Int64(addr_i64)))
    return fx.Int32(buffer_ops.buffer_load(rs, idx, vec_width=1, dtype=T.i32,
                                           cache_modifier=_CM_SYS))


def _latomic_add(lds_i32, idx, val):
    return fx.Int32(llvm.AtomicRMWOp(
        llvm.AtomicBinOp.add, _lds_ptr3(lds_i32, idx * fx.Int32(4)), _raw(fx.Int32(val)),
        llvm.AtomicOrdering.monotonic, syncscope="workgroup", alignment=4).result)


@functools.cache
def compile_ep_prep(*, H, TOPK, EL, BM, EPT, QUANT, MULTI, MDYN=False):
    """Sort coverage: grid * 256 * EPT >= M * TOPK (MULTI) or 256 * EPT (single)."""
    assert H % 512 == 0
    assert EL <= 32, "padded-count scan runs over 32 lanes"
    CH = H // 512  # 16 B (8 bf16) loads per lane per row
    # LDS (i32): cnt[EL+1] (slot EL = dummy for non-local entries), base[EL],
    # gcnt[EL], off[EL+1], per wave: quant row count + 64 compacted rows
    L_CNT, L_BASE, L_GCNT, L_OFF = 0, EL + 1, 2 * EL + 1, 3 * EL + 1
    # tile order (see _order_*): per expert light-block prefix; per XCD range: first
    # block, light blocks before it
    L_NL = 4 * EL + 2
    L_BST, L_Q = 5 * EL + 2, 5 * EL + 12  # [9] each
    L_RCNT = 5 * EL + 22
    L_ROWS = L_RCNT + NT // 64
    LDS_I32 = L_ROWS + NT
    kname = (f"ep_prep_h{H}_k{TOPK}_e{EL}_bm{BM}_ept{EPT}_q{int(QUANT)}_m{int(MULTI)}"
             + ("_dyn" if MDYN else ""))

    @fx.struct
    class SharedStorage:
        raw: fx.Array[fx.Uint8, LDS_I32 * 4, 16]

    # ---- x row quant -------------------------------------------------------------
    @flyc.jit
    def _mark_row(lds_i32, lds, wave, my_t, loc):
        if loc > fx.Int32(0):
            slot = _latomic_add(lds_i32, fx.Int32(L_RCNT) + wave, 1)
            lds[fx.Int32(L_ROWS) + wave * fx.Int32(64) + slot] = my_t

    @flyc.jit
    def _quant(lds_i32, lds, arg_x, arg_x8, arg_xs, arg_ids, i32_m, i32_e0, gw, tw, lane,
               wave):
        """Rows gw + i*tw (i < 64) of x (bf16 [M, H]) -> fp8 + f32 row scale."""
        ids = global_typed_ptr(arg_ids, T.i32)
        rs_x = buffer_ops.create_buffer_resource_from_addr(_raw(fx.Int64(arg_x)))
        rs_x8 = buffer_ops.create_buffer_resource_from_addr(_raw(fx.Int64(arg_x8)))
        xs = global_typed_ptr(arg_xs, T.f32)
        # lane i: does row gw + i*tw have a local expert? compact those rows in LDS
        my_t = gw + lane * tw
        my_tc = fx.max(fx.min(my_t, i32_m - fx.Int32(1)), fx.Int32(0))
        loc = fx.Int32(0)
        for k in range_constexpr(TOPK):
            e = fx.Int32(ids[my_tc * fx.Int32(TOPK) + fx.Int32(k)]) - i32_e0
            loc = loc | ((e >= fx.Int32(0)) & (e < fx.Int32(EL))).select(fx.Int32(1), fx.Int32(0))
        loc = (my_t < i32_m).select(loc, fx.Int32(0))
        _mark_row(lds_i32, lds, wave, my_t, loc)
        rocdl.s_waitcnt(lgkmcnt=0)
        n_do = fx.Int32(rocdl.readfirstlane(T.i32, _raw(lds[fx.Int32(L_RCNT) + wave])))
        rows = fx.Int32(L_ROWS) + wave * fx.Int32(64)

        def load_row(t):
            return [Vec(buffer_ops.buffer_load(
                rs_x, t * fx.Int32(H // 2) + (fx.Int32(j * 64) + lane) * fx.Int32(4),
                vec_width=4, dtype=T.i32)) for j in range_constexpr(CH)]

        def do_row(t, raw):
            vals = []
            amax = fx.Float32(0.0)
            for j in range_constexpr(CH):
                v = Vec(raw[j].bitcast(fx.BFloat16)).to(fx.Float32)
                vals.append(v)
                for k in range_constexpr(8):
                    amax = fx.max(amax, fx.max(v[k], -v[k]))
            for bit in range_constexpr(6):
                amax = fx.max(amax, amax.shuffle_xor(1 << bit, 64))
            s = (amax == 0.0).select(fx.Float32(1.0), amax / 240.0)
            inv = 1.0 / s
            _store_scale(xs, t, s, lane)
            for j in range_constexpr(CH):
                v = vals[j]
                p0 = _pack_activation(v[0] * inv, v[1] * inv, v[2] * inv, v[3] * inv)
                p1 = _pack_activation(v[4] * inv, v[5] * inv, v[6] * inv, v[7] * inv)
                buffer_ops.buffer_store(
                    Vec.from_elements([p0, p1], fx.Int32), rs_x8,
                    t * fx.Int32(H // 4) + (fx.Int32(j * 64) + lane) * fx.Int32(2))

        # two rows per iteration: row j+1's loads are in flight while row j is packed
        for i in range((n_do + fx.Int32(1)) // fx.Int32(2)):
            j0 = fx.Int32(i) * fx.Int32(2)
            t0 = fx.Int32(rocdl.readfirstlane(T.i32, _raw(lds[rows + j0])))
            t1 = fx.Int32(rocdl.readfirstlane(T.i32, _raw(
                lds[rows + fx.min(j0 + fx.Int32(1), n_do - fx.Int32(1))])))
            r0 = load_row(t0)
            r1 = load_row(t1)
            do_row(t0, r0)
            do_row(t1, r1)  # odd count: t1 == t0, same values rewritten

    @flyc.jit
    def _store_scale(xs, t, s, lane):
        if lane == fx.Int32(0):
            xs[t] = s

    # ---- sort helpers ------------------------------------------------------------
    assert 64 % TOPK == 0

    @flyc.jit
    def _rc_store(rc, t, n, ok):
        if ok:
            rc[t] = n

    def _count(lds_i32, arg_ids, arg_w, rc, first, n_ent, i32_e0, tx):
        """entries first + j*NT + tx: (local expert or EL, slot in its expert, weight,
        entry, the token has exactly one local route). A token's TOPK entries sit in
        TOPK neighbouring lanes (first, NT multiples of TOPK): one ballot counts its
        local routes; the group's first lane writes the count to rc[token]."""
        ids = global_typed_ptr(arg_ids, T.i32)
        wts = global_typed_ptr(arg_w, T.f32)
        lane = tx % fx.Int32(64)
        g0 = fx.Int64(lane - lane % fx.Int32(TOPK))
        ent = []
        for j in range_constexpr(EPT):
            idx = first + fx.Int32(j * NT) + tx
            ok = idx < n_ent
            idc = ok.select(idx, fx.Int32(0))
            e = fx.Int32(ids[idc]) - i32_e0
            loc = ok & (e >= fx.Int32(0)) & (e < fx.Int32(EL))
            es = loc.select(e, fx.Int32(EL))
            bal = fx.Int64(rocdl.ballot(T.i64, _raw(loc)))
            n_loc = fx.Int32(fx.math.ctpop(arith.trunci(
                T.i32, _raw((bal >> g0) & fx.Int64((1 << TOPK) - 1)))))
            _rc_store(rc, idc // fx.Int32(TOPK), n_loc, ok & (lane % fx.Int32(TOPK) == fx.Int32(0)))
            ent.append((es, _latomic_add(lds_i32, fx.Int32(L_CNT) + es, 1),
                        fx.Float32(wts[idc]), idc, n_loc == fx.Int32(1)))
        return ent

    @flyc.jit
    def _prefix(lds, tx):
        """wave 0, lane e < EL: padded count of expert e -> exclusive scan (ds_bpermute
        Hillis-Steele over 32 lanes) -> off[e]; off[EL] = total."""
        if tx < fx.Int32(64):
            c = (tx < fx.Int32(EL)).select(lds[fx.Int32(L_GCNT) + fx.min(tx, fx.Int32(EL - 1))],
                                           fx.Int32(0))
            pad = (c + fx.Int32(BM - 1)) // fx.Int32(BM) * fx.Int32(BM)
            inc = pad
            for bit in range_constexpr(5):
                d = 1 << bit
                src = fx.max(tx - fx.Int32(d), fx.Int32(0))
                other = fx.Int32(rocdl.ds_bpermute(T.i32, _raw(src * fx.Int32(4)), _raw(inc)))
                inc = inc + (tx >= fx.Int32(d)).select(other, fx.Int32(0))
            if tx < fx.Int32(EL):
                lds[fx.Int32(L_OFF) + tx] = inc - pad
            if tx == fx.Int32(EL - 1):
                lds[fx.Int32(L_OFF + EL)] = inc

    # ---- GEMM tile order ---------------------------------------------------------
    # The GEMMs map WG bx to XCD x = bx % 8 (the hardware deals WGs round robin to
    # the XCDs, and inside one to its CUs in WG order) and give XCD x the m-blocks
    # order[bst[x] : bst[x+1]] (x NB n-tiles each, m-major) in WG order. Blocks are
    # cut, in expert order (neighbouring blocks share W in the XCD's L2), into 8
    # contiguous ranges of equal estimated cost (a block costs max(CB, its row
    # variant 32 / 64 / 128): a small block streams its W tiles in the time MFMAs
    # take for ~CB rows (measured: 4096 tok g1 fp8 R32 43 us, R128 90 us; R64 ~ R32
    # at 1024); a block goes to the range holding its cost midpoint), so a hot
    # expert spreads over several XCDs. Inside a range the blocks cheaper than a
    # full one (BM > CB only) go last (expert order otherwise): the CUs' dynamic
    # dispatch runs long tiles first.
    CB = 64
    NX = 8
    KF = max(CB, BM)  # cost of a full block

    def _pcost(pr):
        """max(CB, row variant): the GEMMs run 32 / 64 / 128-row variants."""
        v = fx.Int32(32)
        for r in (64, 128):
            if r <= BM:
                v = (pr > fx.Int32(r // 2)).select(fx.Int32(r), v)
        return (pr > fx.Int32(0)).select(fx.max(fx.Int32(CB), v), fx.Int32(0))

    def _light(pr):
        """partial block cheaper than a full one (only when BM > CB)."""
        return ((pr > fx.Int32(0)) & (_pcost(pr) < fx.Int32(KF))).select(fx.Int32(1), fx.Int32(0))

    def _scan_excl(v, tx):
        inc = v
        for bit in range_constexpr(5):
            d = 1 << bit
            src = fx.max(tx - fx.Int32(d), fx.Int32(0))
            other = fx.Int32(rocdl.ds_bpermute(T.i32, _raw(src * fx.Int32(4)), _raw(inc)))
            inc = inc + (tx >= fx.Int32(d)).select(other, fx.Int32(0))
        return inc - v, inc

    def _wave_sum(v):
        for bit in range_constexpr(6):
            v = v + v.shuffle_xor(1 << bit, 64)
        return v

    @flyc.jit
    def _order(lds, tx):
        """wave 0, lane e < EL: cost / light-block prefixes over the experts; per
        cut x: blocks whose cost midpoint lies below x C / 8 (-> bst[x]) and light
        blocks before bst[x] (-> Q[x]), as wave sums of per-expert counts."""
        if tx < fx.Int32(64):
            zero = fx.Int32(0)
            act = tx < fx.Int32(EL)
            c = act.select(lds[fx.Int32(L_GCNT) + fx.min(tx, fx.Int32(EL - 1))], zero)
            nf = c // fx.Int32(BM)
            pr = c % fx.Int32(BM)
            lt = _light(pr)
            p, ctot = _scan_excl(nf * fx.Int32(KF) + _pcost(pr), tx)
            nl, nltot = _scan_excl(lt, tx)
            ctot = fx.Int32(rocdl.readlane(T.i32, _raw(ctot), EL - 1))
            nltot = fx.Int32(rocdl.readlane(T.i32, _raw(nltot), EL - 1))
            pb = lds[fx.Int32(L_OFF) + fx.min(tx, fx.Int32(EL - 1))] // fx.Int32(BM) + nf
            if act:
                lds[fx.Int32(L_NL) + tx] = nl
            nb_all = lds[fx.Int32(L_OFF + EL)] // fx.Int32(BM)
            for x in range_constexpr(1, NX):
                xc = fx.Int32(x) * ctot
                # full block j: 8 (p + (j + 1/2) KF) < x C <=> odd o = 2j+1 < (xC - 8p) / 4KF
                d = xc - fx.Int32(8) * p
                n_odd = (d - fx.Int32(1)) // fx.Int32(4 * KF)
                nfu = (d > zero).select(fx.min((n_odd + fx.Int32(1)) // fx.Int32(2), nf), zero)
                ok_p = (pr > zero) & (fx.Int32(4) * (fx.Int32(2) * p + fx.Int32(2 * KF) * nf
                                                     + _pcost(pr)) < xc)
                bst = _wave_sum(act.select(nfu + ok_p.select(fx.Int32(1), zero), zero))
                q = _wave_sum(((lt > zero) & (pb < bst) & act).select(fx.Int32(1), zero))
                if tx == fx.Int32(x):
                    lds[fx.Int32(L_BST + x)] = bst
                    lds[fx.Int32(L_Q + x)] = q
            if tx == fx.Int32(0):
                lds[fx.Int32(L_BST)] = zero
                lds[fx.Int32(L_Q)] = zero
                lds[fx.Int32(L_BST + NX)] = nb_all
                lds[fx.Int32(L_Q + NX)] = nltot

    def _slot(lds, e, b, light):
        """order slot of block b (global) of expert e: heavy blocks of the range
        first (minus the light ones between the range start and b), then light."""
        x = fx.Int32(0)
        for k in range_constexpr(1, NX):
            x = x + (lds[fx.Int32(L_BST + k)] <= b).select(fx.Int32(1), fx.Int32(0))
        lo = lds[fx.Int32(L_BST) + x]
        qx = lds[fx.Int32(L_Q) + x]
        nl_before = lds[fx.Int32(L_NL) + e] - qx  # light blocks in [lo, b)
        nheavy = (lds[fx.Int32(L_BST + 1) + x] - lo) - (lds[fx.Int32(L_Q + 1) + x] - qx)
        return light.select(lo + nheavy + nl_before, b - nl_before)

    @flyc.jit
    def _scatter(lds, sids, sw, e, pos, w, idx, single):
        """sorted id = token | slot << 24 | (token has exactly one local route) << 31
        (the GEMMs mask the token with 0xFFFFFF; gemm2 may store such rows instead of
        adding them atomically)."""
        if e < fx.Int32(EL):
            dst = lds[fx.Int32(L_OFF) + e] + lds[fx.Int32(L_BASE) + e] + pos
            sids[dst] = ((idx // fx.Int32(TOPK)) | ((idx % fx.Int32(TOPK)) << fx.Int32(24))
                         | single.select(fx.Int32(SID_SINGLE), fx.Int32(0)))
            sw[dst] = w

    TPE = NT // EL  # threads per expert for padding rows / block ids

    @flyc.jit
    def _bst_store(lds, sched, tx):
        if tx <= fx.Int32(NX):
            sched[tx] = lds[fx.Int32(L_BST) + tx]

    @flyc.jit
    def _pad_rows(lds, sids, sw, seids, sched, tx, i32_m):
        """thread tx: expert tx // TPE, every TPE-th padding row (< BM per expert),
        block id and order slot of that expert's blocks (see _order_*); threads
        <= 8 also write the XCD range starts."""
        e = tx // fx.Int32(TPE)
        j = tx % fx.Int32(TPE)
        o0 = lds[fx.Int32(L_OFF) + e]
        o1 = lds[fx.Int32(L_OFF + 1) + e]
        c = lds[fx.Int32(L_GCNT) + e]
        for k in range_constexpr(-(-BM // TPE)):
            r = o0 + c + j + fx.Int32(k * TPE)
            if r < o1:
                sids[r] = i32_m | fx.Int32(TOPK << 24)
                sw[r] = fx.Float32(0.0)
        b0 = o0 // fx.Int32(BM)
        nb = (o1 - o0) // fx.Int32(BM)
        nfull = c // fx.Int32(BM)
        lt = _light(c % fx.Int32(BM)) > fx.Int32(0)
        for k in range((nb + fx.Int32(TPE - 1)) // fx.Int32(TPE)):
            b = fx.Int32(k * TPE) + j
            if b < nb:
                seids[b0 + b] = e
                sched[fx.Int32(SCHED_BORDER) + _slot(lds, e, b0 + b, lt & (b == nfull))] = b0 + b
        _bst_store(lds, sched, tx)

    def _rc_ptr(arg_seids, i32_m):
        """per-token local route count, after the block expert ids and the schedule."""
        nblk = (i32_m * fx.Int32(TOPK)) // fx.Int32(BM) + fx.Int32(EL)
        return global_typed_ptr(fx.Int64(arg_seids) + fx.Int64(fx.Int32(2) * nblk
                                                               + fx.Int32(SCHED_BORDER)) * fx.Int64(4),
                                T.i32)

    def _finish_sort(lds_i32, lds, arg_sids, arg_sw, arg_seids, arg_nv, ent, i32_m, i32_mv,
                     me, tx):
        sids = global_typed_ptr(arg_sids, T.i32)
        sw = global_typed_ptr(arg_sw, T.f32)
        for es, pos, w, idc, single in ent:
            _scatter(lds, sids, sw, es, pos, w, idc, single)
        _pad_and_nv(lds_i32, lds, sids, sw, arg_seids, arg_nv, i32_m, i32_mv, me, tx)

    @flyc.jit
    def _pad_and_nv(lds_i32, lds, sids, sw, arg_seids, arg_nv, i32_m, i32_mv, me, tx):
        if me == fx.Int32(0):
            # the GEMM schedule lives in the same buffer, after the cap // BM block ids
            nblk = (i32_m * fx.Int32(TOPK)) // fx.Int32(BM) + fx.Int32(EL)
            _order(lds, tx)
            gpu.barrier()
            _pad_rows(lds, sids, sw, global_typed_ptr(arg_seids, T.i32),
                      global_typed_ptr(fx.Int64(arg_seids) + fx.Int64(nblk) * fx.Int64(4), T.i32),
                      tx, i32_m)
            if tx == fx.Int32(0):
                nv = global_typed_ptr(arg_nv, T.i32)
                nv[fx.Int32(0)] = lds[fx.Int32(L_OFF + EL)]
                nv[fx.Int32(1)] = i32_mv

    @flyc.jit
    def _single_sort(lds_i32, lds, arg_ids, arg_w, arg_sids, arg_sw, arg_seids, arg_nv,
                     i32_m, i32_mv, i32_e0, tx):
        """whole sort in one WG, LDS only (m_valid * TOPK <= 256 * EPT)."""
        ent = _count(lds_i32, arg_ids, arg_w, _rc_ptr(arg_seids, i32_m), fx.Int32(0),
                     i32_mv * fx.Int32(TOPK), i32_e0, tx)
        gpu.barrier()
        _copy_counts(lds, tx)
        gpu.barrier()
        _prefix(lds, tx)
        gpu.barrier()
        _finish_sort(lds_i32, lds, arg_sids, arg_sw, arg_seids, arg_nv, ent, i32_m, i32_mv,
                     fx.Int32(0), tx)

    @flyc.jit
    def _copy_counts(lds, tx):
        if tx < fx.Int32(EL):
            lds[fx.Int32(L_GCNT) + tx] = lds[fx.Int32(L_CNT) + tx]
            lds[fx.Int32(L_BASE) + tx] = fx.Int32(0)

    @flyc.jit
    def _reserve(lds, arg_ws, tx):
        if tx < fx.Int32(EL):
            lds[fx.Int32(L_BASE) + tx] = _gatomic_add(arg_ws, tx, lds[fx.Int32(L_CNT) + tx])

    @flyc.jit
    def _read_counts(lds, arg_ws, tx):
        if tx < fx.Int32(EL):
            lds[fx.Int32(L_GCNT) + tx] = _gload_sys(arg_ws, tx)

    @flyc.jit
    def _grid_sync(arg_ws, n_sort, tx):
        """barrier of the n_sort sort WGs (resident: lowest block ids, few)."""
        if tx == fx.Int32(0):
            _gatomic_add(arg_ws, fx.Int32(EL), 1)
            seen = _gload_sys(arg_ws, fx.Int32(EL))
            while seen < n_sort:
                rocdl.s_sleep(1)
                seen = _gload_sys(arg_ws, fx.Int32(EL))

    @flyc.jit
    def _finish(arg_ws, grid, tx):
        """the last WG past phase 2 resets counts, barrier and done counters."""
        if tx == fx.Int32(0):
            old = _gatomic_add(arg_ws, fx.Int32(EL + 1), 1)
            if old == grid - fx.Int32(1):
                for e in range_constexpr(EL + 2):
                    _gatomic_xchg(arg_ws, fx.Int32(e), 0)

    @flyc.jit
    def _entry(arg_ids, arg_w, arg_x, arg_x8, arg_xs, arg_sids, arg_sw, arg_seids, arg_nv,
               arg_ws, i32_m, i32_e0, arg_mv):
        lds_ptr = fx.SharedAllocator().allocate(SharedStorage).peek().raw.ptr
        lds_i32 = fx.Int32(fx.ptrtoint(lds_ptr))
        lds = lds_typed_ptr(lds_i32, T.i32)
        tx = fx.Int32(gpu.thread_id("x"))
        bx = fx.Int32(gpu.block_id("x"))
        grid = fx.Int32(gpu.grid_dim.x)
        lane = tx % fx.Int32(64)
        wave = tx // fx.Int32(64)
        if tx < fx.Int32(EL + 1):
            lds[fx.Int32(L_CNT) + tx] = fx.Int32(0)
        if tx < fx.Int32(NT // 64):
            lds[fx.Int32(L_RCNT) + tx] = fx.Int32(0)
        # valid rows: the capacity, or a device count clamped to [0, capacity]
        if const_expr(MDYN):
            i32_mv = fx.max(fx.min(fx.Int32(rocdl.readfirstlane(T.i32, _raw(
                fx.Int32(global_typed_ptr(arg_mv, T.i32)[fx.Int32(0)])))), i32_m), fx.Int32(0))
        else:
            i32_mv = i32_m
        gpu.barrier()
        # WGs [0, n_sort): sort; [n_sort, grid): x quant (no sync with the sort)
        n_sort = fx.max((i32_mv * fx.Int32(TOPK) + fx.Int32(EPT * NT - 1)) // fx.Int32(EPT * NT),
                        fx.Int32(1))
        if const_expr(not MULTI):
            n_sort = fx.Int32(1)
        if bx < n_sort:
            _sort_role(lds_i32, lds, arg_ids, arg_w, arg_sids, arg_sw, arg_seids, arg_nv,
                       arg_ws, i32_m, i32_mv, i32_e0, n_sort, bx, tx)
        if const_expr(QUANT):
            if bx >= n_sort:
                _quant(lds_i32, lds, arg_x, arg_x8, arg_xs, arg_ids, i32_mv, i32_e0,
                       (bx - n_sort) * fx.Int32(NT // 64) + wave,
                       (grid - n_sort) * fx.Int32(NT // 64), lane, wave)

    def _multi_sort(lds_i32, lds, arg_ids, arg_w, arg_sids, arg_sw, arg_seids, arg_nv,
                    arg_ws, i32_m, i32_mv, i32_e0, n_sort, bx, tx):
        ent = _count(lds_i32, arg_ids, arg_w, _rc_ptr(arg_seids, i32_m),
                     bx * fx.Int32(EPT * NT), i32_mv * fx.Int32(TOPK), i32_e0, tx)
        gpu.barrier()
        _reserve(lds, arg_ws, tx)
        rocdl.s_waitcnt(vmcnt=0)
        gpu.barrier()
        _grid_sync(arg_ws, n_sort, tx)
        gpu.barrier()
        _read_counts(lds, arg_ws, tx)
        gpu.barrier()
        _prefix(lds, tx)
        gpu.barrier()
        _finish(arg_ws, n_sort, tx)  # counts already read by this WG
        _finish_sort(lds_i32, lds, arg_sids, arg_sw, arg_seids, arg_nv, ent, i32_m, i32_mv,
                     bx, tx)

    @flyc.jit
    def _sort_dyn(lds_i32, lds, arg_ids, arg_w, arg_sids, arg_sw, arg_seids, arg_nv,
                  arg_ws, i32_m, i32_mv, i32_e0, n_sort, bx, tx):
        """MULTI + MDYN: the valid entries may fit one WG -> LDS-only sort."""
        if n_sort == fx.Int32(1):
            _single_sort(lds_i32, lds, arg_ids, arg_w, arg_sids, arg_sw, arg_seids,
                         arg_nv, i32_m, i32_mv, i32_e0, tx)
        else:
            _multi_sort(lds_i32, lds, arg_ids, arg_w, arg_sids, arg_sw, arg_seids, arg_nv,
                        arg_ws, i32_m, i32_mv, i32_e0, n_sort, bx, tx)

    @flyc.jit
    def _sort_role(lds_i32, lds, arg_ids, arg_w, arg_sids, arg_sw, arg_seids, arg_nv,
                   arg_ws, i32_m, i32_mv, i32_e0, n_sort, bx, tx):
        if const_expr(not MULTI):
            _single_sort(lds_i32, lds, arg_ids, arg_w, arg_sids, arg_sw, arg_seids,
                         arg_nv, i32_m, i32_mv, i32_e0, tx)
        else:
            if const_expr(MDYN):
                _sort_dyn(lds_i32, lds, arg_ids, arg_w, arg_sids, arg_sw, arg_seids,
                          arg_nv, arg_ws, i32_m, i32_mv, i32_e0, n_sort, bx, tx)
            else:
                _multi_sort(lds_i32, lds, arg_ids, arg_w, arg_sids, arg_sw, arg_seids,
                            arg_nv, arg_ws, i32_m, i32_mv, i32_e0, n_sort, bx, tx)

    @flyc.kernel(name=kname, known_block_size=[NT, 1, 1])
    def kernel(arg_ids: fx.Int64, arg_w: fx.Int64, arg_x: fx.Int64, arg_x8: fx.Int64,
               arg_xs: fx.Int64, arg_sids: fx.Int64, arg_sw: fx.Int64, arg_seids: fx.Int64,
               arg_nv: fx.Int64, arg_ws: fx.Int64, i32_m: fx.Int32, i32_e0: fx.Int32,
               arg_mv: fx.Int64):
        _entry(arg_ids, arg_w, arg_x, arg_x8, arg_xs, arg_sids, arg_sw, arg_seids, arg_nv,
               arg_ws, i32_m, i32_e0, arg_mv)

    @flyc.jit
    def launch(arg_ids: fx.Int64, arg_w: fx.Int64, arg_x: fx.Int64, arg_x8: fx.Int64,
               arg_xs: fx.Int64, arg_sids: fx.Int64, arg_sw: fx.Int64, arg_seids: fx.Int64,
               arg_nv: fx.Int64, arg_ws: fx.Int64, i32_m: fx.Int32, i32_e0: fx.Int32,
               arg_mv: fx.Int64, i32_grid: fx.Int32, stream: fx.Stream):
        kernel(arg_ids, arg_w, arg_x, arg_x8, arg_xs, arg_sids, arg_sw, arg_seids, arg_nv,
               arg_ws, i32_m, i32_e0, arg_mv).launch(grid=(fx.Int64(i32_grid), 1, 1),
                                                     block=(NT, 1, 1), stream=stream)

    return launch


_WS = {}


def _workspace(device, n):
    key = (device, n)
    if key not in _WS:
        _WS[key] = torch.zeros(n, dtype=torch.int32, device=device)
    return _WS[key]


def ep_prep(topk_ids, topk_weights, x, *, e0, block_m, quant, num_local=32, m_valid=None):
    """-> (sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, x8, xs,
    sched, route_counts); ``x8``/``xs`` are None unless ``quant`` (x bf16 [M, H] ->
    fp8 + f32 row scale). ``sched``: the GEMMs' tile schedule (see SCHED_BORDER):
    XCD bucket starts and the m-block order (``sched[SCHED_BORDER:]``, first
    num_valid_ids[0] // block_m valid). ``route_counts``: int32 [M], local routes per
    token. Sorted ids carry SID_SINGLE for tokens with exactly one local route.
    ``m_valid``: optional int32 device tensor; only rows [0, m_valid[0]) of the
    (capacity-sized) inputs are sorted / quantized, num_valid_ids[1] = m_valid."""
    M, K = topk_ids.shape
    H = x.shape[1]
    dev = topk_ids.device
    cap = M * K + num_local * block_m
    sids = torch.empty(cap, dtype=torch.int32, device=dev)
    sw = torch.empty(cap, dtype=torch.float32, device=dev)
    nblk = cap // block_m
    assert nblk == (M * K) // block_m + num_local and M < 65536
    seids_sched = torch.empty(2 * nblk + SCHED_BORDER + M, dtype=torch.int32, device=dev)
    seids, sched = seids_sched[:nblk], seids_sched[nblk:2 * nblk + SCHED_BORDER]
    rc = seids_sched[2 * nblk + SCHED_BORDER:]
    nv = torch.empty(2, dtype=torch.int32, device=dev)
    x8 = xs = None
    if quant:
        x8 = torch.empty((M, H), dtype=torch.float8_e4m3fnuz, device=dev)
        xs = torch.empty(M, dtype=torch.float32, device=dev)
    # sort WGs: one (LDS only) up to 256*8 entries, else one per 256*8 entries with a
    # global atomic per (WG, expert) and a barrier among them; quant WGs after them:
    # one wave per row, <= 64 rows per wave
    ept = 8
    multi = M * K > NT * ept
    n_sort = -(-(M * K) // (NT * ept)) if multi else 1
    assert n_sort <= NUM_CU, f"M={M} too large"
    n_quant = min(4 * NUM_CU, -(-M // (NT // 64))) if quant else 0
    assert not quant or M <= n_quant * (NT // 64) * 64
    grid = n_sort + n_quant
    launch = compile_ep_prep(H=H, TOPK=K, EL=num_local, BM=block_m, EPT=ept, QUANT=quant,
                             MULTI=multi, MDYN=m_valid is not None)
    ws = _workspace(dev, num_local + 2)
    _run_compiled(
        launch, topk_ids.data_ptr(), topk_weights.data_ptr(), x.data_ptr(),
        x8.data_ptr() if quant else 0, xs.data_ptr() if quant else 0, sids.data_ptr(),
        sw.data_ptr(), seids.data_ptr(), nv.data_ptr(), ws.data_ptr(), int(M), int(e0),
        m_valid.data_ptr() if m_valid is not None else 0,
        int(grid), torch.cuda.current_stream())
    return sids, sw, seids, nv, x8, xs, sched, rc
