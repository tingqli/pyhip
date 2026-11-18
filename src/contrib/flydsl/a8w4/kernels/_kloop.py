# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""The K loop of the EP GEMM tiles (ep3k ``ep_gemm``, ep1k ``ep_small``, megamoe ``ep_mega``)
and the dispatch to its row variants.

fd2 mxfp4 W is streamed from HBM straight into VGPRs and decoded to fp8 there; A (fp8, or
bf16 quantized while staged) goes through LDS.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from ._cvt import _fp4x8_to_fp8x8_fast, _fp8_lut_predelta, _fp8x4_pack, _perm_u32
from ._mem import (
    _global_i32_buffer_tiles,
    _global_i32_buffer_view,
    _lds_f32_store_masked,
    lds_typed_ptr,
)


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


def _frag(v_i32, elem, n):
    """MFMA operand rmem fragment of ``n`` ``elem`` from a vector of i32."""
    t = fx.make_rmem_tensor(fx.make_layout(n, 1), elem)
    t.store(Vec(v_i32).bitcast(elem))
    return t


@flyc.jit
def _masked_lds_store(atom, reg, dst, ok):
    if ok:
        fx.copy(atom, reg, dst)


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
    ``pre_a()``: after the W prefetch is issued, before the first A load; ``post_a()``:
    after the first A tile's loads are issued, before they go to LDS (e.g. finish a
    table from loads ``pre_a`` issued without waiting for A in between).
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
    assert PF >= 1 and KT % (2 * PF) == 0 and KT // (2 * PF) >= 2, (
        f"K loop: K={K} gives {KT} K tiles of {TK}, prefetch {PF} needs a multiple of "
        f"{2 * PF} and at least {4 * PF}: K is too short for this config")
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

    def run(R, row_base16, hook=None, pre_a=None, post_a=None):
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
        if const_expr(post_a is not None):
            post_a()
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
