# Package copy of doc-mimo/kernels/ep8_small.py (test harness removed, env knobs frozen).
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""EP8 MoE, **small-batch fused path**: routing + (x quant) + gemm1 + gemm2 in ONE
persistent launch (gfx942, FlyDSL), for M up to ``MAXM`` tokens.

Why: at decode sizes the 3-launch pipeline (prep, gemm1, gemm2) pays three kernel
ramps (first W bytes arrive several us after launch) plus the prep kernel itself.

* Routing is computed by EVERY WG (no global sort): each wave loads the topk ids
  of 64 tokens, builds a local-expert bitmap per token and one 64-bit ballot mask
  per (expert, 64 tokens). Expert e's rows are its tokens in increasing token
  order (deterministic, so all WGs agree): row k of e = the k-th set bit of e's
  masks; m-blocks of BM rows, padded per expert (same layout as moe_sorting).
* Setup (each WG a share, write-through stores + a two-level counter): zero
  ``out`` and (a8w4) quantize x rows to fp8 (bit-identical to quantize_rows).
  Only a8w4 gemm1 items (x8) and gemm2 items (zeroed out) wait for it.
* Items, static round robin over the grid, all gemm1 items first:
  gemm1 item = (m-block, 16*C1*NW inter cols) -> act(gate)*up (SiLU / SiTUv2) as
  **bf16** rows of ``inter`` (deterministic position = sorted row);
  gemm2 item = (m-block, 16*C2*NW out cols): waits for the m-block's gemm1 counter,
  stages the bf16 rows (a8w4: quantized per (row, 128) on the fly, the 3-launch
  path's intermediate quant: ``_make_core(AQ8=True)``) and adds the routed result
  with bf16x2 atomics.
Every item issues its W prefetch before any wait. The last WG resets the counters.
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
from .._vendor.act import gate_up_act, situ_params
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
from . import ep_gemm as EP  # noqa: E402
from . import tp_gemm1 as G1  # noqa: E402
from .ep_prep import _gatomic_add, _gatomic_xchg, _gload_sys, _latomic_add  # noqa: E402


HERE = os.path.dirname(os.path.abspath(__file__))


NUM_CU = 304
CM_COH = 17  # sc0|sc1: device-coherent loads of data written by other WGs (other XCDs)
# counter workspace (int32): phase-A done (top), exit (top), NSUB phase-A / exit
# sub-counters (a single address hit by every WG serializes ~40 ns per atomic),
# then one g1-done counter per m-block. 64 B apart to keep atomics on separate lines.
NSUB = 32
STRIDE = 16
C_PHA, C_EXIT = 0, STRIDE
C_SUBA = 2 * STRIDE
C_SUBX = C_SUBA + NSUB * STRIDE
C_G1 = C_SUBX + NSUB * STRIDE


def _fence_release():
    """Release of data stored write-through (cache modifier CM_COH): wait for the
    store acks. (An agent-scope fence would emit buffer_wbl2 = write back the
    whole XCD L2: 7-14 us per call.)"""
    rocdl.s_waitcnt(vmcnt=0)


def _rsrc(addr):
    return buffer_ops.create_buffer_resource_from_addr(_raw(fx.Int64(addr)))


@flyc.jit
def _zero_slice_wt(arg_zero, i32_n16, bx, grid, tx, NT):
    """WG bx of ``grid`` zeroes its share of ``i32_n16`` 16 B units (write-through)."""
    if i32_n16 > fx.Int32(0):
        rs = buffer_ops.create_buffer_resource_from_addr(
            _raw(fx.Int64(arg_zero)), num_records_bytes=_raw(fx.Int64(i32_n16) * fx.Int64(16)))
        per = (i32_n16 + grid - fx.Int32(1)) // grid
        z = Vec.filled(4, 0, fx.Int32)
        for i in range((per + fx.Int32(NT - 1)) // fx.Int32(NT)):
            u = fx.Int32(i) * fx.Int32(NT) + tx
            off = (u < per).select(bx * per + u, i32_n16)
            buffer_ops.buffer_store(z, rs, off * fx.Int32(4), cache_modifier=CM_COH)


@flyc.jit
def _wait_ge(arg_ctr, idx, target, tx):
    """WG waits until ctr[idx] >= target (thread 0 polls coherent loads)."""
    if tx == fx.Int32(0):
        seen = _gload_sys(arg_ctr, idx)
        while seen < target:
            rocdl.s_sleep(8)
            seen = _gload_sys(arg_ctr, idx)
    gpu.barrier()


@flyc.jit
def _signal(arg_ctr, idx, tx):
    """all threads released their stores (fence before the call); one increment."""
    gpu.barrier()
    if tx == fx.Int32(0):
        _gatomic_add(arg_ctr, idx, 1)


@flyc.jit
def _signal2(arg_ctr, top, sub0, bx, grid, tx):
    """two-level count of the grid's WGs: WG bx bumps sub-counter bx % NSUB; the
    last of a sub-group bumps ``top`` (which reaches NSUB when all WGs are in).
    Returns (on tx 0) whether this WG was the very last."""
    gpu.barrier()
    if tx == fx.Int32(0):
        g = bx % fx.Int32(NSUB)
        n_in_g = (grid - g + fx.Int32(NSUB - 1)) // fx.Int32(NSUB)
        old = _gatomic_add(arg_ctr, fx.Int32(sub0) + g * fx.Int32(STRIDE), 1)
        if old == n_in_g - fx.Int32(1):
            _gatomic_add(arg_ctr, top, 1)


@functools.cache
def compile_ep_small(*, H, I, EL, TOPK, ADT, BM, NW, C1, C2, PF, ROWV, MAXM, SCH=3, DBG=0,
                     ACT="silu"):
    FP8 = ADT == "fp8"
    assert ADT in ("fp8", "bf16")
    assert ACT in ("silu", "situv2"), ACT
    assert EL <= 32 and TOPK == 8
    NT = 64 * NW
    NI1 = 16 * C1 * NW  # gemm1 inter cols per item (wave: C1 gate + C1 up 16-col blocks)
    NB1 = I // NI1
    NO2 = 16 * C2 * NW  # gemm2 output cols per item
    NB2 = H // NO2
    ARB1 = 128 if FP8 else 256
    ARB2 = 128 if FP8 else 256
    CAP = MAXM * TOPK + EL * BM  # sorted rows incl. padding (worst case)
    MW = -(-MAXM // 64)  # 64-token words
    assert ROWV[-1] == BM and all(r % 16 == 0 for r in ROWV)
    assert MAXM <= NUM_CU * NW  # setup x quant: <= 1 row per wave at >= 1 WG/CU
    LDS_A = max(2 * BM * ARB1, 2 * BM * ARB2 + (3 * BM * 4 if FP8 else 0))
    # LDS after the A region (i32 index): masks [EL][MW][2], cum [EL][MW+1] (set bits
    # before word w), off [EL+1] (padded row offsets), tok8 [MAXM][2] (local expert
    # id per topk slot, 0xFF = not local)
    L_MASK = LDS_A // 4
    L_CUM = L_MASK + EL * MW * 2
    L_OFF = L_CUM + EL * (MW + 1)
    L_TOK8 = L_OFF + EL + 1
    LDS_BYTES = (L_TOK8 + MAXM * 2) * 4
    assert LDS_BYTES <= 32768, LDS_BYTES  # 2 WGs per CU
    kname = (f"ep_small_{ADT}_h{H}_i{I}_e{EL}_bm{BM}_nw{NW}_c{C1}x{C2}_pf{PF}"
             f"_r{'x'.join(map(str, ROWV))}_m{MAXM}_s{SCH}_d{DBG}"
             + ("" if ACT == "silu" else f"_{ACT}"))

    @fx.struct
    class SharedStorage:
        raw: fx.Array[fx.Uint8, LDS_BYTES, 16]

    # ---- routing (every WG) -------------------------------------------------------
    @flyc.jit
    def _route_word(lds, a, w, lane):
        """one wave, tokens 64w + lane: local-expert bytes -> tok8, ballots -> masks."""
        t = fx.Int32(w * 64) + lane
        ok = t < a["m"]
        tc = ok.select(t, fx.Int32(0))
        ids = global_typed_ptr(a["ids"], T.i32)
        bits = fx.Int32(0)
        b8 = [fx.Int32(0), fx.Int32(0)]
        for k in range_constexpr(TOPK):
            le = fx.Int32(ids[tc * fx.Int32(TOPK) + fx.Int32(k)]) - a["e0"]
            loc = ok & (le >= fx.Int32(0)) & (le < fx.Int32(EL))
            bits = bits | loc.select(fx.Int32(1) << le, fx.Int32(0))
            b8[k // 4] = b8[k // 4] | (loc.select(le, fx.Int32(0xFF)) << fx.Int32(8 * (k % 4)))
        _tok8_store(lds, tc, b8, ok)  # lanes past M must not clobber token 0
        for e in range_constexpr(EL):
            m = fx.Int64(rocdl.ballot(T.i64, _raw(((bits >> fx.Int32(e)) & fx.Int32(1))
                                                   != fx.Int32(0))))
            lds[fx.Int32(L_MASK + (e * MW + w) * 2)] = fx.Int32(arith.trunci(T.i32, _raw(m)))
            lds[fx.Int32(L_MASK + (e * MW + w) * 2 + 1)] = fx.Int32(
                arith.trunci(T.i32, _raw(m >> fx.Int64(32))))

    @flyc.jit
    def _tok8_store(lds, t, b8, ok):
        if ok:
            lds[fx.Int32(L_TOK8) + t * fx.Int32(2)] = b8[0]
            lds[fx.Int32(L_TOK8 + 1) + t * fx.Int32(2)] = b8[1]

    @flyc.jit
    def _route_counts(lds, tx):
        """wave 0, lane e < EL: cum[e][*] and the padded offsets off[e]."""
        if tx < fx.Int32(64):
            e = fx.min(tx, fx.Int32(EL - 1))
            c = fx.Int32(0)
            for w in range_constexpr(MW):
                if tx < fx.Int32(EL):
                    lds[fx.Int32(L_CUM) + e * fx.Int32(MW + 1) + fx.Int32(w)] = c
                for h in range_constexpr(2):
                    c = c + fx.Int32(fx.math.ctpop(_raw(lds[fx.Int32(L_MASK + w * 2 + h)
                                                         + e * fx.Int32(MW * 2)])))
            if tx < fx.Int32(EL):
                lds[fx.Int32(L_CUM + MW) + e * fx.Int32(MW + 1)] = c
            c = (tx < fx.Int32(EL)).select(c, fx.Int32(0))
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

    def _route(lds, a, wave, lane, tx):
        for w in range_constexpr(MW):  # word w on wave w % NW (disjoint LDS outputs)
            _route_on_wave(lds, a, w, wave, lane)
        gpu.barrier()
        _route_counts(lds, tx)
        gpu.barrier()

    @flyc.jit
    def _route_on_wave(lds, a, w, wave, lane):
        if wave == fx.Int32(w % NW):
            _route_word(lds, a, w, lane)

    def _row_token(lds, e, k, cnt_e, i32_m):
        """token of row k of expert e (k-th set bit of e's masks); M if padding."""
        wsel = fx.Int32(0)
        for j in range_constexpr(1, MW):
            wsel = wsel + (lds[fx.Int32(L_CUM + j) + e * fx.Int32(MW + 1)] <= k).select(
                fx.Int32(1), fx.Int32(0))
        r = k - lds[fx.Int32(L_CUM) + e * fx.Int32(MW + 1) + wsel]
        mi = fx.Int32(L_MASK) + (e * fx.Int32(MW) + wsel) * fx.Int32(2)
        lo = lds[mi]
        hi = lds[mi + fx.Int32(1)]
        clo = fx.Int32(fx.math.ctpop(_raw(lo)))
        in_hi = r >= clo
        word = in_hi.select(hi, lo)
        r2 = in_hi.select(r - clo, r)
        pos = fx.Int32(0)
        for s in (16, 8, 4, 2, 1):
            c = fx.Int32(fx.math.ctpop(_raw(word & ((fx.Int32(1) << (pos + fx.Int32(s)))
                                                    - fx.Int32(1)))))
            pos = pos + (c <= r2).select(fx.Int32(s), fx.Int32(0))
        t = wsel * fx.Int32(64) + in_hi.select(fx.Int32(32), fx.Int32(0)) + pos
        return (k < cnt_e).select(t, i32_m)

    def _slot_of(lds, t, e):
        """topk slot of token t's entry for local expert e (t < M)."""
        b0 = lds[fx.Int32(L_TOK8) + t * fx.Int32(2)]
        b1 = lds[fx.Int32(L_TOK8 + 1) + t * fx.Int32(2)]
        slot = fx.Int32(0)
        for k in range_constexpr(TOPK):
            b = ((b0 if k < 4 else b1) >> fx.Int32(8 * (k % 4))) & fx.Int32(0xFF)
            slot = (b == e).select(fx.Int32(k), slot)
        return slot

    def _block_info(lds, mb):
        """m-block mb -> (local expert, valid rows in the block, row index of the block
        inside the expert, expert row count)."""
        row0 = mb * fx.Int32(BM)
        e = fx.Int32(0)
        for k in range_constexpr(EL):
            e = e + (lds[fx.Int32(L_OFF + 1 + k)] <= row0).select(fx.Int32(1), fx.Int32(0))
        e = fx.Int32(rocdl.readfirstlane(T.i32, _raw(e)))
        cnt = lds[fx.Int32(L_CUM + MW) + e * fx.Int32(MW + 1)]
        k0 = row0 - lds[fx.Int32(L_OFF) + e]
        nv = fx.Int32(rocdl.readfirstlane(T.i32, _raw(fx.min(cnt - k0, fx.Int32(BM)))))
        return e, nv, k0, cnt

    def _dispatch(nv, run_rows):
        EP._dispatch_rows(ROWV, [nv <= fx.Int32(r) for r in ROWV[:-1]], run_rows)

    # ---- setup share (zero out, x quant) -------------------------------------------
    @flyc.jit
    def _quant_row(arg_x, arg_x8, arg_xs, t, i32_m, lane):
        """one wave: bf16 x row t -> fp8 + f32 scale (bit-identical to quantize_rows)."""
        if t < i32_m:
            rs_x = buffer_ops.create_buffer_resource_from_addr(_raw(fx.Int64(arg_x)))
            rs_x8 = buffer_ops.create_buffer_resource_from_addr(_raw(fx.Int64(arg_x8)))
            CHL = H // 512
            raw = [Vec(buffer_ops.buffer_load(
                rs_x, t * fx.Int32(H // 2) + (fx.Int32(j * 64) + lane) * fx.Int32(4),
                vec_width=4, dtype=T.i32)) for j in range_constexpr(CHL)]
            vals = []
            amax = fx.Float32(0.0)
            for j in range_constexpr(CHL):
                v = Vec(raw[j].bitcast(fx.BFloat16)).to(fx.Float32)
                vals.append(v)
                for k in range_constexpr(8):
                    amax = fx.max(amax, fx.max(v[k], -v[k]))
            for bit in range_constexpr(6):
                amax = fx.max(amax, amax.shuffle_xor(1 << bit, 64))
            s = (amax == 0.0).select(fx.Float32(1.0), amax / 240.0)
            inv = 1.0 / s
            # masked buffer stores need a bounded resource (masked-off lanes go OOB)
            rs_xs = buffer_ops.create_buffer_resource_from_addr(
                _raw(fx.Int64(arg_xs)), num_records_bytes=_raw(fx.Int64(i32_m) * fx.Int64(4)))
            buffer_ops.buffer_store(s, rs_xs, t, mask=lane == fx.Int32(0), cache_modifier=CM_COH)
            for j in range_constexpr(CHL):
                v = vals[j]
                p0 = _pack_activation(v[0] * inv, v[1] * inv, v[2] * inv, v[3] * inv)
                p1 = _pack_activation(v[4] * inv, v[5] * inv, v[6] * inv, v[7] * inv)
                buffer_ops.buffer_store(
                    Vec.from_elements([p0, p1], fx.Int32), rs_x8,
                    t * fx.Int32(H // 4) + (fx.Int32(j * 64) + lane) * fx.Int32(2),
                    cache_modifier=CM_COH)

    @flyc.jit
    def _setup(a, bx, lane, wave, tx):
        """this WG's share: zero ``out`` (+ a8w4 x quant), released write-through,
        counted with the two-level counter at C_PHA."""
        _zero_slice_wt(a["out"], a["m"] * fx.Int32(H * 2 // 16), bx, a["grid"], tx, NT)
        if const_expr(FP8):
            _quant_row(a["x"], a["x8"], a["xs"], bx * fx.Int32(NW) + wave, a["m"], lane)
        _fence_release()
        _signal2(a["ctr"], fx.Int32(C_PHA), C_SUBA, bx, a["grid"], tx)

    @flyc.jit
    def _setup_if(a, first, lane, wave, tx):
        if first:
            _setup(a, fx.Int32(gpu.block_id("x")), lane, wave, tx)

    def _wait_setup(a, tx):
        _wait_ge(a["ctr"], fx.Int32(C_PHA), fx.min(a["grid"], fx.Int32(NSUB)), tx)

    # ---- items -------------------------------------------------------------------
    def _g1_item(lds_ptr, a, mb, nb, first, lane, wave, tx):
        lds_i32 = fx.Int32(fx.ptrtoint(lds_ptr))
        lds = lds_typed_ptr(lds_i32, T.i32)
        lane_div_16 = lane // fx.Int32(16)
        lane_mod_16 = lane % fx.Int32(16)
        e, nv, k0, cnt = _block_info(lds, mb)
        bx_m = mb * fx.Int32(BM)
        col0 = nb * fx.Int32(NI1) + wave * fx.Int32(16 * C1)  # first inter col of wave
        cols = []
        for gu in range_constexpr(2):
            for c in range_constexpr(C1):
                n0 = e * fx.Int32(2 * I) + fx.Int32(gu * I) + col0 + fx.Int32(c * 16)
                cols.append(((gu, c), n0, (gu, c // 2)))
        exps = {(gu, c, v): fx.Int32(_global_i32_at(
            a["w1e"], e * fx.Int32(2 * I) + fx.Int32(gu * I) + col0 + fx.Int32(c * 16)
            + lane_div_16 * fx.Int32(4) + fx.Int32(v)))
            for gu in range(2) for c in range(C1) for v in range(4)}
        AB = 1 if FP8 else 2
        core = EP._make_core(
            FP8=FP8, K=H, W_ROWS=EL * 2 * I, BM=BM, PF=PF, bcm=2, cols=cols, lds_ptr=lds_ptr,
            LDS_BYTES=LDS_A, lane=lane, tx=tx, NT=NT, arg_w=a["w1"], arg_ws=a["w1s"],
            arg_a=a["x8"] if FP8 else a["x"],
            a_num_bytes=fx.Int64(a["m"]) * fx.Int64(H * AB), ARB=ARB1,
            acm=CM_COH if FP8 else 2, SCH=SCH, TR=True)

        def row_base16(row):
            return _row_token(lds, e, k0 + row, cnt, a["m"]) * fx.Int32(H * AB // 16)

        def run_rows(R):
            def pre_a():
                if const_expr(FP8):  # x8 rows come from the setup phase
                    _setup_if(a, first, lane, wave, tx)
                    _wait_setup(a, tx)

            acc = core(R, row_base16, pre_a=pre_a)
            MR = R // 16
            fac = {k: G1._channel_factor(v) for k, v in exps.items()}
            rows = [fx.Int32(mi * 16) + lane_mod_16 for mi in range(MR)]
            if const_expr(FP8):
                rs = [_gload_sys(a["xs"], fx.min(_row_token(lds, e, k0 + r, cnt, a["m"]),
                                                 a["m"] - fx.Int32(1))).bitcast(fx.Float32)
                      for r in rows]
            out_rsrc = buffer_ops.create_buffer_resource_from_addr(
                _raw(fx.Int64(a["inter"])), num_records_bytes=_raw(
                    fx.Int64(bx_m + fx.Int32(BM)) * fx.Int64(I * 2)))
            ccol = col0 + lane_div_16 * fx.Int32(4)
            situ = None
            if const_expr(ACT == "situv2"):
                situ = situ_params(*[fx.Float32(a[k]) for k in ("sb", "sbr", "lb", "lbr", "lim")])
            for mi in range_constexpr(MR):
                for c in range_constexpr(C1):
                    g4 = Vec(fx.memref_load_vec(acc[0, c][mi]))
                    u4 = Vec(fx.memref_load_vec(acc[1, c][mi]))
                    if const_expr(FP8):
                        gv = [fx.Float32(g4[v]) * (rs[mi] * fac[0, c, v]) for v in range(4)]
                        uv = [fx.Float32(u4[v]) * (rs[mi] * fac[1, c, v]) for v in range(4)]
                    else:
                        gv = [fx.Float32(g4[v]) * fac[0, c, v] for v in range(4)]
                        uv = [fx.Float32(u4[v]) * fac[1, c, v] for v in range(4)]
                    y = gate_up_act(ACT, gv, uv, situ)
                    w2 = Vec.from_elements([EP._bf16x2_rne(y[0], y[1]),
                                            EP._bf16x2_rne(y[2], y[3])], fx.Int32)
                    buffer_ops.buffer_store(
                        w2, out_rsrc,
                        ((bx_m + rows[mi]) * fx.Int32(I) + ccol + fx.Int32(c * 16)) // fx.Int32(2),
                        mask=rows[mi] < nv, cache_modifier=CM_COH)
            if const_expr(not FP8):  # a16w4: the setup share runs after the first item
                _setup_if(a, first, lane, wave, tx)
            _fence_release()
            _signal(a["ctr"], fx.Int32(C_G1) + mb * fx.Int32(STRIDE), tx)

        _dispatch(nv, run_rows)

    def _g2_item(lds_ptr, a, mb, nb, first, lane, wave, tx):
        lds_i32 = fx.Int32(fx.ptrtoint(lds_ptr))
        lds = lds_typed_ptr(lds_i32, T.i32)
        lds_f32 = lds_typed_ptr(lds_i32, T.f32)
        lane_div_16 = lane // fx.Int32(16)
        lane_mod_16 = lane % fx.Int32(16)
        e, nv, k0, cnt = _block_info(lds, mb)
        bx_m = mb * fx.Int32(BM)
        col0 = nb * fx.Int32(NO2) + wave * fx.Int32(16 * C2)
        cols = [((0, c), e * fx.Int32(H) + col0 + fx.Int32(c * 16), (0, c // 2))
                for c in range(C2)]
        exps = [[fx.Int32(_global_i32_at(a["w2e"], e * fx.Int32(H) + col0 + fx.Int32(c * 16)
                                         + lane_div_16 * fx.Int32(4) + fx.Int32(v)))
                 for v in range(4)] for c in range(C2)]
        core = EP._make_core(
            FP8=FP8, K=I, W_ROWS=EL * H, BM=BM, PF=PF, bcm=2, cols=cols, lds_ptr=lds_ptr,
            LDS_BYTES=LDS_A, lane=lane, tx=tx, NT=NT, arg_w=a["w2"], arg_ws=a["w2s"],
            arg_a=a["inter"], a_num_bytes=fx.Int64(CAP) * fx.Int64(I * 2), ARB=ARB2,
            acm=CM_COH, SCH=SCH, TR=True, AQ8=FP8)
        RQ = 2 * BM * ARB2 // 4  # AQ8 ratio slots [2][BM], fin [BM]

        def row_base16(row):
            return (bx_m + row) * fx.Int32(I * 2 // 16)

        def hook(acc, kt, s):
            if const_expr(s != 0):
                return
            for mi in range_constexpr(len(acc[0, 0])):
                rt = lds_f32[fx.Int32(RQ + mi * 16) + (kt % fx.Int32(2)) * fx.Int32(BM)
                             + lane_mod_16]
                for c in range_constexpr(C2):
                    a4 = Vec(acc[0, c][mi].load())
                    acc[0, c][mi].store(Vec.from_elements([a4[v] * rt for v in range(4)],
                                                          fx.Float32))

        def run_rows(R):
            def pre_a():
                _setup_if(a, first, lane, wave, tx)  # before any wait (no deadlock)
                _wait_ge(a["ctr"], fx.Int32(C_G1) + mb * fx.Int32(STRIDE), fx.Int32(NB1), tx)
                _wait_setup(a, tx)  # out zeroed before the atomics

            acc = core(R, row_base16, hook if FP8 else None, pre_a=pre_a)
            MR = R // 16
            fac = [[G1._channel_factor(x) for x in xs4] for xs4 in exps]
            wts = global_typed_ptr(a["wts"], T.f32)
            for mi in range_constexpr(MR):
                r = fx.Int32(mi * 16) + lane_mod_16
                tok = _row_token(lds, e, k0 + r, cnt, a["m"])
                ok = (r < nv) & (tok < a["m"])
                tokc = ok.select(tok, fx.Int32(0))
                wt = fx.Float32(wts[tokc * fx.Int32(TOPK) + _slot_of(lds, tokc, e)])
                if const_expr(FP8):
                    wt = wt * lds_f32[fx.Int32(RQ + 2 * BM) + r]
                for c in range_constexpr(C2):
                    a4 = Vec(acc[0, c][mi].load())
                    v4 = [a4[v] * (wt * fac[c][v]) for v in range(4)]
                    base = tokc * fx.Int32(H) + col0 + fx.Int32(c * 16) + lane_div_16 * fx.Int32(4)
                    EP._atomic_bf16x4(a["out"], base, v4, ok)

        _dispatch(nv, run_rows)

    @flyc.jit
    def _item(lds_ptr, a, item, ng1, first, lane, wave, tx):
        if item < ng1:
            _g1_item(lds_ptr, a, item // fx.Int32(NB1), item % fx.Int32(NB1), first, lane,
                     wave, tx)
        else:
            j = item - ng1
            _g2_item(lds_ptr, a, j // fx.Int32(NB2), j % fx.Int32(NB2), first, lane, wave, tx)

    @flyc.jit
    def _late_setup(a, bx, ntot, lane, wave, tx):
        if bx >= ntot:  # no item: the setup share is still due
            _setup(a, bx, lane, wave, tx)

    @flyc.jit
    def _finish(arg_ctr, grid, nmb, tx):
        """two-level exit count; the last WG out resets every counter."""
        gpu.barrier()
        if tx == fx.Int32(0):
            g = fx.Int32(gpu.block_id("x")) % fx.Int32(NSUB)
            n_in_g = (grid - g + fx.Int32(NSUB - 1)) // fx.Int32(NSUB)
            old = _gatomic_add(arg_ctr, fx.Int32(C_SUBX) + g * fx.Int32(STRIDE), 1)
            if old == n_in_g - fx.Int32(1):
                top = _gatomic_add(arg_ctr, fx.Int32(C_EXIT), 1)
                if top == fx.min(grid, fx.Int32(NSUB)) - fx.Int32(1):
                    _gatomic_xchg(arg_ctr, fx.Int32(C_PHA), 0)
                    _gatomic_xchg(arg_ctr, fx.Int32(C_EXIT), 0)
                    for k in range_constexpr(NSUB):
                        _gatomic_xchg(arg_ctr, fx.Int32(C_SUBA + k * STRIDE), 0)
                        _gatomic_xchg(arg_ctr, fx.Int32(C_SUBX + k * STRIDE), 0)
                    for k in range(nmb):
                        _gatomic_xchg(arg_ctr, fx.Int32(C_G1) + fx.Int32(k) * fx.Int32(STRIDE), 0)

    @flyc.jit
    def _entry(arg_x, arg_x8, arg_xs, arg_ids, arg_wts, arg_w1, arg_w1s, arg_w1e, arg_w2,
               arg_w2s, arg_w2e, arg_out, arg_inter, arg_ctr, i32_m, i32_e0, f32_sb, f32_sbr,
               f32_lb, f32_lbr, f32_lim):
        lds_ptr = fx.SharedAllocator().allocate(SharedStorage).peek().raw.ptr
        lds_i32 = fx.Int32(fx.ptrtoint(lds_ptr))
        lds = lds_typed_ptr(lds_i32, T.i32)
        tx = fx.Int32(gpu.thread_id("x"))
        bx = fx.Int32(gpu.block_id("x"))
        grid = fx.Int32(gpu.grid_dim.x)
        lane = tx % fx.Int32(64)
        wave = fx.Int32(rocdl.readfirstlane(T.i32, _raw(tx // fx.Int32(64))))
        a = dict(x=arg_x, x8=arg_x8, xs=arg_xs, ids=arg_ids, wts=arg_wts, w1=arg_w1,
                 w1s=arg_w1s, w1e=arg_w1e, w2=arg_w2, w2s=arg_w2s, w2e=arg_w2e, out=arg_out,
                 inter=arg_inter, ctr=arg_ctr, m=i32_m, e0=i32_e0, grid=grid, sb=f32_sb,
                 sbr=f32_sbr, lb=f32_lb, lbr=f32_lbr, lim=f32_lim)
        _route(lds, a, wave, lane, tx)
        nmb = fx.Int32(rocdl.readfirstlane(T.i32, _raw(lds[fx.Int32(L_OFF + EL)]))) // fx.Int32(BM)
        ng1 = nmb * fx.Int32(NB1)
        ntot = ng1 + nmb * fx.Int32(NB2)
        if const_expr(DBG == 1):
            ntot = ng1
        if const_expr(DBG == 2):
            ntot = fx.Int32(0)
        for item in range(bx, ntot, grid):
            _item(lds_ptr, a, fx.Int32(item), ng1, fx.Int32(item) == bx, lane, wave, tx)
            gpu.barrier()  # LDS A slots reused by the next item
        _late_setup(a, bx, ntot, lane, wave, tx)
        _finish(arg_ctr, grid, nmb, tx)

    @flyc.kernel(name=kname, known_block_size=[NT, 1, 1])
    def kernel(arg_x: fx.Int64, arg_x8: fx.Int64, arg_xs: fx.Int64, arg_ids: fx.Int64,
               arg_wts: fx.Int64, arg_w1: fx.Int64, arg_w1s: fx.Int64, arg_w1e: fx.Int64,
               arg_w2: fx.Int64, arg_w2s: fx.Int64, arg_w2e: fx.Int64, arg_out: fx.Int64,
               arg_inter: fx.Int64, arg_ctr: fx.Int64, i32_m: fx.Int32, i32_e0: fx.Int32,
               f32_sb: fx.Float32, f32_sbr: fx.Float32, f32_lb: fx.Float32,
               f32_lbr: fx.Float32, f32_lim: fx.Float32):
        _entry(arg_x, arg_x8, arg_xs, arg_ids, arg_wts, arg_w1, arg_w1s, arg_w1e, arg_w2,
               arg_w2s, arg_w2e, arg_out, arg_inter, arg_ctr, i32_m, i32_e0, f32_sb, f32_sbr,
               f32_lb, f32_lbr, f32_lim)

    @flyc.jit
    def launch(arg_x: fx.Int64, arg_x8: fx.Int64, arg_xs: fx.Int64, arg_ids: fx.Int64,
               arg_wts: fx.Int64, arg_w1: fx.Int64, arg_w1s: fx.Int64, arg_w1e: fx.Int64,
               arg_w2: fx.Int64, arg_w2s: fx.Int64, arg_w2e: fx.Int64, arg_out: fx.Int64,
               arg_inter: fx.Int64, arg_ctr: fx.Int64, i32_m: fx.Int32, i32_e0: fx.Int32,
               f32_sb: fx.Float32, f32_sbr: fx.Float32, f32_lb: fx.Float32,
               f32_lbr: fx.Float32, f32_lim: fx.Float32, i32_grid: fx.Int32,
               stream: fx.Stream):
        kernel(arg_x, arg_x8, arg_xs, arg_ids, arg_wts, arg_w1, arg_w1s, arg_w1e, arg_w2,
               arg_w2s, arg_w2e, arg_out, arg_inter, arg_ctr, i32_m, i32_e0, f32_sb,
               f32_sbr, f32_lb, f32_lbr,
               f32_lim).launch(grid=(fx.Int64(i32_grid), 1, 1), block=(NT, 1, 1),
                               stream=stream)

    return launch, dict(CAP=CAP, NMBMAX=CAP // BM, LDS_BYTES=LDS_BYTES, NI1=NI1, NO2=NO2)


_BUF = {}


def _scratch(dev, adt, maxm, H, I, cap, nmbmax):
    key = (dev, adt, maxm, H, I, cap)
    if key not in _BUF:
        _BUF[key] = dict(
            x8=torch.empty((maxm, H), dtype=torch.uint8, device=dev),
            xs=torch.empty(maxm, dtype=torch.float32, device=dev),
            inter=torch.empty((cap, I), dtype=torch.bfloat16, device=dev),
            ctr=torch.zeros(C_G1 + nmbmax * STRIDE, dtype=torch.int32, device=dev),
        )
    return _BUF[key]


SMALL_CFG = dict(bm=32, nw=2, c1=1, c2=4, pf=4, rowv=(16, 32), wpc=2)


def ep_moe_small(x, W, topk_weights, topk_ids, *, e0, adt="fp8", maxm=512, cfg=None,
                 act="silu", **situ):
    """Fused small-batch EP8 MoE (one launch): bf16 x [M, H] -> bf16 [M, H]; same
    contract as ``ep_gemm.ep_moe`` (incl. ``act`` / ``situ``). ``M <= maxm``
    (compile-time capacity)."""
    cfg = dict(SMALL_CFG, **(cfg or {}))
    M, H = x.shape
    E = W["w2"].shape[0]
    I = W["w2"].shape[2] * 2
    K = topk_ids.shape[1]
    assert M <= maxm
    launch, info = compile_ep_small(
        H=H, I=I, EL=E, TOPK=K, ADT=adt, BM=cfg["bm"], NW=cfg["nw"], C1=cfg["c1"],
        C2=cfg["c2"], PF=cfg["pf"], ROWV=tuple(cfg["rowv"]), MAXM=maxm,
        SCH=cfg.get("sch", 3), DBG=cfg.get("dbg", 0), ACT=act)
    s = _scratch(x.device, adt, maxm, H, I, info["CAP"], info["NMBMAX"])
    out = torch.empty((M, H), dtype=torch.bfloat16, device=x.device)
    grid = NUM_CU * cfg["wpc"]
    _run_compiled(
        launch, x.data_ptr(), s["x8"].data_ptr(), s["xs"].data_ptr(), topk_ids.data_ptr(),
        topk_weights.data_ptr(), W["w1"].data_ptr(), W["w1_scale"].data_ptr(),
        W["w1_exp"].data_ptr(), W["w2"].data_ptr(), W["w2_scale"].data_ptr(),
        W["w2_exp"].data_ptr(), out.data_ptr(), s["inter"].data_ptr(), s["ctr"].data_ptr(),
        int(M), int(e0), *G1.situ_args(**situ), int(grid), torch.cuda.current_stream())
    return out
