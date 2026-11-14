# SPDX-License-Identifier: MIT
"""TP fused prep (experiment; gfx942, FlyDSL): routing sort over ALL E (<= 256)
experts + x row quant + zeroing of the bf16 output in ONE launch, replacing aiter
``moe_sorting`` (sort + ``out`` zeroing) + ``quantize_rows`` on the TP path.

Derived from kernels/ep_prep.py (EP: <= 32 local experts). Changes:
  * E <= 256: the padded-count scan runs over all 4 waves (64-lane ds_bpermute scan
    per wave, wave totals through LDS);
  * no EP tile schedule / route counts / SID_SINGLE (the TP GEMMs do not use them);
  * every token is local: the quant WGs walk all rows (no ids load / ballot) and
    (``ZERO``) also zero the token's ``out`` row (the same 16 B lanes as the x loads);
  * padding rows / block expert ids: one wave per expert, lanes = consecutive rows /
    blocks (coalesced; ep_prep gives each expert NT // EL threads, i.e. 1 thread and
    BM scattered stores for 256 experts), spread over all sort WGs;
  * counter reset by all threads of the last WG;
  * tight capacity: M*K // BM + min(E, M*K) blocks (smaller GEMM grids at small M).

Outputs = moe_sorting's (sorted_ids, sorted_weights, sorted_expert_ids,
num_valid_ids = [padded rows, M], zeroed out) + quantize_rows' (x8, xs).
"""
import functools
import os
import sys

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from a8w4._vendor import buffer_ops  # noqa: E402
from a8w4._vendor.a8w4_ops import _pack_activation  # noqa: E402
from a8w4._vendor.mxfp4_gemm_common import global_typed_ptr, lds_typed_ptr  # noqa: E402
from a8w4._vendor.tensor_shim import _run_compiled  # noqa: E402
from a8w4._vendor.tensor_shim import _to_raw as _raw  # noqa: E402
from a8w4.kernels.ep_prep import (  # noqa: E402
    NUM_CU, _gatomic_add, _gatomic_xchg, _gload_sys, _latomic_add)

NT = 256
NW = NT // 64


@functools.cache
def compile_tp_prep(*, H, TOPK, E, BM, EPT, MULTI, ZERO):
    """Sort coverage: grid * 256 * EPT >= M * TOPK (MULTI) or 256 * EPT (single)."""
    assert H % 512 == 0 and 1 <= E <= NT
    CH = H // 512  # 16 B (8 bf16) loads per lane per row
    # LDS (i32): cnt[E+1] (slot E = dummy for invalid entries), base[E], gcnt[E],
    # off[E+1], per-wave scan totals, last-WG flag
    L_CNT, L_BASE, L_GCNT, L_OFF = 0, E + 1, 2 * E + 1, 3 * E + 1
    L_WS = 4 * E + 2
    L_LAST = L_WS + NW
    LDS_I32 = L_LAST + 1
    kname = f"tp_prep_h{H}_k{TOPK}_e{E}_bm{BM}_ept{EPT}_m{int(MULTI)}_z{int(ZERO)}"

    @fx.struct
    class SharedStorage:
        raw: fx.Array[fx.Uint8, LDS_I32 * 4, 16]

    # ---- x row quant (+ out zeroing) --------------------------------------------
    @flyc.jit
    def _store_scale(xs, t, s, lane):
        if lane == fx.Int32(0):
            xs[t] = s

    @flyc.jit
    def _quant(arg_x, arg_x8, arg_xs, arg_out, i32_m, gw, tw, lane):
        """rows gw + i*tw of x (bf16 [M, H]) -> fp8 + f32 row scale (bit identical to
        quantize_rows); ZERO: the same rows of out (bf16 [M, H]) := 0."""
        rs_x = buffer_ops.create_buffer_resource_from_addr(_raw(fx.Int64(arg_x)))
        rs_x8 = buffer_ops.create_buffer_resource_from_addr(_raw(fx.Int64(arg_x8)))
        rs_o = buffer_ops.create_buffer_resource_from_addr(_raw(fx.Int64(arg_out)))
        xs = global_typed_ptr(arg_xs, T.f32)
        z4 = Vec.filled(4, 0, fx.Int32)
        n_do = (gw < i32_m).select((i32_m - gw + tw - fx.Int32(1)) // tw, fx.Int32(0))

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
            for j in range_constexpr(CH if ZERO else 0):
                buffer_ops.buffer_store(
                    z4, rs_o, t * fx.Int32(H // 2) + (fx.Int32(j * 64) + lane) * fx.Int32(4))

        # two rows per iteration: row j+1's loads are in flight while row j is packed
        for i in range((n_do + fx.Int32(1)) // fx.Int32(2)):
            j0 = fx.Int32(i) * fx.Int32(2)
            t0 = gw + j0 * tw
            t1 = gw + fx.min(j0 + fx.Int32(1), n_do - fx.Int32(1)) * tw
            r0 = load_row(t0)
            r1 = load_row(t1)
            do_row(t0, r0)
            do_row(t1, r1)  # odd count: t1 == t0, same values rewritten

    # ---- sort ----------------------------------------------------------------------
    def _count(lds_i32, arg_ids, arg_w, first, n_ent, tx):
        """entries first + j*NT + tx -> (expert or E, slot in its expert, weight, entry)."""
        ids = global_typed_ptr(arg_ids, T.i32)
        wts = global_typed_ptr(arg_w, T.f32)
        ent = []
        for j in range_constexpr(EPT):
            idx = first + fx.Int32(j * NT) + tx
            ok = idx < n_ent
            idc = ok.select(idx, fx.Int32(0))
            e = fx.Int32(ids[idc])
            es = (ok & (e >= fx.Int32(0)) & (e < fx.Int32(E))).select(e, fx.Int32(E))
            ent.append((es, _latomic_add(lds_i32, fx.Int32(L_CNT) + es, 1),
                        fx.Float32(wts[idc]), idc))
        return ent

    @flyc.jit
    def _st_if(ptr, idx, val, cond):
        if cond:
            ptr[idx] = val

    def _prefix(lds, tx, lane, wave):
        """thread e < E: padded count of expert e -> exclusive scan -> off[e];
        off[E] = total. All threads (barrier inside)."""
        c = (tx < fx.Int32(E)).select(lds[fx.Int32(L_GCNT) + fx.min(tx, fx.Int32(E - 1))],
                                      fx.Int32(0))
        pad = (c + fx.Int32(BM - 1)) // fx.Int32(BM) * fx.Int32(BM)
        inc = pad
        for bit in range_constexpr(6):
            d = 1 << bit
            src = fx.max(lane - fx.Int32(d), fx.Int32(0))
            other = fx.Int32(rocdl.ds_bpermute(T.i32, _raw(src * fx.Int32(4)), _raw(inc)))
            inc = inc + (lane >= fx.Int32(d)).select(other, fx.Int32(0))
        _st_if(lds, fx.Int32(L_WS) + wave, inc, lane == fx.Int32(63))
        gpu.barrier()
        for w in range_constexpr(NW - 1):
            inc = inc + (wave > fx.Int32(w)).select(lds[fx.Int32(L_WS + w)], fx.Int32(0))
        _st_if(lds, fx.Int32(L_OFF) + fx.min(tx, fx.Int32(E - 1)), inc - pad, tx < fx.Int32(E))
        _st_if(lds, fx.Int32(L_OFF + E), inc, tx == fx.Int32(E - 1))

    @flyc.jit
    def _scatter(lds, sids, sw, e, pos, w, idx):
        if e < fx.Int32(E):
            dst = lds[fx.Int32(L_OFF) + e] + lds[fx.Int32(L_BASE) + e] + pos
            sids[dst] = (idx // fx.Int32(TOPK)) | ((idx % fx.Int32(TOPK)) << fx.Int32(24))
            sw[dst] = w

    @flyc.jit
    def _pad_expert(lds, sids, sw, seids, e, lane, i32_m):
        """one wave: expert e's padding rows (lane = row) and block expert ids."""
        o0 = fx.Int32(rocdl.readfirstlane(T.i32, _raw(lds[fx.Int32(L_OFF) + e])))
        o1 = fx.Int32(rocdl.readfirstlane(T.i32, _raw(lds[fx.Int32(L_OFF + 1) + e])))
        c = fx.Int32(rocdl.readfirstlane(T.i32, _raw(lds[fx.Int32(L_GCNT) + e])))
        for h in range_constexpr(-(-BM // 64)):
            r = o0 + c + lane + fx.Int32(64 * h)
            if r < o1:
                sids[r] = i32_m | fx.Int32(TOPK << 24)
                sw[r] = fx.Float32(0.0)
        b0 = o0 // fx.Int32(BM)
        nb = (o1 - o0) // fx.Int32(BM)
        for k in range((nb + fx.Int32(63)) // fx.Int32(64)):
            b = fx.Int32(k) * fx.Int32(64) + lane
            if b < nb:
                seids[b0 + b] = e

    @flyc.jit
    def _pad_all(lds, arg_sids, arg_sw, arg_seids, i32_m, me, n_sort, wave, lane):
        """experts gwv, gwv + nwv, ... (gwv: this wave's index over the sort WGs)."""
        sids = global_typed_ptr(arg_sids, T.i32)
        sw = global_typed_ptr(arg_sw, T.f32)
        seids = global_typed_ptr(arg_seids, T.i32)
        gwv = me * fx.Int32(NW) + wave
        nwv = n_sort * fx.Int32(NW)
        for i in range((fx.Int32(E) - gwv + nwv - fx.Int32(1)) // nwv):
            _pad_expert(lds, sids, sw, seids, gwv + fx.Int32(i) * nwv, lane, i32_m)

    @flyc.jit
    def _nv_store(lds, arg_nv, i32_m, me, tx):
        if me == fx.Int32(0):
            if tx == fx.Int32(0):
                nv = global_typed_ptr(arg_nv, T.i32)
                nv[fx.Int32(0)] = lds[fx.Int32(L_OFF + E)]
                nv[fx.Int32(1)] = i32_m

    def _finish_sort(lds, arg_sids, arg_sw, arg_seids, arg_nv, ent, i32_m, me, n_sort, tx,
                     lane, wave):
        sids = global_typed_ptr(arg_sids, T.i32)
        sw = global_typed_ptr(arg_sw, T.f32)
        for es, pos, w, idc in ent:
            _scatter(lds, sids, sw, es, pos, w, idc)
        _pad_all(lds, arg_sids, arg_sw, arg_seids, i32_m, me, n_sort, wave, lane)
        _nv_store(lds, arg_nv, i32_m, me, tx)

    @flyc.jit
    def _init_cnt(lds, tx):
        if tx < fx.Int32(E):
            lds[fx.Int32(L_CNT) + tx] = fx.Int32(0)
        if tx == fx.Int32(0):
            lds[fx.Int32(L_CNT + E)] = fx.Int32(0)

    @flyc.jit
    def _copy_counts(lds, tx):
        if tx < fx.Int32(E):
            lds[fx.Int32(L_GCNT) + tx] = lds[fx.Int32(L_CNT) + tx]
            lds[fx.Int32(L_BASE) + tx] = fx.Int32(0)

    @flyc.jit
    def _reserve(lds, arg_ws, tx):
        if tx < fx.Int32(E):
            n = lds[fx.Int32(L_CNT) + tx]
            lds[fx.Int32(L_BASE) + tx] = fx.Int32(0)
            if n > fx.Int32(0):
                lds[fx.Int32(L_BASE) + tx] = _gatomic_add(arg_ws, tx, n)

    @flyc.jit
    def _read_counts(lds, arg_ws, tx):
        if tx < fx.Int32(E):
            lds[fx.Int32(L_GCNT) + tx] = _gload_sys(arg_ws, tx)

    @flyc.jit
    def _grid_sync(arg_ws, n_sort, tx):
        """barrier of the n_sort sort WGs (resident: lowest block ids, few)."""
        if tx == fx.Int32(0):
            _gatomic_add(arg_ws, fx.Int32(E), 1)
            seen = _gload_sys(arg_ws, fx.Int32(E))
            while seen < n_sort:
                rocdl.s_sleep(1)
                seen = _gload_sys(arg_ws, fx.Int32(E))

    @flyc.jit
    def _finish_flag(lds, arg_ws, n_sort, tx):
        """counts already read by this WG: the last WG to get here resets them."""
        if tx == fx.Int32(0):
            old = _gatomic_add(arg_ws, fx.Int32(E + 1), 1)
            lds[fx.Int32(L_LAST)] = (old == n_sort - fx.Int32(1)).select(fx.Int32(1),
                                                                         fx.Int32(0))

    @flyc.jit
    def _finish_reset(lds, arg_ws, tx):
        if lds[fx.Int32(L_LAST)] == fx.Int32(1):
            if tx < fx.Int32(E):
                _gatomic_xchg(arg_ws, tx, 0)
            if tx < fx.Int32(2):
                _gatomic_xchg(arg_ws, fx.Int32(E) + tx, 0)

    def _single_sort(lds_i32, lds, arg_ids, arg_w, arg_sids, arg_sw, arg_seids, arg_nv,
                     i32_m, tx, lane, wave):
        """whole sort in one WG, LDS only (M * TOPK <= 256 * EPT)."""
        ent = _count(lds_i32, arg_ids, arg_w, fx.Int32(0), i32_m * fx.Int32(TOPK), tx)
        gpu.barrier()
        _copy_counts(lds, tx)
        gpu.barrier()
        _prefix(lds, tx, lane, wave)
        gpu.barrier()
        _finish_sort(lds, arg_sids, arg_sw, arg_seids, arg_nv, ent, i32_m, fx.Int32(0),
                     fx.Int32(1), tx, lane, wave)

    def _multi_sort(lds_i32, lds, arg_ids, arg_w, arg_sids, arg_sw, arg_seids, arg_nv,
                    arg_ws, i32_m, n_sort, bx, tx, lane, wave):
        ent = _count(lds_i32, arg_ids, arg_w, bx * fx.Int32(EPT * NT),
                     i32_m * fx.Int32(TOPK), tx)
        gpu.barrier()
        _reserve(lds, arg_ws, tx)
        rocdl.s_waitcnt(vmcnt=0)
        gpu.barrier()
        _grid_sync(arg_ws, n_sort, tx)
        gpu.barrier()
        _read_counts(lds, arg_ws, tx)
        gpu.barrier()
        _prefix(lds, tx, lane, wave)
        gpu.barrier()
        _finish_flag(lds, arg_ws, n_sort, tx)
        gpu.barrier()
        _finish_reset(lds, arg_ws, tx)
        _finish_sort(lds, arg_sids, arg_sw, arg_seids, arg_nv, ent, i32_m, bx, n_sort, tx,
                     lane, wave)

    @flyc.jit
    def _sort_role(lds_i32, lds, arg_ids, arg_w, arg_sids, arg_sw, arg_seids, arg_nv,
                   arg_ws, i32_m, n_sort, bx, tx, lane, wave):
        _init_cnt(lds, tx)
        gpu.barrier()
        if const_expr(MULTI):
            _multi_sort(lds_i32, lds, arg_ids, arg_w, arg_sids, arg_sw, arg_seids, arg_nv,
                        arg_ws, i32_m, n_sort, bx, tx, lane, wave)
        else:
            _single_sort(lds_i32, lds, arg_ids, arg_w, arg_sids, arg_sw, arg_seids,
                         arg_nv, i32_m, tx, lane, wave)

    @flyc.jit
    def _entry(arg_ids, arg_w, arg_x, arg_x8, arg_xs, arg_out, arg_sids, arg_sw,
               arg_seids, arg_nv, arg_ws, i32_m):
        lds_ptr = fx.SharedAllocator().allocate(SharedStorage).peek().raw.ptr
        lds_i32 = fx.Int32(fx.ptrtoint(lds_ptr))
        lds = lds_typed_ptr(lds_i32, T.i32)
        tx = fx.Int32(gpu.thread_id("x"))
        bx = fx.Int32(gpu.block_id("x"))
        grid = fx.Int32(gpu.grid_dim.x)
        lane = tx % fx.Int32(64)
        wave = fx.Int32(rocdl.readfirstlane(T.i32, _raw(tx // fx.Int32(64))))
        # WGs [0, n_sort): sort; [n_sort, grid): x quant + out zeroing (no sync)
        n_sort = fx.max((i32_m * fx.Int32(TOPK) + fx.Int32(EPT * NT - 1)) // fx.Int32(EPT * NT),
                        fx.Int32(1))
        if const_expr(not MULTI):
            n_sort = fx.Int32(1)
        if bx < n_sort:
            _sort_role(lds_i32, lds, arg_ids, arg_w, arg_sids, arg_sw, arg_seids, arg_nv,
                       arg_ws, i32_m, n_sort, bx, tx, lane, wave)
        if bx >= n_sort:
            _quant(arg_x, arg_x8, arg_xs, arg_out, i32_m,
                   (bx - n_sort) * fx.Int32(NW) + wave, (grid - n_sort) * fx.Int32(NW), lane)

    @flyc.kernel(name=kname, known_block_size=[NT, 1, 1])
    def kernel(arg_ids: fx.Int64, arg_w: fx.Int64, arg_x: fx.Int64, arg_x8: fx.Int64,
               arg_xs: fx.Int64, arg_out: fx.Int64, arg_sids: fx.Int64, arg_sw: fx.Int64,
               arg_seids: fx.Int64, arg_nv: fx.Int64, arg_ws: fx.Int64, i32_m: fx.Int32):
        _entry(arg_ids, arg_w, arg_x, arg_x8, arg_xs, arg_out, arg_sids, arg_sw, arg_seids,
               arg_nv, arg_ws, i32_m)

    @flyc.jit
    def launch(arg_ids: fx.Int64, arg_w: fx.Int64, arg_x: fx.Int64, arg_x8: fx.Int64,
               arg_xs: fx.Int64, arg_out: fx.Int64, arg_sids: fx.Int64, arg_sw: fx.Int64,
               arg_seids: fx.Int64, arg_nv: fx.Int64, arg_ws: fx.Int64, i32_m: fx.Int32,
               i32_grid: fx.Int32, stream: fx.Stream):
        kernel(arg_ids, arg_w, arg_x, arg_x8, arg_xs, arg_out, arg_sids, arg_sw, arg_seids,
               arg_nv, arg_ws, i32_m).launch(grid=(fx.Int64(i32_grid), 1, 1),
                                             block=(NT, 1, 1), stream=stream)

    return launch


_WS = {}


def _workspace(device, n):
    key = (device, n)
    if key not in _WS:
        _WS[key] = torch.zeros(n, dtype=torch.int32, device=device)
    return _WS[key]


def tp_prep(topk_ids, topk_weights, x, *, block_m, num_experts, zero=True, ept=8,
            tight=True):
    """-> (sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, x8, xs, out):
    the moe_sorting contract over all ``num_experts`` experts (capacity M*K // BM +
    min(E, M*K) blocks if ``tight``, else M*K // BM + E), fp8 x + f32 row scales
    (as quantize_rows) and the bf16 output [M, H], zeroed if ``zero``."""
    M, K = topk_ids.shape
    H = x.shape[1]
    E = num_experts
    dev = x.device
    assert M < (1 << 24)
    nblk = (M * K) // block_m + (min(E, M * K) if tight else E)
    cap = nblk * block_m
    sids = torch.empty(cap, dtype=torch.int32, device=dev)
    sw = torch.empty(cap, dtype=torch.float32, device=dev)
    seids = torch.empty(nblk, dtype=torch.int32, device=dev)
    nv = torch.empty(2, dtype=torch.int32, device=dev)
    x8 = torch.empty((M, H), dtype=torch.float8_e4m3fnuz, device=dev)
    xs = torch.empty(M, dtype=torch.float32, device=dev)
    out = torch.empty((M, H), dtype=torch.bfloat16, device=dev)
    multi = M * K > NT * ept
    n_sort = -(-(M * K) // (NT * ept)) if multi else 1
    assert n_sort <= NUM_CU, f"M={M} too large"
    n_quant = min(4 * NUM_CU, -(-M // NW))
    grid = n_sort + n_quant
    launch = compile_tp_prep(H=H, TOPK=K, E=E, BM=block_m, EPT=ept, MULTI=multi,
                             ZERO=zero)
    ws = _workspace(dev, E + 2)
    _run_compiled(
        launch, topk_ids.data_ptr(), topk_weights.data_ptr(), x.data_ptr(), x8.data_ptr(),
        xs.data_ptr(), out.data_ptr(), sids.data_ptr(), sw.data_ptr(), seids.data_ptr(),
        nv.data_ptr(), ws.data_ptr(), int(M), int(grid), torch.cuda.current_stream())
    return sids, sw, seids, nv, x8, xs, out
