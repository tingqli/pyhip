# SPDX-License-Identifier: MIT
"""TP fused prep v2c = v2b + routes per sort WG by size (experiment copy with the SKIP
diagnostics; the package copy is a8w4/kernels/tp_prep.py, used by moe.py's TP kinds).
Gfx942, FlyDSL: routing sort over ALL E (<= 256)
experts + x row quant + zeroing of the bf16 output in ONE launch, replacing aiter
``moe_sorting`` (sort + ``out`` zeroing) + ``quantize_rows`` on the TP path.

WGs [0, n_sort): sort, RNG = 256 * EPT routes each (at least NSMIN WGs, for the
padding; by default RNG targets ~64 sort WGs, 256 .. 2048: a WG's scatter is EPT fully
divergent 4 B stores per thread and array, its histogram reads all M*K ids); [n_sort, grid): quant, one wave per row (as kernels/ep_prep.py; every token is
local, so all rows), each row's ``out`` row zeroed with the same 16 B lanes (``ZERO``).
The two roles never synchronize.

Sort (no global atomics / grid barrier / workspace): a sort WG first issues the loads of
its own routes (ids + weights, kept in registers), then histograms ALL M*K expert ids
itself (16 B loads, HU per thread per round trip, from L2 after the first WG; LDS
atomics), split at its range start (``bef``: routes before its range, ``rest``: from it
on), so it knows every expert's total and its own base without talking to the other WGs;
padded counts -> exclusive scan over all 4 waves (ds_bpermute scan + wave totals in LDS);
then its own routes: LDS atomic slot -> sorted_ids / sorted_weights. Padding rows and
block expert ids: expert e is done by sort wave e mod (4 n_sort): offsets read from LDS
(broadcast), lanes = consecutive rows / blocks (coalesced). Out-of-range / invalid
routes count into per-lane dummy slots [E, E + 64) (one shared slot serialized every
LDS atomic of a wave).

History (tp_prep_v1.py, tp_prep_v2a.py): v1 = ep_prep's per-expert global atomics +
grid barrier + counter reset: ~5 dependent global round trips under the quant WGs' HBM
load. v2a = this histogram but HU 4 (several round trips), loads of the own routes after
the histogram (another round trip), a shared dummy slot, padding by a 64-step readlane
loop with a runtime modulo on few waves (issue bound, 5 - 11 us), and the quant of a
wave's odd last row done twice (x8 / out stores doubled at T <= 4864).

Capacity: M*K // BM + min(E, M*K) blocks (``tight``; a valid bound since only
non-empty experts get blocks), else M*K // BM + E (as aiter / ep_prep).
Outputs = moe_sorting's (sorted_ids, sorted_weights, sorted_expert_ids,
num_valid_ids = [padded rows, M], zeroed out) + quantize_rows' (x8, xs).
"""
import functools
import os
import sys

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from a8w4.kernels._atomic import _latomic_add  # noqa: E402
from a8w4.kernels._cvt import _fp8x4_pack  # noqa: E402
from a8w4.kernels._hw import NUM_CU  # noqa: E402
from a8w4.experiments._launch import _run_compiled  # noqa: E402
from a8w4.kernels._mem import (  # noqa: E402
    _raw,
    buffer_load,
    buffer_store,
    create_buffer_resource_from_addr,
    global_typed_ptr,
    lds_typed_ptr,
)

NT = 256
NW = NT // 64
ND = 64  # dummy LDS counters per array (one per lane)


def _uni(v):
    return fx.Int32(rocdl.readfirstlane(T.i32, _raw(v)))


@functools.cache
def compile_tp_prep(*, H, TOPK, E, BM, EPT, HU, ZERO, SKIP=""):
    """EPT: own routes per sort thread (RNG = 256 * EPT per sort WG); HU: 16 B id loads
    per thread per histogram round trip. SKIP (diagnostics only; output invalid): comma
    list of parts left out: hist, scatter, pad, sort, quant."""
    assert H % 512 == 0 and 1 <= E <= NT and TOPK % 4 == 0
    skip = set(SKIP.split(",")) - {""}
    assert skip <= {"hist", "scatter", "pad", "sort", "quant"}, SKIP
    # ints, not a helper function: FlyDSL's JIT cache key covers int / bool / str
    # closure values but not callables (an on(part) helper got the full kernel reused)
    DO_HIST, DO_SCAT, DO_PAD, DO_SORT, DO_QUANT = (
        int(p not in skip) for p in ("hist", "scatter", "pad", "sort", "quant"))
    RNG = NT * EPT
    CH = H // 512  # 16 B (8 bf16) loads per lane per row
    EA = E + ND
    # LDS (i32): cnt / bef / rest [E + ND] each, gcnt[E], off[E+1], per-wave scan totals
    L_CNT, L_BEF, L_REST = 0, EA, 2 * EA
    L_GCNT = 3 * EA
    L_OFF = L_GCNT + E
    L_WS = L_OFF + E + 1
    LDS_I32 = L_WS + NW
    kname = f"tp_prep2b_h{H}_k{TOPK}_e{E}_bm{BM}_t{EPT}_u{HU}_z{int(ZERO)}" + "".join(
        f"_no{p}" for p in sorted(skip))

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
        rs_x = create_buffer_resource_from_addr(_raw(fx.Int64(arg_x)))
        rs_x8 = create_buffer_resource_from_addr(_raw(fx.Int64(arg_x8)))
        rs_o = create_buffer_resource_from_addr(_raw(fx.Int64(arg_out)))
        xs = global_typed_ptr(arg_xs, T.f32)
        z4 = Vec.filled(4, 0, fx.Int32)
        n_do = (gw < i32_m).select((i32_m - gw + tw - fx.Int32(1)) // tw, fx.Int32(0))

        def load_row(t):
            return [Vec(buffer_load(
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
                p0 = _fp8x4_pack(v[0] * inv, v[1] * inv, v[2] * inv, v[3] * inv)
                p1 = _fp8x4_pack(v[4] * inv, v[5] * inv, v[6] * inv, v[7] * inv)
                buffer_store(
                    Vec.from_elements([p0, p1], fx.Int32), rs_x8,
                    t * fx.Int32(H // 4) + (fx.Int32(j * 64) + lane) * fx.Int32(2))
            for j in range_constexpr(CH if ZERO else 0):
                buffer_store(
                    z4, rs_o, t * fx.Int32(H // 2) + (fx.Int32(j * 64) + lane) * fx.Int32(4))

        # two rows per iteration (row j+1's loads in flight while row j is packed), then
        # the odd last row alone
        for i in range(n_do // fx.Int32(2)):
            t0 = gw + fx.Int32(i) * fx.Int32(2) * tw
            t1 = t0 + tw
            r0 = load_row(t0)
            r1 = load_row(t1)
            do_row(t0, r0)
            do_row(t1, r1)
        if n_do % fx.Int32(2) == fx.Int32(1):
            t = gw + (n_do - fx.Int32(1)) * tw
            do_row(t, load_row(t))

    # ---- sort ----------------------------------------------------------------------
    @flyc.jit
    def _st_if(ptr, idx, val, cond):
        if cond:
            ptr[idx] = val

    @flyc.jit
    def _init(lds, tx):
        for i in range_constexpr(-(-3 * EA // NT)):
            idx = fx.Int32(i * NT) + tx
            if idx < fx.Int32(3 * EA):
                lds[fx.Int32(L_CNT) + idx] = fx.Int32(0)

    def _slot(e, ok, lane):
        """counter slot of a route: its expert, or the lane's dummy slot."""
        good = ok & (e >= fx.Int32(0)) & (e < fx.Int32(E))
        return good.select(e, fx.Int32(E) + lane)

    def _load_own(arg_ids, arg_w, first, n_ent, tx, lane):
        """this WG's routes [first, first + RNG) -> [(slot, route, weight)] (loads only
        issued here; out of range -> 0 -> dummy slot)."""
        nb = _raw(fx.Int64(n_ent) * fx.Int64(4))
        rs_i = create_buffer_resource_from_addr(_raw(fx.Int64(arg_ids)),
                                                           num_records_bytes=nb)
        rs_w = create_buffer_resource_from_addr(_raw(fx.Int64(arg_w)),
                                                           num_records_bytes=nb)
        ent = []
        for j in range_constexpr(EPT):
            idx = first + fx.Int32(j * NT) + tx
            e = fx.Int32(buffer_load(rs_i, idx, vec_width=1, dtype=T.i32))
            w = fx.Float32(buffer_load(rs_w, idx, vec_width=1, dtype=T.f32))
            ent.append((e, idx, w))
        return [(_slot(e, idx < n_ent, lane), idx, w) for e, idx, w in ent]

    @flyc.jit
    def _hist(lds_i32, arg_ids, first, n_ent, tx, lane):
        """all n_ent routes: count expert e into bef[e] (route < first) or rest[e]."""
        rs = create_buffer_resource_from_addr(
            _raw(fx.Int64(arg_ids)), num_records_bytes=_raw(fx.Int64(n_ent) * fx.Int64(4)))
        per = NT * 4 * HU
        for it in range((n_ent + fx.Int32(per - 1)) // fx.Int32(per)):
            b0 = fx.Int32(it) * fx.Int32(per)
            vs = [Vec(buffer_load(
                rs, b0 + (fx.Int32(u * NT) + tx) * fx.Int32(4), vec_width=4, dtype=T.i32))
                for u in range_constexpr(HU)]
            for u in range_constexpr(HU):
                i0 = b0 + (fx.Int32(u * NT) + tx) * fx.Int32(4)
                for k in range_constexpr(4):
                    idx = i0 + fx.Int32(k)
                    arr = (idx < first).select(fx.Int32(L_BEF), fx.Int32(L_REST))
                    _latomic_add(lds_i32, arr + _slot(fx.Int32(vs[u][k]), idx < n_ent, lane), 1)

    def _prefix(lds, tx, lane, wave):
        """thread e < E: padded count of expert e -> exclusive scan -> off[e];
        off[E] = total; gcnt[e] = count. All threads (barrier inside)."""
        ec = fx.min(tx, fx.Int32(E - 1))
        c = (tx < fx.Int32(E)).select(lds[fx.Int32(L_BEF) + ec] + lds[fx.Int32(L_REST) + ec],
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
        _st_if(lds, fx.Int32(L_OFF) + ec, inc - pad, tx < fx.Int32(E))
        _st_if(lds, fx.Int32(L_GCNT) + ec, c, tx < fx.Int32(E))
        _st_if(lds, fx.Int32(L_OFF + E), inc, tx == fx.Int32(E - 1))

    @flyc.jit
    def _scatter(lds, sids, sw, e, pos, w, idx):
        if e < fx.Int32(E):
            dst = lds[fx.Int32(L_OFF) + e] + lds[fx.Int32(L_BEF) + e] + pos
            sids[dst] = (idx // fx.Int32(TOPK)) | ((idx % fx.Int32(TOPK)) << fx.Int32(24))
            sw[dst] = w

    def _scatter_own(lds_i32, lds, arg_sids, arg_sw, ent):
        """own routes: slot in the expert (LDS atomic) -> sorted row."""
        sids = global_typed_ptr(arg_sids, T.i32)
        sw = global_typed_ptr(arg_sw, T.f32)
        pos = [_latomic_add(lds_i32, fx.Int32(L_CNT) + s, 1) for s, _, _ in ent]
        for (s, idx, w), p in zip(ent, pos):
            _scatter(lds, sids, sw, s, p, w, idx)

    @flyc.jit
    def _pad_one(sids, sw, seids, e, o0, o1, c, lane, i32_m):
        """one wave, expert e (rows [o0, o1), c routes): padding rows, block ids."""
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
        """sort wave gw: experts gw, gw + nwv, ... (nwv = all sort waves)."""
        sids = global_typed_ptr(arg_sids, T.i32)
        sw = global_typed_ptr(arg_sw, T.f32)
        seids = global_typed_ptr(arg_seids, T.i32)
        gw = me * fx.Int32(NW) + wave
        nwv = n_sort * fx.Int32(NW)
        for k in range((fx.Int32(E) - gw + nwv - fx.Int32(1)) // nwv):
            e = gw + fx.Int32(k) * nwv
            o0 = _uni(lds[fx.Int32(L_OFF) + e])
            o1 = _uni(lds[fx.Int32(L_OFF + 1) + e])
            c = _uni(lds[fx.Int32(L_GCNT) + e])
            if o1 > o0:
                _pad_one(sids, sw, seids, e, o0, o1, c, lane, i32_m)

    @flyc.jit
    def _nv_store(lds, arg_nv, i32_m, me, tx):
        if me == fx.Int32(0):
            if tx == fx.Int32(0):
                nv = global_typed_ptr(arg_nv, T.i32)
                nv[fx.Int32(0)] = lds[fx.Int32(L_OFF + E)]
                nv[fx.Int32(1)] = i32_m

    @flyc.jit
    def _sort_role(lds_i32, lds, arg_ids, arg_w, arg_sids, arg_sw, arg_seids, arg_nv,
                   i32_m, n_sort, me, tx, lane, wave):
        n_ent = i32_m * fx.Int32(TOPK)
        first = me * fx.Int32(RNG)
        ent = _load_own(arg_ids, arg_w, first, n_ent, tx, lane)
        _init(lds, tx)
        gpu.barrier()
        for _ in range_constexpr(DO_HIST):
            _hist(lds_i32, arg_ids, first, n_ent, tx, lane)
        gpu.barrier()
        _prefix(lds, tx, lane, wave)
        gpu.barrier()
        for _ in range_constexpr(DO_SCAT):
            _scatter_own(lds_i32, lds, arg_sids, arg_sw, ent)
        for _ in range_constexpr(DO_PAD):
            _pad_all(lds, arg_sids, arg_sw, arg_seids, i32_m, me, n_sort, wave, lane)
        _nv_store(lds, arg_nv, i32_m, me, tx)

    @flyc.jit
    def _entry(arg_ids, arg_w, arg_x, arg_x8, arg_xs, arg_out, arg_sids, arg_sw,
               arg_seids, arg_nv, i32_m, i32_nsort):
        lds_ptr = fx.SharedAllocator().allocate(SharedStorage).peek().raw.ptr
        lds_i32 = fx.Int32(fx.ptrtoint(lds_ptr))
        lds = lds_typed_ptr(lds_i32, T.i32)
        tx = fx.Int32(gpu.thread_id("x"))
        bx = fx.Int32(gpu.block_id("x"))
        grid = fx.Int32(gpu.grid_dim.x)
        lane = tx % fx.Int32(64)
        wave = fx.Int32(rocdl.readfirstlane(T.i32, _raw(tx // fx.Int32(64))))
        n_sort = i32_nsort
        for _ in range_constexpr(DO_SORT):
            if bx < n_sort:
                _sort_role(lds_i32, lds, arg_ids, arg_w, arg_sids, arg_sw, arg_seids, arg_nv,
                           i32_m, n_sort, bx, tx, lane, wave)
        for _ in range_constexpr(DO_QUANT):
            if bx >= n_sort:
                _quant(arg_x, arg_x8, arg_xs, arg_out, i32_m,
                       (bx - n_sort) * fx.Int32(NW) + wave, (grid - n_sort) * fx.Int32(NW),
                       lane)

    @flyc.kernel(name=kname, known_block_size=[NT, 1, 1])
    def kernel(arg_ids: fx.Int64, arg_w: fx.Int64, arg_x: fx.Int64, arg_x8: fx.Int64,
               arg_xs: fx.Int64, arg_out: fx.Int64, arg_sids: fx.Int64, arg_sw: fx.Int64,
               arg_seids: fx.Int64, arg_nv: fx.Int64, i32_m: fx.Int32, i32_nsort: fx.Int32):
        _entry(arg_ids, arg_w, arg_x, arg_x8, arg_xs, arg_out, arg_sids, arg_sw, arg_seids,
               arg_nv, i32_m, i32_nsort)

    @flyc.jit
    def launch(arg_ids: fx.Int64, arg_w: fx.Int64, arg_x: fx.Int64, arg_x8: fx.Int64,
               arg_xs: fx.Int64, arg_out: fx.Int64, arg_sids: fx.Int64, arg_sw: fx.Int64,
               arg_seids: fx.Int64, arg_nv: fx.Int64, i32_m: fx.Int32, i32_nsort: fx.Int32,
               i32_grid: fx.Int32, stream: fx.Stream):
        kernel(arg_ids, arg_w, arg_x, arg_x8, arg_xs, arg_out, arg_sids, arg_sw, arg_seids,
               arg_nv, i32_m, i32_nsort).launch(grid=(fx.Int64(i32_grid), 1, 1),
                                                block=(NT, 1, 1), stream=stream)

    return launch


def _hist_unroll(n_ent):
    """16 B id loads per thread per histogram round trip: cover n_ent in one trip up to
    16 (64 KB); fewer at small batches (a partial trip still pays every lane's work)."""
    return min(16, max(1, -(-n_ent // (NT * 4))))


def _route_range(n_ent):
    """routes per sort WG: ~64 sort WGs (r256 at T = 1024 7.6 us vs r2048 11.9; r2048
    at T >= 16384, where more WGs mean more redundant histogram L2 reads), 256 .. 2048."""
    r = NT
    while r < 2048 and r * 64 < n_ent:
        r *= 2
    return r


def tp_prep(topk_ids, topk_weights, x, *, block_m, num_experts, zero=True, rng=None,
            tight=True, nsmin=16, hu=None, skip=""):
    """-> (sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, x8, xs, out):
    the moe_sorting contract over all ``num_experts`` experts, fp8 x + f32 row scales
    (as quantize_rows) and the bf16 output [M, H], zeroed if ``zero``. ``rng``: routes
    per sort WG (a multiple of 256; default by size); ``nsmin``: minimum sort WGs (padding parallelism);
    ``hu``: histogram loads per thread per round trip (default: by size)."""
    M, K = topk_ids.shape
    H = x.shape[1]
    E = num_experts
    dev = x.device
    rng = rng or _route_range(M * K)
    assert 1 <= M < (1 << 24) and rng % NT == 0
    nblk = (M * K) // block_m + (min(E, M * K) if tight else E)
    cap = nblk * block_m
    sids = torch.empty(cap, dtype=torch.int32, device=dev)
    sw = torch.empty(cap, dtype=torch.float32, device=dev)
    seids = torch.empty(nblk, dtype=torch.int32, device=dev)
    nv = torch.empty(2, dtype=torch.int32, device=dev)
    x8 = torch.empty((M, H), dtype=torch.float8_e4m3fnuz, device=dev)
    xs = torch.empty(M, dtype=torch.float32, device=dev)
    out = torch.empty((M, H), dtype=torch.bfloat16, device=dev)
    n_sort = max(-(-(M * K) // rng), nsmin, 1)
    n_quant = min(4 * NUM_CU, -(-M // NW))
    launch = compile_tp_prep(H=H, TOPK=K, E=E, BM=block_m, EPT=rng // NT,
                             HU=hu or _hist_unroll(M * K), ZERO=zero, SKIP=skip)
    _run_compiled(
        launch, topk_ids.data_ptr(), topk_weights.data_ptr(), x.data_ptr(), x8.data_ptr(),
        xs.data_ptr(), out.data_ptr(), sids.data_ptr(), sw.data_ptr(), seids.data_ptr(),
        nv.data_ptr(), int(M), int(n_sort), int(n_sort + n_quant),
        torch.cuda.current_stream())
    return sids, sw, seids, nv, x8, xs, out
