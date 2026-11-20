# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""megamoe large kernel (gfx942, FlyDSL): the EP8 a8w4 MoE layer for prefill-sized batches
as ONE kernel launch per call: x quant, the count exchange and the plan, the dispatch,
gemm1, gemm2 and the combine.

Design: aiter ``ops/flydsl/kernels/mega_moe/mega_moe_large_m.py`` (the MegaMoEV2
large-input path, gfx950), on the a8w4 ep tiles of ``_kloop._make_core`` (fd2 mxfp4 W
streamed into VGPRs, fp8 A through LDS; 128-row m-tiles, row variants 32 / 64 / 128):

- Roles (workgroup id; the grid is one workgroup per CU, all resident): 0 the planner,
  [1, P] the producers (producer p serves destination rank p % 8 and the local experts
  e = p // 8 (mod P / 8) there), the rest quantize x.
- Quant: one wave per token row: x bf16 -> fp8 + f32 scales (amax / 240): QX 1 one per
  (row, 128 cols), gemm1 rescales its accumulator at every 128-K group (as ``ep_mega``);
  QX 0 ONE per row, as ``a8w4_moe``'s ep3k (no rescaling: the row scale goes into the
  epilogue); COMB "add": the row of ``out`` (this rank's output in the symmetric heap) is
  zeroed. Then the quant count goes up.
- Producers, first: every route of this rank to one of their (destination, expert)
  segments goes into the segment's list (LDS counters: the route's rank in the segment);
  the segment sizes go into the destination's count matrix, then DONE_UNIT / (P / 8) into
  its ``cdone[me]``.
- Planner: waits for the count rows of all ranks, lays out its experts' rows (expert
  order, each expert padded to whole m-tiles; inside an expert the sources follow each
  other starting with this rank, so an expert's first m-tile holds local rows), writes
  each source rank's segment starts (``pbase``) into that rank, resets the m-tiles'
  arrival counts, publishes the m-tile table locally and ``pflag[me]`` = epoch on every
  rank.
- Producers, then: once the quant is done and the destination's ``pflag`` arrived, each
  wave sends 8 routes of a segment at a time (fp8 row, row scale, weight, source id,
  written through to the destination's rows), waits for the stores and adds the rows to
  the destination's ``tready`` of the (at most 2) m-tiles they fall in. Segments go in
  expert order, so the destinations' m-tiles fill up in about the order they are taken.
- Units, every workgroup after its role: gemm1 units (m-tile t, 128 inter cols j; t-major)
  and then gemm2 units (m-tile, NO out cols) are dealt to 8 shards (XCD x takes units u =
  x mod 8 from its own counter, then the other shards' leftovers): the m-tiles of one
  expert with the same j (the same W1 slab) run on one XCD. gemm1 waits until the m-tile's
  rows arrived; TR epilogue (act, fp8 a2 + one scale per (row, 128 cols), written
  through); then the m-tile's gemm1 count goes up. gemm2 waits for the I / 128 gemm1 units
  of its m-tile; the rows times their weights go back to the routes' source ranks: COMB
  "add": BF16 atomics into ``out[token]`` (as ``ep_mega``); COMB "fp8" (aiter's transport):
  fp8 + one f32 scale per 128 cols into ``cin[token, slot]`` (half the XGMI bytes; the
  atomics are bound by them at large batches).
- Finish: the last workgroup to finish (two-level count) sets ``fflag[me]`` = epoch on every
  rank and resets the local counters. COMB "add": it waits for every rank's ``fflag`` (all
  adds into this rank's ``out`` are done: when the kernel ends, ``out`` is complete).
  COMB "fp8": every workgroup waits for them, then workgroup b sums the slots of tokens b,
  b + NUM_CU, ... (f32) into ``out``.

Epoch: ``ctl[LC_EPOCH] + 1``, stored back by the last workgroup: the same on every rank as
long as every rank makes the same calls, right in CUDA graph replays. A rank starts call N + 1
only after every rank finished call N (``fflag``): a rank's buffers are written for the next
call only after their readers are done. Peers' data lives in the uncached MORI heap and is
written with system-scope stores; a flag or count is stored only after the data stores
completed (``s_waitcnt vmcnt(0)``): no L2 write-back or invalidation.

Needs the whole GPU: the grid (``NUM_CU`` workgroups) must be resident at once.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from pyhip.contrib.flydsl.host_launch import flydsl_kernel

from ._act import gate_up_act, situ_params
from ._atomic import (
    _atomic_add_at,
    _atomic_load_at,
    _inv_l1,
    _latomic_add,
    _sys_load_at,
    _sys_store_at,
    _xcc_id,
)
from ._cvt import _channel_factor, _fp8x4_pack
from ._hw import CM_NT, CM_SC1, CM_SYS, LDS_BYTES_MAX, NUM_CU, _opaque_tx, _u
from ._kloop import _dispatch_rows, _make_core
from ._mem import (
    _buffer_i32_scalar_read,
    _global_base_ptr1,
    _global_i32_at,
    _global_i32_buffer_tiles,
    _ld,
    _lds_f32_store_masked,
    _raw,
    _rsrc,
    _store,
    buffer_load,
    buffer_store,
    get_element_ptr,
    global_typed_ptr,
    lds_typed_ptr,
    lds_vec_load,
)
from ._protocol import (
    E_GLOBAL,
    E_LOCAL,
    LARGE_BM,
    LC_EPOCH,
    LC_HEAD,
    LC_PLAN,
    LC_QDONE,
    LC_SUB,
    LC_TE,
    LC_TOP,
    NPES,
    SMALL_LINE,
    large_ctl_ints,
    large_scap,
    large_sym_layout,
)

P2P = CM_SYS | CM_NT  # stores into a rank's (uncached) heap: written through
WT = CM_SC1  # x8 / a2: written through (read by another XCD)


def _lane_bcast(v, src_lane):
    """i32 ``v`` of lane ``src_lane`` (per lane) -> every lane (ds_bpermute)."""
    return fx.Int32(rocdl.ds_bpermute(T.i32, _raw(src_lane * fx.Int32(4)), _raw(fx.Int32(v))))


def _lane_value(v, k):
    return fx.Int32(rocdl.readlane(T.i32, _raw(fx.Int32(v)), _raw(fx.Int32(k))))


def _i64_of(lo, hi):
    return fx.Int64((fx.Uint64(fx.Uint32(hi)) << fx.Uint64(32)) | fx.Uint64(fx.Uint32(lo)))


def _wave_scan(v, lane):
    """Inclusive prefix sum of i32 ``v`` over the 64 lanes of a wave."""
    for bit in range_constexpr(6):
        d = 1 << bit
        other = _lane_bcast(v, fx.max(lane - fx.Int32(d), fx.Int32(0)))
        v = v + (lane >= fx.Int32(d)).select(other, fx.Int32(0))
    return v


@flydsl_kernel
def ep_mega_large_kernel(*, H, I, TOPK, TCAP, P=16, NO=256, ARB=128, PF=2, QX=1, COMB="add",
                         ACT="silu", BCM1=2, BCM2=2):
    """The large megamoe kernel for (H, I, top-k) and up to TCAP tokens per rank (a power of
    2). ``P``: producer workgroups (a multiple of 8); ``NO``: gemm2 output cols per unit;
    ``ARB`` / ``PF``: K bytes per A tile / W prefetch distance of both GEMMs; ``QX``: x
    scales, 1 one per (row, 128 cols) (gemm1 rescales its accumulator at every 128-K group,
    as ep_mega), 0 one per token row; ``COMB``: "add" (BF16 atomics into the source rank's
    ``out``) or "fp8" (fp8 rows + scales into its ``cin``, summed there); ``BCM1`` /
    ``BCM2``: gemm1 / gemm2 W load cache modifier (2 = nt)."""
    assert ACT in ("silu", "silu_clamp", "situv2"), ACT
    assert QX in (0, 1) and COMB in ("add", "fp8"), (QX, COMB)
    FP8C = COMB == "fp8"
    NW = 8
    NT = 64 * NW
    NI = 128
    N2 = 2 * I
    BM = LARGE_BM
    NG, ND = I // NI, H // NO  # gemm1 / gemm2 units per m-tile
    G2N = I // 128  # 128-K scale groups of gemm2
    G1N = H // 128  # of gemm1 (QX 1)
    XSN = G1N if QX else 1  # x scales per row
    NIW, NOW = NI // NW, NO // NW
    NC1, NC2 = NIW // 16, NOW // 16
    GPW, WPG = NI // 128, NW // (NI // 128)  # gemm1 epilogue: groups per tile, waves per group
    GRID = NUM_CU
    KP = P // NPES  # producers per destination
    EPP = E_LOCAL // KP  # (destination, expert) segments per producer
    NQ = GRID - 1 - P  # quant workgroups
    SCAP = large_scap(TOPK, TCAP)
    MAXT = SCAP // BM
    TSH = TCAP.bit_length() - 1  # token bits of a source id
    TOPK_SHIFT = TOPK.bit_length() - 1
    TOKEN_MASK = TCAP - 1
    CH = 8  # routes per producer wave between two waits
    CQ = H // 512  # quant: 16 B (8 bf16) per lane per 512 columns
    LINE = SMALL_LINE
    DONE_UNIT = 64
    ROWV = (32, 64, 128)
    RPS = NT // 32  # gemm2 scatter rows per pass (32 lanes x 2 cols per row)
    SCAN_B = 8  # producer scan: route ids per thread with their loads issued together
    off, _ = large_sym_layout(H, TOPK, TCAP)
    O_RX, O_RSX, O_RW, O_RID = off["rx"], off["rsx"], off["rw"], off["rid"]
    O_CNT, O_PBASE, O_CDONE = off["cnt"], off["pbase"], off["cdone"]
    O_PFLAG, O_FFLAG, O_TREADY, O_OUT = off["pflag"], off["fflag"], off["tready"], off["out"]
    O_CIN, O_CINS = off["cin"], off["cins"]
    LPR = NO // 8  # COMB fp8, gemm2 row stores: lanes per row (8 cols each)
    SRP = NT // LPR  # rows per pass
    assert not FP8C or (H == 8 * NT and NT % LPR == 0 and NO % 128 == 0), (
        "fp8 combine: a row is 8 cols per thread, 16 lanes a 128-col group")
    LC_G1 = large_ctl_ints(TOPK, TCAP)[0]
    assert 1 << TOPK_SHIFT == TOPK and TOPK <= 8 and TCAP == 1 << TSH and TSH + 3 <= 24, (
        TOPK, TCAP)
    assert P % NPES == 0 and KP & (KP - 1) == 0 and E_LOCAL % KP == 0 and DONE_UNIT % KP == 0
    assert 0 < P < GRID - 1 and EPP <= 64, P
    assert H % 1024 == 0 and I % NI == 0 and H % NO == 0 and NO % 64 == 0 and NOW % 16 == 0
    assert NIW % 16 == 0 and 128 % NIW == 0 and XSN <= 64
    RATIO = 2 * BM * ARB  # bytes of the A double buffer; then gemm1's row scales [BM] f32
    L_XS = RATIO  # (QX 1: ratio tables [G1N][BM] + fin [BM]), gemm2's [G2N][BM] + fin [BM]
    # LDS words (outside the units): [0, 4) broadcast; the planner's count matrix [256],
    # segment starts [256], first m-tile [33] and rows [32] of each expert; a producer's
    # segment sizes [EPP], starts [EPP], chunk prefix [EPP + 1], routes dropped [1]
    L_BC = 0
    L_PC = 16
    L_PB = L_PC + NPES * E_LOCAL
    L_TF = L_PB + NPES * E_LOCAL
    L_TOT = L_TF + E_LOCAL + 1
    L_CNTP = 16
    L_PBS = L_CNTP + 64
    L_CHP = L_PBS + 64
    LDS_BYTES = max(RATIO + (G1N + 1 if QX else 1) * BM * 4, RATIO + (G2N + 1) * BM * 4,
                    BM * NO * 2, BM * NW * 4, (L_TOT + E_LOCAL) * 4, (L_CHP + 65) * 4)
    assert LDS_BYTES <= LDS_BYTES_MAX, LDS_BYTES
    kname = (f"a8w4_megamoe_large_h{H}_i{I}_k{TOPK}_t{TCAP}_p{P}_no{NO}_a{ARB}_pf{PF}"
             + ("" if QX else "_xrow") + ("_cfp8" if FP8C else "")
             + ("" if (BCM1, BCM2) == (2, 2) else f"_bcm{BCM1}{BCM2}")
             + ("" if ACT == "silu" else f"_{ACT}"))

    @fx.struct
    class SharedStorage:
        raw: fx.Array[fx.Uint8, LDS_BYTES, 16]

    NAMES64 = ("x", "ids", "tw", "sym", "stab", "ctl", "x8", "xs", "lists", "a2", "a2s",
               "w1", "w1s", "w1e", "w2", "w2s", "w2e")
    NAMES32 = ("n", "me")
    NAMESF = ("sb", "sbr", "lb", "lbr", "lim")
    N64 = len(NAMES64)

    @flyc.kernel(name=kname, known_block_size=[NT, 1, 1])
    def kernel(x: fx.Int64, ids: fx.Int64, tw: fx.Int64, sym: fx.Int64, stab: fx.Int64,
               ctl: fx.Int64, x8: fx.Int64, xs: fx.Int64, lists: fx.Int64, a2: fx.Int64,
               a2s: fx.Int64, w1: fx.Int64, w1s: fx.Int64, w1e: fx.Int64, w2: fx.Int64,
               w2s: fx.Int64, w2e: fx.Int64, n: fx.Int32, me: fx.Int32, sb: fx.Float32,
               sbr: fx.Float32, lb: fx.Float32, lbr: fx.Float32, lim: fx.Float32):
        lds_ptr = fx.SharedAllocator().allocate(SharedStorage).peek().raw.ptr
        lds_i32 = fx.Int32(fx.ptrtoint(lds_ptr))
        lds = lds_typed_ptr(lds_i32, T.i32)
        lds_f32 = lds_typed_ptr(lds_i32, T.f32)
        tx0 = fx.Int32(gpu.thread_id("x"))
        lane0 = tx0 % fx.Int32(64)
        wave0 = _u(tx0 // fx.Int32(64))
        bid = fx.Int32(gpu.block_id("x"))
        ctl0 = fx.Int64(ctl)
        sym0 = fx.Int64(sym)
        epoch = _u(_sys_load_at(ctl0 + fx.Int64(LC_EPOCH * 4))) + fx.Int32(1)

        def ctl_at(i):
            return ctl0 + fx.Int64(fx.Int32(i) * fx.Int32(4))

        def lane0_add(addr, value):
            """Atomic add by lane 0 only (old value in lane 0). The address goes through an
            opaque VGPR zero: at a uniform address LLVM's atomic optimizer would sum the
            lanes in a 64-step loop."""
            taken = fx.Int32(0)
            if lane0 == fx.Int32(0):
                zero = fx.Int32(llvm.inline_asm(T.i32, [], "v_mov_b32 $0, 0", "=v",
                                                has_side_effects=True))
                taken = _atomic_add_at(fx.Int64(addr) + fx.Int64(zero), value, "agent-one-as")
            return taken

        def to_all(v, slot):
            """wave 0's uniform ``v`` -> every wave (LDS word ``slot``)."""
            if wave0 == fx.Int32(0):
                _store(lds, fx.Int32(L_BC + slot), v)
            rocdl.s_waitcnt(lgkmcnt=0)
            gpu.barrier()
            r = _u(lds[fx.Int32(L_BC + slot)])
            rocdl.s_waitcnt(lgkmcnt=0)
            gpu.barrier()
            return r

        def wait_flags(base, target):
            """wave 0: until the NPES i32 words at base + s * LINE all hold ``target``."""
            if wave0 == fx.Int32(0):
                addr = base + fx.Int64(fx.min(lane0, fx.Int32(NPES - 1)) * fx.Int32(LINE))
                seen = _sys_load_at(addr)
                pending = fx.Int64(rocdl.ballot(T.i64, _raw(seen != target))) != fx.Int64(0)
                while pending:
                    rocdl.s_sleep(2)
                    seen = _sys_load_at(addr)
                    pending = fx.Int64(rocdl.ballot(T.i64, _raw(seen != target))) != fx.Int64(0)
                _inv_l1()
            gpu.barrier()

        def spin_ge(addr, target, scope):
            """wave 0: poll until the i32 at addr >= target, then invalidate the L1."""
            seen = _atomic_load_at(addr, scope)
            while seen < target:
                rocdl.s_sleep(2)
                seen = _atomic_load_at(addr, scope)
            _inv_l1()

        def reload():
            """Kernel args read again from the kernarg segment through an opaque base: with
            loop-invariant args LLVM hoists every unit kind's address math out of the unit
            loop and runs out of registers (as ep_mega)."""
            ptr4 = ir.Type.parse("!llvm.ptr<4>")
            seg = llvm.call_intrinsic(ptr4, "llvm.amdgcn.kernarg.segment.ptr", [], [], [])
            base = fx.Int64(llvm.inline_asm(T.i64, [llvm.ptrtoint(T.i64, seg)], "s_mov_b64 $0, $1",
                                            "=s,s", has_side_effects=True))

            def load(off_, ty, align):
                return llvm.LoadOp(ty, llvm.inttoptr(ptr4, (base + fx.Int64(off_)).ir_value()),
                                   alignment=align).result

            a = {k: fx.Int64(load(8 * i, T.i64, 8)) for i, k in enumerate(NAMES64)}
            a.update({k: fx.Int32(load(8 * N64 + 4 * j, T.i32, 4)) for j, k in enumerate(NAMES32)})
            a.update({k: fx.Float32(load(8 * N64 + 4 * (len(NAMES32) + j), T.f32, 4))
                      for j, k in enumerate(NAMESF)})
            return a

        def peer(a, rank):
            """rank's heap base (per lane)."""
            return fx.Int64(global_typed_ptr(a["stab"], T.i64, align=8)[rank])

        def peer_u(a, rank):
            """rank's heap base, a wave-uniform ``rank``: in SGPRs."""
            tab = _rsrc(a["stab"], NPES * 8)
            return _i64_of(_u(fx.Int32(_ld(tab, rank * fx.Int32(2)))),
                           _u(fx.Int32(_ld(tab, rank * fx.Int32(2) + fx.Int32(1)))))

        # -------------------------------------------------------------- quant
        def quant(a, qi):
            """Wave w of quant workgroup qi: tokens qi NW + w, + NQ NW, ...: the bf16 row ->
            fp8 + f32 scales (amax / 240, 1 for all zeros; one per row, QX 1: per 128 cols),
            written through; COMB "add": the row of ``out`` zeroed."""
            tx, lane, wave = _opaque_tx(tx0)
            n_ = a["n"]
            x_rs = _rsrc(a["x"], fx.Int64(n_) * fx.Int64(H * 2))
            x8_rs = _rsrc(a["x8"], fx.Int64(n_) * fx.Int64(H))
            xs_rs = _rsrc(a["xs"], fx.Int64(n_) * fx.Int64(XSN * 4))
            out_rs = _rsrc(a["sym"] + fx.Int64(O_OUT), fx.Int64(n_) * fx.Int64(H * 2))
            zero4 = Vec.filled(4, 0, fx.Int32)
            t = qi * fx.Int32(NW) + wave
            while t < n_:
                vals = []
                for c in range_constexpr(CQ):
                    raw = Vec(_ld(x_rs, t * fx.Int32(H // 2) + fx.Int32(c * 256)
                                  + lane * fx.Int32(4), 4))
                    f = Vec(raw.bitcast(fx.BFloat16)).to(fx.Float32)
                    vals.append([fx.Float32(f[i]) for i in range(8)])
                ams = []
                for v8 in vals:
                    am = fx.Float32(0.0)
                    for v in v8:
                        am = fx.maxnumf(am, fx.absf(v))
                    ams.append(am)
                if const_expr(QX):  # 16 lanes = one 128-col group of chunk c
                    for c in range_constexpr(CQ):
                        for s in range_constexpr(4):
                            ams[c] = fx.maxnumf(ams[c], ams[c].shuffle_xor(1 << s, 64))
                else:
                    am = ams[0]
                    for c in range_constexpr(1, CQ):
                        am = fx.maxnumf(am, ams[c])
                    for s in range_constexpr(6):
                        am = fx.maxnumf(am, am.shuffle_xor(1 << s, 64))
                    ams = [am] * CQ
                scs = [(am == fx.Float32(0.0)).select(fx.Float32(1.0), am / fx.Float32(240.0))
                       for am in ams]
                for c in range_constexpr(CQ):
                    inv = fx.Float32(1.0) / scs[c]
                    q = Vec.from_elements(
                        [_fp8x4_pack(*[vals[c][4 * h + i] * inv for i in range(4)])
                         for h in range(2)], fx.Int32)
                    buffer_store(q, x8_rs, t * fx.Int32(H // 4) + fx.Int32(c * 128)
                                 + lane * fx.Int32(2), cache_modifier=WT)
                    if const_expr(QX):
                        buffer_store(scs[c], xs_rs, t * fx.Int32(G1N) + fx.Int32(c * 4)
                                     + lane // fx.Int32(16),
                                     mask=lane % fx.Int32(16) == fx.Int32(0), cache_modifier=WT)
                if const_expr(not QX):
                    buffer_store(scs[0], xs_rs, t, mask=lane == fx.Int32(0), cache_modifier=WT)
                if const_expr(not FP8C):
                    for c in range_constexpr(CQ):
                        buffer_store(zero4, out_rs, t * fx.Int32(H // 2) + fx.Int32(c * 256)
                                     + lane * fx.Int32(4))
                t = t + fx.Int32(NQ * NW)
            rocdl.s_waitcnt(vmcnt=0)  # x8 / xs written through, out zeroed
            gpu.barrier()
            if wave0 == fx.Int32(0):
                lane0_add(ctl_at(LC_QDONE), 1)

        # -------------------------------------------------------------- producers
        def scan(a, p):
            """Producer p: the routes to its segments -> lists[g] (rank in the segment from
            an LDS counter); segment sizes -> the destination's count matrix, then its
            cdone[me] += DONE_UNIT / KP."""
            d = p % fx.Int32(NPES)
            k = p // fx.Int32(NPES)
            me_ = a["me"]
            if tx0 < fx.Int32(EPP):
                _store(lds, fx.Int32(L_CNTP) + tx0, fx.Int32(0))
            rocdl.s_waitcnt(lgkmcnt=0)
            gpu.barrier()
            nr = a["n"] * fx.Int32(TOPK)
            ids_rs = _rsrc(a["ids"], fx.Int64(nr) * fx.Int64(4))
            lists_p = global_typed_ptr(a["lists"], T.i32)
            for r0 in range(fx.Int32(0), nr, SCAN_B * NT):
                gs = [fx.Int32(_ld(ids_rs, r0 + fx.Int32(j * NT) + tx0,
                                   mask=r0 + fx.Int32(j * NT) + tx0 < nr))
                      for j in range(SCAN_B)]
                for j in range_constexpr(SCAN_B):
                    r = r0 + fx.Int32(j * NT) + tx0
                    g = gs[j]
                    el = g & fx.Int32(E_LOCAL - 1)
                    mine = ((r < nr) & (g >= fx.Int32(0)) & (g < fx.Int32(E_GLOBAL))
                            & ((g >> fx.Int32(5)) == d) & ((el & fx.Int32(KP - 1)) == k))
                    if mine:
                        pos = _latomic_add(lds_i32, fx.Int32(L_CNTP) + el // fx.Int32(KP), 1)
                        _store(lists_p, g * fx.Int32(TCAP) + pos, r)
            rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0)
            gpu.barrier()
            if tx0 < fx.Int32(EPP):
                e = k + fx.Int32(KP) * tx0
                _sys_store_at(peer(a, d) + fx.Int64(O_CNT)
                              + fx.Int64((me_ * fx.Int32(E_LOCAL) + e) * fx.Int32(4)),
                              lds[fx.Int32(L_CNTP) + tx0])
            rocdl.s_waitcnt(vmcnt=0)  # the counts landed
            gpu.barrier()
            if tx0 == fx.Int32(0):
                _atomic_add_at(peer(a, d) + fx.Int64(O_CDONE) + fx.Int64(me_ * fx.Int32(LINE)),
                               DONE_UNIT // KP, "one-as")

        def send(a, p):
            """Producer p, once the quant is done and its destination's plan is here: wave w
            sends chunks w, w + NW, ... of its segments (expert order), CH routes each."""
            tx, lane, wave = _opaque_tx(tx0)
            d = p % fx.Int32(NPES)
            k = p // fx.Int32(NPES)
            me_ = a["me"]
            if wave == fx.Int32(0):
                spin_ge(ctl_at(LC_QDONE), fx.Int32(NQ), "agent-one-as")
                f = _sys_load_at(sym0 + fx.Int64(O_PFLAG) + fx.Int64(d * fx.Int32(LINE)))
                while f != epoch:
                    rocdl.s_sleep(2)
                    f = _sys_load_at(sym0 + fx.Int64(O_PFLAG) + fx.Int64(d * fx.Int32(LINE)))
                _inv_l1()
                j = fx.min(lane, fx.Int32(EPP - 1))
                e = k + fx.Int32(KP) * j
                c = (lane < fx.Int32(EPP)).select(lds[fx.Int32(L_CNTP) + j], fx.Int32(0))
                b = _sys_load_at(sym0 + fx.Int64(O_PBASE)
                                 + fx.Int64((d * fx.Int32(E_LOCAL) + e) * fx.Int32(4)))
                nch = (c + fx.Int32(CH - 1)) // fx.Int32(CH)
                inc = _wave_scan(nch, lane)
                if lane < fx.Int32(EPP):
                    _store(lds, fx.Int32(L_PBS) + j, b)
                    _store(lds, fx.Int32(L_CHP) + j, inc - nch)
                over = rocdl.ballot(T.i64, _raw((lane < fx.Int32(EPP)) & (b + c > fx.Int32(SCAP))))
                if lane == fx.Int32(EPP - 1):
                    _store(lds, fx.Int32(L_CHP + EPP), inc)
                    _store(lds, fx.Int32(L_CHP + EPP + 1),
                           (fx.Int64(over) != fx.Int64(0)).select(fx.Int32(1), fx.Int32(0)))
            rocdl.s_waitcnt(lgkmcnt=0)
            gpu.barrier()
            total = _u(lds[fx.Int32(L_CHP + EPP)])
            nr = a["n"] * fx.Int32(TOPK)
            dbase = peer_u(a, d)
            rx_rs = _rsrc(dbase + fx.Int64(O_RX), SCAP * H)
            rsx_rs = _rsrc(dbase + fx.Int64(O_RSX), SCAP * XSN * 4)
            rw_rs = _rsrc(dbase + fx.Int64(O_RW), SCAP * 4)
            rid_rs = _rsrc(dbase + fx.Int64(O_RID), SCAP * 4)
            x8_rs = _rsrc(a["x8"], fx.Int64(a["n"]) * fx.Int64(H))
            xs_rs = _rsrc(a["xs"], fx.Int64(a["n"]) * fx.Int64(XSN * 4))
            w_rs = _rsrc(a["tw"], fx.Int64(nr) * fx.Int64(4))
            lists_rs = _rsrc(a["lists"], E_GLOBAL * TCAP * 4)
            q = wave
            while q < total:
                j = fx.Int32(0)
                for jj in range_constexpr(1, EPP):
                    j = j + (lds[fx.Int32(L_CHP + jj)] <= q).select(fx.Int32(1), fx.Int32(0))
                j = _u(j)
                c0 = _u(lds[fx.Int32(L_CHP) + j])
                cnt = _u(lds[fx.Int32(L_CNTP) + j])
                base = _u(lds[fx.Int32(L_PBS) + j])
                g = d * fx.Int32(E_LOCAL) + k + fx.Int32(KP) * j
                i0 = (q - c0) * fx.Int32(CH)
                m = fx.min(cnt - i0, fx.Int32(CH))
                rts = fx.Int32(_ld(lists_rs, g * fx.Int32(TCAP) + i0 + lane, mask=lane < m))
                # all loads of the chunk before any store (a store may alias a later load
                # for all LLVM knows)
                pending = []
                for jj in range_constexpr(CH):
                    rt = _lane_value(rts, jj)
                    row = base + i0 + fx.Int32(jj)
                    ok = (fx.Int32(jj) < m) & (row < fx.Int32(SCAP))
                    token = rt >> fx.Int32(TOPK_SHIFT)
                    data = [_ld(x8_rs, token * fx.Int32(H // 4) + fx.Int32(part * 256)
                                + lane * fx.Int32(4), 4, ok) for part in range_constexpr(H // 1024)]
                    sc = _ld(xs_rs, token * fx.Int32(XSN) + (lane if XSN > 1 else fx.Int32(0)), 1,
                             ok & (lane < fx.Int32(XSN)))
                    wt = _ld(w_rs, rt, 1, ok)
                    pending.append((rt, row, ok, token, data, sc, wt))
                for rt, row, ok, token, data, sc, wt in pending:
                    for part in range_constexpr(H // 1024):
                        buffer_store(data[part], rx_rs, row * fx.Int32(H // 4)
                                     + fx.Int32(part * 256) + lane * fx.Int32(4), mask=ok,
                                     cache_modifier=P2P)
                    first = ok & (lane == fx.Int32(0))
                    sid = (((rt & fx.Int32(TOPK - 1)) << fx.Int32(24)) | (me_ << fx.Int32(TSH))
                           | token)
                    buffer_store(sc, rsx_rs,
                                 row * fx.Int32(XSN) + (lane if XSN > 1 else fx.Int32(0)),
                                 mask=ok & (lane < fx.Int32(XSN)), cache_modifier=P2P)
                    buffer_store(wt, rw_rs, row, mask=first, cache_modifier=P2P)
                    buffer_store(sid, rid_rs, row, mask=first, cache_modifier=P2P)
                rocdl.s_waitcnt(vmcnt=0)  # the chunk's rows landed
                r0 = base + i0
                r1 = fx.min(r0 + m, fx.Int32(SCAP))
                t0 = r0 // fx.Int32(BM)
                e0 = (t0 + fx.Int32(1)) * fx.Int32(BM)
                n0 = fx.min(r1, e0) - r0
                n1 = r1 - e0
                if (lane == fx.Int32(0)) & (n0 > fx.Int32(0)):
                    _atomic_add_at(dbase + fx.Int64(O_TREADY) + fx.Int64(t0 * fx.Int32(4)), n0,
                                   "one-as")
                if (lane == fx.Int32(1)) & (n1 > fx.Int32(0)):
                    _atomic_add_at(dbase + fx.Int64(O_TREADY) + fx.Int64((t0 + fx.Int32(1))
                                                                         * fx.Int32(4)), n1,
                                   "one-as")
                q = q + fx.Int32(NW)
            if const_expr(FP8C):
                if _u(lds[fx.Int32(L_CHP + EPP + 1)]) != fx.Int32(0):
                    drop_slots(a, d, k, lists_rs, wave, lane)

        def drop_slots(a, d, k, lists_rs, wave, lane):
            """COMB "fp8", producer (d, k) with routes past the destination's s_cap (not
            sent): their slots of this rank's ``cin`` -> 0 (the combine adds them)."""
            cin_rs = _rsrc(a["sym"] + fx.Int64(O_CIN), TCAP * TOPK * H)
            cins_rs = _rsrc(a["sym"] + fx.Int64(O_CINS), TCAP * TOPK * (H // 128) * 4)
            zero4 = Vec.filled(4, 0, fx.Int32)
            for j in range(fx.Int32(0), fx.Int32(EPP), fx.Int32(1)):
                g = d * fx.Int32(E_LOCAL) + k + fx.Int32(KP) * j
                cnt = _u(lds[fx.Int32(L_CNTP) + j])
                i = fx.max(fx.Int32(SCAP) - _u(lds[fx.Int32(L_PBS) + j]), fx.Int32(0)) + wave
                while i < cnt:
                    rt = fx.Int32(_ld(lists_rs, g * fx.Int32(TCAP) + i))
                    for part in range_constexpr(H // 1024):
                        buffer_store(zero4, cin_rs, rt * fx.Int32(H // 4) + fx.Int32(part * 256)
                                     + lane * fx.Int32(4), cache_modifier=P2P)
                    buffer_store(fx.Float32(0.0), cins_rs, rt * fx.Int32(H // 128) + lane,
                                 mask=lane < fx.Int32(H // 128), cache_modifier=P2P)
                    i = i + fx.Int32(NW)

        # -------------------------------------------------------------- planner
        def planner(a):
            """Workgroup 0: the count rows of all ranks -> this rank's row layout, the
            sources' segment starts, the m-tile table; then the flags."""
            me_ = a["me"]
            wait_flags(sym0 + fx.Int64(O_CDONE), epoch * fx.Int32(DONE_UNIT))
            if tx0 < fx.Int32(NPES * E_LOCAL):
                _store(lds, fx.Int32(L_PC) + tx0,
                       _sys_load_at(sym0 + fx.Int64(O_CNT) + fx.Int64(tx0 * fx.Int32(4))))
            rocdl.s_waitcnt(lgkmcnt=0)
            gpu.barrier()
            if wave0 == fx.Int32(0):
                e = fx.min(lane0, fx.Int32(E_LOCAL - 1))
                tot = fx.Int32(0)
                pres = []
                for kk in range_constexpr(NPES):  # sources from this rank on
                    s = (me_ + fx.Int32(kk)) & fx.Int32(NPES - 1)
                    pres.append(tot)
                    tot = tot + lds[fx.Int32(L_PC) + s * fx.Int32(E_LOCAL) + e]
                tot = (lane0 < fx.Int32(E_LOCAL)).select(tot, fx.Int32(0))
                nt = (tot + fx.Int32(BM - 1)) // fx.Int32(BM)
                inc = _wave_scan(nt, lane0)
                first = inc - nt
                if lane0 < fx.Int32(E_LOCAL):
                    _store(lds, fx.Int32(L_TF) + e, first)
                    _store(lds, fx.Int32(L_TOT) + e, tot)
                    for kk in range_constexpr(NPES):
                        s = (me_ + fx.Int32(kk)) & fx.Int32(NPES - 1)
                        _store(lds, fx.Int32(L_PB) + s * fx.Int32(E_LOCAL) + e,
                               first * fx.Int32(BM) + pres[kk])
                if lane0 == fx.Int32(E_LOCAL - 1):
                    _store(lds, fx.Int32(L_TF + E_LOCAL), inc)
            rocdl.s_waitcnt(lgkmcnt=0)
            gpu.barrier()
            all_t = _u(lds[fx.Int32(L_TF + E_LOCAL)])
            tiles = fx.min(all_t, fx.Int32(MAXT))
            if tx0 < fx.Int32(NPES * E_LOCAL):
                s = tx0 // fx.Int32(E_LOCAL)
                e = tx0 % fx.Int32(E_LOCAL)
                _sys_store_at(peer(a, s) + fx.Int64(O_PBASE)
                              + fx.Int64((me_ * fx.Int32(E_LOCAL) + e) * fx.Int32(4)),
                              lds[fx.Int32(L_PB) + tx0])
            for t in range(tx0, tiles, NT):
                _sys_store_at(sym0 + fx.Int64(O_TREADY) + fx.Int64(t * fx.Int32(4)), fx.Int32(0))
                e = fx.Int32(0)
                for kk in range_constexpr(1, E_LOCAL):
                    e = e + (lds[fx.Int32(L_TF + kk)] <= t).select(fx.Int32(1), fx.Int32(0))
                rows = fx.min(lds[fx.Int32(L_TOT) + e] - (t - lds[fx.Int32(L_TF) + e])
                              * fx.Int32(BM), fx.Int32(BM))
                _sys_store_at(ctl_at(fx.Int32(LC_TE) + t), e | (rows << fx.Int32(8)))
            rocdl.s_waitcnt(vmcnt=0)  # segment starts, tready resets, m-tile table landed
            gpu.barrier()
            if tx0 < fx.Int32(NPES):
                _sys_store_at(peer(a, tx0) + fx.Int64(O_PFLAG) + fx.Int64(me_ * fx.Int32(LINE)),
                              epoch)
            if tx0 == fx.Int32(0):
                _sys_store_at(ctl_at(LC_PLAN + 1), tiles)
                _sys_store_at(ctl_at(LC_PLAN + 2), all_t * fx.Int32(BM))
                rocdl.s_waitcnt(vmcnt=0)
                _sys_store_at(ctl_at(LC_PLAN), epoch)

        # -------------------------------------------------------------- gemm1
        def gemm1_unit(a, t, nb, tx, lane, wave):
            lane_div_16 = lane // fx.Int32(16)
            lane_mod_16 = lane % fx.Int32(16)
            te = _u(_sys_load_at(ctl_at(fx.Int32(LC_TE) + t)))
            e = te & fx.Int32(255)
            n_valid = te >> fx.Int32(8)
            row0 = t * fx.Int32(BM)
            col0 = nb * fx.Int32(NI) + wave * fx.Int32(NIW)
            cols = []
            for gu in range_constexpr(2):
                for c in range_constexpr(NC1):
                    n0 = e * fx.Int32(N2) + fx.Int32(gu * I) + col0 + fx.Int32(c * 16)
                    cols.append(((gu, c), n0, (gu, c // 2)))
            exps = {
                (gu, c, v): fx.Int32(_global_i32_at(
                    a["w1e"], e * fx.Int32(N2) + fx.Int32(gu * I) + col0 + fx.Int32(c * 16)
                    + lane_div_16 * fx.Int32(4) + fx.Int32(v)))
                for gu in range(2) for c in range(NC1) for v in range(4)
            }
            core = _make_core(
                K=H, W_ROWS=E_LOCAL * N2, BM=BM, PF=PF, bcm=BCM1, cols=cols,
                lds_ptr=lds_ptr, LDS_BYTES=LDS_BYTES, lane=lane, tx=tx, NT=NT, arg_w=a["w1"],
                arg_ws=a["w1s"], arg_a=a["sym"] + fx.Int64(O_RX), a_num_bytes=SCAP * H,
                ARB=ARB, acm=2, SCH=3, TR=True)

            def row_base16(row):  # rows past the m-tile's valid ones read past the end: 0
                return (row < n_valid).select((row0 + row) * fx.Int32(H // 16),
                                              fx.Int32(SCAP * (H // 16)))

            # LDS f32 index of row r's x scale for the epilogue: QX 0 the row scale, QX 1
            # the last group's (fin, after the ratio tables)
            L_FIN = L_XS // 4 + (G1N * BM if QX else 0)

            def build_ratios(R):
                """QX 1, LDS: ratio[g][r] = s[r][g-1] / s[r][g] (ratio[0] = 1), fin[r] =
                s[r][G-1]; a zero scale (padding row) acts as 1."""
                xs_tiles = _global_i32_buffer_tiles(a["sym"] + fx.Int64(O_RSX), SCAP * G1N * 4, 1)
                xs_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Int32)
                one = fx.Float32(1.0)
                for pp in range_constexpr(-(-(R * G1N) // NT)):
                    idx = tx + fx.Int32(pp * NT)
                    r = idx // fx.Int32(G1N)
                    g = idx % fx.Int32(G1N)
                    ok = r < fx.Int32(R)
                    rr = ok.select(r, fx.Int32(0))
                    srow = (rr < n_valid).select((row0 + rr) * fx.Int32(G1N),
                                                 fx.Int32(SCAP * G1N))
                    cur = _buffer_i32_scalar_read(xs_tiles, srow + g, xs_atom).bitcast(fx.Float32)
                    prv = _buffer_i32_scalar_read(
                        xs_tiles, srow + fx.max(g - fx.Int32(1), fx.Int32(0)), xs_atom
                    ).bitcast(fx.Float32)
                    cur = (cur == fx.Float32(0.0)).select(one, cur)
                    prv = (prv == fx.Float32(0.0)).select(one, prv)
                    ratio = (g == fx.Int32(0)).select(one, prv / cur)
                    _lds_f32_store_masked(lds_f32, fx.Int32(L_XS // 4) + g * fx.Int32(BM) + rr,
                                          ratio, ok)
                    _lds_f32_store_masked(lds_f32, fx.Int32(L_FIN) + rr, cur,
                                          ok & (g == fx.Int32(G1N - 1)))

            def hook(acc, kt, s):
                kk = s * 32  # K offset of MFMA step s in the ARB-K tile
                if const_expr(kk % 128 != 0):
                    return
                gidx = kt * fx.Int32(ARB // 128) + fx.Int32(kk // 128)
                for key in acc:
                    for mi in range_constexpr(len(acc[key])):  # TR: lane = row mi*16 + L%16
                        rt = lds_f32[fx.Int32(L_XS // 4 + mi * 16) + gidx * fx.Int32(BM)
                                     + lane_mod_16]
                        a4 = Vec(acc[key][mi].load())
                        acc[key][mi].store(Vec.from_elements([a4[v] * rt for v in range(4)],
                                                             fx.Float32))

            def epilogue(R, acc):
                """dequant -> act(gate)*up in registers -> row amax over the lane's columns,
                the 4 lane groups and the NW waves (LDS [row][NW]) -> fp8 a2 + f32 scale per
                (row, 128 cols), written through."""
                MR = R // 16
                rows = [fx.Int32(mi * 16) + lane_mod_16 for mi in range(MR)]
                valid = [r < n_valid for r in rows]
                fac = {k: _channel_factor(v) for k, v in exps.items()}
                rsc = [fx.Float32(lds_f32[fx.Int32(L_FIN) + r]) for r in rows]
                situ = None
                if const_expr(ACT != "silu"):
                    situ = situ_params(*[fx.Float32(a[k]) for k in NAMESF])
                ys = {}
                for mi in range_constexpr(MR):
                    for c in range_constexpr(NC1):
                        g4 = Vec(fx.memref_load_vec(acc[0, c][mi]))
                        u4 = Vec(fx.memref_load_vec(acc[1, c][mi]))
                        gv = [fx.Float32(g4[v]) * (rsc[mi] * fac[0, c, v]) for v in range(4)]
                        uv = [fx.Float32(u4[v]) * (rsc[mi] * fac[1, c, v]) for v in range(4)]
                        ys[mi, c] = gate_up_act(ACT, gv, uv, situ)
                col = nb * fx.Int32(NI) + wave * fx.Int32(NIW) + lane_div_16 * fx.Int32(4)
                for mi in range_constexpr(MR):
                    am = fx.Float32(0.0)
                    for c in range_constexpr(NC1):
                        for v in range_constexpr(4):
                            am = fx.maxnumf(am, fx.absf(ys[mi, c][v]))
                    am = fx.maxnumf(am, am.shuffle_xor(16, 64))
                    am = fx.maxnumf(am, am.shuffle_xor(32, 64))
                    lds_f32[rows[mi] * fx.Int32(NW) + wave] = am  # 4 lanes, same value
                gpu.barrier()
                out_rs = _rsrc(a["a2"], fx.Int64(row0 + fx.Int32(BM)) * fx.Int64(I))
                os_rs = _rsrc(a["a2s"], fx.Int64(row0 + fx.Int32(BM)) * fx.Int64(G2N * 4))
                grp = wave // fx.Int32(WPG)
                for mi in range_constexpr(MR):
                    am = fx.Float32(0.0)
                    pv = Vec(lds_vec_load(lds_i32, (rows[mi] * fx.Int32(NW) + grp * fx.Int32(WPG))
                                          * fx.Int32(4), Vec.make_type(WPG, fx.Float32),
                                          fx.Float32, align=4 * WPG))
                    for jw in range_constexpr(WPG):
                        am = fx.maxnumf(am, fx.Float32(pv[jw]))
                    s = (am == 0.0).select(fx.Float32(1.0), am * fx.Float32(1.0 / 240.0))
                    inv = fx.Float32(rocdl.rcp(T.f32, _raw(s)))
                    srow = row0 + rows[mi]
                    for c in range_constexpr(NC1):
                        q = _fp8x4_pack(*[ys[mi, c][v] * inv for v in range(4)])
                        buffer_store(
                            q, out_rs, (srow * fx.Int32(I) + col + fx.Int32(c * 16)) // fx.Int32(4),
                            mask=valid[mi], cache_modifier=WT)
                    buffer_store(
                        s, os_rs, srow * fx.Int32(G2N) + nb * fx.Int32(GPW) + grp,
                        mask=valid[mi] & (lane_div_16 == fx.Int32(0))
                        & (wave % fx.Int32(WPG) == fx.Int32(0)), cache_modifier=WT)

            def run_rows(R):
                def pre_a():  # the W prefetch is in flight
                    if wave == fx.Int32(0):
                        spin_ge(a["sym"] + fx.Int64(O_TREADY) + fx.Int64(t * fx.Int32(4)),
                                n_valid, "one-as")
                    gpu.barrier()
                    if const_expr(QX):
                        build_ratios(R)  # visible after the K loop's first barrier
                    else:
                        # the rows' x scales -> LDS (visible after the K loop's first
                        # barrier); padding rows: 1
                        xs_rs = _rsrc(a["sym"] + fx.Int64(O_RSX), SCAP * 4)
                        sc = fx.Int32(_ld(xs_rs, row0 + tx, mask=tx < n_valid)).bitcast(fx.Float32)
                        sc = (tx < n_valid).select(sc, fx.Float32(1.0))
                        _lds_f32_store_masked(lds_f32, fx.Int32(L_FIN) + tx, sc,
                                              tx < fx.Int32(R))

                acc = core(R, row_base16, hook if QX else None, pre_a=pre_a)
                epilogue(R, acc)
                rocdl.s_waitcnt(vmcnt=0)  # a2 written through
                gpu.barrier()
                if wave == fx.Int32(0):
                    lane0_add(ctl_at(fx.Int32(LC_G1) + t * fx.Int32(16)), 1)

            _dispatch_rows(ROWV, [n_valid <= fx.Int32(r) for r in ROWV[:-1]], run_rows)

        # -------------------------------------------------------------- gemm2
        def gemm2_unit(a, t, nb, tx, lane, wave):
            lane_div_16 = lane // fx.Int32(16)
            lane_mod_16 = lane % fx.Int32(16)
            te = _u(_sys_load_at(ctl_at(fx.Int32(LC_TE) + t)))
            e = te & fx.Int32(255)
            n_valid = te >> fx.Int32(8)
            row0 = t * fx.Int32(BM)
            col0 = nb * fx.Int32(NO) + wave * fx.Int32(NOW)
            cols = [((0, c), e * fx.Int32(H) + col0 + fx.Int32(c * 16), (0, c // 2))
                    for c in range(NC2)]
            exps = [fx.Int32(_global_i32_at(a["w2e"], e * fx.Int32(H) + col0 + fx.Int32(c * 16)
                                            + lane_mod_16)) for c in range(NC2)]
            core = _make_core(
                K=I, W_ROWS=E_LOCAL * H, BM=BM, PF=PF, bcm=BCM2, cols=cols,
                lds_ptr=lds_ptr, LDS_BYTES=LDS_BYTES, lane=lane, tx=tx, NT=NT, arg_w=a["w2"],
                arg_ws=a["w2s"], arg_a=a["a2"], a_num_bytes=SCAP * I, ARB=ARB, acm=2, SCH=3,
                TR=False)

            def row_base16(row):
                return (row0 + row) * fx.Int32(I // 16)

            def build_ratios(R):
                """LDS: ratio[g][r] = s[r][g-1] / s[r][g] (ratio[0] = 1), fin[r] = s[r][G-1];
                a zero scale acts as 1."""
                as_tiles = _global_i32_buffer_tiles(a["a2s"], SCAP * G2N * 4, 1)
                as_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Int32)
                for pp in range_constexpr(-(-(R * G2N) // NT)):
                    idx = tx + fx.Int32(pp * NT)
                    r = idx // fx.Int32(G2N)
                    g = idx % fx.Int32(G2N)
                    ok = r < fx.Int32(R)
                    rr = ok.select(r, fx.Int32(0))
                    srow = (row0 + rr) * fx.Int32(G2N)
                    cur = _buffer_i32_scalar_read(as_tiles, srow + g, as_atom).bitcast(fx.Float32)
                    prv = _buffer_i32_scalar_read(
                        as_tiles, srow + fx.max(g - fx.Int32(1), fx.Int32(0)), as_atom
                    ).bitcast(fx.Float32)
                    cur_ok = (cur == fx.Float32(0.0)).select(fx.Float32(1.0), cur)
                    ratio = (g == fx.Int32(0)).select(fx.Float32(1.0), prv / cur_ok)
                    _lds_f32_store_masked(lds_f32, fx.Int32(RATIO // 4) + g * fx.Int32(BM) + rr,
                                          ratio, ok)
                    _lds_f32_store_masked(lds_f32, fx.Int32(RATIO // 4 + G2N * BM) + rr, cur,
                                          ok & (g == fx.Int32(G2N - 1)))

            def ratio4(base_f32, mi):
                """this lane's 4 rows (L//16)*4 + v of 16-row block mi from an f32 [BM] row."""
                off_ = (base_f32 + fx.Int32(mi * 16) + lane_div_16 * fx.Int32(4)) * fx.Int32(4)
                return Vec(lds_vec_load(lds_i32, off_, Vec.make_type(4, fx.Float32), fx.Float32,
                                        align=16))

            def hook(acc, kt, s):
                kk = s * 32
                if const_expr(kk % 128 != 0):
                    return
                gidx = kt * fx.Int32(ARB // 128) + fx.Int32(kk // 128)
                for mi in range_constexpr(len(acc[0, 0])):
                    rt = ratio4(fx.Int32(RATIO // 4) + gidx * fx.Int32(BM), mi)
                    for c in range_constexpr(NC2):
                        a4 = Vec(acc[0, c][mi].load())
                        acc[0, c][mi].store(Vec.from_elements([a4[v] * rt[v] for v in range(4)],
                                                              fx.Float32))

            def scatter(R):
                """rows of the bf16 LDS tile [R, NO] back to the routes' source ranks: BF16x2
                atomics into out[token] (128 B per row per pass); COMB "fp8": fp8 + one f32
                scale per 128 cols into cin[token, slot] (8 cols per lane, 16 lanes a group)."""
                tab = global_typed_ptr(a["stab"], T.i64, align=8)
                rid = global_typed_ptr(a["sym"] + fx.Int64(O_RID), T.i32)
                if const_expr(FP8C):
                    c8 = (tx % fx.Int32(LPR)) * fx.Int32(8)
                    for pp in range_constexpr(-(-R // SRP)):
                        rib = fx.Int32(pp * SRP) + tx // fx.Int32(LPR)
                        ribc = fx.min(rib, fx.Int32(R - 1))
                        sid = fx.Int32(rid[row0 + ribc])
                        ok = (rib < fx.Int32(R)) & (rib < n_valid)
                        rank = ok.select((sid >> fx.Int32(TSH)) & fx.Int32(NPES - 1), fx.Int32(0))
                        slot = ((sid & fx.Int32(TOKEN_MASK)) * fx.Int32(TOPK)
                                + ((sid >> fx.Int32(24)) & fx.Int32(TOPK - 1)))
                        pk = lds_vec_load(lds_i32, (ribc * fx.Int32(NO) + c8) * fx.Int32(2),
                                          Vec.make_type(4, fx.Int32), fx.Int32, align=16)
                        f = Vec(pk.bitcast(fx.BFloat16)).to(fx.Float32)
                        vals = [fx.Float32(f[i]) for i in range(8)]
                        am = fx.Float32(0.0)
                        for v in vals:
                            am = fx.maxnumf(am, fx.absf(v))
                        for s in range_constexpr(4):  # 16 lanes = 128 cols
                            am = fx.maxnumf(am, am.shuffle_xor(1 << s, 64))
                        sc = (am == fx.Float32(0.0)).select(fx.Float32(1.0),
                                                            am / fx.Float32(240.0))
                        inv = fx.Float32(1.0) / sc
                        q = Vec.from_elements(
                            [_fp8x4_pack(*[vals[4 * h + i] * inv for i in range(4)])
                             for h in range(2)], fx.Int32)
                        base = fx.Int64(tab[rank])
                        dst = (base + fx.Int64(O_CIN) + fx.Int64(slot) * fx.Int64(H)
                               + fx.Int64(nb * fx.Int32(NO)))
                        buffer_store(q, _rsrc(dst, ok.select(fx.Int32(NO), fx.Int32(0))),
                                     c8 // fx.Int32(4), cache_modifier=P2P)
                        sdst = (base + fx.Int64(O_CINS)
                                + (fx.Int64(slot) * fx.Int64(H // 128)
                                   + fx.Int64(nb * fx.Int32(NO // 128))) * fx.Int64(4))
                        buffer_store(sc, _rsrc(sdst, ok.select(fx.Int32(NO // 128 * 4),
                                                               fx.Int32(0))),
                                     c8 // fx.Int32(128), mask=(tx % fx.Int32(16)) == fx.Int32(0),
                                     cache_modifier=P2P)
                    return
                cs = (tx % fx.Int32(32)) * fx.Int32(2)
                rows = []
                for pp in range_constexpr(-(-R // RPS)):
                    rib = fx.Int32(pp * RPS) + tx // fx.Int32(32)
                    sid = fx.Int32(rid[row0 + fx.min(rib, fx.Int32(R - 1))])
                    ok = (rib < fx.Int32(R)) & (rib < n_valid)
                    rank = ok.select((sid >> fx.Int32(TSH)) & fx.Int32(NPES - 1), fx.Int32(0))
                    row_addr = fx.Int64(tab[rank]) + fx.Int64(O_OUT) \
                        + fx.Int64(sid & fx.Int32(TOKEN_MASK)) * fx.Int64(H * 2) \
                        + fx.Int64((nb * fx.Int32(NO) + cs) * fx.Int32(2))
                    rows.append((rib, row_addr, ok))
                for rib, row_addr, ok in rows:
                    if ok:
                        for s in range_constexpr(NO // 64):
                            pk = lds_vec_load(lds_i32, (rib * fx.Int32(NO) + cs + fx.Int32(s * 64))
                                              * fx.Int32(2), Vec.make_type(2, fx.BFloat16),
                                              fx.BFloat16, align=4)
                            llvm.AtomicRMWOp(
                                llvm.AtomicBinOp.fadd,
                                get_element_ptr(_global_base_ptr1(row_addr), fx.Int32(s * 128)),
                                _raw(pk), llvm.AtomicOrdering.monotonic, syncscope="agent",
                                alignment=4)

            def epilogue(R, acc):
                """acc * fin * channel factor * route weight -> bf16 LDS tile [R, NO] -> scatter."""
                MR = R // 16
                fac = [_channel_factor(xe) for xe in exps]
                sw_tiles = _global_i32_buffer_tiles(a["sym"] + fx.Int64(O_RW), SCAP * 4, 1)
                sw_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Int32)
                lds_bf16 = lds_typed_ptr(lds_i32, T.bf16)
                vals = {}
                for mi in range_constexpr(MR):
                    rows4 = fx.Int32(mi * 16) + lane_div_16 * fx.Int32(4)
                    wts = [_buffer_i32_scalar_read(sw_tiles, row0 + rows4 + fx.Int32(v), sw_atom)
                           .bitcast(fx.Float32) for v in range(4)]
                    fin = ratio4(fx.Int32(RATIO // 4 + G2N * BM), mi)
                    wts = [wts[v] * fin[v] for v in range(4)]
                    for c in range_constexpr(NC2):
                        a4 = Vec(acc[0, c][mi].load())
                        vals[mi, c] = [a4[v] * (wts[v] * fac[c]) for v in range(4)]
                gpu.barrier()  # ratio table / A region free for the staging tile
                for mi in range_constexpr(MR):
                    rows4 = fx.Int32(mi * 16) + lane_div_16 * fx.Int32(4)
                    for c in range_constexpr(NC2):
                        col = wave * fx.Int32(NOW) + fx.Int32(c * 16) + lane_mod_16
                        bf4 = Vec.from_elements(vals[mi, c], fx.Float32).to(fx.BFloat16)
                        for v in range_constexpr(4):
                            lds_bf16[(rows4 + fx.Int32(v)) * fx.Int32(NO) + col] = bf4[v]
                rocdl.s_waitcnt(lgkmcnt=0)
                gpu.barrier()
                scatter(R)

            def run_rows(R):
                def pre_a():  # the W prefetch is in flight
                    if wave == fx.Int32(0):
                        spin_ge(ctl_at(fx.Int32(LC_G1) + t * fx.Int32(16)), fx.Int32(NG),
                                "agent-one-as")
                    gpu.barrier()
                    build_ratios(R)  # visible after the K loop's first barrier

                acc = core(R, row_base16, hook, pre_a=pre_a)
                epilogue(R, acc)

            _dispatch_rows(ROWV, [n_valid <= fx.Int32(r) for r in ROWV[:-1]], run_rows)

        # -------------------------------------------------------------- units
        def run_unit(a, u, u1):
            tx, lane, wave = _opaque_tx(tx0)
            if u < u1:
                gemm1_unit(a, u // fx.Int32(NG), u % fx.Int32(NG), tx, lane, wave)
            else:
                v = u - u1
                gemm2_unit(a, v // fx.Int32(ND), v % fx.Int32(ND), tx, lane, wave)

        def units():
            """Every workgroup after its role: gemm1 then gemm2 units from the 8 shards."""
            tl_ = fx.Int32(0)
            if wave0 == fx.Int32(0):
                seen = _sys_load_at(ctl_at(LC_PLAN))
                while seen != epoch:
                    rocdl.s_sleep(2)
                    seen = _sys_load_at(ctl_at(LC_PLAN))
                _inv_l1()
                tl_ = _u(_sys_load_at(ctl_at(LC_PLAN + 1)))
            tiles = to_all(tl_, 1)
            u1 = tiles * fx.Int32(NG)
            total = u1 + tiles * fx.Int32(ND)
            xcd = _u(_xcc_id())
            # shard c: units c, c + 8, ... (gemm1 first); the own shard first, then the
            # others' leftovers (a plain load first: an atomic per leftover workgroup slows
            # the units still running)
            q = fx.Int32(0)
            while q < fx.Int32(8):
                cand = (xcd + q) & fx.Int32(7)
                n_c = (total - cand + fx.Int32(7)) // fx.Int32(8)
                head = ctl_at(fx.Int32(LC_HEAD) + cand * fx.Int32(16))
                got = n_c
                if wave0 == fx.Int32(0):
                    take = q == fx.Int32(0)
                    if q != fx.Int32(0):
                        take = _u(_atomic_load_at(head, "agent-one-as")) < n_c
                    if take:
                        got = _u(lane0_add(head, 1))
                c = to_all(got, 0)
                if c < n_c:
                    run_unit(reload(), cand + fx.Int32(8) * c, u1)
                    gpu.barrier()  # the unit's LDS reads are done
                else:
                    q = q + fx.Int32(1)

        def combine(a):
            """COMB "fp8", once every rank's rows are here: this rank's tokens b, b + GRID, ...:
            out[t] = the sum of its valid slots' rows (f32, slot order; a negative id adds
            nothing). A thread: 8 cols, its 128-col group's scale."""
            n_ = a["n"]
            cin_rs = _rsrc(a["sym"] + fx.Int64(O_CIN), TCAP * TOPK * H)
            cins_rs = _rsrc(a["sym"] + fx.Int64(O_CINS), TCAP * TOPK * (H // 128) * 4)
            out_rs = _rsrc(a["sym"] + fx.Int64(O_OUT), fx.Int64(n_) * fx.Int64(H * 2))
            ids_p = global_typed_ptr(a["ids"], T.i32)
            t = bid
            while t < n_:
                acc = Vec.filled(8, 0.0, fx.Float32)
                for k in range_constexpr(TOPK):
                    g = _u(fx.Int32(ids_p[t * fx.Int32(TOPK) + fx.Int32(k)]))
                    ok = (g >= fx.Int32(0)) & (g < fx.Int32(E_GLOBAL))
                    sl = t * fx.Int32(TOPK) + fx.Int32(k)
                    v = Vec(buffer_load(cin_rs, sl * fx.Int32(H // 4) + tx0 * fx.Int32(2),
                                        vec_width=2, dtype=T.i32, mask=ok, cache_modifier=CM_SYS))
                    s = fx.Int32(buffer_load(cins_rs, sl * fx.Int32(H // 128) + tx0 // fx.Int32(16),
                                             vec_width=1, dtype=T.i32, mask=ok,
                                             cache_modifier=CM_SYS)).bitcast(fx.Float32)
                    fs = []
                    for w in range_constexpr(2):
                        for hi in (False, True):
                            p2 = Vec(rocdl.cvt_pk_f32_fp8(T.vec(2, T.f32), _raw(fx.Int32(v[w])),
                                                          hi))
                            fs.extend(fx.Float32(p2[i]) for i in range(2))
                    acc = acc + Vec.from_elements([fs[i] * s for i in range(8)], fx.Float32)
                buffer_store(acc.to(fx.BFloat16).bitcast(fx.Int32), out_rs,
                             t * fx.Int32(H // 2) + tx0 * fx.Int32(4))
                t = t + fx.Int32(GRID)

        def finish(a):
            """Count this workgroup out; the last one tells every rank its adds / rows are
            done and resets the counters (COMB "add": after all ranks' adds arrived, so
            ``out`` is complete at the kernel end); COMB "fp8": every workgroup waits for
            all ranks' rows, then sums its tokens."""
            rocdl.s_waitcnt(vmcnt=0)  # this workgroup's adds / rows are done
            gpu.barrier()
            top = fx.Int32(-1)
            if wave0 == fx.Int32(0):
                g = bid % fx.Int32(8)
                n_in_g = (fx.Int32(GRID + 7) - g) // fx.Int32(8)
                old = _u(lane0_add(ctl_at(fx.Int32(LC_SUB) + g * fx.Int32(16)), 1))
                if old == n_in_g - fx.Int32(1):
                    top = _u(lane0_add(ctl_at(LC_TOP), 1))
            if to_all(top, 2) == fx.Int32(7):
                if tx0 < fx.Int32(NPES):
                    _sys_store_at(peer(a, tx0) + fx.Int64(O_FFLAG)
                                  + fx.Int64(a["me"] * fx.Int32(LINE)), epoch)
                if const_expr(not FP8C):
                    wait_flags(sym0 + fx.Int64(O_FFLAG), epoch)
                tiles = _u(_sys_load_at(ctl_at(LC_PLAN + 1)))
                for i in range(tx0, tiles, NT):
                    _sys_store_at(ctl_at(fx.Int32(LC_G1) + i * fx.Int32(16)), fx.Int32(0))
                if tx0 < fx.Int32(8):
                    _sys_store_at(ctl_at(fx.Int32(LC_HEAD) + tx0 * fx.Int32(16)), fx.Int32(0))
                    _sys_store_at(ctl_at(fx.Int32(LC_SUB) + tx0 * fx.Int32(16)), fx.Int32(0))
                if tx0 == fx.Int32(0):
                    _sys_store_at(ctl_at(LC_TOP), fx.Int32(0))
                    _sys_store_at(ctl_at(LC_QDONE), fx.Int32(0))
                    _sys_store_at(ctl_at(LC_EPOCH), epoch)
            if const_expr(FP8C):
                wait_flags(sym0 + fx.Int64(O_FFLAG), epoch)
                combine(reload())

        a0 = reload()
        if bid == fx.Int32(0):
            planner(a0)
        else:
            if bid <= fx.Int32(P):
                scan(a0, bid - fx.Int32(1))
            else:
                quant(a0, bid - fx.Int32(P + 1))
        if (bid >= fx.Int32(1)) & (bid <= fx.Int32(P)):
            send(reload(), bid - fx.Int32(1))
        units()
        finish(reload())

    @flyc.jit
    def launch(x: fx.Int64, ids: fx.Int64, tw: fx.Int64, sym: fx.Int64, stab: fx.Int64,
               ctl: fx.Int64, x8: fx.Int64, xs: fx.Int64, lists: fx.Int64, a2: fx.Int64,
               a2s: fx.Int64, w1: fx.Int64, w1s: fx.Int64, w1e: fx.Int64, w2: fx.Int64,
               w2s: fx.Int64, w2e: fx.Int64, n: fx.Int32, me: fx.Int32, sb: fx.Float32,
               sbr: fx.Float32, lb: fx.Float32, lbr: fx.Float32, lim: fx.Float32,
               stream: fx.Stream):
        kernel(x, ids, tw, sym, stab, ctl, x8, xs, lists, a2, a2s, w1, w1s, w1e, w2, w2s, w2e,
               n, me, sb, sbr, lb, lbr, lim,
               value_attrs={"rocdl.waves_per_eu": 2},
               ).launch(grid=(GRID, 1, 1), block=(NT, 1, 1), stream=stream)

    return launch
