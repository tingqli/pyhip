# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Layouts of the buffers one kernel writes and another kernel reads.

Writer and reader import the same names from here.
"""

# ---- EP: tile schedule written by ep_prep, read by ep_gemm ----------------------------
# (i32, right after the cap // BM block expert ids): [0, 9): XCD bucket starts (m-block
# slots); [SCHED_BORDER, + nblk): the m-block order.
SCHED_BORDER = 16

# ---- megamoe kernels: the fp8 x they send (the op moe_ep's dim x_quant) ----------------
# one f32 scale per (row, 128 columns) (kernel key QX 1; MORI's blockwise fp8), or per row
# (QX 0; as a8w4_moe's own x quant)
X_QUANTS = ("per_1x128", "per_token")


def qx_of(x_quant):
    """The kernel key QX of ``x_quant``."""
    assert x_quant in X_QUANTS, x_quant
    return int(x_quant == "per_1x128")

# ---- megamoe: workspace / sched of ep_mega_comm and ep_mega (layout: ep_mega_comm) ----
NPES, E_LOCAL = 8, 32
E_GLOBAL = NPES * E_LOCAL
CFLAG = NPES * E_GLOBAL
FFLAG = CFLAG + NPES
ARRIVED = FFLAG + NPES
QUEUES = 8
QSTRIDE = 16  # i32 between two ticket counters (separate cache lines)
SCHED_EXPECTED = QUEUES * QSTRIDE
ROW_BITS = 18
BLOCK_M = 128
# status (local, i32 [8]): [1] rows the last call needed (ep_prep); [STATUS_EPOCH] calls
# completed (ep_finish advances it; ep_prep / ep_finish flag the call with it + 1)
STATUS_EPOCH = 2

# ---- megamoe small (ep_mega_small): a rank's symmetric buffer, its control block ----
SMALL_LINE = 64  # bytes per cross-rank flag / counter (one cache line each)


def small_sym_layout(H, TOPK, TMAX):
    """(byte offset of each region, total bytes) of a rank's symmetric buffer: fixed slots
    of TMAX rows per local expert (``rx`` fp8 rows, ``rs`` f32 scale per (row, 128 cols),
    ``rw`` route weight, ``rid`` source id), ``run`` (rows the sources took per expert,
    their atomics), ``done`` (source rank s's producers' count of finished calls, in
    DONE_UNIT steps), ``cin`` (bf16 rows the experts send back, per (token, topk slot) of
    this rank), ``cflag`` (rank s sent all its rows back: the epoch)."""
    rows = E_LOCAL * TMAX
    sizes = (("rx", rows * H), ("rs", rows * (H // 128) * 4), ("rw", rows * 4),
             ("rid", rows * 4), ("run", E_LOCAL * SMALL_LINE), ("done", NPES * SMALL_LINE),
             ("cin", TMAX * TOPK * H * 2), ("cflag", NPES * SMALL_LINE))
    off, pos = {}, 0
    for name, nbytes in sizes:
        off[name] = pos
        pos += -(-nbytes // 256) * 256
    return off, pos


def small_max_tiles(TOPK, TMAX, BM):
    """m-tiles a rank can receive: TMAX tokens, each with <= min(topk, 32) routes here."""
    return TMAX * min(TOPK, E_LOCAL) // BM + E_LOCAL


# control block (local, int32; one 64 B line per counter): [SC_EPOCH] the calls completed
# (the epoch of a call is it + 1), the combine count (top, then one per block id % 8), the
# unit counter of each XCD, then the gemm1 units done of each m-tile
SC_EPOCH, SC_TOP = 0, 32
SC_SUB = 48
SC_HEAD = SC_SUB + 8 * 16
SC_TILE = SC_HEAD + 8 * 16

# ---- megamoe large (ep_mega_large): a rank's symmetric buffer, its control block ----
LARGE_BM = 128  # rows of an m-tile; each local expert's received rows are padded to it


def large_scap(TOPK, TCAP):
    """Rows a rank can receive (a multiple of LARGE_BM): twice the uniform-routing average
    (TCAP tokens per rank x TOPK routes) plus one partial m-tile per local expert. Rows past
    it are dropped (the plan reports the rows needed)."""
    return 2 * TCAP * TOPK + E_LOCAL * LARGE_BM


def large_sym_layout(H, TOPK, TCAP):
    """(byte offset of each region, total bytes) of a rank's symmetric buffer. Received
    rows, expert-major (local expert e's rows from its first m-tile on, padded to LARGE_BM;
    inside an expert the sources' segments follow each other, this rank's own first):
    ``rx`` fp8 [S, H], ``rsx`` their f32 scales (one per row, or one per (row, 128 cols):
    [S, H / 128] is reserved), ``rw`` the route weights, ``rid`` the
    source ids (slot << 24 | rank << log2(TCAP) | token); ``cnt`` [NPES][E_LOCAL] (row s:
    source s's routes to this rank's experts), ``pbase`` [E_GLOBAL] (the row where this
    rank's routes to global expert g start on rank g // E_LOCAL, written by its planner),
    per rank s one line of ``cdone`` (s's count row is here: DONE_UNIT per call), ``pflag``
    (s wrote this rank's ``pbase`` entries: the epoch) and ``fflag`` (s's adds into this
    rank's ``out`` / rows into its ``cin`` are done: the epoch); ``tready`` the rows arrived
    per m-tile; ``out`` bf16 [TCAP, H], this rank's output (every rank's experts add into
    it, or the combine sums ``cin`` into it); ``cin`` fp8 [TCAP x TOPK, H] and ``cins`` its
    f32 scales [TCAP x TOPK, H / 128]: the expert rows of this rank's (token, slot) routes,
    sent back (COMB "fp8")."""
    S = large_scap(TOPK, TCAP)
    sizes = (("rx", S * H), ("rsx", S * (H // 128) * 4), ("rw", S * 4), ("rid", S * 4),
             ("cnt", NPES * E_LOCAL * 4), ("pbase", E_GLOBAL * 4), ("cdone", NPES * SMALL_LINE),
             ("pflag", NPES * SMALL_LINE), ("fflag", NPES * SMALL_LINE),
             ("tready", S // LARGE_BM * 4), ("out", TCAP * H * 2), ("cin", TCAP * TOPK * H),
             ("cins", TCAP * TOPK * (H // 128) * 4))
    off, pos = {}, 0
    for name, nbytes in sizes:
        off[name] = pos
        pos += -(-nbytes // 256) * 256
    return off, pos


# control block (local, int32; a 16-int line per counter): [LC_EPOCH] the calls completed
# (the epoch of a call is it + 1), the quant workgroups done, the plan (epoch, m-tiles, rows
# needed), the finish count (one per block id % 8, then the top), the unit counter of each
# of the 8 shards, per m-tile its local expert | valid rows << 8, per m-tile (one line each)
# the gemm1 units done
LC_EPOCH, LC_QDONE, LC_PLAN = 0, 16, 32
LC_SUB = 48
LC_TOP = LC_SUB + 8 * 16
LC_HEAD = LC_TOP + 16
LC_TE = LC_HEAD + 8 * 16


def large_ctl_ints(TOPK, TCAP):
    """(LC_G1, ints of the control block)."""
    tiles = large_scap(TOPK, TCAP) // LARGE_BM
    g1 = -(-(LC_TE + tiles) // 16) * 16
    return g1, g1 + 16 * tiles
