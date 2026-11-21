# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""megamoe_ep_a8w4: the EP8 MoE layer (dispatch + a8w4 experts + combine) as one
persistent kernel, for large batches (prefill, >= 2048 tokens in total).

Replaces MORI dispatch -> ``a8w4_moe_dyn`` -> MORI combine. Per call, on one stream,
without host synchronization:

  ep_quant   x bf16 -> fp8 + one f32 scale per (token, 128 columns); out[:T] zeroed
  ep_prep    count this rank's routes per global expert, exchange the counts with all
             ranks, derive every expert's rows on its destination (1 workgroup)
  ep_plan    every route's destination row and send order; this rank's blocks
  mega       kernels/ep_mega.py: producers write each route's row straight into its
             destination, gemm1 / gemm2 tasks run on the blocks as their rows arrive,
             gemm2 adds every route into out[token] of its source rank (BF16 atomics)
  ep_finish  all ranks' adds are done -> out is complete

Needs: 8 ranks of one node (EP8, 32 local experts each), MORI shmem initialized
(``mori.shmem.shmem_torch_process_group_init``: the buffers peers write live in its
symmetric heap), and the whole GPU while it runs (one workgroup on every CU). All kernels
are FlyDSL (``kernels/ep_mega.py``, ``kernels/ep_mega_comm.py``).

CUDA graphs: capture after one eager call (it compiles the kernels); a replay is a call
with the captured token count, so pad the batch to it with topk_ids -1 (as sglang's
``num_token_non_padded`` mask does): negative ids are no routes, their output rows stay 0.
The cross-rank flags carry a device-side call counter, so replays, eager calls and several
graphs mix freely as long as every rank makes the same calls in the same order.

``MegaMoeSmallEpA8W4``: the same layer for decode-sized batches (<= ``tmax`` tokens over
all ranks, 1024 by default) as ONE kernel per call (``kernels/ep_mega_small.py``: x quant
and dispatch into fixed per-expert slots, gemm1 / gemm2, the combine), same contract.

``MegaMoeLargeEpA8W4``: the layer for prefill-sized batches as ONE kernel per call
(``kernels/ep_mega_large.py``, after aiter's MegaMoEV2 large-input kernel: x quant, the
count exchange and plan, dispatch, gemm1 / gemm2 and the combine), same contract.
"""

import torch

from .kernels._act import situ_args
from .kernels._protocol import (
    ARRIVED,
    BLOCK_M,
    BUFFER_BYTES_MAX,
    E_LOCAL,
    LC_PLAN,
    NPES,
    QUEUES,
    ROW_BITS,
    SC_TILE,
    SCHED_EXPECTED,
    large_ctl_ints,
    large_scap,
    large_sym_layout,
    qx_of,
    small_max_tiles,
    small_sym_layout,
)
from .kernels import ep_mega as K
from .kernels import ep_mega_comm as C
from .kernels import ep_mega_large as KL
from .kernels import ep_mega_small as KS


class MegaMoeEpA8W4:
    """One rank's EP8 MoE layer. ``W``: ``prepare_weights`` of the rank's 32 experts
    (global experts [32 rank, 32 rank + 32)); ``max_tokens``: tokens per rank and call.
    ``routes``: routes per token, the real top-k when the ``topk`` slots are padded with -1
    (default ``topk``). ``s_cap``: rows this rank can receive (default: every route of every
    rank, 8 x max_tokens x routes, plus one partial block per expert, so no route is
    dropped; with a smaller ``s_cap`` the routes past it are, see ``rows_needed``).
    ``producers``: producer workgroups (a multiple of 8); ``lag``: gemm2 trails gemm1 by
    this many block pairs of a queue. ``act`` / ``situ``: as ``a8w4_moe``. ``x_quant``:
    the fp8 x sent (``X_QUANTS``: one scale per (row, 128 columns) or per row; the same on
    every rank; ``qx`` = its kernel key, which ``moe_ep``'s solver sets per call)."""

    def __init__(self, W, *, rank, max_tokens, topk=8, world=8, s_cap=None, producers=16,
                 lag=2, act="silu", situ=None, x_quant="per_1x128", routes=None):
        import mori.shmem as ms
        from mori.shmem import mori_shmem_create_tensor

        E, N2, _ = W["w1"].shape
        assert world == NPES and E == E_LOCAL, "EP8 x 32 local experts"
        assert topk in (1, 2, 4, 8) and producers % QUEUES == 0
        self.W, self.rank, self.topk = W, rank, topk
        self.routes = routes or topk
        assert 0 < self.routes <= topk, (self.routes, topk)
        self.H, self.I = W["w2"].shape[1], N2 // 2
        self.producers, self.lag, self.act = producers, lag, act
        self.situ = situ_args(**(situ or {}))
        self.token_shift = max(int(max_tokens - 1).bit_length(), 1)
        self.tcap = 1 << self.token_shift
        bm = BLOCK_M
        s_cap = s_cap or NPES * self.tcap * self.routes + E_LOCAL * bm
        self.s_cap = -(-s_cap // bm) * bm
        assert self.s_cap < 1 << ROW_BITS
        assert self.s_cap * self.H < BUFFER_BYTES_MAX, (
            f"{self.s_cap} receive rows of {self.H} bytes pass 4 GiB: lower max_tokens")
        self.nblk = self.s_cap // bm
        H, I, S = self.H, self.I, self.s_cap
        dev = torch.device("cuda", torch.cuda.current_device())

        def sym(shape, dtype):
            t = mori_shmem_create_tensor(shape, dtype)
            t.zero_()
            return t

        self.recv_a = sym((S, H), torch.uint8)
        self.recv_scale = sym((S, H // 128), torch.float32)
        self.down_ids = sym((S,), torch.int32)
        self.recv_w = sym((S,), torch.float32)
        self.ws = sym((ARRIVED + self.nblk,), torch.int32)
        self.out = sym((self.tcap, H), torch.bfloat16)
        torch.cuda.synchronize()
        ms.shmem_barrier_all()
        self.tables = {
            name: torch.tensor([ms.shmem_ptr_p2p(getattr(self, name).data_ptr(), rank, p)
                                for p in range(world)], dtype=torch.int64, device=dev)
            for name in ("recv_a", "recv_scale", "down_ids", "recv_w", "ws", "out")}
        i32 = dict(dtype=torch.int32, device=dev)
        self.route_idx = torch.empty(self.tcap * topk, **i32)
        self.send_route = torch.empty(self.tcap * topk, **i32)
        self.send_info = torch.empty(self.tcap * topk, **i32)
        self.num_valid = torch.zeros(1, **i32)
        self.status = torch.zeros(8, **i32)
        self.plan = torch.empty(C.PLAN_INTS, **i32)
        self.expert_ids = torch.zeros(self.nblk, **i32)
        self.sched = torch.zeros(SCHED_EXPECTED + 2 * self.nblk, **i32)
        # ep_plan: one workgroup per producer wave, each sorting up to 1024 routes
        self.slices = producers * 8
        assert self.tcap * topk <= self.slices * 1024, "too many routes per ep_plan slice"
        self.a2 = torch.empty((S, I), dtype=torch.uint8, device=dev)
        self.a2_scale = torch.empty((S, I // 128), dtype=torch.float32, device=dev)
        self.x8 = torch.empty((self.tcap, H), dtype=torch.uint8, device=dev)
        self.xs = torch.empty((self.tcap, H // 128), dtype=torch.float32, device=dev)
        self.grid = min(torch.cuda.get_device_properties(dev).multi_processor_count,
                        K.max_tasks(self.nblk, H, I, producers))
        # W loads: non-temporal while an expert's rows fit one block (no reuse); from
        # ``big_tokens`` tokens on this rank (uniform routing: rows per expert >= BLOCK_M)
        # the blocks of an expert re-read its W1: cached gemm1 W, gemm2 W bcm 1
        # (-1.6 .. -3.6% at 4096 - 32768 tokens in total, +1.0 .. +1.5% at 2048 - 3072;
        # every call on its own weight copy, experiments/log_20261115.md)
        kw = dict(H=H, I=I, TOPK=topk, token_shift=self.token_shift, producers=producers,
                  lag=lag, ACT=act)
        self.qx = qx_of(x_quant)
        if not self.qx:
            kw["QX"] = 0
        self.launch = K.ep_mega_kernel.get(**kw)
        self.launch_big = K.ep_mega_kernel.get(**kw, BCM1=0, BCM2=1)
        self.big_tokens = BLOCK_M * E_LOCAL // topk
        self.quant_ks = [C.mega_quant_kernel.get(H=H, QX=qx) for qx in (0, 1)]
        self.prep_k = C.mega_prep_kernel.get()
        self.plan_k, self.finish_k = C.mega_plan_kernel.get(), C.mega_finish_kernel.get()

    def __call__(self, x, topk_ids, topk_weights, x_scale=None, marks=None):
        """x: bf16 [T, H] (quantized here), or fp8 FNUZ [T, H] with ``x_scale`` f32
        [T, H/128] (``qx`` 1) / [T] (``qx`` 0); topk_ids int32 [T, topk] (global ids;
        negative: no route, e.g. padding), topk_weights f32 [T, topk].
        Returns bf16 [T, H], a view of the symmetric output buffer (valid until the next
        call). ``marks``: a list that gets (stage, CUDA event) after every stage (timing)."""

        def mark(name):
            if marks is not None:
                ev = torch.cuda.Event(enable_timing=True)
                ev.record()
                marks.append((name, ev))

        mark("start")
        T, H = x.shape
        assert T <= self.tcap and H == self.H
        assert topk_ids.dtype == torch.int32 and topk_weights.dtype == torch.float32
        topk_ids, topk_weights = topk_ids.contiguous(), topk_weights.contiguous()
        tab = self.tables
        routes = T * self.topk
        quant = x_scale is None
        if quant:
            x8, xs = self.x8, self.xs
        else:
            shape = (T, H // 128) if self.qx else (T,)
            assert x_scale.shape == shape and x_scale.dtype == torch.float32, (x_scale.shape, shape)
            x8, xs = x.contiguous().view(torch.uint8), x_scale.contiguous()
        if T:
            waves = T * (H // 512) if self.qx else T
            self.quant_ks[self.qx](x if quant else None, x8, xs, self.out, T, int(quant),
                                   -(-waves // 4))
        mark("quant")
        self.prep_k(topk_ids, self.route_idx, self.ws, tab["ws"], self.num_valid, self.status,
                    self.sched, self.plan, routes, self.rank, self.s_cap)
        mark("prep")
        self.plan_k(topk_ids, self.route_idx, self.plan, self.send_route, self.send_info,
                    self.down_ids, self.expert_ids, self.sched, routes, self.rank, self.s_cap,
                    self.slices)
        mark("plan")
        W = self.W
        mega = self.launch_big if T >= self.big_tokens else self.launch
        mega(x8, xs, topk_weights, self.send_route, self.send_info, tab["recv_a"],
             tab["recv_scale"], tab["down_ids"], tab["recv_w"], tab["ws"], tab["out"],
             self.recv_a, self.recv_scale, self.down_ids, self.recv_w, self.ws, W["w1"],
             W["w1_scale"], W["w1_exp"], W["w2"], W["w2_scale"], W["w2_exp"], self.a2,
             self.a2_scale, self.expert_ids, self.num_valid, self.sched, routes, T, self.rank,
             self.nblk, *self.situ, self.grid)
        mark("mega")
        self.finish_k(self.ws, tab["ws"], self.status, self.rank)
        mark("finish")
        return self.out[:T]

    def x_ref(self, n):
        """f32 [n, H]: the last call's x as the quant kernel quantized it (bf16 x; n: its
        tokens)."""
        H = self.H
        x8 = self.x8[:n].view(torch.float8_e4m3fnuz).float()
        if self.qx:
            return (x8.view(n, H // 128, 128) * self.xs[:n].unsqueeze(-1)).view(n, H)
        return x8 * self.xs.view(-1)[:n].view(n, 1)

    def rows_needed(self):
        """Rows the last call needed on this rank (> s_cap: routes were dropped)."""
        return int(self.status[1].item())


# kernel keys of MegaMoeSmallEpA8W4's knobs: m-tile rows, gemm2 cols per unit, A bytes per
# K tile, W prefetch depth, producer workgroups, workgroups per CU, gemm1 / gemm2 W cache
# modifier (experiments/20261118_mega_small: only bm changes the time beyond +-2%)
SMALL_KNOBS = dict(bm=64, no=256, arb=256, pf=2, producers=16, wpc=1, bcm1=2, bcm2=2)
SMALL_KEYS = dict(bm="BM", no="NO", arb="ARB", pf="PF", producers="P", wpc="WPC", bcm1="BCM1",
                  bcm2="BCM2")


class MegaMoeSmallEpA8W4:
    """One rank's EP8 MoE layer for decode-sized batches: ONE kernel per call
    (``kernels/ep_mega_small.py``, after aiter's MegaMoEV2 small-input kernel): x quant and
    dispatch straight into fixed per-expert slots on the destination, gemm1 / gemm2, and the
    combine (each rank sums its tokens' rows).

    ``W``: ``prepare_weights`` of the rank's 32 experts (global experts [32 rank, 32 rank +
    32)), for the shape; ``tmax``: the most tokens a call brings over ALL ranks (each local
    expert has tmax rows of slots, the combine tmax x topk rows: ~196 MB of the MORI heap
    at tmax 1024, H 4096, top-8). Same contract as ``MegaMoeEpA8W4`` otherwise: every rank
    makes the same calls, negative topk ids are no routes (their rows stay 0), CUDA graph
    capture after one eager call."""

    def __init__(self, W, *, rank, tmax=1024, topk=8, world=8):
        import mori.shmem as ms
        from mori.shmem import mori_shmem_create_tensor

        E, N2, _ = W["w1"].shape
        assert world == NPES and E == E_LOCAL, "EP8 x 32 local experts"
        self.W, self.rank, self.topk, self.tmax = W, rank, topk, tmax
        self.H, self.I = W["w2"].shape[1], N2 // 2
        dev = torch.device("cuda", torch.cuda.current_device())
        self.sym = mori_shmem_create_tensor((small_sym_layout(self.H, topk, tmax)[1],),
                                            torch.uint8)
        self.sym.zero_()
        torch.cuda.synchronize()
        ms.shmem_barrier_all()
        self.stab = torch.tensor([ms.shmem_ptr_p2p(self.sym.data_ptr(), rank, p)
                                  for p in range(world)], dtype=torch.int64, device=dev)
        bms = (16, 32, 64)  # a2 rows / tile counters for every m-tile size
        rows = max(small_max_tiles(topk, tmax, bm) * bm for bm in bms)
        tiles = max(small_max_tiles(topk, tmax, bm) for bm in bms)
        self.ctl = torch.zeros(SC_TILE + 16 * tiles, dtype=torch.int32, device=dev)
        self.a2 = torch.empty((rows, self.I), dtype=torch.uint8, device=dev)
        self.a2s = torch.empty((rows, self.I // 128), dtype=torch.float32, device=dev)
        self.out = torch.empty((tmax, self.H), dtype=torch.bfloat16, device=dev)

    def kernel(self, act="silu", x_quant="per_1x128", **knobs):
        """The kernel specialization of ``knobs`` (``SMALL_KNOBS`` names)."""
        k = dict(SMALL_KNOBS, **knobs)
        return KS.ep_mega_small_kernel.get(H=self.H, I=self.I, TOPK=self.topk, TMAX=self.tmax,
                                           ACT=act, QX=qx_of(x_quant),
                                           **{SMALL_KEYS[n]: v for n, v in k.items()})

    def __call__(self, x, topk_ids, topk_weights, W=None, act="silu", situ=None,
                 x_quant="per_1x128", **knobs):
        """x bf16 [T, H] (this rank's tokens), topk_ids int32 [T, topk] (global ids;
        negative: no route), topk_weights f32 [T, topk]; ``W``: the call's experts (default:
        the constructor's); ``x_quant``: the fp8 x sent (``X_QUANTS``; the same on every
        rank). Returns bf16 [T, H], a view of this object's output buffer (valid until the
        next call)."""
        T, H = x.shape
        assert T <= self.tmax and H == self.H, (T, self.tmax, H)
        assert x.dtype == torch.bfloat16 and topk_ids.dtype == torch.int32
        assert topk_weights.dtype == torch.float32 and topk_ids.shape == (T, self.topk)
        W = self.W if W is None else W
        x, topk_ids, topk_weights = x.contiguous(), topk_ids.contiguous(), topk_weights.contiguous()
        self.kernel(act, x_quant, **knobs)(
            x, topk_ids, topk_weights, self.out, self.sym, self.stab, self.ctl, self.a2,
            self.a2s, W["w1"], W["w1_scale"], W["w1_exp"], W["w2"], W["w2_scale"], W["w2_exp"],
            T, self.rank, *situ_args(**(situ or {})))
        return self.out[:T]


# kernel keys of MegaMoeLargeEpA8W4's knobs: producer workgroups, gemm2 cols per unit, A bytes
# per K tile, W prefetch depth, the combine ("add": BF16 atomics, "fp8": fp8 rows summed by
# the source rank), gemm1 / gemm2 W cache modifier (experiments/20261118_mega_large)
LARGE_KNOBS = dict(producers=16, no=256, arb=128, pf=2, comb="add", bcm1=2, bcm2=2)
LARGE_KEYS = dict(producers="P", no="NO", arb="ARB", pf="PF", comb="COMB", bcm1="BCM1",
                  bcm2="BCM2")


class MegaMoeLargeEpA8W4:
    """One rank's EP8 MoE layer for prefill-sized batches: ONE kernel per call
    (``kernels/ep_mega_large.py``, after aiter's MegaMoEV2 large-input kernel): x quant
    (``x_quant``: one fp8 scale per (row, 128 columns), or per row as ``a8w4_moe``), the
    count exchange and the plan, the dispatch, gemm1 / gemm2 and the combine (BF16 atomics
    into the source ranks' outputs, or fp8 rows the source ranks sum: knob ``comb``).

    ``W``: ``prepare_weights`` of the rank's 32 experts (global experts [32 rank, 32 rank +
    32)), for the shape; ``max_tokens``: tokens per rank and call (rounded up to a power of
    2, ``tcap``); ``routes``: routes per token, the real top-k when the ``topk`` slots are
    padded with -1 (default ``topk``). A rank receives at most ``s_cap`` = 8 tcap routes +
    32 x 128 rows: every route of every rank, so no route is dropped. Symmetric heap:
    1.30 GB at tcap 4096 (H 4096, top-8; ~linear in tcap). Same contract as
    ``MegaMoeEpA8W4`` otherwise: every rank makes the same calls (with the same ``comb`` and
    ``x_quant``: they are the wire format), negative topk ids are no routes (their rows stay
    0), CUDA graph capture after one eager call."""

    def __init__(self, W, *, rank, max_tokens, topk=8, world=8, routes=None):
        import mori.shmem as ms
        from mori.shmem import mori_shmem_create_tensor

        from .kernels._protocol import E_GLOBAL

        E, N2, _ = W["w1"].shape
        assert world == NPES and E == E_LOCAL, "EP8 x 32 local experts"
        self.W, self.rank, self.topk = W, rank, topk
        self.routes = routes or topk
        self.H, self.I = W["w2"].shape[1], N2 // 2
        self.tcap = 1 << max(int(max_tokens - 1).bit_length(), 1)
        self.s_cap = large_scap(self.routes, self.tcap)
        dev = torch.device("cuda", torch.cuda.current_device())
        off, nbytes = large_sym_layout(self.H, topk, self.tcap, self.routes)
        self.sym = mori_shmem_create_tensor((nbytes,), torch.uint8)
        self.sym.zero_()
        torch.cuda.synchronize()
        ms.shmem_barrier_all()
        self.stab = torch.tensor([ms.shmem_ptr_p2p(self.sym.data_ptr(), rank, p)
                                  for p in range(world)], dtype=torch.int64, device=dev)
        self.ctl = torch.zeros(large_ctl_ints(topk, self.tcap, self.routes)[1],
                               dtype=torch.int32, device=dev)
        H, I, S = self.H, self.I, self.s_cap
        self.x8 = torch.empty((self.tcap, H), dtype=torch.uint8, device=dev)
        self.xs = torch.empty(self.tcap * (H // 128), dtype=torch.float32, device=dev)
        self.lists = torch.empty(E_GLOBAL * self.tcap, dtype=torch.int32, device=dev)
        self.a2 = torch.empty((S, I), dtype=torch.uint8, device=dev)
        self.a2s = torch.empty((S, I // 128), dtype=torch.float32, device=dev)
        o = off["out"]
        self.out = self.sym[o:o + self.tcap * H * 2].view(torch.bfloat16).view(self.tcap, H)
        self._qx = 1

    def kernel(self, act="silu", x_quant="per_1x128", **knobs):
        """The kernel specialization of ``knobs`` (``LARGE_KNOBS`` names)."""
        k = dict(LARGE_KNOBS, **knobs)
        return KL.ep_mega_large_kernel.get(H=self.H, I=self.I, TOPK=self.topk, TCAP=self.tcap,
                                           ACT=act, QX=qx_of(x_quant), ROUTES=self.routes,
                                           **{LARGE_KEYS[n]: v for n, v in k.items()})

    def __call__(self, x, topk_ids, topk_weights, W=None, act="silu", situ=None,
                 x_quant="per_1x128", **knobs):
        """x bf16 [T, H] (this rank's tokens), topk_ids int32 [T, topk] (global ids;
        negative: no route), topk_weights f32 [T, topk]; ``W``: the call's experts (default:
        the constructor's). Returns bf16 [T, H], a view of the symmetric output buffer
        (valid until the next call)."""
        T, H = x.shape
        assert T <= self.tcap and H == self.H, (T, self.tcap, H)
        assert x.dtype == torch.bfloat16 and topk_ids.dtype == torch.int32
        assert topk_weights.dtype == torch.float32 and topk_ids.shape == (T, self.topk)
        W = self.W if W is None else W
        x, topk_ids, topk_weights = x.contiguous(), topk_ids.contiguous(), topk_weights.contiguous()
        self.kernel(act, x_quant, **knobs)(
            x, topk_ids, topk_weights, self.sym, self.stab, self.ctl, self.x8, self.xs,
            self.lists, self.a2, self.a2s, W["w1"], W["w1_scale"], W["w1_exp"], W["w2"],
            W["w2_scale"], W["w2_exp"], T, self.rank, *situ_args(**(situ or {})))
        self._qx = qx_of(x_quant)
        return self.out[:T]

    def x_ref(self, n):
        """f32 [n, H]: the last call's x as the kernel quantized it (n: its tokens)."""
        x8 = self.x8[:n].view(torch.float8_e4m3fnuz).float()
        if self._qx:
            s = self.xs[:n * (self.H // 128)].view(n, self.H // 128)
            return (x8.view(n, self.H // 128, 128) * s.unsqueeze(-1)).view(n, self.H)
        return x8 * self.xs[:n].view(n, 1)

    def rows_needed(self):
        """Rows the last call needed on this rank (> s_cap: routes were dropped)."""
        return int(self.ctl[LC_PLAN + 2].item())
