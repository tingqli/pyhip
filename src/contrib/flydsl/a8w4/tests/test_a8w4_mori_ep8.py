# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""MORI dispatch -> ``a8w4_moe_dyn`` -> MORI combine on 8 GPUs: the EP8 MoE layer of
XiaomiMiMo/MiMo-V2.6-Flash(-RL) (hidden 4096, moe_intermediate 2048, 256 routed experts,
top-8, SiLU; 32 experts per rank) as sglang runs it with ``--moe-a2a-backend mori``
(deepep mode normal: MORI IntraNode kernels).

A batch of T tokens (the model's total) is split evenly over the ranks (DP attention);
every rank repeats the sglang steps (``moriep.py``, ``mxfp4_aiter_a16w4_moe.py``):

    quant     aiter fp8 quant, one f32 scale per (token, 128)           dispatch_a
    dispatch  MORI dispatch of x + scales + top-k weights / global ids    dispatch_b
    moe       a8w4_moe_dyn on the receive buffer, capacity-sized output   _apply_mori
              eager: the received rows (count read on the host), est = rows;
              CUDA graph (--graph): the whole buffer (the kernels read the
              count), est = T (DP metadata)
    combine   MORI combine (bf16): sums the ranks' partial outputs        combine

    pytest -q a8w4/tests/test_a8w4_mori_ep8.py     # correctness vs the fp32 reference
    python a8w4/tests/test_a8w4_mori_ep8.py        # perf: T = 64 .. 4096 tokens
    python a8w4/tests/test_a8w4_mori_ep8.py -t 256 1024 --iters 50 --graph

Run from the directory that holds ``a8w4/``; needs 8 gfx942 GPUs and mori.

Timing: every iteration flushes L2 and switches to the next of ``R.WCOPIES`` weight copies
(the flush does not evict weights that every call re-reads from the MALL; a model's next
layer reads other weights), then a host barrier, a GPU
spin while the host enqueues the whole layer (host overhead excluded, as in a CUDA-graph
replay; the eager count read is not timed) and a MORI device barrier, so all ranks start
together. Each iteration runs the layer twice:
  * isolated: a device barrier before every stage, each stage timed alone;
  * back to back, as in a forward pass: e2e of the slowest rank.
Routing is uniform (random scores, renormalized softmax top-8).
"""

import argparse
import os
import statistics
import sys
import time

import pytest
import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from a8w4 import a8w4_moe_dyn  # noqa: E402
from a8w4 import ref as R  # noqa: E402
from a8w4.bench import dist as D  # noqa: E402
from a8w4.bench.dist import E_GLOBAL, E_LOCAL, H, I, TOPK, WORLD  # noqa: E402
from a8w4.moe import MOE_DYN, shape_problem  # noqa: E402

try:
    import mori
except ImportError:  # the test skips
    mori = None

SHAPE = (H, I, E_LOCAL, E_GLOBAL, TOPK)
MAX_TOKENS = 4096  # per rank and dispatch (sglang SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK)
FP8 = torch.float8_e4m3fnuz
COS_TOL = D.COS_TOL
STAGES = ("quant", "dispatch", "moe", "combine")
BENCH_TOKENS = [64, 128, 256, 512, 1024, 2048, 4096]  # T, the total over the ranks
# tokens per rank of the correctness test: a small batch, an uneven one (idle ranks,
# a single token) and the full dispatch capacity
CHECK_TOKENS = [[64] * WORLD, [0, 1, 37, 300, 5, 128, 999, 0], [MAX_TOKENS] * WORLD]
SPIN_US = D.SPIN_US  # GPU spin while the host enqueues one layer

# shared with the other 8-GPU tests and experiments (a8w4/bench/dist.py)
_skip_reason = D.skip_reason
_gather_rows = D.gather_rows
_align = D.align_mori
_spin_cycles = D.spin_cycles
_Spans = D.Spans
_free_port = D.free_port
split_tokens = D.split_tokens


class EpMoeLayer:
    """One rank: its 32 experts (random mxfp4 weights) and the MORI op that sglang's
    ``init_mori_op`` builds for EP8 on one node with fp8 dispatch."""

    def __init__(self, rank):
        from aiter import QuantType, get_hip_quant

        self.rank, self.e0 = rank, rank * E_LOCAL
        self.W, self.raw = R.make_weights(E_LOCAL, H, I, seed=rank)
        self._quant = get_hip_quant(QuantType.per_1x128)
        self.op = mori.ops.EpDispatchCombineOp(mori.ops.EpDispatchCombineConfig(
            data_type=FP8, rank=rank, world_size=WORLD, hidden_dim=H, scale_dim=H // 128,
            scale_type_size=4, max_token_type_size=2,  # bf16 model
            max_num_inp_token_per_rank=MAX_TOKENS, num_experts_per_rank=E_LOCAL,
            num_experts_per_token=TOPK, warp_num_per_block=16, block_num=80,
            kernel_type=mori.ops.EpDispatchCombineKernelType.IntraNode, gpu_per_node=WORLD,
            rdma_block_num=0, num_qp_per_pe=2, quant_type="none"))
        assert self.op.max_num_tokens_to_recv() <= 32768  # one a8w4_moe_dyn call (sglang chunk)

    def quant(self, x):
        if x.shape[0] == 0:  # sglang skips the quant kernel on an empty batch
            return x.to(FP8), x.new_empty((0, H // 128), dtype=torch.float32)
        return self._quant(x, quant_dtype=FP8)

    def next_weights(self):
        """Switch to the next of ``R.WCOPIES`` copies of the weights (made on first use)."""
        if not hasattr(self, "_Ws"):
            self._Ws, self._wi = R.weight_copies(self.W), 0
        self._wi = (self._wi + 1) % len(self._Ws)
        self.W = self._Ws[self._wi]

    def dispatch(self, xq, xs, tw, ti):
        """-> receive buffers: x, top-k weights, x scales, top-k ids, [received rows]."""
        return self.op.dispatch(xq, tw, xs, ti)

    def moe_args(self, count, ns, graph):
        """(rows, est) of sglang's call (``_mori_token_bounds``): eager takes the received
        rows (``count``, read on the host); a CUDA-graph capture the whole buffer, with the
        DP batch (total tokens) as the row estimate."""
        if graph:
            cap = self.op.max_num_tokens_to_recv()
            return cap, max(1, min(cap, sum(ns)))
        return count, max(1, count)

    def moe(self, recv, rows, est):
        """a8w4_moe_dyn on rows [0, rows) of the receive buffer (the kernels stop at the
        received count); ``est``: the row estimate (config, grids)."""
        x, tw, xs, ti, count = recv
        out = torch.empty((x.shape[0], H), dtype=torch.bfloat16, device=x.device)
        if rows:
            a8w4_moe_dyn(x[:rows], self.W, tw[:rows], ti[:rows], count, x_scale=xs[:rows],
                         expert_offset=self.e0, num_experts=E_GLOBAL, est_tokens=est,
                         out=out[:rows])
        return out

    def combine(self, out, ti):
        """-> [MAX_TOKENS, H] bf16, the first len(ti) rows valid."""
        return self.op.combine(out, None, ti)[0]


# =============================================================================
# correctness
# =============================================================================


def _check(layer, ns, x, tw, ti, graph):
    """One layer forward (MoE: sglang's eager / CUDA-graph call) vs the fp32 reference
    (each rank evaluates its experts on all ranks' dispatched tokens, summed over the
    ranks) -> [(received rows, cos diff)] of every rank."""
    rank, n = layer.rank, ns[layer.rank]
    xq, xs = layer.quant(x)
    recv = layer.dispatch(xq, xs, tw, ti)
    count = int(recv[4].item())
    y = layer.combine(layer.moe(recv, *layer.moe_args(count, ns, graph)), ti)[:n].float()

    x_ref = (xq.float().view(n, H // 128, 128) * xs.unsqueeze(-1)).view(n, H)
    X, TW, TI = (_gather_rows(t, ns) for t in (x_ref, tw, ti))
    expect = int(((TI // E_LOCAL) == rank).any(-1).sum())  # a token goes once to each rank
    ref = R.ref_moe(X, TW, TI, layer.raw, layer.e0, quant_x=False)
    dist.all_reduce(ref)
    off = sum(ns[:rank])
    diff = R.cos_diff(ref[off:off + n], y) if n else 0.0
    res = [None] * WORLD
    dist.all_gather_object(res, (count, expect, diff))
    bad = [(r, *v) for r, v in enumerate(res) if v[0] != v[1] or not v[2] < COS_TOL]
    assert not bad, (f"tokens per rank {ns}, graph call {graph}: (rank, received, expected, "
                     f"cos diff) {bad}")
    return [(v[0], v[2]) for v in res]


# =============================================================================
# timing
# =============================================================================


def _timed(layer, x, tw, ti, args, isolated, spin):
    """One timed layer forward (``args``: the MoE's rows, est) -> ({stage: us}, e2e us,
    host enqueue us)."""
    R.flush_cache()
    layer.next_weights()
    dist.barrier()
    torch.cuda._sleep(spin)
    _align()
    t0 = time.perf_counter()
    span = _Spans(isolated)
    span("quant")
    xq, xs = layer.quant(x)
    span("dispatch")
    recv = layer.dispatch(xq, xs, tw, ti)
    span("moe")
    out = layer.moe(recv, *args)
    span("combine")
    layer.combine(out, ti)
    span()
    host = (time.perf_counter() - t0) * 1e6
    torch.cuda.synchronize()
    return (*span.times(), host)


def _bench(layer, x, tw, ti, count, args, iters, warmup, spin):
    for _ in range(warmup):
        layer.combine(layer.moe(layer.dispatch(*layer.quant(x), tw, ti), *args), ti)
    torch.cuda.synchronize()
    iso, e2e, late = [], [], 0
    for _ in range(iters):  # isolated and back-to-back runs interleaved: drift hits both
        t, _, h1 = _timed(layer, x, tw, ti, args, True, spin)
        _, t2, h2 = _timed(layer, x, tw, ti, args, False, spin)
        iso.append(t)
        e2e.append(t2)
        late += (h1 > SPIN_US) + (h2 > SPIN_US)  # the GPU may have waited for the host
    cfg = (MOE_DYN.pick(shape_problem(SHAPE, args[1], "silu", op=MOE_DYN))["cfg"] if args[0]
           else "-")
    return dict(recv=count, cfg=cfg, iso=iso, e2e=e2e, late=late)


def _report(points, results, iters, graph):
    med = statistics.median
    call = ("CUDA-graph call: whole receive buffer, est = tokens" if graph
            else "eager call: the received rows, est = received rows")
    lines = [
        "",
        f"MiMo-V2.6-Flash(-RL) MoE layer, EP8 (H {H}, I {I}, {E_GLOBAL} experts, {E_LOCAL} per "
        f"rank, top-{TOPK}): MORI IntraNode dispatch (fp8, f32 scale per 128) -> a8w4_moe_dyn "
        f"({call}) -> MORI combine (bf16)",
        f"us, median of {iters} iterations. tokens: the batch, split evenly over the ranks. "
        "quant / dispatch / moe / combine: each stage timed alone (device barrier before it), "
        "slowest rank (moe: rank mean / slowest rank); sum = quant + dispatch + slowest moe + "
        "combine; e2e: the stages back to back, slowest rank.",
        "",
        "| tokens | tokens/rank | recv rows/rank mean / max | a8w4 cfg | quant | dispatch "
        "| moe mean / max | combine | sum | e2e |",
        "|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for ns, res in zip(points, results):
        stage = [{s: med([t[s] for t in r["iso"]]) for s in STAGES} for r in res]  # per rank
        worst = {s: max(st[s] for st in stage) for s in STAGES}
        moe_mean = statistics.fmean(st["moe"] for st in stage)
        e2e = med([max(r["e2e"][i] for r in res) for i in range(iters)])
        recv = [r["recv"] for r in res]
        cfgs = "/".join(sorted({r["cfg"] for r in res}))
        per = f"{min(ns)}" if min(ns) == max(ns) else f"{min(ns)}-{max(ns)}"
        lines.append(
            f"| {sum(ns)} | {per} | {statistics.fmean(recv):.0f} / {max(recv)} | {cfgs} "
            f"| {worst['quant']:.1f} | {worst['dispatch']:.1f} | {moe_mean:.1f} / "
            f"{worst['moe']:.1f} | {worst['combine']:.1f} | {sum(worst.values()):.1f} "
            f"| {e2e:.1f} |")
    print("\n".join(lines), flush=True)


# =============================================================================
# driver
# =============================================================================


def _worker(rank, port, points, graphs, iters, warmup):
    D.init(rank, port)
    layer = EpMoeLayer(rank)
    spin = _spin_cycles(SPIN_US)
    results = []
    for ns in points:
        x, tw, ti = R.make_routing(ns[rank], H, E_GLOBAL, TOPK, seed=7919 * rank + ns[rank])
        for graph in graphs:
            checked = _check(layer, ns, x, tw, ti, graph)
            if rank == 0:
                print(f"{sum(ns)} tokens {ns}, {'graph' if graph else 'eager'} MoE call: "
                      f"received rows {[c for c, _ in checked]}, cos diff vs reference <= "
                      f"{max(d for _, d in checked):.1e}", flush=True)
        if not iters:
            continue
        count = checked[rank][0]
        args = layer.moe_args(count, ns, graphs[0])
        res = [None] * WORLD
        dist.all_gather_object(res, _bench(layer, x, tw, ti, count, args, iters, warmup, spin))
        results.append(res)
        late = sum(r["late"] for r in res)
        if rank == 0 and late:
            print(f"  {late} of {2 * iters * WORLD} rank runs: host enqueue > the {SPIN_US} us "
                  "GPU spin (medians absorb a few)", flush=True)
    if results and rank == 0:
        _report(points, results, iters, graphs[0])
    D.finish()


def run(points, graphs=(False,), iters=0, warmup=3):
    """Spawn the 8 ranks. ``points``: tokens per rank, one list per point; every point is
    checked against the reference with each MoE call of ``graphs`` (False: sglang's eager
    call, True: its CUDA-graph call); ``iters`` > 0 also times it (one call; rank 0
    prints)."""
    assert not iters or len(graphs) == 1, graphs
    D.spawn(_worker, (points, graphs, iters, warmup))


def test_mori_a8w4_layer():
    reason = _skip_reason()
    if reason:
        pytest.skip(reason)
    run(CHECK_TOKENS, graphs=(False, True))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-t", "--tokens", type=int, nargs="+", default=BENCH_TOKENS,
                    help="tokens of the batch in total, split evenly over the 8 ranks, each "
                         f"<= {WORLD * MAX_TOKENS} (default: %(default)s)")
    ap.add_argument("--graph", action="store_true",
                    help="MoE with sglang's CUDA-graph call (whole receive buffer, est = "
                         "tokens) instead of its eager call (received rows)")
    ap.add_argument("--iters", type=int, default=20, help="timed iterations per point")
    ap.add_argument("--warmup", type=int, default=3)
    args = ap.parse_args()
    assert all(0 < n <= WORLD * MAX_TOKENS for n in args.tokens), args.tokens
    reason = _skip_reason()
    if reason:
        sys.exit(reason)
    run([split_tokens(n) for n in args.tokens], graphs=(args.graph,), iters=max(1, args.iters),
        warmup=args.warmup)
