# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""megamoe_ep_a8w4 on 8 GPUs: the EP8 MoE layer of XiaomiMiMo/MiMo-V2.6-Flash(-RL)
(hidden 4096, moe_intermediate 2048, 256 routed experts, top-8; 32 experts per rank) as one
persistent kernel (``a8w4.megamoe.MegaMoeEpA8W4``), against the fp32 reference and, for
speed, against MORI dispatch -> ``a8w4_moe_dyn`` -> MORI combine (``test_a8w4_mori_ep8``).

A batch of T tokens (the model's total) is split evenly over the ranks (DP attention).

    pytest -q a8w4/tests/test_a8w4_megamoe_ep8.py           # correctness vs the fp32 reference
    pytest -q a8w4/tests/test_a8w4_megamoe_ep8.py -k graph  # only the CUDA graph replay
    pytest -q a8w4/tests/test_a8w4_megamoe_ep8.py -k small  # only MegaMoeSmallEpA8W4
    python a8w4/tests/test_a8w4_megamoe_ep8.py              # perf: T = 2048 .. 32768, vs MORI
    python a8w4/tests/test_a8w4_megamoe_ep8.py -t 8192 --iters 50 --act situv2

Run from the directory that holds ``a8w4/``; needs 8 gfx942 GPUs and mori.

Correctness: every point runs twice, the points interleaved (stale data from an earlier
call would show up); each rank's output vs the reference (each rank evaluates its experts
on all ranks' fp8 tokens, summed over the ranks), cos diff < 5e-3; with both fp8 x quants
(``x_quant``: one scale per (token, 128 columns) or per token).
CUDA graph replay (``test_megamoe_ep8_graph``): two graphs (96 and 1536 tokens per rank;
the second runs the big-batch W cache variant, its experts span several blocks) captured
after one eager call each, as sglang's graph runner does, replayed several times in a row,
the other graph and eager calls in between. Every call gets new inputs padded as sglang
pads a batch: the rows past the real tokens (all of them on an idle rank) and some single
slots have topk_ids -1; uniform and hot-expert routing alternate, so the row layout
changes from call to call. Each call vs the reference, padded rows exactly 0; a call
unfinished after ``GRAPH_TIMEOUT_S`` (a cross-rank wait that never ends) fails the test
instead of hanging it.
Timing (as test_a8w4_mori_ep8): every iteration flushes L2 and takes the next of
``R.WCOPIES`` weight copies (the flush leaves re-read weights in the MALL), then a host
barrier, a GPU spin while the host enqueues the layer, and a MORI device barrier, so all
ranks start together; e2e = the layer of the slowest rank, median over the iterations; the
two paths alternate per iteration. Stage split of megamoe (events between its launches,
slowest rank): quant / prep (count exchange) / plan / mega / finish (wait for all ranks).

The small-batch layer ``a8w4.megamoe.MegaMoeSmallEpA8W4`` (one kernel per call, <= 1024
tokens over all ranks): ``test_megamoe_small_ep8`` (eager: uneven batches with idle ranks
and a single token, dropped slots, both routings, m-tiles of 32 and 64 rows, both x
quants) and ``test_megamoe_small_ep8_graph`` (the replay schedule above, with batches of
its range; the two graphs use different x quants).
The prefill layer ``a8w4.megamoe.MegaMoeLargeEpA8W4`` (``-k large``): the same two tests
(eager also at the full capacity, both combines, both x quants; then rows past its receive
capacity, which must be dropped).
"""

import argparse
import functools
import itertools
import os
import sys
import time

import pytest
import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from a8w4 import ref as R  # noqa: E402
from a8w4.bench import dist as D  # noqa: E402
from a8w4.bench.dist import E_GLOBAL, E_LOCAL, H, I, TOPK, WORLD  # noqa: E402,F401

MAX_TOKENS = 4096  # per rank (the MORI path's dispatch capacity)
COS_TOL = D.COS_TOL
BENCH_TOKENS = [2048, 4096, 8192, 16384, 32768]  # T, the total over the ranks
# tokens per rank of the correctness test: uniform, an uneven batch (idle ranks, a single
# token) and the full capacity
CHECK_TOKENS = [[256] * WORLD, [0, 1, 37, 300, 5, 128, 999, 0], [MAX_TOKENS] * WORLD]
# CUDA graph replay: tokens per rank of the two graphs (a, b) and of the eager calls (e);
# the steps: consecutive replays of one graph (what a launch-argument epoch broke), every
# transition between a, b and e
GRAPH_TOKENS = {"a": 96, "b": 1536, "e": 300}
GRAPH_STEPS = "aaaebbbeabba"
GRAPH_TIMEOUT_S = 60
SPIN_US = D.SPIN_US
STAGES = D.MEGA_STAGES

# shared with the other 8-GPU tests and experiments (a8w4/bench/dist.py); the old names
# stay for the archived experiment scripts that import them
_skip_reason = D.skip_reason
split_tokens = D.split_tokens
_check = D.check_megamoe
_spin_cycles = D.spin_cycles
_timed = D.timed
_free_port = D.free_port


def _worker(rank, port, points, iters, warmup, acts):
    D.init(rank, port)
    from a8w4.megamoe import MegaMoeEpA8W4

    W, raw = R.make_weights(E_LOCAL, H, I, seed=rank)
    max_tokens = max(max(ns) for ns in points)
    megas = {act: MegaMoeEpA8W4(W, rank=rank, max_tokens=max_tokens, topk=TOPK, act=act,
                                situ=R.SITU if act == "situv2" else None) for act in acts}
    # the correctness check also runs every act with one x scale per row
    checked = [(act, "per_1x128", m) for act, m in megas.items()] + [
        (act, "per_token", MegaMoeEpA8W4(W, rank=rank, max_tokens=max_tokens, topk=TOPK, act=act,
                                         situ=R.SITU if act == "situv2" else None,
                                         x_quant="per_token")) for act in acts]
    data = [R.make_routing(ns[rank], H, E_GLOBAL, TOPK, seed=7919 * rank + ns[rank])
            for ns in points]
    # correctness: every point twice, interleaved
    for rep in range(2):
        for ns, (x, tw, ti) in zip(points, data):
            for act, xq, mega in checked:
                res = _check(mega, raw, ns, x, tw, ti, act, R.SITU if act == "situv2" else None)
                if rank == 0:
                    print(f"{sum(ns)} tokens {ns} ({act}, {xq}, call {rep + 1}): rows needed "
                          f"{[r for r, _ in res]} (cap {mega.s_cap}), cos diff vs reference <= "
                          f"{max(d for _, d in res):.1e}", flush=True)
    if iters:
        _bench(rank, points, data, megas, W, iters, warmup)
    D.finish()


def _bench(rank, points, data, megas, W, iters, warmup):
    """e2e of megamoe (+ its stages) and of the MORI path, alternating per iteration."""
    spin = _spin_cycles(SPIN_US)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import test_a8w4_mori_ep8 as M

    layer = M.EpMoeLayer(rank)  # also provides the device barrier (_align)
    nxt = itertools.cycle(R.weight_copies(W)).__next__  # both paths: the same experts
    rows = []
    for ns, (x, tw, ti) in zip(points, data):
        for act, mega in megas.items():
            marks = []
            run_mega = lambda: mega(x, ti, tw, marks=marks)  # noqa: E731
            if hasattr(layer, "set_act"):
                layer.set_act(act)
            layer.W = W
            xq, xs = layer.quant(x)
            recv = layer.dispatch(xq, xs, tw, ti)
            count = int(recv[4].item())
            layer.combine(layer.moe(recv, count, max(1, count)), ti)
            run_mori = lambda: layer.combine(  # noqa: E731
                layer.moe(layer.dispatch(*layer.quant(x), tw, ti), count, max(1, count)), ti)
            for _ in range(warmup):
                run_mega()
                marks.clear()
                run_mori()
            torch.cuda.synchronize()
            t_mega, t_mori, st = [], [], []
            for _ in range(iters):
                mega.W = nxt()
                t, s = _timed(run_mega, spin, marks)
                t_mega.append(t)
                st.append(s)
                layer.W = nxt()
                t_mori.append(_timed(run_mori, spin)[0])
            res = [None] * WORLD
            dist.all_gather_object(res, (t_mega, t_mori, st, mega.rows_needed()))
            if rank == 0:
                e2e = D.slowest_median([r[0] for r in res])
                mori_e2e = D.slowest_median([r[1] for r in res])
                stages = D.stage_max([r[2] for r in res], STAGES)
                rows.append((sum(ns), act, e2e, mori_e2e, stages, max(r[3] for r in res)))
                print(f"T {sum(ns)} {act}: megamoe {e2e:.1f} us, MORI path {mori_e2e:.1f} us",
                      flush=True)
    if rank == 0:
        lines = [
            "",
            f"EP8 MoE layer (H {H}, I {I}, {E_GLOBAL} experts, {E_LOCAL} per rank, top-{TOPK}), "
            f"us, slowest rank, median of {iters} iterations. MORI path: quant -> MORI dispatch "
            "-> a8w4_moe_dyn (received rows) -> MORI combine; megamoe stages: slowest rank's "
            "median each.",
            "",
            "| tokens | tokens/rank | act | MORI path | megamoe | speedup | quant | prep | plan "
            "| mega | finish | rows/rank max |",
            "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for T, act, e2e, mori_e2e, st, rmax in rows:
            lines.append(
                f"| {T} | {T // WORLD} | {act} | {mori_e2e:.1f} | {e2e:.1f} | {mori_e2e / e2e:.2f}x | "
                + " | ".join(f"{st[s]:.1f}" for s in STAGES) + f" | {rmax} |")
        print("\n".join(lines), flush=True)


def run(points, iters=0, warmup=3, acts=("silu",)):
    """Spawn the 8 ranks: check every point (tokens per rank, one list per point) against
    the reference; ``iters`` > 0 also times it against the MORI path (rank 0 prints)."""
    D.spawn(_worker, (points, iters, warmup, acts))


def test_megamoe_ep8():
    reason = _skip_reason()
    if reason:
        pytest.skip(reason)
    run(CHECK_TOKENS)


def _valid_tokens(n, step, rank):
    """Real tokens of a call padded to ``n``: k n / 6 (k = 0: an idle rank), varying over
    the ranks and, in total, over the steps."""
    return n * ((5 * step + 3 * rank) % 7) // 6


def _padded_routing(n, n_valid, seed, routing="uniform"):
    """Inputs of an ``n``-token call with ``n_valid`` real tokens, padded as sglang pads a
    batch: topk_ids -1 in the rows past them (``num_token_non_padded``), and in about 1/16
    of the real rows' slots as well (single dropped routes)."""
    x, tw, ti = R.make_routing(n, H, E_GLOBAL, TOPK, seed=seed, routing=routing)
    gen = torch.Generator(device="cuda").manual_seed(seed)
    drop = torch.rand(ti.shape, device="cuda", generator=gen) < 1 / 16
    drop[n_valid:] = True
    return x, tw, ti.masked_fill(drop, -1)


def _bounded(fn, *args):
    """fn(*args), then wait for its GPU work at most ``GRAPH_TIMEOUT_S``. Past that, this
    rank exits at once: a hung kernel blocks a normal exit as well, while a dead rank makes
    the spawn stop the others and fail."""
    out = fn(*args)
    done = torch.cuda.Event()
    done.record()
    t0 = time.monotonic()
    while not done.query():
        if time.monotonic() - t0 > GRAPH_TIMEOUT_S:
            print(f"rank {torch.cuda.current_device()}: megamoe call unfinished after "
                  f"{GRAPH_TIMEOUT_S} s, a cross-rank wait never ended", file=sys.stderr,
                  flush=True)
            os._exit(1)
        time.sleep(1e-3)
    return out


def _replay(graph, out):
    graph.replay()
    return out


def _graph_worker(rank, port, act):
    D.init(rank, port)
    from a8w4.megamoe import MegaMoeEpA8W4

    situ = R.SITU if act == "situv2" else None
    W, raw = R.make_weights(E_LOCAL, H, I, seed=rank)
    mega = MegaMoeEpA8W4(W, rank=rank, max_tokens=MAX_TOKENS, topk=TOPK, act=act, situ=situ)
    graphs = {}
    for kind in "ab":
        n = GRAPH_TOKENS[kind]
        x, tw, ti = _padded_routing(n, n, seed=rank)  # the graph's static inputs
        mega(x, ti, tw)  # warm-up (compiles), as sglang's graph runner does before capture
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            y = mega(x, ti, tw)
        graphs[kind] = (g, y, (x, tw, ti))
    for step, kind in enumerate(GRAPH_STEPS):
        n = GRAPH_TOKENS[kind]
        n_valid = _valid_tokens(n, step, rank)
        x, tw, ti = _padded_routing(n, n_valid, seed=7919 * rank + 31 * step + 1,
                                    routing=R.ROUTINGS[step % 2])
        if kind == "e":
            call = functools.partial(_bounded, mega, x, ti, tw)
        else:
            g, y, static = graphs[kind]
            for dst, src in zip(static, (x, tw, ti)):
                dst.copy_(src)
            x, tw, ti = static
            call = functools.partial(_bounded, _replay, g, y)
        res = _check(mega, raw, [n] * WORLD, x, tw, ti, act, situ, run=call)
        assert not mega.out[n_valid:n].any(), f"rank {rank} step {step}: padded rows not 0"
        if rank == 0:
            what = "eager call" if kind == "e" else f"replay of graph {kind}"
            print(f"step {step} {what}, {n} tokens per rank, {R.ROUTINGS[step % 2]} routing, "
                  f"real {[_valid_tokens(n, step, r) for r in range(WORLD)]}: rows needed "
                  f"{[r for r, _ in res]}, cos diff vs reference <= "
                  f"{max(d for _, d in res):.1e}", flush=True)
    D.finish()


def test_megamoe_ep8_graph():
    reason = _skip_reason()
    if reason:
        pytest.skip(reason)
    D.spawn(_graph_worker, ("silu",))


# ---- MegaMoeSmallEpA8W4 --------------------------------------------------------------

SMALL_TMAX = 1024  # tokens over all ranks
# tokens per rank: decode-sized, an uneven batch (idle ranks, a single token; 570 in all),
# one rank alone, every rank idle, the full fixed slots
SMALL_CHECK_TOKENS = [[8] * WORLD, [0, 1, 37, 300, 5, 128, 99, 0], [1] + [0] * (WORLD - 1),
                      [0] * WORLD, [SMALL_TMAX // WORLD] * WORLD]
SMALL_GRAPH_TOKENS = {"a": 12, "b": 96, "e": 40}
SMALL_GRAPH_XQ = {"a": "per_1x128", "b": "per_token", "e": "per_token"}


def _check_small(m, raw, ns, x, tw, ti, act, situ, run, x_quant="per_1x128", **knobs):
    """One small-layer call (``run()``) vs the fp32 reference on every rank's fp8 tokens
    (quantized per ``x_quant`` as the kernel does); asserts on every rank."""
    rank, n = m.rank, ns[m.rank]
    y = run()[:n].float()
    xq = R.quant_x(x, x_quant) if n else x.float()
    X, TW, TI = (D.gather_rows(t, ns) for t in (xq, tw, ti))
    ref = R.ref_moe(X, TW, TI, raw, rank * E_LOCAL, act, situ, quant_x=False)
    dist.all_reduce(ref)
    off = sum(ns[:rank])
    diff = R.cos_diff(ref[off:off + n], y) if n else 0.0
    res = [None] * len(ns)
    dist.all_gather_object(res, diff)
    bad = [(r, d) for r, d in enumerate(res) if not d < COS_TOL]
    assert not bad, f"tokens per rank {ns} {x_quant} {knobs}: (rank, cos diff) {bad}"
    return max(res)


def _small_worker(rank, port, act):
    D.init(rank, port)
    from a8w4.megamoe import MegaMoeSmallEpA8W4

    situ = R.SITU if act == "situv2" else None
    W, raw = R.make_weights(E_LOCAL, H, I, seed=rank)
    m = MegaMoeSmallEpA8W4(W, rank=rank, tmax=SMALL_TMAX, topk=TOPK)
    for rep in range(2):
        for i, ns in enumerate(SMALL_CHECK_TOKENS):
            routing = R.ROUTINGS[(rep + i) % 2]
            seed = 7919 * rank + 31 * i + rep
            if rep:  # about 1/16 of the slots dropped (topk_ids -1)
                x, tw, ti = _padded_routing(ns[rank], ns[rank], seed=seed, routing=routing)
            else:
                x, tw, ti = R.make_routing(ns[rank], H, E_GLOBAL, TOPK, seed=seed,
                                           routing=routing)
            for bm, xq in itertools.product((32, 64), R.X_QUANTS):
                call = functools.partial(_bounded, functools.partial(
                    m, x, ti, tw, act=act, situ=situ, x_quant=xq, bm=bm))
                d = _check_small(m, raw, ns, x, tw, ti, act, situ, call, x_quant=xq, bm=bm)
                if rank == 0:
                    print(f"small {act} bm{bm} {xq} {routing}, tokens per rank {ns} (call "
                          f"{rep + 1}): cos diff vs reference <= {d:.1e}", flush=True)
    D.finish()


def test_megamoe_small_ep8():
    reason = _skip_reason()
    if reason:
        pytest.skip(reason)
    D.spawn(_small_worker, ("silu",))


def _small_graph_worker(rank, port, act):
    D.init(rank, port)
    from a8w4.megamoe import MegaMoeSmallEpA8W4

    situ = R.SITU if act == "situv2" else None
    W, raw = R.make_weights(E_LOCAL, H, I, seed=rank)
    m = MegaMoeSmallEpA8W4(W, rank=rank, tmax=SMALL_TMAX, topk=TOPK)
    graphs = {}
    for kind in "ab":
        n, xq = SMALL_GRAPH_TOKENS[kind], SMALL_GRAPH_XQ[kind]
        x, tw, ti = _padded_routing(n, n, seed=rank)  # the graph's static inputs
        m(x, ti, tw, act=act, situ=situ, x_quant=xq)  # warm-up (compiles), then capture
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            y = m(x, ti, tw, act=act, situ=situ, x_quant=xq)
        graphs[kind] = (g, y, (x, tw, ti))
    for step, kind in enumerate(GRAPH_STEPS):
        n, xq = SMALL_GRAPH_TOKENS[kind], SMALL_GRAPH_XQ[kind]
        n_valid = _valid_tokens(n, step, rank)
        x, tw, ti = _padded_routing(n, n_valid, seed=7919 * rank + 31 * step + 1,
                                    routing=R.ROUTINGS[step % 2])
        if kind == "e":
            call = functools.partial(
                _bounded, functools.partial(m, x, ti, tw, act=act, situ=situ, x_quant=xq))
        else:
            g, y, static = graphs[kind]
            for dst, src in zip(static, (x, tw, ti)):
                dst.copy_(src)
            x, tw, ti = static
            call = functools.partial(_bounded, _replay, g, y)
        d = _check_small(m, raw, [n] * WORLD, x, tw, ti, act, situ, call, x_quant=xq)
        assert not m.out[n_valid:n].any(), f"rank {rank} step {step}: padded rows not 0"
        if rank == 0:
            what = "eager call" if kind == "e" else f"replay of graph {kind}"
            print(f"small step {step} {what}, {n} tokens per rank, {R.ROUTINGS[step % 2]} "
                  f"routing, real {[_valid_tokens(n, step, r) for r in range(WORLD)]}: cos "
                  f"diff vs reference <= {d:.1e}", flush=True)
    D.finish()


def test_megamoe_small_ep8_graph():
    reason = _skip_reason()
    if reason:
        pytest.skip(reason)
    D.spawn(_small_graph_worker, ("silu",))


# ---- MegaMoeLargeEpA8W4 --------------------------------------------------------------

LARGE_MAX_TOKENS = 4096  # per rank
# tokens per rank: uneven (idle ranks, a single token), every rank idle, one rank alone,
# prefill-sized, the full capacity
LARGE_CHECK_TOKENS = [[0, 1, 37, 300, 5, 128, 999, 0], [0] * WORLD, [1] + [0] * (WORLD - 1),
                      [1024] * WORLD, [LARGE_MAX_TOKENS] * WORLD]
LARGE_KNOB_SETS = (dict(), dict(x_quant="per_token"),
                   dict(comb="fp8", bcm1=0, bcm2=1, producers=32),
                   dict(comb="fp8", x_quant="per_token"))
LARGE_GRAPH_TOKENS = {"a": 96, "b": 1536, "e": 300}
LARGE_GRAPH_KNOBS = {"a": dict(), "b": dict(comb="fp8", bcm1=0, bcm2=1, x_quant="per_token"),
                     "e": dict(x_quant="per_token")}


def _check_large(m, raw, ns, x, tw, ti, act, situ, run, **knobs):
    """One large-layer call (``run()``) vs the fp32 reference on every rank's fp8 tokens as
    the kernel quantized them (``m.x_ref``); asserts on every rank."""
    rank, n = m.rank, ns[m.rank]
    y = run()[:n].float()
    xq = m.x_ref(n) if n else x.float()
    X, TW, TI = (D.gather_rows(t, ns) for t in (xq, tw, ti))
    ref = R.ref_moe(X, TW, TI, raw, rank * E_LOCAL, act, situ, quant_x=False)
    dist.all_reduce(ref)
    off = sum(ns[:rank])
    diff = R.cos_diff(ref[off:off + n], y) if n else 0.0
    res = [None] * len(ns)
    dist.all_gather_object(res, (m.rows_needed(), diff))
    bad = [(r, *v) for r, v in enumerate(res) if v[0] > m.s_cap or not v[1] < COS_TOL]
    assert not bad, f"tokens per rank {ns} {knobs}: (rank, rows needed, cos diff) {bad}"
    return max(d for _, d in res)


def _check_large_overflow(m, raw, act, situ, **knobs):
    """Rows past s_cap: every rank's LARGE_MAX_TOKENS / 2 tokens send their TOPK slots to
    rank 0's experts 0 .. TOPK - 1 (a permutation per token), 8 x the uniform average. Rank
    0's layout: expert e's rows from e WORLD n, source s's from + s n (sources from rank 0
    on); rows past s_cap are dropped, here whole (source, expert) segments -> the reference
    without those routes (they must add nothing: not the rows of an earlier call)."""
    rank, n = m.rank, LARGE_MAX_TOKENS // 2
    x, tw, _ = R.make_routing(n, H, E_GLOBAL, TOPK, seed=97 * rank + 5)
    g = torch.Generator(device="cuda").manual_seed(rank)
    ti = torch.argsort(torch.rand((n, TOPK), device="cuda", generator=g), dim=1).to(torch.int32)
    starts = torch.arange(TOPK * WORLD).view(TOPK, WORLD).T * n  # [source, expert]
    assert ((starts + n <= m.s_cap) | (starts >= m.s_cap)).all(), "a segment cut"
    kept = (starts + n <= m.s_cap).cuda()
    ns = [n] * WORLD
    y = _bounded(functools.partial(m, x, ti, tw, act=act, situ=situ, **knobs))[:n].float()
    X, TW, TI = (D.gather_rows(t, ns) for t in (m.x_ref(n), tw, ti))
    src = torch.arange(WORLD, device="cuda").repeat_interleave(n).unsqueeze(1)
    TI = TI.masked_fill(~kept[src, TI.long()], -1)
    ref = R.ref_moe(X, TW, TI, raw, rank * E_LOCAL, act, situ, quant_x=False)
    dist.all_reduce(ref)
    d = R.cos_diff(ref[rank * n:(rank + 1) * n], y)
    res = [None] * WORLD
    dist.all_gather_object(res, (m.rows_needed(), d))
    assert res[0][0] == WORLD * TOPK * n and all(r[1] < COS_TOL for r in res), (
        f"overflow {knobs}: (rows needed, cos diff) per rank {res} (cap {m.s_cap})")
    return max(r[1] for r in res)


def _large_worker(rank, port, act):
    D.init(rank, port)
    from a8w4.megamoe import MegaMoeLargeEpA8W4

    situ = R.SITU if act == "situv2" else None
    W, raw = R.make_weights(E_LOCAL, H, I, seed=rank)
    m = MegaMoeLargeEpA8W4(W, rank=rank, max_tokens=LARGE_MAX_TOKENS, topk=TOPK)
    for rep in range(2):
        for i, ns in enumerate(LARGE_CHECK_TOKENS):
            routing = R.ROUTINGS[(rep + i) % 2]
            seed = 7919 * rank + 31 * i + rep
            if rep:  # about 1/16 of the slots dropped (topk_ids -1)
                x, tw, ti = _padded_routing(ns[rank], ns[rank], seed=seed, routing=routing)
            else:
                x, tw, ti = R.make_routing(ns[rank], H, E_GLOBAL, TOPK, seed=seed,
                                           routing=routing)
            for knobs in LARGE_KNOB_SETS:
                call = functools.partial(
                    _bounded, functools.partial(m, x, ti, tw, act=act, situ=situ, **knobs))
                d = _check_large(m, raw, ns, x, tw, ti, act, situ, call, **knobs)
                if rank == 0:
                    print(f"large {act} {knobs} {routing}, tokens per rank {ns} (call {rep + 1})"
                          f": cos diff vs reference <= {d:.1e}", flush=True)
    for knobs in LARGE_KNOB_SETS[0::2]:  # both combines, their slots written by the calls above
        d = _check_large_overflow(m, raw, act, situ, **knobs)
        if rank == 0:
            print(f"large {act} {knobs}, rows past s_cap dropped: cos diff vs reference <= "
                  f"{d:.1e}", flush=True)
    D.finish()


def test_megamoe_large_ep8():
    reason = _skip_reason()
    if reason:
        pytest.skip(reason)
    D.spawn(_large_worker, ("silu",))


def _large_graph_worker(rank, port, act):
    D.init(rank, port)
    from a8w4.kernels._protocol import qx_of
    from a8w4.megamoe import MegaMoeLargeEpA8W4

    situ = R.SITU if act == "situv2" else None
    W, raw = R.make_weights(E_LOCAL, H, I, seed=rank)
    m = MegaMoeLargeEpA8W4(W, rank=rank, max_tokens=LARGE_MAX_TOKENS, topk=TOPK)
    knobs = LARGE_GRAPH_KNOBS
    graphs = {}
    for kind in "ab":
        n = LARGE_GRAPH_TOKENS[kind]
        x, tw, ti = _padded_routing(n, n, seed=rank)  # the graph's static inputs
        m(x, ti, tw, act=act, situ=situ, **knobs[kind])  # warm-up (compiles), then capture
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            y = m(x, ti, tw, act=act, situ=situ, **knobs[kind])
        graphs[kind] = (g, y, (x, tw, ti))
    for step, kind in enumerate(GRAPH_STEPS):
        n = LARGE_GRAPH_TOKENS[kind]
        n_valid = _valid_tokens(n, step, rank)
        x, tw, ti = _padded_routing(n, n_valid, seed=7919 * rank + 31 * step + 1,
                                    routing=R.ROUTINGS[step % 2])
        if kind == "e":
            call = functools.partial(
                _bounded, functools.partial(m, x, ti, tw, act=act, situ=situ, **knobs[kind]))
        else:
            g, y, static = graphs[kind]
            for dst, src in zip(static, (x, tw, ti)):
                dst.copy_(src)
            x, tw, ti = static
            m._qx = qx_of(knobs[kind].get("x_quant", "per_1x128"))  # the replay's (x_ref)
            call = functools.partial(_bounded, _replay, g, y)
        d = _check_large(m, raw, [n] * WORLD, x, tw, ti, act, situ, call)
        assert not m.out[n_valid:n].any(), f"rank {rank} step {step}: padded rows not 0"
        if rank == 0:
            what = "eager call" if kind == "e" else f"replay of graph {kind}"
            print(f"large step {step} {what}, {n} tokens per rank, {R.ROUTINGS[step % 2]} "
                  f"routing, real {[_valid_tokens(n, step, r) for r in range(WORLD)]}: cos "
                  f"diff vs reference <= {d:.1e}", flush=True)
    D.finish()


def test_megamoe_large_ep8_graph():
    reason = _skip_reason()
    if reason:
        pytest.skip(reason)
    D.spawn(_large_graph_worker, ("silu",))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-t", "--tokens", type=int, nargs="+", default=BENCH_TOKENS,
                    help="tokens of the batch in total, split evenly over the 8 ranks, each "
                         f"<= {WORLD * MAX_TOKENS} (default: %(default)s)")
    ap.add_argument("--act", nargs="+", default=["silu"], choices=R.ACTS)
    ap.add_argument("--iters", type=int, default=20, help="timed iterations per point (0: check only)")
    ap.add_argument("--warmup", type=int, default=3)
    args = ap.parse_args()
    assert all(0 < n <= WORLD * MAX_TOKENS for n in args.tokens), args.tokens
    reason = _skip_reason()
    if reason:
        sys.exit(reason)
    run([split_tokens(n) for n in args.tokens], iters=args.iters, warmup=args.warmup,
        acts=tuple(args.act))
