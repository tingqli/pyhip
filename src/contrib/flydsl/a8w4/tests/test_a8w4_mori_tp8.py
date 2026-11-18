# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""``a8w4_moe`` on each rank's slice of moe_intermediate -> all-reduce on 8 GPUs: the TP8 MoE
layer of XiaomiMiMo/MiMo-V2.6-Flash(-RL) (hidden 4096, moe_intermediate 2048 = 8 x 256, 256
routed experts, top-8, SiLU) as sglang runs it with ``--tp 8`` (no expert parallelism, no
all-to-all); the TP counterpart of test_a8w4_mori_ep8.py.

Every rank holds the whole batch (TP attention replicates the hidden states and the routing)
and all experts with 1/8 of moe_intermediate. Per rank, as sglang's
``Mxfp4AiterA16W4MoEMethod.apply`` followed by ``tensor_model_parallel_all_reduce``:

    moe        a8w4_moe(x, W_rank, topk_weights, topk_ids): this rank's partial sum (the tuned
               table picks the TP kernels; x is quantized per row inside)
    allreduce  sum of the partials: aiter's custom all-reduce (sglang's default on ROCm, out of
               place; RCCL above its size limit) or torch.distributed.all_reduce (RCCL)

    pytest -q a8w4/tests/test_a8w4_mori_tp8.py     # correctness vs the fp32 reference
    python a8w4/tests/test_a8w4_mori_tp8.py        # perf: T = 64 .. 4096 tokens
    python a8w4/tests/test_a8w4_mori_tp8.py --act silu situv2 --allreduce torch --graph \\
        --model-dim 4096 --inter-dim 2048 --experts 256 --topk 8 -t 512 4096

Run from the directory that holds ``a8w4/``; needs 8 gfx942 GPUs.

Timing as in test_a8w4_mori_ep8.py: every iteration flushes L2 and uses the next of
``R.WCOPIES`` weight copies (the flush leaves re-read weights in the MALL; ``--graph``
captures a graph pair per copy), then a
host barrier, a GPU spin while the host enqueues the layer (host overhead excluded) and a
device-side barrier (a 1-element RCCL all-reduce), so all ranks start together. Each iteration
runs the layer twice: with a barrier before every stage (each stage timed alone) and back to
back (e2e of the slowest rank). ``--graph`` replays CUDA graphs captured as sglang captures
decode: graph-safe MoE configs, and aiter's all-reduce reads its (registered) input in place
instead of copying it into its own buffer first.
"""

import argparse
import contextlib
import itertools
import os
import statistics
import sys
import time

import pytest
import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from a8w4 import a8w4_moe  # noqa: E402
from a8w4 import ref as R  # noqa: E402
from a8w4.bench import dist as D  # noqa: E402
from a8w4.moe import MOE, shape_dims  # noqa: E402

WORLD = D.WORLD
MODEL = (4096, 2048, 256, 8)  # MiMo-V2.6-Flash: hidden, moe_intermediate, experts, top-k


def _set_model(model):
    """Set the layer shape every function reads; each rank holds I / WORLD of moe_intermediate."""
    global H, I, E, TOPK, I_LOCAL, SHAPE
    H, I, E, TOPK = model
    assert I % WORLD == 0, I
    I_LOCAL = I // WORLD
    SHAPE = (H, I_LOCAL, E, E, TOPK)


_set_model(MODEL)
COS_TOL = 5e-3
STAGES = ("moe", "allreduce")
BENCH_TOKENS = [64, 128, 256, 512, 1024, 2048, 4096]
# tp2k base (1), tp2k nt4 (77, 600), tp1kq eager / tp1k under capture (4096)
CHECK_TOKENS = [1, 77, 600, 4096]
SPIN_US = D.SPIN_US  # GPU spin while the host enqueues one layer

# shared with the other 8-GPU tests and experiments (a8w4/bench/dist.py); no MORI here:
# the device barrier is a 1-element RCCL all-reduce
_align = D.align_rccl
_spin_cycles = D.spin_cycles
_free_port = D.free_port


def _skip_reason():
    return D.skip_reason(need_mori=False)


class TpMoeLayer:
    """One rank: all experts with its slice of moe_intermediate (random mxfp4 weights) and the
    all-reduces of the MoE output."""

    def __init__(self, rank, allreduces):
        self.rank = rank
        self.W, self.raw = R.make_weights(E, H, I_LOCAL, seed=rank)
        self._Ws = [self.W]
        self.set_act("silu")
        self.ca = None
        if "aiter" in allreduces:
            from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce

            # As sglang's TP group: attached to the group's CPU (gloo) group, default limits.
            self.ca = CustomAllreduce(group=dist.new_group(backend="gloo"),
                                      device=torch.device("cuda", rank))
            assert not self.ca.disabled, "aiter custom all-reduce is disabled"

    def set_act(self, act):
        """Expert activation: silu, or situv2 with the test parameters (they do not change the
        timing)."""
        self.act, self.situ = act, (R.SITU if act == "situv2" else None)

    def moe(self, x, tw, ti):
        return a8w4_moe(x, self.W, tw, ti, act=self.act, situ=self.situ)

    def weights(self, n=R.WCOPIES):
        """``n`` copies of the weights (the original first; made once)."""
        if len(self._Ws) < n:
            self._Ws = R.weight_copies(self._Ws[0], n)
        return self._Ws[:n]

    def all_reduce(self, y, kind):
        """sglang ``GroupCoordinator.all_reduce``: ``kind`` "aiter" runs the custom all-reduce
        when the size fits (out of place), anything else RCCL in place -> the sum."""
        if kind == "aiter" and self.ca.should_custom_ar(y):
            return self.ca.custom_all_reduce(y)
        dist.all_reduce(y)
        return y

    def ar_path(self, y, kind):
        return "aiter custom" if kind == "aiter" and self.ca.should_custom_ar(y) else "rccl"

    def capture(self, x, tw, ti, kind, copies=1):
        """CUDA graphs of the MoE and of the all-reduce, captured as sglang captures decode (one
        graph per stage here, to time them), a pair per weight copy -> ([(moe graph,
        all-reduce graph)], the first pair's output)."""
        self.all_reduce(self.moe(x, tw, ti), kind)  # warm-up: compiles, allocates workspaces
        torch.cuda.synchronize()
        graphs, outs, W = [], [], self.W
        # aiter registers the captured inputs' addresses with its peers when capture() exits
        with self.ca.capture() if kind == "aiter" else contextlib.nullcontext():
            for Wc in self.weights(copies):
                self.W = Wc
                g_moe, g_ar = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
                with torch.cuda.graph(g_moe):
                    y = self.moe(x, tw, ti)
                with torch.cuda.graph(g_ar):
                    outs.append(self.all_reduce(y, kind))
                graphs.append((g_moe, g_ar))
        self.W = W
        torch.cuda.synchronize()
        return graphs, outs[0]


# =============================================================================
# correctness
# =============================================================================


def _reference(layer, x, tw, ti):
    """fp32 reference of the layer: each rank's slice, summed over the ranks."""
    ref = R.ref_moe(x, tw, ti, layer.raw, 0, layer.act, layer.situ)
    dist.all_reduce(ref)
    return ref


def _forward(layer, x, tw, ti, graph, kind, copies=1):
    """One layer forward -> (output, the graphs when ``graph``: captured, a pair per weight
    copy, then the first pair replayed)."""
    if not graph:
        return layer.all_reduce(layer.moe(x, tw, ti), kind), None
    graphs, z = layer.capture(x, tw, ti, kind, copies)
    for g in graphs[0]:
        g.replay()
    return z, graphs


def _check(ref, z, tokens, graph, kind):
    """Cos diff of every rank's output vs the reference, asserted on every rank -> the worst."""
    diff = R.cos_diff(ref, z.float())
    res = [None] * WORLD
    dist.all_gather_object(res, diff)
    assert all(d < COS_TOL for d in res), (
        f"{tokens} tokens, graph {graph}, all-reduce {kind}: cos diff per rank {res}")
    return max(res)


# =============================================================================
# timing
# =============================================================================


def _stages(layer, x, tw, ti, graphs, kind):
    """[(stage, fn)] of one layer forward: eager calls, or replays of the captured graphs."""
    if graphs:
        return [("moe", graphs[0].replay), ("allreduce", graphs[1].replay)]
    out = {}
    return [("moe", lambda: out.update(y=layer.moe(x, tw, ti))),
            ("allreduce", lambda: layer.all_reduce(out["y"], kind))]


def _timed(stages, isolated, spin):
    """One timed layer forward -> ({stage: us}, e2e us, host enqueue us)."""
    R.flush_cache()
    dist.barrier()
    torch.cuda._sleep(spin)
    _align()
    t0 = time.perf_counter()
    span = D.Spans(isolated, align=_align)
    for name, fn in stages:
        span(name)
        fn()
    span()
    host = (time.perf_counter() - t0) * 1e6
    torch.cuda.synchronize()
    return (*span.times(), host)


def _bench(layer, x, tw, ti, graphs, kind, iters, warmup, spin):
    """``graphs``: a (moe, all-reduce) pair per weight copy, or None (eager). Every timed
    forward takes the next copy."""
    if graphs:
        nxt = itertools.cycle([_stages(layer, x, tw, ti, g, kind) for g in graphs]).__next__
    else:
        stages, W = _stages(layer, x, tw, ti, None, kind), layer.W
        Ws = itertools.cycle(layer.weights())

        def nxt():
            layer.W = next(Ws)
            return stages
    for _ in range(warmup):
        for _, fn in nxt():
            fn()
    torch.cuda.synchronize()
    iso, e2e, late = [], [], 0
    for _ in range(iters):  # isolated and back-to-back runs interleaved: drift hits both
        t, _, h1 = _timed(nxt(), True, spin)
        _, t2, h2 = _timed(nxt(), False, spin)
        iso.append(t)
        e2e.append(t2)
        late += (h1 > SPIN_US) + (h2 > SPIN_US)  # the GPU may have waited for the host
    if not graphs:
        layer.W = W
    cfg = MOE.name_of(MOE.pick(shape_dims(SHAPE, x.shape[0], layer.act),
                               capturing=bool(graphs)))
    return dict(cfg=cfg, ar=layer.ar_path(x, kind), iso=iso, e2e=e2e, late=late)


def _e2e(res, iters):
    """Median over the iterations of the slowest rank's back-to-back layer time."""
    return statistics.median([max(r["e2e"][i] for r in res) for i in range(iters)])


def _report(tokens, results, iters, graph, kind, act):
    med = statistics.median
    lines = [
        "",
        f"TP8 MoE layer (H {H}, I {I} = {WORLD} x {I_LOCAL} per rank, {E} experts, top-{TOPK}, "
        f"{act}): a8w4_moe ({'CUDA graphs' if graph else 'eager'}) -> all-reduce "
        f"({'aiter custom' if kind == 'aiter' else 'torch'})",
        f"us, median of {iters} iterations. tokens: the batch, held by every rank. moe / allreduce:"
        " each stage timed alone (device barrier before it), slowest rank (moe: rank mean / "
        "slowest rank); sum = slowest moe + allreduce; e2e: the stages back to back, slowest rank.",
        "",
        "| tokens | a8w4 cfg | moe mean / max | allreduce | all-reduce path | sum | e2e |",
        "|---:|---|---:|---:|---|---:|---:|",
    ]
    for t, res in zip(tokens, results):
        stage = [{s: med([it[s] for it in r["iso"]]) for s in STAGES} for r in res]  # per rank
        worst = {s: max(st[s] for st in stage) for s in STAGES}
        moe_mean = statistics.fmean(st["moe"] for st in stage)
        lines.append(
            f"| {t} | {res[0]['cfg']} | {moe_mean:.1f} / {worst['moe']:.1f} "
            f"| {worst['allreduce']:.1f} | {res[0]['ar']} | {sum(worst.values()):.1f} "
            f"| {_e2e(res, iters):.1f} |")
    print("\n".join(lines), flush=True)


def _report_acts(tokens, results, iters):
    """SiTUv2 vs SiLU: the MoE stage (slowest rank) and the layer e2e."""
    lines = ["", "| tokens | moe silu | moe situv2 | ratio | e2e silu | e2e situv2 | ratio |",
             "|---:|---:|---:|---:|---:|---:|---:|"]
    for p, t in enumerate(tokens):
        a, b = results["silu"][p], results["situv2"][p]
        moe = [max(statistics.median(it["moe"] for it in r["iso"]) for r in x) for x in (a, b)]
        e2e = [_e2e(x, iters) for x in (a, b)]
        lines.append(f"| {t} | {moe[0]:.1f} | {moe[1]:.1f} | {moe[1] / moe[0]:.3f} "
                     f"| {e2e[0]:.1f} | {e2e[1]:.1f} | {e2e[1] / e2e[0]:.3f} |")
    print("\n".join(lines), flush=True)


# =============================================================================
# driver
# =============================================================================


def _worker(rank, port, tokens, graphs, iters, warmup, model, acts, allreduces):
    _set_model(model)
    D.init(rank, port, mori=False)
    layer = TpMoeLayer(rank, allreduces)
    spin = _spin_cycles(SPIN_US)
    results = {act: [] for act in acts}
    for t in tokens:
        x, tw, ti = R.make_routing(t, H, E, TOPK, seed=t)
        for buf in (x, tw, ti):  # TP: every rank holds the same batch and routing
            dist.broadcast(buf, 0)
        for act in acts:  # the activations alternate per point: drift hits all of them
            layer.set_act(act)
            ref = _reference(layer, x, tw, ti)
            timed = None
            for graph in graphs:
                for kind in allreduces:
                    timing = iters and (graph, kind) == (graphs[0], allreduces[0])
                    z, captured = _forward(layer, x, tw, ti, graph, kind,
                                           R.WCOPIES if timing else 1)
                    diff = _check(ref, z, t, graph, kind)
                    if rank == 0:
                        print(f"{t} tokens, {act}, {'graph' if graph else 'eager'}, all-reduce "
                              f"{layer.ar_path(x, kind)}: cos diff vs reference <= {diff:.1e}",
                              flush=True)
                    if timing:
                        timed = captured
                    del captured
            del ref
            if not iters:
                continue
            res = [None] * WORLD
            dist.all_gather_object(
                res, _bench(layer, x, tw, ti, timed, allreduces[0], iters, warmup, spin))
            results[act].append(res)
            del timed
            late = sum(r["late"] for r in res)
            if rank == 0 and late:
                print(f"  {late} of {2 * iters * WORLD} rank runs: host enqueue > the {SPIN_US} "
                      "us GPU spin (medians absorb a few)", flush=True)
    if iters and rank == 0:
        for act in acts:
            _report(tokens, results[act], iters, graphs[0], allreduces[0], act)
        if set(acts) == {"silu", "situv2"}:
            _report_acts(tokens, results, iters)
    D.finish()


def run(tokens, graphs=(False,), iters=0, warmup=3, model=MODEL, acts=("silu",),
        allreduces=("aiter",)):
    """Spawn the 8 ranks. Every token count is checked against the reference with each mode of
    ``graphs`` (False: eager, True: CUDA graphs) and each all-reduce of ``allreduces`` ("aiter",
    "torch"); ``iters`` > 0 also times it (one mode and one all-reduce; rank 0 prints)."""
    assert not iters or (len(graphs) == 1 and len(allreduces) == 1), (graphs, allreduces)
    D.spawn(_worker, (list(tokens), tuple(graphs), iters, warmup, tuple(model), tuple(acts),
                      tuple(allreduces)))


def test_a8w4_tp8_layer():
    reason = _skip_reason()
    if reason:
        pytest.skip(reason)
    run(CHECK_TOKENS, graphs=(False, True), allreduces=("aiter", "torch"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-t", "--tokens", type=int, nargs="+", default=BENCH_TOKENS,
                    help="tokens of the batch (every rank holds all of them) "
                         "(default: %(default)s)")
    ap.add_argument("--allreduce", default="aiter", choices=["aiter", "torch"],
                    help="aiter: aiter's custom all-reduce, sglang's default on ROCm (RCCL "
                         "above its size limit); torch: torch.distributed.all_reduce (RCCL)")
    ap.add_argument("--graph", action="store_true",
                    help="replay CUDA graphs captured as sglang captures decode instead of "
                         "eager calls")
    ap.add_argument("--act", nargs="+", default=["silu"], choices=["silu", "situv2"],
                    help="expert activation(s); both: side-by-side comparison")
    ap.add_argument("--model-dim", type=int, default=MODEL[0])
    ap.add_argument("--inter-dim", type=int, default=MODEL[1],
                    help="moe_intermediate_size of the model; each rank holds 1/8 of it")
    ap.add_argument("--experts", type=int, default=MODEL[2], help="routed experts")
    ap.add_argument("--topk", type=int, default=MODEL[3])
    ap.add_argument("--iters", type=int, default=20, help="timed iterations per point")
    ap.add_argument("--warmup", type=int, default=3)
    args = ap.parse_args()
    assert all(n > 0 for n in args.tokens), args.tokens
    reason = _skip_reason()
    if reason:
        sys.exit(reason)
    run(args.tokens, graphs=(args.graph,), iters=max(1, args.iters), warmup=args.warmup,
        model=(args.model_dim, args.inter_dim, args.experts, args.topk), acts=args.act,
        allreduces=(args.allreduce,))
