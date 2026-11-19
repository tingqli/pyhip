"""Phase + per-unit timeline of MegaMoeLargeEpA8W4 (kernel knob tl=1), cold calls.

    python a8w4/experiments/20261118_mega_large/timeline_large.py -t 8192 [-k producers=32 ...]
        [--act silu] [--rank 0] [--iters 3]

Per rank (us from the rank's earliest workgroup start; s_memrealtime 100 MHz): the plan,
producers (scan end, send start / end), quant end, unit loop start / end, the finish; per
unit kind (gemm1 / gemm2): duration, the part before the K loop (incl. the wait), the gap to
the workgroup's next unit; units per workgroup. Every call is cold (L2 flush + the next of
8 weight copies, all ranks aligned).

The landed kernel has no TL knob: ``pre_cleanup.diff`` (in this dir) turns the final
``kernels/ep_mega_large.py`` back into the instrumented experiment kernel (``patch
kernels/ep_mega_large.py < pre_cleanup.diff``; it also has the rejected LAG / LCOMB / COMB
1 options, COMB as 0 / 1 / 2). Its host side (not archived, re-add by hand):
``_protocol.large_ctl_ints(TOPK, TCAP, tl_words=0)`` returned ``(LC_G1, LC_TL, ints)`` with
``LC_TL = LC_G1 + 16 * tiles`` and ``ints = LC_TL + tl_words``; ``cin`` was ``TCAP * TOPK *
H * 2`` bytes (no ``cins`` region: COMB 2 put the scales after its fp8 rows);
``MegaMoeLargeEpA8W4`` allocated ``large_ctl_ints(topk, tcap, NUM_CU * KL.TLW)``, had the
knobs ``lag=0, comb=0, tl=0, lcomb=0`` (keys LAG, COMB, TL, LCOMB) and
``timeline() = ctl[lc_tl:lc_tl + NUM_CU * TLW].view(int64).view(NUM_CU, TLW // 2)``.
The timelines of ``../log_20261118c.md`` ran one x scale per row (QX 0, then the default):
``-k x_quant=per_token``.
"""
import argparse
import itertools
import os
import statistics
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
from a8w4 import ref as R  # noqa: E402
from a8w4.bench import dist as D  # noqa: E402
from a8w4.bench.dist import E_GLOBAL, E_LOCAL, H, I, TOPK, WORLD  # noqa: E402


def med(v):
    return statistics.median(v) if v else float("nan")


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(q * len(v)))] if v else float("nan")


def analyse(t, P, u1):
    """t: int64 [grid, TLW/2] stamps of one call -> dict of numbers (us)."""
    g = t.shape[0]
    t0 = int(t[:, 0][t[:, 0] > 0].min())

    def us(v):
        return (int(v) - t0) / 100.0

    d = dict(plan=us(t[0, 1]),
             scan=max(us(t[b, 1]) for b in range(1, P + 1)),
             send0=med([us(t[b, 2]) for b in range(1, P + 1)]),
             send1=max(us(t[b, 3]) for b in range(1, P + 1)),
             send1med=med([us(t[b, 3]) for b in range(1, P + 1)]),
             quant=max(us(t[b, 1]) for b in range(P + 1, g)),
             loop0=med([us(t[b, 4]) for b in range(g)]),
             loop1_med=med([us(t[b, 5]) for b in range(g)]),
             loop1_min=min(us(t[b, 5]) for b in range(g)),
             loop1_max=max(us(t[b, 5]) for b in range(g)),
             fin=max(us(t[b, 6]) for b in range(g) if t[b, 6] > 0))
    dur = {0: [], 1: []}
    pre = {0: [], 1: []}
    gaps, nunits, first_g2 = [], [], []
    for b in range(g):
        k = 0
        prev_end = None
        while k < 60:
            s, w, e, uid = (int(t[b, 16 + 4 * k]), int(t[b, 17 + 4 * k]), int(t[b, 18 + 4 * k]),
                            int(t[b, 19 + 4 * k]))
            if s == 0 or e < s or s < t0:
                break
            kind = int(uid >= u1)
            dur[kind].append((e - s) / 100.0)
            pre[kind].append((w - s) / 100.0)
            if kind == 1 and (not first_g2 or len(first_g2) <= b):
                pass
            if prev_end is not None:
                gaps.append((s - prev_end) / 100.0)
            prev_end = e
            k += 1
        nunits.append(k)
    for kind, name in ((0, "g1"), (1, "g2")):
        d[f"{name}_n"] = len(dur[kind])
        d[f"{name}_dur"] = med(dur[kind])
        d[f"{name}_dur90"] = pct(dur[kind], 0.9)
        d[f"{name}_pre"] = med(pre[kind])
        d[f"{name}_pre90"] = pct(pre[kind], 0.9)
        d[f"{name}_sum"] = sum(dur[kind]) / g
        d[f"{name}_presum"] = sum(pre[kind]) / g
    d["gap"] = med(gaps)
    d["gapsum"] = sum(gaps) / g
    d["units"] = (min(nunits), med(nunits), max(nunits))
    # producers' wave 0, chunks 0 and 1: lists loaded, rows loaded, stores done, published
    # (relative to the send start)
    for c in range(2):
        for i, nm in enumerate(("lst", "ld", "st", "pub")):
            v = [(int(t[b, 8 + 4 * c + i]) - int(t[b, 2])) / 100.0 for b in range(1, P + 1)
                 if t[b, 8 + 4 * c + i] > t[b, 2] > 0]
            d[f"c{c}{nm}"] = med(v)
    return d


def worker(rank, port, ns, knobs, act, iters, show):
    D.init(rank, port)
    from a8w4.kernels._protocol import LC_PLAN
    from a8w4.megamoe import MegaMoeLargeEpA8W4

    situ = R.SITU if act == "situv2" else None
    W, raw = R.make_weights(E_LOCAL, H, I, seed=rank)
    m = MegaMoeLargeEpA8W4(W, rank=rank, max_tokens=max(4096, max(ns)), topk=TOPK)
    nxt = itertools.cycle(R.weight_copies(W)).__next__
    x, tw, ti = R.make_routing(ns[rank], H, E_GLOBAL, TOPK, seed=7919 * rank + ns[rank])
    kn = dict(knobs, tl=1)
    P = kn.get("producers", 16)
    spin = D.spin_cycles(D.SPIN_US)
    for _ in range(2):
        m(x, ti, tw, act=act, situ=situ, **kn)
    torch.cuda.synchronize()
    res = []
    for _ in range(iters):
        W_ = nxt()
        t, _ = D.timed(lambda: m(x, ti, tw, W=W_, act=act, situ=situ, **kn), spin)
        tiles = int(m.ctl[LC_PLAN + 1].item())
        res.append((t, analyse(m.timeline().cpu(), P, tiles * (I // 128))))
    allr = [None] * WORLD
    dist.all_gather_object(allr, res)
    if rank == 0:
        for it in range(iters):
            e2e = max(allr[r][it][0] for r in range(WORLD))
            print(f"--- call {it}: e2e (slowest rank) {e2e:.1f} us", flush=True)
            for r in show:
                dd = allr[r][it][1]
                print(f"rank {r}: " + ", ".join(
                    f"{k} {v:.1f}" if isinstance(v, float) else f"{k} {v}" for k, v in dd.items()),
                    flush=True)
            for k in ("plan", "send1", "loop1_max", "fin"):
                vals = [allr[r][it][1][k] for r in range(WORLD)]
                print(f"  {k} per rank: " + " ".join(f"{v:.1f}" for v in vals), flush=True)
    D.finish()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--tokens", type=int, default=8192)
    ap.add_argument("--act", default="silu")
    ap.add_argument("--iters", type=int, default=2)
    ap.add_argument("--show", type=int, nargs="+", default=[0])
    ap.add_argument("-k", "--knob", nargs="*", default=[])
    a = ap.parse_args()
    knobs = {k: int(v) if v.lstrip("-").isdigit() else v for k, v in (s.split("=") for s in a.knob)}
    D.spawn(worker, (D.split_tokens(a.tokens), knobs, a.act, a.iters, a.show))
