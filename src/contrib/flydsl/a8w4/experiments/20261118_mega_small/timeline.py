"""Phase + per-unit timeline of MegaMoeSmallEpA8W4 (kernel built with TL=1), cold calls.

Needs the TL instrumentation: apply timeline_instrumentation.diff to kernels/ep_mega_small.py
(patch -R; it is the diff from the instrumented to the landed file) and add tl / "TL" to
megamoe.SMALL_KNOBS / SMALL_KEYS (and 128 * 2 * NUM_CU ints to MegaMoeSmallEpA8W4.ctl).

Per rank (us, from the rank's earliest workgroup start; s_memrealtime 100 MHz): phase ends,
and per unit kind: duration, the part before the K loop (gemm2: incl. the m-tile wait),
the gap to the workgroup's next unit; units per workgroup.
"""
import argparse
import os
import itertools
import statistics
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
from a8w4 import ref as R  # noqa: E402
from a8w4.bench import dist as D  # noqa: E402
from a8w4.bench.dist import E_GLOBAL, E_LOCAL, H, I, TOPK, WORLD  # noqa: E402

TLW = 128


def med(v):
    return statistics.median(v) if v else float("nan")


def analyse(t):
    """t: int64 [grid, 64] stamps of one call -> dict of numbers (us)."""
    t0 = int(t[:, 0].min())
    g = t.shape[0]

    def us(v):
        return (int(v) - t0) / 100.0

    d = dict(slot=max(us(t[b, 6]) for b in range(16) if t[b, 6]),
             stored=max(us(t[b, 7]) for b in range(16)),
             prod=max(us(t[b, 1]) for b in range(16)), plan=med([us(t[b, 2]) for b in range(g)]),
             units_med=med([us(t[b, 3]) for b in range(g)]),
             units_max=max(us(t[b, 3]) for b in range(g)), out=max(us(t[b, 4]) for b in range(g)),
             comb=max(us(t[b, 5]) for b in range(g)))
    dur = {0: [], 1: []}
    pre = {0: [], 1: []}
    gaps, first, nunits = [], [], []
    for b in range(g):
        n = 0
        prev_end = None
        for k in range(8):
            s, w, e, uid = (int(t[b, 8 + 3 * k]), int(t[b, 9 + 3 * k]), int(t[b, 10 + 3 * k]),
                            int(t[b, 32 + k]))
            if s == 0 or e < s or s < t0:
                break
            kind = uid >> 20
            dur[kind].append((e - s) / 100.0)
            pre[kind].append((w - s) / 100.0)
            if prev_end is None:
                first.append((s - int(t[b, 2])) / 100.0)
            else:
                gaps.append((s - prev_end) / 100.0)
            prev_end = e
            n += 1
        nunits.append(n)
    d.update(g1=med(dur[0]), g1pre=med(pre[0]), g2=med(dur[1]), g2pre=med(pre[1]),
             g2pre_max=max(pre[1]) if pre[1] else float("nan"), gap=med(gaps),
             first=med(first), n1=len(dur[0]), n2=len(dur[1]), upw=med(nunits),
             upw_max=max(nunits))
    return d


def worker(rank, port, totals, knobs, iters):
    D.init(rank, port)
    from a8w4.kernels._hw import NUM_CU
    from a8w4.kernels._protocol import SC_TILE, small_max_tiles
    from a8w4.megamoe import MegaMoeSmallEpA8W4

    W, _ = R.make_weights(E_LOCAL, H, I, seed=rank)
    nxt = itertools.cycle(R.weight_copies(W)).__next__
    m = MegaMoeSmallEpA8W4(W, rank=rank, tmax=1024, topk=TOPK)
    tiles = max(small_max_tiles(TOPK, 1024, bm) for bm in (16, 32, 64))
    sc_tl = SC_TILE + 16 * tiles
    grid = NUM_CU * knobs.get("wpc", 1)
    spin = D.spin_cycles(D.SPIN_US)
    for total in totals:
        ns = D.split_tokens(total)
        x, tw, ti = R.make_routing(ns[rank], H, E_GLOBAL, TOPK, seed=7919 * rank + total)
        for _ in range(3):
            m(x, ti, tw, W=nxt(), tl=1, **knobs)
        torch.cuda.synchronize()
        rows = []
        for _ in range(iters):
            Wc = nxt()
            m.ctl[sc_tl:sc_tl + TLW * grid].zero_()
            e2e, _ = D.timed(lambda: m(x, ti, tw, W=Wc, tl=1, **knobs), spin)
            t = m.ctl[sc_tl:sc_tl + TLW * grid].view(torch.int64).view(grid, TLW // 2).cpu()
            d = analyse(t)
            d["e2e"] = e2e
            rows.append(d)
        mm = {k: med([r[k] for r in rows]) for k in rows[0]}
        res = [None] * WORLD
        dist.all_gather_object(res, mm)
        if rank == 0:
            print(f"\n{total} tokens, knobs {knobs}: per rank median of {iters} cold calls (us)")
            keys = list(mm)
            print("| rank | " + " | ".join(keys) + " |")
            for r, x_ in enumerate(res):
                print(f"| {r} | " + " | ".join(f"{x_[k]:.1f}" for k in keys) + " |", flush=True)
    D.finish()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--tokens", type=int, nargs="+", default=[64, 512])
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("-k", "--knob", nargs="*", default=[])
    a = ap.parse_args()
    knobs = {k: int(v) for k, v in (s.split("=") for s in a.knob)}
    D.spawn(worker, (a.tokens, knobs, a.iters))
