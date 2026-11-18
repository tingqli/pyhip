"""8-GPU correctness check of MegaMoeSmallEpA8W4 vs the fp32 reference (dev scratch)."""
import argparse
import os
import sys
import time

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
from a8w4 import ref as R  # noqa: E402
from a8w4.bench import dist as D  # noqa: E402
from a8w4.bench.dist import E_GLOBAL, E_LOCAL, H, I, TOPK, WORLD  # noqa: E402


def bounded(fn, rank, what):
    out = fn()
    ev = torch.cuda.Event()
    ev.record()
    t0 = time.monotonic()
    while not ev.query():
        if time.monotonic() - t0 > 30:
            print(f"rank {rank}: {what} unfinished after 30 s", file=sys.stderr, flush=True)
            os._exit(1)
        time.sleep(1e-3)
    return out


def worker(rank, port, points, knobs, act, reps):
    D.init(rank, port)
    from a8w4.megamoe import MegaMoeSmallEpA8W4

    situ = R.SITU if act == "situv2" else None
    W, raw = R.make_weights(E_LOCAL, H, I, seed=rank)
    m = MegaMoeSmallEpA8W4(W, rank=rank, tmax=1024, topk=TOPK)
    for rep in range(reps):
        for pi, ns in enumerate(points):
            routing = R.ROUTINGS[(rep + pi) % 2]
            x, tw, ti = R.make_routing(ns[rank], H, E_GLOBAL, TOPK, seed=7919 * rank + ns[rank] + rep,
                                       routing=routing)
            if rep % 2 and ns[rank]:  # some dropped slots
                g = torch.Generator(device="cuda").manual_seed(rank + 17 * rep)
                ti = ti.masked_fill(torch.rand(ti.shape, device="cuda", generator=g) < 1 / 16, -1)
            torch.cuda.synchronize()
            dist.barrier()
            y = bounded(lambda: m(x, ti, tw, act=act, situ=situ, **knobs), rank,
                        f"call {ns} rep {rep}").clone()
            n = ns[rank]
            xq = R.dequant_fp8(*R.quant_fp8_groups(x)) if n else x.float()
            X, TW, TI = (D.gather_rows(t, ns) for t in (xq, tw, ti))
            ref = R.ref_moe(X, TW, TI, raw, rank * E_LOCAL, act, situ, quant_x=False)
            dist.all_reduce(ref)
            off = sum(ns[:rank])
            d = R.cos_diff(ref[off:off + n], y.float()) if n else 0.0
            nan = bool(torch.isnan(y).any()) if n else False
            res = [None] * WORLD
            dist.all_gather_object(res, (d, nan))
            if rank == 0:
                print(f"rep {rep} {routing} tokens/rank {ns}: cos diff max {max(r[0] for r in res):.2e}"
                      f" nan {[r[1] for r in res]}", flush=True)
    D.finish()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--tokens", type=int, nargs="+", default=[64, 512])
    ap.add_argument("--uneven", action="store_true")
    ap.add_argument("--act", default="silu")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("-k", "--knob", nargs="*", default=[])
    a = ap.parse_args()
    knobs = {k: int(v) for k, v in (s.split("=") for s in a.knob)}
    points = [D.split_tokens(n) for n in a.tokens]
    if a.uneven:
        points += [[0, 1, 37, 300, 5, 128, 99, 0], [0] * 8, [1] + [0] * 7]
    D.spawn(worker, (points, knobs, a.act, a.reps))
