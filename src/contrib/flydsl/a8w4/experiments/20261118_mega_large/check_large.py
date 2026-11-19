"""8-GPU correctness check of MegaMoeLargeEpA8W4 vs the fp32 reference (dev scratch).

    python a8w4/experiments/20261118_mega_large/check_large.py -t 2048 32768 [--uneven]
        [--act situv2] [-k producers=32 no=128 x_quant=per_token] [--reps 2] [--max-tokens 4096]

The reference uses the kernel's own fp8 x (``m.x_ref``: the scales of the call's
``x_quant``), so the difference is accumulation order and the bf16 output.
"""
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

FP8 = torch.float8_e4m3fnuz


def bounded(fn, rank, what, timeout=60):
    out = fn()
    ev = torch.cuda.Event()
    ev.record()
    t0 = time.monotonic()
    while not ev.query():
        if time.monotonic() - t0 > timeout:
            print(f"rank {rank}: {what} unfinished after {timeout} s", file=sys.stderr, flush=True)
            os._exit(1)
        time.sleep(1e-3)
    return out


def worker(rank, port, points, knobs, act, reps, max_tokens):
    D.init(rank, port)
    from a8w4.megamoe import MegaMoeLargeEpA8W4

    situ = R.SITU if act == "situv2" else None
    W, raw = R.make_weights(E_LOCAL, H, I, seed=rank)
    m = MegaMoeLargeEpA8W4(W, rank=rank, max_tokens=max_tokens, topk=TOPK)
    for rep in range(reps):
        for pi, ns in enumerate(points):
            routing = R.ROUTINGS[(rep + pi) % 2]
            x, tw, ti = R.make_routing(ns[rank], H, E_GLOBAL, TOPK,
                                       seed=7919 * rank + ns[rank] + rep, routing=routing)
            if rep % 2 and ns[rank]:  # some dropped slots
                g = torch.Generator(device="cuda").manual_seed(rank + 17 * rep)
                ti = ti.masked_fill(torch.rand(ti.shape, device="cuda", generator=g) < 1 / 16, -1)
            torch.cuda.synchronize()
            dist.barrier()
            y = bounded(lambda: m(x, ti, tw, act=act, situ=situ, **knobs), rank,
                        f"call {ns} rep {rep}").clone()
            n = ns[rank]
            xq = m.x_ref(n) if n else x.float()
            X, TW, TI = (D.gather_rows(t, ns) for t in (xq, tw, ti))
            ref = R.ref_moe(X, TW, TI, raw, rank * E_LOCAL, act, situ, quant_x=False)
            dist.all_reduce(ref)
            off = sum(ns[:rank])
            d = R.cos_diff(ref[off:off + n], y.float()) if n else 0.0
            nan = bool(torch.isnan(y).any()) if n else False
            res = [None] * WORLD
            dist.all_gather_object(res, (d, nan, m.rows_needed()))
            if rank == 0:
                print(f"rep {rep} {routing} tokens/rank {ns}: cos diff max "
                      f"{max(r[0] for r in res):.2e} nan {[r[1] for r in res]} rows needed "
                      f"{[r[2] for r in res]} (cap {m.s_cap})", flush=True)
    D.finish()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--tokens", type=int, nargs="+", default=[2048, 32768])
    ap.add_argument("--uneven", action="store_true")
    ap.add_argument("--act", default="silu")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("-k", "--knob", nargs="*", default=[])
    a = ap.parse_args()
    knobs = {k: int(v) if v.lstrip("-").isdigit() else v for k, v in (s.split("=") for s in a.knob)}
    points = [D.split_tokens(n) for n in a.tokens]
    if a.uneven:
        points += [[0, 1, 37, 300, 5, 128, 999, 0], [0] * 8, [1] + [0] * 7]
    D.spawn(worker, (points, knobs, a.act, a.reps, a.max_tokens))
