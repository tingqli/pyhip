"""Does the W cache-modifier gain (bcm 0 = cached vs 2 = nt, 20261115) survive a
realistic cold start? ``ref.time_cold`` flushes by reading 512 MB; if that left
weights in the 256 MB MALL, cached W loads could look faster than they are. Here
every call reads its own copy of the weights, as consecutive layers do: ``--layers``
copies (8 x ~400 MB >> MALL) run back to back without flush, each call timed by its
own events. ``time_cold`` on a single copy is measured too, for comparison.

    cd src/contrib/flydsl && HIP_VISIBLE_DEVICES=0 python a8w4/experiments/20261115_ep_prio/rotate.py

"old" = the 20261114 configs (all W loads nt, gemm2 PF 2), "new" = the variants first
landed on 20261115 (gemm2 W cached + PF 1 at bm <= 64, gemm2 W cached at l128, both
gemms cached in t16c / l128c), defined here explicitly (CFGS no longer has them).
"""
import argparse
import copy
import os
import statistics
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from a8w4 import ref as R  # noqa: E402
from a8w4.kernels import ep_gemm as EG  # noqa: E402

H, I, E, EGL, K = 4096, 2048, 32, 256, 8


def var(base, b1=None, b2=None, pf2=None):
    """CFGS[base] without bcm / pf, then gemm1 / gemm2 W bcm and gemm2 PF if given."""
    c = copy.deepcopy(EG.CFGS[base])
    for g in ("g1", "g2"):
        c[g].pop("bcm", None)
        c[g].pop("pf", None)
    if b1 is not None:
        c["g1"]["bcm"] = b1
    if b2 is not None:
        c["g2"]["bcm"] = b2
    if pf2 is not None:
        c["g2"]["pf"] = pf2
    return c


def old(name):
    return var(name)


NEW = {"t16c": var("t16", 0, 0, 1), "s16": var("s16", None, 0, 1),
       "s32": var("s32", None, 0, 1), "m64": var("m64", None, 0, 1),
       "l128": var("l128", None, 0), "l128c": var("l128", 0, 0)}

CASES = {  # tokens -> {variant: cfg}
    8: {"t16": old("t16"), "t16c": NEW["t16c"]},
    16: {"t16": old("t16"), "t16c": NEW["t16c"]},
    24: {"t16": old("t16"), "t16c": NEW["t16c"], "s16 new": NEW["s16"]},
    64: {"s16 old": old("s16"), "s16 new": NEW["s16"]},
    256: {"s16 old": old("s16"), "s16 new": NEW["s16"]},
    512: {"s32 old": old("s32"), "s32 new": NEW["s32"]},
    1024: {"m64 old": old("m64"), "m64 new": NEW["m64"]},
    2048: {"m64 old": old("m64"), "m64 new": NEW["m64"], "l128 old": old("l128")},
    4096: {"l128 old": old("l128"), "l128 new": NEW["l128"], "l128c": NEW["l128c"]},
    16384: {"l128 old": old("l128"), "l128 new": NEW["l128"], "l128c": NEW["l128c"]},
}


def rotate_pass(fns):
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(len(fns) + 1)]
    torch.cuda._sleep(4_000_000)  # let the host enqueue the whole pass first
    ev[0].record()
    for i, f in enumerate(fns):
        f()
        ev[i + 1].record()
    torch.cuda.synchronize()
    return [ev[i].elapsed_time(ev[i + 1]) * 1e3 for i in range(len(fns))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tokens", type=int, nargs="*", default=sorted(CASES))
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=7)
    a = ap.parse_args()
    W0, _ = R.make_weights(E, H, I)
    Ws = [W0] + [{k: v.clone() if torch.is_tensor(v) else v for k, v in W0.items()}
                 for _ in range(a.layers - 1)]
    mb = sum(v.numel() * v.element_size() for v in W0.values() if torch.is_tensor(v)) / 2**20
    print(f"{a.layers} weight copies of {mb:.0f} MB")
    for T in a.tokens:
        V = CASES[T]
        x, tw, ti = R.make_routing(T, H, EGL, K, seed=7 + T)
        rot, cold = {n: [] for n in V}, {n: [] for n in V}
        for rnd in range(a.rounds):
            names = list(V) if rnd % 2 == 0 else list(V)[::-1]
            for n in names:
                fns = [lambda W=W, c=V[n]: EG.ep_moe(x, W, tw, ti, e0=0, cfg=c)
                       for W in Ws]
                if rnd == 0:
                    rotate_pass(fns)  # compile / warm up
                rot[n] += rotate_pass(fns)
                cold[n].append(R.time_cold(fns[0], iters=10))
        b = list(V)[0]
        rb, cb = statistics.median(rot[b]), statistics.median(cold[b])
        print(f"== tokens {T}: rotate ({a.layers} layers x {a.rounds} rounds) | time_cold")
        for n in V:
            r, c = statistics.median(rot[n]), statistics.median(cold[n])
            print(f"  {n:9s} rotate {r:8.1f} us ({(r / rb - 1) * 100:+5.1f}%) | "
                  f"cold {c:8.1f} us ({(c / cb - 1) * 100:+5.1f}%)", flush=True)


if __name__ == "__main__":
    main()
