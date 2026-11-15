"""W cache modifier / gemm2 PF variants of the EP configs under the corrected timing:
``ref.time_cold`` (512 MB read flush + sleep) where every call reads the next of
``--copies`` weight copies (flushcheck.py: the flush does not evict weights that are
re-read every call from the MALL, which made cached W loads look 5-13% faster).

    cd src/contrib/flydsl && HIP_VISIBLE_DEVICES=0 python a8w4/experiments/20261115_ep_prio/coldrot.py 1024 2048

Variants: old = nt W loads, gemm2 PF 2 (the configs before 20261115);
g2pf1 = old + gemm2 PF 1; g2c = gemm2 W cached (+ PF 1 at bm <= 64, as first landed);
c = both gemms cached (+ gemm2 PF 1 at bm 16, t16c).
"""
import argparse
import itertools
import os
import statistics
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from a8w4 import ref as R  # noqa: E402
from a8w4.kernels import ep_gemm as EG  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from rotate import NEW, old, var  # noqa: E402

H, I, E, EGL, K = 4096, 2048, 32, 256, 8
SMALL = {"t16 old": old("t16"), "t16c": NEW["t16c"]}
S16 = {"s16 old": old("s16"), "s16 g2c": NEW["s16"]}
S32 = {"s32 old": old("s32"), "s32 g2c": NEW["s32"]}
M64 = {"m64 old": old("m64"), "m64 g2pf1": var("m64", pf2=1), "m64 g2c": NEW["m64"]}
L128 = {"l128 old": old("l128"), "l128 g2c": NEW["l128"], "l128c": NEW["l128c"]}
CASES = {8: SMALL, 16: SMALL, 64: S16, 256: S16, 512: S32, 768: S32, 1024: M64,
         1536: M64, 2048: {**M64, "l128 old": old("l128")}, 2560: {**L128, "m64 old": old("m64")},
         4096: L128, 8192: L128, 12288: L128, 16384: L128, 32768: L128}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tokens", type=int, nargs="*", default=sorted(CASES))
    ap.add_argument("--copies", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--iters", type=int, default=16)
    a = ap.parse_args()
    W0, _ = R.make_weights(E, H, I)
    Ws = [W0] + [{k: v.clone() if torch.is_tensor(v) else v for k, v in W0.items()}
                 for _ in range(a.copies - 1)]
    nxt = itertools.cycle(Ws).__next__
    for T in a.tokens:
        V = CASES[T]
        x, tw, ti = R.make_routing(T, H, EGL, K, seed=7 + T)
        ts = {n: [] for n in V}
        for rnd in range(a.rounds):
            for n in (list(V) if rnd % 2 == 0 else list(V)[::-1]):
                f = lambda c=V[n]: EG.ep_moe(x, nxt(), tw, ti, e0=0, adt="fp8", cfg=c)  # noqa: E731
                ts[n].append(R.time_cold(f, iters=a.iters))
        b = statistics.median(ts[list(V)[0]])
        print(f"== tokens {T} ({a.copies} W copies, {a.rounds} rounds x {a.iters} iters)")
        for n, t in ts.items():
            m = statistics.median(t)
            print(f"  {n:10s} {m:8.1f} us ({(m / b - 1) * 100:+5.1f}%)  "
                  f"min {min(t):8.1f} max {max(t):8.1f}", flush=True)


if __name__ == "__main__":
    main()
