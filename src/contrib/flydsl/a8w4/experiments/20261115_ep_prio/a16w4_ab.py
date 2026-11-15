"""a16w4 (``adt="bf16"``) shares CFGS with a8w4: A/B of the 20261115 W cache-modifier
change on that path ("old" = all W loads nt and gemm2 PF 2, as in rotate.py; "new" = the
first landed variants, rotate.NEW; ``--final``: the configs finally landed, CFGS).
``--wcopies``: weight copies the timed calls cycle through (a16w4_ab.txt: 1, the flawed
same-weights protocol; a16w4_final.txt: 8).

    cd src/contrib/flydsl && HIP_VISIBLE_DEVICES=3 python a8w4/experiments/20261115_ep_prio/a16w4_ab.py --final
"""
import argparse
import itertools
import os
import statistics
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from a8w4 import ref as R  # noqa: E402
from a8w4.kernels import ep_gemm as EG  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from rotate import NEW, old  # noqa: E402

H, I, E, EGL, K = 4096, 2048, 32, 256, 8
CASES = {16: ("t16", "t16c"), 256: ("s16",), 1024: ("m64",), 4096: ("l128",),
         16384: ("l128", "l128c")}
FINAL = {1024: ("m64",), 2048: ("m64",), 4096: ("l128", "l128c"), 16384: ("l128", "l128c")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--wcopies", type=int, default=R.WCOPIES)
    a = ap.parse_args()
    W, _ = R.make_weights(E, H, I)
    nxt = itertools.cycle(R.weight_copies(W, a.wcopies)).__next__
    for T, names in (FINAL if a.final else CASES).items():
        x, tw, ti = R.make_routing(T, H, EGL, K, seed=7 + T)
        V = {f"{names[0]} old": old(names[0])}
        if a.final:
            V.update({f"{n} final": EG.CFGS[n] for n in names})
        else:
            V.update({f"{n} new": NEW[n] for n in names})
        ts = {n: [] for n in V}
        ref = None
        for rnd in range(7):
            for n, c in (V.items() if rnd % 2 == 0 else list(V.items())[::-1]):
                f = lambda c=c: EG.ep_moe(x, nxt(), tw, ti, e0=0, adt="bf16", cfg=c)  # noqa: E731
                if rnd == 0:
                    out = f().float()
                    ref = out if ref is None else ref
                    assert R.cos_diff(out, ref) < 1e-5, n
                ts[n].append(R.time_cold(f, iters=10))
        b = statistics.median(ts[f"{names[0]} old"])
        print(f"== a16w4 tokens {T}")
        for n, t in ts.items():
            m = statistics.median(t)
            print(f"  {n:10s} {m:8.1f} us ({(m / b - 1) * 100:+5.1f}%)", flush=True)


if __name__ == "__main__":
    main()
