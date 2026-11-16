"""EP8 gemm scheduling knobs not swept in 20261114_ep_cfg_sweep: s_setprio ping-pong
(``prio``), W prefetch distance (``pf``), A-store step (``st``), W / A cache
modifiers (``bcm`` / ``acm``).

    cd src/contrib/flydsl && HIP_VISIBLE_DEVICES=2 python a8w4/experiments/20261115_ep_prio/sweep.py 1024 4096 [--set main|ext]

Uniform routing over 256 experts (32 local), cold L2, interleaved rounds; reports the
median over rounds of each round's median (R.time_cold) and the spread.
"""
import argparse
import copy
import os
import statistics
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from a8w4 import ref as R  # noqa: E402
from a8w4.kernels import ep_gemm as EG  # noqa: E402

H, I, E, EGL, K = 4096, 2048, 32, 256, 8


def var(base, g1=None, g2=None, **kw):
    c = copy.deepcopy(EG.CFGS[base])
    c["g1"].update(g1 or {})
    c["g2"].update(g2 or {})
    c.update(kw)
    return c


def variants(base, which):
    v = {base: var(base)}
    if which in ("main", "all"):
        v.update({
            f"{base}_g1prio": var(base, g1=dict(prio=1)),
            f"{base}_g2prio": var(base, g2=dict(prio=1)),
            f"{base}_prio": var(base, g1=dict(prio=1), g2=dict(prio=1)),
            f"{base}_g1pf1": var(base, g1=dict(pf=1)),
            f"{base}_g1pf4": var(base, g1=dict(pf=4)),
            f"{base}_g2pf1": var(base, g2=dict(pf=1)),
            f"{base}_g2pf4": var(base, g2=dict(pf=4)),
        })
    if which == "pf1":  # confirmation of the m64 g2 pf=1 point
        v.update({
            f"{base}_g2pf1": var(base, g2=dict(pf=1)),
            f"{base}_pf1": var(base, g1=dict(pf=1), g2=dict(pf=1)),
            f"{base}_g2pf1b": var(base, g2=dict(pf=1)),  # same config twice: noise floor
        })
    if which == "bcm":  # W cache modifier combinations (default bcm=2: nt)
        v.update({
            f"{base}_g2bcm0": var(base, g2=dict(bcm=0)),
            f"{base}_g2bcm0pf1": var(base, g2=dict(bcm=0, pf=1)),
            f"{base}_bcm0": var(base, g1=dict(bcm=0), g2=dict(bcm=0)),
            f"{base}_bcm0g2pf1": var(base, g1=dict(bcm=0), g2=dict(bcm=0, pf=1)),
        })
    if which in ("ext", "all"):
        v.update({
            f"{base}_g1st1": var(base, g1=dict(st=1)),
            f"{base}_g1st3": var(base, g1=dict(st=3)),
            f"{base}_g2st1": var(base, g2=dict(st=1)),
            f"{base}_g2st3": var(base, g2=dict(st=3)),
            f"{base}_g1bcm0": var(base, g1=dict(bcm=0)),
            f"{base}_g1bcm1": var(base, g1=dict(bcm=1)),
            f"{base}_g2bcm0": var(base, g2=dict(bcm=0)),
            f"{base}_g2bcm1": var(base, g2=dict(bcm=1)),
            f"{base}_acm0": var(base, g1=dict(acm=0), g2=dict(acm=0)),
        })
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tokens", type=int, nargs="+")
    ap.add_argument("--set", default="main", choices=["main", "ext", "all", "pf1", "bcm"])
    ap.add_argument("--base", default=None, help="CFGS entry (default: EG.pick_cfg)")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--iters", type=int, default=10)
    args = ap.parse_args()
    W, raw = R.make_weights(E, H, I)
    for T in args.tokens:
        base = args.base or next(n for n, c in EG.CFGS.items() if c is EG.pick_cfg(T))
        V = variants(base, args.set)
        x, tw, ti = R.make_routing(T, H, EGL, K, seed=7 + T)
        ref = R.ref_moe(x, tw, ti, raw, expert_offset=0).float() if T <= 2048 else None
        base_out = None
        rows, bad = {}, {}
        for rnd in range(args.rounds):
            for n, c in V.items():
                if n in bad:
                    continue
                try:
                    f = lambda c=c: EG.ep_moe(x, W, tw, ti, e0=0, cfg=c)  # noqa: E731
                    if rnd == 0:
                        out = f().float()
                        if base_out is None:
                            base_out = out
                            if ref is not None:
                                print(f"  {n}: cos vs ref {R.cos_diff(out, ref):.2e}", flush=True)
                        cb = R.cos_diff(out, base_out)
                        if cb > 1e-5:
                            print(f"  {n}: MISMATCH vs base cos {cb:.2e}", flush=True)
                    rows.setdefault(n, []).append(R.time_cold(f, iters=args.iters))
                except Exception as e:  # noqa: BLE001
                    bad[n] = str(e)[:120]
                    print(f"  {n}: FAILED {bad[n]}", flush=True)
        b = statistics.median(rows[base])
        print(f"== tokens {T} (base {base}, {args.rounds} interleaved rounds, cold)")
        for n, t in rows.items():
            m = statistics.median(t)
            print(f"{n:18s} {m:9.1f} us  ({(m / b - 1) * 100:+5.1f}%)  "
                  f"min {min(t):8.1f} max {max(t):8.1f}", flush=True)


if __name__ == "__main__":
    main()
