"""Split-K gemm1 for tiny batches (EP8 MiMo shape, the "t16" config).

At 1 - 8 tokens the EP8 gemm1 has only (active experts x 16) WGs, each streaming
512 KB of W1 (K = 4096): far fewer WGs than the 304 CUs, so the kernel is bound by
one WG's W stream latency. ``ep_gemm_sk.py`` (a copy of kernels/ep_gemm.py) splits
the K loop of each (m-block, n-block) tile over SK WGs; every WG writes its f32
partial accumulators through to a scratch buffer and takes a ticket on the tile's
counter, the last arriver sums the SK partials (fixed split order) and runs the
unchanged epilogue (act + fp8 quant), then resets the counter.

    cd src/contrib/flydsl
    HIP_VISIBLE_DEVICES=0 python a8w4/experiments/20261114_splitk_gemm1/run.py check
    HIP_VISIBLE_DEVICES=0 python a8w4/experiments/20261114_splitk_gemm1/run.py bench 1 2 4 8 16

Variants: "base" (= kernels/ep_gemm ep_moe cfg t16), "sk{S}p{PF}[a{ARB}][t{N}]"
(split-K gemm1 S ways, W prefetch distance PF, A row bytes per K tile ARB; S=1 =
this file's kernel without split, e.g. "sk1p4" = PF 4; t{N}: adaptive, split only
when the valid tile count <= N, decided on device).
"""
import argparse
import os
import re
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "../../..")))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "../20261114_r2_baseline"))
from a8w4 import ref as R  # noqa: E402
from a8w4.kernels import ep_gemm as EG  # noqa: E402
from a8w4.kernels import ep_prep as PR  # noqa: E402
import ep_gemm_sk as SKG  # noqa: E402
from breakdown import ep_routing, kernel_times, short  # noqa: E402

H, I, E, EGL, K = 4096, 2048, 32, 256, 8
DEFAULT_VARIANTS = "base,sk1p2,sk1p4,sk2p2,sk2p4,sk4p1,sk4p2,sk8p1,sk4p2a128,sk8p2a128,sk16p1a128"


def parse_variant(v):
    if v == "base":
        return None
    m = re.fullmatch(r"sk(\d+)p(\d+)(?:a(\d+))?(?:t(\d+))?", v)
    assert m, v
    return dict(sk=int(m[1]), pf=int(m[2]), arb=int(m[3] or 256), sk_tiles=int(m[4] or 0))


def ep_moe_sk(x, W, tw, ti, *, sk, pf, arb, sk_tiles=0, poison=False):
    """ep_moe with cfg t16 (bm16, plain order, single-store), gemm1 from ep_gemm_sk."""
    M = x.shape[0]
    bm = 16
    sids, sw, seids, nv, x8, xs, sched, rc = PR.ep_prep(
        ti, tw, x, e0=0, block_m=bm, quant=True, num_local=E)
    sorted_size = seids.numel() * bm
    out = torch.empty((M, H), dtype=torch.bfloat16, device=x.device)
    a = torch.empty((sorted_size, I), dtype=torch.uint8, device=x.device)
    a_s = torch.empty((sorted_size, I // 128), dtype=torch.float32, device=x.device)
    if poison:  # rows gemm1 fails to write show up as 240 * 1e3 in the output
        a.fill_(0x7F)
        a_s.fill_(1e3)
    SKG.ep_gemm1(x8, xs, W, sids, seids, nv, a, a_s, adt="fp8", block_m=bm, zero=out,
                 zero_rc=rc, rowv=(bm,), sched=None, ni=128, nw=8, sch=3, pf=pf, arb=arb,
                 sk=sk, sk_tiles=sk_tiles)
    EG.ep_gemm2(a, a_s, W, sids, sw, seids, nv, out, block_m=bm, rowv=(bm,),
                sched=None, single_store=True, no=64, nw=4, sch=3)
    return out


def make_fn(v, x, W, tw, ti, poison=False):
    p = parse_variant(v)
    if p is None:
        return lambda: EG.ep_moe(x, W, tw, ti, e0=0, cfg=EG.CFGS["t16"])
    return lambda: ep_moe_sk(x, W, tw, ti, poison=poison, **p)


def inputs(T, local, seed):
    if local > 0:
        return ep_routing(T, H, EGL, E, K, local, seed=seed)
    return R.make_routing(T, H, EGL, K, seed=seed)  # uniform over the 256 experts


def check(W, raw, variants, tokens):
    bad = 0
    for local in (1, 2, 0):
        for T in tokens:
            x, tw, ti = inputs(T, local, seed=100 + T)
            ref = R.ref_moe(x, tw, ti, raw, 0)
            base = make_fn("base", x, W, tw, ti)().float()
            for v in variants:
                fn = make_fn(v, x, W, tw, ti, poison=True)
                o1 = fn().float()
                outs = [fn().float() for _ in range(20)]  # counters reset every call
                torch.cuda.synchronize()
                cos = R.cos_diff(o1, ref)
                dif = (o1 - base).abs().max().item()
                # rows with >= 3 local routes: gemm2's bf16 atomics (any kernel, also
                # "base") add in a run-dependent order -> ~1 ulp differences
                stable = all(R.cos_diff(o, o1) < 1e-5 for o in outs)
                ok = cos < 5e-3 and stable and torch.isfinite(o1).all().item()
                bad += not ok
                print(f"local={local} T={T:3d} {v:12s} cos {cos:.1e}  max|d base| {dif:.3e}"
                      f"  stable {stable}  {'OK' if ok else 'FAIL'}", flush=True)
    # CUDA graph capture + replay (the split-K scratch / counters are reused)
    x, tw, ti = inputs(4, 1, seed=5)
    for v in variants:
        if v == "base":
            continue
        fn = make_fn(v, x, W, tw, ti)
        ref = fn().float()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            fn()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = fn()
        errs = []
        for _ in range(10):
            g.replay()
            torch.cuda.synchronize()
            errs.append(R.cos_diff(out.float(), ref))
        ok = max(errs) < 1e-6
        bad += not ok
        print(f"graph {v:12s} max cos vs eager {max(errs):.1e} {'OK' if ok else 'FAIL'}",
              flush=True)
    print("ALL OK" if bad == 0 else f"{bad} FAILURES")


def bench(W, variants, tokens, local, iters, rounds, prof):
    for T in tokens:
        x, tw, ti = inputs(T, local, seed=T)
        fns = {v: make_fn(v, x, W, tw, ti) for v in variants}
        res = {v: [] for v in variants}
        for _ in range(rounds):  # interleaved A/B (session drift ~15%)
            for v, fn in fns.items():
                res[v].append(R.time_cold(fn, iters=iters))
        med = {v: sorted(t)[len(t) // 2] for v, t in res.items()}
        b = med[variants[0]]
        line = f"local={local} T={T:4d} e2e(us)"
        for v in variants:
            line += f"  {v} {med[v]:.1f}({(med[v] / b - 1) * 100:+.0f}%)"
        print(line, flush=True)
        if prof:
            g1 = {}
            for v, fn in fns.items():
                kt = kernel_times(fn, iters=iters)
                g1[v] = sum(t for k, t in kt.items() if short(k) == "ep_gemm1")
            print(f"local={local} T={T:4d} gemm1(us)" + "".join(
                f"  {v} {g1[v]:.1f}" for v in variants), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("check", "bench"))
    ap.add_argument("tokens", type=int, nargs="*", default=[1, 2, 4, 8, 16])
    ap.add_argument("--variants", default=DEFAULT_VARIANTS)
    ap.add_argument("--local", type=int, default=1, help="local experts per token (0: uniform)")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--prof", action="store_true", help="also the gemm1 kernel time")
    a = ap.parse_args()
    variants = a.variants.split(",")
    W, raw = R.make_weights(E, H, I)
    if a.mode == "check":
        check(W, raw, variants, a.tokens)
    else:
        bench(W, variants, a.tokens, a.local, a.iters, a.rounds, a.prof)


if __name__ == "__main__":
    main()
