"""TP8 MiMo shape (I = 256, 256 local experts): the split-K EP gemm1 (ep_gemm_sk) in
place of tp_gemm1 in the tuned "tp2k" path (aiter moe_sorting -> row quant -> gemm1
-> tp_gemm2). Everything but gemm1 is the tuned config's (same sorting, same
gemm2, same q8s A2 buffer: fp8 [sorted, I] | f32 [sorted, I/128]).

    cd src/contrib/flydsl
    HIP_VISIBLE_DEVICES=0 python a8w4/experiments/20261114_splitk_gemm1/run_tp.py check
    HIP_VISIBLE_DEVICES=0 python a8w4/experiments/20261114_splitk_gemm1/run_tp.py bench 1 2 4 8 16

Variants: "base" (the tuned tp2k config) or "sk{S}p{PF}[a{ARB}][w{NW}][t{N}]" (see
run.py; t{N}: split only when the valid tile count <= N, decided on device).
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
from a8w4.kernels import tp_gemm2 as G2  # noqa: E402
from a8w4.kernels.row_quant import quantize_rows  # noqa: E402
from a8w4.moe import _sort, config_name, lookup_config, run_config  # noqa: E402
import ep_gemm_sk as SKG  # noqa: E402
from breakdown import kernel_times, short  # noqa: E402

SHAPE = (4096, 256, 256, 256, 8)
H, I, E, EGL, K = SHAPE
DEFAULT_VARIANTS = "base,sk1p2,sk2p2,sk4p2,sk8p1,sk8p2a128,sk16p1a128,sk4p2w4,sk8p1w4"


def parse_variant(v):
    if v == "base":
        return None
    m = re.fullmatch(r"sk(\d+)p(\d+)(?:a(\d+))?(?:w(\d+))?(?:t(\d+))?", v)
    assert m, v
    return dict(sk=int(m[1]), pf=int(m[2]), arb=int(m[3] or 256), nw=int(m[4] or 8),
                sk_tiles=int(m[5] or 0))


def tp_sk(x, W, tw, ti, cfg, *, sk, pf, arb, nw, sk_tiles=0):
    """_run_tp2k with gemm1 = ep_gemm_sk.ep_gemm1 (q8s layout, no zeroing: the
    sorting zeroes out)."""
    bm = cfg["block_m"]
    groups = I // 128
    sids, sw, seids, nv, out = _sort(ti, tw, SHAPE, 0, bm)
    x8, xs = quantize_rows(x)
    sorted_size = seids.numel() * bm
    buf = torch.empty(sorted_size * (I + groups * 4), dtype=torch.uint8, device=x.device)
    nd = sorted_size * I
    a2u = buf[:nd].view(sorted_size, I)
    a2s = buf[nd:].view(torch.float32).view(sorted_size, groups)
    rowv = tuple(r for r in (16, 32, 64, 128) if r <= bm)
    SKG.ep_gemm1(x8, xs, W, sids, seids, nv, a2u, a2s, adt="fp8", block_m=bm, rowv=rowv,
                 zero=None, sched=None, ni=128, nw=nw, sch=3, pf=pf, arb=arb, sk=sk,
                 sk_tiles=sk_tiles)
    kw = dict(tile_n=128, b_cache_mod=2, xcd_swizzle=0, dyn_rows=True)
    if cfg["g2"] == "nt4":
        kw.update(n_tiles=4, epilog="direct64", row_vgpr=True)
    G2.a8w4_gemm2(a2u.view(torch.float8_e4m3fnuz), a2s, W["w2"], W["w2_scale"], sids, sw,
                  seids, nv, out, num_experts=E, tile_m=bm, a_group=128, fast_dec=2, **kw)
    return out


def make_fn(v, x, W, tw, ti, T):
    cfg = lookup_config(SHAPE, T)
    assert cfg["kind"] == "tp2k", config_name(cfg)
    p = parse_variant(v)
    if p is None:
        name = config_name(cfg)
        return lambda: run_config(name, x, W, tw, ti, expert_offset=0, num_experts=EGL)
    return lambda: tp_sk(x, W, tw, ti, cfg, **p)


def check(W, raw, variants, tokens):
    bad = 0
    for T in tokens:
        x, tw, ti = R.make_routing(T, H, EGL, K, seed=7 + T)
        ref = R.ref_moe(x, tw, ti, raw, 0)
        base = make_fn("base", x, W, tw, ti, T)().float()
        for v in variants:
            fn = make_fn(v, x, W, tw, ti, T)
            o1 = fn().float()
            outs = [fn().float() for _ in range(10)]
            torch.cuda.synchronize()
            cos = R.cos_diff(o1, ref)
            cb = R.cos_diff(o1, base)
            # every token has 8 routes: gemm2's bf16 atomics add in a run-dependent
            # order, so any two runs (also of "base") differ by ~1e-5 (1 - cos)
            jit = max(R.cos_diff(o, o1) for o in outs)
            ok = cos < 5e-3 and cb < 1e-4 and jit < 1e-4
            bad += not ok
            print(f"T={T:3d} {v:12s} cos {cos:.1e}  cos vs base {cb:.1e}  run jitter {jit:.1e}"
                  f"  {'OK' if ok else 'FAIL'}", flush=True)
    print("ALL OK" if bad == 0 else f"{bad} FAILURES")


def bench(W, variants, tokens, iters, rounds, prof):
    for T in tokens:
        x, tw, ti = R.make_routing(T, H, EGL, K, seed=7 + T)
        fns = {v: make_fn(v, x, W, tw, ti, T) for v in variants}
        res = {v: [] for v in variants}
        for _ in range(rounds):
            for v, fn in fns.items():
                res[v].append(R.time_cold(fn, iters=iters))
        med = {v: sorted(t)[len(t) // 2] for v, t in res.items()}
        b = med[variants[0]]
        print(f"tp8 T={T:4d} [{config_name(lookup_config(SHAPE, T))}] e2e(us)" + "".join(
            f"  {v} {med[v]:.1f}({(med[v] / b - 1) * 100:+.0f}%)" for v in variants), flush=True)
        if prof:
            g1 = {}
            for v, fn in fns.items():
                kt = kernel_times(fn, iters=iters)
                g1[v] = sum(t for k, t in kt.items() if short(k) in ("ep_gemm1", "a8w4_gemm1"))
            print(f"tp8 T={T:4d} gemm1(us)" + "".join(f"  {v} {g1[v]:.1f}" for v in variants),
                  flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("check", "bench"))
    ap.add_argument("tokens", type=int, nargs="*", default=[1, 2, 4, 8, 16])
    ap.add_argument("--variants", default=DEFAULT_VARIANTS)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--prof", action="store_true")
    a = ap.parse_args()
    variants = a.variants.split(",")
    W, raw = R.make_weights(E, H, I)
    if a.mode == "check":
        check(W, raw, variants, a.tokens)
    else:
        bench(W, variants, a.tokens, a.iters, a.rounds, a.prof)


if __name__ == "__main__":
    main()
