"""NOTE: imports run.py, which uses the API removed in refactor D; run this at commit
5d0b861 or earlier.

Does tp_prep's row order inside an expert (LDS atomic arrival, not token order as
moe_sorting) change the time of the GEMM kernels that follow? Times ONLY the tuned
config's GEMM kernels (cold L2, interleaved rounds) on the routing from
  aiter  moe_sorting (+ quantize_rows)
  prep   tp_prep
  prepo  tp_prep, each expert's rows re-sorted by token on the host (same set of rows)
  aitero moe_sorting's rows, re-sorted the same way
    HIP_VISIBLE_DEVICES=0 python a8w4/experiments/20261114_tp_prep/order.py 4096 8192
"""
import sys

import torch

import run as RN
from run import E, EGL, F, H, K, Q, R, SHAPE, _block_m, _sort, _tp2k_tail, lookup_config, \
    quantize_rows, config_name


def _token_order(sids, sw, seids, nv, bm):
    """each expert's valid rows sorted by (token, slot); padding rows stay last."""
    s, w = sids.clone(), sw.clone()
    n = int(nv[0])
    ex = seids[:n // bm].long().repeat_interleave(bm)
    tok = sids[:n].long() & 0xFFFFFF
    pad = tok >= int(nv[1])
    key = (ex << 40) | (pad.long() << 39) | (tok << 8) | (sids[:n].long() >> 24)
    order = torch.argsort(key)
    s[:n], w[:n] = sids[:n][order], sw[:n][order]
    return s, w


def main():
    tokens = [int(t) for t in sys.argv[1:]] or [4096, 8192, 16384]
    W, _ = R.make_weights(E, H, SHAPE[1])
    for T in tokens:
        x, tw, ti = R.make_routing(T, H, EGL, K, seed=7 + T)
        cfg = lookup_config(SHAPE, T)
        bm = _block_m(cfg)
        x8, xs = quantize_rows(x)
        a = _sort(ti, tw, SHAPE, 0, bm)[:4]
        p = RN.prep(ti, tw, x, bm)[:4]
        po = _token_order(p[0], p[1], p[2], p[3], bm) + tuple(p[2:4])
        ao = _token_order(a[0], a[1], a[2], a[3], bm) + tuple(a[2:4])
        n = int(a[3][0])
        same = [int((x[0][:n] == a[0][:n]).sum()) for x in (p, po, ao)]
        print(f"T={T:5d} rows equal to moe_sorting's (of {n}): prep {same[0]} prepo "
              f"{same[1]} aitero {same[2]}; first expert rows: aiter "
              f"{(a[0][:12] & 0xFFFFFF).tolist()} prep {(p[0][:12] & 0xFFFFFF).tolist()}",
              flush=True)
        out = torch.empty((T, H), dtype=torch.bfloat16, device="cuda")

        def gemm(r):
            sids, sw, seids, nv = r
            out.zero_()
            kind = cfg["kind"]
            if kind == "tp2k":
                return _tp2k_tail(cfg, x8, xs, W, sids, sw, seids, nv, out)
            if kind == "tp1k":
                ks = cfg["ksplit"]
                return F.a8w4_fmoe_bm64(x8, xs, W, sids, sw, seids, nv, out, topk=K,
                                        b_cache_mod=cfg["bcm"], waves_per_eu=2 if
                                        SHAPE[1] // ks <= 256 else 1, ksplit=ks, **RN.AKW)
            return Q.a8w4_fmoe_bm64q(x8, xs, W, sids, sw, seids, nv, out, topk=K,
                                     b_cache_mod=cfg["bcm"], **RN.AKW)

        ref = gemm(a).float().clone()
        fns = dict(aiter=lambda: gemm(a), prep=lambda: gemm(p), prepo=lambda: gemm(po),
                   aitero=lambda: gemm(ao))
        for v, fn in fns.items():
            assert R.cos_diff(ref, fn().float()) < 1e-4, v
        res = {v: [] for v in fns}
        for _ in range(7):
            for v, fn in fns.items():
                res[v].append(R.time_cold(fn, iters=20))
        med = {v: sorted(t)[len(t) // 2] for v, t in res.items()}
        print(f"T={T:5d} [{config_name(cfg)}] GEMM-only us (incl. out.zero_): " + "  ".join(
            f"{v} {t:.1f}" for v, t in med.items()), flush=True)


if __name__ == "__main__":
    main()
