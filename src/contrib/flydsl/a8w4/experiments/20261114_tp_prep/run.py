"""TP8 MiMo shape (I = 256, 256 experts): the fused TP prep (tp_prep.py: sort + x quant
+ out zeroing in one launch) in place of aiter moe_sorting + quantize_rows, feeding the
tuned config's own GEMM kernels (tp2k / tp1k / tp1kq), or, at small T, the split-K EP
gemm1 (20261114_splitk_gemm1) + tp_gemm2.

    cd src/contrib/flydsl
    HIP_VISIBLE_DEVICES=0 python a8w4/experiments/20261114_tp_prep/run.py unit 1 16 1024 8192
    HIP_VISIBLE_DEVICES=0 python a8w4/experiments/20261114_tp_prep/run.py check 1 1024 4096
    HIP_VISIBLE_DEVICES=0 python a8w4/experiments/20261114_tp_prep/run.py bench 1024 2048 --prof

Variants: "base" (tuned config, aiter sort + quant), "prep" (tp_prep v2 + the tuned
config's kernels), "prep1" (v1: tp_prep_v1.py), "prepr{N}" (v2, N routes per sort
WG), "prepwide" (v2, aiter-sized capacity), "prep@<config>" (tp_prep + an explicit
config, e.g. "prep@tp2k bm64 g1bcm2 g2nt4"), "sk{S}p{PF}t{N}" (tp_prep + split-K EP
gemm1 with S splits when <= N valid tiles + the tuned tp2k config's gemm2).
"""
import argparse
import os
import re
import sys
from collections import defaultdict

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "../../..")))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "../20261114_splitk_gemm1"))
sys.path.insert(0, os.path.join(HERE, "../20261114_r2_baseline"))
from a8w4 import ref as R  # noqa: E402
from a8w4.kernels import tp_fmoe as F  # noqa: E402
from a8w4.kernels import tp_fmoe_q as Q  # noqa: E402
from a8w4.kernels import tp_gemm1 as G1  # noqa: E402
from a8w4.kernels import tp_gemm2 as G2  # noqa: E402
from a8w4.kernels.row_quant import quantize_rows  # noqa: E402
from a8w4.moe import _sort, config_name, lookup_config, parse_config, run_config  # noqa: E402
import ep_gemm_sk as SKG  # noqa: E402
import tp_prep as TPP  # noqa: E402
import tp_prep_v1 as TPP1  # noqa: E402
from breakdown import kernel_times  # noqa: E402

SHAPE = (4096, 256, 256, 256, 8)
H, I, E, EGL, K = SHAPE
AKW = dict(act="silu")


def _block_m(cfg):
    return {"tp2k": cfg.get("block_m"), "tp1k": F.BM, "tp1kq": Q.BM}[cfg["kind"]]


def _tp2k_tail(cfg, x8, xs, W, sids, sw, seids, nv, out, sk=None):
    bm = cfg["block_m"]
    groups = I // 128
    sorted_size = seids.numel() * bm
    buf = torch.empty(sorted_size * (I + groups * 4), dtype=torch.uint8, device=x8.device)
    nd = sorted_size * I
    a2s = buf[nd:].view(torch.float32).view(sorted_size, groups)
    if sk is None:
        G1.a8w4_gemm1(x8, xs, W["w1"], W["w1_scale"], W["w1_exp"], sids, seids, nv, buf,
                      num_experts=E, tile_m=bm, epilog="q8s", inter_dim=I, topk=K,
                      tile_n=128, b_cache_mod=cfg["g1_bcm"], xcd_swizzle=1, dyn_m=bm > 32,
                      fast_dec=2, **AKW)
    else:
        rowv = tuple(r for r in (16, 32, 64, 128) if r <= bm)
        SKG.ep_gemm1(x8, xs, W, sids, seids, nv, buf[:nd].view(sorted_size, I), a2s,
                     adt="fp8", block_m=bm, rowv=rowv, zero=None, sched=None, ni=128, nw=8,
                     sch=3, pf=sk["pf"], arb=256, sk=sk["sk"], sk_tiles=sk["sk_tiles"])
    a2 = buf[:nd].view(torch.float8_e4m3fnuz).view(sorted_size, I)
    kw = dict(tile_n=128, b_cache_mod=2, xcd_swizzle=0, dyn_rows=True)
    if cfg["g2"] == "nt4":
        kw.update(n_tiles=4, epilog="direct64", row_vgpr=True)
    G2.a8w4_gemm2(a2, a2s, W["w2"], W["w2_scale"], sids, sw, seids, nv, out,
                  num_experts=E, tile_m=bm, a_group=128, fast_dec=2, **kw)
    return out


def prep(ti, tw, x, bm, ver=2, **kw):
    if ver == 1:
        return TPP1.tp_prep(ti, tw, x, block_m=bm, num_experts=E, **kw)
    return TPP.tp_prep(ti, tw, x, block_m=bm, num_experts=E, **kw)


def run_prep(cfg, x, W, tw, ti, sk=None, ver=2, **pkw):
    """tp_prep + the GEMM kernels of ``cfg`` (same calls as a8w4.moe._run_*)."""
    kind = cfg["kind"]
    sids, sw, seids, nv, x8, xs, out = prep(ti, tw, x, _block_m(cfg), ver, **pkw)
    if kind == "tp2k":
        return _tp2k_tail(cfg, x8, xs, W, sids, sw, seids, nv, out, sk)
    assert sk is None
    if kind == "tp1k":
        ks = cfg["ksplit"]
        return F.a8w4_fmoe_bm64(x8, xs, W, sids, sw, seids, nv, out, topk=K,
                                b_cache_mod=cfg["bcm"],
                                waves_per_eu=2 if I // ks <= 256 else 1, ksplit=ks, **AKW)
    assert kind == "tp1kq", kind
    return Q.a8w4_fmoe_bm64q(x8, xs, W, sids, sw, seids, nv, out, topk=K,
                             b_cache_mod=cfg["bcm"], **AKW)


def make_fn(v, x, W, tw, ti, T):
    tuned = lookup_config(SHAPE, T)
    if v == "base":
        name = config_name(tuned)
        return lambda: run_config(name, x, W, tw, ti, expert_offset=0, num_experts=EGL)
    if v.startswith("base@"):
        name = v[5:]
        return lambda: run_config(name, x, W, tw, ti, expert_offset=0, num_experts=EGL)
    if v == "prep":
        return lambda: run_prep(tuned, x, W, tw, ti)
    if v == "prepwide":  # ep_prep-style capacity (M*K // BM + E blocks)
        return lambda: run_prep(tuned, x, W, tw, ti, tight=False)
    if v == "prep1":
        return lambda: run_prep(tuned, x, W, tw, ti, ver=1)
    m = re.fullmatch(r"prepr(\d+)", v)
    if m:
        return lambda: run_prep(tuned, x, W, tw, ti, rng=int(m[1]))
    if v.startswith("prep@"):
        cfg = parse_config(v[5:])
        return lambda: run_prep(cfg, x, W, tw, ti)
    m = re.fullmatch(r"sk(\d+)p(\d+)t(\d+)", v)
    assert m, v
    assert tuned["kind"] == "tp2k", config_name(tuned)
    sk = dict(sk=int(m[1]), pf=int(m[2]), sk_tiles=int(m[3]))
    return lambda: run_prep(tuned, x, W, tw, ti, sk=sk)


# ---- unit: tp_prep vs aiter moe_sorting + quantize_rows ----------------------------


def _region_keys(sids, sw, seids, nv, bm):
    """(expert << 32 | sorted id) of every valid row, sorted, with the weights."""
    n = int(nv[0])
    ex = seids[:n // bm].long().repeat_interleave(bm)
    key = (ex << 32) | (sids[:n].long() & 0xFFFFFFFF)
    order = torch.argsort(key)
    return key[order], sw[:n][order]


def unit(tokens, bms, graph, ver, pkw):
    bad = 0
    for T in tokens:
        x, tw, ti = R.make_routing(T, H, EGL, K, seed=7 + T)
        xa8, xas = quantize_rows(x)
        for bm in bms:
            sa, wa, ea, nva, _ = _sort(ti, tw, SHAPE, 0, bm)
            junk = torch.full((T, H), float("nan"), dtype=torch.bfloat16, device="cuda")
            del junk  # the allocator hands this block to tp_prep's out
            outs = [prep(ti, tw, x, bm, ver, **pkw) for _ in range(3)]
            torch.cuda.synchronize()
            ka, wka = _region_keys(sa, wa, ea, nva, bm)
            msgs = []
            for it, (s, w, e, nv, x8, xs, out) in enumerate(outs):
                if nv.tolist() != nva.tolist():
                    msgs.append(f"call{it} nv {nv.tolist()} vs {nva.tolist()}")
                    continue
                nb = int(nv[0]) // bm
                if not torch.equal(e[:nb], ea[:nb]):
                    msgs.append(f"call{it} block experts differ")
                k, wk = _region_keys(s, w, e, nv, bm)
                if not torch.equal(k, ka):
                    msgs.append(f"call{it} sorted ids differ")
                elif not torch.equal(wk, wka):
                    msgs.append(f"call{it} weights differ")
                if not (torch.equal(x8.view(torch.uint8), xa8.view(torch.uint8))
                        and torch.equal(xs, xas)):
                    msgs.append(f"call{it} x8/xs differ")
                if int(torch.count_nonzero(out)) or bool(torch.isnan(out).any()):
                    msgs.append(f"call{it} out not zeroed")
            cap = outs[0][2].numel()
            if graph:
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    gout = prep(ti, tw, x, bm, ver, **pkw)
                for it in range(4):
                    gout[0].fill_(-1)
                    gout[6].fill_(1.0)
                    g.replay()
                    torch.cuda.synchronize()
                    s, w, e, nv, x8, xs, out = gout
                    k, wk = _region_keys(s, w, e, nv, bm)
                    if not (nv.tolist() == nva.tolist() and torch.equal(k, ka)
                            and torch.equal(wk, wka) and int(torch.count_nonzero(out)) == 0):
                        msgs.append(f"graph replay {it} differs")
            ok = not msgs
            bad += not ok
            print(f"T={T:5d} bm={bm:3d} nv={nva.tolist()} blocks tp_prep {cap} vs aiter "
                  f"{ea.numel()}  {'OK' if ok else 'FAIL: ' + '; '.join(msgs)}", flush=True)
    print("ALL OK" if bad == 0 else f"{bad} FAILURES")


# ---- e2e check / bench ---------------------------------------------------------------


def check(W, raw, variants, tokens):
    bad = 0
    for T in tokens:
        x, tw, ti = R.make_routing(T, H, EGL, K, seed=7 + T)
        ref = R.ref_moe(x, tw, ti, raw, 0)
        base = make_fn("base", x, W, tw, ti, T)().float()
        for v in variants:
            fn = make_fn(v, x, W, tw, ti, T)
            o1 = fn().float()
            outs = [fn().float() for _ in range(5)]
            torch.cuda.synchronize()
            cos = R.cos_diff(o1, ref)
            cb = R.cos_diff(o1, base)
            # gemm2's bf16 atomics add in a run-dependent order: runs differ by ~1e-5
            jit = max(R.cos_diff(o, o1) for o in outs)
            ok = cos < 5e-3 and cb < 1e-4 and jit < 1e-4
            bad += not ok
            print(f"T={T:5d} [{config_name(lookup_config(SHAPE, T))}] {v:28s} cos {cos:.1e}  "
                  f"vs base {cb:.1e}  jitter {jit:.1e}  {'OK' if ok else 'FAIL'}", flush=True)
    print("ALL OK" if bad == 0 else f"{bad} FAILURES")


def _short(name):
    for p in ("tp_prep", "ep_gemm1", "a8w4_gemm1", "a8w4_gemm2", "a8w4_fmoe_bm64q",
              "a8w4_fmoe_bm64", "moe_sorting", "kernel_0"):
        if p in name:
            return p
    return name[:30]


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
        print(f"tp8 T={T:5d} [{config_name(lookup_config(SHAPE, T))}] e2e(us)" + "".join(
            f"  {v} {med[v]:.1f}({(med[v] / b - 1) * 100:+.1f}%)" for v in variants),
            flush=True)
        if prof:
            for v, fn in fns.items():
                parts = defaultdict(float)
                for k, t in kernel_times(fn, iters=iters).items():
                    parts[_short(k)] += t
                print(f"    {v:28s} " + "  ".join(
                    f"{k} {t:.1f}" for k, t in sorted(parts.items(), key=lambda kv: -kv[1])),
                    flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("unit", "check", "bench"))
    ap.add_argument("tokens", type=int, nargs="*",
                    default=[1, 2, 4, 8, 16, 1024, 2048, 4096, 8192, 16384, 32768])
    ap.add_argument("--variants", default="base,prep")
    ap.add_argument("--bm", default="32,64")
    ap.add_argument("--graph", action="store_true")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--prof", action="store_true")
    ap.add_argument("--ver", type=int, default=2, help="unit: tp_prep version")
    ap.add_argument("--rng", type=int, default=None, help="unit: v2 routes per sort WG")
    a = ap.parse_args()
    if a.mode == "unit":
        unit(a.tokens, [int(b) for b in a.bm.split(",")], a.graph, a.ver,
             {} if a.rng is None else dict(rng=a.rng))
        return
    variants = a.variants.split(";") if ";" in a.variants else a.variants.split(",")
    W, raw = R.make_weights(E, H, I)
    if a.mode == "check":
        check(W, raw, variants, a.tokens)
    else:
        bench(W, variants, a.tokens, a.iters, a.rounds, a.prof)


if __name__ == "__main__":
    main()
