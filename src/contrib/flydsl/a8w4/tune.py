# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Tuner for ``a8w4_moe``: measures every candidate kernel config (TP and EP families)
per token count and stores ALL results; the runtime picks the fastest config that is
allowed in its context (activation, CUDA-graph capture, token count).

    # one process per GPU, disjoint token lists, then merge:
    HIP_VISIBLE_DEVICES=0 python -m a8w4.tune tune --case mimo-v2.6-flash-tp8 -t 1 2 4 --out /tmp/p0.csv
    python -m a8w4.tune merge /tmp/p*.csv            # into configs/a8w4_moe_tuned.csv
    python -m a8w4.tune show --case mimo-v2.6-flash-tp8
    python -m a8w4.tune verify --case mimo-v2.6-flash-ep8 -t 1 100 3000

Run from the directory that contains ``a8w4/`` (or put it on PYTHONPATH);
``python a8w4/tune.py ...`` works too. Timing: cold caches (L2 + MALL flushed),
host launch overhead excluded, median of ``--iters`` (see ``ref.time_cold``).
"""

import argparse
import csv
import os
import sys
import time

if __package__ in (None, ""):  # run as a script: make the package importable
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import a8w4  # noqa: E402,F401

    __package__ = "a8w4"

import torch  # noqa: E402

from . import ref as R  # noqa: E402
from .moe import (  # noqa: E402
    DEFAULT_CSV,
    FIELDS,
    _load_table,
    a8w4_moe,
    config_name,
    default_config,
    lookup_config,
    parse_config,
    run_config,
    supported,
)

# (H, I, E local, E global, topk, expert_offset). XiaomiMiMo/MiMo-V2.6-Flash(-RL):
# hidden 4096, moe_intermediate 2048, 256 routed experts, top-8, SiLU.
CASES = {
    "mimo-v2.6-flash-tp8": (4096, 256, 256, 256, 8, 0),
    "mimo-v2.6-flash-tp4": (4096, 512, 256, 256, 8, 0),
    "mimo-v2.6-flash-ep8": (4096, 2048, 32, 256, 8, 0),
}
TOKENS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 320, 384, 448, 512, 640,
          768, 1024, 1280, 1536, 2048, 2560, 3072, 4096, 6144, 8192, 12288, 16384, 24576, 32768]
COS_TOL = 5e-3  # vs the fp8-emulating fp32 reference (typical 1e-5 .. 1e-4)
EP_CFGS = ("t16", "s16", "s32", "m64", "l128")


def candidates(shape):
    """Every config worth measuring on ``shape`` (structurally supported ones)."""
    H, I, E, EGl, K = shape
    out = []
    for bm in (32, 64):
        for g1 in (1, 2):
            for g2 in ("base", "nt4"):
                out.append(dict(kind="tp2k", block_m=bm, g1_bcm=g1, g2=g2))
    # fused: each CTA takes 256 / 384 / 512 inter columns (256: 2 waves/SIMD, best for
    # small/medium batches; 384-512: 1 wave/SIMD, best for large batches at large I)
    splits = sorted({I // c for c in (256, 384, 512) if I % c == 0})
    for bcm in (1, 2):
        for ks in splits:
            out.append(dict(kind="tp1k", bcm=bcm, ksplit=ks))
    out.append(dict(kind="tp1kq", bcm=1))
    out += [dict(kind="ep3k", cfg=n) for n in EP_CFGS]
    out.append(dict(kind="ep1k", maxm=512))
    return [c for c in out if supported(c, shape)]


def _key(r):
    return tuple(str(r[k]) for k in ("model_dim", "inter_dim", "experts", "experts_global",
                                     "topk", "routing", "num_tokens", "config"))


def write_rows(path, new_rows):
    """Merge rows into ``path`` (same shape / point / config replaced), sorted."""
    old = []
    if os.path.exists(path):
        with open(path) as fh:
            old = list(csv.DictReader(fh))
    keys = {_key(r) for r in new_rows}
    rows = [r for r in old if _key(r) not in keys] + list(new_rows)
    rows.sort(key=lambda r: (int(r["model_dim"]), int(r["inter_dim"]), int(r["experts"]),
                             int(r["experts_global"]), int(r["topk"]), r["routing"],
                             int(r["num_tokens"]), float(r["us"])))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def _case(args):
    if args.shape:
        H, I, E, EGl, K = (int(v) for v in args.shape.split(","))
        return (H, I, E, EGl, K), args.expert_offset
    *shape, e0 = CASES[args.case]
    return tuple(shape), e0


def tune(shape, e0, tokens, out_path, routing="uniform", iters=20, rounds=3, log=print):
    """Per token count, ``rounds`` rounds, each with a fresh routing draw (at small
    batches the per-rank work depends on the draw): check every candidate against
    the reference, time it (candidates interleaved, so clock / power drift hits all
    of them); store the median per candidate."""
    H, I, E, EGl, K = shape
    W, raw = R.make_weights(E, H, I)
    cands = candidates(shape)
    log(f"shape H{H} I{I} E{E}/{EGl} top{K} e0 {e0}: {len(cands)} candidates")
    for tok in tokens:
        names = [config_name(c) for c in cands if supported(c, shape, tokens=tok)]
        times = {n: [] for n in names}
        cos = dict.fromkeys(names, 0.0)
        bad = {}
        for rnd in range(rounds):
            x, tw, ti = R.make_routing(tok, H, EGl, K, seed=101 * tok + rnd, routing=routing)
            ref = R.ref_moe(x, tw, ti, raw, e0)
            for name in names:
                if name in bad:
                    continue
                fn = lambda name=name: run_config(name, x, W, tw, ti,  # noqa: E731
                                                  expert_offset=e0, num_experts=EGl)
                try:
                    c = R.cos_diff(ref, fn().float())
                except Exception as ex:  # noqa: BLE001
                    bad[name] = f"FAIL {type(ex).__name__}: {str(ex)[:120]}"
                    continue
                if not c < COS_TOL:
                    bad[name] = f"BAD cos {c:.1e}"
                    continue
                cos[name] = max(cos[name], c)
                times[name].append(R.time_cold(fn, iters=iters))
        rows = []
        for name in names:
            if name in bad:
                continue
            us = sorted(times[name])[len(times[name]) // 2]
            rows.append(dict(model_dim=H, inter_dim=I, experts=E, experts_global=EGl, topk=K,
                             routing=routing, num_tokens=tok, config=name, us=f"{us:.1f}",
                             cos=f"{cos[name]:.1e}"))
        msgs = [f"{n}: {m}" for n, m in bad.items()]
        rows.sort(key=lambda r: float(r["us"]))
        if rows:
            write_rows(out_path, rows)
        top = " | ".join(f"{r['config']} {r['us']}" for r in rows[:3])
        log(f"tok {tok:6d}: {top}" + (f"  [{'; '.join(msgs)}]" if msgs else ""))
        torch.cuda.empty_cache()


def _points(csv_path, shape, routing):
    rows = _load_table(csv_path, os.path.getmtime(csv_path)).get((*shape, routing))
    if not rows:
        raise SystemExit(f"no rows for {shape} {routing} in {csv_path}")
    return rows


def show(shape, csv_path, routing="uniform"):
    """Markdown: per point the fastest config, the fastest of each family and the
    graph-safe pick when it differs."""
    print("| tokens | best | us | best TP family | us | best EP family | us | graph-safe pick |")
    print("|---:|---|---:|---|---:|---|---:|---|")
    for n, cands in _points(csv_path, shape, routing):
        us0, best = cands[0]
        tp = next(((u, c) for u, c in cands if c["kind"].startswith("tp")), (None, None))
        ep = next(((u, c) for u, c in cands if c["kind"].startswith("ep")), (None, None))
        safe = lookup_config(shape, n, capturing=True, routing=routing, csv_path=csv_path)
        cell = lambda uc: (config_name(uc[1]), f"{uc[0]:.1f}") if uc[1] else ("-", "-")  # noqa: E731
        print(f"| {n} | {config_name(best)} | {us0:.1f} | {' | '.join(cell(tp))} | "
              f"{' | '.join(cell(ep))} | "
              f"{'' if safe == best else config_name(safe)} |")


def verify(shape, e0, tokens, csv_path, routing="uniform", iters=20, log=print):
    """``a8w4_moe`` (table lookup) at each token count, also between tuned points:
    cos vs the reference, time vs the table entry and vs the untuned default."""
    H, I, E, EGl, K = shape
    W, raw = R.make_weights(E, H, I)
    pts = dict(_points(csv_path, shape, routing))
    for tok in tokens:
        x, tw, ti = R.make_routing(tok, H, EGl, K, seed=7 + tok, routing=routing)
        ref = R.ref_moe(x, tw, ti, raw, e0)
        cfg = lookup_config(shape, tok, routing=routing, csv_path=csv_path)
        fn = lambda: a8w4_moe(x, W, tw, ti, expert_offset=e0, num_experts=EGl,  # noqa: E731
                              routing=routing, csv_path=csv_path)
        c = R.cos_diff(ref, fn().float())
        us = R.time_cold(fn, iters=iters)
        dflt = default_config(shape, tok)
        dus = R.time_cold(lambda: run_config(dflt, x, W, tw, ti, expert_offset=e0,
                                             num_experts=EGl), iters=iters)
        n = next((p for p in sorted(pts) if p >= tok), max(pts))
        tus = next(u for u, cc in pts[n] if cc == cfg)
        log(f"tok {tok:6d} -> point {n:6d}: {config_name(cfg):24s} {us:8.1f} us (table {tus:8.1f})"
            f" | default {config_name(dflt):24s} {dus:8.1f} us | x{dus / us:.2f} | cos {c:.1e}"
            + ("" if c < COS_TOL else " BAD"))


def merge(parts, csv_path):
    rows = []
    for p in parts:
        with open(p) as fh:
            rows += list(csv.DictReader(fh))
    write_rows(csv_path, rows)
    print(f"merged {len(rows)} rows from {len(parts)} files into {csv_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("mode", choices=["tune", "merge", "show", "verify", "candidates"])
    p.add_argument("parts", nargs="*", help="merge: tune output files")
    p.add_argument("--case", default="mimo-v2.6-flash-tp8", choices=sorted(CASES))
    p.add_argument("--shape", default=None, help="H,I,E_local,E_global,topk (overrides --case)")
    p.add_argument("--expert-offset", type=int, default=0)
    p.add_argument("-t", type=int, nargs="*", default=None, help="token counts")
    p.add_argument("--routing", default="uniform", choices=R.ROUTINGS)
    p.add_argument("--csv", default=DEFAULT_CSV)
    p.add_argument("--out", default=None, help="tune: output file (default --csv)")
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--rounds", type=int, default=3, help="tune: interleaved timing rounds")
    args = p.parse_args()
    log = lambda s: print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True)  # noqa: E731
    if args.mode == "merge":
        return merge(args.parts, args.csv)
    shape, e0 = _case(args)
    if args.mode == "candidates":
        for c in candidates(shape):
            print(config_name(c))
    elif args.mode == "tune":
        tune(shape, e0, args.t or TOKENS, args.out or args.csv, args.routing, args.iters,
             args.rounds, log)
    elif args.mode == "show":
        show(shape, args.csv, args.routing)
    else:
        verify(shape, e0, args.t or [1, 3, 20, 100, 300, 1000, 3000, 10000], args.csv,
               args.routing, args.iters, log)


if __name__ == "__main__":
    main()
