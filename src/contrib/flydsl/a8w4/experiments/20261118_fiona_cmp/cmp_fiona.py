#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""FionaZZ92/aiter 28e6b92 (gfx942 backport of aiter's MegaMoEV2 small-M kernel,
``SmallMRuntime``) vs pyhip ``mega_small`` (``MegaMoeSmallEpA8W4``) and ``mega``
(``MegaMoeEpA8W4``) on the MiMo-V2.6-Flash EP8 layer (H 4096, I 2048, 256 experts, top-8),
8 GPUs, ONE job: the same inputs, the same raw mxfp4 weights, variants interleaved.

    cd src/contrib/flydsl
    python a8w4/experiments/20261118_fiona_cmp/cmp_fiona.py -m 12 24 48 96 128 192 256 512

``-m``: tokens PER RANK (FionaZZ92's M; the pyhip table's ``tokens`` = 8 M).

FionaZZ92 side: a checkout of FionaZZ92/aiter 28e6b92 at $FIONA_AITER (default
/tmp/fiona_aiter), imported with AITER_TRITON_ONLY=1 (no CK build). ``--fiona-env sglang``
(default) = what FionaZZ92/sglang 040fefd's ``mimo_mega_decode.prepare_weights`` sets
(UNPACK reuse, OPERANDS native_ab_early, PROFILE register_pressure: the setup of its
published table); ``default`` = aiter's defaults (UNPACK reuse, OPERANDS legacy, PROFILE
original). Weights: the raw mxfp4 tensors of ``ref.make_weights`` shuffled as that adapter
does (``shuffle_weight_a16w4(w1, 16, True)`` ...); runtime max_tok_per_rank 512.

pyhip side: ``mega_small`` bm 64 (the table's choice at 128-1024 tokens), tmax 1024 while
8 M <= 1024, else a second instance with tmax 4096 (beyond the solver's limit, only to
follow the same design up); ``mega`` (``--mega``, from 8 M >= 1024) with the table's knobs
(p16 l2 bcm22 up to 2048 tokens, p16 l4 bcm01 above).

Protocols (``--protocols``):
  cold     kernelkit's: every call L2 flush, the next of 8 weight copies, host barrier,
           GPU spin, MORI device barrier; e2e = the slowest rank; variants alternate ABBA
           per iteration; median (p10-p90) over iterations; ``<first>#2`` = noise floor
  graph8   a CUDA graph of 8 back-to-back calls, call i on weight copy i (8 layers);
           us per call = replay time / 8, slowest rank, median over rounds (ABBA rounds)
  graph8w  the same with all 8 calls on copy 0: FionaZZ92's bench_mimo_mega_decode_ep8.py
           (warm weights; 8 calls per graph)
Correctness: one eager call per variant and point vs an fp32 reference on the raw mxfp4
weights (x and the intermediate not quantized): cos diff and relative L2, max over ranks.
"""

import argparse
import itertools
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
FLYDSL = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))  # src/contrib/flydsl
FIONA = os.environ.get("FIONA_AITER", "/tmp/fiona_aiter")
sys.path[:0] = [FIONA, os.path.join(FIONA, "aiter", "jit", "utils"), FLYDSL]

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

WORLD, H, I, E_GLOBAL, TOPK = 8, 4096, 2048, 256, 8
E_LOCAL = E_GLOBAL // WORLD
FIONA_MTPR = 512
FIONA_ENV = {
    "sglang": dict(AITER_MEGA_GFX942_UNPACK="reuse", AITER_MEGA_GFX942_OPERANDS="native_ab_early",
                   AITER_MEGA_GFX942_PROFILE="register_pressure"),
    "default": dict(AITER_MEGA_GFX942_UNPACK="reuse", AITER_MEGA_GFX942_OPERANDS="legacy",
                    AITER_MEGA_GFX942_PROFILE="original"),
}
TIMEOUT_S = 120


def _patch_aiter():
    """AITER_TRITON_ONLY skips the top-level exports ref.make_weights imports."""
    import aiter
    from aiter.ops.enum import QuantType
    from aiter.ops.quant import get_torch_quant
    from aiter.utility import dtypes

    for k, v in dict(QuantType=QuantType, get_torch_quant=get_torch_quant, dtypes=dtypes).items():
        if not hasattr(aiter, k):
            setattr(aiter, k, v)


def _bounded(fn):
    """fn(), then wait for its GPU work at most TIMEOUT_S (a hung cross-rank wait would
    block the whole job): past that, exit this rank at once."""
    out = fn()
    ev = torch.cuda.Event()
    ev.record()
    t0 = time.monotonic()
    while not ev.query():
        if time.monotonic() - t0 > TIMEOUT_S:
            print(f"rank {torch.cuda.current_device()}: call unfinished after {TIMEOUT_S} s",
                  file=sys.stderr, flush=True)
            os._exit(1)
        time.sleep(1e-3)
    return out


def _pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, max(0, round(q * (len(v) - 1))))]


def worker(rank, port, args):
    from a8w4 import ref as R
    from a8w4.bench import dist as D

    D.init(rank, port)
    _patch_aiter()
    from aiter.ops.flydsl.kernels.mega_moe import gfx942_scaled_mma as G
    from aiter.ops.flydsl.kernels.mega_moe.mega_moe_small_m import SmallMRuntime
    from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4

    from a8w4.megamoe import MegaMoeEpA8W4, MegaMoeSmallEpA8W4
    from a8w4.moe_ep import _mega_variant

    say = (lambda *a: print(*a, flush=True)) if rank == 0 else (lambda *a: None)
    say(f"fiona gfx942 mode: MMA fp8={G.USE_NATIVE_FP8} UNPACK={G.UNPACK_MODE} "
        f"OPERANDS={G.OPERAND_MODE} id={G.GFX942_MODE_ID} "
        f"PROFILE={os.environ.get('AITER_MEGA_GFX942_PROFILE')}")

    t0 = time.time()
    W, raw = R.make_weights(E_LOCAL, H, I, seed=rank)
    native = (shuffle_weight_a16w4(raw["w1q"], 16, True).contiguous(),
              shuffle_scale_a16w4(raw["w1s"].reshape(-1, H // 32), E_LOCAL, True).contiguous(),
              shuffle_weight_a16w4(raw["w2q"], 16, False).contiguous(),
              shuffle_scale_a16w4(raw["w2s"].reshape(-1, I // 32), E_LOCAL, False).contiguous())
    NC = args.copies
    Wc = R.weight_copies(W, NC)
    Nc = [native] + [tuple(t.clone() for t in native) for _ in range(NC - 1)]
    say(f"weights + {NC} copies of both layouts: {time.time() - t0:.0f} s")

    Ms = args.m
    t0 = time.time()
    fiona = SmallMRuntime(rank=rank, world_size=WORLD, model_dim=H, inter_dim=I,
                          experts=E_GLOBAL, topk=TOPK, quant="a8w4", w1=native[0],
                          w1_scale=native[1], w2=native[2], w2_scale=native[3],
                          max_tok_per_rank=FIONA_MTPR, swiglu_limit=0.0)
    small = {}
    for tmax in sorted({1024 if m * WORLD <= 1024 else 4096 for m in Ms}):
        small[tmax] = MegaMoeSmallEpA8W4(W, rank=rank, tmax=tmax, topk=TOPK)
    need_mega = args.mega and any(m * WORLD >= 1024 for m in Ms)
    mega = MegaMoeEpA8W4(W, rank=rank, max_tokens=FIONA_MTPR, topk=TOPK) if need_mega else None
    say(f"instances: {time.time() - t0:.0f} s")

    # ---- variants: run(x, tw, ti, copy) -> bf16 [M, H] (a view of the variant's buffer)
    def run_fiona(x, tw, ti, c):
        fiona.bind_weights(*Nc[c])
        return fiona(x, tw, ti)

    def small_runner(tmax):
        def run(x, tw, ti, c):
            return small[tmax](x, ti, tw, W=Wc[c], bm=args.bm)
        return run

    def mega_runner(total):
        prod, lag, bcm = (16, 2, (2, 2)) if total <= 2048 else (16, 4, (0, 1))
        launch, slices, grid = _mega_variant(mega.H, mega.I, mega.topk, mega.token_shift,
                                             mega.nblk, prod, lag, "silu", bcm)

        def run(x, tw, ti, c):
            mega.launch = mega.launch_big = launch
            mega.slices, mega.grid, mega.W = slices, grid, Wc[c]
            return mega(x, ti, tw)
        return run, f"p{prod} l{lag} bcm{bcm[0]}{bcm[1]}"

    # ---- fp32 reference on the raw mxfp4 weights (no activation quantization)
    dec = {}

    def expert(e):
        if e not in dec:
            dec[e] = (R._mxfp4_to_f32(raw["w1q"][e], raw["w1s"][e]),
                      R._mxfp4_to_f32(raw["w2q"][e], raw["w2s"][e]))
        return dec[e]

    def reference(x, tw, ti):
        n = x.shape[0]
        ns = [n] * WORLD
        X, TW, TI = (D.gather_rows(t, ns) for t in (x.float(), tw, ti))
        out = torch.zeros((X.shape[0], H), dtype=torch.float32, device=x.device)
        loc = TI.long() - rank * E_LOCAL
        loc = torch.where((loc >= 0) & (loc < E_LOCAL), loc, torch.full_like(loc, -1))
        for e in loc[loc >= 0].unique().tolist():
            tok, slot = torch.nonzero(loc == e, as_tuple=True)
            w1, w2 = expert(e)
            gate, up = (X[tok] @ w1.t()).split(I, dim=-1)
            a = torch.nn.functional.silu(gate) * up
            out.index_add_(0, tok, (a @ w2.t()) * TW[tok, slot].view(-1, 1))
        dist.all_reduce(out)
        return out[rank * n:(rank + 1) * n]

    spin = D.spin_cycles(D.SPIN_US)
    report = []
    for M in Ms:
        total = M * WORLD
        x, tw, ti = R.make_routing(M, H, E_GLOBAL, TOPK, seed=7919 * rank + M)
        tmax = 1024 if total <= 1024 else 4096
        variants = {"fiona": run_fiona, f"mega_small(t{tmax})": small_runner(tmax)}
        mega_knobs = None
        if need_mega and total >= 1024:
            run, mega_knobs = mega_runner(total)
            variants["mega"] = run
        names = list(variants)

        # warm-up: compile, fiona's W2 table of every copy
        t0 = time.time()
        for name, run in variants.items():
            for c in range(NC):
                _bounded(lambda: run(x, tw, ti, c))
        torch.cuda.synchronize()
        dist.barrier()
        cfg = fiona._active_config
        fcfg = (f"s1 bm{cfg.stage1.sort_block_m} tn{cfg.stage1.tile_n} "
                f"wpe{cfg.stage1.waves_per_eu_hint} dcu{cfg.stage1.num_dispatch_cu} "
                f"bnt{cfg.stage1.b_nt}; s2 bm{cfg.stage2.block_m} bn{cfg.stage2.block_n}")
        say(f"\n=== M {M}/rank ({total} tokens): warm-up {time.time() - t0:.0f} s; fiona cfg "
            f"{fcfg}; mega {mega_knobs}")

        # correctness (copy 0 = the reference's weights)
        ref = reference(x, tw, ti)
        acc = {}
        for name, run in variants.items():
            y = _bounded(lambda: run(x, tw, ti, 0)).float().clone()
            torch.cuda.synchronize()
            cos = R.cos_diff(ref, y)
            rel = float((y - ref).norm() / ref.norm().clamp(min=1e-30))
            fin = bool(torch.isfinite(y).all())
            res = [None] * WORLD
            dist.all_gather_object(res, (cos, rel, fin))
            acc[name] = (max(r[0] for r in res), max(r[1] for r in res), all(r[2] for r in res))
        say("accuracy vs fp32 ref (max over ranks): " + "; ".join(
            f"{n}: cos {a[0]:.2e} relL2 {a[1]:.4f}{'' if a[2] else ' NONFINITE'}"
            for n, a in acc.items()))
        row = dict(M=M, total=total, fiona_cfg=fcfg, mega_knobs=mega_knobs, accuracy=acc)

        cnt = itertools.count()
        if "cold" in args.protocols:
            labels = names + [names[0] + "#2"]
            times = {n: [] for n in labels}
            for it in range(args.iters):
                for lab in (labels if it % 2 == 0 else labels[::-1]):
                    run = variants[lab.split("#")[0]]
                    c = next(cnt) % NC
                    times[lab].append(D.timed(lambda: run(x, tw, ti, c), spin)[0])
            res = [None] * WORLD
            dist.all_gather_object(res, times)
            out = {}
            for lab in labels:
                slow = [max(r[lab][i] for r in res) for i in range(args.iters)]
                out[lab] = (statistics.median(slow), _pct(slow, 0.1), _pct(slow, 0.9))
            row["cold"] = out
            say("cold (us, slowest rank, median [p10, p90]): " + "; ".join(
                f"{k} {v[0]:.1f} [{v[1]:.1f}, {v[2]:.1f}]" for k, v in out.items()))

        for proto in ("graph8", "graph8w"):
            if proto not in args.protocols:
                continue
            copies = list(range(NC)) if proto == "graph8" else [0] * NC
            graphs = {}
            for name, run in variants.items():
                torch.cuda.synchronize()
                dist.barrier()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    for c in copies:
                        run(x, tw, ti, c)
                torch.cuda.synchronize()
                dist.barrier()
                for _ in range(3):
                    _bounded(g.replay)
                graphs[name] = g
            torch.cuda.synchronize()
            dist.barrier()
            labels = names + [names[0] + "#2"]
            times = {n: [] for n in labels}
            for rd in range(args.rounds):
                for lab in (labels if rd % 2 == 0 else labels[::-1]):
                    g = graphs[lab.split("#")[0]]
                    torch.cuda.synchronize()
                    dist.barrier()
                    D.align_mori()
                    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    s.record()
                    for _ in range(args.replays):
                        g.replay()
                    e.record()
                    e.synchronize()
                    times[lab].append(s.elapsed_time(e) * 1e3 / (args.replays * len(copies)))
            res = [None] * WORLD
            dist.all_gather_object(res, times)
            out = {}
            for lab in labels:
                slow = [max(r[lab][i] for r in res) for i in range(args.rounds)]
                out[lab] = (statistics.median(slow), min(slow), max(slow))
            row[proto] = out
            say(f"{proto} (us per call, slowest rank, median [min, max] over rounds): " + "; ".join(
                f"{k} {v[0]:.1f} [{v[1]:.1f}, {v[2]:.1f}]" for k, v in out.items()))
            del graphs
            torch.cuda.synchronize()
        report.append(row)

    if rank == 0:
        if args.out:
            with open(args.out, "w") as f:
                json.dump(dict(args=vars(args), fiona_env={k: os.environ.get(k) for k in
                                                          FIONA_ENV["sglang"]},
                               rows=report), f, indent=1)
        lines = []
        for proto in args.protocols:
            lines += ["", f"### {proto}", "",
                      "| M/rank | tokens | fiona | mega_small | mega | fiona / mega_small | "
                      "fiona#2 delta |", "|---:|---:|---:|---:|---:|---:|---:|"]
            for r in report:
                if proto not in r:
                    continue
                t = r[proto]
                fi = t["fiona"][0]
                sm = next(v[0] for k, v in t.items() if k.startswith("mega_small"))
                mg = t.get("mega", (float("nan"),))[0]
                lines.append(f"| {r['M']} | {r['total']} | {fi:.1f} | {sm:.1f} | {mg:.1f} | "
                             f"{fi / sm:.2f}x | {100 * (t['fiona#2'][0] / fi - 1):+.1f}% |")
        lines += ["", "### accuracy vs fp32 (cos diff / rel L2, max over ranks)", "",
                  "| M/rank | fiona | mega_small | mega |", "|---:|---|---|---|"]
        for r in report:
            a = r["accuracy"]
            sm = next(v for k, v in a.items() if k.startswith("mega_small"))
            mg = a.get("mega")
            lines.append(f"| {r['M']} | {a['fiona'][0]:.2e} / {a['fiona'][1]:.4f} | "
                         f"{sm[0]:.2e} / {sm[1]:.4f} | "
                         + (f"{mg[0]:.2e} / {mg[1]:.4f} |" if mg else "- |"))
        print("\n".join(lines), flush=True)
    D.finish()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-m", type=int, nargs="+", default=[12, 24, 48, 96, 128, 192, 256, 512],
                    help="tokens per rank (<= 512)")
    ap.add_argument("--protocols", nargs="*", default=["cold", "graph8", "graph8w"],
                    choices=["cold", "graph8", "graph8w"],
                    help="none: compile + accuracy check only")
    ap.add_argument("--fiona-env", default="sglang", choices=list(FIONA_ENV))
    ap.add_argument("--mega", action=argparse.BooleanOptionalAction, default=True,
                    help="also time pyhip mega at >= 1024 tokens in total")
    ap.add_argument("--bm", type=int, default=64, help="mega_small m-tile rows")
    ap.add_argument("--copies", type=int, default=8)
    ap.add_argument("--iters", type=int, default=30, help="cold: timed calls per variant")
    ap.add_argument("--rounds", type=int, default=7, help="graph: rounds per variant")
    ap.add_argument("--replays", type=int, default=20, help="graph: replays per round")
    ap.add_argument("--out", default="", help="json with every number (rank 0)")
    args = ap.parse_args()
    assert all(0 < m <= FIONA_MTPR for m in args.m), args.m
    os.environ.update(FIONA_ENV[args.fiona_env])
    os.environ.setdefault("AITER_TRITON_ONLY", "1")
    os.environ.setdefault("MORI_SHMEM_HEAP_SIZE", "8G")
    from a8w4.bench import dist as D

    reason = D.skip_reason()
    if reason:
        sys.exit(reason)
    D.spawn(worker, (args,))


if __name__ == "__main__":
    main()
