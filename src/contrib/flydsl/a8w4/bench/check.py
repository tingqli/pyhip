# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Correctness of variants, and helpers for experiment scripts.

    python -m a8w4.bench check --case tp8 -t 1 77 600 4096 -v @table "tp1k bcm2 ks1" --graph
    python -m a8w4.bench check --shape 4096,2048,32,256,1 -t 64 4096 -v l128 "x=l128:g1.pf=1"

Per token count, for every variant: cos diff vs the fp32 reference (< 5e-3), vs the
first variant (the base; > 1e-4 flagged: the bf16 atomics alone give ~1e-5), the worst
of ``--repeats`` further calls vs the first (stale state, e.g. a counter that is not
reset, shows up here), bit equality with the base (``--bitwise``: deterministic only
with one route per token, i.e. topk 1, see README 9.4) and with ``--graph`` CUDA-graph
capture + replays vs the eager result. Exit status 1 on any failure.

Helpers (import from experiment scripts): ``region_keys`` (row set of every expert region
of a sorted routing, order-independent), ``poison_`` (fill buffers so rows a kernel fails
to write stand out), ``stability``, ``graph_replay``.
"""

import argparse

import torch

from .. import ref as R
from ..moe import MOE, shape_vocab
from . import core

POISON = {torch.bfloat16: float("nan"), torch.float32: float("nan"), torch.uint8: 0x7F,
          torch.int32: -1}


def region_keys(sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, block_m):
    """(expert << 32 | sorted id) of every valid row, sorted, and the rows' weights: the
    row set of each expert region regardless of the order inside it (e.g. tp_prep vs
    aiter moe_sorting: torch.equal on both results)."""
    n = int(num_valid_ids[0])
    ex = sorted_expert_ids[:n // block_m].long().repeat_interleave(block_m)
    key = (ex << 32) | (sorted_ids[:n].long() & 0xFFFFFFFF)
    order = torch.argsort(key)
    return key[order], sorted_weights[:n][order]


def poison_(*tensors, value=None):
    """Fill ``tensors`` with a value no kernel writes (NaN / 0x7F fp8 = 240 / -1), so rows a
    kernel skips show up in the result (run before the kernel under test)."""
    for t in tensors:
        t.fill_(POISON[t.dtype] if value is None else value)
    return tensors


def stability(fn, repeats=5):
    """Worst cos diff of ``repeats`` further calls of ``fn()`` vs the first."""
    first = fn().float()
    return max((R.cos_diff(first, fn().float()) for _ in range(repeats)), default=0.0)


def graph_replay(fn, replays=5):
    """Capture ``fn()`` in a CUDA graph (after a side-stream warm-up, as frameworks do) and
    replay it -> worst cos diff of the replays vs an eager call."""
    eager = fn().float()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = fn()
    worst = 0.0
    for _ in range(replays):
        g.replay()
        torch.cuda.synchronize()
        worst = max(worst, R.cos_diff(eager, out.float()))
    return worst


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m a8w4.bench check", description=__doc__,
                                 epilog=core.VARIANT_HELP,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-t", "--tokens", type=int, nargs="+", required=True)
    ap.add_argument("-v", "--variants", nargs="+", default=["@table"])
    core.add_case_args(ap)
    ap.add_argument("--seed", type=int, default=7, help="routing seed = --seed + T")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--graph", action="store_true", help="also CUDA-graph capture + replay")
    ap.add_argument("--bitwise", action="store_true", help="also bit equality with the base")
    core.add_out_args(ap, "check")
    a = ap.parse_args(argv)
    if not a.verbose:
        core.quiet()
    case = core.case_from_args(a)
    variants = [core.parse_variant(s) for s in a.variants]
    H, I, E, EGl, K = case.shape
    rec = core.Recorder(a.out, a.name, core.run_meta("check", a))
    rec.text(case.describe() + "\n")
    W, raw = R.make_weights(E, H, I)
    rows, bad = [], 0
    for T in a.tokens:
        x, tw, ti = case.inputs(T, a.seed + T, a.local_experts_per_token)
        ref = core.ref_out(case, x, tw, ti, raw)
        base = None
        for v in variants:
            r = core.resolve(v, case, T)
            if isinstance(r, str):
                rows.append([T, v.label, "-", "n/a", "", "", "", "", r])
                continue
            fn = lambda r=r: r.run(x, W, tw, ti)  # noqa: E731
            try:
                out = fn()
                torch.cuda.synchronize()
                c = R.cos_diff(ref, out.float())
                if base is None:
                    base = out.clone()
                cb = R.cos_diff(base.float(), out.float())
                jit = stability(fn, a.repeats)
                bw = bool(torch.equal(base, out)) if a.bitwise else None
                g = None
                if a.graph and r.cfg is not None and MOE.why_not(
                        r.cfg, shape_vocab(case.shape, T, case.act), capturing=True):
                    g = "n/a (not graph-safe)"
                elif a.graph:
                    g = graph_replay(fn)
                fails = [s for s, f in (("ref", not c < core.COS_TOL),
                                        ("base", cb > core.DIFF_TOL), ("repeat", jit > core.DIFF_TOL),
                                        ("graph", isinstance(g, float) and g > core.DIFF_TOL),
                                        ("bitwise", a.bitwise and not bw)) if f]
                ok = "OK" if not fails else "FAIL " + ",".join(fails)
            except Exception as ex:  # noqa: BLE001
                c = cb = jit = g = bw = None
                ok = f"FAIL {type(ex).__name__}: {str(ex)[:160]}"
            bad += ok != "OK"
            rows.append([T, r.label, f"`{r.config}`", core.fmt_cos(c), core.fmt_cos(cb),
                         core.fmt_cos(jit), "-" if bw is None else ("yes" if bw else "no"),
                         core.fmt_cos(g) if isinstance(g, float) else (g or "-"), ok])
            rec.row(T=T, variant=r.label, config=r.config, cos_ref=c, cos_base=cb, repeat=jit,
                    bitwise=bw, graph=g, result=ok)
        torch.cuda.empty_cache()
    rec.text(core.md_table(["tokens", "variant", "config", "cos ref", "cos base", "repeat",
                            "bitwise", "graph", "result"], [[str(c) for c in r] for r in rows],
                           right=(0, 3, 4, 5)))
    rec.text("\nALL OK" if not bad else f"\n{bad} FAILED")
    rec.done()
    if bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
