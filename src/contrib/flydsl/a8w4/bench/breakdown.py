# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Per-kernel time breakdown (torch profiler) and e2e of variants.

    python -m a8w4.bench breakdown --case tp8 -t 1024 4096 32768
    python -m a8w4.bench breakdown --case ep8 -t 1 2 4 --local-experts-per-token 1
    python -m a8w4.bench breakdown --case ep8 -t 8192 -v @table "ep3k l128"

Every timed call (e2e and profiled) is cold: L2 flush, the next of ``--wcopies``
weight copies, GPU sleep (host overhead excluded). Kernel times: median over
``--iters`` profiled calls per kernel family (the name up to its parameters); "kernels"
is their sum, e2e - kernels the gaps between launches.
"""

import argparse
import re
import statistics
from collections import defaultdict

import torch

from .. import ref as R
from . import core

_FAMILY = re.compile(r"^([a-z][a-z0-9]*(?:_[a-z0-9]+)*?)(?:_(?:fp8|bf16|h\d+)(?:_.*)?)?$")
_KNOWN = ("moe_sorting", "quant", "amax", "fill", "elementwise", "copy", "zero")


def family(name):
    """Kernel family of a kernel name: "ep_gemm1_fp8_h4096_..." -> "ep_gemm1"."""
    m = _FAMILY.match(name)
    if m:
        return m[1]
    return next((k for k in _KNOWN if k in name), name[:40])


def kernel_times(fn, Ws, iters):
    """{kernel family: median us} of ``fn(W)`` over ``iters`` cold calls, each on the next
    weight copy."""
    from torch.profiler import ProfilerActivity, profile

    for _ in range(3):
        fn(Ws[0])
    torch.cuda.synchronize()
    per = defaultdict(list)
    for i in range(iters):
        W = Ws[(i + 1) % len(Ws)]
        R.flush_cache()
        torch.cuda._sleep(2_000_000)
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            fn(W)
            torch.cuda.synchronize()
        seen = defaultdict(float)
        for ev in prof.events():
            if ev.device_type == torch.autograd.DeviceType.CUDA:
                seen[family(ev.name)] += getattr(ev, "device_time_total", None) or ev.cuda_time_total
        for k, v in seen.items():
            per[k].append(v)
    return {k: statistics.median(v) for k, v in per.items()}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m a8w4.bench breakdown", description=__doc__,
                                 epilog=core.VARIANT_HELP,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-t", "--tokens", type=int, nargs="+", required=True)
    ap.add_argument("-v", "--variants", nargs="+", default=["@table"])
    core.add_case_args(ap)
    ap.add_argument("--wcopies", type=int, default=R.WCOPIES)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--seed", type=int, default=7, help="routing seed = --seed + T")
    core.add_out_args(ap, "breakdown")
    a = ap.parse_args(argv)
    if not a.verbose:
        core.quiet()
    case = core.case_from_args(a)
    variants = [core.parse_variant(s) for s in a.variants]
    H, I, E, EGl, K = case.shape
    rec = core.Recorder(a.out, a.name, core.run_meta("breakdown", a))
    routing = (f"{a.local_experts_per_token} local experts per token"
               if a.local_experts_per_token else f"{case.routing} routing")
    rec.text(f"{case.describe()}; inputs: {routing}, seed {a.seed} + T; every call cold "
             f"(flush + next of {a.wcopies} weight copies); medians of {a.iters} calls\n")
    W, raw = R.make_weights(E, H, I)
    Ws = R.weight_copies(W, a.wcopies)
    timer = core.Timer(Ws)
    rows, fams = [], {}
    for T in a.tokens:
        x, tw, ti = case.inputs(T, a.seed + T, a.local_experts_per_token)
        ref = core.ref_out(case, x, tw, ti, raw)
        routes, _ = core.route_blocks(ti, case, [])
        for v in variants:
            r = core.resolve(v, case, T)
            if isinstance(r, str):
                rec.text(f"- {T} tokens, {v.label}: {r}")
                continue
            fn = lambda Wc, r=r: r.run(x, Wc, tw, ti)  # noqa: E731
            cos = R.cos_diff(ref, fn(W).float())
            e2e = statistics.median(timer.times(fn, "coldrot", a.iters))
            kt = kernel_times(fn, Ws, a.iters)
            for k, t in kt.items():
                fams[k] = fams.get(k, 0.0) + t
            rows.append((T, routes, r, e2e, kt, cos))
            rec.row(T=T, variant=r.label, config=r.config, routes=routes, e2e_us=e2e,
                    kernels=kt, cos_ref=cos)
        torch.cuda.empty_cache()
    cols = sorted(fams, key=lambda k: -fams[k])
    table = [[T, routes, r.label, f"`{r.config}`", f"{e2e:.1f}", f"{sum(kt.values()):.1f}"]
             + [f"{kt[k]:.1f}" if k in kt else "-" for k in cols] + [core.fmt_cos(cos)]
             for T, routes, r, e2e, kt, cos in rows]
    rec.text(core.md_table(["tokens", "local routes", "variant", "config", "e2e us", "kernels"]
                           + cols + ["cos ref"], table,
                           right=tuple(i for i in range(len(cols) + 7) if i not in (2, 3))))
    rec.done()


if __name__ == "__main__":
    main()
