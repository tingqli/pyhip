# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""A/B timing of configs / variants on one GPU.

    python -m a8w4.bench ab --case ep8 -t 1024 2048 -v "ep3k m64" "pf1=m64:g2.pf=1"
    python -m a8w4.bench ab --case tp8 -t 16 -v "tp2k bm32 g1bcm2 g2nt4" \\
        "tp2k bm32 g1bcm1 g2nt4" --protocol coldrot cold     # is a gain a cache artifact?
    python -m a8w4.bench ab --case ep8 -t 4096 -v l128 "sk=py:a8w4/experiments/X/run.py:moe" \\
        --out a8w4/experiments/X --name sk

Per token count and routing sample, every variant first runs once against the fp32
reference (cos diff >= 5e-3: BAD, an exception: FAIL; both dropped) and against the first
variant, the base (cos diff > 1e-4: flagged DIFF). Then ``--rounds`` rounds time
``--iters`` calls of every (protocol, variant), the variant order reversed every other
round. The base also runs as "<base>#2": that copy's difference is the noise floor. A
difference gets a verdict (faster / slower) only beyond the noise band, max(|copy's
delta|, half the base's round spread, 0.3%); otherwise "~". Configs with cached W / A
loads are refused unless a protocol rotates >= 8 weight copies (core.PROTOCOLS).
"""

import argparse
import dataclasses
import statistics

import torch

from .. import ref as R
from . import core


def parser():
    ap = argparse.ArgumentParser(prog="python -m a8w4.bench ab", description=__doc__,
                                 epilog=core.VARIANT_HELP,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-t", "--tokens", type=int, nargs="+", required=True)
    ap.add_argument("-v", "--variants", nargs="+", default=["@table"],
                    help="variant specs (below); the first is the base (default: @table)")
    core.add_case_args(ap)
    g = ap.add_argument_group("timing")
    g.add_argument("--protocol", nargs="+", default=["coldrot"], choices=core.PROTOCOLS,
                   help="several: each variant under each; verdicts that differ are flagged")
    g.add_argument("--wcopies", type=int, default=R.WCOPIES,
                   help="weight copies coldrot / rotate cycle through (default: %(default)s)")
    g.add_argument("--rounds", type=int, default=5)
    g.add_argument("--iters", type=int, default=16, help="timed calls per round")
    g.add_argument("--samples", type=int, default=1,
                   help="routing samples per token count (seed = --seed + T + 1000 s)")
    g.add_argument("--seed", type=int, default=7)
    g.add_argument("--no-dup", action="store_true", help="do not time the base twice")
    g.add_argument("--allow-biased", action="store_true",
                   help="time cached-load configs without a rotating protocol (to measure "
                        "the bias)")
    g = ap.add_argument_group("checks")
    g.add_argument("--no-ref", action="store_true", help="skip the fp32 reference")
    g.add_argument("--bitwise", action="store_true",
                   help="report whether outputs equal the base's bit for bit (deterministic "
                        "only with one route per token: --shape H,I,E,E_global,1)")
    core.add_out_args(ap, "ab")
    return ap


def _correctness(case, res, x, tw, ti, W, raw, a):
    """One call of every variant -> (variants that passed, {label: (cos ref, cos base,
    bitwise) or why it was dropped})."""
    ref = None if a.no_ref else core.ref_out(case, x, tw, ti, raw)
    base, ok, notes = None, [], {}
    for r in res:
        try:
            out = r.run(x, W, tw, ti)
            torch.cuda.synchronize()
        except Exception as ex:  # noqa: BLE001 - report the variant, keep the others
            notes[r.label] = f"FAIL {type(ex).__name__}: {str(ex)[:200]}"
            continue
        c = R.cos_diff(ref, out.float()) if ref is not None else None
        if c is not None and not c < core.COS_TOL:
            notes[r.label] = f"BAD cos diff vs reference {c:.1e}"
            continue
        if base is None:
            if r is not res[0]:
                raise SystemExit(f"the base {res[0].label!r} failed: {notes[res[0].label]}")
            base = out.clone()
        notes[r.label] = (c, R.cos_diff(base.float(), out.float()),
                          bool(torch.equal(base, out)) if a.bitwise else None)
        ok.append(r)
    return ok, notes


def main(argv=None):
    a = parser().parse_args(argv)
    if not a.verbose:
        core.quiet()
    case = core.case_from_args(a)
    variants = [core.parse_variant(s) for s in a.variants]
    labels = [v.label for v in variants]
    if len(set(labels)) < len(labels):
        raise SystemExit(f"duplicate variant labels {labels}: name them (label=spec)")
    base_label = labels[0]
    dup_label = None
    if not a.no_dup:
        dup_label = base_label + "#2"
        variants.insert(1, dataclasses.replace(variants[0], label=dup_label))
    plan = {T: [core.resolve(v, case, T) for v in variants] for T in a.tokens}
    for T, res in plan.items():  # before any timing: fail fast
        if isinstance(res[0], str):
            raise SystemExit(f"the base at {T} tokens: {res[0]}")
        core.check_protocols([r for r in res if not isinstance(r, str)], a.protocol,
                             a.wcopies, a.allow_biased)
    H, I, E, EGl, K = case.shape
    rec = core.Recorder(a.out, a.name, core.run_meta("ab", a))
    rot = [p for p in a.protocol if p in ("coldrot", "rotate")]
    rec.text(f"{case.describe()}\n\nprotocol {', '.join(a.protocol)}"
             + (f" ({a.wcopies} weight copies)" if rot else "")
             + f", {a.rounds} rounds x {a.iters} calls; us = median of the round medians; "
             f"verdict only beyond the noise band (max(|{dup_label or 'copy'} delta|, half the "
             f"base's round spread, {core.MIN_NOISE}%))\n")
    W, raw = R.make_weights(E, H, I)
    timer = core.Timer(R.weight_copies(W, a.wcopies) if rot else [W])
    agg = {}  # (protocol, label, T) -> [(median, noise)] per sample
    cfgs = {}  # label -> {config names}
    for T in a.tokens:
        for s in range(a.samples):
            seed = a.seed + T + 1000 * s
            x, tw, ti = case.inputs(T, seed, a.local_experts_per_token)
            res = [r for r in plan[T] if not isinstance(r, str)]
            ok, notes = _correctness(case, res, x, tw, ti, W, raw, a)
            fns = {r.label: (lambda Wc, r=r: r.run(x, Wc, tw, ti)) for r in ok}
            times = core.interleaved(fns, a.protocol, timer, a.rounds, a.iters)
            routes, blocks = core.route_blocks(ti, case, [r.bm for r in ok])
            head = (f"#### {T} tokens" + (f", sample {s}" if a.samples > 1 else "")
                    + f" (seed {seed}): {routes} local routes"
                    + "".join(f", {n} blocks of {bm}" for bm, n in blocks.items()))
            rows, verdicts = [], {}
            for p in a.protocol:
                b = times[p, base_label]
                bmed = statistics.median(b)
                noise = core.noise_band(b, times.get((p, dup_label)))
                for r in ok:
                    rr = times[p, r.label]
                    m = statistics.median(rr)
                    d = (m / bmed - 1) * 100
                    v = ("base" if r.label == base_label else
                         "noise floor" if r.label == dup_label else core.verdict(d, noise))
                    verdicts[p, r.label] = (d, v)
                    c, cb, bw = notes[r.label]
                    rows.append([r.label, f"`{r.config}`", p, f"{m:.1f}", f"{d:+.1f}",
                                 f"{min(rr):.1f}-{max(rr):.1f}", v, core.fmt_cos(c),
                                 core.fmt_cos(cb) + (" DIFF" if cb > core.DIFF_TOL else "")]
                                + (["yes" if bw else "no"] if a.bitwise else []))
                    agg.setdefault((p, r.label, T), []).append((m, noise))
                    cfgs.setdefault(r.label, set()).add(r.config)
                    rec.row(T=T, sample=s, seed=seed, protocol=p, variant=r.label, spec=r.spec,
                            config=r.config, us=m, rounds=rr, delta_pct=d, noise_pct=noise,
                            verdict=v, cos_ref=c, cos_base=cb, bitwise=bw, cached=r.cached,
                            routes=routes, blocks=blocks, wcopies=a.wcopies if p in rot else 1)
                rows.append([f"*noise band +-{noise:.1f}%*"] + [""] * (len(rows[-1]) - 1))
            table = core.md_table(
                ["variant", "config", "protocol", "us", "delta %", "rounds min-max", "verdict",
                 "cos ref", "cos base"] + (["bitwise"] if a.bitwise else []), rows,
                right=(3, 4, 5, 7, 8))
            drop = [f"- {v.label}: {r}" for v, r in zip(variants, plan[T]) if isinstance(r, str)]
            drop += [f"- {n}: {why}" for n, why in notes.items() if isinstance(why, str)]
            rec.text(head + "\n\n" + table + ("\n\ndropped:\n" + "\n".join(drop) if drop else ""))
            if len(a.protocol) > 1:
                sens = []
                for r in ok:
                    vs = {p: verdicts[p, r.label] for p in a.protocol}
                    if r.label not in (base_label, dup_label) and len({v for _, v in vs.values()}) > 1:
                        sens.append(f"- {r.label}: " + ", ".join(
                            f"{p} {d:+.1f}% ({v})" for p, (d, v) in vs.items())
                            + " -> PROTOCOL-SENSITIVE: trust coldrot")
                rec.text("\nprotocols agree on every verdict" if not sens else
                         "\n" + "\n".join(sens))
            rec.text("")
            torch.cuda.empty_cache()
    if len(a.tokens) * a.samples > 1:
        order = labels[:1] + ([dup_label] if dup_label else []) + labels[1:]
        for p in a.protocol:
            rows = []
            for lab in order:
                if lab not in cfgs:
                    continue
                cells = []
                for T in a.tokens:
                    pts, bpts = agg.get((p, lab, T)), agg.get((p, base_label, T))
                    if not pts:
                        cells.append("-")
                        continue
                    us = statistics.fmean(m for m, _ in pts)
                    if lab == base_label:
                        cells.append(f"{us:.1f} us")
                        continue
                    d = (us / statistics.fmean(m for m, _ in bpts) - 1) * 100
                    noise = max(n for _, n in bpts)
                    cells.append(f"{d:+.1f}%" + ("~" if abs(d) <= noise else ""))
                rows.append([lab, " / ".join(f"`{c}`" for c in sorted(cfgs[lab]))] + cells)
            rec.text(f"#### summary ({p}): base in us, the others delta vs the base; "
                     "~ = within the noise band\n\n"
                     + core.md_table(["variant", "config"] + [str(T) for T in a.tokens], rows,
                                     right=tuple(range(2, 2 + len(a.tokens)))) + "\n")
    rec.done()


if __name__ == "__main__":
    main()
