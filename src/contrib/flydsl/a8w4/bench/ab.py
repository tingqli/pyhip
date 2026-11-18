# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""A/B timing of configs / variants on one GPU (``pyhip.kit.bench.ab`` on an a8w4 case).

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
loads are refused unless a protocol rotates >= 8 weight copies (``kit.timing``).
"""

import argparse

from pyhip.kit import bench as kb

from .. import ref as R
from ..tune import TUNING
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


def main(argv=None):
    a = parser().parse_args(argv)
    if not a.verbose:
        core.quiet()
    case = core.case_from_args(a)
    rec = core.Recorder(a.out, a.name, core.run_meta("ab", a))
    kb.ab(TUNING, case.dims(), a.tokens, a.variants, protocols=a.protocol, rounds=a.rounds,
          iters=a.iters, samples=a.samples, seed=a.seed, dup=not a.no_dup,
          allow_biased=a.allow_biased, ref=not a.no_ref, bitwise=a.bitwise,
          diff_tol=core.DIFF_TOL, ncopies=a.wcopies, rec=rec,
          inputs=lambda T, seed: case.interface(T, seed, a.local_experts_per_token),
          annotate=core.annotate(case), alias=core.alias, unit="tokens",
          describe=case.describe())


if __name__ == "__main__":
    main()
