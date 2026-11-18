# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Tuner of ``a8w4_moe``: ``python -m a8w4.tune <cmd> ...`` is ``python -m pyhip.kit <cmd>
a8w4.tune:TUNING ...`` (``pyhip.kit.tune``: every grid config of every kernel family, per
token count, ALL results stored; the dispatch picks the fastest one allowed in its
context), plus the a8w4 reports on the table (``tables.py``).

    python -m a8w4.tune problems                                # tp8-silu, tp8-situv2, ...
    python -m a8w4.tune candidates tp8-silu
    # one process per GPU, disjoint token lists, then merge:
    HIP_VISIBLE_DEVICES=0 python -m a8w4.tune tune tp8-silu -t 1 2 4 --out /tmp/p0.csv
    python -m a8w4.tune merge /tmp/p*.csv                       # into configs/a8w4_moe_tuned.csv
    python -m a8w4.tune show tp8-silu
    python -m a8w4.tune verify ep8-silu -t 1 100 3000
    # another shape / routing: --set model_dim=... inter_dim=... routing=hot
    # the shipped table's problems on 8 idle GPUs (4 per problem), BAD report, merged:
    python -m a8w4.tune tune-all --gpus 0-7 --out /tmp/a8w4_tune
    python -m a8w4.tune compare configs/a8w4_moe_tuned.csv /tmp/a8w4_tune/a8w4_moe_tuned.new.csv
    python -m a8w4.tune readme --check                          # README 7.2-7.4 vs the table
    python -m a8w4.tune stats                                   # situv2 vs silu rows

Run from the directory that contains ``a8w4/`` (or put it on PYTHONPATH). Everything
about tuning is ``TUNING`` below: the problems (``ref.moe_test``'s shapes x activations),
the token counts, the table's keys and the rotated weights. SiTUv2 runs with
``ref.SITU`` (its betas / clamp are launch arguments: no effect on the time). See
configs/README.md for the commands that produced the shipped table.
"""

import argparse
import os
import sys

if __package__ in (None, ""):  # run as a script: make the package importable
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import a8w4  # noqa: E402,F401

    __package__ = "a8w4"

from pyhip.kit import tune as kit_tune  # noqa: E402

from ._ops import DEFAULT_CSV  # noqa: E402
from .moe import MOE  # noqa: E402

# the token counts tuned: the points of the bucket dim num_tokens
TOKENS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 320, 384, 448, 512,
          640, 768, 1024, 1280, 1536, 2048, 2560, 3072, 4096, 6144, 8192, 12288, 16384, 24576,
          32768)
# how the shipped table is tuned: the problems are ref.SHAPES x activations (moe_test's
# @dim shape / act), num_tokens at TOKENS; the table is keyed by the shape, routing and
# act (not expert_offset: it only renames the local experts); the timed calls rotate
# copies of the weights W
TUNING = kit_tune.Tuning(
    MOE, problems=MOE.problems("shape", "act"), bucket="num_tokens", points=TOKENS,
    keys=("model_dim", "inter_dim", "experts", "experts_global", "topk", "routing", "act"),
    rotate=("W",))
# the table's columns: the key dims, the bucket num_tokens, the config name and its time,
# moe_check's metric
FIELDS = TUNING.fields(["cos"])
# the problems of the shipped table (configs/README.md): tune-all's default
JOBS = ("tp8-situv2", "ep8-situv2", "tp8-silu", "ep8-silu", "tp4-silu", "tp4-situv2")
REPORTS = ("compare", "readme", "stats")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return kit_tune.parser("python -m a8w4.tune").print_help()
    if argv[0] in REPORTS:
        from . import tables

        p = argparse.ArgumentParser(prog=f"python -m a8w4.tune {argv[0]}")
        p.add_argument("parts", nargs="*", help="compare: OLD NEW tables")
        p.add_argument("--csv", default=DEFAULT_CSV)
        p.add_argument("--write", action="store_true", help="readme: update README.md")
        p.add_argument("--check", action="store_true",
                       help="readme: exit 1 when README 7.2-7.4 differ from the table")
        a = p.parse_args(argv[1:])
        if argv[0] == "compare":
            if len(a.parts) != 2:
                raise SystemExit("compare OLD.csv NEW.csv")
            return tables.compare(*a.parts)
        if argv[0] == "readme":
            return tables.readme(a.csv, write=a.write, check=a.check)
        return tables.stats(a.csv)
    cmd, rest = argv[0], argv[1:]
    if cmd == "tune-all" and not kit_tune.parser().parse_args([cmd, "m:op", *rest]).args:
        rest = [*JOBS, *rest]  # default: the shipped table's problems
    return kit_tune.main([cmd, f"{__package__}.tune:TUNING", *rest], prog="python -m a8w4.tune")


if __name__ == "__main__":
    sys.exit(main())
