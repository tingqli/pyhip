# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""python -m a8w4.bench <tool> ...

    ab         A/B timing of configs / variants on one GPU (noise floor, protocol guard)
    breakdown  per-kernel times (torch profiler) + e2e of variants
    check      correctness: reference, base, run-to-run stability, CUDA-graph replay
    dist       megamoe (EP8, 8 GPUs) variants: producers / lag / W cache / comm modes

``python -m a8w4.bench <tool> -h`` for the options. Run from the directory that holds
``a8w4/`` (pyhip: src/contrib/flydsl).
"""

import importlib
import sys

TOOLS = ("ab", "breakdown", "check", "dist")


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in TOOLS:
        print(__doc__)
        sys.exit(0 if sys.argv[1:2] in ([], ["-h"], ["--help"]) else 2)
    importlib.import_module(f".{sys.argv[1]}", __package__).main(sys.argv[2:])


if __name__ == "__main__":
    main()
