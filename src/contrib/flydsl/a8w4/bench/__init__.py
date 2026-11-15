# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Benchmark / experiment tools of the a8w4 package (not needed at inference time).

    python -m a8w4.bench ab         -t 1024 4096 -v "ep3k m64" "m64:g2.pf=1"   # A/B timing
    python -m a8w4.bench breakdown  -t 1024 32768 --case tp8                   # per kernel
    python -m a8w4.bench check      -t 1 77 4096 -v @table "tp1k bcm2 ks1"      # correctness
    python -m a8w4.bench dist       -t 4096 32768 -v default lag=4 bcm=0/1      # megamoe, 8 GPUs

Run from the directory that holds ``a8w4/`` (pyhip: ``src/contrib/flydsl``). Every tool
times with the deployment-like protocol by default (L2 flush + a different weight copy
per call, see ``core.PROTOCOLS``) and writes ``--out DIR``: ``<name>.jsonl`` (one record
per measurement, run metadata first) and ``<name>.md`` (the printed tables).
"""
