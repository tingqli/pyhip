# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Layouts of the buffers one kernel writes and another kernel reads.

Writer and reader import the same names from here.
"""

# ---- EP: tile schedule written by ep_prep, read by ep_gemm ----------------------------
# (i32, right after the cap // BM block expert ids): [0, 9): XCD bucket starts (m-block
# slots); [SCHED_BORDER, + nblk): the m-block order.
SCHED_BORDER = 16

# ---- megamoe: workspace / sched of ep_mega_comm and ep_mega (layout: ep_mega_comm) ----
NPES, E_LOCAL = 8, 32
E_GLOBAL = NPES * E_LOCAL
CFLAG = NPES * E_GLOBAL
FFLAG = CFLAG + NPES
ARRIVED = FFLAG + NPES
QUEUES = 8
QSTRIDE = 16  # i32 between two ticket counters (separate cache lines)
SCHED_EXPECTED = QUEUES * QSTRIDE
ROW_BITS = 18
BLOCK_M = 128
