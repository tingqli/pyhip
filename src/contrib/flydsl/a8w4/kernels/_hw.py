# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""MI300X (gfx942) facts the kernels rely on, and helpers tied to the wave / XCD structure.

Grids sized from ``NUM_CU`` (persistent kernels, grid barriers) need the whole GPU: with
fewer free CUs they deadlock instead of slowing down.
"""

import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.expr import rocdl
from flydsl.expr.typing import T

from ._mem import _raw

NUM_CU = 304
NUM_XCD = 8  # the hardware deals WGs round robin to the XCDs: WG i runs on XCD i % 8
LDS_BYTES_MAX = 65536  # per WG; a kernel over it fails at launch without raising

# cache policy bits of buffer / global loads and stores (the aux operand)
CM_SC0, CM_NT, CM_SC1 = 1, 2, 16
CM_SYS = CM_SC0 | CM_SC1  # system scope: loads bypass, stores write through the XCD L2s

DPP_QUAD_SWAP = 0xB1  # quad_perm [1, 0, 3, 2]: lane L <- lane L ^ 1


def _u(v):
    """Wave-uniform copy (readfirstlane) of an i32."""
    return fx.Int32(rocdl.readfirstlane(T.i32, _raw(fx.Int32(v))))


def _opaque_tx(tx):
    """(tx, lane, wave) re-derived from an asm copy of tx right before a tile body:
    LLVM cannot hoist the body's per-lane address math (out of the tile loop / to
    the kernel start), whose long live ranges spilled up to ~80 VGPRs (bf16); with
    it no config spills and the fp8 kernels need ~15 VGPRs less."""
    t = fx.Int32(llvm.inline_asm(T.i32, [_raw(tx)], "v_mov_b32 $0, $1", "=v,v",
                                 has_side_effects=True))
    return t, t % fx.Int32(64), fx.Int32(rocdl.readfirstlane(T.i32, _raw(t // fx.Int32(64))))
