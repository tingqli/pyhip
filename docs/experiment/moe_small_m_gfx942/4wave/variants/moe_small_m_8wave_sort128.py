# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
# Historical variant, kept only as the A/B reference of docs/experiment/moe_small_m_gfx942/4wave (loaded from
# this file by tools/ab_kernels.py and tools/make_bench_ab.py; not part of the pyhip package).

"""gfx942 FP8 block-scale MoE GateUp and Down for experts with at most 128 routes (128-row sort blocks).

With <= 128 routes per expert the BM256 kernels (moe_gateup/down_a8w8_blockscale_gfx942_lds) run at least half
of their MFMAs on padding rows, and the call is bound by the weights: every active expert's weights are read
once whatever its route count. These kernels target that regime (docs/record/moe_small_m_gfx942.md):

- One workgroup (8 waves) takes one sorted 128-row block and one N tile; every wave owns 2 x 16 weight rows
  (GateUp: 16 gate rows and the matching 16 up rows, 128 output columns per task; Down: 32 output columns,
  256 per task).
- B is aiter shuffle_weight(16,16): a 16 B load per lane is MFMA-ready, so each wave streams its weight rows
  straight into VGPRs through a ring of `ring` K blocks (ring - 1 in flight); no LDS and no barrier on the B
  path. Loads past the last K block point out of the descriptor and move no data. ring=2 is the default: under
  the 750 W cap (sustained calls) it beats 4 and 8 by 0-4% at every slice count.
- A (GateUp: rows gathered by token; Down: the sorted FP8 rows GateUp wrote) is loaded two K blocks ahead into
  VGPRs and written into a double-buffered 16 KiB LDS tile, one barrier per K block.
- The rows a block uses are found at run time (ballot of its sorted IDs). The K loop and epilogue exist for 1 to
  8 sixteen-row slices and the block takes the smallest that covers its routes, so MFMAs, A traffic and stores
  follow the routes (to the next 16 rows) instead of the 128-row block.
- Block scales as in the BM256 kernels: per K block the 4 chained MFMAs of a (slice, side) give a K128 partial,
  retired as acc += partial * (scale_a[row] * scale_b) under the next slice's MFMAs.

GateUp writes the Down input: SiLU(gate) * up quantized to FP8 per (row, 128 columns) with FP32 scale amax / 240,
at sorted row block * 128 + row of [blocks * 128, I] (scales [I/128, blocks * 128]). Rows past the used slices
are not written: Down never reads padding rows. Down adds routing weight * (row @ W2^T) into out [tokens, H] with
BF16 atomics, or stores the rows of tokens with one local route (single_store).
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from .gemm_a8w8_blockscale_gfx942_common import pack_bf16x2_rne
from .helpers import BufferTensor, run_compiled
from .moe_gateup_a8w8_blockscale_gfx942_lds import FP8_MAX, LOG2E, OOB_OFFSET, TILE_K, TOKEN_MASK, _lds_dwords

SORT_BLOCK_M = 128
GATEUP_TILE_N = 128
DOWN_TILE_N = 256
ROW_DWORDS = TILE_K // 4
A_TILE_DWORDS = SORT_BLOCK_M * ROW_DWORDS
B_BLOCK_BYTES = 16 * TILE_K
# Down epilogue transpose: per wave 16 rows x 16 dwords; an even pitch keeps the 8 B writes aligned.
TRANSPOSE_PITCH = 18
SLICE_VARIANTS = (1, 2, 3, 4, 5, 6, 7, 8)


def validate_parameters(mode, tokens, n_dim, k_dim, topk, num_experts, capacity):
    """mode "gateup": n_dim = inter_dim, k_dim = hidden_dim; "down": n_dim = hidden_dim, k_dim = inter_dim."""
    tile_n = GATEUP_TILE_N if mode == "gateup" else DOWN_TILE_N
    if not 0 < tokens <= TOKEN_MASK:
        raise ValueError(f"tokens must be in [1, {TOKEN_MASK}]")
    if n_dim <= 0 or n_dim % tile_n:
        raise ValueError(f"{mode}: N must be a positive multiple of {tile_n}")
    k_blocks = k_dim // TILE_K
    if k_dim <= 0 or k_dim % TILE_K or (2 * k_blocks if mode == "gateup" else k_blocks) > 64:
        raise ValueError(f"{mode}: K must be a multiple of 128 and the task's B scales must fit one VGPR")
    if not 0 < topk < 256 or num_experts <= 0:
        raise ValueError("require 0 < topk < 256 and num_experts > 0")
    weight_rows = 2 * n_dim if mode == "gateup" else n_dim
    extents = {"weight": weight_rows * k_dim}
    if mode == "gateup":
        extents.update(a=tokens * k_dim, a2=capacity * n_dim)
    else:
        extents.update(a2=capacity * k_dim, output=tokens * n_dim * 2)
    # Offsets past a block still add up to 64 KiB of K-block soffset.
    over = [f"{name}={size:,}" for name, size in extents.items() if size >= OOB_OFFSET - (1 << 20)]
    if over:
        raise ValueError("buffer byte extents too large: " + ", ".join(over))


def _rsrc_of(ptr, num_records_bytes):
    tensor = fx.rocdl.make_buffer_tensor(fx.make_view(ptr, fx.make_layout(1, 1)), max_size=False, num_records_bytes=num_records_bytes)
    return fx.rocdl.get_buffer_rsrc(fx.get_iter(tensor))


@functools.cache
def compile_moe_small_m(mode, n_dim, k_dim, topk, num_experts, ring=2, b_aux=2, single_store=False, slice_variants=SLICE_VARIANTS):
    """Launcher of the GateUp ("gateup") or Down ("down") kernel; see the module doc.

    ring: K blocks of B per wave ring (ring - 1 in flight); b_aux: cache policy bits of the B loads (2 = nt).
    """
    assert mode in ("gateup", "down"), mode
    gateup = mode == "gateup"
    validate_parameters(mode, 1, n_dim, k_dim, topk, num_experts, SORT_BLOCK_M)
    KB = k_dim // TILE_K
    assert ring % 2 == 0 and KB % ring == 0, "the K loop is unrolled by ring, which must divide K / 128"
    variants = sorted(set(slice_variants))
    assert variants[-1] == 8 and variants[0] >= 1, variants
    if gateup:
        I = n_dim
        n_tiles = I // GATEUP_TILE_N
        weight_rows = 2 * I
    else:
        H = n_dim
        n_tiles = H // DOWN_TILE_N
        weight_rows = H
    scale_groups = weight_rows // 128

    @fx.struct
    class SharedTiles:
        # A double buffer. After the K loop: GateUp's FP8 output tile (buffer 0) and row maxima (buffer 1), or
        # Down's per-wave transposes.
        a: fx.Array[fx.Int32, 2 * A_TILE_DWORDS, 16]
        # The A scales of the two buffered K blocks, one per row.
        sa: fx.Array[fx.Float32, 2 * SORT_BLOCK_M, 16]

    @flyc.kernel(known_block_size=[512, 1, 1])
    def small_m_kernel(
        pA: fx.Pointer,
        pB: fx.Pointer,
        pOut: fx.Pointer,
        pScaleOut: fx.Pointer,
        pScaleA: fx.Pointer,
        pScaleB: fx.Pointer,
        pSortedIds: fx.Pointer,
        pSortedWeights: fx.Pointer,
        pExpertIds: fx.Pointer,
        pNumValid: fx.Pointer,
        pRouteCounts: fx.Pointer,
        num_tokens: fx.Int32,
        num_blocks: fx.Int32,
    ):
        tid = fx.thread_idx.x
        lane = tid % 64
        wave = fx.Int32(rocdl.readfirstlane(T.i32, (tid // 64).ir_value()))
        pid = fx.Int32(fx.block_idx.x)
        zero = fx.Int32(0).ir_value()

        def load_i32(rsrc, voffset):
            return fx.Int32(rocdl.raw_ptr_buffer_load(T.i32, rsrc, fx.Int32(voffset).ir_value(), zero, zero))

        def uniform(value):
            return fx.Int32(rocdl.readfirstlane(T.i32, value.ir_value()))

        valid_rows = uniform(load_i32(_rsrc_of(pNumValid, 4), 0))
        valid_blocks = (valid_rows + (SORT_BLOCK_M - 1)) // SORT_BLOCK_M
        valid_blocks = (valid_blocks < num_blocks).select(valid_blocks, num_blocks)
        num_tasks = valid_blocks * n_tiles
        # Hardware deals workgroups round-robin over 8 XCDs: each XCD gets a contiguous task range, so the N tiles
        # of a block (the same A rows) share one L2.
        chunk = (num_tasks + 7) >> 3
        xcd_slot = pid >> 3
        task = (pid & 7) * chunk + xcd_slot
        active = (xcd_slot < chunk) & (task < num_tasks)

        if active:
            block = task // n_tiles
            n_tile = task - block * n_tiles
            row_base = block * SORT_BLOCK_M
            capacity = num_blocks * SORT_BLOCK_M
            sorted_rsrc = _rsrc_of(pSortedIds, num_blocks * (SORT_BLOCK_M * 4))
            expert = uniform(load_i32(_rsrc_of(pExpertIds, num_blocks * 4), block * 4))

            def decode(fused):
                """(token, slot, valid) of a sorted ID."""
                token = fused & TOKEN_MASK
                slot = (fused >> 24) & 255
                return token, slot, (token < num_tokens) & (slot < topk)

            # Rows the block uses (highest valid row + 1), identical in every wave.
            low_valid = decode(load_i32(sorted_rsrc, (row_base + lane) * 4))[2]
            high_valid = decode(load_i32(sorted_rsrc, (row_base + 64 + lane) * 4))[2]
            low_mask = fx.Int64(rocdl.ballot(fx.Int64.ir_type, low_valid.ir_value()))
            high_mask = fx.Int64(rocdl.ballot(fx.Int64.ir_type, high_valid.ir_value()))
            used_rows = (high_mask != fx.Int64(0)).select(
                fx.Int64(128) - fx.math.ctlz(high_mask), fx.Int64(64) - fx.math.ctlz(low_mask)
            )

            # The B scales of every K block of the task in one VGPR, read with v_readlane.
            if const_expr(gateup):
                scale_side = lane // KB
                scale_offset = ((expert * scale_groups + n_tile + scale_side * (I // 128)) * KB + lane - scale_side * KB) * 4
                scale_valid = lane < 2 * KB
            else:
                scale_offset = ((expert * scale_groups + n_tile * 2 + wave // 4) * KB + lane) * 4
                scale_valid = lane < KB
            sb_lanes = fx.Float32(
                rocdl.raw_ptr_buffer_load(
                    T.f32,
                    _rsrc_of(pScaleB, num_experts * scale_groups * (KB * 4)),
                    scale_valid.select(scale_offset, fx.Int32(OOB_OFFSET)).ir_value(),
                    zero,
                    zero,
                )
            )

            def scale_b(side, k_block):
                index = k_block + side * KB if const_expr(gateup) else k_block
                return fx.Float32(rocdl.readlane(T.f32, sb_lanes.ir_value(), fx.Int32(index).ir_value()))

            b_tensor = fx.rocdl.make_buffer_tensor(
                fx.make_view(pB + fx.Int64(expert) * (weight_rows * k_dim // 4), fx.make_layout(1, 1)),
                max_size=False,
                num_records_bytes=weight_rows * k_dim,
            )
            b_packet = BufferTensor(fx.make_view(fx.get_iter(b_tensor), fx.make_layout(4, 1)))
            if const_expr(gateup):
                b_first_rows = [n_tile * GATEUP_TILE_N + wave * 16, n_tile * GATEUP_TILE_N + wave * 16 + I]
            else:
                b_first_rows = [n_tile * DOWN_TILE_N + wave * 32, n_tile * DOWN_TILE_N + wave * 32 + 16]
            # Lane l, load j of a 16-row group's K block: weight row l % 16, K 64 j + 16 (l // 16) + [0, 16).
            b_voffsets = [[b_first_rows[side] * k_dim + j * 1024 + lane * 16 for j in range(2)] for side in range(2)]
            # The A reads take the same K: row lane % 16 of a slice, 16 B column 4 j + lane // 16.
            a_read = [_lds_dwords(lane % 16, j * 4 + lane // 16) for j in range(2)]

            if const_expr(gateup):
                a_rows_total = num_tokens
            else:
                a_rows_total = capacity
            a_tensor = fx.rocdl.make_buffer_tensor(
                fx.make_view(pA, fx.make_layout(1, 1)), max_size=False, num_records_bytes=a_rows_total * k_dim
            )
            a_packet = BufferTensor(fx.make_view(fx.get_iter(a_tensor), fx.make_layout(4, 1)))
            scale_a_rsrc = _rsrc_of(pScaleA, a_rows_total * (KB * 4))
            lds = fx.SharedAllocator().allocate(SharedTiles).peek()

            def lds_view(dwords, count=4):
                return fx.make_view(lds.a.ptr + dwords, fx.make_layout(count, 1))

            def barrier():
                rocdl.sched_barrier(0)
                rocdl.s_barrier()
                rocdl.sched_barrier(0)

            def silu_mul(gate_values, up_values):
                result = []
                for element in range_constexpr(4):
                    gate = gate_values[element]
                    exponent = fx.Float32(rocdl.exp2(T.f32, (gate * -LOG2E).ir_value()))
                    inverse = fx.Float32(rocdl.rcp(T.f32, (exponent + 1.0).ir_value()))
                    result.append(gate * inverse * up_values[element])
                return result

            def run(S):
                """K loop and epilogue for S 16-row slices (S * 16 >= the rows this block uses)."""
                copies = 1 if S <= 4 else 2
                lane_rows = [s * 16 + lane % 16 for s in range_constexpr(S)]
                lane_ids = [load_i32(sorted_rsrc, (row_base + row) * 4) for row in lane_rows]
                copy_rows = [tid // 8 + 64 * c for c in range_constexpr(copies)]
                a_offsets = []
                for c in range_constexpr(copies):
                    token, _, valid = decode(load_i32(sorted_rsrc, (row_base + copy_rows[c]) * 4))
                    a_row = token if const_expr(gateup) else row_base + copy_rows[c]
                    a_offsets.append(valid.select(a_row * k_dim + tid % 8 * 16, fx.Int32(OOB_OFFSET)))
                # Thread t < 128 stages the A scale of block row t through LDS.
                scale_row = tid % SORT_BLOCK_M
                token, _, valid = decode(load_i32(sorted_rsrc, (row_base + scale_row) * 4))
                sa_row = token if const_expr(gateup) else row_base + scale_row
                sa_offset = valid.select(sa_row * 4, fx.Int32(OOB_OFFSET))
                a_write = [_lds_dwords(copy_rows[c], tid % 8) for c in range_constexpr(copies)]

                # B ring slot buf * 4 + side * 2 + j; the A rows and scales of K block k stage in slot k % 2.
                b_ring = fx.make_rmem_tensor(fx.make_layout((4, ring * 4), (1, 4)), fx.Int32)
                a_stage = fx.make_rmem_tensor(fx.make_layout((4, 2 * copies), (1, 4)), fx.Int32)
                sa_stage = fx.make_rmem_tensor(fx.make_layout((1, 2), (1, 1)), fx.Float32)
                accum = fx.make_rmem_tensor(fx.make_layout((4, 2 * S), (1, 4)), fx.Float32)
                partial = fx.make_rmem_tensor(fx.make_layout((4, 4), (1, 4)), fx.Float32)
                partial_scale = fx.make_rmem_tensor(fx.make_layout((1, 4), (1, 1)), fx.Float32)

                def block_soffset(k_block, stride):
                    # Past the last K block the load points out of every descriptor and moves no data.
                    return (k_block < KB).select(k_block * stride, fx.Int32(OOB_OFFSET))

                def issue_b(k_block, buf, parts=((0, 0), (0, 1), (1, 0), (1, 1))):
                    soffset = block_soffset(k_block, B_BLOCK_BYTES)
                    for side, j in parts:
                        b_ring[None, buf * 4 + side * 2 + j].store(
                            b_packet.load(voffset_bytes=b_voffsets[side][j], soffset_bytes=soffset, aux=b_aux)
                        )

                def issue_a(k_block, kbuf, parts=None):
                    soffset = block_soffset(k_block, TILE_K)
                    selected = list(range(copies)) if parts is None else list(parts)
                    for c in selected:
                        a_stage[None, kbuf * copies + c].store(a_packet.load(voffset_bytes=a_offsets[c], soffset_bytes=soffset))

                def issue_scale_a(k_block, kbuf):
                    soffset = block_soffset(k_block, a_rows_total * 4)
                    value = rocdl.raw_ptr_buffer_load(T.f32, scale_a_rsrc, fx.Int32(sa_offset).ir_value(), fx.Int32(soffset).ir_value(), zero)
                    sa_stage[None, kbuf].store(Vec.from_elements([fx.Float32(value)], fx.Float32))

                def write_a(kbuf):
                    for c in range_constexpr(copies):
                        lds_view(a_write[c] + kbuf * A_TILE_DWORDS).store(a_stage[None, kbuf * copies + c].load())
                    # Threads t, t + 128, ... store the same scale: no branch inside the K loop.
                    fx.make_view(lds.sa.ptr + (kbuf * SORT_BLOCK_M + scale_row), fx.make_layout(1, 1)).store(sa_stage[None, kbuf].load())

                def retire(slice_index, pbuf, element_index):
                    """acc += partial * scale for element (side, element) of a finished slice's partials."""
                    side, element = divmod(element_index, 4)
                    target = accum[None, slice_index * 2 + side]
                    target[element] = fx.fma(
                        Vec(partial[None, pbuf * 2 + side].load())[element],
                        Vec(partial_scale[None, pbuf * 2 + side].load())[0],
                        target[element],
                    )

                def read_a(s, kbuf):
                    """A words of slice s (both 64-K halves) and the A scale of this lane's row."""
                    words = [
                        Vec(lds_view(a_read[j] + (s * 16 * ROW_DWORDS + kbuf * A_TILE_DWORDS)).load()).bitcast(fx.Int64)
                        for j in range_constexpr(2)
                    ]
                    scale = Vec(fx.make_view(lds.sa.ptr + (kbuf * SORT_BLOCK_M + s * 16 + lane % 16), fx.make_layout(1, 1)).load())[0]
                    return words, scale

                def compute(k_block, buf, kbuf, k_local, memory_ops):
                    """MFMAs of K block k_block; memory_ops (the next blocks' loads, in issue order) go between slices."""
                    sb = [scale_b(side, k_block) for side in range_constexpr(2)]
                    b_words = [
                        [Vec(b_ring[None, buf * 4 + side * 2 + j].load()).bitcast(fx.Int64) for j in range_constexpr(2)]
                        for side in range_constexpr(2)
                    ]
                    # A of slice s + 1 is read before the MFMAs of slice s, which hide its LDS latency.
                    a_next = read_a(0, kbuf)
                    for s in range_constexpr(S):
                        for op in memory_ops[-(-s * len(memory_ops) // S) : -(-(s + 1) * len(memory_ops) // S)]:
                            op()
                        pbuf = (k_local * S + s) % 2
                        # Retired under this slice: the slice before (the previous K block's last for s == 0).
                        previous = s - 1 if s > 0 else S - 1
                        a_words, sa = a_next
                        if const_expr(s + 1 < S):
                            a_next = read_a(s + 1, kbuf)
                        for step in range_constexpr(4):
                            j, h = divmod(step, 2)
                            for side in range_constexpr(2):
                                accumulator = Vec.filled(4, 0.0, fx.Float32) if step == 0 else Vec(partial[None, pbuf * 2 + side].load())
                                partial[None, pbuf * 2 + side].store(
                                    Vec(rocdl.mfma_f32_16x16x32_fp8_fp8(T.f32x4, [b_words[side][j][h], a_words[j][h], accumulator, 0, 0, 0]))
                                )
                                # Keep MFMA -> FMA order: the retired partial has MFMAs of XDL latency cover.
                                rocdl.sched_barrier(0)
                                retire(previous, 1 - pbuf, step * 2 + side)
                                if const_expr(step == 0):
                                    partial_scale[None, pbuf * 2 + side].store(Vec.from_elements([sa * sb[side]], fx.Float32))
                                rocdl.sched_barrier(0)

                # Prologue: the A rows / scales of K blocks 0 and 1, ring - 1 B blocks, block 0 into LDS.
                issue_a(fx.Int32(0), 0)
                issue_scale_a(fx.Int32(0), 0)
                issue_a(fx.Int32(1), 1)
                issue_scale_a(fx.Int32(1), 1)
                for p in range_constexpr(ring - 1):
                    issue_b(fx.Int32(p), p)
                accum.fill(0)
                # The first slice retires these zeros as the "previous" one.
                for p in range_constexpr(4):
                    partial[None, p].store(Vec.filled(4, 0.0, fx.Float32))
                    partial_scale[None, p].store(Vec.filled(1, 0.0, fx.Float32))
                write_a(0)
                rocdl.s_waitcnt(lgkmcnt=0)
                barrier()

                for k_outer in range(0, KB // ring):
                    for k_local in range_constexpr(ring):
                        k_block = fx.Int32(k_outer) * ring + k_local
                        kbuf = k_local % 2
                        b_buf = (k_local + ring - 1) % ring
                        # A before B: the in-order vmcnt wait for A then never covers the newest B blocks.
                        memory_ops = [lambda c=c: issue_a(k_block + 2, kbuf, (c,)) for c in range(copies)]
                        memory_ops.append(lambda: issue_scale_a(k_block + 2, kbuf))
                        memory_ops += [lambda p=p: issue_b(k_block + (ring - 1), b_buf, (p,)) for p in ((0, 0), (0, 1), (1, 0), (1, 1))]
                        compute(k_block, k_local, kbuf, k_local, memory_ops)
                        write_a(1 - kbuf)
                        rocdl.s_waitcnt(lgkmcnt=0)
                        barrier()

                # The last slice of the last K block has no later MFMA shadow.
                last_pbuf = ((ring - 1) * S + S - 1) % 2
                for element_index in range_constexpr(8):
                    retire(S - 1, last_pbuf, element_index)

                if const_expr(gateup):
                    gateup_epilogue(S, accum)
                else:
                    down_epilogue(S, accum, lane_rows, lane_ids)

            def gateup_epilogue(S, accum):
                """FP8 per (row, 128 columns) of SiLU(gate) * up; every dwordx4 store writes 8 rows x 16 B... of one row."""
                row_max_base = A_TILE_DWORDS
                values = [
                    silu_mul(Vec(accum[None, s * 2].load()), Vec(accum[None, s * 2 + 1].load())) for s in range_constexpr(S)
                ]
                for s in range_constexpr(S):
                    peak = fx.Float32(0.0)
                    for value in values[s]:
                        peak = fx.arith.maxnumf(peak, fx.math.absf(value))
                    # Lanes l, l ^ 16, l ^ 32, l ^ 48 hold this wave's other columns of row l % 16.
                    for shift in (16, 32):
                        other = rocdl.ds_bpermute(T.i32, fx.Int32((lane ^ shift) * 4).ir_value(), peak.bitcast(fx.Int32).ir_value())
                        peak = fx.arith.maxnumf(peak, fx.Int32(other).bitcast(fx.Float32))
                    if lane < 16:
                        lds_view(row_max_base + (s * 16 + lane) * 8 + wave, 1).store(Vec.from_elements([peak.bitcast(fx.Int32)], fx.Int32))
                rocdl.s_waitcnt(lgkmcnt=0)
                barrier()

                def row_amax(row):
                    halves = [Vec(lds_view(row_max_base + row * 8 + half * 4).load()).bitcast(fx.Float32) for half in range_constexpr(2)]
                    peak = halves[0][0]
                    for index in range_constexpr(1, 8):
                        peak = fx.arith.maxnumf(peak, halves[index // 4][index % 4])
                    return peak

                for s in range_constexpr(S):
                    row = s * 16 + lane % 16
                    amax = row_amax(row)
                    # |x| <= amax keeps x * 240 / amax within the FP8 range after rounding.
                    inverse = (amax > 0.0).select(fx.Float32(rocdl.rcp(T.f32, amax.ir_value())) * FP8_MAX, fx.Float32(0.0))
                    q = [values[s][element] * inverse for element in range_constexpr(4)]
                    low = rocdl.cvt_pk_fp8_f32(T.i32, q[0], q[1], fx.Int32(0), False)
                    word = fx.Int32(rocdl.cvt_pk_fp8_f32(T.i32, q[2], q[3], low, True))
                    # This wave's 16 columns are 16 B column `wave` of the 128 B row; lane group g holds 4 g .. 4 g + 3.
                    lds_view(_lds_dwords(row, wave) + lane // 16, 1).store(Vec.from_elements([word], fx.Int32))
                rocdl.s_waitcnt(lgkmcnt=0)
                barrier()
                out_rsrc = _rsrc_of(pOut, capacity * I)
                for part in range_constexpr((S * 16 + 63) // 64):
                    out_row = part * 64 + tid // 8
                    offset = (row_base + out_row) * I + n_tile * GATEUP_TILE_N + tid % 8 * 16
                    if const_expr(S * 16 % 64 != 0):
                        offset = (out_row < S * 16).select(offset, fx.Int32(OOB_OFFSET))
                    rocdl.raw_ptr_buffer_store(
                        Vec(lds_view(_lds_dwords(out_row, tid % 8)).load()).ir_value(),
                        out_rsrc,
                        offset.ir_value(),
                        zero,
                        aux=ir.IntegerAttr.get(T.i32, 0),
                    )
                scale_row = tid % SORT_BLOCK_M
                scale_offset = (n_tile * capacity + row_base + scale_row) * 4
                rocdl.raw_ptr_buffer_store(
                    (row_amax(scale_row) * (1.0 / FP8_MAX)).ir_value(),
                    _rsrc_of(pScaleOut, (I // 128) * capacity * 4),
                    (tid < S * 16).select(scale_offset, fx.Int32(OOB_OFFSET)).ir_value(),
                    zero,
                    aux=ir.IntegerAttr.get(T.i32, 0),
                )

            def down_epilogue(S, accum, lane_rows, lane_ids):
                """routing weight * row in BF16, through a per-wave LDS transpose: 4 rows x 64 B per store/atomic."""
                weight_rsrc = _rsrc_of(pSortedWeights, capacity * 4)
                weights = [
                    fx.Float32(rocdl.raw_ptr_buffer_load(T.f32, weight_rsrc, ((row_base + row) * 4).ir_value(), zero, zero))
                    for row in lane_rows
                ]
                token_offsets = []
                for s in range_constexpr(S):
                    token, _, valid = decode(lane_ids[s])
                    row_offset = token * (H * 2)
                    if const_expr(single_store):
                        # Bit 0 marks the token's only local route: a plain store, no other route adds to it.
                        routes = load_i32(_rsrc_of(pRouteCounts, num_tokens * 4), valid.select(token * 4, fx.Int32(OOB_OFFSET)))
                        row_offset = row_offset | (routes == 1).select(fx.Int32(1), fx.Int32(0))
                    token_offsets.append(valid.select(row_offset, fx.Int32(-1)))
                out_base = fx.Int64(fx.ptrtoint(pOut)).ir_value()
                transpose = wave * (16 * TRANSPOSE_PITCH)
                write_base = transpose + lane % 16 * TRANSPOSE_PITCH + lane // 16 * 2
                read_base = transpose + lane // 16 * TRANSPOSE_PITCH + lane % 16
                column = n_tile * (DOWN_TILE_N * 2) + wave * 64 + lane % 16 * 4
                # LLVM does not count the asm stores: retire every load first so no later vmcnt wait covers them.
                rocdl.s_waitcnt(vmcnt=0)
                for s in range_constexpr(S):
                    for side in range_constexpr(2):
                        values = Vec(accum[None, s * 2 + side].load())
                        words = pack_bf16x2_rne([values[element] * weights[s] for element in range_constexpr(4)])
                        lds_view(write_base + side * 8, 2).store(Vec.from_elements(words, fx.Int32))
                    # Store i covers rows 4 i .. 4 i + 3 of the slice (lane // 16), 64 B each.
                    row_offsets = [
                        fx.Int32(rocdl.ds_bpermute(T.i32, ((4 * i + lane // 16) * 4).ir_value(), token_offsets[s].ir_value()))
                        for i in range_constexpr(4)
                    ]
                    reads = [Vec(lds_view(read_base + 4 * i * TRANSPOSE_PITCH, 1).load())[0].ir_value() for i in range_constexpr(4)]
                    # One opaque copy of the 4 LDS reads: one wait, and LLVM cannot sink a read into a store branch.
                    copies = llvm.inline_asm(
                        ir.Type.parse("!llvm.struct<(" + ", ".join(["i32"] * 4) + ")>"),
                        reads,
                        "",
                        ",".join(["=v"] * 4 + [str(i) for i in range(4)]),
                        has_side_effects=True,
                    )
                    for i in range_constexpr(4):
                        value = fx.Int32(llvm.extractvalue(T.i32, copies, [i])).ir_value()
                        row_offset = row_offsets[i]
                        if const_expr(single_store):
                            address = ((row_offset & -2) + column).ir_value()
                            if (row_offset >= 0) & ((row_offset & 1) == 1):
                                llvm.inline_asm(None, [address, value, out_base], "global_store_dword $0, $1, $2", "v,v,s", has_side_effects=True)
                            if (row_offset >= 0) & ((row_offset & 1) == 0):
                                llvm.inline_asm(None, [address, value, out_base], "global_atomic_pk_add_bf16 $0, $1, $2", "v,v,s", has_side_effects=True)
                        else:
                            if row_offset >= 0:
                                llvm.inline_asm(
                                    None, [(row_offset + column).ir_value(), value, out_base], "global_atomic_pk_add_bf16 $0, $1, $2", "v,v,s",
                                    has_side_effects=True,
                                )

            def dispatch(index):
                if const_expr(index == len(variants) - 1):
                    run(variants[index])
                else:
                    if used_rows <= variants[index] * 16:
                        run(variants[index])
                    else:
                        dispatch(index + 1)

            dispatch(0)

    @flyc.jit
    def launch(
        pA: fx.Pointer,
        pB: fx.Pointer,
        pOut: fx.Pointer,
        pScaleOut: fx.Pointer,
        pScaleA: fx.Pointer,
        pScaleB: fx.Pointer,
        pSortedIds: fx.Pointer,
        pSortedWeights: fx.Pointer,
        pExpertIds: fx.Pointer,
        pNumValid: fx.Pointer,
        pRouteCounts: fx.Pointer,
        num_tokens: fx.Int32,
        num_blocks: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        small_m_kernel(
            pA, pB, pOut, pScaleOut, pScaleA, pScaleB, pSortedIds, pSortedWeights, pExpertIds, pNumValid, pRouteCounts,
            num_tokens, num_blocks,
            value_attrs={"llvm.passthrough": [["target-features", "-packed-fp32-ops"]]},
        ).launch(
            grid=((num_blocks * n_tiles + 7) // 8 * 8, 1, 1),
            block=(512, 1, 1),
            stream=stream,
        )

    launch.compile_hints["target_features"] = "-packed-fp32-ops"
    return launch


def _ptr(dtype, tensor):
    return flyc.from_c_void_p(dtype, tensor.data_ptr())


def moe_gateup_small_m(
    a, w1, a_scale_t, w1_scale, sorted_ids, expert_ids, num_valid_ids, a2, a2_scale_t, topk, ring=2, b_aux=2
):
    """a [T, H] FP8, w1 shuffle_weight(16,16) of [E, 2I, H] FP8, a_scale_t [H/128, T], w1_scale [E, 2I/128, H/128] FP32;
    routes as from aiter moe_sorting(128): sorted_ids [blocks * 128], expert_ids [blocks], num_valid_ids [>= 1].
    Writes a2 [blocks * 128, I] FP8 and a2_scale_t [I/128, blocks * 128] FP32 (see the module doc)."""
    tokens, hidden_dim = a.shape
    num_experts, two_inter, _ = w1.shape
    inter_dim = two_inter // 2
    num_blocks = expert_ids.numel()
    capacity = num_blocks * SORT_BLOCK_M
    assert a2.shape == (capacity, inter_dim) and a2.element_size() == 1
    assert a2_scale_t.shape == (inter_dim // 128, capacity) and a2_scale_t.dtype == torch.float32
    assert sorted_ids.numel() >= capacity
    validate_parameters("gateup", tokens, inter_dim, hidden_dim, topk, num_experts, capacity)
    launch = compile_moe_small_m("gateup", inter_dim, hidden_dim, topk, num_experts, ring, b_aux)
    run_compiled(
        launch,
        _ptr(fx.Int32, a), _ptr(fx.Int32, w1), _ptr(fx.Int32, a2), _ptr(fx.Float32, a2_scale_t), _ptr(fx.Float32, a_scale_t),
        _ptr(fx.Float32, w1_scale), _ptr(fx.Int32, sorted_ids), _ptr(fx.Float32, a2_scale_t), _ptr(fx.Int32, expert_ids),
        _ptr(fx.Int32, num_valid_ids), _ptr(fx.Int32, num_valid_ids), tokens, num_blocks, torch.cuda.current_stream(),
    )
    return a2


def moe_down_small_m(
    a2, w2, a2_scale_t, w2_scale, sorted_ids, sorted_weights, expert_ids, num_valid_ids, out, topk, route_counts=None, ring=2,
    b_aux=2,
):
    """out [T, H] BF16 += routing weight * (a2 row @ w2[expert]^T) for every route; a2 / a2_scale_t from
    moe_gateup_small_m, w2 shuffle_weight(16,16) of [E, H, I] FP8, w2_scale [E, H/128, I/128], sorted_weights
    [blocks * 128]. route_counts (moe_route_counts): rows of single-route tokens are stored instead of added; out
    must hold zeros at every other routed row."""
    tokens, hidden_dim = out.shape
    num_experts, _, inter_dim = w2.shape
    num_blocks = expert_ids.numel()
    capacity = num_blocks * SORT_BLOCK_M
    assert a2.shape[0] >= capacity and a2.shape[1] == inter_dim
    assert sorted_ids.numel() >= capacity and sorted_weights.numel() >= capacity
    single_store = route_counts is not None
    assert not single_store or route_counts.shape == (tokens,) and route_counts.dtype == torch.int32
    validate_parameters("down", tokens, hidden_dim, inter_dim, topk, num_experts, capacity)
    launch = compile_moe_small_m("down", hidden_dim, inter_dim, topk, num_experts, ring, b_aux, single_store)
    run_compiled(
        launch,
        _ptr(fx.Int32, a2), _ptr(fx.Int32, w2), _ptr(fx.Int32, out), _ptr(fx.Float32, a2_scale_t), _ptr(fx.Float32, a2_scale_t),
        _ptr(fx.Float32, w2_scale), _ptr(fx.Int32, sorted_ids), _ptr(fx.Float32, sorted_weights), _ptr(fx.Int32, expert_ids),
        _ptr(fx.Int32, num_valid_ids), _ptr(fx.Int32, route_counts if single_store else num_valid_ids), tokens, num_blocks,
        torch.cuda.current_stream(),
    )
    return out


def max_sorted_blocks(tokens, topk, local_experts):
    """Upper bound on the 128-row blocks the sort emits: at most one partial block per local expert."""
    return -(-tokens * min(topk, local_experts) // SORT_BLOCK_M) + local_experts


def fused_moe_small_m(
    a1, a1_scale_t, w1, w1_scale, w2, w2_scale, sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, out, topk,
    num_rows=None, single_store=True, ring=2, b_aux=2, out_zeroed=False,
):
    """fused_moe_a8w8_blockscale_gfx942 on routes from aiter moe_sorting(block_size=128):
    out [tokens, H] BF16 = sum over each token's local routes of routing weight * Down(SiLU(gate) * up).

    out_zeroed: the sort was handed out as moe_buf and zeroed it; Down adds every route with BF16 atomics and no
    other kernel runs (the fastest form, used by fused_moe_a8w8_blockscale_gfx942_sort). Otherwise (empty moe_buf):
    single_store stores the rows of single-route tokens after a route-count kernel and zeroes the other rows, and
    single_store=False zeroes out first. num_rows: optional device int32 tensor with the valid token count; without
    out_zeroed, rows of out at or past it stay untouched.
    """
    from .moe_down_a8w8_blockscale_gfx942_lds import moe_route_counts, moe_zero_rows

    tokens = a1.shape[0]
    inter_dim = w2.shape[2]
    blocks = min(sorted_expert_ids.numel(), max_sorted_blocks(tokens, topk, w1.shape[0]))
    expert_ids = sorted_expert_ids[:blocks]
    capacity = blocks * SORT_BLOCK_M
    a2 = torch.empty((capacity, inter_dim), device=a1.device, dtype=a1.dtype)
    a2_scale_t = torch.empty((inter_dim // 128, capacity), device=a1.device, dtype=torch.float32)
    route_counts = None
    if out_zeroed:
        pass
    elif single_store:
        route_counts = torch.empty(tokens, device=a1.device, dtype=torch.int32)
        moe_route_counts(sorted_ids, num_valid_ids, expert_ids, route_counts, topk, block_m=SORT_BLOCK_M)
        moe_zero_rows(out, route_counts, num_rows)
    elif num_rows is None:
        out.zero_()
    else:
        # Every count != 1 zeroes the row: an all-zero count clears out below num_rows only.
        moe_zero_rows(out, torch.zeros(tokens, device=a1.device, dtype=torch.int32), num_rows)
    moe_gateup_small_m(a1, w1, a1_scale_t, w1_scale, sorted_ids, expert_ids, num_valid_ids, a2, a2_scale_t, topk, ring, b_aux)
    moe_down_small_m(
        a2, w2, a2_scale_t, w2_scale, sorted_ids, sorted_weights, expert_ids, num_valid_ids, out, topk, route_counts, ring, b_aux
    )
    return out
