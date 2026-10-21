# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""gfx950 FP8 blockscale MoE stage1: BF16 SiLU(gate) * up, without clamp.

Weight bytes are exactly AIter FP8 blockscale's shuffle_weight(w1, (16, 16)):
logical [E, 2*I, K], gate rows followed by up rows, physically
[E, 2*I/16, K/16, 16n, 16k]. This is NOT the MXFP4/MXFP8 GUGU shuffle.
The kernel consumes the shuffled buffer directly; no unshuffle is performed.

Scales deliberately keep the standalone GEMM contract, both FP32:
  scale_a[KB, tokens], scale_b[E, 2*I/128, KB], KB=K/128.
A and scale_a are gathered through expert-major packed sorted token IDs.
Output is [tokens, topk, I]; routing weights are NOT applied in stage1.

The compute pipeline is the 8-wave, 256x256, half-M single-FIFO GEMM:
TL[s0]/TR[s0]/BL[s0]/BR[s0], then TL[s1]/TR[s1]/BL[s1]/BR[s1].
Here s0/s1 denote slice0/slice1; left/right mean gate/up.
Each phase interleaves old-partial FP32 FMA with independent new MFMA.
DMA and wait placement follows the aiter blockscale GEMM's split-M schedule;
each A half loads both A64xK128 M-slice entries together, one DMA each.
With k_rotate, expert e walks its K tiles from tile e % KB (wrapping), so
concurrent workgroups do not request the same K column in lockstep.
Workgroups use 256-WG XCD batches, with no M grouping by default.
I must be a multiple of 128 and K a multiple of 256 (at least 256).
The module is importable; CLI modes include --small-accuracy, --accuracy,
--accuracy-matrix, --layout-check and --benchmark.
"""

import argparse
import json
import statistics

import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import arith, const_expr, range_constexpr, rocdl
from flydsl.expr.typing import Float8E4M3FN, Float32, T
from flydsl.expr.typing import Vector as Vec
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as _llvm

# 每个 WG 处理同一 expert 的 256 条排序路由行（含 padding）。
# N 方向分别计算 128 个 gate 和对应的 128 个 up：合计 256 列中间结果，
# 在寄存器中做 SiLU(gate) * up 后仅输出 128 列，而不是输出 256 列。
# BLOCK_M=128 是 top/bottom 各自的行数；BLOCK_N=128 是 gate/up 各自的列数。
# 权重仍是 GGUU（所有 gate 行在前、所有 up 行在后）；融合计算不等于
# gate/up interleave，shuffle_weight(16,16) 只做矩阵内部的分块重排。
SORT_BLOCK_M = 256
BLOCK_M = 128
BLOCK_N = 128
BLOCK_K = 128
TOKEN_MASK = 0xFFFFFF
BUFFER_LIMIT = 0x7FFFFFFF
# As an unsigned voffset this is past every buffer below 4 GiB, so loads return 0.
OOB_OFFSET = -1


def validate_parameters(tokens, intermediate_size, hidden_size, topk, num_experts):
    if not 0 < tokens <= TOKEN_MASK:
        raise ValueError(f"tokens must be in [1, {TOKEN_MASK}]")
    if intermediate_size <= 0 or intermediate_size % 128:
        raise ValueError("intermediate_size must be a positive multiple of 128")
    if hidden_size < 256 or hidden_size % 256:
        raise ValueError("hidden_size must be a multiple of 256 and at least 256")
    if not 0 < topk < 256 or num_experts < topk:
        raise ValueError("require 0 < topk < 256 and num_experts >= topk")
    # Buffer descriptors hold a 32-bit byte count.
    extents = {
        "a (tokens*K)": tokens * hidden_size,
        "a_scale (tokens*K/128*4)": tokens * (hidden_size // 128) * 4,
        "output (tokens*topk*I*2)": tokens * topk * intermediate_size * 2,
        "weight per expert (2*I*K)": 2 * intermediate_size * hidden_size,
    }
    over = [f"{name}={size:,}" for name, size in extents.items() if size >= 2**32]
    if over:
        raise ValueError("buffer byte extents must be < 4 GiB: " + ", ".join(over))


def _buffer_resource(tensor, num_records_bytes, base_byte_offset=0):
    ptr = fx.add_offset(
        fx.recast_iter(fx.Uint8, fx.get_iter(tensor)),
        fx.make_int_tuple(base_byte_offset),
    )
    view = fx.make_view(ptr, fx.make_layout(1, 1))
    buffer = rocdl.make_buffer_tensor(view, num_records_bytes=num_records_bytes)
    return rocdl.get_buffer_rsrc(fx.get_iter(buffer))


def _lds_byte_ptr(ptr, byte_offset):
    return fx.to_llvm_ptr(
        fx.add_offset(fx.recast_iter(fx.Uint8, ptr), fx.make_int_tuple(byte_offset))
    )


def compile_fp8_stage1_giu1(
    intermediate_size,
    hidden_size,
    topk,
    num_experts,
    *,
    xcd_swizzle=True,
    group_size_m=1,
    k_rotate=True,
):
    """Build a half-M, single-FIFO kernel for AIter-native FP8 weight bytes.

    Routing must contain complete 256-row expert blocks, each assigned to a
    valid local expert. Padding uses an invalid token/slot; each real
    (token, slot) must appear exactly once. No EP expert masking or bias.
    Native B slabs are read coalesced into unpadded LDS. Output routing IDs are
    prefetched before the K loop. Each DOT phase pairs four scalar FP32 FMAs with
    one MFMA; the launch disables packed FP32 ops to keep the FMAs scalar.
    The retained pipeline is half-M, one FIFO, B lookahead and scalar B scales;
    no full-M, K-unroll, priority-switch or alternative loader experiments.
    """
    validate_parameters(1, intermediate_size, hidden_size, topk, num_experts)
    if group_size_m <= 0:
        raise ValueError("group_size_m must be positive")
    kb_count = hidden_size // BLOCK_K
    n_tiles = intermediate_size // BLOCK_N
    # Keep A64 padding: tested row-major XOR layouts were slower on the main
    # gfx950 targets, even with precomputed offsets and zero register spills.
    # Saving 2KiB of LDS did not improve measured end-to-end kernel latency.
    a_group8 = 8 * BLOCK_K + 16
    a_group16 = 2 * a_group8 + 32
    a_slice_rows = BLOCK_M // 2
    a_lds_elems = (a_slice_rows // 16) * a_group16
    b_group16 = 16 * BLOCK_K
    b_lds_elems = (BLOCK_N // 16) * b_group16

    def _get_pids(pid, m_tiles, grid_size):
        if const_expr(xcd_swizzle):
            # Match bounded XCD batches without scattering adjacent experts
            # over the full grid. Leave the incomplete final batch linear.
            full_batches = grid_size - grid_size % 256
            if pid < full_batches:
                pid = (pid // 256) * 256 + (pid % 8) * 32 + (pid % 256) // 8
        group_id = pid // (group_size_m * n_tiles)
        first_m = group_id * group_size_m
        remaining = m_tiles - first_m
        group_m = (remaining < group_size_m).select(remaining, group_size_m)
        return (
            first_m + (pid % (group_size_m * n_tiles)) % group_m,
            (pid % (group_size_m * n_tiles)) // group_m,
        )

    get_pids = ASTRewriter.transform(_get_pids)

    @fx.struct
    class LDS:
        ##LDS A ping buffer
        a_t0_slice0: fx.Array[Float8E4M3FN, a_lds_elems, 16]
        a_t0_slice1: fx.Array[Float8E4M3FN, a_lds_elems, 16]
        a_b0_slice0: fx.Array[Float8E4M3FN, a_lds_elems, 16]
        a_b0_slice1: fx.Array[Float8E4M3FN, a_lds_elems, 16]

        ##LDS A pong buffer
        a_t1_slice0: fx.Array[Float8E4M3FN, a_lds_elems, 16]
        a_t1_slice1: fx.Array[Float8E4M3FN, a_lds_elems, 16]
        a_b1_slice0: fx.Array[Float8E4M3FN, a_lds_elems, 16]
        a_b1_slice1: fx.Array[Float8E4M3FN, a_lds_elems, 16]

        ##LDS B ping buffer
        b_l0: fx.Array[Float8E4M3FN, b_lds_elems, 16]
        b_r0: fx.Array[Float8E4M3FN, b_lds_elems, 16]

        ##LDS B pong buffer
        b_l1: fx.Array[Float8E4M3FN, b_lds_elems, 16]
        b_r1: fx.Array[Float8E4M3FN, b_lds_elems, 16]
        # One DMA per K128 tile: 256 row scales duplicated across 512 lanes.
        # Both compute M slices share these entries; do not split the scales.
        scale_a0: fx.Array[Float32, 512, 4]
        scale_a1: fx.Array[Float32, 512, 4]

    @flyc.kernel(known_block_size=[512, 1, 1])
    def fp8_stage1_kernel(
        arg_a: fx.Tensor,
        arg_b: fx.Tensor,
        arg_scale_a: fx.Tensor,
        arg_scale_b: fx.Tensor,
        arg_sorted_ids: fx.Tensor,
        arg_expert_ids: fx.Tensor,
        arg_num_valid_ids: fx.Tensor,
        arg_c: fx.Tensor,
        num_tokens: fx.Int32,
        num_expert_blocks: fx.Int32,
    ):
        tid = fx.thread_idx.x
        lane_id = tid % 64
        wave_id = tid // 64
        wave_m = wave_id // 4
        wave_n = wave_id % 4
        # The launch grid is num_expert_blocks * n_tiles. fx.grid_dim would call
        # __ockl_get_num_groups, which cannot be inlined under -packed-fp32-ops
        # (target-feature mismatch); its VGPR result would make every block
        # index divergent and add waterfall loops around the buffer accesses.
        block_id, n_tile = get_pids(
            fx.block_idx.x, num_expert_blocks, num_expert_blocks * n_tiles
        )
        block_id = fx.Int32(block_id)
        n_tile = fx.Int32(n_tile)
        sorted_base = block_id * SORT_BLOCK_M
        prefix_rsrc = _buffer_resource(arg_num_valid_ids, 4)
        prefix_rows = rocdl.raw_ptr_buffer_load(
            T.i32, prefix_rsrc, fx.Int32(0), fx.Int32(0)
        )
        prefix_rows = fx.Int32(rocdl.readfirstlane(T.i32, arith._to_raw(prefix_rows)))
        # Aiter passes full-capacity sort buffers: blocks past the valid prefix may
        # hold arbitrary expert IDs, so skip them before any expert/route load.
        # Distinct names: the guarded body re-assigns valid_rsrc/valid_rows itself.
        if sorted_base < prefix_rows:
            sorted_rsrc = _buffer_resource(
                arg_sorted_ids,
                arith._to_raw(fx.Int32(num_expert_blocks * SORT_BLOCK_M * 4)),
            )
            expert_rsrc = _buffer_resource(
                arg_expert_ids, arith._to_raw(fx.Int32(num_expert_blocks * 4))
            )
            expert = rocdl.raw_ptr_buffer_load(
                T.i32, expert_rsrc, block_id * 4, fx.Int32(0)
            )
            expert = fx.Int32(rocdl.readfirstlane(T.i32, arith._to_raw(expert)))
            a_rsrc = _buffer_resource(arg_a, fx.Int64(num_tokens) * hidden_size)
            b_slab = BLOCK_N * hidden_size
            b_gate_base = fx.Int64(expert) * (2 * intermediate_size * hidden_size) + (
                fx.Int64(n_tile) * b_slab
            )
            b_gate_rsrc = _buffer_resource(arg_b, b_slab, b_gate_base)
            b_up_rsrc = _buffer_resource(
                arg_b, b_slab, b_gate_base + intermediate_size * hidden_size
            )
            k_off = expert % kb_count
            scale_a_rsrc = _buffer_resource(
                arg_scale_a, fx.Int64(num_tokens) * (kb_count * 4)
            )
            scale_b_rsrc = _buffer_resource(
                arg_scale_b,
                2 * n_tiles * kb_count * 4,
                fx.Int64(expert) * (2 * n_tiles * kb_count * 4),
            )
            lds = fx.SharedAllocator().allocate(LDS).peek()
            # Map the low two logical row bits to the padded 16-row groups.
            # The matching producer below preserves a single 8KiB DMA per M slice.
            read_layout = fx.make_layout(
                ((a_slice_rows // 16, 2, 8), (32, BLOCK_K // 32)),
                ((a_group16, a_group8, BLOCK_K), (1, 32)),
            )
            a_top = [
                [fx.make_view(p, read_layout) for p in pair]
                for pair in (
                    (lds.a_t0_slice0.ptr, lds.a_t0_slice1.ptr),
                    (lds.a_t1_slice0.ptr, lds.a_t1_slice1.ptr),
                )
            ]
            a_bottom = [
                [fx.make_view(p, read_layout) for p in pair]
                for pair in (
                    (lds.a_b0_slice0.ptr, lds.a_b0_slice1.ptr),
                    (lds.a_b1_slice0.ptr, lds.a_b1_slice1.ptr),
                )
            ]
            # Keep native N16/K16 bytes in LDS. Move the existing logical-row
            # permutation to the consumer so each producer wave reads 1024B
            # contiguously instead of eight scattered rows from eight N16 groups.
            b_read_layout = fx.make_layout(
                ((2, BLOCK_N // 16, 8), ((16, 2), BLOCK_K // 32)),
                ((16, 32, b_group16), ((1, 256), 512)),
            )
            b_gate = [
                fx.make_view(p, b_read_layout) for p in (lds.b_l0.ptr, lds.b_l1.ptr)
            ]
            b_up = [
                fx.make_view(p, b_read_layout) for p in (lds.b_r0.ptr, lds.b_r1.ptr)
            ]

            # Transposed C, as in blockscale GEMM: actual B is MMA operand A.
            mma_atom = fx.make_mma_atom(
                fx.rocdl.cdna4.MFMA_Scale(16, 16, 128, Float8E4M3FN)
            )
            mma_atom = fx.atom_set_value(mma_atom, "scale_a", fx.Int32(0))
            mma_atom = fx.atom_set_value(mma_atom, "scale_b", fx.Int32(0))
            tiled_mma = fx.make_tiled_mma(
                mma_atom,
                fx.make_layout((4, 2, 1), (1, 4, 0)),
                (None, None, fx.make_layout((32, 4), (1, 32))),
            )
            thread_mma = tiled_mma.thr_slice(tid)
            copy_atom = fx.make_copy_atom(fx.UniversalCopy128b(), Float8E4M3FN)
            copy_a = fx.make_tiled_copy_B(copy_atom, tiled_mma).get_slice(tid)
            copy_b = fx.make_tiled_copy_A(copy_atom, tiled_mma).get_slice(tid)

            frag_a = thread_mma.make_fragment_B(a_top[0][0])
            frag_bl = thread_mma.make_fragment_A(b_gate[0])
            frag_br = thread_mma.make_fragment_A(b_up[0])
            dest_a = copy_a.retile(frag_a)
            dest_bl = copy_b.retile(frag_bl)
            dest_br = copy_b.retile(frag_br)
            c_tile = fx.make_rmem_tensor(
                fx.make_ordered_layout((BLOCK_N, BLOCK_M), (1, 0)), Float32
            )
            c_tl = thread_mma.make_fragment_C(c_tile)
            c_tr = thread_mma.make_fragment_C(c_tile)
            c_bl = thread_mma.make_fragment_C(c_tile)
            c_br = thread_mma.make_fragment_C(c_tile)
            p_tile = fx.flat_divide(c_tile, (BLOCK_N, 64))[None, None, 0, 0]
            frag_p = thread_mma.make_fragment_C(p_tile)

            def sorted_id(row):
                return fx.Int32(
                    rocdl.raw_ptr_buffer_load(
                        T.i32,
                        sorted_rsrc,
                        fx.Int32((sorted_base + row) * 4),
                        fx.Int32(0),
                    )
                )

            store_ids = [
                sorted_id(bottom * 128 + repeat * 32 + wave_m * 16 + lane_id % 16)
                for bottom in range_constexpr(2)
                for repeat in range_constexpr(4)
            ]

            # One cooperative DWORDx4 DMA covers one complete A64xK128 entry.
            # Within an entry r = wave_id//2 + 4*(wave_id%2) + 8*(lane_id//8).
            # This is the inverse of the (4,2,8) consumer's padded row mapping.
            a_voffsets = []
            for bottom in range_constexpr(2):
                offsets = []
                for m_slice in range_constexpr(2):
                    local_row = wave_id // 2 + (wave_id % 2) * 4 + (lane_id // 8) * 8
                    row = bottom * 128 + m_slice * a_slice_rows + local_row
                    fused = sorted_id(row)
                    token = fused & TOKEN_MASK
                    slot = (fused >> 24) & 255
                    valid = (token < num_tokens) & (slot < topk)
                    offset = token * hidden_size + fx.Int32((tid % 8) * 16)
                    offsets.append(valid.select(offset, fx.Int32(OOB_OFFSET)))
                a_voffsets.append(offsets)

            scale_id = sorted_id(tid % SORT_BLOCK_M)
            scale_token = scale_id & TOKEN_MASK
            scale_valid = (scale_token < num_tokens) & (((scale_id >> 24) & 255) < topk)
            scale_a_voffset = scale_valid.select(scale_token * 4, fx.Int32(OOB_OFFSET))
            wave_dma_base = rocdl.readfirstlane(
                T.i32,
                arith._to_raw(
                    fx.Int32((wave_id % 2) * a_group8 + (wave_id // 2) * a_group16)
                ),
            )
            scale_dma_base = rocdl.readfirstlane(
                T.i32, arith._to_raw(fx.Int32(tid * 4))
            )

            def operand_dma_ptrs(views):
                return [
                    [_lds_byte_ptr(fx.get_iter(view), wave_dma_base) for view in pair]
                    for pair in views
                ]

            at_dma = operand_dma_ptrs(a_top)
            ab_dma = operand_dma_ptrs(a_bottom)
            b_wave_base = rocdl.readfirstlane(
                T.i32, arith._to_raw(fx.Int32(wave_id * b_group16))
            )

            def b_dma_ptrs(views):
                return [_lds_byte_ptr(fx.get_iter(view), b_wave_base) for view in views]

            bl_dma = b_dma_ptrs(b_gate)
            br_dma = b_dma_ptrs(b_up)
            scale_dma = [
                _lds_byte_ptr(p, scale_dma_base)
                for p in (lds.scale_a0.ptr, lds.scale_a1.ptr)
            ]

            # Native FP8 offset = (n//16)*16*K + (k//16)*256 + (n%16)*16 + k%16.
            # A wave copies an N16/K64 slab, two slabs cover K128. No host-side
            # weight conversion; the matching LDS view preserves MFMA operands.
            b_voffset = fx.Int32(wave_id * (16 * hidden_size) + lane_id * 16)

            def k_tile(ki):
                if const_expr(not k_rotate):
                    return ki
                t = k_off + ki
                return (t < kb_count).select(t, t - kb_count)

            def tile_soffset(ki, tile_bytes):
                # Prefetches past the last K tile are all-lane OOB and move no data.
                value = fx.Int32(k_tile(ki) * tile_bytes)
                if const_expr(isinstance(ki, int)):
                    return value
                return (ki < kb_count).select(value, fx.Int32(-1))

            def raw_g2s_a(lds_dst, voffset, soffset):
                rocdl.raw_ptr_buffer_load_lds(
                    a_rsrc,
                    lds_dst,
                    fx.Int32(16),
                    voffset,
                    soffset,
                    fx.Int32(0),
                    fx.Int32(0),
                )

            def async_copy_At(lds_idx, ki):
                soffset = tile_soffset(ki, BLOCK_K)
                for m_slice in range_constexpr(2):
                    raw_g2s_a(at_dma[lds_idx][m_slice], a_voffsets[0][m_slice], soffset)

            def async_copy_Ab(lds_idx, ki):
                soffset = tile_soffset(ki, BLOCK_K)
                for m_slice in range_constexpr(2):
                    raw_g2s_a(ab_dma[lds_idx][m_slice], a_voffsets[1][m_slice], soffset)

            def raw_g2s_b(rsrc, lds_dst, ki):
                soffset = tile_soffset(ki, BLOCK_K * 16)
                for chunk in range_constexpr(2):
                    # The immediate offset advances both the global and the LDS address.
                    rocdl.raw_ptr_buffer_load_lds(
                        rsrc,
                        lds_dst,
                        fx.Int32(16),
                        b_voffset,
                        soffset,
                        fx.Int32(chunk * 1024),
                        fx.Int32(0),
                    )

            def async_copy_Bl(lds_idx, ki):
                raw_g2s_b(b_gate_rsrc, bl_dma[lds_idx], ki)

            def async_copy_Br(lds_idx, ki):
                raw_g2s_b(b_up_rsrc, br_dma[lds_idx], ki)

            def async_copy_scale_a(lds_idx, ki):
                rocdl.raw_ptr_buffer_load_lds(
                    scale_a_rsrc,
                    scale_dma[lds_idx],
                    fx.Int32(4),
                    scale_a_voffset,
                    tile_soffset(ki, num_tokens * 4),
                    fx.Int32(0),
                    fx.Int32(0),
                )

            def lds_rd_At(lds_idx, m_slice):
                fx.copy(copy_atom, copy_a.partition_S(a_top[lds_idx][m_slice]), dest_a)

            def lds_rd_Ab(lds_idx, m_slice):
                fx.copy(
                    copy_atom, copy_a.partition_S(a_bottom[lds_idx][m_slice]), dest_a
                )

            def lds_rd_Bl(lds_idx):
                fx.copy(copy_atom, copy_b.partition_S(b_gate[lds_idx]), dest_bl)

            def lds_rd_Br(lds_idx):
                fx.copy(copy_atom, copy_b.partition_S(b_up[lds_idx]), dest_br)

            scale_copy = fx.make_copy_atom(fx.UniversalCopy32b(), Float32)

            def lds_rd_scale_a(lds_idx, bottom, m_slice):
                values = []
                for m0 in range_constexpr(2):
                    index = wave_m * 256 + bottom * 128 + wave_m * 16 + lane_id % 16
                    index = index + (m_slice * 2 + m0) * 32
                    ptr = (
                        lds.scale_a0.ptr
                        if const_expr(lds_idx == 0)
                        else lds.scale_a1.ptr
                    )
                    src = fx.make_view(
                        fx.add_offset(ptr, fx.make_int_tuple(index)),
                        fx.make_layout(1, 1),
                    )
                    frag = fx.make_fragment_like(src)
                    fx.copy(scale_copy, src, frag)
                    values.append(Vec(frag.load())[0])
                return Vec.from_elements(values, Float32)

            # Gate/up ScaleB of this N tile and K tile are wave-uniform: two SMEM
            # loads into SGPRs. Inline s_buffer_load_dword works around the pinned
            # LLVM: SITargetLowering::lowerSBuffer() drops the MachineMemOperand,
            # so the backend cannot see the source buffer and inserts extra
            # waitcnts that cost significant performance.
            def load_scale_b(ki):
                offset = fx.Int32((n_tile * kb_count + k_tile(ki)) * 4)
                result = _llvm.inline_asm(
                    ir.Type.parse("!llvm.struct<(f32, f32)>"),
                    [
                        arith._to_raw(scale_b_rsrc),
                        arith._to_raw(offset),
                        arith._to_raw(offset + n_tiles * kb_count * 4),
                    ],
                    "s_buffer_load_dword $0, $2, $3\ns_buffer_load_dword $1, $2, $4",
                    "=&s,=&s,s,s,s,~{memory}",
                    has_side_effects=True,
                )
                return Vec.from_elements(
                    [
                        fx.Float32(_llvm.extractvalue(T.f32, result, [i]))
                        for i in range_constexpr(2)
                    ],
                    Float32,
                )

            def compute(c_frag, b_frag, dq_scale_a, dq_scale_b, dq_m_slice):
                # One DOT phase: retire the pending P FIFO into c_frag with the
                # scales of the phase that produced it, then refill P with this
                # phase's b_frag x frag_a products.
                # Per-lane fragments of a wave's 64(M) x 32(N) share of a C tile:
                #   frag_a [Kval, Mrep, Krep]: (32, 2, 1) FP8 -> 16 dwords
                #   b_frag [Kval, Nrep, Krep]: (32, 2, 1) FP8 -> 16 dwords
                #   c_frag [Cval, Nrep, Mrep]: (4, 2, 4) F32  -> 32 dwords
                #   frag_p [Cval, Nrep, Mrep]: (4, 2, 2) F32  -> 16 dwords
                # frag_a holds one M slice; c_frag keeps both (Mrep 0-1 and 2-3).
                # dq_scale_a has two row scales; dq_scale_b is wave-uniform.
                #
                # for m0 in (Mrep):
                #     dq_scale = dq_scale_a[m0] * dq_scale_b
                #     for n0 in (Nrep):
                #         c_frag[:, n0, dq_m_slice * 2 + m0] += dq_scale * frag_p[:, n0, m0]
                #         frag_p[:, n0, m0] = mfma_16x16x128(b_frag[:, n0], frag_a[:, m0], 0)
                #
                # Compiler-only sched_barrier fences keep 4 scalar FMAs -> 1 MFMA;
                # the launch's -packed-fp32-ops prevents re-packing into
                # v_pk_fma_f32. Retired and new partials are independent, so the
                # FMAs overlap the MFMA pipeline. WG sync stays in the caller.
                # The MFMA is the raw intrinsic with zero scales (the unscaled
                # form): on the pinned FlyDSL, fx.gemm rejects these FP8 register
                # fragments in rmem SSA promotion, and their DWORD views are not
                # contiguous 8-dword operands.
                for m0 in range_constexpr(2):
                    dq_scale = Vec(dq_scale_a)[m0] * dq_scale_b
                    rocdl.sched_barrier(0)
                    for n0 in range_constexpr(2):
                        sub_c = c_frag[None, n0, dq_m_slice * 2 + m0]
                        sub_p = frag_p[None, n0, m0]
                        partial = Vec(sub_p.load())
                        accum = Vec(sub_c.load())
                        sub_c.store(
                            Vec.from_elements(
                                [
                                    fx.fma(partial[i], dq_scale, accum[i])
                                    for i in range_constexpr(4)
                                ],
                                Float32,
                            )
                        )
                        rocdl.sched_barrier(0)
                        sub_p.store(
                            rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                                T.vec(4, T.f32),
                                [
                                    Vec(b_frag[None, n0, 0].load()).bitcast(fx.Int32),
                                    Vec(frag_a[None, m0, 0].load()).bitcast(fx.Int32),
                                    Vec.filled(4, 0.0, Float32),
                                    0,
                                    0,
                                    0,
                                    fx.Int32(0),
                                    0,
                                    fx.Int32(0),
                                ],
                            )
                        )
                        rocdl.sched_barrier(0)

            def begin_compute_phase():
                rocdl.sched_barrier(0)
                rocdl.s_waitcnt(lgkmcnt=0)
                rocdl.sched_barrier(0)
                rocdl.s_barrier()
                rocdl.sched_barrier(0)

            def end_compute_phase():
                rocdl.sched_barrier(0)
                rocdl.s_barrier()
                rocdl.sched_barrier(0)

            # Drain routing VMEM before the pipeline's counted DMA batches start.
            rocdl.sched_barrier(0)
            rocdl.s_waitcnt(vmcnt=0)
            rocdl.sched_barrier(0)
            # K tile 0: ScaleA (1 VMEM op), Bl (2), At (2), Br (2) and Ab (2).
            async_copy_scale_a(lds_idx=0, ki=fx.Int32(0))
            rocdl.sched_barrier(0)
            async_copy_Bl(lds_idx=0, ki=0)
            rocdl.sched_barrier(0)
            async_copy_At(lds_idx=0, ki=0)
            rocdl.sched_barrier(0)
            async_copy_Br(lds_idx=0, ki=0)
            rocdl.sched_barrier(0)
            async_copy_Ab(lds_idx=0, ki=0)
            rocdl.sched_barrier(0)
            # Offset the two groups of four waves by one stage. The lower group
            # closes this unmatched barrier after the final FIFO drain.
            if wave_id >= 4:
                rocdl.s_barrier()
            c_tl.fill(0)
            c_tr.fill(0)
            c_bl.fill(0)
            c_br.fill(0)
            # Tile 0 ScaleA, Bl and At have landed; its Br and Ab may still fly.
            rocdl.s_waitcnt(vmcnt=4)
            rocdl.s_barrier()
            rocdl.sched_barrier(0)
            # K tile 1: ScaleA (1), At (2), Bl (2) and Br (2). Its Ab is issued
            # by the loop's first TL phase.
            async_copy_scale_a(lds_idx=1, ki=fx.Int32(1))
            rocdl.sched_barrier(0)
            async_copy_At(lds_idx=1, ki=1)
            rocdl.sched_barrier(0)
            async_copy_Bl(lds_idx=1, ki=1)
            rocdl.sched_barrier(0)
            async_copy_Br(lds_idx=1, ki=1)
            rocdl.sched_barrier(0)
            # All of tile 0 has landed; the seven tile-1 ops may still fly.
            rocdl.s_waitcnt(vmcnt=7)
            rocdl.s_barrier()
            rocdl.sched_barrier(0)
            # B0 of tile 0 is read one phase ahead; the first DQ retires zeros.
            lds_rd_Bl(lds_idx=0)
            frag_p.fill(0)
            # Register fragments (C quadrants, the P FIFO and the B0 lookahead)
            # are promoted to loop-carried SSA automatically; FP8 register storage
            # uses the compiler's integer representation. Initialize the pending
            # P scales once, then let range carry their updates.
            old_a = Vec.filled(2, 0.0, Float32)
            old_b = fx.Float32(0)
            for k_index in range(0, kb_count, 2):
                kiter = fx.Int32(k_index)
                # One K128 tile k (repeat for both unroll_idx values). Tile k
                # loads K tile k_tile(k), rotated per expert when k_rotate is set.
                # Ahs = A[half h, M slice s]; B0/B1 = gate/up; SA = ScaleA.
                # t = tick (k % 2), u = tock; operands use tile k unless noted.
                # Each DOT phase: 4 MFMA + 16 FMA retiring the previous frag_p.
                # LDS reads / async copies (AC) precede that phase's begin barrier.
                # | Phase | MFMA   | DQ (VALU)   | LDS read (slot) | AC (slot, tile) / wait |
                # |-------|--------|-------------|-----------------|------------------------|
                # | TL s0 | A00*B0 | BR s1 (k-1) | A00, SA00 (t)   | A1 (u, k+1)            |
                # | TR s0 | A00*B1 | TL s0       | B1 (t)          | -                      |
                # | BL s0 | A10*B0 | TR s0       | A10, SA10 (t)   | B0 (t, k+2)            |
                # | BR s0 | A10*B1 | BL s0       | -               | B1 (t, k+2)            |
                # | TL s1 | A01*B0 | BR s0       | A01, SA01 (t)   | -                      |
                # | TR s1 | A01*B1 | TL s1       | -               | A0 (t, k+2)            |
                # | BL s1 | A11*B0 | TR s1       | A11, SA11 (t)   | vmcnt(6)               |
                # | BR s1 | A11*B1 | BL s1       | B0 (u, k+1)     | SA (t, k+2), before LR |
                # Every phase: LR/AC/wait -> lgkmcnt(0) -> s_barrier -> DOT -> s_barrier.
                # TL s0 also loads both ScaleB SGPRs (SMEM, not LDS or AC).
                # BR s1's pending partial is retired by the next TL s0 or the drain.
                for unroll_idx in range_constexpr(2):
                    tick = unroll_idx
                    tock = 1 - unroll_idx
                    ki = kiter + unroll_idx
                    for m_slice in range_constexpr(2):
                        lds_rd_At(lds_idx=tick, m_slice=m_slice)
                        top_scale = lds_rd_scale_a(
                            lds_idx=tick, bottom=0, m_slice=m_slice
                        )
                        if const_expr(m_slice == 0):
                            b_scale = load_scale_b(ki=ki)
                            async_copy_Ab(lds_idx=tock, ki=ki + 1)
                        begin_compute_phase()
                        # New TL; consume BR from the preceding M slice/K tile.
                        compute(c_br, frag_bl, old_a, old_b, 1 - m_slice)
                        end_compute_phase()
                        if const_expr(m_slice == 0):
                            lds_rd_Br(lds_idx=tick)
                        else:
                            # Both At[tick] slices were last read two or more phases ago.
                            async_copy_At(lds_idx=tick, ki=ki + 2)
                        begin_compute_phase()
                        compute(c_tl, frag_br, top_scale, b_scale[0], m_slice)
                        end_compute_phase()
                        lds_rd_Ab(lds_idx=tick, m_slice=m_slice)
                        bottom_scale = lds_rd_scale_a(
                            lds_idx=tick, bottom=1, m_slice=m_slice
                        )
                        if const_expr(m_slice == 0):
                            # B registers persist through slice1; LDS can be reused.
                            async_copy_Bl(lds_idx=tick, ki=ki + 2)
                        else:
                            # Only the six ki+2 At/Bl/Br DMAs may remain in flight:
                            # after BL s1's barriers every wave's ki+1 tile has
                            # landed for BR s1's B0 (u, k+1) read.
                            rocdl.s_waitcnt(vmcnt=6)
                        begin_compute_phase()
                        compute(c_tr, frag_bl, top_scale, b_scale[1], m_slice)
                        end_compute_phase()
                        if const_expr(m_slice == 0):
                            async_copy_Br(lds_idx=tick, ki=ki + 2)
                        else:
                            async_copy_scale_a(lds_idx=tick, ki=ki + 2)
                            lds_rd_Bl(lds_idx=tock)
                        begin_compute_phase()
                        compute(c_bl, frag_br, bottom_scale, b_scale[0], m_slice)
                        end_compute_phase()
                        old_a, old_b = bottom_scale, b_scale[1]

            # Drain the final BR s1 partial using the state retained by the loop.
            for m0 in range_constexpr(2):
                dq_scale = Vec(old_a)[m0] * old_b
                for n0 in range_constexpr(2):
                    sub_c = c_br[None, n0, 2 + m0]
                    partial = Vec(frag_p[None, n0, m0].load())
                    accum = Vec(sub_c.load())
                    sub_c.store(
                        Vec.from_elements(
                            [
                                fx.fma(partial[i], dq_scale, accum[i])
                                for i in range_constexpr(4)
                            ],
                            Float32,
                        )
                    )
            if wave_id < 4:
                rocdl.s_barrier()

            valid_rsrc = _buffer_resource(arg_num_valid_ids, 4)
            valid_rows = fx.Int32(
                rocdl.raw_ptr_buffer_load(T.i32, valid_rsrc, fx.Int32(0), fx.Int32(0))
            )
            c_rsrc = _buffer_resource(
                arg_c, fx.Int64(num_tokens) * (topk * intermediate_size * 2)
            )
            pair_type = ir.Type.parse("!llvm.struct<(i32, i32)>")

            def silu_mul(gate, up):
                exponent = rocdl.exp2(T.f32, arith._to_raw(gate * -1.4426950408889634))
                inverse = rocdl.rcp(T.f32, 1.0 + exponent)
                return (gate * inverse) * up

            def store_gate_up(gates, ups, bottom):
                for row_repeat in range_constexpr(4):
                    row = bottom * 128 + row_repeat * 32 + wave_m * 16 + lane_id % 16
                    fused = store_ids[bottom * 4 + row_repeat]
                    token, slot = fused & TOKEN_MASK, (fused >> 24) & 255
                    valid = (
                        (token < num_tokens)
                        & (slot < topk)
                        & (sorted_base + row < valid_rows)
                    )
                    if valid:
                        g0, g1 = Vec(gates[None, 0, row_repeat].load()), Vec(
                            gates[None, 1, row_repeat].load()
                        )
                        u0, u1 = Vec(ups[None, 0, row_repeat].load()), Vec(
                            ups[None, 1, row_repeat].load()
                        )
                        v0 = [silu_mul(g0[i], u0[i]) for i in range_constexpr(4)]
                        v1 = [silu_mul(g1[i], u1[i]) for i in range_constexpr(4)]
                        # In the transposed C, lane_id % 16 selects the row and the
                        # lane group (lane_id // 16) four consecutive columns.
                        # aX/bX are output columns of N repeat 0/1 (64 apart),
                        # relative to the wave's N base:
                        # | lane | p0      | p1      | p2      | p3      |
                        # |------|---------|---------|---------|---------|
                        # |    0 | a0,a1   | a2,a3   | b0,b1   | b2,b3   |
                        # |   16 | a4,a5   | a6,a7   | b4,b5   | b6,b7   |
                        # |   32 | a8,a9   | a10,a11 | b8,b9   | b10,b11 |
                        # |   48 | a12,a13 | a14,a15 | b12,b13 | b14,b15 |
                        # permlane16_swap exchanges the odd groups' p0/p1 with the
                        # even groups' p2/p3 (partner lane = lane_id ^ 16), so each
                        # lane then owns eight consecutive columns of its row:
                        # | lane | swap0[0], swap1[0], swap0[1], swap1[1] | columns |
                        # |------|----------------------------------------|---------|
                        # |    0 | a0..a7                                 | 0-7     |
                        # |   16 | b0..b7                                 | 64-71   |
                        # |   32 | a8..a15                                | 8-15    |
                        # |   48 | b8..b15                                | 72-79   |
                        p0 = rocdl.cvt_pk_bf16_f32(v0[0], v0[1])
                        p1 = rocdl.cvt_pk_bf16_f32(v0[2], v0[3])
                        p2 = rocdl.cvt_pk_bf16_f32(v1[0], v1[1])
                        p3 = rocdl.cvt_pk_bf16_f32(v1[2], v1[3])
                        swap0 = rocdl.permlane16_swap(
                            pair_type,
                            arith._to_raw(p0),
                            arith._to_raw(p2),
                            False,
                            False,
                        )
                        swap1 = rocdl.permlane16_swap(
                            pair_type,
                            arith._to_raw(p1),
                            arith._to_raw(p3),
                            False,
                            False,
                        )
                        packed = Vec.from_elements(
                            [
                                fx.Int32(_llvm.extractvalue(T.i32, swap0, [0])),
                                fx.Int32(_llvm.extractvalue(T.i32, swap1, [0])),
                                fx.Int32(_llvm.extractvalue(T.i32, swap0, [1])),
                                fx.Int32(_llvm.extractvalue(T.i32, swap1, [1])),
                            ],
                            fx.Int32,
                        )
                        lane_group = lane_id // 16
                        col = (
                            n_tile * 128
                            + lane_group % 2 * 64
                            + wave_n * 16
                            + lane_group // 2 * 8
                        )
                        address = (
                            (token * topk + slot) * intermediate_size + fx.Int32(col)
                        ) * 2
                        rocdl.raw_ptr_buffer_store(
                            packed.ir_value(),
                            c_rsrc,
                            address.ir_value(),
                            fx.Int32(0).ir_value(),
                            aux=ir.IntegerAttr.get(T.i32, 0),
                        )

            store_gate_up(c_tl, c_tr, 0)
            store_gate_up(c_bl, c_br, 1)

    @flyc.jit
    def launch_stage1(
        a: fx.Tensor,
        b: fx.Tensor,
        scale_a: fx.Tensor,
        scale_b: fx.Tensor,
        sorted_ids: fx.Tensor,
        expert_ids: fx.Tensor,
        num_valid_ids: fx.Tensor,
        out: fx.Tensor,
        num_tokens: fx.Int32,
        num_expert_blocks: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        fp8_stage1_kernel(
            a,
            b,
            scale_a,
            scale_b,
            sorted_ids,
            expert_ids,
            num_valid_ids,
            out,
            num_tokens,
            num_expert_blocks,
            # Keep the DOT-phase FP32 FMAs scalar: no v_pk_fma_f32 re-packing.
            value_attrs={"llvm.passthrough": [["target-features", "-packed-fp32-ops"]]},
        ).launch(
            grid=(num_expert_blocks * n_tiles, 1, 1), block=(512, 1, 1), stream=stream
        )

    return launch_stage1


def check_native_weight_layout():
    """CPU byte-level check against the actual AIter shuffle, not its flag."""
    from aiter.ops.shuffle import shuffle_weight

    experts, intermediate, hidden = 2, 256, 512
    raw = (torch.arange(experts * 2 * intermediate * hidden, device="cpu") % 251).to(
        torch.uint8
    )
    raw = raw.view(experts, 2 * intermediate, hidden)
    shuffled = shuffle_weight(raw, (16, 16))
    n = torch.arange(2 * intermediate, device="cpu")[:, None]
    k = torch.arange(hidden, device="cpu")[None, :]
    offsets = (n // 16) * (16 * hidden) + (k // 16) * 256 + (n % 16) * 16 + k % 16
    for expert in range(experts):
        torch.testing.assert_close(
            shuffled[expert].reshape(-1)[offsets], raw[expert], rtol=0, atol=0
        )
    # Check the coalesced producer's global and LDS addresses, including both
    # K64 slabs, gate/up blocks, and first/last K128 tiles.
    tid = torch.arange(512, device="cpu")
    for hidden in (256, 512, 6144, 16384):
        for row_tile in (0, 1, 3):
            for kb in (0, hidden // 128 - 1):
                for chunk in (0, 1):
                    row = row_tile * 128 + (tid // 64) * 16 + tid % 16
                    col = kb * 128 + (tid % 64 // 16) * 16 + chunk * 64
                    expected = (
                        (row // 16) * (16 * hidden)
                        + (col // 16) * 256
                        + (row % 16) * 16
                    )
                    actual = (
                        (tid // 64) * (16 * hidden)
                        + (tid % 64) * 16
                        + row_tile * 128 * hidden
                        + kb * (128 * 16)
                        + chunk * 1024
                    )
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    lds_address = (
                        (tid // 64) * (16 * 128) + (tid % 64) * 16 + chunk * 1024
                    )
                    local_row, local_col = row % 128, col % 128
                    consumer_address = (
                        (local_row // 16) * (16 * 128)
                        + (local_col // 16) * 256
                        + (local_row % 16) * 16
                    )
                    torch.testing.assert_close(
                        lds_address, consumer_address, rtol=0, atol=0
                    )
    print("layout: AIter shuffle_weight(16,16) native byte mapping PASS")
    return True


def check_a_slice_layout():
    """Exhaustive CPU check of all bytes written/read in each A64 entry."""
    group8 = 8 * BLOCK_K + 16
    group16 = 2 * group8 + 32
    tid = torch.arange(512, device="cpu")[:, None]
    byte = torch.arange(16, device="cpu")[None, :]
    wave = tid // 64
    local_row = wave // 2 + (wave % 2) * 4 + (tid % 64 // 8) * 8
    col = (tid % 8) * 16 + byte
    producer = (
        (tid // 64 % 2) * group8 + (tid // 128) * group16 + (tid % 64) * 16 + byte
    )
    consumer = (
        (local_row % 4) * group16
        + (local_row // 4 % 2) * group8
        + (local_row // 8) * BLOCK_K
        + col
    )
    entry_bytes = 4 * group16
    # Every adjacent group of eight lanes reads exactly one complete row.
    torch.testing.assert_close(
        col.reshape(64, 128).sort(dim=1).values,
        torch.arange(128).expand(64, 128),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(producer, consumer, rtol=0, atol=0)
    assert producer.unique().numel() == 64 * BLOCK_K
    assert int(producer.max()) < entry_bytes
    logical = local_row * BLOCK_K + col
    torch.testing.assert_close(
        logical.flatten().sort().values, torch.arange(64 * BLOCK_K), rtol=0, atol=0
    )
    # Check all four M slices and both ping/pong generations, including boundaries.
    for ping in range(2):
        physical, routed = [], []
        for bottom in range(2):
            for m_slice in range(2):
                entry = ping * 4 + bottom * 2 + m_slice
                physical.append((entry * entry_bytes + producer).flatten())
                routed.append(
                    (
                        (bottom * 128 + m_slice * 64 + local_row) * BLOCK_K + col
                    ).flatten()
                )
        assert torch.cat(physical).unique().numel() == 256 * BLOCK_K
        torch.testing.assert_close(
            torch.cat(routed).sort().values, torch.arange(256 * BLOCK_K), rtol=0, atol=0
        )
    print("layout: padded A64 byte mapping/coalescing PASS")
    return True


def prepare_inputs(
    tokens,
    intermediate_size,
    hidden_size,
    topk,
    num_experts,
    *,
    routing="balanced",
    data_case="random",
):
    from aiter.fused_moe import moe_sorting
    from aiter.ops.shuffle import shuffle_weight

    validate_parameters(tokens, intermediate_size, hidden_size, topk, num_experts)
    if routing == "balanced":
        choices = (
            torch.arange(tokens * topk, device="cuda").view(tokens, topk) % num_experts
        )
        choices = choices[torch.randperm(tokens, device="cuda")]
        topk_ids = torch.randperm(num_experts, device="cuda")[choices].to(torch.int32)
    elif routing == "random":
        topk_ids = (
            torch.rand((tokens, num_experts), device="cuda")
            .topk(topk, dim=1)
            .indices.to(torch.int32)
        )
    elif routing == "skewed":
        topk_ids = (
            torch.randperm(num_experts, device="cuda")[:topk]
            .to(torch.int32)[None, :]
            .repeat(tokens, 1)
        )
    else:
        raise ValueError(f"unsupported routing: {routing}")
    weights = torch.randn((tokens, topk), device="cuda", dtype=torch.float32)
    sorted_ids, _, expert_ids, valid_ids, _ = moe_sorting(
        topk_ids,
        weights,
        num_experts,
        hidden_size,
        torch.bfloat16,
        block_size=SORT_BLOCK_M,
        accumulate=False,
    )
    sorted_count = int(valid_ids[0].item())
    if sorted_count % SORT_BLOCK_M or sorted_count * 4 >= BUFFER_LIMIT:
        raise ValueError("sorted routes must fit a 31-bit, 256-row-aligned buffer")
    sorted_ids = sorted_ids[:sorted_count].contiguous()
    expert_ids = expert_ids[: sorted_count // SORT_BLOCK_M].contiguous()
    if not bool(((expert_ids >= 0) & (expert_ids < num_experts)).all()):
        raise ValueError("only valid local experts are supported")
    kb = hidden_size // 128
    a_source = torch.randn((tokens, kb, 128), device="cuda") * 0.33
    a_scale_raw = a_source.abs().amax(dim=-1).clamp_min(1e-8) / 448.0
    a = (
        (a_source / a_scale_raw[..., None])
        .to(torch.float8_e4m3fn)
        .view(tokens, hidden_size)
    )
    b_source = (
        torch.randn(
            (num_experts, 2 * intermediate_size // 128, 128, kb, 128), device="cuda"
        )
        * 0.1
    )
    b_scale = b_source.abs().amax(dim=(2, 4)).clamp_min(1e-8) / 448.0
    b_raw = (
        (b_source / b_scale[:, :, None, :, None])
        .to(torch.float8_e4m3fn)
        .view(num_experts, 2 * intermediate_size, hidden_size)
    )
    del a_source, b_source
    # Exercise genuinely different FP32 scales, including different gate/up groups.
    a_scale_raw *= 0.5 + torch.rand_like(a_scale_raw)
    b_scale *= 0.5 + torch.rand_like(b_scale)
    if data_case == "zero_a":
        a.view(torch.uint8).zero_()
    elif data_case == "zero_a_scale":
        a_scale_raw.zero_()
    elif data_case == "zero_b_scale":
        b_scale.zero_()
    elif data_case == "last_k":
        a_scale_raw[:, :-1] = 0
    elif data_case == "zero_gate":
        b_raw.view(torch.uint8)[:, :intermediate_size].zero_()
    elif data_case == "zero_up":
        b_raw.view(torch.uint8)[:, intermediate_size:].zero_()
    elif data_case != "random":
        raise ValueError(f"unsupported data_case: {data_case}")
    # Same weight layout as AIter FP8 blockscale; NOT is_guinterleave=True.
    b = shuffle_weight(b_raw, (16, 16))
    return {
        "a": a,
        "weight": b,
        "weight_raw": b_raw,
        "scale_a": a_scale_raw.t().contiguous(),
        "scale_b": b_scale.contiguous(),
        "topk_ids": topk_ids,
        "sorted_ids": sorted_ids,
        "expert_ids": expert_ids,
        "num_valid_ids": valid_ids,
    }


def stage1_reference(inputs, intermediate_size):
    """Independent logical-weight/topk reference; never use the kernel's routing."""
    a, raw_b = inputs["a"], inputs["weight_raw"]
    tokens, hidden = a.shape
    experts = raw_b.shape[0]
    kb = hidden // 128
    a_dequant = (
        a.float().view(tokens, kb, 128) * inputs["scale_a"].t()[:, :, None]
    ).view(tokens, hidden)
    topk_ids = inputs["topk_ids"]
    out = torch.empty(
        (tokens, topk_ids.shape[1], intermediate_size),
        device=a.device,
        dtype=torch.bfloat16,
    )
    for expert in range(experts):
        token, slot = torch.where(topk_ids == expert)
        if token.numel() == 0:
            continue
        b_dequant = (
            raw_b[expert].float().view(2 * intermediate_size // 128, 128, kb, 128)
            * inputs["scale_b"][expert, :, None, :, None]
        ).view(2 * intermediate_size, hidden)
        projected = a_dequant[token] @ b_dequant.t()
        gate, up = projected.split(intermediate_size, dim=-1)
        out[token, slot] = (torch.nn.functional.silu(gate) * up).to(torch.bfloat16)
    return out


def make_launch_args(inputs, output):
    a = inputs["a"]
    tokens, hidden = a.shape
    experts = inputs["weight"].shape[0]
    topk, intermediate = output.shape[1:]
    validate_parameters(tokens, intermediate, hidden, topk, experts)
    expected = {
        "a": ((tokens, hidden), torch.float8_e4m3fn),
        "weight": ((experts, 2 * intermediate, hidden), torch.float8_e4m3fn),
        "scale_a": ((hidden // 128, tokens), torch.float32),
        "scale_b": ((experts, 2 * intermediate // 128, hidden // 128), torch.float32),
    }
    for name, (shape, dtype) in expected.items():
        value = inputs[name]
        if (
            tuple(value.shape) != shape
            or value.dtype != dtype
            or not value.is_contiguous()
        ):
            raise ValueError(f"{name} requires contiguous {dtype}, shape={shape}")
        if value.device != a.device or not value.is_cuda:
            raise ValueError(f"{name} must be on the same GPU as A")
    for name in ("sorted_ids", "expert_ids", "num_valid_ids"):
        value = inputs[name]
        if (
            value.dtype != torch.int32
            or not value.is_contiguous()
            or value.device != a.device
        ):
            raise ValueError(f"{name} must be contiguous int32 on the input GPU")
        if value.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional")
    if inputs["num_valid_ids"].numel() < 1:
        raise ValueError("num_valid_ids must contain the padded route count")
    blocks = inputs["expert_ids"].numel()
    if not blocks or inputs["sorted_ids"].numel() != blocks * SORT_BLOCK_M:
        raise ValueError("sorted_ids must contain 256 entries per expert block")
    if inputs["sorted_ids"].numel() * 4 >= BUFFER_LIMIT:
        raise ValueError("sorted_ids must fit 31-bit byte offsets")
    if (
        output.shape[0] != tokens
        or output.dtype != torch.bfloat16
        or not output.is_contiguous()
        or output.device != a.device
    ):
        raise ValueError(
            "output must be contiguous BF16 [tokens, topk, I] on the input GPU"
        )
    return (
        a.view(torch.int8),
        inputs["weight"].view(torch.int8),
        inputs["scale_a"],
        inputs["scale_b"],
        inputs["sorted_ids"],
        inputs["expert_ids"],
        inputs["num_valid_ids"],
        output.view(-1),
        tokens,
        blocks,
        torch.cuda.current_stream(),
    )


def run_case(
    tokens=8192,
    intermediate_size=256,
    hidden_size=6144,
    topk=8,
    num_experts=384,
    *,
    routing="balanced",
    data_case="random",
    xcd_swizzle=True,
    group_size_m=1,
    k_rotate="auto",
    benchmark=False,
    compare_jit=False,
    warmup=5,
    iterations=20,
    data_clones=4,
):
    import pyhip

    if compare_jit and not benchmark:
        raise ValueError("compare_jit requires benchmark=True")
    if benchmark and (iterations <= 0 or data_clones <= 0 or warmup < 0):
        raise ValueError("require iterations/clones > 0 and warmup >= 0")
    inputs = prepare_inputs(
        tokens,
        intermediate_size,
        hidden_size,
        topk,
        num_experts,
        routing=routing,
        data_case=data_case,
    )
    output = torch.empty(
        (tokens, topk, intermediate_size), device="cuda", dtype=torch.bfloat16
    )
    args = make_launch_args(inputs, output)
    if k_rotate == "auto":
        # Measured: rotation wins with <= 2 blocks per expert, costs a little beyond.
        k_rotate = inputs["expert_ids"].numel() <= 2 * num_experts
    launcher = compile_fp8_stage1_giu1(
        intermediate_size,
        hidden_size,
        topk,
        num_experts,
        xcd_swizzle=xcd_swizzle,
        group_size_m=group_size_m,
        k_rotate=k_rotate in (True, "on"),
    )
    kernel = flyc.compile[{"opt_level": 2}](launcher, *args)
    reference = stage1_reference(inputs, intermediate_size)
    previous = None
    diff_threshold = 0.00001
    for repeat in range(3):
        output.fill_(float("nan"))
        kernel(*args)
        torch.cuda.synchronize()
        finite = bool(torch.isfinite(output).all() & torch.isfinite(reference).all())
        allclose = finite and torch.allclose(output, reference, rtol=0.02, atol=0.01)
        diff = pyhip.calc_diff(output.float(), reference) if finite else float("inf")
        diff_ok = finite and diff <= diff_threshold
        max_abs = (output.float() - reference.float()).abs().max().item()
        print(
            f"accuracy: repeat={repeat + 1} finite={finite} allclose={allclose} "
            f"calc_diff={diff:.9g} diff_thr={diff_threshold:.9g} "
            f"diff_ok={diff_ok} max_abs={max_abs:.9g}",
            flush=True,
        )
        if not finite:
            raise AssertionError("non-finite or unwritten stage1 output")
        # A small aggregate error does not replace the per-element check.
        pyhip.calc_diff(output.float(), reference, diff_thr=0.00001)
        torch.testing.assert_close(output, reference, rtol=0.05, atol=0.01)
        if previous is not None:
            torch.testing.assert_close(output, previous, rtol=0, atol=0)
        previous = output.clone()
    row = {
        "tokens": tokens,
        "intermediate_size": intermediate_size,
        "hidden_size": hidden_size,
        "topk": topk,
        "num_experts": num_experts,
        "routing": routing,
        "data_case": data_case,
        "xcd_swizzle": xcd_swizzle,
        "group_size_m": group_size_m,
        "k_rotate": k_rotate in (True, "on"),
        "valid_routes": tokens * topk,
        "padded_routes": inputs["sorted_ids"].numel(),
        "correct": True,
        "repeatable": True,
        "finite": finite,
        "allclose": allclose,
        "max_abs": max_abs,
        "diff": diff,
        "diff_thr": diff_threshold,
        "diff_ok": diff_ok,
        "weight_layout": "aiter.shuffle_weight(16,16), GGUU",
        "scale_layout": "A[KB,tokens], B[E,2I/128,KB], FP32",
        "output": "BF16 SiLU(gate)*up, no clamp, no routing weight",
    }
    if benchmark:
        # Setup, shuffle, reference, and compilation stay outside timing.
        arg_sets = [args]
        for _ in range(1, data_clones):
            arg_sets.append(
                tuple(
                    value.clone() if isinstance(value, torch.Tensor) else value
                    for value in args
                )
            )
        kernels = {"flydsl": kernel}
        if compare_jit:
            from pyhip.contrib.moe_gemm_8wave import moe_gemm_8wave_g1u1

            def launch_jit(*launch_args):
                a, weight, scale_a, scale_b, ids, experts, valid, out = launch_args[:8]
                token_count, block_count = launch_args[8:10]
                grid = block_count * (intermediate_size // BLOCK_N)
                moe_gemm_8wave_g1u1(
                    [grid],
                    [512],
                    False,
                    "fp8",
                    SORT_BLOCK_M,
                    2 * BLOCK_N,
                    num_experts,
                    2 * intermediate_size,
                    hidden_size,
                    True,
                    True,
                    topk,
                    ids.data_ptr(),
                    None,
                    experts.data_ptr(),
                    valid.data_ptr(),
                    weight.data_ptr(),
                    scale_b.data_ptr(),
                    a.data_ptr(),
                    scale_a.data_ptr(),
                    out.data_ptr(),
                    token_count,
                    grid,
                )

            kernels["moe_jit"] = launch_jit
            previous = None
            for repeat in range(3):
                output.fill_(float("nan"))
                launch_jit(*args)
                torch.cuda.synchronize()
                if not bool(torch.isfinite(output).all()):
                    raise AssertionError("non-finite or unwritten MoE JIT output")
                jit_diff = pyhip.calc_diff(output.float(), reference, diff_thr=0.00001)
                torch.testing.assert_close(output, reference, rtol=0.05, atol=0.01)
                if previous is not None:
                    torch.testing.assert_close(output, previous, rtol=0, atol=0)
                previous = output.clone()
                print(f"moe_jit accuracy: repeat={repeat + 1} calc_diff={jit_diff:.9g}", flush=True)
        for i in range(warmup):
            for launch in kernels.values():
                launch(*arg_sets[i % data_clones])
        torch.cuda.synchronize()
        effective_flops = 4 * tokens * topk * intermediate_size * hidden_size
        padded_flops = (
            4 * inputs["sorted_ids"].numel() * intermediate_size * hidden_size
        )
        byte_count = sum(
            value.numel() * value.element_size()
            for value in args
            if isinstance(value, torch.Tensor)
        )
        samples_by_kernel = {name: [] for name in kernels}
        for i in range(iterations):
            launches = list(kernels.items())
            if i % 2:
                launches.reverse()
            for name, launch in launches:
                with pyhip.cudaPerf(
                    effective_flops, byte_count, name=f"fp8_stage1_{name}", verbose=0
                ) as perf:
                    launch(*arg_sets[(warmup + i) % data_clones])
                samples_by_kernel[name].append(perf.dt_ms * 1000)
        samples = samples_by_kernel["flydsl"]
        best_us = min(samples)
        # Both rates use the same best latency and count gate/up GEMM FLOPs.
        valid_tflops = effective_flops / best_us / 1e6
        row.update(
            {
                "best_us": best_us,
                "median_us": statistics.median(samples),
                "valid_tflops": valid_tflops,
                "effective_tflops": valid_tflops,  # Retain the existing JSON key.
                "padded_tflops": padded_flops / best_us / 1e6,
                "samples_us": samples,
                "clones": data_clones,
            }
        )
        if compare_jit:
            jit_samples = samples_by_kernel["moe_jit"]
            jit_median = statistics.median(jit_samples)
            row["moe_jit"] = {
                "correct": True,
                "repeatable": True,
                "diff": jit_diff,
                "best_us": min(jit_samples),
                "median_us": jit_median,
                "samples_us": jit_samples,
            }
            row["speedup_vs_jit"] = jit_median / row["median_us"]
    print("STAGE1_RESULT " + json.dumps(row), flush=True)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--small-accuracy", action="store_true")
    modes.add_argument("--accuracy", action="store_true")
    modes.add_argument("--accuracy-matrix", action="store_true")
    modes.add_argument("--benchmark", action="store_true")
    modes.add_argument("--layout-check", action="store_true")
    parser.add_argument("--tokens", type=int, default=8192)
    parser.add_argument("--intermediate-size", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=6144)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--num-experts", "--experts", type=int, default=384)
    # routing 控制 token 选择哪些专家；每个 token 的 topk 个专家互不重复。
    # balanced：循环分配后打乱 token 顺序/专家编号，各专家路由数最多差 1。
    # random：每个 token 独立生成随机专家分数再取 topk，负载仅在期望上均衡。
    # skewed：随机选定同一组 topk 专家，全部 token 都路由到这组，负载极端集中。
    # 这些是合成测试路由，不代表真实模型的 gate 分布。各专家按 256 行补齐，
    # 因此 routing 会影响 padding、WG 数量与权重复用，而不改变计算公式。
    parser.add_argument(
        "--routing", choices=("balanced", "random", "skewed"), default="balanced"
    )
    # data_case 控制 A/权重/scale 的数值，可与 routing 独立组合。
    # 除 random/last_k 外，下列场景在有限输入、无 bias 时均应输出全零。
    # last_k 仍执行完整 K-loop，仅最后一个 K128 分块有贡献，检查尾部/FIFO drain。
    # zero_gate/zero_up 清零的是投影权重，不是路由分数；性能测试通常用 random。
    parser.add_argument(
        "--data-case",
        choices=(
            "random",  # 随机 A/权重及非零 FP32 scale：常规精度/性能测试。
            "zero_a",  # FP8 A 数据清零，保留 scale。
            "zero_a_scale",  # 仅 A-scale 清零，A 数据保留。
            "zero_b_scale",  # 仅 B-scale 清零，权重数据保留。
            "last_k",  # A-scale 除最后一个 K128 分块外全部置零。
            "zero_gate",  # 所有专家前 I 行 gate 权重清零；SiLU(0)=0。
            "zero_up",  # 所有专家后 I 行 up 权重清零。
        ),
        default="random",
    )
    parser.add_argument("--no-xcd-swizzle", action="store_true")
    parser.add_argument("--group-size-m", type=int, default=1)
    # Per-expert K start rotation; auto enables it for <= 2 blocks per expert.
    parser.add_argument("--k-rotate", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--data-clones", type=int, default=4)
    parser.add_argument("--compare-jit", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.compare_jit and not args.benchmark:
        parser.error("--compare-jit requires --benchmark")
    check_native_weight_layout()
    check_a_slice_layout()
    if args.layout_check:
        return
    if "gfx950" not in torch.cuda.get_device_properties().gcnArchName:
        raise RuntimeError("FP8 MFMA requires gfx950")
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    common = {
        "xcd_swizzle": not args.no_xcd_swizzle,
        "group_size_m": args.group_size_m,
        "k_rotate": args.k_rotate,
    }
    if args.accuracy_matrix:
        cases = [
            (96, 128, 256, 2, 3, "balanced", "random"),
            (257, 384, 512, 3, 5, "random", "random"),
            (513, 256, 768, 2, 7, "skewed", "random"),
            (97, 128, 512, 2, 5, "skewed", "zero_a"),
            (97, 128, 512, 2, 5, "random", "zero_a_scale"),
            (97, 128, 512, 2, 5, "random", "zero_b_scale"),
            (257, 256, 512, 3, 5, "balanced", "last_k"),
            (97, 128, 256, 2, 3, "balanced", "zero_gate"),
            (97, 128, 256, 2, 3, "balanced", "zero_up"),
        ]
        rows = [
            run_case(*case[:5], routing=case[5], data_case=case[6], **common)
            for case in cases
        ]
    elif args.accuracy or args.benchmark:
        rows = [
            run_case(
                args.tokens,
                args.intermediate_size,
                args.hidden_size,
                args.topk,
                args.num_experts,
                routing=args.routing,
                data_case=args.data_case,
                benchmark=args.benchmark,
                compare_jit=args.compare_jit,
                warmup=args.warmup,
                iterations=args.iterations,
                data_clones=args.data_clones,
                **common,
            )
        ]
    else:
        rows = [run_case(96, 128, 256, 2, 3, **common)]
    print(
        "\n| Tokens | I | K | Topk | Experts | Case | Correct | Best us | "
        "Valid TFLOPS | Padded TFLOPS |"
    )
    print("|---:|---:|---:|---:|---:|---|:---:|---:|---:|---:|")
    for row in rows:
        best_us = f"{row['best_us']:.3f}" if "best_us" in row else "-"
        valid_tflops = f"{row['valid_tflops']:.2f}" if "valid_tflops" in row else "-"
        padded_tflops = f"{row['padded_tflops']:.2f}" if "padded_tflops" in row else "-"
        print(
            f"| {row['tokens']} | {row['intermediate_size']} | {row['hidden_size']} | "
            f"{row['topk']} | {row['num_experts']} | {row['data_case']} | "
            f"{row['correct']} | {best_us} | {valid_tflops} | {padded_tflops} |"
        )


if __name__ == "__main__":
    main()
