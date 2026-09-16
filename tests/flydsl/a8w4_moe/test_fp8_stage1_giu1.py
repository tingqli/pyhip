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
TL0/TR0/BL0/BR0, then TL1/TR1/BL1/BR1. Left/right mean gate/up here.
Each phase interleaves old-partial FP32 FMA with independent new MFMA.
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
from flydsl._mlir.dialects import llvm as _llvm, vector

SORT_BLOCK_M = 256
BLOCK_M = 128
BLOCK_N = 128
BLOCK_K = 128
TOKEN_MASK = 0xFFFFFF
BUFFER_LIMIT = 0x7FFFFFFF


def validate_parameters(tokens, intermediate_size, hidden_size, topk, num_experts):
    if not 0 < tokens <= TOKEN_MASK:
        raise ValueError(f"tokens must be in [1, {TOKEN_MASK}]")
    if intermediate_size <= 0 or intermediate_size % 128:
        raise ValueError("intermediate_size must be a positive multiple of 128")
    if hidden_size < 256 or hidden_size % 256:
        raise ValueError("hidden_size must be a multiple of 256 and at least 256")
    if not 0 < topk < 256 or num_experts < topk:
        raise ValueError("require 0 < topk < 256 and num_experts >= topk")
    extents = (
        tokens * hidden_size,
        tokens * (hidden_size // 128) * 4,
        tokens * topk * intermediate_size * 2,
        2 * intermediate_size * hidden_size,
    )
    if max(extents) >= BUFFER_LIMIT:
        raise ValueError(
            "A/scales/output and each expert's weight must fit 31-bit byte offsets"
        )


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


def encode_waitcnt_950(vmcnt=63, expcnt=7, lgkmcnt=63):
    return (vmcnt & 15) | (expcnt << 4) | (lgkmcnt << 8) | ((vmcnt >> 4) << 14)


def compile_fp8_stage1_giu1(
    intermediate_size,
    hidden_size,
    topk,
    num_experts,
    *,
    xcd_swizzle=True,
    group_size_m=4,
    prefetch_store_ids=True,
    b_lds_padding=0,
    packed_b_carry=True,
):
    """Build a half-M, single-FIFO kernel for AIter-native FP8 weight bytes.

    Routing must contain complete 256-row expert blocks, each assigned to a
    valid local expert. Padding uses an invalid token/slot; each real
    (token, slot) must appear exactly once. No EP expert masking or bias.
    Native B slabs are read coalesced into LDS without padding by default.
    B's loop carry uses packed DWORDs, not individual FP8 bytes. Output routing IDs
    are prefetched by default; disable prefetch_store_ids for lower VGPR use.
    """
    validate_parameters(1, intermediate_size, hidden_size, topk, num_experts)
    if group_size_m <= 0:
        raise ValueError("group_size_m must be positive")
    if b_lds_padding not in (0, 64):
        raise ValueError("b_lds_padding must be 0 or 64 bytes per N16 group")
    kb_count = hidden_size // BLOCK_K
    n_tiles = intermediate_size // BLOCK_N
    a_group8 = 8 * BLOCK_K + 16
    a_group16 = 2 * a_group8 + 32
    lds_operand_elems = (BLOCK_M // 16) * a_group16
    b_group16 = 16 * BLOCK_K + b_lds_padding
    b_lds_elems = (BLOCK_N // 16) * b_group16

    def _get_pids(pid, m_tiles, grid_size):
        if const_expr(xcd_swizzle):
            per_xcd = (grid_size + 7) // 8
            tall = grid_size % 8
            tall = (tall == 0).select(8, tall)
            xcd = pid % 8
            local = pid // 8
            if xcd < tall:
                pid = xcd * per_xcd + local
            else:
                pid = tall * per_xcd + (xcd - tall) * (per_xcd - 1) + local
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
        a_t0: fx.Array[Float8E4M3FN, lds_operand_elems, 16]
        a_b0: fx.Array[Float8E4M3FN, lds_operand_elems, 16]
        a_t1: fx.Array[Float8E4M3FN, lds_operand_elems, 16]
        a_b1: fx.Array[Float8E4M3FN, lds_operand_elems, 16]
        b_l0: fx.Array[Float8E4M3FN, b_lds_elems, 16]
        b_l1: fx.Array[Float8E4M3FN, b_lds_elems, 16]
        b_r0: fx.Array[Float8E4M3FN, b_lds_elems, 16]
        b_r1: fx.Array[Float8E4M3FN, b_lds_elems, 16]
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
        block_id, n_tile = get_pids(fx.block_idx.x, num_expert_blocks, fx.grid_dim.x)
        block_id = fx.Int32(block_id)
        n_tile = fx.Int32(n_tile)
        sorted_base = block_id * SORT_BLOCK_M
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
        a_rsrc = _buffer_resource(
            arg_a, arith._to_raw(fx.Int32(num_tokens * hidden_size))
        )
        b_rsrc = _buffer_resource(
            arg_b,
            2 * intermediate_size * hidden_size,
            fx.Int64(expert) * (2 * intermediate_size * hidden_size),
        )
        scale_a_rsrc = _buffer_resource(
            arg_scale_a, arith._to_raw(fx.Int32(num_tokens * kb_count * 4))
        )
        scale_b_rsrc = _buffer_resource(
            arg_scale_b,
            2 * n_tiles * kb_count * 4,
            fx.Int64(expert) * (2 * n_tiles * kb_count * 4),
        )
        lds = fx.SharedAllocator().allocate(LDS).peek()
        read_layout = fx.make_layout(
            ((2, BLOCK_M // 16, 8), (32, BLOCK_K // 32)),
            ((a_group8, a_group16, BLOCK_K), (1, 32)),
        )
        a_top = [fx.make_view(p, read_layout) for p in (lds.a_t0.ptr, lds.a_t1.ptr)]
        a_bottom = [fx.make_view(p, read_layout) for p in (lds.a_b0.ptr, lds.a_b1.ptr)]
        # Keep native N16/K16 bytes in LDS. Move the existing logical-row
        # permutation to the consumer so each producer wave reads 1024B
        # contiguously instead of eight scattered rows from eight N16 groups.
        b_read_layout = fx.make_layout(
            ((2, BLOCK_N // 16, 8), ((16, 2), BLOCK_K // 32)),
            ((16, 32, b_group16), ((1, 256), 512)),
        )
        b_gate = [fx.make_view(p, b_read_layout) for p in (lds.b_l0.ptr, lds.b_l1.ptr)]
        b_up = [fx.make_view(p, b_read_layout) for p in (lds.b_r0.ptr, lds.b_r1.ptr)]

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

        def a_part(view, part):
            return fx.flat_divide(view, (64, BLOCK_K))[None, None, part, 0]

        frag_a = thread_mma.make_fragment_B(a_part(a_top[0], 0))
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
                    T.i32, sorted_rsrc, fx.Int32((sorted_base + row) * 4), fx.Int32(0)
                )
            )

        if const_expr(prefetch_store_ids):
            store_ids = [
                sorted_id(bottom * 128 + repeat * 32 + wave_m * 16 + lane_id % 16)
                for bottom in range_constexpr(2)
                for repeat in range_constexpr(4)
            ]

        # Exactly the GEMM producer's row permutation. Two DWORDx4 copies
        # cover one A128xK128 operand; replace its row index with token ID.
        a_voffsets = []
        for bottom in range_constexpr(2):
            offsets = []
            for chunk in range_constexpr(2):
                row = bottom * 128 + (tid // 8 % 8) * 16 + tid // 64 + chunk * 8
                fused = sorted_id(row)
                token = fused & TOKEN_MASK
                slot = (fused >> 24) & 255
                valid = (token < num_tokens) & (slot < topk)
                offset = token * hidden_size + fx.Int32((tid % 8) * 16)
                offsets.append(valid.select(offset, fx.Int32(BUFFER_LIMIT)))
            a_voffsets.append(offsets)

        scale_id = sorted_id(tid % SORT_BLOCK_M)
        scale_token = scale_id & TOKEN_MASK
        scale_valid = (scale_token < num_tokens) & (((scale_id >> 24) & 255) < topk)
        scale_a_voffset = scale_valid.select(scale_token * 4, fx.Int32(BUFFER_LIMIT))
        wave_dma_base = rocdl.readfirstlane(
            T.i32,
            arith._to_raw(
                fx.Int32((wave_id % 2) * a_group8 + (wave_id // 2) * a_group16)
            ),
        )
        scale_dma_base = rocdl.readfirstlane(T.i32, arith._to_raw(fx.Int32(tid * 4)))

        def operand_dma_ptrs(views):
            return [
                [
                    _lds_byte_ptr(
                        fx.get_iter(view), wave_dma_base + chunk * 4 * a_group16
                    )
                    for chunk in range_constexpr(2)
                ]
                for view in views
            ]

        at_dma = operand_dma_ptrs(a_top)
        ab_dma = operand_dma_ptrs(a_bottom)
        b_wave_base = rocdl.readfirstlane(
            T.i32, arith._to_raw(fx.Int32(wave_id * b_group16))
        )

        def b_dma_ptrs(views):
            return [
                [
                    _lds_byte_ptr(fx.get_iter(view), b_wave_base + chunk * 1024)
                    for chunk in range_constexpr(2)
                ]
                for view in views
            ]

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
        b_gate_base = n_tile * (BLOCK_N * hidden_size)
        b_up_base = b_gate_base + intermediate_size * hidden_size

        def ac_a(buf, ki, bottom=0):
            for chunk in range_constexpr(2):
                rocdl.raw_ptr_buffer_load_lds(
                    a_rsrc,
                    ab_dma[buf][chunk] if const_expr(bottom) else at_dma[buf][chunk],
                    fx.Int32(16),
                    a_voffsets[bottom][chunk],
                    fx.Int32(ki * BLOCK_K),
                    fx.Int32(0),
                    fx.Int32(0),
                )

        def ac_b(buf, ki, is_up=0):
            for chunk in range_constexpr(2):
                rocdl.raw_ptr_buffer_load_lds(
                    b_rsrc,
                    br_dma[buf][chunk] if const_expr(is_up) else bl_dma[buf][chunk],
                    fx.Int32(16),
                    b_voffset,
                    fx.Int32(
                        (b_up_base if const_expr(is_up) else b_gate_base)
                        + ki * (BLOCK_K * 16)
                        + chunk * 1024
                    ),
                    fx.Int32(0),
                    fx.Int32(0),
                )

        def ac_scale_a(buf, ki):
            rocdl.raw_ptr_buffer_load_lds(
                scale_a_rsrc,
                scale_dma[buf],
                fx.Int32(4),
                scale_a_voffset,
                fx.Int32(ki * num_tokens * 4),
                fx.Int32(0),
                fx.Int32(0),
            )

        def rd_a(buf, part, bottom=0):
            view = a_bottom[buf] if const_expr(bottom) else a_top[buf]
            fx.copy(copy_atom, copy_a.partition_S(a_part(view, part)), dest_a)

        def rd_bl(buf):
            fx.copy(copy_atom, copy_b.partition_S(b_gate[buf]), dest_bl)

        def rd_br(buf):
            fx.copy(copy_atom, copy_b.partition_S(b_up[buf]), dest_br)

        scale_copy = fx.make_copy_atom(fx.UniversalCopy32b(), Float32)

        def rd_scale_a(buf, bottom, part):
            values = []
            for m0 in range_constexpr(2):
                index = wave_m * 256 + bottom * 128 + wave_m * 16 + lane_id % 16
                index = index + (part * 2 + m0) * 32
                ptr = lds.scale_a0.ptr if const_expr(buf == 0) else lds.scale_a1.ptr
                src = fx.make_view(
                    fx.add_offset(ptr, fx.make_int_tuple(index)), fx.make_layout(1, 1)
                )
                frag = fx.make_fragment_like(src)
                fx.copy(scale_copy, src, frag)
                values.append(Vec(frag.load())[0])
            return Vec.from_elements(values, Float32)

        def rd_scale_b(ki):
            offset = fx.Int32((n_tile * kb_count + ki) * 4)
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

        def compute(c_frag, b_frag, old_scale_a, old_scale_b, old_part):
            # Old partial and new MFMA results are independent. Preserve the
            # GEMM's early-clobber boundary and scalar (not packed) FP32 FMA.
            result_type = ir.Type.parse(
                "!llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, "
                "vector<4xf32>, vector<4xf32>)>"
            )
            asm = (
                "v_fmac_f32 $0, $10, $18\n"
                "v_fmac_f32 $1, $11, $18\n"
                "v_fmac_f32 $2, $12, $18\n"
                "v_fmac_f32 $3, $13, $18\n"
                "v_mfma_f32_16x16x128_f8f6f4 $8, $27, $29, 0\n"
                "v_fmac_f32 $4, $14, $18\n"
                "v_fmac_f32 $5, $15, $18\n"
                "v_fmac_f32 $6, $16, $18\n"
                "v_fmac_f32 $7, $17, $18\n"
                "v_mfma_f32_16x16x128_f8f6f4 $9, $28, $29, 0\n"
            )
            for m0 in range_constexpr(2):
                scale = Vec(old_scale_a)[m0] * old_scale_b
                cs0 = c_frag[None, 0, old_part * 2 + m0]
                cs1 = c_frag[None, 1, old_part * 2 + m0]
                p0 = Vec(frag_p[None, 0, m0].load())
                p1 = Vec(frag_p[None, 1, m0].load())
                c0, c1 = Vec(cs0.load()), Vec(cs1.load())
                b0 = vector.bitcast(
                    T.vec(8, T.i32), b_frag[None, 0, 0].load().ir_value()
                )
                b1 = vector.bitcast(
                    T.vec(8, T.i32), b_frag[None, 1, 0].load().ir_value()
                )
                a = vector.bitcast(
                    T.vec(8, T.i32), frag_a[None, m0, 0].load().ir_value()
                )
                args = [arith._to_raw(p0[i]) for i in range_constexpr(4)]
                args += [arith._to_raw(p1[i]) for i in range_constexpr(4)]
                args += [arith._to_raw(scale)]
                args += [arith._to_raw(c0[i]) for i in range_constexpr(4)]
                args += [arith._to_raw(c1[i]) for i in range_constexpr(4)]
                args += [arith._to_raw(b0), arith._to_raw(b1), arith._to_raw(a)]
                result = _llvm.inline_asm(
                    result_type,
                    args,
                    asm,
                    "=&v,=&v,=&v,=&v,=&v,=&v,=&v,=&v,=&v,=&v,"
                    "v,v,v,v,v,v,v,v,v,0,1,2,3,4,5,6,7,v,v,v",
                    has_side_effects=True,
                )
                cs0.store(
                    Vec.from_elements(
                        [
                            fx.Float32(_llvm.extractvalue(T.f32, result, [i]))
                            for i in range_constexpr(4)
                        ],
                        Float32,
                    )
                )
                cs1.store(
                    Vec.from_elements(
                        [
                            fx.Float32(_llvm.extractvalue(T.f32, result, [i + 4]))
                            for i in range_constexpr(4)
                        ],
                        Float32,
                    )
                )
                frag_p[None, 0, m0].store(
                    _llvm.extractvalue(T.vec(4, T.f32), result, [8])
                )
                frag_p[None, 1, m0].store(
                    _llvm.extractvalue(T.vec(4, T.f32), result, [9])
                )

        def begin_phase():
            rocdl.sched_barrier(0)
            rocdl.s_barrier()
            rocdl.sched_barrier(0)
            rocdl.s_waitcnt(encode_waitcnt_950(lgkmcnt=0))
            rocdl.sched_barrier(0)

        def end_phase():
            rocdl.sched_barrier(0)
            rocdl.s_barrier()
            rocdl.sched_barrier(0)

        def load_b_carry():
            value = frag_bl.load()
            if const_expr(packed_b_carry):
                value = Vec(value).bitcast(fx.Int32)
                # Keep the DWORD boundary opaque to byte-vector legalization.
                # Tied operands make this an identity with no machine instruction.
                value = Vec(
                    _llvm.inline_asm(
                        T.vec(16, T.i32),
                        [value.ir_value()],
                        "",
                        "=v,0",
                        has_side_effects=True,
                    )
                )
            return value

        # Drain routing VMEM before the pipeline's counted DMA batches start.
        rocdl.sched_barrier(0)
        rocdl.s_waitcnt(encode_waitcnt_950(vmcnt=0))
        rocdl.sched_barrier(0)
        ac_scale_a(0, fx.Int32(0))
        rocdl.sched_barrier(0)
        ac_b(0, 0)
        rocdl.sched_barrier(0)
        ac_a(0, 0)
        rocdl.sched_barrier(0)
        ac_b(0, 0, 1)
        rocdl.sched_barrier(0)
        ac_a(0, 0, 1)
        rocdl.sched_barrier(0)
        # The complementary barrier follows the final FIFO drain.
        if wave_id >= 4:
            rocdl.s_barrier()
        c_tl.fill(0)
        c_tr.fill(0)
        c_bl.fill(0)
        c_br.fill(0)
        rocdl.s_waitcnt(encode_waitcnt_950(vmcnt=4))
        rocdl.s_barrier()
        rocdl.sched_barrier(0)
        ac_scale_a(1, fx.Int32(1))
        rocdl.sched_barrier(0)
        ac_a(1, 1)
        rocdl.sched_barrier(0)
        ac_b(1, 1)
        rocdl.sched_barrier(0)
        ac_b(1, 1, 1)
        rocdl.sched_barrier(0)
        rocdl.s_waitcnt(encode_waitcnt_950(vmcnt=7))
        rocdl.s_barrier()
        rocdl.sched_barrier(0)
        rd_bl(0)
        frag_p.fill(0)
        initial = [
            c_tl.load(),
            c_tr.load(),
            c_bl.load(),
            c_br.load(),
            frag_p.load(),
            Vec.filled(2, 0.0, Float32),
            fx.Float32(0),
            load_b_carry(),
        ]
        for k_index, states in range(0, kb_count, 2, init=initial):
            c_tl.store(states[0])
            c_tr.store(states[1])
            c_bl.store(states[2])
            c_br.store(states[3])
            frag_p.store(states[4])
            old_a = Vec(states[5])
            old_b = fx.Float32(states[6])
            if const_expr(packed_b_carry):
                frag_bl.store(Vec(states[7]).bitcast(Float8E4M3FN))
            else:
                frag_bl.store(states[7])
            kiter = fx.Int32(k_index)
            for tile in range_constexpr(2):
                ki = kiter + tile
                for part in range_constexpr(2):
                    rd_a(tile, part)
                    top_scale = rd_scale_a(tile, 0, part)
                    if const_expr(part == 0):
                        b_scale = rd_scale_b(ki)
                        ac_a(1 - tile, ki + 1, 1)
                    begin_phase()
                    # New TL; consume BR from the preceding part/K tile.
                    compute(c_br, frag_bl, old_a, old_b, 1 - part)
                    end_phase()
                    if const_expr(part == 0):
                        rd_br(tile)
                    else:
                        # Both parts have read A_t in both staggered groups.
                        ac_a(tile, ki + 2)
                    begin_phase()
                    compute(c_tl, frag_br, top_scale, b_scale[0], part)
                    end_phase()
                    rd_a(tile, part, 1)
                    bottom_scale = rd_scale_a(tile, 1, part)
                    if const_expr(part == 0):
                        # B registers persist through part1; LDS can be reused.
                        ac_b(tile, ki + 2)
                    begin_phase()
                    compute(c_tr, frag_bl, top_scale, b_scale[1], part)
                    end_phase()
                    if const_expr(part == 0):
                        ac_b(tile, ki + 2, 1)
                    else:
                        ac_scale_a(tile, ki + 2)
                        rocdl.s_waitcnt(encode_waitcnt_950(vmcnt=7))
                        rd_bl(1 - tile)
                    begin_phase()
                    compute(c_bl, frag_br, bottom_scale, b_scale[0], part)
                    end_phase()
                    old_a, old_b = bottom_scale, b_scale[1]
            result = yield [
                c_tl.load(),
                c_tr.load(),
                c_bl.load(),
                c_br.load(),
                frag_p.load(),
                old_a,
                old_b,
                load_b_carry(),
            ]

        c_tl.store(result[0])
        c_tr.store(result[1])
        c_bl.store(result[2])
        c_br.store(result[3])
        frag_p.store(result[4])
        for m0 in range_constexpr(2):
            scale = Vec(result[5])[m0] * fx.Float32(result[6])
            for n0 in range_constexpr(2):
                cs = c_br[None, n0, 2 + m0]
                cs.store(
                    fx.fma(
                        frag_p[None, n0, m0].load(),
                        Vec.filled(4, scale, Float32),
                        cs.load(),
                    )
                )
        if wave_id < 4:
            rocdl.s_barrier()

        valid_rsrc = _buffer_resource(arg_num_valid_ids, 4)
        valid_rows = fx.Int32(
            rocdl.raw_ptr_buffer_load(T.i32, valid_rsrc, fx.Int32(0), fx.Int32(0))
        )
        c_rsrc = _buffer_resource(
            arg_c, arith._to_raw(fx.Int32(num_tokens * topk * intermediate_size * 2))
        )
        pair_type = ir.Type.parse("!llvm.struct<(i32, i32)>")

        def silu_mul(gate, up):
            exponent = rocdl.exp2(T.f32, arith._to_raw(gate * -1.4426950408889634))
            inverse = rocdl.rcp(T.f32, 1.0 + exponent)
            return (gate * inverse) * up

        def store_gate_up(gates, ups, bottom):
            for row_repeat in range_constexpr(4):
                row = bottom * 128 + row_repeat * 32 + wave_m * 16 + lane_id % 16
                fused = (
                    store_ids[bottom * 4 + row_repeat]
                    if const_expr(prefetch_store_ids)
                    else sorted_id(row)
                )
                token, slot = fused & TOKEN_MASK, (fused >> 24) & 255
                valid = (
                    (token < num_tokens)
                    & (slot < topk)
                    & (sorted_base + row < valid_rows)
                )
                g0, g1 = Vec(gates[None, 0, row_repeat].load()), Vec(
                    gates[None, 1, row_repeat].load()
                )
                u0, u1 = Vec(ups[None, 0, row_repeat].load()), Vec(
                    ups[None, 1, row_repeat].load()
                )
                v0 = [silu_mul(g0[i], u0[i]) for i in range_constexpr(4)]
                v1 = [silu_mul(g1[i], u1[i]) for i in range_constexpr(4)]
                p0 = rocdl.cvt_pk_bf16_f32(v0[0], v0[1])
                p1 = rocdl.cvt_pk_bf16_f32(v0[2], v0[3])
                p2 = rocdl.cvt_pk_bf16_f32(v1[0], v1[1])
                p3 = rocdl.cvt_pk_bf16_f32(v1[2], v1[3])
                swap0 = rocdl.permlane16_swap(
                    pair_type, arith._to_raw(p0), arith._to_raw(p2), False, False
                )
                swap1 = rocdl.permlane16_swap(
                    pair_type, arith._to_raw(p1), arith._to_raw(p3), False, False
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
                address = valid.select(address, fx.Int32(BUFFER_LIMIT))
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
        ).launch(
            grid=(num_expert_blocks * n_tiles, 1, 1), block=(512, 1, 1), stream=stream
        )

    return launch_stage1


def check_native_weight_layout(b_lds_padding=0):
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
                        (tid // 64) * (16 * 128 + b_lds_padding)
                        + (tid % 64) * 16
                        + chunk * 1024
                    )
                    local_row, local_col = row % 128, col % 128
                    consumer_address = (
                        (local_row // 16) * (16 * 128 + b_lds_padding)
                        + (local_col // 16) * 256
                        + (local_row % 16) * 16
                    )
                    torch.testing.assert_close(
                        lds_address, consumer_address, rtol=0, atol=0
                    )
    print("layout: AIter shuffle_weight(16,16) native byte mapping PASS")
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
        * 0.2
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
    group_size_m=4,
    benchmark=False,
    warmup=5,
    iterations=20,
    data_clones=4,
):
    import pyhip

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
    launcher = compile_fp8_stage1_giu1(
        intermediate_size,
        hidden_size,
        topk,
        num_experts,
        xcd_swizzle=xcd_swizzle,
        group_size_m=group_size_m,
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
        for i in range(warmup):
            kernel(*arg_sets[i % data_clones])
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
        samples = []
        for i in range(iterations):
            with pyhip.cudaPerf(
                effective_flops, byte_count, name="fp8_stage1", verbose=0
            ) as perf:
                kernel(*arg_sets[(warmup + i) % data_clones])
            samples.append(perf.dt_ms * 1000)
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
    parser.add_argument(
        "--routing", choices=("balanced", "random", "skewed"), default="balanced"
    )
    parser.add_argument(
        "--data-case",
        choices=(
            "random",
            "zero_a",
            "zero_a_scale",
            "zero_b_scale",
            "last_k",
            "zero_gate",
            "zero_up",
        ),
        default="random",
    )
    parser.add_argument("--no-xcd-swizzle", action="store_true")
    parser.add_argument("--group-size-m", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--data-clones", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    check_native_weight_layout()
    if args.layout_check:
        return
    if "gfx950" not in torch.cuda.get_device_properties().gcnArchName:
        raise RuntimeError("FP8 MFMA requires gfx950")
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    common = {"xcd_swizzle": not args.no_xcd_swizzle, "group_size_m": args.group_size_m}
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
