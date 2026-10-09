# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""gfx950 W128x128/A1x128 FP8 GateUp with BF16 routed output.

Ported from the standalone half-M single-FIFO stage1 experiment. Weights use
Aiter's shuffle_weight(16,16), with gate rows followed by up rows. Scales are
FP32: A is physically [K/128, tokens], B is [E, 2*I/128, K/128]. Each workgroup
consumes 256 sorted rows and produces 128 columns of SiLU(gate) * up, without
clamp or routing weights. Sorting capacity may exceed the valid padded prefix.
"""

import functools

import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import arith, const_expr, range_constexpr, rocdl
from flydsl.expr.typing import Float8E4M3FN, Float32, T
from flydsl.expr.typing import Vector as Vec
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as _llvm, vector

from pyhip.codegen.flydsl.helpers import rocdl_aux
from .moe_gemm_2stage.common import get_device_cache_key


SORT_BLOCK_M = 256
BLOCK_M = 128
BLOCK_N = 128
BLOCK_K = 128
TOKEN_MASK = 0xFFFFFF
BUFFER_LIMIT = 0x7FFFFFFF


def _validate_parameters(tokens, intermediate_size, hidden_size, topk, num_experts):
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
        raise ValueError("A/scales/output and each expert's weight must fit 31-bit byte offsets")


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


def _waitcnt(vmcnt=63, expcnt=7, lgkmcnt=63):
    return (vmcnt & 15) | (expcnt << 4) | (lgkmcnt << 8) | ((vmcnt >> 4) << 14)


@functools.cache
def _compile_stage1(device_key, intermediate_size, hidden_size, topk, num_experts,
                    xcd_swizzle=True, group_size_m=1):
    if device_key is None or device_key[2].split(":")[0] != "gfx950":
        raise ValueError("FP8 blockscale GateUp requires gfx950")
    _validate_parameters(1, intermediate_size, hidden_size, topk, num_experts)
    if group_size_m <= 0:
        raise ValueError("group_size_m must be positive")
    kb_count = hidden_size // BLOCK_K
    n_tiles = intermediate_size // BLOCK_N
    a_group8 = 8 * BLOCK_K + 16
    a_group16 = 2 * a_group8 + 32
    a_slice_rows = BLOCK_M // 2
    a_lds_elems = (a_slice_rows // 16) * a_group16
    b_group16 = 16 * BLOCK_K
    b_lds_elems = (BLOCK_N // 16) * b_group16

    def _get_pids(pid, m_tiles, grid_size):
        if const_expr(xcd_swizzle):
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
        a_t0_slice0: fx.Array[Float8E4M3FN, a_lds_elems, 16]
        a_t0_slice1: fx.Array[Float8E4M3FN, a_lds_elems, 16]
        a_b0_slice0: fx.Array[Float8E4M3FN, a_lds_elems, 16]
        a_b0_slice1: fx.Array[Float8E4M3FN, a_lds_elems, 16]
        a_t1_slice0: fx.Array[Float8E4M3FN, a_lds_elems, 16]
        a_t1_slice1: fx.Array[Float8E4M3FN, a_lds_elems, 16]
        a_b1_slice0: fx.Array[Float8E4M3FN, a_lds_elems, 16]
        a_b1_slice1: fx.Array[Float8E4M3FN, a_lds_elems, 16]
        b_l0: fx.Array[Float8E4M3FN, b_lds_elems, 16]
        b_r0: fx.Array[Float8E4M3FN, b_lds_elems, 16]
        b_l1: fx.Array[Float8E4M3FN, b_lds_elems, 16]
        b_r1: fx.Array[Float8E4M3FN, b_lds_elems, 16]
        # 256 row scales duplicated across 512 lanes, shared by both M slices.
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
        num_sorted_ids: fx.Int32,
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
        valid_rsrc = _buffer_resource(arg_num_valid_ids, 4)
        valid_rows = rocdl.raw_ptr_buffer_load(
            T.i32, valid_rsrc, fx.Int32(0), fx.Int32(0), aux=rocdl_aux(0)
        )
        valid_rows = fx.Int32(rocdl.readfirstlane(T.i32, arith._to_raw(valid_rows)))
        # Uniform across the WG, before any expert/route loads or barriers.
        # Inactive capacity blocks may contain arbitrary expert IDs.
        if sorted_base < valid_rows:
            sorted_rsrc = _buffer_resource(
                arg_sorted_ids, arith._to_raw(fx.Int32(num_sorted_ids * 4))
            )
            expert_rsrc = _buffer_resource(
                arg_expert_ids, arith._to_raw(fx.Int32(num_expert_blocks * 4))
            )
            expert = rocdl.raw_ptr_buffer_load(
                T.i32, expert_rsrc, block_id * 4, fx.Int32(0), aux=rocdl_aux(0)
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
            # Native N16/K16 bytes stay in LDS; the consumer permutes the rows.
            b_read_layout = fx.make_layout(
                ((2, BLOCK_N // 16, 8), ((16, 2), BLOCK_K // 32)),
                ((16, 32, b_group16), ((1, 256), 512)),
            )
            b_gate = [fx.make_view(p, b_read_layout) for p in (lds.b_l0.ptr, lds.b_l1.ptr)]
            b_up = [fx.make_view(p, b_read_layout) for p in (lds.b_r0.ptr, lds.b_r1.ptr)]

            # Transposed C: the weight is MMA operand A.
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
                return fx.Int32(rocdl.raw_ptr_buffer_load(
                    T.i32, sorted_rsrc, fx.Int32((sorted_base + row) * 4),
                    fx.Int32(0), aux=rocdl_aux(0),
                ))

            store_ids = [
                sorted_id(bottom * 128 + repeat * 32 + wave_m * 16 + lane_id % 16)
                for bottom in range_constexpr(2)
                for repeat in range_constexpr(4)
            ]
            # One cooperative DWORDx4 DMA covers one complete A64xK128 entry.
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
                    offsets.append(valid.select(offset, fx.Int32(BUFFER_LIMIT)))
                a_voffsets.append(offsets)

            scale_id = sorted_id(tid % SORT_BLOCK_M)
            scale_token = scale_id & TOKEN_MASK
            scale_valid = (scale_token < num_tokens) & (((scale_id >> 24) & 255) < topk)
            scale_a_voffset = scale_valid.select(scale_token * 4, fx.Int32(BUFFER_LIMIT))
            wave_dma_base = rocdl.readfirstlane(
                T.i32,
                arith._to_raw(fx.Int32((wave_id % 2) * a_group8 + (wave_id // 2) * a_group16)),
            )
            scale_dma_base = rocdl.readfirstlane(T.i32, arith._to_raw(fx.Int32(tid * 4)))

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
                return [
                    [_lds_byte_ptr(fx.get_iter(view), b_wave_base + chunk * 1024)
                     for chunk in range_constexpr(2)]
                    for view in views
                ]

            bl_dma = b_dma_ptrs(b_gate)
            br_dma = b_dma_ptrs(b_up)
            scale_dma = [
                _lds_byte_ptr(p, scale_dma_base)
                for p in (lds.scale_a0.ptr, lds.scale_a1.ptr)
            ]
            # Native offset: (n//16)*16*K + (k//16)*256 + (n%16)*16 + k%16.
            b_voffset = fx.Int32(wave_id * (16 * hidden_size) + lane_id * 16)
            b_gate_base = n_tile * (BLOCK_N * hidden_size)
            b_up_base = b_gate_base + intermediate_size * hidden_size

            def ac_a(buf, ki, m_slice, bottom=0):
                rocdl.raw_ptr_buffer_load_lds(
                    a_rsrc,
                    ab_dma[buf][m_slice] if const_expr(bottom) else at_dma[buf][m_slice],
                    fx.Int32(16), a_voffsets[bottom][m_slice], fx.Int32(ki * BLOCK_K),
                    fx.Int32(0), rocdl_aux(0),
                )

            def ac_b(buf, ki, is_up=0):
                for chunk in range_constexpr(2):
                    rocdl.raw_ptr_buffer_load_lds(
                        b_rsrc,
                        br_dma[buf][chunk] if const_expr(is_up) else bl_dma[buf][chunk],
                        fx.Int32(16), b_voffset,
                        fx.Int32((b_up_base if const_expr(is_up) else b_gate_base)
                                 + ki * (BLOCK_K * 16) + chunk * 1024),
                        fx.Int32(0), rocdl_aux(0),
                    )

            def ac_scale_a(buf, ki):
                rocdl.raw_ptr_buffer_load_lds(
                    scale_a_rsrc, scale_dma[buf], fx.Int32(4), scale_a_voffset,
                    fx.Int32(ki * num_tokens * 4), fx.Int32(0), rocdl_aux(0),
                )

            def rd_a(buf, m_slice, bottom=0):
                view = a_bottom[buf][m_slice] if const_expr(bottom) else a_top[buf][m_slice]
                fx.copy(copy_atom, copy_a.partition_S(view), dest_a)

            def rd_bl(buf):
                fx.copy(copy_atom, copy_b.partition_S(b_gate[buf]), dest_bl)

            def rd_br(buf):
                fx.copy(copy_atom, copy_b.partition_S(b_up[buf]), dest_br)

            scale_copy = fx.make_copy_atom(fx.UniversalCopy32b(), Float32)

            def rd_scale_a(buf, bottom, m_slice):
                values = []
                for m0 in range_constexpr(2):
                    index = wave_m * 256 + bottom * 128 + wave_m * 16 + lane_id % 16
                    index = index + (m_slice * 2 + m0) * 32
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
                    [arith._to_raw(scale_b_rsrc), arith._to_raw(offset),
                     arith._to_raw(offset + n_tiles * kb_count * 4)],
                    "s_buffer_load_dword $0, $2, $3\ns_buffer_load_dword $1, $2, $4",
                    "=&s,=&s,s,s,s,~{memory}",
                    has_side_effects=True,
                )
                return Vec.from_elements(
                    [fx.Float32(_llvm.extractvalue(T.f32, result, [i]))
                     for i in range_constexpr(2)], Float32,
                )

            def compute(c_frag, b_frag, old_scale_a, old_scale_b, prev_m_slice):
                # Retire the old partial while issuing independent new MFMAs.
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
                    cs0 = c_frag[None, 0, prev_m_slice * 2 + m0]
                    cs1 = c_frag[None, 1, prev_m_slice * 2 + m0]
                    p0 = Vec(frag_p[None, 0, m0].load())
                    p1 = Vec(frag_p[None, 1, m0].load())
                    c0, c1 = Vec(cs0.load()), Vec(cs1.load())
                    b0 = vector.bitcast(T.vec(8, T.i32), b_frag[None, 0, 0].load().ir_value())
                    b1 = vector.bitcast(T.vec(8, T.i32), b_frag[None, 1, 0].load().ir_value())
                    a = vector.bitcast(T.vec(8, T.i32), frag_a[None, m0, 0].load().ir_value())
                    args = [arith._to_raw(p0[i]) for i in range_constexpr(4)]
                    args += [arith._to_raw(p1[i]) for i in range_constexpr(4)]
                    args += [arith._to_raw(scale)]
                    args += [arith._to_raw(c0[i]) for i in range_constexpr(4)]
                    args += [arith._to_raw(c1[i]) for i in range_constexpr(4)]
                    args += [arith._to_raw(b0), arith._to_raw(b1), arith._to_raw(a)]
                    result = _llvm.inline_asm(
                        result_type, args, asm,
                        "=&v,=&v,=&v,=&v,=&v,=&v,=&v,=&v,=&v,=&v,"
                        "v,v,v,v,v,v,v,v,v,0,1,2,3,4,5,6,7,v,v,v",
                        has_side_effects=True,
                    )
                    cs0.store(Vec.from_elements(
                        [fx.Float32(_llvm.extractvalue(T.f32, result, [i]))
                         for i in range_constexpr(4)], Float32,
                    ))
                    cs1.store(Vec.from_elements(
                        [fx.Float32(_llvm.extractvalue(T.f32, result, [i + 4]))
                         for i in range_constexpr(4)], Float32,
                    ))
                    frag_p[None, 0, m0].store(_llvm.extractvalue(T.vec(4, T.f32), result, [8]))
                    frag_p[None, 1, m0].store(_llvm.extractvalue(T.vec(4, T.f32), result, [9]))

            def begin_compute_phase():
                rocdl.sched_barrier(0)
                rocdl.s_barrier()
                rocdl.sched_barrier(0)
                rocdl.s_waitcnt(_waitcnt(lgkmcnt=0))
                rocdl.sched_barrier(0)

            def end_compute_phase():
                rocdl.sched_barrier(0)
                rocdl.s_barrier()
                rocdl.sched_barrier(0)

            def load_b_carry():
                value = Vec(frag_bl.load()).bitcast(fx.Int32)
                # Preserve the opaque DWORD loop boundary; emits no instruction.
                return Vec(_llvm.inline_asm(
                    T.vec(16, T.i32), [value.ir_value()], "", "=v,0", has_side_effects=True,
                ))

            rocdl.sched_barrier(0)
            rocdl.s_waitcnt(_waitcnt(vmcnt=0))
            rocdl.sched_barrier(0)
            ac_scale_a(0, fx.Int32(0))
            rocdl.sched_barrier(0)
            ac_b(0, 0)
            rocdl.sched_barrier(0)
            ac_a(0, 0, 0)
            ac_a(0, 0, 1)
            rocdl.sched_barrier(0)
            ac_b(0, 0, 1)
            rocdl.sched_barrier(0)
            ac_a(0, 0, 0, 1)
            ac_a(0, 0, 1, 1)
            rocdl.sched_barrier(0)
            if wave_id >= 4:
                rocdl.s_barrier()
            c_tl.fill(0)
            c_tr.fill(0)
            c_bl.fill(0)
            c_br.fill(0)
            rocdl.s_waitcnt(_waitcnt(vmcnt=4))
            rocdl.s_barrier()
            rocdl.sched_barrier(0)
            ac_scale_a(1, fx.Int32(1))
            rocdl.sched_barrier(0)
            ac_a(1, 1, 0)
            ac_a(1, 1, 1)
            rocdl.sched_barrier(0)
            ac_b(1, 1)
            rocdl.sched_barrier(0)
            ac_b(1, 1, 1)
            rocdl.sched_barrier(0)
            rocdl.s_waitcnt(_waitcnt(vmcnt=7))
            rocdl.s_barrier()
            rocdl.sched_barrier(0)
            rd_bl(0)
            frag_p.fill(0)
            initial = [
                c_tl.load(), c_tr.load(), c_bl.load(), c_br.load(), frag_p.load(),
                Vec.filled(2, 0.0, Float32), fx.Float32(0), load_b_carry(),
            ]
            for k_index, states in range(0, kb_count, 2, init=initial):
                c_tl.store(states[0])
                c_tr.store(states[1])
                c_bl.store(states[2])
                c_br.store(states[3])
                frag_p.store(states[4])
                old_a = Vec(states[5])
                old_b = fx.Float32(states[6])
                frag_bl.store(Vec(states[7]).bitcast(Float8E4M3FN))
                kiter = fx.Int32(k_index)
                for k_tile in range_constexpr(2):
                    ki = kiter + k_tile
                    for m_slice in range_constexpr(2):
                        rd_a(k_tile, m_slice)
                        top_scale = rd_scale_a(k_tile, 0, m_slice)
                        if const_expr(m_slice == 0):
                            b_scale = rd_scale_b(ki)
                        ac_a(1 - k_tile, ki + 1, m_slice, 1)
                        begin_compute_phase()
                        compute(c_br, frag_bl, old_a, old_b, 1 - m_slice)
                        end_compute_phase()
                        if const_expr(m_slice == 0):
                            rd_br(k_tile)
                        begin_compute_phase()
                        compute(c_tl, frag_br, top_scale, b_scale[0], m_slice)
                        end_compute_phase()
                        # Both staggered groups have finished TL's LDS reads.
                        ac_a(k_tile, ki + 2, m_slice)
                        rd_a(k_tile, m_slice, 1)
                        bottom_scale = rd_scale_a(k_tile, 1, m_slice)
                        if const_expr(m_slice == 0):
                            ac_b(k_tile, ki + 2)
                        begin_compute_phase()
                        compute(c_tr, frag_bl, top_scale, b_scale[1], m_slice)
                        end_compute_phase()
                        if const_expr(m_slice == 0):
                            ac_b(k_tile, ki + 2, 1)
                            rocdl.s_waitcnt(_waitcnt(vmcnt=8))
                        else:
                            ac_scale_a(k_tile, ki + 2)
                            rocdl.s_waitcnt(_waitcnt(vmcnt=7))
                            rd_bl(1 - k_tile)
                        begin_compute_phase()
                        compute(c_bl, frag_br, bottom_scale, b_scale[0], m_slice)
                        end_compute_phase()
                        old_a, old_b = bottom_scale, b_scale[1]
                result = yield [
                    c_tl.load(), c_tr.load(), c_bl.load(), c_br.load(), frag_p.load(),
                    old_a, old_b, load_b_carry(),
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
                    cs.store(fx.fma(
                        frag_p[None, n0, m0].load(), Vec.filled(4, scale, Float32), cs.load(),
                    ))
            if wave_id < 4:
                rocdl.s_barrier()

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
                    fused = store_ids[bottom * 4 + row_repeat]
                    token, slot = fused & TOKEN_MASK, (fused >> 24) & 255
                    valid = ((token < num_tokens) & (slot < topk)
                             & (sorted_base + row < valid_rows))
                    g0 = Vec(gates[None, 0, row_repeat].load())
                    g1 = Vec(gates[None, 1, row_repeat].load())
                    u0 = Vec(ups[None, 0, row_repeat].load())
                    u1 = Vec(ups[None, 1, row_repeat].load())
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
                        [fx.Int32(_llvm.extractvalue(T.i32, swap0, [0])),
                         fx.Int32(_llvm.extractvalue(T.i32, swap1, [0])),
                         fx.Int32(_llvm.extractvalue(T.i32, swap0, [1])),
                         fx.Int32(_llvm.extractvalue(T.i32, swap1, [1]))], fx.Int32,
                    )
                    lane_group = lane_id // 16
                    col = (n_tile * 128 + lane_group % 2 * 64 + wave_n * 16
                           + lane_group // 2 * 8)
                    address = ((token * topk + slot) * intermediate_size + fx.Int32(col)) * 2
                    address = valid.select(address, fx.Int32(BUFFER_LIMIT))
                    rocdl.raw_ptr_buffer_store(
                        packed.ir_value(), c_rsrc, address.ir_value(),
                        fx.Int32(0).ir_value(), aux=rocdl_aux(0),
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
        num_sorted_ids: fx.Int32,
        stream: fx.Stream,
    ):
        fp8_stage1_kernel(
            a, b, scale_a, scale_b, sorted_ids, expert_ids, num_valid_ids, out,
            num_tokens, num_expert_blocks, num_sorted_ids,
        ).launch(
            grid=(num_expert_blocks * n_tiles, 1, 1), block=(512, 1, 1), stream=stream
        )

    launch_stage1.compile_hints["opt_level"] = 2
    return launch_stage1


def fp8_blockscale_stage1(a, weight, scale_a, scale_b, sorted_ids, expert_ids,
                         num_valid_ids, output):
    """Write BF16 [tokens, topk, I] on the input device's current stream.

    scale_a must already be physically K-block-major (Aiter transpose_scale=True).
    Valid routes form a 256-row-aligned prefix; padding encodes invalid token/slot
    IDs and each real (token, slot) occurs once. No host read of valid counts.
    """
    if a.ndim != 2 or weight.ndim != 3 or output.ndim != 3:
        raise ValueError("expected A[T,H], weight[E,2I,H], output[T,topk,I]")
    tokens, hidden = a.shape
    experts = weight.shape[0]
    topk, intermediate = output.shape[1:]
    _validate_parameters(tokens, intermediate, hidden, topk, experts)
    if not getattr(weight, "is_shuffled", False):
        raise ValueError("stage1 weight must use Aiter shuffle_weight(16,16)")
    if tuple(weight.shape) != (experts, 2 * intermediate, hidden) or output.shape[0] != tokens:
        raise ValueError("stage1 weight/output shape mismatch")
    tensors = ((a, torch.float8_e4m3fn), (weight, torch.float8_e4m3fn),
               (scale_a, torch.float32), (scale_b, torch.float32),
               (sorted_ids, torch.int32), (expert_ids, torch.int32),
               (num_valid_ids, torch.int32), (output, torch.bfloat16))
    for tensor, dtype in tensors:
        if (tensor.dtype != dtype or not tensor.is_cuda or tensor.device != a.device
                or not tensor.is_contiguous()):
            raise ValueError("stage1 tensors must be contiguous, correctly typed and on the same GPU")
    if (scale_a.numel() != tokens * (hidden // 128)
            or scale_b.numel() != experts * (2 * intermediate // 128) * (hidden // 128)):
        raise ValueError("stage1 scale size mismatch")
    if any(t.ndim != 1 for t in (sorted_ids, expert_ids, num_valid_ids)) or num_valid_ids.numel() < 1:
        raise ValueError("stage1 sorting metadata must be 1D with a valid count")
    if max(sorted_ids.numel(), expert_ids.numel()) * 4 >= BUFFER_LIMIT:
        raise ValueError("stage1 sorting metadata must fit 31-bit byte offsets")
    with torch.cuda.device(a.device):
        launcher = _compile_stage1(get_device_cache_key(), intermediate, hidden, topk, experts)
        if expert_ids.numel():
            # These are views only: do not transpose the already transposed scale bytes.
            _run_compiled(
                launcher, a.view(torch.int8), weight.view(torch.int8),
                scale_a.view(hidden // 128, tokens),
                scale_b.view(experts, 2 * intermediate // 128, hidden // 128),
                sorted_ids, expert_ids, num_valid_ids, output.view(-1),
                tokens, expert_ids.numel(), sorted_ids.numel(), torch.cuda.current_stream(a.device),
            )
    return output