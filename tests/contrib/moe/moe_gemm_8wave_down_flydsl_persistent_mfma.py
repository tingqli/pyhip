# SPDX-License-Identifier: MIT

"""Single-dispatch persistent 8-wave MFMA MoE down kernel for gfx950."""

from functools import cache

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.expr import arith, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec


def _atomic_add_i32(addr, value):
    ptr = fx.buffer_ops.create_llvm_ptr(addr, address_space=1)
    return llvm.AtomicRMWOp(
        llvm.AtomicBinOp.add,
        ptr,
        arith._to_raw(value),
        llvm.AtomicOrdering.monotonic,
        syncscope="agent",
    ).res


def _load_i32x4(rsrc, byte_offset, mask=None):
    return Vec(
        fx.buffer_ops.buffer_load(
            rsrc,
            byte_offset // fx.Int32(4),
            vec_width=4,
            dtype=T.i32,
            mask=mask,
        )
    )


def _pack_i32x8(lo, hi):
    return lo.shuffle(hi, list(range(8)))


@cache
def compile_flydsl_moe_gemm_8wave_down_persistent_mfma(
    *, n, k, topk, block_m=256, block_n=64, num_oc_splits=1
):
    assert block_m == 256 and block_n == 64
    assert k % 128 == 0 and n % num_oc_splits == 0
    n_split = n // num_oc_splits
    assert n_split % block_n == 0
    n_tiles_per_split = n_split // block_n
    k_blocks = k // 128
    scale_n_blocks = n // 128

    @fx.struct
    class SharedStorage:
        task_id: fx.Array[fx.Int32, 1, 4]
        b_tile0: fx.Array[fx.Int32, block_n * k // 4, 16]
        b_tile1: fx.Array[fx.Int32, block_n * k // 4, 16]
        b_tile2: fx.Array[fx.Int32, block_n * k // 4, 16]
        b_tile3: fx.Array[fx.Int32, block_n * k // 4, 16]

    @flyc.kernel(known_block_size=[512, 1, 1])
    def kernel(
        out_addr: fx.Int64,
        input_addr: fx.Int64,
        weight_addr: fx.Int64,
        scale_a_addr: fx.Int64,
        scale_b_addr: fx.Int64,
        sorted_ids_addr: fx.Int64,
        sorted_weights_addr: fx.Int64,
        sorted_expert_ids_addr: fx.Int64,
        num_valid_addr: fx.Int64,
        counter_addr: fx.Int64,
        num_tokens: fx.Int32,
    ):
        tid = fx.Int32(gpu.thread_id("x"))
        wave = tid // fx.Int32(64)
        lane = tid % fx.Int32(64)
        lane_div16 = lane // fx.Int32(16)
        lane_mod16 = lane % fx.Int32(16)

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        task_view = fx.make_view(lds.task_id.ptr, fx.make_layout(1, 1))
        b_tiles = [lds.b_tile0, lds.b_tile1, lds.b_tile2, lds.b_tile3]

        def load_b_lds(slot, dword_offset):
            ptr = fx.add_offset(b_tiles[slot].ptr, dword_offset)
            return Vec(fx.make_view(ptr, fx.make_layout(4, 1)).load())

        max_bytes = fx.Int32(0x7FFFFFFF)
        ids_rsrc = fx.buffer_ops.create_buffer_resource_from_addr(sorted_ids_addr, num_records_bytes=max_bytes)
        routes_rsrc = fx.buffer_ops.create_buffer_resource_from_addr(sorted_weights_addr, num_records_bytes=max_bytes)
        experts_rsrc = fx.buffer_ops.create_buffer_resource_from_addr(sorted_expert_ids_addr, num_records_bytes=max_bytes)
        valid_rsrc = fx.buffer_ops.create_buffer_resource_from_addr(num_valid_addr, num_records_bytes=4)
        input_rsrc = fx.buffer_ops.create_buffer_resource_from_addr(input_addr, num_records_bytes=max_bytes)
        weight_rsrc = fx.buffer_ops.create_buffer_resource_from_addr(weight_addr, num_records_bytes=max_bytes)
        scale_a_rsrc = fx.buffer_ops.create_buffer_resource_from_addr(scale_a_addr, num_records_bytes=max_bytes)
        scale_b_rsrc = fx.buffer_ops.create_buffer_resource_from_addr(scale_b_addr, num_records_bytes=max_bytes)
        out_rsrc = fx.buffer_ops.create_buffer_resource_from_addr(out_addr, num_records_bytes=max_bytes)
        max_id = fx.Int32(fx.buffer_ops.buffer_load(valid_rsrc, fx.Int32(0), vec_width=1, dtype=T.i32))

        atom = fx.make_mma_atom(fx.rocdl.cdna4.MFMA_Scale(16, 16, 128, fx.Float8E4M3FN))
        a_mma_frag = fx.make_rmem_tensor(8, fx.Int32)
        b_mma_frag = fx.make_rmem_tensor(8, fx.Int32)
        c_mma_frag = fx.make_rmem_tensor(4, fx.Float32)
        running = fx.Int32(1)
        while running != fx.Int32(0):
            if tid == fx.Int32(0):
                task_view[0] = fx.Int32(_atomic_add_i32(counter_addr, fx.Int32(1)))
            fx.barrier()
            task = fx.Int32(task_view[0])
            blk_m = task // fx.Int32(num_oc_splits)
            blk_oc = task % fx.Int32(num_oc_splits)
            valid_task = blk_m * fx.Int32(block_m) < max_id
            running = valid_task.select(fx.Int32(1), fx.Int32(0))

            if valid_task:
                expert = fx.Int32(
                    fx.buffer_ops.buffer_load(experts_rsrc, blk_m, vec_width=1, dtype=T.i32)
                )
                packed_a = []
                route_rows = []
                valid_rows = []
                output_rows = []
                a_frag_all = []
                a_scales_all = []
                for mi in range_constexpr(2):
                    a_row = wave * fx.Int32(32) + fx.Int32(mi * 16) + lane_mod16
                    sorted_idx = blk_m * fx.Int32(block_m) + a_row
                    packed = fx.Int32(
                        fx.buffer_ops.buffer_load(ids_rsrc, sorted_idx, vec_width=1, dtype=T.i32)
                    )
                    token = packed & fx.Int32(0xFFFFFF)
                    slot = packed >> fx.Int32(24)
                    input_row = token * fx.Int32(topk) + slot
                    valid_a = (token < num_tokens) & (slot < fx.Int32(topk))
                    packed_a.append(packed)
                    per_m_frags = []
                    for kb in range_constexpr(k_blocks):
                        halves = []
                        for step in range_constexpr(2):
                            byte_offset = input_row * fx.Int32(k) + fx.Int32(kb * 128 + step * 64) + lane_div16 * fx.Int32(16)
                            halves.append(_load_i32x4(input_rsrc, byte_offset, mask=valid_a))
                        per_m_frags.append(_pack_i32x8(halves[0], halves[1]))
                    a_frag_all.append(per_m_frags)

                    scale_rows = []
                    route_values = []
                    row_valid_values = []
                    out_row_values = []
                    for ii in range_constexpr(4):
                        out_row = wave * fx.Int32(32) + fx.Int32(mi * 16) + lane_div16 * fx.Int32(4) + fx.Int32(ii)
                        pidx = blk_m * fx.Int32(block_m) + out_row
                        p = fx.Int32(fx.buffer_ops.buffer_load(ids_rsrc, pidx, vec_width=1, dtype=T.i32))
                        tok = p & fx.Int32(0xFFFFFF)
                        sl = p >> fx.Int32(24)
                        ridx = tok * fx.Int32(topk) + sl
                        row_valid = (tok < num_tokens) & (sl < fx.Int32(topk))
                        route_values.append(
                            fx.Float32(fx.buffer_ops.buffer_load(routes_rsrc, pidx, vec_width=1, dtype=T.f32))
                        )
                        row_valid_values.append(row_valid)
                        out_row_values.append(ridx)
                        scales = []
                        for kb in range_constexpr(k_blocks):
                            scale_idx = fx.Int32(kb) * (num_tokens * fx.Int32(topk)) + ridx
                            scales.append(
                                fx.Float32(
                                    fx.buffer_ops.buffer_load(
                                        scale_a_rsrc, scale_idx, vec_width=1, dtype=T.f32, mask=row_valid
                                    )
                                )
                            )
                        scale_rows.append(scales)
                    a_scales_all.append(scale_rows)
                    route_rows.append(route_values)
                    valid_rows.append(row_valid_values)
                    output_rows.append(out_row_values)

                def prefetch_b(n_tile):
                    n_base_pf = blk_oc * fx.Int32(n_split) + fx.Int32(n_tile * block_n)
                    b_expert_base = expert * fx.Int32(n * k) + n_base_pf * fx.Int32(k)
                    values = []
                    for copy_round in range_constexpr(k // 128):
                        dword_offset = tid * fx.Int32(4) + fx.Int32(copy_round * 512 * 4)
                        values.append(_load_i32x4(weight_rsrc, b_expert_base + dword_offset * fx.Int32(4)))
                    return values

                def commit_b(slot, values):
                    for copy_round in range_constexpr(k // 128):
                        dword_offset = tid * fx.Int32(4) + fx.Int32(copy_round * 512 * 4)
                        dst_ptr = fx.add_offset(b_tiles[slot].ptr, dword_offset)
                        fx.make_view(dst_ptr, fx.make_layout(4, 1)).store(values[copy_round])

                current_b = prefetch_b(0)
                commit_b(0, current_b)
                fx.barrier()

                for n_tile in range_constexpr(n_tiles_per_split):
                    n_base = blk_oc * fx.Int32(n_split) + fx.Int32(n_tile * block_n)
                    # Keep the next tile's VMEM values live while the current
                    # tile executes MFMA, then commit them to the next ring slot.
                    # The last iteration harmlessly reloads its own tile; this
                    # keeps the traced value defined on a single path.
                    next_tile = min(n_tile + 1, n_tiles_per_split - 1)
                    next_b = prefetch_b(next_tile)
                    acc = [Vec.filled(4, 0.0, fx.Float32) for _ in range_constexpr(8)]
                    for kb in range_constexpr(k_blocks):
                        b_frag = []
                        b_scales = []
                        for ni in range_constexpr(4):
                            n_col0 = n_base + fx.Int32(ni * 16)
                            n0 = n_col0 // fx.Int32(16)
                            row = fx.Int32(ni * 16) + lane_mod16
                            halves = []
                            for step in range_constexpr(2):
                                k0 = fx.Int32(kb * 2 + step)
                                # In the Aiter (16,16) preshuffle, the four 16-byte
                                # K groups are the four lanes of one i32x4 vector.
                                w_index = (
                                    (n0 - (n_base // fx.Int32(16))) * fx.Int32((k // 64) * 4 * 16 * 16)
                                    + k0 * fx.Int32(4 * 16 * 16)
                                    + lane_div16 * fx.Int32(16 * 16)
                                    + (row % fx.Int32(16)) * fx.Int32(16)
                                )
                                halves.append(load_b_lds(n_tile % 4, w_index // fx.Int32(4)))
                            b_frag.append(_pack_i32x8(halves[0], halves[1]))
                            sb_idx = (
                                expert * fx.Int32(scale_n_blocks * k_blocks)
                                + (n_col0 // fx.Int32(128)) * fx.Int32(k_blocks)
                                + fx.Int32(kb)
                            )
                            b_scales.append(
                                fx.Float32(fx.buffer_ops.buffer_load(scale_b_rsrc, sb_idx, vec_width=1, dtype=T.f32))
                            )

                        rocdl.s_setprio(1)
                        partials = []
                        for mi in range_constexpr(2):
                            for ni in range_constexpr(4):
                                a_mma_frag.store(a_frag_all[mi][kb])
                                b_mma_frag.store(b_frag[ni])
                                c_mma_frag.fill(0)
                                fx.gemm(atom, c_mma_frag, a_mma_frag, b_mma_frag, c_mma_frag)
                                partials.append(Vec(c_mma_frag.load()))
                        for mi in range_constexpr(2):
                            for ni in range_constexpr(4):
                                idx = mi * 4 + ni
                                partial = partials[idx]
                                scaled = [
                                    acc[idx][ii] + partial[ii] * a_scales_all[mi][ii][kb] * b_scales[ni]
                                    for ii in range_constexpr(4)
                                ]
                                acc[idx] = Vec.from_elements(scaled, fx.Float32)
                        rocdl.s_setprio(0)

                    for mi in range_constexpr(2):
                        for ni in range_constexpr(4):
                            idx = mi * 4 + ni
                            n_col = n_base + fx.Int32(ni * 16) + lane_mod16
                            for ii in range_constexpr(4):
                                route = route_rows[mi][ii]
                                row = output_rows[mi][ii]
                                valid_row = valid_rows[mi][ii]
                                out_idx = row * fx.Int32(n) + n_col
                                fx.buffer_ops.buffer_store(
                                    (acc[idx][ii] * route).to(fx.BFloat16).ir_value(),
                                    out_rsrc,
                                    out_idx,
                                    mask=valid_row,
                                )
                    if n_tile + 1 < n_tiles_per_split:
                        commit_b((n_tile + 1) % 4, next_b)
                    fx.barrier()
            fx.barrier()

    @flyc.jit
    def launch(
        out_addr: fx.Int64,
        input_addr: fx.Int64,
        weight_addr: fx.Int64,
        scale_a_addr: fx.Int64,
        scale_b_addr: fx.Int64,
        sorted_ids_addr: fx.Int64,
        sorted_weights_addr: fx.Int64,
        sorted_expert_ids_addr: fx.Int64,
        num_valid_addr: fx.Int64,
        counter_addr: fx.Int64,
        num_tokens: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(
            out_addr,
            input_addr,
            weight_addr,
            scale_a_addr,
            scale_b_addr,
            sorted_ids_addr,
            sorted_weights_addr,
            sorted_expert_ids_addr,
            num_valid_addr,
            counter_addr,
            num_tokens,
            value_attrs={"rocdl.waves_per_eu": 2, "rocdl.flat_work_group_size": "512,512"},
        ).launch(grid=(256, 1, 1), block=(512, 1, 1), stream=stream)

    return launch


class PersistentFlyDSLDownMFMA:
    def __init__(
        self,
        output,
        input_q,
        weight_shuffled,
        input_scales,
        weight_scales,
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        task_counter,
        *,
        num_tokens,
        n,
        k,
        topk,
        num_oc_splits,
    ):
        launcher = compile_flydsl_moe_gemm_8wave_down_persistent_mfma(
            n=n, k=k, topk=topk, num_oc_splits=num_oc_splits
        )
        self.args = (
            fx.Int64(output.data_ptr()),
            fx.Int64(input_q.data_ptr()),
            fx.Int64(weight_shuffled.data_ptr()),
            fx.Int64(input_scales.data_ptr()),
            fx.Int64(weight_scales.data_ptr()),
            fx.Int64(sorted_ids.data_ptr()),
            fx.Int64(sorted_weights.data_ptr()),
            fx.Int64(sorted_expert_ids.data_ptr()),
            fx.Int64(num_valid_ids.data_ptr()),
            fx.Int64(task_counter.data_ptr()),
            fx.Int32(num_tokens),
            fx.Stream(torch.cuda.current_stream().cuda_stream),
        )
        self.compiled = flyc.compile(launcher, *self.args)
        self.output = output
        self.counter = task_counter

    def __call__(self):
        self.counter.zero_()
        self.compiled(*self.args)
        return self.output
