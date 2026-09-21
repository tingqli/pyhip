# SPDX-License-Identifier: MIT

"""Persistent 8-wave A8W4 MoE down kernel for gfx950.

The kernel follows the four-slot B-ring and conditional-displacement barrier
pipeline used by ``moe_8wave_down.py``. Inputs use Aiter's A16W4 GUI shuffle:
FP8 activations, packed FP4 weights, and packed E8M0 per-1x32 scales.
"""

from functools import cache
import os

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled, ptr_arg
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import arith, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from pyhip.contrib.flydsl import helpers as fxh
from pyhip.contrib.flydsl.moe_gemm_2stage.common import (
    torch_tensor_to_pointer as _ptr,
)

from moe_8wave_down_utils import ROCDLBuffer

if os.environ.get("PYHIP_FLYDSL_NOP_MFMA") == "1":
    os.environ["FLYDSL_RUNTIME_ENABLE_CACHE"] = "0"
    from pyhip.contrib.flydsl.mfma_nop_patch import install_mfma_nop_patch

    install_mfma_nop_patch("flydsl_moe_gemm_8wave_down_a8w4_0")

#fxh.dump_ir(True)

def _atomic_add_i32(tensor, value):
    ptr = fx.to_llvm_ptr(fx.get_iter(tensor))
    old = llvm.AtomicRMWOp(
        llvm.AtomicBinOp.add,
        ptr,
        arith._to_raw(value),
        llvm.AtomicOrdering.monotonic,
        syncscope="agent",
        alignment=4,
    ).result
    return fx.Int32(old)


def _recast_tensor(tensor, dtype):
    dst_ptr_type = fx.PointerType.get(
        dtype.ir_type,
        tensor.memspace,
        max(1, dtype.width // 8),
    )
    dst_ptr = fx.recast_iter(dst_ptr_type, fx.get_iter(tensor))
    dst_layout = fx.recast_layout(tensor.layout, tensor.dtype.width, dtype.width)
    return fx.make_view(dst_ptr, dst_layout)


def _ptr_to_tensor(ptr, shape, dtype):
    tensor = fx.make_view(ptr, fx.make_ordered_layout(shape, 0))
    return _recast_tensor(tensor, dtype)


@cache
def flydsl_moe_gemm_8wave_down_a8w4(
    *,
    n: int,
    k: int,
    topk: int,
    num_experts: int,
    block_m: int = 128,
    block_n: int = 128,
    num_oc_splits: int = 1,
):
    """Build an FP8 x packed-FP4 persistent MoE down kernel.

    The returned callable writes weighted per-route BF16 output with shape
    ``[tokens, topk, n]``. The caller performs the final top-k reduction.
    """
    assert block_m in (128, 256)
    assert block_n in (64, 128)
    assert k % 128 == 0
    assert n % num_oc_splits == 0
    n_split = n // num_oc_splits
    assert n_split % block_n == 0
    assert n_split % 128 == 0
    n_tiles_per_split = n_split // block_n
    assert n_tiles_per_split >= 3

    num_warps = 8
    num_threads = num_warps * 64
    num_slots = 4
    m_waves = 8
    n_waves = 1
    num_mma_m = block_m // (m_waves * 16)
    num_mma_n = block_n // (n_waves * 16)
    assert num_mma_n > 0 and num_mma_n % 2 == 0
    k_blocks = k // 128
    scale_k_cols = ((k // 32 + 7) // 8) * 8
    scale_k_packs = (k_blocks + 1) // 2
    scale_stride_k = 64
    scale_stride_m = scale_k_packs * scale_stride_k
    scale_tile_dwords = (block_n // 32) * scale_stride_m
    weight_tile_bytes = block_n * k // 2
    weight_tile_dwords = weight_tile_bytes // 4

    @fx.struct
    class SharedStorage:
        # Four packed-FP4 B tiles: 4 * block_n * k / 2 bytes.
        b_ring: fx.Array[fx.Int32, block_n * k // 2, 16]
        scale_b_ring: fx.Array[fx.Int32, scale_tile_dwords * num_slots, 16]
        sorted_weights: fx.Array[fx.Float32, block_m, 16]
        sorted_ids: fx.Array[fx.Int32, block_m, 16]

    @flyc.kernel(known_block_size=[num_threads, 1, 1])
    def flydsl_moe_gemm_8wave_down_a8w4(
        output: fx.Pointer,
        input_q: fx.Pointer,
        weight_shuffled: fx.Pointer,
        input_scales: fx.Pointer,
        weight_scales: fx.Pointer,
        sorted_ids: fx.Pointer,
        sorted_weights: fx.Pointer,
        sorted_expert_ids: fx.Pointer,
        num_valid_ids: fx.Pointer,
        task_counter: fx.Pointer,
        num_tokens: fx.Int32,
        num_expert_blocks: fx.Int32,
    ):
        tid = fx.thread_idx.x
        wave = tid // fx.Int32(64)
        lane = tid % fx.Int32(64)
        lane_div16 = lane // fx.Int32(16)
        lane_mod16 = lane % fx.Int32(16)
        wave_m = wave % fx.Int32(m_waves)
        wave_n = wave // fx.Int32(m_waves)

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        lds_task_id = fx.recast_iter(fx.Int32, lds.b_ring.ptr)
        sorted_ids_lds = fx.make_view(
            lds.sorted_ids.ptr, fx.make_layout(block_m, 1)
        )
        sorted_weights_lds = fx.make_view(
            lds.sorted_weights.ptr, fx.make_layout(block_m, 1)
        )

        # GUI FP4 layout, in physical order:
        # [dword4, n16, k-lane4, K/128, N/16, ring-slot].
        lds_b_ring = fx.make_view(
            lds.b_ring.ptr,
            fx.make_ordered_layout(
                (4, 16, 4, k_blocks, block_n // 16, num_slots), 0
            ),
        )
        lds_scale_b_ring = fx.make_view(
            lds.scale_b_ring.ptr,
            fx.make_ordered_layout(
                (64, scale_k_packs, block_n // 32, num_slots), 0
            ),
        )

        rows = num_tokens * fx.Int32(topk)
        input_data = _ptr_to_tensor(
            input_q, (16, k // 16, rows), fx.Int32
        )
        output_data = fx.rocdl.make_buffer_tensor(
            _ptr_to_tensor(output, (8, n // 8, rows), fx.Int32), False
        )
        sorted_ids_data = _ptr_to_tensor(
            sorted_ids, (num_expert_blocks * fx.Int32(block_m),), fx.Int32
        )
        sorted_weights_data = _ptr_to_tensor(
            sorted_weights, (num_expert_blocks * fx.Int32(block_m),), fx.Float32
        )
        sorted_expert_data = _ptr_to_tensor(
            sorted_expert_ids, (num_expert_blocks,), fx.Int32
        )
        num_valid_data = _ptr_to_tensor(num_valid_ids, (1,), fx.Int32)
        weight_bytes = _ptr_to_tensor(
            weight_shuffled, (num_experts * n * k // 2,), fx.Uint8
        )
        weight_scale_words = _ptr_to_tensor(
            weight_scales,
            (num_experts * n * scale_k_cols,),
            fx.Int32,
        )
        input_scale_storage = _ptr_to_tensor(
            input_scales,
            (num_expert_blocks * fx.Int32(block_m * scale_k_cols),),
            fx.Int32,
        )
        input_scale_words = fx.make_view(
            fx.get_iter(input_scale_storage),
            fx.make_layout(
                (64, scale_k_packs, block_m // 32, num_expert_blocks),
                (
                    1,
                    scale_stride_k,
                    scale_stride_m,
                    (block_m // 32) * scale_stride_m,
                ),
            ),
        )
        task_counter_data = _ptr_to_tensor(task_counter, (4,), fx.Int32)
        a_mma_frag = fx.make_rmem_tensor(
            [8, num_mma_m, k_blocks], fx.Int32
        )
        a_mma_frag_w = fx.make_view(
            fx.get_iter(a_mma_frag),
            fx.make_ordered_layout([4, 2, num_mma_m, k_blocks], 0),
        )
        b_mma_frag = fx.make_rmem_tensor([4, num_mma_n, k_blocks], fx.Int32)
        a_scale_words = fx.make_rmem_tensor(
            [num_mma_m, scale_k_packs], fx.Int32
        )
        b_scale_words = fx.make_rmem_tensor(
            [num_mma_n // 2, scale_k_packs], fx.Int32
        )
        acc = fx.make_rmem_tensor([4, num_mma_m, num_mma_n], fx.Float32)
        output_frag = fx.make_rmem_tensor(
            [8, num_mma_m, num_mma_n // 2], fx.BFloat16
        )
        pair_type = ir.Type.parse("!llvm.struct<(i32, i32)>")

        max_id = num_valid_data[0]
        running = fx.Boolean(True)
        while running:
            if tid == fx.Int32(0):
                lds_task_id[0] = _atomic_add_i32(task_counter_data, fx.Int32(1))
            fx.barrier()
            task = lds_task_id[0]
            blk_m = task // fx.Int32(num_oc_splits)
            blk_oc = task % fx.Int32(num_oc_splits)
            running = blk_m * fx.Int32(block_m) < max_id

            if running:
                expert = fx.Int32(sorted_expert_data[blk_m])
                weight_rsrc = ROCDLBuffer(weight_bytes)
                weight_scale_buffer = ROCDLBuffer(weight_scale_words)
                sorted_base = blk_m * fx.Int32(block_m)
                sorted_ids_src = fx.make_view(
                    fx.get_iter(sorted_ids_data) + sorted_base,
                    fx.make_layout(block_m, 1),
                )
                sorted_weights_src = fx.make_view(
                    fx.get_iter(sorted_weights_data) + sorted_base,
                    fx.make_layout(block_m, 1),
                )
                ROCDLBuffer(sorted_ids_src).load_async(
                    fx.get_iter(sorted_ids_src),
                    fx.get_iter(sorted_ids_lds),
                    block_m,
                    num_threads,
                )
                ROCDLBuffer(sorted_weights_src).load_async(
                    fx.get_iter(sorted_weights_src),
                    fx.get_iter(sorted_weights_lds),
                    block_m,
                    num_threads,
                )
                fx.rocdl.asyncmark()
                fx.rocdl.wait_asyncmark(0)
                fx.barrier()

                output_rows = {}
                valid_rows = {}
                route_weights = {}
                m_wave_base = wave_m * fx.Int32(num_mma_m * 16)
                for mi in range_constexpr(num_mma_m):
                    row_in_block = m_wave_base + fx.Int32(mi * 16) + lane_mod16
                    packed = fx.Int32(sorted_ids_lds[row_in_block])
                    token = packed & fx.Int32(0xFFFFFF)
                    slot = packed >> fx.Int32(24)
                    output_rows[mi] = token * fx.Int32(topk) + slot
                    valid_rows[mi] = slot < fx.Int32(topk)
                    route_weights[mi] = sorted_weights_lds[row_in_block]

                a_mma_frag.fill(0)
                for mi in range_constexpr(num_mma_m):
                    if valid_rows[mi]:
                        for kb in range_constexpr(k_blocks):
                            for step in range_constexpr(2):
                                a_mma_frag_w[None, step, mi, kb] = input_data[
                                    None,
                                    lane_div16 + (kb * 128 + step * 64) // 16,
                                    output_rows[mi],
                                ].load()

                # Packed E8M0 layout: (M/32, ceil((K/32)/8), 4, 16).
                for mi in range_constexpr(num_mma_m):
                    m16 = m_wave_base // fx.Int32(16) + fx.Int32(mi)
                    a_scale_shift = (m16 % fx.Int32(2)) * fx.Int32(8)
                    for kp in range_constexpr(scale_k_packs):
                        a_scale_words[mi, kp] = (
                            input_scale_words[
                                lane,
                                kp,
                                m16 // fx.Int32(2),
                                blk_m,
                            ]
                            >> a_scale_shift
                        )

                weight_expert_byte = expert * fx.Int32(n * k // 2)
                weight_split_byte = blk_oc * fx.Int32(n_split * k // 2)

                def global_load_b(n_tile):
                    src = (
                        fx.get_iter(weight_bytes)
                        + weight_expert_byte
                        + weight_split_byte
                        + n_tile * fx.Int32(weight_tile_bytes)
                    )
                    weight_rsrc.load_async(
                        src,
                        fx.get_iter(lds_b_ring[None, None, None, None, None, n_tile % 4]),
                        weight_tile_dwords,
                        num_threads,
                    )
                    scale_m_base = (
                        expert * fx.Int32(n // 32)
                        + (
                            blk_oc * fx.Int32(n_split)
                            + n_tile * fx.Int32(block_n)
                        ) // fx.Int32(32)
                    )
                    scale_src = (
                        fx.get_iter(weight_scale_words)
                        + scale_m_base * fx.Int32(scale_stride_m)
                    )
                    weight_scale_buffer.load_async(
                        scale_src,
                        fx.get_iter(
                            lds_scale_b_ring[None, None, None, n_tile % 4]
                        ),
                        scale_tile_dwords,
                        num_threads,
                    )

                def ds_read_b(n_tile):
                    slot_id = n_tile % 4
                    n16_base = wave_n * fx.Int32(num_mma_n)
                    for kb in range_constexpr(k_blocks):
                        for ni in range_constexpr(num_mma_n):
                            b_mma_frag[None, ni, kb] = lds_b_ring[
                                None,
                                lane_mod16,
                                lane_div16,
                                kb,
                                n16_base + fx.Int32(ni),
                                slot_id,
                            ].load()

                    for kp in range_constexpr(scale_k_packs):
                        for np in range_constexpr(num_mma_n // 2):
                            b_scale_words[np, kp] = lds_scale_b_ring[
                                lane, kp, np, slot_id
                            ]

                def compute():
                    acc.fill(0)
                    zero = fx.Int32(0)
                    for kb in range_constexpr(k_blocks):
                        kp = kb // 2
                        k_half = kb % 2
                        for mi in range_constexpr(num_mma_m):
                            a_vec = a_mma_frag[None, mi, kb].load()
                            a_scale = a_scale_words[mi, kp]
                            for ni in range_constexpr(num_mma_n):
                                np = ni // 2
                                n_half = ni % 2
                                b4 = Vec(b_mma_frag[None, ni, kb].load())
                                b_vec = Vec.from_elements(
                                    [
                                        b4[0], b4[1], b4[2], b4[3],
                                        zero, zero, zero, zero,
                                    ],
                                    fx.Int32,
                                )
                                c_vec = acc[None, mi, ni].load()
                                c_vec = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                                    T.vec(4, T.f32),
                                    [
                                        arith._to_raw(b_vec),
                                        arith._to_raw(a_vec),
                                        arith._to_raw(c_vec),
                                        4,
                                        0,
                                        k_half * 2 + n_half,
                                        arith._to_raw(b_scale_words[np, kp]),
                                        k_half * 2,
                                        arith._to_raw(a_scale),
                                    ],
                                )
                                acc[None, mi, ni].store(c_vec)
                                rocdl.sched_barrier(0)

                    for mi in range_constexpr(num_mma_m):
                        for ni in range_constexpr(0, num_mma_n, 2):
                            c0 = acc[None, mi, ni].load() * route_weights[mi]
                            c1 = acc[None, mi, ni + 1].load() * route_weights[mi]
                            d0_a = rocdl.cvt_pk_bf16_f32(c0[0], c0[1])
                            d1_a = rocdl.cvt_pk_bf16_f32(c0[2], c0[3])
                            d0_b = rocdl.cvt_pk_bf16_f32(c1[0], c1[1])
                            d1_b = rocdl.cvt_pk_bf16_f32(c1[2], c1[3])
                            swap0 = rocdl.permlane16_swap(
                                pair_type,
                                arith._to_raw(d0_a),
                                arith._to_raw(d0_b),
                                False,
                                False,
                            )
                            swap1 = rocdl.permlane16_swap(
                                pair_type,
                                arith._to_raw(d1_a),
                                arith._to_raw(d1_b),
                                False,
                                False,
                            )
                            output_values = Vec.from_elements(
                                [
                                    fx.Int32(llvm.extractvalue(T.i32, swap0, [0])),
                                    fx.Int32(llvm.extractvalue(T.i32, swap1, [0])),
                                    fx.Int32(llvm.extractvalue(T.i32, swap0, [1])),
                                    fx.Int32(llvm.extractvalue(T.i32, swap1, [1])),
                                ],
                                fx.Int32,
                            )
                            output_frag[None, mi, ni // 2].store(
                                output_values.bitcast(fx.BFloat16)
                            )

                def global_store(n_tile):
                    n_base = (
                        blk_oc * fx.Int32(n_split)
                        + n_tile * fx.Int32(block_n)
                        + wave_n * fx.Int32(num_mma_n * 16)
                    )
                    for mi in range_constexpr(num_mma_m):
                        for ni in range_constexpr(0, num_mma_n, 2):
                            output_values = output_frag[
                                None, mi, ni // 2
                            ].load().bitcast(fx.Int32)
                            swap_col = (
                                (lane_div16 & fx.Int32(1)) * fx.Int32(2)
                                + (lane_div16 >> fx.Int32(1))
                            )
                            output_data[
                                None,
                                n_base // fx.Int32(8) + fx.Int32(ni * 2) + swap_col,
                                output_rows[mi],
                            ] = output_values

                global_load_b(fx.Int32(0))
                fx.rocdl.asyncmark()
                global_load_b(fx.Int32(1))
                fx.rocdl.asyncmark()
                fx.rocdl.wait_asyncmark(1)

                # Offset the two half-workgroups by one barrier generation so
                # one half can compute while the other performs DS/VMEM work.
                if wave > fx.Int32(3):
                    fx.barrier()
                fx.barrier()

                ds_read_b(fx.Int32(0))
                global_load_b(fx.Int32(2))
                fx.rocdl.asyncmark()
                fx.rocdl.wait_asyncmark(1)
                fxh.s_waitcnt(lgkmcnt=0)
                fx.barrier()
                compute()

                loop_begin = fx.Int32(0)
                loop_end = fx.Int32(n_tiles_per_split - 3)
                loop_step = fx.Int32(1)
                for n_tile in range(loop_begin, loop_end, loop_step):
                    fx.barrier()
                    rocdl.s_setprio(1)
                    ds_read_b(n_tile + fx.Int32(1))
                    global_store(n_tile)
                    global_load_b(n_tile + fx.Int32(3))
                    fx.rocdl.asyncmark()
                    fx.rocdl.wait_asyncmark(1)
                    rocdl.s_setprio(0)

                    fxh.s_waitcnt(lgkmcnt=0)
                    fx.barrier()
                    compute()

                fx.barrier()
                ds_read_b(n_tiles_per_split - 2)
                global_store(n_tiles_per_split - 3)
                fx.rocdl.wait_asyncmark(0)
                fx.barrier()
                compute()

                fx.barrier()
                ds_read_b(n_tiles_per_split - 1)
                global_store(n_tiles_per_split - 2)
                compute()
                global_store(n_tiles_per_split - 1)

                # Balance the extra prologue barrier for waves 4-7.
                if wave < fx.Int32(4):
                    fx.barrier()
            #fx.barrier()

    @flyc.jit
    def launch(
        output: fx.Pointer,
        input_q: fx.Pointer,
        weight_shuffled: fx.Pointer,
        input_scales: fx.Pointer,
        weight_scales: fx.Pointer,
        sorted_ids: fx.Pointer,
        sorted_weights: fx.Pointer,
        sorted_expert_ids: fx.Pointer,
        num_valid_ids: fx.Pointer,
        task_counter: fx.Pointer,
        num_tokens: fx.Int32,
        num_expert_blocks: fx.Int32,
        stream: fx.Stream,
    ):
        flydsl_moe_gemm_8wave_down_a8w4(
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
            num_tokens,
            num_expert_blocks,
            value_attrs={
                "rocdl.waves_per_eu": 2,
                "rocdl.flat_work_group_size": "512,512",
                "passthrough": [
                    ["target-features", "-packed-fp32-ops"] # disable v_pk_mul (which has co-issue problem with MFMA)
                ],
            },
        ).launch(grid=(256, 1, 1), block=(512, 1, 1), stream=stream)

    def callable(
        output: torch.Tensor,
        input_q: torch.Tensor,
        weight_shuffled: torch.Tensor,
        input_scales: torch.Tensor,
        weight_scales: torch.Tensor,
        sorted_ids: torch.Tensor,
        sorted_weights: torch.Tensor,
        sorted_expert_ids: torch.Tensor,
        num_valid_ids: torch.Tensor,
        task_counter: torch.Tensor,
    ):
        num_tokens, input_topk, input_k = input_q.shape
        assert input_topk == topk and input_k == k
        assert weight_shuffled.shape == (num_experts, n, k // 2)
        assert output.shape == (num_tokens, topk, n)
        assert input_scales.shape[1] >= scale_k_cols
        assert weight_scales.shape[0] >= num_experts * n
        assert weight_scales.shape[1] >= scale_k_cols
        assert sorted_weights.shape == sorted_ids.shape

        task_counter.zero_()
        stream = torch.cuda.current_stream()
        _run_compiled(
            launch,
            _ptr(output),
            _ptr(input_q),
            ptr_arg(weight_shuffled),
            ptr_arg(input_scales),
            ptr_arg(weight_scales),
            _ptr(sorted_ids),
            _ptr(sorted_weights),
            _ptr(sorted_expert_ids),
            _ptr(num_valid_ids),
            _ptr(task_counter),
            fx.Int32(num_tokens),
            fx.Int32(sorted_expert_ids.numel()),
            fx.Stream(stream.cuda_stream),
        )
        return output

    return callable
