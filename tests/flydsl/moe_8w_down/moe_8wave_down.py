# SPDX-License-Identifier: MIT

"""Single-dispatch persistent 8-wave MFMA MoE down kernel for gfx950."""

from functools import cache

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import arith, const_expr, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from pyhip.contrib.flydsl import helpers as fxh
from pyhip.contrib.flydsl.moe_gemm_2stage.common import (
    torch_tensor_to_pointer as _ptr,
)

# fxh.dump_ir(True)

from moe_8wave_down_utils import ROCDLBuffer


def _atomic_add_i32(addr, value):
    ptr = fx.buffer_ops.create_llvm_ptr(addr, address_space=1)
    return llvm.AtomicRMWOp(
        llvm.AtomicBinOp.add,
        ptr,
        arith._to_raw(value),
        llvm.AtomicOrdering.monotonic,
        syncscope="agent",
    ).res


def recast_tensor(tensor, dtype):
    dst_ptr_type = fx.PointerType.get(
        dtype.ir_type,
        tensor.memspace,
        dtype.width // 8,
    )
    dst_ptr = fx.recast_iter(dst_ptr_type, fx.get_iter(tensor))
    dst_layout = fx.recast_layout(
        tensor.layout,
        tensor.dtype.width,
        dtype.width,
    )
    return fx.make_view(dst_ptr, dst_layout)


def ptr_to_tensor(ptr, shape, dtype=None, as_buffer=False):
    tensor = fx.make_view(ptr, fx.make_ordered_layout(shape, 0))
    if dtype is not None:
        tensor = recast_tensor(tensor, dtype)
    if as_buffer:
        tensor = fx.rocdl.make_buffer_tensor(tensor, False)
    return tensor


@cache
def flydsl_moe_gemm_8wave_down(
    *, n, k, topk, num_experts, block_m=256, block_n=32, num_oc_splits=1
):
    assert block_m == 256
    assert block_n >= 32 and block_n % 32 == 0
    assert k % 128 == 0 and n % num_oc_splits == 0
    n_split = n // num_oc_splits
    assert n_split % block_n == 0
    assert n_split % 128 == 0
    n_tiles_per_split = n_split // block_n
    assert n_tiles_per_split >= 3
    n16_tiles_per_block = block_n // 16
    k_blocks = k // 128
    scale_n_blocks = n // 128
    scale_n_blocks_per_split = (n_split + 127) // 128
    num_warps = 8
    num_threads = num_warps * 64

    # A 16x16 preshuffled tile has ordered layout
    # [k16, n16, K/k16, N/n16], where k16 elements occupy 16 bytes.

    @fx.struct
    class SharedStorage:
        # Int32 storage makes the ring four times the FP8 element count in bytes.
        b_ring: fx.Array[fx.Int32, block_n * k, 16]
        sorted_weights: fx.Array[fx.Float32, block_m, 16]
        sorted_ids: fx.Array[fx.Int32, block_m, 16]
        scale_b: fx.Array[fx.Float32, scale_n_blocks_per_split * k_blocks, 16]

    @flyc.kernel(known_block_size=[num_threads, 1, 1])
    def flydsl_moe_gemm_8wave_down_kernel(
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

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        # The task ID is dead before the B ring is populated.
        lds_task_id = fx.recast_iter(fx.Int32, lds.b_ring.ptr)
        lds_scale_b = fx.make_view(
            lds.scale_b.ptr,
            fx.make_layout(
                (scale_n_blocks_per_split, k_blocks),
                (k_blocks, 1),
            ),
        )

        b_tile_dwords = block_n * k // 4

        sorted_ids_lds = fx.make_view(lds.sorted_ids.ptr, fx.make_layout(block_m, 1))
        sorted_weights_lds = fx.make_view(lds.sorted_weights.ptr, fx.make_layout(block_m, 1))

        # weight_shuffled:
        # [k16, n16, k/k16, n_split/n16, num_oc_splits, num_experts]
        assert k % 64 == 0
        assert n % 16 == 0
        k16 = 16 // (weight_shuffled.dtype.width // 8)
        n16 = 16
        weight_shuffled = fx.make_view(
            weight_shuffled,
            fx.make_ordered_layout(
                (
                    k16,
                    n16,
                    k // k16,
                    n_split // n16,
                    num_oc_splits,
                    num_experts,
                ),
                0,
            ),
        )

        # The Int32 ring replaces k16 with four contiguous dwords.
        num_slots = 4
        lds_b_ring = fx.make_view(
            lds.b_ring.ptr,
            fx.make_ordered_layout(
                (4, n16, k // k16, block_n // n16, num_slots),
                0,
            ),
        )

        # Activation scales are K-major because transpose_scale=True.
        input_scales = fxh.view_as_torch_tensor(
            input_scales,
            (k_blocks, num_tokens * fx.Int32(topk)),
            fx.Float32,
        )
        weight_scales = fxh.view_as_torch_tensor(
            weight_scales,
            (num_experts, scale_n_blocks, k_blocks),
            fx.Float32,
        )
        sorted_expert_ids = fxh.view_as_torch_tensor(
            sorted_expert_ids,
            (num_expert_blocks,),
            fx.Int32,
        )

        input_scales = fx.rocdl.make_buffer_tensor(input_scales, False)

        assert k % 16 == 0
        assert n % 8 == 0
        input_data = ptr_to_tensor(
            input_q,
            (16, k // 16, num_tokens * topk),
            fx.Int32,
            as_buffer=True,
        )
        output_data = ptr_to_tensor(
            output,
            (8, n // 8, num_tokens * topk),
            fx.Int32,
            as_buffer=True,
        )

        max_id = num_valid_ids[0]

        atom = fx.make_mma_atom(
            fx.rocdl.cdna4.MFMA_Scale(16, 16, 128, fx.Float8E4M3FN)
        )

        running = fx.Boolean(True)
        while running:
            if tid == fx.Int32(0):
                lds_task_id[0] = fx.Int32(_atomic_add_i32(fx.ptrtoint(task_counter), fx.Int32(1)))
            fx.barrier()
            task = lds_task_id[0]
            blk_m = task // fx.Int32(num_oc_splits)
            blk_oc = task % fx.Int32(num_oc_splits)
            running = blk_m * fx.Int32(block_m) < max_id
            if running:
                num_mma_m = 32 // 16
                num_mma_n = block_n // 16
                num_mma_k = k_blocks

                a_mma_frag_r = fx.make_rmem_tensor(
                    [8, num_mma_m, num_mma_k],
                    fx.Int32,
                )
                a_mma_frag_w = fx.make_view(
                    fx.get_iter(a_mma_frag_r),
                    fx.make_ordered_layout(
                        [4, 2, num_mma_m, num_mma_k],
                        0,
                    ),
                )
                a_scales_frag = fx.make_rmem_tensor([2, num_mma_k], fx.Float32)

                b_mma_frag_r = fx.make_rmem_tensor(
                    [8, num_mma_n, num_mma_k],
                    fx.Int32,
                )
                b_mma_frag_w = fx.make_view(
                    fx.get_iter(b_mma_frag_r),
                    fx.make_ordered_layout(
                        [4, 2, num_mma_n, num_mma_k],
                        0,
                    ),
                )
                # B uses one scale per 128x128 block.
                b_scales_frag = fx.make_rmem_tensor([num_mma_k], fx.Float32)

                acc = fx.make_rmem_tensor([4, num_mma_m, num_mma_n], fx.Float32)

                expert = fx.Int32(sorted_expert_ids[blk_m])

                sorted_ids_src = fx.make_view(
                    sorted_ids + blk_m * fx.Int32(block_m),
                    fx.make_layout(block_m, 1),
                )
                sorted_weights_src = fx.make_view(
                    sorted_weights + blk_m * fx.Int32(block_m),
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

                ROCDLBuffer(weight_scales).load_async(
                    fx.get_iter(weight_scales[expert, blk_oc * (n_split // 128), None]),
                    lds.scale_b.ptr,
                    scale_n_blocks_per_split * k_blocks,
                    num_threads,
                )

                # Select one [k16, n16, K/k16, n_split/n16] expert tile.
                weight_expert = weight_shuffled[None, None, None, None, blk_oc, expert]
                weight_rsrc = ROCDLBuffer(weight_expert)

                fx.rocdl.asyncmark()
                fx.rocdl.wait_asyncmark(0)
                fx.barrier()

                route_rows = {}
                valid_rows = {}
                output_rows = {}
                for mi in range_constexpr(num_mma_m):
                    a_row = wave * fx.Int32(32) + fx.Int32(mi * 16) + lane_mod16
                    packed = fx.Int32(sorted_ids_lds[a_row])
                    token = packed & fx.Int32(0xFFFFFF)
                    slot = packed >> fx.Int32(24)
                    output_row = token * fx.Int32(topk) + slot
                    valid_row = slot < fx.Int32(topk)
                    route_rows[mi] = sorted_weights_lds[a_row]
                    valid_rows[mi] = valid_row
                    output_rows[mi] = output_row

                for mi in range_constexpr(num_mma_m):
                    for kb in range_constexpr(num_mma_k):
                        for step in range_constexpr(2):
                            a_mma_frag_w[None, step, mi, kb] = input_data[
                                None,
                                lane_div16 + (kb * 128 + step * 64) // 16,
                                output_rows[mi],
                            ].load()

                for mi in range_constexpr(num_mma_m):
                    for kb in range_constexpr(num_mma_k):
                        if valid_rows[mi]:
                            a_scales_frag[mi, kb] = input_scales[kb, output_rows[mi]]

                def global_load_b(n_tile):
                    """Copy one preshuffled B tile directly into its LDS slot."""
                    weight_rsrc.load_async(
                        fx.get_iter(
                            weight_expert[
                                None,
                                None,
                                None,
                                n_tile * n16_tiles_per_block,
                            ]
                        ),
                        fx.get_iter(lds_b_ring[None, None, None, None, n_tile % 4]),
                        b_tile_dwords,
                        num_threads,
                    )

                def global_store(n_tile):
                    """
                        With physical operands ordered B,A, lane_mod16 selects the
                        routed M row and each accumulator vector contains four
                        contiguous N values. This is the same store-friendly
                        distribution used by PyHIP's vaddr_rows construction.                        
                    """
                    n_base = blk_oc * fx.Int32(n_split) + fx.Int32(n_tile * block_n)

                    for mi in range_constexpr(num_mma_m):
                        for ni in range_constexpr(0, num_mma_n, 2):
                            c0 = acc[None, mi, ni].load() * route_rows[mi]
                            c1 = acc[None, mi, ni+1].load() * route_rows[mi]
                            d0_a = rocdl.cvt_pk_bf16_f32(c0[0], c0[1])
                            d1_a = rocdl.cvt_pk_bf16_f32(c0[2], c0[3])
                            d0_b = rocdl.cvt_pk_bf16_f32(c1[0], c1[1])
                            d1_b = rocdl.cvt_pk_bf16_f32(c1[2], c1[3])
                            pair_type = ir.Type.parse("!llvm.struct<(i32, i32)>")
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
                            swap_12_col = (
                                (lane_div16 & fx.Int32(1)) * fx.Int32(2)
                                + (lane_div16 >> fx.Int32(1))
                            )
                            output_data[
                                None,
                                n_base // 8 + ni * 2 + swap_12_col,
                                output_rows[mi],
                            ] = output_values

                def ds_read_b(n_tile):
                    noff = n_tile * block_n
                    slot = n_tile % 4
                    for kb in range_constexpr(num_mma_k):
                        for ni in range_constexpr(num_mma_n):
                            for step in range_constexpr(2):
                                b_mma_frag_w[None, step, ni, kb] = lds_b_ring[
                                    None,
                                    lane_mod16,
                                    lane_div16 + (kb * 2 + step) * 4,
                                    ni,
                                    slot,
                                ].load()
                        b_scales_frag[kb] = lds_scale_b[noff // 128, kb]

                def compute():
                    acc_init = {}
                    dequant_queue = []
                    for kb in range_constexpr(num_mma_k):
                        for mi in range_constexpr(num_mma_m):
                            for ni in range_constexpr(num_mma_n):
                                c_mma_frag = fx.make_rmem_tensor(4, fx.Float32)
                                c_mma_frag.fill(0)
                                fx.gemm(atom, c_mma_frag, b_mma_frag_r[None, ni, kb], a_mma_frag_r[None, mi, kb], c_mma_frag)
                                rocdl.sched_barrier(0)

                                mfma_scaleAB = a_scales_frag[mi, kb] * b_scales_frag[kb]
                                dequant_queue.append([c_mma_frag, mfma_scaleAB, (mi, ni)])

                                if len(dequant_queue) > 3:
                                    c_mma_frag, mfma_scaleAB, (mi0, ni0) = dequant_queue.pop(0)
                                    if const_expr((mi0, ni0) in acc_init):
                                        acc[None, mi0, ni0].store(acc[None, mi0, ni0].load() + c_mma_frag.load() * mfma_scaleAB)
                                    else:
                                        acc[None, mi0, ni0].store(c_mma_frag.load() * mfma_scaleAB)
                                        acc_init[mi0, ni0] = True
                                rocdl.sched_barrier(0)

                    while const_expr(len(dequant_queue) > 0):
                        c_mma_frag, mfma_scaleAB, (mi0, ni0) = dequant_queue.pop(0)
                        if const_expr((mi0, ni0) in acc_init):
                            acc[None, mi0, ni0].store(acc[None, mi0, ni0].load() + c_mma_frag.load() * mfma_scaleAB)
                        else:
                            acc[None, mi0, ni0].store(c_mma_frag.load() * mfma_scaleAB)
                            acc_init[mi0, ni0] = True

                """
                 如果只依赖 MFMA operand dependency，那么确实只保证 lgkmcnt 在 MFMA 前，不能保证在 barrier 前；
                 fx.barrier() 不是裸 barrier指令，它的 fence 语义使 LLVM 把自动等待放到了 barrier 前；
                 当前显式 fxh.s_waitcnt(lgkmcnt=0) 是防御性流水线约束；避免global加载破坏正在被ds_read访问的LDS内容；
                """
                global_load_b(0)
                fx.rocdl.asyncmark()

                global_load_b(1)
                fx.rocdl.asyncmark()

                fx.rocdl.wait_asyncmark(1)

                if wave > fx.Int32(3):
                    fx.barrier()
                
                fx.barrier()

                ds_read_b(0)
                global_load_b(2)

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

                    ds_read_b(n_tile + 1)
                    global_store(n_tile)
                    global_load_b(n_tile + 3)

                    fx.rocdl.asyncmark()
                    fx.rocdl.wait_asyncmark(1)
                    fxh.s_waitcnt(lgkmcnt=0)
                    fx.barrier()

                    compute()

                fx.barrier()
                ds_read_b(n_tiles_per_split - 2)
                global_store(n_tiles_per_split - 3)

                fx.rocdl.asyncmark()
                fx.rocdl.wait_asyncmark(1)
                fxh.s_waitcnt(lgkmcnt=0)
                fx.barrier()

                compute()

                fx.barrier()
                ds_read_b(n_tiles_per_split - 1)
                global_store(n_tiles_per_split - 2)

                fxh.s_waitcnt(lgkmcnt=0)
                fx.barrier()

                compute()
                global_store(n_tiles_per_split - 1)

                if wave < fx.Int32(4):
                    fx.barrier()
            fx.barrier()

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
        flydsl_moe_gemm_8wave_down_kernel(
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
            value_attrs={"rocdl.waves_per_eu": 2, "rocdl.flat_work_group_size": "512,512"},
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
        runtime_num_experts, output_n, weight_k = weight_shuffled.shape
        assert input_topk == topk and input_k == k
        assert runtime_num_experts == num_experts
        assert output_n == n and weight_k == k
        assert output.shape == (num_tokens, topk, n)
        assert input_scales.numel() == num_tokens * topk * k_blocks
        assert weight_scales.shape == (num_experts, scale_n_blocks, k_blocks)
        assert sorted_weights.shape == sorted_ids.shape

        task_counter.zero_()
        stream = torch.cuda.current_stream()
        _run_compiled(
            launch,
            _ptr(output),
            _ptr(input_q),
            _ptr(weight_shuffled),
            _ptr(input_scales),
            _ptr(weight_scales),
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
