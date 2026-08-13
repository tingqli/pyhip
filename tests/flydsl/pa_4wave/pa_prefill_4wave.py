import functools
import os

os.environ.setdefault("FLYDSL_RUNTIME_ENABLE_CACHE", "0")

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import arith, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import as_ir_value

LOG2E = 1.4426950408889634
_SCHED_MASK_DS_WRITE = 0x200
_SCHED_MASK_TRANS = 0x400
_EXP_DSWR_SYNC_ID = 1


def _tensor_signature(tensor):
    return (
        tensor.dtype,
        tensor.device.type,
        tensor.device.index,
        tuple(tensor.shape),
        tuple(tensor.stride()),
    )


def _maxnumf(lhs, rhs):
    return type(lhs)(arith.maxnumf(arith.unwrap(lhs), arith.unwrap(rhs)))


def _exp2_f32(value):
    from flydsl._mlir.ir import F32Type

    return fx.Float32(llvm.call_intrinsic(F32Type.get(), "llvm.amdgcn.exp2.f32", [arith.unwrap(value)], [], []))


def _exp2_vec_f32(values):
    from flydsl._mlir.dialects import vector
    from flydsl._mlir.ir import F32Type, VectorType

    raw = arith.unwrap(values)
    f32 = F32Type.get()
    result = [llvm.call_intrinsic(
        f32, "llvm.amdgcn.exp2.f32",
        [vector.extract(raw, static_position=[index], dynamic_position=[])], [], [],
    ) for index in range(raw.type.shape[0])]
    return fx.Vector(vector.from_elements(VectorType.get([len(result)], f32), result))


def _fma_f32(lhs, rhs, acc, negate_acc=False):
    from flydsl._mlir.ir import F32Type

    instruction = "v_fma_f32 $0, $1, $2, -$3" if negate_acc else "v_fma_f32 $0, $1, $2, $3"
    return fx.Float32(llvm.inline_asm(
        F32Type.get(), [arith.unwrap(lhs), arith.unwrap(rhs), arith.unwrap(acc)],
        instruction, "=v,v,v,v", has_side_effects=False,
    ))


def _read_hw_wave_slot():
    return fx.Int32(llvm.inline_asm(
        fx.Int32.ir_type, [], "s_getreg_b32 $0, hwreg(HW_REG_HW_ID, 0, 4)",
        "=s", has_side_effects=True,
    ))


def _set_hw_slot_priority(wave_slot, slot0_priority, slot1_priority):
    llvm.inline_asm(
        ir.Type.parse("!llvm.void"), [arith.unwrap(wave_slot)],
        (
            "s_cmp_eq_u32 $0, 0\n\t"
            "s_cbranch_scc0 1f\n\t"
            f"s_setprio {slot0_priority}\n\t"
            "s_branch 2f\n\t"
            "1:\n\t"
            f"s_setprio {slot1_priority}\n\t"
            "2:"
        ),
        "s", has_side_effects=True,
    )


def _cvt_f32_to_bf16(fragment):
    result = fx.make_fragment_like(fragment, dtype=fx.BFloat16)
    result.store(((fragment.load().bitcast(fx.Uint32) + fx.Uint32(0x8000)) >> 16).to(fx.Uint16).bitcast(fx.BFloat16))
    return result


def _pack_f32x4_to_fp8(values):
    packed = fx.Int32(0)
    packed = fx.Int32(rocdl.cvt_pk_fp8_f32(fx.Int32.ir_type, values[0], values[1], packed, False))
    return fx.Int32(rocdl.cvt_pk_fp8_f32(fx.Int32.ir_type, values[2], values[3], packed, True))


def _pack_probability_fp8(probability, start):
    values = fx.Vector.from_elements(
        [probability[start + offset] for offset in range_constexpr(4)], fx.Float32
    )
    return fx.Vector.from_elements([_pack_f32x4_to_fp8(values)], fx.Int32).bitcast(fx.Float8E4M3FNUZ)


@flyc.jit
def _rescale_accumulator_if_needed(output_accumulator, correction):
    if correction < fx.Float32(1.0):
        output_accumulator.store(output_accumulator.load() * correction)


def _store_fp8_probability(score_fragment, probability_operand):
    probability = score_fragment.load()
    for k_group in range_constexpr(2):
        start = k_group * 8
        probability_lo = _pack_probability_fp8(probability, start)
        probability_hi = _pack_probability_fp8(probability, start + 4)
        probability_operand[None, 0, k_group].store(
            probability_lo.shuffle(probability_hi, list(range(8)))
        )


def _store_bf16_probability(score_fragment, probability_storage):
    probability_storage.store(_cvt_f32_to_bf16(score_fragment).load())


def _make_fp8_epilogue_tid(tid, running_sum):
    return fx.Int32(
        llvm.inline_asm(
            fx.Int32.ir_type,
            [arith.unwrap(tid), arith.unwrap(running_sum)],
            "v_and_or_b32 $0, $2, 0, $1",
            "=v,v,v",
            has_side_effects=False,
        )
    )


def _schedule_qk_bf16_d128(num_v_loads, head_dim_qk, mma_k):
    for _ in range_constexpr(num_v_loads):
        rocdl.sched_vmem(1)
        rocdl.sched_mfma(1)
    rocdl.sched_mfma(head_dim_qk // mma_k - num_v_loads)


def _schedule_qk_bf16_d192(num_v_loads):
    for _ in range_constexpr(num_v_loads):
        rocdl.sched_vmem(1)
        rocdl.sched_mfma(3)


def _schedule_qk_fp8(num_v_loads):
    for _ in range_constexpr(num_v_loads):
        rocdl.sched_vmem(1)
        rocdl.sched_mfma(2)


def _schedule_pv_bf16(num_k_copies, num_k_reads, head_dim_v, mma_k):
    for _ in range_constexpr(num_k_copies):
        rocdl.sched_vmem(1)
        rocdl.sched_dswr(1)
    rocdl.sched_mfma(3)
    for _ in range_constexpr(num_k_reads):
        rocdl.sched_dsrd(1)
        rocdl.sched_mfma(1)
    rocdl.sched_mfma(head_dim_v // mma_k - num_k_reads - 3)


def _schedule_pv_fp8(num_k_reads):
    rocdl.sched_vmem(1)
    rocdl.sched_dswr(1)
    rocdl.sched_mfma(7)
    rocdl.sched_vmem(1)
    rocdl.sched_mfma(3)
    rocdl.sched_dswr(1)
    rocdl.sched_mfma(4)
    for _ in range_constexpr(num_k_reads):
        rocdl.sched_dsrd(1)
        rocdl.sched_mfma(1)


def _s_waitcnt(vmcnt=63, expcnt=7, lgkmcnt=63):
    vm_lo = vmcnt & 0xF
    vm_hi = (vmcnt >> 4) & 0x3
    rocdl.s_waitcnt(vm_lo | (expcnt << 4) | (lgkmcnt << 8) | (vm_hi << 14))


def _schedule_fence():
    rocdl.sched_barrier(0)


def _schedule_ds_write(count, sync_id=_EXP_DSWR_SYNC_ID):
    rocdl.sched_group_barrier(_SCHED_MASK_DS_WRITE, count, sync_id)


def _schedule_trans(count, sync_id=_EXP_DSWR_SYNC_ID):
    rocdl.sched_group_barrier(_SCHED_MASK_TRANS, count, sync_id)


def _recast_tensor(tensor, dtype):
    pointer_type = fx.PointerType.get(dtype.ir_type, tensor.memspace, dtype.width // 8)
    iterator = fx.recast_iter(pointer_type, fx.get_iter(tensor))
    layout = fx.recast_layout(tensor.layout, tensor.dtype.width, dtype.width)
    return fx.make_view(iterator, layout)


def _prepare_paged_v_tile(v_tile, permute_bf16_tokens: fx.Constexpr[bool]):
    if const_expr(v_tile.dtype == fx.BFloat16 and permute_bf16_tokens):
        token_permutation = fx.make_layout((8, (2, 2)), (1, (16, 8)))
        v_tile = fx.composition(v_tile, fx.make_tile(None, token_permutation, None))
    return fx.rocdl.make_buffer_tensor(v_tile, max_size=False)


def _compile_hints_for_dtype(dtype):
    return {"fast_fp_math": True} if dtype == torch.float8_e4m3fnuz else {}


@flyc.jit
def _online_softmax(
    score_fragment,
    output_accumulator,
    qk_scale_log2,
    running_max,
    running_sum,
    query_tile_start,
    kv_block_index,
    kv_len,
    query_sequence_length,
    all_kv_valid: fx.Constexpr[bool],
    is_causal: fx.Constexpr[bool],
    split_score_scaling: fx.Constexpr[bool],
    defer_output_rescale: fx.Constexpr[bool],
    interleaved_score_columns: fx.Constexpr[bool],
):
    if const_expr(not all_kv_valid):
        lane_id = fx.thread_idx.x & 63
        column_base = (lane_id < 32).select(fx.Int32(0), fx.Int32(16))
        lane_column_group = (lane_id < 32).select(fx.Int32(0), fx.Int32(8))
        block_base = fx.Int32(kv_block_index * 32)
        if const_expr(is_causal):
            wave_id = fx.thread_idx.x // 64
            query_row = fx.thread_idx.x & 31
            query_position = query_tile_start + wave_id * 32 + query_row
            causal_limit = kv_len - query_sequence_length + query_position
            for index in range_constexpr(16):
                if const_expr(interleaved_score_columns):
                    column = lane_column_group + fx.Int32((index // 8) * 16 + index % 8)
                else:
                    column = column_base + fx.Int32(index)
                if block_base + column > causal_limit:
                    score_fragment[index, 0, 0] = float("-inf")
        else:
            for index in range_constexpr(16):
                if const_expr(interleaved_score_columns):
                    column = lane_column_group + fx.Int32((index // 8) * 16 + index % 8)
                else:
                    column = column_base + fx.Int32(index)
                if block_base + column >= kv_len:
                    score_fragment[index, 0, 0] = float("-inf")

    score = score_fragment.load()
    # Q/K scale is positive, so the cross-lane max can run before scaling.
    row_max = score.reduce("max")
    shuffled_row_max = row_max.shuffle_xor(32, 64)
    if const_expr(split_score_scaling):
        _schedule_fence()
    if const_expr(split_score_scaling):
        scaled_values = [
            score[index] * qk_scale_log2 for index in range_constexpr(11)
        ]
        _schedule_fence()
        row_max = _maxnumf(row_max, shuffled_row_max)
        scaled_values.extend(
            [
                score[index] * qk_scale_log2 for index in range_constexpr(11, score.numel)
            ]
        )
        scaled_score = fx.Vector.from_elements(scaled_values, fx.Float32)
    else:
        row_max = _maxnumf(row_max, shuffled_row_max)
        scaled_score = fx.Vector.from_elements(
            [
                _fma_f32(score[index], qk_scale_log2, fx.Float32(0.0))
                for index in range_constexpr(score.numel)
            ],
            fx.Float32,
        )
    if const_expr(split_score_scaling):
        row_max = row_max * qk_scale_log2
    else:
        row_max = _fma_f32(row_max, qk_scale_log2, fx.Float32(0.0))

    updated_max = running_max
    correction = fx.Float32(1.0)
    if row_max > running_max + fx.Float32(7.0):
        updated_max = row_max + fx.Float32(1.0)
        correction = _exp2_f32(running_max - updated_max)

    probability = _exp2_vec_f32(scaled_score - updated_max)
    tile_sum = probability.reduce("add")
    updated_sum = _fma_f32(running_sum, correction, tile_sum)
    score_fragment.store(probability)

    if const_expr(not defer_output_rescale):
        _rescale_accumulator_if_needed(output_accumulator, correction)
    return updated_max, updated_sum, correction


@functools.cache
def MHA(
    num_qo_heads,
    num_kv_heads,
    head_dim_qk,
    head_dim_v,
    page_size,
    is_causal,
    key_layout="vectorized",
):
    assert head_dim_qk in (128, 192)
    assert head_dim_v == 128
    assert page_size == 32
    assert num_qo_heads % num_kv_heads == 0
    assert key_layout in ("vectorized", "linear")
    if key_layout == "linear":
        assert head_dim_qk == head_dim_v == 128

    block_m = 128
    block_n = 32
    num_threads = 256
    causal_tile_step = 251
    causal_tile_offset = 251
    qk_scale_log2_base = float(LOG2E / (head_dim_qk**0.5))

    @flyc.jit
    def attention_pipeline(
        q_tile,
        k_tile,
        v_tile,
        o_tile,
        query_pos0,
        query_len,
        kv_len,
        full_qo_len,
        kv_page_table,
        num_kv_pages,
        kv_sequence_start,
        kv_head,
        qk_scale_log2,
        v_scale,
        requires_epilogue_reentry_barrier: fx.Constexpr[bool],
    ):
        tid = fx.thread_idx.x
        dtype = q_tile.dtype
        is_fp8 = dtype == fx.Float8E4M3FNUZ
        is_bf16 = dtype == fx.BFloat16
        mma_k = 8 if is_bf16 else 16
        use_hw_slot_priority = is_bf16 and head_dim_qk == 128
        interleave_exp_ds_write = (
            is_bf16 and head_dim_qk == 128 and num_qo_heads >= 4
        )
        hw_wave_slot = _read_hw_wave_slot() if const_expr(use_hw_slot_priority) else None

        def enter_softmax_stage():
            _schedule_fence()
            if const_expr(use_hw_slot_priority):
                _set_hw_slot_priority(hw_wave_slot, 1, 0)
            else:
                rocdl.s_setprio(0)
            _schedule_fence()

        def enter_mma_stage():
            _schedule_fence()
            if const_expr(use_hw_slot_priority):
                _set_hw_slot_priority(hw_wave_slot, 3, 2)
            else:
                rocdl.s_setprio(2)
            _schedule_fence()

        mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(32, 32, mma_k, dtype))
        atom_values = mma_k // 2
        vector_values = 128 // dtype.width
        k_permutation = fx.make_layout((atom_values, 2, 2), (1, vector_values, atom_values))
        wave_layout = fx.make_layout((1, 4, 1), (1, 1, 0))
        mma_tile = fx.make_tile(None, None, k_permutation)
        qk_tiled_mma = fx.make_tiled_mma(mma_atom, wave_layout, mma_tile)
        pv_tiled_mma = fx.make_tiled_mma(mma_atom, wave_layout, mma_tile)
        qk_thread_mma = qk_tiled_mma.thr_slice(tid)
        pv_thread_mma = pv_tiled_mma.thr_slice(tid)

        q_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), dtype)
        q_thread_copy = fx.make_tiled_copy_B(q_copy_atom, qk_tiled_mma).get_slice(tid)
        q_fragment = qk_thread_mma.make_fragment_B(q_tile)
        fx.copy(q_copy_atom, q_thread_copy.partition_S(q_tile), q_thread_copy.retile(q_fragment))

        k_mma_tile = fx.Tensor(fx.make_view(
            fx.get_iter(k_tile), fx.make_layout((block_n, head_dim_qk), (head_dim_qk, 1))
        ))
        v_mma_tile = fx.Tensor(fx.make_view(
            fx.get_iter(v_tile), fx.make_layout((head_dim_v, block_n), (block_n, 1))
        ))
        k_fragment = qk_thread_mma.make_fragment_A(k_mma_tile)
        v_fragment = pv_thread_mma.make_fragment_A(v_mma_tile)
        k_fragment.fill(0)
        v_fragment.fill(0)
        score_fragment = qk_thread_mma.make_fragment_C(
            fx.make_rmem_tensor(fx.make_layout((block_n, block_m), (block_m, 1)), fx.Float32)
        )
        transposed_output_tile = fx.select(o_tile, [1, 0])
        output_accumulator = pv_thread_mma.make_fragment_C(transposed_output_tile)
        if const_expr(is_bf16):
            probability_storage = fx.make_fragment_like(score_fragment, dtype=fx.BFloat16)
            probability_operand = fx.make_view(
                fx.get_iter(probability_storage), fx.make_layout((4, 1, (2, 2)), (1, 0, (4, 8)))
            )
        else:
            probability_operand = pv_thread_mma.make_fragment_B(fx.make_rmem_tensor(
                fx.make_layout((block_m, block_n), (block_n, 1)), dtype
            ))
        score_fragment.fill(0)
        probability_operand.fill(0)

        k_lds_stride = head_dim_qk + (8 if is_bf16 else 16)

        @fx.union
        class SharedStorage:
            k_lds: fx.Array[dtype, 2 * block_n * k_lds_stride, 16]
            o_lds: fx.Array[fx.BFloat16, block_m * (head_dim_v // 2), 16]

        shared = fx.SharedAllocator().allocate(SharedStorage)
        k_lds_layout = fx.make_layout(
            (block_n, head_dim_qk, 2), (k_lds_stride, 1, block_n * k_lds_stride)
        )
        k_lds_storage = fx.make_view(shared.k_lds.peek().ptr, k_lds_layout)
        if const_expr(is_bf16):
            k_row_permutation = fx.make_layout((4, 2, 4), (1, 16, 4))
            k_lds = fx.composition(k_lds_storage, fx.make_tile(k_row_permutation, None, None))
        else:
            k_lds = k_lds_storage
        output_swizzle = fx.SwizzleType.get(3, 3, 3)
        o_lds_store = shared.o_lds.peek().view(fx.make_composed_layout(
            fx.static(output_swizzle), fx.make_ordered_layout((head_dim_v // 2, block_m), (0, 1))
        ))
        o_lds_read = shared.o_lds.peek().view(fx.make_composed_layout(
            fx.static(output_swizzle), fx.make_ordered_layout((block_m, head_dim_v // 2), (1, 0))
        ))

        if const_expr(is_bf16):
            k_global_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), dtype)
            k_lds_store_atom = fx.make_copy_atom(fx.UniversalCopy128b(), dtype)
            num_k_copies = head_dim_qk // 64
            prefetched_k = [
                fx.make_rmem_tensor(fx.make_layout((8, num_k_copies), (1, 8)), dtype),
                fx.make_rmem_tensor(fx.make_layout((8, num_k_copies), (1, 8)), dtype),
            ]

            def prefetch_k_bf16(logical_page_id, physical_page_id, register_slot):
                logical_page_id = fx.Int32(arith.minsi(
                    arith.unwrap(fx.Int32(logical_page_id)), arith.unwrap(num_kv_pages - 1)
                ))
                for atom_index in range_constexpr(num_k_copies):
                    linear_atom = tid + atom_index * num_threads
                    source_row = linear_atom & (block_n - 1)
                    d_group = linear_atom // block_n
                    if const_expr(key_layout == "linear"):
                        source_offset = (
                            (kv_sequence_start + logical_page_id * block_n + source_row)
                            * num_kv_heads * head_dim_qk
                            + kv_head * head_dim_qk
                            + d_group * vector_values
                        )
                    else:
                        source_offset = (
                            physical_page_id * page_size * num_kv_heads * head_dim_qk
                            + kv_head * page_size * head_dim_qk
                            + d_group * page_size * vector_values
                            + source_row * vector_values
                        )
                    source = fx.make_view(
                        fx.get_iter(k_tile) + source_offset, fx.make_layout(8, 1)
                    )
                    fx.copy(k_global_copy_atom, source, prefetched_k[register_slot][None, atom_index])

            def store_k_to_lds_bf16(register_slot, lds_slot):
                for atom_index in range_constexpr(num_k_copies):
                    linear_atom = tid + atom_index * num_threads
                    source_row = linear_atom & (block_n - 1)
                    d_group = linear_atom // block_n
                    destination_offset = ((lds_slot & 1) * block_n * k_lds_stride
                                          + source_row * k_lds_stride + d_group * vector_values)
                    destination = fx.make_view(
                        fx.get_iter(k_lds_storage) + destination_offset, fx.make_layout(8, 1)
                    )
                    fx.copy(k_lds_store_atom, prefetched_k[register_slot][None, atom_index], destination)

            prefetch_k = prefetch_k_bf16
            store_k_to_lds = store_k_to_lds_bf16
        else:
            k_global_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy64b(), fx.Uint32)
            k_lds_store_atom = fx.make_copy_atom(fx.UniversalCopy64b(), fx.Uint32)
            k_tile_u32 = _recast_tensor(k_tile, fx.Uint32)
            num_k_copies = head_dim_qk // 64
            prefetched_k = [
                fx.make_rmem_tensor(fx.make_layout((2, num_k_copies), (1, 2)), fx.Uint32),
                fx.make_rmem_tensor(fx.make_layout((2, num_k_copies), (1, 2)), fx.Uint32),
            ]
            k_row = tid // 8
            k_chunk_in_group = tid & 7
            k_source_row = (k_row & 3) + ((k_row // 4) & 1) * 16 + (k_row // 8) * 4

            def prefetch_k_fp8(logical_page_id, physical_page_id, register_slot):
                prefetched_k[register_slot].fill(0)
                for atom_index in range_constexpr(num_k_copies):
                    chunk = k_chunk_in_group + atom_index * 8
                    d_group = chunk // 2
                    d_half = chunk & 1
                    source_offset = (
                        physical_page_id * page_size * num_kv_heads * head_dim_qk
                        + kv_head * page_size * head_dim_qk
                        + d_group * page_size * 16
                        + k_source_row * 16 + d_half * 8
                    )
                    source = fx.make_view(
                        fx.get_iter(k_tile_u32) + source_offset // 4, fx.make_layout(2, 1)
                    )
                    fx.copy(k_global_copy_atom, source, prefetched_k[register_slot][None, atom_index])

            def store_k_to_lds_fp8(register_slot, lds_slot):
                for atom_index in range_constexpr(num_k_copies):
                    chunk = k_chunk_in_group + atom_index * 8
                    destination_offset = ((lds_slot & 1) * block_n * k_lds_stride
                                          + k_row * k_lds_stride + chunk * 8)
                    destination = fx.make_view(
                        fx.get_iter(k_lds) + destination_offset, fx.make_layout(8, 1)
                    )
                    fx.copy(k_lds_store_atom, prefetched_k[register_slot][None, atom_index],
                            _recast_tensor(destination, fx.Uint32))

            prefetch_k = prefetch_k_fp8
            store_k_to_lds = store_k_to_lds_fp8

        k_lds_copy_atom = fx.make_copy_atom(fx.UniversalCopy128b(), dtype)
        k_lds_copy = fx.make_tiled_copy_A(k_lds_copy_atom, qk_tiled_mma).get_slice(tid)

        def partition_k_lds(lds_slot):
            return k_lds_copy.partition_S(k_lds[None, None, lds_slot])

        v_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), dtype)
        v_copy = fx.make_tiled_copy_A(v_copy_atom, pv_tiled_mma).get_slice(tid)
        num_v_loads = fx.size(v_fragment.shape).get_static_leaf_int * v_fragment.dtype.width // 128
        num_k_fragment_bits = fx.size(k_fragment.shape).get_static_leaf_int * k_fragment.dtype.width

        output_accumulator.fill(0.0)

        def compute_qk():
            for k_group in range_constexpr(head_dim_qk // (2 * mma_k)):
                for k_atom in range_constexpr(2):
                    accumulator = score_fragment[None, 0, 0]
                    fx.mma_atom_call(
                        mma_atom, accumulator, k_fragment[None, 0, (k_atom, k_group)],
                        q_fragment[None, 0, (k_atom, k_group)], accumulator,
                    )

        def schedule_qk_and_v_loads():
            if const_expr(is_bf16 and head_dim_qk == 128):
                _schedule_qk_bf16_d128(num_v_loads, head_dim_qk, mma_k)
            elif const_expr(is_fp8):
                _schedule_qk_fp8(num_v_loads)
            else:
                _schedule_qk_bf16_d192(num_v_loads)
            rocdl.sched_vmem(100)
            rocdl.sched_mfma(100)

        def schedule_pv_and_next_k():
            num_k_reads = num_k_fragment_bits // 128
            if const_expr(is_bf16):
                _schedule_pv_bf16(num_k_copies, num_k_reads, head_dim_v, mma_k)
            else:
                _schedule_pv_fp8(num_k_reads)
            _schedule_fence()

        def process_kv_block(
            kv_block_index,
            k_pipeline_slot,
            running_max,
            running_sum,
            current_v_page_id,
            prefetch_k_page_id,
            is_all_kv_valid: fx.Constexpr[bool] = True,
        ):
            lookahead_page_id = kv_page_table[kv_block_index + 3]

            score_fragment.fill(0.0)
            compute_qk()
            fx.copy(
                v_copy_atom,
                v_copy.partition_S(v_tile[None, None, current_v_page_id]),
                v_copy.retile(v_fragment),
            )
            schedule_qk_and_v_loads()

            enter_softmax_stage()
            prefetch_k(
                kv_block_index + 2, prefetch_k_page_id, k_pipeline_slot ^ 1
            )

            running_max, running_sum, correction = _online_softmax(
                score_fragment, output_accumulator, qk_scale_log2, running_max, running_sum,
                query_pos0, kv_block_index, kv_len, full_qo_len, is_all_kv_valid, is_causal,
                is_fp8,
                interleave_exp_ds_write,
                False,
            )

            store_k_to_lds(k_pipeline_slot, k_pipeline_slot ^ 1)
            if const_expr(interleave_exp_ds_write):
                _schedule_trans(16)
                _schedule_ds_write(1)
                _schedule_trans(1)
                _schedule_ds_write(1)

            if const_expr(interleave_exp_ds_write):
                _rescale_accumulator_if_needed(output_accumulator, correction)
            if const_expr(is_fp8):
                _store_fp8_probability(score_fragment, probability_operand)
            else:
                _store_bf16_probability(score_fragment, probability_storage)

            enter_mma_stage()

            fx.gemm(
                mma_atom, output_accumulator, v_fragment,
                probability_operand, output_accumulator,
            )

            gpu.barrier()
            fx.copy(
                k_lds_copy_atom, partition_k_lds(k_pipeline_slot ^ 1),
                k_lds_copy.retile(k_fragment),
            )
            schedule_pv_and_next_k()

            return running_max, running_sum, lookahead_page_id

        current_max = fx.Float32(float("-inf"))
        running_sum = fx.Float32(0.0)
        current_page_id = kv_page_table[0]
        next_page_id = kv_page_table[1]
        prefetch_page_id = kv_page_table[2]

        prefetch_k(0, current_page_id, 0)
        store_k_to_lds(0, 0)
        prefetch_k(1, next_page_id, 0)
        gpu.barrier()
        fx.copy(k_lds_copy_atom, partition_k_lds(0), k_lds_copy.retile(k_fragment))
        enter_mma_stage()

        if const_expr(is_causal):
            causal_base = kv_len - full_qo_len + query_pos0
            num_fully_valid_pages = (causal_base + 1) // block_n
            num_fast_path_pages = (num_fully_valid_pages // 2) * 2
            num_intersecting_pages = (causal_base + query_len + block_n - 1) // block_n
            num_pages_to_process = (num_intersecting_pages < num_kv_pages).select(
                num_intersecting_pages, num_kv_pages
            )
        else:
            num_fast_path_pages = num_kv_pages - 2
            if (num_kv_pages & 1) == 1:
                num_fast_path_pages = num_kv_pages - 1
            num_pages_to_process = num_kv_pages

        loop_state = [current_max, running_sum, current_page_id, next_page_id, prefetch_page_id]
        for kv_block_index, state in range(0, num_fast_path_pages, 2, init=loop_state):
            current_max, running_sum, current_page_id, next_page_id, prefetch_page_id = state
            current_max, running_sum, lookahead_page_id = process_kv_block(
                kv_block_index, 0, current_max, running_sum, current_page_id, prefetch_page_id
            )
            current_page_id, next_page_id, prefetch_page_id = (
                next_page_id, prefetch_page_id, lookahead_page_id
            )
            current_max, running_sum, lookahead_page_id = process_kv_block(
                kv_block_index + 1, 1, current_max, running_sum,
                current_page_id, prefetch_page_id,
            )
            current_page_id, next_page_id, prefetch_page_id = (
                next_page_id, prefetch_page_id, lookahead_page_id
            )
            loop_state = yield [
                current_max, running_sum, current_page_id, next_page_id, prefetch_page_id
            ]

        for kv_block_index, state in range(
            num_fast_path_pages, num_pages_to_process, 2, init=loop_state
        ):
            current_max, running_sum, current_page_id, next_page_id, prefetch_page_id = state
            current_max, running_sum, lookahead_page_id = process_kv_block(
                kv_block_index, 0, current_max, running_sum,
                current_page_id, prefetch_page_id, is_all_kv_valid=False,
            )
            current_page_id, next_page_id, prefetch_page_id = (
                next_page_id, prefetch_page_id, lookahead_page_id
            )
            if fx.Int32(kv_block_index + 1) < num_pages_to_process:
                current_max, running_sum, lookahead_page_id = process_kv_block(
                    kv_block_index + 1, 1, current_max, running_sum,
                    current_page_id, prefetch_page_id,
                    is_all_kv_valid=False,
                )
                current_page_id, next_page_id, prefetch_page_id = (
                    next_page_id, prefetch_page_id, lookahead_page_id
                )
            loop_state = yield [
                current_max, running_sum, current_page_id, next_page_id, prefetch_page_id
            ]

        running_sum = loop_state[1]
        denominator = running_sum + running_sum.shuffle_xor(32, 64)
        output_accumulator.store(output_accumulator.load() * (v_scale / denominator))
        output_fragment_bf16 = _cvt_f32_to_bf16(output_accumulator)
        epilogue_tid = tid
        if const_expr(is_fp8):
            epilogue_tid = _make_fp8_epilogue_tid(tid, running_sum)
        cshuffle_store_atom = fx.make_copy_atom(fx.UniversalCopy64b(), fx.BFloat16)
        cshuffle_store = fx.make_tiled_copy_C(cshuffle_store_atom, pv_tiled_mma).get_slice(epilogue_tid)
        cshuffle_read_atom = fx.make_copy_atom(fx.UniversalCopy128b(), fx.BFloat16)
        cshuffle_read = fx.make_tiled_copy_tv(
            cshuffle_read_atom, fx.make_layout((32, 8), (8, 1)), fx.make_layout((4, 8), (8, 1))
        ).get_slice(epilogue_tid)
        store_source_halves = fx.logical_divide(
            cshuffle_store.retile(output_fragment_bf16), (None, 2, None)
        )
        store_destination = cshuffle_store.partition_D(o_lds_store)
        read_source = cshuffle_read.partition_S(o_lds_read)
        output_halves = fx.logical_divide(o_tile, (None, head_dim_v // 2))
        output_fragment = fx.make_fragment_like(read_source)
        output_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16)

        gpu.barrier()
        for half in range_constexpr(2):
            fx.copy(
                cshuffle_store_atom, store_source_halves[None, (None, half), None], store_destination
            )
            gpu.barrier()
            fx.copy(cshuffle_read_atom, read_source, output_fragment)
            if const_expr(half == 0 or requires_epilogue_reentry_barrier):
                gpu.barrier()
            fx.copy(
                output_copy_atom, output_fragment,
                cshuffle_read.partition_D(output_halves[None, (None, half)]),
            )

    @flyc.jit
    def process_work_item(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        kv_indptr,
        kv_page_indices,
        q_descale,
        kv_last_page_lens,
        output,
        batch_index,
        head_index,
        query_tile_index,
        tid,
        k_scale,
        v_scale,
        requires_epilogue_reentry_barrier: fx.Constexpr[bool],
    ):
        query_pos0 = query_tile_index * block_m
        query_start = cu_seqlens_q[batch_index] + query_pos0
        query_end = fx.Int32(arith.minsi(
            arith.unwrap(query_start + block_m), arith.unwrap(cu_seqlens_q[batch_index + 1])
        ))
        query_len = query_end - query_start
        full_qo_len = cu_seqlens_q[batch_index + 1] - cu_seqlens_q[batch_index]
        kv_start = kv_indptr[batch_index]
        num_kv_pages = kv_indptr[batch_index + 1] - kv_start
        if const_expr(key_layout == "linear"):
            kv_len = cu_seqlens_k[batch_index + 1] - cu_seqlens_k[batch_index]
        else:
            kv_len = (num_kv_pages - 1) * page_size + kv_last_page_lens[batch_index]

        qo_head = head_index
        kv_head = (qo_head * num_kv_heads) // num_qo_heads

        q_tile = fx.make_view(
            fx.get_iter(q) + query_start * num_qo_heads * head_dim_qk,
            fx.make_ordered_layout((block_m, num_qo_heads, head_dim_qk), (2, 1, 0)),
        )
        q_tile = fx.rocdl.make_buffer_tensor(
            q_tile, max_size=False,
            num_records_bytes=query_len * num_qo_heads * head_dim_qk * (q_tile.dtype.width // 8),
        )[None, qo_head, None]

        q_scale_tile = fx.make_view(
            fx.get_iter(q_descale) + query_start * num_qo_heads,
            fx.make_ordered_layout((block_m, num_qo_heads), (1, 0)),
        )
        q_scale_tile = fx.rocdl.make_buffer_tensor(
            q_scale_tile, max_size=False, num_records_bytes=query_len * num_qo_heads * 4
        )[None, qo_head]
        query_row = (tid // 64) * 32 + (tid & 31)
        qk_scale_log2 = q_scale_tile[query_row] * k_scale * fx.Float32(qk_scale_log2_base)

        o_tile = fx.make_view(
            fx.get_iter(output) + query_start * num_qo_heads * head_dim_v,
            fx.make_ordered_layout((block_m, num_qo_heads, head_dim_v), (2, 1, 0)),
        )
        o_tile = fx.rocdl.make_buffer_tensor(
            o_tile, max_size=False, num_records_bytes=query_len * num_qo_heads * head_dim_v * 2
        )[None, qo_head, None]

        k_tile = fx.rocdl.make_buffer_tensor(k, max_size=False)
        v_tile = v[None, kv_head, None, None, None]
        v_tile = fx.group(fx.select(v_tile, (2, 3, 1, 0)), 1, 3)
        v_tile = _prepare_paged_v_tile(v_tile, True)

        kv_page_table = fx.rocdl.make_buffer_ptr(
            fx.get_iter(kv_page_indices) + kv_start,
            num_records_bytes=(
                fx.Int64(num_kv_pages) * (kv_page_indices.dtype.width // 8)
            ),
        )
        attention_pipeline(
            q_tile, k_tile, v_tile, o_tile, query_pos0, query_len, kv_len, full_qo_len,
            kv_page_table, num_kv_pages,
            cu_seqlens_k[batch_index], kv_head, qk_scale_log2, v_scale,
            requires_epilogue_reentry_barrier,
        )

    @flyc.kernel(known_block_size=[num_threads, 1, 1])
    def attention_kernel_static(
        q: fx.Tensor,
        k: fx.Tensor,
        v: fx.Tensor,
        cu_seqlens_q: fx.Tensor,
        cu_seqlens_k: fx.Tensor,
        kv_indptr: fx.Tensor,
        kv_page_indices: fx.Tensor,
        q_descale: fx.Tensor,
        k_descale: fx.Tensor,
        v_descale: fx.Tensor,
        kv_last_page_lens: fx.Tensor,
        output: fx.Tensor,
    ):
        tid = fx.thread_idx.x
        work_ticket = fx.Int32(fx.block_idx.x)
        works_per_head = (cu_seqlens_q[1] - cu_seqlens_q[0] + block_m - 1) // block_m
        if const_expr(is_causal):
            physical_tile = work_ticket // num_qo_heads
            head_index = work_ticket - physical_tile * num_qo_heads
            half_tile = physical_tile // 2
            balanced_work = ((physical_tile & 1) == 0).select(half_tile, works_per_head - 1 - half_tile)
            affine_work = (physical_tile * causal_tile_step + causal_tile_offset) % works_per_head
            query_tile_index = (works_per_head == 256).select(affine_work, balanced_work)
        else:
            head_index = work_ticket // works_per_head
            query_tile_index = work_ticket - head_index * works_per_head
        process_work_item(
            q, k, v, cu_seqlens_q, cu_seqlens_k, kv_indptr, kv_page_indices,
            q_descale, kv_last_page_lens, output,
            fx.Int32(0), head_index, query_tile_index, tid, k_descale[0], v_descale[0], False,
        )

    @flyc.kernel(known_block_size=[num_threads, 1, 1])
    def attention_kernel(
        q: fx.Tensor,
        k: fx.Tensor,
        v: fx.Tensor,
        cu_seqlens_q: fx.Tensor,
        cu_seqlens_k: fx.Tensor,
        kv_indptr: fx.Tensor,
        kv_page_indices: fx.Tensor,
        q_descale: fx.Tensor,
        k_descale: fx.Tensor,
        v_descale: fx.Tensor,
        kv_last_page_lens: fx.Tensor,
        output: fx.Tensor,
        work_counter: fx.Tensor,
    ):
        tid = fx.thread_idx.x
        batch_size = fx.size(cu_seqlens_q.shape).to_py_value() - 1

        @flyc.jit
        def fetch_work(work_counter, tid):
            if tid == 0:
                address = fx.ptrtoint(fx.get_iter(work_counter))
                llvm_pointer = llvm.inttoptr(ir.Type.parse("!llvm.ptr<1>"), as_ir_value(address))
                old = llvm.AtomicRMWOp(
                    llvm.AtomicBinOp.add, llvm_pointer, as_ir_value(fx.Int32(1)), llvm.AtomicOrdering.monotonic,
                    syncscope="agent", alignment=4,
                )
                work_counter[fx.block_idx.x + 1] = fx.Int32(old.result)
                _s_waitcnt(vmcnt=0)
            gpu.barrier()
            ticket = work_counter[fx.block_idx.x + 1]
            _s_waitcnt(vmcnt=0)
            gpu.barrier()
            return ticket

        @flyc.jit
        def advance_work_ticket(ticket_delta, query_tile_index, head_index, batch_index, works_per_head):
            query_tile_index += ticket_delta
            while (batch_index < batch_size) & (query_tile_index >= works_per_head):
                query_tile_index -= works_per_head
                head_index += 1
                if head_index >= num_qo_heads:
                    head_index = 0
                    batch_index += 1
                    if batch_index < batch_size:
                        works_per_head = (cu_seqlens_q[batch_index + 1] - cu_seqlens_q[batch_index]
                                          + block_m - 1) // block_m
            return query_tile_index, head_index, batch_index, works_per_head

        work_ticket = fx.Int32(fx.block_idx.x)
        batch_index = fx.Int32(0)
        head_index = fx.Int32(0)
        query_tile_index = fx.Int32(0)
        works_per_head = (cu_seqlens_q[1] - cu_seqlens_q[0] + block_m - 1) // block_m
        k_scale = k_descale[0]
        v_scale = v_descale[0]
        query_tile_index, head_index, batch_index, works_per_head = advance_work_ticket(
            work_ticket, query_tile_index, head_index, batch_index, works_per_head
        )

        while batch_index < batch_size:
            process_work_item(
                q, k, v, cu_seqlens_q, cu_seqlens_k, kv_indptr, kv_page_indices,
                q_descale, kv_last_page_lens, output,
                batch_index, head_index, query_tile_index, tid, k_scale, v_scale,
                True,
            )

            next_ticket = fetch_work(work_counter, tid)
            ticket_delta = next_ticket - work_ticket
            work_ticket = next_ticket
            query_tile_index, head_index, batch_index, works_per_head = advance_work_ticket(
                ticket_delta, query_tile_index, head_index, batch_index, works_per_head
            )

    @flyc.jit
    def launch(
        q: fx.Tensor,
        k: fx.Tensor,
        v: fx.Tensor,
        cu_seqlens_q: fx.Tensor,
        cu_seqlens_k: fx.Tensor,
        kv_indptr: fx.Tensor,
        kv_page_indices: fx.Tensor,
        q_descale: fx.Tensor,
        k_descale: fx.Tensor,
        v_descale: fx.Tensor,
        kv_last_page_lens: fx.Tensor,
        output: fx.Tensor,
        work_counter: fx.Tensor,
        num_workgroups: fx.Int32,
        static_schedule: fx.Constexpr[bool],
        stream: fx.Stream,
    ):
        num_query_tokens = q.shape[0].to_py_value()
        num_physical_pages = v.shape[0].to_py_value()
        vector_size = 128 // k.dtype.width
        q = fx.make_view(
            fx.get_iter(q),
            fx.make_ordered_layout(
                (num_query_tokens, num_qo_heads, head_dim_qk), (2, 1, 0)
            ),
        )
        if fx.const_expr(key_layout == "linear"):
            num_kv_tokens = k.shape[0].to_py_value()
            k = fx.make_view(
                fx.get_iter(k),
                fx.make_ordered_layout(
                    (num_kv_tokens, num_kv_heads, head_dim_qk), (2, 1, 0)
                ),
            )
        else:
            k = fx.make_view(
                fx.get_iter(k),
                fx.make_ordered_layout(
                    (
                        num_physical_pages,
                        num_kv_heads,
                        head_dim_qk // vector_size,
                        page_size,
                        vector_size,
                    ),
                    (4, 3, 2, 1, 0),
                ),
            )
        v = fx.make_view(
            fx.get_iter(v),
            fx.make_ordered_layout(
                (num_physical_pages, num_kv_heads, page_size // vector_size, head_dim_v, vector_size),
                (4, 3, 2, 1, 0),
            ),
        )
        q_descale = fx.make_view(
            fx.get_iter(q_descale),
            fx.make_ordered_layout(
                (num_query_tokens, num_qo_heads, 1), (2, 1, 0)
            ),
        )
        k_descale = fx.make_view(fx.get_iter(k_descale), fx.make_layout(1, 1))
        v_descale = fx.make_view(fx.get_iter(v_descale), fx.make_layout(1, 1))
        output = fx.make_view(
            fx.get_iter(output),
            fx.make_ordered_layout(
                (num_query_tokens, num_qo_heads, head_dim_v), (2, 1, 0)
            ),
        )
        value_attrs = {"passthrough": [["target-features", "-packed-fp32-ops"]]}
        if static_schedule:
            attention_kernel_static(
                q, k, v, cu_seqlens_q, cu_seqlens_k, kv_indptr,
                kv_page_indices, q_descale, k_descale, v_descale,
                kv_last_page_lens, output,
                value_attrs=value_attrs,
            ).launch(grid=(num_workgroups, 1, 1), block=(num_threads, 1, 1), stream=stream)
        else:
            attention_kernel(
                q, k, v, cu_seqlens_q, cu_seqlens_k, kv_indptr,
                kv_page_indices, q_descale, k_descale, v_descale,
                kv_last_page_lens, output, work_counter,
                value_attrs=value_attrs,
            ).launch(grid=(num_workgroups, 1, 1), block=(num_threads, 1, 1), stream=stream)

    def callable(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        kv_indptr,
        kv_page_indices,
        max_seqlen_q,
        max_seqlen_k,
        causal,
        q_descale,
        k_descale,
        v_descale,
        kv_last_page_lens,
        out,
        stream=None,
    ):
        stream = torch.cuda.current_stream() if stream is None else stream
        assert causal == is_causal
        assert not causal or max_seqlen_k >= max_seqlen_q
        if cu_seqlens_k is None:
            if key_layout != "vectorized":
                raise ValueError(
                    "cu_seqlens_k=None requires key_layout='vectorized'"
                )
            cu_seqlens_k = cu_seqlens_q
        assert q.dtype in (torch.float8_e4m3fnuz, torch.bfloat16)
        if key_layout == "linear":
            assert q.dtype == torch.bfloat16
            assert head_dim_qk == head_dim_v == 128
        assert k.dtype == q.dtype
        assert v.dtype == q.dtype
        tensors = (q, k, v, cu_seqlens_q, cu_seqlens_k, kv_indptr, kv_page_indices,
                   q_descale, k_descale, v_descale, kv_last_page_lens, out)
        assert all(tensor.is_cuda and tensor.is_contiguous() for tensor in tensors)
        assert cu_seqlens_q.dtype == torch.int32
        assert cu_seqlens_k.dtype == torch.int32
        assert kv_indptr.dtype == torch.int32
        assert kv_page_indices.dtype == torch.int32
        assert kv_last_page_lens.dtype == torch.int32
        assert q_descale.dtype == k_descale.dtype == v_descale.dtype == torch.float32
        assert out.dtype == torch.bfloat16
        assert q.shape[1:] == (num_qo_heads, head_dim_qk)
        num_query_tokens = q.shape[0]
        assert q_descale.shape == (num_query_tokens, num_qo_heads, 1)
        assert out.shape == (num_query_tokens, num_qo_heads, head_dim_v)
        vector_size = 16 // q.element_size()
        num_physical_pages = v.shape[0]
        if key_layout == "linear":
            assert k.shape[1:] == (num_kv_heads, head_dim_qk)
            assert k.shape[0] == int(cu_seqlens_k[-1].item())
        else:
            assert k.shape == (
                num_physical_pages,
                num_kv_heads,
                head_dim_qk // vector_size,
                page_size,
                vector_size,
            )
        assert v.shape == (
            num_physical_pages, num_kv_heads,
            page_size // vector_size, head_dim_v, vector_size,
        )
        assert k_descale.numel() == 1
        assert v_descale.numel() == 1
        assert cu_seqlens_q.ndim == cu_seqlens_k.ndim == kv_indptr.ndim == kv_page_indices.ndim == 1
        assert kv_last_page_lens.ndim == 1
        assert cu_seqlens_q.shape == cu_seqlens_k.shape == kv_indptr.shape
        assert kv_last_page_lens.shape[0] == cu_seqlens_q.shape[0] - 1
        assert k.numel() * k.element_size() <= 2**31 - 1
        assert "gfx942" in torch.cuda.get_device_properties().gcnArchName
        batch_size = cu_seqlens_q.shape[0] - 1
        static_schedule = batch_size == 1
        if static_schedule:
            works_per_head = (num_query_tokens + block_m - 1) // block_m
            num_workgroups = num_qo_heads * works_per_head
        else:
            multiprocessor_count = torch.cuda.get_device_properties().multi_processor_count
            num_workgroups = multiprocessor_count * 2
        if static_schedule:
            work_counter = getattr(launch, "_static_work_counter", None)
            if work_counter is None:
                work_counter = torch.empty(1, device="cuda", dtype=torch.int32)
                launch._static_work_counter = work_counter
        else:
            with torch.cuda.stream(stream):
                work_counter = torch.zeros(num_workgroups + 1, device="cuda", dtype=torch.int32)
                work_counter[0] = num_workgroups

        compiled_cache = getattr(launch, "_compiled", {})
        cache_key = (
            static_schedule,
            num_workgroups,
            torch.cuda.current_device(),
            torch.cuda.get_device_properties().gcnArchName,
            *(_tensor_signature(tensor) for tensor in (
                q, k, v, cu_seqlens_q, cu_seqlens_k, kv_indptr, kv_page_indices,
                q_descale, k_descale, v_descale, kv_last_page_lens, out,
                work_counter,
            )),
        )
        compiled = compiled_cache.get(cache_key)
        if compiled is None:
            saved_compile_hints = launch.compile_hints
            try:
                launch.compile_hints = {**saved_compile_hints, **_compile_hints_for_dtype(q.dtype)}
                compiled = flyc.compile(
                    launch, q, k, v, cu_seqlens_q, cu_seqlens_k, kv_indptr,
                    kv_page_indices, q_descale, k_descale, v_descale,
                    kv_last_page_lens, out, work_counter, num_workgroups,
                    static_schedule, stream,
                )
            finally:
                launch.compile_hints = saved_compile_hints
            compiled_cache[cache_key] = compiled
            launch._compiled = compiled_cache
        else:
            compiled(
                q, k, v, cu_seqlens_q, cu_seqlens_k, kv_indptr,
                kv_page_indices, q_descale, k_descale, v_descale,
                kv_last_page_lens, out, work_counter, num_workgroups,
                static_schedule, stream,
            )

    return callable
