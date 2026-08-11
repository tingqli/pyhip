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


def _fma_vec_f32(lhs, rhs, acc):
    return fx.Vector.from_elements(
        [_fma_f32(lhs[index], rhs, acc[index]) for index in range_constexpr(lhs.numel)], fx.Float32,
    )


def _scale_center_vec_f32(values, scale, center):
    return fx.Vector.from_elements(
        [_fma_f32(values[index], scale, center, negate_acc=True) for index in range_constexpr(values.numel)],
        fx.Float32,
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


def _s_waitcnt(vmcnt=63, expcnt=7, lgkmcnt=63):
    vm_lo = vmcnt & 0xF
    vm_hi = (vmcnt >> 4) & 0x3
    rocdl.s_waitcnt(vm_lo | (expcnt << 4) | (lgkmcnt << 8) | (vm_hi << 14))


def _recast_tensor(tensor, dtype):
    pointer_type = fx.PointerType.get(dtype.ir_type, tensor.memspace, dtype.width // 8)
    iterator = fx.recast_iter(pointer_type, fx.get_iter(tensor))
    layout = fx.recast_layout(tensor.layout, tensor.dtype.width, dtype.width)
    return fx.make_view(iterator, layout)


@flyc.jit
def _online_softmax(
    frag_s,
    frag_o,
    qk_scale_log2,
    old_max,
    running_sum,
    query_pos0,
    kv_block,
    kv_len,
    full_qo_len,
    is_all_kv_valid: fx.Constexpr[bool],
    is_causal: fx.Constexpr[bool],
    overlap_max_shuffle: fx.Constexpr[bool],
    split_shuffle_valu: fx.Constexpr[bool],
):
    if const_expr(not is_all_kv_valid):
        lane_id = fx.thread_idx.x & 63
        column_base = (lane_id < 32).select(fx.Int32(0), fx.Int32(16))
        block_base = fx.Int32(kv_block * 32)
        if const_expr(is_causal):
            wave_id = fx.thread_idx.x // 64
            query_row = fx.thread_idx.x & 31
            query_pos = query_pos0 + wave_id * 32 + query_row
            causal_limit = kv_len - full_qo_len + query_pos
            for index in range_constexpr(16):
                if block_base + column_base + fx.Int32(index) > causal_limit:
                    frag_s[index, 0, 0] = float("-inf")
        else:
            for index in range_constexpr(16):
                if block_base + column_base + fx.Int32(index) >= kv_len:
                    frag_s[index, 0, 0] = float("-inf")

    score = frag_s.load()
    if const_expr(overlap_max_shuffle):
        # Q/K scale is positive, so the cross-lane max can run before scaling.
        row_max = score.reduce("max")
        shuffled_row_max = row_max.shuffle_xor(32, 64)
        if const_expr(split_shuffle_valu):
            rocdl.sched_barrier(0)
    if const_expr(split_shuffle_valu):
        scaled_values = [
            _fma_f32(score[index], qk_scale_log2, fx.Float32(0.0))
            for index in range_constexpr(score.numel // 2)
        ]
        rocdl.sched_barrier(0)
        row_max = _maxnumf(row_max, shuffled_row_max)
        scaled_values.extend(
            [
                _fma_f32(score[index], qk_scale_log2, fx.Float32(0.0))
                for index in range_constexpr(score.numel // 2, score.numel)
            ]
        )
        scaled_score = fx.Vector.from_elements(scaled_values, fx.Float32)
    else:
        if const_expr(overlap_max_shuffle):
            row_max = _maxnumf(row_max, shuffled_row_max)
        scaled_score = fx.Vector.from_elements(
            [
                _fma_f32(score[index], qk_scale_log2, fx.Float32(0.0))
                for index in range_constexpr(score.numel)
            ],
            fx.Float32,
        )
    if const_expr(overlap_max_shuffle):
        row_max = _fma_f32(row_max, qk_scale_log2, fx.Float32(0.0))
    else:
        row_max = scaled_score.reduce("max")
        row_max = _maxnumf(row_max, row_max.shuffle_xor(32, 64))

    new_max = old_max
    correction = fx.Float32(1.0)
    if row_max > old_max + fx.Float32(7.0):
        new_max = row_max + fx.Float32(1.0)
        correction = _exp2_f32(old_max - new_max)

    probability = _exp2_vec_f32(scaled_score - new_max)
    tile_sum = probability.reduce("add")
    new_sum = _fma_f32(running_sum, correction, tile_sum)
    frag_s.store(probability)

    def rescale_output():
        frag_o.store(frag_o.load() * correction)

    @flyc.jit
    def rescale_if_needed():
        if correction < fx.Float32(1.0):
            rescale_output()

    rescale_if_needed()
    return new_max, new_sum


@functools.cache
def MHA(
    num_qo_heads,
    num_kv_heads,
    head_dim_qk,
    head_dim_v,
    page_size,
    is_causal,
):
    assert head_dim_qk in (128, 192)
    assert head_dim_v == 128
    assert page_size == 32
    assert num_qo_heads % num_kv_heads == 0

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
        page_table,
        num_kv_pages,
        qk_scale_log2,
        v_scale,
        single_work_item: fx.Constexpr[bool],
    ):
        tid = fx.thread_idx.x
        element_type = q_tile.dtype
        mma_k = {fx.Float8E4M3FNUZ: 16, fx.BFloat16: 8}[element_type]
        use_hw_slot_priority = element_type == fx.BFloat16 and head_dim_qk == 128
        hw_wave_slot = _read_hw_wave_slot() if const_expr(use_hw_slot_priority) else None

        def set_stage0_priority():
            if const_expr(use_hw_slot_priority):
                _set_hw_slot_priority(hw_wave_slot, 1, 0)
            else:
                rocdl.s_setprio(0)

        def set_stage1_priority():
            if const_expr(use_hw_slot_priority):
                _set_hw_slot_priority(hw_wave_slot, 3, 2)
            else:
                rocdl.s_setprio(2)

        mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(32, 32, mma_k, element_type))
        atom_values = mma_k // 2
        vector_values = 128 // element_type.width
        k_permutation = fx.make_layout((atom_values, 2, 2), (1, vector_values, atom_values))
        wave_layout = fx.make_layout((1, 4, 1), (1, 1, 0))
        mma_tile = fx.make_tile(None, None, k_permutation)
        tiled_mma_qk = fx.make_tiled_mma(mma_atom, wave_layout, mma_tile)
        tiled_mma_pv = fx.make_tiled_mma(mma_atom, wave_layout, mma_tile)
        thread_mma_qk = tiled_mma_qk.thr_slice(tid)
        thread_mma_pv = tiled_mma_pv.thr_slice(tid)

        q_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), element_type)
        q_copy = fx.make_tiled_copy_B(q_copy_atom, tiled_mma_qk).get_slice(tid)
        frag_q = thread_mma_qk.make_fragment_B(q_tile)
        fx.copy(q_copy_atom, q_copy.partition_S(q_tile), q_copy.retile(frag_q))

        k_fake = fx.Tensor(fx.make_view(
            fx.get_iter(k_tile), fx.make_layout((block_n, head_dim_qk), (head_dim_qk, 1))
        ))
        v_fake = fx.Tensor(fx.make_view(
            fx.get_iter(v_tile), fx.make_layout((head_dim_v, block_n), (block_n, 1))
        ))
        frag_k = thread_mma_qk.make_fragment_A(k_fake)
        frag_v = thread_mma_pv.make_fragment_A(v_fake)
        frag_k.fill(0)
        frag_v.fill(0)
        frag_s = thread_mma_qk.make_fragment_C(
            fx.make_rmem_tensor(fx.make_layout((block_n, block_m), (block_m, 1)), fx.Float32)
        )
        o_transposed = fx.select(o_tile, [1, 0])
        frag_o = thread_mma_pv.make_fragment_C(o_transposed)
        if const_expr(element_type == fx.BFloat16):
            probability_storage = fx.make_fragment_like(frag_s, dtype=fx.BFloat16)
            probability_operand = fx.make_view(
                fx.get_iter(probability_storage), fx.make_layout((4, 1, (2, 2)), (1, 0, (4, 8)))
            )
        else:
            probability_operand = thread_mma_pv.make_fragment_B(fx.make_rmem_tensor(
                fx.make_layout((block_m, block_n), (block_n, 1)), element_type
            ))
        frag_s.fill(0)
        probability_operand.fill(0)

        k_lds_stride = head_dim_qk + (8 if element_type == fx.BFloat16 else 16)

        @fx.union
        class SharedStorage:
            k_lds: fx.Array[element_type, 2 * block_n * k_lds_stride, 16]
            o_lds: fx.Array[fx.BFloat16, block_m * (head_dim_v // 2), 16]

        shared = fx.SharedAllocator().allocate(SharedStorage)
        k_lds_layout = fx.make_layout(
            (block_n, head_dim_qk, 2), (k_lds_stride, 1, block_n * k_lds_stride)
        )
        k_lds_storage = fx.make_view(shared.k_lds.peek().ptr, k_lds_layout)
        if const_expr(element_type == fx.BFloat16):
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

        def is_valid_static_block(block_index):
            return const_expr(block_index >= 0) if const_expr(isinstance(block_index, int)) else True

        if const_expr(element_type == fx.BFloat16):
            k_global_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), element_type)
            k_lds_store_atom = fx.make_copy_atom(fx.UniversalCopy128b(), element_type)
            num_k_copies = head_dim_qk // 64
            prefetched_k = [
                fx.make_rmem_tensor(fx.make_layout((8, num_k_copies), (1, 8)), element_type),
                fx.make_rmem_tensor(fx.make_layout((8, num_k_copies), (1, 8)), element_type),
            ]

            def global_load_k(block_index, page_id, fragment_id):
                if const_expr(is_valid_static_block(block_index)):
                    for atom_index in range_constexpr(num_k_copies):
                        linear_atom = tid + atom_index * num_threads
                        source_row = linear_atom & (block_n - 1)
                        d_group = linear_atom // block_n
                        source_offset = (page_id * block_n * head_dim_qk
                                         + d_group * block_n * vector_values
                                         + source_row * vector_values)
                        source = fx.make_view(
                            fx.get_iter(k_tile) + source_offset, fx.make_layout(8, 1)
                        )
                        fx.copy(k_global_copy_atom, source, prefetched_k[fragment_id][None, atom_index])
                    return num_k_copies
                return 0

            def lds_store_k(block_index, fragment_id, stage):
                if const_expr(is_valid_static_block(block_index)):
                    for atom_index in range_constexpr(num_k_copies):
                        linear_atom = tid + atom_index * num_threads
                        source_row = linear_atom & (block_n - 1)
                        d_group = linear_atom // block_n
                        destination_offset = ((stage & 1) * block_n * k_lds_stride
                                              + source_row * k_lds_stride + d_group * vector_values)
                        destination = fx.make_view(
                            fx.get_iter(k_lds_storage) + destination_offset, fx.make_layout(8, 1)
                        )
                        fx.copy(k_lds_store_atom, prefetched_k[fragment_id][None, atom_index], destination)
        else:
            k_global_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy64b(), fx.Uint32)
            k_lds_store_atom = fx.make_copy_atom(fx.UniversalCopy64b(), fx.Uint32)
            k_tile_u32 = _recast_tensor(k_tile, fx.Uint32)
            num_k_copies = 3
            prefetched_k = [
                fx.make_rmem_tensor(fx.make_layout((2, num_k_copies), (1, 2)), fx.Uint32),
                fx.make_rmem_tensor(fx.make_layout((2, num_k_copies), (1, 2)), fx.Uint32),
            ]
            k_row = tid // 8
            k_chunk_in_group = tid & 7
            k_source_row = (k_row & 3) + ((k_row // 4) & 1) * 16 + (k_row // 8) * 4

            def global_load_k(block_index, page_id, fragment_id):
                if const_expr(is_valid_static_block(block_index)):
                    prefetched_k[fragment_id].fill(0)
                    for atom_index in range_constexpr(num_k_copies):
                        chunk = k_chunk_in_group + atom_index * 8
                        d_group = chunk // 2
                        d_half = chunk & 1
                        source_offset = (page_id * block_n * head_dim_qk + d_group * block_n * 16
                                         + k_source_row * 16 + d_half * 8)
                        source = fx.make_view(
                            fx.get_iter(k_tile_u32) + source_offset // 4, fx.make_layout(2, 1)
                        )
                        fx.copy(k_global_copy_atom, source, prefetched_k[fragment_id][None, atom_index])
                    return num_k_copies
                return 0

            def lds_store_k(block_index, fragment_id, stage):
                if const_expr(is_valid_static_block(block_index)):
                    for atom_index in range_constexpr(num_k_copies):
                        chunk = k_chunk_in_group + atom_index * 8
                        destination_offset = ((stage & 1) * block_n * k_lds_stride
                                              + k_row * k_lds_stride + chunk * 8)
                        destination = fx.make_view(
                            fx.get_iter(k_lds) + destination_offset, fx.make_layout(8, 1)
                        )
                        fx.copy(k_lds_store_atom, prefetched_k[fragment_id][None, atom_index],
                                _recast_tensor(destination, fx.Uint32))

        k_lds_copy_atom = fx.make_copy_atom(fx.UniversalCopy128b(), element_type)
        k_lds_copy = fx.make_tiled_copy_A(k_lds_copy_atom, tiled_mma_qk).get_slice(tid)

        def partition_k_lds(stage):
            return k_lds_copy.partition_S(k_lds[None, None, stage])

        v_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), element_type)
        v_copy = fx.make_tiled_copy_A(v_copy_atom, tiled_mma_pv).get_slice(tid)
        num_v_loads = fx.size(frag_v.shape).get_static_leaf_int * frag_v.dtype.width // 128
        num_k_fragment_bits = fx.size(frag_k.shape).get_static_leaf_int * frag_k.dtype.width

        frag_o.fill(0.0)

        def gemm_qk():
            for k_group in range_constexpr(head_dim_qk // (2 * mma_k)):
                for k_atom in range_constexpr(2):
                    accumulator = frag_s[None, 0, 0]
                    fx.mma_atom_call(
                        mma_atom, accumulator, frag_k[None, 0, (k_atom, k_group)],
                        frag_q[None, 0, (k_atom, k_group)], accumulator,
                    )

        def kv_step(
            kv_block,
            lds_stage,
            current_prefetch,
            next_prefetch,
            current_max,
            running_sum,
            page_id0,
            page_id1,
            page_id2,
            is_all_kv_valid: fx.Constexpr[bool] = True,
        ):
            page_id3 = page_table[kv_block + 3]

            if const_expr(is_valid_static_block(kv_block)):
                frag_s.fill(0.0)
                gemm_qk()
                fx.copy(v_copy_atom, v_copy.partition_S(v_tile[None, None, page_id0]), v_copy.retile(frag_v))

                if const_expr(element_type == fx.BFloat16 and head_dim_qk == 128):
                    for _ in range_constexpr(num_v_loads):
                        rocdl.sched_vmem(1)
                        rocdl.sched_mfma(1)
                    rocdl.sched_mfma(head_dim_qk // mma_k - num_v_loads)
                else:
                    for _ in range_constexpr(num_v_loads):
                        rocdl.sched_vmem(1)
                        rocdl.sched_mfma(3)
                rocdl.sched_vmem(100)
                rocdl.sched_mfma(100)

            rocdl.sched_barrier(0)
            set_stage0_priority()
            rocdl.sched_barrier(0)

            global_load_k(kv_block + 2, page_id2, next_prefetch)

            if const_expr(is_valid_static_block(kv_block)):
                current_max, running_sum = _online_softmax(
                    frag_s, frag_o, qk_scale_log2, current_max, running_sum,
                    query_pos0, kv_block, kv_len, full_qo_len, is_all_kv_valid, is_causal,
                    element_type in (fx.BFloat16, fx.Float8E4M3FNUZ),
                    element_type == fx.Float8E4M3FNUZ,
                )

            lds_store_k(kv_block + 1, current_prefetch, lds_stage ^ 1)

            if const_expr(is_valid_static_block(kv_block)):
                if const_expr(element_type == fx.Float8E4M3FNUZ):
                    probability = frag_s.load()
                    for k_group in range_constexpr(2):
                        start = k_group * 8
                        probability_lo = _pack_probability_fp8(probability, start)
                        probability_hi = _pack_probability_fp8(probability, start + 4)
                        probability_operand[None, 0, k_group].store(
                            probability_lo.shuffle(probability_hi, list(range(8)))
                        )
                else:
                    probability_storage.store(_cvt_f32_to_bf16(frag_s).load())

            rocdl.sched_barrier(0)
            set_stage1_priority()
            rocdl.sched_barrier(0)

            if const_expr(is_valid_static_block(kv_block)):
                fx.gemm(mma_atom, frag_o, frag_v, probability_operand, frag_o)

            gpu.barrier()
            if const_expr(is_valid_static_block(kv_block + 1)):
                fx.copy(k_lds_copy_atom, partition_k_lds(lds_stage ^ 1), k_lds_copy.retile(frag_k))

            if const_expr(element_type == fx.BFloat16):
                for _ in range_constexpr(num_k_copies):
                    rocdl.sched_vmem(1)
                    rocdl.sched_dswr(1)
                rocdl.sched_mfma(3)
                for _ in range_constexpr(num_k_fragment_bits // 128):
                    rocdl.sched_dsrd(1)
                    rocdl.sched_mfma(1)
                rocdl.sched_mfma(head_dim_v // mma_k - num_k_fragment_bits // 128 - 3)
            else:
                rocdl.sched_vmem(1)
                rocdl.sched_dswr(1)
                rocdl.sched_mfma(7)
                rocdl.sched_vmem(1)
                rocdl.sched_mfma(3)
                rocdl.sched_dswr(1)
                rocdl.sched_mfma(4)
                for _ in range_constexpr(num_k_fragment_bits // 128):
                    rocdl.sched_dsrd(1)
                    rocdl.sched_mfma(1)
            rocdl.sched_barrier(0)

            return current_max, running_sum, page_id1, page_id2, page_id3

        current_max = fx.Float32(float("-inf"))
        running_sum = fx.Float32(0.0)
        page0, page1, page2 = page_table[0], page_table[1], page_table[2]

        global_load_k(0, page0, 0)
        lds_store_k(0, 0, 0)
        global_load_k(1, page1, 0)
        gpu.barrier()
        fx.copy(k_lds_copy_atom, partition_k_lds(0), k_lds_copy.retile(frag_k))
        rocdl.sched_barrier(0)
        set_stage1_priority()
        rocdl.sched_barrier(0)

        if const_expr(is_causal):
            causal_base = kv_len - full_qo_len + query_pos0
            full_pages = (causal_base + 1) // block_n
            valid_pages = (full_pages // 2) * 2
            intersecting_pages = (causal_base + query_len + block_n - 1) // block_n
            pages_to_process = (intersecting_pages < num_kv_pages).select(intersecting_pages, num_kv_pages)
        else:
            valid_pages = num_kv_pages - 2
            if (num_kv_pages & 1) == 1:
                valid_pages = num_kv_pages - 1
            pages_to_process = num_kv_pages

        results = [current_max, running_sum, page0, page1, page2]
        for page_index, state in range(0, valid_pages, 2, init=results):
            current_max, running_sum, page0, page1, page2 = state
            current_max, running_sum, page0, page1, page2 = kv_step(
                page_index, 0, 0, 1, current_max, running_sum, page0, page1, page2
            )
            current_max, running_sum, page0, page1, page2 = kv_step(
                page_index + 1, 1, 1, 0, current_max, running_sum, page0, page1, page2
            )
            results = yield [current_max, running_sum, page0, page1, page2]

        for page_index, state in range(valid_pages, pages_to_process, 2, init=results):
            current_max, running_sum, page0, page1, page2 = state
            current_max, running_sum, page0, page1, page2 = kv_step(
                page_index, 0, 0, 1, current_max, running_sum, page0, page1, page2,
                is_all_kv_valid=False
            )
            if fx.Int32(page_index + 1) < pages_to_process:
                current_max, running_sum, page0, page1, page2 = kv_step(
                    page_index + 1, 1, 1, 0, current_max, running_sum, page0, page1, page2,
                    is_all_kv_valid=False
                )
            results = yield [current_max, running_sum, page0, page1, page2]

        running_sum = results[1]
        denominator = running_sum + running_sum.shuffle_xor(32, 64)
        frag_o.store(frag_o.load() * (v_scale / denominator))
        frag_o_bf16 = _cvt_f32_to_bf16(frag_o)
        cshuffle_store_atom = fx.make_copy_atom(fx.UniversalCopy64b(), fx.BFloat16)
        cshuffle_store = fx.make_tiled_copy_C(cshuffle_store_atom, tiled_mma_pv).get_slice(tid)
        cshuffle_read_atom = fx.make_copy_atom(fx.UniversalCopy128b(), fx.BFloat16)
        cshuffle_read = fx.make_tiled_copy_tv(
            cshuffle_read_atom, fx.make_layout((32, 8), (8, 1)), fx.make_layout((4, 8), (8, 1))
        ).get_slice(tid)
        store_source_halves = fx.logical_divide(cshuffle_store.retile(frag_o_bf16), (None, 2, None))
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
            if const_expr(half == 0 or not single_work_item):
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
        kv_indptr,
        kv_page_indices,
        q_descale,
        kv_last_page_lens,
        output,
        batch_index,
        head_index,
        current_work,
        tid,
        k_scale,
        v_scale,
        single_work_item: fx.Constexpr[bool],
    ):
        query_pos0 = current_work * block_m
        query_start = cu_seqlens_q[batch_index] + query_pos0
        query_end = fx.Int32(arith.minsi(
            arith.unwrap(query_start + block_m), arith.unwrap(cu_seqlens_q[batch_index + 1])
        ))
        query_len = query_end - query_start
        full_qo_len = cu_seqlens_q[batch_index + 1] - cu_seqlens_q[batch_index]
        kv_start = kv_indptr[batch_index]
        num_kv_pages = kv_indptr[batch_index + 1] - kv_start
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

        k_tile = k[None, kv_head, None, None, None]
        k_tile = fx.group(fx.select(k_tile, (2, 3, 1, 0)), 1, 3)
        k_tile = fx.rocdl.make_buffer_tensor(k_tile, max_size=False)
        v_tile = v[None, kv_head, None, None, None]
        v_tile = fx.group(fx.select(v_tile, (2, 3, 1, 0)), 1, 3)
        if const_expr(v_tile.dtype == fx.BFloat16):
            token_permutation = fx.make_layout((8, (2, 2)), (1, (16, 8)))
            v_tile = fx.composition(v_tile, fx.make_tile(None, token_permutation, None))
        v_tile = fx.rocdl.make_buffer_tensor(v_tile, max_size=False)

        attention_pipeline(
            q_tile, k_tile, v_tile, o_tile, query_pos0, query_len, kv_len, full_qo_len,
            fx.get_iter(kv_page_indices) + kv_start, num_kv_pages, qk_scale_log2, v_scale, single_work_item,
        )

    @flyc.kernel(known_block_size=[num_threads, 1, 1])
    def attention_kernel_static(
        q: fx.Tensor,
        k: fx.Tensor,
        v: fx.Tensor,
        cu_seqlens_q: fx.Tensor,
        kv_indptr: fx.Tensor,
        kv_page_indices: fx.Tensor,
        q_descale: fx.Tensor,
        k_descale: fx.Tensor,
        v_descale: fx.Tensor,
        kv_last_page_lens: fx.Tensor,
        output: fx.Tensor,
    ):
        tid = fx.thread_idx.x
        linear_work = fx.Int32(fx.block_idx.x)
        works_per_head = (cu_seqlens_q[1] - cu_seqlens_q[0] + block_m - 1) // block_m
        if const_expr(is_causal):
            physical_tile = linear_work // num_qo_heads
            head_index = linear_work - physical_tile * num_qo_heads
            half_tile = physical_tile // 2
            balanced_work = ((physical_tile & 1) == 0).select(half_tile, works_per_head - 1 - half_tile)
            affine_work = (physical_tile * causal_tile_step + causal_tile_offset) % works_per_head
            current_work = (works_per_head == 256).select(affine_work, balanced_work)
        else:
            head_index = linear_work // works_per_head
            current_work = linear_work - head_index * works_per_head
        process_work_item(
            q, k, v, cu_seqlens_q, kv_indptr, kv_page_indices, q_descale, kv_last_page_lens, output,
            fx.Int32(0), head_index, current_work, tid, k_descale[0], v_descale[0], True,
        )

    @flyc.kernel(known_block_size=[num_threads, 1, 1])
    def attention_kernel(
        q: fx.Tensor,
        k: fx.Tensor,
        v: fx.Tensor,
        cu_seqlens_q: fx.Tensor,
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
        def skip_work_items(count, current_work, head_index, batch_index, works_per_head):
            current_work += count
            while (batch_index < batch_size) & (current_work >= works_per_head):
                current_work -= works_per_head
                head_index += 1
                if head_index >= num_qo_heads:
                    head_index = 0
                    batch_index += 1
                    if batch_index < batch_size:
                        works_per_head = (cu_seqlens_q[batch_index + 1] - cu_seqlens_q[batch_index]
                                          + block_m - 1) // block_m
            return current_work, head_index, batch_index, works_per_head

        linear_work = fx.Int32(fx.block_idx.x)
        batch_index = fx.Int32(0)
        head_index = fx.Int32(0)
        current_work = fx.Int32(0)
        works_per_head = (cu_seqlens_q[1] - cu_seqlens_q[0] + block_m - 1) // block_m
        k_scale = k_descale[0]
        v_scale = v_descale[0]
        current_work, head_index, batch_index, works_per_head = skip_work_items(
            linear_work, current_work, head_index, batch_index, works_per_head
        )

        while batch_index < batch_size:
            process_work_item(
                q, k, v, cu_seqlens_q, kv_indptr, kv_page_indices, q_descale, kv_last_page_lens, output,
                batch_index, head_index, current_work, tid, k_scale, v_scale, False,
            )

            next_work = linear_work + num_workgroups
            if not static_schedule:
                next_work = fetch_work(work_counter, tid)
            work_delta = next_work - linear_work
            linear_work = next_work
            current_work, head_index, batch_index, works_per_head = skip_work_items(
                work_delta, current_work, head_index, batch_index, works_per_head
            )

    @flyc.jit
    def launch(
        q: fx.Tensor,
        k: fx.Tensor,
        v: fx.Tensor,
        cu_seqlens_q: fx.Tensor,
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
        num_physical_pages = k.shape[0].to_py_value()
        vector_size = 128 // k.dtype.width
        q = fx.make_view(
            fx.get_iter(q), fx.make_ordered_layout((num_query_tokens, num_qo_heads, head_dim_qk), (2, 1, 0))
        )
        k = fx.make_view(
            fx.get_iter(k),
            fx.make_ordered_layout(
                (num_physical_pages, num_kv_heads, head_dim_qk // vector_size, page_size, vector_size),
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
            fx.get_iter(q_descale), fx.make_ordered_layout((num_query_tokens, num_qo_heads, 1), (2, 1, 0))
        )
        k_descale = fx.make_view(fx.get_iter(k_descale), fx.make_layout(1, 1))
        v_descale = fx.make_view(fx.get_iter(v_descale), fx.make_layout(1, 1))
        output = fx.make_view(
            fx.get_iter(output), fx.make_ordered_layout((num_query_tokens, num_qo_heads, head_dim_v), (2, 1, 0))
        )
        value_attrs = {"passthrough": [["target-features", "-packed-fp32-ops"]]}
        if static_schedule:
            attention_kernel_static(
                q, k, v, cu_seqlens_q, kv_indptr, kv_page_indices, q_descale, k_descale, v_descale,
                kv_last_page_lens, output,
                value_attrs=value_attrs,
            ).launch(grid=(num_workgroups, 1, 1), block=(num_threads, 1, 1), stream=stream)
        else:
            attention_kernel(
                q, k, v, cu_seqlens_q, kv_indptr, kv_page_indices, q_descale, k_descale, v_descale,
                kv_last_page_lens, output, work_counter, num_workgroups, static_schedule,
                value_attrs=value_attrs,
            ).launch(grid=(num_workgroups, 1, 1), block=(num_threads, 1, 1), stream=stream)

    def callable(
        q,
        k,
        v,
        cu_seqlens_q,
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
        assert q.dtype in (torch.float8_e4m3fnuz, torch.bfloat16)
        assert k.dtype == q.dtype
        assert v.dtype == q.dtype
        assert out.dtype == torch.bfloat16
        assert q.shape[1:] == (num_qo_heads, head_dim_qk)
        vector_size = 16 // q.element_size()
        assert k.shape[1:] == (num_kv_heads, head_dim_qk // vector_size, page_size, vector_size)
        assert v.shape[1:] == (num_kv_heads, page_size // vector_size, head_dim_v, vector_size)
        assert k_descale.numel() == 1
        assert v_descale.numel() == 1
        assert k.numel() * k.element_size() <= 2**31 - 1
        assert "gfx942" in torch.cuda.get_device_properties().gcnArchName

        multiprocessor_count = torch.cuda.get_device_properties().multi_processor_count
        batch_size = cu_seqlens_q.shape[0] - 1
        static_schedule = batch_size == 1
        if static_schedule:
            works_per_head = (q.shape[0] + block_m - 1) // block_m
            num_workgroups = num_qo_heads * works_per_head
        else:
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
        cache_key = (static_schedule, q.dtype)
        compiled = compiled_cache.get(cache_key)
        if compiled is None:
            compiled = flyc.compile(
                launch, q, k, v, cu_seqlens_q, kv_indptr, kv_page_indices, q_descale, k_descale, v_descale,
                kv_last_page_lens, out, work_counter, num_workgroups, static_schedule, stream,
            )
            compiled_cache[cache_key] = compiled
            launch._compiled = compiled_cache
        else:
            compiled(
                q, k, v, cu_seqlens_q, kv_indptr, kv_page_indices, q_descale, k_descale, v_descale,
                kv_last_page_lens, out, work_counter, num_workgroups, static_schedule, stream,
            )

    return callable
