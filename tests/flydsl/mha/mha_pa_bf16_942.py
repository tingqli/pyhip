"""gfx942 BF16 paged MHA, retaining the original persistent 8-wave BN32 pipeline.

K is staged through LDS, V stays in registers, and both probability/output
conversion use the original round-half-up BF16 helper. No alternate kernel is
dispatched. Empty KV sequences are outside this backend's current contract.
"""

import functools
import math
import os

os.environ.setdefault("FLYDSL_RUNTIME_ENABLE_CACHE", "0")

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl.expr.typing import as_ir_value
from flydsl.expr import arith, gpu, rocdl
from flydsl._mlir.dialects import llvm

import pyhip.contrib.flydsl.helpers as fxh

try:
    from ._dsl import select, group, composition, flat_divide, static_view
except ImportError:
    from _dsl import select, group, composition, flat_divide, static_view


def _maxnumf(a, b):
    """Non-NaN-propagating f32 max used by the wave softmax reduction."""
    return type(a)(arith.maxnumf(arith.unwrap(a), arith.unwrap(b)))


def _uniform(value):
    """Make workgroup-uniform scheduler metadata explicit to AMDGPU lowering."""
    return fx.Int32(rocdl.readfirstlane(fx.Int32.ir_type, fx.Int32(value).ir_value()))


def _late_thread_id():
    """Materialize lane/address arithmetic at its use, not across long loops."""
    value = fx.Int32(llvm.inline_asm(
        fx.Int32.ir_type, [fx.Int32(fx.thread_idx.x).ir_value()],
        "v_mov_b32 $0, $1", "=v,v", has_side_effects=True,
    ))
    # The opaque identity hides LLVM's known range; re-expose the exact legal
    # 512-thread domain so division/modulo stay shifts/masks, not signed divides.
    return value & 511


@flyc.jit
def online_softmax(fragS, fragO, sm_scale_log2, old_max, l_in,
                   q_pos0, kv_block_n, kv_len, qo_len,
                   is_all_kv_valid: fx.Constexpr[bool],
                   KV_BLOCK_SIZE: fx.Constexpr[int],
                   is_causal: fx.Constexpr[bool],
                   rescale_always: fx.Constexpr[bool]):
    """
    old_max/l_in是会被更新的，使用SSA方式return更新后值，不要使用mutable container例如list来修改

    is_causal为True时， kv_len >= qo_len, 并且attention只需要计算causal_mask合法区域即可：

                rows = torch.arange(qo_len, device="cuda").unsqueeze(1)
                cols = torch.arange(kv_len, device="cuda").unsqueeze(0)
                causal_mask = cols <= (kv_len - qo_len + rows)
     - num_kv_pages 只需循环到某个位置即可，后面的page都不用参考
     - 某个kv-page之前都是non-causal的，之后才需要施加causal_mask
    - causal_mask 施加于 32x32 的 score 矩阵上，
    """
    # assert 0, f"{fragS}"
    if fx.const_expr(not is_all_kv_valid):
        # mask out invalid kv positions
        mask_tid = _late_thread_id()
        lane_id = mask_tid & 63
        bf16_col_lane = (lane_id < 32).select(fx.Int32(0), fx.Int32(8))
        col_block = fx.Int32(kv_block_n * KV_BLOCK_SIZE)
        if fx.const_expr(not is_causal):
            # Keep both sides explicitly i32.  A Python constexpr loop index
            # otherwise promotes this comparison to MLIR index, whose ordered
            # comparison is unsigned; a negative limit would then look huge
            # and leave invalid tail columns unmasked.
            for i in fx.range_constexpr(16):
                column = bf16_col_lane + fx.Int32((i // 8) * 16 + i % 8)
                kv_pos = col_block + column
                if kv_pos >= kv_len:
                    fragS[i,0,0] = float("-inf")
        else:
            # Bottom-right causal mask:
            #   kv_pos <= kv_len - qo_len + q_pos
            wave_id = mask_tid // 64
            row_lane = fx.thread_idx.x & 31
            q_pos = q_pos0 + wave_id * 32 + row_lane
            causal_limit = kv_len - qo_len + q_pos
            for i in fx.range_constexpr(16):
                column = bf16_col_lane + fx.Int32((i // 8) * 16 + i % 8)
                kv_pos = col_block + column
                if kv_pos > causal_limit:
                    fragS[i,0,0] = float("-inf")


    scores = fragS.load() * sm_scale_log2

    row_max = scores.reduce("max")
    row_max = _maxnumf(row_max, row_max.shuffle_xor(32, 64))

    new_max = old_max
    corr = fx.Float32(1.0)
    threshold = fxh.eltwise_op("v_add_f32", old_max, fx.Float32(7.0))
    if row_max > threshold:
        new_max = fxh.eltwise_op("v_add_f32", row_max, fx.Float32(1.0))
        # do not use inline asm inside scf.If, use intrinsic instead
        corr = fxh.eltwise_op("llvm.amdgcn.exp2.f32", old_max - new_max)

    probs = fxh.eltwise_op("v_exp_f32", scores - new_max)
    row_sum = probs.reduce("add")

    # this fake instruction avoids spills for some reason, but seems to be not required anymore
    # row_sum = fxh.eltwise_op("; fake inst", row_sum, 0.0)
    l_out = fxh.eltwise_op("v_fma_f32", l_in, corr, row_sum)
    fragS.store(probs)

    # Preserve the lazy rescale: most tiles have corr=1 and should not execute
    # a full vector multiply. Address lifetime fixes leave room for this branch.
    def rescale_output():
        fragO.store(fxh.eltwise_op("v_mul_f32", fragO.load(), corr))

    @flyc.jit
    def rescale_if_needed():
        if corr < fx.Float32(1.0):
            rescale_output()
    if fx.const_expr(rescale_always):
        rescale_output()
    else:
        rescale_if_needed()
    probability = fxh.cvt_f32_to_bf16(fragS)
    probability = fx.make_view(
        fx.get_iter(probability),
        fx.make_layout((4, 1, (2, 2)), (1, 0, (4, 8))),
    )
    return new_max, l_out, probability


@functools.cache
def _build_attention(
    num_qo_heads,
    num_kv_heads,
    head_dim_qk,
    head_dim_v,
    page_size,
    is_causal,
    quant_query_mode="per-token",
    *,
    softmax_scale=None,
    with_lse=False,
):
    """
    cu_seqlens_q: [batch_size + 1] cu_seqlens_q[i] ~ cu_seqlens_q[i+1] is the range of query tokens in batch i
    kv_indptr   : [batch_size + 1] kv_indptr[i] ~ kv_indptr[i+1] is the range of virtual page ids in batch i
    kv_page_indices : [num_pages] kv_page_indices[i] is the physical page id of virtual page i (used to index into K and V)

    k_vector_size is number of elements that 16 bytes can hold

    persistent kernel, each 8wave workgroup occupies one CU to handles part(BM) of the query/output tokens. and loop
    over cu_seqlens_q to find next part of query tokens to handle， until all query tokens are handled.

    任务复杂，从极简pipeline开始构建，保证框架正确之后再开始性能调优迭代
    """
    BM, BN = 256, 32
    num_threads = 512
    num_waves = num_threads // 64
    assert page_size in [32, 64, 128]
    num_BN_per_page = page_size // BN
    LOG2E = 1.4426950408889634
    # Preserve the original default expression, rather than preprocessing Q.
    sm_scale_log2 = float(
        LOG2E / (head_dim_qk**0.5)
        if softmax_scale is None else LOG2E * softmax_scale
    )

    assert (page_size % BN) == 0, f"{page_size=} must be a multiple of {BN=}"

    assert quant_query_mode in ["per-token", "per-tensor"], f"quant_query_mode={quant_query_mode} is not supported"
    per_token = quant_query_mode == "per-token"
    # Wider O or the extra LSE state leaves no room for a prefetched full V
    # operand alongside Q/K/S. Load V after softmax for these specializations;
    # retain the original QK/V overlap for the default V128/no-LSE path.
    defer_v = head_dim_v == 192 or (head_dim_qk == 192 and with_lse)
    stream_k = head_dim_v == 192
    bounded_k = head_dim_qk == 192 or head_dim_v == 192 or page_size == 128

    @flyc.jit
    def attn_pipeline(q_tile, # [BM, head_dim_qk]
                      k_tile, # [BN, (k_vector_size, head_dim_qk // k_vector_size), num_physical_pages, num_BN_per_page]
                      v_tile, # [head_dim_v, (k_vector_size, BN // k_vector_size), num_physical_pages, num_BN_per_page]
                      o_tile, # [BM, head_dim_v]
                      lse_tile, # [BM], pointing at this sequence's query tile/head
                      q_pos0, query_len, kv_len, full_qo_len,
                      ptr_kv_page_table,
                      num_kv_pages, last_page_len,
                      qk_scale_log2, v_s):
        tid = fx.thread_idx.x
        lane_id = fx.thread_idx.x % 64
        wave_id = fx.thread_idx.x // 64
        wave_m = wave_id // 4

        flyobj = fxh.FlyObjCache()
        tmma1 = flyobj.create_thr_mma(k_tile.dtype, (1, 8, 1), 32)
        tmma2 = flyobj.create_thr_mma(v_tile.dtype, (1, 8, 1), 32)

        # Keep the BF16 K-row permutation paired with the probability view in
        # online_softmax: lane 0 covers keys 0..7/16..23, lane 32 covers
        # 8..15/24..31. This avoids a register reorder before the P@V MFMAs.
        k_row_layout = fx.make_layout((4, 2, 2, 2), (1, 8, 4, 16))
        k_tile = composition(
            k_tile,
            fx.make_tile(k_row_layout, None, None, None),
        )

        if fx.const_expr(head_dim_qk == 192 and page_size == 32):
            # Q is loaded once per persistent work item, not once per KV tile.
            # Recompute its lane address here instead of spilling that address
            # throughout the overlapped page32 hot loop.
            q_atom = flyobj.get_buffer_copy_atom(q_tile.dtype, 128)
            q_copy = fx.make_tiled_copy_B(q_atom, tmma1).get_slice(_late_thread_id())
            fragQ = tmma1.make_fragment_B(q_tile)
            fx.copy(q_atom, q_copy.partition_S(q_tile), q_copy.retile(fragQ))
        else:
            fragQ = flyobj.load_tiled_mma_fragB(tmma1, q_tile)

        k_fake = fx.Tensor(
            fx.make_view(
                fx.get_iter(k_tile),
                fx.make_layout((BN, head_dim_qk), (head_dim_qk, 1)),
            )
        )
        v_fake = fx.Tensor(
            fx.make_view(
                fx.get_iter(v_tile),
                fx.make_layout((head_dim_v, BN), (BN, 1)),
            )
        )
        fragK = static_view(tmma1.make_fragment_A(k_fake))
        fragV = static_view(tmma2.make_fragment_A(v_fake))
        num_bits_fragK = (fx.size(fragK.shape).get_static_leaf_int * fragK.dtype.width)
        num_bits_fragV = (fx.size(fragV.shape).get_static_leaf_int * fragV.dtype.width)
        num_vm_cnt_load_v = (num_bits_fragV)//128

        fakeCt = fx.make_rmem_tensor(fx.make_layout((BN, BM), (BM, 1)), fx.Float32)
        fragS = static_view(tmma1.make_fragment_C(fakeCt))
        fragO = static_view(tmma2.make_fragment_C(select(o_tile, [1, 0])))

        # let all 512 threads participate in the copy so no extra if condition involved
        # 512*16/32 = 256, so all head_dim <= 256 can be padded to 256
        copy_atom_bits = 128

        @fx.union
        class SharedStorage:
            k_lds: fx.Array[k_tile.dtype, 2 * BN * head_dim_qk, 16]
            o_lds: fx.Array[o_tile.dtype, (BM//8) * head_dim_v, 16]

        # mask,base,shift, swizzle always in unit of 128b,
        swz_base = ((128 // k_tile.dtype.width) - 1).bit_length()
        swz = fx.SwizzleType.get(3, swz_base, 3)
        lds = fx.SharedAllocator().allocate(SharedStorage)
        layout_k_lds = fx.make_composed_layout(
            fx.static(swz),
            fx.make_ordered_layout((BN, head_dim_qk, 2), (1, 0, 2)),
        )
        lds_k = lds.k_lds.peek().view(layout_k_lds)

        # assert 0, f"{lds_ku32} {lds_k}"

        def is_valid_block_n(bn):
            #return fx.const_expr(bn >= 0 and bn < num_kv_pages) if fx.const_expr(isinstance(bn, int)) else True
            return fx.const_expr(bn >= 0) if fx.const_expr(isinstance(bn, int)) else True

        num_copy_threads = BN * head_dim_qk * k_tile.dtype.width // copy_atom_bits
        assert BN * head_dim_qk * k_tile.dtype.width % copy_atom_bits == 0
        if fx.const_expr(head_dim_qk == 192):
            # Fit 768 b128 atoms into 384 threads; each thread copies two atoms.
            num_copy_threads //= 2
        assert num_copy_threads <= num_threads

        # [TRICKY]
        # Keep the original global->register->LDS pipeline packed in dwords.
        # Recasting the contiguous per-thread slice preserves its byte address
        # (including the LDS swizzle) and vector<Nxi32> loop-carried values.
        def recast_tensor(src, new_dtype):
            result_type = fx.PointerType.get(new_dtype.ir_type, src.memspace, new_dtype.width//8)
            new_iter = fx.recast_iter(result_type, fx.get_iter(src))
            new_layout = fx.recast_layout(src.layout, src.dtype.width, new_dtype.width)
            return fx.make_view(new_iter, new_layout)

        lds_k_u32 = recast_tensor(lds_k, fx.Uint32)
        k_tile_u32 = recast_tensor(k_tile, fx.Uint32)

        glk_thrcopy, glk_load_atom = flyobj.get_tiled_copy_coalesced_mn(
            k_tile_u32[None, None, 0, 0],
            copy_atom_bits=copy_atom_bits,
            num_threads=num_copy_threads,
        )
        glk_srck = static_view(glk_thrcopy.partition_S(k_tile_u32))
        glk_dstk = static_view(glk_thrcopy.partition_D(lds_k_u32))

        glk_store_atom = flyobj.get_universal_copy_atom(
            fx.Uint32, copy_atom_bits
        )
        glk_frag = fx.make_fragment_like(glk_dstk[None, None, None, 0])
        num_vm_cnt_load_k = (fx.size(glk_frag.shape).get_static_leaf_int * glk_frag.dtype.width)//copy_atom_bits
        prefetch_fragk_list = [
            fx.make_fragment_like(glk_srck[None, None, None, 0, 0]),
            fx.make_fragment_like(glk_srck[None, None, None, 0, 0]),
        ]

        def global_load_k(block_n, page_id, bn_id, frag_id):
            if fx.const_expr(is_valid_block_n(block_n)):
                source = glk_srck[None, None, None, page_id, bn_id]
                if fx.const_expr(num_copy_threads == num_threads):
                    fx.copy(glk_load_atom, source, prefetch_fragk_list[frag_id])
                else:
                    if _late_thread_id() < num_copy_threads:
                        fx.copy(glk_load_atom, source, prefetch_fragk_list[frag_id])
                return num_vm_cnt_load_k
            else:
                return 0

        def ds_store_k(block_n, frag_id, lds_buff_id):
            if fx.const_expr(is_valid_block_n(block_n)):
                if fx.const_expr(num_copy_threads == num_threads):
                    fx.copy(glk_store_atom, prefetch_fragk_list[frag_id], glk_dstk[None, None, None, lds_buff_id & 1])
                else:
                    if _late_thread_id() < num_copy_threads:
                        fx.copy(glk_store_atom, prefetch_fragk_list[frag_id], glk_dstk[None, None, None, lds_buff_id & 1])

        fragO.fill(0.0)

        v_copy_atom = flyobj.get_universal_copy_atom(v_tile.dtype, 128)

        def load_v(page, part):
            v_thrcopy = fx.make_tiled_copy_A(v_copy_atom, tmma2).get_slice(_late_thread_id())
            fx.copy(v_copy_atom, v_thrcopy.partition_S(v_tile[None, None, page, part]),
                v_thrcopy.retile(fragV))

        def kv_step(page_n, lds_buff_id, cur_max, l_in,
                    kv_page_id0, kv_page_id1, kv_page_id2, kv_page_id3,
                    is_all_kv_valid: fx.Constexpr[bool] = True):
            # first block_n in pipeline is -3
            # The explicit-loop induction variable is MLIR index. Convert
            # before indexing a buffer descriptor (whose offset is i32),
            # otherwise older layout lowering emits invalid extsi i64->i32.
            kv_page_id4 = _uniform(ptr_kv_page_table[fx.Int32(page_n + 4)])

            kv_page_0123 = [kv_page_id0, kv_page_id1, kv_page_id2, kv_page_id3]

            for bn_i in fx.range_constexpr(num_BN_per_page):
                bn0_page = kv_page_0123[(bn_i + 0)//num_BN_per_page]
                bn0_part = (bn_i + 0) % num_BN_per_page
                bn3_page = kv_page_0123[(bn_i + 3)//num_BN_per_page]
                bn3_part = (bn_i + 3) % num_BN_per_page

                block_n = page_n * num_BN_per_page + bn_i

                # Q@K part for block_n
                prefetch_frag_id = lds_buff_id^1
                vm_cnt = 0

                ds_store_k(block_n + 1, prefetch_frag_id, lds_buff_id^1) # +2, +1
                vm_cnt += global_load_k(block_n + 3, bn3_page, bn3_part, prefetch_frag_id)

                if fx.const_expr(is_valid_block_n(block_n)):
                    fragS.fill(0.0)
                    if fx.const_expr(stream_k):
                        # V192 keeps 96 O accumulators per lane. Consume K in
                        # 16-column pieces in the identical MFMA reduction
                        # order instead of keeping a complete D192 K fragment.
                        k_atom = flyobj.get_universal_copy_atom(k_tile.dtype, 128)
                        k_copy = fx.make_tiled_copy_A(k_atom, tmma1).get_slice(_late_thread_id())
                        k_source = k_copy.partition_S(lds_k[None, None, lds_buff_id])
                        part_layout = fx.make_layout((4, 1, (2, 1)), (1, 0, (4, 8)))
                        k_part = fx.make_rmem_tensor(part_layout, k_tile.dtype)
                        k_target = k_copy.retile(k_part)[None, None, 0]
                        for k16 in fx.range_constexpr(head_dim_qk // 16):
                            fx.copy(k_atom, k_source[None, None, k16], k_target)
                            q_part = fx.make_view(fx.get_iter(fragQ) + k16 * 8, part_layout)
                            fx.gemm(tmma1, fragS, k_part, q_part, fragS)
                            rocdl.sched_barrier(0)
                    else:
                        fx.gemm(tmma1, fragS, fragK, fragQ, fragS)
                    # The next K tile is loaded below before any subsequent
                    # QK. Do not carry the retired operand through softmax/PV
                    # (including the conditional last-page merge).
                    fragK.fill(0)
                    if fx.const_expr(not defer_v):
                        load_v(bn0_page, bn0_part)
                        vm_cnt += num_vm_cnt_load_v
                        # Overlap V loads with QK only while both operand sets fit.
                        fx.rocdl.sched_group_barrier(0x200, 1, 0)
                        fx.rocdl.sched_mfma(2)
                        fx.rocdl.sched_vmem(1)
                        for _ in fx.range_constexpr(num_vm_cnt_load_v//2):
                            fx.rocdl.sched_mfma(3)
                            fx.rocdl.sched_vmem(2)
                        fx.rocdl.sched_vmem(100)
                        fx.rocdl.sched_mfma(100)

                rocdl.sched_barrier(0)
                fxh.s_waitcnt(vmcnt=vm_cnt, lgkmcnt=0)
                rocdl.s_barrier() # ::::::::: wave-group barrier ::::::::: 切换调度
                rocdl.s_setprio(0)
                rocdl.sched_barrier(0)

                if fx.const_expr(is_valid_block_n(block_n)):
                    # q_pos0, kv_len
                    cur_max, l_in, probability_operand = online_softmax(
                        fragS, fragO, qk_scale_log2, cur_max, l_in,
                        q_pos0, block_n, kv_len, full_qo_len,
                        is_all_kv_valid, BN, is_causal, head_dim_qk == 192 and page_size == 128,
                    )

                rocdl.sched_barrier(0)
                rocdl.s_barrier()
                rocdl.s_setprio(1)
                rocdl.sched_barrier(0)

                # MFMA-stage :
                #   1st half: P@V part for block_n
                #   2nd half: Q@K part for block_n+1

                if fx.const_expr(is_valid_block_n(block_n)):
                    if fx.const_expr(defer_v):
                        load_v(bn0_page, bn0_part)
                    fxh.s_waitcnt(vmcnt=0)
                    fx.gemm(tmma2, fragO, fragV, probability_operand, fragO)
                    fragV.fill(0)

                if fx.const_expr(is_valid_block_n(block_n + 1) and not stream_k):
                    # End the P@V operand lifetime before materializing the
                    # full next K fragment. Address work may overlap MFMA, but
                    # two complete matrix operand sets must not coexist.
                    if fx.const_expr(bounded_k and page_size != 32):
                        rocdl.sched_barrier(0)
                    # Swizzled LDS offsets are cheap to recompute here, but
                    # keeping every per-atom address for both rings live over
                    # the entire D192 loop consumes dozens of VGPRs.
                    if fx.const_expr(bounded_k):
                        lds_atom = flyobj.get_universal_copy_atom(k_tile.dtype, 128)
                        lds_copy = fx.make_tiled_copy_A(lds_atom, tmma1).get_slice(_late_thread_id())
                        fx.copy(lds_atom, lds_copy.partition_S(lds_k[None, None, lds_buff_id^1]),
                                lds_copy.retile(fragK))
                    else:
                        flyobj.load_tiled_mma_fragA(tmma1, lds_k, [None, None, lds_buff_id^1], dst=fragK)


                # leave some LDS bandwidth in head of MFMA-stage
                # because head of online-softmax-stage needs LDS
                for _ in fx.range_constexpr(num_bits_fragK//128//2):
                    fx.rocdl.sched_group_barrier(0x100, 2, 0)
                    fx.rocdl.sched_mfma(3)
                fx.rocdl.sched_mfma(100)
                #fx.rocdl.sched_group_barrier(0x200, 1, 0)
                fx.rocdl.sched_barrier(0)
                lds_buff_id = lds_buff_id^1

            return lds_buff_id, cur_max, l_in, kv_page_id1, kv_page_id2, kv_page_id3, kv_page_id4

        if (_late_thread_id() // 256) == 1:
            gpu.barrier()
        cur_max, l_in, page0, page1, page2, page3 = fx.Float32(float("-inf")), fx.Float32(0.0), 0,0,0,_uniform(ptr_kv_page_table[0])
        lds_buff_id = 1
        lds_buff_id, cur_max, l_in, page0, page1, page2, page3 = kv_step(-3, lds_buff_id, cur_max, l_in, page0, page1, page2, page3)
        lds_buff_id, cur_max, l_in, page0, page1, page2, page3 = kv_step(-2, lds_buff_id, cur_max, l_in, page0, page1, page2, page3)
        lds_buff_id, cur_max, l_in, page0, page1, page2, page3 = kv_step(-1, lds_buff_id, cur_max, l_in, page0, page1, page2, page3)

        if fx.const_expr(is_causal):
            # Bottom-right causal diagonal for this Q tile:
            #   kv_pos <= kv_len - full_qo_len + q_pos
            #
            # Pages [0, causal_full_pages) are valid for even the first query
            # row in this tile, so they need no element mask.  Round this
            # prefix down to an even count because the hot loop processes two
            # pages with compile-time LDS buffer IDs 0/1.
            causal_base = kv_len - full_qo_len + q_pos0
            causal_full_pages = (causal_base + 1) // page_size
            num_kv_pages_valid = (causal_full_pages // 2) * 2

            # Only pages intersecting at least one active query row need to be
            # visited by the masked tail.  Later pages are fully causal-masked
            # for the whole Q tile and must be skipped, rather than sent
            # through online softmax as an all-minus-infinity block.
            causal_pages = (causal_base + query_len + page_size - 1) // page_size
            num_kv_pages_to_process = (causal_pages < num_kv_pages).select(
                causal_pages, num_kv_pages
            )
        else:
            # Reserve the final one or two pages for the masked tail.  The last
            # physical page may be ragged; for an even page count its partner
            # is handled by the same specialized pair.
            num_kv_pages_valid = num_kv_pages - 2
            if (num_kv_pages & 1) == 1:
                num_kv_pages_valid = num_kv_pages - 1
            num_kv_pages_to_process = num_kv_pages

        # Seed the loop-carried result outside the loop.  For one-page inputs
        # num_kv_pages_valid is zero, so a value assigned only by `yield`
        # would not dominate the epilogue (and FlyDSL rejects the IR).
        results = [cur_max, l_in, page0, page1, page2, page3]
        for page_i, state in range(0, num_kv_pages_valid, 2, init=results):
            cur_max, l_in, page0, page1, page2, page3 = state
            lds_buff_id, cur_max, l_in, page0, page1, page2, page3 = kv_step(page_i, lds_buff_id, cur_max, l_in, page0, page1, page2, page3)
            lds_buff_id, cur_max, l_in, page0, page1, page2, page3 = kv_step(page_i+1, lds_buff_id, cur_max, l_in, page0, page1, page2, page3)
            results = yield [cur_max, l_in, page0, page1, page2, page3]

        # Process the specialized tail in page pairs.  Non-causal has only one
        # or two tail pages; causal may have several pages intersected by this
        # Q tile's diagonal.
        # Keep lds_buff_id as the compile-time constants 0/1: kv_step uses it
        # to index Python fragment lists, so deriving it from the dynamic
        # induction variable (page_i & 1) is not legal FlyDSL.
        for page_i, state in range(
            num_kv_pages_valid, num_kv_pages_to_process, 2, init=results
        ):
            cur_max, l_in, page0, page1, page2, page3 = state
            lds_buff_id, cur_max, l_in, page0, page1, page2, page3 = kv_step(
                page_i,
                lds_buff_id, cur_max, l_in, page0, page1, page2, page3,
                is_all_kv_valid=False,
            )

            if fx.Int32(page_i + 1) < num_kv_pages_to_process:
                lds_buff_id, cur_max, l_in, page0, page1, page2, page3 = kv_step(
                    page_i+1,
                    lds_buff_id, cur_max, l_in, page0, page1, page2, page3,
                    is_all_kv_valid=False,
                )
            results = yield [cur_max, l_in, page0, page1, page2, page3]

        cur_max, l_in, page0, page1, page2, page3 = results
        l = fxh.eltwise_op("v_add_f32", l_in, l_in.shuffle_xor(32, 64))
        fragO.store(fragO.load() * (v_s / l))

        if fx.const_expr(with_lse):
            lse_tid = _late_thread_id()
            query_in_tile = (lse_tid // 64) * 32 + (lse_tid & 31)
            if ((lse_tid & 63) < 32) & (query_in_tile < query_len):
                wave_sum = fx.Float32(l)
                log_l = fx.Float32(llvm.call_intrinsic(
                    fx.Float32.ir_type, "llvm.log2.f32", [arith.unwrap(wave_sum)], [], []
                ))
                lse_tile[fx.Int64(query_in_tile)] = (wave_sum > 0.0).select(
                    (fx.Float32(cur_max) + log_l) * fx.Float32(math.log(2.0)),
                    fx.Float32(float("-inf")),
                )

        fragO_bf16 = fxh.cvt_f32_to_bf16(fragO)

        if fx.const_expr(0):
            # direct store to vmem
            if (_late_thread_id() // 256) == 0:
                gpu.barrier()

            flyobj.store_tiled_mma_fragC(tmma2, fragO_bf16, select(o_tile, [1,0]), copy_atom_bits=64)
        else:
            # 128-bit C-shuffle epilogue:
            #   MFMA C registers --64b--> LDS --128b--> registers --128b--> HBM.
            # The first barrier also makes it safe to reuse the K/O union storage;
            # the last one guarantees every LDS read finishes before the next
            # persistent work item starts using the union as K storage again.
            if (_late_thread_id() // 256) == 0:
                gpu.barrier()

            # C-shuffle aliases the same output LDS bytes through two layouts:
            # tmma2 writes its logical C=(N, M) fragment with N contiguous, while
            # the epilogue reads the physical tensor as row-major (M, N).  The
            # bf16 swizzle removes the bank conflicts from the 64-bit C stores.
            swz_o = fx.SwizzleType.get(3, 3, 3)
            layout_o_lds_store = fx.make_composed_layout(
                fx.static(swz_o),
                fx.make_ordered_layout((head_dim_v, BM//num_waves), order=(0, 1)),
            )
            assert head_dim_v % 8 == 0, f"{head_dim_v=} must be a multiple of 8"
            num_dw4_items = (head_dim_v // 8) * (BM//num_waves)
            layout_o_lds_read = fx.make_composed_layout(
                fx.static(swz_o),
                fx.make_ordered_layout((8, num_dw4_items), order=(0, 1)),
            )
            o_lds_store = lds.o_lds.peek().view(layout_o_lds_store)
            o_lds_read = lds.o_lds.peek().view(layout_o_lds_read)

            # Do not keep all eight output-wave masks and C-shuffle addresses
            # live through the attention loop. Their values are used only here.
            epilogue_tid = _late_thread_id()
            lane_id = epilogue_tid % 64
            epilogue_wave = epilogue_tid // 64
            cshuf_atom_w = flyobj.get_universal_copy_atom(fx.BFloat16, 64)
            cshuf_store = fx.make_tiled_copy_C(cshuf_atom_w, tmma2).get_slice(lane_id)
            cshuf_atom_r = flyobj.get_universal_copy_atom(fx.BFloat16, 128)
            out_atom_w = flyobj.get_buffer_copy_atom(fx.BFloat16, 128)

            # o_lds_read [32, head_dim_v]
            # o_tile (BM, head_dim_v):(d0, 1)
            o_tile = select(o_tile, [1,0]) # (head_dim_v, BM):(1, d0)
            o_tile = flat_divide(o_tile, [8, 32]) # (8, 32, head_dim_v//8, BM//32):(1, d0)
            o_tile = group(select(o_tile, [0, 2, 1, 3]), 1, 3)

            fragO_r = cshuf_store.retile(fragO_bf16)
            thrv_o_lds_store = cshuf_store.partition_D(o_lds_store)

            for src_wave in fx.range_constexpr(num_waves):
                # due to limited LDS space for output C-shuffle, do it one wave after another
                if epilogue_wave == src_wave:
                    fx.copy(cshuf_atom_w, fragO_r, thrv_o_lds_store)

                gpu.barrier()

                frag = fx.make_fragment_like(o_lds_read[None, 0])
                for item in range(epilogue_tid, num_dw4_items, num_threads):
                    src = o_lds_read[None, item]
                    dst = o_tile[None, item, src_wave]
                    fx.copy(cshuf_atom_r, src, frag)
                    fx.copy(out_atom_w, frag, dst)
                gpu.barrier()


    @flyc.kernel(known_block_size=[num_threads, 1, 1])
    def attn_kernel(
        Q_: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        cu_seqlens_q: fx.Tensor,
        cu_seqlens_k: fx.Tensor,
        kv_indptr: fx.Tensor,
        kv_page_indices: fx.Tensor,
        q_descale: fx.Tensor,
        k_descale: fx.Tensor,
        v_descale: fx.Tensor,
        kv_last_page_lens: fx.Tensor,
        O_: fx.Tensor,
        LSE: fx.Tensor,
        work_counter: fx.Tensor,
    ):
        tid = fx.thread_idx.x

        batch_size = fx.size(cu_seqlens_q.shape).to_py_value() - 1

        #assert 0, f"{Q_}\n{K_}\n{V_}\n{cu_seqlens_q}\n{kv_indptr}\n{kv_page_indices}\n{q_descale}\n{k_descale}\n{v_descale}\n{kv_last_page_lens}\n{out}"

        #if tid == 0:
        #    fx.printf("[{}.{}.{}] batch_size = {}", i_wg, i_head_qo, i_head_kv,  batch_size)

        @flyc.jit
        def fetch_work(work_counter, tid):
            # Only lane 0 of wave 0 performs one device-scope fetch-add for
            # the whole workgroup.  Store the result in a per-workgroup global
            # mailbox, then use a workgroup barrier to broadcast it to all
            # eight waves.  A wave shuffle alone cannot cross wave boundaries.
            # Materialize the mailbox offset here, not as another live 64-bit
            # counter pointer across every attention iteration.
            mailbox = fx.Int32(llvm.inline_asm(
                fx.Int32.ir_type, [fx.Int32(fx.block_idx.x).ir_value()],
                "s_mov_b32 $0, $1", "=s,s", has_side_effects=True,
            )) + 1
            if _late_thread_id() == 0:
                addr = fx.ptrtoint(fx.get_iter(work_counter))
                llvm_ptr = llvm.inttoptr(
                    ir.Type.parse("!llvm.ptr<1>"), as_ir_value(addr)
                )
                old = llvm.AtomicRMWOp(
                    llvm.AtomicBinOp.add,
                    llvm_ptr,
                    as_ir_value(fx.Int32(1)),
                    llvm.AtomicOrdering.monotonic,
                    syncscope="agent",
                    alignment=4,
                )
                work_counter[mailbox] = fx.Int32(old.result)
                fxh.s_waitcnt(vmcnt=0)
            gpu.barrier()
            ticket = work_counter[mailbox]
            fxh.s_waitcnt(vmcnt=0)
            gpu.barrier()
            # Every lane reads the same CTA mailbox after the barriers. If
            # left as a vector value, this poisons the persistent loop's batch
            # and query bounds, forcing a waterfall for each Q/O descriptor.
            return _uniform(ticket)

        # Dynamic ticket dispenser: the host initializes the counter to the
        # number of initially resident workgroups.  Each workgroup first owns
        # its block id, then fetches additional work when it finishes.
        linear_work_idx = fx.Int32(fx.block_idx.x)
        batch_i = fx.Int32(0)
        head_i = fx.Int32(0)
        cur_work_idx = fx.Int32(0)
        works_per_head = fx.Int32(((_uniform(cu_seqlens_q[1]) - _uniform(cu_seqlens_q[0])) + (BM - 1))//(BM))
        k_s = k_descale[0]
        v_s = v_descale[0]

        @flyc.jit
        def skip_works(num_works, cur_work_idx, head_i, batch_i, works_per_head):
            cur_work_idx += num_works
            while (batch_i < batch_size) & (cur_work_idx >= works_per_head):
                cur_work_idx -= works_per_head
                head_i = head_i + 1
                if head_i >= num_qo_heads:
                    head_i = 0
                    batch_i = batch_i + 1
                    if batch_i < batch_size:
                        works_per_head = ((_uniform(cu_seqlens_q[batch_i + 1]) - _uniform(cu_seqlens_q[batch_i])) + (BM - 1))//(BM)
            return cur_work_idx, head_i, batch_i, works_per_head

        cur_work_idx, head_i, batch_i, works_per_head = skip_works(
            linear_work_idx, cur_work_idx, head_i, batch_i, works_per_head
        )

        while batch_i < batch_size:
            # process the work
            query_pos0 = cur_work_idx * BM
            sequence_start = _uniform(cu_seqlens_q[batch_i])
            sequence_end = _uniform(cu_seqlens_q[batch_i + 1])
            query_start = sequence_start + query_pos0
            query_end = fx.Int32(arith.minsi(arith.unwrap(query_start + BM), arith.unwrap(sequence_end)))
            query_len = query_end - query_start
            full_qo_len = sequence_end - sequence_start

            kv_ind_start = _uniform(kv_indptr[batch_i])   # workgroup-uniform i32
            kv_ind_end = _uniform(kv_indptr[batch_i + 1])
            num_kv_pages = kv_ind_end - kv_ind_start # i32
            last_page_len = _uniform(kv_last_page_lens[batch_i])
            kv_len = (num_kv_pages - 1) * page_size + last_page_len


            """
            page_size 是一个在kv-length维度上的天然的分块，因为我们步进 BN 选择了32,
            因此page_size也要求是32的倍数以降低复杂度。
            """
            head_qo = head_i
            head_kv = (head_qo * num_kv_heads) // num_qo_heads

            # process:
            #      Q_[query_start:query_end, head_qo, head_dim]
            #      O_[query_start:query_end, head_qo, head_dim]
            # q_descale[query_start:query_end, head_qo, 1]
            q_tile = fx.make_view(fx.get_iter(Q_) + query_start * num_qo_heads * head_dim_qk,
                                  fx.make_ordered_layout((BM, num_qo_heads, head_dim_qk),(2, 1, 0)))
            q_tile = fx.rocdl.make_buffer_tensor(q_tile, max_size=False,
                                                 num_records_bytes = query_len * num_qo_heads * head_dim_qk * (q_tile.dtype.width // 8))
            q_tile = q_tile[None, head_qo, None]

            if fx.const_expr(per_token):
                qs_tile = fx.make_view(fx.get_iter(q_descale) + query_start * num_qo_heads,
                                    fx.make_ordered_layout((BM, num_qo_heads),(1, 0)))
                qs_tile = fx.rocdl.make_buffer_tensor(qs_tile, max_size=False,
                                                    num_records_bytes = query_len * num_qo_heads * (qs_tile.dtype.width // 8))
                qs_tile = qs_tile[None, head_qo]
                # [TRICKY#1] this scale assumes 1 32x32 MFMA
                scale_tid = _late_thread_id() if fx.const_expr(head_dim_qk == 192 and page_size == 32) else tid
                query_in_tile = (fx.Int32(scale_tid // 64) * fx.Int32(32)) + fx.Int32(scale_tid % 32)
                value_q_descale = qs_tile[query_in_tile]
            else:
                # per-tensor
                value_q_descale = (fx.get_iter(q_descale))[0]

            qk_scale_log2 = value_q_descale * k_s * fx.Float32(sm_scale_log2)

            o_tile = fx.make_view(fx.get_iter(O_) + query_start * num_qo_heads * head_dim_v,
                                  fx.make_ordered_layout((BM, num_qo_heads, head_dim_v),(2, 1, 0)))
            o_tile = fx.rocdl.make_buffer_tensor(o_tile, max_size=False,
                                                 num_records_bytes = query_len * num_qo_heads * head_dim_v * (o_tile.dtype.width // 8))
            o_tile = o_tile[None, head_qo, None]

            lse_tile = LSE
            if fx.const_expr(with_lse):
                # query_start includes the per-sequence prefix AND this tile's
                # offset. One i64 pointer offset avoids mixed-width lowering.
                lse_tile = fx.make_view(
                    fx.get_iter(LSE) + (fx.Int64(query_start) * num_qo_heads + fx.Int64(head_qo)),
                    fx.make_layout(BM, num_qo_heads),
                )

            # Vectorized K: [page, BN-page, kv_head, D/vector, BN, vector]
            # Public V is [page, kv_head, page/vector, D, vector]. The launch
            # view splits page/vector into [num_BN_per_page, BN/vector].
            #       =>
            # k_tile: [BN, (k_vector_size, head_dim // k_vector_size), num_physical_pages, num_BN_per_page]
            # v_tile: [head_dim, (k_vector_size, BN // k_vector_size), num_physical_pages, num_BN_per_page]
            k_tile = K[None, None, head_kv, None, None, None]
            k_tile = select(k_tile, (3, 4, 2, 0, 1))
            k_tile = group(k_tile, 1, 3)
            # V has a public [page, head, page/vector, D, vector] layout. The
            # launch view splits page/vector into BN pages without moving data.
            v_tile = V[None, None, head_kv, None, None, None] # [num_physical_pages, num_BN_per_page, BN // k_vector_size, head_dim, k_vector_size]
            v_tile = select(v_tile, (3, 4, 2, 0, 1))    # [head_dim, k_vector_size, BN // k_vector_size, num_physical_pages, num_BN_per_page]
            v_tile = group(v_tile, 1, 3)                # [head_dim, (k_vector_size, BN // k_vector_size), num_physical_pages, num_BN_per_page]

            buf_kv_page_table = fx.make_view(fx.get_iter(kv_page_indices) + kv_ind_start,
                                             fx.make_layout(num_kv_pages,1))
            buf_kv_page_table = fx.rocdl.make_buffer_tensor(buf_kv_page_table, max_size=False)

            attn_pipeline(q_tile, k_tile, v_tile, o_tile, lse_tile,
                          query_pos0, query_len, kv_len, full_qo_len,
                          buf_kv_page_table,
                          num_kv_pages,  last_page_len,
                          qk_scale_log2, v_s)

            next_linear_work_idx = fetch_work(work_counter, tid)
            linear_work_delta = next_linear_work_idx - linear_work_idx
            linear_work_idx = next_linear_work_idx
            cur_work_idx, head_i, batch_i, works_per_head = skip_works(
                linear_work_delta, cur_work_idx, head_i, batch_i, works_per_head
            )


    @flyc.jit
    def launch(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        cu_seqlens_q: fx.Tensor,
        cu_seqlens_k: fx.Tensor,
        kv_indptr: fx.Tensor,
        kv_page_indices: fx.Tensor,
        q_descale: fx.Tensor,
        k_descale: fx.Tensor,
        v_descale: fx.Tensor,
        kv_last_page_lens: fx.Tensor,
        out: fx.Tensor,
        LSE: fx.Tensor,
        work_counter: fx.Tensor,
        num_workgroups: fx.Int32,
        stream: fx.Stream,
    ):
        num_query_tokens = Q.shape[0].to_py_value()
        num_physical_pages = V.shape[0].to_py_value()
        k_vector_size = 128 // K.dtype.width
        Q = fxh.view_as_torch_tensor(Q, (num_query_tokens, num_qo_heads, head_dim_qk))
        K = fxh.view_as_torch_tensor(
            K,
            (
                num_physical_pages,
                num_kv_heads,
                head_dim_qk // k_vector_size,
                num_BN_per_page,
                BN,
                k_vector_size,
            ),
        )
        K = select(K, (0, 3, 1, 2, 4, 5))
        V = fxh.view_as_torch_tensor(V, (num_physical_pages, num_kv_heads, num_BN_per_page, BN//k_vector_size, head_dim_v, k_vector_size))
        V = select(V, (0, 2, 1, 3, 4, 5))

        if fx.const_expr(not per_token):
            q_descale = fxh.view_as_torch_tensor(q_descale, (1,))
        else:
            q_descale = fxh.view_as_torch_tensor(q_descale, (num_query_tokens, num_qo_heads, 1))
        k_descale = fxh.view_as_torch_tensor(k_descale, (1,))
        v_descale = fxh.view_as_torch_tensor(v_descale, (1,))
        out = fxh.view_as_torch_tensor(out, (num_query_tokens, num_qo_heads, head_dim_v))
        if fx.const_expr(with_lse):
            LSE = fxh.view_as_torch_tensor(LSE, (num_query_tokens, num_qo_heads))
        value_attrs = {
            "passthrough": [
                ["target-features", "-packed-fp32-ops"] # disable v_pk_mul (which has co-issue problem with MFMA)
            ],
        }
        attn_kernel(
            Q,
            K,
            V,
            cu_seqlens_q,
            cu_seqlens_k,
            kv_indptr,
            kv_page_indices,
            q_descale,
            k_descale,
            v_descale,
            kv_last_page_lens,
            out,
            LSE,
            work_counter,
            value_attrs=value_attrs,
        ).launch(grid=(num_workgroups, 1, 1), block=(num_threads, 1, 1), stream=stream)

    return launch


class _PagedAttention:
    bf16_backend = "native-8wave"

    def __init__(self, heads, kv_heads, dq, dv, page_size, causal, quant_query_mode):
        self.heads, self.kv_heads = heads, kv_heads
        self.dq, self.dv, self.page_size = dq, dv, page_size
        self.causal, self.quant_query_mode = causal, quant_query_mode
        self.memory_mode, self.persistent = "lds", True
        self._builders = {}
        self._compiled = {}

    def __call__(self, Q, K, V, cu_seqlens_q, cu_seqlens_k, kv_indptr, kv_page_indices,
                 max_seqlen_q, max_seqlen_k, causal, q_descale, k_descale, v_descale,
                 kv_last_page_lens, out=None, sink_ptr=None, stream=None, *, return_lse=False, lse=None, softmax_scale=None):
        """Run the original BF16 pipeline with contiguous, caller-owned buffers.

        K is [pages, KV heads, Dq/8, page_size, 8]; V is
        [pages, KV heads, page_size/8, Dv, 8]. Descales are contiguous FP32,
        finite and strictly positive. Q scales may be scalar in either mode;
        per-token mode also accepts one value per token/head.

        Prefix sums, page IDs, last-page lengths and maximum lengths must be
        consistent and in bounds. Each active sequence must have nonempty KV;
        causal sequences additionally require KV length >= query length.
        Only metadata shape/type/device are checked, never its GPU values.
        In particular, mixed batches containing empty KV remain unsupported.
        cu_seqlens_k is optional and unused by this vectorized paged pipeline.

        A supplied LSE buffer is written even with return_lse=False. LSE uses
        natural logarithms and has shape [query tokens, query heads]. Outputs
        must not overlap inputs or each other. Empty Q returns without a
        launch or scheduler allocation. The original per-call counter
        allocation is retained; its device-side seed is graph-capture safe.
        """
        if not isinstance(causal, bool) or causal != self.causal:
            raise ValueError("causal must match the factory")
        if sink_ptr is not None:
            raise NotImplementedError("gfx942 BF16 full MHA does not support sinks")
        if not isinstance(return_lse, bool):
            raise ValueError("return_lse must be a bool")
        if not isinstance(Q, torch.Tensor) or not Q.is_cuda:
            raise ValueError("Q/K/V must be tensors on the same gfx942 GPU")
        device = Q.device
        properties = torch.cuda.get_device_properties(device)
        if getattr(properties, "gcnArchName", "").split(":", 1)[0] != "gfx942":
            raise NotImplementedError("this BF16 backend requires gfx942")
        for name, tensor in (("Q", Q), ("K", K), ("V", V)):
            if not isinstance(tensor, torch.Tensor) or tensor.device != device:
                raise ValueError(f"{name} must be a tensor on the input GPU")
            if tensor.dtype != torch.bfloat16:
                raise NotImplementedError("gfx942 BF16 attention requires BF16 Q/K/V")
            if tensor.layout != torch.strided or not tensor.is_contiguous():
                raise ValueError(f"{name} must be contiguous")
            if tensor.numel() * tensor.element_size() >= 2**31:
                raise NotImplementedError(f"{name} byte span must fit signed int32 addressing")
        if Q.ndim != 3 or Q.shape[1:] != (self.heads, self.dq):
            raise ValueError("Q must be [tokens, query heads, head_dim_qk]")
        if (K.ndim != 5 or V.ndim != 5
                or K.shape != (V.shape[0], self.kv_heads, self.dq // 8, self.page_size, 8)
                or V.shape[1:] != (self.kv_heads, self.page_size // 8, self.dv, 8)):
            raise ValueError("K/V must use the factory's BF16 vectorized paged layouts")
        num_query_tokens = Q.shape[0]
        for name, bound in (("max_seqlen_q", max_seqlen_q), ("max_seqlen_k", max_seqlen_k)):
            if not isinstance(bound, int) or isinstance(bound, bool) or not 0 <= bound < 2**31:
                raise ValueError(f"{name} must be a nonnegative signed int32 bound")
        if causal and max_seqlen_k < max_seqlen_q:
            raise ValueError("bottom-right causal attention requires max_seqlen_k >= max_seqlen_q")
        if num_query_tokens and max_seqlen_q == 0:
            raise ValueError("nonempty Q requires max_seqlen_q > 0")

        if cu_seqlens_k is None:
            cu_seqlens_k = cu_seqlens_q
        metadata = (
            ("cu_seqlens_q", cu_seqlens_q), ("cu_seqlens_k", cu_seqlens_k),
            ("kv_indptr", kv_indptr), ("kv_page_indices", kv_page_indices),
            ("kv_last_page_lens", kv_last_page_lens),
        )
        for name, tensor in metadata:
            if (not isinstance(tensor, torch.Tensor) or tensor.ndim != 1
                    or tensor.dtype != torch.int32 or tensor.device != device
                    or tensor.layout != torch.strided or not tensor.is_contiguous()):
                raise ValueError(f"{name} must be contiguous device int32 metadata")
        batch_size = cu_seqlens_q.numel() - 1
        if (batch_size < 0 or (num_query_tokens > 0 and batch_size == 0)
                or cu_seqlens_k.shape != cu_seqlens_q.shape
                or kv_indptr.shape != cu_seqlens_q.shape
                or kv_last_page_lens.numel() != batch_size):
            raise ValueError("inconsistent batch metadata shapes")
        if num_query_tokens and (max_seqlen_k == 0 or K.shape[0] == 0 or kv_page_indices.numel() == 0):
            raise NotImplementedError("empty KV is not supported by the original BF16 pipeline")

        for name, tensor in (("q_descale", q_descale), ("k_descale", k_descale), ("v_descale", v_descale)):
            if (not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.float32
                    or tensor.device != device or tensor.layout != torch.strided
                    or not tensor.is_contiguous()):
                raise ValueError(f"{name} must be contiguous FP32 on the input GPU")
        if k_descale.numel() != 1 or v_descale.numel() != 1:
            raise ValueError("K/V descales must each contain one value")
        if (q_descale.numel() != 1
                and (self.quant_query_mode != "per-token"
                     or q_descale.numel() != num_query_tokens * self.heads)):
            raise ValueError("Q descale must be scalar or, in per-token mode, one value per token/head")
        if isinstance(softmax_scale, torch.Tensor) or isinstance(softmax_scale, bool):
            raise ValueError("softmax_scale must be a host scalar or None")
        scale = None if softmax_scale is None else float(softmax_scale)
        if scale is not None and (not math.isfinite(scale) or scale <= 0):
            raise ValueError("softmax_scale must be finite and positive")

        out_shape, lse_shape = (num_query_tokens, self.heads, self.dv), (num_query_tokens, self.heads)
        if num_query_tokens * self.heads * self.dv * 2 >= 2**31:
            raise NotImplementedError("output byte span must fit signed int32 addressing")
        for name, tensor, shape, dtype in (
            ("out", out, out_shape, torch.bfloat16), ("lse", lse, lse_shape, torch.float32)
        ):
            if tensor is not None and (
                not isinstance(tensor, torch.Tensor) or tensor.shape != shape or tensor.dtype != dtype
                or tensor.device != device or tensor.layout != torch.strided or not tensor.is_contiguous()
            ):
                raise ValueError(f"{name} must be contiguous {dtype} {shape} on the input GPU")
        stream = torch.cuda.current_stream(device) if stream is None else stream
        if getattr(stream, "device", None) != device or not hasattr(stream, "cuda_stream"):
            raise ValueError("stream must belong to the input GPU")

        with torch.cuda.device(device), torch.cuda.stream(stream):
            if out is None:
                out = torch.empty(out_shape, device=device, dtype=torch.bfloat16)
            if return_lse and lse is None:
                lse = torch.empty(lse_shape, device=device, dtype=torch.float32)
            if num_query_tokens == 0:
                return (out, lse) if return_lse else out

            with_lse = lse is not None
            mode = "per-tensor" if q_descale.numel() == 1 else "per-token"
            specialization = (scale, with_lse, mode)
            launch = self._builders.get(specialization)
            if launch is None:
                launch = _build_attention(
                    self.heads, self.kv_heads, self.dq, self.dv, self.page_size,
                    self.causal, mode, softmax_scale=scale, with_lse=with_lse,
                )
                self._builders[specialization] = launch

            # Preserve the baseline's allocation/seed overhead and ticket
            # scheduling: slot 0 is the counter, slots 1..N are CTA mailboxes.
            # This is per-call state, not a counter shared by concurrent streams.
            num_workgroups = properties.multi_processor_count
            work_counter = torch.zeros(num_workgroups + 1, device=device, dtype=torch.int32)
            # Device-side fill is capture-safe; scalar assignment otherwise
            # stages a pageable CPU tensor and fails inside a CUDA graph.
            work_counter[:1].fill_(num_workgroups)
            q_scale, k_scale, v_scale = (tensor.view(-1) for tensor in (q_descale, k_descale, v_descale))
            args = (
                Q, K, V, cu_seqlens_q, cu_seqlens_k, kv_indptr, kv_page_indices,
                q_scale, k_scale, v_scale, kv_last_page_lens, out,
                lse if with_lse else k_scale, work_counter, num_workgroups, stream,
            )
            signature = tuple(
                (arg.device, arg.dtype, tuple(arg.shape), tuple(arg.stride()))
                if isinstance(arg, torch.Tensor) else ("stream",) if hasattr(arg, "cuda_stream") else arg
                for arg in args
            )
            cache_key = (launch, properties.gcnArchName, signature)
            compiled = self._compiled.get(cache_key)
            if compiled is None:
                # flyc.compile both compiles and performs the first launch.
                self._compiled[cache_key] = flyc.compile(launch, *args)
            else:
                compiled(*args)
        return (out, lse) if return_lse else out


@functools.cache
def PagedAttention(num_qo_heads, num_kv_heads, head_dim_qk, head_dim_v, page_size,
                   is_causal, quant_query_mode="per-token", key_layout="vectorized",
                   window_left=-1, has_sink=False, *, memory_mode="lds", persistent=None):
    """Return BF16 gfx942 full MHA without changing its native 8-wave algorithm.

    persistent=None/True select the original ticket scheduler; False is not
    implemented. Only vectorized K and memory_mode='lds' are available. The
    original D128/D192, V128/V192 and page32/64/128 parameterization is retained;
    the factory does not substitute a different backend for any configuration.
    """
    if memory_mode != "lds":
        raise NotImplementedError("gfx942 BF16 supports memory_mode='lds' only")
    if persistent is not None and not isinstance(persistent, bool):
        raise ValueError("persistent must be a bool or None")
    if persistent is False:
        raise NotImplementedError("gfx942 BF16 always uses the original persistent scheduler")
    if key_layout != "vectorized":
        raise NotImplementedError("gfx942 BF16 full MHA supports vectorized K only")
    if window_left != -1 or has_sink:
        raise NotImplementedError("gfx942 BF16 full MHA does not support SWA or sinks")
    if not isinstance(is_causal, bool) or not isinstance(has_sink, bool):
        raise ValueError("is_causal and has_sink must be bools")
    if any(not isinstance(value, int) or isinstance(value, bool) for value in (
        num_qo_heads, num_kv_heads, head_dim_qk, head_dim_v, page_size
    )):
        raise ValueError("head counts, dimensions and page_size must be integers")
    if num_qo_heads <= 0 or num_kv_heads <= 0 or num_qo_heads % num_kv_heads:
        raise ValueError("query heads must be a positive multiple of KV heads")
    if head_dim_qk not in (128, 192) or head_dim_v not in (128, 192) or page_size not in (32, 64, 128):
        raise NotImplementedError("gfx942 BF16 supports D128/D192, V128/V192 and pages 32/64/128")
    if quant_query_mode not in ("per-token", "per-tensor"):
        raise ValueError("query scale mode must be 'per-token' or 'per-tensor'")
    return _PagedAttention(
        num_qo_heads, num_kv_heads, head_dim_qk, head_dim_v, page_size, is_causal, quant_query_mode
    )
