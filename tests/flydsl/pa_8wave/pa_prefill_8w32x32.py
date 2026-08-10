import functools
import os

os.environ.setdefault("FLYDSL_RUNTIME_ENABLE_CACHE", "0")

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl.expr.typing import T, as_ir_value
from flydsl.expr import arith, const_expr, gpu, range_constexpr, rocdl
from flydsl._mlir.dialects import llvm, vector
from flydsl.expr.typing import Vector as Vec

import pyhip.contrib.flydsl.helpers as fxh

fxh.dump_ir(True)

import pyhip

if __name__ == "__main__":
    pyhip.set_device()


def _maxnumf(a, b):
    """Non-NaN-propagating f32 max used by the wave softmax reduction."""
    return type(a)(arith.maxnumf(arith.unwrap(a), arith.unwrap(b)))

@flyc.jit
def online_softmax(fragS, fragO, sm_scale_log2, old_max, l_in,
                   q_pos0, kv_block_n, kv_len, qo_len,
                   is_all_kv_valid: fx.Constexpr[bool],
                   KV_BLOCK_SIZE: fx.Constexpr[int],
                   is_causal: fx.Constexpr[bool]):
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
        lane_id = fx.thread_idx.x & 63
        col_lane = (lane_id < 32).select(fx.Int32(0), fx.Int32(16))
        col_block = fx.Int32(kv_block_n * KV_BLOCK_SIZE)
        if fx.const_expr(not is_causal):
            # Keep both sides explicitly i32.  A Python constexpr loop index
            # otherwise promotes this comparison to MLIR index, whose ordered
            # comparison is unsigned; a negative limit would then look huge
            # and leave invalid tail columns unmasked.
            for i in fx.range_constexpr(16):
                kv_pos = col_block + col_lane + fx.Int32(i)
                if kv_pos >= kv_len:
                    fragS[i,0,0] = float("-inf")
        else:
            # Bottom-right causal mask:
            #   kv_pos <= kv_len - qo_len + q_pos
            wave_id = fx.thread_idx.x // 64
            row_lane = fx.thread_idx.x & 31
            q_pos = q_pos0 + wave_id * 32 + row_lane
            causal_limit = kv_len - qo_len + q_pos
            for i in fx.range_constexpr(16):
                kv_pos = col_block + col_lane + fx.Int32(i)
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

    # Rebase the accumulated numerator only when the lazy max advances.
    def rescale_output():
        fragO.store(fxh.eltwise_op("v_mul_f32", fragO.load(), corr))

    @flyc.jit
    def rescale_if_needed():
        if corr < fx.Float32(1.0):
            rescale_output()

    rescale_if_needed()
    return new_max, l_out


@functools.cache
def PagedAttention(num_qo_heads, num_kv_heads, head_dim_qk, head_dim_v, page_size, is_causal):
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
    assert page_size in [32, 64, 128]
    num_BN_per_page = page_size // BN
    LOG2E = 1.4426950408889634
    sm_scale_log2 = float(LOG2E / (head_dim_qk**0.5))

    assert (page_size % BN) == 0, f"{page_size=} must be a multiple of {BN=}"

    @flyc.jit
    def attn_pipeline(q_tile, # [query_start:query_end, head_qo, head_dim_qk]
                      k_tile, # [BN, (k_vector_size, head_dim_qk // k_vector_size), num_physical_pages, num_BN_per_page]
                      v_tile, # [head_dim_v, (k_vector_size, BN // k_vector_size), num_physical_pages, num_BN_per_page]
                      o_tile, # [query_start:query_end, head_qo, head_dim_v]
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

        """
        [TRICKY]:
        P@V的gemm中V的reduction维度的layout：
            lane[0] 0..15
            lane[32] 16..31
        而P的reduction维度的layout，跟K的n维度的关系满足MFMA的输出layout:
            lane[0] 0..3/8..11/16..29/24..27
            lane[32] 4..7/12..15/20..23/28..31
        
        为了避免额外的reorder开销：
            对K的layout中n维度进行remap, 用这个layout进行compose (4, 2, 4):(1, 16, 4) 使得P的lane[0]/[32]跟V一致
            对P的寄存器排布，按照 fx.gemm 对 A/B 输入的要求重新解释
        """
        k_tile = fx.composition(
            k_tile,
            fx.make_tile(fx.make_layout((4, 2, 4), (1, 16, 4)), None, None, None),
        )

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
        fragK = tmma1.make_fragment_A(k_fake)
        fragV = tmma2.make_fragment_A(v_fake)
        num_bits_fragK = (fx.size(fragK.shape).get_static_leaf_int * fragK.dtype.width)
        num_bits_fragV = (fx.size(fragV.shape).get_static_leaf_int * fragV.dtype.width)
        num_vm_cnt_load_v = (num_bits_fragV)//128

        fakeCt = fx.make_rmem_tensor(fx.make_layout((BN, BM), (BM, 1)), fx.Float32)
        fragS = tmma1.make_fragment_C(fakeCt)
        fragO = tmma2.make_fragment_C(fx.select(o_tile, [1, 0]))
        """
        [TRICKY]:
        
        """
        prob_operand = fx.make_rmem_tensor(
            fx.make_layout((8, 1, 2), (1, 0, 8)),
            v_tile.dtype,
        )

        # let all 512 threads participate in the copy so no extra if condition involved
        # 512*16/32 = 256, so all head_dim <= 256 can be padded to 256
        copy_atom_bits = 64 if head_dim_qk == 128 else 128

        @fx.union
        class SharedStorage:
            k_lds: fx.Array[k_tile.dtype, 2 * BN * head_dim_qk, 16]
            o_lds: fx.Array[o_tile.dtype, BM * head_dim_v, 16]

        # mask,base,shift, swizzle always in unit of 128b,
        swz_base = ((128 // k_tile.dtype.width) - 1).bit_length()
        swz = fx.SwizzleType.get(3, swz_base, 3)
        lds = fx.SharedAllocator().allocate(SharedStorage)
        layout_k_lds = fx.make_composed_layout(
            fx.static(swz),
            fx.make_ordered_layout((BN, head_dim_qk, 2), (1, 0, 2)),
        )
        lds_k = lds.k_lds.peek().view(layout_k_lds)

        # C-shuffle aliases the same output LDS bytes through two layouts:
        # tmma2 writes its logical C=(N, M) fragment with N contiguous, while
        # the epilogue reads the physical tensor as row-major (M, N).  The
        # bf16 swizzle removes the bank conflicts from the 64-bit C stores.
        swz_o = fx.SwizzleType.get(3, 3, 3)
        layout_o_lds_store = fx.make_composed_layout(
            fx.static(swz_o),
            fx.make_ordered_layout((head_dim_v, BM), order=(0, 1)),
        )
        layout_o_lds_read = fx.make_composed_layout(
            fx.static(swz_o),
            fx.make_ordered_layout((BM, head_dim_v), order=(1, 0)),
        )
        o_lds_store = lds.o_lds.peek().view(layout_o_lds_store)
        o_lds_read = lds.o_lds.peek().view(layout_o_lds_read)

        # assert 0, f"{lds_ku32} {lds_k}"

        def is_valid_block_n(bn):
            #return fx.const_expr(bn >= 0 and bn < num_kv_pages) if fx.const_expr(isinstance(bn, int)) else True
            return fx.const_expr(bn >= 0) if fx.const_expr(isinstance(bn, int)) else True

        # k_tile layout: Tensor<f8E4M3FNUZ, global, ((4,2,4),(16,8),?):((16,256,64),(1,512),4096)>
        # assert 0, f"{k_tile}"
        num_copy_threads = BN * head_dim_qk * k_tile.dtype.width // copy_atom_bits
        assert BN * head_dim_qk * k_tile.dtype.width % copy_atom_bits == 0
        assert num_copy_threads <= num_threads

        # [TRICKY]
        # Keep the global->register->LDS pipeline packed in 32-bit dwords.  If
        # the FP8 fragment crosses a loop backedge as vector<16xi8>, LLVM
        # scalarizes it into byte values and later emits shifts/v_perm to pack
        # it again for ds_write_b128.  Recasting the already-partitioned,
        # contiguous per-thread slice preserves its byte address (including
        # the LDS swizzle) while making the loop-carried value vector<Nxi32>.
        def recast_tensor(src, new_dtype):
            result_type = fx.PointerType.get(new_dtype.ir_type, src.memspace, new_dtype.width//8)
            new_iter = fx.recast_iter(result_type, fx.get_iter(src))
            new_layout = fx.recast_layout(src.layout, src.dtype.width, new_dtype.width)
            return fx.make_view(new_iter, new_layout)

        lds_k_u32 = recast_tensor(lds_k, fx.Uint32)
        k_tile_u32 = recast_tensor(k_tile, fx.Uint32)

        glk_thrcopy, _ = flyobj.get_tiled_copy_coalesced_mn(
            k_tile_u32[None, None, 0, 0],
            copy_atom_bits=copy_atom_bits,
            num_threads=num_copy_threads,
        )
        glk_srck = glk_thrcopy.partition_S(k_tile_u32)
        glk_dstk = glk_thrcopy.partition_D(lds_k_u32)

        glk_cp_atom = flyobj.get_universal_copy_atom(fx.Uint32, copy_atom_bits)
        glk_frag = fx.make_fragment_like(glk_dstk[None, None, None, 0])
        num_vm_cnt_load_k = (fx.size(glk_frag.shape).get_static_leaf_int * glk_frag.dtype.width)//copy_atom_bits
        prefetch_fragk_list = [
            fx.make_fragment_like(glk_srck[None, None, None, 0, 0]),
            fx.make_fragment_like(glk_srck[None, None, None, 0, 0]),
        ]

        def global_load_k(block_n, page_id, bn_id, frag_id):
            if fx.const_expr(is_valid_block_n(block_n)):
                if fx.const_expr(num_copy_threads == num_threads):
                    fx.copy(glk_cp_atom, glk_srck[None, None, None, page_id, bn_id], prefetch_fragk_list[frag_id])
                else:
                    if tid < num_copy_threads:
                        fx.copy(glk_cp_atom, glk_srck[None, None, None, page_id, bn_id], prefetch_fragk_list[frag_id])
                return num_vm_cnt_load_k
            else:
                return 0

        def ds_store_k(block_n, frag_id, lds_buff_id):
            if fx.const_expr(is_valid_block_n(block_n)):
                if fx.const_expr(num_copy_threads == num_threads):
                    fx.copy(glk_cp_atom, prefetch_fragk_list[frag_id], glk_dstk[None, None, None, lds_buff_id & 1])
                else:
                    if tid < num_copy_threads:
                        fx.copy(glk_cp_atom, prefetch_fragk_list[frag_id], glk_dstk[None, None, None, lds_buff_id & 1])
    
        fragO.fill(0.0)

        v_copy_atom = flyobj.get_universal_copy_atom(v_tile.dtype, 128)
        v_tcopy = flyobj.get_tiled_mma_copy(v_copy_atom, tmma2, "A")
        v_thrcopy = v_tcopy.get_slice(tid)

        def kv_step(page_n, lds_buff_id, cur_max, l_in,
                    kv_page_id0, kv_page_id1, kv_page_id2, kv_page_id3,
                    is_all_kv_valid: fx.Constexpr[bool] = True):
            # first block_n in pipeline is -3
            kv_page_id4 = ptr_kv_page_table[page_n + 4]

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
                vm_cnt += global_load_k(block_n + 3, bn3_page, bn3_part, prefetch_frag_id)   # 
                
                if fx.const_expr(is_valid_block_n(block_n)):
                    fragS.fill(0.0)
                    #s_waitcnt(lgkmcnt=0)
                    fx.gemm(tmma1, fragS, fragK, fragQ, fragS)
                    fx.copy(
                        v_copy_atom,
                        v_thrcopy.partition_S(v_tile[None, None, bn0_page, bn0_part]),
                        v_thrcopy.retile(fragV),
                    )

                    vm_cnt += num_vm_cnt_load_v
                    #assert 0, f"{vm_cnt} {num_vm_cnt_load_v}"

                    # Issue all eight V loads across the first eight QK MFMAs. The
                    # final eight MFMAs hide the latency of the last V load.
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
                    cur_max, l_in = online_softmax(fragS, fragO, qk_scale_log2, cur_max, l_in,
                                                q_pos0, block_n, kv_len, full_qo_len,
                                                is_all_kv_valid,
                                                BN,
                                                is_causal)

                rocdl.sched_barrier(0)
                rocdl.s_barrier()
                rocdl.s_setprio(1)
                rocdl.sched_barrier(0)

                # MFMA-stage :
                #   1st half: P@V part for block_n
                #   2nd half: Q@K part for block_n+1
                
                if fx.const_expr(is_valid_block_n(block_n)):
                    vecS = fragS.load()
                    packed_words = []
                    for fn in fx.range_constexpr(4):
                        i = fn * 4
                        lo = rocdl.cvt_pk_fp8_f32(T.i32, vecS[i], vecS[i + 1], fx.Int32(0), False)
                        packed = rocdl.cvt_pk_fp8_f32(T.i32, vecS[i + 2], vecS[i + 3], lo, True)
                        packed_words.append(packed)
                    packed_fp8 = Vec.from_elements(packed_words, fx.Int32).bitcast(prob_operand.dtype)
                    prob_operand.store(packed_fp8)
                    fxh.s_waitcnt(vmcnt=0)
                    fx.gemm(tmma2, fragO, fragV, prob_operand, fragO)

                if fx.const_expr(is_valid_block_n(block_n + 1)):
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

        if wave_m == 1:
            gpu.barrier()
        cur_max, l_in, page0, page1, page2, page3 = fx.Float32(float("-inf")), fx.Float32(0.0), 0,0,0,ptr_kv_page_table[0]
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

        fragO_bf16 = fxh.cvt_f32_to_bf16(fragO)

        if fx.const_expr(0):
            # direct store to vmem
            if wave_m == 0:
                gpu.barrier()

            flyobj.store_tiled_mma_fragC(tmma2, fragO_bf16, fx.select(o_tile, [1,0]), copy_atom_bits=64)
        else:
            # 128-bit C-shuffle epilogue:
            #   MFMA C registers --64b--> LDS --128b--> registers --128b--> HBM.
            # The first barrier also makes it safe to reuse the K/O union storage;
            # the last one guarantees every LDS read finishes before the next
            # persistent work item starts using the union as K storage again.
            cshuf_atom_w = flyobj.get_universal_copy_atom(fx.BFloat16, 64)
            cshuf_store = fx.make_tiled_copy_C(cshuf_atom_w, tmma2).get_slice(tid)
            cshuf_read, cshuf_atom_r = flyobj.get_tiled_copy_coalesced_mn(
                o_lds_read, copy_atom_bits=128, num_threads=num_threads
            )
            out_atom_w = flyobj.get_buffer_copy_atom(fx.BFloat16, 128)

            gpu.barrier()
            fx.copy(
                cshuf_atom_w,
                cshuf_store.retile(fragO_bf16),
                cshuf_store.partition_D(o_lds_store),
            )
            gpu.barrier()
            if wave_m == 0:
                gpu.barrier()

            o_lds_thread = cshuf_read.partition_S(o_lds_read)
            o_thread = cshuf_read.partition_D(o_tile)
            o_coalesced = fx.make_fragment_like(o_lds_thread)
            fx.copy(cshuf_atom_r, o_lds_thread, o_coalesced)
            gpu.barrier()
            fx.copy(out_atom_w, o_coalesced, o_thread)


    @flyc.kernel(known_block_size=[num_threads, 1, 1])
    def attn_kernel(
        Q_: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        cu_seqlens_q: fx.Tensor,
        kv_indptr: fx.Tensor,
        kv_page_indices: fx.Tensor,
        q_descale: fx.Tensor,
        k_descale: fx.Tensor,
        v_descale: fx.Tensor,
        kv_last_page_lens: fx.Tensor,
        O_: fx.Tensor,
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
            if tid == 0:
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
                work_counter[fx.block_idx.x + 1] = fx.Int32(old.result)
                fxh.s_waitcnt(vmcnt=0)
            gpu.barrier()
            ticket = work_counter[fx.block_idx.x + 1]
            fxh.s_waitcnt(vmcnt=0)
            gpu.barrier()
            return ticket

        # Dynamic ticket dispenser: the host initializes the counter to the
        # number of initially resident workgroups.  Each workgroup first owns
        # its block id, then fetches additional work when it finishes.
        linear_work_idx = fx.Int32(fx.block_idx.x)
        batch_i = fx.Int32(0)
        head_i = fx.Int32(0)
        cur_work_idx = fx.Int32(0)
        works_per_head = fx.Int32(((cu_seqlens_q[1] - cu_seqlens_q[0]) + (BM - 1))//(BM))
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
                        works_per_head = ((cu_seqlens_q[batch_i + 1] - cu_seqlens_q[batch_i]) + (BM - 1))//(BM)
            return cur_work_idx, head_i, batch_i, works_per_head

        cur_work_idx, head_i, batch_i, works_per_head = skip_works(
            linear_work_idx, cur_work_idx, head_i, batch_i, works_per_head
        )

        while batch_i < batch_size:
            # process the work
            query_pos0 = cur_work_idx * BM
            query_start = cu_seqlens_q[batch_i] + query_pos0
            query_end = fx.Int32(arith.minsi(arith.unwrap(query_start + BM), arith.unwrap(cu_seqlens_q[batch_i + 1])))
            query_len = query_end - query_start
            full_qo_len = cu_seqlens_q[batch_i + 1] - cu_seqlens_q[batch_i]

            kv_ind_start = kv_indptr[batch_i]   # i32
            kv_ind_end = kv_indptr[batch_i + 1] # i32
            num_kv_pages = kv_ind_end - kv_ind_start # i32
            last_page_len = kv_last_page_lens[batch_i]
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

            qs_tile = fx.make_view(fx.get_iter(q_descale) + query_start * num_qo_heads,
                                   fx.make_ordered_layout((BM, num_qo_heads),(1, 0)))
            qs_tile = fx.rocdl.make_buffer_tensor(qs_tile, max_size=False,
                                                  num_records_bytes = query_len * num_qo_heads * (qs_tile.dtype.width // 8))
            qs_tile = qs_tile[None, head_qo]

            # [TRICKY#1] this scale assumes 1 32x32 MFMA
            query_in_tile = (fx.Int32(tid // 64) * fx.Int32(32)) + fx.Int32(tid % 32)
            qk_scale_log2 = qs_tile[query_in_tile] * k_s * fx.Float32(sm_scale_log2)

            o_tile = fx.make_view(fx.get_iter(O_) + query_start * num_qo_heads * head_dim_v,
                                  fx.make_ordered_layout((BM, num_qo_heads, head_dim_v),(2, 1, 0)))
            o_tile = fx.rocdl.make_buffer_tensor(o_tile, max_size=False,
                                                 num_records_bytes = query_len * num_qo_heads * head_dim_v * (o_tile.dtype.width // 8))
            o_tile = o_tile[None, head_qo, None]

            # K: [num_physical_pages, num_BN_per_page, num_kv_heads, head_dim // k_vector_size, BN, k_vector_size]
            # V: [num_physical_pages, num_BN_per_page, num_kv_heads, BN // k_vector_size, head_dim, k_vector_size]
            #       =>
            # k_tile: [BN, (k_vector_size, head_dim // k_vector_size), num_physical_pages, num_BN_per_page]
            # v_tile: [head_dim, (k_vector_size, BN // k_vector_size), num_physical_pages, num_BN_per_page]
            k_tile = K[None, None, head_kv, None, None, None] # [num_physical_pages, num_BN_per_page, head_dim // k_vector_size, BN, k_vector_size]
            v_tile = V[None, None, head_kv, None, None, None] # [num_physical_pages, num_BN_per_page, BN // k_vector_size, head_dim, k_vector_size]
            k_tile = fx.select(k_tile, (3, 4, 2, 0, 1))    # [BN, k_vector_size, head_dim // k_vector_size, num_physical_pages, num_BN_per_page]
            k_tile = fx.group(k_tile, 1, 3)                # [BN, (k_vector_size, head_dim // k_vector_size), num_physical_pages, num_BN_per_page]
            v_tile = fx.select(v_tile, (3, 4, 2, 0, 1))    # [head_dim, k_vector_size, BN // k_vector_size, num_physical_pages, num_BN_per_page]
            v_tile = fx.group(v_tile, 1, 3)                # [head_dim, (k_vector_size, BN // k_vector_size), num_physical_pages, num_BN_per_page]

            attn_pipeline(q_tile, k_tile, v_tile, o_tile,
                          query_pos0, query_len, kv_len, full_qo_len,
                          fx.get_iter(kv_page_indices) + kv_ind_start,
                          num_kv_pages,  last_page_len,
                          qk_scale_log2, v_s)

            # The next ticket is global rather than grid-stride.  Faster
            # workgroups therefore naturally process more query tiles.
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
        kv_indptr: fx.Tensor,
        kv_page_indices: fx.Tensor,
        q_descale: fx.Tensor,
        k_descale: fx.Tensor,
        v_descale: fx.Tensor,
        kv_last_page_lens: fx.Tensor,
        out: fx.Tensor,
        work_counter: fx.Tensor,
        num_workgroups: fx.Int32,
        stream: fx.Stream,
    ):
        num_query_tokens = Q.shape[0].to_py_value()
        num_physical_pages = K.shape[0].to_py_value()
        k_vector_size = 128 // K.dtype.width
        Q = fxh.view_as_torch_tensor(Q, (num_query_tokens, num_qo_heads, head_dim_qk))
        K = fxh.view_as_torch_tensor(K, (num_physical_pages, num_kv_heads, head_dim_qk//k_vector_size, num_BN_per_page, BN, k_vector_size))
        K = fx.select(K, (0, 3, 1, 2, 4, 5))
        V = fxh.view_as_torch_tensor(V, (num_physical_pages, num_kv_heads, num_BN_per_page, BN//k_vector_size, head_dim_v, k_vector_size))
        V = fx.select(V, (0, 2, 1, 3, 4, 5))

        q_descale = fxh.view_as_torch_tensor(q_descale, (num_query_tokens, num_qo_heads, 1))
        k_descale = fxh.view_as_torch_tensor(k_descale, (1,))
        v_descale = fxh.view_as_torch_tensor(v_descale, (1,))
        out = fxh.view_as_torch_tensor(out, (num_query_tokens, num_qo_heads, head_dim_v))
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
            kv_indptr,
            kv_page_indices,
            q_descale,
            k_descale,
            v_descale,
            kv_last_page_lens,
            out,
            work_counter,
            value_attrs=value_attrs,
        ).launch(grid=(num_workgroups, 1, 1), block=(num_threads, 1, 1), stream=stream)

    def callable(
        Q: torch.Tensor,  # [num_query_tokens, num_qo_heads, head_dim]
        K: torch.Tensor,  # [num_physical_pages, num_kv_heads, (head_dim // k_vector_size, page_size, k_vector_size)]
        V: torch.Tensor,  # [num_physical_pages, num_kv_heads, (page_size // k_vector_size, head_dim, k_vector_size)]
        cu_seqlens_q: torch.Tensor,  # [batch_size + 1] cu_seqlens_q[i] ~ cu_seqlens_q[i+1] is the range of query tokens in batch i
        kv_indptr: torch.Tensor,  # [batch_size + 1]    kv_indptr[i] ~ kv_indptr[i+1] is the range of virtual page ids in batch i
        kv_page_indices: torch.Tensor,  # [num_pages] kv_page_indices[i] is the physical page id of virtual page i (used to index into K and V)
        max_seqlen_q: int,  # a hint for scheduler
        max_seqlen_k: int,  # a hint for scheduler
        causal: bool,
        q_descale: torch.Tensor,  # per-token/per-tensor descaling factor for Q, shape [num_query_tokens, num_qo_heads, 1]
        k_descale: torch.Tensor,  # per-tensor descaling factor for K, shape [1]  (per-layer scalar, not per-head nor per-sequence)
        v_descale: torch.Tensor,  # per-tensor descaling factor for V, shape [1]  (per-layer scalar, not per-head nor per-sequence)
        kv_last_page_lens: torch.Tensor,  # [batch_size] kv_last_page_lens[i] is the number of valid tokens in the last page of batch i, used to mask out invalid tokens in the last page
        out: torch.Tensor,  # [num_query_tokens, num_qo_heads, head_dim]
        stream=None,
    ):
        stream = torch.cuda.current_stream() if stream is None else stream

        assert causal == is_causal
        assert not causal or max_seqlen_k >= max_seqlen_q, (
            "bottom-right causal attention requires max_seqlen_k >= max_seqlen_q"
        )
        assert k_descale.numel() == 1
        assert v_descale.numel() == 1
        num_query_tokens, _num_qo_heads, _head_dim = Q.shape
        assert _num_qo_heads == num_qo_heads
        assert _head_dim == head_dim_qk
        num_physical_pages, _num_kv_heads, _head_dim_grps, _page_size, k_vector_size = K.shape
        assert _num_kv_heads == num_kv_heads
        assert _head_dim_grps * k_vector_size == head_dim_qk
        assert _page_size == page_size
        assert V.shape == (
            num_physical_pages,
            num_kv_heads,
            _page_size // k_vector_size,
            head_dim_v,
            k_vector_size,
        )
        batch_size = cu_seqlens_q.shape[0] - 1
        assert kv_indptr.shape[0] == batch_size + 1
        assert kv_last_page_lens.shape[0] == batch_size
        # some internal logic use i32 address
        assert K.numel()*K.element_size() <= 2**31 - 1, f"KV cache size ={K.numel()*K.element_size()} > 2**31 - 1"

        if 0:
            # reference implementation using torch.nn.functional.scaled_dot_product_attention
            # de-vectorize & de-quantize K and V
            q_ref = Q.float() * q_descale
            k_cache_ref = (
                K.permute(0, 3, 1, 2, 4).reshape(num_physical_pages, page_size, num_kv_heads, head_dim_qk).float()
                * k_descale
            )
            v_cache_ref = (
                V.permute(0, 2, 4, 1, 3).reshape(num_physical_pages, page_size, num_kv_heads, head_dim_v).float()
                * v_descale
            )
            # reference
            for batch_idx in range(batch_size):
                page0 = kv_indptr[batch_idx]
                page1 = kv_indptr[batch_idx + 1]
                query0 = cu_seqlens_q[batch_idx]
                query1 = cu_seqlens_q[batch_idx + 1]
                pages = kv_page_indices[page0:page1].long().to("cuda")
                kv_len = (page1 - page0 - 1) * page_size + kv_last_page_lens[batch_idx]
                #print(batch_idx, kv_last_page_lens[batch_idx], kv_len)
                k_ref = k_cache_ref[pages].view(-1, num_kv_heads, head_dim_qk)[:kv_len].float()
                v_ref = v_cache_ref[pages].view(-1, num_kv_heads, head_dim_v)[:kv_len].float()
                k_ref = k_ref.repeat_interleave(num_qo_heads // num_kv_heads, dim=1)
                v_ref = v_ref.repeat_interleave(num_qo_heads // num_kv_heads, dim=1)
                # rows = torch.arange(qo_len, device="cuda").unsqueeze(1)
                # cols = torch.arange(kv_len, device="cuda").unsqueeze(0)
                # causal_mask = cols <= (kv_len - qo_len + rows) if causal else None
                out[query0:query1] = (
                    torch.nn.functional.scaled_dot_product_attention(
                        q_ref[query0:query1].transpose(0, 1).unsqueeze(0),
                        k_ref.transpose(0, 1).unsqueeze(0),
                        v_ref.transpose(0, 1).unsqueeze(0),
                        # attn_mask=causal_mask,
                        is_causal=causal,
                    )
                    .squeeze(0)
                    .transpose(0, 1)
                )
            return out

        multi_processor_count = torch.cuda.get_device_properties().multi_processor_count
        with torch.cuda.stream(stream):
            # slot 0: global ticket counter; slots 1..N: per-workgroup mailbox
            work_counter = torch.zeros(
                multi_processor_count + 1, device="cuda", dtype=torch.int32
            )
            work_counter[0] = multi_processor_count

        cf = getattr(launch, "_cf", None)
        if cf is None:
            cf = flyc.compile(
                launch,
                Q,
                K,
                V,
                cu_seqlens_q,
                kv_indptr,
                kv_page_indices,
                q_descale,
                k_descale,
                v_descale,
                kv_last_page_lens,
                out,
                work_counter,
                multi_processor_count,
                stream,
            )
            launch._cf = cf
        else:
            cf(
                Q,
                K,
                V,
                cu_seqlens_q,
                kv_indptr,
                kv_page_indices,
                q_descale,
                k_descale,
                v_descale,
                kv_last_page_lens,
                out,
                work_counter,
                multi_processor_count,
                stream,
            )

    return callable
