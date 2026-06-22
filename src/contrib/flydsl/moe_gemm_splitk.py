import argparse
import os
import sys

import torch
from functools import cache

import flydsl.compiler as flyc  # noqa: E402
from flydsl.compiler.kernel_function import CompilationContext  # noqa: E402
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl, vector, arith
from flydsl.expr.typing import BFloat16, Float32, T, Float8E4M3FNUZ
from flydsl.expr.typing import Vector as Vec
from flydsl.runtime.device import get_rocm_arch  # noqa: E402
from flydsl.utils.env import DebugEnvManager
from flydsl._mlir import ir
import flydsl
from flydsl._mlir.dialects import llvm
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr.utils.arith import _to_raw as _raw

# debug
if 1:
    DebugEnvManager.enable_debug_info = True
    ir._globals.register_traceback_file_inclusion(__file__)
    ir._globals.register_traceback_file_exclusion(os.path.dirname(flydsl.__file__))
    ir._globals.set_loc_tracebacks_frame_limit(40)
    ir._globals.set_loc_tracebacks_enabled(True)
    os.environ.setdefault("FLYDSL_RUNTIME_ENABLE_CACHE", "0")

def div_up(x, y):
    return (x + y - 1) // y

def compile_gemm(N, K, weight_dtype, TOPK, BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N, stage='gateup', alg='splitk'):

    TILE_K = 64
    assert BLOCK_TILE_SIZE_M <= 256, "BLOCK_SIZE_M must be less than or equal to 256 due to LDS size limit for sorted ids."
    if alg == 'splitk':
        assert BLOCK_TILE_SIZE_N % 64 == 0, "For split-k, BLOCK_TILE_SIZE_N needs to be multiple of 64 due to reduce layout."
        assert K % (32 * 4) == 0, "K must be a multiple of 128 for split-k algorithm."
        c_reduce_lds_size = 16 * 64 * 4 # save LDS size instead of BLOCK_TILE_SIZE_M * BLOCK_TILE_SIZE_N * 4
        @fx.union
        class SharedStorage:
            sorted_lds: fx.Array[fx.Int32, 256, 16]
            c_reduce_lds: fx.Array[fx.Float32, c_reduce_lds_size, 16]

    def _encode_waitcnt(vmcnt=63, expcnt=7, lgkmcnt=63):
        """Encode s_waitcnt bitfield for CDNA3 (gfx94x)."""
        vm_lo = vmcnt & 0xF
        vm_hi = (vmcnt >> 4) & 0x3
        return vm_lo | (expcnt << 4) | (lgkmcnt << 8) | (vm_hi << 14)

    class TensorWithIndex:
        # view: real tensor
        # tile_m, tile_k: tile size in M/K dimension for each copy from global to shared
        # tile_m_in_copy, tile_k_in_copy: tile size in M/K dimension for tiled_copy(due to m, k values could be extracted from tiled_copy)
        # tiled_copy_index: thread mapping for index
        # tiled_copy: thread mapping for copy
        # tid: thread id for copy
        # lds_index: index tensor in LDS buffer which contains m index of view
        def __init__(self, view, tile_m, tile_k, tile_m_in_copy, tile_k_in_copy, tiled_copy_index:fx.TiledCopy, tiled_copy:fx.TiledCopy, tid, lds_index, is_read_from_mem=True):
            self.view = view
            self.tile_m = tile_m
            self.tile_k = tile_k
            self.tile_m_in_copy = tile_m_in_copy
            self.tile_k_in_copy = tile_k_in_copy
            self.tid = tid
            self.tile_copy_mn = tiled_copy.tile_mn
            # split (tile_m, tile_k) by tile_copy_mn, shape: [(tile_m_in_copy, tile_k_in_copy), (rep_m, rep_k)]
            tile_layout = fx.zipped_divide(fx.make_layout((tile_m, tile_k), (1, tile_m)), self.tile_copy_mn)
            _, rep_shape  = tile_layout.shape
            _, rep_stride = tile_layout.stride
            self.rep_layout = fx.make_layout(rep_shape, rep_stride)
            self.tiled_copy_tv_layout = tiled_copy.layout_src_tv_tiled if is_read_from_mem else tiled_copy.layout_dst_tv_tiled
            self.is_read_from_mem = is_read_from_mem

            # split into (1, tile_k) blocks
            rank = fx.get_shape(self.view).rank
            dims = [1] * (rank - 1)
            # shape: [(1, tile_k), (m, rep_k)]
            self.tensor_blocks_in_k = fx.zipped_divide(view, fx.make_tile(*dims, tile_k))

            # read index
            lds = fx.make_view(lds_index, fx.make_layout(BLOCK_TILE_SIZE_M, 1))
            cp_atom_lds = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Int32)
            lds_thr = tiled_copy_index.get_slice(tid).partition_S(lds)
            self.index_frag = fx.make_fragment_like(lds_thr)
            fx.copy(cp_atom_lds, lds_thr, self.index_frag)

            offset_thread = fx.get_scalar(fx.crd2idx((self.tid, fx.Int32(0)), self.tiled_copy_tv_layout))
            self.offset_thread_k = offset_thread // tile_m_in_copy

        def copy(self, copy_atom, k_idx, frag:fx.Tensor):
            shape = fx.get_shape(self.rep_layout)
            rep_m = fx.size(shape[0]).to_py_value()
            rep_k = fx.size(shape[1]).to_py_value()
            m_size_in_copy = self.tile_m_in_copy
            value_size = fx.get_shape(frag)[0][0].to_py_value()
            # is there repeat in value dimension for frag?
            assert fx.get_shape(frag)[0][1].to_py_value() == 1, "rep for value dimension should be 1"

            rank = fx.get_shape(self.view).rank
            block_cord = [None] * (rank - 1) + [k_idx]
            # current iter block (M dimension is not indexed), shape: [(1, tile_k), m]
            tensor_block = self.tensor_blocks_in_k[None, (*block_cord,)]
            for m in range_constexpr(rep_m):
                # current iter subblock with correct M index, shape: [(1, tile_k)]
                if const_expr(rank == 2):
                    tensor_sub_block = tensor_block[None, self.index_frag[0, m] & 0xffffff]
                else:
                    tensor_sub_block = tensor_block[None, self.index_frag[0, m] & 0xffffff, (self.index_frag[0, m] >> 24)]
                for k in range_constexpr(rep_k):
                    # get block k index
                    offset_block = fx.crd2idx((m, k), self.rep_layout).to_py_value()
                    offset_block_k = offset_block // self.tile_m
                    # NOTE: assum K is linear in memory
                    offset_k_in_tile = offset_block_k + self.offset_thread_k
                    reg = frag[None, m, k]
                    mem = fx.make_view(fx.add_offset(fx.get_iter(tensor_sub_block), offset_k_in_tile), fx.make_layout(value_size, 1))
                    if const_expr(self.is_read_from_mem):
                        fx.copy_atom_call(copy_atom, mem, reg)
                    else:
                        fx.copy_atom_call(copy_atom, reg, mem)

    def _gemm_splitk(TILE_M, TILE_N, TILE_K,
                     blk_n: int,                        # block index for N dimension
                     arg_p_input:fx.Tensor,             # [M, K]
                     arg_p_weight:fx.Tensor,            # [(16,N/16), (8, K/8)]
                     lds,
                     ):
        tid = gpu.thread_idx.x
        a_tensor = fx.rocdl.make_buffer_tensor(arg_p_input, max_size=False)
        b_tensor = fx.rocdl.make_buffer_tensor(arg_p_weight, max_size=False)
        # shape: [n_in_tile, k_in_tile, k_tile]
        b_tile  = fx.flat_divide(b_tensor, fx.make_tile(TILE_N, TILE_K * 4))[None, None, blk_n, None]
        cp_atom_r = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16)

        tiled_mma = fx.make_tiled_mma(
            fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, fx.BFloat16)),
            # splitk, 4 waves in K dimension
            fx.make_layout((1, 1, 4), (0, 0, 1)),
            fx.make_tile(None, None, fx.make_layout((4, 4 * 4, 2), (1, 8, 4)))
        )
        b_tiled_thr = fx.make_tiled_copy_B(cp_atom_r, tiled_mma).get_slice(tid)
        cp_atom_lds = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Int32)
        tiled_copy_sortid_lds = fx.make_tiled_copy(cp_atom_lds, fx.make_layout(((16, 16), 1), ((1, 0), 0)), fx.make_tile(16))
        a_tensor_thr = TensorWithIndex(a_tensor, TILE_M, TILE_K * 4, 16, 32, tiled_copy_sortid_lds, fx.make_tiled_copy_A(cp_atom_r, tiled_mma), tid, lds.sorted_lds)

        b_tensor_thr = b_tiled_thr.partition_S(b_tile)
        b_frag = fx.make_rmem_tensor(
            fx.make_layout((4, TILE_N // 16, (2, TILE_K // 32)), (1, TILE_K // 32 * 8, (4, 8))), fx.BFloat16)
        a_frag = fx.make_rmem_tensor(
            fx.make_layout((4, TILE_M // 16, (2, TILE_K // 32)), (1, TILE_K // 32 * 8, (4, 8))), fx.BFloat16)
        c_frag = fx.make_rmem_tensor(
            # shape(c=b*a): [value, n_req, m_req]
            fx.make_layout(((4, 1), TILE_N // 16, TILE_M // 16), ((1, 0), TILE_M // 16 * 4, 4)), fx.Float32)
        c_frag.fill(0)

        a_frag_retile = fx.make_tiled_copy_A(cp_atom_r, tiled_mma).get_slice(tid).retile(a_frag)
        b_frag_retile = b_tiled_thr.retile(b_frag)
        acc_init = c_frag.load()
        for k, state in range(0, K // TILE_K // 4, 1, init=[acc_init]):
            c_frag.store(state[0])
            k_i32 = fx.Int32(k)
            a_tensor_thr.copy(cp_atom_r, k_i32, a_frag_retile)
            fx.copy(cp_atom_r, b_tensor_thr[None, None, None, k_i32], b_frag_retile)
            fx.gemm(tiled_mma, c_frag, b_frag, a_frag, c_frag)

            results = yield [c_frag.load()]
        c_frag.store(results)
        # [v, n, m] -> [v, m, n]
        c_frag = fx.select(c_frag, [0, 2, 1])

        # Reduce across 4 waves. To save lds size, will reuse (16*4)x64 floats for one loop
        swz = fx.SwizzleType.get(3, 3, 3)
        c_lds = fx.make_view(lds.c_reduce_lds,
                             fx.make_composed_layout(fx.static(swz), fx.make_ordered_layout((16 * 4, 64), order=(1, 0))))
        cp_atom_lds_w = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Float32)
        c_tiled_lds_w = fx.make_tiled_copy(cp_atom_lds_w,
                                         # (4wave*16)*4
                                         fx.make_layout(((16, 4, 4), (4, 4)), ((1, 256, 16), (64, 1024))),
                                         fx.make_tile(16 * 4, 16 * 4))
        c_tensor_thr_lds_w = c_tiled_lds_w.get_slice(tid).partition_D(c_lds)

        if const_expr(TILE_N == 64):
            cp_atom_lds_r = fx.make_copy_atom(fx.UniversalCopy64b(), fx.Float32)
            c_tiled_lds_r = fx.make_tiled_copy(cp_atom_lds_r,
                                            # thread mapping: (4wavex4)x16, repeat 4 times in m dimension for reduce
                                            fx.make_layout(((16, 4, 4), (2, 4)), ((64 * 2, 1, 4), (64, 16))),
                                            fx.make_tile(16 * 4, 16 * 2))
            tile_sub_n = 32
        else:
            cp_atom_lds_r = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Float32)
            c_tiled_lds_r = fx.make_tiled_copy(cp_atom_lds_r,
                                            # thread mapping: (4wavex4)x16, repeat 4 times in m dimension for reduce
                                            fx.make_layout(((16, 4, 4), (4, 4)), ((256, 1, 4), (64, 16))),
                                            fx.make_tile(16 * 4, 16 * 4))
            tile_sub_n = 64
        c_tensor_thr_lds_r = c_tiled_lds_r.get_slice(tid).partition_S(c_lds)

        # shape: [(4, 1), rep_m, rep_n]
        c_frag_vec = c_frag.load()
        # shape: [v, rm, rn]
        shape_v = fx.size(fx.get_shape(c_tensor_thr_lds_r)[0][0]).to_py_value()
        stride_v = 1
        stride_sub_rn = shape_v * stride_v
        stride_rn = stride_sub_rn * (64 // tile_sub_n)
        stride_rm = stride_rn * TILE_N // tile_sub_n
        c_frag_reduce = fx.make_rmem_tensor(
            fx.make_layout((shape_v, TILE_M // (4 * 4), (64 // tile_sub_n, TILE_N // 64)),
                           (stride_v, stride_rm, (stride_sub_rn, stride_rn))), fx.Float32)
        for m in range_constexpr(TILE_M // 16):
            for n in range_constexpr(TILE_N // 64):
                items = []
                for i in range_constexpr(16):
                    idx = fx.get_scalar(fx.crd2idx((i % 4, m, n * 4 + i // 4), c_frag.layout))
                    items.append(c_frag_vec[idx])
                sub_c_frag = fx.make_fragment_like(c_tensor_thr_lds_w)
                sub_c_frag.store(Vec.from_elements(items, fx.Float32))
                fx.copy(cp_atom_lds_w, sub_c_frag, c_tensor_thr_lds_w)
                gpu.barrier()

                sub_c_frag_reduce = fx.make_fragment_like(c_tensor_thr_lds_r)
                fx.copy(cp_atom_lds_r, c_tensor_thr_lds_r, sub_c_frag_reduce)
                acc = sub_c_frag_reduce[(None, 0), None, None].load()
                for i in range_constexpr(1, 4):
                    acc += sub_c_frag_reduce[(None, i), None, None].load()

                c_frag_reduce[None, m, (None, n)].store(acc)

        return c_frag_reduce

    gemm_splitk = ASTRewriter.transform(_gemm_splitk)

    @flyc.kernel
    def moe_2stage_gateup(p_input: fx.Tensor,            # bf16 [M, K]
                          p_weight: fx.Tensor,           # bf16 [N/16, K/8 * 16 * 8,]
                          p_output: fx.Tensor,           # bf16 [M, N]
                          # sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids
                          p_sorted_ids: fx.Tensor,
                          p_sorted_weights: fx.Tensor,
                          p_sorted_expert_ids: fx.Tensor,
                          p_num_valid_ids: fx.Tensor,
                          p_w_scale: fx.Tensor,
                          M: int,
                          ):
        tid = gpu.thread_idx.x
        blk_n = gpu.block_idx.x
        e_idx = gpu.block_idx.y

        arg_p_input = fx.Tensor(fx.make_view(fx.get_iter(p_input), fx.make_layout((M, K), (K, 1))))
        num_valid_buf = fx.Tensor(fx.make_view(fx.recast_iter(fx.Int32, fx.get_iter(p_num_valid_ids)), fx.make_layout(1, 1)))
        max_valid_id = num_valid_buf[0]
        if e_idx * BLOCK_TILE_SIZE_M < max_valid_id:
            lds = fx.SharedAllocator().allocate(SharedStorage)
            lds.sorted_lds = lds.sorted_lds.peek()
            lds.c_reduce_lds = lds.c_reduce_lds.peek()
            arg_p_sorted_ids = fx.Tensor(fx.make_view(fx.recast_iter(fx.Int32, fx.add_offset(fx.get_iter(p_sorted_ids), e_idx * BLOCK_TILE_SIZE_M)), fx.make_layout(BLOCK_TILE_SIZE_M, 1)))
            arg_p_sorted_expert_ids = fx.recast_iter(fx.Int32, fx.get_iter(p_sorted_expert_ids))
            expert_id = arg_p_sorted_expert_ids[e_idx]
            # there is a reduce in gemm_splitk which will read/write from lds, the BLOCK_TILE_SIZE_N will impact the coalesced access:
            # BLOCK_TILE_SIZE_N BLOCK_TILE_SIZE_N//2(after silu) LDS_read_per_lane  MEM_write_per_lane
            # 64                32                               2=(32/16 threads)  2=(32/16 threads)
            # 128               64                               4=(64/16 threads)  4=(64/16 threads)
            # 256: will split into 2x128
            if const_expr(alg == 'splitk'):
                if const_expr(BLOCK_TILE_SIZE_N % 128 == 0):
                    group_layout_silu = fx.make_layout(((64, 2, N // 128), K), ((1, N // 2, 64), N))
                else:
                    group_layout_silu = fx.make_layout(((32, 2, N // 64), K), ((1, N // 2, 32), N))
            else:
                # at least 32(=64/2) contiguous rows in N dimension for silu
                group_layout_silu = fx.make_layout(((32, 2, N // 64), K), ((1, N // 2, 32), N))
            arg_p_weight = fx.Tensor(fx.make_view(fx.add_offset(fx.get_iter(p_weight), fx.Int64(expert_id * N * K)),
                                                  fx.composition(fx.make_layout(((16, N // 16), (8, K // 8)), ((8, 16 * K), (1, 128))),   # preshuffle layout: [16, (8, K // 8)]
                                                                 group_layout_silu)))                                                     # NOTE: assume permuted adjacent 32 rows will fall in the same wave to do silu
            
            # sorted ids: global -> LDS (scalar load/store, only first BLOCK_TILE_SIZE_M threads participate)
            sorted_ids_buf = fx.rocdl.make_buffer_tensor(arg_p_sorted_ids, max_size=False)
            if tid < BLOCK_TILE_SIZE_M:
                lds_view = fx.make_view(lds.sorted_lds, fx.make_layout(BLOCK_TILE_SIZE_M, 1))
                # fx.memref_store(val, lds_view, tid)
                lds_view[tid] = sorted_ids_buf[tid]
            gpu.barrier()

            # prepare c_tensor(reuse lds.c_reduce_lds before gemm)
            if const_expr(alg == 'splitk'):
                if const_expr(BLOCK_TILE_SIZE_N % 128 == 0):
                    # due to silu, actually the N block is only 64 points=(16x4)64b
                    cp_atom_w = fx.make_copy_atom(fx.rocdl.BufferCopy64b(), fx.BFloat16)
                    c_tiled_g = fx.make_tiled_copy(cp_atom_w,
                                                # 线程划分: 4x16，4 wave切分m方向16行，每个线程写入n方向连续4个点
                                                fx.make_layout(((16, 4, 4), 4), ((64, 1, 4), 16)),
                                                fx.make_tile(16, 16 * 4))
                    tile_m_copy = 16
                    tile_n_copy = 64
                else:
                    # due to silu, actually the N block is only 32 points=(16x2)32b
                    cp_atom_w = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.BFloat16)
                    c_tiled_g = fx.make_tiled_copy(cp_atom_w,
                                                # 线程划分: 4x16，4 wave切分m方向16行，每个线程写入n方向连续2个点
                                                fx.make_layout(((16, 4, 4), 2), ((32, 1, 4), 16)),
                                                fx.make_tile(16, 16 * 2))
                    tile_m_copy = 16
                    tile_n_copy = 32

            arg_p_output = fx.Tensor(fx.make_view(fx.get_iter(p_output), fx.make_layout((M, TOPK, N // 2), (TOPK * N // 2, N // 2, 1))))
            out_tensor = fx.rocdl.make_buffer_tensor(arg_p_output, max_size=False, num_records_bytes=M * TOPK * N // 2 * fx.BFloat16.width // 8)
            tiled_copy_sortid_lds = fx.make_tiled_copy(fx.make_copy_atom(fx.UniversalCopy32b(), fx.Int32),
                                                       fx.make_layout(((16, 16), 1), ((0, 1), 0)),
                                                       fx.make_tile(16))
            c_tensor = TensorWithIndex(out_tensor, BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N // 2, tile_m_copy, tile_n_copy, tiled_copy_sortid_lds, c_tiled_g, tid, lds.sorted_lds, is_read_from_mem=False)

            c_frag = gemm_splitk(BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N, TILE_K, blk_n, arg_p_input, arg_p_weight, lds)

            # silu: gate/up are interleaved in the N dimension
            # c_frag has shape (value, M_reps, N_reps) after reduce
            v_reps = fx.size(fx.get_shape(c_frag)[0]).to_py_value()
            m_reps = fx.size(fx.get_shape(c_frag)[1]).to_py_value()
            n_reps = fx.size(fx.get_shape(c_frag)[2]).to_py_value()

            # c_frag_bf16 stores the silu result (half the N dimension since gate+up → 1 output)
            c_frag_bf16 = fx.make_rmem_tensor(
                fx.make_layout(((v_reps, 1), m_reps, n_reps // 2), ((1, 0), n_reps // 2 * v_reps, v_reps)), fx.BFloat16)

            log2_exp1 = -1.4426950408889634
            for i in range_constexpr(n_reps // 2):
                gate = c_frag[None, None, 2 * i + 0].load()
                up   = c_frag[None, None, 2 * i + 1].load()
                gate_log2 = gate * log2_exp1
                acc = []
                for j in range_constexpr(gate.numel):
                    tmp = rocdl.exp2(T.f32, _raw(gate_log2[j]))
                    acc.append((gate[j] * rocdl.rcp(T.f32, 1.0 + tmp)) * up[j])
                acc = vector.from_elements(T.vec(gate.numel, fx.Float32.ir_type), acc)
                # failed: acc = (gate * (1.0 / (1.0 + flydsl.expr.math.exp2(gate * log2_exp1, fastmath=flydsl.expr.arith.FastMathFlags.fast)))) * up
                round_bit = fx.Uint32(0x8000).ir_value().bitcast(fx.Float32.ir_type)
                # c_frag_bf16[None, None, i].store(acc.to(fx.BFloat16))
                acc = ((acc + round_bit).bitcast(fx.Uint32) >> 16).to(fx.Uint16).bitcast(fx.BFloat16)
                c_frag_bf16[None, None, i].store(acc)

            c_tensor.copy(cp_atom_w, blk_n, c_tiled_g.get_slice(tid).retile(c_frag_bf16))

    @flyc.jit
    def launch(
        p_input: fx.Tensor,
        p_weight: fx.Tensor,
        p_output: fx.Tensor,
        p_sorted_ids: fx.Tensor,
        p_sorted_weights: fx.Tensor,
        p_sorted_expert_ids: fx.Tensor,
        p_num_valid_ids: fx.Tensor,
        p_w_scale: fx.Tensor,
        M: int,
        stream: fx.Stream,
    ):
        CompilationContext.get_current()
        num_n_blocks = div_up(N, BLOCK_TILE_SIZE_N)
        grid = p_sorted_expert_ids.shape.to_py_value()
        B = p_input.shape[0].to_py_value()
        E = p_weight.shape[0].to_py_value()
        if B * TOPK <= E:
            grid = B * TOPK
        moe_2stage_gateup(
            p_input, p_weight, p_output,
            p_sorted_ids, p_sorted_weights, p_sorted_expert_ids,
            p_num_valid_ids, p_w_scale, M,
        ).launch(
            grid=(num_n_blocks, grid, 1),
            block=(256, 1, 1),
            stream=stream,
        )

    return launch
