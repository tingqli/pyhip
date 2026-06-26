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

def compile_gemm(N, K, weight_dtype, quant_type, TOPK, BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N, stage='gateup', alg='splitk'):

    TILE_K = 64
    assert BLOCK_TILE_SIZE_M <= 256, "BLOCK_SIZE_M must be less than or equal to 256 due to LDS size limit for sorted ids."
    assert weight_dtype in ['bf16', 'fp8'], "weight_dtype must be either 'bf16' or 'fp8'"
    assert quant_type in ['no', 'ptpc', 'per_tensor'], "quant_type must be either 'no', 'ptpc' or 'per_tensor'"

    if stage == 'gateup' and alg == 'splitk':
        assert BLOCK_TILE_SIZE_N % 64 == 0, "For split-k, BLOCK_TILE_SIZE_N needs to be multiple of 64 due to reduce layout."
        assert K % (32 * 4) == 0, "K must be a multiple of 128 for split-k algorithm."
        c_reduce_lds_size = 16 * 64 * 4 # save LDS size instead of BLOCK_TILE_SIZE_M * BLOCK_TILE_SIZE_N * 4
        @fx.union
        class SharedStorage:
            sorted_lds: fx.Array[fx.Int32, 256, 16]
            c_reduce_lds: fx.Array[fx.Float32, c_reduce_lds_size, 16]
    elif stage == 'down' and alg == 'splitk':
        @fx.struct
        class SharedStorage:
            sorted_lds: fx.Array[fx.Int32, 256, 16]
    elif stage == 'gateup' and alg == 'batch1':
        c_reduce_lds_size = 16 * 64 * 4 # save LDS size instead of BLOCK_TILE_SIZE_M * BLOCK_TILE_SIZE_N * 4
        @fx.struct
        class SharedStorage:
            c_reduce_lds: fx.Array[fx.Float32, c_reduce_lds_size, 16]

    if weight_dtype == 'bf16':
        weight_dtype = fx.BFloat16
    elif weight_dtype == 'fp8':
        weight_dtype = fx.Float8E4M3FNUZ

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
        def __init__(self, view, tile_m, tile_k, tiled_copy_index:fx.TiledCopy, tiled_copy:fx.TiledCopy, tid, lds_index, is_read_from_mem=True, TOPK=None, is_atomic_write=False):
            assert not (is_atomic_write and is_read_from_mem)
            self.view = view
            self.tile_m = tile_m
            self.tile_k = tile_k
            self.is_read_from_mem = is_read_from_mem
            self.TOPK = TOPK
            self.is_atomic_write = is_atomic_write

            # split into (1, tile_k) blocks
            rank = fx.get_shape(self.view).rank
            dims = [1] * (rank - 1)
            # shape: [(1, tile_k), (m, rep_k)]
            self.tensor_blocks_in_k = fx.zipped_divide(view, fx.make_tile(*dims, tile_k))

            # read index
            lds = fx.make_view(lds_index.ptr, fx.make_layout(BLOCK_TILE_SIZE_M, 1))
            cp_atom_lds = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Int32)
            lds_thr = tiled_copy_index.get_slice(tid).partition_S(lds)
            self.index_frag = fx.make_fragment_like(lds_thr)
            fx.copy(cp_atom_lds, lds_thr, self.index_frag)

            dtype = fx.PointerType.get(fx.Int8.ir_type, 1, 512)
            ptr = fx.inttoptr(dtype, fx.Int32(0))
            self.fake_tensor = fx.make_view(ptr, fx.make_layout((tile_m, tile_k), (1, tile_m)))
            self.fake_tensor_thr = tiled_copy.get_slice(tid).partition_S(self.fake_tensor) if is_read_from_mem else tiled_copy.get_slice(tid).partition_D(self.fake_tensor)
            # since init ptr is zero, it will be the offset of the thread in the tile after partition_S
            offset_thread = fx.Int32(fx.ptrtoint(fx.get_iter(self.fake_tensor_thr)))
            self.offset_thread_k = offset_thread // tile_m

        def copy(self, copy_atom, k_idx, frag:fx.Tensor):
            layout = fx.get_layout(self.fake_tensor_thr)
            shape = fx.get_shape(self.fake_tensor_thr)
            rep_m = fx.size(shape[1]).to_py_value()
            rep_k = fx.size(shape[2]).to_py_value()
            value_size = fx.get_shape(frag)[0].to_py_value()
            stride_size = fx.get_stride(frag)[0].to_py_value()

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
                if const_expr(not self.is_atomic_write):
                    for k in range_constexpr(rep_k):
                        # get block k index
                        offset_block = fx.crd2idx((0, m, k), layout).to_py_value()
                        offset_block_k = offset_block // self.tile_m
                        # NOTE: assume K is linear in memory
                        offset_k_in_tile = offset_block_k + self.offset_thread_k
                        reg = frag[None, m, k]
                        mem = fx.make_view(fx.get_iter(tensor_sub_block) + offset_k_in_tile, fx.make_layout(value_size, stride_size))
                        if const_expr(self.is_read_from_mem):
                            fx.copy(copy_atom, mem, reg)
                        else:
                            fx.copy(copy_atom, reg, mem)
                else:
                    # fx.UniversalAtomic(fx.AtomicOp.Add) could not lower to `global_atomic_pk_add_bf16`, hack to emit
                    if (self.index_frag[0, m] >> 24) < TOPK:
                        for k in range_constexpr(rep_k):
                            # get block k index
                            offset_block = fx.crd2idx((0, m, k), layout).to_py_value()
                            offset_block_k = offset_block // self.tile_m
                            # NOTE: assume K is linear in memory
                            offset_k_in_tile = offset_block_k + self.offset_thread_k
                            reg = frag[None, m, k]
                            mem = fx.make_view(fx.get_iter(tensor_sub_block) + offset_k_in_tile, fx.make_layout(value_size, stride_size))
                            reg_vec = reg.load()
                            ptr_base = fx.get_iter(mem)
                            for i in range_constexpr(reg_vec.numel // 2):
                                pair = Vec.from_elements([reg_vec[i * 2], reg_vec[i * 2 + 1]], fx.BFloat16)
                                ptr = ptr_base + i * 2
                                addr = fx.ptrtoint(ptr)
                                llvm_ptr_ty = ir.Type.parse("!llvm.ptr<1>")
                                llvm_ptr = llvm.IntToPtrOp(llvm_ptr_ty, addr.ir_value())
                                llvm.AtomicRMWOp(llvm.AtomicBinOp.fadd, llvm_ptr, pair, llvm.AtomicOrdering.monotonic, syncscope="agent", alignment=4)

    TensorWithIndex.copy = ASTRewriter.transform(TensorWithIndex.copy)

    def select(tensor:fx.Tensor, order):
        rank = fx.get_shape(tensor).rank
        assert len(order) == rank
        stride = fx.get_stride(tensor)
        shape = fx.get_shape(tensor)
        new_layout = fx.make_layout([shape[i] for i in order], [stride[i] for i in order])
        return fx.make_view(fx.get_iter(tensor), new_layout)
    
    def cvt_fp8_bf16(src_tensor:fx.Tensor, dst_tensor:fx.Tensor):
        size = fx.size(fx.get_shape(src_tensor)).to_py_value()

        items = []
        src_vec = src_tensor.load().bitcast(fx.Uint32)
        for i in range_constexpr(size // 4):
            src_val = src_vec[i]
            pk0_f32 = llvm.inline_asm(T.f32x2, 
                                    [src_val.ir_value()], "v_cvt_pk_f32_fp8 $0, $1", "=v,v", has_side_effects=False)
            pk1_f32 = llvm.inline_asm(T.f32x2,
                                    [src_val.ir_value()], "v_cvt_pk_f32_fp8_sdwa $0, $1 src0_sel:WORD_1", "=v,v", has_side_effects=False)
            tmp = (pk0_f32.bitcast(fx.Uint32) >> 16).to(fx.Uint16).bitcast(fx.BFloat16)
            items.append(tmp[0])
            items.append(tmp[1])
            tmp = (pk1_f32.bitcast(fx.Uint32) >> 16).to(fx.Uint16).bitcast(fx.BFloat16)
            items.append(tmp[0])
            items.append(tmp[1])
        vec = Vec.from_elements(items, fx.BFloat16)
        layout = fx.get_layout(dst_tensor)
        for i in range_constexpr(size):
            crd = fx.idx2crd(i, layout)
            dst_tensor[crd] = vec[i]

    def _apply_scale_silu_bf16(c_frag, tid, expert_id, blk_n, contiguous_n, p_w_scale):
        v_reps = fx.size(fx.get_shape(c_frag)[0]).to_py_value()
        m_reps = fx.size(fx.get_shape(c_frag)[1]).to_py_value()
        n_reps = fx.size(fx.get_shape(c_frag)[2]).to_py_value()

        if const_expr(weight_dtype != fx.BFloat16):
            if const_expr(quant_type == 'ptpc'):
                group_layout_silu = fx.make_layout(((contiguous_n, 2, N // (2 * contiguous_n)), 1), ((1, N // 2, contiguous_n), 0))
                arg_p_scale = fx.make_view(fx.get_iter(p_w_scale) + expert_id * N,
                                            fx.composition(fx.make_layout(N, 1), group_layout_silu))
                scale_tile = fx.flat_divide(arg_p_scale, fx.make_tile(BLOCK_TILE_SIZE_N, 1))[None, None, blk_n, 0]
                cp_atom_scale = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32)
                tiled_copy_scale = fx.make_tiled_copy(cp_atom_scale,
                                                      fx.make_layout(((16, 4, 4), contiguous_n // 16), ((contiguous_n // 16, 0, 0), 1)),
                                                      fx.make_tile(contiguous_n, 1))
                scale_frag_tensor = tiled_copy_scale.get_slice(tid).partition_S(scale_tile)
                scale_frag = fx.make_fragment_like(scale_frag_tensor)
                fx.copy(cp_atom_scale, scale_frag_tensor, scale_frag)
                for n in range_constexpr(n_reps):
                    scale_vec = scale_frag[None, n, 0].load()
                    for m in range_constexpr(m_reps):
                        c_vec = c_frag[None, m, n].load()
                        vec = c_vec * scale_vec
                        c_frag[None, m, n].store(vec)
            elif const_expr(quant_type == 'per_tensor'):
                arg_p_scale = fx.make_view(fx.get_iter(p_w_scale) + expert_id, fx.make_layout(1, 1))
                scale = arg_p_scale[0]
                c_frag.store(c_frag.load() * scale)

        # c_frag_bf16 stores the silu result (half the N dimension since gate+up -> 1 output)
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
            acc = Vec.from_elements(acc, fx.Float32)
            round_bit = fx.Uint32(0x8000)
            acc = ((acc.bitcast(fx.Uint32) + round_bit) >> 16).to(fx.Uint16).bitcast(fx.BFloat16)
            c_frag_bf16[None, None, i].store(acc)

        return c_frag_bf16

    def _apply_down_scale(c_frag, tid, expert_id, blk_n, p_w_scale):
        if const_expr(weight_dtype != fx.BFloat16):
            if const_expr(quant_type == 'ptpc'):
                arg_p_scale = fx.make_view(fx.get_iter(p_w_scale) + expert_id * N, fx.make_layout(N, 1))
                scale_tile = fx.flat_divide(arg_p_scale, fx.make_tile(BLOCK_TILE_SIZE_N))[None, blk_n]
                cp_atom_scale = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Float32)
                tiled_copy_scale = fx.make_tiled_copy(cp_atom_scale,
                                                      fx.make_layout(((16, 4), 4), ((0, 4), 1)),
                                                      fx.make_tile(16))
                scale_frag_tensor = tiled_copy_scale.get_slice(tid).partition_S(scale_tile)
                scale_frag = fx.make_fragment_like(scale_frag_tensor)
                fx.copy(cp_atom_scale, scale_frag_tensor, scale_frag)
                m_reps = fx.size(fx.get_shape(c_frag)[1]).to_py_value()
                n_reps = fx.size(fx.get_shape(c_frag)[2]).to_py_value()
                for n in range_constexpr(n_reps):
                    scale_vec = scale_frag[None, n].load()
                    for m in range_constexpr(m_reps):
                        c_vec = c_frag[None, m, n].load()
                        vec = c_vec * scale_vec
                        c_frag[None, m, n].store(vec)
            elif const_expr(quant_type == 'per_tensor'):
                arg_p_scale = fx.make_view(fx.get_iter(p_w_scale) + expert_id, fx.make_layout(1, 1))
                scale = arg_p_scale[0]
                c_frag.store(c_frag.load() * scale)

    def _cvt_f32_to_bf16(c_frag):
        c_frag_bf16 = fx.make_fragment_like(c_frag, dtype=fx.BFloat16)
        round_bit = fx.Uint32(0x8000)
        c_frag_bf16.store(((c_frag.load().bitcast(fx.Uint32) + round_bit) >> 16).to(fx.Uint16).bitcast(fx.BFloat16))
        return c_frag_bf16

    def _gemm_splitk(TILE_M, TILE_N, TILE_K,
                     blk_n: int,                        # block index for N dimension
                     arg_p_input:fx.Tensor,             # [M, K] or [M, TOPK, K]
                     arg_p_weight:fx.Tensor,            # [(16,N/16), (8, K/8)]
                     lds,
                     splitk_waves=4,
                     a_with_index=True,
                     ):
        tid = gpu.thread_idx.x

        tile_k_per_wg = TILE_K * splitk_waves

        a_tensor = fx.rocdl.make_buffer_tensor(arg_p_input, max_size=False)
        b_tensor = fx.rocdl.make_buffer_tensor(arg_p_weight, max_size=False)
        # shape: [n_in_tile, k_in_tile, k_tile]
        b_tile = fx.flat_divide(b_tensor, fx.make_tile(TILE_N, tile_k_per_wg))[None, None, blk_n, None]
        a_cp_atom_r = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), arg_p_input.dtype)
        b_cp_atom_r = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), weight_dtype)

        # tiled copy is created based on the tiled_mma, so the tiled_mma should be same size for tiled copy
        rep_k_per_lane = 4 if const_expr(weight_dtype != fx.BFloat16) else 2
        k_perm = fx.make_tile(None, None, fx.make_layout((4, 4 * splitk_waves, rep_k_per_lane), (1, 4 * rep_k_per_lane, 4)))
        tiled_mma = fx.make_tiled_mma(
            fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, fx.BFloat16)),
            # splitk for gateup/down
            fx.make_layout((1, 1, splitk_waves), (0, 0, 1)),
            k_perm
        )
        if const_expr(a_with_index):
            cp_atom_lds = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Int32)
            tiled_copy_sortid_lds = fx.make_tiled_copy(cp_atom_lds, fx.make_layout(((16, 4 * splitk_waves), 1), ((1, 0), 0)), fx.make_tile(16))
            a_tensor_thr = TensorWithIndex(a_tensor, TILE_M, tile_k_per_wg, tiled_copy_sortid_lds, fx.make_tiled_copy_A(a_cp_atom_r, tiled_mma), tid, lds.sorted_lds)
            a_fake_tensor = fx.make_view(fx.get_iter(arg_p_input), fx.make_layout((TILE_M, tile_k_per_wg), (tile_k_per_wg, 1)))
            a_frag = tiled_mma.make_fragment_A(a_fake_tensor)
        else:
            a_tile = fx.flat_divide(a_tensor, fx.make_tile(TILE_M, tile_k_per_wg))[None, None, 0, None]
            a_tiled_thr = fx.make_tiled_copy_A(a_cp_atom_r, tiled_mma).get_slice(tid)
            a_tensor_thr = a_tiled_thr.partition_S(a_tile)
            a_frag = tiled_mma.make_fragment_A(a_tile[None, None, 0])

        a_frag_retile = fx.make_tiled_copy_A(a_cp_atom_r, tiled_mma).get_slice(tid).retile(a_frag)

        if const_expr(weight_dtype == fx.BFloat16):
            b_tiled_thr = fx.make_tiled_copy_B(b_cp_atom_r, tiled_mma).get_slice(tid)
            b_tensor_thr = b_tiled_thr.partition_S(b_tile)
            b_frag = tiled_mma.make_fragment_B(b_tile[None, None, 0])
            b_frag_retile = b_tiled_thr.retile(b_frag)
        else:
            # b_frag will be decompressed from fp8
            b_fake_tensor = fx.make_view(fx.get_iter(arg_p_input), fx.make_layout((TILE_N, tile_k_per_wg), (tile_k_per_wg, 1)))
            b_frag = tiled_mma.make_fragment_B(b_fake_tensor)

            tile_size = tiled_mma.tile_size_mnk
            tile_mn = fx.make_tile(fx.make_layout(fx.select(tile_size, [1]), 1), fx.make_layout(fx.select(tile_size, [2]), 1))
            b_tiled_thr = fx.make_tiled_copy(b_cp_atom_r, tiled_mma.tv_layout_B_tiled, tile_mn).get_slice(tid)
            b_tensor_thr = b_tiled_thr.partition_S(b_tile)
            b_frag_retile = fx.make_fragment_like(b_tensor_thr[None, None, None, 0], fx.Uint8)

        c_fake_tensor = fx.make_view(fx.get_iter(arg_p_input), fx.make_layout((TILE_N, TILE_M), (TILE_M, 1)))
        c_frag = tiled_mma.make_fragment_C(c_fake_tensor)
        c_frag.fill(0)
        acc_init = c_frag.load()

        for k, state in range(0, K // TILE_K // splitk_waves, 1, init=[acc_init]):
            c_frag.store(state[0])
            k_i32 = fx.Int32(k)
            if const_expr(a_with_index):
                a_tensor_thr.copy(a_cp_atom_r, k_i32, a_frag_retile)
            else:
                fx.copy(a_cp_atom_r, a_tensor_thr[None, None, None, k_i32], a_frag_retile)
            fx.copy(b_cp_atom_r, b_tensor_thr[None, None, None, k_i32], b_frag_retile)
            if const_expr(weight_dtype != fx.BFloat16):
                # decompress fp8 to bf16
                cvt_fp8_bf16(b_frag_retile, b_frag)
            fx.gemm(tiled_mma, c_frag, b_frag, a_frag, c_frag)

            results = yield [c_frag.load()]
        c_frag.store(results)
        # [v, n, m] -> [v, m, n]
        c_frag = select(c_frag, [0, 2, 1])

        if const_expr(splitk_waves == 1):
            return c_frag

        # Reduce across 4 waves. To save lds size, will reuse (16*4)x64 floats for one loop
        swz = fx.SwizzleType.get(3, 3, 3)
        c_lds = fx.make_view(lds.c_reduce_lds.ptr,
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
                gpu.barrier()

        return c_frag_reduce

    gemm_splitk = ASTRewriter.transform(_gemm_splitk)

    @flyc.kernel
    def moe_2stage_gateup(p_input: fx.Tensor,            # bf16 [M, K]
                          p_weight: fx.Tensor,           # quantized/bf16 [N/16, K/8 * 16 * 8]
                          p_output: fx.Tensor,           # bf16 [M, TOPK, N//2]
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

        arg_p_input = fx.make_view(fx.get_iter(p_input), fx.make_layout((M, K), (K, 1)))
        num_valid_buf = fx.make_view(fx.recast_iter(fx.Int32, fx.get_iter(p_num_valid_ids)), fx.make_layout(1, 1))
        max_valid_id = num_valid_buf[0]
        if const_expr(p_weight.dtype != fx.BFloat16):
            p_weight = fx.recast_iter(fx.Uint8, fx.get_iter(p_weight))
        else:
            p_weight = fx.get_iter(p_weight)
        if e_idx * BLOCK_TILE_SIZE_M < max_valid_id:
            lds = fx.SharedAllocator().allocate(SharedStorage)
            lds.sorted_lds = lds.sorted_lds.peek()
            lds.c_reduce_lds = lds.c_reduce_lds.peek()
            arg_p_sorted_ids = fx.make_view(fx.recast_iter(fx.Int32, fx.get_iter(p_sorted_ids) + e_idx * BLOCK_TILE_SIZE_M), fx.make_layout(BLOCK_TILE_SIZE_M, 1))
            arg_p_sorted_expert_ids = fx.recast_iter(fx.Int32, fx.get_iter(p_sorted_expert_ids))
            expert_id = arg_p_sorted_expert_ids[e_idx]
            # there is a reduce in gemm_splitk which will read/write from lds, the BLOCK_TILE_SIZE_N will impact the coalesced access:
            # BLOCK_TILE_SIZE_N BLOCK_TILE_SIZE_N//2(after silu) LDS_read_per_lane  MEM_write_per_lane
            # 64                32                               2=(32/16 threads)  2=(32/16 threads)
            # 128               64                               4=(64/16 threads)  4=(64/16 threads)
            # 256: will split into 2x128
            if const_expr(alg == 'splitk'):
                contiguous_n = 64 if const_expr(BLOCK_TILE_SIZE_N % 128 == 0) else 32
            else:
                contiguous_n = 32

            group_layout_silu = fx.make_layout(((contiguous_n, 2, N // (contiguous_n * 2)), K), ((1, N // 2, contiguous_n), N))
            element_num = 16 // (p_weight.dtype.width // 8)
            arg_p_weight = fx.make_view(p_weight + fx.Int64(expert_id * N * K),
                                        # preshuffle layout: [16, (8, K // 8)]
                                        fx.composition(fx.make_layout(((16, N // 16), (element_num, K // element_num)), ((element_num, 16 * K), (1, 16 * element_num))),
                                                       group_layout_silu))                                                     # NOTE: assume permuted adjacent 32 rows will fall in the same wave to do silu
            
            # sorted ids: global -> LDS (scalar load/store, only first BLOCK_TILE_SIZE_M threads participate)
            sorted_ids_buf = fx.rocdl.make_buffer_tensor(arg_p_sorted_ids, max_size=False)
            if tid < BLOCK_TILE_SIZE_M:
                lds_view = fx.make_view(lds.sorted_lds.ptr, fx.make_layout(BLOCK_TILE_SIZE_M, 1))
                # fx.memref_store(val, lds_view, tid)
                lds_view[tid] = sorted_ids_buf[tid]
            gpu.barrier()

            # prepare c_tensor(reuse lds.c_reduce_lds before gemm)
            if const_expr(alg == 'splitk'):
                cp_atom_w = fx.make_copy_atom(fx.rocdl.BufferCopy64b() if const_expr(BLOCK_TILE_SIZE_N % 128 == 0) else fx.rocdl.BufferCopy32b(), fx.BFloat16)
                c_tiled_g = fx.make_tiled_copy(cp_atom_w,
                                            # thread mapping: 4 wavex(4x16), (contiguous_n // 16) elements per lane
                                            fx.make_layout(((16, 4, 4), contiguous_n // 16), ((contiguous_n, 1, 4), 16)),
                                            fx.make_tile(16, contiguous_n))
            arg_p_output = fx.make_view(fx.get_iter(p_output), fx.make_layout((M, TOPK, N // 2), (TOPK * N // 2, N // 2, 1)))
            out_tensor = fx.rocdl.make_buffer_tensor(arg_p_output, max_size=False, num_records_bytes=M * TOPK * N // 2 * fx.BFloat16.width // 8)
            tiled_copy_sortid_lds = fx.make_tiled_copy(
                fx.make_copy_atom(fx.UniversalCopy32b(), fx.Int32),
                fx.make_layout(((16, 16), 1), ((0, 1), 0)),
                fx.make_tile(16),
            )
            c_tensor = TensorWithIndex(out_tensor, BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N // 2, tiled_copy_sortid_lds, c_tiled_g, tid, lds.sorted_lds, is_read_from_mem=False)

            c_frag = gemm_splitk(BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N, TILE_K, blk_n, arg_p_input, arg_p_weight, lds, splitk_waves=4)

            c_frag_bf16 = _apply_scale_silu_bf16(c_frag, tid, expert_id, blk_n, contiguous_n, p_w_scale)

            c_tensor.copy(cp_atom_w, blk_n, c_tiled_g.get_slice(tid).retile(c_frag_bf16))

    @flyc.kernel
    def moe_2stage_down(p_input: fx.Tensor,            # bf16 [M, TOPK, K]
                        p_weight: fx.Tensor,           # quantized/bf16 [N/16, K/8 * 16 * 8]
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

        arg_p_input = fx.make_view(fx.get_iter(p_input), fx.make_layout((M, TOPK, K), (TOPK * K, K, 1)))
        num_valid_buf = fx.make_view(fx.recast_iter(fx.Int32, fx.get_iter(p_num_valid_ids)), fx.make_layout(1, 1))
        max_valid_id = num_valid_buf[0]
        if const_expr(p_weight.dtype != fx.BFloat16):
            p_weight = fx.recast_iter(fx.Uint8, fx.get_iter(p_weight))
        else:
            p_weight = fx.get_iter(p_weight)
        if e_idx * BLOCK_TILE_SIZE_M < max_valid_id:
            lds = fx.SharedAllocator().allocate(SharedStorage).peek()
            arg_p_sorted_ids = fx.make_view(fx.recast_iter(fx.Int32, fx.get_iter(p_sorted_ids) + e_idx * BLOCK_TILE_SIZE_M), fx.make_layout(BLOCK_TILE_SIZE_M, 1))
            arg_p_sorted_expert_ids = fx.recast_iter(fx.Int32, fx.get_iter(p_sorted_expert_ids))
            expert_id = arg_p_sorted_expert_ids[e_idx]
            element_num = 16 // (p_weight.dtype.width // 8)
            arg_p_weight = fx.make_view(p_weight + fx.Int64(expert_id * N * K),
                                        # preshuffle layout: [16, (8, K // 8)]
                                        fx.make_layout(((16, N // 16), (element_num, K // element_num)), ((element_num, 16 * K), (1, 16 * element_num))))
            
            # sorted ids: global -> LDS (scalar load/store, only first BLOCK_TILE_SIZE_M threads participate)
            sorted_ids_buf = fx.rocdl.make_buffer_tensor(arg_p_sorted_ids, max_size=False)
            lds_view = fx.make_view(lds.sorted_lds.ptr, fx.make_layout(BLOCK_TILE_SIZE_M, 1))
            for idx in range(tid, BLOCK_TILE_SIZE_M, 64):
                # fx.memref_store(val, lds_view, tid)
                lds_view[idx] = sorted_ids_buf[idx]
            gpu.barrier()

            cp_atom_weight = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
            arg_p_sorted_weights = fx.make_view(fx.recast_iter(fx.Float32, fx.get_iter(p_sorted_weights) + e_idx * BLOCK_TILE_SIZE_M), fx.make_layout(BLOCK_TILE_SIZE_M, 1))
            sorted_weights_buf = fx.rocdl.make_buffer_tensor(arg_p_sorted_weights, max_size=False)
            cp_atom_lds = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Int32)
            tiled_copy_sortid_lds = fx.make_tiled_copy(cp_atom_lds, fx.make_layout(((16, 4), 1), ((1, 0), 0)), fx.make_tile(16))
            sorted_weights_tensor = tiled_copy_sortid_lds.get_slice(tid).partition_S(sorted_weights_buf)
            sorted_weight_frag = fx.make_fragment_like(sorted_weights_tensor, fx.Float32)
            fx.copy(cp_atom_weight, sorted_weights_tensor, sorted_weight_frag)

            c_frag = gemm_splitk(BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N, TILE_K, blk_n, arg_p_input, arg_p_weight, lds, splitk_waves=1)

            _apply_down_scale(c_frag, tid, expert_id, blk_n, p_w_scale)

            # mul weight
            sorted_weight_frag_vec = sorted_weight_frag.load()
            for m in range_constexpr(BLOCK_TILE_SIZE_M // 16):
                w = sorted_weight_frag_vec[m]
                v = c_frag[None, m, None].load()
                v *= w
                c_frag[None, m, None].store(v)

            c_frag_bf16 = _cvt_f32_to_bf16(c_frag)

            # write to mem
            if const_expr(p_output.shape.rank == 3):  # gateup output shape: [M, TOPK, N]
                arg_p_output = fx.make_view(fx.get_iter(p_output), fx.make_layout((M, TOPK, N), (TOPK * N, N, 1)))
                arg_p_output = fx.rocdl.make_buffer_tensor(arg_p_output, max_size=False, num_records_bytes=M * TOPK * N * fx.BFloat16.width // 8)
                cp_atom_w = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.BFloat16)
                is_atomic_write = False
            else:
                arg_p_output = fx.make_view(fx.get_iter(p_output), fx.make_layout((M, N), (N, 1)))
                # arg_p_output = fx.rocdl.make_buffer_tensor(arg_p_output, max_size=False, num_records_bytes=M * TOPK * N * fx.BFloat16.width // 8)
                # cp_atom_w = fx.make_copy_atom(fx.rocdl.BufferAtomicAdd(fx.BFloat16), fx.BFloat16)
                cp_atom_w = fx.make_copy_atom(fx.UniversalAtomic(fx.AtomicOp.Add, fx.BFloat16), fx.BFloat16)
                is_atomic_write = True
            c_tiled_g = fx.make_tiled_copy(cp_atom_w,
                                        # 16x4 threads, each writes 4 points in N dimension
                                        fx.make_layout(((16, 4), 4), ((1, 64), 16)),
                                        fx.make_tile(16, 16))
            c_tensor = TensorWithIndex(arg_p_output, BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N, tiled_copy_sortid_lds, c_tiled_g, tid, lds.sorted_lds, is_read_from_mem=False, TOPK=TOPK, is_atomic_write=is_atomic_write)
            c_tensor.copy(cp_atom_w, blk_n, c_tiled_g.get_slice(tid).retile(c_frag_bf16))

    @flyc.kernel
    def moe_2stage_gateup_batch1(p_input: fx.Tensor,            # bf16 [M, K]
                                 p_weight: fx.Tensor,           # quantized/bf16 [N/16, K/8 * 16 * 8]
                                 p_output: fx.Tensor,           # bf16 [M, TOPK, N//2]
                                 p_topk_ids: fx.Tensor,         # int32 [M, TOPK]
                                 p_w_scale: fx.Tensor,
                                ):
        tid = gpu.thread_idx.x
        blk_n = gpu.block_idx.x
        e_idx = gpu.block_idx.y

        arg_p_input = fx.make_view(fx.get_iter(p_input), fx.make_layout((1, K), (K, 1)))
        if const_expr(p_weight.dtype != fx.BFloat16):
            p_weight = fx.recast_iter(fx.Uint8, fx.get_iter(p_weight))
        else:
            p_weight = fx.get_iter(p_weight)
        arg_p_expert_ids = fx.recast_iter(fx.Int32, fx.get_iter(p_topk_ids))
        expert_id = arg_p_expert_ids[e_idx]
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        # there is a reduce in gemm_splitk which will read/write from lds, the BLOCK_TILE_SIZE_N will impact the coalesced access:
        # BLOCK_TILE_SIZE_N BLOCK_TILE_SIZE_N//2(after silu) LDS_read_per_lane  MEM_write_per_lane
        # 64                32                               2=(32/16 threads)  2=(32/16 threads)
        # 128               64                               4=(64/16 threads)  4=(64/16 threads)
        # 256: will split into 2x128
        contiguous_n = 64 if const_expr(BLOCK_TILE_SIZE_N % 128 == 0) else 32

        group_layout_silu = fx.make_layout(((contiguous_n, 2, N // (contiguous_n * 2)), K), ((1, N // 2, contiguous_n), N))
        element_num = 16 // (p_weight.dtype.width // 8)
        arg_p_weight = fx.make_view(p_weight + fx.Int64(expert_id * N * K),
                                    # preshuffle layout: [16, (8, K // 8)]
                                    fx.composition(fx.make_layout(((16, N // 16), (element_num, K // element_num)), ((element_num, 16 * K), (1, 16 * element_num))),
                                                    group_layout_silu))                                                     # NOTE: assume permuted adjacent 32 rows will fall in the same wave to do silu
        
        c_frag = gemm_splitk(BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N, TILE_K, blk_n, arg_p_input, arg_p_weight, lds, splitk_waves=4, a_with_index=False)

        c_frag_bf16 = _apply_scale_silu_bf16(c_frag, tid, expert_id, blk_n, contiguous_n, p_w_scale)

        arg_p_output = fx.make_view(fx.get_iter(p_output), fx.make_layout((1, TOPK, N // 2), (TOPK * N // 2, N // 2, 1)))
        out_tensor = fx.rocdl.make_buffer_tensor(arg_p_output, max_size=False, num_records_bytes=1 * TOPK * N // 2 * fx.BFloat16.width // 8)
        cp_atom_w = fx.make_copy_atom(fx.rocdl.BufferCopy64b() if const_expr(BLOCK_TILE_SIZE_N % 128 == 0) else fx.rocdl.BufferCopy32b(), fx.BFloat16)
        c_tiled_g = fx.make_tiled_copy(cp_atom_w,
                                    # thread mapping: 4 wavex(4x16), (contiguous_n // 16) elements per lane
                                    fx.make_layout(((16, 4, 4), contiguous_n // 16), ((contiguous_n, 1, 4), 16)),
                                    fx.make_tile(16, contiguous_n))
        c_tile = fx.flat_divide(out_tensor[None, e_idx, None], fx.make_tile(BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N // 2))[None, None, None, blk_n]
        c_dst = c_tiled_g.get_slice(tid).partition_S(c_tile)
        c_src = c_tiled_g.get_slice(tid).retile(c_frag_bf16)

        # Copy each (rep_m, rep_n) value slice individually. A single bulk copy of the retiled
        # register fails rmem->SSA promotion (the broadcast register memref cannot be vectorized as
        # a whole); per-slice copies keep the register access promotable, matching the indexed
        # gateup path.
        rep_m = fx.size(fx.get_shape(c_src)[1]).to_py_value()
        rep_n = fx.size(fx.get_shape(c_src)[2]).to_py_value()
        for m in range_constexpr(rep_m):
            for n in range_constexpr(rep_n):
                fx.copy(cp_atom_w, c_src[None, m, n], c_dst[None, m, n, 0])

    @flyc.kernel
    def moe_2stage_down_batch1(p_input: fx.Tensor,            # bf16 [M, TOPK, K]
                               p_weight: fx.Tensor,           # quantized/bf16 [N/16, K/8 * 16 * 8]
                               p_output: fx.Tensor,           # bf16 [M, N]
                               p_topk_ids: fx.Tensor,
                               p_topk_weights: fx.Tensor,
                               p_w_scale: fx.Tensor,
                               ):
        tid = gpu.thread_idx.x
        blk_n = gpu.block_idx.x
        e_idx = gpu.block_idx.y

        # batch1: input is gemm1_out[0, e_idx, :] (single token, expert slot e_idx). Point at that
        # row and broadcast it across the TILE_M MFMA rows (stride 0); every computed row is then
        # identical, so any single row is the real result.
        arg_p_input = fx.make_view(fx.get_iter(p_input) + fx.Int64(e_idx * K), fx.make_layout((BLOCK_TILE_SIZE_M, K), (0, 1)))
        if const_expr(p_weight.dtype != fx.BFloat16):
            p_weight = fx.recast_iter(fx.Uint8, fx.get_iter(p_weight))
        else:
            p_weight = fx.get_iter(p_weight)
        arg_p_topk_ids = fx.recast_iter(fx.Int32, fx.get_iter(p_topk_ids))
        arg_p_topk_weights = fx.recast_iter(fx.Float32, fx.get_iter(p_topk_weights))
        expert_id = arg_p_topk_ids[e_idx]
        topk_weight = arg_p_topk_weights[e_idx]
        element_num = 16 // (p_weight.dtype.width // 8)
        arg_p_weight = fx.make_view(p_weight + fx.Int64(expert_id * N * K),
                                    # preshuffle layout: [16, (8, K // 8)]
                                    fx.make_layout(((16, N // 16), (element_num, K // element_num)), ((element_num, 16 * K), (1, 16 * element_num))))

        c_frag = gemm_splitk(BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N, TILE_K, blk_n, arg_p_input, arg_p_weight, None, splitk_waves=1, a_with_index=False)

        _apply_down_scale(c_frag, tid, expert_id, blk_n, p_w_scale)

        # mul weight
        c_frag.store(c_frag.load() * topk_weight)

        c_frag_bf16 = _cvt_f32_to_bf16(c_frag)

        # write to mem
        arg_p_output = fx.make_view(fx.get_iter(p_output), fx.make_layout((1, N), (N, 1)))
        cp_atom_w = fx.make_copy_atom(fx.UniversalAtomic(fx.AtomicOp.Add, fx.BFloat16), fx.BFloat16)
        c_tiled_g = fx.make_tiled_copy(cp_atom_w,
                                    # 16x4 threads, each writes 4 points in N dimension
                                    fx.make_layout(((16, 4), 4), ((1, 64), 16)),
                                    fx.make_tile(16, 16))
        c_tile = fx.flat_divide(arg_p_output, fx.make_tile(BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N))[None, None, None, blk_n]
        c_dst = c_tiled_g.get_slice(tid).partition_S(c_tile)
        c_src = c_tiled_g.get_slice(tid).retile(c_frag_bf16)
        rep_m = fx.size(fx.get_shape(c_src)[1]).to_py_value()
        rep_n = fx.size(fx.get_shape(c_src)[2]).to_py_value()
        if tid % 16 == 0:
            for m in range_constexpr(rep_m):
                for n in range_constexpr(rep_n):
                    reg_vec = c_src[None, m, n].load()
                    ptr_base = fx.get_iter(c_dst[None, m, n, 0])
                    for i in range_constexpr(reg_vec.numel // 2):
                        pair = Vec.from_elements([reg_vec[i * 2], reg_vec[i * 2 + 1]], fx.BFloat16)
                        ptr = ptr_base + i * 2
                        addr = fx.ptrtoint(ptr)
                        llvm_ptr_ty = ir.Type.parse("!llvm.ptr<1>")
                        llvm_ptr = llvm.IntToPtrOp(llvm_ptr_ty, addr.ir_value())
                        llvm.AtomicRMWOp(llvm.AtomicBinOp.fadd, llvm_ptr, pair, llvm.AtomicOrdering.monotonic, syncscope="agent", alignment=4)

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
        if const_expr(stage == 'gateup'):
            moe_2stage_gateup(
                p_input, p_weight, p_output,
                p_sorted_ids, p_sorted_weights, p_sorted_expert_ids,
                p_num_valid_ids, p_w_scale, M,
            ).launch(
                grid=(num_n_blocks, grid, 1),
                block=(256, 1, 1),
                stream=stream,
            )
        else:
            moe_2stage_down(
                p_input, p_weight, p_output,
                p_sorted_ids, p_sorted_weights, p_sorted_expert_ids,
                p_num_valid_ids, p_w_scale, M,
            ).launch(
                grid=(num_n_blocks, grid, 1),
                block=(64, 1, 1),
                stream=stream,
            )

    @flyc.jit
    def launch_batch1(
        p_input: fx.Tensor,
        p_weight: fx.Tensor,
        p_output: fx.Tensor,
        p_topk_ids: fx.Tensor,
        p_topk_weights: fx.Tensor,
        p_w_scale: fx.Tensor,
        stream: fx.Stream,
    ):
        CompilationContext.get_current()
        num_n_blocks = div_up(N, BLOCK_TILE_SIZE_N)
        grid = fx.size(p_topk_ids.shape).to_py_value()
        if const_expr(stage == 'gateup'):
            moe_2stage_gateup_batch1(
                p_input, p_weight, p_output, p_topk_ids, p_w_scale
            ).launch(
                grid=(num_n_blocks, grid, 1),
                block=(256, 1, 1),
                stream=stream,
            )
        else:
            moe_2stage_down_batch1(
                p_input, p_weight, p_output, p_topk_ids, p_topk_weights, p_w_scale
            ).launch(
                grid=(num_n_blocks, grid, 1),
                block=(64, 1, 1),
                stream=stream,
            )

    if const_expr(alg == 'batch1'):
        return launch_batch1
    return launch
