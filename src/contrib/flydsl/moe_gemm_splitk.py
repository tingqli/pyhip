import os


import flydsl.compiler as flyc  # noqa: E402
from flydsl.compiler.kernel_function import CompilationContext  # noqa: E402
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.utils.env import DebugEnvManager
from flydsl._mlir import ir
import flydsl
from flydsl._mlir.dialects import llvm
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr.utils.arith import _to_raw as _raw

# debug
if 0:
    DebugEnvManager.enable_debug_info = True
    ir._globals.register_traceback_file_inclusion(__file__)
    ir._globals.register_traceback_file_exclusion(os.path.dirname(flydsl.__file__))
    ir._globals.set_loc_tracebacks_frame_limit(40)
    ir._globals.set_loc_tracebacks_enabled(True)
    os.environ.setdefault("FLYDSL_RUNTIME_ENABLE_CACHE", "0")


def _as_ptr(p):
    """Convert memref or pointer to a pointer/iterator suitable for fx.make_view.
    Handles both raw fx.Pointer values and memref values passed by flydsl runtime."""
    try:
        return fx.get_iter(p)
    except Exception:
        return p


def div_up(x, y):
    return (x + y - 1) // y


def compile_gemm(
    N,
    K,
    weight_dtype,
    weight_quant_type,
    TOPK,
    BLOCK_TILE_SIZE_M,
    BLOCK_TILE_SIZE_N,
    stage="gateup",
    alg="splitk",
    E=None,
    USE_ATOMIC_WRITE=True,
    act_quant_type=None,
    tile_k=None,
):

    TILE_K = 64
    # Optional TILE_K override for the prefill_1x4 alg. The env fallback lets test_moe.py /
    # profile scripts pick BK without threading a kwarg through every caller. bf16 prefill_1x4
    # supports BK in {64, 128} (the per-ki gemm loop); fp8 stays 128.
    if tile_k is None and os.environ.get("MOE_PREFILL_TILE_K"):
        tile_k = int(os.environ["MOE_PREFILL_TILE_K"])
    # weight_quant_type governs the WEIGHT scale form; act_quant_type governs the ACTIVATION
    # scale form (native-fp8 prefill only) and defaults to weight_quant_type (legacy behavior
    # where a single quant_type drove both).
    if act_quant_type is None:
        act_quant_type = weight_quant_type
    assert (
        BLOCK_TILE_SIZE_M <= 256
    ), "BLOCK_SIZE_M must be less than or equal to 256 due to LDS size limit for sorted ids."
    assert weight_dtype in [
        "bf16",
        "fp8",
    ], "weight_dtype must be either 'bf16' or 'fp8'"
    assert weight_quant_type in [
        "no",
        "ptpc",
        "per_tensor",
    ], "weight_quant_type must be either 'no', 'ptpc' or 'per_tensor'"
    assert act_quant_type in [
        "no",
        "ptpc",
        "per_tensor",
    ], "act_quant_type must be either 'no', 'ptpc' or 'per_tensor'"
    # Supported native-fp8 prefill (weight, act) combos: weight ptpc requires act ptpc;
    # weight per_tensor allows act ptpc or per_tensor.
    if weight_dtype == "fp8" and alg in ("prefill_2x2", "prefill_1x4"):
        assert (weight_quant_type == "ptpc" and act_quant_type == "ptpc") or (
            weight_quant_type == "per_tensor"
            and act_quant_type in ("ptpc", "per_tensor")
        ), (
            f"unsupported prefill quant combo (weight={weight_quant_type}, "
            f"act={act_quant_type})"
        )

    if stage == "gateup" and alg == "splitk":
        assert (
            BLOCK_TILE_SIZE_N % 64 == 0
        ), "For split-k, BLOCK_TILE_SIZE_N needs to be multiple of 64 due to reduce layout."
        assert K % (32 * 4) == 0, "K must be a multiple of 128 for split-k algorithm."
        c_reduce_lds_size = (
            16 * 64 * 4
        )  # save LDS size instead of BLOCK_TILE_SIZE_M * BLOCK_TILE_SIZE_N * 4

        @fx.union
        class SharedStorage:
            sorted_lds: fx.Array[fx.Int32, 256, 16]
            c_reduce_lds: fx.Array[fx.Float32, c_reduce_lds_size, 16]

    elif stage == "down" and alg == "splitk":

        @fx.struct
        class SharedStorage:
            sorted_lds: fx.Array[fx.Int32, 256, 16]

    elif stage == "gateup" and alg == "batch1":
        c_reduce_lds_size = (
            16 * 64 * 4
        )  # save LDS size instead of BLOCK_TILE_SIZE_M * BLOCK_TILE_SIZE_N * 4

        @fx.struct
        class SharedStorage:
            c_reduce_lds: fx.Array[fx.Float32, c_reduce_lds_size, 16]

    elif stage == "gateup" and alg == "prefill_2x2":
        # CShuffle read packs value(8 bf16) x chan_thread(4) x waveN(2) = 64 channels per
        # tile, i.e. contiguous_n = BN//2 must be a multiple of 64 -> BN a multiple of 128.
        # Supported range for the prefill_2x2 alg is BN in [128, 256] (BM in [64, 256]); the
        # dual-buffer CShuffle epilogue below stages both C halves in GemmBuffers at once.
        assert 128 <= BLOCK_TILE_SIZE_N <= 256 and BLOCK_TILE_SIZE_N % 128 == 0, (
            "For prefill_2x2 alg, BLOCK_TILE_SIZE_N must be in [128, 256] and a multiple of 128 "
            "(CShuffle channel pack = 8 bf16 x 4 lanes x 2 waves = 64 = contiguous_n = BN//2)"
        )
        assert 64 <= BLOCK_TILE_SIZE_M <= 256 and BLOCK_TILE_SIZE_M % 64 == 0, (
            "For prefill_2x2 alg, BLOCK_TILE_SIZE_M must be in [64, 256] and a multiple of 64 "
            "(each M-half >= 32 for the 4-wave load scheme)"
        )
        # fp8 issues 128-bit g2r loads (16 fp8/thread); widen the K-tile to 128 so a
        # 32xTILE_K sub-tile is 32x128 = 256 threads x 16 fp8. bf16's 128-bit load is
        # already 8 bf16, so it keeps TILE_K=64.
        TILE_K = 128 if weight_dtype == "fp8" else 64
        a_lds_size_half = BLOCK_TILE_SIZE_M // 2 * TILE_K
        b_lds_size_half = BLOCK_TILE_SIZE_N // 2 * TILE_K
        # Native fp8 feeds MFMA(16,16,32) directly from fp8 LDS (no bf16 decompress).
        lds_elem = fx.Float8E4M3FNUZ if weight_dtype == "fp8" else fx.BFloat16

        @fx.struct
        class GemmBuffers:
            at_lds: fx.Array[lds_elem, a_lds_size_half, 16]
            ab_lds: fx.Array[lds_elem, a_lds_size_half, 16]
            bl_lds: fx.Array[lds_elem, b_lds_size_half, 16]
            br_lds: fx.Array[lds_elem, b_lds_size_half, 16]

        @fx.union
        class SharedStorage:
            sorted_lds: fx.Array[fx.Int32, 256, 16]
            gemm: GemmBuffers

        # The CShuffle epilogue stages BOTH C halves (each BM//2 x contiguous_n bf16) into
        # two non-overlapping GemmBuffers regions at once (dual-buffer -> 2 barriers instead
        # of 4). Confirm the allocated GemmBuffers >= 2 halves.
        _cshuffle_elem_bytes = 1 if weight_dtype == "fp8" else 2
        _gemm_lds_bytes = (2 * a_lds_size_half + 2 * b_lds_size_half) * _cshuffle_elem_bytes
        _cshuffle_bytes = 2 * (BLOCK_TILE_SIZE_M // 2 * (BLOCK_TILE_SIZE_N // 2) * 2)
        assert _cshuffle_bytes <= _gemm_lds_bytes, (
            f"CShuffle needs {_cshuffle_bytes} B of LDS but GemmBuffers only allocates "
            f"{_gemm_lds_bytes} B (BM={BLOCK_TILE_SIZE_M}, BN={BLOCK_TILE_SIZE_N})"
        )

    elif stage == "gateup" and alg == "prefill_1x4":
        # 1x4: the 4 waves tile the N(channel) direction (vs prefill_2x2's 2x2 wave grid),
        # and the full TILE_M is shared across all waves (no M-split). B (weight gate/up)
        # loads direct global->register (no LDS); only A (activation) is staged through LDS
        # with a ping-pong double buffer (a_ping / a_pong). Same BN/BM range as prefill_2x2.
        assert 128 <= BLOCK_TILE_SIZE_N <= 256 and BLOCK_TILE_SIZE_N % 128 == 0, (
            "For prefill_1x4 alg, BLOCK_TILE_SIZE_N must be in [128, 256] and a multiple of 128 "
            "(each wave owns contiguous_n//4 = BN//8 output channels, a multiple of 16)"
        )
        assert 64 <= BLOCK_TILE_SIZE_M <= 256 and BLOCK_TILE_SIZE_M % 64 == 0, (
            "For prefill_1x4 alg, BLOCK_TILE_SIZE_M must be in [64, 256] and a multiple of 64"
        )
        # fp8 issues 128-bit g2r loads (16 fp8/thread) -> widen K-tile to 128; bf16 keeps 64.
        # tile_k override (default 64 for bf16 / 128 for fp8) enables the BK sweep; the per-ki
        # gemm loop in _gemm_1x4 handles bf16 TILE_K in {64, 128} and fp8 TILE_K in {128, 256}.
        TILE_K = tile_k if tile_k is not None else (128 if weight_dtype == "fp8" else 64)
        if weight_dtype == "fp8":
            assert TILE_K in (128, 256), f"prefill_1x4 fp8 TILE_K must be 128 or 256, got {TILE_K}"
        else:
            assert TILE_K in (64, 128), f"prefill_1x4 bf16 TILE_K must be 64 or 128, got {TILE_K}"
        assert K % TILE_K == 0, f"prefill_1x4 K={K} must be a multiple of TILE_K={TILE_K}"
        a_lds_size = BLOCK_TILE_SIZE_M * TILE_K  # full A tile; ping-pong needs two buffers
        lds_elem = fx.Float8E4M3FNUZ if weight_dtype == "fp8" else fx.BFloat16

        @fx.struct
        class GemmBuffers:
            a_ping: fx.Array[lds_elem, a_lds_size, 16]
            a_pong: fx.Array[lds_elem, a_lds_size, 16]

        @fx.union
        class SharedStorage:
            sorted_lds: fx.Array[fx.Int32, 256, 16]
            gemm: GemmBuffers

        # The CShuffle epilogue stages the single (BM x contiguous_n) bf16 output into the
        # A LDS (reused after the main loop). Confirm it fits.
        _cshuffle_elem_bytes = 1 if weight_dtype == "fp8" else 2
        _gemm_lds_bytes = 2 * a_lds_size * _cshuffle_elem_bytes
        _cshuffle_bytes = BLOCK_TILE_SIZE_M * (BLOCK_TILE_SIZE_N // 2) * 2
        assert _cshuffle_bytes <= _gemm_lds_bytes, (
            f"CShuffle needs {_cshuffle_bytes} B of LDS but GemmBuffers only allocates "
            f"{_gemm_lds_bytes} B (BM={BLOCK_TILE_SIZE_M}, BN={BLOCK_TILE_SIZE_N})"
        )

    if weight_dtype == "bf16":
        weight_dtype = fx.BFloat16
    elif weight_dtype == "fp8":
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
        def __init__(
            self,
            view,
            tile_m,
            tile_k,
            tiled_copy_index: fx.TiledCopy,
            tiled_copy: fx.TiledCopy,
            tid,
            lds_index,
            is_read_from_mem=True,
            TOPK=None,
            is_atomic_write=False,
            index_size=None,
            index_offset=0,
            index_frag=None,
        ):
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
            self.tensor_blocks_in_k = fx.zipped_divide(
                view, fx.make_tile(*dims, tile_k)
            )

            # read index (or reuse a pre-computed one when LDS is no longer available,
            # e.g. sorted_lds has been overwritten by the CShuffle epilogue).
            if index_frag is not None:
                self.index_frag = index_frag
            else:
                if index_size is None:
                    index_size = BLOCK_TILE_SIZE_M
                lds = fx.make_view(
                    lds_index.ptr + index_offset, fx.make_layout(index_size, 1)
                )
                cp_atom_lds = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Int32)
                lds_thr = tiled_copy_index.get_slice(tid).partition_S(lds)
                self.index_frag = fx.make_fragment_like(lds_thr)
                fx.copy(cp_atom_lds, lds_thr, self.index_frag)

            dtype = fx.PointerType.get(fx.Int8.ir_type, 1, 512)
            ptr = fx.inttoptr(dtype, fx.Int32(0))
            self.fake_tensor = fx.make_view(
                ptr, fx.make_layout((tile_m, tile_k), (1, tile_m))
            )
            self.fake_tensor_thr = (
                tiled_copy.get_slice(tid).partition_S(self.fake_tensor)
                if is_read_from_mem
                else tiled_copy.get_slice(tid).partition_D(self.fake_tensor)
            )
            # since init ptr is zero, it will be the offset of the thread in the tile after partition_S
            offset_thread = fx.Int32(fx.ptrtoint(fx.get_iter(self.fake_tensor_thr)))
            self.offset_thread_k = offset_thread // tile_m

        def copy(self, copy_atom, k_idx, frag: fx.Tensor):
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
                    tensor_sub_block = tensor_block[
                        None, self.index_frag[0, m] & 0xFFFFFF
                    ]
                else:
                    tensor_sub_block = tensor_block[
                        None,
                        self.index_frag[0, m] & 0xFFFFFF,
                        (self.index_frag[0, m] >> 24),
                    ]
                if const_expr(not self.is_atomic_write):
                    for k in range_constexpr(rep_k):
                        # get block k index
                        offset_block = fx.crd2idx((0, m, k), layout).to_py_value()
                        offset_block_k = offset_block // self.tile_m
                        # NOTE: assume K is linear in memory
                        offset_k_in_tile = offset_block_k + self.offset_thread_k
                        reg = frag[None, m, k]
                        mem = fx.make_view(
                            fx.get_iter(tensor_sub_block) + offset_k_in_tile,
                            fx.make_layout(value_size, stride_size),
                        )
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
                            mem = fx.make_view(
                                fx.get_iter(tensor_sub_block) + offset_k_in_tile,
                                fx.make_layout(value_size, stride_size),
                            )
                            reg_vec = reg.load()
                            ptr_base = fx.get_iter(mem)
                            for i in range_constexpr(reg_vec.numel // 2):
                                pair = Vec.from_elements(
                                    [reg_vec[i * 2], reg_vec[i * 2 + 1]], fx.BFloat16
                                )
                                ptr = ptr_base + i * 2
                                addr = fx.ptrtoint(ptr)
                                llvm_ptr_ty = ir.Type.parse("!llvm.ptr<1>")
                                llvm_ptr = llvm.IntToPtrOp(llvm_ptr_ty, addr.ir_value())
                                llvm.AtomicRMWOp(
                                    llvm.AtomicBinOp.fadd,
                                    llvm_ptr,
                                    pair,
                                    llvm.AtomicOrdering.monotonic,
                                    syncscope="agent",
                                    alignment=4,
                                )

    TensorWithIndex.copy = ASTRewriter.transform(TensorWithIndex.copy)

    def select(tensor: fx.Tensor, order):
        rank = fx.get_shape(tensor).rank
        assert len(order) == rank
        stride = fx.get_stride(tensor)
        shape = fx.get_shape(tensor)
        new_layout = fx.make_layout(
            [shape[i] for i in order], [stride[i] for i in order]
        )
        return fx.make_view(fx.get_iter(tensor), new_layout)

    def cvt_fp8_bf16(src_tensor: fx.Tensor, dst_tensor: fx.Tensor):
        size = fx.size(fx.get_shape(src_tensor)).to_py_value()

        items = []
        src_vec = src_tensor.load().bitcast(fx.Uint32)
        for i in range_constexpr(size // 4):
            src_val = src_vec[i]
            pk0_f32 = llvm.inline_asm(
                T.f32x2,
                [src_val.ir_value()],
                "v_cvt_pk_f32_fp8 $0, $1",
                "=v,v",
                has_side_effects=False,
            )
            pk1_f32 = llvm.inline_asm(
                T.f32x2,
                [src_val.ir_value()],
                "v_cvt_pk_f32_fp8_sdwa $0, $1 src0_sel:WORD_1",
                "=v,v",
                has_side_effects=False,
            )
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
        # The reduce makes gate/up adjacent (2i, 2i+1).
        v_reps = fx.size(fx.get_shape(c_frag)[0]).to_py_value()
        m_reps = fx.size(fx.get_shape(c_frag)[1]).to_py_value()
        n_reps = fx.size(fx.get_shape(c_frag)[2]).to_py_value()

        if const_expr(weight_dtype != fx.BFloat16):
            if const_expr(weight_quant_type == "ptpc"):
                group_layout_silu = fx.make_layout(
                    ((contiguous_n, 2, N // (2 * contiguous_n)), 1),
                    ((1, N // 2, contiguous_n), 0),
                )
                arg_p_scale = fx.make_view(
                    _as_ptr(p_w_scale) + expert_id * N,
                    fx.composition(fx.make_layout(N, 1), group_layout_silu),
                )
                scale_tile = fx.flat_divide(
                    arg_p_scale, fx.make_tile(BLOCK_TILE_SIZE_N, 1)
                )[None, None, blk_n, 0]
                cp_atom_scale = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32)
                tiled_copy_scale = fx.make_tiled_copy(
                    cp_atom_scale,
                    fx.make_layout(
                        ((16, 4, 4), contiguous_n // 16),
                        ((contiguous_n // 16, 0, 0), 1),
                    ),
                    fx.make_tile(contiguous_n, 1),
                )
                scale_frag_tensor = tiled_copy_scale.get_slice(tid).partition_S(
                    scale_tile
                )
                scale_frag = fx.make_fragment_like(scale_frag_tensor)
                fx.copy(cp_atom_scale, scale_frag_tensor, scale_frag)
                for n in range_constexpr(n_reps):
                    scale_vec = scale_frag[None, n, 0].load()
                    for m in range_constexpr(m_reps):
                        c_vec = c_frag[None, m, n].load()
                        vec = c_vec * scale_vec
                        c_frag[None, m, n].store(vec)
            elif const_expr(weight_quant_type == "per_tensor"):
                arg_p_scale = fx.make_view(
                    _as_ptr(p_w_scale) + expert_id, fx.make_layout(1, 1)
                )
                scale = arg_p_scale[0]
                c_frag.store(c_frag.load() * scale)

        # c_frag_bf16 stores the silu result (half the N dimension since gate+up -> 1 output)
        n_half = n_reps // 2
        if const_expr(v_reps == 1):
            # v_reps==1 (TILE_N=32): flat value mode with stride 0 to avoid
            # ((1,1),...):((1,0),...) producing two stride-1 leaves in findContigSegment.
            c_frag_bf16 = fx.make_rmem_tensor(
                fx.make_layout((1, m_reps, n_half), (0, n_half, 1)), fx.BFloat16
            )
        else:
            c_frag_bf16 = fx.make_rmem_tensor(
                fx.make_layout(
                    ((v_reps, 1), m_reps, n_half), ((1, 0), n_half * v_reps, v_reps)
                ),
                fx.BFloat16,
            )

        log2_exp1 = -1.4426950408889634
        for i in range_constexpr(n_reps // 2):
            gate = c_frag[None, None, 2 * i + 0].load()
            up = c_frag[None, None, 2 * i + 1].load()
            gate_log2 = gate * log2_exp1
            acc = []
            for j in range_constexpr(gate.numel):
                tmp = rocdl.exp2(T.f32, _raw(gate_log2[j]))
                acc.append((gate[j] * rocdl.rcp(T.f32, 1.0 + tmp)) * up[j])
            acc = Vec.from_elements(acc, fx.Float32)
            round_bit = fx.Uint32(0x8000)
            acc = (
                ((acc.bitcast(fx.Uint32) + round_bit) >> 16)
                .to(fx.Uint16)
                .bitcast(fx.BFloat16)
            )
            c_frag_bf16[None, None, i].store(acc)

        return c_frag_bf16

    def _apply_down_scale(c_frag, tid, expert_id, blk_n, p_w_scale):
        if const_expr(weight_dtype != fx.BFloat16):
            if const_expr(weight_quant_type == "ptpc"):
                arg_p_scale = fx.make_view(
                    _as_ptr(p_w_scale) + expert_id * N, fx.make_layout(N, 1)
                )
                scale_tile = fx.flat_divide(
                    arg_p_scale, fx.make_tile(BLOCK_TILE_SIZE_N)
                )[None, blk_n]
                cp_atom_scale = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Float32)
                tiled_copy_scale = fx.make_tiled_copy(
                    cp_atom_scale,
                    fx.make_layout(((16, 4), 4), ((0, 4), 1)),
                    fx.make_tile(16),
                )
                scale_frag_tensor = tiled_copy_scale.get_slice(tid).partition_S(
                    scale_tile
                )
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
            elif const_expr(weight_quant_type == "per_tensor"):
                arg_p_scale = fx.make_view(
                    _as_ptr(p_w_scale) + expert_id, fx.make_layout(1, 1)
                )
                scale = arg_p_scale[0]
                c_frag.store(c_frag.load() * scale)

    def _cvt_f32_to_bf16(c_frag):
        c_frag_bf16 = fx.make_fragment_like(c_frag, dtype=fx.BFloat16)
        round_bit = fx.Uint32(0x8000)
        c_frag_bf16.store(
            ((c_frag.load().bitcast(fx.Uint32) + round_bit) >> 16)
            .to(fx.Uint16)
            .bitcast(fx.BFloat16)
        )
        return c_frag_bf16

    def _silu_pair_bf16(gate_frag, up_frag, gate_scale=None, up_scale=None, a_scale=None):
        # silu(gate) * up, element-wise over identically-laid-out gate/up fragments.
        # Used by the 4-wave compute path where gate (left N-half) and up (right N-half)
        # land in separate quadrant fragments with matching layout. Iterate (m, n)
        # explicitly so the result keeps the fragment's [v, m, n] positions. Optional
        # per-N-channel fp8 weight scales (shape [value, rep_n]) and an optional per-row
        # fp8 activation scale (a_scale[m], one per C M-row) are folded into the read so
        # native-fp8 dequant happens before the non-linear silu.
        log2_exp1 = -1.4426950408889634
        round_bit = fx.Uint32(0x8000)
        out_bf16 = fx.make_fragment_like(gate_frag, dtype=fx.BFloat16)
        m_reps = fx.size(fx.get_shape(gate_frag)[1]).to_py_value()
        n_reps = fx.size(fx.get_shape(gate_frag)[2]).to_py_value()
        for m in range_constexpr(m_reps):
            if const_expr(a_scale is not None):
                a_sc = a_scale[m]
            for n in range_constexpr(n_reps):
                gate = gate_frag[None, m, n].load()
                up = up_frag[None, m, n].load()
                if const_expr(gate_scale is not None):
                    sc_g = gate_scale[None, n].load()
                    sc_u = up_scale[None, n].load()
                acc = []
                for j in range_constexpr(gate.numel):
                    g = gate[j]
                    u = up[j]
                    if const_expr(gate_scale is not None):
                        g = g * sc_g[j]
                        u = u * sc_u[j]
                    if const_expr(a_scale is not None):
                        g = g * a_sc
                        u = u * a_sc
                    tmp = rocdl.exp2(T.f32, _raw(g * log2_exp1))
                    acc.append((g * rocdl.rcp(T.f32, 1.0 + tmp)) * u)
                acc = Vec.from_elements(acc, fx.Float32)
                acc = (
                    ((acc.bitcast(fx.Uint32) + round_bit) >> 16)
                    .to(fx.Uint16)
                    .bitcast(fx.BFloat16)
                )
                out_bf16[None, m, n].store(acc)
        return out_bf16

    def _gemm_splitk(
        TILE_M,
        TILE_N,
        TILE_K,
        blk_n: int,  # block index for N dimension
        arg_p_input: fx.Tensor,  # [M, K] or [M, TOPK, K]
        arg_p_weight: fx.Tensor,  # [(16,N/16), (8, K/8)]
        lds,
        splitk_waves=4,
        a_with_index=True,
    ):
        tid = gpu.thread_idx.x

        tile_k_per_wg = TILE_K * splitk_waves

        a_tensor = fx.rocdl.make_buffer_tensor(arg_p_input, max_size=False)
        b_tensor = fx.rocdl.make_buffer_tensor(arg_p_weight, max_size=False)
        # shape: [n_in_tile, k_in_tile, k_tile]
        b_tile = fx.flat_divide(b_tensor, fx.make_tile(TILE_N, tile_k_per_wg))[
            None, None, blk_n, None
        ]
        a_cp_atom_r = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), arg_p_input.dtype)
        b_cp_atom_r = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), weight_dtype)

        # tiled copy is created based on the tiled_mma, so the tiled_mma should be same size for tiled copy
        rep_k_per_lane = 4 if const_expr(weight_dtype != fx.BFloat16) else 2
        k_perm = fx.make_tile(
            None,
            None,
            fx.make_layout(
                (4, 4 * splitk_waves, rep_k_per_lane), (1, 4 * rep_k_per_lane, 4)
            ),
        )
        tiled_mma = fx.make_tiled_mma(
            fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, fx.BFloat16)),
            # splitk for gateup/down
            fx.make_layout((1, 1, splitk_waves), (0, 0, 1)),
            k_perm,
        )
        if const_expr(a_with_index):
            cp_atom_lds = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Int32)
            tiled_copy_sortid_lds = fx.make_tiled_copy(
                cp_atom_lds,
                fx.make_layout(((16, 4 * splitk_waves), 1), ((1, 0), 0)),
                fx.make_tile(16),
            )
            a_tensor_thr = TensorWithIndex(
                a_tensor,
                TILE_M,
                tile_k_per_wg,
                tiled_copy_sortid_lds,
                fx.make_tiled_copy_A(a_cp_atom_r, tiled_mma),
                tid,
                lds.sorted_lds,
            )
            a_fake_tensor = fx.make_view(
                fx.get_iter(arg_p_input),
                fx.make_layout((TILE_M, tile_k_per_wg), (tile_k_per_wg, 1)),
            )
            a_frag = tiled_mma.make_fragment_A(a_fake_tensor)
        else:
            a_tile = fx.flat_divide(a_tensor, fx.make_tile(TILE_M, tile_k_per_wg))[
                None, None, 0, None
            ]
            a_tiled_thr = fx.make_tiled_copy_A(a_cp_atom_r, tiled_mma).get_slice(tid)
            a_tensor_thr = a_tiled_thr.partition_S(a_tile)
            a_frag = tiled_mma.make_fragment_A(a_tile[None, None, 0])

        a_frag_retile = (
            fx.make_tiled_copy_A(a_cp_atom_r, tiled_mma).get_slice(tid).retile(a_frag)
        )

        if const_expr(weight_dtype == fx.BFloat16):
            b_tiled_thr = fx.make_tiled_copy_B(b_cp_atom_r, tiled_mma).get_slice(tid)
            b_tensor_thr = b_tiled_thr.partition_S(b_tile)
            b_frag = tiled_mma.make_fragment_B(b_tile[None, None, 0])
            b_frag_retile = b_tiled_thr.retile(b_frag)
        else:
            # b_frag will be decompressed from fp8
            b_fake_tensor = fx.make_view(
                fx.get_iter(arg_p_input),
                fx.make_layout((TILE_N, tile_k_per_wg), (tile_k_per_wg, 1)),
            )
            b_frag = tiled_mma.make_fragment_B(b_fake_tensor)

            tile_size = tiled_mma.tile_size_mnk
            tile_mn = fx.make_tile(
                fx.make_layout(fx.select(tile_size, [1]), 1),
                fx.make_layout(fx.select(tile_size, [2]), 1),
            )
            b_tiled_thr = fx.make_tiled_copy(
                b_cp_atom_r, tiled_mma.tv_layout_B_tiled, tile_mn
            ).get_slice(tid)
            b_tensor_thr = b_tiled_thr.partition_S(b_tile)
            b_frag_retile = fx.make_fragment_like(
                b_tensor_thr[None, None, None, 0], fx.Uint8
            )

        c_fake_tensor = fx.make_view(
            fx.get_iter(arg_p_input), fx.make_layout((TILE_N, TILE_M), (TILE_M, 1))
        )
        c_frag = tiled_mma.make_fragment_C(c_fake_tensor)
        c_frag.fill(0)
        acc_init = c_frag.load()

        for k, state in range(0, K // TILE_K // splitk_waves, 1, init=[acc_init]):
            c_frag.store(state[0])
            k_i32 = fx.Int32(k)
            if const_expr(a_with_index):
                a_tensor_thr.copy(a_cp_atom_r, k_i32, a_frag_retile)
            else:
                fx.copy(
                    a_cp_atom_r, a_tensor_thr[None, None, None, k_i32], a_frag_retile
                )
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

        if const_expr(TILE_N == 32):
            c_lds = fx.make_view(
                lds.c_reduce_lds.ptr, fx.make_ordered_layout((16 * 4, 32), order=(1, 0))
            )
            cp_atom_lds_w = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Float32)
            c_tiled_lds_w = fx.make_tiled_copy(
                cp_atom_lds_w,
                # (4wave*16)*4
                fx.make_layout(((16, 4, 4), (4, 2)), ((1, 256, 16), (64, 1024))),
                fx.make_tile(16 * 4, 16 * 2),
            )
            c_tensor_thr_lds_w = c_tiled_lds_w.get_slice(tid).partition_D(c_lds)
        else:
            # Reduce across 4 waves. To save lds size, will reuse (16*4)x64 floats for one loop
            swz = fx.SwizzleType.get(3, 3, 3)
            c_lds = fx.make_view(
                lds.c_reduce_lds.ptr,
                fx.make_composed_layout(
                    fx.static(swz), fx.make_ordered_layout((16 * 4, 64), order=(1, 0))
                ),
            )
            cp_atom_lds_w = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Float32)
            c_tiled_lds_w = fx.make_tiled_copy(
                cp_atom_lds_w,
                # (4wave*16)*4
                fx.make_layout(((16, 4, 4), (4, 4)), ((1, 256, 16), (64, 1024))),
                fx.make_tile(16 * 4, 16 * 4),
            )
            c_tensor_thr_lds_w = c_tiled_lds_w.get_slice(tid).partition_D(c_lds)

        if const_expr(TILE_N == 32):
            cp_atom_lds_r = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32)
            c_tiled_lds_r = fx.make_tiled_copy(
                cp_atom_lds_r,
                # thread mapping: (4wavex4)x16, repeat 4 times in m dimension for reduce
                fx.make_layout(((16, 4, 4), (1, 4)), ((32 * 2, 1, 4), (32, 16))),
                fx.make_tile(16 * 4, 16 * 1),
            )
            tile_sub_n = 16
        elif const_expr(TILE_N == 64):
            cp_atom_lds_r = fx.make_copy_atom(fx.UniversalCopy64b(), fx.Float32)
            c_tiled_lds_r = fx.make_tiled_copy(
                cp_atom_lds_r,
                # thread mapping: (4wavex4)x16, repeat 4 times in m dimension for reduce
                fx.make_layout(((16, 4, 4), (2, 4)), ((64 * 2, 1, 4), (64, 16))),
                fx.make_tile(16 * 4, 16 * 2),
            )
            tile_sub_n = 32
        else:
            cp_atom_lds_r = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Float32)
            c_tiled_lds_r = fx.make_tiled_copy(
                cp_atom_lds_r,
                # thread mapping: (4wavex4)x16, repeat 4 times in m dimension for reduce
                fx.make_layout(((16, 4, 4), (4, 4)), ((256, 1, 4), (64, 16))),
                fx.make_tile(16 * 4, 16 * 4),
            )
            tile_sub_n = 64
        c_tensor_thr_lds_r = c_tiled_lds_r.get_slice(tid).partition_S(c_lds)

        # shape: [(4, 1), rep_m, rep_n]
        c_frag_vec = c_frag.load()
        # shape: [v, rm, rn]
        shape_v = fx.size(fx.get_shape(c_tensor_thr_lds_r)[0][0]).to_py_value()
        read_rep_n = fx.size(fx.get_shape(c_tensor_thr_lds_r)[2]).to_py_value()
        if const_expr(shape_v == 1):
            # TILE_N==32: flat layout (0-stride value mode) to avoid two stride-1 leaves
            c_frag_reduce = fx.make_rmem_tensor(
                fx.make_layout((1, TILE_M // 16, read_rep_n), (0, read_rep_n, 1)),
                fx.Float32,
            )
        else:
            stride_v = 1
            stride_sub_rn = shape_v * stride_v
            stride_rn = stride_sub_rn * (64 // tile_sub_n)
            stride_rm = stride_rn * TILE_N // tile_sub_n
            c_frag_reduce = fx.make_rmem_tensor(
                fx.make_layout(
                    (shape_v, TILE_M // (4 * 4), (64 // tile_sub_n, TILE_N // 64)),
                    (stride_v, stride_rm, (stride_sub_rn, stride_rn)),
                ),
                fx.Float32,
            )
        n_blocks = max(1, TILE_N // 64)
        w_size = fx.size(fx.get_shape(c_tensor_thr_lds_w)).to_py_value()
        for m in range_constexpr(TILE_M // 16):
            for n in range_constexpr(n_blocks):
                items = []
                for i in range_constexpr(w_size):
                    n_idx = n * (w_size // 4) + i // 4
                    idx = fx.get_scalar(fx.crd2idx((i % 4, m, n_idx), c_frag.layout))
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

                if const_expr(shape_v == 1):
                    c_frag_reduce[0, m, None].store(acc)
                else:
                    c_frag_reduce[None, m, (None, n)].store(acc)
                gpu.barrier()

        return c_frag_reduce

    gemm_splitk = ASTRewriter.transform(_gemm_splitk)

    def _gemm_2x2(
        TILE_M,
        TILE_N,
        TILE_K,
        blk_n: int,  # block index for N dimension (in units of TILE_N)
        arg_p_input: fx.Tensor,  # [M, K]; A rows are gathered via lds.sorted_lds
        arg_p_weight: fx.Tensor,  # preshuffle layout with group_layout_silu composed
        lds,  # SharedStorage with sorted_lds, at_lds, ab_lds, bl_lds, br_lds
    ):
        """4-wave 2x2 tiled GEMM without reduce. Each wave computes one quadrant."""
        tid = gpu.thread_idx.x

        a_tensor = fx.rocdl.make_buffer_tensor(arg_p_input, max_size=False)
        b_tensor = fx.rocdl.make_buffer_tensor(arg_p_weight, max_size=False)

        # Split A into top/bottom halves, B into left/right halves.
        # M is runtime; size the A staging tiles from a static (TILE_M, K) fake so
        # flat_divide stays static (the real rows are gathered via TensorWithIndex below).
        a_size_buf = fx.rocdl.make_buffer_tensor(
            fx.make_view(
                fx.get_iter(arg_p_input), fx.make_layout((TILE_M, K), (K, 1))
            ),
            max_size=False,
        )
        # shape: [m_in_tile, k_in_tile, k_tile]
        at_tile = fx.flat_divide(a_size_buf, fx.make_tile(TILE_M // 2, TILE_K))[None, None, 0, None]
        ab_tile = fx.flat_divide(a_size_buf, fx.make_tile(TILE_M // 2, TILE_K))[None, None, 1, None]
        # shape: [n_in_tile, k_in_tile, k_tile]
        bl_tile = fx.flat_divide(b_tensor, fx.make_tile(TILE_N // 2, TILE_K))[None, None, blk_n * 2 + 0, None]
        br_tile = fx.flat_divide(b_tensor, fx.make_tile(TILE_N // 2, TILE_K))[None, None, blk_n * 2 + 1, None]

        # memory->register copy atoms + g2r thread layout. Each thread issues one 128-bit
        # (16-byte) buffer_load: bf16 -> 8 bf16 over a 32x64 sub-tile; native fp8 -> 16 fp8
        # over a 32x128 sub-tile (TILE_K widened to 128 for fp8). 256 threads x 16 bytes
        # tile the sub-tile in both cases; the fp8 layout doubles value (8->16) and the
        # K-group stride (256->512) vs bf16.
        if const_expr(weight_dtype == fx.BFloat16):
            buf_cp_atom_r = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16)
            g2r_tv_layout = fx.make_layout(((8, 8, 4), 8), ((256, 1, 8), 32))
        else:
            buf_cp_atom_r = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), weight_dtype)
            g2r_tv_layout = fx.make_layout(((8, 8, 4), 16), ((512, 1, 8), 32))
        b_cp_atom_r = buf_cp_atom_r
        # thread layout: [8 * 4 wave, value]
        ab_mem_cp_layout_g2r = fx.make_tiled_copy(
            buf_cp_atom_r,
            g2r_tv_layout,
            fx.make_tile(8 * 4, TILE_K),
        )
        b_mem_cp_layout_g2r = fx.make_tiled_copy(
            b_cp_atom_r,
            g2r_tv_layout,
            fx.make_tile(8 * 4, TILE_K),
        )
        at_mem_tensor_thr = ab_mem_cp_layout_g2r.get_slice(tid).partition_S(at_tile)
        ab_mem_tensor_thr = ab_mem_cp_layout_g2r.get_slice(tid).partition_S(ab_tile)
        bl_mem_tensor_thr = b_mem_cp_layout_g2r.get_slice(tid).partition_S(bl_tile)
        br_mem_tensor_thr = b_mem_cp_layout_g2r.get_slice(tid).partition_S(br_tile)
        at_cp_frag = fx.make_fragment_like(at_mem_tensor_thr[None, None, None, 0])
        ab_cp_frag = fx.make_fragment_like(ab_mem_tensor_thr[None, None, None, 0])
        bl_cp_frag = fx.make_fragment_like(bl_mem_tensor_thr[None, None, None, 0])
        br_cp_frag = fx.make_fragment_like(br_mem_tensor_thr[None, None, None, 0])


        # Gather A rows via sorted_lds (top half -> at, bottom half -> ab). The index copy
        # mirrors ab_mem_cp_layout_g2r's M-mapping: M-row = (lane//8) + 8*wave, replicated
        # across the 8 K-lanes (lane%8); rep_m at M-stride 32.
        cp_atom_sortid_a = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Int32)
        tiled_copy_sortid_a = fx.make_tiled_copy(
            cp_atom_sortid_a,
            fx.make_layout(((8, 8, 4), 1), ((0, 1, 8), 0)),
            fx.make_tile(32),
        )
        at_idx = TensorWithIndex(
            a_tensor,
            TILE_M // 2,
            TILE_K,
            tiled_copy_sortid_a,
            ab_mem_cp_layout_g2r,
            tid,
            lds.sorted_lds,
            index_size=TILE_M // 2,
            index_offset=0,
        )
        ab_idx = TensorWithIndex(
            a_tensor,
            TILE_M // 2,
            TILE_K,
            tiled_copy_sortid_a,
            ab_mem_cp_layout_g2r,
            tid,
            lds.sorted_lds,
            index_size=TILE_M // 2,
            index_offset=TILE_M // 2,
        )

        # sorted_lds is unioned with at_lds: seed all index_frag reads (c_top/c_bot in the
        # caller, plus at_idx/ab_idx above) before any thread overwrites that LDS region with
        # the A-tile in the prefetch below.
        gpu.barrier()

        if const_expr(weight_dtype == fx.BFloat16):
            swz = fx.SwizzleType.get(3, 3, 3)
        else:
            swz = fx.SwizzleType.get(3, 4, 3)
        at_lds = fx.make_view(lds.gemm.at_lds.ptr, fx.make_composed_layout(fx.static(swz), fx.make_ordered_layout((TILE_M // 2, TILE_K), order=(1, 0))))
        ab_lds = fx.make_view(lds.gemm.ab_lds.ptr, fx.make_composed_layout(fx.static(swz), fx.make_ordered_layout((TILE_M // 2, TILE_K), order=(1, 0))))
        bl_lds = fx.make_view(lds.gemm.bl_lds.ptr, fx.make_composed_layout(fx.static(swz), fx.make_ordered_layout((TILE_N // 2, TILE_K), order=(1, 0))))
        br_lds = fx.make_view(lds.gemm.br_lds.ptr, fx.make_composed_layout(fx.static(swz), fx.make_ordered_layout((TILE_N // 2, TILE_K), order=(1, 0))))

        uni_cp_atom_r = fx.make_copy_atom(fx.UniversalCopy128b(), weight_dtype)
        uni_cp_atom_w = fx.make_copy_atom(fx.UniversalCopy128b(), weight_dtype)
        ab_lds_cp_layout_r2s = fx.make_tiled_copy(
            uni_cp_atom_w,
            g2r_tv_layout,
            fx.make_tile(8 * 4, TILE_K),
        )
        at_lds_tensor_thr_w = ab_lds_cp_layout_r2s.get_slice(tid).partition_D(at_lds)
        ab_lds_tensor_thr_w = ab_lds_cp_layout_r2s.get_slice(tid).partition_D(ab_lds)
        bl_lds_tensor_thr_w = ab_lds_cp_layout_r2s.get_slice(tid).partition_D(bl_lds)
        br_lds_tensor_thr_w = ab_lds_cp_layout_r2s.get_slice(tid).partition_D(br_lds)

        at_cp_frag_retile = ab_lds_cp_layout_r2s.get_slice(tid).retile(at_cp_frag)
        ab_cp_frag_retile = ab_lds_cp_layout_r2s.get_slice(tid).retile(ab_cp_frag)
        # B LDS write reads the weight staging fragment directly (native fp8, no decompress).
        bl_cp_frag_retile = ab_lds_cp_layout_r2s.get_slice(tid).retile(bl_cp_frag)
        br_cp_frag_retile = ab_lds_cp_layout_r2s.get_slice(tid).retile(br_cp_frag)

        # LDS->register layout (2x2 wave tiled MMA). bf16 = MFMA(16,16,16); native fp8 =
        # MFMA(16,16,32) with the gfx942 fp8 K-permutation (matches preshuffle_gemm_v2).
        if const_expr(weight_dtype == fx.BFloat16):
            mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, fx.BFloat16))
            k_perm = fx.make_layout((4, 4, 2), (1, 8, 4))
        else:
            mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, weight_dtype))
            k_perm = fx.make_layout((8, 4, 2), (1, 16, 8))
        tiled_mma = fx.make_tiled_mma(
            mma_atom,
            # 2x2 wave layout
            fx.make_layout((2, 2, 1), (1, 2, 0)),
            # K-permutation
            fx.make_tile(None, None, k_perm),
        )

        at_lds_tensor_thr_r = fx.make_tiled_copy_A(uni_cp_atom_r, tiled_mma).get_slice(tid).partition_S(at_lds)
        ab_lds_tensor_thr_r = fx.make_tiled_copy_A(uni_cp_atom_r, tiled_mma).get_slice(tid).partition_S(ab_lds)
        bl_lds_tensor_thr_r = fx.make_tiled_copy_B(uni_cp_atom_r, tiled_mma).get_slice(tid).partition_S(bl_lds)
        br_lds_tensor_thr_r = fx.make_tiled_copy_B(uni_cp_atom_r, tiled_mma).get_slice(tid).partition_S(br_lds)

        at_frag = tiled_mma.make_fragment_A(at_lds)
        ab_frag = tiled_mma.make_fragment_A(ab_lds)
        bl_frag = tiled_mma.make_fragment_B(bl_lds)
        br_frag = tiled_mma.make_fragment_B(br_lds)

        at_frag_retile = fx.make_tiled_copy_A(uni_cp_atom_r, tiled_mma).get_slice(tid).retile(at_frag)
        ab_frag_retile = fx.make_tiled_copy_A(uni_cp_atom_r, tiled_mma).get_slice(tid).retile(ab_frag)
        bl_frag_retile = fx.make_tiled_copy_B(uni_cp_atom_r, tiled_mma).get_slice(tid).retile(bl_frag)
        br_frag_retile = fx.make_tiled_copy_B(uni_cp_atom_r, tiled_mma).get_slice(tid).retile(br_frag)

        # C fragments (one per quadrant). Mirror dense gemm_2x2: make_fragment_C on
        # an (M//2, N//2) tile from flat_divide, then select [0,2,1] -> [v, m, n].
        c_fake_buf = fx.rocdl.make_buffer_tensor(
            fx.make_view(
                fx.get_iter(arg_p_input),
                fx.make_layout((TILE_M, TILE_N), (TILE_N, 1)),
            ),
            max_size=False,
        )
        c_fake = fx.flat_divide(c_fake_buf, fx.make_tile(TILE_M // 2, TILE_N // 2))[
            None, None, 0, 0
        ]
        c_tl_frag = select(tiled_mma.make_fragment_C(c_fake), [0, 2, 1])
        c_tr_frag = select(tiled_mma.make_fragment_C(c_fake), [0, 2, 1])
        c_bl_frag = select(tiled_mma.make_fragment_C(c_fake), [0, 2, 1])
        c_br_frag = select(tiled_mma.make_fragment_C(c_fake), [0, 2, 1])

        c_tl_frag.fill(0)
        c_tr_frag.fill(0)
        c_bl_frag.fill(0)
        c_br_frag.fill(0)

        acc_init = [c_tl_frag.load(), c_tr_frag.load(), c_bl_frag.load(), c_br_frag.load()]

        # Prefetch iteration 0: global->register, register->LDS
        fx.copy(b_cp_atom_r, bl_mem_tensor_thr[None, None, None, 0], bl_cp_frag)
        at_idx.copy(buf_cp_atom_r, 0, at_cp_frag)
        ab_idx.copy(buf_cp_atom_r, 0, ab_cp_frag)
        fx.copy(b_cp_atom_r, br_mem_tensor_thr[None, None, None, 0], br_cp_frag)
        fx.copy(uni_cp_atom_w, bl_cp_frag_retile, bl_lds_tensor_thr_w)
        fx.copy(uni_cp_atom_w, at_cp_frag_retile, at_lds_tensor_thr_w)
        fx.copy(uni_cp_atom_w, ab_cp_frag_retile, ab_lds_tensor_thr_w)
        fx.copy(uni_cp_atom_w, br_cp_frag_retile, br_lds_tensor_thr_w)

        # Prefetch iteration 1: global->register
        fx.copy(b_cp_atom_r, bl_mem_tensor_thr[None, None, None, 1], bl_cp_frag)
        rocdl.sched_barrier(0)
        at_idx.copy(buf_cp_atom_r, 1, at_cp_frag)
        rocdl.sched_barrier(0)
        ab_idx.copy(buf_cp_atom_r, 1, ab_cp_frag)
        rocdl.sched_barrier(0)
        fx.copy(b_cp_atom_r, br_mem_tensor_thr[None, None, None, 1], br_cp_frag)
        rocdl.sched_barrier(0)
        gpu.barrier()

        # LDS read for first GEMM
        fx.copy(uni_cp_atom_r, bl_lds_tensor_thr_r, bl_frag_retile)
        fx.copy(uni_cp_atom_r, at_lds_tensor_thr_r, at_frag_retile)
        rocdl.s_waitcnt(_encode_waitcnt(lgkmcnt=0))

        mem_b_half_cnt = bl_cp_frag.load().numel * weight_dtype.width // 8 // 16
        mem_a_half_cnt = at_cp_frag.load().numel * weight_dtype.width // 8 // 16
        lds_b_half_cnt = bl_frag.load().numel * weight_dtype.width // 8 // 16
        lds_a_half_cnt = at_frag.load().numel * weight_dtype.width // 8 // 16

        # MFMA K per instruction: bf16 = 16, native fp8 = 32.
        k_per_mma = 16 if const_expr(weight_dtype == fx.BFloat16) else 32

        def hot_loop_scheduler(vmem_cnt, dsrd_cnt):
            mfma_cnt = (TILE_M // 2 // 2 // 16) * (TILE_N // 2 // 2 // 16) * (TILE_K // k_per_mma)
            dswr_cnt = vmem_cnt
            for _ in range_constexpr(dswr_cnt):
                rocdl.sched_dswr(1)
                rocdl.sched_mfma(3)
                rocdl.sched_vmem(1)
            for _ in range_constexpr(dsrd_cnt):
                rocdl.sched_dsrd(1)
                rocdl.sched_mfma(1)
            rocdl.sched_mfma(mfma_cnt - dsrd_cnt - dswr_cnt * 3)

        for k, state in range(0, K // TILE_K, 1, init=acc_init):
            c_tl_frag.store(state[0])
            c_tr_frag.store(state[1])
            c_bl_frag.store(state[2])
            c_br_frag.store(state[3])
            k_i32 = fx.Int32(k)

            # bl @ at -> c_tl
            fx.gemm(tiled_mma, c_tl_frag, bl_frag, at_frag, c_tl_frag)
            rocdl.s_waitcnt(_encode_waitcnt(vmcnt=mem_a_half_cnt + mem_a_half_cnt + mem_b_half_cnt))
            fx.copy(uni_cp_atom_w, bl_cp_frag_retile, bl_lds_tensor_thr_w)
            fx.copy(b_cp_atom_r, bl_mem_tensor_thr[None, None, None, k_i32 + 2], bl_cp_frag)
            gpu.barrier()
            fx.copy(uni_cp_atom_r, ab_lds_tensor_thr_r, ab_frag_retile)
            hot_loop_scheduler(mem_b_half_cnt, lds_a_half_cnt)
            rocdl.sched_barrier(0)

            # bl @ ab -> c_bl
            fx.gemm(tiled_mma, c_bl_frag, bl_frag, ab_frag, c_bl_frag)
            rocdl.s_waitcnt(_encode_waitcnt(vmcnt=mem_a_half_cnt + mem_b_half_cnt + mem_b_half_cnt))
            fx.copy(uni_cp_atom_w, at_cp_frag_retile, at_lds_tensor_thr_w)
            at_idx.copy(buf_cp_atom_r, k_i32 + 2, at_cp_frag)
            gpu.barrier()
            fx.copy(uni_cp_atom_r, br_lds_tensor_thr_r, br_frag_retile)
            hot_loop_scheduler(mem_a_half_cnt, lds_b_half_cnt)
            rocdl.sched_barrier(1)

            # br @ at -> c_tr
            fx.gemm(tiled_mma, c_tr_frag, br_frag, at_frag, c_tr_frag)
            rocdl.s_waitcnt(_encode_waitcnt(vmcnt=mem_b_half_cnt + mem_b_half_cnt + mem_a_half_cnt))
            fx.copy(uni_cp_atom_w, ab_cp_frag_retile, ab_lds_tensor_thr_w)
            ab_idx.copy(buf_cp_atom_r, k_i32 + 2, ab_cp_frag)
            gpu.barrier()
            fx.copy(uni_cp_atom_r, bl_lds_tensor_thr_r, bl_frag_retile)
            hot_loop_scheduler(mem_a_half_cnt, lds_b_half_cnt)
            rocdl.sched_barrier(2)

            # br @ ab -> c_br
            fx.gemm(tiled_mma, c_br_frag, br_frag, ab_frag, c_br_frag)
            rocdl.s_waitcnt(_encode_waitcnt(vmcnt=mem_b_half_cnt + mem_a_half_cnt + mem_a_half_cnt))
            fx.copy(uni_cp_atom_w, br_cp_frag_retile, br_lds_tensor_thr_w)
            fx.copy(b_cp_atom_r, br_mem_tensor_thr[None, None, None, k_i32 + 2], br_cp_frag)
            gpu.barrier()
            fx.copy(uni_cp_atom_r, at_lds_tensor_thr_r, at_frag_retile)
            hot_loop_scheduler(mem_b_half_cnt, lds_a_half_cnt)
            rocdl.sched_barrier(3)

            results = yield [c_tl_frag.load(), c_tr_frag.load(), c_bl_frag.load(), c_br_frag.load()]

        c_tl_frag.store(results[0])
        c_tr_frag.store(results[1])
        c_bl_frag.store(results[2])
        c_br_frag.store(results[3])

        # [v, n, m] -> [v, m, n]
        c_tl_frag = select(c_tl_frag, [0, 2, 1])
        c_tr_frag = select(c_tr_frag, [0, 2, 1])
        c_bl_frag = select(c_bl_frag, [0, 2, 1])
        c_br_frag = select(c_br_frag, [0, 2, 1])
        return c_tl_frag, c_tr_frag, c_bl_frag, c_br_frag

    gemm_2x2 = ASTRewriter.transform(_gemm_2x2)

    def _gemm_1x4(
        TILE_M,
        TILE_N,
        TILE_K,
        blk_n: int,  # block index for N dimension (in units of TILE_N)
        arg_p_input: fx.Tensor,  # [M, K]; A rows are gathered via lds.sorted_lds
        arg_p_weight: fx.Tensor,  # preshuffle layout with group_layout_silu composed
        lds,  # SharedStorage with sorted_lds, a_ping, a_pong
    ):
        """1x4 tiled GEMM: the 4 waves tile N(channel); the full TILE_M is shared across
        all waves (no M-split). Each wave owns contiguous_n//4 output channels of BOTH the
        gate and the up projection (two C fragments) so silu stays wave-internal. A
        (activation) is gathered via sorted_lds and staged through an LDS ping-pong
        (a_ping/a_pong); B (weight gate/up) loads direct global->register (no LDS). Pipeline
        mirrors preshuffle_gemm_v2 (A 2-stage LDS ping-pong). B-first MFMA (weight is the
        MFMA M-side) so each C fragment's value dim runs along channel (4 contiguous
        channels/lane), letting the epilogue store 64-bit instead of the A-first 16-bit."""
        tid = gpu.thread_idx.x
        contiguous_n = TILE_N // 2

        a_tensor = fx.rocdl.make_buffer_tensor(arg_p_input, max_size=False)
        b_tensor = fx.rocdl.make_buffer_tensor(arg_p_weight, max_size=False)

        # tiled_mma: B-first (mma_M=channel from weight, mma_N=token from activation); the 4
        # waves tile M(channel) so each wave still owns contiguous_n//4 output channels.
        if const_expr(weight_dtype == fx.BFloat16):
            mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, fx.BFloat16))
            k_perm = fx.make_layout((4, 4, 2), (1, 8, 4))
        else:
            mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, weight_dtype))
            k_perm = fx.make_layout((8, 4, 2), (1, 16, 8))
        tiled_mma = fx.make_tiled_mma(
            mma_atom,
            fx.make_layout((4, 1, 1), (1, 0, 0)),
            fx.make_tile(None, None, k_perm),
        )

        # ---- A (activation): gather + LDS ping-pong ----
        # Static (TILE_M, K) fake keeps flat_divide static; real rows gathered below.
        a_size_buf = fx.rocdl.make_buffer_tensor(
            fx.make_view(fx.get_iter(arg_p_input), fx.make_layout((TILE_M, K), (K, 1))),
            max_size=False,
        )
        a_tile = fx.flat_divide(a_size_buf, fx.make_tile(TILE_M, TILE_K))[None, None, 0, None]
        if const_expr(weight_dtype == fx.BFloat16):
            buf_cp_atom_r = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16)
            # value = contiguous K-elements per thread over the (32, TILE_K) A sub-tile:
            # TILE_K=64 -> 8 bf16 (one ds/buffer 128b op); TILE_K=128 -> 16 bf16 (two 128b
            # ops, rep=2) with sub0 stride widened to 512 so K-blocks step by 16 not 8.
            if const_expr(TILE_K == 128):
                g2r_tv_layout = fx.make_layout(((8, 8, 4), 16), ((512, 1, 8), 32))
            else:
                g2r_tv_layout = fx.make_layout(((8, 8, 4), 8), ((256, 1, 8), 32))
        else:
            buf_cp_atom_r = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), weight_dtype)
            # fp8: contiguous K-elems/thread over the (32, TILE_K) A sub-tile. TILE_K=128 -> 16
            # fp8 (one 128b op); TILE_K=256 -> 32 fp8 (two 128b ops, rep=2) with sub0 stride
            # widened to 1024 so K-blocks step by 32 not 16.
            if const_expr(TILE_K == 256):
                g2r_tv_layout = fx.make_layout(((8, 8, 4), 32), ((1024, 1, 8), 32))
            else:
                g2r_tv_layout = fx.make_layout(((8, 8, 4), 16), ((512, 1, 8), 32))
        a_mem_cp_g2r = fx.make_tiled_copy(buf_cp_atom_r, g2r_tv_layout, fx.make_tile(8 * 4, TILE_K))
        # index copy for A gather: M-row = (lane//8) + 8*wave, replicated across the 8
        # K-lanes; rep_m at M-stride 32 (full TILE_M).
        cp_atom_sortid_a = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Int32)
        tiled_copy_sortid_a = fx.make_tiled_copy(
            cp_atom_sortid_a,
            fx.make_layout(((8, 8, 4), 1), ((0, 1, 8), 0)),
            fx.make_tile(32),
        )
        a_idx = TensorWithIndex(
            a_tensor, TILE_M, TILE_K, tiled_copy_sortid_a, a_mem_cp_g2r, tid,
            lds.sorted_lds, index_size=TILE_M, index_offset=0,
        )
        a_mem_thr = a_mem_cp_g2r.get_slice(tid).partition_S(a_tile)
        a_cp_frag = fx.make_fragment_like(a_mem_thr[None, None, None, 0])

        # sorted_lds is unioned with a_ping: seed all index_frag reads (caller's c_out index
        # + a_idx above) before overwriting that LDS region with the A tile below.
        gpu.barrier()

        if const_expr(weight_dtype == fx.BFloat16):
            swz = fx.SwizzleType.get(3, 3, 3)
        else:
            swz = fx.SwizzleType.get(3, 4, 3)
        a_ping = fx.make_view(lds.gemm.a_ping.ptr, fx.make_composed_layout(fx.static(swz), fx.make_ordered_layout((TILE_M, TILE_K), order=(1, 0))))
        a_pong = fx.make_view(lds.gemm.a_pong.ptr, fx.make_composed_layout(fx.static(swz), fx.make_ordered_layout((TILE_M, TILE_K), order=(1, 0))))

        uni_cp_atom = fx.make_copy_atom(fx.UniversalCopy128b(), weight_dtype)
        # A LDS write (r2s): 128-bit -> ds_write_b128; LDS read below stays 128-bit -> ds_read_b128.
        uni_cp_atom_w = fx.make_copy_atom(fx.UniversalCopy128b(), weight_dtype)
        a_r2s = fx.make_tiled_copy(uni_cp_atom_w, g2r_tv_layout, fx.make_tile(8 * 4, TILE_K))
        a_lds_w = [a_r2s.get_slice(tid).partition_D(a_ping), a_r2s.get_slice(tid).partition_D(a_pong)]
        a_cp_frag_retile = a_r2s.get_slice(tid).retile(a_cp_frag)
        # B-first: activation is the MFMA B-operand (make_fragment_B / make_tiled_copy_B).
        a_lds_r = [
            fx.make_tiled_copy_B(uni_cp_atom, tiled_mma).get_slice(tid).partition_S(a_ping),
            fx.make_tiled_copy_B(uni_cp_atom, tiled_mma).get_slice(tid).partition_S(a_pong),
        ]
        a_frag = tiled_mma.make_fragment_B(a_ping)
        a_frag_retile = fx.make_tiled_copy_B(uni_cp_atom, tiled_mma).get_slice(tid).retile(a_frag)

        # ---- B (weight gate/up): direct global->register (no LDS), 2-stage double buffer ----
        # B-first: weight is the MFMA A-operand (make_fragment_A / make_tiled_copy_A).
        bl_tile = fx.flat_divide(b_tensor, fx.make_tile(contiguous_n, TILE_K))[None, None, blk_n * 2 + 0, None]
        br_tile = fx.flat_divide(b_tensor, fx.make_tile(contiguous_n, TILE_K))[None, None, blk_n * 2 + 1, None]
        b_g2r = fx.make_tiled_copy_A(buf_cp_atom_r, tiled_mma).get_slice(tid)
        bl_g2r = b_g2r.partition_S(bl_tile)
        br_g2r = b_g2r.partition_S(br_tile)
        bl_frag_st = [tiled_mma.make_fragment_A(bl_tile[None, None, 0]), tiled_mma.make_fragment_A(bl_tile[None, None, 0])]
        br_frag_st = [tiled_mma.make_fragment_A(br_tile[None, None, 0]), tiled_mma.make_fragment_A(br_tile[None, None, 0])]
        bl_ret_st = [b_g2r.retile(bl_frag_st[0]), b_g2r.retile(bl_frag_st[1])]
        br_ret_st = [b_g2r.retile(br_frag_st[0]), b_g2r.retile(br_frag_st[1])]

        # ---- C fragments (gate + up), one make_fragment_C each ----
        # B-first: make_fragment_C over the (channel, token) tile; the value dim then runs
        # along channel (4 contiguous channels/lane) for a 64-bit epilogue store.
        c_fake_buf = fx.rocdl.make_buffer_tensor(
            fx.make_view(fx.get_iter(arg_p_input), fx.make_layout((contiguous_n, TILE_M), (TILE_M, 1))),
            max_size=False,
        )
        c_fake = fx.flat_divide(c_fake_buf, fx.make_tile(contiguous_n, TILE_M))[None, None, 0, 0]
        c_gate = tiled_mma.make_fragment_C(c_fake)
        c_up = tiled_mma.make_fragment_C(c_fake)
        c_gate.fill(0)
        c_up.fill(0)

        num_tiles = K // TILE_K

        # ---- instruction-scheduling hints ----
        # 128-bit loads / ds ops per stage; MFMA count for the two gemms.
        k_per_mma = 16 if const_expr(weight_dtype == fx.BFloat16) else 32
        _m_reps = fx.size(fx.get_shape(c_gate)[1]).to_py_value()
        _n_reps = fx.size(fx.get_shape(c_gate)[2]).to_py_value()
        mfma_per_gemm = _m_reps * _n_reps * (TILE_K // k_per_mma)
        mem_a_cnt = a_cp_frag.load().numel * weight_dtype.width // 8 // 16
        mem_b_cnt = bl_frag_st[0].load().numel * weight_dtype.width // 8 // 16
        # per-ki interleave: k_perm groups 2 MFMA-K atoms, so k_iters = TILE_K / (2*k_per_mma).
        # fragment K dim is (2 atoms, k_iters) -> gemm coord = (None, ki); the retile/LDS-read
        # views have a flat k_iters dim -> coord = ki. This is what lets TILE_K scale to 128.
        k_iters = TILE_K // (2 * k_per_mma)
        # full A(tile) LDS read (ds_read), done once per stage (cross-stage rotation)
        lds_a_cnt = a_frag.load().numel * weight_dtype.width // 8 // 16

        def hot_loop_scheduler():
            # Fixed interleave: each buffer_load(vmem)+4 mfma; each ds_read(dsrd)+1 mfma;
            # each ds_write(dswr)+2 mfma (dsrd before dswr); then the remaining mfma.
            mfma_cnt = 2 * mfma_per_gemm
            n_vmem = mem_a_cnt + 2 * mem_b_cnt   # A g2r + B gate/up g2r (buffer_load)
            n_dswr = mem_a_cnt                    # A staging -> LDS store (ds_write)
            n_dsrd = lds_a_cnt                    # A LDS -> register full tile (ds_read)
            used = 0
            rocdl.sched_dsrd(2)
            for _ in range_constexpr(n_vmem):
                rocdl.sched_dsrd(1)
                rocdl.sched_vmem(1)
                rocdl.sched_mfma(4)
                used += 4
            for _ in range_constexpr(n_dsrd - n_vmem - 2):
                rocdl.sched_dsrd(1)
                rocdl.sched_mfma(1)
                used += 1
            rocdl.sched_mfma(mfma_cnt - n_dswr * 2 - used)
            for _ in range_constexpr(n_dswr):
                rocdl.sched_dswr(1)
                rocdl.sched_mfma(2)
                used += 2
            if const_expr(mfma_cnt - used > 0):
                rocdl.sched_mfma(mfma_cnt - used)

        def pipeline_stage(read_i, k_next, do_prefetch):
            write_i = read_i ^ 1
            # prefetch next B (gate/up) + A (global -> register)
            if const_expr(do_prefetch):
                a_idx.copy(buf_cp_atom_r, k_next, a_cp_frag)
                fx.copy(buf_cp_atom_r, bl_g2r[None, None, None, k_next], bl_ret_st[write_i])
                fx.copy(buf_cp_atom_r, br_g2r[None, None, None, k_next], br_ret_st[write_i])
            # read this stage's own A tile LDS[read_i] -> a_frag at the head, then compute
            fx.copy(uni_cp_atom, a_lds_r[read_i], a_frag_retile)
            for ki in range_constexpr(k_iters):
                fx.gemm(
                    tiled_mma,
                    c_gate,
                    bl_frag_st[read_i][None, None, (None, ki)],
                    a_frag[None, None, (None, ki)],
                    c_gate,
                )
                fx.gemm(
                    tiled_mma,
                    c_up,
                    br_frag_st[read_i][None, None, (None, ki)],
                    a_frag[None, None, (None, ki)],
                    c_up,
                )
            if const_expr(do_prefetch):
                # A(k_next) staging -> LDS[write] for a later stage's head read
                fx.copy(uni_cp_atom_w, a_cp_frag_retile, a_lds_w[write_i])
                hot_loop_scheduler()
            rocdl.sched_barrier(0)
            gpu.barrier()

        # Prologue: gather A(0) -> LDS[0]; load B(0) -> stage 0.
        a_idx.copy(buf_cp_atom_r, fx.Int32(0), a_cp_frag)
        fx.copy(buf_cp_atom_r, bl_g2r[None, None, None, fx.Int32(0)], bl_ret_st[0])
        fx.copy(buf_cp_atom_r, br_g2r[None, None, None, fx.Int32(0)], br_ret_st[0])
        rocdl.s_waitcnt(_encode_waitcnt(vmcnt=0))
        fx.copy(uni_cp_atom_w, a_cp_frag_retile, a_lds_w[0])
        gpu.barrier()

        acc_init = [c_gate.load(), c_up.load()]
        for iv, state in range(0, num_tiles // 2 - 1, 1, init=acc_init):
            c_gate.store(state[0])
            c_up.store(state[1])
            kb = fx.Int32(iv * 2)
            pipeline_stage(0, kb + 1, True)
            pipeline_stage(1, kb + 2, True)
            results = yield [c_gate.load(), c_up.load()]
        c_gate.store(results[0])
        c_up.store(results[1])
        kb = fx.Int32(num_tiles - 2)
        pipeline_stage(0, kb + 1, True)
        pipeline_stage(1, fx.Int32(0), False)
        return c_gate, c_up

    gemm_1x4 = ASTRewriter.transform(_gemm_1x4)

    @flyc.kernel
    def moe_2stage_gateup(
        p_input: fx.Pointer,  # bf16 [M, K]
        p_weight: fx.Pointer,  # quantized/bf16 [N/16, K/8 * 16 * 8]
        p_output: fx.Pointer,  # bf16 [M, TOPK, N//2]
        # sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids
        p_sorted_ids: fx.Pointer,
        p_sorted_weights: fx.Pointer,
        p_sorted_expert_ids: fx.Pointer,
        p_num_valid_ids: fx.Pointer,
        p_w_scale: fx.Pointer,
        M: fx.Int32,
    ):
        tid = gpu.thread_idx.x
        blk_n = gpu.block_idx.x
        e_idx = gpu.block_idx.y

        arg_p_input = fx.make_view(_as_ptr(p_input), fx.make_layout((M, K), (K, 1)))
        num_valid_buf = fx.make_view(
            fx.recast_iter(fx.Int32, _as_ptr(p_num_valid_ids)), fx.make_layout(1, 1)
        )
        max_valid_id = num_valid_buf[0]
        if const_expr(weight_dtype != fx.BFloat16):
            p_weight = fx.recast_iter(fx.Uint8, _as_ptr(p_weight))
        if e_idx * BLOCK_TILE_SIZE_M < max_valid_id:
            lds = fx.SharedAllocator().allocate(SharedStorage)
            lds.sorted_lds = lds.sorted_lds.peek()
            lds.c_reduce_lds = lds.c_reduce_lds.peek()
            arg_p_sorted_ids = fx.make_view(
                fx.recast_iter(
                    fx.Int32, _as_ptr(p_sorted_ids) + e_idx * BLOCK_TILE_SIZE_M
                ),
                fx.make_layout(BLOCK_TILE_SIZE_M, 1),
            )
            arg_p_sorted_expert_ids = fx.recast_iter(
                fx.Int32, _as_ptr(p_sorted_expert_ids)
            )
            expert_id = arg_p_sorted_expert_ids[e_idx]
            # there is a reduce in gemm_splitk which will read/write from lds, the BLOCK_TILE_SIZE_N will impact the coalesced access:
            # BLOCK_TILE_SIZE_N BLOCK_TILE_SIZE_N//2(after silu) LDS_read_per_lane  MEM_write_per_lane
            # 64                32                               2=(32/16 threads)  2=(32/16 threads)
            # 128               64                               4=(64/16 threads)  4=(64/16 threads)
            # 256: will split into 2x128
            if const_expr(alg == "splitk"):
                contiguous_n = 64 if const_expr(BLOCK_TILE_SIZE_N % 128 == 0) else 32
            else:
                contiguous_n = 64

            group_layout_silu = fx.make_layout(
                ((contiguous_n, 2, N // (contiguous_n * 2)), K),
                ((1, N // 2, contiguous_n), N),
            )
            element_num = 16 // (p_weight.dtype.width // 8)
            arg_p_weight = fx.make_view(
                p_weight + fx.Int64(expert_id * N * K),
                # preshuffle layout: [16, (8, K // 8)]
                fx.composition(
                    fx.make_layout(
                        ((16, N // 16), (element_num, K // element_num)),
                        ((element_num, 16 * K), (1, 16 * element_num)),
                    ),
                    group_layout_silu,
                ),
            )  # NOTE: assume permuted adjacent 32 rows will fall in the same wave to do silu

            # sorted ids: global -> LDS (scalar load/store, only first BLOCK_TILE_SIZE_M threads participate)
            sorted_ids_buf = fx.rocdl.make_buffer_tensor(
                arg_p_sorted_ids, max_size=False
            )
            if tid < BLOCK_TILE_SIZE_M:
                lds_view = fx.make_view(
                    lds.sorted_lds.ptr, fx.make_layout(BLOCK_TILE_SIZE_M, 1)
                )
                # fx.memref_store(val, lds_view, tid)
                lds_view[tid] = sorted_ids_buf[tid]
            gpu.barrier()

            # prepare c_tensor(reuse lds.c_reduce_lds before gemm)
            if const_expr(alg == "splitk"):
                cp_atom_w = fx.make_copy_atom(
                    (
                        fx.rocdl.BufferCopy64b()
                        if const_expr(BLOCK_TILE_SIZE_N % 128 == 0)
                        else fx.rocdl.BufferCopy32b()
                    ),
                    fx.BFloat16,
                )
                c_tiled_g = fx.make_tiled_copy(
                    cp_atom_w,
                    # thread mapping: 4 wavex(4x16), (contiguous_n // 16) elements per lane
                    fx.make_layout(
                        ((16, 4, 4), contiguous_n // 16), ((contiguous_n, 1, 4), 16)
                    ),
                    fx.make_tile(16, contiguous_n),
                )
            arg_p_output = fx.make_view(
                _as_ptr(p_output),
                fx.make_layout((M, TOPK, N // 2), (TOPK * N // 2, N // 2, 1)),
            )
            out_tensor = fx.rocdl.make_buffer_tensor(
                arg_p_output,
                max_size=False,
                num_records_bytes=M * TOPK * N // 2 * fx.BFloat16.width // 8,
            )
            tiled_copy_sortid_lds = fx.make_tiled_copy(
                fx.make_copy_atom(fx.UniversalCopy32b(), fx.Int32),
                fx.make_layout(((16, 16), 1), ((0, 1), 0)),
                fx.make_tile(16),
            )
            c_tensor = TensorWithIndex(
                out_tensor,
                BLOCK_TILE_SIZE_M,
                BLOCK_TILE_SIZE_N // 2,
                tiled_copy_sortid_lds,
                c_tiled_g,
                tid,
                lds.sorted_lds,
                is_read_from_mem=False,
            )

            c_frag = gemm_splitk(
                BLOCK_TILE_SIZE_M,
                BLOCK_TILE_SIZE_N,
                TILE_K,
                blk_n,
                arg_p_input,
                arg_p_weight,
                lds,
                splitk_waves=4,
            )

            c_frag_bf16 = _apply_scale_silu_bf16(
                c_frag, tid, expert_id, blk_n, contiguous_n, p_w_scale
            )

            c_tensor.copy(
                cp_atom_w, blk_n, c_tiled_g.get_slice(tid).retile(c_frag_bf16)
            )

    @flyc.kernel
    def moe_2stage_down(
        p_input: fx.Pointer,  # bf16 [M, TOPK, K]
        p_weight: fx.Pointer,  # quantized/bf16 [N/16, K/8 * 16 * 8]
        p_output: fx.Pointer,  # bf16 [M, N]
        # sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids
        p_sorted_ids: fx.Pointer,
        p_sorted_weights: fx.Pointer,
        p_sorted_expert_ids: fx.Pointer,
        p_num_valid_ids: fx.Pointer,
        p_w_scale: fx.Pointer,
        M: fx.Int32,
    ):
        tid = gpu.thread_idx.x
        blk_n = gpu.block_idx.x
        e_idx = gpu.block_idx.y

        arg_p_input = fx.make_view(
            _as_ptr(p_input), fx.make_layout((M, TOPK, K), (TOPK * K, K, 1))
        )
        num_valid_buf = fx.make_view(
            fx.recast_iter(fx.Int32, _as_ptr(p_num_valid_ids)), fx.make_layout(1, 1)
        )
        max_valid_id = num_valid_buf[0]
        if const_expr(weight_dtype != fx.BFloat16):
            p_weight = fx.recast_iter(fx.Uint8, _as_ptr(p_weight))
        if e_idx * BLOCK_TILE_SIZE_M < max_valid_id:
            lds = fx.SharedAllocator().allocate(SharedStorage).peek()
            arg_p_sorted_ids = fx.make_view(
                fx.recast_iter(
                    fx.Int32, _as_ptr(p_sorted_ids) + e_idx * BLOCK_TILE_SIZE_M
                ),
                fx.make_layout(BLOCK_TILE_SIZE_M, 1),
            )
            arg_p_sorted_expert_ids = fx.recast_iter(
                fx.Int32, _as_ptr(p_sorted_expert_ids)
            )
            expert_id = arg_p_sorted_expert_ids[e_idx]
            element_num = 16 // (p_weight.dtype.width // 8)
            arg_p_weight = fx.make_view(
                p_weight + fx.Int64(expert_id * N * K),
                # preshuffle layout: [16, (8, K // 8)]
                fx.make_layout(
                    ((16, N // 16), (element_num, K // element_num)),
                    ((element_num, 16 * K), (1, 16 * element_num)),
                ),
            )

            # sorted ids: global -> LDS (scalar load/store, only first BLOCK_TILE_SIZE_M threads participate)
            sorted_ids_buf = fx.rocdl.make_buffer_tensor(
                arg_p_sorted_ids, max_size=False
            )
            lds_view = fx.make_view(
                lds.sorted_lds.ptr, fx.make_layout(BLOCK_TILE_SIZE_M, 1)
            )
            for idx in range(tid, BLOCK_TILE_SIZE_M, 64):
                # fx.memref_store(val, lds_view, tid)
                lds_view[idx] = sorted_ids_buf[idx]
            gpu.barrier()

            cp_atom_weight = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
            arg_p_sorted_weights = fx.make_view(
                fx.recast_iter(
                    fx.Float32, _as_ptr(p_sorted_weights) + e_idx * BLOCK_TILE_SIZE_M
                ),
                fx.make_layout(BLOCK_TILE_SIZE_M, 1),
            )
            sorted_weights_buf = fx.rocdl.make_buffer_tensor(
                arg_p_sorted_weights, max_size=False
            )
            cp_atom_lds = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Int32)
            tiled_copy_sortid_lds = fx.make_tiled_copy(
                cp_atom_lds, fx.make_layout(((16, 4), 1), ((1, 0), 0)), fx.make_tile(16)
            )
            sorted_weights_tensor = tiled_copy_sortid_lds.get_slice(tid).partition_S(
                sorted_weights_buf
            )
            sorted_weight_frag = fx.make_fragment_like(
                sorted_weights_tensor, fx.Float32
            )
            fx.copy(cp_atom_weight, sorted_weights_tensor, sorted_weight_frag)

            c_frag = gemm_splitk(
                BLOCK_TILE_SIZE_M,
                BLOCK_TILE_SIZE_N,
                TILE_K,
                blk_n,
                arg_p_input,
                arg_p_weight,
                lds,
                splitk_waves=1,
            )

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
            if const_expr(not USE_ATOMIC_WRITE):  # gateup output shape: [M, TOPK, N]
                arg_p_output = fx.make_view(
                    _as_ptr(p_output), fx.make_layout((M, TOPK, N), (TOPK * N, N, 1))
                )
                arg_p_output = fx.rocdl.make_buffer_tensor(
                    arg_p_output,
                    max_size=False,
                    num_records_bytes=M * TOPK * N * fx.BFloat16.width // 8,
                )
                cp_atom_w = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.BFloat16)
                is_atomic_write = False
            else:
                arg_p_output = fx.make_view(
                    _as_ptr(p_output), fx.make_layout((M, N), (N, 1))
                )
                # arg_p_output = fx.rocdl.make_buffer_tensor(arg_p_output, max_size=False, num_records_bytes=M * TOPK * N * fx.BFloat16.width // 8)
                # cp_atom_w = fx.make_copy_atom(fx.rocdl.BufferAtomicAdd(fx.BFloat16), fx.BFloat16)
                cp_atom_w = fx.make_copy_atom(
                    fx.UniversalAtomic(fx.AtomicOp.Add, fx.BFloat16), fx.BFloat16
                )
                is_atomic_write = True
            c_tiled_g = fx.make_tiled_copy(
                cp_atom_w,
                # 16x4 threads, each writes 4 points in N dimension
                fx.make_layout(((16, 4), 4), ((1, 64), 16)),
                fx.make_tile(16, 16),
            )
            c_tensor = TensorWithIndex(
                arg_p_output,
                BLOCK_TILE_SIZE_M,
                BLOCK_TILE_SIZE_N,
                tiled_copy_sortid_lds,
                c_tiled_g,
                tid,
                lds.sorted_lds,
                is_read_from_mem=False,
                TOPK=TOPK,
                is_atomic_write=is_atomic_write,
            )
            c_tensor.copy(
                cp_atom_w, blk_n, c_tiled_g.get_slice(tid).retile(c_frag_bf16)
            )

    @flyc.kernel
    def moe_2stage_gateup_batch1(
        p_input: fx.Pointer,  # bf16 [M, K]
        p_weight: fx.Pointer,  # quantized/bf16 [N/16, K/8 * 16 * 8]
        p_output: fx.Pointer,  # bf16 [M, TOPK, N//2]
        p_topk_ids: fx.Pointer,  # int32 [M, TOPK]
        p_w_scale: fx.Pointer,
    ):
        tid = gpu.thread_idx.x
        blk_n = gpu.block_idx.x
        e_idx = gpu.block_idx.y

        arg_p_input = fx.make_view(_as_ptr(p_input), fx.make_layout((1, K), (K, 1)))
        if const_expr(weight_dtype != fx.BFloat16):
            p_weight = fx.recast_iter(fx.Uint8, _as_ptr(p_weight))
        arg_p_expert_ids = fx.recast_iter(fx.Int32, _as_ptr(p_topk_ids))
        expert_id = arg_p_expert_ids[e_idx]
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        # gate/up group width. BN==32 keeps the 4-wave split-K but uses a dedicated reduce below;
        # BN>=64 uses the coalesced reduce in gemm_splitk.
        contiguous_n = min(64, BLOCK_TILE_SIZE_N // 2)

        group_layout_silu = fx.make_layout(
            ((contiguous_n, 2, N // (contiguous_n * 2)), K),
            ((1, N // 2, contiguous_n), N),
        )
        element_num = 16 // (p_weight.dtype.width // 8)
        arg_p_weight = fx.make_view(
            p_weight + fx.Int64(expert_id * N * K),
            # preshuffle layout: [16, (8, K // 8)]
            fx.composition(
                fx.make_layout(
                    ((16, N // 16), (element_num, K // element_num)),
                    ((element_num, 16 * K), (1, 16 * element_num)),
                ),
                group_layout_silu,
            ),
        )  # NOTE: assume permuted adjacent 32 rows will fall in the same wave to do silu

        c_frag = gemm_splitk(
            BLOCK_TILE_SIZE_M,
            BLOCK_TILE_SIZE_N,
            TILE_K,
            blk_n,
            arg_p_input,
            arg_p_weight,
            lds,
            splitk_waves=4,
            a_with_index=False,
        )

        c_frag_bf16 = _apply_scale_silu_bf16(
            c_frag, tid, expert_id, blk_n, contiguous_n, p_w_scale
        )

        arg_p_output = fx.make_view(
            _as_ptr(p_output),
            fx.make_layout((1, TOPK, N // 2), (TOPK * N // 2, N // 2, 1)),
        )
        out_tensor = fx.rocdl.make_buffer_tensor(
            arg_p_output,
            max_size=False,
            num_records_bytes=1 * TOPK * N // 2 * fx.BFloat16.width // 8,
        )
        cp_atom_w = fx.make_copy_atom(
            (
                fx.rocdl.BufferCopy64b()
                if const_expr(BLOCK_TILE_SIZE_N % 128 == 0)
                else (
                    fx.rocdl.BufferCopy32b()
                    if const_expr(BLOCK_TILE_SIZE_N >= 64)
                    else fx.rocdl.BufferCopy16b()
                )
            ),
            fx.BFloat16,
        )
        c_tiled_g = fx.make_tiled_copy(
            cp_atom_w,
            # thread mapping: 4 wavex(4x16), (contiguous_n // 16) elements per lane
            fx.make_layout(
                ((16, 4, 4), max(1, contiguous_n // 16)),
                ((max(1, contiguous_n), 1, 4), 16),
            ),
            fx.make_tile(16, max(16, contiguous_n)),
        )
        c_tile = fx.flat_divide(
            out_tensor[None, e_idx, None],
            fx.make_tile(BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N // 2),
        )[None, None, None, blk_n]
        c_dst = c_tiled_g.get_slice(tid).partition_S(c_tile)
        c_src = c_tiled_g.get_slice(tid).retile(c_frag_bf16)

        fx.copy(cp_atom_w, c_src, c_dst[None, None, None, 0])

    @flyc.kernel
    def moe_2stage_gateup_prefill_2x2(
        p_input: fx.Pointer,  # bf16 or native-fp8 [M, K]
        p_weight: fx.Pointer,  # bf16/fp8 [N/16, K/8 * 16 * 8]
        p_output: fx.Pointer,  # bf16 [M, TOPK, N//2]
        p_sorted_ids: fx.Pointer,
        p_sorted_weights: fx.Pointer,
        p_sorted_expert_ids: fx.Pointer,
        p_num_valid_ids: fx.Pointer,
        p_w_scale: fx.Pointer,  # weight fp8 scale (per-output-channel ptpc / per-tensor)
        p_a_scale: fx.Pointer,  # input fp8 scale (per-token ptpc / per-tensor)
        M: fx.Int32,
    ):
        tid = gpu.thread_idx.x
        blk_n = gpu.block_idx.x
        e_idx = gpu.block_idx.y

        # Native fp8: A (input) and B (weight) are both fp8; reinterpret the byte pointers.
        if const_expr(weight_dtype != fx.BFloat16):
            in_ptr = fx.recast_iter(weight_dtype, _as_ptr(p_input))
        else:
            in_ptr = _as_ptr(p_input)
        arg_p_input = fx.make_view(in_ptr, fx.make_layout((M, K), (K, 1)))
        num_valid_buf = fx.make_view(
            fx.recast_iter(fx.Int32, _as_ptr(p_num_valid_ids)), fx.make_layout(1, 1)
        )
        max_valid_id = num_valid_buf[0]
        if const_expr(weight_dtype != fx.BFloat16):
            p_weight = fx.recast_iter(weight_dtype, _as_ptr(p_weight))
        if e_idx * BLOCK_TILE_SIZE_M < max_valid_id:
            lds = fx.SharedAllocator().allocate(SharedStorage)
            lds.sorted_lds = lds.sorted_lds.peek()
            lds.gemm = lds.gemm.peek()
            arg_p_sorted_ids = fx.make_view(
                fx.recast_iter(
                    fx.Int32, _as_ptr(p_sorted_ids) + e_idx * BLOCK_TILE_SIZE_M
                ),
                fx.make_layout(BLOCK_TILE_SIZE_M, 1),
            )
            arg_p_sorted_expert_ids = fx.recast_iter(
                fx.Int32, _as_ptr(p_sorted_expert_ids)
            )
            expert_id = arg_p_sorted_expert_ids[e_idx]

            # BLOCK_TILE_SIZE_N is at least 64, so contiguous_n is 32 then each wave has 16 in the gemm
            contiguous_n = BLOCK_TILE_SIZE_N // 2
            group_layout_silu = fx.make_layout(
                ((contiguous_n, 2, N // (contiguous_n * 2)), K),
                ((1, N // 2, contiguous_n), N),
            )
            element_num = 16 // (p_weight.dtype.width // 8)
            arg_p_weight = fx.make_view(
                p_weight + fx.Int64(expert_id * N * K),
                # preshuffle layout [16, (8, K // 8)] composed with the gate/up silu grouping
                fx.composition(
                    fx.make_layout(
                        ((16, N // 16), (element_num, K // element_num)),
                        ((element_num, 16 * K), (1, 16 * element_num)),
                    ),
                    group_layout_silu,
                ),
            )

            # sorted ids: global -> LDS (only first BLOCK_TILE_SIZE_M threads participate)
            sorted_ids_buf = fx.rocdl.make_buffer_tensor(
                arg_p_sorted_ids, max_size=False
            )
            if tid < BLOCK_TILE_SIZE_M:
                lds_view = fx.make_view(
                    lds.sorted_lds.ptr, fx.make_layout(BLOCK_TILE_SIZE_M, 1)
                )
                lds_view[tid] = sorted_ids_buf[tid]
            gpu.barrier()

            # Output [M, TOPK, N//2] and the per-row sorted indices, built BEFORE gemm_2x2:
            # sorted_lds is unioned with at_lds, so c_top/c_bot must seed their index_frag from
            # sorted_lds now; gemm_2x2 then overwrites that LDS region with the A-tile. The
            # index also serves the ptpc input scale gather (per token = per C M-row) below.
            arg_p_output = fx.make_view(
                _as_ptr(p_output),
                fx.make_layout((M, TOPK, N // 2), (TOPK * N // 2, N // 2, 1)),
            )
            out_tensor = fx.rocdl.make_buffer_tensor(
                arg_p_output,
                max_size=False,
                num_records_bytes=M * TOPK * N // 2 * fx.BFloat16.width // 8,
            )
            buf_atom_w128 = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16)
            # CShuffle read/scatter layout: per wave 16(token=M) x 4(channel=N) lanes,
            # channel-first then token; 8 contiguous channels (128b) per lane; 4 waves 2x2.
            # One tile covers 32 token x 64 channel; partition yields rep_token = (BM//2)//32
            # and rep_channel = contiguous_n//64 (contiguous_n = BN//2, a multiple of 64).
            # tile (d0=32 token, d1=64 channel) is dim0-major (d0 stride 1, d1 stride 32):
            # value=8 channels -> addr 32; chan_thread(4) -> addr 256; token_thread(16) ->
            # addr 1; waveM(2) -> addr 16; waveN(2) -> addr 1024.
            c_rw_copy = fx.make_tiled_copy(
                buf_atom_w128,
                fx.make_layout(((4, 16, 2, 2), 8), ((256, 1, 16, 1024), 32)),
                fx.make_tile(32, 64),
            )
            # index copy: token(M)-row = (lane//4) + 16*waveM, replicated over channel threads;
            # rep_m at token-stride 32.
            c_index_copy = fx.make_tiled_copy(
                fx.make_copy_atom(fx.UniversalCopy32b(), fx.Int32),
                fx.make_layout(((4, 16, 2, 2), 1), ((0, 1, 16, 0), 0)),
                fx.make_tile(32),
            )
            c_top = TensorWithIndex(
                out_tensor,
                BLOCK_TILE_SIZE_M // 2,
                contiguous_n,
                c_index_copy,
                c_rw_copy,
                tid,
                lds.sorted_lds,
                is_read_from_mem=False,
                TOPK=TOPK,
                index_size=BLOCK_TILE_SIZE_M // 2,
                index_offset=0,
            )
            c_bot = TensorWithIndex(
                out_tensor,
                BLOCK_TILE_SIZE_M // 2,
                contiguous_n,
                c_index_copy,
                c_rw_copy,
                tid,
                lds.sorted_lds,
                is_read_from_mem=False,
                TOPK=TOPK,
                index_size=BLOCK_TILE_SIZE_M // 2,
                index_offset=BLOCK_TILE_SIZE_M // 2,
            )

            # a_scale (ptpc) is per-token, folded into the silu per c_*_frag rep_m whose
            # token follows the mma layout (token = lane%16 + 16*waveM), NOT the CShuffle read
            # layout that c_top/c_bot use for the scatter. Read a dedicated mma-mapped index
            # before gemm_2x2 overwrites sorted_lds.
            if const_expr(weight_dtype != fx.BFloat16 and act_quant_type == "ptpc"):
                asc_index_copy = fx.make_tiled_copy(
                    fx.make_copy_atom(fx.UniversalCopy32b(), fx.Int32),
                    fx.make_layout(((16, 4, 2, 2), 1), ((1, 0, 16, 0), 0)),
                    fx.make_tile(32),
                )
                # Read the mma-mapped sorted index directly into fragments (top/bottom
                # M-half) instead of building a TensorWithIndex just for index_frag.
                cp_atom_idx = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Int32)
                asc_lds_top = fx.make_view(
                    lds.sorted_lds.ptr, fx.make_layout(BLOCK_TILE_SIZE_M // 2, 1)
                )
                asc_top_thr = asc_index_copy.get_slice(tid).partition_S(asc_lds_top)
                asc_idx_top = fx.make_fragment_like(asc_top_thr)
                fx.copy(cp_atom_idx, asc_top_thr, asc_idx_top)
                asc_lds_bot = fx.make_view(
                    lds.sorted_lds.ptr + BLOCK_TILE_SIZE_M // 2,
                    fx.make_layout(BLOCK_TILE_SIZE_M // 2, 1),
                )
                asc_bot_thr = asc_index_copy.get_slice(tid).partition_S(asc_lds_bot)
                asc_idx_bot = fx.make_fragment_like(asc_bot_thr)
                fx.copy(cp_atom_idx, asc_bot_thr, asc_idx_bot)

            c_tl_frag, c_tr_frag, c_bl_frag, c_br_frag = gemm_2x2(
                BLOCK_TILE_SIZE_M,
                BLOCK_TILE_SIZE_N,
                TILE_K,
                blk_n,
                arg_p_input,
                arg_p_weight,
                lds,
            )

            m_reps = fx.size(fx.get_shape(c_tl_frag)[1]).to_py_value()

            # Native-fp8 dequant: real C = C_fp8 * a_scale[token] * b_scale[channel].
            # weight_quant_type picks the b_scale form and act_quant_type the a_scale form,
            # independently. Any per_tensor factor is a plain scalar pre-multiply; ptpc factors
            # are folded into the silu read (b_scale per-output-channel gate/up frags; a_scale
            # per token = per C M-row, gathered for the top/bottom M-half).
            gate_scale = None
            up_scale = None
            a_sc_top = None
            a_sc_bot = None
            if const_expr(weight_dtype != fx.BFloat16):
                if const_expr(weight_quant_type == "per_tensor"):
                    scale = fx.make_view(
                        _as_ptr(p_w_scale) + expert_id, fx.make_layout(1, 1)
                    )[0]
                    if const_expr(act_quant_type == "per_tensor"):
                        scale = scale * fx.make_view(
                            fx.recast_iter(fx.Float32, _as_ptr(p_a_scale)),
                            fx.make_layout(1, 1),
                        )[0]
                    c_tl_frag.store(c_tl_frag.load() * scale)
                    c_tr_frag.store(c_tr_frag.load() * scale)
                    c_bl_frag.store(c_bl_frag.load() * scale)
                    c_br_frag.store(c_br_frag.load() * scale)
                if const_expr(weight_quant_type == "ptpc"):
                    scale_gate = fx.make_view(
                        _as_ptr(p_w_scale) + expert_id * N + blk_n * contiguous_n,
                        fx.make_layout(contiguous_n, 1),
                    )
                    scale_up = fx.make_view(
                        _as_ptr(p_w_scale) + expert_id * N + N // 2 + blk_n * contiguous_n,
                        fx.make_layout(contiguous_n, 1),
                    )
                    cp_atom_scale = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32)
                    scale_copy = fx.make_tiled_copy(
                        cp_atom_scale,
                        fx.make_layout(((16, 4, 2, 2), 4), ((0, 4, 0, 16), 1)),
                        fx.make_tile(32),
                    )
                    sg_thr = scale_copy.get_slice(tid).partition_S(scale_gate)
                    su_thr = scale_copy.get_slice(tid).partition_S(scale_up)
                    gate_scale = fx.make_fragment_like(sg_thr)
                    up_scale = fx.make_fragment_like(su_thr)
                    fx.copy(cp_atom_scale, sg_thr, gate_scale)
                    fx.copy(cp_atom_scale, su_thr, up_scale)
                if const_expr(act_quant_type == "ptpc"):
                    a_scale_tensor = fx.rocdl.make_buffer_tensor(
                        fx.make_view(
                            fx.recast_iter(fx.Float32, _as_ptr(p_a_scale)),
                            fx.make_layout(M, 1),
                        ),
                        max_size=False,
                    )
                    a_sc_top = []
                    a_sc_bot = []
                    for m in range_constexpr(m_reps):
                        a_sc_top.append(a_scale_tensor[asc_idx_top[0, m] & 0xFFFFFF])
                        a_sc_bot.append(a_scale_tensor[asc_idx_bot[0, m] & 0xFFFFFF])

            # silu(gate) * up, element-wise. After group_layout_silu, left N-half = gate
            # (c_tl/c_bl), right N-half = up (c_tr/c_br); same N-col -> same output channel.
            # c_tl/c_tr -> top M-half output, c_bl/c_br -> bottom M-half output.
            c_top_bf16 = _silu_pair_bf16(c_tl_frag, c_tr_frag, gate_scale, up_scale, a_sc_top)
            c_bot_bf16 = _silu_pair_bf16(c_bl_frag, c_br_frag, gate_scale, up_scale, a_sc_bot)

            # 128-bit CShuffle epilogue. The mma fragment gives each lane only 4 contiguous
            # channels (64-bit writes). Stage BOTH silu halves into two non-overlapping LDS
            # regions in GemmBuffers, then read them back with 8 contiguous channels per lane
            # so the scatter issues 128-bit writes. Dual-buffer -> 2 barriers instead of 4.
            cshuf_atom_w = fx.make_copy_atom(fx.UniversalCopy64b(), fx.BFloat16)
            cshuf_atom_r = fx.make_copy_atom(fx.UniversalCopy128b(), fx.BFloat16)
            # store layout = c_*_bf16's native fragment layout (value=4 contiguous channels,
            # token=lane%16+16*(wave%2), channel=value+4*(lane//16)+16*(wave//2)); the same
            # mapping the old 64-bit scatter used, so retile(c_*_bf16) is a no-shuffle store.
            c_store_copy = fx.make_tiled_copy(
                cshuf_atom_w,
                fx.make_layout(((16, 4, 2, 2), 4), ((1, 128, 16, 512), 32)),
                fx.make_tile(32, 32),
            )
            cshuf_ptr = fx.recast_iter(fx.BFloat16, lds.gemm.at_lds.ptr)
            # Two LDS tiles (token, channel), channel-contiguous, back-to-back in GemmBuffers.
            # The 64-bit store's token stride (contiguous_n bf16 elems) is bank-aligned -> the
            # plain layout is 16-way bank-conflicted; XOR-swizzle both tiles (same view used by
            # the 64-bit store AND the 128-bit read -> consistent) to de-conflict. C-staging is
            # bf16, so the swizzle is bf16's (3,3,3). Needs no extra LDS (unlike padding).
            swz_c = fx.SwizzleType.get(3, 3, 3)
            _cshuf_half = BLOCK_TILE_SIZE_M // 2 * contiguous_n
            lds_top = fx.make_view(
                cshuf_ptr,
                fx.make_composed_layout(
                    fx.static(swz_c),
                    fx.make_ordered_layout((BLOCK_TILE_SIZE_M // 2, contiguous_n), order=(1, 0)),
                ),
            )
            lds_bot = fx.make_view(
                cshuf_ptr + _cshuf_half,
                fx.make_composed_layout(
                    fx.static(swz_c),
                    fx.make_ordered_layout((BLOCK_TILE_SIZE_M // 2, contiguous_n), order=(1, 0)),
                ),
            )

            # gemm_2x2's LDS reads must finish before we overwrite GemmBuffers.
            gpu.barrier()
            # store both halves to their own LDS region (no inter-store dependency)
            fx.copy(
                cshuf_atom_w,
                c_store_copy.get_slice(tid).retile(c_top_bf16),
                c_store_copy.get_slice(tid).partition_D(lds_top),
            )
            fx.copy(
                cshuf_atom_w,
                c_store_copy.get_slice(tid).retile(c_bot_bf16),
                c_store_copy.get_slice(tid).partition_D(lds_bot),
            )
            gpu.barrier()
            # read + scatter each half (rd_top is consumed before rd_bot -> no extra VGPRs)
            rd_top = fx.make_fragment_like(c_rw_copy.get_slice(tid).partition_S(lds_top))
            fx.copy(cshuf_atom_r, c_rw_copy.get_slice(tid).partition_S(lds_top), rd_top)
            c_top.copy(buf_atom_w128, blk_n, rd_top)
            rd_bot = fx.make_fragment_like(c_rw_copy.get_slice(tid).partition_S(lds_bot))
            fx.copy(cshuf_atom_r, c_rw_copy.get_slice(tid).partition_S(lds_bot), rd_bot)
            c_bot.copy(buf_atom_w128, blk_n, rd_bot)

    @flyc.kernel
    def moe_2stage_gateup_prefill_1x4(
        p_input: fx.Pointer,  # bf16 or native-fp8 [M, K]
        p_weight: fx.Pointer,  # bf16/fp8 [N/16, K/8 * 16 * 8]
        p_output: fx.Pointer,  # bf16 [M, TOPK, N//2]
        p_sorted_ids: fx.Pointer,
        p_sorted_weights: fx.Pointer,
        p_sorted_expert_ids: fx.Pointer,
        p_num_valid_ids: fx.Pointer,
        p_w_scale: fx.Pointer,  # weight fp8 scale (per-output-channel ptpc / per-tensor)
        p_a_scale: fx.Pointer,  # input fp8 scale (per-token ptpc / per-tensor)
        M: fx.Int32,
    ):
        tid = gpu.thread_idx.x
        blk_n = gpu.block_idx.x
        e_idx = gpu.block_idx.y

        if const_expr(weight_dtype != fx.BFloat16):
            in_ptr = fx.recast_iter(weight_dtype, _as_ptr(p_input))
        else:
            in_ptr = _as_ptr(p_input)
        arg_p_input = fx.make_view(in_ptr, fx.make_layout((M, K), (K, 1)))
        num_valid_buf = fx.make_view(
            fx.recast_iter(fx.Int32, _as_ptr(p_num_valid_ids)), fx.make_layout(1, 1)
        )
        max_valid_id = num_valid_buf[0]
        if const_expr(weight_dtype != fx.BFloat16):
            p_weight = fx.recast_iter(weight_dtype, _as_ptr(p_weight))
        if e_idx * BLOCK_TILE_SIZE_M < max_valid_id:
            lds = fx.SharedAllocator().allocate(SharedStorage)
            lds.sorted_lds = lds.sorted_lds.peek()
            lds.gemm = lds.gemm.peek()
            arg_p_sorted_ids = fx.make_view(
                fx.recast_iter(
                    fx.Int32, _as_ptr(p_sorted_ids) + e_idx * BLOCK_TILE_SIZE_M
                ),
                fx.make_layout(BLOCK_TILE_SIZE_M, 1),
            )
            arg_p_sorted_expert_ids = fx.recast_iter(
                fx.Int32, _as_ptr(p_sorted_expert_ids)
            )
            expert_id = arg_p_sorted_expert_ids[e_idx]

            contiguous_n = BLOCK_TILE_SIZE_N // 2
            group_layout_silu = fx.make_layout(
                ((contiguous_n, 2, N // (contiguous_n * 2)), K),
                ((1, N // 2, contiguous_n), N),
            )
            element_num = 16 // (p_weight.dtype.width // 8)
            arg_p_weight = fx.make_view(
                p_weight + fx.Int64(expert_id * N * K),
                fx.composition(
                    fx.make_layout(
                        ((16, N // 16), (element_num, K // element_num)),
                        ((element_num, 16 * K), (1, 16 * element_num)),
                    ),
                    group_layout_silu,
                ),
            )

            sorted_ids_buf = fx.rocdl.make_buffer_tensor(
                arg_p_sorted_ids, max_size=False
            )
            if tid < BLOCK_TILE_SIZE_M:
                lds_view = fx.make_view(
                    lds.sorted_lds.ptr, fx.make_layout(BLOCK_TILE_SIZE_M, 1)
                )
                lds_view[tid] = sorted_ids_buf[tid]
            gpu.barrier()

            # Output [M, TOPK, N//2] + the per-row scatter index, built BEFORE gemm_1x4:
            # sorted_lds is unioned with a_ping, so c_out must seed its index_frag from
            # sorted_lds now; gemm_1x4 then overwrites that LDS region with the A tile.
            arg_p_output = fx.make_view(
                _as_ptr(p_output),
                fx.make_layout((M, TOPK, N // 2), (TOPK * N // 2, N // 2, 1)),
            )
            out_tensor = fx.rocdl.make_buffer_tensor(
                arg_p_output,
                max_size=False,
                num_records_bytes=M * TOPK * N // 2 * fx.BFloat16.width // 8,
            )
            buf_atom_w128 = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16)
            # CShuffle read/scatter over the single (BM x contiguous_n) region: reuse the
            # 2x2-wave read layout (token = lane//4 + 16*waveM, 8 contiguous channels/lane,
            # 128b). The read is decoupled from the gemm's 1x4 wave layout (it just walks the
            # staged LDS), so rep_token = BM//32 and rep_channel = contiguous_n//64.
            c_rw_copy = fx.make_tiled_copy(
                buf_atom_w128,
                fx.make_layout(((4, 16, 2, 2), 8), ((256, 1, 16, 1024), 32)),
                fx.make_tile(32, 64),
            )
            c_index_copy = fx.make_tiled_copy(
                fx.make_copy_atom(fx.UniversalCopy32b(), fx.Int32),
                fx.make_layout(((4, 16, 2, 2), 1), ((0, 1, 16, 0), 0)),
                fx.make_tile(32),
            )
            c_out = TensorWithIndex(
                out_tensor,
                BLOCK_TILE_SIZE_M,
                contiguous_n,
                c_index_copy,
                c_rw_copy,
                tid,
                lds.sorted_lds,
                is_read_from_mem=False,
                TOPK=TOPK,
                index_size=BLOCK_TILE_SIZE_M,
                index_offset=0,
            )

            # ptpc a_scale is per-token; B-first packs 4 CONTIGUOUS channels per lane in the
            # value dim, so token = lane%16 + 16*token_rep (one id per token_rep, shared by
            # the 4 channel values). Gather the per-token_rep sorted id here, before gemm_1x4
            # overwrites sorted_lds.
            if const_expr(weight_dtype != fx.BFloat16 and act_quant_type == "ptpc"):
                asc_index_copy = fx.make_tiled_copy(
                    fx.make_copy_atom(fx.UniversalCopy32b(), fx.Int32),
                    fx.make_layout(((16, 4, 4), 1), ((1, 0, 0), 0)),
                    fx.make_tile(16),
                )
                cp_atom_idx = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Int32)
                asc_lds = fx.make_view(
                    lds.sorted_lds.ptr, fx.make_layout(BLOCK_TILE_SIZE_M, 1)
                )
                asc_thr = asc_index_copy.get_slice(tid).partition_S(asc_lds)
                asc_idx = fx.make_fragment_like(asc_thr)
                fx.copy(cp_atom_idx, asc_thr, asc_idx)

            c_gate_frag, c_up_frag = gemm_1x4(
                BLOCK_TILE_SIZE_M,
                BLOCK_TILE_SIZE_N,
                TILE_K,
                blk_n,
                arg_p_input,
                arg_p_weight,
                lds,
            )

            m_reps = fx.size(fx.get_shape(c_gate_frag)[1]).to_py_value()
            n_reps = fx.size(fx.get_shape(c_gate_frag)[2]).to_py_value()

            if const_expr(weight_dtype != fx.BFloat16 and act_quant_type == "ptpc"):
                # B-first dequant with per-token act scale: value dim = 4 contiguous channels,
                # m_rep = channel_rep, n_rep = token_rep. a_scale is per token (one scalar per
                # token_rep, shared by the 4 channel values). b_scale is per-output-channel
                # (one per value + channel_rep) when weight is ptpc, else a per_tensor scalar.
                # Fold both into c before the plain silu.
                if const_expr(weight_quant_type == "ptpc"):
                    scale_gate = fx.make_view(
                        _as_ptr(p_w_scale) + expert_id * N + blk_n * contiguous_n,
                        fx.make_layout(contiguous_n, 1),
                    )
                    scale_up = fx.make_view(
                        _as_ptr(p_w_scale) + expert_id * N + N // 2 + blk_n * contiguous_n,
                        fx.make_layout(contiguous_n, 1),
                    )
                    cp_atom_scale = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32)
                    # channel = v + 4*(lane//16) + 16*wave (+ 64*channel_rep): gather 4 per
                    # value into [v, channel_rep] to match the C fragment channel layout.
                    scale_copy = fx.make_tiled_copy(
                        cp_atom_scale,
                        fx.make_layout(((16, 4, 4), 4), ((0, 4, 16), 1)),
                        fx.make_tile(64),
                    )
                    sg_thr = scale_copy.get_slice(tid).partition_S(scale_gate)
                    su_thr = scale_copy.get_slice(tid).partition_S(scale_up)
                    gate_scale = fx.make_fragment_like(sg_thr)
                    up_scale = fx.make_fragment_like(su_thr)
                    fx.copy(cp_atom_scale, sg_thr, gate_scale)
                    fx.copy(cp_atom_scale, su_thr, up_scale)
                else:
                    b_scalar = fx.make_view(
                        _as_ptr(p_w_scale) + expert_id, fx.make_layout(1, 1)
                    )[0]

                a_scale_tensor = fx.rocdl.make_buffer_tensor(
                    fx.make_view(
                        fx.recast_iter(fx.Float32, _as_ptr(p_a_scale)),
                        fx.make_layout(M, 1),
                    ),
                    max_size=False,
                )
                # a_scale is per token = per token_rep n (independent of channel_rep m), so
                # gather all n up front: removes the redundant gather across m and lets the
                # indexed loads issue together instead of serializing inside the inner loop.
                a_sc_n = [
                    a_scale_tensor[asc_idx[0, n] & 0xFFFFFF]
                    for n in range_constexpr(n_reps)
                ]
                for m in range_constexpr(m_reps):
                    if const_expr(weight_quant_type == "ptpc"):
                        sg_v = gate_scale[None, m].load()
                        su_v = up_scale[None, m].load()
                    for n in range_constexpr(n_reps):
                        a_sc = a_sc_n[n]
                        cg = c_gate_frag[None, m, n].load()
                        cu = c_up_frag[None, m, n].load()
                        cg_items = []
                        cu_items = []
                        for v in range_constexpr(4):
                            if const_expr(weight_quant_type == "ptpc"):
                                sg = sg_v[v]
                                su = su_v[v]
                            else:
                                sg = b_scalar
                                su = b_scalar
                            cg_items.append(cg[v] * sg * a_sc)
                            cu_items.append(cu[v] * su * a_sc)
                        c_gate_frag[None, m, n].store(Vec.from_elements(cg_items, fx.Float32))
                        c_up_frag[None, m, n].store(Vec.from_elements(cu_items, fx.Float32))
            elif const_expr(weight_dtype != fx.BFloat16 and act_quant_type == "per_tensor"):
                b_scale = fx.make_view(
                    _as_ptr(p_w_scale) + expert_id, fx.make_layout(1, 1)
                )[0]
                a_scale0 = fx.make_view(
                    fx.recast_iter(fx.Float32, _as_ptr(p_a_scale)), fx.make_layout(1, 1)
                )[0]
                scale = b_scale * a_scale0
                c_gate_frag.store(c_gate_frag.load() * scale)
                c_up_frag.store(c_up_frag.load() * scale)

            c_out_bf16 = _silu_pair_bf16(c_gate_frag, c_up_frag)

            # 128-bit CShuffle epilogue (single region). Reconstruct the 1x4 tiled_mma and
            # stage c_out_bf16 into the A LDS via make_tiled_copy_C (framework-consistent with
            # the make_fragment_C layout). B-first makes the value dim 4 contiguous channels,
            # so the store is 64-bit; read it back channel-contiguous (8 bf16/lane) so the
            # scatter issues 128-bit writes.
            if const_expr(weight_dtype == fx.BFloat16):
                _mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, fx.BFloat16))
                _k_perm = fx.make_layout((4, 4, 2), (1, 8, 4))
            else:
                _mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, weight_dtype))
                _k_perm = fx.make_layout((8, 4, 2), (1, 16, 8))
            _tiled_mma = fx.make_tiled_mma(
                _mma_atom,
                fx.make_layout((4, 1, 1), (1, 0, 0)),
                fx.make_tile(None, None, _k_perm),
            )
            cshuf_atom_w = fx.make_copy_atom(fx.UniversalCopy64b(), fx.BFloat16)
            cshuf_atom_r = fx.make_copy_atom(fx.UniversalCopy128b(), fx.BFloat16)
            cshuf_ptr = fx.recast_iter(fx.BFloat16, lds.gemm.a_ping.ptr)
            # B-first: store through the (channel=M, token=N) transpose view so the value dim
            # (4 contiguous channels) is channel-contiguous -> 64-bit ds_write; read back the
            # aliased (token, channel) view channel-contiguous (8 bf16/lane) for the 128b
            # scatter. Both views share the same LDS bytes AND linear-offset formula, so the
            # same XOR swizzle keeps them consistent. The swizzle is required: the token stride
            # (contiguous_n elems) is bank-aligned, so an unswizzled 64-bit store is 16-way
            # bank-conflicted; the swizzle spreads it (needs no extra LDS, unlike padding).
            # C-staging is bf16 in both the bf16 and fp8 paths (it holds the bf16 output), so
            # the de-conflict swizzle is bf16's (3,3,3) in both cases -- NOT the fp8 input swz.
            swz_c = fx.SwizzleType.get(3, 3, 3)
            lds_c_store = fx.make_view(
                cshuf_ptr,
                fx.make_composed_layout(
                    fx.static(swz_c),
                    fx.make_ordered_layout((contiguous_n, BLOCK_TILE_SIZE_M), order=(0, 1)),
                ),
            )
            lds_c = fx.make_view(
                cshuf_ptr,
                fx.make_composed_layout(
                    fx.static(swz_c),
                    fx.make_ordered_layout((BLOCK_TILE_SIZE_M, contiguous_n), order=(1, 0)),
                ),
            )

            gpu.barrier()  # gemm_1x4's LDS reads must finish before reusing GemmBuffers
            store_c = fx.make_tiled_copy_C(cshuf_atom_w, _tiled_mma).get_slice(tid)
            fx.copy(cshuf_atom_w, store_c.retile(c_out_bf16), store_c.partition_D(lds_c_store))
            gpu.barrier()
            rd = fx.make_fragment_like(c_rw_copy.get_slice(tid).partition_S(lds_c))
            fx.copy(cshuf_atom_r, c_rw_copy.get_slice(tid).partition_S(lds_c), rd)
            c_out.copy(buf_atom_w128, blk_n, rd)

    @flyc.kernel
    def moe_2stage_down_batch1(
        p_input: fx.Pointer,  # bf16 [M, TOPK, K]
        p_weight: fx.Pointer,  # quantized/bf16 [N/16, K/8 * 16 * 8]
        p_output: fx.Pointer,  # bf16 [M, N]
        p_topk_ids: fx.Pointer,
        p_topk_weights: fx.Pointer,
        p_w_scale: fx.Pointer,
    ):
        tid = gpu.thread_idx.x
        blk_n = gpu.block_idx.x
        e_idx = gpu.block_idx.y

        # batch1: input is gemm1_out[0, e_idx, :] (single token, expert slot e_idx). Point at that
        # row and broadcast it across the TILE_M MFMA rows (stride 0); every computed row is then
        # identical, so any single row is the real result.
        arg_p_input = fx.make_view(
            _as_ptr(p_input) + fx.Int64(e_idx * K),
            fx.make_layout((BLOCK_TILE_SIZE_M, K), (0, 1)),
        )
        if const_expr(weight_dtype != fx.BFloat16):
            p_weight = fx.recast_iter(fx.Uint8, _as_ptr(p_weight))
        arg_p_topk_ids = fx.recast_iter(fx.Int32, _as_ptr(p_topk_ids))
        arg_p_topk_weights = fx.recast_iter(fx.Float32, _as_ptr(p_topk_weights))
        expert_id = arg_p_topk_ids[e_idx]
        topk_weight = arg_p_topk_weights[e_idx]
        element_num = 16 // (p_weight.dtype.width // 8)
        arg_p_weight = fx.make_view(
            p_weight + fx.Int64(expert_id * N * K),
            # preshuffle layout: [16, (8, K // 8)]
            fx.make_layout(
                ((16, N // 16), (element_num, K // element_num)),
                ((element_num, 16 * K), (1, 16 * element_num)),
            ),
        )

        c_frag = gemm_splitk(
            BLOCK_TILE_SIZE_M,
            BLOCK_TILE_SIZE_N,
            TILE_K,
            blk_n,
            arg_p_input,
            arg_p_weight,
            None,
            splitk_waves=1,
            a_with_index=False,
        )

        _apply_down_scale(c_frag, tid, expert_id, blk_n, p_w_scale)

        # mul weight
        c_frag.store(c_frag.load() * topk_weight)

        c_frag_bf16 = _cvt_f32_to_bf16(c_frag)

        # write to mem
        arg_p_output = fx.make_view(_as_ptr(p_output), fx.make_layout((1, N), (N, 1)))
        cp_atom_w = fx.make_copy_atom(
            fx.UniversalAtomic(fx.AtomicOp.Add, fx.BFloat16), fx.BFloat16
        )
        c_tiled_g = fx.make_tiled_copy(
            cp_atom_w,
            # 16x4 threads, each writes 4 points in N dimension
            fx.make_layout(((16, 4), 4), ((1, 64), 16)),
            fx.make_tile(16, 16),
        )
        c_tile = fx.flat_divide(
            arg_p_output, fx.make_tile(BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N)
        )[None, None, None, blk_n]
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
                        pair = Vec.from_elements(
                            [reg_vec[i * 2], reg_vec[i * 2 + 1]], fx.BFloat16
                        )
                        ptr = ptr_base + i * 2
                        addr = fx.ptrtoint(ptr)
                        llvm_ptr_ty = ir.Type.parse("!llvm.ptr<1>")
                        llvm_ptr = llvm.IntToPtrOp(llvm_ptr_ty, addr.ir_value())
                        llvm.AtomicRMWOp(
                            llvm.AtomicBinOp.fadd,
                            llvm_ptr,
                            pair,
                            llvm.AtomicOrdering.monotonic,
                            syncscope="agent",
                            alignment=4,
                        )

    @flyc.jit
    def launch(
        p_input: fx.Pointer,
        p_weight: fx.Pointer,
        p_output: fx.Pointer,
        p_sorted_ids: fx.Pointer,
        p_sorted_weights: fx.Pointer,
        p_sorted_expert_ids: fx.Pointer,
        p_num_valid_ids: fx.Pointer,
        p_w_scale: fx.Pointer,
        M: fx.Int32,
        task_num: fx.Int32,
        stream: fx.Stream,
    ):
        CompilationContext.get_current()
        num_n_blocks = div_up(N, BLOCK_TILE_SIZE_N)
        if const_expr(E is not None):
            if M * TOPK <= E:
                task_num = M * TOPK
        if const_expr(stage == "gateup"):
            moe_2stage_gateup(
                p_input,
                p_weight,
                p_output,
                p_sorted_ids,
                p_sorted_weights,
                p_sorted_expert_ids,
                p_num_valid_ids,
                p_w_scale,
                M,
            ).launch(
                grid=(num_n_blocks, task_num, 1),
                block=(256, 1, 1),
                stream=stream,
            )
        else:
            moe_2stage_down(
                p_input,
                p_weight,
                p_output,
                p_sorted_ids,
                p_sorted_weights,
                p_sorted_expert_ids,
                p_num_valid_ids,
                p_w_scale,
                M,
            ).launch(
                grid=(num_n_blocks, task_num, 1),
                block=(64, 1, 1),
                stream=stream,
            )

    @flyc.jit
    def launch_batch1(
        p_input: fx.Pointer,
        p_weight: fx.Pointer,
        p_output: fx.Pointer,
        p_topk_ids: fx.Pointer,
        p_topk_weights: fx.Pointer,
        p_w_scale: fx.Pointer,
        task_num: fx.Int32,
        stream: fx.Stream,
    ):
        CompilationContext.get_current()
        num_n_blocks = div_up(N, BLOCK_TILE_SIZE_N)
        if const_expr(stage == "gateup"):
            moe_2stage_gateup_batch1(
                p_input, p_weight, p_output, p_topk_ids, p_w_scale
            ).launch(
                grid=(num_n_blocks, task_num, 1),
                block=(256, 1, 1),
                stream=stream,
            )
        else:
            moe_2stage_down_batch1(
                p_input, p_weight, p_output, p_topk_ids, p_topk_weights, p_w_scale
            ).launch(
                grid=(num_n_blocks, task_num, 1),
                block=(64, 1, 1),
                stream=stream,
            )

    @flyc.jit
    def launch_prefill_2x2(
        p_input: fx.Pointer,
        p_weight: fx.Pointer,
        p_output: fx.Pointer,
        p_sorted_ids: fx.Pointer,
        p_sorted_weights: fx.Pointer,
        p_sorted_expert_ids: fx.Pointer,
        p_num_valid_ids: fx.Pointer,
        p_w_scale: fx.Pointer,
        p_a_scale: fx.Pointer,
        M: fx.Int32,
        task_num: fx.Int32,
        stream: fx.Stream,
    ):
        CompilationContext.get_current()
        num_n_blocks = div_up(N, BLOCK_TILE_SIZE_N)
        if const_expr(E is not None):
            if M * TOPK <= E:
                task_num = M * TOPK
        moe_2stage_gateup_prefill_2x2(
            p_input,
            p_weight,
            p_output,
            p_sorted_ids,
            p_sorted_weights,
            p_sorted_expert_ids,
            p_num_valid_ids,
            p_w_scale,
            p_a_scale,
            M,
        ).launch(
            grid=(num_n_blocks, task_num, 1),
            block=(256, 1, 1),
            stream=stream,
        )

    @flyc.jit
    def launch_prefill_1x4(
        p_input: fx.Pointer,
        p_weight: fx.Pointer,
        p_output: fx.Pointer,
        p_sorted_ids: fx.Pointer,
        p_sorted_weights: fx.Pointer,
        p_sorted_expert_ids: fx.Pointer,
        p_num_valid_ids: fx.Pointer,
        p_w_scale: fx.Pointer,
        p_a_scale: fx.Pointer,
        M: fx.Int32,
        task_num: fx.Int32,
        stream: fx.Stream,
    ):
        CompilationContext.get_current()
        num_n_blocks = div_up(N, BLOCK_TILE_SIZE_N)
        if const_expr(E is not None):
            if M * TOPK <= E:
                task_num = M * TOPK
        moe_2stage_gateup_prefill_1x4(
            p_input,
            p_weight,
            p_output,
            p_sorted_ids,
            p_sorted_weights,
            p_sorted_expert_ids,
            p_num_valid_ids,
            p_w_scale,
            p_a_scale,
            M,
        ).launch(
            grid=(num_n_blocks, task_num, 1),
            block=(256, 1, 1),
            stream=stream,
        )

    if const_expr(alg == "prefill_1x4"):
        return launch_prefill_1x4
    if const_expr(alg == "prefill_2x2"):
        return launch_prefill_2x2
    if const_expr(alg == "batch1"):
        return launch_batch1
    return launch