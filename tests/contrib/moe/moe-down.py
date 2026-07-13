import os

import flydsl.compiler as flyc  # noqa: E402
from flydsl.compiler.kernel_function import CompilationContext  # noqa: E402
import flydsl.expr as fx
from flydsl.expr import const_expr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.utils.env import DebugEnvManager
from flydsl._mlir import ir
import flydsl
from flydsl._mlir.dialects import fly, llvm, vector, gpu, scf, rocdl
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr.utils.arith import _to_raw as _raw
import torch


import pyhip
import pyhip.contrib.flydsl as fxu
import math

# fxu.enable_dump_ir(True)

_, stream = pyhip.set_device()


def div_up(x, y):
    return (x + y - 1) // y

def load_fragment(thr_view: fx.Tensor):
    """
    make_fragment_like() reserves space for mode with stride-0, which is unexpected.
    this function loads a thread-view tensor into a fragment tensor, which is compact
    and only contains non-zero-stride modes, while profile is preserved.
    """
    tview_shape = thr_view.shape.to_py_value()
    tview_stride = thr_view.stride.to_py_value()
    nz_shape = []
    nz_stride = []
    nz_frag_stride = []
    fstride = 1
    def collect_nz_modes(shape, stride):
        nonlocal nz_shape, nz_stride, fstride
        frag_stride = []
        for s, d in zip(shape,stride):
            if isinstance(d, int):
                if d != 0:
                    nz_shape.append(s)
                    nz_stride.append(d)
                    nz_frag_stride.append(fstride)
                    frag_stride.append(fstride) # fragment stride is compact
                    fstride *= s
                else:
                    frag_stride.append(0) # fragment stride keeps all modes, even those with 0 stride
            else:
                frag_stride.append(collect_nz_modes(s, d))
        return frag_stride
    frag_stride = collect_nz_modes(tview_shape, tview_stride)
    nz_cnt = fstride
    # print(" thr_view shape: ", nz_shape, " stride: ", nz_stride, " frag_stride: ", frag_stride, " nz_cnt: ", nz_cnt)

    if len(nz_shape) == 0:
        nz_shape = 1
        nz_stride = 0
    thr_view_nz = fx.make_view(fx.get_iter(thr_view), fx.make_layout(nz_shape, nz_stride))
    frag = fx.make_rmem_tensor(fx.make_layout(nz_shape, nz_frag_stride), thr_view.dtype)

    vec = thr_view_nz.load()
    frag.store(vec) # store to rmem tensor usually do nothing after lowering

    # reshape back to thread-view domain
    #frag = fx.composition(frag, fx.make_layout(tview_shape, frag_stride))
    frag = fx.make_view(fx.get_iter(frag), fx.make_layout(tview_shape, frag_stride))

    return frag

def all_elements(*tensors, scalar=False):
    """Iterate broadcasted element views from multiple FlyDSL tensors.

    The first tensor is treated as the leader for iteration shape/rank. Other
    tensors must be broadcast-compatible with that leader per mode (size 1 is
    broadcastable). Iteration skips mode 0 and advances modes [1..rank-1] in a
    row-major style, with leader strides used to detect singular modes
    (stride==0 means that mode is iterated once at coordinate 0).

    Args:
        *tensors: FlyDSL tensors/views sharing a compatible layout profile.
        scalar: If True, prepends a synthetic leading size-1 mode to each input
            to support scalar-like iteration in fused loops.

    Yields:
        list: One sliced element-view per input tensor at the current logical
        coordinate, suitable for per-element load/store or copy-atom handling.
    """
    def _htuple2flat(htuple):
        if isinstance(htuple, (tuple, list)):
            flat = []
            for h in htuple:
                flat.extend(_htuple2flat(h))
            return flat
        else:
            return (htuple,)

    def _flat2htuple(flat, ht_guide):
        if isinstance(ht_guide, (tuple, list)):
            htuple = []
            for guide in ht_guide:
                ele = _flat2htuple(flat, guide)
                htuple.append(ele)
            return htuple
        else:
            return flat.pop(0)

    leader_shape = None
    flat_tensors = []
    flat_shapes = []
    layout0 = tensors[0].layout
    stride0 = _htuple2flat(
        layout0.outer.stride.to_py_value()
        if isinstance(layout0, fx.ComposedLayout)
        else layout0.stride.to_py_value()
    )
    if scalar:
        stride0.insert(0, 0)
    for i in fx.range_constexpr(len(tensors)):
        assert tensors[i].shape.is_static
        shape = tensors[i].shape.to_py_value()
        static_shape = _htuple2flat(shape)
        slice_all = _flat2htuple([None for _ in static_shape], shape)
        ft = tensors[i][slice_all]
        if scalar:
            # prepend a 1 mode for slicing the scalar tensor
            ft = fx.make_view(
                fx.get_iter(ft), fx.prepend(ft.layout, fx.make_layout(1, 0))
            )
            static_shape.insert(0, 1)
        flat_tensors.append(ft)
        flat_shapes.append(static_shape)
        if i == 0:
            leader_shape = static_shape
        else:
            assert len(static_shape) == len(
                leader_shape
            ), f"{i}'th rank is not consistent with leader"
            for s, m in zip(static_shape, leader_shape):
                assert (
                    s == 1 or (s == m and m > 1) or (m == 1)
                ), f"{i}'th shape {static_shape} is not broadcastable to leader's shape {leader_shape}"

    coord = [0 for _ in leader_shape]

    rank = len(leader_shape)
    r = 1
    while r < rank:
        ret = []
        for fshape, ftensor in zip(flat_shapes, flat_tensors):
            crd = [None]
            for c, s in zip(coord[1:], fshape[1:]):
                crd.append(min(c, s - 1))
            # print(crd, ftensor, fx.slice(ftensor, crd))
            ret.append(fx.slice(ftensor, crd))
        yield ret

        r = 1
        while r < rank:
            coord[r] += 1
            if coord[r] < leader_shape[r] and stride0[r] > 0:
                break
            # finished rank r : full size iterated (stride==0 means singular)
            coord[r] = 0
            r += 1


def create_thr_mma(dtype, wave_mnk):
    mfma_M = 16
    mfma_N = 16
    mfma_K = {fx.Float8E4M3FNUZ: 32, fx.BFloat16: 16, fx.Float16: 16, fx.Float32: 4}[
        dtype
    ]
    mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(mfma_M, mfma_N, mfma_K, dtype))

    wave_m, wave_n, wave_k = wave_mnk
    thr_layout_mnk = fx.make_layout(
        (wave_m, wave_n, wave_k), (1, wave_m, 0 if wave_k == 1 else wave_m * wave_n)
    )

    atom_frgv = mfma_K // 4  # how many elements in a fragment vector (per-thread)
    num_frgv_in_DW4 = 128 // (
        atom_frgv * dtype.width
    )  # to use DW4 load, how many atom_frgv needs to be packed
    num_elements_in_DW4 = 128 // dtype.width
    k_perm = fx.make_layout(
        (atom_frgv, 4, num_frgv_in_DW4), (1, num_elements_in_DW4, atom_frgv)
    )
    permutation_mnk = (None, None, k_perm)
    tiled_mma = fx.make_tiled_mma(mma_atom, thr_layout_mnk, permutation_mnk)

    return tiled_mma.get_slice(fx.thread_idx.x)


def _as_ptr(p):
    """Convert memref or pointer to a pointer/iterator suitable for fx.make_view.
    Handles both raw fx.Pointer values and memref values passed by flydsl runtime."""
    try:
        return fx.get_iter(p)
    except Exception:
        return p


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
    if weight_dtype == "bf16":
        weight_dtype = fx.BFloat16
    elif weight_dtype == "fp8":
        weight_dtype = fx.Float8E4M3FNUZ

    def moe_gather_tokens(
        ptr_row_index, src_tensor, dst_tensor, num_threads=256, copy_atom_bits=128
    ):
        assert (
            dst_tensor.layout.rank == 2
        ), f"expect dst_tensor to be in [num_tokens, hidden_states]"
        assert (
            src_tensor.layout.rank == 3
        ), f"expect src_tensor to be in [M, TOPK, hidden_states]"
        assert src_tensor.dtype == dst_tensor.dtype
        num_rows = fx.size(dst_tensor.layout.shape[0]).to_py_value()
        num_cols = fx.size(dst_tensor.layout.shape[1]).to_py_value()
        cp_atom = fx.make_copy_atom(fx.UniversalCopy(copy_atom_bits), src_tensor.dtype)
        row_tensor = fx.make_view(
            ptr_row_index, fx.make_layout((num_rows, num_cols), (1, 0))
        )
        col_tensor = fx.make_view(
            fx.make_int_tuple(0), fx.make_layout((num_rows, num_cols), (0, 1))
        )
        num_vals = copy_atom_bits // (src_tensor.dtype.width)
        thread_n = num_cols // num_vals
        thread_m = num_threads // thread_n
        tile_mn = (thread_m, thread_n * num_vals)
        stride = lambda m, n: m + n * tile_mn[0]
        thrcopy = fx.make_tiled_copy(
            cp_atom,
            fx.make_layout(
                ((thread_m, thread_n), num_vals),
                ((stride(1, 0), stride(0, num_vals)), stride(0, 1)),
            ),
            tile_mn,
        ).get_slice(fx.thread_idx.x)
        thrv_row = thrcopy.partition_D(row_tensor)
        thrv_col = thrcopy.partition_D(col_tensor)
        thrv_dst = thrcopy.partition_D(dst_tensor)
        # preload all row indicies into fragment
        frag_row = load_fragment(thrv_row)
        for dst, row, col in all_elements(thrv_dst, frag_row, thrv_col):
            # each element (smallest left-inner-most mode) is a copy-atom
            sorted_id = row[0].bitcast(fx.Uint32)
            atom_A = fx.make_view(
                fx.get_iter(src_tensor)
                + src_tensor.layout(sorted_id & 0xFFFFFF, sorted_id >> 24, col[0]),
                cp_atom.layout_src_tv[1],
            )
            fx.copy(cp_atom, atom_A, dst)
        fx.gpu.barrier()

    @flyc.kernel
    def moe_2stage_down_prefill_1x4(
        p_input: fx.Pointer,  # bf16 [M, TOPK, K]           K = HIDDEN_STATES//TP
        p_weight: fx.Pointer,  # quantized/bf16 [E, N, K]   N = HIDDEN_STATES
        p_output: fx.Pointer,  # bf16 [M, TOPK, N]
        p_sorted_ids: fx.Pointer,  # f32 [num_tokens_sorted]
        p_sorted_weights: fx.Pointer,  # f32 [num_tokens_sorted]
        p_sorted_expert_ids: fx.Pointer,  # int32 [num_blocks] num_tokens_sorted <= num_blocks * BLOCK_TILE_SIZE_M
        p_num_valid_ids: fx.Pointer,  # int32 [2]  value: (true_valid_tokens(M*TOPK), M)
        p_w_scale: fx.Pointer,  # weight fp8 scale (per-output-channel ptpc / per-tensor)
        p_a_scale: fx.Pointer,  # input fp8 scale (per-token ptpc / per-tensor)
        M: fx.Int32,
    ):
        tid = fx.gpu.thread_idx.x
        blk_n = fx.gpu.block_idx.x  # always 0
        e_idx = fx.gpu.block_idx.y

        if fx.const_expr(weight_dtype != fx.BFloat16):
            in_ptr = fx.recast_iter(weight_dtype, _as_ptr(p_input))
        else:
            in_ptr = _as_ptr(p_input)

        arg_p_input = fx.make_view(
            in_ptr, fx.make_ordered_layout((M, TOPK, K), (2, 1, 0))
        )
        arg_p_output = fx.make_view(
            _as_ptr(p_output),
            fx.make_ordered_layout((M, TOPK, N), (2, 1, 0)),
        )
        num_valid_buf = fx.make_view(
            fx.recast_iter(fx.Int32, _as_ptr(p_num_valid_ids)), fx.make_layout(1, 1)
        )
        max_valid_id = num_valid_buf[0]

        if fx.const_expr(weight_dtype != fx.BFloat16):
            p_weight = fx.recast_iter(weight_dtype, _as_ptr(p_weight))

        if e_idx * BLOCK_TILE_SIZE_M < max_valid_id:
            arg_p_sorted_ids = fx.make_view(
                fx.recast_iter(
                    fx.Int32, _as_ptr(p_sorted_ids) + e_idx * BLOCK_TILE_SIZE_M
                ),
                fx.make_layout(BLOCK_TILE_SIZE_M, 1),
            )
            arg_p_sorted_weights = fx.make_view(
                fx.recast_iter(
                    fx.Float32, _as_ptr(p_sorted_weights) + e_idx * BLOCK_TILE_SIZE_M
                ),
                fx.make_layout(BLOCK_TILE_SIZE_M, 1),
            )
            arg_p_sorted_expert_ids = fx.recast_iter(
                fx.Int32, _as_ptr(p_sorted_expert_ids)
            )
            expert_id = arg_p_sorted_expert_ids[e_idx]

            # 16bytes/DW4
            element_num = 16 // (p_weight.dtype.width // 8)
            arg_p_weight = fx.make_view(
                p_weight + fx.Int64(expert_id * N * K),
                fx.make_layout(
                    ((16, N // 16), (element_num, K // element_num)),
                    ((element_num, 16 * K), (1, 16 * element_num)),
                ),
            )

            sorted_ids_buf = fx.rocdl.make_buffer_tensor(
                arg_p_sorted_ids, max_size=False
            )
            cp_atom_128b = fx.make_copy_atom(fx.UniversalCopy128b(), weight_dtype)
            cp_atom_64b = fx.make_copy_atom(fx.UniversalCopy64b(), fx.BFloat16)

            BLOCK_M = BLOCK_TILE_SIZE_M
            BLOCK_N = 64
            BLOCK_K = 64 // (
                weight_dtype.width // 8
            )  # BLOCK_K * byte_size(weight_dtype) = 128 bytes

            act_dtype = weight_dtype  # fp8 / bf16

            # mask,base,shift
            if const_expr(weight_dtype == fx.BFloat16):
                swz = fx.SwizzleType.get(3, 3, 3)
            else:
                swz = fx.SwizzleType.get(3, 4, 3)

            @fx.union
            class SharedStorage:
                A: fx.Array[act_dtype, BLOCK_M * K]
                C: fx.Array[fx.BFloat16, BLOCK_M * BLOCK_N]

            # swizzle happens in unit of 128b,
            lds = fx.SharedAllocator().allocate(SharedStorage)
            ldsA0 = lds.A.peek().view(
                fx.make_composed_layout(
                    fx.static(swz), fx.make_layout((BLOCK_M, K), (K, 1))
                )
            )

            moe_gather_tokens(fx.get_iter(arg_p_sorted_ids), arg_p_input, ldsA0)

            weight = fx.flat_divide(
                arg_p_weight, (BLOCK_N, BLOCK_K)
            )  # (BLOCK_N, BLOCK_K, num_blocks_N, num_blocks_K)
            ldsA = fx.flat_divide(
                ldsA0, (BLOCK_M, BLOCK_K)
            )  # S<3,3,3> o (BLOCK_M, BLOCK_K, num_blocks_M, num_blocks_K)

            nBM = 1
            nBN = div_up(N, BLOCK_N)
            nBK = div_up(K, BLOCK_K)

            mm = create_thr_mma(weight_dtype, (4, 1, 1))
            fragA = mm.make_fragment_B(ldsA)
            frag_weights = [mm.make_fragment_A(weight[None, None, 0, None]),
                            mm.make_fragment_A(weight[None, None, 0, None])]

            c_fake_tensor = fx.make_view(
                fx.get_iter(arg_p_input),
                fx.make_ordered_layout((BLOCK_N, BLOCK_M), (0, 1)),
            )
            fragC = [
                mm.make_fragment_C(c_fake_tensor),
                mm.make_fragment_C(c_fake_tensor),
            ]
            fragC_bf16 = fx.make_fragment_like(fragC[0], fx.BFloat16)

            # copy gathered tokens into fragA
            thrcopy_act = fx.make_tiled_copy_B(cp_atom_128b, mm).get_slice(
                fx.thread_idx.x
            )
            fx.copy(
                cp_atom_128b, thrcopy_act.partition_S(ldsA), thrcopy_act.retile(fragA)
            )

            thrcopy_weight = fx.make_tiled_copy_A(cp_atom_128b, mm).get_slice(
                fx.thread_idx.x
            )
            thrcopy_C = fx.make_tiled_copy_C(cp_atom_64b, mm).get_slice(fx.thread_idx.x)

            peekC = lds.C.peek()
            layout = fx.make_composed_layout(
                fx.static(swz),
                fx.make_ordered_layout((BLOCK_M, BLOCK_N), (1, 0)),
            )

            # thrcopy_C stores into transposed C, thus the 2 modes need to be swapped
            # (which allows 4xbf16 to be physically continous)
            #   dequant scales also need to be loaded using tiled-copy C
            #   (because they are applied to C matrix)
            #
            #  pc-scales load only once for a fragC and reused for all fragC along different tokens
            #  pt-scales are gathered from M*TOPK, for (BLOCK_M x 16)
            #
            # broad-cast scales
            with_dequant_pc = False
            if const_expr(weight_quant_type == "per_tensor"):
                arg_w_scale = fx.make_view(
                    _as_ptr(p_w_scale) + expert_id, fx.make_layout((N, 1), (0, 0))
                )
                with_dequant_pc = True
            if const_expr(weight_quant_type == "ptpc"):
                arg_w_scale = fx.make_view(
                    _as_ptr(p_w_scale) + expert_id * N, fx.make_layout((N, 1), (1, 0))
                )
                with_dequant_pc = True

            if const_expr(with_dequant_pc):
                # each value belong to same column needs a scale
                arg_w_scale = fx.flat_divide(
                    arg_w_scale, (BLOCK_N, 1)
                )  # (BLOCK_N, 1, num_block_N, 1)
                cp_atom_pc_scales = fx.make_copy_atom(
                    fx.UniversalCopy128b(), p_w_scale.dtype
                )
                thrcopy_pc_scales = fx.make_tiled_copy_C(
                    cp_atom_pc_scales, mm
                ).get_slice(fx.thread_idx.x)
                thrv_w_scale = thrcopy_pc_scales.partition_S(arg_w_scale)
                frag_pc_scales = [
                    mm.make_fragment_C(arg_w_scale[None, None, 0, 0]),
                    mm.make_fragment_C(arg_w_scale[None, None, 0, 0]),
                ]
            else:
                frag_pc_scales = [None, None]

            with_dequant_pt = False
            if const_expr(act_quant_type == "per_tensor"):
                arg_a_scale = fx.make_view(
                    fx.recast_iter(fx.Float32, _as_ptr(p_a_scale)),
                    fx.make_layout((M, TOPK), (0, 0)),
                )
                with_dequant_pt = True
            if const_expr(act_quant_type == "ptpc"):
                arg_a_scale = fx.make_view(
                    fx.recast_iter(fx.Float32, _as_ptr(p_a_scale)),
                    fx.make_layout((M, TOPK), (TOPK, 1)),
                )
                with_dequant_pt = True

            if const_expr(with_dequant_pt):
                # if we can load per-token scales according to "make_tiled_copy_C"
                # our dequant algo will be more robust can work in any tiledMma.
                # tiled_copy_C can loads data into C-tile-format, but we need to
                # gather them according to sorted_id
                cp_atom_pt_scales = fx.make_copy_atom(
                    fx.UniversalCopy32b(), p_a_scale.dtype
                )
                thrcopy_pt_scales = fx.make_tiled_copy_C(
                    cp_atom_pt_scales, mm
                ).get_slice(fx.thread_idx.x)

                coord_tensor = fx.make_view(
                    fx.get_iter(arg_p_sorted_ids),
                    fx.make_layout((BLOCK_N, BLOCK_M), (0, 1)),
                )
                thrv_coord = thrcopy_pt_scales.partition_S(coord_tensor)
                frag_coord = load_fragment(thrv_coord)
                frag_pt_scales = mm.make_fragment_C(coord_tensor)

                for dst, coord in all_elements(
                    thrcopy_pt_scales.retile(frag_pt_scales), frag_coord
                ):
                    sorted_id = coord[0].bitcast(fx.Uint32)
                    atom_A = fx.make_view(
                        fx.get_iter(arg_a_scale)
                        + arg_a_scale.layout(sorted_id & 0xFFFFFF, sorted_id >> 24),
                        cp_atom_pt_scales.layout_src_tv[1],
                    )
                    fx.copy(cp_atom_pt_scales, atom_A, dst)

            col_tensor = fx.make_view(
                fx.make_int_tuple(0), fx.make_layout((N, BLOCK_M), (1, 0))
            )
            row_tensor = fx.make_view(
                fx.get_iter(arg_p_sorted_ids), fx.make_layout((N, BLOCK_M), (0, 1))
            )
            weight_tensor = fx.make_view(
                fx.get_iter(arg_p_sorted_weights), fx.make_layout((N, BLOCK_M), (0, 1))
            )
            col_tensor = fx.flat_divide(col_tensor, (BLOCK_N, BLOCK_M))
            row_tensor = fx.flat_divide(row_tensor, (BLOCK_N, BLOCK_M))
            weight_tensor = fx.flat_divide(weight_tensor, (BLOCK_N, BLOCK_M))

            thrv_dst_col = thrcopy_C.partition_D(col_tensor)
            thrv_dst_row = thrcopy_C.partition_D(row_tensor)
            thrv_weight = thrcopy_C.partition_D(weight_tensor)
            frag_row = load_fragment(thrv_dst_row[None, None, None, 0, 0])
            frag_sorted_weight = load_fragment(thrv_weight[None, None, None, 0, 0])

            if const_expr(with_dequant_pt):
                # combine per-token scales with per-token weights
                for frag_pt, frag_sw in all_elements(
                    frag_pt_scales, frag_sorted_weight
                ):
                    frag_pt.store(frag_pt.load() * frag_sw.load())

                frag_sorted_weight = frag_pt_scales

            def copy_C_to_global(frag, m, n):
                for src_frag, row, col in all_elements(
                    frag, frag_row, thrv_dst_col[None, None, None, n, m]
                ):
                    sorted_id = row[0].bitcast(fx.Uint32)
                    atom_C = fx.make_view(
                        fx.get_iter(arg_p_output)
                        + arg_p_output.layout(
                            sorted_id & 0xFFFFFF, sorted_id >> 24, col[0]
                        ),
                        cp_atom_64b.layout_dst_tv[1],
                    )
                    fx.copy(cp_atom_64b, src_frag, atom_C)

            def f32_to_bf16(x):
                round_bit = fx.Uint32(0x8000).ir_value().bitcast(fx.Float32.ir_type)
                return (
                    ((x + round_bit).bitcast(fx.Uint32) >> 16)
                    .to(fx.Uint16)
                    .bitcast(fx.BFloat16)
                )

            def preload_weight(n, fragW):
                for k in fx.range_constexpr(nBK):
                    fx.copy(
                        cp_atom_128b,
                        thrcopy_weight.partition_S(weight)[None, None, None, n, k],
                        thrcopy_weight.retile(fragW[None, None, None, k]),
                    )

            def preload_pc_scales(n, fragPCS):
                if const_expr(with_dequant_pc):
                    fx.copy(
                        cp_atom_pc_scales,
                        thrv_w_scale[None, None, None, n, 0],
                        fragPCS,
                    )

            def gemm_compute(fragW, fragC):
                fragC.fill(0)
                for k in fx.range_constexpr(nBK):
                    fx.gemm(mm, fragC, fragW[None, None, None, k], fragA[None, None, None, 0, k], fragC)

            def postprocess_store(n, fragC, fragPCS):
                if const_expr(with_dequant_pc):
                    for fc, fpt, fpc in all_elements(
                        fragC, frag_sorted_weight, fragPCS
                    ):
                        fc.store(fc.load() * (fpt.load() * fpc.load()))
                else:
                    for fc, fsw in all_elements(fragC, frag_sorted_weight):
                        fc.store(fc.load() * fsw.load())
                vec_f32 = fragC.load()
                fragC_bf16.store(f32_to_bf16(vec_f32))
                copy_C_to_global(fragC_bf16, 0, n)

            # prelog
            preload_weight(0, frag_weights[0])
            # prelog
            gemm_compute(frag_weights[0], fragC[0])
            preload_weight(1, frag_weights[1])
            preload_pc_scales(0, frag_pc_scales[0])

            num_mfma_inst = (BLOCK_M // 16) * (
                K // (16 if weight_dtype.width == 16 else 32)
            )
            num_stores = BLOCK_M // 16
            num_loads = K // ((4 * 8) if weight_dtype.width == 16 else (4 * 16))

            def hot_loop_scheduler():
                mfma_step = num_mfma_inst // (num_loads + num_stores)
                fx.rocdl.sched_mfma(2)
                for _ in fx.range_constexpr(num_loads):
                    fx.rocdl.sched_vmem(1)
                    fx.rocdl.sched_mfma(mfma_step)

                for _ in fx.range_constexpr(num_stores):
                    fx.rocdl.sched_vmem(1)
                    fx.rocdl.sched_mfma(mfma_step)

                mfma_rest = num_mfma_inst - (num_loads + num_stores) * mfma_step
                if mfma_rest > 0:
                    fx.rocdl.sched_mfma(mfma_rest)
                fx.rocdl.sched_barrier(0)

            for n, state in range(0, nBN - 2, 2, init=[]):
                postprocess_store(n, fragC[0], frag_pc_scales[0])
                gemm_compute(frag_weights[1], fragC[1])
                preload_weight(n + 2, frag_weights[0])
                preload_pc_scales(n + 1, frag_pc_scales[1])

                hot_loop_scheduler()

                postprocess_store(n + 1, fragC[1], frag_pc_scales[1])
                gemm_compute(frag_weights[0], fragC[0])
                preload_weight(n + 3, frag_weights[1])
                preload_pc_scales(n + 2, frag_pc_scales[0])
                hot_loop_scheduler()

            # epilogue
            gemm_compute(frag_weights[1], fragC[1])
            preload_pc_scales(nBN - 1, frag_pc_scales[1])
            postprocess_store(nBN - 2, fragC[0], frag_pc_scales[0])
            # epilogue
            postprocess_store(nBN - 1, fragC[1], frag_pc_scales[1])


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
        if fx.const_expr(E is not None):
            if M * TOPK <= E:
                task_num = M * TOPK
        value_attrs = {
            "rocdl.waves_per_eu": 2,
            "passthrough": [
                ["amdgpu-agpr-alloc", "256,256"],
            ],
        }
        value_attrs = None
        moe_2stage_down_prefill_1x4(
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
            value_attrs=value_attrs,
        ).launch(
            grid=(1, task_num, 1),
            block=(256, 1, 1),
            stream=stream,
        )

    return launch_prefill_1x4


def _fly_dispatch(cache_key, build_jit, args):
    """Run a FlyDSL launch via a prebuilt CallState (flyc.compile).

    On the first use (cache miss) ``flyc.compile`` traces, compiles AND executes
    the kernel once with ``args`` -- so we must NOT call it again here (the down
    stage uses atomic-add accumulation; a second call would double the output).
    On every later use we invoke the cached fast-dispatch callable exactly once.
    """
    import flydsl.compiler as flyc

    compiled = _FLY_COMPILED_CACHE.get(cache_key)
    if compiled is None:
        _FLY_COMPILED_CACHE[cache_key] = flyc.compile(build_jit(), *args)
    else:
        compiled(*args)


def test(pt_file):
    moe_down_data = torch.load(pt_file, torch.get_default_device())
    down_in = moe_down_data["down_in"]
    w2 = moe_down_data["w2"]
    gemm2_out = moe_down_data["gemm2_out"]
    sorted_ids = moe_down_data["sorted_ids"]
    sorted_weights = moe_down_data["sorted_weights"]
    sorted_expert_ids = moe_down_data["sorted_expert_ids"]
    num_valid_ids = moe_down_data["num_valid_ids"]
    w2_scale_arg = moe_down_data["w2_scale_arg"]
    a_scale = moe_down_data["a_scale"]
    B = moe_down_data["B"]
    grid = moe_down_data["grid"]
    N = moe_down_data["N"]
    K = moe_down_data["K"]
    weight_dtype = moe_down_data["weight_dtype"]
    weight_quant_type = moe_down_data["weight_quant_type"]

    if "act_quant_type" in moe_down_data:
        act_quant_type = moe_down_data["act_quant_type"]
    else:
        act_quant_type = None

    TOPK = moe_down_data["TOPK"]
    BLOCK_TILE_SIZE_M = moe_down_data["BLOCK_TILE_SIZE_M"]
    BLOCK_TILE_SIZE_N = moe_down_data["BLOCK_TILE_SIZE_N"]
    stage = moe_down_data["stage"]
    alg = moe_down_data["alg"]
    E = moe_down_data["E"]
    USE_ATOMIC_WRITE = moe_down_data["USE_ATOMIC_WRITE"]

    if 0:
        down_in *= 1000
        print(sorted_ids[:BLOCK_TILE_SIZE_M].view(-1, 8))
        print(down_in[1, 0, :])
    moe = compile_gemm(
        N,
        K,
        weight_dtype,
        weight_quant_type,
        TOPK,
        BLOCK_TILE_SIZE_M,
        BLOCK_TILE_SIZE_N,
        stage=stage,
        alg=alg,
        E=E,
        USE_ATOMIC_WRITE=USE_ATOMIC_WRITE,
        act_quant_type=act_quant_type,
        tile_k=None,
    )

    _TORCH_TO_FX = {
        torch.bfloat16: fx.BFloat16,
        torch.float32: fx.Float32,
        torch.int32: fx.Int32,
        torch.float8_e4m3fnuz: fx.Uint8,
        torch.float8_e4m3fn: fx.Uint8,
    }

    def _ptr(t):
        return flyc.from_c_void_p(_TORCH_TO_FX[t.dtype], t.data_ptr())

    if act_quant_type == "ptpc":
        import aiter

        down_in, a_scale = aiter.get_hip_quant(aiter.QuantType.per_Token)(
            down_in.view(B * TOPK, -1), quant_dtype=w2.dtype
        )
        a_scale = a_scale.to(torch.float32).contiguous()
    elif act_quant_type == "per_tensor":
        fmax = torch.finfo(w2.dtype).max
        a_scale = down_in.float().abs().amax() / fmax
        down_in = (down_in.float() / a_scale).clamp(-fmax, fmax).to(w2.dtype)
        a_scale = a_scale.reshape(1).to(torch.float32)
    else:
        # no quant
        pass

    moedown_out = torch.empty([B, TOPK, N], dtype=gemm2_out.dtype)

    args = (
        _ptr(down_in),
        _ptr(w2),
        _ptr(moedown_out),
        _ptr(sorted_ids),
        _ptr(sorted_weights),
        _ptr(sorted_expert_ids),
        _ptr(num_valid_ids),
        _ptr(w2_scale_arg),
        _ptr(a_scale),
        B,
        grid,
        stream,
    )

    compiled = flyc.compile(moe, *args)

    num_flops = sorted_expert_ids.numel() * (BLOCK_TILE_SIZE_M * N * K * 2)
    num_bytes = sorted_expert_ids.numel() * (
        N * K * w2.element_size()
        + BLOCK_TILE_SIZE_M * K * down_in.element_size()
        + BLOCK_TILE_SIZE_M * N * gemm2_out.element_size()
    )
    pyhip.run_perftest(
        compiled,
        *args,
        num_name=f"moe-down-{weight_dtype}-{weight_quant_type}-{act_quant_type}",
        num_flops=num_flops,
        num_bytes=num_bytes,
        num_verbose=1,
    )

    cur_out = torch.sum(moedown_out, dim=1)

    pyhip.allclose(cur_out, gemm2_out, rtol=1e-2, atol=1e-2)


test("moe_down_data_bf16_no_no.pt")
#test("moe_down_data_fp8_ptpc_ptpc.pt")
#test("moe_down_data_fp8_per_tensor_ptpc.pt")
# test("moe_down_data_fp8_ptpc_ptpc.pt")


# for k in fx.range_constexpr(8):
#    rocdl.sched_vmem(1)
#    rocdl.sched_mfma(8)
# rocdl.sched_dswr(1)
# rocdl.sched_mfma(2)
# rocdl.sched_vmem(1)
