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
from flydsl._mlir.dialects.fly_rocdl import TargetAddressSpace
import torch

import pyhip
import pyhip.contrib.flydsl.helpers as fxh

# fxh.dump_ir(False)

_, stream = pyhip.set_device()


"""
python的灵活强大的语法可以简化代码编写, 但是flydsl的编写有点吃力之处：
 - 非常的不够pythonic, 没有简化语法，反而非常复杂，一个简单的操作要分很多步骤
 - 没有一套标准的API可以处理任意复杂的情况，几乎是任何算法都要重新实现helper封装

 比如，使用index索引tensor的其中一个mode进行读写，也就是gather/scatter操作
 在torch里面简单的使用tensor作为slice索引即可，但是flydsl里面的slice非常局限
 只有 None / arith-value 两种。

 理论上要做到这一点，需要引入一层额外的 code-generation，从python的语法中解析出语义
 映射到某种代码生成机制。

 再比如我们要使用coalescing方式加载数据到 fragment tensor 中的时候，也不应该如此复杂
 应该简单的使用pythonic的方式： 

    frag = fx.coalescing_loader( mem[任意slice语义] )       # frag内部自带了 layout映射关系
    mem[任意slice语义] = frag                               # 这个赋值中frag内部自带了 layout映射关系

    frag = fx.coalescing_loader_empty(shape, dtype)  # 
    
    frag 寄存器作为计算的中间存储

    partition 本质上

"""


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

    flyobj = fxh.FlyObjCache()

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

        flyobj.bid = e_idx

        arg_p_input = fxh.view_as_torch_tensor(p_input, (M, TOPK, K), weight_dtype)
        arg_p_output = fxh.view_as_torch_tensor(p_output, (M, TOPK, N))
        max_valid_id = fxh.view_as_torch_tensor(p_num_valid_ids, (1,), fx.Int32)[0]

        if e_idx * BLOCK_TILE_SIZE_M < max_valid_id:
            arg_p_sorted_ids = fxh.view_as_torch_tensor(
                p_sorted_ids + e_idx * BLOCK_TILE_SIZE_M, (BLOCK_TILE_SIZE_M,), fx.Int32
            )
            arg_p_sorted_weights = fxh.view_as_torch_tensor(
                p_sorted_weights + e_idx * BLOCK_TILE_SIZE_M,
                (BLOCK_TILE_SIZE_M,),
                fx.Float32,
            )
            expert_id = fxh.view_as_torch_tensor(p_sorted_expert_ids, (1,), fx.Int32)[
                e_idx
            ]

            # 16bytes/DW4
            element_num = 16 // (weight_dtype.width // 8)
            arg_p_weight = fx.make_view(
                fxh._as_ptr(p_weight, weight_dtype) + fx.Int64(expert_id * N * K),
                fx.make_layout(
                    ((16, N // 16), (element_num, K // element_num)),
                    ((element_num, 16 * K), (1, 16 * element_num)),
                ),
            )

            arg_p_weight = fx.rocdl.make_buffer_tensor(arg_p_weight, max_size=False)

            sorted_ids_buf = fx.rocdl.make_buffer_tensor(
                arg_p_sorted_ids, max_size=False
            )

            BLOCK_M = BLOCK_TILE_SIZE_M
            BLOCK_N = 64
            BLOCK_K = 64 // (weight_dtype.width // 8)

            # mask,base,shift, swizzle always in unit of 128b,
            swz_base = ((128 // weight_dtype.width) - 1).bit_length()
            swz = fx.SwizzleType.get(3, swz_base, 3)

            act_dtype = weight_dtype  # fp8 / bf16

            @fx.union
            class SharedStorage:
                A: fx.Array[act_dtype, BLOCK_M * K]
                C: fx.Array[fx.BFloat16, 2 * BLOCK_M * BLOCK_N]

            lds = fx.SharedAllocator().allocate(SharedStorage)
            ldsA0 = lds.A.peek().view(
                fx.make_composed_layout(fx.static(swz), fxh.torch_layout(BLOCK_M, K))
            )
            layoutC = fx.make_composed_layout(
                fx.static(swz),
                fx.make_ordered_layout((BLOCK_M, BLOCK_N, 2), (1, 0, 2)),
            )
            ldsC = lds.C.peek().view(layoutC)
            ldsCt = fx.select(ldsC, [1, 0, 2])

            # cp_atom = flyobj.get_universal_copy_atom(arg_p_input.dtype, 128)
            tcopy, cp_atom = flyobj.get_tiled_copy_coalesced_mn(
                ldsA0, copy_atom_bits=128, num_threads=256
            )
            for dst, row, col in fxh.all_element_of_tensors(
                ldsA0,
                fxh.make_1d_coord_tensor(ldsA0, 0, fx.get_iter(arg_p_sorted_ids)),
                fxh.make_1d_coord_tensor(ldsA0, 1, fx.make_int_tuple(0)),
                tiled_copy=tcopy,
            ):
                sorted_id = row[0].bitcast(fx.Uint32)
                topk = sorted_id >> 24
                # avoid using if brach in such small loop-body
                valid = topk < TOPK
                token_id = valid.select(sorted_id & 0xFFFFFF, 0)
                topk = valid.select(topk, 0)
                atom_A = fxh.atom_tensor(arg_p_input, (token_id, topk, col[0]), 128)
                fx.copy(cp_atom, atom_A, dst)
            fx.gpu.barrier()

            # (BLOCK_N, BLOCK_K, num_blocks_N, num_blocks_K)
            weight = fx.flat_divide(arg_p_weight, (BLOCK_N, BLOCK_K))
            ldsA = fx.flat_divide(ldsA0, (BLOCK_M, BLOCK_K))

            nBM = 1
            nBN = fxh.div_up(N, BLOCK_N)
            nBK = fxh.div_up(K, BLOCK_K)

            mm = flyobj.create_thr_mma(weight_dtype, (4, 1, 1))

            c_fake_tensor = fx.make_view(
                fx.get_iter(arg_p_input),
                fx.make_ordered_layout((BLOCK_N, BLOCK_M), (0, 1)),
            )
            fragC = [
                mm.make_fragment_C(c_fake_tensor),
                mm.make_fragment_C(c_fake_tensor),
            ]
            fragC_bf16 = fx.make_fragment_like(fragC[0], fx.BFloat16)

            frag_act = flyobj.load_tiled_mma_fragB(mm, ldsA, copy_atom_bits=128)
            fx.gpu.barrier()  # make sure all threads finished using ldsA (since it's reused by ldsC)

            arg_w_scale = None
            if const_expr(weight_quant_type == "per_tensor"):
                arg_w_scale = fx.make_view(
                    fxh._as_ptr(p_w_scale) + expert_id, fx.make_layout((N, 1), (0, 0))
                )
                arg_w_scale = fx.flat_divide(arg_w_scale, (BLOCK_N, 1))
            if const_expr(weight_quant_type == "ptpc"):
                arg_w_scale = fx.make_view(
                    fxh._as_ptr(p_w_scale) + expert_id * N,
                    fx.make_layout((N, 1), (1, 0)),
                )
                # (BLOCK_N, 1, num_block_N, 1)
                arg_w_scale = fx.flat_divide(arg_w_scale, (BLOCK_N, 1))

            arg_a_scale = None
            if const_expr(act_quant_type == "per_tensor"):
                arg_a_scale = fx.make_view(
                    fx.recast_iter(fx.Float32, fxh._as_ptr(p_a_scale)),
                    fx.make_layout((M, TOPK), (0, 0)),
                )
            if const_expr(act_quant_type == "ptpc"):
                arg_a_scale = fx.make_view(
                    fx.recast_iter(fx.Float32, fxh._as_ptr(p_a_scale)),
                    fx.make_layout((M, TOPK), (TOPK, 1)),
                )

            sorted_weights = fx.make_view(
                fx.get_iter(arg_p_sorted_weights),
                fx.make_layout((BLOCK_N, BLOCK_M), (0, 1)),
            )
            # load rows/token weights using C layout
            frag_sorted_weight = flyobj.load_tiled_mma_fragC(
                mm, sorted_weights, copy_atom_bits=32
            )

            if fx.const_expr(arg_a_scale is not None):
                """load & combine per-token scales with per-token weights, and store into lds.C"""
                cp_atom = flyobj.get_universal_copy_atom(p_a_scale.dtype, 32)
                coord_tensor = fx.make_view(
                    fx.get_iter(arg_p_sorted_ids),
                    fx.make_layout((BLOCK_N, BLOCK_M), (0, 1)),
                )
                frag_coord = flyobj.load_tiled_mma_fragC(
                    mm, coord_tensor, copy_atom_bits=32
                )
                frag_pt_scales = mm.make_fragment_C(coord_tensor)
                frag_pt_scalesr = flyobj.get_tiled_mma_retile(
                    mm, frag_pt_scales, "C", copy_atom=cp_atom
                )

                for dst, coord in fxh.all_elements(frag_pt_scalesr, frag_coord):
                    sorted_id = coord[0].bitcast(fx.Uint32)
                    atom_A = fxh.atom_tensor(
                        arg_a_scale,
                        (sorted_id & 0xFFFFFF, sorted_id >> 24),
                        32,
                    )
                    fx.copy(cp_atom, atom_A, dst)

                # combine per-token scales with per-token weights
                for frag_pt, frag_sw in fxh.all_elements(
                    frag_pt_scales, frag_sorted_weight
                ):
                    frag_pt.store(frag_pt.load() * frag_sw.load())

                frag_sorted_weight = frag_pt_scales

            def f32_to_bf16(x):
                round_bit = fx.Uint32(0x8000).ir_value().bitcast(fx.Float32.ir_type)
                return (
                    ((x + round_bit).bitcast(fx.Uint32) >> 16)
                    .to(fx.Uint16)
                    .bitcast(fx.BFloat16)
                )

            def gemm_compute(fragW, fragPCS, fragC):
                fragC.fill(0)
                for k in fx.range_constexpr(nBK):
                    fx.gemm(
                        mm,
                        fragC,
                        fragW[None, None, None, k],
                        frag_act[None, None, None, 0, k],
                        fragC,
                    )
                if fx.const_expr(fragPCS is not None):
                    for fc, fpc in fxh.all_elements(fragC, fragPCS):
                        fc.store(fc.load() * fpc.load())

            row_tensor = fx.make_view(
                fx.get_iter(arg_p_sorted_ids),
                fx.make_layout((BLOCK_M, BLOCK_N), (1, 0)),
            )
            col_tensor = fx.make_view(
                fx.make_int_tuple(0), fx.make_layout((BLOCK_M, N), (0, 1))
            )
            col_tensor = fx.flat_divide(col_tensor, (BLOCK_M, BLOCK_N))

            tcopyLDS, _ = flyobj.get_tiled_copy_coalesced_mn(
                ldsC[None, None, 0], copy_atom_bits=128, num_threads=256
            )

            thrv_ldsC = tcopyLDS.partition_S(ldsC)

            thrv_dst_col = tcopyLDS.partition_D(col_tensor)
            frag_row = fxh.load_fragment(tcopyLDS.partition_S(row_tensor))

            cp_atom_128b = flyobj.get_universal_copy_atom(fx.BFloat16, 128)

            copy_atom_ = flyobj.get_universal_copy_atom(fragC_bf16.dtype, 64)
            tcopy = flyobj.get_tiled_mma_copy(copy_atom_, mm, "C")
            fragC_bf16r = flyobj.get_retile(tcopy, fragC_bf16)

            thrv_ldsCt = flyobj.get_partition_D(tcopy, ldsCt)

            def postprocess_store2lds(fragC, ldsc_idx):
                for fc, fsw in fxh.all_elements(fragC, frag_sorted_weight):
                    fc.store(fc.load() * fsw.load())
                vec_f32 = fragC.load()
                fragC_bf16.store(f32_to_bf16(vec_f32))
                fx.copy(copy_atom_, fragC_bf16r, thrv_ldsCt[None, None, None, ldsc_idx])

            fragOut = fx.make_fragment_like(thrv_ldsC[None, None, None, 0])

            def postprocess_store2vmem(n, ldsc_idx):
                fx.copy(cp_atom_128b, thrv_ldsC[None, None, None, ldsc_idx], fragOut)

                for src, row, col in fxh.all_elements(
                    fragOut,
                    frag_row,
                    thrv_dst_col[None, None, None, 0, n],
                ):
                    sorted_id = row[0].bitcast(fx.Uint32)
                    topk = sorted_id >> 24
                    atom_C = fxh.atom_tensor(
                        arg_p_output, (sorted_id & 0xFFFFFF, topk, col[0]), 128
                    )
                    if fx.const_expr(1):
                        valid = topk < TOPK
                        dummy = llvm.inline_asm(
                            ir.Type.parse("i64"),
                            [
                                topk.ir_value(),
                                fx.ptrtoint(fx.get_iter(atom_C)).ir_value(),
                                src.load().ir_value(),
                            ],
                            f"v_cmp_lt_i32_e64 vcc, $1, {int(TOPK)}\n\t"
                            f"s_and_saveexec_b64 $0, vcc \n\t"
                            f"global_store_dwordx4 $2, $3, off\n\t"
                            f"s_or_b64 exec, exec, $0\n\t",
                            "=s,v,v,v,~{vcc}",
                            has_side_effects=True,
                        )
                    else:
                        if topk < TOPK:
                            fx.copy(cp_atom_128b, src, atom_C)

            """
            apply per-token scale & weights, cvt-bf16, write-fragC to LDS
            load fragC from LDS, write to global memory
            """

            def hot_loop_scheduler():
                """
                // to cross the SCHED_BARRIER during scheduling.
                //     MASK = 0x0000 0000: No instructions may be scheduled across SCHED_BARRIER.
                //     MASK = 0x0000 0001: ALL, non-memory, non-side-effect producing instructions may be
                //                         scheduled across SCHED_BARRIER, i.e. allow ALU instructions to pass.
                //     MASK = 0x0000 0002: VALU instructions may be scheduled across SCHED_BARRIER.
                //     MASK = 0x0000 0004: SALU instructions may be scheduled across SCHED_BARRIER.
                //     MASK = 0x0000 0008: MFMA/WMMA instructions may be scheduled across SCHED_BARRIER.
                //     MASK = 0x0000 0010: ALL VMEM instructions may be scheduled across SCHED_BARRIER.
                //     MASK = 0x0000 0020: VMEM read instructions may be scheduled across SCHED_BARRIER.
                //     MASK = 0x0000 0040: VMEM write instructions may be scheduled across SCHED_BARRIER.
                //     MASK = 0x0000 0080: ALL DS instructions may be scheduled across SCHED_BARRIER.
                //     MASK = 0x0000 0100: ALL DS read instructions may be scheduled accoss SCHED_BARRIER.
                //     MASK = 0x0000 0200: ALL DS write instructions may be scheduled across SCHED_BARRIER.
                """
                num_mfma_inst = (BLOCK_M // 16) * (
                    K // (16 if weight_dtype.width == 16 else 32)
                )
                num_stores = (BLOCK_M // (256 // (BLOCK_N//8)))
                num_loads = K // ((4 * 8) if weight_dtype.width == 16 else (4 * 16))

                # print(num_loads, num_stores, num_mfma_inst)
                nloads = num_loads
                nstores = num_stores
                mfma_step = num_mfma_inst // (nloads + nstores)

                nmfma = num_mfma_inst - mfma_step * (nloads + nstores)
                if nmfma > 0:
                    fx.rocdl.sched_mfma(nmfma)

                for _ in fx.range_constexpr(nloads):
                    fx.rocdl.sched_mfma(mfma_step)
                    fx.rocdl.sched_group_barrier(0x10, 1, 0)

                for _ in fx.range_constexpr(nstores):
                    fx.rocdl.sched_mfma(mfma_step)
                    fx.rocdl.sched_group_barrier(0x10, 1, 0)

                fx.rocdl.sched_barrier(0)

            # prelog
            frag_weights = [None, None]
            frag_pc_scales = [None, None]
            frag_weights[0] = flyobj.load_tiled_mma_fragA(
                mm, weight, [None, None, 0, None]
            )
            if fx.const_expr(arg_w_scale is not None):
                frag_pc_scales[0] = flyobj.load_tiled_mma_fragC(
                    mm,
                    arg_w_scale,
                    [None, None, 0, 0],
                    copy_atom_bits=32 if weight_quant_type == "per_tensor" else 128,
                )

            # prelog
            gemm_compute(frag_weights[0], frag_pc_scales[0], fragC[0])
            frag_weights[1] = flyobj.load_tiled_mma_fragA(
                mm, weight, [None, None, 1, None]
            )
            if fx.const_expr(arg_w_scale is not None):
                frag_pc_scales[1] = flyobj.load_tiled_mma_fragC(
                    mm,
                    arg_w_scale,
                    [None, None, 1, 0],
                    copy_atom_bits=32 if weight_quant_type == "per_tensor" else 128,
                )

            postprocess_store2lds(fragC[0], 0)
            fx.gpu.barrier()
            """
            sched_group_barrier only search independent instructions within current basic-block
            and syn-threads/barrier is a boundary of basic-block, we need to respect this rules
            and clearly organize instructions into natural basic-blocks and apply sched_group_barrier
            to each of them:
                basic-block1: post-process & LDS write | prefetch part of next weight block
                    wait barrier
                basic-block2: gemm compute | prefetch part of next weight block | LDS-read + global-write
            """
            for n, state in range(0, nBN - 2, 2, init=[]):
                fxh.asm_mark("aaa")
                postprocess_store2vmem(n, 0)
                flyobj.load_tiled_mma_fragA(
                    mm, weight, [None, None, n + 2, None], frag_weights[0]
                )
                if fx.const_expr(
                    arg_w_scale is not None and weight_quant_type != "per_tensor"
                ):
                    flyobj.load_tiled_mma_fragC(
                        mm, arg_w_scale, [None, None, n + 2, 0], frag_pc_scales[0]
                    )
                gemm_compute(frag_weights[1], frag_pc_scales[1], fragC[1])
                postprocess_store2lds(fragC[1], 1)

                hot_loop_scheduler()
                fx.gpu.barrier()

                fxh.asm_mark("bbb")

                postprocess_store2vmem(n + 1, 1)
                flyobj.load_tiled_mma_fragA(
                    mm, weight, [None, None, n + 3, None], frag_weights[1]
                )
                # fxh.asm_mark("ccc")

                if fx.const_expr(
                    arg_w_scale is not None and weight_quant_type != "per_tensor"
                ):
                    flyobj.load_tiled_mma_fragC(
                        mm, arg_w_scale, [None, None, n + 3, 0], frag_pc_scales[1]
                    )
                gemm_compute(frag_weights[0], frag_pc_scales[0], fragC[0])
                postprocess_store2lds(fragC[0], 0)

                hot_loop_scheduler()
                fx.gpu.barrier()

            # epilogue
            postprocess_store2vmem(nBN - 2, 0)
            gemm_compute(frag_weights[1], frag_pc_scales[1], fragC[1])
            postprocess_store2lds(fragC[1], 1)
            fx.gpu.barrier()
            postprocess_store2vmem(nBN - 1, 1)

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
        num_n_blocks = fxh.div_up(N, BLOCK_TILE_SIZE_N)
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
        num_copies=1,
    )

    cur_out = torch.sum(moedown_out, dim=1)

    pyhip.allclose(cur_out, gemm2_out, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    test("moe_down_data_bf16_no_no.pt")
    # test("moe_down_data_fp8_ptpc_ptpc.pt")
    # test("moe_down_data_fp8_per_tensor_ptpc.pt")
    # test("moe_down_data_fp8_ptpc_ptpc.pt")

    # for k in fx.range_constexpr(8):
    #    rocdl.sched_vmem(1)
    #    rocdl.sched_mfma(8)
    # rocdl.sched_dswr(1)
    # rocdl.sched_mfma(2)
    # rocdl.sched_vmem(1)
