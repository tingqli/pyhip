import argparse
import os

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

def compile_gemm(TILE_M, TILE_N, N, K, alg='splitk'):
    gpu_arch = get_rocm_arch()
    is_gfx942 = str(gpu_arch).startswith("gfx942")

    TILE_K = 64
    if alg == 'splitk':
        c_lds_size = TILE_M * TILE_N * 4
        @fx.struct
        class SharedStorage:
            C_lds: fx.Array[fx.Float32, c_lds_size, 16]
    else:
        # a和b的lds将被均分为两部分：at, ab, bl, br，每个小块大小为TILE_M/2*TILE_K
        a_lds_size_half = TILE_M // 2 * TILE_K
        b_lds_size_half = TILE_N // 2 * TILE_K
        @fx.struct
        class SharedStorage:
            at_lds: fx.Array[fx.BFloat16, a_lds_size_half, 16]
            ab_lds: fx.Array[fx.BFloat16, a_lds_size_half, 16]
            bl_lds: fx.Array[fx.BFloat16, b_lds_size_half, 16]
            br_lds: fx.Array[fx.BFloat16, b_lds_size_half, 16]

    @flyc.kernel
    def gemm_splitk(arg_c_: fx.Tensor,
                    arg_a_: fx.Tensor,
                    arg_b_: fx.Tensor,
                    M: int):
        tid = fx.thread_idx.x
        blk_x, blk_y, _ = fx.block_idx
        arg_a = fx.Tensor(fx.make_view(
            fx.get_iter(arg_a_),
            fx.make_layout((M, K), (K, 1))
        ))
        # pre-shuffled
        arg_b = fx.Tensor(fx.make_view(
            fx.get_iter(arg_b_),
            # layout分成两部分：第一部分(m, n)描述一个wave内部划分为16行，(8列每组)x4组；第二部分为重复第一部分的次数
            # shape: (16, (8, 4)),   (N//16, K//32)
            # stride:(8,  (1, 128)), (K*16, 512))
            # 重新排列为第0维、第1维
            fx.make_layout(((16, N // 16), (8, 4, K // 32)), ((8, 16 * K), (1, 128, 512)))
        ))
        arg_c = fx.Tensor(fx.make_view(
            fx.get_iter(arg_c_),
            fx.make_layout((M, N), (N, 1))
        ))
        a_tensor = fx.rocdl.make_buffer_tensor(arg_a, max_size=False)
        b_tensor = fx.rocdl.make_buffer_tensor(arg_b, max_size=False)
        c_tensor = fx.rocdl.make_buffer_tensor(arg_c, max_size=False)
        # shape: [m_in_tile, k_in_tile, k_tile]
        a_tile = fx.flat_divide(a_tensor, fx.make_tile(TILE_M, TILE_K * 4))[None, None, blk_x, None]
        # shape: [n_in_tile, k_in_tile, k_tile]
        b_tile = fx.flat_divide(b_tensor, fx.make_tile(TILE_N, TILE_K * 4))[None, None, blk_y, None]
        # shape: [m_in_tile, n_in_tile]
        c_tile = fx.flat_divide(c_tensor, fx.make_tile(TILE_M, TILE_N))[None, None, blk_x, blk_y]
        cp_atom_r = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16)

        tiled_mma = fx.make_tiled_mma(
            fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, fx.BFloat16)),
            # splitk, 4个wave分布在K维度
            fx.make_layout((1, 1, 4), (0, 0, 1)),
            # K维度((4*2)*4)*4，线程分布依次为：连续的4点为一个lane，跳过8点为一组并重复4组*4wave，最后为重复第二次调用
            fx.make_tile(None, None, fx.make_layout((4, 4 * 4, 2), (1, 8, 4)))
        )
        # fx.utils.print_typst(tiled_mma.tv_layout_A_tiled, file="layout_tiled_mma.typ")
        a_tiled_thr = fx.make_tiled_copy_A(cp_atom_r, tiled_mma).get_slice(tid)
        b_tiled_thr = fx.make_tiled_copy_B(cp_atom_r, tiled_mma).get_slice(tid)

        a_tensor_thr = a_tiled_thr.partition_S(a_tile)
        b_tensor_thr = b_tiled_thr.partition_S(b_tile)
        # 等价于如下显示生成的layout
        if const_expr(0):
            g2r_ab = fx.make_tiled_copy(cp_atom_r,
                                        # tv，第一部分为(16x4)x4wave=256 线程分布；第二部分为value的布局；坐标换算时是以column first的方式进行
                                        fx.make_layout(((16, 4 * 4), 8), ((1, 128), 16)),
                                        fx.make_tile(16, 32 * 4))
            ab_thr = g2r_ab.get_slice(tid)
            # shape: [v, m_rep, k_rep, k_tile]
            a_tensor_thr = ab_thr.partition_S(a_tile)
            # shape: [v, n_rep, k_rep, k_tile]
            b_tensor_thr = ab_thr.partition_S(b_tile)

        # tiled_mma.make_fragment_A 与 fx.make_fragment_like(a_tensor_thr[None, None, None, 0])不等价，区别：
        #  layout按照mma_atom要求切分，比如value维度为4（mfma_16x16x16时），但a_tensor_thr是按照tiled_copy切分的，value很可能为8
        a_frag = tiled_mma.make_fragment_A(a_tile[None, None, 0])
        b_frag = tiled_mma.make_fragment_B(b_tile[None, None, 0])
        # 1, 矩阵乘法从a*b变为b*a，c的layout不能直接用make_fragment_C
        # 2, splitk需要reduce，因此c_tile每个wave都有一份，使用make_tiled_copy来构造layout但最终又不用于copy，并且还要特殊处理broadcast。
        #    下面使用make_tiled_copy是错误的示例，没有考虑broadcast；简单的做法是基于tiledmma来构造layout
        #    layout_c = fx.make_tiled_copy(fx.make_copy_atom(fx.rocdl.BufferCopy64b(), fx.BFloat16),
        #                            fx.make_layout((16, (4, 4)), (1, (16, 64))),
        #                            fx.make_tile(16, 16))
        c_frag = fx.make_rmem_tensor(
            # 未交换a*b时16x16排布规则：4个连续的M为dim0，步长为1，然后N方向跳过16个元素为dim2，步长为4，最后在M方向重复4次，为dim1，步长为64；
            # shape(c=b*a): [value, n_req, m_req]
            fx.make_layout(((4, 1), TILE_N // 16, TILE_M // 16), ((1, 0), TILE_M // 16 * 4, 4)), fx.Float32)
        c_frag.store(Vec.filled(TILE_M * TILE_N // 64, 0, fx.Float32))

        a_frag_retile = a_tiled_thr.retile(a_frag)
        b_frag_retile = b_tiled_thr.retile(b_frag)

        acc_init = c_frag.load()
        loop_start = fx.Index(0)
        loop_end = fx.Index(K // TILE_K // 4)
        loop_step = fx.Index(1)
        if const_expr(0):
            # 完全展开
            for k in range_constexpr(0, K // TILE_K // 4):
                fx.copy(cp_atom_r, a_tensor_thr[None, None, None, k], a_frag_retile)
                fx.copy(cp_atom_r, b_tensor_thr[None, None, None, k], b_frag_retile)
                for sub_k in range_constexpr(TILE_K // 32):
                    fx.gemm(tiled_mma, c_frag, b_frag[None, None, (None, sub_k)], a_frag[None, None, (None, sub_k)], c_frag)
        else:
            for k, state in range(loop_start, loop_end, loop_step, init=[acc_init]):
                c_frag.store(state[0])
                k_i32 = fx.Int32(k)
                fx.copy(cp_atom_r, a_tensor_thr[None, None, None, k_i32], a_frag_retile)
                fx.copy(cp_atom_r, b_tensor_thr[None, None, None, k_i32], b_frag_retile)
                for sub_k in range_constexpr(TILE_K // 32):
                    fx.gemm(tiled_mma, c_frag, b_frag[None, None, (None, sub_k)], a_frag[None, None, (None, sub_k)], c_frag)
                results = yield [c_frag.load()]
            c_frag.store(results)
        # [v, n, m] -> [v, m, n]
        c_frag = fx.select(c_frag, [0, 2, 1])

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        swz = fx.SwizzleType.get(3, 3, 3)
        c_lds = fx.make_view(lds.C_lds,
                             fx.make_composed_layout(fx.static(swz), fx.make_ordered_layout((TILE_M * 4, TILE_N), order=(1, 0))))
        cp_atom_lds = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Float32)
        c_tiled_lds = fx.make_tiled_copy(cp_atom_lds,
                                         # 线程划分: 16x4，4 wave切分m方向64行，每个线程写入n方向连续4个点
                                         fx.make_layout(((16, 4, 4), 4), ((1, 256, 16), 64)),
                                         fx.make_tile(16 * 4, 16))
        #fx.utils.print_typst(c_tiled_lds, file="layout.typ")
        c_tensor_thr_lds_w = c_tiled_lds.get_slice(tid).partition_D(c_lds)
        fx.copy(cp_atom_lds, c_frag, c_tensor_thr_lds_w)
        gpu.barrier()

        # 一次写出凑够128字节，128/2=64个元素
        assert TILE_N % 64 == 0, "TILE_N needs to be multiple of 64 for coalesced store(64x2=128 bytes) to global memory"

        c_tiled_lds = fx.make_tiled_copy(cp_atom_lds,
                                         # 线程划分: 4x16，4 wave切分m方向16行，每个线程读取n方向连续4个点，在m方向重复4次
                                         fx.make_layout(((16, 4, 4), (4, 4)), ((256, 1, 4), (64, 16))),
                                         fx.make_tile(16 * 4, 16 * 4))
        #fx.utils.print_typst(c_tiled_lds, file="layout-c-tiled.typ")
        c_tensor_thr_lds_r = c_tiled_lds.get_slice(tid).partition_S(c_lds)
        c_frag_reduce =fx.make_fragment_like(c_tensor_thr_lds_r)
        fx.copy(cp_atom_lds, c_tensor_thr_lds_r, c_frag_reduce)
        acc = c_frag_reduce[(None, 0), None, None].load()
        for i in range_constexpr(1, 4):
            acc += c_frag_reduce[(None, i), None, None].load()
        # if tid < 16:
        #     fx.printf("---------tid = {} acc[0]={}", tid, acc)

        c_frag_bf16 = fx.make_fragment_like(c_frag_reduce[(None, 0), None, None], dtype=fx.BFloat16)
        # 快速计算f32->bf16
        round_bit = fx.Uint32(0x8000).ir_value().bitcast(fx.Float32.ir_type)
        acc = ((acc + round_bit).bitcast(fx.Uint32) >> 16).to(fx.Uint16).bitcast(fx.BFloat16)
        # acc = acc.to(fx.BFloat16)
        c_frag_bf16.store(acc)

        cp_atom_w = fx.make_copy_atom(fx.rocdl.BufferCopy64b(), fx.BFloat16)
        c_tiled_g = fx.make_tiled_copy(cp_atom_w,
                                       # 线程划分: 4x16，4 wave切分m方向16行，每个线程写入n方向连续4个点
                                       fx.make_layout(((16, 4, 4), 4), ((64, 1, 4), 16)),
                                       fx.make_tile(16, 16 * 4))
        # fx.utils.print_typst(c_tiled_g, file="layout-c-out.typ")
        c_tensor_thr_g = c_tiled_g.get_slice(tid).partition_D(c_tile)
        fx.copy(cp_atom_w, c_tiled_g.get_slice(tid).retile(c_frag_bf16), c_tensor_thr_g)

    # copy from https://github.com/ROCm/gfx9-gluon-tutorials/blob/main/kernels/gemm/a16w16/v8_beyond_hotloop/matmul_kernel.py
    def get_pids(
        pid,
        M,
        N,
        BM,
        BN,
        GRID_MN,
        NUM_XCDS,
        GROUP_SIZE_M,
    ):
        num_pid_m = (M + BM - 1) // BM
        num_pid_n = (N + BN - 1) // BN

        if const_expr(NUM_XCDS != 1):
            ## pid remapping on xcds
            # Number of pids per XCD in the new arrangement
            pids_per_xcd = (GRID_MN + NUM_XCDS - 1) // NUM_XCDS
            # When GRID_MN cannot divide NUM_XCDS, some xcds will have
            # pids_per_xcd pids, the other will have pids_per_xcd - 1 pids.
            # We calculate the number of xcds that have pids_per_xcd pids as
            # tall_xcds
            tall_xcds = GRID_MN % NUM_XCDS
            # TODO
            # tall_xcds = NUM_XCDS if tall_xcds == 0 else tall_xcds
            tall_xcds = (tall_xcds == 0).select(NUM_XCDS, tall_xcds)
            # Compute current XCD and local pid within the XCD
            xcd = pid % NUM_XCDS
            local_pid = pid // NUM_XCDS
            # Calculate new pid based on the new grouping
            # Note that we need to consider the following two cases:
            # 1. the current pid is on a tall xcd
            # 2. the current pid is on a short xcd
            # TODO
            # if xcd < tall_xcds:
            #     pid = xcd * pids_per_xcd + local_pid
            # else:
            #     pid = tall_xcds * pids_per_xcd + (xcd - tall_xcds) * (pids_per_xcd - 1) + local_pid
            pid = (xcd < tall_xcds).select(xcd * pids_per_xcd + local_pid,
                                           tall_xcds * pids_per_xcd + (xcd - tall_xcds) * (pids_per_xcd - 1) + local_pid)

        if const_expr(GROUP_SIZE_M == 1):
            pid_m = pid // num_pid_n
            pid_n = pid % num_pid_n
        else:
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            group_id = pid // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            # TODO
            #group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            group_size_m = (num_pid_m - first_pid_m < GROUP_SIZE_M).select(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
            pid_n = (pid % num_pid_in_group) // group_size_m

        return pid_m, pid_n

    @flyc.kernel
    def gemm_4wave(arg_c_: fx.Tensor,
                   arg_a_: fx.Tensor,
                   arg_b_: fx.Tensor,
                   M: int):
        tid = fx.thread_idx.x
        # 308 has 4 xcds
        blk_x, blk_y = get_pids(fx.block_idx.x, M, N, TILE_M, TILE_N, fx.grid_dim.x, 4, 4)
        arg_a = fx.Tensor(fx.make_view(
            fx.get_iter(arg_a_),
            fx.make_layout((M, K), (K, 1))
        ))
        # pre-shuffled
        arg_b = fx.Tensor(fx.make_view(
            fx.get_iter(arg_b_),
            # layout分成两部分：第一部分(m, n)描述一个wave内部划分为16行，(8列每组)x4组；第二部分为重复第一部分的次数
            # shape: (16, (8, 4)),   (N//16, K//32)
            # stride:(8,  (1, 128)), (K*16, 512))
            # 重新排列为第0维、第1维
            fx.make_layout(((16, N // 16), (8, 4, K // 32)), ((8, 16 * K), (1, 128, 512)))
        ))
        arg_c = fx.Tensor(fx.make_view(
            fx.get_iter(arg_c_),
            fx.make_layout((M, N), (N, 1))
        ))
        a_tensor = fx.rocdl.make_buffer_tensor(arg_a, max_size=False)
        b_tensor = fx.rocdl.make_buffer_tensor(arg_b, max_size=False)
        c_tensor = fx.rocdl.make_buffer_tensor(arg_c, max_size=False)
        # shape: [m_in_tile, k_in_tile, k_tile]
        at_tile = fx.flat_divide(a_tensor, fx.make_tile(TILE_M // 2, TILE_K))[None, None, blk_x * 2 + 0, None]
        ab_tile = fx.flat_divide(a_tensor, fx.make_tile(TILE_M // 2, TILE_K))[None, None, blk_x * 2 + 1, None]
        # shape: [n_in_tile, k_in_tile, k_tile]
        bl_tile = fx.flat_divide(b_tensor, fx.make_tile(TILE_N // 2, TILE_K))[None, None, blk_y * 2 + 0, None]
        br_tile = fx.flat_divide(b_tensor, fx.make_tile(TILE_N // 2, TILE_K))[None, None, blk_y * 2 + 1, None]
        # shape: [m_in_tile, n_in_tile]
        c_tl_tile = fx.flat_divide(c_tensor, fx.make_tile(TILE_M // 2, TILE_N // 2))[None, None, 2 * blk_x + 0, 2 * blk_y + 0]
        c_tr_tile = fx.flat_divide(c_tensor, fx.make_tile(TILE_M // 2, TILE_N // 2))[None, None, 2 * blk_x + 0, 2 * blk_y + 1]
        c_bl_tile = fx.flat_divide(c_tensor, fx.make_tile(TILE_M // 2, TILE_N // 2))[None, None, 2 * blk_x + 1, 2 * blk_y + 0]
        c_br_tile = fx.flat_divide(c_tensor, fx.make_tile(TILE_M // 2, TILE_N // 2))[None, None, 2 * blk_x + 1, 2 * blk_y + 1]

        # memory->lds layout
        buf_cp_atom_r = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16)
        assert TILE_M % 32 == 0 and TILE_N % 32 == 0, "TILE_M and TILE_N need to be multiple of 32 for loading scheme: (8x4wave)x8"
        ab_mem_cp_layout_g2r = fx.make_tiled_copy(buf_cp_atom_r,
                                                  # 线程划分[8 * 4 wave, 8]
                                                  fx.make_layout(((8, 8, 4), 8), ((256, 1, 8), 32)),
                                                  fx.make_tile(8 * 4, TILE_K))
        # fx.utils.print_typst(ab_mem_cp_layout_g2r, file="layout-ab_mem_cp_layout_g2r.typ")
        at_mem_tensor_thr = ab_mem_cp_layout_g2r.get_slice(tid).partition_S(at_tile)
        ab_mem_tensor_thr = ab_mem_cp_layout_g2r.get_slice(tid).partition_S(ab_tile)
        bl_mem_tensor_thr = ab_mem_cp_layout_g2r.get_slice(tid).partition_S(bl_tile)
        br_mem_tensor_thr = ab_mem_cp_layout_g2r.get_slice(tid).partition_S(br_tile)
        at_cp_frag = fx.make_fragment_like(at_mem_tensor_thr[None, None, None, 0])
        ab_cp_frag = fx.make_fragment_like(ab_mem_tensor_thr[None, None, None, 0])
        bl_cp_frag = fx.make_fragment_like(bl_mem_tensor_thr[None, None, None, 0])
        br_cp_frag = fx.make_fragment_like(br_mem_tensor_thr[None, None, None, 0])

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        swz = fx.SwizzleType.get(3, 3, 3)
        at_lds = fx.make_view(lds.at_lds, fx.make_composed_layout(fx.static(swz), fx.make_ordered_layout((TILE_M // 2, TILE_K), order=(1, 0))))
        ab_lds = fx.make_view(lds.ab_lds, fx.make_composed_layout(fx.static(swz), fx.make_ordered_layout((TILE_M // 2, TILE_K), order=(1, 0))))
        bl_lds = fx.make_view(lds.bl_lds, fx.make_composed_layout(fx.static(swz), fx.make_ordered_layout((TILE_N // 2, TILE_K), order=(1, 0))))
        br_lds = fx.make_view(lds.br_lds, fx.make_composed_layout(fx.static(swz), fx.make_ordered_layout((TILE_N // 2, TILE_K), order=(1, 0))))

        uni_cp_atom = fx.make_copy_atom(fx.UniversalCopy128b(), fx.BFloat16)
        ab_lds_cp_layout_g2r = fx.make_tiled_copy(uni_cp_atom,
                                                  # 线程划分[8 * 4 wave, 8]
                                                  fx.make_layout(((8, 8, 4), 8), ((256, 1, 8), 32)),
                                                  fx.make_tile(8 * 4, TILE_K))
        at_lds_tensor_thr_w = ab_lds_cp_layout_g2r.get_slice(tid).partition_D(at_lds)
        ab_lds_tensor_thr_w = ab_lds_cp_layout_g2r.get_slice(tid).partition_D(ab_lds)
        bl_lds_tensor_thr_w = ab_lds_cp_layout_g2r.get_slice(tid).partition_D(bl_lds)
        br_lds_tensor_thr_w = ab_lds_cp_layout_g2r.get_slice(tid).partition_D(br_lds)

        # lds -> reg layout
        tiled_mma = fx.make_tiled_mma(
            fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, fx.BFloat16)),
            # wave 为2x2
            fx.make_layout((2, 2, 1), (1, 2, 0)),
            # K维度重复两次
            fx.make_tile(None, None, fx.make_layout((4, 4, 2), (1, 8, 4)))
        )
        # fx.utils.print_typst(tiled_mma, file="layout_tiled_mma.typ")

        at_lds_tensor_thr_r = fx.make_tiled_copy_A(uni_cp_atom, tiled_mma).get_slice(tid).partition_S(at_lds)
        ab_lds_tensor_thr_r = fx.make_tiled_copy_A(uni_cp_atom, tiled_mma).get_slice(tid).partition_S(ab_lds)
        bl_lds_tensor_thr_r = fx.make_tiled_copy_B(uni_cp_atom, tiled_mma).get_slice(tid).partition_S(bl_lds)
        br_lds_tensor_thr_r = fx.make_tiled_copy_B(uni_cp_atom, tiled_mma).get_slice(tid).partition_S(br_lds)

        at_frag = tiled_mma.make_fragment_A(at_lds)
        ab_frag = tiled_mma.make_fragment_A(ab_lds)
        bl_frag = tiled_mma.make_fragment_B(bl_lds)
        br_frag = tiled_mma.make_fragment_B(br_lds)

        at_frag_retile = fx.make_tiled_copy_A(uni_cp_atom, tiled_mma).get_slice(tid).retile(at_frag)
        ab_frag_retile = fx.make_tiled_copy_A(uni_cp_atom, tiled_mma).get_slice(tid).retile(ab_frag)
        bl_frag_retile = fx.make_tiled_copy_B(uni_cp_atom, tiled_mma).get_slice(tid).retile(bl_frag)
        br_frag_retile = fx.make_tiled_copy_B(uni_cp_atom, tiled_mma).get_slice(tid).retile(br_frag)

        c_tl_frag = fx.select(tiled_mma.make_fragment_C(c_tl_tile), [0, 2, 1])
        c_tr_frag = fx.select(tiled_mma.make_fragment_C(c_tr_tile), [0, 2, 1])
        c_bl_frag = fx.select(tiled_mma.make_fragment_C(c_bl_tile), [0, 2, 1])
        c_br_frag = fx.select(tiled_mma.make_fragment_C(c_br_tile), [0, 2, 1])

        c_tl_frag.store(Vec.filled(TILE_M * TILE_N // 64 // 4, 0, fx.Float32))
        c_tr_frag.store(Vec.filled(TILE_M * TILE_N // 64 // 4, 0, fx.Float32))
        c_bl_frag.store(Vec.filled(TILE_M * TILE_N // 64 // 4, 0, fx.Float32))
        c_br_frag.store(Vec.filled(TILE_M * TILE_N // 64 // 4, 0, fx.Float32))

        acc_init = [c_tl_frag.load(), c_tr_frag.load(), c_bl_frag.load(), c_br_frag.load()]

        # prefetch
        # gr0: all
        fx.copy(buf_cp_atom_r, bl_mem_tensor_thr[None, None, None, 0], bl_cp_frag)
        fx.copy(buf_cp_atom_r, at_mem_tensor_thr[None, None, None, 0], at_cp_frag)
        fx.copy(buf_cp_atom_r, ab_mem_tensor_thr[None, None, None, 0], ab_cp_frag)
        fx.copy(buf_cp_atom_r, br_mem_tensor_thr[None, None, None, 0], br_cp_frag)
        # lw0: all
        fx.copy(uni_cp_atom, bl_cp_frag, bl_lds_tensor_thr_w)
        fx.copy(uni_cp_atom, at_cp_frag, at_lds_tensor_thr_w)
        fx.copy(uni_cp_atom, ab_cp_frag, ab_lds_tensor_thr_w)
        fx.copy(uni_cp_atom, br_cp_frag, br_lds_tensor_thr_w)

        # gr1: all
        fx.copy(buf_cp_atom_r, bl_mem_tensor_thr[None, None, None, 1], bl_cp_frag)
        fx.copy(buf_cp_atom_r, at_mem_tensor_thr[None, None, None, 1], at_cp_frag)
        fx.copy(buf_cp_atom_r, ab_mem_tensor_thr[None, None, None, 1], ab_cp_frag)
        fx.copy(buf_cp_atom_r, br_mem_tensor_thr[None, None, None, 1], br_cp_frag)
        gpu.barrier()

        # lr: bl0, at0
        fx.copy(uni_cp_atom, bl_lds_tensor_thr_r, bl_frag_retile)
        fx.copy(uni_cp_atom, at_lds_tensor_thr_r, at_frag_retile)

        def hot_loop_scheduler():
            if const_expr(is_gfx942):
                for _ in range_constexpr(4):
                    rocdl.sched_dswr(1)
                    rocdl.sched_mfma(4)
                for _ in range_constexpr(4):
                    rocdl.sched_vmem(1)
                    rocdl.sched_mfma(4)
                for _ in range_constexpr(8):
                    rocdl.sched_mfma(1)
                    rocdl.sched_dsrd(1)
                rocdl.sched_mfma(64 - 16 - 16 - 8)

        assert K // TILE_K >= 2, "this kernel requires at least 2 iterations"
        for k, state in range(0, K // TILE_K - 0, 1, init=acc_init):
            c_tl_frag.store(state[0])
            c_tr_frag.store(state[1])
            c_bl_frag.store(state[2])
            c_br_frag.store(state[3])
            k_i32 = fx.Int32(k)

            # bl0 @ at0
            for sub_k in range_constexpr(TILE_K // 32):
                fx.gemm(tiled_mma, c_tl_frag, bl_frag[None, None, (None, sub_k)], at_frag[None, None, (None, sub_k)], c_tl_frag)
            # lw: bl1
            fx.copy(uni_cp_atom, bl_cp_frag, bl_lds_tensor_thr_w)
            # gr: bl2
            fx.copy(buf_cp_atom_r, bl_mem_tensor_thr[None, None, None, k_i32 + 2], bl_cp_frag)
            # lr: ab0
            gpu.barrier()
            fx.copy(uni_cp_atom, ab_lds_tensor_thr_r, ab_frag_retile)
            hot_loop_scheduler()
            rocdl.sched_barrier(0)

            # bl0 @ ab0
            for sub_k in range_constexpr(TILE_K // 32):
                fx.gemm(tiled_mma, c_bl_frag, bl_frag[None, None, (None, sub_k)], ab_frag[None, None, (None, sub_k)], c_bl_frag)
            # lw: at1
            fx.copy(uni_cp_atom, at_cp_frag, at_lds_tensor_thr_w)
            # gr: at2
            fx.copy(buf_cp_atom_r, at_mem_tensor_thr[None, None, None, k_i32 + 2], at_cp_frag)
            # lr: br0
            gpu.barrier()
            fx.copy(uni_cp_atom, br_lds_tensor_thr_r, br_frag_retile)
            hot_loop_scheduler()
            rocdl.sched_barrier(1)

            # br0 @ at0
            for sub_k in range_constexpr(TILE_K // 32):
                fx.gemm(tiled_mma, c_tr_frag, br_frag[None, None, (None, sub_k)], at_frag[None, None, (None, sub_k)], c_tr_frag)
            # lw: ab1
            fx.copy(uni_cp_atom, ab_cp_frag, ab_lds_tensor_thr_w)
            # gr: ab2
            fx.copy(buf_cp_atom_r, ab_mem_tensor_thr[None, None, None, k_i32 + 2], ab_cp_frag)
            # lr: bl1
            gpu.barrier()
            fx.copy(uni_cp_atom, bl_lds_tensor_thr_r, bl_frag_retile)
            hot_loop_scheduler()
            rocdl.sched_barrier(2)

            # br0 @ ab0
            for sub_k in range_constexpr(TILE_K // 32):
                fx.gemm(tiled_mma, c_br_frag, br_frag[None, None, (None, sub_k)], ab_frag[None, None, (None, sub_k)], c_br_frag)
            # lw: br1
            fx.copy(uni_cp_atom, br_cp_frag, br_lds_tensor_thr_w)
            # gr: br2
            fx.copy(buf_cp_atom_r, br_mem_tensor_thr[None, None, None, k_i32 + 2], br_cp_frag)
            # lr: at1
            gpu.barrier()
            fx.copy(uni_cp_atom, at_lds_tensor_thr_r, at_frag_retile)
            hot_loop_scheduler()
            rocdl.sched_barrier(3)

            results = yield [c_tl_frag.load(), c_tr_frag.load(), c_bl_frag.load(), c_br_frag.load()]

        c_tl_frag.store(results[0])
        c_tr_frag.store(results[1])
        c_bl_frag.store(results[2])
        c_br_frag.store(results[3])        
        c_tl_frag = fx.select(c_tl_frag, [0, 2, 1])
        c_tr_frag = fx.select(c_tr_frag, [0, 2, 1])
        c_bl_frag = fx.select(c_bl_frag, [0, 2, 1])
        c_br_frag = fx.select(c_br_frag, [0, 2, 1])

        c_frag_bf16 = fx.make_fragment_like(c_tl_frag, dtype=fx.BFloat16)
        round_bit = fx.Uint32(0x8000).ir_value().bitcast(fx.Float32.ir_type)
        buf_atom_w64 = fx.make_copy_atom(fx.rocdl.BufferCopy64b(), fx.BFloat16)
        c_layout_w = fx.make_tiled_copy(buf_atom_w64,
                                        # wave先M，再N（匹配mma中的设置）
                                        fx.make_layout(((16, 4, 2, 2), 4), ((1, 128, 16, 512), 32)),
                                        fx.make_tile(32, 32))
        # fx.utils.print_typst(c_layout_w, file="layout-c-out.typ")

        # acc = acc.to(fx.BFloat16)
        c_frag_bf16.store(((c_tl_frag.load() + round_bit).bitcast(fx.Uint32) >> 16).to(fx.Uint16).bitcast(fx.BFloat16))
        fx.copy(buf_atom_w64, c_layout_w.get_slice(tid).retile(c_frag_bf16), c_layout_w.get_slice(tid).partition_D(c_tl_tile))
        c_frag_bf16.store(((c_tr_frag.load() + round_bit).bitcast(fx.Uint32) >> 16).to(fx.Uint16).bitcast(fx.BFloat16))
        fx.copy(buf_atom_w64, c_layout_w.get_slice(tid).retile(c_frag_bf16), c_layout_w.get_slice(tid).partition_D(c_tr_tile))
        c_frag_bf16.store(((c_bl_frag.load() + round_bit).bitcast(fx.Uint32) >> 16).to(fx.Uint16).bitcast(fx.BFloat16))
        fx.copy(buf_atom_w64, c_layout_w.get_slice(tid).retile(c_frag_bf16), c_layout_w.get_slice(tid).partition_D(c_bl_tile))
        c_frag_bf16.store(((c_br_frag.load() + round_bit).bitcast(fx.Uint32) >> 16).to(fx.Uint16).bitcast(fx.BFloat16))
        fx.copy(buf_atom_w64, c_layout_w.get_slice(tid).retile(c_frag_bf16), c_layout_w.get_slice(tid).partition_D(c_br_tile))

    def _encode_waitcnt(vmcnt=63, expcnt=7, lgkmcnt=63):
        """Encode s_waitcnt bitfield for CDNA3 (gfx94x)."""
        vm_lo = vmcnt & 0xF
        vm_hi = (vmcnt >> 4) & 0x3
        return vm_lo | (expcnt << 4) | (lgkmcnt << 8) | (vm_hi << 14)

    @flyc.kernel(known_block_size=[512, 1, 1])
    def gemm_8wave(arg_c_: fx.Tensor,
                   arg_a_: fx.Tensor,
                   arg_b_: fx.Tensor,
                   M: int):
        tid = fx.thread_idx.x
        wave_id = tid // 64
        # 308 has 4 xcds
        blk_x, blk_y = get_pids(fx.block_idx.x, M, N, TILE_M, TILE_N, fx.grid_dim.x, 4, 4)
        arg_a = fx.Tensor(fx.make_view(
            fx.get_iter(arg_a_),
            fx.make_layout((M, K), (K, 1))
        ))
        # pre-shuffled
        arg_b = fx.Tensor(fx.make_view(
            fx.get_iter(arg_b_),
            # layout分成两部分：第一部分(m, n)描述一个wave内部划分为16行，(8列每组)x4组；第二部分为重复第一部分的次数
            # shape: (16, (8, 4)),   (N//16, K//32)
            # stride:(8,  (1, 128)), (K*16, 512))
            # 重新排列为第0维、第1维
            fx.make_layout(((16, N // 16), (8, 4, K // 32)), ((8, 16 * K), (1, 128, 512)))
        ))
        arg_c = fx.Tensor(fx.make_view(
            fx.get_iter(arg_c_),
            fx.make_layout((M, N), (N, 1))
        ))
        a_tensor = fx.rocdl.make_buffer_tensor(arg_a, max_size=False)
        b_tensor = fx.rocdl.make_buffer_tensor(arg_b, max_size=False)
        c_tensor = fx.rocdl.make_buffer_tensor(arg_c, max_size=False)
        # shape: [m_in_tile, k_in_tile, k_tile]
        at_tile = fx.flat_divide(a_tensor, fx.make_tile(TILE_M // 2, TILE_K))[None, None, blk_x * 2 + 0, None]
        ab_tile = fx.flat_divide(a_tensor, fx.make_tile(TILE_M // 2, TILE_K))[None, None, blk_x * 2 + 1, None]
        # shape: [n_in_tile, k_in_tile, k_tile]
        bl_tile = fx.flat_divide(b_tensor, fx.make_tile(TILE_N // 2, TILE_K))[None, None, blk_y * 2 + 0, None]
        br_tile = fx.flat_divide(b_tensor, fx.make_tile(TILE_N // 2, TILE_K))[None, None, blk_y * 2 + 1, None]
        # shape: [m_in_tile, n_in_tile]
        c_tl_tile = fx.flat_divide(c_tensor, fx.make_tile(TILE_M // 2, TILE_N // 2))[None, None, 2 * blk_x + 0, 2 * blk_y + 0]
        c_tr_tile = fx.flat_divide(c_tensor, fx.make_tile(TILE_M // 2, TILE_N // 2))[None, None, 2 * blk_x + 0, 2 * blk_y + 1]
        c_bl_tile = fx.flat_divide(c_tensor, fx.make_tile(TILE_M // 2, TILE_N // 2))[None, None, 2 * blk_x + 1, 2 * blk_y + 0]
        c_br_tile = fx.flat_divide(c_tensor, fx.make_tile(TILE_M // 2, TILE_N // 2))[None, None, 2 * blk_x + 1, 2 * blk_y + 1]

        # memory->lds layout
        buf_cp_atom_r = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16)
        assert TILE_M % (32 * 2) == 0 and TILE_N % (32 * 2) == 0, "TILE_M/TILE_N need to be multiple of 64 for loading scheme"
        ab_mem_cp_layout_g2r = fx.make_tiled_copy(buf_cp_atom_r,
                                                  # 线程划分[8 * 8 wave, 8]
                                                  fx.make_layout(((8, 8, 8), 8), ((512, 1, 8), 64)),
                                                  fx.make_tile(8 * 8, TILE_K))
        # fx.utils.print_typst(ab_mem_cp_layout_g2r, file="layout-ab_mem_cp_layout_g2r.typ")
        at_mem_tensor_thr = ab_mem_cp_layout_g2r.get_slice(tid).partition_S(at_tile)
        ab_mem_tensor_thr = ab_mem_cp_layout_g2r.get_slice(tid).partition_S(ab_tile)
        bl_mem_tensor_thr = ab_mem_cp_layout_g2r.get_slice(tid).partition_S(bl_tile)
        br_mem_tensor_thr = ab_mem_cp_layout_g2r.get_slice(tid).partition_S(br_tile)
        at_cp_frag = fx.make_fragment_like(at_mem_tensor_thr[None, None, None, 0])
        ab_cp_frag = fx.make_fragment_like(ab_mem_tensor_thr[None, None, None, 0])
        bl_cp_frag = fx.make_fragment_like(bl_mem_tensor_thr[None, None, None, 0])
        br_cp_frag = fx.make_fragment_like(br_mem_tensor_thr[None, None, None, 0])

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        swz = fx.SwizzleType.get(3, 3, 3)
        at_lds = fx.make_view(lds.at_lds, fx.make_composed_layout(fx.static(swz), fx.make_ordered_layout((TILE_M // 2, TILE_K), order=(1, 0))))
        ab_lds = fx.make_view(lds.ab_lds, fx.make_composed_layout(fx.static(swz), fx.make_ordered_layout((TILE_M // 2, TILE_K), order=(1, 0))))
        bl_lds = fx.make_view(lds.bl_lds, fx.make_composed_layout(fx.static(swz), fx.make_ordered_layout((TILE_N // 2, TILE_K), order=(1, 0))))
        br_lds = fx.make_view(lds.br_lds, fx.make_composed_layout(fx.static(swz), fx.make_ordered_layout((TILE_N // 2, TILE_K), order=(1, 0))))

        uni_cp_atom = fx.make_copy_atom(fx.UniversalCopy128b(), fx.BFloat16)
        ab_lds_cp_layout_g2r = fx.make_tiled_copy(uni_cp_atom,
                                                  # 线程划分[8 * 8 wave, 8]
                                                  fx.make_layout(((8, 8, 8), 8), ((512, 1, 8), 64)),
                                                  fx.make_tile(8 * 8, TILE_K))
        at_lds_tensor_thr_w = ab_lds_cp_layout_g2r.get_slice(tid).partition_D(at_lds)
        ab_lds_tensor_thr_w = ab_lds_cp_layout_g2r.get_slice(tid).partition_D(ab_lds)
        bl_lds_tensor_thr_w = ab_lds_cp_layout_g2r.get_slice(tid).partition_D(bl_lds)
        br_lds_tensor_thr_w = ab_lds_cp_layout_g2r.get_slice(tid).partition_D(br_lds)

        # lds -> reg layout
        tiled_mma = fx.make_tiled_mma(
            fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, fx.BFloat16)),
            # wave 为2x4
            fx.make_layout((2, 4, 1), (1, 2, 0)),
            # K维度重复两次
            fx.make_tile(None, None, fx.make_layout((4, 4, 2), (1, 8, 4)))
        )
        # fx.utils.print_typst(tiled_mma, file="layout_tiled_mma.typ")

        at_lds_tensor_thr_r = fx.make_tiled_copy_A(uni_cp_atom, tiled_mma).get_slice(tid).partition_S(at_lds)
        ab_lds_tensor_thr_r = fx.make_tiled_copy_A(uni_cp_atom, tiled_mma).get_slice(tid).partition_S(ab_lds)
        bl_lds_tensor_thr_r = fx.make_tiled_copy_B(uni_cp_atom, tiled_mma).get_slice(tid).partition_S(bl_lds)
        br_lds_tensor_thr_r = fx.make_tiled_copy_B(uni_cp_atom, tiled_mma).get_slice(tid).partition_S(br_lds)

        at_frag = tiled_mma.make_fragment_A(at_lds)
        ab_frag = tiled_mma.make_fragment_A(ab_lds)
        bl_frag = tiled_mma.make_fragment_B(bl_lds)
        br_frag = tiled_mma.make_fragment_B(br_lds)

        at_frag_retile = fx.make_tiled_copy_A(uni_cp_atom, tiled_mma).get_slice(tid).retile(at_frag)
        ab_frag_retile = fx.make_tiled_copy_A(uni_cp_atom, tiled_mma).get_slice(tid).retile(ab_frag)
        bl_frag_retile = fx.make_tiled_copy_B(uni_cp_atom, tiled_mma).get_slice(tid).retile(bl_frag)
        br_frag_retile = fx.make_tiled_copy_B(uni_cp_atom, tiled_mma).get_slice(tid).retile(br_frag)

        c_tl_frag = fx.select(tiled_mma.make_fragment_C(c_tl_tile), [0, 2, 1])
        c_tr_frag = fx.select(tiled_mma.make_fragment_C(c_tr_tile), [0, 2, 1])
        c_bl_frag = fx.select(tiled_mma.make_fragment_C(c_bl_tile), [0, 2, 1])
        c_br_frag = fx.select(tiled_mma.make_fragment_C(c_br_tile), [0, 2, 1])

        c_tl_frag.store(Vec.filled(TILE_M * TILE_N // 64 // 8, 0, fx.Float32))
        c_tr_frag.store(Vec.filled(TILE_M * TILE_N // 64 // 8, 0, fx.Float32))
        c_bl_frag.store(Vec.filled(TILE_M * TILE_N // 64 // 8, 0, fx.Float32))
        c_br_frag.store(Vec.filled(TILE_M * TILE_N // 64 // 8, 0, fx.Float32))

        acc_init = [c_tl_frag.load(), c_tr_frag.load(), c_bl_frag.load(), c_br_frag.load()]

        # prefetch
        # gr0: all
        fx.copy(buf_cp_atom_r, bl_mem_tensor_thr[None, None, None, 0], bl_cp_frag)
        fx.copy(buf_cp_atom_r, at_mem_tensor_thr[None, None, None, 0], at_cp_frag)
        fx.copy(buf_cp_atom_r, ab_mem_tensor_thr[None, None, None, 0], ab_cp_frag)
        fx.copy(buf_cp_atom_r, br_mem_tensor_thr[None, None, None, 0], br_cp_frag)
        # lw0: all
        fx.copy(uni_cp_atom, bl_cp_frag, bl_lds_tensor_thr_w)
        fx.copy(uni_cp_atom, at_cp_frag, at_lds_tensor_thr_w)
        fx.copy(uni_cp_atom, ab_cp_frag, ab_lds_tensor_thr_w)
        fx.copy(uni_cp_atom, br_cp_frag, br_lds_tensor_thr_w)

        # gr1: all
        fx.copy(buf_cp_atom_r, bl_mem_tensor_thr[None, None, None, 1], bl_cp_frag)
        fx.copy(buf_cp_atom_r, at_mem_tensor_thr[None, None, None, 1], at_cp_frag)
        fx.copy(buf_cp_atom_r, ab_mem_tensor_thr[None, None, None, 1], ab_cp_frag)
        fx.copy(buf_cp_atom_r, br_mem_tensor_thr[None, None, None, 1], br_cp_frag)
        gpu.barrier()

        # lr: bl0, at0
        fx.copy(uni_cp_atom, bl_lds_tensor_thr_r, bl_frag_retile)
        fx.copy(uni_cp_atom, at_lds_tensor_thr_r, at_frag_retile)
        rocdl.s_waitcnt(_encode_waitcnt(lgkmcnt=0))
        if wave_id >= 4:
            gpu.barrier()

        assert K // TILE_K >= 2, "this kernel requires at least 2 iterations"
        for k, state in range(0, K // TILE_K - 0, 1, init=acc_init):
            c_tl_frag.store(state[0])
            c_tr_frag.store(state[1])
            c_bl_frag.store(state[2])
            c_br_frag.store(state[3])
            k_i32 = fx.Int32(k)

            rocdl.sched_barrier(0)
            rocdl.s_setprio(1)
            rocdl.sched_barrier(0)
            # bl0 @ at0
            for sub_k in range_constexpr(TILE_K // 32):
                fx.gemm(tiled_mma, c_tl_frag, bl_frag[None, None, (None, sub_k)], at_frag[None, None, (None, sub_k)], c_tl_frag)
            rocdl.sched_barrier(0)
            rocdl.s_setprio(0)
            gpu.barrier()
            rocdl.sched_barrier(0)
            # lr: ab0
            fx.copy(uni_cp_atom, ab_lds_tensor_thr_r, ab_frag_retile)
            # lw: bl1
            fx.copy(uni_cp_atom, bl_cp_frag, bl_lds_tensor_thr_w)
            # gr: bl2
            fx.copy(buf_cp_atom_r, bl_mem_tensor_thr[None, None, None, k_i32 + 2], bl_cp_frag)
            gpu.barrier()
            rocdl.sched_barrier(0)

            rocdl.sched_barrier(0)
            rocdl.s_setprio(1)
            rocdl.sched_barrier(0)
            # bl0 @ ab0
            for sub_k in range_constexpr(TILE_K // 32):
                fx.gemm(tiled_mma, c_bl_frag, bl_frag[None, None, (None, sub_k)], ab_frag[None, None, (None, sub_k)], c_bl_frag)
            rocdl.sched_barrier(0)
            rocdl.s_setprio(0)
            gpu.barrier()
            rocdl.sched_barrier(0)
            # lr: br0
            fx.copy(uni_cp_atom, br_lds_tensor_thr_r, br_frag_retile)
            # lw: at1
            fx.copy(uni_cp_atom, at_cp_frag, at_lds_tensor_thr_w)
            # gr: at2
            fx.copy(buf_cp_atom_r, at_mem_tensor_thr[None, None, None, k_i32 + 2], at_cp_frag)
            gpu.barrier()
            rocdl.sched_barrier(0)

            rocdl.sched_barrier(0)
            rocdl.s_setprio(1)
            rocdl.sched_barrier(0)
            # br0 @ at0
            for sub_k in range_constexpr(TILE_K // 32):
                fx.gemm(tiled_mma, c_tr_frag, br_frag[None, None, (None, sub_k)], at_frag[None, None, (None, sub_k)], c_tr_frag)
            rocdl.sched_barrier(0)
            rocdl.s_setprio(0)
            gpu.barrier()
            rocdl.sched_barrier(0)
            # lr: bl1
            fx.copy(uni_cp_atom, bl_lds_tensor_thr_r, bl_frag_retile)
            # lw: ab1
            fx.copy(uni_cp_atom, ab_cp_frag, ab_lds_tensor_thr_w)
            # gr: ab2
            fx.copy(buf_cp_atom_r, ab_mem_tensor_thr[None, None, None, k_i32 + 2], ab_cp_frag)
            gpu.barrier()
            rocdl.sched_barrier(0)

            rocdl.sched_barrier(0)
            rocdl.s_setprio(1)
            rocdl.sched_barrier(0)
            # br0 @ ab0
            for sub_k in range_constexpr(TILE_K // 32):
                fx.gemm(tiled_mma, c_br_frag, br_frag[None, None, (None, sub_k)], ab_frag[None, None, (None, sub_k)], c_br_frag)
            rocdl.sched_barrier(0)
            rocdl.s_setprio(0)
            gpu.barrier()
            rocdl.sched_barrier(0)
            # lr: at1
            fx.copy(uni_cp_atom, at_lds_tensor_thr_r, at_frag_retile)
            # lw: br1
            fx.copy(uni_cp_atom, br_cp_frag, br_lds_tensor_thr_w)
            # gr: br2
            fx.copy(buf_cp_atom_r, br_mem_tensor_thr[None, None, None, k_i32 + 2], br_cp_frag)
            gpu.barrier()
            rocdl.sched_barrier(0)

            results = yield [c_tl_frag.load(), c_tr_frag.load(), c_bl_frag.load(), c_br_frag.load()]

        c_tl_frag.store(results[0])
        c_tr_frag.store(results[1])
        c_bl_frag.store(results[2])
        c_br_frag.store(results[3])        
        c_tl_frag = fx.select(c_tl_frag, [0, 2, 1])
        c_tr_frag = fx.select(c_tr_frag, [0, 2, 1])
        c_bl_frag = fx.select(c_bl_frag, [0, 2, 1])
        c_br_frag = fx.select(c_br_frag, [0, 2, 1])

        if wave_id < 4:
            gpu.barrier()

        c_frag_bf16 = fx.make_fragment_like(c_tl_frag, dtype=fx.BFloat16)
        round_bit = fx.Uint32(0x8000).ir_value().bitcast(fx.Float32.ir_type)
        buf_atom_w64 = fx.make_copy_atom(fx.rocdl.BufferCopy64b(), fx.BFloat16)
        c_layout_w = fx.make_tiled_copy(buf_atom_w64,
                                        # wave先M，再N（匹配mma中的设置）
                                        fx.make_layout(((16, 4, 2, 4), 4), ((1, 128, 16, 512), 32)),
                                        fx.make_tile(32, 64))
        # fx.utils.print_typst(c_layout_w, file="layout-c-out.typ")

        # acc = acc.to(fx.BFloat16)
        c_frag_bf16.store(((c_tl_frag.load() + round_bit).bitcast(fx.Uint32) >> 16).to(fx.Uint16).bitcast(fx.BFloat16))
        fx.copy(buf_atom_w64, c_layout_w.get_slice(tid).retile(c_frag_bf16), c_layout_w.get_slice(tid).partition_D(c_tl_tile))
        c_frag_bf16.store(((c_tr_frag.load() + round_bit).bitcast(fx.Uint32) >> 16).to(fx.Uint16).bitcast(fx.BFloat16))
        fx.copy(buf_atom_w64, c_layout_w.get_slice(tid).retile(c_frag_bf16), c_layout_w.get_slice(tid).partition_D(c_tr_tile))
        c_frag_bf16.store(((c_bl_frag.load() + round_bit).bitcast(fx.Uint32) >> 16).to(fx.Uint16).bitcast(fx.BFloat16))
        fx.copy(buf_atom_w64, c_layout_w.get_slice(tid).retile(c_frag_bf16), c_layout_w.get_slice(tid).partition_D(c_bl_tile))
        c_frag_bf16.store(((c_br_frag.load() + round_bit).bitcast(fx.Uint32) >> 16).to(fx.Uint16).bitcast(fx.BFloat16))
        fx.copy(buf_atom_w64, c_layout_w.get_slice(tid).retile(c_frag_bf16), c_layout_w.get_slice(tid).partition_D(c_br_tile))

    @flyc.jit
    def launch(
        arg_c: fx.Tensor,
        arg_a: fx.Tensor,
        arg_b: fx.Tensor,
        M: int,
        stream: fx.Stream
    ):
        CompilationContext.get_current()
        if const_expr(alg == 'splitk'):
            gemm_splitk(arg_c, arg_a, arg_b, M).launch(grid=(div_up(M, TILE_M), div_up(N, TILE_N), 1), block=(256, 1, 1), stream=stream)
        elif const_expr(alg == '4wave'):
            value_attrs = None
            if const_expr((TILE_M >= 128 and TILE_N > 128) or (TILE_M > 128 and TILE_N >= 128)):
                value_attrs = {"rocdl.waves_per_eu": 1,
                               "passthrough": [["amdgpu-agpr-alloc", "256,256"],]
                              }
            gemm_4wave(arg_c, arg_a, arg_b, M,
                       value_attrs=value_attrs,
                       ).launch(grid=(div_up(M, TILE_M) * div_up(N, TILE_N), 1, 1), block=(256, 1, 1), stream=stream)
        else:
            value_attrs = None
            gemm_8wave(arg_c, arg_a, arg_b, M,
                       value_attrs=value_attrs,
                       ).launch(grid=(div_up(M, TILE_M) * div_up(N, TILE_N), 1, 1), block=(512, 1, 1), stream=stream)
    
    return launch

#####################################################################
from pyhip import cudaPerf
from torch import Tensor
import pytest

SHUFFLE = True

def div_up(x, y):
    return (x + y - 1) // y
def _run_aiter(
                x: Tensor,  # A:[M, K] bf16
                weight: Tensor,  # B:[N, K/2] f4x2
                weight_scale: Tensor,  # B_scale:[N, K/32] e8m0 paded
            ):
    from aiter import gemm_a4w4, per_1x32_f4_quant_hip, gemm_a16w16
    M = x.shape[0]
    N = weight.shape[0]
    K = weight.shape[1]
    if x.dtype == torch.bfloat16:
        out = torch.empty([M, N], dtype=torch.bfloat16, device=x.device)
        gemm_a16w16(x, weight, out, splitK=4, bpreshuffle=True)
        return out

    # use hip quant kernel for performance
    x_q, x_s = per_1x32_f4_quant_hip(x, shuffle=True)

    # 32 alignment is enough for dim0 padding of output for
    # gemm_a4w4 kernel
    y = torch.empty(
        (M + 31) // 32 * 32,
        weight.shape[0],
        device=x_q.device,
        dtype=x.dtype,
    )

    gemm_a4w4(
        x_q, weight, x_s, weight_scale.view(x_s.dtype), y, bpreshuffle=True
    )

    return y[:M]

def _run_batch(num_warps, kernel_type, M=1, weight_type=torch.bfloat16, TILE_M=16, TILE_N=32, run_count=10, N=4096, K=4096):
    BUF_COPY = 32
    A = torch.randn([BUF_COPY, M, K], dtype=torch.bfloat16)
    from aiter.ops.shuffle import shuffle_weight
    import aiter
    if weight_type == torch.float4_e2m1fn_x2:
        from aiter.utility import fp4_utils
        w_ = torch.randint(-2, 3, [N, K], dtype=torch.bfloat16) / 2
        w_qt, w_qt_scale_ = aiter.get_torch_quant(aiter.QuantType.per_1x32)(w_, quant_dtype=weight_type)
        w_f32 = fp4_utils.mxfp4_to_f32(w_qt).to(dtype=torch.bfloat16).reshape(N, K // 32, 32)
        w_scale_f32 = fp4_utils.e8m0_to_f32(w_qt_scale_).to(dtype=torch.bfloat16).reshape(N, K // 32, 1)
        w_ref = (w_f32 * w_scale_f32).reshape(N, K)
        assert K % 256 == 0, f'e8m0_shuffle assume there will be 8 groups of 32 elements in K dim, current K={K} is not supported'
        w_qt_scale = fp4_utils.e8m0_shuffle(w_qt_scale_)
        w = [shuffle_weight(w_qt) for _ in range(BUF_COPY)]
        w_scale = [w_qt_scale.clone() for _ in range(BUF_COPY)]
    elif weight_type == torch.bfloat16:
        w_ = torch.randn([N, K], dtype=torch.bfloat16)
        if SHUFFLE:
            w_shuffled = shuffle_weight(w_).reshape(N // 16, -1)
            w = [w_shuffled.clone() for _ in range(BUF_COPY)]
            w_org = [x.reshape([N, K]) for x in w]
        else:
            w = [w_ for _ in range(BUF_COPY)]
        w_scale = [None] * BUF_COPY
        w_ref = w_
    elif weight_type == torch.float8_e4m3fn:
        w_qt = torch.randint(-2, 3, [N, K], dtype=torch.float32).to(weight_type)
        w_qt_scale = torch.randint(-2, 3, [N // 128, K // 128], dtype=torch.float32)
        # # TODO
        # w_qt_scale[:] = 1
        w_f32 = w_qt.to(dtype=torch.float32).reshape(N // 128, 128, K // 128, 128)
        w_scale_f32 = w_qt_scale.reshape(N // 128, 1, K // 128, 1)
        w_ref = (w_f32 * w_scale_f32).reshape(N, K).to(dtype=torch.bfloat16)
        w = [shuffle_weight(w_qt) for _ in range(BUF_COPY)]
        w_scale = [w_qt_scale.clone() for _ in range(BUF_COPY)]
    else:
        assert 0, f'Only fp4 weight is supported in this test, current weight_type={weight_type}'

    flops = 2 * M * N * K
    if weight_type == torch.bfloat16:
        ele_size = 2
    elif weight_type == torch.float4_e2m1fn_x2:
        ele_size = 0.5
    else:
        ele_size = 1
    mem_size = M * K * 2 + N * K * ele_size

    def run(kernel, stream, A, weight, weight_scale, p_debug_buf=None):
        M, K = A.shape
        N = w_ref.shape[0]
        gemm_out = torch.empty([M, N], dtype=A.dtype, device=A.device)

        kernel(
            gemm_out.view(-1),         # bf16 [M, N]
            A.view(-1),                # bf16 [M, K]
            weight.view(-1),           # bf16 [K, N]
            M,
            stream)

        return gemm_out

    tflops_res = []
    latencies = []
    bw = []
    if kernel_type == 'torch':
        i = 0
        for _ in range(run_count):
            with cudaPerf(flops, mem_size, name=f"{kernel_type}[{M=},{str(weight_type).split('.')[1]}]") as p:
                ref = torch.nn.functional.linear(A[i], w_org[i])
            tflops_res.append(p.tflops())
            latencies.append(p.dt())
            bw.append(p.bw())
        ref_out = A[0] @ w_org[0].t()
        cur_out = torch.nn.functional.linear(A[0], w_org[0])
        if not torch.allclose(ref_out, cur_out, rtol=0.1, atol=0.03):
            print(cur_out)
            idx = torch.where(torch.abs(ref_out - cur_out) > 0.03)
            if len(idx[0]):
                print(f'idx = {idx}\nref={ref_out[idx]}\ncur={cur_out[idx]}\n{len(idx[0])}')
            assert 0, f"{kernel_type=}, {M=}, {weight_type=}, {TILE_M=}, {TILE_N=}, {run_count=}"

    elif kernel_type == 'aiter':
        # aiter needs preshuffle weights
        i = 0
        for _ in range(run_count):
            with cudaPerf(flops, mem_size, name=f"{kernel_type}[{M=},{str(weight_type).split('.')[1]}]") as p:
                _run_aiter(x=A[i], weight=w_org[i], weight_scale=w_scale[i])
            i = (i + 1) % BUF_COPY
            tflops_res.append(p.tflops())
            latencies.append(p.dt())
            bw.append(p.bw())
    else:
        def _as_i8(t):
            return t.view(torch.int8) if "float8" in str(t.dtype) else t

        def _make_args(c, a, b_shuf, M, include_bias=False):
            args = [
                c.view(-1),
                _as_i8(a.view(-1)),
                _as_i8(b_shuf.view(-1)),
                #sa.view(-1) if sa.numel() > 0 else sa,
                #sb.view(-1) if sb.numel() > 0 else sb,
            ]
            if include_bias:
                args.append(torch.empty(0, device=c.device, dtype=c.dtype))
            args.extend([M, torch.cuda.current_stream()])
            return tuple(args)
        ref_out = A[0] @ w_ref.t()
        hints = {
            #"maxnreg": 256,
            "opt_level": 2,
            #"llvm_options": ""
        }
        args = _make_args(torch.zeros_like(ref_out), A[0], w[0], M)
        if kernel_type == 'splitk':
            launcher = compile_gemm(TILE_M, TILE_N, N, K, alg='splitk')
            gemm = flyc.compile[hints](launcher, *args)

        elif kernel_type == '4wave':
            launcher = compile_gemm(TILE_M, TILE_N, N, K, alg='4wave')
            hints['llvm_options'] = {
                "amdgpu-mfma-vgpr-form": False,
            }
            gemm = flyc.compile[hints](launcher, *args)
        elif kernel_type == '8wave':
            launcher = compile_gemm(TILE_M, TILE_N, N, K, alg='8wave')
            gemm = flyc.compile[hints](launcher, *args)
        else:
            assert 0, f"unsupported kernel type {kernel_type}"

        cur_out = run(gemm, torch.cuda.current_stream(), A=A[0], weight=w[0], weight_scale=w_scale[0])
        i = 0
        for _ in range(run_count):
            with cudaPerf(flops, mem_size, name=f"{kernel_type}[{M=},{str(weight_type).split('.')[1]}]") as p:
                run(gemm, torch.cuda.current_stream(), A=A[i], weight=w[i], weight_scale=w_scale[i])
            i = (i + 1) % BUF_COPY
            tflops_res.append(p.tflops())
            latencies.append(p.dt())
            bw.append(p.bw())

        if not torch.allclose(ref_out, cur_out, rtol=0.1, atol=0.03):
            print(cur_out)
            idx = torch.where(torch.abs(ref_out - cur_out) > 0.03)
            if len(idx[0]):
                print(f'idx = {idx}\nref={ref_out[idx]}\ncur={cur_out[idx]}\n{len(idx[0])}')
            assert 0, f"{kernel_type=}, {M=}, {weight_type=}, {TILE_M=}, {TILE_N=}, {run_count=}"
        else:
            print(f"{kernel_type}[{M=} {weight_type=}] acc OK")
    if run_count > 0:
        return {'flops': sum(tflops_res[1:])/len(tflops_res[1:]),              # tflops
                'latency': sum(latencies[1:])/len(latencies[1:]) * 1e6,        # us
                'bw': sum(bw[1:]) / len(bw[1:])}                               # GB/s

def is_arch_type(arch):
    props = torch.cuda.get_device_properties()
    return arch in props.gcnArchName

def get_fp8type():
    return torch.float8_e4m3fn if is_arch_type('950') else torch.float8_e4m3fnuz

def get_fp4type_if_valid():
    return torch.float4_e2m1fn_x2 if is_arch_type('950') else None

def entry_common(num_warps, kernel_type, M, prec=[torch.bfloat16], TILE_M=32, TILE_N=64, N=4096, K=4096, run_count=10):
    perf = {}
    perf[kernel_type] = {}
    for weight_type in prec:
        if weight_type is None: continue
        perf_prec = {}
        for i in M:
            perf_prec[i] = _run_batch(num_warps, kernel_type, M=i, weight_type=weight_type, TILE_M=TILE_M, TILE_N=TILE_N, run_count=run_count, N=N, K=K)
        perf[kernel_type][str(weight_type)] = perf_prec
    
    return perf

def init_env():
    torch.set_printoptions(linewidth=3000, sci_mode=False, edgeitems=8, )
    torch.set_default_device('cuda')
    torch.manual_seed(0)

def test_acc(TILE_M=32, TILE_N=64, N=4096, K=4096):
    init_env()
    M = list(range(2, 64))
    # fix TILE_M=16, TILE_N=32
    M += list(range(128, 256))
    M += [i * 256 for i in range(1, 4)]
    M += [i * 2048 for i in range(1, 5)]
    M += list(range(2048 * 3, 2048 * 3 + 256))
    # TILE_M/N is configurable
    entry_common('mxn_2s', M=M, prec=[torch.bfloat16], TILE_M=TILE_M, TILE_N=TILE_N, N=N, K=K, run_count=0)

def show_perf(perf, dict_tile_mn):
    print('\nsummary:')
    for kernel, vals in perf.items():
        for prec, vals_ in vals.items():
            for b, data in vals_.items():
                if kernel != 'aiter':
                    TILE_M, TILE_N = dict_tile_mn[f'{b}']
                    print(f'{kernel}[{prec:<4} B={b:<4}({TILE_M}x{TILE_N})]: {data["latency"]:5.0f} us, {data["bw"]:6.1f} GB/s, {data["flops"]:4.1f} tflops')
                else:
                    print(f'{kernel}[{prec:<4} B={b:<4}]: {data["latency"]:5.0f} us, {data["bw"]:6.1f} GB/s, {data["flops"]:4.1f} tflops')

@pytest.mark.parametrize("M", [[1, 2, 4, 8, 12, 16, 32, 64]])
def test_perf(num_warps, alg, M, TILE_M=32, TILE_N=64, N=4096, K=4096):
    init_env()
    perf = {}
    # perf.update(entry_common(num_warps, 'torch', M, prec=[torch.bfloat16], N=N, K=K, TILE_M=TILE_M, TILE_N=TILE_N))
    # TILE_M/N is configurable
    # perf.update(entry_common('mxn_2s', M=M, prec=[torch.bfloat16, get_fp8type()], TILE_M=TILE_M, TILE_N=TILE_N, N=N, K=K))
    # perf.update(entry_common('mxn_2s', M=M, prec=[get_fp8type()], TILE_M=TILE_M, TILE_N=TILE_N, N=N, K=K))
    perf.update(entry_common(num_warps, alg, M=M, prec=[torch.bfloat16], TILE_M=TILE_M, TILE_N=TILE_N, N=N, K=K))
    return perf

def merge(a: dict, b: dict, path=[]):
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                merge(a[key], b[key], path + [str(key)])
            elif a[key] != b[key]:
                raise Exception('Conflict at ' + '.'.join(path + [str(key)]))
        else:
            a[key] = b[key]
    return a

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="config input of test",
    )
    parser.add_argument(
        "-w",
        "--wave",
        type=int,
        choices=[4, 8],
        default=4,
        help="""select wave number 4 or 8""",
    )
    args = parser.parse_args()
    print(f'selected num_warps={args.wave}')

    #TILE_M = 16
    #TILE_N = 128
    N, K = 4096*1, 1024*8 # 4096*8*2 /128 = 512
    #N, K = 64*1024*1, 1024*16
    Ms = [16, 32, 64, 128, 256]
    Ms = [4096,] # 64, 128]
    #Ms = [13, 64, 257, 1024, 4096, 4096*2, 4096*4]
    # N, K = 64*4, 256*4
    perf = {}
    dict_tile_mn = {}

    def get_tile_mn(M):
        num_CU = torch.cuda.get_device_properties().multi_processor_count
        solutions = []
        for tile_m in [16, 32, 64]:
            for tile_n in [32, 64, 128]:
                works = div_up(M, tile_m) * div_up(N, tile_n)
                if works >= num_CU:
                    round = works // num_CU
                    reminder = works % num_CU
                    solutions.append((round, reminder, tile_m, tile_n))
                else:
                    reminder = num_CU - works % num_CU
                    solutions.append((100000, reminder, tile_m, tile_n))
        # prefer less rounds; then less reminder
        TILE_M, TILE_N = sorted(solutions)[0][2:]
        return TILE_M, TILE_N

    TILE_M, TILE_N = get_tile_mn(64)
    for M in Ms:
        TILE_M, TILE_N = get_tile_mn(M)
        # if N == 9216 and K == 4096:
        #     if M in [16]:
        #         TILE_M = 16
        #         TILE_N = 64
        #     elif M in [32]:
        #         TILE_M = 32
        #         TILE_N = 64
        #     elif M in [64, 128]:
        #         TILE_M = 32
        #         TILE_N = 128

        #TILE_M, TILE_N = 16, 64
        TILE_M, TILE_N = 256, 128
        dict_tile_mn[f'{M}'] = (TILE_M, TILE_N)
        print(f'final selected TILE_M={TILE_M}, TILE_N={TILE_N}')
        #test_acc(TILE_M=TILE_M, TILE_N=TILE_N, N=N, K=K)
        # perf = merge(perf, test_perf(num_warps=args.wave, alg='splitk', M=[M], TILE_M=TILE_M, TILE_N=TILE_N, N=N, K=K))
        perf = merge(perf, test_perf(num_warps=args.wave, alg='torch', M=[M], TILE_M=TILE_M, TILE_N=TILE_N, N=N, K=K))
        TILE_M, TILE_N = 256, 256
        perf = merge(perf, test_perf(num_warps=args.wave, alg='4wave', M=[M], TILE_M=TILE_M, TILE_N=TILE_N, N=N, K=K))
        TILE_M, TILE_N = 256, 128
        perf = merge(perf, test_perf(num_warps=args.wave, alg='8wave', M=[M], TILE_M=TILE_M, TILE_N=TILE_N, N=N, K=K))
    show_perf(perf, dict_tile_mn)