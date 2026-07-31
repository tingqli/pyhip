# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

import torch
import math
import flydsl.compiler as flyc
import flydsl.expr as fx
import flydsl.compiler as flyc
from flydsl.expr.typing import BFloat16, Float8E4M3FN, Float8E4M3FNUZ, Float16, Float32, Int8, Int32, T, Vector
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl, vector, arith
import os
from flydsl._mlir.dialects import llvm as _llvm

from flydsl.expr.typing import Vector as Vec
from flydsl._mlir.dialects import fly as fly_dialect
from flydsl._mlir import ir
from flydsl.expr.typing import T as _T
from flydsl.compiler.ast_rewriter import ASTRewriter
from enum import Enum

def anchor_frag(frag):
    # 空 asm 延长 fragment 的 VGPR live range，防止后续 ds_read 过早复用仍被 MFMA
    # 消费的寄存器而破坏交织（参照 test_gemm.py 的 anchor_b_frag）。
    words = frag.load().bitcast(fx.Int32)
    num_words = words.numel
    result_type = ir.Type.parse(f"!llvm.struct<({', '.join(['i32'] * num_words)})>")
    operands = [arith._to_raw(words[i]) for i in range_constexpr(num_words)]
    constraints = ",".join(["=r"] * num_words + ["r"] * num_words)
    _llvm.inline_asm(
        result_type,
        operands,
        ";",
        constraints,
        has_side_effects=True,
    )

def hot_loop_scheduler(group_id, vmem_cnt, lds_cnt, mfma_cnt):
    mfma_prolog = 4
    assert (vmem_cnt*4 + lds_cnt + mfma_prolog) <= mfma_cnt

    mfma_epilog = mfma_cnt - mfma_prolog - vmem_cnt*4 - lds_cnt
    #每一条MFMA指令单独schedule, 避免多个一组组内会打乱顺序。
    for _ in range_constexpr(mfma_prolog):
        rocdl.sched_group_barrier(rocdl.mask_mfma, 1, group_id)
    for _ in range_constexpr(lds_cnt):
        rocdl.sched_group_barrier(rocdl.mask_dsrd, 1, group_id)
        rocdl.sched_group_barrier(rocdl.mask_mfma, 1, group_id)
    for _ in range_constexpr(vmem_cnt):
        rocdl.sched_group_barrier(rocdl.mask_vmem_rd, 1, group_id)
        rocdl.sched_group_barrier(rocdl.mask_mfma, 1, group_id)
        rocdl.sched_group_barrier(rocdl.mask_mfma, 1, group_id)
        rocdl.sched_group_barrier(rocdl.mask_mfma, 1, group_id)
        rocdl.sched_group_barrier(rocdl.mask_mfma, 1, group_id)
    for _ in range_constexpr(mfma_epilog):
        rocdl.sched_group_barrier(rocdl.mask_mfma, 1, group_id)

# VMEM_WRITE / VALU 掩码（flydsl 未导出 vmem_wr 常量，直接用 bit 值）。
_MASK_VALU = 0x002
_MASK_VMEM_WR = 0x040


# def scheduler_store_overlap(group_id):
#     # MFMA 领先：每发 4 条 MFMA，穿插 store 的 VALU(cvt/permlane) 与 buffer_store(vmem_wr)，
#     # 用 MFMA 计算掩盖 store 的写延迟（MFMA 必须领先，否则 store 会挡住计算流水）。
#     for _ in range_constexpr(8):
#         rocdl.sched_group_barrier(rocdl.mask_mfma, 4, group_id)
#         rocdl.sched_group_barrier(_MASK_VALU, 6, group_id)
#         rocdl.sched_group_barrier(_MASK_VMEM_WR, 1, group_id)

def scheduler_store_overlap(group_id, store_cnt, lds_cnt, mfma_cnt):
    assert store_cnt >= lds_cnt and mfma_cnt >= (lds_cnt + store_cnt)
    for _ in range_constexpr(lds_cnt):
        rocdl.sched_group_barrier(rocdl.mask_mfma, 1, group_id)
        rocdl.sched_group_barrier(rocdl.mask_dsrd, 1, group_id)
        rocdl.sched_group_barrier(rocdl.mask_mfma, 1, group_id)
        rocdl.sched_group_barrier(_MASK_VMEM_WR, 1, group_id)
    for _ in range_constexpr(store_cnt - lds_cnt):
        rocdl.sched_group_barrier(rocdl.mask_mfma, 1, group_id)
        rocdl.sched_group_barrier(_MASK_VMEM_WR, 1, group_id)
    for _ in range_constexpr(mfma_cnt - lds_cnt - store_cnt):
        rocdl.sched_group_barrier(rocdl.mask_mfma, 1, group_id)


    
# every 8 contineous row pad 16 elements. (need 128/8-1) * 16 elements padding totally.
def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")

def enable_dump_ir(enable_debug_info=True):
    if enable_debug_info:
        import flydsl
        from flydsl.utils.env import DebugEnvManager
        from flydsl._mlir import ir

        DebugEnvManager.enable_debug_info = enable_debug_info
        DebugEnvManager.dump_asm = True
        DebugEnvManager.dump_ir = True
        DebugEnvManager.dump_dir = "my_ir_dumps"
        ir._globals.register_traceback_file_inclusion(__file__)
        ir._globals.register_traceback_file_exclusion(os.path.dirname(flydsl.__file__))
        ir._globals.set_loc_tracebacks_frame_limit(40)
        ir._globals.set_loc_tracebacks_enabled(True)
        os.environ.setdefault("FLYDSL_RUNTIME_ENABLE_CACHE", "0")


def encode_waitcnt_950(vmcnt=63, expcnt=7, lgkmcnt=63):
    """Encode s_waitcnt bitfield for CDNA3 (gfx94x)."""
    vm_lo = vmcnt & 0xF
    vm_hi = (vmcnt >> 4) & 0x3
    return vm_lo | (expcnt << 4) | (lgkmcnt << 8) | (vm_hi << 14)

def wait_barrier(count):
    _llvm.inline_asm(
        res=None,
        operands_=[],
        asm_string=f"s_waitcnt vmcnt({count})\ns_barrier\n",
        # asm_string=f"s_waitcnt vmcnt({count})\ns_barrier\ns_waitcnt lgkmcnt(0)\n",
        constraints="",
        has_side_effects=True,
    )

def waitvmcnt_barrier(vmcnt):
        rocdl.s_waitcnt(encode_waitcnt_950(vmcnt=vmcnt))
        rocdl.s_waitcnt(encode_waitcnt_950(lgkmcnt=0))
        rocdl.s_barrier()

class MfmaMode(Enum):
    # rocdl.mfma intrinsic，每个 c_slice 的 k0/k1 是两条独立指令，调度器可自由重排
    # （通常把 k 提到外层、交织不同 accumulator 以掩盖 MFMA 依赖延迟）。
    ROCDL = 0
    # k0/k1 塞进同一条 inline_asm，强制两条 MFMA 背靠背复用同一 accumulator，
    # 命中 GFXIPARCH-1380 的 SRCC/VDST read/write suppression。
    INLINE_ASM_K2 = 1
    # 交给 fx.gemm（tiled MMA）自行生成。
    FX_GEMM = 2


class Mfma16x16x64:
    def __init__(self, n_tiles_a, n_tiles_b, mma_atom, mode=MfmaMode.ROCDL):
        self.mma_atom = mma_atom
        self.n_tiles_a = n_tiles_a
        self.n_tiles_b = n_tiles_b
        self.mode = mode

    def call_BxA(self, aa, bb, c):
        if self.mode == MfmaMode.ROCDL:
            # 每个 c_slice 独立发两条 k intrinsic，调度器自由重排。
            for n in range_constexpr(self.n_tiles_b):
                for m in range_constexpr(self.n_tiles_a):
                    c_slice = c[None, n, m]
                    acc = arith._to_raw(c_slice.load())
                    for k in range_constexpr(2):
                        a = bb[None, n, k].load()
                        b = aa[None, m, k].load()
                        acc = rocdl.mfma_f32_16x16x32_bf16(
                            T.vec(4, T.f32),
                            [arith._to_raw(a), arith._to_raw(b), arith._to_raw(acc), 0, 0, 0],
                        )
                    c_slice.store(acc)
        elif self.mode == MfmaMode.INLINE_ASM_K2:
            # 把同一 accumulator 的 k0/k1 两条 MFMA 放进同一条 inline_asm，
            # 调度器无法拆开 -> 背靠背复用 Matrix C/D，命中 GFXIPARCH-1380 suppression。
            for n in range_constexpr(self.n_tiles_b):
                for m in range_constexpr(self.n_tiles_a):
                    c_slice = c[None, n, m]
                    a0 = vector.bitcast(_T.vec(4, _T.i32), bb[None, n, 0].load())
                    b0 = vector.bitcast(_T.vec(4, _T.i32), aa[None, m, 0].load())
                    a1 = vector.bitcast(_T.vec(4, _T.i32), bb[None, n, 1].load())
                    b1 = vector.bitcast(_T.vec(4, _T.i32), aa[None, m, 1].load())
                    acc = c_slice.load()
                    res = _llvm.inline_asm(
                        _T.vec(4, _T.f32),
                        [
                            arith._to_raw(a0), arith._to_raw(b0),
                            arith._to_raw(a1), arith._to_raw(b1),
                            arith._to_raw(acc),
                        ],
                        "v_mfma_f32_16x16x32_bf16 $0, $1, $2, $0\n"
                        "v_mfma_f32_16x16x32_bf16 $0, $3, $4, $0",
                        "=a,v,v,v,v,0",
                        has_side_effects=False,
                    )
                    c_slice.store(res)
        else:
            fx.gemm(self.mma_atom, c, bb, aa, c)



def div_up(x, y):
    return (x + y - 1) // y

def compile_gemm(
    TILE_M,
    TILE_N,
    TILE_K,
    N,
    K,
    dtype="bf16",
    lds_swizzle=False,
    pid_swizzle=False,
    permlane_epilogue=True,
    mfma_mode=MfmaMode.ROCDL,
    preshuffle_b=False,
):
    BLOCK_M = TILE_M // 2
    BLOCK_N = TILE_N // 2
    BLOCK_K = TILE_K
    def _get_pids_950(pid, M, GRID_MN, NUM_XCDS, GROUP_SIZE_M):
        num_pid_m = (M + TILE_M - 1) // TILE_M
        num_pid_n = div_up(N, TILE_N)

        if const_expr(NUM_XCDS != 1):
            pids_per_xcd = (GRID_MN + NUM_XCDS - 1) // NUM_XCDS
            tall_xcds = GRID_MN % NUM_XCDS
            tall_xcds = (tall_xcds == 0).select(NUM_XCDS, tall_xcds)
            xcd = pid % NUM_XCDS
            local_pid = pid // NUM_XCDS
            if xcd < tall_xcds:
                pid = xcd * pids_per_xcd + local_pid
            else:
                pid = tall_xcds * pids_per_xcd + (xcd - tall_xcds) * (pids_per_xcd - 1) + local_pid

        if const_expr(GROUP_SIZE_M == 1):
            pid_m = pid // num_pid_n
            pid_n = pid % num_pid_n
        else:
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            group_id = pid // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            remaining_pid_m = num_pid_m - first_pid_m
            group_size_m = (remaining_pid_m < GROUP_SIZE_M).select(remaining_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
            pid_n = (pid % num_pid_in_group) // group_size_m

        return pid_m, pid_n
    get_pids_950 = ASTRewriter.transform(_get_pids_950)

    # preshuffle B 可能会有一些padding的浪费
    PADDING_ELEMS = 16
    PADDING_NUM = PADDING_ELEMS * (16 - 1)
    if const_expr(lds_swizzle):
        PADDING_NUM = 0
    @fx.struct
    class LDS_PADDING:
        lds0_a_t: fx.Array[BFloat16, BLOCK_M*BLOCK_K+PADDING_NUM, 16]
        lds0_a_b: fx.Array[BFloat16, BLOCK_M*BLOCK_K+PADDING_NUM, 16]
        lds0_b_l: fx.Array[BFloat16, BLOCK_N*BLOCK_K+PADDING_NUM, 16]
        lds0_b_r: fx.Array[BFloat16, BLOCK_N*BLOCK_K+PADDING_NUM, 16]
        lds1_a_t: fx.Array[BFloat16, BLOCK_M*BLOCK_K+PADDING_NUM, 16]
        lds1_a_b: fx.Array[BFloat16, BLOCK_M*BLOCK_K+PADDING_NUM, 16]
        lds1_b_l: fx.Array[BFloat16, BLOCK_N*BLOCK_K+PADDING_NUM, 16]
        lds1_b_r: fx.Array[BFloat16, BLOCK_N*BLOCK_K+PADDING_NUM, 16]
    

    element_type = fx.BFloat16

    @flyc.kernel
    def gemm_kernel(
        argA: fx.Tensor,
        argB: fx.Tensor,
        argC: fx.Tensor,
        M: int
    ):
        tid = fx.thread_idx.x
        num_pid_n = div_up(N, TILE_N)
        if const_expr(pid_swizzle):
            bid_x, bid_y = get_pids_950(fx.block_idx.x, M, fx.grid_dim.x, 8, 4)
        else:
            bid_x = fx.block_idx.x // num_pid_n
            bid_y = fx.block_idx.x % num_pid_n


        a_iter = fx.get_iter(argA)
        b_iter = fx.get_iter(argB)
        A_2d = fx.Tensor(fx.make_view(
            a_iter,
            fx.make_layout((M, K), (K, 1)),
        ))
        
        # A_2d = fx.Tensor(fx.make_view(fx.get_iter(A), fx.make_layout((M, K), (K, 1))))
        # B_2d = fx.Tensor(fx.make_view(fx.get_iter(B), fx.make_layout((N, K), (K, 1))))
        # C_2d = fx.Tensor(fx.make_view(fx.get_iter(C), fx.make_layout((M, N), (N, 1))))
        B_2d = fx.Tensor(fx.make_view(b_iter, fx.make_layout((N, K), (K, 1))))
        C_2d = fx.Tensor(fx.make_view(
            fx.get_iter(argC),
            fx.make_layout((M, N), (N, 1)),
        ))


        A = fx.rocdl.make_buffer_tensor(A_2d,  max_size=False)
        B = fx.rocdl.make_buffer_tensor(B_2d,  max_size=False)
        C = fx.rocdl.make_buffer_tensor(C_2d,  max_size=False)
        c_store_rsrc = fx.buffer_ops.create_buffer_resource(argC, max_size=True)

        bA_t = fx.flat_divide(A, (BLOCK_M, BLOCK_K))[None, None, bid_x*2 + 0, None]  # (BM, BK, k)
        bA_b = fx.flat_divide(A, (BLOCK_M, BLOCK_K))[None, None, bid_x*2 + 1, None]  # (BM, BK, k)
        bB_l = fx.flat_divide(B, (BLOCK_N, BLOCK_K))[None, None, bid_y*2 + 0, None]  # (BN, BK, k)
        bB_r = fx.flat_divide(B, (BLOCK_N, BLOCK_K))[None, None, bid_y*2 + 1, None]  # (BN, BK, k)
        
        bC_tl = fx.flat_divide(C, (BLOCK_M, BLOCK_N))[None, None, bid_x*2 + 0, bid_y*2 + 0]  # (BM, BN)
        bC_tr = fx.flat_divide(C, (BLOCK_M, BLOCK_N))[None, None, bid_x*2 + 0, bid_y*2 + 1]  # (BM, BN)
        bC_bl = fx.flat_divide(C, (BLOCK_M, BLOCK_N))[None, None, bid_x*2 + 1, bid_y*2 + 0]  # (BM, BN)
        bC_br = fx.flat_divide(C, (BLOCK_M, BLOCK_N))[None, None, bid_x*2 + 1, bid_y*2 + 1]  # (BM, BN)

        ################################################################################################
        ################################################################################################
        ################read subA, subB tensor layout for padding, swizzle and B preshuffled case.######
        #swizzle case:
        if const_expr(lds_swizzle):
            # swizzle 应用到静态形状的 tile 视图（而非 dynamic-M 的全局 A/B）：
            # 组合相同 num_shift 的 swizzle，形状全静态 -> layout-lowering 可正常 lower。
            # M 是 runtime 值，若组合到全局 A((M,K)) 会因动态 extent 无法 lower。
            # num_shift 仅依赖 K（编译期常量），分支 scope 隔离，故在此就地计算。
            _num_shift = K.bit_length() - 1 - 3
            _sw = fx.static(fx.SwizzleType.get(3, 3, _num_shift))
            bA_t = fx.Tensor(fx.make_view(fx.get_iter(bA_t), fx.make_composed_layout(_sw, fx.get_layout(bA_t))))
            bA_b = fx.Tensor(fx.make_view(fx.get_iter(bA_b), fx.make_composed_layout(_sw, fx.get_layout(bA_b))))
            bB_l = fx.Tensor(fx.make_view(fx.get_iter(bB_l), fx.make_composed_layout(_sw, fx.get_layout(bB_l))))
            bB_r = fx.Tensor(fx.make_view(fx.get_iter(bB_r), fx.make_composed_layout(_sw, fx.get_layout(bB_r))))
        #padding case:
        else:
            # A, B read layout
            bA_layout = fx.make_layout(((8, BLOCK_M//8), BLOCK_K, K//BLOCK_K), ((BLOCK_M//8*K, K), 1, BLOCK_K))
            bA_t = fx.Tensor(fx.make_view(fx.get_iter(bA_t), bA_layout))
            bA_b = fx.Tensor(fx.make_view(fx.get_iter(bA_b), bA_layout))
            bB_layout = fx.make_layout(((8, BLOCK_N//8), BLOCK_K, K//BLOCK_K), ((BLOCK_N//8*K, K), 1, BLOCK_K))
            bB_l = fx.Tensor(fx.make_view(fx.get_iter(bB_l), bB_layout))
            bB_r = fx.Tensor(fx.make_view(fx.get_iter(bB_r), bB_layout))

        # preshuffle B case：，与 swizzle/padding 无关。
        # 只在 preshuffle 时覆盖 bB_l/bB_r，A tensor 根据上面的padding和swizzle设置。
        if const_expr(preshuffle_b):
            _subB = fx.make_layout(((16, BLOCK_N//16), (8, BLOCK_K//8), K//BLOCK_K), ((8, 16*K), (1, 128), 1024))
            bB_l = fx.Tensor(fx.make_view(fx.get_iter(bB_l), _subB))
            bB_r = fx.Tensor(fx.make_view(fx.get_iter(bB_r), _subB))


        ################################################################################################
        ################################################################################################
        ###################### rd/wr LDS tensor view for padding, swizzle and preshuffleB###############
        #padding for rd and wr:
        lds_layout_rd =fx.make_layout(((16, 8), (32, 2)), ((512+PADDING_ELEMS, 64), (1, 32)))
        lds_layout_wr =fx.make_layout(((8, 16), 64), ((64, 8*64+PADDING_ELEMS), 1))
        # read and write LDS tensor view for swizzle.
        if const_expr(lds_swizzle):
            lds_layout_wr =fx.make_ordered_layout((BLOCK_M, BLOCK_K), (1, 0))
            lds_layout_rd = lds_layout_wr
            lds_layout_rd = fx.make_composed_layout(
                fx.static(fx.SwizzleType.get(3, 3, 3)),
                lds_layout_wr,
            )
        #lds b rd/wr layout settting:
        lds_layout_b_wr = lds_layout_wr
        lds_layout_b_rd = lds_layout_rd
        if const_expr(preshuffle_b):
            _b_lds = fx.make_layout(((16, BLOCK_N//16), (8, BLOCK_K//8)), ((8, 1024), (1, 128)))
            lds_layout_b_wr = _b_lds
            lds_layout_b_rd = _b_lds
            
        ################################################################################################
        ################################################################################################
        lds = fx.SharedAllocator().allocate(LDS_PADDING).peek()

        #LDS 0
        lds0_A_t_rd = fx.make_view(lds.lds0_a_t.ptr, lds_layout_rd)
        lds0_A_b_rd = fx.make_view(lds.lds0_a_b.ptr, lds_layout_rd)
        lds0_A_t_wr = fx.make_view(lds.lds0_a_t.ptr, lds_layout_wr)
        lds0_A_b_wr = fx.make_view(lds.lds0_a_b.ptr, lds_layout_wr)
        lds0_B_l_rd = fx.make_view(lds.lds0_b_l.ptr, lds_layout_b_rd)
        lds0_B_r_rd = fx.make_view(lds.lds0_b_r.ptr, lds_layout_b_rd)
        lds0_B_l_wr = fx.make_view(lds.lds0_b_l.ptr, lds_layout_b_wr)
        lds0_B_r_wr = fx.make_view(lds.lds0_b_r.ptr, lds_layout_b_wr)


        #LDS 1
        lds1_A_t_rd = fx.make_view(lds.lds1_a_t.ptr, lds_layout_rd)
        lds1_A_b_rd = fx.make_view(lds.lds1_a_b.ptr, lds_layout_rd)
        lds1_A_t_wr = fx.make_view(lds.lds1_a_t.ptr, lds_layout_wr)
        lds1_A_b_wr = fx.make_view(lds.lds1_a_b.ptr, lds_layout_wr)
        lds1_B_l_rd = fx.make_view(lds.lds1_b_l.ptr, lds_layout_b_rd)
        lds1_B_r_rd = fx.make_view(lds.lds1_b_r.ptr, lds_layout_b_rd)
        lds1_B_l_wr = fx.make_view(lds.lds1_b_l.ptr, lds_layout_b_wr)
        lds1_B_r_wr = fx.make_view(lds.lds1_b_r.ptr, lds_layout_b_wr)
        
        ################################################################################################
        ################################################################################################
        ######################## dma copy tiles ########################################################
        async_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopyLDS128b(), 128)
        lsd_copy_atom = fx.make_copy_atom(fx.UniversalCopy128b(), fx.BFloat16)
        buffer_copy_atom_bf16 = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16)
        buffer_copy_atom_f32 = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Float32)
        
        # DMA copy tiles
        ac_tile_mn = fx.make_tile(32, 64)
        ac_tv_layout =  fx.make_layout(((8, 8, 4), 8), ((8*4*8, 1, 8), 4*8))
        ac_tiled_copy = fx.make_tiled_copy(buffer_copy_atom_bf16, ac_tv_layout, ac_tile_mn)
        ac_thr = ac_tiled_copy.get_slice(tid)
        # preshuffle B 专属 tiled copy：tv=((16n,8k,2n),8k0):((1,256,16),32)，按 preshuffle 物理
        # 连续排布 -> coalesced gather；非 preshuffle 时 B 沿用 ac_thr（padding 路径不变）。
        if const_expr(preshuffle_b):
            b_tv_layout = fx.make_layout(((16, 8, 2), 8), ((1, 256, 16), 32))
            b_tiled_copy = fx.make_tiled_copy(buffer_copy_atom_bf16, b_tv_layout, fx.make_tile(32, 64))
            b_thr = b_tiled_copy.get_slice(tid)
        else:
            b_thr = ac_thr
        ################################################################################################
        ################################################################################################
        ######################## dma copy partition src/dest ###########################################
        ac_src_A_t = ac_thr.partition_S(bA_t)
        ac_src_A_b = ac_thr.partition_S(bA_b)
        ac_src_B_l = b_thr.partition_S(bB_l)
        ac_src_B_r = b_thr.partition_S(bB_r)
        #LDS0
        ac_dest0_A_t = ac_thr.partition_D(lds0_A_t_wr)
        ac_dest0_A_b = ac_thr.partition_D(lds0_A_b_wr)
        ac_dest0_B_l = b_thr.partition_D(lds0_B_l_wr)
        ac_dest0_B_r = b_thr.partition_D(lds0_B_r_wr)
        #LDS1
        ac_dest1_A_t = ac_thr.partition_D(lds1_A_t_wr)
        ac_dest1_A_b = ac_thr.partition_D(lds1_A_b_wr)
        ac_dest1_B_l = b_thr.partition_D(lds1_B_l_wr)
        ac_dest1_B_r = b_thr.partition_D(lds1_B_r_wr)

        # tiled MMA, thread MMA
        mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, fx.BFloat16))
        #tiled_mma = fx.make_tiled_mma(mma_atom, fx.make_layout((2, 2, 1), (2, 1, 0)))
        tiled_mma = fx.make_tiled_mma(mma_atom, fx.make_layout((2, 2, 1), (1, 2, 0)))
        thr_mma = tiled_mma.thr_slice(tid)
        # MMA copy A, B, C tiled copy
        s2r_tiled_copy_A = fx.make_tiled_copy_A(buffer_copy_atom_bf16, tiled_mma)
        s2r_tiled_copy_B = fx.make_tiled_copy_B(buffer_copy_atom_bf16, tiled_mma)
        # C tiled copy. make_tiled_copy_C is not used because C= B*A
  
        #MMA fragments
        #fragA layout:((a_val), m_rep, k_rep)
        #fragB layout:((b_val), n_rep, k_rep)
        #fragC layout:((c_val), m_rep, n_rep)

        #op1是A，op是B, fx.gemm(mma_atom, result, op1, op2, op3)的代码行为应该是：
        #C=A*B的情况下m_iter是m_rep, n_iter就是n_rep
        # m_iter = op1.shape[1]
        # n_iter = op2.shape[1]
        # k_iter = op1.shape[2] 
        # for m in range (m_iter):
        #     for n in range (n_iter):
        #         for k in range (k_iter):
        #             frag_C[None, m, n] += frag_A[None, m, k] * frag_B[None, k, n]

        #c=B*A, fx.gemm(mma_atom, C, B, A, C)
        #所以m_iter = n_rep, n_iter = m_rep, 
        #对frgaC的访问，frag_C[None, m_iter, n_iter]实际上是frag_C[None, n_rep, m_rep]
        frag_A_t = thr_mma.make_fragment_A(lds0_A_t_rd)
        frag_A_b = thr_mma.make_fragment_A(lds0_A_b_rd)
        frag_B_l = thr_mma.make_fragment_B(lds0_B_l_rd)
        frag_B_r = thr_mma.make_fragment_B(lds0_B_r_rd)
        #frag_C(val, m_rep, n_rep] -> frag_C[val, n_rep, m_rep]
        frag_C_tl = thr_mma.make_fragment_C(fx.select(bC_tl,[1,0]))
        frag_C_tr = thr_mma.make_fragment_C(fx.select(bC_tr,[1,0]))
        frag_C_bl = thr_mma.make_fragment_C(fx.select(bC_bl,[1,0]))
        frag_C_br = thr_mma.make_fragment_C(fx.select(bC_br,[1,0]))

        # print(f'##frag_A_t={frag_A_t}')
        # print(f'##frag_A_b={frag_A_b}')
        # print(f'##frag_B_l={frag_B_l}')
        # print(f'##frag_B_r={frag_B_r}')
        
        # print(f'##frag_C_tl={frag_C_tl}')
        # print(f'##frag_C_tr={frag_C_tr}')
        # print(f'##frag_C_bl={frag_C_bl}')
        # print(f'##frag_C_br={frag_C_br}')
        # from LDS to reigster partition
        ldsA_rd_thread = s2r_tiled_copy_A.get_slice(tid)
        ldsB_rd_thread = s2r_tiled_copy_B.get_slice(tid)
        s2r_src0_A_t = ldsA_rd_thread.partition_S(lds0_A_t_rd)
        s2r_src0_A_b = ldsA_rd_thread.partition_S(lds0_A_b_rd)
        s2r_src0_B_l = ldsB_rd_thread.partition_S(lds0_B_l_rd)
        s2r_src0_B_r = ldsB_rd_thread.partition_S(lds0_B_r_rd)
        
        s2r_src1_A_t = ldsA_rd_thread.partition_S(lds1_A_t_rd)
        s2r_src1_A_b = ldsA_rd_thread.partition_S(lds1_A_b_rd)
        s2r_src1_B_l = ldsB_rd_thread.partition_S(lds1_B_l_rd)
        s2r_src1_B_r = ldsB_rd_thread.partition_S(lds1_B_r_rd)
        ###MMA fragments retile to des
        dest_frag_A_t = ldsA_rd_thread.retile(frag_A_t)
        dest_frag_A_b = ldsA_rd_thread.retile(frag_A_b)
        dest_frag_B_l = ldsB_rd_thread.retile(frag_B_l)
        dest_frag_B_r = ldsB_rd_thread.retile(frag_B_r)

        frag_C_tl.store(Vector.filled(BLOCK_M * BLOCK_N // 64 // 4, 0, fx.Float32))
        frag_C_tr.store(Vector.filled(BLOCK_M * BLOCK_N // 64 // 4, 0, fx.Float32))
        frag_C_bl.store(Vector.filled(BLOCK_M * BLOCK_N // 64 // 4, 0, fx.Float32))
        frag_C_br.store(Vector.filled(BLOCK_M * BLOCK_N // 64 // 4, 0, fx.Float32))
        acc_init = [frag_C_tl.load(), frag_C_tr.load(), frag_C_bl.load(), frag_C_br.load()]
        
        rocdl.sched_barrier(0)
        fx.copy(async_copy_atom, ac_src_B_l[None, None, None, 0], ac_dest0_B_l)
        rocdl.sched_barrier(0)
        fx.copy(async_copy_atom, ac_src_A_t[None, None, None, 0], ac_dest0_A_t)
        rocdl.sched_barrier(0)
        fx.copy(async_copy_atom, ac_src_A_b[None, None, None, 0], ac_dest0_A_b)
        rocdl.sched_barrier(0)
        fx.copy(async_copy_atom, ac_src_B_r[None, None, None, 0], ac_dest0_B_r)
        rocdl.sched_barrier(0)

        fx.copy(async_copy_atom, ac_src_B_l[None, None, None, 1], ac_dest1_B_l)
        rocdl.sched_barrier(0)

        fx.copy(async_copy_atom, ac_src_A_t[None, None, None, 1], ac_dest1_A_t)
        rocdl.sched_barrier(0)

        fx.copy(async_copy_atom, ac_src_A_b[None, None, None, 1], ac_dest1_A_b)
        rocdl.sched_barrier(0)
        fx.copy(async_copy_atom, ac_src_B_r[None, None, None, 1], ac_dest1_B_r)
        rocdl.sched_barrier(0)

        nrM = BLOCK_M // (16*2)
        nrN = BLOCK_N // (16*2)
        nrK = BLOCK_K // 32
        mfma_cnt = nrM * nrN * nrK
        assert nrM == nrN
        elems_per_lane = 8 if dtype == "bf16" else 16
        vm_load_cnt = (BLOCK_M*BLOCK_K) // 256 // elems_per_lane
        ds_rd_cnt = vm_load_cnt * 2
        vm_store_cnt = (BLOCK_M*BLOCK_N) // 256 // elems_per_lane
        if not permlane_epilogue:
            vm_store_cnt *= 2

        waitvmcnt_barrier(6*vm_load_cnt)
        gpu.barrier()
        fx.copy(lsd_copy_atom, s2r_src0_B_l, dest_frag_B_l, pred=None)
        fx.copy(lsd_copy_atom, s2r_src0_A_t, dest_frag_A_t, pred=None)
        rocdl.sched_barrier(0)
        
        frag_C_tl.fill(0)
        frag_C_tr.fill(0)
        frag_C_bl.fill(0)
        frag_C_br.fill(0)
        rocdl.sched_barrier(0)

        for kidx, states in range(0, K // BLOCK_K - 2, 2, init=acc_init):    
        # for kiter in const_expr.range(0, K // BLOCK_K - 2, 2):
            frag_C_tl.store(states[0])
            frag_C_tr.store(states[1])
            frag_C_bl.store(states[2])
            frag_C_br.store(states[3])
            kiter = fx.Int32(kidx)
            mfma_16x16x64 = Mfma16x16x64(4, 4, mma_atom, mode=mfma_mode)

            waitvmcnt_barrier(5*vm_load_cnt)
            mfma_16x16x64.call_BxA(frag_A_t, frag_B_l, frag_C_tl)
            fx.copy(lsd_copy_atom, s2r_src0_A_b, dest_frag_A_b, pred=None)
            fx.copy(async_copy_atom, ac_src_B_l[None, None, None, kiter+2], ac_dest0_B_l)
            hot_loop_scheduler(0, vm_load_cnt, ds_rd_cnt, mfma_cnt)
            anchor_frag(frag_B_l)
            rocdl.sched_barrier(0)

            waitvmcnt_barrier(5*vm_load_cnt)
            mfma_16x16x64.call_BxA(frag_A_b, frag_B_l, frag_C_bl)
            fx.copy(lsd_copy_atom, s2r_src0_B_r, dest_frag_B_r, pred=None)
            fx.copy(async_copy_atom, ac_src_A_t[None, None, None, kiter+2], ac_dest0_A_t)
            hot_loop_scheduler(1, vm_load_cnt, ds_rd_cnt, mfma_cnt)
            anchor_frag(frag_B_l)
            rocdl.sched_barrier(0)
            
            waitvmcnt_barrier(5*vm_load_cnt)
            mfma_16x16x64.call_BxA(frag_A_t, frag_B_r, frag_C_tr)
            fx.copy(lsd_copy_atom, s2r_src1_B_l, dest_frag_B_l, pred=None)
            fx.copy(async_copy_atom, ac_src_A_b[None, None, None, kiter+2], ac_dest0_A_b)
            hot_loop_scheduler(2, vm_load_cnt, ds_rd_cnt, mfma_cnt)
            anchor_frag(frag_B_r)
            rocdl.sched_barrier(0)
        
            waitvmcnt_barrier(5*vm_load_cnt)
            mfma_16x16x64.call_BxA(frag_A_b, frag_B_r, frag_C_br)
            fx.copy(lsd_copy_atom, s2r_src1_A_t, dest_frag_A_t, pred=None)
            fx.copy(async_copy_atom, ac_src_B_r[None, None, None, kiter+2], ac_dest0_B_r)
            hot_loop_scheduler(3, vm_load_cnt, ds_rd_cnt, mfma_cnt)
            anchor_frag(frag_B_r)
            rocdl.sched_barrier(0)

            waitvmcnt_barrier(5*vm_load_cnt)
            mfma_16x16x64.call_BxA(frag_A_t, frag_B_l, frag_C_tl)
            fx.copy(lsd_copy_atom, s2r_src1_A_b, dest_frag_A_b, pred=None)
            fx.copy(async_copy_atom, ac_src_B_l[None, None, None, kiter+3], ac_dest1_B_l)
            hot_loop_scheduler(4, vm_load_cnt, ds_rd_cnt, mfma_cnt)
            anchor_frag(frag_B_l)
            rocdl.sched_barrier(0)
            
            waitvmcnt_barrier(5*vm_load_cnt)
            mfma_16x16x64.call_BxA(frag_A_b, frag_B_l, frag_C_bl)
            fx.copy(lsd_copy_atom, s2r_src1_B_r, dest_frag_B_r, pred=None)
            fx.copy(async_copy_atom, ac_src_A_t[None, None, None, kiter+3], ac_dest1_A_t)
            hot_loop_scheduler(5, vm_load_cnt, ds_rd_cnt, mfma_cnt)
            anchor_frag(frag_B_l)
            rocdl.sched_barrier(0)
            
            waitvmcnt_barrier(5*vm_load_cnt)
            mfma_16x16x64.call_BxA(frag_A_t, frag_B_r, frag_C_tr)
            fx.copy(lsd_copy_atom, s2r_src0_B_l, dest_frag_B_l, pred=None)
            fx.copy(async_copy_atom, ac_src_A_b[None, None, None, kiter+3], ac_dest1_A_b)
            hot_loop_scheduler(6, vm_load_cnt, ds_rd_cnt, mfma_cnt)
            anchor_frag(frag_B_r)
            rocdl.sched_barrier(0)
            
            waitvmcnt_barrier(5*vm_load_cnt)
            mfma_16x16x64.call_BxA(frag_A_b, frag_B_r, frag_C_br)
            fx.copy(lsd_copy_atom, s2r_src0_A_t, dest_frag_A_t, pred=None)
            fx.copy(async_copy_atom, ac_src_B_r[None, None, None, kiter+3], ac_dest1_B_r)
            hot_loop_scheduler(7, vm_load_cnt, ds_rd_cnt, mfma_cnt)
            anchor_frag(frag_B_r)
            rocdl.sched_barrier(0)

            results = yield [frag_C_tl.load(), frag_C_tr.load(), frag_C_bl.load(), frag_C_br.load()]
        #frag_C(val, n_rep, m_rep] -> frag_C[val, m_rep, n_rep]
        frag_C_tl.store(results[0])
        frag_C_tr.store(results[1])
        frag_C_bl.store(results[2])
        frag_C_br.store(results[3])

            
        mfma_16x16x64 = Mfma16x16x64(4, 4, mma_atom, mode=mfma_mode)

        waitvmcnt_barrier(5*vm_load_cnt)
        mfma_16x16x64.call_BxA(frag_A_t, frag_B_l, frag_C_tl)
        fx.copy(lsd_copy_atom, s2r_src0_A_b, dest_frag_A_b, pred=None)
        hot_loop_scheduler(0, 0, ds_rd_cnt, mfma_cnt)
        rocdl.sched_barrier(0)

        waitvmcnt_barrier(4*vm_load_cnt)
        mfma_16x16x64.call_BxA(frag_A_b, frag_B_l, frag_C_bl)
        fx.copy(lsd_copy_atom, s2r_src0_B_r, dest_frag_B_r, pred=None)
        hot_loop_scheduler(1, 0, ds_rd_cnt, mfma_cnt)
        rocdl.sched_barrier(0)

        waitvmcnt_barrier(3*vm_load_cnt)
        mfma_16x16x64.call_BxA(frag_A_t, frag_B_r, frag_C_tr)
        fx.copy(lsd_copy_atom, s2r_src1_B_l, dest_frag_B_l, pred=None)
        hot_loop_scheduler(2, 0, ds_rd_cnt, mfma_cnt)
        rocdl.sched_barrier(0)

        waitvmcnt_barrier(2*vm_load_cnt)
        mfma_16x16x64.call_BxA(frag_A_b, frag_B_r, frag_C_br)
        fx.copy(lsd_copy_atom, s2r_src1_A_t, dest_frag_A_t, pred=None)
        hot_loop_scheduler(3, 0, ds_rd_cnt, mfma_cnt)
        rocdl.sched_barrier(0)

        # epilogue store：提前建立 store 辅助，使其能与最后的 MFMA 交织，互相掩盖延迟。
        if const_expr(permlane_epilogue):
            # permlane 方式：两个相邻 16x16 tile 经 permlane16_swap 重排后，
            # 每个 lane 一次 store 8 个连续 bf16（128-bit 合并写）。
            # 注意：v9 的 C 寄存器->坐标映射由 c_tv_layout 决定，output-M 波是
            # wave_id%2、output-N 波是 wave_id//2，
            pair_type = ir.Type.parse("!llvm.struct<(i32, i32)>")
            lane_id = tid % 64
            wave_id = tid // 64
            wave_m = wave_id % 2
            wave_n = wave_id // 2
            lane_group = lane_id // 16
            fragment_mode_0_repeat = TILE_N // 64
            fragment_mode_1_repeat = TILE_M // 64

            def store_quadrant(c_frag, bC, quadrant_m, quadrant_n):
                for row_repeat in range_constexpr(fragment_mode_1_repeat):
                    for col_repeat in range_constexpr(0, fragment_mode_0_repeat, 2):
                        acc_a = Vec(c_frag[None, col_repeat, row_repeat].load())
                        acc_b = Vec(c_frag[None, col_repeat + 1, row_repeat].load())
                        d0_a = rocdl.cvt_pk_bf16_f32(acc_a[0], acc_a[1])
                        d1_a = rocdl.cvt_pk_bf16_f32(acc_a[2], acc_a[3])
                        d0_b = rocdl.cvt_pk_bf16_f32(acc_b[0], acc_b[1])
                        d1_b = rocdl.cvt_pk_bf16_f32(acc_b[2], acc_b[3])
                        swap0 = rocdl.permlane16_swap(
                            pair_type,
                            arith._to_raw(d0_a),
                            arith._to_raw(d0_b),
                            False,
                            False,
                        )
                        swap1 = rocdl.permlane16_swap(
                            pair_type,
                            arith._to_raw(d1_a),
                            arith._to_raw(d1_b),
                            False,
                            False,
                        )
                        packed = Vec.from_elements(
                            [
                                fx.Int32(_llvm.extractvalue(T.i32, swap0, [0])),
                                fx.Int32(_llvm.extractvalue(T.i32, swap1, [0])),
                                fx.Int32(_llvm.extractvalue(T.i32, swap0, [1])),
                                fx.Int32(_llvm.extractvalue(T.i32, swap1, [1])),
                            ],
                            fx.Int32,
                        )
                        row = (
                            bid_x * TILE_M
                            + quadrant_m * (TILE_M // 2)
                            + row_repeat * 32
                            + wave_m * 16
                            + lane_id % 16
                        )
                        col = (
                            bid_y * TILE_N
                            + quadrant_n * (TILE_N // 2)
                            + col_repeat * 32
                            + lane_group % 2 * 32
                            + wave_n * 16
                            + lane_group // 2 * 8
                        )
                        byte_offset = (row * N + col) * 2
                        fx.buffer_ops.buffer_store(
                            packed,
                            c_store_rsrc,
                            byte_offset,
                            offset_is_bytes=True,
                        )
        else:
            # 简单 bf16 存储：f32 累加器转 bf16，
            c_tile_mn = fx.make_tile(32, 32)
            # wave ((2, 2, 1), (1, 2, 0)):
            c_tv_layout =  fx.make_layout((((16, 4), 2, 2), 4), (((1, 128), 16, 512) , 32)) 
            store_atom_bf16 = fx.make_copy_atom(fx.rocdl.BufferCopy64b(), fx.BFloat16)
            tiled_copy_C_bf16 = fx.make_tiled_copy(store_atom_bf16, c_tv_layout, c_tile_mn)
            thr_copy_C_bf16 = tiled_copy_C_bf16.get_slice(tid)

            def store_quadrant(c_frag, bC, quadrant_m, quadrant_n):
                c_sel = fx.select(c_frag, [0, 2, 1])
                c_bf16 = fx.make_fragment_like(c_sel, dtype=fx.BFloat16)
                c_bf16.store(c_sel.load().to(fx.BFloat16))
                fx.copy(store_atom_bf16, thr_copy_C_bf16.retile(c_bf16), thr_copy_C_bf16.partition_D(bC))

        waitvmcnt_barrier(vm_load_cnt)
        mfma_16x16x64.call_BxA(frag_A_t, frag_B_l, frag_C_tl)
        fx.copy(lsd_copy_atom, s2r_src1_A_b, dest_frag_A_b, pred=None)
        hot_loop_scheduler(4, 0, ds_rd_cnt, mfma_cnt)
        rocdl.sched_barrier(0)

        waitvmcnt_barrier(0)
        # bl 的 FMA 与 tl 的存储互相掩盖
        mfma_16x16x64.call_BxA(frag_A_b, frag_B_l, frag_C_bl)
        fx.copy(lsd_copy_atom, s2r_src1_B_r, dest_frag_B_r, pred=None)
        store_quadrant(frag_C_tl, bC_tl, 0, 0)
        scheduler_store_overlap(5, vm_store_cnt, ds_rd_cnt, mfma_cnt)
        rocdl.sched_barrier(0)

        # tr 的 FMA 掩盖 bl 的存储
        mfma_16x16x64.call_BxA(frag_A_t, frag_B_r, frag_C_tr)
        store_quadrant(frag_C_bl, bC_bl, 1, 0)
        scheduler_store_overlap(6, vm_store_cnt, 0, mfma_cnt)
        rocdl.sched_barrier(0)

        # br 的 FMA 掩盖 tr 的存储
        mfma_16x16x64.call_BxA(frag_A_b, frag_B_r, frag_C_br)
        store_quadrant(frag_C_tr, bC_tr, 0, 1)
        scheduler_store_overlap(7, vm_store_cnt, 0, mfma_cnt)
        rocdl.sched_barrier(0)

        # 最后 br 单独存储
        store_quadrant(frag_C_br, bC_br, 1, 1)

    @flyc.jit
    def launch_gemm(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
        M: int,
        stream: fx.Stream = fx.Stream(None),
    ):
        
        value_attrs = {"rocdl.waves_per_eu": 1,
                    "passthrough": [["amdgpu-agpr-alloc", "256,256"],]
                    }
        gemm_kernel(A, B, C, M, value_attrs=value_attrs,).launch(grid=(div_up(M, TILE_M)*div_up(N, TILE_N), 1, 1), block=(256, 1, 1), stream=stream)
        
    launch_gemm.compile_hints["llvm_options"] = {
        "amdgpu-mfma-vgpr-form": False,
    }
    return launch_gemm


import pyhip



def _load_shuffle_weight():
    # host 端 shuffle_weight 在 tests.utils，延迟加载避免非 preshuffle 路径依赖它。
    import sys as _sys, os.path as _osp
    _flydsl_root = _osp.abspath(_osp.join(_osp.dirname(__file__), "..", ".."))
    if _flydsl_root not in _sys.path:
        _sys.path.insert(0, _flydsl_root)
    from tests.utils import shuffle_weight
    return shuffle_weight


def run_test(M, N, K, USE_SWIZZLE=False, PRESHUFFLE_B=False, perf=False,
             TILEM=256, TILEN=256, TILEK=64, run_count=16, data_clones=32):
    shuffle_weight = _load_shuffle_weight() if PRESHUFFLE_B else None
    enable_dump_ir(False)

    OUT_DTYPE = torch.bfloat16
    OUT_ATOL = 0.03
    OUT_RTOL = 0.01
    OUT_BYTES = 2
    # MFMA 发射方式: ROCDL(调度器自由重排,k外提) / INLINE_ASM_K2(k0k1背靠背) / FX_GEMM
    # MFMA_MODE = MfmaMode[os.environ.get("MFMA_MODE", "ROCDL")]
    MFMA_MODE = MfmaMode["ROCDL"]
    # True: permlane 方式（一次 store 8 个 bf16）；False: 简单 bf16 tiled-copy 存储。两者都输出 bf16。
    PERMLANE_EPILOGUE = True

    A = torch.randn(M, K, dtype=torch.bfloat16).cuda() / math.sqrt(K)
    B = torch.randn(N, K, dtype=torch.bfloat16).cuda() / math.sqrt(K)
    C = torch.zeros(M, N, dtype=OUT_DTYPE).cuda()
    expected = A.to(torch.float32) @ B.to(torch.float32).T
    # preshuffle 时喂给 kernel 的是 shuffle 后的 B；expected 仍用原始 B。
    weight = shuffle_weight(B, layout=(16, 16)) if PRESHUFFLE_B else B

    hints = {"opt_level": 2, "llvm_options": {"amdgpu-mfma-vgpr-form": False}}
    stream = torch.cuda.current_stream()
    launcher_gemm = compile_gemm(
        TILE_M=TILEM,
        TILE_N=TILEN,
        TILE_K=TILEK,
        N=N,
        K=K,
        dtype="bf16",
        lds_swizzle=USE_SWIZZLE,
        pid_swizzle=True,
        permlane_epilogue=PERMLANE_EPILOGUE,
        mfma_mode=MFMA_MODE,
        preshuffle_b=PRESHUFFLE_B)

    compiled_gemm = flyc.compile[hints](launcher_gemm, A, weight, C, M, stream)
    compiled_gemm(A, weight, C, M, stream)
    torch.cuda.synchronize()

    torch.set_printoptions(linewidth=3000, sci_mode=False, edgeitems=8)
    is_correct = torch.allclose(expected, C.to(torch.float32), atol=OUT_ATOL, rtol=OUT_RTOL)
    print(f'####M={M} N={N} K={K} {USE_SWIZZLE=} {PRESHUFFLE_B=} {is_correct=}')

    if not perf:
        return is_correct

    # ---- perf（多份数据轮转，排除 L2 cache 影响）----
    As = [torch.randn(M, K, dtype=torch.bfloat16).cuda() for _ in range(data_clones)]
    Bs = [torch.randn(N, K, dtype=torch.bfloat16).cuda() for _ in range(data_clones)]
    if PRESHUFFLE_B:
        Bs = [shuffle_weight(_b, layout=(16, 16)) for _b in Bs]
    Cs = [torch.zeros(M, N, dtype=OUT_DTYPE).cuda() for _ in range(data_clones)]

    flops = 2 * M * N * K
    mem_bytes = (M * K + N * K) * 2 + M * N * OUT_BYTES  # bf16 A+B + C

    di = 0
    latencies = []
    torch_latencies = []
    for _ in range(run_count):
        di = (di + 1) % data_clones
        with pyhip.cudaPerf(flops, mem_bytes, name=f"gemm_{di}") as p:
            compiled_gemm(As[di], Bs[di], Cs[di], M, stream)
        latencies.append(p.dt_ms)
    for _ in range(run_count):
        di = (di + 1) % data_clones
        with pyhip.cudaPerf(flops, mem_bytes, name=f"torch_{di}") as p:
            _ = torch.nn.functional.linear(As[di], Bs[di])
        torch_latencies.append(p.dt_ms)

    latencies.sort()
    torch_latencies.sort()
    best_ms = latencies[0]
    tflops = flops / (best_ms * 1e-3) / 1e12
    bw_gbs = mem_bytes / (best_ms * 1e-3) / 1e9

    print(f"\n=== perf  M={M} N={N} K={K} USE_SWIZZLE={USE_SWIZZLE} PRESHUFFLE_B={PRESHUFFLE_B} ===")
    print(f"gemm:  {best_ms*1e3:.1f} us  {tflops:.2f} TFLOPS  {bw_gbs:.1f} GB/s")
    print(f"torch: {torch_latencies[0]*1e3:.1f} us")
    print(f"ratio: {torch_latencies[0]/best_ms:.2f}x")
    return is_correct


run_test(M=256 * 32,N=256 * 32,K=64 * 128,USE_SWIZZLE=0,PRESHUFFLE_B=0,perf=1,)
run_test(M=256 * 32,N=256 * 32,K=64 * 128,USE_SWIZZLE=1,PRESHUFFLE_B=0,perf=1,)
run_test(M=256 * 32,N=256 * 32,K=64 * 128,USE_SWIZZLE=0,PRESHUFFLE_B=1,perf=1,)
run_test(M=256 * 32,N=256 * 32,K=64 * 128,USE_SWIZZLE=1,PRESHUFFLE_B=1,perf=1,)