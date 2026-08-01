"""旋转反相 fused-attention kernel 的精度与性能测试。

GEMM1计算S^T=K@Q^T，online softmax就地生成P^T，GEMM2计算O^T=V^T@P^T。K走LDS
ping-pong，V从paged global直接进入fragment。stage0执行K global预取与softmax；stage1执行K LDS
写读、GEMM2和下一tile的V预取/GEMM1，并跨runtime loop回边维持resident-wave反相。

当前维护记录见attn_4wave/attn_opt.md。
"""

import os

os.environ.setdefault("FLYDSL_RUNTIME_ENABLE_CACHE", "0")

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, const_expr, gpu, range_constexpr, rocdl
from flydsl._mlir.dialects import llvm

_SCHED_MASK_VALU = 0x002
_SCHED_MASK_DS_WRITE = 0x200
_SCHED_MASK_TRANS = 0x400
_EXP_DSWR_SYNC_ID = 1


def _sched_valu(count, sync_id=_EXP_DSWR_SYNC_ID):
    rocdl.sched_group_barrier(_SCHED_MASK_VALU, count, sync_id)


def _sched_ds_write(count, sync_id=_EXP_DSWR_SYNC_ID):
    rocdl.sched_group_barrier(_SCHED_MASK_DS_WRITE, count, sync_id)


def _sched_trans(count, sync_id=_EXP_DSWR_SYNC_ID):
    rocdl.sched_group_barrier(_SCHED_MASK_TRANS, count, sync_id)


# debug
if 1:
    from flydsl.utils.env import DebugEnvManager
    from flydsl._mlir import ir
    import flydsl

    DebugEnvManager.enable_debug_info = True
    ir._globals.register_traceback_file_inclusion(__file__)
    ir._globals.register_traceback_file_exclusion(os.path.dirname(flydsl.__file__))
    ir._globals.set_loc_tracebacks_frame_limit(40)
    ir._globals.set_loc_tracebacks_enabled(True)
    os.environ.setdefault("FLYDSL_RUNTIME_ENABLE_CACHE", "0")

# 模块级缓存:静态配置 -> flyc.compile 的快速派发 callable。参考 tests/contrib/moe/test_moe.py。
# flyc.compile 返回预建 CallState 的 callable(每次调用 ~6us),避开完整 JitFunction.__call__
# 路径(~140us:签名绑定 + 重建 cache-key + globals/runtime 检查),降低 host 侧开销。
_FLY_COMPILED_CACHE = {}


def fly_compiled(key, build_launch, args):
    """用 flyc.compile 编译并缓存,返回快速派发 callable。

    首次(cache miss):flyc.compile 追踪、编译并**执行一次** kernel(填充输出),缓存 callable。
    后续(cache hit):直接调用缓存 callable 一次。
    """
    compiled = _FLY_COMPILED_CACHE.get(key)
    if compiled is None:
        compiled = flyc.compile(build_launch(), *args)  # 编译 + 执行一次
        _FLY_COMPILED_CACHE[key] = compiled
    else:
        compiled(*args)
    return compiled


def _cvt_f32_to_bf16(c_frag):
    """f32 -> bf16:add 0x8000 舍入 + 截断(round-half-up),比 .to(fx.BFloat16) 的 RNE + NaN 处理少指令。
    移植自 src/contrib/flydsl/moe_gemm_splitk.py::_cvt_f32_to_bf16。"""
    c_frag_bf16 = fx.make_fragment_like(c_frag, dtype=fx.BFloat16)
    round_bit = fx.Uint32(0x8000)
    c_frag_bf16.store(
        ((c_frag.load().bitcast(fx.Uint32) + round_bit) >> 16)
        .to(fx.Uint16)
        .bitcast(fx.BFloat16)
    )
    return c_frag_bf16


LOG2E = 1.4426950408889634  # log2(e):exp(x)=exp2(x*LOG2E),flash softmax 用 exp2 走单 v_exp_f32
FP8_PROB_SCALE = 240.0
FP8_V_ROW_ORDER = (
    0,
    1,
    2,
    3,
    16,
    17,
    18,
    19,
    4,
    5,
    6,
    7,
    20,
    21,
    22,
    23,
    8,
    9,
    10,
    11,
    24,
    25,
    26,
    27,
    12,
    13,
    14,
    15,
    28,
    29,
    30,
    31,
)
FP8_MMA32_V_ROW_ORDER = (
    0,
    1,
    2,
    3,
    8,
    9,
    10,
    11,
    16,
    17,
    18,
    19,
    24,
    25,
    26,
    27,
    4,
    5,
    6,
    7,
    12,
    13,
    14,
    15,
    20,
    21,
    22,
    23,
    28,
    29,
    30,
    31,
)


def _cvt_f32x4_to_fp8(vec):
    """用gfx942原生v_cvt_pk_fp8_f32将4个f32打包为E4M3FNUZ。"""
    scaled = vec * fx.Float32(FP8_PROB_SCALE)
    packed = fx.Int32(0)
    packed = fx.Int32(
        rocdl.cvt_pk_fp8_f32(fx.Int32.ir_type, scaled[0], scaled[1], packed, False)
    )
    packed = fx.Int32(
        rocdl.cvt_pk_fp8_f32(fx.Int32.ir_type, scaled[2], scaled[3], packed, True)
    )
    return fx.Vector.from_elements([packed], fx.Int32).bitcast(fx.Float8E4M3FNUZ)


def _preshuffle_v(V, qkv_dtype, mma_mn=16):
    """转成kernel的[D,(8,NB),N/BN]物理布局；FP8额外补偿score row置换。"""
    H, N, D = V.shape
    if qkv_dtype == "fp8":
        row_order = FP8_MMA32_V_ROW_ORDER if mma_mn == 32 else FP8_V_ROW_ORDER
        V = V.reshape(H, N // 32, 32, D)[:, :, row_order, :]
    return V.reshape(H, N // 8, 8, D).permute(0, 1, 3, 2).contiguous()


def _maxnumf(a, b):
    """非 NaN 传播 max(单 v_max_f32),移植自 pa_decode_fp8.py。softmax 输入有限或 -inf,不会 NaN。"""
    return type(a)(arith.maxnumf(arith.unwrap(a), arith.unwrap(b)))


def _exp2_amdgcn(x):
    """标量 f32 = 2^x:llvm.amdgcn.exp2.f32(单 v_exp_f32,省 OCML 的 v_ldexp)。指数须在 fast-range(≤0)。"""
    from flydsl._mlir.ir import F32Type

    return fx.Float32(
        llvm.call_intrinsic(
            F32Type.get(), "llvm.amdgcn.exp2.f32", [arith.unwrap(x)], [], []
        )
    )


def _exp2_vec_amdgcn(vec):
    """fx.Vector f32 逐元素 2^vec(amdgcn intrinsic)。"""
    from flydsl._mlir.dialects import vector as _vd
    from flydsl._mlir.ir import F32Type, VectorType

    raw = arith.unwrap(vec)
    n = raw.type.shape[0]
    f32 = F32Type.get()
    outs = [
        llvm.call_intrinsic(
            f32,
            "llvm.amdgcn.exp2.f32",
            [_vd.extract(raw, static_position=[i], dynamic_position=[])],
            [],
            [],
        )
        for i in range(n)
    ]
    return fx.Vector(_vd.from_elements(VectorType.get([n], f32), outs))


def _fma_f32_inline(a, b, c, negate_c=False):
    """强制单条scalar v_fma_f32，避免LLVM合并成不能与MFMA共发的v_pk算术。"""
    from flydsl._mlir.ir import F32Type

    instruction = (
        "v_fma_f32 $0, $1, $2, -$3" if negate_c else "v_fma_f32 $0, $1, $2, $3"
    )
    return fx.Float32(
        llvm.inline_asm(
            F32Type.get(),
            [arith.unwrap(a), arith.unwrap(b), arith.unwrap(c)],
            instruction,
            "=v,v,v,v",
            has_side_effects=False,
        )
    )


def _read_hw_wave_slot():
    """读取gfx9 HW_ID.WAVE_ID[3:0]，即当前SIMD内的物理wave slot。"""
    return fx.Int32(
        llvm.inline_asm(
            fx.Int32.ir_type,
            [],
            "s_getreg_b32 $0, hwreg(HW_REG_HW_ID, 0, 4)",
            "=s",
            has_side_effects=True,
        )
    )


def _set_hw_slot_priority(wave_slot, slot0_priority, slot1_priority):
    """按物理wave slot选择立即数s_setprio；非slot0按slot1处理。"""
    from flydsl._mlir import ir

    llvm.inline_asm(
        ir.Type.parse("!llvm.void"),
        [arith.unwrap(wave_slot)],
        (
            "s_cmp_eq_u32 $0, 0\n\t"
            "s_cbranch_scc0 1f\n\t"
            f"s_setprio {slot0_priority}\n\t"
            "s_branch 2f\n\t"
            "1:\n\t"
            f"s_setprio {slot1_priority}\n\t"
            "2:"
        ),
        "s",
        has_side_effects=True,
    )


def _fma_vec_f32_inline(a, b, c):
    values = []
    for i in range_constexpr(a.numel):
        values.append(_fma_f32_inline(a[i], b, c[i]))
    return fx.Vector.from_elements(values, fx.Float32)


def _scale_center_vec_f32_inline(score, scale, center):
    values = []
    for i in range_constexpr(score.numel):
        values.append(_fma_f32_inline(score[i], scale, center, negate_c=True))
    return fx.Vector.from_elements(values, fx.Float32)


def _make_klds_view(ptr, BN, D, is_fp8):
    """K LDS视图：BF16使用SwizzleType(3,3,3)，FP8保持plain row-major。"""
    base = fx.make_layout((2, BN, D), (BN * D, D, 1))
    if is_fp8:
        return fx.make_view(ptr, base)
    swz = fx.SwizzleType.get(3, 3, 3)
    return fx.make_view(ptr, fx.make_composed_layout(fx.static(swz), base))


def _make_ktiles(K_, N, D, BN, koff, is_fp8, mma_mn):
    """K coop读源tile；BF16物理重排32行，FP8保持plain row-major。"""
    layout = (
        fx.make_layout((BN, D, N // BN, 1), (D, 1, BN * D, 0))
        if is_fp8
        else (
            fx.make_layout(
                ((4, 2, 2, 2), D, N // BN, 1),
                ((D, 8 * D, 4 * D, 16 * D), 1, BN * D, 0),
            )
            if mma_mn == 32
            else fx.make_layout(
                ((4, 4, 2), D, N // BN, 1),
                ((D, 8 * D, 4 * D), 1, BN * D, 0),
            )
        )
    )
    return fx.rocdl.make_buffer_tensor(
        fx.make_view(
            fx.get_iter(K_) + koff,
            layout,
        ),
        max_size=False,
    )


def build(
    M,
    N,
    D,
    BM,
    BN,
    H=1,
    qkv_dtype="bf16",
    mma_mn=16,
):
    """构造旋转反相 MHA kernel(flash softmax + multi-head)。

    stage0执行K global预取与softmax；stage1执行K LDS写读、GEMM2和下一次V预取/GEMM1。
    H: head 数(multi-head,grid.y=head)。
    BF16使用阈值8的lazy rebase；FP8为保证probability量化有限而使用精确rebase。
    """
    assert qkv_dtype in ("bf16", "fp8")
    assert mma_mn in (16, 32), "MMA_MN must be 16 or 32"
    assert mma_mn != 32 or BM == 128, "MMA_MN=32 only supports BM=128"
    is_fp8 = qkv_dtype == "fp8"
    element_type = fx.Float8E4M3FNUZ if is_fp8 else fx.BFloat16
    mma_k = (32 if is_fp8 else 16) * 16 // mma_mn
    assert BN == 32
    assert BM in (64, 128)
    assert BN % 16 == 0 and D % mma_k == 0 and N % BN == 0 and M % BM == 0
    assert (N // BN) % 2 == 0, "KV tile 数需为偶数(循环展开 2 次)"
    WAVES = 4
    QUERY_REPEATS = BM // (WAVES * mma_mn)
    M_REPEATS = BN // mma_mn
    K_ATOMS_PER_GROUP = 2 if mma_mn == 32 else 1
    REDUCE_SHUFFLES = (16, 32) if mma_mn == 16 else (32,)
    USE_HW_SLOT_PRIORITY = mma_mn == 32 and not is_fp8 and D == 128
    NT = WAVES * 64  # 线程数
    VECN = 128 // element_type.width
    assert QUERY_REPEATS in (1, 2) and D % VECN == 0
    bits_per_thread = BN * D * element_type.width // NT
    assert bits_per_thread % 64 == 0, "K tile必须能由256线程按至少64-bit均分"
    coop_copy_bits = 128 if bits_per_thread % 128 == 0 else 64
    COOP_VEC = coop_copy_bits // element_type.width
    MMA32_V_LOADS = D // 16
    MMA32_GEMM_MFMAS = D // mma_k
    MMA32_DSRD_LEAD_MFMAS = 0 if is_fp8 else 3
    K_COPY_ATOMS = bits_per_thread // coop_copy_bits
    INTERLEAVE_EXP_DSWR = mma_mn == 32 and not is_fp8 and D == 128
    sm_scale = float(1.0 / (D**0.5))  # softmax 缩放 1/sqrt(D)
    sm_scale_log2 = float(
        sm_scale * LOG2E
    )  # 把 LOG2E 折进缩放:exp2(S*sm_scale*LOG2E-m) 省掉逐元素 *LOG2E
    rebase_threshold = 0.0 if is_fp8 else 8.0

    @flyc.kernel
    def attn_kernel(Q_: fx.Tensor, K_: fx.Tensor, V_: fx.Tensor, O_: fx.Tensor):
        tid = fx.thread_idx.x
        bm = fx.block_idx.x  # 第几个 query tile
        h = fx.block_idx.y  # head 索引(multi-head)
        hw_wave_slot = _read_hw_wave_slot() if USE_HW_SLOT_PRIORITY else None

        def set_stage0_priority():
            if const_expr(USE_HW_SLOT_PRIORITY):
                _set_hw_slot_priority(hw_wave_slot, 1, 0)
            else:
                rocdl.s_setprio(0)

        def set_stage1_priority():
            if const_expr(USE_HW_SLOT_PRIORITY):
                _set_hw_slot_priority(hw_wave_slot, 3, 2)
            else:
                rocdl.s_setprio(2)

        qo_off = h * (M * D)  # Q/O 的 head 偏移(元素)
        kv_off = h * (N * D)  # K/V 的 head 偏移(元素)

        # 多头:每 head 的 Q/K/V/O 在全局按 head 偏移基址(iter+offset)
        Q = fx.Tensor(
            fx.make_view(fx.get_iter(Q_) + qo_off, fx.make_layout((M, D), (D, 1)))
        )
        VECV = 8
        NB = BN // VECV
        # O 存成转置视图 O^T[D,M]:GEMM2 转置后 C=O^T,4/lane 沿 D 连续 -> 64-bit 写
        output_view = fx.Tensor(
            fx.make_view(fx.get_iter(O_) + qo_off, fx.make_layout((D, M), (1, D)))
        )
        Qb = fx.rocdl.make_buffer_tensor(Q, max_size=False)
        Ob = fx.rocdl.make_buffer_tensor(output_view, max_size=False)

        q_tile = fx.flat_divide(Qb, fx.make_tile(BM, D))[None, None, bm, 0]  # [BM, D]
        k_tiles = _make_ktiles(K_, N, D, BN, kv_off, is_fp8, mma_mn)  # [BN,D,N//BN,1]
        o_tile = fx.flat_divide(Ob, fx.make_tile(D, BM))[
            None, None, 0, bm
        ]  # [D, BM] = O^T tile

        mma = fx.make_mma_atom(fx.rocdl.MFMA(mma_mn, mma_mn, mma_k, element_type))
        k_perm1 = (
            (
                fx.make_layout((8, 2, 2), (1, 16, 8))
                if is_fp8
                else fx.make_layout((4, 2, 2), (1, 8, 4))
            )
            if mma_mn == 32
            else (
                fx.make_layout((8, 4, 2), (1, 16, 8))
                if is_fp8
                else fx.make_layout((4, 4, 2), (1, 8, 4))
            )
        )
        k_perm2 = (
            k_perm1
            if mma_mn == 32
            else (fx.make_layout((8, 4), (1, 8)) if is_fp8 else k_perm1)
        )
        # BF16的row permutation在全局K view中补偿；FP8使用plain row + 连续K32 operand。
        # GEMM1 = K@Q^T: wave 沿 query-M(MFMA 的 N 维)-> (1,WAVES,1);K 维(D)加 k_perm -> K 128-bit
        tmma1 = fx.make_tiled_mma(
            mma,
            fx.make_layout((1, WAVES, 1), (1, 1, 0)),
            fx.make_tile(None, None, k_perm1),
        )
        # GEMM2 = (S@V)^T = V^T@S^T:交换 A/B -> C=O^T[D,Mq];wave 沿 query-M(现为 N 维)-> (1,WAVES,1);
        # K 维(Nk)加 k_perm -> A(V^T)每 lane 8 Nk -> 128-bit;C 累加器 4/lane 沿 D 连续 -> O 64-bit 写出
        tmma2 = fx.make_tiled_mma(
            mma,
            fx.make_layout((1, WAVES, 1), (1, 1, 0)),
            fx.make_tile(None, None, k_perm2),
        )
        thr1 = tmma1.thr_slice(tid)
        thr2 = tmma2.thr_slice(tid)

        cp_cg = fx.make_copy_atom(
            fx.rocdl.BufferCopy(coop_copy_bits), element_type
        )  # 协作 global -> reg(合并)
        cp_cs = fx.make_copy_atom(
            fx.UniversalCopy(coop_copy_bits), element_type
        )  # reg -> LDS
        cp_kr = fx.make_copy_atom(
            fx.UniversalCopy128b(), element_type
        )  # k_lds -> frag_K(k_perm -> 128-bit)
        q_copy_bits = 128
        v_copy_bits = 64 if is_fp8 else 128
        cp_vg = fx.make_copy_atom(
            fx.rocdl.BufferCopy(v_copy_bits),
            element_type,
        )  # V paged global -> frag_V(直读,不经 LDS)
        cp_qg = fx.make_copy_atom(
            fx.rocdl.BufferCopy(q_copy_bits), element_type
        )  # Q global -> frag_Q(k_perm -> 128-bit)
        cp_oc = fx.make_copy_atom(
            fx.rocdl.BufferCopy64b(), fx.BFloat16
        )  # O 输出(GEMM2 转置 -> C=O^T,4/lane 沿 D 连续 -> 64-bit 写出)

        # LDS: K 双缓冲(ping-pong),2*[BN,D];S 不入 LDS(register trick)
        @fx.struct
        class SharedStorage:
            k_lds: fx.Array[element_type, 2 * BN * D, 16]

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        # BF16 swizzle去bank conflict；FP8不能复用该字节宽度相关映射。
        k_lds2 = _make_klds_view(lds.k_lds.ptr, BN, D, is_fp8)
        # V直读paged global：每tile [D,(8,NB)]，BF16/FP8分别使用128/64-bit load。
        v_g = fx.rocdl.make_buffer_tensor(
            fx.make_view(
                fx.get_iter(V_) + kv_off,
                fx.make_layout(
                    (D, (VECV, NB), N // BN),
                    (VECV, (1, D * VECV), BN * D),
                ),
            ),
            max_size=False,
        )
        # frag_V 的 B 模板:干净 [D,BN] tile(非 paged 嵌套),读取源才用 paged 分区
        v_fake = fx.flat_divide(
            fx.rocdl.make_buffer_tensor(
                fx.make_view(
                    fx.get_iter(V_) + kv_off, fx.make_layout((D, BN), (BN, 1))
                ),
                max_size=False,
            ),
            fx.make_tile(D, BN),
        )[None, None, 0, 0]

        # 固定256线程覆盖完整[BN,D]；FP8 D192使用64-bit atom均分。
        coop_col_threads = 16 if D % (16 * COOP_VEC) == 0 else 8
        coop_row_threads = NT // coop_col_threads
        assert coop_row_threads <= BN and BN % coop_row_threads == 0
        assert D % coop_col_threads == 0
        coop_thr = fx.make_layout(
            (coop_row_threads, coop_col_threads), (coop_col_threads, 1)
        )
        coop_val = fx.make_layout(
            (BN // coop_row_threads, D // coop_col_threads),
            (D // coop_col_threads, 1),
        )
        coop_g = fx.make_tiled_copy_tv(cp_cg, coop_thr, coop_val).get_slice(tid)
        coop_s = fx.make_tiled_copy_tv(cp_cs, coop_thr, coop_val).get_slice(tid)

        # Q(GEMM1 的 B 操作数)只载入一次；BM64/128时每wave分别处理16/32行。
        frag_Q = thr1.make_fragment_B(q_tile)
        tcQ = fx.make_tiled_copy_B(cp_qg, tmma1).get_slice(tid)
        fx.copy(cp_qg, tcQ.partition_S(q_tile), tcQ.retile(frag_Q))

        # fragment(形状固定)
        frag_O = thr2.make_fragment_C(o_tile)  # C=O[M,D](跨 KV 循环累加)
        tcK = fx.make_tiled_copy_A(cp_kr, tmma1).get_slice(tid)  # k_lds -> frag_K
        tcV = fx.make_tiled_copy_A(cp_vg, tmma2).get_slice(
            tid
        )  # paged global -> frag_V(直读)

        frag_O.fill(0)

        # 双缓冲 prologue:coop K(0)->frag; 写 k_lds[0]; coop K(1)->frag; barrier
        frag_ldK = fx.make_fragment_like(coop_g.partition_S(k_tiles[None, None, 0, 0]))
        fx.copy(cp_cg, coop_g.partition_S(k_tiles[None, None, 0, 0]), frag_ldK)
        fx.copy(
            cp_cs, frag_ldK, coop_s.partition_D(k_lds2[0, None, None])
        )  # 写 stage 0
        fx.copy(
            cp_cg, coop_g.partition_S(k_tiles[None, None, 1, 0]), frag_ldK
        )  # 预取 K(1) coop
        gpu.barrier()

        acc_init = frag_O.load()
        kcar_init = frag_ldK.load()  # frag_ldK 持 K(kv+1) coop 数据(loop-carried)

        # 循环内 fragment 提到循环外:形状固定,只分配一次,循环内复用(fill/store/copy 仍在循环内)
        frag_K = thr1.make_fragment_A(k_lds2[0, None, None])  # k_lds -> frag_K
        frag_St = thr1.make_fragment_C(
            fx.make_rmem_tensor(fx.make_layout((BN, BM), (BM, 1)), fx.Float32)
        )  # GEMM1 C=S^T
        frag_Sb = thr2.make_fragment_B(
            fx.make_rmem_tensor(fx.make_layout((BM, BN), (BN, 1)), element_type)
        )  # GEMM2 B=S^T
        frag_ldK_next = fx.make_fragment_like(
            coop_g.partition_S(k_tiles[None, None, 0, 0])
        )  # coop 预取 K(kv+2)
        frag_V = thr2.make_fragment_A(v_fake)  # V 直读 -> frag_V

        # K LDS读数量随D计算；VMEM/MFMA配额沿用当前旋转反相调度。
        WARP = NT // WAVES  # 64
        n_dsrd = BN * D * element_type.width // (WARP * 128)

        def hot_loop_scheduler(is_first_gemm, stagger_v_loads=False):
            if const_expr(mma_mn == 32 and D == 128):
                if is_first_gemm:
                    # VMEM:MFMA固定1:1，剩余MFMA尾置以维持resident-wave反相。
                    for _ in range_constexpr(MMA32_V_LOADS):
                        rocdl.sched_vmem(1)
                        rocdl.sched_mfma(1)
                    rocdl.sched_mfma(MMA32_GEMM_MFMAS - MMA32_V_LOADS)
                else:
                    # BF16先用3条MFMA覆盖K LDS写；FP8直接进入严格1:1 DSRD shadow。
                    for _ in range_constexpr(K_COPY_ATOMS):
                        rocdl.sched_vmem(1)
                        rocdl.sched_dswr(1)
                    if const_expr(MMA32_DSRD_LEAD_MFMAS > 0):
                        rocdl.sched_mfma(MMA32_DSRD_LEAD_MFMAS)
                    for _ in range_constexpr(n_dsrd):
                        rocdl.sched_dsrd(1)
                        rocdl.sched_mfma(1)
                    rocdl.sched_mfma(MMA32_GEMM_MFMAS - n_dsrd - MMA32_DSRD_LEAD_MFMAS)
            elif is_first_gemm:
                # 第二静态 substep 在第 5/6 条 V load 处有 VMEM 队列背压，保持总 MFMA 配额为 24。
                if stagger_v_loads:
                    for _ in range_constexpr(3):
                        rocdl.sched_vmem(1)
                        rocdl.sched_mfma(3)
                    for _ in range_constexpr(2):
                        rocdl.sched_vmem(1)
                        rocdl.sched_mfma(4)
                    rocdl.sched_vmem(1)
                    rocdl.sched_mfma(3)
                    for _ in range_constexpr(2):
                        rocdl.sched_vmem(1)
                        rocdl.sched_mfma(2)
                else:
                    for _ in range_constexpr(8):
                        rocdl.sched_vmem(1)
                        rocdl.sched_mfma(3)

                # rocdl.sched_vmem(100)
                # rocdl.sched_mfma(100)
            else:
                rocdl.sched_vmem(1)
                rocdl.sched_dswr(1)
                rocdl.sched_mfma(7)
                rocdl.sched_vmem(1)
                rocdl.sched_mfma(3)
                rocdl.sched_dswr(1)
                rocdl.sched_mfma(4)
                for _ in range_constexpr(n_dsrd):
                    rocdl.sched_dsrd(1)
                    rocdl.sched_mfma(1)
            rocdl.sched_barrier(0)

        def gemm1_mt(mt):
            # Fixing n_rep=mt gives one independent mma_mn-row query accumulator group.
            for kg in range_constexpr(D // (K_ATOMS_PER_GROUP * mma_k)):
                for ki in range_constexpr(K_ATOMS_PER_GROUP):
                    for m in range_constexpr(M_REPEATS):
                        acc = frag_St[None, m, mt]
                        if const_expr(mma_mn == 32):
                            fx.mma_atom_call(
                                mma,
                                acc,
                                frag_K[None, m, (ki, kg)],
                                frag_Q[None, mt, (ki, kg)],
                                acc,
                            )
                        else:
                            fx.mma_atom_call(
                                mma,
                                acc,
                                frag_K[None, m, kg],
                                frag_Q[None, mt, kg],
                                acc,
                            )

        # 展开 2 次:偶/奇两步 LDS stage(wr)变编译期常量,消掉 kv%2;fragment 全部复用
        # K 读做成 prefetch:frag_K 在上一步 GEMM2 之后就读好,GEMM1 直接用(藏 LDS 读延迟)
        def kv_step(kv_i, wr, ld_cur, ld_next, m0, m1, l0, l1, stagger_v_loads):
            # V 直读 paged global -> frag_V
            fx.copy(cp_vg, tcV.partition_S(v_g[None, None, kv_i]), tcV.retile(frag_V))
            # Split GEMM1 by its independent 16-row query accumulator groups.
            frag_St.fill(0)
            for mt in range_constexpr(QUERY_REPEATS):
                gemm1_mt(mt)
            hot_loop_scheduler(True, stagger_v_loads)
            # stage0:K global预取/LDS写与softmax。
            rocdl.sched_barrier(0)
            set_stage0_priority()
            rocdl.sched_barrier(0)
            fx.copy(
                cp_cg, coop_g.partition_S(k_tiles[None, None, kv_i + 2, 0]), ld_next
            )
            m_in, l_in = [m0, m1], [l0, l1]
            m_out, l_out, corr = [m0, m1], [l0, l1], [None, None]
            for mt in range_constexpr(QUERY_REPEATS):
                score = frag_St[None, None, mt].load()
                tmax = score.reduce("max")
                for sh in REDUCE_SHUFFLES:
                    tmax = _maxnumf(tmax, tmax.shuffle_xor(sh, 64))
                tmax = _fma_f32_inline(tmax, fx.Float32(sm_scale_log2), fx.Float32(0.0))
                nm = m_in[mt]
                corr_mt = fx.Float32(1.0)
                if tmax > m_in[mt] + fx.Float32(rebase_threshold):
                    nm = tmax
                    corr_mt = _exp2_amdgcn(m_in[mt] - nm)
                corr[mt] = corr_mt
                centered = _scale_center_vec_f32_inline(
                    score, fx.Float32(sm_scale_log2), nm
                )
                probability = _exp2_vec_amdgcn(centered)
                l_out[mt] = _fma_vec_f32_inline(l_in[mt], corr[mt], probability)
                m_out[mt] = nm
                frag_St[None, None, mt].store(probability)
            fx.copy(cp_cs, ld_cur, coop_s.partition_D(k_lds2[wr, None, None]))
            if const_expr(INTERLEAVE_EXP_DSWR):
                # Keep one complete copy, but place its two generated writes
                # separately after probability EXP 2 and 6. The first TRANS
                # group also includes the preceding running-max correction EXP.
                _sched_trans(3)
                _sched_ds_write(1)
                _sched_trans(4)
                _sched_ds_write(1)
                _sched_trans(10)
            for mt in range_constexpr(
                QUERY_REPEATS
            ):  # 旧 O 按 correction 缩放(GEMM2 累加前)
                output_tile = frag_O[None, None, mt]

                def rescale_output():
                    output_tile.store(output_tile.load() * corr[mt])

                @flyc.jit
                def rescale_if_needed():
                    if corr[mt] < fx.Float32(1.0):
                        rescale_output()

                rescale_if_needed()
            m0, m1, l0, l1 = m_out[0], m_out[1], l_out[0], l_out[1]
            # P^T直接作GEMM2的B。FP8的K=32 atom把两个16行half合并为8元素/lane。
            if const_expr(is_fp8):
                if const_expr(mma_mn == 32):
                    probability = frag_St.load()
                    for k in range_constexpr(2):
                        base = k * 8
                        prob_lo = _cvt_f32x4_to_fp8(
                            fx.Vector.from_elements(
                                [probability[base + i] for i in range_constexpr(4)],
                                fx.Float32,
                            )
                        )
                        prob_hi = _cvt_f32x4_to_fp8(
                            fx.Vector.from_elements(
                                [probability[base + 4 + i] for i in range_constexpr(4)],
                                fx.Float32,
                            )
                        )
                        frag_Sb[None, 0, k].store(
                            prob_lo.shuffle(prob_hi, list(range(8)))
                        )
                else:
                    for mt in range_constexpr(QUERY_REPEATS):
                        prob_lo = _cvt_f32x4_to_fp8(frag_St[None, 0, mt].load())
                        prob_hi = _cvt_f32x4_to_fp8(frag_St[None, 1, mt].load())
                        frag_Sb[None, mt, 0].store(
                            prob_lo.shuffle(prob_hi, list(range(8)))
                        )
            elif const_expr(mma_mn == 32):
                frag_Stb = _cvt_f32_to_bf16(frag_St)
                frag_Sb.store(frag_Stb.load())
            else:
                for mn in range_constexpr(BN // 16):
                    frag_Stb = _cvt_f32_to_bf16(frag_St[None, mn, None])
                    frag_Sb[None, None, mn].store(frag_Stb.load())
            # stage1:GEMM2与下一tile的V预取/GEMM1，跨runtime loop回边。
            rocdl.sched_barrier(0)
            set_stage1_priority()
            rocdl.sched_barrier(0)
            fx.gemm(mma, frag_O, frag_V, frag_Sb, frag_O)  # O^T = V^T @ P^T(交换 A/B)
            gpu.barrier()  # k_lds[wr] 写完可见
            # prefetch 下一步 K:读 k_lds[wr] -> frag_K(移到 GEMM2 之后,藏 LDS 读延迟)
            fx.copy(cp_kr, tcK.partition_S(k_lds2[wr, None, None]), tcK.retile(frag_K))
            hot_loop_scheduler(False)
            return m0, m1, l0, l1

        # prologue:prefetch 第一个 frag_K = k_lds[0]
        fx.copy(cp_kr, tcK.partition_S(k_lds2[0, None, None]), tcK.retile(frag_K))
        fragK_init = frag_K.load()
        l_zero = fx.Vector.zeros_like(frag_St[None, None, 0].load())
        stage_loop_init = [
            acc_init,
            kcar_init,
            fragK_init,
            fx.Float32(float("-inf")),
            fx.Float32(float("-inf")),
            l_zero,
            l_zero,
        ]
        rocdl.sched_barrier(0)
        set_stage1_priority()
        rocdl.sched_barrier(0)
        for kv, state in range(
            fx.Index(0), fx.Index(N // BN // 2), fx.Index(1), init=stage_loop_init
        ):
            frag_O.store(state[0])
            frag_ldK.store(state[1])
            frag_K.store(state[2])
            m0, m1, l0, l1 = state[3], state[4], state[5], state[6]
            kv0 = fx.Int32(kv) * 2
            m0, m1, l0, l1 = kv_step(
                kv0, 1, frag_ldK, frag_ldK_next, m0, m1, l0, l1, False
            )
            m0, m1, l0, l1 = kv_step(
                kv0 + 1, 0, frag_ldK_next, frag_ldK, m0, m1, l0, l1, True
            )

            stage_results = yield [
                frag_O.load(),
                frag_ldK.load(),
                frag_K.load(),
                m0,
                m1,
                l0,
                l1,
            ]

        frag_O.store(stage_results[0])
        rocdl.sched_barrier(0)
        rocdl.s_setprio(0)
        rocdl.sched_barrier(0)
        l_final = [stage_results[5], stage_results[6]]

        # ---- epilogue:O /= l(每 Mq-tile 用最终 running sum 归一化)----
        for mt in range_constexpr(QUERY_REPEATS):
            l_final[mt] = l_final[mt].reduce("add")
            for sh in REDUCE_SHUFFLES:
                l_final[mt] = l_final[mt] + l_final[mt].shuffle_xor(sh, 64)
            output_tile = frag_O[None, None, mt]
            prob_scale = FP8_PROB_SCALE if is_fp8 else 1.0
            output_tile.store(
                output_tile.load() * (fx.Float32(1.0 / prob_scale) / l_final[mt])
            )

        # O: f32 -> bf16 -> global(每wave写16/32行；_cvt_f32_to_bf16省RNE+NaN指令)
        frag_Ob = _cvt_f32_to_bf16(frag_O)
        tcO = fx.make_tiled_copy_C(cp_oc, tmma2).get_slice(tid)
        fx.copy(cp_oc, tcO.retile(frag_Ob), tcO.partition_S(o_tile))

    @flyc.jit
    def launch(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        output: fx.Tensor,
        stream: fx.Stream,
    ):
        attn_kernel(Q, K, V, output).launch(
            grid=(M // BM, H, 1), block=(NT, 1, 1), stream=stream
        )

    return launch


def torch_ref(Q, K, V, qkv_dtype):
    """Non-causal multi-head attention reference."""
    S = torch.einsum("hmd,hnd->hmn", Q.float(), K.float())  # [H,M,N] f32
    S = S * (1.0 / (Q.shape[-1] ** 0.5))
    if qkv_dtype == "fp8":
        running_max = torch.full_like(S[:, :, :1], float("-inf"))
        running_sum = torch.zeros_like(running_max)
        output = torch.zeros(
            Q.shape[0], Q.shape[1], V.shape[2], dtype=torch.float32, device=Q.device
        )
        for start in range(0, S.shape[-1], 32):
            score = S[:, :, start : start + 32]
            tile_max = score.amax(dim=-1, keepdim=True)
            new_max = torch.maximum(running_max, tile_max)
            correction = torch.exp(running_max - new_max)
            probability = torch.exp(score - new_max)
            probability_fp8 = (
                (probability * FP8_PROB_SCALE).to(torch.float8_e4m3fnuz).float()
            )
            output = output * correction + torch.einsum(
                "hmn,hnd->hmd", probability_fp8, V[:, start : start + 32].float()
            )
            running_sum = running_sum * correction + probability.sum(
                dim=-1, keepdim=True
            )
            running_max = new_max
        output = output / (running_sum * FP8_PROB_SCALE)
    else:
        P = torch.softmax(S, dim=-1).to(torch.bfloat16).float()
        output = torch.einsum("hmn,hnd->hmd", P, V.float())
    return output.to(torch.bfloat16)


def main():
    torch.manual_seed(0)
    torch.set_default_device("cuda")

    # 旋转反相MHA；H/BM/MULT/D控制问题尺寸。
    H = int(os.environ.get("H", "8"))
    BM = int(os.environ.get("BM", "128"))
    BN = 32
    mult = int(os.environ.get("MULT", "16"))
    D = int(os.environ.get("D", "192"))
    mma_mn = int(os.environ.get("MMA_MN", "16"))
    qkv_dtype = os.environ.get("QKV_DTYPE", "bf16").lower()
    check = os.environ.get("CHECK", "1") != "0"
    assert qkv_dtype in ("bf16", "fp8")
    torch_dtype = torch.float8_e4m3fnuz if qkv_dtype == "fp8" else torch.bfloat16
    M, N = BM * mult, BM * mult

    Q = torch.randn(H, M, D, dtype=torch.float32).to(torch_dtype)
    K = torch.randn(H, N, D, dtype=torch.float32).to(torch_dtype)
    V = torch.randn(H, N, D, dtype=torch.float32).to(torch_dtype)
    # 预 shuffle V 成 paged 布局 [H, N//8, D, 8];torch_ref 仍用原始 V
    V_shuf = _preshuffle_v(V, qkv_dtype, mma_mn)

    stream = torch.cuda.current_stream()
    o_fly = torch.empty(H, M, D, dtype=torch.bfloat16)
    args = (Q, K, V_shuf, o_fly, stream)
    print(
        f"[cfg] H={H} M={M} N={N} D={D} BM={BM} MMA={mma_mn} "
        f"QKV={qkv_dtype} pipeline=stage_antiphase priority=2"
    )
    kernel = fly_compiled(
        (M, N, D, BM, BN, H, qkv_dtype, mma_mn),
        lambda: build(
            M,
            N,
            D,
            BM,
            BN,
            H=H,
            qkv_dtype=qkv_dtype,
            mma_mn=mma_mn,
        ),
        args,
    )
    torch.cuda.synchronize()

    # ---- 精度 ----
    if check:
        o_ref = torch_ref(Q, K, V, qkv_dtype)
        diff = (o_fly.float() - o_ref.float()).abs()
        rel = diff.norm() / o_ref.float().norm().clamp_min(1e-6)
        print(
            f"[acc] max_abs={diff.max().item():.4f} mean_abs={diff.mean().item():.5f} rel_l2={rel.item():.5f}"
        )
    else:
        print("[acc] skipped (CHECK=0)")

    # ---- 性能:多 buffer 轮换 + cudaPerf 计时 ----
    from pyhip import cudaPerf

    flops = H * 4 * M * N * D  # 每 head gemm1+gemm2 各 2*M*N*D
    mem_bytes = sum(t.numel() * t.element_size() for t in (Q, K, V_shuf, o_fly))

    BUF_COPY = 10
    Qs = [
        torch.randn(H, M, D, dtype=torch.float32).to(torch_dtype)
        for _ in range(BUF_COPY)
    ]
    Ks = [
        torch.randn(H, N, D, dtype=torch.float32).to(torch_dtype)
        for _ in range(BUF_COPY)
    ]
    Vs = [
        torch.randn(H, N, D, dtype=torch.float32).to(torch_dtype)
        for _ in range(BUF_COPY)
    ]
    V_shufs = [_preshuffle_v(v, qkv_dtype, mma_mn) for v in Vs]
    o_flys = [torch.empty(H, M, D, dtype=torch.bfloat16) for _ in range(BUF_COPY)]

    run_count = 50

    def perf(fn, name):
        for _ in range(10):  # warmup
            fn(0)
        torch.cuda.synchronize()
        tfs, uss = [], []
        i = 0
        for _ in range(run_count):
            with cudaPerf(flops, mem_bytes, name=name, verbose=0) as p:
                fn(i)
            i = (i + 1) % BUF_COPY
            tfs.append(p.tflops())
            uss.append(p.dt() * 1e6)
        tfs.sort()
        uss.sort()
        return uss[run_count // 2], tfs[run_count // 2]

    def launch(index):
        kernel(Qs[index], Ks[index], V_shufs[index], o_flys[index], stream)

    microseconds, tflops = perf(launch, "attn_stage_antiphase")
    print(
        f"[perf] fly: {microseconds:8.1f} us  {tflops:7.1f} TFLOPS  "
        f"({mem_bytes / microseconds / 1e3:.0f} GB/s)"
    )


if __name__ == "__main__":
    main()
