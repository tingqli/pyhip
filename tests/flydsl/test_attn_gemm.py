"""222T旋转反相 fused-attention kernel 的精度与性能测试。

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
from flydsl.expr import arith, gpu, range_constexpr, rocdl
from flydsl._mlir.dialects import llvm

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


def _make_klds_view(ptr, BN, D):
    """K LDS 视图 [2,BN,D](D 连续,stage 偏移 BN*D):SwizzleType(3,3,3) 组合行主序去 bank 冲突
    -> K 读 ds_read2st64_b64。"""
    base = fx.make_layout((2, BN, D), (BN * D, D, 1))
    swz = fx.SwizzleType.get(3, 3, 3)
    return fx.make_view(ptr, fx.make_composed_layout(fx.static(swz), base))


def _make_ktiles(K_, N, D, BN, koff):
    """K coop 读源 tile [BN,D,N//BN,1](head 偏移 koff=h*N*D 元素)。perm_M 施加在全局 K cache 而非 MMA:
    每 tile 内 Nk 按正向 k_perm (4,4,2):(D,8D,4D) 重排 -> plain 读入 LDS 后,GEMM1 不加 perm_M 也能
    得到相同的 8-连续-Nk C 布局(N 方向已 shuffle)。假设 BN=32。"""
    return fx.rocdl.make_buffer_tensor(
        fx.make_view(
            fx.get_iter(K_) + koff,
            fx.make_layout(
                ((4, 4, 2), D, N // BN, 1), ((D, 8 * D, 4 * D), 1, BN * D, 0)
            ),
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
):
    """构造222T旋转反相 MHA kernel(flash softmax + multi-head)。

    stage0执行K global预取与softmax；stage1执行K LDS写读、GEMM2和下一次V预取/GEMM1。
    H: head 数(multi-head,grid.y=head)。
    固定使用lazy rebase：局部最大值超过参考值8（log2域）时才重缩放l/O。
    """
    assert BN == 32
    assert BM % 32 == 0 and BN % 16 == 0 and D % 16 == 0 and N % BN == 0 and M % BM == 0
    assert (N // BN) % 2 == 0, "KV tile 数需为偶数(循环展开 2 次)"
    WAVES = BM // 32  # 每 wave 负责 32 行 query
    NT = WAVES * 64  # 线程数
    VECN = 128 // fx.BFloat16.width  # 协作加载向量宽度(8 bf16 = 128b)
    assert BM // 32 == WAVES and D % VECN == 0 and BN % 16 == 0
    sm_scale = float(1.0 / (D**0.5))  # softmax 缩放 1/sqrt(D)
    sm_scale_log2 = float(
        sm_scale * LOG2E
    )  # 把 LOG2E 折进缩放:exp2(S*sm_scale*LOG2E-m) 省掉逐元素 *LOG2E

    @flyc.kernel
    def attn_kernel(Q_: fx.Tensor, K_: fx.Tensor, V_: fx.Tensor, O_: fx.Tensor):
        tid = fx.thread_idx.x
        bm = fx.block_idx.x  # 第几个 query tile
        h = fx.block_idx.y  # head 索引(multi-head)
        qo_off = h * (M * D)  # Q/O 的 head 偏移(元素)
        kv_off = h * (N * D)  # K/V 的 head 偏移(元素)

        # 多头:每 head 的 Q/K/V/O 在全局按 head 偏移基址(iter+offset)
        Q = fx.Tensor(
            fx.make_view(fx.get_iter(Q_) + qo_off, fx.make_layout((M, D), (D, 1)))
        )
        NB = BN // 8
        # O 存成转置视图 O^T[D,M]:GEMM2 转置后 C=O^T,4/lane 沿 D 连续 -> 64-bit 写
        output_view = fx.Tensor(
            fx.make_view(fx.get_iter(O_) + qo_off, fx.make_layout((D, M), (1, D)))
        )
        Qb = fx.rocdl.make_buffer_tensor(Q, max_size=False)
        Ob = fx.rocdl.make_buffer_tensor(output_view, max_size=False)

        q_tile = fx.flat_divide(Qb, fx.make_tile(BM, D))[None, None, bm, 0]  # [BM, D]
        k_tiles = _make_ktiles(
            K_, N, D, BN, kv_off
        )  # [BN, D, N//BN, 1](perm_M 已全局重排)
        o_tile = fx.flat_divide(Ob, fx.make_tile(D, BM))[
            None, None, 0, bm
        ]  # [D, BM] = O^T tile

        mma = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, fx.BFloat16))
        k_perm = fx.make_layout((4, 4, 2), (1, 8, 4))
        # perm_M 挪到全局 K(_make_ktiles) -> MMA 不加 perm_M(N 方向已 shuffle)
        # GEMM1 = K@Q^T: wave 沿 query-M(MFMA 的 N 维)-> (1,WAVES,1);K 维(D)加 k_perm -> K 128-bit
        tmma1 = fx.make_tiled_mma(
            mma,
            fx.make_layout((1, WAVES, 1), (1, 1, 0)),
            fx.make_tile(None, None, k_perm),
        )
        # GEMM2 = (S@V)^T = V^T@S^T:交换 A/B -> C=O^T[D,Mq];wave 沿 query-M(现为 N 维)-> (1,WAVES,1);
        # K 维(Nk)加 k_perm -> A(V^T)每 lane 8 Nk -> 128-bit;C 累加器 4/lane 沿 D 连续 -> O 64-bit 写出
        tmma2 = fx.make_tiled_mma(
            mma,
            fx.make_layout((1, WAVES, 1), (1, 1, 0)),
            fx.make_tile(None, None, k_perm),
        )
        thr1 = tmma1.thr_slice(tid)
        thr2 = tmma2.thr_slice(tid)

        cp_cg = fx.make_copy_atom(
            fx.rocdl.BufferCopy128b(), fx.BFloat16
        )  # 协作 global -> reg(合并)
        cp_cs = fx.make_copy_atom(fx.UniversalCopy128b(), fx.BFloat16)  # reg -> LDS
        cp_kr = fx.make_copy_atom(
            fx.UniversalCopy128b(), fx.BFloat16
        )  # k_lds -> frag_K(k_perm -> 128-bit)
        cp_vg = fx.make_copy_atom(
            fx.rocdl.BufferCopy128b(), fx.BFloat16
        )  # V paged global -> frag_V(直读,不经 LDS)
        cp_qg = fx.make_copy_atom(
            fx.rocdl.BufferCopy128b(), fx.BFloat16
        )  # Q global -> frag_Q(k_perm -> 128-bit)
        cp_oc = fx.make_copy_atom(
            fx.rocdl.BufferCopy64b(), fx.BFloat16
        )  # O 输出(GEMM2 转置 -> C=O^T,4/lane 沿 D 连续 -> 64-bit 写出)

        # LDS: K 双缓冲(ping-pong),2*[BN,D];S 不入 LDS(register trick)
        @fx.struct
        class SharedStorage:
            k_lds: fx.Array[fx.BFloat16, 2 * BN * D, 16]

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        # K LDS 视图:swizzle(3,3,3) 去 bank 冲突(swz 默认 / gpermswz 共用)
        k_lds2 = _make_klds_view(lds.k_lds.ptr, BN, D)
        # V 直读 paged global 读源(head 偏移 kv_off):每 kv-tile [D,(8,NB)], v(8) inner -> 128-bit buffer_load
        v_g = fx.rocdl.make_buffer_tensor(
            fx.make_view(
                fx.get_iter(V_) + kv_off,
                fx.make_layout((D, (8, NB), N // BN), (8, (1, D * 8), BN * D)),
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

        # 协作加载 [BN,D] tile(线程沿 D 连续 -> 合并访问)
        coop_thr = fx.make_layout((16, D // VECN), (D // VECN, 1))
        coop_val = fx.make_layout((BN // 16, VECN), (VECN, 1))
        coop_g = fx.make_tiled_copy_tv(cp_cg, coop_thr, coop_val).get_slice(tid)
        coop_s = fx.make_tiled_copy_tv(cp_cs, coop_thr, coop_val).get_slice(tid)

        # Q(GEMM1 的 B 操作数)只载入一次,每 wave 自己的 32 行
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
            fx.make_rmem_tensor(fx.make_layout((BM, BN), (BN, 1)), fx.BFloat16)
        )  # GEMM2 B=S^T
        frag_ldK_next = fx.make_fragment_like(
            coop_g.partition_S(k_tiles[None, None, 0, 0])
        )  # coop 预取 K(kv+2)
        frag_V = thr2.make_fragment_A(v_fake)  # V 直读 -> frag_V

        # hot_loop_scheduler 的指令数按实际 tile 尺寸算(BN=32,D=128 -> 8 dsrd / 2 dswr / 2 vmem / 32 mfma/GEMM)
        WARP = NT // WAVES  # 64
        n_dsrd = BN * D // (WARP * VECN)  # K LDS 读(frag_K,每 wave 读 [BN,D],128-bit)

        def hot_loop_scheduler(is_first_gemm, stagger_v_loads=False):
            if is_first_gemm:
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

                rocdl.sched_vmem(100)
                rocdl.sched_mfma(100)
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
            # GEMM1 fragments are [value, m_rep=2, n_rep=2, k_rep=8].
            # Fixing n_rep=mt gives one independent 16-row query accumulator group.
            for k in range_constexpr(D // 16):
                for m in range_constexpr(BN // 16):
                    acc = frag_St[None, m, mt]
                    fx.mma_atom_call(
                        mma,
                        acc,
                        frag_K[None, m, k],
                        frag_Q[None, mt, k],
                        acc,
                    )

        # 展开 2 次:偶/奇两步 LDS stage(wr)变编译期常量,消掉 kv%2;fragment 全部复用
        # K 读做成 prefetch:frag_K 在上一步 GEMM2 之后就读好,GEMM1 直接用(藏 LDS 读延迟)
        def kv_step(kv_i, wr, ld_cur, ld_next, m0, m1, l0, l1, stagger_v_loads):
            # V 直读 paged global -> frag_V
            fx.copy(cp_vg, tcV.partition_S(v_g[None, None, kv_i]), tcV.retile(frag_V))
            # Split GEMM1 by its two independent query-row accumulator groups.
            frag_St.fill(0)
            for mt in range_constexpr(2):
                gemm1_mt(mt)
            hot_loop_scheduler(True, stagger_v_loads)
            # stage1 到 V(i)+GEMM1(i) 结束；stage0 从 K(i+2) global 预取开始。
            rocdl.sched_barrier(0)
            rocdl.s_setprio(0)
            rocdl.sched_barrier(0)
            fx.copy(
                cp_cg, coop_g.partition_S(k_tiles[None, None, kv_i + 2, 0]), ld_next
            )
            m_in, l_in = [m0, m1], [l0, l1]
            m_out, l_out, corr = [None, None], [None, None], [None, None]
            for mt in range_constexpr(2):
                score = frag_St[None, None, mt].load()
                tmax = score.reduce("max")
                for sh in (16, 32):
                    tmax = _maxnumf(tmax, tmax.shuffle_xor(sh, 64))
                tmax = _fma_f32_inline(tmax, fx.Float32(sm_scale_log2), fx.Float32(0.0))
                nm = m_in[mt]
                corr_mt = fx.Float32(1.0)
                if tmax > m_in[mt] + fx.Float32(8.0):
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
            for mt in range_constexpr(2):  # 旧 O 按 correction 缩放(GEMM2 累加前)
                output_tile = frag_O[None, None, mt]

                def rescale_output():
                    output_tile.store(output_tile.load() * corr[mt])

                @flyc.jit
                def rescale_if_needed():
                    if corr[mt] < fx.Float32(1.0):
                        rescale_output()

                rescale_if_needed()
            m0, m1, l0, l1 = m_out[0], m_out[1], l_out[0], l_out[1]
            # P^T直接作GEMM2的B；拆半转换以分散VALU。
            frag_Stb0 = _cvt_f32_to_bf16(frag_St[None, 0, None])
            frag_Sb[None, None, 0].store(frag_Stb0.load())
            frag_Stb1 = _cvt_f32_to_bf16(frag_St[None, 1, None])
            frag_Sb[None, None, 1].store(frag_Stb1.load())
            # stage1 从 K(i+1) LDS 写开始，跨回边延续到下一次 V load + GEMM1。
            rocdl.sched_barrier(0)
            rocdl.s_setprio(2)
            rocdl.sched_barrier(0)
            fx.copy(cp_cs, ld_cur, coop_s.partition_D(k_lds2[wr, None, None]))
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
        rocdl.s_setprio(2)
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
        for mt in range_constexpr(2):
            l_final[mt] = l_final[mt].reduce("add")
            for sh in (16, 32):
                l_final[mt] = l_final[mt] + l_final[mt].shuffle_xor(sh, 64)
            output_tile = frag_O[None, None, mt]
            output_tile.store(output_tile.load() * (fx.Float32(1.0) / l_final[mt]))

        # O: f32 -> bf16 -> global(每 wave 写自己的 32 行;_cvt_f32_to_bf16 省 RNE+NaN 指令)
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


def torch_ref(Q, K, V):
    """Non-causal multi-head attention reference."""
    S = torch.einsum("hmd,hnd->hmn", Q.float(), K.float())  # [H,M,N] f32
    S = S * (1.0 / (Q.shape[-1] ** 0.5))
    P = torch.softmax(S, dim=-1)
    output = torch.einsum("hmn,hnd->hmd", P.to(torch.bfloat16).float(), V.float()).to(
        torch.bfloat16
    )
    return output


def main():
    torch.manual_seed(0)
    torch.set_default_device("cuda")

    # 222T旋转反相MHA；H/MULT仅控制问题尺寸。
    H = int(os.environ.get("H", "8"))
    BM, BN = 128, 32
    mult = int(os.environ.get("MULT", "16"))
    M, N, D = BM * mult, BM * mult, 128

    Q = torch.randn(H, M, D, dtype=torch.bfloat16)
    K = torch.randn(H, N, D, dtype=torch.bfloat16)
    V = torch.randn(H, N, D, dtype=torch.bfloat16)
    # 预 shuffle V 成 paged 布局 [H, N//8, D, 8];torch_ref 仍用原始 V
    V_shuf = V.reshape(H, N // 8, 8, D).permute(0, 1, 3, 2).contiguous()

    stream = torch.cuda.current_stream()
    o_fly = torch.empty(H, M, D, dtype=torch.bfloat16)
    args = (Q, K, V_shuf, o_fly, stream)
    print(f"[cfg] H={H} M={M} N={N} D={D} pipeline=stage_antiphase priority=2")
    kernel = fly_compiled(
        (M, N, D, BM, BN, H),
        lambda: build(M, N, D, BM, BN, H=H),
        args,
    )
    torch.cuda.synchronize()

    # ---- 精度 ----
    o_ref = torch_ref(Q, K, V)
    diff = (o_fly.float() - o_ref.float()).abs()
    rel = diff.norm() / o_ref.float().norm().clamp_min(1e-6)
    print(
        f"[acc] max_abs={diff.max().item():.4f} mean_abs={diff.mean().item():.5f} rel_l2={rel.item():.5f}"
    )

    # ---- 性能:多 buffer 轮换 + cudaPerf 计时 ----
    from pyhip import cudaPerf

    flops = H * 4 * M * N * D  # 每 head gemm1+gemm2 各 2*M*N*D
    mem_bytes = (Q.numel() + K.numel() + V_shuf.numel() + o_fly.numel()) * 2

    BUF_COPY = 10
    Qs = [torch.randn(H, M, D, dtype=torch.bfloat16) for _ in range(BUF_COPY)]
    Ks = [torch.randn(H, N, D, dtype=torch.bfloat16) for _ in range(BUF_COPY)]
    Vs = [torch.randn(H, N, D, dtype=torch.bfloat16) for _ in range(BUF_COPY)]
    V_shufs = [v.reshape(H, N // 8, 8, D).permute(0, 1, 3, 2).contiguous() for v in Vs]
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
