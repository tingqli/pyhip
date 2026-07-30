"""融合双 GEMM(无 softmax attention)精度 + 性能测试 —— register-resident S + 全 128-bit 读。

  gemm1: pv = Q[M,D] @ K[N,D]^T   -> [M, N]
  gemm2: v  = pv[M,N] @ V[N,D]     -> [M, D]

优化核心(见 docs/attn_gemm_optimization.md):
- register trick:GEMM1 算 S^T=K@Q^T,其 C 累加器经 fx.select 重排后直接作 GEMM2 的 A,S 不入 LDS。
- perm_M(GEMM1 的 M=Nk 维加 k_perm):C 累加器每 lane 持 8 连续 Nk,对齐 GEMM2 的 k_perm A。
- 全 128-bit 读:K/Q 沿 D 加 k_perm(ds_read/buffer_load 128-bit);V 预 shuffle 成 paged
  布局 [N//8,D,8],直接从 global paged 视图 buffer_load 到 frag_V(不经 LDS,128-bit)。
- f32→bf16 用 _cvt_f32_to_bf16(add-0x8000 舍入+截断),省 .to(bf16) 的 RNE+NaN 指令。
- 软件流水线:prologue 预取 K(0);K 走 LDS 双缓冲 ping-pong(读 k_lds[kv%2]=上轮写好,写 K(kv+1) 到
  k_lds[(kv+1)%2],write+barrier 与 GEMM2 重叠 -> 移出 GEMM1 关键路径);V 直读 global。
  BN=32(腾 VGPR 预算供双缓冲)-> M=N=20480: 229.8 TFLOPS(1.9x rocBLAS 122),VGPR 190。2 waves/SIMD。
精度 bf16。
"""

import os
from pathlib import Path

os.environ.setdefault("FLYDSL_RUNTIME_ENABLE_CACHE", "0")

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, const_expr, gpu, range_constexpr, rocdl
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


def _find_final_isa(dump_dir):
    candidates = sorted((Path(dump_dir) / "attn_kernel_0").glob("*_final_isa.s"))
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected one final attn_kernel_0 ISA dump in {dump_dir}, found {candidates}"
        )
    return candidates[0]


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


def _encode_waitcnt(vmcnt=63, expcnt=7, lgkmcnt=63):
    """Encode s_waitcnt bitfield for CDNA3 (gfx94x)."""
    vm_lo = vmcnt & 0xF
    vm_hi = (vmcnt >> 4) & 0x3
    return vm_lo | (expcnt << 4) | (lgkmcnt << 8) | (vm_hi << 14)


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


def build(M, N, D, BM, BN, H=1, softmax=True):
    """构造融合 MHA kernel(flash softmax + multi-head),返回 launch 包装。

    perm_M 挪到全局 K(_make_ktiles)+ 展开2x + K-prefetch:266.0 TFLOPS @M=N=40960(2.24x rocBLAS)。
    H: head 数(multi-head,grid.y=head)。
    softmax: 编译时开关。True=在线 flash softmax;False=纯双 GEMM(S@V,无 softmax/scale)-> ~260T 基线。
    softmax=True 时固定使用 lazy rebase:局部最大值超过参考值 8(log2 域)时才重缩放 l/O。
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
    use_long_sequence_schedule = N >= 32768

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
        O = fx.Tensor(
            fx.make_view(fx.get_iter(O_) + qo_off, fx.make_layout((D, M), (1, D)))
        )
        Qb = fx.rocdl.make_buffer_tensor(Q, max_size=False)
        Ob = fx.rocdl.make_buffer_tensor(O, max_size=False)

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

        def hot_loop_scheduler(is_first_gemm):
            if is_first_gemm:
                for _ in range_constexpr(8):
                    rocdl.sched_vmem(1)
                    rocdl.sched_mfma(3)

                rocdl.sched_vmem(100)
                rocdl.sched_mfma(100)
            else:
                if const_expr(use_long_sequence_schedule):
                    rocdl.sched_vmem(1)
                    rocdl.sched_dswr(1)
                    rocdl.sched_mfma(7)
                    rocdl.sched_vmem(1)
                    rocdl.sched_mfma(3)
                    rocdl.sched_dswr(1)
                    rocdl.sched_mfma(4)
                else:
                    for _ in range_constexpr(2):
                        rocdl.sched_vmem(1)
                        rocdl.sched_dswr(1)
                        rocdl.sched_mfma(7)
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
        def kv_step(kv_i, wr, ld_cur, ld_next, m0, m1, l0, l1):
            # V 直读 paged global -> frag_V
            fx.copy(cp_vg, tcV.partition_S(v_g[None, None, kv_i]), tcV.retile(frag_V))
            # Split GEMM1 by its two independent query-row accumulator groups.
            frag_St.fill(0)
            for mt in range_constexpr(2):
                gemm1_mt(mt)
            hot_loop_scheduler(True)
            fx.copy(
                cp_cg, coop_g.partition_S(k_tiles[None, None, kv_i + 2, 0]), ld_next
            )
            if const_expr(softmax):
                m_in, l_in = [m0, m1], [l0, l1]
                m_out, l_out, corr = [None, None], [None, None], [None, None]
                for mt in range_constexpr(2):
                    v = frag_St[None, None, mt].load() * sm_scale_log2
                    tmax = v.reduce("max")
                    for sh in (16, 32):
                        tmax = _maxnumf(tmax, tmax.shuffle_xor(sh, 64))
                    nm = m_in[mt]
                    corr_mt = fx.Float32(1.0)
                    if tmax > m_in[mt] + fx.Float32(8.0):
                        nm = tmax
                        corr_mt = _exp2_amdgcn(m_in[mt] - nm)
                    corr[mt] = corr_mt
                    p = _exp2_vec_amdgcn(v - nm)
                    ts = p.reduce("add")
                    for sh in (16, 32):
                        ts = ts + ts.shuffle_xor(sh, 64)
                    # 用一次舍入的两条 scalar FMA 替代 packed MUL + 两条 ADD；精度由回归测试约束。
                    l_out[mt] = fx.fma(l_in[mt], corr[mt], ts)
                    m_out[mt] = nm
                    frag_St[None, None, mt].store(p)
                for mt in range_constexpr(2):  # 旧 O 按 correction 缩放(GEMM2 累加前)
                    ot = frag_O[None, None, mt]

                    def rescale_output():
                        ot.store(ot.load() * corr[mt])

                    @flyc.jit
                    def rescale_if_needed():
                        if corr[mt] < fx.Float32(1.0):
                            rescale_output()

                    rescale_if_needed()
                m0, m1, l0, l1 = m_out[0], m_out[1], l_out[0], l_out[1]
            # S^T(启softmax时=P^T)直接作GEMM2的B；长序列拆半转换以分散VALU。
            if const_expr(use_long_sequence_schedule):
                frag_Stb0 = _cvt_f32_to_bf16(frag_St[None, 0, None])
                frag_Sb[None, None, 0].store(frag_Stb0.load())
                # 写K(kv+1)=ld_cur -> k_lds[wr]，由scheduler与GEMM2重叠。
                fx.copy(cp_cs, ld_cur, coop_s.partition_D(k_lds2[wr, None, None]))
                frag_Stb1 = _cvt_f32_to_bf16(frag_St[None, 1, None])
                frag_Sb[None, None, 1].store(frag_Stb1.load())
            else:
                frag_Stb = _cvt_f32_to_bf16(frag_St)
                frag_Sb.store(fx.select(frag_Stb, [0, 2, 1]).load())
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
        # flash softmax 在线统计初值(per Mq-tile,每 lane 冗余):m=-inf, l=0（仅 softmax 时携带）
        loop_init = [acc_init, kcar_init, fragK_init]
        if const_expr(softmax):
            loop_init += [
                fx.Float32(float("-inf")),
                fx.Float32(float("-inf")),
                fx.Float32(0.0),
                fx.Float32(0.0),
            ]

        _encode_waitcnt(vmcnt=0)
        rocdl.sched_barrier(0)
        # 每轮处理 2 个 kv(偶=2*kv, 奇=2*kv+1);frag_ldK/frag_ldK_next 做 coop ping-pong;frag_K 携带 prefetch
        for kv, state in range(
            fx.Index(0), fx.Index(N // BN // 2), fx.Index(1), init=loop_init
        ):
            frag_O.store(state[0])
            frag_ldK.store(state[1])  # 恢复 frag_ldK = K(2*kv+1) coop 数据
            frag_K.store(state[2])  # 恢复上一步 prefetch 的 frag_K = K(2*kv)
            if const_expr(softmax):
                m0, m1, l0, l1 = state[3], state[4], state[5], state[6]
            else:
                m0 = m1 = l0 = l1 = None
            kv0 = fx.Int32(kv) * 2
            m0, m1, l0, l1 = kv_step(
                kv0, 1, frag_ldK, frag_ldK_next, m0, m1, l0, l1
            )  # 偶:写 k_lds[1]
            m0, m1, l0, l1 = kv_step(
                kv0 + 1, 0, frag_ldK_next, frag_ldK, m0, m1, l0, l1
            )  # 奇:写 k_lds[0]

            yield_vals = [
                frag_O.load(),
                frag_ldK.load(),
                frag_K.load(),
            ]  # frag_K = K(2*kv+2)
            if const_expr(softmax):
                yield_vals += [m0, m1, l0, l1]
            results = yield yield_vals
        frag_O.store(results[0])
        if const_expr(softmax):
            # ---- epilogue:O /= l(每 Mq-tile 用最终 running sum 归一化)----
            l_final = [results[5], results[6]]
            for mt in range_constexpr(2):
                ot = frag_O[None, None, mt]
                ot.store(ot.load() * (fx.Float32(1.0) / l_final[mt]))

        # O: f32 -> bf16 -> global(每 wave 写自己的 32 行;_cvt_f32_to_bf16 省 RNE+NaN 指令)
        frag_Ob = _cvt_f32_to_bf16(frag_O)
        tcO = fx.make_tiled_copy_C(cp_oc, tmma2).get_slice(tid)
        fx.copy(cp_oc, tcO.retile(frag_Ob), tcO.partition_S(o_tile))

    @flyc.jit
    def launch(
        Q: fx.Tensor, K: fx.Tensor, V: fx.Tensor, O: fx.Tensor, stream: fx.Stream
    ):
        attn_kernel(Q, K, V, O).launch(
            grid=(M // BM, H, 1), block=(NT, 1, 1), stream=stream
        )

    return launch


def torch_ref(Q, K, V, causal=False, softmax=True):
    """完整 MHA 参考(多头):Q/K/V = [H,M,D]/[H,N,D]/[H,N,D],O = softmax(Q@K^T/sqrt(D))@V。
    softmax=False 时退化为 (Q@K^T)@V(不 scale),用于阶段A 单独验证多头布线。"""
    S = torch.einsum("hmd,hnd->hmn", Q.float(), K.float())  # [H,M,N] f32
    if softmax:
        S = S * (1.0 / (Q.shape[-1] ** 0.5))
        if causal:
            Mq, Nk = S.shape[-2:]
            mask = torch.arange(Nk)[None, :] > (torch.arange(Mq)[:, None] + (Nk - Mq))
            S = S.masked_fill(mask[None], float("-inf"))
        P = torch.softmax(S, dim=-1)  # softmax over N(KV),f32
    else:
        P = S  # 阶段A:无 softmax、无 scale
    O = torch.einsum("hmn,hnd->hmd", P.to(torch.bfloat16).float(), V.float()).to(
        torch.bfloat16
    )
    return S, O


def main():
    torch.manual_seed(0)
    torch.set_default_device("cuda")

    # multi-head flash MHA(kernel 内含在线 softmax);non-causal。H/MULT 可用环境变量扫尺寸。
    H = int(os.environ.get("H", "8"))
    BM, BN = 128, 32
    _mult = int(os.environ.get("MULT", "16"))
    M, N, D = BM * _mult, BM * _mult, 128  # 每 head M=N=BM*MULT(默认 2048)

    Q = torch.randn(H, M, D, dtype=torch.bfloat16)
    K = torch.randn(H, N, D, dtype=torch.bfloat16)
    V = torch.randn(H, N, D, dtype=torch.bfloat16)
    # 预 shuffle V 成 paged 布局 [H, N//8, D, 8];torch_ref 仍用原始 V
    V_shuf = V.reshape(H, N // 8, 8, D).permute(0, 1, 3, 2).contiguous()

    _backend = os.environ.get("ATTN_FLY_BACKEND", "fly")
    assert _backend in ("fly", "jit_all_vgpr", "compare_jit")
    _use_jit_backend = _backend != "fly"
    if _use_jit_backend:
        from pyhip.core.fly_isa_priority import (
            build_all_vgpr_jit_attention_kernel,
            preshuffle_jit_key,
            preshuffle_jit_value,
        )

        K_jit = preshuffle_jit_key(K)
        V_jit = preshuffle_jit_value(V)
    else:
        K_jit = V_jit = None

    stream = torch.cuda.current_stream()
    o_fly = torch.empty(H, M, D, dtype=torch.bfloat16)
    args = (Q, K, V_shuf, o_fly, stream)

    _softmax = (
        os.environ.get("SOFTMAX", "1") == "1"
    )  # 编译时开关:0=无 softmax 纯双 GEMM(应 ~260T)
    if _use_jit_backend and not _softmax:
        raise ValueError("the archived jit_all_vgpr backend requires SOFTMAX=1")
    _priority_mode = os.environ.get("ATTN_FLY_PRIORITY", "off")
    _priority_mode = {"0": "off", "1": "post_isa", "on": "post_isa"}.get(
        _priority_mode, _priority_mode
    )
    assert _priority_mode in ("off", "post_isa", "compare")
    if _backend != "fly" and _priority_mode != "off":
        raise ValueError(
            "ATTN_FLY_PRIORITY is only supported with ATTN_FLY_BACKEND=fly"
        )
    _use_post_isa = (
        _backend == "fly" and _priority_mode != "off" and _softmax and N >= 32768
    )
    _priority_period = None
    _priority_events = None
    _dump_dir = (
        Path(__file__).resolve().parents[2]
        / ".cache"
        / "fly-attn-priority"
        / f"h{H}-m{M}-n{N}"
    )
    if _use_post_isa:
        from pyhip.core.fly_isa_priority import parse_priority_events

        _priority_period = int(os.environ.get("ATTN_FLY_PRIORITY_PERIOD", "64"))
        _priority_events = parse_priority_events(
            os.environ.get("ATTN_FLY_PRIORITY_EVENTS"), period=_priority_period
        )
        os.environ["FLYDSL_DUMP_IR"] = "1"
        os.environ["FLYDSL_DUMP_DIR"] = str(_dump_dir)
    print(
        f"[cfg] H={H} M={M} N={N} D={D} backend={_backend} softmax={_softmax} "
        f"softmax_impl=lazy_delta8 priority={_priority_mode} active={_use_post_isa} "
        f"period={_priority_period} events={_priority_events}"
    )

    compiled = None
    if _backend != "jit_all_vgpr":
        compiled = fly_compiled(
            (M, N, D, BM, BN, H, _softmax),
            lambda: build(M, N, D, BM, BN, H=H, softmax=_softmax),
            args,
        )
    kernel = compiled
    if _use_jit_backend:
        root = Path(__file__).resolve().parents[2]
        jit_kernel, artifact = build_all_vgpr_jit_attention_kernel(
            root
            / "archive/gemm/attn-gemm-jit-setprio-best-gfx942-m40960-n40960-237p1t.s",
            root / ".cache/jit-attn-all-vgpr",
            m=M,
            n=N,
            h=H,
        )
        print(
            f"[jit-all-vgpr] assembly={artifact.assembly_path} code_object={artifact.code_object_path}"
        )
        if _backend == "jit_all_vgpr":
            kernel = jit_kernel
    if _use_post_isa:
        from pyhip.core.fly_isa_priority import build_attention_priority_kernel

        kernel, artifact = build_attention_priority_kernel(
            _find_final_isa(_dump_dir),
            _dump_dir / "post-isa",
            m=M,
            h=H,
            block_m=BM,
            period=_priority_period,
            events=_priority_events,
        )
        print(
            f"[post-isa] assembly={artifact.assembly_path} code_object={artifact.code_object_path}"
        )
        kernel(Q, K, V_shuf, o_fly, stream)
    elif _backend == "jit_all_vgpr":
        kernel(Q, K_jit, V_jit, o_fly, stream)
    elif _backend == "compare_jit":
        jit_kernel(Q, K_jit, V_jit, o_fly, stream)
    torch.cuda.synchronize()

    # ---- 精度 ----
    _, o_ref = torch_ref(Q, K, V, softmax=_softmax)
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
    if _use_jit_backend:
        K_jits = [preshuffle_jit_key(k) for k in Ks]
        V_jits = [preshuffle_jit_value(v) for v in Vs]
    else:
        K_jits = V_jits = None
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

    def paired_perf(control, candidate):
        pair_count = int(os.environ.get("ATTN_FLY_PAIR_COUNT", "12"))
        max_control_drift = float(os.environ.get("ATTN_FLY_MAX_CONTROL_DRIFT", "0.05"))

        def measure(fn, index, name):
            with cudaPerf(flops, mem_bytes, name=name, verbose=0) as measurement:
                fn(index)
            return measurement.dt() * 1e6

        ratios, controls, candidates = [], [], []
        index = 0
        for _ in range(pair_count):
            control_before = measure(control, index, "attn_fly_base_before")
            candidate_first = measure(
                candidate, (index + 1) % BUF_COPY, "attn_fly_priority_first"
            )
            candidate_second = measure(
                candidate, (index + 2) % BUF_COPY, "attn_fly_priority_second"
            )
            control_after = measure(
                control, (index + 3) % BUF_COPY, "attn_fly_base_after"
            )
            index = (index + 4) % BUF_COPY

            control_us = (control_before + control_after) / 2
            candidate_us = (candidate_first + candidate_second) / 2
            control_drift = abs(control_after - control_before) / control_us
            if control_drift <= max_control_drift:
                controls.append(control_us)
                candidates.append(candidate_us)
                ratios.append(candidate_us / control_us)

        if not ratios:
            raise RuntimeError(
                f"no valid C-X-X-C pairs: all {pair_count} controls drifted by more than {max_control_drift:.1%}"
            )

        controls.sort()
        candidates.sort()
        ratios.sort()
        middle = len(ratios) // 2
        return {
            "valid": len(ratios),
            "total": pair_count,
            "control_us": controls[middle],
            "candidate_us": candidates[middle],
            "ratio": ratios[middle],
        }

    if _backend == "compare_jit":
        paired = paired_perf(
            lambda i: compiled(Qs[i], Ks[i], V_shufs[i], o_flys[i], stream),
            lambda i: jit_kernel(Qs[i], K_jits[i], V_jits[i], o_flys[i], stream),
        )
        tf_base = flops / paired["control_us"] / 1e6
        tf_jit = flops / paired["candidate_us"] / 1e6
        print(
            f"[paired-jit] valid={paired['valid']}/{paired['total']} "
            f"fly={paired['control_us']:.1f}us/{tf_base:.1f}T "
            f"jit_all_vgpr={paired['candidate_us']:.1f}us/{tf_jit:.1f}T "
            f"time_ratio={paired['ratio']:.5f} speedup={1 / paired['ratio'] - 1:.2%}"
        )
    elif _priority_mode == "compare" and _use_post_isa:
        paired = paired_perf(
            lambda i: compiled(Qs[i], Ks[i], V_shufs[i], o_flys[i], stream),
            lambda i: kernel(Qs[i], Ks[i], V_shufs[i], o_flys[i], stream),
        )
        tf_base = flops / paired["control_us"] / 1e6
        tf_priority = flops / paired["candidate_us"] / 1e6
        print(
            f"[paired] valid={paired['valid']}/{paired['total']} "
            f"base={paired['control_us']:.1f}us/{tf_base:.1f}T "
            f"priority={paired['candidate_us']:.1f}us/{tf_priority:.1f}T "
            f"time_ratio={paired['ratio']:.5f} speedup={1 / paired['ratio'] - 1:.2%}"
        )
    else:
        if _backend == "jit_all_vgpr":

            def launch(i):
                kernel(Qs[i], K_jits[i], V_jits[i], o_flys[i], stream)

        else:

            def launch(i):
                kernel(Qs[i], Ks[i], V_shufs[i], o_flys[i], stream)

        microseconds, tflops = perf(
            launch,
            f"attn_{_backend}",
        )
        print(
            f"[perf] {_backend}: {microseconds:8.1f} us  {tflops:7.1f} TFLOPS  "
            f"({mem_bytes / microseconds / 1e3:.0f} GB/s)"
        )


if __name__ == "__main__":
    main()
