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

os.environ.setdefault("FLYDSL_RUNTIME_ENABLE_CACHE", "0")

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
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
        ((c_frag.load().bitcast(fx.Uint32) + round_bit) >> 16).to(fx.Uint16).bitcast(fx.BFloat16)
    )
    return c_frag_bf16

def _encode_waitcnt(vmcnt=63, expcnt=7, lgkmcnt=63):
    """Encode s_waitcnt bitfield for CDNA3 (gfx94x)."""
    vm_lo = vmcnt & 0xF
    vm_hi = (vmcnt >> 4) & 0x3
    return vm_lo | (expcnt << 4) | (lgkmcnt << 8) | (vm_hi << 14)


def _make_klds_view(ptr, BN, D):
    """K LDS 视图 [2,BN,D](D 连续,stage 偏移 BN*D):SwizzleType(3,3,3) 组合行主序去 bank 冲突
    -> K 读 ds_read2st64_b64。swz(默认=v12)与 gpermswz 共用此 LDS 布局。"""
    base = fx.make_layout((2, BN, D), (BN * D, D, 1))
    swz = fx.SwizzleType.get(3, 3, 3)
    return fx.make_view(ptr, fx.make_composed_layout(fx.static(swz), base))


def _make_ktiles(K_, Kb, N, D, BN, mode):
    """K coop 读源 tile [BN,D,N//BN,1]。mode="gpermswz":把 perm_M 施加在全局 K cache 而非 MMA:
    每 tile 内 Nk 按正向 k_perm (4,4,2):(D,8D,4D) 重排 -> plain 读入 LDS 后,GEMM1 不加 perm_M 也能
    得到相同的 8-连续-Nk C 布局(N 方向已 shuffle)。否则自然 flat_divide。假设 BN=32。"""
    if mode == "gpermswz":
        return fx.rocdl.make_buffer_tensor(
            fx.make_view(
                fx.get_iter(K_),
                fx.make_layout(((4, 4, 2), D, N // BN, 1), ((D, 8 * D, 4 * D), 1, BN * D, 0)),
            ),
            max_size=False,
        )
    return fx.flat_divide(Kb, fx.make_tile(BN, D))

def build(M, N, D, BM, BN, klds_mode="gpermswz"):
    """构造融合 attention kernel(无 softmax, register-resident S),返回 launch 包装。

    klds_mode: "gpermswz"(默认, perm_M 挪全局 K + 展开2x + K-prefetch, 266.0 TFLOPS @M=N=40960, 2.24x)/
               "swz"(perm_M 在 MMA=v12 路径;此 unroll+K-prefetch 结构下 =219.6,慢于 gpermswz)。
    """
    assert BM % 32 == 0 and BN % 16 == 0 and D % 16 == 0 and N % BN == 0 and M % BM == 0
    assert (N // BN) % 2 == 0, "KV tile 数需为偶数(循环展开 2 次)"
    WAVES = BM // 32                    # 每 wave 负责 32 行 query
    NT = WAVES * 64                     # 线程数
    VECN = 128 // fx.BFloat16.width     # 协作加载向量宽度(8 bf16 = 128b)
    assert BM // 32 == WAVES and D % VECN == 0 and BN % 16 == 0

    @flyc.kernel
    def attn_kernel(Q_: fx.Tensor, K_: fx.Tensor, V_: fx.Tensor, O_: fx.Tensor):
        tid = fx.thread_idx.x
        bm = fx.block_idx.x  # 第几个 query tile

        Q = fx.Tensor(fx.make_view(fx.get_iter(Q_), fx.make_layout((M, D), (D, 1))))
        K = fx.Tensor(fx.make_view(fx.get_iter(K_), fx.make_layout((N, D), (D, 1))))
        # V paged: V_shuf[nb,d,v]=V[nb*8+v,d]。每 kv-tile 在 global 是连续 [BN*D] 块 -> 按 [N,D] 连续视图协作加载
        NB = BN // 8
        # O 存成转置视图 O^T[D,M]:stride (1,D) -> O[m,d] 连续 D;GEMM2 转置后 C=O^T,4/lane 沿 D 连续 -> 64-bit 写
        O = fx.Tensor(fx.make_view(fx.get_iter(O_), fx.make_layout((D, M), (1, D))))
        Qb = fx.rocdl.make_buffer_tensor(Q, max_size=False)
        Kb = fx.rocdl.make_buffer_tensor(K, max_size=False)
        Ob = fx.rocdl.make_buffer_tensor(O, max_size=False)

        q_tile = fx.flat_divide(Qb, fx.make_tile(BM, D))[None, None, bm, 0]  # [BM, D]
        k_tiles = _make_ktiles(K_, Kb, N, D, BN, klds_mode)                  # [BN, D, N//BN, 1](gpermswz=全局重排)
        o_tile = fx.flat_divide(Ob, fx.make_tile(D, BM))[None, None, 0, bm]  # [D, BM] = O^T tile

        mma = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, fx.BFloat16))
        k_perm = fx.make_layout((4, 4, 2), (1, 8, 4))
        # gpermswz: perm_M 挪到全局 K(_make_ktiles) -> MMA 不再加 perm_M(N 方向已 shuffle);否则 perm_M=k_perm
        _pm = None if klds_mode == "gpermswz" else k_perm
        # GEMM1 = K@Q^T: wave 沿 query-M(MFMA 的 N 维)-> (1,WAVES,1);K 维(D)加 k_perm -> K 128-bit;
        # M 维(Nk)加 perm_M -> C 累加器 frag_St 每 lane 8 连续 Nk(对齐 GEMM2 的 k_perm A)
        tmma1 = fx.make_tiled_mma(mma, fx.make_layout((1, WAVES, 1), (1, 1, 0)), fx.make_tile(_pm, None, k_perm))
        # GEMM2 = (S@V)^T = V^T@S^T:交换 A/B -> C=O^T[D,Mq];wave 沿 query-M(现为 N 维)-> (1,WAVES,1);
        # K 维(Nk)加 k_perm -> A(V^T)每 lane 8 Nk -> 128-bit;C 累加器 4/lane 沿 D 连续 -> O 64-bit 写出
        tmma2 = fx.make_tiled_mma(mma, fx.make_layout((1, WAVES, 1), (1, 1, 0)), fx.make_tile(None, None, k_perm))
        thr1 = tmma1.thr_slice(tid)
        thr2 = tmma2.thr_slice(tid)

        cp_cg = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16)  # 协作 global -> reg(合并)
        cp_cs = fx.make_copy_atom(fx.UniversalCopy128b(), fx.BFloat16)     # reg -> LDS
        cp_kr = fx.make_copy_atom(fx.UniversalCopy128b(), fx.BFloat16)     # k_lds -> frag_K(k_perm -> 128-bit)
        cp_vg = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16)  # V paged global -> frag_V(直读,不经 LDS)
        cp_qg = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16)   # Q global -> frag_Q(k_perm -> 128-bit)
        cp_oc = fx.make_copy_atom(fx.rocdl.BufferCopy64b(), fx.BFloat16)   # O 输出(GEMM2 转置 -> C=O^T,4/lane 沿 D 连续 -> 64-bit 写出)

        # LDS: K 双缓冲(ping-pong),2*[BN,D];S 不入 LDS(register trick)
        @fx.struct
        class SharedStorage:
            k_lds: fx.Array[fx.BFloat16, 2 * BN * D, 16]

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        # K LDS 视图:swizzle(3,3,3) 去 bank 冲突(swz 默认 / gpermswz 共用)
        k_lds2 = _make_klds_view(lds.k_lds.ptr, BN, D)
        # V 直读 paged global 读源:每 kv-tile [D,(8,NB)], v(8) inner(stride 1)-> 128-bit buffer_load(不经 LDS)
        v_g = fx.rocdl.make_buffer_tensor(
            fx.make_view(fx.get_iter(V_), fx.make_layout((D, (8, NB), N // BN), (8, (1, D * 8), BN * D))),
            max_size=False,
        )
        # frag_V 的 B 模板:干净 [D,BN] tile(非 paged 嵌套),读取源才用 paged 分区
        v_fake = fx.flat_divide(
            fx.rocdl.make_buffer_tensor(
                fx.make_view(fx.get_iter(V_), fx.make_layout((D, BN), (BN, 1))), max_size=False
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
        frag_O = thr2.make_fragment_C(o_tile)                             # C=O[M,D](跨 KV 循环累加)
        tcK = fx.make_tiled_copy_A(cp_kr, tmma1).get_slice(tid)            # k_lds -> frag_K
        tcV = fx.make_tiled_copy_A(cp_vg, tmma2).get_slice(tid)            # paged global -> frag_V(直读)

        frag_O.fill(0)

        # 双缓冲 prologue:coop K(0)->frag; 写 k_lds[0]; coop K(1)->frag; barrier
        frag_ldK = fx.make_fragment_like(coop_g.partition_S(k_tiles[None, None, 0, 0]))
        fx.copy(cp_cg, coop_g.partition_S(k_tiles[None, None, 0, 0]), frag_ldK)
        fx.copy(cp_cs, frag_ldK, coop_s.partition_D(k_lds2[0, None, None]))              # 写 stage 0
        fx.copy(cp_cg, coop_g.partition_S(k_tiles[None, None, 1, 0]), frag_ldK)          # 预取 K(1) coop
        gpu.barrier()

        acc_init = frag_O.load()
        kcar_init = frag_ldK.load()  # frag_ldK 持 K(kv+1) coop 数据(loop-carried)

        # 循环内 fragment 提到循环外:形状固定,只分配一次,循环内复用(fill/store/copy 仍在循环内)
        frag_K = thr1.make_fragment_A(k_lds2[0, None, None])                                # k_lds -> frag_K
        frag_St = thr1.make_fragment_C(fx.make_rmem_tensor(fx.make_layout((BN, BM), (BM, 1)), fx.Float32))  # GEMM1 C=S^T
        frag_Sb = thr2.make_fragment_B(fx.make_rmem_tensor(fx.make_layout((BM, BN), (BN, 1)), fx.BFloat16))  # GEMM2 B=S^T
        frag_ldK_next = fx.make_fragment_like(coop_g.partition_S(k_tiles[None, None, 0, 0]))  # coop 预取 K(kv+2)
        frag_V = thr2.make_fragment_A(v_fake)                                               # V 直读 -> frag_V

        # hot_loop_scheduler 的指令数按实际 tile 尺寸算(BN=32,D=128 -> 8 dsrd / 2 dswr / 2 vmem / 32 mfma/GEMM)
        WARP = NT // WAVES  # 64
        mfma_per_gemm = (BN // 16) * ((BM // WAVES) // 16) * (D // 16)  # 每个 GEMM 的 MFMA 数
        n_dsrd = BN * D // (WARP * VECN)    # K LDS 读(frag_K,每 wave 读 [BN,D],128-bit)
        n_dswr = BN * D // (NT * VECN)      # coop K LDS 写(NT 线程协作)
        n_vmem = n_dswr                     # coop K global 读(与 coop 写同数)

        def hot_loop_scheduler(is_first_gemm):
            if is_first_gemm:
                for _ in range_constexpr(8):
                    rocdl.sched_vmem(1)
                    rocdl.sched_mfma(3)

                rocdl.sched_vmem(100)
                rocdl.sched_mfma(100)
            else:
                for _ in range_constexpr(n_vmem):
                    rocdl.sched_vmem(1)
                    rocdl.sched_dswr(1)
                    rocdl.sched_mfma(7)
                    # rocdl.sched_group_barrier(2, 4, 0)
                #rocdl.sched_mfma(mfma_per_gemm - 4 * n_dswr - n_dsrd)
                for _ in range_constexpr(n_dsrd):
                    rocdl.sched_dsrd(1)
                    rocdl.sched_mfma(1)
            rocdl.sched_barrier(0)

        # 展开 2 次:偶/奇两步 LDS stage(wr)变编译期常量,消掉 kv%2;fragment 全部复用
        # K 读做成 prefetch:frag_K 在上一步 GEMM2 之后就读好,GEMM1 直接用(藏 LDS 读延迟)
        def kv_step(kv_i, wr, ld_cur, ld_next):
            # V 直读 paged global -> frag_V
            fx.copy(cp_vg, tcV.partition_S(v_g[None, None, kv_i]), tcV.retile(frag_V))
            # GEMM1 = K@Q^T(frag_K 已在上一步/prologue prefetch)
            frag_St.fill(0)
            fx.gemm(mma, frag_St, frag_K, frag_Q, frag_St)
            hot_loop_scheduler(True)
            fx.copy(cp_cg, coop_g.partition_S(k_tiles[None, None, kv_i + 2, 0]), ld_next)
            # register trick:S^T 直接作 GEMM2 的 B
            frag_Stb = _cvt_f32_to_bf16(frag_St)  # add-0x8000 舍入,省 RNE+NaN 指令
            frag_Sb.store(fx.select(frag_Stb, [0, 2, 1]).load())
            # 写 K(kv+1)=ld_cur -> k_lds[wr] + 预取 coop K(kv+2) -> ld_next(与 GEMM2 重叠)
            fx.copy(cp_cs, ld_cur, coop_s.partition_D(k_lds2[wr, None, None]))
            fx.gemm(mma, frag_O, frag_V, frag_Sb, frag_O)  # O^T = V^T @ S^T(交换 A/B)
            gpu.barrier()  # k_lds[wr] 写完可见
            # prefetch 下一步 K:读 k_lds[wr] -> frag_K(移到 GEMM2 之后,藏 LDS 读延迟)
            fx.copy(cp_kr, tcK.partition_S(k_lds2[wr, None, None]), tcK.retile(frag_K))
            hot_loop_scheduler(False)

        # prologue:prefetch 第一个 frag_K = k_lds[0]
        fx.copy(cp_kr, tcK.partition_S(k_lds2[0, None, None]), tcK.retile(frag_K))
        fragK_init = frag_K.load()

        _encode_waitcnt(vmcnt=0)
        rocdl.sched_barrier(0)
        # 每轮处理 2 个 kv(偶=2*kv, 奇=2*kv+1);frag_ldK/frag_ldK_next 做 coop ping-pong;frag_K 携带 prefetch
        for kv, state in range(fx.Index(0), fx.Index(N // BN // 2), fx.Index(1), init=[acc_init, kcar_init, fragK_init]):
            frag_O.store(state[0])
            frag_ldK.store(state[1])  # 恢复 frag_ldK = K(2*kv+1) coop 数据
            frag_K.store(state[2])    # 恢复上一步 prefetch 的 frag_K = K(2*kv)
            kv0 = fx.Int32(kv) * 2
            kv_step(kv0, 1, frag_ldK, frag_ldK_next)      # 偶:写 k_lds[1],prefetch 读 k_lds[1]
            kv_step(kv0 + 1, 0, frag_ldK_next, frag_ldK)  # 奇:写 k_lds[0],prefetch 读 k_lds[0]

            results = yield [frag_O.load(), frag_ldK.load(), frag_K.load()]  # frag_K = K(2*kv+2)
        frag_O.store(results[0])

        # O: f32 -> bf16 -> global(每 wave 写自己的 32 行;_cvt_f32_to_bf16 省 RNE+NaN 指令)
        frag_Ob = _cvt_f32_to_bf16(frag_O)
        tcO = fx.make_tiled_copy_C(cp_oc, tmma2).get_slice(tid)
        fx.copy(cp_oc, tcO.retile(frag_Ob), tcO.partition_S(o_tile))

    @flyc.jit
    def launch(Q: fx.Tensor, K: fx.Tensor, V: fx.Tensor, O: fx.Tensor, stream: fx.Stream):
        attn_kernel(Q, K, V, O).launch(grid=(M // BM, 1, 1), block=(NT, 1, 1), stream=stream)

    return launch


def torch_ref(Q, K, V):
    """torch bf16 参考: S=Q@K^T 舍入到 bf16, 再 O=S@V 舍入到 bf16(与 kernel 数值路径一致)。"""
    S = (Q @ K.transpose(0, 1)).to(torch.bfloat16)  # bf16 matmul, f32 内部累加
    O = (S @ V).to(torch.bfloat16)
    return S, O


def main():
    torch.manual_seed(0)
    torch.set_default_device("cuda")


    # 实际使用的layout:  ``K = [NumBlocks, NumKVHeads, HeadDim / kVectorSize, PageSize, kVectorSize]`` and
    #   ``V = [NumBlocks, NumKVHeads, PageSize / kVectorSize, HeadDim, kVectorSize]``.

    M, N, D = 8192, 8192, 128
    # BN=32 + K LDS 双缓冲:VGPR 190(<256, 2 waves/SIMD),双缓冲藏 K LDS 读延迟 -> 229.8 TFLOPS。
    # (BN=64 单缓冲=183;BN=64 双缓冲 VGPR 265->1 wave 反而慢;BN=32 才有双缓冲的寄存器预算。)
    BM, BN = 128, 32
    M, N, D = BM * 80 * 4, BM * 80 * 4, 128

    Q = torch.randn(M, D, dtype=torch.bfloat16)
    K = torch.randn(N, D, dtype=torch.bfloat16)
    V = torch.randn(N, D, dtype=torch.bfloat16)
    # 方案A: 预 shuffle V 成 paged 布局 [N//8, D, 8];torch_ref 仍用原始 V
    V_shuf = V.reshape(N // 8, 8, D).permute(0, 2, 1).contiguous()

    stream = torch.cuda.current_stream()
    o_fly = torch.empty(M, D, dtype=torch.bfloat16)
    args = (Q, K, V_shuf, o_fly, stream)

    # KLDS 环境变量选布局:"gpermswz"(默认, 266.0 TFLOPS @M=N=40960, 2.24x)/ "swz"(v12 路径, 此结构下=219.6)
    _klds_mode = os.environ.get("KLDS", "gpermswz")
    print(f"[cfg] klds_mode={_klds_mode}")

    # flyc.compile 编译并缓存:首次编译 + 执行一次(填充 o_fly),返回快速派发 callable
    compiled = fly_compiled((M, N, D, BM, BN, _klds_mode), lambda: build(M, N, D, BM, BN, klds_mode=_klds_mode), args)
    torch.cuda.synchronize()

    # ---- 精度 ----
    _, o_ref = torch_ref(Q, K, V)
    diff = (o_fly.float() - o_ref.float()).abs()
    rel = diff.norm() / o_ref.float().norm().clamp_min(1e-6)
    print(f"[acc] max_abs={diff.max().item():.4f} mean_abs={diff.mean().item():.5f} rel_l2={rel.item():.5f}")

    # ---- 性能:多 buffer 轮换避免 L2 命中 + cudaPerf 计时(参考 tests/contrib/moe/test_moe.py)----
    from pyhip import cudaPerf

    flops = 4 * M * N * D  # gemm1 + gemm2, 各 2*M*N*D
    mem_bytes = (Q.numel() + K.numel() + V_shuf.numel() + o_fly.numel()) * 2  # bf16 读写

    # 轮换 BUF_COPY 组输入:每次计时读不同显存 -> L2 冷 -> 测真实 HBM(而非 L2 命中的虚高)
    BUF_COPY = 60
    Qs = [torch.randn(M, D, dtype=torch.bfloat16) for _ in range(BUF_COPY)]
    Ks = [torch.randn(N, D, dtype=torch.bfloat16) for _ in range(BUF_COPY)]
    Vs = [torch.randn(N, D, dtype=torch.bfloat16) for _ in range(BUF_COPY)]
    V_shufs = [v.reshape(N // 8, 8, D).permute(0, 2, 1).contiguous() for v in Vs]
    o_flys = [torch.empty(M, D, dtype=torch.bfloat16) for _ in range(BUF_COPY)]

    run_count = 50

    def perf(fn, name):
        for _ in range(10):  # warmup(稳频)
            fn(0)
        torch.cuda.synchronize()
        tfs, uss = [], []
        i = 0
        for _ in range(run_count):
            # cudaPerf: 进入时 GPU sleep 掩盖 host 派发,再用 CUDA event 计 kernel 时间
            with cudaPerf(flops, mem_bytes, name=name, verbose=0) as p:
                fn(i)
            i = (i + 1) % BUF_COPY  # 轮换 buffer
            tfs.append(p.tflops())
            uss.append(p.dt() * 1e6)
        tfs.sort()
        uss.sort()
        return uss[run_count // 2], tfs[run_count // 2]  # 中位数

    us_fly, tf_fly = perf(lambda i: compiled(Qs[i], Ks[i], V_shufs[i], o_flys[i], stream), "attn_fly")
    us_ref, tf_ref = perf(lambda i: torch_ref(Qs[i], Ks[i], Vs[i]), "attn_torch")
    print(f"[perf] fly  : {us_fly:8.1f} us  {tf_fly:7.1f} TFLOPS  ({mem_bytes / us_fly / 1e3:.0f} GB/s)")
    print(f"[perf] torch: {us_ref:8.1f} us  {tf_ref:7.1f} TFLOPS")


if __name__ == "__main__":
    main()
