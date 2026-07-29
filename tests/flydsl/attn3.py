
import functools
import os

os.environ.setdefault("FLYDSL_RUNTIME_ENABLE_CACHE", "0")

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, const_expr, gpu, range_constexpr, rocdl
from flydsl._mlir.dialects import llvm

import pyhip.contrib.flydsl.helpers as fxh

fxh.dump_ir(True)

import pyhip
pyhip.set_device()

"""
循序渐进开发kernel，先把基础组件完成并组建极简pipeline保证组件功能正常
"""

def _cvt_f32_to_bf16(c_frag):
    """f32 -> bf16:add 0x8000 舍入 + 截断(round-half-up),比 .to(fx.BFloat16) 的 RNE + NaN 处理少指令。
    移植自 src/contrib/flydsl/moe_gemm_splitk.py::_cvt_f32_to_bf16。"""
    c_frag_bf16 = fx.make_fragment_like(c_frag, dtype=fx.BFloat16)
    round_bit = fx.Uint32(0x8000)
    c_frag_bf16.store(
        ((c_frag.load().bitcast(fx.Uint32) + round_bit) >> 16).to(fx.Uint16).bitcast(fx.BFloat16)
    )
    return c_frag_bf16

def s_waitcnt(vmcnt=-1, lgkmcnt=-1):
    asm_string = ""
    asm_string += "s_waitcnt "
    if vmcnt >= 0: asm_string += f"vmcnt({vmcnt}) "
    if lgkmcnt >= 0: asm_string += f"lgkmcnt({lgkmcnt}) "

    llvm.inline_asm(
        res=None,
        operands_=[],
        asm_string=asm_string,
        constraints="",
        has_side_effects=True,
    )


@functools.cache
def MHA(M, N, D, BM, BN, H):
    num_threads = 512
    @flyc.kernel(known_block_size=[num_threads, 1, 1])
    def attn_kernel(Q_: fx.Tensor, K_: fx.Tensor, V_: fx.Tensor, O_: fx.Tensor, S_: fx.Tensor):
        tid = fx.thread_idx.x
        lane_id = fx.thread_idx.x % 64
        wave_id = fx.thread_idx.x // 64
        wave_m = wave_id // 4

        bm = fx.block_idx.x  # 第几个 query tile
        h = fx.block_idx.y  # head 索引(multi-head)
        qo_off = h * (M * D)  # Q/O 的 head 偏移(元素)
        kv_off = h * (N * D)  # K/V 的 head 偏移(元素)
        # 
        # paged layout of V in torch: [H, N//8, D, 8]
        #           V.reshape(H, N // 8, 8, D).permute(0, 1, 3, 2).contiguous()
        # in cute-layout, we put them in [D, N] order as MFMA tile A format [M(D), K(N)]
        # and it was prefetched cooperatively by all waves into LDS in unit of [D, BN]
        # this load would be very inefficient if V is not preshuffled (because BN is only
        # 32x2 = 64bytes). To get best coalescing, special TV-layout is required
        paged_kv_layout = fx.make_layout((D, (8, N // 8)), (8, (1, 8 * D)))

        Q = fx.Tensor(fx.make_view(fx.get_iter(Q_) + qo_off, fx.make_layout((M, D), (D, 1))))
        O = fx.Tensor(fx.make_view(fx.get_iter(O_) + qo_off, fx.make_layout((M, D), (D, 1))))
        K = fx.Tensor(fx.make_view(fx.get_iter(K_) + kv_off, fx.make_layout((N, D), (D, 1))))
        V = fx.Tensor(fx.make_view(fx.get_iter(V_) + kv_off, paged_kv_layout))

        S = fx.Tensor(fx.make_view(fx.get_iter(S_) + qo_off, fx.make_layout((M, N), (N, 1))))
        s_tile = fx.flat_divide(S, fx.make_tile(BM, BN))[None, None, bm, None]
        #assert 0, f"\n{M=} {N=} {D=} {BM=} {BN=} {H=}\n{S}\n{s_tile}\n"

        q_tile = fx.flat_divide(Q, fx.make_tile(BM, D))[None, None, bm, 0]  # [BM, D]
        o_tile = fx.flat_divide(O, fx.make_tile(BM, D))[None, None, bm, 0]  # [BM, D]
        q_tile = fx.rocdl.make_buffer_tensor(q_tile, max_size=False) # (BM,D):(D,1)
        o_tile = fx.rocdl.make_buffer_tensor(o_tile, max_size=False) # (BM,D):(D,1)

        k_tile = fx.flat_divide(K, fx.make_tile(BN, D))[None, None, None, 0] # [BN, D, N//BN]
        v_tile = fx.flat_divide(V, fx.make_tile(D, BN))[None, None, 0, None] # [BN, D, N//BN]
        k_tile = fx.rocdl.make_buffer_tensor(k_tile, max_size=False) # (BN, D, N//BN):(1, D, BN*D)
        v_tile = fx.rocdl.make_buffer_tensor(v_tile, max_size=False) # (D, (8,4), N//BN):(8, (1,8*D), 32*D)
        # assert 0, f"\n{M=} {N=} {D=} {BM=} {BN=} {H=}\n{q_tile}\n{o_tile}\n{k_tile}\n{v_tile}\n"

        """
        to be compatible with V's layout, remap k_tile's BN mode 32:d => (4, 4, 2):(1d, 8d, 4d)
        by composition 1st mode with (4, 4, 2):(1, 8, 4)
        """
        print(k_tile)
        k_tile = fx.composition(k_tile, fx.make_tile(fx.make_layout((4,4,2),(1,8,4)), None, None))
        print(k_tile)

        @fx.struct
        class SharedStorage:
            k_lds: fx.Array[fx.BFloat16, 4 * BN * D, 16]
            v_lds: fx.Array[fx.BFloat16, 4 * BN * D, 16]

        # mask,base,shift, swizzle always in unit of 128b,
        swz_base = ((128 // K_.dtype.width) - 1).bit_length()
        swz = fx.SwizzleType.get(3, swz_base, 3)

        lds = fx.SharedAllocator().allocate(SharedStorage)
        layout_k_lds = fx.make_composed_layout(
            fx.static(swz),
            fx.make_ordered_layout((BN, D, 2), (1, 0, 2)),
        )
        layout_v_lds = fx.make_composed_layout(
            fx.static(swz),
            fx.make_ordered_layout((D, BN, 2), (1, 0, 2)),
        )
        lds_k = lds.k_lds.peek().view(layout_k_lds)
        lds_v = lds.v_lds.peek().view(layout_v_lds)

        flyobj = fxh.FlyObjCache()
        glk_thrcopy, glk_cp_atom = flyobj.get_tiled_copy_coalesced_mn(
                                        k_tile[None, None, 0], copy_atom_bits=128, num_threads = num_threads)
        glk_cp_atom2 = flyobj.get_universal_copy_atom(k_tile.dtype, 128)

        glk_src = glk_thrcopy.partition_S(k_tile)
        glk_dst = glk_thrcopy.partition_D(lds_k)
        glk_frag = fx.make_fragment_like(glk_dst[None, None, None, 0])
        #assert 0, f"\n{glk_src}\n{glk_frag}\n"
        num_vm_cnt_load_k = (fx.size(glk_frag.shape).get_static_leaf_int * glk_frag.dtype.width)//128
        def global_load_k(block_n):
            fx.copy(glk_cp_atom, glk_src[None, None, None, block_n], glk_frag)
            return num_vm_cnt_load_k
        def ds_store_k(lds_buff_id):
            fx.copy(glk_cp_atom2, glk_frag, glk_dst[None, None, None, lds_buff_id & 1])

        # assert 0, f"\n{v_tile}\n{lds_v}\n"
        glv_thrcopy, glv_cp_atom = flyobj.get_tiled_copy_coalesced_mn(
                                    v_tile[None, None, 0], copy_atom_bits=128, num_threads = num_threads)
        glv_cp_atom2 = flyobj.get_universal_copy_atom(v_tile.dtype, 128)
        glv_src = glv_thrcopy.partition_S(v_tile)
        glv_dst = glv_thrcopy.partition_D(lds_v)
        glv_frag = fx.make_fragment_like(glv_dst[None, None, None, 0])
        num_vm_cnt_load_v = (fx.size(glv_frag.shape).get_static_leaf_int * glv_frag.dtype.width)//128
        def global_load_v(block_n):
            fx.copy(glv_cp_atom, glv_src[None, None, None, block_n], glv_frag)
            return num_vm_cnt_load_v
        def ds_store_v(lds_buff_id):
            fx.copy(glv_cp_atom2, glv_frag, glv_dst[None, None, None, lds_buff_id & 1])

        #def ds_load_k(lds_buff_id):
        tmma1 = flyobj.create_thr_mma(fx.BFloat16, (1, 8, 1))
        tmma2 = flyobj.create_thr_mma(fx.BFloat16, (1, 8, 1))

        # B(q_tile) @ A(k_tile) -> f32(p_tile)
        fragQ = flyobj.load_tiled_mma_fragB(tmma1, q_tile) # [4, n_bm, n_bk]

        fakeCt = fx.make_rmem_tensor(fx.make_layout((BN, BM), (BM, 1)), fx.Float32)
        fragS = tmma1.make_fragment_C(fakeCt)
        fragO = tmma2.make_fragment_C(fx.select(o_tile, [1, 0]))

        fragO.fill(0.0)

        """
        8wave 方法可以融合两个完全无关的 4wave 任务，各做各的不share任何东西，只是用条件barrier交替调度
        两个 4wave 任务的热点子任务（mfma/mem/valu）,或者两个4wave任务可以作为一个整体共享部分资源，
        这样可以进一步减少冗余操作提高性能.

        此处8-wave在query token维度并行分割任务，共享kv-cache数据，条件barrier并行调度pipeline的相邻两个stages
        在等待数据就位时需要注意，如果是8wave协同发起的数据加载，需要提前一个barrier发起等待，因为另外4wave会延迟
        一个stage执行。同样的，如果要覆盖一个8wave协同使用的LDS buffer之前，需要保证提前一个barrier发起lgkm的等待
        保证对LDS的访问已经完成。

        32x128个bf16的k/v数据需要被每个wave都完整读入，共32*128*16/128/64=8条ds_read_b128指令， kv一块就是16条
        4个wave一起读，cdna3架构下每个指令需要32cycles, 消耗约32条16x16mfma指令的cycles数， 而这么多数据共产生的mfma
        运算总共也只有64条（包括Q*K,P*V)，光从LDS加载数据就消耗了大约一半的时间，比较不划算（除非外存访问是瓶颈）。

        另一种办法就是不使用LDS，每个wave直接发起冗余外存读取指令，利用cache融合这些同时发起的冗余读取，从而减少LDS的使用和等待时间。
        一个32x128的bf16数据块需要 32个vgpr寄存器，提前预取需要双份，kv一起共需要32x4=128个vgpr寄存器
        fragO/fragQ/fragS加一起也需要接近112个vgpr寄存器, 非常紧张。

        因此 luocheng 使用了 部分LDS方案，只把 K/V 中的一个读入LDS，降低寄存器消耗
        
        我们先尝试一下直接从外存读入 k/v 的8wave方案，此时没有任何共享行为
        不需要依赖barrier的vmcnt等待，

        [misc  + online-softmax]   [mfma P*V      mfma Q*K]


        K的预取比V提前一拍

        fragK[0..1] : k0, k1
        fragV[0..1] : v0, __

        loop:======================

            MFMA S = Q @ fragK[0]

            ::::::::: wave-group barrier ::::::::: 切换调度
            prefetch v1 -> fragV[1]
            prefetch k2 -> fragK[0] # 因为已经使用完K0, 因此预取可以复用相同的fragK寄存器

            softmax(S)=>P

            wait prefetch before to finish
            ::::::::: wave-group barrier ::::::::: 切换调度

            MFMA O += P @ fragV[0]

            MFMA S = Q @ fragK[1]

            ::::::::: wave-group barrier ::::::::: 切换调度
            prefetch v2 -> fragV[0] # 因为已经使用完fragV[0], 因此预取可以复用相同的寄存器
            prefetch k3 -> fragK[1] # 因为已经使用完fragK[1], 因此预取可以复用相同的寄存器

            softmax(S)=>P

            wait prefetch before to finish
            ::::::::: wave-group barrier ::::::::: 切换调度

            MFMA O += P @ fragV[1]

        """

        #global_load_k(0)
        #global_load_v(0)

        # (64 + 64)个vgpr寄存器
        #fragK = [tmma1.make_fragment_A(lds_k[None, None, 0]) for _ in fx.range_constexpr(2)]
        #fragV = [tmma2.make_fragment_A(lds_v[None, None, 0]) for _ in fx.range_constexpr(2)]
        fragK = [None, None]
        fragV = [None, None]

        fragK[0] = flyobj.load_tiled_mma_fragA(tmma1, k_tile, [None, None, 0])
        fragK[1] = flyobj.load_tiled_mma_fragA(tmma1, k_tile, [None, None, 1])
        fragV[0] = flyobj.load_tiled_mma_fragA(tmma2, v_tile, [None, None, 0])

        num_vm_cnt_load_kv = (fx.size(fragK[0].shape).get_static_leaf_int * fragK[0].dtype.width)//128
        s_waitcnt(num_vm_cnt_load_kv*2) # wait k[0]

        if wave_m == 1:
            gpu.barrier()

        num_blocks_n = N // BN
        for block_n in fx.range_constexpr(0, num_blocks_n, 2):
            fragS.fill(0.0)
            fx.gemm(tmma1, fragS, fragK[0], fragQ, fragS)

            rocdl.sched_barrier(0)
            gpu.barrier() # ::::::::: wave-group barrier ::::::::: 切换调度
            rocdl.sched_barrier(0)

            vm_cnt = 0
            if fx.const_expr(block_n+2 < num_blocks_n):
                fragK[0] = flyobj.load_tiled_mma_fragA(tmma1, k_tile, [None, None, block_n+2])
                vm_cnt += num_vm_cnt_load_kv

            if fx.const_expr(block_n+1 < num_blocks_n):
                fragV[1] = flyobj.load_tiled_mma_fragA(tmma2, v_tile, [None, None, block_n+1])
                vm_cnt += num_vm_cnt_load_kv

            rocdl.sched_barrier(0)
            s_waitcnt(vm_cnt) # wait previous load finish
            gpu.barrier() # ::::::::: wave-group barrier ::::::::: 切换调度
            rocdl.sched_barrier(0)

            fragS_bf16 = _cvt_f32_to_bf16(fragS)
            fragS_bf16 = fx.select(fragS_bf16, [0,2,1]) # [fragv, n_bn, n_bm] -> [fragv, n_bm, n_bn]
            fx.gemm(tmma2, fragO, fragV[0], fragS_bf16, fragO)  

            fragS.fill(0.0)
            fx.gemm(tmma1, fragS, fragK[1], fragQ, fragS)

            rocdl.sched_barrier(0)
            gpu.barrier() # ::::::::: wave-group barrier ::::::::: 切换调度
            rocdl.sched_barrier(0)

            vm_cnt = 0
            if fx.const_expr(block_n+1+2 < num_blocks_n):
                fragK[1] = flyobj.load_tiled_mma_fragA(tmma1, k_tile, [None, None, block_n+1+2])
                vm_cnt += num_vm_cnt_load_kv

            if fx.const_expr(block_n+1+1 < num_blocks_n):
                fragV[0] = flyobj.load_tiled_mma_fragA(tmma2, v_tile, [None, None, block_n+1+1])
                vm_cnt += num_vm_cnt_load_kv

            rocdl.sched_barrier(0)
            s_waitcnt(vm_cnt) # wait previous load finish
            gpu.barrier() # ::::::::: wave-group barrier ::::::::: 切换调度
            rocdl.sched_barrier(0)

            fragS_bf16 = _cvt_f32_to_bf16(fragS)
            fragS_bf16 = fx.select(fragS_bf16, [0,2,1]) # [fragv, n_bn, n_bm] -> [fragv, n_bm, n_bn]
            fx.gemm(tmma2, fragO, fragV[1], fragS_bf16, fragO)  

        if wave_m == 0:
            gpu.barrier()

        # save fragO
        fragO_bf16 = _cvt_f32_to_bf16(fragO)
        #assert 0, f"\n{fragO_bf16}\n{o_tile}\n"
        flyobj.store_tiled_mma_fragC(tmma2, fragO_bf16, fx.select(o_tile, [1,0]), copy_atom_bits=64)
        #fx.printf("{}", fragO[0,0,0])


    @flyc.jit
    def launch(Q: fx.Tensor, K: fx.Tensor, V: fx.Tensor, O: fx.Tensor, S:fx.Tensor, stream: fx.Stream):
        # V = V.reshape(H, N // 8, 8, D).permute(0, 1, 3, 2)
        attn_kernel(Q, K, V, O, S).launch(grid=(M // BM, H, 1), block=(num_threads, 1, 1), stream=stream)


    def callable(
        Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, O: torch.Tensor, S: torch.Tensor, stream = None
    ):
        stream = torch.cuda.current_stream() if stream is None else stream
        cf = getattr(launch, "_cf", None)
        if cf is None:
            cf = flyc.compile(launch, Q, K, V, O, S, stream)
            launch._cf = cf
        else:
            cf(Q, K, V, O, S, stream)

    return callable


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
    O = torch.einsum("hmn,hnd->hmd", P.to(torch.bfloat16).float(), V.float()).to(torch.bfloat16)
    return S, O

def main():
    torch.manual_seed(0)
    torch.set_default_device("cuda")

    # multi-head flash MHA(kernel 内含在线 softmax);non-causal。H/MULT 可用环境变量扫尺寸。
    H = int(os.environ.get("H", "80"))
    BM, BN = 256, 32
    _mult = int(os.environ.get("MULT", "1"))
    M, N, D = BM * _mult, BM * _mult, 128  # 每 head M=N=BM*MULT(默认 2048)

    Q = torch.randn(H, M, D, dtype=torch.bfloat16)*0.1
    K = torch.randn(H, N, D, dtype=torch.bfloat16)*0.1
    V = torch.randn(H, N, D, dtype=torch.bfloat16)*0.1

    S = torch.empty(H, M, N, dtype=torch.bfloat16)

    # 预 shuffle V 成 paged 布局 [H, N//8, D, 8];torch_ref 仍用原始 V
    V_shuf = V.reshape(H, N // 8, 8, D).permute(0, 1, 3, 2).contiguous()

    stream = torch.cuda.current_stream()
    o_fly = torch.empty(H, M, D, dtype=torch.bfloat16)
    #args = (Q, K, V_shuf, o_fly, S, stream)

    _softmax = os.environ.get("SOFTMAX", "0") == "1"  # 编译时开关:0=无 softmax 纯双 GEMM(应 ~260T)
    print(f"[cfg] H={H} M={M} N={N} D={D} softmax={_softmax} softmax_impl=lazy_delta8")

    flydsl_mha = MHA(M, N, D, BM, BN, H)
    flydsl_mha(Q, K, V_shuf, o_fly, S, stream)

    torch.cuda.synchronize()

    # ---- 精度 ----
    s_ref, o_ref = torch_ref(Q, K, V, softmax=_softmax)
    #print(o_ref)
    #print(o_fly)
    #assert pyhip.allclose(s_ref.to(torch.bfloat16), S, rtol=1e-2, atol=1e-2)

    assert pyhip.allclose(o_ref, o_fly, rtol=1e-2, atol=1e-2)
    diff = (o_fly.float() - o_ref.float()).abs()
    rel = diff.norm() / o_ref.float().norm().clamp_min(1e-6)
    print(f"[acc] max_abs={diff.max().item():.4f} mean_abs={diff.mean().item():.5f} rel_l2={rel.item():.5f}")

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

    run_count = 10

    def perf(fn, name):
        for _ in range(2):  # warmup
            fn(0)
        torch.cuda.synchronize()
        tfs, uss = [], []
        i = 0
        for _ in range(run_count):
            with cudaPerf(flops, mem_bytes, name=name, verbose=1) as p:
                fn(i)
            i = (i + 1) % BUF_COPY
            tfs.append(p.tflops())
            uss.append(p.dt() * 1e6)
        tfs.sort()
        uss.sort()
        return uss[run_count // 2], tfs[run_count // 2]

    us_fly, tf_fly = perf(lambda i: flydsl_mha(Qs[i], Ks[i], V_shufs[i], o_flys[i], S, stream), f"attn_fly_{M}_{N}_{D}_{H}")
    print(f"[perf] fly  : {us_fly:8.1f} us  {tf_fly:7.1f} TFLOPS  ({mem_bytes / us_fly / 1e3:.0f} GB/s)")


if __name__ == "__main__":
    main()
