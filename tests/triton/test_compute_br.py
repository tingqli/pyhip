# -*- coding: utf-8 -*-
"""Unit test for compute_br: chunk recurrence with h0=0, return final state."""

import math

import torch
import triton
import triton.language as tl

CHUNK_SIZE = 64


# ---------------------------------------------------------------------------
# Triton exp (copied from sglang op.py to be self-contained)
# ---------------------------------------------------------------------------
@triton.jit
def exp(x):
    return tl.math.exp(x)


# ---------------------------------------------------------------------------
# Triton kernel: compute_br
# ---------------------------------------------------------------------------
@triton.heuristics(
    {
        "USE_G": lambda args: args["g"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit(do_not_specialize=["T"])
def chunk_cp_compute_br_kernel(
    k,
    v,
    w,
    g,
    ht,
    cu_seqlens,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos = tl.load(cu_seqlens + i_n).to(tl.int32)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos = i_n * T
    NT = tl.cdiv(T, BT)

    b_h1 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 64:
        b_h2 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 128:
        b_h3 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 192:
        b_h4 = tl.zeros([64, BV], dtype=tl.float32)

    v += ((bos * H + i_h) * V).to(tl.int64)
    k += ((bos * Hg + i_h // (H // Hg)) * K).to(tl.int64)
    w += ((bos * H + i_h) * K).to(tl.int64)
    stride_v = H * V
    stride_k = Hg * K
    stride_w = H * K

    for i_t in range(NT):
        p_w = tl.make_block_ptr(
            w, (T, K), (stride_w, 1), (i_t * BT, 0), (BT, 64), (1, 0)
        )
        b_v = tl.dot(tl.load(p_w, boundary_check=(0, 1)), b_h1.to(w.dtype.element_ty))
        if K > 64:
            p_w = tl.make_block_ptr(
                w, (T, K), (stride_w, 1), (i_t * BT, 64), (BT, 64), (1, 0)
            )
            b_v += tl.dot(
                tl.load(p_w, boundary_check=(0, 1)), b_h2.to(w.dtype.element_ty)
            )
        if K > 128:
            p_w = tl.make_block_ptr(
                w, (T, K), (stride_w, 1), (i_t * BT, 128), (BT, 64), (1, 0)
            )
            b_v += tl.dot(
                tl.load(p_w, boundary_check=(0, 1)), b_h3.to(w.dtype.element_ty)
            )
        if K > 192:
            p_w = tl.make_block_ptr(
                w, (T, K), (stride_w, 1), (i_t * BT, 192), (BT, 64), (1, 0)
            )
            b_v += tl.dot(
                tl.load(p_w, boundary_check=(0, 1)), b_h4.to(w.dtype.element_ty)
            )

        p_v = tl.make_block_ptr(
            v, (T, V), (stride_v, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
        )
        b_v = tl.load(p_v, boundary_check=(0, 1)) - b_v

        last_idx = min((i_t + 1) * BT, T) - 1
        if USE_G:
            m_t = (i_t * BT + tl.arange(0, BT)) < T
            b_g_last = tl.load(g + bos * H + last_idx * H + i_h)
            p_g = tl.make_block_ptr(
                g + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,)
            )
            b_g = tl.load(p_g, boundary_check=(0,))
            b_v = b_v * tl.where(m_t, exp(b_g_last - b_g), 0)[:, None]
            b_g_last = exp(b_g_last)
            b_h1 = b_h1 * b_g_last
            if K > 64:
                b_h2 = b_h2 * b_g_last
            if K > 128:
                b_h3 = b_h3 * b_g_last
            if K > 192:
                b_h4 = b_h4 * b_g_last

        b_v = b_v.to(k.dtype.element_ty)
        p_k = tl.make_block_ptr(
            k, (K, T), (1, stride_k), (0, i_t * BT), (64, BT), (0, 1)
        )
        b_h1 += tl.dot(tl.load(p_k, boundary_check=(0, 1)), b_v)
        if K > 64:
            p_k = tl.make_block_ptr(
                k, (K, T), (1, stride_k), (64, i_t * BT), (64, BT), (0, 1)
            )
            b_h2 += tl.dot(tl.load(p_k, boundary_check=(0, 1)), b_v)
        if K > 128:
            p_k = tl.make_block_ptr(
                k, (K, T), (1, stride_k), (128, i_t * BT), (64, BT), (0, 1)
            )
            b_h3 += tl.dot(tl.load(p_k, boundary_check=(0, 1)), b_v)
        if K > 192:
            p_k = tl.make_block_ptr(
                k, (K, T), (1, stride_k), (192, i_t * BT), (64, BT), (0, 1)
            )
            b_h4 += tl.dot(tl.load(p_k, boundary_check=(0, 1)), b_v)

    ht_base = ht + i_nh * K * V
    tl.store(
        tl.make_block_ptr(ht_base, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0)),
        b_h1.to(ht.dtype.element_ty),
        boundary_check=(0, 1),
    )
    if K > 64:
        tl.store(
            tl.make_block_ptr(
                ht_base, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0)
            ),
            b_h2.to(ht.dtype.element_ty),
            boundary_check=(0, 1),
        )
    if K > 128:
        tl.store(
            tl.make_block_ptr(
                ht_base, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0)
            ),
            b_h3.to(ht.dtype.element_ty),
            boundary_check=(0, 1),
        )
    if K > 192:
        tl.store(
            tl.make_block_ptr(
                ht_base, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0)
            ),
            b_h4.to(ht.dtype.element_ty),
            boundary_check=(0, 1),
        )

    # iterate for each trunk in the zig/zag, i 是trunk index :
    #   h_in = 0 if i==0 else h_out                                            [Dk, Dv]
    #   v_new[:,:,i] = U~[:,:,i] - W←[:,:,i] @ h_in[:,:,i]                     [64, Dv] - [64, Dk]@[Dk,Dv] = [64, Dv]
    #   K-> = k[:,:,i] * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]),[64, Dk] * [64] = [64, Dk]
    #   h_in-> = h_in * gamma_C                                                [Dk,Dv] * [1] = [Dk, Dv]
    #   h_out = h_in-> + K->^T@v_new                                           [Dk,Dv] + [Dk,64]@[64,Dv] = [Dk, Dv]
    # 最终的h_out就是Br

    # Compute affine pairs (b, M) on 1 GPU:
    # Br: [2*B, H, Dk, Dv], `B`通常是batch, 代表有多少个sequence. `2`是seg(zigzag的两半)
    # [B, H, dV/BV] parallel,每个WG负责2个seg的所有chunk B系数的计算。 BV = 32


def compute_br(k, w, u, g, cu_seqlens=None):
    """Run recurrence with h0=0, return only h_final [N, H, K, V]."""
    B, T, Hg, K = k.shape
    H = w.shape[2]
    V = u.shape[-1]
    N = len(cu_seqlens) - 1 if cu_seqlens is not None else B
    ht = torch.empty(N, H, K, V, dtype=torch.float32, device=k.device)

    def grid(meta):
        return (triton.cdiv(V, meta["BV"]), N * H)

    chunk_cp_compute_br_kernel[grid](
        k=k,
        v=u,
        w=w,
        g=g,
        ht=ht,
        cu_seqlens=cu_seqlens,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=CHUNK_SIZE,
        BV=32,
        num_warps=4,
        num_stages=2,
    )
    return ht


# ---------------------------------------------------------------------------
# PyTorch reference
# ---------------------------------------------------------------------------
def compute_br_ref(k, w, u, g, cu_seqlens=None):
    """
    Pure PyTorch reference for compute_br.

    For each sequence (defined by cu_seqlens), run chunk recurrence with h0=0:
        For each chunk t:
            v_new = u_chunk - W_chunk @ h          # [BT, V]
            v_new *= exp(g_last - g_i)              # decay v_new
            h *= exp(g_last)                        # decay h
            h += K_chunk^T @ v_new                  # update h

    Inputs:
        k:  [B, T, Hg, K]   (may have GQA grouped heads)
        w:  [B, T, H, K]
        u:  [B, T, H, V]
        g:  [B, T, H]       (cumulative gate, already cumsum'd)
        cu_seqlens: [N+1]    (cumulative sequence lengths)

    Returns: [N, H, K, V]  final state per sequence per head.
    """
    B, T_total, Hg, K = k.shape
    H = w.shape[2]
    V = u.shape[-1]
    BT = CHUNK_SIZE
    repeat = H // Hg

    if cu_seqlens is not None:
        N = len(cu_seqlens) - 1
    else:
        N = B
        cu_seqlens = torch.arange(0, (B + 1) * T_total, T_total, device=k.device)

    result = torch.zeros(N, H, K, V, dtype=torch.float32, device=k.device)

    for n in range(N):
        bos = cu_seqlens[n].item()
        eos = cu_seqlens[n + 1].item()
        seq_len = eos - bos
        NT = math.ceil(seq_len / BT)

        for h_idx in range(H):
            h = torch.zeros(K, V, dtype=torch.float32, device=k.device)
            hg_idx = h_idx // repeat

            for t in range(NT):
                t_start = t * BT
                t_end = min((t + 1) * BT, seq_len)
                chunk_len = t_end - t_start

                # Extract chunk data — k: [B, T, Hg, K], w: [B, T, H, K], u: [B, T, H, V]
                # Data is packed: token at global position (bos + t_start + i)
                # k layout: k.view(-1, Hg, K), stride along T is Hg*K
                k_chunk = k.view(-1, Hg, K)[
                    bos + t_start : bos + t_end, hg_idx, :
                ]  # [chunk_len, K]
                w_chunk = w.view(-1, H, K)[
                    bos + t_start : bos + t_end, h_idx, :
                ]  # [chunk_len, K]
                u_chunk = u.view(-1, H, V)[
                    bos + t_start : bos + t_end, h_idx, :
                ]  # [chunk_len, V]

                # v_new = u - W @ h
                v_new = u_chunk - w_chunk @ h.to(w_chunk.dtype)  # [chunk_len, V]

                # Apply gate decay
                if g is not None:
                    g_chunk = g.view(-1, H)[
                        bos + t_start : bos + t_end, h_idx
                    ]  # [chunk_len]
                    g_last = g_chunk[chunk_len - 1]
                    decay = torch.exp(g_last - g_chunk)  # [chunk_len]
                    v_new = v_new * decay[:, None]
                    h = h * torch.exp(g_last).float()

                # h += K^T @ v_new
                v_new = v_new.to(k_chunk.dtype)
                h = h + (k_chunk.float().T @ v_new.float())  # [K, V]

            result[n, h_idx] = h

    return result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
import pytest


def _run_compute_br_test(B, T, H, Hg, K, V, dtype, cu_seqlens=None):
    """Helper: run triton vs ref, print stats, assert close."""
    device = "cuda"
    N = len(cu_seqlens) - 1 if cu_seqlens is not None else B

    k = torch.randn(B, T, Hg, K, dtype=dtype, device=device) * 0.02
    w = torch.randn(B, T, H, K, dtype=dtype, device=device) * 0.02
    u = torch.randn(B, T, H, V, dtype=dtype, device=device) * 0.02
    g = torch.randn(B, T, H, dtype=torch.float32, device=device) * 0.01

    out_triton = compute_br(k=k, w=w, u=u, g=g, cu_seqlens=cu_seqlens)
    out_ref = compute_br_ref(k=k, w=w, u=u, g=g, cu_seqlens=cu_seqlens)

    assert out_triton.shape == out_ref.shape == (N, H, K, V)

    abs_diff = (out_triton - out_ref).abs()
    max_diff = abs_diff.max().item()
    rel_diff = (abs_diff / (out_ref.abs() + 1e-6)).max().item()
    cos_sim = torch.nn.functional.cosine_similarity(
        out_triton.reshape(-1).unsqueeze(0).float(),
        out_ref.reshape(-1).unsqueeze(0).float(),
    ).item()

    tag = f"B={B}, T={T}, H={H}, Hg={Hg}, K={K}, V={V}, dtype={dtype}"
    if cu_seqlens is not None:
        tag += ", varlen"
    print(
        f"[{tag}] max_abs={max_diff:.6f}, max_rel={rel_diff:.6f}, cos_sim={cos_sim:.8f}"
    )

    atol = 5e-3
    rtol = 5e-2
    torch.testing.assert_close(out_triton, out_ref, atol=atol, rtol=rtol)


# ---- 主要测试: 不同 sequence length ----
@pytest.mark.parametrize("seq_len", [2048, 4096, 8192, 32768])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_compute_br_seqlen(seq_len, dtype):
    """Single batch, H=Hg=2, K=V=64, varying sequence length."""
    torch.manual_seed(42)
    _run_compute_br_test(B=1, T=seq_len, H=2, Hg=2, K=64, V=64, dtype=dtype)


# ---- 不同 K 大小 (覆盖 kernel 分支 K>64, K>128) ----
@pytest.mark.parametrize("seq_len", [2048, 4096])
@pytest.mark.parametrize("K", [64, 128])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_compute_br_K_sizes(seq_len, K, dtype):
    torch.manual_seed(42)
    _run_compute_br_test(B=1, T=seq_len, H=2, Hg=2, K=K, V=64, dtype=dtype)


# ---- GQA: H != Hg ----
@pytest.mark.parametrize("seq_len", [2048, 4096])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_compute_br_gqa(seq_len, dtype):
    torch.manual_seed(42)
    _run_compute_br_test(B=1, T=seq_len, H=4, Hg=1, K=64, V=64, dtype=dtype)


# ---- varlen: 多条不等长 sequence 拼接 ----
@pytest.mark.parametrize(
    "seq_lens",
    [
        [2048, 2048],
        [4096, 4096],
        [2048, 4096, 2048],
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_compute_br_varlen(seq_lens, dtype):
    torch.manual_seed(42)
    device = "cuda"
    total = sum(seq_lens)
    cu_seqlens = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(seq_lens), 0).tolist()),
        dtype=torch.long,
        device=device,
    )
    _run_compute_br_test(
        B=1, T=total, H=2, Hg=2, K=64, V=64, dtype=dtype, cu_seqlens=cu_seqlens
    )


# ---- 无 gate (g=None) ----
@pytest.mark.parametrize("seq_len", [2048, 4096])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_compute_br_no_gate(seq_len, dtype):
    torch.manual_seed(123)
    device = "cuda"
    B, H, Hg, K, V = 1, 2, 2, 64, 64
    T = seq_len
    k = torch.randn(B, T, Hg, K, dtype=dtype, device=device) * 0.02
    w = torch.randn(B, T, H, K, dtype=dtype, device=device) * 0.02
    u = torch.randn(B, T, H, V, dtype=dtype, device=device) * 0.02

    out_triton = compute_br(k=k, w=w, u=u, g=None, cu_seqlens=None)
    out_ref = compute_br_ref(k=k, w=w, u=u, g=None, cu_seqlens=None)

    assert out_triton.shape == out_ref.shape == (B, H, K, V)
    torch.testing.assert_close(out_triton, out_ref, atol=5e-3, rtol=5e-2)


# ---------------------------------------------------------------------------
# Perf test
# ---------------------------------------------------------------------------
def perf_compute_br(
    B=1,
    T=8192,
    H=16,
    Hg=2,
    K=192,
    V=128,
    dtype=torch.bfloat16,
    num_iters=10,
    data_clones=10,
    warmup=1,
):
    """Benchmark compute_br with clone-rotated inputs to avoid cache effects."""
    import pyhip

    device = "cuda"
    torch.manual_seed(0)

    # --- allocate DATA_CLONES copies ---
    ks = [
        torch.randn(B, T, Hg, K, dtype=dtype, device=device) * 0.01
        for _ in range(data_clones)
    ]
    ws = [
        torch.randn(B, T, H, K, dtype=dtype, device=device) * 0.01
        for _ in range(data_clones)
    ]
    us = [
        torch.randn(B, T, H, V, dtype=dtype, device=device) * 0.01
        for _ in range(data_clones)
    ]
    gs = [
        torch.randn(B, T, H, dtype=torch.float32, device=device) * 0.01
        for _ in range(data_clones)
    ]

    N = B
    # flops: per chunk ~ 2*BT*K*V (W@h) + 2*BT*K*V (K^T@v) = 4*BT*K*V
    # NT chunks total, so ~ NT * 4 * BT * K * V = 4 * T * K * V
    # across N seqs and H heads
    flops = N * H * 4 * T * K * V
    # rw_bytes: read k, w, u, g; write ht
    rw_bytes = (
        B * T * Hg * K * 2  # k (bf16)
        + B * T * H * K * 2  # w (bf16)
        + B * T * H * V * 2  # u (bf16)
        + B * T * H * 4  # g (fp32)
        + N * H * K * V * 4  # ht output (fp32)
    )

    # --- warmup ---
    di = 0
    for _ in range(warmup):
        di = (di + 1) % data_clones
        compute_br(k=ks[di], w=ws[di], u=us[di], g=gs[di], cu_seqlens=None)
    torch.cuda.synchronize()

    # --- timed runs ---
    latency = []
    for i in range(num_iters):
        di = (di + 1) % data_clones
        with pyhip.cudaPerf(flops, rw_bytes, name=f"compute_br_{di}", verbose=0) as p:
            compute_br(k=ks[di], w=ws[di], u=us[di], g=gs[di], cu_seqlens=None)
        latency.append(p.dt_ms)

    latency.sort()
    median = latency[len(latency) // 2]
    best = latency[0]
    tflops = flops / (best * 1e-3) * 1e-12
    bw_gb = rw_bytes / (best * 1e-3) * 1e-9

    print(f"\n{'='*60}")
    print(f"compute_br perf  B={B}, T={T}, H={H}, Hg={Hg}, K={K}, V={V}, {dtype}")
    print(f"  best   = {best:.4f} ms")
    print(f"  median = {median:.4f} ms")
    print(f"  TFLOPS = {tflops:.2f}")
    print(f"  BW     = {bw_gb:.2f} GB/s")
    print(f"  all    = {[f'{x:.4f}' for x in latency[:10]]} ...")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    import sys

    if "--perf" in sys.argv:
        # Parse optional args: --perf --B=2 --T=16000 --H=64 --Hg=16 --K=128 --V=128 --num_iters=20 --data_clones=20 --warmup=1
        kwargs = {}
        for arg in sys.argv[1:]:
            if arg.startswith("--") and "=" in arg:
                key, val = arg[2:].split("=", 1)
                if key in (
                    "B",
                    "T",
                    "H",
                    "Hg",
                    "K",
                    "V",
                    "num_iters",
                    "data_clones",
                    "warmup",
                ):
                    kwargs[key] = int(val)
        perf_compute_br(**kwargs)
    else:
        cu = torch.tensor([0, 16000, 32000], dtype=torch.long, device="cuda")
        _run_compute_br_test(1, 32000, 64, 16, 128, 128, torch.bfloat16, cu_seqlens=cu)
        # pytest.main([__file__, "-v", "-s"])
