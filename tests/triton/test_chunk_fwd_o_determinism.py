# -*- coding: utf-8 -*-
"""
Determinism test for sglang chunk_fwd_o.

Calls chunk_fwd_o N times with the EXACT same inputs and reports
the max absolute diff across runs.  A truly deterministic kernel must
produce bit-equal outputs (max_diff == 0).

Usage:
    python test_chunk_fwd_o_determinism.py
    python test_chunk_fwd_o_determinism.py --T 65536 --H 64 --K 128 --V 128 --runs 5
    python test_chunk_fwd_o_determinism.py --T 65536 --varlen 32768 32768
"""

import argparse
import os

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from sglang.srt.layers.attention.fla.chunk_o import chunk_fwd_o
from sglang.srt.layers.attention.fla.cumsum import chunk_local_cumsum
from sglang.srt.layers.attention.fla.index import prepare_chunk_indices
from sglang.srt.layers.attention.fla.op import exp, safe_exp


# ---------------------------------------------------------------------------
# Variant A: explicit pointer arithmetic (NO tl.make_block_ptr).
# Same math as sglang chunk_fwd_kernel_o, but loads/stores use
# `tl.load(base + offs[:,None]*stride + offs[None,:], mask=...)` directly.
# ---------------------------------------------------------------------------
@triton.jit(do_not_specialize=["T"])
def chunk_fwd_kernel_o_explicit(
    q,
    k,
    v,
    h,
    g,
    o,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v = tl.program_id(0)
    i_t = tl.program_id(1)
    i_bh = tl.program_id(2)
    i_b = i_bh // H
    i_h = i_bh % H

    if IS_VARLEN:
        i_tg = i_t
        i_n = tl.load(chunk_indices + i_t * 2).to(tl.int32)
        i_t = tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos = tl.load(cu_seqlens + i_n).to(tl.int32)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T_local = eos - bos
    else:
        i_tg = i_b * tl.cdiv(T, BT) + i_t
        bos = i_b * T
        T_local = T

    # base pointer offsets per (batch_seq, head)
    q_ptr = q + (bos * Hg + i_h // (H // Hg)) * K
    k_ptr = k + (bos * Hg + i_h // (H // Hg)) * K
    v_ptr = v + (bos * H + i_h) * V
    o_ptr = o + (bos * H + i_h) * V
    h_ptr = h + (i_tg * H + i_h).to(tl.int64) * V * K

    offs_t = i_t * BT + tl.arange(0, BT)  # [BT]
    offs_v = i_v * BV + tl.arange(0, BV)  # [BV]
    mask_t = offs_t < T_local  # [BT]

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    b_A = tl.zeros([BT, BT], dtype=tl.float32)

    # K-loop (cdiv(K, BK) iterations; K, BK constexpr → unrolled)
    for i_k in range(tl.cdiv(K, BK)):
        offs_k = i_k * BK + tl.arange(0, BK)  # [BK]
        # q: [BT, BK], strides (Hg*K, 1)
        b_q = tl.load(
            q_ptr + offs_t[:, None] * (Hg * K) + offs_k[None, :],
            mask=mask_t[:, None] & (offs_k[None, :] < K),
            other=0.0,
        )
        # k: [BK, BT], same memory as q (Hg*K, 1) but transposed view
        b_k = tl.load(
            k_ptr + offs_t[None, :] * (Hg * K) + offs_k[:, None],
            mask=mask_t[None, :] & (offs_k[:, None] < K),
            other=0.0,
        )
        # h: [BV, BK], shape (V, K), strides (K, 1)
        b_h = tl.load(
            h_ptr + offs_v[:, None] * K + offs_k[None, :],
            mask=(offs_v[:, None] < V) & (offs_k[None, :] < K),
            other=0.0,
        )
        b_o += tl.dot(b_q, tl.trans(b_h))
        b_A += tl.dot(b_q, b_k)

    if USE_G:
        g_ptr = g + bos * H + i_h
        b_g = tl.load(g_ptr + offs_t * H, mask=mask_t, other=0.0)
        b_o = b_o * exp(b_g)[:, None]
        b_A = b_A * safe_exp(b_g[:, None] - b_g[None, :])

    o_i = tl.arange(0, BT)
    m_A = o_i[:, None] >= o_i[None, :]
    b_A = tl.where(m_A, b_A, 0.0)

    # v: [BT, BV], strides (H*V, 1)
    b_v = tl.load(
        v_ptr + offs_t[:, None] * (H * V) + offs_v[None, :],
        mask=mask_t[:, None] & (offs_v[None, :] < V),
        other=0.0,
    )

    b_o = b_o * scale + tl.dot(b_A.to(b_v.dtype), b_v) * scale
    tl.store(
        o_ptr + offs_t[:, None] * (H * V) + offs_v[None, :],
        b_o.to(o_ptr.dtype.element_ty),
        mask=mask_t[:, None] & (offs_v[None, :] < V),
    )


def chunk_fwd_o_explicit(
    q,
    k,
    v,
    h,
    g=None,
    scale=None,
    cu_seqlens=None,
    chunk_size=64,
    num_warps=4,
    num_stages=2,
):
    """Drop-in replacement for sglang.chunk_fwd_o using explicit pointer arith."""
    B, T, Hg, K = q.shape
    H, V = v.shape[-2], v.shape[-1]
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    chunk_indices = (
        prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    )
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    if scale is None:
        scale = K**-0.5
    o = torch.zeros_like(v)
    BK = 64
    BV = 64
    grid = (triton.cdiv(V, BV), NT, B * H)
    chunk_fwd_kernel_o_explicit[grid](
        q,
        k,
        v,
        h,
        g,
        o,
        cu_seqlens,
        chunk_indices,
        scale,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
        USE_G=g is not None,
        IS_VARLEN=cu_seqlens is not None,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return o


def chunk_fwd_o_explicit_ns1(
    q, k, v, h, g=None, scale=None, cu_seqlens=None, chunk_size=64
):
    return chunk_fwd_o_explicit(
        q, k, v, h, g, scale, cu_seqlens, chunk_size, num_warps=4, num_stages=1
    )


def chunk_fwd_o_explicit_nw2(
    q, k, v, h, g=None, scale=None, cu_seqlens=None, chunk_size=64
):
    return chunk_fwd_o_explicit(
        q, k, v, h, g, scale, cu_seqlens, chunk_size, num_warps=2, num_stages=2
    )


def chunk_fwd_o_explicit_nw1(
    q, k, v, h, g=None, scale=None, cu_seqlens=None, chunk_size=64
):
    return chunk_fwd_o_explicit(
        q, k, v, h, g, scale, cu_seqlens, chunk_size, num_warps=1, num_stages=1
    )


def chunk_fwd_o_torch_ref(
    q,  # [B, T, Hg, K]
    k,  # [B, T, Hg, K]
    v,  # [B, T, H,  V]   (this is v_new)
    h,  # [B, NT, H, V, K]  per-chunk start state
    g=None,  # [B, T, H]  cumsum of log decay
    scale=None,
    cu_seqlens=None,
    chunk_size=64,
):
    """Deterministic PyTorch reference mirroring `chunk_fwd_kernel_o`:

        inter[t] = exp(g[t]) * (Q[t] @ h[t].T)
        intra[t] = (tril(Q[t] @ K[t].T) * exp(g_row - g_col)) @ V[t]
        O[t]     = (inter + intra) * scale

    fp32 accumulators; A is round-tripped through v.dtype before the second
    matmul, matching the kernel.  Hg < H handled via repeat_interleave.
    """
    B, T, Hg, K = q.shape
    _, _, H, V = v.shape
    BT = chunk_size
    if scale is None:
        scale = K**-0.5
    head_ratio = H // Hg
    assert H % Hg == 0
    out_dtype = v.dtype
    o = torch.zeros_like(v)

    # (flat_chunk_offset, batch_idx, bos, eos)
    if cu_seqlens is not None:
        assert B == 1, "varlen path expects B=1"
        cu = cu_seqlens.tolist()
        seqs = []
        flat = 0
        for i_n in range(len(cu) - 1):
            bos, eos = cu[i_n], cu[i_n + 1]
            seqs.append((flat, 0, bos, eos))
            flat += (eos - bos + BT - 1) // BT
    else:
        NT_b = (T + BT - 1) // BT
        seqs = [(b * NT_b, b, 0, T) for b in range(B)]

    for chunk_base, b, bos, eos in seqs:
        T_n = eos - bos
        NT_n = (T_n + BT - 1) // BT
        for i_t in range(NT_n):
            t0 = bos + i_t * BT
            t1 = min(t0 + BT, eos)
            Lt = t1 - t0

            q_blk = q[b, t0:t1].float().repeat_interleave(head_ratio, dim=1)  # [Lt,H,K]
            k_blk = k[b, t0:t1].float().repeat_interleave(head_ratio, dim=1)
            v_blk = v[b, t0:t1].float()  # [Lt, H, V]
            h_blk = h[b, chunk_base + i_t].float()  # [H, V, K]

            inter = torch.einsum("lhk,hvk->lhv", q_blk, h_blk)
            A = torch.einsum("lhk,mhk->hlm", q_blk, k_blk)

            if g is not None:
                g_blk = g[b, t0:t1].float()  # [Lt, H]
                inter = inter * torch.exp(g_blk).unsqueeze(-1)
                gd = g_blk.transpose(0, 1)  # [H, Lt]
                A = A * torch.exp(gd[:, :, None] - gd[:, None, :])

            mask = torch.tril(torch.ones(Lt, Lt, dtype=torch.bool, device=q.device))
            A = torch.where(mask[None], A, A.new_zeros(()))

            # mimic kernel: cast A to v.dtype before second matmul
            A_v = A.to(out_dtype).float()
            intra = torch.einsum("hlm,mhv->lhv", A_v, v_blk)

            ob = (inter + intra) * scale
            o[b, t0:t1] = ob.to(out_dtype)

    return o


def build_inputs(T, H, K, V, varlen, device, dtype):
    torch.manual_seed(0)
    q = torch.randn(1, T, H, K, dtype=dtype, device=device)
    k = F.normalize(
        torch.randn(1, T, H, K, dtype=torch.float32, device=device), p=2, dim=-1
    ).to(dtype)
    v = torch.randn(1, T, H, V, dtype=dtype, device=device)
    g = F.logsigmoid(torch.rand(1, T, H, dtype=dtype, device=device))
    g_cum = chunk_local_cumsum(
        g,
        chunk_size=64,
        cu_seqlens=(
            torch.tensor(
                [0, T] if not varlen else [0, *torch.tensor(varlen).cumsum(0).tolist()],
                dtype=torch.long,
                device=device,
            )
        ),
    )

    if varlen:
        cu = torch.tensor(
            [0, *torch.tensor(varlen).cumsum(0).tolist()],
            dtype=torch.long,
            device=device,
        )
    else:
        cu = torch.tensor([0, T], dtype=torch.long, device=device)

    NT = T // 64
    # h is the per-chunk start state; chunk_fwd_o only reads it.
    # Must match q/k/v dtype (chunk_fwd_o does tl.dot(q, h) which requires
    # same operand dtype).  In the real GDN flow, h is allocated via
    # k.new_empty(...) so it inherits k's dtype.
    h = torch.randn(1, NT, H, V, K, dtype=dtype, device=device)
    return q, k, v, g_cum, h, cu


VARIANTS = {
    "kernel": chunk_fwd_o,  # original sglang kernel (make_block_ptr)
    "explicit": chunk_fwd_o_explicit,  # variant A: explicit pointer arithmetic
    "explicit_ns1": chunk_fwd_o_explicit_ns1,  # explicit, num_stages=1 (no pipelining)
    "explicit_nw2": chunk_fwd_o_explicit_nw2,  # explicit, num_warps=2
    "explicit_nw1": chunk_fwd_o_explicit_nw1,  # explicit, num_warps=1, num_stages=1
}


def run_test(T, H, K, V, varlen, runs, dtype, variant="kernel"):
    device = torch.device("cuda")
    q, k, v, g, h, cu = build_inputs(T, H, K, V, varlen, device, dtype)
    scale = K**-0.5
    fwd = VARIANTS[variant]

    print(
        f"[determinism test] variant={variant} T={T} H={H} K={K} V={V} "
        f"dtype={dtype} varlen={varlen} runs={runs}"
    )

    outputs = []
    for i in range(runs):
        # L2 cache flush to mimic real workloads
        torch.cuda.empty_cache()
        flush = torch.empty(256 * 1024 * 1024 // 4, dtype=torch.int32, device=device)
        flush.zero_()

        o = fwd(
            q=q,
            k=k,
            v=v,
            h=h,
            g=g,
            scale=scale,
            cu_seqlens=cu,
        )
        torch.cuda.synchronize()
        outputs.append(o.detach().clone())

    o0 = outputs[0]
    print(f"  o[0]  shape={tuple(o0.shape)}  dtype={o0.dtype}")
    max_diff = 0.0
    nonzero_pairs = 0
    for i in range(1, runs):
        d = (outputs[i].float() - o0.float()).abs().max().item()
        bit_eq = torch.equal(outputs[i], o0)
        flag = "BIT-EQUAL" if bit_eq else f"DIFF max={d:.3e}"
        print(f"  run{i} vs run0: {flag}")
        if not bit_eq:
            nonzero_pairs += 1
        max_diff = max(max_diff, d)

    # If non-deterministic, compare each run against torch ref to see whose
    # error is bigger and whether some runs are way off the ground truth.
    if nonzero_pairs > 0:
        print()
        print("  [vs torch ref]  (ref is fp32 deterministic ground truth)")
        o_ref = chunk_fwd_o_torch_ref(
            q=q, k=k, v=v, h=h, g=g, scale=scale, cu_seqlens=cu
        )
        ref_abs_mean = o_ref.float().abs().mean().item()
        for i, oi in enumerate(outputs):
            diff = (oi.float() - o_ref.float()).abs()
            mx = diff.max().item()
            mn = diff.mean().item()
            rel = mn / (ref_abs_mean + 1e-9)
            print(
                f"    run{i} vs ref: max_abs={mx:.3e}  mean_abs={mn:.3e}  "
                f"rel={rel:.3e}"
            )

    print()
    if nonzero_pairs == 0:
        print(f"[PASS] chunk_fwd_o is deterministic across {runs} runs")
        return True
    else:
        print(
            f"[FAIL] chunk_fwd_o is NON-DETERMINISTIC: "
            f"{nonzero_pairs}/{runs - 1} runs differ from run0, "
            f"max_diff={max_diff:.3e}"
        )
        return False


def run_parity(T, H, K, V, varlen, dtype, variant="kernel"):
    """Compare torch ref against the selected kernel variant."""
    device = torch.device("cuda")
    q, k, v, g, h, cu = build_inputs(T, H, K, V, varlen, device, dtype)
    scale = K**-0.5
    fwd = VARIANTS[variant]

    o_kernel = fwd(q=q, k=k, v=v, h=h, g=g, scale=scale, cu_seqlens=cu)
    o_ref = chunk_fwd_o_torch_ref(q=q, k=k, v=v, h=h, g=g, scale=scale, cu_seqlens=cu)
    diff = (o_kernel.float() - o_ref.float()).abs()
    max_d = diff.max().item()
    mean_d = diff.mean().item()
    ref_abs = o_ref.float().abs().mean().item()
    rel = mean_d / (ref_abs + 1e-9)
    # bf16 mantissa ~7 bits; with two matmuls + scale, max_abs ~ a few * 1e-2.
    ok = max_d < 5e-2 and rel < 5e-3
    print(
        f"[parity] variant={variant} T={T:>6} varlen={str(varlen):<20} "
        f"max_abs={max_d:.3e}  mean_abs={mean_d:.3e}  rel={rel:.3e}  "
        f"-> {'PASS' if ok else 'FAIL'}"
    )
    return ok


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--T", type=int, default=65536, help="single-seq length (ignored if --varlen)"
    )
    p.add_argument("--H", type=int, default=64)
    p.add_argument("--K", type=int, default=128)
    p.add_argument("--V", type=int, default=128)
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument(
        "--varlen",
        type=int,
        nargs="+",
        default=None,
        help="varlen seq lengths, e.g. --varlen 16384 32768 16384",
    )
    p.add_argument(
        "--mode",
        choices=["determinism", "parity", "both"],
        default="determinism",
        help="determinism = repeat-run check; parity = ref vs kernel; both = both",
    )
    p.add_argument(
        "--variant",
        choices=list(VARIANTS.keys()),
        default="kernel",
        help="which kernel implementation to test",
    )
    args = p.parse_args()

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[
        args.dtype
    ]
    T = sum(args.varlen) if args.varlen else args.T

    # Run a small suite by default to catch the bug under different shapes.
    cases = [
        # (T, varlen)
        (T, args.varlen),
    ]
    # Also exercise some common shapes if user didn't override.
    if args.T == 65536 and args.varlen is None:
        cases += [
            (4096, None),
            (8192, None),
            (16384, None),
            (32768, None),
            (65536, [32768, 32768]),
        ]

    all_pass = True
    for tt, vl in cases:
        if args.mode in ("determinism", "both"):
            ok = run_test(
                tt, args.H, args.K, args.V, vl, args.runs, dtype, variant=args.variant
            )
            all_pass = all_pass and ok
        if args.mode in ("parity", "both"):
            ok = run_parity(tt, args.H, args.K, args.V, vl, dtype, variant=args.variant)
            all_pass = all_pass and ok
        print("-" * 70)

    exit(0 if all_pass else 1)
