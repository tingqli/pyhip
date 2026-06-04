# -*- coding: utf-8 -*-
# Context Parallelism — Zigzag variant for Gated Delta Rule
# Ported from rtp-llm, adapted to reuse sglang's existing FLA kernels.
#
# Each rank computes on its own zigzag tokens directly, no QKV all-gather needed.
# Each rank has 2 segments (front half + back half of zigzag).
# Total 2*cp_size segments form a causal chain.
#
# Phase 1: Each rank computes (b, M) for its 2 segments locally
# Phase 2: All-gather (b, M) from all ranks
# Phase 3: cp_merge to compute h0_true for each segment
# Phase 4: Rerun fwd_h + fwd_o with correct h0
#
# New Triton kernels (compute_br, compute_M_total, cp_merge) use [N, H, K, V]
# layout internally. The public API uses sglang convention [N, H, V, K] for
# initial_state / final_state; transposition happens at the boundary.

from typing import Optional

import torch
import torch.distributed as dist
import triton
import triton.language as tl

from sglang.srt.layers.attention.fla.chunk_delta_h import chunk_gated_delta_rule_fwd_h
from sglang.srt.layers.attention.fla.chunk_fwd import chunk_gated_delta_rule_fwd_intra
from sglang.srt.layers.attention.fla.chunk_o import chunk_fwd_o

# from sglang.srt.layers.attention.fla.cp.utils import (
#     build_segment_cu_seqlens,
#     causal_positions,
#     zigzag_causal_order,
# )
from sglang.srt.layers.attention.fla.cumsum import chunk_local_cumsum
from sglang.srt.layers.attention.fla.index import prepare_chunk_indices
from sglang.srt.layers.attention.fla.l2norm import l2norm_fwd
from sglang.srt.layers.attention.fla.op import exp

CHUNK_SIZE = 64


__all__ = ["chunk_gated_delta_rule_fwd_cp_zigzag", "build_segment_cu_seqlens"]


from typing import Tuple

import torch

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def zigzag_causal_order(cp_size: int) -> list:
    """Map all-gather layout to causal order.

    All-gather layout: [rank0_seg0, rank0_seg1, rank1_seg0, rank1_seg1, ...]
    Causal order (zigzag): rank0_seg0, rank1_seg0, ..., rankN_seg0,
                           rankN_seg1, ..., rank1_seg1, rank0_seg1

    Returns indices into the all-gather layout that produce causal order.
    """
    num_segs = 2 * cp_size
    order = []
    for pos in range(num_segs):
        if pos < cp_size:
            rank = pos
            seg = 0
        else:
            rank = num_segs - 1 - pos
            seg = 1
        order.append(rank * 2 + seg)
    return order


def causal_positions(rank: int, cp_size: int) -> Tuple[int, int]:
    """Return the causal chain positions of this rank's seg0 and seg1."""
    seg0_pos = rank
    seg1_pos = 2 * cp_size - 1 - rank
    return seg0_pos, seg1_pos


def build_segment_cu_seqlens(cu_seqlens: torch.Tensor) -> torch.Tensor:
    """Build cu_seqlens that treats each sequence's two halves as separate sequences.

    Input cu_seqlens: [0, L0, L0+L1, ...]  (batch+1 entries)
    Output: [0, L0/2, L0, L0+L1/2, L0+L1, ...]  (2*batch+1 entries)
    """
    lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    half_lengths = lengths // 2
    batch_size = lengths.shape[0]
    seg_cu = torch.zeros(
        2 * batch_size + 1, dtype=cu_seqlens.dtype, device=cu_seqlens.device
    )
    for b in range(batch_size):
        seg_cu[2 * b + 1] = seg_cu[2 * b] + half_lengths[b]
        seg_cu[2 * b + 2] = seg_cu[2 * b + 1] + half_lengths[b]
    return seg_cu


# ---------------------------------------------------------------------------
# Triton kernel: compute affine bias b = h_final with h0=0
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

    # store only final state — layout [N*H, K, V]
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
# Triton kernel: compute affine matrix M_total
# ---------------------------------------------------------------------------


@triton.heuristics(
    {
        "USE_G": lambda args: args["g"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit(do_not_specialize=["T"])
def chunk_cp_compute_M_total_kernel(
    k,
    w,
    g,
    M_total,
    cu_seqlens,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    USE_G: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_col, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos = tl.load(cu_seqlens + i_n).to(tl.int32)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos = i_n * T
    NT = tl.cdiv(T, BT)

    o_row = tl.arange(0, 64)
    o_col = tl.arange(0, BK) + i_col * BK
    b_dh1 = (o_row[:, None] == o_col[None, :]).to(tl.float32)
    if K > 64:
        b_dh2 = ((o_row + 64)[:, None] == o_col[None, :]).to(tl.float32)
    if K > 128:
        b_dh3 = ((o_row + 128)[:, None] == o_col[None, :]).to(tl.float32)
    if K > 192:
        b_dh4 = ((o_row + 192)[:, None] == o_col[None, :]).to(tl.float32)

    k += ((bos * Hg + i_h // (H // Hg)) * K).to(tl.int64)
    w += ((bos * H + i_h) * K).to(tl.int64)
    stride_k = Hg * K
    stride_w = H * K

    for i_t in range(NT):
        p_w = tl.make_block_ptr(
            w, (T, K), (stride_w, 1), (i_t * BT, 0), (BT, 64), (1, 0)
        )
        b_wdh = tl.dot(
            tl.load(p_w, boundary_check=(0, 1)), b_dh1.to(w.dtype.element_ty)
        )
        if K > 64:
            p_w = tl.make_block_ptr(
                w, (T, K), (stride_w, 1), (i_t * BT, 64), (BT, 64), (1, 0)
            )
            b_wdh += tl.dot(
                tl.load(p_w, boundary_check=(0, 1)), b_dh2.to(w.dtype.element_ty)
            )
        if K > 128:
            p_w = tl.make_block_ptr(
                w, (T, K), (stride_w, 1), (i_t * BT, 128), (BT, 64), (1, 0)
            )
            b_wdh += tl.dot(
                tl.load(p_w, boundary_check=(0, 1)), b_dh3.to(w.dtype.element_ty)
            )
        if K > 192:
            p_w = tl.make_block_ptr(
                w, (T, K), (stride_w, 1), (i_t * BT, 192), (BT, 64), (1, 0)
            )
            b_wdh += tl.dot(
                tl.load(p_w, boundary_check=(0, 1)), b_dh4.to(w.dtype.element_ty)
            )

        last_idx = min((i_t + 1) * BT, T) - 1
        if USE_G:
            m_t = (i_t * BT + tl.arange(0, BT)) < T
            b_g_last = tl.load(g + bos * H + last_idx * H + i_h)
            p_g = tl.make_block_ptr(
                g + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,)
            )
            b_g = tl.load(p_g, boundary_check=(0,))
            b_wdh = b_wdh * tl.where(m_t, exp(b_g_last - b_g), 0)[:, None]
            b_g_last_exp = exp(b_g_last)
            b_dh1 = b_dh1 * b_g_last_exp
            if K > 64:
                b_dh2 = b_dh2 * b_g_last_exp
            if K > 128:
                b_dh3 = b_dh3 * b_g_last_exp
            if K > 192:
                b_dh4 = b_dh4 * b_g_last_exp

        b_wdh = b_wdh.to(k.dtype.element_ty)
        p_k = tl.make_block_ptr(
            k, (K, T), (1, stride_k), (0, i_t * BT), (64, BT), (0, 1)
        )
        b_dh1 -= tl.dot(tl.load(p_k, boundary_check=(0, 1)), b_wdh)
        if K > 64:
            p_k = tl.make_block_ptr(
                k, (K, T), (1, stride_k), (64, i_t * BT), (64, BT), (0, 1)
            )
            b_dh2 -= tl.dot(tl.load(p_k, boundary_check=(0, 1)), b_wdh)
        if K > 128:
            p_k = tl.make_block_ptr(
                k, (K, T), (1, stride_k), (128, i_t * BT), (64, BT), (0, 1)
            )
            b_dh3 -= tl.dot(tl.load(p_k, boundary_check=(0, 1)), b_wdh)
        if K > 192:
            p_k = tl.make_block_ptr(
                k, (K, T), (1, stride_k), (192, i_t * BT), (64, BT), (0, 1)
            )
            b_dh4 -= tl.dot(tl.load(p_k, boundary_check=(0, 1)), b_wdh)

    M_base = M_total + i_nh * K * K
    tl.store(
        tl.make_block_ptr(M_base, (K, K), (K, 1), (0, i_col * BK), (64, BK), (1, 0)),
        b_dh1.to(M_total.dtype.element_ty),
        boundary_check=(0, 1),
    )
    if K > 64:
        tl.store(
            tl.make_block_ptr(
                M_base, (K, K), (K, 1), (64, i_col * BK), (64, BK), (1, 0)
            ),
            b_dh2.to(M_total.dtype.element_ty),
            boundary_check=(0, 1),
        )
    if K > 128:
        tl.store(
            tl.make_block_ptr(
                M_base, (K, K), (K, 1), (128, i_col * BK), (64, BK), (1, 0)
            ),
            b_dh3.to(M_total.dtype.element_ty),
            boundary_check=(0, 1),
        )
    if K > 192:
        tl.store(
            tl.make_block_ptr(
                M_base, (K, K), (K, 1), (192, i_col * BK), (64, BK), (1, 0)
            ),
            b_dh4.to(M_total.dtype.element_ty),
            boundary_check=(0, 1),
        )


def compute_M_total(k, w, g, cu_seqlens=None):
    """Compute affine matrix M for each segment [N, H, K, K]."""
    B, T, Hg, K = k.shape
    H = w.shape[2]
    BK = min(64, K)
    N = len(cu_seqlens) - 1 if cu_seqlens is not None else B
    M_total = torch.empty(N, H, K, K, dtype=torch.float32, device=k.device)

    def grid(meta):
        return (triton.cdiv(K, meta["BK"]), N * H)

    chunk_cp_compute_M_total_kernel[grid](
        k=k,
        w=w,
        g=g,
        M_total=M_total,
        cu_seqlens=cu_seqlens,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        BT=CHUNK_SIZE,
        BK=BK,
        num_warps=4,
        num_stages=2,
    )
    return M_total


# ---------------------------------------------------------------------------
# Triton kernel: cp_merge — walk the causal chain of (M, b) affines on h0
# ---------------------------------------------------------------------------


@triton.jit(do_not_specialize=["num_ranks"])
def cp_merge_kernel(
    h_out,
    ag_hm,
    h0,
    causal_order_ptr,
    num_ranks,
    N: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BV: tl.constexpr,
    BK: tl.constexpr,
    HAS_H0: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n = i_nh // H
    i_h = i_nh % H

    stride_rank = N * H * K * (V + K)
    ag_base = (i_n * H + i_h) * K * (V + K)

    if HAS_H0:
        p_h0 = tl.make_block_ptr(
            h0 + (i_n * H + i_h) * K * V,
            (K, V),
            (V, 1),
            (0, i_v * BV),
            (BK, BV),
            (1, 0),
        )
        b_h = tl.load(p_h0, boundary_check=(0, 1)).to(tl.float32)
    else:
        b_h = tl.zeros([BK, BV], dtype=tl.float32)

    for r in range(num_ranks):
        r_actual = tl.load(causal_order_ptr + r).to(tl.int64)
        base = r_actual * stride_rank + ag_base
        p_b = tl.make_block_ptr(
            ag_hm + base, (K, V), (V + K, 1), (0, i_v * BV), (BK, BV), (1, 0)
        )
        p_m = tl.make_block_ptr(
            ag_hm + base + V, (K, K), (V + K, 1), (0, 0), (BK, BK), (1, 0)
        )
        b_b = tl.load(p_b, boundary_check=(0, 1)).to(tl.float32)
        b_m = tl.load(p_m, boundary_check=(0, 1)).to(tl.float32)
        b_h = tl.dot(b_m, b_h) + b_b

    p_out = tl.make_block_ptr(
        h_out + (i_n * H + i_h) * K * V,
        (K, V),
        (V, 1),
        (0, i_v * BV),
        (BK, BV),
        (1, 0),
    )
    tl.store(p_out, b_h.to(p_out.dtype.element_ty), boundary_check=(0, 1))


def cp_merge(ag_hm, h0, num_ranks, N, H, K, V, causal_order):
    """Apply num_ranks affine transforms on h0 in causal order.

    h0 layout: [N, H, K, V].
    ag_hm layout: [2*cp_size, N, H, K, V+K].
    Returns [N, H, K, V].
    """
    BK = triton.next_power_of_2(K)
    BV = 32
    h_out = torch.empty(N, H, K, V, dtype=torch.float32, device=ag_hm.device)

    def grid(meta):
        return (triton.cdiv(V, meta["BV"]), N * H)

    cp_merge_kernel[grid](
        h_out=h_out,
        ag_hm=ag_hm,
        h0=h0,
        causal_order_ptr=causal_order,
        num_ranks=num_ranks,
        N=N,
        H=H,
        K=K,
        V=V,
        BV=BV,
        BK=BK,
        HAS_H0=h0 is not None,
        num_warps=4,
        num_stages=2,
    )
    return h_out


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def chunk_gated_delta_rule_fwd_cp_zigzag(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: Optional[torch.Tensor],
    output_final_state: bool,
    cp_group: dist.ProcessGroup,
    scale: Optional[float] = None,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
    seg_cu: Optional[torch.LongTensor] = None,
    causal_order: Optional[torch.Tensor] = None,
    chunk_indices: Optional[torch.Tensor] = None,
    fwd_o_fn=None,
):
    """
    CP-parallel gated delta rule forward — zigzag variant.

    Args:
        initial_state: [N, H, K, V] float32.
        All other args match sglang's chunk_gated_delta_rule_fwd.

    Returns:
        o: output tensor
        h_all: per-chunk hidden states
        final_state: [N, H, K, V] if output_final_state else None
    """
    rank = dist.get_rank(cp_group)
    cp_size = dist.get_world_size(cp_group)
    num_segs = 2 * cp_size

    if use_qk_l2norm_in_kernel:
        q = l2norm_fwd(q)
        k = l2norm_fwd(k)

    if scale is None:
        scale = k.shape[-1] ** -0.5

    # ---- Phase 1: local compute for both segments ----
    if seg_cu is None:
        seg_cu = build_segment_cu_seqlens(cu_seqlens)

    g = chunk_local_cumsum(g, chunk_size=CHUNK_SIZE, cu_seqlens=seg_cu)

    # Reuse sglang's fused intra kernel (kkt + solve_tril + recompute_w_u)
    if chunk_indices is None:
        chunk_indices = prepare_chunk_indices(seg_cu, CHUNK_SIZE)
    w, u, A = chunk_gated_delta_rule_fwd_intra(
        k=k,
        v=v,
        g=g,
        beta=beta,
        cu_seqlens=seg_cu,
        chunk_indices=chunk_indices,
    )

    # Compute affine pairs (b, M) for each segment — [2*N, H, K, V] / [2*N, H, K, K]
    b = compute_br(k=k, w=w, u=u, g=g, cu_seqlens=seg_cu)
    M = compute_M_total(k=k, w=w, g=g, cu_seqlens=seg_cu)

    b0 = b[0::2].contiguous()
    b1 = b[1::2].contiguous()
    M0 = M[0::2].contiguous()
    M1 = M[1::2].contiguous()

    N = cu_seqlens.shape[0] - 1
    H = w.shape[2]
    K = k.shape[3]
    V = v.shape[-1]

    # ---- Phase 2: all-gather affine pairs ----
    packed = torch.stack(
        [
            torch.cat([b0, M0], dim=-1),
            torch.cat([b1, M1], dim=-1),
        ],
        dim=0,
    )  # [2, N, H, K, V+K]

    gathered = torch.empty(
        num_segs, *packed.shape[1:], device=packed.device, dtype=packed.dtype
    )
    dist.all_gather_into_tensor(
        gathered.view(num_segs, -1),
        packed.view(2, -1),
        group=cp_group,
    )

    if causal_order is None:
        causal_order = torch.tensor(
            zigzag_causal_order(cp_size), dtype=torch.long, device=packed.device
        )

    # ---- Phase 3: cp_merge to get h0 for each segment ----
    # initial_state is [N, H, K, V] — used directly by cp_merge (same layout)
    h0_global = initial_state.float() if initial_state is not None else None
    seg0_pos, seg1_pos = causal_positions(rank, cp_size)

    h0_seg0 = cp_merge(gathered, h0_global, seg0_pos, N, H, K, V, causal_order)
    h0_seg1 = cp_merge(gathered, h0_global, seg1_pos, N, H, K, V, causal_order)

    # ---- Phase 4: rerun fwd_h + fwd_o with correct h0 ----
    # Interleave h0_seg0 and h0_seg1 to match seg_cu layout
    h0_combined_kv = torch.empty(2 * N, H, K, V, dtype=torch.float32, device=k.device)
    h0_combined_kv[0::2] = h0_seg0
    h0_combined_kv[1::2] = h0_seg1

    # sglang's chunk_gated_delta_rule_fwd_h expects [pool_size, H, V, K] layout
    h0_combined_vk = h0_combined_kv.transpose(-1, -2).contiguous()
    seg_indices = torch.arange(2 * N, dtype=torch.int32, device=k.device)

    h_all, v_new_all = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=h0_combined_vk,
        initial_state_indices=seg_indices,
        cu_seqlens=seg_cu,
        chunk_indices=chunk_indices,
    )
    _fwd_o = fwd_o_fn if fwd_o_fn is not None else chunk_fwd_o
    o = _fwd_o(
        q=q,
        k=k,
        v=v_new_all,
        h=h_all,
        g=g,
        scale=scale,
        cu_seqlens=seg_cu,
    )

    # final_state in [N, H, K, V] layout
    final_state = (
        cp_merge(gathered, h0_global, num_segs, N, H, K, V, causal_order)
        if output_final_state
        else None
    )

    return o, h_all, final_state, v_new_all
