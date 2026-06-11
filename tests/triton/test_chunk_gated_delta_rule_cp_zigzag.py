# -*- coding: utf-8 -*-
"""
Test chunk_gated_delta_rule CP zigzag correctness.

Usage:
    GPU_COUNT=2 python test_chunk_gated_delta_rule_cp_zigzag.py --seqlens 4096 --H 32 --Hg 8 --K 128 --V 128
"""

import argparse
import logging
import multiprocessing as mp
import os
import socket
from typing import List

import torch
import torch.nn.functional as F
from torch.profiler import (
    ProfilerActivity,
    profile,
    schedule,
    tensorboard_trace_handler,
)
from triton.testing import do_bench

from sglang.srt.layers.attention.fla.index import prepare_chunk_indices
import torch.distributed as dist

from sglang.srt.layers.attention.fla.chunk_delta_h import (
    chunk_gated_delta_rule_fwd_h,
)
from sglang.srt.layers.attention.fla.chunk_fwd import (
    chunk_gated_delta_rule_fwd_intra,
)

from sglang.srt.layers.attention.fla.cumsum import chunk_local_cumsum

from sglang.srt.layers.attention.fla.chunk_o import chunk_fwd_o

# from sglang.srt.layers.attention.fla.cp.chunk_cp_zigzag import (
#     chunk_gated_delta_rule_fwd_cp_zigzag,
# )
from pyhip.contrib.triton.chunk_gated_delta_rule_cp_zigzag import (
    chunk_gated_delta_rule_fwd_cp_zigzag,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")

GPU_COUNT = int(os.environ.get("GPU_COUNT", "2"))
# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

# ---------------------------------------------------------------------------
# Helpers: zigzag layout (ported from rtp-llm cp/utils.py)
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


def _zigzag_seg_starts(seq_len: int, cp_size: int, rank: int) -> tuple:
    """Return (seg0_start, seg1_start, half) physical offsets in the full
    sequence for this rank's two halves."""
    half = seq_len // (2 * cp_size)
    seg0_start = rank * half
    seg1_start = seq_len - (rank + 1) * half
    return seg0_start, seg1_start, half


def _build_local_from_full(
    full: torch.Tensor, seq_lengths: List[int], cp_size: int, rank: int
) -> torch.Tensor:
    """Slice each sequence's zigzag halves out of `full` (shape [1, T_total, ...])
    and concat them in [seg0_seq0, seg1_seq0, seg0_seq1, seg1_seq1, ...] order."""
    parts = []
    offset = 0
    for sl in seq_lengths:
        s0, s1, half = _zigzag_seg_starts(sl, cp_size, rank)
        parts.append(full[:, offset + s0 : offset + s0 + half])
        parts.append(full[:, offset + s1 : offset + s1 + half])
        offset += sl
    return torch.cat(parts, dim=1).contiguous()


def _find_free_port() -> int:
    """Find a free TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _bench_synced(
    fn,
    *,
    warmup: int = 10,
    rep: int = 50,
    flush_mb: int = 512,
    use_barrier: bool = False,
):
    """Like triton.testing.do_bench but with FIXED iteration counts so multiple
    ranks running collective ops stay in lockstep.

    - L2 flush via a `flush_mb` MB buffer zeroed before every timed iter.
    - `use_barrier=True` adds a dist.barrier() before each timed iter so the
      two ranks start the same iteration at (close to) the same wall time.
    Returns (p20, p50, p80) in milliseconds.
    """
    flush_buf = torch.empty(
        flush_mb * 1024 * 1024 // 4, dtype=torch.int32, device="cuda"
    )

    # warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    if use_barrier and dist.is_initialized():
        dist.barrier()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    for i in range(rep):
        flush_buf.zero_()
        if use_barrier and dist.is_initialized():
            dist.barrier()
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()

    times = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    n = len(times)
    return times[int(0.2 * n)], times[n // 2], times[int(0.8 * n)]


# Reference: single-GPU with deterministic chunk_fwd_o_ref
def chunk_fwd_o_ref(q, k, v, h, g=None, scale=None, cu_seqlens=None, chunk_size=64):
    """Deterministic PyTorch reference for chunk_fwd_o.

    Computes the same formula as the Triton kernel but in fp32 for
    reproducibility.  This avoids the non-determinism observed in the
    Triton ``make_block_ptr`` implementation on certain Triton versions.
    """
    B, T, Hg, K = q.shape
    H, V = v.shape[-2], v.shape[-1]
    BT = min(chunk_size, max(16, 2 ** (T - 1).bit_length()))  # next_power_of_2
    BT = min(BT, chunk_size)
    if scale is None:
        scale = K**-0.5

    ci = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = len(ci) if ci is not None else (T + BT - 1) // BT
    o = torch.zeros(B, T, H, V, dtype=torch.float32, device=q.device)
    repeat = H // Hg

    for idx in range(NT):
        if ci is not None:
            i_n = ci[idx, 0].item()
            i_t = ci[idx, 1].item()
            bos = cu_seqlens[i_n].item()
            eos = cu_seqlens[i_n + 1].item()
        else:
            i_t = idx
            bos, eos = 0, T
        start = bos + i_t * BT
        end = min(start + BT, eos)
        CL = end - start

        for b in range(B):
            q_c = q[b, start:end].float().permute(1, 0, 2)  # [Hg, CL, K]
            k_c = k[b, start:end].float().permute(1, 0, 2)  # [Hg, CL, K]
            v_c = v[b, start:end].float().permute(1, 0, 2)  # [H, CL, V]
            h_c = h[b, idx].float()  # [H, V, K]

            if repeat > 1:
                q_c = q_c.repeat_interleave(repeat, dim=0)
                k_c = k_c.repeat_interleave(repeat, dim=0)

            # Inter-chunk: [H, CL, K] @ [H, K, V] -> [H, CL, V]
            o_inter = torch.bmm(q_c, h_c.transpose(-1, -2))

            # Intra-chunk: [H, CL, CL]
            A = torch.bmm(q_c, k_c.transpose(-1, -2))

            if g is not None:
                g_c = g[b, start:end].float().permute(1, 0)  # [H, CL]
                o_inter = o_inter * torch.exp(g_c).unsqueeze(-1)
                # g_diff[h, i, j] = g_c[h, i] - g_c[h, j]; for causal (i>=j)
                # with decreasing cumsum this is <=0 → exp gives the decay.
                g_diff = g_c.unsqueeze(2) - g_c.unsqueeze(1)  # [H, CL, CL]
                A = A * torch.where(
                    g_diff <= 0, torch.exp(g_diff), torch.zeros_like(g_diff)
                )

            mask = torch.tril(torch.ones(CL, CL, device=q.device))
            A = A * mask.unsqueeze(0)
            o_c = torch.bmm(A, v_c)
            o[b, start:end] = ((o_inter + o_c) * scale).permute(1, 0, 2)

    return o.to(q.dtype)


def sglang_chunk_gated_delta_rule_fwd(
    q,
    k,
    v,
    g,
    beta,
    scale,
    initial_state,
    cu_seqlens,
    chunk_indices,
    initial_state_indices,
    use_ref_o=False,
):

    g_ref = chunk_local_cumsum(g=g, chunk_size=64, cu_seqlens=cu_seqlens)
    w_ref, u_ref, _ = chunk_gated_delta_rule_fwd_intra(
        k=k,
        v=v,
        g=g_ref,
        beta=beta,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    h_ref, vn_ref = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w_ref,
        u=u_ref,
        g=g_ref,
        initial_state=initial_state,
        initial_state_indices=initial_state_indices,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )

    func_chunk_fwd_o = chunk_fwd_o_ref if use_ref_o else chunk_fwd_o
    o_ref = func_chunk_fwd_o(
        q=q,
        k=k,
        v=vn_ref,
        h=h_ref,
        g=g_ref,
        scale=scale,
        cu_seqlens=cu_seqlens,
    )
    return o_ref, h_ref, vn_ref, g_ref, w_ref, u_ref


# ---------------------------------------------------------------------------
# Worker: unified (handles both fixed-batch and varlen)
# ---------------------------------------------------------------------------
def prepare_input(rank, world_size, nccl_port, seq_lengths, H, K, V, Hg=None):
    if Hg is None:
        Hg = H
    assert H % Hg == 0, f"H ({H}) must be divisible by Hg ({Hg})"
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(nccl_port)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=world_size,
        device_id=device,
    )
    dtype = torch.bfloat16

    torch.manual_seed(0)
    N = len(seq_lengths)
    T_total = sum(seq_lengths)
    scale = K**-0.5

    q_full = torch.randn(1, T_total, Hg, K, dtype=dtype, device=device)
    k_full = F.normalize(
        torch.randn(1, T_total, Hg, K, dtype=torch.float32, device=device),
        p=2,
        dim=-1,
    ).to(dtype)
    v_full = torch.randn(1, T_total, H, V, dtype=dtype, device=device)
    g_full = F.logsigmoid(torch.rand(1, T_total, H, dtype=dtype, device=device))
    beta_full = torch.rand(1, T_total, H, dtype=dtype, device=device).sigmoid()
    # [B, H, K, V] state
    h0_kv = torch.randn(N, H, K, V, dtype=torch.float32, device=device)
    # transpose to [B,H, V, K] state
    h0_vk = h0_kv.transpose(-1, -2).contiguous()
    full_cu = torch.zeros(N + 1, dtype=torch.long, device=device)  # [batch + 1]

    # [batch + 1]
    for i, sl in enumerate(seq_lengths):
        full_cu[i + 1] = full_cu[i] + sl

    # Build local zigzag inputs
    q_l = _build_local_from_full(q_full, seq_lengths, world_size, rank)
    k_l = _build_local_from_full(k_full, seq_lengths, world_size, rank)
    v_l = _build_local_from_full(v_full, seq_lengths, world_size, rank)
    g_l = _build_local_from_full(g_full, seq_lengths, world_size, rank)
    b_l = _build_local_from_full(beta_full, seq_lengths, world_size, rank)

    local_lengths = [
        sl // world_size for sl in seq_lengths
    ]  # [batch], local length for each batch
    local_cu = torch.zeros(N + 1, dtype=torch.long, device=device)
    # [batch + 1], cumulative lengh for each batch.
    for i, ll in enumerate(local_lengths):
        local_cu[i + 1] = local_cu[i] + ll
    # [2*batch + 1], cumulative length for each half-segment.
    seg_cu = build_segment_cu_seqlens(local_cu)
    causal_order = torch.tensor(
        zigzag_causal_order(world_size),
        dtype=torch.long,
        device=device,
    )
    initial_state_indices = torch.arange(N, dtype=torch.int32, device=device)
    chunk_indices_ref = prepare_chunk_indices(full_cu, 64)
    chunk_indices_l = prepare_chunk_indices(seg_cu, 64)
    return (
        scale,
        q_full,
        k_full,
        v_full,
        g_full,
        q_l,
        k_l,
        v_l,
        g_l,
        b_l,
        beta_full,
        h0_kv,
        h0_vk,
        full_cu,
        local_cu,
        seg_cu,
        causal_order,
        initial_state_indices,
        chunk_indices_ref,
        chunk_indices_l,
        device,
    )


# ---------------------------------------------------------------------------
# Worker: unified (handles both fixed-batch and varlen)
# ---------------------------------------------------------------------------


def acc_check_all_hidden_states(h_all, h_ref, seq_lengths, world_size, rank):
    NT_chunk = 64
    h_state_diff = 0.0
    cp_h_offset = 0
    ref_h_offset = 0
    for sl in seq_lengths:
        s0, s1, half = _zigzag_seg_starts(sl, world_size, rank)
        nt_seg = half // NT_chunk  # chunks per half-segment for this seq
        ref_seg0_start = ref_h_offset + s0 // NT_chunk
        ref_seg1_start = ref_h_offset + s1 // NT_chunk
        diff_h0 = (
            (
                h_all[0, cp_h_offset : cp_h_offset + nt_seg].float()
                - h_ref[0, ref_seg0_start : ref_seg0_start + nt_seg].float()
            )
            .abs()
            .max()
            .item()
        )
        diff_h1 = (
            (
                h_all[0, cp_h_offset + nt_seg : cp_h_offset + 2 * nt_seg].float()
                - h_ref[0, ref_seg1_start : ref_seg1_start + nt_seg].float()
            )
            .abs()
            .max()
            .item()
        )
        # print(f"  rank{rank} h_diff seq sl={sl} seg0={diff_h0:.6f} seg1={diff_h1:.6f}")
        h_state_diff = max(h_state_diff, diff_h0, diff_h1)
        cp_h_offset += 2 * nt_seg
        ref_h_offset += sl // NT_chunk
    # print(
    #     f"################[input checking passed]: rank {rank}  h_state_diff={h_state_diff:.6f} "
    # )
    assert (
        h_state_diff < 1e-3
    ), f"rank {rank} h_state_diff={h_state_diff:.6f} exceeds 1e-3"


def acc_check_v_new(v_z, vn_ref, seq_lengths, world_size, rank):
    # Compare v_new (DeltaV) first: zigzag-local segments vs reference slices.
    v_diff = 0.0
    local_offset, full_offset = 0, 0
    for sl in seq_lengths:
        s0, s1, half = _zigzag_seg_starts(sl, world_size, rank)
        dv0 = (
            (
                v_z[:, local_offset : local_offset + half].float()
                - vn_ref[:, full_offset + s0 : full_offset + s0 + half].float()
            )
            .abs()
            .max()
            .item()
        )
        dv1 = (
            (
                v_z[:, local_offset + half : local_offset + 2 * half].float()
                - vn_ref[:, full_offset + s1 : full_offset + s1 + half].float()
            )
            .abs()
            .max()
            .item()
        )
        v_diff = max(v_diff, dv0, dv1)
        local_offset += 2 * half
        full_offset += sl
    # print(
    #     f"################[input checking passed]: rank {rank} v_diff={v_diff:.6f}  "
    # )
    assert v_diff < 1e-3, f"rank {rank} v_diff={v_diff:.6f} exceeds 1e-3"


def acc_check_qkvg(
    q_l,
    k_l,
    v_l,
    g_l,
    q_full,
    k_full,
    v_full,
    g_full,
    seq_lengths,
    world_size,
    rank,
    seg_cu,
    g_ref,
):
    # ---- Extra: verify q/k/g bit-equality at corresponding physical positions ----
    # The CP function reassigns g via chunk_local_cumsum internally; replicate it here.
    g_l_cumsum = chunk_local_cumsum(g_l, chunk_size=64, cu_seqlens=seg_cu)
    q_diff = k_diff = g_diff = 0.0
    local_offset, full_offset = 0, 0
    for sl in seq_lengths:
        s0, s1, half = _zigzag_seg_starts(sl, world_size, rank)
        for seg_idx, src_start in enumerate([s0, s1]):
            lo = local_offset + seg_idx * half
            fo = full_offset + src_start
            q_diff = max(
                q_diff,
                (q_l[:, lo : lo + half].float() - q_full[:, fo : fo + half].float())
                .abs()
                .max()
                .item(),
            )
            k_diff = max(
                k_diff,
                (k_l[:, lo : lo + half].float() - k_full[:, fo : fo + half].float())
                .abs()
                .max()
                .item(),
            )
            g_diff = max(
                g_diff,
                (
                    g_l_cumsum[:, lo : lo + half].float()
                    - g_ref[:, fo : fo + half].float()
                )
                .abs()
                .max()
                .item(),
            )
        local_offset += 2 * half
        full_offset += sl
    # print(
    #     f"  rank{rank} q_diff={q_diff:.6e} k_diff={k_diff:.6e} g_diff(cumsumed)={g_diff:.6e}"
    # )
    assert q_diff == 0.0, f"rank {rank} q_diff={q_diff} not bit-equal"
    assert k_diff == 0.0, f"rank {rank} k_diff={k_diff} not bit-equal"
    assert g_diff == 0.0, f"rank {rank} g_diff={g_diff} not bit-equal"


def acc_worker(rank, world_size, nccl_port, seq_lengths, H, K, V, Hg=None):
    try:
        (
            scale,
            q_full,
            k_full,
            v_full,
            g_full,
            q_l,
            k_l,
            v_l,
            g_l,
            b_l,
            beta_full,
            h0_kv,
            h0_vk,
            full_cu,
            local_cu,
            seg_cu,
            causal_order,
            initial_state_indices,
            chunk_indices_ref,
            chunk_indices_l,
            device,
        ) = prepare_input(rank, world_size, nccl_port, seq_lengths, H, K, V, Hg)

        use_ref_o_func = False
        o_ref, h_ref, vn_ref, g_ref, w_ref, u_ref = sglang_chunk_gated_delta_rule_fwd(
            q=q_full,
            k=k_full,
            v=v_full,
            g=g_full,
            beta=beta_full,
            scale=scale,
            # h0_vk would be inplace updated as final state
            initial_state=h0_vk,
            cu_seqlens=full_cu,
            chunk_indices=chunk_indices_ref,
            initial_state_indices=initial_state_indices,
            use_ref_o=use_ref_o_func,
        )
        fs_ref = h0_vk.transpose(-1, -2).contiguous()
        # CP zigzag
        if use_ref_o_func:
            o_z, h_all, fs_z, v_z = chunk_gated_delta_rule_fwd_cp_zigzag(
                q=q_l,
                k=k_l,
                v=v_l,
                g=g_l,
                beta=b_l,
                scale=scale,
                initial_state=h0_kv,
                output_final_state=True,
                cp_group=dist.group.WORLD,
                cu_seqlens=local_cu,
                seg_cu=seg_cu,
                causal_order=causal_order,
                chunk_indices=chunk_indices_l,
                fwd_o_fn=chunk_fwd_o_ref,
            )
        else:
            o_z, h_all, fs_z, v_z = chunk_gated_delta_rule_fwd_cp_zigzag(
                q=q_l,
                k=k_l,
                v=v_l,
                g=g_l,
                beta=b_l,
                scale=scale,
                initial_state=h0_kv,
                output_final_state=True,
                cp_group=dist.group.WORLD,
                cu_seqlens=local_cu,
                seg_cu=seg_cu,
                causal_order=causal_order,
                fwd_o_fn=chunk_fwd_o,
            )

        # Debug: compare h_all per chunk
        # CP h_all layout: [seq0_seg0, seq0_seg1, seq1_seg0, seq1_seg1, ...]
        # where each entry has NT_seg_i = (sl_i // (2*cp_size)) // 64 chunks.
        # Ref h_ref layout: [seq0_chunks, seq1_chunks, ...] with NT_i = sl_i//64 chunks.

        acc_check_all_hidden_states(h_all, h_ref, seq_lengths, world_size, rank)
        acc_check_v_new(v_z, vn_ref, seq_lengths, world_size, rank)
        acc_check_qkvg(
            q_l,
            k_l,
            v_l,
            g_l,
            q_full,
            k_full,
            v_full,
            g_full,
            seq_lengths,
            world_size,
            rank,
            seg_cu,
            g_ref,
        )
        # Compare output and final hidden state:
        o_diff = 0.0
        local_offset, full_offset = 0, 0
        for sl in seq_lengths:
            s0, s1, half = _zigzag_seg_starts(sl, world_size, rank)
            d0 = (
                (
                    o_z[:, local_offset : local_offset + half].float()
                    - o_ref[:, full_offset + s0 : full_offset + s0 + half].float()
                )
                .abs()
                .max()
                .item()
            )
            d1 = (
                (
                    o_z[:, local_offset + half : local_offset + 2 * half].float()
                    - o_ref[:, full_offset + s1 : full_offset + s1 + half].float()
                )
                .abs()
                .max()
                .item()
            )
            o_diff = max(o_diff, d0, d1)
            print(f"  rank{rank} o_diff seg0={d0:.6f} seg1={d1:.6f}")
            local_offset += 2 * half
            full_offset += sl

        fs_diff = (
            (fs_z.float() - fs_ref.float()).abs().max().item()
            if fs_z is not None
            else 0.0
        )
        passed = max(o_diff, fs_diff) < 1e-2

        dist.barrier()
        logging.info(
            f"  rank {rank}: o={o_diff:.6f} fs={fs_diff:.6f} {'PASS' if passed else 'FAIL'}"
        )
        dist.barrier()
        torch.cuda.synchronize()
        dist.destroy_process_group()

        assert passed, f"rank {rank} failed: o={o_diff:.6f} fs={fs_diff:.6f}"

    except Exception as e:
        print(f"Rank {rank} error: {e}")
        import traceback

        dist.barrier()
        torch.cuda.synchronize()
        dist.destroy_process_group()
        traceback.print_exc()
        raise


def bench_worker(
    rank, world_size, nccl_port, seq_lengths, H, K, V, Hg=None, profile_trace=False
):
    try:
        (
            scale,
            q_full,
            k_full,
            v_full,
            g_full,
            q_l,
            k_l,
            v_l,
            g_l,
            b_l,
            beta_full,
            h0_kv,
            h0_vk,
            full_cu,
            local_cu,
            seg_cu,
            causal_order,
            initial_state_indices,
            chunk_indices_ref,
            chunk_indices_l,
            device,
        ) = prepare_input(rank, world_size, nccl_port, seq_lengths, H, K, V, Hg)

        def _ref_call():
            sglang_chunk_gated_delta_rule_fwd(
                q=q_full,
                k=k_full,
                v=v_full,
                g=g_full,
                beta=beta_full,
                scale=scale,
                initial_state=h0_vk,
                cu_seqlens=full_cu,
                chunk_indices=chunk_indices_ref,
                initial_state_indices=initial_state_indices,
            )

        def _cp_call():
            chunk_gated_delta_rule_fwd_cp_zigzag(
                q=q_l,
                k=k_l,
                v=v_l,
                g=g_l,  # cumsum is in-place inside
                beta=b_l,
                scale=scale,
                initial_state=h0_kv,
                output_final_state=True,
                cp_group=dist.group.WORLD,
                cu_seqlens=local_cu,
                seg_cu=seg_cu,
                causal_order=causal_order,
                chunk_indices=chunk_indices_l,
                fwd_o_fn=chunk_fwd_o,
            )

        # # Only rank 0 runs single-GPU reference (no collectives).
        # if rank == 0:
        #     ref_p20, ref_p50, ref_p80 = _bench_synced(
        #         _ref_call, warmup=2, rep=100, use_barrier=False
        #     )
        # else:
        #     ref_p20 = ref_p50 = ref_p80 = float("nan")

        dist.barrier()
        # CP path runs collectives -> all ranks must use identical iter counts
        # AND barrier between iters, otherwise NCCL desyncs and TCPStore dies.
        cp_p20, cp_p50, cp_p80 = _bench_synced(
            _cp_call, warmup=2, rep=100, use_barrier=True
        )
        dist.barrier()

        label = "+".join(str(s) for s in seq_lengths)
        # if rank == 0:
        #     logging.info(
        #         f"  [bench seqlens={label} H={H} K={K} V={V}] "
        #         f"ref(1GPU) p50={ref_p50*1000:.1f}us (p20/p80={ref_p20*1000:.1f}/{ref_p80*1000:.1f})"
        #     )
        logging.info(
            f"  [bench seqlens={label} cp_size={world_size} rank={rank}] "
            f"cp p50={cp_p50*1000:.1f}us (p20/p80={cp_p20*1000:.1f}/{cp_p80*1000:.1f})"
        )
        dist.barrier()

        # ------------------------------------------------------------------
        # Optional torch.profiler pass: dump per-rank Chrome trace + print
        # per-kernel CUDA-time table.  Active iters are kept small to avoid
        # huge traces; this is independent of the timing bench above.
        # ------------------------------------------------------------------
        if profile_trace:
            trace_dir = os.environ.get(
                "PROFILE_DIR",
                f"./traces_seqlens{label}_cp{world_size}",
            )
            os.makedirs(trace_dir, exist_ok=True)
            sched = schedule(wait=1, warmup=2, active=5, repeat=1)
            n_steps = 1 + 2 + 5  # wait + warmup + active

            # ---- CP profile (all ranks) ----
            dist.barrier()
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=sched,
                on_trace_ready=tensorboard_trace_handler(
                    trace_dir, worker_name=f"cp_rank{rank}"
                ),
                record_shapes=False,
                with_stack=False,
            ) as prof_cp:
                for _ in range(n_steps):
                    dist.barrier()
                    _cp_call()
                    prof_cp.step()
            torch.cuda.synchronize()
            dist.barrier()

            if rank == 0:
                logging.info(
                    f"\n[profile cp seqlens={label} cp_size={world_size} rank=0] "
                    f"top kernels:\n"
                    + prof_cp.key_averages().table(
                        sort_by="cuda_time_total", row_limit=20
                    )
                )

            # ---- Reference profile (rank 0 only, no collectives) ----
            if rank == 0:
                with profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    schedule=sched,
                    on_trace_ready=tensorboard_trace_handler(
                        trace_dir, worker_name="ref_rank0"
                    ),
                    record_shapes=False,
                    with_stack=False,
                ) as prof_ref:
                    for _ in range(n_steps):
                        _ref_call()
                        prof_ref.step()
                torch.cuda.synchronize()
                logging.info(
                    f"\n[profile ref(1GPU) seqlens={label}] top kernels:\n"
                    + prof_ref.key_averages().table(
                        sort_by="cuda_time_total", row_limit=20
                    )
                )
                logging.info(f"\n[profile] traces written to: {trace_dir}")
            dist.barrier()

        torch.cuda.synchronize()
        dist.destroy_process_group()

    except Exception as e:
        print(f"Rank {rank} error: {e}")
        import traceback

        dist.barrier()
        torch.cuda.synchronize()
        dist.destroy_process_group()
        traceback.print_exc()
        raise


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_test(
    seq_lengths, H=16, K=128, V=128, Hg=None, bench=False, profile_trace=False
):
    nccl_port = _find_free_port()
    label = "+".join(str(s) for s in seq_lengths)
    hg_str = f" Hg={Hg}" if Hg is not None and Hg != H else ""
    logging.info(f"[seqlens={label}  H={H}{hg_str} K={K} V={V}]")
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    processes = []
    for rank in range(GPU_COUNT):
        p = mp.Process(
            target=bench_worker if bench else acc_worker,
            args=(rank, GPU_COUNT, nccl_port, seq_lengths, H, K, V, Hg)
            + ((profile_trace,) if bench else ()),
            name=f"rank-{rank}",
        )
        p.start()
        processes.append(p)
    failed = False
    for p in processes:
        p.join(timeout=300)
        if p.exitcode != 0:
            logging.error(f"  {p.name} exited with code {p.exitcode}")
            failed = True
    return not failed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test CP zigzag correctness")
    parser.add_argument(
        "--seqlens",
        type=int,
        nargs="+",
        default=None,
        help="Sequence lengths (e.g. --seqlens 4096 8192)",
    )
    parser.add_argument(
        "--H", type=int, default=64, help="Number of V heads (num_v_heads)"
    )
    parser.add_argument(
        "--Hg",
        type=int,
        default=None,
        help="Number of Q/K heads (num_qk_heads). Defaults to H if not set.",
    )
    parser.add_argument("--K", type=int, default=128)
    parser.add_argument("--V", type=int, default=128)
    parser.add_argument(
        "--bench",
        action="store_true",
        help="After correctness check, run do_bench on the reference and CP zigzag forward.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="With --bench, also dump per-rank torch.profiler Chrome traces and per-kernel CUDA-time tables. Set PROFILE_DIR to override the output dir.",
    )
    args = parser.parse_args()

    if args.seqlens:
        # Run single user-specified config
        ok = run_test(
            args.seqlens,
            H=args.H,
            K=args.K,
            V=args.V,
            Hg=args.Hg,
            bench=args.bench,
            profile_trace=args.profile,
        )
        exit(0 if ok else 1)

    # Default: run a suite of configs
    # Each config: (seq_lengths, H, Hg) — Hg=None means Hg==H
    configs = [
        # Hg == H (no GQA)
        # ([256], args.H, None),
        # ([512], args.H, None),
        # ([4096], args.H, None),
        # Hg < H (GQA-like, matching Qwen3.5 head ratio)
        ([8192], 64, 16),
        ([64000], 64, 16),
        # ([4096, 8192], 32, 8),
    ]
    results = []
    for sl, H, Hg in configs:
        ok = run_test(
            sl,
            H=H,
            K=args.K,
            V=args.V,
            Hg=Hg,
            bench=args.bench,
            profile_trace=args.profile,
        )
        results.append((sl, H, Hg, ok))

    logging.info("\n===== Summary =====")
    all_pass = True
    for sl, H, Hg, ok in results:
        label = "+".join(str(s) for s in sl)
        hg_str = f" Hg={Hg}" if Hg is not None and Hg != H else ""
        logging.info(f"  {label} H={H}{hg_str}: {'PASS' if ok else 'FAIL'}")
        if not ok:
            all_pass = False
    exit(0 if all_pass else 1)
