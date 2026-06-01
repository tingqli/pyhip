# -*- coding: utf-8 -*-
"""
Test chunk_gated_delta_rule CP zigzag correctness.

Usage:
    GPU_COUNT=2 python test_chunk_gated_delta_rule_cp_zigzag.py
    GPU_COUNT=2 python test_chunk_gated_delta_rule_cp_zigzag.py --seqlens 4096 8192
    GPU_COUNT=2 python test_chunk_gated_delta_rule_cp_zigzag.py --seqlens 256 --H 8 --K 64 --V 64
"""

import argparse
import logging
import multiprocessing as mp
import os
import socket
from typing import List

import torch
import torch.nn.functional as F

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


# ---------------------------------------------------------------------------
# Worker: unified (handles both fixed-batch and varlen)
# ---------------------------------------------------------------------------


def _worker(rank, world_size, nccl_port, seq_lengths, H, K, V):
    try:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(nccl_port)
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
        dtype = torch.bfloat16

        torch.manual_seed(0)
        N = len(seq_lengths)
        T_total = sum(seq_lengths)
        scale = K**-0.5

        q_full = torch.randn(1, T_total, H, K, dtype=dtype, device=device)
        k_full = F.normalize(
            torch.randn(1, T_total, H, K, dtype=torch.float32, device=device),
            p=2,
            dim=-1,
        ).to(dtype)
        v_full = torch.randn(1, T_total, H, V, dtype=dtype, device=device)
        g_full = F.logsigmoid(torch.rand(1, T_total, H, dtype=dtype, device=device))
        beta_full = torch.rand(1, T_total, H, dtype=dtype, device=device).sigmoid()
        h0_kv = torch.randn(N, H, K, V, dtype=torch.float32, device=device)
        h0_vk = h0_kv.transpose(-1, -2).contiguous()

        full_cu = torch.zeros(N + 1, dtype=torch.long, device=device)
        for i, sl in enumerate(seq_lengths):
            full_cu[i + 1] = full_cu[i] + sl

        # Build local zigzag inputs
        q_l = _build_local_from_full(q_full, seq_lengths, world_size, rank)
        k_l = _build_local_from_full(k_full, seq_lengths, world_size, rank)
        v_l = _build_local_from_full(v_full, seq_lengths, world_size, rank)
        g_l = _build_local_from_full(g_full, seq_lengths, world_size, rank)
        b_l = _build_local_from_full(beta_full, seq_lengths, world_size, rank)

        local_lengths = [sl // world_size for sl in seq_lengths]
        local_cu = torch.zeros(N + 1, dtype=torch.long, device=device)
        for i, ll in enumerate(local_lengths):
            local_cu[i + 1] = local_cu[i] + ll
        seg_cu = build_segment_cu_seqlens(local_cu)
        causal_order = torch.tensor(
            zigzag_causal_order(world_size),
            dtype=torch.long,
            device=device,
        )

        # Reference: single-GPU with deterministic chunk_fwd_o_ref
        ref_state = h0_vk.clone()
        initial_state_indices = torch.arange(N, dtype=torch.int32, device=device)
        chunk_indices_ref = prepare_chunk_indices(full_cu, 64)
        g_ref = chunk_local_cumsum(g_full.clone(), chunk_size=64, cu_seqlens=full_cu)
        w_ref, u_ref, _ = chunk_gated_delta_rule_fwd_intra(
            k=k_full.clone(),
            v=v_full.clone(),
            g=g_ref,
            beta=beta_full.clone(),
            cu_seqlens=full_cu,
            chunk_indices=chunk_indices_ref,
        )
        h_ref, vn_ref = chunk_gated_delta_rule_fwd_h(
            k=k_full.clone(),
            w=w_ref,
            u=u_ref,
            g=g_ref,
            initial_state=ref_state,
            initial_state_indices=initial_state_indices,
            cu_seqlens=full_cu,
            chunk_indices=chunk_indices_ref,
        )

        use_ref = 0
        if use_ref:
            o_ref = chunk_fwd_o_ref(
                q=q_full,
                k=k_full,
                v=vn_ref,
                h=h_ref,
                g=g_ref,
                scale=scale,
                cu_seqlens=full_cu,
            )
        else:
            o_ref = chunk_fwd_o(
                q=q_full,
                k=k_full,
                v=vn_ref,
                h=h_ref,
                g=g_ref,
                scale=scale,
                cu_seqlens=full_cu,
            )
        fs_ref = ref_state.transpose(-1, -2).contiguous()

        # CP zigzag
        if use_ref:
            o_z, h_all, fs_z, v_z = chunk_gated_delta_rule_fwd_cp_zigzag(
                q=q_l.clone(),
                k=k_l.clone(),
                v=v_l.clone(),
                g=g_l.clone(),
                beta=b_l.clone(),
                scale=scale,
                initial_state=h0_kv.clone(),
                output_final_state=True,
                cp_group=dist.group.WORLD,
                cu_seqlens=local_cu,
                seg_cu=seg_cu,
                causal_order=causal_order,
                fwd_o_fn=chunk_fwd_o_ref,
            )
        else:
            o_z, h_all, fs_z, v_z = chunk_gated_delta_rule_fwd_cp_zigzag(
                q=q_l.clone(),
                k=k_l.clone(),
                v=v_l.clone(),
                g=g_l.clone(),
                beta=b_l.clone(),
                scale=scale,
                initial_state=h0_kv.clone(),
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
        assert (
            h_state_diff < 1e-3
        ), f"rank {rank} h_state_diff={h_state_diff:.6f} exceeds 1e-3"

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

        assert v_diff < 1e-3, f"rank {rank} v_diff={v_diff:.6f} exceeds 1e-3"
        print(
            f"################[input checking passed]: rank {rank} v_diff={v_diff:.6f}  h_state_diff={h_state_diff:.6f} "
        )

        # ---- Extra: verify q/k/g bit-equality at corresponding physical positions ----
        # The CP function reassigns g via chunk_local_cumsum internally; replicate it here.
        g_l_cumsum = chunk_local_cumsum(g_l.clone(), chunk_size=64, cu_seqlens=seg_cu)
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
        print(
            f"  rank{rank} q_diff={q_diff:.6e} k_diff={k_diff:.6e} g_diff(cumsumed)={g_diff:.6e}"
        )
        assert q_diff == 0.0, f"rank {rank} q_diff={q_diff} not bit-equal"
        assert k_diff == 0.0, f"rank {rank} k_diff={k_diff} not bit-equal"
        assert g_diff == 0.0, f"rank {rank} g_diff={g_diff} not bit-equal"

        # Compare
        o_diff = 0.0
        local_offset, full_offset = 0, 0
        for sl in seq_lengths:
            s0, s1, half = _zigzag_seg_starts(sl, world_size, rank)
            # cur_out_0 = o_z[:, local_offset : local_offset + half].float()
            # ref_out_0 = o_ref[:, full_offset + s0 : full_offset + s0 + half].float()
            # if not torch.allclose(ref_out_0, cur_out_0, rtol=0.01, atol=0.01):
            #     # print(ref_out)
            #     # print(cur_out)
            #     # print(ref_out[0].tolist())
            #     # print(cur_out[0].tolist())
            #     idx = torch.where(torch.abs(ref_out_0 - cur_out_0) > 0.01)
            #     if len(idx[0]):
            #         print(
            #             f"idx = {idx}\nref={ref_out_0[idx]}\ncur={cur_out_0[idx]}\n{len(idx[0])}"
            #         )
            #     # assert 0
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

        traceback.print_exc()
        raise


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_test(seq_lengths, H=16, K=128, V=128):
    nccl_port = _find_free_port()
    label = "+".join(str(s) for s in seq_lengths)
    logging.info(f"[seqlens={label}  H={H} K={K} V={V}]")
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    processes = []
    for rank in range(GPU_COUNT):
        p = mp.Process(
            target=_worker,
            args=(rank, GPU_COUNT, nccl_port, seq_lengths, H, K, V),
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
    parser.add_argument("--H", type=int, default=64)
    parser.add_argument("--K", type=int, default=128)
    parser.add_argument("--V", type=int, default=128)
    args = parser.parse_args()

    if args.seqlens:
        # Run single user-specified config
        ok = run_test(args.seqlens, H=args.H, K=args.K, V=args.V)
        exit(0 if ok else 1)

    # Default: run a suite of configs
    configs = [
        [256],
        [512],
        [1024],
        [2048],
        [4096],
        [8192],
        [32768],
        [256, 256],
        [4096, 4096],
        [8192, 16384],
        [4096, 8192, 4096],
    ]
    results = []
    for sl in configs:
        ok = run_test(sl, H=args.H, K=args.K, V=args.V)
        results.append((sl, ok))

    logging.info("\n===== Summary =====")
    all_pass = True
    for sl, ok in results:
        label = "+".join(str(s) for s in sl)
        logging.info(f"  {label}: {'PASS' if ok else 'FAIL'}")
        if not ok:
            all_pass = False
    exit(0 if all_pass else 1)
