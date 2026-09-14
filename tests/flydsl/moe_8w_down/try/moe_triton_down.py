# SPDX-License-Identifier: MIT
"""Alternative tiled MFMA down on the same OCP FP8 and M256 packed ABI."""

import torch
import triton as tr
import triton.language as tl


@tr.jit
def _down(A, B, SA, SB, IDS, ROUTES, EIDS, VALID, OUT,
          TOKENS: tl.constexpr, TOPK: tl.constexpr, N: tl.constexpr,
          CAPACITY: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
          XCD: tl.constexpr, N_GROUP: tl.constexpr):
    pid = tl.program_id(0)
    m_parts: tl.constexpr = 256 // BM
    n_tiles: tl.constexpr = N // BN
    n_tasks: tl.constexpr = n_tiles // N_GROUP
    total = tl.load(VALID) // 256 * m_parts * n_tasks
    if pid < total:
        chunk = total // XCD
        task = tl.where(pid < chunk * XCD, (pid % XCD) * chunk + pid // XCD, pid)
        mt = task // n_tasks
        parent, part = mt // m_parts, mt % m_parts
        row = parent * 256 + part * BM + tl.arange(0, BM)
        encoded = tl.load(IDS + row).to(tl.uint32)
        token, slot = encoded & 0xFFFFFF, encoded >> 24
        live = (token < TOKENS) & (slot < TOPK)
        if tl.sum(live.to(tl.int32), 0) > 0:
            expert = tl.load(EIDS + parent)
            source = token.to(tl.int32) * TOPK + slot.to(tl.int32)
            kk = tl.arange(0, 128)
            a0 = tl.load(A + source[:, None] * 256 + kk[None, :], live[:, None], 0.0)
            a1 = tl.load(A + source[:, None] * 256 + 128 + kk[None, :], live[:, None], 0.0)
            sa0 = tl.load(SA + source, live, 0)
            sa1 = tl.load(SA + TOKENS * TOPK + source, live, 0)
            routing = tl.load(ROUTES + row)
            for turn in range(N_GROUP):
                nn = (task % n_tasks * N_GROUP + turn) * BN + tl.arange(0, BN)
                # shuffle_weight((16,16)): [N16,K32,Ksub2,n16,k16].
                woff = expert * N * 256 + nn[None, :] // 16 * 4096 + kk[:, None] // 16 * 256 + nn[None, :] % 16 * 16 + kk[:, None] % 16
                b0 = tl.load(B + woff)
                b1 = tl.load(B + woff + 2048)
                p0 = tl.dot(a0, b0)
                p1 = tl.dot(a1, b1)
                sb0 = tl.load(SB + expert * (N // 128 * 2) + nn // 128 * 2)
                sb1 = tl.load(SB + expert * (N // 128 * 2) + nn // 128 * 2 + 1)
                c0 = p0 * (sa0[:, None] * sb0[None, :])
                value = tl.fma(p1, sa1[:, None] * sb1[None, :], c0)
                value = (value * routing[:, None]).to(tl.bfloat16)
                dst = parent * (256 * N) + nn[None, :] // 64 * (256 * 64) + (part * BM + tl.arange(0, BM))[:, None] * 64 + nn[None, :] % 64
                tl.store(OUT + dst, value, live[:, None], cache_modifier=".wt")


def make_triton_down(*, n, k, topk, num_experts, output_layout="packed", bm=64, bn=128,
                    xcd_count=8, n_group=1, num_warps=4, matrix_instr_nonkdim=16, num_stages=2,
                    num_oc_splits=4):
    assert k == 256 and n % (bn * n_group) == 0 and 256 % bm == 0
    assert output_layout == "packed" and 0 < topk <= min(num_experts, 255)

    def down(output, a, b, sa, sb, ids, routes, eids, valid, counter):
        assert a.shape[1:] == (topk, 256) and b.shape == (num_experts, n, 256)
        assert a.dtype == b.dtype == torch.float8_e4m3fn and output.dtype == torch.bfloat16
        assert output.shape == (eids.numel() * 256, n)
        assert all(t.is_cuda and t.is_contiguous() and t.device == a.device for t in (output, a, b, sa, sb, ids, routes, eids, valid, counter))
        _down[(eids.numel() * (256 // bm) * (n // bn // n_group),)](
            a, b, sa, sb, ids, routes, eids, valid, output,
            a.shape[0], topk, n, eids.numel(), bm, bn, xcd_count, n_group,
            num_warps=num_warps, num_stages=num_stages, matrix_instr_nonkdim=matrix_instr_nonkdim,
            enable_fp_fusion=False,
        )
        return output
    down.config = {"backend": "triton", "block_m": bm, "block_n": bn,
                   "num_waves": num_warps, "n_group": n_group, "matrix_instr_nonkdim": matrix_instr_nonkdim,
                   "num_stages": num_stages, "num_oc_splits": num_oc_splits,
                   "persistent": False, "xcd_count": xcd_count, "skip_empty_tasks": True,
                   "output_layout": "packed", "weight_layout": (16, 16)}
    return down