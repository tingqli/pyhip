# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Weight preparation (weight-static: run once per weight at load time).

Input = the standard aiter mxfp4 MoE preshuffle (what aiter's a16w4 / a8w4 paths
consume):
  w1       [E, 2I, H/2]  fp4x2 / uint8, rows [gate (I); up (I)], preshuffled with
                         ``shuffle_weight_a16w4(w1, 16, False)``
  w2       [E, H, I/2]   fp4x2 / uint8, ``shuffle_weight_a16w4(w2, 16, False)``
  w*_scale e8m0 per 1x32 K group, ``shuffle_scale_a16w4(s, E, False)``
``shuffle_mxfp4`` does that from unshuffled checkpoint tensors.

Output (one dict for every kernel, TP and EP): the "fd2" fast-decode form
  w1, w2            nibble order k0..k7 -> (k_j | k_{j+4} << 4) per dword, -0 -> +0,
                    W of groups that decode to 0 (scale <= channel max - 12) zeroed
  w1_scale          per 32-K group: min(channel max code - group code, 11) * 8
  w2_scale          same, followed by W2's per-channel max codes (int32 [E*H]);
                    the TP gemm2 / fused kernels read them at that offset
  w1_exp, w2_exp    per-output-channel max e8m0 code (int32 [E, 2I] / [E, H]);
                    ``w2_exp`` is a view into the tail of ``w2_scale`` (no copy)
  w1_fast_dec, w2_fast_dec = 2
Bit-exact with decoding the original weights (``doc-mimo`` README section 7).
"""

import torch

from ._vendor.a8w4_ops import channel_exponents
from .kernels.tp_gemm1 import prep_w1_fast

FAST_DEC = 2


def prepare_weights(w1, w2, w1_scale, w2_scale):
    """Preshuffled mxfp4 W1 / W2 + preshuffled e8m0 scales -> the weight dict used by
    ``a8w4_moe`` (see the module docstring). Shapes: w1 [E, 2I, H/2], w2 [E, H, I/2]."""
    E, n2, hk = w1.shape
    assert w2.shape[0] == E and w2.shape[1] == hk * 2 and w2.shape[2] * 2 == n2 // 2, (
        w1.shape, w2.shape)
    out = dict(w1_fast_dec=FAST_DEC, w2_fast_dec=FAST_DEC)
    for i, (w, s) in ((1, (w1, w1_scale)), (2, (w2, w2_scale))):
        w_u8 = w.contiguous().view(torch.uint8)
        s_u8 = s.view(torch.uint8).contiguous().view(-1)
        exp = channel_exponents(w_u8, s_u8)  # from the ORIGINAL scales
        wf, sf = prep_w1_fast(w_u8, s_u8, exp, FAST_DEC)
        out[f"w{i}"] = wf
        if i == 2:
            n = sf.numel()
            assert n % 4 == 0
            sf = torch.cat([sf, exp.contiguous().view(-1).view(torch.uint8)])
            exp = sf[n:].view(torch.int32).view(exp.shape)
        out[f"w{i}_scale"] = sf
        out[f"w{i}_exp"] = exp
    return out


def shuffle_mxfp4(w1q, w1s, w2q, w2s):
    """Unshuffled mxfp4 checkpoint tensors -> the preshuffled inputs of
    ``prepare_weights`` (uses aiter's shuffles).
    w1q [E, 2I, H/2] u8, w1s [E, 2I, H/32] e8m0, w2q [E, H, I/2], w2s [E, H, I/32]."""
    from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4

    E = w1q.shape[0]
    return (shuffle_weight_a16w4(w1q, 16, False),
            shuffle_weight_a16w4(w2q, 16, False),
            shuffle_scale_a16w4(w1s.reshape(-1, w1s.shape[-1]), E, False),
            shuffle_scale_a16w4(w2s.reshape(-1, w2s.shape[-1]), E, False))


def weight_shape(W):
    """(experts, model_dim H, inter_dim I) of a prepared weight dict."""
    E, H, ik = W["w2"].shape
    return E, H, ik * 2
