# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Test / tuning helpers: random weights + routing, fp32 reference, cold-cache timing.

Not needed at inference time. Weight generation uses aiter's mxfp4 quantizer and
preshuffle (``aiter.get_torch_quant`` / ``aiter.ops.shuffle``); everything else is
plain torch.
"""

import torch

from .weights import prepare_weights, shuffle_mxfp4

ACTS = ("silu", "situv2")
# SiTUv2 params used by the tests: the clamp sits inside the gate/up range of the
# random data (std ~64), so clamped and free values both occur.
SITU = dict(situ_beta=4.0, situ_linear_beta=25.0, swiglu_limit=64.0)
ROUTINGS = ("uniform", "hot")
HOT_EXPERT_DIV = 32  # "hot": 1/32 of the experts get a +2 score bias
HOT_EXPERT_BIAS = 2.0


# =============================================================================
# data
# =============================================================================


def make_weights(experts, model_dim, inter_dim, seed=0, device="cuda"):
    """Random bf16 W1 [E, 2I, H] / W2 [E, H, I] -> mxfp4 (per 1x32 e8m0) ->
    (prepared weights, raw dict(w1q, w1s, w2q, w2s) of the unshuffled tensors).
    Quantized one expert at a time to keep the bf16 temporaries small."""
    from aiter import QuantType, dtypes, get_torch_quant

    tq = get_torch_quant(QuantType.per_1x32)
    g = torch.Generator(device=device).manual_seed(seed)
    w1q, w1s, w2q, w2s = [], [], [], []
    for _ in range(experts):
        w1 = torch.randn((1, 2 * inter_dim, model_dim), dtype=torch.bfloat16, device=device,
                         generator=g)
        w2 = torch.randn((1, model_dim, inter_dim), dtype=torch.bfloat16, device=device,
                         generator=g)
        q, s = tq(w1, quant_dtype=dtypes.fp4x2)
        w1q.append(q.view(1, 2 * inter_dim, model_dim // 2))
        w1s.append(s.view(1, 2 * inter_dim, model_dim // 32))
        q, s = tq(w2, quant_dtype=dtypes.fp4x2)
        w2q.append(q.view(1, model_dim, inter_dim // 2))
        w2s.append(s.view(1, model_dim, inter_dim // 32))
    raw = dict(w1q=torch.cat(w1q), w1s=torch.cat(w1s), w2q=torch.cat(w2q), w2s=torch.cat(w2s))
    W = prepare_weights(*shuffle_mxfp4(raw["w1q"], raw["w1s"], raw["w2q"], raw["w2s"]))
    return W, raw


def make_routing(tokens, model_dim, num_experts, topk, seed=0, routing="uniform",
                 device="cuda"):
    """x bf16 [tokens, H] and a renormalized softmax top-k over ``num_experts``
    (global) experts: (x, topk_weights f32 [tokens, topk], topk_ids i32)."""
    g = torch.Generator(device=device).manual_seed(1000 + seed)
    x = torch.randn((tokens, model_dim), dtype=torch.bfloat16, device=device, generator=g)
    score = torch.randn((tokens, num_experts), dtype=torch.float32, device=device, generator=g)
    if routing == "hot":
        hot = torch.randperm(num_experts, device=device, generator=g)[
            : max(1, num_experts // HOT_EXPERT_DIV)]
        score[:, hot] += HOT_EXPERT_BIAS
    else:
        assert routing == "uniform", routing
    w, ids = torch.topk(torch.softmax(score, dim=-1), topk, dim=-1)
    w = w / w.sum(dim=-1, keepdim=True)
    return x, w.float().contiguous(), ids.to(torch.int32).contiguous()


# =============================================================================
# fp32 reference
# =============================================================================


def act_ref(gate, up, act="silu", situ=None):
    """fp32 act(gate) * up; SiTUv2 with the kernels' clamp (gate <= L, |up| <= L)."""
    if act == "silu":
        return torch.nn.functional.silu(gate) * up
    assert act == "situv2", act
    situ = situ or {}
    lim = float(situ.get("swiglu_limit", float("inf")))
    beta = float(situ.get("situ_beta", 1.0))
    lbeta = float(situ.get("situ_linear_beta", 1.0))
    gate, up = gate.clamp(max=lim), up.clamp(min=-lim, max=lim)
    return beta * torch.tanh(gate / beta) * torch.sigmoid(gate) * (lbeta * torch.tanh(up / lbeta))


_FP4 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def _mxfp4_to_f32(q, s):
    """packed e2m1 [..., K/2] (low nibble first) + e8m0 [..., K/32] -> f32 [..., K]."""
    lut = torch.tensor(_FP4 + [-v for v in _FP4], dtype=torch.float32, device=q.device)
    q = q.view(torch.uint8)
    v = torch.stack([lut[(q & 0xF).long()], lut[(q >> 4).long()]], dim=-1).flatten(-2)
    sc = torch.exp2(s.view(torch.uint8).float() - 127.0)
    return (v.view(*v.shape[:-1], -1, 32) * sc.unsqueeze(-1)).flatten(-2)


def _quant_fp8(a, group):
    """fp8-FNUZ round trip with one scale per ``group`` columns (amax / 240)."""
    g = a.view(a.shape[0], -1, group)
    s = g.abs().amax(-1, keepdim=True).clamp(min=1e-30) / 240.0
    return ((g / s).to(torch.float8_e4m3fnuz).float() * s).view(a.shape)


def ref_moe(x, topk_weights, topk_ids, raw, expert_offset=0, act="silu", situ=None):
    """fp32 reference of this rank's partial MoE [tokens, H]: experts
    [expert_offset, expert_offset + E) of the global ``topk_ids``; x and the
    activation output are fp8-quantized as the kernels do (x per row, the
    intermediate per (row, 128 cols)), so the remaining differences are
    accumulation order and the bf16 output."""
    tokens, H = x.shape
    E, n2, _ = raw["w1q"].shape
    inter = n2 // 2
    loc = topk_ids.long() - expert_offset
    loc = torch.where((loc >= 0) & (loc < E), loc, torch.full_like(loc, -1))
    xf = _quant_fp8(x.float(), H)
    out = torch.zeros((tokens, H), dtype=torch.float32, device=x.device)
    for e in loc[loc >= 0].unique().tolist():
        tok, slot = torch.nonzero(loc == e, as_tuple=True)
        w1 = _mxfp4_to_f32(raw["w1q"][e], raw["w1s"][e])
        gate, up = (xf[tok] @ w1.t()).split(inter, dim=-1)
        a = _quant_fp8(act_ref(gate, up, act, situ), 128)
        w2 = _mxfp4_to_f32(raw["w2q"][e], raw["w2s"][e])
        out.index_add_(0, tok, (a @ w2.t()) * topk_weights[tok, slot].view(-1, 1))
    return out


def cos_diff(x, y):
    """1 - cosine similarity (0 = identical direction; 0 when both are all zero, e.g.
    an EP rank that got no routes)."""
    x, y = x.double(), y.double()
    den = (x * x + y * y).sum()
    if float(den) == 0.0:
        return 0.0
    return float(1 - 2 * (x * y).sum() / den)


# =============================================================================
# cold-cache timing
# =============================================================================

_FLUSH = {}


def flush_cache():
    """Read 512 MB: evicts L2 and the 256 MB MALL with clean lines (a read-modify-
    write flush leaves dirty lines whose write-back then steals HBM bandwidth)."""
    if "buf" not in _FLUSH:
        _FLUSH["buf"] = torch.ones(128 << 20, dtype=torch.int32, device="cuda")
    _FLUSH["buf"].amax()


def time_cold(fn, iters=20, warmup=3, flush=True):
    """Median GPU time (us) of ``fn`` with cold caches. A GPU-side sleep after the
    flush lets the host enqueue all of fn's launches before the start event, so
    host overhead is excluded (as in a CUDA graph replay)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
          for _ in range(iters)]
    ts = []
    for e0, e1 in ev:
        if flush:
            flush_cache()
        torch.cuda._sleep(2_000_000)  # ~1 ms of GPU spin
        e0.record()
        fn()
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) * 1e3)
    ts.sort()
    return ts[len(ts) // 2]
