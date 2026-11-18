# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Test / tuning helpers: random weights + routing, fp32 references, cold-cache timing.

Not needed at inference time. Weight generation uses aiter's mxfp4 quantizer and
preshuffle (``aiter.get_torch_quant`` / ``aiter.ops.shuffle``); everything else is
plain torch.

References: ``moe_ref`` / ``moe_dyn_ref`` take the same arguments as ``a8w4_moe`` /
``a8w4_moe_dyn`` (the prepared weights, decoded by ``decode_fd2``); ``ref_moe`` is the
older form on the unshuffled mxfp4 tensors. All share one fp32 core.

The problem side of the ops ``moe`` and ``moe_dyn`` (``kernelkit``, registered into
``_ops.ops``; the kernels are the solvers in ``moe.py``): ``<op>_ref`` the interface and
the answer, ``<op>_problem`` the inputs of one problem (its parameters but ``seed``: the
dims), ``<op>_check`` the tolerance, ``<op>_dims`` the dims of a call. Cold-cache timing
(``flush_cache``, ``weight_copies``,
``time_cold``, ``WCOPIES``) is ``kernelkit.timing``, re-exported here.
"""

import functools

import torch
from kernelkit import dim
from kernelkit.timing import NCOPIES as WCOPIES  # noqa: F401 - re-exported for the tools
from kernelkit.timing import copies as weight_copies  # noqa: F401
from kernelkit.timing import flush_cache, time_cold  # noqa: F401

from ._ops import op_register
from .weights import prepare_weights, shuffle_mxfp4, weight_shape

ACTS = ("silu", "situv2")
# SiTUv2 params used by the tests: the clamp (16) sits well inside the gate/up range
# of the random data (std ~64), so most values are clamped and a missing or wrong
# clamp changes the result far beyond the test tolerance (cos diff ~0.1).
SITU = dict(situ_beta=4.0, situ_linear_beta=25.0, swiglu_limit=16.0)
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


def _fd2_lut(device):
    return torch.tensor(_FP4 + [-v for v in _FP4], dtype=torch.float32, device=device)


def _fd2_to_f32(w, s, exp):
    """One matrix of the fd2 form (``weights.py``) -> f32 [N, K]. w: u8 [N, K/2],
    preshuffled, nibbles k0..k7 of a dword as (k_j | k_{j+4} << 4); s: u8 [N * K/32],
    preshuffled, (channel max - group exponent) * 8; exp: i32 [N], channel max e8m0."""
    N, K = w.shape[0], w.shape[1] * 2
    G = K // 32
    b = w.reshape(-1, 4)
    nib = torch.cat([b & 0xF, b >> 4], dim=-1)  # per dword: k0..k3, k4..k7
    # undo shuffle_weight_a16w4(w, 16): (N/16, K/128, 4, 16 rows, 32 nibbles)
    v = _fd2_lut(w.device)[nib.long()].view(N // 16, K // 128, 4, 16, 32)
    v = v.permute(0, 3, 1, 2, 4).reshape(N, G, 32)
    # undo shuffle_scale_a16w4: (N/32, K/256, k lane 4, n lane 16, k pack 2, n pack 2)
    d = s.reshape(N // 32, K // 256, 4, 16, 2, 2).permute(0, 5, 3, 1, 4, 2).reshape(N, G)
    code = exp.reshape(N, 1).float() - d.float() / 8
    return (v * torch.exp2(code - 127.0).unsqueeze(-1)).reshape(N, K)


def decode_fd2(W, e):
    """Local expert ``e`` of a prepared weight dict -> (w1 f32 [2I, H], w2 f32 [H, I]),
    the values the kernels multiply with: the mxfp4 weights, except that groups whose
    scale is >= 12 below their channel's max decode to 0."""
    out = []
    for i in (1, 2):
        w = W[f"w{i}"][e].view(torch.uint8)
        n = w.shape[0] * w.shape[1] // 16  # scale bytes: one per 32 K values
        s = W[f"w{i}_scale"].view(torch.uint8)[e * n:(e + 1) * n]
        out.append(_fd2_to_f32(w, s, W[f"w{i}_exp"][e]))
    return tuple(out)


def _quant_fp8(a, group):
    """fp8-FNUZ round trip with one scale per ``group`` columns (amax / 240)."""
    g = a.view(a.shape[0], a.shape[1] // group, group)
    s = g.abs().amax(-1, keepdim=True).clamp(min=1e-30) / 240.0
    return ((g / s).to(torch.float8_e4m3fnuz).float() * s).view(a.shape)


def quant_fp8_groups(x, group=128):
    """x [M, H] -> (fp8 FNUZ [M, H], f32 scale [M, H/group]), one scale per (row,
    ``group`` columns): the layout of a blockwise-fp8 EP dispatch (``a8w4_moe_dyn``)."""
    g = x.float().view(x.shape[0], -1, group)
    s = g.abs().amax(-1, keepdim=True).clamp(min=1e-30) / 240.0
    return (g / s).to(torch.float8_e4m3fnuz).view(x.shape), s.view(x.shape[0], -1)


def dequant_fp8(x, x_scale):
    """fp8 x [M, H] with f32 scales [M, H/g] (one per row and g columns) or [M] (one per
    row) -> f32 [M, H]."""
    M, H = x.shape
    if x_scale.dim() == 1:
        return x.float() * x_scale.view(M, 1)
    g = x_scale.shape[1]
    return (x.float().view(M, g, H // g) * x_scale.unsqueeze(-1)).view(M, H)


def _moe_f32(x, topk_weights, topk_ids, expert, n_local, expert_offset, act, situ, quant_x):
    """The fp32 reference core: this rank's partial MoE [tokens, H] over local experts
    [0, n_local) = global [expert_offset, expert_offset + n_local); ``expert(e)`` ->
    (w1 f32 [2I, H], w2 f32 [H, I]). x (when ``quant_x``) and the activation output
    are fp8-quantized as the kernels do (x per row, the intermediate per (row, 128
    columns)), so the remaining differences are accumulation order and bf16 output."""
    tokens, H = x.shape
    loc = topk_ids.long() - expert_offset
    loc = torch.where((loc >= 0) & (loc < n_local), loc, torch.full_like(loc, -1))
    xf = _quant_fp8(x.float(), H) if quant_x else x.float()
    out = torch.zeros((tokens, H), dtype=torch.float32, device=x.device)
    for e in loc[loc >= 0].unique().tolist():
        tok, slot = torch.nonzero(loc == e, as_tuple=True)
        w1, w2 = expert(e)
        gate, up = (xf[tok] @ w1.t()).split(w1.shape[0] // 2, dim=-1)
        a = _quant_fp8(act_ref(gate, up, act, situ), 128)
        out.index_add_(0, tok, (a @ w2.t()) * topk_weights[tok, slot].view(-1, 1))
    return out


def ref_moe(x, topk_weights, topk_ids, raw, expert_offset=0, act="silu", situ=None,
            quant_x=True):
    """fp32 reference of this rank's partial MoE [tokens, H] on the unshuffled mxfp4
    tensors ``raw`` (``make_weights``): experts [expert_offset, expert_offset + E) of
    the global ``topk_ids``. ``quant_x=False``: x is used as given (an already
    fp8-quantized input, dequantized). See ``moe_ref`` for the prepared weights."""
    def expert(e):
        return (_mxfp4_to_f32(raw["w1q"][e], raw["w1s"][e]),
                _mxfp4_to_f32(raw["w2q"][e], raw["w2s"][e]))

    return _moe_f32(x, topk_weights, topk_ids, expert, raw["w1q"].shape[0], expert_offset,
                    act, situ, quant_x)


@op_register
def moe_ref(x, W, topk_weights, topk_ids, *, expert_offset=0, num_experts=None, act="silu",
            situ=None):
    """fp32 reference of ``a8w4_moe`` (same arguments), on the prepared weights ``W``."""
    return _moe_f32(x, topk_weights, topk_ids, functools.partial(decode_fd2, W),
                    weight_shape(W)[0], expert_offset, act, situ, quant_x=True)


@op_register
def moe_dyn_ref(x, W, topk_weights, topk_ids, m_valid, *, x_scale=None, expert_offset=0,
                num_experts=None, act="silu", situ=None, est_tokens=None, out=None):
    """fp32 reference of ``a8w4_moe_dyn`` (same arguments): rows [0, m_valid[0]) as
    ``moe_ref`` (an fp8 x is dequantized and used as given), the rows past them 0."""
    n = int(m_valid[0])
    xv = x[:n] if x_scale is None else dequant_fp8(x[:n], x_scale[:n])
    res = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
    res[:n] = _moe_f32(xv, topk_weights[:n], topk_ids[:n], functools.partial(decode_fd2, W),
                       weight_shape(W)[0], expert_offset, act, situ, quant_x=x_scale is None)
    return res


def cos_diff(x, y):
    """1 - cosine similarity (0 = identical direction; 0 when both are all zero, e.g.
    an EP rank that got no routes)."""
    x, y = x.double(), y.double()
    den = (x * x + y * y).sum()
    if float(den) == 0.0:
        return 0.0
    return float(1 - 2 * (x * y).sum() / den)


# =============================================================================
# ops moe / moe_dyn: problems, tolerance, dims of a call
# =============================================================================

COS_TOL = 5e-3  # vs the fp8-emulating fp32 reference (typical 1e-5 .. 1e-4)
# name: (model_dim, inter_dim, experts, experts_global, topk, expert_offset), the
# MiMo-V2.6-Flash per-rank shapes (EP: rank 3, global expert ids [96, 128))
SHAPES = {
    "tp8": (4096, 256, 256, 256, 8, 0),
    "tp4": (4096, 512, 256, 256, 8, 0),
    "ep8": (4096, 2048, 32, 256, 8, 96),
}
_SHAPE = "model_dim, inter_dim, experts, experts_global, topk, expert_offset"


@functools.lru_cache(maxsize=1)  # the tests run problem by problem
def _weights(experts, model_dim, inter_dim):
    return make_weights(experts, model_dim, inter_dim)[0]


@op_register
@dim("act", ACTS, id="{}")
@dim("routing", ROUTINGS, id="{}")
def moe_problem(model_dim, inter_dim, experts, experts_global, topk, expert_offset, num_tokens,
                routing="uniform", act="silu", *, seed=0):
    """Inputs of ``moe_ref`` / ``a8w4_moe``: random mxfp4 weights of ``experts`` local
    experts (``make_weights``, cached per shape), ``make_routing(seed=seed)`` over
    ``experts_global``; SiTUv2 with ``SITU``."""
    x, tw, ti = make_routing(num_tokens, model_dim, experts_global, topk, seed=seed,
                             routing=routing)
    return dict(x=x, W=_weights(experts, model_dim, inter_dim), topk_weights=tw, topk_ids=ti,
                expert_offset=expert_offset, num_experts=experts_global, act=act,
                situ=SITU if act == "situv2" else None)


@op_register
def moe_check(out, expected, **_):
    """bf16 output of the reference's shape within ``COS_TOL`` -> {"cos": cos diff}."""
    assert out.dtype == torch.bfloat16 and out.shape == expected.shape, (out.dtype, out.shape)
    d = cos_diff(expected, out.float())
    assert d < COS_TOL, f"cos diff {d:.2e} >= {COS_TOL}"
    return {"cos": d}


@op_register
def moe_dims(x, W, topk_weights, topk_ids, *, expert_offset=0, num_experts=None, act="silu",
             situ=None):
    """Also checks the arguments (``ops.moe(...)`` calls it on every call)."""
    E, H, I = weight_shape(W)
    M, topk = topk_ids.shape
    assert x.dtype == torch.bfloat16 and x.shape == (M, H) and x.is_contiguous(), x.shape
    assert topk_ids.dtype == torch.int32 and topk_weights.dtype == torch.float32
    assert topk_weights.shape == (M, topk)
    return dict(num_tokens=M, model_dim=H, inter_dim=I, experts=E,
                experts_global=num_experts or E, topk=topk, expert_offset=expert_offset,
                act=act)


@op_register
@dim(_SHAPE, {"ep8": SHAPES["ep8"]}, name="shape")
@dim("fp8", {"bf16": False, "fp8": True})
@dim("act", ACTS, id="{}")
@dim("routing", ROUTINGS, id="{}")
def moe_dyn_problem(model_dim, inter_dim, experts, experts_global, topk, expert_offset,
                    capacity, num_tokens, fp8=False, act="silu", routing="uniform", *, seed=0):
    """Inputs of ``moe_dyn_ref`` / ``a8w4_moe_dyn``: an EP all-to-all receive buffer of
    ``capacity`` rows. Rows [0, num_tokens) are tokens with >= 1 local route (global
    ids); the rows past them are stale (NaN x, random ids) and must not be read. fp8: x
    in fp8 with per-(row, 128) scales (``quant_fp8_groups``). The host's estimate of the
    valid rows (``est_tokens``) is exact."""
    x, tw, ti = make_routing(capacity, model_dim, experts_global, topk, seed=seed,
                             routing=routing)
    g = torch.Generator(device="cuda").manual_seed(seed)
    lo, hi = expert_offset, expert_offset + experts
    forced = torch.randint(lo, hi, (capacity,), device="cuda", generator=g)
    has_local = ((ti >= lo) & (ti < hi)).any(-1)
    ti[:, 0] = torch.where(has_local, ti[:, 0], forced.to(torch.int32))
    ti[num_tokens:] = torch.randint(0, experts_global, ti[num_tokens:].shape, device="cuda",
                                    generator=g, dtype=torch.int32)
    x[num_tokens:] = float("nan")
    xs = None
    if fp8:
        x, xs = quant_fp8_groups(x)
    return dict(x=x, W=_weights(experts, model_dim, inter_dim), topk_weights=tw, topk_ids=ti,
                m_valid=torch.tensor([num_tokens], dtype=torch.int32, device="cuda"),
                x_scale=xs, expert_offset=expert_offset, num_experts=experts_global, act=act,
                situ=SITU if act == "situv2" else None, est_tokens=num_tokens)


@op_register
def moe_dyn_check(result, expected, *, m_valid, **_):
    """``moe_check`` on rows [0, m_valid[0]) (the rows past them are undefined). The
    first parameter is not called ``out``: that is an argument of the interface."""
    assert result.shape == expected.shape, (result.shape, expected.shape)
    n = int(m_valid[0])
    return moe_check(result[:n], expected[:n])


@op_register
def moe_dyn_dims(x, W, topk_weights, topk_ids, m_valid, *, x_scale=None, expert_offset=0,
                 num_experts=None, act="silu", situ=None, est_tokens=None, out=None):
    """``capacity``: the buffer's rows; ``num_tokens``: the valid rows as the host
    estimates them (``est_tokens`` clamped to the capacity; default the capacity).
    m_valid is not read: it is on the device, reading it would sync. Also checks the
    arguments (``ops.moe_dyn(...)`` calls it on every call)."""
    E, H, I = weight_shape(W)
    M, topk = topk_ids.shape
    assert x.shape == (M, H) and x.is_contiguous(), x.shape
    assert x.dtype in (torch.bfloat16, torch.float8_e4m3fnuz), x.dtype
    assert (x.dtype == torch.bfloat16) == (x_scale is None), "fp8 x needs x_scale"
    assert topk_ids.dtype == torch.int32 and topk_weights.dtype == torch.float32
    assert m_valid.dtype == torch.int32 and m_valid.is_cuda, m_valid
    est = M if est_tokens is None else max(0, min(M, int(est_tokens)))
    return dict(model_dim=H, inter_dim=I, experts=E, experts_global=num_experts or E,
                topk=topk, expert_offset=expert_offset, act=act, capacity=M, num_tokens=est,
                fp8=x_scale is not None)
