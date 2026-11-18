# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""A8W4 MoE for gfx942: one entry point for TP and EP, kernel picked from a tuned table.

    W = prepare_weights(w1, w2, w1_scale, w2_scale)            # once per weight
    out = a8w4_moe(x, W, topk_weights, topk_ids,                # bf16 [M, H]
                   expert_offset=rank * E_local,                # EP; 0 for TP
                   num_experts=E_global)                        # EP; None for TP

Contract: ``out = sum over (t, k) with topk_ids[t, k] in [expert_offset,
expert_offset + E) of topk_weights[t, k] * W2_e(act(W1_e x_t))``, E = local experts
in ``W``. TP (each rank: all experts, a 1/tp slice of inter_dim) is the case
expert_offset = 0, num_experts = E; EP (each rank: E whole experts) passes the global
ids. Either way ``out`` is this rank's partial sum; the caller reduces across ranks.

``a8w4_moe_dyn`` is the EP variant for all-to-all receive buffers (e.g. MORI
dispatch): capacity-sized inputs whose valid row count is a device tensor, x in bf16
or in fp8 with per-(token, 128) scales; no host sync, CUDA-graph safe.

Solvers = the functions ``moe_<solver>`` below (all consume the same ``W``; see
README.md). ``@op_register`` adds them, after the problem side of ``ref.py``
(``moe_ref`` / ``moe_problem`` / ...), to the package's ops (``_ops.py``): ``MOE`` =
``ops.moe``, ``MOE_DYN`` = ``ops.moe_dyn``. A config is ``{"solver": name, knob: value,
...}``, the knobs' values are the ``@knob``s; a config that cannot run a call raises (the
kernels' asserts). ``MOE(...)`` / ``MOE.pick(problem)`` pick one from the tuned table
(``pyhip.kit`` dispatch): only tuned keys are guaranteed, an untuned one raises
ValueError.
  tp2k   prep -> gemm1 (q8s epilogue) -> gemm2 (atomic scatter)
  tp1k   prep -> one fused gemm1 + gemm2 kernel (BM 64)
  tp1kq  same, persistent work-queue kernel (inter_dim 256 only; NOT graph-safe)
  ep3k   ep_prep (local sort + x quant + tile schedule) -> ep gemm1 -> ep gemm2
  ep1k   one persistent launch: routing + x quant + gemm1 + gemm2 (M <= 512)
TP solvers' prep: all experts local -> tp_prep (ONE launch: sort over all experts + x row
quant + out zeroing); an expert slice (EP shapes) -> aiter moe_sorting with an
expert_mask + quantize_rows.
"""

import copy

import torch

from pyhip.kit import knob, meta, no_capture, stack

from . import ref as _ref  # noqa: F401 - registers the ops' ref / problem / check / dims first
from ._ops import DEFAULT_CSV, op_register, ops  # noqa: F401 - DEFAULT_CSV re-exported
from .kernels import ep_gemm as EG
from .kernels import ep_small as ES
from .kernels import tp_fmoe as F
from .kernels import tp_fmoe_q as Q
from .kernels import tp_gemm1 as G1
from .kernels import tp_gemm2 as G2
from .kernels import tp_prep as TPP
from .kernels.row_quant import quantize_rows
from .weights import weight_shape

EP_SMALL_MAXM = 512  # ep1k compile-time token capacity
EP_CFGS = ("t16", "s16", "s32", "m64", "l128", "l128c")  # the ep3k presets of its @knob

_DROP = "-"  # an ep3k override value that removes the key (the kernel default applies)


def ep_preset(cfg, over=None):
    """``EG.CFGS[cfg]`` with ``over`` applied: {"g2.pf": 1, "bm": 64, ...}, "g1." /
    "g2." keys go to ep_gemm1 / ep_gemm2, a value "-" drops the key. Without
    overrides the preset itself (not a copy)."""
    if not over:
        return EG.CFGS[cfg]
    c = copy.deepcopy(EG.CFGS[cfg])
    for k, v in over.items():
        d, key = (c[k[:2]], k[3:]) if k[:3] in ("g1.", "g2.") else (c, k)
        if isinstance(v, str) and v == _DROP:
            d.pop(key, None)
        else:
            d[key] = v
    return c


def _ep_cached(preset):
    return any(preset[g].get("bcm", 2) != 2 or preset[g].get("acm", 2) != 2
               for g in ("g1", "g2"))


ep3k_space = stack(  # moe_ep3k and moe_dyn_ep3k
    knob("cfg", EP_CFGS, id="{}"),
    meta(tile_m=lambda cfg, **over: ep_preset(cfg, over)["bm"],
         cached_loads=lambda cfg, **over: _ep_cached(ep_preset(cfg, over))))


# =============================================================================
# solvers
# =============================================================================

_MASKS = {}


def _expert_mask(n_global, e0, n_local, device):
    key = (n_global, e0, n_local, str(device))
    if key not in _MASKS:
        m = torch.zeros(n_global, dtype=torch.int32, device=device)
        m[e0:e0 + n_local] = 1
        _MASKS[key] = m
    return _MASKS[key]


def _sort(topk_ids, topk_weights, shape, e0, block_m):
    """aiter moe_sorting (local expert ids with an expert_mask in EP) -> (sorted_ids,
    sorted_weights, sorted_expert_ids, num_valid_ids, zeroed bf16 out [M, H])."""
    from aiter.fused_moe import moe_sorting

    H, _, E, EGl, _ = shape
    mask = _expert_mask(EGl, e0, E, topk_ids.device) if (E != EGl or e0) else None
    return moe_sorting(topk_ids, topk_weights, EGl, H, torch.bfloat16, block_m,
                       expert_mask=mask)[:5]


def _prep(topk_ids, topk_weights, x, shape, e0, block_m):
    """-> (sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, zeroed out,
    x8, xs): one tp_prep launch when every expert is local, else _sort + quantize_rows."""
    H, _, E, EGl, topk = shape
    if (E == EGl and not e0 and E <= TPP.NT and H % 512 == 0 and topk % 4 == 0
            and x.shape[0] > 0):
        sids, sw, seids, nv, x8, xs, out = TPP.tp_prep(topk_ids, topk_weights, x,
                                                       block_m=block_m, num_experts=E)
        return sids, sw, seids, nv, out, x8, xs
    return (*_sort(topk_ids, topk_weights, shape, e0, block_m), *quantize_rows(x))


def _shape(W, topk_ids, num_experts):
    E, H, I = weight_shape(W)
    return H, I, E, num_experts or E, topk_ids.shape[1]


@op_register
@knob("block_m", (32, 64), id="bm{}")
@knob("g1_bcm", (1, 2), id="g1bcm{}")
@knob("g2", ("base", "nt4"), id="g2{}")
@meta(tile_m=lambda block_m: block_m, cached_loads=lambda g1_bcm: g1_bcm != 2)
def moe_tp2k(x, W, topk_weights, topk_ids, *, expert_offset=0, num_experts=None, act="silu",
             situ=None, block_m=32, g1_bcm=1, g2="base"):
    """prep -> gemm1 (q8s epilogue) -> gemm2 (atomic scatter)."""
    shape = _shape(W, topk_ids, num_experts)
    H, I, E, _, _ = shape
    groups = I // 128
    sids, sw, seids, nv, out, x8, xs = _prep(topk_ids, topk_weights, x, shape, expert_offset,
                                             block_m)
    sorted_size = seids.numel() * block_m
    buf = torch.empty(sorted_size * (I + groups * 4), dtype=torch.uint8, device=x.device)
    G1.a8w4_gemm1(x8, xs, W["w1"], W["w1_scale"], W["w1_exp"], sids, seids, nv, buf,
                  num_experts=E, inter_dim=I, tile_m=block_m, tile_n=128,
                  b_cache_mod=g1_bcm, xcd_swizzle=1, dyn_m=block_m > 32, act=act,
                  **(situ or {}))
    nd = sorted_size * I
    a2 = buf[:nd].view(torch.float8_e4m3fnuz).view(sorted_size, I)
    a2s = buf[nd:].view(torch.float32).view(sorted_size, groups)
    G2.a8w4_gemm2(a2, a2s, W["w2"], W["w2_scale"], sids, sw, seids, nv, out,
                  num_experts=E, tile_m=block_m, tile_n=128, b_cache_mod=2, xcd_swizzle=0,
                  **(dict(n_tiles=4) if g2 == "nt4" else {}))
    return out


# each CTA takes 256 / 384 / 512 inter columns (256: 2 waves/SIMD, best for small/medium
# batches; 384-512: 1 wave/SIMD, best for large batches at large inter_dim)
@op_register
@knob("bcm", (1, 2))
@knob("ksplit", lambda inter_dim: sorted({inter_dim // c for c in (256, 384, 512)
                                           if inter_dim % c == 0}), id="ks{}")
@meta(tile_m=F.BM, cached_loads=lambda bcm: bcm != 2)
def moe_tp1k(x, W, topk_weights, topk_ids, *, expert_offset=0, num_experts=None, act="silu",
             situ=None, bcm=1, ksplit=1):
    """prep -> one fused gemm1 + gemm2 kernel (BM 64)."""
    shape = _shape(W, topk_ids, num_experts)
    sids, sw, seids, nv, out, x8, xs = _prep(topk_ids, topk_weights, x, shape, expert_offset,
                                             F.BM)
    return F.a8w4_fmoe_bm64(x8, xs, W, sids, sw, seids, nv, out, topk=shape[4],
                            b_cache_mod=bcm, waves_per_eu=2 if shape[1] // ksplit <= 256 else 1,
                            ksplit=ksplit, act=act, **(situ or {}))


@op_register
@knob("bcm", (1,))
@no_capture("tp1kq passes its work-queue epoch as a host launch argument")
@meta(tile_m=Q.BM, cached_loads=lambda bcm: bcm != 2)
def moe_tp1kq(x, W, topk_weights, topk_ids, *, expert_offset=0, num_experts=None, act="silu",
              situ=None, bcm=1):
    """tp1k as a persistent work-queue kernel."""
    shape = _shape(W, topk_ids, num_experts)
    sids, sw, seids, nv, out, x8, xs = _prep(topk_ids, topk_weights, x, shape, expert_offset,
                                             Q.BM)
    return Q.a8w4_fmoe_bm64q(x8, xs, W, sids, sw, seids, nv, out, topk=shape[4],
                             b_cache_mod=bcm, act=act, **(situ or {}))


@op_register
@ep3k_space
def moe_ep3k(x, W, topk_weights, topk_ids, *, expert_offset=0, num_experts=None, act="silu",
             situ=None, cfg="t16", **over):
    """ep_prep (local sort + x quant + tile schedule) -> ep gemm1 -> ep gemm2. ``cfg``: an
    ``EG.CFGS`` preset; ``over``: overrides of it (``ep_preset``)."""
    return EG.ep_moe(x, W, topk_weights, topk_ids, e0=expert_offset, cfg=ep_preset(cfg, over),
                     act=act, **(situ or {}))


@op_register
@knob("maxm", (EP_SMALL_MAXM,), id="m{}")
@meta(tile_m=None, cached_loads=False)
def moe_ep1k(x, W, topk_weights, topk_ids, *, expert_offset=0, num_experts=None, act="silu",
             situ=None, maxm=EP_SMALL_MAXM):
    """one persistent launch: routing + x quant + gemm1 + gemm2 (M <= maxm)."""
    return ES.ep_moe_small(x, W, topk_weights, topk_ids, e0=expert_offset, maxm=maxm, act=act,
                           **(situ or {}))


def _est(M, est_tokens):
    return M if est_tokens is None else max(1, min(M, int(est_tokens)))


@op_register
@ep3k_space
def moe_dyn_ep3k(x, W, topk_weights, topk_ids, m_valid, *, x_scale=None, expert_offset=0,
                 num_experts=None, act="silu", situ=None, est_tokens=None, out=None,
                 cfg="t16", **over):
    """ep3k on a capacity-sized receive buffer; the valid count stays on the device."""
    return EG.ep_moe_dyn(x, x_scale, W, topk_weights, topk_ids, m_valid, e0=expert_offset,
                         est_tokens=_est(x.shape[0], est_tokens), out=out,
                         cfg=ep_preset(cfg, over), act=act, **(situ or {}))


MOE, MOE_DYN = ops.moe, ops.moe_dyn


def shape_problem(shape, tokens=None, act=None, routing=None, op=MOE):
    """The problem of ``op`` for ``shape`` = (H, I, E local, E global, topk), ``tokens``
    rows (``num_tokens``), ``act`` and ``routing``; the other dims None (unknown). For
    the tools that work on shape tuples."""
    H, I, E, EGl, topk = shape
    return dict(dict.fromkeys(op.dim_names), model_dim=H, inter_dim=I, experts=E,
                experts_global=EGl, topk=topk, num_tokens=tokens, act=act, routing=routing)


# =============================================================================
# entry points
# =============================================================================


def a8w4_moe(x, W, topk_weights, topk_ids, *, expert_offset=0, num_experts=None, act="silu",
             situ=None, **dispatch):
    """A8W4 MoE partial sum of this rank's experts: bf16 x [M, H] -> bf16 [M, H].

    W: ``prepare_weights`` output (E local experts). topk_weights f32 [M, topk],
    topk_ids int32 [M, topk] (global expert ids). expert_offset / num_experts: the
    rank owns global experts [expert_offset, expert_offset + E) of num_experts
    (defaults: 0 / E = TP). act: "silu" | "situv2" (situ: dict(situ_beta,
    situ_linear_beta, swiglu_limit); all solvers).

    ``MOE(...)`` (``ops.moe``; ``moe_dims`` checks the inputs). ``dispatch`` goes to it:
    ``config=`` (a config dict or name to force, e.g. ``"tp1k bcm1 ks1"``), ``table=``
    (another tuned table), dim hints (``routing=``). Only tuned keys are
    guaranteed: an untuned one raises ValueError, a forced config the kernels cannot
    run raises their AssertionError.
    Under CUDA-graph capture only graph-safe configs are picked. M == 0 returns an
    empty output: the kernels need rows."""
    if x.shape[0] == 0:
        return x.new_empty(x.shape)
    return MOE(x, W, topk_weights, topk_ids, expert_offset=expert_offset,
               num_experts=num_experts, act=act, situ=situ, **dispatch)


def a8w4_moe_dyn(x, W, topk_weights, topk_ids, m_valid, *, x_scale=None, expert_offset=0,
                 num_experts=None, act="silu", situ=None, est_tokens=None, out=None,
                 **dispatch):
    """EP MoE on a capacity-sized all-to-all receive buffer (e.g. MORI dispatch).

    Only rows [0, m_valid[0]) of ``x`` / ``topk_ids`` / ``topk_weights`` [M, ...] are
    valid; ``m_valid`` is an int32 device tensor read by the kernels (no host sync,
    one CUDA graph serves any count up to M). ``x``: bf16 [M, H] (``x_scale`` None,
    quantized per row by the prep) or fp8 FNUZ [M, H] with ``x_scale`` f32 [M, H/128]
    (per (token, 128) block, e.g. MORI's blockwise fp8 dispatch) or [M] (per token).
    ``topk_ids`` are global; the rank owns [expert_offset, expert_offset + E).
    ``act`` / ``situ`` / ``dispatch``: as ``a8w4_moe`` (``MOE_DYN``, ep3k only).
    ``est_tokens`` (default M): host estimate of the valid rows, only used to pick
    the config and size the grids (the workgroups stride over the actual tiles, so
    any value is correct). Returns bf16 [M, H] (written into ``out`` if given); rows
    >= m_valid are undefined. EP kernels only (<= 32 local experts, M < 65536)."""
    if x.shape[0] == 0:
        return out if out is not None else x.new_empty(x.shape, dtype=torch.bfloat16)
    return MOE_DYN(x, W, topk_weights, topk_ids, m_valid, x_scale=x_scale,
                   expert_offset=expert_offset, num_experts=num_experts, act=act, situ=situ,
                   est_tokens=est_tokens, out=out, **dispatch)
