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

Kernel families = the functions ``moe_<kind>`` below (all consume the same ``W``; see
README.md). ``@op_register`` adds them, after the reference / test functions of
``ref.py``, to the package's ops (``_ops.py``): ``MOE`` = ``ops.moe``, ``MOE_DYN`` =
``ops.moe_dyn``. A config is ``{"kind": kind, knob: value, ...}``, its knobs' grid is
the ``@param``s, its limits the ``@requires``.
  tp2k   prep -> gemm1 (q8s epilogue) -> gemm2 (atomic scatter)
  tp1k   prep -> one fused gemm1 + gemm2 kernel (BM 64)
  tp1kq  same, persistent work-queue kernel (inter_dim 256 only; NOT graph-safe)
  ep3k   ep_prep (local sort + x quant + tile schedule) -> ep gemm1 -> ep gemm2
  ep1k   one persistent launch: routing + x quant + gemm1 + gemm2 (M <= 512)
TP kinds' prep: all experts local -> tp_prep (ONE launch: sort over all experts + x row
quant + out zeroing); an expert slice (EP shapes) -> aiter moe_sorting with an
expert_mask + quantize_rows.
"""

import copy
import csv
import functools
import os

import torch

from pyhip.kit import meta, param, requires, stack

from ._ops import op_register, ops
from .kernels import ep_gemm as EG
from .kernels import ep_small as ES
from .kernels import tp_fmoe as F
from .kernels import tp_fmoe_q as Q
from .kernels import tp_gemm1 as G1
from .kernels import tp_gemm2 as G2
from .kernels import tp_prep as TPP
from .kernels.row_quant import quantize_rows
from .ref import ACTS  # importing ref.py registers the ops' ref / test / check / dims first
from .weights import weight_shape

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.path.join(HERE, "configs", "a8w4_moe_tuned.csv")
EP_MAX_LOCAL_EXPERTS = 32  # ep_prep / ep_small: one wave holds the per-expert scans
EP_SMALL_MAXM = 512  # ep1k compile-time token capacity
EP_CFGS = ("t16", "s16", "s32", "m64", "l128", "l128c")  # the ep3k presets of the grid


# =============================================================================
# limits (shared by several kernel families)
# =============================================================================

kernel_acts = requires(lambda act: act in ACTS, "act={act!r}: the kernels implement silu, situv2")
tp_limits = stack(
    kernel_acts,
    requires(lambda model_dim, inter_dim: inter_dim % 256 == 0 and model_dim % 512 == 0,
             "TP kernels need inter_dim % 256 == 0 and model_dim % 512 == 0 "
             "(inter_dim={inter_dim}, model_dim={model_dim})"))
ep_limits = stack(
    kernel_acts,
    requires(lambda experts: experts <= EP_MAX_LOCAL_EXPERTS,
             "experts={experts} > 32: the EP kernels scan the local experts in one wave"),
    requires(lambda model_dim, topk: 64 % topk == 0 and model_dim % 512 == 0,
             "EP kernels need 64 % topk == 0 and model_dim % 512 == 0 "
             "(topk={topk}, model_dim={model_dim})"),
    requires(lambda num_tokens: num_tokens < 65536, "num_tokens={num_tokens} >= 65536"))

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


def _arb_fits(inter_dim, preset):
    # gemm2's K loop (K = inter_dim) needs >= 4 * PF K tiles: inter_dim >= 1024 (ARB 128)
    # or 2048 (ARB 256)
    arb = preset["g2"].get("arb", 256)
    return inter_dim // arb >= 8 and (inter_dim // arb) % 4 == 0


def _ep_cached(preset):
    return any(preset[g].get("bcm", 2) != 2 or preset[g].get("acm", 2) != 2
               for g in ("g1", "g2"))


ep3k_space = stack(  # moe_ep3k and moe_dyn_ep3k
    param("cfg", EP_CFGS, id="{}"),
    ep_limits,
    requires(lambda cfg: cfg in EG.CFGS, "no ep3k preset {cfg!r} in ep_gemm.CFGS"),
    requires(lambda inter_dim, cfg, **over: _arb_fits(inter_dim, ep_preset(cfg, over)),
             "inter_dim={inter_dim} is too short for gemm2's K loop of {cfg}"),
    meta(tile_m=lambda cfg, **over: ep_preset(cfg, over)["bm"],
         cached_loads=lambda cfg, **over: _ep_cached(ep_preset(cfg, over))))


# =============================================================================
# kernel families
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
@param("block_m", (32, 64), id="bm{}")
@param("g1_bcm", (1, 2), id="g1bcm{}")
@param("g2", ("base", "nt4"), id="g2{}")
@tp_limits
@requires(lambda g2, inter_dim: g2 != "nt4" or inter_dim == 256,
          "g2nt4 needs inter_dim == 256 (inter_dim={inter_dim})")
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
@param("bcm", (1, 2))
@param("ksplit", lambda inter_dim: sorted({inter_dim // c for c in (256, 384, 512)
                                           if inter_dim % c == 0}), id="ks{}")
@tp_limits
@requires(lambda inter_dim, ksplit: (inter_dim // 128) % ksplit == 0
          and inter_dim // ksplit <= 512,
          "inter_dim={inter_dim} does not split into ks{ksplit} chunks of 128..512 columns")
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
@param("bcm", (1,))
@tp_limits
@requires(lambda inter_dim: inter_dim == 256,
          "tp1kq needs inter_dim == 256 (inter_dim={inter_dim})")
@requires(lambda capturing: not capturing, "tp1kq is not CUDA-graph safe (host launch epoch)")
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
@param("maxm", (EP_SMALL_MAXM,), id="m{}")
@ep_limits
@requires(lambda topk, inter_dim: topk == 8 and inter_dim % 2048 == 0,
          "ep1k needs topk == 8 and inter_dim % 2048 == 0 (topk={topk}, inter_dim={inter_dim})")
@requires(lambda num_tokens, maxm: num_tokens <= maxm, "num_tokens={num_tokens} > maxm {maxm}")
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


# =============================================================================
# the frozen functions on (H, I, E local, E global, topk) shape tuples
# =============================================================================

_UNKNOWN = dict.fromkeys(MOE.vocab)


def shape_vocab(shape, tokens=None, act=None):
    """``MOE`` vocabulary of ``shape`` = (H, I, E local, E global, topk), the key of the
    tuned table (None: unknown, the limits that need it are skipped)."""
    H, I, E, EGl, topk = shape
    return dict(_UNKNOWN, model_dim=H, inter_dim=I, experts=E, experts_global=EGl, topk=topk,
                num_tokens=tokens, act=act)


def config_name(cfg):
    """``"tp2k bm64 g1bcm1 g2base"``, ``"ep3k m64"``, ... (``MOE.name_of``)."""
    return MOE.name_of(cfg)


def parse_config(name):
    cfg = MOE.parse(name)
    assert cfg["kind"] != "ep3k" or cfg["cfg"] in EG.CFGS, name
    return cfg


def supported(cfg, shape, *, act="silu", capturing=False, tokens=None):
    """Whether ``cfg`` can run ``shape`` = (H, I, E local, E global, topk) with this
    activation / under CUDA-graph capture / for ``tokens`` rows. Structural limits
    only (the ``@requires`` above); a tuned table holds just configs that ran on its
    shape."""
    return MOE.why_not(cfg, shape_vocab(shape, tokens, act), capturing=capturing) is None


def run_config(cfg, x, W, topk_weights, topk_ids, *, expert_offset=0, num_experts=None,
               act="silu", situ=None):
    """Run one explicit config (dict or name); no table lookup."""
    if isinstance(cfg, str):
        cfg = parse_config(cfg)
    return MOE.run(cfg, x, W, topk_weights, topk_ids, expert_offset=expert_offset,
                   num_experts=num_experts, act=act, situ=situ)


# =============================================================================
# tuned table
# =============================================================================

FIELDS = ["model_dim", "inter_dim", "experts", "experts_global", "topk", "routing", "act",
          "num_tokens", "config", "us", "cos"]


@functools.lru_cache(maxsize=8)
def _load_table(path, mtime):
    """{(H, I, E, E_global, topk, routing, act): sorted [(num_tokens, [(us, cfg), ...])]}:
    every candidate measured at every point, fastest first. Rows without an ``act``
    (tables written before the column existed) are SiLU."""
    del mtime  # cache key only: reload when the file changes
    pts = {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            key = (int(r["model_dim"]), int(r["inter_dim"]), int(r["experts"]),
                   int(r["experts_global"]), int(r["topk"]), r["routing"],
                   r.get("act") or "silu")
            pts.setdefault(key, {}).setdefault(int(r["num_tokens"]), []).append(
                (float(r["us"]), parse_config(r["config"])))
    return {k: sorted((t, sorted(c, key=lambda x: x[0])) for t, c in v.items())
            for k, v in pts.items()}


def default_config(shape, tokens, act="silu", capturing=False):
    """No table row: the EP auto thresholds when the EP kernels fit, else tp2k."""
    for lim, name in EG.AUTO:
        if lim is None or tokens <= lim:
            cfg = dict(kind="ep3k", cfg=name)
            if supported(cfg, shape, act=act, capturing=capturing, tokens=tokens):
                return cfg
            break
    return dict(kind="tp2k", block_m=64, g1_bcm=1, g2="base")


def lookup_config(shape, tokens, *, act="silu", capturing=False, routing="uniform",
                  csv_path=DEFAULT_CSV):
    """Fastest supported config of the first tuned point >= ``tokens`` (the largest
    point if beyond the table) among the rows of ``act``; ``default_config`` if the
    shape is not tuned. Memoized per table file version: the returned dict is shared,
    do not modify it."""
    return _lookup(tuple(shape), tokens, act, capturing, routing, csv_path,
                   _version(csv_path))


def _version(path):
    """Cache key of the table file: its mtime (None: no table)."""
    try:
        return os.stat(path).st_mtime if path else None
    except OSError:
        return None


@functools.lru_cache(maxsize=4096)  # the decision of each call of a8w4_moe
def _lookup(shape, tokens, act, capturing, routing, csv_path, version):
    for _, cfg in _point(_table_rows(shape, routing, csv_path, act, version), tokens):
        if supported(cfg, shape, act=act, capturing=capturing, tokens=tokens):
            return cfg
    return default_config(shape, tokens, act, capturing)


def _table_rows(shape, routing, csv_path, act, version):
    """Tuned points of (shape, routing, act). A shape tuned only for SiLU serves the
    other activations from its SiLU rows (the configs rank alike, README 7.8)."""
    if version is None:
        return None
    table = _load_table(csv_path, version)
    return table.get((*shape, routing, act)) or table.get((*shape, routing, "silu"))


def _point(rows, tokens):
    """Candidates of the first tuned point >= tokens (the largest point beyond); none
    without rows."""
    if not rows:
        return ()
    return next((c for n, c in rows if n >= tokens), rows[-1][1])


def dyn_config(shape, tokens, *, act="silu", routing="uniform", csv_path=DEFAULT_CSV):
    """Config of ``a8w4_moe_dyn`` (EP kernels only): the fastest supported ``ep3k``
    config of the tuned point >= ``tokens`` (rows of ``act``), else the EP auto
    thresholds. Memoized like ``lookup_config``."""
    return _dyn(tuple(shape), tokens, act, routing, csv_path, _version(csv_path))


@functools.lru_cache(maxsize=4096)
def _dyn(shape, tokens, act, routing, csv_path, version):
    ok = lambda c: c["kind"] == "ep3k" and supported(c, shape, act=act)  # noqa: E731
    for _, cfg in _point(_table_rows(shape, routing, csv_path, act, version), tokens):
        if ok(cfg):
            return cfg
    names = [n for lim, n in EG.AUTO if lim is None or tokens <= lim]
    for name in names + [n for _, n in EG.AUTO]:
        if ok(dict(kind="ep3k", cfg=name)):
            return dict(kind="ep3k", cfg=name)
    raise ValueError(f"a8w4_moe_dyn: (H, I, E, E_global, topk) = {shape} is not "
                     "supported by the EP kernels (see supported())")


# =============================================================================
# entry point
# =============================================================================


def a8w4_moe(x, W, topk_weights, topk_ids, *, expert_offset=0, num_experts=None, act="silu",
             situ=None, config=None, routing="uniform", csv_path=DEFAULT_CSV):
    """A8W4 MoE partial sum of this rank's experts: bf16 x [M, H] -> bf16 [M, H].

    W: ``prepare_weights`` output (E local experts). topk_weights f32 [M, topk],
    topk_ids int32 [M, topk] (global expert ids). expert_offset / num_experts: the
    rank owns global experts [expert_offset, expert_offset + E) of num_experts
    (defaults: 0 / E = TP). act: "silu" | "situv2" (situ: dict(situ_beta,
    situ_linear_beta, swiglu_limit); all kernel families). config: a config (dict or
    name, see ``config_name``) to force, else the tuned table at ``csv_path`` decides.
    Under CUDA-graph capture only graph-safe configs are picked."""
    E, H, I = weight_shape(W)
    M, topk = topk_ids.shape
    assert x.dtype == torch.bfloat16 and x.shape == (M, H) and x.is_contiguous(), x.shape
    assert topk_ids.dtype == torch.int32 and topk_weights.dtype == torch.float32
    assert topk_weights.shape == (M, topk)
    if M == 0:
        return x.new_empty((0, H))
    shape = (H, I, E, num_experts or E, topk)
    if config is None:
        config = lookup_config(shape, M, act=act,
                               capturing=torch.cuda.is_current_stream_capturing(),
                               routing=routing, csv_path=csv_path)
    return run_config(config, x, W, topk_weights, topk_ids, expert_offset=expert_offset,
                      num_experts=num_experts, act=act, situ=situ)


def a8w4_moe_dyn(x, W, topk_weights, topk_ids, m_valid, *, x_scale=None, expert_offset=0,
                 num_experts=None, act="silu", situ=None, est_tokens=None, out=None,
                 config=None, routing="uniform", csv_path=DEFAULT_CSV):
    """EP MoE on a capacity-sized all-to-all receive buffer (e.g. MORI dispatch).

    Only rows [0, m_valid[0]) of ``x`` / ``topk_ids`` / ``topk_weights`` [M, ...] are
    valid; ``m_valid`` is an int32 device tensor read by the kernels (no host sync,
    one CUDA graph serves any count up to M). ``x``: bf16 [M, H] (``x_scale`` None,
    quantized per row by the prep) or fp8 FNUZ [M, H] with ``x_scale`` f32 [M, H/128]
    (per (token, 128) block, e.g. MORI's blockwise fp8 dispatch) or [M] (per token).
    ``topk_ids`` are global; the rank owns [expert_offset, expert_offset + E).
    ``act`` / ``situ``: as ``a8w4_moe``.
    ``est_tokens`` (default M): host estimate of the valid rows, only used to pick
    the config and size the grids (the workgroups stride over the actual tiles, so
    any value is correct). Returns bf16 [M, H] (written into ``out`` if given); rows
    >= m_valid are undefined. EP kernels only (<= 32 local experts, M < 65536; see
    ``supported``)."""
    E, H, I = weight_shape(W)
    M, topk = topk_ids.shape
    assert x.shape == (M, H) and x.is_contiguous(), x.shape
    assert x.dtype in (torch.bfloat16, torch.float8_e4m3fnuz), x.dtype
    assert (x.dtype == torch.bfloat16) == (x_scale is None), "fp8 x needs x_scale"
    assert topk_ids.dtype == torch.int32 and topk_weights.dtype == torch.float32
    assert m_valid.dtype == torch.int32 and m_valid.is_cuda, m_valid
    if M == 0:
        return out if out is not None else x.new_empty((0, H), dtype=torch.bfloat16)
    shape = (H, I, E, num_experts or E, topk)
    est = _est(M, est_tokens)
    if config is None:
        config = dyn_config(shape, est, act=act, routing=routing, csv_path=csv_path)
    elif isinstance(config, str):
        config = parse_config(config)
    assert config["kind"] == "ep3k", f"{config_name(config)}: a8w4_moe_dyn runs ep3k only"
    return MOE_DYN.run(config, x, W, topk_weights, topk_ids, m_valid, x_scale=x_scale,
                       expert_offset=expert_offset, num_experts=num_experts, act=act,
                       situ=situ, est_tokens=est, out=out)
