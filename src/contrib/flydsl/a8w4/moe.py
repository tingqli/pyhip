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

Kernel families (all consume the same ``W``; see README.md):
  tp2k   moe_sorting -> x row quant -> gemm1 (q8s epilogue) -> gemm2 (atomic scatter)
  tp1k   moe_sorting -> x row quant -> one fused gemm1 + gemm2 kernel (BM 64)
  tp1kq  same, persistent work-queue kernel (inter_dim 256 only; NOT graph-safe)
  ep3k   ep_prep (local sort + x quant + tile schedule) -> ep gemm1 -> ep gemm2
  ep1k   one persistent launch: routing + x quant + gemm1 + gemm2 (M <= 512)
"""

import csv
import functools
import os

import torch

from .kernels import ep_gemm as EG
from .kernels import ep_small as ES
from .kernels import tp_fmoe as F
from .kernels import tp_fmoe_q as Q
from .kernels import tp_gemm1 as G1
from .kernels import tp_gemm2 as G2
from .kernels.row_quant import quantize_rows
from .weights import weight_shape

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.path.join(HERE, "configs", "a8w4_moe_tuned.csv")
KINDS = ("tp2k", "tp1k", "tp1kq", "ep3k", "ep1k")
ACTS = ("silu", "situv2")  # every kernel family implements both
EP_MAX_LOCAL_EXPERTS = 32  # ep_prep / ep_small: one wave holds the per-expert scans
EP_SMALL_MAXM = 512  # ep1k compile-time token capacity


# =============================================================================
# configs: dict <-> canonical name
# =============================================================================


def config_name(cfg):
    k = cfg["kind"]
    if k == "tp2k":
        return f"tp2k bm{cfg['block_m']} g1bcm{cfg['g1_bcm']} g2{cfg['g2']}"
    if k == "tp1k":
        return f"tp1k bcm{cfg['bcm']} ks{cfg['ksplit']}"
    if k == "tp1kq":
        return f"tp1kq bcm{cfg['bcm']}"
    if k == "ep3k":
        return f"ep3k {cfg['cfg']}"
    assert k == "ep1k", k
    return f"ep1k m{cfg['maxm']}"


def parse_config(name):
    p = name.split()
    k = p[0]
    if k == "tp2k":
        return dict(kind=k, block_m=int(p[1][2:]), g1_bcm=int(p[2][5:]), g2=p[3][2:])
    if k == "tp1k":
        return dict(kind=k, bcm=int(p[1][3:]), ksplit=int(p[2][2:]))
    if k == "tp1kq":
        return dict(kind=k, bcm=int(p[1][3:]))
    if k == "ep3k":
        assert p[1] in EG.CFGS, p[1]
        return dict(kind=k, cfg=p[1])
    assert k == "ep1k", name
    return dict(kind=k, maxm=int(p[1][1:]))


# =============================================================================
# shape + capabilities
# =============================================================================


def supported(cfg, shape, *, act="silu", capturing=False, tokens=None):
    """Whether ``cfg`` can run ``shape`` = (H, I, E local, E global, topk) with this
    activation / under CUDA-graph capture / for ``tokens`` rows. Structural limits
    only; a tuned table holds just configs that ran on its shape."""
    H, I, E, EGl, topk = shape
    k = cfg["kind"]
    if act not in ACTS:
        return False
    if k.startswith("tp"):
        if I % 256 or H % 512:
            return False
        if k == "tp2k":
            return cfg["g2"] != "nt4" or I == 256
        if k == "tp1k":
            return (I // 128) % cfg["ksplit"] == 0 and I // cfg["ksplit"] <= 512
        return I == 256 and not capturing  # tp1kq: host-side launch epoch
    # EP family: <= 32 local experts, 64 % topk == 0, M < 65536
    if E > EP_MAX_LOCAL_EXPERTS or 64 % topk or H % 512:
        return False
    if tokens is not None and tokens >= 65536:
        return False
    if k == "ep3k":
        # gemm2's K loop (K = I) needs >= 4 * PF K tiles: I >= 1024 (ARB 128) / 2048
        arb = EG.CFGS[cfg["cfg"]]["g2"].get("arb", 256)
        return I // arb >= 8 and (I // arb) % 4 == 0
    return topk == 8 and I % 2048 == 0 and (tokens is None or tokens <= cfg["maxm"])


# =============================================================================
# runners
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


def _run_tp2k(cfg, x, W, tw, ti, shape, e0, akw):
    H, I, E, _, topk = shape
    bm = cfg["block_m"]
    groups = I // 128
    sids, sw, seids, nv, out = _sort(ti, tw, shape, e0, bm)
    x8, xs = quantize_rows(x)
    sorted_size = seids.numel() * bm
    buf = torch.empty(sorted_size * (I + groups * 4), dtype=torch.uint8, device=x.device)
    G1.a8w4_gemm1(x8, xs, W["w1"], W["w1_scale"], W["w1_exp"], sids, seids, nv, buf,
                  num_experts=E, tile_m=bm, epilog="q8s", inter_dim=I, topk=topk,
                  tile_n=128, b_cache_mod=cfg["g1_bcm"], xcd_swizzle=1, dyn_m=bm > 32,
                  fast_dec=2, **akw)
    nd = sorted_size * I
    a2 = buf[:nd].view(torch.float8_e4m3fnuz).view(sorted_size, I)
    a2s = buf[nd:].view(torch.float32).view(sorted_size, groups)
    kw = dict(tile_n=128, b_cache_mod=2, xcd_swizzle=0, dyn_rows=True)
    if cfg["g2"] == "nt4":
        kw.update(n_tiles=4, epilog="direct64", row_vgpr=True)
    G2.a8w4_gemm2(a2, a2s, W["w2"], W["w2_scale"], sids, sw, seids, nv, out,
                  num_experts=E, tile_m=bm, a_group=128, fast_dec=2, **kw)
    return out


def _run_tp1k(cfg, x, W, tw, ti, shape, e0, akw):
    I, topk = shape[1], shape[4]
    sids, sw, seids, nv, out = _sort(ti, tw, shape, e0, F.BM)
    x8, xs = quantize_rows(x)
    ks = cfg["ksplit"]
    return F.a8w4_fmoe_bm64(x8, xs, W, sids, sw, seids, nv, out, topk=topk,
                            b_cache_mod=cfg["bcm"], waves_per_eu=2 if I // ks <= 256 else 1,
                            ksplit=ks, **akw)


def _run_tp1kq(cfg, x, W, tw, ti, shape, e0, akw):
    sids, sw, seids, nv, out = _sort(ti, tw, shape, e0, Q.BM)
    x8, xs = quantize_rows(x)
    return Q.a8w4_fmoe_bm64q(x8, xs, W, sids, sw, seids, nv, out, topk=shape[4],
                             b_cache_mod=cfg["bcm"], **akw)


def _run_ep3k(cfg, x, W, tw, ti, shape, e0, akw):
    return EG.ep_moe(x, W, tw, ti, e0=e0, adt="fp8", cfg=EG.CFGS[cfg["cfg"]], **akw)


def _run_ep1k(cfg, x, W, tw, ti, shape, e0, akw):
    return ES.ep_moe_small(x, W, tw, ti, e0=e0, adt="fp8", maxm=cfg["maxm"], **akw)


RUNNERS = dict(tp2k=_run_tp2k, tp1k=_run_tp1k, tp1kq=_run_tp1kq, ep3k=_run_ep3k,
               ep1k=_run_ep1k)


def run_config(cfg, x, W, topk_weights, topk_ids, *, expert_offset=0, num_experts=None,
               act="silu", situ=None):
    """Run one explicit config (dict or name); no table lookup."""
    if isinstance(cfg, str):
        cfg = parse_config(cfg)
    E, H, I = weight_shape(W)
    shape = (H, I, E, num_experts or E, topk_ids.shape[1])
    akw = dict(act=act, **(situ or {}))
    return RUNNERS[cfg["kind"]](cfg, x, W, topk_weights, topk_ids, shape, expert_offset, akw)


# =============================================================================
# tuned table
# =============================================================================

FIELDS = ["model_dim", "inter_dim", "experts", "experts_global", "topk", "routing",
          "num_tokens", "config", "us", "cos"]


@functools.lru_cache(maxsize=8)
def _load_table(path, mtime):
    """{(H, I, E, E_global, topk, routing): sorted [(num_tokens, [(us, cfg), ...])]}:
    every candidate measured at every point, fastest first."""
    del mtime  # cache key only: reload when the file changes
    pts = {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            key = (int(r["model_dim"]), int(r["inter_dim"]), int(r["experts"]),
                   int(r["experts_global"]), int(r["topk"]), r["routing"])
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
    point if beyond the table); ``default_config`` if the shape is not tuned."""
    rows = _table_rows(shape, routing, csv_path)
    if rows:
        return _pick(rows, shape, tokens, act, capturing)
    return default_config(shape, tokens, act, capturing)


def _table_rows(shape, routing, csv_path):
    if not csv_path or not os.path.exists(csv_path):
        return None
    return _load_table(csv_path, os.path.getmtime(csv_path)).get((*shape, routing))


def _point(rows, tokens):
    """Candidates of the first tuned point >= tokens (the largest point beyond)."""
    return next((c for n, c in rows if n >= tokens), rows[-1][1])


def _pick(rows, shape, tokens, act, capturing):
    for _, cfg in _point(rows, tokens):
        if supported(cfg, shape, act=act, capturing=capturing, tokens=tokens):
            return cfg
    return default_config(shape, tokens, act, capturing)


def dyn_config(shape, tokens, *, act="silu", routing="uniform", csv_path=DEFAULT_CSV):
    """Config of ``a8w4_moe_dyn`` (EP kernels only): the fastest supported ``ep3k``
    config of the tuned point >= ``tokens``, else the EP auto thresholds."""
    ok = lambda c: c["kind"] == "ep3k" and supported(c, shape, act=act)  # noqa: E731
    rows = _table_rows(shape, routing, csv_path)
    if rows:
        for _, cfg in _point(rows, tokens):
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
    est = M if est_tokens is None else max(1, min(M, int(est_tokens)))
    if config is None:
        config = dyn_config(shape, est, act=act, routing=routing, csv_path=csv_path)
    elif isinstance(config, str):
        config = parse_config(config)
    assert config["kind"] == "ep3k", f"{config_name(config)}: a8w4_moe_dyn runs ep3k only"
    return EG.ep_moe_dyn(x, x_scale, W, topk_weights, topk_ids, m_valid, e0=expert_offset,
                         est_tokens=est, out=out, cfg=EG.CFGS[config["cfg"]], act=act,
                         **(situ or {}))
