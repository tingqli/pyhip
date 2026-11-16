# SPDX-License-Identifier: Apache-2.0
# Derived from aiter 7cd70aca8 aiter/ops/flydsl/kernels/act.py
# (Copyright (C) 2025-2026 FlyDSL Project Contributors); silu and situv2 only.
# Modifications Copyright (C) 2026 Advanced Micro Devices, Inc.
"""Gate / up activation of the gemm1 epilogues: ``act(gate) * up`` on f32 registers.

exp2 / rcp based sigmoid and sign-restored tanh, in batch forms that emit all exp2 before
the reciprocals. :func:`gate_up_act` picks silu or situv2 so kernels carry no ``act`` branch.
"""

from typing import NamedTuple

import flydsl.expr as fx
from flydsl.expr import const_expr, rocdl
from flydsl.expr.typing import T

from ._mem import _raw

LOG2E = 1.4426950408889634


def sigmoid_batch(xs):
    """Emit all exponentials before their reciprocals to preserve batch scheduling."""
    e = [
        fx.Float32(rocdl.exp2(T.f32, _raw(x * fx.Float32(-LOG2E)))) for x in xs
    ]
    return [fx.Float32(rocdl.rcp(T.f32, _raw(fx.Float32(1.0) + ei))) for ei in e]


def sigmoid_f32(g):
    return sigmoid_batch([g])[0]


def clamp_gate_up(g, u, neg_limit):
    """Upper-bound the gate and symmetrically clamp the up value."""
    return -fx.max(-g, neg_limit), fx.max(-fx.max(-u, neg_limit), neg_limit)


def silu_mul_batch(gs, us):
    sig = sigmoid_batch(gs)
    return [gs[i] * sig[i] * us[i] for i in range(len(gs))]


def tanh_batch(xs):
    """Sign-restored tanh with exp2 and reciprocal operations grouped by stage."""
    neg_two_log2e = fx.Float32(-2.0 * LOG2E)
    es = []
    for x in xs:
        abs_x = fx.max(x, -x)
        es.append(fx.Float32(rocdl.exp2(T.f32, _raw(abs_x * neg_two_log2e))))
    recips = [fx.Float32(rocdl.rcp(T.f32, _raw(fx.Float32(1.0) + e))) for e in es]
    zero = fx.Float32(0.0)
    out = []
    for i, x in enumerate(xs):
        tanh_abs = (fx.Float32(1.0) - es[i]) * recips[i]
        out.append((x > zero).select(tanh_abs, -tanh_abs))
    return out


def tanh_f32(x):
    return tanh_batch([x])[0]


def situ_mul(g, u, beta, beta_rcp, lbeta, lbeta_rcp):
    gate = beta * tanh_f32(g * beta_rcp) * sigmoid_f32(g)
    up = lbeta * tanh_f32(u * lbeta_rcp)
    return gate * up


def situ_mul_batch(gs, us, beta, beta_rcp, lbeta, lbeta_rcp, neg_clamp_limit):
    """SiTUv2 activation (aiter mixed_moe situ_mul_vec4):
        situ(g)    = beta * tanh(g / beta) * sigmoid(g)
        situ_up(u) = linear_beta * tanh(u / linear_beta)
        out        = situ(clamp_gate(g)) * situ_up(clamp_lin(u))
    clamp_gate: g <= +limit (upper only); clamp_lin: u in [-limit, +limit].

    beta/beta_rcp/lbeta/lbeta_rcp and neg_clamp_limit are runtime fx.Float32
    scalars (nothing baked; one kernel serves any beta/limit). neg_clamp_limit is
    -swiglu_limit (host-negated), so the clamp matches the a8w4/mixed_moe situv2
    path exactly: a +inf limit -> -inf -> maximumf no-op = no clamp; a finite
    limit clamps. Do NOT drop the clamp -- at large linear_beta the model expects
    the clamp and no-clamp diverges badly (a16w4 must match a8w4).
    """
    out = []
    for i in range(len(gs)):
        # clamp_gate: g <= +lim (upper only); clamp_lin: u in [-lim, +lim].
        g, u = clamp_gate_up(gs[i], us[i], neg_clamp_limit)
        out.append(situ_mul(g, u, beta, beta_rcp, lbeta, lbeta_rcp))
    return out


class SituParams(NamedTuple):
    """Runtime SiTUv2 scalars, as :func:`situ_mul_batch` wants them.

    Built by :func:`situ_params` so the ``-swiglu_limit`` negation happens once, next
    to the reason for it, instead of inside every caller's epilogue loop.
    """

    beta: object
    beta_rcp: object
    linear_beta: object
    linear_beta_rcp: object
    neg_clamp_limit: object


def situ_params(beta, beta_rcp, linear_beta, linear_beta_rcp, swiglu_limit):
    """Bundle the five runtime SiTUv2 f32 scalars (all fx.Float32, nothing baked)."""
    return SituParams(beta, beta_rcp, linear_beta, linear_beta_rcp, -swiglu_limit)


def gate_up_act(act, gs, us, situ=None):
    """Fused gate+up epilogue activation: ``act(gate) * up``.

    ``act`` is a compile-time string, "silu" or "situv2"; ``situ`` carries the
    runtime situv2 parameters.
    """
    if const_expr(act == "situv2"):
        return situ_mul_batch(gs, us, *situ)
    return silu_mul_batch(gs, us)


def situ_args(situ_beta=1.0, situ_linear_beta=1.0, swiglu_limit=float("inf")):
    """Runtime SiTUv2 launch args (beta, 1/beta, linear_beta, 1/linear_beta,
    clamp limit; +inf = no clamp), as flydsl_a16w4_gemm1. Ignored by silu."""
    beta, lbeta = float(situ_beta), float(situ_linear_beta)
    if beta <= 0.0 or lbeta <= 0.0:
        raise ValueError(f"situ_beta/situ_linear_beta must be > 0, got {beta}/{lbeta}")
    return beta, 1.0 / beta, lbeta, 1.0 / lbeta, float(swiglu_limit)
