# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""The a8w4 side of the bench tools, on top of ``pyhip.kit.bench`` (variants, A/B and
check engines, run records) and ``pyhip.kit.timing`` (protocols, ABBA rounds, noise band;
both re-exported here): the cases (``ref.SHAPES`` or a shape tuple) and their inputs
(``--local-experts-per-token``), the routing annotation of a sample (local routes,
m-blocks per ``@meta`` tile_m), bare ep3k preset names in variant specs, and the
``resolve`` adapter (``run(x, W, tw, ti)``) breakdown and experiment scripts use.
"""

import dataclasses
import logging
import os

import torch

from pyhip.kit import bench as kb
from pyhip.kit.bench import Recorder, md_table, meta_line  # noqa: F401 - re-exported
from pyhip.kit.bench import check_protocols  # noqa: F401
from pyhip.kit.bench import fmt as fmt_cos  # noqa: F401
from pyhip.kit.timing import (  # noqa: F401 - re-exported for the bench tools
    MIN_COPIES,
    MIN_NOISE,
    PROTOCOLS,
    Timer,
    interleaved,
    noise_band,
    verdict,
)

from .. import ref as R
from ..kernels import ep_gemm as EG
from ..moe import MOE, shape_dims
from ..ref import COS_TOL, SHAPES  # noqa: F401 - COS_TOL: re-exported for the tools

PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIFF_TOL = 1e-4  # cos diff vs the base above this: flagged (bf16 atomics alone: ~1e-5)


def quiet():
    """Silence the log lines that bury results: aiter's NUMA / module-import notes and
    torch C++ warnings (Gloo peer connects, ROCTracer duplicate flows) of spawned workers."""
    os.environ.setdefault("AITER_LOG_LEVEL", "ERROR")
    os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")
    logging.getLogger("aiter").setLevel(logging.ERROR)


def run_meta(tool, args=None):
    return kb.run_meta(tool, args, prog="python -m a8w4.bench", path=PKG_DIR)


# =============================================================================
# cases and inputs
# =============================================================================


@dataclasses.dataclass
class Case:
    name: str
    shape: tuple  # (H, I, E local, E global, topk)
    e0: int = 0
    act: str = "silu"
    routing: str = "uniform"

    @property
    def situ(self):
        return R.SITU if self.act == "situv2" else None

    def inputs(self, tokens, seed, local=0):
        """x, topk_weights, topk_ids of one routing sample. ``local`` > 0: every token
        routes to exactly ``local`` distinct local experts (EP shapes: the rank's work is
        then deterministic; uniform routing leaves ~1/3 of the EP8 ranks idle at 1 token)."""
        H, I, E, EGl, K = self.shape
        if local:
            return local_routing(tokens, H, EGl, E, K, local, seed, self.e0)
        return R.make_routing(tokens, H, EGl, K, seed=seed, routing=self.routing)

    def kwargs(self):
        """The interface's keyword arguments of the case."""
        return dict(expert_offset=self.e0, num_experts=self.shape[3], act=self.act,
                    situ=self.situ)

    def interface(self, tokens, seed, local=0):
        """The interface arguments of one sample (``moe_ref`` / ``a8w4_moe``): the routing
        of ``inputs``, the cached random weights of the shape (``ref.moe_test``'s)."""
        H, I, E, EGl, K = self.shape
        x, tw, ti = self.inputs(tokens, seed, local)
        return dict(x=x, W=R._weights(E, H, I), topk_weights=tw, topk_ids=ti, **self.kwargs())

    def dims(self, tokens=None):
        """The ``ops.moe`` dims of the case at ``tokens``."""
        return dict(shape_dims(self.shape, tokens, self.act, self.routing), expert_offset=self.e0)

    def describe(self):
        H, I, E, EGl, K = self.shape
        return (f"case {self.name}: H {H}, I {I}, {E} local / {EGl} experts, top-{K}, "
                f"expert_offset {self.e0}, {self.act}, {self.routing} routing")


def get_case(name=None, shape=None, e0=0, act="silu", routing="uniform"):
    """A ``ref.SHAPES`` entry (tp8 / tp4 / ep8; the old names mimo-v2.6-flash-<name> work
    too) or ``shape`` "H,I,E_local,E_global,topk"."""
    if shape:
        H, I, E, EGl, K = (int(v) for v in shape.split(","))
        return Case(f"H{H}-I{I}-E{E}of{EGl}-top{K}", (H, I, E, EGl, K), e0, act, routing)
    full = str(name).rpartition("-")[2] if str(name) not in SHAPES else name
    if full not in SHAPES:
        raise SystemExit(f"unknown case {name!r}: one of {sorted(SHAPES)}, or --shape "
                         "H,I,E,E_global,topk")
    *shp, e0c = SHAPES[full]
    return Case(full, tuple(shp), e0 or e0c, act, routing)


def add_case_args(ap, local=True):
    g = ap.add_argument_group("case")
    g.add_argument("--case", default=None,
                   help=f"one of {sorted(SHAPES)} (ref.SHAPES)")
    g.add_argument("--shape", default=None, help="H,I,E_local,E_global,topk (instead of --case)")
    g.add_argument("--expert-offset", type=int, default=0)
    g.add_argument("--act", default="silu", choices=R.ACTS)
    g.add_argument("--routing", default="uniform", choices=R.ROUTINGS)
    if local:
        g.add_argument("--local-experts-per-token", type=int, default=0, metavar="L",
                       help="EP: every token routes to exactly L local experts (deterministic "
                            "per-rank work at tiny batches; 0: --routing over all experts)")


def case_from_args(a):
    if not (a.case or a.shape):
        raise SystemExit("--case (tp8 / tp4 / ep8) or --shape is required")
    return get_case(a.case, a.shape, a.expert_offset, a.act, a.routing)


def add_out_args(ap, name):
    g = ap.add_argument_group("output")
    g.add_argument("--out", default=None, help="directory: append to <out>/<name>.md and "
                                               "<name>.jsonl (e.g. an experiments/ dir)")
    g.add_argument("--name", default=name, help="file name stem (default: %(default)s)")
    g.add_argument("--verbose", action="store_true", help="keep aiter / torch log lines")


def local_routing(T, H, EGl, E, K, local, seed, e0=0):
    """x, tw, ti in which every token routes to ``local`` distinct experts of [e0, e0 + E)
    and to K - local other experts."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn((T, H), dtype=torch.bfloat16, device="cuda", generator=g)
    remote = torch.tensor([e for e in range(EGl) if not e0 <= e < e0 + E], device="cuda")
    loc = e0 + torch.argsort(torch.rand((T, E), device="cuda", generator=g), dim=1)[:, :local]
    pick = torch.argsort(torch.rand((T, remote.numel()), device="cuda", generator=g), dim=1)
    ti = torch.cat([loc, remote[pick[:, :K - local]]], dim=1).to(torch.int32).contiguous()
    tw = torch.rand((T, K), device="cuda", generator=g)
    return x, (tw / tw.sum(-1, keepdim=True)).float().contiguous(), ti


def route_blocks(ti, case, bms):
    """(local routes, {bm: m-blocks}) of a routing sample: the work the kernels get. EP8
    has ~T/32 rows per expert, so e.g. l128's block count at 4096 tokens moves +-9%
    between samples (log_20261115 3.4): compare configs on the same samples."""
    E = case.shape[2]
    loc = ti.flatten().long() - case.e0
    loc = loc[(loc >= 0) & (loc < E)]
    cnt = torch.bincount(loc, minlength=E)
    return int(loc.numel()), {bm: int(((cnt + bm - 1) // bm).sum())
                              for bm in sorted({b for b in bms if b})}


def annotate(case):
    """``kit.bench.ab``'s annotation of a sample: local routes and the m-blocks of every
    tile_m of the variants."""
    def f(inputs, resolved):
        bms = [MOE.meta(r.cfg, "tile_m") if r.cfg is not None else None for r in resolved]
        routes, blocks = route_blocks(inputs["topk_ids"], case, bms)
        return (f": {routes} local routes" + "".join(f", {n} blocks of {bm}"
                                                     for bm, n in blocks.items()),
                dict(routes=routes, blocks=blocks))
    return f


# =============================================================================
# variants
# =============================================================================

VARIANT_HELP = """\
variant spec: [label=]<base>[:<override>,...]  or  [label=]py:<module | file.py>:<func>
  base      a config name (python -m a8w4.tune candidates <case>), e.g. "tp1k bcm1 ks1",
            "ep3k l128"; a bare ep_gemm.CFGS name ("l128" = "ep3k l128"); "@table" (the
            tuned table's pick at each token count)
  override  ep3k only: g1.<k>=v / g2.<k>=v (ep_gemm1 / ep_gemm2 kwargs: pf, bcm, acm, sch,
            st, prio, nw, ni / no, arb, tr, wpe), <k>=v (bm, rowv=32/64/128, sched, sst);
            v "-" drops the key (kernel default)
  py        func(x, W, topk_weights, topk_ids, *, expert_offset, num_experts, act, situ)
            -> bf16 [M, H]: any pipeline, e.g. an experiment's kernel copy
  label     the name in the tables (default: the spec)
examples: "ep3k m64"  "pf1=m64:g2.pf=1"  "c=l128:g1.bcm=0,g2.bcm=0"
"""


def alias(base):
    """A bare ep_gemm.CFGS name is that ep3k preset; an unknown ep3k preset is refused."""
    base = f"ep3k {base}" if base in EG.CFGS else base
    kind, _, cfg = base.partition(" ")
    if kind == "ep3k" and cfg not in EG.CFGS:
        raise SystemExit(f"{base!r} is not a config name: no preset {cfg!r} in "
                         f"ep_gemm.CFGS\n\n{VARIANT_HELP}")
    return base


def parse_variant(text):
    return kb.parse_variant(MOE, text, alias)


@dataclasses.dataclass
class Resolved:
    """A variant at one token count: ``run(x, W, tw, ti)`` -> bf16 out."""
    label: str
    spec: str
    config: str
    run: object
    cfg: object = None  # the config dict, overrides included (None: py variant)
    bm: object = None  # the solver's @meta tile_m (rows per block; None: none)
    cached: object = None  # @meta cached_loads: reads W / A with cached loads (None: unknown)


def resolve(v, case, tokens):
    """``kit.bench.resolve`` on the case -> Resolved with ``run(x, W, tw, ti)``, or a str
    (no tuned config for ``@table``)."""
    r = kb.resolve(MOE, v, case.dims(tokens))
    if isinstance(r, str):
        return r
    kw = case.kwargs()
    run = (lambda x, W, tw, ti, r=r: r.run(dict(x=x, W=W, topk_weights=tw, topk_ids=ti, **kw)))
    bm = MOE.meta(r.cfg, "tile_m") if r.cfg is not None else None
    return Resolved(r.label, r.spec, r.config, run, r.cfg, bm, r.cached)


def ref_out(case, x, tw, ti, raw):
    return R.ref_moe(x, tw, ti, raw, case.e0, case.act, case.situ)
