# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Shared pieces of the bench tools: cases, timing protocols, variant specs, routing
samples, run metadata, result files and the noise / verdict rules.

Timing protocols (experiments/log_20261115.md 3.3). Only ``coldrot`` with >= 8 weight
copies matches a model (each layer reads other weights, so W is never left in the
256 MB MALL between two calls of the same layer); the others exist to measure the bias:

  coldrot  512 MB read flush + GPU sleep; each call reads the next of N weight copies
  rotate   N weight copies back to back, no flush (depends on the previous call's MALL)
  cold     flush + sleep, the same weights every call: the MALL keeps them, so configs
           that read W through the cache (bcm 0 / 1) look 5-13% faster than they are
  bigread  2 GB read flush + sleep, the same weights (the same bias as cold)
  wflush   512 MB write + read flush + sleep, the same weights (dirty lines: cached-W
           configs look 9-48% slower)
  warm     no flush, the same weights
"""

import dataclasses
import datetime
import importlib
import importlib.util
import json
import logging
import os
import shlex
import socket
import statistics
import subprocess
import sys

import torch

from .. import ref as R
from ..kernels import ep_gemm as EG
from ..moe import MOE, shape_vocab
from ..tune import CASES, COS_TOL  # noqa: F401 - COS_TOL: re-exported for the tools

PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROTOCOLS = ("coldrot", "rotate", "cold", "bigread", "wflush", "warm")
MIN_COPIES = 8  # 2 copies still favor cached W at 16 tokens (log_20261115 3.3)
MIN_NOISE = 0.3  # % : below this a difference is never called
DIFF_TOL = 1e-4  # cos diff vs the base above this: flagged (bf16 atomics alone: ~1e-5)


def quiet():
    """Silence the log lines that bury results: aiter's NUMA / module-import notes and
    torch C++ warnings (Gloo peer connects, ROCTracer duplicate flows) of spawned workers."""
    os.environ.setdefault("AITER_LOG_LEVEL", "ERROR")
    os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")
    logging.getLogger("aiter").setLevel(logging.ERROR)


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

    def describe(self):
        H, I, E, EGl, K = self.shape
        return (f"case {self.name}: H {H}, I {I}, {E} local / {EGl} experts, top-{K}, "
                f"expert_offset {self.e0}, {self.act}, {self.routing} routing")


def get_case(name=None, shape=None, e0=0, act="silu", routing="uniform"):
    """A ``tune.CASES`` entry (full name or its suffix: tp8 / tp4 / ep8) or ``shape``
    "H,I,E_local,E_global,topk"."""
    if shape:
        H, I, E, EGl, K = (int(v) for v in shape.split(","))
        return Case(f"H{H}-I{I}-E{E}of{EGl}-top{K}", (H, I, E, EGl, K), e0, act, routing)
    full = name if name in CASES else next((c for c in sorted(CASES)
                                            if c.endswith("-" + str(name))), None)
    if full is None:
        raise SystemExit(f"unknown case {name!r}: one of {sorted(CASES)} or the suffix "
                         "after the last '-' (tp8, tp4, ep8), or --shape H,I,E,E_global,topk")
    *shp, e0c = CASES[full]
    return Case(full, tuple(shp), e0 or e0c, act, routing)


def add_case_args(ap, local=True):
    g = ap.add_argument_group("case")
    g.add_argument("--case", default=None,
                   help=f"one of {sorted(CASES)} or its suffix (tp8, tp4, ep8)")
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
        raise SystemExit("--case (tp8 / tp4 / ep8 / a tune.CASES name) or --shape is required")
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


# =============================================================================
# variants
# =============================================================================

VARIANT_HELP = """\
variant spec: [label=]<base>[:<override>,...]  or  [label=]py:<module | file.py>:<func>
  base      a config name (python -m a8w4.tune candidates --case ...), e.g. "tp1k bcm1 ks1",
            "ep3k l128"; a bare ep_gemm.CFGS name ("l128" = "ep3k l128"); "@table" (the
            tuned table's pick at each token count) or "@auto" (the untuned default)
  override  ep3k only: g1.<k>=v / g2.<k>=v (ep_gemm1 / ep_gemm2 kwargs: pf, bcm, acm, sch,
            st, prio, nw, ni / no, arb, tr, wpe), <k>=v (bm, rowv=32/64/128, sched, sst);
            v "-" drops the key (kernel default)
  py        func(x, W, topk_weights, topk_ids, *, expert_offset, num_experts, act, situ)
            -> bf16 [M, H]: any pipeline, e.g. an experiment's kernel copy
  label     the name in the tables (default: the spec)
examples: "ep3k m64"  "pf1=m64:g2.pf=1"  "c=l128:g1.bcm=0,g2.bcm=0"
"""

_DROP = "-"


@dataclasses.dataclass
class Variant:
    label: str
    spec: str
    base: str = ""
    overrides: dict = dataclasses.field(default_factory=dict)
    func: object = None


def _value(v):
    v = v.strip()
    if v == _DROP:
        return _DROP
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    if v.lower() == "none":
        return None
    if "/" in v:
        return tuple(_value(p) for p in v.split("/"))
    for t in (int, float):
        try:
            return t(v)
        except ValueError:
            pass
    return v


def _load_func(target):
    mod, _, fn = target.rpartition(":")
    if not mod or not fn:
        raise SystemExit(f"py variant {target!r}: expected py:<module | file.py>:<function>")
    if mod.endswith(".py"):
        path = os.path.abspath(mod)
        sys.path.insert(0, os.path.dirname(path))  # the script's sibling imports
        # a private module name: a script called e.g. profile.py must not shadow the stdlib
        name = "a8w4_bench_user_" + os.path.splitext(os.path.basename(path))[0]
        spec = importlib.util.spec_from_file_location(name, path)
        m = importlib.util.module_from_spec(spec)
        sys.modules[name] = m
        spec.loader.exec_module(m)
    else:
        m = importlib.import_module(mod)
    return getattr(m, fn)


def parse_variant(text):
    text = text.strip()
    label = None
    eq, colon = text.find("="), text.find(":")
    if eq > 0 and (colon < 0 or eq < colon):
        label, text = text[:eq].strip(), text[eq + 1:].strip()
    if text.startswith("py:"):
        func = _load_func(text[3:])
        return Variant(label or text, text, base="py", func=func)
    base, _, ov = text.partition(":")
    base = base.strip()
    if base in EG.CFGS:
        base = f"ep3k {base}"
    if base not in ("@table", "@auto"):
        try:
            cfg = MOE.parse(base)
        except (IndexError, KeyError, ValueError):
            cfg = None
        if cfg is None or (cfg["kind"] == "ep3k" and cfg["cfg"] not in EG.CFGS):
            raise SystemExit(f"variant {text!r}: {base!r} is not a config name\n\n{VARIANT_HELP}")
    v = Variant(label or text, text, base=base)
    for item in filter(None, (s.strip() for s in ov.split(","))):
        k, eq2, val = item.partition("=")
        if not eq2:
            raise SystemExit(f"variant {text!r}: override {item!r} is not key=value")
        v.overrides[k.strip()] = _value(val)
    if v.overrides and not base.startswith("ep3k"):
        raise SystemExit(f"variant {text!r}: overrides apply to ep3k configs only")
    return v


@dataclasses.dataclass
class Resolved:
    """A variant at one token count: ``run(x, W, tw, ti)`` -> bf16 out."""
    label: str
    spec: str
    config: str
    run: object
    cfg: object = None  # the config dict, overrides included (None: py variant)
    bm: object = None  # the implementation's @meta tile_m (rows per block; None: none)
    cached: object = None  # @meta cached_loads: reads W / A with cached loads (None: unknown)


def resolve(v, case, tokens):
    """-> Resolved, or a str: why ``v`` cannot run ``tokens`` on ``case``. The overrides
    of an ep3k variant become config entries: ``moe_ep3k(**over)`` applies them, and its
    ``@requires`` / ``@meta`` see them."""
    kw = dict(expert_offset=case.e0, num_experts=case.shape[3], act=case.act, situ=case.situ)
    if v.func is not None:
        f = v.func
        return Resolved(v.label, v.spec, v.spec, lambda x, W, tw, ti: f(x, W, tw, ti, **kw))
    try:
        if v.base == "@table":
            cfg = MOE.pick(shape_vocab(case.shape, tokens, case.act, case.routing))
        elif v.base == "@auto":
            cfg = MOE.pick(shape_vocab(case.shape, tokens, case.act), table=None)
        else:
            cfg = MOE.parse(v.base)
    except ValueError as e:  # no config can run the case
        return str(e)
    cfg = {**cfg, **v.overrides}
    why = MOE.why_not(cfg, shape_vocab(case.shape, tokens, case.act))
    if why:
        return f"{MOE.name_of(cfg)}: {why}"
    return Resolved(v.label, v.spec, MOE.name_of(cfg),
                    lambda x, W, tw, ti: MOE.run(cfg, x, W, tw, ti, **kw), cfg=cfg,
                    bm=MOE.meta(cfg, "tile_m"), cached=MOE.meta(cfg, "cached_loads"))


def check_protocols(resolved, protocols, wcopies, allow_biased):
    """Refuse to time configs with cached loads under protocols that keep the weights
    in the MALL only (the 20261115 mistake: a whole retune on a biased protocol)."""
    safe = "coldrot" in protocols and wcopies >= MIN_COPIES
    risky = sorted({r.label for r in resolved if r.cached is not False})
    if safe or not risky:
        return
    msg = (f"variants {risky} read W / A with cached loads (or are py variants: unknown), "
           f"but no protocol of {list(protocols)} rotates >= {MIN_COPIES} weight copies "
           f"(coldrot, --wcopies {wcopies}): the MALL keeps the weights between calls and "
           "favors cached loads by 5-13% (experiments/log_20261115.md 3.3)")
    if not allow_biased:
        raise SystemExit(msg + ". Add coldrot, or pass --allow-biased to measure the bias.")
    print("WARNING: " + msg, flush=True)


# =============================================================================
# timing
# =============================================================================


class Timer:
    """Call times of ``fn(W)`` under a protocol; ``Ws``: weight copies, Ws[0] the original."""

    def __init__(self, Ws):
        self.Ws, self._i, self._buf = Ws, 0, {}

    def _next(self):
        self._i = (self._i + 1) % len(self.Ws)
        return self.Ws[self._i]

    def _pre(self, protocol):
        if protocol in ("coldrot", "cold"):
            R.flush_cache()
        elif protocol == "wflush":
            if "w" not in self._buf:
                self._buf["w"] = torch.empty(512 << 20, dtype=torch.uint8, device="cuda")
            self._buf["w"].zero_()
            R.flush_cache()
        elif protocol == "bigread":
            if "r" not in self._buf:
                self._buf["r"] = torch.ones(512 << 20, dtype=torch.int32, device="cuda")
            self._buf["r"].amax()
        else:
            assert protocol == "warm", protocol

    def times(self, fn, protocol, iters, warmup=3):
        """``iters`` GPU times (us) of ``fn(W)``; host launch overhead excluded (a GPU sleep
        lets the host enqueue fn before the start event)."""
        if protocol == "rotate":
            return self._rotate(fn, iters)
        for _ in range(warmup):
            fn(self.Ws[0])
        torch.cuda.synchronize()
        out = []
        for _ in range(iters):
            W = self._next() if protocol == "coldrot" else self.Ws[0]
            self._pre(protocol)
            torch.cuda._sleep(2_000_000)  # ~1 ms
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            fn(W)
            e1.record()
            torch.cuda.synchronize()
            out.append(e0.elapsed_time(e1) * 1e3)
        return out

    def _rotate(self, fn, iters):
        out, n = [], len(self.Ws)
        for p in range(-(-iters // n) + 1):  # pass 0: warm-up
            ev = [torch.cuda.Event(enable_timing=True) for _ in range(n + 1)]
            torch.cuda._sleep(4_000_000)
            ev[0].record()
            for i, W in enumerate(self.Ws):
                fn(W)
                ev[i + 1].record()
            torch.cuda.synchronize()
            if p:
                out += [ev[i].elapsed_time(ev[i + 1]) * 1e3 for i in range(n)]
        return out


def interleaved(fns, protocols, timer, rounds, iters):
    """{(protocol, label): [median of each round]}: each round times every (protocol,
    variant) once; the variant order reverses every other round (ABBA), so clock and
    power drift hit all variants alike."""
    res = {(p, n): [] for p in protocols for n in fns}
    for rnd in range(rounds):
        names = list(fns) if rnd % 2 == 0 else list(fns)[::-1]
        for p in protocols:
            for n in names:
                res[p, n].append(statistics.median(timer.times(fns[n], p, iters)))
    return res


def noise_band(base, dup=None):
    """% band within which a difference to ``base`` (round medians) is noise: the dup's
    |delta|, half the base's round spread, at least MIN_NOISE."""
    b = statistics.median(base)
    band = (max(base) - min(base)) / b * 50 if len(base) > 1 else 0.0
    if dup:
        band = max(band, abs(statistics.median(dup) / b - 1) * 100)
    return max(band, MIN_NOISE)


def verdict(delta, noise):
    return "~" if abs(delta) <= noise else ("faster" if delta < 0 else "slower")


# =============================================================================
# run metadata and result files
# =============================================================================


def git_rev(path=PKG_DIR):
    """Short HEAD of the repo holding ``path`` (+dirty: uncommitted changes outside
    experiments/)."""
    try:
        run = lambda *a: subprocess.run(["git", "-C", path, *a], capture_output=True,  # noqa: E731
                                        text=True, timeout=10).stdout.strip()
        rev = run("rev-parse", "--short", "HEAD")
        if not rev:
            return None
        dirty = run("status", "--porcelain", "--", ".", ":(exclude)experiments")
        return rev + ("+dirty" if dirty else "")
    except (OSError, subprocess.SubprocessError):
        return None


def power_cap():
    vis = (os.environ.get("HIP_VISIBLE_DEVICES") or os.environ.get("ROCR_VISIBLE_DEVICES")
           or "0").split(",")[0]
    try:
        out = subprocess.run(["rocm-smi", "-d", vis, "--showmaxpower", "--json"],
                             capture_output=True, text=True, timeout=30).stdout
        for card in json.loads(out).values():
            for k, v in card.items():
                if "Max Graphics Package Power" in k:
                    return float(v)
    except (OSError, ValueError, subprocess.SubprocessError, AttributeError):
        pass
    return None


def run_meta(tool, args=None):
    import flydsl

    p = torch.cuda.get_device_properties(torch.cuda.current_device())
    return dict(
        tool=tool, time=datetime.datetime.now().isoformat(timespec="seconds"),
        cmd=shlex.join(["python", "-m", "a8w4.bench", tool]
                       + sys.argv[1 + (sys.argv[1:2] == [tool]):]),
        cwd=os.getcwd(), host=socket.gethostname(), git=git_rev(),
        torch=torch.__version__, flydsl=getattr(flydsl, "__version__", "?"),
        gpu=p.name, arch=p.gcnArchName.split(":")[0], cus=p.multi_processor_count,
        visible=os.environ.get("HIP_VISIBLE_DEVICES"), power_cap_w=power_cap(),
        args=dict(vars(args)) if args is not None else {})


def meta_line(m):
    cap = f", cap {m['power_cap_w']:.0f} W" if m.get("power_cap_w") else ""
    return (f"> {m['time']} | git {m['git']} | FlyDSL {m['flydsl']} | torch {m['torch']} | "
            f"{m['gpu']} ({m['arch']}, {m['cus']} CU{cap}) | HIP_VISIBLE_DEVICES "
            f"{m['visible']}\n> `{m['cmd']}`")


class Recorder:
    """Prints text; with ``out``: also appends it to <out>/<name>.md and the measurement
    records to <out>/<name>.jsonl (each run starts with a {"meta": ...} line)."""

    def __init__(self, out, name, meta):
        self.md = self.jl = None
        if out:
            os.makedirs(out, exist_ok=True)
            self.md = os.path.join(out, f"{name}.md")
            self.jl = os.path.join(out, f"{name}.jsonl")
            with open(self.jl, "a") as f:
                f.write(json.dumps({"meta": meta}, default=str) + "\n")
        self.text("\n" + meta_line(meta) + "\n")

    def text(self, s):
        print(s, flush=True)
        if self.md:
            with open(self.md, "a") as f:
                f.write(s + "\n")

    def row(self, **rec):
        if self.jl:
            with open(self.jl, "a") as f:
                f.write(json.dumps(rec, default=str) + "\n")

    def done(self):
        if self.md:
            print(f"(written: {self.md}, {self.jl})", flush=True)


def md_table(head, rows, right=()):
    """Markdown table; columns whose index is in ``right`` are right-aligned."""
    out = ["| " + " | ".join(head) + " |",
           "|" + "|".join("---:" if i in right else "---" for i in range(len(head))) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def fmt_cos(c):
    return "-" if c is None else ("0" if c == 0 else f"{c:.1e}")


def ref_out(case, x, tw, ti, raw):
    return R.ref_moe(x, tw, ti, raw, case.e0, case.act, case.situ)
