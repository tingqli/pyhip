"""A/B timing and correctness of an op's configs by name (torch; ``python -m pyhip.kit``).

    python -m pyhip.kit ab    a8w4:moe ep8-silu -t 1024 4096 -v "ep3k m64" "pf1=ep3k m64:g2.pf=1"
    python -m pyhip.kit check a8w4:moe tp8-silu -t 1 77 4096 -v @table "tp1k bcm2 ks1" --graph

The case and the inputs are ``kit.tune``'s: a tune case (``--set`` overrides), the bucket
value per ``-t``, ``<op>_test`` with seed ``--seed`` + value (+ 1000 * sample). The
answer is ``<op>_ref``, the comparison ``<op>_check`` (its first metric is the column
reported, e.g. cos). The timed calls cycle through copies of the registry's ``rotate``
inputs (``kit.timing``).

variant spec: [label=]<base>[:<k>=<v>,...]  or  [label=]py:<module | file.py>:<func>
  base      a config name (``python -m pyhip.kit candidates <op> <case>``) or "@table" (the
            tuned table's pick at each bucket value)
  k=v       overrides passed to a solver with ``**over`` (v: int, float, true /
            false, none, a/b/c a tuple, "-", anything else a string)
  py        func(**the interface arguments) -> the output: any pipeline, e.g. an
            experiment's kernel copy
  label     the name in the tables (default: the spec)

ab: per bucket value and sample, every variant first runs once against ``<op>_ref``
(fails the check: BAD, raises: FAIL; both dropped) and against the first variant, the
base. Then ``--rounds`` ABBA rounds time ``--iters`` calls of every (protocol, variant).
The base also runs as "<base>#2": that copy's difference is the noise floor; a
difference gets a verdict only beyond the noise band (``kit.timing.noise_band``).
Implementations with ``@meta(cached_loads=True)`` (they read the rotated inputs through
the cache) are refused unless a protocol rotates >= MIN_COPIES copies.

check: per bucket value, every variant vs ``<op>_ref``, vs the base, the worst of
``--repeats`` further calls vs the first, bit equality with the base (``--bitwise``) and
CUDA-graph capture + replays vs eager (``--graph``; ``@no_capture`` solvers:
n/a). ``--diff-tol``: the largest metric vs the base / repeat / graph that passes.
"""

import argparse
import dataclasses
import datetime
import importlib
import importlib.util
import json
import os
import shlex
import socket
import statistics
import subprocess
import sys

import torch

from .timing import MIN_COPIES, MIN_NOISE, NCOPIES, PROTOCOLS, Timer, copies, interleaved
from .timing import noise_band, verdict
from .tune import _bucket, _check, _inputs, case, resolve as resolve_op

__all__ = ["Variant", "Resolved", "parse_variant", "resolve", "check_protocols", "ab", "check",
           "run_meta", "meta_line", "Recorder", "md_table", "fmt", "main"]


# ---- variants ------------------------------------------------------------------------

@dataclasses.dataclass
class Variant:
    label: str
    spec: str
    base: str = ""
    overrides: dict = dataclasses.field(default_factory=dict)
    func: object = None


@dataclasses.dataclass
class Resolved:
    """A variant at one bucket value: ``run(inputs)`` -> the output."""
    label: str
    spec: str
    config: str
    run: object
    cfg: object = None  # the config dict, overrides included (None: py variant)
    cached: object = None  # @meta cached_loads (None: unknown)


_DROP = "-"


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
        name = "kit_bench_user_" + os.path.splitext(os.path.basename(path))[0]
        spec = importlib.util.spec_from_file_location(name, path)
        m = importlib.util.module_from_spec(spec)
        sys.modules[name] = m
        spec.loader.exec_module(m)
    else:
        m = importlib.import_module(mod)
    return getattr(m, fn)


def parse_variant(op, text, alias=None):
    """A variant spec (see the module docstring) -> Variant. ``alias``: base name ->
    config name (e.g. short names of a package)."""
    text = text.strip()
    label = None
    eq, colon = text.find("="), text.find(":")
    if eq > 0 and (colon < 0 or eq < colon):
        label, text = text[:eq].strip(), text[eq + 1:].strip()
    if text.startswith("py:"):
        return Variant(label or text, text, base="py", func=_load_func(text[3:]))
    base, _, ov = text.partition(":")
    base = base.strip()
    base = alias(base) if alias else base
    v = Variant(label or text, text, base=base)
    for item in filter(None, (s.strip() for s in ov.split(","))):
        k, eq2, val = item.partition("=")
        if not eq2:
            raise SystemExit(f"variant {text!r}: override {item!r} is not key=value")
        v.overrides[k.strip()] = _value(val)
    if base != "@table":
        try:
            op.name_of({**op.parse(base), **v.overrides})
        except (KeyError, TypeError, ValueError) as e:
            raise SystemExit(f"variant {text!r}: {e}") from None
    return v


def resolve(op, v, dims):
    """-> Resolved, or a str: why ``v`` has no config for ``dims`` (an untuned key for
    ``@table``). The overrides become config entries (``**over`` of the solver; its
    ``@meta`` sees them). A config the solver cannot run raises at its first
    run."""
    if v.func is not None:
        f = v.func
        return Resolved(v.label, v.spec, v.spec, lambda inputs: f(**inputs))
    try:
        cfg = op.pick(dims) if v.base == "@table" else op.parse(v.base)
    except ValueError as e:  # no tuned config for the case
        return str(e)
    cfg = {**cfg, **v.overrides}
    try:
        cached = op.meta(cfg, "cached_loads", dims)
    except KeyError:
        cached = None
    except TypeError as e:  # overrides on a solver without **over
        return str(e)
    return Resolved(v.label, v.spec, op.name_of(cfg),
                    lambda inputs, cfg=cfg: op.run(cfg, **inputs), cfg=cfg, cached=cached)


def check_protocols(resolved, protocols, ncopies, allow_biased):
    """Refuse to time variants with cached loads under protocols that keep the inputs in
    the MALL only (a whole retune once ran on such a protocol)."""
    safe = "coldrot" in protocols and ncopies >= MIN_COPIES
    risky = sorted({r.label for r in resolved if r.cached is not False})
    if safe or not risky:
        return
    msg = (f"variants {risky} read the rotated inputs with cached loads (or are py variants / "
           f"lack @meta(cached_loads=...): unknown), but no protocol of {list(protocols)} "
           f"rotates >= {MIN_COPIES} copies (coldrot; copies: {ncopies}): the MALL keeps the "
           "inputs between calls and favors cached loads by 5-13%")
    if not allow_biased:
        raise SystemExit(msg + ". Add coldrot, or pass --allow-biased to measure the bias.")
    print("WARNING: " + msg, flush=True)


# ---- run records -----------------------------------------------------------------------

def git_rev(path=None):
    """Short HEAD of the repo holding ``path`` (default: the working directory), +dirty
    for uncommitted changes outside experiments/."""
    path = path or os.getcwd()
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


def run_meta(tool, args=None, prog="python -m pyhip.kit", path=None):
    """The run's metadata: command, git revision of ``path``, versions, GPU, power cap."""
    try:
        import flydsl
        flydsl_v = getattr(flydsl, "__version__", "?")
    except ImportError:
        flydsl_v = None
    p = torch.cuda.get_device_properties(torch.cuda.current_device())
    return dict(
        tool=tool, time=datetime.datetime.now().isoformat(timespec="seconds"),
        cmd=shlex.join([*prog.split(), tool] + sys.argv[1 + (sys.argv[1:2] == [tool]):]),
        cwd=os.getcwd(), host=socket.gethostname(), git=git_rev(path),
        torch=torch.__version__, flydsl=flydsl_v,
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


def fmt(c):
    return "-" if c is None else ("0" if c == 0 else f"{c:.1e}")


# ---- engine ------------------------------------------------------------------------------

def _metric(m):
    """(name, value) of the first metric of a check's result; (None, None) without one."""
    return next(iter(m.items()), (None, None)) if m else (None, None)


def _clone(out):
    return out.clone() if torch.is_tensor(out) else out


def _same(a, b):
    return bool(torch.equal(a, b)) if torch.is_tensor(a) else a == b


class _Inputs:
    """inputs(value, seed) -> the interface arguments: ``<op>_test`` on the case with the
    bucket set, or the caller's function."""

    def __init__(self, op, c, fn=None):
        self.op, self.c, self.fn, self.b = op, c, fn, _bucket(op)

    def dims(self, t):
        return dict(self.c, **{self.b: t})

    def __call__(self, t, seed):
        return self.fn(t, seed) if self.fn else _inputs(self.op, self.dims(t), seed)


def _rotated(op, inputs, n):
    """[{name: copy}] of the registry's rotate inputs (the first: the inputs' own)."""
    names = op.registry.rotate
    cs = {k: copies(inputs[k], n) for k in names}
    return [{k: cs[k][i] for k in names} for i in range(n)]


def _correctness(op, res, inputs, ref, bitwise):
    """One call of every variant -> (variants that passed, {label: (ref metric, base
    metric, bitwise) or why it was dropped}, the metric's name)."""
    expected = op.ref(**inputs) if ref else None
    base, ok, notes, name = None, [], {}, None
    for r in res:
        try:
            out = r.run(inputs)
            torch.cuda.synchronize()
        except Exception as ex:  # noqa: BLE001 - report the variant, keep the others
            notes[r.label] = f"FAIL {type(ex).__name__}: {str(ex)[:200]}"
            continue
        c = None
        if expected is not None:
            try:
                name, c = _metric(_check(op, out, expected, inputs))
            except AssertionError as ex:
                notes[r.label] = f"BAD vs reference: {ex}"
                continue
        if base is None:
            if r is not res[0]:
                raise SystemExit(f"the base {res[0].label!r} failed: {notes[res[0].label]}")
            base = _clone(out)
        try:
            name2, cb = _metric(_check(op, out, base, inputs))
            name = name or name2
        except AssertionError:
            cb = float("inf")
        notes[r.label] = (c, cb, _same(base, out) if bitwise else None)
        ok.append(r)
    return ok, notes, name


def ab(op, c, points, specs, *, protocols=("coldrot",), rounds=5, iters=16, samples=1, seed=7,
       dup=True, allow_biased=False, ref=True, bitwise=False, diff_tol=None, ncopies=NCOPIES,
       rec=None, inputs=None, annotate=None, alias=None, unit=None, describe=None):
    """A/B timing of the variants ``specs`` on case ``c`` (see the module docstring).
    Hooks: ``inputs(value, seed)`` -> interface arguments (default ``<op>_test``);
    ``annotate(inputs, resolved)`` -> (heading text, record fields) of a sample;
    ``alias`` for base names; ``unit``: the bucket's name in headings."""
    rec = rec or Recorder(None, None, run_meta("ab"))
    gen = _Inputs(op, c, inputs)
    unit = unit or gen.b
    variants = [parse_variant(op, s, alias) for s in specs]
    labels = [v.label for v in variants]
    if len(set(labels)) < len(labels):
        raise SystemExit(f"duplicate variant labels {labels}: name them (label=spec)")
    base_label = labels[0]
    dup_label = None
    if dup:
        dup_label = base_label + "#2"
        variants.insert(1, dataclasses.replace(variants[0], label=dup_label))
    plan = {T: [resolve(op, v, gen.dims(T)) for v in variants] for T in points}
    for T, res in plan.items():  # before any timing: fail fast
        if isinstance(res[0], str):
            raise SystemExit(f"the base at {unit} {T}: {res[0]}")
        check_protocols([r for r in res if not isinstance(r, str)], protocols, ncopies,
                        allow_biased)
    rot = [p for p in protocols if p in ("coldrot", "rotate")]
    rec.text((describe or f"{op.name} {c}") + f"\n\nprotocol {', '.join(protocols)}"
             + (f" ({ncopies} copies of {list(op.registry.rotate)})" if rot else "")
             + f", {rounds} rounds x {iters} calls; us = median of the round medians; "
             f"verdict only beyond the noise band (max(|{dup_label or 'copy'} delta|, half the "
             f"base's round spread, {MIN_NOISE}%))\n")
    agg, cfgs, timer = {}, {}, None
    for T in points:
        for s in range(samples):
            sd = seed + T + 1000 * s
            inp = gen(T, sd)
            if timer is None:  # the rotated inputs are the same objects at every value
                timer = Timer(_rotated(op, inp, ncopies) if rot
                              else [{k: inp[k] for k in op.registry.rotate}])
            res = [r for r in plan[T] if not isinstance(r, str)]
            ok, notes, mname = _correctness(op, res, inp, ref, bitwise)
            fns = {r.label: (lambda Wc, r=r: r.run({**inp, **Wc})) for r in ok}
            times = interleaved(fns, protocols, timer, rounds, iters)
            extra_head, extra_rec = annotate(inp, ok) if annotate else ("", {})
            head = (f"#### {T} {unit}" + (f", sample {s}" if samples > 1 else "")
                    + f" (seed {sd})" + extra_head)
            rows, verdicts = [], {}
            m = mname or "check"
            for p in protocols:
                b = times[p, base_label]
                bmed = statistics.median(b)
                noise = noise_band(b, times.get((p, dup_label)))
                for r in ok:
                    rr = times[p, r.label]
                    med = statistics.median(rr)
                    d = (med / bmed - 1) * 100
                    v = ("base" if r.label == base_label else
                         "noise floor" if r.label == dup_label else verdict(d, noise))
                    verdicts[p, r.label] = (d, v)
                    cr, cb, bw = notes[r.label]
                    flag = diff_tol is not None and cb is not None and cb > diff_tol
                    rows.append([r.label, f"`{r.config}`", p, f"{med:.1f}", f"{d:+.1f}",
                                 f"{min(rr):.1f}-{max(rr):.1f}", v, fmt(cr),
                                 fmt(cb) + (" DIFF" if flag else "")]
                                + (["yes" if bw else "no"] if bitwise else []))
                    agg.setdefault((p, r.label, T), []).append((med, noise))
                    cfgs.setdefault(r.label, set()).add(r.config)
                    rec.row(T=T, sample=s, seed=sd, protocol=p, variant=r.label, spec=r.spec,
                            config=r.config, us=med, rounds=rr, delta_pct=d, noise_pct=noise,
                            verdict=v, **{f"{m}_ref": cr, f"{m}_base": cb}, bitwise=bw,
                            cached=r.cached, copies=ncopies if p in rot else 1, **extra_rec)
                rows.append([f"*noise band +-{noise:.1f}%*"] + [""] * (len(rows[-1]) - 1))
            table = md_table(
                ["variant", "config", "protocol", "us", "delta %", "rounds min-max", "verdict",
                 f"{m} ref", f"{m} base"] + (["bitwise"] if bitwise else []), rows,
                right=(3, 4, 5, 7, 8))
            drop = [f"- {v.label}: {r}" for v, r in zip(variants, plan[T]) if isinstance(r, str)]
            drop += [f"- {n}: {why}" for n, why in notes.items() if isinstance(why, str)]
            rec.text(head + "\n\n" + table + ("\n\ndropped:\n" + "\n".join(drop) if drop else ""))
            if len(protocols) > 1:
                sens = []
                for r in ok:
                    vs = {p: verdicts[p, r.label] for p in protocols}
                    if r.label not in (base_label, dup_label) and len({v for _, v in vs.values()}) > 1:
                        sens.append(f"- {r.label}: " + ", ".join(
                            f"{p} {d:+.1f}% ({v})" for p, (d, v) in vs.items())
                            + " -> PROTOCOL-SENSITIVE: trust coldrot")
                rec.text("\nprotocols agree on every verdict" if not sens else
                         "\n" + "\n".join(sens))
            rec.text("")
    if len(points) * samples > 1:
        order = labels[:1] + ([dup_label] if dup_label else []) + labels[1:]
        for p in protocols:
            rows = []
            for lab in order:
                if lab not in cfgs:
                    continue
                cells = []
                for T in points:
                    pts, bpts = agg.get((p, lab, T)), agg.get((p, base_label, T))
                    if not pts:
                        cells.append("-")
                        continue
                    us = statistics.fmean(m for m, _ in pts)
                    if lab == base_label:
                        cells.append(f"{us:.1f} us")
                        continue
                    d = (us / statistics.fmean(m for m, _ in bpts) - 1) * 100
                    noise = max(n for _, n in bpts)
                    cells.append(f"{d:+.1f}%" + ("~" if abs(d) <= noise else ""))
                rows.append([lab, " / ".join(f"`{c}`" for c in sorted(cfgs[lab]))] + cells)
            rec.text(f"#### summary ({p}): base in us, the others delta vs the base; "
                     "~ = within the noise band\n\n"
                     + md_table(["variant", "config"] + [str(T) for T in points], rows,
                                right=tuple(range(2, 2 + len(points)))) + "\n")
    rec.done()


def _stability(op, fn, inputs, repeats):
    """Worst metric of ``repeats`` further calls of ``fn()`` vs the first."""
    first = _clone(fn())
    return max((_metric(_check(op, fn(), first, inputs))[1] or 0.0 for _ in range(repeats)),
               default=0.0)


def _graph_replay(op, fn, inputs, replays=5):
    """Capture ``fn()`` in a CUDA graph (after a side-stream warm-up, as frameworks do) and
    replay it -> worst metric of the replays vs an eager call."""
    eager = _clone(fn())
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = fn()
    worst = 0.0
    for _ in range(replays):
        g.replay()
        torch.cuda.synchronize()
        worst = max(worst, _metric(_check(op, out, eager, inputs))[1] or 0.0)
    return worst


def check(op, c, points, specs, *, seed=7, repeats=5, graph=False, bitwise=False,
          diff_tol=None, rec=None, inputs=None, alias=None, unit=None, describe=None):
    """Correctness of the variants ``specs`` on case ``c`` (see the module docstring).
    Returns the number of failed (value, variant) items."""
    rec = rec or Recorder(None, None, run_meta("check"))
    gen = _Inputs(op, c, inputs)
    unit = unit or gen.b
    variants = [parse_variant(op, s, alias) for s in specs]
    rec.text((describe or f"{op.name} {c}") + "\n")
    rows, bad, m = [], 0, None
    for T in points:
        inp = gen(T, seed + T)
        expected = op.ref(**inp)
        base = None
        for v in variants:
            r = resolve(op, v, gen.dims(T))
            if isinstance(r, str):
                rows.append([T, v.label, "-", "n/a", "", "", "", "", r])
                continue
            fn = lambda r=r: r.run(inp)  # noqa: E731
            cr = cb = jit = g = bw = None
            fails = []
            try:
                out = fn()
                torch.cuda.synchronize()
                try:
                    m, cr = _metric(_check(op, out, expected, inp))
                except AssertionError:
                    fails.append("ref")
                if base is None:
                    base = _clone(out)
                try:
                    cb = _metric(_check(op, out, base, inp))[1]
                except AssertionError:
                    fails.append("base")
                jit = _stability(op, fn, inp, repeats)
                bw = _same(base, out) if bitwise else None
                if graph and r.cfg is not None and not op.capturable(r.cfg):
                    g = "n/a (not graph-safe)"
                elif graph:
                    g = _graph_replay(op, fn, inp)
                tol = diff_tol if diff_tol is not None else float("inf")
                fails += [s for s, f in (("base", (cb or 0) > tol), ("repeat", (jit or 0) > tol),
                                         ("graph", isinstance(g, float) and g > tol),
                                         ("bitwise", bitwise and not bw)) if f]
                ok = "OK" if not fails else "FAIL " + ",".join(dict.fromkeys(fails))
            except Exception as ex:  # noqa: BLE001
                ok = f"FAIL {type(ex).__name__}: {str(ex)[:160]}"
            bad += ok != "OK"
            rows.append([T, r.label, f"`{r.config}`", fmt(cr), fmt(cb), fmt(jit),
                         "-" if bw is None else ("yes" if bw else "no"),
                         fmt(g) if isinstance(g, float) else (g or "-"), ok])
            rec.row(T=T, variant=r.label, config=r.config, **{f"{m or 'check'}_ref": cr,
                                                              f"{m or 'check'}_base": cb},
                    repeat=jit, bitwise=bw, graph=g, result=ok)
        torch.cuda.empty_cache()
    m = m or "check"
    rec.text(md_table([unit, "variant", "config", f"{m} ref", f"{m} base", "repeat", "bitwise",
                       "graph", "result"], [[str(x) for x in r] for r in rows],
                      right=(0, 3, 4, 5)))
    rec.text("\nALL OK" if not bad else f"\n{bad} FAILED")
    rec.done()
    return bad


# ---- command line ------------------------------------------------------------------------

def parser(prog="python -m pyhip.kit"):
    ap = argparse.ArgumentParser(prog=prog, description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["ab", "check"])
    ap.add_argument("op", help="<module>:<op>")
    ap.add_argument("case", help="a tune case (python -m pyhip.kit cases <op>)")
    ap.add_argument("-t", "--points", nargs="+", required=True, help="bucket values")
    ap.add_argument("-v", "--variants", nargs="+", default=["@table"],
                    help="variant specs; the first is the base (default: @table)")
    ap.add_argument("--set", nargs="+", default=[], metavar="NAME=VALUE")
    ap.add_argument("--seed", type=int, default=7, help="input seed = --seed + value")
    ap.add_argument("--diff-tol", type=float, default=None,
                    help="largest metric vs the base / repeat / graph that passes")
    ap.add_argument("--bitwise", action="store_true", help="also bit equality with the base")
    g = ap.add_argument_group("ab")
    g.add_argument("--protocol", nargs="+", default=["coldrot"], choices=PROTOCOLS)
    g.add_argument("--copies", type=int, default=NCOPIES)
    g.add_argument("--rounds", type=int, default=5)
    g.add_argument("--iters", type=int, default=16, help="timed calls per round")
    g.add_argument("--samples", type=int, default=1,
                   help="input samples per value (seed + 1000 * sample)")
    g.add_argument("--no-dup", action="store_true", help="do not time the base twice")
    g.add_argument("--allow-biased", action="store_true")
    g.add_argument("--no-ref", action="store_true", help="skip the reference")
    g = ap.add_argument_group("check")
    g.add_argument("--repeats", type=int, default=5)
    g.add_argument("--graph", action="store_true", help="also CUDA-graph capture + replay")
    g = ap.add_argument_group("output")
    g.add_argument("--out", default=None, help="directory: append to <out>/<name>.md and "
                                               "<name>.jsonl")
    g.add_argument("--name", default=None, help="file name stem (default: the command)")
    return ap


def main(argv=None, prog="python -m pyhip.kit"):
    a = parser(prog).parse_args(argv)
    from . import _guess

    op = resolve_op(a.op)
    c = case(op, a.case, a.set)
    pts = [_guess(v) for v in a.points]
    rec = Recorder(a.out, a.name or a.cmd, run_meta(a.cmd, a, prog))
    desc = f"{op.name} case {a.case}: {({k: v for k, v in c.items() if v is not None})}"
    if a.cmd == "ab":
        return ab(op, c, pts, a.variants, protocols=a.protocol, rounds=a.rounds, iters=a.iters,
                  samples=a.samples, seed=a.seed, dup=not a.no_dup, allow_biased=a.allow_biased,
                  ref=not a.no_ref, bitwise=a.bitwise, diff_tol=a.diff_tol, ncopies=a.copies,
                  rec=rec, describe=desc)
    return 1 if check(op, c, pts, a.variants, seed=a.seed, repeats=a.repeats, graph=a.graph,
                      bitwise=a.bitwise, diff_tol=a.diff_tol, rec=rec, describe=desc) else 0
