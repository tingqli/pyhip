"""Tune an op into its tuned table, and inspect the table (torch; ``python -m pyhip.kit``).

    python -m pyhip.kit problems   a8w4.tune:TUNING                # groups and their problems
    python -m pyhip.kit configs    a8w4.tune:TUNING mimo-flash2.6-tp8-silu   # every config
    python -m pyhip.kit tune       a8w4.tune:TUNING mimo-flash2.6-tp8 -t 1 2 4 --out /tmp/p0.csv
    python -m pyhip.kit merge      a8w4.tune:TUNING /tmp/p*.csv    # into the tuning's table
    python -m pyhip.kit show       a8w4.tune:TUNING mimo-flash2.6-tp8   # per point: the fastest
    python -m pyhip.kit verify     a8w4.tune:TUNING mimo-flash2.6-ep8-silu -t 1 100 3000
    python -m pyhip.kit tune-all   a8w4.tune:TUNING [PROBLEM ...] --gpus 0-7 --per-job 4 --out /tmp/t
    python -m pyhip.kit compare    OLD.csv NEW.csv

``<module>:<name>``: an importable module and the name of the ``Tuning`` in it that
declares how to tune the op (see ``Tuning``): the problems, the bucket dim and its
points, the key dims the table stores, the inputs the timed calls rotate, the table.
A PROBLEM argument is a problem id or a group name (``add_problem``: all its problems).

``--set name=value`` overrides dim values of a problem (another shape, a routing).
``tune``: per point (default: the problem's), ``--rounds`` rounds, each with fresh
inputs (``<op>_problem``, seed 101 * point + round); per round every config runs once
(one that raises cannot run the problem: n/a), is checked against ``<op>_ref``
(``<op>_check``: a failure is BAD) and timed cold (``kit.timing.time_cold``, the
``rotate`` inputs cycling through ``--copies`` copies), the configs interleaved. Every
config that ran and passed is stored: the median time, the worst metric of the check.
"""

import argparse
import fcntl
import glob
import importlib
import itertools
import json
import os
import re
import subprocess
import sys
import time

import torch

from . import _cell, _guess, _unknown
from . import table as _table
from .timing import NCOPIES, copies, time_cold

__all__ = ["Tuning", "resolve", "tune", "show", "verify", "merge", "tune_all", "main"]

_DEFAULT = object()  # table=: the registry's


class Tuning:
    """How to tune ``op`` (all of it, in one place), declared by chaining:

        TUNING = (Tuning(op, keys=..., rotate=("W",))
                  .set_bucket("num_tokens")
                  .add_problem("mimo-tp8", num_tokens=TOKENS, model_dim=4096, inter_dim=256))

    * ``set_bucket(dim)``: the dim tuned at points, matched inexactly by the dispatch (the
      first tuned point >= the call's value; the other key dims match exactly);
    * ``add_problem(name, **dims)``: a group of problems named ``name``, see there; call
      it once per group;
    * ``keys``: the dims the table stores (default: every dim but the bucket); a dim
      left out must not change which config is fastest (e.g. an expert offset);
    * ``rotate``: interface arguments the timed calls cycle through copies of (e.g.
      the weights: ``kit.timing.copies``);
    * ``table``: the table ``tune`` / ``merge`` write and ``show`` / ``verify`` read
      (default: the registry's, which the dispatch reads).

    ``problems`` {id: problem (the bucket None)}, ``points`` {id: the bucket's values},
    ``groups`` {name: [id, ...]}.
    """

    def __init__(self, op, *, keys=None, rotate=(), table=_DEFAULT):
        where = f"Tuning({op.name})"
        if keys is not None:
            keys = tuple(keys)
            if _unknown(keys, op.dim_names) or len(set(keys)) < len(keys):
                raise TypeError(f"{where}: keys {list(keys)}: distinct dims of "
                                f"{op.name}_problem {list(op.dim_names)}")
        if isinstance(rotate, str) or _unknown(rotate, op._ref_names):
            raise TypeError(f"{where}: rotate {rotate!r}: names of {op.name}_ref's parameters")
        self.op, self.bucket, self._keys, self.rotate = op, None, keys, tuple(rotate)
        self.table = op.registry.table if table is _DEFAULT else table
        self.problems, self.points, self.groups = {}, {}, {}

    def __repr__(self):
        return (f"Tuning({self.op.name}, bucket={self.bucket!r}, groups={list(self.groups)}, "
                f"{len(self.problems)} problems)")

    @property
    def keys(self):
        if self._keys is not None:
            return self._keys
        return tuple(n for n in self.op.dim_names if n != self.bucket)

    @property
    def column(self):
        """The bucket's column: ``~<dim>``."""
        if self.bucket is None:
            raise TypeError(f"{self!r}: no bucket; call set_bucket")
        return _table.BUCKET + self.bucket

    def set_bucket(self, dim):
        """The dim tuned at points (each group's values of it); before ``add_problem``."""
        op, where = self.op, f"Tuning({self.op.name}).set_bucket({dim!r})"
        if dim not in op.dim_names:
            raise TypeError(f"{where}: not a dim {list(op.dim_names)}")
        if self.problems:
            raise TypeError(f"{where}: call it before add_problem")
        if self._keys is not None and dim in self._keys:
            raise TypeError(f"{where}: keys {list(self._keys)} hold the bucket")
        self.bucket = dim
        return self

    def add_problem(self, name, **dims):
        """Add the group ``name``: the Cartesian product of the values of ``dims`` (a list,
        tuple or range: several values; anything else: one) and of the ``@dim`` axes of
        ``<op>_problem`` none of whose dims is given (all their values, e.g. every act). A
        dim neither given nor on such an axis takes its ``<op>_problem`` default. The
        bucket's values are the points of the group's problems (not given: its ``@dim``
        values). A problem's id: ``name``, then "-" and the id of each value that varies
        (its ``@dim`` id, else ``<dim><value>``), in the order of the dims."""
        op, b = self.op, self.bucket
        where = f"Tuning({op.name}).add_problem({name!r})"
        if b is None:
            raise TypeError(f"{where}: call set_bucket first")
        if not re.fullmatch(r"[\w.+-]+", str(name)):
            raise TypeError(f"{where}: a name of letters, digits and _ . + -")
        bad = _unknown(dims, op.dim_names)
        if bad:
            raise TypeError(f"{where}: {bad} are not dims of {op.name}_problem "
                            f"{list(op.dim_names)}")
        vals = {k: tuple(v) if isinstance(v, (list, tuple, range)) else (v,)
                for k, v in dims.items()}
        empty = [k for k, v in vals.items() if not v]
        if empty:
            raise TypeError(f"{where}: no values for {empty}")
        points = vals.pop(b, None)
        if points is None:
            ax = op._axes.get(b)
            if ax is None or len(ax.names) > 1:
                raise TypeError(f"{where}: give the points of the bucket: {b}=[...]")
            points = tuple(d[b] for _, d in ax.points())
        order = op.dim_names.index
        axes, fixed = [], {}  # axes: (position of the first dim, [(id, {dim: value})])
        for k, vs in vals.items():
            if isinstance(dims[k], (list, tuple, range)):
                axes.append((order(k), [(self._id(k, v), {k: v}) for v in vs]))
            else:
                fixed[k] = vs[0]
        for ax in op._axes.values():
            given = [n for n in ax.names if n in dims or n == b]
            if len(given) == len(ax.names):
                continue
            if given:
                raise TypeError(f"{where}: give all of the @dim {ax.name!r} dims "
                                f"{list(ax.names)} or none of them (all its values)")
            axes.append((min(map(order, ax.names)), ax.points()))
        covered = {*dims, b, *(n for ax in op._axes.values() for n in ax.names)}
        missing = [n for n in op.dim_names if n not in covered and n not in op._defaults]
        if missing:
            raise TypeError(f"{where}: give {missing} (no {op.name}_problem default, no @dim)")
        axes.sort(key=lambda a: a[0])
        combos = list(itertools.product(*(pts for _, pts in axes)))
        new = {}
        for combo in combos:
            prob = {**op._blank, **op._defaults, **fixed}
            for _, d in combo:
                prob.update(d)
            new["-".join([name, *(tok for tok, _ in combo)])] = {**prob, b: None}
        if len(new) < len(combos):
            raise TypeError(f"{where}: two problems print as the same id: {list(new)}")
        clash = sorted(({*new, name} & {*self.problems, *self.groups}))
        if clash:
            raise TypeError(f"{where}: {clash} already name a group or a problem")
        self.groups[name] = list(new)
        self.problems.update(new)
        self.points.update(dict.fromkeys(new, points))
        return self

    def _id(self, k, v):
        """The id of value ``v`` of dim ``k`` in problem ids."""
        ax = self.op._axes.get(k)
        if ax is not None and ax.names == (k,):
            try:
                return ax.token({k: v})
            except ValueError:  # a dict @dim without this value
                pass
        return f"{k}{v}"

    def fields(self, metrics):
        """The table header: key columns, the bucket column (``~<dim>``), config, us, the
        check's metrics."""
        return [*self.keys, self.column, "config", "us", *metrics]

    def select(self, names):
        """The problem ids of ``names`` (problem ids or group names), each once."""
        out = []
        for n in names:
            ids = self.groups.get(n) or ([n] if n in self.problems else None)
            if ids is None:
                raise SystemExit(f"{self.op.name}: no tune problem or group {n!r}; groups: "
                                 f"{list(self.groups)}; problems: {list(self.problems)}")
            out += [i for i in ids if i not in out]
        return out

    def problem(self, pid, sets=()):
        """Problem ``pid`` with ``sets`` ("name=value") applied."""
        if pid not in self.problems:
            raise SystemExit(f"{self.op.name}: no tune problem {pid!r}; problems: "
                             f"{list(self.problems)}")
        prob = dict(self.problems[pid])
        for s in sets:
            k, eq, v = s.partition("=")
            if not eq or k not in self.op.dim_names:
                raise SystemExit(f"--set {s!r}: expected name=value, name in "
                                 f"{list(self.op.dim_names)}")
            prob[k] = _cell(v, self.op._defaults.get(k))
        return prob


def resolve(spec):
    """``"module:name"`` -> the ``Tuning`` ``name`` of ``module``."""
    mod, _, name = spec.rpartition(":")
    if not mod or not name:
        raise SystemExit(f"{spec!r}: expected <module>:<Tuning name>, e.g. a8w4.tune:TUNING")
    t = getattr(importlib.import_module(mod), name, None)
    if not isinstance(t, Tuning):
        raise SystemExit(f"{spec}: not a pyhip.kit.tune.Tuning ({type(t).__name__})")
    return t


def _check(op, out, expected, inputs):
    """``<op>_check`` (default: torch.testing.assert_close) -> its metrics dict."""
    if op.check is not None:
        return op.check(out, expected, **inputs) or {}
    if torch.is_tensor(out) and torch.is_tensor(expected):
        expected = expected.to(out.dtype)
    torch.testing.assert_close(out, expected)
    return {}


class _Rotation:
    """Copies of the ``names`` inputs, made once per input object; ``next()`` -> {name:
    the next copy}."""

    def __init__(self, names, n):
        self.names, self.n, self.cyc = names, n, {}

    def bind(self, inputs):
        for k in self.names:
            got = self.cyc.get(k)
            if got is None or got[0] is not inputs[k]:
                self.cyc[k] = (inputs[k], itertools.cycle(copies(inputs[k], self.n)).__next__)
        return self

    def next(self):
        return {k: f() for k, (_, f) in self.cyc.items()}


def _inputs(op, problem, seed):
    return op.problem(**problem, seed=seed) if op._seeded else op.problem(**problem)


def tune(t, problem, points, out, *, rounds=3, iters=20, ncopies=NCOPIES, log=print, pid=""):
    """Tune ``problem`` of ``t`` at ``points`` into the table ``out`` (see the module
    docstring)."""
    op, b = t.op, t.bucket
    known = {k: v for k, v in problem.items() if v is not None and k != b}
    log(f"{op.name} {pid} {known}: {len(op.configs(dict(problem, **{b: points[0]})))} "
        f"configs, {ncopies} copies of {list(t.rotate)}")
    rot = _Rotation(t.rotate, ncopies)
    for p in points:
        prob = dict(problem, **{b: p})
        cfgs = op.configs(prob)
        names = [op.name_of(cfg) for cfg in cfgs]
        times = {n: [] for n in names}
        metrics = {n: {} for n in names}
        bad, na = {}, {}
        for rnd in range(rounds):
            inputs = _inputs(op, prob, 101 * p + rnd)
            rot.bind(inputs)
            expected = op.ref(**inputs)
            for cfg, name in zip(cfgs, names):
                if name in bad or name in na:
                    continue
                run = lambda over, cfg=cfg: op.run(cfg, **{**inputs, **over})  # noqa: E731
                try:
                    res = run({})
                except Exception as ex:  # noqa: BLE001 - the solver cannot run it
                    na[name] = type(ex).__name__
                    continue
                try:
                    m = _check(op, res, expected, inputs)
                except AssertionError as ex:
                    bad[name] = f"BAD {ex}"
                    continue
                for k, v in m.items():
                    metrics[name][k] = max(metrics[name].get(k, v), v)
                times[name].append(time_cold(lambda run=run: run(rot.next()), iters=iters))
        rows = []
        for name in names:
            if name in bad or name in na:
                continue
            us = sorted(times[name])[len(times[name]) // 2]
            rows.append(dict({k: prob[k] for k in t.keys}, **{t.column: p}, config=name,
                             us=f"{us:.1f}", **{k: f"{v:.1e}" for k, v in metrics[name].items()}))
        rows.sort(key=lambda r: float(r["us"]))
        if rows:
            _table.merge(out, rows, t.fields([k for k in rows[0] if k not in
                                              (*t.keys, t.column, "config", "us")]))
        top = " | ".join(f"{r['config']} {r['us']}" for r in rows[:3])
        msgs = [f"{n}: {m}" for n, m in bad.items()]
        errs = sorted(set(na.values()))
        log(f"{b} {p:6}: {top}" + (f"  [{'; '.join(msgs)}]" if msgs else "")
            + (f"  ({len(na)} n/a: {', '.join(errs)})" if na else ""))
        torch.cuda.empty_cache()


def _points(op, problem, path):
    pts = op.points(problem, table=path)
    if not pts:
        known = {k: v for k, v in problem.items() if v is not None}
        raise SystemExit(f"{op.name}: no rows for {known} in {path}")
    return pts


def show(t, problem, path=None, out=print):
    """Markdown, per point: the fastest config, the fastest of each solver and the
    capture pick when it differs."""
    op, b, path = t.op, t.bucket, path or t.table
    pts = _points(op, problem, path)
    solvers = [s for s in op.solvers
               if any(cfg["solver"] == s for _, cs in pts for _, cfg in cs)]
    out(f"| {b} | best | us | " + " | ".join(f"best {s} | us" for s in solvers)
        + " | capture pick |")
    out("|---:|---|---:|" + "---|---:|" * len(solvers) + "---|")
    for n, cfgs in pts:
        us0, best = cfgs[0]
        cells = []
        for s in solvers:
            u, cfg = next(((u, cfg) for u, cfg in cfgs if cfg["solver"] == s), (None, None))
            cells += [op.name_of(cfg), f"{u:.1f}"] if cfg else ["-", "-"]
        try:
            safe = op.pick(dict(problem, **{b: n}), capturing=True, table=path)
            cap = "" if safe == best else op.name_of(safe)
        except ValueError:
            cap = "none"
        out(f"| {n} | {op.name_of(best)} | {us0:.1f} | " + " | ".join(cells) + f" | {cap} |")


def verify(t, problem, points, path=None, *, iters=20, ncopies=NCOPIES, log=print):
    """``op(...)`` (the table's pick) at each bucket value, also between tuned points:
    the check against ``<op>_ref`` and the time vs the table entry."""
    op, b, path = t.op, t.bucket, path or t.table
    pts = dict(_points(op, problem, path))
    rot = _Rotation(t.rotate, ncopies)
    for v in points:
        prob = dict(problem, **{b: v})
        inputs = _inputs(op, prob, 7 + v)
        rot.bind(inputs)
        expected = op.ref(**inputs)
        cfg = op.pick(prob, table=path)
        hints = {k: prob[k] for k in t.keys if k not in op._ref_names}
        fn = lambda over: op(**{**inputs, **over}, table=path, **hints)  # noqa: E731
        try:
            m = _check(op, fn({}), expected, inputs)
            msg = "".join(f" | {k} {x:.1e}" for k, x in m.items())
        except AssertionError as ex:
            msg = f" | BAD {ex}"
        us = time_cold(lambda: fn(rot.next()), iters=iters)
        n = next((p for p in sorted(pts) if p >= v), max(pts))
        tus = next(u for u, cc in pts[n] if cc == cfg)
        log(f"{b} {v:6} -> point {n:6}: {op.name_of(cfg):24s} {us:8.1f} us (table {tus:8.1f})"
            + msg)


def merge(parts, path, log=print):
    """Merge the tables ``parts`` into ``path`` (``kit.table.merge``)."""
    rows, head = [], None
    for p in parts:
        f, rs = _table.read(p)
        head = head or f
        rows += rs
    _table.merge(path, rows, head)
    log(f"merged {len(rows)} rows from {len(parts)} files into {path}")


# ---- several GPUs --------------------------------------------------------------------

def gpu_list(spec):
    """"0-7" / "0,2,4" -> [0, ..., 7] / [0, 2, 4]."""
    out = []
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        out += list(range(int(lo), int(hi or lo) + 1))
    return out


def busy_gpus(gpus, limit=10, samples=3):
    """GPUs of ``gpus`` (physical indices) with > ``limit`` % GFX activity in any of
    ``samples`` rocm-smi readings 1 s apart (a tuner next to another job measures both)."""
    busy = {}
    for i in range(samples):
        out = subprocess.run(["rocm-smi", "--showuse", "--json"], capture_output=True,
                             text=True, timeout=60).stdout
        for card, d in json.loads(out).items():
            g = int(re.sub(r"\D", "", card))
            use = float(d.get("GPU use (%)", 0))
            if g in gpus and use > limit:
                busy[g] = max(busy.get(g, 0.0), use)
        if i + 1 < samples:
            time.sleep(1)
    return busy


def lock_gpus(gpus):
    """Exclusive /tmp/gpu-<i>.lock of every GPU (the convention of aiter's
    validate-kernel-pr: other agents' jobs wait or fail instead of sharing the GPU)."""
    held = []
    for g in gpus:
        fh = open(f"/tmp/gpu-{g}.lock", "w")
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise SystemExit(f"GPU {g} is locked by another job (/tmp/gpu-{g}.lock)")
        held.append(fh)
    return held


def tune_all(spec, t, pids, gpus, per_job, points, out_dir, *, rounds=3, iters=20,
             ncopies=NCOPIES, sets=(), force=False, log=print):
    """Every problem of ``pids`` on ``per_job`` GPUs (GPU g of the group takes
    points[g::per_job]; ``points`` None: each problem's own); groups of ``gpus`` take the
    next problem when they free up. Then the BAD lines of all logs and a merged
    <out_dir>/<table name>.new.csv."""
    if len(gpus) < per_job:
        raise SystemExit(f"--per-job {per_job} needs at least that many --gpus ({gpus})")
    for pid in pids:
        if pid not in t.problems:
            raise SystemExit(f"problem {pid!r}: one of {list(t.problems)}")
    if not force:
        busy = busy_gpus(gpus)
        if busy:
            raise SystemExit(f"GPUs busy (GFX use %): {busy}; tuning next to another job "
                             "skews the table. Wait, pick other --gpus, or --force")
    held = lock_gpus(gpus)  # noqa: F841 - released at exit
    os.makedirs(out_dir, exist_ok=True)
    groups = [gpus[i:i + per_job] for i in range(0, len(gpus) - per_job + 1, per_job)]
    env = dict(os.environ, AITER_LOG_LEVEL=os.environ.get("AITER_LOG_LEVEL", "ERROR"))
    todo, running, failed = list(pids), {}, []
    t0 = time.time()
    while todo or running:
        for gi, procs in list(running.items()):
            if all(p.poll() is not None for _, p in procs):
                failed += [name for name, p in procs if p.returncode]
                del running[gi]
        for gi, grp in enumerate(groups):
            if gi in running or not todo:
                continue
            pid = todo.pop(0)
            pts = list(points or t.points[pid])
            procs = []
            for g, gpu in enumerate(grp):
                shard = pts[g::per_job]
                if not shard:
                    continue
                stem = os.path.join(out_dir, f"{pid}_{g}")
                cmd = [sys.executable, "-m", "pyhip.kit", "tune", spec, pid, "-t",
                       *map(str, shard), "--out", stem + ".csv", "--iters", str(iters),
                       "--rounds", str(rounds), "--copies", str(ncopies)]
                if sets:
                    cmd += ["--set", *sets]
                with open(stem + ".log", "w") as fh:
                    procs.append((stem + ".log", subprocess.Popen(
                        cmd, stdout=fh, stderr=subprocess.STDOUT,
                        env=dict(env, HIP_VISIBLE_DEVICES=str(gpu)))))
            running[gi] = procs
            log(f"{pid}: GPUs {grp}, {len(pts)} points")
        time.sleep(2)
    log(f"all jobs done in {(time.time() - t0) / 60:.1f} min")
    logs = sorted(glob.glob(os.path.join(out_dir, "*_[0-9]*.log")))
    bad = [f"{os.path.basename(f)}: {ln.strip()}" for f in logs for ln in open(f)
           if re.search(r"\bBAD\b", ln)]
    print("\n".join(["BAD configs:"] + bad) if bad else "no BAD configs")
    if failed:
        print(f"FAILED processes (see their logs): {failed}")
    parts = sorted(glob.glob(os.path.join(out_dir, "*_[0-9]*.csv")))
    stem = os.path.splitext(os.path.basename(t.table or f"{t.op.name}.csv"))[0]
    new = os.path.join(out_dir, f"{stem}.new.csv")
    if os.path.exists(new):
        os.remove(new)
    if parts:
        merge(parts, new)
    print(f"next: python -m pyhip.kit compare {t.table} {new}\n"
          f"      cp {new} {t.table}")
    return 1 if failed else 0


# ---- command line ---------------------------------------------------------------------

def parser(prog="python -m pyhip.kit"):
    p = argparse.ArgumentParser(prog=prog, description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["problems", "configs", "tune", "merge", "show", "verify",
                                   "tune-all", "compare"])
    p.add_argument("spec", help="<module>:<Tuning name> (compare: the old table)")
    p.add_argument("args", nargs="*", help="problems or groups (configs / tune / show / "
                                           "verify; tune-all: default all), tables (merge, "
                                           "compare)")
    p.add_argument("-t", "--points", nargs="+", default=None,
                   help="bucket values (default: the problem's points; verify: a few)")
    p.add_argument("--set", nargs="+", default=[], metavar="NAME=VALUE",
                   help="override dim values of the problem")
    p.add_argument("--table", default=None, help="the table (default: the tuning's)")
    p.add_argument("--out", default=None, help="tune: output table (default --table); "
                                               "tune-all: output directory")
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--rounds", type=int, default=3, help="tune: interleaved timing rounds")
    p.add_argument("--copies", type=int, default=NCOPIES,
                   help="copies of the rotate inputs the timed calls cycle through")
    p.add_argument("--gpus", default="0-7", help="tune-all: GPUs, e.g. 0-7 or 0,1,2,3")
    p.add_argument("--per-job", type=int, default=4, help="tune-all: GPUs per problem")
    p.add_argument("--force", action="store_true", help="tune-all: skip the idle-GPU check")
    return p


def main(argv=None, prog="python -m pyhip.kit"):
    a = parser(prog).parse_args(argv)
    if a.cmd == "compare":
        if len(a.args) != 1:
            raise SystemExit("compare OLD.csv NEW.csv")
        return _table.compare(a.spec, a.args[0])
    t = resolve(a.spec)
    op = t.op
    log = lambda s: print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True)  # noqa: E731
    path = a.table or t.table
    pts = [_guess(v) for v in a.points] if a.points else None
    if a.cmd == "problems":
        for g, pids in t.groups.items():
            print(g)
            for pid in pids:
                p = t.points[pid]
                print(f"  {pid} {({k: v for k, v in t.problems[pid].items() if v is not None})}"
                      f" {t.bucket}: {len(p)} points {p[0]}..{p[-1]}")
        return 0
    if a.cmd == "merge":
        return merge(a.args, path)
    if a.cmd == "tune-all":
        return tune_all(a.spec, t, t.select(a.args) if a.args else list(t.problems),
                        gpu_list(a.gpus), a.per_job, pts, a.out or "/tmp/kit_tune",
                        rounds=a.rounds, iters=a.iters, ncopies=a.copies, sets=a.set,
                        force=a.force, log=log)
    if not a.args:
        raise SystemExit(f"{a.cmd}: problems or groups; groups: {list(t.groups)}; problems: "
                         f"{list(t.problems)}")
    pids = t.select(a.args)
    for pid in pids:
        prob = t.problem(pid, a.set)
        if a.cmd == "configs":
            if len(pids) > 1:
                print(f"# {pid}")
            for cfg in op.configs(dict(prob, **{t.bucket: (pts or t.points[pid])[0]})):
                print(op.name_of(cfg))
        elif a.cmd == "tune":
            tune(t, prob, pts or list(t.points[pid]), a.out or path, rounds=a.rounds,
                 iters=a.iters, ncopies=a.copies, log=log, pid=pid)
        elif a.cmd == "show":
            if len(pids) > 1:
                print(f"\n#### {pid}\n")
            show(t, prob, path)
        else:
            verify(t, prob, pts or [1, 3, 20, 100, 300, 1000, 3000, 10000], path,
                   iters=a.iters, ncopies=a.copies, log=log)
    return 0
