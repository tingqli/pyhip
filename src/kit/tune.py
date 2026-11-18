"""Tune an op into its tuned table, and inspect the table (torch; ``python -m pyhip.kit``).

    python -m pyhip.kit problems   a8w4.tune:TUNING             # the tune problems: tp8-silu ...
    python -m pyhip.kit candidates a8w4.tune:TUNING tp8-silu    # the grid configs
    python -m pyhip.kit tune       a8w4.tune:TUNING tp8-silu -t 1 2 4 --out /tmp/p0.csv
    python -m pyhip.kit merge      a8w4.tune:TUNING /tmp/p*.csv # into the tuning's table
    python -m pyhip.kit show       a8w4.tune:TUNING tp8-silu    # per point: the fastest configs
    python -m pyhip.kit verify     a8w4.tune:TUNING ep8-silu -t 1 100 3000
    python -m pyhip.kit tune-all   a8w4.tune:TUNING [PROBLEM ...] --gpus 0-7 --per-job 4 --out /tmp/t
    python -m pyhip.kit compare    OLD.csv NEW.csv

``<module>:<name>``: an importable module and the name of the ``Tuning`` in it that
declares how to tune the op (see ``Tuning``): the problems, the bucket dim and its
points, the key dims the table stores, the inputs the timed calls rotate, the table.

``--set name=value`` overrides dim values of a problem (another shape, a routing).
``tune``: per point, ``--rounds`` rounds, each with fresh inputs (``<op>_test``, seed
101 * point + round); per round every grid config runs once (one that raises cannot run
the problem: n/a), is checked against ``<op>_ref`` (``<op>_check``: a failure is BAD)
and timed cold (``kit.timing.time_cold``, the ``rotate`` inputs cycling through
``--copies`` copies), the configs interleaved. Every config that ran and passed is
stored: the median time, the worst metric of the check.
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
    """How to tune ``op`` (all of it, in one place):

    * ``problems``: [(id, problem)] (``op.problems(...)``); the bucket is set per point;
    * ``bucket``: the dim tuned at ``points``, matched inexactly by the dispatch (the
      first tuned point >= the call's value; the other key dims match exactly);
    * ``keys``: the dims the table stores (default: every dim but the bucket); a dim
      left out must not change which config is fastest (e.g. an expert offset);
    * ``rotate``: interface arguments the timed calls cycle through copies of (e.g.
      the weights: ``kit.timing.copies``);
    * ``table``: the table ``tune`` / ``merge`` write and ``show`` / ``verify`` read
      (default: the registry's, which the dispatch reads).
    """

    def __init__(self, op, *, problems, bucket, points, keys=None, rotate=(), table=_DEFAULT):
        where = f"Tuning({op.name})"
        if bucket not in op.dim_names:
            raise TypeError(f"{where}: bucket {bucket!r} is not a dim {list(op.dim_names)}")
        if op.registry.bucket not in (None, bucket):
            raise TypeError(f"{where}: bucket {bucket!r}, but the dispatch buckets "
                            f"{op.registry.bucket!r} (Registry(bucket=))")
        keys = tuple(n for n in op.dim_names if n != bucket) if keys is None else tuple(keys)
        if _unknown(keys, op.dim_names) or bucket in keys or len(set(keys)) < len(keys):
            raise TypeError(f"{where}: keys {list(keys)}: distinct dims, not the bucket")
        if isinstance(rotate, str) or _unknown(rotate, op._ref_names):
            raise TypeError(f"{where}: rotate {rotate!r}: names of {op.name}_ref's parameters")
        self.op, self.bucket, self.points = op, bucket, tuple(points)
        self.keys, self.rotate = keys, tuple(rotate)
        self.table = op.registry.table if table is _DEFAULT else table
        self.problems = {}
        for pid, prob in problems:
            if pid in self.problems:
                raise TypeError(f"{where}: two problems {pid!r}")
            self.problems[pid] = {**op._blank, **prob, bucket: None}

    def __repr__(self):
        return (f"Tuning({self.op.name}, problems={list(self.problems)}, bucket={self.bucket!r}, "
                f"{len(self.points)} points)")

    def fields(self, metrics):
        """The table header: key columns, bucket, config, us, the check's metrics."""
        return [*self.keys, self.bucket, "config", "us", *metrics]

    def problem(self, pid, sets=()):
        """Problem ``pid`` with ``sets`` ("name=value") applied."""
        if pid not in self.problems:
            raise SystemExit(f"{self.op.name}: no tune problem {pid!r}; problems: "
                             f"{list(self.problems)}")
        c = dict(self.problems[pid])
        for s in sets:
            k, eq, v = s.partition("=")
            if not eq or k not in self.op.dim_names:
                raise SystemExit(f"--set {s!r}: expected name=value, name in "
                                 f"{list(self.op.dim_names)}")
            c[k] = _cell(v, self.op._test_defaults.get(k))
        return c


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


def _inputs(op, dims, seed):
    return op.test(**dims, seed=seed) if op._seeded else op.test(**dims)


def tune(t, c, points, out, *, rounds=3, iters=20, ncopies=NCOPIES, log=print, pid=""):
    """Tune problem ``c`` of ``t`` at ``points`` into the table ``out`` (see the module
    docstring)."""
    op, b = t.op, t.bucket
    known = {k: v for k, v in c.items() if v is not None and k != b}
    log(f"{op.name} {pid} {known}: {len(op.configs(dict(c, **{b: points[0]})))} candidates, "
        f"{ncopies} copies of {list(t.rotate)}")
    rot = _Rotation(t.rotate, ncopies)
    for p in points:
        dims = dict(c, **{b: p})
        cfgs = op.configs(dims)
        names = [op.name_of(cfg) for cfg in cfgs]
        times = {n: [] for n in names}
        metrics = {n: {} for n in names}
        bad, na = {}, {}
        for rnd in range(rounds):
            inputs = _inputs(op, dims, 101 * p + rnd)
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
            rows.append(dict({k: dims[k] for k in t.keys}, **{b: p}, config=name,
                             us=f"{us:.1f}", **{k: f"{v:.1e}" for k, v in metrics[name].items()}))
        rows.sort(key=lambda r: float(r["us"]))
        if rows:
            _table.merge(out, rows, t.fields([k for k in rows[0] if k not in
                                              (*t.keys, b, "config", "us")]))
        top = " | ".join(f"{r['config']} {r['us']}" for r in rows[:3])
        msgs = [f"{n}: {m}" for n, m in bad.items()]
        kinds = sorted(set(na.values()))
        log(f"{b} {p:6}: {top}" + (f"  [{'; '.join(msgs)}]" if msgs else "")
            + (f"  ({len(na)} n/a: {', '.join(kinds)})" if na else ""))
        torch.cuda.empty_cache()


def _points(op, c, path):
    pts = op.points(c, table=path)
    if not pts:
        known = {k: v for k, v in c.items() if v is not None}
        raise SystemExit(f"{op.name}: no rows for {known} in {path}")
    return pts


def show(t, c, path=None, out=print):
    """Markdown, per point: the fastest config, the fastest of each solver kind
    and the capture pick when it differs."""
    op, b, path = t.op, t.bucket, path or t.table
    pts = _points(op, c, path)
    kinds = [k for k in op.impls if any(cfg["kind"] == k for _, cs in pts for _, cfg in cs)]
    out(f"| {b} | best | us | " + " | ".join(f"best {k} | us" for k in kinds)
        + " | capture pick |")
    out("|---:|---|---:|" + "---|---:|" * len(kinds) + "---|")
    for n, cands in pts:
        us0, best = cands[0]
        cells = []
        for k in kinds:
            u, cfg = next(((u, cfg) for u, cfg in cands if cfg["kind"] == k), (None, None))
            cells += [op.name_of(cfg), f"{u:.1f}"] if cfg else ["-", "-"]
        try:
            safe = op.pick(dict(c, **{b: n}), capturing=True, table=path)
            cap = "" if safe == best else op.name_of(safe)
        except ValueError:
            cap = "none"
        out(f"| {n} | {op.name_of(best)} | {us0:.1f} | " + " | ".join(cells) + f" | {cap} |")


def verify(t, c, points, path=None, *, iters=20, ncopies=NCOPIES, log=print):
    """``op(...)`` (the table's pick) at each bucket value, also between tuned points:
    the check against ``<op>_ref`` and the time vs the table entry."""
    op, b, path = t.op, t.bucket, path or t.table
    pts = dict(_points(op, c, path))
    rot = _Rotation(t.rotate, ncopies)
    for v in points:
        dims = dict(c, **{b: v})
        inputs = _inputs(op, dims, 7 + v)
        rot.bind(inputs)
        expected = op.ref(**inputs)
        cfg = op.pick(dims, table=path)
        hints = {k: dims[k] for k in t.keys if k not in op._ref_names}
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
    points[g::per_job]); groups of ``gpus`` take the next problem when they free up.
    Then the BAD lines of all logs and a merged <out_dir>/<table name>.new.csv."""
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
            procs = []
            for g, gpu in enumerate(grp):
                shard = points[g::per_job]
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
            log(f"{pid}: GPUs {grp}, {len(points)} points")
        time.sleep(2)
    log(f"all jobs done in {(time.time() - t0) / 60:.1f} min")
    logs = sorted(glob.glob(os.path.join(out_dir, "*_[0-9]*.log")))
    bad = [f"{os.path.basename(f)}: {ln.strip()}" for f in logs for ln in open(f)
           if re.search(r"\bBAD\b", ln)]
    print("\n".join(["BAD candidates:"] + bad) if bad else "no BAD candidates")
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
    p.add_argument("cmd", choices=["problems", "candidates", "tune", "merge", "show", "verify",
                                   "tune-all", "compare"])
    p.add_argument("spec", help="<module>:<Tuning name> (compare: the old table)")
    p.add_argument("args", nargs="*", help="a problem (candidates / tune / show / verify), "
                                           "problems (tune-all; default all), tables (merge, "
                                           "compare)")
    p.add_argument("-t", "--points", nargs="+", default=None,
                   help="bucket values (default: the tuning's points; verify: a few)")
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
        for pid, c in t.problems.items():
            print(pid, {k: v for k, v in c.items() if v is not None})
        return 0
    if a.cmd == "merge":
        return merge(a.args, path)
    if a.cmd == "tune-all":
        return tune_all(a.spec, t, a.args or list(t.problems), gpu_list(a.gpus), a.per_job,
                        pts or list(t.points), a.out or "/tmp/kit_tune",
                        rounds=a.rounds, iters=a.iters, ncopies=a.copies, sets=a.set,
                        force=a.force, log=log)
    if len(a.args) != 1:
        raise SystemExit(f"{a.cmd}: one problem, one of {list(t.problems)}")
    c = t.problem(a.args[0], a.set)
    if a.cmd == "candidates":
        for cfg in op.configs(dict(c, **{t.bucket: (pts or t.points or [None])[0]})):
            print(op.name_of(cfg))
    elif a.cmd == "tune":
        tune(t, c, pts or list(t.points), a.out or path, rounds=a.rounds, iters=a.iters,
             ncopies=a.copies, log=log, pid=a.args[0])
    elif a.cmd == "show":
        show(t, c, path)
    else:
        verify(t, c, pts or [1, 3, 20, 100, 300, 1000, 3000, 10000], path, iters=a.iters,
               ncopies=a.copies, log=log)
    return 0
