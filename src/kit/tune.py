"""Tune an op into its tuned table, and inspect the table (torch; ``python -m pyhip.kit``).

    python -m pyhip.kit cases      a8w4:moe                     # the tune cases: tp8-silu ...
    python -m pyhip.kit candidates a8w4:moe tp8-silu            # the grid configs
    python -m pyhip.kit tune       a8w4:moe tp8-silu -t 1 2 4 --out /tmp/p0.csv
    python -m pyhip.kit merge      a8w4:moe /tmp/p*.csv         # into the registry's table
    python -m pyhip.kit show       a8w4:moe tp8-silu            # per point: the fastest configs
    python -m pyhip.kit verify     a8w4:moe ep8-silu -t 1 100 3000
    python -m pyhip.kit tune-all   a8w4:moe [CASE ...] --gpus 0-7 --per-job 4 --out /tmp/t
    python -m pyhip.kit compare    OLD.csv NEW.csv

``<module>:<op>``: an importable module holding one ``Registry`` and the op's name. The
registry declares the table, its ``bucket`` and the bucket's tune ``points``, the dims
the table does not store (``unkeyed``) and the interface arguments the timed calls cycle
through copies of (``rotate``).

A tune case is a point of ``<op>_test``'s grid without the bucket axis (its id joins the
other axes' ids, e.g. ``tp8-silu``); ``--set name=value`` overrides dim values (another
shape, a routing). ``tune``: per point, ``--rounds`` rounds, each with fresh inputs
(``<op>_test``, seed 101 * point + round); per round every grid config runs once
(one that raises cannot run the case: n/a), is checked against ``<op>_ref``
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

from . import Registry, _cell, _guess
from . import table as _table
from .timing import NCOPIES, copies, time_cold

__all__ = ["resolve", "cases", "case", "fields", "tune", "show", "verify", "merge", "tune_all",
           "main"]


def resolve(spec):
    """``"module:op"`` -> the Op (the module holds exactly one Registry)."""
    mod, _, name = spec.rpartition(":")
    if not mod or not name:
        raise SystemExit(f"{spec!r}: expected <module>:<op>, e.g. a8w4:moe")
    m = importlib.import_module(mod)
    regs = {id(v): v for v in vars(m).values() if isinstance(v, Registry)}
    if len(regs) != 1:
        raise SystemExit(f"{mod}: expected one Registry, found {len(regs)}")
    reg = next(iter(regs.values()))
    if name not in reg:
        raise SystemExit(f"{mod}: no op {name!r}; ops: {list(reg)}")
    return reg[name]


def _bucket(op):
    b = op.registry.bucket
    if b is None or b not in op.dim_names:
        raise SystemExit(f"{op.name}: tuning needs Registry(bucket=) naming a dim")
    return b


def cases(op):
    """{id: case}: the points of ``<op>_test``'s grid without the bucket axis; the bucket
    is None in ``case``."""
    b = _bucket(op)
    axes = [p for p in op._axes if b not in p.names]
    if any(b in p.names and len(p.names) > 1 for p in op._axes):
        raise SystemExit(f"{op.name}_test: the bucket {b!r} shares a @dim axis with other "
                         "names; tuning replaces that axis")
    out = {}
    for combo in itertools.product(*(p.points() for p in axes)):
        c = dict(op._test_defaults)
        for _, vals in combo:
            c.update(vals)
        c[b] = None
        out["-".join(tok for tok, _ in combo)] = c
    return out


def case(op, cid, sets=()):
    """The tune case ``cid`` with ``sets`` ("name=value") applied."""
    all_ = cases(op)
    if cid not in all_:
        raise SystemExit(f"{op.name}: no tune case {cid!r}; cases: {list(all_)}")
    c = dict(all_[cid])
    for s in sets:
        k, eq, v = s.partition("=")
        if not eq or k not in op.dim_names:
            raise SystemExit(f"--set {s!r}: expected name=value, name in {list(op.dim_names)}")
        c[k] = _cell(v, op._test_defaults.get(k))
    return c


def _keys(op):
    """The dims the table stores, in dim order (bucket excluded)."""
    reg = op.registry
    return [n for n in op.dim_names if n not in reg.unkeyed and n != reg.bucket]


def fields(op, metrics):
    """The table header: key columns, bucket, config, us, the check's metrics."""
    return [*_keys(op), _bucket(op), "config", "us", *metrics]


def _check(op, out, expected, inputs):
    """``<op>_check`` (default: torch.testing.assert_close) -> its metrics dict."""
    if op.check is not None:
        return op.check(out, expected, **inputs) or {}
    if torch.is_tensor(out) and torch.is_tensor(expected):
        expected = expected.to(out.dtype)
    torch.testing.assert_close(out, expected)
    return {}


class _Rotation:
    """Copies of the ``rotate`` inputs, made once per input object; ``next()`` -> {name:
    the next copy}."""

    def __init__(self, op, n):
        self.names, self.n, self.cyc = op.registry.rotate, n, {}

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


def tune(op, c, points, out, *, rounds=3, iters=20, ncopies=NCOPIES, log=print, cid=""):
    """Tune case ``c`` at ``points`` into the table ``out`` (see the module docstring)."""
    b = _bucket(op)
    known = {k: v for k, v in c.items() if v is not None and k != b}
    log(f"{op.name} {cid} {known}: {len(op.configs(dict(c, **{b: points[0]})))} candidates, "
        f"{ncopies} copies of {list(op.registry.rotate)}")
    rot = _Rotation(op, ncopies)
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
            rows.append(dict({k: dims[k] for k in _keys(op)}, **{b: p}, config=name,
                             us=f"{us:.1f}", **{k: f"{v:.1e}" for k, v in metrics[name].items()}))
        rows.sort(key=lambda r: float(r["us"]))
        if rows:
            _table.merge(out, rows, fields(op, [k for k in rows[0] if k not in
                                                 (*_keys(op), b, "config", "us")]))
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


def show(op, c, path=None, out=print):
    """Markdown, per point: the fastest config, the fastest of each solver kind
    and the capture pick when it differs."""
    b, path = _bucket(op), path or op.registry.table
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


def verify(op, c, points, path=None, *, iters=20, ncopies=NCOPIES, log=print):
    """``op(...)`` (the table's pick) at each bucket value, also between tuned points:
    the check against ``<op>_ref`` and the time vs the table entry."""
    b, path = _bucket(op), path or op.registry.table
    pts = dict(_points(op, c, path))
    rot = _Rotation(op, ncopies)
    for t in points:
        dims = dict(c, **{b: t})
        inputs = _inputs(op, dims, 7 + t)
        rot.bind(inputs)
        expected = op.ref(**inputs)
        cfg = op.pick(dims, table=path)
        hints = {k: dims[k] for k in _keys(op) if k not in op._ref_names}
        fn = lambda over: op(**{**inputs, **over}, table=path, **hints)  # noqa: E731
        try:
            m = _check(op, fn({}), expected, inputs)
            msg = "".join(f" | {k} {v:.1e}" for k, v in m.items())
        except AssertionError as ex:
            msg = f" | BAD {ex}"
        us = time_cold(lambda: fn(rot.next()), iters=iters)
        n = next((p for p in sorted(pts) if p >= t), max(pts))
        tus = next(u for u, cc in pts[n] if cc == cfg)
        log(f"{b} {t:6} -> point {n:6}: {op.name_of(cfg):24s} {us:8.1f} us (table {tus:8.1f})"
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


def tune_all(spec, op, cids, gpus, per_job, points, out_dir, *, rounds=3, iters=20,
             ncopies=NCOPIES, sets=(), force=False, log=print):
    """Every case of ``cids`` on ``per_job`` GPUs (GPU g of the group takes
    points[g::per_job]); groups of ``gpus`` take the next case when they free up. Then
    the BAD lines of all logs and a merged <out_dir>/<table name>.new.csv."""
    if len(gpus) < per_job:
        raise SystemExit(f"--per-job {per_job} needs at least that many --gpus ({gpus})")
    known = cases(op)
    for cid in cids:
        if cid not in known:
            raise SystemExit(f"case {cid!r}: one of {list(known)}")
    if not force:
        busy = busy_gpus(gpus)
        if busy:
            raise SystemExit(f"GPUs busy (GFX use %): {busy}; tuning next to another job "
                             "skews the table. Wait, pick other --gpus, or --force")
    held = lock_gpus(gpus)  # noqa: F841 - released at exit
    os.makedirs(out_dir, exist_ok=True)
    groups = [gpus[i:i + per_job] for i in range(0, len(gpus) - per_job + 1, per_job)]
    env = dict(os.environ, AITER_LOG_LEVEL=os.environ.get("AITER_LOG_LEVEL", "ERROR"))
    todo, running, failed = list(cids), {}, []
    t0 = time.time()
    while todo or running:
        for gi, procs in list(running.items()):
            if all(p.poll() is not None for _, p in procs):
                failed += [name for name, p in procs if p.returncode]
                del running[gi]
        for gi, grp in enumerate(groups):
            if gi in running or not todo:
                continue
            cid = todo.pop(0)
            procs = []
            for g, gpu in enumerate(grp):
                shard = points[g::per_job]
                if not shard:
                    continue
                stem = os.path.join(out_dir, f"{cid}_{g}")
                cmd = [sys.executable, "-m", "pyhip.kit", "tune", spec, cid, "-t",
                       *map(str, shard), "--out", stem + ".csv", "--iters", str(iters),
                       "--rounds", str(rounds), "--copies", str(ncopies)]
                if sets:
                    cmd += ["--set", *sets]
                with open(stem + ".log", "w") as fh:
                    procs.append((stem + ".log", subprocess.Popen(
                        cmd, stdout=fh, stderr=subprocess.STDOUT,
                        env=dict(env, HIP_VISIBLE_DEVICES=str(gpu)))))
            running[gi] = procs
            log(f"{cid}: GPUs {grp}, {len(points)} points")
        time.sleep(2)
    log(f"all jobs done in {(time.time() - t0) / 60:.1f} min")
    logs = sorted(glob.glob(os.path.join(out_dir, "*_[0-9]*.log")))
    bad = [f"{os.path.basename(f)}: {ln.strip()}" for f in logs for ln in open(f)
           if re.search(r"\bBAD\b", ln)]
    print("\n".join(["BAD candidates:"] + bad) if bad else "no BAD candidates")
    if failed:
        print(f"FAILED processes (see their logs): {failed}")
    parts = sorted(glob.glob(os.path.join(out_dir, "*_[0-9]*.csv")))
    stem = os.path.splitext(os.path.basename(op.registry.table or f"{op.name}.csv"))[0]
    new = os.path.join(out_dir, f"{stem}.new.csv")
    if os.path.exists(new):
        os.remove(new)
    if parts:
        merge(parts, new)
    print(f"next: python -m pyhip.kit compare {op.registry.table} {new}\n"
          f"      cp {new} {op.registry.table}")
    return 1 if failed else 0


# ---- command line ---------------------------------------------------------------------

def parser(prog="python -m pyhip.kit"):
    p = argparse.ArgumentParser(prog=prog, description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["cases", "candidates", "tune", "merge", "show", "verify",
                                   "tune-all", "compare"])
    p.add_argument("op", help="<module>:<op> (compare: the old table)")
    p.add_argument("args", nargs="*", help="a case (candidates / tune / show / verify), cases "
                                           "(tune-all; default all), tables (merge, compare)")
    p.add_argument("-t", "--points", nargs="+", default=None,
                   help="bucket values (default: the registry's points; verify: a few)")
    p.add_argument("--set", nargs="+", default=[], metavar="NAME=VALUE",
                   help="override dim values of the case")
    p.add_argument("--table", default=None, help="the table (default: the registry's)")
    p.add_argument("--out", default=None, help="tune: output table (default --table); "
                                               "tune-all: output directory")
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--rounds", type=int, default=3, help="tune: interleaved timing rounds")
    p.add_argument("--copies", type=int, default=NCOPIES,
                   help="copies of the rotate inputs the timed calls cycle through")
    p.add_argument("--gpus", default="0-7", help="tune-all: GPUs, e.g. 0-7 or 0,1,2,3")
    p.add_argument("--per-job", type=int, default=4, help="tune-all: GPUs per case")
    p.add_argument("--force", action="store_true", help="tune-all: skip the idle-GPU check")
    return p


def main(argv=None, prog="python -m pyhip.kit"):
    a = parser(prog).parse_args(argv)
    if a.cmd == "compare":
        if len(a.args) != 1:
            raise SystemExit("compare OLD.csv NEW.csv")
        return _table.compare(a.op, a.args[0])
    op = resolve(a.op)
    log = lambda s: print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True)  # noqa: E731
    path = a.table or op.registry.table
    pts = [_guess(v) for v in a.points] if a.points else None
    if a.cmd == "cases":
        for cid, c in cases(op).items():
            print(cid, {k: v for k, v in c.items() if v is not None})
        return 0
    if a.cmd == "merge":
        return merge(a.args, path)
    if a.cmd == "tune-all":
        return tune_all(a.op, op, a.args or list(cases(op)), gpu_list(a.gpus), a.per_job,
                        pts or list(op.registry.points or ()), a.out or "/tmp/kit_tune",
                        rounds=a.rounds, iters=a.iters, ncopies=a.copies, sets=a.set,
                        force=a.force, log=log)
    if len(a.args) != 1:
        raise SystemExit(f"{a.cmd}: one case, one of {list(cases(op))}")
    c = case(op, a.args[0], a.set)
    if a.cmd == "candidates":
        b = _bucket(op)
        for cfg in op.configs(dict(c, **{b: (pts or op.registry.points or [None])[0]})):
            print(op.name_of(cfg))
    elif a.cmd == "tune":
        tune(op, c, pts or list(op.registry.points), a.out or path, rounds=a.rounds,
             iters=a.iters, ncopies=a.copies, log=log, cid=a.args[0])
    elif a.cmd == "show":
        show(op, c, path)
    else:
        verify(op, c, pts or [1, 3, 20, 100, 300, 1000, 3000, 10000], path, iters=a.iters,
               ncopies=a.copies, log=log)
    return 0
