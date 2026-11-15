# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Tuner for ``a8w4_moe``: measures every candidate kernel config (TP and EP families)
per token count and stores ALL results; the runtime picks the fastest config that is
allowed in its context (activation, CUDA-graph capture, token count).

    # one process per GPU, disjoint token lists, then merge:
    HIP_VISIBLE_DEVICES=0 python -m a8w4.tune tune --case mimo-v2.6-flash-tp8 -t 1 2 4 --out /tmp/p0.csv
    python -m a8w4.tune merge /tmp/p*.csv            # into configs/a8w4_moe_tuned.csv
    python -m a8w4.tune show --case mimo-v2.6-flash-tp8
    python -m a8w4.tune verify --case mimo-v2.6-flash-ep8 -t 1 100 3000
    # SiTUv2 experts: their own rows (act column), same commands with --act situv2
    python -m a8w4.tune tune --act situv2 --case mimo-v2.6-flash-ep8 -t 1 2 4 --out /tmp/s0.csv
    # the whole table on 8 idle GPUs (4 per (act, case) job), BAD / FAIL report, merged
    python -m a8w4.tune tune-all --gpus 0-7 --out /tmp/a8w4_tune
    python -m a8w4.tune compare configs/a8w4_moe_tuned.csv /tmp/a8w4_tune/a8w4_moe_tuned.new.csv
    python -m a8w4.tune readme --check               # README 7.2-7.4 vs the table
    python -m a8w4.tune stats                        # situv2 vs silu rows

Run from the directory that contains ``a8w4/`` (or put it on PYTHONPATH);
``python a8w4/tune.py ...`` works too. Timing: cold caches (L2 flushed, and every
timed call reads the next of ``--wcopies`` weight copies: the flush does not evict
weights that every call re-reads from the MALL), host launch overhead excluded,
median of ``--iters`` (see ``ref.time_cold``, ``ref.weight_copies``).
SiTUv2 runs with ``ref.SITU`` (its betas / clamp are launch arguments: no effect on
the time). See configs/README.md for the commands that produced the shipped table.
"""

import argparse
import csv
import fcntl
import glob
import itertools
import json
import os
import re
import subprocess
import sys
import time

if __package__ in (None, ""):  # run as a script: make the package importable
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import a8w4  # noqa: E402,F401

    __package__ = "a8w4"

import torch  # noqa: E402

from . import ref as R  # noqa: E402
from .moe import (  # noqa: E402
    DEFAULT_CSV,
    FIELDS,
    _load_table,
    a8w4_moe,
    config_name,
    default_config,
    lookup_config,
    parse_config,
    run_config,
    supported,
)

# (H, I, E local, E global, topk, expert_offset). XiaomiMiMo/MiMo-V2.6-Flash(-RL):
# hidden 4096, moe_intermediate 2048, 256 routed experts, top-8, SiLU.
CASES = {
    "mimo-v2.6-flash-tp8": (4096, 256, 256, 256, 8, 0),
    "mimo-v2.6-flash-tp4": (4096, 512, 256, 256, 8, 0),
    "mimo-v2.6-flash-ep8": (4096, 2048, 32, 256, 8, 0),
}
TOKENS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 320, 384, 448, 512, 640,
          768, 1024, 1280, 1536, 2048, 2560, 3072, 4096, 6144, 8192, 12288, 16384, 24576, 32768]
COS_TOL = 5e-3  # vs the fp8-emulating fp32 reference (typical 1e-5 .. 1e-4)
EP_CFGS = ("t16", "s16", "s32", "m64", "l128", "l128c")
# (act, case) groups of the shipped table (configs/README.md)
TABLE_JOBS = ("situv2:mimo-v2.6-flash-tp8", "situv2:mimo-v2.6-flash-ep8",
              "silu:mimo-v2.6-flash-tp8", "silu:mimo-v2.6-flash-ep8", "silu:mimo-v2.6-flash-tp4")


def candidates(shape):
    """Every config worth measuring on ``shape`` (structurally supported ones)."""
    H, I, E, EGl, K = shape
    out = []
    for bm in (32, 64):
        for g1 in (1, 2):
            for g2 in ("base", "nt4"):
                out.append(dict(kind="tp2k", block_m=bm, g1_bcm=g1, g2=g2))
    # fused: each CTA takes 256 / 384 / 512 inter columns (256: 2 waves/SIMD, best for
    # small/medium batches; 384-512: 1 wave/SIMD, best for large batches at large I)
    splits = sorted({I // c for c in (256, 384, 512) if I % c == 0})
    for bcm in (1, 2):
        for ks in splits:
            out.append(dict(kind="tp1k", bcm=bcm, ksplit=ks))
    out.append(dict(kind="tp1kq", bcm=1))
    out += [dict(kind="ep3k", cfg=n) for n in EP_CFGS]
    out.append(dict(kind="ep1k", maxm=512))
    return [c for c in out if supported(c, shape)]


def _situ(act):
    """SiTUv2 parameters of the tuner (``ref.SITU``); None for SiLU."""
    return R.SITU if act == "situv2" else None


def _key(r):
    return tuple(str(r[k]) for k in ("model_dim", "inter_dim", "experts", "experts_global",
                                     "topk", "routing", "act", "num_tokens", "config"))


def _read_rows(path):
    """Rows of a table file; rows of files written before the act column are SiLU."""
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r["act"] = r.get("act") or "silu"
    return rows


def write_rows(path, new_rows):
    """Merge rows into ``path`` (same shape / act / point / config replaced), sorted."""
    old = _read_rows(path) if os.path.exists(path) else []
    keys = {_key(r) for r in new_rows}
    rows = [r for r in old if _key(r) not in keys] + list(new_rows)
    rows.sort(key=lambda r: (int(r["model_dim"]), int(r["inter_dim"]), int(r["experts"]),
                             int(r["experts_global"]), int(r["topk"]), r["routing"], r["act"],
                             int(r["num_tokens"]), float(r["us"])))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def _case(args):
    if args.shape:
        H, I, E, EGl, K = (int(v) for v in args.shape.split(","))
        return (H, I, E, EGl, K), args.expert_offset
    *shape, e0 = CASES[args.case]
    return tuple(shape), e0


def tune(shape, e0, tokens, out_path, routing="uniform", iters=20, rounds=3, log=print,
         act="silu", wcopies=R.WCOPIES):
    """Per token count, ``rounds`` rounds, each with a fresh routing draw (at small
    batches the per-rank work depends on the draw): check every candidate against
    the reference, time it (candidates interleaved, so clock / power drift hits all
    of them; each timed call reads the next of ``wcopies`` weight copies); store the
    median per candidate. ``act``: the experts' activation (its rows are stored under
    that act)."""
    H, I, E, EGl, K = shape
    situ = _situ(act)
    W, raw = R.make_weights(E, H, I)
    nxt = itertools.cycle(R.weight_copies(W, wcopies)).__next__
    cands = candidates(shape)
    log(f"shape H{H} I{I} E{E}/{EGl} top{K} e0 {e0} act {act}: {len(cands)} candidates, "
        f"{wcopies} weight copies")
    for tok in tokens:
        names = [config_name(c) for c in cands if supported(c, shape, act=act, tokens=tok)]
        times = {n: [] for n in names}
        cos = dict.fromkeys(names, 0.0)
        bad = {}
        for rnd in range(rounds):
            x, tw, ti = R.make_routing(tok, H, EGl, K, seed=101 * tok + rnd, routing=routing)
            ref = R.ref_moe(x, tw, ti, raw, e0, act, situ)
            for name in names:
                if name in bad:
                    continue
                fn = lambda name=name, W=W: run_config(name, x, W, tw, ti,  # noqa: E731
                                                       expert_offset=e0, num_experts=EGl,
                                                       act=act, situ=situ)
                try:
                    c = R.cos_diff(ref, fn().float())
                except Exception as ex:  # noqa: BLE001
                    bad[name] = f"FAIL {type(ex).__name__}: {str(ex)[:120]}"
                    continue
                if not c < COS_TOL:
                    bad[name] = f"BAD cos {c:.1e}"
                    continue
                cos[name] = max(cos[name], c)
                times[name].append(R.time_cold(lambda fn=fn: fn(W=nxt()), iters=iters))
        rows = []
        for name in names:
            if name in bad:
                continue
            us = sorted(times[name])[len(times[name]) // 2]
            rows.append(dict(model_dim=H, inter_dim=I, experts=E, experts_global=EGl, topk=K,
                             routing=routing, act=act, num_tokens=tok, config=name,
                             us=f"{us:.1f}", cos=f"{cos[name]:.1e}"))
        msgs = [f"{n}: {m}" for n, m in bad.items()]
        rows.sort(key=lambda r: float(r["us"]))
        if rows:
            write_rows(out_path, rows)
        top = " | ".join(f"{r['config']} {r['us']}" for r in rows[:3])
        log(f"tok {tok:6d}: {top}" + (f"  [{'; '.join(msgs)}]" if msgs else ""))
        torch.cuda.empty_cache()


def _points(csv_path, shape, routing, act):
    rows = _load_table(csv_path, os.path.getmtime(csv_path)).get((*shape, routing, act))
    if not rows:
        raise SystemExit(f"no rows for {shape} {routing} {act} in {csv_path}")
    return rows


def show(shape, csv_path, routing="uniform", act="silu"):
    """Markdown: per point the fastest config, the fastest of each family and the
    graph-safe pick when it differs."""
    print("| tokens | best | us | best TP family | us | best EP family | us | graph-safe pick |")
    print("|---:|---|---:|---|---:|---|---:|---|")
    for n, cands in _points(csv_path, shape, routing, act):
        us0, best = cands[0]
        tp = next(((u, c) for u, c in cands if c["kind"].startswith("tp")), (None, None))
        ep = next(((u, c) for u, c in cands if c["kind"].startswith("ep")), (None, None))
        safe = lookup_config(shape, n, act=act, capturing=True, routing=routing,
                             csv_path=csv_path)
        cell = lambda uc: (config_name(uc[1]), f"{uc[0]:.1f}") if uc[1] else ("-", "-")  # noqa: E731
        print(f"| {n} | {config_name(best)} | {us0:.1f} | {' | '.join(cell(tp))} | "
              f"{' | '.join(cell(ep))} | "
              f"{'' if safe == best else config_name(safe)} |")


def verify(shape, e0, tokens, csv_path, routing="uniform", iters=20, log=print, act="silu",
           wcopies=R.WCOPIES):
    """``a8w4_moe`` (table lookup) at each token count, also between tuned points:
    cos vs the reference, time vs the table entry and vs the untuned default."""
    H, I, E, EGl, K = shape
    situ = _situ(act)
    W, raw = R.make_weights(E, H, I)
    nxt = itertools.cycle(R.weight_copies(W, wcopies)).__next__
    pts = dict(_points(csv_path, shape, routing, act))
    for tok in tokens:
        x, tw, ti = R.make_routing(tok, H, EGl, K, seed=7 + tok, routing=routing)
        ref = R.ref_moe(x, tw, ti, raw, e0, act, situ)
        cfg = lookup_config(shape, tok, act=act, routing=routing, csv_path=csv_path)
        fn = lambda W=W: a8w4_moe(x, W, tw, ti, expert_offset=e0,  # noqa: E731
                                  num_experts=EGl, act=act, situ=situ, routing=routing,
                                  csv_path=csv_path)
        c = R.cos_diff(ref, fn().float())
        us = R.time_cold(lambda: fn(nxt()), iters=iters)
        dflt = default_config(shape, tok, act)
        dus = R.time_cold(lambda: run_config(dflt, x, nxt(), tw, ti, expert_offset=e0,
                                             num_experts=EGl, act=act, situ=situ), iters=iters)
        n = next((p for p in sorted(pts) if p >= tok), max(pts))
        tus = next(u for u, cc in pts[n] if cc == cfg)
        log(f"tok {tok:6d} -> point {n:6d}: {config_name(cfg):24s} {us:8.1f} us (table {tus:8.1f})"
            f" | default {config_name(dflt):24s} {dus:8.1f} us | x{dus / us:.2f} | cos {c:.1e}"
            + ("" if c < COS_TOL else " BAD"))


def merge(parts, csv_path):
    rows = []
    for p in parts:
        rows += _read_rows(p)
    write_rows(csv_path, rows)
    print(f"merged {len(rows)} rows from {len(parts)} files into {csv_path}")


def _gpu_list(spec):
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


def _lock(gpus):
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


def tune_all(jobs, gpus, per_job, tokens, out_dir, args, force=False, log=print):
    """Every (act, case) job of ``jobs`` on ``per_job`` GPUs (GPU g of the group takes
    tokens[g::per_job]); groups of ``gpus`` take the next job when they free up. Then the
    BAD / FAIL lines of all logs and a merged <out_dir>/a8w4_moe_tuned.new.csv."""
    if len(gpus) < per_job:
        raise SystemExit(f"--per-job {per_job} needs at least that many --gpus ({gpus})")
    for j in jobs:
        act, _, case = j.partition(":")
        if act not in R.ACTS or case not in CASES:
            raise SystemExit(f"job {j!r}: expected <act>:<case>, act in {R.ACTS}, case in "
                             f"{sorted(CASES)}")
    if not force:
        busy = busy_gpus(gpus)
        if busy:
            raise SystemExit(f"GPUs busy (GFX use %): {busy}; tuning next to another job "
                             "skews the table. Wait, pick other --gpus, or --force")
    held = _lock(gpus)  # noqa: F841 - released at exit
    os.makedirs(out_dir, exist_ok=True)
    groups = [gpus[i:i + per_job] for i in range(0, len(gpus) - per_job + 1, per_job)]
    env = dict(os.environ, AITER_LOG_LEVEL=os.environ.get("AITER_LOG_LEVEL", "ERROR"))
    todo, running, failed = list(jobs), {}, []
    t0 = time.time()
    while todo or running:
        for gi, procs in list(running.items()):
            if all(p.poll() is not None for _, p in procs):
                failed += [name for name, p in procs if p.returncode]
                del running[gi]
        for gi, grp in enumerate(groups):
            if gi in running or not todo:
                continue
            act, _, case = todo.pop(0).partition(":")
            procs = []
            for g, gpu in enumerate(grp):
                shard = tokens[g::per_job]
                if not shard:
                    continue
                stem = os.path.join(out_dir, f"{act}_{case}_{g}")
                cmd = [sys.executable, os.path.abspath(__file__), "tune", "--act", act, "--case",
                       case, "-t", *map(str, shard), "--out", stem + ".csv", "--iters",
                       str(args.iters), "--rounds", str(args.rounds), "--wcopies", str(args.wcopies),
                       "--routing", args.routing]
                with open(stem + ".log", "w") as fh:
                    procs.append((stem + ".log", subprocess.Popen(
                        cmd, stdout=fh, stderr=subprocess.STDOUT,
                        env=dict(env, HIP_VISIBLE_DEVICES=str(gpu)))))
            running[gi] = procs
            log(f"{act} {case}: GPUs {grp}, {len(tokens)} token counts")
        time.sleep(2)
    log(f"all jobs done in {(time.time() - t0) / 60:.1f} min")
    logs = sorted(glob.glob(os.path.join(out_dir, "*_[0-9]*.log")))
    bad = [f"{os.path.basename(f)}: {ln.strip()}" for f in logs for ln in open(f)
           if re.search(r"\b(BAD|FAIL)\b", ln)]
    print("\n".join(["BAD / FAIL candidates:"] + bad) if bad else "no BAD / FAIL candidates")
    if failed:
        print(f"FAILED processes (see their logs): {failed}")
    parts = sorted(glob.glob(os.path.join(out_dir, "*_[0-9]*.csv")))
    new = os.path.join(out_dir, "a8w4_moe_tuned.new.csv")
    if os.path.exists(new):
        os.remove(new)
    merge(parts, new)
    print(f"next: python -m a8w4.tune compare {DEFAULT_CSV} {new}\n"
          f"      cp {new} {DEFAULT_CSV}   # then: python -m a8w4.tune readme --write")
    return 1 if failed else 0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("mode", choices=["tune", "merge", "show", "verify", "candidates", "tune-all",
                                    "compare", "readme", "stats"])
    p.add_argument("parts", nargs="*", help="merge: tune output files; compare: OLD NEW tables")
    p.add_argument("--case", default="mimo-v2.6-flash-tp8", choices=sorted(CASES))
    p.add_argument("--shape", default=None, help="H,I,E_local,E_global,topk (overrides --case)")
    p.add_argument("--expert-offset", type=int, default=0)
    p.add_argument("-t", type=int, nargs="*", default=None, help="token counts")
    p.add_argument("--routing", default="uniform", choices=R.ROUTINGS)
    p.add_argument("--act", default="silu", choices=R.ACTS,
                   help="experts' activation: tune / show / verify its rows (situv2: ref.SITU)")
    p.add_argument("--csv", default=DEFAULT_CSV)
    p.add_argument("--out", default=None,
                   help="tune: output file (default --csv); tune-all: output directory "
                        "(default /tmp/a8w4_tune)")
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--rounds", type=int, default=3, help="tune: interleaved timing rounds")
    p.add_argument("--wcopies", type=int, default=R.WCOPIES,
                   help="weight copies the timed calls cycle through (1: the same weights "
                        "every call, which the MALL keeps: favors cached W loads)")
    p.add_argument("--gpus", default="0-7", help="tune-all: GPUs, e.g. 0-7 or 0,1,2,3")
    p.add_argument("--per-job", type=int, default=4, help="tune-all: GPUs per (act, case) job")
    p.add_argument("--jobs", nargs="+", default=list(TABLE_JOBS),
                   help="tune-all: <act>:<case> jobs (default: the shipped table's)")
    p.add_argument("--force", action="store_true", help="tune-all: skip the idle-GPU check")
    p.add_argument("--write", action="store_true", help="readme: update README.md")
    p.add_argument("--check", action="store_true",
                   help="readme: exit 1 when README 7.2-7.4 differ from the table")
    args = p.parse_args()
    log = lambda s: print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True)  # noqa: E731
    if args.mode == "merge":
        return merge(args.parts, args.csv)
    if args.mode in ("compare", "readme", "stats"):
        from . import tables

        if args.mode == "compare":
            if len(args.parts) != 2:
                raise SystemExit("compare OLD.csv NEW.csv")
            return tables.compare(*args.parts)
        if args.mode == "readme":
            sys.exit(tables.readme(args.csv, write=args.write, check=args.check))
        return tables.stats(args.csv)
    if args.mode == "tune-all":
        sys.exit(tune_all(args.jobs, _gpu_list(args.gpus), args.per_job, args.t or TOKENS,
                          args.out or "/tmp/a8w4_tune", args, args.force, log))
    shape, e0 = _case(args)
    if args.mode == "candidates":
        for c in candidates(shape):
            print(config_name(c))
    elif args.mode == "tune":
        tune(shape, e0, args.t or TOKENS, args.out or args.csv, args.routing, args.iters,
             args.rounds, log, args.act, args.wcopies)
    elif args.mode == "show":
        show(shape, args.csv, args.routing, args.act)
    else:
        verify(shape, e0, args.t or [1, 3, 20, 100, 300, 1000, 3000, 10000], args.csv,
               args.routing, args.iters, log, args.act, args.wcopies)


if __name__ == "__main__":
    main()
