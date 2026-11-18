# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Experiment registry and hygiene (``python -m a8w4.exp <tool>``), so that "what was tried,
under which protocol, with what result" is one query instead of reading every log.

    python -m a8w4.exp grep bcm                    # rows mentioning bcm (regex, any column)
    python -m a8w4.exp grep --area mega --verdict pending idea
    python -m a8w4.exp ls --since 20261115
    python -m a8w4.exp add --dir 20261116_x --area ep --knob "m64 g1.pf=1" --tokens 1024-2048 \\
        --protocol coldrot8 --delta "-1.2%" --verdict landed --log "log_20261116.md 3.1"
    python -m a8w4.exp new gemm2_bm256             # experiments/<today>_gemm2_bm256/ + notes
    python -m a8w4.exp lint                        # all experiment dirs (or given ones)
    python -m a8w4.exp clean FILE ... [--inplace]  # drop aiter / Gloo / torch log lines

experiments/index.csv columns: date, dir ("-": an idea, no run), area (tp / ep / mega / mori
/ table / protocol / flydsl), knob, tokens, protocol (coldrot8: flush + 8 weight copies, the only
unbiased one for cached-W configs; cold1: flush, one copy, unbiased for nt-only configs; -),
delta, verdict (landed / rejected / pending / invalidated / finding / baseline / idea), log
(file + section), note. ``invalidated`` rows were measured on a biased protocol: do not cite
their numbers.
"""

import argparse
import csv
import datetime
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.join(HERE, "experiments")
INDEX = os.path.join(EXP_DIR, "index.csv")
FIELDS = ["date", "dir", "area", "knob", "tokens", "protocol", "delta", "verdict", "log", "note"]
VERDICTS = ("landed", "rejected", "pending", "invalidated", "finding", "baseline", "idea")
NOISE = [
    r"^\[aiter\] ",  # aiter logger: NUMA balancing warning, "import [module] under ..."
    r"^\[Gloo\] ",  # gloo peer connects
    r"^\[[WI]\d{4} [\d:.]+ .*\] ",  # torch C++ log lines (ROCTracer duplicate flow, ...)
    r"^WARNING: AMD GPU device\(s\) is/are in a low-power state",
    r"^Check power control/runtime_status",
]
NOTES_TEMPLATE = """# {name} ({date})

- goal / hypothesis, with the mechanism that would make it faster:
- change (commit or diff in this dir; not a full copy of a kernel file):
- commands (re-runnable, from src/contrib/flydsl):
  `python -m a8w4.bench ab --case ... -t ... -v ... --out a8w4/experiments/{dir}`
- protocol: coldrot, 8 weight copies (the only way the bench tools time); anything else needs a reason
- results: the tables ab.md / dist.md got (do not copy numbers by hand)
- conclusion + mechanism (a gain without a mechanism: suspect the measurement first):
- index: `python -m a8w4.exp add --dir {dir} --area ... --knob ... --verdict ...`
"""


def read_index(path=None):
    path = path or INDEX
    if not os.path.exists(path):
        return []
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _show(rows, wide=False):
    if not rows:
        print("no rows")
        return
    for r in rows:
        print(f"{r['date']} {r['area']:9s} {r['verdict']:11s} {r['knob']} | {r['tokens']} | "
              f"{r['protocol']} | {r['delta']} | {r['dir']} | {r['log']}")
        if wide and r.get("note"):
            print(f"{'':21s}note: {r['note']}")
    print(f"({len(rows)} rows; --wide for notes)" if not wide else f"({len(rows)} rows)")


def _filter(rows, a):
    out = []
    for r in rows:
        if a.area and not any(x in r["area"].split("/") for x in a.area):
            continue
        if a.verdict and r["verdict"] not in a.verdict:
            continue
        if a.since and r["date"] < a.since:
            continue
        if getattr(a, "pattern", None) and not re.search(a.pattern, " ".join(r.values()), re.I):
            continue
        out.append(r)
    return out


def _filters(ap):
    ap.add_argument("--area", nargs="+", default=None)
    ap.add_argument("--verdict", nargs="+", default=None, choices=VERDICTS)
    ap.add_argument("--since", default=None, help="YYYYMMDD")
    ap.add_argument("--wide", action="store_true", help="also the notes")


def cmd_grep(argv):
    ap = argparse.ArgumentParser(prog="python -m a8w4.exp grep")
    ap.add_argument("pattern", nargs="?", default=None, help="regex, case-insensitive")
    _filters(ap)
    a = ap.parse_args(argv)
    _show(_filter(read_index(), a), a.wide)


def cmd_ls(argv):
    ap = argparse.ArgumentParser(prog="python -m a8w4.exp ls")
    _filters(ap)
    a = ap.parse_args(argv)
    _show(_filter(read_index(), a), a.wide)


def cmd_add(argv):
    ap = argparse.ArgumentParser(prog="python -m a8w4.exp add")
    ap.add_argument("--date", default=datetime.date.today().strftime("%Y%m%d"))
    ap.add_argument("--dir", default="-", help='experiment dir under experiments/ ("-": idea)')
    ap.add_argument("--area", required=True)
    ap.add_argument("--knob", required=True)
    ap.add_argument("--tokens", default="-")
    ap.add_argument("--protocol", default="coldrot8")
    ap.add_argument("--delta", default="-")
    ap.add_argument("--verdict", required=True, choices=VERDICTS)
    ap.add_argument("--log", default="-", help="log file + section")
    ap.add_argument("--note", default="")
    # values such as "-1.5%" would parse as options: bind them to their flag
    argv = list(argv)
    for i in range(len(argv) - 1):
        if argv[i] in ("--delta", "--note", "--knob", "--tokens") and argv[i + 1][:1] == "-":
            argv[i], argv[i + 1] = f"{argv[i]}={argv[i + 1]}", None
    a = ap.parse_args([t for t in argv if t is not None])
    if a.dir != "-" and not os.path.isdir(os.path.join(EXP_DIR, a.dir)):
        raise SystemExit(f"no directory experiments/{a.dir}")
    if a.verdict == "landed" and a.protocol.startswith("cold1"):
        print("note: landed on a one-weight-copy protocol: fine only if every variant reads W "
              "non-temporally (nt); else re-measure with coldrot8", file=sys.stderr)
    row = {k: getattr(a, k) for k in FIELDS}
    new = not os.path.exists(INDEX)
    with open(INDEX, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)
    _show([row])


def cmd_new(argv):
    ap = argparse.ArgumentParser(prog="python -m a8w4.exp new")
    ap.add_argument("name", help="short snake_case name")
    ap.add_argument("--date", default=datetime.date.today().strftime("%Y%m%d"))
    a = ap.parse_args(argv)
    if not re.fullmatch(r"[a-z0-9_]+", a.name):
        raise SystemExit("name: lower-case letters, digits, underscores")
    d = f"{a.date}_{a.name}"
    path = os.path.join(EXP_DIR, d)
    os.makedirs(path, exist_ok=False)
    with open(os.path.join(path, "NOTES.md"), "w") as fh:
        fh.write(NOTES_TEMPLATE.format(name=a.name, date=a.date, dir=d))
    print(f"created {path}/NOTES.md\nbefore writing a script, check: python -m a8w4.exp grep "
          f"'<knob>'; time with python -m a8w4.bench ab ... --out a8w4/experiments/{d}")


def _noisy(line):
    return any(re.search(p, line) for p in NOISE)


def cmd_clean(argv):
    ap = argparse.ArgumentParser(prog="python -m a8w4.exp clean")
    ap.add_argument("files", nargs="+")
    ap.add_argument("--inplace", action="store_true")
    a = ap.parse_args(argv)
    for f in a.files:
        lines = open(f, errors="replace").read().splitlines(keepends=True)
        keep = [ln for ln in lines if not _noisy(ln)]
        if a.inplace:
            if len(keep) != len(lines):
                with open(f, "w") as fh:
                    fh.writelines(keep)
            print(f"{f}: dropped {len(lines) - len(keep)} lines")
        else:
            sys.stdout.writelines(keep)


def lint_dir(path, indexed):
    """Problems of one experiment dir: [(level, message)]."""
    out = []
    name = os.path.basename(path.rstrip("/"))
    stdlib = getattr(sys, "stdlib_module_names", ())
    for f in sorted(os.listdir(path)):
        p = os.path.join(path, f)
        if f.endswith(".py"):
            mod = f[:-3]
            if mod in stdlib or mod in ("torch", "aiter", "flydsl", "mori", "triton"):
                out.append(("ERROR", f"{f} shadows the module {mod!r} (e.g. profile.py breaks "
                                     "torch's import): rename it"))
            src = open(p, errors="replace").read()
            if re.search(r"[\"'](/sgl-workspace|/root|/home)/", src):
                out.append(("WARN", f"{f}: absolute path: use paths relative to the script"))
            if re.search(r"time_cold\(|elapsed_time\(", src) and not re.search(
                    r"weight_copies|wcopies|WCOPIES|a8w4\.bench|itertools\.cycle|\bWs\b", src):
                out.append(("WARN", f"{f}: times ONE weight copy: configs with cached W "
                                    "loads look 5-13% faster (python -m a8w4.bench ab, or "
                                    "ref.weight_copies)"))
            if re.search(r"sys\.path\.insert\([^)]*\.\./20\d{6}_", src):
                out.append(("WARN", f"{f}: imports another experiment's script: shared code "
                                    "belongs in a8w4/bench"))
        elif f.endswith((".txt", ".log")):
            n = sum(_noisy(ln) for ln in open(p, errors="replace"))
            if n:
                out.append(("INFO", f"{f}: {n} log-noise lines (python -m a8w4.exp clean "
                                    f"--inplace {os.path.relpath(p)})"))
    if name not in indexed:
        out.append(("WARN", "not in experiments/index.csv (python -m a8w4.exp add --dir "
                            f"{name} ...)"))
    return out


def cmd_lint(argv):
    ap = argparse.ArgumentParser(prog="python -m a8w4.exp lint")
    ap.add_argument("dirs", nargs="*", help="experiment dirs (default: all)")
    ap.add_argument("--quiet-info", action="store_true", help="hide INFO lines")
    a = ap.parse_args(argv)
    rows = read_index()
    bad = [i + 2 for i, r in enumerate(rows) if None in r or None in r.values()
           or r["verdict"] not in VERDICTS]
    if bad:
        print(f"ERROR index.csv: malformed rows (line numbers) {bad}")
    indexed = {r["dir"] for r in rows}
    dirs = a.dirs or sorted(d for d in os.listdir(EXP_DIR)
                            if os.path.isdir(os.path.join(EXP_DIR, d)) and re.match(r"20\d{6}_", d))
    errors = 0
    for d in dirs:
        path = d if os.path.isdir(d) else os.path.join(EXP_DIR, d)
        probs = [p for p in lint_dir(path, indexed) if not (a.quiet_info and p[0] == "INFO")]
        for level, msg in probs:
            print(f"{level:5s} {os.path.basename(path.rstrip('/'))}: {msg}")
        errors += sum(level == "ERROR" for level, _ in probs)
    missing = sorted(r["dir"] for r in rows if r["dir"] != "-"
                     and not os.path.isdir(os.path.join(EXP_DIR, r["dir"])))
    if missing:
        print(f"WARN  index.csv: rows point to missing dirs {sorted(set(missing))}")
    sys.exit(1 if errors or bad else 0)


TOOLS = {"grep": cmd_grep, "ls": cmd_ls, "add": cmd_add, "new": cmd_new, "lint": cmd_lint,
         "clean": cmd_clean}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in TOOLS:
        print(__doc__)
        sys.exit(0 if sys.argv[1:2] in ([], ["-h"], ["--help"]) else 2)
    TOOLS[sys.argv[1]](sys.argv[2:])


if __name__ == "__main__":
    main()
