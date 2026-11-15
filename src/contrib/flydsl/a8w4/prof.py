# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Profiling helpers around any command (``python -m a8w4.prof <tool> [options] -- CMD``).

    # L2 / memory counters per kernel: one rocprofv3 pass per counter group, summarized
    python -m a8w4.prof counters --group l2 ea --kernel gemm --out /tmp/cnt -- \\
        python -m a8w4.bench ab --case ep8 -t 1024 -v m64 --rounds 1 --iters 3 --no-dup --no-ref
    # VGPR / SGPR / spills / LDS / scratch of every FlyDSL kernel CMD compiles
    python -m a8w4.prof isa --grep ep_gemm -- \\
        python -m a8w4.bench check --case ep8 -t 1024 -v m64 --repeats 0

counters: rocprofv3 collects a few counters per pass, so each ``--group`` (or ``--pmc``
list) runs CMD once. The summary is the mean per dispatch of every counter, per kernel name
(the names carry the compile options, e.g. bcm, so variants stay apart); the raw CSVs are
deleted unless ``--keep-raw`` (they reach ~60 MB). Profile a short CMD: everything it
launches is counted (filter with ``--kernel``).
isa: runs CMD with a fresh, empty FlyDSL cache (a cached kernel is not compiled, so not
dumped) and the IR / ISA dump on, then reads every ``*/21_final_isa.s``. waves/SIMD: the
VGPR limit (512 per SIMD lane, allocated in 8s, VGPRs + AGPRs). LDS > 64 KB raises nowhere:
LLVM and the launch only print errors (hipErrorIllegalState) and the output is not written.
"""

import argparse
import csv
import glob
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict

COUNTER_GROUPS = {
    "l2": "TCC_HIT_sum TCC_MISS_sum TCC_READ_sum TCC_STREAMING_REQ_sum",
    "ea": "TCC_EA0_RDREQ_sum TCC_EA0_RDREQ_32B_sum TCC_EA0_RDREQ_DRAM_sum",
    "wr": "TCC_EA0_WRREQ_sum TCC_EA0_WRREQ_64B_sum TCC_EA0_WRREQ_DRAM_sum",
}
ISA_FIELDS = ("vgpr_count", "agpr_count", "sgpr_count", "vgpr_spill_count", "sgpr_spill_count",
              "group_segment_fixed_size", "private_segment_fixed_size",
              "max_flat_workgroup_size")


def _split(argv):
    if "--" not in argv:
        raise SystemExit("usage: python -m a8w4.prof <counters | isa> [options] -- CMD ...")
    i = argv.index("--")
    if i == len(argv) - 1:
        raise SystemExit("no command after --")
    return argv[:i], argv[i + 1:]


def _env():
    return dict(os.environ, AITER_LOG_LEVEL=os.environ.get("AITER_LOG_LEVEL", "ERROR"))


def _md(head, rows):
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
                     + ["| " + " | ".join(map(str, r)) + " |" for r in rows])


def parse_counters(out_dir, kernel=None):
    """{kernel name: (dispatches, {counter: mean per dispatch})} of all rocprofv3 counter
    CSVs under ``out_dir``."""
    acc = defaultdict(lambda: defaultdict(dict))
    for f in glob.glob(os.path.join(out_dir, "**", "*counter_collection.csv"), recursive=True):
        with open(f) as fh:
            for r in csv.DictReader(fh):
                name = r["Kernel_Name"]
                if kernel and not re.search(kernel, name):
                    continue
                key = (f, r["Dispatch_Id"])  # dispatch ids restart in every pass
                d = acc[name][r["Counter_Name"]]
                d[key] = d.get(key, 0.0) + float(r["Counter_Value"])
    out = {}
    for name, cs in acc.items():
        n = max(len(v) for v in cs.values())
        out[name] = (n, {c: sum(v.values()) / len(v) for c, v in sorted(cs.items())})
    return out


def counters(argv):
    opts, cmd = _split(argv)
    ap = argparse.ArgumentParser(prog="python -m a8w4.prof counters")
    ap.add_argument("--group", nargs="+", default=["l2", "ea"], choices=sorted(COUNTER_GROUPS))
    ap.add_argument("--pmc", nargs="+", default=None,
                    help="explicit counter lists, one per pass (quoted), instead of --group")
    ap.add_argument("--kernel", default=None, help="regex on the kernel name")
    ap.add_argument("--out", default=None, help="directory (default: a temp dir)")
    ap.add_argument("--keep-raw", action="store_true")
    a = ap.parse_args(opts)
    if not shutil.which("rocprofv3"):
        raise SystemExit("rocprofv3 not found (ROCm >= 6.2: /opt/rocm/bin)")
    out = a.out or tempfile.mkdtemp(prefix="a8w4_counters_")
    passes = a.pmc or [COUNTER_GROUPS[g] for g in a.group]
    for i, pmc in enumerate(passes):
        d = os.path.join(out, f"pass{i}")
        print(f"[pass {i + 1}/{len(passes)}] {pmc}", flush=True)
        r = subprocess.run(["rocprofv3", "--pmc", *pmc.split(), "-d", d, "-o", "c",
                            "--output-format", "csv", "--", *cmd], env=_env(),
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        if r.returncode:
            raise SystemExit(f"rocprofv3 pass {i} failed ({r.returncode}):\n{r.stderr[-3000:]}")
    res = parse_counters(out, a.kernel)
    rows = []
    for name, (n, vals) in sorted(res.items()):
        hit, miss = vals.get("TCC_HIT_sum"), vals.get("TCC_MISS_sum")
        rate = f"{hit / (hit + miss) * 100:.1f}%" if hit is not None and miss else "-"
        rows.append([re.sub(r"\(.*", "", name)[-100:], n,
                     "<br>".join(f"{c} {v / 1e6:.3f}M" for c, v in vals.items()), rate])
    text = _md(["kernel", "dispatches", "mean per dispatch", "L2 hit"], rows)
    print(text)
    with open(os.path.join(out, "summary.md"), "w") as fh:
        fh.write(f"`{' '.join(cmd)}`\n\npasses: {passes}\n\n{text}\n")
    if not a.keep_raw:
        for i in range(len(passes)):
            shutil.rmtree(os.path.join(out, f"pass{i}"), ignore_errors=True)
    print(f"(summary: {os.path.join(out, 'summary.md')})")


def parse_isa(path):
    """Kernel metadata fields of one FlyDSL ``21_final_isa.s``."""
    meta = {}
    for ln in open(path):
        m = re.match(r"\s+\.(\w+):\s+(\S+)", ln)
        if m and m[1] in ISA_FIELDS + ("name",) and m[1] not in meta:
            meta[m[1]] = m[2] if m[1] == "name" else int(m[2])
    return meta


def waves_per_simd(vgpr, agpr=0):
    """VGPR-limited waves per SIMD on gfx942 (wave64: 512 registers per lane, unified
    VGPR + AGPR file allocated in granules of 8)."""
    regs = -(-((-(-vgpr // 4) * 4) + agpr) // 8) * 8
    return min(8, 512 // max(regs, 8))


def isa(argv):
    opts, cmd = _split(argv)
    ap = argparse.ArgumentParser(prog="python -m a8w4.prof isa")
    ap.add_argument("--out", default=None, help="dump directory (default: a temp dir)")
    ap.add_argument("--grep", default=None, help="regex on the kernel name")
    a = ap.parse_args(opts)
    out = a.out or tempfile.mkdtemp(prefix="a8w4_isa_")
    cache = tempfile.mkdtemp(prefix="a8w4_isa_cache_")  # empty: every kernel compiles
    env = dict(_env(), FLYDSL_DUMP_IR="1", FLYDSL_DEBUG_DUMP_ASM="1", FLYDSL_DUMP_DIR=out,
               FLYDSL_RUNTIME_CACHE_DIR=cache)
    r = subprocess.run(cmd, env=env)
    shutil.rmtree(cache, ignore_errors=True)
    rows, warn = [], []
    for f in sorted(glob.glob(os.path.join(out, "*", "*final_isa.s"))):
        m = parse_isa(f)
        name = m.get("name", os.path.basename(os.path.dirname(f)))
        if a.grep and not re.search(a.grep, name):
            continue
        v, ag = m.get("vgpr_count", 0), m.get("agpr_count", 0)
        lds = m.get("group_segment_fixed_size", 0)
        spill = m.get("vgpr_spill_count", 0) + m.get("sgpr_spill_count", 0)
        rows.append([name, v, ag, m.get("sgpr_count", "-"), spill, lds,
                     m.get("private_segment_fixed_size", 0), m.get("max_flat_workgroup_size", "-"),
                     waves_per_simd(v, ag)])
        if spill:
            warn.append(f"{name}: {spill} spills")
        if lds > 65536:
            warn.append(f"{name}: LDS {lds} B > 64 KB: the launch fails without an exception "
                        "(hipErrorIllegalState printed, output not written)")
    print(_md(["kernel", "vgpr", "agpr", "sgpr", "spills", "LDS B", "scratch B", "WG size",
               "waves/SIMD (VGPR)"], rows))
    print("\n".join(["", *warn]) if warn else "\nno spills, LDS <= 64 KB")
    print(f"(dump: {out}; CMD exit status {r.returncode})")
    if not rows:
        print("no kernel was compiled: did CMD launch any FlyDSL kernel?")


def main():
    tools = {"counters": counters, "isa": isa}
    if len(sys.argv) < 2 or sys.argv[1] not in tools:
        print(__doc__)
        sys.exit(0 if sys.argv[1:2] in ([], ["-h"], ["--help"]) else 2)
    tools[sys.argv[1]](sys.argv[2:])


if __name__ == "__main__":
    main()
