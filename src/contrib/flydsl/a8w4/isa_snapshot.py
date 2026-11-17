# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""ISA snapshot / diff of every kernel variant the package compiles: the safety net of
a structural refactor (``prof isa-snapshot`` / ``prof isa-diff``).

    python -m pyhip.contrib.flydsl.a8w4.prof isa-snapshot --out /tmp/a8w4_isa/base
    python -m pyhip.contrib.flydsl.a8w4.prof isa-snapshot --out /tmp/a8w4_isa/new
    python -m pyhip.contrib.flydsl.a8w4.prof isa-diff /tmp/a8w4_isa/base /tmp/a8w4_isa/new

isa-snapshot runs a fixed matrix of jobs (``--list`` prints them), several worker
processes per GPU, each with an empty FlyDSL cache (a cached kernel is not compiled, so
not seen) and the compile hooks below. Matrix:

  moe:<case>:<act>:<family>  ``MOE.run`` of every tuner candidate of the family (and
                             every ep3k preset) at every tuned token count plus the
                             token counts where a compile-time choice flips (tp_prep
                             unroll / route range, ep_prep multi-WG)
  dyn:<act>:<x>              ``a8w4_moe_dyn`` (ep8 shape) per ep3k preset, x = bf16 /
                             fp8 per (token, 128) / fp8 per token, capacity 256 / 4096
  mega:<act>                 the megamoe kernels, compiled only (no MORI needed)

Not covered: bench-only knob overrides and megamoe token capacities other than 4096.

Every compiled kernel module's final ISA is normalized (kernel symbol names ->
``KERNEL<i>``, comments and blank lines dropped), stored content-addressed in
``OUT/isa/<sha>.s`` and recorded in ``OUT/jobs/<job>.jsonl`` with the matrix item that
first compiled it. isa-diff compares the two sets of normalized ISA hashes: equal sets
mean byte-identical code generation up to kernel names. Exit status 1 on any difference.

isa-cover checks the matrix against a real workload: it runs CMD (and every Python
process it spawns, through a ``sitecustomize`` hook) with the same capture and lists
the kernels CMD compiled that the snapshot lacks:

    python -m pyhip.contrib.flydsl.a8w4.prof isa-cover /tmp/a8w4_isa/base -- \
        python -m pytest -q a8w4/tests/test_a8w4_moe.py
"""

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

PKG = __package__
ACTS = ("silu", "situv2")
CASE_KEYS = {"tp8": "mimo-v2.6-flash-tp8", "tp4": "mimo-v2.6-flash-tp4",
             "ep8": "mimo-v2.6-flash-ep8"}
FAMILIES = ("tp2k", "tp1k", "tp1kq", "ep3k", "ep1k")
DYN_X = ("bf16", "fp8g128", "fp8tok")
DYN_CAPS = (256, 4096)  # ep_prep: single-WG (M * topk <= 2048) / multi-WG
MEGA_MAX_TOKENS = 4096
RES_FIELDS = ("vgpr_count", "agpr_count", "sgpr_count", "vgpr_spill_count",
              "sgpr_spill_count", "group_segment_fixed_size", "private_segment_fixed_size")


# =============================================================================
# matrix
# =============================================================================


def isa_tokens():
    """Tuned token counts + the counts where a token-dependent compile option flips:
    tp_prep's histogram unroll (every 128 tokens up to 2048 at topk 8) and route range
    (> 2048 / 4096 / 8192), ep_prep's multi-WG path (> 256)."""
    from . import tune

    extra = [128 * h for h in range(1, 17)] + [257, 2049, 4097, 8193]
    return sorted(set(tune.TOKENS) | set(extra))


def _case(name):
    from . import tune

    H, I, E, EGl, K, e0 = tune.CASES[CASE_KEYS[name]]
    return (H, I, E, EGl, K), e0


def _family_configs(shape, family):
    from . import moe, tune
    from .kernels import ep_gemm as EG

    cands = [c for c in tune.candidates(shape) if c["kind"] == family]
    if family == "ep3k":  # presets the tuner skips (e.g. l128_v1)
        names = {moe.MOE.name_of(c) for c in cands}
        for p in EG.CFGS:
            c = dict(kind="ep3k", cfg=p)
            if (moe.MOE.name_of(c) not in names
                    and moe.MOE.why_not(c, moe.shape_vocab(shape)) is None):
                cands.append(c)
    return cands


def jobs():
    """{job name: (kind, args)} of the full matrix."""
    out = {}
    for case in CASE_KEYS:
        shape, _ = _case(case)
        for act in ACTS:
            for fam in FAMILIES:
                if _family_configs(shape, fam):
                    out[f"moe:{case}:{act}:{fam}"] = ("moe", (case, act, fam))
    for act in ACTS:
        for xf in DYN_X:
            out[f"dyn:{act}:{xf}"] = ("dyn", (act, xf))
    for act in ACTS:
        out[f"mega:{act}"] = ("mega", (act,))
    return out


def _situ(act):
    from . import ref as R

    return R.SITU if act == "situv2" else None


def _moe_items(case, act, fam, tokens):
    from . import moe
    from . import ref as R

    shape, e0 = _case(case)
    H, I, E, EGl, K = shape
    cfgs = _family_configs(shape, fam)
    st = {}
    yield "make_weights", lambda: st.update(W=R.make_weights(E, H, I)[0])
    for t in tokens:
        x, tw, ti = R.make_routing(t, H, EGl, K)
        for c in cfgs:
            if moe.MOE.why_not(c, moe.shape_vocab(shape, t, act)) is None:
                yield (f"t={t} {moe.MOE.name_of(c)}",
                       lambda c=c, x=x, tw=tw, ti=ti: moe.MOE.run(
                           c, x, st["W"], tw, ti, expert_offset=e0, num_experts=EGl,
                           act=act, situ=_situ(act)))


def _dyn_items(act, xf):
    import torch

    from . import moe
    from . import ref as R
    from .kernels import ep_gemm as EG

    shape, e0 = _case("ep8")
    H, I, E, EGl, K = shape
    st = {}
    yield "make_weights", lambda: st.update(W=R.make_weights(E, H, I)[0])
    for cap in DYN_CAPS:
        x, tw, ti = R.make_routing(cap, H, EGl, K)
        m_valid = torch.tensor([cap * 3 // 4], dtype=torch.int32, device="cuda")
        if xf == "bf16":
            xs = None
        elif xf == "fp8g128":
            x, xs = R.quant_fp8_groups(x, 128)
        else:
            xs = x.float().abs().amax(-1).clamp(min=1e-30) / 240.0
            x = (x.float() / xs[:, None]).to(torch.float8_e4m3fnuz)
        for p in EG.CFGS:
            c = dict(kind="ep3k", cfg=p)
            if moe.MOE.why_not(c, moe.shape_vocab(shape, cap, act)) is None:
                yield (f"cap={cap} {moe.MOE.name_of(c)}",
                       lambda c=c, x=x, xs=xs, tw=tw, ti=ti, m_valid=m_valid: moe.a8w4_moe_dyn(
                           x, st["W"], tw, ti, m_valid, x_scale=xs, expert_offset=e0,
                           num_experts=EGl, act=act, situ=_situ(act), config=c))


def _mega_items(act):
    """The compiles of ``MegaMoeEpA8W4.__init__`` (ep8 shape, default producers / lag)."""
    from .kernels import ep_mega as K
    from .kernels import ep_mega_comm as C

    (H, I, E, EGl, topk), _ = _case("ep8")
    token_shift = max(int(MEGA_MAX_TOKENS - 1).bit_length(), 1)
    kw = dict(H=H, I=I, TOPK=topk, token_shift=token_shift, producers=16, lag=2, ACT=act)
    kernels = {
        "ep_mega": (K.ep_mega_kernel, kw),
        "ep_mega big": (K.ep_mega_kernel, dict(kw, BCM1=0, BCM2=1)),
        "quant": (C.mega_quant_kernel, dict(H=H)),
        "prep": (C.mega_prep_kernel, {}),
        "plan": (C.mega_plan_kernel, {}),
        "finish": (C.mega_finish_kernel, {}),
    }
    for name, (kern, key) in kernels.items():
        yield name, lambda kern=kern, key=key: kern.preload(**key)


def _items(kind, args, tokens):
    if kind == "moe":
        return _moe_items(*args, tokens)
    if kind == "dyn":
        return _dyn_items(*args)
    return _mega_items(*args)


# =============================================================================
# ISA capture
# =============================================================================


def normalize(text):
    """Final ISA with kernel symbol names -> KERNEL<i> (in order of appearance),
    comment and blank lines dropped: (normalized text, [kernel names])."""
    names = re.findall(r"^\s*\.amdhsa_kernel\s+(\S+)", text, re.M)
    for n in sorted(set(names), key=len, reverse=True):  # longest first: no partial hits
        text = text.replace(n, f"\0{names.index(n)}\0")
    text = re.sub(r"\0(\d+)\0", r"KERNEL\1", text)
    lines = []
    for ln in text.splitlines():
        ln = re.sub(r"\s*;.*$", "", ln).rstrip()
        if ln:
            lines.append(ln)
    return "\n".join(lines) + "\n", names


def code_part(norm):
    """Instructions of every function body (label .. .Lfunc_end): equal code parts with
    different full texts mean a metadata-only change (e.g. kernarg layout)."""
    out, on = [], False
    for ln in norm.splitlines():
        if re.fullmatch(r"KERNEL\d+:", ln):
            on = True
        elif ln.startswith(".Lfunc_end"):
            on = False
        if on:
            out.append(ln)
    return "\n".join(out)


def resources(norm):
    res = {}
    for m in re.finditer(r"^\s+\.(\w+):\s+(\d+)$", norm, re.M):
        if m[1] in RES_FIELDS and m[1] not in res:
            res[m[1]] = int(m[2])
    return res


def _sha(s):
    return hashlib.sha256(s.encode()).hexdigest()[:20]


class _Recorder:
    def __init__(self, out, job, scratch):
        self.out, self.job, self.scratch = Path(out), job, Path(scratch)
        self.item, self.seq, self.started = None, 0, 0
        (self.out / "isa").mkdir(parents=True, exist_ok=True)
        (self.out / "jobs").mkdir(parents=True, exist_ok=True)
        self.fh = open(self.out / "jobs" / f"{_fname(job)}.jsonl", "w")

    def write(self, rec):
        self.fh.write(json.dumps(rec) + "\n")
        self.fh.flush()

    def add_isa(self, path):
        raw = Path(path).read_text()
        norm, names = normalize(raw)
        h = _sha(norm)
        dst = self.out / "isa" / f"{h}.s"
        if not dst.exists():
            tmp = dst.with_suffix(f".{os.getpid()}.tmp")
            tmp.write_text(norm)
            os.replace(tmp, dst)
        self.write(dict(job=self.job, item=self.item, seq=self.seq, kernels=names, full=h,
                        code=_sha(code_part(norm)), res=resources(norm)))
        self.seq += 1


def _fname(job):
    return job.replace(":", "__")


def _install_hooks(rec):
    """Patch FlyDSL 0.3.2's dump path: the per-stage MLIR dumps are not written (the
    final ISA is all we need) and each final ISA goes to the recorder."""
    import flydsl.compiler.jit_function as jf

    orig_isa = jf._dump_isa
    orig_names = jf._infer_kernel_names_from_asm

    def infer_names(asm):  # called once at the start of every dumped compile
        rec.started += 1
        return orig_names(asm)

    def dump_ir(stage, *, dump_dir, asm):
        dump_dir.mkdir(parents=True, exist_ok=True)
        return dump_dir / f"{stage}.mlir"

    def dump_isa(*, dump_dir, ctx, asm, verify, stage_name="15_final_isa"):
        tmp = rec.scratch / "isa"
        out = orig_isa(dump_dir=tmp, ctx=ctx, asm=asm, verify=verify, stage_name="final")
        if out is None:
            rec.write(dict(job=rec.job, item=rec.item, seq=rec.seq, error="no ISA dumped"))
            rec.seq += 1
        else:
            rec.add_isa(out)
            os.unlink(out)
        return out

    jf._infer_kernel_names_from_asm = infer_names
    jf._dump_ir = dump_ir
    jf._dump_isa = dump_isa
    jf._extract_llvm_ir = lambda module: None  # the .ll dump: not needed


def worker(argv):
    ap = argparse.ArgumentParser(prog="prof isa-worker")
    ap.add_argument("--job", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", default=None)
    a = ap.parse_args(argv)
    import torch

    kind, args = jobs()[a.job]
    tokens = [int(t) for t in a.tokens.split(",")] if a.tokens else isa_tokens()
    rec = _Recorder(a.out, a.job, os.environ["FLYDSL_DUMP_DIR"])
    _install_hooks(rec)
    for label, run in _items(kind, args, tokens):
        rec.item = label
        try:
            run()
            torch.cuda.synchronize()
        except Exception as e:  # noqa: BLE001 (recorded; the diff shows it)
            rec.write(dict(job=a.job, item=label, error=f"{type(e).__name__}: {e}"[:500]))
    rec.write(dict(job=a.job, done=True, compiles=rec.started, recorded=rec.seq))
    if rec.started != rec.seq:
        raise SystemExit(f"{rec.started} compiles but {rec.seq} ISAs recorded")


# =============================================================================
# driver
# =============================================================================


def _git_head():
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        r = subprocess.run(["git", "-C", here, "rev-parse", "HEAD"], capture_output=True,
                           text=True)
        dirty = subprocess.run(["git", "-C", here, "status", "--porcelain", "--", "."],
                               capture_output=True, text=True).stdout.strip()
        return r.stdout.strip() + ("-dirty" if dirty else "")
    except OSError:
        return "?"


def snapshot(argv):
    from .tune import _gpu_list, busy_gpus

    ap = argparse.ArgumentParser(prog="prof isa-snapshot")
    ap.add_argument("--out", required=False)
    ap.add_argument("--gpus", default="0-7")
    ap.add_argument("--workers-per-gpu", type=int, default=3)
    ap.add_argument("--jobs", nargs="+", default=None, help="regexes on job names")
    ap.add_argument("--tokens", default=None, help="comma list (default: isa_tokens())")
    ap.add_argument("--list", action="store_true", help="print the job names and exit")
    a = ap.parse_args(argv)
    names = list(jobs())
    if a.jobs:
        names = [n for n in names if any(re.search(p, n) for p in a.jobs)]
    if a.list:
        print("\n".join(names))
        return
    if not a.out:
        raise SystemExit("--out is required")
    gpus = _gpu_list(a.gpus)
    busy = busy_gpus(gpus)
    if busy:
        raise SystemExit(f"GPUs busy: {busy}")
    locks = []
    for g in gpus:
        fh = open(f"/tmp/gpu-{g}.lock", "w")
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise SystemExit(f"GPU {g} is locked by another job (/tmp/gpu-{g}.lock)")
        locks.append(fh)
    out = Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    (out / "logs").mkdir(parents=True)
    t0 = time.time()
    meta = dict(git=_git_head(), started=time.strftime("%Y-%m-%d %H:%M:%S"), jobs=names,
                tokens=a.tokens or ",".join(map(str, isa_tokens())), argv=argv)
    try:
        import flydsl

        meta["flydsl"] = getattr(flydsl, "__version__", "?")
    except ImportError:
        pass
    slots = [g for g in gpus for _ in range(a.workers_per_gpu)]
    pending, running, done = list(names), {}, []
    scratch_root = Path(tempfile.mkdtemp(prefix="a8w4_isa_"))
    try:
        while pending or running:
            while pending and len(running) < len(slots):
                free = [s for s in range(len(slots)) if s not in {v[1] for v in running.values()}]
                job, slot = pending.pop(0), free[0]
                scratch = scratch_root / _fname(job)
                (scratch / "cache").mkdir(parents=True)
                env = dict(os.environ, HIP_VISIBLE_DEVICES=str(slots[slot]),
                           FLYDSL_DUMP_IR="1", FLYDSL_DUMP_DIR=str(scratch),
                           FLYDSL_RUNTIME_CACHE_DIR=str(scratch / "cache"),
                           AITER_LOG_LEVEL=os.environ.get("AITER_LOG_LEVEL", "ERROR"))
                cmd = [sys.executable, "-m", f"{PKG}.prof", "isa-worker", "--job", job,
                       "--out", str(out)] + (["--tokens", a.tokens] if a.tokens else [])
                log = open(out / "logs" / f"{_fname(job)}.log", "w")
                p = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
                running[p] = (job, slot, time.time(), log, scratch)
            time.sleep(2)
            for p in [p for p in running if p.poll() is not None]:
                job, slot, ts, log, scratch = running.pop(p)
                log.close()
                shutil.rmtree(scratch, ignore_errors=True)
                recs = _load_job(out, job)
                nk = sum(1 for r in recs if "full" in r)
                ne = sum(1 for r in recs if "error" in r)
                ok = p.returncode == 0 and any(r.get("done") for r in recs)
                done.append(dict(job=job, rc=p.returncode, kernels=nk, errors=ne,
                                 seconds=round(time.time() - ts), complete=ok))
                print(f"[{len(done)}/{len(names)}] {job}: {nk} kernels, {ne} errors, "
                      f"{time.time() - ts:.0f} s{'' if ok else f', INCOMPLETE (rc {p.returncode})'}",
                      flush=True)
    finally:
        for p in running:
            p.kill()
        shutil.rmtree(scratch_root, ignore_errors=True)
        for fh in locks:
            fh.close()
    meta.update(seconds=round(time.time() - t0), results=done)
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    recs = load(out)
    full = {r["full"] for r in recs if "full" in r}
    print(f"{sum('full' in r for r in recs)} kernel compiles, {len(full)} distinct ISAs, "
          f"{sum('error' in r for r in recs)} errors, {meta['seconds']} s -> {out}")
    if not all(d["complete"] for d in done):
        sys.exit(1)


# =============================================================================
# coverage of a command
# =============================================================================

_SITE = """import os

if os.environ.get("A8W4_ISA_OUT"):
    import atexit

    from {pkg} import isa_snapshot as S

    _rec = S._Recorder(os.environ["A8W4_ISA_OUT"], "proc-%d" % os.getpid(),
                       os.path.join(os.environ["FLYDSL_DUMP_DIR"], str(os.getpid())))
    S._install_hooks(_rec)
    _rec.item = "cmd"
    atexit.register(lambda: _rec.write(dict(job=_rec.job, done=True,
                                            compiles=_rec.started, recorded=_rec.seq)))
"""


def cover(argv):
    if "--" not in argv:
        raise SystemExit("usage: prof isa-cover SNAPSHOT [--out DIR] -- CMD ...")
    i = argv.index("--")
    ap = argparse.ArgumentParser(prog="prof isa-cover")
    ap.add_argument("snapshot")
    ap.add_argument("--out", default=None, help="capture directory (default: a temp dir)")
    a = ap.parse_args(argv[:i])
    cmd = argv[i + 1:]
    out = Path(a.out or tempfile.mkdtemp(prefix="a8w4_isa_cover_"))
    scratch = Path(tempfile.mkdtemp(prefix="a8w4_isa_"))
    (scratch / "cache").mkdir()
    (scratch / "site").mkdir()
    (scratch / "site" / "sitecustomize.py").write_text(_SITE.format(pkg=PKG))
    pp = os.environ.get("PYTHONPATH")
    env = dict(os.environ, A8W4_ISA_OUT=str(out), FLYDSL_DUMP_IR="1",
               FLYDSL_DUMP_DIR=str(scratch), FLYDSL_RUNTIME_CACHE_DIR=str(scratch / "cache"),
               PYTHONPATH=str(scratch / "site") + (os.pathsep + pp if pp else ""))
    try:
        rc = subprocess.run(cmd, env=env, stdout=None, stderr=subprocess.STDOUT).returncode
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    base = {r["full"] for r in load(a.snapshot) if "full" in r}
    recs = load(out)
    got = [r for r in recs if "full" in r]
    lost = [r for r in recs if r.get("done") and r["compiles"] != r["recorded"]]
    miss = {}
    for r in got:
        if r["full"] not in base:
            miss.setdefault(r["full"], r)
    print(f"CMD exit status {rc}; {len(got)} compiles in {len({r['job'] for r in recs})} "
          f"processes, {len({r['full'] for r in got})} distinct ISAs, {len(miss)} not in "
          f"{a.snapshot}")
    for h, r in miss.items():
        print(f"  {','.join(r['kernels'])} {h}")
    if lost:
        print(f"compiles without a captured ISA: {lost}")
    sys.exit(1 if rc or miss or lost or not got else 0)


# =============================================================================
# diff
# =============================================================================


def _load_job(out, job):
    f = Path(out) / "jobs" / f"{_fname(job)}.jsonl"
    return [json.loads(ln) for ln in open(f)] if f.exists() else []


def load(out):
    recs = []
    for f in sorted((Path(out) / "jobs").glob("*.jsonl")):
        recs += [json.loads(ln) for ln in open(f)]
    return recs


def _where(r):
    return f"{r['job']} | {r['item']} | {','.join(r['kernels'])}"


def _res_delta(ra, rb):
    keys = sorted(set(ra) | set(rb))
    return ", ".join(f"{k.replace('_count', '').replace('_fixed_size', '')} "
                     f"{ra.get(k)}->{rb.get(k)}" for k in keys if ra.get(k) != rb.get(k))


def diff(argv):
    ap = argparse.ArgumentParser(prog="prof isa-diff")
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--show", type=int, default=1,
                    help="print a unified diff of the first N changed kernel pairs")
    ap.add_argument("--max", type=int, default=30, help="list at most N entries per section")
    a = ap.parse_args(argv)
    A, B = load(a.a), load(a.b)
    ha, hb = defaultdict(list), defaultdict(list)
    for recs, h in ((A, ha), (B, hb)):
        for r in recs:
            if "full" in r:
                h[r["full"]].append(r)
    only_a = [h for h in ha if h not in hb]
    only_b = [h for h in hb if h not in ha]
    na, nb = sum(map(len, ha.values())), sum(map(len, hb.values()))
    print(f"A {a.a}: {na} compiles, {len(ha)} distinct ISAs")
    print(f"B {a.b}: {nb} compiles, {len(hb)} distinct ISAs")
    print(f"common {len(set(ha) & set(hb))}, only in A {len(only_a)}, only in B {len(only_b)}")
    bad = bool(only_a or only_b)

    jobs_a = {r["job"] for r in A}
    jobs_b = {r["job"] for r in B}
    if jobs_a != jobs_b:
        print(f"\njobs only in A: {sorted(jobs_a - jobs_b)}\njobs only in B: {sorted(jobs_b - jobs_a)}")
    inc = [j for recs in (A, B) for j in sorted({r['job'] for r in recs})
           if not any(r.get("done") for r in recs if r["job"] == j)]
    if inc:
        print(f"\nINCOMPLETE jobs (worker died): {sorted(set(inc))}")
        bad = True
    ea = {(r["job"], r["item"], r["error"]) for r in A if "error" in r}
    eb = {(r["job"], r["item"], r["error"]) for r in B if "error" in r}
    for tag, e in (("errors only in A", ea - eb), ("errors only in B", eb - ea)):
        if e:
            bad = True
            print(f"\n{tag} ({len(e)}):")
            for j, i, m in sorted(e)[: a.max]:
                print(f"  {j} | {i}: {m}")
    if ea & eb:
        print(f"\n{len(ea & eb)} errors in both (pre-existing)")

    # pair the changed ISAs by (job, item, n-th compile of the item) to show what changed
    def positions(recs, changed):
        by_item, out = defaultdict(list), {}
        for r in recs:
            if "full" in r:
                by_item[(r["job"], r["item"])].append(r)
        for (j, i), rs in by_item.items():
            for n, r in enumerate(sorted(rs, key=lambda r: r["seq"])):
                if r["full"] in changed:
                    out[(j, i, n)] = r
        return out

    ra, rb = positions(A, set(only_a)), positions(B, set(only_b))
    pairs = [(ra[k], rb[k]) for k in sorted(set(ra) & set(rb))]
    shown = 0
    if pairs:
        print(f"\nchanged ({len(pairs)} paired by job/item/position):")
        for x, y in pairs[: a.max]:
            kind = "metadata only" if x["code"] == y["code"] else "code"
            print(f"  {_where(x)} -> {','.join(y['kernels'])}: {kind}"
                  f"{'; ' + _res_delta(x['res'], y['res']) if x['res'] != y['res'] else ''}")
        for x, y in pairs[: a.show]:
            fa = Path(a.a) / "isa" / f"{x['full']}.s"
            fb = Path(a.b) / "isa" / f"{y['full']}.s"
            r = subprocess.run(["diff", "-u", str(fa), str(fb)], capture_output=True, text=True)
            lines = r.stdout.splitlines()
            print(f"\n--- {_where(x)}\n" + "\n".join(lines[:80])
                  + (f"\n... ({len(lines) - 80} more lines)" if len(lines) > 80 else ""))
            shown += 1
    paired_a = {id(p[0]) for p in pairs}
    paired_b = {id(p[1]) for p in pairs}
    for tag, hs, h, paired in (("only in A", only_a, ha, paired_a),
                               ("only in B", only_b, hb, paired_b)):
        rest = [r for x in hs for r in h[x] if id(r) not in paired]
        if rest:
            print(f"\nunpaired, {tag} ({len(rest)}):")
            for r in rest[: a.max]:
                print(f"  {_where(r)} {r['full']}")
    print("\nISA IDENTICAL" if not bad else "\nISA DIFFERS")
    sys.exit(1 if bad else 0)
