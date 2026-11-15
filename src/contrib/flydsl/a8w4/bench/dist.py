# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""8-GPU harness: helpers the multi-GPU tests and experiments share, and an A/B tool for
megamoe (``a8w4.megamoe.MegaMoeEpA8W4``) variants in ONE job (interleaved, ABBA).

    python -m a8w4.bench dist -t 4096 32768 -v default lag=4 "c=bcm=0/1" mode=local

Variant spec: ``[label=]default`` or ``[label=]key=value,...`` with
  producers=N   producer workgroups (a multiple of 8; default 16)
  lag=N         gemm2 trails gemm1 by N block pairs of a queue (default 2)
  bcm=B1/B2     gemm1 / gemm2 W cache modifier of every call (default: the instance's own
                choice, nt below ``big_tokens`` per rank, 0/1 from there)
  mode=M        p2p (normal) | ldisp | lcomb | local: dispatch writes and / or combine
                atomics go to local scratch instead of the peers (the flag / counter
                table stays remote), so p2p - local = the exposed communication
                (log_20261115 3.6); before every timed non-p2p call an untimed p2p call
                refills the receive buffers. Only p2p outputs are checked.
A T-token batch is split evenly over the 8 ranks (DP attention). Timing as
tests/test_a8w4_megamoe_ep8.py: L2 flush + the next of ``--wcopies`` weight copies, host
barrier, GPU spin, MORI device barrier; e2e = the slowest rank, median over iterations;
stages = the slowest rank's median. The first variant also runs as "<first>#2" (noise
floor; ``--no-dup``).
"""

import argparse
import dataclasses
import inspect
import itertools
import os
import socket
import statistics
import sys

import torch
import torch.distributed as dist

from .. import ref as R

WORLD = 8
H, I, E_GLOBAL, TOPK = 4096, 2048, 256, 8  # MiMo-V2.6-Flash EP8
E_LOCAL = E_GLOBAL // WORLD
COS_TOL = 5e-3
SPIN_US = 3000  # GPU spin while the host enqueues one layer
MEGA_STAGES = ("quant", "prep", "plan", "mega", "finish")
FP8 = torch.float8_e4m3fnuz


# =============================================================================
# process group / timing helpers (tests/test_a8w4_{megamoe,mori}_*.py use these)
# =============================================================================


def skip_reason(need_mori=True):
    """Why the 8-GPU tests cannot run here (None: they can)."""
    if need_mori:
        try:
            import mori  # noqa: F401
        except ImportError:
            return "needs mori"
    if torch.cuda.device_count() < WORLD:
        return f"needs {WORLD} GPUs"
    if "gfx942" not in torch.cuda.get_device_properties(0).gcnArchName:
        return "a8w4 kernels need gfx942 (fp8 FNUZ MFMA)"
    return None


def free_port():
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def split_tokens(total, world=WORLD):
    """Tokens per rank of a ``total``-token batch (DP attention: even shares)."""
    return [total // world + (r < total % world) for r in range(world)]


def init(rank, port, world=WORLD, mori=True):
    """This process's rank of a local ``world``-GPU group (gloo + nccl); ``mori``: also
    MORI shmem on it (megamoe's symmetric buffers, MORI dispatch / combine)."""
    os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"] = "localhost", str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="cpu:gloo,cuda:nccl", rank=rank, world_size=world,
                            device_id=torch.device("cuda", rank))
    if mori:
        import mori as _mori

        torch._C._distributed_c10d._register_process_group("mori", dist.group.WORLD)
        _mori.shmem.shmem_torch_process_group_init("mori")


def finish():
    torch.cuda.synchronize()
    dist.barrier()
    dist.destroy_process_group()


def spawn(worker, args, world=WORLD):
    """Run ``worker(rank, port, *args)`` on ``world`` processes."""
    torch.multiprocessing.spawn(worker, args=(free_port(), *args), nprocs=world, join=True)


def gather_rows(t, ns):
    """Every rank's rows of ``t`` ([ns[rank], ...]), concatenated in rank order."""
    buf = t.new_zeros((max(max(ns), 1), *t.shape[1:]))
    buf[: t.shape[0]] = t
    parts = [torch.empty_like(buf) for _ in range(len(ns))]
    dist.all_gather(parts, buf)
    return torch.cat([p[:n] for p, n in zip(parts, ns)])


def align_mori():
    """Device-side barrier of all ranks (MORI shmem) on the current stream."""
    import mori

    mori.shmem.shmem_barrier_on_stream(torch.cuda.current_stream())


_BARRIER = []


def align_rccl():
    """Device-side barrier of all ranks without MORI: a 1-element RCCL all-reduce the
    current stream waits for."""
    if not _BARRIER:
        _BARRIER.append(torch.zeros(1, device="cuda"))
    dist.all_reduce(_BARRIER[0])


def spin_cycles(us):
    """``torch.cuda._sleep`` cycles for about ``us`` microseconds of GPU spin."""
    torch.cuda._sleep(1000)
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    torch.cuda._sleep(1 << 22)
    e.record()
    e.synchronize()
    return int((1 << 22) * us / (s.elapsed_time(e) * 1e3))


def timed(fn, spin, marks=None, align=align_mori):
    """One timed call, all ranks starting together (flush, host barrier, GPU spin while
    the host enqueues, device barrier) -> (e2e us, {stage: us} from ``marks``: the
    (stage, event) list megamoe fills)."""
    R.flush_cache()
    dist.barrier()
    torch.cuda._sleep(spin)
    align()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    fn()
    e.record()
    torch.cuda.synchronize()
    stages = {}
    if marks:
        stages = {name: a.elapsed_time(b) * 1e3 for (_, a), (name, b) in zip(marks, marks[1:])}
        marks.clear()
    return s.elapsed_time(e) * 1e3, stages


class Spans:
    """Events around each stage of a layer; ``isolated``: a device barrier (``align``)
    between the stages, outside the timed spans."""

    def __init__(self, isolated, align=align_mori):
        self.isolated, self.align, self.spans = isolated, align, []

    def __call__(self, name=None):
        if self.spans:
            self.spans[-1][2].record()
            if self.isolated and name:
                self.align()
        if name:
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            self.spans.append((name, s, e))

    def times(self):
        """-> ({stage: us}, us from the first start to the last end)."""
        t = {n: s.elapsed_time(e) * 1e3 for n, s, e in self.spans}
        return t, self.spans[0][1].elapsed_time(self.spans[-1][2]) * 1e3


def slowest_median(per_rank):
    """per_rank: [rank][iteration] us -> median over iterations of the slowest rank."""
    return statistics.median([max(r[i] for r in per_rank) for i in range(len(per_rank[0]))])


def stage_max(per_rank, stages):
    """per_rank: [rank][iteration]{stage: us} -> {stage: slowest rank's median}."""
    return {s: max(statistics.median([it[s] for it in r]) for r in per_rank) for s in stages}


def check_megamoe(mega, raw, ns, x, tw, ti, act, situ, run=None):
    """One megamoe call (``run()``, default ``mega(x, ti, tw)``) vs the fp32 reference
    (each rank evaluates its experts on all ranks' fp8 tokens, summed over the ranks)
    -> [(rows needed, cos diff)] of every rank; asserts every rank is within tolerance."""
    rank, n = mega.rank, ns[mega.rank]
    y = (run() if run else mega(x, ti, tw))[:n].float()
    x_ref = (mega.x8[:n].view(FP8).float().view(n, H // 128, 128)
             * mega.xs[:n].unsqueeze(-1)).view(n, H)
    X, TW, TI = (gather_rows(t, ns) for t in (x_ref, tw, ti))
    ref = R.ref_moe(X, TW, TI, raw, rank * E_LOCAL, act, situ, quant_x=False)
    dist.all_reduce(ref)
    off = sum(ns[:rank])
    diff = R.cos_diff(ref[off:off + n], y) if n else 0.0
    res = [None] * len(ns)
    dist.all_gather_object(res, (mega.rows_needed(), diff))
    bad = [(r, *v) for r, v in enumerate(res) if v[0] > mega.s_cap or not v[1] < COS_TOL]
    assert not bad, f"tokens per rank {ns}: (rank, rows needed, cos diff) {bad}"
    return res


# =============================================================================
# megamoe variants
# =============================================================================

MODES = ("p2p", "ldisp", "lcomb", "local")
DISP = ("recv_a", "recv_scale", "down_ids", "recv_w")
KEYS = ("producers", "lag", "bcm", "mode")


def _defaults():
    from ..megamoe import MegaMoeEpA8W4

    p = inspect.signature(MegaMoeEpA8W4.__init__).parameters
    return p["producers"].default, p["lag"].default


@dataclasses.dataclass
class MegaVariant:
    label: str
    spec: str
    producers: int = 16
    lag: int = 2
    bcm: tuple = None
    mode: str = "p2p"


def parse_mega_variant(text):
    text = text.strip()
    label = None
    head, eq, rest = text.partition("=")
    if eq and head.strip() not in KEYS:
        label, text = head.strip(), rest.strip()
    producers, lag = _defaults()
    v = MegaVariant(label or text, text, producers, lag)
    if text in ("", "default"):
        return v
    for item in text.split(","):
        k, _, val = (s.strip() for s in item.partition("="))
        if k in ("producers", "lag"):
            setattr(v, k, int(val))
        elif k == "bcm":
            v.bcm = tuple(int(b) for b in val.split("/"))
            assert len(v.bcm) == 2, f"bcm=B1/B2, got {val!r}"
        elif k == "mode":
            if val not in MODES:
                raise SystemExit(f"mode {val!r}: one of {MODES}")
            v.mode = val
        else:
            raise SystemExit(f"megamoe variant {item!r}: keys are {KEYS} (or 'default')")
    return v


def _mode_tables(mega, scratch, mode):
    """Peer tables of ``mode``: the dispatch / combine targets point at local scratch."""
    real = dict(mega.tables)
    if mode == "p2p":
        return real
    if not scratch:
        bufs = {n: torch.empty((WORLD,) + tuple(getattr(mega, n).shape),
                               dtype=getattr(mega, n).dtype, device="cuda")
                for n in DISP + ("out",)}
        scratch["bufs"] = bufs  # kept alive with the tables
        scratch["ptrs"] = {n: torch.tensor([b[p].data_ptr() for p in range(WORLD)],
                                           dtype=torch.int64, device="cuda")
                           for n, b in bufs.items()}
    names = {"ldisp": DISP, "lcomb": ("out",), "local": DISP + ("out",)}[mode]
    return dict(real, **{n: scratch["ptrs"][n] for n in names})


def _sweep_worker(rank, port, points, variants, iters, warmup, wcopies, act, out, name, meta):
    init(rank, port)
    from ..kernels import ep_mega as K
    from ..megamoe import MegaMoeEpA8W4
    from . import core

    situ = R.SITU if act == "situv2" else None
    W, raw = R.make_weights(E_LOCAL, H, I, seed=rank)
    nxt = itertools.cycle(R.weight_copies(W, wcopies)).__next__
    max_tokens = max(max(ns) for ns in points)
    megas, scratch, runs = {}, {}, {}
    for v in variants:
        key = (v.producers, v.lag)
        if key not in megas:
            megas[key] = MegaMoeEpA8W4(W, rank=rank, max_tokens=max_tokens, topk=TOPK,
                                       producers=v.producers, lag=v.lag, act=act, situ=situ)
            scratch[key] = {}
        m = megas[key]
        launch = None
        if v.bcm is not None:
            launch = K.compile_ep_mega(H=m.H, I=m.I, TOPK=m.topk, token_shift=m.token_shift,
                                       producers=m.producers, lag=m.lag, ACT=m.act,
                                       BCM1=v.bcm[0], BCM2=v.bcm[1])
        runs[v.label] = (m, launch, _mode_tables(m, scratch[key], v.mode), v.mode)

    def call(label, x, ti, tw, marks=None):
        m, launch, tables, _ = runs[label]
        saved = m.launch, m.launch_big, m.tables
        if launch is not None:
            m.launch = m.launch_big = launch
        m.tables = tables
        try:
            return m(x, ti, tw, marks=marks)
        finally:
            m.launch, m.launch_big, m.tables = saved

    data = [R.make_routing(ns[rank], H, E_GLOBAL, TOPK, seed=7919 * rank + ns[rank])
            for ns in points]
    worst = {}
    for v in variants:
        if v.mode != "p2p":
            continue
        m = runs[v.label][0]
        for ns, (x, tw, ti) in zip(points, data):
            res = check_megamoe(m, raw, ns, x, tw, ti, act, situ,
                                run=lambda: call(v.label, x, ti, tw))
            worst[v.label] = max(worst.get(v.label, 0.0), max(d for _, d in res))
    spin = spin_cycles(SPIN_US)
    labels = [v.label for v in variants]
    rows = []
    for ns, (x, tw, ti) in zip(points, data):
        marks = []
        for _ in range(warmup):
            for lab in labels:
                call(lab, x, ti, tw, marks)
                marks.clear()
        torch.cuda.synchronize()
        ts = {lab: [] for lab in labels}
        st = {lab: [] for lab in labels}
        for it in range(iters):
            for lab in (labels if it % 2 == 0 else labels[::-1]):
                m = runs[lab][0]
                if runs[lab][3] != "p2p":  # a normal call: receivers hold this call's rows
                    m(x, ti, tw)
                m.W = nxt()
                t, s = timed(lambda lab=lab: call(lab, x, ti, tw, marks), spin, marks)
                ts[lab].append(t)
                st[lab].append(s)
        res = [None] * WORLD
        dist.all_gather_object(res, (ts, st))
        if rank == 0:
            rows.append((sum(ns), {lab: (slowest_median([r[0][lab] for r in res]),
                                         stage_max([r[1][lab] for r in res], MEGA_STAGES))
                                   for lab in labels}))
    if rank == 0:
        rec = core.Recorder(out, name, meta)
        rec.text(f"megamoe EP8 (H {H}, I {I}, {E_GLOBAL} experts, {E_LOCAL} per rank, top-{TOPK}"
                 f", {act}); {iters} iterations, variants interleaved (ABBA), {wcopies} weight "
                 "copies; e2e = slowest rank, median; stages = slowest rank's median; worst cos "
                 "vs reference: " + ", ".join(f"{k} {w:.1e}" for k, w in worst.items()) + "\n")
        base, dup = labels[0], (labels[1] if labels[1:2] == [labels[0] + "#2"] else None)
        for T, res in rows:
            b = res[base][0]
            noise = max(core.MIN_NOISE, abs(res[dup][0] / b - 1) * 100 if dup else 0.0)
            table = []
            for v in variants:
                e2e, stg = res[v.label]
                d = (e2e / b - 1) * 100
                verdict = ("base" if v.label == base else "noise floor" if v.label == dup
                           else core.verdict(d, noise))
                table.append([v.label, f"`{v.spec}`", f"{e2e:.1f}", f"{d:+.1f}", verdict]
                             + [f"{stg[s]:.1f}" for s in MEGA_STAGES])
                rec.row(T=T, tokens_per_rank=T // WORLD, variant=v.label, spec=v.spec,
                        producers=v.producers, lag=v.lag, bcm=v.bcm, mode=v.mode, e2e_us=e2e,
                        delta_pct=d, noise_pct=noise, verdict=verdict, stages=stg,
                        cos_ref=worst.get(v.label), wcopies=wcopies)
            rec.text(f"#### {T} tokens ({T // WORLD} per rank), noise band +-{noise:.1f}%\n\n"
                     + core.md_table(["variant", "spec", "e2e us", "delta %", "verdict"]
                                     + list(MEGA_STAGES), table, right=(2, 3, 5, 6, 7, 8, 9))
                     + "\n")
        rec.done()
    finish()


def main(argv=None):
    from . import core

    ap = argparse.ArgumentParser(prog="python -m a8w4.bench dist", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-t", "--tokens", type=int, nargs="+", default=[4096, 8192, 32768],
                    help="tokens of the batch in total (split evenly over the 8 ranks)")
    ap.add_argument("-v", "--variants", nargs="+", default=["default"])
    ap.add_argument("--act", default="silu", choices=R.ACTS)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--wcopies", type=int, default=R.WCOPIES)
    ap.add_argument("--no-dup", action="store_true")
    core.add_out_args(ap, "dist")
    a = ap.parse_args(argv)
    if not a.verbose:
        core.quiet()
    reason = skip_reason()
    if reason:
        sys.exit(reason)
    variants = [parse_mega_variant(s) for s in a.variants]
    labels = [v.label for v in variants]
    if len(set(labels)) < len(labels):
        raise SystemExit(f"duplicate variant labels {labels}: name them (label=spec)")
    if not a.no_dup:
        variants.insert(1, dataclasses.replace(variants[0], label=variants[0].label + "#2"))
    meta = core.run_meta("dist", a)
    points = [split_tokens(n) for n in a.tokens]
    spawn(_sweep_worker, (points, variants, a.iters, a.warmup, a.wcopies, a.act, a.out, a.name,
                          meta))


if __name__ == "__main__":
    main()
