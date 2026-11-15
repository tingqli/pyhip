"""How much of megamoe's xGMI traffic is still exposed? (method: luwei/942-fp8-megamoe
tests/contrib/moe/ep8/profile_ep8_mega_overlap.py, applied to the a8w4 megamoe)

Modes (peer pointer tables of ``MegaMoeEpA8W4``; the flag / counter table ``ws`` stays
remote, so prep, arrival waits, task order and the finish barrier are unchanged):
  p2p     normal
  ldisp   producers write the routes (recv_a, recv_scale, down_ids, recv_w) into a local
          scratch copy per destination instead of the destination (the destination
          computes on the rows of the previous p2p call: same inputs, same work)
  lcomb   gemm2's BF16 atomics go to a local scratch copy of every source's ``out``
  local   both
Only p2p's output is correct (checked once). Modes interleave per iteration; e2e is
the slowest rank of each iteration (cold L2, aligned start as in the megamoe test),
median over iterations; p2p - local = the exposed communication.

    cd src/contrib/flydsl && python a8w4/experiments/20261115_mega_overlap/overlap.py -t 4096 32768
"""
import argparse
import os
import statistics
import sys

import torch
import torch.distributed as dist

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "../../..")))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "../../tests")))
import test_a8w4_megamoe_ep8 as TM  # noqa: E402
from a8w4 import ref as R  # noqa: E402

MODES = ("p2p", "ldisp", "lcomb", "local")
DISP = ("recv_a", "recv_scale", "down_ids", "recv_w")


def _worker(rank, port, points, iters, warmup, producers, lag):
    import mori

    os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"] = "localhost", str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="cpu:gloo,cuda:nccl", rank=rank, world_size=TM.WORLD,
                            device_id=torch.device("cuda", rank))
    torch._C._distributed_c10d._register_process_group("mori", dist.group.WORLD)
    mori.shmem.shmem_torch_process_group_init("mori")
    from a8w4.megamoe import MegaMoeEpA8W4

    W, raw = R.make_weights(TM.E_LOCAL, TM.H, TM.I, seed=rank)
    max_tokens = max(max(ns) for ns in points)
    mega = MegaMoeEpA8W4(W, rank=rank, max_tokens=max_tokens, topk=TM.TOPK,
                         producers=producers, lag=lag)
    real = dict(mega.tables)
    scratch = {n: torch.empty((TM.WORLD,) + tuple(getattr(mega, n).shape),
                              dtype=getattr(mega, n).dtype, device="cuda")
               for n in DISP + ("out",)}
    loc = {n: torch.tensor([scratch[n][p].data_ptr() for p in range(TM.WORLD)],
                           dtype=torch.int64, device="cuda") for n in scratch}
    tabs = {
        "p2p": real,
        "ldisp": dict(real, **{n: loc[n] for n in DISP}),
        "lcomb": dict(real, out=loc["out"]),
        "local": dict(real, **{n: loc[n] for n in DISP + ("out",)}),
    }
    data = [R.make_routing(ns[rank], TM.H, TM.E_GLOBAL, TM.TOPK, seed=7919 * rank + ns[rank])
            for ns in points]
    worst = 0.0
    for ns, (x, tw, ti) in zip(points, data):
        res = TM._check(mega, raw, ns, x, tw, ti, "silu", None)
        worst = max(worst, max(d for _, d in res))
    spin = TM._spin_cycles(TM.SPIN_US)
    rows = []
    for ns, (x, tw, ti) in zip(points, data):
        marks = []

        def fn(mode):
            mega.tables = tabs[mode]
            mega(x, ti, tw, marks=marks)
            mega.tables = real

        for _ in range(warmup):
            for m in MODES:
                fn(m)
                marks.clear()
        torch.cuda.synchronize()
        ts = {m: [] for m in MODES}
        st = {m: [] for m in MODES}
        for _ in range(iters):
            fn("p2p")  # the destinations' receive buffers hold this call's rows
            marks.clear()
            for m in MODES:
                t, s = TM._timed(lambda m=m: fn(m), spin, marks)
                ts[m].append(t)
                st[m].append(s)
        res = [None] * TM.WORLD
        dist.all_gather_object(res, (ts, st))
        if rank == 0:
            med = statistics.median
            out = {}
            for m in MODES:
                e2e = med([max(r[0][m][i] for r in res) for i in range(iters)])
                stages = {s: max(med([it[s] for it in r[1][m]]) for r in res)
                          for s in TM.STAGES}
                out[m] = (e2e, stages)
            rows.append((sum(ns), out))
    if rank == 0:
        print(f"producers {producers} lag {lag}, {iters} iterations, cos(p2p) <= {worst:.1e}")
        for T, out in rows:
            b = out["p2p"][0]
            for m in MODES:
                e2e, s = out[m]
                print(f"T {T:5d} {m:6s} e2e {e2e:7.1f} us ({e2e - b:+6.1f})  "
                      + "  ".join(f"{k} {s[k]:.1f}" for k in TM.STAGES), flush=True)
    torch.cuda.synchronize()
    dist.barrier()
    dist.destroy_process_group()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--tokens", type=int, nargs="+", default=[4096, 8192, 32768])
    ap.add_argument("--producers", type=int, default=16)
    ap.add_argument("--lag", type=int, default=2)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=2)
    a = ap.parse_args()
    reason = TM._skip_reason()
    if reason:
        sys.exit(reason)
    points = [TM.split_tokens(n) for n in a.tokens]
    torch.multiprocessing.spawn(
        _worker, args=(TM._free_port(), points, a.iters, a.warmup, a.producers, a.lag),
        nprocs=TM.WORLD, join=True)


if __name__ == "__main__":
    main()
