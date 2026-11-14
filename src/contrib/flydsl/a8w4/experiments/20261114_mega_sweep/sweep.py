"""megamoe (EP8 persistent kernel) knob sweep: ``producers`` (producer WGs, a multiple
of 8) x ``lag`` (gemm2 trails gemm1 by this many block pairs of a queue). One 8-rank job
per setting (a fresh MegaMoeEpA8W4); every point is checked against the fp32 reference
once, then timed as in a8w4/tests/test_a8w4_megamoe_ep8.py (flush, host + device
barrier, slowest rank, median), megamoe only.

    cd src/contrib/flydsl
    python a8w4/experiments/20261114_mega_sweep/sweep.py --producers 8 16 32 --lag 1 2 4
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
        fn = lambda: mega(x, ti, tw, marks=marks)  # noqa: E731
        for _ in range(warmup):
            fn()
            marks.clear()
        torch.cuda.synchronize()
        ts, st = [], []
        for _ in range(iters):
            t, s = TM._timed(fn, spin, marks)
            ts.append(t)
            st.append(s)
        res = [None] * TM.WORLD
        dist.all_gather_object(res, (ts, st))
        if rank == 0:
            med = statistics.median
            e2e = med([max(r[0][i] for r in res) for i in range(iters)])
            stages = {s: max(med([it[s] for it in r[1]]) for r in res) for s in TM.STAGES}
            rows.append((sum(ns), e2e, stages))
    if rank == 0:
        for T, e2e, st in rows:
            print(f"producers {producers:2d} lag {lag} T {T:5d}: e2e {e2e:7.1f} us  "
                  + "  ".join(f"{s} {st[s]:.1f}" for s in TM.STAGES)
                  + f"  (cos <= {worst:.1e})", flush=True)
    torch.cuda.synchronize()
    dist.barrier()
    dist.destroy_process_group()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--tokens", type=int, nargs="+", default=TM.BENCH_TOKENS)
    ap.add_argument("--producers", type=int, nargs="+", default=[8, 16, 32])
    ap.add_argument("--lag", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    a = ap.parse_args()
    reason = TM._skip_reason()
    if reason:
        sys.exit(reason)
    points = [TM.split_tokens(n) for n in a.tokens]
    for p in a.producers:
        for lag in a.lag:
            torch.multiprocessing.spawn(
                _worker, args=(TM._free_port(), points, a.iters, a.warmup, p, lag),
                nprocs=TM.WORLD, join=True)


if __name__ == "__main__":
    main()
