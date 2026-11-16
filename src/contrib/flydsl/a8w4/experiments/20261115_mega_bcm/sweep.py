"""megamoe W cache modifier (gemm1 / gemm2 W loads: bcm 2 = nt vs 0 / 1), cf.
20261115_ep_prio (single-GPU EP: gemm2 bcm 0 / 1 is 1-5% faster). Written before
megamoe.py picked the variant by batch (bcm 2/2 below 512 tokens per rank, else 0/1).

One 8-rank job; ``mega.launch`` and ``launch_big`` are both set to each variant in turn
(same buffers, same plan), each variant checked against the fp32 reference, then the
variants interleave per iteration (timing as tests/test_a8w4_megamoe_ep8.py: flush,
aligned start, slowest rank, median). ``--wcopies N``: every call reads the next of N weight
copies (result*.txt: 1, the same weights every call, which the MALL keeps through the flush
and so favors cached W loads; rotate*.txt: 8, see ../log_20261115.md).

    cd src/contrib/flydsl && python a8w4/experiments/20261115_mega_bcm/sweep.py -t 4096 32768
"""
import itertools
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

VARIANTS = {"bcm22": (2, 2), "bcm20": (2, 0), "bcm00": (0, 0), "bcm21": (2, 1), "bcm01": (0, 1),
            "bcm22b": (2, 2)}  # bcm22b: the base again (noise floor)


def _worker(rank, port, points, iters, warmup, variants, wcopies):
    import mori

    os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"] = "localhost", str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="cpu:gloo,cuda:nccl", rank=rank, world_size=TM.WORLD,
                            device_id=torch.device("cuda", rank))
    torch._C._distributed_c10d._register_process_group("mori", dist.group.WORLD)
    mori.shmem.shmem_torch_process_group_init("mori")
    from a8w4.kernels import ep_mega as K
    from a8w4.megamoe import MegaMoeEpA8W4

    W, raw = R.make_weights(TM.E_LOCAL, TM.H, TM.I, seed=rank)
    max_tokens = max(max(ns) for ns in points)
    mega = MegaMoeEpA8W4(W, rank=rank, max_tokens=max_tokens, topk=TM.TOPK)
    Ws = R.weight_copies(W, wcopies)
    nxt = itertools.cycle(Ws).__next__
    launches = {n: K.ep_mega_kernel.get(H=mega.H, I=mega.I, TOPK=mega.topk,
                                        token_shift=mega.token_shift, producers=mega.producers,
                                        lag=mega.lag, ACT=mega.act, BCM1=b1, BCM2=b2)
                for n, (b1, b2) in variants.items()}
    data = [R.make_routing(ns[rank], TM.H, TM.E_GLOBAL, TM.TOPK, seed=7919 * rank + ns[rank])
            for ns in points]
    worst = {}
    for n, launch in launches.items():
        mega.launch = mega.launch_big = launch
        for ns, (x, tw, ti) in zip(points, data):
            res = TM._check(mega, raw, ns, x, tw, ti, "silu", None)
            worst[n] = max(worst.get(n, 0.0), max(d for _, d in res))
    spin = TM._spin_cycles(TM.SPIN_US)
    rows = []
    for ns, (x, tw, ti) in zip(points, data):
        marks = []

        def fn(n):
            mega.launch = mega.launch_big = launches[n]
            mega.W = nxt()
            mega(x, ti, tw, marks=marks)

        for _ in range(warmup):
            for n in launches:
                fn(n)
                marks.clear()
        torch.cuda.synchronize()
        ts = {n: [] for n in launches}
        st = {n: [] for n in launches}
        for it in range(iters):
            order = list(launches)
            if it % 2:
                order.reverse()
            for n in order:
                t, s = TM._timed(lambda n=n: fn(n), spin, marks)
                ts[n].append(t)
                st[n].append(s)
        res = [None] * TM.WORLD
        dist.all_gather_object(res, (ts, st))
        if rank == 0:
            med = statistics.median
            out = {}
            for n in launches:
                e2e = med([max(r[0][n][i] for r in res) for i in range(iters)])
                stages = {s: max(med([it[s] for it in r[1][n]]) for r in res)
                          for s in TM.STAGES}
                out[n] = (e2e, stages)
            rows.append((sum(ns), out))
    if rank == 0:
        print(f"{iters} iterations (variants interleaved, ABBA order), {wcopies} weight "
              "copies; worst cos vs ref: "
              + ", ".join(f"{n} {w:.1e}" for n, w in worst.items()))
        for T, out in rows:
            b = out[next(iter(launches))][0]
            for n in launches:
                e2e, s = out[n]
                print(f"T {T:5d} {n:6s} e2e {e2e:7.1f} us ({(e2e / b - 1) * 100:+5.1f}%)  "
                      + "  ".join(f"{k} {s[k]:.1f}" for k in TM.STAGES), flush=True)
    torch.cuda.synchronize()
    dist.barrier()
    dist.destroy_process_group()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--tokens", type=int, nargs="+", default=TM.BENCH_TOKENS)
    ap.add_argument("-v", "--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--wcopies", type=int, default=R.WCOPIES)
    a = ap.parse_args()
    reason = TM._skip_reason()
    if reason:
        sys.exit(reason)
    points = [TM.split_tokens(n) for n in a.tokens]
    variants = {n: VARIANTS[n] for n in a.variants}
    torch.multiprocessing.spawn(
        _worker, args=(TM._free_port(), points, a.iters, a.warmup, variants, a.wcopies),
        nprocs=TM.WORLD, join=True)


if __name__ == "__main__":
    main()
