# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""megamoe_ep_a8w4 on 8 GPUs: the EP8 MoE layer of XiaomiMiMo/MiMo-V2.6-Flash(-RL)
(hidden 4096, moe_intermediate 2048, 256 routed experts, top-8; 32 experts per rank) as one
persistent kernel (``a8w4.megamoe.MegaMoeEpA8W4``), against the fp32 reference and, for
speed, against MORI dispatch -> ``a8w4_moe_dyn`` -> MORI combine (``test_a8w4_mori_ep8``).

A batch of T tokens (the model's total) is split evenly over the ranks (DP attention).

    pytest -q a8w4/tests/test_a8w4_megamoe_ep8.py        # correctness vs the fp32 reference
    python a8w4/tests/test_a8w4_megamoe_ep8.py           # perf: T = 2048 .. 32768, vs MORI
    python a8w4/tests/test_a8w4_megamoe_ep8.py -t 8192 --iters 50 --act situv2

Run from the directory that holds ``a8w4/``; needs 8 gfx942 GPUs and mori.

Correctness: every point runs twice, the points interleaved (stale data from an earlier
call would show up); each rank's output vs the reference (each rank evaluates its experts
on all ranks' fp8 tokens, summed over the ranks), cos diff < 5e-3.
Timing (as test_a8w4_mori_ep8): every iteration flushes L2 + MALL, then a host barrier, a
GPU spin while the host enqueues the layer, and a MORI device barrier, so all ranks start
together; e2e = the layer of the slowest rank, median over the iterations; the two paths
alternate per iteration. Stage split of megamoe (events between its launches, slowest
rank): quant / prep (count exchange) / plan / mega / finish (wait for all ranks).
"""

import argparse
import os
import socket
import statistics
import sys

import pytest
import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from a8w4 import ref as R  # noqa: E402

try:
    import mori
except ImportError:  # the test skips
    mori = None

H, I, E_GLOBAL, TOPK, WORLD = 4096, 2048, 256, 8, 8
E_LOCAL = E_GLOBAL // WORLD
MAX_TOKENS = 4096  # per rank (the MORI path's dispatch capacity)
FP8 = torch.float8_e4m3fnuz
COS_TOL = 5e-3
BENCH_TOKENS = [2048, 4096, 8192, 16384, 32768]  # T, the total over the ranks
# tokens per rank of the correctness test: uniform, an uneven batch (idle ranks, a single
# token) and the full capacity
CHECK_TOKENS = [[256] * WORLD, [0, 1, 37, 300, 5, 128, 999, 0], [MAX_TOKENS] * WORLD]
SPIN_US = 3000
STAGES = ("quant", "prep", "plan", "mega", "finish")


def _skip_reason():
    if mori is None:
        return "needs mori"
    if torch.cuda.device_count() < WORLD:
        return f"needs {WORLD} GPUs"
    if "gfx942" not in torch.cuda.get_device_properties(0).gcnArchName:
        return "a8w4 kernels need gfx942 (fp8 FNUZ MFMA)"
    return None


def split_tokens(total):
    """Tokens per rank of a ``total``-token batch (DP attention: even shares)."""
    return [total // WORLD + (r < total % WORLD) for r in range(WORLD)]


def _gather_rows(t, ns):
    """Every rank's rows of ``t`` ([ns[rank], ...]), concatenated in rank order."""
    buf = t.new_zeros((max(max(ns), 1), *t.shape[1:]))
    buf[: t.shape[0]] = t
    parts = [torch.empty_like(buf) for _ in range(WORLD)]
    dist.all_gather(parts, buf)
    return torch.cat([p[:n] for p, n in zip(parts, ns)])


def _check(mega, raw, ns, x, tw, ti, act, situ):
    """One megamoe call vs the fp32 reference -> [(rows needed, cos diff)] of every rank."""
    rank, n = mega.rank, ns[mega.rank]
    y = mega(x, ti, tw)[:n].float()
    x_ref = (mega.x8[:n].view(FP8).float().view(n, H // 128, 128)
             * mega.xs[:n].unsqueeze(-1)).view(n, H)
    X, TW, TI = (_gather_rows(t, ns) for t in (x_ref, tw, ti))
    ref = R.ref_moe(X, TW, TI, raw, rank * E_LOCAL, act, situ, quant_x=False)
    dist.all_reduce(ref)
    off = sum(ns[:rank])
    diff = R.cos_diff(ref[off:off + n], y) if n else 0.0
    res = [None] * WORLD
    dist.all_gather_object(res, (mega.rows_needed(), diff))
    bad = [(r, *v) for r, v in enumerate(res) if v[0] > mega.s_cap or not v[1] < COS_TOL]
    assert not bad, f"tokens per rank {ns}: (rank, rows needed, cos diff) {bad}"
    return res


def _align():
    """Device-side barrier of all ranks (MORI shmem) on the current stream."""
    mori.shmem.shmem_barrier_on_stream(torch.cuda.current_stream())


def _spin_cycles(us):
    torch.cuda._sleep(1000)
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    torch.cuda._sleep(1 << 22)
    e.record()
    e.synchronize()
    return int((1 << 22) * us / (s.elapsed_time(e) * 1e3))


def _timed(fn, spin, marks=None):
    """One timed call -> (e2e us, {stage: us} if ``marks``)."""
    R.flush_cache()
    dist.barrier()
    torch.cuda._sleep(spin)
    _align()
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


def _worker(rank, port, points, iters, warmup, acts):
    os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"] = "localhost", str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="cpu:gloo,cuda:nccl", rank=rank, world_size=WORLD,
                            device_id=torch.device("cuda", rank))
    torch._C._distributed_c10d._register_process_group("mori", dist.group.WORLD)
    mori.shmem.shmem_torch_process_group_init("mori")
    from a8w4.megamoe import MegaMoeEpA8W4

    W, raw = R.make_weights(E_LOCAL, H, I, seed=rank)
    max_tokens = max(max(ns) for ns in points)
    megas = {act: MegaMoeEpA8W4(W, rank=rank, max_tokens=max_tokens, topk=TOPK, act=act,
                                situ=R.SITU if act == "situv2" else None) for act in acts}
    data = [R.make_routing(ns[rank], H, E_GLOBAL, TOPK, seed=7919 * rank + ns[rank])
            for ns in points]
    # correctness: every point twice, interleaved
    for rep in range(2):
        for ns, (x, tw, ti) in zip(points, data):
            for act, mega in megas.items():
                res = _check(mega, raw, ns, x, tw, ti, act, R.SITU if act == "situv2" else None)
                if rank == 0:
                    print(f"{sum(ns)} tokens {ns} ({act}, call {rep + 1}): rows needed "
                          f"{[r for r, _ in res]} (cap {mega.s_cap}), cos diff vs reference <= "
                          f"{max(d for _, d in res):.1e}", flush=True)
    if iters:
        _bench(rank, points, data, megas, W, iters, warmup)
    torch.cuda.synchronize()
    dist.barrier()
    dist.destroy_process_group()


def _bench(rank, points, data, megas, W, iters, warmup):
    """e2e of megamoe (+ its stages) and of the MORI path, alternating per iteration."""
    spin = _spin_cycles(SPIN_US)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import test_a8w4_mori_ep8 as M

    layer = M.EpMoeLayer(rank)  # also provides the device barrier (_align)
    layer.W = W  # the same experts as megamoe
    rows = []
    for ns, (x, tw, ti) in zip(points, data):
        for act, mega in megas.items():
            marks = []
            run_mega = lambda: mega(x, ti, tw, marks=marks)  # noqa: E731
            if hasattr(layer, "set_act"):
                layer.set_act(act)
            xq, xs = layer.quant(x)
            recv = layer.dispatch(xq, xs, tw, ti)
            count = int(recv[4].item())
            layer.combine(layer.moe(recv, count, max(1, count)), ti)
            run_mori = lambda: layer.combine(  # noqa: E731
                layer.moe(layer.dispatch(*layer.quant(x), tw, ti), count, max(1, count)), ti)
            for _ in range(warmup):
                run_mega()
                marks.clear()
                run_mori()
            torch.cuda.synchronize()
            t_mega, t_mori, st = [], [], []
            for _ in range(iters):
                t, s = _timed(run_mega, spin, marks)
                t_mega.append(t)
                st.append(s)
                t_mori.append(_timed(run_mori, spin)[0])
            res = [None] * WORLD
            dist.all_gather_object(res, (t_mega, t_mori, st, mega.rows_needed()))
            if rank == 0:
                med = statistics.median
                e2e = med([max(r[0][i] for r in res) for i in range(iters)])
                mori_e2e = med([max(r[1][i] for r in res) for i in range(iters)])
                stages = {s: max(med([it[s] for it in r[2]]) for r in res) for s in STAGES}
                rows.append((sum(ns), act, e2e, mori_e2e, stages, max(r[3] for r in res)))
                print(f"T {sum(ns)} {act}: megamoe {e2e:.1f} us, MORI path {mori_e2e:.1f} us",
                      flush=True)
    if rank == 0:
        lines = [
            "",
            f"EP8 MoE layer (H {H}, I {I}, {E_GLOBAL} experts, {E_LOCAL} per rank, top-{TOPK}), "
            f"us, slowest rank, median of {iters} iterations. MORI path: quant -> MORI dispatch "
            "-> a8w4_moe_dyn (received rows) -> MORI combine; megamoe stages: slowest rank's "
            "median each.",
            "",
            "| tokens | tokens/rank | act | MORI path | megamoe | speedup | quant | prep | plan "
            "| mega | finish | rows/rank max |",
            "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for T, act, e2e, mori_e2e, st, rmax in rows:
            lines.append(
                f"| {T} | {T // WORLD} | {act} | {mori_e2e:.1f} | {e2e:.1f} | {mori_e2e / e2e:.2f}x | "
                + " | ".join(f"{st[s]:.1f}" for s in STAGES) + f" | {rmax} |")
        print("\n".join(lines), flush=True)


def _free_port():
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def run(points, iters=0, warmup=3, acts=("silu",)):
    """Spawn the 8 ranks: check every point (tokens per rank, one list per point) against
    the reference; ``iters`` > 0 also times it against the MORI path (rank 0 prints)."""
    torch.multiprocessing.spawn(_worker, args=(_free_port(), points, iters, warmup, acts),
                                nprocs=WORLD, join=True)


def test_megamoe_ep8():
    reason = _skip_reason()
    if reason:
        pytest.skip(reason)
    run(CHECK_TOKENS)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-t", "--tokens", type=int, nargs="+", default=BENCH_TOKENS,
                    help="tokens of the batch in total, split evenly over the 8 ranks, each "
                         f"<= {WORLD * MAX_TOKENS} (default: %(default)s)")
    ap.add_argument("--act", nargs="+", default=["silu"], choices=R.ACTS)
    ap.add_argument("--iters", type=int, default=20, help="timed iterations per point (0: check only)")
    ap.add_argument("--warmup", type=int, default=3)
    args = ap.parse_args()
    assert all(0 < n <= WORLD * MAX_TOKENS for n in args.tokens), args.tokens
    reason = _skip_reason()
    if reason:
        sys.exit(reason)
    run([split_tokens(n) for n in args.tokens], iters=args.iters, warmup=args.warmup,
        acts=tuple(args.act))
