"""Trace one EP mega-kernel call on 8 ranks and summarize where the time goes.

    torchrun --nproc_per_node=8 tests/contrib/moe/ep8/trace_ep8_mega.py [--producers 32] [--lag 2] [--one-queue] [--save trace.pt]

Per rank: kernel span, producer time, GateUp / Down task counts (and how many ran on another XCD than their queue's)
with mean wait and run time, when the last ones end, and how busy the CUs are (task time over span x CUs used).
"""

import argparse
import os
import statistics
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(__file__))
import bench_ep8_mori_flydsl as bench  # noqa: E402

TICK_US = 0.01  # s_memrealtime runs at 100 MHz
KINDS = {1: "P", 2: "G", 3: "D"}


def summarize(trace, queue_cap):
    """queue_cap: trace slots per queue (slot = queue * queue_cap + ticket), 0 for one queue."""
    rows = [(i, r) for i, r in enumerate(trace.view(-1, 8).tolist()) if r[3] & 0xFF in KINDS]
    t0 = min(r[4] for _, r in rows)
    rel = lambda t: ((t - t0) & 0xFFFFFFFF) * TICK_US
    span = max(rel(r[2]) for _, r in rows)
    out = dict(span=span)
    for kind in "PGD":
        sel = [(i, r) for i, r in rows if KINDS[r[3] & 0xFF] == kind]
        out[kind + "_n"] = len(sel)
        out[kind + "_wait"] = statistics.mean(rel(r[1]) - rel(r[0]) for _, r in sel) if sel else 0.0
        out[kind + "_run"] = statistics.mean(rel(r[2]) - rel(r[1]) for _, r in sel) if sel else 0.0
        # GateUp / Down: shader clocks of the run in word 5, so the mean clock it ran at (power-limited).
        out[kind + "_mhz"] = statistics.mean(
            (r[5] & 0xFFFFFFFF) / max(rel(r[2]) - rel(r[1]), TICK_US) for _, r in sel
        ) if sel and kind != "P" else 0.0
        out[kind + "_last_end"] = max((rel(r[2]) for _, r in sel), default=0.0)
        out[kind + "_first_start"] = min((rel(r[1]) for _, r in sel), default=0.0)
        # Tasks that ran on another XCD than their queue's.
        out[kind + "_stolen"] = sum(queue_cap and (r[3] >> 24) != i // queue_cap for i, r in sel)
    # XCC id and HW_ID bits [8, 16): CU, SH and SE.
    cus = {r[3] & ~0xFF for _, r in rows}
    out["cus"] = len(cus)
    out["busy"] = sum(rel(r[2]) - rel(r[0]) for _, r in rows) / (span * len(cus))
    # Ticket time (entry -> start) and the median gap before it (previous task's end -> entry) on the same CU.
    out["ticket"] = statistics.mean(rel(r[0]) - rel(r[4]) for _, r in rows)
    per_cu = {}
    for _, r in rows:
        per_cu.setdefault(r[3] & ~0xFF, []).append((rel(r[4]), rel(r[2])))
    gaps = sorted(b[0] - a[1] for spans in per_cu.values() for a, b in zip(sorted(spans), sorted(spans)[1:]))
    out["gap"] = gaps[len(gaps) // 2] if gaps else 0.0
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--producers", type=int, default=16)
    parser.add_argument("--lag", type=int, default=2)
    parser.add_argument("--one-queue", action="store_true")
    parser.add_argument("--strict-sync", action="store_true")
    parser.add_argument("--relaunch", action="store_true")
    parser.add_argument("--queue-order", default="paired", choices=["interleaved", "contiguous", "paired"])
    parser.add_argument("--tasks-per-wg", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--routing", default="random", choices=["random", "balanced"])
    parser.add_argument("--combine", default="bf16", choices=["bf16", "fp8"])
    parser.add_argument("--save", default="")
    args = parser.parse_args()
    rank, world = bench.setup()
    from pyhip.contrib.flydsl import ep_moe_gfx942
    from pyhip.contrib.flydsl.ep_moe_mega_gfx942 import trace_entries

    if rank == 0:
        ep_moe_gfx942.build_kernels()
    dist.barrier()
    ep_moe_gfx942.build_kernels()
    layer = bench.make_layer(rank, world, args.tokens, args.seed, args.routing)
    mega = bench.Mega(rank, world, layer, args.tokens, "mega", args.producers, args.lag, not args.one_queue, not args.strict_sync,
                      not args.relaunch, args.queue_order, args.tasks_per_wg, args.combine)
    op = mega.op
    entries = trace_entries(op, args.producers)
    op.trace = torch.zeros(entries, dtype=torch.int32, device="cuda")
    for _ in range(4):
        op.trace.zero_()
        mega.run(layer["x"], layer["ids"], layer["weights"])
    torch.cuda.synchronize()
    stats = summarize(op.trace.cpu(), 0 if args.one_queue else entries // 8 // 8)
    stats["blocks"] = (int(op.num_valid.item()) + 255) // 256
    # Kernel start after the count exchange (ep_prep), comparable across ranks.
    rows = [r for r in op.trace.view(-1, 8).tolist() if r[3] & 0xFF]
    stats["start_lag"] = ((min(r[4] & 0xFFFFFFFF for r in rows) - int(op.status[2])) & 0xFFFFFFFF) * TICK_US
    st = op.status.tolist()
    # ep_prep entry and exit relative to the end of the count exchange (signed, us).
    stats["prep"] = [round((((st[i] - st[2]) + (1 << 31)) % (1 << 32) - (1 << 31)) * TICK_US) for i in (3, 4)]
    gathered = [None] * world
    dist.all_gather_object(gathered, stats)
    if args.save:
        traces = [None] * world
        dist.all_gather_object(traces, op.trace.cpu())
        if rank == 0:
            torch.save(dict(args=vars(args), traces=traces, stats=gathered), args.save)
    if rank == 0:
        print(f"producers={args.producers} lag={args.lag} queues={1 if args.one_queue else 8} "
              f"sync={'strict' if args.strict_sync else 'relaxed'} {'relaunch' if args.relaunch else 'persistent'} "
              f"{args.queue_order} (us from the first task)")
        print("rank blocks  span | P_run P_end | G_n stol G_wait G_run G_end | D_n stol D_first D_wait D_run D_end | CUs busy "
              "| ticket gap | MHz G/D | start")
        for r, s in enumerate(gathered):
            print(f"{r:4d} {s['blocks']:6d} {s['span']:5.0f} | {s['P_run']:5.0f} {s['P_last_end']:5.0f} | "
                  f"{s['G_n']:4d} {s['G_stolen']:4d} {s['G_wait']:6.1f} {s['G_run']:5.0f} {s['G_last_end']:5.0f} | "
                  f"{s['D_n']:4d} {s['D_stolen']:4d} {s['D_first_start']:7.0f} {s['D_wait']:6.1f} {s['D_run']:5.0f} "
                  f"{s['D_last_end']:5.0f} | {s['cus']:4d} {s['busy']:.2f} | {s['ticket']:6.1f} {s['gap']:5.1f} | "
                  f"{s['G_mhz']:4.0f} {s['D_mhz']:4.0f} | {s['start_lag']:5.0f} {s['prep']}")
    dist.barrier()
    bench.mori.shmem.shmem_finalize()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
