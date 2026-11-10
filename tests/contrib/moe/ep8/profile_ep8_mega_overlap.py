"""How much of the EP8 mega kernel's communication is hidden under its compute.

    torchrun --nproc_per_node=8 tests/contrib/moe/ep8/profile_ep8_mega_overlap.py [--iters 100] [--rounds 4]

Times the layer with the dispatch data and/or the combine atomics sent to local scratch buffers instead of the peers
(timing only: the output is then wrong). Arrival signals stay remote, so the schedule and the synchronization are
unchanged; what disappears is the xGMI traffic. p2p minus local is the communication time left exposed.
Per mode, from its fastest round (other jobs on the GPUs only slow a round down): the slowest rank's median layer
time, and from one traced call right after (same power state) the kernel span, producer time, mean GateUp / Down
run time and wait, and the effective clock.
"""

import argparse
import os
import statistics
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(__file__))
import bench_ep8_mori_flydsl as bench  # noqa: E402
import trace_ep8_mega  # noqa: E402

DISPATCH = ("recv_a", "recv_scale", "down_ids", "recv_w")
MODES = {"p2p": (), "local-dispatch": DISPATCH, "local-combine": ("out",), "local": DISPATCH + ("out",)}


def traced(mega, layer, entries):
    """Trace summary of the last of a few back-to-back calls."""
    op = mega.op
    op.trace = torch.zeros(entries, dtype=torch.int32, device="cuda")
    for _ in range(4):
        op.trace.zero_()
        mega.run(layer["x"], layer["ids"], layer["weights"])
    torch.cuda.synchronize()
    stats = trace_ep8_mega.summarize(op.trace.cpu(), entries // 8 // 8)
    op.trace = None
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rank, world = bench.setup()
    from pyhip.contrib.flydsl import ep_moe_gfx942
    from pyhip.contrib.flydsl.ep_moe_mega_gfx942 import trace_entries

    if rank == 0:
        ep_moe_gfx942.build_kernels()
    dist.barrier()
    ep_moe_gfx942.build_kernels()
    layer = bench.make_layer(rank, world, args.tokens, args.seed)
    mega = bench.Mega(rank, world, layer, args.tokens)
    op = mega.op
    # A real call first: the local modes then compute on the real received rows (zeros would draw less power).
    mega.run(layer["x"], layer["ids"], layer["weights"])
    torch.cuda.synchronize()
    peers = dict(op.tables)
    scratch = {name: torch.zeros_like(getattr(op, name)) for name in DISPATCH + ("out",)}
    local = {name: torch.full_like(peers[name], buf.data_ptr()) for name, buf in scratch.items()}
    _, phases = mega.phases()
    entries = trace_entries(op, mega.op.producers)
    results = {mode: [] for mode in MODES}
    for round_index in range(args.rounds):
        for mode, names in MODES.items():
            op.tables = {**peers, **{name: local[name] for name in names}}
            med, _ = bench.run_timed(phases, args.iters, args.warmup)
            row = dict(total=med["total"], **traced(mega, layer, entries))
            rows = [None] * world
            dist.all_gather_object(rows, row)
            results[mode].append(rows)
            if rank == 0:
                print(f"round {round_index} {mode:15s} layer {max(r['total'] for r in rows):6.0f} us", flush=True)
    op.tables = peers
    if rank == 0:
        print(f"tokens/rank={args.tokens} iters={args.iters} rounds={args.rounds}, fastest round per mode; "
              "layer and span = slowest rank, others = rank mean")
        print("mode            layer_us span_us | P_run P_end | G_wait G_run D_run | busy | MHz G/D")
        summary = {}
        for mode, rounds in results.items():
            rows = min(rounds, key=lambda rows: max(r["total"] for r in rows))
            s = dict(total=max(r["total"] for r in rows), span=max(r["span"] for r in rows))
            s.update({key: statistics.mean(r[key] for r in rows)
                      for key in ("P_run", "P_last_end", "G_wait", "G_run", "D_run", "busy", "G_mhz", "D_mhz")})
            summary[mode] = s
            print(f"{mode:15s} {s['total']:8.0f} {s['span']:7.0f} | {s['P_run']:5.0f} {s['P_last_end']:5.0f} | "
                  f"{s['G_wait']:6.1f} {s['G_run']:5.0f} {s['D_run']:5.0f} | {s['busy']:.2f} | {s['G_mhz']:4.0f} {s['D_mhz']:4.0f}")
        for mode in ("local-dispatch", "local-combine", "local"):
            print(f"exposed by {mode.removeprefix('local-') if mode != 'local' else 'both'}: "
                  f"{summary['p2p']['total'] - summary[mode]['total']:.0f} us (layer), "
                  f"{summary['p2p']['span'] - summary[mode]['span']:.0f} us (span)")
    dist.barrier()


if __name__ == "__main__":
    main()
