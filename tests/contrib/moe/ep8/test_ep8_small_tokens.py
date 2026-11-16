"""EP8 mega MoE with few tokens per rank: every token against the MORI baseline and the step1 path, and latency.

    torchrun --nproc_per_node=8 tests/contrib/moe/ep8/test_ep8_small_tokens.py [--capacity 16384]

The mega and step1 EPMoEGfx942 (built for --capacity tokens per rank, like sglang builds them for its MORI capacity) and
the MORI op run the cases in order, each on the state the previous one left: the same count on every rank, then uneven
counts with idle (0-token) ranks. Random routing, BF16 combine.
"""

import argparse
import os
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(__file__))
import bench_ep8_mori_flydsl as bench  # noqa: E402

UNIFORM = [256, 1, 2, 4, 8, 16, 32, 64, 128]
UNEVEN = [[0, 1, 2, 3, 5, 8, 13, 21], [0, 0, 0, 0, 64, 64, 64, 64], [1, 0, 0, 0, 0, 0, 0, 0],
          [255, 0, 17, 1, 0, 100, 3, 64]]


def quant(x):
    if x.shape[0] == 0:
        return (torch.empty((0, bench.H), dtype=bench.dtypes.fp8, device=x.device),
                torch.empty((0, bench.H // 128), dtype=torch.float32, device=x.device))
    return bench.get_hip_quant(bench.QuantType.per_1x128)(x, quant_dtype=bench.dtypes.fp8)


def mori_forward(op, layer, a, scale, ids, w):
    state, phases = bench.Baseline(op, dict(layer, ids=ids, weights=w)).phases()
    state["a"], state["scale"] = a, scale
    for _, fn in phases[1:]:
        fn()
    return state["out"][: a.shape[0]]


def worst_row_diff(x, y):
    """Largest calc_diff of a single token row (0 for no rows)."""
    if x.shape[0] == 0:
        return 0.0
    x, y = x.double(), y.double()
    den = (x * x + y * y).sum(dim=1)
    sim = torch.where(den > 0, 2 * (x * y).sum(dim=1) / den, torch.ones_like(den))
    return float((1 - sim).max())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--capacity", type=int, default=16384, help="tokens per rank the EPMoEGfx942 are built for")
    parser.add_argument("--mori-capacity", type=int, default=4096,
                        help="the MORI op's (its MoE's buffers exceed 2 GB above 4096)")
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--tol", type=float, default=1e-3, help="largest calc_diff of a token row")
    args = parser.parse_args()

    # Two EPMoEGfx942 and the MORI op for 16384 tokens/rank outgrow MORI's default symmetric heap.
    os.environ.setdefault("MORI_SHMEM_HEAP_SIZE", "16GiB")
    rank, world = bench.setup()
    from pyhip.contrib.flydsl import ep_moe_gfx942

    if rank == 0:
        ep_moe_gfx942.build_kernels()
    dist.barrier()
    ep_moe_gfx942.build_kernels()
    layer = bench.make_layer(rank, world, 32, 0)
    mega = bench.Mega(rank, world, layer, args.capacity)
    step1 = bench.Mega(rank, world, layer, args.capacity, mode="step1")
    mori_op = bench.make_mori_op(rank, world, args.mori_capacity, 80, 16)

    results = []
    for case, counts in enumerate([[t] * world for t in UNIFORM] + UNEVEN):
        x, ids, w = bench.make_routes(rank, counts[rank], 100 + case)
        a, scale = quant(x)
        out = mega.op.forward(a, scale, ids, w).clone()
        rows = mega.op.rows_needed()
        ref_step1 = step1.op.forward(a, scale, ids, w).clone()
        ref_mori = mori_forward(mori_op, layer, a, scale, ids, w).clone()
        r = dict(rows=rows, s_cap=mega.op.s_cap, vs_mori=worst_row_diff(out, ref_mori),
                 vs_step1=worst_row_diff(out, ref_step1),
                 max_abs=float((out.float() - ref_mori.float()).abs().max()) if counts[rank] else 0.0)
        if len(set(counts)) == 1:
            L = dict(layer, x=x, ids=ids, weights=w)
            mega.L = L
            r["mega_us"] = bench.run_timed(mega.phases()[1], args.iters, args.warmup)[0]["total"]
            r["mori_us"] = bench.run_timed(bench.Baseline(mori_op, L).phases()[1], args.iters, args.warmup)[0]["total"]
        gathered = [None] * world
        dist.all_gather_object(gathered, r)
        results.append((counts, gathered))

    failed = False
    if rank == 0:
        print(f"capacity {args.capacity} tokens/rank, s_cap {mega.op.s_cap} rows; worst token row over all ranks")
        print("tokens per rank                  max rows   vs MORI   vs step1   max |diff|   mega us   MORI us")
        for counts, g in results:
            worst = max(max(r["vs_mori"], r["vs_step1"]) for r in g)
            failed |= worst > args.tol or any(r["rows"] > r["s_cap"] for r in g)
            timing = ""
            if "mega_us" in g[0]:
                timing = f"{max(r['mega_us'] for r in g):9.1f} {max(r['mori_us'] for r in g):9.1f}"
            label = str(counts[0]) if len(set(counts)) == 1 else " ".join(map(str, counts))
            print(f"{label:32s} {max(r['rows'] for r in g):8d}  {max(r['vs_mori'] for r in g):8.2e}  "
                  f"{max(r['vs_step1'] for r in g):8.2e}  {max(r['max_abs'] for r in g):10.3e}  {timing}")
        print("FAILED" if failed else "OK")
    dist.barrier()
    bench.mori.shmem.shmem_finalize()
    dist.destroy_process_group()
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
