"""EP8 baseline of one MiMo-V2.6-Flash MoE layer: per-1x128 FP8 quant -> MORI dispatch -> aiter.fused_moe
(AITER_FMOE_FP8_BLOCKSCALE_FLYDSL=1 hook) -> MORI BF16 combine, configured like sglang moriep.py (IntraNode,
80 blocks x 16 warps, FP8 dispatch with FP32 1x128 scales, combine without weights).

    torchrun --nproc_per_node=8 tests/contrib/moe/ep8/bench_ep8_mori_flydsl.py --tokens 4096

Each rank owns 32 of the 256 experts and sends `--tokens` tokens routed to random top-8 experts.
"""

import argparse
import dataclasses
import json
import os
import statistics
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("AITER_FMOE_FP8_BLOCKSCALE_FLYDSL", "1")

import aiter.fused_moe as fm  # noqa: E402
import mori  # noqa: E402
from aiter import ActivationType, QuantType, dtypes, get_hip_quant  # noqa: E402
from aiter.ops.shuffle import shuffle_weight  # noqa: E402
from test_fmoe_a8w8_blockscale_asm_gfx942 import block_quant  # noqa: E402

H, I, EXPERTS, TOPK = 4096, 2048, 256, 8


def setup():
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dist.init_process_group("cpu:gloo,cuda:nccl", device_id=torch.device("cuda", local))
    import torch._C._distributed_c10d as c10d

    c10d._register_process_group("default", dist.group.WORLD)
    mori.shmem.shmem_torch_process_group_init("default")
    return rank, world


def make_mori_op(rank, world, capacity, block_num, warps):
    kwargs = dict(
        data_type=dtypes.fp8, rank=rank, world_size=world, hidden_dim=H, scale_dim=H // 128, scale_type_size=4,
        max_token_type_size=2, max_num_inp_token_per_rank=capacity, num_experts_per_rank=EXPERTS // world,
        num_experts_per_token=TOPK, warp_num_per_block=warps, block_num=block_num, max_total_recv_tokens=0,
        kernel_type=mori.ops.EpDispatchCombineKernelType.IntraNode, gpu_per_node=8, rdma_block_num=0,
        num_qp_per_pe=2, quant_type="none",
    )
    valid = {f.name for f in dataclasses.fields(mori.ops.EpDispatchCombineConfig)}
    return mori.ops.EpDispatchCombineOp(mori.ops.EpDispatchCombineConfig(**{k: v for k, v in kwargs.items() if k in valid}))


def make_routes(rank, tokens, seed):
    """This rank's tokens and their top-8 routes."""
    g = torch.Generator().manual_seed(seed + rank)
    x = (torch.randn((tokens, H), generator=g) * 0.5).to(torch.bfloat16).cuda()
    ids = torch.rand((tokens, EXPERTS), generator=g).topk(TOPK, dim=1).indices.to(torch.int32).cuda()
    w = torch.rand((tokens, TOPK), generator=g) + 0.1
    return x, ids, (w / w.sum(dim=1, keepdim=True)).cuda()


def make_layer(rank, world, tokens, seed):
    """This rank's tokens, routes and local expert weights (the same global weights on every run)."""
    x, ids, w = make_routes(rank, tokens, seed)
    torch.manual_seed(seed + 1000 + rank)
    e = EXPERTS // world
    w1, w1_scale = block_quant(e, 2 * I, H, 0.2)
    w2, w2_scale = block_quant(e, H, I, 0.02)
    mask = torch.zeros(EXPERTS, dtype=torch.int32, device="cuda")
    mask[rank * e:(rank + 1) * e] = 1
    return dict(x=x, ids=ids, weights=w, w1_raw=w1, w2_raw=w2, w1=shuffle_weight(w1, layout=(16, 16)),
                w2=shuffle_weight(w2, layout=(16, 16)), w1_scale=w1_scale, w2_scale=w2_scale, mask=mask)


class Baseline:
    def __init__(self, op, layer):
        self.op, self.L = op, layer
        self.quant = get_hip_quant(QuantType.per_1x128)

    def phases(self):
        L = self.L
        state = {}

        def quant():
            state["a"], state["scale"] = self.quant(L["x"], quant_dtype=dtypes.fp8)

        def dispatch():
            out = self.op.dispatch(state["a"], L["weights"], state["scale"], L["ids"])
            state["recv_x"], state["recv_w"], state["recv_scale"], state["recv_ids"], state["recv_n"] = out

        def moe():
            state["moe_out"] = fm.fused_moe(
                state["recv_x"], L["w1"], L["w2"], state["recv_w"], state["recv_ids"], expert_mask=L["mask"],
                activation=ActivationType.Silu, quant_type=QuantType.per_128x128, w1_scale=L["w1_scale"],
                w2_scale=L["w2_scale"], a1_scale=state["recv_scale"], dtype=dtypes.bf16,
                num_local_tokens=state["recv_n"], ep_has_fake_route=False,
            )

        def combine():
            state["out"] = self.op.combine(state["moe_out"], None, L["ids"])[0]

        return state, [("quant", quant), ("dispatch", dispatch), ("moe", moe), ("combine", combine)]


def run_timed(phases, iters, warmup):
    for _ in range(warmup):
        for _, fn in phases:
            fn()
    torch.cuda.synchronize()
    dist.barrier()
    ev = [[torch.cuda.Event(enable_timing=True) for _ in range(len(phases) + 1)] for _ in range(iters)]
    for it in range(iters):
        ev[it][0].record()
        for p, (_, fn) in enumerate(phases):
            fn()
            ev[it][p + 1].record()
    torch.cuda.synchronize()
    res = {name: [ev[it][p].elapsed_time(ev[it][p + 1]) * 1e3 for it in range(iters)] for p, (name, _) in enumerate(phases)}
    res["total"] = [ev[it][0].elapsed_time(ev[it][-1]) * 1e3 for it in range(iters)]
    loop = ev[0][0].elapsed_time(ev[-1][-1]) * 1e3 / iters
    if os.environ.get("EP8_TIMES") and dist.get_rank() == 0:
        gaps = [ev[it][-1].elapsed_time(ev[it + 1][0]) * 1e3 for it in range(iters - 1)]
        print("per-iteration total us:", " ".join(f"{t:.0f}" for t in res["total"]))
        print("gaps between iterations us:", " ".join(f"{t:.0f}" for t in gaps))
    return {k: statistics.median(v) for k, v in res.items()}, loop


class Mega:
    """pyhip EPMoEGfx942: P2P dispatch + GateUp + Down with the combine fused into its epilogue."""

    def __init__(self, rank, world, layer, tokens, mode="mega", producers=16, lag=2, xcd_queues=True, relaxed=True,
                 persistent=True, queue_order="paired", tasks_per_wg=1):
        from pyhip.contrib.flydsl.ep_moe_gfx942 import EPMoEGfx942

        self.L = layer
        self.op = EPMoEGfx942(rank, world, layer["w1"], layer["w1_scale"], layer["w2"], layer["w2_scale"], TOPK, tokens,
                              mode=mode, producers=producers, lag=lag, xcd_queues=xcd_queues, relaxed=relaxed,
                              persistent=persistent, queue_order=queue_order, tasks_per_wg=tasks_per_wg)
        self.quant = get_hip_quant(QuantType.per_1x128)

    def run(self, x, ids, weights):
        a, scale = self.quant(x, quant_dtype=dtypes.fp8)
        return self.op.forward(a, scale, ids, weights)

    def phases(self):
        L = self.L
        state = {}

        def quant():
            state["a"], state["scale"] = self.quant(L["x"], quant_dtype=dtypes.fp8)

        def mega():
            state["out"] = self.op.forward(state["a"], state["scale"], L["ids"], L["weights"])

        return state, [("quant", quant), ("mega", mega)]


def bench(impl, rank, world, args):
    state, phases = impl.phases()
    recv_n = local_routes = -1
    baseline = isinstance(impl, Baseline)
    if baseline:
        real_check = fm._flydsl_gfx942_fp8_blockscale_unsupported
        reasons = []
        fm._flydsl_gfx942_fp8_blockscale_unsupported = lambda *a, **k: reasons.append(real_check(*a, **k)) or reasons[-1]
    for name, fn in phases:
        fn()
        if name == "dispatch":
            recv_n = int(state["recv_n"].item())
            local_routes = int(((state["recv_ids"][:recv_n] // (EXPERTS // world)) == rank).sum().item())
    torch.cuda.synchronize()
    if baseline:
        fm._flydsl_gfx942_fp8_blockscale_unsupported = real_check
        if rank == 0:
            print(f"flydsl hook rejected: {reasons!r}")
    out = state["out"].clone()
    med, loop = run_timed(phases, args.iters, args.warmup)
    if isinstance(impl, Mega):
        stages = {}
        for _ in range(5):
            impl.op.stage_events = []
            for _, fn in phases:
                fn()
            torch.cuda.synchronize()
            ev = impl.op.stage_events
            for (_, a), (name, b) in zip(ev[:-1], ev[1:]):
                stages.setdefault(name, []).append(a.elapsed_time(b) * 1e3)
        impl.op.stage_events = None
        med.update({f"[{k}]": statistics.median(v) for k, v in stages.items()})
    # Repeated calls must reproduce the first output (stale data / missing synchronization shows up here).
    repeat = float((state["out"].float() - out.float()).abs().max())
    row = dict(rank=rank, recv_tokens=recv_n, local_routes=local_routes, loop_us=loop, repeat_maxdiff=repeat,
               **{f"{k}_us": v for k, v in med.items()})
    rows = [None] * world
    dist.all_gather_object(rows, row)
    if rank == 0:
        keys = [f"{name}_us" for name, _ in phases] + ["total_us", "loop_us"] + [k for k in rows[0] if k.startswith("[")]
        print(f"[{type(impl).__name__}] tokens/rank={args.tokens}" + (f" mori capacity={args.capacity}" if baseline else ""))
        print("rank recv  routes  " + "  ".join(f"{k.removesuffix('_us'):>9s}" for k in keys) + "  repeat_maxdiff")
        for r in rows:
            print(f"{r['rank']:4d} {r['recv_tokens']:5d} {r['local_routes']:6d}  " + "  ".join(f"{r[k]:9.1f}" for k in keys)
                  + f"  {r['repeat_maxdiff']:.3g}")
        print("max            " + "  ".join(f"{max(r[k] for r in rows):9.1f}" for k in keys))
    return rows, out


def check_alternating(mega, rank, world, args, layer):
    """mega on inputs A, B, A, B, ... must match the step1 path on each (stale data between calls shows up here)."""
    import pyhip

    ref_op = Mega(rank, world, layer, args.tokens, mode="step1")
    inputs = [(layer["x"], layer["ids"], layer["weights"]), make_routes(rank, args.tokens, args.seed + 77)]
    refs = [ref_op.run(*inp).clone() for inp in inputs]
    worst = 0.0
    for it in range(6):
        out = mega.run(*inputs[it % 2]).float()
        ref = refs[it % 2].float()
        worst = max(worst, float(pyhip.calc_diff(out, ref)))
    torch.cuda.synchronize()
    worsts = [None] * world
    dist.all_gather_object(worsts, worst)
    if rank == 0:
        print("alternating inputs, worst calc_diff vs step1 per rank: " + " ".join(f"{w:.2e}" for w in worsts))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=4096, help="tokens sent by each rank")
    parser.add_argument("--capacity", type=int, default=4096, help="MORI max_num_inp_token_per_rank")
    parser.add_argument("--block-num", type=int, default=80)
    parser.add_argument("--warps", type=int, default=16)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", default="both", choices=["baseline", "mega", "both"])
    parser.add_argument("--ep-mode", default="mega", choices=["mega", "step1"], help="EPMoEGfx942 path")
    parser.add_argument("--producers", type=int, default=16)
    parser.add_argument("--lag", type=int, default=2, help="Down lags GateUp by this many units of a queue")
    parser.add_argument("--one-queue", action="store_true", help="one task queue instead of one per XCD")
    parser.add_argument("--strict-sync", action="store_true", help="scoped L2 write-back / invalidation per task")
    parser.add_argument("--relaunch", action="store_true", help="one task per workgroup instead of persistent ones")
    parser.add_argument("--queue-order", default="paired", choices=["interleaved", "contiguous", "paired"])
    parser.add_argument("--tasks-per-wg", type=int, default=1)
    parser.add_argument("--check-alt", action="store_true", help="compare against step1 on alternating inputs")
    parser.add_argument("--json", default="")
    args = parser.parse_args()

    rank, world = setup()
    if args.mode != "baseline":
        from pyhip.contrib.flydsl import ep_moe_gfx942

        if rank == 0:
            ep_moe_gfx942.build_kernels()
        dist.barrier()
        ep_moe_gfx942.build_kernels()
    layer = make_layer(rank, world, args.tokens, args.seed)
    report, outputs = {}, {}
    if args.mode in ("baseline", "both"):
        op = make_mori_op(rank, world, args.capacity, args.block_num, args.warps)
        report["baseline"], outputs["baseline"] = bench(Baseline(op, layer), rank, world, args)
    if args.mode in ("mega", "both"):
        mega = Mega(rank, world, layer, args.tokens, args.ep_mode, args.producers, args.lag, not args.one_queue, not args.strict_sync,
                    not args.relaunch, args.queue_order, args.tasks_per_wg)
        if args.check_alt:
            check_alternating(mega, rank, world, args, layer)
        report["mega"], outputs["mega"] = bench(mega, rank, world, args)
        rows = mega.op.rows_needed()
        assert rows <= mega.op.s_cap, f"rank {rank} needs {rows} rows > s_cap {mega.op.s_cap}"
    if len(outputs) == 2:
        import pyhip

        ref, out = outputs["baseline"].float(), outputs["mega"].float()
        diff = dict(calc_diff=float(pyhip.calc_diff(out, ref)), max_abs=float((out - ref).abs().max()),
                    ref_absmax=float(ref.abs().max()))
        diffs = [None] * world
        dist.all_gather_object(diffs, diff)
        if rank == 0:
            for r, d in enumerate(diffs):
                print(f"rank {r}: mega vs baseline calc_diff {d['calc_diff']:.2e} max_abs {d['max_abs']:.3e} "
                      f"(ref absmax {d['ref_absmax']:.2f})")
            report["diff"] = diffs
    if rank == 0 and args.json:
        with open(args.json, "w") as fp:
            json.dump(dict(args=vars(args), **report), fp, indent=1)
    dist.barrier()
    mori.shmem.shmem_finalize()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
