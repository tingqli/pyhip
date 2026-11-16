# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Perf and accuracy of pyhip's gfx942 FP8 block-scale MoE, called directly without aiter.

Times pyhip.contrib.flydsl.moe_a8w8_blockscale_gfx942.fused_moe_a8w8_blockscale_gfx942 (route counts, zero rows,
GateUp 2x4_pf1 with FP8 sorted-row output, Down 4x2_pf1 single_store) on one EP rank that owns experts
[0, experts / ep). Its inputs are built in torch in aiter's layouts: shuffle_weight(16, 16) weights and
moe_sorting(block_size=256) routes, with the sort capacity the fused entry uses. The sort itself is not timed.

--mode a2a (default): a MORI all-to-all receive buffer. Of the --tokens tokens of the EP group in one step, those
    with a local route come first and garbage rows (NaN scales) fill the buffer up to --rows rows (default:
    --tokens); num_rows holds the received count. SGLang's MORI buffer has SGLANG_MORI_MOE_MAX_INPUT_TOKENS rows.
--mode ep: all --tokens tokens are this rank's input; those without a local route get zero rows.

Accuracy: calc_diff of the output against an FP32 reference that re-quantizes the activation output to FP8 per
(row, 128 columns) as GateUp does. Rows past num_rows must stay untouched and rows without a local route zero.

Timing: pyhip.cudaPerf around every call: a GPU sleep before its start event hides the host launches, and a sync
follows each call. Calls rotate over clones of the activations, weights and scales so that every call reads cold
data: by default enough clones that >= 1 GiB of other data is read between two uses of one (MI300X MALL: 256 MB).
With several activations, the timed calls alternate between them. Every call starts on an idle GPU: on the MiMo EP8
shape this read 9% more than calls queued back to back at 4096 tokens and 4% less at 32768 (MI300X), so compare
numbers taken with the same method.

    python -m pytest --import-mode=importlib -q test_moe_a8w8_blockscale_gfx942.py       # correctness
    HIP_VISIBLE_DEVICES=0 python test_moe_a8w8_blockscale_gfx942.py      # EP8 a2a, T = 4096 16384 32768, SiLU vs SiTUv2
    HIP_VISIBLE_DEVICES=0 python test_moe_a8w8_blockscale_gfx942.py -t 8192 --rows 32768 --act situv2 --swiglu-limit inf
    HIP_VISIBLE_DEVICES=0 python test_moe_a8w8_blockscale_gfx942.py --mode ep --model-dim 2048 --inter-dim 1024 \
        --experts 64 --ep 4 --topk 4 -t 8192

SiTUv2 needs a pyhip with act / situ support (pip install --no-deps <this repository>); SiLU runs on any pyhip.
"""

import argparse
import inspect
import itertools
import json
import math
import statistics
import sys

import pytest
import torch

import pyhip
from pyhip.contrib.flydsl.moe_a8w8_blockscale_gfx942 import (
    SORT_BLOCK_M,
    fused_moe_a8w8_blockscale_gfx942,
    max_sorted_blocks,
    unsupported_reason,
)

BLK = 128
FP8 = torch.float8_e4m3fnuz
FP8_MAX = torch.finfo(FP8).max
# MiMo-V2.6-Flash: hidden, moe_intermediate, routed experts, top-k.
MODEL = (4096, 2048, 256, 8)
ACTS = ("silu", "situv2")
# The SiTUv2 parameters of pyhip.contrib.flydsl.a8w4.ref.SITU.
SITU = dict(situ_beta=4.0, situ_linear_beta=25.0, swiglu_limit=16.0)
ROUTINGS = ("random", "balanced", "ep_uniform")
CLONED = ("a1", "a1_scale_t", "w1", "w1_scale", "w2", "w2_scale")
COLD_BYTES = 1 << 30
CLONE_LIMIT_BYTES = 32 << 30
HAS_ACT = "act" in inspect.signature(fused_moe_a8w8_blockscale_gfx942).parameters


def is_gfx942():
    return torch.cuda.is_available() and torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] == "gfx942"


def nbytes(tensor):
    return tensor.numel() * tensor.element_size()


def shuffle_weight(weight):
    """aiter.ops.shuffle.shuffle_weight(weight, layout=(16, 16)) of FP8 [..., N, K]."""
    n, k = weight.shape[-2:]
    return weight.view(-1, n // 16, 16, k // 32, 2, 16).permute(0, 1, 3, 4, 2, 5).contiguous().view(weight.shape)


def block_quant(experts, n, k, std):
    """FP8 weights [experts, n, k] with distinct FP32 scales per 128x128 block."""
    scale = torch.empty((experts, n // BLK, k // BLK), device="cuda")
    weight = torch.empty((experts, n, k), device="cuda", dtype=FP8)
    for e in range(experts):
        source = torch.randn((n // BLK, BLK, k // BLK, BLK), device="cuda") * std
        scale[e] = source.abs().amax(dim=(1, 3)).clamp_min(1e-8) / FP8_MAX
        weight[e] = (source / scale[e][:, None, :, None]).to(FP8).view(n, k)
    return weight, scale * (0.5 + torch.rand_like(scale))


def dequant_blocks(weight, scale):
    n, k = weight.shape
    return (weight.float().view(n // BLK, BLK, k // BLK, BLK) * scale[:, None, :, None]).view(n, k)


def make_routes(tokens, topk, experts, local_experts, routing, generator):
    """Global top-k ids [tokens, topk] and weights; this rank owns experts [0, local_experts)."""
    if routing == "ep_uniform":
        # Every token has exactly one local route and every local expert gets tokens / local_experts of them.
        assert experts - local_experts >= topk - 1, "ep_uniform needs topk - 1 remote experts"
        local = torch.empty(tokens, dtype=torch.int64)
        local[torch.randperm(tokens, generator=generator)] = torch.arange(tokens) % local_experts
        remote = torch.rand((tokens, experts - local_experts), generator=generator).topk(topk - 1, dim=1).indices
        is_local = torch.arange(topk) == torch.randint(0, topk, (tokens, 1), generator=generator)
        ids = torch.empty((tokens, topk), dtype=torch.int64)
        ids[is_local] = local
        ids[~is_local] = remote.flatten() + local_experts
    elif routing == "random":
        ids = torch.rand((tokens, experts), generator=generator).topk(topk, dim=1).indices
    elif routing == "balanced":
        choices = torch.arange(tokens * topk).view(tokens, topk) % experts
        ids = torch.randperm(experts, generator=generator)[choices[torch.randperm(tokens, generator=generator)]]
    else:
        raise ValueError(f"routing must be one of {ROUTINGS}, got {routing!r}")
    weights = torch.rand((tokens, topk), generator=generator) + 0.1
    return ids.to(torch.int32).cuda(), (weights / weights.sum(dim=1, keepdim=True)).cuda()


def sort_routes(topk_ids, topk_weights, rows, local_experts):
    """aiter moe_sorting(block_size=256) of the routes of topk_ids (the valid rows) to experts [0, local_experts).

    Per local expert, its (slot << 24) | token IDs in token order, padded to whole blocks with (topk << 24) | valid
    rows, as aiter. The buffers have the capacity fused_moe_a8w8_blockscale_gfx942 uses (max_sorted_blocks of the
    buffer rows); blocks past the valid ones hold garbage the kernels must not read. Also returns the routes
    (token, slot, expert, weight).
    """
    valid, topk = topk_ids.shape
    ids = topk_ids.long()
    token, slot = (ids < local_experts).nonzero(as_tuple=True)
    expert = ids[token, slot]
    order = torch.argsort(expert * rows + token)
    token, slot, expert = token[order], slot[order], expert[order]
    weight = topk_weights[token, slot]
    counts = torch.bincount(expert, minlength=local_experts)
    blocks = (counts + SORT_BLOCK_M - 1) // SORT_BLOCK_M
    first_row = (torch.cumsum(blocks, 0) - blocks) * SORT_BLOCK_M
    first_route = torch.cumsum(counts, 0) - counts
    position = (first_row - first_route)[expert] + torch.arange(token.numel(), device="cuda")
    num_valid = int(blocks.sum()) * SORT_BLOCK_M
    capacity = max_sorted_blocks(rows, topk, local_experts)
    sorted_ids = torch.randint(0, 1 << 31, (capacity * SORT_BLOCK_M,), device="cuda", dtype=torch.int32)
    sorted_ids[:num_valid] = torch.tensor((topk << 24) | valid).to(torch.int32)
    sorted_ids[position] = ((slot << 24) | token).to(torch.int32)
    sorted_weights = torch.full((capacity * SORT_BLOCK_M,), float("nan"), device="cuda")
    sorted_weights[:num_valid] = 0.0
    sorted_weights[position] = weight
    sorted_expert_ids = torch.full((capacity,), 1 << 30, device="cuda", dtype=torch.int32)
    sorted_expert_ids[: num_valid // SORT_BLOCK_M] = torch.repeat_interleave(
        torch.arange(local_experts, device="cuda", dtype=torch.int32), blocks
    )
    sort = dict(
        sorted_ids=sorted_ids,
        sorted_weights=sorted_weights,
        sorted_expert_ids=sorted_expert_ids,
        num_valid_ids=torch.tensor([num_valid, rows], device="cuda", dtype=torch.int32),
    )
    return sort, (token, slot, expert, weight)


def make_case(tokens, mode="a2a", rows=None, hidden_dim=MODEL[0], inter_dim=MODEL[1], experts=MODEL[2],
              topk=MODEL[3], ep=8, routing="random", seed=0):
    """The fused entry's inputs on one EP rank, plus the unshuffled weights for the reference (see the module doc)."""
    if experts % ep or not 0 < topk <= experts:
        raise ValueError(f"need experts % ep == 0 and 0 < topk <= experts, got {experts=} {ep=} {topk=}")
    local = experts // ep
    generator = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    a_source = torch.randn((tokens, hidden_dim // BLK, BLK), device="cuda") * 0.33
    a_scale = a_source.abs().amax(dim=-1).clamp_min(1e-8) / FP8_MAX
    a = (a_source / a_scale[..., None]).to(FP8).view(tokens, hidden_dim)
    del a_source
    a_scale *= 0.5 + torch.rand_like(a_scale)
    w1_raw, w1_scale = block_quant(local, 2 * inter_dim, hidden_dim, 0.2)
    w2_raw, w2_scale = block_quant(local, hidden_dim, inter_dim, 0.02)
    topk_ids, topk_weights = make_routes(tokens, topk, experts, local, routing, generator)
    if mode == "a2a":
        received = (topk_ids < local).any(dim=1).nonzero().squeeze(1)
        valid, rows = received.numel(), rows or tokens
        if rows < valid:
            raise ValueError(f"rows={rows} < {valid} received tokens")
        a1 = torch.randn((rows, hidden_dim), device="cuda").to(FP8)
        a1[:valid] = a[received]
        a1_scale = torch.full((rows, hidden_dim // BLK), float("nan"), device="cuda")
        a1_scale[:valid] = a_scale[received]
        topk_ids, topk_weights = topk_ids[received], topk_weights[received]
        num_rows = torch.tensor([valid], device="cuda", dtype=torch.int32)
    elif mode == "ep":
        if rows not in (None, tokens):
            raise ValueError("rows applies to mode a2a only")
        a1, a1_scale, valid, rows, num_rows = a, a_scale, tokens, tokens, None
    else:
        raise ValueError(f"mode must be a2a or ep, got {mode!r}")
    sort, routes = sort_routes(topk_ids, topk_weights, rows, local)
    return dict(
        a1=a1, a1_scale_t=a1_scale.t().contiguous(), w1=shuffle_weight(w1_raw), w1_scale=w1_scale,
        w2=shuffle_weight(w2_raw), w2_scale=w2_scale, w1_raw=w1_raw, w2_raw=w2_raw, sort=sort, routes=routes,
        num_rows=num_rows, valid=valid, rows=rows, topk=topk, local_experts=local, inter_dim=inter_dim,
    )


def act_ref(gate, up, act, situ):
    """FP32 act(gate, up): SiLU(gate) * up, or SiTUv2 as pyhip.contrib.flydsl.a8w4.ref.act_ref (clamp first)."""
    if act == "silu":
        return torch.nn.functional.silu(gate) * up
    beta, linear_beta, limit = situ["situ_beta"], situ["situ_linear_beta"], situ["swiglu_limit"]
    gate, up = gate.clamp(max=limit), up.clamp(min=-limit, max=limit)
    return beta * torch.tanh(gate / beta) * torch.sigmoid(gate) * linear_beta * torch.tanh(up / linear_beta)


def reference(case, act, situ):
    """FP32 output of the valid rows; the activation output is re-quantized to FP8 per (row, 128) as GateUp does."""
    valid, inter_dim = case["valid"], case["inter_dim"]
    a1 = case["a1"][:valid].float()
    a = (a1.view(valid, -1, BLK) * case["a1_scale_t"].t()[:valid, :, None]).view(valid, -1)
    token, _, expert, weight = case["routes"]
    out = torch.zeros_like(a)
    for e in torch.unique(expert).tolist():
        routes = (expert == e).nonzero().squeeze(1)
        w1 = dequant_blocks(case["w1_raw"][e], case["w1_scale"][e])
        w2 = dequant_blocks(case["w2_raw"][e], case["w2_scale"][e])
        gate, up = (a[token[routes]] @ w1.t()).split(inter_dim, dim=-1)
        h = act_ref(gate, up, act, situ).view(routes.numel(), -1, BLK)
        h_scale = h.abs().amax(dim=-1, keepdim=True).clamp_min(1e-10) / FP8_MAX
        h = ((h / h_scale).to(FP8).float() * h_scale).view(routes.numel(), inter_dim)
        out.index_add_(0, token[routes], h @ w2.t() * weight[routes, None])
    return out


def run_moe(case, inputs, out, act, situ):
    """The fused entry on inputs (the case or one of its clones); SiLU calls pass no act, as aiter's hook."""
    sort = case["sort"]
    act_kwargs = {} if act == "silu" else dict(act=act, situ=situ)
    return fused_moe_a8w8_blockscale_gfx942(
        inputs["a1"], inputs["a1_scale_t"], inputs["w1"], inputs["w1_scale"], inputs["w2"], inputs["w2_scale"],
        sort["sorted_ids"], sort["sorted_weights"], sort["sorted_expert_ids"], sort["num_valid_ids"], out,
        case["topk"], num_rows=case["num_rows"], **act_kwargs,
    )


def check(case, out, expected, diff_thr):
    """calc_diff of the valid rows against the reference and the list of failed checks."""
    valid = case["valid"]
    diff = pyhip.calc_diff(out[:valid].float(), expected)
    problems = [] if diff <= diff_thr else [f"calc_diff {diff:.3g} > {diff_thr:g}"]
    if not out[valid:].isnan().all():
        problems.append("rows past num_rows were written")
    unrouted = torch.bincount(case["routes"][0], minlength=valid) == 0
    if not (out[:valid][unrouted] == 0).all():
        problems.append("rows without a local route are not zero")
    return diff, problems


def clone_count(case, requested):
    """(clones, bytes per clone, clones needed for cold reads when memory capped the count, else None)."""
    per_clone = sum(nbytes(case[key]) for key in CLONED)
    if requested:
        return requested, per_clone, None
    experts_hit = case["routes"][2].unique().numel()
    weight_bytes = sum(nbytes(case[key]) for key in ("w1", "w1_scale", "w2", "w2_scale"))
    read = (nbytes(case["a1"]) + nbytes(case["a1_scale_t"])) * case["valid"] // case["rows"]
    read += weight_bytes * experts_hit // case["local_experts"]
    needed = max(2, 1 + math.ceil(COLD_BYTES / read))
    count = max(2, min(needed, CLONE_LIMIT_BYTES // per_clone))
    return count, per_clone, needed if count < needed else None


def time_acts(case, clones, out, acts, situ, iters, warmup):
    """GPU microseconds of every timed call per activation; consecutive calls read different clones."""
    perfs = {act: pyhip.cudaPerf(name=f"moe_a8w8_blockscale {act}", verbose=0) for act in acts}
    if not all(perf.enable for perf in perfs.values()):
        raise RuntimeError("pyhip.cudaPerf is disabled by the CUDAPERF environment variable")
    calls = itertools.count()

    def run(act):
        run_moe(case, clones[next(calls) % len(clones)], out, act, situ)

    for act in acts:
        for _ in range(warmup):
            run(act)
    torch.cuda.synchronize()
    # Interleaved, so clock or thermal drift affects every activation alike.
    for _ in range(iters):
        for act in acts:
            with perfs[act]:
                run(act)
    return {act: [latency * 1e6 for latency in perf.latencies] for act, perf in perfs.items()}


def run_point(tokens, mode="a2a", rows=None, hidden_dim=MODEL[0], inter_dim=MODEL[1], experts=MODEL[2],
              topk=MODEL[3], ep=8, routing="random", seed=0, acts=ACTS, situ=SITU, iters=40, warmup=4,
              clones=0, diff_thr=1e-4):
    """Accuracy and timing of every activation on one case; dict(skipped=reason) for unsupported shapes."""
    reason = unsupported_reason(rows or tokens, hidden_dim, inter_dim, topk, experts // ep)
    if reason:
        return dict(tokens=tokens, skipped=reason)
    case = make_case(tokens, mode, rows, hidden_dim, inter_dim, experts, topk, ep, routing, seed)
    expected = {act: reference(case, act, situ) for act in acts}
    del case["w1_raw"], case["w2_raw"]
    out = torch.empty((case["rows"], hidden_dim), device="cuda", dtype=torch.bfloat16)
    results = {}
    for act in acts:
        out.fill_(float("nan"))
        run_moe(case, case, out, act, situ)
        torch.cuda.synchronize()
        diff, problems = check(case, out, expected[act], diff_thr)
        results[act] = dict(calc_diff=diff, problems=problems)
    del expected
    count, per_clone, needed = clone_count(case, clones)
    copies = [case] + [{key: case[key].clone() for key in CLONED} for _ in range(count - 1)]
    samples = time_acts(case, copies, out, acts, situ, iters, warmup)
    del copies
    routes = case["routes"][0].numel()
    # GateUp 2 * routes * 2I * H + Down 2 * routes * H * I.
    flops = 6 * routes * inter_dim * hidden_dim
    for act in acts:
        times = sorted(samples[act])
        median = statistics.median(times)
        results[act].update(
            median_us=median, p10_us=times[len(times) // 10], p90_us=times[len(times) * 9 // 10],
            tflops=flops / median / 1e6, samples=len(times),
        )
    return dict(
        tokens=tokens, mode=mode, rows=case["rows"], valid=case["valid"], routes=routes,
        experts_hit=case["routes"][2].unique().numel(), local_experts=case["local_experts"], flops=flops,
        clones=count, clone_gib=count * per_clone / 2**30, clones_needed=needed, acts=results,
    )


@pytest.mark.parametrize("mode", ["a2a", "ep"])
@pytest.mark.parametrize("act", ACTS)
def test_moe(mode, act):
    if not is_gfx942():
        pytest.skip("gfx942 kernels")
    if act != "silu" and not HAS_ACT:
        pytest.skip(f"pyhip at {pyhip.__file__} has no act / situ")
    point = run_point(2048, mode=mode, acts=[act], iters=2, warmup=1, clones=2)
    result = point["acts"][act]
    assert not result["problems"], result["problems"]
    assert result["samples"] == 2 and result["median_us"] > 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-t", "--tokens", type=int, nargs="+", default=[4096, 16384, 32768],
                        help="a2a: tokens of the EP group per step; ep: tokens of this rank (default: %(default)s)")
    parser.add_argument("--mode", default="a2a", choices=["a2a", "ep"])
    parser.add_argument("--rows", type=int, help="a2a receive-buffer rows (default: --tokens)")
    parser.add_argument("--model-dim", type=int, default=MODEL[0], help="hidden size")
    parser.add_argument("--inter-dim", type=int, default=MODEL[1], help="moe_intermediate_size")
    parser.add_argument("--experts", type=int, default=MODEL[2], help="global routed experts")
    parser.add_argument("--topk", type=int, default=MODEL[3])
    parser.add_argument("--ep", type=int, default=8, help="EP size: this rank owns experts [0, experts / ep)")
    parser.add_argument("--routing", default="random", choices=ROUTINGS)
    parser.add_argument("--act", nargs="+", default=list(ACTS), choices=ACTS, help="activations, timed side by side")
    parser.add_argument("--situ-beta", type=float, default=SITU["situ_beta"])
    parser.add_argument("--situ-linear-beta", type=float, default=SITU["situ_linear_beta"])
    parser.add_argument("--swiglu-limit", type=float, default=SITU["swiglu_limit"], help="SiTUv2 clamp; inf: none")
    parser.add_argument("--iters", type=int, default=40, help="timed calls per activation: the samples of its median")
    parser.add_argument("--warmup", type=int, default=4, help="untimed calls per activation")
    parser.add_argument("--clones", type=int, default=0, help="input copies the calls rotate over (0: enough for cold)")
    parser.add_argument("--diff-thr", type=float, default=1e-4, help="calc_diff limit")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", help="write the results to this file")
    args = parser.parse_args()
    if not is_gfx942():
        sys.exit("gfx942 kernels: no gfx942 GPU")
    if args.experts % args.ep or not 0 < args.topk <= args.experts:
        parser.error("need --experts divisible by --ep and 0 < --topk <= --experts")
    if args.mode == "ep" and args.rows is not None:
        parser.error("--rows applies to --mode a2a only")
    if args.iters < 1 or args.warmup < 0 or args.clones == 1 or args.clones < 0:
        parser.error("need --iters >= 1, --warmup >= 0 and --clones 0 or >= 2")
    acts = list(dict.fromkeys(args.act))
    situ = dict(situ_beta=args.situ_beta, situ_linear_beta=args.situ_linear_beta, swiglu_limit=args.swiglu_limit)
    if "situv2" in acts:
        if not HAS_ACT:
            sys.exit(f"pyhip at {pyhip.__file__} has no act / situ: install a pyhip with SiTUv2 or pass --act silu")
        from pyhip.contrib.flydsl.moe_gateup_a8w8_blockscale_gfx942_lds import activation_args

        try:
            activation_args("situv2", situ)
        except ValueError as error:
            parser.error(str(error))

    local = args.experts // args.ep
    print(f"pyhip {pyhip.__file__} (act / situ: {'yes' if HAS_ACT else 'no'}), {torch.cuda.get_device_name()}")
    print(f"MoE: hidden {args.model_dim}, inter {args.inter_dim}, {args.experts} experts / EP{args.ep} = {local} local, "
          f"top-{args.topk}, {args.routing} routing, mode {args.mode}"
          + (f"; SiTUv2 {situ}" if "situv2" in acts else ""), flush=True)
    points, failed = [], False
    for tokens in args.tokens:
        point = run_point(
            tokens, args.mode, args.rows, args.model_dim, args.inter_dim, args.experts, args.topk, args.ep,
            args.routing, args.seed, acts, situ, args.iters, args.warmup, args.clones, args.diff_thr,
        )
        points.append(point)
        if "skipped" in point:
            print(f"T={tokens}: skipped, {point['skipped']}", flush=True)
            continue
        cold = "" if point["clones_needed"] is None else f", cold reads need {point['clones_needed']}: may be warm"
        received = f"{point['valid']} received, " if args.mode == "a2a" else ""
        print(f"T={tokens}: {point['rows']} rows, {received}{point['routes']} local routes, "
              f"{point['experts_hit']}/{point['local_experts']} experts hit, {point['flops'] / 1e12:.3f} TFLOP; "
              f"{point['clones']} clones ({point['clone_gib']:.2f} GiB){cold}")
        for act, result in point["acts"].items():
            failed |= bool(result["problems"])
            status = "ok" if not result["problems"] else "FAIL: " + "; ".join(result["problems"])
            print(f"  {act:7s} calc_diff {result['calc_diff']:.2e} {status:4s}  median {result['median_us']:8.1f} us"
                  f"  p10-p90 {result['p10_us']:.1f}-{result['p90_us']:.1f}  {result['tflops']:7.1f} TFLOPS"
                  f"  ({result['samples']} calls)")
        if len(acts) == 2:
            ratio = point["acts"][acts[1]]["median_us"] / point["acts"][acts[0]]["median_us"]
            print(f"  {acts[1]} / {acts[0]}: {ratio:.4f}", flush=True)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(dict(args=vars(args), pyhip=pyhip.__file__, device=torch.cuda.get_device_name(), points=points),
                      f, indent=1)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
