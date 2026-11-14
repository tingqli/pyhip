"""Per-kernel breakdown (cold cache, torch profiler) + e2e of the tuned configs on the
MiMo TP8 / EP8 shapes.

    cd src/contrib/flydsl && HIP_VISIBLE_DEVICES=0 python a8w4/experiments/20261114_r2_baseline/breakdown.py tp8 1 2 4 8 16
    ... breakdown.py ep8 1 2 4 8 16      # EP: every token routed to exactly `local` local experts

EP routing: ``--local L`` (default 1) makes each token's first L routes local experts
(distinct, random), the rest remote, so the rank's work is deterministic (uniform
routing gives ~1/3 empty ranks at 1 token); ``--local 0``: uniform routing over the
256 global experts (the tuned table's routing; use it for large batches).
"""
import argparse
import os
import sys
from collections import defaultdict

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from a8w4 import ref as R  # noqa: E402
from a8w4.moe import a8w4_moe, config_name, lookup_config, run_config  # noqa: E402

SHAPES = {"tp8": (4096, 256, 256, 256, 8), "ep8": (4096, 2048, 32, 256, 8)}


def ep_routing(T, H, EGl, E, K, local, seed):
    """x, tw, ti with exactly ``local`` distinct local experts (ids < E) per token."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn((T, H), dtype=torch.bfloat16, device="cuda", generator=g)
    ids = []
    for _ in range(T):
        loc = torch.randperm(E, device="cuda", generator=g)[:local]
        rem = E + torch.randperm(EGl - E, device="cuda", generator=g)[:K - local]
        ids.append(torch.cat([loc, rem]))
    ti = torch.stack(ids).to(torch.int32).contiguous()
    tw = torch.rand((T, K), device="cuda", generator=g)
    tw = (tw / tw.sum(-1, keepdim=True)).float().contiguous()
    return x, tw, ti


def kernel_times(fn, iters=20):
    """{kernel name: median us} over ``iters`` cold-cache calls."""
    from torch.profiler import ProfilerActivity, profile

    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    per = defaultdict(list)
    for _ in range(iters):
        R.flush_cache()
        torch.cuda._sleep(2_000_000)
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            fn()
            torch.cuda.synchronize()
        seen = defaultdict(float)
        for ev in prof.events():
            if ev.device_type == torch.autograd.DeviceType.CUDA:
                seen[ev.name] += ev.device_time_total if hasattr(ev, "device_time_total") else ev.cuda_time_total
        for k, v in seen.items():
            per[k].append(v)
    return {k: sorted(v)[len(v) // 2] for k, v in per.items()}


def short(name):
    for p in ("ep_prep", "ep_gemm1", "ep_gemm2", "ep_small", "a8w4_gemm1", "a8w4_gemm2",
              "a8w4_fmoe_bm64q", "a8w4_fmoe_bm64", "moe_sorting", "kernel_0", "amax", "fill",
              "elementwise"):
        if p in name:
            return p
    return name[:40]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shape", choices=SHAPES)
    ap.add_argument("tokens", type=int, nargs="+")
    ap.add_argument("--local", type=int, default=1)
    ap.add_argument("--config", default=None)
    ap.add_argument("--iters", type=int, default=20)
    a = ap.parse_args()
    H, I, E, EGl, K = SHAPES[a.shape]
    W, raw = R.make_weights(E, H, I)
    for T in a.tokens:
        if a.shape == "ep8" and a.local > 0:
            x, tw, ti = ep_routing(T, H, EGl, E, K, a.local, seed=T)
        else:
            x, tw, ti = R.make_routing(T, H, EGl, K, seed=7 + T)
        cfg = a.config or config_name(lookup_config((H, I, E, EGl, K), T))
        fn = lambda: run_config(cfg, x, W, tw, ti, expert_offset=0, num_experts=EGl)  # noqa: E731
        ref = R.ref_moe(x, tw, ti, raw, 0)
        cos = R.cos_diff(fn().float(), ref)
        e2e = R.time_cold(fn, iters=a.iters)
        kt = kernel_times(fn, iters=a.iters)
        parts = defaultdict(float)
        for k, v in kt.items():
            parts[short(k)] += v
        tot = sum(parts.values())
        print(f"{a.shape} T={T:5d} cfg='{cfg}' e2e {e2e:7.1f} us  kernels {tot:7.1f}  cos {cos:.1e}  "
              + "  ".join(f"{k} {v:.1f}" for k, v in sorted(parts.items(), key=lambda kv: -kv[1])),
              flush=True)


if __name__ == "__main__":
    main()
