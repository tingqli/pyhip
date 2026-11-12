"""Standalone GateUp (FP8 out, 2x4_pf1) on one EP8 rank's balanced load: 4096 tokens x top-8 over 32 local experts =
128 full blocks, as the mega kernel's GateUp tasks see on each rank. A scales either transposed [K/128, T] (as the
single-kernel MoE path reads them) or token-major [T, K/128] (as the mega kernel reads recv_scale).

    HIP_VISIBLE_DEVICES=0 python tests/contrib/moe/ep8/att_gateup_scale_layout.py --token-major 0|1
    ATT: HIP_VISIBLE_DEVICES=0 rocprofv3 --att --att-library-path DECODER_DIR --att-target-cu 1 \\
             --att-shader-engine-mask 0xffff --att-simd-select 0x1 --att-buffer-size 0x40000000 \\
             --kernel-include-regex gateup_kernel --kernel-iteration-range "[3, 3]" -d OUT -o att \\
             -- python tests/contrib/moe/ep8/att_gateup_scale_layout.py --token-major 1 --calls 4
         python tests/contrib/moe/bench_fused_moe_a2a_gfx942.py phases OUT/ui_output_agent_*_dispatch_*
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from test_gateup_a8w8_blockscale_gfx942 import make_inputs, run_kernel  # noqa: E402

T, I, H, TOPK, E = 4096, 2048, 4096, 8, 32


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--token-major", type=int, default=0)
    parser.add_argument("--calls", type=int, default=8, help="untimed calls first (ATT picks one of them)")
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()
    inputs = make_inputs(T, I, H, TOPK, E, routing="balanced")
    blocks = inputs["expert_ids"].numel()
    a2 = torch.empty((blocks * 256, I), dtype=torch.uint8, device="cuda").view(inputs["a"].dtype)
    a2_scale_t = torch.empty((I // 128, blocks * 256), dtype=torch.float32, device="cuda")
    if args.token_major:
        inputs["a_scale_t"] = inputs["a_scale_t"].t().contiguous()

    def call():
        run_kernel(inputs, a2, out_scale_t=a2_scale_t, topk=TOPK, a_scale_token_major=bool(args.token_major))

    for _ in range(args.calls):
        call()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(args.iters):
        call()
    end.record()
    torch.cuda.synchronize()
    print(f"blocks {blocks} token_major {args.token_major}: {start.elapsed_time(end) / args.iters * 1e3:.1f} us per call")


if __name__ == "__main__":
    main()
