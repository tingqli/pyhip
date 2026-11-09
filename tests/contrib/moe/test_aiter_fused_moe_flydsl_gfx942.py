"""aiter.fused_moe with AITER_FMOE_FP8_BLOCKSCALE_FLYDSL=1 on gfx942: the FP8 block-scale MoE runs the pyhip FlyDSL
GateUp + Down (pyhip.contrib.flydsl.moe_a8w8_blockscale_gfx942) instead of the 1-stage asm fmoe_fp8_blockscale_g1u1.

Covers how SGLang calls fused_moe: EP with expert_mask and BF16 activations quantized inside fused_moe; MORI
all-to-all with FP8 activations, token-major a1_scale, num_local_tokens below the receive-buffer rows and an output
buffer; and no EP. Needs an aiter with the hook (luocheng25/aiter tq/moe_gemm_blockscale) and pyhip importable.

perf: HIP_VISIBLE_DEVICES=0 python test_aiter_fused_moe_flydsl_gfx942.py --tokens 32768
"""

import argparse
import os
import statistics
from unittest import mock

import aiter.fused_moe as fm
import pyhip
import pytest
import torch
from aiter import ActivationType, QuantType, dtypes
from aiter.ops.quant import get_hip_quant
from test_fmoe_a8w8_blockscale_asm_gfx942 import (
    check,
    is_gfx942,
    make_inputs,
    reference,
)

ENV = "AITER_FMOE_FP8_BLOCKSCALE_FLYDSL"

pytestmark = [
    pytest.mark.skipif(not is_gfx942(), reason="gfx942 only"),
    pytest.mark.skipif(not hasattr(fm, "_fused_moe_flydsl_gfx942_fp8_blockscale"), reason="aiter without the FlyDSL hook"),
]


def take(inputs, rows):
    """The inputs of the tokens rows (a 1-D index), routes recomputed."""
    _, inter_dim, hidden_dim, topk, experts, ep = inputs["shape"]
    topk_ids = inputs["topk_ids"][rows]
    token, slot = (topk_ids < experts).nonzero(as_tuple=True)
    return dict(
        inputs,
        a=inputs["a"][rows],
        a_scale_t=inputs["a_scale_t"][:, rows].contiguous(),
        topk_ids=topk_ids,
        topk_weights=inputs["topk_weights"][rows],
        routes=(token, slot, topk_ids[token, slot].long()),
        shape=(rows.numel(), inter_dim, hidden_dim, topk, experts, ep),
    )


def a2a_inputs(inputs, rows):
    """MORI-style receive buffer of rows tokens: the tokens with a local route first, garbage after them."""
    received = take(inputs, (inputs["topk_ids"] < inputs["shape"][4]).any(dim=1).nonzero().squeeze(1))
    n = received["shape"][0]
    assert n <= rows
    hidden_dim = inputs["shape"][2]
    a = torch.randn((rows, hidden_dim), device="cuda").to(dtypes.fp8)
    a[:n] = received["a"]
    a1_scale = torch.full((rows, hidden_dim // 128), float("nan"), device="cuda")
    a1_scale[:n] = received["a_scale_t"].t()
    topk_ids = inputs["topk_ids"][torch.randint(0, inputs["shape"][0], (rows,), device="cuda")]
    topk_ids[:n] = received["topk_ids"]
    topk_weights = torch.rand((rows, inputs["shape"][3]), device="cuda")
    topk_weights[:n] = received["topk_weights"]
    buffers = dict(a=a, a1_scale=a1_scale, topk_ids=topk_ids, topk_weights=topk_weights,
                   num_local_tokens=torch.tensor([n], device="cuda", dtype=torch.int32))
    return received, buffers


def call(inputs, hidden_states, a1_scale=None, topk_ids=None, topk_weights=None, w=None, **kwargs):
    w = w or inputs
    return fm.fused_moe(
        hidden_states, w["w1"], w["w2"], inputs["topk_weights"] if topk_weights is None else topk_weights,
        inputs["topk_ids"] if topk_ids is None else topk_ids, expert_mask=inputs["expert_mask"],
        activation=ActivationType.Silu, quant_type=QuantType.per_128x128, w1_scale=w["w1_scale"],
        w2_scale=w["w2_scale"], a1_scale=a1_scale, dtype=dtypes.bf16, **kwargs,
    )


def requantized(inputs):
    """BF16 activations and the inputs fused_moe quantizes them to (aiter per-1x128 quant)."""
    tokens, _, hidden_dim = inputs["shape"][:3]
    x = (inputs["a"].float().view(tokens, -1, 128) * inputs["a_scale_t"].t()[:, :, None]).view(tokens, hidden_dim)
    x = x.to(torch.bfloat16)
    a, scale = get_hip_quant(QuantType.per_1x128)(x, quant_dtype=dtypes.fp8)
    return x, dict(inputs, a=a, a_scale_t=scale.t().contiguous())


@pytest.fixture
def flydsl(monkeypatch):
    """Enables the hook and fails the test if fused_moe falls back to the default path."""
    monkeypatch.setenv(ENV, "1")
    monkeypatch.setenv(f"{ENV}_MIN_TOKENS", "1")
    spy = mock.patch.object(
        fm, "_fused_moe_flydsl_gfx942_fp8_blockscale", wraps=fm._fused_moe_flydsl_gfx942_fp8_blockscale
    )
    with spy as hook:
        yield
        assert hook.called, "fused_moe did not take the FlyDSL path"


SHAPES = [
    (1000, 512, 1024, 4, 8, 4, "random"),
    (4096, 2048, 4096, 8, 32, 8, "random"),
]


@pytest.mark.parametrize("tokens,inter_dim,hidden_dim,topk,experts,ep,routing", SHAPES)
def test_ep_bf16_input(flydsl, tokens, inter_dim, hidden_dim, topk, experts, ep, routing):
    inputs = make_inputs(tokens, inter_dim, hidden_dim, topk, experts, ep, routing)
    x, quantized = requantized(inputs)
    out = call(inputs, x)
    # Rows of tokens without a local route are zero.
    check(quantized, out, reference(quantized))


@pytest.mark.parametrize("tokens,inter_dim,hidden_dim,topk,experts,ep,routing", SHAPES)
def test_a2a_fp8_input(flydsl, tokens, inter_dim, hidden_dim, topk, experts, ep, routing):
    received, buffers = a2a_inputs(make_inputs(tokens, inter_dim, hidden_dim, topk, experts, ep, routing), tokens)
    n = received["shape"][0]
    output = torch.full((tokens, hidden_dim), float("nan"), device="cuda", dtype=torch.bfloat16)
    out = call(received, buffers["a"], buffers["a1_scale"], buffers["topk_ids"], buffers["topk_weights"],
               num_local_tokens=buffers["num_local_tokens"], output=output)
    assert out.data_ptr() == output.data_ptr()
    check(received, out[:n], reference(received))
    assert out[n:].isnan().all(), "rows past num_local_tokens were written"


def test_no_ep(flydsl):
    inputs = make_inputs(1000, 512, 1024, 4, 8, 1, "random")
    x, quantized = requantized(inputs)
    check(quantized, call(inputs, x), reference(quantized))


def time_calls(fn, clones, iters, warmup):
    for i in range(warmup):
        fn(clones[i % len(clones)])
    torch.cuda.synchronize()
    events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(iters)]
    for i, (start, end) in enumerate(events):
        start.record()
        fn(clones[i % len(clones)])
        end.record()
    torch.cuda.synchronize()
    return [start.elapsed_time(end) * 1000 for start, end in events]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tokens", type=int, default=32768, help="EP-group tokens per step = receive-buffer rows")
    parser.add_argument("--routing", default="random", choices=["random", "ep_uniform", "balanced"])
    parser.add_argument("--mode", default="a2a", choices=["a2a", "ep"],
                        help="a2a: FP8 receive buffer + num_local_tokens; ep: every token on this rank, BF16 input")
    parser.add_argument("--clones", type=int, default=5)
    parser.add_argument("--iters", type=int, default=16)
    parser.add_argument("--rounds", type=int, default=4)
    args = parser.parse_args()

    I, H, TOPK, E, EP = 2048, 4096, 8, 32, 8
    inputs = make_inputs(args.tokens, I, H, TOPK, E, EP, args.routing)
    if args.mode == "a2a":
        case, buffers = a2a_inputs(inputs, args.tokens)
        hidden, kwargs = buffers["a"], dict(a1_scale=buffers["a1_scale"], topk_ids=buffers["topk_ids"],
                                            topk_weights=buffers["topk_weights"],
                                            num_local_tokens=buffers["num_local_tokens"])
        rows = case["shape"][0]
    else:
        hidden, case = requantized(inputs)
        kwargs, rows = {}, args.tokens
    expected = reference(case)
    clones = [inputs] + [{k: inputs[k].clone() for k in ("w1", "w1_scale", "w2", "w2_scale")} for _ in range(args.clones - 1)]
    print(f"{args.mode}: {args.tokens} buffer rows, {rows} valid tokens, {case['routes'][0].numel()} local routes", flush=True)
    os.environ[f"{ENV}_MIN_TOKENS"] = "1"
    samples = {"asm (default)": [], "flydsl": []}
    for r in range(args.rounds):
        order = ["asm (default)", "flydsl"] if r % 2 == 0 else ["flydsl", "asm (default)"]
        for name in order:
            os.environ[ENV] = "1" if name == "flydsl" else "0"
            samples[name] += time_calls(lambda w: call(case, hidden, w=w, **kwargs), clones, args.iters, warmup=8)
    for name, values in samples.items():
        os.environ[ENV] = "1" if name == "flydsl" else "0"
        diff = pyhip.calc_diff(call(case, hidden, **kwargs)[:rows].float(), expected)
        print(f"  {name:14s} median {statistics.median(values):8.1f} us  p10-p90 "
              f"{sorted(values)[len(values) // 10]:.1f}-{sorted(values)[len(values) * 9 // 10]:.1f}  calc_diff {diff:.2e}")
    ratio = statistics.median(samples["asm (default)"]) / statistics.median(samples["flydsl"])
    print(f"  speedup {ratio:.2f}x", flush=True)


if __name__ == "__main__":
    main()
