# SPDX-License-Identifier: MIT
"""Targeted correctness for the opt-in MFMA32 path, including persistent reuse."""

import pytest
import torch

from moe_multistage_down_mfma32 import (
    MFMA32_EXPERIMENT_CONFIG, flydsl_moe_gemm_8wave_down_mfma32,
)
from test_8stage import make_case


@pytest.mark.parametrize("n,splits,tokens,ctas,fold", [
    (128, 1, 65, 256, True),
    (256, 1, 257, 256, True),
    (384, 1, 257, 256, True),
    (640, 1, 257, 256, True),
    (6144, 4, 513, 8, True),
    (512, 4, 129, 256, False),
])
def test_mfma32(n, splits, tokens, ctas, fold):
    if not torch.cuda.is_available() or not torch.cuda.get_device_properties().gcnArchName.startswith("gfx950"):
        pytest.skip("native gfx950 MFMA32 required")
    args, reference, guard = make_case(tokens, n, 256, 8, 2, 2026, "aiter", (32, 16))
    options = {**MFMA32_EXPERIMENT_CONFIG, "num_oc_splits": splits,
               "persistent_workgroups": ctas, "fold_routing": fold, "defer_k1": fold}
    kernel = flydsl_moe_gemm_8wave_down_mfma32(n=n, k=256, topk=2, num_experts=8, **options)

    def check():
        torch.testing.assert_close(args[0], reference, rtol=0.01, atol=0.01)
        assert torch.isfinite(args[0]).all()
        assert torch.isnan(guard[:n]).all() and torch.isnan(guard[-n:]).all()

    kernel(*args)
    torch.cuda.synchronize()
    check()
    capture = torch.cuda.CUDAGraph()
    with torch.cuda.graph(capture):
        kernel(*args)
    for _ in range(3):
        args[0].fill_(torch.nan)
        args[-1].fill_(0x123456)
        capture.replay()
        torch.cuda.synchronize()
        check()
    # A new launch must reset the queue, including for empty work.
    args[8].zero_()
    args[0].fill_(torch.nan)
    kernel(*args)
    torch.cuda.synchronize()
    assert torch.isnan(args[0]).all()
    assert args[-1].item() == ctas


@pytest.mark.parametrize("candidate,n,tokens,experts,topk,persistent", [
    ("mfma32_xcd", 512, 33, 3, 2, False),
    ("mfma32_defer14", 512, 33, 3, 2, False),
    ("mfma32_defer14", 6144, 513, 8, 8, False),
    ("mfma32_defer16", 6144, 513, 8, 8, False),
    ("mfma32_defer14_cache", 6144, 129, 8, 8, False),
    ("mfma32_defer14_cache_skip", 6144, 129, 8, 8, False),
    ("mfma32_defer14", 512, 257, 8, 2, True),
])
def test_mfma32_exact_xcd_pipeline(candidate, n, tokens, experts, topk, persistent):
    from tune_mfma32_xcd import check_pipeline, make_pipeline, reference, shared_case

    if not torch.cuda.is_available() or not torch.cuda.get_device_properties().gcnArchName.startswith("gfx950"):
        pytest.skip("native gfx950 MFMA32 required")
    args, expected_routes, weights = shared_case(tokens, n, experts, topk, 2026)
    baseline = make_pipeline("mfma16_xcd", n=n, topk=topk, experts=experts)
    baseline(*args)
    torch.cuda.synchronize()
    decoded = reference.unpack_routes(baseline.workspace.data, args[5], args[8], tokens, topk)
    torch.testing.assert_close(decoded, expected_routes, rtol=0.01, atol=0.01)
    expected_sum = decoded.sum(dim=1)
    guard = torch.full((expected_sum.numel() + 2 * n,), torch.nan, dtype=torch.bfloat16, device=args[0].device)
    args = (guard[n:-n].view_as(expected_sum), *args[1:])
    args[2].copy_(weights[(32, 16)])
    pipeline = make_pipeline(candidate, n=n, topk=topk, experts=experts,
                             persistent=persistent, xcd_swizzle=not persistent, persistent_workgroups=8)

    def check():
        check_pipeline(pipeline, args, expected_routes, expected_sum)
        assert torch.isnan(guard[:n]).all() and torch.isnan(guard[-n:]).all()
        if not persistent:
            assert args[-1].item() == 0x123456

    args[-1].fill_(0x123456)
    pipeline.poison_workspace(*args)
    pipeline(*args)
    torch.cuda.synchronize()
    check()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        pipeline(*args)
    for _ in range(3):
        pipeline.poison_workspace(*args)
        args[0].fill_(torch.nan)
        args[-1].fill_(0x123456)
        graph.replay()
        torch.cuda.synchronize()
        check()
    saved_ids, saved_valid = args[5].clone(), args[8].clone()
    args[5].fill_((topk << 24) | tokens)
    pipeline.poison_workspace(*args)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.count_nonzero(args[0]).item() == 0
    args[8].zero_()
    pipeline.poison_workspace(*args)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.count_nonzero(args[0]).item() == 0
    assert args[-1].item() == (8 if persistent else 0x123456)
    args[5].copy_(saved_ids)
    args[8].copy_(saved_valid)
    graph.replay()
    torch.cuda.synchronize()
    check()


@pytest.mark.parametrize("defer", [False, True])
def test_mfma32_exact_xcd_cancellation(defer):
    from aiter.ops.shuffle import shuffle_weight
    from tune_mfma32_xcd import make_pipeline

    if not torch.cuda.is_available() or not torch.cuda.get_device_properties().gcnArchName.startswith("gfx950"):
        pytest.skip("native gfx950 MFMA32 required")
    tokens, topk, n, k, experts = 3, 2, 512, 256, 2
    a = torch.zeros((tokens, topk, k), dtype=torch.bfloat16, device="cuda")
    a[..., 0] = a[..., 128] = 1
    w = torch.zeros((experts, n, k), dtype=torch.bfloat16, device="cuda")
    w[0, :, 0], w[0, :, 128], w[1, :, 0] = 16, -5, -4
    sa = torch.ones((2, tokens * topk), dtype=torch.float32, device="cuda")
    sb = torch.ones((experts, n // 128, 2), dtype=torch.float32, device="cuda")
    sb[0, :, 1] = 2.3968749046325684
    ids = torch.full((experts * 256,), (topk << 24) | tokens, dtype=torch.int32, device="cuda")
    routes = torch.zeros(ids.shape, dtype=torch.float32, device="cuda")
    for expert in range(experts):
        ids[expert * 256:expert * 256 + tokens] = torch.arange(tokens, dtype=torch.int32, device="cuda") | (expert << 24)
        routes[expert * 256:expert * 256 + tokens] = 0.5
    output = torch.full((tokens, n), torch.nan, dtype=torch.bfloat16, device="cuda")
    args = (output, a.to(torch.float8_e4m3fn), shuffle_weight(w.to(torch.float8_e4m3fn), layout=(32, 16)),
            sa, sb, ids, routes, torch.arange(experts, dtype=torch.int32, device="cuda"),
            torch.tensor([experts * 256], dtype=torch.int32, device="cuda"),
            torch.full((1,), 0x123456, dtype=torch.int32, device="cuda"))
    pipeline = make_pipeline("mfma32_defer14" if defer else "mfma32_xcd", n=n, topk=topk, experts=experts)
    pipeline(*args)
    torch.cuda.synchronize()
    torch.testing.assert_close(output, torch.full_like(output, 0.015625), rtol=0, atol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        pipeline(*args)
    pipeline.poison_workspace(*args)
    output.fill_(torch.nan)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(output, torch.full_like(output, 0.015625), rtol=0, atol=0)
    assert args[-1].item() == 0x123456