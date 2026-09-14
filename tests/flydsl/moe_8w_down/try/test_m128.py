# SPDX-License-Identifier: MIT
"""Independent M128 CTA rolling prefetch, padded output and graph correctness."""

import pytest
import torch
from tune_m128 import make_pipeline, check_pipeline
from tune_mfma32_xcd import reference, repack_weight


@pytest.mark.parametrize("name,n,tokens,experts,topk", [
    ("m128_pf3", 512, 33, 3, 2),
    ("m128_pf3", 6144, 513, 8, 8),
    ("m128_pf2", 6144, 257, 4, 2),
    ("m128_pf2r2", 512, 33, 3, 2),
    ("m128_pf3r3", 6144, 257, 4, 2),
    ("m128_pf4r4", 6144, 513, 8, 8),
    ("m128_pf3late", 6144, 257, 4, 2),
    ("m128_pf3noskip", 512, 33, 3, 2),
    ("m128_nodpp", 512, 33, 3, 2),
    ("m128_pkfp32", 6144, 513, 8, 8),
    ("m128_pk_nodpp", 6144, 513, 8, 8),
    ("m128_pk_r3", 6144, 257, 4, 2),
    ("m128_direct", 6144, 513, 8, 8),
    ("m64_oc8r3", 6144, 513, 8, 8),
    ("m256_rolling8", 6144, 513, 8, 8),
    ("m128_8w", 6144, 513, 8, 8),
    ("m256_4w", 6144, 513, 8, 8),
    ("m128_dma_mix", 6144, 513, 8, 8),
    ("m128_store_mix", 6144, 513, 8, 8),
    ("m128_vmem_mix", 6144, 513, 8, 8),
    ("m128_shortlive", 6144, 513, 8, 8),
    ("m128_shortlive_pk", 6144, 513, 8, 8),
    ("m128_k128", 6144, 513, 8, 8),
    ("m128_k128pk", 6144, 513, 8, 8),
    ("m128_raw", 6144, 513, 8, 8),
    ("m128_sorted_a", 6144, 513, 8, 8),
    ("m256_4w_k128", 6144, 513, 8, 8),
    ("m128_group4", 6144, 33, 3, 2),
    ("m128_nmajor", 6144, 33, 3, 2),
    ("m128_outn", 6144, 513, 8, 8),
    ("tr64n128", 6144, 513, 8, 8),
    ("tr128n128g4", 6144, 513, 8, 8),
    ("m128_atomic", 6144, 513, 8, 8),
    ("m128_mfma32_8_True", 6144, 513, 8, 8),
    ("m128_n32", 6144, 513, 8, 8),
    ("m128_n32pk", 6144, 513, 8, 8),
    ("m128_fast_sum", 6144, 513, 8, 8),
    ("m128_bstream", 6144, 513, 8, 8),
    ("m128_bstream_r3", 6144, 513, 8, 8),
    ("m128_bstream_short", 6144, 513, 8, 8),
    ("m128_direct_meta", 6144, 513, 8, 8),
    ("m128_early_guard", 6144, 513, 8, 8),
    ("m128_wide", 6144, 513, 8, 8),
    ("m128_pad264", 6144, 513, 8, 8),
    ("m128_relaxed", 6144, 513, 8, 8),
    ("m128_relaxed_pk", 6144, 513, 8, 8),
    ("m128_rotate1", 6144, 513, 8, 8),
    ("m128_mfma32_roll8_14", 6144, 513, 8, 8),
    ("m128_8w_mix_r3", 6144, 513, 8, 8),
    ("m128_priority3", 6144, 513, 8, 8),
    ("m128_static256", 6144, 513, 8, 8),
])
def test_m128_pipeline(name, n, tokens, experts, topk):
    if not torch.cuda.is_available() or not torch.cuda.get_device_properties().gcnArchName.startswith("gfx950"):
        pytest.skip("gfx950 required")
    inputs, expected_routes = reference.make_case(tokens, n, experts, topk, 2026)
    guard = torch.full((tokens * n + 2 * n,), torch.nan, dtype=torch.bfloat16, device="cuda")
    output = guard[n:-n].view(tokens, n)
    args = (output, *inputs)
    baseline = make_pipeline("m256", n=n, topk=topk, experts=experts)
    baseline(*args)
    torch.cuda.synchronize()
    decoded = reference.unpack_routes(baseline.workspace.data, inputs[4], inputs[7], tokens, topk)
    torch.testing.assert_close(decoded, expected_routes, rtol=0.01, atol=0.01)
    expected = decoded.sum(dim=1)
    kernel = make_pipeline(name, n=n, topk=topk, experts=experts)
    if kernel.config.get("weight_layout") == (32, 16):
        args[2].copy_(repack_weight(args[2]))
    args[-1].fill_(0x123456)
    kernel.poison_workspace(*args)
    kernel(*args)
    torch.cuda.synchronize()

    def check():
        check_pipeline(kernel, args, expected_routes, expected)
        assert torch.isnan(guard[:n]).all() and torch.isnan(guard[-n:]).all()
        assert args[-1].item() == 0x123456

    check()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        kernel(*args)
    for _ in range(3):
        kernel.poison_workspace(*args)
        output.fill_(torch.nan)
        graph.replay()
        torch.cuda.synchronize()
        check()
    ids, valid = args[5].clone(), args[8].clone()
    args[5].fill_((topk << 24) | tokens)
    kernel.poison_workspace(*args)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.count_nonzero(output).item() == 0
    args[8].zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.count_nonzero(output).item() == 0
    args[5].copy_(ids)
    args[8].copy_(valid)
    graph.replay()
    torch.cuda.synchronize()
    check()


def test_m128_selected_cancellation():
    from aiter.ops.shuffle import shuffle_weight

    if not torch.cuda.is_available() or not torch.cuda.get_device_properties().gcnArchName.startswith("gfx950"):
        pytest.skip("gfx950 required")
    tokens, topk, n, experts = 3, 2, 1024, 2
    a = torch.zeros((tokens, topk, 256), dtype=torch.bfloat16, device="cuda")
    a[..., 0] = a[..., 128] = 1
    b = torch.zeros((experts, n, 256), dtype=torch.bfloat16, device="cuda")
    b[0, :, 0], b[0, :, 128], b[1, :, 0] = 16, -5, -4
    sa = torch.ones((2, tokens * topk), dtype=torch.float32, device="cuda")
    sb = torch.ones((experts, n // 128, 2), dtype=torch.float32, device="cuda")
    sb[0, :, 1] = 2.3968749046325684
    ids = torch.full((experts * 256,), (topk << 24) | tokens, dtype=torch.int32, device="cuda")
    routes = torch.zeros(ids.shape, dtype=torch.float32, device="cuda")
    for e in range(experts):
        ids[e * 256:e * 256 + tokens] = torch.arange(tokens, dtype=torch.int32, device="cuda") | (e << 24)
        routes[e * 256:e * 256 + tokens] = 0.5
    output = torch.full((tokens, n), torch.nan, dtype=torch.bfloat16, device="cuda")
    args = (output, a.to(torch.float8_e4m3fn), shuffle_weight(b.to(torch.float8_e4m3fn), layout=(16, 16)),
            sa, sb, ids, routes, torch.arange(experts, dtype=torch.int32, device="cuda"),
            torch.tensor([experts * 256], dtype=torch.int32, device="cuda"),
            torch.full((1,), 0x123456, dtype=torch.int32, device="cuda"))
    pipeline = make_pipeline("m128_priority3", n=n, topk=topk, experts=experts)
    pipeline(*args)
    torch.cuda.synchronize()
    torch.testing.assert_close(output, torch.full_like(output, 0.015625), rtol=0, atol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        pipeline(*args)
    output.fill_(torch.nan)
    pipeline.poison_workspace(*args)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(output, torch.full_like(output, 0.015625), rtol=0, atol=0)
    assert args[-1].item() == 0x123456