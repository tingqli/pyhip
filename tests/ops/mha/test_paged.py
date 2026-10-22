"""Paged MHA (SHUFFLE-5D KV) correctness: BF16 gfx942/gfx950, FP8 gfx942 and SWA."""

import torch
import pytest

from tests.ops.mha._mha import (BF16_942, BF16_942_GRID, BF16_950, BF16_950_PERSISTENT, FP8, SWA, Workload,
                                accuracy, check_paged, gpu_arch, make_call, make_case, require, torch_reference,
                                vectorize_kv)


def test_bf16_256_default_factory():
    """Public D256 keeps the validated paged path; smaller dimensions stay unchanged."""
    native = BF16_942.load()
    from pyhip.ops.mha.flydsl import mha_pa_bf16_256_paged_942 as paged
    for page in (32, 64, 128):
        for persistent in (None, True, False):
            default = native.PagedAttention(24, 2, 256, 256, page, False, persistent=persistent)
            assert default._launch is paged._launch
            assert default.persistent is (persistent is not False)
            assert default.bf16_backend.endswith("-pages1-lateS41")
    for dq in (128, 192):
        for dv in (128, 192):
            assert native.PagedAttention(24, 2, dq, dv, 64, False)._launch is native._launch_attention


@pytest.mark.parametrize("variant", ("8wave", "persistent"))
@pytest.mark.parametrize("dq", (128, 192))
def test_bf16_mha(dq, variant):
    """Short ragged/empty/poisoned edges not covered by benchmark shapes."""
    require("gfx942", "gfx950")
    arch = gpu_arch()
    backends = [BF16_942 if variant == "persistent" else BF16_942_GRID] if arch == "gfx942" else [
        BF16_950_PERSISTENT if variant == "persistent" else BF16_950]
    kv = (63, 0, 193, 257) if arch == "gfx950" else (63, 65, 193, 257)
    w = Workload("ragged_edges", (0, 7, 33, 129), kv, dq=dq, heads=6, kv_heads=2, causal=True)
    check_paged(w, backends, layout="padded" if backends[0].strided else "contiguous", nonunit_scales=True,
                poison_tail=arch != "gfx942", softmax_scale=0.0625)


@pytest.mark.parametrize("kv_len", (31, 64, 65, 128, 129))
@pytest.mark.parametrize("dq", (128, 192, 256))
def test_bf16_stage_boundaries(kv_len, dq):
    """One/two/three BN64 tiles exercise prologue, handoff and drain."""
    require("gfx942")
    w = Workload("bf16_stage_boundaries", (65,), (kv_len,), dq=dq, dv=256 if dq == 256 else 128,
                 heads=24 if dq == 256 else 16, kv_heads=2 if dq == 256 else 1)
    check_paged(w, [BF16_942])


@pytest.mark.parametrize("pattern", ("uniform", "values", "keys"))
def test_bf16_stage_operands(pattern):
    """Distinguish V staging, softmax normalization and QK handoff bugs."""
    require("gfx942")
    case = make_case((256,), (128,), dq=128, heads=1, poison_tail=False,
                     source_dtype=torch.bfloat16, seed=20260905)
    if pattern in ("uniform", "values"):
        case.q.zero_()
    if pattern in ("uniform", "keys"):
        case.v_pages.fill_(1)
        case.k, case.v = vectorize_kv(case.k_pages, case.v_pages)
    call, _, _ = make_call(case, BF16_942, False)
    actual = call()
    oracle = torch_reference(case, False)
    accuracy(actual, oracle, f"bf16_stage_operands/{pattern}")
    saved = actual.clone()
    for _ in range(2):
        torch.testing.assert_close(call(), saved, rtol=0, atol=0)


@pytest.mark.parametrize("page", (32, 64, 128))
@pytest.mark.parametrize("dq,dv", ((128, 128), (192, 128), (128, 192), (192, 192), (256, 256)))
def test_bf16_layout_contract(page, dq, dv):
    """SHUFFLE-5D, both schedulers, ragged prefixes, scales and optional LSE."""
    require("gfx942")
    case = make_case((0, 7, 33, 513), (31, 33, 97, 545), dq=dq, dv=dv, page=page,
                     heads=24 if dq == 256 else 12, kv_heads=2 if dq == 256 else 3,
                     mode="per-tensor" if page == 32 else "per-token",
                     q_offset=5, table_offset=3, nonunit_scales=True, poison_tail=True,
                     source_dtype=torch.bfloat16)
    valid = slice(case.q_offset, case.q_offset + sum(case.q_lens))
    for causal in (False, True):
        oracle, lse_oracle = torch_reference(case, causal, 0.0625, return_lse=True)
        for backend in (BF16_942, BF16_942_GRID):
            call, out, kernel = make_call(case, backend, causal)
            # The no-LSE hot path is a separate specialization; validate both.
            actual = call(softmax_scale=0.0625)
            accuracy(actual[valid], oracle[valid], f"{backend.name}/no_lse/d{dq}v{dv}p{page}")
            lse = torch.full(case.q.shape[:2], -123.0, device="cuda")
            actual, actual_lse = call(return_lse=True, lse=lse, softmax_scale=0.0625)
            assert actual.data_ptr() == out.data_ptr() and actual_lse.data_ptr() == lse.data_ptr()
            assert kernel.persistent is (backend == BF16_942)
            accuracy(actual[valid], oracle[valid], f"{backend.name}/layout/d{dq}v{dv}p{page}")
            torch.testing.assert_close(actual_lse[valid], lse_oracle[valid], atol=0.002, rtol=0.002)
            first, first_lse = actual.clone(), actual_lse.clone()
            for _ in range(2):
                assert call(lse=lse, softmax_scale=0.0625).data_ptr() == out.data_ptr()
                torch.testing.assert_close(out, first, atol=0, rtol=0)
                torch.testing.assert_close(lse, first_lse, atol=0, rtol=0)
            for buffer in (out, lse):
                assert bool((buffer[:valid.start] == -123).all())
                assert bool((buffer[valid.stop:] == -123).all())


@pytest.mark.parametrize("backend", (BF16_942, BF16_942_GRID), ids=lambda b: b.name)
@pytest.mark.parametrize("dim", (128, 256))
def test_bf16_streams_and_graph(backend, dim):
    """Persistent iterations must not share scheduler state across streams."""
    require("gfx942")
    case = make_case((1537,), (193,), dq=dim, dv=dim, heads=24 if dim == 256 else 16,
                     kv_heads=2 if dim == 256 else 1, poison_tail=False,
                     source_dtype=torch.bfloat16)
    oracle = torch_reference(case, False)
    calls = [make_call(case, backend, False)[:2] for _ in range(2)]
    streams = [torch.cuda.Stream() for _ in calls]
    for stream, (call, _) in zip(streams, calls):
        stream.wait_stream(torch.cuda.current_stream())
        call(stream=stream)
    for stream, (_, out) in zip(streams, calls):
        stream.synchronize()
        accuracy(out, oracle, f"{backend.name}/concurrent_stream")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=streams[0]):
        calls[0][0](stream=streams[0])
    for _ in range(2):
        graph.replay()
    torch.cuda.synchronize()
    accuracy(calls[0][1], oracle, f"{backend.name}/graph_replay")


@pytest.mark.parametrize("dq", (128, 192))
def test_swa(dq):
    """One fused ragged/empty/poisoned/scale case per head dimension."""
    require("gfx942", "gfx950")
    w = Workload("swa_ragged_edges", (0, 7, 33, 129), (63, 0, 193, 257),
                 dq=dq, heads=6, kv_heads=2, causal=True, window=128, sink=True)
    check_paged(w, [SWA], layout="padded", nonunit_scales=True, poison_tail=True, softmax_scale=0.0625)


@pytest.mark.parametrize("dq", (128, 192))
@pytest.mark.parametrize("causal", (False, True))
@pytest.mark.parametrize("mode", ("per-token", "per-tensor"))
def test_fp8_mha(dq, causal, mode):
    require("gfx942")
    w = Workload("fp8_lds_edges", (0, 7, 33, 129), (63, 65, 193, 257), dq=dq,
                 heads=6, kv_heads=2, causal=causal, scale_mode=mode)
    check_paged(w, [FP8], poison_tail=True)
