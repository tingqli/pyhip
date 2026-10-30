"""Paged MHA (SHUFFLE-5D KV) correctness: BF16 gfx942/gfx950, FP8 gfx942 and SWA."""

from dataclasses import replace

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


@pytest.mark.parametrize("dq,dv,return_lse", ((128, 128, False), (192, 128, False), (192, 128, True), (128, 192, False)))
def test_bf16_strided_q(dq, dv, return_lse):
    """Q strided over tokens and heads (padded rows, head-major) is bit-exact with contiguous Q."""
    require("gfx942")
    case = make_case((0, 7, 33, 513), (31, 33, 97, 545), dq=dq, dv=dv, heads=12, kv_heads=3,
                     q_offset=5, nonunit_scales=True, source_dtype=torch.bfloat16)
    padded = torch.zeros(case.q.shape[0], 13, dq + 16, device="cuda", dtype=case.q.dtype)[:, :12, :dq]
    padded.copy_(case.q)
    head_major = case.q.transpose(0, 1).contiguous().transpose(0, 1)
    for backend in (BF16_942, BF16_942_GRID):
        results = []
        for q in (case.q, padded, head_major):
            lse = torch.full(case.q.shape[:2], -123.0, device="cuda") if return_lse else None
            results.append(make_call(replace(case, q=q), backend, True)[0](return_lse=return_lse, lse=lse))
        for actual in results[1:]:
            torch.testing.assert_close(actual, results[0], rtol=0, atol=0)


def test_bf16_q_layout_rejections():
    """D256 reads contiguous Q only; overlapping Q rows would defeat the zero-filled tail rows."""
    require("gfx942")
    case = make_case((7,), (33,), dq=256, dv=256, heads=24, kv_heads=2, layout="padded",
                     source_dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="contiguous"):
        make_call(case, BF16_942, False)[0]()
    case = make_case((7,), (33,), dq=192, heads=12, kv_heads=3, source_dtype=torch.bfloat16)
    case = replace(case, q=case.q[:1].expand(case.q.shape[0], -1, -1))
    with pytest.raises(ValueError, match="rows at least Dq apart"):
        make_call(case, BF16_942, False)[0]()


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


# (Q, K, Q/K descale) for BF16 and FP8, giving equal logits per row: near +1e11 an FMA
# S * scale - m against the rounded max is thousands off (the FP8 K = 120 makes S * scale
# inexact); near -2e31 they sit below the former -1e30 floor.
EXTREME_LOGITS = {
    "near_1e11": ((65536.0, 66048.0, 1.0), (128.0, 120.0, 2.0**9)),
    "below_-1e30": ((2.0**50, -(2.0**50), 1.0), (128.0, -120.0, 2.0**45)),
}


def _constant_case(backend, dq, q_value, k_value, descale, small_k=None):
    """Constant Q/K (K = small_k on the first two pages of each sequence), V = 1: every output is 1."""
    case = make_case((65, 129), (193, 257), dtype=backend.dtype, dq=dq, dv=256 if dq == 256 else 128,
                     heads=4, kv_heads=2, mode="per-tensor", poison_tail=False,
                     window_left=128 if backend == SWA else -1, source_dtype=torch.bfloat16)

    def fill(tensor, value):
        tensor.copy_(torch.full(tensor.shape, value, device=tensor.device).to(tensor.dtype))

    fill(case.q, q_value)
    fill(case.k_pages, k_value)
    fill(case.v_pages, 1.0)
    if small_k is not None:
        for pos in (0, 1, 4, 5):
            fill(case.k_pages[case.page_order[pos]], small_k)
    case.qs.fill_(descale)
    case.ks.fill_(descale)
    case.k, case.v = vectorize_kv(case.k_pages, case.v_pages)
    return case


@pytest.mark.parametrize("logits", tuple(EXTREME_LOGITS))
@pytest.mark.parametrize("causal", (False, True))
@pytest.mark.parametrize("backend,dq", [(BF16_942, 128), (BF16_942, 192), (BF16_942, 256), (BF16_942_GRID, 128),
                                        (FP8, 128), (FP8, 192), (SWA, 128), (BF16_950, 128)],
                         ids=lambda value: value.name if hasattr(value, "name") else f"d{value}")
def test_extreme_logits(backend, dq, causal, logits):
    """Constant Q/K and V = 1 give O = 1 at logits near 1e11 and below -1e30."""
    require(*(("gfx942", "gfx950") if backend.arch == "both" else (backend.arch,)))
    if backend == SWA and not causal:
        pytest.skip("single-wave SWA is causal")
    q_value, k_value, descale = EXTREME_LOGITS[logits][backend.fp8]
    case = _constant_case(backend, dq, q_value, k_value, descale)
    call, out, _ = make_call(case, backend, causal)
    call()
    torch.testing.assert_close(out.float(), torch.ones_like(out, dtype=torch.float32), rtol=0, atol=0.01)


@pytest.mark.parametrize("backend,dq,lse", [(FP8, 128, False), (BF16_942, 128, False), (BF16_942, 192, False),
                                            (BF16_942, 128, True)],
                         ids=("fp8", "bf16-d128", "bf16-d192", "bf16-d128-lse"))
def test_logits_cross_exact_threshold(backend, dq, lse):
    """Tiles 0-1 keep |m| < 2^22 (FMA centering); tile 2 raises m to ~2^28 (FP8) or ~2^36
    (BF16), where only the exact rule keeps P(max) = 1 (FP8 would pass 240, BF16 2^128)."""
    require("gfx942")
    q_value, k_value, descale = (128.0, 120.0, 41.25) if backend.fp8 else (65536.0, 66048.0, 1.0)
    case = _constant_case(backend, dq, q_value, k_value, descale, small_k=1.0)
    for causal in (False, True):
        call, out, _ = make_call(case, backend, causal)
        call(lse=torch.empty(case.q.shape[:2], device="cuda") if lse else None)
        torch.testing.assert_close(out.float(), torch.ones_like(out, dtype=torch.float32), rtol=0, atol=0.01)


@pytest.mark.parametrize("backend", (BF16_942, FP8, SWA, BF16_950), ids=lambda b: b.name)
def test_softmax_scale_validation(backend):
    """Host scalars only; softmax_scale * log2(e) must be a positive finite FP32."""
    require(*(("gfx942", "gfx950") if backend.arch == "both" else (backend.arch,)))
    case = make_case((7,), (65,), dtype=backend.dtype, dq=128, heads=2, kv_heads=1, poison_tail=False,
                     window_left=16 if backend == SWA else -1, source_dtype=torch.bfloat16)
    call, _, _ = make_call(case, backend, backend == SWA)
    for scale in (torch.tensor(0.1), True):
        with pytest.raises(TypeError, match="softmax_scale"):
            call(softmax_scale=scale)
    for scale in (0.0, -1.0, float("nan"), float("inf"), 1e-46, 3e38):
        with pytest.raises(ValueError, match="softmax_scale"):
            call(softmax_scale=scale)
