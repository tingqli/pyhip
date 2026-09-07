"""Unified MHA contracts; one input/oracle/assertion layer for all backends.

Coverage origins: FP8 native 85-case suite and quantified outer 942 cases;
origin/main BF16's 3 cases; gfx950 page/stride/metadata/merge/persistent/sink
contracts. SWA-specific window matrices live in test_mha_pa_swa.py.
"""

import inspect
import itertools
import math
from itertools import accumulate

import pytest
import torch

if __package__:
    from ._testing import (BACKENDS, PRIMARY_BACKENDS, FP8, FP8_REG, BF16_942, BF16_950,
        BF16_950_PERSISTENT, make_case, make_call, assert_case, assert_close,
        torch_reference, i32, dispatch_names, vectorize_kv, output_buffer)
else:
    from _testing import (BACKENDS, PRIMARY_BACKENDS, FP8, FP8_REG, BF16_942, BF16_950,
        BF16_950_PERSISTENT, make_case, make_call, assert_case, assert_close,
        torch_reference, i32, dispatch_names, vectorize_kv, output_buffer)


PAGE_TAILS = (1, 63, 64, 65, 79, 95, 127, 128, 129, 192, 193, 255, 256, 257, 320, 321)
DQS = (128, 192)


@pytest.fixture(params=BACKENDS, ids=lambda backend: backend.name)
def backend(request):
    selected = request.param
    if not selected.available:
        pytest.skip(f"{selected.name} requires native {selected.arch}")
    return selected


def _case(backend, *args, **kwargs):
    # The origin/main BF16 kernel was tested with zero padding, not poisoned
    # unused V tokens; the native FP8/950 paths explicitly support NaN tails.
    kwargs.setdefault("poison_tail", backend != BF16_942)
    return make_case(*args, dtype=backend.dtype, **kwargs)


def _require(backend, condition, reason):
    if not condition:
        pytest.skip(f"{backend.name}: {reason}")


@pytest.mark.parametrize("dq", DQS)
@pytest.mark.parametrize("causal", (False, True))
@pytest.mark.parametrize("kv", PAGE_TAILS)
def test_page_parity_and_tails(backend, dq, causal, kv):
    q = 257 if backend.causal_short_kv or not causal else min(257, kv)
    assert_case(_case(backend, (q,), (kv,), dq=dq), backend, causal)


@pytest.mark.parametrize("dq,page,q,kv,causal", ((128, 32, 9, 9, False), (192, 32, 9, 9, False), (192, 64, 1024, 1024, True)))
@pytest.mark.parametrize("with_lse", (False, True))
def test_original_bf16_cases(backend, dq, page, q, kv, causal, with_lse):
    _require(backend, backend == BF16_942, "origin/main BF16 specialization")
    assert_case(_case(backend, (q,), (kv,), dq=dq, page=page, heads=16), backend, causal, with_lse=with_lse)


@pytest.mark.parametrize("q,kv,causal", (
    (37, 64, False), (129, 192, False), (65, 128, True), (256, 256, True),
    *((65, kv, False) for kv in (65, 79, 95, 127)),
))
def test_original_quantized_fp8(backend, q, kv, causal):
    _require(backend, backend.fp8, "FP8 quantized per-token/per-tensor inputs")
    assert_case(_case(backend, (q,), (kv,), heads=2, quantized=True, reverse_pages=True), backend, causal, with_lse=False)


@pytest.mark.parametrize("causal", (False, True))
@pytest.mark.parametrize("layout", ("contiguous", "padded", "head-major"))
def test_ragged_gqa_offsets_and_scales(backend, causal, layout):
    _require(backend, layout == "contiguous" or backend.strided, "contiguous Q/O contract")
    q, kv = (0, 7, 129, 513), (63, 0, 193, 321)
    if backend == BF16_942:
        q, kv = (0, 7, 129, 259), (63, 64, 193, 321)
    elif backend.arch == "gfx950":
        q = (0, 7, 129, 259)
    case = _case(backend, q, kv, heads=6, kv_heads=2, layout=layout, q_offset=5, table_offset=3,
                 nonunit_scales=True, magnitude=4.0 if backend.fp8 else 1.0)
    assert_case(case, backend, causal, layout=layout, lse_atol=0.002 if backend.fp8 else None)


@pytest.mark.parametrize("dq", DQS)
@pytest.mark.parametrize("mode", ("per-token", "per-tensor"))
@pytest.mark.parametrize("causal", (False, True))
def test_descales_large_logits_and_softmax_scale(backend, dq, mode, causal):
    case = _case(backend, (129,), (321,), dq=dq, heads=4, kv_heads=2, mode=mode,
                 magnitude=4.0, nonunit_scales=True)
    assert_case(case, backend, causal, softmax_scale=0.0625)


@pytest.mark.parametrize("dq", DQS)
def test_exact_zero_logits_value_and_lse(backend, dq):
    case = _case(backend, (512,), (256,), dq=dq)
    case.q.zero_()
    case.v_pages.fill_(1)
    case.pack(copy=True)
    out, lse = assert_case(case, backend, False)
    torch.testing.assert_close(out, torch.ones_like(out), rtol=0, atol=0)
    torch.testing.assert_close(lse, torch.full_like(lse, math.log(256)), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("dq", DQS)
def test_exact_output_pattern_and_guards(backend, dq):
    case = _case(backend, (0, 17, 259), (64, 65, 193), dq=dq, heads=6, kv_heads=2, q_offset=3, table_offset=2)
    case.q.zero_()
    pattern = (((torch.arange(128, device="cuda") % 16) - 8).float()[None, None, None, :]
               + torch.arange(2, device="cuda")[None, None, :, None] * 16)
    case.v_pages.copy_(pattern.expand_as(case.v_pages).to(backend.dtype))
    case.pack(copy=True)
    out, _ = assert_case(case, backend, False, with_lse=False)
    expected = pattern[0, 0].to(backend.dtype).float().repeat_interleave(3, 0).to(torch.bfloat16)
    begin, end = case.q_offset, case.q_offset + sum(case.q_lens)
    torch.testing.assert_close(out[begin:end], expected[None].expand(end - begin, -1, -1), rtol=0, atol=0)


@pytest.mark.parametrize("dq", DQS)
def test_fp8_bf16_rounding_and_lazy_range(backend, dq):
    _require(backend, backend.fp8, "FNUZ numerical contract")
    case = _case(backend, (257,), (64,), dq=dq, heads=2)
    case.q.zero_()
    case.v.fill_(1)
    call, out, _ = make_call(case, backend, False)
    for value in (1.00390625, 1.01171875, -1.00390625, -1.01171875):
        case.vs.fill_(value)
        torch.testing.assert_close(call(), torch.full_like(out, value, dtype=torch.float32).bfloat16(), rtol=0, atol=0)
    case = _case(backend, (512,), (320,), dq=dq)
    case.q.fill_(1)
    keys = torch.ones_like(case.k_pages, dtype=torch.float32)
    keys[case.page_order[0]] = 0
    case.k_pages = keys.to(backend.dtype)
    case.v_pages.fill_(1)
    case.pack(copy=True)
    for log_max in (6.0, 7.0, 8.0):
        case.ks.fill_(log_max / (math.sqrt(dq) * math.log2(math.e)))
        actual, _ = assert_case(case, backend, False)
        torch.testing.assert_close(actual, torch.ones_like(actual), rtol=0, atol=0)


@pytest.mark.parametrize("dq,causal", ((128, False), (192, False), (192, True)))
def test_fp8_memory_modes_are_bit_exact(backend, dq, causal):
    _require(backend, backend == FP8, "compare the two native FP8 memory paths once")
    case = _case(backend, (1025,), (777,), dq=dq, heads=6, kv_heads=2, nonunit_scales=True)
    a = make_call(case, FP8, causal)[0]()
    b = make_call(case, FP8_REG, causal)[0]()
    torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("causal", (False, True))
@pytest.mark.parametrize("kv", (2560, 2583))
def test_target_shape_and_repeated_no_lse(backend, causal, kv):
    q = 10240 if backend.causal_short_kv or not causal else kv
    assert_case(_case(backend, (q,), (kv,), heads=16), backend, causal, with_lse=False, repeats=5)


@pytest.mark.parametrize("dq", DQS)
def test_live_page_and_cache_mutations(backend, dq):
    case = _case(backend, (17, 23), (129, 256), dq=dq, heads=6, kv_heads=2, table_offset=2, poison_tail=False)
    lse = torch.empty(case.q.shape[:2], device="cuda")
    call, out, kernel = make_call(case, backend, False, lse=lse)
    count = None
    for mutation in range(3):
        if mutation == 1:
            case.page_order[2:] = list(reversed(case.page_order[2:]))
            case.indices.copy_(i32(case.page_order))
        elif mutation == 2:
            case.k_pages.copy_((case.k_pages.float() * 2).to(backend.dtype))
            case.v_pages.copy_((case.v_pages.float() * 0.5).to(backend.dtype))
            case.pack(copy=True)
        call()
        assert_close(case, backend, out, lse, False)
        if count is not None:
            assert len(kernel._compiled) == count
        count = len(kernel._compiled)


@pytest.mark.parametrize("dq", DQS)
@pytest.mark.parametrize("causal", (False, True))
def test_shared_physical_pages(backend, dq, causal):
    case = _case(backend, (9, 33), (129, 257), dq=dq, heads=6, kv_heads=2, q_offset=3, table_offset=2,
                 nonunit_scales=True, poison_tail=False)
    case.k_pages, case.v_pages = case.k_pages[:3], case.v_pages[:3]
    case.page_order[2:] = [2, 0, 2, 1, 2, 0, 1, 0]
    case.indices.copy_(i32(case.page_order))
    case.pack()
    assert_case(case, backend, causal)


@pytest.mark.parametrize("dq", DQS)
@pytest.mark.parametrize("causal", (False, True))
@pytest.mark.parametrize("with_lse", (False, True))
def test_runtime_lengths_poisoned_tail(backend, dq, causal, with_lse):
    q = 7937 if causal and backend.arch == "gfx950" else 257
    if causal and not backend.causal_short_kv:
        q = 1
    case = _case(backend, (q,), (321,), dq=dq, heads=16 if q == 7937 else 4, poison_tail=False)
    lse = torch.empty(case.q.shape[:2], device="cuda") if with_lse else None
    call, out, kernel = make_call(case, backend, causal, lse=lse)
    keys, values, count = case.k_pages.clone(), case.v_pages.clone(), None
    for length in (1, 64, 65, 128, 193, 321, 320):
        n = (length + 63) // 64
        case.kv_lens = (length,)
        case.indptr.copy_(i32([0, n]))
        case.last.fill_((length - 1) % 64 + 1)
        case.k_pages.copy_(keys)
        case.v_pages.copy_(values)
        if length % 64:
            physical = case.page_order[n - 1]
            fill = 0 if backend == BF16_942 else float("nan")
            case.k_pages[physical, length % 64:] = fill
            case.v_pages[physical, length % 64:] = fill
        case.pack(copy=True)
        call(max_seqlen_q=q, max_seqlen_k=321)
        first = out.clone()
        for _ in range(2):
            call(max_seqlen_q=q, max_seqlen_k=321)
            torch.testing.assert_close(out, first, rtol=0, atol=0)
        assert_close(case, backend, out, lse, causal)
        if count is not None:
            assert len(kernel._compiled) == count
        count = len(kernel._compiled)


def test_empty_requests(backend):
    assert_case(_case(backend, (0,), (0,), heads=2), backend, False)
    if backend.empty_kv:
        assert_case(_case(backend, (65,), (0,), heads=2), backend, False)
    else:
        case = _case(backend, (65,), (0,), heads=2)
        with pytest.raises(NotImplementedError, match="empty KV"):
            make_call(case, backend, False)[0]()


def _check_stream_graphs(case, backend, *, causal=True, replays=4, calls_per_graph=1):
    call, _, kernel = make_call(case, backend, causal)
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    captured = []
    for stream in streams:
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            out, lse = call(out=None, return_lse=True, stream=stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                for _ in range(calls_per_graph):
                    call(out=out, lse=lse, stream=stream)
            captured.append((stream, graph, out, lse))
    for _ in range(replays):
        for stream, graph, _, _ in captured:
            with torch.cuda.stream(stream):
                graph.replay()
    for stream, _, out, lse in captured:
        torch.cuda.current_stream().wait_stream(stream)
        assert_close(case, backend, out, lse, causal)
    if backend == BF16_950_PERSISTENT:
        stream_ids = {stream.cuda_stream for stream in streams}
        active = [counter for (_, sid, _), counter in kernel._scheduler_counters.items() if sid in stream_ids]
        assert len(active) == 2 and active[0].data_ptr() != active[1].data_ptr()
        for (_, _, grid), counter in kernel._scheduler_counters.items():
            assert counter.tolist() == [grid, 0]


@pytest.mark.parametrize("dq", DQS)
def test_stream_graph_and_allocation(backend, dq):
    _check_stream_graphs(_case(backend, (127,), (193,), dq=dq), backend)


@pytest.mark.parametrize("window", (-1, 128))
def test_warmed_dispatch_contract(backend, monkeypatch, window):
    _require(backend, window < 0 or backend.arch == "gfx950", "gfx950 retained SWA extension")
    case = _case(backend, (257,), (901,), heads=16, window_left=window, has_sink=window >= 0)
    call, _, kernel = make_call(case, backend, window >= 0)
    call()
    if backend.single_dispatch:
        def forbidden(*args, **kwargs):
            pytest.fail("warmed call allocated a tensor")
        with monkeypatch.context() as patch:
            for name in ("empty", "empty_like", "zeros", "tensor"):
                patch.setattr(torch, name, forbidden)
            call()
            torch.cuda.synchronize()
        events = dispatch_names(call)
        assert len(events) == 1 and "attention" in events[0]
    else:
        # Original prefill allocates/seeds its per-call persistent counter.
        events = dispatch_names(call)
        assert sum("attn_kernel" in name for name in events) == 1
    assert not hasattr(kernel, "prepare_kv")


@pytest.mark.parametrize("q", (7937, 8193))
@pytest.mark.parametrize("dq,sink", ((192, False), (128, False), (128, True), (192, True)))
def test_gfx950_causal_merge(backend, q, dq, sink):
    _require(backend, backend.arch == "gfx950", "gfx950 paired-head/tail scheduler")
    assert_case(_case(backend, (q,), (q + 7,), dq=dq, heads=16, has_sink=sink), backend, True)


@pytest.mark.parametrize("dq", DQS)
@pytest.mark.parametrize("causal", (False, True))
def test_gfx950_full_sink(backend, dq, causal):
    _require(backend, backend.arch == "gfx950", "full-attention sink extension")
    case = _case(backend, (33, 259), (0, 65), dq=dq, has_sink=True, q_offset=5, table_offset=2)
    for sink in (-80.0, 0.0, 80.0, -float("inf")):
        case.sinks.fill_(sink)
        assert_case(case, backend, causal)


@pytest.mark.parametrize("dq", DQS)
@pytest.mark.parametrize("causal,window", ((False, -1), (True, -1), (True, 128)))
def test_persistent_ticket_reuse(backend, dq, causal, window, monkeypatch):
    _require(backend, backend == BF16_950_PERSISTENT, "gfx950 persistent scheduler")
    case = _case(backend, (12289,), (901,), dq=dq, heads=16, window_left=window, has_sink=window >= 0)
    call, out, kernel = make_call(case, backend, causal)
    static = make_call(case, BF16_950, causal)[0]().clone()
    for _ in range(12):
        torch.testing.assert_close(call(), static, rtol=0, atol=0)
    for (_, _, grid), counter in kernel._scheduler_counters.items():
        assert counter.tolist() == [grid, 0]
    def forbidden(*args, **kwargs):
        pytest.fail("persistent warmed dispatch allocated/reset a tensor")
    with monkeypatch.context() as patch:
        for name in ("empty", "empty_like", "zeros", "tensor"):
            patch.setattr(torch, name, forbidden)
        call()
    assert len(dispatch_names(call)) == 1
    assert_close(case, backend, out, None, causal)


@pytest.mark.parametrize("dq", DQS)
def test_persistent_runtime_query_mapping(backend, dq):
    _require(backend, backend == BF16_950_PERSISTENT, "gfx950 persistent scheduler")
    case = _case(backend, (257, 513, 257), (129, 321, 193), dq=dq, heads=6, kv_heads=2,
                 q_offset=3, table_offset=2, poison_tail=False)
    lse = torch.full(case.q.shape[:2], -123.0, device="cuda", dtype=torch.float32)
    call, out, kernel = make_call(case, backend, False, lse=lse)
    count = None
    for lengths in ((257, 513, 257), (0, 1, 1026), (1027, 0, 0), (0, 0, 0), (513, 257, 257)):
        case.q_lens = lengths
        case.cq.copy_(i32(list(accumulate(lengths, initial=3))))
        out.fill_(-123)
        lse.fill_(-123)
        call(max_seqlen_q=1027, max_seqlen_k=321)
        assert_close(case, backend, out, lse, False)
        assert (out[:3] == -123).all() and (out[3 + sum(lengths):] == -123).all()
        if count is not None:
            assert len(kernel._compiled) == count
        count = len(kernel._compiled)
        for (_, _, grid), counter in kernel._scheduler_counters.items():
            assert counter.tolist() == [grid, 0]


@pytest.mark.parametrize("dq", DQS)
@pytest.mark.parametrize("dv", DQS)
@pytest.mark.parametrize("page", (32, 64, 128))
@pytest.mark.parametrize("mode", ("per-token", "per-tensor"))
@pytest.mark.parametrize("causal", (False, True))
def test_bf16_942_original_extended_pages(backend, dq, dv, page, mode, causal):
    _require(backend, backend == BF16_942, "origin/main page32/64/128 BF16 parameterization")
    case = _case(backend, (129,), (321,), dq=dq, dv=dv, page=page, mode=mode)
    assert_case(case, backend, causal)


# Architecture-specific contracts are parameterized only over their actual
# backends. These exercise the retained 8-wave code, not the one-wave SWA.
@pytest.mark.parametrize("backend", (BF16_950, BF16_950_PERSISTENT), indirect=True, ids=lambda b: b.name)
class TestGfx950Extensions:
    @pytest.mark.parametrize("dq", DQS)
    @pytest.mark.parametrize("kv", (256, 321))
    @pytest.mark.parametrize("window,sink", ((-1, False), (-1, True), (128, False), (128, True)))
    def test_default_no_lse(self, backend, dq, kv, window, sink):
        case = _case(backend, (257,), (kv,), dq=dq, heads=16, window_left=window, has_sink=sink)
        assert_case(case, backend, window >= 0, with_lse=False, repeats=11)

    @pytest.mark.parametrize("q,kv,causal,sink", ((256, 256, False, False), (10240, 2583, False, False),
                                               (129, 321, False, True), (129, 321, True, True)))
    def test_original_dispatch_and_sink_logits(self, backend, q, kv, causal, sink):
        case = _case(backend, (q,), (kv,), heads=4 if sink else 16, kv_heads=2 if sink else 1,
                     has_sink=sink, magnitude=4.0 if sink else 1.0)
        assert_case(case, backend, causal, repeats=10 if q == 256 else 3)

    def test_merge_ragged_empty_rows(self, backend):
        case = _case(backend, (33, 4097, 65), (0, 193, 129), heads=16, q_offset=3, table_offset=2)
        assert_case(case, backend, True)

    @pytest.mark.parametrize("dq", DQS)
    @pytest.mark.parametrize("window", (0, 1, 63, 64, 65, 127, 128, 129, 512))
    @pytest.mark.parametrize("sink", (False, True))
    def test_window_boundaries(self, backend, dq, window, sink):
        case = _case(backend, (257,), (777,), dq=dq, window_left=window, has_sink=sink)
        assert_case(case, backend, True)

    @pytest.mark.parametrize("dq", DQS)
    @pytest.mark.parametrize("window", (-1, 0, 128))
    @pytest.mark.parametrize("sink", (-80.0, 0.0, 80.0))
    def test_sink_denominator(self, backend, dq, window, sink):
        case = _case(backend, (33, 259), (0, 65), dq=dq, q_offset=5, table_offset=2,
                     window_left=window, has_sink=True, nonunit_scales=True)
        case.sinks.fill_(sink)
        assert_case(case, backend, True)

    @pytest.mark.parametrize("dq", DQS)
    @pytest.mark.parametrize("layout", ("padded", "head-major"))
    @pytest.mark.parametrize("causal,window,sink", ((False, -1, False), (True, -1, True),
                                                  (True, 0, False), (True, 128, True)))
    def test_ragged_scheduler_modes(self, backend, dq, layout, causal, window, sink):
        case = _case(backend, (0, 7, 1025, 0, 513), (63, 0, 129, 0, 901), dq=dq,
                     heads=6, kv_heads=2, q_offset=5, table_offset=3, layout=layout,
                     window_left=window, has_sink=sink, nonunit_scales=True)
        assert_case(case, backend, causal, layout=layout, repeats=5, softmax_scale=0.0625)
        if window == 128:
            legacy = _case(backend, (0, 7, 129, 259), (63, 0, 193, 901), dq=dq,
                           heads=6, kv_heads=2, q_offset=5, table_offset=3, layout=layout,
                           window_left=128, has_sink=True, nonunit_scales=True)
            assert_case(legacy, backend, True, layout=layout)

    @pytest.mark.parametrize("dq,window,q,kv,exact", ((128, 128, 257, 8193, False),
        (192, 128, 257, 8193, False), (128, 1, 16384, 32769, True), (192, 128, 16384, 32769, True)))
    @pytest.mark.parametrize("sink", (False, True))
    def test_excluded_prefix_and_pruned_grid(self, backend, dq, window, q, kv, exact, sink):
        case = _case(backend, (q,), (kv,), dq=dq, heads=16 if exact else 4,
                     window_left=window, has_sink=sink)
        if exact:
            case.q.zero_()
            case.k_pages.zero_()
            case.v_pages.fill_(1)
            case.pack(copy=True)
            if sink:
                case.sinks.zero_()
        case.indices[:(kv - q - window) // 64] = 2**30
        out, lse = assert_case(case, backend, True, repeats=3 if exact else 10)
        if exact:
            denominator = window + 1 + int(sink)
            torch.testing.assert_close(out, torch.full_like(out, (window + 1) / denominator), rtol=0, atol=0)
            torch.testing.assert_close(lse, torch.full_like(lse, math.log(denominator)), rtol=1e-6, atol=1e-6)

    @pytest.mark.parametrize("dq,window", ((128, 64), (128, 65), (192, 128), (192, 129)))
    @pytest.mark.parametrize("q", (16128, 16129, 16385))
    def test_pruning_thresholds(self, backend, dq, window, q):
        case = _case(backend, (q,), (901,), dq=dq, heads=16, window_left=window,
                     has_sink=True, nonunit_scales=True, magnitude=4.0)
        assert_case(case, backend, True, softmax_scale=0.0625)

    @pytest.mark.parametrize("dq", DQS)
    def test_runtime_lengths_and_sinks(self, backend, dq):
        case = _case(backend, (65,), (2049,), dq=dq, mode="per-tensor", window_left=128,
                     has_sink=True, nonunit_scales=True, poison_tail=False)
        lse = torch.empty(case.q.shape[:2], device="cuda")
        call, out, kernel = make_call(case, backend, True, lse=lse)
        compiled = None
        for length, sink in ((2049, 0.0), (257, -80.0), (128, 80.0), (0, -float("inf")), (193, 1.0)):
            case.kv_lens = (length,)
            case.indptr.copy_(i32([0, (length + 63) // 64]))
            case.last.fill_((length - 1) % 64 + 1 if length else 0)
            case.sinks.fill_(sink)
            call(max_seqlen_q=65, max_seqlen_k=2049)
            assert_close(case, backend, out, lse, True)
            if compiled is not None:
                assert frozenset(kernel._compiled) == compiled
            compiled = frozenset(kernel._compiled)

    @pytest.mark.parametrize("dq", DQS)
    def test_exact_window_and_sink(self, backend, dq):
        case = _case(backend, (129,), (257,), dq=dq, heads=2, window_left=128, has_sink=True)
        case.q.zero_()
        case.k_pages.zero_()
        case.v_pages.fill_(1)
        case.pack(copy=True)
        for sink in (0.0, -float("inf")):
            case.sinks.fill_(sink)
            out, lse = assert_case(case, backend, True)
            denominator = 130 if sink == 0 else 129
            torch.testing.assert_close(out, torch.full_like(out, 129 / denominator), rtol=0, atol=0)
            torch.testing.assert_close(lse, torch.full_like(lse, math.log(denominator)), rtol=1e-6, atol=1e-6)

    @pytest.mark.parametrize("dq,window,q", ((128, -1, 4097), (192, 128, 4097),
                                           (128, 128, 127), (192, 0, 127)))
    def test_stream_isolation_and_repeated_graphs(self, backend, dq, window, q):
        case = _case(backend, (q,), (777 if q == 4097 else 193,), dq=dq,
                     heads=16 if q == 4097 else 4, window_left=window, has_sink=window == 128)
        _check_stream_graphs(case, backend, replays=8, calls_per_graph=2)

    def test_sink_buffer_contract(self, backend):
        case = _case(backend, (9,), (65,), dq=128, window_left=128, has_sink=True)
        call, _, _ = make_call(case, backend, True)
        for invalid in (None, torch.zeros(3, device="cuda"), torch.zeros(4, device="cuda", dtype=torch.bfloat16),
                        torch.zeros(8, device="cuda")[::2], torch.zeros(4)):
            with pytest.raises(ValueError, match="sink_ptr"):
                call(sink_ptr=invalid)
        case.sinks = None
        with pytest.raises(ValueError, match="has_sink"):
            make_call(case, backend, True)[0](sink_ptr=torch.zeros(4, device="cuda"))

    @pytest.mark.parametrize("dq", DQS)
    def test_aiter_window_reference(self, backend, dq):
        if __package__:
            from ._references import aiter_call, probe_reference, ReferenceUnavailable
        else:
            from _references import aiter_call, probe_reference, ReferenceUnavailable
        case = _case(backend, (257,), (777,), dq=dq, heads=16, window_left=128, has_sink=True, poison_tail=False)
        assert_case(case, backend, True)
        try:
            actual = probe_reference(aiter_call(case, True))
        except ReferenceUnavailable as exc:
            pytest.skip(str(exc))
        assert_close(case, backend, actual, None, True)


def test_gfx950_factory_contract():
    factory = BF16_950.load().PagedAttention
    assert not factory(16, 1, 192, 128, 64, False).persistent
    assert factory(16, 1, 192, 128, 64, False, persistent=True).persistent
    for options in ({"persistent": "yes"}, {"window_left": 128}, {"window_left": -2}):
        with pytest.raises(ValueError):
            factory(16, 1, 192, 128, 64, False, **options)
    with pytest.raises(NotImplementedError):
        factory(16, 1, 192, 128, 32, False)


@pytest.mark.parametrize("backend", (BF16_950,), indirect=True, ids=lambda b: b.name)
@pytest.mark.parametrize("dq", DQS)
@pytest.mark.parametrize("causal", (False, True))
def test_explicit_opus_comparison(backend, dq, causal):
    if __package__:
        from ._references import aiter_opus_call, probe_reference, ReferenceUnavailable
    else:
        from _references import aiter_opus_call, probe_reference, ReferenceUnavailable
    case = _case(backend, (257,), (777,), dq=dq, heads=16, poison_tail=False)
    try:
        call = aiter_opus_call(case, causal)
        actual = probe_reference(call)
    except ReferenceUnavailable as exc:
        pytest.skip(str(exc))
    assert_close(case, backend, actual, None, causal)
    assert any(f"gqa_d{dq}" in name for name in dispatch_names(call))


@pytest.mark.parametrize("window,sink", ((128, False), (-1, True), (128, True)))
def test_opus_rejects_unsupported_semantics(window, sink):
    from types import SimpleNamespace
    if __package__:
        from ._references import aiter_opus_call
    else:
        from _references import aiter_opus_call
    with pytest.raises(ValueError, match="no SWA or sink"):
        aiter_opus_call(SimpleNamespace(window_left=window, sinks=object() if sink else None), True)


def test_invalid_runtime_buffers(backend):
    case = _case(backend, (9,), (65,))
    call, _, _ = make_call(case, backend, False)
    for kwargs in ({"out": torch.empty(9, 4, 128, device="cuda", dtype=torch.float32)},
                   {"lse": torch.empty(9, 4, device="cuda", dtype=torch.bfloat16)}, {"softmax_scale": 0}):
        with pytest.raises(ValueError):
            call(**kwargs)
    case.q = case.q.float()
    with pytest.raises(NotImplementedError):
        call()


def test_uniform_public_signatures():
    factories = [backend.load().PagedAttention for backend in PRIMARY_BACKENDS]
    assert len({str(inspect.signature(factory)) for factory in factories}) == 1
    calls = [factory(4, 1, 192, 128, 64, False).__call__ for factory in factories]
    assert len({str(inspect.signature(call)) for call in calls}) == 1


@pytest.mark.parametrize("selected", PRIMARY_BACKENDS, ids=lambda b: b.name)
def test_public_scope(selected):
    factory = selected.load().PagedAttention
    for shape in ((4, 1, 64, 128, 64), (7, 2, 192, 128, 64)):
        with pytest.raises((ValueError, NotImplementedError)):
            factory(*shape, False)
    with pytest.raises(NotImplementedError):
        factory(4, 1, 192, 128, 64, False, key_layout="linear")
    with pytest.raises((ValueError, NotImplementedError)):
        factory(4, 1, 192, 128, 64, False, memory_mode="unsupported")
    if selected != BF16_950:
        with pytest.raises(NotImplementedError):
            factory(4, 1, 192, 128, 64, True, has_sink=True)


@pytest.mark.parametrize("dq", DQS)
@pytest.mark.parametrize("causal", (False, True))
@pytest.mark.parametrize("reference_kind", ("paged", "ck_linear"))
def test_aiter_reference(backend, dq, causal, reference_kind):
    if __package__:
        from ._references import aiter_call, aiter_linear_call, probe_reference, ReferenceUnavailable
    else:
        from _references import aiter_call, aiter_linear_call, probe_reference, ReferenceUnavailable
    case = _case(backend, (257,), (777,), dq=dq, heads=16, poison_tail=False,
                 mode="per-tensor" if backend.fp8 else "per-token")
    actual, _ = assert_case(case, backend, causal, with_lse=False)
    try:
        factory = aiter_call if reference_kind == "paged" else aiter_linear_call
        reference = probe_reference(factory(case, causal))
    except ReferenceUnavailable as exc:
        pytest.skip(str(exc))
    expected, _ = torch_reference(case, causal)
    tolerance = 0.1 if backend.fp8 else 0.02
    torch.testing.assert_close(reference.float(), expected, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(actual.float(), reference.float(), rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("backend", (FP8, FP8_REG), indirect=True, ids=lambda b: b.name)
def test_fp8_scalar_scale_runtime_aliases_and_graph(backend):
    case = _case(backend, (513,), (320,), dq=192, heads=6, kv_heads=2, poison_tail=False)
    case.qs = torch.full((1,), 0.75, device="cuda")
    lse = torch.empty(case.q.shape[:2], device="cuda")
    call, out, _ = make_call(case, backend, True, lse=lse)
    for length, ids in ((65, [1, 1]), (128, [3, 0]), (193, [4, 1, 4, 2]), (320, [2, 0, 3, 1, 4])):
        case.kv_lens, case.page_order = (length,), ids
        case.indices[:len(ids)].copy_(i32(ids))
        case.indptr.copy_(i32([0, len(ids)]))
        case.last.fill_((length - 1) % 64 + 1)
        call()
        assert_close(case, backend, out, lse, True)
    _check_stream_graphs(case, backend)


def test_runner_result_paths_and_register_accounting(tmp_path):
    from types import SimpleNamespace
    if __package__:
        from ._runner import result_path, resource_fields, profile_round
    else:
        from _runner import result_path, resource_fields, profile_round
    args = SimpleNamespace(output=tmp_path / "run.json", mode="all")
    assert result_path(args, "resources") != result_path(args, "performance")
    args.mode = "audit"
    assert result_path(args, "resources") == args.output
    text = "\n".join(f".{name}: {value}" for name, value in {
        "group_segment_fixed_size": 0, "private_segment_fixed_size": 0,
        "vgpr_count": 387, "agpr_count": 131, "sgpr_count": 36,
        "vgpr_spill_count": 0, "sgpr_spill_count": 0}.items())
    fields = resource_fields(text + "\n.amdhsa_accum_offset 256")
    assert fields["vector_register_count"] == fields["accum_offset"] == 256
    assert fields["agpr_count"] == 131
    with pytest.raises(ValueError, match="iterations"):
        profile_round(lambda: pytest.fail("must reject before launch"), iterations=1)


@pytest.mark.parametrize("error", (RuntimeError, AssertionError, ValueError))
def test_reference_probe_does_not_swallow_errors(error):
    if __package__:
        from ._references import probe_reference, ReferenceUnavailable
    else:
        from _references import probe_reference, ReferenceUnavailable
    def missing():
        raise RuntimeError("no matching kernel found")
    def broken():
        raise error("unexpected kernel failure")
    with pytest.raises(ReferenceUnavailable):
        probe_reference(missing)
    with pytest.raises(error, match="unexpected kernel failure"):
        probe_reference(broken)


@pytest.mark.parametrize("error", (ImportError, OSError))
def test_reference_probe_reports_lazy_dependency_failure(error):
    if __package__:
        from ._references import probe_reference, ReferenceUnavailable
    else:
        from _references import probe_reference, ReferenceUnavailable
    def missing_dependency():
        raise error("optional native module unavailable")
    with pytest.raises(ReferenceUnavailable, match="optional native module unavailable"):
        probe_reference(missing_dependency)


@pytest.mark.parametrize("dq", DQS)
def test_fp8_packed_address_invariants(dq):
    key, value = {}, {}
    for tid in range(512):
        for round_id in range(dq // 64):
            byte = (tid + round_id * 512) * 8
            chunk, within = divmod(byte, 1024)
            address = chunk * 1040 + (within ^ ((within & 256) >> 2))
            for i in range(8):
                assert address + i not in key
                key[address + i] = byte + i
        for round_id in range(2):
            byte = (tid + round_id * 512) * 8
            chunk, within = divmod(byte, 1024)
            for i in range(8):
                value[chunk * 1040 + within + i] = byte + i
    assert len(key) == 64 * dq and len(value) == 64 * 128
    for lane in range(64):
        half = lane // 32
        row = (lane & 3) | ((lane & 4) << 2) | ((lane & 24) >> 1)
        for sub, d in itertools.product(range(2), range(dq // 32)):
            lds = half * 1040 + ((row * 16) ^ ((row & 16) << 2)) + sub * 512 + d * 2080
            src = half * 1024 + row * 16 + sub * 512 + d * 2048
            assert [key[lds + i] for i in range(16)] == list(range(src, src + 16))
        for sub, n, k in itertools.product(range(2), repeat=3):
            lds = half * 2080 + (lane & 31) * 16 + sub * 1040 + n * 512 + k * 4160
            src = half * 2048 + (lane & 31) * 16 + sub * 1024 + n * 512 + k * 4096
            assert [value[lds + i] for i in range(16)] == list(range(src, src + 16))
        for i in range(16):
            col = half * 4 + (i // 4) * 8 + i % 4
            assert ((col & 3) | ((col & 4) << 2) | ((col & 24) >> 1)) == half * 16 + i
    for a, b in ((0, 20), (4, 16), (8, 28), (12, 24), (32, 52), (36, 48), (40, 60), (44, 56)):
        banks = []
        for lane in (*range(a, a + 4), *range(b, b + 4)):
            row = (lane & 3) | ((lane & 4) << 2) | ((lane & 24) >> 1)
            byte = lane // 32 * 1040 + ((row * 16) ^ ((row & 16) << 2))
            banks.extend((byte // 4 + i) % 32 for i in range(4))
        assert len(set(banks)) == 32


def test_cshuffle_identity_and_banks():
    memory = {}
    for tid, n, group in itertools.product(range(512), range(2), range(4)):
        row = (tid >> 6) * 32 + (tid & 31)
        col = n * 32 + group * 8 + ((tid >> 5) & 1) * 4
        address = (row * 64 + col) ^ ((row & 15) * 4)
        for i in range(4):
            assert address + i not in memory
            memory[address + i] = row * 64 + col + i
    assert len(memory) == 256 * 64
    for tid, block in itertools.product(range(512), range(4)):
        element = tid * 8 + block * 512 * 8
        row = element // 64
        address = element ^ ((row & 14) * 4)
        for word, value in itertools.product(range(4), range(2)):
            physical = word ^ (2 if row & 1 else 0)
            assert memory[address + physical * 2 + value] == element + word * 2 + value
    for first in range(0, 64, 16):
        banks = []
        for lane in range(first, first + 16):
            address = ((lane & 31) * 64 + (lane >> 5) * 4) ^ ((lane & 15) * 4)
            banks.extend((address // 2 + i) % 32 for i in range(2))
        assert len(set(banks)) == 32


if __name__ == "__main__":
    if __package__:
        from ._runner import main
    else:
        from _runner import main
    main(__file__, suite="pa")