"""Native one-wave BF16 paged SWA contracts for gfx942 and gfx950.

Keep CPU contracts independent of GPU availability. GPU cases select the same
SWA backend on each architecture; neither legacy tests nor multi-wave kernels
are imported. Performance shapes also remain strict functional tests. Timing
and emitted-ISA/resource checks belong to the shared performance/audit runner.
"""

import contextlib
import itertools
import json
import math
import statistics
from itertools import accumulate

import pytest
import torch

if __package__:
    from ._testing import (
        SWA, Case, assert_case, assert_close, dispatch_names, gpu_arch, i32,
        make_call, make_case, torch_reference, vectorize_kv,
    )
else:
    from _testing import (
        SWA, Case, assert_case, assert_close, dispatch_names, gpu_arch, i32,
        make_call, make_case, torch_reference, vectorize_kv,
    )


ARCHES = ("gfx942", "gfx950")
DQS = (128, 192)
BLOCK_NS = (16, 32, 64)
QUERY_TILES = (16, 32)
TILE_CONFIGS = tuple(itertools.product(QUERY_TILES, BLOCK_NS))
WINDOWS = (0, 1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 512, 1024)
KV_LENGTHS = (0, 1, 15, 16, 31, 32, 63, 64, 65, 127, 128, 129, 192, 193, 255, 256, 257, 321)
SWA_CASES = (
    pytest.param(33, 129, 128, True, id="partial-pages-sink"),
    pytest.param(257, 777, 128, True, id="many-tiles-sink"),
    pytest.param(17, 1, 0, False, id="all-masked-prefix"),
    pytest.param(129, 193, 1, True, id="one-left-sink"),
    pytest.param(33, 65, 64, False, id="page-boundary"),
)
RUNTIME_LENGTHS_AND_SINKS = (
    (2049, 0.0), (257, -80.0), (128, 80.0),
    (0, -float("inf")), (193, 1.0), (65, 0.0),
)
# Original CLI defaults, including all explicit QT/BN candidates and auto.
PERFORMANCE_CASES = (
    pytest.param(16384, 131072, 128, id="q16384-kv131072-window128"),
)
GATHER_LINEAR_KV_LENGTHS = (32768, 65536, 131072)
GATHER_LINEAR_SOURCE = "23cc6d1e95b1611493e21232bef5d9962b7b73c9:tests/flydsl/pa_4wave/test_pa_prefill.py:1010-1139,1409-1439"


@pytest.fixture(params=ARCHES, ids=ARCHES)
def backend(request):
    arch = gpu_arch()
    if arch != request.param:
        pytest.skip(f"native single-wave SWA requires {request.param}; detected {arch}")
    assert SWA.available and SWA.arch == "both" and SWA.dtype == torch.bfloat16
    return SWA


def _assert_guards(case: Case, *buffers):
    begin, end = case.q_offset, case.q_offset + sum(case.q_lens)
    for buffer in buffers:
        if buffer is not None:
            assert (buffer[:begin] == -123).all(), "query prefix was overwritten"
            assert (buffer[end:] == -123).all(), "query suffix was overwritten"


def _assert_result(case: Case, backend, out, lse=None):
    assert_close(case, backend, out, lse, True)
    _assert_guards(case, out, lse)


def _assert_compiled_keys(kernel, previous):
    current = frozenset(kernel._compiled)
    if previous is not None:
        assert current == previous, "device metadata/content must not specialize the kernel"
    return current


def _references():
    # The shared adapter imports AITER lazily, translating only unavailable
    # imports/built specializations into ReferenceUnavailable.
    if __package__:
        from . import _references as references
    else:
        import _references as references
    return references


# Numerical semantics: preserve the complete original Cartesian products.
@pytest.mark.parametrize("dq", DQS)
@pytest.mark.parametrize("block_n", BLOCK_NS)
@pytest.mark.parametrize("query_tile", QUERY_TILES)
@pytest.mark.parametrize("q_len,kv_len,window,sink", SWA_CASES)
def test_swa(backend, dq, block_n, query_tile, q_len, kv_len, window, sink):
    case = make_case((q_len,), (kv_len,), heads=4, dq=dq, window_left=window, has_sink=sink)
    assert_case(case, backend, True, block_n=block_n, query_tile=query_tile)


@pytest.mark.parametrize("dq", DQS)
@pytest.mark.parametrize("window", WINDOWS)
@pytest.mark.parametrize("sink", (False, True))
def test_window_boundaries(backend, dq, window, sink):
    case = make_case((97,), (393,), heads=4, dq=dq, window_left=window, has_sink=sink)
    assert_case(case, backend, True)


@pytest.mark.parametrize("dq", DQS)
@pytest.mark.parametrize("kv_len", KV_LENGTHS)
def test_poisoned_pages_and_all_masked_rows(backend, dq, kv_len):
    case = make_case((257,), (kv_len,), heads=3, dq=dq, window_left=128, poison_tail=True)
    if kv_len == 0:
        # Even the dummy physical page must not leak into empty-row results.
        case.k_pages.fill_(float("nan"))
        case.v_pages.fill_(float("nan"))
        case.pack(copy=True)
    out, lse = assert_case(case, backend, True)
    masked = max(0, 257 - kv_len)
    assert torch.isfinite(out).all(), "masked/poisoned cache values reached output"
    assert (out[:masked] == 0).all()
    assert torch.isneginf(lse[:masked]).all()


@pytest.mark.parametrize("dq", DQS)
@pytest.mark.parametrize("layout", ("contiguous", "padded", "head-major"))
@pytest.mark.parametrize("window", (1, 128))
def test_ragged_strides_offsets_and_scales(backend, dq, layout, window):
    case = make_case((0, 7, 129, 259), (63, 0, 193, 901), heads=6, kv_heads=2,
                     dq=dq, window_left=window, has_sink=True, nonunit_scales=True,
                     layout=layout, q_offset=5, table_offset=3)
    assert_case(case, backend, True, layout=layout, softmax_scale=0.0625)


@pytest.mark.parametrize("dq", DQS)
@pytest.mark.parametrize("mode", ("per-token", "per-tensor"))
@pytest.mark.parametrize("query_tile", QUERY_TILES)
def test_large_logits_lazy_max_and_descales(backend, dq, mode, query_tile):
    case = make_case((129,), (513,), heads=4, kv_heads=2, dq=dq, mode=mode,
                     nonunit_scales=True, magnitude=4.0, window_left=128, has_sink=True)
    assert_case(case, backend, True, query_tile=query_tile, softmax_scale=0.0625)


@pytest.mark.parametrize("dq", DQS)
@pytest.mark.parametrize("query_tile", QUERY_TILES)
@pytest.mark.parametrize("sink", (-80.0, 0.0, 80.0, -float("inf")))
def test_sink_empty_rows(backend, dq, query_tile, sink):
    case = make_case((33, 259), (0, 65), heads=4, dq=dq, window_left=128,
                     has_sink=True, q_offset=5, table_offset=2, nonunit_scales=True)
    case.sinks.fill_(sink)
    assert_case(case, backend, True, query_tile=query_tile)


@pytest.mark.parametrize("dq", DQS)
@pytest.mark.parametrize("window", (0, 16, 128))
def test_exact_inclusive_window_and_sink(backend, dq, window):
    case = make_case((129,), (257,), heads=2, dq=dq, window_left=window, has_sink=True)
    case.q.zero_()
    case.k_pages.zero_()
    case.v_pages.fill_(1)
    case.sinks.zero_()
    case.k, case.v = vectorize_kv(case.k_pages, case.v_pages)
    out, lse = assert_case(case, backend, True)
    torch.testing.assert_close(out, torch.full_like(out, (window + 1) / (window + 2)), rtol=0, atol=0)
    torch.testing.assert_close(lse, torch.full_like(lse, math.log(window + 2)), rtol=1e-6, atol=1e-6)
    case.sinks.fill_(-float("inf"))
    out, lse = assert_case(case, backend, True)
    torch.testing.assert_close(out, torch.ones_like(out), rtol=0, atol=0)
    torch.testing.assert_close(lse, torch.full_like(lse, math.log(window + 1)), rtol=1e-6, atol=1e-6)


# Device contents, not host bounds or physical capacity, determine the work.
@pytest.mark.parametrize("dq", DQS)
@pytest.mark.parametrize("query_tile", QUERY_TILES)
def test_runtime_lengths_pages_and_sinks(backend, dq, query_tile):
    case = make_case((65,), (2049,), heads=4, dq=dq, mode="per-tensor", window_left=128,
                     has_sink=True, nonunit_scales=True, poison_tail=False)
    lse = torch.full(case.q.shape[:2], -123, device=case.q.device, dtype=torch.float32)
    _, out, kernel = make_call(case, backend, True, lse=lse, query_tile=query_tile)
    keys, values = case.k_pages.clone(), case.v_pages.clone()
    compiled = None
    for length, sink in RUNTIME_LENGTHS_AND_SINKS:
        case.kv_lens = (length,)
        case.indptr.copy_(i32([0, (length + 63) // 64]))
        case.last.fill_((length - 1) % 64 + 1 if length else 0)
        case.sinks.fill_(sink)
        case.k_pages.copy_(keys)
        case.v_pages.copy_(values)
        if length % 64:
            physical = case.page_order[(length - 1) // 64]
            case.k_pages[physical, length % 64:] = float("nan")
            case.v_pages[physical, length % 64:] = float("nan")
        case.pack(copy=True)
        # Exercise the optional KV prefix too; the host maxima stay fixed.
        result = kernel(case.q, case.k, case.v, case.cq, None, case.indptr, case.indices,
                        65, 2049, True, case.qs, case.ks, case.vs, case.last,
                        out=out, lse=lse, sink_ptr=case.sinks)
        assert result is out
        compiled = _assert_compiled_keys(kernel, compiled)
        _assert_result(case, backend, out, lse)


@pytest.mark.parametrize("dq", DQS)
@pytest.mark.parametrize("query_tile", QUERY_TILES)
def test_runtime_query_mapping_and_empty_grid(backend, dq, query_tile):
    case = make_case((257, 513, 257), (129, 321, 193), heads=6, kv_heads=2, dq=dq,
                     q_offset=3, table_offset=2, window_left=128, has_sink=True,
                     nonunit_scales=True, poison_tail=False)
    lse = torch.full(case.q.shape[:2], -123, device=case.q.device, dtype=torch.float32)
    call, out, kernel = make_call(case, backend, True, lse=lse, query_tile=query_tile)
    compiled = None
    for lengths in ((257, 513, 257), (0, 1, 1026), (1027, 0, 0), (0, 0, 0), (513, 257, 257)):
        case.q_lens = lengths
        case.cq.copy_(i32(list(accumulate(lengths, initial=case.q_offset))))
        out.fill_(-123)
        lse.fill_(-123)
        assert call(max_seqlen_q=1027, max_seqlen_k=321) is out
        compiled = _assert_compiled_keys(kernel, compiled)
        _assert_result(case, backend, out, lse)


@pytest.mark.parametrize("dq", DQS)
def test_page_table_and_cache_mutations(backend, dq):
    case = make_case((33, 65), (192, 256), heads=4, kv_heads=2, dq=dq, table_offset=2,
                     window_left=128, has_sink=True)
    call, out, kernel = make_call(case, backend, True)
    compiled = None
    for mutation in ("original", "reverse", "alias", "cache"):
        if mutation == "reverse":
            case.page_order[2:] = reversed(case.page_order[2:])
            case.indices.copy_(i32(case.page_order))
        elif mutation == "alias":
            case.page_order[2:] = case.page_order[:1] * (len(case.page_order) - 2)
            case.indices.copy_(i32(case.page_order))
        elif mutation == "cache":
            case.k_pages.mul_(2)
            case.v_pages.mul_(0.5)
            case.pack(copy=True)
        assert call() is out
        compiled = _assert_compiled_keys(kernel, compiled)
        _assert_result(case, backend, out)


@pytest.mark.parametrize("dq", DQS)
def test_shared_physical_pages_across_sequences(backend, dq):
    case = make_case((9, 33), (129, 257), heads=6, kv_heads=2, dq=dq,
                     table_offset=2, q_offset=3, nonunit_scales=True,
                     window_left=128, has_sink=True, poison_tail=False)
    case.k_pages, case.v_pages = case.k_pages[:3], case.v_pages[:3]
    case.page_order[2:] = [2, 0, 2, 1, 2, 0, 1, 0]
    case.indices.copy_(i32(case.page_order))
    case.k, case.v = vectorize_kv(case.k_pages, case.v_pages)
    assert_case(case, backend, True)


@pytest.mark.parametrize("dq", DQS)
@pytest.mark.parametrize("query_tile", QUERY_TILES)
def test_excluded_prefix_is_not_read(backend, dq, query_tile):
    case = make_case((257,), (8193,), heads=4, dq=dq, window_left=128, has_sink=True)
    case.indices[:(8193 - 257 - 128) // 64] = 2**30
    # Leave the oracle's valid page_order untouched; only excluded device
    # entries are invalid. Clamping a speculative prefix load is not enough.
    assert_case(case, backend, True, query_tile=query_tile, repeats=10)


@pytest.mark.parametrize("dq", DQS)
@pytest.mark.parametrize("window", (0, 128))
def test_empty_queries(backend, dq, window):
    case = make_case((0,), (0,), heads=2, dq=dq, window_left=window, has_sink=True)
    out, lse = assert_case(case, backend, True)
    assert out.shape == (0, 2, 128) and lse.shape == (0, 2)
    call, _, kernel = make_call(case, backend, True)
    compiled = frozenset(kernel._compiled)
    assert dispatch_names(call) == []
    _assert_compiled_keys(kernel, compiled)


# Stream isolation, graph replay, caller-owned outputs, and no workspace.
@pytest.mark.parametrize("dq", DQS)
@pytest.mark.parametrize("window", (0, 128))
def test_streams_graphs_and_output_allocation(backend, dq, window):
    case = make_case((127,), (193,), heads=4, dq=dq, window_left=window, has_sink=True)
    call, _, kernel = make_call(case, backend, True)
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    outputs, lses, graphs = [], [], []
    compiled = None
    for stream in streams:
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            out, lse = call(out=None, return_lse=True, stream=stream)
            assert out.dtype == torch.bfloat16 and lse.dtype == torch.float32
            assert out.shape == (127, 4, 128) and lse.shape == (127, 4)
            compiled = _assert_compiled_keys(kernel, compiled)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                assert call(out=out, lse=lse, stream=stream) is out
                assert call(out=out, lse=lse, stream=stream) is out
            outputs.append(out)
            lses.append(lse)
            graphs.append(graph)
    assert outputs[0].data_ptr() != outputs[1].data_ptr()
    assert lses[0].data_ptr() != lses[1].data_ptr()
    for _ in range(5):
        for stream, graph in zip(streams, graphs):
            with torch.cuda.stream(stream):
                graph.replay()
    for stream in streams:
        torch.cuda.current_stream().wait_stream(stream)
    _assert_compiled_keys(kernel, compiled)
    for out, lse in zip(outputs, lses):
        _assert_result(case, backend, out, lse)
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
    torch.testing.assert_close(lses[0], lses[1], rtol=0, atol=0)


@pytest.mark.parametrize("dq", DQS)
@pytest.mark.parametrize("window", (0, 128))
@pytest.mark.parametrize("query_tile", QUERY_TILES)
def test_direct_only_one_launch_no_workspace(backend, dq, window, query_tile, monkeypatch):
    case = make_case((257,), (901,), heads=16, dq=dq, window_left=window, has_sink=True)
    call, out, kernel = make_call(case, backend, True, query_tile=query_tile)
    assert call() is out
    torch.cuda.synchronize()
    assert backend.load().THREADS == 64
    for name in ("prepare_kv", "attend_linear", "_workspace"):
        assert not hasattr(kernel, name)

    def no_allocation(*args, **kwargs):
        pytest.fail("warmed single-wave dispatch must not allocate a workspace")

    before = torch.cuda.memory_allocated()
    with monkeypatch.context() as patch:
        for name in ("empty", "empty_like", "zeros", "zeros_like", "full", "full_like", "tensor"):
            patch.setattr(torch, name, no_allocation)
        assert call() is out
        torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() == before
    names = dispatch_names(call)
    assert len(names) == 1 and "_swa" in names[0], names
    _assert_result(case, backend, out)


@pytest.mark.parametrize("dq", DQS)
@pytest.mark.parametrize("kv_len", (256, 321))
@pytest.mark.parametrize("has_sink", (False, True))
def test_default_no_lse_is_deterministic(backend, dq, kv_len, has_sink):
    case = make_case((257,), (kv_len,), heads=16, dq=dq, window_left=128, has_sink=has_sink)
    assert_case(case, backend, True, repeats=10, with_lse=False)


@pytest.mark.parametrize("dq", DQS)
@pytest.mark.parametrize("heads,kv_heads", ((1, 1), (4, 4), (8, 2), (6, 3)))
def test_mha_gqa_and_storage_offsets(backend, dq, heads, kv_heads):
    case = make_case((65,), (257,), heads=heads, kv_heads=kv_heads, dq=dq,
                     window_left=128, nonunit_scales=True, has_sink=True)
    backing_q = torch.zeros(67, heads + 1, dq + 16, device=case.q.device, dtype=torch.bfloat16)
    q = backing_q[1:66, :heads, 8:8 + dq]
    q.copy_(case.q)
    case.q = q
    backing_out = torch.full((67, heads + 1, 144), -123, device=q.device, dtype=torch.bfloat16)
    out = backing_out[1:66, :heads, 8:136]
    assert q.storage_offset() > 0 and out.storage_offset() > 0
    call, _, _ = make_call(case, backend, True, out=out)
    assert call() is out
    _assert_result(case, backend, out)
    assert (backing_out[0] == -123).all() and (backing_out[-1] == -123).all()
    assert (backing_out[:, heads:] == -123).all()
    assert (backing_out[:, :heads, :8] == -123).all()
    assert (backing_out[:, :heads, 136:] == -123).all()


# Invalid-buffer values are constructed inside the test, after arch selection.
def _strided_copy(tensor):
    shape = (*tensor.shape[:-1], tensor.shape[-1] * 2)
    backing = torch.empty(shape, device=tensor.device, dtype=tensor.dtype)
    view = backing[..., ::2]
    view.copy_(tensor)
    return view


def _forbid_launch(monkeypatch, backend, kernel):
    def unexpected_launch(*args, **kwargs):
        pytest.fail("invalid runtime buffers reached compilation or launch")

    class NoLaunchCache(dict):
        get = staticmethod(unexpected_launch)

    # The factory is cached: guard both previously compiled and cold paths,
    # without compiling a GPU kernel just to test input validation.
    monkeypatch.setattr(kernel, "_compiled", NoLaunchCache())
    monkeypatch.setattr(backend.load().flyc, "compile", unexpected_launch)


INVALID_RUNTIME_BUFFERS = (
    pytest.param("sink_ptr", lambda c: None, ValueError, "sink", id="sink-missing"),
    pytest.param("sink_ptr", lambda c: c.sinks[:3], ValueError, "sink", id="sink-shape"),
    pytest.param("sink_ptr", lambda c: c.sinks.cpu(), ValueError, "sink", id="sink-device"),
    pytest.param("sink_ptr", lambda c: _strided_copy(c.sinks), ValueError, "sink", id="sink-stride"),
    pytest.param("sink_ptr", lambda c: c.sinks.bfloat16(), ValueError, "sink", id="sink-dtype"),
    pytest.param("out", lambda c: torch.empty(9, 4, 128, device=c.q.device), ValueError, "output", id="out-dtype"),
    pytest.param("out", lambda c: torch.empty(8, 4, 128, device=c.q.device, dtype=torch.bfloat16), ValueError, "output", id="out-shape"),
    pytest.param("out", lambda c: torch.empty(9, 4, 128, dtype=torch.bfloat16), ValueError, "output", id="out-device"),
    pytest.param("out", lambda c: torch.empty(9, 4, 256, device=c.q.device, dtype=torch.bfloat16)[..., ::2], ValueError, "output", id="out-stride"),
    pytest.param("lse", lambda c: torch.empty(9, 4, device=c.q.device, dtype=torch.bfloat16), ValueError, "LSE", id="lse-dtype"),
    pytest.param("lse", lambda c: torch.empty(9, 3, device=c.q.device), ValueError, "LSE", id="lse-shape"),
    pytest.param("lse", lambda c: torch.empty(9, 4), ValueError, "LSE", id="lse-device"),
    pytest.param("lse", lambda c: torch.empty(4, 9, device=c.q.device).t(), ValueError, "LSE", id="lse-stride"),
    pytest.param("q", lambda c: c.q.flatten(), ValueError, "Q must", id="q-rank"),
    pytest.param("q", lambda c: c.q[:, :3], ValueError, "Q must", id="q-heads"),
    pytest.param("q", lambda c: _strided_copy(c.q), ValueError, "Q must", id="q-stride"),
    *(
        pytest.param(field, lambda c, field=field, dtype=dtype: getattr(c, field).to(dtype),
                     NotImplementedError, "BF16", id=f"{field}-{dtype}")
        for field, dtype in itertools.product(("q", "k", "v"), (torch.float32, torch.float16, torch.float8_e4m3fnuz))
    ),
    *(
        pytest.param(field, lambda c, field=field: getattr(c, field).flatten(),
                     ValueError, "SHUFFLE-5D", id=f"{field}-shape")
        for field in ("k", "v")
    ),
    *(
        pytest.param(field, lambda c, field=field: _strided_copy(getattr(c, field)),
                     ValueError, "K/V must", id=f"{field}-stride")
        for field in ("k", "v")
    ),
    *(
        pytest.param(field, lambda c, field=field: getattr(c, field).cpu(),
                     ValueError, "K/V must", id=f"{field}-device")
        for field in ("k", "v")
    ),
    pytest.param("cq", lambda c: c.cq[:1], ValueError, "batch metadata", id="cq-length"),
    pytest.param("indptr", lambda c: c.indptr[:1], ValueError, "batch metadata", id="indptr-length"),
    pytest.param("last", lambda c: c.last[:0], ValueError, "batch metadata", id="last-length"),
    pytest.param("ck", lambda c: c.ck[:1], ValueError, "KV prefix", id="ck-length"),
    *(
        pytest.param(field, lambda c, field=field: getattr(c, field).long(),
                     ValueError, "metadata", id=f"{field}-dtype")
        for field in ("cq", "ck", "indptr", "indices", "last")
    ),
    *(
        pytest.param(field, lambda c, field=field: getattr(c, field).cpu(),
                     ValueError, "metadata", id=f"{field}-device")
        for field in ("cq", "ck", "indptr", "indices", "last")
    ),
    *(
        pytest.param(field, lambda c, field=field: _strided_copy(getattr(c, field)),
                     ValueError, "metadata", id=f"{field}-stride")
        for field in ("cq", "ck", "indptr", "indices")
    ),
    pytest.param("cq", lambda c: c.cq[None, :], ValueError, "metadata", id="cq-rank"),
    *(
        pytest.param(field, lambda c, field=field: getattr(c, field).bfloat16(),
                     ValueError, "descales", id=f"{field}-dtype")
        for field in ("qs", "ks", "vs")
    ),
    *(
        pytest.param(field, lambda c, field=field: getattr(c, field).cpu(),
                     ValueError, "descales", id=f"{field}-device")
        for field in ("qs", "ks", "vs")
    ),
    pytest.param("qs", lambda c: _strided_copy(c.qs), ValueError, "descales", id="qs-stride"),
    *(
        pytest.param(field, lambda c: torch.ones(2, device=c.q.device),
                     ValueError, "descales", id=f"{field}-size")
        for field in ("qs", "ks", "vs")
    ),
    *(
        pytest.param("softmax_scale", lambda c, value=value: value,
                     ValueError, "softmax_scale", id=f"scale-{value}")
        for value in (0.0, -1.0, float("nan"), float("inf"), -float("inf"))
    ),
    pytest.param("max_seqlen_q", lambda c: -1, ValueError, "maximum lengths", id="negative-max-q"),
    pytest.param("max_seqlen_k", lambda c: -1, ValueError, "maximum lengths", id="negative-max-k"),
)


@pytest.mark.parametrize("target,invalid,error,match", INVALID_RUNTIME_BUFFERS)
def test_invalid_buffers_fail_before_launch(backend, monkeypatch, target, invalid, error, match):
    case = make_case((9,), (65,), heads=4, window_left=128, has_sink=True)
    call, out, kernel = make_call(case, backend, True)
    value = invalid(case)
    runtime = {}
    if target in ("q", "k", "v", "cq", "ck", "indptr", "indices", "last", "qs", "ks", "vs"):
        setattr(case, target, value)
    else:
        runtime[target] = value
    _forbid_launch(monkeypatch, backend, kernel)
    with pytest.raises(error, match=match):
        call(**runtime)
    assert (out == -123).all()
    assert not kernel._compiled


def test_sink_requires_factory_opt_in(backend, monkeypatch):
    case = make_case((9,), (65,), heads=4, window_left=128, has_sink=False)
    call, _, kernel = make_call(case, backend, True)
    sink = torch.zeros(case.heads, device=case.q.device)
    _forbid_launch(monkeypatch, backend, kernel)
    with pytest.raises(ValueError, match="has_sink"):
        call(sink_ptr=sink)


def test_runtime_requires_causal(backend, monkeypatch):
    case = make_case((9,), (65,), heads=4, window_left=128)
    _, out, kernel = make_call(case, backend, True)
    _forbid_launch(monkeypatch, backend, kernel)
    with pytest.raises(ValueError, match="causal"):
        kernel(case.q, case.k, case.v, case.cq, case.ck, case.indptr, case.indices,
               9, 65, False, case.qs, case.ks, case.vs, case.last, out=out)


# CPU-only factory and address-model checks: never use the backend fixture.
@pytest.mark.parametrize("dq", DQS)
@pytest.mark.parametrize("window,tile", ((0, 16), (16, 16), (17, 32), (128, 32)))
def test_explicit_scope_and_default_tiles(dq, window, tile):
    module = SWA.load()
    kernel = module.PagedAttention(16, 1, dq, 128, 64, window_left=window)
    assert module.THREADS == 64
    assert (kernel.query_tile, kernel.block_n) == (tile, tile)


@pytest.mark.parametrize("dq", DQS)
@pytest.mark.parametrize("query_tile,block_n", TILE_CONFIGS)
@pytest.mark.parametrize("mode", ("per-token", "per-tensor"))
def test_explicit_tiles_and_scale_modes(dq, query_tile, block_n, mode):
    kernel = SWA.load().PagedAttention(16, 1, dq, 128, 64, quant_query_mode=mode,
                                     query_tile=query_tile, block_n=block_n)
    assert (kernel.query_tile, kernel.block_n) == (query_tile, block_n)


@pytest.mark.parametrize("config,options,error", (
    pytest.param((16, 1, 64, 128, 64), {}, NotImplementedError, id="unsupported-dq"),
    pytest.param((16, 1, 192, 64, 64), {}, NotImplementedError, id="unsupported-dv"),
    pytest.param((16, 1, 192, 128, 32), {}, NotImplementedError, id="unsupported-page"),
    pytest.param((16, 1, 192, 128, 64), {"key_layout": "linear"}, NotImplementedError, id="unsupported-layout"),
    *(
        pytest.param((heads, kv_heads, 192, 128, 64), {}, ValueError, id=f"heads-{heads}-{kv_heads}")
        for heads, kv_heads in ((7, 2), (0, 1), (4, 0), (-4, 1), (4, -1))
    ),
    *(
        pytest.param((16, 1, 192, 128, 64), options, ValueError, id=name)
        for name, options in (
            ("noncausal", {"is_causal": False}),
            ("negative-window", {"window_left": -1}),
            ("noninteger-window", {"window_left": 1.5}),
            ("overflow-window", {"window_left": 2**31}),
            ("query-tile", {"query_tile": 64}),
            ("block-n", {"block_n": 8}),
            ("scale-mode", {"quant_query_mode": "per-head"}),
        )
    ),
))
def test_factory_validation(config, options, error):
    with pytest.raises(error):
        SWA.load().PagedAttention(*config, **options)


@pytest.mark.parametrize("arch", ARCHES)
@pytest.mark.parametrize("logical_k", (16, 32))
def test_native_mfma_atom_selection(arch, logical_k):
    # This checks the host-side atom contract, not emitted ISA. The shared
    # audit checks target-specific forbidden instructions and resource usage.
    assert SWA.load()._atom_k(logical_k, int(arch[3:])) == (16 if arch == "gfx942" else logical_k)


@pytest.mark.parametrize("dq", DQS)
@pytest.mark.parametrize("block_n", BLOCK_NS)
def test_fragment_address_coverage(dq, block_n):
    # Each K/V element is loaded exactly once. Both architectures retain this
    # load layout, including the BN32/64 interleaved K-row permutation.
    for tile in range(0, 64, block_n):
        key_addresses, value_addresses = [], []
        atom_k = 16 if block_n == 16 else 32
        for lane in range(64):
            for n in range(block_n // 16):
                row = lane & 15 if block_n == 16 else (lane & 3) + ((lane & 12) << 1) + (n & 1) * 4 + (n // 2) * 32
                for k in range(dq // 32):
                    key_addresses.extend((((lane >> 4) + k * 4) * 64 + tile + row) * 8 + i for i in range(8))
            for n in range(8):
                for k in range(block_n // atom_k):
                    token = tile + (lane >> 4) * (atom_k // 4) + k * atom_k
                    value_addresses.extend((token // 8 * 128 + n * 16 + (lane & 15)) * 8 + (token & 7) + i
                                           for i in range(atom_k // 4))
        expected_k = {(d // 8 * 64 + token) * 8 + d % 8 for token in range(tile, tile + block_n) for d in range(dq)}
        expected_v = {(token // 8 * 128 + d) * 8 + token % 8 for token in range(tile, tile + block_n) for d in range(128)}
        assert len(key_addresses) == len(set(key_addresses)) == block_n * dq
        assert len(value_addresses) == len(set(value_addresses)) == block_n * 128
        assert set(key_addresses) == expected_k and set(value_addresses) == expected_v


@pytest.mark.parametrize("query_tile", QUERY_TILES)
@pytest.mark.parametrize("block_n", BLOCK_NS)
def test_query_output_layout_and_window_coverage(query_tile, block_n):
    output = [((lane & 15) + m * 16, (lane >> 4) * 4 + n * 16 + i)
              for lane in range(64) for m in range(query_tile // 16) for n in range(8) for i in range(4)]
    assert len(output) == len(set(output)) == query_tile * 128
    assert set(output) == set(itertools.product(range(query_tile), range(128)))
    for q_len, kv_len, window in itertools.product((1, 17, 33, 129), (0, 1, 63, 65, 256), (0, 1, 16, 31, 64, 128, 512)):
        for qstart in range(0, q_len, query_tile):
            valid_q = min(query_tile, q_len - qstart)
            first = max(0, qstart + kv_len - q_len - window) & -block_n
            end = max(0, min(kv_len, qstart + valid_q + kv_len - q_len))
            visited = [col for tile in range(first, end, block_n) for col in range(tile, tile + block_n)]
            for row in range(qstart, qstart + valid_q):
                diagonal = kv_len - q_len + row
                accepted = {col for col in visited if ((diagonal - col) & 0xFFFFFFFF) <= window}
                expected = set(range(max(0, diagonal - window), max(0, min(kv_len, diagonal + 1))))
                assert accepted == expected


@pytest.mark.parametrize("kv_len", (0, 2))
@pytest.mark.parametrize("window", (0, 1, 128))
@pytest.mark.parametrize("sink", (None, -80.0, 0.0, 80.0, -float("inf")))
def test_cpu_reference_nan_padding_and_empty_rows(kv_len, window, sink):
    # Oracle query guards are intentionally NaN, unlike caller-owned output
    # guards (-123). Active rows must stay finite even with poisoned KV tails.
    q = torch.zeros(7, 2, 128, dtype=torch.bfloat16)
    keys = torch.zeros(1, 64, 1, 128, dtype=torch.bfloat16)
    values = torch.ones_like(keys)
    keys[:, kv_len:] = float("nan")
    values[:, kv_len:] = float("nan")
    k, v = vectorize_kv(keys, values)
    case = Case(
        q=q, k_pages=keys, v_pages=values, k=k, v=v,
        cq=torch.tensor([2, 5], dtype=torch.int32), ck=torch.tensor([0, kv_len], dtype=torch.int32),
        indptr=torch.tensor([0, int(kv_len > 0)], dtype=torch.int32),
        indices=torch.tensor([0], dtype=torch.int32), last=torch.tensor([kv_len], dtype=torch.int32),
        qs=torch.ones(7, 2, 1), ks=torch.ones(1), vs=torch.ones(1),
        q_lens=(3,), kv_lens=(kv_len,), page_order=[0], q_offset=2, table_offset=0,
        mode="per-token", window_left=window,
        sinks=None if sink is None else torch.full((2,), sink, dtype=torch.float32),
    )
    out, lse = torch_reference(case, True)
    assert torch.isnan(out[:2]).all() and torch.isnan(out[5:]).all()
    assert torch.isnan(lse[:2]).all() and torch.isnan(lse[5:]).all()
    counts = (0, 1, min(2, window + 1)) if kv_len else (0, 0, 0)
    sink_mass = 0.0 if sink is None else math.exp(sink)
    expected_out, expected_lse = [], []
    for count in counts:
        total = count + sink_mass
        expected_out.append(count / total if total else 0.0)
        expected_lse.append(math.log(total) if total else -float("inf"))
    target = torch.tensor(expected_out)[:, None, None].expand(3, 2, 128)
    target_lse = torch.tensor(expected_lse)[:, None].expand(3, 2)
    torch.testing.assert_close(out[2:5], target, rtol=2e-6, atol=0)
    torch.testing.assert_close(lse[2:5], target_lse, rtol=2e-6, atol=1e-6)
    # Shared assertions must ignore expected oracle NaN guards without
    # accepting NaNs in live output, or requiring kernels to write padding.
    actual = torch.full((7, 2, 128), -123, dtype=torch.bfloat16)
    actual_lse = torch.full((7, 2), -123, dtype=torch.float32)
    actual[2:5].copy_(out[2:5])
    actual_lse[2:5].copy_(lse[2:5])
    _assert_result(case, SWA, actual, actual_lse)


# Same-input AITER comparisons use unit descales; missing references alone skip.
@pytest.mark.parametrize("dq", DQS)
@pytest.mark.parametrize("window,has_sink", ((0, False), (1, True), (128, False), (128, True)))
@pytest.mark.parametrize("poison_tail", (False, True), ids=("zero-tail", "nan-tail"))
@pytest.mark.parametrize("softmax_scale", (None, 0.0625))
@pytest.mark.parametrize("reference_kind", ("paged", "ck_linear"))
def test_aiter_function_comparison(backend, dq, window, has_sink, poison_tail, softmax_scale, reference_kind):
    case = make_case((257,), (777,), heads=16, kv_heads=1, dq=dq, dtype=torch.bfloat16,
                     window_left=window, has_sink=has_sink, nonunit_scales=False,
                     poison_tail=poison_tail)
    actual, _ = assert_case(case, backend, True, softmax_scale=softmax_scale)
    reference_out = torch.full_like(actual, -123)
    references = _references()
    try:
        factory = references.aiter_call if reference_kind == "paged" else references.aiter_linear_call
        reference_call = factory(case, True, out=reference_out, softmax_scale=softmax_scale)
        references.probe_reference(reference_call)
    except references.ReferenceUnavailable as error:
        pytest.skip(str(error))
    # No RuntimeError/AssertionError catch, xfail, equal_nan, or relaxed tail
    # tolerance: a present reference must satisfy the same SWA/sink semantics.
    assert_close(case, backend, reference_out, None, True, softmax_scale=softmax_scale)
    torch.testing.assert_close(actual.float(), reference_out.float(), rtol=0.02, atol=0.02)


@pytest.mark.parametrize("error_type", (RuntimeError, AssertionError, ValueError))
def test_reference_probe_preserves_correctness_failures(error_type):
    references = _references()
    error = error_type("unexpected attention correctness failure")

    def broken():
        raise error

    with pytest.raises(error_type) as caught:
        references.probe_reference(broken)
    assert caught.value is error


@pytest.mark.parametrize("dq", DQS)
@pytest.mark.parametrize("q_len,kv_len,window", PERFORMANCE_CASES)
@pytest.mark.parametrize("query_tile,block_n", (pytest.param(None, None, id="auto"), *TILE_CONFIGS))
def test_original_performance_configurations(backend, dq, q_len, kv_len, window, query_tile, block_n):
    case = make_case((q_len,), (kv_len,), heads=16, kv_heads=1, dq=dq, dtype=torch.bfloat16,
                     window_left=window, has_sink=True, poison_tail=False, seed=20260905)
    # Match the original warmed, preallocated-output, no-LSE benchmark path.
    assert_case(case, backend, True, query_tile=query_tile, block_n=block_n, with_lse=False)


def gather_linear_calls(case, *, softmax_scale=None):
    """Explicit comparison only: gather once per total call, never a SWA fallback."""
    if __package__:
        from ._gather import gather_swa_kv_call
    else:
        from _gather import gather_swa_kv_call
    gather, linear_kv = gather_swa_kv_call(case)
    linear = _references().aiter_linear_call(case, True, softmax_scale=softmax_scale, linear_kv=linear_kv)

    def gather_linear():
        gather()
        return linear()

    return {"gather": gather, "aiter_ck_linear_prepared": linear,
            "gather_aiter_ck_linear": gather_linear}, linear_kv


def compare_gather_linear_events(candidates, *, warmup=20, iterations=100, rounds=5):
    """Original per-sample candidate rotation, GPU-event medians of five rounds.

    A single event interval surrounds gather+linear. Do not sum independently
    measured components or report only the attention dispatch for this path.
    """
    if not candidates or warmup < 0 or iterations < 1 or rounds < 1:
        raise ValueError("candidates, positive iterations/rounds and nonnegative warmup required")
    names = list(candidates)
    samples = {name: [] for name in names}
    for trial in range(rounds):
        for index in range(warmup):
            offset = (trial + index) % len(names)
            for name in names[offset:] + names[:offset]:
                candidates[name]()
        torch.cuda.synchronize()
        # Allocate timing events outside all measured intervals.
        pending = []
        for index in range(iterations):
            offset = (trial + index) % len(names)
            for name in names[offset:] + names[:offset]:
                pending.append((name, torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)))
        for name, start, end in pending:
            start.record()
            candidates[name]()
            end.record()
        torch.cuda.synchronize()
        raw = {name: [] for name in names}
        for name, start, end in pending:
            us = start.elapsed_time(end) * 1000
            if not math.isfinite(us) or us <= 0:
                raise ValueError("event latency must be finite and positive")
            raw[name].append(us)
        for name in names:
            samples[name].append({"median_us": statistics.median(raw[name]), "raw_us": raw[name]})
    return {name: statistics.median(row["median_us"] for row in values) for name, values in samples.items()}, samples


def gather_linear_workloads(args):
    if __package__:
        from . import _runner
        from ._perf_cases import select_workloads
    else:
        import _runner
        from _perf_cases import select_workloads
    if args.matrix == "documented" and not args.case:
        cases = select_workloads(tuple(f"swa_kv{kv}_d*" for kv in GATHER_LINEAR_KV_LENGTHS))
    else:
        cases = _runner.benchmark_workloads(args)
    cases = tuple(case for case in cases if _runner.case_selected(args, case))
    if not cases:
        raise ValueError("no gather+linear workload matches the filters")
    if not any(case.unsupported(SWA) is None for case in cases):
        raise ValueError("no supported SWA workload selected for gather+linear comparison")
    return cases


def benchmark_gather_linear(args):
    """SWA vs original full-cache Triton gather + explicit AITER CK varlen.

    Use --mode performance --gather-linear. Original defaults: Q16K,
    KV32K/64K/128K, W128/sink; extend D192 to both D128/D192 and both GPUs.
    """
    if __package__:
        from . import _runner
    else:
        import _runner
    output = _runner.result_path(args, "swa_gather_linear")
    result = {"environment": _runner.environment(), "records": [], "unavailable": [], "complete": False,
              "comparison_complete": False,
              "comparison_source": GATHER_LINEAR_SOURCE, "isolated": not args.allow_contention}
    _runner.save(output, result)
    workloads = gather_linear_workloads(args)
    if not SWA.available:
        result["unavailable"].append({"backend": SWA.name, "reason": "requires native gfx942 or gfx950"})
        _runner.save(output, result)
        raise RuntimeError("SWA gather+linear comparison requires native gfx942 or gfx950")
    warmup = 20 if args.warmup is None else args.warmup
    iterations = 100 if args.iterations is None else args.iterations
    rounds = 5 if args.rounds is None else args.rounds
    references = _references()
    for workload in workloads:
        reason = workload.unsupported(SWA)
        if reason:
            result["unavailable"].append({"case": workload.name, "reason": reason})
            continue
        if not args.allow_contention:
            _runner.require_idle_device()
        case = _runner.make_performance_case(workload, SWA)
        configurations = [("auto", {})]
        if args.tiles:
            configurations += [(f"q{qt}_bn{bn}", {"query_tile": qt, "block_n": bn}) for qt, bn in TILE_CONFIGS]
        for config, options in configurations:
            direct, out, _ = make_call(case, SWA, True, **options)
            direct()
            reference, _ = assert_close(case, SWA, out, None, True)
            first = out.clone()
            for _ in range(3):
                torch.testing.assert_close(direct(), first, rtol=0, atol=0)
            calls, unavailable = {"swa_direct": direct}, {}
            workspace_bytes = None
            try:
                comparisons, linear_kv = gather_linear_calls(case)
                # Initialization and full gather correctness are outside timing.
                gathered = references.probe_reference(comparisons["gather"])
                keys, values = case.logical_kv()
                for actual, expected in zip(gathered, (torch.cat(keys), torch.cat(values))):
                    torch.testing.assert_close(actual.float(), expected, rtol=0, atol=0)
                for name in ("aiter_ck_linear_prepared", "gather_aiter_ck_linear"):
                    actual = references.probe_reference(comparisons[name])
                    torch.testing.assert_close(actual.float(), reference, rtol=0.02, atol=0.02)
                    stable = actual.clone()
                    for _ in range(3):
                        torch.testing.assert_close(comparisons[name](), stable, rtol=0, atol=0)
                calls.update(comparisons)
                workspace_bytes = sum(tensor.numel() * tensor.element_size() for tensor in linear_kv)
            except references.ReferenceUnavailable as exc:
                unavailable["gather_aiter_ck_linear"] = str(exc)
                if args.aiter == "required":
                    result["unavailable"].append({"case": workload.name, "reason": str(exc)})
                    _runner.save(output, result)
                    raise
            dispatches = {name: dispatch_names(call) for name, call in calls.items()}
            if "gather" in calls:
                assert len(dispatches["gather"]) == 1, dispatches
                assert len(dispatches["swa_direct"]) == 1 and "_swa" in dispatches["swa_direct"][0], dispatches
                assert len(dispatches["gather_aiter_ck_linear"]) == len(dispatches["gather"]) + len(dispatches["aiter_ck_linear_prepared"]), dispatches
            if not args.allow_contention:
                _runner.require_idle_device()
            times, samples = compare_gather_linear_events(calls, warmup=warmup, iterations=iterations, rounds=rounds)
            if not args.allow_contention:
                _runner.require_idle_device()
            entry = {**workload.to_dict(), "config": config, "backend": SWA.name,
                "event_interval_us": times, "rounds": samples, "dispatch_names": dispatches,
                "tflops": {name: None if name == "gather" else workload.flops / us / 1e6 for name, us in times.items()},
                "speedup_vs_gather_linear": times.get("gather_aiter_ck_linear", 0) / times["swa_direct"] if "gather_aiter_ck_linear" in times else None,
                "linear_workspace_bytes": workspace_bytes, "reference_unavailable": unavailable,
                "comparison_complete": not unavailable,
                "timing_note": "GPU event interval; gather+linear includes both launches and gaps, not a sum of standalone times",
                "gather_note": "all logical KV tokens; fixed slot mapping and preallocated workspaces prepared outside timing",
                "protocol": {"timer": "events", "sample_warmup": warmup, "iterations": iterations, "rounds": rounds,
                             "statistic": "median of all samples; median of rounds", "candidate_order": "rotated every sample", "buffers": 1}}
            result["records"].append(entry)
            _runner.save(output, result)
            print("SWA_GATHER_LINEAR_RESULT", json.dumps({k: v for k, v in entry.items() if k != "rounds"}), flush=True)
    result["complete"] = True
    result["comparison_complete"] = bool(result["records"]) and all(row["comparison_complete"] for row in result["records"])
    _runner.save(output, result)
    return result


@pytest.mark.parametrize("dq", DQS)
@pytest.mark.parametrize("q_lens,kv_lens,kv_heads", (((129,), (1154,), 1), ((3, 33, 17), (0, 129, 65), 2)))
def test_gather_full_cache_and_live_values(backend, dq, q_lens, kv_lens, kv_heads, monkeypatch):
    if __package__:
        from ._gather import gather_swa_kv_call
    else:
        from _gather import gather_swa_kv_call
    case = make_case(q_lens, kv_lens, heads=4, kv_heads=kv_heads, dq=dq, window_left=1, poison_tail=True)
    try:
        gather, buffers = gather_swa_kv_call(case)
        _references().probe_reference(gather)
    except _references().ReferenceUnavailable as exc:
        pytest.skip(str(exc))
    for mutation in (False, True):
        if mutation:
            case.k_pages.mul_(0.5)
            case.v_pages.neg_()
            case.pack(copy=True)
        result = gather()
        assert result[0] is buffers[0] and result[1] is buffers[1]
        keys, values = case.logical_kv()
        torch.testing.assert_close(result[0].float(), torch.cat(keys), rtol=0, atol=0)
        torch.testing.assert_close(result[1].float(), torch.cat(values), rtol=0, atol=0)
        assert result[0].shape[0] == sum(kv_lens)  # Not a window-pruned suffix.
    torch.cuda.synchronize()
    def no_allocation(*args, **kwargs):
        pytest.fail("warmed gather must reuse its slot mapping and linear workspace")
    with monkeypatch.context() as patch:
        for name in ("empty", "empty_like", "arange", "cat", "tensor"):
            patch.setattr(torch, name, no_allocation)
        assert gather()[0] is buffers[0]
        torch.cuda.synchronize()
    assert len(dispatch_names(gather)) == 1


@pytest.mark.parametrize("dq", DQS)
@pytest.mark.parametrize("q_lens,kv_lens", (((129,), (1154,)), ((129, 256, 17), (193, 447, 337))))
@pytest.mark.parametrize("window,sink", ((0, False), (128, True)))
def test_gather_linear_comparison_matches_reference(backend, dq, q_lens, kv_lens, window, sink):
    case = make_case(q_lens, kv_lens, heads=16, dq=dq, window_left=window, has_sink=sink,
                     poison_tail=True, source_dtype=torch.bfloat16)
    direct, _ = assert_case(case, backend, True, with_lse=False)
    refs = _references()
    try:
        calls, buffers = gather_linear_calls(case)
        refs.probe_reference(calls["gather"])
        for name in ("aiter_ck_linear_prepared", "gather_aiter_ck_linear"):
            actual = refs.probe_reference(calls[name])
            assert_close(case, backend, actual, None, True)
            torch.testing.assert_close(actual.float(), direct.float(), rtol=0.02, atol=0.02)
    except refs.ReferenceUnavailable as exc:
        pytest.skip(str(exc))
    assert buffers[0].shape == (sum(kv_lens), 1, dq)
    dispatch = {name: dispatch_names(call) for name, call in calls.items()}
    assert len(dispatch["gather_aiter_ck_linear"]) == len(dispatch["gather"]) + len(dispatch["aiter_ck_linear_prepared"])


def test_gather_linear_event_timer_includes_both_calls_and_gap(monkeypatch):
    clock, order = [0.0], []
    class Event:
        def __init__(self, **kwargs):
            self.timestamp = 0.0
        def record(self):
            self.timestamp = clock[0]
        def elapsed_time(self, other):
            return (other.timestamp - self.timestamp) / 1000
    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    def step(name, elapsed):
        order.append(name)
        clock[0] += elapsed
    calls = {"direct": lambda: step("direct", 2), "gather": lambda: step("gather", 3),
             "linear": lambda: step("linear", 5)}
    def combined():
        # Model an extra launch gap: total must be measured, not 3+5 inferred.
        step("gather_linear", 9)
    calls["gather_linear"] = combined
    times, samples = compare_gather_linear_events(calls, warmup=2, iterations=3, rounds=2)
    assert times == {"direct": 2, "gather": 3, "linear": 5, "gather_linear": 9}
    assert all(len(rows) == 2 and all(len(row["raw_us"]) == 3 for row in rows) for rows in samples.values())
    names = list(calls)
    expected = []
    for trial in range(2):
        for count in (2, 3):
            for index in range(count):
                offset = (trial + index) % 4
                expected.extend(names[offset:] + names[:offset])
    assert order == expected


def test_gather_linear_composition_gathers_on_every_call(monkeypatch):
    if __package__:
        from . import _gather
    else:
        import _gather
    order = []
    buffers = (torch.empty(3, 1, 128), torch.empty(3, 1, 128))
    output = torch.empty(2, 4, 128)
    def gather():
        order.append("gather")
        return buffers
    def linear():
        order.append("linear")
        return output
    def factory(case, causal, **kwargs):
        assert causal and kwargs["linear_kv"] is buffers
        return linear
    monkeypatch.setattr(_gather, "gather_swa_kv_call", lambda case: (gather, buffers))
    monkeypatch.setattr(_references(), "aiter_linear_call", factory)
    calls, returned = gather_linear_calls(object())
    assert returned is buffers and not order
    assert calls["aiter_ck_linear_prepared"]() is output
    for _ in range(2):
        assert calls["gather_aiter_ck_linear"]() is output
    assert order == ["linear", "gather", "linear", "gather", "linear"]


def test_gather_linear_supplied_buffers_are_not_regathered(monkeypatch):
    import sys
    from types import SimpleNamespace
    references = _references()
    q = torch.zeros(2, 4, 128, dtype=torch.bfloat16)
    k, v = (torch.ones(3, 1, 128, dtype=torch.bfloat16) for _ in range(2))
    sink = torch.ones(4)
    def unexpected():
        pytest.fail("supplied linear buffers must not trigger Python logical_kv/gather")
    case = SimpleNamespace(q=q, q_offset=0, table_offset=0, dq=128, dv=128, heads=4, kv_heads=1,
        q_lens=(2,), kv_lens=(3,), qs=torch.ones(2, 4, 1), ks=torch.ones(1), vs=torch.ones(1),
        cq=torch.tensor([0, 2], dtype=torch.int32), ck=torch.tensor([0, 3], dtype=torch.int32),
        window_left=128, sinks=sink, logical_kv=unexpected)
    captured = []
    def ck(*args, **kwargs):
        captured.append((args, kwargs))
        kwargs["out"].zero_()
        return kwargs["out"], None, None, None
    monkeypatch.setattr(references, "_aiter", lambda: None)
    monkeypatch.setitem(sys.modules, "aiter.ops.mha", SimpleNamespace(mha_varlen_fwd=ck))
    call = references.aiter_linear_call(case, True, linear_kv=(k, v), softmax_scale=0.0625)
    assert call().shape == (2, 4, 128)
    args, kwargs = captured[0]
    assert args[1] is k and args[2] is v and args[9] == 0.0625
    assert args[12:15] == (True, 128, 0) and kwargs["sink_ptr"] is sink
    with pytest.raises(ValueError, match="linear K"):
        references.aiter_linear_call(case, True, linear_kv=(k.float(), v))


@pytest.mark.parametrize("failure,policy", ((None, "required"), ("missing", "auto"),
                                           ("missing", "required"), ("numerical", "auto")))
def test_gather_linear_benchmark_preserves_failures_and_timing_scope(monkeypatch, tmp_path, failure, policy):
    from types import SimpleNamespace
    if __package__:
        from . import _runner
        from ._perf_cases import Workload
        from ._testing import Backend
    else:
        import _runner
        from _perf_cases import Workload
        from _testing import Backend
    import sys
    test_module = sys.modules[__name__]
    work = Workload("swa_mock", (2,), (3,), dq=128, heads=2, causal=True, window=128)
    q = torch.ones(2, 2, 128, dtype=torch.bfloat16)
    buffers = tuple(torch.ones(3, 1, 128, dtype=torch.bfloat16) for _ in range(2))
    case = SimpleNamespace(q=q, logical_kv=lambda: ([buffers[0].float()], [buffers[1].float()]))
    def direct():
        return q
    def gather():
        return buffers
    def linear():
        return q
    def combined():
        gather()
        return linear()
    def factories(_case):
        if failure == "missing":
            raise _references().ReferenceUnavailable("no optional reference")
        if failure == "numerical":
            raise AssertionError("bad gather output")
        return {"gather": gather, "aiter_ck_linear_prepared": linear, "gather_aiter_ck_linear": combined}, buffers
    monkeypatch.setattr(_runner, "environment", lambda: {"gpu": "CPU mock"})
    monkeypatch.setattr(Backend, "available", property(lambda self: True))
    monkeypatch.setattr(_runner, "make_performance_case", lambda *args: case)
    monkeypatch.setattr(_runner, "require_idle_device", lambda: None)
    monkeypatch.setattr(test_module, "gather_linear_workloads", lambda args: (work,))
    monkeypatch.setattr(test_module, "make_call", lambda *args, **kwargs: (direct, q, None))
    monkeypatch.setattr(test_module, "assert_close", lambda *args: (q.float(), None))
    monkeypatch.setattr(test_module, "gather_linear_calls", factories)
    monkeypatch.setattr(_references(), "probe_reference", lambda call: call())
    dispatches = {direct: ["_swa32_kernel_0"], gather: ["_gather_swa_kv_kernel"], linear: ["ck_fmha"],
                  combined: ["_gather_swa_kv_kernel", "ck_fmha"]}
    monkeypatch.setattr(test_module, "dispatch_names", lambda call: dispatches[call])
    measured = []
    def measure(calls, **kwargs):
        measured.append((calls, kwargs))
        times = {"swa_direct": 2, "gather": 3, "aiter_ck_linear_prepared": 5, "gather_aiter_ck_linear": 9}
        return {name: times[name] for name in calls}, {name: [] for name in calls}
    monkeypatch.setattr(test_module, "compare_gather_linear_events", measure)
    args = SimpleNamespace(output=tmp_path / "comparison.json", mode="performance", allow_contention=False,
                           warmup=None, iterations=None, rounds=None, tiles=False, aiter=policy)
    expected = (pytest.raises(AssertionError, match="bad gather") if failure == "numerical" else
                pytest.raises(_references().ReferenceUnavailable) if failure == "missing" and policy == "required" else
                contextlib.nullcontext())
    with expected:
        result = benchmark_gather_linear(args)
        assert result["complete"] and result["comparison_complete"] == (failure is None)
        row, = result["records"]
        if failure is None:
            assert row["event_interval_us"]["gather_aiter_ck_linear"] == 9
            assert row["tflops"]["gather"] is None and row["speedup_vs_gather_linear"] == 4.5
            assert row["linear_workspace_bytes"] == 2 * 3 * 128 * 2
        else:
            assert row["reference_unavailable"] and row["speedup_vs_gather_linear"] is None
    if failure == "numerical" or failure == "missing" and policy == "required":
        assert not measured and not json.loads(args.output.read_text())["complete"]
    else:
        assert measured[0][1] == {"warmup": 20, "iterations": 100, "rounds": 5}


def test_gather_linear_cli_plan_is_original_six_cases(monkeypatch, capsys):
    if __package__:
        from . import _runner
    else:
        import _runner
    def no_gpu(*args, **kwargs):
        pytest.fail("gather+linear plan must not initialize/query GPU or AITER")
    for name in ("get_device_properties", "is_available", "_lazy_init"):
        monkeypatch.setattr(torch.cuda, name, no_gpu)
    monkeypatch.setattr(_references(), "_aiter", no_gpu)
    monkeypatch.setattr(_runner.sys, "argv", ["test", "--mode", "performance", "--gather-linear", "--list-cases"])
    _runner.main(__file__, suite="swa")
    plan = json.loads(capsys.readouterr().out)
    assert plan["gpu_queried"] is False
    assert {row["name"] for row in plan["workloads"]} == {
        f"swa_kv{kv}_d{dq}" for kv in (32768, 65536, 131072) for dq in DQS}
    assert all(row["measurement_protocol"] == {"timer": "events", "sample_warmup": 20,
        "iterations": 100, "rounds": 5, "buffers": 1, "candidate_order": "rotated every sample"} for row in plan["workloads"])


@pytest.mark.parametrize("extra", (("--aiter", "off"), ("--timer", "profiler"), ("--buffers", "10"), ("--require-baseline",)))
def test_gather_linear_cli_rejects_mislabelled_protocol(monkeypatch, extra):
    if __package__:
        from . import _runner
    else:
        import _runner
    monkeypatch.setattr(_runner.sys, "argv", ["test", "--mode", "performance", "--gather-linear", *extra])
    with pytest.raises(SystemExit) as error:
        _runner.main(__file__, suite="swa")
    assert error.value.code == 2


if __name__ == "__main__":
    try:
        from ._runner import main
    except ImportError:
        from _runner import main
    main(__file__, suite="swa")