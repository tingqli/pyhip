"""Native linear BF16 D256 contracts; unchanged independent oracle, no timing.

GPU discovery is deferred to fixture setup. Run from the repository root with
the existing ROCm/FlyDSL test environment and the desired current device.
"""

import random
import re

import pytest
import torch

from experiments.attention.flydsl.flash_attn_api import flash_attn_varlen_d256 as api
from experiments.attention.flydsl.mha import test_mha_pa as common


@pytest.fixture(scope="module", autouse=True)
def native_resources():
    yield
    from experiments.attention.flydsl.mha import mha_pa_bf16_256_linear_942 as core

    # Opaque asynchronous LDS operands must never be spilled before their wait.
    failures = []
    for signature, compiled in core._COMPILED.items():
        ir = compiled._keepalive.ir
        for field in ("private_segment_fixed_size", "vgpr_spill_count", "sgpr_spill_count"):
            values = re.findall(rf"\b{field} = (\d+)", ir)
            if not values or any(int(value) != 0 for value in values):
                failures.append((signature, field, values))
    assert not failures, failures


@pytest.fixture(scope="module")
def gfx942():
    if not torch.cuda.is_available():
        pytest.skip("requires gfx942")
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    if getattr(props, "gcnArchName", "").split(":", 1)[0] != "gfx942":
        pytest.skip("requires gfx942")


def _case(q_lens, kv_lens, *, heads=24, kv_heads=2, magnitude=1.0):
    return common.make_case(
        q_lens, kv_lens, dq=256, dv=256, page=64, heads=heads,
        kv_heads=kv_heads, source_dtype=torch.bfloat16, magnitude=magnitude,
        poison_tail=True, seed=20260924,
    )


def _inputs(case, page=1, order=None, **options):
    """Convert independent logical KV, never the API's output, to linear storage."""
    keys, values = case.logical_kv()
    device, dtype = case.q.device, torch.bfloat16
    table = None
    if order is None:
        k, v = (torch.cat(parts).to(dtype).contiguous() if parts else
                torch.empty((0, case.kv_heads, 256), device=device, dtype=dtype)
                for parts in (keys, values))
    else:
        counts = [(n + page - 1) // page for n in case.kv_lens]
        ids = list(range(sum(counts)))
        if order == "random":
            random.Random(20260924).shuffle(ids)
            if len(ids) > 1 and ids == list(range(len(ids))):
                ids = ids[1:] + ids[:1]
        elif order == "reversed":
            ids.reverse()
        else:
            assert order == "contiguous"
        if order != "contiguous" and len(ids) > 1:
            assert ids != list(range(len(ids))), "exercise a real page permutation"
        k = torch.full((len(ids) * page, case.kv_heads, 256), float("nan"),
                       device=device, dtype=dtype)
        v = torch.full_like(k, float("nan"))
        width = max(counts, default=0)
        # Inactive columns may be invalid; physical tail rows remain poisoned.
        table = torch.full((len(counts), width + bool(width)), -1,
                           device=device, dtype=torch.int32)
        start = 0
        for batch, (length, count) in enumerate(zip(case.kv_lens, counts)):
            pages = torch.tensor(ids[start:start + count], device=device, dtype=torch.int64)
            table[batch, :count] = pages.to(torch.int32)
            logical = torch.arange(length, device=device)
            rows = pages[logical // page] * page + logical % page
            k[rows], v[rows] = keys[batch].to(dtype), values[batch].to(dtype)
            start += count
    arguments = dict(q=case.q, k=k, v=v, cu_seqlens_q=case.cq, cu_seqlens_k=case.ck,
                     max_seqlen_q=max(case.q_lens, default=0),
                     max_seqlen_k=max(case.kv_lens, default=0),
                     page_size=page, block_table=table)
    arguments.update(options)
    return arguments


def _assert_result(actual, reference, out=None):
    if isinstance(reference, tuple):
        assert isinstance(actual, tuple) and len(actual) == 2
        output, lse = actual
        reference, ref_lse = reference
        assert lse.dtype == torch.float32 and lse.is_contiguous()
        # The original oracle uses natural log, not the kernel's internal log2.
        torch.testing.assert_close(lse, ref_lse, rtol=0.002, atol=0.002)
    else:
        assert isinstance(actual, torch.Tensor)
        output = actual
    assert output.shape == reference.shape and output.device == reference.device
    assert output.dtype == torch.bfloat16 and output.is_contiguous()
    if out is not None:
        assert output is out
    if output.numel():
        common.accuracy(output, reference, "linear-d256", tolerance=0.02)
    # accuracy's scalar metric is undefined for empty tensors; shape/dtype above
    # and the exact-repeat checks below still apply to the full-empty path.


def _check(case, arguments):
    reference = common.torch_reference(
        case, arguments.get("causal", False), arguments.get("softmax_scale"),
        return_lse=arguments.get("return_lse", False),
    )
    actual = api.flash_attn_varlen_func(**arguments)
    out = arguments.get("out")
    _assert_result(actual, reference, out)
    saved = tuple(t.clone() for t in actual) if isinstance(actual, tuple) else actual.clone()
    for _ in range(2):
        if out is not None:
            out.fill_(float("nan"))
        repeated = api.flash_attn_varlen_func(**arguments)
        if out is not None:
            assert (repeated[0] if isinstance(repeated, tuple) else repeated) is out
        torch.testing.assert_close(repeated, saved, rtol=0, atol=0)
    return saved


def _guarded(tensor, padding=8):
    backing = torch.full((tensor.numel() + 2 * padding,), -123,
                         device=tensor.device, dtype=tensor.dtype)
    return backing, backing[padding:-padding].view_as(tensor)


def _no_launch():
    pytest.fail("invalid input reached the native core")


@pytest.mark.parametrize("page,order,persistent", [
    pytest.param(1, None, False, id="page1-linear-grid"),
    pytest.param(4, "random", True, id="page4-random-persistent"),
])
def test_kv_tails(gfx942, page, order, persistent):
    # One ragged launch covers every boundary, rather than a Cartesian matrix.
    tails = (1, 3, 4, 31, 32, 33, 63, 64, 65, 129)
    case = _case((65,) * len(tails), tails)
    _check(case, _inputs(case, page, order, persistent=persistent))


@pytest.mark.parametrize("page,order,persistent,heads,hk,causal,scale", [
    pytest.param(1, "random", False, 24, 2, False, None, id="page1-random-grid-h24-hk2"),
    pytest.param(1, "reversed", True, 6, 2, True, 0.125, id="page1-reversed-persistent-h6-hk2"),
    pytest.param(4, "random", True, 24, 2, True, None, id="page4-random-persistent-h24-hk2"),
    pytest.param(4, "reversed", False, 4, 4, False, 0.03125, id="page4-reversed-grid-mha"),
])
def test_paged_layouts(gfx942, page, order, persistent, heads, hk, causal, scale):
    case = _case((0, 7, 0, 129), (31, 33, 0, 193), heads=heads, kv_heads=hk)
    options = dict(persistent=persistent, causal=causal, softmax_scale=scale, return_lse=True)
    # Dense KV has 257 rows: page_size=4 must NOT require padding without a table.
    dense = _check(case, _inputs(case, page, **options))
    for layout in ("contiguous", order):
        paged = _check(case, _inputs(case, page, layout, **options))
        torch.testing.assert_close(paged[0], dense[0], rtol=0.02, atol=0.02)
        torch.testing.assert_close(paged[1], dense[1], rtol=0.002, atol=0.002)


@pytest.mark.parametrize("qlens,klens,page,order,persistent", [
    pytest.param((), (), 1, None, False, id="no-batches"),
    pytest.param((0, 0), (0, 0), 4, "reversed", True, id="fully-empty-paged"),
    pytest.param((0, 0), (3, 65), 1, None, True, id="zero-q-linear"),
    pytest.param((0, 0), (3, 65), 4, "random", False, id="zero-q-paged"),
])
def test_empty_queries(gfx942, qlens, klens, page, order, persistent):
    case = _case(qlens, klens)
    for return_lse in (False, True):
        _check(case, _inputs(case, page, order, persistent=persistent, return_lse=return_lse))


@pytest.mark.parametrize("page,persistent", [(1, False), (4, None)], ids=["grid", "default-persistent"])
def test_missing_ck_self_attention(gfx942, page, persistent):
    case = _case((0, 17, 33), (0, 17, 33), heads=4, kv_heads=1)
    arguments = _inputs(case, page, causal=True, persistent=persistent, return_lse=True)
    explicit = _check(case, arguments)
    arguments["cu_seqlens_k"] = None
    inferred = _check(case, arguments)
    torch.testing.assert_close(inferred, explicit, rtol=0, atol=0)
    if persistent is None:
        arguments["persistent"] = True
        torch.testing.assert_close(api.flash_attn_varlen_func(**arguments), inferred, rtol=0, atol=0)


@pytest.mark.parametrize("page,persistent", [(1, False), (4, True)], ids=["grid", "persistent"])
def test_guarded_out_reuse(gfx942, page, persistent):
    case = _case((129,), (65,))
    backing, out = _guarded(case.q)
    assert out.storage_offset() == 8 and out.data_ptr() % 16 == 0
    arguments = _inputs(case, page, "reversed", persistent=persistent, out=out)
    first = _check(case, arguments)
    # Reuse the very same output for changed data; update the independent Case
    # as well, including its old packed buffers (the oracle reads v_pages).
    case.v_pages.mul_(-2)
    case.k, case.v = common.vectorize_kv(case.k_pages, case.v_pages)
    arguments["v"].mul_(-2)
    arguments["return_lse"] = True
    changed, _ = _check(case, arguments)
    assert not torch.equal(first, changed)
    assert bool((backing[:8] == -123).all()) and bool((backing[-8:] == -123).all())


@pytest.mark.parametrize("page,persistent", [(1, False), (4, True)], ids=["grid", "persistent"])
def test_streams_and_graph(gfx942, page, persistent):
    case = _case((1537,), (193,))
    arguments = _inputs(case, page, "random", persistent=persistent, return_lse=True)
    outputs = [torch.empty_like(case.q) for _ in range(2)]
    expected = _check(case, dict(arguments, out=outputs[0]))  # Warm before capture.
    current = torch.cuda.current_stream(case.q.device)
    streams = [torch.cuda.Stream(device=case.q.device) for _ in outputs]
    snapshots = []
    for index, (stream, out) in enumerate(zip(streams, outputs)):
        stream.wait_stream(current)
        with torch.cuda.stream(stream):
            out.fill_(float("nan"))
        if index == 0:
            # Explicit stream differs from the caller's current stream.
            actual = api.flash_attn_varlen_func(**arguments, out=out, stream=stream)
        else:
            with torch.cuda.stream(stream):
                actual = api.flash_attn_varlen_func(**arguments, out=out)
        assert actual[0] is out and torch.cuda.current_stream(case.q.device) == current
        with torch.cuda.stream(stream):
            snapshots.append(tuple(t.clone() for t in actual))
    for stream, snapshot in zip(streams, snapshots):
        stream.synchronize()
        torch.testing.assert_close(snapshot, expected, rtol=0, atol=0)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=streams[1]):
        captured = api.flash_attn_varlen_func(**arguments, out=outputs[1])
    for _ in range(2):
        outputs[1].fill_(float("nan"))
        graph.replay()
        current.synchronize()
        torch.testing.assert_close(captured, expected, rtol=0, atol=0)
    case.q.neg_()  # Captured addresses must read new values, not cached outputs.
    reference = common.torch_reference(case, False, return_lse=True)
    graph.replay()
    current.synchronize()
    _assert_result(captured, reference, outputs[1])


@pytest.mark.parametrize("name,match", [
    ("cu_seqlens_q", "Bounds must start"),
    ("cu_seqlens_k", "Bounds must start"),
    ("block_table", "invalid active physical page IDs"),
])
def test_metadata_mutation_revalidation(gfx942, monkeypatch, name, match):
    case = _case((3, 5), (5, 7))
    arguments = _inputs(case, 4, "reversed", return_lse=True)
    _check(case, arguments)
    metadata = arguments[name]
    saved = metadata.clone()
    metadata.view(-1)[0] = -1  # Same identity/storage, new version, invalid data.
    with monkeypatch.context() as patch:
        patch.setattr(api, "_core_run", _no_launch)
        with pytest.raises(ValueError, match=match):
            api.flash_attn_varlen_func(**arguments)
    metadata.copy_(saved)
    _check(case, arguments)
    if name == "cu_seqlens_q":
        # A valid in-place repartition also changes the expected attention rows.
        case.cq[1] = 4
        case.q_lens = (4, 4)
        _check(case, arguments)


@pytest.mark.parametrize("change", ["fresh", "inplace"])
def test_graph_metadata_requires_rewarm(gfx942, change):
    case = _case((7,), (33,))
    arguments = _inputs(case, 4, "reversed", return_lse=True)
    _check(case, arguments)
    if change == "fresh":
        arguments["cu_seqlens_q"] = case.cq.clone()
    else:
        case.cq.add_(0)  # Identical values still invalidate the version receipt.
    stream = torch.cuda.Stream(device=case.q.device)
    stream.wait_stream(torch.cuda.current_stream(case.q.device))
    with torch.cuda.graph(torch.cuda.CUDAGraph(), stream=stream):
        with pytest.raises(RuntimeError, match="Unwarmed or changed metadata"):
            api.flash_attn_varlen_func(**arguments)
    expected = _check(case, arguments)  # Revalidation outside capture is legal.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actual = api.flash_attn_varlen_func(**arguments)
    for _ in range(2):
        graph.replay()
        torch.cuda.current_stream(case.q.device).synchronize()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("pattern,page,persistent,causal", [
    pytest.param("constructed", 1, False, False, id="constructed-grid"),
    pytest.param("constructed", 4, True, True, id="constructed-causal-persistent"),
    pytest.param("high-magnitude", 4, False, False, id="large-qk-grid"),
])
def test_rare_rescale(gfx942, pattern, page, persistent, causal):
    case = _case((129,), (193,), magnitude=8.0 if pattern == "high-magnitude" else 1.0)
    if pattern == "constructed":
        # Same construction as the old module's rare-rescale regression: later
        # BN64 tiles jump by 8 in natural-log logits, forcing the >7 log2 branch.
        case.q.fill_(0.25)
        for tile, physical in enumerate(case.page_order):
            logical = torch.arange(tile * 64, min((tile + 1) * 64, 193), device=case.q.device)
            count = logical.numel()
            case.k_pages[physical, :count] = (tile * 2 + (logical % 64 >= 32))[:, None, None]
            case.v_pages[physical, :count, :, :128] = (logical.float() / 193)[:, None, None]
            case.v_pages[physical, :count, :, 128:] = (-logical.float() / 97)[:, None, None]
        case.k, case.v = common.vectorize_kv(case.k_pages, case.v_pages)
    arguments = _inputs(case, page, "reversed", persistent=persistent, causal=causal)
    _check(case, arguments)
    _check(case, dict(arguments, return_lse=True))


@pytest.mark.parametrize("change,error,match", [
    pytest.param({name: value}, NotImplementedError, name, id=name)
    for name, value in (
        ("min_seqlen_q", 1), ("dropout_p", 0.1), ("logits_soft_cap", 1.0),
        ("deterministic", True), ("return_attn_probs", True), ("how_v3_bf16_cvt", 0),
        ("bias", 0), ("alibi_slopes", 0), ("sink_ptr", 0),
        ("cu_seqlens_q_padded", 0), ("cu_seqlens_k_padded", 0),
        ("window_size", (-1, -1, 1)), ("layout", "vectorized"),
        ("key_layout", "vectorized"), ("num_waves", 4), ("page_size", 2),
    )
] + [
    pytest.param({"softmax_scale": value}, ValueError, "softmax_scale", id=f"scale-{value}")
    for value in (0.0, -1.0, float("inf"), float("nan"), True, "bad")
] + [
    pytest.param({name: value}, ValueError, name, id=f"invalid-{name}-{value}")
    for name, value in (("causal", 1), ("return_lse", 1), ("persistent", 1),
                        ("max_seqlen_q", -1), ("max_seqlen_q", True),
                        ("max_seqlen_k", 2**31), ("max_seqlen_k", 1.5))
])
def test_invalid_options(monkeypatch, change, error, match):
    # All these options are rejected before device discovery or native import.
    monkeypatch.setattr(api, "_core_run", _no_launch)
    arguments = dict(q=None, k=None, v=None, cu_seqlens_q=None, cu_seqlens_k=None,
                     max_seqlen_q=0, max_seqlen_k=0)
    arguments.update(change)
    with pytest.raises(error, match=match):
        api.flash_attn_varlen_func(**arguments)


@pytest.mark.parametrize("change,error,match", [
    pytest.param(lambda a: dict(q=a["q"].cpu()), ValueError, "Q must be a device", id="cpu-q"),
    pytest.param(lambda a: dict(q=a["q"].float()), ValueError, "Q must be contiguous", id="q-dtype"),
    pytest.param(lambda a: dict(k=a["k"].transpose(0, 1)), ValueError, "K must be contiguous", id="k-strides"),
    pytest.param(lambda a: dict(v=a["v"].float()), ValueError, "V must be contiguous", id="v-dtype"),
    pytest.param(lambda a: dict(q=a["q"].flatten(0, 1)), NotImplementedError, "linear", id="rank"),
    pytest.param(lambda a: dict(q=a["q"][..., :128].contiguous()), NotImplementedError, "linear", id="dimension"),
    pytest.param(lambda a: dict(v=a["v"][:-1]), ValueError, "K/V shapes", id="kv-shape"),
    pytest.param(lambda a: dict(q=a["q"][:, :23].contiguous()), ValueError, "positive multiple", id="head-ratio"),
    pytest.param(lambda a: dict(q=a["q"][:, :0].contiguous()), ValueError, "positive multiple", id="zero-heads"),
    pytest.param(lambda a: dict(q=a["q"].detach().requires_grad_()), NotImplementedError, "autograd", id="grad"),
    pytest.param(lambda a: dict(out=a["q"][:1].clone()), ValueError, "out must match", id="out-shape"),
    pytest.param(lambda a: dict(out=a["q"].float()), ValueError, "out must be contiguous", id="out-dtype"),
    pytest.param(lambda a: dict(out=a["q"].transpose(0, 1)), ValueError, "out must be contiguous", id="out-strides"),
    pytest.param(lambda a: dict(stream=object()), ValueError, "stream must be", id="stream-type"),
] + [
    pytest.param(lambda a, name=name: {name: _guarded(a["q"] if name == "out" else a[name], 1)[1]},
                 ValueError, "16-byte", id=f"unaligned-{name}")
    for name in ("q", "k", "v", "out")
])
def test_invalid_tensors(gfx942, monkeypatch, change, error, match):
    arguments = _inputs(_case((3, 5), (5, 7)), 4, "reversed")
    monkeypatch.setattr(api, "_core_run", _no_launch)
    arguments.update(change(arguments))
    with pytest.raises(error, match=match):
        api.flash_attn_varlen_func(**arguments)


@pytest.mark.parametrize("change,match", [
    pytest.param(lambda a: dict(cu_seqlens_q=a["cu_seqlens_q"].long()), "cu_seqlens_q must", id="dtype"),
    pytest.param(lambda a: dict(cu_seqlens_q=a["cu_seqlens_q"].view(1, -1)), "equal 1D shapes", id="rank"),
    pytest.param(lambda a: dict(cu_seqlens_k=a["cu_seqlens_k"][:2]), "equal 1D shapes", id="batch-count"),
    pytest.param(lambda a: dict(cu_seqlens_q=a["cu_seqlens_q"][:0]), "equal 1D shapes", id="empty-bounds"),
    pytest.param(lambda a: dict(cu_seqlens_q=common.i32([1, 3, 8])), "Bounds must start", id="start"),
    pytest.param(lambda a: dict(cu_seqlens_q=common.i32([0, 3, 7])), "Bounds must start", id="end"),
    pytest.param(lambda a: dict(cu_seqlens_q=common.i32([0, 9, 8])), "monotonic", id="nonmonotonic"),
    pytest.param(lambda a: dict(max_seqlen_q=4), "maxima", id="q-maximum"),
    pytest.param(lambda a: dict(max_seqlen_k=6), "maxima", id="kv-maximum"),
    pytest.param(lambda a: dict(cu_seqlens_k=common.i32([0, 0, 12]), max_seqlen_k=12), "Active Q", id="empty-active-kv"),
    pytest.param(lambda a: dict(cu_seqlens_k=common.i32([0, 2, 12]), max_seqlen_k=10, causal=True), "KV >= Q", id="causal-short-kv"),
    pytest.param(lambda a: dict(block_table=a["block_table"].long()), "block_table must", id="table-dtype"),
    pytest.param(lambda a: dict(block_table=a["block_table"][:1]), "shape", id="table-batches"),
    pytest.param(lambda a: dict(block_table=a["block_table"][:, :1].contiguous()), "missing columns", id="table-width"),
    pytest.param(lambda a: dict(block_table=torch.full_like(a["block_table"], -1)), "invalid active", id="negative-page"),
    pytest.param(lambda a: dict(block_table=torch.full_like(a["block_table"], a["k"].shape[0] // 4)), "invalid active", id="past-last-page"),
    pytest.param(lambda a: dict(k=a["k"][:-1], v=a["v"][:-1]), "multiple of page_size", id="partial-physical-page"),
    pytest.param(lambda a: dict(block_table=None), "end at Q/K token counts", id="dense-k-end"),
    pytest.param(lambda a: dict(cu_seqlens_k=None), "equal-token self-attention", id="missing-ck-paged"),
    pytest.param(lambda a: dict(cu_seqlens_k=None, block_table=None), "equal-token self-attention", id="missing-ck-cross-attention"),
])
def test_invalid_metadata(gfx942, monkeypatch, change, match):
    arguments = _inputs(_case((3, 5), (5, 7)), 4, "reversed")
    monkeypatch.setattr(api, "_core_run", _no_launch)
    arguments.update(change(arguments))
    with pytest.raises(ValueError, match=match):
        api.flash_attn_varlen_func(**arguments)


def test_metadata_requires_version_counters(gfx942, monkeypatch):
    arguments = _inputs(_case((3,), (5,)))
    with torch.inference_mode():
        arguments["cu_seqlens_q"] = arguments["cu_seqlens_q"].clone()
    monkeypatch.setattr(api, "_core_run", _no_launch)
    with pytest.raises(ValueError, match="version counters"):
        api.flash_attn_varlen_func(**arguments)


@pytest.mark.parametrize("name", ["q", "k", "v", "cu_seqlens_q", "cu_seqlens_k", "block_table"])
def test_output_overlap(gfx942, monkeypatch, name):
    arguments = _inputs(_case((3, 5), (5, 7)), 4, "reversed")
    q, source = arguments["q"], arguments[name]
    if name in ("q", "k", "v"):
        backing = torch.empty(q.numel() + source.numel() + 8, device=q.device, dtype=q.dtype)
        arguments[name] = backing[:source.numel()].view_as(source).copy_(source)
        # Distinct aligned data pointers, but overlapping byte intervals.
        arguments["out"] = backing[8:8 + q.numel()].view_as(q)
    else:
        out = torch.empty_like(q)
        alias = out.view(-1).view(torch.int32)[:source.numel()].view_as(source)
        arguments[name] = alias.copy_(source)
        arguments["out"] = out
    monkeypatch.setattr(api, "_core_run", _no_launch)
    with pytest.raises(ValueError, match="out must not overlap"):
        api.flash_attn_varlen_func(**arguments)