"""D256 linear/page1/page4 correctness of flash_attn_varlen_func (BF16 gfx942)."""

import re

import pytest
import torch

from pyhip.ops.mha.flydsl.mha_pa_bf16_942 import flash_attn_varlen_func
from tests.ops.mha._mha import linear_inputs, require


@pytest.fixture(scope="module")
def device():
    return require("gfx942")


def _check(arguments, expected):
    actual = flash_attn_varlen_func(**arguments)
    assert actual[0].dtype == torch.bfloat16 and actual[0].is_contiguous()
    torch.testing.assert_close(actual[0].float(), expected[0], rtol=0.02, atol=0.02)
    torch.testing.assert_close(actual[1], expected[1], rtol=0.002, atol=0.002)
    return actual


def _guarded_out(arguments):
    """An output view with 8 guard elements of 123 on each side."""
    q = arguments["q"]
    storage = torch.full((q.numel() + 16,), 123.0, device=q.device, dtype=q.dtype)
    arguments["out"] = storage[8:-8].view_as(q)
    return storage


def _guards_intact(storage):
    return bool((storage[:8] == 123).all()) and bool((storage[-8:] == 123).all())


@pytest.mark.parametrize("page,persistent,heads,hk,causal", [
    (None, True, 12, 1, True), (1, False, 6, 2, False), (4, True, 24, 2, True),
])
def test_layout_tail_and_lse(device, page, persistent, heads, hk, causal):
    args, reference = linear_inputs(device, page=page, persistent=persistent, heads=heads,
                                    kv_heads=hk, causal=causal, scale=0.5 if page == 1 else 0.0625)
    first = _check(args, reference)
    repeated = _check(args, reference)
    torch.testing.assert_close(first, repeated, rtol=0, atol=0)
    no_lse = flash_attn_varlen_func(**(args | {"return_lse": False}))
    torch.testing.assert_close(no_lse.float(), reference[0], rtol=0.02, atol=0.02)


def test_empty_self_attention_and_guards(device):
    args, reference = linear_inputs(device, qlens=(0, 0), klens=(3, 65))
    _check(args, reference)
    args, reference = linear_inputs(device, qlens=(0, 17, 33), klens=(0, 17, 33), heads=3)
    args["cu_seqlens_k"] = None
    storage = _guarded_out(args)
    first = _check(args, reference)
    assert first[0] is args["out"] and _guards_intact(storage)


@pytest.mark.parametrize("length,heads,hk", ((129, 6, 2), (2048, 12, 1), (2051, 12, 1)))
def test_causal_balance(device, length, heads, hk):
    """Balanced B1 causal tasks retain LSE, partial blocks, output guards and graph replay."""
    args, expected = linear_inputs(device, qlens=(length,), klens=(length,), heads=heads,
                                   kv_heads=hk, causal=True, persistent=True)
    q = args["q"]
    storage = _guarded_out(args)
    result = _check(args, expected)
    saved = result[0].clone()
    torch.testing.assert_close(_check(args, expected)[0], saved, rtol=0, atol=0)
    stream = torch.cuda.Stream(device=q.device)
    stream.wait_stream(torch.cuda.current_stream(q.device))
    no_lse = args | {"return_lse": False}
    with torch.cuda.stream(stream):
        flash_attn_varlen_func(**no_lse)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            flash_attn_varlen_func(**no_lse)
    torch.cuda.current_stream(q.device).wait_stream(stream)
    args["out"].fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize(q.device)
    torch.testing.assert_close(args["out"].float(), expected[0], rtol=0.02, atol=0.02)
    assert _guards_intact(storage)


@pytest.mark.parametrize("length", (1, 31, 32, 33, 63, 64, 65, 95, 96, 97, 127, 128, 129, 257, 2583))
def test_single_request_tail(device, length):
    """Single-request static bounds and N32 drain retain all heads/rows and guards."""
    for persistent in (False, True):
        args, expected = linear_inputs(device, qlens=(129,), klens=(length,), heads=6, kv_heads=2,
                                       causal=False, persistent=persistent,
                                       magnitude=3.0 if length == 2583 else 1.0)
        q = args["q"]
        storage = _guarded_out(args)
        actual = flash_attn_varlen_func(**(args | {"return_lse": False}))
        torch.testing.assert_close(actual.float(), expected[0], rtol=0.02, atol=0.02)
        saved = actual.clone()
        flash_attn_varlen_func(**(args | {"return_lse": False}))
        torch.testing.assert_close(actual, saved, rtol=0, atol=0)
        assert _guards_intact(storage)
        if length == 95:
            stream = torch.cuda.Stream(device=q.device)
            stream.wait_stream(torch.cuda.current_stream(q.device))
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                flash_attn_varlen_func(**(args | {"return_lse": False}))
            graph.replay()
            torch.cuda.synchronize(q.device)
            torch.testing.assert_close(actual, saved, rtol=0, atol=0)


def test_streams_graph_and_metadata_invalidation(device):
    args, reference = linear_inputs(device, page=4)
    current = torch.cuda.current_stream()
    streams = [torch.cuda.Stream() for _ in range(2)]
    outputs = [torch.empty_like(args["q"]) for _ in streams]
    for stream, out in zip(streams, outputs):
        stream.wait_stream(current)
        result = flash_attn_varlen_func(**args, out=out, stream=stream)
        current.wait_stream(stream)
        torch.testing.assert_close(result[0].float(), reference[0], rtol=0.02, atol=0.02)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=streams[0]):
        result = flash_attn_varlen_func(**args, out=outputs[0])
    for _ in range(2):
        graph.replay()
        current.synchronize()
        torch.testing.assert_close(result[0].float(), reference[0], rtol=0.02, atol=0.02)
    original = args["block_table"].clone()
    args["block_table"][0, 0] = -1
    with pytest.raises(ValueError, match="invalid active"):
        flash_attn_varlen_func(**args)
    args["block_table"].copy_(original)
    _check(args, reference)
    args["cu_seqlens_q"].add_(0)
    with torch.cuda.graph(torch.cuda.CUDAGraph(), stream=streams[1]):
        with pytest.raises(RuntimeError, match="Unwarmed or changed"):
            flash_attn_varlen_func(**args)


@pytest.mark.parametrize("option", ({"dropout_p": 0.1}, {"num_waves": 4}, {"page_size": 2}, {"bias": 0}))
def test_unsupported_options(option):
    args = dict(q=None, k=None, v=None, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0)
    with pytest.raises(NotImplementedError):
        flash_attn_varlen_func(**(args | option))


def test_invalid_scale_metadata_and_alias(device):
    args, _ = linear_inputs(device, page=4, reference=False)
    for scale in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="softmax_scale"):
            flash_attn_varlen_func(**(args | {"softmax_scale": scale}))
    with pytest.raises(ValueError, match="overlap"):
        flash_attn_varlen_func(**(args | {"out": args["q"]}))
    backing = torch.empty(args["q"].numel() + args["k"].numel(), device=device, dtype=torch.bfloat16)
    key_alias = backing[:args["k"].numel()].view_as(args["k"]).copy_(args["k"])
    out_alias = backing[8:8 + args["q"].numel()].view_as(args["q"])
    with pytest.raises(ValueError, match="overlap"):
        flash_attn_varlen_func(**(args | {"k": key_alias, "out": out_alias}))
    with pytest.raises(ValueError, match="maxima"):
        flash_attn_varlen_func(**(args | {"max_seqlen_q": 1}))
    with pytest.raises(ValueError, match="contiguous"):
        flash_attn_varlen_func(**(args | {"q": args["q"].float()}))
    with pytest.raises(NotImplementedError, match="autograd"):
        flash_attn_varlen_func(**(args | {"q": args["q"].detach().requires_grad_()}))


@pytest.fixture(scope="module", autouse=True)
def no_scratch_or_spills():
    """Every D256 linear specialization compiled by this module uses no scratch and no spills."""
    yield
    if not torch.cuda.is_initialized():
        return
    from pyhip.ops.mha.flydsl import mha_pa_bf16_256_linear_942 as core

    for compiled in core._COMPILED.values():
        for field in ("private_segment_fixed_size", "vgpr_spill_count", "sgpr_spill_count"):
            values = re.findall(rf"\b{field}\s*=\s*(\d+)", compiled._keepalive.ir)
            assert values and not any(map(int, values)), field
