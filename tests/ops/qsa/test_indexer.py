"""Public QSA indexer correctness with synthetic inputs and independent references."""

import pytest
import torch

from tests.ops.qsa import _indexer as helpers


# Sequence/extend lengths cover short rows, the 512-key boundary and packed prefixes.
PREFILL_CASES = (
    ((1,), (1,)),
    ((7,), (7,)),
    ((2051,), (2051,)),
    ((2052,), (2052,)),
    ((2177,), (129,)),
    ((2055, 16, 13), (7, 0, 9)),
)
# Compressed-key lengths, page-table width and heads; zero rows model decode padding.
DECODE_CASES = (
    ((0, 1, 15, 16, 17), 4, 4),
    ((511, 512, 513), 64, 4),
    ((513,), 33, 4),
    ((3000,), 256, 4),
    (tuple(3000 + 37 * i for i in range(29)) + (0, 0, 0), 256, 8),
    ((65536,), 4096, 4),
)
DECODE_FORWARD_CASES = (
    ((1,), 0, 64),
    ((5, 8, 3, 12), 1, 256),
    ((12000, 11888, 11667, 11851), 0, 16384),
)


@pytest.fixture
def device():
    return helpers._gpu()


def _addresses(source):
    return {name: tensor.data_ptr() for name, tensor in source.inputs.items()
            if isinstance(tensor, torch.Tensor)}


def _check_prefill(source):
    expected = helpers.reference_prep(source.inputs, source.state)
    output = helpers.indexer.prefill_indexer(**source.inputs)
    helpers.check(source, actual=output, expected=expected)
    return output


@pytest.mark.parametrize("seq_lens,extend_lens", PREFILL_CASES,
                         ids=("short", "tail", "topk-limit", "topk-over", "prefix", "multi"))
def test_prefill_operator(seq_lens, extend_lens, device):
    _check_prefill(helpers.synthetic(seq_lens, extend_lens, device))


def test_prefill_frame_and_output_ownership(device):
    source = helpers.synthetic((2055, 13), (7, 9), device, cache_dtype=torch.float32)
    # The public ABI allows a noncontiguous RoPE axis stride, but unit token stride.
    storage = torch.empty((3, source.rows + 5), dtype=torch.int64, device=device)
    storage[:, :source.rows].copy_(source.inputs["positions"])
    source.inputs["positions"] = storage[:, :source.rows]
    output = _check_prefill(source)
    saved = output.clone()
    helpers.reset_state(source)
    source.inputs["qk"].neg_()
    later = _check_prefill(source)
    assert later.data_ptr() != output.data_ptr()
    helpers.assert_exact(output, saved, "a later call must not overwrite an earlier result")


@pytest.mark.parametrize("lengths,pages,heads", DECODE_CASES,
                         ids=("short-pages", "topk-boundary", "compact-pages", "single", "ragged-padded", "max-keys"))
def test_decode_operator(lengths, pages, heads, device):
    source = helpers.decode_case(lengths, pages, device, heads=heads)
    output = helpers.indexer.decode_indexer(**source.inputs)
    helpers.check_decode(source, output)


def test_decode_graph_replay(device):
    source = helpers.decode_case((3000, 0, 700, 4096), 256, device, pool=600)
    helpers.indexer.decode_indexer(**source.inputs)
    torch.cuda.synchronize(device)
    addresses = _addresses(source)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = helpers.indexer.decode_indexer(**source.inputs)
    lengths = (513, 512, 0, 4095)
    fresh = helpers.decode_case(lengths, 256, device, seed=19, pool=600)
    for name, tensor in source.inputs.items():
        tensor.copy_(fresh.inputs[name])
    source.host["compressed_lengths"] = lengths
    graph.replay()
    helpers.check_decode(source, output)
    assert _addresses(source) == addresses


@pytest.mark.parametrize("lengths,padding,context", DECODE_FORWARD_CASES,
                         ids=("short", "mixed-padded", "long-ragged"))
def test_decode_forward_operator(lengths, padding, context, device):
    source = helpers.decode_forward_case(lengths, device, padding=padding, context=context)
    expected = helpers.reference_prep(source.inputs, source.state, decode=True)
    output = helpers.indexer.decode_forward(**source.inputs)
    helpers.check_decode_forward(source, output, expected=expected)


def test_decode_forward_graph_replay(device):
    lengths = (4003, 6, 1023, 2)
    source = helpers.decode_forward_case(lengths, device, padding=2, context=8192)
    expected = helpers.reference_prep(source.inputs, source.state, decode=True)
    warm = helpers.indexer.decode_forward(**source.inputs)
    helpers.check_decode_forward(source, warm, expected=expected)
    torch.cuda.synchronize(device)
    helpers.reset_state(source)
    addresses = _addresses(source)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = helpers.indexer.decode_forward(**source.inputs)
    # Advance only independent history before crossing the next compression boundary.
    source.state = expected.state
    helpers.reset_state(source)
    helpers.decode_step(source, tuple(length + 1 for length in lengths))
    expected = helpers.reference_prep(source.inputs, source.state, decode=True)
    graph.replay()
    helpers.check_decode_forward(source, output, expected=expected)
    assert _addresses(source) == addresses
