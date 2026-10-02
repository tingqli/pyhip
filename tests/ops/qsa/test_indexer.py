"""Public QSA indexer correctness with synthetic inputs and independent references."""

import os
from pathlib import Path
import subprocess
import sys

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


def test_prefill_position_mismatch_traps(device):
    """A query position off the host prefix+index layout stops the GPU queue in top-k, which aborts the process."""
    script = ("import torch\n"
              "from tests.ops.qsa import _indexer as helpers\n"
              "source = helpers.synthetic((2177,), (129,), helpers._gpu())\n"
              "source.inputs['logical_positions'][5] += 1\n"
              "helpers.indexer.prefill_indexer(**source.inputs)\n"
              "torch.cuda.synchronize()\n"
              "print('no trap')\n")
    # The trap is deliberate: skip ROCr's GPU core dump file.
    result = subprocess.run([sys.executable, "-c", script], cwd=Path(__file__).resolve().parents[3],
                            capture_output=True, text=True, errors="replace", timeout=900,
                            env=dict(os.environ, HSA_DISABLE_COREDUMP_ON_EXCEPTION="1"))
    assert result.returncode != 0 and "no trap" not in result.stdout
    assert "qsa_indexer_topk" in result.stderr


def test_prep_kernels_compile_once(device):
    """Row counts, group counts, position strides and decode batches of any value (1, multiples of
    16, others) reuse one compiled q_prep, k_compress and decode_prep; a first prefill whose rows
    need no logits still compiles the logits kernel."""
    kernels = (helpers.indexer._indexer_q_prep, helpers.indexer._indexer_k_compress,
               helpers.indexer._indexer_decode_prep)

    def counts():
        return [len(kernel.device_caches[device.index][0]) for kernel in kernels]

    helpers.indexer.indexer_logits._COMPILED.pop(device, None)
    helpers.indexer.prefill_indexer(**helpers.synthetic((9,), (9,), device).inputs)
    assert device in helpers.indexer.indexer_logits._COMPILED
    helpers.indexer.decode_forward(**helpers.decode_forward_case((7,), device, context=64).inputs)
    first = counts()
    for seq_lens, extend_lens in (((1,), (1,)), ((64,), (64,)), ((60,), (60,)), ((2177,), (129,)),
                                  ((4096, 63), (4096, 63))):
        helpers.indexer.prefill_indexer(**helpers.synthetic(seq_lens, extend_lens, device).inputs)
    for rows in (2, 16, 32):
        lengths = tuple(range(5, 5 + rows))
        helpers.indexer.decode_forward(**helpers.decode_forward_case(lengths, device, context=64).inputs)
    assert counts() == first


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


@pytest.mark.parametrize("pattern", ("random", "ties", "dense-threshold", "nan", "inf"))
def test_decode_topk_matches_row_kernel(pattern, device):
    """The CTA-per-row decode top-k reproduces the one-wave prefill row kernel bit for bit."""
    from pyhip.ops.qsa.flydsl import indexer_topk

    counts = (20000, 600, 65536, 513)
    stride = 65536 + 512
    generator = torch.Generator(device=device).manual_seed(11)
    logits = torch.randn((len(counts), stride), generator=generator, device=device)
    if pattern == "ties":
        logits.fill_(1.0)
    elif pattern == "dense-threshold":
        logits = (logits * 2).round() / 2
    elif pattern == "nan":
        logits[:, 7::997] = float("nan")
    elif pattern == "inf":
        logits[:, 3::1013] = float("inf")
        logits[:, 5::1009] = float("-inf")
    lengths = torch.tensor(counts, dtype=torch.int32, device=device)
    positions = lengths.long() * 4 + 2
    sequences = (positions + 1).int()
    decode = torch.full((len(counts), 2051), -7, dtype=torch.int32, device=device)
    indexer_topk.launch_decode(logits, lengths, positions, sequences, decode)
    row_info = torch.stack((positions.int(), sequences), dim=1).contiguous()
    rows = torch.full_like(decode, -7)
    # The row kernel takes each row's (min bits, ~max bits) from the logits kernel; NaN orders above +inf.
    causal = torch.arange(stride, device=device)[None] < lengths[:, None]
    finite = causal & ~logits.isnan()
    high = torch.where(finite, logits, float("-inf")).amax(1).view(torch.int32)
    high = torch.where((causal & logits.isnan()).any(1), 0x7FC00000, high)
    stats = torch.stack((torch.where(finite, logits, float("inf")).amin(1).view(torch.int32), ~high),
                        dim=1).contiguous()
    indexer_topk.launch(logits, stride, 0, len(counts), positions, row_info, rows, stats)
    torch.cuda.synchronize(device)
    assert torch.equal(decode, rows)
