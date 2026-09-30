# Copyright 2023-2024 SGLang Team
# SPDX-License-Identifier: Apache-2.0
"""Basic QSA calls, selected-token accuracy, input errors and graph updates."""

import pytest
import torch

from tests.ops.qsa._attention import _base, _call, _gpu, _make_case, _runtime, reference


def _check(value):
    storage = torch.full((len(value.q) + 2, *value.q.shape[1:]), 123.0,
                         dtype=value.q.dtype, device=value.q.device)
    output = storage[1:-1]
    output.fill_(float("nan"))
    assert _call(value, output) is output
    expected = reference(value, list(range(len(value.q))))
    torch.testing.assert_close(output.float(), expected, rtol=.02, atol=.02)
    torch.testing.assert_close(output.float(), _base(value), rtol=.02, atol=.02)
    first = output.clone()
    output.fill_(float("nan"))
    _call(value, output)
    torch.testing.assert_close(output, first, rtol=0, atol=0)
    assert bool((storage[[0, -1]] == 123).all())
    return output


@pytest.mark.parametrize("queries,prefixes,heads,hk,shared", [
    pytest.param((0,), (0,), 12, 1, False, id="empty"),
    pytest.param((1,), (0,), 3, 1, False, id="one-dense"),
    pytest.param((65,), (0,), 6, 1, False, id="short-dense-tail"),
    pytest.param((16,), (30000,), 12, 1, False, id="sparse-packed"),
    pytest.param((17,), (30000,), 3, 1, False, id="sparse-raw"),
    pytest.param((32,), (30000,), 6, 1, True, id="sparse-shared"),
    pytest.param((9,), (2047,), 12, 2, False, id="prefix-boundary-hk2"),
    pytest.param((7, 0, 9), (0, 5, 3000), 6, 2, True, id="ragged-hk2"),
])
def test_attention(queries, prefixes, heads, hk, shared):
    value = _make_case(queries, prefixes, heads, _gpu(), shared=shared)
    for name in ("k", "v"):
        original = getattr(value, name)
        if hk == 2:
            original = torch.cat((original, -original), dim=1)
        storage = torch.full((len(original) + 4, hk, 256), float("nan"),
                             dtype=original.dtype, device=original.device)
        storage[:len(original)].copy_(original)
        setattr(value, name, storage[:len(original)])
    _check(value)


def test_output_alias_and_causal_nan_tail():
    value = _make_case((8,), (30000,), 12, _gpu())
    with pytest.raises(ValueError, match="overlap"):
        _call(value, value.q)
    # Unselected lanes of the last packed block must not affect causal tails.
    value.k[30005:].fill_(float("nan"))
    value.v[30005:].fill_(float("nan"))
    output = _call(value)
    torch.testing.assert_close(output[:5].float(), reference(value, list(range(5))), rtol=.02, atol=.02)


@pytest.mark.parametrize("lengths,prefixes", [
    ((12, 16, 20), (3000, 9000, 9004)),
    ((13, 17, 21), (3000, 9000, 9004)),
    ((1, 65, 129), (0, 0, 0)),
    ((64, 128, 192), (0, 0, 0)),
    ((12, 13, 16), (3000, 9000, 9004)),
    ((13, 16, 17), (3000, 9000, 9004)),
    ((64, 65, 128), (0, 0, 0)),
], ids=("packed", "raw", "dense-tail", "dense-aligned", "packed-raw-switch",
        "raw-packed-switch", "dense-alignment-switch"))
def test_runtime_lengths_reuse_compilation(lengths, prefixes):
    from pyhip.ops.qsa.flydsl import attention_direct_packed

    device = _gpu()
    counts = []
    for rows, prefix in zip(lengths, prefixes):
        value = _make_case((rows,), (prefix,), 12, device)
        _check(value)
        counts.append((len(attention_direct_packed._COMPILED), len(_runtime.direct._COMPILED),
                       len(_runtime.union._COMPILED), len(_runtime.dense._BOUNDED_COMPILED)))
    assert counts == [counts[0]] * len(counts), counts


def test_invalid_selection_error_buffer():
    """Inspect errors without executing the public asynchronous device assert."""
    value = _make_case((5,), (30000,), 12, _gpu())
    workspace = _runtime._Workspace(value.q, value.k, value.v, value.indices,
                                    value.query_lens, value.prefix_lens, value.scale)
    inputs = workspace.bind(value.q, value.k, value.v, value.indices)
    for corruption in ("valid", "duplicate", "block", "tail", "padding", "out-of-range"):
        indices = value.indices.clone()
        if corruption == "duplicate":
            indices[0, 4:8].copy_(indices[0, :4])
        elif corruption == "block":
            indices[0, 1].add_(1)
        elif corruption == "tail":
            indices[0, 2048] = 29999
        elif corruption == "padding":
            indices[0, 2049] = 0
        elif corruption == "out-of-range":
            indices[0, :4] = torch.arange(30004, 30008, dtype=torch.int32, device=value.q.device)
        _runtime.prepare.attention_recover_scatter[(5,)](
            indices, inputs.query_positions, inputs.kv_lens, inputs.query_sequence_ids,
            inputs.block_indices, workspace.errors, num_warps=1,
        )
        _runtime.prepare.attention_order_masks_validate[(1,)](
            workspace.errors, workspace.valid, ROWS=5, num_warps=4,
        )
        assert workspace.errors.cpu().tolist() == [int(corruption != "valid"), 0, 0, 0, 0]
        assert bool(workspace.valid.cpu()) == (corruption == "valid")


def test_graph_selection_and_kv_update():
    device = _gpu()
    value = _make_case((32,), (30000,), 6, device)
    independent = value.indices.clone()
    shared = _make_case((32,), (30000,), 6, device, seed=29, shared=True).indices
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):
        output = _call(value)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            _call(value, output)
    torch.cuda.current_stream(device).wait_stream(stream)
    for selection in (shared, independent):
        value.indices.copy_(selection)
        value.k.mul_(.75)
        value.v.neg_()
        output.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize(device)
        torch.testing.assert_close(output.float(), reference(value, list(range(32))), rtol=.02, atol=.02)
        first = output.clone()
        output.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize(device)
        torch.testing.assert_close(output, first, rtol=0, atol=0)
