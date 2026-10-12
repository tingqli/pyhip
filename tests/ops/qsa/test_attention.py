# Copyright 2023-2024 SGLang Team
# SPDX-License-Identifier: Apache-2.0
"""Basic QSA calls, selected-token accuracy, input errors and graph updates."""

from collections import OrderedDict
import math
import os
import subprocess
import sys

import pytest
import torch
import triton
import triton.language as tl

from pyhip.ops.qsa.flydsl.attention_prepare import _route
from tests.ops.qsa._attention import ROOT, _base, _call, _gpu, _make_case, _runtime, reference


@pytest.fixture(params=(None, 0), ids=("default-plan", "always-plan"))
def union_min_rows(request, monkeypatch):
    """Run with the production threshold and with union planning forced for small inputs."""
    if request.param is not None:
        monkeypatch.setattr(_runtime, "UNION_MIN_ROWS", request.param)
        monkeypatch.setattr(_runtime, "UNION_MIN_ROWS_PER_HEAD", request.param)
    monkeypatch.setattr(_runtime, "_workspaces", OrderedDict())
    return _runtime.UNION_MIN_ROWS


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
    pytest.param((1,), (0,), 3, 1, False, id="one-row"),
    pytest.param((65,), (0,), 6, 1, False, id="short-causal-tail"),
    pytest.param((16,), (30000,), 12, 1, False, id="sparse-packed"),
    pytest.param((17,), (30000,), 3, 1, False, id="sparse-nonmultiple"),
    pytest.param((32,), (30000,), 6, 1, True, id="sparse-shared"),
    pytest.param((9,), (2047,), 12, 2, False, id="prefix-boundary-hk2"),
    pytest.param((7, 0, 9), (0, 5, 3000), 6, 2, True, id="ragged-hk2"),
])
def test_attention(queries, prefixes, heads, hk, shared, union_min_rows):
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


@pytest.mark.parametrize("queries,prefixes,hk", [
    pytest.param((1, 2, 3), (0, 0, 0), 1, id="kv-below-four"),
    pytest.param((3, 5, 2), (0, 2, 4097), 2, id="partial-hk2"),
    pytest.param((61,), (30002,), 1, id="partial-long"),
])
def test_union_partial_final_blocks(queries, prefixes, hk, monkeypatch):
    """Every tile on union: a request's partial final block (KV of 1-3 tokens or not a multiple of 4)
    reads only that request, so the NaN tokens after the last request never reach the output."""
    allocate = _runtime.prepare.allocate_plan

    def forced(**kwargs):
        plan = allocate(**kwargs)
        plan.union_ratio, plan.routing = math.inf, False
        return plan

    monkeypatch.setattr(_runtime.prepare, "allocate_plan", forced)
    monkeypatch.setattr(_runtime, "UNION_MIN_ROWS", 0)
    monkeypatch.setattr(_runtime, "UNION_MIN_ROWS_PER_HEAD", 0)
    monkeypatch.setattr(_runtime, "_workspaces", OrderedDict())
    value = _make_case(queries, prefixes, 6, _gpu())
    for name in ("k", "v"):
        original = getattr(value, name)
        if hk == 2:
            original = torch.cat((original, -original), dim=1)
        storage = torch.full((len(original) + 4, hk, 256), float("nan"),
                             dtype=original.dtype, device=original.device)
        storage[:len(original)].copy_(original)
        setattr(value, name, storage[:len(original)])
    _check(value)
    assert bool(next(reversed(_runtime._workspaces.values())).union.active.all())


def test_first_call_compiles_every_variant(monkeypatch):
    """The first call per head shape compiles every variant it can use (no plan, each union tile
    incl. BQ21/BQ42), so later layouts compile no K1-K3 variant or union/direct family."""
    monkeypatch.setattr(_runtime, "_workspaces", OrderedDict())
    monkeypatch.setattr(_runtime, "_warmed", set())
    device = _gpu()
    cus = torch.cuda.get_device_properties(device).multi_processor_count
    kernels = [getattr(_runtime.prepare, name) for name in
               ("attention_recover_scatter", "attention_compact", "attention_order_masks")]

    def counts():
        return ([len(kernel.device_caches[device.index][0]) for kernel in kernels],
                len(_runtime.union._COMPILED), len(_runtime.direct._COMPILED))

    for heads, full in ((12, 768), (6, 21 * cus * 2), (3, 42 * cus * 2)):
        _call(_make_case((1,), (0,), heads, device))
        first = counts()
        for queries, prefixes in (((full,), (0,)), ((800, 700), (0, 3000)), ((1024,), (30000,)), ((9,), (0,))):
            _call(_make_case(queries, prefixes, heads, device))
            assert counts() == first, (heads, queries, prefixes)


@pytest.mark.parametrize("lengths,prefixes", [
    ((12, 16, 20), (3000, 9000, 9004)),
    ((13, 17, 21), (3000, 9000, 9004)),
    ((1, 65, 129), (0, 0, 0)),
    ((64, 128, 192), (0, 0, 0)),
    ((12, 13, 16), (3000, 9000, 9004)),
    ((13, 16, 17), (3000, 9000, 9004)),
    ((64, 65, 128), (0, 0, 0)),
], ids=("aligned", "nonmultiple", "causal-tail", "causal-aligned", "aligned-nonmultiple-switch",
        "nonmultiple-aligned-switch", "causal-alignment-switch"))
def test_runtime_lengths_reuse_compilation(lengths, prefixes, union_min_rows):
    device = _gpu()
    counts = []
    for rows, prefix in zip(lengths, prefixes):
        value = _make_case((rows,), (prefix,), 12, device)
        _check(value)
        counts.append((len(_runtime.direct._COMPILED), len(_runtime.union._COMPILED)))
    assert counts == [counts[0]] * len(counts), counts


def test_prepare_compiles_once(monkeypatch):
    """Task counts (sort widths 128-4096, two sort CTAs), BQ16/BQ21 and KV scan widths reuse K1-K3."""
    monkeypatch.setattr(_runtime, "UNION_MIN_ROWS", 0)
    monkeypatch.setattr(_runtime, "UNION_MIN_ROWS_PER_HEAD", 0)
    monkeypatch.setattr(_runtime, "_workspaces", OrderedDict())
    device = _gpu()
    kernels = [getattr(_runtime.prepare, name) for name in
               ("attention_recover_scatter", "attention_compact", "attention_order_masks")]
    layouts = [((1,) * n, (0,) * (n - 1) + (prefix,), 12)
               for n, prefix in ((100, 0), (700, 1500), (1500, 3000), (2100, 0), (4200, 0))]
    layouts += [((3400,), (0,), 6), ((64, 64), (0, 0), 6)]
    counts = []
    for queries, prefixes, heads in layouts:
        value = _make_case(queries, prefixes, heads, device)
        output = _call(value)
        rows = [0, len(value.q) - 1]
        torch.testing.assert_close(output[rows].float(), reference(value, rows), rtol=.02, atol=.02)
        counts.append([len(kernel.device_caches[device.index][0]) for kernel in kernels])
    assert counts == [counts[0]] * len(counts), counts


def test_raw_direct_over_budget(monkeypatch):
    monkeypatch.setattr(_runtime, "_workspaces", OrderedDict())
    device = _gpu()
    _check(_make_case((13,), (3000,), 6, device))
    counts = len(_runtime.direct._COMPILED)
    monkeypatch.setattr(_runtime.direct, "MAX_PACKED_KV_BYTES", 0)
    for queries, prefixes in (((17,), (30000,)), ((7, 0, 9), (0, 5, 3000))):
        _check(_make_case(queries, prefixes, 6, device, shared=True))
        assert next(reversed(_runtime._workspaces.values())).direct.packed_key is None
    assert len(_runtime.direct._COMPILED) == counts


@triton.jit
def _route_probe(Counts, Costs, Result, TILES, TASK_SHARE, DIRECT_STEP, DIRECT_FIXED):
    tl.store(Result, _route(Counts, Costs, TILES, TASK_SHARE, DIRECT_STEP, DIRECT_FIXED, 64, 8))


@pytest.mark.parametrize("proposal,blocks,share,expected", [
    # Six long union tasks would each hold one CU far longer than the direct batch.
    pytest.param([1] * 6 + [0] * 26, [3500] * 6 + [3600] * 26, 1 / 32, [0] * 32, id="latency-direct"),
    # Four long union tasks set the union time; the short ones stay union, the long ones go direct.
    pytest.param([1] * 64, [1920] * 4 + [160] * 60, 1 / 80, [0] * 4 + [1] * 60, id="latency-demote"),
    pytest.param([1] * 600 + [0] * 600, [640] * 600 + [5120] * 600, 1 / 80, [1] * 600 + [0] * 600,
                 id="throughput-keep"),
])
def test_route_guard(proposal, blocks, share, expected):
    device = _gpu()
    tiles = len(proposal)
    counts = torch.tensor(blocks, dtype=torch.int32, device=device)
    costs = torch.tensor([[650, 65, p] for p in proposal], dtype=torch.int32, device=device)
    result = torch.empty(1, dtype=torch.int32, device=device)
    _route_probe[(1,)](counts, costs, result, tiles, share, 0.0038, 50.0, num_warps=4)
    threshold = int(result.item())
    steps = [-(-b // 16) for b in blocks]
    assert [int(p == 1 and s <= threshold) for p, s in zip(proposal, steps)] == expected, threshold


def test_route_keeps_direct_tail(monkeypatch):
    # Tiles just above the per-tile ratio stay direct: union and gated direct write one output.
    monkeypatch.setattr(_runtime, "_workspaces", OrderedDict())
    _check(_make_case((1024,), (30000,), 12, _gpu(), shared=True))
    active = next(reversed(_runtime._workspaces.values())).union.active
    assert bool((active == 1).any()) and bool((active == 0).any())


@pytest.mark.parametrize("ordered", (True, False), ids=("ascending", "unordered"))
def test_row_order(ordered, monkeypatch):
    """Rows in ascending (PyHIP indexer) or any other order stay exact."""
    monkeypatch.setattr(_runtime, "_workspaces", OrderedDict())
    value = _make_case((1024,), (30000,), 6, _gpu(), shared=True)
    if ordered:
        blocks = value.indices[:, :2048].view(-1, 512, 4)
        order = torch.where(blocks[:, :, 0] >= 0, blocks[:, :, 0], 1 << 30).argsort(dim=1)
        value.indices[:, :2048] = blocks.gather(1, order[:, :, None].expand(-1, -1, 4)).reshape(-1, 2048)
    _check(value)


def test_h3_full_prefill_bq42(monkeypatch):
    """Full single-request H3 prefill uses 42-row union tiles with 64-bit membership."""
    monkeypatch.setattr(_runtime, "_workspaces", OrderedDict())
    device = _gpu()
    rows = 42 * torch.cuda.get_device_properties(device).multi_processor_count * 2
    value = _make_case((rows,), (0,), 3, device)
    output = _call(value)
    plan = next(reversed(_runtime._workspaces.values())).union
    assert plan.query_tile == 42 and plan.membership.dtype == torch.int64
    picks = sorted({0, 41, 42, 2047, 2051, rows - 1, *range(0, rows, rows // 61)})
    torch.testing.assert_close(output[picks].float(), reference(value, picks), rtol=.02, atol=.02)
    torch.testing.assert_close(_call(value), output, rtol=0, atol=0)


def test_union_plan_threshold(monkeypatch):
    for heads, threshold in ((12, 768), (6, 384), (3, 384)):
        for rows in (threshold - 1, threshold):
            value = _make_case((rows,), (0,), heads, _gpu())
            workspace = _runtime._Workspace(value.q, value.k, value.v, value.indices,
                                            value.query_lens, value.prefix_lens, value.scale)
            assert (workspace.union is not None) == (rows == threshold)
    monkeypatch.setattr(_runtime, "UNION_MIN_ROWS", 40)
    monkeypatch.setattr(_runtime, "UNION_MIN_ROWS_PER_HEAD", 0)
    monkeypatch.setattr(_runtime, "_workspaces", OrderedDict())
    for rows in (39, 40):
        value = _make_case((rows,), (3000,), 12, _gpu(), shared=True)
        _check(value)
        workspace = next(reversed(_runtime._workspaces.values()))
        assert (workspace.union is not None) == (rows >= 40)
        assert workspace.direct.gated == (rows >= 40)


def test_invalid_selection_error_buffer():
    """With an error buffer K1 reports invalid rows instead of trapping."""
    value = _make_case((5,), (30000,), 12, _gpu())
    workspace = _runtime._Workspace(value.q, value.k, value.v, value.indices,
                                    value.query_lens, value.prefix_lens, value.scale)
    inputs = workspace.bind(value.q, value.k, value.v, value.indices)
    errors = torch.empty(5, dtype=torch.int32, device=value.q.device)
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
            inputs.block_indices, errors, num_warps=1,
        )
        assert errors.cpu().tolist() == [int(corruption != "valid"), 0, 0, 0, 0]


def test_invalid_selection_traps():
    """A public call with an invalid row stops the GPU queue in K1, which aborts the process."""
    _gpu()
    script = ("import torch\n"
              "from tests.ops.qsa._attention import _call, _gpu, _make_case\n"
              "value = _make_case((5,), (30000,), 12, _gpu())\n"
              "value.indices[0, 1] += 1\n"
              "_call(value)\n"
              "torch.cuda.synchronize()\n"
              "print('no trap')\n")
    # The trap is deliberate: skip ROCr's GPU core dump file.
    result = subprocess.run([sys.executable, "-c", script], cwd=ROOT, capture_output=True, text=True,
                            errors="replace", timeout=900,
                            env=dict(os.environ, HSA_DISABLE_COREDUMP_ON_EXCEPTION="1"))
    assert result.returncode != 0 and "no trap" not in result.stdout
    assert "attention_recover_scatter" in result.stderr


def test_graph_selection_and_kv_update(union_min_rows):
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


def test_graph_keeps_scratch_across_growth(union_min_rows, monkeypatch):
    """Growing the stream's shared scratch rebinds eager layouts but never moves a captured one."""
    monkeypatch.setattr(_runtime, "_arenas", {})
    device = _gpu()
    small = _make_case((32,), (30000,), 6, device)
    medium = _make_case((48,), (40000,), 6, device, seed=23)
    large = _make_case((64,), (60000,), 6, device, seed=29, shared=True)
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):
        output = _call(small)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            _call(small, output)
        first_medium = _call(medium)
        arena = next(iter(_runtime._arenas.values()))
        generation = arena.generation
        first_large = _call(large)
        assert arena.generation > generation
        output.fill_(float("nan"))
        graph.replay()
        eager_small, second_medium, second_large = _call(small), _call(medium), _call(large)
    torch.cuda.synchronize(device)
    torch.testing.assert_close(output, eager_small, rtol=0, atol=0)
    torch.testing.assert_close(output.float(), reference(small, list(range(32))), rtol=.02, atol=.02)
    torch.testing.assert_close(second_medium, first_medium, rtol=0, atol=0)
    torch.testing.assert_close(second_large, first_large, rtol=0, atol=0)
    torch.testing.assert_close(first_large.float(), reference(large, list(range(64))), rtol=.02, atol=.02)
