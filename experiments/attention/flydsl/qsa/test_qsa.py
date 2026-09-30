# Copyright 2023-2024 SGLang Team
# SPDX-License-Identifier: Apache-2.0

"""QSA bundle correctness checks adapted for the package implementation API.

Only the TP2/TP4 M=12000 cases carry the perf marker; no test performs timing.
TP8 is correctness-only, and small TP1 cases retain baseline compatibility.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path
from typing import Literal

import pytest
import torch

from .contract import AttentionInputs, CaseSpec, load_model_shape
from .inputs import default_spec, make_inputs, validate_inputs
from .reference import check_output, sparse_reference


def _case_spec(
    *, name: str, layout: str, selection: str, attention_tp: int
) -> CaseSpec:
    if layout == "ragged":
        return CaseSpec(
            name=name,
            query_lens=(7, 0, 9, 4),
            # A prefix-only request must still advance the packed KV offset.
            prefix_lens=(0, 0, 0, 0) if name == "no_prefix" else (3000, 5, 12000, 3),
            attention_tp=attention_tp,
            selection=selection,
            selection_group=4,
        )
    if layout == "257":
        return default_spec(
            name=name,
            query_tokens=257,
            prefix_tokens=3000,
            attention_tp=attention_tp,
            selection=selection,
        )
    if layout == "12000":
        return default_spec(name=name, attention_tp=attention_tp, selection=selection)
    raise ValueError(f"Unknown test layout: {layout}")


def _assert_local_shapes(*, inputs: AttentionInputs) -> None:
    query_heads, kv_heads = {1: (24, 2), 2: (12, 1), 4: (6, 1), 8: (3, 1)}[
        inputs.spec.attention_tp
    ]
    query_tokens = sum(inputs.spec.query_lens)
    kv_tokens = query_tokens + sum(inputs.spec.prefix_lens)
    assert inputs.q.shape == (query_tokens, query_heads, 256)
    assert inputs.k.shape == inputs.v.shape == (kv_tokens, kv_heads, 256)


def _assert_causal_masks_cpu(*, inputs: AttentionInputs) -> None:
    validate_inputs(inputs=inputs)
    _assert_local_shapes(inputs=inputs)
    assert inputs.q.device.type == "cpu"
    ratio = inputs.model.indexer_compress_ratio
    visible = inputs.query_positions + 1
    counts = (visible // ratio).clamp_max(inputs.model.block_topk) * ratio
    counts += visible % ratio
    valid = inputs.indices >= 0
    expected_valid = torch.arange(inputs.model.final_topk)[None, :] < counts[:, None]
    torch.testing.assert_close(valid, expected_valid)
    assert bool(torch.all(~valid | (inputs.indices < visible[:, None])))
    kv_lens = inputs.kv_lens[inputs.query_sequence_ids.long()]
    assert bool(torch.all(~valid | (inputs.indices < kv_lens[:, None])))
    assert bool(torch.all(counts > 0))
    for tensor in (inputs.q, inputs.k, inputs.v):
        assert tensor.is_contiguous() and tensor.device.type == "cpu"
    assert inputs.scale == 0.0625


def _with_nan_kv_guards(
    *, inputs: AttentionInputs
) -> tuple[AttentionInputs, list[torch.Tensor]]:
    fields = {name: getattr(inputs, name) for name in inputs.__struct_fields__}
    guards = []
    for name in ("k", "v"):
        original = fields[name]
        storage = torch.full(
            (original.shape[0] + 8, *original.shape[1:]),
            float("nan"),
            dtype=original.dtype,
            device=original.device,
        )
        storage[: original.shape[0]].copy_(original)
        fields[name] = storage[: original.shape[0]]
        guards.append(storage[original.shape[0] :])
    return AttentionInputs(**fields), guards


def _guarded_output(*, inputs: AttentionInputs) -> tuple[torch.Tensor, torch.Tensor]:
    storage = torch.full(
        (inputs.q.shape[0] + 2, *inputs.q.shape[1:]),
        123.0,
        dtype=inputs.q.dtype,
        device=inputs.q.device,
    )
    return storage, storage[1:-1]


def test_frozen_baseline_snapshot_cpu():
    root = Path(__file__).parent
    manifest = json.loads((root / "source_manifest.json").read_text(encoding="utf-8"))
    source = (root / manifest["snapshot_path"]).read_text(encoding="utf-8")
    lines = source.splitlines(keepends=True)
    actual = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef) and node.name in manifest["functions"]:
            first = min([node.lineno] + [item.lineno for item in node.decorator_list])
            body = "".join(lines[first - 1 : node.end_lineno])
            actual[node.name] = hashlib.sha256(body.encode("utf-8")).hexdigest()
    assert actual == manifest["functions"], "Frozen SGLang baseline kernels changed"


@pytest.mark.parametrize("name", ("no_prefix", "chunk_prefill"))
@pytest.mark.parametrize("attention_tp", (1, 2, 4, 8))
def test_default_model_contract_cpu(name, attention_tp):
    model = load_model_shape()
    spec = default_spec(name=name, attention_tp=attention_tp)
    assert spec.query_lens == (12000,)
    assert spec.prefix_lens == ((0,) if name == "no_prefix" else (12000,))
    assert spec.attention_tp == attention_tp
    assert spec.selection == "independent" and spec.selection_group == 8
    assert spec.seed == 17
    assert (model.num_attention_heads, model.num_key_value_heads, model.head_dim) == (
        24,
        2,
        256,
    )
    assert model.indexer_budget == 2048 and model.indexer_compress_ratio == 4
    assert model.block_topk == 512 and model.final_topk == 2051


@pytest.mark.parametrize("name", ("no_prefix", "chunk_prefill"))
@pytest.mark.parametrize("selection", ("independent", "shared", "recent"))
@pytest.mark.parametrize("attention_tp", (1, 2, 4, 8))
def test_causal_block_and_tail_contract_cpu(name, selection, attention_tp):
    spec = default_spec(
        name=name,
        query_tokens=2057 if name == "no_prefix" else 17,
        prefix_tokens=2040,
        attention_tp=attention_tp,
        selection=selection,
    )
    inputs = make_inputs(spec=spec, device="cpu")
    _assert_causal_masks_cpu(inputs=inputs)
    counts = (inputs.indices >= 0).sum(dim=1)
    prefix = spec.prefix_lens[0]
    for visible, count in (
        (1, 1),
        (4, 4),
        (2048, 2048),
        (2051, 2051),
        (2052, 2048),
        (2053, 2049),
    ):
        row = visible - prefix - 1
        if 0 <= row < counts.numel():
            assert counts[row].item() == count


@pytest.mark.parametrize("name", ("no_prefix", "chunk_prefill"))
@pytest.mark.parametrize("selection", ("independent", "shared", "recent"))
@pytest.mark.parametrize("attention_tp", (1, 2, 4, 8))
def test_ragged_prefix_contract_cpu(name, selection, attention_tp):
    spec = _case_spec(
        name=name, layout="ragged", selection=selection, attention_tp=attention_tp
    )
    inputs = make_inputs(spec=spec, device="cpu")
    _assert_causal_masks_cpu(inputs=inputs)
    assert inputs.cu_q.tolist() == [0, 7, 7, 16, 20]
    assert inputs.query_sequence_ids.tolist() == [0] * 7 + [2] * 9 + [3] * 4
    assert inputs.cu_k.tolist() == (
        [0, 7, 7, 16, 20] if name == "no_prefix" else [0, 3007, 3012, 15021, 15028]
    )
    if name == "chunk_prefill":
        assert inputs.query_positions[[0, 7, 16]].tolist() == [3000, 12000, 3]


@pytest.mark.parametrize(
    "corruption",
    ("future_token", "padding", "block", "sequence", "position", "kv_offset"),
)
def test_invalid_causal_contract_rejected_cpu(corruption):
    spec = CaseSpec(
        name="chunk_prefill",
        query_lens=(3, 0, 2),
        prefix_lens=(5, 7, 1),
        attention_tp=8,
    )
    inputs = make_inputs(spec=spec, device="cpu")
    if corruption == "future_token":
        inputs.indices[0, 0] = inputs.query_positions[0] + 1
    elif corruption == "padding":
        inputs.indices[0, -1] = 0
    elif corruption == "block":
        inputs.block_indices[2, 1] = inputs.block_indices[2, 0]
    elif corruption == "sequence":
        inputs.query_sequence_ids[0] = 2
    elif corruption == "position":
        inputs.query_positions[0] += 1
    else:
        inputs.cu_k[1] += 1
    with pytest.raises(AssertionError):
        validate_inputs(inputs=inputs)


def test_reference_masks_padding_and_ragged_prefix_cpu():
    spec = CaseSpec(
        name="chunk_prefill",
        query_lens=(2, 0, 1),
        prefix_lens=(2, 3, 1),
        attention_tp=8,
    )
    inputs = make_inputs(spec=spec, device="cpu")
    inputs.q.zero_()
    inputs.k.zero_()
    for sequence, (start, end) in enumerate(zip(inputs.cu_k[:-1], inputs.cu_k[1:])):
        values = 10 * (sequence + 1) + torch.arange(int(end - start))
        inputs.v[int(start) : int(end)] = values[:, None, None]
    validate_inputs(inputs=inputs)
    actual = sparse_reference(inputs=inputs, rows=[0, 1, 2])
    expected = torch.tensor([11.0, 11.5, 30.5])[:, None, None].expand_as(actual)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)


@pytest.fixture
def gpu_device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA/ROCm GPU unavailable; GPU correctness was not checked")
    pytest.importorskip("triton")
    pytest.importorskip("flydsl")
    if torch.version.hip is None:
        pytest.skip("The implemented FlyDSL kernel targets gfx942")
    if (
        torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName.split(
            ":"
        )[0]
        != "gfx942"
    ):
        pytest.skip("The implemented FlyDSL kernel targets gfx942")
    return torch.device("cuda", torch.cuda.current_device())


def _check_implementation_against_baseline(
    *, inputs: AttentionInputs, sample_count: int, prepared: object | None = None
) -> None:
    from . import baseline, implementation

    validate_inputs(inputs=inputs)
    _assert_local_shapes(inputs=inputs)
    baseline_prepared = baseline.prepare(inputs=inputs)
    baseline_out = torch.full_like(inputs.q, float("nan"))
    baseline.run(inputs=inputs, prepared=baseline_prepared, out=baseline_out)
    check_output(inputs=inputs, output=baseline_out, sample_count=sample_count)

    if prepared is None:
        prepared = implementation.prepare(inputs=inputs)
        assert prepared.mode == "auto"
    assert isinstance(prepared, implementation.DispatchPlan)
    out = torch.full_like(inputs.q, float("nan"))
    implementation.run(inputs=inputs, prepared=prepared, out=out)
    check_output(
        inputs=inputs, output=out, sample_count=sample_count, rtol=2e-2, atol=2e-2
    )
    torch.testing.assert_close(out, baseline_out, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("name", ("no_prefix", "chunk_prefill"))
@pytest.mark.parametrize("selection", ("independent", "shared", "recent"))
@pytest.mark.parametrize(
    "attention_tp,layout",
    (
        pytest.param(2, "ragged", id="tp2-ragged"),
        pytest.param(4, "ragged", id="tp4-ragged"),
        pytest.param(8, "ragged", id="tp8-ragged"),
        pytest.param(2, "257", id="tp2-m257"),
        pytest.param(4, "257", id="tp4-m257"),
        pytest.param(8, "257", id="tp8-m257"),
        pytest.param(2, "12000", id="tp2-m12000", marks=pytest.mark.perf),
        pytest.param(4, "12000", id="tp4-m12000", marks=pytest.mark.perf),
        pytest.param(8, "12000", id="tp8-m12000-correctness"),
    ),
)
def test_implementation_matches_baseline_and_fp32(
    gpu_device, name, selection, attention_tp, layout
):
    spec = _case_spec(
        name=name, layout=layout, selection=selection, attention_tp=attention_tp
    )
    inputs = make_inputs(spec=spec, device=gpu_device)
    sample_count = inputs.q.shape[0] if layout == "ragged" else 32
    _check_implementation_against_baseline(inputs=inputs, sample_count=sample_count)


@pytest.mark.parametrize("name", ("no_prefix", "chunk_prefill"))
@pytest.mark.parametrize("attention_tp", (2, 4, 8))
@pytest.mark.parametrize("block_n", (32, 64))
def test_direct_matches_baseline_and_fp32(gpu_device, name, attention_tp, block_n):
    from . import implementation

    inputs = make_inputs(
        spec=default_spec(
            name=name,
            query_tokens=37,
            prefix_tokens=3000,
            attention_tp=attention_tp,
        ),
        device=gpu_device,
    )
    prepared = implementation.prepare(
        inputs=inputs, mode="direct", dense_limit=0, block_n=block_n
    )
    assert prepared.mode == "direct"
    assert prepared.dense.query_counts == (0,)
    assert prepared.union is None and prepared.direct is not None
    assert prepared.direct.block_n == block_n
    assert prepared.direct.query_tile == 4
    assert not prepared.direct.gated
    _check_implementation_against_baseline(
        inputs=inputs, prepared=prepared, sample_count=37
    )


@pytest.mark.parametrize("name", ("no_prefix", "chunk_prefill"))
@pytest.mark.parametrize("layout", ("ragged", "257"))
def test_tp1_baseline_compatibility(gpu_device, name, layout):
    inputs = make_inputs(
        spec=_case_spec(
            name=name, layout=layout, selection="independent", attention_tp=1
        ),
        device=gpu_device,
    )
    sample_count = inputs.q.shape[0] if layout == "ragged" else 32
    _check_implementation_against_baseline(inputs=inputs, sample_count=sample_count)


@pytest.mark.parametrize("attention_tp", (1, 2, 4, 8))
def test_zero_prefix_chunk_matches_baseline_prefill(gpu_device, attention_tp):
    from . import baseline

    outputs = []
    for name in ("no_prefix", "chunk_prefill"):
        spec = default_spec(
            name=name, query_tokens=41, prefix_tokens=0, attention_tp=attention_tp
        )
        inputs = make_inputs(spec=spec, device=gpu_device)
        _assert_local_shapes(inputs=inputs)
        prepared = baseline.prepare(inputs=inputs)
        out = torch.full_like(inputs.q, float("nan"))
        baseline.run(inputs=inputs, prepared=prepared, out=out)
        check_output(inputs=inputs, output=out, sample_count=out.shape[0])
        outputs.append(out)
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)


@pytest.mark.parametrize("attention_tp,query_tile", ((2, 8), (4, 16), (8, 32)))
@pytest.mark.parametrize("mode,dense_limit", (("auto", 2051), ("union", 0)))
def test_local_union_query_tile_cap_and_coverage(
    gpu_device, attention_tp, query_tile, mode, dense_limit
):
    from . import implementation

    inputs = make_inputs(
        spec=CaseSpec(
            name="chunk_prefill",
            query_lens=(37, 0, 65, 37),
            prefix_lens=(0, 5, 2047, 12000),
            attention_tp=attention_tp,
        ),
        device=gpu_device,
    )
    prepared = implementation.prepare(
        inputs=inputs, query_tile=32, mode=mode, dense_limit=dense_limit
    )
    assert prepared.mode == mode
    union = prepared.union
    assert union is not None
    group = inputs.q.shape[1] // inputs.k.shape[1]
    assert (
        union.query_tile
        == query_tile
        == min(32, 1 << ((128 // group).bit_length() - 1))
    )
    assert union.group_padded == group
    counts = tuple(
        min(length, max(0, dense_limit - prefix))
        for length, prefix in zip(inputs.spec.query_lens, inputs.spec.prefix_lens)
    )
    assert prepared.dense.query_counts == counts

    metadata, query_tiles = [], [-1] * inputs.q.shape[0]
    coverage = [0] * inputs.q.shape[0]
    q_start, k_start = 0, 0
    for length, prefix, count in zip(
        inputs.spec.query_lens, inputs.spec.prefix_lens, counts
    ):
        coverage[q_start : q_start + count] = [1] * count
        local = count
        while local < length:
            end = min(length, (local // query_tile + 1) * query_tile)
            first, rows = q_start + local, end - local
            query_tiles[first : first + rows] = [len(metadata)] * rows
            metadata.append([first, rows, k_start, prefix + length, prefix + local])
            local = end
        q_start += length
        k_start += prefix + length
    assert union.metadata.cpu().tolist() == metadata
    assert union.num_tiles == len(metadata)
    assert union.query_tiles.cpu().tolist() == query_tiles
    active = union.active.cpu().tolist()
    assert all(flag in (0, 1) for flag in active)
    for flag, (first, rows, *_) in zip(active, metadata):
        if flag:
            for row in range(first, first + rows):
                coverage[row] += 1
    if mode == "union":
        assert prepared.direct is None
        assert all(active)
    else:
        direct = prepared.direct
        assert direct is not None and direct.gated
        assert direct.active.data_ptr() == union.active.data_ptr()
        assert direct.query_tiles.data_ptr() == union.query_tiles.data_ptr()
        assert set(active) == {0, 1}
        for first, rows, *_ in direct.metadata.cpu().tolist():
            for row in range(first, first + rows):
                assert query_tiles[row] >= 0
                if not active[query_tiles[row]]:
                    coverage[row] += 1
    assert coverage == [1] * inputs.q.shape[0]
    _check_implementation_against_baseline(
        inputs=inputs, prepared=prepared, sample_count=inputs.q.shape[0]
    )


@pytest.mark.parametrize("query_tile", (8, 10, 32))
@pytest.mark.parametrize("attention_tp", (2, 4, 8))
def test_unaligned_union_nan_guards_and_all_masked_tiles(
    gpu_device, query_tile, attention_tp
):
    from . import implementation

    inputs = make_inputs(
        spec=default_spec(
            name="chunk_prefill",
            query_tokens=9,
            prefix_tokens=56,
            attention_tp=attention_tp,
        ),
        device=gpu_device,
    )
    inputs, guards = _with_nan_kv_guards(inputs=inputs)
    plan = implementation.prepare(
        inputs=inputs,
        query_tile=query_tile,
        mode="union",
        dense_limit=0,
        max_union_inflation=float("inf"),
    )
    storage, out = _guarded_output(inputs=inputs)
    out.fill_(float("nan"))
    implementation.run(inputs=inputs, prepared=plan, out=out)
    check_output(inputs=inputs, output=out, sample_count=out.shape[0])
    assert bool((storage[[0, -1]] == 123).all())
    assert all(bool(torch.isnan(guard).all()) for guard in guards)


@pytest.mark.parametrize("attention_tp", (2, 4, 8))
@pytest.mark.parametrize("block_n", (32, 64))
def test_direct_cancellation_scales_after_fp32_dot(gpu_device, attention_tp, block_n):
    from . import implementation

    inputs = make_inputs(
        spec=default_spec(name="no_prefix", query_tokens=32, attention_tp=attention_tp),
        device=gpu_device,
    )
    inputs.q.zero_()
    inputs.k.zero_()
    inputs.v.zero_()
    inputs.q[1, :, 0::2] = 1
    inputs.q[1, :, 1::2] = 129 / 128
    inputs.k[0, :, 0::2] = 129 / 64
    inputs.k[0, :, 1::2] = -2
    inputs.k[1] = -inputs.k[0]
    inputs.v[0].fill_(1)
    inputs.v[1].fill_(-1)
    plan = implementation.prepare(
        inputs=inputs, mode="direct", dense_limit=0, block_n=block_n
    )
    assert plan.union is None and plan.direct is not None
    assert plan.dense.query_counts == (0,)
    out = torch.full_like(inputs.q, float("nan"))
    implementation.run(inputs=inputs, prepared=plan, out=out)
    check_output(inputs=inputs, output=out, sample_count=out.shape[0])


@pytest.mark.parametrize("attention_tp", (2, 4, 8))
def test_union_graph_rebuild_preserves_current_indices(gpu_device, attention_tp):
    from . import implementation

    inputs = make_inputs(
        spec=default_spec(
            name="chunk_prefill",
            query_tokens=65,
            prefix_tokens=3000,
            attention_tp=attention_tp,
        ),
        device=gpu_device,
    )
    plan = implementation.prepare(
        inputs=inputs, mode="union", dense_limit=0, max_union_inflation=float("inf")
    )
    assert plan.union is not None and plan.direct is None
    out = torch.full_like(inputs.q, float("nan"))
    warmup = torch.cuda.Stream(device=gpu_device)
    warmup.wait_stream(torch.cuda.current_stream(gpu_device))
    with torch.cuda.stream(warmup):
        for _ in range(2):
            implementation.rebuild_plan(inputs=inputs, plan=plan)
            implementation.run(inputs=inputs, prepared=plan, out=out)
    torch.cuda.current_stream(gpu_device).wait_stream(warmup)
    check_output(inputs=inputs, output=out, sample_count=65)
    original = out.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        implementation.rebuild_plan(inputs=inputs, plan=plan)
        implementation.run(inputs=inputs, prepared=plan, out=out)
    changed = make_inputs(
        spec=default_spec(
            name="chunk_prefill",
            query_tokens=65,
            prefix_tokens=3000,
            attention_tp=attention_tp,
            seed=51,
        ),
        device=gpu_device,
    )
    assert not torch.equal(inputs.indices, changed.indices)
    assert not torch.equal(inputs.block_indices, changed.block_indices)
    inputs.indices.copy_(changed.indices)
    inputs.block_indices.copy_(changed.block_indices)
    validate_inputs(inputs=inputs)
    out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize(gpu_device)
    check_output(inputs=inputs, output=out, sample_count=65)
    assert not torch.equal(out, original)


@pytest.mark.parametrize("query_tile", (8, 32))
@pytest.mark.parametrize("attention_tp", (2, 4, 8))
def test_forced_union_disjoint_blocks_and_softmax_rescale(
    gpu_device, query_tile, attention_tp
):
    from . import implementation

    inputs = make_inputs(
        spec=default_spec(
            name="chunk_prefill",
            query_tokens=33,
            prefix_tokens=12000,
            attention_tp=attention_tp,
        ),
        device=gpu_device,
    )
    for row in range(inputs.q.shape[0]):
        blocks = torch.arange(512, dtype=torch.int32, device=gpu_device)
        blocks += (row % 4) * 600
        inputs.block_indices[row] = blocks
        inputs.indices[row].fill_(-1)
        inputs.indices[row, :2048] = (
            blocks[:, None] * 4 + torch.arange(4, dtype=torch.int32, device=gpu_device)
        ).flatten()
        visible = 12001 + row
        inputs.indices[row, 2048 : 2048 + visible % 4] = torch.arange(
            visible // 4 * 4, visible, dtype=torch.int32, device=gpu_device
        )
    validate_inputs(inputs=inputs)
    # Disjoint early tiles are all masked for some queries; larger scores exercise rescaling.
    inputs.q.mul_(3)
    inputs.k.mul_(3)
    plan = implementation.prepare(
        inputs=inputs,
        query_tile=query_tile,
        mode="union",
        dense_limit=0,
        max_union_inflation=float("inf"),
    )
    out = torch.empty_like(inputs.q)
    implementation.run(inputs=inputs, prepared=plan, out=out)
    check_output(inputs=inputs, output=out, sample_count=33)
    first = out.clone()
    implementation.run(inputs=inputs, prepared=plan, out=out)
    torch.testing.assert_close(out, first, rtol=0, atol=0)


@pytest.mark.parametrize("attention_tp", (2, 4, 8))
@pytest.mark.parametrize("mode", ("auto", "direct"))
@pytest.mark.parametrize("block_n", (32, 64))
def test_block_native_modes_match_fp32(
    gpu_device: torch.device,
    attention_tp: int,
    mode: Literal["auto", "direct"],
    block_n: int,
) -> None:
    from . import implementation

    inputs = make_inputs(
        spec=_case_spec(
            name="chunk_prefill",
            layout="ragged",
            selection="independent",
            attention_tp=attention_tp,
        ),
        device=gpu_device,
    )
    inputs, guards = _with_nan_kv_guards(inputs=inputs)
    validate_inputs(inputs=inputs)
    _assert_local_shapes(inputs=inputs)
    prepared = implementation.prepare(
        inputs=inputs, mode=mode, dense_limit=0, block_n=block_n
    )
    assert prepared.dense.query_counts == (0,) * len(inputs.spec.query_lens)
    if mode == "direct":
        assert prepared.union is None and prepared.direct is not None
    storage, out = _guarded_output(inputs=inputs)
    out.fill_(float("nan"))
    implementation.run(inputs=inputs, prepared=prepared, out=out)
    check_output(inputs=inputs, output=out, sample_count=out.shape[0])
    assert bool((storage[[0, -1]] == 123).all())
    assert all(bool(torch.isnan(guard).all()) for guard in guards)


@pytest.mark.parametrize("attention_tp", (2, 4, 8))
@pytest.mark.parametrize("mode", ("auto", "direct"))
@pytest.mark.parametrize("prefix_tokens", (0, 2047, 2050, 2051, 12000))
def test_dense_dispatch_boundaries_match_fp32(
    gpu_device: torch.device,
    attention_tp: int,
    mode: Literal["auto", "direct"],
    prefix_tokens: int,
) -> None:
    from . import implementation

    inputs = make_inputs(
        spec=default_spec(
            name="chunk_prefill",
            query_tokens=33,
            prefix_tokens=prefix_tokens,
            attention_tp=attention_tp,
        ),
        device=gpu_device,
    )
    validate_inputs(inputs=inputs)
    _assert_local_shapes(inputs=inputs)
    prepared = implementation.prepare(inputs=inputs, mode=mode, dense_limit=2051)
    expected_count = min(33, max(0, 2051 - prefix_tokens))
    assert prepared.dense.query_counts == (expected_count,)
    assert len(prepared.dense.calls) == int(expected_count > 0)
    storage, out = _guarded_output(inputs=inputs)
    out.fill_(float("nan"))
    implementation.run(inputs=inputs, prepared=prepared, out=out)
    check_output(inputs=inputs, output=out, sample_count=out.shape[0])
    assert bool((storage[[0, -1]] == 123).all())


@pytest.mark.parametrize("attention_tp", (2, 4, 8))
@pytest.mark.parametrize("limit", (0, 64, 2051))
def test_dense_only_writes_eligible_ragged_prefixes(
    gpu_device: torch.device, attention_tp: int, limit: int
) -> None:
    from . import dense

    inputs = make_inputs(
        spec=CaseSpec(
            name="chunk_prefill",
            query_lens=(7, 0, 9, 9, 9, 9, 7, 5, 9),
            prefix_lens=(0, 5, 55, 56, 2047, 2050, 2051, 12000, 3),
            attention_tp=attention_tp,
        ),
        device=gpu_device,
    )
    inputs, guards = _with_nan_kv_guards(inputs=inputs)
    # The prefix-only request also guards the first request's unaligned KV end.
    inputs.k[7:12].fill_(float("nan"))
    inputs.v[7:12].fill_(float("nan"))
    validate_inputs(inputs=inputs)
    _assert_local_shapes(inputs=inputs)
    prepared = dense.prepare(inputs=inputs, limit=limit)
    expected_counts = tuple(
        min(length, max(0, limit - prefix))
        for length, prefix in zip(inputs.spec.query_lens, inputs.spec.prefix_lens)
    )
    assert prepared.query_counts == expected_counts
    expected_calls, written_rows, untouched_rows = [], [], []
    q_start, k_start = 0, 0
    for request, (length, prefix, count) in enumerate(
        zip(inputs.spec.query_lens, inputs.spec.prefix_lens, expected_counts)
    ):
        if count:
            expected_calls.append((request, q_start, count, k_start, prefix + count))
        written_rows.extend(range(q_start, q_start + count))
        untouched_rows.extend(range(q_start + count, q_start + length))
        q_start += length
        k_start += prefix + length
    assert [
        (call.request, call.q_start, call.q_count, call.k_start, call.kv_count)
        for call in prepared.calls
    ] == expected_calls
    for call in prepared.calls:
        assert call.cu_q.tolist() == [0, call.q_count]
        assert call.cu_k.tolist() == [0, call.kv_count]

    storage, out = _guarded_output(inputs=inputs)
    out[written_rows] = float("nan")
    dense.run(inputs=inputs, prepared=prepared, out=out)
    if written_rows:
        actual = out[written_rows].float()
        expected = sparse_reference(inputs=inputs, rows=written_rows)
        assert bool(torch.isfinite(actual).all())
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    assert bool((out[untouched_rows] == 123).all())
    assert bool((storage[[0, -1]] == 123).all())
    assert all(bool(torch.isnan(guard).all()) for guard in guards)


@pytest.mark.parametrize("attention_tp", (2, 4, 8))
@pytest.mark.parametrize("mode", ("auto", "direct", "union"))
def test_sparse_only_rejects_output_alias(gpu_device, attention_tp, mode):
    from . import implementation

    inputs = make_inputs(
        spec=default_spec(
            name="chunk_prefill",
            query_tokens=33,
            prefix_tokens=3000,
            attention_tp=attention_tp,
        ),
        device=gpu_device,
    )
    plan = implementation.prepare(inputs=inputs, mode=mode, dense_limit=0)
    out = inputs.v.flatten()[: inputs.q.numel()].view_as(inputs.q)
    with pytest.raises(ValueError, match="overlap"):
        implementation.run(inputs=inputs, prepared=plan, out=out)


@pytest.mark.parametrize("attention_tp", (2, 4, 8))
def test_direct_unsorted_replacement_input(gpu_device, attention_tp):
    from . import implementation

    inputs = make_inputs(
        spec=default_spec(
            name="chunk_prefill",
            query_tokens=37,
            prefix_tokens=3000,
            attention_tp=attention_tp,
        ),
        device=gpu_device,
    )
    plan = implementation.prepare(
        inputs=inputs, mode="direct", dense_limit=0, sort_blocks=False
    )
    changed = make_inputs(
        spec=default_spec(
            name="chunk_prefill",
            query_tokens=37,
            prefix_tokens=3000,
            attention_tp=attention_tp,
            seed=99,
        ),
        device=gpu_device,
    )
    implementation.rebuild_plan(inputs=changed, plan=plan)
    out = torch.full_like(changed.q, float("nan"))
    implementation.run(inputs=changed, prepared=plan, out=out)
    check_output(inputs=changed, output=out, sample_count=37)


@pytest.mark.parametrize("attention_tp", (2, 4, 8))
@pytest.mark.parametrize("block_n", (32, 64))
def test_auto_graph_gate_flips_rebuild_sorted_direct(gpu_device, attention_tp, block_n):
    from . import implementation

    def case(selection):
        return make_inputs(
            spec=default_spec(
                name="chunk_prefill",
                query_tokens=67,
                prefix_tokens=12000,
                attention_tp=attention_tp,
                selection=selection,
                selection_group=32,
            ),
            device=gpu_device,
        )

    inputs, changed = case("shared"), case("independent")
    plan = implementation.prepare(inputs=inputs, block_n=block_n)
    assert plan.union is not None and plan.direct is not None
    out = torch.empty_like(inputs.q)
    for _ in range(2):
        implementation.run(inputs=inputs, prepared=plan, out=out)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        implementation.rebuild_plan(inputs=inputs, plan=plan)
        implementation.run(inputs=inputs, prepared=plan, out=out)
    shared_gate = plan.union.active.clone()
    original_indices, original_blocks = (
        inputs.indices.clone(),
        inputs.block_indices.clone(),
    )
    for index, blocks in (
        (changed.indices, changed.block_indices),
        (original_indices, original_blocks),
    ):
        inputs.indices.copy_(index)
        inputs.block_indices.copy_(blocks)
        out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize(gpu_device)
        check_output(inputs=inputs, output=out, sample_count=67)
        if index is changed.indices:
            assert not torch.equal(shared_gate, plan.union.active)


def test_rebuild_uses_input_device_not_ambient_device(gpu_device):
    from . import implementation

    if torch.cuda.device_count() < 2:
        pytest.skip("Needs two GPUs to verify device-context restoration")
    target = (gpu_device.index + 1) % torch.cuda.device_count()
    inputs = make_inputs(
        spec=default_spec(
            name="chunk_prefill", query_tokens=37, prefix_tokens=3000, attention_tp=4
        ),
        device=f"cuda:{target}",
    )
    ambient = torch.cuda.current_device()
    plan = implementation.prepare(inputs=inputs)
    implementation.rebuild_plan(inputs=inputs, plan=plan)
    assert torch.cuda.current_device() == ambient
    out = torch.empty_like(inputs.q)
    implementation.run(inputs=inputs, prepared=plan, out=out)
    check_output(inputs=inputs, output=out, sample_count=37)
    assert torch.cuda.current_device() == ambient


@pytest.fixture(scope="module", autouse=True)
def compiled_kernels_do_not_spill():
    yield
    if not torch.cuda.is_initialized() or torch.version.hip is None:
        return
    from . import dense, direct, kernel

    caches = {
        "union": kernel._COMPILED,
        "direct": direct._COMPILED,
        "dense_native": dense.native._COMPILED,
        "dense_bounded": dense._BOUNDED_COMPILED,
    }
    for name, cache in caches.items():
        for key, compiled in cache.items():
            for field in (
                "private_segment_fixed_size",
                "vgpr_spill_count",
                "sgpr_spill_count",
            ):
                values = re.findall(rf"\b{field}\s*=\s*(\d+)", compiled._keepalive.ir)
                assert values and not any(map(int, values)), (name, key, field, values)
