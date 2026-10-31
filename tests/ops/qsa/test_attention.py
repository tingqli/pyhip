"""Pytest entry for sparse_prefill_op_test.py (functional and graph cases, groups 12/6/3),
plus PyHIP's slot-table pools, the per-row pack rule and packed K/V over 2 GiB."""

from collections import OrderedDict
import dataclasses
import math

import pytest
import torch

from tests.ops.qsa import sparse_prefill_op_test as op
from tests.ops.qsa._attention import (
    _call,
    _call_paged,
    _gpu,
    _make_case,
    _paged,
    _runtime,
    reference,
)

# The op test makes CUDA the default device on import; keep it local to its cases.
torch.set_default_device(None)

GROUPS = (12, 6, 3)


def _run(test, case, group, layout):
    _gpu()  # the QSA_REPLAY_GPU device, as for the cases below (attention allows one GPU per process)
    with torch.device("cuda"):
        test(case, group * op.CASES[layout][2])


@pytest.mark.parametrize("group", GROUPS)
@pytest.mark.parametrize("case", [*op.FUNCTIONAL_CASES, *op.CONSTANT_CASES])
def test_sparse_prefill(case, group):
    _run(op.test_qsa_sparse_attention, case, group, case)


@pytest.mark.parametrize("group", GROUPS)
@pytest.mark.parametrize("case", list(op.GRAPH_CASES))
def test_graph_replay(case, group):
    _run(op.test_qsa_graph_replay, case, group, op.GRAPH_CASES[case][0])


@pytest.fixture(params=(None, 0), ids=("default-plan", "always-plan"))
def union_min_rows(request, monkeypatch):
    """Run with the production threshold and with union planning forced for small inputs."""
    if request.param is not None:
        monkeypatch.setattr(_runtime, "_union_min_rows", lambda group: request.param)
    monkeypatch.setattr(_runtime, "_workspaces", OrderedDict())
    return request.param


def _platform(monkeypatch, **fields):
    """Override platform fields for the direct planner."""
    entry = dataclasses.replace(_runtime.direct.platform(), **fields)
    monkeypatch.setattr(_runtime.direct, "platform", lambda: entry)


def _all_union(monkeypatch):
    """Every tile on union: no ratio budget, no route demotion, plans from one row on."""
    allocate = _runtime.prepare.allocate_plan

    def forced(**kwargs):
        plan = allocate(**kwargs)
        plan.union_ratio, plan.routing = math.inf, False
        return plan

    monkeypatch.setattr(_runtime.prepare, "allocate_plan", forced)
    monkeypatch.setattr(_runtime, "_union_min_rows", lambda group: 0)
    monkeypatch.setattr(_runtime, "_workspaces", OrderedDict())


def _latest():
    return next(reversed(_runtime._workspaces.values()))


def test_pack_by_rows_matches_raw(monkeypatch):
    """Past the 64 MiB budget a context of at most 32 tokens per query row is packed; the
    packed and raw paths give the same output bit for bit."""
    monkeypatch.setattr(_runtime, "_workspaces", OrderedDict())
    value = _make_case((4096,), (126976,), 6, _gpu())
    packed = _call(value)
    assert _latest().direct.packed_key is not None
    _platform(monkeypatch, pack_tokens_per_row=0)
    monkeypatch.setattr(_runtime, "_workspaces", OrderedDict())
    raw = _call(value)
    assert _latest().direct.packed_key is None
    assert torch.equal(packed.view(torch.int16), raw.view(torch.int16))
    rows = list(range(64))
    torch.testing.assert_close(
        packed[rows].float(), reference(value, rows), rtol=0.02, atol=0.02
    )


def test_kv_over_2gib(monkeypatch):
    """Packed K/V of 2 GiB or more (batched 1M-token contexts) in one call: requests are
    addressed from their own bases."""
    monkeypatch.setattr(_runtime, "_workspaces", OrderedDict())
    value = _make_case((2,) * 5, (2**20 - 2,) * 5, 3, _gpu())
    assert value.k.numel() * value.k.element_size() >= 2**31
    output = _call(value)
    rows = list(range(len(value.q)))
    torch.testing.assert_close(
        output.float(), reference(value, rows), rtol=0.02, atol=0.02
    )
    assert len(_runtime._workspaces) == 1


def _same_paged(value, pool_k, pool_v, slots):
    expected = _call(value)
    for table in (slots, slots.clone()):
        output = _call_paged(value, pool_k, pool_v, table)
        assert bool(torch.isfinite(output.float()).all())
        assert torch.equal(output.view(torch.int16), expected.view(torch.int16))


@pytest.mark.parametrize(
    "queries,prefixes,heads,shared",
    [
        pytest.param((9,), (130,), 3, False, id="no-plan"),
        pytest.param((1024,), (3000,), 12, False, id="union-direct-packed"),
        pytest.param((1024,), (3001,), 6, True, id="shared-nonmultiple"),
        pytest.param((2048,), (30000,), 3, True, id="union-bq32"),
        pytest.param((64,), (70001,), 12, False, id="raw-direct"),
        pytest.param((800,), (70000,), 6, False, id="raw-gated"),
        pytest.param((480, 1000, 301), (11392, 3, 4001), 12, False, id="ragged"),
        pytest.param((768, 768), (0, 5), 12, True, id="two-requests"),
    ],
)
def test_kv_slots_match_packed(queries, prefixes, heads, shared, union_min_rows):
    """K/V read through a slot table from 64-token pages of a NaN pool give the packed call's
    output bit for bit; the pool's NaN rows after a request's partial final block never
    reach it."""
    value = _make_case(queries, prefixes, heads, _gpu(), shared=shared)
    _same_paged(value, *_paged(value))


def test_kv_slots_pool_above_2gib(monkeypatch):
    """Pool byte offsets of 2**31 and more (unsigned buffer offsets) for union and raw direct."""
    monkeypatch.setattr(_runtime, "_workspaces", OrderedDict())
    rows = 6_500_000 // 64 * 64
    for queries, prefixes, heads, shared in (
        ((2048,), (30000,), 6, True),
        ((64,), (70001,), 12, False),
    ):
        value = _make_case(queries, prefixes, heads, _gpu(), shared=shared)
        pool_k, pool_v, slots = _paged(
            value, pool_rows=rows, first_page=4_400_000 // 64
        )
        assert int(slots.min()) * pool_k[0].numel() * 2 >= 2**31
        _same_paged(value, pool_k, pool_v, slots)
        assert not _latest().metadata["wide"]
        del pool_k, pool_v


WIDE_ROWS = 9_000_000 // 64 * 64


@pytest.mark.parametrize(
    "queries,prefixes,heads,shared",
    [
        pytest.param((2048,), (30000,), 6, True, id="union-packed"),
        pytest.param((64,), (70001,), 12, False, id="raw-direct"),
        pytest.param((800,), (70000,), 3, False, id="raw-gated"),
        pytest.param((480, 1000, 301), (11392, 3, 4001), 12, False, id="ragged"),
    ],
)
def test_kv_slots_pool_above_4gib(queries, prefixes, heads, shared, monkeypatch):
    """A pool of more than 4 GiB, its pages straddling 4 GiB: union, pack and the 64-bit raw
    direct match the packed call bit for bit."""
    monkeypatch.setattr(_runtime, "_workspaces", OrderedDict())
    value = _make_case(queries, prefixes, heads, _gpu(), shared=shared)
    pool_k, pool_v, slots = _paged(
        value, pool_rows=WIDE_ROWS, first_page=8_000_000 // 64
    )
    assert int((slots.long() * pool_k[0].numel() * 2 >= 2**32).sum()) > len(slots) // 4
    _same_paged(value, pool_k, pool_v, slots)
    assert _latest().metadata["wide"]


def test_kv_slots_union_partial_blocks_above_4gib(monkeypatch):
    """Every tile on union with all pages above 4 GiB: partial final blocks (KV of 1-3 tokens
    or not a multiple of 4) read only their request's rows, never the pool's NaN rows after
    them."""
    _all_union(monkeypatch)
    for queries, prefixes in (
        ((1, 2, 3), (0, 0, 0)),
        ((3, 5, 2), (0, 2, 4097)),
        ((61,), (30002,)),
    ):
        value = _make_case(queries, prefixes, 6, _gpu())
        pool_k, pool_v, slots = _paged(
            value, pool_rows=WIDE_ROWS, first_page=2**32 // 512 // 64
        )
        _same_paged(value, pool_k, pool_v, slots)
        workspace = _latest()
        assert workspace.metadata["wide"] and bool(workspace.union.active.all())
        del pool_k, pool_v


def test_kv_slots_input_errors():
    value = _make_case((16,), (3000,), 12, _gpu())
    pool_k, pool_v, slots = _paged(value)
    for table in (
        slots.long(),
        slots[:-1],
        torch.stack((slots, slots), 1)[:, 0],
        slots.cpu(),
    ):
        with pytest.raises(ValueError, match="kv_slots|lengths"):
            _call_paged(value, pool_k, pool_v, table)
