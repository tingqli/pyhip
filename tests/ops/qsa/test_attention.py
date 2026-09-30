# Copyright 2023-2024 SGLang Team
# SPDX-License-Identifier: Apache-2.0
"""QSA attention recovery, packing, routing and task-order kernel tests."""

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import triton

from tests.ops.qsa._attention import (
    _audit,
    _base,
    _call,
    _gpu,
    _make_case,
    _resources,
    _rows,
    _runtime,
    check,
    reference,
)


def test_direct_recovered_order():
    """Unordered selection stays exact; direct shares sorted recovery and guards tails."""
    device = _gpu()
    for queries, prefixes, heads, hk in (
        ((7, 0, 9), (30000, 5, 30000), 12, 1),
        ((9,), (30000,), 24, 2),
        ((5,), (30000,), 3, 1),
    ):
        value = _make_case(queries, prefixes, heads, device)
        if hk == 2:
            value.k = torch.cat((value.k, -value.k), dim=1)
            value.v = torch.cat((value.v, -value.v), dim=1)
        for field in ("k", "v"):
            original = getattr(value, field)
            storage = torch.full((original.shape[0] + 4, hk, 256), float("nan"),
                                 device=device, dtype=original.dtype)
            storage[:original.shape[0]].copy_(original)
            setattr(value, field, storage[:original.shape[0]])
        result = check(value)
        assert result["direct_rows"] > 0
        key = (device, torch.cuda.current_stream(device).cuda_stream,
               value.query_lens, value.prefix_lens, heads, hk, value.scale)
        workspace = _runtime._workspaces[key]
        assert workspace.direct.source_blocks is workspace.metadata["block_indices"]
        first = _call(value).clone()
        complete = value.indices[:, :2048].reshape(-1, 512, 4).clone()
        value.indices[:, :2048].copy_(complete.flip(1).reshape(-1, 2048))
        torch.testing.assert_close(_call(value), first, rtol=0, atol=0)
        _audit(value)
    # Exercise the paired direct kernel's short N/BN32 tails without claiming
    # the normal public dispatcher would send dense-eligible rows to direct.
    for length in (1, 15, 16, 17, 31, 32, 33, 65, 255, 256, 257, 511, 512, 513):
        value = _make_case((length,), (0,), 12, device)
        _call(value)
        key = (device, torch.cuda.current_stream(device).cuda_stream,
               value.query_lens, value.prefix_lens, 12, 1, value.scale)
        workspace = _runtime._workspaces[key]
        inputs = workspace.bind(value.q, value.k, value.v, value.indices)
        plan = _runtime.direct.prepare(inputs=inputs)
        storage = torch.full((length + 2, 12, 256), 123.0, device=device, dtype=value.q.dtype)
        output = storage[1:-1]
        _runtime.direct.run(inputs=inputs, prepared=plan, out=output)
        assert bool((storage[[0, -1]] == 123).all())
        torch.testing.assert_close(output.float(), reference(value, list(range(length))), rtol=0.02, atol=0.02)
    # A short request can cross any cached-index chunk, including the final
    # 512-block boundary. Keep physical KV tails poisoned on this forced path.
    for prefix in (763, 1019, 1275, 1531, 1787, 2043):
        value = _make_case((9,), (prefix,), 24, device)
        for field in ("k", "v"):
            original = getattr(value, field)
            original = torch.cat((original, -original), dim=1)
            storage = torch.full((original.shape[0] + 4, 2, 256), float("nan"),
                                 device=device, dtype=original.dtype)
            storage[:original.shape[0]].copy_(original)
            setattr(value, field, storage[:original.shape[0]])
        workspace = _runtime._Workspace(value.q, value.k, value.v, value.indices,
                                        value.query_lens, value.prefix_lens, value.scale)
        inputs = workspace.bind(value.q, value.k, value.v, value.indices)
        _runtime.prepare.attention_recover_scatter[(9,)](
            value.indices, inputs.query_positions, inputs.kv_lens,
            inputs.query_sequence_ids, inputs.block_indices, workspace.errors, num_warps=1,
        )
        assert not bool(workspace.errors.any())
        plan = _runtime.direct.prepare(inputs=inputs)
        storage = torch.full((11, 24, 256), 123.0, device=device, dtype=value.q.dtype)
        output = storage[1:-1]
        _runtime.direct.run(inputs=inputs, prepared=plan, out=output)
        assert bool((storage[[0, -1]] == 123).all())
        torch.testing.assert_close(output.float(), reference(value, list(range(9))), rtol=0.02, atol=0.02)
    # Inspect the recovery error buffer directly so invalid-ABI checks do not
    # poison this test process with the public asynchronous device assertion.
    value = _make_case((5,), (30000,), 12, device)
    workspace = _runtime._Workspace(value.q, value.k, value.v, value.indices,
                                    value.query_lens, value.prefix_lens, value.scale)
    inputs = workspace.bind(value.q, value.k, value.v, value.indices)
    for corruption in ("none", "duplicate", "block", "tail", "padding"):
        indices = value.indices.clone()
        if corruption == "duplicate":
            indices[0, 4:8].copy_(indices[0, :4])
        elif corruption == "block":
            indices[0, 1].add_(1)
        elif corruption == "tail":
            indices[0, 2048] = 29999
        elif corruption == "padding":
            indices[0, 2049] = 0
        _runtime.prepare.attention_recover_scatter[(5,)](
            indices, inputs.query_positions, inputs.kv_lens,
            inputs.query_sequence_ids, inputs.block_indices, workspace.errors, num_warps=1,
        )
        assert workspace.errors.cpu().tolist() == [int(corruption != "none"), 0, 0, 0, 0]
        _runtime.prepare.attention_order_masks_validate[(1,)](workspace.errors, workspace.valid, ROWS=5, num_warps=4)
        assert bool(workspace.valid.cpu()) == (corruption == "none")


@pytest.mark.parametrize("rows", (1, 1023, 1024, 1025, 12000, 261632))
def test_error_check_chunked_graph(rows):
    device = _gpu()
    errors = torch.zeros(rows, dtype=torch.int32, device=device)
    storage = torch.full((3,), True, dtype=torch.bool, device=device)
    valid = storage[1:2]
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):
        _runtime.prepare.attention_order_masks_validate[(1,)](errors, valid, ROWS=rows, num_warps=4)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            _runtime.prepare.attention_order_masks_validate[(1,)](errors, valid, ROWS=rows, num_warps=4)
    torch.cuda.current_stream(device).wait_stream(stream)
    for row in sorted({0, min(1023, rows - 1), min(1024, rows - 1), rows - 1}):
        for error in (1, -1, 0):
            errors[row] = error
            valid.fill_(error != 0)
            graph.replay()
            torch.cuda.synchronize(device)
            assert bool(valid.cpu()) == (error == 0)
            assert bool(storage[0].cpu()) and bool(storage[2].cpu())


@pytest.mark.parametrize("reverse", (False, True))
def test_prepare_recovery_validation(reverse):
    """Single-pass recovery retains every block, tail, padding and duplicate check."""
    device = _gpu()
    rng = np.random.default_rng(28)
    data, positions, lengths, expected_blocks, expected_errors = [], [], [], [], []
    for visible in (1, 2, 3, 4, 5, 7, 8, 2047, 2048, 2049, 2050, 2051, 2052, 12000, 30001):
        count = min(visible // 4, 512)
        for shuffle in (False, True):
            selected = (np.arange(count) if visible // 4 <= 512
                        else rng.choice(visible // 4, count, replace=False))
            if shuffle:
                rng.shuffle(selected)
            base = np.full(2051, -1, np.int32)
            base[:count * 4] = (selected[:, None] * 4 + np.arange(4)).reshape(-1)
            base[count * 4:count * 4 + visible % 4] = np.arange(visible // 4 * 4, visible)
            variants = [base]
            for column in sorted({0, 1, 2, 3, 4, 2047, 2048, 2049, 2050, count * 4}):
                for value in (-2147483648, -2, -1, 0, visible, 2147483644, 2147483647):
                    changed = base.copy()
                    changed[column] = value
                    variants.append(changed)
            if count >= 2:
                changed = base.copy()
                changed[4:8] = changed[:4]
                variants.append(changed)
            for indices in variants:
                recovered, error = [], False
                for block in indices[:count * 4].reshape(-1, 4).tolist():
                    valid = (block[0] >= 0 and block[0] % 4 == 0
                             and block == list(range(block[0], block[0] + 4)) and block[-1] < visible)
                    error |= not valid
                    recovered.append(block[0] // 4 if valid else -1)
                error |= len(set(recovered)) != count
                tail = list(range(visible // 4 * 4, visible))
                error |= indices[count * 4:count * 4 + len(tail)].tolist() != tail
                error |= bool(np.any(indices[count * 4 + len(tail):] != -1))
                data.append(indices)
                positions.append(visible - 1)
                lengths.append(visible + 4)
                expected_blocks.append(sorted(recovered) + [-1] * (512 - count))
                expected_errors.append(int(error))
    indices = torch.from_numpy(np.stack(data)).to(device)
    pos, lens = (torch.tensor(v, dtype=torch.int32, device=device) for v in (positions, lengths))
    sequences = torch.arange(len(data), dtype=torch.int32, device=device)
    blocks = torch.empty((len(data), 512), dtype=torch.int32, device=device)
    errors = torch.empty(len(data), dtype=torch.int32, device=device)
    _runtime.prepare.attention_recover_scatter[(len(data),)](
        indices, pos, lens, sequences, blocks, errors, REVERSE=reverse, num_warps=1)
    np.testing.assert_array_equal(blocks.cpu().numpy(), expected_blocks)
    np.testing.assert_array_equal(errors.cpu().numpy(), expected_errors)


@pytest.mark.parametrize("max_blocks", (511, 1024, 1025, 3000, 4097, 8012, 16385, 65536, 1048575))
def test_union_compact_chunked(max_blocks):
    device = _gpu()
    rng = np.random.default_rng(20260928 + max_blocks)
    rows = [1, 3, 10, 16, 32] * 3
    capacity = triton.cdiv(min(max_blocks, 32 * 513), 16) * 16
    dense = np.zeros((len(rows), max_blocks), dtype=np.uint32)
    metadata = np.zeros((len(rows), 5), dtype=np.int32)
    expected = []
    for tile, count in enumerate(rows):
        position = max_blocks * 4 - count - 1
        metadata[tile] = (0, count, 0, max_blocks * 4, position)
        size = 0 if tile < 5 else min(max_blocks, 500 if tile < 10 else count * 400)
        ids = np.sort(rng.choice(max_blocks, size=size, replace=False))
        bits = rng.integers(1, 1 << count, size=size, dtype=np.uint32)
        bits[:min(size, 250)] = np.uint32((1 << count) - 1)
        dense[tile, ids] = bits
        common = (bits == np.uint32((1 << count) - 1)) & (ids * 4 + 3 <= position)
        order = np.concatenate((ids[common], ids[~common]))
        expected.append((position, count, order, int(common.sum())))
    source = torch.from_numpy(dense.view(np.int32)).to(device)
    meta = torch.from_numpy(metadata).to(device)
    for packed, rho in ((False, 4.0), (True, 4.0), (False, float("inf"))):
        blocks = torch.full((len(rows), capacity), -777, dtype=torch.int32, device=device)
        membership = torch.full_like(blocks, -777)
        counts = torch.empty((len(rows), 2), dtype=torch.int32, device=device)
        active = torch.empty(len(rows), dtype=torch.int32, device=device)
        _runtime.prepare.attention_compact[(len(rows),)](
            source, blocks, membership, counts, meta, active, max_blocks, capacity,
            min(1024, triton.next_power_of_2(max_blocks)), rho, packed, num_warps=4)
        b, m = blocks.cpu().numpy(), membership.cpu().numpy().view(np.uint32)
        c, a = counts.cpu().numpy(), active.cpu().numpy()
        for tile, (position, count, order, common) in enumerate(expected):
            widths = [min((position + i + 1) // 4, 512) * 4 + (position + i + 1) % 4 for i in range(count)]
            enabled = (160 * math.ceil(len(order) / 16) <= 17 * sum(math.ceil(n / 32) for n in widths)
                       if packed else len(order) * count <= rho * sum(math.ceil(n / 4) for n in widths))
            assert a[tile] == enabled
            np.testing.assert_array_equal(c[tile], (len(order), common // 16 if enabled else 0))
            written = len(order) if enabled else 0
            if enabled:
                np.testing.assert_array_equal(b[tile, :written], order)
                np.testing.assert_array_equal(m[tile, :written], dense[tile, order])
            assert np.all(b[tile, written:] == -777)
            assert np.all(m[tile, written:] == np.uint32(2**32 - 777))


def test_direct_packed_kv():
    """Packed scratch is refreshed for graph replays and masks partial KV blocks."""
    device = _gpu()
    for length, prefix, heads, hk in (
        (8, 0, 12, 1), (24, 0, 24, 2), (68, 0, 12, 1),
        (32, 30000, 24, 2), (256, 30000, 12, 1),
    ):
        value = _make_case((length,), (prefix,), heads, device)
        if hk == 2:
            value.k = torch.cat((value.k, -value.k), dim=1)
            value.v = torch.cat((value.v, -value.v), dim=1)
        for field in ("k", "v"):
            original = getattr(value, field)
            storage = torch.full((original.shape[0] + 4, hk, 256), float("nan"),
                                 device=device, dtype=original.dtype)
            storage[:original.shape[0]].copy_(original)
            setattr(value, field, storage[:original.shape[0]])
        workspace = _runtime._Workspace(value.q, value.k, value.v, value.indices,
                                        value.query_lens, value.prefix_lens, value.scale)
        inputs = workspace.bind(value.q, value.k, value.v, value.indices)
        _runtime.prepare.attention_recover_scatter[(length,)](
            value.indices, inputs.query_positions, inputs.kv_lens,
            inputs.query_sequence_ids, inputs.block_indices, workspace.errors, num_warps=1,
        )
        assert not bool(workspace.errors.any())
        plan = _runtime.direct.prepare(inputs=inputs)
        assert plan.query_tile == 1 and plan.packed_key is not None and plan.packed_value is not None
        assert plan.packed_key.data_ptr() != value.k.data_ptr()
        assert plan.packed_value.data_ptr() != value.v.data_ptr()
        output_storage = torch.full((length + 2, heads, 256), 123.0, device=device, dtype=value.q.dtype)
        output = output_storage[1:-1]
        ids = list(range(length)) if length <= 68 else _rows(value)
        _runtime.direct.run(inputs=inputs, prepared=plan, out=output)
        torch.testing.assert_close(output[ids].float(), reference(value, ids), rtol=.02, atol=.02)
        before = output.clone()
        plan.packed_key.fill_(float("nan"))
        plan.packed_value.fill_(float("nan"))
        _runtime.direct.run(inputs=inputs, prepared=plan, out=output)
        torch.testing.assert_close(output, before, rtol=0, atol=0)
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            _runtime.direct.run(inputs=inputs, prepared=plan, out=output)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                _runtime.direct.run(inputs=inputs, prepared=plan, out=output)
        torch.cuda.current_stream(device).wait_stream(stream)
        value.v.neg_()
        value.k.mul_(2)
        output.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize(device)
        torch.testing.assert_close(output[ids].float(), reference(value, ids), rtol=.02, atol=.02)
        assert bool((output_storage[[0, -1]] == 123).all())

        if prefix:
            independent = value.indices.clone()
            shared = _make_case((length,), (prefix,), heads, device, shared=True).indices
            stream = torch.cuda.Stream(device=device)
            stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(stream):
                public_output = _call(value)
                public_graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(public_graph, stream=stream):
                    _call(value, public_output)
            torch.cuda.current_stream(device).wait_stream(stream)
            for selected in (shared, independent, shared, independent):
                value.indices.copy_(selected)
                value.v.neg_()
                public_output.fill_(float("nan"))
                public_graph.replay()
                torch.cuda.synchronize(device)
                torch.testing.assert_close(public_output[ids].float(), reference(value, ids), rtol=.02, atol=.02)

    # NaNs inside the last packed block must not contaminate queries whose
    # causal tail ends earlier than those tokens (zero probability is not enough).
    value = _make_case((8,), (0,), 12, device)
    value.k[5:].fill_(float("nan"))
    value.v[5:].fill_(float("nan"))
    workspace = _runtime._Workspace(value.q, value.k, value.v, value.indices,
                                    value.query_lens, value.prefix_lens, value.scale)
    inputs = workspace.bind(value.q, value.k, value.v, value.indices)
    _runtime.prepare.attention_recover_scatter[(8,)](
        value.indices, inputs.query_positions, inputs.kv_lens,
        inputs.query_sequence_ids, inputs.block_indices, workspace.errors, num_warps=1,
    )
    plan = _runtime.direct.prepare(inputs=inputs)
    output = torch.empty_like(value.q)
    _runtime.direct.run(inputs=inputs, prepared=plan, out=output)
    torch.testing.assert_close(output[:5].float(), reference(value, list(range(5))), rtol=.02, atol=.02)


def test_dense_task_order():
    """Balanced worker rows preserve the task prefix and packed dense graph writes."""
    for queries in (1, 64, 65, 127, 128, 129, 512, 2048, 2051):
        for heads in (1, 3, 6, 12, 16, 24):
            for cus in (1, 7, 80, 160):
                blocks = math.ceil(queries / 128)
                tasks = heads * blocks
                actual = []
                for worker in range(min(tasks, cus)):
                    for work in range(worker, math.ceil(tasks / cus) * cus, cus):
                        row, column = divmod(work, cus)
                        rank = row * cus + (cus - 1 - column if row % 2 else column)
                        if rank < tasks:
                            actual.append((rank % heads, blocks - 1 - rank // heads))
                assert sorted(actual) == [(h, b) for h in range(heads) for b in range(blocks)]
    device = _gpu()
    for queries, prefixes, heads, hk in (
        ((2051,), (0,), 24, 2),
        ((129, 0, 257), (31, 5, 63), 6, 1),
        ((65,), (1986,), 3, 1),
    ):
        value = _make_case(queries, prefixes, heads, device)
        if hk == 2:
            value.k = torch.cat((value.k, -value.k), dim=1)
            value.v = torch.cat((value.v, -value.v), dim=1)
        for name in ("k", "v"):
            original = getattr(value, name)
            storage = torch.full((original.shape[0] + 4, hk, 256), float("nan"),
                                 device=device, dtype=original.dtype)
            storage[:original.shape[0]].copy_(original)
            setattr(value, name, storage[:original.shape[0]])
        assert check(value)["dense_rows"] == sum(queries)
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            guard = torch.full((sum(queries) + 2, heads, 256), 123.0, device=device, dtype=value.q.dtype)
            output = guard[1:-1]
            _call(value, output)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                _call(value, output)
        torch.cuda.current_stream(device).wait_stream(stream)
        value.v.neg_()
        output.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize(device)
        assert bool((guard[[0, -1]] == 123).all())
        torch.testing.assert_close(output, _base(value), rtol=0.02, atol=0.02)
        ids = _rows(value)
        torch.testing.assert_close(output[ids].float(), reference(value, ids), rtol=0.02, atol=0.02)


def test_union_direct_routing():
    """Padded-work boundaries and graph routing retain exact current selections."""
    device = _gpu()
    # Include both sides of every integer cutoff, including partially filled
    # query groups. Sparse work uses the actual causal tail token count.
    cases = []
    for rows in (1, 2, 3, 6, 9, 10, 15, 16, 31, 32):
        for position in (2051, 2052, 2053, 2054, 30000):
            direct_work = sum(math.ceil((min((position + i + 1) // 4, 512) * 4
                                        + (position + i + 1) % 4) / 32) * 32 * 16
                              for i in range(rows))
            limit = 17 * direct_work // (10 * 64 * 128)
            for count in sorted({1, 513, max(1, limit * 16 - 1), max(1, limit * 16), limit * 16 + 1}):
                cases.append((rows, position, count, direct_work))
    capacity = 4096
    meta = np.array([(0, rows, 0, 32768, position) for rows, position, _, _ in cases], dtype=np.int32)
    dense = np.zeros((len(cases), capacity), dtype=np.int32)
    for tile, (_, _, count, _) in enumerate(cases):
        dense[tile, :count] = 1
    dense_gpu, meta_gpu = torch.from_numpy(dense).to(device), torch.from_numpy(meta).to(device)
    blocks = torch.empty_like(dense_gpu)
    membership = torch.empty_like(dense_gpu)
    counts = torch.empty((len(cases), 2), dtype=torch.int32, device=device)
    active = torch.empty(len(cases), dtype=torch.int32, device=device)
    for packed in (False, True):
        _runtime.prepare.attention_compact[(len(cases),)](
            dense_gpu, blocks, membership, counts, meta_gpu, active,
            capacity, capacity, capacity, 4.0, packed, num_warps=4,
        )
        actual = active.cpu().tolist()
        np.testing.assert_array_equal(counts[:, 0].cpu().numpy(), np.array([c[2] for c in cases]))
        for enabled, (rows, position, count, direct_work) in zip(actual, cases):
            if packed:
                expected = 10 * math.ceil(count / 16) * 64 * 128 <= 17 * direct_work
            else:
                total = sum(min((position + i + 1) // 4, 512) + bool((position + i + 1) % 4) for i in range(rows))
                expected = count * rows <= 4 * total
            assert bool(enabled) == expected

    for heads, hk, queries, prefixes in (
        (12, 1, (64,), (30000,)), (6, 1, (64,), (30000,)),
        (24, 2, (64,), (30000,)), (12, 1, (65,), (30000,)),
        (12, 1, (32, 32), (30000, 30000)),
    ):
        value = _make_case(queries, prefixes, heads, device, seed=83)
        if hk == 2:
            value.k = torch.cat((value.k, -value.k), dim=1)
            value.v = torch.cat((value.v, -value.v), dim=1)
        independent = value.indices.clone()
        shared = _make_case(queries, prefixes, heads, device, seed=89, shared=True).indices
        packed = len(queries) == 1 and (sum(queries) + sum(prefixes)) % 4 == 0
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            output = _call(value)
            key = (device, stream.cuda_stream, queries, prefixes, heads, hk, value.scale)
            plan = _runtime._workspaces[key].union
            assert plan.packed_direct == packed
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                _call(value, output)
            for selection in (shared, independent, shared):
                value.indices.copy_(selection)
                value.v.neg_()
                output.fill_(float('nan'))
                graph.replay()
                torch.cuda.synchronize(device)
                ids = _rows(value)
                torch.testing.assert_close(output[ids].float(), reference(value, ids), rtol=.02, atol=.02)
                routes = _audit(value)
                assert routes['union_rows'] > 0 if selection is shared else routes['direct_rows'] > 0
        torch.cuda.current_stream(device).wait_stream(stream)


def test_direct_pack_limit_host(monkeypatch):
    """Budget both packed tensors before allocating, using only host metadata."""
    limit = 64 * 1024 * 1024

    def forbidden(*args, **kwargs):
        raise AssertionError("Packed eligibility must not read device data or free memory")

    for hk in (1, 2, 4):
        for extra_rows in (-4, 0, 4):
            rows = limit // (1024 * hk) + extra_rows
            kw = {"device": "meta", "dtype": torch.bfloat16}
            inputs = SimpleNamespace(
                q=torch.empty((4, 12, 256), **kw),
                k=torch.empty((rows, hk, 256), **kw),
                v=torch.empty((rows, hk, 256), **kw),
                query_lens=(4,), prefix_lens=(rows - 4,),
                query_positions=torch.empty(4, device="meta", dtype=torch.int32),
                query_sequence_ids=torch.empty(4, device="meta", dtype=torch.int32),
                block_indices=torch.empty((4, 512), device="meta", dtype=torch.int32),
            )
            required = inputs.k.numel() * inputs.k.element_size() + inputs.v.numel() * inputs.v.element_size()
            expected = required <= limit
            allocations = []
            empty_like = torch.empty_like

            def allocate(tensor):
                assert expected, "Over-budget plan must fall back before any packed allocation"
                allocations.append(tensor)
                return empty_like(tensor)

            with monkeypatch.context() as patch:
                for method in ("cpu", "item", "tolist", "__bool__"):
                    patch.setattr(torch.Tensor, method, forbidden)
                patch.setattr(torch.cuda, "mem_get_info", forbidden)
                patch.setattr(torch.cuda, "synchronize", forbidden)
                patch.setattr(torch, "empty_like", allocate)
                plan = _runtime.direct.prepare(inputs=inputs)
            assert len(allocations) == (2 if expected else 0)
            assert plan.query_tile == (1 if expected else 4)
            assert plan.num_tiles == (4 if expected else 1)
            assert plan.source_blocks is inputs.block_indices
            if expected:
                assert plan.packed_key.shape == inputs.k.shape and plan.packed_value.shape == inputs.v.shape
                assert sum(t.numel() * t.element_size() for t in (plan.packed_key, plan.packed_value)) == required
            else:
                assert plan.packed_key is None and plan.packed_value is None


def test_h6_query_tiles_host(monkeypatch):
    """The large full-prefill specialization preserves exact row coverage."""
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _: SimpleNamespace(multi_processor_count=80))
    for queries, prefixes, heads, hk, expected_tile in (
        ((5410,), (0,), 6, 1, 16), ((5411,), (0,), 6, 1, 21),
        ((11888,), (0,), 6, 1, 21), ((12000,), (0,), 6, 1, 21),
        ((12000,), (1,), 6, 1, 16), ((6000, 6000), (0, 0), 6, 1, 16),
        ((12000,), (0,), 12, 1, 10), ((12000,), (0,), 3, 1, 32),
        ((12000,), (0,), 12, 2, 16),
    ):
        skips = tuple(min(q, max(0, 2051 - p)) for q, p in zip(queries, prefixes))
        inputs = SimpleNamespace(q=SimpleNamespace(shape=(sum(queries), heads, 256), device=torch.device("cpu")),
                     k=SimpleNamespace(shape=(sum(queries) + sum(prefixes), hk, 256)),
                                 query_lens=queries, prefix_lens=prefixes,
                                 max_seqlen_k=max(q + p for q, p in zip(queries, prefixes)))
        plan = _runtime.prepare.allocate_plan(inputs=inputs, query_tile=32, skip_counts=skips)
        assert plan.query_tile == expected_tile and plan.group_padded == heads // hk
        meta = plan.metadata.tolist()
        actual = [row for first, count, *_ in meta for row in range(first, first + count)]
        expected, start = [], 0
        for count, skip in zip(queries, skips):
            expected.extend(range(start + skip, start + count))
            start += count
        assert actual == expected
        query_tiles = plan.query_tiles.tolist()
        for tile, (first, count, *_) in enumerate(meta):
            assert query_tiles[first:first + count] == [tile] * count
        assert all(query_tiles[row] == -1 for row in set(range(sum(queries))) - set(expected))
        if expected_tile == 21:
            sizes = [row[1] for row in meta]
            assert len(meta) == math.ceil(len(expected) / 21)
            assert max(sizes) - min(sizes) <= 1 and max(sizes) <= 21
            assert min(sizes) >= 20 and plan.grid == 160


def test_union_task_order():
    """Dynamic gates, head-major tasks and partial sort/snake rows keep coverage."""
    device = _gpu()
    for tiles, heads in ((1, 1), (81, 1), (995, 1), (2051, 2), (4097, 1)):
        tasks, grid = tiles * heads, min(tiles * heads, 80)
        capacity = math.ceil(tasks / grid) * grid
        size = min(4096, 1 << (capacity - 1).bit_length())
        counts = np.zeros((tiles, 2), dtype=np.int32)
        counts[:, 0] = (np.arange(tiles) * 79) % 1500 + 1
        active = (np.arange(tiles) % 3 != 0).astype(np.int32)
        counts_gpu = torch.from_numpy(counts).to(device)
        order = torch.empty(capacity, dtype=torch.int32, device=device)
        for enabled in (active, 1 - active):
            active_gpu = torch.from_numpy(enabled.copy()).to(device)
            expected = np.full(capacity, -1, dtype=np.int32)
            for start in range(0, capacity, size):
                ids = list(range(start, min(start + size, tasks)))
                ids.sort(key=lambda t: (-math.ceil(int(counts[t % tiles, 0]) / 16)
                                       if enabled[t % tiles] else 0, t))
                ids.extend([-1] * (size - len(ids)))
                for local, task in enumerate(ids):
                    row, column = divmod(start + local, grid)
                    destination = row * grid + (grid - 1 - column if row % 2 else column)
                    if destination < capacity:
                        expected[destination] = task
            for shift in (max(1, (tasks - 1).bit_length()), 26):
                # Exercise the int64 fallback without allocating millions of tasks.
                wide = (math.ceil(int(counts[:, 0].max()) / 16) << shift) + tasks - 1 >= 2**31
                _runtime.prepare.attention_order_masks_validate[(math.ceil(capacity / size),)](
                    None, None, counts_gpu, active_gpu, order, 0, tasks, heads, 1, grid, size,
                    shift, wide, False, num_warps=8,
                )
                actual = order.cpu().numpy()
                np.testing.assert_array_equal(actual, expected)
                np.testing.assert_array_equal(np.sort(actual[actual >= 0]), np.arange(tasks))
