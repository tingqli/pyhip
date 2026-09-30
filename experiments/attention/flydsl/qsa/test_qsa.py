# Copyright 2023-2024 SGLang Team
# SPDX-License-Identifier: Apache-2.0
"""QSA normal correctness, real-data replay and performance in one entry point.

Pytest runs numerical/normal-use checks; -m perf enables timing explicitly.
CLI --check-only skips timing. All new results live under mytest/mydata.
"""

import argparse
import gc
import hashlib
import importlib
import importlib.metadata
from itertools import accumulate
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import triton

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    __package__ = "experiments.attention.flydsl.qsa"

from . import qsa
from .sglang.baseline import baseline

_runtime = importlib.import_module(".qsa", __package__)
ROOT = Path(__file__).resolve().parents[4]
DATA = ROOT / "mytest/mydata"
REAL_INPUTS = DATA / "qsa_real_study_20260925/capture/inputs"
TP_SIZES = (2, 4, 8)
BENCHMARK_BUFFERS = 10
BENCHMARK_SAMPLES = 128
CASES = (
    ((0,), (0,), 12), ((1,), (0,), 12), ((64,), (0,), 12),
    ((7, 0, 9, 9, 9, 7), (0, 5, 55, 56, 2050, 3000), 12),
    ((33,), (2047,), 12), ((33,), (2051,), 6),
    ((65,), (30000,), 12), ((65,), (30000,), 6), ((65,), (30000,), 3),
    ((2048,), (0,), 12), ((2051,), (0,), 12),
    ((2057,), (0,), 12), ((33,), (12000,), 12),
)


def _hash(tensor):
    raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes() if tensor.numel() else b""
    return hashlib.sha256(raw).hexdigest()


def _metadata(q, k, v, indices, query_lens, prefix_lens, scale=0.0625, captured=None):
    lengths = tuple(qn + pn for qn, pn in zip(query_lens, prefix_lens))
    kw = {"dtype": torch.int32, "device": q.device}
    return SimpleNamespace(q=q, k=k, v=v, indices=indices, query_lens=tuple(query_lens),
                           prefix_lens=tuple(prefix_lens), scale=scale, captured=captured,
                           cu_q=torch.tensor(tuple(accumulate(query_lens, initial=0)), **kw),
                           cu_k=torch.tensor(tuple(accumulate(lengths, initial=0)), **kw),
                           kv_lens=torch.tensor(lengths, **kw),
                           positions=torch.tensor([p + i for n, p in zip(query_lens, prefix_lens) for i in range(n)], **kw),
                           sequence_ids=torch.tensor([s for s, n in enumerate(query_lens) for _ in range(n)], **kw))


def _make_case(queries, prefixes, heads, device, seed=17, shared=False):
    rows, total = sum(queries), sum(queries) + sum(prefixes)
    generator = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn((rows, heads, 256), generator=generator, device=device, dtype=torch.bfloat16)
    k = torch.randn((total, 1, 256), generator=generator, device=device, dtype=q.dtype)
    v = torch.randn(k.shape, generator=generator, device=device, dtype=q.dtype)
    indices = np.full((rows, 2051), -1, dtype=np.int32)
    rng, row = np.random.default_rng(seed), 0
    for count, prefix in zip(queries, prefixes):
        priority = None
        for local in range(count):
            visible = prefix + local + 1
            blocks = visible // 4
            if blocks <= 512:
                chosen = np.arange(blocks)
            elif shared:
                if priority is None or local % 32 == 0:
                    priority = rng.permutation((prefix + count) // 4)
                chosen = priority[priority < blocks][:512]
            else:
                chosen = rng.choice(blocks, 512, replace=False)
            tokens = np.concatenate(((chosen[:, None] * 4 + np.arange(4)).reshape(-1), np.arange(blocks * 4, visible)))
            indices[row, :len(tokens)] = tokens
            row += 1
    return _metadata(q, k, v, torch.from_numpy(indices).to(device), queries, prefixes)


def _load(path, device):
    value = torch.load(path, map_location="cpu", weights_only=True)
    meta, tensors = value["metadata"], value["tensors"]
    for name, tensor in tensors.items():
        assert _hash(tensor) == meta["tensor_metadata"][name]["sha256"], (path, name)
    result = _metadata(*(tensors[name].to(device) for name in ("q", "k", "v", "indices")),
                       meta["query_lens"], meta["prefix_lens"], meta["scale"], tensors["output"].to(device))
    result.capture = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return result


def _tp_case(value, tp_size):
    """Local-head replay, not a claim of a newly captured multi-GPU TP run."""
    if tp_size not in TP_SIZES or value.q.shape[1] != 12 or value.k.shape[1] != 1:
        raise ValueError("TP2/4/8 replay requires a TP2 H12/HK1 source capture")
    heads = 24 // tp_size
    output = None if value.captured is None else value.captured[:, :heads].contiguous()
    result = _metadata(value.q[:, :heads].contiguous(), value.k, value.v, value.indices,
                       value.query_lens, value.prefix_lens, value.scale, output)
    result.capture = dict(value.capture, source_tp_size=2, local_tp_size=tp_size,
                          local_heads=heads, derived_head_slice=tp_size != 2,
                          distributed_tp_run=False)
    return result


def _call(inputs, out=None):
    return qsa(inputs.q, inputs.k, inputs.v, inputs.indices, query_lens=inputs.query_lens,
               prefix_lens=inputs.prefix_lens, softmax_scale=inputs.scale, out=out)


def _base(inputs, out=None):
    return baseline(inputs.q, inputs.k, inputs.v, inputs.indices, inputs.cu_q, inputs.cu_k, inputs.kv_lens,
                    max_seqlen_q=max(inputs.query_lens, default=0), has_prefix=any(inputs.prefix_lens),
                    softmax_scale=inputs.scale, out=out)


def _rows(inputs, count=32):
    result, start = set(), 0
    for length, prefix in zip(inputs.query_lens, inputs.prefix_lens):
        result.update(start + n for n in (0, 1, 2, 3, 4, 7, 8, 31, 32, 2047 - prefix,
                      2048 - prefix, 2050 - prefix, 2051 - prefix, 2052 - prefix, length - 1) if 0 <= n < length)
        start += length
    if start:
        result.update(np.linspace(0, start - 1, min(start, count), dtype=int).tolist())
    return sorted(result)


@torch.no_grad()
def reference(inputs, rows):
    """FP32 per-query selected-token oracle; never uses union scratch."""
    outputs = []
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        for start in range(0, len(rows), 4):
            ids = torch.tensor(rows[start:start + 4], device=inputs.q.device)
            tokens = inputs.indices[ids].long()
            slots = inputs.cu_k[inputs.sequence_ids[ids].long(), None].long() + tokens.clamp_min(0)
            keys, values = inputs.k[slots].float(), inputs.v[slots].float()
            queries = inputs.q[ids].float().reshape(-1, inputs.k.shape[1], inputs.q.shape[1] // inputs.k.shape[1], 256)
            scores = torch.einsum("bghd,bkgd->bghk", queries, keys) * inputs.scale
            scores.masked_fill_(tokens[:, None, None, :] < 0, -float("inf"))
            outputs.append(torch.einsum("bghk,bkgd->bghd", scores.softmax(-1), values).reshape(-1, inputs.q.shape[1], 256))
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous
    return torch.cat(outputs) if outputs else inputs.q.float()


def _audit(inputs):
    if not inputs.q.shape[0]:
        return {"rows": 0, "dense_rows": 0, "union_rows": 0, "direct_rows": 0}
    key = (inputs.q.device, torch.cuda.current_stream(inputs.q.device).cuda_stream,
           inputs.query_lens, inputs.prefix_lens, inputs.q.shape[1], inputs.k.shape[1], inputs.scale)
    workspace = _runtime._workspaces[key]
    blocks = workspace.metadata["block_indices"].cpu().numpy()
    indices, positions = inputs.indices.cpu().numpy(), inputs.positions.cpu().tolist()
    for row, position in enumerate(positions):
        count = min((position + 1) // 4, 512)
        chosen = blocks[row, :count]
        assert len(set(chosen.tolist())) == count and np.all(chosen >= 0) and np.all(chosen < (position + 1) // 4)
        tokens = np.concatenate(((chosen[:, None] * 4 + np.arange(4)).reshape(-1), np.arange((position + 1) // 4 * 4, position + 1)))
        np.testing.assert_array_equal(np.sort(indices[row, :len(tokens)]), tokens)
        assert np.all(indices[row, len(tokens):] == -1)
    dense_rows = sum(workspace.dense.query_counts)
    assert workspace.dense.query_counts == tuple(min(n, max(0, 2051 - p)) for n, p in zip(inputs.query_lens, inputs.prefix_lens))
    result = {"rows": inputs.q.shape[0], "dense_rows": dense_rows, "union_rows": 0, "direct_rows": 0}
    if workspace.union is not None:
        plan = workspace.union
        metadata, counts, active = plan.metadata.cpu().tolist(), plan.counts.cpu().tolist(), plan.active.cpu().tolist()
        members = plan.dense_membership.cpu().numpy().view(np.uint32)
        compact, bits = plan.blocks.cpu().numpy(), plan.membership.cpu().numpy().view(np.uint32)
        assert sum(item[1] for item in metadata) == inputs.q.shape[0] - dense_rows
        union_rows = 0
        for tile, (first, rows, _, _, position) in enumerate(metadata):
            expected = {}
            for local in range(rows):
                for b in blocks[first + local]:
                    if b >= 0:
                        expected[int(b)] = expected.get(int(b), 0) | (1 << local)
                visible = position + local + 1
                if visible % 4:
                    b = visible // 4
                    expected[b] = expected.get(b, 0) | (1 << local)
            assert {int(b): int(members[tile, b]) for b in members[tile].nonzero()[0]} == expected
            assert counts[tile][0] == len(expected)
            if active[tile]:
                count = counts[tile][0]
                assert {int(b): int(m) for b, m in zip(compact[tile, :count], bits[tile, :count])} == expected
            total = sum(min((position + i + 1) // 4, 512) + bool((position + i + 1) % 4) for i in range(rows))
            if plan.packed_direct:
                union_work = math.ceil(len(expected) / 16) * 64 * 128
                direct_work = sum(math.ceil((min((position + i + 1) // 4, 512) * 4
                                            + (position + i + 1) % 4) / 32) * 32 * 16
                                  for i in range(rows))
                assert active[tile] == (10 * union_work <= 17 * direct_work)
            else:
                assert active[tile] == (len(expected) * rows <= 4 * total)
            union_rows += rows * active[tile]
        result.update(union_rows=union_rows, direct_rows=inputs.q.shape[0] - dense_rows - union_rows)
    return result


@torch.no_grad()
def check(inputs, *, compare_baseline=True):
    """Normal calls jointly cover numerical output, guards, reuse and current selection."""
    original_hashes = {n: _hash(getattr(inputs, n)) for n in ("q", "k", "v", "indices")}
    storage = torch.full((inputs.q.shape[0] + 2, *inputs.q.shape[1:]), 123.0, device=inputs.q.device, dtype=inputs.q.dtype)
    output = storage[1:-1]
    output.fill_(float("nan"))
    assert _call(inputs, output) is output
    assert bool(torch.isfinite(output).all()) and bool((storage[[0, -1]] == 123).all())
    if compare_baseline:
        torch.testing.assert_close(output, _base(inputs), rtol=0.02, atol=0.02)
    if inputs.captured is not None:
        torch.testing.assert_close(output, inputs.captured, rtol=0.02, atol=0.02)
    ids = list(range(inputs.q.shape[0])) if inputs.q.shape[0] <= 65 else _rows(inputs)
    torch.testing.assert_close(output[ids].float(), reference(inputs, ids), rtol=0.02, atol=0.02)
    first = output.clone()
    output.fill_(float("nan"))
    _call(inputs, output)
    torch.testing.assert_close(output, first, rtol=0, atol=0)
    routes = _audit(inputs)
    assert original_hashes == {n: _hash(getattr(inputs, n)) for n in original_hashes}
    return routes


def _gpu():
    if not torch.cuda.is_available() or torch.version.hip is None:
        pytest.skip("requires ROCm gfx942")
    torch.cuda.set_device(int(os.environ.get("QSA_REPLAY_GPU", "0")))
    if not torch.cuda.get_device_properties().gcnArchName.startswith("gfx942"):
        pytest.skip("requires gfx942")
    return torch.device("cuda", torch.cuda.current_device())


@pytest.mark.parametrize("queries,prefixes,heads", CASES)
def test_qsa(queries, prefixes, heads):
    device = _gpu()
    value = _make_case(queries, prefixes, heads, device)
    if sum(queries):
        for name in ("k", "v"):
            original = getattr(value, name)
            storage = torch.full((original.shape[0] + 4, 1, 256), float("nan"), device=device, dtype=original.dtype)
            storage[:original.shape[0]].copy_(original)
            setattr(value, name, storage[:original.shape[0]])
        with pytest.raises(ValueError, match="overlap"):
            _call(value, value.q)
    if prefixes == (12000,):
        # Disjoint selections and large logits exercise rescaling. The old baseline
        # pre-scales BF16 Q and is not the FP32 oracle for this construction.
        for row in range(queries[0]):
            blocks = torch.arange(512, device=device) + (row % 4) * 600
            value.indices[row, :2048] = (blocks[:, None] * 4 + torch.arange(4, device=device)).flatten()
        value.q.mul_(3)
        value.k.mul_(3)
    check(value, compare_baseline=prefixes != (12000,))
    if prefixes == (2047,):
        check(_make_case(queries, prefixes, heads, device, shared=True))
    if prefixes == (30000,) and heads == 12:
        torch.testing.assert_close(qsa(value.q, value.k, value.v, value.indices), _call(value), rtol=0, atol=0)
        for scale in (float("nan"), 0, -1, True, torch.tensor(0.0625, device=device)):
            with pytest.raises(ValueError, match="scale"):
                qsa(value.q, value.k, value.v, value.indices, softmax_scale=scale)
        with pytest.raises(ValueError, match="Host"):
            qsa(value.q, value.k, value.v, value.indices, query_lens=(1,))
        with pytest.raises(ValueError, match="16 Q heads"):
            qsa(torch.empty((queries[0], 17, 256), device=device, dtype=value.q.dtype), value.k, value.v, value.indices)
    if prefixes == (30000,):
        independent = value.indices.clone()
        changed = _make_case(queries, prefixes, heads, device, seed=51, shared=True)
        value.indices.copy_(changed.indices)
        check(value)
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            output = _call(value)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                _call(value, output)
        torch.cuda.current_stream(device).wait_stream(stream)
        value.v.neg_()
        for indices in (independent, changed.indices):
            value.indices.copy_(indices)
            output.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize(device)
            torch.testing.assert_close(output.float(), reference(value, list(range(queries[0]))), rtol=0.02, atol=0.02)


def _real_files():
    directory = Path(os.environ.get("QSA_REAL_INPUT_DIR", REAL_INPUTS))
    paths = sorted(directory.glob("tp*_layer*_m*.pt"))
    if not paths and "QSA_REAL_INPUT_DIR" in os.environ:
        raise FileNotFoundError(f"No captured QSA inputs in {directory}")
    return paths


@pytest.mark.parametrize("path", _real_files(), ids=lambda p: p.stem)
@pytest.mark.parametrize("tp_size", TP_SIZES, ids=lambda tp: f"tp{tp}")
def test_real_qsa(path, tp_size):
    check(_tp_case(_load(path, _gpu()), tp_size))


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
        _runtime._qsa_recover_blocks[(9,)](
            value.indices, inputs.query_positions, inputs.kv_lens,
            inputs.query_sequence_ids, inputs.block_indices, workspace.errors, num_warps=4,
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
        _runtime._qsa_recover_blocks[(5,)](
            indices, inputs.query_positions, inputs.kv_lens,
            inputs.query_sequence_ids, inputs.block_indices, workspace.errors, num_warps=4,
        )
        assert workspace.errors.cpu().tolist() == [int(corruption != "none"), 0, 0, 0, 0]
        _runtime._qsa_check_errors[(1,)](workspace.errors, workspace.valid, 5, 1024, num_warps=4)
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
        _runtime._qsa_check_errors[(1,)](errors, valid, rows, 1024, num_warps=4)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            _runtime._qsa_check_errors[(1,)](errors, valid, rows, 1024, num_warps=4)
    torch.cuda.current_stream(device).wait_stream(stream)
    for row in sorted({0, min(1023, rows - 1), min(1024, rows - 1), rows - 1}):
        for error in (1, -1, 0):
            errors[row] = error
            valid.fill_(error != 0)
            graph.replay()
            torch.cuda.synchronize(device)
            assert bool(valid.cpu()) == (error == 0)
            assert bool(storage[0].cpu()) and bool(storage[2].cpu())


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
        _runtime.union.union_qsa_compact_membership[(len(rows),)](
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
        _runtime._qsa_recover_blocks[(length,)](
            value.indices, inputs.query_positions, inputs.kv_lens,
            inputs.query_sequence_ids, inputs.block_indices, workspace.errors, num_warps=4,
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
    _runtime._qsa_recover_blocks[(8,)](
        value.indices, inputs.query_positions, inputs.kv_lens,
        inputs.query_sequence_ids, inputs.block_indices, workspace.errors, num_warps=4,
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
        _runtime.union.union_qsa_compact_membership[(len(cases),)](
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


def test_direct_scratch_reuse(monkeypatch):
    """Size from host KV shape, reuse without GPU readback, and pin graph buffers."""
    device = _gpu()
    for tp_size in TP_SIZES:
        heads = 24 // tp_size
        value = _make_case((68,), (30000,), heads, device, seed=93)
        shared = _make_case((68,), (30000,), heads, device, seed=97, shared=True).indices
        independent = value.indices.clone()
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            output = _call(value)
            key = (device, stream.cuda_stream, value.query_lens, value.prefix_lens, heads, 1, value.scale)
            workspace = _runtime._workspaces[key]
            plan = workspace.direct
            assert plan.packed_key.shape == value.k.shape and plan.packed_value.shape == value.v.shape
            expected_bytes = 2 * value.k.numel() * value.k.element_size()
            assert sum(t.untyped_storage().nbytes() for t in (plan.packed_key, plan.packed_value)) == expected_bytes
            pointers = (plan.packed_key.data_ptr(), plan.packed_value.data_ptr(), workspace.valid.data_ptr())

            def forbidden(*args, **kwargs):
                raise AssertionError("Hot QSA must reuse scratch without host reads or Torch bool reduction")

            # The zero-spill validation kernel must replace Tensor equality/all,
            # not merely run beside the old scratch-using Torch reduction.
            with monkeypatch.context() as patch:
                patch.setattr(torch.Tensor, "cpu", forbidden)
                patch.setattr(torch.Tensor, "item", forbidden)
                patch.setattr(torch.Tensor, "tolist", forbidden)
                patch.setattr(torch.Tensor, "__bool__", forbidden)
                patch.setattr(torch.Tensor, "__eq__", forbidden)
                patch.setattr(torch.Tensor, "all", forbidden)
                patch.setattr(torch, "empty", forbidden)
                patch.setattr(torch, "empty_like", forbidden)
                _call(value, output)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    _call(value, output)
            assert workspace.captured
            assert pointers == (workspace.direct.packed_key.data_ptr(), workspace.direct.packed_value.data_ptr(), workspace.valid.data_ptr())
            for selection in (shared, independent):
                value.indices.copy_(selection)
                value.v.neg_()
                output.fill_(float("nan"))
                graph.replay()
                torch.cuda.synchronize(device)
                ids = _rows(value)
                torch.testing.assert_close(output[ids].float(), reference(value, ids), rtol=.02, atol=.02)
                assert pointers == (workspace.direct.packed_key.data_ptr(), workspace.direct.packed_value.data_ptr(), workspace.valid.data_ptr())
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


@pytest.mark.parametrize("tp_size", TP_SIZES, ids=lambda tp: f"tp{tp}")
def test_direct_pack_limit(tp_size, monkeypatch):
    """The real byte cutoff retains raw routing, output guards and graph updates."""
    from . import _direct_packed

    device = _gpu()
    limit = 64 * 1024 * 1024

    def forbidden(*args, **kwargs):
        raise AssertionError("Over-budget QSA must not allocate or run packed KV")

    for hk in (1, 2):
        heads = (24 // tp_size) * hk
        for extra_rows in (0, 4):
            rows = limit // (1024 * hk) + extra_rows
            value = _make_case((68,), (rows - 68,), heads, device, seed=103)
            if hk == 2:
                value.k = torch.cat((value.k, -value.k), dim=1)
                value.v = torch.cat((value.v, -value.v), dim=1)
            independent = value.indices.clone()
            shared = _make_case(value.query_lens, value.prefix_lens, heads, device, seed=107, shared=True).indices
            expected = extra_rows == 0
            stream = torch.cuda.Stream(device=device)
            stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(stream):
                storage = torch.full((70, heads, 256), 123.0, device=device, dtype=value.q.dtype)
                output = storage[1:-1]
                with monkeypatch.context() as patch:
                    if not expected:
                        patch.setattr(torch, "empty_like", forbidden)
                        patch.setattr(_direct_packed, "run", forbidden)
                    _call(value, output)
                key = (device, stream.cuda_stream, value.query_lens, value.prefix_lens, heads, hk, value.scale)
                workspace = _runtime._workspaces[key]
                plan = workspace.direct
                assert workspace.union.packed_direct == expected
                assert plan.query_tile == (1 if expected else 4)
                assert plan.num_tiles == math.ceil(68 / plan.query_tile)
                if expected:
                    assert sum(t.untyped_storage().nbytes() for t in (plan.packed_key, plan.packed_value)) == limit
                else:
                    assert plan.packed_key is None and plan.packed_value is None
                torch.testing.assert_close(output.float(), reference(value, list(range(68))), rtol=.02, atol=.02)
                assert _audit(value)["direct_rows"] > 0
                before = output.clone()
                with monkeypatch.context() as patch:
                    for method in ("cpu", "item", "tolist", "__bool__"):
                        patch.setattr(torch.Tensor, method, forbidden)
                    patch.setattr(torch, "empty_like", forbidden)
                    if not expected:
                        patch.setattr(_direct_packed, "run", forbidden)
                    _call(value, output)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        _call(value, output)
                torch.testing.assert_close(output, before, rtol=0, atol=0)
                assert workspace.captured
                for selection in (shared, independent):
                    value.indices.copy_(selection)
                    value.v.neg_()
                    output.fill_(float("nan"))
                    graph.replay()
                    torch.cuda.synchronize(device)
                    torch.testing.assert_close(output.float(), reference(value, list(range(68))), rtol=.02, atol=.02)
                    assert bool((storage[[0, -1]] == 123).all())
                    routes = _audit(value)
                    assert routes["union_rows"] > 0 if selection is shared else routes["direct_rows"] > 0
                    assert workspace.direct is plan and workspace.union.packed_direct == expected
            torch.cuda.current_stream(device).wait_stream(stream)


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
            _runtime.union.union_qsa_order_tasks[(math.ceil(capacity / size),)](
                counts_gpu, active_gpu, order, tasks, heads, 1, grid, size,
                max(1, (tasks - 1).bit_length()), num_warps=8,
            )
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
            actual = order.cpu().numpy()
            np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(np.sort(actual[actual >= 0]), np.arange(tasks))


def test_sglang_adapter(monkeypatch, tmp_path):
    """One normal backend flow covers cache/padding, profile capture and lazy exclusion."""
    pytest.importorskip("sglang")
    from .sglang import plugin
    from sglang.srt.distributed.parallel_state_wrapper import ParallelState
    from sglang.srt.layers.attention.qsa.config import QSAProfile
    from sglang.srt.layers.attention.qwen_sparse_attn_backend import QwenSparseAttnBackend
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    from unittest.mock import Mock

    value = _make_case((33,), (3000,), 12, _gpu())
    pool = Mock()
    pool.get_key_buffer.return_value = value.k
    pool.get_value_buffer.return_value = value.v
    backend = QwenSparseAttnBackend.__new__(QwenSparseAttnBackend)
    backend.runner = SimpleNamespace(is_draft_worker=False, kv_cache_dtype=torch.bfloat16, ps=ParallelState.trivial())
    backend.token_to_kv_pool = pool
    backend.req_to_token_pool = SimpleNamespace(req_to_token=torch.arange(3033, device=value.q.device)[None])
    backend.qsa_profile = QSAProfile("compressed", 8, 1, 128, 2048, 4, "mrope", False)
    layer = SimpleNamespace(tp_q_head_num=12, tp_k_head_num=1, tp_v_head_num=1, head_dim=256,
                            v_head_dim=256, scaling=0.0625, layer_id=3, logit_cap=0,
                            sliding_window_size=-1, is_cross_attention=False, pos_encoding_mode="NONE")
    batch = SimpleNamespace(forward_mode=ForwardMode.EXTEND, extend_seq_lens_cpu=[33], seq_lens_cpu=[3033],
                            extend_seq_lens=torch.tensor([33], dtype=torch.int32, device=value.q.device),
                            req_pool_indices=torch.tensor([0], device=value.q.device),
                            out_cache_loc=torch.arange(33, device=value.q.device))
    module = importlib.import_module("sglang.srt.layers.attention.qwen_sparse_attn_backend")
    original = QwenSparseAttnBackend.forward_extend
    padded = torch.cat((value.q, torch.full((3, 12, 256), float("nan"), device=value.q.device, dtype=value.q.dtype)))
    expected = original(backend, padded, value.k, value.v, layer, batch, topk_indices=value.indices)
    plain, chunk = module.sparse_gqa_fwd_interface_triton, module.sparse_gqa_fwd_interface_triton_ck
    monkeypatch.setattr(module, "sparse_gqa_fwd_interface_triton", lambda *a: plugin._around_sparse(plain, *a))
    monkeypatch.setattr(module, "sparse_gqa_fwd_interface_triton_ck", lambda *a: plugin._around_sparse(chunk, *a, chunk=True))
    monkeypatch.setenv("PYHIP_QSA_VALIDATE", "1")
    monkeypatch.setenv("PYHIP_QSA_REPORT_DIR", str(tmp_path))
    monkeypatch.setenv("PYHIP_QSA_DUMP_LAYERS", "3")
    monkeypatch.setenv("PYHIP_QSA_DUMP_ROWS", "33")
    monkeypatch.setenv("PYHIP_QSA_DUMP_DIR", str(tmp_path))
    monkeypatch.setattr(plugin, "_state", None)
    pool.reset_mock()
    actual = plugin._around_forward(original, backend, padded, value.k, value.v, layer, batch, topk_indices=value.indices)
    pool.set_kv_buffer.assert_called_once()
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)
    assert bool((actual[-3:] == 0).all())
    manager = SimpleNamespace(ps=backend.runner.ps)
    failed = lambda *_: SimpleNamespace(success=False)
    succeeded = lambda *_: SimpleNamespace(success=True)
    plugin._profile_start(failed, manager)
    assert not plugin._state.profiling
    plugin._profile_start(succeeded, manager)
    actual = plugin._around_forward(original, backend, padded, value.k, value.v, layer, batch,
                                     save_kv_cache=False, topk_indices=value.indices)
    saved = value.q.clone()
    value.q.zero_()
    plugin._profile_stop(succeeded, manager)
    dump_path, = tmp_path.glob("tp0_layer3_m33_*.pt")
    dump = torch.load(dump_path, weights_only=True)
    torch.testing.assert_close(dump["tensors"]["q"], saved.cpu(), rtol=0, atol=0)
    report = json.loads((tmp_path / "qsa_tp0.json").read_text())
    assert report["calls_per_layer"] == {"3": 1} and len(report["validation"]) == 1
    check(_load(dump_path, value.q.device))
    plugin._profile_stop(failed, manager)
    assert not plugin._state.profiling
    batch.forward_mode = ForwardMode.DECODE
    fallback = Mock(return_value="unchanged")
    assert plugin._around_forward(fallback, backend, value.q, value.k, value.v, layer, batch) == "unchanged"


def _gate(folder, phase, gpu):
    from tests.ops.gr_read.test_gr_read import read_hardware, validate_hardware

    snapshot = read_hardware(gpu, Path("/opt/rocm-7.14/bin/amd-smi"))
    props = torch.cuda.get_device_properties(gpu)
    pci = f"{props.pci_domain_id:04x}:{props.pci_bus_id:02x}:{props.pci_device_id:02x}.0"
    snapshot["runtime_pci"] = pci
    (folder / f"hardware_{phase}.json").write_text(json.dumps(snapshot, indent=2))
    validate_hardware(snapshot)
    assert pci.lower() == snapshot["card"]["PCI Bus"].lower()


def benchmark(value, folder, gpu, *, buffers=BENCHMARK_BUFFERS, warmup=2, samples=BENCHMARK_SAMPLES):
    from pyhip.testing.misc import cudaPerf
    from tests.ops.gr_read.test_gr_read import tensor_address

    if buffers < 1 or samples < buffers or warmup < 0:
        raise ValueError("Require samples >= buffers >= 1 and warmup >= 0")
    if not folder.resolve().is_relative_to(DATA.resolve()):
        raise ValueError("Results must stay under mytest/mydata")
    folder.mkdir(parents=True, exist_ok=False)
    sources = [*Path(__file__).parent.glob("*.py"), *Path(__file__).parent.glob("sglang/*.py"),
               Path(__file__).parent.parent / "mha/_common.py",
               Path(__file__).parent.parent / "mha/mha_pa_bf16_256_linear_942.py",
               ROOT / "src/pyhip/testing/misc.py", ROOT / "tests/ops/gr_read/test_gr_read.py"]
    hashes = lambda: {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    report = {"complete": False, "raw": [], "buffers": buffers, "warmup": warmup, "samples": samples,
              "scope": "qsa: recover+validate+rebuild+dispatch; base: frozen sparse kernel; preallocated outputs, no JIT/indexer/KV gather",
              "query_lens": value.query_lens, "prefix_lens": value.prefix_lens, "scale": value.scale,
              "q_shape": list(value.q.shape), "k_shape": list(value.k.shape), "dtype": str(value.q.dtype),
              "torch": torch.__version__, "hip": torch.version.hip, "source_sha256": hashes(),
              "capture": getattr(value, "capture", None), "gpu": gpu,
              "packages": {n: importlib.metadata.version(n) for n in ("triton", "flydsl")}}
    try:
        _gate(folder, "before", gpu)
        check(value)
        values = [value] + [_metadata(*(getattr(value, n).clone() for n in ("q", "k", "v", "indices")),
                            value.query_lens, value.prefix_lens, value.scale, value.captured) for _ in range(buffers - 1)]
        outputs, bases = [torch.empty_like(v.q) for v in values], [torch.empty_like(v.q) for v in values]
        report["addresses"] = []
        for data, output, base_out in zip(values, outputs, bases):
            _call(data, output)
            _base(data, base_out)
            torch.testing.assert_close(output, outputs[0], rtol=0, atol=0)
            torch.testing.assert_close(output, base_out, rtol=0.02, atol=0.02)
            report["addresses"].append({name: tensor_address(t, output=name in ("out", "base_out")) for name, t in (
                ("q", data.q), ("k", data.k), ("v", data.v), ("indices", data.indices), ("out", output), ("base_out", base_out))})
            for _ in range(warmup):
                _call(data, output); _base(data, base_out)
        assert all(len({a[name]["pointer"] for a in report["addresses"]}) == buffers for name in report["addresses"][0])
        expected_qsa = outputs[0].clone()
        for output, base_out in zip(outputs, bases):
            output.fill_(float("nan")); base_out.fill_(float("nan"))
        torch.cuda.synchronize(gpu)
        _gate(folder, "before_samples", gpu)
        timer = cudaPerf(name="qsa", verbose=0)
        if not timer.enable:
            raise RuntimeError("CUDAPERF disabled timing")
        for sample in range(samples):
            bi = sample % buffers
            for name in (("base", "qsa") if sample % 2 == 0 else ("qsa", "base")):
                with timer:
                    (_base if name == "base" else _call)(values[bi], bases[bi] if name == "base" else outputs[bi])
                elapsed = timer.latencies[-1] * 1e6
                report["raw"].append({"scope": name, "sample": sample, "buffer": bi, "us": elapsed})
                assert math.isfinite(elapsed) and elapsed > 0
        for data, output, expected in zip(values, outputs, bases):
            torch.testing.assert_close(output, expected_qsa, rtol=0, atol=0)
            torch.testing.assert_close(output, expected, rtol=0.02, atol=0.02)
            for name in ("q", "k", "v", "indices"):
                torch.testing.assert_close(getattr(data, name), getattr(value, name), rtol=0, atol=0)
        work = int((value.indices >= 0).sum()) * 4 * value.q.shape[1] * 256
        report["summary"] = {}
        base_us = statistics.median(r["us"] for r in report["raw"] if r["scope"] == "base")
        for name in ("base", "qsa"):
            elapsed = statistics.median(r["us"] for r in report["raw"] if r["scope"] == name)
            report["summary"][name] = {"median_us": elapsed, "ratio_to_base": elapsed / base_us,
                                         "effective_tflops": work / elapsed / 1e6,
                                         "paired_ratio_median": statistics.median(
                                             next(r["us"] for r in report["raw"] if r["scope"] == name and r["sample"] == i) /
                                             next(r["us"] for r in report["raw"] if r["scope"] == "base" and r["sample"] == i)
                                             for i in range(samples))}
        report["routes"] = _audit(value)
        report["useful_flops"] = work
        report["timed_outputs_bitexact"] = True
        assert hashes() == report["source_sha256"], "Source changed during measurement"
        for source in sources:
            target = folder / "source" / source.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        _gate(folder, "after", gpu)
        report["complete"] = True
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        (folder / "result.json").write_text(json.dumps(report, indent=2))
    return report


@pytest.mark.perf
@pytest.mark.parametrize("path", _real_files(), ids=lambda p: p.stem)
@pytest.mark.parametrize("tp_size", TP_SIZES, ids=lambda tp: f"tp{tp}")
def test_qsa_performance(path, tp_size):
    device = _gpu()
    output = Path(os.environ["QSA_REPLAY_OUTPUT"])
    assert output.resolve().is_relative_to(DATA.resolve())
    result = benchmark(_tp_case(_load(path, device), tp_size), output / f"{path.stem}_tp{tp_size}", device.index)
    assert result["complete"]
    if os.environ.get("QSA_REPLAY_REQUIRE_HALF") == "1":
        assert result["summary"]["qsa"]["ratio_to_base"] <= 0.5


@pytest.fixture(scope="module", autouse=True)
def _resources():
    yield
    if not torch.cuda.is_initialized():
        return
    for name, cache in (("union_qsa_bf16_d256", _runtime.union._COMPILED),
                        ("direct_qsa_bf16_d256", _runtime.direct._COMPILED),
                        ("dense_mha_bf16_d256", _runtime.dense.native._COMPILED),
                        ("dense_qsa_bf16_d256_bounded", _runtime.dense._BOUNDED_COMPILED)):
        for compiled in cache.values():
            assert re.findall(r'#gpu\.kernel_metadata<"([^"]+)"', compiled._keepalive.ir) == [name]
            for field in ("private_segment_fixed_size", "vgpr_spill_count", "sgpr_spill_count"):
                values = re.findall(rf"\b{field}\s*=\s*(\d+)", compiled._keepalive.ir)
                assert values and not any(map(int, values))

    from . import _direct_packed

    for compiled in _direct_packed._COMPILED.values():
        text = compiled._keepalive.ir
        assert re.findall(r'#gpu\.kernel_metadata<"([^\"]+)"', text) == [
            "direct_pack_kv_bf16_d256", "direct_qsa_bf16_d256",
        ]
        for field in ("private_segment_fixed_size", "vgpr_spill_count", "sgpr_spill_count"):
            values = re.findall(rf"\b{field}\s*=\s*(\d+)", text)
            assert len(values) == 2 and not any(map(int, values))

    # Inspect actual Triton ELFs too; attention-only checks missed planner
    # SGPR spills and large-context private scratch in earlier versions.
    readelf = Path(os.environ.get("ROCM_PATH", "/opt/rocm")) / "llvm/bin/llvm-readelf"
    functions = (_runtime._qsa_recover_blocks, _runtime._qsa_check_errors,
                 _runtime.union.union_qsa_scatter_membership, _runtime.union.union_qsa_compact_membership,
                 _runtime.union.union_qsa_score_masks, _runtime.union.union_qsa_order_tasks)
    seen = set()
    for function in functions:
        for cache in function.device_caches.values():
            for compiled in cache[0].values():
                binary = compiled.asm["hsaco"]
                digest = hashlib.sha256(binary).digest()
                if digest in seen:
                    continue
                seen.add(digest)
                notes = subprocess.run([str(readelf), "--notes", "-"], input=binary,
                                       stdout=subprocess.PIPE, check=True).stdout.decode()
                for field in ("private_segment_fixed_size", "vgpr_spill_count", "sgpr_spill_count"):
                    values = re.findall(r"\." + field + r":\s+(\d+)", notes)
                    assert values and not any(map(int, values)), (function.__name__, field, values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--inputs", nargs="*", type=Path)
    parser.add_argument("--tp-sizes", nargs="+", type=int, choices=TP_SIZES, default=TP_SIZES,
                        help="local-head replay sizes; TP4/8 derive H6/H3 from each TP2 capture")
    parser.add_argument("--buffers", type=int, default=BENCHMARK_BUFFERS,
                        help="independent input/output buffers (default: 10)")
    parser.add_argument("--samples", type=int, default=BENCHMARK_SAMPLES,
                        help="samples per implementation (default: 128)")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--require-half", action="store_true", help="require full QSA <= half the frozen baseline")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.buffers < 1 or args.samples < args.buffers:
        parser.error("Require samples >= buffers >= 1")
    if any(os.environ.get(n) for n in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "HSA_CU_MASK", "ROC_GLOBAL_CU_MASK")):
        raise RuntimeError("Use unmasked physical GPU indices")
    assert args.output.resolve().is_relative_to(DATA.resolve())
    args.output.mkdir(parents=True, exist_ok=False)
    torch.cuda.set_device(args.gpu)
    paths = args.inputs if args.inputs is not None else _real_files()
    if not paths:
        raise ValueError("No captured inputs; pass --inputs or QSA_REAL_INPUT_DIR")
    for path in paths:
        original = _load(path, f"cuda:{args.gpu}")
        for tp_size in args.tp_sizes:
            value = _tp_case(original, tp_size)
            routes = check(value)
            label = f"{path.stem}_tp{tp_size}"
            result = {"complete": True, "routes": routes, "capture": value.capture} if args.check_only else benchmark(
                value, args.output / label, args.gpu, buffers=args.buffers, samples=args.samples)
            print(label, result.get("summary", routes), flush=True)
            if args.require_half and not args.check_only:
                assert result["summary"]["qsa"]["ratio_to_base"] <= 0.5
            del value
        del original
        gc.collect()
    (args.output / "checks.json").write_text(json.dumps({"inputs": [str(p) for p in paths], "tp_sizes": args.tp_sizes,
                                                        "passed": len(paths) * len(args.tp_sizes), "check_only": args.check_only,
                                                        "buffers": args.buffers, "samples": args.samples}, indent=2))


if __name__ == "__main__":
    main()
