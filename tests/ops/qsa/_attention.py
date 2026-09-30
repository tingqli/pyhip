# Copyright 2023-2024 SGLang Team
# SPDX-License-Identifier: Apache-2.0
"""Shared QSA attention cases, numerical checks and compiled-resource assertions."""

import hashlib
import importlib
from itertools import accumulate
import math
import os
from pathlib import Path
import re
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.attention.flydsl.qsa.sglang.attention_baseline import baseline
from pyhip.ops.qsa.flydsl.attention import attention

_runtime = importlib.import_module("pyhip.ops.qsa.flydsl.attention")
ROOT = Path(__file__).resolve().parents[3]
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
    return attention(inputs.q, inputs.k, inputs.v, inputs.indices, query_lens=inputs.query_lens,
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
        masks = plan.score_masks.cpu().numpy().view(np.uint32)
        assert sum(item[1] for item in metadata) == inputs.q.shape[0] - dense_rows
        assert not np.any(members), "Consumed membership scratch must be zero for the next call"
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
            assert counts[tile][0] == len(expected)
            if active[tile]:
                count = counts[tile][0]
                assert {int(b): int(m) for b, m in zip(compact[tile, :count], bits[tile, :count])} == expected
                common = sorted(b for b, mask in expected.items()
                                if mask == (1 << rows) - 1 and b * 4 + 3 <= position)
                other = sorted(set(expected) - set(common))
                np.testing.assert_array_equal(compact[tile, :count], common + other)
                assert counts[tile][1] == len(common) // 16
                for nt in range(counts[tile][1], math.ceil(count / 16)):
                    query, quarter = np.arange(plan.query_tile)[:, None], np.arange(4)[None, :]
                    mask = np.zeros((plan.query_tile, 4), dtype=np.uint32)
                    for group in range(4):
                        slot = nt * 16 + quarter * 2 + (group // 2) * 8 + group % 2
                        safe = np.minimum(slot, count - 1)
                        valid = (slot < count) & (query < rows)
                        valid &= ((bits[tile, safe] >> query.astype(np.uint32)) & 1) != 0
                        for offset in range(4):
                            token = compact[tile, safe] * 4 + offset
                            keep = valid & (token <= position + query) & (token < metadata[tile][3])
                            mask |= keep.astype(np.uint32) << (group * 4 + offset)
                    np.testing.assert_array_equal(masks[tile, nt], mask)
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


def _real_files():
    directory = Path(os.environ.get("QSA_REAL_INPUT_DIR", REAL_INPUTS))
    paths = sorted(directory.glob("tp*_layer*_m*.pt"))
    if not paths and "QSA_REAL_INPUT_DIR" in os.environ:
        raise FileNotFoundError(f"No captured QSA inputs in {directory}")
    return paths


@pytest.fixture(scope="module", autouse=True)
def _resources():
    yield
    if not torch.cuda.is_initialized():
        return
    for name, cache in (("attention_union_bf16_d256", _runtime.union._COMPILED),
                        ("attention_direct_bf16_d256", _runtime.direct._COMPILED),
                        ("dense_mha_bf16_d256", _runtime.dense.native._COMPILED),
                        ("attention_dense_bf16_d256_bounded", _runtime.dense._BOUNDED_COMPILED)):
        for compiled in cache.values():
            assert re.findall(r'#gpu\.kernel_metadata<"([^"]+)"', compiled._keepalive.ir) == [name]
            for field in ("private_segment_fixed_size", "vgpr_spill_count", "sgpr_spill_count"):
                values = re.findall(rf"\b{field}\s*=\s*(\d+)", compiled._keepalive.ir)
                assert values and not any(map(int, values))

    from pyhip.ops.qsa.flydsl import attention_direct_packed

    for compiled in attention_direct_packed._COMPILED.values():
        text = compiled._keepalive.ir
        assert re.findall(r'#gpu\.kernel_metadata<"([^\"]+)"', text) == [
            "attention_direct_bf16_d256", "attention_pack_kv_bf16_d256",
        ]
        for field in ("private_segment_fixed_size", "vgpr_spill_count", "sgpr_spill_count"):
            values = re.findall(rf"\b{field}\s*=\s*(\d+)", text)
            assert len(values) == 2 and not any(map(int, values))

    # Inspect actual Triton ELFs too; attention-only checks missed planner
    # SGPR spills and large-context private scratch in earlier versions.
    from experiments.attention.flydsl.qsa.sglang.attention_validation import _selected_attention_fp32

    readelf = Path(os.environ.get("ROCM_PATH", "/opt/rocm")) / "llvm/bin/llvm-readelf"
    functions = (_runtime.prepare.attention_recover_scatter, _runtime.prepare.attention_compact,
                 _runtime.prepare.attention_order_masks_validate, _runtime.prepare.attention_scatter_prepared,
                 _selected_attention_fp32)
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
