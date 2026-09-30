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
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    __package__ = "experiments.attention.flydsl.qsa"

from . import qsa
from .sglang.baseline import baseline

_runtime = importlib.import_module(".qsa", __package__)
ROOT = Path(__file__).resolve().parents[4]
DATA = ROOT / "mytest/mydata"
REAL_INPUTS = DATA / "qsa_real_study_20260925/capture/inputs"
CASES = (
    ((0,), (0,), 12), ((1,), (0,), 12), ((64,), (0,), 12),
    ((7, 0, 9, 9, 9, 7), (0, 5, 55, 56, 2050, 3000), 12),
    ((33,), (2047,), 12), ((33,), (2051,), 6),
    ((65,), (30000,), 12), ((65,), (30000,), 6), ((65,), (30000,), 3),
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
                if local % 32 == 0:
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
        np.testing.assert_array_equal(indices[row, :len(tokens)], tokens)
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
def test_real_qsa(path):
    check(_load(path, _gpu()))


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


def benchmark(value, folder, gpu, *, buffers=10, warmup=2, samples=10):
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
def test_qsa_performance(path):
    device = _gpu()
    output = Path(os.environ["QSA_REPLAY_OUTPUT"])
    assert output.resolve().is_relative_to(DATA.resolve())
    result = benchmark(_load(path, device), output / path.stem, device.index)
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--inputs", nargs="*", type=Path)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--require-half", action="store_true", help="require full QSA <= half the frozen baseline")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if any(os.environ.get(n) for n in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "HSA_CU_MASK", "ROC_GLOBAL_CU_MASK")):
        raise RuntimeError("Use unmasked physical GPU indices")
    assert args.output.resolve().is_relative_to(DATA.resolve())
    args.output.mkdir(parents=True, exist_ok=False)
    torch.cuda.set_device(args.gpu)
    paths = args.inputs if args.inputs is not None else _real_files()
    if not paths:
        raise ValueError("No captured inputs; pass --inputs or QSA_REAL_INPUT_DIR")
    for path in paths:
        value = _load(path, f"cuda:{args.gpu}")
        routes = check(value)
        result = {"complete": True, "routes": routes} if args.check_only else benchmark(value, args.output / path.stem, args.gpu)
        print(path.stem, result.get("summary", routes), flush=True)
        if args.require_half and not args.check_only:
            assert result["summary"]["qsa"]["ratio_to_base"] <= 0.5
        del value
        gc.collect()
    (args.output / "checks.json").write_text(json.dumps({"inputs": [str(p) for p in paths], "passed": len(paths), "check_only": args.check_only}, indent=2))


if __name__ == "__main__":
    main()
