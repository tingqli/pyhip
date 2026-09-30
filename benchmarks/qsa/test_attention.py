# Copyright 2023-2024 SGLang Team
# SPDX-License-Identifier: Apache-2.0
"""QSA attention correctness, real-data replay and performance in one entry point.

Pytest runs numerical/normal-use checks; -m perf enables timing explicitly.
CLI --check-only skips timing; --synthetic [ROWS ...] needs no captures.
All new results live under mytest/mydata.
"""

import argparse
import gc
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import sys
from types import SimpleNamespace

import pytest
import torch

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pyhip.ops.qsa.flydsl.attention import attention
from tests.ops.qsa._attention import (
    BENCHMARK_BUFFERS,
    BENCHMARK_SAMPLES,
    CASES,
    DATA,
    ROOT,
    TP_SIZES,
    _audit,
    _base,
    _call,
    _gpu,
    _hash,
    _load,
    _make_case,
    _metadata,
    _real_files,
    _resources,
    _rows,
    _runtime,
    _tp_case,
    check,
    reference,
)
from tests.ops.qsa._benchmark import source_files, write_summary


@pytest.mark.parametrize("queries,prefixes,heads", CASES)
def test_attention(queries, prefixes, heads):
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
        torch.testing.assert_close(attention(value.q, value.k, value.v, value.indices), _call(value), rtol=0, atol=0)
        for scale in (float("nan"), 0, -1, True, torch.tensor(0.0625, device=device)):
            with pytest.raises(ValueError, match="scale"):
                attention(value.q, value.k, value.v, value.indices, softmax_scale=scale)
        with pytest.raises(ValueError, match="Host"):
            attention(value.q, value.k, value.v, value.indices, query_lens=(1,))
        with pytest.raises(ValueError, match="16 Q heads"):
            attention(torch.empty((queries[0], 17, 256), device=device, dtype=value.q.dtype), value.k, value.v, value.indices)
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


@pytest.mark.parametrize("path", _real_files(), ids=lambda p: p.stem)
@pytest.mark.parametrize("tp_size", TP_SIZES, ids=lambda tp: f"tp{tp}")
def test_real_attention(path, tp_size):
    check(_tp_case(_load(path, _gpu()), tp_size))


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


@pytest.mark.parametrize("tp_size", TP_SIZES, ids=lambda tp: f"tp{tp}")
def test_direct_pack_limit(tp_size, monkeypatch):
    """The real byte cutoff retains raw routing, output guards and graph updates."""
    from pyhip.ops.qsa.flydsl import attention_direct_packed

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
                        patch.setattr(attention_direct_packed, "run", forbidden)
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
                        patch.setattr(attention_direct_packed, "run", forbidden)
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


def test_h6_balanced_graph():
    """Balanced H6 tiles rebuild exact masks after changing selection and values."""
    device = _gpu()
    value = _make_case((5460,), (0,), 6, device, seed=93)
    check(value)
    key = (device, torch.cuda.current_stream(device).cuda_stream, value.query_lens,
           value.prefix_lens, 6, 1, value.scale)
    plan = _runtime._workspaces[key].union
    assert plan.query_tile == 21
    sizes = plan.metadata[:, 1].cpu().tolist()
    assert min(sizes) >= 20 and max(sizes) == 21
    independent = value.indices.clone()
    shared = _make_case((5460,), (0,), 6, device, seed=97, shared=True).indices
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):
        output = _call(value)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            _call(value, output)
    torch.cuda.current_stream(device).wait_stream(stream)
    for selection in (shared, independent, shared):
        value.indices.copy_(selection)
        value.v.neg_()
        output.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize(device)
        ids = _rows(value)
        torch.testing.assert_close(output[ids].float(), reference(value, ids), rtol=.02, atol=.02)
        with torch.cuda.stream(stream):
            _audit(value)


def test_sglang_adapter(monkeypatch, tmp_path):
    """One normal backend flow covers cache/padding, profile capture and lazy exclusion."""
    pytest.importorskip("sglang")
    from experiments.attention.flydsl.qsa.sglang import plugin
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


def test_validation_failure_capture(monkeypatch, tmp_path):
    """A failed service comparison saves the exact replay and still raises."""
    from experiments.attention.flydsl.qsa.sglang import attention_validation as validation, plugin
    from unittest.mock import Mock

    monkeypatch.setenv("PYHIP_QSA_VALIDATE", "1")
    monkeypatch.setenv("PYHIP_QSA_DUMP_DIR", str(tmp_path))
    state = plugin._State()
    q = torch.zeros((2, 12, 256), dtype=torch.bfloat16)
    k = torch.zeros((6, 1, 256), dtype=torch.bfloat16)
    v = torch.ones_like(k)
    indices = torch.full((2, 2051), -1, dtype=torch.int32)
    output, expected = torch.zeros_like(q), torch.zeros_like(q)
    output[1, 5, 220] = 1
    state.runtime = SimpleNamespace(attention=Mock(return_value=output))
    backend = SimpleNamespace(runner=SimpleNamespace(ps=SimpleNamespace(tp_rank=0)))
    context = (backend, SimpleNamespace(layer_id=47), (1, 1), (3, 1))
    original = Mock(return_value=expected)
    monkeypatch.setattr(validation, "reference", Mock(return_value=expected.float()))
    with pytest.raises(AssertionError, match="Tensor-likes are not close"):
        state.execute(context, q, k, v, indices, .0625, original, ())
    assert not state.checked and not state.checks
    original.assert_called_once()
    path, = tmp_path.glob("failure_tp0_layer47_m2_*.pt")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    assert saved["metadata"]["validation_failed"]
    assert saved["metadata"]["query_lens"] == (1, 1)
    assert saved["metadata"]["prefix_lens"] == (3, 1)
    for name, value in (("q", q), ("k", k), ("v", v), ("indices", indices),
                        ("output", output), ("expected", expected.float()), ("legacy_expected", expected)):
        torch.testing.assert_close(saved["tensors"][name], value, rtol=0, atol=0)
        assert _hash(value) == saved["metadata"]["tensor_metadata"][name]["sha256"]


@pytest.mark.parametrize("heads", (12, 6, 3))
def test_t13_q_prescaling_reference(heads, monkeypatch, tmp_path):
    """BF16 prescaling destroys a zero dot product; the service must check the true math."""
    from experiments.attention.flydsl.qsa.sglang import attention_validation as validation, plugin

    device = _gpu()
    q = torch.tensor([1., 1.5], device=device, dtype=torch.bfloat16).repeat(128).expand(1, heads, 256).contiguous()
    k = torch.zeros((4, 1, 256), device=device, dtype=torch.bfloat16)
    k[::2] = torch.tensor([1.5, -1.], device=device, dtype=torch.bfloat16).repeat(128)
    v = torch.full_like(k, -4.)
    v[::2] = 4.
    indices = torch.full((1, 2051), -1, dtype=torch.int32, device=device)
    indices[0, :4] = torch.arange(4, device=device)
    value = _metadata(q, k, v, indices, (1,), (3,))
    expected64 = (q.double()[0] @ k[:, 0].double().T * value.scale).softmax(-1) @ v[:, 0].double()
    torch.testing.assert_close(expected64, torch.zeros_like(expected64), rtol=0, atol=0)
    legacy = _base(value)
    with pytest.raises(AssertionError):
        torch.testing.assert_close(legacy.double()[0], expected64, rtol=.02, atol=.02)
    precise = validation.reference(q, k, v, indices, query_lens=(1,), prefix_lens=(3,), scale=value.scale)
    torch.testing.assert_close(precise.double()[0], expected64, rtol=0, atol=1e-6)
    monkeypatch.setenv("PYHIP_QSA_VALIDATE", "1")
    monkeypatch.setenv("PYHIP_QSA_REPORT_DIR", str(tmp_path))
    backend = SimpleNamespace(runner=SimpleNamespace(ps=SimpleNamespace(tp_rank=0)))
    state = plugin._State()
    output = state.execute((backend, SimpleNamespace(layer_id=47), (1,), (3,)), q, k, v, indices,
                           value.scale, lambda: legacy, ())
    torch.testing.assert_close(output.double()[0], expected64, rtol=.02, atol=.02)
    assert state.checks[0]["reference"] == "fp32_selected_tokens" and not state.checks[0]["legacy_close"]
    assert state.checks[0]["elements"] == heads * 256


def test_t13_legacy_agreement_cannot_hide_bad_output(monkeypatch, tmp_path):
    """Even agreement with the old baseline cannot bypass the mathematical reference."""
    from experiments.attention.flydsl.qsa.sglang import attention_validation as validation, plugin
    from unittest.mock import Mock

    monkeypatch.setenv("PYHIP_QSA_VALIDATE", "1")
    monkeypatch.setenv("PYHIP_QSA_DUMP_DIR", str(tmp_path))
    q = torch.zeros((1, 3, 256), dtype=torch.bfloat16)
    k = v = torch.zeros((4, 1, 256), dtype=torch.bfloat16)
    indices = torch.full((1, 2051), -1, dtype=torch.int32)
    wrong = torch.ones_like(q)
    state = plugin._State()
    state.runtime = SimpleNamespace(attention=Mock(return_value=wrong))
    monkeypatch.setattr(validation, "reference", Mock(return_value=torch.zeros_like(q, dtype=torch.float32)))
    backend = SimpleNamespace(runner=SimpleNamespace(ps=SimpleNamespace(tp_rank=0)))
    with pytest.raises(AssertionError):
        state.execute((backend, SimpleNamespace(layer_id=47), (1,), (3,)), q, k, v, indices,
                      .0625, lambda: wrong, ())
    assert not state.checked and not state.checks


@pytest.mark.parametrize("heads,hk", ((12, 1), (6, 1), (3, 1), (12, 2), (6, 2), (3, 2)))
def test_validation_packed_requests(heads, hk):
    """The independent oracle covers empty requests, sparse sets, prefixes and physical tails."""
    from experiments.attention.flydsl.qsa.sglang.attention_validation import reference as service_reference

    value = _make_case((7, 0, 5, 9), (0, 17, 2047, 3000), heads * hk, _gpu())
    if hk == 2:
        value.k = torch.cat((value.k, -value.k), dim=1)
        value.v = torch.cat((value.v, -value.v), dim=1)
    expected = service_reference(value.q, value.k, value.v, value.indices, query_lens=value.query_lens,
                                 prefix_lens=value.prefix_lens, scale=value.scale)
    torch.testing.assert_close(expected, reference(value, list(range(len(value.q)))), rtol=2e-4, atol=2e-5)
    torch.testing.assert_close(_call(value).float(), expected, rtol=.02, atol=.02)
    for row in (0, 6, 7, 11, 12, 20):
        selected = value.indices[row].cpu().long()
        selected = selected[selected >= 0]
        selected += int(value.cu_k[value.sequence_ids[row]])
        q = value.q[row].cpu().double().reshape(hk, heads, 256)
        k = value.k[selected.to(value.k.device)].cpu().double()
        v = value.v[selected.to(value.v.device)].cpu().double()
        scores = torch.einsum("ghd,ngd->ghn", q, k) * value.scale
        expected64 = torch.einsum("ghn,ngd->ghd", scores.softmax(-1), v).reshape(heads * hk, 256)
        torch.testing.assert_close(expected[row].cpu().double(), expected64, rtol=2e-4, atol=2e-5)


@pytest.mark.parametrize("tp_size", TP_SIZES)
def test_t13_captured_mixed_prefill(tp_size, monkeypatch, tmp_path):
    """Captured service failure: unchanged runtime must pass an accurate full-output oracle."""
    from experiments.attention.flydsl.qsa.sglang import plugin

    paths = sorted((DATA / "qsa_t13_20260929_01/reproduce/failure_inputs").glob("failure_tp0_layer47_m16352_*.pt"))
    if not paths:
        pytest.skip("requires the QSA-T13 captured mixed-prefill input")
    assert len(paths) == 1
    value = _tp_case(_load(paths[0], _gpu()), tp_size)
    monkeypatch.setenv("PYHIP_QSA_VALIDATE", "1")
    monkeypatch.setenv("PYHIP_QSA_REPORT_DIR", str(tmp_path))
    backend = SimpleNamespace(runner=SimpleNamespace(ps=SimpleNamespace(tp_rank=0)))
    state = plugin._State()
    output = state.execute((backend, SimpleNamespace(layer_id=47), value.query_lens, value.prefix_lens),
                           value.q, value.k, value.v, value.indices, value.scale, lambda: _base(value), ())
    # H6/H3 change union query grouping relative to the original H12 capture;
    # cross-layout accumulation is tolerance-equivalent, not bit-exact.
    tolerance = 0 if tp_size == 2 else .02
    torch.testing.assert_close(output, value.captured, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(output, _call(value), rtol=0, atol=0)
    assert state.checks[0]["reference"] == "fp32_selected_tokens"
    if tp_size in (2, 4):
        assert not state.checks[0]["legacy_close"]
    assert _audit(value)["dense_rows"] == 4038


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
    sources = source_files()
    hashes = lambda: {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    report = {"complete": False, "raw": [], "buffers": buffers, "warmup": warmup, "samples": samples,
              "scope": "attention: recover+validate+rebuild+dispatch; base: frozen sparse kernel; preallocated outputs, no JIT/indexer/KV gather",
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
        expected_attention = outputs[0].clone()
        for output, base_out in zip(outputs, bases):
            output.fill_(float("nan")); base_out.fill_(float("nan"))
        torch.cuda.synchronize(gpu)
        _gate(folder, "before_samples", gpu)
        timer = cudaPerf(name="attention", verbose=0)
        if not timer.enable:
            raise RuntimeError("CUDAPERF disabled timing")
        for sample in range(samples):
            bi = sample % buffers
            for name in (("base", "attention") if sample % 2 == 0 else ("attention", "base")):
                with timer:
                    (_base if name == "base" else _call)(values[bi], bases[bi] if name == "base" else outputs[bi])
                elapsed = timer.latencies[-1] * 1e6
                report["raw"].append({"scope": name, "sample": sample, "buffer": bi, "us": elapsed})
                assert math.isfinite(elapsed) and elapsed > 0
        for data, output, expected in zip(values, outputs, bases):
            torch.testing.assert_close(output, expected_attention, rtol=0, atol=0)
            torch.testing.assert_close(output, expected, rtol=0.02, atol=0.02)
            for name in ("q", "k", "v", "indices"):
                torch.testing.assert_close(getattr(data, name), getattr(value, name), rtol=0, atol=0)
        work = int((value.indices >= 0).sum()) * 4 * value.q.shape[1] * 256
        report["summary"] = {}
        base_us = statistics.median(r["us"] for r in report["raw"] if r["scope"] == "base")
        for name in ("base", "attention"):
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
def test_attention_performance(path, tp_size):
    device = _gpu()
    output = Path(os.environ["QSA_REPLAY_OUTPUT"])
    assert output.resolve().is_relative_to(DATA.resolve())
    result = benchmark(_tp_case(_load(path, device), tp_size), output / f"{path.stem}_tp{tp_size}", device.index)
    assert result["complete"]
    if os.environ.get("QSA_REPLAY_REQUIRE_HALF") == "1":
        assert result["summary"]["attention"]["ratio_to_base"] <= 0.5


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--inputs", nargs="+", type=Path)
    parser.add_argument("--synthetic", nargs="*", type=int, metavar="ROWS",
                        help="synthetic full-prefill rows (default: 64 2051 12000 32768), at each local TP size")
    parser.add_argument("--tp-sizes", nargs="+", type=int, choices=TP_SIZES, default=TP_SIZES,
                        help="local-head replay sizes; TP4/8 derive H6/H3 from each TP2 capture")
    parser.add_argument("--buffers", type=int, default=BENCHMARK_BUFFERS,
                        help="independent input/output buffers (default: 10)")
    parser.add_argument("--samples", type=int, default=BENCHMARK_SAMPLES,
                        help="samples per implementation (default: 128)")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--require-half", action="store_true", help="require full attention <= half the frozen baseline")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.buffers < 1 or args.samples < args.buffers:
        parser.error("Require samples >= buffers >= 1")
    synthetic_rows = (args.synthetic or (64, 2051, 12000, 32768)) if args.synthetic is not None else ()
    if any(rows < 1 for rows in synthetic_rows):
        parser.error("Synthetic row counts must be positive")
    if len(set(synthetic_rows)) != len(synthetic_rows) or len(set(args.tp_sizes)) != len(args.tp_sizes):
        parser.error("Synthetic row counts and TP sizes must be unique")
    paths = args.inputs if args.inputs is not None else ([] if args.synthetic is not None else _real_files())
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Captured QSA input does not exist: {path}")
    labels = [p.stem for p in paths] + [f"synthetic_m{rows}" for rows in synthetic_rows]
    if len(set(labels)) != len(labels):
        parser.error("Input stems and synthetic case labels must be unique")
    if not paths and not synthetic_rows:
        parser.error("No captured inputs; pass --inputs, QSA_REAL_INPUT_DIR or --synthetic")
    if any(os.environ.get(n) for n in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "HSA_CU_MASK", "ROC_GLOBAL_CU_MASK")):
        raise RuntimeError("Use unmasked physical GPU indices")
    assert args.output.resolve().is_relative_to(DATA.resolve())
    args.output.mkdir(parents=True, exist_ok=False)
    torch.cuda.set_device(args.gpu)
    reports = {f"{label}_tp{tp}": {"complete": False, "error": "not run"}
               for label in labels for tp in args.tp_sizes}

    def run_case(value, label, tp_size, *, synthetic=False):
        folder = args.output / label
        result = {"complete": False, "check_only": args.check_only, "synthetic": synthetic,
                  "local_tp_size": tp_size, "distributed_tp_run": False,
                  "query_lens": value.query_lens, "prefix_lens": value.prefix_lens, "scale": value.scale,
                  "q_shape": list(value.q.shape), "k_shape": list(value.k.shape), "dtype": str(value.q.dtype),
                  "capture": getattr(value, "capture", None), "gpu": args.gpu}
        reports[label] = result
        if args.check_only:
            folder.mkdir(parents=True, exist_ok=False)
        try:
            result["routes"] = check(value)
            if args.check_only:
                result["complete"] = True
            else:
                result.update(benchmark(value, folder, args.gpu, buffers=args.buffers, samples=args.samples))
            if args.require_half and not args.check_only:
                assert result["summary"]["attention"]["ratio_to_base"] <= 0.5
            print(label, result.get("summary", result["routes"]), flush=True)
        except BaseException as error:
            if not args.check_only and (folder / "result.json").is_file():
                result.update(json.loads((folder / "result.json").read_text()))
            result.update(complete=False, error=f"{type(error).__name__}: {error}")
            raise
        finally:
            if folder.is_dir():
                (folder / "result.json").write_text(json.dumps(result, indent=2))

    expected = (len(paths) + len(synthetic_rows)) * len(args.tp_sizes)
    try:
        for path in paths:
            original = _load(path, f"cuda:{args.gpu}")
            for tp_size in args.tp_sizes:
                value = _tp_case(original, tp_size)
                run_case(value, f"{path.stem}_tp{tp_size}", tp_size)
                del value
            del original
            gc.collect()
        for rows in synthetic_rows:
            for tp_size in args.tp_sizes:
                value = _make_case((rows,), (0,), 24 // tp_size, f"cuda:{args.gpu}")
                run_case(value, f"synthetic_m{rows}_tp{tp_size}", tp_size, synthetic=True)
                del value
            gc.collect()
    finally:
        passed = sum(result["complete"] for result in reports.values())
        (args.output / "checks.json").write_text(json.dumps({"inputs": [str(p) for p in paths], "tp_sizes": args.tp_sizes,
                                                            "synthetic_rows": synthetic_rows, "expected": expected,
                                                            "passed": passed, "complete": passed == expected,
                                                            "check_only": args.check_only,
                                                            "buffers": args.buffers, "samples": args.samples}, indent=2))
        write_summary(args.output, reports)


if __name__ == "__main__":
    main()
