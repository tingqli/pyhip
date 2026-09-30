"""Prefill QSA indexer: SGLang-eager replay of real captures, synthetic layouts and timing.

Pytest runs correctness only (-m perf times). The CLI replays captured inputs or
--synthetic prefill, or runs a filtered --decode/--decode-forward matrix. All modes
support --check-only without timing or hardware gates. Base
is SGLang's own ``QSAIndexer.forward_cuda`` on this ROCm stack (Torch MQA fallback,
radix fast_topk, Triton expand), PyHIP is the plugin adapter plus ``indexer.py``,
either with SGLang's index_qk_proj GEMM (bit-exact prep; ``check``) or with the
plugin's default hipBLASLt projection (rounding-level changes; ``check_projection``).
The indexer weights and heads are replicated, so per-rank work is TP2/4/8-invariant.
"""

import argparse
import contextlib
import hashlib
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
    __package__ = "benchmarks.qsa"

from tests.ops.qsa._indexer import (  # noqa: E402
    BENCHMARK_BUFFERS,
    BENCHMARK_SAMPLES,
    CASES,
    DATA,
    DECODE_BENCH,
    DECODE_CASES,
    DECODE_FORWARD_BENCH,
    DECODE_FORWARD_CASES,
    RATIO,
    ROOT,
    _batch,
    _buffer,
    _decode_args,
    _decode_batch,
    _decode_metadata,
    _gpu,
    _metadata,
    _pool,
    _production_rope,
    _real_files,
    base,
    check,
    check_decode,
    check_decode_forward,
    check_projection,
    decode_base,
    decode_case,
    decode_forward_case,
    decode_pyhip,
    decode_step,
    fast_projection,
    load,
    pyhip,
    synthetic,
)
from pyhip.ops.qsa.flydsl import indexer  # noqa: E402
from experiments.attention.flydsl.qsa.sglang import plugin  # noqa: E402
from tests.ops.qsa._benchmark import gate as _gate  # noqa: E402
from tests.ops.qsa._benchmark import recording_matrix, source_files  # noqa: E402


@pytest.mark.parametrize("seq_lens,extend_lens", CASES,
                         ids=[f"s{'-'.join(map(str, s))}_e{'-'.join(map(str, e))}" for s, e in CASES])
def test_synthetic(seq_lens, extend_lens):
    print(check(synthetic(seq_lens, extend_lens, _gpu())))


@pytest.mark.parametrize("path", _real_files(), ids=lambda p: p.stem)
def test_real_capture(path):
    print(check(load(path, _gpu())))


@pytest.mark.parametrize("path", _real_files(), ids=lambda p: p.stem)
def test_real_capture_hipblaslt_projection(path):
    print(check_projection(load(path, _gpu())))


def test_ineligible_calls_keep_sglang():
    device = _gpu()
    case = synthetic((300,), (300,), device)
    batch = _batch(case)
    metadata = _metadata(case, _pool(case.state, case.layer))
    assert plugin._indexer_inputs(case.module, case.hidden, case.positions, batch, metadata) is not None
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    for change in (dict(forward_mode=ForwardMode.DECODE), dict(seq_lens_cpu=None),
                   dict(seq_lens_cpu=torch.tensor([70000]))):
        assert plugin._indexer_inputs(case.module, case.hidden, case.positions,
                                      SimpleNamespace(**{**vars(batch), **change}), metadata) is None
    fp32 = case.hidden.float()
    assert plugin._indexer_inputs(case.module, fp32, case.positions, batch, metadata) is None
    # 65536 compressed keys (262144 tokens, the model maximum) is the largest eligible request.
    for length, eligible in ((262144, True), (262148, False)):
        case = synthetic((length,), (4,), device)
        inputs = plugin._indexer_inputs(case.module, case.hidden, case.positions, _batch(case),
                                        _metadata(case, _pool(case.state, case.layer)))
        assert (inputs is not None) == eligible, length


def test_plugin_validation_then_fast_projection(monkeypatch):
    """PYHIP_QSA_VALIDATE checks a layout once with SGLang's GEMM; repeats use the hipBLASLt projection."""
    device = _gpu()
    case = synthetic((33000,), (16616,), device)
    monkeypatch.setenv("PYHIP_QSA_INDEXER", "1")
    monkeypatch.setenv("PYHIP_QSA_VALIDATE", "1")
    state = plugin._State()
    original = type(case.module).forward_cuda
    with _production_rope(), torch.no_grad():
        for _ in range(2):
            state.indexer(original, case.module, case.hidden, case.positions, _batch(case),
                          _metadata(case, _pool(case.state, case.layer)))
    torch.cuda.synchronize()
    (report,) = state.indexer_checks
    print(report)
    assert report["key_state_equal"] and report["rope_state_equal"] and report["compressed_equal"]
    assert report["worst_relative_boundary_violation"] <= 1e-5 and state.indexer_gemm is True


@pytest.mark.parametrize("heads", (4, 8))
@pytest.mark.parametrize("lengths,pages", DECODE_CASES,
                         ids=[f"n{'-'.join(map(str, n[:4]))}{'x%d' % len(n) if len(n) > 4 else ''}_p{p}"
                              for n, p in DECODE_CASES])
def test_decode(lengths, pages, heads):
    print(check_decode(decode_case(lengths, pages, _gpu(), heads=heads)))


def test_decode_graph_replay():
    """One capture serves every later length/page table written into its static buffers."""
    device = _gpu()
    case = decode_case((3000, 0, 700, 16384), 1024, device, pool=2100)
    state = plugin._State()
    decode_pyhip(case, state)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = decode_pyhip(case, state)
    for lengths in ((3000, 0, 700, 16384), (1, 512, 513, 9999), (16384, 16383, 0, 4)):
        fresh = decode_case(lengths, 1024, device, seed=sum(lengths), pool=2100)
        for name in ("q", "cache", "table", "lengths", "positions", "sequences"):
            getattr(case, name).copy_(getattr(fresh, name))
        graph.replay()
        # Exact-zero ReLU ties (and near-zero sign flips) may change which tied block fast_topk keeps.
        print(check_decode(case, output.clone()))


def test_decode_ineligible_calls_keep_sglang():
    device = _gpu()
    case = decode_case((600,), 64, device)
    args = _decode_args(case)[:6]
    assert plugin._decode_eligible(*args)
    padded = torch.zeros((1, 8, 128), dtype=torch.bfloat16, device=device)
    changes = [dict(q=case.q.float()), dict(q=case.q[:, :, :64]), dict(q=case.q.repeat(1, 3, 1)[:, :6]),
               dict(q=padded[:, ::2]), dict(cache=case.cache.view(-1, 8, 1, 128)), dict(table=case.table.long()),
               dict(lengths=case.lengths.long()), dict(width=case.width - 16),
               dict(module=SimpleNamespace(**{**vars(case.module), "block_topk": 2048}))]
    names = ("module", "q", "cache", "table", "lengths", "width")
    for change in changes:
        values = dict(zip(names, args), **change)
        assert not plugin._decode_eligible(*(values[n] for n in names)), change.keys()


@pytest.mark.parametrize("lengths,padding", DECODE_FORWARD_CASES,
                         ids=[f"n{'-'.join(map(str, n[:4]))}{'x%d' % len(n) if len(n) > 4 else ''}_pad{p}"
                              for n, p in DECODE_FORWARD_CASES])
def test_decode_forward(lengths, padding):
    device = _gpu()
    case = decode_forward_case(lengths, device, padding=padding, context=max(4096, -(-max(lengths) // 64) * 64 + 64))
    base_pool, new_pool = _pool(case.state, case.layer), _pool(case.state, case.layer)
    batch = _decode_batch(case)
    with _production_rope(), torch.no_grad():
        expected = case.module.forward_cuda(case.hidden, case.positions, batch, _decode_metadata(case, base_pool))
        inputs = plugin._decode_forward_inputs(case.module, case.hidden, case.positions, batch,
                                               _decode_metadata(case, new_pool))
        assert inputs is not None, "adapter rejected an eligible graph decode call"
        actual, q, _ = indexer._decode_forward(inputs.pop("qk"), **inputs)
    print(check_decode_forward(case, base_pool, new_pool, actual, q, expected))


@pytest.mark.parametrize("path", _real_files()[:2], ids=lambda p: p.stem)
def test_decode_forward_real_weights(path):
    device = _gpu()
    case = decode_forward_case((12000, 11888, 4, 7), device, padding=1, capture=path, context=16384)
    base_pool, new_pool = _pool(case.state, case.layer), _pool(case.state, case.layer)
    batch = _decode_batch(case)
    with _production_rope(), torch.no_grad():
        expected = case.module.forward_cuda(case.hidden, case.positions, batch, _decode_metadata(case, base_pool))
        inputs = plugin._decode_forward_inputs(case.module, case.hidden, case.positions, batch,
                                               _decode_metadata(case, new_pool))
        actual, q, _ = indexer._decode_forward(inputs.pop("qk"), **inputs)
    print(check_decode_forward(case, base_pool, new_pool, actual, q, expected))


def test_decode_forward_graph_steps(monkeypatch):
    """Capture the hooked decode forward once, then replay 6 steps (crossing group boundaries) against
    SGLang's eager graph-metadata path on an identical pool history."""
    from sglang.srt.layers.attention.qsa.qsa_indexer import QSAIndexer

    device = _gpu()
    monkeypatch.setenv("PYHIP_QSA_INDEXER_DECODE", "1")
    lengths = [4001, 6, 1022, 2]
    case = decode_forward_case(tuple(lengths), device, padding=2, context=8192)
    base_pool, new_pool, warm_pool = (_pool(case.state, case.layer) for _ in range(3))
    state, batch, original = plugin._State(), _decode_batch(case), QSAIndexer.forward_cuda
    with _production_rope(), torch.no_grad():
        state.indexer(original, case.module, case.hidden, case.positions, batch, _decode_metadata(case, warm_pool))
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = state.indexer(original, case.module, case.hidden, case.positions, batch,
                                   _decode_metadata(case, new_pool))
        for step in range(6):
            if step:
                lengths = [n + 1 for n in lengths]
                decode_step(case, lengths)
            graph.replay()
            actual = output.clone()
            expected = original(case.module, case.hidden, case.positions, batch, _decode_metadata(case, base_pool))
            inputs = plugin._decode_forward_inputs(case.module, case.hidden, case.positions, batch,
                                                   _decode_metadata(case, warm_pool))
            q = indexer._decode_forward(inputs.pop("qk"), **inputs)[1]
            report = check_decode_forward(case, base_pool, new_pool, actual, q, expected)
            print(step, lengths, report)
            case.state = {name: _buffer(base_pool, name).clone() for name in ("key_state", "rope_state", "compressed")}


def test_decode_validation_graph(monkeypatch):
    """PYHIP_QSA_VALIDATE=1 captures SGLang's decode forward and the PyHIP one together; device counters
    cover every replay (6 steps crossing group boundaries) and the pool history still matches SGLang."""
    from sglang.srt.layers.attention.qsa.qsa_indexer import QSAIndexer
    from sglang.srt.model_executor.runner_utils.capture_mode import model_capture_mode

    device = _gpu()
    monkeypatch.setenv("PYHIP_QSA_INDEXER_DECODE", "1")
    monkeypatch.setenv("PYHIP_QSA_VALIDATE", "1")
    lengths = [4001, 6, 1022, 2]
    case = decode_forward_case(tuple(lengths), device, padding=2, context=8192)
    base_pool, new_pool, warm_pool = (_pool(case.state, case.layer) for _ in range(3))
    state, batch, original = plugin._State(), _decode_batch(case), QSAIndexer.forward_cuda
    with _production_rope(), torch.no_grad():
        state.indexer(original, case.module, case.hidden, case.positions, batch, _decode_metadata(case, warm_pool))
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with model_capture_mode(), torch.cuda.graph(graph):
            output = state.indexer(original, case.module, case.hidden, case.positions, batch,
                                   _decode_metadata(case, new_pool))
        for step in range(6):
            if step:
                lengths = [n + 1 for n in lengths]
                decode_step(case, lengths)
            graph.replay()
            actual = output.clone()
            expected = original(case.module, case.hidden, case.positions, batch, _decode_metadata(case, base_pool))
            inputs = plugin._decode_forward_inputs(case.module, case.hidden, case.positions, batch,
                                                   _decode_metadata(case, warm_pool))
            q = indexer._decode_forward(inputs.pop("qk"), **inputs)[1]
            print(step, lengths, check_decode_forward(case, base_pool, new_pool, actual, q, expected))
            case.state = {name: _buffer(base_pool, name).clone() for name in ("key_state", "rope_state", "compressed")}
    counts = state.decode_summary()["forward:3"]
    print(counts)
    # 1 eager warmup + 6 replays, 4 real rows each; boundaries at lengths 8/1024/4 (step 2) and 4004 (step 3).
    assert (counts["calls"], counts["rows"], counts["compressed_rows"]) == (7, 28, 4)
    assert not any(counts[name] for name in plugin._DECODE_FAILURES)
    state.check_decode()


@pytest.mark.parametrize("fault", ("q", "compressed", "selection"))
def test_decode_validation_detects_faults(monkeypatch, fault):
    """Each class of PyHIP decode error is counted on device and fails the next prefill's host check."""
    from sglang.srt.layers.attention.qsa.qsa_indexer import QSAIndexer

    device = _gpu()
    monkeypatch.setenv("PYHIP_QSA_INDEXER_DECODE", "1")
    monkeypatch.setenv("PYHIP_QSA_VALIDATE", "1")
    case = decode_forward_case((4003, 1024, 2051), device, padding=1, context=8192)
    decode_forward = indexer._decode_forward

    def faulty(qk, **inputs):
        tokens, q, logits = decode_forward(qk, **inputs)
        if fault == "q":
            q = q.clone()
            q[0, 0, 0] += 1
        elif fault == "compressed":
            locs = inputs["write_locs"].long()
            inputs["compressed"][locs[locs != 0]] += 1
        else:
            # Swap row 0's first chosen block for its lowest-scoring unchosen block.
            chosen = set((tokens[0, :2048:4] // 4).tolist()) - {-1}
            low = min((b for b in range(int(inputs["lengths"][0])) if b not in chosen), key=lambda b: float(logits[0, b]))
            tokens = tokens.clone()
            tokens[0, :4] = torch.arange(4 * low, 4 * low + 4, dtype=tokens.dtype, device=tokens.device)
        return tokens, q, logits

    monkeypatch.setattr(indexer, "_decode_forward", faulty)
    state = plugin._State()
    with _production_rope(), torch.no_grad():
        state.indexer(QSAIndexer.forward_cuda, case.module, case.hidden, case.positions, _decode_batch(case),
                      _decode_metadata(case, _pool(case.state, case.layer)))
    counts = state.decode_summary()["forward:3"]
    print(fault, counts)
    field = dict(q="q_mismatch", compressed="compressed_mismatch", selection="selection_violations")[fault]
    assert counts[field] == 1 and counts["rows"] == 3 and counts["compressed_rows"] == 1
    assert not any(counts[name] for name in plugin._DECODE_FAILURES if name != field)
    prefill = synthetic((2051,), (2051,), device)
    with pytest.raises(AssertionError, match="decode differs from SGLang"), _production_rope(), torch.no_grad():
        state.indexer(type(prefill.module).forward_cuda, prefill.module, prefill.hidden, prefill.positions,
                      _batch(prefill), _metadata(prefill, _pool(prefill.state, prefill.layer)))


def test_decode_select_validation(monkeypatch):
    """The select-only hook validates against SGLang's selection too; the forward's reference call bypasses it."""
    device = _gpu()
    monkeypatch.setenv("PYHIP_QSA_VALIDATE", "1")
    case = decode_case((3000, 513, 17, 0), 256, device)
    state = plugin._State()
    print(check_decode(case, decode_pyhip(case, state)))
    counts = state.decode_summary()["select:3"]
    assert counts["calls"] == 1 and counts["rows"] == 4 and not any(counts[n] for n in plugin._DECODE_FAILURES)
    # Inside the validated forward's reference call the hook hands SGLang's result back unvalidated. (SGLang's own
    # selection is not repeatable on near-ties, so its set is not compared here.)
    state.decode_reference = True
    assert state.decode(lambda *args: "sglang", *_decode_args(case)) == "sglang"
    assert state.decode_summary()["select:3"]["calls"] == 1
    state.decode_reference = False
    decode_select = indexer._decode_select

    def reversed_ranking(*args):
        tokens, logits = decode_select(*args)
        return tokens, -logits

    monkeypatch.setattr(indexer, "_decode_select", reversed_ranking)
    decode_pyhip(case, state)
    counts = state.decode_summary()["select:3"]
    assert counts["calls"] == 2 and counts["selection_violations"] == 2
    with pytest.raises(AssertionError, match="decode differs from SGLang"):
        state.check_decode()


def benchmark(case, folder, gpu, *, buffers=BENCHMARK_BUFFERS, warmup=2, samples=BENCHMARK_SAMPLES):
    """Time full ``forward_cuda`` (incl. index_qk_proj) for base and both PyHIP projections, AB/BA."""
    from pyhip.testing.misc import cudaPerf

    if buffers < 1 or samples < buffers or warmup < 0:
        raise ValueError("Require samples >= buffers >= 1 and warmup >= 0")
    if not folder.resolve().is_relative_to(DATA.resolve()):
        raise ValueError("Results must stay under mytest/mydata")
    folder.mkdir(parents=True, exist_ok=False)
    sources = source_files()
    hashes = lambda: {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    arms = ("base", "pyhip_exact", "pyhip")
    report = dict(complete=False, raw=[], buffers=buffers, warmup=warmup, samples=samples, capture=case.capture,
                  scope="base: SGLang QSAIndexer.forward_cuda; pyhip_exact: plugin adapter + indexer.py with "
                        "SGLang's index_qk_proj; pyhip: same with the plugin's default hipBLASLt projection; "
                        "fresh pool copy per buffer; outputs allocated by each implementation",
                  seq_lens=case.seq_lens, extend_lens=case.extend_lens, torch=torch.__version__,
                  hip=torch.version.hip, source_sha256=hashes(), gpu=gpu,
                  tp_note="indexer heads/weights are replicated: TP2/4/8 ranks run this identical workload",
                  packages={n: importlib.metadata.version(n) for n in ("triton", "flydsl", "sglang", "amd-aiter")})
    try:
        _gate(folder, "before", gpu)
        report["check"] = check(case)
        report["check_projection"] = check_projection(case)
        runs = dict(base=lambda value, pool: base(value, pool), pyhip_exact=lambda value, pool: pyhip(value, pool),
                    pyhip=lambda value, pool: pyhip(value, pool, projection=fast_projection))
        values = []
        for index in range(buffers):
            copy = SimpleNamespace(**vars(case))
            copy.hidden = case.hidden.clone()
            values.append((copy, {name: _pool(case.state, case.layer) for name in arms}))
        expected = {}
        for value, pools in values:
            for name in arms:
                result = runs[name](value, pools[name])
                if name != "base":
                    expected.setdefault(name, result)
                    assert torch.equal(result, expected[name])
            for _ in range(warmup):
                for name in arms:
                    runs[name](value, pools[name])
        torch.cuda.synchronize(gpu)
        _gate(folder, "before_samples", gpu)
        timer = cudaPerf(name="indexer", verbose=0)
        if not timer.enable:
            raise RuntimeError("CUDAPERF disabled timing")
        for sample in range(samples):
            index = sample % buffers
            value, pools = values[index]
            for name in (arms if sample % 2 == 0 else arms[::-1]):
                with timer:
                    result = runs[name](value, pools[name])
                elapsed = timer.latencies[-1] * 1e6
                report["raw"].append(dict(scope=name, sample=sample, buffer=index, us=elapsed))
                assert math.isfinite(elapsed) and elapsed > 0
                if name != "base":
                    assert torch.equal(result, expected[name])
        report["summary"] = {}
        base_us = statistics.median(r["us"] for r in report["raw"] if r["scope"] == "base")
        for name in arms:
            us = [r["us"] for r in report["raw"] if r["scope"] == name]
            paired = [next(r["us"] for r in report["raw"] if r["scope"] == name and r["sample"] == i)
                      / next(r["us"] for r in report["raw"] if r["scope"] == "base" and r["sample"] == i)
                      for i in range(samples)]
            report["summary"][name] = dict(median_us=statistics.median(us), mean_us=statistics.fmean(us),
                                           min_us=min(us), max_us=max(us),
                                           ratio_to_base=statistics.median(us) / base_us,
                                           paired_ratio_median=statistics.median(paired))
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
        (folder / "result.json").write_text(json.dumps(report, indent=2, default=str))
    return report


@pytest.mark.perf
@pytest.mark.parametrize("path", _real_files(), ids=lambda p: p.stem)
def test_indexer_performance(path):
    device = _gpu()
    output = Path(os.environ["QSA_REPLAY_OUTPUT"])
    assert output.resolve().is_relative_to(DATA.resolve())
    assert benchmark(load(path, device), output / path.stem, device.index)["complete"]


def decode_benchmark(rows, keys, folder, gpu, *, buffers=BENCHMARK_BUFFERS, warmup=2, samples=BENCHMARK_SAMPLES):
    """Time CUDA-graph replays of ``select_decode_tokens``: SGLang vs the PyHIP hook, AB/BA."""
    from pyhip.testing.misc import cudaPerf

    if buffers < 1 or samples < buffers or warmup < 0:
        raise ValueError("Require samples >= buffers >= 1 and warmup >= 0")
    if not folder.resolve().is_relative_to(DATA.resolve()):
        raise ValueError("Results must stay under mytest/mydata")
    folder.mkdir(parents=True, exist_ok=False)
    sources = source_files()
    hashes = lambda: {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    arms = ("base", "pyhip")
    device = torch.device("cuda", gpu)
    report = dict(complete=False, raw=[], buffers=buffers, warmup=warmup, samples=samples, rows=rows, keys=keys,
                  table_pages=4096,
                  scope="one CUDA-graph replay of QSAIndexer.select_decode_tokens per arm: base = SGLang (Torch "
                        "paged MQA fallback over the full table width, fast_topk, Triton expand); pyhip = plugin "
                        "hook (FlyDSL paged logits over each row's length, same fast_topk and expand); one graph "
                        "per arm and buffer, independent random pools/tables/queries per buffer",
                  torch=torch.__version__, hip=torch.version.hip, source_sha256=hashes(), gpu=gpu,
                  tp_note="indexer heads/weights are replicated: TP2/4/8 ranks run this identical workload",
                  packages={n: importlib.metadata.version(n) for n in ("triton", "flydsl", "sglang", "amd-aiter")})
    try:
        _gate(folder, "before", gpu)
        cases = [decode_case((keys,) * rows, 4096, device, seed=1000 * index + rows) for index in range(buffers)]
        report["check"] = check_decode(cases[0])
        runs = dict(base=decode_base, pyhip=decode_pyhip)
        graphs, outputs, expected = {}, {}, {}
        for index, case in enumerate(cases):
            for name in arms:
                for _ in range(1 + warmup):
                    runs[name](case)
                torch.cuda.synchronize(gpu)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    outputs[name, index] = runs[name](case)
                graph.replay()
                graphs[name, index] = graph
            torch.cuda.synchronize(gpu)
            for name in arms:
                expected[name, index] = outputs[name, index].sort(dim=1).values.clone()
            report.setdefault("base_vs_pyhip_different_token_set_rows", []).append(
                int((expected["base", index] != expected["pyhip", index]).any(dim=1).sum()))
        for _ in range(warmup):
            for key in graphs:
                graphs[key].replay()
        torch.cuda.synchronize(gpu)
        _gate(folder, "before_samples", gpu)
        timer = cudaPerf(name="indexer_decode", verbose=0)
        if not timer.enable:
            raise RuntimeError("CUDAPERF disabled timing")
        for sample in range(samples):
            index = sample % buffers
            for name in (arms if sample % 2 == 0 else arms[::-1]):
                with timer:
                    graphs[name, index].replay()
                elapsed = timer.latencies[-1] * 1e6
                report["raw"].append(dict(scope=name, sample=sample, buffer=index, us=elapsed))
                assert math.isfinite(elapsed) and elapsed > 0
                assert torch.equal(outputs[name, index].sort(dim=1).values, expected[name, index])
        report["summary"] = {}
        base_us = statistics.median(r["us"] for r in report["raw"] if r["scope"] == "base")
        for name in arms:
            us = [r["us"] for r in report["raw"] if r["scope"] == name]
            paired = [next(r["us"] for r in report["raw"] if r["scope"] == name and r["sample"] == i)
                      / next(r["us"] for r in report["raw"] if r["scope"] == "base" and r["sample"] == i)
                      for i in range(samples)]
            report["summary"][name] = dict(median_us=statistics.median(us), mean_us=statistics.fmean(us),
                                           min_us=min(us), max_us=max(us),
                                           ratio_to_base=statistics.median(us) / base_us,
                                           paired_ratio_median=statistics.median(paired))
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
        (folder / "result.json").write_text(json.dumps(report, indent=2, default=str))
    return report


def decode_forward_benchmark(rows, length, folder, gpu, *, buffers=BENCHMARK_BUFFERS, warmup=2,
                             samples=BENCHMARK_SAMPLES):
    """Time CUDA-graph replays of one decode ``QSAIndexer.forward_cuda`` (hidden states in, token
    selections out): SGLang, SGLang with the stage-1 selection hook, and the PyHIP decode forward."""
    from unittest import mock

    from pyhip.testing.misc import cudaPerf
    from sglang.srt.layers.attention.qsa.qsa_indexer import QSAIndexer
    from sglang.srt.model_executor.runner_utils.capture_mode import model_capture_mode

    if buffers < 1 or samples < buffers or warmup < 0:
        raise ValueError("Require samples >= buffers >= 1 and warmup >= 0")
    if not folder.resolve().is_relative_to(DATA.resolve()):
        raise ValueError("Results must stay under mytest/mydata")
    folder.mkdir(parents=True, exist_ok=False)
    sources = source_files()
    hashes = lambda: {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    arms = ("base", "select", "forward")
    device = torch.device("cuda", gpu)
    report = dict(complete=False, raw=[], buffers=buffers, warmup=warmup, samples=samples, rows=rows, length=length,
                  table_pages=4096,
                  scope="one CUDA-graph replay of a decode QSAIndexer.forward_cuda per arm (index_qk_proj GEMM, q "
                        "norm/RoPE, pending-ring store, fixed-shape compression, MQA, top-k, expand): base = SGLang "
                        "(unfused prep, Torch MQA fallback over the table width); select = SGLang with the stage-1 "
                        "select_decode_tokens hook; forward = PyHIP decode forward (1 prep kernel + stage-1 "
                        "selection). Graph metadata from SGLang's own refresh; one graph and KV pool per arm/buffer; "
                        "all graphs share one CUDA-graph memory pool like SGLang's runner, so every output is "
                        "checked right after its own replay",
                  torch=torch.__version__, hip=torch.version.hip, source_sha256=hashes(), gpu=gpu,
                  tp_note="indexer heads/weights are replicated: TP2/4/8 ranks run this identical workload",
                  packages={n: importlib.metadata.version(n) for n in ("triton", "flydsl", "sglang", "amd-aiter")})
    original_select = QSAIndexer.select_decode_tokens

    def hooked_select(module, *args):
        return plugin._around_decode(original_select, module, *args)

    try:
        _gate(folder, "before", gpu)
        # Graph replays read these inputs/pools by address: keep every buffer alive until timing ends.
        state, graphs, outputs, expected, checks, keep = plugin._State(), {}, {}, {}, [], []
        graph_pool = torch.cuda.graph_pool_handle()
        for index in range(buffers):
            case = decode_forward_case((length,) * rows, device, seed=1000 * index + rows, context=262144)
            batch, pools = _decode_batch(case), {arm: _pool(case.state, case.layer) for arm in arms}
            case.state, snapshots = None, {}
            runs = dict(
                base=lambda meta: QSAIndexer.forward_cuda(case.module, case.hidden, case.positions, batch, meta),
                select=lambda meta: QSAIndexer.forward_cuda(case.module, case.hidden, case.positions, batch, meta),
                forward=lambda meta: state.indexer(QSAIndexer.forward_cuda, case.module, case.hidden, case.positions,
                                                   batch, meta))
            for arm in arms:
                meta = _decode_metadata(case, pools[arm])
                with (mock.patch.object(QSAIndexer, "select_decode_tokens", hooked_select) if arm == "select"
                      else contextlib.nullcontext()), \
                        mock.patch.dict(os.environ, {"PYHIP_QSA_INDEXER_DECODE": "1" if arm == "forward" else "0"}), \
                        _production_rope(), torch.no_grad():
                    for _ in range(1 + warmup):
                        runs[arm](meta)
                    torch.cuda.synchronize(gpu)
                    graph = torch.cuda.CUDAGraph()
                    with model_capture_mode(), torch.cuda.graph(graph, pool=graph_pool):
                        outputs[arm, index] = runs[arm](meta)
                graph.replay()
                graphs[arm, index] = graph
                snapshots[arm] = outputs[arm, index].clone()
                expected[arm, index] = snapshots[arm].sort(dim=1).values
            torch.cuda.synchronize(gpu)
            with _production_rope(), torch.no_grad():
                q = case.module.project_qk(case.hidden[:case.rows], case.positions[:, :case.rows])[0]
            meta = _decode_metadata(case, pools["forward"])
            cache, table, lengths, width = meta.get_decode_mqa_inputs(case.layer)
            view = SimpleNamespace(q=q, cache=cache, table=table, lengths=lengths, width=width, module=case.module,
                                   positions=meta.decode_logical_positions, sequences=meta.get_seqlens_int32())
            check = dict(buffer=index, select=check_decode(view, snapshots["select"], snapshots["base"]),
                         forward=check_decode(view, snapshots["forward"], snapshots["base"]))
            for name, lo in (("key_state", RATIO), ("rope_state", RATIO), ("compressed", 1)):
                check[f"{name}_equal"] = all(torch.equal(_buffer(pools[arm], name)[lo:], _buffer(pools["base"], name)[lo:])
                                             for arm in arms)
                assert check[f"{name}_equal"], (index, name)
            checks.append(check)
            keep.append((case, batch, pools))
        report["checks"] = checks
        for _ in range(warmup):
            for key in graphs:
                graphs[key].replay()
        torch.cuda.synchronize(gpu)
        # Return this process's cached eager/setup blocks; the VRAM gate targets other tenants.
        torch.cuda.empty_cache()
        _gate(folder, "before_samples", gpu)
        timer = cudaPerf(name="indexer_decode_forward", verbose=0)
        if not timer.enable:
            raise RuntimeError("CUDAPERF disabled timing")
        for sample in range(samples):
            index = sample % buffers
            for arm in (arms if sample % 2 == 0 else arms[::-1]):
                with timer:
                    graphs[arm, index].replay()
                elapsed = timer.latencies[-1] * 1e6
                report["raw"].append(dict(scope=arm, sample=sample, buffer=index, us=elapsed))
                assert math.isfinite(elapsed) and elapsed > 0
                assert torch.equal(outputs[arm, index].sort(dim=1).values, expected[arm, index])
        report["summary"] = {}
        base_us = statistics.median(r["us"] for r in report["raw"] if r["scope"] == "base")
        for arm in arms:
            us = [r["us"] for r in report["raw"] if r["scope"] == arm]
            paired = [next(r["us"] for r in report["raw"] if r["scope"] == arm and r["sample"] == i)
                      / next(r["us"] for r in report["raw"] if r["scope"] == "base" and r["sample"] == i)
                      for i in range(samples)]
            report["summary"][arm] = dict(median_us=statistics.median(us), mean_us=statistics.fmean(us),
                                          min_us=min(us), max_us=max(us),
                                          ratio_to_base=statistics.median(us) / base_us,
                                          paired_ratio_median=statistics.median(paired))
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
        (folder / "result.json").write_text(json.dumps(report, indent=2, default=str))
    return report


def _check_only(folder, run, **metadata):
    """Persist a correctness-only case, including failures, without constructing a timer."""
    if not folder.resolve().is_relative_to(DATA.resolve()):
        raise ValueError("Results must stay under mytest/mydata")
    folder.mkdir(parents=True, exist_ok=False)
    report = dict(complete=False, check_only=True, raw=[], **metadata)
    try:
        report["check"] = run()
        report["complete"] = True
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        (folder / "result.json").write_text(json.dumps(report, indent=2, default=str))
    return report


def _check_decode_forward(rows, length, device):
    """Check all three forward arms once on independent pools, without timing or graph capture."""
    from unittest import mock

    from sglang.srt.layers.attention.qsa.qsa_indexer import QSAIndexer

    case = decode_forward_case((length,) * rows, device, seed=rows, context=262144)
    pools = {arm: _pool(case.state, case.layer) for arm in ("base", "select", "forward")}
    batch, state = _decode_batch(case), plugin._State()
    original_select = QSAIndexer.select_decode_tokens

    def hooked_select(module, *args):
        return state.decode(original_select, module, *args)

    with _production_rope(), torch.no_grad():
        expected = case.module.forward_cuda(case.hidden, case.positions, batch,
                                            _decode_metadata(case, pools["base"])).clone()
        with mock.patch.object(QSAIndexer, "select_decode_tokens", hooked_select):
            selected = case.module.forward_cuda(case.hidden, case.positions, batch,
                                                _decode_metadata(case, pools["select"])).clone()
        inputs = plugin._decode_forward_inputs(case.module, case.hidden, case.positions, batch,
                                               _decode_metadata(case, pools["forward"]))
        assert inputs is not None, "adapter rejected an eligible graph decode call"
        actual, q, _ = indexer._decode_forward(inputs.pop("qk"), **inputs)
    return dict(select=check_decode_forward(case, pools["base"], pools["select"], selected, q, expected),
                forward=check_decode_forward(case, pools["base"], pools["forward"], actual, q, expected))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=0)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--inputs", nargs="+", type=Path)
    inputs.add_argument("--synthetic", action="store_true",
                        help="prefill with synthetic sequence/extend lengths (12000, 12000), without captures")
    parser.add_argument("--buffers", type=int, default=BENCHMARK_BUFFERS)
    parser.add_argument("--samples", type=int, default=BENCHMARK_SAMPLES)
    parser.add_argument("--check-only", action="store_true", help="correctness only; no hardware gates or timing")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--decode", action="store_true", help="formal decode matrix (DECODE_BENCH) instead of prefill")
    modes.add_argument("--decode-forward", action="store_true",
                       help="formal decode forward_cuda matrix (DECODE_FORWARD_BENCH) instead of prefill")
    parser.add_argument("--rows", nargs="+", type=int, choices=sorted({rows for rows, _ in DECODE_BENCH}),
                        help="filter graph rows for either decode matrix; defaults to all rows")
    parser.add_argument("--keys", nargs="+", type=int, choices=sorted({keys for _, keys in DECODE_BENCH}),
                        help="filter compressed key counts for --decode; defaults to all counts")
    parser.add_argument("--lengths", nargs="+", type=int,
                        choices=sorted({length for _, length in DECODE_FORWARD_BENCH}),
                        help="filter sequence lengths in tokens for --decode-forward; defaults to all lengths")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.gpu < 0:
        parser.error("gpu must be nonnegative")
    if args.buffers < 1 or args.samples < args.buffers:
        parser.error("Require samples >= buffers >= 1")
    if args.keys is not None and not args.decode:
        parser.error("--keys requires --decode")
    if args.lengths is not None and not args.decode_forward:
        parser.error("--lengths requires --decode-forward")
    for option in ("rows", "keys", "lengths"):
        values = getattr(args, option)
        if values is not None and len(set(values)) != len(values):
            parser.error(f"--{option} values must be distinct")
    paths, shapes = [], []
    if args.decode or args.decode_forward:
        if args.inputs is not None or args.synthetic:
            parser.error("--inputs and --synthetic are prefill-only")
        matrix = DECODE_BENCH if args.decode else DECODE_FORWARD_BENCH
        sizes = args.keys if args.decode else args.lengths
        shapes = [(rows, size) for rows, size in matrix
                  if (args.rows is None or rows in args.rows) and (sizes is None or size in sizes)]
        if not shapes:
            parser.error("No decode shapes selected")
    else:
        if args.rows is not None:
            parser.error("--rows requires --decode or --decode-forward")
        paths = [] if args.synthetic else args.inputs if args.inputs is not None else _real_files()
        if not args.synthetic and not paths:
            parser.error("No input captures found; provide --inputs or use --synthetic")
        if len({path.stem for path in paths}) != len(paths):
            parser.error("Input captures must have distinct stems for per-case output")
        for path in paths:
            if not path.is_file():
                parser.error(f"Input capture does not exist: {path}")
    if any(os.environ.get(n) for n in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES",
                                        "HSA_CU_MASK", "ROC_GLOBAL_CU_MASK")):
        raise RuntimeError("Use unmasked physical GPU indices")
    if not args.output.resolve().is_relative_to(DATA.resolve()):
        parser.error("Results must stay under mytest/mydata")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.cuda.set_device(args.gpu)
    device = torch.device("cuda", args.gpu)
    results = {}
    if args.decode or args.decode_forward:
        names = [f"decode_r{rows}_k{size}" if args.decode else f"decode_forward_r{rows}_n{size}"
                 for rows, size in shapes]
        with recording_matrix(args.output, names) as reports:
            for name, (rows, size) in zip(names, shapes):
                if args.check_only:
                    if args.decode:
                        run = lambda: check_decode(decode_case((size,) * rows, 4096, device, seed=rows))
                    else:
                        run = lambda: _check_decode_forward(rows, size, device)
                    report = _check_only(args.output / name, run, case=name, gpu=args.gpu, rows=rows,
                                         table_pages=4096, **({"keys": size} if args.decode else {"length": size}))
                else:
                    run = decode_benchmark if args.decode else decode_forward_benchmark
                    report = run(rows, size, args.output / name, args.gpu, buffers=args.buffers, samples=args.samples)
                reports[name] = report
                results[name] = report if args.check_only else report["summary"]
                print(name, results[name], flush=True)
                torch.cuda.empty_cache()
        (args.output / "checks.json").write_text(json.dumps(dict(shapes=shapes, results=results,
                                                                 buffers=args.buffers, samples=args.samples,
                                                                 check_only=args.check_only),
                                                            indent=2))
        return
    names = ["synthetic_s12000_e12000"] if args.synthetic else [path.stem for path in paths]
    with recording_matrix(args.output, names) as reports:
        for name, path in zip(names, [None] if args.synthetic else paths):
            case = synthetic((12000,), (12000,), device) if path is None else load(path, device)
            result = _check_only(args.output / name, lambda: check(case), case=name, gpu=args.gpu,
                                 seq_lens=case.seq_lens, extend_lens=case.extend_lens, capture=case.capture) \
                if args.check_only else benchmark(case, args.output / name, args.gpu,
                                                  buffers=args.buffers, samples=args.samples)
            reports[name] = result
            results[name] = result if args.check_only else result["summary"]
            print(name, results[name], flush=True)
            del case
    (args.output / "checks.json").write_text(json.dumps(dict(inputs=[str(p) for p in paths], results=results,
                                                             buffers=args.buffers, samples=args.samples,
                                                             check_only=args.check_only, synthetic=args.synthetic),
                                                        indent=2, default=str))


if __name__ == "__main__":
    main()
