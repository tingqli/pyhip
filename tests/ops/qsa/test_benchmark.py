"""CPU-only contracts for complete QSA benchmark receipts and failure handling."""

import csv
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from tests.ops.qsa import _benchmark as reports


def test_summary_keeps_unstarted_cases_and_tp(tmp_path):
    good = dict(complete=True, local_tp_size=4, gpu=6, routes=dict(dense_rows=2, union_rows=3, direct_rows=1),
                raw=[dict(scope="attention", sample=0, buffer=0, us=5.0)],
                summary=dict(attention=dict(median_us=5.0)))
    result = reports.write_summary(tmp_path, {"good": good, "pending": dict(complete=False, error="not run")})
    assert not result["complete"] and result["raw_samples"] == 1
    with (tmp_path / "summary.csv").open() as stream:
        row, = csv.DictReader(stream)
    assert row["tp_size"] == "4" and row["direct_rows"] == "1"
    with (tmp_path / "raw.csv").open() as stream:
        assert len(list(csv.DictReader(stream))) == 1


def test_matrix_persists_failure_and_raw(tmp_path):
    with pytest.raises(RuntimeError, match="failure"):
        with reports.recording_matrix(tmp_path, ["finished", "failed", "unstarted"]) as matrix:
            matrix["finished"] = dict(complete=True)
            folder = tmp_path / "failed"
            folder.mkdir()
            (folder / "result.json").write_text(json.dumps(dict(complete=False, error="failure",
                raw=[dict(scope="kernel", sample=0, buffer=0, us=4.0)])))
            raise RuntimeError("failure")
    result = json.loads((tmp_path / "summary.json").read_text())
    assert not result["complete"] and result["raw_samples"] == 1
    assert result["cases"]["failed"]["error"] == "failure"
    assert result["cases"]["unstarted"]["error"] == "not run"


def test_matrix_does_not_hide_setup_failure(tmp_path):
    with pytest.raises(ValueError, match="checksum"):
        with reports.recording_matrix(tmp_path, ["first", "second"]) as matrix:
            matrix["first"] = dict(complete=True)
            raise ValueError("checksum")
    assert not json.loads((tmp_path / "summary.json").read_text())["complete"]


def test_empty_summary_is_not_success(tmp_path):
    assert not reports.write_summary(tmp_path, {})["complete"]
    with pytest.raises(FileExistsError):
        reports.write_summary(tmp_path, {})


def test_component_check_only_never_times_or_gates(tmp_path, monkeypatch):
    import torch
    from pyhip.testing import misc

    monkeypatch.setattr(reports, "DATA", tmp_path)
    monkeypatch.setattr(reports, "source_files", lambda: [])

    def forbidden(*args, **kwargs):
        raise AssertionError("check-only must not time or gate")

    monkeypatch.setattr(reports, "gate", forbidden)
    monkeypatch.setattr(misc, "cudaPerf", forbidden)
    calls = []

    def make_case(index):
        calls.append(("make", index))
        return SimpleNamespace(runs={"kernel": lambda: calls.append("run")},
                               reset={"kernel": lambda: calls.append("reset")},
                               checks={"kernel": lambda: calls.append("check")},
                               tensors={"input": torch.zeros(2)}, metadata={}, flops={})

    result = reports.measure_components(make_case, tmp_path / "component", 6, check_only=True)
    assert result["complete"] and not result["raw"]
    assert calls == [("make", 0), "reset", "run", "check"]


@pytest.mark.parametrize("fail", (False, True))
def test_component_sampling_exports_all_raw_cpu(tmp_path, monkeypatch, fail):
    """A fake clock tests bookkeeping only; these values are never GPU performance evidence."""
    import torch
    from pyhip.testing import misc

    monkeypatch.setattr(reports, "DATA", tmp_path)
    monkeypatch.setattr(reports, "source_files", lambda: [])
    gates, state = [], {"timing": False, "samples": 0}
    monkeypatch.setattr(reports, "gate", lambda folder, phase, gpu: gates.append(phase))

    class Clock:
        enable = True

        def __init__(self, **kwargs):
            self.latencies = []

        def __enter__(self):
            state["timing"] = True
            return self

        def __exit__(self, *args):
            state["timing"] = False
            state["samples"] += 1
            self.latencies.append(1e-6 * state["samples"])

    monkeypatch.setattr(misc, "cudaPerf", Clock)

    def make_case(index):
        def reset():
            assert not state["timing"]

        def check():
            assert not state["timing"]
            if fail and state["samples"] == 3:
                raise ValueError("output check failed")

        return SimpleNamespace(runs={name: lambda: None for name in ("a", "b")},
                               reset={name: reset for name in ("a", "b")},
                               checks={name: check for name in ("a", "b")},
                               tensors={"input": torch.zeros(2)}, metadata={}, flops={"a": 16})

    folder = tmp_path / "sampling"
    if fail:
        with pytest.raises(ValueError, match="output check failed"):
            reports.measure_components(make_case, folder, 6, buffers=2, samples=4, warmup=1)
    else:
        reports.measure_components(make_case, folder, 6, buffers=2, samples=4, warmup=1)
    result = json.loads((folder / "result.json").read_text())
    assert result["complete"] is not fail
    assert len(result["raw"]) == (3 if fail else 8)
    with (folder / "raw.csv").open() as stream:
        assert len(list(csv.DictReader(stream))) == len(result["raw"])
    assert gates == (["before", "before_samples"] if fail else ["before", "before_samples", "after"])


@pytest.mark.parametrize("name", ("attention", "indexer", "indexer_components"))
def test_cli_refuses_existing_root(tmp_path, monkeypatch, name):
    from benchmarks.qsa import test_attention, test_indexer
    from tests.ops.qsa import test_indexer as components

    module = {"attention": test_attention, "indexer": test_indexer, "indexer_components": components}[name]
    monkeypatch.setattr(module, "DATA", tmp_path)
    args = ["qsa", "--gpu", "6", "--check-only", "--output", str(tmp_path)]
    if name != "indexer_components":
        args += ["--synthetic"]
    monkeypatch.setattr(sys, "argv", args)
    with pytest.raises(FileExistsError):
        module.main()