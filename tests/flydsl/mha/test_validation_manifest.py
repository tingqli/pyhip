"""Pure-stdlib evidence inventory regressions; no ROCm imports or GPU access."""

import hashlib
import json

import pytest

from . import validation_manifest as manifest


def test_junit_counts_skips_failures_and_errors_without_double_counting(tmp_path):
    source = tmp_path / "suite.xml"
    source.write_text('<testsuites><testsuite tests="4"><testcase name="ok"/>'
                      '<testcase name="skip"><skipped message="needs gfx950"/></testcase>'
                      '<testcase name="bad"><failure message="mismatch"/></testcase>'
                      '<testcase name="error"><error message="setup"/></testcase></testsuite></testsuites>')
    row = manifest.junit_summary(source)
    assert (row["tests"], row["passed"], row["skipped"], row["failures"], row["errors"]) == (4, 1, 1, 1, 1)
    assert row["skip_reasons"] == {"needs gfx950": 1}
    assert [case["name"] for case in row["failed_tests"]] == ["bad", "error"]


@pytest.fixture
def local_evidence(monkeypatch, tmp_path):
    monkeypatch.setattr(manifest, "ROOT", tmp_path)
    monkeypatch.setattr(manifest, "HERE", tmp_path / "source")
    manifest.HERE.mkdir()
    (manifest.HERE / "kernel.py").write_text("# fixture source\n")
    monkeypatch.setattr(manifest.subprocess, "check_output", lambda *a, **k: "fixture-git-value\n")
    directory = tmp_path / "results"
    directory.mkdir()
    (directory / "isa.s").write_text("s_endpgm\n")
    data = {"executed": False, "complete": True, "records": [{
        "isa": "/missing/historical/path.s", "retained_isa": "isa.s",
        "isa_sha256": hashlib.sha256((directory / "isa.s").read_bytes()).hexdigest(), "resources": {}}]}
    (directory / "resources.json").write_text(json.dumps(data))
    return directory


def test_manifest_verifies_retained_isa_not_stale_source_path(local_evidence):
    output = local_evidence / "manifest.json"
    result = manifest.summarize(local_evidence, output)
    assert not result["gpu_queried"] and len(result["isa"]) == 1
    assert result["isa"][0]["sha_verified"] and result["isa"][0]["inside_results_directory"]
    assert result["json_reports"][0]["executed"] is False
    assert result["source_files_sha256"]["source/kernel.py"]
    with pytest.raises(FileExistsError):
        manifest.summarize(local_evidence, output)


def test_manifest_rejects_corrupt_isa(local_evidence):
    (local_evidence / "isa.s").write_text("different\n")
    with pytest.raises(ValueError, match="SHA mismatch"):
        manifest.summarize(local_evidence, local_evidence / "manifest.json")


def test_manifest_keeps_invalid_json_attempts(local_evidence):
    (local_evidence / "failed.json").touch()
    result = manifest.summarize(local_evidence, local_evidence / "manifest.json")
    row = next(row for row in result["json_reports"] if row["path"] == "failed.json")
    assert not row["valid_json"]