"""Offline throughput reporting must not mix timer scopes or rewrite evidence."""

import json

import pytest

from .export_performance import export, performance_rows


def report(*, gather=False, complete=True):
    times = {"swa_direct": 10.0, "gather": 3.0, "gather_aiter_ck_linear": 25.0} if gather else {"current": 10.0, "original_8wave": 20.0}
    return {"complete": complete, "environment": {"gpu": "MI325X", "arch": "gfx942", "compute_units": 304},
            "records": [{"name": "example", "backend": "swa_bf16" if gather else "bf16_942",
                         "effective_flops": 1_000_000_000, "event_interval_us": times,
                         "attention_us": None, "total_gpu_us": {name: us - 1 for name, us in times.items()},
                         "q_lens": [32], "kv_lens": [64]}]}


def test_export_uses_each_candidate_interval_not_attention_or_summed_components():
    rows = performance_rows(report(gather=True), "run.json")
    by_name = {row["candidate"]: row for row in rows}
    assert by_name["swa_direct"]["tflops"] == 100
    assert by_name["gather"]["tflops"] is None and not by_name["gather"]["tflops_applicable"]
    assert by_name["gather_aiter_ck_linear"]["tflops"] == 40
    assert all(row["timer_metric"] == "event_interval_us" and row["compute_units"] == 304 for row in rows)


def test_export_supports_original_current_and_profiler_reports():
    data = report()
    row = data["records"][0]
    row["workload"] = {"name": row.pop("name"), "q_lens": [32], "kv_lens": [64]}
    row["attention_us"] = row.pop("event_interval_us")
    rows = performance_rows(data, "original.json")
    assert [value["tflops"] for value in rows] == [100, 50]
    assert all(value["timer_metric"] == "attention_us" for value in rows)


def test_export_does_not_assign_flydsl_baseline_to_another_candidate():
    data = report()
    record = data["records"][0]
    record["baseline_checks"] = [{"status": "passed", "measured_tflops": 100}]
    record["comparison_baseline_checks"] = {"original_8wave": [{"status": "unmatched", "measured_tflops": 50}]}
    rows = performance_rows(data, "comparison.json")
    assert rows[0]["baseline_checks"][0]["status"] == "passed"
    assert rows[1]["baseline_checks"][0]["status"] == "unmatched"
    assert rows[1]["baseline_checks"][0]["measured_tflops"] == 50
    data = report(gather=True)
    data["records"][0]["baseline_checks"] = [{"status": "passed"}]
    assert all(row["baseline_checks"] == [] for row in performance_rows(data, "references.json"))


def test_export_retains_requested_relative_gate_only_on_current():
    data = report()
    record = data["records"][0]
    record["requested_reference"] = {"path": "specified.py", "sha256": "source"}
    record["requested_reference_check"] = {"status": "passed", "max_latency_ratio": 1.05}
    rows = performance_rows(data, "relative.json")
    assert rows[0]["requested_reference_check"] == record["requested_reference_check"]
    assert rows[1]["requested_reference_check"] is None
    assert all(row["requested_reference"] == record["requested_reference"] for row in rows)


@pytest.mark.parametrize("bad", (0, -1, True, float("nan"), float("inf")))
def test_export_rejects_invalid_measurement(bad):
    data = report()
    data["records"][0]["event_interval_us"]["current"] = bad
    with pytest.raises(ValueError, match="latency"):
        performance_rows(data, "bad.json")


def test_export_rejects_inconsistent_or_gather_tflops():
    data = report()
    data["records"][0]["tflops"] = {"current": 99}
    with pytest.raises(ValueError, match="disagrees"):
        performance_rows(data, "bad.json")
    data = report(gather=True)
    data["records"][0]["tflops"] = {"gather": 0}
    with pytest.raises(ValueError, match="gather-only"):
        performance_rows(data, "bad.json")


def test_export_retains_partial_status_and_original_bytes(tmp_path):
    source = tmp_path / "source.json"
    content = json.dumps(report(complete=False)).encode()
    source.write_bytes(content)
    target = tmp_path / "summary"
    result = export([source], target)
    assert not result["gpu_queried"] and all(not row["report_complete"] for row in result["rows"])
    assert source.read_bytes() == content
    assert "100.000" in target.with_suffix(".md").read_text()
    assert "original_8wave" in target.with_suffix(".csv").read_text()
    with pytest.raises(FileExistsError):
        export([source], target)


def test_compile_only_is_not_reported_as_measured_performance(tmp_path):
    source = tmp_path / "compile.json"
    source.write_text(json.dumps({"complete": True, "records": [{"executed": False, "resources": {}}]}))
    with pytest.raises(ValueError, match="no measured performance"):
        export([source], tmp_path / "summary")
    data = report()
    data["executed"] = False
    with pytest.raises(ValueError, match="unexecuted"):
        performance_rows(data, "not-native.json")