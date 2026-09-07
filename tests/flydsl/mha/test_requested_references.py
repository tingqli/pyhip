"""CPU-only contracts for the explicitly requested BF16/FP8 performance gates."""

import builtins
import json
import subprocess
from types import SimpleNamespace

import pytest

from . import compare_requested_references as comparison


@pytest.fixture(autouse=True)
def no_rocm_or_real_processes(monkeypatch):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split(".")[0] in ("torch", "flydsl", "pyhip"):
            pytest.fail("CPU reference tests must not import ROCm dependencies")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(comparison.subprocess, "Popen", lambda *a, **k: pytest.fail("no real subprocess in CPU contracts"))


def test_requested_sources_are_exact_and_not_old_dtype_mapping():
    bf16 = comparison.reference_identity("bf16")
    fp8 = comparison.reference_identity("fp8")
    assert bf16["path"] == "tests/flydsl/test_attn_8wave_32x32_lkgv.py" and bf16["factory"] == "MHA"
    assert fp8["path"] == "tests/flydsl/pa_8wave/pa_prefill_8w32x32.py" and fp8["factory"] == "PagedAttention"
    assert fp8["sha256"] == "620209a023ccb5ea566489774d19edae880dfbcee298613233ed6f01f3b59849"


def test_reference_hash_cannot_be_silently_relabelled(tmp_path):
    relative, _ = comparison.REFERENCE_SOURCES["bf16"]
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    path.write_text("not the selected reference")
    with pytest.raises(ValueError, match="reference SHA changed"):
        comparison.reference_identity("bf16", tmp_path)


@pytest.mark.parametrize("cu", (80, 304))
def test_bf16_default_uses_actual_cu_and_dense_contract(cu):
    case = comparison.cases(cu)[0]
    assert (case.q, case.kv, case.heads, case.kv_heads, case.dq, case.dv, case.page) == (256 * cu, 256 * cu, 8, 8, 128, 128, 32)
    assert not case.causal and case.buffers == 10
    assert case.flops == 4 * 8 * (256 * cu) ** 2 * 128
    assert case.to_dict()["protocol"]["round_statistic"] == "upper_median_us"


def test_fp8_driver_default_is_scalar_quantized_causal32k():
    case = next(case for case in comparison.cases(304) if case.name == "fp8_prefill_main")
    assert (case.q, case.kv, case.heads, case.kv_heads, case.dq, case.dv, case.page) == (32768, 32768, 16, 1, 192, 128, 64)
    assert case.causal and case.scale_mode == "per-tensor" and case.buffers == 12
    assert case.flops == 2 * 16 * (32768 * 32769 // 2) * (192 + 128)
    assert "FNUZ quantization" in case.to_dict()["input_protocol"]


@pytest.mark.parametrize("cu", (0, -1, True, 3.5))
def test_plan_rejects_invalid_cu(cu):
    with pytest.raises(ValueError, match="compute_units"):
        comparison.cases(cu)


def eligibility(**changes):
    values = dict(native=True, correctness=True, isolated=True, source_matched=True, protocol_matched=True)
    return {**values, **changes}


@pytest.mark.parametrize("kind,current,status", (("bf16", 104, "passed"), ("bf16", 106, "failed"),
                                                 ("fp8", 100, "passed"), ("fp8", 100.01, "failed"), ("fp8", 90, "passed")))
def test_relative_acceptance_never_uses_absolute_mi308_tflops(kind, current, status):
    check = comparison.gate(kind, current, 100, **eligibility())
    assert check["status"] == status
    assert check["max_latency_ratio"] == (1.05 if kind == "bf16" else 1.0)


@pytest.mark.parametrize("flag", ("native", "correctness", "isolated", "source_matched", "protocol_matched"))
def test_unverified_or_unsupported_cannot_pass(flag):
    check = comparison.gate("fp8", 1, 100, **eligibility(**{flag: False}))
    assert check["status"] == "unmatched" and check["mismatch_reasons"]


@pytest.mark.parametrize("value", (0, -1, True, float("nan"), float("inf")))
def test_gate_requires_real_finite_latencies(value):
    with pytest.raises(ValueError, match="latencies"):
        comparison.gate("fp8", value, 100, **eligibility())


def test_upper_median_latency_is_not_average_or_separately_sorted_tflops():
    assert comparison.round_statistic(list(range(1, 11)), "bf16") == 6
    assert comparison.round_statistic(list(range(1, 11)), "fp8") == 5.5


@pytest.mark.parametrize("kind", ("bf16", "fp8"))
def test_matched_timer_rotates_all_buffers_and_retains_raw_samples(kind):
    case = comparison.select_cases(304, [kind])[0]
    order, active, synchronizations = [], [], []

    class Timer:
        def __init__(self, flops, name):
            self.name = name
            assert flops == case.flops
        def __enter__(self):
            active.append(self.name)
            return self
        def __exit__(self, *_args):
            active.pop()
        def dt(self):
            return (90 if self.name == "current" else 100) / 1e6

    def call(name, index):
        if active:
            assert active[-1] == name
            order.append((name, index))
    pools = {name: [lambda name=name, index=index: call(name, index) for index in range(case.buffers)]
             for name in ("reference", "current")}
    latency, rounds = comparison.measure(pools, case, Timer, lambda: synchronizations.append(1))
    assert latency == {"reference": 100, "current": 90}
    assert len(synchronizations) == 5 and len(order) == 100
    assert all(len(rows) == 5 and all(len(row["raw_us"]) == 10 for row in rows) for rows in rounds.values())
    for index in range(0, len(order), 2):
        assert {order[index][0], order[index + 1][0]} == {"reference", "current"}
        assert order[index][1] == order[index + 1][1]


def test_cpu_plan_never_imports_or_queries_gpu(monkeypatch, tmp_path, capsys):
    output = tmp_path / "plan.json"
    monkeypatch.setattr(comparison.sys, "argv", ["compare", "--list-cases", "--compute-units", "304", "--output", str(output)])
    assert comparison.main() == 0
    data = json.loads(output.read_text())
    assert len(data["workloads"]) == 9 and not data["gpu_queried"] and not data["executed"]
    assert not data["complete"] and not data["acceptance_passed"] and not data["records"]
    assert data["workloads"][0]["q"] == 77824
    assert "src/misc.py" in data["protocol_sources_sha256"]
    capsys.readouterr()


def test_blocked_preflight_saves_failure_before_native_import(monkeypatch, tmp_path):
    output = tmp_path / "blocked.json"
    monkeypatch.setattr(comparison.sys, "argv", ["compare", "--output", str(output)])
    monkeypatch.setattr(comparison, "preflight", lambda *_: (_ for _ in ()).throw(RuntimeError("GPU health timed out")))
    monkeypatch.setattr(comparison, "run_native", lambda *_: pytest.fail("must not enter native path"))
    with pytest.raises(RuntimeError, match="GPU health timed out"):
        comparison.main()
    report = json.loads(output.read_text())
    assert not report["complete"] and not report["acceptance_passed"] and not report["records"]
    assert report["progress"]["stage"] == "health_preflight"


def test_preflight_only_is_never_native_acceptance(monkeypatch, tmp_path):
    output = tmp_path / "health.json"
    monkeypatch.setattr(comparison.sys, "argv", ["compare", "--preflight-only", "--output", str(output)])
    monkeypatch.setattr(comparison, "preflight", lambda *_: {"settings_changed": False})
    monkeypatch.setattr(comparison, "run_native", lambda *_: pytest.fail("health-only must not run a kernel"))
    assert comparison.main() == 0
    report = json.loads(output.read_text())
    assert not report["executed"] and not report["complete"] and not report["acceptance_passed"]
    assert report["progress"]["stage"] == "health_preflight_complete"


def test_health_timeout_kills_only_its_reader_and_does_not_reset(monkeypatch):
    calls = []

    class Reader:
        pid = 1234
        def communicate(self, timeout):
            calls.append(("communicate", timeout))
            if len(calls) == 1:
                raise subprocess.TimeoutExpired("read", timeout)
            return "", ""
        def kill(self):
            calls.append(("kill", self.pid))
    monkeypatch.setattr(comparison.subprocess, "Popen", lambda *a, **k: Reader())
    with pytest.raises(RuntimeError, match="no ROCm imports"):
        comparison.health_read(["amd-smi", "process"], timeout_seconds=3)
    assert calls == [("communicate", 3), ("kill", 1234), ("communicate", 1)]


@pytest.mark.parametrize("variable", ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"))
def test_missing_or_remapped_device_rejected_before_driver_read(monkeypatch, variable):
    for name in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        monkeypatch.setenv(name, "0")
    monkeypatch.delenv(variable)
    with pytest.raises(RuntimeError, match="explicit physical GPU0"):
        comparison.preflight("unused-smi")
    monkeypatch.setenv(variable, "1")
    with pytest.raises(RuntimeError, match="explicit physical GPU0"):
        comparison.preflight("unused-smi")


@pytest.mark.parametrize("variable,value", (("FLYDSL_COMPILE_ONLY", "1"), ("FLYDSL_COMPILE_ARCH", "gfx950")))
def test_compile_only_cannot_enter_native_acceptance(monkeypatch, variable, value):
    for name in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        monkeypatch.setenv(name, "0")
    monkeypatch.setenv(variable, value)
    with pytest.raises(RuntimeError, match="compile-only"):
        comparison.preflight("unused-smi")


def test_fp8_adapter_uses_requested_factory_without_old_memory_mode():
    calls = []
    output = object()

    def factory(*args, **kwargs):
        calls.append(("factory", args, kwargs))
        return lambda *a, **k: calls.append(("call", a, k)) or output
    case = SimpleNamespace(heads=16, kv_heads=1, dq=192, dv=128, page=64, mode="per-tensor",
        q=SimpleNamespace(shape=(32768, 16, 192), device="cuda"), k=object(), v=object(),
        cq=object(), ck=object(), indptr=object(), indices=object(), qs=object(), ks=object(), vs=object(), last=object(),
        q_lens=(32768,), kv_lens=(32768,))
    torch = SimpleNamespace(bfloat16="bf16", empty=lambda *a, **k: output)
    call = comparison.paged_reference_call(case, SimpleNamespace(PagedAttention=factory), True, torch)
    assert call() is output
    assert calls[0] == ("factory", (16, 1, 192, 128, 64, True, "per-tensor"), {})
    assert calls[1][2] == {"out": output}


def test_unknown_shape_filter_does_not_silently_shrink():
    with pytest.raises(ValueError, match="unknown"):
        comparison.select_cases(304, ["bf16"], ["fp8_prefill_main"])