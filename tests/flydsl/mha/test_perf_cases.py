"""CPU-only tests for performance coverage, timing and hardware-policy guards."""

import contextlib
from dataclasses import replace
import json
import subprocess
from types import SimpleNamespace

import pytest
import torch

if __package__:
    from . import _hardware, _runner, _testing, reproduce_baselines
    from ._perf_cases import BASELINES, Protocol, Workload, baseline_status, documented_workloads, select_workloads
    from ._testing import Backend, BF16_942, BF16_950, FP8, SWA
else:
    import _hardware
    import _runner
    import _testing
    import reproduce_baselines
    from _perf_cases import BASELINES, Protocol, Workload, baseline_status, documented_workloads, select_workloads
    from _testing import Backend, BF16_942, BF16_950, FP8, SWA


@pytest.fixture(autouse=True)
def no_gpu(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("CPU performance-contract tests must not query or launch a GPU")
    for name in ("get_device_properties", "current_stream", "synchronize", "is_available", "init", "_lazy_init"):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "check_output", forbidden)


def options(**updates):
    values = dict(mode="performance", matrix="documented", case=[], suite="all", dq=None, dv=None,
                  page=None, q=None, kv=None, heads=None, kv_heads=None, batch=None, causal=None,
                  window=None, sink=None, scale_mode=None, timer=None, buffers=None, warmup=None,
                  rounds=None, iterations=None, output=None, require_baseline=False,
                  allow_contention=False, tiles=False, aiter="off", ptl="current")
    values.update(updates)
    return SimpleNamespace(**values)


def test_original_22_workloads_are_retained():
    cases = {case.name: case for case in documented_workloads()}
    expected = set()
    for dq in (128, 192):
        expected.update((f"full_d{dq}_p64", f"causal32k_d{dq}_p64"))
        expected.update(f"swa_kv{kv}_d{dq}" for kv in (32768, 65536, 131072))
    expected.update(("full_d192_p32", "causal32k_d192_p32"))
    for page in (32, 64):
        expected.update((f"batch4_d192_p{page}", f"batch4_d128_h1_p{page}", f"singlehead40k_p{page}",
                         f"singlehead_causal32k_p{page}", f"h3_p{page}"))
    assert len(expected) == 22 and expected <= cases.keys()
    assert len(cases) == 44
    assert cases["h3_p64"].q_lens == cases["h3_p64"].kv_lens == (63225, 7)
    assert cases["h3_p64"].heads == cases["h3_p64"].kv_heads == 14


@pytest.mark.parametrize("dq", (128, 192))
def test_original_swa_scaling_and_window_sweeps(dq):
    cases = documented_workloads()
    assert {case.window for case in cases if case.name.startswith("swa_window") and case.dq == dq} == {0, 16, 64, 512, 1024}
    assert {case.q_lens[0] for case in cases if case.name.startswith("swa_query") and case.dq == dq} == {256, 2048, 4096, 65536}
    assert all(case.sink and case.causal for case in cases if case.name.startswith("swa_"))


def test_distinct_documented_timing_protocols():
    fp8, = select_workloads(("fp8_native_410t",))
    bf16, = select_workloads(("historical_bf16_250t_shape",))
    h3, = select_workloads(("historical_h3_events",))
    assert fp8.protocol == Protocol("profiler", 1200, 5, 100, 1, 20)
    assert bf16.protocol == Protocol("events", 10, 1, 50, 10, 0)
    assert h3.protocol == Protocol("events", 3, 1, 10, 10, 0)
    assert (bf16.dq, bf16.heads, bf16.page, bf16.q_lens, bf16.kv_lens) == (128, 1, 32, (40960,), (40960,))


def test_exact_effective_flops():
    fp8, = select_workloads(("fp8_native_410t",))
    bf16, = select_workloads(("historical_bf16_250t_shape",))
    swa, = select_workloads(("swa_kv131072_d192",))
    assert fp8.flops == 268435456000
    assert bf16.flops == 858993459200
    assert swa.flops == 21642608640
    assert Workload("masked", (5,), (2,), dq=128, heads=1, causal=True).flops == 2*3*256
    assert Workload("window0", (5,), (2,), dq=128, heads=1, causal=True, window=0).flops == 2*2*256


def measured_env(**updates):
    env = {"arch": "gfx942", "gpu": "AMD Instinct MI308X", "flydsl": "0.3.1",
           "torch": "2.12.1+rocm7.2", "hip": "7.2.53211",
           "limits": {"gpu_data": [{"limit": {"ptl_state": "Disabled", "ptl_format": "N/A"}}]}}
    env.update(updates)
    return env


def test_fp8_requires_documented_policy_and_throughput():
    workload, = select_workloads(("fp8_native_410t",))
    disabled, = baseline_status(workload, FP8.name, measured_env(), 414)
    assert disabled["status"] == "unmatched"
    assert "PTL Enabled/VECTOR,F8" in disabled["mismatch_reasons"][0]
    enabled = measured_env(limits={"gpu_data": [{"limit": {"ptl_state": "Enabled", "ptl_format": "VECTOR,F8"}}]})
    passed, = baseline_status(workload, FP8.name, enabled, 410)
    failed, = baseline_status(workload, FP8.name, enabled, 217)
    assert passed["status"] == "passed" and not passed["compiler_matches_original"]
    assert failed["status"] == "failed"


def test_bf16_250t_is_not_an_eight_wave_baseline():
    workload, = select_workloads(("historical_bf16_250t_shape",))
    row, = baseline_status(workload, BF16_942.name, measured_env(), 250)
    assert row["status"] == "unmatched"
    assert "historical_4wave_static" in row["mismatch_reasons"][0]
    swa, = select_workloads(("swa_kv131072_d192",))
    row, = baseline_status(swa, SWA.name, measured_env(), 253)
    assert row["status"] == "unmatched" and any("MI350X/gfx950" in reason for reason in row["mismatch_reasons"])


def test_capability_checks_do_not_change_original_shape():
    original, = select_workloads(("causal32k_d192_p32",))
    assert original.unsupported(FP8) and original.unsupported(BF16_950)
    assert original.unsupported(BF16_942) is None
    short = Workload("short_causal", (10240,), (2560,), causal=True)
    assert short.unsupported(BF16_942) and short.unsupported(FP8) is None
    assert short.q_lens == (10240,)
    swa, = select_workloads(("swa_window64_d128",))
    assert swa.unsupported(SWA) is None and swa.unsupported(BF16_950) is None
    assert swa.unsupported(FP8) and swa.unsupported(BF16_942)


def test_case_filters_and_protocol_overrides():
    selected = select_workloads(("swa_window*_d128",))
    assert len(selected) == 5
    with pytest.raises(ValueError, match="unknown"):
        select_workloads(("not-a-workload",))
    workload, = select_workloads(("historical_bf16_250t_shape",))
    assert _runner.measurement_protocol(options(), workload) == workload.protocol
    assert _runner.measurement_protocol(options(warmup=2), workload).warmup == 2
    assert _runner.case_selected(options(page=[32], dq=[128]), workload)
    assert not _runner.case_selected(options(page=[64]), workload)


def test_cli_list_cases_is_cpu_only(monkeypatch, capsys):
    monkeypatch.setattr(_runner.sys, "argv", ["test", "--mode", "performance", "--suite", "all",
                                              "--list-cases", "--case", "fp8_native_410t"])
    _runner.main(__file__)
    plan = json.loads(capsys.readouterr().out)
    assert plan["gpu_queried"] is False and len(plan["workloads"]) == 1
    assert plan["workloads"][0]["name"] == "fp8_native_410t"


def test_cli_rejects_acceptance_under_contention(monkeypatch):
    monkeypatch.setattr(_runner.sys, "argv", ["test", "--require-baseline", "--allow-contention"])
    with pytest.raises(SystemExit) as error:
        _runner.main(__file__)
    assert error.value.code == 2


def test_process_guard_preserves_other_workloads():
    assert not _hardware.other_processes([{"process_list": [{"process_info": "No running processes detected"}]}], 1)
    own = {"process_info": {"pid": 1, "mem_usage": {"value": 64}}}
    other = {"process_info": {"pid": 2, "mem_usage": {"value": 149000000000}, "cu_occupancy": 14}}
    assert _hardware.other_processes([own, other], 1) == [other["process_info"]]


def test_default_policy_never_changes_hardware(monkeypatch, tmp_path):
    monkeypatch.setattr(_hardware, "limits", lambda: pytest.fail("current policy must not query/set PTL"))
    with _hardware.ptl_experiment("current", tmp_path / "no-file.json"):
        pass
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("state", ("N/A", None, "Unknown"))
def test_non_ptl_device_rejects_policy_before_setter(monkeypatch, tmp_path, state):
    for name in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        monkeypatch.setenv(name, "0")
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda i:
                        SimpleNamespace(gcnArchName="gfx942", name="AMD Instinct MI325X", multi_processor_count=304))
    monkeypatch.setattr(_hardware, "ensure_idle", lambda *a: None)
    monkeypatch.setattr(_hardware, "limits", lambda: {"gpu_data": [{"limit": {"ptl_state": state, "ptl_format": "N/A"}}]})
    monkeypatch.setattr(_hardware.subprocess, "run", lambda *a, **k: pytest.fail("unsupported PTL must never reach a setter"))
    with pytest.raises(RuntimeError, match="does not report supported PTL"):
        with _hardware.ptl_experiment("VECTOR,F8", tmp_path / "unsupported.json"):
            pytest.fail("unsupported PTL must not run the experiment")
    assert not list(tmp_path.iterdir())


def test_mi325_cannot_be_relabelled_as_mi308_acceptance():
    workload, = select_workloads(("fp8_native_410t",))
    env = measured_env(gpu="AMD Instinct MI325X", compute_units=304, flydsl="0.2.2",
                       limits={"gpu_data": [{"limit": {"ptl_state": "N/A", "ptl_format": "N/A"}}]})
    row, = baseline_status(workload, FP8.name, env, 1000)
    assert row["status"] == "unmatched" and not row["exact_environment_match"]
    assert row["latency_ratio_vs_documented"] is None
    assert "requires MI308X/gfx942" in row["mismatch_reasons"]
    assert "requires PTL Enabled/VECTOR,F8" in row["mismatch_reasons"]


def test_event_timer_uses_all_rotating_buffers(monkeypatch):
    # Fully mocked events/calls: no device allocation, query or synchronization.
    order = []
    class Event:
        def __init__(self, **kwargs):
            pass
        def record(self):
            pass
        def synchronize(self):
            pass
        def elapsed_time(self, end):
            return 0.01
    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    calls = [lambda index=index: order.append(index) for index in range(10)]
    result = _runner.event_round(calls, iterations=50, warmup=10)
    assert order == list(range(10))*6
    assert result["interval"]["median_us"] == 10
    assert len(result["interval"]["raw_us"]) == 50 and result["buffer_count"] == 10


def test_benchmark_gate_fails_before_allocation(monkeypatch, tmp_path):
    args = options(case=["fp8_native_410t"], output=tmp_path / "blocked.json", require_baseline=True)
    monkeypatch.setattr(_runner, "environment", measured_env)
    monkeypatch.setattr(_runner, "make_case", lambda *a, **kw: pytest.fail("must not allocate before baseline preflight"))
    with pytest.raises(RuntimeError, match="baseline conditions"):
        _runner.benchmark(args, [FP8])
    report = json.loads(args.output.read_text())
    assert not report["complete"] and report["records"] == []


@pytest.mark.parametrize("fail", (None, "body", "format", "permission", "partial_enable"))
def test_ptl_restored_on_success_and_failure(monkeypatch, tmp_path, fail):
    # Simulate the manager; never invoke sudo, amd-smi, or a GPU runtime.
    for name in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        monkeypatch.setenv(name, "0")
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda i: SimpleNamespace(gcnArchName="gfx942"))
    monkeypatch.setattr(_hardware, "ensure_idle", lambda *a: None)
    state = {"ptl_state": "Disabled", "ptl_format": "N/A"}
    monkeypatch.setattr(_hardware, "limits", lambda: {"gpu_data": [{"limit": dict(state)}]})
    commands = []
    def run(command, **kwargs):
        commands.append(command)
        option, value = command[-2:]
        if option == "--ptl-status" and value == "1" and fail == "permission":
            return subprocess.CompletedProcess(command, 1, "", "sudo: a password is required")
        if option == "--ptl-format" and fail == "format":
            return subprocess.CompletedProcess(command, 1, "", "format change failed")
        if option == "--ptl-status":
            state["ptl_state"] = "Enabled" if value == "1" else "Disabled"
            if value == "0":
                state["ptl_format"] = "N/A"
            elif fail == "partial_enable":
                return subprocess.CompletedProcess(command, 1, "", "partial state change")
        else:
            state["ptl_format"] = value
        return subprocess.CompletedProcess(command, 0, "", "")
    monkeypatch.setattr(_hardware.subprocess, "run", run)
    path = tmp_path / "ptl.json"
    expected = (pytest.raises(ValueError, match="numerical failure") if fail == "body" else
                pytest.raises(subprocess.CalledProcessError) if fail else contextlib.nullcontext())
    with expected:
        with _hardware.ptl_experiment("VECTOR,F8", path):
            assert state["ptl_state"] == "Enabled"
            if fail == "body":
                raise ValueError("numerical failure")
            assert fail is None
    report = json.loads(path.read_text())
    assert report["restored"] and report["after"] == report["before"]
    expected_commands = [["--ptl-status", "1"]]
    if fail not in ("permission", "partial_enable"):
        expected_commands.append(["--ptl-format", "VECTOR,F8"])
    if fail != "permission":
        expected_commands.append(["--ptl-status", "0"])
    assert [command[-2:] for command in commands] == expected_commands


def test_all_named_baselines_have_workloads():
    names = {case.name for case in documented_workloads()}
    assert all(baseline.workload in names for baseline in BASELINES)


def enabled_env(**updates):
    return measured_env(limits={"gpu_data": [{"limit": {"ptl_state": "Enabled", "ptl_format": "VECTOR,F8"}}]}, **updates)


def test_baseline_rejects_relabelled_shape_or_protocol():
    workload, = select_workloads(("fp8_native_410t",))
    changed = replace(workload, q_lens=(512,), protocol=Protocol(warmup=1))
    check, = baseline_status(changed, FP8.name, enabled_env(flydsl="0.2.2"), 420)
    assert check["status"] == "unmatched" and not check["exact_environment_match"]
    assert check["latency_ratio_vs_documented"] is None


@pytest.mark.parametrize("value", (0, -1, float("inf"), float("nan")))
def test_baseline_requires_finite_positive_measurement(value):
    workload, = select_workloads(("fp8_native_410t",))
    with pytest.raises(ValueError, match="finite|positive"):
        baseline_status(workload, FP8.name, enabled_env(), value)


def test_unknown_original_runtime_is_not_an_exact_match():
    workload, = select_workloads(("historical_bf16_250t_shape",))
    check, = baseline_status(workload, "historical_4wave_static", measured_env(), 250)
    assert check["compiler_matches_original"] is None
    assert check["runtime_matches_original"] is None and not check["exact_environment_match"]


def test_custom_swa_preserves_multiwindow_and_dimensions():
    args = options(matrix="custom", suite="swa", q=[256], kv=[4096], dq=[192], dv=[192],
                   page=[32], window=[16, 64], scale_mode="per-tensor")
    workloads = _runner.benchmark_workloads(args)
    assert len(workloads) == 2 and {workload.window for workload in workloads} == {16, 64}
    assert all(workload.dv == 192 and workload.page == 32 and workload.scale_mode == "per-tensor"
               and workload.backends == (SWA.name,) for workload in workloads)
    assert all(workload.unsupported(SWA) for workload in workloads)


@pytest.mark.parametrize("case_name,backend,quantized,source_dtype", (
    ("fp8_native_410t", FP8, False, torch.float32),
    ("full_d192_p64", FP8, True, torch.bfloat16),
    ("full_d192_p64", BF16_942, False, torch.bfloat16),
))
def test_performance_input_protocol(monkeypatch, case_name, backend, quantized, source_dtype):
    workload, = select_workloads((case_name,))
    monkeypatch.setattr(_runner, "make_case", lambda *args, **kwargs: (args, kwargs))
    positional, kwargs = _runner.make_performance_case(workload, backend, 2)
    assert positional == (workload.q_lens, workload.kv_lens)
    assert kwargs["quantized"] is quantized and kwargs["source_dtype"] == source_dtype
    assert kwargs["padding_before_quantization"] == (source_dtype == torch.bfloat16)
    assert kwargs["seed"] == workload.seed + 2 and kwargs["dv"] == workload.dv


@pytest.fixture
def benchmark_runtime(monkeypatch):
    # Only host orchestration is tested here; these synthetic timings are not GPU results.
    state = SimpleNamespace(created=[], validated=[], isolation_checks=0, profile_us=650.0, pools=[])
    monkeypatch.setattr(_runner, "environment", enabled_env)
    monkeypatch.setattr(Backend, "available", property(lambda self: True))
    def idle():
        state.isolation_checks += 1
    monkeypatch.setattr(_runner, "require_idle_device", idle)
    def make_case(q_lens, kv_lens, **kwargs):
        case = SimpleNamespace(q_lens=q_lens, kv_lens=kv_lens, **kwargs)
        state.created.append(case)
        return case
    monkeypatch.setattr(_runner, "make_case", make_case)
    def make_call(case, backend, causal, **kwargs):
        out = torch.ones(1)
        return lambda: out, out, None
    monkeypatch.setattr(_runner, "make_call", make_call)
    def check(case, backend, out, lse, causal):
        state.validated.append(case)
        torch.testing.assert_close(out, torch.ones_like(out), rtol=0, atol=0)
        return out.clone(), None
    monkeypatch.setattr(_runner, "assert_close", check)
    def profile(call, **kwargs):
        call()
        sample = {"mean_us": state.profile_us, "raw_us": [state.profile_us] * 3, "kept_indices": [1, 2]}
        return {"attention": sample, "total": sample, "dispatches_per_call": 1}
    monkeypatch.setattr(_runner, "profile_round", profile)
    def events(calls, **kwargs):
        state.pools.append(calls)
        sample = {"mean_us": 3422.933, "median_us": 3422.933, "raw_us": [3422.933] * kwargs["iterations"]}
        return {"interval": sample, "timer": "events", "buffer_count": len(calls)}
    monkeypatch.setattr(_runner, "event_round", events)
    return state


def test_benchmark_strict_gate_and_event_metric(benchmark_runtime, tmp_path):
    args = options(case=["fp8_native_410t"], require_baseline=True, output=tmp_path / "fp8.json")
    result = _runner.benchmark(args, [FP8])
    assert result["complete"] and result["records"][0]["baseline_checks"][0]["status"] == "passed"
    assert benchmark_runtime.isolation_checks >= 3
    args = options(case=["historical_bf16_250t_shape"], output=tmp_path / "events.json")
    result = _runner.benchmark(args, [BF16_942])
    row, = result["records"]
    assert row["attention_us"] is None and row["total_gpu_us"] is None
    assert row["event_interval_us"]["flydsl"] == 3422.933
    assert len(benchmark_runtime.pools[0]) == len({id(call) for call in benchmark_runtime.pools[0]}) == 10
    assert len(benchmark_runtime.validated) == 11


@pytest.mark.parametrize("changes", ({"warmup": 1}, {"rounds": 1}, {"timer": "events"}))
def test_strict_protocol_preflight_before_allocation(benchmark_runtime, tmp_path, changes):
    args = options(case=["fp8_native_410t"], require_baseline=True, output=tmp_path / "preflight.json", **changes)
    with pytest.raises(RuntimeError, match="baseline conditions"):
        _runner.benchmark(args, [FP8])
    assert not benchmark_runtime.created


def test_changed_protocol_cannot_leave_exact_match_or_ratio(benchmark_runtime, monkeypatch, tmp_path):
    monkeypatch.setattr(_runner, "environment", lambda: enabled_env(flydsl="0.2.2"))
    args = options(case=["fp8_native_410t"], warmup=1, output=tmp_path / "diagnostic.json")
    result = _runner.benchmark(args, [FP8])
    check, = result["records"][0]["baseline_checks"]
    assert check["status"] == "unmatched" and not check["exact_environment_match"]
    assert check["latency_ratio_vs_documented"] is None


def test_failed_throughput_preserves_incomplete_report(benchmark_runtime, tmp_path):
    benchmark_runtime.profile_us = 1300.0
    args = options(case=["fp8_native_410t"], require_baseline=True, output=tmp_path / "failed.json")
    with pytest.raises(RuntimeError, match="baseline not reproduced"):
        _runner.benchmark(args, [FP8])
    saved = json.loads(args.output.read_text())
    assert not saved["complete"] and saved["records"][0]["baseline_checks"][0]["status"] == "failed"


def test_unsupported_causal_does_not_launch_or_rewrite(benchmark_runtime, tmp_path):
    args = options(matrix="custom", suite="pa", q=[10240], kv=[2560], causal=[1], dq=[192],
                   output=tmp_path / "unsupported.json")
    result = _runner.benchmark(args, [BF16_942])
    assert not benchmark_runtime.created and not result["records"]
    assert "KV>=Q" in result["unavailable"][0]["reason"]


def test_swa_measurement_without_defined_gate_cannot_pass(benchmark_runtime, monkeypatch, tmp_path):
    monkeypatch.setattr(_runner, "environment", lambda: measured_env(arch="gfx950", gpu="AMD Instinct MI350X"))
    args = options(case=["swa_kv131072_d192"], require_baseline=True, output=tmp_path / "no-gate.json")
    with pytest.raises(RuntimeError, match="baseline conditions"):
        _runner.benchmark(args, [SWA])
    assert not benchmark_runtime.created


def test_ptl_remapping_rejected_before_gpu_query(monkeypatch, tmp_path):
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "1")
    with pytest.raises(RuntimeError, match="GPU0|HIP_VISIBLE_DEVICES"):
        with _hardware.ptl_experiment("VECTOR,F8", tmp_path / "blocked.json"):
            pytest.fail("remapped device must be rejected")


def test_busy_gpu_never_reaches_ptl_setter(monkeypatch, tmp_path):
    for name in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        monkeypatch.setenv(name, "0")
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda i: SimpleNamespace(gcnArchName="gfx942"))
    monkeypatch.setattr(_hardware, "ensure_idle", lambda *a: (_ for _ in ()).throw(RuntimeError("other process")))
    with pytest.raises(RuntimeError, match="other process"):
        with _hardware.ptl_experiment("VECTOR,F8", tmp_path / "busy.json"):
            pytest.fail("busy GPU must be rejected before any setting is read or changed")


@pytest.mark.parametrize("quantized", (False, True))
def test_documented_input_generation_matches_cpu_transcription(monkeypatch, quantized):
    # Execute tensor generation on CPU and compare with a direct transcription
    # of the original README input protocol, including zero-before-quantize.
    for name in ("randn", "ones", "tensor", "linspace"):
        function = getattr(torch, name)
        def cpu(*args, _function=function, **kwargs):
            if kwargs.get("device") == "cuda":
                kwargs["device"] = "cpu"
            return _function(*args, **kwargs)
        monkeypatch.setattr(torch, name, cpu)
    dtype = torch.float8_e4m3fnuz if quantized else torch.bfloat16
    case = _testing.make_case((3,), (5,), dtype=dtype, dq=128, heads=2,
        source_dtype=torch.bfloat16, padding_before_quantization=True, quantized=quantized,
        poison_tail=False, seed=20260905)
    torch.manual_seed(20260905)
    q = torch.randn(3, 2, 128, dtype=torch.bfloat16)
    k = torch.randn(1, 64, 1, 128, dtype=torch.bfloat16)
    v = torch.randn(1, 64, 1, 128, dtype=torch.bfloat16)
    order = torch.randperm(1).tolist()
    k[order[0], 5:] = 0
    v[order[0], 5:] = 0
    if quantized:
        limit = torch.finfo(dtype).max
        qs = q.float().abs().amax(-1, keepdim=True).clamp_min(1e-12) / limit
        ks, vs = (value.float().abs().max().reshape(1) / limit for value in (k, v))
        q, k, v = ((value.float() / scale).to(dtype) for value, scale in ((q, qs), (k, ks), (v, vs)))
    else:
        qs, ks, vs = torch.ones(3, 2, 1), torch.ones(1), torch.ones(1)
    for actual, expected in ((case.q, q), (case.k_pages, k), (case.v_pages, v),
                             (case.qs, qs), (case.ks, ks), (case.vs, vs)):
        torch.testing.assert_close(actual.float(), expected.float(), rtol=0, atol=0)
    assert case.page_order == order and case.q.device.type == "cpu"


@pytest.fixture
def reproducer_runtime(monkeypatch, benchmark_runtime):
    state = benchmark_runtime
    state.cache_clears = []
    monkeypatch.setattr(reproduce_baselines, "environment", enabled_env)
    monkeypatch.setattr(reproduce_baselines, "require_idle_device", lambda: None)
    monkeypatch.setattr(reproduce_baselines, "make_call", _runner.make_call)
    monkeypatch.setattr(reproduce_baselines, "assert_close", _runner.assert_close)
    monkeypatch.setattr(reproduce_baselines, "profile_round", _runner.profile_round)
    monkeypatch.setattr(reproduce_baselines, "event_round", _runner.event_round)
    original = SimpleNamespace(PagedAttention=SimpleNamespace(cache_clear=lambda: state.cache_clears.append(True)))
    monkeypatch.setattr(reproduce_baselines, "load_original", lambda *a: original)
    def original_call(case, backend, module, causal):
        return _runner.make_call(case, backend, causal)[0]
    monkeypatch.setattr(reproduce_baselines, "original_call", original_call)
    monkeypatch.setattr(reproduce_baselines, "original_swa_call", lambda case, module, causal: original_call(case, SWA, module, causal))
    state.env = SimpleNamespace(debug=SimpleNamespace(dump_ir=True, dump_asm=True, enable_debug_info=True))
    return state


def test_reproducer_uses_documented_quantization_and_gates(reproducer_runtime, tmp_path):
    args = options(backend="fp8", case=["fp8_native_410t", "full_d192_p64"], warmup=1, four_wave=False,
                   output=tmp_path / "reproducer.json")
    result = reproduce_baselines._run(args, tmp_path, reproducer_runtime.env)
    assert result["complete"] and [case.quantized for case in reproducer_runtime.created] == [False, True]
    assert all(row["original_current_bit_exact"] for row in result["records"])
    check, = result["records"][0]["baseline_checks"]
    assert check["status"] == "unmatched" and check["latency_ratio_vs_documented"] is None
    assert set(result["records"][0]["latency_us"]) == {"current", "original_8wave"}


def test_reproducer_resets_original_cache_per_workload_not_buffer(reproducer_runtime, tmp_path):
    args = options(backend="bf16", case=["historical_bf16_*"], four_wave=False,
                   output=tmp_path / "bf16.json")
    result = reproduce_baselines._run(args, tmp_path, reproducer_runtime.env)
    assert len(result["records"]) == len(reproducer_runtime.cache_clears) == 2
    assert len(reproducer_runtime.created) == 20
    assert all(len(pool) == 10 for pool in reproducer_runtime.pools)
    assert all(row["attention_us"] is None and row["total_gpu_us"] is None for row in result["records"])


def test_reproducer_swa_requires_original_native_gpu(reproducer_runtime, tmp_path):
    args = options(backend="swa", four_wave=False, output=tmp_path / "swa.json")
    with pytest.raises(RuntimeError, match="original SWA baseline requires gfx950"):
        reproduce_baselines._run(args, tmp_path, reproducer_runtime.env)
    assert not reproducer_runtime.created
    assert not json.loads(args.output.read_text())["complete"]


def test_reproducer_original_swa_is_not_skipped_on_gfx950(reproducer_runtime, monkeypatch, tmp_path):
    monkeypatch.setattr(reproduce_baselines, "environment", lambda: measured_env(arch="gfx950", gpu="AMD Instinct MI350X"))
    args = options(backend="swa", case=["swa_kv131072_d192"], warmup=1, four_wave=False,
                   output=tmp_path / "swa_native_plan.json")
    result = reproduce_baselines._run(args, tmp_path, reproducer_runtime.env)
    assert set(result["records"][0]["latency_us"]) == {"current", "original_1wave"}


def test_reproducer_restores_debug_flags_on_failure(monkeypatch, tmp_path):
    debug = SimpleNamespace(dump_ir=True, dump_asm=False, enable_debug_info=True)
    monkeypatch.setitem(_runner.sys.modules, "flydsl.utils", SimpleNamespace(env=SimpleNamespace(debug=debug)))
    def fail(args, directory, env):
        env.debug.dump_ir = env.debug.dump_asm = env.debug.enable_debug_info = False
        raise ValueError("comparison failed")
    monkeypatch.setattr(reproduce_baselines, "_run", fail)
    with pytest.raises(ValueError, match="comparison failed"):
        reproduce_baselines.run(options(), tmp_path)
    assert vars(debug) == dict(dump_ir=True, dump_asm=False, enable_debug_info=True)


def test_four_wave_adapter_binds_each_case_and_reports_real_dimensions(monkeypatch):
    outputs = []
    def factory(*factory_args, **options):
        def call(*args, **kwargs):
            outputs.append((factory_args, options, args, kwargs))
            return args[-1]
        return call
    module = SimpleNamespace(MHA=factory)
    def case(q, kv):
        values = dict(heads=2, kv_heads=1, dq=128, dv=192, page=32, window_left=-1,
                      q_lens=(q,), kv_lens=(kv,), q=torch.ones(q, 2, 128), sinks=None)
        values.update({name: torch.ones(1) for name in ("k", "v", "cq", "ck", "indptr", "indices", "qs", "ks", "vs", "last")})
        return SimpleNamespace(**values)
    small, large = case(3, 5), case(7, 9)
    call_small = reproduce_baselines.branch_four_wave_call(small, module, False)
    call_large = reproduce_baselines.branch_four_wave_call(large, module, False)
    assert call_small().shape == (3, 2, 192) and call_large().shape == (7, 2, 192)
    assert [row[2][7:9] for row in outputs] == [(3, 5), (7, 9)]
    assert all(row[0][3] == 192 for row in outputs)


@pytest.mark.parametrize("protocol", (
    dict(iterations=0), dict(rounds=0), dict(buffers=0), dict(warmup=-1), dict(buffers=10),
))
def test_invalid_protocol_rejected_on_cpu(protocol):
    with pytest.raises(ValueError):
        Protocol(**protocol)


def test_cli_plan_saves_effective_protocol(monkeypatch, capsys, tmp_path):
    output = tmp_path / "plan.json"
    monkeypatch.setattr(_runner.sys, "argv", ["test", "--list-cases", "--case", "fp8_native_410t",
        "--warmup", "7", "--output", str(output)])
    _runner.main(__file__)
    capsys.readouterr()
    plan = json.loads(output.read_text())
    assert plan["gpu_queried"] is False and plan["executed"] is False
    assert plan["workloads"][0]["protocol"]["warmup"] == 1200
    assert plan["workloads"][0]["measurement_protocol"]["warmup"] == 7