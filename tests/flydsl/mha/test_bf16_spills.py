"""Spill-fix CPU guards and pinned-original native numerical comparison."""

import contextlib
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

if __package__:
    from . import compile_bf16_942, watch_performance, recheck_performance, reproduce_baselines
    from ._testing import BF16_942, make_case, make_call, assert_close
    from .validate_preservation import load_original, original_call
else:
    import compile_bf16_942
    import watch_performance
    import recheck_performance
    import reproduce_baselines
    from _testing import BF16_942, make_case, make_call, assert_close
    from validate_preservation import load_original, original_call


@pytest.mark.parametrize("message", ("error: could not allocate output register for constraint 'a'", "LLVM ERROR: bad code", "file:2: error: failed"))
def test_codegen_errors_cannot_pass_by_emitting_isa(message):
    with pytest.raises(RuntimeError, match="invalid code generation"):
        compile_bf16_942.reject_codegen_errors(message)
    compile_bf16_942.reject_codegen_errors("normal compiler output")


def test_streamed_k_preserves_fragment_reduction_order():
    # Fragment layout (4,1,(2,D/16)):(1,0,(4,8)) and 32x32x8 atoms.
    for dimension in (128, 192):
        original = [part * 8 + half * 4 + value for part in range(dimension // 16)
                    for half in range(2) for value in range(4)]
        streamed = [offset + value for offset in range(0, dimension // 2, 8) for value in range(8)]
        assert streamed == original
    assert all((tid & 511) == tid for tid in range(512))


@pytest.mark.parametrize("workgroups", (80, 304))
def test_compile_case_uses_explicit_workgroup_metadata(monkeypatch, tmp_path, workgroups):
    def forbidden(*args, **kwargs):
        pytest.fail("metadata-only compilation must not query or initialize the GPU")
    for name in ("_lazy_init", "get_device_properties", "current_stream", "synchronize", "is_available"):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    launch = SimpleNamespace(compile_hints={})
    monkeypatch.setattr(compile_bf16_942.module, "_build_attention", lambda *a, **k: launch)
    resource = {"group_segment_fixed_size": 24576, "private_segment_fixed_size": 0,
                "vgpr_count": 252, "sgpr_count": 106, "vgpr_spill_count": 0,
                "sgpr_spill_count": 0, "agpr_count": 0}
    def compile_meta(actual_launch, *args):
        assert actual_launch is launch and args[-2] == workgroups
        assert args[-3].shape == (workgroups + 1,)
        assert all(arg.device.type == "meta" for arg in args if isinstance(arg, torch.Tensor))
        (tmp_path / "test_final_isa.s").write_text("\n".join(f".{k}: {v}" for k, v in resource.items()))
    monkeypatch.setattr(compile_bf16_942.flyc, "compile", compile_meta)
    row = compile_bf16_942.compile_case(tmp_path, dq=192, dv=128, page=32, causal=False,
                                      with_lse=False, workgroups=workgroups)
    assert row["workgroups"] == workgroups and row["counter_shape"] == [workgroups + 1]
    assert not row["gpu_queried"] and not row["executed"]
    assert row["resources"]["private_segment_fixed_size"] == 0


@pytest.mark.parametrize("workgroups", (0, -1, True, 1.5))
def test_compile_case_rejects_invalid_workgroups(tmp_path, workgroups):
    with pytest.raises(ValueError, match="workgroups must be a positive integer"):
        compile_bf16_942.compile_case(tmp_path, dq=192, dv=128, page=32, causal=False,
                                    with_lse=False, workgroups=workgroups)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("busy,allow_resident", ((False, False), (True, False), (True, True)))
def test_watcher_uses_stable_samples_and_process_guard(monkeypatch, tmp_path, busy, allow_resident):
    launched = []
    class Monitor:
        def __init__(self, *args, **kwargs):
            self.stdout = iter(["gpu,gfx_activity\n", "0,100\n", "0,0\n", "0,100\n", "0,0\n", "0,0\n"])
            self.stopped = False
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def poll(self):
            return 0 if self.stopped else None
        def terminate(self):
            self.stopped = True
        def wait(self):
            return 0
    monkeypatch.setattr(watch_performance.subprocess, "Popen", Monitor)
    monkeypatch.setattr(watch_performance.subprocess, "run", lambda command: launched.append(command) or
                        watch_performance.subprocess.CompletedProcess(command, 0))
    def idle(device):
        if busy:
            raise RuntimeError("resident worker")
    monkeypatch.setattr(watch_performance, "ensure_idle", idle)
    report = tmp_path / "watch.json"
    extra = ["--allow-resident-workers"] if allow_resident else []
    monkeypatch.setattr(watch_performance.sys, "argv", ["watch", "--record", str(report), *extra, "--", "test-command"])
    blocked = busy and not allow_resident
    with pytest.raises(RuntimeError, match="stream ended") if blocked else contextlib.nullcontext():
        assert watch_performance.main() == 0
    saved = json.loads(report.read_text())
    assert saved["samples_seen"] == 5
    assert bool(launched) is not blocked and saved["triggered"] is not blocked


def test_low_utilization_queue_never_changes_resident_workers_policy(monkeypatch, tmp_path):
    commands = []
    def occupied(_device):
        raise RuntimeError("resident worker")
    monkeypatch.setattr(recheck_performance, "ensure_idle", occupied)
    monkeypatch.setattr(recheck_performance.subprocess, "run", lambda command, **kwargs:
                        commands.append(command) or recheck_performance.subprocess.CompletedProcess(command, 0))
    monkeypatch.setattr(recheck_performance.sys, "argv", ["queue", "--when-low", "--output-dir", str(tmp_path)])
    assert recheck_performance.main() == 0
    assert len(commands) == 2
    for command in commands:
        assert command[command.index("--ptl") + 1] == "current"
        assert "--allow-contention" in command and "--require-baseline" not in command
    assert json.loads((tmp_path / "queue.json").read_text())["diagnostic_only"]


def test_nonexclusive_reproducer_rejects_ptl_change(monkeypatch, tmp_path):
    monkeypatch.setattr(reproduce_baselines.sys, "argv", ["reproduce", "--backend", "bf16",
        "--allow-contention", "--ptl", "VECTOR,BF16", "--output", str(tmp_path / "blocked.json")])
    with pytest.raises(SystemExit) as error:
        reproduce_baselines.main()
    assert error.value.code == 2 and not (tmp_path / "blocked.json").exists()


@pytest.fixture(scope="module")
def pinned_original(tmp_path_factory):
    if not BF16_942.available:
        pytest.skip("requires native gfx942; compile-only is not numerical validation")
    from flydsl.utils import env
    directory = tmp_path_factory.mktemp("bf16_spill_original")
    saved = (env.debug.dump_ir, env.debug.dump_dir, env.debug.dump_asm, env.debug.enable_debug_info)
    try:
        with contextlib.chdir(directory):
            module = load_original(BF16_942.module, directory)
        env.debug.dump_ir = env.debug.dump_asm = env.debug.enable_debug_info = False
        yield module
    finally:
        env.debug.dump_ir, env.debug.dump_dir, env.debug.dump_asm, env.debug.enable_debug_info = saved


@pytest.mark.parametrize("dq", (128, 192))
@pytest.mark.parametrize("dv", (128, 192))
@pytest.mark.parametrize("page", (32, 64, 128))
@pytest.mark.parametrize("causal", (False, True))
def test_spill_fix_bit_exact_original(pinned_original, dq, dv, page, causal):
    pinned_original.PagedAttention.cache_clear()
    case = make_case((0, 17, 259), (33, 65, 321), dtype=torch.bfloat16, dq=dq, dv=dv, page=page,
                     heads=4, kv_heads=2, nonunit_scales=True, poison_tail=False)
    current = make_call(case, BF16_942, causal)[0]
    original = original_call(case, BF16_942, pinned_original, causal)
    expected = original().clone()
    result = current()
    assert_close(case, BF16_942, result, None, causal)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    for _ in range(3):
        torch.testing.assert_close(current(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("index", (0, 1))
def test_final_spill_evidence_matches_source(index):
    here = Path(__file__).resolve().parent
    status = json.loads((here / "results/bf16_spill_status.json").read_text())
    report = here / "results" / status["resource_reports"][index]
    if not report.exists():
        pytest.skip("final compile matrix has not been generated")
    data = json.loads(report.read_text())
    assert data["complete"] and not data["executed"] and not data["gpu_queried"]
    assert data["source_sha256"] == hashlib.sha256((here / "mha_pa_bf16_942.py").read_bytes()).hexdigest()
    assert len(data["records"]) == 48
    for row in data["records"]:
        resource = row["resources"]
        assert resource["private_segment_fixed_size"] == resource["vgpr_spill_count"] == row["scratch_instructions"] == 0
        assert resource["agpr_count"] == 0
        # Residual NC/V128/LSE SGPR lane transfers are reported, not hidden.
        assert resource["sgpr_spill_count"] <= (2 if row["with_lse"] else 0)