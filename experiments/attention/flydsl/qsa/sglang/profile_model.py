"""Temporary TP2 model profile/capture using the user's unchanged external scripts."""

import argparse
from collections import Counter
import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import queue
import signal
import socket
import subprocess
import sys
import threading
import urllib.request

if __package__:
    from .plugin import build_target
else:
    from plugin import build_target

ROOT = Path(__file__).resolve().parents[5]
DATA = ROOT / "mytest/mydata"
LAUNCH = Path("/opt/sglang/scripts/launch_qwen38_flash_next_fp8_mi308x_pure_tp_4_or_8_or_2.sh")
WARMUP = Path("/opt/evaluation7/check_acc_long_oai.py")
PROFILE = Path("/opt/evaluation7/run_pure_text_profile.sh")
MODEL = Path("/models/Qwen3.8-Flash-Next-PTPC-FP8")
URL = "http://127.0.0.1:9080"


def save(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sources():
    git = ["git", "-C", "/opt/sglang"]
    output = lambda args: subprocess.check_output(git + args, text=True)
    paths = set(output(["diff", "--name-only", "HEAD"]).splitlines())
    paths.update(output(["ls-files", "--others", "--exclude-standard"]).splitlines())
    return {"sglang_head": output(["rev-parse", "HEAD"]).strip(),
            "sglang_status": output(["status", "--porcelain=v1"]),
            "sglang_dirty_sha256": {p: digest(Path("/opt/sglang") / p) for p in paths if (Path("/opt/sglang") / p).is_file()},
            "input_sha256": {str(p): digest(p) for p in (LAUNCH, WARMUP, PROFILE, MODEL / "config.json", MODEL / "tokenizer_config.json")}}


def snapshot(output, phase, *, free):
    smi = [sys.executable, "/opt/rocm-7.14/bin/amd-smi"]
    query = lambda args: json.loads(subprocess.check_output(smi + args + ["--json"], text=True))
    devices = query(["list"])
    limits = query(["static", "--limit", "--gpu", "0", "1"])
    processes = query(["process", "--gpu", "0", "1"])
    cards = {}
    for card in Path("/sys/class/drm").glob("card[0-9]*"):
        path = card / "device"
        if (path / "gpu_busy_percent").is_file():
            cards[path.resolve().name.lower()] = {key: int((path / key).read_text()) for key in (
                "gpu_busy_percent", "mem_info_vram_total", "mem_info_vram_used")}
    save(output / f"gpu_{phase}.json", {"devices": devices, "limits": limits, "processes": processes, "cards": cards})
    for row in limits["gpu_data"]:
        assert row["limit"]["ptl_state"] == "Enabled" and row["limit"]["ptl_format"] == "VECTOR,F8", row
    for device in devices:
        if device["gpu"] in (0, 1):
            state = cards[device["bdf"].lower()]
            assert state["gpu_busy_percent"] <= 5, (phase, state)
            if free:
                assert state["mem_info_vram_used"] <= 0.2 * state["mem_info_vram_total"], state


def request(endpoint, data=None):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    req = urllib.request.Request(URL + endpoint, data=None if data is None else json.dumps(data).encode(),
                                 headers={"Content-Type": "application/json"})
    with opener.open(req, timeout=600) as response:
        return json.load(response)


def stop(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=90)
    except subprocess.TimeoutExpired:
        pass
    # The group can outlive its leader; never leave model workers behind.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=30)


def run(command, path, env, timeout):
    with path.open("x") as log:
        process = subprocess.Popen(command, cwd=path.parent, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
            if code:
                raise subprocess.CalledProcessError(code, command)
        finally:
            stop(process)


def validate(output, baseline, dump_layers, dump_rows):
    traces = sorted((output / "profiles").rglob("*.trace.json.gz"))
    assert len(traces) == 2, traces
    results = sorted(output.glob("sglang_*.jsonl"))
    assert len(results) == 1, results
    rows = [json.loads(row) for row in results[0].read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["completed"] == 4 and rows[0]["total_output_tokens"] == 20, rows
    summaries = []
    for rank in (0, 1):
        path, = [p for p in traces if f"-TP-{rank}.trace.json.gz" in p.name]
        with gzip.open(path, "rt") as stream:
            events = json.load(stream)["traceEvents"]
        kernels = [e for e in events if e.get("cat") == "kernel" and e.get("ph") == "X"]
        assert kernels and any(e.get("cat") == "cpu_op" for e in events)
        durations, counts = Counter(), Counter()
        for event in kernels:
            durations[event["name"]] += event["dur"]
            counts[event["name"]] += 1
        report = None
        if not baseline:
            report = json.loads((output / "qsa" / f"qsa_tp{rank}.json").read_text())
            assert len(report["calls_per_layer"]) == 12 and set(report["calls_per_layer"].values()) == {4}, report
            assert len(report["validation"]) >= 24, report
            assert any("union_qsa_bf16_d256" in name for name in durations)
            expected = {(layer, size) for layer in dump_layers for size in dump_rows}
            actual = set()
            for name in report["input_snapshots"]:
                # The snapshot filename identifies layer/rows; the replay loader verifies tensor hashes.
                parts = name.split("_")
                assert (output / "inputs" / name).is_file() and parts[0] == f"tp{rank}"
                actual.add((int(parts[1][5:]), int(parts[2][1:])))
            assert actual == expected, (actual, expected)
        summaries.append({"rank": rank, "path": str(path.relative_to(output)), "sha256": digest(path),
                          "kernel_count": len(kernels), "kernel_sum_us": sum(durations.values()), "qsa": report,
                          "top_kernels": [{"name": name, "us": us, "count": counts[name]} for name, us in durations.most_common(20)]})
    save(output / "trace_validation.json", summaries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", action="store_true", help="disable the plugin; use the original model path")
    parser.add_argument("--dump-layers", type=int, nargs="+", default=[])
    parser.add_argument("--dump-rows", type=int, nargs="+", default=[11888, 12000])
    args = parser.parse_args()
    output = Path(os.path.abspath(args.output))
    if not output.resolve().is_relative_to(DATA.resolve()):
        raise ValueError("Use a new directory under mytest/mydata")
    if args.baseline and args.dump_layers:
        parser.error("Input capture requires the QSA plugin")
    if any(os.environ.get(k) for k in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "HSA_CU_MASK", "ROC_GLOBAL_CU_MASK")):
        raise RuntimeError("This TP2 launcher uses unmasked physical GPUs 0 and 1")
    output.mkdir(parents=True, exist_ok=False)
    for name in ("tmp", "profiles"):
        (output / name).mkdir()
    target = None if args.baseline else build_target(output / "plugin")
    cache = DATA / ".cache"
    configs = ROOT / "mytest/sglang_tp2_base_20260925_01/aiter_configs_snapshot"
    overrides = {
        "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1", "PYTHONPATH": "" if target is None else str(target),
        "TP_SIZE": "2", "HOST": "127.0.0.1", "PORT": "9080", "MODEL_PATH": str(MODEL), "BENCH_MODEL": str(MODEL),
        "SERVED_MODEL_NAME": "Qwen/Qwen3.8-Flash-Next-PTPC-FP8", "MEM_FRACTION_STATIC": "0.95",
        "CHUNKED_PREFILL_SIZE": "16384", "MAX_RUNNING_REQUESTS": "32", "CUDA_GRAPH_MAX_BS_DECODE": "32",
        "PLE_OFFLOAD_EMBEDDING": "0", "AITER_MOE_PADDING_SIZE": "64", "BENCH_HOST": "127.0.0.1", "BENCH_PORT": "9080",
        "INPUT_TOKENS": "12000", "OUTPUT_TOKENS": "5", "NUM_PROMPTS": "4", "MAX_CONCURRENCY": "1", "DATASET_NAME": "random",
        "LOG_FILE": str(output / "server.log"), "PROFILE_LOG_FILE": str(output / "profile.log"),
        "SGLANG_TORCH_PROFILER_DIR": str(output / "profiles"), "SGLANG_PROFILE_WITH_STACK": "1", "SGLANG_PROFILE_RECORD_SHAPES": "1",
        "PYHIP_QSA_PREFILL": "0" if args.baseline else "1", "PYHIP_QSA_VALIDATE": "1", "PYHIP_QSA_REPORT_DIR": str(output / "qsa"),
        "SGLANG_PLUGINS": "pyhip_no_plugins" if args.baseline else "pyhip_flydsl_qsa", "TMPDIR": str(output / "tmp"),
        "XDG_CACHE_HOME": str(cache), "SGLANG_CACHE_DIR": str(cache / "sglang"), "SGLANG_JIT_CACHE_DIR": str(cache / "sglang_jit"),
        "TORCHINDUCTOR_CACHE_DIR": str(cache / "sglang/inductor"), "TRITON_CACHE_DIR": str(cache / "triton"),
        "FLYDSL_RUNTIME_CACHE_DIR": str(cache / "flydsl"), "TORCH_EXTENSIONS_DIR": str(cache / "torch_extensions"), "AITER_JIT_DIR": str(cache / "aiter"),
    }
    for kind, filename in (("GEMM_A8W8_BPRESHUFFLE", "a8w8_bpreshuffle_tuned_gemm.csv"),
                           ("GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE", "a8w8_blockscale_bpreshuffle_tuned_gemm.csv"),
                           ("GEMM_BF16", "bf16_tuned_gemm.csv"), ("FMOE", "tuned_fmoe.csv")):
        assert (configs / filename).is_file()
        overrides[f"AITER_CONFIG_{kind}"] = str(configs / filename)
    if args.dump_layers:
        overrides.update(PYHIP_QSA_DUMP_LAYERS=",".join(map(str, args.dump_layers)),
                         PYHIP_QSA_DUMP_ROWS=",".join(map(str, args.dump_rows)), PYHIP_QSA_DUMP_DIR=str(output / "inputs"))
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYHIP_QSA_")}
    env.update(overrides)
    identity = sources()
    save(output / "metadata.json", {"sources": identity, "environment": overrides, "python": sys.executable,
         "packages": {n: importlib.metadata.version(n) for n in ("torch", "triton", "flydsl", "sglang")},
         "commands": {"capture": [sys.executable, *sys.argv], "launch": ["bash", str(LAUNCH)],
                      "warmup": [sys.executable, str(WARMUP)], "profile": ["bash", "-o", "pipefail", str(PROFILE)]},
         "extra_warmup_rows": [11888, 12000], "scope": "Profiler trace, not unprofiled throughput; input cloning perturbs traces when requested",
         "gpu_gate": "Idle/PTL before profile; free VRAM <=20% before launch/after teardown; model VRAM is allowed while serving",
         "profile_source_sha256": digest(Path(__file__)), "aiter_config_sha256": {p.name: digest(p) for p in configs.glob("*.csv")}})
    snapshot(output, "before_launch", free=True)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 9080))
    server = subprocess.Popen(["bash", str(LAUNCH)], cwd=output, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True)
    save(output / "server_process.json", {"pid": server.pid, "pgid": server.pid})
    events = queue.Queue()

    def drain():
        with (output / "launcher.log").open("x") as log:
            for line in server.stdout:
                log.write(line)
                if "The server is fired up and ready to roll!" in line:
                    events.put("ready")
        events.put("exited")

    reader = threading.Thread(target=drain, daemon=True)
    reader.start()
    status = {"complete": False, "capture_validated": False}
    try:
        if events.get(timeout=1800) != "ready":
            raise RuntimeError("Server exited before readiness")
        info = request("/server_info")
        save(output / "server_info.json", info)
        save(output / "model_info.json", request("/model_info"))
        assert info["tp_size"] == 2 and info["model_path"] == str(MODEL), info
        run([sys.executable, str(WARMUP)], output / "warmup.log", env, 600)
        warmup = (output / "warmup.log").read_text()
        assert "Answer:" in warmup and "Answer: None" not in warmup
        for rows in (11888, 12000):
            reply = request("/generate", {"input_ids": [42] * rows, "sampling_params": {"max_new_tokens": 1, "temperature": 0}})
            save(output / f"warmup_{rows}.json", reply)
            assert reply["meta_info"]["completion_tokens"] == 1, reply
        snapshot(output, "before_profile", free=False)
        run(["bash", "-o", "pipefail", str(PROFILE)], output / "profile_driver.log", env, 1800)
        snapshot(output, "after_profile", free=False)
        validate(output, args.baseline, args.dump_layers, args.dump_rows)
        status["capture_validated"] = True
    finally:
        stop(server)
        reader.join(timeout=10)
        server.stdout.close()
        status["server_returncode"] = server.returncode
        try:
            snapshot(output, "after_teardown", free=True)
            status["cleanup_verified"] = True
            final = sources()
            save(output / "sources_after.json", final)
            assert identity == final, "External sources changed during capture"
            status["external_sources_unchanged"] = True
            status["complete"] = status["capture_validated"]
        finally:
            save(output / "capture_status.json", status)
    print(f"PROFILE_COMPLETE={output}", flush=True)


if __name__ == "__main__":
    main()