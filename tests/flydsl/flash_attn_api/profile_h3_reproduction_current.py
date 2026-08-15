"""Profile current H3 BF16 attention paths with report-compatible telemetry.

The configuration follows H3_REPRODUCTION_COMPARISON_REPORT.md: physical GPU 4,
auto DPM, segments (63225, 7), BF16, H=14, D=128, seed 1101, three warmups,
70 CUDA-event-timed dispatches, and 1 ms SCLK/power/temperature sampling.

Run AITER MI308 and MI300 in separate processes because loaded code objects are
cached by kernel name for the life of a process.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

os.environ.setdefault("AITER_AOT_IMPORT", "1")
os.environ.setdefault("FLYDSL_RUNTIME_ENABLE_CACHE", "0")

import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))

H3_SEGMENTS = (63225, 7)
H3_HEADS = 14
H3_HEAD_DIM = 128
H3_TOTAL_TOKENS = sum(H3_SEGMENTS)
H3_SCALE = H3_HEAD_DIM**-0.5
H3_FLOPS = sum(
    4 * length * length * H3_HEADS * H3_HEAD_DIM
    for length in H3_SEGMENTS
)
ASM_FILENAME = "fwd_hd128_bf16_rtna_group.co"
THROTTLE_COUNTERS = (
    "accumulation_counter",
    "ppt_accumulated",
    "prochot_accumulated",
    "socket_thermal_accumulated",
    "vr_thermal_accumulated",
    "hbm_thermal_accumulated",
)


def run_command(command):
    try:
        result = subprocess.run(
            command, check=True, capture_output=True, text=True
        )
        return {
            "command": command,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    except (OSError, subprocess.CalledProcessError) as error:
        return {"command": command, "error": str(error)}


def run_json_command(command):
    result = run_command(command)
    if "stdout" in result:
        raw = result.pop("stdout")
        try:
            result["json"] = json.loads(raw)
        except json.JSONDecodeError as error:
            result["json_error"] = str(error)
            result["stdout"] = raw
    return result


def read_text(path):
    return Path(path).read_text().strip()


def read_scaled_number(path, divisor):
    try:
        return int(Path(path).read_text()) / divisor
    except (OSError, ValueError):
        return None


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        while chunk := file.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def physical_gpu_index():
    visible = os.environ.get("HIP_VISIBLE_DEVICES") or os.environ.get(
        "CUDA_VISIBLE_DEVICES"
    )
    if not visible or len(visible.split(",")) != 1:
        raise RuntimeError("set HIP_VISIBLE_DEVICES to one physical GPU")
    return int(visible)


def gpu_bdf(physical_gpu):
    result = subprocess.run(
        ["amd-smi", "list", "--json"],
        check=True,
        capture_output=True,
        text=True,
    )
    devices = json.loads(result.stdout)
    return next(
        device["bdf"] for device in devices if device["gpu"] == physical_gpu
    )


def gpu_runtime_state(physical_gpu, bdf):
    pci_path = Path("/sys/bus/pci/devices") / bdf
    process_result = run_json_command(
        ["amd-smi", "process", "-g", str(physical_gpu), "-G", "--json"]
    )
    static_result = run_json_command(
        ["amd-smi", "static", "-g", str(physical_gpu), "--json"]
    )
    process_list = process_result["json"][0]["process_list"]
    ptl = static_result["json"]["gpu_data"][0]["limit"]
    running_processes = [
        entry["process_info"]
        for entry in process_list
        if isinstance(entry.get("process_info"), dict)
    ]
    return {
        "gpu_busy_percent": int(read_text(pci_path / "gpu_busy_percent")),
        "vram_used_bytes": int(read_text(pci_path / "mem_info_vram_used")),
        "vram_total_bytes": int(read_text(pci_path / "mem_info_vram_total")),
        "dpm_force_performance_level": read_text(
            pci_path / "power_dpm_force_performance_level"
        ),
        "ptl_state": ptl["ptl_state"],
        "ptl_format": ptl["ptl_format"],
        "running_processes": running_processes,
        "process_query": process_result,
    }


def require_idle_gpu(physical_gpu, bdf):
    state = gpu_runtime_state(physical_gpu, bdf)
    expected_ptl_format = os.environ.get("ATTN_PROFILE_EXPECT_PTL_FORMAT")
    max_vram_bytes = int(
        os.environ.get("ATTN_PROFILE_MAX_INITIAL_VRAM_MIB", "1024")
    ) * 1024 * 1024
    allow_resident = (
        os.environ.get("ATTN_PROFILE_ALLOW_RESIDENT_PROCESSES", "0") == "1"
    )
    resident_processes_are_idle = all(
        process.get("cu_occupancy") == 0
        for process in state["running_processes"]
    )
    errors = []
    if state["running_processes"] and not (
        allow_resident and resident_processes_are_idle
    ):
        errors.append(f"running GPU processes: {state['running_processes']}")
    if state["gpu_busy_percent"] != 0:
        errors.append(f"gpu_busy_percent={state['gpu_busy_percent']}")
    if state["vram_used_bytes"] > max_vram_bytes:
        errors.append(
            f"initial VRAM={state['vram_used_bytes'] / 2**20:.1f} MiB"
        )
    if state["dpm_force_performance_level"] != "auto":
        errors.append(
            f"DPM={state['dpm_force_performance_level']!r}, expected 'auto'"
        )
    if expected_ptl_format and state["ptl_state"] != "Enabled":
        errors.append(f"PTL={state['ptl_state']!r}, expected 'Enabled'")
    if expected_ptl_format and state["ptl_format"] != expected_ptl_format:
        errors.append(
            f"PTL format={state['ptl_format']!r}, "
            f"expected {expected_ptl_format!r}"
        )
    if errors:
        raise RuntimeError("GPU preflight failed: " + "; ".join(errors))
    state["resident_process_exception"] = {
        "enabled": allow_resident,
        "all_reported_cu_occupancy_zero": resident_processes_are_idle,
        "process_count": len(state["running_processes"]),
        "max_initial_vram_mib": max_vram_bytes // 1024 // 1024,
    }
    return state


def throttle_counters(physical_gpu):
    result = run_json_command(
        ["amd-smi", "metric", "-g", str(physical_gpu), "-v", "--json"]
    )
    return result["json"]["gpu_data"][0]["throttle"]


def labeled_sensor(hwmon, family, labels):
    labels = {label.lower() for label in labels}
    for label_path in sorted(hwmon.glob(f"{family}*_label")):
        label = label_path.read_text().strip()
        if label.lower() in labels:
            input_path = label_path.with_name(
                label_path.name.replace("_label", "_input")
            )
            if input_path.is_file():
                return input_path, label
    raise RuntimeError(
        f"cannot find {family} sensor {sorted(labels)} under {hwmon}"
    )


class SensorSampler:
    def __init__(self, bdf, interval):
        hwmon_dirs = list(
            (Path("/sys/bus/pci/devices") / bdf / "hwmon").glob("hwmon*")
        )
        if len(hwmon_dirs) != 1:
            raise RuntimeError(f"expected one hwmon for {bdf}: {hwmon_dirs}")
        hwmon = hwmon_dirs[0]
        sclk_path, sclk_label = labeled_sensor(
            hwmon, "freq", {"sclk", "gfxclk"}
        )
        power_path, power_label = labeled_sensor(
            hwmon, "power", {"ppt", "socket power"}
        )
        junction_path, junction_label = labeled_sensor(
            hwmon, "temp", {"junction", "hotspot"}
        )
        mem_path, mem_label = labeled_sensor(
            hwmon, "temp", {"mem", "memory", "hbm"}
        )
        self.paths = {
            "sclk_mhz": (sclk_path, 1e6),
            "power_w": (power_path, 1e6),
            "junction_c": (junction_path, 1e3),
            "mem_c": (mem_path, 1e3),
        }
        self.metadata = {
            "hwmon": str(hwmon),
            "interval_seconds": interval,
            "power_cap_w": read_scaled_number(
                power_path.with_name(
                    power_path.name.replace("_input", "_cap")
                ),
                1e6,
            ),
            "sensors": {
                "sclk_mhz": {"path": str(sclk_path), "label": sclk_label},
                "power_w": {"path": str(power_path), "label": power_label},
                "junction_c": {
                    "path": str(junction_path),
                    "label": junction_label,
                },
                "mem_c": {"path": str(mem_path), "label": mem_label},
            },
        }
        self.interval = interval
        self.samples = []
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.thread = None

    def _sample_once(self):
        row = {"time": time.perf_counter()}
        for name, (path, divisor) in self.paths.items():
            row[name] = int(path.read_text()) / divisor
        with self.lock:
            self.samples.append(row)

    def _run(self):
        while not self.stop_event.is_set():
            self._sample_once()
            self.stop_event.wait(self.interval)

    def start(self):
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join()

    def between(self, begin, end):
        with self.lock:
            return [
                sample
                for sample in self.samples
                if begin <= sample["time"] <= end
            ]


def summarize_sensors(samples):
    if not samples:
        raise RuntimeError("dispatch has no sensor samples")
    return {
        "sensor_count": len(samples),
        "sclk_mean_mhz": statistics.mean(
            sample["sclk_mhz"] for sample in samples
        ),
        "sclk_min_mhz": min(sample["sclk_mhz"] for sample in samples),
        "sclk_max_mhz": max(sample["sclk_mhz"] for sample in samples),
        "power_mean_w": statistics.mean(
            sample["power_w"] for sample in samples
        ),
        "power_max_w": max(sample["power_w"] for sample in samples),
        "junction_max_c": max(sample["junction_c"] for sample in samples),
        "mem_max_c": max(sample["mem_c"] for sample in samples),
    }


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def correlation(lhs, rhs):
    lhs_mean = statistics.mean(lhs)
    rhs_mean = statistics.mean(rhs)
    numerator = sum(
        (x - lhs_mean) * (y - rhs_mean) for x, y in zip(lhs, rhs)
    )
    denominator = math.sqrt(
        sum((x - lhs_mean) ** 2 for x in lhs)
        * sum((y - rhs_mean) ** 2 for y in rhs)
    )
    return numerator / denominator if denominator else None


def high_runs(values, threshold):
    runs = []
    for index, value in enumerate(values):
        if value < threshold:
            continue
        if not runs or index != runs[-1][-1] + 1:
            runs.append([index])
        else:
            runs[-1].append(index)
    return runs


def analyze_rows(rows):
    elapsed = [row["elapsed_ms"] for row in rows]
    tflops = [row["h3_tflops"] for row in rows]
    sclk = [row["sclk_mean_mhz"] for row in rows]
    steady = rows[2:] if len(rows) > 2 else rows
    high_threshold = min(tflops) + (max(tflops) - min(tflops)) * 0.65
    runs = high_runs(tflops, high_threshold)
    starts = [run[0] for run in runs]
    dispatch_intervals = [
        end - begin for begin, end in zip(starts, starts[1:])
    ]
    time_intervals = [
        rows[end]["wall_start_seconds"] - rows[begin]["wall_start_seconds"]
        for begin, end in zip(starts, starts[1:])
    ]
    cycle_detected = (
        len(starts) >= 4
        and bool(dispatch_intervals)
        and max(dispatch_intervals) - min(dispatch_intervals) <= 2
    )
    return {
        "sample_count": len(rows),
        "mean_elapsed_ms": statistics.mean(elapsed),
        "median_elapsed_ms": statistics.median(elapsed),
        "min_elapsed_ms": min(elapsed),
        "max_elapsed_ms": max(elapsed),
        "mean_h3_tflops": statistics.mean(tflops),
        "median_h3_tflops": statistics.median(tflops),
        "tflops_cv_percent": statistics.pstdev(tflops)
        / statistics.mean(tflops)
        * 100.0,
        "observed_sclk_min_mhz": min(row["sclk_min_mhz"] for row in rows),
        "observed_sclk_max_mhz": max(row["sclk_max_mhz"] for row in rows),
        "steady_observed_sclk_min_mhz": min(
            row["sclk_min_mhz"] for row in steady
        ),
        "steady_observed_sclk_max_mhz": max(
            row["sclk_max_mhz"] for row in steady
        ),
        "steady_dispatch_mean_sclk_p05_mhz": percentile(
            [row["sclk_mean_mhz"] for row in steady], 0.05
        ),
        "steady_dispatch_mean_sclk_median_mhz": percentile(
            [row["sclk_mean_mhz"] for row in steady], 0.5
        ),
        "steady_dispatch_mean_sclk_p95_mhz": percentile(
            [row["sclk_mean_mhz"] for row in steady], 0.95
        ),
        "steady_dispatch_mean_sclk_stddev_mhz": statistics.pstdev(
            [row["sclk_mean_mhz"] for row in steady]
        ),
        "sensor_sample_count": sum(row["sensor_count"] for row in rows),
        "mean_power_w": statistics.mean(
            row["power_mean_w"] for row in rows
        ),
        "correlation_tflops_sclk": correlation(tflops, sclk),
        "peak_to_floor_drop_percent": (1.0 - min(tflops) / max(tflops))
        * 100.0,
        "high_threshold_tflops": high_threshold,
        "high_runs": [[run[0], run[-1]] for run in runs],
        "dispatch_intervals": dispatch_intervals,
        "time_intervals_seconds": time_intervals,
        "cycle_detected": cycle_detected,
    }


def accuracy(reference, candidate):
    reference_f32 = reference.float()
    candidate_f32 = candidate.float()
    difference = candidate_f32 - reference_f32
    return {
        "cosine": torch.nn.functional.cosine_similarity(
            candidate_f32.flatten(), reference_f32.flatten(), dim=0
        ).item(),
        "relative_l2": (
            torch.linalg.vector_norm(difference)
            / torch.linalg.vector_norm(reference_f32).clamp_min(1e-12)
        ).item(),
        "max_abs": difference.abs().max().item(),
        "tail_cosine": torch.nn.functional.cosine_similarity(
            candidate_f32[-H3_SEGMENTS[-1] :].flatten(),
            reference_f32[-H3_SEGMENTS[-1] :].flatten(),
            dim=0,
        ).item(),
        "finite": bool(torch.isfinite(candidate_f32).all()),
    }


def make_linear_inputs():
    generator = torch.Generator(device="cuda").manual_seed(1101)
    shape = (H3_TOTAL_TOKENS, H3_HEADS, H3_HEAD_DIM)
    q, k, v = (
        torch.randn(
            shape,
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        for _ in range(3)
    )
    cu = torch.tensor(
        [0, *torch.tensor(H3_SEGMENTS).cumsum(0).tolist()],
        device="cuda",
        dtype=torch.int32,
    )
    return q, k, v, cu


def segment_reference(q, k, v):
    output = torch.empty_like(q)
    start = 0
    for length in H3_SEGMENTS:
        stop = start + length
        output[start:stop] = torch.nn.functional.scaled_dot_product_attention(
            q[start:stop].transpose(0, 1).unsqueeze(0),
            k[start:stop].transpose(0, 1).unsqueeze(0),
            v[start:stop].transpose(0, 1).unsqueeze(0),
            scale=H3_SCALE,
        ).squeeze(0).transpose(0, 1)
        start = stop
    return output


def prepare_flydsl_launchers(selected):
    from flash_attn_varlen_8wave import flash_attn_varlen_func

    q, k, v, cu = make_linear_inputs()
    reference = segment_reference(q, k, v)
    launchers = {}
    for name, num_waves in (
        ("new_flydsl_4wave", 4),
        ("new_flydsl_8wave", 8),
    ):
        if name not in selected:
            continue
        output = torch.empty_like(q)
        key_layout = "vectorized" if num_waves == 4 else "linear"

        def launch(
            output=output,
            num_waves=num_waves,
            key_layout=key_layout,
        ):
            return flash_attn_varlen_func(
                q,
                k,
                v,
                cu,
                cu,
                max(H3_SEGMENTS),
                max(H3_SEGMENTS),
                softmax_scale=H3_SCALE,
                out=output,
                key_layout=key_layout,
                num_waves=num_waves,
            )

        launch()
        torch.cuda.synchronize()
        launchers[name] = {
            "launch": launch,
            "native_flops": H3_FLOPS,
            "accuracy": accuracy(reference, output),
        }
    return launchers, []


def install_asm_override(variant):
    aiter_root = Path("/app/aiter")
    source = (
        aiter_root / "hsa/gfx942/fmha_v3_fwd" / variant / ASM_FILENAME
    )
    if not source.is_file():
        raise FileNotFoundError(source)
    temporary = tempfile.TemporaryDirectory(prefix=f"aiter-{variant.lower()}-")
    selected_dir = Path(temporary.name) / "gfx942/fmha_v3_fwd/MI308"
    selected_dir.mkdir(parents=True)
    selected_path = selected_dir / ASM_FILENAME
    selected_path.symlink_to(source)
    if selected_path.resolve() != source.resolve():
        raise RuntimeError("ASM override did not resolve to requested source")
    os.environ["AITER_ASM_DIR"] = temporary.name
    return temporary, {
        "variant": variant,
        "selected_path": str(selected_path),
        "resolved_path": str(selected_path.resolve()),
        "sha256": file_sha256(source),
    }


def prepare_aiter_launchers(selected):
    q, k, v, cu = make_linear_inputs()
    reference = segment_reference(q, k, v)
    launchers = {}
    resources = []

    if "triton" in selected:
        from aiter.ops.triton.attention.mha import flash_attn_varlen_func

        output = None

        def launch_triton():
            nonlocal output
            result = flash_attn_varlen_func(
                q,
                k,
                v,
                cu,
                cu,
                max(H3_SEGMENTS),
                max(H3_SEGMENTS),
                dropout_p=0.0,
                softmax_scale=H3_SCALE,
                causal=False,
            )
            output = result[0] if isinstance(result, (tuple, list)) else result
            return output

        launch_triton()
        torch.cuda.synchronize()
        launchers["triton"] = {
            "launch": launch_triton,
            "native_flops": H3_FLOPS,
            "accuracy": accuracy(reference, output),
        }

    asm_names = [name for name in selected if name.startswith("asm_")]
    if len(asm_names) > 1:
        raise RuntimeError("profile one ASM variant per process")
    if asm_names:
        from aiter.ops.mha import fmha_v3_varlen_fwd

        variant = "MI300" if asm_names[0] == "asm_mi300" else "MI308"
        temporary, metadata = install_asm_override(variant)
        resources.append(temporary)
        output = torch.empty_like(q)

        def launch_asm():
            return fmha_v3_varlen_fwd(
                q,
                k,
                v,
                cu,
                cu,
                max(H3_SEGMENTS),
                max(H3_SEGMENTS),
                0,
                0.0,
                H3_SCALE,
                0.0,
                False,
                False,
                -1,
                -1,
                False,
                False,
                1,
                output,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )[0]

        launch_asm()
        torch.cuda.synchronize()
        launchers[asm_names[0]] = {
            "launch": launch_asm,
            "native_flops": H3_FLOPS,
            "accuracy": accuracy(reference, output),
            "asm": metadata,
        }
    return launchers, resources


def profile_dispatches(
    name,
    launch,
    native_flops,
    physical_gpu,
    bdf,
    warmup,
    iters,
    sensor_interval,
):
    for _ in range(warmup):
        launch()
    torch.cuda.synchronize()

    before = throttle_counters(physical_gpu)
    sampler = SensorSampler(bdf, sensor_interval)
    sampler.start()
    section_start = time.perf_counter()
    dispatches = []
    print(
        "sample,impl,index,elapsed_ms,native_tflops,h3_tflops,"
        "sclk_mean_mhz,sclk_min_mhz,sclk_max_mhz,power_mean_w,"
        "power_max_w,junction_max_c,mem_max_c,sensor_count",
        flush=True,
    )
    for index in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        wall_start = time.perf_counter()
        start.record()
        launch()
        stop.record()
        stop.synchronize()
        wall_stop = time.perf_counter()
        elapsed_ms = start.elapsed_time(stop)
        sensors = summarize_sensors(sampler.between(wall_start, wall_stop))
        row = {
            "impl": name,
            "index": index,
            "wall_start_seconds": wall_start - section_start,
            "wall_elapsed_ms": (wall_stop - wall_start) * 1e3,
            "elapsed_ms": elapsed_ms,
            "native_tflops": native_flops / (elapsed_ms * 1e9),
            "h3_tflops": H3_FLOPS / (elapsed_ms * 1e9),
            **sensors,
        }
        dispatches.append(row)
        print(
            f"sample,{name},{index},{elapsed_ms:.3f},"
            f"{row['native_tflops']:.3f},{row['h3_tflops']:.3f},"
            f"{row['sclk_mean_mhz']:.1f},{row['sclk_min_mhz']:.1f},"
            f"{row['sclk_max_mhz']:.1f},{row['power_mean_w']:.1f},"
            f"{row['power_max_w']:.1f},{row['junction_max_c']:.1f},"
            f"{row['mem_max_c']:.1f},{row['sensor_count']}",
            flush=True,
        )
    sampler.stop()
    after = throttle_counters(physical_gpu)
    throttle_delta = {
        key: (
            after.get(key) - before.get(key)
            if isinstance(after.get(key), (int, float))
            and isinstance(before.get(key), (int, float))
            else None
        )
        for key in THROTTLE_COUNTERS
    }
    return {
        "name": name,
        "dispatches": dispatches,
        "analysis": analyze_rows(dispatches),
        "throttle_before": before,
        "throttle_after": after,
        "throttle_delta": throttle_delta,
        "sensor_metadata": sampler.metadata,
    }


def main():
    selected = tuple(
        name.strip()
        for name in os.environ.get(
            "ATTN_PROFILE_IMPLS", "new_flydsl_8wave"
        ).split(",")
        if name.strip()
    )
    known = {
        "new_flydsl_4wave",
        "new_flydsl_8wave",
        "triton",
        "asm_mi308",
        "asm_mi300",
    }
    unknown = set(selected) - known
    if unknown:
        raise ValueError(f"unknown implementations: {sorted(unknown)}")

    physical_gpu = physical_gpu_index()
    bdf = gpu_bdf(physical_gpu)
    preflight = require_idle_gpu(physical_gpu, bdf)
    warmup = int(os.environ.get("ATTN_PROFILE_WARMUP", "3"))
    iters = int(os.environ.get("ATTN_PROFILE_ITERS", "70"))
    sensor_interval_ms = float(
        os.environ.get("ATTN_PROFILE_SENSOR_INTERVAL_MS", "1")
    )
    output_path = Path(
        os.environ.get(
            "ATTN_PROFILE_OUTPUT", "/tmp/h3-reproduction-current.json"
        )
    )

    torch.set_grad_enabled(False)
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    if "gfx942" not in properties.gcnArchName or properties.multi_processor_count != 80:
        raise RuntimeError(
            f"expected 80-CU gfx942, got {properties.gcnArchName} "
            f"with {properties.multi_processor_count} CUs"
        )

    flydsl_names = {"new_flydsl_4wave", "new_flydsl_8wave"}
    if any(name in flydsl_names for name in selected):
        launchers, resources = prepare_flydsl_launchers(selected)
    else:
        launchers, resources = prepare_aiter_launchers(selected)

    for name in selected:
        result_accuracy = launchers[name]["accuracy"]
        print(f"accuracy,{name},{json.dumps(result_accuracy, sort_keys=True)}")
        if not result_accuracy["finite"] or result_accuracy["relative_l2"] >= 0.005:
            raise AssertionError(f"{name} correctness failed: {result_accuracy}")

    results = []
    for name in selected:
        profile = profile_dispatches(
            name,
            launchers[name]["launch"],
            launchers[name]["native_flops"],
            physical_gpu,
            bdf,
            warmup,
            iters,
            sensor_interval_ms / 1e3,
        )
        profile["accuracy"] = launchers[name]["accuracy"]
        if "asm" in launchers[name]:
            profile["asm"] = launchers[name]["asm"]
        results.append(profile)
        print(f"summary,{name},{json.dumps(profile['analysis'], sort_keys=True)}")

    runtime_after_profile = gpu_runtime_state(physical_gpu, bdf)

    output = {
        "schema_version": 2,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": {
            "segments": H3_SEGMENTS,
            "heads": H3_HEADS,
            "head_dim": H3_HEAD_DIM,
            "seed": 1101,
            "h3_flops": H3_FLOPS,
            "warmup": warmup,
            "iters": iters,
            "sensor_interval_ms": sensor_interval_ms,
            "implementations": selected,
        },
        "environment": {
            "hostname": platform.node(),
            "python": sys.version,
            "torch": torch.__version__,
            "torch_hip": torch.version.hip,
            "flydsl": package_version("flydsl"),
            "physical_gpu": physical_gpu,
            "bdf": bdf,
            "gcn_arch": properties.gcnArchName,
            "cu_count": properties.multi_processor_count,
            "preflight": preflight,
            "runtime_after_profile": runtime_after_profile,
            "sensor_metadata": results[0]["sensor_metadata"],
        },
        "results": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2))
    print(f"profile_output={output_path}")

    torch.cuda.synchronize()
    for resource in resources:
        resource.cleanup()


if __name__ == "__main__":
    main()