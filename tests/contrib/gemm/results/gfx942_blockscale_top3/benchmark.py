"""Same-process palindromic A/B timing of saved gemm_fp8_gfx942 code objects."""
import argparse
import glob
import hashlib
import importlib.util
import json
import os
import statistics
import threading
import time
from pathlib import Path

import amdsmi
import torch

import pyhip
from pyhip.core.hiptools import _build_hip_func

TEST = str(Path(__file__).resolve().parents[2] / "test_a8w8_f32_blockscale.py")
SYMBOL = "gemm_a8w8_blockscale_8wave_gfx942"
HWMON = Path("/sys/bus/pci/devices/0000:0a:00.0/hwmon/hwmon0")

parser = argparse.ArgumentParser()
parser.add_argument("--versions", required=True, help="name=cache_dir:jit_line,...")
parser.add_argument("--shape", required=True)
parser.add_argument("--preshuffle", type=int, default=0)
parser.add_argument("--rounds", type=int, default=16)
parser.add_argument("--output", required=True)
args = parser.parse_args()
m, n, k = (int(v) for v in args.shape.split(","))
preshuffle = bool(args.preshuffle)

amdsmi.amdsmi_init()
handle = amdsmi.amdsmi_get_processor_handles()[0]
gates = []


def gate(stage):
    memory = amdsmi.amdsmi_get_gpu_vram_usage(handle)
    record = {
        "stage": stage, "time_ns": time.monotonic_ns(),
        "activity": amdsmi.amdsmi_get_gpu_activity(handle)["gfx_activity"],
        "vram_used_mb": memory["vram_used"],
        "ptl": amdsmi.amdsmi_get_gpu_ptl_state(handle),
        "formats": [amdsmi.AmdSmiPtlData(v).name for v in amdsmi.amdsmi_get_gpu_ptl_formats(handle)],
        "pids": len(amdsmi.amdsmi_get_gpu_process_list(handle)),
    }
    gates.append(record)
    print("GATE", json.dumps(record), flush=True)
    assert record["ptl"] and set(record["formats"]) == {"VECTOR", "F8"}, record
    assert record["vram_used_mb"] <= 0.2 * memory["vram_total"], record
    if stage == "entry":
        assert record["activity"] <= 5 and record["pids"] == 0, record
    else:
        # amdsmi reports host PIDs; entry had none, so a single PID is this process.
        assert record["pids"] <= 1, record


gate("entry")
spec = importlib.util.spec_from_file_location("gemm_test", TEST)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

torch.manual_seed(0)
x = (torch.rand((m, k), device="cuda") / 10).to(module.dtypes.fp8)
weight = (torch.rand((n, k), device="cuda") / 10).to(module.dtypes.fp8)
scale_a = torch.rand((m, k // 128), device="cuda")
scale_b = torch.rand((pyhip.div_up(n, 128), k // 128), device="cuda")
reference = module.run_torch(x, weight, scale_a, scale_b).cpu()
scale_a = scale_a.T.contiguous()
weight = module.shuffle_weight(weight, layout=(16, 16)) if preshuffle else weight
buffers = [tuple(t.clone() for t in (x, weight, scale_a, scale_b)) for _ in range(32)]
output = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
grid = [pyhip.div_up(m, 256) * pyhip.div_up(n, 256)]

kernels = {}
for item in args.versions.split(","):
    name, location = item.split("=")
    cache, line = location.split(":")
    pattern = f"{cache}/{SYMBOL}-1-preshuffle_b={preshuffle}-N={n}-K={k}-*gfx942.py-{line}.co"
    matches = glob.glob(pattern)
    assert len(matches) == 1, (pattern, matches)
    kernels[name] = (_build_hip_func(matches[0], "", None, SYMBOL), matches[0])


def launch(kernel, buffer):
    activation, weights, activation_scale, weight_scale = buffer
    kernel(list(grid), [512], activation, weights, output, activation_scale, weight_scale, m)


def check(name, candidate=None):
    with torch.device("cpu"):
        if candidate is None:
            candidate = output.cpu()
        assert module.checkAllclose(reference, candidate, msg=name, tol_err_ratio=0) == 0
        return pyhip.calc_diff(reference, candidate, diff_thr=1e-5)


diffs = {}
for name, (kernel, _) in kernels.items():
    output.fill_(float("nan"))
    for buffer in buffers:
        launch(kernel, buffer)
    torch.cuda.synchronize()
    diffs[name] = check(name)

telemetry = []
stop = threading.Event()


def sample():
    while not stop.wait(0.005):
        telemetry.append((time.monotonic_ns(),
                          int((HWMON / "freq1_input").read_text()) / 1e6,
                          int((HWMON / "power1_input").read_text()) / 1e6))


thread = threading.Thread(target=sample, daemon=True)
thread.start()
gate("before_samples")
names = list(kernels)
order = names + names[::-1]
flops = 2 * m * n * k
samples = []
timed_outputs = {}
for round_index in range(args.rounds):
    for position, name in enumerate(order):
        buffer_index = (round_index * 2 + position // len(names)) % len(buffers)
        begin_ns = time.monotonic_ns()
        with pyhip.cudaPerf(flops, 0, name=name, verbose=0) as perf:
            launch(kernels[name][0], buffers[buffer_index])
        samples.append({"round": round_index, "version": name, "buffer": buffer_index,
                        "begin_ns": begin_ns, "end_ns": time.monotonic_ns(),
                        "us": perf.latencies[0] * 1e6})
        if round_index == args.rounds - 1 and position >= len(names):
            timed_outputs[name] = output.cpu()
print("RAW_SAMPLES", json.dumps(samples), flush=True)
gate("after_samples")
stop.set()
thread.join()
for name, candidate in timed_outputs.items():
    assert check(name, candidate) == diffs[name]

summary = {}
for name in names:
    us = [s["us"] for s in samples if s["version"] == name]
    summary[name] = {"n": len(us), "median_us": statistics.median(us), "min_us": min(us),
                     "max_us": max(us), "median_tflops": flops / (statistics.median(us) * 1e6),
                     "diff": diffs[name]}
base = names[0]
paired = {}
for name in names[1:]:
    ratios = []
    for r in range(args.rounds):
        a = [s["us"] for s in samples if s["round"] == r and s["version"] == base]
        b = [s["us"] for s in samples if s["round"] == r and s["version"] == name]
        ratios.append(statistics.mean(b) / statistics.mean(a))
    paired[name] = {"median_ratio_vs_" + base: statistics.median(ratios), "ratios": ratios}
window = [t for t in telemetry if gates[-2]["time_ns"] <= t[0] <= gates[-1]["time_ns"]]
Path(args.output).write_text(json.dumps({
    "shape": [m, n, k], "preshuffle": preshuffle, "order": order,
    "input_pointers": [[tensor.data_ptr() for tensor in group] for group in buffers],
    "output_pointer": output.data_ptr(),
    "code_objects": {name: [path, hashlib.sha256(Path(path).read_bytes()).hexdigest()]
                     for name, (_, path) in kernels.items()},
    "gates": gates, "samples": samples, "summary": summary, "paired": paired,
    "telemetry": telemetry,
    "window_sclk_median_mhz": statistics.median(t[1] for t in window) if window else None,
}, indent=1))
for name in names:
    s = summary[name]
    extra = f" ratio={paired[name]['median_ratio_vs_' + base]:.4f}" if name in paired else ""
    print(f"ABBA {m}x{n}x{k} pre={int(preshuffle)} {name}: median={s['median_us']:.1f}us "
          f"{s['median_tflops']:.1f}TF [{s['min_us']:.1f},{s['max_us']:.1f}] diff={s['diff']:.3e}{extra}")
print("window_sclk_median_mhz", statistics.median(t[1] for t in window) if window else None)
amdsmi.amdsmi_shut_down()
