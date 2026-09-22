# SPDX-License-Identifier: MIT
"""Full Graph latency using the H64 report's 100 weights / 2 passes / 3 replays protocol."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
from statistics import median
import subprocess
import time

if __package__:
    from .test_gr_read import assert_close, capture, make_inputs, parse_rows, reference, use_checkout_package
else:
    from test_gr_read import assert_close, capture, make_inputs, parse_rows, reference, use_checkout_package


def hardware_snapshot():
    physical = os.getenv("HIP_VISIBLE_DEVICES") or os.getenv("CUDA_VISIBLE_DEVICES") or "0"
    if not physical.isdigit() or os.getenv("ROCR_VISIBLE_DEVICES"):
        raise RuntimeError("select one physical GPU with HIP_VISIBLE_DEVICES")
    if os.getenv("CUDA_VISIBLE_DEVICES", physical) != physical:
        raise RuntimeError("HIP_VISIBLE_DEVICES and CUDA_VISIBLE_DEVICES must match")
    if os.getenv("HSA_CU_MASK") or os.getenv("ROC_GLOBAL_CU_MASK"):
        raise RuntimeError("benchmark requires an unmasked GPU")
    env = {k: v for k, v in os.environ.items() if k not in (
        "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL")}
    def query(command):
        return json.loads(subprocess.run(command, env=env, text=True, capture_output=True,
                                         check=True, timeout=30).stdout)
    card = query(["rocm-smi", "-d", physical, "--showuse", "--showmemuse", "--showbus", "--json"])[f"card{physical}"]
    limit = query(["amd-smi", "static", "-g", physical, "--limit", "--json"])["gpu_data"][0]["limit"]
    return {"physical_gpu": int(physical), "card": card, "limit": limit}


def check_hardware(snapshot, *, entry=False):
    card, limit = snapshot["card"], snapshot["limit"]
    use, memory = int(card["GPU use (%)"]), int(card["GPU Memory Allocated (VRAM%)"])
    if (entry and (use != 0 or memory != 0)) or use > 5 or memory > 20:
        raise RuntimeError(f"GPU occupied: {snapshot}; benchmark stopped")
    if limit.get("ptl_state") != "Enabled" or limit.get("ptl_format") != "VECTOR,F8":
        raise RuntimeError(f"expected PTL Enabled/VECTOR,F8: {snapshot}")


def time_graph(graph, calls, samples):
    # Same CUDA Event timer as be1555f support.time_graph; no Python launch timing.
    import torch
    for _ in range(3):
        graph.replay()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    values = []
    for _ in range(samples):
        start.record()
        for _ in range(3):
            graph.replay()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end) * 1000 / (calls * 3))
    return values


def benchmark_rows(rows, pairs, args, emit):
    import torch
    use_checkout_package()
    from pyhip.contrib.flydsl.gr_read import GRReadDecode, prepare_weights
    gen = torch.Generator(device="cuda").manual_seed(args.seed + rows)
    xs = [torch.randn(rows, 10240, device="cuda", dtype=torch.bfloat16, generator=gen) for _ in pairs]
    readers = [GRReadDecode(rows, *prepare_weights(wd, wu)) for wd, wu in pairs]
    emit({"type": "addresses", "rows": rows, "buffers": [
        {name: {"pointer": t.data_ptr(), "storage_offset": t.storage_offset(), "mod4096": t.data_ptr() % 4096}
         for name, t in zip(("X", "WD", "WU", "P", "Y"), (x, r.w_down, r.w_up, r.partial, r.output))}
        for x, r in zip(xs, readers)]})
    calls = [lambda x=x, r=r: r(x) for x, r in zip(xs, readers)]
    graph = capture(calls + calls)
    graph.replay()
    for x, r, (wd, wu) in zip(xs, readers, pairs):
        assert_close(r.output, reference(x, wd, wu), f"T={rows}: initial FP64")
    torch.cuda.synchronize()
    # Clear our own initialization/check work from the utilization window.
    time.sleep(2)
    snap = hardware_snapshot()
    emit({"type": "hardware", "phase": "before_samples", "rows": rows, **snap})
    check_hardware(snap)
    timings = []
    for round_index in range(args.rounds):
        values = time_graph(graph, len(pairs) * 2, args.samples)
        timings.extend(values)
        emit({"type": "samples", "rows": rows, "round": round_index, "us": values})
    for x in xs:
        x.mul_(0.99).add_(0.015625)
    graph.replay()
    for x, r, (wd, wu) in zip(xs, readers, pairs):
        assert_close(r.output, reference(x, wd, wu), f"T={rows}: changed-input FP64")
    latency = median(timings)
    result = {"type": "result", "rows": rows, "median_us": latency, "fp64_passed": True,
              "effective_tflops": 4 * rows * 10240 * 320 / (latency * 1e6)}
    emit(result)
    print(f"T={rows:2}  H64 final Graph: {latency:8.3f} us  {result['effective_tflops']:.3f} effective TFLOPS", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=parse_rows, nargs="+", default=list(range(1, 33)))
    parser.add_argument("--weights", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--seed", type=int, default=707)
    parser.add_argument("--output", type=Path, help="new JSONL file; defaults to results/graph_<timestamp>.jsonl")
    args = parser.parse_args()
    if min(args.weights, args.rounds, args.samples) < 1 or not __debug__:
        parser.error("counts must be positive; run without python -O")
    path = args.output or Path(__file__).with_name("results") / f"graph_{datetime.now(timezone.utc):%Y%m%dT%H%M%S%f}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as out:
        def emit(record):
            out.write(json.dumps(record) + "\n")
            out.flush()
        snap = hardware_snapshot()
        emit({"type": "hardware", "phase": "entry", **snap})
        check_hardware(snap, entry=True)
        import torch
        props = torch.cuda.get_device_properties(0)
        emit({"type": "environment", "gpu": props.name, "arch": props.gcnArchName,
              "compute_units": props.multi_processor_count, "torch": torch.__version__, "hip": torch.version.hip,
              "flydsl": importlib.metadata.version("flydsl"), "weight_source": "synthetic",
              "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "sources": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (
                  *Path(__file__).parent.glob("*.py"),
                  *(Path(__file__).resolve().parents[3] / "src/contrib/flydsl/gr_read").glob("*.py"))}})
        with torch.inference_mode():
            pairs = [make_inputs(1, args.seed + i)[1:] for i in range(args.weights)]
            results = []
            for rows in args.rows:
                results.append(benchmark_rows(rows, pairs, args, emit))
        torch.cuda.synchronize()
        time.sleep(2)
        snap = hardware_snapshot()
        emit({"type": "hardware", "phase": "exit", **snap})
        check_hardware(snap)
        emit({"type": "summary", "completed": True, "rows": args.rows})
    print("\n| T | Decode Graph us | Effective TFLOPS |")
    print("|---:|---:|---:|")
    for result in results:
        print(f"| {result['rows']} | {result['median_us']:.3f} | {result['effective_tflops']:.3f} |")
    print(f"Raw samples: {path}")


if __name__ == "__main__":
    main()
