"""Accuracy, eager timing, and optional CUDA graph timing for depthwise Conv3D."""

import argparse
import json
import math
import statistics
import sys
import time
from contextlib import nullcontext, redirect_stdout
from pathlib import Path

import torch
import torch.nn.functional as F

import pyhip
from pyhip.ops.conv import conv_depthwise as depthwise

_PADDING = (0, 2, 2)
_DEFAULT_SHAPE = (1, 512, 61, 45, 80)


def _positive_int(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _warmup(run, device, iterations):
    """Keep first-use compilation/loading separate from warmed timing."""
    _synchronize(device)
    start = time.perf_counter()
    output = run()
    _synchronize(device)
    first_call_ms = (time.perf_counter() - start) * 1000
    start = time.perf_counter()
    for _ in range(iterations):
        output = run()
    _synchronize(device)
    return output, first_call_ms, (time.perf_counter() - start) * 1000


def _measure(run, device, iterations):
    """Time a batch of eager calls or graph replays on the current stream."""
    _synchronize(device)
    if device.type == "cuda":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
    start = time.perf_counter()
    for _ in range(iterations):
        run()
    if device.type == "cuda":
        end_event.record()
        end_event.synchronize()
        gpu_ms = start_event.elapsed_time(end_event) / iterations
    else:
        gpu_ms = None
    eager_ms = (time.perf_counter() - start) * 1000 / iterations
    return gpu_ms, eager_ms


def _capture_graph(run, device, warmup):
    """Warm up on a side stream, then capture one call with a persistent output."""
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):
        for _ in range(warmup):
            run()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    start = time.perf_counter()
    with torch.cuda.graph(graph, stream=stream):
        output = run()
    stream.synchronize()
    return graph, output, (time.perf_counter() - start) * 1000


def _summary(samples):
    return {
        "samples_ms": samples,
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def _time_runners(runners, device, iterations, rounds, wall_time_key):
    samples = {name: {"gpu": [], "wall": []} for name in runners}
    for round_index in range(rounds):
        order = list(runners)
        if round_index % 2:
            order.reverse()
        for name in order:
            gpu_ms, wall_ms = _measure(runners[name], device, iterations)
            if gpu_ms is not None:
                samples[name]["gpu"].append(gpu_ms)
            samples[name]["wall"].append(wall_ms)
    return {
        name: {
            "gpu_time": _summary(values["gpu"]) if values["gpu"] else None,
            wall_time_key: _summary(values["wall"]),
        }
        for name, values in samples.items()
    }


def _add_metrics(results, flops, tensor_bytes, wall_time_key):
    baseline_ms = None
    for result in results.values():
        latency = result["gpu_time"] or result[wall_time_key]
        median_ms = latency["median_ms"]
        if baseline_ms is None:
            baseline_ms = median_ms
        result.update(
            tflops=flops / median_ms / 1e9,
            effective_gbps=tensor_bytes / median_ms / 1e6,
            speedup=baseline_ms / median_ms,
        )


def _markdown_report(report):
    lines = ["# Depthwise Conv3D benchmark", ""]

    def table(headers, rows):
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join("---" for _ in headers) + " |")
        for row in rows:
            cells = [str(value).replace("|", "\\|").replace("\n", " ") for value in row]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    table(
        ["Setting", "Value"],
        [
            ["Input shape", str(report["input_shape"])],
            ["Filter / padding", f"{report['filter']} / {report['padding']}"],
            ["Groups / bias", f"{report['groups']} / {report['bias']}"],
            ["Device", f"{report['device']} ({report['device_name']})"],
            ["Architecture", report["arch"] or "CPU"],
            ["PyTorch / ROCm", f"{report['torch']} / {report['rocm'] or '-'}"],
            ["Timing", f"{report['rounds']} rounds × {report['iterations']} calls"],
            [
                "Warmup / seed",
                f"{report['warmup_iterations']} calls / {report['seed']}",
            ],
        ],
    )
    if report["cuda_graph"]:
        lines.extend(
            [
                (
                    "Each graph captures one Conv3D and reuses its input/output storage. "
                    "Capture and warmup costs are excluded from replay timing."
                ),
                "",
            ]
        )
    for result in report["results"]:
        dtype = {"torch.float16": "FP16", "torch.bfloat16": "BF16"}[result["dtype"]]
        dispatch = result["dispatch"]
        lines.extend(
            [
                f"## {dtype}",
                "",
                f"Requested method: `{report['method']}`. Selected backend: `{dispatch['backend']}`.",
                "",
            ]
        )
        if dispatch["backend"] == "hip":
            lines.extend(
                [
                    f"Arithmetic: `{dispatch['arithmetic']}`. Output tile: {dispatch['output_tile']}.",
                    "",
                ]
            )
        elif "fallback_reason" in dispatch:
            lines.extend([f"Torch fallback: {dispatch['fallback_reason']}.", ""])
        setup_headers = ["Backend", "First call (ms)", "Eager warmup (ms)"]
        if result["cuda_graph"] is not None:
            setup_headers.extend(["Graph capture (ms)", "Graph warmup (ms)"])
        setup_rows = []
        for name, timing in result["timings"].items():
            row = [name, f"{timing['first_call_ms']:.2f}", f"{timing['warmup_ms']:.2f}"]
            if result["cuda_graph"] is not None:
                graph = result["cuda_graph"][name]
                row.extend([f"{graph['capture_ms']:.2f}", f"{graph['warmup_ms']:.2f}"])
            setup_rows.append(row)
        lines.extend(["### Setup", ""])
        table(setup_headers, setup_rows)
        modes = [("Eager execution", result["timings"], "eager_time", "Eager (ms)")]
        if result["cuda_graph"] is not None:
            modes.append(
                (
                    "CUDA graph replay",
                    result["cuda_graph"],
                    "replay_time",
                    "Replay (ms)",
                )
            )
        for title, timings, wall_key, wall_label in modes:
            lines.extend([f"### {title}", ""])
            rows = []
            for name, timing in timings.items():
                gpu = timing["gpu_time"]
                rows.append(
                    [
                        name,
                        f"{gpu['median_ms']:.4f}" if gpu else "-",
                        f"{timing[wall_key]['median_ms']:.4f}",
                        f"{timing['tflops']:.2f}",
                        f"{timing['effective_gbps']:.2f}",
                        f"{timing['speedup']:.2f}x",
                        f"{timing['accuracy']['normalized_diff']:.3e}",
                    ]
                )
            table(
                [
                    "Backend",
                    "GPU (ms)",
                    wall_label,
                    "TFLOP/s",
                    "Effective GB/s",
                    "Speedup",
                    "Normalized diff",
                ],
                rows,
            )
        if result["profile"] is not None:
            lines.extend(
                [
                    "### Profile",
                    "",
                    "```text",
                    result["profile"]["summary"].rstrip(),
                    "```",
                    "",
                    f"Trace directory: `{result['profile']['directory']}`.",
                    "",
                ]
            )
    return "\n".join(lines)


def _check_accuracy(actual, expected, dtype):
    if not torch.isfinite(actual).all():
        raise AssertionError("depthwise Conv3D returned nonfinite output")
    normalized_diff = pyhip.calc_diff(expected, actual)
    if not math.isfinite(normalized_diff) or normalized_diff >= 1e-4:
        raise AssertionError(f"normalized diff {normalized_diff} exceeds 1e-4")
    # Random inputs have unit scale and 75 filter taps. These bounds match the
    # large-shape refactor checks; focused pytest cases use tighter small inputs.
    atol, rtol = (0.25, 0.01) if dtype == torch.bfloat16 else (0.04, 0.003)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    return {
        "normalized_diff": normalized_diff,
        "atol": atol,
        "rtol": rtol,
    }


def _dispatch_info(x, weight, bias, method):
    if method == "torch":
        return {"backend": "torch"}
    reason = depthwise._hip_unavailable_reason(
        x, weight, bias, (1, 1, 1), _PADDING, (1, 1, 1), x.shape[1]
    )
    if reason:
        if method == "hip":
            raise ValueError(f"HIP depthwise Conv3D: {reason}")
        return {"backend": "torch", "fallback_reason": reason}
    from pyhip.runtime.hiptools import amdgpu_arch

    config = depthwise._kernel_config(amdgpu_arch(), x.dtype)
    return {
        "backend": "hip",
        "arithmetic": "bf16_fma" if config["BF16_FMA"] else "native_dot",
        "output_tile": config["OUTPUT_TILE"],
    }


def _profile(runners, device, directory):
    directory.mkdir(parents=True, exist_ok=True)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(
        activities=activities,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(str(directory)),
        record_shapes=True,
    ) as profiler:
        for _ in range(5):
            for name, run in runners.items():
                with torch.profiler.record_function(name):
                    run()
        _synchronize(device)
    sort_by = "self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total"
    return {
        "summary": profiler.key_averages().table(sort_by=sort_by, row_limit=10),
        "directory": str(directory),
    }


def _benchmark_dtype(args, dtype, device):
    n, c, d, h, w = _DEFAULT_SHAPE
    generator = torch.Generator(device=device).manual_seed(args.seed)
    x = torch.randn(_DEFAULT_SHAPE, device=device, dtype=dtype, generator=generator)
    weight = torch.randn(
        (c, 1, 3, 5, 5), device=device, dtype=dtype, generator=generator
    )
    bias = (
        torch.randn((c,), device=device, dtype=dtype, generator=generator)
        if args.bias
        else None
    )
    dispatch = _dispatch_info(x, weight, bias, args.method)
    method = None if args.method == "auto" else args.method

    def torch_conv():
        return F.conv3d(x, weight, bias, padding=_PADDING, groups=c)

    def pyhip_conv():
        return depthwise.conv_depthwise_3d(
            x, weight, bias, padding=_PADDING, groups=c, method=method
        )

    runners = {"Torch": torch_conv, "PyHIP": pyhip_conv}
    results = {}
    reference = None
    for name, run in runners.items():
        output, first_call_ms, warmup_ms = _warmup(run, device, args.warmup)
        if name == "Torch":
            reference = output
            backend_accuracy = {"normalized_diff": 0.0}
        else:
            accuracy = _check_accuracy(output, reference, dtype)
            backend_accuracy = accuracy
        results[name] = {
            "first_call_ms": first_call_ms,
            "warmup_ms": warmup_ms,
            "accuracy": backend_accuracy,
        }
    eager_timings = _time_runners(
        runners, device, args.iters, args.rounds, "eager_time"
    )
    for name, result in results.items():
        result.update(eager_timings[name])

    output_elements = n * c * (d - 2) * h * w
    flops = 2 * output_elements * 3 * 5 * 5
    tensor_bytes = (
        x.numel()
        + weight.numel()
        + output_elements
        + (bias.numel() if bias is not None else 0)
    ) * x.element_size()
    _add_metrics(results, flops, tensor_bytes, "eager_time")
    graph_results = None
    if args.cuda_graph:
        graphs = {}
        graph_results = {}
        for name, run in runners.items():
            graph, captured_output, capture_ms = _capture_graph(
                run, device, args.warmup
            )
            graphs[name] = (graph, captured_output)
            graph.replay()
            replay_accuracy = _check_accuracy(captured_output, reference, dtype)
            _, first_replay_ms, replay_warmup_ms = _warmup(
                graph.replay, device, args.warmup
            )
            graph_results[name] = {
                "capture_ms": capture_ms,
                "first_replay_ms": first_replay_ms,
                "warmup_ms": replay_warmup_ms,
                "accuracy": replay_accuracy,
            }
        replay_runners = {
            name: graph.replay for name, (graph, _output) in graphs.items()
        }
        replay_timings = _time_runners(
            replay_runners, device, args.iters, args.rounds, "replay_time"
        )
        for name, result in graph_results.items():
            result.update(replay_timings[name])
            # Validate the persistent output again after all timed replays.
            result["accuracy"] = _check_accuracy(graphs[name][1], reference, dtype)
        _add_metrics(graph_results, flops, tensor_bytes, "replay_time")
    del reference, output
    profile = None
    if args.profile is not None:
        profile = _profile(
            runners, device, args.profile / str(dtype).removeprefix("torch.")
        )
    return {
        "dtype": str(dtype),
        "dispatch": dispatch,
        "accuracy": accuracy,
        "timings": results,
        "cuda_graph": graph_results,
        "profile": profile,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=("fp16", "bf16", "both"), default="bf16")
    parser.add_argument("--method", choices=("auto", "hip", "torch"), default="auto")
    parser.add_argument("--device", default="cuda:0", help="cuda:N or cpu")
    parser.add_argument("--iters", type=_positive_int, default=100)
    parser.add_argument("--rounds", type=_positive_int, default=7)
    parser.add_argument("--warmup", type=_positive_int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-bias", action="store_false", dest="bias")
    parser.add_argument(
        "--cuda-graph",
        action="store_true",
        help="also capture and benchmark Torch/PyHIP graph replay after eager timing",
    )
    parser.add_argument(
        "--output", type=Path, help="save metadata and raw timing samples as JSON"
    )
    parser.add_argument(
        "--profile",
        type=Path,
        nargs="?",
        const=Path("log/conv3d_depthwise"),
        help="profile both backends after timing; optionally set a trace directory",
    )
    args = parser.parse_args()
    try:
        device = torch.device(args.device)
    except RuntimeError as error:
        parser.error(str(error))
    if device.type not in ("cpu", "cuda"):
        parser.error("--device must be cpu or cuda:N")
    if args.cuda_graph and device.type != "cuda":
        parser.error("--cuda-graph requires a CUDA/ROCm device")
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA/ROCm device is unavailable; use --device cpu for Torch")
    device_context = (
        torch.cuda.device(device) if device.type == "cuda" else nullcontext()
    )
    with device_context, torch.no_grad():
        if device.type == "cuda":
            device = torch.device("cuda", torch.cuda.current_device())
            properties = torch.cuda.get_device_properties(device)
            device_name = properties.name
            arch = getattr(properties, "gcnArchName", None)
        else:
            device_name, arch = "CPU", None
        dtypes = {
            "fp16": [torch.float16],
            "bf16": [torch.bfloat16],
            "both": [torch.float16, torch.bfloat16],
        }[args.dtype]
        report = {
            "input_shape": list(_DEFAULT_SHAPE),
            "filter": [3, 5, 5],
            "padding": list(_PADDING),
            "groups": _DEFAULT_SHAPE[1],
            "bias": args.bias,
            "seed": args.seed,
            "device": str(device),
            "device_name": device_name,
            "arch": arch,
            "torch": torch.__version__,
            "rocm": torch.version.hip,
            "method": args.method,
            "iterations": args.iters,
            "rounds": args.rounds,
            "warmup_iterations": args.warmup,
            "cuda_graph": args.cuda_graph,
        }
        # Compiler diagnostics belong on stderr so stdout can be saved as Markdown.
        with redirect_stdout(sys.stderr):
            report["results"] = [
                _benchmark_dtype(args, dtype, device) for dtype in dtypes
            ]
    print(_markdown_report(report))
    if args.output is not None:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"\nJSON results saved to `{args.output}`.")


if __name__ == "__main__":
    main()
