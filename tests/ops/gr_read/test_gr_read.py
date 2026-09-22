# SPDX-License-Identifier: MIT
"""GRRead basic correctness and performance tests with an inline reference and CLI.

Direct execution checks all 21 batch sizes, including 33/64/128/256/512,
before timing Down/Up/Total and a full-row torch.compile comparison.
Use --check-only for correctness without timing. Pytest checks basic numerical
results, empty inputs, and tails without running performance tests.
Retain every timing sample. Stop on a failed gate without waiting or changing hardware.
"""

import argparse
from datetime import datetime
from functools import cache
import gc
import json
import math
import os
from pathlib import Path
import shutil
from statistics import median
import subprocess
import sys
import traceback

C, H, R = 4, 2560, 320
CHECK_ROWS = 1024
DOWN_TOLERANCE = dict(rtol=0.015625, atol=2e-5)
OUTPUT_TOLERANCE = dict(rtol=1e-2, atol=5e-3)
DEFAULT_BATCHES = (33, 64, 128, 256, 512) + tuple(k * 1024 for k in (1, 2, 4, 8, 10, 12, 16, 20, 24, 28, 30, 32, 36, 48, 60, 64))
SCOPES = ("down", "up", "total")
TIMING_SCOPES = (*SCOPES, "torch_compile")


def parse_batch(value):
    text = value.strip().lower()
    try:
        rows = int(text[:-1]) * 1024 if text.endswith("k") else int(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("batch must be an integer or an integer followed by K") from error
    if rows < 1:
        raise argparse.ArgumentTypeError("batch must be a positive integer")
    return rows


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    batches = parser.add_mutually_exclusive_group()
    batches.add_argument("--rows", type=parse_batch, help="run one batch size, e.g. 4K")
    batches.add_argument("--batches", nargs="+", type=parse_batch,
                         help=f"batch sizes to test; defaults to all {len(DEFAULT_BATCHES)} sizes, including 33/64/128/256/512")
    parser.add_argument("--gpu", type=int, default=3, help="physical ROCm GPU index (default: 3)")
    parser.add_argument("--check-only", action="store_true", help="check all selected batches without hardware gates or timing")
    parser.add_argument("--seed", type=int, default=131)
    parser.add_argument("--warmup", type=int, default=2, help="warmup calls per scope")
    parser.add_argument("--iters", type=int, default=10, help="samples per scope; report the median of all samples")
    parser.add_argument("--buffers", type=int, default=10, help="independent rotating X/weights/P/Y buffer sets")
    parser.add_argument("--amd-smi", type=Path, help="AMD SMI executable or Python CLI with PTL reporting")
    parser.add_argument("--output", type=Path, help="new output directory; defaults to results/one_stop_* and must not exist")
    args = parser.parse_args(argv)
    if args.gpu < 0 or args.warmup < 0 or args.iters < 1 or args.buffers < 1:
        parser.error("gpu/warmup must be nonnegative; iters/buffers must be positive")
    args.batches = [args.rows] if args.rows is not None else list(args.batches or DEFAULT_BATCHES)
    if len(set(args.batches)) != len(args.batches):
        parser.error("batch sizes must be distinct")
    return args


def prepare_cli_environment(args):
    """Select the GPU before importing Torch without installing dependencies or changing hardware."""
    repo = Path(__file__).resolve().parents[3]
    paths = (repo, repo / "src", Path("/opt/aiter"),
             Path(f"/usr/local/lib/python{sys.version_info.major}.{sys.version_info.minor}/dist-packages"))
    for path in paths:
        if path.is_dir() and str(path) not in sys.path:
            sys.path.append(str(path))
    if os.environ.get("HSA_CU_MASK") or os.environ.get("ROC_GLOBAL_CU_MASK"):
        raise RuntimeError("GRRead reference tests require an unmasked GPU")
    for key in ("ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL", "CUDAPERF", "COMPILE_ONLY",
                "FLYDSL_DUMP_IR", "FLYDSL_DEBUG_DUMP_ASM", "FLYDSL_DUMP_DIR", "FLYDSL_RUNTIME_RUN_ONLY"):
        os.environ.pop(key, None)
    os.environ["HIP_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["FLYDSL_RUNTIME_ENABLE_CACHE"] = "1"


def write_json(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def use_checkout_package():
    import importlib.util
    repo = Path(__file__).resolve().parents[3]
    expected = repo / "src/__init__.py"
    module = sys.modules.get("pyhip")
    if module is not None:
        if Path(module.__file__).resolve() != expected:
            raise RuntimeError(f"wrong PyHIP checkout: {module.__file__}")
        return
    spec = importlib.util.spec_from_file_location("pyhip", expected,
                                                  submodule_search_locations=[str(repo / "src")])
    module = importlib.util.module_from_spec(spec)
    sys.modules["pyhip"] = module
    spec.loader.exec_module(module)


@cache
def dependencies():
    # 延迟导入：CLI先选GPU；--help和参数解析不初始化Torch，也不依赖pytest。
    use_checkout_package()
    import torch
    import torch.nn.functional as F
    import flydsl.compiler as flyc
    from pyhip.misc import cudaPerf
    if torch.version.hip is None or not torch.cuda.is_available():
        raise RuntimeError("ROCm GPU required")
    props = torch.cuda.get_device_properties(0)
    if not props.gcnArchName.startswith("gfx942"):
        raise RuntimeError("GRRead currently requires gfx942")

    @torch.compile(fullgraph=True)
    def _mix_reference(x, w_down, w_up):
        p = F.silu(F.linear(x, w_down) / C)
        logits = F.linear(p, w_up)
        gates = torch.sigmoid(logits).unflatten(-1, (C, H))
        return p, (gates * x.unflatten(-1, (C, H))).mean(dim=-2)

    return torch, flyc, _mix_reference, cudaPerf


@cache
def torch_compile_mix():
    """Standalone copy of SGLang's output-only _mix_compute, with default compile options."""
    torch, _, _, _ = dependencies()
    import torch.nn.functional as F

    # Mirrors hyperconnection.py at SGLang 2843214f6ed923e992a74ee4d7a0cda5d7deddbf.
    # Keep this separate from the chunked P/Y correctness reference: benchmark
    # the full T in one call, returning only Y as the actual model does.
    def _mix_compute(
        hyper_input_normed: torch.Tensor,
        input_mix_weight_down: torch.Tensor,
        input_mix_weight_up: torch.Tensor,
        hc: int,
        hs: int,
    ) -> torch.Tensor:
        input_mix_weight = F.silu(
            F.linear(hyper_input_normed, input_mix_weight_down) / hc
        )
        input_mix_weight = F.linear(input_mix_weight, input_mix_weight_up)
        input_mix_weight = torch.sigmoid(input_mix_weight)
        input_mix_weight = input_mix_weight.unflatten(-1, (hc, hs))
        output = (
            input_mix_weight * hyper_input_normed.unflatten(-1, (hc, hs))
        ).mean(dim=-2)
        return output

    return torch.compile(_mix_compute)


def prepare_reader(x, w_down, w_up):
    dependencies()
    from pyhip.contrib.flydsl.gr_read import GRReadPrefill, prepare_weights
    packed_down, packed_up = prepare_weights(w_down, w_up)
    reader = GRReadPrefill(x.shape[0], packed_down, packed_up)
    reader._check_input(x)
    return reader


def prepare_gr_read(x, w_down, w_up):
    """Compatibility tuple for historical scripts; current tests use the public object."""
    reader = prepare_reader(x, w_down, w_up)
    return (reader.w_down, reader.w_up, reader.partial, reader.output,
            reader.down, reader.up, reader.down_config, reader.up_config[1])


def run_down(x, w_down, partial, down):
    """Launch Down with prepared buffers and return the BF16 intermediate."""
    torch, _, _, _ = dependencies()
    if x.shape[0]:
        with torch.cuda.device(x.device):
            down(x, w_down, partial, x.shape[0], torch.cuda.current_stream(x.device))
    return partial


def run_up(x, w_up, partial, output, up):
    """Launch Up with prepared buffers and return the BF16 output."""
    torch, _, _, _ = dependencies()
    if x.shape[0]:
        with torch.cuda.device(x.device):
            up(x, w_up, partial, output, x.shape[0], torch.cuda.current_stream(x.device))
    return output


def run_gr_read(x, w_down, w_up, partial, output, down, up):
    """Launch Down then Up on the current stream, without host row splitting."""
    torch, _, _, _ = dependencies()
    if x.shape[0]:
        with torch.cuda.device(x.device):
            stream = torch.cuda.current_stream(x.device)
            down(x, w_down, partial, x.shape[0], stream)
            up(x, w_up, partial, output, x.shape[0], stream)
    return output


def reference_bf16(x, w_down, w_up):
    torch, _, mix, _ = dependencies()
    activation = torch.empty((x.shape[0], R), device=x.device, dtype=torch.bfloat16)
    output = torch.empty((x.shape[0], H), device=x.device, dtype=torch.bfloat16)
    with torch.autocast("cuda", enabled=False):
        for begin in range(0, x.shape[0], CHECK_ROWS):
            end = min(begin + CHECK_ROWS, x.shape[0])
            p, y = mix(x[begin:end], w_down, w_up)
            assert p.dtype == y.dtype == torch.bfloat16
            activation[begin:end], output[begin:end] = p, y
    return activation, output


def check_close(actual, expected, tolerance):
    torch, _, _, _ = dependencies()
    assert actual.shape == expected.shape
    error_squared = expected_squared = 0.0
    for begin in range(0, actual.shape[0], CHECK_ROWS):
        a, e = actual[begin:begin + CHECK_ROWS].double(), expected[begin:begin + CHECK_ROWS].double()
        torch.testing.assert_close(a, e, **tolerance)
        error_squared += (a - e).square().sum().item()
        expected_squared += e.square().sum().item()
    return (error_squared / expected_squared if expected_squared else error_squared) ** 0.5


def make_inputs(torch, rows, seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn((rows, 10240), device="cuda", dtype=torch.bfloat16, generator=generator)
    wd = torch.randn((320, 10240), device="cuda", dtype=torch.bfloat16, generator=generator) * 0.02
    wu = torch.randn((10240, 320), device="cuda", dtype=torch.bfloat16, generator=generator) * 0.02
    return x, wd, wu


def check_batch(rows, args):
    torch, _, _, _ = dependencies()
    with torch.no_grad():
        x, wd, wu = make_inputs(torch, rows, args.seed)
        reader = prepare_reader(x, wd, wu)
        partial, output = reader.partial, reader.output
        dm, dw, dn, dk = reader.down_config
        um, un = reader.up_config
        p_expected, y_expected = reference_bf16(x, wd, wu)
        assert partial.dtype == output.dtype == torch.bfloat16
        assert partial.shape == (rows, R)
        partial.fill_(torch.nan)
        assert reader.run_down(x) is partial
        p_error = check_close(partial, p_expected, DOWN_TOLERANCE)
        output.fill_(torch.nan)
        assert reader.run_up(x) is output
        y_error = check_close(output, y_expected, OUTPUT_TOLERANCE)
        partial.fill_(torch.nan); output.fill_(torch.nan)
        assert reader(x) is output
        check_close(partial, p_expected, DOWN_TOLERANCE)
        check_close(output, y_expected, OUTPUT_TOLERANCE)
        result = {"complete": True, "rows": rows, "down_n_splits": dn,
                  "down_block_m": dm, "down_num_waves": dw, "down_block_k": dk,
                  "up_block_m": um, "n_splits": un, "P_rel_l2": p_error, "Y_rel_l2": y_error,
                  "P_tolerance": DOWN_TOLERANCE, "Y_tolerance": OUTPUT_TOLERANCE,
                  "partial_shape": list(partial.shape), "checked_scopes": list(SCOPES)}
        print(f"  T={rows:5d} Down M{dm}/W{dw}/N{dn}/BK{dk} / Up M{um}/N{un} "
              f"Down rel_l2={p_error:.6g} Output rel_l2={y_error:.6g} PASS", flush=True)
        return result


def read_hardware(gpu, amd_smi=None):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("ROCPROF", "ROCP_", "AQLPROFILE")) and k not in (
        "LD_PRELOAD", "HSA_TOOLS_LIB", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL")}
    query = subprocess.run(["rocm-smi", "-d", str(gpu), "--showuse", "--showmemuse", "--showperflevel",
                            "--showmaxpower", "--showbus", "--json"],
                           env=env, text=True, capture_output=True, check=True, timeout=30)
    card = json.loads(query.stdout)[f"card{gpu}"]
    bundled = Path("/tmp/amd-smi-lib-26.2.2-rocm-7.2.3/opt/rocm-7.2.3/libexec/amdsmi_cli/amdsmi_cli.py")
    cli = amd_smi or (bundled if bundled.is_file() else shutil.which("amd-smi"))
    if cli is None:
        raise RuntimeError("PTL reporting requires a compatible --amd-smi")
    cli = Path(cli).resolve()
    command = [str(cli)]
    if cli.suffix == ".py":
        command = [sys.executable, str(cli)]
        base = cli.parents[2]
        if (base / "share/amd_smi").is_dir():
            env["PYTHONPATH"] = os.pathsep.join((str(base / "share/amd_smi"), str(cli.parent)))
            env["LD_LIBRARY_PATH"] = os.pathsep.join((str(base / "lib"), str(base / "share/amd_smi/amdsmi"), env.get("LD_LIBRARY_PATH", "")))
    query = subprocess.run(command + ["static", "-g", str(gpu), "--limit", "--json"],
                           env=env, text=True, capture_output=True, check=True, timeout=30)
    limit = next(item["limit"] for item in json.loads(query.stdout)["gpu_data"] if str(item["gpu"]) == str(gpu))
    return {"gpu": gpu, "card": card, "limit": limit, "settings_written": False}


def validate_hardware(snapshot):
    card, limit = snapshot["card"], snapshot["limit"]
    use, vram = int(card["GPU use (%)"]), int(card["GPU Memory Allocated (VRAM%)"])
    if use > 5 or vram > 20:
        raise RuntimeError(f"GPU{snapshot['gpu']} busy: use={use}%, VRAM={vram}%; stop without retry")
    formats = {part.strip().upper() for part in str(limit.get("ptl_format")).split(",")}
    if str(limit.get("ptl_state")).lower() != "enabled" or formats != {"VECTOR", "F8"}:
        raise RuntimeError("Timing requires PTL Enabled/VECTOR,F8; no hardware settings were changed")


def hardware_gate(args, folder, phase):
    import time
    time.sleep(2)
    snapshot = read_hardware(args.gpu, args.amd_smi)
    # 门禁未通过也先保留实际状态；不重新查询直到通过。
    write_json(folder / f"hardware_{phase}.json", snapshot)
    validate_hardware(snapshot)
    print(f"  GPU{args.gpu} {phase}: PTL Enabled/VECTOR,F8, use={snapshot['card']['GPU use (%)']}%, "
          f"VRAM={snapshot['card']['GPU Memory Allocated (VRAM%)']}%", flush=True)
    return snapshot


def tensor_address(tensor, *, output=False):
    pointer, base = tensor.data_ptr(), tensor.untyped_storage().data_ptr()
    if output:
        assert tensor.is_contiguous() and tensor.storage_offset() == 0 and pointer == base
        assert pointer % 256 == 0, "performance Y must start at its aligned allocation base"
    return {"pointer": pointer, "storage_base": base, "storage_offset": tensor.storage_offset(),
            "mod256": pointer % 256, "mod4096": pointer % 4096}


def benchmark_batch(rows, args, folder):
    before = hardware_gate(args, folder, "before")
    torch, _, _, cuda_perf = dependencies()
    compiled_mix = torch_compile_mix()
    with torch.no_grad():
        x, wd, wu = make_inputs(torch, rows, args.seed)
        inputs = [x] + [x.clone() for _ in range(args.buffers - 1)]
        # Both implementations rotate over independent weight allocations.
        # Each pair has identical logical values; packing happens only here.
        weights = [(wd, wu)] + [(wd.clone(), wu.clone()) for _ in range(args.buffers - 1)]
        readers = [prepare_reader(input_x, *pair) for input_x, pair in zip(inputs, weights)]
        expected_p = expected_y = None
        expected_torch = None
        torch_outputs = [None] * args.buffers
        addresses = []
        calls = {scope: [] for scope in TIMING_SCOPES}
        for bi, (input_x, reader, (raw_down, raw_up)) in enumerate(zip(inputs, readers, weights)):
            dm, dw, dn, dk = reader.down_config
            um, un = reader.up_config
            partial, output = reader.partial, reader.output
            addresses.append({name: tensor_address(tensor, output=name == "Y") for name, tensor in zip(
                ("X", "W_down", "W_up", "P", "Y"),
                (input_x, reader.w_down, reader.w_up, partial, output))})
            calls["down"].append(lambda x=input_x, r=reader: r.run_down(x))
            calls["up"].append(lambda x=input_x, r=reader: r.run_up(x))
            calls["total"].append(lambda x=input_x, r=reader: r(x))
            calls["torch_compile"].append(
                lambda x=input_x, wd=raw_down, wu=raw_up: compiled_mix(x, wd, wu, C, H))
            partial.fill_(torch.nan); output.fill_(torch.nan)
            reader(input_x)
            if expected_p is None:
                expected_p, expected_y = partial.clone(), output.clone()
            torch.testing.assert_close(partial, expected_p, rtol=0, atol=0)
            torch.testing.assert_close(output, expected_y, rtol=0, atol=0)
            # Compile the actual full shape and check every buffer before timing.
            torch_outputs[bi] = calls["torch_compile"][bi]()
            if expected_torch is None:
                expected_torch = torch_outputs[bi].clone()
            torch.testing.assert_close(torch_outputs[bi], expected_torch, rtol=0, atol=0)
            check_close(output, torch_outputs[bi], OUTPUT_TOLERANCE)
            addresses[-1].update({name: tensor_address(tensor) for name, tensor in (
                ("Torch_X", input_x), ("Torch_W_down", raw_down), ("Torch_W_up", raw_up),
                ("Torch_Y_prepared", torch_outputs[bi]))})
        assert all(len({row[name]["pointer"] for row in addresses}) == args.buffers for name in addresses[0])
        write_json(folder / "addresses.json", addresses)
        for scope in TIMING_SCOPES:
            for index in range(args.warmup): calls[scope][index % args.buffers]()
        torch.cuda.synchronize()
        ready = hardware_gate(args, folder, "before_samples")
        timings = {}
        with (folder / "samples.jsonl").open("x") as raw:
            # Retain the original stage-by-stage sampling order; append Torch.
            # These are whole-scope medians, not an interleaved A/B experiment.
            for scope in TIMING_SCOPES:
                perf = cuda_perf(name=f"gr_read_{scope}", verbose=0)
                if not perf.enable:
                    raise RuntimeError("CUDAPERF disables GRRead timing")
                for index in range(args.iters):
                    bi = index % args.buffers
                    with perf: actual = calls[scope][bi]()
                    us = perf.latencies[-1] * 1e6
                    assert math.isfinite(us) and us > 0
                    record = {"scope": scope, "sample": index, "buffer": bi, "us": us}
                    if scope == "torch_compile":
                        torch_outputs[bi] = actual
                        record["output_address"] = tensor_address(actual)
                    raw.write(json.dumps(record) + "\n")
                    raw.flush()
                    # Validate each timed Torch output before a later call can
                    # release/reuse it. No validation is inside the timer.
                    if scope == "torch_compile":
                        torch.testing.assert_close(actual, expected_torch, rtol=0, atol=0)
                samples = [value * 1e6 for value in perf.latencies]
                assert len(samples) == args.iters
                for bi in sorted({index % args.buffers for index in range(args.iters)}):
                    reader = readers[bi]
                    actual = (torch_outputs[bi] if scope == "torch_compile" else
                              reader.partial if scope == "down" else reader.output)
                    expected = (expected_torch if scope == "torch_compile" else
                                expected_p if scope == "down" else expected_y)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                flops = (2 if scope in ("down", "up") else 4) * rows * 10240 * 320
                elapsed = median(samples)
                timings[scope] = {"elapsed_us": elapsed, "samples_us": samples, "gemm_FLOPs": flops,
                                  "effective_TFLOPS": flops / elapsed / 1e6}
                print(f"  T={rows:5d} Down M{dm}/W{dw}/N{dn}/BK{dk} / Up M{um}/N{un} "
                      f"{scope}: {elapsed:.3f} us", flush=True)
        after = hardware_gate(args, folder, "after")
        return {"complete": True, "rows": rows, "down_n_splits": dn,
                "down_block_m": dm, "down_num_waves": dw, "down_block_k": dk,
                "up_block_m": um, "n_splits": un, "timings": timings,
                "hardware_before": before, "hardware_before_samples": ready, "hardware_after": after,
                "buffers": args.buffers, "warmup_each": args.warmup, "samples_each": args.iters,
                "timed_outputs_bitexact": True, "all_samples_retained": True,
                "Total_measured_directly": True, "settings_written": False,
                "speedup_vs_torch_compile": timings["torch_compile"]["elapsed_us"] / timings["total"]["elapsed_us"],
                "torch_compile": {"full_rows": rows, "returns": "Y", "options": "default", "cuda_graph": False,
                                  "shared_X": True, "independent_raw_weight_buffers": args.buffers,
                                  "output_allocation": "native", "Y_tolerance": OUTPUT_TOLERANCE},
                "sampling_order": list(TIMING_SCOPES)}


def release_buffers():
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def print_summary(results):
    print("\n| Batch | Down | Up | Down us / TFLOPS | Up us / TFLOPS | Total us / TFLOPS | Torch compile us / TFLOPS | Speedup |", flush=True)
    print("|---:|---:|---:|---:|---:|---:|---:|---:|", flush=True)
    for result in results:
        cells = [f"{result['timings'][scope]['elapsed_us']:.3f} / {result['timings'][scope]['effective_TFLOPS']:.3f}" for scope in TIMING_SCOPES]
        down = (f"M{result['down_block_m']}/W{result['down_num_waves']}"
                f"/N{result['down_n_splits']}/BK{result['down_block_k']}")
        print(f"| {result['rows']} | {down} | M{result['up_block_m']}/N{result['n_splits']} | "
              f"{' | '.join(cells)} | {result['speedup_vs_torch_compile']:.3f}x |", flush=True)
    print("Speedup = Torch compile Total / PyHIP Total; eager cudaPerf, full-row calls, no CUDA Graph.", flush=True)


def run_suite(args, output):
    result = {"complete": False, "phase": "correctness", "batches": args.batches, "gpu": args.gpu,
              "seed": args.seed, "checks": [], "performance": [], "check_only": args.check_only,
              "protocol": {"buffers": args.buffers, "warmup": args.warmup, "iters": args.iters,
                           "timing_scopes": list(TIMING_SCOPES), "scope_order": "sequential", "cuda_graph": False},
              "settings_written": False}
    active_rows = None
    try:
        torch, _, _, _ = dependencies()
        result["environment"] = {"torch": torch.__version__, "hip": torch.version.hip,
                                 "torch_compile_source": "inline SGLang _mix_compute at 2843214f6ed923e992a74ee4d7a0cda5d7deddbf"}
        print(f"[1/2] Correctness checks for all {len(args.batches)} batch sizes (no idle-GPU requirement)", flush=True)
        for rows in args.batches:
            active_rows = rows
            try:
                checked = check_batch(rows, args)
                assert checked["complete"] and checked["rows"] == rows
                write_json(output / f"check_{rows}.json", checked)
                result["checks"].append(checked)
            finally:
                release_buffers()
        print(f"All {len(result['checks'])} batch sizes passed correctness checks.", flush=True)
        if not args.check_only:
            result["phase"] = "performance"
            print(f"[2/2] Performance: {args.buffers} buffers, {args.warmup} warmup calls, {args.iters} samples/scope, median", flush=True)
            for rows in args.batches:
                active_rows = rows
                folder = output / f"timing_{rows}"
                folder.mkdir()
                try:
                    measured = benchmark_batch(rows, args, folder)
                    assert measured["complete"] and measured["rows"] == rows
                    write_json(output / f"timing_{rows}.json", measured)
                    result["performance"].append(measured)
                finally:
                    release_buffers()
            print_summary(result["performance"])
        else:
            print("--check-only: skipping performance gates and timing.", flush=True)
        result.update(complete=True, phase="complete")
    except Exception:
        result.update(error=traceback.format_exc(), failed_rows=active_rows)
        print(f"{result['phase']} failed at T={active_rows}; stopping and retaining completed results and raw samples.", file=sys.stderr, flush=True)
    write_json(output / "summary.json", result)
    return result


def main(argv=None):
    args = parse_args(argv)
    prepare_cli_environment(args)
    output = args.output or Path(__file__).resolve().parent / "results" / f"one_stop_{datetime.now():%Y%m%d_%H%M%S_%f}_{os.getpid()}"
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    print(f"GRRead test output: {output}", flush=True)
    result = run_suite(args, output)
    if not result["complete"]:
        print(result["error"], file=sys.stderr, flush=True)
        return 1
    print(f"PASS; results and all samples saved to: {output / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# 直接运行上方CLI不需要pytest；pytest复用同一基础正确性函数。
import pytest


@pytest.mark.parametrize("rows", (0, 1, 31, 32, 47, 48, 49, 63, 65, 127, 129, 255, 257, 511, 513,
                                  1023, 1025, 2047, 2049, 2560, 2561, 3072, 3073,
                                  4095, 4097, 65537, *DEFAULT_BATCHES))
def test_gr_read(rows):
    torch = pytest.importorskip("torch")
    if torch.version.hip is None or not torch.cuda.is_available():
        pytest.skip("ROCm GPU required")
    if not torch.cuda.get_device_properties().gcnArchName.startswith("gfx942"):
        pytest.skip("gfx942 required")
    try:
        check_batch(rows, argparse.Namespace(seed=131))
    finally:
        release_buffers()


def require_rocm():
    torch = pytest.importorskip("torch")
    if torch.version.hip is None or not torch.cuda.is_available():
        pytest.skip("ROCm GPU required")
    if not torch.cuda.get_device_properties().gcnArchName.startswith("gfx942"):
        pytest.skip("gfx942 required")
    use_checkout_package()
    return torch


def test_prepared_weights_shared_and_current_stream():
    torch = require_rocm()
    from pyhip.contrib.flydsl.gr_read import GRReadPrefill, prepare_weights

    with torch.inference_mode():
        x, wd, wu = make_inputs(torch, 64, 917)
        packed = prepare_weights(wd, wu)
        before_weights = tuple(w.clone() for w in packed)
        first = GRReadPrefill(64, *packed)
        second = GRReadPrefill(129, *packed)
        assert first.w_down.data_ptr() == second.w_down.data_ptr() == packed[0].data_ptr()
        assert first.w_up.data_ptr() == second.w_up.data_ptr() == packed[1].data_ptr()
        assert first.partial.data_ptr() != second.partial.data_ptr()
        assert first.output.data_ptr() != second.output.data_ptr()
        before_x = x.clone()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            assert first(x) is first.output
        torch.cuda.current_stream().wait_stream(stream)
        expected = reference_bf16(x, wd, wu)[1]
        torch.testing.assert_close(first.output, expected, **OUTPUT_TOLERANCE)
        saved = first.output.clone()
        x2 = make_inputs(torch, 129, 918)[0]
        second(x2)
        torch.testing.assert_close(first.output, saved, rtol=0, atol=0)
        x.mul_(0.99).add_(0.015625)
        first(x)
        torch.testing.assert_close(first.output, reference_bf16(x, wd, wu)[1], **OUTPUT_TOLERANCE)
        assert torch.equal(before_x.mul(0.99).add(0.015625), x)
        assert all(torch.equal(w, original) for w, original in zip(packed, before_weights))


def test_prepared_input_contract():
    torch = require_rocm()
    from pyhip.contrib.flydsl.gr_read import GRReadPrefill, prepare_weights

    x, wd, wu = make_inputs(torch, 64, 919)
    packed = prepare_weights(wd, wu)
    reader = GRReadPrefill(64, *packed)
    with pytest.raises(ValueError):
        reader(x[:63])
    with pytest.raises(ValueError):
        reader(x.float())
    noncontiguous = torch.empty((64, 20480), device=x.device, dtype=x.dtype)[:, ::2]
    with pytest.raises(ValueError):
        reader(noncontiguous)
    with pytest.raises(ValueError):
        GRReadPrefill(True, *packed)
    with pytest.raises(ValueError):
        GRReadPrefill(64, *packed, output=torch.empty_like(x))


def test_prepared_multiple_devices():
    torch = require_rocm()
    if torch.cuda.device_count() < 2 or not torch.cuda.get_device_properties(1).gcnArchName.startswith("gfx942"):
        pytest.skip("two gfx942 devices required")
    from pyhip.contrib.flydsl.gr_read import GRReadPrefill, prepare_weights

    with torch.inference_mode(), torch.cuda.device(0):
        x, wd, wu = make_inputs(torch, 64, 920)
        pd, pu = prepare_weights(wd, wu)
        first = GRReadPrefill(64, pd, pu)
        pd1, pu1 = pd.to("cuda:1"), pu.to("cuda:1")
        second = GRReadPrefill(64, pd1, pu1)
        x1 = x.to("cuda:1")
        expected = first(x).clone()
        actual = second(x1).to("cuda:0")
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert torch.cuda.current_device() == 0
        with torch.cuda.device(1):
            first.run_down(x)
            first.run_up(x)
            assert torch.cuda.current_device() == 1
        torch.testing.assert_close(first.output, expected, rtol=0, atol=0)
