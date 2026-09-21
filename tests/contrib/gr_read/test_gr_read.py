# SPDX-License-Identifier: MIT
"""GRRead basic correctness and performance tests with an inline reference and CLI.

Direct execution checks all 21 batch sizes, including 32/64/128/256/512, before timing Down/Up/Total.
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
DEFAULT_BATCHES = (32, 64, 128, 256, 512) + tuple(k * 1024 for k in (1, 2, 4, 8, 10, 12, 16, 20, 24, 28, 30, 32, 36, 48, 60, 64))
SCOPES = ("down", "up", "total")


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
                         help=f"batch sizes to test; defaults to all {len(DEFAULT_BATCHES)} sizes, including 32/64/128/256/512")
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


@cache
def dependencies():
    # 延迟导入：CLI先选GPU；--help和参数解析不初始化Torch，也不依赖pytest。
    import torch
    import torch.nn.functional as F
    import flydsl.compiler as flyc
    from pyhip.testing.misc import cudaPerf
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


def prepare_gr_read(x, w_down, w_up):
    """Prepare shuffled weights, buffers, and compiled launches for this input."""
    torch, flyc, _, _ = dependencies()
    from pyhip.ops.gr_read.flydsl.common import (
        K, preshuffle_weight, select_down_config, select_n_splits, validate_rows,
    )
    from pyhip.ops.gr_read.flydsl.prefill_down import make_down
    from pyhip.ops.gr_read.flydsl.prefill_up import make_up

    rows = x.shape[0]
    validate_rows(rows)
    if w_down.shape != (R, K) or w_up.shape != (K, R):
        raise ValueError("expected W_down[320,10240] and W_up[10240,320]")
    if w_down.dtype != torch.bfloat16 or w_up.dtype != torch.bfloat16:
        raise ValueError("GRRead weights must be BF16")
    if w_down.device != w_up.device or not w_down.is_cuda or torch.version.hip is None:
        raise ValueError("weights must be on the same ROCm device")
    if x.shape != (rows, K) or x.dtype != w_down.dtype or x.device != w_down.device:
        raise ValueError("input must match the weights' dtype and device and have shape [rows,10240]")
    if not x.is_contiguous():
        raise ValueError("input must be contiguous")
    props = torch.cuda.get_device_properties(x.device)
    if props.gcnArchName.split(":", 1)[0] != "gfx942":
        raise ValueError("GRRead currently targets gfx942")
    down_config = select_down_config(rows, props.multi_processor_count)
    down_block_m, down_num_waves, down_n_splits, down_block_k = down_config
    n_splits = select_n_splits(rows, props.multi_processor_count)
    with torch.cuda.device(x.device):
        # 保留H64内stream优先的布局；这里只搬迁准备逻辑，不改变权重排列。
        up_interleaved = w_up.detach().reshape(C, H // 64, 2, 4, 2, 4, R).permute(1, 0, 2, 4, 3, 5, 6).contiguous().reshape(K, R)
        w_down = preshuffle_weight(w_down)
        w_up = preshuffle_weight(up_interleaved)
        partial = torch.empty((rows, R), dtype=x.dtype, device=x.device)
        output = torch.empty((rows, H), dtype=x.dtype, device=x.device)
        down = up = None
        if rows:
            # compile会执行launcher；直接使用本次足量二维X，不再额外分配占位输入。
            stream = torch.cuda.current_stream(x.device)
            down = flyc.compile(
                make_down(n_splits=down_n_splits, block_m=down_block_m,
                          num_waves=down_num_waves, block_k=down_block_k),
                x, w_down, partial, rows, stream)
            up = flyc.compile(make_up(n_splits=n_splits), x, w_up, partial, output, rows, stream)
    return w_down, w_up, partial, output, down, up, down_config, n_splits


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
        w_down, w_up, partial, output, down, up, down_config, n_splits = prepare_gr_read(x, wd, wu)
        down_block_m, down_num_waves, down_n_splits, down_block_k = down_config
        p_expected, y_expected = reference_bf16(x, wd, wu)
        assert partial.dtype == output.dtype == torch.bfloat16
        assert partial.shape == (rows, R)
        # 单独检查两基础函数，再检查完整调用；共用同一参考，不做结构/内部选型测试。
        partial.fill_(torch.nan)
        assert run_down(x, w_down, partial, down) is partial
        p_error = check_close(partial, p_expected, DOWN_TOLERANCE)
        output.fill_(torch.nan)
        assert run_up(x, w_up, partial, output, up) is output
        y_error = check_close(output, y_expected, OUTPUT_TOLERANCE)
        partial.fill_(torch.nan); output.fill_(torch.nan)
        assert run_gr_read(x, w_down, w_up, partial, output, down, up) is output
        check_close(partial, p_expected, DOWN_TOLERANCE)
        check_close(output, y_expected, OUTPUT_TOLERANCE)
        result = {"complete": True, "rows": rows, "down_n_splits": down_n_splits,
              "down_block_m": down_block_m, "down_num_waves": down_num_waves, "down_block_k": down_block_k,
                  "n_splits": n_splits, "P_rel_l2": p_error, "Y_rel_l2": y_error,
                  "P_tolerance": DOWN_TOLERANCE, "Y_tolerance": OUTPUT_TOLERANCE,
                  "partial_shape": list(partial.shape), "checked_scopes": list(SCOPES)}
        print(f"  T={rows:5d}  Down M{down_block_m}/W{down_num_waves}/N{down_n_splits}/BK{down_block_k} "
              f"/ Up N{n_splits}  "
              f"Down rel_l2={p_error:.6g}  Output rel_l2={y_error:.6g}  PASS", flush=True)
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
    with torch.no_grad():
        x, wd, wu = make_inputs(torch, rows, args.seed)
        inputs = [x] + [x.clone() for _ in range(args.buffers - 1)]
        buffers = [(input_x, *prepare_gr_read(input_x, wd, wu)) for input_x in inputs]
        expected_p = expected_y = None
        addresses = []
        calls = {scope: [] for scope in SCOPES}
        for input_x, w_down, w_up, partial, output, down, up, down_config, n_splits in buffers:
            down_block_m, down_num_waves, down_n_splits, down_block_k = down_config
            addresses.append({name: tensor_address(tensor, output=name == "Y") for name, tensor in zip(
                ("X", "W_down", "W_up", "P", "Y"), (input_x, w_down, w_up, partial, output))})
            calls["down"].append((input_x, w_down, partial, down))
            calls["up"].append((input_x, w_up, partial, output, up))
            calls["total"].append((input_x, w_down, w_up, partial, output, down, up))
            partial.fill_(torch.nan); output.fill_(torch.nan)
            run_gr_read(*calls["total"][-1])
            if expected_p is None:
                expected_p, expected_y = partial.clone(), output.clone()
            torch.testing.assert_close(partial, expected_p, rtol=0, atol=0)
            torch.testing.assert_close(output, expected_y, rtol=0, atol=0)
        assert all(len({row[name]["pointer"] for row in addresses}) == args.buffers for name in addresses[0])
        write_json(folder / "addresses.json", addresses)
        for scope, launch in zip(SCOPES, (run_down, run_up, run_gr_read)):
            for index in range(args.warmup):
                launch(*calls[scope][index % args.buffers])
        torch.cuda.synchronize()
        ready = hardware_gate(args, folder, "before_samples")
        timings = {}
        with (folder / "samples.jsonl").open("x") as raw:
            for scope, launch in zip(SCOPES, (run_down, run_up, run_gr_read)):
                perf = cuda_perf(name=f"gr_read_{scope}", verbose=0)
                if not perf.enable:
                    raise RuntimeError("CUDAPERF disables GRRead timing")
                for index in range(args.iters):
                    bi = index % args.buffers
                    launch_args = calls[scope][bi]
                    with perf:
                        launch(*launch_args)
                    us = perf.latencies[-1] * 1e6
                    assert math.isfinite(us) and us > 0
                    raw.write(json.dumps({"scope": scope, "sample": index, "buffer": bi, "us": us}) + "\n")
                    raw.flush()
                samples = [value * 1e6 for value in perf.latencies]
                assert len(samples) == args.iters
                for bi in sorted({index % args.buffers for index in range(args.iters)}):
                    _, _, _, partial, output, _, _, _, _ = buffers[bi]
                    torch.testing.assert_close(partial if scope == "down" else output,
                                               expected_p if scope == "down" else expected_y, rtol=0, atol=0)
                flops = (4 if scope == "total" else 2) * rows * 10240 * 320
                elapsed = median(samples)
                timings[scope] = {"elapsed_us": elapsed, "samples_us": samples, "gemm_FLOPs": flops,
                                  "effective_TFLOPS": flops / elapsed / 1e6}
                print(f"  T={rows:5d} Down M{down_block_m}/W{down_num_waves}/N{down_n_splits}/BK{down_block_k} "
                      f"/ Up N{n_splits} {scope:5s}: {elapsed:.3f} us, "
                      f"{timings[scope]['effective_TFLOPS']:.3f} effective TFLOPS", flush=True)
        after = hardware_gate(args, folder, "after")
        return {"complete": True, "rows": rows, "down_n_splits": down_n_splits,
                "down_block_m": down_block_m, "down_num_waves": down_num_waves, "down_block_k": down_block_k,
                "n_splits": n_splits, "timings": timings,
                "hardware_before": before, "hardware_before_samples": ready, "hardware_after": after,
                "buffers": args.buffers, "warmup_each": args.warmup, "samples_each": args.iters,
                "timed_outputs_bitexact": True, "all_samples_retained": True, "Total_measured_directly": True,
                "settings_written": False}


def release_buffers():
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def print_summary(results):
    print("\n| Batch | Down | Up | Down us / TFLOPS | Up us / TFLOPS | Total us / TFLOPS |", flush=True)
    print("|---:|---:|---:|---:|---:|---:|", flush=True)
    for result in results:
        cells = [f"{result['timings'][scope]['elapsed_us']:.3f} / {result['timings'][scope]['effective_TFLOPS']:.3f}" for scope in SCOPES]
        down = (f"M{result['down_block_m']}/W{result['down_num_waves']}"
                f"/N{result['down_n_splits']}/BK{result['down_block_k']}")
        print(f"| {result['rows']} | {down} | N{result['n_splits']} | {' | '.join(cells)} |", flush=True)


def run_suite(args, output):
    result = {"complete": False, "phase": "correctness", "batches": args.batches, "gpu": args.gpu,
              "seed": args.seed, "checks": [], "performance": [], "check_only": args.check_only,
              "protocol": {"buffers": args.buffers, "warmup": args.warmup, "iters": args.iters},
              "settings_written": False}
    active_rows = None
    try:
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


@pytest.mark.parametrize("rows", (0, 1, 31, 33, 63, 65, 127, 129, 255, 257, 511, 513,
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