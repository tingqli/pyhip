#!/usr/bin/env python3
"""Compare gfx950 GEMM kernels with one cudaPerf timing harness.

The timed closures only launch kernels. Input conversion, B transpose/preshuffle,
output allocation, compilation, and correctness checks happen before timing.

The FP8 paths are not bitwise-equivalent formats:
  * FlyDSL and pyhip JIT use E4M3FN inputs and BF16 output.
  * gfx950-gluon-tutorials uses E5M2 inputs and FP16 output.
The table reports that distinction explicitly while comparing kernel throughput.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import statistics
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[2]
GLUON_ENV_VARS = (
    "LLVM_PASS_PLUGIN_PATH",
    "LLVM_PASS_PLUGIN_KEEP_TARGET_MACHINE",
    "TRITON_FORCE_MFMA_AGPR",
    "TRITON_AMDGCNAS_PLUGIN",
)


@dataclass
class BenchmarkCase:
    name: str
    family: str
    waves: int
    input_dtype: str
    output_dtype: str
    launch: Callable[[], None]
    output: object
    reference: object
    rtol: float
    atol: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=4096)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--waves", choices=("4", "8", "all"), default="all")
    parser.add_argument("--dtype", choices=("bf16", "fp8", "all"), default="all")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument(
        "--gluon-repo",
        type=Path,
        default=None,
        help="gfx950-gluon-tutorials checkout (or set GFX950_GLUON_TUTORIALS)",
    )
    parser.add_argument(
        "--gluon-config",
        choices=("auto", "base", "full"),
        default="auto",
        help="auto uses full for 4-wave and base/no-AGPR for 8-wave",
    )
    parser.add_argument(
        "--llir-plugin",
        type=Path,
        default=None,
        help="LLIR scheduler .so matched to the active Triton LLVM (or set GFX950_LLIR_PLUGIN)",
    )
    parser.add_argument(
        "--local-fp8-layout",
        choices=("preshuffle", "row-major", "both", "none"),
        default="both",
        help="local tests/contrib/gemm/test_fp8_8wave.py variants to include",
    )
    parser.add_argument("--skip-correctness", action="store_true")
    parser.add_argument("--csv", type=Path, default=None)
    return parser.parse_args()


def resolve_gluon_repo(requested: Path | None) -> Path:
    candidates = [
        requested,
        Path(os.environ["GFX950_GLUON_TUTORIALS"]) if os.environ.get("GFX950_GLUON_TUTORIALS") else None,
        Path("/host_lc/gfx950-gluon-tutorials"),
        Path("/tmp/gfx950-gluon-tutorials"),
    ]
    for candidate in candidates:
        if candidate is not None and (candidate / "kernels/gemm").is_dir():
            return candidate.resolve()
    raise FileNotFoundError("gfx950-gluon-tutorials not found; pass --gluon-repo or set " "GFX950_GLUON_TUTORIALS")


def configure_gluon(repo: Path, config: str, requested_plugin: Path | None) -> None:
    for name in GLUON_ENV_VARS:
        os.environ.pop(name, None)
    if config != "full":
        return
    plugin = requested_plugin
    if plugin is None and os.environ.get("GFX950_LLIR_PLUGIN"):
        plugin = Path(os.environ["GFX950_LLIR_PLUGIN"])
    if plugin is None:
        plugin = repo / "plugins/llir_scheduler/libLlirSched.so"
    if not plugin.is_file():
        raise FileNotFoundError(f"LLIR scheduler plugin not found: {plugin}")
    os.environ["LLVM_PASS_PLUGIN_PATH"] = str(plugin.resolve())
    os.environ["LLVM_PASS_PLUGIN_KEEP_TARGET_MACHINE"] = "1"
    os.environ["TRITON_FORCE_MFMA_AGPR"] = "1"
    os.environ["TRITON_AMDGCNAS_PLUGIN"] = "1"


def load_triton(global_symbols: bool):
    if not global_symbols:
        import triton

        return triton

    import ctypes

    old_flags = sys.getdlopenflags()
    try:
        sys.setdlopenflags(os.RTLD_NOW | os.RTLD_GLOBAL)
        import triton
        import triton._C.libtriton as libtriton

        ctypes.CDLL(libtriton.__file__, mode=os.RTLD_NOW | os.RTLD_GLOBAL)
    finally:
        sys.setdlopenflags(old_flags)
    return triton


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_gluon_modules(repo: Path, config: str):
    triton = load_triton(global_symbols=config == "full")

    utils = repo / "kernels/gemm/utils"
    sys.path.insert(0, str(utils))
    if config == "full":
        plugin_dir = repo / "plugins/amdgcnas"
        sys.path.insert(0, str(plugin_dir))
        import amdgcnas_plugin
        from triton import knobs

        knobs.runtime.add_stages_inspection_hook = amdgcnas_plugin.inspect_stages_hook

    paths = {
        (4, "bf16"): repo / "kernels/gemm/intra_wave/a16w16/v9_beyond_hotloop/matmul_kernel.py",
        (8, "bf16"): repo / "kernels/gemm/inter_wave/a16w16/matmul_kernel.py",
        (4, "fp8"): repo / "kernels/gemm/intra_wave/a8w8/matmul_kernel.py",
        (8, "fp8"): repo / "kernels/gemm/inter_wave/a8w8/matmul_kernel.py",
    }
    modules = {key: load_module(f"gfx950_gluon_{key[0]}w_{key[1]}", path) for key, path in paths.items()}
    return triton, modules


def selected_values(value: str, choices: tuple):
    return choices if value == "all" else (type(choices[0])(value),)


def make_flydsl_case(module, waves: int, dtype: str, m: int, n: int, k: int):
    a, b, output, reference, kernel_args = module._make_gemm_950_problem(dtype, m, n, k)
    launcher = module.compile_gemm_950(256, 256, n, k, waves, dtype)
    kernel = module.flyc.compile[{"opt_level": 2}](launcher, *kernel_args)

    def launch():
        kernel(*kernel_args)

    case = BenchmarkCase(
        name=f"flydsl-{waves}w-{dtype}",
        family="flydsl",
        waves=waves,
        input_dtype="bf16" if dtype == "bf16" else "e4m3fn",
        output_dtype="bf16",
        launch=launch,
        output=output,
        reference=reference,
        rtol=0.1 if dtype == "bf16" else 0.05,
        atol=0.03 if dtype == "bf16" else 0.5,
    )
    return case, a, b


def make_gluon_case(triton, module, waves: int, dtype: str, a, b):
    import torch

    m, k = a.shape
    n = b.shape[0]
    if dtype == "bf16":
        output = torch.empty((m, n), device=a.device, dtype=torch.bfloat16)
        b_logical = b.t()
        reference = a @ b_logical
        grid_mn = triton.cdiv(m, 256) * triton.cdiv(n, 256)
        if waves == 4:

            def launch():
                module.v9_beyond_hotloop[(grid_mn, 1)](
                    a,
                    b_logical,
                    output,
                    m,
                    n,
                    k,
                    a.stride(0),
                    a.stride(1),
                    b_logical.stride(0),
                    b_logical.stride(1),
                    output.stride(0),
                    output.stride(1),
                    BLOCK_M=256,
                    BLOCK_N=256,
                    BLOCK_K=64,
                    GRID_MN=grid_mn,
                    NUM_XCDS=8,
                    GROUP_SIZE_M=4,
                    num_warps=4,
                    llvm_fn_attrs=("amdgpu-agpr-alloc=256" if os.environ.get("TRITON_FORCE_MFMA_AGPR") else ""),
                )

        else:

            def launch():
                module.a16w16_kernel[(grid_mn,)](
                    a,
                    b,
                    output,
                    m,
                    n,
                    k,
                    a.stride(0),
                    a.stride(1),
                    b.stride(1),
                    b.stride(0),
                    output.stride(0),
                    output.stride(1),
                    BLOCK_M=256,
                    BLOCK_N=256,
                    BLOCK_K=64,
                    WARPS_M=2,
                    WARPS_N=4,
                    GRID_MN=grid_mn,
                    NUM_XCDS=8,
                    GROUP_SIZE_M=4,
                    num_warps=8,
                    llvm_fn_attrs=(("amdgpu-agpr-alloc", "0,0"),),
                )

        return BenchmarkCase(
            name=f"gluon-{waves}w-bf16",
            family="gluon",
            waves=waves,
            input_dtype="bf16",
            output_dtype="bf16",
            launch=launch,
            output=output,
            reference=reference,
            rtol=0.1,
            atol=0.1,
        )

    a_bf8 = a.to(torch.float8_e5m2)
    b_bf8 = b.to(torch.float8_e5m2)
    output = torch.empty((m, n), device=a.device, dtype=torch.float16)
    reference = a_bf8.to(torch.float16) @ b_bf8.to(torch.float16).t()
    if waves == 4:
        b_logical = b_bf8.t()
        grid_mn = triton.cdiv(m, 256) * triton.cdiv(n, 256)

        def launch():
            module.a8w8_kernel[(grid_mn, 1)](
                a_bf8,
                b_logical,
                output,
                m,
                n,
                k,
                a_bf8.stride(0),
                a_bf8.stride(1),
                b_logical.stride(0),
                b_logical.stride(1),
                output.stride(0),
                output.stride(1),
                BLOCK_M=256,
                BLOCK_N=256,
                BLOCK_K=128,
                GRID_MN=grid_mn,
                NUM_XCDS=8,
                GROUP_SIZE_M=4,
                num_warps=4,
                llvm_fn_attrs=("amdgpu-agpr-alloc=256" if os.environ.get("TRITON_FORCE_MFMA_AGPR") else ""),
            )

    else:

        def launch():
            grid_mn = triton.cdiv(m, 256) * triton.cdiv(n, 256)
            module.a8w8_kernel[(grid_mn,)](
                a_bf8,
                b_bf8,
                output,
                m,
                n,
                k,
                a_bf8.stride(0),
                a_bf8.stride(1),
                b_bf8.stride(1),
                b_bf8.stride(0),
                output.stride(0),
                output.stride(1),
                BLOCK_M=256,
                BLOCK_N=256,
                BLOCK_K=128,
                WARPS_M=2,
                WARPS_N=4,
                GRID_MN=grid_mn,
                NUM_XCDS=8,
                GROUP_SIZE_M=4,
                num_warps=8,
                llvm_fn_attrs=(("amdgpu-agpr-alloc", "0,0"),),
            )

    return BenchmarkCase(
        name=f"gluon-{waves}w-bf8",
        family="gluon",
        waves=waves,
        input_dtype="e5m2",
        output_dtype="fp16",
        launch=launch,
        output=output,
        reference=reference,
        rtol=0.0,
        atol=0.5,
    )


def make_local_fp8_case(a, b, preshuffle: bool):
    import pyhip
    import torch
    from pyhip.contrib.gemm_fp8 import gemm_8wave_fp8bf16fp16

    m, k = a.shape
    n = b.shape[0]
    output = torch.empty((m, n), device=a.device, dtype=torch.bfloat16)
    b_arg = pyhip.pre_shuffle(b, mfma_MN=16) if preshuffle else b
    grid = [pyhip.div_up(m, 256) * pyhip.div_up(n, 256)]

    def launch():
        gemm_8wave_fp8bf16fp16(
            grid,
            [512],
            "fp8",
            preshuffle,
            False,
            256,
            256,
            n,
            k,
            a.data_ptr(),
            b_arg.data_ptr(),
            output.data_ptr(),
            None,
            None,
            m,
        )

    suffix = "preshuffle" if preshuffle else "row-major"
    reference = a.to(torch.bfloat16) @ b.to(torch.bfloat16).t()
    return BenchmarkCase(
        name=f"pyhip-jit-8w-e4m3fn-{suffix}",
        family="pyhip-jit",
        waves=8,
        input_dtype="e4m3fn",
        output_dtype="bf16",
        launch=launch,
        output=output,
        reference=reference,
        rtol=0.05,
        atol=0.5,
    )


def check_case(case: BenchmarkCase) -> None:
    import torch

    case.output.fill_(float("nan"))
    case.launch()
    torch.cuda.synchronize()
    torch.testing.assert_close(
        case.output.float(),
        case.reference.float(),
        rtol=case.rtol,
        atol=case.atol,
    )


def quartiles(values: list[float]):
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0], ordered[0], ordered[0]
    half = len(ordered) // 2
    return (
        statistics.median(ordered),
        statistics.median(ordered[:half]),
        statistics.median(ordered[half:]),
    )


def benchmark_group(cases, m: int, n: int, k: int, warmup: int, rounds: int, iterations: int):
    import torch
    from pyhip import cudaPerf

    for _ in range(warmup):
        for case in cases:
            case.launch()
    torch.cuda.synchronize()

    samples = {case.name: [] for case in cases}
    for round_index in range(rounds):
        shift = round_index % len(cases)
        order = cases[shift:] + cases[:shift]
        if (round_index // len(cases)) % 2:
            order = list(reversed(order))
        for case in order:
            timer = cudaPerf(
                flops=2 * m * n * k * iterations,
                name=case.name,
                verbose=0,
            )
            with timer:
                for _ in range(iterations):
                    case.launch()
            samples[case.name].append(timer.dt() * 1e3 / iterations)

    baseline = quartiles(samples[cases[0].name])[0]
    rows = []
    for case in cases:
        median, q1, q3 = quartiles(samples[case.name])
        rows.append(
            {
                "name": case.name,
                "family": case.family,
                "waves": case.waves,
                "input_dtype": case.input_dtype,
                "output_dtype": case.output_dtype,
                "m": m,
                "n": n,
                "k": k,
                "median_ms": median,
                "q1_ms": q1,
                "q3_ms": q3,
                "tflops": 2 * m * n * k / (median * 1e9),
                "delta_vs_flydsl_pct": (median / baseline - 1.0) * 100.0,
            }
        )
    return rows


def print_rows(rows) -> None:
    print("name,waves,input_dtype,output_dtype,median_ms,q1_ms,q3_ms,tflops," "delta_vs_flydsl_pct")
    for row in rows:
        print(
            f"{row['name']},{row['waves']},{row['input_dtype']},{row['output_dtype']},"
            f"{row['median_ms']:.6f},{row['q1_ms']:.6f},{row['q3_ms']:.6f},"
            f"{row['tflops']:.1f},{row['delta_vs_flydsl_pct']:+.3f}"
        )


def write_csv(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_auto(args: argparse.Namespace) -> int:
    repo = resolve_gluon_repo(args.gluon_repo)
    with tempfile.TemporaryDirectory(prefix="gfx950-gemm-compare-") as temp_dir:
        temp_dir = Path(temp_dir)
        common = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--m",
            str(args.m),
            "--n",
            str(args.n),
            "--k",
            str(args.k),
            "--dtype",
            args.dtype,
            "--warmup",
            str(args.warmup),
            "--rounds",
            str(args.rounds),
            "--iterations",
            str(args.iterations),
            "--gluon-repo",
            str(repo),
            "--local-fp8-layout",
            args.local_fp8_layout,
        ]
        if args.llir_plugin is not None:
            common.extend(["--llir-plugin", str(args.llir_plugin)])
        if args.skip_correctness:
            common.append("--skip-correctness")

        rows = []
        for waves, config in (("4", "full"), ("8", "base")):
            child_csv = temp_dir / f"{waves}wave.csv"
            command = [
                *common,
                "--waves",
                waves,
                "--gluon-config",
                config,
                "--csv",
                str(child_csv),
            ]
            child_env = os.environ.copy()
            for name in GLUON_ENV_VARS:
                child_env.pop(name, None)
            subprocess.run(command, check=True, env=child_env)
            with child_csv.open(newline="") as stream:
                for row in csv.DictReader(stream):
                    for key in ("waves", "m", "n", "k"):
                        row[key] = int(row[key])
                    for key in (
                        "median_ms",
                        "q1_ms",
                        "q3_ms",
                        "tflops",
                        "delta_vs_flydsl_pct",
                    ):
                        row[key] = float(row[key])
                    rows.append(row)

        print("\n[combined auto: 4-wave full, 8-wave base]")
        print_rows(rows)
        if args.csv is not None:
            write_csv(args.csv, rows)
            print(f"csv={args.csv}")
    return 0


def main() -> int:
    args = parse_args()
    if args.gluon_config == "auto":
        if args.waves == "all":
            return run_auto(args)
        args.gluon_config = "full" if args.waves == "4" else "base"
    if args.gluon_config == "full" and args.waves in ("8", "all"):
        raise ValueError("8-wave Gluon requires base/no-AGPR; use --gluon-config auto or base")

    repo = resolve_gluon_repo(args.gluon_repo)

    sys.path.insert(0, str(ROOT))
    import torch

    if not torch.cuda.is_available() or "gfx950" not in torch.cuda.get_device_properties().gcnArchName:
        raise RuntimeError("this benchmark requires gfx950")

    from tests.flydsl import test_gemm as flydsl_gemm

    waves_values = selected_values(args.waves, (4, 8))
    dtype_values = selected_values(args.dtype, ("bf16", "fp8"))
    flydsl_cases = {}
    for dtype in dtype_values:
        for waves in waves_values:
            flydsl_cases[(waves, dtype)] = make_flydsl_case(flydsl_gemm, waves, dtype, args.m, args.n, args.k)

    configure_gluon(repo, args.gluon_config, args.llir_plugin)
    triton, gluon_modules = load_gluon_modules(repo, args.gluon_config)

    all_rows = []
    print(f"gluon_repo={repo}")
    print(f"gluon_commit={os.popen(f'git -C {repo} rev-parse HEAD').read().strip()}")
    print(f"gluon_config={args.gluon_config}")

    for dtype in dtype_values:
        for waves in waves_values:
            flydsl_case, a, b = flydsl_cases[(waves, dtype)]
            cases = [
                flydsl_case,
                make_gluon_case(triton, gluon_modules[(waves, dtype)], waves, dtype, a, b),
            ]
            if dtype == "fp8" and waves == 8 and args.local_fp8_layout != "none":
                if args.local_fp8_layout in ("preshuffle", "both"):
                    cases.append(make_local_fp8_case(a, b, True))
                if args.local_fp8_layout in ("row-major", "both"):
                    cases.append(make_local_fp8_case(a, b, False))

            if not args.skip_correctness:
                for case in cases:
                    check_case(case)
                    print(f"correctness=PASS {case.name}")
            rows = benchmark_group(
                cases,
                args.m,
                args.n,
                args.k,
                args.warmup,
                args.rounds,
                args.iterations,
            )
            print(f"\n[{waves}-wave {dtype}]")
            print_rows(rows)
            all_rows.extend(rows)

    if args.csv is not None:
        write_csv(args.csv, all_rows)
        print(f"csv={args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
