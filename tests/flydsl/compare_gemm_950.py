#!/usr/bin/env python3
"""Benchmark and validate current GFX950 FlyDSL GEMMs.

The benchmark command compares FlyDSL kernels with local pyhip JIT FP8 kernels.
The validate command runs the functional matrix, and verify-8wave checks the
8-wave source and final-ISA pipeline contract.
"""

from __future__ import annotations

import argparse
import ast
import csv
import re
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TILES = ((128, 128), (128, 256), (256, 128), (256, 256))
DEFAULT_SOURCE = ROOT / "tests/flydsl/test_gemm.py"
DEFAULT_ASM = ROOT / "tests/flydsl/asm/gfx950_current/gemm_8wave_bf16_m4096_n4096_k4096.s"


@dataclass
class BenchmarkCase:
    name: str
    family: str
    waves: int
    tile_m: int
    tile_n: int
    input_dtype: str
    output_dtype: str
    launch: Callable[[], None]
    output: object
    reference: object
    rtol: float
    atol: float


def parse_int_values(value: str) -> tuple[int, ...]:
    values = tuple(int(item) for item in value.split(",") if item)
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated integer")
    return values


def parse_tiles(value: str) -> tuple[tuple[int, int], ...]:
    tiles = []
    for item in value.split(","):
        match = item.lower().split("x")
        if len(match) != 2:
            raise argparse.ArgumentTypeError(f"invalid tile: {item}")
        tiles.append((int(match[0]), int(match[1])))
    if not tiles:
        raise argparse.ArgumentTypeError("expected at least one tile")
    return tuple(tiles)


def parse_args(argv=None) -> argparse.Namespace:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    commands = {"benchmark", "validate", "verify-8wave"}
    if not raw_args or (raw_args[0] not in commands and raw_args[0] not in {"-h", "--help"}):
        raw_args.insert(0, "benchmark")

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="benchmark FlyDSL and local pyhip JIT kernels",
    )
    benchmark_parser.add_argument("--m", type=int, default=4096)
    benchmark_parser.add_argument("--n", type=int, default=4096)
    benchmark_parser.add_argument("--k-values", type=parse_int_values, default=(4096,))
    benchmark_parser.add_argument("--waves", choices=("4", "8", "all"), default="all")
    benchmark_parser.add_argument("--dtype", choices=("bf16", "fp8", "all"), default="all")
    benchmark_parser.add_argument(
        "--tiles",
        type=parse_tiles,
        default=DEFAULT_TILES,
        help="comma-separated tiles such as 128x128,256x256",
    )
    benchmark_parser.add_argument(
        "--jit-layout",
        choices=("preshuffle", "row-major", "both", "none"),
        default="both",
        help="local pyhip JIT variants; only available for 8-wave FP8 at tile 256x256",
    )
    benchmark_parser.add_argument("--warmup", type=int, default=20)
    benchmark_parser.add_argument("--rounds", type=int, default=24)
    benchmark_parser.add_argument("--iterations", type=int, default=100)
    benchmark_parser.add_argument("--skip-correctness", action="store_true")
    benchmark_parser.add_argument("--csv", type=Path, default=None)

    validate_parser = subparsers.add_parser(
        "validate",
        help="run pytest and the full functional matrix",
    )
    validate_parser.add_argument("--full-size", type=int, default=4096)
    validate_parser.add_argument("--full-k", type=int, default=4096)
    validate_parser.add_argument("--full-launches", type=int, default=10)
    validate_parser.add_argument("--two-tile-launches", type=int, default=20)
    validate_parser.add_argument("--skip-pytest", action="store_true")
    validate_parser.add_argument("--csv", type=Path, default=None)

    verify_parser = subparsers.add_parser(
        "verify-8wave",
        help="verify the 8-wave source and final-ISA VMEM pipeline",
    )
    verify_parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    verify_parser.add_argument("--asm", type=Path, default=DEFAULT_ASM)

    return parser.parse_args(raw_args)


def selected_values(value: str, choices: tuple):
    return choices if value == "all" else (type(choices[0])(value),)


def make_flydsl_case(module, waves: int, tile_m: int, tile_n: int, dtype: str, m: int, n: int, k: int):
    a, b, output, reference, kernel_args = module._make_gemm_950_problem(dtype, m, n, k)
    launcher = module.compile_gemm_950(tile_m, tile_n, n, k, waves, dtype)
    kernel = module.flyc.compile[{"opt_level": 2}](launcher, *kernel_args)

    def launch():
        kernel(*kernel_args)

    return (
        BenchmarkCase(
            name=f"flydsl-{waves}w-{dtype}-{tile_m}x{tile_n}",
            family="flydsl",
            waves=waves,
            tile_m=tile_m,
            tile_n=tile_n,
            input_dtype="bf16" if dtype == "bf16" else "e4m3fn",
            output_dtype="bf16",
            launch=launch,
            output=output,
            reference=reference,
            rtol=0.1 if dtype == "bf16" else 0.05,
            atol=0.03 if dtype == "bf16" else 0.5,
        ),
        a,
        b,
    )


def make_jit_case(a, b, preshuffle: bool):
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
    return BenchmarkCase(
        name=f"pyhip-jit-8w-fp8-256x256-{suffix}",
        family="pyhip-jit",
        waves=8,
        tile_m=256,
        tile_n=256,
        input_dtype="e4m3fn",
        output_dtype="bf16",
        launch=launch,
        output=output,
        reference=a.to(torch.bfloat16) @ b.to(torch.bfloat16).t(),
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
            timer = cudaPerf(flops=2 * m * n * k * iterations, name=case.name, verbose=0)
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
                "tile_m": case.tile_m,
                "tile_n": case.tile_n,
                "input_dtype": case.input_dtype,
                "output_dtype": case.output_dtype,
                "m": m,
                "n": n,
                "k": k,
                "median_ms": median,
                "q1_ms": q1,
                "q3_ms": q3,
                "tflops": 2 * m * n * k / (median * 1e9),
                "delta_vs_first_pct": (median / baseline - 1.0) * 100.0,
            }
        )
    return rows


def print_rows(rows) -> None:
    print("name,k,median_ms,q1_ms,q3_ms,tflops,delta_vs_first_pct")
    for row in rows:
        print(
            f"{row['name']},{row['k']},{row['median_ms']:.6f},{row['q1_ms']:.6f},"
            f"{row['q3_ms']:.6f},{row['tflops']:.1f},{row['delta_vs_first_pct']:+.3f}"
        )


def write_csv(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_benchmark(args: argparse.Namespace) -> int:
    import torch
    from tests.flydsl import test_gemm as flydsl_gemm

    if not torch.cuda.is_available() or "gfx950" not in torch.cuda.get_device_properties().gcnArchName:
        raise RuntimeError("this benchmark requires gfx950")

    waves_values = selected_values(args.waves, (4, 8))
    dtype_values = selected_values(args.dtype, ("bf16", "fp8"))
    all_rows = []

    for k in args.k_values:
        for dtype in dtype_values:
            cases = []
            jit_inputs = None
            for waves in waves_values:
                for tile_m, tile_n in args.tiles:
                    case, a, b = make_flydsl_case(
                        flydsl_gemm,
                        waves,
                        tile_m,
                        tile_n,
                        dtype,
                        args.m,
                        args.n,
                        k,
                    )
                    cases.append(case)
                    if waves == 8 and dtype == "fp8" and (tile_m, tile_n) == (256, 256):
                        jit_inputs = (a, b)

            if jit_inputs is not None and args.jit_layout != "none":
                a, b = jit_inputs
                if args.jit_layout in ("preshuffle", "both"):
                    cases.append(make_jit_case(a, b, True))
                if args.jit_layout in ("row-major", "both"):
                    cases.append(make_jit_case(a, b, False))

            if not args.skip_correctness:
                for case in cases:
                    check_case(case)
                    print(f"correctness=PASS K={k} {case.name}", flush=True)

            rows = benchmark_group(
                cases,
                args.m,
                args.n,
                k,
                args.warmup,
                args.rounds,
                args.iterations,
            )
            print(f"\n[K={k} dtype={dtype}]", flush=True)
            print_rows(rows)
            all_rows.extend(rows)

    if args.csv is not None:
        write_csv(args.csv, all_rows)
        print(f"csv={args.csv}")
    return 0


def run_validation_matrix(rows, *, m, n, k, waves, tile_m, tile_n, dtype, launches) -> None:
    import flydsl.compiler as flyc
    import torch
    from tests.flydsl.test_gemm import _make_gemm_950_problem, compile_gemm_950

    torch.manual_seed(20260727)
    _, _, output, reference, kernel_args = _make_gemm_950_problem(dtype, m, n, k)
    launcher = compile_gemm_950(tile_m, tile_n, n, k, waves, dtype)
    kernel = flyc.compile[{"opt_level": 2}](launcher, *kernel_args)
    rtol, atol = ((0.1, 0.03) if dtype == "bf16" else (0.05, 0.5))

    for launch in range(launches):
        output.fill_(float("nan"))
        kernel(*kernel_args)
        torch.cuda.synchronize()
        close = torch.isclose(output.float(), reference.float(), rtol=rtol, atol=atol)
        bad = int((~close).sum().item())
        if bad:
            raise AssertionError(
                f"waves={waves} tile={tile_m}x{tile_n} dtype={dtype} "
                f"shape={m}x{n}x{k} launch={launch} bad={bad}"
            )

    rows.append(
        {
            "waves": waves,
            "tile_m": tile_m,
            "tile_n": tile_n,
            "dtype": dtype,
            "m": m,
            "n": n,
            "k": k,
            "launches": launches,
            "status": "PASS",
        }
    )
    print(
        f"PASS waves={waves} tile={tile_m}x{tile_n} dtype={dtype} "
        f"shape={m}x{n}x{k} launches={launches}",
        flush=True,
    )


def run_validation(args: argparse.Namespace) -> int:
    if not args.skip_pytest:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "tests/flydsl/test_gemm.py",
                "-k",
                "gemm_950",
                "-q",
            ],
            cwd=ROOT,
            check=True,
        )

    rows = []
    for waves in (4, 8):
        for tile_m, tile_n in DEFAULT_TILES:
            for dtype in ("bf16", "fp8"):
                run_validation_matrix(
                    rows,
                    m=args.full_size,
                    n=args.full_size,
                    k=args.full_k,
                    waves=waves,
                    tile_m=tile_m,
                    tile_n=tile_n,
                    dtype=dtype,
                    launches=args.full_launches,
                )

    for waves in (4, 8):
        for tile_m, tile_n in DEFAULT_TILES:
            for dtype in ("bf16", "fp8"):
                run_validation_matrix(
                    rows,
                    m=tile_m * 2,
                    n=tile_n * 2,
                    k=128 if dtype == "bf16" else 256,
                    waves=waves,
                    tile_m=tile_m,
                    tile_n=tile_n,
                    dtype=dtype,
                    launches=args.two_tile_launches,
                )

    if args.csv is not None:
        write_csv(args.csv, rows)
        print(f"csv={args.csv}")
    print(f"PASS configurations={len(rows)} launches={sum(row['launches'] for row in rows)}")
    return 0


def _function_node(tree: ast.AST, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise RuntimeError(f"cannot find function {name}")


def _call_name(node: ast.Call) -> str:
    value = node.func
    parts = []
    while isinstance(value, ast.Attribute):
        parts.append(value.attr)
        value = value.value
    if isinstance(value, ast.Name):
        parts.append(value.id)
    return ".".join(reversed(parts))


def _wait_vmcnt_argument(source: str, call: ast.Call) -> str:
    if _call_name(call) != "rocdl.s_waitcnt" or len(call.args) != 1:
        raise AssertionError(f"not an s_waitcnt call: {_call_name(call)}")
    encoded = call.args[0]
    if not isinstance(encoded, ast.Call) or _call_name(encoded) != "encode_waitcnt_950":
        raise AssertionError("s_waitcnt must directly wrap encode_waitcnt_950")
    keyword = next((item for item in encoded.keywords if item.arg == "vmcnt"), None)
    if keyword is None:
        raise AssertionError("s_waitcnt is missing vmcnt")
    return ast.get_source_segment(source, keyword.value)


def verify_8wave_source(path: Path) -> None:
    source = path.read_text()
    kernel = _function_node(ast.parse(source), "gemm_8wave_950")
    begin_compute = _function_node(kernel, "begin_compute_phase")
    end_compute = _function_node(kernel, "end_compute_phase")
    begin_priorities = [
        ast.get_source_segment(source, call.args[0])
        for call in ast.walk(begin_compute)
        if isinstance(call, ast.Call) and _call_name(call) == "rocdl.s_setprio"
    ]
    end_priorities = [
        ast.get_source_segment(source, call.args[0])
        for call in ast.walk(end_compute)
        if isinstance(call, ast.Call) and _call_name(call) == "rocdl.s_setprio"
    ]
    if begin_priorities != ["1"] or end_priorities != ["0"]:
        raise AssertionError(f"priority phases: begin={begin_priorities}, end={end_priorities}")

    loop = next(
        node
        for node in ast.walk(kernel)
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Tuple)
        and any(isinstance(item, ast.Name) and item.id == "state" for item in node.target.elts)
    )

    waits = []
    for statement in loop.body:
        if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
            continue
        call = statement.value
        if _call_name(call) == "rocdl.s_waitcnt":
            waits.append(_wait_vmcnt_argument(source, call))
    expected = ["ab_br_vmcnt", "ab_br_vmcnt", "bl_at_vmcnt", "bl_at_vmcnt"] * 2
    if waits != expected:
        raise AssertionError(f"main-loop wait arguments: expected={expected}, actual={waits}")

    tail_calls = [
        node
        for node in ast.walk(kernel)
        if isinstance(node, ast.Call)
        and node.lineno > loop.end_lineno
        and _call_name(node) == "rocdl.s_waitcnt"
    ]
    tail_calls.sort(key=lambda node: node.lineno)
    tail_arguments = [_wait_vmcnt_argument(source, node) for node in tail_calls]
    expected_tail_arguments = [
        "2 * a_vmem_count + 3 * b_vmem_count",
        "2 * a_vmem_count + 2 * b_vmem_count",
        "2 * a_vmem_count + b_vmem_count",
        "a_vmem_count + b_vmem_count",
        "b_vmem_count",
        "0",
    ]
    if tail_arguments != expected_tail_arguments:
        raise AssertionError(
            f"tail wait arguments: expected={expected_tail_arguments}, actual={tail_arguments}"
        )

    assignments = {}
    for statement in kernel.body:
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            continue
        target = statement.targets[0]
        if isinstance(target, ast.Name) and target.id in {"prologue_vmcnt", "ab_br_vmcnt", "bl_at_vmcnt"}:
            assignments[target.id] = ast.get_source_segment(source, statement.value)
    expected_assignments = {
        "prologue_vmcnt": "3 * a_vmem_count + 3 * b_vmem_count",
        "ab_br_vmcnt": "2 * a_vmem_count + 3 * b_vmem_count",
        "bl_at_vmcnt": "3 * a_vmem_count + 2 * b_vmem_count",
    }
    if assignments != expected_assignments:
        raise AssertionError(f"wait formulas: expected={expected_assignments}, actual={assignments}")


def _hot_loop(lines: list[str]) -> tuple[int, int, list[str]]:
    labels = {}
    for index, line in enumerate(lines):
        if match := re.match(r"^(\.LBB\d+_\d+):", line):
            labels[match.group(1)] = index
    candidates = []
    for end, line in enumerate(lines):
        match = re.search(r"s_cbranch_\w+\s+(\.LBB\d+_\d+)", line)
        if match and match.group(1) in labels and labels[match.group(1)] < end:
            start = labels[match.group(1)]
            candidates.append((end - start, start, end))
    if not candidates:
        raise AssertionError("cannot find hot-loop backedge")
    _, start, end = max(candidates)
    return start, end, lines[start : end + 1]


def _waits(lines: list[str]) -> list[int]:
    return [int(value) for line in lines for value in re.findall(r"s_waitcnt vmcnt\((\d+)\)", line)]


def verify_8wave_assembly(path: Path) -> None:
    lines = path.read_text().splitlines()
    loop_start, loop_end, hot = _hot_loop(lines)
    prologue = _waits(lines[:loop_start])
    main = _waits(hot)
    tail = _waits(lines[loop_end + 1 :])
    if not prologue or prologue[-1] != 12:
        raise AssertionError(f"prologue waits: {prologue}")
    if main != [10] * 8:
        raise AssertionError(f"main-loop waits: {main}")
    if not any(tail[index : index + 6] == [10, 8, 6, 4, 2, 0] for index in range(len(tail) - 5)):
        raise AssertionError(f"tail waits: {tail}")

    wait_indices = [index for index, line in enumerate(hot) if re.search(r"s_waitcnt vmcnt\(10\)", line)]
    dma_counts = []
    phases = []
    for index, wait_index in enumerate(wait_indices):
        next_wait = wait_indices[(index + 1) % len(wait_indices)]
        interval = (
            hot[wait_index + 1 : next_wait]
            if next_wait > wait_index
            else hot[wait_index + 1 :] + hot[:next_wait]
        )
        dma_counts.append(sum("buffer_load_dwordx4" in line and "lds" in line for line in interval))
        events = []
        for line in interval:
            if re.search(r"\bs_barrier\b", line):
                events.append("B")
            elif "v_mfma" in line and (not events or events[-1] != "M"):
                events.append("M")
            elif re.search(r"\bds_read", line) and (not events or events[-1] != "D"):
                events.append("D")
            elif "buffer_load_dwordx4" in line and "lds" in line and (not events or events[-1] != "V"):
                events.append("V")
        phases.append(events)
    if dma_counts != [2] * 8:
        raise AssertionError(f"DMA loads between waits: {dma_counts}")
    if sum("s_barrier" in line for line in hot) != 16:
        raise AssertionError("main loop must contain 16 barriers")
    if not all(phase[:3] == ["B", "M", "B"] for phase in phases):
        raise AssertionError(f"region phases: {phases}")

    control_events = []
    for line in hot:
        if match := re.search(r"s_waitcnt vmcnt\((\d+)\)", line):
            control_events.append(f"W{match.group(1)}")
        elif match := re.search(r"s_setprio\s+(\d+)", line):
            control_events.append(f"P{match.group(1)}")
        elif re.search(r"\bs_barrier\b", line):
            control_events.append("B")
        elif "v_mfma" in line and (not control_events or control_events[-1] != "M"):
            control_events.append("M")
    expected_control_events = ["W10", "P1", "B", "M", "P0", "B"] * 8
    if control_events != expected_control_events:
        raise AssertionError(f"priority phases: {control_events}")

    print(f"prologue vmcnt: {prologue[-1]}")
    print(f"main-loop vmcnt: {main}")
    print(f"DMA loads between waits: {dma_counts}")
    print("region phase: wait -> prio(1) -> barrier -> MFMA -> prio(0) -> barrier -> LDS/DMA")
    print("tail vmcnt: 10, 8, 6, 4, 2, 0")


def run_verify_8wave(args: argparse.Namespace) -> int:
    verify_8wave_source(args.source)
    verify_8wave_assembly(args.asm)
    print("PASS: 8-wave source formulas and ISA VMEM group ledger are correct.")
    return 0


def main(argv=None) -> int:
    args = parse_args(argv)
    sys.path.insert(0, str(ROOT))
    if args.command == "benchmark":
        return run_benchmark(args)
    if args.command == "validate":
        return run_validation(args)
    if args.command == "verify-8wave":
        return run_verify_8wave(args)
    raise AssertionError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
