#!/usr/bin/env python3
# SPDX-License-Identifier: MIT

"""Measure per-kernel HBM traffic and bandwidth with rocprofv3.

Examples:
    python prof-hbm.py test_a8w4.py
    python prof-hbm.py --kernel-regex 'flydsl|opus' test_a8w4.py
    python prof-hbm.py -k 'gemm' test_gemm.py -- --shape 4096 4096 4096

Profiler options must precede TARGET. Arguments following TARGET are passed to it;
use ``--`` when a target argument could be confused with a profiler option.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

READ_COUNTER = "TCC_EA0_RDREQ_DRAM_32B_sum"
WRITE_COUNTER = "TCC_EA0_WRREQ_WRITE_DRAM_32B_sum"
BYTES_PER_REQUEST = 32.0
OUTPUT_BASENAME = "hbm"


@dataclass(frozen=True)
class DispatchSample:
    kernel_name: str
    start_timestamp_ns: int
    duration_ns: int
    read_bytes: float
    write_bytes: float

    @property
    def bandwidth_gb_s(self) -> float:
        return (self.read_bytes + self.write_bytes) / self.duration_ns


@dataclass(frozen=True)
class KernelSummary:
    kernel_name: str
    duration_us: float
    read_gb: float
    write_gb: float
    total_gb: float
    bandwidth_gb_s: float


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a Python script under rocprofv3 and report actual HBM traffic "
            "for every captured kernel."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "The name filter is applied only while printing; rocprofv3 still "
            "collects both counters for every kernel.\n"
            "Example: %(prog)s -k 'gemm|reduce' workload.py -- --size 4096"
        ),
    )
    parser.add_argument(
        "-k",
        "--kernel-regex",
        default=".*",
        help="print kernel names matching this regular expression (default: all)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        help="rocprofv3 output directory (default: a timestamped directory)",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter used for TARGET (default: current interpreter)",
    )
    parser.add_argument("target", type=Path, help="Python script to profile")
    parser.add_argument(
        "target_args",
        nargs=argparse.REMAINDER,
        help="arguments passed unchanged to TARGET",
    )
    args = parser.parse_args(argv)

    try:
        args.kernel_pattern = re.compile(args.kernel_regex)
    except re.error as exc:
        parser.error(f"invalid --kernel-regex: {exc}")

    args.target = args.target.expanduser().resolve()
    if not args.target.is_file():
        parser.error(f"target script does not exist: {args.target}")

    if args.output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        args.output_dir = Path.cwd() / f"rocprof-hbm-{stamp}-{os.getpid()}"
    else:
        args.output_dir = args.output_dir.expanduser().resolve()

    if args.output_dir.exists():
        parser.error(f"output directory already exists: {args.output_dir}")

    return args


def run_rocprof(args: argparse.Namespace) -> subprocess.CompletedProcess[bytes]:
    rocprof = shutil.which("rocprofv3")
    if rocprof is None:
        raise RuntimeError("rocprofv3 was not found in PATH")

    command = [
        rocprof,
        "--pmc",
        READ_COUNTER,
        WRITE_COUNTER,
        "-f",
        "csv",
        "-d",
        str(args.output_dir),
        "-o",
        OUTPUT_BASENAME,
        "--",
        args.python,
        str(args.target),
        *args.target_args,
    ]
    print("Running:", shlex.join(command), flush=True)
    return subprocess.run(command, check=False)


def _parse_int(row: dict[str, str], field: str, line_number: int) -> int:
    try:
        return int(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"CSV line {line_number}: invalid {field!r}") from exc


def _parse_float(row: dict[str, str], field: str, line_number: int) -> float:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"CSV line {line_number}: invalid {field!r}") from exc
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"CSV line {line_number}: invalid {field!r} value {value}")
    return value


def parse_counter_csv(csv_path: Path) -> tuple[list[DispatchSample], list[str]]:
    dispatches: dict[tuple[str, str, str, str], dict[str, object]] = {}
    warnings: list[str] = []

    with csv_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {
            "Process_Id",
            "Agent_Id",
            "Queue_Id",
            "Dispatch_Id",
            "Kernel_Name",
            "Counter_Name",
            "Counter_Value",
            "Start_Timestamp",
            "End_Timestamp",
        }
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(
                "counter CSV is missing columns: " + ", ".join(sorted(missing))
            )

        for line_number, row in enumerate(reader, start=2):
            counter_name = row["Counter_Name"]
            if counter_name not in (READ_COUNTER, WRITE_COUNTER):
                continue

            key = (
                row["Process_Id"],
                row["Agent_Id"],
                row["Queue_Id"],
                row["Dispatch_Id"],
            )
            kernel_name = row["Kernel_Name"]
            start_ns = _parse_int(row, "Start_Timestamp", line_number)
            end_ns = _parse_int(row, "End_Timestamp", line_number)
            if end_ns <= start_ns:
                warnings.append(
                    f"CSV line {line_number}: skipped non-positive duration for "
                    f"dispatch {row['Dispatch_Id']}"
                )
                continue

            entry = dispatches.setdefault(
                key,
                {
                    "kernel_name": kernel_name,
                    "start_ns": start_ns,
                    "end_ns": end_ns,
                    "counters": {},
                },
            )
            if (
                entry["kernel_name"] != kernel_name
                or entry["start_ns"] != start_ns
                or entry["end_ns"] != end_ns
            ):
                raise ValueError(
                    f"CSV line {line_number}: inconsistent metadata for dispatch "
                    f"{row['Dispatch_Id']}"
                )

            counters = entry["counters"]
            assert isinstance(counters, dict)
            if counter_name in counters:
                raise ValueError(
                    f"CSV line {line_number}: duplicate {counter_name} for "
                    f"dispatch {row['Dispatch_Id']}"
                )
            counters[counter_name] = _parse_float(row, "Counter_Value", line_number)

    samples: list[DispatchSample] = []
    for key, entry in dispatches.items():
        counters = entry["counters"]
        assert isinstance(counters, dict)
        missing_counters = {READ_COUNTER, WRITE_COUNTER}.difference(counters)
        if missing_counters:
            warnings.append(
                f"dispatch {key[-1]} ({entry['kernel_name']}): skipped because "
                f"{', '.join(sorted(missing_counters))} is missing"
            )
            continue
        start_ns = entry["start_ns"]
        end_ns = entry["end_ns"]
        assert isinstance(start_ns, int) and isinstance(end_ns, int)
        samples.append(
            DispatchSample(
                kernel_name=str(entry["kernel_name"]),
                start_timestamp_ns=start_ns,
                duration_ns=end_ns - start_ns,
                read_bytes=float(counters[READ_COUNTER]) * BYTES_PER_REQUEST,
                write_bytes=float(counters[WRITE_COUNTER]) * BYTES_PER_REQUEST,
            )
        )

    samples.sort(key=lambda sample: sample.start_timestamp_ns)
    return samples, warnings


def summarize(samples: Iterable[DispatchSample]) -> list[KernelSummary]:
    return [
        KernelSummary(
            kernel_name=sample.kernel_name,
            duration_us=sample.duration_ns / 1.0e3,
            read_gb=sample.read_bytes / 1.0e9,
            write_gb=sample.write_bytes / 1.0e9,
            total_gb=(sample.read_bytes + sample.write_bytes) / 1.0e9,
            bandwidth_gb_s=sample.bandwidth_gb_s,
        )
        for sample in samples
    ]


def print_summaries(
    summaries: Sequence[KernelSummary], pattern: re.Pattern[str]
) -> int:
    selected = [item for item in summaries if pattern.search(item.kernel_name)]
    headers = (
        "Kernel prefix",
        "Time (us)",
        "BW",
        "Total",
        "Read",
        "Write",
        "Kernel",
    )
    rows = [
        (
            item.kernel_name.replace("\n", " ")[:50],
            f"{item.duration_us:.3f}",
            str(int(item.bandwidth_gb_s)),
            f"{item.total_gb:.3f}",
            f"{item.read_gb:.3f}",
            f"{item.write_gb:.3f}",
            item.kernel_name.replace("\n", " "),
        )
        for item in selected
    ]
    widths = [
        max([len(headers[index]), *(len(row[index]) for row in rows)])
        for index in range(len(headers) - 1)
    ]

    def format_row(row: Sequence[str]) -> str:
        columns = [row[0].ljust(widths[0])]
        columns.extend(
            value.rjust(width) for value, width in zip(row[1:-1], widths[1:])
        )
        return "  ".join((*columns, row[-1]))

    print("\n" + format_row(headers))
    print(format_row(tuple("-" * width for width in widths) + ("------",)))
    for row in rows:
        print(format_row(row))
    print(
        f"\nPrinted {len(selected)} of {len(summaries)} kernel dispatches "
        f"matching /{pattern.pattern}/."
    )
    return len(selected)


def find_counter_csv(output_dir: Path) -> Path:
    expected = output_dir / f"{OUTPUT_BASENAME}_counter_collection.csv"
    if expected.is_file():
        return expected
    matches = sorted(output_dir.glob("*_counter_collection.csv"))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"no counter collection CSV found in {output_dir}")
    raise RuntimeError(
        f"multiple counter collection CSV files found in {output_dir}: "
        + ", ".join(str(path) for path in matches)
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        completed = run_rocprof(args)
    except (OSError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    try:
        csv_path = find_counter_csv(args.output_dir)
        samples, warnings = parse_counter_csv(csv_path)
        summaries = summarize(samples)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: failed to analyze rocprofv3 output: {exc}", file=sys.stderr)
        return completed.returncode or 1

    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)

    if not samples:
        print("error: no complete HBM counter samples were captured", file=sys.stderr)
        return completed.returncode or 1

    print(f"\nCounter CSV: {csv_path}")
    print(
        f"Traffic = 32 * ({READ_COUNTER} + {WRITE_COUNTER}); "
        "Total/Read/Write use decimal GB; BW uses decimal GB/s."
    )
    print_summaries(summaries, args.kernel_pattern)

    if completed.returncode:
        print(
            f"warning: rocprofv3/target exited with status {completed.returncode}",
            file=sys.stderr,
        )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
