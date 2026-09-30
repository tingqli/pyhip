# Copyright 2023-2024 SGLang Team
# SPDX-License-Identifier: Apache-2.0
"""Single-kernel correctness tests and explicit --check-only / --perf CLI.

Performance uses ten independent buffers and the shared gated cudaPerf runner.
Component medians are not a whole-attention latency. No timing runs at import.
"""

import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tests.ops.qsa import _attention as helpers
from tests.ops.qsa._attention_kernels import SCOPES, SENTINEL, _packed_reference, _plan_reference, make_case
from tests.ops.qsa._benchmark import measure_components, write_summary


@pytest.mark.parametrize("tp_size", helpers.TP_SIZES)
@pytest.mark.parametrize("rows,prefix", ((1, 2051), (5, 2055), (68, 30000)))
def test_attention_kernels(tp_size, rows, prefix):
    case = make_case(0, helpers._gpu(), tp_size=tp_size, rows=rows, prefix=prefix)
    for labels in (SCOPES, SCOPES[::-1]):
        for label in labels:
            case.reset[label]()
            case.runs[label]()
            case.checks[label]()


@pytest.mark.perf
@pytest.mark.parametrize("tp_size", helpers.TP_SIZES)
def test_attention_kernel_perf(tp_size):
    output = os.environ.get("QSA_REPLAY_OUTPUT")
    if not output:
        pytest.skip("Set QSA_REPLAY_OUTPUT under mytest/mydata for kernel timing")
    device = helpers._gpu()
    measure_components(lambda buffer: make_case(buffer, device, tp_size=tp_size),
                       Path(output) / "attention_kernels" / f"tp{tp_size}", device.index)


@pytest.mark.parametrize("hk", (1, 2))
def test_packed_reference_host(hk):
    source = torch.arange(8 * hk * 256, dtype=torch.int32).view(8, hk, 256)
    key, value = _packed_reference(source, source)
    for token in range(8):
        for dim in range(256):
            key_column = (dim % 64 // 32) * 128 + (dim % 32 // 8) * 32 + (token % 4) * 8 + dim % 8
            rank = (dim // 128) * 512 + (dim % 8 // 2) * 128 + (dim % 128 // 8) * 8 + (dim % 2) * 4 + token % 4
            assert torch.equal(key[token // 4 * 4 + dim // 64, :, key_column], source[token, :, dim])
            assert torch.equal(value[token // 4 * 4 + rank // 256, :, rank % 256], source[token, :, dim])


def test_plan_reference_host(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _: SimpleNamespace(multi_processor_count=80))
    value = helpers._make_case((5,), (3,), 12, torch.device("cpu"))
    inputs = SimpleNamespace(**vars(value), max_seqlen_k=8)
    plan = helpers._runtime.prepare.allocate_plan(inputs=inputs, query_tile=32)
    expected = _plan_reference(value, plan)
    np.testing.assert_array_equal(expected["dense"], [[31, 30]])
    np.testing.assert_array_equal(expected["blocks"][0, :2], [0, 1])
    np.testing.assert_array_equal(expected["membership"][0, :2], [31, 30])
    np.testing.assert_array_equal(expected["counts"], [[2, 0]])
    np.testing.assert_array_equal(expected["order"], [0])
    np.testing.assert_array_equal(expected["masks"][0, 0, :5, 0], [15, 31, 63, 127, 255])
    assert not expected["masks"][0, 0, 5:].any()
    assert not expected["masks"][0, 0, :, 1:].any()
    assert np.all(expected["blocks"][0, 2:] == SENTINEL)


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true", help="validate one buffer per TP without timing")
    mode.add_argument("--perf", action="store_true")
    parser.add_argument("--output", type=Path, required=True, help="new directory under mytest/mydata")
    parser.add_argument("--buffers", type=int, choices=(10,), default=helpers.BENCHMARK_BUFFERS)
    parser.add_argument("--samples", type=int, default=helpers.BENCHMARK_SAMPLES)
    parser.add_argument("--tp-sizes", nargs="+", default=["2", "4", "8"], help="space- or comma-separated TP2/4/8")
    parser.add_argument("--scopes", nargs="+", default=list(SCOPES), help=", ".join(SCOPES))
    parser.add_argument("--rows", type=int, default=68, help="sparse query rows")
    parser.add_argument("--prefix", type=int, default=30000, help="sparse prefix, >=2051; total KV must be four-aligned")
    args = parser.parse_args(argv)
    try:
        args.tp_sizes = [int(part) for item in args.tp_sizes for part in item.split(",")]
    except ValueError:
        parser.error("--tp-sizes requires 2, 4 or 8")
    args.scopes = [part for item in args.scopes for part in item.split(",")]
    if not args.tp_sizes or len(set(args.tp_sizes)) != len(args.tp_sizes) or not set(args.tp_sizes) <= set(helpers.TP_SIZES):
        parser.error("--tp-sizes requires distinct values from 2, 4, 8")
    if not args.scopes or len(set(args.scopes)) != len(args.scopes) or not set(args.scopes) <= set(SCOPES):
        parser.error(f"--scopes requires distinct names from {SCOPES}")
    if args.gpu < 0 or args.samples < args.buffers:
        parser.error("Require gpu >= 0 and samples >= buffers")
    if args.rows < 1 or args.prefix < 2051 or (args.rows + args.prefix) % 4:
        parser.error("Require rows > 0, prefix >= 2051, and (rows + prefix) divisible by four")
    if (args.rows + args.prefix) * 1024 > helpers._runtime.direct.MAX_PACKED_KV_BYTES:
        parser.error("KV exceeds the canonical packed scratch budget")
    if not args.output.resolve().is_relative_to(helpers.DATA.resolve()):
        parser.error("--output must be under mytest/mydata (including its resolved symlink target)")
    return args


def test_cli_host():
    base = ["--gpu", "3", "--output", str(helpers.DATA / "attention_kernels_cli_host")]
    args = _parse_args([*base, "--check-only", "--tp-sizes", "2,4,8"])
    assert args.buffers == 10 and args.samples == 128 and args.tp_sizes == [2, 4, 8]
    assert tuple(args.scopes) == SCOPES
    for extra in ([], ["--check-only", "--perf"], ["--perf", "--buffers", "1"],
                  ["--perf", "--scopes", "whole_chain"], ["--perf", "--tp-sizes", "2,2"]):
        with pytest.raises(SystemExit) as error:
            _parse_args([*base, *extra])
        assert error.value.code == 2


def main(argv=None):
    args = _parse_args(argv)
    if not torch.cuda.is_available() or torch.version.hip is None:
        raise RuntimeError("Requires ROCm gfx942")
    torch.cuda.set_device(args.gpu)
    if torch.cuda.get_device_properties(args.gpu).gcnArchName.split(":")[0] != "gfx942":
        raise RuntimeError("Requires gfx942")
    args.output.mkdir(parents=True, exist_ok=False)
    reports = {f"tp{tp}": dict(complete=False, tp_size=tp, error="not run") for tp in args.tp_sizes}
    try:
        for tp_size in args.tp_sizes:
            label = f"tp{tp_size}"
            folder = args.output / label
            try:
                measure_components(
                    lambda buffer: make_case(buffer, torch.device("cuda", args.gpu), tp_size=tp_size,
                                             rows=args.rows, prefix=args.prefix),
                    folder, args.gpu, buffers=args.buffers, samples=args.samples, warmup=2,
                    check_only=args.check_only, scopes=args.scopes)
            finally:
                result = folder / "result.json"
                if result.is_file():
                    reports[label] = dict(json.loads(result.read_text()), tp_size=tp_size)
    finally:
        write_summary(args.output, reports)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())