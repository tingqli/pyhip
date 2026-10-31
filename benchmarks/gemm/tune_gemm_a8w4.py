"""Tune/export the Aiter-compatible A8W4 API, then compare full calls.

The runtime's flydsl.autotune performs both searches: Aiter-only for the
baseline, then Aiter + PyHIP for the public API. There is no second selector
or winner CSV. --retune forces a search and exports artifacts when
FLYDSL_AUTOTUNE_CONFIG_DIR is set. Without it, normal cache/artifact lookup
applies; --check-only still tunes on a miss, just like the MoE benchmark.

    FLYDSL_AUTOTUNE_CONFIG_DIR=/tmp/a8w4_configs \
        python benchmarks/gemm/tune_gemm_a8w4.py \
        --shapes 32,64,512 4096,4096,8192 --retune --output /tmp/a8w4.json

Final timing uses the original rotating cudaPerf protocol, including output
allocation, PyHIP's scale repack and any Aiter Split-K reduction. Input
quantization and weight/scale preshuffle are prepared outside timing.
"""

import argparse
from contextlib import contextmanager
import csv
import gc
import json
import math
import os
from pathlib import Path
import statistics
import traceback

import torch

import pyhip
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.quant import per_1x32_mx_quant_hip
from aiter.ops.shuffle import shuffle_mxfp8fp4_b, shuffle_scale
from aiter.utility.fp4_utils import e8m0_to_f32, mxfp4_to_f32
from pyhip.ops.gemm import tuned_gemm_a8w4 as api


@contextmanager
def environment(name, value):
    previous = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


def _quantize(rows, K, scale, dtype):
    x = (torch.randn((rows, K), device="cuda") * scale).to(torch.bfloat16)
    q, s = per_1x32_mx_quant_hip(
        x, quant_dtype=dtype, scale_type=dtypes.fp8_e8m0, shuffle=False,
    )
    return q, s.view(torch.uint8)


def make_buffer(M, N, K):
    """Prepare shared shuffled inputs; construct the reference from raw quantized data."""
    a, sa = _quantize(M, K, 0.75, dtypes.fp8)
    b, sb = _quantize(N, K, 3.0, dtypes.fp4x2)
    b_ref = mxfp4_to_f32(b) * e8m0_to_f32(sb).repeat_interleave(32, dim=1)
    ref = torch.empty((M, N), device="cuda", dtype=torch.bfloat16)
    for row in range(0, M, 8192):
        scale = e8m0_to_f32(sa[row:row + 8192]).repeat_interleave(32, dim=1)
        ref[row:row + 8192] = ((a[row:row + 8192].float() * scale) @ b_ref.T).bfloat16()
    return dict(
        A=a, B=shuffle_mxfp8fp4_b(b), ScaleA=shuffle_scale(sa),
        ScaleB=shuffle_scale(sb), ref=ref,
    )


def run(buf, backend, splitk):
    args = (buf["A"], buf["B"], buf["ScaleA"], buf["ScaleB"])
    if backend == "aiter":
        # gfx950 has a native FlyDSL kernel but no public Aiter A8W4 op implementation.
        return api._gemm_a8w4_mxfp8(
            *args, torch.bfloat16, False, "", splitk, aiter_only=True,
        )
    return api.gemm_a8w4_mxfp8(*args, a_preshuffle=False, splitk=splitk)


def check_output(out, reference):
    if out.shape != reference.shape or out.dtype != reference.dtype:
        raise AssertionError("A8W4 output shape/dtype mismatch")
    if not torch.isfinite(out).all().item() or not torch.isfinite(reference).all().item():
        raise AssertionError("A8W4 output/reference contains nonfinite values")
    diff = pyhip.calc_diff(out, reference)
    if not math.isfinite(diff) or diff > api._DIFF_LIMIT:
        raise AssertionError(f"A8W4 calc_diff={diff} exceeds {api._DIFF_LIMIT}")
    return {
        "calc_diff": diff,
        "strict_allclose": torch.allclose(out, reference, rtol=0.02, atol=0.01),
    }


def buffer_addresses(buf):
    return {
        name: {
            "pointer": t.data_ptr(), "storage_base": t.untyped_storage().data_ptr(),
            "storage_offset": t.storage_offset(), "mod256": t.data_ptr() % 256,
            "mod4096": t.data_ptr() % 4096,
        }
        for name, t in buf.items() if name != "ref"
    }


def run_case(shape, args):
    M, N, K = shape
    row = dict(M=M, N=N, K=K, status="running", choices={}, raw=[], correctness={})
    first = make_buffer(M, N, K)
    previous_record = api.record_dispatch
    try:
        api.record_dispatch = True
        with environment("FLYDSL_AUTOTUNE", "1" if args.retune else "0"):
            for backend in ("aiter", "auto"):
                out = run(first, backend, args.splitk)
                row["correctness"][backend] = [check_output(out, first["ref"])]
                row["choices"][backend] = api.last_dispatch.copy()
        api.record_dispatch = False
        if args.check_only:
            row["status"] = "checked"
            return row

        pool = [first] + [make_buffer(M, N, K) for _ in range(args.buffers - 1)]
        row["addresses"] = [buffer_addresses(buf) for buf in pool]
        outputs = {"aiter": {}, "auto": {}}
        timed_outputs = {"aiter": [], "auto": []}
        with environment("FLYDSL_AUTOTUNE", "0"):
            for i, buf in enumerate(pool):
                for backend in ("aiter", "auto"):
                    outputs[backend][i] = run(buf, backend, args.splitk)
            for backend in outputs:
                for i, buf in enumerate(pool):
                    check_output(outputs[backend][i], buf["ref"])
            # Reference checks and telemetry must not replace the normal warmup order.
            for i, buf in enumerate(pool):
                for backend in (("aiter", "auto") if i % 2 else ("auto", "aiter")):
                    outputs[backend][i] = run(buf, backend, args.splitk)
            torch.cuda.synchronize()
            for sample in range(args.samples):
                i = (sample + 1) % len(pool)
                order = ("aiter", "auto") if sample % 2 == 0 else ("auto", "aiter")
                for backend in order:
                    with pyhip.cudaPerf(name=backend, verbose=0) as timer:
                        outputs[backend][i] = run(pool[i], backend, args.splitk)
                    us = timer.dt_ms * 1000
                    if not math.isfinite(us) or us <= 0:
                        raise RuntimeError(f"Invalid A8W4 timing sample: {us}")
                    timed_outputs[backend].append((i, outputs[backend][i]))
                    row["raw"].append(dict(
                        backend=backend, sample=sample, buffer=i, us=us,
                        order=order, output_pointer=outputs[backend][i].data_ptr(),
                    ))
        for backend in timed_outputs:
            # Check the outputs of the timed calls, without relaunching another kernel.
            row["correctness"][backend] = [
                check_output(out, pool[i]["ref"]) for i, out in timed_outputs[backend]
            ]
            times = [r["us"] for r in row["raw"] if r["backend"] == backend]
            med, best = statistics.median(times), min(times)
            row[backend] = dict(
                median_us=med, min_us=best, max_us=max(times),
                effective_tflops=2 * M * N * K / med / 1e6,
            )
        row["speedup"] = row["aiter"]["median_us"] / row["auto"]["median_us"]
        row["status"] = "complete"
        return row
    finally:
        api.record_dispatch = previous_record


def parse_shapes(values, input_path=None):
    shapes = [tuple(int(v) for v in s.split(",")) for s in values]
    if input_path is not None:
        with Path(input_path).open(newline="") as file:
            shapes += [(int(r["M"]), int(r["N"]), int(r["K"])) for r in csv.DictReader(file)]
    shapes = list(dict.fromkeys(shapes))
    if not shapes:
        raise ValueError("pass --shapes M,N,K or -i with M,N,K columns")
    if any(len(s) != 3 or min(s) <= 0 for s in shapes):
        raise ValueError("every shape must contain three positive integers")
    if any(n % 32 or k % 128 for _, n, k in shapes):
        raise ValueError("this Aiter/PyHIP comparison requires N%32==0 and K%128==0")
    return shapes


def print_table(rows, file=None):
    print("| M,N,K | selected backend | Aiter min/median us | auto min/median us | median speedup | auto TFLOPS |", file=file)
    print("|---|---|---:|---:|---:|---:|", file=file)
    for row in rows:
        if row["status"] != "complete":
            print(f"| {row['M']},{row['N']},{row['K']} | {row['status']} | - | - | - | - |", file=file)
            continue
        print(
            f"| {row['M']},{row['N']},{row['K']} | {row['choices']['auto']['_impl']} "
            f"| {row['aiter']['min_us']:.2f}/{row['aiter']['median_us']:.2f} "
            f"| {row['auto']['min_us']:.2f}/{row['auto']['median_us']:.2f} "
            f"| {row['speedup']:.3f}x | {row['auto']['effective_tflops']:.1f} |", file=file,
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shapes", nargs="*", default=[])
    parser.add_argument("-i", "--input", type=Path, help="optional CSV with M,N,K columns")
    parser.add_argument("--retune", action="store_true", help="force both searches; export artifacts if configured")
    parser.add_argument("--check-only", action="store_true", help="skip comparison timing, not tuning on cache misses")
    parser.add_argument("--buffers", type=int, default=50)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--splitk", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, help="new JSON report path")
    parser.add_argument("--md", type=Path, help="new Markdown summary path")
    args = parser.parse_args(argv)
    try:
        shapes = parse_shapes(args.shapes, args.input)
    except (OSError, KeyError, ValueError) as error:
        parser.error(str(error))
    if args.buffers < 1 or args.samples < args.buffers:
        parser.error("samples must be >= buffers > 0 so every timed buffer is checked")
    if args.splitk is not None and args.splitk < 1:
        parser.error("splitk must be positive")
    for path in (args.output, args.md):
        if path is not None and path.exists():
            parser.error(f"refusing to overwrite {path}")
    if get_gfx_runtime() != "gfx950":
        parser.error("this native A8W4 comparison requires gfx950")
    if not args.check_only and not all(pyhip.cudaPerf(name=name).enable for name in ("aiter", "auto")):
        parser.error("cudaPerf timing is disabled by CUDAPERF")
    torch.manual_seed(args.seed)
    properties = torch.cuda.get_device_properties()
    report = dict(
        status="running", rows=[], arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        implementation=api._implementation_key(),
        device=dict(name=properties.name, arch=properties.gcnArchName, pci_bus_id=properties.pci_bus_id),
        protocol="FlyDSL autotune for selection; full API cudaPerf samples alternate Aiter/auto with rotating buffers",
    )
    try:
        for shape in shapes:
            report["rows"].append(run_case(shape, args))
            print_table(report["rows"][-1:])
            if args.output is not None:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
            gc.collect()
            torch.cuda.empty_cache()
        report["status"] = "complete"
    except Exception:
        report["status"] = "failed"
        report["error"] = traceback.format_exc()
        raise
    finally:
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
    if args.md is not None:
        args.md.parent.mkdir(parents=True, exist_ok=True)
        with args.md.open("w") as file:
            print("# A8W4 full-API comparison\n", file=file)
            print_table(report["rows"], file)


if __name__ == "__main__":
    main()
