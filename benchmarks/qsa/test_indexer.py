"""Synthetic public QSA indexer checks and default operator/kernel timing.

Prefill and decode start after projection. Decode timing is one public-operator
graph replay. Fixture construction, projection, resets, compilation and
independent PyTorch references are untimed. Default execution reports operator
and individual kernel timing.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from types import SimpleNamespace

import torch

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.ops.qsa import _indexer as helpers  # noqa: E402
from tests.ops.qsa._benchmark import (  # noqa: E402
    DATA,
    ROOT,
    capture_kernels,
    default_output,
    format_table,
    measure_kernels,
    recording_matrix,
    source_files,
    write_summary,
)

BENCHMARK_BUFFERS, BENCHMARK_SAMPLES = 10, 128
MODES = {"prefill": "prefill_indexer", "decode": "decode_indexer"}
OPERATORS = dict(prefill_indexer=helpers.indexer.prefill_indexer, decode_indexer=helpers.indexer.decode_indexer)


def _source_hashes(sources):
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}


def _save_sources(folder, sources, expected):
    for source in sources:
        name = str(source.relative_to(ROOT))
        content = source.read_bytes()
        assert hashlib.sha256(content).hexdigest() == expected[name], f"Source changed: {name}"
        target = folder / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(content)


def _check_output(source, mode, output, expected):
    if mode == "prefill_indexer":
        return helpers.check(source, actual=output, expected=expected)
    return helpers.check_decode(source, output, expected=expected)


def _operator_table(name, report):
    checks = [{**check, **check.get("selection", {})} for check in report["checks"]]
    gaps = [check["worst_relative_boundary_violation"] for check in checks]
    timing = report["summary"].get(report["operator"], {})
    row = (report["operator"], len(checks), sum(check["different_token_sets"] for check in checks),
           max(gaps) if gaps else None, timing.get("median_us"), timing.get("mean_us"),
           timing.get("min_us"), timing.get("max_us"))
    status = "complete" if report["complete"] else report.get("error", "incomplete")
    title = f"\n{name}: GPU {report['gpu']} ({status})"
    return "\n".join([title, format_table(("Operator", "Buffers", "Diff_rows", "Worst_gap", "Median_us",
                                             "Mean_us", "Min_us", "Max_us"), [row])])


@torch.no_grad()
def benchmark(make_source, folder, gpu, *, mode="prefill_indexer", check_only=True,
              buffers=BENCHMARK_BUFFERS, warmup=2, samples=BENCHMARK_SAMPLES):
    """Ten independent buffers, actual-output checks and complete raw receipts."""
    if mode not in OPERATORS or gpu < 0 or buffers != BENCHMARK_BUFFERS or samples < buffers or warmup < 0:
        raise ValueError("Require a public operator, gpu >= 0, 10 buffers, samples >= 10 and warmup >= 0")
    folder = Path(folder)
    if not folder.resolve().is_relative_to(DATA.resolve()):
        raise ValueError("Results must stay under mytest/mydata")
    folder.mkdir(parents=True, exist_ok=False)
    graph_mode = mode != "prefill_indexer"
    report = dict(
        complete=False, check_only=check_only, raw=[], summary={}, gpu=gpu, buffers=buffers,
        warmup=warmup, samples=samples, operator=mode, fixture="synthetic", source_sha256={}, torch=torch.__version__,
        hip=torch.version.hip, scope=f"public pyhip.ops.qsa.flydsl.indexer.{mode}; "
            + ("one CUDA-graph replay" if graph_mode else "one eager call")
            + "; projection, metadata construction, reset, reference and checks excluded",
        baseline=None, reference="independent PyTorch BF16 prep / FP64 selection; not timed",
        heads=4, head_dim=128, token_width=2051,
        output_allocation="private graph output per buffer" if graph_mode else "public operator allocation per call",
        addresses=[], checks=[],
    )
    try:
        from tests.ops.qsa._benchmark import tensor_address

        sources = source_files()
        report["source_sha256"] = _source_hashes(sources)
        if not torch.cuda.is_available() or torch.version.hip is None:
            raise RuntimeError("requires ROCm gfx942")
        if not torch.cuda.get_device_properties(gpu).gcnArchName.startswith("gfx942"):
            raise RuntimeError("requires gfx942")
        cases = []
        storage_bases = set()
        with torch.cuda.device(gpu):
            for buffer in range(buffers):
                source = make_source(buffer)
                expected = helpers.reference_prep(source.inputs, source.state, decode=mode == "decode_indexer")
                helpers.reset_state(source)
                output = OPERATORS[mode](**source.inputs)
                checked = _check_output(source, mode, output, expected)
                graph = None
                if graph_mode:
                    # Warm the actual public entry point before capture. Separate
                    # graph pools retain independent outputs for all buffers.
                    torch.cuda.synchronize(gpu)
                    helpers.reset_state(source)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        output = OPERATORS[mode](**source.inputs)
                    helpers.reset_state(source)
                    graph.replay()
                    checked = _check_output(source, mode, output, expected)
                case = SimpleNamespace(source=source, expected=expected, graph=graph,
                                       output=output, validated=output.clone())
                cases.append(case)
                report["checks"].append(dict(buffer=buffer, **checked))
                addresses = {name: tensor_address(tensor) for name, tensor in
                             {**source.inputs, "output": output}.items()
                             if isinstance(tensor, torch.Tensor) and tensor.numel()}
                report["addresses"].append(addresses)
                current = {address["storage_base"] for address in addresses.values()}
                assert storage_bases.isdisjoint(current), "buffers must not share input/output storage"
                storage_bases.update(current)
            report["host_frame"] = cases[0].source.host
            if mode == "prefill_indexer":
                report.update(seq_lens=cases[0].source.seq_lens, extend_lens=cases[0].source.extend_lens)
            report["input_shapes"] = {name: list(tensor.shape) for name, tensor in cases[0].source.inputs.items()
                                       if isinstance(tensor, torch.Tensor)}

            def run(case):
                if case.graph is None:
                    case.output = OPERATORS[mode](**case.source.inputs)
                else:
                    case.graph.replay()
                return case.output

            def check_actual(case, actual=None):
                # Exact equality to an independently validated output is itself
                # a numeric check; changed (possibly tied) selections need the
                # full FP64 boundary check. No replacement runtime invocation.
                output = case.output if actual is None else actual
                if not torch.equal(output, case.validated):
                    _check_output(case.source, mode, output, case.expected)
                elif case.expected is not None:
                    helpers.assert_state(case.source, case.expected)

            if not check_only:
                for case in cases:
                    for _ in range(warmup):
                        helpers.reset_state(case.source)
                        run(case)
                        check_actual(case)

            torch.cuda.synchronize(gpu)
            _save_sources(folder, sources, report["source_sha256"])
            if not check_only:
                from pyhip.testing.misc import cudaPerf

                timer = cudaPerf(name=f"qsa_{mode}", verbose=0)
                if not timer.enable:
                    raise RuntimeError("CUDAPERF disabled timing")
                for sample in range(samples):
                    buffer = sample % buffers
                    case = cases[buffer]
                    helpers.reset_state(case.source)
                    with timer:
                        run(case)
                    elapsed = timer.latencies[-1] * 1e6
                    report["raw"].append(dict(scope=mode, sample=sample, buffer=buffer, us=elapsed,
                                               output_address=tensor_address(case.output)))
                    assert math.isfinite(elapsed) and elapsed > 0
                    check_actual(case)
                values = [row["us"] for row in report["raw"]]
                report["summary"][mode] = dict(median_us=statistics.median(values), mean_us=statistics.fmean(values),
                                                min_us=min(values), max_us=max(values))
                torch.cuda.synchronize(gpu)
                assert _source_hashes(sources) == report["source_sha256"], "Source changed during measurement"
            kernel_cases = []
            try:
                for case in cases:
                    kernel_cases.append(capture_kernels(
                        lambda case=case: OPERATORS[mode](**case.source.inputs),
                        lambda case=case: helpers.reset_state(case.source),
                        lambda actual, case=case: check_actual(case, actual)))
                report["kernels"] = measure_kernels(kernel_cases, folder / "kernels", samples=samples,
                                                    warmup=warmup, check_only=check_only)
            finally:
                for case in kernel_cases:
                    case.kernels.close()
            assert _source_hashes(sources) == report["source_sha256"], "Source changed during measurement"
        report["complete"] = True
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        with (folder / "result.json").open("x") as stream:
            json.dump(report, stream, indent=2, default=str)
        print(_operator_table(folder.name, report), flush=True)
        write_summary(folder, {folder.name: report})
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--mode", choices=("all", *MODES), default="all")
    timing = parser.add_mutually_exclusive_group()
    timing.add_argument("--check-only", action="store_true", help="correctness only; no timing")
    timing.add_argument("--perf", action="store_true", help="performance is the default")
    parser.add_argument("--rows", nargs="+", type=int, help="decode batch sizes (default: 1 32)")
    parser.add_argument("--lengths", nargs="+", type=int, help="prefill or decode token lengths (default: 12000)")
    parser.add_argument("--buffers", type=int, choices=(BENCHMARK_BUFFERS,), default=BENCHMARK_BUFFERS)
    parser.add_argument("--samples", type=int, default=BENCHMARK_SAMPLES)
    parser.add_argument("--output", type=Path, default=default_output("indexer"))
    args = parser.parse_args(argv)
    if args.gpu < 0 or args.samples < args.buffers:
        parser.error("Require gpu >= 0 and samples >= 10")
    if args.rows is not None and args.mode == "prefill":
        parser.error("--rows requires --mode decode; use --lengths for prefill")
    for option, values in (("rows", args.rows), ("lengths", args.lengths)):
        if values is not None and (min(values) < 1 or len(set(values)) != len(values)):
            parser.error(f"--{option} values must be positive and distinct")
    if max(args.lengths or [12000]) > helpers.indexer.MAX_COMPRESSED_KEYS * 4:
        parser.error("Token lengths exceed the supported context")
    if not args.output.resolve().is_relative_to(DATA.resolve()):
        parser.error("Results must stay under mytest/mydata")
    # Refuse an existing result root before any GPU initialization.
    args.output.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda", args.gpu)
    jobs = []
    if args.mode in ("all", "prefill"):
        for length in args.lengths or [12000]:
            jobs.append((f"prefill_n{length}", "prefill", lambda buffer, n=length: helpers.synthetic(
                (n,), (n,), device, seed=11 + buffer)))
    if args.mode in ("all", "decode"):
        for rows in args.rows or (1, 32):
            for length in args.lengths or [12000]:
                jobs.append((f"decode_r{rows}_n{length}", "decode",
                             lambda buffer, r=rows, n=length: helpers.decode_case(
                                 (n,) * r, device, context=(n + 63) // 64 * 64, seed=r + 1000 * buffer)))
    with recording_matrix(args.output, [name for name, _, _ in jobs],
                          check_only=args.check_only) as reports:
        for name, mode, make_source in jobs:
            reports[name] = benchmark(make_source, args.output / name, args.gpu, mode=MODES[mode],
                                       check_only=args.check_only, buffers=args.buffers, samples=args.samples)
    print(f"Results: {args.output}", flush=True)


if __name__ == "__main__":
    main()
