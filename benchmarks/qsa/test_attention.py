"""Synthetic full QSA, forced direct/union and complete-causal-prefix comparison.

No model, captured dataset or integration hooks are used. Every timed output is
checked; allocation, references and JIT are outside the unchanged cudaPerf timer.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.qsa._synthetic import DENSE_ROWS, make_case, reference_rows
from pyhip.testing.misc import cudaPerf
from tests.ops.qsa._attention import _metadata, _runtime
from tests.ops.qsa._benchmark import (
    DATA, ROOT, capture_kernels, default_output, format_table, measure_kernels,
    recording_matrix, source_files, write_summary,
)

SCOPES = ("qsa", "forced_direct", "forced_union", "prefix_qsa", "prefix_direct", "prefix_union")


def _case(value, scope):
    workspace = _runtime._Workspace(value.q, value.k, value.v, value.indices,
                                    value.query_lens, value.prefix_lens, value.scale)
    inputs = workspace.bind(value.q, value.k, value.v, value.indices)
    if scope.endswith("direct"):
        # Direct-only: no union plan, ungated direct over every row.
        workspace.union = None
        workspace.direct = _runtime.direct.prepare(inputs=inputs)
    elif scope.endswith("union"):
        if workspace.union is None:
            workspace.union = _runtime.prepare.allocate_plan(inputs=inputs, query_tile=32)
            workspace.direct = _runtime.direct.prepare(inputs=inputs, union=workspace.union)
        workspace.union.union_ratio = math.inf
        workspace.union.routing = False
    return SimpleNamespace(value=value, scope=scope, workspace=workspace, output=torch.empty_like(value.q))


def _run(case):
    w, v = case.workspace, case.value
    inputs = w.bind(v.q, v.k, v.v, v.indices)
    _runtime.prepare.run(inputs=inputs, plan=w.union)
    if w.union is not None:
        _runtime.union.run(inputs=inputs, plan=w.union, out=case.output)
    _runtime.direct.run(inputs=inputs, prepared=w.direct, out=case.output)
    return case.output


def _routes(case):
    plan = case.workspace.union
    rows = len(case.value.q)
    union = 0
    union_pairs = selected_pairs = 0
    if plan is not None:
        meta, active, counts = plan.metadata.cpu().tolist(), plan.active.cpu().tolist(), plan.counts.cpu().tolist()
        for row, enabled, count in zip(meta, active, counts):
            first, n, _, _, pos = row
            if enabled:
                union += n
                union_pairs += n * count
                selected_pairs += sum(min((pos + i + 1) // 4, 512) + bool((pos + i + 1) % 4) for i in range(n))
        if case.scope.endswith("union"):
            assert all(active)
    if case.scope.endswith("direct"):
        assert plan is None
    return dict(rows=rows, union_planned=plan is not None, union_rows=union, direct_rows=rows - union,
                union_inflation=union_pairs / selected_pairs if selected_pairs else None)


def _host_hash(tensor):
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def _scope_table(name, report):
    rows = []
    for scope, route in report["routes"].items():
        timing = report["summary"].get(scope, {})
        rows.append((scope, route["rows"], route["union_rows"], route["direct_rows"],
                     route["union_inflation"], timing.get("median_us"), timing.get("min_us"),
                     timing.get("max_us"), timing.get("effective_tflops")))
    status = "complete" if report["complete"] else report.get("error", "incomplete")
    title = f"\n{name}: TP{report['tp_size']}, {report['rows']} rows, GPU {report['gpu']} ({status})"
    return "\n".join([title, format_table(("Scope", "Rows", "Union", "Direct", "Inflation",
                                             "Median_us", "Min_us", "Max_us", "TFLOPS"), rows)])


@torch.no_grad()
def benchmark(rows, tp_size, folder, gpu, *, buffers=10, samples=128, check_only=False, seed=20260929):
    from tests.ops.qsa._benchmark import tensor_address

    folder = Path(folder)
    if rows < 1 or tp_size not in (2, 4, 8) or buffers != 10 or samples < buffers:
        raise ValueError("Require positive rows, TP2/4/8, 10 buffers and samples >= 10")
    if not folder.resolve().is_relative_to(DATA.resolve()):
        raise ValueError("Results must stay under mytest/mydata")
    folder.mkdir(parents=True, exist_ok=False)
    sources = source_files()
    hashes = lambda: {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    report = dict(complete=False, raw=[], summary={}, gpu=gpu, tp_size=tp_size, buffers=buffers,
                  warmup=2, samples=samples, rows=rows, synthetic=True, check_only=check_only,
                  source_sha256=hashes(), routes={}, scope="qsa/forced_union: full rebuilt plan and dispatch; "
                  "forced_direct: no union plan. Private workspace; argument/cache lookup, output allocation "
                  "and JIT excluded equally. prefix_* uses only the identical complete causal prefix.")
    try:
        with torch.cuda.device(gpu):
            host, config = make_case(rows, 24 // tp_size, seed=seed)
            report["synthetic_config"] = config
            prefix_rows = min(rows, DENSE_ROWS)
            selected = sorted(set(np.linspace(0, rows - 1, min(rows, 48), dtype=int).tolist()
                                  + [n for n in (0, 3, 31, 32, 2047, 2050, 2051, rows - 1) if n < rows]))
            prefix_selected = sorted(set(n for n in selected if n < prefix_rows) | {prefix_rows - 1})
            oracle_rows = sorted(set(selected) | set(prefix_selected))
            oracle = reference_rows(host, oracle_rows)
            location = {row: i for i, row in enumerate(oracle_rows)}
            expected = {scope: oracle[[location[n] for n in (prefix_selected if scope.startswith("prefix") else selected)]]
                        for scope in SCOPES}
            bundles, validated = [], {}
            for buffer in range(1 if check_only else buffers):
                value = _metadata(*(getattr(host, name).to(torch.device("cuda", gpu), copy=True)
                                    for name in ("q", "k", "v", "indices")), (rows,), (0,))
                prefix = _metadata(value.q[:prefix_rows], value.k[:prefix_rows], value.v[:prefix_rows],
                                   value.indices[:prefix_rows], (prefix_rows,), (0,))
                bundle = {scope: _case(prefix if scope.startswith("prefix") else value, scope) for scope in SCOPES}
                bundles.append(bundle)
                for scope, case in bundle.items():
                    for _ in range(3):
                        case.output.fill_(float("nan"))
                        _run(case)
                    actual = case.output.cpu()
                    ids = prefix_selected if scope.startswith("prefix") else selected
                    torch.testing.assert_close(actual[ids].float(), expected[scope], rtol=.02, atol=.02)
                    assert bool(torch.isfinite(actual).all())
                    if buffer:
                        assert torch.equal(actual.view(torch.int16), validated[scope].view(torch.int16))
                    else:
                        validated[scope] = actual
                        report["routes"][scope] = _routes(case)
            report["addresses"] = [{scope: tensor_address(case.output, output=True) for scope, case in bundle.items()}
                                   for bundle in bundles]
            report["input_addresses"] = [
                {name: tensor_address(getattr(bundle["qsa"].value, name))
                 for name in ("q", "k", "v", "indices")} for bundle in bundles]
            for name in ("q", "k", "v", "indices"):
                assert len({row[name]["storage_base"] for row in report["input_addresses"]}) == len(bundles)
            report["input_sha256"] = {name: _host_hash(getattr(host, name)) for name in ("q", "k", "v", "indices")}
            report["checked_rows"] = dict(full=selected, prefix=prefix_selected)
            report["output_sha256"] = {scope: _host_hash(value) for scope, value in validated.items()}
            torch.cuda.synchronize(gpu)
            # Useful source/validation evidence work, not an idle loop or delay.
            for source in sources:
                target = folder / "source" / source.relative_to(ROOT)
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("xb") as stream:
                    stream.write(source.read_bytes())
            if not check_only:
                timer = cudaPerf(name="qsa_branches", verbose=0)
                if not timer.enable:
                    raise RuntimeError("CUDAPERF disabled timing")
                for sample in range(samples):
                    buffer = sample % buffers
                    order = SCOPES[sample % len(SCOPES):] + SCOPES[:sample % len(SCOPES)]
                    for scope in (order if sample % 2 == 0 else order[::-1]):
                        case = bundles[buffer][scope]
                        with timer:
                            _run(case)
                        us = timer.latencies[-1] * 1e6
                        report["raw"].append(dict(scope=scope, sample=sample, buffer=buffer, us=us))
                        assert math.isfinite(us) and us > 0
                        actual = case.output.cpu()
                        assert torch.equal(actual.view(torch.int16), validated[scope].view(torch.int16))
                for scope in SCOPES:
                    times = [r["us"] for r in report["raw"] if r["scope"] == scope]
                    work_rows = prefix_rows if scope.startswith("prefix") else rows
                    work = int((host.indices[:work_rows] >= 0).sum()) * 4 * host.q.shape[1] * 256
                    median = statistics.median(times)
                    report["summary"][scope] = dict(median_us=median, min_us=min(times), max_us=max(times),
                                                     effective_tflops=work / median / 1e6, useful_flops=work,
                                                     **report["routes"][scope])
                assert hashes() == report["source_sha256"]
            kernel_cases = []
            try:
                for bundle in bundles:
                    case = bundle["qsa"]

                    def check_kernel(actual):
                        torch.testing.assert_close(actual.cpu(), validated["qsa"], rtol=0, atol=0)

                    kernel_cases.append(capture_kernels(
                        lambda case=case: _run(case),
                        lambda case=case: case.output.fill_(float("nan")), check_kernel))
                report["kernels"] = measure_kernels(kernel_cases, folder / "kernels",
                                                    samples=samples, check_only=check_only)
            finally:
                for case in kernel_cases:
                    case.kernels.close()
            assert hashes() == report["source_sha256"]
            report["complete"] = True
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        with (folder / "result.json").open("x") as stream:
            json.dump(report, stream, indent=2)
        print(_scope_table(folder.name, report), flush=True)
        # Per-scope route fields live in summary, not a misleading combined route.
        compact = dict(report)
        compact.pop("routes", None)
        write_summary(folder, {folder.name: compact})
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--rows", nargs="+", type=int, default=[12000])
    parser.add_argument("--tp-sizes", nargs="+", type=int, choices=(2, 4, 8), default=[2, 4, 8])
    parser.add_argument("--buffers", type=int, choices=(10,), default=10)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--output", type=Path, default=default_output("attention"))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-only", action="store_true", help="validate without timing")
    mode.add_argument("--perf", action="store_true", help="performance is the default")
    args = parser.parse_args(argv)
    if min(args.rows) < 1 or args.samples < 10 or args.gpu < 0:
        parser.error("Require positive rows, gpu >= 0 and samples >= 10")
    if len(set(args.rows)) != len(args.rows) or len(set(args.tp_sizes)) != len(args.tp_sizes):
        parser.error("Rows and TP sizes must be unique")
    if not args.output.resolve().is_relative_to(DATA.resolve()):
        parser.error("Results must stay under mytest/mydata")
    args.output.mkdir(parents=True, exist_ok=False)
    jobs = [(f"m{rows}_tp{tp}", rows, tp) for rows in args.rows for tp in args.tp_sizes]
    with recording_matrix(args.output, [name for name, _, _ in jobs],
                          check_only=args.check_only) as reports:
        for name, rows, tp in jobs:
            result = benchmark(rows, tp, args.output / name, args.gpu, buffers=args.buffers,
                               samples=args.samples, seed=args.seed, check_only=args.check_only)
            reports[name] = {key: value for key, value in result.items() if key != "routes"}
    print(f"Results: {args.output}", flush=True)


@pytest.mark.parametrize("check_only,failure", [(False, None), (True, None), (False, "case")])
def test_matrix_reports(tmp_path, check_only, failure):
    def run():
        with recording_matrix(tmp_path, ["first", "second"], check_only=check_only) as reports:
            for name in reports:
                reports[name] = dict(complete=True, summary={}, raw=[
                    dict(scope="kernel", sample=0, buffer=0, us=1.0)])
                if failure == "case":
                    raise RuntimeError("case failed")

    if failure:
        with pytest.raises(RuntimeError, match=failure):
            run()
    else:
        run()
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["complete"] == (failure is None)
    assert summary["raw_samples"] == (1 if failure == "case" else 2)
    status = json.loads((tmp_path / "matrix_status.json").read_text())
    assert status == dict(check_only=check_only, error="RuntimeError: case failed" if failure else None)


if __name__ == "__main__":
    main()
