"""Shared QSA evidence format and component timing."""

import csv
import ctypes
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
from types import SimpleNamespace

from benchmarks.gr_read.bench_gr_read_compare import tensor_address

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "mytest/mydata"


def default_output(name):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return DATA / f"qsa_{name}_{stamp}_{os.getpid()}"


def kernel_name(symbol):
    for name in ("attention_recover_scatter", "attention_compact", "attention_order_masks_validate",
                 "attention_dense", "attention_union", "attention_pack_kv", "attention_direct",
                 "indexer_q_prep", "indexer_k_compress", "indexer_decode_prep",
                 "qsa_indexer_decode_logits", "qsa_indexer_decode_topk", "qsa_indexer_logits",
                 "qsa_indexer_topk", "assert_async"):
        if name in symbol:
            return name
    return symbol


class _Dim3(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint) for name in ("width", "height", "depth")]


class _KernelParams(ctypes.Structure):
    _fields_ = [
        ("block", _Dim3), ("extra", ctypes.c_void_p), ("function", ctypes.c_void_p),
        ("grid", _Dim3), ("arguments", ctypes.c_void_p), ("shared_bytes", ctypes.c_uint),
    ]


class KernelGraphs:
    """Replay captured HIP kernel parameters while retaining the original graph."""

    def __init__(self, owner):
        import torch

        self.owner = owner
        self.library = ctypes.CDLL("libamdhip64.so")
        pointer = ctypes.c_void_p
        handles = ctypes.POINTER(pointer)
        size = ctypes.c_size_t
        signatures = {
            "hipGraphGetNodes": [pointer, handles, ctypes.POINTER(size)],
            "hipGraphNodeGetDependencies": [pointer, handles, ctypes.POINTER(size)],
            "hipGraphNodeGetType": [pointer, ctypes.POINTER(ctypes.c_int)],
            "hipGraphKernelNodeGetParams": [pointer, ctypes.POINTER(_KernelParams)],
            "hipGraphCreate": [handles, ctypes.c_uint],
            "hipGraphAddKernelNode": [handles, pointer, handles, size, ctypes.POINTER(_KernelParams)],
            "hipGraphInstantiate": [handles, pointer, handles, pointer, size],
            "hipGraphLaunch": [pointer, pointer],
            "hipGraphExecDestroy": [pointer],
            "hipGraphDestroy": [pointer],
        }
        for name, arguments in signatures.items():
            function = getattr(self.library, name)
            function.argtypes, function.restype = arguments, ctypes.c_int
        self.library.hipKernelNameRefByPtr.argtypes = [pointer, pointer]
        self.library.hipKernelNameRefByPtr.restype = ctypes.c_char_p
        self.library.hipKernelNameRef.argtypes = [pointer]
        self.library.hipKernelNameRef.restype = ctypes.c_char_p
        self.executables = []
        nodes = self._nodes("hipGraphGetNodes", owner.raw_cuda_graph())
        dependencies = {node: set(self._nodes("hipGraphNodeGetDependencies", node)) for node in nodes}
        ordered = []
        while dependencies:
            ready = [node for node in nodes if node in dependencies and not dependencies[node]]
            if len(ready) != 1:
                raise ValueError("Expected the operator's single-stream kernel chain")
            node = ready[0]
            ordered.append(node)
            del dependencies[node]
            for parents in dependencies.values():
                parents.discard(node)
        self.parameters, self.metadata = [], []
        stream = torch.cuda.current_stream().cuda_stream
        for node in ordered:
            kind = ctypes.c_int()
            self._call("hipGraphNodeGetType", node, ctypes.byref(kind))
            if kind.value != 0:
                raise ValueError(f"Non-kernel graph node: {kind.value}")
            params = _KernelParams()
            self._call("hipGraphKernelNodeGetParams", node, ctypes.byref(params))
            name = self.library.hipKernelNameRefByPtr(params.function, stream)
            if not name:
                name = self.library.hipKernelNameRef(params.function)
            if not name:
                raise RuntimeError("HIP could not identify a captured kernel")
            self.parameters.append(params)
            self.metadata.append(dict(
                name=name.decode(), function=params.function,
                grid=[params.grid.width, params.grid.height, params.grid.depth],
                block=[params.block.width, params.block.height, params.block.depth],
                shared_bytes=params.shared_bytes,
            ))

    def _call(self, name, *arguments):
        status = getattr(self.library, name)(*arguments)
        if status:
            raise RuntimeError(f"{name} failed with HIP status {status}")

    def _nodes(self, name, handle):
        count = ctypes.c_size_t()
        self._call(name, handle, None, ctypes.byref(count))
        nodes = (ctypes.c_void_p * count.value)()
        self._call(name, handle, nodes, ctypes.byref(count))
        return list(nodes)

    def segment(self, first, end):
        if first == end:
            return None
        graph, executable, previous = ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_void_p()
        self._call("hipGraphCreate", ctypes.byref(graph), 0)
        try:
            for params in self.parameters[first:end]:
                node = ctypes.c_void_p()
                self._call("hipGraphAddKernelNode", ctypes.byref(node), graph,
                           ctypes.byref(previous) if previous.value else None,
                           int(bool(previous.value)), ctypes.byref(params))
                previous = node
            self._call("hipGraphInstantiate", ctypes.byref(executable), graph, None, None, 0)
        finally:
            self._call("hipGraphDestroy", graph)
        self.executables.append(executable)
        return executable

    def replay(self, executable):
        import torch

        if executable is not None:
            self._call("hipGraphLaunch", executable, torch.cuda.current_stream().cuda_stream)

    def close(self):
        for executable in self.executables:
            self._call("hipGraphExecDestroy", executable)
        self.executables.clear()


def capture_kernels(operation, reset, check):
    import torch

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        reset()
        check(operation())
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        reset()
        with torch.cuda.graph(graph, stream=stream):
            output = operation()
        graph.instantiate()
        reset()
        graph.replay()
        check(output)
        kernels = KernelGraphs(graph)
        segments = [tuple(kernels.segment(first, end) for first, end in (
            (0, position), (position, position + 1), (position + 1, len(kernels.metadata))))
            for position in range(len(kernels.metadata))]
    torch.cuda.current_stream().wait_stream(stream)
    return SimpleNamespace(kernels=kernels, segments=segments, output=output, reset=reset, check=check)


def measure_kernels(cases, folder, *, samples=128, warmup=2, check_only=False):
    """Time actual captured kernels; all dependencies and output checks are untimed."""
    from pyhip.testing.misc import cudaPerf

    folder.mkdir()
    report = dict(complete=False, raw=[], summary={}, metadata=cases[0].kernels.metadata,
                  scope="single captured kernel; reset/prefix/suffix/check/JIT excluded", samples=samples)
    names = [row["name"] for row in report["metadata"]]
    labels = [f"{position}:{kernel_name(name)}" for position, name in enumerate(names)]
    try:
        assert all([row["name"] for row in case.kernels.metadata] == names for case in cases)
        for case in cases:
            for segments in case.segments:
                for _ in range(1 if check_only else warmup):
                    case.reset()
                    for segment in segments:
                        case.kernels.replay(segment)
                    case.check(case.output)
        if not check_only:
            timer = cudaPerf(name="qsa_kernels", verbose=0)
            if not timer.enable:
                raise RuntimeError("CUDAPERF disabled timing")
            for sample in range(samples):
                buffer = sample % len(cases)
                case = cases[buffer]
                order = list(range(len(names)))
                offset = sample % len(order)
                order = order[offset:] + order[:offset]
                for position in order if sample % 2 == 0 else order[::-1]:
                    before, kernel, after = case.segments[position]
                    case.reset()
                    case.kernels.replay(before)
                    with timer:
                        case.kernels.replay(kernel)
                    case.kernels.replay(after)
                    elapsed = timer.latencies[-1] * 1e6
                    assert math.isfinite(elapsed) and elapsed > 0
                    case.check(case.output)
                    report["raw"].append(dict(scope=labels[position], sample=sample, buffer=buffer, us=elapsed))
            for label in labels:
                values = [row["us"] for row in report["raw"] if row["scope"] == label]
                report["summary"][label] = dict(median_us=statistics.median(values),
                    mean_us=statistics.fmean(values), min_us=min(values), max_us=max(values))
        report["complete"] = True
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        for case in cases:
            case.kernels.close()
        with (folder / "result.json").open("x") as stream:
            json.dump(report, stream, indent=2)
    return report


def source_files():
    """Hash implementation dependencies as well as the harness, not old forwarding stubs alone."""
    directories = (
        ROOT / "src/pyhip/ops/qsa", ROOT / "src/pyhip/ops/mha",
        ROOT / "tests/ops/qsa", ROOT / "benchmarks/qsa", ROOT / "src/pyhip/testing",
    )
    files = {p for directory in directories for p in directory.rglob("*.py") if "__pycache__" not in p.parts}
    files.update((ROOT / "src/pyhip/testing/misc.py", ROOT / "benchmarks/gr_read/bench_gr_read_compare.py"))
    return sorted(files)


def format_table(headers, rows):
    """Aligned plain-text table; numeric columns are right-aligned and None prints as '-'."""
    def text(value):
        if value is None:
            return "-"
        if isinstance(value, float):
            return f"{value:.3e}" if value and abs(value) < 1e-3 else f"{value:.3f}"
        return str(value)

    cells = [[text(value) for value in row] for row in rows]
    numeric = [all(value is None or (isinstance(value, (int, float)) and not isinstance(value, bool))
                   for value in column) for column in zip(*rows)] if rows else [False] * len(headers)
    widths = [max([len(header), *(len(row[index]) for row in cells)]) for index, header in enumerate(headers)]

    def line(values):
        return "  ".join(value.rjust(width) if right else value.ljust(width)
                         for value, width, right in zip(values, widths, numeric)).rstrip()

    return "\n".join([line(headers), line(["-" * width for width in widths]), *(line(row) for row in cells)])


def write_summary(folder, reports, *, echo=True):
    """Write machine-readable per-case summary and CSV tables, including incomplete cases."""
    summaries, rows, raw = {}, [], []
    for label, report in reports.items():
        summaries[label] = {key: value for key, value in report.items() if key not in ("raw", "addresses")}
        for scope, values in report.get("summary", {}).items():
            rows.append(dict(case=label, scope=scope, complete=report["complete"], gpu=report.get("gpu"),
                             tp_size=report.get("tp_size", report.get("local_tp_size",
                                      (report.get("capture") or {}).get("local_tp_size"))),
                             **report.get("routes", {}), **values))
        raw.extend(dict(case=label, **sample) for sample in report.get("raw", ()))
    result = dict(complete=bool(reports) and all(r["complete"] for r in reports.values()),
                  cases=summaries, raw_samples=len(raw))
    with (folder / "summary.json").open("x") as stream:
        json.dump(result, stream, indent=2, default=str)
    for name, records, required in (("summary.csv", rows, ("case", "scope", "complete", "median_us")),
                                    ("raw.csv", raw, ("case", "scope", "sample", "buffer", "us"))):
        fields = list(dict.fromkeys((*required, *(key for row in records for key in row))))
        with (folder / name).open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)
    kernels = [dict(case=label, kernel=name, complete=report["complete"], **values)
               for label, report in reports.items()
               for name, values in report.get("kernels", {}).get("summary", {}).items()]
    with (folder / "kernels.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("case", "kernel", "complete", "median_us",
                                                   "mean_us", "min_us", "max_us"))
        writer.writeheader()
        writer.writerows(kernels)
    with (folder / "kernels.txt").open("x") as stream:
        table = format_table(("Case", "Kernel", "Median_us", "Status"), [
            (row["case"], row["kernel"], row["median_us"], "valid" if row["complete"] else "invalid")
            for row in kernels])
        print(table, file=stream)
    if echo and kernels:
        print(table, flush=True)
    return result


@contextmanager
def recording_matrix(folder, labels, *, check_only=True):
    """Record a whole matrix, retaining completed results when a case fails."""
    labels = list(labels)
    if not labels or len(set(labels)) != len(labels):
        raise ValueError("Expected a nonempty matrix with unique labels")
    reports = {label: dict(complete=False, error="not run") for label in labels}
    body_error = None
    try:
        yield reports
    except BaseException as error:
        body_error = f"{type(error).__name__}: {error}"
        raise
    finally:
        for label, report in reports.items():
            path = folder / label / "result.json"
            if not report["complete"] and path.is_file():
                reports[label] = json.loads(path.read_text())
        with (folder / "matrix_status.json").open("x") as stream:
            json.dump(
                dict(
                    check_only=check_only,
                    error=body_error,
                ),
                stream,
                indent=2,
            )
        # Each case already printed its kernel table.
        write_summary(folder, reports, echo=False)
