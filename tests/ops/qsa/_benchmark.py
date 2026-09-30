"""Shared QSA evidence format and explicit, gated component timing."""

import csv
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "mytest/mydata"


def source_files():
    """Hash implementation dependencies as well as the harness, not old forwarding stubs alone."""
    directories = (
        ROOT / "src/pyhip/ops/qsa", ROOT / "src/pyhip/ops/mha",
        ROOT / "tests/ops/qsa", ROOT / "benchmarks/qsa",
        ROOT / "experiments/attention/flydsl/qsa",
    )
    files = {p for directory in directories for p in directory.rglob("*.py") if "__pycache__" not in p.parts}
    files.update((ROOT / "src/pyhip/testing/misc.py", ROOT / "tests/ops/gr_read/test_gr_read.py"))
    return sorted(files)


def gate(folder, phase, gpu):
    """Single read-only check; retain failed snapshots without waiting, retrying or changing hardware."""
    import torch
    from tests.ops.gr_read.test_gr_read import read_hardware, validate_hardware

    torch.cuda.synchronize(gpu)
    snapshot = read_hardware(gpu, Path("/opt/rocm-7.14/bin/amd-smi"))
    props = torch.cuda.get_device_properties(gpu)
    pci = f"{props.pci_domain_id:04x}:{props.pci_bus_id:02x}:{props.pci_device_id:02x}.0"
    snapshot["runtime_pci"] = pci
    with (folder / f"hardware_{phase}.json").open("x") as stream:
        json.dump(snapshot, stream, indent=2)
    validate_hardware(snapshot)
    assert pci.lower() == snapshot["card"]["PCI Bus"].lower()


def write_summary(folder, reports):
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
    return result


@contextmanager
def recording_matrix(folder, labels):
    """Finalize every planned case, including a failing load/setup and unstarted cases."""
    labels = list(labels)
    if not labels or len(set(labels)) != len(labels):
        raise ValueError("Expected a nonempty matrix with unique labels")
    reports = {label: dict(complete=False, error="not run") for label in labels}
    try:
        yield reports
    finally:
        for label, report in reports.items():
            path = folder / label / "result.json"
            if not report["complete"] and path.is_file():
                reports[label] = json.loads(path.read_text())
        write_summary(folder, reports)


def measure_components(make_case, folder, gpu, *, buffers=10, warmup=2, samples=128, check_only=False,
                       scopes=None):
    """Time each prepared kernel separately; setup/reset/check are always outside cudaPerf.

    A case owns independent tensor buffers, ``runs``/``checks``/``reset`` mappings,
    optional ``flops`` per scope and ``metadata``. No sum of these medians is a full-call latency.
    Checks consume the actual timed output before another component can overwrite it.
    """
    import torch
    from pyhip.testing.misc import cudaPerf
    from tests.ops.gr_read.test_gr_read import tensor_address

    if buffers < 1 or samples < buffers or warmup < 0:
        raise ValueError("Require samples >= buffers >= 1 and warmup >= 0")
    folder = Path(folder)
    if not folder.resolve().is_relative_to(DATA.resolve()):
        raise ValueError("Results must stay under mytest/mydata")
    folder.mkdir(parents=True, exist_ok=False)
    sources = source_files()
    hashes = lambda: {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    report = dict(complete=False, check_only=check_only, raw=[], summary={}, gpu=gpu, buffers=buffers,
                  warmup=warmup, samples=samples, source_sha256=hashes(), torch=torch.__version__,
                  hip=torch.version.hip, scope="individual prepared kernel; setup/reset/reference excluded")
    try:
        if not check_only:
            gate(folder, "before", gpu)
        cases = [make_case(index) for index in range(1 if check_only else buffers)]
        labels = list(cases[0].runs) if scopes is None else list(scopes)
        if not labels or len(set(labels)) != len(labels) or not set(labels).issubset(cases[0].runs):
            raise ValueError(f"Unknown or duplicate scopes: {labels}; available: {list(cases[0].runs)}")
        report["metadata"] = cases[0].metadata
        report["addresses"] = [{name: tensor_address(tensor) for name, tensor in case.tensors.items()}
                               for case in cases]
        for name in cases[0].tensors:
            assert len({case.tensors[name].data_ptr() for case in cases}) == len(cases), name
        for case in cases:
            for name in labels:
                for _ in range(1 + (0 if check_only else warmup)):
                    case.reset[name]()
                    case.runs[name]()
                    case.checks[name]()
        if not check_only:
            gate(folder, "before_samples", gpu)
            timer = cudaPerf(name="qsa_components", verbose=0)
            if not timer.enable:
                raise RuntimeError("CUDAPERF disabled timing")
            for sample in range(samples):
                index = sample % buffers
                case = cases[index]
                order = labels[sample % len(labels):] + labels[:sample % len(labels)]
                for name in (order if sample % 2 == 0 else order[::-1]):
                    case.reset[name]()
                    with timer:
                        case.runs[name]()
                    elapsed = timer.latencies[-1] * 1e6
                    report["raw"].append(dict(scope=name, sample=sample, buffer=index, us=elapsed))
                    assert math.isfinite(elapsed) and elapsed > 0
                    case.checks[name]()
            for name in labels:
                values = [row["us"] for row in report["raw"] if row["scope"] == name]
                median = statistics.median(values)
                row = dict(median_us=median, mean_us=statistics.fmean(values), min_us=min(values), max_us=max(values))
                if name in cases[0].flops:
                    row.update(useful_flops=cases[0].flops[name], effective_tflops=cases[0].flops[name] / median / 1e6)
                report["summary"][name] = row
            gate(folder, "after", gpu)
        assert hashes() == report["source_sha256"], "Source changed during measurement"
        for source in sources:
            target = folder / "source" / source.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        report["complete"] = True
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        with (folder / "result.json").open("x") as stream:
            json.dump(report, stream, indent=2, default=str)
        write_summary(folder, {folder.name: report})
    return report