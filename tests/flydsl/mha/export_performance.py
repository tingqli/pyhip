"""Export measured MHA latency/TFLOPS from explicit reports without running a GPU.

Input JSON and raw samples are never modified. Do not point this at a historical
summary and infer a new validation run: every row retains its actual source.
"""

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path


def performance_rows(report, source):
    environment = report.get("environment", {})
    rows = []
    for record in report.get("records", []):
        if record.get("event_interval_us") is not None:
            metric, times = "event_interval_us", record["event_interval_us"]
        elif record.get("attention_us") is not None:
            metric, times = "attention_us", record["attention_us"]
        else:
            continue  # Resource/plan records have no measured throughput.
        if record.get("executed") is False or report.get("executed") is False:
            raise ValueError(f"{source}: compile-only/unexecuted records cannot claim measured performance")
        workload = record.get("workload", record)
        flops = record.get("effective_flops", workload.get("effective_flops"))
        if isinstance(flops, bool) or not isinstance(flops, (int, float)) or not math.isfinite(flops) or flops <= 0:
            raise ValueError(f"{source}: measured attention requires positive finite effective_flops")
        if not isinstance(times, dict) or not times:
            raise ValueError(f"{source}: measured intervals must be a nonempty candidate mapping")
        for candidate, us in times.items():
            if isinstance(us, bool) or not isinstance(us, (int, float)) or not math.isfinite(us) or us <= 0:
                raise ValueError(f"{source}: invalid {candidate} latency")
            tflops = None if candidate == "gather" else flops / us / 1e6
            reported = record.get("tflops", {}).get(candidate)
            if candidate == "gather" and reported is not None:
                raise ValueError(f"{source}: gather-only must not report attention TFLOPS")
            if reported is not None and (isinstance(reported, bool) or not isinstance(reported, (int, float))
                                         or not math.isfinite(reported)
                                         or not math.isclose(reported, tflops, rel_tol=1e-9)):
                raise ValueError(f"{source}: {candidate} TFLOPS disagrees with effective FLOPs / {metric}")
            # Reproducer checks are per implementation; a CK reference must
            # never inherit the measured FlyDSL candidate's acceptance label.
            checks = record.get("comparison_baseline_checks", {}).get(candidate)
            if checks is None:
                checks = record.get("baseline_checks", []) if candidate in ("current", "flydsl") else []
            rows.append({
                "source": source, "report_complete": report.get("complete", False),
                "case": workload["name"], "backend": record["backend"],
                "config": record.get("config", "auto"), "candidate": candidate,
                "gpu": environment.get("gpu"), "arch": environment.get("arch"),
                "compute_units": environment.get("compute_units"), "flydsl": environment.get("flydsl"),
                "timer_metric": metric, "latency_us": us, "effective_flops": flops,
                "tflops": tflops, "tflops_applicable": tflops is not None,
                "isolated": record.get("isolated", report.get("isolated")),
                "q_lens": workload.get("q_lens"), "kv_lens": workload.get("kv_lens"),
                "heads": workload.get("heads"), "kv_heads": workload.get("kv_heads"),
                "dq": workload.get("dq"), "dv": workload.get("dv"), "page": workload.get("page"),
                "causal": workload.get("causal"), "window": workload.get("window"),
                "sink": workload.get("sink"), "scale_mode": workload.get("scale_mode"),
                "protocol": record.get("protocol"), "baseline_checks": checks,
                "comparison_baseline_checks": record.get("comparison_baseline_checks", {}),
                "requested_reference": record.get("requested_reference"),
                "requested_reference_check": record.get("requested_reference_check") if candidate == "current" else None,
                "comparison_complete": record.get("comparison_complete"),
                "reference_unavailable": record.get("reference_unavailable", {}),
            })
    return rows


def export(reports, output):
    output = output.resolve()
    targets = [output.with_suffix(suffix) for suffix in (".json", ".csv", ".md")]
    if any(path.exists() for path in targets):
        raise FileExistsError("choose a new output prefix; existing evidence will not be overwritten")
    sources, rows = [], []
    for path in reports:
        path = path.resolve()
        content = path.read_bytes()
        report = json.loads(content)
        source = Path(os.path.relpath(path, output.parent)).as_posix()
        extracted = performance_rows(report, source)
        if not extracted:
            raise ValueError(f"{path}: no measured performance rows (plan/compile-only is not a pass)")
        sources.append({"path": source, "sha256": hashlib.sha256(content).hexdigest(),
                        "complete": report.get("complete", False), "environment": report.get("environment", {}),
                        "unavailable": report.get("unavailable", []), "rows": len(extracted)})
        rows.extend(extracted)
    if not rows:
        raise ValueError("at least one measured performance report is required")
    result = {"generated_from_reports_only": True, "gpu_queried": False, "sources": sources, "rows": rows,
              "mixed_timer_metrics": len({row["timer_metric"] for row in rows}) > 1,
              "note": "TFLOPS uses visible QK/PV FLOPs and each row's own timer; gather alone has no attention TFLOPS. "
                      "A summary is not an independent correctness run or historical baseline acceptance."}
    fields = list(rows[0])
    csv_text = io.StringIO()
    writer = csv.DictWriter(csv_text, fields)
    writer.writeheader()
    for row in rows:
        writer.writerow({key: json.dumps(value, allow_nan=False) if isinstance(value, (dict, list)) else value
                         for key, value in row.items()})
    markdown = ["# MHA measured latency and effective TFLOPS", "", result["note"], "",
                "| Case | Backend/config | Candidate | GPU/CU | FlyDSL | Metric | µs | TFLOPS | Source complete | Report |",
                "|---|---|---|---|---|---|---:|---:|---|---|"]
    for row in rows:
        throughput = "N/A (gather only)" if row["tflops"] is None else f"{row['tflops']:.3f}"
        source = row["source"].replace(" ", "%20")
        markdown.append(f"| {row['case']} | {row['backend']}/{row['config']} | {row['candidate']} | "
                        f"{row['gpu']}/{row['compute_units']} | {row['flydsl']} | {row['timer_metric']} | "
                        f"{row['latency_us']:.3f} | {throughput} | "
                        f"{row['report_complete']} | [JSON]({source}) |")
    markdown += ["", "Full shape, GPU/CU/compiler, effective FLOPs and protocol are retained in the CSV/JSON; "
                 "raw samples and skip/unavailable reasons remain in the linked source reports.", ""]
    output.parent.mkdir(parents=True, exist_ok=True)
    for target, text in zip(targets, (json.dumps(result, indent=2, allow_nan=False) + "\n",
                                     csv_text.getvalue(), "\n".join(markdown))):
        with target.open("x") as stream:
            stream.write(text)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True, help="new output prefix for JSON, CSV and Markdown")
    args = parser.parse_args()
    result = export(args.reports, args.output)
    print(f"Exported {len(result['rows'])} measured candidate rows with effective TFLOPS; no GPU execution")


if __name__ == "__main__":
    main()