"""Inventory one machine's saved evidence without relabelling historical runs.

Only reads files and Git state. Native results are taken from supplied JUnit
and performance records, never inferred from case collection or ISA emission.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def junit_summary(path):
    root = ET.parse(path).getroot()
    cases = list(root.iter("testcase"))
    skipped = [case.find("skipped") for case in cases if case.find("skipped") is not None]
    failed = [case for case in cases if case.find("failure") is not None]
    errors = [case for case in cases if case.find("error") is not None]
    reasons = Counter(item.get("message", "") for item in skipped)
    return {"tests": len(cases), "passed": len(cases) - len(skipped) - len(failed) - len(errors),
            "skipped": len(skipped), "failures": len(failed), "errors": len(errors),
            "skip_reasons": dict(reasons),
            "skip_first_lines": dict(Counter(item.get("message", "").splitlines()[0]
                                              if item.get("message", "") else "" for item in skipped)),
            "failed_tests": [{"classname": case.get("classname"), "name": case.get("name")}
                             for case in [*failed, *errors]]}


def summarize(directory, output):
    directory, output = directory.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError("use a new manifest path; never overwrite prior evidence")
    artifacts, json_reports, tests, retained = [], [], [], []
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path == output or "__pycache__" in path.parts:
            continue
        relative = path.relative_to(directory).as_posix()
        artifacts.append({"path": relative, "sha256": digest(path), "bytes": path.stat().st_size})
        if path.suffix == ".xml":
            tests.append({"path": relative, **junit_summary(path)})
        if path.suffix != ".json":
            continue
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            json_reports.append({"path": relative, "valid_json": False})
            continue
        if not isinstance(data, dict) or "records" not in data:
            continue
        json_reports.append({"path": relative, "valid_json": True, "records": len(data["records"]),
                             "complete": data.get("complete"), "executed": data.get("executed"),
                             "environment": data.get("environment", data.get("compiler")),
                             "source_sha256": data.get("source_sha256"),
                             "comparison_complete": data.get("comparison_complete"),
                             "unavailable": data.get("unavailable", [])})
        for index, record in enumerate(data["records"]):
            entries = [("record", record)]
            entries += [(key, record[key]) for key in ("original", "refactored") if key in record]
            for label, entry in entries:
                if "isa_sha256" not in entry:
                    continue
                saved = entry.get("retained_isa", entry.get("isa"))
                source = Path(saved)
                if not source.is_absolute():
                    candidates = (path.parent / source, ROOT / source, HERE / source)
                    source = next((candidate for candidate in candidates if candidate.is_file()), candidates[0])
                exists = source.is_file()
                matched = exists and digest(source) == entry["isa_sha256"]
                if not matched:
                    raise ValueError(f"{relative} record {index}/{label}: ISA missing or SHA mismatch: {source}")
                retained.append({"report": relative, "record": index, "label": label,
                                 "path": Path(os.path.relpath(source, directory)).as_posix(),
                                 "isa_sha256": entry["isa_sha256"], "sha_verified": True,
                                 "resources": entry.get("resources"),
                                 "inside_results_directory": source.is_relative_to(directory)})
    git = lambda *args: subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    source_files = sorted({*HERE.glob("*.py"), *HERE.glob("*.md"), *HERE.glob(".gitignore")})
    settings = ROOT / ".vscode/settings.json"
    if settings.is_file():
        source_files.append(settings)
    result = {"generated_utc": datetime.now(timezone.utc).isoformat(), "gpu_queried": False,
              "results_directory": directory.relative_to(ROOT).as_posix(),
              "commit": git("rev-parse", "HEAD"), "branch": git("branch", "--show-current"),
              "worktree_status": git("status", "--short"),
              "delivery": "user requested uncommitted worktree; no commit or push",
              "source_files_sha256": {p.relative_to(ROOT).as_posix(): digest(p) for p in source_files},
              "artifacts": artifacts, "junit_runs": tests, "json_reports": json_reports, "isa": retained,
              "note": "JUnit runs may overlap; do not sum their passes as unique tests. Failed/partial attempts remain in this inventory. "
                      "Collection and target ISA emission are not native passes; individual source reports are authoritative."}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.results, args.output)
    print(f"Inventoried {len(result['artifacts'])} files, {len(result['junit_runs'])} JUnit runs and {len(result['isa'])} verified ISAs")


if __name__ == "__main__":
    main()