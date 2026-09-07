"""Consolidate saved MHA evidence, preserving the provenance of every run."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET

if __package__:
    from ._runner import resource_fields, save
    from .validate_preservation import SOURCES
else:
    from _runner import resource_fields, save
    from validate_preservation import SOURCES


HERE = Path(__file__).resolve().parent


def summarize(root):
    root = root.resolve()
    resource_inputs = [root / name for name in ("resources.json", "resources_remaining.json", "resources_swa.json")]
    records, runs = {}, []
    for path in resource_inputs:
        data = json.loads(path.read_text())
        runs.append({"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                     "environment": data["environment"]})
        for entry in data["records"]:
            key = tuple(entry.get(name) for name in ("backend", "target", "dq", "causal", "with_lse", "query_tile", "block_n"))
            if key in records:
                previous = records[key]
                assert previous["resources"] == entry["resources"], key
                assert previous["instructions"] == entry["instructions"], key
            records[key] = {**entry, "source_run": path.name,
                            "source_hashes": data["environment"]["sources_sha256"]}
    for entry in records.values():
        source = Path(entry["isa"])
        assert hashlib.sha256(source.read_bytes()).hexdigest() == entry["isa_sha256"], source
        text = source.read_text()
        entry["resources"] = resource_fields(text)
        entry["has_spills"] = any(entry["resources"][key] for key in
                                   ("private_segment_fixed_size", "vgpr_spill_count", "sgpr_spill_count"))
        tag = f"{entry['backend']}_{entry['target']}_d{entry['dq']}_c{int(entry['causal'])}_lse{int(entry['with_lse'])}"
        if entry["query_tile"] is not None:
            tag += f"_q{entry['query_tile']}_bn{entry['block_n']}"
        target = root / "isa" / f"{tag}.isa.txt"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        entry["isa_original_dump"] = entry["isa"]
        entry["isa"] = str(target.relative_to(HERE))
    ordered = sorted(records.values(), key=lambda entry: (entry["backend"], entry["target"], entry["dq"],
                     entry["causal"], entry["with_lse"], entry["query_tile"] or 0, entry["block_n"] or 0))
    assert len(ordered) == 88, f"expected the full 88-specialization matrix, got {len(ordered)}"
    save(root / "resources_all.json", {"runs": runs, "executed": False, "records": ordered,
         "note": "Consolidated saved compile-only evidence; per-record source_run/hash is authoritative, not current source hashes."})

    report = {"generated_utc": datetime.now(timezone.utc).isoformat(), "sources": {
        name: {"revision": rev, "git_path": path, "sha256": digest}
        for name, (rev, path, digest) in SOURCES.items()},
        "current_files_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in HERE.glob("*.py")},
        "resource_records": len(ordered), "resource_backend_counts": dict(Counter(row["backend"] for row in ordered)),
        "gfx950_native_executed": False}
    native = json.loads((root / "preservation_942.json").read_text())
    report["native_preservation"] = [{key: row[key] for key in ("backend", "dq", "q", "kv", "causal",
        "output_bit_exact", "attention_us", "attention_delta_pct", "total_gpu_delta_pct")} for row in native["records"]]
    cross = json.loads((root / "preservation_950_compile.json").read_text())
    important = ("vgpr_count", "agpr_count", "group_segment_fixed_size", "private_segment_fixed_size", "vgpr_spill_count")
    report["gfx950_compile_preservation"] = {"records": len(cross["records"]),
        "same_vector_accumulator_lds_scratch": all(all(row["original"]["resources"][key] == row["refactored"]["resources"][key]
            for key in important) for row in cross["records"]),
        "machine_instructions_identical": sum(row["instructions_equal"] for row in cross["records"]),
        "note": "Native gfx950 functional/performance tests have not run; SGPR and compiler-inserted instructions may differ."}
    retained_cross = {**cross, "records": []}
    for index, row in enumerate(cross["records"]):
        retained = dict(row)
        for label in ("original", "refactored"):
            entry = dict(row[label])
            source = Path(entry["isa"])
            assert hashlib.sha256(source.read_bytes()).hexdigest() == entry["isa_sha256"]
            target = root / "isa_preservation" / f"{index:02d}_{row['backend']}_{label}.isa.txt"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            entry["isa_original_dump"] = entry["isa"]
            entry["isa"] = str(target.relative_to(HERE))
            retained[label] = entry
        retained_cross["records"].append(retained)
    save(root / "preservation_950_retained.json", retained_cross)
    # Retain exact historical numbers from the requested branch, without
    # relabeling them as new gfx950 measurements or mixing compiler versions.
    historical = root / "historical"
    historical.mkdir(parents=True, exist_ok=True)
    report["historical"] = []
    for git_path, name in (("tests/flydsl/pa_1wave/final_results.json", "gfx950_swa_original.json"),
                           ("tests/flydsl/pa_8wave/tile_refactor_results.json", "gfx950_pa_tile_refactor.json")):
        revision = SOURCES["mha_pa_bf16_950"][0]
        source = subprocess.check_output(["git", "show", f"{revision}:{git_path}"], cwd=HERE.parents[2])
        (historical / name).write_bytes(source)
        report["historical"].append({"revision": revision, "git_path": git_path,
            "file": str((historical / name).relative_to(HERE)), "sha256": hashlib.sha256(source).hexdigest(),
            "current_run": False})
    performance_path = root / "performance_all.json"
    if performance_path.exists():
        data = json.loads(performance_path.read_text())
        report["performance"] = {"environment": data["environment"], "unavailable": data["unavailable"],
            "records": [{key: value for key, value in row.items() if key != "rounds"} for row in data["records"]]}
        # A nonsemantic post-validation cleanup has an independently verified
        # manifest. Never silently relabel measurements with a new source hash.
        path = HERE / "mha_pa_bf16_942.py"
        current_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        measured_hash = data["environment"]["sources_sha256"][path.name]
        if current_hash != measured_hash:
            cleanup = json.loads((root / "final_cleanup.json").read_text())
            assert cleanup["measured_sha256"] == measured_hash and cleanup["final_sha256"] == current_hash
            assert cleanup["executable_ast_equal"]
            report["final_cleanup"] = cleanup
    functional_path = root / "functional_final.xml"
    if functional_path.exists():
        suites = ET.parse(functional_path).getroot()
        totals = Counter()
        reasons = Counter()
        for suite in suites.iter("testsuite"):
            for key in ("tests", "failures", "errors", "skipped"):
                totals[key] += int(suite.get(key, 0))
        for skipped in suites.iter("skipped"):
            reasons[skipped.get("message", "")] += 1
        report["functional"] = {**totals, "passed": totals["tests"] - totals["skipped"] - totals["failures"] - totals["errors"],
                                "skip_reasons": dict(reasons)}
        report["functional"]["artifacts_sha256"] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (functional_path, root / "functional_final.log")}
        assert totals["failures"] == totals["errors"] == 0, report["functional"]
    save(root / "summary.json", report)
    print("MHA_EVIDENCE_SUMMARY", {key: report[key] for key in ("resource_records", "gfx950_compile_preservation")})
    if "functional" in report:
        print("FUNCTIONAL", {key: value for key, value in report["functional"].items() if key != "skip_reasons"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=HERE / "results")
    summarize(parser.parse_args().results)