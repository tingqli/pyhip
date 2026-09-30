"""Recompute retained performance claims without GPU execution."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import statistics
from pathlib import Path


def audit_result(path: Path, *, check_current: bool = False) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    if not report["complete"] or report["errors"]:
        raise AssertionError(f"Incomplete/rejected result: {path}")
    for scope, summary in report["summary"].items():
        values = [item["us"] for item in report["raw"] if item["scope"] == scope]
        assert values and min(values) > 0
        median = statistics.median(values)
        assert median == summary["median_us"]
        tflops = report["useful_flops"] / (median * 1e6)
        assert abs(tflops - summary["effective_tflops"]) < 1e-10
        target = summary["target_tflops"]
        assert summary["target_met"] == (None if target is None else tflops >= target)
    for phase in ("before", "before_samples", "after"):
        gate = json.loads(
            (path.parent / f"hardware_{phase}.json").read_text(encoding="utf-8")
        )
        assert not gate["settings_written"]
        assert int(gate["card"]["GPU use (%)"]) <= 5
        assert int(gate["card"]["GPU Memory Allocated (VRAM%)"]) <= 20
        assert gate["runtime"]["bdf"].lower() == gate["card"]["PCI Bus"].lower()
        assert gate["limit"]["ptl_state"].lower() == "enabled"
        assert set(gate["limit"]["ptl_format"].split(",")) == {"VECTOR", "F8"}
    for name, digest in report["source_sha256"].items():
        snapshot = path.parent / "source" / name
        assert hashlib.sha256(snapshot.read_bytes()).hexdigest() == digest
    root = Path(__file__).parent
    if check_current:
        for name in (
            "kernel.py",
            "plan.py",
            "direct.py",
            "dense.py",
            "implementation.py",
        ):
            snapshot = path.parent / "source" / name
            if not snapshot.exists():
                raise AssertionError(
                    f"Historical result lacks current implementation {name}"
                )
            frozen = snapshot.read_text(encoding="utf-8")
            current = (root / name).read_text(encoding="utf-8")
            assert ast.dump(ast.parse(frozen)) == ast.dump(ast.parse(current)), name
    for artifact in report["artifacts"]:
        for field in (
            "private_segment_fixed_size",
            "vgpr_spill_count",
            "sgpr_spill_count",
        ):
            assert artifact[field] and not any(artifact[field])
        for binary in artifact["code_objects"]:
            content = (path.parent / binary["path"]).read_bytes()
            assert content.startswith(b"\x7fELF")
            assert hashlib.sha256(content).hexdigest() == binary["sha256"]
    return {"path": str(path), "samples": len(report["raw"]), **report["summary"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--check-current", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            [
                audit_result(path, check_current=args.check_current)
                for path in args.results
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
