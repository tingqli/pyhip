"""Serial idle-GPU performance queue; no fallback or concurrent measurements."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

if __package__:
    from ._hardware import ensure_idle
else:
    from _hardware import ensure_idle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--when-low", action="store_true", help="if workers remain, measure diagnostic current-policy data only")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    here = Path(__file__).resolve().parent
    # A low-usage window can be a server restart; it is not a reservation.
    # Keep --when-low diagnostic even if a single process snapshot is empty,
    # rather than racing a new worker with a hardware-policy change.
    diagnostic = args.when_low
    if not diagnostic:
        ensure_idle("0")
    bf16_policy, fp8_policy = ("current", "current") if diagnostic else ("VECTOR,BF16", "VECTOR,F8")
    compare_flags = ["--allow-contention"] if diagnostic else []
    gate_flags = ["--allow-contention"] if diagnostic else ["--require-baseline"]
    commands = (
        ("bf16", [sys.executable, str(here / "reproduce_baselines.py"), "--backend", "bf16",
                  "--ptl", bf16_policy, *compare_flags,
                  "--output", str(args.output_dir / "bf16_original_vs_spill_fixed.json")]),
        ("fp8", [sys.executable, str(here / "test_mha_pa.py"), "--mode", "performance",
                 "--backend", "fp8_942", "--case", "fp8_native_410t", "--ptl", fp8_policy,
                 *gate_flags, "--aiter", "off", "--output", str(args.output_dir / "fp8_410t_gate.json")]),
    )
    report = {"status": "running", "diagnostic_only": diagnostic, "runs": []}
    def save():
        (args.output_dir / "queue.json").write_text(json.dumps(report, indent=2) + "\n")
    save()
    for name, command in commands:
        if not diagnostic:
            ensure_idle("0")
        print("PERFORMANCE_QUEUE_START", name, flush=True)
        with (args.output_dir / f"{name}.log").open("w") as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        report["runs"].append({"name": name, "command": command, "returncode": completed.returncode})
        save()
        print("PERFORMANCE_QUEUE_END", name, completed.returncode, flush=True)
        if completed.returncode:
            report["status"] = "failed"
            save()
            return completed.returncode
    report["status"] = "complete"
    save()
    return 0


if __name__ == "__main__":
    sys.exit(main())