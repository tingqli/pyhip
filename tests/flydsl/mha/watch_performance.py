"""Watch amd-smi's utilization stream and run a supplied benchmark once idle.

No polling sleeps, process termination, clocks, power or PTL changes here.
The invoked benchmark retains its own isolation and optional PTL guards.
"""

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

if __package__:
    from ._hardware import SMI, ensure_idle
else:
    from _hardware import SMI, ensure_idle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--interval", type=int, default=2)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--allow-resident-workers", action="store_true",
                        help="trigger a current-policy diagnostic on low utilization; the child must not change hardware")
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or min(args.interval, args.samples) < 1 or not 0 <= args.threshold <= 100:
        parser.error("provide a command and positive interval/samples; threshold must be 0..100")
    args.record.parent.mkdir(parents=True, exist_ok=True)
    state = {"status": "watching", "gpu": args.gpu, "pid": os.getpid(), "command": command,
             "threshold_percent": args.threshold, "stable_samples": args.samples,
             "allow_resident_workers": args.allow_resident_workers,
             "interval_seconds": args.interval, "samples_seen": 0, "triggered": False}

    def record():
        state["time_utc"] = datetime.now(timezone.utc).isoformat()
        args.record.write_text(json.dumps(state, indent=2) + "\n")

    record()
    stable = 0
    previous = None
    # amd-smi supplies the timed event stream; the watcher blocks reading it.
    with subprocess.Popen([SMI, "metric", "--gpu", args.gpu, "--usage", "--csv", "--watch", str(args.interval)],
                          stdout=subprocess.PIPE, text=True, bufsize=1) as monitor:
        try:
            headers = None
            for line in monitor.stdout:
                values = next(csv.reader([line]))
                if "gfx_activity" in values:
                    headers = values
                    continue
                if not headers or len(values) != len(headers):
                    continue
                sample = dict(zip(headers, values))
                try:
                    activity = float(sample["gfx_activity"])
                except (KeyError, ValueError):
                    continue
                state.update(samples_seen=state["samples_seen"] + 1, gfx_activity_percent=activity)
                stable = stable + 1 if activity <= args.threshold else 0
                state["consecutive_low_samples"] = stable
                if activity != previous:
                    print("GPU_WATCH", args.gpu, activity, "%", flush=True)
                    previous = activity
                if stable >= args.samples:
                    try:
                        if not args.allow_resident_workers:
                            ensure_idle(args.gpu)
                    except RuntimeError as exc:
                        state["blocking_processes"] = str(exc)
                        stable = 0
                    else:
                        state.update(status="running", triggered=True)
                        state.pop("blocking_processes", None)
                        record()
                        print("GPU_IDLE_RUN", command, flush=True)
                        monitor.terminate()
                        monitor.wait()
                        result = subprocess.run(command)
                        state.update(status="complete" if result.returncode == 0 else "failed", returncode=result.returncode)
                        record()
                        return result.returncode
                record()
            raise RuntimeError("amd-smi utilization stream ended before an idle interval")
        except BaseException as exc:
            state.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error=str(exc))
            record()
            raise
        finally:
            if monitor.poll() is None:
                monitor.terminate()
                monitor.wait()


if __name__ == "__main__":
    sys.exit(main())