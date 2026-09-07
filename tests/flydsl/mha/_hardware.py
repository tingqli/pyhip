"""Read-only process guard for the logical benchmark device; never sets policy."""

import json
import os
import subprocess
import csv
from datetime import datetime, timezone


SMI = "/opt/rocm/bin/amd-smi"


def other_processes(data, own_pid):
    """Parse amd-smi output without ignoring memory-resident idle workers."""
    others = []
    def visit(value):
        if isinstance(value, dict):
            info = value.get("process_info")
            if isinstance(info, dict):
                pid = info.get("pid", info.get("process_id"))
                if pid is None or int(pid) != own_pid:
                    others.append(info)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
    visit(data)
    return others


def ensure_idle(device="0"):
    data = json.loads(subprocess.check_output([SMI, "process", "--gpu", device, "--json"], text=True, timeout=15))
    others = other_processes(data, os.getpid())
    if others:
        raise RuntimeError(f"GPU {device} has another process; refusing idle-guarded benchmark: {others}")


def require_idle_device():
    import torch
    prop = torch.cuda.get_device_properties(0)
    bdf = f"{prop.pci_domain_id:04x}:{prop.pci_bus_id:02x}:{prop.pci_device_id:02x}.0"
    ensure_idle(bdf)


def wait_until_idle(device="0", *, stable_samples=3, record_path=None):
    """Wait indefinitely on SMI's monitoring stream, not a tight polling loop.

    Require consecutive 0% gfx/UMC samples AND no other process, including
    resident workers. Query errors/timeouts fail closed, never authorize reset.
    The monitor alone is terminated on exit; no GPU workload is terminated.
    """
    if stable_samples < 1:
        raise ValueError("stable_samples must be positive")
    command = [SMI, "metric", "--gpu", device, "--usage", "--csv", "--watch", "5"]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
    stable = 0
    log = open(record_path, "a") if record_path else None
    print(f"Waiting without deadline for GPU {device}: {stable_samples} idle samples and no resident workers", flush=True)
    try:
        fields = None
        for values in csv.reader(process.stdout):
            if "gpu" in values and "gfx_activity" in values and "umc_activity" in values:
                fields = values
                continue
            if not values or values[0].startswith("'CTRL'"):
                continue
            if fields is None or len(values) != len(fields):
                raise RuntimeError(f"unrecognized GPU monitor CSV row: {values}")
            row = dict(zip(fields, values))
            data = json.loads(subprocess.check_output([SMI, "process", "--gpu", device, "--json"], text=True, timeout=15))
            others = other_processes(data, os.getpid())
            quiet = float(row["gfx_activity"]) == 0 and float(row["umc_activity"]) == 0 and not others
            stable = stable + 1 if quiet else 0
            record = {"time_utc": datetime.now(timezone.utc).isoformat(), "metrics": row,
                      "other_processes": others, "consecutive_idle": stable}
            if log:
                log.write(json.dumps(record) + "\n")
                log.flush()
            print(f"GPU {device}: gfx={row['gfx_activity']}%, UMC={row['umc_activity']}%, other_processes={len(others)}, idle={stable}/{stable_samples}", flush=True)
            if stable >= stable_samples:
                return
        raise RuntimeError("GPU monitoring stream ended before an idle window was established")
    finally:
        if log:
            log.close()
        process.terminate()
        try:
            process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()