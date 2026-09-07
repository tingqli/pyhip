"""Explicit, reversible GPU0 PTL experiments; never used by kernel dispatch.

Hardware changes require an opt-in at the benchmark CLI. No clock, power, NUMA
or other GPU settings are changed. All commands and restoration are recorded.
"""

import contextlib
import json
import os
from pathlib import Path
import subprocess


SMI = "/opt/rocm/bin/amd-smi"


def limits():
    return json.loads(subprocess.check_output([SMI, "static", "--gpu", "0", "--limit", "--json"], text=True))


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
    data = json.loads(subprocess.check_output([SMI, "process", "--gpu", device, "--json"], text=True))
    others = other_processes(data, os.getpid())
    if others:
        raise RuntimeError(f"GPU {device} has another process; refusing isolated benchmark/PTL change: {others}")


def require_idle_device():
    import torch
    prop = torch.cuda.get_device_properties(0)
    bdf = f"{prop.pci_domain_id:04x}:{prop.pci_bus_id:02x}:{prop.pci_device_id:02x}.0"
    ensure_idle(bdf)


@contextlib.contextmanager
def ptl_experiment(policy, record_path):
    """Leave state untouched for 'current', else restore the exact prior limits."""
    if policy == "current":
        yield None
        return
    if policy not in ("VECTOR,F8", "VECTOR,BF16"):
        raise ValueError(f"unsupported explicit PTL experiment: {policy}")
    # Avoid interpreting a remapped logical CUDA device as physical GPU0.
    if any(os.environ.get(name, "0") != "0" for name in
            ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")):
        raise RuntimeError("PTL experiments require HIP_VISIBLE_DEVICES=0")
    ensure_idle()
    import torch
    prop = torch.cuda.get_device_properties(0)
    if not prop.gcnArchName.startswith("gfx942"):
        raise RuntimeError("the authorized PTL experiment is restricted to gfx942 GPU0")
    before = limits()
    original = before["gpu_data"][0]["limit"]
    if original.get("ptl_state") not in ("Enabled", "Disabled"):
        raise RuntimeError("GPU0 does not report supported PTL controls; use --ptl current (no hardware changes)")
    if original["ptl_state"] != "Disabled":
        raise RuntimeError("PTL experiment expects the original Disabled state; refusing to overwrite another policy")
    record = {"policy": policy, "before": before, "commands": [], "restored": False}
    path = Path(record_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    def save():
        path.write_text(json.dumps(record, indent=2) + "\n")
    def change(option, value):
        command = ["sudo", "-n", SMI, "set", "--gpu", "0", option, value]
        result = subprocess.run(command, capture_output=True, text=True)
        record["commands"].append({"command": command, "returncode": result.returncode,
                                   "stdout": result.stdout, "stderr": result.stderr})
        save()
        result.check_returncode()
    changed = False
    save()
    try:
        change("--ptl-status", "1")
        changed = True
        change("--ptl-format", policy)
        record["during"] = limits()
        active = record["during"]["gpu_data"][0]["limit"]
        if active["ptl_state"] != "Enabled" or active["ptl_format"] != policy:
            raise RuntimeError(f"PTL readback does not match requested policy: {active}")
        save()
        yield record
    finally:
        try:
            if not changed:
                # A failing setter may still have changed the device. Check
                # before deciding whether rollback is needed; a permission
                # failure with unchanged state must not retry sudo.
                record["after_failed_enable"] = limits()
                changed = record["after_failed_enable"] != before
            if changed:
                change("--ptl-status", "0")
        finally:
            record["after"] = limits()
            record["restored"] = record["after"] == before
            save()
            print("GPU0_PTL_RESTORED", record["restored"], flush=True)
        if not record["restored"]:
            raise RuntimeError(f"GPU0 policy restoration failed; see {path}")