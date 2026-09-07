"""Read-only process guard for the logical benchmark device; never sets policy."""

import json
import os
import subprocess


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
    data = json.loads(subprocess.check_output([SMI, "process", "--gpu", device, "--json"], text=True))
    others = other_processes(data, os.getpid())
    if others:
        raise RuntimeError(f"GPU {device} has another process; refusing idle-guarded benchmark: {others}")


def require_idle_device():
    import torch
    prop = torch.cuda.get_device_properties(0)
    bdf = f"{prop.pci_domain_id:04x}:{prop.pci_bus_id:02x}:{prop.pci_device_id:02x}.0"
    ensure_idle(bdf)