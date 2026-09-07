"""Environment metadata and JSON output for the single MHA test entry."""

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

import torch

if __package__:
    from ._testing import gpu_arch
else:
    from _testing import gpu_arch


HERE = Path(__file__).resolve().parent


def environment():
    result = {"time_utc": datetime.now(timezone.utc).isoformat(), "arch": gpu_arch(),
              "torch": torch.__version__, "hip": torch.version.hip,
              "flydsl": importlib.metadata.version("flydsl"),
              "python": sys.version, "interpreter": sys.executable,
              "device_environment": {name: os.environ.get(name) for name in
                  ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")},
              "sources_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob("*.py")}}
    if torch.cuda.is_available():
        prop = torch.cuda.get_device_properties(0)
        result.update(gpu=prop.name, compute_units=prop.multi_processor_count)
        bdf = f"{prop.pci_domain_id:04x}:{prop.pci_bus_id:02x}:{prop.pci_device_id:02x}.0"
        try:
            result["limits"] = json.loads(subprocess.check_output(["/opt/rocm/bin/amd-smi", "static", "--gpu", bdf, "--limit", "--json"], text=True))
        except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
            result["limits_unavailable"] = str(exc)
    return result


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")