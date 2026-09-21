# SPDX-License-Identifier: MIT
"""Small, Torch-free CSV helpers shared by cross_compare and moe_driver.tuned."""

import csv
import json
import math
from pathlib import Path


MODEL_FIELDS = (
    "model_dim", "inter_dim_tp", "experts", "topk", "quant_scheme", "activation",
    "preshuffle", "swiglu_limit", "beta", "linear_beta", "output_dtype",
)
KEY_FIELDS = ("gfx", "cu_num", *MODEL_FIELDS, "tokens")
CSV_FIELDS = ("model", *KEY_FIELDS, "driver", "us", "diff", "params",
              "benchmark_protocol", "aiter_status", "report")


def config_values(spec):
    """Torch-free boundary from global CLI specs or MOEconfig dictionaries."""
    values = dict(preshuffle=True, swiglu_limit=None, beta=1.0, linear_beta=1.0, output_dtype="bfloat16")
    values.update({name: spec[name] for name in MODEL_FIELDS if name in spec})
    if "inter_dim_tp" not in spec:
        values["inter_dim_tp"] = spec["inter_dim"] // spec["TP"]
        values["quant_scheme"] = spec["quant_scheme_str"]
    if values["activation"] == "swiglu" and values["swiglu_limit"] is None:
        values["swiglu_limit"] = 7.0
    values["output_dtype"] = str(values["output_dtype"]).removeprefix("torch.")
    for name in ("beta", "linear_beta", "swiglu_limit"):
        if values[name] is not None:
            values[name] = float(values[name])
    return values


def model_key(spec, device):
    """Canonical local-shard config; actual M is a separate exact key."""
    values = {**config_values(spec), **device}
    return {name: "" if values[name] is None else str(values[name])
            for name in KEY_FIELDS if name != "tokens"}


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not set(CSV_FIELDS).issubset(reader.fieldnames or ()):
            raise ValueError(f"unrecognized MoE tuned CSV: {path}; regenerate it for the function-based driver API")
        return list(reader)


def best_row(batch):
    valid = [row for row in batch["rows"] if row["status"] == "OK"
             and row.get("e2e_latency_s") is not None
             and math.isfinite(row["e2e_latency_s"]) and row["e2e_latency_s"] > 0]
    return min(valid, key=lambda row: row["e2e_latency_s"], default=None)


def update_csv(path, report, batch):
    """Replace one measured key, preserving other shapes. Single writer; no locks."""
    path = Path(path)
    key = {**model_key(batch["spec"], report["device"]), "tokens": str(batch["spec"]["tokens"])}
    rows = read_rows(path) if path.exists() else []
    rows = [row for row in rows if any(row[name] != value for name, value in key.items())]
    best = best_row(batch)
    if best is not None:
        params = {}
        if best["candidate"] == "aiter":
            # Replay the measured row, not a future/changed Aiter CSV winner.
            params["tuned_config"] = batch["tuning"]["tuned_config"]
        rows.append(dict(model=report["model"], **key, driver=best["candidate"],
                         us=best["e2e_latency_s"] * 1e6, diff=best["diff"],
                         params=json.dumps(params, sort_keys=True, allow_nan=False),
                         benchmark_protocol=report["benchmark_protocol"],
                         aiter_status=batch["rows"][0]["status"],
                         report=str(Path(report["artifact_dir"]) / "comparison.json")))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)