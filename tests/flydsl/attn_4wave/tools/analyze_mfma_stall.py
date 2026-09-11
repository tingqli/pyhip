#!/usr/bin/env python3
"""Rebuild a physical-SIMD MFMA/stall ledger from gfx9 ATT JSON files."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import cached_property, lru_cache
from pathlib import Path
from typing import Any

import numpy as np


TICK_CYCLES = 4
MFMA_EXEC_CYCLES = 16
WAVE_FILE_RE = re.compile(r"se(\d+)_sm(\d+)_sl(\d+)_wv(\d+)\.json")
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]

CATEGORIES = (
    "VMEM issue",
    "VMEM wait",
    "LDS issue",
    "LDS wait",
    "VALU execution",
    "barrier",
    "other",
)
CATEGORY_RANK = {name: index for index, name in enumerate(CATEGORIES)}


@dataclass(frozen=True)
class Geometry:
    n_blocks: int
    cores_per_n: int
    mfma_per_core: int
    core_mfma_counts: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if self.n_blocks <= 0 or self.cores_per_n <= 0 or self.mfma_per_core <= 0:
            raise ValueError("geometry dimensions must be positive")
        if self.core_mfma_counts is not None and (
            len(self.core_mfma_counts) != self.cores_per_n
            or any(count <= 0 for count in self.core_mfma_counts)
        ):
            raise ValueError("core MFMA counts must be positive and match cores-per-n")

    @cached_property
    def counts_per_core(self) -> tuple[int, ...]:
        return self.core_mfma_counts or (self.mfma_per_core,) * self.cores_per_n

    @cached_property
    def core_offsets(self) -> tuple[int, ...]:
        offsets = [0]
        for count in self.counts_per_core:
            offsets.append(offsets[-1] + count)
        return tuple(offsets)

    @property
    def mfma_per_n(self) -> int:
        return self.core_offsets[-1]

    @property
    def expected_mfma_per_wave(self) -> int:
        return self.n_blocks * self.mfma_per_n

    @cached_property
    def phase_names(self) -> tuple[str, ...]:
        names = ["inactive", "prologue"]
        for core in range(self.cores_per_n):
            names.append(f"core{core}")
            if core + 1 < self.cores_per_n:
                names.append(f"core{core}->{core + 1}")
        names.extend(("tail", "drain"))
        return tuple(names)

    @property
    def core_phase_ids(self) -> set[int]:
        return {2 + 2 * core for core in range(self.cores_per_n)}

    @property
    def boundary_phase_ids(self) -> set[int]:
        return {3 + 2 * core for core in range(self.cores_per_n - 1)}

    @property
    def tail_phase_id(self) -> int:
        return 2 * self.cores_per_n + 1

    @property
    def drain_phase_id(self) -> int:
        return self.tail_phase_id + 1

    @property
    def loop_phase_ids(self) -> set[int]:
        return self.core_phase_ids | self.boundary_phase_ids | {self.tail_phase_id}


@dataclass(frozen=True)
class CodeInfo:
    asm: str
    opcode: str
    category: str
    source: str


@lru_cache(maxsize=16384)
def display_path(path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def instruction_category(assembly: str) -> str:
    opcode = assembly.strip().lower()
    if opcode.startswith("v_mfma"):
        return "MFMA"
    if opcode.startswith("s_waitcnt"):
        has_vmem = "vmcnt" in opcode
        has_lds = "lgkmcnt" in opcode
        if has_vmem and has_lds:
            return "wait-mixed"
        if has_vmem:
            return "wait-vmcnt"
        if has_lds:
            return "wait-lgkmcnt"
        return "wait-other"
    if opcode.startswith(("buffer_load", "global_load", "flat_load")):
        return "VMEM-load"
    if opcode.startswith(("buffer_store", "global_store", "flat_store")):
        return "VMEM-store"
    if opcode.startswith("ds_read"):
        return "DS-read"
    if opcode.startswith("ds_write"):
        return "DS-write"
    if opcode.startswith("ds_"):
        return "LDS/crosslane"
    if opcode.startswith("s_barrier"):
        return "barrier"
    if opcode.startswith(("v_exp", "v_rcp")):
        return "TRANS"
    if opcode.startswith("v_"):
        return "VALU"
    if opcode.startswith("s_load"):
        return "SMEM"
    if opcode.startswith("s_"):
        return "SALU/control"
    return "other"


def opcode_of(assembly: str) -> str:
    fields = assembly.strip().split()
    return fields[0] if fields else "other"


def tick_floor(cycle: int, origin: int) -> int:
    return max(0, (cycle - origin) // TICK_CYCLES)


def tick_ceil(cycle: int, origin: int) -> int:
    return max(0, (cycle - origin + TICK_CYCLES - 1) // TICK_CYCLES)


def paint(array: np.ndarray, begin: int, end: int, origin: int, value: Any) -> None:
    if end <= begin:
        return
    left = tick_floor(begin, origin)
    right = min(array.shape[-1], tick_ceil(end, origin))
    if right > left:
        array[..., left:right] = value


def quantile(values: list[int], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return float(ordered[low])
    return ordered[low] * (high - position) + ordered[high] * (position - low)


def distribution(values: list[int]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "mean": statistics.mean(values) if values else 0.0,
        "p50": statistics.median(values) if values else 0.0,
        "p95": quantile(values, 0.95),
        "max": max(values, default=0),
    }


def histogram(values: list[int]) -> dict[str, int]:
    return {str(key): count for key, count in sorted(Counter(values).items())}


def load_code(dispatch: Path) -> tuple[list[CodeInfo], int]:
    rows = json.loads((dispatch / "code.json").read_text(encoding="utf-8"))["code"]
    max_index = max(int(row[2]) for row in rows)
    code = [CodeInfo("", "other", "other", "") for _ in range(max_index + 1)]
    for row in rows:
        assembly = str(row[0]).strip()
        code[int(row[2])] = CodeInfo(
            assembly,
            opcode_of(assembly),
            instruction_category(assembly),
            str(row[3] or ""),
        )
    return code, len(rows)


def load_trace(
    dispatch: Path, geometry: Geometry
) -> tuple[list[CodeInfo], dict[tuple[int, int, int], list[dict]], dict]:
    code, static_instruction_count = load_code(dispatch)
    groups: dict[tuple[int, int, int], list[dict]] = defaultdict(list)
    trace = Counter()
    opcode_stats: dict[str, Counter] = defaultdict(Counter)

    for path in sorted(dispatch.glob("se*_sm*_sl*_wv*.json")):
        match = WAVE_FILE_RE.fullmatch(path.name)
        if match is None:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        wave_payload = payload["wave"]
        trace["wave_files"] += 1
        trace["complete_wave_files"] += int(
            payload["num_insts"] == payload["num_stitched"]
        )
        records = []
        mfma_ordinal = 0
        for raw in wave_payload["instructions"]:
            pc_index = int(raw[4])
            info = code[pc_index]
            attempt = int(raw[0])
            stall = int(raw[2])
            duration = int(raw[3])
            issue = attempt + stall
            record = {
                "attempt": attempt,
                "issue": issue,
                "complete": attempt + duration,
                "stall": stall,
                "issue_cost": duration - stall,
                "pc_index": pc_index,
                "opcode": info.opcode,
                "category": info.category,
                "asm": info.asm,
                "source": display_path(info.source) if info.source else "",
                "mfma_ordinal": mfma_ordinal if info.category == "MFMA" else None,
            }
            if info.category == "MFMA":
                mfma_ordinal += 1
            opcode_stats[info.opcode]["count"] += 1
            opcode_stats[info.opcode]["stall_cycles"] += stall
            opcode_stats[info.opcode]["service_cycles"] += duration - stall
            records.append(record)

        if mfma_ordinal == 0:
            trace["uniform_early_exit_wave_files"] += 1
            continue
        if mfma_ordinal != geometry.expected_mfma_per_wave:
            raise RuntimeError(
                f"{path}: expected {geometry.expected_mfma_per_wave} MFMA, "
                f"found {mfma_ordinal}"
            )
        trace["active_wave_files"] += 1
        key = (int(match.group(1)), int(wave_payload["cu"]), int(wave_payload["simd"]))
        groups[key].append(
            {
                "path": str(path.resolve()),
                "file": path.name,
                "begin": int(wave_payload["begin"]),
                "end": int(wave_payload["end"]),
                "slot": int(wave_payload["slot"]),
                "records": records,
            }
        )

    if not groups:
        raise RuntimeError(f"no active wave traces found under {dispatch}")
    if trace["wave_files"] != trace["complete_wave_files"]:
        raise RuntimeError("trace contains incomplete wave files")
    trace_data = dict(trace)
    trace_data["physical_simds"] = len(groups)
    trace_data["static_instructions"] = static_instruction_count
    trace_data["static_mfma_instructions"] = sum(
        info.category == "MFMA" for info in code
    )
    trace_data["opcode_stats"] = {
        opcode: dict(values) for opcode, values in sorted(opcode_stats.items())
    }
    return code, groups, trace_data


def build_blocker_names(code: list[CodeInfo]) -> list[str]:
    categories = sorted({info.category for info in code})
    return [
        "inactive",
        "scheduler/ready",
        *(f"stall:{category}" for category in categories),
        *(f"issue:{category}" for category in categories),
    ]


def paint_group(
    waves: list[dict], blocker_names: list[str], geometry: Geometry
) -> dict[str, Any]:
    waves = sorted(waves, key=lambda wave: (wave["begin"], wave["slot"]))
    origin = min(wave["begin"] for wave in waves)
    end = max(wave["end"] for wave in waves)
    ticks = tick_ceil(end, origin)
    slot_count = max(wave["slot"] for wave in waves) + 1
    phase = np.zeros((slot_count, ticks), dtype=np.uint8)
    blocker = np.zeros((slot_count, ticks), dtype=np.uint16)
    pc = np.zeros((slot_count, ticks), dtype=np.uint16)
    active = np.zeros((slot_count, ticks), dtype=np.bool_)
    mfma = np.zeros((slot_count, ticks), dtype=np.bool_)
    n_index = np.full((slot_count, ticks), -1, dtype=np.int16)
    blocker_to_id = {name: index for index, name in enumerate(blocker_names)}

    for wave in waves:
        slot = wave["slot"]
        paint(active[slot], wave["begin"], wave["end"], origin, True)
        paint(phase[slot], wave["begin"], wave["end"], origin, 1)
        paint(
            blocker[slot],
            wave["begin"],
            wave["end"],
            origin,
            blocker_to_id["scheduler/ready"],
        )
        mfmas = [record for record in wave["records"] if record["category"] == "MFMA"]
        for n_block in range(geometry.n_blocks):
            n_base = n_block * geometry.mfma_per_n
            core_ranges = []
            for core in range(geometry.cores_per_n):
                begin_index = n_base + geometry.core_offsets[core]
                core_records = mfmas[begin_index : n_base + geometry.core_offsets[core + 1]]
                core_begin = core_records[0]["issue"]
                core_end = core_records[-1]["issue"] + MFMA_EXEC_CYCLES
                core_ranges.append((core_begin, core_end))
                paint(phase[slot], core_begin, core_end, origin, 2 + 2 * core)
            for core in range(geometry.cores_per_n - 1):
                paint(
                    phase[slot],
                    core_ranges[core][1],
                    core_ranges[core + 1][0],
                    origin,
                    3 + 2 * core,
                )
            tile_end = (
                mfmas[(n_block + 1) * geometry.mfma_per_n]["issue"]
                if n_block + 1 < geometry.n_blocks
                else core_ranges[-1][1]
            )
            paint(
                phase[slot],
                core_ranges[-1][1],
                tile_end,
                origin,
                geometry.tail_phase_id,
            )
            paint(n_index[slot], core_ranges[0][0], tile_end, origin, n_block)
        paint(
            phase[slot],
            mfmas[-1]["issue"] + MFMA_EXEC_CYCLES,
            wave["end"],
            origin,
            geometry.drain_phase_id,
        )
        for record in wave["records"]:
            paint(
                blocker[slot],
                record["attempt"],
                record["issue"],
                origin,
                blocker_to_id[f"stall:{record['category']}"],
            )
            paint(
                blocker[slot],
                record["issue"],
                record["complete"],
                origin,
                blocker_to_id[f"issue:{record['category']}"],
            )
            paint(
                pc[slot],
                record["attempt"],
                record["complete"],
                origin,
                record["pc_index"] + 1,
            )
            if record["category"] == "MFMA":
                paint(
                    mfma[slot],
                    record["issue"],
                    record["issue"] + MFMA_EXEC_CYCLES,
                    origin,
                    True,
                )
    return {
        "origin": origin,
        "phase": phase,
        "blocker": blocker,
        "pc": pc,
        "active": active,
        "mfma": mfma,
        "n_index": n_index,
    }


def static_distribution(
    launch_workgroups: int,
    active_workgroups: int,
    cu_count: int,
    waves_per_wg: int,
    simds_per_cu: int,
    resident_waves: int,
) -> dict[str, Any]:
    if waves_per_wg % simds_per_cu:
        raise ValueError("waves-per-wg must divide evenly across SIMD-per-CU")
    quotient, remainder = divmod(active_workgroups, cu_count)
    tasks_per_cu = [quotient + int(index < remainder) for index in range(cu_count)]
    waves_per_task_per_simd = waves_per_wg // simds_per_cu
    waves_per_simd = [
        tasks * waves_per_task_per_simd
        for tasks in tasks_per_cu
        for _ in range(simds_per_cu)
    ]
    batches_per_simd = [math.ceil(waves / resident_waves) for waves in waves_per_simd]
    max_tasks = max(tasks_per_cu)
    wave_capacity = sum(
        simds_per_cu
        * max(waves_per_simd[index : index + simds_per_cu])
        for index in range(0, len(waves_per_simd), simds_per_cu)
    )
    batch_capacity = sum(
        simds_per_cu
        * max(batches_per_simd[index : index + simds_per_cu])
        for index in range(0, len(batches_per_simd), simds_per_cu)
    )
    return {
        "launch_workgroups": launch_workgroups,
        "active_workgroups": active_workgroups,
        "uniform_early_exit_workgroups": launch_workgroups - active_workgroups,
        "cu_count": cu_count,
        "tasks_per_cu_histogram": histogram(tasks_per_cu),
        "zero_task_cus": sum(tasks == 0 for tasks in tasks_per_cu),
        "cu_capacity_loss_fraction": 1 - active_workgroups / (cu_count * max_tasks),
        "critical_path_inflation_fraction": (
            max_tasks / (active_workgroups / cu_count) - 1
        ),
        "waves_per_wg": waves_per_wg,
        "simds_per_cu": simds_per_cu,
        "resident_waves_per_simd": resident_waves,
        "waves_per_simd_histogram": histogram(waves_per_simd),
        "resident_batches_per_simd_histogram": histogram(batches_per_simd),
        "zero_wave_simds": sum(waves == 0 for waves in waves_per_simd),
        "simd_wave_capacity_loss_fraction": 1 - sum(waves_per_simd) / wave_capacity,
        "simd_batch_capacity_loss_fraction": 1 - sum(batches_per_simd) / batch_capacity,
    }


def lifecycle(groups: dict[tuple[int, int, int], list[dict]]) -> dict[str, Any]:
    segment_values = {name: [] for name in ("prologue", "steady", "epilogue", "lifetime")}
    gaps = []
    sampled_waves_per_simd = []
    resident_batches = 0
    for waves in groups.values():
        by_slot: dict[int, list[tuple[int, int, int, int]]] = defaultdict(list)
        for wave in waves:
            mfmas = [record for record in wave["records"] if record["category"] == "MFMA"]
            by_slot[wave["slot"]].append(
                (
                    wave["begin"],
                    wave["end"],
                    mfmas[0]["issue"],
                    mfmas[-1]["issue"] + MFMA_EXEC_CYCLES,
                )
            )
        for waves_in_slot in by_slot.values():
            waves_in_slot.sort()
        batch_counts = {len(waves_in_slot) for waves_in_slot in by_slot.values()}
        if len(batch_counts) != 1:
            raise RuntimeError("resident slots have unequal active-wave counts")
        sampled_waves_per_simd.append(sum(len(value) for value in by_slot.values()))
        lifetimes = []
        for batch_index in range(batch_counts.pop()):
            batch = [by_slot[slot][batch_index] for slot in sorted(by_slot)]
            begin = min(value[0] for value in batch)
            first_mfma = min(value[2] for value in batch)
            last_mfma_end = max(value[3] for value in batch)
            end = max(value[1] for value in batch)
            parts = {
                "prologue": first_mfma - begin,
                "steady": last_mfma_end - first_mfma,
                "epilogue": end - last_mfma_end,
                "lifetime": end - begin,
            }
            if parts["prologue"] + parts["steady"] + parts["epilogue"] != parts["lifetime"]:
                raise RuntimeError("lifecycle ledger does not close")
            for name, value in parts.items():
                segment_values[name].append(value)
            lifetimes.append((begin, end))
            resident_batches += 1
        gaps.extend(
            max(0, right[0] - left[1])
            for left, right in zip(lifetimes, lifetimes[1:])
        )
    lifetime_cycles = sum(segment_values["lifetime"])
    return {
        "resident_batches": resident_batches,
        "sampled_active_waves_per_simd_histogram": histogram(sampled_waves_per_simd),
        "segments": {
            name: {
                **distribution(values),
                "cycles": sum(values),
                "lifecycle_fraction": sum(values) / lifetime_cycles,
            }
            for name, values in segment_values.items()
            if name != "lifetime"
        },
        "lifetime_cycles": lifetime_cycles,
        "inter_batch_gap": {
            **distribution(gaps),
            "cycles": sum(gaps),
            "horizon_fraction": sum(gaps) / (lifetime_cycles + sum(gaps)),
        },
    }


def classify(blocker: str, phase_name: str) -> tuple[str, str]:
    if blocker.startswith(("stall:VMEM-", "issue:VMEM-")):
        detail = "issue-stall" if blocker.startswith("stall:") else "service"
        return "VMEM issue", detail
    if blocker.startswith(
        ("stall:wait-vmcnt", "issue:wait-vmcnt", "stall:wait-mixed", "issue:wait-mixed")
    ):
        return "VMEM wait", "completion wait"
    if blocker.startswith(("stall:DS-", "issue:DS-", "stall:LDS/", "issue:LDS/")):
        detail = "issue-stall" if blocker.startswith("stall:") else "service"
        return "LDS issue", detail
    if blocker.startswith(("stall:wait-lgkmcnt", "issue:wait-lgkmcnt")):
        return "LDS wait", "completion wait"
    if blocker.startswith(("issue:VALU", "issue:TRANS")):
        return "VALU execution", "service"
    if blocker.startswith(("stall:barrier", "issue:barrier")):
        detail = "stall" if blocker.startswith("stall:") else "service"
        return "barrier", detail
    if phase_name == "tail":
        return "other", "structural tail"
    if blocker.startswith(("stall:VALU", "stall:TRANS")):
        return "other", "VALU dependency"
    if blocker.startswith("stall:MFMA"):
        return "other", "MFMA unavailable"
    if blocker.startswith(("stall:SALU", "issue:SALU")):
        return "other", "SALU/control"
    if blocker == "scheduler/ready":
        return "other", "scheduler ready"
    if blocker.startswith("issue:"):
        return "other", "other service"
    return "other", "residual"


def steady_masks(
    painted: dict[str, Any], geometry: Geometry, first_n: int, last_n: int, resident_waves: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    active = painted["active"]
    phase = painted["phase"]
    n_index = painted["n_index"]
    active_count = active.sum(axis=0)
    in_loop = np.isin(phase, list(geometry.loop_phase_ids))
    in_n = (n_index >= first_n) & (n_index < last_n)
    steady = active_count == resident_waves
    steady &= np.all(~active | (in_loop & in_n), axis=0)
    busy = steady & np.any(painted["mfma"], axis=0)
    return steady, busy, steady & ~busy


def tick_entries(
    tick: int,
    painted: dict[str, Any],
    code: list[CodeInfo],
    blocker_names: list[str],
    geometry: Geometry,
) -> list[dict[str, Any]]:
    entries = []
    for slot_value in np.flatnonzero(painted["active"][:, tick]):
        slot = int(slot_value)
        blocker = blocker_names[int(painted["blocker"][slot, tick])]
        phase_name = geometry.phase_names[int(painted["phase"][slot, tick])]
        category, detail = classify(blocker, phase_name)
        pc_index = int(painted["pc"][slot, tick]) - 1
        info = code[pc_index] if pc_index >= 0 else CodeInfo("<ready>", "<ready>", "", "")
        entries.append(
            {
                "slot": slot,
                "category": category,
                "detail": detail,
                "blocker": blocker,
                "phase": phase_name,
                "pc_index": pc_index,
                "opcode": info.opcode,
                "asm": info.asm,
                "source": display_path(info.source) if info.source else "",
            }
        )
    return entries


def classify_idle_tick(entries: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    chosen = min((entry["category"] for entry in entries), key=CATEGORY_RANK.__getitem__)
    return chosen, [entry for entry in entries if entry["category"] == chosen]


def counter_summary(counter: Counter, denominator: int) -> dict[str, dict[str, float]]:
    return {
        str(name): {
            "cycles": float(value),
            "equivalent_16cycle_slots": float(value) / MFMA_EXEC_CYCLES,
            "fraction": float(value) / denominator if denominator else 0.0,
        }
        for name, value in sorted(counter.items(), key=lambda item: item[1], reverse=True)
    }


def prioritized_stall(
    code: list[CodeInfo],
    groups: dict[tuple[int, int, int], list[dict]],
    blocker_names: list[str],
    geometry: Geometry,
    first_n: int,
    last_n: int,
    resident_waves: int,
) -> dict[str, Any]:
    main = Counter()
    details: dict[str, Counter] = defaultdict(Counter)
    opcodes: dict[str, Counter] = defaultdict(Counter)
    sources: dict[str, Counter] = defaultdict(Counter)
    phases: dict[str, Counter] = defaultdict(Counter)
    joint = Counter()
    all_same = Counter()
    total = busy = idle = 0

    for waves in groups.values():
        painted = paint_group(waves, blocker_names, geometry)
        steady_mask, busy_mask, idle_mask = steady_masks(
            painted, geometry, first_n, last_n, resident_waves
        )
        total += int(np.count_nonzero(steady_mask)) * TICK_CYCLES
        busy += int(np.count_nonzero(busy_mask)) * TICK_CYCLES
        idle += int(np.count_nonzero(idle_mask)) * TICK_CYCLES
        for tick_value in np.flatnonzero(idle_mask):
            tick = int(tick_value)
            entries = tick_entries(tick, painted, code, blocker_names, geometry)
            chosen, selected = classify_idle_tick(entries)
            main[chosen] += TICK_CYCLES
            share = TICK_CYCLES / len(selected)
            for entry in selected:
                details[chosen][entry["detail"]] += share
                phases[chosen][entry["phase"]] += share
                opcodes[chosen][entry["opcode"]] += share
                source = (
                    f"{entry['asm']} @ {entry['source']}"
                    if entry["source"]
                    else entry["asm"]
                )
                sources[chosen][source] += share
            mapped = [f"{entry['category']}@{entry['phase']}" for entry in entries]
            joint[" + ".join(sorted(mapped))] += TICK_CYCLES
            if len({entry["category"] for entry in entries}) == 1:
                all_same[entries[0]["category"]] += TICK_CYCLES

    if busy + idle != total:
        raise RuntimeError("MFMA-union ledger does not close")
    if sum(main.values()) != idle:
        raise RuntimeError("exclusive category ledger does not close")
    for category in CATEGORIES:
        if not math.isclose(sum(details[category].values()), main[category], abs_tol=1e-6):
            raise RuntimeError(f"subcategory ledger does not close for {category}")

    return {
        "first_n": first_n,
        "last_n_exclusive": last_n,
        "total_cycles": total,
        "mfma_busy_cycles": busy,
        "mfma_busy_fraction": busy / total,
        "mfma_idle_cycles": idle,
        "mfma_idle_fraction": idle / total,
        "total_16cycle_slots": total / MFMA_EXEC_CYCLES,
        "busy_16cycle_slots": busy / MFMA_EXEC_CYCLES,
        "idle_16cycle_slots": idle / MFMA_EXEC_CYCLES,
        "categories": {
            category: {
                "cycles": main[category],
                "equivalent_16cycle_slots": main[category] / MFMA_EXEC_CYCLES,
                "stall_fraction": main[category] / idle,
                "steady_fraction": main[category] / total,
                "details": counter_summary(details[category], main[category]),
                "opcodes": counter_summary(opcodes[category], main[category]),
                "phases": counter_summary(phases[category], main[category]),
                "sources": counter_summary(sources[category], main[category]),
            }
            for category in CATEGORIES
        },
        "all_waves_same_category": counter_summary(all_same, total),
        "joint_category_phase": counter_summary(joint, total),
    }


def find_wave(
    groups: dict[tuple[int, int, int], list[dict]], wave_file: str
) -> tuple[tuple[int, int, int], dict]:
    matches = [
        (key, wave)
        for key, waves in groups.items()
        for wave in waves
        if wave["file"] == Path(wave_file).name
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one wave named {wave_file}, found {len(matches)}")
    return matches[0]


def merge_tick_ranges(ticks: list[int], origin: int) -> list[list[int]]:
    if not ticks:
        return []
    ranges = []
    begin = previous = ticks[0]
    for tick in ticks[1:]:
        if tick != previous + 1:
            ranges.append([origin + begin * TICK_CYCLES, origin + (previous + 1) * TICK_CYCLES])
            begin = tick
        previous = tick
    ranges.append([origin + begin * TICK_CYCLES, origin + (previous + 1) * TICK_CYCLES])
    return ranges


def analyze_stage(
    code: list[CodeInfo],
    groups: dict[tuple[int, int, int], list[dict]],
    blocker_names: list[str],
    geometry: Geometry,
    wave_file: str,
    n_block: int,
    core: int,
) -> dict[str, Any]:
    key, wave = find_wave(groups, wave_file)
    if not 0 <= n_block < geometry.n_blocks:
        raise ValueError("stage N index is outside the configured geometry")
    if not 0 <= core < geometry.cores_per_n:
        raise ValueError("stage core index is outside the configured geometry")
    mfmas = [record for record in wave["records"] if record["category"] == "MFMA"]
    first_ordinal = n_block * geometry.mfma_per_n + geometry.core_offsets[core]
    stage_mfmas = mfmas[first_ordinal : first_ordinal + geometry.counts_per_core[core]]
    begin = stage_mfmas[0]["issue"]
    end = stage_mfmas[-1]["issue"] + MFMA_EXEC_CYCLES
    painted = paint_group(groups[key], blocker_names, geometry)
    left = tick_floor(begin, painted["origin"])
    right = tick_ceil(end, painted["origin"])
    total = (right - left) * TICK_CYCLES
    busy = int(np.count_nonzero(np.any(painted["mfma"][:, left:right], axis=0))) * TICK_CYCLES
    categories = Counter()
    details = Counter()
    idle_rows = []
    for tick in range(left, right):
        if np.any(painted["mfma"][:, tick]):
            continue
        entries = tick_entries(tick, painted, code, blocker_names, geometry)
        chosen, selected = classify_idle_tick(entries)
        categories[chosen] += TICK_CYCLES
        share = TICK_CYCLES / len(selected)
        for entry in selected:
            details[(chosen, entry["detail"])] += share
        idle_rows.append(
            {
                "begin": painted["origin"] + tick * TICK_CYCLES,
                "end": painted["origin"] + (tick + 1) * TICK_CYCLES,
                "category": chosen,
                "states": entries,
            }
        )
    merged_idle = []
    for row in idle_rows:
        if merged_idle and merged_idle[-1]["end"] == row["begin"] and merged_idle[-1]["category"] == row["category"]:
            merged_idle[-1]["end"] = row["end"]
        else:
            merged_idle.append(
                {"begin": row["begin"], "end": row["end"], "category": row["category"]}
            )
    idle = total - busy
    if sum(categories.values()) != idle:
        raise RuntimeError("stage ledger does not close")
    return {
        "physical_location": {"shader_engine": key[0], "cu": key[1], "simd": key[2]},
        "wave_file": display_path(wave["path"]),
        "resident_slot": wave["slot"],
        "n_block": n_block,
        "core": core,
        "first_mfma_ordinal": first_ordinal,
        "last_mfma_ordinal": first_ordinal + geometry.counts_per_core[core] - 1,
        "begin": begin,
        "end": end,
        "total_cycles": total,
        "mfma_instruction_records": len(stage_mfmas),
        "selected_wave_raw_mfma_stall_cycles": sum(record["stall"] for record in stage_mfmas),
        "mfma_busy_cycles": busy,
        "mfma_busy_fraction": busy / total,
        "mfma_idle_cycles": idle,
        "mfma_idle_fraction": idle / total,
        "categories": {
            category: {
                "cycles": categories[category],
                "equivalent_16cycle_slots": categories[category] / MFMA_EXEC_CYCLES,
                "stage_fraction": categories[category] / total,
            }
            for category in CATEGORIES
        },
        "details": {
            f"{category}/{detail}": cycles
            for (category, detail), cycles in sorted(details.items())
        },
        "idle_runs": merged_idle,
        "first_mfma": stage_mfmas[0],
        "last_mfma": stage_mfmas[-1],
        "mfma_issues": [
            {
                "ordinal": record["mfma_ordinal"],
                "issue": record["issue"],
                "stall": record["stall"],
                "pc_index": record["pc_index"],
            }
            for record in stage_mfmas
        ],
    }


def analyze_record_exposure(
    code: list[CodeInfo],
    groups: dict[tuple[int, int, int], list[dict]],
    blocker_names: list[str],
    geometry: Geometry,
    wave_file: str,
    attempt: int,
    pc_index: int | None,
    first_n: int,
    last_n: int,
    resident_waves: int,
) -> dict[str, Any]:
    key, wave = find_wave(groups, wave_file)
    matches = [
        record
        for record in wave["records"]
        if record["attempt"] == attempt
        and (pc_index is None or record["pc_index"] == pc_index)
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one record at attempt={attempt}, found {len(matches)}")
    record = matches[0]
    painted = paint_group(groups[key], blocker_names, geometry)
    _, _, idle_mask = steady_masks(painted, geometry, first_n, last_n, resident_waves)
    left = tick_floor(record["attempt"], painted["origin"])
    right = tick_ceil(record["issue"], painted["origin"])
    exposed_ticks = [tick for tick in range(left, right) if idle_mask[tick]]
    owned_ticks = []
    target_category, _ = classify(
        f"stall:{record['category']}",
        geometry.phase_names[int(painted["phase"][wave["slot"], left])],
    )
    for tick in exposed_ticks:
        entries = tick_entries(tick, painted, code, blocker_names, geometry)
        chosen, selected = classify_idle_tick(entries)
        if chosen == target_category and any(
            entry["slot"] == wave["slot"] and entry["pc_index"] == record["pc_index"]
            for entry in selected
        ):
            owned_ticks.append(tick)
    return {
        "physical_location": {"shader_engine": key[0], "cu": key[1], "simd": key[2]},
        "wave_file": display_path(wave["path"]),
        "resident_slot": wave["slot"],
        "attempt": record["attempt"],
        "issue": record["issue"],
        "complete": record["complete"],
        "raw_stall_cycles": record["stall"],
        "service_cycles": record["issue_cost"],
        "pc_index": record["pc_index"],
        "opcode": record["opcode"],
        "asm": record["asm"],
        "source": record["source"],
        "steady_union_idle_intervals": merge_tick_ranges(exposed_ticks, painted["origin"]),
        "steady_union_idle_cycles": len(exposed_ticks) * TICK_CYCLES,
        "exclusive_owner_intervals": merge_tick_ranges(owned_ticks, painted["origin"]),
        "exclusive_owner_cycles": len(owned_ticks) * TICK_CYCLES,
    }


def markdown_report(data: dict[str, Any]) -> str:
    static = data["static"]
    life = data["lifecycle"]
    steady = data["steady"]
    lines = [
        "# Seven-layer physical MFMA stall report",
        "",
        f"Trace: `{data['dispatch']}`",
        "",
        "## 1-2. Static distribution",
        "",
        f"- launch/active/early-exit WG: {static['launch_workgroups']}/{static['active_workgroups']}/{static['uniform_early_exit_workgroups']}",
        f"- tasks/CU: `{static['tasks_per_cu_histogram']}`; CU capacity loss: {100 * static['cu_capacity_loss_fraction']:.3f}%",
        f"- waves/SIMD: `{static['waves_per_simd_histogram']}`",
        f"- resident batches/SIMD: `{static['resident_batches_per_simd_histogram']}`",
        f"- incremental SIMD wave/batch loss: {100 * static['simd_wave_capacity_loss_fraction']:.3f}% / {100 * static['simd_batch_capacity_loss_fraction']:.3f}%",
        "",
        "## 3-5. Sampled SIMD lifecycle",
        "",
        "| Segment | cycles/batch | p50 | p95 | lifecycle |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name in ("prologue", "steady", "epilogue"):
        value = life["segments"][name]
        lines.append(
            f"| {name} | {value['mean']:.2f} | {value['p50']:.1f} | {value['p95']:.1f} | {100 * value['lifecycle_fraction']:.3f}% |"
        )
    lines.extend(
        [
            "",
            f"Inter-batch gap: {life['inter_batch_gap']['mean']:.2f} cycles average; {100 * life['inter_batch_gap']['horizon_fraction']:.3f}% of the observed horizon.",
            "",
            "## 6. Internal steady MFMA union",
            "",
            f"Window: N{steady['first_n']}..N{steady['last_n_exclusive'] - 1}; coverage of full steady: {100 * data['internal_window_coverage']:.3f}%.",
            "",
            "| Category | Cycles | 16-cycle slots | Idle share | Steady share |",
            "| --- | ---: | ---: | ---: | ---: |",
            f"| MFMA busy | {steady['mfma_busy_cycles']:,} | {steady['busy_16cycle_slots']:,.2f} | - | {100 * steady['mfma_busy_fraction']:.3f}% |",
        ]
    )
    for category in CATEGORIES:
        value = steady["categories"][category]
        lines.append(
            f"| {category} | {value['cycles']:,.0f} | {value['equivalent_16cycle_slots']:,.2f} | {100 * value['stall_fraction']:.3f}% | {100 * value['steady_fraction']:.3f}% |"
        )
    lines.extend(["", "## 7. Bottleneck summary", ""])
    for category in CATEGORIES:
        value = steady["categories"][category]
        top_opcode = next(iter(value["opcodes"]), "none")
        top_phase = next(iter(value["phases"]), "none")
        lines.append(
            f"- **{category}**: {100 * value['steady_fraction']:.3f}% steady; top opcode `{top_opcode}`; top phase `{top_phase}`."
        )

    stage = data.get("stage")
    if stage:
        location = stage["physical_location"]
        lines.extend(
            [
                "",
                f"## Selected {stage['mfma_instruction_records']}-MFMA stage",
                "",
                f"`{stage['wave_file']}`, SE{location['shader_engine']}/CU{location['cu']}/SIMD{location['simd']}/slot{stage['resident_slot']}, N{stage['n_block']}/core{stage['core']}, ordinal {stage['first_mfma_ordinal']}..{stage['last_mfma_ordinal']}.",
                "",
                f"Dynamic window: `[{stage['begin']}, {stage['end']})`; {stage['total_cycles']} cycles. Physical MFMA union busy: {stage['mfma_busy_cycles']} cycles ({100 * stage['mfma_busy_fraction']:.3f}%).",
                "",
                "| Category | Cycles | 16-cycle slots | Stage share |",
                "| --- | ---: | ---: | ---: |",
                f"| MFMA busy | {stage['mfma_busy_cycles']} | {stage['mfma_busy_cycles'] / MFMA_EXEC_CYCLES:.2f} | {100 * stage['mfma_busy_fraction']:.3f}% |",
            ]
        )
        for category in CATEGORIES:
            value = stage["categories"][category]
            if value["cycles"]:
                lines.append(
                    f"| {category} | {value['cycles']} | {value['equivalent_16cycle_slots']:.2f} | {100 * value['stage_fraction']:.3f}% |"
                )
        lines.extend(["", "Idle runs:", ""])
        for run in stage["idle_runs"]:
            lines.append(
                f"- `[{run['begin']}, {run['end']})`: {run['category']} ({run['end'] - run['begin']} cycles)"
            )

    exposure = data.get("record_exposure")
    if exposure:
        lines.extend(
            [
                "",
                "## Selected record exposure",
                "",
                f"`{exposure['asm']}` (PC index {exposure['pc_index']}, `{exposure['source']}`)",
                "",
                f"Raw stall: `[{exposure['attempt']}, {exposure['issue']})` = {exposure['raw_stall_cycles']} cycles. Steady physical union-idle intersection: `{exposure['steady_union_idle_intervals']}` = {exposure['steady_union_idle_cycles']} cycles. Exclusive owner contribution: {exposure['exclusive_owner_cycles']} cycles.",
            ]
        )
    lines.extend(
        [
            "",
            "## Closure checks",
            "",
            "- prologue + steady + epilogue == lifecycle",
            "- MFMA busy + idle == selected steady window",
            "- seven exclusive categories == MFMA idle",
            "- every category detail table == its category",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a seven-layer physical MFMA/stall report from gfx9 ATT"
    )
    parser.add_argument("dispatch", type=Path)
    parser.add_argument("--n-blocks", type=int, required=True)
    parser.add_argument("--cores-per-n", type=int, required=True)
    core_counts = parser.add_mutually_exclusive_group(required=True)
    core_counts.add_argument("--mfma-per-core", type=int)
    core_counts.add_argument("--core-mfma-counts", type=int, nargs="+", help="MFMA counts in execution order within each N tile")
    parser.add_argument("--first-n", type=int, required=True)
    parser.add_argument("--last-n-exclusive", type=int, required=True)
    parser.add_argument("--launch-workgroups", type=int, required=True)
    parser.add_argument("--active-workgroups", type=int, required=True)
    parser.add_argument("--cu-count", type=int, required=True)
    parser.add_argument("--waves-per-wg", type=int, required=True)
    parser.add_argument("--simds-per-cu", type=int, required=True)
    parser.add_argument("--resident-waves", type=int, required=True)
    parser.add_argument("--stage-wave")
    parser.add_argument("--stage-n", type=int)
    parser.add_argument("--stage-core", type=int)
    parser.add_argument("--record-wave")
    parser.add_argument("--record-attempt", type=int)
    parser.add_argument("--record-pc-index", type=int)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()
    if args.core_mfma_counts is not None and (
        len(args.core_mfma_counts) != args.cores_per_n
        or any(count <= 0 for count in args.core_mfma_counts)
    ):
        parser.error("--core-mfma-counts must contain one positive count per core")
    stage_values = (args.stage_wave, args.stage_n, args.stage_core)
    if any(value is not None for value in stage_values) and not all(
        value is not None for value in stage_values
    ):
        parser.error("--stage-wave, --stage-n and --stage-core must be used together")
    if args.record_attempt is not None and not (args.record_wave or args.stage_wave):
        parser.error("--record-attempt needs --record-wave or --stage-wave")
    if args.record_pc_index is not None and args.record_attempt is None:
        parser.error("--record-pc-index needs --record-attempt")
    return args


def main() -> None:
    args = parse_args()
    dispatch = args.dispatch.resolve()
    counts = tuple(args.core_mfma_counts) if args.core_mfma_counts is not None else None
    geometry = Geometry(args.n_blocks, args.cores_per_n, max(counts) if counts else args.mfma_per_core, counts)
    code, groups, trace = load_trace(dispatch, geometry)
    blocker_names = build_blocker_names(code)
    static = static_distribution(
        args.launch_workgroups,
        args.active_workgroups,
        args.cu_count,
        args.waves_per_wg,
        args.simds_per_cu,
        args.resident_waves,
    )
    life = lifecycle(groups)
    steady = prioritized_stall(
        code,
        groups,
        blocker_names,
        geometry,
        args.first_n,
        args.last_n_exclusive,
        args.resident_waves,
    )
    data = {
        "schema_version": 2,
        "dispatch": display_path(dispatch),
        "geometry": {
            "n_blocks": geometry.n_blocks,
            "cores_per_n": geometry.cores_per_n,
            "mfma_per_core": geometry.mfma_per_core,
            "mfma_execution_cycles": MFMA_EXEC_CYCLES,
            "att_tick_cycles": TICK_CYCLES,
        },
        "static": static,
        "trace": trace,
        "lifecycle": life,
        "internal_window_coverage": steady["total_cycles"]
        / life["segments"]["steady"]["cycles"],
        "steady": steady,
    }
    if geometry.core_mfma_counts is not None:
        data["geometry"].pop("mfma_per_core")
        data["geometry"]["core_mfma_counts"] = list(geometry.core_mfma_counts)
    if args.stage_wave is not None:
        data["stage"] = analyze_stage(
            code,
            groups,
            blocker_names,
            geometry,
            args.stage_wave,
            args.stage_n,
            args.stage_core,
        )
    if args.record_attempt is not None:
        data["record_exposure"] = analyze_record_exposure(
            code,
            groups,
            blocker_names,
            geometry,
            args.record_wave or args.stage_wave,
            args.record_attempt,
            args.record_pc_index,
            args.first_n,
            args.last_n_exclusive,
            args.resident_waves,
        )
    markdown = markdown_report(data)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(markdown, encoding="utf-8")
    print(markdown, end="")


if __name__ == "__main__":
    main()