#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""SE dispatch examples in one file: model, GPU probes, raw recheck and tests.

Subcommands:
  model         CPU illustration of ordered admission versus completion rounds.
  run           Five bounded gfx950 GPU examples; --out must be a new directory.
  analyze       CPU recheck of saved per-wave records, including the legacy run.
  moe-evidence  Recheck the existing M128 sort256/sort128 placement comparison.
    plot-mappings Plot identity/width4/width8 physical XCD maps and saved timings.
  test          CPU-only unit tests, without importing Torch or initializing HIP.

The embedded HIP source is compiled in a temporary directory and removed after
the build; no separately maintained C++ or helper/test script is needed.
Entry/exit timestamps observe execution, not the
hardware's internal admission event. Nothing waits for another CTA on the GPU.
"""

import argparse
from collections import Counter, defaultdict
import ctypes
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import gzip
import hashlib
import heapq
import html
import json
from pathlib import Path
import shutil
from statistics import mean
import subprocess
import sys
import tempfile
import unittest


BASE = Path(__file__).resolve().parent
DEFAULT_MOE_DATA = BASE.parents[1] / "moe_8w_down/try/sort_alignment_20260913/default/results.json"
LEGACY_SOURCE_SHA256 = {
    "probe-se-dispatch.py": "d92631657da60f601fc09e6f15341dd19b07b2b65f228ce17096065164444bf3",
    "se_dispatch_probe.cpp": "5f83488225015ec7d1dae1ce61fe988b477c766a5c929ba67daa6c6dfb92e09a",
}
PRE_PLOT_SCRIPT_SHA256 = "10b481eaf90bfdfb90281e16c956d9d2ca9a7d03c1b051c4fee434bd4f85f892"

# Byte-identical to the previously measured HIP source. Its captured hash is
# checked when analyzing legacy data; the old Python fingerprint is recognized,
# not falsely claimed to match this newly consolidated analyzer.
HIP_SOURCE = r'''// SPDX-License-Identifier: MIT
// Passive per-wave timestamps: no inter-CTA barrier, counter, or work queue.
#include <hip/hip_runtime.h>
#include <cstdint>

#ifndef LDS_BYTES
#define LDS_BYTES 98304
#endif

__device__ __forceinline__ unsigned read_xcc() {
    unsigned value;
    asm volatile("s_getreg_b32 %0, hwreg(HW_REG_XCC_ID, 0, 4)" : "=s"(value));
    return value;
}

__device__ __forceinline__ unsigned read_hw_id() {
    unsigned value;
    asm volatile("s_getreg_b32 %0, hwreg(HW_REG_HW_ID)" : "=s"(value));
    return value;
}

// Modes: 0 all short; 1 bid0 long; 2 one SE's initial capacity all long;
// 3 same, but its first block ends early; 4 equal long count spread over SEs.
__global__ __launch_bounds__(256) void se_dispatch_probe(
        uint64_t* records, unsigned mode, unsigned short_ticks,
        unsigned release_ticks, unsigned long_ticks, unsigned prefix_blocks,
        unsigned xcd_count, unsigned target_xcc, unsigned target_se) {
    const uint64_t entry = wall_clock64();
    const unsigned tid = threadIdx.x, bid = blockIdx.x;
    const unsigned xcc = read_xcc(), hw = read_hw_id();
    const unsigned se = (hw >> 13) & 7;
    __shared__ __align__(256) volatile unsigned char reserve[LDS_BYTES];
    reserve[tid] = static_cast<unsigned char>(tid);
    if (tid == 0) reserve[LDS_BYTES - 1] = 0xA5;
    __syncthreads();

    unsigned ticks = short_ticks;
    if (mode == 1 && bid == 0) ticks = long_ticks;
    if ((mode == 2 || mode == 3) && xcc == target_xcc && se == target_se
            && bid < prefix_blocks) {
        ticks = mode == 3 && bid < xcd_count * 4 ? release_ticks : long_ticks;
    }
    if (mode == 4 && xcc == target_xcc && bid < prefix_blocks / 4)
        ticks = long_ticks;

    const uint64_t begin = wall_clock64();
    uint64_t end;
    do {
        // Bounded time budget, no load polling or atomic synchronization.
        asm volatile("s_nop 7" ::: "memory");
        end = wall_clock64();
    } while (end - begin < ticks);
    // Only a local CTA barrier: LDS stays live until every wave has finished.
    __syncthreads();
    const unsigned checksum = reserve[tid] ^ reserve[LDS_BYTES - 1];
    const uint64_t exit = wall_clock64();
    if (tid % 64 == 0) {
        uint64_t* row = records + (bid * 4 + tid / 64) * 10;
        row[0] = xcc; row[1] = hw;
        row[2] = entry; row[3] = begin; row[4] = end; row[5] = exit;
        row[6] = bid; row[7] = tid / 64; row[8] = ticks; row[9] = checksum;
    }
}

extern "C" int probe_attributes(int* values) {
    hipFuncAttributes attributes{};
    hipError_t error = hipFuncGetAttributes(&attributes, reinterpret_cast<const void*>(se_dispatch_probe));
    if (error != hipSuccess) return static_cast<int>(error);
    int occupancy = 0;
    error = hipOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy, se_dispatch_probe, 256, 0);
    if (error != hipSuccess) return static_cast<int>(error);
    values[0] = static_cast<int>(attributes.sharedSizeBytes);
    values[1] = static_cast<int>(attributes.localSizeBytes);
    values[2] = attributes.numRegs;
    values[3] = occupancy;
    return 0;
}

extern "C" int launch_probe(void* records, unsigned grid, unsigned mode,
        unsigned short_ticks, unsigned release_ticks, unsigned long_ticks,
        unsigned prefix_blocks, unsigned xcd_count, unsigned target_xcc,
        unsigned target_se, void* stream) {
    hipLaunchKernelGGL(se_dispatch_probe, dim3(grid), dim3(256), 0,
        reinterpret_cast<hipStream_t>(stream), static_cast<uint64_t*>(records),
        mode, short_ticks, release_ticks, long_ticks, prefix_blocks,
        xcd_count, target_xcc, target_se);
    return static_cast<int>(hipGetLastError());
}'''

MODES = {0: "uniform_short", 1: "one_straggler", 2: "one_se_saturated",
         3: "one_se_partial_release", 4: "spread_same_work"}
FIELDS = ["xcc", "hw_id", "entry", "work_begin", "work_end", "exit",
          "bid", "wave", "budget_ticks", "checksum"]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def source_hashes():
    return {"script": digest(Path(__file__).read_bytes()), "embedded_hip": digest(HIP_SOURCE.encode())}


def check_provenance(captured):
    if captured == source_hashes():
        return "current single-file script and embedded HIP match the capture"
    if captured == LEGACY_SOURCE_SHA256 and digest(HIP_SOURCE.encode()) == captured["se_dispatch_probe.cpp"]:
        return "legacy capture recognized; GPU source identical; current analyzer independently recomputes old results"
    if captured == {"script": PRE_PLOT_SCRIPT_SHA256, "embedded_hip": digest(HIP_SOURCE.encode())}:
        return "pre-plot single-file capture recognized; HIP unchanged; current analyzer independently recomputes old results"
    raise ValueError("Unrecognized or changed source fingerprint; do not silently relabel old results")


def write_json(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


@dataclass(frozen=True)
class Block:
    bid: int
    xcc: int
    se: int
    cu: int
    entry: int
    ready: int
    work_end: int
    exit: int
    budget: int

    @property
    def location(self):
        return self.xcc, self.se, self.cu


def decode_blocks(raw, grid):
    assert len(raw) == grid
    blocks = []
    for bid, waves in enumerate(raw):
        assert len(waves) == 4 and all(len(row) == len(FIELDS) for row in waves)
        locations, budgets = set(), set()
        for wave, row in enumerate(waves):
            xcc, hw, entry, begin, end, exit_tick, rb, rw, budget, checksum = row
            assert rb == bid and rw == wave and entry <= begin <= end <= exit_tick
            assert end - begin >= budget > 0 and checksum == ((wave * 64) ^ 0xA5)
            locations.add((xcc, (hw >> 13) & 7, (hw >> 8) & 15))
            budgets.add(budget)
        assert len(locations) == len(budgets) == 1
        xcc, se, cu = locations.pop()
        blocks.append(Block(bid, xcc, se, cu, min(r[2] for r in waves),
            max(r[3] for r in waves), min(r[4] for r in waves),
            max(r[5] for r in waves), budgets.pop()))
    return blocks


def longest_idle_gap(intervals, begin, end):
    """Conservative gaps outside all observed [first-entry,last-exit] spans."""
    if end <= begin:
        return begin, begin
    cursor, best = begin, (begin, begin)
    for left, right in sorted(intervals):
        if right <= begin or left >= end:
            continue
        left, right = max(left, begin), min(right, end)
        if left > cursor and left - cursor > best[1] - best[0]:
            best = cursor, left
        cursor = max(cursor, right)
    if end - cursor > best[1] - best[0]:
        best = cursor, end
    return best


def refill_witnesses(blocks, *, prefix, target, begin, end, guard, short_ticks):
    """Same CU reused by a post-prefix CTA while a different SE remains busy."""
    predecessors = defaultdict(list)
    for block in blocks:
        if block.bid < prefix and block.xcc == target[0] and block.se != target[1] and block.budget == short_ticks:
            predecessors[block.location].append(block)
    witnesses = []
    for later in sorted(blocks, key=lambda b: b.entry):
        if later.bid < prefix or later.xcc != target[0] or later.se == target[1]:
            continue
        if not begin + guard < later.entry < end - guard:
            continue
        earlier = [p for p in predecessors[later.location] if p.exit + guard < later.entry]
        if earlier:
            previous = max(earlier, key=lambda p: p.exit)
            witnesses.append({"predecessor_bid": previous.bid, "successor_bid": later.bid,
                "location": list(later.location), "predecessor_exit": previous.exit,
                "successor_entry": later.entry, "busy_window_begin": begin, "busy_window_end": end})
    return witnesses


def expected_budget(bid, xcc, se, mode, settings):
    short, release, long = (settings[k] for k in ("short_ticks", "release_ticks", "long_ticks"))
    target = (settings["target_xcc"], settings["target_se"])
    if mode == 1 and bid == 0:
        return long
    if mode in (2, 3) and (xcc, se) == target and bid < settings["prefix_blocks"]:
        return release if mode == 3 and bid < settings["xcd_count"] * 4 else long
    if mode == 4 and xcc == target[0] and bid < settings["prefix_blocks"] // 4:
        return long
    return short


def analyze(raw, mode, settings):
    blocks = decode_blocks(raw, settings["grid"])
    assert all(b.budget == expected_budget(b.bid, b.xcc, b.se, mode, settings) for b in blocks)
    locations = {b.location for b in blocks}
    assert len(locations) == settings["cu_count"], ("observed CU count", len(locations))
    domains = Counter((x, s) for x, s, _ in locations)
    assert len(domains) == settings["xcd_count"] * 4
    assert len(set(domains.values())) == 1
    prefix, guard, short = settings["prefix_blocks"], settings["guard_ticks"], settings["short_ticks"]
    target = (blocks[0].xcc, blocks[0].se) if mode == 1 else (settings["target_xcc"], settings["target_se"])
    target_cus = {loc for loc in locations if loc[:2] == target}
    holders = [b for b in blocks if b.bid < prefix and (b.xcc, b.se) == target and b.budget > short]
    fast = [b for b in blocks if b.xcc == target[0] and b.se != target[1]]
    late_holders = [b for b in holders if b.budget == settings["long_ticks"]]
    early_holders = [b for b in holders if b.budget == settings["release_ticks"]]
    if mode == 1:
        assert len(holders) == 1
    if mode in (2, 3):
        assert len(holders) == len(target_cus) * settings["resident_ctas_per_cu"]
    if mode == 3:
        assert len(early_holders) == 1, "partial-release case did not select exactly one early block"
    target_counts = Counter(b.location for b in holders)
    resources_filled = bool(holders) and set(target_counts) == target_cus and all(
        count == settings["resident_ctas_per_cu"] for count in target_counts.values())
    full_begin = max((b.ready for b in holders), default=0)
    first_release = min((b.work_end for b in holders), default=0)
    saturated = resources_filled and first_release - full_begin > 2 * guard
    gap = longest_idle_gap([(b.entry, b.exit) for b in fast], full_begin + guard, first_release - guard) if saturated else (0, 0)
    future_after_gap = sum(b.entry > gap[1] for b in fast) if gap[1] > gap[0] else 0
    all_long_still_running_until = min((b.work_end for b in late_holders), default=0)
    witness_begin = max((b.ready for b in late_holders), default=0)
    if mode == 3:
        witness_begin = max(witness_begin, max(b.exit for b in early_holders))
    witnesses = refill_witnesses(blocks, prefix=prefix, target=target, begin=witness_begin,
        end=all_long_still_running_until, guard=guard, short_ticks=short) if late_holders else []
    phases = defaultdict(set)
    for b in blocks:
        phases[(b.xcc, b.bid // settings["xcd_count"] % 4)].add(b.se)
    origin = min(b.entry for b in blocks)
    ticks_us = settings["ticks_per_us"]
    sample = {"mode": MODES[mode], "origin_tick": origin, "target": list(target),
        "observed_cus": len(locations), "observed_se_count": len(domains),
        "horizon_us": (max(b.exit for b in blocks) - origin) / ticks_us,
        "budget_histogram": dict(Counter(b.budget for b in blocks)),
        "bid_mod_xcd_mismatches": sum(b.xcc != b.bid % settings["xcd_count"] for b in blocks),
        "fixed_four_phase_se_mapping": all(len(v) == 1 for v in phases.values()),
        "holders": [asdict(b) for b in holders], "target_initial_capacity_filled": saturated,
        "target_capacity_per_cu": dict((str(key), value) for key, value in target_counts.items()),
        "full_capacity_begin_us": (full_begin - origin) / ticks_us if holders else None,
        "first_possible_release_us": (first_release - origin) / ticks_us if holders else None,
        "all_remaining_long_blocks_busy_until_us": (all_long_still_running_until - origin) / ticks_us if late_holders else None,
        "early_release_exit_us": (max(b.exit for b in early_holders) - origin) / ticks_us if early_holders else None,
        "fast_se_idle_gap_us": (gap[1] - gap[0]) / ticks_us,
        "fast_se_idle_gap_ticks": list(gap), "fast_ctas_starting_after_gap": future_after_gap,
        "post_prefix_refill_witness_count": len(witnesses), "refill_witnesses": witnesses[:8],
        "first_refill_us": (witnesses[0]["successor_entry"] - origin) / ticks_us if witnesses else None,
        "strict_completion_round_refuted": mode in (1, 3) and bool(witnesses),
        "admission_blocking_consistent": saturated and gap[1] - gap[0] > 2 * guard and future_after_gap > 0,
        "per_se_last_exit_us": {f"xcc{x}.se{s}": (max(b.exit for b in blocks if b.xcc == x and b.se == s) - origin) / ticks_us for x, s in domains}}
    return sample, blocks


def draw_svg(path, blocks, sample, ticks_per_us):
    """Target XCD only: one row/CU, passive [entry,exit] block spans."""
    xcc = sample["target"][0]
    chosen = [b for b in blocks if b.xcc == xcc]
    lanes = sorted({(b.se, b.cu) for b in chosen})
    origin = sample["origin_tick"]
    span = max(b.exit for b in chosen) - origin
    width, left, scale, row_h = 1100, 120, 930 / span, 20
    height = 90 + len(lanes) * row_h
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
           '<rect width="100%" height="100%" fill="white"/>',
           '<style>text{font:12px sans-serif}</style>',
           f'<text x="10" y="20">{html.escape(sample["mode"])} / XCD{xcc}; long=red, early=orange, short=blue; execution spans, not admission timestamps</text>']
    long_budget, short_budget = max(b.budget for b in chosen), min(b.budget for b in chosen)
    for index, (se, cu) in enumerate(lanes):
        y = 50 + index * row_h
        svg.append(f'<text x="8" y="{y+12}">SE{se} CU{cu}</text>')
        for block in [b for b in chosen if b.se == se and b.cu == cu]:
            color = '#dc2626' if block.budget == long_budget and long_budget > short_budget else '#2563eb' if block.budget == short_budget else '#d97706'
            x = left + (block.entry - origin) * scale
            w = max(.6, (block.exit - block.entry) * scale)
            svg.append(f'<rect x="{x:.3f}" y="{y}" width="{w:.3f}" height="12" fill="{color}" opacity="0.65"><title>bid={block.bid}; {(block.entry-origin)/ticks_per_us:.3f}–{(block.exit-origin)/ticks_per_us:.3f} us</title></rect>')
    for i in range(6):
        x = left + 930 * i / 5
        svg.append(f'<text x="{x:.1f}" y="{height-15}">{span*i/5/ticks_per_us:.1f} us</text>')
    svg.append('</svg>')
    path.write_text('\n'.join(svg) + '\n')


def device_info(torch):
    from pyhip.core.hiptools import get_lib, hip_check_error
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    if not props.gcnArchName.startswith("gfx950"):
        raise RuntimeError("This probe currently validates gfx950 only")
    lib = get_lib()
    query = lib.hipDeviceGetAttribute
    query.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
    query.restype = ctypes.c_int
    def attr(number):
        result = ctypes.c_int()
        hip_check_error(query(ctypes.byref(result), number, device))
        return result.value
    pci = ctypes.create_string_buffer(64)
    lib.hipDeviceGetPCIBusId.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
    hip_check_error(lib.hipDeviceGetPCIBusId(pci, len(pci), device))
    return {"name": props.name, "arch": props.gcnArchName, "logical_device": device,
            "pci_bus_id": pci.value.decode(), "cu_count": props.multi_processor_count,
            "xcd_count": attr(10018), "lds_bytes_per_cu": attr(10002),
            "wall_clock_khz": attr(10017), "torch": torch.__version__, "hip": torch.version.hip,
            "python": sys.executable}


def build_library(out, lds):
    hipcc = Path(shutil.which("hipcc") or "/opt/rocm/bin/hipcc").resolve()
    target = out / f"se_dispatch_lds{lds}.so"
    assert not target.exists()
    # HIP compilation reads the source once for device and again for host.
    # A pipe/stdin is consumed by the first pass, losing the host wrappers.
    with tempfile.TemporaryDirectory(prefix="se-dispatch-") as directory:
        source = Path(directory) / "probe.cpp"
        source.write_text(HIP_SOURCE)
        command = [str(hipcc), "-std=c++17", "-O3", "-shared", "-fPIC", "--offload-arch=gfx950",
                   f"-DLDS_BYTES={lds}", str(source), "-o", str(target)]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
    (out / f"build_lds{lds}.log").write_text(completed.stdout + completed.stderr)
    if completed.returncode:
        raise RuntimeError(f"HIP compilation failed ({completed.returncode}): {completed.stderr}")
    library = ctypes.CDLL(str(target))
    library.probe_attributes.argtypes = [ctypes.POINTER(ctypes.c_int)]
    library.probe_attributes.restype = ctypes.c_int
    attributes = (ctypes.c_int * 4)()
    error = library.probe_attributes(attributes)
    if error:
        raise RuntimeError(f"HIP attribute/occupancy query failed: {error}")
    library.launch_probe.argtypes = [ctypes.c_void_p, *([ctypes.c_uint] * 9), ctypes.c_void_p]
    library.launch_probe.restype = ctypes.c_int
    return library, {"static_lds_bytes": attributes[0], "private_bytes": attributes[1],
                     "registers": attributes[2], "resident_ctas_per_cu": attributes[3],
                     "build_command": command, "binary_sha256": digest(target.read_bytes())}


def print_summary(samples):
    print("| LDS | Pattern | Horizon us | Fast-SE idle us | Post-prefix refills | Round refuted |")
    print("|---:|---|---:|---:|---:|---|")
    for lds in sorted({row["lds_bytes"] for row in samples}):
        for mode in MODES.values():
            rows = [r for r in samples if r["lds_bytes"] == lds and r["mode"] == mode]
            if rows:
                print(f"| {lds} | {mode} | {mean(r['horizon_us'] for r in rows):.3f} | {mean(r['fast_se_idle_gap_us'] for r in rows):.3f} | {sum(r['post_prefix_refill_witness_count'] for r in rows)} | {any(r['strict_completion_round_refuted'] for r in rows)} |")


def run(args):
    import torch
    if args.out.exists():
        raise ValueError("Choose a new output directory; previous evidence is never overwritten")
    assert 0 < args.short_us < args.release_us < args.long_us <= 5000
    assert 0 < args.guard_us < args.short_us / 2 and 1 <= args.repeats <= 10
    assert args.lds and len(set(args.lds)) == len(args.lds)
    info = device_info(torch)
    assert 0 <= args.target_xcc < info["xcd_count"] and 0 <= args.target_se < 4
    args.out.mkdir(parents=True)
    before = source_hashes()
    ticks_us = info["wall_clock_khz"] / 1000
    samples, resources = [], {}
    for lds in args.lds:
        assert info["lds_bytes_per_cu"] // 3 < lds <= info["lds_bytes_per_cu"]
        library, resource = build_library(args.out, lds)
        resources[str(lds)] = resource
        assert resource["static_lds_bytes"] >= lds and resource["private_bytes"] == 0
        occupancy = resource["resident_ctas_per_cu"]
        assert occupancy in (1, 2) and occupancy == info["lds_bytes_per_cu"] // resource["static_lds_bytes"]
        prefix = info["cu_count"] * occupancy
        assert args.grid >= prefix * 4 and args.grid % (info["xcd_count"] * 4) == 0
        settings = {"grid": args.grid, "cu_count": info["cu_count"], "xcd_count": info["xcd_count"],
                    "resident_ctas_per_cu": occupancy, "prefix_blocks": prefix,
                    "target_xcc": args.target_xcc, "target_se": args.target_se,
                    "ticks_per_us": ticks_us,
                    **{key + "_ticks": round(getattr(args, key + "_us") * ticks_us) for key in ("short", "release", "long", "guard")}}
        records = torch.empty((args.grid, 4, len(FIELDS)), dtype=torch.int64, device="cuda")
        stream = torch.cuda.current_stream().cuda_stream
        def launch(mode):
            status = library.launch_probe(records.data_ptr(), args.grid, mode, settings["short_ticks"],
                settings["release_ticks"], settings["long_ticks"], prefix, info["xcd_count"],
                args.target_xcc, args.target_se, stream)
            if status:
                raise RuntimeError(f"HIP launch error: {status}")
            torch.cuda.synchronize()
        launch(0)
        print("PROBE_RESOURCES", lds, json.dumps(resource), flush=True)
        for repeat in range(args.repeats):
            for mode in list(MODES) if repeat % 2 == 0 else list(MODES)[::-1]:
                records.fill_(-1)
                launch(mode)
                raw = records.cpu().tolist()
                key = f"lds{lds}_{MODES[mode]}_{repeat}"
                with gzip.open(args.out / f"{key}.json.gz", "wt") as file:
                    json.dump({"fields": FIELDS, "settings": settings, "mode": mode, "raw": raw}, file)
                sample, blocks = analyze(raw, mode, settings)
                sample.update(key=key, lds_bytes=lds, repeat=repeat)
                write_json(args.out / f"{key}.json", sample)
                if repeat == 0 and mode in (1, 2, 3, 4):
                    draw_svg(args.out / f"{key}.svg", blocks, sample, ticks_us)
                samples.append(sample)
                print("SE_SAMPLE", json.dumps({k: sample[k] for k in ("key", "horizon_us", "target_initial_capacity_filled", "fast_se_idle_gap_us", "post_prefix_refill_witness_count", "strict_completion_round_refuted", "admission_blocking_consistent")}), flush=True)
    assert source_hashes() == before
    for lds in args.lds:
        for repeat in range(args.repeats):
            a = next(s for s in samples if s["lds_bytes"] == lds and s["repeat"] == repeat and s["mode"] == "one_se_saturated")
            b = next(s for s in samples if s["lds_bytes"] == lds and s["repeat"] == repeat and s["mode"] == "spread_same_work")
            assert a["budget_histogram"] == b["budget_histogram"], "equal-work control mismatch"
    result = {"device": info, "resources": resources, "source_sha256": before,
              "utc": datetime.now(timezone.utc).isoformat(), "samples": samples,
              "strict_completion_round_refuted": any(s["strict_completion_round_refuted"] for s in samples),
              "notes": ["No inter-CTA barrier, atomics, queue settings, or CU mask; only bounded local delays.",
                        "Entry/exit are observed execution timestamps, not exact hardware admission/retirement.",
                        "Refill witnesses compare one physical XCD, not cross-XCD clock ordering.",
                        "A counterexample refutes strict completion rounds; none found is inconclusive.",
                        "Idle gaps support admission/HOL blocking, not exact firmware or queue depths."]}
    write_json(args.out / "results.json", result)
    print_summary(samples)
    print("SE_DISPATCH_PROBE_PASS", len(samples), "strict_round_refuted", result["strict_completion_round_refuted"], flush=True)


def reanalyze(args):
    result = json.loads((args.out / "results.json").read_text())
    print("PROVENANCE", check_provenance(result["source_sha256"]))
    for lds, resource in result["resources"].items():
        assert digest((args.out / f"se_dispatch_lds{lds}.so").read_bytes()) == resource["binary_sha256"]
    for row in result["samples"]:
        with gzip.open(args.out / f"{row['key']}.json.gz", "rt") as file:
            stored = json.load(file)
        assert stored["fields"] == FIELDS
        sample, _ = analyze(stored["raw"], stored["mode"], stored["settings"])
        sample.update(key=row["key"], lds_bytes=row["lds_bytes"], repeat=row["repeat"])
        assert json.loads(json.dumps(sample)) == row
        assert json.loads((args.out / f"{row['key']}.json").read_text()) == row
    print_summary(result["samples"])
    print("SE_DISPATCH_RAW_RECHECK_PASS", len(result["samples"]))


def admission_model(durations, slots_per_se=1, completion_round=False):
    """Explicit CPU model, NOT an inference from or measurement of hardware."""
    assert slots_per_se > 0 and all(duration > 0 for duration in durations)
    available = [[0] * slots_per_se for _ in range(4)]
    now, result = 0, []
    for bid, duration in enumerate(durations):
        se = bid % 4
        if completion_round and bid and bid % 4 == 0:
            now = max(row["end_us"] for row in result)
        ready = heapq.heappop(available[se])
        start = max(now, ready)
        result.append({"bid": bid, "se": se, "start_us": start,
                       "end_us": start + duration, "dispatch_wait_us": start - now})
        heapq.heappush(available[se], start + duration)
        now = start
    return result


def mapped_task(bid, tasks, width):
    chunk = tasks // width if width else 0
    return bid % width * chunk + bid // width if width and bid < chunk * width else bid


def model_examples(_args):
    durations = [10, 100, 5, 5, 20, 10, 10, 10]
    ordered = admission_model(durations)
    barrier = admission_model(durations, completion_round=True)
    print("CPU MODEL ONLY: four SEs, one slot each, no look-ahead buffer; not measured GPU timing.")
    print("| WG | SE | Budget us | Ordered start/end | Wait-all start/end |")
    print("|---:|---:|---:|---|---|")
    for a, b, duration in zip(ordered, barrier, durations):
        print(f"| {a['bid']} | {a['se']} | {duration} | {a['start_us']}/{a['end_us']} | {b['start_us']}/{b['end_us']} |")
    assert ordered[4]["start_us"] == 10 and ordered[5]["start_us"] == ordered[6]["start_us"] == 100
    print("At10us SE0 accepts WG4; SE1 blocks WG5 until100us. SE2/3 are free since5us but WG6/7 cannot bypass.")
    print("WG4 still executes10–30us while dispatch is blocked: blocked admission is not blocked execution.")
    old_live = [row < 3 or expert < 5 for expert in range(384) for row in range(4)]
    native_live = [value for value in old_live if value]
    print("SORTING MODEL (five fourth-block experts placed first for illustration, not the actual expert ordering):")
    for label, live in (("sort256/M128", old_live), ("sort128/M128", native_live)):
        print(label, "Mmod4", [sum(live[i::4]) for i in range(4)])
        tasks = len(live) * 8
        for width in (0, 4):
            counts = Counter({(xcd, phase): 0 for xcd in range(8) for phase in range(4)})
            for bid in range(tasks):
                task = mapped_task(bid, tasks, width)
                counts[(bid % 8, bid // 8 % 4)] += live[task // 8]
            print(" width", width, "XCD0 SE-PHASE live counts", [counts[(0, p)] for p in range(4)])
    print("MODEL_EXAMPLES_PASS; use run/analyze for physical evidence and moe-evidence for actual GEMM records")


def moe_evidence(args):
    """Audit existing real-GEMM evidence; no other experiment scripts imported."""
    import torch
    path = args.data.resolve()
    data = json.loads(path.read_text())
    assert data["source_unchanged"] and data["shape"] == {"n": 6144, "k": 256, "experts": 384, "topk": 8}
    batch = next(b for b in data["results"] if b["batch"] == 16384)
    raw = torch.load(path.parent / "batch_16384_placement.pt", map_location="cpu", weights_only=True)
    names = ("m128_sort256_identity", "m128_sort256_w4", "m128_identity", "m128_w4")
    print("SAVED GPU EVIDENCE: previous same-process comparisons, NOT a new GEMM benchmark.")
    print("| Candidate | Sort M | Live M mod4 | Empty M | SE max/mean | Down us | Full us |")
    print("|---|---:|---|---:|---:|---:|---:|")
    for name in names:
        config, routing = batch["configs"][name], batch["routing"][name]
        records, live = raw[name].tolist(), raw[name + "_live_m"].tolist()
        tasks = len(live) * config["num_oc_splits"]
        width = config["xcd_count"] if config["xcd_swizzle"] else 0
        per_se, per_xcd, a_domains = Counter(), Counter(), defaultdict(set)
        for bid, (xcc, hw, task, active) in enumerate(records):
            assert task == mapped_task(bid, tasks, width) and xcc == bid % 8
            assert active == int(task < tasks and live[task // 8])
            per_se[(xcc, (hw >> 13) & 7)] += active
            per_xcd[xcc] += active
            if active:
                a_domains[task // 8].add(xcc)
        expected = next(row for row in batch["placement"] if row["candidate"] == name)
        assert {f"xcd{x}.se{s}": value for (x, s), value in per_se.items()} == expected["se_active_counts"]
        assert {str(x): value for x, value in per_xcd.items()} == expected["xcd_active_counts"]
        load = max(per_se.values()) / mean(per_se.values())
        assert abs(load - expected["se_max_over_mean"]) < 1e-12
        mod4 = [sum(live[i::4]) for i in range(4)]
        assert mod4 == routing["active_by_mod4"]
        samples = [s for s in batch["samples"] if s["candidate"] == name]
        assert len(samples) == data["rounds"] and all(s["routes"]["status"] == s["sum"]["status"] == "PASS" for s in samples)
        down, full = (mean(s[key] for s in samples) for key in ("down_us", "full_us"))
        assert abs(down - batch["means"][name]["down_us"]) < 1e-9
        assert abs(full - batch["means"][name]["full_us"]) < 1e-9
        print(f"| {name} | {config['sort_block_m']} | {mod4} | {sum(not v for v in live)} | {load:.6f} | {down:.3f} | {full:.3f} |")
        print("A_consumer_XCD_histogram", name, dict(Counter(len(v) for v in a_domains.values())))
    print("M128_EVIDENCE_RECHECK_PASS; old padding explanation does not establish the remaining native-sort speedup")


def mapping_grid(records, live, oc_splits, width):
    """Invert recorded block->task assignment to the actual XCD for each M/OC."""
    tasks = len(live) * oc_splits
    grid = [[-1] * oc_splits for _ in live]
    per_se, per_xcd = Counter(), Counter()
    for bid, (xcc, hw, task, active) in enumerate(records):
        assert 0 <= xcc < 8 and task == mapped_task(bid, tasks, width)
        assert active == int(task < tasks and live[task // oc_splits])
        per_se[(xcc, (hw >> 13) & 7)] += active
        per_xcd[xcc] += active
        if task < tasks:
            m, oc = divmod(task, oc_splits)
            assert grid[m][oc] == -1, "duplicate logical task"
            grid[m][oc] = xcc
    assert all(value >= 0 for row in grid for value in row), "missing logical task"
    return grid, per_se, per_xcd


def plot_mappings(args):
    """One figure, saved native-sort data only: maps, zoom and timing bars."""
    import torch
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap, to_rgb
    from matplotlib.patches import Patch

    source = args.data.resolve()
    data = json.loads(source.read_text())
    batch = next(b for b in data["results"] if b["batch"] == 16384)
    assert data["source_unchanged"] and batch["status"] == "PASS"
    raw_path = source.parent / "batch_16384_placement.pt"
    raw = torch.load(raw_path, map_location="cpu", weights_only=True)
    names = ("m128_identity", "m128_w4", "m128_w8")
    labels = ("identity", "width4", "width8")
    out = args.figure.resolve()
    assert out.suffix.lower() == ".png" and args.rows > 0 and args.start_m >= 0
    targets = (out, out.with_suffix(".svg"), out.with_suffix(".json"))
    assert not any(p.exists() for p in targets), "choose a new figure name; no overwrites"
    plots = []
    for name, label in zip(names, labels):
        config = batch["configs"][name]
        assert config["sort_block_m"] == config["block_m"] == config["packed_rows"] == 128
        assert config["num_waves"] == 4 and config["num_oc_splits"] == 8
        live = raw[name + "_live_m"].tolist()
        assert all(live), "this comparison expects native nonempty sorting blocks"
        width = config["xcd_count"] if config["xcd_swizzle"] else 0
        grid, per_se, per_xcd = mapping_grid(raw[name].tolist(), live, 8, width)
        placement = next(p for p in batch["placement"] if p["candidate"] == name)
        assert {f"xcd{x}.se{s}": value for (x, s), value in per_se.items()} == placement["se_active_counts"]
        assert {str(x): value for x, value in per_xcd.items()} == placement["xcd_active_counts"]
        samples = [s for s in batch["samples"] if s["candidate"] == name]
        assert len(samples) == data["rounds"]
        assert all(s["routes"]["status"] == s["sum"]["status"] == "PASS" for s in samples)
        means = {key: mean(s[key] for s in samples) for key in ("down_us", "full_us")}
        assert all(abs(means[key] - batch["means"][name][key]) < 1e-9 for key in means)
        plots.append({"candidate": name, "label": label, "width": width, "grid": grid,
                      **means, "samples": [{key: s[key] for key in ("round", "down_us", "full_us")} for s in samples],
                      "xcd_live_tasks": dict(per_xcd), "se_max_over_mean": placement["se_max_over_mean"],
                      "a_xcd_count_histogram": dict(Counter(len(set(row)) for row in grid))})
    common = [{k: v for k, v in batch["configs"][name].items() if k not in ("xcd_swizzle", "xcd_count")} for name in names]
    assert common[0] == common[1] == common[2], "comparison changes more than the mapping"
    m_count = len(plots[0]["grid"])
    assert all(len(p["grid"]) == m_count for p in plots)
    stop = min(m_count, args.start_m + args.rows)
    assert args.start_m < stop
    colors = ["#2563eb", "#f59e0b", "#16a34a", "#dc2626", "#7c3aed", "#0891b2", "#db2777", "#64748b"]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-.5, 8.5), cmap.N)
    with plt.rc_context({"font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11}):
        fig = plt.figure(figsize=(17, 10), facecolor="white")
        layout = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 1.0], height_ratios=[1, .86],
                                 left=.055, right=.97, top=.81, bottom=.13, hspace=.42, wspace=.45)
        for column, p in enumerate(plots):
            grid = np.asarray(p["grid"], dtype=np.int32)
            zoom = fig.add_subplot(layout[0, column])
            zoom.imshow(grid[args.start_m:stop], cmap=cmap, norm=norm, interpolation="nearest", aspect="auto")
            zoom.set_title(p["label"], weight="bold", pad=12)
            zoom.set_xticks(range(8), labels=range(8))
            zoom.set_yticks(range(stop - args.start_m), labels=range(args.start_m, stop))
            zoom.set_xlabel("OC slice")
            zoom.set_ylabel("M128 block (zoom)" if column == 0 else "")
            zoom.set_xticks(np.arange(-.5, 8), minor=True)
            zoom.set_yticks(np.arange(-.5, stop - args.start_m), minor=True)
            zoom.grid(which="minor", color="white", linewidth=1.0)
            zoom.tick_params(which="minor", length=0)
            for row in range(stop - args.start_m):
                for oc in range(8):
                    value = int(grid[row + args.start_m, oc])
                    r, g, b = to_rgb(colors[value])
                    text_color = "#111827" if .2126*r + .7152*g + .0722*b > .57 else "white"
                    zoom.text(oc, row, str(value), ha="center", va="center", color=text_color, fontsize=9, weight="bold")
            overview = fig.add_subplot(layout[1, column])
            overview.imshow(grid, cmap=cmap, norm=norm, interpolation="nearest", aspect="auto",
                            extent=(-.5, 7.5, m_count - .5, -.5))
            overview.set_title(f"All {m_count} M blocks", fontsize=11)
            overview.set_xticks(range(8), labels=range(8))
            overview.set_yticks([0, 144, 289, 434, 578, 723, 868, 1012, m_count - 1])
            overview.set_xlabel("OC slice")
            overview.set_ylabel("M128 block (full range)" if column == 0 else "")
        bar_colors = ["#94a3b8", "#0f766e", "#475569"]
        for row, (key, title) in enumerate((("down_us", "Down only"), ("full_us", "Down + inverse + reduce"))):
            ax = fig.add_subplot(layout[row, 3])
            values = [p[key] for p in plots]
            ax.barh(range(3), values, color=bar_colors, height=.46)
            for y, p in enumerate(plots):
                points = [s[key] for s in p["samples"]]
                ax.scatter(points, [y] * len(points), s=18, color="#111827", zorder=3)
                ax.text(p[key] + max(values) * .025, y, f"{p[key]:.3f}", va="center", fontsize=11, weight="bold")
            ax.set_yticks(range(3), labels=labels)
            ax.invert_yaxis()
            ax.set_xlim(0, max(values) * 1.27)
            ax.set_xlabel("Latency (microseconds; lower is better)")
            ax.set_title(title, fontsize=12, weight="bold", pad=12)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_axisbelow(True)
            ax.grid(axis="x", alpha=.2)
        fig.suptitle("M128 / sort128: identity vs width4 vs width8", y=.976, fontsize=20, weight="bold")
        fig.text(.5, .934, "tokens 16384 | TOPK 8 | experts 384 | N 6144 | K 256 | MI350X, 8 physical XCDs",
                 ha="center", color="#475569", fontsize=12)
        fig.legend(handles=[Patch(facecolor=color, label=f"XCD {i}") for i, color in enumerate(colors)],
                   loc="upper center", bbox_to_anchor=(.5, .91), ncol=8, frameon=False)
        fig.text(.5, .865, "Cell color / number = recorded physical XCD for (M, OC).  Zoom and full-range views share the same data.",
                 ha="center", fontsize=11, color="#334155")
        fig.text(.5, .068, "All three use all 8 XCDs and have SE task max/mean = 1.002593.  The maps are NOT execution timelines.",
                 ha="center", color="#334155", fontsize=11)
        fig.text(.5, .037, "Bars: same-run two-round means; dots: individual rounds.  Timings exclude sorting; placement logging was separate.  No new GPU run.",
                 ha="center", color="#64748b", fontsize=10)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=180)
        fig.savefig(out.with_suffix(".svg"))
        plt.close(fig)
    write_json(out.with_suffix(".json"), {"source_results": str(source), "source_results_sha256": digest(source.read_bytes()),
        "source_placement_sha256": digest(raw_path.read_bytes()), "plot_script_sha256": digest(Path(__file__).read_bytes()),
        "shape": data["shape"], "batch": 16384, "zoom": [args.start_m, stop], "plots": plots,
        "note": "Physical XCD readback indexed by logical M/OC; not time order. Latencies recomputed from saved ordinary samples."})
    print("MAPPING_COMPARISON_FIGURE_PASS", out, "3 mappings", m_count, "M blocks; timings", [p["down_us"] for p in plots])


class DispatchAnalysisTests(unittest.TestCase):
    def witness(self, successor, *, predecessor=None, end=800):
        previous = predecessor or Block(1, 0, 1, 0, 1, 2, 22, 25, 20)
        return refill_witnesses([previous, successor], prefix=4,
            target=(0, 0), begin=10, end=end, guard=5, short_ticks=20)

    def test_refill_while_long_peer_runs(self):
        self.assertEqual(len(self.witness(Block(5, 0, 1, 0, 100, 101, 121, 125, 20))), 1)

    def test_completion_round_has_no_refill_witness(self):
        self.assertFalse(self.witness(Block(5, 0, 1, 0, 810, 811, 831, 835, 20)))

    def test_initial_cohort_is_not_new_round(self):
        self.assertFalse(self.witness(Block(3, 0, 1, 0, 100, 101, 121, 125, 20)))

    def test_other_cu_does_not_prove_reuse(self):
        self.assertFalse(self.witness(Block(5, 0, 1, 1, 100, 101, 121, 125, 20)))

    def test_other_xcd_cannot_refute_within_xcd_rule(self):
        self.assertFalse(self.witness(Block(5, 1, 1, 0, 100, 101, 121, 125, 20)))

    def test_timestamp_guard_excludes_ambiguous_edges(self):
        self.assertFalse(self.witness(Block(5, 0, 1, 0, 28, 29, 49, 51, 20)))
        self.assertFalse(self.witness(Block(5, 0, 1, 0, 798, 799, 819, 821, 20)))

    def test_idle_gap_merges_overlapping_spans(self):
        self.assertEqual(longest_idle_gap([(0, 30), (20, 40), (80, 100)], 10, 90), (40, 80))

    def test_full_coverage_has_no_gap(self):
        self.assertEqual(longest_idle_gap([(0, 100)], 10, 90), (10, 10))

    def test_partial_release_waits_for_one_not_every_holder(self):
        later = Block(5, 0, 1, 0, 250, 251, 271, 275, 20)
        predecessor = Block(1, 0, 1, 0, 0, 1, 21, 25, 20)
        rows = refill_witnesses([predecessor, later], prefix=4,
            target=(0, 0), begin=220, end=800, guard=5, short_ticks=20)
        self.assertEqual(len(rows), 1)

    def test_decode_validates_every_wave(self):
        raw = [[[0, (1 << 13) | (3 << 8) | (w << 4), 10, 20, 120, 125,
                 0, w, 100, (w * 64) ^ 0xA5] for w in range(4)]]
        blocks = decode_blocks(raw, 1)
        self.assertEqual(blocks[0].location, (0, 1, 3))
        self.assertEqual(blocks[0].work_end, 120)
        raw[0][3][9] = 0
        with self.assertRaises(AssertionError):
            decode_blocks(raw, 1)

    def test_ordered_model_and_completion_barrier_differ(self):
        durations = [10, 100, 5, 5, 20, 10, 10, 10]
        ordered = admission_model(durations)
        barrier = admission_model(durations, completion_round=True)
        self.assertEqual([r["start_us"] for r in ordered], [0, 0, 0, 0, 10, 100, 100, 100])
        self.assertEqual(barrier[4]["start_us"], 100)
        self.assertEqual(ordered[4]["end_us"], 30)

    def test_acceptance_does_not_require_completely_idle_cu(self):
        rows = admission_model([800, 20, 20, 20, 20, 20, 20, 20], slots_per_se=2)
        self.assertEqual(rows[4]["start_us"], 0)
        self.assertEqual(rows[0]["end_us"], 800)

    def test_mapping_is_bijective_with_remainder(self):
        for tasks, width in ((7, 8), (12288, 4), (9256, 8)):
            result = [mapped_task(b, tasks, width) for b in range(tasks + 9)]
            self.assertEqual(sorted(result[:tasks]), list(range(tasks)))
            self.assertEqual(result[tasks:], list(range(tasks, tasks + 9)))

    def test_known_legacy_hip_matches_embedded_source(self):
        self.assertEqual(digest(HIP_SOURCE.encode()), LEGACY_SOURCE_SHA256["se_dispatch_probe.cpp"])
        self.assertIn("legacy", check_provenance(LEGACY_SOURCE_SHA256))

    def test_unknown_provenance_is_not_silently_accepted(self):
        with self.assertRaises(ValueError):
            check_provenance({**LEGACY_SOURCE_SHA256, "probe-se-dispatch.py": "changed"})

    def test_pre_plot_singlefile_capture_is_recognized(self):
        captured = {"script": PRE_PLOT_SCRIPT_SHA256, "embedded_hip": digest(HIP_SOURCE.encode())}
        self.assertIn("pre-plot", check_provenance(captured))

    def test_grid_uses_recorded_xcd_not_predicted_color(self):
        tasks = 3 * 8
        for width in (0, 4, 8):
            records = [[(bid + 3) % 8, 0, mapped_task(bid, tasks, width), 1] for bid in range(tasks)]
            grid, _, _ = mapping_grid(records, [True] * 3, 8, width)
            for xcc, _, task, _ in records:
                self.assertEqual(grid[task // 8][task % 8], xcc)

    def test_grid_handles_mapping_boundary_and_capacity_tail(self):
        tasks, width = 5 * 8, 8
        records = [[bid % 8, 0, mapped_task(bid, tasks, width), int(bid < tasks)] for bid in range(tasks + 8)]
        grid, _, per_xcd = mapping_grid(records, [True] * 5, 8, width)
        self.assertEqual(len(grid), 5)
        self.assertEqual(sum(per_xcd.values()), tasks)
        self.assertGreater(len(set(grid[0])), 1)  # chunk5 cuts through an M row


def self_test(_args):
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(DispatchAnalysisTests))
    if not result.wasSuccessful():
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("model", help="CPU illustrative schedule, not a hardware simulation").set_defaults(handler=model_examples)
    sub.add_parser("test", help="CPU-only built-in unit tests").set_defaults(handler=self_test)
    evidence = sub.add_parser("moe-evidence", help="audit existing M128 GEMM timings and raw placement")
    evidence.add_argument("--data", type=Path, default=DEFAULT_MOE_DATA)
    evidence.set_defaults(handler=moe_evidence)
    plotting = sub.add_parser("plot-mappings", help="one figure with actual identity/width4/width8 XCD grids and timing bars")
    plotting.add_argument("--data", type=Path, default=DEFAULT_MOE_DATA)
    plotting.add_argument("--figure", type=Path, required=True, help="new PNG path; also writes SVG and verified JSON")
    plotting.add_argument("--start-m", type=int, default=0)
    plotting.add_argument("--rows", type=int, default=12)
    plotting.set_defaults(handler=plot_mappings)
    capture = sub.add_parser("run", help="compile embedded HIP and run all five bounded examples")
    capture.add_argument("--out", type=Path, required=True)
    capture.add_argument("--lds", type=int, nargs="+", default=[98304, 68616])
    capture.add_argument("--grid", type=int, default=2048)
    capture.add_argument("--repeats", type=int, default=3)
    capture.add_argument("--short-us", type=float, default=20)
    capture.add_argument("--release-us", type=float, default=200)
    capture.add_argument("--long-us", type=float, default=800)
    capture.add_argument("--guard-us", type=float, default=5)
    capture.add_argument("--target-xcc", type=int, default=0)
    capture.add_argument("--target-se", type=int, default=0)
    capture.set_defaults(handler=run)
    check = sub.add_parser("analyze", help="CPU-only raw recheck of new or recognized historical records")
    check.add_argument("--out", type=Path, required=True)
    check.set_defaults(handler=reanalyze)
    args = parser.parse_args()
    if hasattr(args, "out"):
        args.out = args.out.resolve()
    args.handler(args)


if __name__ == "__main__":
    main()