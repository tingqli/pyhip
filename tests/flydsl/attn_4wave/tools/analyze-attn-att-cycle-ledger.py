#!/usr/bin/env python3
"""Build a non-double-counted ATT cycle ledger for the PyHIP attention kernel.

The rocprofv3 ``code.json`` stall column sums stalls independently for every
wave. Two resident waves can wait during the same physical SIMD cycle, so that
sum cannot be compared directly with wall time. This analyzer instead:

1. merges all resident-wave issue intervals on each physical SIMD;
2. identifies physical intervals where no instruction issued;
3. inspects every active wave's blocked PC during each interval;
4. splits the interval equally across the simultaneous blockers;
5. attributes the trace's excess over the ideal model in proportion to the
    observed internal blocker weights.

The equal split and proportional normalization are attribution policies, not
claims that hardware assigns ownership that way. The closure totals remain
independent of those policies.
"""

import argparse
import bisect
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

WAVE_FILE_RE = re.compile(r"se(\d+)_sm(\d+)_sl(\d+)_wv(\d+)\.json")
PYTHON_LINE_RE = re.compile(r"test_attn_gemm_jit\.py:(\d+)")
ISSUE_CYCLES = 4
MFMA_BUSY_CYCLES = 16


def classify_instruction(asm):
    opcode = asm.lower()
    if "s_barrier" in opcode:
        return "barrier"
    if opcode.startswith("s_waitcnt") and "vmcnt" in opcode:
        return "VMEM-wait"
    if opcode.startswith("s_waitcnt") and "lgkmcnt" in opcode:
        return "LDS/SMEM-wait"
    if opcode.startswith("buffer_load"):
        return "VMEM-load"
    if opcode.startswith("buffer_store"):
        return "VMEM-store"
    if opcode.startswith("ds_"):
        return "LDS/crosslane"
    if "v_mfma" in opcode:
        return "MFMA"
    if opcode.startswith(("v_exp", "v_rcp")):
        return "TRANS"
    if opcode.startswith("v_"):
        return "VALU"
    if opcode.startswith("s_"):
        return "SALU/control"
    return "other"


def stage_for_line(python_line, category):
    if python_line in (113, 121):
        return "V global load"
    if python_line in (130, 132):
        return "future-K global prefetch"
    if python_line in (140, 141, 145, 147):
        return "K LDS write"
    if python_line == 153:
        return "K LDS read"
    if python_line in (169, 174):
        return "GEMM1/progressive K wait"
    if python_line in (202, 209, 216, 223, 225, 228, 229, 231, 245, 247, 249, 251, 253):
        return "softmax max reduction"
    if python_line in (258, 260, 269, 278, 281, 285):
        return "softmax center/EXP"
    if python_line in (310, 321, 322, 327, 328, 330, 332, 334, 339, 341, 343, 345, 347):
        return "softmax sum reduction/state"
    if python_line in (289, 294, 300, 356, 362, 369):
        return "probability f32->bf16"
    if python_line in (378, 380, 385):
        return "conditional O rescale"
    if python_line == 448:
        return "GEMM2"
    if python_line in (459, 462, 463, 496, 497):
        return "K stage write/wait/barrier"
    if python_line in (455, 508, 518):
        return "loop/address control"
    return category


def load_code(dispatch_dir):
    with (dispatch_dir / "code.json").open() as stream:
        data = json.load(stream)
    if not data.get("code"):
        raise RuntimeError(
            f"ATT trace contains no decoded instructions: {dispatch_dir}"
        )
    return {row[2]: row for row in data["code"]}


def resolve_python_line(source_loc, generated_cpp_lines):
    if not generated_cpp_lines or not source_loc or ":" not in source_loc:
        return None
    try:
        generated_line = int(source_loc.rsplit(":", 1)[1])
    except ValueError:
        return None
    # DWARF points at the generated .loc line; the following quoted assembly
    # line carries PyHIP's original Python source comment.
    start = max(0, generated_line - 1)
    for index in range(start, min(start + 4, len(generated_cpp_lines))):
        match = PYTHON_LINE_RE.search(generated_cpp_lines[index])
        if match:
            return int(match.group(1))
    return None


def load_simd_waves(dispatch_dir, code_by_index):
    groups = defaultdict(list)
    for path in sorted(dispatch_dir.glob("se*_sm*_sl*_wv*.json")):
        match = WAVE_FILE_RE.fullmatch(path.name)
        if not match:
            continue
        with path.open() as stream:
            wave = json.load(stream)["wave"]
        records = []
        for raw in wave["instructions"]:
            row = code_by_index[raw[4]]
            records.append(
                {
                    "issue": raw[0],
                    "duration": raw[3],
                    "category": classify_instruction(row[0]),
                    "pc_index": raw[4],
                    "asm": row[0],
                }
            )
        key = (int(match.group(1)), int(match.group(2)), int(wave["cu"]))
        groups[key].append(
            {
                "begin": wave["begin"],
                "end": wave["end"],
                "records": records,
                "issue_times": [record["issue"] for record in records],
            }
        )
    if not groups:
        raise RuntimeError(f"No per-wave ATT files found in {dispatch_dir}")
    return groups


def merge_intervals(intervals):
    merged = []
    for begin, end in sorted(intervals):
        if not merged or begin > merged[-1][1]:
            merged.append([begin, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return merged


def intersect_intervals(left, right):
    intersections = []
    left_index = 0
    right_index = 0
    while left_index < len(left) and right_index < len(right):
        begin = max(left[left_index][0], right[right_index][0])
        end = min(left[left_index][1], right[right_index][1])
        if begin < end:
            intersections.append([begin, end])
        if left[left_index][1] < right[right_index][1]:
            left_index += 1
        else:
            right_index += 1
    return intersections


def sum_interval_cycles(intervals):
    return sum(end - begin for begin, end in intervals)


def split_by_intervals(begin, end, covered_intervals, covered_starts):
    """Split [begin, end) into covered and uncovered physical intervals."""
    index = bisect.bisect_right(covered_starts, begin) - 1
    if index < 0 or covered_intervals[index][1] <= begin:
        index += 1
    cursor = begin
    while cursor < end:
        if index >= len(covered_intervals) or covered_intervals[index][0] >= end:
            yield cursor, end, False
            return
        covered_begin, covered_end = covered_intervals[index]
        if cursor < covered_begin:
            segment_end = min(end, covered_begin)
            yield cursor, segment_end, False
            cursor = segment_end
            continue
        segment_end = min(end, covered_end)
        yield cursor, segment_end, True
        cursor = segment_end
        if cursor >= covered_end:
            index += 1


def timestamp_in_intervals(timestamp, intervals, starts):
    index = bisect.bisect_right(starts, timestamp) - 1
    return index >= 0 and intervals[index][0] <= timestamp < intervals[index][1]


def blockers_at(waves, timestamp):
    blockers = []
    for wave in waves:
        if not (wave["begin"] <= timestamp < wave["end"]):
            continue
        index = bisect.bisect_right(wave["issue_times"], timestamp) - 1
        if index < 0:
            continue
        record = wave["records"][index]
        if record["duration"] > 4 and record["issue"] + record["duration"] > timestamp:
            blockers.append((record["pc_index"], record["category"]))
        else:
            blockers.append((-1, "scheduler/ready"))
    return blockers or [(-2, "no-active-wave")]


def analyze_simd(waves):
    events = sorted(
        (record["issue"], wave_index, record)
        for wave_index, wave in enumerate(waves)
        for record in wave["records"]
    )
    issue_intervals = merge_intervals(
        (record["issue"], record["issue"] + ISSUE_CYCLES) for _, _, record in events
    )
    mfma_issue_intervals = merge_intervals(
        (record["issue"], record["issue"] + ISSUE_CYCLES)
        for _, _, record in events
        if record["category"] == "MFMA"
    )
    non_mfma_issue_intervals = merge_intervals(
        (record["issue"], record["issue"] + ISSUE_CYCLES)
        for _, _, record in events
        if record["category"] != "MFMA"
    )
    mfma_shadow_intervals = merge_intervals(
        (record["issue"] + ISSUE_CYCLES, record["issue"] + MFMA_BUSY_CYCLES)
        for _, _, record in events
        if record["category"] == "MFMA"
    )
    mfma_count = sum(record["category"] == "MFMA" for _, _, record in events)
    begin = min(wave["begin"] for wave in waves)
    end = max(wave["end"] for wave in waves)
    interval_cycles = end - begin
    issue_cycles = sum_interval_cycles(issue_intervals)

    shadow_non_mfma = intersect_intervals(
        mfma_shadow_intervals, non_mfma_issue_intervals
    )
    shadow_mfma = intersect_intervals(mfma_shadow_intervals, mfma_issue_intervals)
    shadow_mfma_non_mfma = intersect_intervals(shadow_mfma, non_mfma_issue_intervals)
    mfma_non_mfma = intersect_intervals(mfma_issue_intervals, non_mfma_issue_intervals)
    shadow_cycles = sum_interval_cycles(mfma_shadow_intervals)
    shadow_non_mfma_cycles = sum_interval_cycles(shadow_non_mfma)
    shadow_mfma_only_cycles = sum_interval_cycles(shadow_mfma) - sum_interval_cycles(
        shadow_mfma_non_mfma
    )
    shadow_no_issue_cycles = (
        shadow_cycles - shadow_non_mfma_cycles - shadow_mfma_only_cycles
    )
    total_non_mfma_cycles = sum_interval_cycles(non_mfma_issue_intervals)
    total_mfma_only_cycles = sum_interval_cycles(
        mfma_issue_intervals
    ) - sum_interval_cycles(mfma_non_mfma)
    total_no_issue_cycles = interval_cycles - issue_cycles
    timeline_matrix = Counter(
        {
            ("mfma_shadow", "non_mfma_issue"): shadow_non_mfma_cycles,
            ("mfma_shadow", "mfma_only_issue"): shadow_mfma_only_cycles,
            ("mfma_shadow", "no_issue"): shadow_no_issue_cycles,
            ("outside_shadow", "non_mfma_issue"): total_non_mfma_cycles
            - shadow_non_mfma_cycles,
            ("outside_shadow", "mfma_only_issue"): total_mfma_only_cycles
            - shadow_mfma_only_cycles,
            ("outside_shadow", "no_issue"): total_no_issue_cycles
            - shadow_no_issue_cycles,
        }
    )
    if sum(timeline_matrix.values()) != interval_cycles:
        raise RuntimeError(
            "MFMA-shadow timeline matrix does not close to the SIMD interval"
        )

    pc_cycles = Counter()
    category_cycles = Counter()
    combination_cycles = Counter()
    no_issue_pc_cycles = defaultdict(lambda: defaultdict(float))
    no_issue_category_cycles = defaultdict(lambda: defaultdict(float))
    no_issue_combination_cycles = defaultdict(Counter)
    shadow_starts = [interval[0] for interval in mfma_shadow_intervals]
    non_mfma_events_by_time = defaultdict(list)
    for _, _, record in events:
        if record["category"] != "MFMA":
            non_mfma_events_by_time[record["issue"]].append(record)
    non_mfma_issue_pc_cycles = defaultdict(lambda: defaultdict(float))
    non_mfma_issue_category_cycles = defaultdict(lambda: defaultdict(float))
    for issue_time, records in non_mfma_events_by_time.items():
        region = (
            "mfma_shadow"
            if timestamp_in_intervals(issue_time, mfma_shadow_intervals, shadow_starts)
            else "outside_shadow"
        )
        share = ISSUE_CYCLES / len(records)
        for record in records:
            non_mfma_issue_pc_cycles[region][record["pc_index"]] += share
            non_mfma_issue_category_cycles[region][record["category"]] += share

    internal_idle_cycles = 0
    for (_, current_end), (next_begin, _) in zip(issue_intervals, issue_intervals[1:]):
        if next_begin <= current_end:
            continue
        gap_duration = next_begin - current_end
        gap_midpoint = (current_end + next_begin - 1) // 2
        gap_blockers = blockers_at(waves, gap_midpoint)
        gap_combination = " + ".join(sorted(category for _, category in gap_blockers))
        internal_idle_cycles += gap_duration
        combination_cycles[gap_combination] += gap_duration
        gap_share = gap_duration / len(gap_blockers)
        for pc_index, category in gap_blockers:
            pc_cycles[pc_index] += gap_share
            category_cycles[category] += gap_share

        for segment_begin, segment_end, in_shadow in split_by_intervals(
            current_end, next_begin, mfma_shadow_intervals, shadow_starts
        ):
            duration = segment_end - segment_begin
            midpoint = (segment_begin + segment_end - 1) // 2
            blockers = blockers_at(waves, midpoint)
            region = "mfma_shadow" if in_shadow else "outside_shadow"
            combination = " + ".join(sorted(category for _, category in blockers))
            no_issue_combination_cycles[region][combination] += duration
            share = duration / len(blockers)
            for pc_index, category in blockers:
                no_issue_pc_cycles[region][pc_index] += share
                no_issue_category_cycles[region][category] += share

    edge_cycles = interval_cycles - issue_cycles - internal_idle_cycles
    if edge_cycles < 0:
        raise RuntimeError("Issue-union accounting exceeded the SIMD interval")
    no_issue_pc_cycles["outside_shadow"][-3] += edge_cycles
    no_issue_category_cycles["outside_shadow"]["trace-edge"] += edge_cycles
    no_issue_combination_cycles["outside_shadow"]["trace-edge"] += edge_cycles
    return {
        "interval_cycles": interval_cycles,
        "issue_cycles": issue_cycles,
        "physical_idle_cycles": internal_idle_cycles + edge_cycles,
        "internal_idle_cycles": internal_idle_cycles,
        "edge_cycles": edge_cycles,
        "pc_cycles": pc_cycles,
        "category_cycles": category_cycles,
        "combination_cycles": combination_cycles,
        "timeline_matrix": timeline_matrix,
        "mfma_count": mfma_count,
        "logical_mfma_shadow_cycles": mfma_count * (MFMA_BUSY_CYCLES - ISSUE_CYCLES),
        "physical_mfma_shadow_cycles": shadow_cycles,
        "no_issue_pc_cycles": no_issue_pc_cycles,
        "no_issue_category_cycles": no_issue_category_cycles,
        "no_issue_combination_cycles": no_issue_combination_cycles,
        "non_mfma_issue_pc_cycles": non_mfma_issue_pc_cycles,
        "non_mfma_issue_category_cycles": non_mfma_issue_category_cycles,
    }


def average_counter(counter, simd_count, tasks_per_simd):
    return {key: value / simd_count / tasks_per_simd for key, value in counter.items()}


def format_cycles(value):
    return f"{value:,.3f}"


def instruction_metadata(pc_index, code_by_index, generated_cpp_lines):
    if pc_index == -1:
        return (
            None,
            "scheduler/ready",
            "scheduler/ready",
            None,
            "no blocked instruction",
        )
    if pc_index == -2:
        return None, "no-active-wave", "no-active-wave", None, "no active wave"
    if pc_index == -3:
        return None, "trace-edge", "trace-edge", None, "trace interval edge"
    row = code_by_index[pc_index]
    category = classify_instruction(row[0])
    python_line = resolve_python_line(row[3], generated_cpp_lines)
    return row[5], category, stage_for_line(python_line, category), python_line, row[0]


def make_pc_rows(
    pc_task, code_by_index, generated_cpp_lines, task_scale, tiles_per_task
):
    rows = []
    for pc_index, raw_task_cycles in pc_task.items():
        pc_addr, category, stage, python_line, asm = instruction_metadata(
            pc_index, code_by_index, generated_cpp_lines
        )
        scaled_task_cycles = raw_task_cycles * task_scale
        rows.append(
            {
                "pc_index": pc_index,
                "pc_addr": pc_addr,
                "category": category,
                "stage": stage,
                "python_line": python_line,
                "asm": asm,
                "physical_cycles_per_task": raw_task_cycles,
                "scaled_cycles_per_task": scaled_task_cycles,
                "scaled_cycles_per_tile": scaled_task_cycles / tiles_per_task,
                "physical_idle_cycles_per_task": raw_task_cycles,
                "residual_cycles_per_task": scaled_task_cycles,
                "residual_cycles_per_tile": scaled_task_cycles / tiles_per_task,
            }
        )
    rows.sort(key=lambda row: row["scaled_cycles_per_task"], reverse=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dispatch_dir", type=Path)
    parser.add_argument("--ideal-task-cycles", type=float, required=True)
    parser.add_argument("--wall-task-cycles", type=float, required=True)
    parser.add_argument("--tiles-per-task", type=int, required=True)
    parser.add_argument("--tasks-per-simd", type=int, default=0)
    parser.add_argument("--generated-cpp", type=Path)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    code_by_index = load_code(args.dispatch_dir)
    groups = load_simd_waves(args.dispatch_dir, code_by_index)
    wave_counts = [len(waves) for waves in groups.values()]
    tasks_per_simd = args.tasks_per_simd or int(statistics.median(wave_counts))
    if any(count != tasks_per_simd for count in wave_counts):
        raise RuntimeError(f"Inconsistent wave count per SIMD: {wave_counts}")
    if args.tiles_per_task <= 0 or tasks_per_simd <= 0:
        parser.error("tile and task counts must be positive")

    generated_cpp_lines = None
    if args.generated_cpp:
        generated_cpp_lines = args.generated_cpp.read_text().splitlines()

    analyses = [analyze_simd(waves) for waves in groups.values()]
    simd_count = len(analyses)
    trace_task_cycles = (
        statistics.mean(item["interval_cycles"] for item in analyses) / tasks_per_simd
    )
    issue_task_cycles = (
        statistics.mean(item["issue_cycles"] for item in analyses) / tasks_per_simd
    )
    physical_idle_task_cycles = (
        statistics.mean(item["physical_idle_cycles"] for item in analyses)
        / tasks_per_simd
    )
    attributable_idle_task_cycles = (
        statistics.mean(item["internal_idle_cycles"] for item in analyses)
        / tasks_per_simd
    )
    trace_edge_task_cycles = (
        statistics.mean(item["edge_cycles"] for item in analyses) / tasks_per_simd
    )
    model_idle_task_cycles = args.ideal_task_cycles - issue_task_cycles
    att_excess_task_cycles = trace_task_cycles - args.ideal_task_cycles
    wall_boundary_task_cycles = args.wall_task_cycles - trace_task_cycles
    wall_residual_task_cycles = args.wall_task_cycles - args.ideal_task_cycles
    if physical_idle_task_cycles <= 0 or att_excess_task_cycles < 0:
        raise RuntimeError(
            "The supplied ideal model is incompatible with the ATT trace"
        )
    attribution_scale = att_excess_task_cycles / attributable_idle_task_cycles

    pc_raw = Counter()
    category_raw = Counter()
    combination_raw = Counter()
    timeline_matrix_raw = Counter()
    no_issue_pc_raw = defaultdict(Counter)
    no_issue_category_raw = defaultdict(Counter)
    no_issue_combination_raw = defaultdict(Counter)
    non_mfma_issue_pc_raw = defaultdict(Counter)
    non_mfma_issue_category_raw = defaultdict(Counter)
    total_mfma_count = 0
    logical_mfma_shadow_cycles = 0
    physical_mfma_shadow_cycles = 0
    for item in analyses:
        pc_raw.update(item["pc_cycles"])
        category_raw.update(item["category_cycles"])
        combination_raw.update(item["combination_cycles"])
        timeline_matrix_raw.update(item["timeline_matrix"])
        for region in ("mfma_shadow", "outside_shadow"):
            no_issue_pc_raw[region].update(item["no_issue_pc_cycles"][region])
            no_issue_category_raw[region].update(
                item["no_issue_category_cycles"][region]
            )
            no_issue_combination_raw[region].update(
                item["no_issue_combination_cycles"][region]
            )
            non_mfma_issue_pc_raw[region].update(
                item["non_mfma_issue_pc_cycles"][region]
            )
            non_mfma_issue_category_raw[region].update(
                item["non_mfma_issue_category_cycles"][region]
            )
        total_mfma_count += item["mfma_count"]
        logical_mfma_shadow_cycles += item["logical_mfma_shadow_cycles"]
        physical_mfma_shadow_cycles += item["physical_mfma_shadow_cycles"]
    pc_task = average_counter(pc_raw, simd_count, tasks_per_simd)
    category_task = average_counter(category_raw, simd_count, tasks_per_simd)
    combination_task = average_counter(combination_raw, simd_count, tasks_per_simd)
    timeline_matrix_task = average_counter(
        timeline_matrix_raw, simd_count, tasks_per_simd
    )

    category_rows = []
    for category, raw_task_cycles in sorted(
        category_task.items(), key=lambda item: item[1], reverse=True
    ):
        category_rows.append(
            {
                "category": category,
                "physical_idle_cycles_per_task": raw_task_cycles,
                "residual_cycles_per_task": raw_task_cycles * attribution_scale,
                "residual_cycles_per_tile": raw_task_cycles
                * attribution_scale
                / args.tiles_per_task,
            }
        )

    pc_rows = make_pc_rows(
        pc_task,
        code_by_index,
        generated_cpp_lines,
        attribution_scale,
        args.tiles_per_task,
    )
    stage_task = Counter()
    for row in pc_rows:
        stage_task[row["stage"]] += row["scaled_cycles_per_task"]

    timeline_rows = []
    for region in ("mfma_shadow", "outside_shadow"):
        for state in ("non_mfma_issue", "mfma_only_issue", "no_issue"):
            task_cycles = timeline_matrix_task[(region, state)]
            timeline_rows.append(
                {
                    "region": region,
                    "state": state,
                    "cycles_per_task": task_cycles,
                    "cycles_per_tile": task_cycles / args.tiles_per_task,
                }
            )

    mfma_count_per_task = total_mfma_count / simd_count / tasks_per_simd
    logical_mfma_shadow_task_cycles = (
        logical_mfma_shadow_cycles / simd_count / tasks_per_simd
    )
    physical_mfma_shadow_task_cycles = (
        physical_mfma_shadow_cycles / simd_count / tasks_per_simd
    )
    shadow_non_mfma_task_cycles = timeline_matrix_task[
        ("mfma_shadow", "non_mfma_issue")
    ]
    shadow_mfma_only_task_cycles = timeline_matrix_task[
        ("mfma_shadow", "mfma_only_issue")
    ]
    shadow_no_issue_task_cycles = timeline_matrix_task[("mfma_shadow", "no_issue")]
    mfma_shadow_summary = {
        "mfma_count_per_task": mfma_count_per_task,
        "mfma_count_per_tile": mfma_count_per_task / args.tiles_per_task,
        "logical_shadow_cycles_per_task": logical_mfma_shadow_task_cycles,
        "logical_shadow_cycles_per_tile": logical_mfma_shadow_task_cycles
        / args.tiles_per_task,
        "physical_shadow_union_cycles_per_task": physical_mfma_shadow_task_cycles,
        "physical_shadow_union_cycles_per_tile": physical_mfma_shadow_task_cycles
        / args.tiles_per_task,
        "overlapping_shadow_cycles_per_task": logical_mfma_shadow_task_cycles
        - physical_mfma_shadow_task_cycles,
        "overlapping_shadow_cycles_per_tile": (
            logical_mfma_shadow_task_cycles - physical_mfma_shadow_task_cycles
        )
        / args.tiles_per_task,
        "non_mfma_issue_cycles_per_task": shadow_non_mfma_task_cycles,
        "non_mfma_issue_cycles_per_tile": shadow_non_mfma_task_cycles
        / args.tiles_per_task,
        "mfma_only_issue_cycles_per_task": shadow_mfma_only_task_cycles,
        "mfma_only_issue_cycles_per_tile": shadow_mfma_only_task_cycles
        / args.tiles_per_task,
        "no_issue_cycles_per_task": shadow_no_issue_task_cycles,
        "no_issue_cycles_per_tile": shadow_no_issue_task_cycles / args.tiles_per_task,
        "logical_unhidden_cycles_per_task": logical_mfma_shadow_task_cycles
        - shadow_non_mfma_task_cycles,
        "logical_unhidden_cycles_per_tile": (
            logical_mfma_shadow_task_cycles - shadow_non_mfma_task_cycles
        )
        / args.tiles_per_task,
    }

    no_issue_regions = []
    non_mfma_issue_regions = []
    for region in ("mfma_shadow", "outside_shadow"):
        region_category_task = average_counter(
            no_issue_category_raw[region], simd_count, tasks_per_simd
        )
        attributable_region_category_task = {
            category: cycles
            for category, cycles in region_category_task.items()
            if category != "trace-edge"
        }
        region_pc_task = average_counter(
            no_issue_pc_raw[region], simd_count, tasks_per_simd
        )
        region_pc_task.pop(-3, None)
        region_pc_rows = make_pc_rows(
            region_pc_task,
            code_by_index,
            generated_cpp_lines,
            attribution_scale,
            args.tiles_per_task,
        )
        region_stage_task = Counter()
        for row in region_pc_rows:
            region_stage_task[row["stage"]] += row["physical_cycles_per_task"]
        no_issue_regions.append(
            {
                "region": region,
                "physical_cycles_per_task": sum(region_category_task.values()),
                "physical_cycles_per_tile": sum(region_category_task.values())
                / args.tiles_per_task,
                "attributed_residual_cycles_per_task": sum(
                    attributable_region_category_task.values()
                )
                * attribution_scale,
                "attributed_residual_cycles_per_tile": sum(
                    attributable_region_category_task.values()
                )
                * attribution_scale
                / args.tiles_per_task,
                "categories": [
                    {
                        "category": category,
                        "physical_cycles_per_task": cycles,
                        "physical_cycles_per_tile": cycles / args.tiles_per_task,
                        "attributed_residual_cycles_per_tile": (
                            cycles * attribution_scale / args.tiles_per_task
                            if category != "trace-edge"
                            else 0.0
                        ),
                    }
                    for category, cycles in sorted(
                        region_category_task.items(),
                        key=lambda item: item[1],
                        reverse=True,
                    )
                ],
                "stages": [
                    {
                        "stage": stage,
                        "physical_cycles_per_task": cycles,
                        "physical_cycles_per_tile": cycles / args.tiles_per_task,
                        "attributed_residual_cycles_per_tile": cycles
                        * attribution_scale
                        / args.tiles_per_task,
                    }
                    for stage, cycles in sorted(
                        region_stage_task.items(),
                        key=lambda item: item[1],
                        reverse=True,
                    )
                ],
                "top_pcs": region_pc_rows[: args.topk],
                "top_combinations": [
                    {
                        "combination": combination,
                        "physical_cycles_per_task": cycles,
                        "physical_cycles_per_tile": cycles / args.tiles_per_task,
                    }
                    for combination, cycles in sorted(
                        average_counter(
                            no_issue_combination_raw[region], simd_count, tasks_per_simd
                        ).items(),
                        key=lambda item: item[1],
                        reverse=True,
                    )[: args.topk]
                ],
            }
        )

        issue_category_task = average_counter(
            non_mfma_issue_category_raw[region], simd_count, tasks_per_simd
        )
        issue_pc_task = average_counter(
            non_mfma_issue_pc_raw[region], simd_count, tasks_per_simd
        )
        issue_pc_rows = make_pc_rows(
            issue_pc_task, code_by_index, generated_cpp_lines, 1.0, args.tiles_per_task
        )
        issue_stage_task = Counter()
        for row in issue_pc_rows:
            issue_stage_task[row["stage"]] += row["physical_cycles_per_task"]
        non_mfma_issue_regions.append(
            {
                "region": region,
                "physical_cycles_per_task": sum(issue_category_task.values()),
                "physical_cycles_per_tile": sum(issue_category_task.values())
                / args.tiles_per_task,
                "categories": [
                    {
                        "category": category,
                        "cycles_per_task": cycles,
                        "cycles_per_tile": cycles / args.tiles_per_task,
                    }
                    for category, cycles in sorted(
                        issue_category_task.items(),
                        key=lambda item: item[1],
                        reverse=True,
                    )
                ],
                "stages": [
                    {
                        "stage": stage,
                        "cycles_per_task": cycles,
                        "cycles_per_tile": cycles / args.tiles_per_task,
                    }
                    for stage, cycles in sorted(
                        issue_stage_task.items(), key=lambda item: item[1], reverse=True
                    )
                ],
                "top_pcs": issue_pc_rows[: args.topk],
            }
        )

    stage_rows = [
        {
            "stage": stage,
            "residual_cycles_per_task": cycles,
            "residual_cycles_per_tile": cycles / args.tiles_per_task,
        }
        for stage, cycles in sorted(
            stage_task.items(), key=lambda item: item[1], reverse=True
        )
    ]

    closure = {
        "ideal_task_cycles": args.ideal_task_cycles,
        "wall_task_cycles": args.wall_task_cycles,
        "trace_task_cycles": trace_task_cycles,
        "issue_union_cycles_per_task": issue_task_cycles,
        "physical_no_issue_cycles_per_task": physical_idle_task_cycles,
        "attributable_no_issue_cycles_per_task": attributable_idle_task_cycles,
        "trace_edge_cycles_per_task": trace_edge_task_cycles,
        "model_budgeted_no_issue_cycles_per_task": model_idle_task_cycles,
        "att_excess_cycles_per_task": att_excess_task_cycles,
        "wall_boundary_cycles_per_task": wall_boundary_task_cycles,
        "wall_residual_cycles_per_task": wall_residual_task_cycles,
        "wall_residual_cycles_per_tile": wall_residual_task_cycles
        / args.tiles_per_task,
        "attribution_scale": attribution_scale,
    }

    print(f"Dispatch: {args.dispatch_dir}")
    print(
        f"SIMDs={simd_count}, tasks/SIMD={tasks_per_simd}, tiles/task={args.tiles_per_task}, "
        f"wave files={sum(wave_counts)}"
    )
    print("\nCycle closure (cycles/task)")
    for name, value in closure.items():
        print(f"  {name:44s} {format_cycles(value)}")
    reconstructed = (
        args.ideal_task_cycles + att_excess_task_cycles + wall_boundary_task_cycles
    )
    print(f"  {'reconstructed_wall_task_cycles':44s} {format_cycles(reconstructed)}")

    print("\nMFMA logical shadow closure")
    for name, value in mfma_shadow_summary.items():
        print(f"  {name:44s} {format_cycles(value)}")

    print("\nMFMA-shadow x physical issue matrix")
    print(f"  {'region':18s} {'state':18s} {'cycles/task':>16s} {'cycles/tile':>16s}")
    for row in timeline_rows:
        print(
            f"  {row['region']:18s} {row['state']:18s} "
            f"{format_cycles(row['cycles_per_task']):>16s} "
            f"{format_cycles(row['cycles_per_tile']):>16s}"
        )

    print("\nNo-issue decomposition")
    print(
        f"  {'region':18s} {'physical/task':>16s} {'physical/tile':>16s} "
        f"{'residual/tile':>16s}"
    )
    for row in no_issue_regions:
        print(
            f"  {row['region']:18s} "
            f"{format_cycles(row['physical_cycles_per_task']):>16s} "
            f"{format_cycles(row['physical_cycles_per_tile']):>16s} "
            f"{format_cycles(row['attributed_residual_cycles_per_tile']):>16s}"
        )

    print("\nNo-issue blocker categories by region (physical cycles/tile)")
    for row in no_issue_regions:
        print(f"  {row['region']}:")
        for category in row["categories"]:
            print(
                f"    {category['category']:23s} "
                f"{format_cycles(category['physical_cycles_per_tile']):>12s}"
            )

    print("\nNon-MFMA issue work by region (physical cycles/tile)")
    for row in non_mfma_issue_regions:
        print(f"  {row['region']}: {format_cycles(row['physical_cycles_per_tile'])}")
        for stage in row["stages"][:10]:
            print(
                f"    {stage['stage']:35s} {format_cycles(stage['cycles_per_tile']):>12s}"
            )

    print("\nResidual attribution by blocker category")
    print(
        f"  {'category':25s} {'raw idle/task':>16s} {'residual/task':>16s} {'residual/tile':>16s}"
    )
    for row in category_rows:
        print(
            f"  {row['category']:25s} "
            f"{format_cycles(row['physical_idle_cycles_per_task']):>16s} "
            f"{format_cycles(row['residual_cycles_per_task']):>16s} "
            f"{format_cycles(row['residual_cycles_per_tile']):>16s}"
        )

    print("\nResidual attribution by source stage")
    print(f"  {'stage':35s} {'residual/task':>16s} {'residual/tile':>16s}")
    for row in stage_rows:
        print(
            f"  {row['stage']:35s} "
            f"{format_cycles(row['residual_cycles_per_task']):>16s} "
            f"{format_cycles(row['residual_cycles_per_tile']):>16s}"
        )

    print(f"\nTop-{args.topk} blocking PCs")
    print(
        f"  {'#':>3s} {'PC':>10s} {'category':18s} {'pyline':>8s} "
        f"{'residual/tile':>16s} {'stage':31s} ASM"
    )
    for rank, row in enumerate(pc_rows[: args.topk], 1):
        pc_text = f"0x{row['pc_addr']:x}" if row["pc_addr"] is not None else "-"
        line_text = str(row["python_line"]) if row["python_line"] else "-"
        print(
            f"  {rank:3d} {pc_text:>10s} {row['category']:18s} {line_text:>8s} "
            f"{format_cycles(row['residual_cycles_per_tile']):>16s} "
            f"{row['stage']:31s} {row['asm'][:70]}"
        )

    report = {
        "dispatch_dir": str(args.dispatch_dir),
        "simd_count": simd_count,
        "tasks_per_simd": tasks_per_simd,
        "tiles_per_task": args.tiles_per_task,
        "attribution_policy": {
            "simultaneous_blockers": "equal share of each physical no-issue interval",
            "model_budget": "attribute ATT excess over the ideal model in proportion to internal blocker weights",
        },
        "closure": closure,
        "mfma_shadow_summary": mfma_shadow_summary,
        "mfma_shadow_timeline_matrix": timeline_rows,
        "no_issue_regions": no_issue_regions,
        "non_mfma_issue_regions": non_mfma_issue_regions,
        "categories": category_rows,
        "stages": stage_rows,
        "top_pcs": pc_rows[: args.topk],
        "top_combinations": [
            {"combination": key, "physical_idle_cycles_per_task": value}
            for key, value in sorted(
                combination_task.items(), key=lambda item: item[1], reverse=True
            )[: args.topk]
        ],
    }
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with args.json.open("w") as stream:
            json.dump(report, stream, indent=2, sort_keys=True)
            stream.write("\n")
        print(f"\nJSON: {args.json}")


if __name__ == "__main__":
    main()
