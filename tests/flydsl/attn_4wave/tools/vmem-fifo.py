#!/usr/bin/env python3
"""运行并分析gfx942单wave VMEM issue/backpressure ATT实验。

该探针故意不使用 s_memtime、ABBA 或 burst 内 waitcnt。80 个 workgroup 先通过
全局原子 barrier 集合，每个 workgroup 静态占满 64 KiB LDS，从而每 CU 恰好
一个 workgroup。barrier 后只执行 SALU 冷却，再由每个 wave 连续发射静态展开、
无wait的 ``buffer_load_dwordx4`` / ``buffer_store_dwordx4``。支持纯load、纯store、
load:store=1:1和2:1四种序列；load使用互不重叠的VGPR/AGPR目标，无WAW。

``run``子命令生成可完整拼接的ATT目标kernel；``analyze``子命令读取rocprofv3 UI，
输出目标单wave的逐VMEM stall和issue gap。ATT数据不代表memory completion latency。
"""

import argparse
import json
import math
import os
import re
import statistics
from pathlib import Path
from typing import Any, cast

os.environ.setdefault("PYHIP_JIT_LOG", "0")
os.environ.setdefault("PYHIP_DEBUG_LOG", "")

import torch  # pyright: ignore[reportMissingImports]

from pyhip.core.asmjit import JIT, jit  # pyright: ignore[reportMissingImports]

UINT32 = "unsigned int"
VOID_POINTER = "void*"
WAVE_SIZE = 64
LDS_BYTES = 64 * 1024
DEFAULT_BUFFER_MIB = 1280
MAX_OPS = 100
DEFAULT_OPS = 96
MAX_VGPR_TARGET_SLOTS = 36
MAX_AGPR_TARGET_SLOTS = 64
ACCESS_PATTERNS = {
    "l": ("load",),
    "s": ("store",),
    "ls": ("load", "store"),
    "lls": ("load", "load", "store"),
}
ACCESS_PATTERN_CODES = {
    "load": "l",
    "store": "s",
    "load-store-1to1": "ls",
    "load-store-2to1": "lls",
}
VMEM_PREFIXES = ("buffer_load_dwordx4 ", "buffer_store_dwordx4 ")
WAVE_FILE_RE = re.compile(r"se(?P<se>\d+)_sm(?P<simd>\d+)_sl\d+_wv\d+\.json$")
LOAD_DESTINATION_RE = re.compile(r"buffer_load_dwordx4\s+([va](?:\d+|\[\d+:\d+\]))")
STORE_SOURCE_RE = re.compile(r"buffer_store_dwordx4\s+(v(?:\d+|\[\d+:\d+\]))")
GFX9_ATT_TIME_QUANTUM_CYCLES = 4
STORE_DWORDS = tuple(0x13579BDF + index for index in range(4))


def _operation_sequence(num_ops, access_pattern):
    unit = ACCESS_PATTERNS[access_pattern]
    if num_ops % len(unit):
        raise ValueError(f"{access_pattern}要求--ops能被{len(unit)}整除")
    return unit * (num_ops // len(unit))


def _infer_access_pattern(operations):
    for name, code in ACCESS_PATTERN_CODES.items():
        unit = ACCESS_PATTERNS[code]
        if len(operations) % len(unit) == 0 and operations == unit * (len(operations) // len(unit)):
            return name
    return None


@jit(no_pass=["pass_dse", "pass_dce"])
def vmem_single_wave_issue_burst(
    jit_builder: JIT,
    num_ops,
    access_pattern,
    load_slots,
    alignment_nops,
    cooldown_nops,
    start_delay_ticks,
    grid_blocks: UINT32,  # pyright: ignore[reportInvalidTypeForm]
    buffer_bytes,
    load_data: VOID_POINTER,  # pyright: ignore[reportInvalidTypeForm]
    store_data: VOID_POINTER,  # pyright: ignore[reportInvalidTypeForm]
    sync_state: VOID_POINTER,  # pyright: ignore[reportInvalidTypeForm]
    output: VOID_POINTER,  # pyright: ignore[reportInvalidTypeForm]
):
    """全 CU 集合后发射无 wait 的静态 VMEM burst。"""

    operation_sequence = _operation_sequence(num_ops, access_pattern)
    assert 1 <= num_ops <= MAX_OPS
    assert load_slots == operation_sequence.count("load")
    assert alignment_nops in (0, 1)
    store_slots = operation_sequence.count("store")

    access_dwords = 4
    bytes_per_lane = access_dwords * 4
    bytes_per_wave_op = WAVE_SIZE * bytes_per_lane
    ops_per_offset = 4096 // bytes_per_wave_op
    max_operations_per_kind = max(operation_sequence.count("load"), operation_sequence.count("store"))
    offset_count = (max_operations_per_kind + ops_per_offset - 1) // ops_per_offset
    bytes_per_wave_stream = max_operations_per_kind * bytes_per_wave_op

    lds_base = jit_builder.alloc_lds(LDS_BYTES, align=256)
    lds_address = jit_builder.gpr("vu32", jit_builder.threadIdx.x[0] * 4 + lds_base)
    jit_builder.ds_write_b32(lds_address, jit_builder.threadIdx.x[0])
    jit_builder.s_waitcnt(mod="lgkmcnt(0)")
    jit_builder.s_barrier()

    load_buffer = jit_builder.Buffer(load_data, buffer_bytes) if load_slots else None
    store_buffer = jit_builder.Buffer(store_data, buffer_bytes) if store_slots else None
    global_wave = jit_builder.gpr("vu32", jit_builder.blockIdx.x[0])
    vector_offset = jit_builder.gpr(
        "vu32",
        global_wave * bytes_per_wave_stream + jit_builder.lane_id[0] * bytes_per_lane,
    )
    vector_offsets = jit_builder.gpr(offset_count, "vu32")
    for chunk in range(offset_count):
        vector_offsets[chunk] = vector_offset + chunk * 4096

    vgpr_slots = min(load_slots, MAX_VGPR_TARGET_SLOTS)
    agpr_slots = load_slots - vgpr_slots
    assert agpr_slots <= MAX_AGPR_TARGET_SLOTS
    load_values = jit_builder.gpr(vgpr_slots, access_dwords, "vu32", align=access_dwords) if vgpr_slots else None
    load_accumulators = jit_builder.gpr(agpr_slots, access_dwords, "au32", align=access_dwords) if agpr_slots else None
    store_value = jit_builder.gpr(access_dwords, "vu32", align=access_dwords) if store_slots else None
    if store_slots:
        for dword_index in range(access_dwords):
            store_value[dword_index] = STORE_DWORDS[dword_index]

    counter_address = jit_builder.gpr("vu32", 0)
    target_address = jit_builder.gpr("vu32", 4)
    ready_address = jit_builder.gpr("vu32", 8)
    atomic_old = jit_builder.gpr("vu32")
    one = jit_builder.gpr("vu32", 1)
    zero = jit_builder.gpr("vu32", 0)
    arrived = jit_builder.gpr("su32", 0)
    with jit_builder.ExecMask(jit_builder.threadIdx.x[0] == 0):
        jit_builder.global_atomic_add(atomic_old, counter_address, one, sync_state, mod="sc0")
        jit_builder.s_waitcnt(mod="vmcnt(0)")
        with jit_builder.While(arrived[0] < grid_blocks[0]):
            jit_builder.global_atomic_add(atomic_old, counter_address, zero, sync_state, mod="sc0")
            jit_builder.s_waitcnt(mod="vmcnt(0)")
            jit_builder.v_readfirstlane_b32(arrived, atomic_old)
            jit_builder.s_nop(8)

        with jit_builder.If(jit_builder.blockIdx.x[0] == 0):
            target = jit_builder.gpr(2, "su32", align=2)
            jit_builder.s_memrealtime(target)
            jit_builder.s_waitcnt(mod="lgkmcnt(0)")
            jit_builder.s_add_u32(target[0], target[0], start_delay_ticks)
            target_value = jit_builder.gpr("vu32", target[0])
            jit_builder.global_atomic_add(atomic_old, target_address, target_value, sync_state, mod="sc0")
            jit_builder.s_waitcnt(mod="vmcnt(0)")
            jit_builder.global_atomic_add(atomic_old, ready_address, one, sync_state, mod="sc0")
            jit_builder.s_waitcnt(mod="vmcnt(0)")

        ready = jit_builder.gpr("su32", 0)
        with jit_builder.While(ready[0] == 0):
            jit_builder.global_atomic_add(atomic_old, ready_address, zero, sync_state, mod="sc0")
            jit_builder.s_waitcnt(mod="vmcnt(0)")
            jit_builder.v_readfirstlane_b32(ready, atomic_old)

        target_low = jit_builder.gpr("su32", 0)
        jit_builder.global_atomic_add(atomic_old, target_address, zero, sync_state, mod="sc0")
        jit_builder.s_waitcnt(mod="vmcnt(0)")
        jit_builder.v_readfirstlane_b32(target_low, atomic_old)

        # 先排空全局barrier的VMEM流量，再等待共同目标tick；目标tick后立即进入burst。
        for _ in range(cooldown_nops):
            jit_builder.s_nop(15)

        realtime_now = jit_builder.gpr(2, "su32", align=2)
        remaining = jit_builder.gpr("si32", 1)
        with jit_builder.While(remaining[0] > 0):
            jit_builder.s_memrealtime(realtime_now)
            jit_builder.s_waitcnt(mod="lgkmcnt(0)")
            jit_builder.s_sub_u32(remaining, target_low, realtime_now[0])
    jit_builder.s_barrier()
    for _ in range(alignment_nops):
        jit_builder.s_nop(0)

    load_index = 0
    store_index = 0
    for operation in operation_sequence:
        if operation == "load":
            load_target = (
                load_values[load_index] if load_index < vgpr_slots else load_accumulators[load_index - vgpr_slots]
            )
            chunk = load_index // ops_per_offset
            offset12 = (load_index % ops_per_offset) * bytes_per_wave_op
            load_buffer.load_dwordx4(load_target, vector_offsets[chunk], 0, offset12=offset12)
            load_index += 1
        else:
            chunk = store_index // ops_per_offset
            offset12 = (store_index % ops_per_offset) * bytes_per_wave_op
            store_buffer.store_dwordx4(store_value, vector_offsets[chunk], 0, offset12=offset12)
            store_index += 1

    jit_builder.s_waitcnt(mod="vmcnt(0)")
    sink = jit_builder.gpr("vu32", 0x13579BDF)
    for load_slot in range(vgpr_slots):
        for dword_index in range(access_dwords):
            jit_builder.v_xor_b32(sink, sink, load_values[load_slot, dword_index])
    if agpr_slots:
        accumulator_value = jit_builder.gpr("vu32")
        for load_slot in range(agpr_slots):
            for dword_index in range(access_dwords):
                jit_builder.v_accvgpr_read_b32(accumulator_value, load_accumulators[load_slot, dword_index])
                jit_builder.v_xor_b32(sink, sink, accumulator_value)

    hw_id = jit_builder.gpr("su32")
    xcc_id = jit_builder.gpr("su32")
    jit_builder.s_getreg_b32(hw_id, mod="hwreg(HW_REG_HW_ID, 0, 20)")
    jit_builder.s_getreg_b32(xcc_id, mod="hwreg(HW_REG_XCC_ID, 0, 4)")
    record = jit_builder.gpr(4, "vu32", align=4)
    record[0] = sink
    record[1] = hw_id
    record[2] = xcc_id
    record[3] = jit_builder.blockIdx.x[0] | (jit_builder.warp_id[0] << 16)
    record_address = jit_builder.gpr("vu32", global_wave * 16)
    with jit_builder.ExecMask(jit_builder.lane_id[0] == 0):
        jit_builder.global_store_dwordx4(record_address, record, output)
    jit_builder.s_waitcnt(mod="vmcnt(0)")


def _cu_key(hw_id, xcc_id):
    return (xcc_id, (hw_id >> 13) & 0x3, (hw_id >> 8) & 0xF)


def _validate_mapping(output, grid_blocks):
    rows = output.cpu().tolist()
    cus = set()
    block_cus = {}
    block_waves = {}
    block_simds = {}
    for sink, hw_id, xcc_id, packed_location in rows:
        block = int(packed_location) & 0xFFFF
        wave = (int(packed_location) >> 16) & 0xFFFF
        cu = _cu_key(int(hw_id), int(xcc_id))
        simd = (int(hw_id) >> 4) & 0x3
        cus.add(cu)
        block_cus.setdefault(block, set()).add(cu)
        block_waves.setdefault(block, set()).add(wave)
        block_simds.setdefault(block, set()).add(simd)
        if int(sink) == 0:
            raise RuntimeError("结果sink为0，目标VMEM burst可能未执行")

    if len(cus) != grid_blocks:
        raise RuntimeError(f"只覆盖 {len(cus)}/{grid_blocks} 个 CU")
    if any(len(value) != 1 for value in block_cus.values()):
        raise RuntimeError("至少一个 workgroup 跨越多个 CU")
    if len({next(iter(value)) for value in block_cus.values()}) != grid_blocks:
        raise RuntimeError("多个 workgroup 落在同一个 CU")
    if any(value != {0} for value in block_waves.values()):
        raise RuntimeError("单wave workgroup的wave ID不是0")
    if any(len(value) != 1 for value in block_simds.values()):
        raise RuntimeError("单wave workgroup映射到多个SIMD")
    return sorted(cus)


def _latest_isa(
    num_ops,
    access_pattern,
    load_slots,
    alignment_nops,
    cooldown_nops,
    start_delay_ticks,
    buffer_bytes,
):
    pattern = (
        f"vmem_single_wave_issue_burst-*-num_ops={num_ops}-access_pattern={access_pattern}-"
        f"load_slots={load_slots}-"
        f"alignment_nops={alignment_nops}-"
        f"cooldown_nops={cooldown_nops}-start_delay_ticks={start_delay_ticks}-"
        f"buffer_bytes={buffer_bytes}-*.s"
    )
    candidates = sorted(
        Path.home().joinpath(".pyhip").glob(pattern),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _validate_isa(path, operation_sequence):
    if path is None:
        raise RuntimeError("未找到刚编译的ISA文件")
    text = path.read_text()
    lines = text.splitlines()
    vmem_indices = [index for index, line in enumerate(lines) if any(prefix in line for prefix in VMEM_PREFIXES)]
    vmem_lines = [lines[index] for index in vmem_indices]
    actual_sequence = tuple("load" if "buffer_load_dwordx4 " in line else "store" for line in vmem_lines)
    if actual_sequence != operation_sequence:
        raise RuntimeError(f"ISA VMEM序列为{actual_sequence}，预期{operation_sequence}")
    if any(re.search(r"(?:^|\s)nt(?:\s|$)", line) for line in vmem_lines):
        raise RuntimeError("目标VMEM指令意外带有nt属性")
    burst_lines = lines[vmem_indices[0] : vmem_indices[-1] + 1]
    if any("s_waitcnt" in line for line in burst_lines):
        raise RuntimeError("目标VMEM burst内意外出现s_waitcnt")

    load_indices = [index for index, operation in enumerate(actual_sequence) if operation == "load"]
    load_lines = [vmem_lines[index] for index in load_indices]
    load_destinations = {
        match.group(1) for line in load_lines if (match := LOAD_DESTINATION_RE.search(line)) is not None
    }
    if len(load_destinations) != len(load_lines):
        raise RuntimeError(f"ISA中有{len(load_destinations)}/{len(load_lines)}组唯一load目标寄存器")

    store_indices = [index for index, operation in enumerate(actual_sequence) if operation == "store"]
    store_lines = [vmem_lines[index] for index in store_indices]
    store_sources = {match.group(1) for line in store_lines if (match := STORE_SOURCE_RE.search(line)) is not None}
    if store_lines and len(store_sources) != 1:
        raise RuntimeError(f"ISA中store使用{len(store_sources)}组源寄存器，预期1组")


def _percentile(values, fraction):
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _summary(values):
    if not values:
        return None
    return {
        "count": len(values),
        "min": min(values),
        "p10": _percentile(values, 0.10),
        "median": statistics.median(values),
        "p90": _percentile(values, 0.90),
        "max": max(values),
    }


def _resolve_ui_directory(directory):
    if (directory / "code.json").is_file():
        return directory
    candidates = sorted(directory.glob("ui_output_agent_*"))
    if len(candidates) != 1:
        raise RuntimeError(f"{directory} 下应恰好有一个ui_output_agent_*目录")
    return candidates[0]


def analyze_ui(directory, gap_threshold, long_gap_threshold):
    directory = _resolve_ui_directory(directory)
    code = json.loads((directory / "code.json").read_text())["code"]
    vmem_indices = [index for index, row in enumerate(code) if str(row[0]).strip().startswith(VMEM_PREFIXES)]
    if not vmem_indices:
        raise RuntimeError("code.json中没有buffer_load/store_dwordx4")
    vmem_ordinals = {index: ordinal for ordinal, index in enumerate(vmem_indices, 1)}
    operation_ordinals = {}
    operation_counts = {"load": 0, "store": 0}
    for code_index in vmem_indices:
        assembly = str(code[code_index][0]).strip()
        operation = "load" if assembly.startswith(VMEM_PREFIXES[0]) else "store"
        operation_counts[operation] += 1
        operation_ordinals[code_index] = operation_counts[operation]

    wave_files = sorted(directory.glob("se*_sm*_sl*_wv*.json"))
    if len(wave_files) != 1:
        raise RuntimeError(f"最终方法只接受1条wave trace，实际找到{len(wave_files)}条")
    wave_file = wave_files[0]
    match = WAVE_FILE_RE.match(wave_file.name)
    if match is None:
        raise RuntimeError(f"无法解析wave文件名: {wave_file.name}")
    payload = json.loads(wave_file.read_text())
    wave = payload["wave"]
    simd = int(match.group("simd"))
    events = []
    for timestamp, token_type, stall, duration, code_index in wave["instructions"]:
        if code_index not in vmem_ordinals:
            continue
        assembly = str(code[code_index][0]).strip()
        operation = "load" if assembly.startswith(VMEM_PREFIXES[0]) else "store"
        operand_match = (
            LOAD_DESTINATION_RE.search(assembly) if operation == "load" else STORE_SOURCE_RE.search(assembly)
        )
        events.append(
            {
                "operation": operation,
                "trace_begin_timestamp": int(timestamp),
                "issue_begin_timestamp": int(timestamp) + int(stall),
                "token_type": int(token_type),
                "stall_cycles": int(stall),
                "issue_cycles": int(duration) - int(stall),
                "code_index": int(code_index),
                "static_vmem_ordinal": vmem_ordinals[code_index],
                "static_operation_ordinal": operation_ordinals[code_index],
                "operand": None if operand_match is None else operand_match.group(1),
                "simd": simd,
            }
        )
    if len(events) != len(vmem_indices):
        raise RuntimeError(f"ATT只看到{len(events)}/{len(vmem_indices)}条VMEM，trace不完整")
    events.sort(key=lambda row: (row["issue_begin_timestamp"], row["code_index"]))
    dynamic_static_ordinals = [event["static_vmem_ordinal"] for event in events]
    expected_static_ordinals = list(range(1, len(vmem_indices) + 1))
    if dynamic_static_ordinals != expected_static_ordinals:
        raise RuntimeError(f"ATT动态VMEM顺序错位: {dynamic_static_ordinals}")

    dynamic_operations = tuple(event["operation"] for event in events)
    access_pattern = _infer_access_pattern(dynamic_operations)

    timeline = []
    gaps = []
    stalls_by_operation = {"load": [], "store": []}
    gaps_by_transition = {"load->load": [], "load->store": [], "store->load": [], "store->store": []}
    seen_destinations = set()
    first_reused_load = None
    dynamic_operation_counts = {"load": 0, "store": 0}
    previous_issue_begin = None
    previous_operation = None
    for vmem_number, event in enumerate(events, 1):
        operation = event["operation"]
        dynamic_operation_counts[operation] += 1
        operation_number = dynamic_operation_counts[operation]
        operand = event["operand"]
        target_reused = operation == "load" and operand is not None and operand in seen_destinations
        if operation == "load" and operand is not None:
            seen_destinations.add(operand)
        if target_reused and first_reused_load is None:
            first_reused_load = operation_number
        issue_begin = event["issue_begin_timestamp"]
        gap = None if previous_issue_begin is None else issue_begin - previous_issue_begin
        transition = None if previous_operation is None else f"{previous_operation}->{operation}"
        if gap is not None:
            gaps.append(gap)
            gaps_by_transition[transition].append(gap)
        stalls_by_operation[operation].append(event["stall_cycles"])
        timeline.append(
            {
                "vmem_number": vmem_number,
                "operation": operation,
                "operation_number": operation_number,
                "static_vmem_ordinal": event["static_vmem_ordinal"],
                "static_operation_ordinal": event["static_operation_ordinal"],
                "simd": event["simd"],
                "operand": operand,
                "target_reused": target_reused,
                "trace_begin_timestamp": event["trace_begin_timestamp"],
                "stall_cycles": event["stall_cycles"],
                "issue_begin_timestamp": issue_begin,
                "issue_cycles": event["issue_cycles"],
                "issue_gap_from_previous_cycles": gap,
                "transition_from_previous": transition,
            }
        )
        previous_issue_begin = issue_begin
        previous_operation = operation

    return {
        "directory": str(directory),
        "shader_engine": int(match.group("se")),
        "cu": int(wave["cu"]),
        "simd": simd,
        "static_vmem": len(vmem_indices),
        "dynamic_vmem": len(events),
        "static_operation_counts": operation_counts,
        "dynamic_operation_counts": dynamic_operation_counts,
        "access_pattern": access_pattern,
        "first_reused_load": first_reused_load,
        "time_axis": "gfx9_estimated_issue_begin",
        "issue_begin_formula": "trace_begin_timestamp + stall_cycles",
        "time_quantum_cycles": GFX9_ATT_TIME_QUANTUM_CYCLES,
        "gap_threshold_cycles": gap_threshold,
        "long_gap_threshold_cycles": long_gap_threshold,
        "gap_summary": _summary(gaps),
        "stall_summary_by_operation": {
            operation: _summary(values) for operation, values in stalls_by_operation.items() if values
        },
        "gap_summary_by_transition": {
            transition: _summary(values) for transition, values in gaps_by_transition.items() if values
        },
        "first_gap_position": next((position for position, gap in enumerate(gaps, 1) if gap >= gap_threshold), None),
        "first_long_gap_position": next(
            (position for position, gap in enumerate(gaps, 1) if gap >= long_gap_threshold),
            None,
        ),
        "issue_timeline": timeline,
    }


def render_analysis(report):
    gap = report["gap_summary"]
    counts = report["dynamic_operation_counts"]
    lines = [
        "# 单wave VMEM issue时间线",
        "",
        f"- UI: `{report['directory']}`",
        f"- SE/CU/SIMD: {report['shader_engine']}/{report['cu']}/{report['simd']}",
        f"- access pattern: {report['access_pattern']}",
        f"- VMEM: {report['dynamic_vmem']}；load/store: {counts['load']}/{counts['store']}",
        f"- first reused load: {report['first_reused_load']}",
        f"- 时间量化: {report['time_quantum_cycles']} shader cycles",
        f"- gap median/p90/max: {gap['median']}/{gap['p90']}/{gap['max']}",
        f"- first >= {report['gap_threshold_cycles']} gap: {report['first_gap_position']}",
        f"- first >= {report['long_gap_threshold_cycles']} gap: {report['first_long_gap_position']}",
        "",
        "## 分类统计",
        "",
        "| category | count | min | p10 | median | p90 | max |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for operation, summary in report["stall_summary_by_operation"].items():
        lines.append(
            f"| {operation} stall | {summary['count']} | {summary['min']} | {summary['p10']} | "
            f"{summary['median']} | {summary['p90']} | {summary['max']} |"
        )
    for transition, summary in report["gap_summary_by_transition"].items():
        lines.append(
            f"| {transition} gap | {summary['count']} | {summary['min']} | {summary['p10']} | "
            f"{summary['median']} | {summary['p90']} | {summary['max']} |"
        )
    lines.extend(
        [
            "",
            "## 逐条时间线",
            "",
            "| VMEM | op | op number | operand | reused load | stall | issue | transition | gap |",
            "|---:|---|---:|---|---|---:|---:|---|---:|",
        ]
    )
    for event in report["issue_timeline"]:
        lines.append(
            f"| {event['vmem_number']} | {event['operation']} | {event['operation_number']} | "
            f"{event['operand']} | {event['target_reused']} | {event['stall_cycles']} | "
            f"{event['issue_cycles']} | {event['transition_from_previous']} | "
            f"{event['issue_gap_from_previous_cycles']} |"
        )
    return "\n".join(lines) + "\n"


def _build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="运行ATT目标kernel")
    run_parser.add_argument("--device", type=int, default=0)
    run_parser.add_argument("--grid-blocks", type=int, help="workgroup数量；默认覆盖全部CU")
    run_parser.add_argument("--ops", type=int, default=DEFAULT_OPS, help="每wave的VMEM总指令数")
    run_parser.add_argument(
        "--access-pattern",
        choices=tuple(ACCESS_PATTERN_CODES),
        default="load",
        help="VMEM访问序列；混合模式严格按给定比例交错",
    )
    run_parser.add_argument("--launches", type=int, default=2, help="ATT建议抓第二次匹配launch")
    run_parser.add_argument("--alignment-nops", type=int, choices=(0, 1), default=0)
    run_parser.add_argument("--cooldown-nops", type=int, default=64)
    run_parser.add_argument(
        "--start-delay-ticks",
        type=int,
        default=5000,
        help="全GPU 100 MHz realtime目标距离发布时刻的tick数",
    )
    run_parser.add_argument("--buffer-mib", type=int, default=DEFAULT_BUFFER_MIB)
    run_parser.add_argument(
        "--launch-address-mode",
        choices=("same", "disjoint"),
        default="disjoint",
        help="same复用地址；disjoint让各launch访问互不重叠的地址区间",
    )

    analyze_parser = subparsers.add_parser("analyze", help="分析rocprofv3 ATT UI")
    analyze_parser.add_argument("directory", type=Path)
    analyze_parser.add_argument("--gap-threshold", type=int, default=16)
    analyze_parser.add_argument("--long-gap-threshold", type=int, default=100)
    analyze_parser.add_argument("--json", type=Path)
    analyze_parser.add_argument("--markdown", type=Path)
    return parser


def run_probe(args):
    if not 1 <= args.ops <= MAX_OPS:
        raise ValueError(f"--ops 必须在 1..{MAX_OPS}")
    access_pattern_code = ACCESS_PATTERN_CODES[args.access_pattern]
    operation_sequence = _operation_sequence(args.ops, access_pattern_code)
    load_slots = operation_sequence.count("load")
    store_slots = operation_sequence.count("store")
    if load_slots > MAX_VGPR_TARGET_SLOTS + MAX_AGPR_TARGET_SLOTS:
        raise ValueError(f"load数量不能超过{MAX_VGPR_TARGET_SLOTS + MAX_AGPR_TARGET_SLOTS}")
    if args.launches < 1:
        raise ValueError("--launches 必须为正数")
    if args.cooldown_nops < 0:
        raise ValueError("--cooldown-nops 不能为负")
    if args.start_delay_ticks <= 0:
        raise ValueError("--start-delay-ticks 必须为正数")

    torch.cuda.set_device(args.device)
    device = torch.device(f"cuda:{args.device}")
    properties = torch.cuda.get_device_properties(args.device)
    if "gfx942" not in properties.gcnArchName:
        raise RuntimeError(f"该探针只在 gfx942 验证，实际为 {properties.gcnArchName}")
    args.grid_blocks = properties.multi_processor_count if args.grid_blocks is None else args.grid_blocks
    if not 1 <= args.grid_blocks <= properties.multi_processor_count:
        raise ValueError(f"--grid-blocks 必须在 1..{properties.multi_processor_count}")

    buffer_bytes = args.buffer_mib * 1024 * 1024
    max_operations_per_kind = max(load_slots, store_slots)
    bytes_per_launch = args.grid_blocks * max_operations_per_kind * WAVE_SIZE * 16
    if bytes_per_launch > buffer_bytes:
        raise ValueError(f"每次launch需要{bytes_per_launch} bytes，超过--buffer-mib提供的{buffer_bytes} bytes")
    launch_stride = 0
    if args.launch_address_mode == "disjoint":
        alignment = 2 * 1024 * 1024
        launch_stride = ((bytes_per_launch + alignment - 1) // alignment) * alignment
    allocation_bytes = buffer_bytes + launch_stride * (args.launches - 1)
    load_data = torch.empty(allocation_bytes, dtype=torch.uint8, device=device) if load_slots else None
    store_data = torch.empty(allocation_bytes, dtype=torch.uint8, device=device) if store_slots else None
    if load_data is not None:
        load_data.fill_(1)
    if store_data is not None:
        store_data.fill_(0)
    sync_state = torch.zeros(3, dtype=torch.uint32, device=device)
    output = torch.zeros((args.grid_blocks, 4), dtype=torch.uint32, device=device)
    torch.cuda.synchronize()

    for launch_index in range(args.launches):
        sync_state.zero_()
        output.zero_()
        torch.cuda.synchronize()
        cast(Any, vmem_single_wave_issue_burst)(
            [args.grid_blocks],
            [WAVE_SIZE],
            args.ops,
            access_pattern_code,
            load_slots,
            args.alignment_nops,
            args.cooldown_nops,
            args.start_delay_ticks,
            args.grid_blocks,
            buffer_bytes,
            0 if load_data is None else load_data.data_ptr() + launch_index * launch_stride,
            0 if store_data is None else store_data.data_ptr() + launch_index * launch_stride,
            sync_state.data_ptr(),
            output.data_ptr(),
        )
        torch.cuda.synchronize()
        host_sync = sync_state.cpu().tolist()
        if int(host_sync[0]) != args.grid_blocks:
            raise RuntimeError(f"全局 barrier 只有 {int(host_sync[0])}/{args.grid_blocks} 个 WG")
        if int(host_sync[1]) == 0 or int(host_sync[2]) != 1:
            raise RuntimeError(f"共同起跑时钟发布失败: sync_state={host_sync}")
        covered_cus = _validate_mapping(output, args.grid_blocks)
        if store_data is not None:
            store_base = launch_index * launch_stride
            stored_dwords = store_data[store_base : store_base + 16].view(torch.uint32).cpu().tolist()
            if tuple(stored_dwords) != STORE_DWORDS:
                raise RuntimeError(f"store写回校验失败: {stored_dwords}")

    isa = _latest_isa(
        args.ops,
        access_pattern_code,
        load_slots,
        args.alignment_nops,
        args.cooldown_nops,
        args.start_delay_ticks,
        buffer_bytes,
    )
    _validate_isa(isa, operation_sequence)
    print(
        f"done device={args.device} grid_blocks={args.grid_blocks} covered_CUs={covered_cus} "
        f"waves/CU=1 width=4 dwords/lane "
        f"ops/wave={args.ops} access_pattern={args.access_pattern} "
        f"loads={load_slots} stores={store_slots} load_target_reuse=False "
        f"vgpr_load_slots={min(load_slots, MAX_VGPR_TARGET_SLOTS)} "
        f"agpr_load_slots={max(0, load_slots - MAX_VGPR_TARGET_SLOTS)} requests/CU={args.ops} "
        f"launches={args.launches} launch_address_mode={args.launch_address_mode} "
        f"launch_stride={launch_stride} "
        f"alignment_nops={args.alignment_nops} start_delay_ticks={args.start_delay_ticks} isa={isa}"
    )


def analyze_trace(args):
    if args.gap_threshold <= 0 or args.long_gap_threshold < args.gap_threshold:
        raise ValueError("阈值必须满足0 < gap <= long-gap")
    report = analyze_ui(args.directory, args.gap_threshold, args.long_gap_threshold)
    markdown = render_analysis(report)
    if args.json:
        args.json.write_text(json.dumps(report, indent=2) + "\n")
    if args.markdown:
        args.markdown.write_text(markdown)
    print(markdown, end="")


def main():
    args = _build_parser().parse_args()
    if args.command == "run":
        run_probe(args)
    else:
        analyze_trace(args)


if __name__ == "__main__":
    main()
