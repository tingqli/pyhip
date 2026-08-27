#!/usr/bin/env python3
"""搜索gfx942四wave workgroup的buffer load/store带宽配置。"""

import argparse
import csv
import json
import os
import random
import re
import statistics
from pathlib import Path
from typing import Any, cast

os.environ.setdefault("PYHIP_JIT_LOG", "0")
os.environ.setdefault("PYHIP_DEBUG_LOG", "")

import torch  # pyright: ignore[reportMissingImports]

from pyhip.core.asmjit import JIT, jit  # pyright: ignore[reportMissingImports]
from pyhip.misc import cudaPerf  # pyright: ignore[reportMissingImports]

UINT32 = "unsigned int"
VOID_POINTER = "void*"
THREADS = 256
WAVES_PER_WORKGROUP = 4
LDS_BYTES = 64 * 1024
PIPELINE_DEPTH = 32
CONTIGUOUS_THREADS = (256, 64, 16, 8, 4)
LANE_BYTES = (16, 8, 4)
OPERATIONS = ("load", "store")
CACHE_POLICIES = {
    "default": "",
    "sc0": "sc0",
    "nt": "nt",
    "sc1": "sc1",
    "sc0-nt": "sc0 nt",
    "sc0-sc1": "sc0 sc1",
    "nt-sc1": "nt sc1",
    "sc0-nt-sc1": "sc0 nt sc1",
}
DEFAULT_CACHE_POLICIES = tuple(CACHE_POLICIES)
CACHE_POLICY_CODES = {name: f"c{index}" for index, name in enumerate(CACHE_POLICIES)}
CACHE_POLICY_FROM_CODE = {code: name for name, code in CACHE_POLICY_CODES.items()}
DEFAULT_BYTES_PER_WORKGROUP = 8 * 1024 * 1024
DEFAULT_BUFFERS = 8
DEFAULT_WARMUPS = 2
DEFAULT_SAMPLES = 9


def _instruction_suffix(lane_bytes):
    return {4: "dword", 8: "dwordx2", 16: "dwordx4"}[lane_bytes]


def _cache_mod(cache_policy_code):
    return CACHE_POLICIES[CACHE_POLICY_FROM_CODE[cache_policy_code]]


@jit(no_pass=["pass_dse", "pass_dce"])
def vmem_bandwidth_kernel(
    jit_builder: JIT,
    operation,
    contiguous_threads,
    lane_bytes,
    cache_policy_code,
    bytes_per_workgroup,
    buffer_bytes: UINT32,  # pyright: ignore[reportInvalidTypeForm]
    data: VOID_POINTER,  # pyright: ignore[reportInvalidTypeForm]
    sink_output: VOID_POINTER,  # pyright: ignore[reportInvalidTypeForm]
):
    """四wave协作扫描等长stripe；只改变线程连续度、指令宽度和cache位。"""

    assert operation in OPERATIONS
    assert contiguous_threads in CONTIGUOUS_THREADS
    assert lane_bytes in LANE_BYTES
    assert cache_policy_code in CACHE_POLICY_FROM_CODE
    assert bytes_per_workgroup % (THREADS * lane_bytes * PIPELINE_DEPTH) == 0

    lds_base = jit_builder.alloc_lds(LDS_BYTES, align=256)
    lds_address = jit_builder.gpr("vu32", jit_builder.threadIdx.x[0] * 4 + lds_base)
    jit_builder.ds_write_b32(lds_address, jit_builder.threadIdx.x[0])
    jit_builder.s_waitcnt(mod="lgkmcnt(0)")
    jit_builder.s_barrier()

    data_buffer = jit_builder.Buffer(data, buffer_bytes)
    thread_id = jit_builder.threadIdx.x[0]
    group_id = thread_id // contiguous_threads
    lane_in_group = thread_id % contiguous_threads
    groups_per_workgroup = THREADS // contiguous_threads
    group_bytes = bytes_per_workgroup // groups_per_workgroup
    block_base = jit_builder.blockIdx.x[0] * bytes_per_workgroup
    instruction_stride = contiguous_threads * lane_bytes
    vector_offsets = jit_builder.gpr(PIPELINE_DEPTH, "vu32")
    for slot in range(PIPELINE_DEPTH):
        vector_offsets[slot] = (
            block_base + group_id * group_bytes + lane_in_group * lane_bytes + slot * instruction_stride
        )

    dwords = lane_bytes // 4
    load_values = jit_builder.gpr(PIPELINE_DEPTH, dwords, "vu32", align=dwords) if operation == "load" else None
    store_value = jit_builder.gpr(dwords, "vu32", align=dwords) if operation == "store" else None
    if operation == "store":
        store_value = cast(Any, store_value)
        for dword in range(dwords):
            store_value[dword] = 0x13579BDF + dword

    iteration_stride = contiguous_threads * lane_bytes * PIPELINE_DEPTH
    iterations = group_bytes // iteration_stride
    iteration = jit_builder.gpr("su32", 0)
    scalar_offset = jit_builder.gpr("su32", 0)
    instruction = _instruction_suffix(lane_bytes)
    cache_mod = _cache_mod(cache_policy_code)
    with jit_builder.While(iteration[0] < iterations):
        for slot in range(PIPELINE_DEPTH):
            mod = "offen"
            if cache_mod:
                mod += f" {cache_mod}"
            if operation == "load":
                load_values = cast(Any, load_values)
                getattr(jit_builder, f"buffer_load_{instruction}")(
                    load_values[slot], vector_offsets[slot], data_buffer.desc, scalar_offset, mod=mod
                )
            else:
                getattr(jit_builder, f"buffer_store_{instruction}")(
                    store_value, vector_offsets[slot], data_buffer.desc, scalar_offset, mod=mod
                )
        jit_builder.s_waitcnt(mod="vmcnt(0)")
        scalar_offset[0] += iteration_stride
        iteration[0] += 1

    sink = jit_builder.gpr("vu32", 0x2468ACE0)
    if operation == "load":
        load_values = cast(Any, load_values)
        for slot in range(PIPELINE_DEPTH):
            for dword in range(dwords):
                jit_builder.v_add_u32(sink, sink, load_values[slot, dword])
    hw_id = jit_builder.gpr("su32")
    xcc_id = jit_builder.gpr("su32")
    jit_builder.s_getreg_b32(hw_id, mod="hwreg(HW_REG_HW_ID, 0, 20)")
    jit_builder.s_getreg_b32(xcc_id, mod="hwreg(HW_REG_XCC_ID, 0, 4)")
    record = jit_builder.gpr(4, "vu32", align=4)
    record[0] = sink
    record[1] = hw_id
    record[2] = xcc_id
    record[3] = jit_builder.blockIdx.x[0] | (jit_builder.warp_id[0] << 16)
    record_index = jit_builder.blockIdx.x[0] * WAVES_PER_WORKGROUP + jit_builder.warp_id[0]
    record_address = jit_builder.gpr("vu32", record_index * 16)
    with jit_builder.ExecMask(jit_builder.lane_id[0] == 0):
        jit_builder.global_store_dwordx4(record_address, record, sink_output)
    jit_builder.s_waitcnt(mod="vmcnt(0)")


def _latest_isa(operation, contiguous_threads, lane_bytes, cache_policy_code, bytes_per_workgroup):
    pattern = (
        "vmem_bandwidth_kernel-*-"
        f"operation={operation}-contiguous_threads={contiguous_threads}-lane_bytes={lane_bytes}-"
        f"cache_policy_code={cache_policy_code}-bytes_per_workgroup={bytes_per_workgroup}-*.s"
    )
    candidates = sorted(
        Path.home().joinpath(".pyhip").glob(pattern),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _validate_isa(path, operation, lane_bytes, cache_policy):
    if path is None:
        raise RuntimeError("未找到刚编译的ISA文件")
    mnemonic = f"buffer_{operation}_{_instruction_suffix(lane_bytes)}"
    target_lines = [line.strip() for line in path.read_text().splitlines() if mnemonic in line]
    if not target_lines:
        raise RuntimeError(f"ISA中没有{mnemonic}")
    expected_modifiers = set(CACHE_POLICIES[cache_policy].split())
    cache_tokens = {"sc0", "nt", "sc1"}
    for line in target_lines:
        actual_modifiers = set(line.split()) & cache_tokens
        if actual_modifiers != expected_modifiers:
            raise RuntimeError(f"cache修饰符为{sorted(actual_modifiers)}，预期{sorted(expected_modifiers)}: {line}")
    if len(target_lines) != PIPELINE_DEPTH:
        raise RuntimeError(f"ISA中有{len(target_lines)}条目标VMEM，预期{PIPELINE_DEPTH}")
    if operation == "load":
        destination_pattern = re.compile(rf"{mnemonic}\s+([va](?:\d+|\[\d+:\d+\]))")
        destinations = {
            match.group(1) for line in target_lines if (match := destination_pattern.search(line)) is not None
        }
        if len(destinations) != PIPELINE_DEPTH:
            raise RuntimeError(f"ISA中有{len(destinations)}组唯一load目标，预期{PIPELINE_DEPTH}")
    else:
        source_pattern = re.compile(rf"{mnemonic}\s+(v(?:\d+|\[\d+:\d+\]))")
        sources = {match.group(1) for line in target_lines if (match := source_pattern.search(line)) is not None}
        if len(sources) != 1:
            raise RuntimeError(f"ISA中store使用{len(sources)}组源寄存器，预期1组")
    if re.search(r"^\s*scratch_(?:load|store)", path.read_text(), re.MULTILINE):
        raise RuntimeError("ISA意外包含scratch访问")
    return len(target_lines)


def _parse_csv_list(value):
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _unique(values):
    return tuple(dict.fromkeys(values))


def _validate_case(contiguous_threads, lane_bytes, bytes_per_workgroup):
    if contiguous_threads not in CONTIGUOUS_THREADS:
        raise ValueError(f"continuous threads必须是{CONTIGUOUS_THREADS}")
    if lane_bytes not in LANE_BYTES:
        raise ValueError(f"lane bytes必须是{LANE_BYTES}")
    quantum = THREADS * lane_bytes * PIPELINE_DEPTH
    if bytes_per_workgroup < quantum:
        raise ValueError(f"--bytes-per-workgroup至少为{quantum}")
    if bytes_per_workgroup % quantum:
        raise ValueError(f"--bytes-per-workgroup必须被{quantum}整除")


def _cu_key(hw_id, xcc_id):
    return (xcc_id, (hw_id >> 13) & 0x3, (hw_id >> 8) & 0xF)


def _validate_mapping(records, grid_blocks):
    rows = records.cpu().tolist()
    block_cus = {}
    block_waves = {}
    block_simds = {}
    for _, hw_id, xcc_id, packed_location in rows:
        block = int(packed_location) & 0xFFFF
        wave = (int(packed_location) >> 16) & 0xFFFF
        block_cus.setdefault(block, set()).add(_cu_key(int(hw_id), int(xcc_id)))
        block_waves.setdefault(block, set()).add(wave)
        block_simds.setdefault(block, set()).add((int(hw_id) >> 4) & 0x3)
    expected_blocks = set(range(grid_blocks))
    if set(block_cus) != expected_blocks:
        raise RuntimeError(f"记录只覆盖block {sorted(block_cus)}，预期{sorted(expected_blocks)}")
    if any(len(cus) != 1 for cus in block_cus.values()):
        raise RuntimeError("至少一个workgroup跨越多个CU")
    if len({next(iter(cus)) for cus in block_cus.values()}) != grid_blocks:
        raise RuntimeError("多个workgroup落在同一个CU")
    if any(waves != set(range(WAVES_PER_WORKGROUP)) for waves in block_waves.values()):
        raise RuntimeError("至少一个workgroup没有完整覆盖wave 0-3")
    if any(simds != set(range(WAVES_PER_WORKGROUP)) for simds in block_simds.values()):
        raise RuntimeError("至少一个workgroup的四个wave没有分布到四个SIMD")


def _launch_case(args, data, sink, operation, contiguous_threads, lane_bytes, cache_policy):
    buffer_bytes = data.numel()
    cache_policy_code = CACHE_POLICY_CODES[cache_policy]
    cast(Any, vmem_bandwidth_kernel)(
        [args.grid_blocks],
        [THREADS],
        operation,
        contiguous_threads,
        lane_bytes,
        cache_policy_code,
        args.bytes_per_workgroup,
        buffer_bytes,
        data.data_ptr(),
        sink.data_ptr(),
    )


def _validate_execution(args, data, sink, operation, contiguous_threads, lane_bytes, cache_policy):
    sink.zero_()
    _launch_case(args, data, sink, operation, contiguous_threads, lane_bytes, cache_policy)
    torch.cuda.synchronize()
    _validate_mapping(sink, args.grid_blocks)
    rows = sink.cpu().tolist()
    if operation == "load":
        group_bytes = args.bytes_per_workgroup // (THREADS // contiguous_threads)
        loaded_values = []
        for _, _, _, packed_location in rows:
            block = int(packed_location) & 0xFFFF
            wave = (int(packed_location) >> 16) & 0xFFFF
            thread_id = wave * 64
            group_id = thread_id // contiguous_threads
            lane_in_group = thread_id % contiguous_threads
            byte_offset = block * args.bytes_per_workgroup + group_id * group_bytes + lane_in_group * lane_bytes
            loaded_bytes = bytes(data[byte_offset : byte_offset + 4].cpu().tolist())
            loaded_values.append(int.from_bytes(loaded_bytes, "little"))
        loaded_dwords = PIPELINE_DEPTH * (lane_bytes // 4)
        expected_sinks = [(0x2468ACE0 + int(value) * loaded_dwords) & 0xFFFFFFFF for value in loaded_values]
    else:
        expected_sinks = [0x2468ACE0] * len(rows)
    sink_values = [int(row[0]) for row in rows]
    if sink_values != expected_sinks:
        raise RuntimeError(f"sink校验失败: got={sink_values[:8]} expected={expected_sinks[:8]}")
    if operation == "store":
        first_values = []
        last_values = []
        for block in range(args.grid_blocks):
            first_offset = block * args.bytes_per_workgroup
            last_offset = (block + 1) * args.bytes_per_workgroup - 4
            first_values.append(int.from_bytes(bytes(data[first_offset : first_offset + 4].cpu().tolist()), "little"))
            last_values.append(int.from_bytes(bytes(data[last_offset : last_offset + 4].cpu().tolist()), "little"))
        expected_last = 0x13579BDF + lane_bytes // 4 - 1
        if any(int(value) != 0x13579BDF for value in first_values) or any(
            int(value) != expected_last for value in last_values
        ):
            raise RuntimeError(
                f"store写回校验失败: first={first_values[:8]} last={last_values[:8]} expected_last={expected_last:#x}"
            )


def _measure_case(args, buffers, sink, operation, contiguous_threads, lane_bytes, cache_policy):
    buffers[0].fill_(1)
    torch.cuda.synchronize()
    _validate_execution(
        args,
        buffers[0],
        sink,
        operation,
        contiguous_threads,
        lane_bytes,
        cache_policy,
    )
    _launch_case(args, buffers[0], sink, operation, contiguous_threads, lane_bytes, cache_policy)
    torch.cuda.synchronize()
    for warmup in range(args.warmups * len(buffers)):
        _launch_case(
            args, buffers[warmup % len(buffers)], sink, operation, contiguous_threads, lane_bytes, cache_policy
        )
    torch.cuda.synchronize()

    elapsed_ms = []
    for sample in range(args.samples):
        timer = cudaPerf(rw_bytes=args.grid_blocks * args.bytes_per_workgroup, name="", verbose=0)
        with timer:
            _launch_case(
                args,
                buffers[sample % len(buffers)],
                sink,
                operation,
                contiguous_threads,
                lane_bytes,
                cache_policy,
            )
        elapsed_ms.append(timer.dt() * 1e3)

    median_ms = statistics.median(elapsed_ms)
    transferred_bytes = args.grid_blocks * args.bytes_per_workgroup
    bandwidth_gbps = transferred_bytes * 1e-6 / median_ms
    isa = _latest_isa(
        operation,
        contiguous_threads,
        lane_bytes,
        CACHE_POLICY_CODES[cache_policy],
        args.bytes_per_workgroup,
    )
    static_vmem = _validate_isa(isa, operation, lane_bytes, cache_policy)
    return {
        "operation": operation,
        "contiguous_threads": contiguous_threads,
        "lane_bytes": lane_bytes,
        "cache_policy": cache_policy,
        "cache_modifiers": CACHE_POLICIES[cache_policy],
        "median_ms": median_ms,
        "bandwidth_gbps": bandwidth_gbps,
        "samples_ms": elapsed_ms,
        "static_vmem": static_vmem,
        "isa": str(isa),
    }


def _write_results(args, payload):
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2) + "\n")
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as stream:
            fieldnames = (
                "operation",
                "contiguous_threads",
                "lane_bytes",
                "cache_policy",
                "cache_modifiers",
                "median_ms",
                "bandwidth_gbps",
                "static_vmem",
            )
            writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(payload["results"])


def _print_ranking(results, top):
    for operation in OPERATIONS:
        ranked = sorted(
            (result for result in results if result["operation"] == operation),
            key=lambda result: result["bandwidth_gbps"],
            reverse=True,
        )
        if not ranked:
            continue
        print(f"\n{operation} top {min(top, len(ranked))}")
        print("rank  threads  bytes/lane  cache          median_ms  GB/s")
        for rank, result in enumerate(ranked[:top], 1):
            print(
                f"{rank:>4}  {result['contiguous_threads']:>7}  {result['lane_bytes']:>10}  "
                f"{result['cache_policy']:<13}  {result['median_ms']:>9.4f}  "
                f"{result['bandwidth_gbps']:>8.1f}"
            )


def run_benchmark(args):
    operations = _unique(_parse_csv_list(args.operations))
    contiguous_values = _unique(map(int, _parse_csv_list(args.contiguous_threads)))
    lane_values = _unique(map(int, _parse_csv_list(args.lane_bytes)))
    cache_values = _unique(_parse_csv_list(args.cache_policies))
    if not operations or any(operation not in OPERATIONS for operation in operations):
        raise ValueError(f"--operations必须来自{OPERATIONS}")
    if not cache_values or any(policy not in CACHE_POLICIES for policy in cache_values):
        raise ValueError(f"--cache-policies必须来自{tuple(CACHE_POLICIES)}")
    for contiguous_threads in contiguous_values:
        for lane_bytes in lane_values:
            _validate_case(contiguous_threads, lane_bytes, args.bytes_per_workgroup)

    cases = [
        (operation, contiguous_threads, lane_bytes, cache_policy)
        for operation in operations
        for contiguous_threads in contiguous_values
        for lane_bytes in lane_values
        for cache_policy in cache_values
    ]
    random.Random(args.seed).shuffle(cases)
    if args.dry_run:
        if args.grid_blocks is None:
            allocation = "grid=auto; transfer/allocation resolved at run time"
        else:
            buffer_bytes = args.grid_blocks * args.bytes_per_workgroup
            allocation = (
                f"grid={args.grid_blocks}; transfer/launch={buffer_bytes / 2**30:.3f} GiB; "
                f"allocation={buffer_bytes * args.buffers / 2**30:.3f} GiB"
            )
        print(f"matrix={len(cases)} cases; bytes/workgroup={args.bytes_per_workgroup}; {allocation}")
        for case_index, (operation, contiguous_threads, lane_bytes, cache_policy) in enumerate(cases, 1):
            print(
                f"[{case_index:03d}/{len(cases)}] {operation} threads={contiguous_threads} "
                f"bytes/lane={lane_bytes} cache={cache_policy} modifiers={CACHE_POLICIES[cache_policy]!r}"
            )
        return

    torch.cuda.set_device(args.device)
    properties = torch.cuda.get_device_properties(args.device)
    if "gfx942" not in properties.gcnArchName:
        raise RuntimeError(f"该工具只在gfx942验证，实际为{properties.gcnArchName}")
    args.grid_blocks = properties.multi_processor_count if args.grid_blocks is None else args.grid_blocks
    if not 1 <= args.grid_blocks <= properties.multi_processor_count:
        raise ValueError(f"--grid-blocks必须在1..{properties.multi_processor_count}")
    buffer_bytes = args.grid_blocks * args.bytes_per_workgroup
    if buffer_bytes > 0xFFFFFFFF:
        raise ValueError(f"单个buffer为{buffer_bytes} bytes，超过32位buffer descriptor范围")
    allocation_bytes = buffer_bytes * args.buffers
    print(
        f"matrix={len(cases)} cases transfer/launch={buffer_bytes / 2**30:.3f} GiB "
        f"buffers={args.buffers} allocation={allocation_bytes / 2**30:.3f} GiB"
    )
    device = torch.device(f"cuda:{args.device}")
    buffers = [torch.empty(buffer_bytes, dtype=torch.uint8, device=device) for _ in range(args.buffers)]
    for index, buffer in enumerate(buffers):
        buffer.fill_(index + 1)
    sink = torch.zeros((args.grid_blocks * WAVES_PER_WORKGROUP, 4), dtype=torch.uint32, device=device)
    torch.cuda.synchronize()

    results = []
    for case_index, (operation, contiguous_threads, lane_bytes, cache_policy) in enumerate(cases, 1):
        result = _measure_case(
            args,
            buffers,
            sink,
            operation,
            contiguous_threads,
            lane_bytes,
            cache_policy,
        )
        result["execution_index"] = case_index
        results.append(result)
        print(
            f"[{case_index:03d}/{len(cases)}] {operation} threads={contiguous_threads} "
            f"bytes/lane={lane_bytes} cache={cache_policy} "
            f"median={result['median_ms']:.4f} ms bandwidth={result['bandwidth_gbps']:.1f} GB/s"
        )

    payload = {
        "device": args.device,
        "architecture": properties.gcnArchName,
        "grid_blocks": args.grid_blocks,
        "threads_per_workgroup": THREADS,
        "waves_per_workgroup": WAVES_PER_WORKGROUP,
        "pipeline_depth": PIPELINE_DEPTH,
        "lds_bytes_per_workgroup": LDS_BYTES,
        "bytes_per_workgroup": args.bytes_per_workgroup,
        "buffers": args.buffers,
        "warmups": args.warmups,
        "samples": args.samples,
        "timer": "pyhip.cudaPerf",
        "seed": args.seed,
        "address_mapping": (
            "group=thread_id//contiguous_threads; lane=thread_id%contiguous_threads; "
            "each group scans an independent equal-sized stripe"
        ),
        "matrix": {
            "operations": operations,
            "contiguous_threads": contiguous_values,
            "lane_bytes": lane_values,
            "cache_policies": cache_values,
            "cache_modifiers": {policy: CACHE_POLICIES[policy] for policy in cache_values},
        },
        "results": results,
    }
    _write_results(args, payload)
    _print_ranking(results, args.top)


def _build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--grid-blocks", type=int, help="workgroup数量；默认覆盖全部CU")
    parser.add_argument("--bytes-per-workgroup", type=int, default=DEFAULT_BYTES_PER_WORKGROUP)
    parser.add_argument("--buffers", type=int, default=DEFAULT_BUFFERS)
    parser.add_argument("--warmups", type=int, default=DEFAULT_WARMUPS)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--operations", default=",".join(OPERATIONS))
    parser.add_argument("--contiguous-threads", default=",".join(map(str, CONTIGUOUS_THREADS)))
    parser.add_argument("--lane-bytes", default=",".join(map(str, LANE_BYTES)))
    parser.add_argument("--cache-policies", default=",".join(DEFAULT_CACHE_POLICIES))
    parser.add_argument("--seed", type=int, default=942)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true", help="只输出随机化矩阵和显存需求")
    parser.add_argument("--json", type=Path)
    parser.add_argument("--csv", type=Path)
    return parser


def main():
    args = _build_parser().parse_args()
    if args.buffers < 1 or args.warmups < 0 or args.samples < 1 or args.top < 1:
        raise ValueError("buffers/samples/top必须为正数，warmups不能为负")
    run_benchmark(args)


if __name__ == "__main__":
    main()
