#!/usr/bin/env python3
"""用持续满载的 PyHIP JIT kernel 测量不同指令组合下的 GPU 频率。

kernel 使用固定 100 MHz 的 ``s_memrealtime`` 自行控制执行时间，并同时读取
随 shader clock 变化的 ``s_memtime``。两者的增量比可以给出短至 1 ms 的整段
有效 SCLK；后台线程再从 sysfs 采集 SCLK、PPT 功耗和温度轨迹。
"""

import argparse
import csv
import json
import math
import re
import statistics
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch

from pyhip.core.asmjit import JIT, jit

REALTIME_HZ = 100_000_000
REALTIME_TICKS_PER_MS = REALTIME_HZ // 1000
THREADS_PER_BLOCK = 256
WAVE_SIZE = 64
WAVES_PER_BLOCK = THREADS_PER_BLOCK // WAVE_SIZE
RECORD_BYTES = 32
MAX_OUTSTANDING_LOADS = 16
MAX_DURATION_MS = 30_000.0
MFMA_LATENCY_CYCLES = 16
MEM_WAIT_MIN_CYCLES = 1024
DS_WAIT_MIN_CYCLES = 256
MEM_WAIT_MFMAS = MEM_WAIT_MIN_CYCLES // MFMA_LATENCY_CYCLES
DS_WAIT_MFMAS = DS_WAIT_MIN_CYCLES // MFMA_LATENCY_CYCLES
MFMA_EXP_MAX_BLOCKS_PER_CU = 2
FP8_DENSITY_THROTTLE_MAX_CYCLES_PER_MFMA = 60.356
FP8_DENSITY_HIGH_FREQUENCY_MIN_CYCLES_PER_MFMA = 60.982
BF16_DENSITY_THROTTLE_MAX_CYCLES_PER_MFMA = 59.718
BF16_DENSITY_HIGH_FREQUENCY_MIN_CYCLES_PER_MFMA = 60.978
assert MEM_WAIT_MFMAS * MFMA_LATENCY_CYCLES == MEM_WAIT_MIN_CYCLES
assert DS_WAIT_MFMAS * MFMA_LATENCY_CYCLES == DS_WAIT_MIN_CYCLES

# PyHIP 将这些注解字符串直接用作生成 kernel 的 C ABI 类型。
UINT32 = "unsigned int"
VOID_POINTER = "void*"

WORKLOADS = {
    "mfma": "纯 MFMA",
    "mfma_valu": "MFMA + 可共发 scalar VALU",
    "mfma_valu_burst": "连续 MFMA burst + 连续 VALU burst",
    "bf16_32": "生产同款 BF16 32x32 MFMA burst + 连续 VALU burst",
    "fp8_mfma_valu_burst": "生产同款 FP8 32x32 MFMA burst + 连续 VALU burst",
    "mfma_exp": "MFMA + EXP 交织",
    "exp": "纯 EXP",
    "valu": "纯 scalar VALU FMA",
    "mfma_mem": "MFMA + 延迟消费的 non-temporal HBM load",
    "mfma_ds_mem": "MFMA + LDS read + non-temporal HBM load 交织",
    "mem": "纯 non-temporal HBM load",
}
SPECIAL_WORKLOADS = {
    "mfma_valu_burst",
    "bf16_32",
    "fp8_mfma_valu_burst",
}
DEFAULT_WORKLOADS = tuple(name for name in WORKLOADS if name not in SPECIAL_WORKLOADS)


def _make_mfma_registers(jit_builder):
    if jit_builder.gfx >= 950:
        operand_a = jit_builder.gpr(8, "vu32", align=2)
        operand_b = jit_builder.gpr(8, "vu32", align=2)
        operand_a[...] = 0x01010101
        operand_b[...] = 0x01010101
    else:
        operand_a = jit_builder.gpr(2, "vu32", align=2)
        operand_b = jit_builder.gpr(2, "vu32", align=2)
        operand_a[...] = 0x3F803F80
        operand_b[...] = 0x3F803F80
    accumulators = jit_builder.gpr(4, 4, "vf32", align=4)
    accumulators[...] = 0.0
    return accumulators, operand_a, operand_b


def _emit_mfma(jit_builder, accumulators, operand_a, operand_b, slot):
    if jit_builder.gfx >= 950:
        jit_builder.v_mfma_f32_16x16x128_f8f6f4(accumulators[slot], operand_a, operand_b, 0)
    else:
        jit_builder.v_mfma_f32_16x16x16_bf16(accumulators[slot], operand_a, operand_b, 0)


def _make_bf16_32x32_mfma_registers(jit_builder):
    if not jit_builder.gfx < 950:
        raise RuntimeError(
            "bf16_32 当前只用于验证 gfx94x production MFMA"
        )
    operand_a = jit_builder.gpr(2, "vu32", 0x3F803F80, align=2)
    operand_b = jit_builder.gpr(2, "vu32", 0x3F803F80, align=2)
    accumulators = jit_builder.gpr(4, 16, "vf32", align=4)
    accumulators[...] = 0.0
    return accumulators, operand_a, operand_b


def _emit_bf16_32x32_mfma(jit_builder, accumulators, operand_a, operand_b, slot):
    jit_builder.v_mfma_f32_32x32x8_bf16(
        accumulators[slot], operand_a, operand_b, 0
    )


def _make_fp8_mfma_registers(jit_builder):
    if not jit_builder.gfx < 950:
        raise RuntimeError("fp8_mfma_valu_burst 当前只用于验证 gfx94x 生产 FP8 MFMA")
    operand_a = jit_builder.gpr(2, "vu32", 0x40404040, align=2)
    operand_b = jit_builder.gpr(2, "vu32", 0x40404040, align=2)
    accumulators = jit_builder.gpr(4, 16, "vf32", align=4)
    accumulators[...] = 0.0
    return accumulators, operand_a, operand_b


def _emit_fp8_mfma(jit_builder, accumulators, operand_a, operand_b, slot):
    jit_builder.v_mfma_f32_32x32x16_fp8_fp8(
        accumulators[slot], operand_a, operand_b, 0
    )


def _read_counter(jit_builder, instruction):
    value = jit_builder.gpr(2, "su32", align=2)
    getattr(jit_builder, instruction)(value)
    jit_builder.s_waitcnt(mod="lgkmcnt(0)")
    return value


def _subtract_u64(jit_builder, stop, start):
    jit_builder.s_sub_u32(stop[0], stop[0], start[0])
    jit_builder.s_subb_u32(stop[1], stop[1], start[1])


@jit(no_pass=["pass_dse", "pass_dce"])
def mfma_timeline(
    jit_builder: JIT,
    inner_unroll,
    bin_ticks: UINT32,  # pyright: ignore[reportInvalidTypeForm]
    num_bins: UINT32,  # pyright: ignore[reportInvalidTypeForm]
    output: VOID_POINTER,  # pyright: ignore[reportInvalidTypeForm]
):
    """逐 bin 记录纯 MFMA 的 shader cycles 与固定 100 MHz realtime ticks。"""

    mfma_dst, mfma_a, mfma_b = _make_mfma_registers(jit_builder)
    bin_index = jit_builder.gpr("su32", 0)
    record_offset = jit_builder.gpr(
        "su32",
        (jit_builder.blockIdx.x[0] * WAVES_PER_BLOCK + jit_builder.warp_id[0]) * num_bins[0] * 16,
    )
    record_address = jit_builder.gpr("vu32", record_offset)
    record = jit_builder.gpr(4, "vu32", align=4)

    with jit_builder.While(bin_index[0] < num_bins[0]):
        realtime_start = _read_counter(jit_builder, "s_memrealtime")
        cycles_start = _read_counter(jit_builder, "s_memtime")
        realtime_now = jit_builder.gpr(2, "su32", align=2)
        elapsed_low = jit_builder.gpr("su32", 0)

        with jit_builder.While(elapsed_low[0] < bin_ticks[0]):
            for mfma_index in range(inner_unroll):
                _emit_mfma(jit_builder, mfma_dst, mfma_a, mfma_b, mfma_index % 4)
            jit_builder.s_memrealtime(realtime_now)
            jit_builder.s_waitcnt(mod="lgkmcnt(0)")
            jit_builder.s_sub_u32(elapsed_low, realtime_now[0], realtime_start[0])

        cycles_stop = _read_counter(jit_builder, "s_memtime")
        realtime_stop = _read_counter(jit_builder, "s_memrealtime")
        _subtract_u64(jit_builder, cycles_stop, cycles_start)
        _subtract_u64(jit_builder, realtime_stop, realtime_start)

        record[0] = realtime_stop[0]
        record[1] = realtime_stop[1]
        record[2] = cycles_stop[0]
        record[3] = cycles_stop[1]
        with jit_builder.ExecMask(jit_builder.lane_id[0] == 0):
            jit_builder.global_store_dwordx4(record_address, record, output)
        jit_builder.s_waitcnt(mod="vmcnt(0)")
        record_address[0] += 16
        bin_index[0] += 1


@jit(no_pass=["pass_dse", "pass_dce"])
def sustained_load(
    jit_builder: JIT,
    workload,
    inner_unroll,
    valu_per_mfma,
    mfma_burst,
    valu_burst,
    loads_per_group,
    grid_blocks,
    buffer_bytes,
    target_ticks: UINT32,  # pyright: ignore[reportInvalidTypeForm]
    data: VOID_POINTER,  # pyright: ignore[reportInvalidTypeForm]
    output: VOID_POINTER,  # pyright: ignore[reportInvalidTypeForm]
):
    """让所有 resident waves 持续执行指定 workload，直到达到目标墙钟 tick。"""

    assert workload in WORKLOADS
    has_mfma = workload in (
        "mfma",
        "mfma_valu",
        "mfma_valu_burst",
        "bf16_32",
        "fp8_mfma_valu_burst",
        "mfma_exp",
        "mfma_mem",
        "mfma_ds_mem",
    )
    has_memory = workload in ("mfma_mem", "mfma_ds_mem", "mem")
    has_exp = workload in ("mfma_exp", "exp")
    has_ds = workload == "mfma_ds_mem"

    valu_src0 = jit_builder.gpr(4, "vf32", 0.25, align=4)
    valu_src1 = jit_builder.gpr(4, "vf32", 0.5, align=4)
    valu_dst = jit_builder.gpr(4, "vf32", 1.0, align=4)

    if workload == "bf16_32":
        mfma_dst, mfma_a, mfma_b = _make_bf16_32x32_mfma_registers(jit_builder)
    elif workload == "fp8_mfma_valu_burst":
        mfma_dst, mfma_a, mfma_b = _make_fp8_mfma_registers(jit_builder)
    elif has_mfma:
        mfma_dst, mfma_a, mfma_b = _make_mfma_registers(jit_builder)

    if has_exp:
        exp_src = jit_builder.gpr(4, "vf32", 0.25, align=4)
        exp_dst = jit_builder.gpr(4, "vf32", 1.0, align=4)

    if has_memory:
        data_buffer = jit_builder.Buffer(data, buffer_bytes)
        load_values = jit_builder.gpr(MAX_OUTSTANDING_LOADS, 4, "vu32", align=4)
        load_sink = jit_builder.gpr("vu32", 0)
        vector_offset = jit_builder.gpr("vu32", jit_builder.threadIdx.x[0] * 16)
        scalar_offset = jit_builder.gpr("su32", jit_builder.blockIdx.x[0] * THREADS_PER_BLOCK * 16)
        grid_stride_bytes = grid_blocks * THREADS_PER_BLOCK * 16
        buffer_mask = buffer_bytes - 1

    if has_ds:
        lds_base = jit_builder.alloc_lds(THREADS_PER_BLOCK * 16, align=16)
        lds_address = jit_builder.gpr("vu32", jit_builder.threadIdx.x[0] * 16 + lds_base)
        lds_seed = jit_builder.gpr(4, "vu32", 0x12345678, align=4)
        ds_value = jit_builder.gpr(4, "vu32", align=4)
        ds_sink = jit_builder.gpr("vu32", 0)
        jit_builder.ds_write_b128(lds_address, lds_seed)
        jit_builder.s_waitcnt(mod="lgkmcnt(0)")
        jit_builder.s_barrier()

    realtime_start = _read_counter(jit_builder, "s_memrealtime")
    cycles_start = _read_counter(jit_builder, "s_memtime")
    realtime_now = jit_builder.gpr(2, "su32", align=2)
    elapsed_low = jit_builder.gpr("su32", 0)
    batch_count = jit_builder.gpr("su32", 0)

    with jit_builder.While(elapsed_low[0] < target_ticks[0]):
        if workload in ("mfma_mem", "mfma_ds_mem"):
            for pipeline_index in range(inner_unroll // MEM_WAIT_MFMAS):
                for load_index in range(loads_per_group):
                    load_slot = load_index
                    data_buffer.load_dwordx4(
                        load_values[load_slot],
                        vector_offset,
                        scalar_offset,
                        non_temporal=True,
                    )
                    scalar_offset[0] += grid_stride_bytes
                    scalar_offset[0] = scalar_offset[0] & buffer_mask

                if has_ds:
                    jit_builder.ds_read_b128(ds_value, lds_address)

                # 同一 accumulator 的依赖链使每条 MFMA 的 latency 累加，用于保证 wait 前的 cycle 下界。
                delay_slot = pipeline_index % 4
                for mfma_index in range(MEM_WAIT_MFMAS):
                    _emit_mfma(jit_builder, mfma_dst, mfma_a, mfma_b, delay_slot)
                    if has_ds and mfma_index + 1 == DS_WAIT_MFMAS:
                        jit_builder.s_waitcnt(mod="lgkmcnt(0)")
                        jit_builder.v_xor_b32(ds_sink, ds_sink, ds_value[0])

                jit_builder.s_waitcnt(mod="vmcnt(0)")
                for load_slot in range(loads_per_group):
                    jit_builder.v_xor_b32(load_sink, load_sink, load_values[load_slot, 0])

        elif workload == "mem":
            outstanding_loads = 0
            for _ in range(inner_unroll):
                for _ in range(loads_per_group):
                    load_slot = outstanding_loads % MAX_OUTSTANDING_LOADS
                    data_buffer.load_dwordx4(
                        load_values[load_slot],
                        vector_offset,
                        scalar_offset,
                        non_temporal=True,
                    )
                    scalar_offset[0] += grid_stride_bytes
                    scalar_offset[0] = scalar_offset[0] & buffer_mask
                    outstanding_loads += 1
                    if outstanding_loads % MAX_OUTSTANDING_LOADS == 0:
                        jit_builder.s_waitcnt(mod="vmcnt(0)")
                        jit_builder.v_xor_b32(load_sink, load_sink, load_values[load_slot, 0])

            if outstanding_loads % MAX_OUTSTANDING_LOADS:
                jit_builder.s_waitcnt(mod="vmcnt(0)")
                last_slot = (outstanding_loads - 1) % MAX_OUTSTANDING_LOADS
                jit_builder.v_xor_b32(load_sink, load_sink, load_values[last_slot, 0])

        elif workload in (
            "mfma_valu_burst",
            "bf16_32",
            "fp8_mfma_valu_burst",
        ):
            for mfma_index in range(mfma_burst):
                if workload == "bf16_32":
                    _emit_bf16_32x32_mfma(
                        jit_builder, mfma_dst, mfma_a, mfma_b, mfma_index % 4
                    )
                elif workload == "fp8_mfma_valu_burst":
                    _emit_fp8_mfma(
                        jit_builder, mfma_dst, mfma_a, mfma_b, mfma_index % 4
                    )
                else:
                    _emit_mfma(
                        jit_builder, mfma_dst, mfma_a, mfma_b, mfma_index % 4
                    )
            for valu_index in range(valu_burst):
                valu_slot = valu_index % 4
                jit_builder.v_fmac_f32(valu_dst[valu_slot], valu_src0[valu_slot], valu_src1[valu_slot])

        else:
            for group_index in range(inner_unroll):
                slot = group_index % 4
                if has_mfma:
                    _emit_mfma(jit_builder, mfma_dst, mfma_a, mfma_b, slot)

                if workload == "mfma_valu":
                    for valu_index in range(valu_per_mfma):
                        valu_slot = (slot + valu_index) % 4
                        jit_builder.v_add_f32(
                            valu_dst[valu_slot],
                            valu_src0[valu_slot],
                            valu_src1[valu_slot],
                        )
                elif has_exp:
                    jit_builder.v_exp_f32(exp_dst[slot], exp_src[slot])
                elif workload == "valu":
                    jit_builder.v_fmac_f32(valu_dst[slot], valu_src0[slot], valu_src1[slot])

        if has_memory:
            jit_builder.s_waitcnt(mod="vmcnt(0)")

        batch_count[0] += 1
        jit_builder.s_memrealtime(realtime_now)
        jit_builder.s_waitcnt(mod="lgkmcnt(0)")
        jit_builder.s_sub_u32(elapsed_low, realtime_now[0], realtime_start[0])

    cycles_stop = _read_counter(jit_builder, "s_memtime")
    realtime_stop = _read_counter(jit_builder, "s_memrealtime")
    _subtract_u64(jit_builder, cycles_stop, cycles_start)
    _subtract_u64(jit_builder, realtime_stop, realtime_start)

    sink_scalar = jit_builder.gpr("su32", 0)
    sink_component = jit_builder.gpr("su32")
    if has_mfma:
        jit_builder.v_readfirstlane_b32(sink_component, mfma_dst[0, 0])
        jit_builder.s_xor_b32(sink_scalar, sink_scalar, sink_component)
    if workload in (
        "valu",
        "mfma_valu",
        "mfma_valu_burst",
        "bf16_32",
        "fp8_mfma_valu_burst",
    ):
        jit_builder.v_readfirstlane_b32(sink_component, valu_dst[0])
        jit_builder.s_xor_b32(sink_scalar, sink_scalar, sink_component)
    if has_exp:
        jit_builder.v_readfirstlane_b32(sink_component, exp_dst[0])
        jit_builder.s_xor_b32(sink_scalar, sink_scalar, sink_component)
    if has_memory:
        jit_builder.v_readfirstlane_b32(sink_component, load_sink)
        jit_builder.s_xor_b32(sink_scalar, sink_scalar, sink_component)
    if has_ds:
        jit_builder.v_readfirstlane_b32(sink_component, ds_sink)
        jit_builder.s_xor_b32(sink_scalar, sink_scalar, sink_component)
    jit_builder.s_xor_b32(sink_scalar, sink_scalar, data[0])

    hw_id = jit_builder.gpr("su32")
    xcc_id = jit_builder.gpr("su32")
    jit_builder.s_getreg_b32(hw_id, mod="hwreg(HW_REG_HW_ID, 0, 20)")
    jit_builder.s_getreg_b32(xcc_id, mod="hwreg(HW_REG_XCC_ID, 0, 4)")

    record_offset = jit_builder.gpr(
        "su32",
        (jit_builder.blockIdx.x[0] * WAVES_PER_BLOCK + jit_builder.warp_id[0]) * RECORD_BYTES,
    )
    record_address = jit_builder.gpr("vu32", record_offset)
    record = jit_builder.gpr(8, "vu32", align=4)
    record[0] = realtime_stop[0]
    record[1] = realtime_stop[1]
    record[2] = cycles_stop[0]
    record[3] = cycles_stop[1]
    record[4] = hw_id
    record[5] = xcc_id
    record[6] = sink_scalar
    record[7] = batch_count
    with jit_builder.ExecMask(jit_builder.lane_id[0] == 0):
        jit_builder.global_store_dwordx4(record_address, record[0:3], output)
        jit_builder.global_store_dwordx4(record_address, record[4:7], output, mod="offset:16")
    jit_builder.s_waitcnt(mod="vmcnt(0)")


def _read_text(path):
    return Path(path).read_text(encoding="utf-8").strip()


def _find_labeled_sensor(hwmon, family, accepted_labels):
    for label_path in sorted(hwmon.glob(f"{family}*_label")):
        label = _read_text(label_path).lower()
        if label not in accepted_labels:
            continue
        prefix = label_path.name.removesuffix("_label")
        for suffix in ("_average", "_input"):
            value_path = hwmon / f"{prefix}{suffix}"
            if value_path.exists():
                return value_path, label
    raise RuntimeError(f"在 {hwmon} 中找不到 {family} 传感器，候选标签={sorted(accepted_labels)}")


@dataclass(frozen=True)
class SensorPaths:
    bdf: str
    pci_path: Path
    hwmon: Path
    sclk: Path
    power: Path
    junction: Path
    memory_temp: Path
    sclk_label: str
    power_label: str
    max_sclk_mhz: float

    def sample(self):
        return {
            "timestamp_ns": time.monotonic_ns(),
            "sclk_mhz": int(_read_text(self.sclk)) / 1_000_000.0,
            "power_w": int(_read_text(self.power)) / 1_000_000.0,
            "junction_c": int(_read_text(self.junction)) / 1000.0,
            "memory_temp_c": int(_read_text(self.memory_temp)) / 1000.0,
            "gpu_busy_percent": int(_read_text(self.pci_path / "gpu_busy_percent")),
        }


def _discover_sensors(properties):
    bdf = f"{properties.pci_domain_id:04x}:{properties.pci_bus_id:02x}:" f"{properties.pci_device_id:02x}.0"
    pci_path = Path("/sys/bus/pci/devices") / bdf
    hwmon_dirs = list((pci_path / "hwmon").glob("hwmon*"))
    if len(hwmon_dirs) != 1:
        raise RuntimeError(f"{bdf} 应有一个 hwmon 目录，实际为 {hwmon_dirs}")
    hwmon = hwmon_dirs[0]
    sclk, sclk_label = _find_labeled_sensor(hwmon, "freq", {"sclk", "gfxclk"})
    power, power_label = _find_labeled_sensor(hwmon, "power", {"ppt", "socket power"})
    junction, _ = _find_labeled_sensor(hwmon, "temp", {"junction", "hotspot"})
    memory_temp, _ = _find_labeled_sensor(hwmon, "temp", {"mem", "memory", "hbm"})
    dpm_text = _read_text(pci_path / "pp_dpm_sclk")
    dpm_values = [float(value) for value in re.findall(r"(\d+(?:\.\d+)?)\s*Mhz", dpm_text, re.I)]
    if not dpm_values:
        raise RuntimeError(f"无法解析 {pci_path / 'pp_dpm_sclk'}: {dpm_text!r}")
    return SensorPaths(
        bdf=bdf,
        pci_path=pci_path,
        hwmon=hwmon,
        sclk=sclk,
        power=power,
        junction=junction,
        memory_temp=memory_temp,
        sclk_label=sclk_label,
        power_label=power_label,
        max_sclk_mhz=max(dpm_values),
    )


class SensorSampler:
    def __init__(self, sensors, interval_ms):
        self.sensors = sensors
        self.interval_seconds = interval_ms / 1000.0
        self.samples = []
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, name="gpu-sysfs-sampler", daemon=True)

    def _append(self):
        self.samples.append(self.sensors.sample())

    def _run(self):
        self._append()
        self._ready.set()
        while not self._stop.wait(self.interval_seconds):
            self._append()
        self._append()

    def start(self):
        self._thread.start()
        if not self._ready.wait(timeout=2.0):
            raise RuntimeError("sysfs 采样线程启动超时")

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2.0)
        if self._thread.is_alive():
            raise RuntimeError("sysfs 采样线程停止超时")


def _percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    index = fraction * (len(ordered) - 1)
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _summarize(values):
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


def _summarize_sysfs(samples, start_ns, stop_ns):
    during = [sample for sample in samples if start_ns <= sample["timestamp_ns"] <= stop_ns]
    latter_half_ns = start_ns + (stop_ns - start_ns) // 2
    latter_half = [sample for sample in during if sample["timestamp_ns"] >= latter_half_ns]
    return {
        "sample_count": len(during),
        "steady_sample_count": len(latter_half),
        "sclk_mhz": _summarize([sample["sclk_mhz"] for sample in during]),
        "steady_sclk_mhz": _summarize([sample["sclk_mhz"] for sample in latter_half]),
        "power_w": _summarize([sample["power_w"] for sample in during]),
        "steady_power_w": _summarize([sample["power_w"] for sample in latter_half]),
        "junction_c": _summarize([sample["junction_c"] for sample in during]),
        "memory_temp_c": _summarize([sample["memory_temp_c"] for sample in during]),
        "samples": [{**sample, "t_ms": (sample["timestamp_ns"] - start_ns) / 1e6} for sample in samples],
    }


def _decode_mfma_timeline(output, threshold_mhz, low_bins):
    host = output.cpu()
    bins = []
    cumulative_ms = 0.0
    for bin_index in range(host.shape[1]):
        realtime_ticks = [int(value) for value in host[:, bin_index, 0].tolist()]
        shader_cycles = [int(value) for value in host[:, bin_index, 1].tolist()]
        if not all(realtime_ticks):
            raise RuntimeError(f"timeline bin {bin_index} 存在未写回的 wave")
        frequencies = [cycles * 100.0 / ticks for ticks, cycles in zip(realtime_ticks, shader_cycles)]
        duration_ms = statistics.median(realtime_ticks) / REALTIME_TICKS_PER_MS
        start_ms = cumulative_ms
        cumulative_ms += duration_ms
        bins.append(
            {
                "index": bin_index,
                "start_ms": start_ms,
                "end_ms": cumulative_ms,
                "duration_ms": duration_ms,
                "effective_sclk_mhz": _summarize(frequencies),
            }
        )

    first_sustained_low_bin = None
    for bin_index in range(len(bins) - low_bins + 1):
        if all(
            bins[index]["effective_sclk_mhz"]["median"] < threshold_mhz
            for index in range(bin_index, bin_index + low_bins)
        ):
            first_sustained_low_bin = bin_index
            break
    high_duration_ms = cumulative_ms if first_sustained_low_bin is None else bins[first_sustained_low_bin]["start_ms"]
    return {
        "threshold_mhz": threshold_mhz,
        "required_consecutive_low_bins": low_bins,
        "first_sustained_low_bin": first_sustained_low_bin,
        "continuous_high_duration_ms": high_duration_ms,
        "bins": bins,
    }


def _run_mfma_timeline(args, sensors, properties):
    if len(args.durations) != 1:
        raise RuntimeError("timeline 模式要求 --duration-ms 只包含一个值")
    duration_ms = args.durations[0]
    num_bins = round(duration_ms / args.timeline_bin_ms)
    if num_bins <= 0 or not math.isclose(num_bins * args.timeline_bin_ms, duration_ms, abs_tol=1e-9):
        raise RuntimeError("--duration-ms 必须能被 --timeline-bin-ms 整除")
    bin_ticks = round(args.timeline_bin_ms * REALTIME_TICKS_PER_MS)
    grid_blocks = properties.multi_processor_count * args.blocks_per_cu
    record_count = grid_blocks * WAVES_PER_BLOCK

    warmup = torch.zeros((record_count, 1, 2), dtype=torch.uint64, device=f"cuda:{args.device}")
    mfma_timeline([grid_blocks], [THREADS_PER_BLOCK], args.inner_unroll, bin_ticks, 1, warmup.data_ptr())
    torch.cuda.synchronize()
    if args.cooldown_ms:
        time.sleep(args.cooldown_ms / 1000.0)

    results = []
    for repeat in range(args.repeats):
        output = torch.zeros((record_count, num_bins, 2), dtype=torch.uint64, device=f"cuda:{args.device}")
        sampler = SensorSampler(sensors, args.sample_interval_ms)
        sampler.start()
        event_start = torch.cuda.Event(enable_timing=True)
        event_stop = torch.cuda.Event(enable_timing=True)
        host_start_ns = time.monotonic_ns()
        event_start.record()
        mfma_timeline(
            [grid_blocks],
            [THREADS_PER_BLOCK],
            args.inner_unroll,
            bin_ticks,
            num_bins,
            output.data_ptr(),
        )
        event_stop.record()
        event_stop.synchronize()
        host_stop_ns = time.monotonic_ns()
        sampler.stop()
        timeline = _decode_mfma_timeline(output, args.timeline_threshold_mhz, args.timeline_low_bins)
        result = {
            "repeat": repeat,
            "target_duration_ms": duration_ms,
            "event_duration_ms": event_start.elapsed_time(event_stop),
            "wall_duration_ms": (host_stop_ns - host_start_ns) / 1e6,
            "grid_blocks": grid_blocks,
            "blocks_per_cu": args.blocks_per_cu,
            "timeline": timeline,
            "sysfs": _summarize_sysfs(sampler.samples, host_start_ns, host_stop_ns),
        }
        results.append(result)
        print(
            f"timeline repeat={repeat} high_duration={timeline['continuous_high_duration_ms']:.3f} ms "
            f"first_low_bin={timeline['first_sustained_low_bin']} event={result['event_duration_ms']:.3f} ms"
        )
        if args.cooldown_ms:
            time.sleep(args.cooldown_ms / 1000.0)

    return {
        "schema_version": 1,
        "mode": "mfma_timeline",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "device": {
            "torch_device": args.device,
            "name": properties.name,
            "arch": properties.gcnArchName,
            "bdf": sensors.bdf,
            "compute_units": properties.multi_processor_count,
            "max_sclk_mhz": sensors.max_sclk_mhz,
        },
        "config": {
            "duration_ms": duration_ms,
            "timeline_bin_ms": args.timeline_bin_ms,
            "timeline_threshold_mhz": args.timeline_threshold_mhz,
            "timeline_low_bins": args.timeline_low_bins,
            "repeats": args.repeats,
            "blocks_per_cu": args.blocks_per_cu,
            "inner_unroll": args.inner_unroll,
            "sample_interval_ms": args.sample_interval_ms,
            "cooldown_ms": args.cooldown_ms,
        },
        "results": results,
    }


def _workload_grid_blocks(workload, requested_blocks_per_cu, compute_units):
    blocks_per_cu = requested_blocks_per_cu
    if workload == "mfma_exp":
        blocks_per_cu = min(blocks_per_cu, MFMA_EXP_MAX_BLOCKS_PER_CU)
    return blocks_per_cu * compute_units


def _decode_records(
    output,
    target_ticks,
    event_ms,
    workload,
    inner_unroll,
    valu_per_mfma,
    mfma_burst,
    valu_burst,
    loads_per_group,
    expected_cus,
):
    host = output.cpu()
    records = []
    frequencies = []
    unique_cus = set()
    unique_simds = set()
    records_by_simd = {}
    total_batches = 0
    for row in host.tolist():
        realtime_ticks = int(row[0])
        shader_cycles = int(row[1])
        hw_and_xcc = int(row[2])
        sink_and_batches = int(row[3])
        if realtime_ticks == 0:
            continue
        hw_id = hw_and_xcc & 0xFFFFFFFF
        xcc_id = (hw_and_xcc >> 32) & 0xFFFFFFFF
        batch_count = (sink_and_batches >> 32) & 0xFFFFFFFF
        frequency_mhz = shader_cycles * 100.0 / realtime_ticks
        cu_key = (xcc_id, (hw_id >> 13) & 0x3, (hw_id >> 8) & 0xF)
        simd_key = (*cu_key, (hw_id >> 4) & 0x3)
        unique_cus.add(cu_key)
        unique_simds.add(simd_key)
        records_by_simd.setdefault(simd_key, []).append((shader_cycles, batch_count))
        frequencies.append(frequency_mhz)
        total_batches += batch_count
        records.append([realtime_ticks, shader_cycles, hw_id, xcc_id, batch_count])

    if len(records) != output.shape[0]:
        raise RuntimeError(f"只有 {len(records)}/{output.shape[0]} 个 wave 写回了有效计时结果")

    total_groups = total_batches * inner_unroll
    metrics = {}
    if workload in ("mfma", "mfma_valu", "mfma_exp", "mfma_mem", "mfma_ds_mem"):
        matrix_flops_per_mfma = 65_536 if JIT.gfx >= 950 else 8_192
        metrics["matrix_tflops"] = total_groups * matrix_flops_per_mfma / (event_ms * 1e9)
    if workload in ("valu", "mfma_valu"):
        valu_per_group = 1 if workload == "valu" else valu_per_mfma
        flops_per_valu = 2 if workload == "valu" else 1
        metrics["valu_tflops"] = total_groups * valu_per_group * WAVE_SIZE * flops_per_valu / (event_ms * 1e9)
    if workload in ("mfma_exp", "exp"):
        metrics["exp_gops"] = total_groups * WAVE_SIZE / (event_ms * 1e6)
    if workload in (
        "mfma_valu_burst",
        "bf16_32",
        "fp8_mfma_valu_burst",
    ):
        matrix_flops_per_mfma = (
            32_768
            if workload == "fp8_mfma_valu_burst"
            else 16_384
            if workload == "bf16_32"
            else 65_536
            if JIT.gfx >= 950
            else 8_192
        )
        total_mfmas = total_batches * mfma_burst
        metrics["matrix_tflops"] = total_mfmas * matrix_flops_per_mfma / (event_ms * 1e9)
        if workload in (
            "mfma_valu_burst",
            "bf16_32",
            "fp8_mfma_valu_burst",
        ):
            total_valu = total_batches * valu_burst
            metrics["valu_tflops"] = total_valu * WAVE_SIZE * 2 / (event_ms * 1e9)
    if workload == "mem":
        memory_loads = total_groups * loads_per_group
        metrics["memory_tb_per_s"] = memory_loads * WAVE_SIZE * 16 / (event_ms * 1e9)
    elif workload in ("mfma_mem", "mfma_ds_mem"):
        memory_loads = total_groups * loads_per_group // MEM_WAIT_MFMAS
        metrics["memory_tb_per_s"] = memory_loads * WAVE_SIZE * 16 / (event_ms * 1e9)
    if workload == "mfma_ds_mem":
        ds_reads = total_groups // MEM_WAIT_MFMAS
        metrics["lds_tb_per_s"] = ds_reads * WAVE_SIZE * 16 / (event_ms * 1e9)

    mfmas_per_batch = None
    if workload in ("mfma", "mfma_valu", "mfma_exp", "mfma_mem", "mfma_ds_mem"):
        mfmas_per_batch = inner_unroll
    elif workload in (
        "mfma_valu_burst",
        "bf16_32",
        "fp8_mfma_valu_burst",
    ):
        mfmas_per_batch = mfma_burst
    simd_cycles_per_mfma = None
    if mfmas_per_batch is not None:
        simd_cycles_per_mfma = _summarize(
            [
                statistics.median(shader_cycles for shader_cycles, _ in simd_records)
                / sum(batch_count * mfmas_per_batch for _, batch_count in simd_records)
                for simd_records in records_by_simd.values()
            ]
        )

    target_duration_ms = target_ticks / REALTIME_TICKS_PER_MS
    hardware_duration_ms = statistics.median(record[0] for record in records) / REALTIME_TICKS_PER_MS
    if len(unique_cus) != expected_cus:
        raise RuntimeError(f"workload 只覆盖了 {len(unique_cus)}/{expected_cus} 个 CU")
    event_excess_ms = event_ms - hardware_duration_ms
    if event_excess_ms > 10.0 and event_ms > hardware_duration_ms * 1.5:
        raise RuntimeError(
            f"整次 launch {event_ms:.3f} ms 明显长于每 wave {hardware_duration_ms:.3f} ms，"
            "workgroup 可能发生了分批调度"
        )
    return {
        "target_ticks": target_ticks,
        "hardware_duration_ms": hardware_duration_ms,
        "target_error_percent": 100.0 * (hardware_duration_ms - target_duration_ms) / target_duration_ms,
        "event_overhead_ms": event_ms - hardware_duration_ms,
        "event_overhead_percent": 100.0 * (event_ms - hardware_duration_ms) / hardware_duration_ms,
        "effective_sclk_mhz": _summarize(frequencies),
        "simd_cycles_per_mfma": simd_cycles_per_mfma,
        "unique_cu_count": len(unique_cus),
        "unique_simd_count": len(unique_simds),
        "total_batches": total_batches,
        "throughput": metrics,
        "records": records,
    }


def _run_once(
    workload,
    duration_ms,
    args,
    sensors,
    grid_blocks,
    data,
    output,
    valu_burst=None,
):
    if valu_burst is None:
        valu_burst = args.valu_burst
    target_ticks = round(duration_ms * REALTIME_TICKS_PER_MS)
    output.zero_()
    sampler = SensorSampler(sensors, args.sample_interval_ms)
    sampler.start()

    event_start = torch.cuda.Event(enable_timing=True)
    event_stop = torch.cuda.Event(enable_timing=True)
    host_start_ns = time.monotonic_ns()
    event_start.record()
    sustained_load(
        [grid_blocks],
        [THREADS_PER_BLOCK],
        workload,
        args.inner_unroll,
        args.valu_per_mfma,
        args.mfma_burst,
        valu_burst,
        args.loads_per_group,
        grid_blocks,
        data.numel(),
        target_ticks,
        data.data_ptr(),
        output.data_ptr(),
    )
    event_stop.record()
    event_stop.synchronize()
    host_stop_ns = time.monotonic_ns()
    sampler.stop()

    event_ms = event_start.elapsed_time(event_stop)
    hardware = _decode_records(
        output,
        target_ticks,
        event_ms,
        workload,
        args.inner_unroll,
        args.valu_per_mfma,
        args.mfma_burst,
        valu_burst,
        args.loads_per_group,
        torch.cuda.get_device_properties(args.device).multi_processor_count,
    )
    effective_mhz = hardware["effective_sclk_mhz"]["median"]
    return {
        "workload": workload,
        "description": WORKLOADS[workload],
        "target_duration_ms": duration_ms,
        "grid_blocks": grid_blocks,
        "blocks_per_cu": grid_blocks // torch.cuda.get_device_properties(args.device).multi_processor_count,
        "mfma_burst": args.mfma_burst if workload in SPECIAL_WORKLOADS else None,
        "valu_burst": valu_burst if workload in SPECIAL_WORKLOADS else None,
        "event_duration_ms": event_ms,
        "wall_duration_ms": (host_stop_ns - host_start_ns) / 1e6,
        "effective_sclk_drop_mhz": sensors.max_sclk_mhz - effective_mhz,
        "effective_sclk_drop_percent": 100.0 * (sensors.max_sclk_mhz - effective_mhz) / sensors.max_sclk_mhz,
        "hardware_timer": hardware,
        "sysfs": _summarize_sysfs(sampler.samples, host_start_ns, host_stop_ns),
    }


def _run_dispatch_train(
    workload,
    duration_ms,
    args,
    sensors,
    grid_blocks,
    data,
    output,
):
    """连续提交多个短 kernel，检查 kernel 边界是否重置 MFMA 降频计时。"""

    target_ticks = round(duration_ms * REALTIME_TICKS_PER_MS)
    output.zero_()
    sampler = SensorSampler(sensors, args.sample_interval_ms)
    sampler.start()

    events = []
    host_start_ns = time.monotonic_ns()
    for dispatch_index in range(args.dispatch_train_count):
        event_start = torch.cuda.Event(enable_timing=True)
        event_stop = torch.cuda.Event(enable_timing=True)
        event_start.record()
        sustained_load(
            [grid_blocks],
            [THREADS_PER_BLOCK],
            workload,
            args.inner_unroll,
            args.valu_per_mfma,
            args.mfma_burst,
            args.valu_burst,
            args.loads_per_group,
            grid_blocks,
            data.numel(),
            target_ticks,
            data.data_ptr(),
            output[dispatch_index].data_ptr(),
        )
        event_stop.record()
        events.append((event_start, event_stop))
        if args.dispatch_gap_ms and dispatch_index + 1 < args.dispatch_train_count:
            event_stop.synchronize()
            time.sleep(args.dispatch_gap_ms / 1000.0)

    events[-1][1].synchronize()
    host_stop_ns = time.monotonic_ns()
    sampler.stop()

    expected_cus = torch.cuda.get_device_properties(args.device).multi_processor_count
    dispatches = []
    frequencies = []
    for dispatch_index, (event_start, event_stop) in enumerate(events):
        event_ms = event_start.elapsed_time(event_stop)
        hardware = _decode_records(
            output[dispatch_index],
            target_ticks,
            event_ms,
            workload,
            args.inner_unroll,
            args.valu_per_mfma,
            args.mfma_burst,
            args.valu_burst,
            args.loads_per_group,
            expected_cus,
        )
        frequencies.extend(
            record[1] * 100.0 / record[0] for record in hardware["records"]
        )
        dispatches.append(
            {
                "index": dispatch_index,
                "event_duration_ms": event_ms,
                "hardware_timer": hardware,
            }
        )

    effective_sclk = _summarize(frequencies)
    dispatch_densities = [
        dispatch["hardware_timer"]["simd_cycles_per_mfma"]["median"]
        for dispatch in dispatches
        if dispatch["hardware_timer"]["simd_cycles_per_mfma"] is not None
    ]
    return {
        "workload": workload,
        "description": f"{WORKLOADS[workload]}，连续短 dispatch train",
        "target_duration_ms": duration_ms * args.dispatch_train_count,
        "per_dispatch_target_ms": duration_ms,
        "dispatch_train_count": args.dispatch_train_count,
        "dispatch_gap_ms": args.dispatch_gap_ms,
        "grid_blocks": grid_blocks,
        "blocks_per_cu": grid_blocks // expected_cus,
        "mfma_burst": args.mfma_burst if workload.endswith("_burst") else None,
        "valu_burst": args.valu_burst if workload.endswith("_burst") else None,
        "event_duration_ms": sum(dispatch["event_duration_ms"] for dispatch in dispatches),
        "wall_duration_ms": (host_stop_ns - host_start_ns) / 1e6,
        "effective_sclk_drop_mhz": sensors.max_sclk_mhz - effective_sclk["median"],
        "effective_sclk_drop_percent": 100.0
        * (sensors.max_sclk_mhz - effective_sclk["median"])
        / sensors.max_sclk_mhz,
        "hardware_timer": {
            "effective_sclk_mhz": effective_sclk,
            "simd_cycles_per_mfma": _summarize(dispatch_densities),
            "unique_cu_count": expected_cus,
            "throughput": {},
        },
        "dispatches": dispatches,
        "sysfs": _summarize_sysfs(sampler.samples, host_start_ns, host_stop_ns),
    }


def _parse_csv(value, converter):
    return [converter(item.strip()) for item in value.split(",") if item.strip()]


def _classify_mfma_density(mfma_opcodes, cycles_per_mfma, stall_per_mfma):
    if mfma_opcodes == ["v_mfma_f32_32x32x16_fp8_fp8"]:
        throttle_max = FP8_DENSITY_THROTTLE_MAX_CYCLES_PER_MFMA
        high_frequency_min = FP8_DENSITY_HIGH_FREQUENCY_MIN_CYCLES_PER_MFMA
    elif mfma_opcodes == ["v_mfma_f32_32x32x8_bf16"]:
        throttle_max = BF16_DENSITY_THROTTLE_MAX_CYCLES_PER_MFMA
        high_frequency_min = BF16_DENSITY_HIGH_FREQUENCY_MIN_CYCLES_PER_MFMA
        if (
            cycles_per_mfma <= throttle_max
            and stall_per_mfma >= cycles_per_mfma
        ):
            return "stall_diluted_candidate"
    else:
        return "unclassified_opcode"
    if cycles_per_mfma <= throttle_max:
        return "throttle_side"
    if cycles_per_mfma >= high_frequency_min:
        return "high_frequency_side"
    return "boundary"


def _parse_labeled_paths(value):
    labeled_paths = []
    for item in _parse_csv(value, str):
        if "=" not in item:
            raise ValueError(f"ATT trace 必须使用 label=path 格式: {item!r}")
        label, path = item.split("=", 1)
        if not label or not path:
            raise ValueError(f"ATT trace 必须使用非空 label=path 格式: {item!r}")
        labeled_paths.append((label, Path(path)))
    return labeled_paths


def _selected_valu_bursts(workload, args):
    if workload == "fp8_mfma_valu_burst" and args.fp8_valu_scan:
        return args.fp8_valu_bursts
    if workload == "bf16_32" and args.bf16_valu_scan:
        return args.bf16_valu_bursts
    return [args.valu_burst]


def _att_instruction_class(instruction):
    opcode = instruction.strip().split()[0]
    if "mfma" in opcode:
        return "mfma"
    if opcode.startswith(("v_pk_", "v_dot", "v_wmma")):
        return "packed_valu"
    if opcode.startswith(("v_exp", "v_rcp")):
        return "exp_rcp"
    if opcode.startswith("v_"):
        return "scalar_valu"
    if opcode.startswith("s_waitcnt"):
        return "waitcnt"
    if opcode.startswith("s_barrier"):
        return "barrier"
    return "other"


def _analyze_att_mfma_density(label, root):
    ui_directories = sorted(root.glob("ui_output_agent_*"))
    stats_paths = sorted(root.glob("stats_ui_output_agent_*.csv"))
    if len(ui_directories) != 1 or len(stats_paths) != 1:
        raise RuntimeError(
            f"{root} 下应各有一个 ui_output_agent_* 和 stats_ui_output_agent_*.csv"
        )
    ui_directory = ui_directories[0]
    capture_log = root.parent / f"{root.name}-capture.log"
    trace_level_complete = None
    if capture_log.is_file():
        capture_text = capture_log.read_text(encoding="utf-8", errors="replace")
        incomplete_markers = re.findall(
            r"Stitch Incomplete|Wave incomplete|trace was cutoff|"
            r"parser could not fully match",
            capture_text,
            flags=re.IGNORECASE,
        )
        if incomplete_markers:
            raise RuntimeError(
                f"{capture_log} 报告 ATT 整体不完整: {sorted(set(incomplete_markers))}"
            )
        trace_level_complete = True
    code = json.loads((ui_directory / "code.json").read_text(encoding="utf-8"))["code"]
    mfma_opcodes = sorted(
        {
            row[0].strip().split()[0]
            for row in code
            if _att_instruction_class(row[0]) == "mfma"
        }
    )
    mfma_instruction_ids = {
        instruction_id
        for instruction_id, row in enumerate(code)
        if _att_instruction_class(row[0]) == "mfma"
    }
    if not mfma_instruction_ids:
        raise RuntimeError(f"{ui_directory / 'code.json'} 中没有 MFMA")

    max_consecutive_mfma = 0
    current_mfma_run = 0
    for row in code:
        if _att_instruction_class(row[0]) == "mfma":
            current_mfma_run += 1
            max_consecutive_mfma = max(max_consecutive_mfma, current_mfma_run)
        else:
            current_mfma_run = 0

    simds = {}
    wave_count = 0
    for wave_path in sorted(ui_directory.glob("se*.json")):
        with wave_path.open(encoding="utf-8") as handle:
            header = handle.read(512)
        metadata_match = re.search(
            r'^\{"duration":(\d+),.*?"num_insts":(\d+),"num_stitched":(\d+),'
            r'"wave":\{"begin":(\d+),"cu":\d+,"end":(\d+)',
            header,
        )
        simd_match = re.search(r"_sm(\d+)_sl\d+_wv\d+\.json$", wave_path.name)
        if metadata_match is None or simd_match is None:
            raise RuntimeError(f"无法从 {wave_path} 的文件头或文件名解析 wave 元数据")
        duration, num_insts, num_stitched, begin, end = map(
            int, metadata_match.groups()
        )
        if num_insts != num_stitched:
            raise RuntimeError(
                f"{wave_path} 不完整: {num_stitched}/{num_insts}"
            )
        simd_id = int(simd_match.group(1))
        simd = simds.setdefault(
            simd_id,
            {
                "begin": begin,
                "end": end,
                "active_cycles": 0,
                "dynamic_instructions": 0,
            },
        )
        simd["begin"] = min(simd["begin"], begin)
        simd["end"] = max(simd["end"], end)
        simd["active_cycles"] += duration
        simd["dynamic_instructions"] += num_insts
        wave_count += 1

    instruction_mix = {
        name: 0
        for name in (
            "mfma",
            "scalar_valu",
            "packed_valu",
            "exp_rcp",
            "waitcnt",
            "barrier",
            "other",
        )
    }
    total_latency = 0
    total_stall = 0
    total_idle = 0
    with stats_paths[0].open(newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                hitcount = int(row["Hitcount"])
                latency = int(row["Latency"])
                stall = int(row["Stall"])
                idle = int(row["Idle"])
            except ValueError:
                continue
            instruction_mix[_att_instruction_class(row["Instruction"])] += hitcount
            total_latency += latency
            total_stall += stall
            total_idle += idle
    if instruction_mix["mfma"] == 0:
        raise RuntimeError(f"{stats_paths[0]} 中没有 MFMA hit")
    mfma_count = instruction_mix["mfma"]
    if not simds or mfma_count % len(simds):
        raise RuntimeError(
            f"MFMA hit {mfma_count} 不能均分到 {len(simds)} 个采样 SIMD"
        )
    mfma_per_simd = mfma_count // len(simds)
    cycles_per_mfma_by_simd = [
        (values["end"] - values["begin"]) / mfma_per_simd
        for _, values in sorted(simds.items())
    ]
    cycles_per_mfma = statistics.median(cycles_per_mfma_by_simd)
    stall_per_mfma = total_stall / mfma_count
    idle_per_mfma = total_idle / mfma_count
    latency_per_mfma = total_latency / mfma_count
    issue_density_by_simd = [
        values["dynamic_instructions"] / values["active_cycles"]
        for _, values in sorted(simds.items())
    ]
    issue_density = statistics.median(issue_density_by_simd)
    return {
        "label": label,
        "root": str(root),
        "wave_count": wave_count,
        "all_waves_complete": True,
        "capture_log": str(capture_log) if capture_log.is_file() else None,
        "trace_level_complete": trace_level_complete,
        "simds": sorted(simds),
        "cycles_per_mfma_by_simd": cycles_per_mfma_by_simd,
        "cycles_per_mfma_median": cycles_per_mfma,
        "latency_per_mfma": latency_per_mfma,
        "stall_per_mfma": stall_per_mfma,
        "idle_per_mfma": idle_per_mfma,
        "instruction_issue_density": issue_density,
        "instruction_issue_density_by_simd": issue_density_by_simd,
        "mfma_opcodes": mfma_opcodes,
        "density_classification": _classify_mfma_density(
            mfma_opcodes, cycles_per_mfma, stall_per_mfma
        ),
        "static_max_consecutive_mfma": max_consecutive_mfma,
        "instruction_mix": instruction_mix,
        "scalar_valu_per_mfma": instruction_mix["scalar_valu"] / mfma_count,
        "all_valu_per_mfma": (
            instruction_mix["scalar_valu"]
            + instruction_mix["packed_valu"]
            + instruction_mix["exp_rcp"]
        )
        / mfma_count,
    }


def _run_att_validation(args):
    results = [
        _analyze_att_mfma_density(label, root) for label, root in args.att_trace_paths
    ]
    print(
        "ATT label                         cycles/MFMA stall/MFMA issue-density "
        "class                    scalar-VALU/MFMA"
    )
    for result in results:
        print(
            f"{result['label']:32s} {result['cycles_per_mfma_median']:11.3f} "
            f"{result['stall_per_mfma']:10.3f} "
            f"{result['instruction_issue_density']:13.6f} "
            f"{result['density_classification']:24s} "
            f"{result['scalar_valu_per_mfma']:16.3f}"
        )
    return {
        "schema_version": 1,
        "mode": "att_mfma_density_validation",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "thresholds": {
            "fp8_32x32x16": {
                "throttle_side_max_cycles_per_mfma": FP8_DENSITY_THROTTLE_MAX_CYCLES_PER_MFMA,
                "high_frequency_side_min_cycles_per_mfma": FP8_DENSITY_HIGH_FREQUENCY_MIN_CYCLES_PER_MFMA,
            },
            "bf16_32x32x8": {
                "throttle_side_max_cycles_per_mfma": BF16_DENSITY_THROTTLE_MAX_CYCLES_PER_MFMA,
                "high_frequency_side_min_cycles_per_mfma": BF16_DENSITY_HIGH_FREQUENCY_MIN_CYCLES_PER_MFMA,
                "stall_diluted_candidate_requires_stall_per_mfma_ge_cycles_per_mfma": True,
            },
        },
        "results": results,
    }


def _validate_args(parser, args):
    try:
        args.durations = _parse_csv(args.duration_ms, float)
    except ValueError as error:
        parser.error(f"--duration-ms 格式错误: {error}")
    if not args.durations or any(duration < 1.0 or duration > MAX_DURATION_MS for duration in args.durations):
        parser.error(f"每个 duration 必须在 1..{MAX_DURATION_MS:g} ms 之间")

    args.selected_workloads = _parse_csv(args.workloads, str)
    if args.selected_workloads == ["all"]:
        args.selected_workloads = list(DEFAULT_WORKLOADS)
    unknown = [name for name in args.selected_workloads if name not in WORKLOADS]
    if unknown:
        parser.error(f"未知 workload: {', '.join(unknown)}")
    try:
        args.fp8_valu_bursts = _parse_csv(args.fp8_valu_scan, int)
    except ValueError as error:
        parser.error(f"--fp8-valu-scan 格式错误: {error}")
    if not args.fp8_valu_bursts:
        args.fp8_valu_bursts = [args.valu_burst]
    if any(value < 0 for value in args.fp8_valu_bursts):
        parser.error("--fp8-valu-scan 中的值不能为负数")
    if args.fp8_valu_scan and args.selected_workloads != ["fp8_mfma_valu_burst"]:
        parser.error("--fp8-valu-scan 只支持单独选择 fp8_mfma_valu_burst")
    try:
        args.bf16_valu_bursts = _parse_csv(args.bf16_valu_scan, int)
    except ValueError as error:
        parser.error(f"--bf16-valu-scan 格式错误: {error}")
    if not args.bf16_valu_bursts:
        args.bf16_valu_bursts = [args.valu_burst]
    if any(value < 0 for value in args.bf16_valu_bursts):
        parser.error("--bf16-valu-scan 中的值不能为负数")
    if args.bf16_valu_scan and args.selected_workloads != ["bf16_32"]:
        parser.error("--bf16-valu-scan 只支持单独选择 bf16_32")
    if args.fp8_valu_scan and args.bf16_valu_scan:
        parser.error("--fp8-valu-scan 与 --bf16-valu-scan 不能同时使用")
    try:
        args.att_trace_paths = _parse_labeled_paths(args.att_traces)
    except ValueError as error:
        parser.error(str(error))
    if args.att_trace_paths and (args.fp8_valu_scan or args.bf16_valu_scan):
        parser.error("--att-traces 与 VALU scan 不能同时使用")
    if args.inner_unroll <= 0:
        parser.error("--inner-unroll 必须为正数")
    if any(name in ("mfma_mem", "mfma_ds_mem") for name in args.selected_workloads):
        if args.inner_unroll % MEM_WAIT_MFMAS:
            parser.error(f"包含延迟消费内存的 workload 时，--inner-unroll 必须为 {MEM_WAIT_MFMAS} 的倍数")
    if "mem" in args.selected_workloads and args.inner_unroll % MAX_OUTSTANDING_LOADS:
        parser.error(f"纯 mem workload 要求 --inner-unroll 为 {MAX_OUTSTANDING_LOADS} 的倍数")
    if not 0 <= args.valu_per_mfma <= 4:
        parser.error("--valu-per-mfma 必须在 0..4 之间")
    if args.mfma_burst <= 0 or args.valu_burst < 0:
        parser.error("--mfma-burst 必须为正数，--valu-burst 不能为负数")
    if not 1 <= args.loads_per_group <= 4:
        parser.error("--loads-per-group 必须在 1..4 之间")
    if args.blocks_per_cu <= 0 or args.repeats <= 0:
        parser.error("--blocks-per-cu 和 --repeats 必须为正数")
    if args.dispatch_train_count <= 0 or args.dispatch_gap_ms < 0:
        parser.error("--dispatch-train-count 必须为正数，--dispatch-gap-ms 不能为负数")
    if args.dispatch_train_count == 1 and args.dispatch_gap_ms:
        parser.error("--dispatch-gap-ms 只在 --dispatch-train-count > 1 时有效")
    if args.dispatch_train_count > 1 and (args.fp8_valu_scan or args.bf16_valu_scan):
        parser.error("dispatch train 与 VALU scan 不能同时使用")
    if args.sample_interval_ms <= 0 or args.cooldown_ms < 0:
        parser.error("采样间隔必须为正数，冷却时间不能为负数")
    buffer_bytes = args.buffer_mib * 1024 * 1024
    if args.buffer_mib <= 0 or buffer_bytes & (buffer_bytes - 1):
        parser.error("--buffer-mib 换算为字节后必须是 2 的幂")


def _print_result(result, max_sclk_mhz):
    hardware = result["hardware_timer"]
    frequency = hardware["effective_sclk_mhz"]
    throughput = hardware["throughput"]
    throughput_text = " ".join(f"{name}={value:.3f}" for name, value in throughput.items())
    density = hardware.get("simd_cycles_per_mfma")
    density_text = "" if density is None else f" cycles/MFMA={density['median']:.3f}"
    print(
        f"{result['workload']:10s} target={result['target_duration_ms']:8.3f} ms "
        f"event={result['event_duration_ms']:8.3f} ms "
        f"SCLK={frequency['median']:7.1f} MHz "
        f"drop={result['effective_sclk_drop_percent']:6.2f}%/{max_sclk_mhz:.0f}MHz "
        f"CU={hardware['unique_cu_count']:3d}{density_text} {throughput_text}"
    )
    for dispatch in result.get("dispatches", []):
        frequency = dispatch["hardware_timer"]["effective_sclk_mhz"]
        density = dispatch["hardware_timer"]["simd_cycles_per_mfma"]
        density_text = "" if density is None else f" cycles/MFMA={density['median']:.3f}"
        print(
            f"  dispatch={dispatch['index']:3d} event={dispatch['event_duration_ms']:8.3f} ms "
            f"SCLK={frequency['median']:7.1f} MHz "
            f"[{frequency['min']:.1f}, {frequency['max']:.1f}]{density_text}"
        )


def _print_comparison(results):
    print("\n按 target/repeat 对比整段有效 SCLK（相对 DPM 最高档）：")
    print("target(ms) repeat workload      effective(MHz) drop(MHz) drop(%) steady_sysfs(MHz) power(W)")
    for result in results:
        steady_sclk = result["sysfs"]["steady_sclk_mhz"]
        steady_power = result["sysfs"]["steady_power_w"]
        steady_sclk_text = "-" if steady_sclk is None else f"{steady_sclk['median']:.1f}"
        steady_power_text = "-" if steady_power is None else f"{steady_power['median']:.1f}"
        print(
            f"{result['target_duration_ms']:10.3f} {result['repeat']:6d} "
            f"{result['workload']:12s} "
            f"{result['hardware_timer']['effective_sclk_mhz']['median']:14.1f} "
            f"{result['effective_sclk_drop_mhz']:9.1f} "
            f"{result['effective_sclk_drop_percent']:7.2f} "
            f"{steady_sclk_text:17s} {steady_power_text:8s}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0, help="当前进程可见的 torch/HIP 设备编号")
    parser.add_argument(
        "--workloads",
        default="all",
        help=f"逗号分隔；all={','.join(DEFAULT_WORKLOADS)}；特殊扫描模式={','.join(set(WORKLOADS) - set(DEFAULT_WORKLOADS))}",
    )
    parser.add_argument(
        "--duration-ms",
        default="1,10,100,1000,3000",
        help="逗号分隔的单次 kernel 时长，范围 1..30000 ms",
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--blocks-per-cu", type=int, default=4)
    parser.add_argument("--inner-unroll", type=int, default=64)
    parser.add_argument("--valu-per-mfma", type=int, default=3)
    parser.add_argument("--mfma-burst", type=int, default=64)
    parser.add_argument("--valu-burst", type=int, default=0)
    parser.add_argument(
        "--fp8-valu-scan",
        default="",
        help="逗号分隔的 VALU burst；用于一次进程扫描 fp8_mfma_valu_burst 密度边界",
    )
    parser.add_argument(
        "--bf16-valu-scan",
        default="",
        help="逗号分隔的 VALU burst；用于一次进程扫描 bf16_32 密度边界",
    )
    parser.add_argument(
        "--att-traces",
        default="",
        help="逗号分隔的 label=trace-root；离线计算生产 kernel 的每 SIMD cycles/MFMA",
    )
    parser.add_argument(
        "--dispatch-train-count",
        type=int,
        default=1,
        help="每个样本连续提交的短 kernel 数；大于 1 时保留每个 dispatch 的双时钟频率",
    )
    parser.add_argument(
        "--dispatch-gap-ms",
        type=float,
        default=0.0,
        help="dispatch train 中同步后插入的 host idle；0 表示同一 stream 无间隙提交",
    )
    parser.add_argument("--loads-per-group", type=int, default=1)
    parser.add_argument("--buffer-mib", type=int, default=512)
    parser.add_argument("--sample-interval-ms", type=float, default=10.0)
    parser.add_argument("--cooldown-ms", type=float, default=1000.0)
    parser.add_argument("--timeline-bin-ms", type=float, default=0.0, help="大于 0 时启用纯 MFMA kernel 内时间线")
    parser.add_argument("--timeline-threshold-mhz", type=float, default=1800.0)
    parser.add_argument("--timeline-low-bins", type=int, default=3)
    parser.add_argument("--allow-busy", action="store_true", help="允许启动时目标 GPU 非空闲")
    parser.add_argument("--json", help="保存完整结果 JSON")
    args = parser.parse_args()
    _validate_args(parser, args)

    if args.att_trace_paths:
        payload = _run_att_validation(args)
        if args.json:
            output_path = Path(args.json)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(f"完整结果已写入 {output_path}")
        return

    torch.cuda.set_device(args.device)
    properties = torch.cuda.get_device_properties(args.device)
    arch = properties.gcnArchName
    if not (arch.startswith("gfx94") or arch.startswith("gfx950")):
        raise RuntimeError(f"当前只支持 gfx94x/gfx950，检测到 {arch}")

    sensors = _discover_sensors(properties)
    initial_state = sensors.sample()
    dpm_mode = _read_text(sensors.pci_path / "power_dpm_force_performance_level")
    if initial_state["gpu_busy_percent"] != 0 and not args.allow_busy:
        raise RuntimeError(
            f"目标 GPU {sensors.bdf} 当前 busy={initial_state['gpu_busy_percent']}%，"
            "请使用空闲卡或显式传入 --allow-busy"
        )
    if dpm_mode != "auto":
        raise RuntimeError(f"目标 GPU DPM 模式为 {dpm_mode!r}，本测试要求 auto")

    if args.timeline_bin_ms > 0:
        if args.timeline_bin_ms < 0.1 or args.timeline_low_bins <= 0:
            parser.error("timeline bin 至少 0.1 ms，连续低频 bin 数必须为正数")
        payload = _run_mfma_timeline(args, sensors, properties)
        if args.json:
            output_path = Path(args.json)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(f"完整结果已写入 {output_path}")
        return

    max_grid_blocks = properties.multi_processor_count * args.blocks_per_cu
    record_count = max_grid_blocks * WAVES_PER_BLOCK
    buffer_bytes = args.buffer_mib * 1024 * 1024
    needs_memory = any(name in ("mem", "mfma_mem", "mfma_ds_mem") for name in args.selected_workloads)
    allocation_bytes = buffer_bytes if needs_memory else 4096
    data = torch.zeros(allocation_bytes, dtype=torch.uint8, device=f"cuda:{args.device}")
    output_shape = (record_count, RECORD_BYTES // 8)
    if args.dispatch_train_count > 1:
        output_shape = (args.dispatch_train_count, *output_shape)
    output = torch.zeros(output_shape, dtype=torch.uint64, device=f"cuda:{args.device}")
    torch.cuda.synchronize()

    print(
        f"GPU={args.device} {properties.name} arch={arch} BDF={sensors.bdf} "
        f"CU={properties.multi_processor_count} max_grid={max_grid_blocks}x{THREADS_PER_BLOCK} "
        f"max_sclk={sensors.max_sclk_mhz:.0f}MHz DPM={dpm_mode}"
    )
    print("预编译并短暂预热所有 workload...")
    for workload in args.selected_workloads:
        grid_blocks = _workload_grid_blocks(workload, args.blocks_per_cu, properties.multi_processor_count)
        warmup_output = output[0] if args.dispatch_train_count > 1 else output
        valu_bursts = _selected_valu_bursts(workload, args)
        for valu_burst in valu_bursts:
            sustained_load(
                [grid_blocks],
                [THREADS_PER_BLOCK],
                workload,
                args.inner_unroll,
                args.valu_per_mfma,
                args.mfma_burst,
                valu_burst,
                args.loads_per_group,
                grid_blocks,
                data.numel(),
                1000,
                data.data_ptr(),
                warmup_output.data_ptr(),
            )
    torch.cuda.synchronize()
    if args.cooldown_ms:
        time.sleep(args.cooldown_ms / 1000.0)

    results = []
    for duration_ms in args.durations:
        for repeat in range(args.repeats):
            for workload in args.selected_workloads:
                grid_blocks = _workload_grid_blocks(workload, args.blocks_per_cu, properties.multi_processor_count)
                valu_bursts = _selected_valu_bursts(workload, args)
                for valu_burst in valu_bursts:
                    if args.dispatch_train_count > 1:
                        workload_output = output[:, : grid_blocks * WAVES_PER_BLOCK]
                        result = _run_dispatch_train(
                            workload,
                            duration_ms,
                            args,
                            sensors,
                            grid_blocks,
                            data,
                            workload_output,
                        )
                    else:
                        workload_output = output[: grid_blocks * WAVES_PER_BLOCK]
                        result = _run_once(
                            workload,
                            duration_ms,
                            args,
                            sensors,
                            grid_blocks,
                            data,
                            workload_output,
                            valu_burst=valu_burst,
                        )
                    result["repeat"] = repeat
                    results.append(result)
                    _print_result(result, sensors.max_sclk_mhz)
                    if args.cooldown_ms:
                        time.sleep(args.cooldown_ms / 1000.0)

    _print_comparison(results)

    payload = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "device": {
            "torch_device": args.device,
            "name": properties.name,
            "arch": arch,
            "bdf": sensors.bdf,
            "compute_units": properties.multi_processor_count,
            "max_sclk_mhz": sensors.max_sclk_mhz,
            "dpm_mode": dpm_mode,
            "initial_state": initial_state,
            "sensor_paths": {
                "hwmon": str(sensors.hwmon),
                "sclk": str(sensors.sclk),
                "power": str(sensors.power),
                "junction": str(sensors.junction),
                "memory_temp": str(sensors.memory_temp),
            },
        },
        "config": {
            "workloads": args.selected_workloads,
            "durations_ms": args.durations,
            "repeats": args.repeats,
            "blocks_per_cu": args.blocks_per_cu,
            "mfma_exp_max_blocks_per_cu": MFMA_EXP_MAX_BLOCKS_PER_CU,
            "threads_per_block": THREADS_PER_BLOCK,
            "inner_unroll": args.inner_unroll,
            "valu_per_mfma": args.valu_per_mfma,
            "mfma_burst": args.mfma_burst,
            "valu_burst": args.valu_burst,
            "fp8_valu_scan": args.fp8_valu_bursts if args.fp8_valu_scan else None,
            "bf16_valu_scan": args.bf16_valu_bursts if args.bf16_valu_scan else None,
            "dispatch_train_count": args.dispatch_train_count,
            "dispatch_gap_ms": args.dispatch_gap_ms,
            "loads_per_group": args.loads_per_group,
            "mfma_latency_cycles": MFMA_LATENCY_CYCLES,
            "mem_wait_min_cycles": MEM_WAIT_MIN_CYCLES,
            "mem_wait_mfmas": MEM_WAIT_MFMAS,
            "ds_wait_min_cycles": DS_WAIT_MIN_CYCLES,
            "ds_wait_mfmas": DS_WAIT_MFMAS,
            "buffer_mib": args.buffer_mib,
            "sample_interval_ms": args.sample_interval_ms,
            "cooldown_ms": args.cooldown_ms,
        },
        "results": results,
    }
    if args.json:
        output_path = Path(args.json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"完整结果已写入 {output_path}")


if __name__ == "__main__":
    main()
