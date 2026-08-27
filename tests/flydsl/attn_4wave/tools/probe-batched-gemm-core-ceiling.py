#!/usr/bin/env python3
"""Measure an FP8 batched-GEMM core co-issue ceiling on gfx94x.

V0 models balanced, equally shaped GEMMs. A and B are FP8 and D is BF16.
Each workgroup owns one BM x BN tile. VMEM loads and MFMAs deliberately use
independent registers; D stores use a fixed payload. The result is therefore a
GEMM core co-issue ceiling, not a correct GEMM implementation.

LDS is allocated only to cap occupancy at the requested waves/SIMD. The kernel
does not issue LDS instructions.
"""

import argparse
import csv
import ctypes
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import warnings
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch
from pyhip.core.asmjit import JIT, jit
from pyhip.core.hiptools import get_lib, hip_check_error

VOID_POINTER = "void*"
WAVE_SIZE = 64
SIMDS_PER_CU = 4
LANES_PER_TRANSACTION = 8
BYTES_PER_LANE = 16
BYTES_PER_WAVE_OP = WAVE_SIZE * BYTES_PER_LANE
FLOPS_PER_MFMA = 2 * 16 * 16 * 32
MFMA_EXECUTION_CYCLES = 16
PERF_DETERMINISM_SCLK_MHZ = 1800
EXPECTED_POWER_CAP_W = 650
LDS_BYTES_PER_CU = 64 * 1024
LDS_ALLOCATION_GRANULARITY = 256
MAX_WAVES_PER_SIMD = 8
VECTOR_REGISTERS_PER_SIMD = 512
VECTOR_REGISTER_ALLOCATION_GRANULARITY = 8
DEFAULT_AMDSMI_ROOT = Path(
    os.environ.get(
        "AMDSMI_ROOT",
        "/tmp/amd-smi-lib-26.2.2-rocm-7.2.3/opt/rocm-7.2.3",
    )
)

SCHEDULES = (
    "2stage_0",
    "2stage_prio",
    "2stage_barrier",
    "interleave",
)
SCHEDULE_CODES = {schedule: code for code, schedule in enumerate(SCHEDULES)}
GRID_ORDERS = ("batch_m_n", "batch_n_m")
GRID_ORDER_CODES = {grid_order: code for code, grid_order in enumerate(GRID_ORDERS)}
CACHE_POLICIES = ("temporal", "non_temporal")
MIN_SAMPLE_US = 100.0


def _resident_workgroups_per_cu(waves_per_block, waves_per_simd):
    if not 1 <= waves_per_simd <= MAX_WAVES_PER_SIMD:
        raise RuntimeError(f"waves-per-simd must be in [1, {MAX_WAVES_PER_SIMD}]")
    resident_waves = SIMDS_PER_CU * waves_per_simd
    if resident_waves % waves_per_block:
        raise RuntimeError(f"{waves_per_block} waves/workgroup cannot produce exactly " f"{waves_per_simd} waves/SIMD")
    return resident_waves // waves_per_block


def _lds_bytes(waves_per_block, waves_per_simd):
    resident_workgroups = _resident_workgroups_per_cu(waves_per_block, waves_per_simd)
    return LDS_BYTES_PER_CU // resident_workgroups // LDS_ALLOCATION_GRANULARITY * LDS_ALLOCATION_GRANULARITY


def _register_occupancy(artifact, waves_per_block):
    assembly_path = artifact.get("assembly_path")
    if assembly_path and Path(assembly_path).is_file():
        assembly = Path(assembly_path).read_text(encoding="utf-8")

        def metadata_count(name):
            matches = re.findall(rf"\.set\s+\S+\.{name},\s*(\d+)", assembly)
            if not matches:
                raise RuntimeError(f"missing {name} in ISA metadata: {assembly_path}")
            return int(matches[-1])

        vgprs = metadata_count("num_vgpr")
        agprs = metadata_count("num_agpr")
        sgprs = metadata_count("numbered_sgpr")
    else:
        used_gprs = artifact.get("used_gprs", ())

        def register_count(prefix):
            indices = [
                int(register[1:]) for register in used_gprs if register.startswith(prefix) and register[1:].isdigit()
            ]
            return max(indices, default=-1) + 1

        vgprs = register_count("v")
        agprs = register_count("a")
        sgprs = register_count("s")
        if not vgprs and not agprs:
            raise RuntimeError("final ISA register usage is unavailable")

    vector_registers = vgprs + agprs
    allocated_vector_registers = (
        math.ceil(vector_registers / VECTOR_REGISTER_ALLOCATION_GRANULARITY) * VECTOR_REGISTER_ALLOCATION_GRANULARITY
    )
    register_waves = min(
        MAX_WAVES_PER_SIMD,
        VECTOR_REGISTERS_PER_SIMD // allocated_vector_registers,
    )
    waves_per_workgroup_per_simd = waves_per_block // SIMDS_PER_CU
    register_workgroups = register_waves // waves_per_workgroup_per_simd
    if register_workgroups < 1:
        raise RuntimeError("final ISA register usage cannot support one workgroup")
    achievable_waves = min(
        MAX_WAVES_PER_SIMD,
        register_workgroups * waves_per_workgroup_per_simd,
    )
    return {
        "sgpr_count": sgprs,
        "vgpr_count": vgprs,
        "agpr_count": agprs,
        "vector_registers_per_wave": vector_registers,
        "allocated_vector_registers_per_wave": (allocated_vector_registers),
        "max_waves_per_simd_by_vector_registers": register_waves,
        "max_workgroups_per_cu_by_vector_registers": (register_workgroups),
        "achievable_waves_per_simd": achievable_waves,
    }


def _driver_occupancy(artifact, waves_per_block):
    kernel = artifact.get("kernel")
    if kernel is None:
        raise RuntimeError("compiled kernel handle is unavailable")
    hip_kernel = kernel.build()
    hip_kernel.lazy_load_func()
    runtime = get_lib()
    occupancy = runtime.hipModuleOccupancyMaxActiveBlocksPerMultiprocessor
    occupancy.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_size_t,
    ]
    occupancy.restype = ctypes.c_int32
    active_blocks = ctypes.c_int()
    hip_check_error(
        occupancy(
            ctypes.byref(active_blocks),
            hip_kernel.p_func,
            waves_per_block * WAVE_SIZE,
            0,
        )
    )
    waves_per_workgroup_per_simd = waves_per_block // SIMDS_PER_CU
    return {
        "max_active_workgroups_per_cu": active_blocks.value,
        "max_waves_per_simd": min(
            MAX_WAVES_PER_SIMD,
            active_blocks.value * waves_per_workgroup_per_simd,
        ),
    }


def _effective_waves_per_simd(register_occupancy, requested_waves_per_simd):
    achievable = min(
        register_occupancy["achievable_waves_per_simd"],
        register_occupancy["driver"]["max_waves_per_simd"],
    )
    if achievable < requested_waves_per_simd:
        warnings.warn(
            "requested "
            f"{requested_waves_per_simd} waves/SIMD, but final ISA uses "
            f"{register_occupancy['vgpr_count']} VGPR + "
            f"{register_occupancy['agpr_count']} AGPR "
            f"({register_occupancy['allocated_vector_registers_per_wave']} "
            "allocated vector registers/wave); compiled-resource "
            "occupancy limits this workgroup "
            f"to {achievable} waves/SIMD",
            RuntimeWarning,
        )
    return min(requested_waves_per_simd, achievable)


def _percentile(ordered, fraction):
    index = fraction * (len(ordered) - 1)
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _summary(values):
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "median": statistics.median(ordered),
        "mean": statistics.fmean(ordered),
        "p25": _percentile(ordered, 0.25),
        "p75": _percentile(ordered, 0.75),
        "max": ordered[-1],
    }


def _compact_state(state):
    keys = (
        "physical_device",
        "gpu_busy_percent",
        "vram_allocated_percent",
        "performance_level",
        "sclk",
        "mclk",
        "fclk",
        "power_cap_w",
        "ptl_state",
        "ptl_format",
        "numa_balancing",
    )
    return {key: state[key] for key in keys}


def _run_json(command, env=None):
    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    return json.loads(result.stdout), result.stderr


def _amdsmi_command(root, *arguments):
    cli = root / "libexec/amdsmi_cli/amdsmi_cli.py"
    env = os.environ.copy()
    if not cli.is_file():
        system_cli = shutil.which("amd-smi")
        if system_cli is None:
            raise RuntimeError(f"AMDSMI CLI not found under {root} or on PATH")
        return [system_cli, *arguments], env
    python_paths = [root / "share/amd_smi", root / "libexec/amdsmi_cli"]
    library_paths = [root / "lib", root / "share/amd_smi/amdsmi"]
    env["PYTHONPATH"] = ":".join(str(path) for path in python_paths) + (
        f":{env['PYTHONPATH']}" if env.get("PYTHONPATH") else ""
    )
    env["LD_LIBRARY_PATH"] = ":".join(str(path) for path in library_paths) + (
        f":{env['LD_LIBRARY_PATH']}" if env.get("LD_LIBRARY_PATH") else ""
    )
    return [sys.executable, str(cli), *arguments], env


def read_gpu_state(physical_device, amdsmi_root):
    rocm_payload, rocm_stderr = _run_json(
        [
            "rocm-smi",
            "-d",
            str(physical_device),
            "--showuse",
            "--showmemuse",
            "--showclocks",
            "--showperflevel",
            "--showpower",
            "--showpids",
            "--json",
        ]
    )
    card = rocm_payload[f"card{physical_device}"]
    static_command, env = _amdsmi_command(
        amdsmi_root,
        "static",
        "-g",
        str(physical_device),
        "--limit",
        "--json",
    )
    static_payload, static_stderr = _run_json(static_command, env)
    limit = static_payload["gpu_data"][0]["limit"]
    return {
        "physical_device": physical_device,
        "gpu_busy_percent": int(card["GPU use (%)"]),
        "vram_allocated_percent": int(card["GPU Memory Allocated (VRAM%)"]),
        "memory_activity_percent": int(card["GPU Memory Read/Write Activity (%)"]),
        "performance_level": card["Performance Level"],
        "sclk": card["sclk clock speed:"],
        "mclk": card["mclk clock speed:"],
        "fclk": card["fclk clock speed:"],
        "socket_power_w": float(card["Current Socket Graphics Package Power (W)"]),
        "ptl_state": limit["ptl_state"],
        "ptl_format": limit["ptl_format"],
        "power_cap_w": limit["ppt0"]["socket_power_limit"]["value"],
        "numa_balancing": int(Path("/proc/sys/kernel/numa_balancing").read_text().strip()),
        "rocm_smi_stderr": rocm_stderr.strip(),
        "amdsmi_stderr": static_stderr.strip(),
    }


def set_experiment_state(physical_device, amdsmi_root):
    set_status, env = _amdsmi_command(
        amdsmi_root,
        "set",
        "-g",
        str(physical_device),
        "-S",
        "1",
    )
    subprocess.run(set_status, check=True, env=env)
    set_format, env = _amdsmi_command(
        amdsmi_root,
        "set",
        "-g",
        str(physical_device),
        "-F",
        "VECTOR,F8",
    )
    subprocess.run(set_format, check=True, env=env)
    subprocess.run(
        [
            "rocm-smi",
            "-d",
            str(physical_device),
            "--setperfdeterminism",
            str(PERF_DETERMINISM_SCLK_MHZ),
        ],
        check=True,
    )


def restore_experiment_state(physical_device, amdsmi_root, original_state):
    errors = []
    try:
        subprocess.run(
            [
                "rocm-smi",
                "-d",
                str(physical_device),
                "--resetperfdeterminism",
            ],
            check=True,
        )
    except subprocess.CalledProcessError as error:
        errors.append(str(error))
    try:
        subprocess.run(
            [
                "rocm-smi",
                "-d",
                str(physical_device),
                "--setperflevel",
                "auto",
            ],
            check=True,
        )
    except subprocess.CalledProcessError as error:
        errors.append(str(error))
    try:
        if original_state["ptl_format"] not in {"N/A", "UNKNOWN,UNKNOWN"}:
            set_format, env = _amdsmi_command(
                amdsmi_root,
                "set",
                "-g",
                str(physical_device),
                "-F",
                original_state["ptl_format"],
            )
            subprocess.run(set_format, check=True, env=env)
        set_status, env = _amdsmi_command(
            amdsmi_root,
            "set",
            "-g",
            str(physical_device),
            "-S",
            "1" if original_state["ptl_state"] == "Enabled" else "0",
        )
        subprocess.run(set_status, check=True, env=env)
    except subprocess.CalledProcessError as error:
        errors.append(str(error))
    if errors:
        raise RuntimeError("failed to restore GPU state: " + "; ".join(errors))


@dataclass(frozen=True)
class DerivedConfig:
    waves_per_workgroup: int
    wave_m: int
    wave_n: int
    m_tiles: int
    n_tiles: int
    n_tile_groups: int
    k_tiles: int
    m_padded: int
    n_padded: int
    k_padded: int
    workgroups: int
    c_tiles_per_wave: int
    mfma_per_wave_per_k: int
    a_bytes_per_wave_per_k: int
    b_bytes_per_wave_per_k: int
    d_bytes_per_wave: int
    a_reads_per_wave_per_k: int
    b_reads_per_wave_per_k: int
    d_writes_per_wave: int
    a_register_vgprs: int
    a_storage_bytes: int
    b_storage_bytes: int
    d_storage_bytes: int
    useful_flops: int
    executed_flops: int
    mfma_instructions: int
    a_read_instructions: int
    b_read_instructions: int
    d_write_instructions: int


def _ceil_div(value, divisor):
    return (value + divisor - 1) // divisor


def _derive(args):
    for name in (
        "batch",
        "m",
        "n",
        "k",
        "bm",
        "bn",
        "bk",
        "waves_m",
        "waves_n",
        "waves_per_simd",
        "n_tiles_per_wg",
        "accumulator_destinations",
    ):
        if getattr(args, name) < 1:
            raise RuntimeError(f"{name.replace('_', '-')} must be positive")

    waves = args.waves_m * args.waves_n
    if waves not in (4, 8, 16):
        raise RuntimeError("waves-m * waves-n must be 4, 8, or 16")
    if args.bm % args.waves_m or args.bn % args.waves_n:
        raise RuntimeError("BM/BN must be divisible by waves-m/waves-n")
    wave_m = args.bm // args.waves_m
    wave_n = args.bn // args.waves_n
    if wave_m % 16 or wave_n % 16:
        raise RuntimeError("wave-M and wave-N must be multiples of 16")
    if args.bk % 32:
        raise RuntimeError("BK must be a multiple of FP8 MFMA K=32")
    if args.schedule == "2stage_barrier" and waves < 8:
        raise RuntimeError("2stage_barrier requires at least 8 waves/workgroup")
    if args.cross_n_prefetch and args.schedule not in (
        "2stage_0",
        "2stage_prio",
    ):
        raise RuntimeError("cross-N prefetch requires 2stage_0 or 2stage_prio")
    if args.cross_n_spread_stores and not args.cross_n_prefetch:
        raise RuntimeError("cross-N store spreading requires cross-N prefetch")

    _resident_workgroups_per_cu(waves, args.waves_per_simd)

    m_tiles = _ceil_div(args.m, args.bm)
    n_tiles = _ceil_div(args.n, args.bn)
    n_tile_groups = _ceil_div(n_tiles, args.n_tiles_per_wg)
    k_tiles = _ceil_div(args.k, args.bk)
    m_padded = m_tiles * args.bm
    n_padded = n_tile_groups * args.n_tiles_per_wg * args.bn
    k_padded = k_tiles * args.bk
    workgroups = args.batch * m_tiles * n_tile_groups

    c_tiles_per_wave = (wave_m // 16) * (wave_n // 16)
    mfma_per_wave_per_k = c_tiles_per_wave * (args.bk // 32)
    a_bytes_per_wave_per_k = wave_m * args.bk
    b_bytes_per_wave_per_k = wave_n * args.bk
    d_bytes_per_wave = wave_m * wave_n * 2

    if args.a_in_reg:
        a_register_bytes = wave_m * k_padded
        if a_register_bytes % (WAVE_SIZE * 4):
            raise RuntimeError("register-resident A bytes/wave must be VGPR aligned")
        a_reads_per_wave_per_k = 0
        a_register_vgprs = a_register_bytes // (WAVE_SIZE * 4)
    else:
        if a_bytes_per_wave_per_k % BYTES_PER_WAVE_OP:
            raise RuntimeError("A bytes/wave/K must be divisible by 1024 in V0")
        a_reads_per_wave_per_k = a_bytes_per_wave_per_k // BYTES_PER_WAVE_OP
        a_register_vgprs = 0
    if b_bytes_per_wave_per_k % BYTES_PER_WAVE_OP:
        raise RuntimeError("B bytes/wave/K must be divisible by 1024 in V0")
    if d_bytes_per_wave % BYTES_PER_WAVE_OP:
        raise RuntimeError("D bytes/wave must be divisible by 1024 in V0")
    b_reads_per_wave_per_k = b_bytes_per_wave_per_k // BYTES_PER_WAVE_OP
    d_writes_per_wave = d_bytes_per_wave // BYTES_PER_WAVE_OP

    useful_flops = 2 * args.batch * args.m * args.n * args.k
    executed_flops = 2 * args.batch * m_padded * n_padded * k_padded
    mfma_instructions = workgroups * waves * args.n_tiles_per_wg * k_tiles * mfma_per_wave_per_k
    if mfma_instructions * FLOPS_PER_MFMA != executed_flops:
        raise RuntimeError("derived MFMA work does not match padded GEMM")

    a_read_instructions = workgroups * waves * args.n_tiles_per_wg * k_tiles * a_reads_per_wave_per_k
    b_read_instructions = workgroups * waves * args.n_tiles_per_wg * k_tiles * b_reads_per_wave_per_k
    d_write_instructions = workgroups * waves * args.n_tiles_per_wg * d_writes_per_wave

    return DerivedConfig(
        waves_per_workgroup=waves,
        wave_m=wave_m,
        wave_n=wave_n,
        m_tiles=m_tiles,
        n_tiles=n_tiles,
        n_tile_groups=n_tile_groups,
        k_tiles=k_tiles,
        m_padded=m_padded,
        n_padded=n_padded,
        k_padded=k_padded,
        workgroups=workgroups,
        c_tiles_per_wave=c_tiles_per_wave,
        mfma_per_wave_per_k=mfma_per_wave_per_k,
        a_bytes_per_wave_per_k=a_bytes_per_wave_per_k,
        b_bytes_per_wave_per_k=b_bytes_per_wave_per_k,
        d_bytes_per_wave=d_bytes_per_wave,
        a_reads_per_wave_per_k=a_reads_per_wave_per_k,
        b_reads_per_wave_per_k=b_reads_per_wave_per_k,
        d_writes_per_wave=d_writes_per_wave,
        a_register_vgprs=a_register_vgprs,
        a_storage_bytes=(1 if args.a_in_reg else args.batch * m_padded * k_padded),
        b_storage_bytes=args.batch * n_padded * k_padded,
        d_storage_bytes=args.batch * m_padded * n_padded * 2,
        useful_flops=useful_flops,
        executed_flops=executed_flops,
        mfma_instructions=mfma_instructions,
        a_read_instructions=a_read_instructions,
        b_read_instructions=b_read_instructions,
        d_write_instructions=d_write_instructions,
    )


def _mfma_partition(total, parts):
    base = total // parts
    remainder = total % parts
    return tuple(base + int(index < remainder) for index in range(parts))


def _logical_access_multiset(args):
    """Return wave-level A/B/D access keys in actual linear WG order."""
    derived = _derive(args)
    accesses = {"a": Counter(), "b": Counter(), "d": Counter()}
    for block in range(derived.workgroups):
        if args.grid_order == "batch_m_n":
            batch_m, n_tile_group = divmod(block, derived.n_tile_groups)
            batch_id, m_tile = divmod(batch_m, derived.m_tiles)
        else:
            batch_n, m_tile = divmod(block, derived.m_tiles)
            batch_id, n_tile_group = divmod(batch_n, derived.n_tile_groups)
        n_tile_begin = n_tile_group * args.n_tiles_per_wg
        for n_tile_in_wg in range(args.n_tiles_per_wg):
            n_tile = n_tile_begin + n_tile_in_wg
            for wave in range(derived.waves_per_workgroup):
                wave_m, wave_n = divmod(wave, args.waves_n)
                a_stream = (batch_id * derived.m_tiles + m_tile) * args.waves_m + wave_m
                b_stream = (batch_id * derived.n_padded // args.bn + n_tile) * args.waves_n + wave_n
                d_stream = (
                    (batch_id * derived.m_tiles + m_tile) * (derived.n_padded // args.bn) + n_tile
                ) * derived.waves_per_workgroup + wave
                for k_tile in range(derived.k_tiles):
                    for read in range(derived.a_reads_per_wave_per_k):
                        accesses["a"][(a_stream, k_tile, read)] += 1
                    for read in range(derived.b_reads_per_wave_per_k):
                        accesses["b"][(b_stream, k_tile, read)] += 1
                for write in range(derived.d_writes_per_wave):
                    accesses["d"][(d_stream, write)] += 1
    return accesses


@jit(no_pass=["pass_dse", "pass_dce"])
def batched_gemm_core_ceiling(
    builder: JIT,
    batch,
    m,
    n,
    k,
    bm,
    bn,
    bk,
    waves_m,
    waves_n,
    n_tiles_per_wg,
    waves_per_simd,
    accumulator_destinations,
    cross_n_prefetch,
    cross_n_spread_stores,
    a_in_reg,
    grid_order_code,
    schedule_code,
    non_temporal,
    a: VOID_POINTER,  # pyright: ignore[reportInvalidTypeForm]
    b: VOID_POINTER,  # pyright: ignore[reportInvalidTypeForm]
    d: VOID_POINTER,  # pyright: ignore[reportInvalidTypeForm]
):
    waves = waves_m * waves_n
    wave_m_size = bm // waves_m
    wave_n_size = bn // waves_n
    m_tiles = (m + bm - 1) // bm
    n_tiles = (n + bn - 1) // bn
    n_tile_groups = (n_tiles + n_tiles_per_wg - 1) // n_tiles_per_wg
    n_tiles_padded = n_tile_groups * n_tiles_per_wg
    k_tiles = (k + bk - 1) // bk
    k_padded = k_tiles * bk
    c_tiles = (wave_m_size // 16) * (wave_n_size // 16)
    mfma_per_k = c_tiles * (bk // 32)
    a_bytes_per_k = wave_m_size * bk
    b_bytes_per_k = wave_n_size * bk
    d_bytes_per_wave = wave_m_size * wave_n_size * 2
    a_reads = 0 if a_in_reg else a_bytes_per_k // BYTES_PER_WAVE_OP
    b_reads = b_bytes_per_k // BYTES_PER_WAVE_OP
    reads_per_k = a_reads + b_reads
    d_writes = d_bytes_per_wave // BYTES_PER_WAVE_OP
    schedule = SCHEDULES[schedule_code]
    grid_order = GRID_ORDERS[grid_order_code]
    nt = bool(non_temporal)
    cross_n = bool(cross_n_prefetch)
    spread_stores = bool(cross_n_spread_stores)

    assert waves in (4, 8, 16)
    assert reads_per_k > 0 and d_writes > 0
    assert accumulator_destinations > 0
    assert not cross_n or schedule in ("2stage_0", "2stage_prio")
    assert not spread_stores or cross_n
    assert not spread_stores or d_writes <= k_tiles * mfma_per_k
    assert schedule != "2stage_barrier" or waves >= 8
    builder.alloc_lds(_lds_bytes(waves, waves_per_simd), align=16)

    block = builder.blockIdx.x[0]
    if grid_order == "batch_m_n":
        batch_m = block if n_tile_groups == 1 else block // n_tile_groups
        batch_id = batch_m if m_tiles == 1 else batch_m // m_tiles
        n_tile_group = 0 if n_tile_groups == 1 else block - batch_m * n_tile_groups
        m_tile = 0 if m_tiles == 1 else batch_m - batch_id * m_tiles
    else:
        batch_n = block if m_tiles == 1 else block // m_tiles
        batch_id = batch_n if n_tile_groups == 1 else batch_n // n_tile_groups
        m_tile = 0 if m_tiles == 1 else block - batch_n * m_tiles
        n_tile_group = 0 if n_tile_groups == 1 else batch_n - batch_id * n_tile_groups
    n_tile_begin = n_tile_group * n_tiles_per_wg

    wave = builder.warp_id[0]
    wave_m = wave // waves_n
    wave_n = wave % waves_n

    a_stream_bytes = wave_m_size * k_padded
    b_stream_bytes = wave_n_size * k_padded
    a_stream_id = (batch_id * m_tiles + m_tile) * waves_m + wave_m
    b_stream_id = (batch_id * n_tiles_padded + n_tile_begin) * waves_n + wave_n
    d_stream_id = ((batch_id * m_tiles + m_tile) * n_tiles_padded + n_tile_begin) * waves + wave

    def make_stream_buffer(pointer, stream_id, stream_stride, descriptor_bytes=None):
        buffer = builder.Buffer(
            pointer,
            stream_stride if descriptor_bytes is None else descriptor_bytes,
        )
        offset = builder.s_mul_u32_u64(stream_id, stream_stride)
        base = buffer.get_base()
        builder.s_add_u32(base[0], base[0], offset[0])
        builder.s_addc_u32(base[1], base[1], offset[1])
        return buffer

    a_buffer = None if a_in_reg else make_stream_buffer(a, a_stream_id, a_stream_bytes)
    b_group_bytes = ((n_tiles_per_wg - 1) * waves_n + 1) * b_stream_bytes
    d_group_bytes = ((n_tiles_per_wg - 1) * waves + 1) * d_bytes_per_wave
    b_buffer = make_stream_buffer(b, b_stream_id, b_stream_bytes, b_group_bytes)
    d_buffer = make_stream_buffer(d, d_stream_id, d_bytes_per_wave, d_group_bytes)
    if a_in_reg:
        # Keep the unused kernarg live across B/D descriptor construction.
        # Otherwise the JIT allocator can issue A and B loads into the same
        # SGPR pair before lgkmcnt(0), allowing A to overwrite B on completion.
        builder.gpr("su32", a[0] ^ a[1])

    lane_group = builder.lane_id[0] // LANES_PER_TRANSACTION
    lane_in_group = builder.lane_id[0] % LANES_PER_TRANSACTION
    vector_offset = builder.gpr(
        "vu32",
        lane_group * (LANES_PER_TRANSACTION * BYTES_PER_LANE) + lane_in_group * BYTES_PER_LANE,
    )

    if a_in_reg:
        a_registers = builder.gpr(
            max(2, wave_m_size * k_padded // (WAVE_SIZE * 4)),
            "vu32",
            align=2,
        )
        operand_a = a_registers[0:1]
    else:
        operand_a_registers = builder.gpr(2, "vu32", align=2)
        operand_a = operand_a_registers[0:1]
    operand_b_registers = builder.gpr(2, "vu32", align=2)
    operand_b = operand_b_registers[0:1]
    store_payload = builder.gpr(4, "vu32", align=4)
    # Destinations are write-only: immediate-zero C removes accumulator RAW.
    mfma_destinations = builder.gpr(accumulator_destinations, 4, "af32", align=4)

    load_values = builder.gpr(2, reads_per_k, 4, "vu32", align=4)
    load_sink = builder.gpr("vu32", 0)
    operation_offsets = builder.gpr(reads_per_k, "su32")

    def prepare_batch(n_tile_in_wg, k_tile):
        operation = 0
        if not a_in_reg:
            for read_index in range(a_reads):
                operation_offsets[operation] = k_tile * a_bytes_per_k + read_index * BYTES_PER_WAVE_OP
                operation += 1
        for read_index in range(b_reads):
            operation_offsets[operation] = (
                n_tile_in_wg * waves_n * b_stream_bytes + k_tile * b_bytes_per_k + read_index * BYTES_PER_WAVE_OP
            )
            operation += 1

    def issue_operation(bank, operation):
        target = a_buffer if not a_in_reg and operation < a_reads else b_buffer
        target.load_dwordx4(
            load_values[bank, operation],
            vector_offset,
            operation_offsets[operation],
            non_temporal=nt,
        )

    def issue_batch(bank, n_tile_in_wg, k_tile):
        prepare_batch(n_tile_in_wg, k_tile)
        for operation in range(reads_per_k):
            issue_operation(bank, operation)

    def consume_operation(bank, operation):
        builder.v_xor_b32(
            load_sink,
            load_sink,
            load_values[bank, operation, 0],
        )

    def consume_batch(bank):
        for operation in range(reads_per_k):
            consume_operation(bank, operation)

    def emit_mfma(begin, count):
        for mfma_index in range(begin, begin + count):
            destination = mfma_destinations[mfma_index % accumulator_destinations]
            builder.v_mfma_f32_16x16x32_fp8_fp8(
                destination,
                operand_a,
                operand_b,
                0,
            )

    use_priority = schedule in ("2stage_prio", "2stage_barrier")
    use_barrier = schedule == "2stage_barrier"
    interleave = schedule == "interleave"
    half_waves = waves // 2

    d_offsets = builder.gpr(d_writes, "su32")

    store_positions = tuple((write_index + 1) * k_tiles * mfma_per_k // d_writes for write_index in range(d_writes))
    stores_per_k = tuple(
        sum(k_tile * mfma_per_k < position <= (k_tile + 1) * mfma_per_k for position in store_positions)
        for k_tile in range(k_tiles)
    )

    def issue_store(n_tile_in_wg, write_index):
        d_offsets[write_index] = n_tile_in_wg * waves * d_bytes_per_wave + write_index * BYTES_PER_WAVE_OP
        d_buffer.store_dwordx4(
            store_payload,
            vector_offset,
            d_offsets[write_index],
            ext_mod="sc0 nt" if nt else "",
        )

    def issue_stores(n_tile_in_wg):
        for write_index in range(d_writes):
            issue_store(n_tile_in_wg, write_index)

    def emit_mfma_with_stores(n_tile_in_wg, k_tile):
        stage_begin = k_tile * mfma_per_k
        mfma_begin = 0
        for write_index, position in enumerate(store_positions):
            if stage_begin < position <= stage_begin + mfma_per_k:
                mfma_end = position - stage_begin
                emit_mfma(mfma_begin, mfma_end - mfma_begin)
                issue_store(n_tile_in_wg, write_index)
                mfma_begin = mfma_end
        emit_mfma(mfma_begin, mfma_per_k - mfma_begin)

    for n_tile_in_wg in range(n_tiles_per_wg):
        if use_barrier:
            with builder.If(builder.warp_id[0] >= half_waves):
                builder.s_barrier()
            builder.s_barrier()

        bank_phase = n_tile_in_wg if cross_n else 0
        if not cross_n or n_tile_in_wg == 0:
            issue_batch(bank_phase & 1, n_tile_in_wg, 0)
        for k_tile in range(k_tiles):
            current_bank = (bank_phase + k_tile) & 1
            next_bank = (bank_phase + k_tile + 1) & 1
            issue_next_k = k_tile + 1 < k_tiles
            issue_next_n = cross_n and not issue_next_k and n_tile_in_wg + 1 < n_tiles_per_wg
            if cross_n and not spread_stores and n_tile_in_wg > 0 and k_tile == 1:
                issue_stores(n_tile_in_wg - 1)
            if interleave and issue_next_k:
                prepare_batch(n_tile_in_wg, k_tile + 1)
                mfma_counts = _mfma_partition(mfma_per_k, reads_per_k)
                mfma_begin = 0
                for operation, mfma_count in enumerate(mfma_counts):
                    issue_operation(next_bank, operation)
                    builder.s_waitcnt(mod=f"vmcnt({reads_per_k})")
                    consume_operation(current_bank, operation)
                    emit_mfma(mfma_begin, mfma_count)
                    mfma_begin += mfma_count
            else:
                if issue_next_k:
                    issue_batch(next_bank, n_tile_in_wg, k_tile + 1)
                    wait_target = reads_per_k
                    if cross_n and not spread_stores and n_tile_in_wg > 0 and k_tile == 1:
                        wait_target += d_writes
                    if spread_stores and n_tile_in_wg > 0:
                        if k_tile > 0:
                            wait_target += stores_per_k[k_tile - 1]
                        elif n_tile_in_wg > 1:
                            wait_target += stores_per_k[-1]
                    builder.s_waitcnt(mod=f"vmcnt({wait_target})")
                elif issue_next_n:
                    issue_batch(next_bank, n_tile_in_wg + 1, 0)
                    wait_target = reads_per_k
                    if spread_stores and n_tile_in_wg > 0:
                        wait_target += stores_per_k[k_tile - 1]
                    builder.s_waitcnt(mod=f"vmcnt({wait_target})")
                else:
                    wait_target = 0
                    if spread_stores and n_tile_in_wg > 0:
                        wait_target = stores_per_k[k_tile - 1]
                    builder.s_waitcnt(mod=f"vmcnt({wait_target})")
                consume_batch(current_bank)
                if use_barrier:
                    builder.s_barrier()
                if use_priority:
                    builder.s_setprio(1)
                if spread_stores and n_tile_in_wg > 0:
                    emit_mfma_with_stores(n_tile_in_wg - 1, k_tile)
                else:
                    emit_mfma(0, mfma_per_k)
                if use_priority:
                    builder.s_setprio(0)
                if use_barrier:
                    builder.s_barrier()

        if use_barrier:
            with builder.If(builder.warp_id[0] < half_waves):
                builder.s_barrier()

        if not cross_n:
            issue_stores(n_tile_in_wg)
            builder.s_waitcnt(mod="vmcnt(0)")

    if cross_n:
        # The last tile has no successor over which to spread its stores.
        issue_stores(n_tiles_per_wg - 1)
        builder.s_waitcnt(mod="vmcnt(0)")


def _short_compile_key(kernel_key):
    digest = hashlib.sha256(kernel_key.encode("utf-8")).hexdigest()[:16]
    return f"key={digest}"


_original_split_args = batched_gemm_core_ceiling.split_args


def _split_args_with_short_key(args):
    compile_args, runtime_args, kernel_key = _original_split_args(args)
    return compile_args, runtime_args, _short_compile_key(kernel_key)


batched_gemm_core_ceiling.split_args = _split_args_with_short_key


def _artifact_for_launch(artifact):
    if "assembly_path" in artifact and "kernel" in artifact:
        return artifact
    for kernel, cached_artifact in batched_gemm_core_ceiling.kernel_cache.values():
        if cached_artifact is artifact:
            artifact.setdefault("assembly_path", str(Path(kernel.src_fpath).with_suffix(".s")))
            artifact.setdefault("kernel", kernel)
            return artifact
    raise RuntimeError("compiled artifact is not present in the JIT cache")


def _launch(args, derived, a, b, d):
    artifact = batched_gemm_core_ceiling(
        [derived.workgroups],
        [derived.waves_per_workgroup * WAVE_SIZE],
        args.batch,
        args.m,
        args.n,
        args.k,
        args.bm,
        args.bn,
        args.bk,
        args.waves_m,
        args.waves_n,
        args.n_tiles_per_wg,
        args.waves_per_simd,
        args.accumulator_destinations,
        args.cross_n_prefetch,
        args.cross_n_spread_stores,
        args.a_in_reg,
        GRID_ORDER_CODES[args.grid_order],
        SCHEDULE_CODES[args.schedule],
        args.cache_policy == "non_temporal",
        a.data_ptr(),
        b.data_ptr(),
        d.data_ptr(),
    )
    return _artifact_for_launch(artifact)


def _assert_no_lds_instructions(artifact):
    assembly_path = Path(artifact["assembly_path"])
    assembly = assembly_path.read_text(encoding="utf-8")
    instructions = re.findall(r"^\s*(ds_[a-zA-Z0-9_]+)", assembly, re.MULTILINE)
    if instructions:
        raise RuntimeError("ceiling kernel unexpectedly uses LDS instructions: " f"{sorted(set(instructions))}")


def _allocate(derived, buffer_copies):
    return [
        (
            torch.empty(
                derived.a_storage_bytes,
                dtype=torch.uint8,
                device="cuda",
            ),
            torch.empty(
                derived.b_storage_bytes,
                dtype=torch.uint8,
                device="cuda",
            ),
            torch.empty(
                derived.d_storage_bytes // 2,
                dtype=torch.bfloat16,
                device="cuda",
            ),
        )
        for _ in range(buffer_copies)
    ]


def _config_dict(args, derived):
    return {
        "problem": {
            "batch": args.batch,
            "m": args.m,
            "n": args.n,
            "k": args.k,
            "a_dtype": "fp8",
            "b_dtype": "fp8",
            "d_dtype": "bf16",
            "balanced_batch": True,
        },
        "candidate": {
            "bm": args.bm,
            "bn": args.bn,
            "bk": args.bk,
            "waves_m": args.waves_m,
            "waves_n": args.waves_n,
            "n_tiles_per_wg": args.n_tiles_per_wg,
            "waves_per_workgroup": derived.waves_per_workgroup,
            "requested_waves_per_simd": args.waves_per_simd,
            "accumulator_destinations": args.accumulator_destinations,
            "cross_n_prefetch": args.cross_n_prefetch,
            "cross_n_spread_stores": args.cross_n_spread_stores,
            "a_in_reg": args.a_in_reg,
            "grid_order": args.grid_order,
            "schedule": args.schedule,
            "cache_policy": args.cache_policy,
        },
        "derived": {key: value for key, value in derived.__dict__.items()},
    }


def _validate_timed_artifact(args, derived, artifact):
    _assert_no_lds_instructions(artifact)
    occupancy = _register_occupancy(artifact, derived.waves_per_workgroup)
    occupancy["driver"] = _driver_occupancy(artifact, derived.waves_per_workgroup)
    actual = _effective_waves_per_simd(occupancy, args.waves_per_simd)
    if actual != args.waves_per_simd:
        raise RuntimeError(f"candidate requested Q{args.waves_per_simd}, but final resources allow Q{actual}")
    expected_workgroups = _resident_workgroups_per_cu(derived.waves_per_workgroup, args.waves_per_simd)
    if occupancy["driver"]["max_active_workgroups_per_cu"] != expected_workgroups:
        raise RuntimeError(
            "LDS did not produce the exact requested workgroups/CU: "
            f"expected {expected_workgroups}, got "
            f"{occupancy['driver']['max_active_workgroups_per_cu']}"
        )
    return occupancy


def _run_bench(args):
    derived = _derive(args)
    if os.environ.get("HIP_VISIBLE_DEVICES") != str(args.physical_device):
        raise RuntimeError("HIP_VISIBLE_DEVICES must equal --physical-device")
    original = read_gpu_state(args.physical_device, args.amdsmi_root)
    if original["performance_level"] != "auto":
        raise RuntimeError(f"GPU must start in auto: {original}")
    if original["gpu_busy_percent"] > 5 or original["vram_allocated_percent"] > args.max_initial_vram_percent:
        raise RuntimeError(f"GPU is not idle: {original}")
    if abs(original["power_cap_w"] - EXPECTED_POWER_CAP_W) > 0.5:
        raise RuntimeError(f"expected 650 W power cap: {original}")

    payload = None
    state_change_attempted = False
    try:
        state_change_attempted = True
        set_experiment_state(args.physical_device, args.amdsmi_root)
        managed = read_gpu_state(args.physical_device, args.amdsmi_root)
        torch.cuda.set_device(args.device)
        properties = torch.cuda.get_device_properties(args.device)
        if not properties.gcnArchName.startswith("gfx94"):
            raise RuntimeError(f"gfx94x required, got {properties.gcnArchName}")
        bytes_per_buffer = derived.a_storage_bytes + derived.b_storage_bytes + derived.d_storage_bytes
        required = bytes_per_buffer * args.buffer_copies
        free_bytes, _ = torch.cuda.mem_get_info(args.device)
        if required > int(free_bytes * 0.9):
            raise RuntimeError(
                f"candidate needs {required / 2**30:.2f} GiB, " f"but only {free_bytes / 2**30:.2f} GiB is free"
            )
        buffers = _allocate(derived, args.buffer_copies)
        artifact = _launch(args, derived, *buffers[0])
        for launch_index in range(1, args.warmups):
            _launch(
                args,
                derived,
                *buffers[launch_index % args.buffer_copies],
            )
        torch.cuda.synchronize()
        occupancy = _validate_timed_artifact(args, derived, artifact)

        elapsed_ms = []
        useful_tflops = []
        executed_tflops = []
        events = []
        launch_index = args.warmups
        for _ in range(args.samples):
            start = torch.cuda.Event(enable_timing=True)
            stop = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(args.launches_per_sample):
                _launch(
                    args,
                    derived,
                    *buffers[launch_index % args.buffer_copies],
                )
                launch_index += 1
            stop.record()
            events.append((start, stop))
            if args.sample_sync == "each":
                torch.cuda.synchronize()

        if args.sample_sync == "end":
            torch.cuda.synchronize()
        for start, stop in events:
            total_ms = start.elapsed_time(stop)
            if total_ms * 1000.0 < MIN_SAMPLE_US:
                raise RuntimeError(
                    f"timed sample is only {total_ms * 1000.0:.2f} us; "
                    "increase --launches-per-sample or problem size"
                )
            per_launch_ms = total_ms / args.launches_per_sample
            elapsed_ms.append(per_launch_ms)
            useful_tflops.append(derived.useful_flops * args.launches_per_sample / (total_ms * 1.0e9))
            executed_tflops.append(derived.executed_flops * args.launches_per_sample / (total_ms * 1.0e9))

        config = _config_dict(args, derived)
        config["candidate"]["actual_waves_per_simd"] = args.waves_per_simd
        config["candidate"]["occupancy_lds_bytes"] = _lds_bytes(derived.waves_per_workgroup, args.waves_per_simd)
        payload = {
            "schema_version": 1,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "device": {
                "arch": properties.gcnArchName,
                "cu_count": properties.multi_processor_count,
                "physical_device": args.physical_device,
            },
            "method": {
                "name": "gemm_core_coissue_ceiling",
                "timing": "CUDA events around uninstrumented dispatches",
                "warmup_dispatches": args.warmups,
                "samples": args.samples,
                "launches_per_sample": args.launches_per_sample,
                "sample_sync": args.sample_sync,
                "buffer_copies": args.buffer_copies,
                "buffer_rotation": "round_robin_across_warmups_and_samples",
                "bytes_per_buffer": bytes_per_buffer,
                "rotated_tensors": (["B", "D"] if args.a_in_reg else ["A", "B", "D"]),
                "a_buffer_semantics": (
                    "unaccessed one-byte placeholder per buffer" if args.a_in_reg else "round-robin VMEM source"
                ),
                "vmem_mfma_dependency": False,
                "lds_data_access": False,
                "d_store_dependency": False,
            },
            **config,
            "occupancy": occupancy,
            "kernel_ms": _summary(elapsed_ms),
            "useful_tflops": _summary(useful_tflops),
            "executed_tflops": _summary(executed_tflops),
            "initial_state": _compact_state(original),
            "managed_state": _compact_state(managed),
        }
    finally:
        if state_change_attempted:
            restore_experiment_state(args.physical_device, args.amdsmi_root, original)
        restored = read_gpu_state(args.physical_device, args.amdsmi_root)
        for key in (
            "performance_level",
            "ptl_state",
            "ptl_format",
            "numa_balancing",
        ):
            if restored[key] != original[key]:
                raise RuntimeError(f"GPU restoration mismatch for {key}: " f"{original[key]!r} -> {restored[key]!r}")
    payload["restored_state"] = _compact_state(restored)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _run_pmc(args):
    derived = _derive(args)
    torch.cuda.set_device(args.device)
    buffers = _allocate(derived, args.buffer_copies)
    for launch_index in range(args.warmups):
        _launch(
            args,
            derived,
            *buffers[launch_index % args.buffer_copies],
        )
    artifact = _launch(
        args,
        derived,
        *buffers[args.warmups % args.buffer_copies],
    )
    torch.cuda.synchronize()
    occupancy = _validate_timed_artifact(args, derived, artifact)
    payload = {
        **_config_dict(args, derived),
        "occupancy": occupancy,
        "method": {
            "warmup_dispatches": args.warmups,
            "buffer_copies": args.buffer_copies,
            "buffer_rotation": "round_robin",
            "profiled_buffer_index": args.warmups % args.buffer_copies,
            "rotated_tensors": (["B", "D"] if args.a_in_reg else ["A", "B", "D"]),
        },
        "expected": {
            "waves": (derived.workgroups * derived.waves_per_workgroup),
            "mfma_instructions": derived.mfma_instructions,
            "read_instructions": (derived.a_read_instructions + derived.b_read_instructions),
            "write_instructions": derived.d_write_instructions,
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _run_pmc_analyze(args):
    expected = json.loads(args.expected.read_text(encoding="utf-8"))["expected"]
    dispatches = {}
    with args.csv.open(newline="", encoding="utf-8") as csv_file:
        for row in csv.DictReader(csv_file):
            if "batched_gemm_core_ceiling" not in row["Kernel_Name"]:
                continue
            dispatches.setdefault(int(row["Dispatch_Id"]), {})[row["Counter_Name"]] = float(row["Counter_Value"])
    if not dispatches:
        raise RuntimeError("no batched_gemm_core_ceiling counters found")
    counters = dispatches[max(dispatches)]
    checks = {
        "waves_exact": counters["SQ_WAVES"] == expected["waves"],
        "mfma_exact": (counters["SQ_INSTS_MFMA"] == expected["mfma_instructions"]),
        "read_exact": (counters["SQ_INSTS_VMEM_RD"] == expected["read_instructions"]),
        "write_exact": (counters["SQ_INSTS_VMEM_WR"] == expected["write_instructions"]),
    }
    result = {
        "dispatch_id": max(dispatches),
        "expected": expected,
        "counters": counters,
        "checks": checks,
        "valid": all(checks.values()),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if not result["valid"]:
        raise RuntimeError(f"SQ closure failed: {checks}")


def _candidate_arguments(parser):
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--physical-device", type=int, default=4)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--bm", type=int, required=True)
    parser.add_argument("--bn", type=int, required=True)
    parser.add_argument("--bk", type=int, required=True)
    parser.add_argument("--waves-m", type=int, required=True)
    parser.add_argument("--waves-n", type=int, required=True)
    parser.add_argument(
        "--n-tiles-per-wg",
        type=int,
        default=1,
        help="number of consecutive BN tiles computed by each workgroup",
    )
    parser.add_argument("--waves-per-simd", type=int, required=True)
    parser.add_argument(
        "--accumulator-destinations",
        type=int,
        choices=(1, 2, 3, 4),
        default=1,
    )
    parser.add_argument("--cross-n-prefetch", action="store_true")
    parser.add_argument("--cross-n-spread-stores", action="store_true")
    parser.add_argument("--a-in-reg", action="store_true")
    parser.add_argument("--grid-order", choices=GRID_ORDERS, default="batch_m_n")
    parser.add_argument("--schedule", choices=SCHEDULES, default="2stage_0")
    parser.add_argument("--cache-policy", choices=CACHE_POLICIES, default="temporal")
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument(
        "--buffer-copies",
        type=int,
        default=1,
        help="number of A/B/D address sets used in round-robin order",
    )
    parser.add_argument("--json", type=Path)


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    bench = subparsers.add_parser("bench")
    _candidate_arguments(bench)
    bench.add_argument("--samples", type=int, default=12)
    bench.add_argument("--launches-per-sample", type=int, default=1)
    bench.add_argument(
        "--sample-sync",
        choices=("each", "end"),
        default="each",
        help="synchronize after each sample or after all samples",
    )
    bench.add_argument("--max-initial-vram-percent", type=int, default=20)
    bench.add_argument("--amdsmi-root", type=Path, default=DEFAULT_AMDSMI_ROOT)

    pmc = subparsers.add_parser("pmc-run")
    _candidate_arguments(pmc)

    analyze = subparsers.add_parser("pmc-analyze")
    analyze.add_argument("--expected", type=Path, required=True)
    analyze.add_argument("--csv", type=Path, required=True)
    analyze.add_argument("--json", type=Path)

    subparsers.add_parser("self-test")
    return parser


def _self_test():
    args = _parser().parse_args(
        [
            "bench",
            "--batch",
            "193",
            "--m",
            "1536",
            "--n",
            "4096",
            "--k",
            "192",
            "--bm",
            "64",
            "--bn",
            "512",
            "--bk",
            "64",
            "--waves-m",
            "1",
            "--waves-n",
            "8",
            "--waves-per-simd",
            "4",
            "--a-in-reg",
        ]
    )
    derived = _derive(args)
    assert args.accumulator_destinations == 1
    assert args.buffer_copies == 1
    assert derived.waves_per_workgroup == 8
    assert derived.wave_m == 64 and derived.wave_n == 64
    assert derived.m_tiles == 24
    assert derived.n_tiles == 8
    assert derived.n_tile_groups == 8
    assert derived.k_tiles == 3
    assert derived.workgroups == 37_056
    assert derived.mfma_per_wave_per_k == 32
    assert derived.a_reads_per_wave_per_k == 0
    assert derived.b_reads_per_wave_per_k == 4
    assert derived.d_writes_per_wave == 8
    assert derived.a_register_vgprs == 48
    assert derived.mfma_instructions * FLOPS_PER_MFMA == derived.executed_flops
    assert derived.useful_flops == derived.executed_flops

    args.n_tiles_per_wg = 8
    derived = _derive(args)
    assert derived.n_tile_groups == 1
    assert derived.workgroups == 4_632
    assert derived.mfma_instructions == 28_459_008
    assert derived.b_read_instructions == 3_557_376
    assert derived.d_write_instructions == 2_371_584
    assert derived.a_register_vgprs == 48

    args.n = 5 * args.bn
    args.n_tiles_per_wg = 4
    derived = _derive(args)
    assert derived.n_tiles == 5
    assert derived.n_tile_groups == 2
    assert derived.n_padded == 8 * args.bn
    assert derived.workgroups == args.batch * derived.m_tiles * 2
    assert derived.useful_flops < derived.executed_flops
    args.n = 4096
    args.n_tiles_per_wg = 8

    args.n = 24 * args.bn
    args.n_tiles_per_wg = 24
    derived = _derive(args)
    assert derived.n_tiles == 24
    assert derived.n_tile_groups == 1
    assert derived.n_padded == args.n
    args.n = 4096
    args.n_tiles_per_wg = 8

    args.a_in_reg = False
    derived = _derive(args)
    assert derived.a_reads_per_wave_per_k == 4
    assert derived.a_register_vgprs == 0

    args.batch = 2
    args.m = 128
    args.n = 4096
    args.a_in_reg = False
    args.schedule = "2stage_0"
    args.waves_m = 1
    args.waves_n = 8
    args.grid_order = "batch_m_n"
    args.n_tiles_per_wg = 1
    baseline_accesses = _logical_access_multiset(args)
    for n_tiles_per_wg in (2, 4, 8):
        args.n_tiles_per_wg = n_tiles_per_wg
        assert _logical_access_multiset(args) == baseline_accesses
    args.grid_order = "batch_n_m"
    assert _logical_access_multiset(args) == baseline_accesses

    args.m = 1500
    derived = _derive(args)
    assert derived.m_padded == 1536
    assert derived.useful_flops < derived.executed_flops

    args.batch = 193
    args.waves_m = 1
    args.waves_n = 4
    args.schedule = "2stage_barrier"
    try:
        _derive(args)
    except RuntimeError as error:
        assert "requires at least 8" in str(error)
    else:
        raise AssertionError("W4 2stage_barrier unexpectedly accepted")
    print("self-test passed: geometry, work, VMEM, and padding")


def main():
    args = _parser().parse_args()
    if args.command == "self-test":
        _self_test()
    elif args.command == "bench":
        if args.samples < 1 or args.warmups < 1 or args.buffer_copies < 1:
            raise RuntimeError("samples, warmups, and buffer-copies must be positive")
        if args.launches_per_sample < 1:
            raise RuntimeError("launches-per-sample must be positive")
        if not 0 <= args.max_initial_vram_percent <= 100:
            raise RuntimeError("max-initial-vram-percent must be in [0, 100]")
        _run_bench(args)
    elif args.command == "pmc-run":
        if args.warmups < 0 or args.buffer_copies < 1:
            raise RuntimeError("warmups cannot be negative and buffer-copies must be positive")
        _run_pmc(args)
    else:
        _run_pmc_analyze(args)


if __name__ == "__main__":
    main()
