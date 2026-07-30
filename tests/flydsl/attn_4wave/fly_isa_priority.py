import ctypes
import hashlib
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import torch

from pyhip.core.hiptools import (
    get_lib,
    hipModuleGetFunction,
    hipModuleLoad,
    hip_check_error,
)

ATTN_MFMA_PERIOD = 64
ATTN_MFMA_COUNT = 128
ATTN_PRIORITY_EVENTS = ((46, 2), (64, 0))
ATTN_KERNARG_SIZE = 164
ATTN_POINTER_OFFSETS = (0, 40, 80, 128)
ATTN_IDENTITY_MAX_SPACERS = (79, 83, 79, 83, 77, 113, 77, 113)


def preshuffle_jit_key(key, *, block_n=32):
    """将逻辑K[H,N,D]转换为归档JIT机器码的32-token物理布局。"""
    heads, sequence, head_dim = key.shape
    if block_n != 32 or head_dim != 128:
        raise ValueError(
            f"the archived JIT layout requires block_n=32 and D=128, got {block_n=}, D={head_dim}"
        )
    if sequence % block_n != 0:
        raise ValueError(f"N={sequence} must be divisible by block_n={block_n}")
    tile_count = sequence // block_n
    grouped = key.reshape(heads, tile_count, 8, 4, head_dim)
    grouped = grouped[:, :, (0, 2, 4, 6, 1, 3, 5, 7)]
    return (
        grouped.reshape(heads, tile_count, 2, 16, head_dim)
        .reshape(heads, tile_count, 2, 16, head_dim // 8, 8)
        .permute(0, 1, 2, 4, 3, 5)
        .contiguous()
    )


def preshuffle_jit_value(value, *, block_n=32):
    """将逻辑V[H,N,D]转换为归档JIT机器码的GEMM2物理布局。"""
    heads, sequence, head_dim = value.shape
    if block_n != 32 or head_dim != 128:
        raise ValueError(
            f"the archived JIT layout requires block_n=32 and D=128, got {block_n=}, D={head_dim}"
        )
    if sequence % block_n != 0:
        raise ValueError(f"N={sequence} must be divisible by block_n={block_n}")
    return (
        value.reshape(heads, sequence // block_n, 4, 8, head_dim // 16, 16)
        .permute(0, 1, 4, 2, 5, 3)
        .contiguous()
    )


def convert_accvgprs_to_vgprs(isa):
    """将gfx942统一寄存器文件中的显式AGPR机械重命名到空闲VGPR区间。"""
    if "amdgcn-amd-amdhsa--gfx942" not in isa:
        raise ValueError("expected a gfx942 AMDGPU assembly")

    num_vgpr_match = re.search(
        r"^\s*\.set\s+(\S+)\.num_vgpr,\s*(\d+)\s*$", isa, re.MULTILINE
    )
    num_agpr_match = re.search(
        r"^\s*\.set\s+(\S+)\.num_agpr,\s*(\d+)\s*$", isa, re.MULTILINE
    )
    if num_vgpr_match is None or num_agpr_match is None:
        raise ValueError("expected explicit num_vgpr and num_agpr resource symbols")
    if num_vgpr_match.group(1) != num_agpr_match.group(1):
        raise ValueError("num_vgpr and num_agpr symbols refer to different kernels")

    symbol = num_vgpr_match.group(1)
    num_vgprs = int(num_vgpr_match.group(2))
    num_agprs = int(num_agpr_match.group(2))
    if num_agprs == 0:
        raise ValueError("assembly does not use AGPRs")

    operand_pattern = re.compile(r"\ba\[(\d+)(?::(\d+))?\]")
    operand_ranges = [
        (int(match.group(1)), int(match.group(2) or match.group(1)))
        for line in isa.splitlines()
        for match in operand_pattern.finditer(line.split(";", 1)[0])
    ]
    if not operand_ranges:
        raise ValueError(
            "assembly metadata reports AGPRs but no AGPR operands were found"
        )
    if min(begin for begin, _ in operand_ranges) != 0:
        raise ValueError(
            "AGPR operands must begin at a0 for contiguous mechanical remapping"
        )
    if max(end for _, end in operand_ranges) + 1 != num_agprs:
        raise ValueError("AGPR operand range does not match num_agpr metadata")

    vgpr_pattern = re.compile(r"\bv(?:\[(\d+)(?::(\d+))?\]|(\d+))")
    vgpr_ranges = []
    for line in isa.splitlines():
        for match in vgpr_pattern.finditer(line.split(";", 1)[0]):
            begin = int(match.group(1) or match.group(3))
            end = int(match.group(2) or begin)
            vgpr_ranges.append((begin, end))
    if vgpr_ranges and max(end for _, end in vgpr_ranges) >= num_vgprs:
        raise ValueError(
            "existing VGPR operands overlap the target AGPR remapping range"
        )

    total_vgprs = num_vgprs + num_agprs
    if total_vgprs > 256:
        raise ValueError(
            f"combined VGPR pressure {total_vgprs} exceeds the 256-register proof boundary"
        )

    def replace_operand(match):
        begin = num_vgprs + int(match.group(1))
        if match.group(2) is None:
            return f"v[{begin}]"
        end = num_vgprs + int(match.group(2))
        return f"v[{begin}:{end}]"

    transformed_lines = []
    for line in isa.splitlines(keepends=True):
        code, separator, comment = line.partition(";")
        transformed_lines.append(
            operand_pattern.sub(replace_operand, code) + separator + comment
        )
    transformed = "".join(transformed_lines)
    replacements = {
        rf"(^\s*\.set\s+{re.escape(symbol)}\.num_vgpr,\s*){num_vgprs}(\s*$)": rf"\g<1>{total_vgprs}\g<2>",
        rf"(^\s*\.set\s+{re.escape(symbol)}\.num_agpr,\s*){num_agprs}(\s*$)": r"\g<1>0\g<2>",
        rf"(^\s*\.amdhsa_accum_offset\s+){num_vgprs}(\s*$)": rf"\g<1>{total_vgprs}\g<2>",
        rf"(^\s*- \.agpr_count:\s*){num_agprs}(\s*$)": r"\g<1>0\g<2>",
        rf"(^; NumVgprs:\s*){num_vgprs}(\s*$)": rf"\g<1>{total_vgprs}\g<2>",
        rf"(^; NumAgprs:\s*){num_agprs}(\s*$)": r"\g<1>0\g<2>",
        rf"(^; AccumOffset:\s*){num_vgprs}(\s*$)": rf"\g<1>{total_vgprs}\g<2>",
    }
    for pattern, replacement in replacements.items():
        transformed, count = re.subn(
            pattern, replacement, transformed, flags=re.MULTILINE
        )
        if count != 1:
            raise ValueError(
                f"expected exactly one resource metadata match for {pattern!r}, found {count}"
            )

    if any(
        operand_pattern.search(line.split(";", 1)[0])
        for line in transformed.splitlines()
    ):
        raise AssertionError("AGPR operands remain after mechanical remapping")
    return transformed


def convert_jit_attention_to_fly_abi(isa):
    """将四指针JIT kernel改为Fly tensor参数的164-byte kernarg槽位。"""
    code_replacements = {
        r"(^[ \t]*s_load_dwordx2[ \t]+s\[6:7\],[ \t]*s\[0:1\],[ \t]*)0x0([ \t]+)": r"\g<1>0x0\g<2>",
        r"(^[ \t]*s_load_dwordx2[ \t]+s\[8:9\],[ \t]*s\[0:1\],[ \t]*)0x8([ \t]+)": r"\g<1>0x28\g<2>",
        r"(^[ \t]*s_load_dwordx2[ \t]+s\[10:11\],[ \t]*s\[0:1\],[ \t]*)0x10([ \t]+)": r"\g<1>0x50\g<2>",
        r"(^[ \t]*s_load_dwordx2[ \t]+s\[12:13\],[ \t]*s\[0:1\],[ \t]*)0x18([ \t]+)": r"\g<1>0x80\g<2>",
        r"(^[ \t]*\.amdhsa_kernarg_size[ \t]+)32([ \t]*$)": r"\g<1>164\g<2>",
    }
    transformed = isa
    for pattern, replacement in code_replacements.items():
        transformed, count = re.subn(
            pattern, replacement, transformed, flags=re.MULTILINE
        )
        if count != 1:
            raise ValueError(
                f"expected exactly one JIT ABI match for {pattern!r}, found {count}"
            )

    metadata_marker = "\t.amdgpu_metadata\n"
    if transformed.count(metadata_marker) != 1:
        raise ValueError("expected exactly one AMDGPU metadata document")
    prefix, metadata = transformed.split(metadata_marker, 1)
    metadata_replacements = {
        r"(^[ \t]*(?:-[ \t]+)?\.offset:[ \t]*)0([ \t]*$)": r"\g<1>0\g<2>",
        r"(^[ \t]*(?:-[ \t]+)?\.offset:[ \t]*)8([ \t]*$)": r"\g<1>40\g<2>",
        r"(^[ \t]*(?:-[ \t]+)?\.offset:[ \t]*)16([ \t]*$)": r"\g<1>80\g<2>",
        r"(^[ \t]*(?:-[ \t]+)?\.offset:[ \t]*)24([ \t]*$)": r"\g<1>128\g<2>",
        r"(^[ \t]*\.kernarg_segment_size:[ \t]*)32([ \t]*$)": r"\g<1>164\g<2>",
        r"(^[ \t]*\.max_flat_workgroup_size:[ \t]*)1024([ \t]*$)": r"\g<1>256\g<2>",
    }
    for pattern, replacement in metadata_replacements.items():
        metadata, count = re.subn(pattern, replacement, metadata, flags=re.MULTILINE)
        if count != 1:
            raise ValueError(
                f"expected exactly one metadata ABI match for {pattern!r}, found {count}"
            )
    return prefix + metadata_marker + metadata


def parse_priority_events(value, *, period=ATTN_MFMA_PERIOD):
    if value is None or not value.strip():
        return ATTN_PRIORITY_EVENTS
    try:
        events = tuple(
            (int(position), int(priority))
            for item in value.split(",")
            for position, priority in (item.split(":", 1),)
        )
    except ValueError as error:
        raise ValueError(
            f"priority events must use '<MFMA>:<priority>,...', got {value!r}"
        ) from error
    _validate_priority_events(period, events)
    return events


def _instruction_opcode(line):
    text = line.strip()
    if not text or text.startswith((".", "#", ";")) or text.endswith(":"):
        return ""
    return text.split(None, 1)[0]


def _instruction_text(line):
    text = line.split(";", 1)[0].strip()
    if not text or text.startswith((".", "#")) or text.endswith(":"):
        return None
    return re.sub(r"\s+", " ", text)


def _replace_instruction_sequence(isa, old, new, *, description):
    lines = isa.splitlines(keepends=True)
    instruction_indices = []
    instructions = []
    for index, line in enumerate(lines):
        instruction = _instruction_text(line)
        if instruction is not None:
            instruction_indices.append(index)
            instructions.append(instruction)

    matches = [
        index
        for index in range(len(instructions) - len(old) + 1)
        if instructions[index : index + len(old)] == list(old)
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one {description} sequence, found {len(matches)}")

    match = matches[0]
    first_line = instruction_indices[match]
    last_line = instruction_indices[match + len(old) - 1]
    original = lines[first_line]
    indent = original[: len(original) - len(original.lstrip())]
    newline = "\r\n" if original.endswith("\r\n") else "\n"
    lines[first_line : last_line + 1] = [
        f"{indent}{instruction}{newline}" for instruction in new
    ]
    return "".join(lines)


def insert_fly_attention_max_fanout(isa):
    """将Fly attention的两级max归约改为xor16/xor32/xor48并行fanout。"""
    validate_attention_isa(isa)
    isa = _replace_instruction_sequence(
        isa,
        ["v_xor_b32_e32 v96, 0x80, v4"],
        ["v_xor_b32_e32 v96, 0x80, v4", "v_xor_b32_e32 v240, 0x40, v96"],
        description="xor48 address setup",
    )
    isa = _replace_instruction_sequence(
        isa,
        [
            "ds_swizzle_b32 v83, v81 offset:swizzle(SWAP,16)",
            "s_waitcnt lgkmcnt(1)",
            "v_max_f32_e32 v79, v79, v79",
            "v_max_f32_e32 v78, v78, v79",
            "ds_bpermute_b32 v79, v96, v78",
            "s_waitcnt lgkmcnt(1)",
            "v_max_f32_e32 v83, v83, v83",
            "v_max_f32_e32 v81, v81, v83",
            "ds_bpermute_b32 v83, v96, v81",
            "s_waitcnt lgkmcnt(1)",
            "v_max_f32_e32 v79, v79, v79",
            "v_max_f32_e32 v200, v78, v79",
            "v_pk_add_f32 v[78:79], v[76:77], s[18:19] op_sel_hi:[1,0]",
            "s_waitcnt lgkmcnt(0)",
            "v_max_f32_e32 v83, v83, v83",
            "v_cmp_gt_f32_e32 vcc, v200, v79",
            "v_max_f32_e32 v201, v81, v83",
            "v_cmp_gt_f32_e64 s[2:3], v201, v78",
        ],
        [
            "ds_swizzle_b32 v83, v81 offset:swizzle(SWAP,16)",
            "ds_bpermute_b32 v200, v96, v78",
            "ds_bpermute_b32 v202, v240, v78",
            "ds_bpermute_b32 v201, v96, v81",
            "ds_bpermute_b32 v203, v240, v81",
            "s_waitcnt lgkmcnt(0)",
            "v_max3_f32 v200, v78, v79, v200",
            "v_max_f32_e32 v200, v200, v202",
            "v_max3_f32 v201, v81, v83, v201",
            "v_max_f32_e32 v201, v201, v203",
            "v_pk_add_f32 v[78:79], v[76:77], s[18:19] op_sel_hi:[1,0]",
            "v_cmp_gt_f32_e32 vcc, v200, v79",
            "v_cmp_gt_f32_e64 s[2:3], v201, v78",
        ],
        description="first max fanout",
    )
    isa = _replace_instruction_sequence(
        isa,
        [
            "ds_swizzle_b32 v113, v112 offset:swizzle(SWAP,16)",
            "s_waitcnt lgkmcnt(1)",
            "v_max_f32_e32 v77, v77, v77",
            "v_max_f32_e32 v76, v76, v77",
            "ds_bpermute_b32 v77, v96, v76",
            "s_waitcnt lgkmcnt(1)",
            "v_max_f32_e32 v113, v113, v113",
            "v_max_f32_e32 v112, v112, v113",
            "ds_bpermute_b32 v113, v96, v112",
            "s_waitcnt lgkmcnt(1)",
            "v_max_f32_e32 v77, v77, v77",
            "v_max_f32_e32 v202, v76, v77",
            "v_pk_add_f32 v[76:77], v[78:79], s[18:19] op_sel_hi:[1,0]",
            "s_waitcnt lgkmcnt(0)",
            "v_max_f32_e32 v113, v113, v113",
            "v_cmp_gt_f32_e32 vcc, v202, v77",
            "v_max_f32_e32 v203, v112, v113",
            "v_cmp_gt_f32_e64 s[2:3], v203, v76",
        ],
        [
            "ds_swizzle_b32 v113, v112 offset:swizzle(SWAP,16)",
            "ds_bpermute_b32 v202, v96, v76",
            "ds_bpermute_b32 v241, v240, v76",
            "ds_bpermute_b32 v203, v96, v112",
            "ds_bpermute_b32 v242, v240, v112",
            "s_waitcnt lgkmcnt(0)",
            "v_max3_f32 v202, v76, v77, v202",
            "v_max_f32_e32 v202, v202, v241",
            "v_max3_f32 v203, v112, v113, v203",
            "v_max_f32_e32 v203, v203, v242",
            "v_pk_add_f32 v[76:77], v[78:79], s[18:19] op_sel_hi:[1,0]",
            "v_cmp_gt_f32_e32 vcc, v202, v77",
            "v_cmp_gt_f32_e64 s[2:3], v203, v76",
        ],
        description="second max fanout",
    )

    replacements = {
        r"(^\s*\.amdhsa_next_free_vgpr\s+)240(\s*$)": r"\g<1>244\g<2>",
        r"(^\s*\.amdhsa_accum_offset\s+)240(\s*$)": r"\g<1>244\g<2>",
        r"(^\s*\.set\s+attn_kernel_0\.num_vgpr,\s*)240(\s*$)": r"\g<1>244\g<2>",
        r"(^\s*- \.vgpr_count:\s*)240(\s*$)": r"\g<1>244\g<2>",
    }
    for pattern, replacement in replacements.items():
        isa, count = re.subn(pattern, replacement, isa, flags=re.MULTILINE)
        if count != 1:
            raise ValueError(
                f"expected one max-fanout resource match for {pattern!r}, found {count}"
            )
    return isa


def _validate_priority_events(period, events):
    if period <= 0:
        raise ValueError(f"MFMA period must be positive, got {period}")

    previous_after = 0
    for after, priority in events:
        if not previous_after < after <= period:
            raise ValueError(
                f"priority event positions must increase within one period: {events}"
            )
        if not 0 <= priority <= 15:
            raise ValueError(f"priority must be in [0, 15], got {priority}")
        previous_after = after


def validate_attention_isa(isa, expected_mfma=ATTN_MFMA_COUNT):
    """校验当前Fly attention机器码形态，避免把插桩静默应用到不同kernel。"""
    if "amdgcn-amd-amdhsa--gfx942" not in isa:
        raise ValueError("expected a gfx942 AMDGPU assembly")
    if ".name:           attn_kernel_0" not in isa:
        raise ValueError("expected attn_kernel_0 metadata")
    if re.search(r"^\s*s_setprio\b", isa, re.MULTILINE):
        raise ValueError("assembly already contains s_setprio")

    mfma_count = sum(
        _instruction_opcode(line).startswith("v_mfma_") for line in isa.splitlines()
    )
    if mfma_count != expected_mfma:
        raise ValueError(
            f"expected {expected_mfma} MFMA instructions, found {mfma_count}"
        )

    kernarg_match = re.search(r"\.kernarg_segment_size:\s*(\d+)", isa)
    if kernarg_match is None or int(kernarg_match.group(1)) != ATTN_KERNARG_SIZE:
        actual = None if kernarg_match is None else int(kernarg_match.group(1))
        raise ValueError(
            f"expected {ATTN_KERNARG_SIZE}-byte kernarg segment, found {actual}"
        )

    load_pattern = re.compile(
        r"^\s*s_load_dwordx2\s+[^,]+,\s*s\[0:1\],\s*(0x[0-9a-fA-F]+|\d+)\s*$",
        re.MULTILINE,
    )
    load_offsets = tuple(int(value, 0) for value in load_pattern.findall(isa))
    if load_offsets != ATTN_POINTER_OFFSETS:
        raise ValueError(
            f"expected tensor pointer loads at {ATTN_POINTER_OFFSETS}, found {load_offsets}"
        )


def remove_fly_attention_identity_max_spacers(isa, *, expected_mfma=ATTN_MFMA_COUNT):
    """删除max归约中已有wait覆盖的8条恒等VALU spacer。"""
    validate_attention_isa(isa, expected_mfma=expected_mfma)
    pattern = re.compile(r"v_max_f32_e32 v(\d+), v\1, v\1")
    output = []
    observed = []
    for line in isa.splitlines(keepends=True):
        instruction = _instruction_text(line)
        match = None if instruction is None else pattern.fullmatch(instruction)
        if match is None:
            output.append(line)
            continue
        observed.append(int(match.group(1)))

    if tuple(observed) != ATTN_IDENTITY_MAX_SPACERS:
        raise ValueError(
            "expected identity max spacer registers "
            f"{ATTN_IDENTITY_MAX_SPACERS}, found {tuple(observed)}"
        )

    transformed = "".join(output)
    validate_attention_isa(transformed, expected_mfma=expected_mfma)
    return transformed


def _move_instruction_subset_after(
    isa, source_sequence, moved_offsets, anchor, *, description
):
    lines = isa.splitlines(keepends=True)
    instruction_indices = []
    instructions = []
    for line_index, line in enumerate(lines):
        instruction = _instruction_text(line)
        if instruction is not None:
            instruction_indices.append(line_index)
            instructions.append(instruction)

    source_matches = [
        index
        for index in range(len(instructions) - len(source_sequence) + 1)
        if instructions[index : index + len(source_sequence)] == list(source_sequence)
    ]
    if len(source_matches) != 1:
        raise ValueError(
            f"expected one {description} source sequence, found {len(source_matches)}"
        )
    anchor_matches = [
        instruction_indices[index]
        for index, instruction in enumerate(instructions)
        if instruction == anchor
    ]
    if len(anchor_matches) != 1:
        raise ValueError(
            f"expected one {description} destination anchor, found {len(anchor_matches)}"
        )

    source_begin = source_matches[0]
    moved_offsets = tuple(moved_offsets)
    if len(set(moved_offsets)) != len(moved_offsets) or any(
        offset < 0 or offset >= len(source_sequence) for offset in moved_offsets
    ):
        raise ValueError(f"invalid {description} moved offsets: {moved_offsets}")
    remove_lines = {
        instruction_indices[source_begin + offset] for offset in moved_offsets
    }
    moved_instructions = [source_sequence[offset] for offset in moved_offsets]
    anchor_line = anchor_matches[0]
    anchor_text = lines[anchor_line]
    indent = anchor_text[: len(anchor_text) - len(anchor_text.lstrip())]
    newline = "\r\n" if anchor_text.endswith("\r\n") else "\n"

    output = []
    for line_index, line in enumerate(lines):
        if line_index in remove_lines:
            continue
        output.append(line)
        if line_index == anchor_line:
            output.extend(
                f"{indent}{instruction}{newline}" for instruction in moved_instructions
            )
    return "".join(output)


def move_fly_attention_sum_pack_to_wait(isa, *, expected_mfma=ATTN_MFMA_COUNT):
    """将每个展开步的一组BF16概率打包移入sum DS等待窗口。"""
    validate_attention_isa(isa, expected_mfma=expected_mfma)
    first_conversion = (
        "v_add_u32_e32 v115, 0x8000, v115",
        "v_add_u32_e32 v116, 0x8000, v116",
        "v_add_u32_e32 v117, 0x8000, v117",
        "v_add_u32_e32 v76, 0x8000, v111",
        "v_add_u32_e32 v111, 0x8000, v112",
        "v_add_u32_e32 v77, 0x8000, v113",
        "v_add_u32_e32 v112, 0x8000, v114",
        "v_add_u32_e32 v110, 0x8000, v110",
        "v_perm_b32 v77, v112, v77, s15",
        "v_perm_b32 v76, v111, v76, s15",
        "v_perm_b32 v119, v110, v117, s15",
        "v_perm_b32 v118, v116, v115, s15",
    )
    second_conversion = (
        "v_add_u32_e32 v200, 0x8000, v117",
        "v_add_u32_e32 v118, 0x8000, v118",
        "v_add_u32_e32 v117, 0x8000, v119",
        "v_add_u32_e32 v113, 0x8000, v113",
        "v_add_u32_e32 v114, 0x8000, v114",
        "v_add_u32_e32 v115, 0x8000, v115",
        "v_add_u32_e32 v116, 0x8000, v116",
        "v_add_u32_e32 v112, 0x8000, v112",
        "v_perm_b32 v115, v116, v115, s15",
        "v_perm_b32 v114, v114, v113, s15",
        "v_perm_b32 v117, v112, v117, s15",
        "v_perm_b32 v116, v118, v200, s15",
    )
    transformed = _move_instruction_subset_after(
        isa,
        first_conversion,
        (0, 1, 2, 7, 10, 11),
        "ds_swizzle_b32 v72, v70 offset:swizzle(SWAP,16)",
        description="first sum-pack",
    )
    transformed = _move_instruction_subset_after(
        transformed,
        second_conversion,
        (3, 4, 5, 6, 8, 9),
        "ds_swizzle_b32 v82, v80 offset:swizzle(SWAP,16)",
        description="second sum-pack",
    )
    validate_attention_isa(transformed, expected_mfma=expected_mfma)
    return transformed


def insert_periodic_mfma_priority(
    isa,
    *,
    period=ATTN_MFMA_PERIOD,
    events=ATTN_PRIORITY_EVENTS,
    expected_mfma=ATTN_MFMA_COUNT,
):
    """在每个MFMA周期的指定指令后插入`s_setprio`，不改变其他机器指令。"""
    events = tuple(events)
    _validate_priority_events(period, events)
    validate_attention_isa(isa, expected_mfma=expected_mfma)
    if expected_mfma % period != 0:
        raise ValueError(
            f"expected MFMA count {expected_mfma} is not divisible by period {period}"
        )

    priority_by_phase = dict(events)
    output = []
    mfma_count = 0
    for line in isa.splitlines(keepends=True):
        output.append(line)
        if not _instruction_opcode(line).startswith("v_mfma_"):
            continue

        mfma_count += 1
        phase = (mfma_count - 1) % period + 1
        if phase not in priority_by_phase:
            continue

        newline = "\r\n" if line.endswith("\r\n") else "\n"
        indent = line[: len(line) - len(line.lstrip())]
        output.append(f"{indent}s_setprio {priority_by_phase[phase]}{newline}")

    if mfma_count != expected_mfma:
        raise AssertionError("MFMA count changed while inserting priority events")
    return "".join(output)


def insert_absolute_mfma_priority(isa, *, events, expected_mfma=ATTN_MFMA_COUNT):
    """在整段机器码的绝对MFMA编号后插入priority事件。"""
    events = tuple(events)
    _validate_priority_events(expected_mfma, events)
    validate_attention_isa(isa, expected_mfma=expected_mfma)
    priority_by_position = dict(events)
    output = []
    mfma_count = 0
    for line in isa.splitlines(keepends=True):
        output.append(line)
        if not _instruction_opcode(line).startswith("v_mfma_"):
            continue
        mfma_count += 1
        if mfma_count not in priority_by_position:
            continue
        newline = "\r\n" if line.endswith("\r\n") else "\n"
        indent = line[: len(line) - len(line.lstrip())]
        output.append(f"{indent}s_setprio {priority_by_position[mfma_count]}{newline}")
    if mfma_count != expected_mfma:
        raise AssertionError(
            "MFMA count changed while inserting absolute priority events"
        )
    return "".join(output)


@dataclass(frozen=True)
class PriorityHsaco:
    assembly_path: Path
    code_object_path: Path


def _find_amdgpu_clang():
    configured = os.environ.get("PYHIP_AMDGPU_CLANG")
    candidates = [configured, "/opt/rocm/llvm/bin/clang++", shutil.which("clang++")]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return candidate
    raise FileNotFoundError("could not find an AMDGPU-capable clang++ assembler")


def build_priority_hsaco(
    source_isa_path,
    output_dir,
    *,
    arch="gfx942",
    period=ATTN_MFMA_PERIOD,
    events=ATTN_PRIORITY_EVENTS,
):
    source_isa_path = Path(source_isa_path)
    output_dir = Path(output_dir)
    source = source_isa_path.read_text(encoding="utf-8")
    transformed = insert_periodic_mfma_priority(source, period=period, events=events)

    digest = hashlib.sha256()
    digest.update(transformed.encode("utf-8"))
    digest.update(arch.encode("ascii"))
    key = digest.hexdigest()[:16]
    output_dir.mkdir(parents=True, exist_ok=True)
    assembly_path = output_dir / f"attn_kernel_0-priority-{key}.s"
    code_object_path = output_dir / f"attn_kernel_0-priority-{key}.co"

    if (
        not assembly_path.exists()
        or assembly_path.read_text(encoding="utf-8") != transformed
    ):
        assembly_path.write_text(transformed, encoding="utf-8")

    if not code_object_path.exists():
        command = [
            _find_amdgpu_clang(),
            "-x",
            "assembler",
            "-target",
            "amdgcn-amd-amdhsa",
            f"-mcpu={arch}",
            str(assembly_path),
            "-o",
            str(code_object_path),
        ]
        subprocess.run(command, check=True, text=True, capture_output=True)

    return PriorityHsaco(assembly_path=assembly_path, code_object_path=code_object_path)


def build_attention_post_isa_hsaco(
    source_isa_path,
    output_dir,
    *,
    arch="gfx942",
    remove_identity_max=False,
    move_sum_pack=False,
    priority_period=None,
    priority_events=None,
    absolute_priority_events=None,
):
    source_isa_path = Path(source_isa_path)
    output_dir = Path(output_dir)
    transformed = source_isa_path.read_text(encoding="utf-8")
    transforms = []
    if remove_identity_max:
        transformed = remove_fly_attention_identity_max_spacers(transformed)
        transforms.append("no-identity-max")
    if move_sum_pack:
        transformed = move_fly_attention_sum_pack_to_wait(transformed)
        transforms.append("sum-pack")
    if priority_events is not None and absolute_priority_events is not None:
        raise ValueError("periodic and absolute priority events are mutually exclusive")
    if priority_events is not None:
        if priority_period is None:
            raise ValueError("priority_period is required with priority_events")
        transformed = insert_periodic_mfma_priority(
            transformed, period=priority_period, events=priority_events
        )
        transforms.append("priority")
    if absolute_priority_events is not None:
        transformed = insert_absolute_mfma_priority(
            transformed, events=absolute_priority_events
        )
        transforms.append("absolute-priority")
    if not transforms:
        raise ValueError("at least one post-ISA transform must be enabled")

    digest = hashlib.sha256()
    digest.update(transformed.encode("utf-8"))
    digest.update(arch.encode("ascii"))
    key = digest.hexdigest()[:16]
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = "attn_kernel_0-" + "-".join(transforms) + f"-{key}"
    assembly_path = output_dir / f"{stem}.s"
    code_object_path = output_dir / f"{stem}.co"

    if (
        not assembly_path.exists()
        or assembly_path.read_text(encoding="utf-8") != transformed
    ):
        assembly_path.write_text(transformed, encoding="utf-8")
    if not code_object_path.exists():
        subprocess.run(
            [
                _find_amdgpu_clang(),
                "-x",
                "assembler",
                "-target",
                "amdgcn-amd-amdhsa",
                f"-mcpu={arch}",
                str(assembly_path),
                "-o",
                str(code_object_path),
            ],
            check=True,
            text=True,
            capture_output=True,
        )
    return PriorityHsaco(assembly_path=assembly_path, code_object_path=code_object_path)


def build_max_fanout_hsaco(source_isa_path, output_dir, *, arch="gfx942"):
    source_isa_path = Path(source_isa_path)
    output_dir = Path(output_dir)
    transformed = insert_fly_attention_max_fanout(
        source_isa_path.read_text(encoding="utf-8")
    )
    digest = hashlib.sha256()
    digest.update(transformed.encode("utf-8"))
    digest.update(arch.encode("ascii"))
    key = digest.hexdigest()[:16]
    output_dir.mkdir(parents=True, exist_ok=True)
    assembly_path = output_dir / f"attn_kernel_0-max-fanout-{key}.s"
    code_object_path = output_dir / f"attn_kernel_0-max-fanout-{key}.co"
    if (
        not assembly_path.exists()
        or assembly_path.read_text(encoding="utf-8") != transformed
    ):
        assembly_path.write_text(transformed, encoding="utf-8")
    if not code_object_path.exists():
        subprocess.run(
            [
                _find_amdgpu_clang(),
                "-x",
                "assembler",
                "-target",
                "amdgcn-amd-amdhsa",
                f"-mcpu={arch}",
                str(assembly_path),
                "-o",
                str(code_object_path),
            ],
            check=True,
            text=True,
            capture_output=True,
        )
    return PriorityHsaco(assembly_path=assembly_path, code_object_path=code_object_path)


def build_all_vgpr_hsaco(source_isa_path, output_dir, *, arch="gfx942"):
    source_isa_path = Path(source_isa_path)
    output_dir = Path(output_dir)
    transformed = convert_jit_attention_to_fly_abi(
        convert_accvgprs_to_vgprs(source_isa_path.read_text(encoding="utf-8"))
    )

    digest = hashlib.sha256()
    digest.update(transformed.encode("utf-8"))
    digest.update(arch.encode("ascii"))
    key = digest.hexdigest()[:16]
    output_dir.mkdir(parents=True, exist_ok=True)
    assembly_path = output_dir / f"attn_jit_setprio_best-all-vgpr-fly-abi-{key}.s"
    code_object_path = output_dir / f"attn_jit_setprio_best-all-vgpr-fly-abi-{key}.co"

    if (
        not assembly_path.exists()
        or assembly_path.read_text(encoding="utf-8") != transformed
    ):
        assembly_path.write_text(transformed, encoding="utf-8")

    if not code_object_path.exists():
        command = [
            _find_amdgpu_clang(),
            "-x",
            "assembler",
            "-target",
            "amdgcn-amd-amdhsa",
            f"-mcpu={arch}",
            str(assembly_path),
            "-o",
            str(code_object_path),
        ]
        subprocess.run(command, check=True, text=True, capture_output=True)

    return PriorityHsaco(assembly_path=assembly_path, code_object_path=code_object_path)


class StaticAttentionKernel:
    """启动只读取四个tensor基址的特化Fly attention code object。"""

    def __init__(
        self,
        code_object_path,
        *,
        grid,
        block=(256, 1, 1),
        symbol="attn_kernel_0",
        kernarg_size=ATTN_KERNARG_SIZE,
        pointer_offsets=ATTN_POINTER_OFFSETS,
    ):
        self.code_object_path = str(Path(code_object_path).resolve())
        self.grid = tuple(grid)
        self.block = tuple(block)
        self.kernarg_size = kernarg_size
        self.pointer_offsets = tuple(pointer_offsets)
        if len(self.grid) != 3 or len(self.block) != 3:
            raise ValueError("grid and block must contain exactly three dimensions")
        if len(self.pointer_offsets) != 4:
            raise ValueError("pointer_offsets must contain exactly four offsets")
        if (
            max(self.pointer_offsets) + ctypes.sizeof(ctypes.c_uint64)
            > self.kernarg_size
        ):
            raise ValueError("pointer offsets exceed the kernarg buffer")
        module = hipModuleLoad(self.code_object_path)
        self._module = module
        self._function = hipModuleGetFunction(module, symbol)

    def __call__(self, q, k, v, output, stream=None):
        kernarg = (ctypes.c_ubyte * self.kernarg_size)()
        for tensor, offset in zip((q, k, v, output), self.pointer_offsets):
            ctypes.c_uint64.from_buffer(kernarg, offset).value = tensor.data_ptr()

        kernarg_size = ctypes.c_uint64(self.kernarg_size)
        extra_type = ctypes.c_void_p * 5
        extra = extra_type(
            1,
            ctypes.addressof(kernarg),
            2,
            ctypes.addressof(kernarg_size),
            3,
        )
        if stream is None:
            stream = torch.cuda.current_stream()
        stream_handle = ctypes.c_void_p(stream.cuda_stream)
        hip_check_error(
            get_lib().hipModuleLaunchKernel(
                self._function,
                *self.grid,
                *self.block,
                0,
                stream_handle,
                0,
                ctypes.byref(extra),
            )
        )


def build_attention_priority_kernel(
    source_isa_path,
    output_dir,
    *,
    m,
    h,
    block_m=128,
    period=ATTN_MFMA_PERIOD,
    events=ATTN_PRIORITY_EVENTS,
):
    if m % block_m != 0:
        raise ValueError(f"M={m} must be divisible by block_m={block_m}")
    artifact = build_priority_hsaco(
        source_isa_path, output_dir, period=period, events=events
    )
    return (
        StaticAttentionKernel(artifact.code_object_path, grid=(m // block_m, h, 1)),
        artifact,
    )


def build_attention_post_isa_kernel(
    source_isa_path,
    output_dir,
    *,
    m,
    h,
    block_m=128,
    remove_identity_max=False,
    move_sum_pack=False,
    priority_period=None,
    priority_events=None,
    absolute_priority_events=None,
):
    if m % block_m != 0:
        raise ValueError(f"M={m} must be divisible by block_m={block_m}")
    artifact = build_attention_post_isa_hsaco(
        source_isa_path,
        output_dir,
        remove_identity_max=remove_identity_max,
        move_sum_pack=move_sum_pack,
        priority_period=priority_period,
        priority_events=priority_events,
        absolute_priority_events=absolute_priority_events,
    )
    kernel = StaticAttentionKernel(artifact.code_object_path, grid=(m // block_m, h, 1))
    return kernel, artifact


def build_all_vgpr_jit_attention_kernel(
    source_isa_path, output_dir, *, m, n, h, block_m=128
):
    if m != 40960 or n != 40960:
        raise ValueError(
            f"the archived JIT kernel is specialized for M=N=40960, got M={m}, N={n}"
        )
    artifact = build_all_vgpr_hsaco(source_isa_path, output_dir)
    kernel = StaticAttentionKernel(
        artifact.code_object_path,
        grid=(m // block_m, h, 1),
        symbol="_Z26attn_gemm_jit_setprio_bestPvS_S_S_",
    )
    return kernel, artifact
