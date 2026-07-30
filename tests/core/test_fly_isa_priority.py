import hashlib
import re
from collections import Counter
from pathlib import Path

import pytest
import torch

from pyhip.core.fly_isa_priority import (  # pyright: ignore[reportMissingImports]
    ATTN_IDENTITY_MAX_SPACERS,
    ATTN_PRIORITY_EVENTS,
    convert_accvgprs_to_vgprs,
    convert_jit_attention_to_fly_abi,
    insert_absolute_mfma_priority,
    insert_periodic_mfma_priority,
    move_fly_attention_sum_pack_to_wait,
    parse_priority_events,
    preshuffle_jit_key,
    preshuffle_jit_value,
    remove_fly_attention_identity_max_spacers,
    validate_attention_isa,
)


def _make_attention_isa(mfma_count=128):
    instructions = [
        '\t.amdgcn_target "amdgcn-amd-amdhsa--gfx942"',
        "attn_kernel_0:",
        "\ts_load_dwordx2 s[4:5], s[0:1], 0x0",
        "\ts_load_dwordx2 s[6:7], s[0:1], 0x28",
        "\ts_load_dwordx2 s[14:15], s[0:1], 0x50",
    ]
    instructions.extend(
        f"\tv_mfma_f32_16x16x16_bf16 v[0:3], v[4:5], v[6:7], v[0:3] ; {i}"
        for i in range(mfma_count)
    )
    instructions.extend(
        [
            "\ts_load_dwordx2 s[0:1], s[0:1], 0x80",
            "\t.kernarg_segment_size: 164",
            "\t.name:           attn_kernel_0",
        ]
    )
    return "\n".join(instructions) + "\n"


def test_insert_periodic_mfma_priority():
    transformed = insert_periodic_mfma_priority(_make_attention_isa())

    mfma_count = 0
    observed = []
    for line in transformed.splitlines():
        opcode = line.strip().split(None, 1)[0] if line.strip() else ""
        if opcode.startswith("v_mfma_"):
            mfma_count += 1
        elif opcode == "s_setprio":
            observed.append((mfma_count, int(line.split()[1])))

    assert observed == [(46, 2), (64, 0), (110, 2), (128, 0)]
    assert transformed.count("s_setprio") == 4
    assert ATTN_PRIORITY_EVENTS == ((46, 2), (64, 0))


def test_insert_absolute_mfma_priority():
    transformed = insert_absolute_mfma_priority(
        _make_attention_isa(), events=((16, 0), (96, 2))
    )

    mfma_count = 0
    observed = []
    for line in transformed.splitlines():
        opcode = line.strip().split(None, 1)[0] if line.strip() else ""
        if opcode.startswith("v_mfma_"):
            mfma_count += 1
        elif opcode == "s_setprio":
            observed.append((mfma_count, int(line.split()[1])))

    assert observed == [(16, 0), (96, 2)]


def test_validate_attention_isa_rejects_changed_machine_shape():
    with pytest.raises(ValueError, match="expected 128 MFMA instructions"):
        validate_attention_isa(_make_attention_isa(mfma_count=127))


def test_remove_fly_attention_identity_max_spacers():
    spacers = "".join(
        f"\tv_max_f32_e32 v{register}, v{register}, v{register}\n"
        for register in ATTN_IDENTITY_MAX_SPACERS
    )
    source = _make_attention_isa().replace(
        "\ts_load_dwordx2 s[0:1], s[0:1], 0x80\n",
        spacers
        + "\tv_max_f32_e32 v79, v78, v79\n"
        + "\ts_load_dwordx2 s[0:1], s[0:1], 0x80\n",
    )

    transformed = remove_fly_attention_identity_max_spacers(source)

    assert "v_max_f32_e32 v79, v79, v79" not in transformed
    assert "v_max_f32_e32 v79, v78, v79" in transformed
    assert transformed.count("v_mfma_f32_16x16x16_bf16") == 128


def test_remove_fly_attention_identity_max_spacers_rejects_shape_drift():
    source = _make_attention_isa().replace(
        "\ts_load_dwordx2 s[0:1], s[0:1], 0x80\n",
        "\tv_max_f32_e32 v79, v79, v79\n" "\ts_load_dwordx2 s[0:1], s[0:1], 0x80\n",
    )
    with pytest.raises(ValueError, match="expected identity max spacer registers"):
        remove_fly_attention_identity_max_spacers(source)


def test_move_fly_attention_sum_pack_to_wait():
    first_conversion = """\
	v_add_u32_e32 v115, 0x8000, v115
	v_add_u32_e32 v116, 0x8000, v116
	v_add_u32_e32 v117, 0x8000, v117
	v_add_u32_e32 v76, 0x8000, v111
	v_add_u32_e32 v111, 0x8000, v112
	v_add_u32_e32 v77, 0x8000, v113
	v_add_u32_e32 v112, 0x8000, v114
	v_add_u32_e32 v110, 0x8000, v110
	v_perm_b32 v77, v112, v77, s15
	v_perm_b32 v76, v111, v76, s15
	v_perm_b32 v119, v110, v117, s15
	v_perm_b32 v118, v116, v115, s15
"""
    second_conversion = """\
	v_add_u32_e32 v200, 0x8000, v117
	v_add_u32_e32 v118, 0x8000, v118
	v_add_u32_e32 v117, 0x8000, v119
	v_add_u32_e32 v113, 0x8000, v113
	v_add_u32_e32 v114, 0x8000, v114
	v_add_u32_e32 v115, 0x8000, v115
	v_add_u32_e32 v116, 0x8000, v116
	v_add_u32_e32 v112, 0x8000, v112
	v_perm_b32 v115, v116, v115, s15
	v_perm_b32 v114, v114, v113, s15
	v_perm_b32 v117, v112, v117, s15
	v_perm_b32 v116, v118, v200, s15
"""
    source = _make_attention_isa().replace(
        "\ts_load_dwordx2 s[0:1], s[0:1], 0x80\n",
        first_conversion
        + "\tds_swizzle_b32 v72, v70 offset:swizzle(SWAP,16)\n"
        + second_conversion
        + "\tds_swizzle_b32 v82, v80 offset:swizzle(SWAP,16)\n"
        + "\ts_load_dwordx2 s[0:1], s[0:1], 0x80\n",
    )

    transformed = move_fly_attention_sum_pack_to_wait(source)

    def instructions(text):
        result = []
        for line in text.splitlines():
            instruction = re.sub(r"\s+", " ", line.split(";", 1)[0].strip())
            if (
                instruction
                and not instruction.startswith((".", "#"))
                and not instruction.endswith(":")
            ):
                result.append(instruction)
        return result

    before = instructions(source)
    after = instructions(transformed)

    assert Counter(after) == Counter(before)
    first_anchor = after.index("ds_swizzle_b32 v72, v70 offset:swizzle(SWAP,16)")
    assert after[first_anchor + 1 : first_anchor + 7] == [
        "v_add_u32_e32 v115, 0x8000, v115",
        "v_add_u32_e32 v116, 0x8000, v116",
        "v_add_u32_e32 v117, 0x8000, v117",
        "v_add_u32_e32 v110, 0x8000, v110",
        "v_perm_b32 v119, v110, v117, s15",
        "v_perm_b32 v118, v116, v115, s15",
    ]
    second_anchor = after.index("ds_swizzle_b32 v82, v80 offset:swizzle(SWAP,16)")
    assert after[second_anchor + 1 : second_anchor + 7] == [
        "v_add_u32_e32 v113, 0x8000, v113",
        "v_add_u32_e32 v114, 0x8000, v114",
        "v_add_u32_e32 v115, 0x8000, v115",
        "v_add_u32_e32 v116, 0x8000, v116",
        "v_perm_b32 v115, v116, v115, s15",
        "v_perm_b32 v114, v114, v113, s15",
    ]


def test_move_fly_attention_sum_pack_to_wait_rejects_shape_drift():
    with pytest.raises(ValueError, match="first sum-pack source sequence"):
        move_fly_attention_sum_pack_to_wait(_make_attention_isa())


def test_insert_rejects_existing_priority():
    isa = _make_attention_isa().replace(
        "attn_kernel_0:\n", "attn_kernel_0:\n\ts_setprio 1\n"
    )
    with pytest.raises(ValueError, match="already contains s_setprio"):
        insert_periodic_mfma_priority(isa)


def test_parse_priority_events():
    assert parse_priority_events(None) == ATTN_PRIORITY_EVENTS
    assert parse_priority_events("7:1,46:0") == ((7, 1), (46, 0))
    assert parse_priority_events("16:0,40:1,80:0,104:1", period=128) == (
        (16, 0),
        (40, 1),
        (80, 0),
        (104, 1),
    )
    with pytest.raises(ValueError, match="must use"):
        parse_priority_events("7=1")


def test_convert_accvgprs_to_vgprs():
    isa = """\
	.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
	buffer_load_dwordx4 a[0:3], v0, s[0:3], 0 offen
	v_mfma_f32_16x16x16_bf16 v[0:3], a[60:61], v[4:5], v[0:3]
    v_mfma_f32_16x16x16_bf16 v[0:3], a[62:63], v[4:5], v[0:3]
    ; keep comment operand a[0:3]
	.amdhsa_accum_offset 156
	.set kernel.num_vgpr, 156
	.set kernel.num_agpr, 64
; NumVgprs: 156
; NumAgprs: 64
; AccumOffset: 156
    - .agpr_count:     64
"""
    transformed = convert_accvgprs_to_vgprs(isa)

    assert "buffer_load_dwordx4 v[156:159]" in transformed
    assert "v_mfma_f32_16x16x16_bf16 v[0:3], v[216:217]" in transformed
    assert "\t.amdhsa_accum_offset 220" in transformed
    assert "\t.set kernel.num_vgpr, 220" in transformed
    assert "\t.set kernel.num_agpr, 0" in transformed
    assert "; NumVgprs: 220" in transformed
    assert "; NumAgprs: 0" in transformed
    assert "; AccumOffset: 220" in transformed
    assert "  - .agpr_count:     0" in transformed
    assert "; keep comment operand a[0:3]" in transformed


def test_preshuffle_jit_inputs_preserve_elements():
    key = torch.arange(2 * 64 * 128).reshape(2, 64, 128)
    value = torch.arange(2 * 64 * 128).reshape(2, 64, 128)

    key_shuffled = preshuffle_jit_key(key)
    value_shuffled = preshuffle_jit_value(value)

    assert key_shuffled.shape == (2, 2, 2, 16, 16, 8)
    assert value_shuffled.shape == (2, 2, 8, 4, 16, 8)
    assert torch.equal(
        key_shuffled.flatten().sort().values, key.flatten().sort().values
    )
    assert torch.equal(
        value_shuffled.flatten().sort().values, value.flatten().sort().values
    )


def test_convert_jit_attention_to_fly_abi():
    isa = """\
	s_load_dwordx2 s[6:7],s[0:1],0x0  ; query
	s_load_dwordx2 s[8:9],s[0:1],0x8  ; key
	s_load_dwordx2 s[10:11],s[0:1],0x10  ; value
	s_load_dwordx2 s[12:13],s[0:1],0x18  ; output
	.amdhsa_kernarg_size 32
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
            - .address_space:  global
                .offset:         0
            - .address_space:  global
                .offset:         8
            - .address_space:  global
                .offset:         16
            - .address_space:  global
                .offset:         24
    .kernarg_segment_size: 32
    .max_flat_workgroup_size: 1024
...
"""
    transformed = convert_jit_attention_to_fly_abi(isa)

    assert "s[8:9],s[0:1],0x28" in transformed
    assert "s[10:11],s[0:1],0x50" in transformed
    assert "s[12:13],s[0:1],0x80" in transformed
    assert "\t.amdhsa_kernarg_size 164" in transformed
    assert [int(value) for value in re.findall(r"\.offset:\s*(\d+)", transformed)] == [
        0,
        40,
        80,
        128,
    ]
    assert ".kernarg_segment_size: 164" in transformed
    assert ".max_flat_workgroup_size: 256" in transformed


def test_convert_archived_attention_to_all_vgpr_fly_abi():
    root = Path(__file__).resolve().parents[2]
    source = (
        root / "archive/gemm/attn-gemm-jit-setprio-best-gfx942-m40960-n40960-237p1t.s"
    ).read_text()

    transformed = convert_jit_attention_to_fly_abi(convert_accvgprs_to_vgprs(source))
    code = "\n".join(line.split(";", 1)[0] for line in transformed.splitlines())

    assert "a[" not in code
    assert code.count("v_mfma_f32_16x16x16_bf16") == 128
    assert code.count("s_setprio") == 4
    assert ".amdhsa_next_free_vgpr 220" in transformed
    assert ".set _Z26attn_gemm_jit_setprio_bestPvS_S_S_.num_vgpr, 220" in transformed
    assert ".set _Z26attn_gemm_jit_setprio_bestPvS_S_S_.num_agpr, 0" in transformed
    assert ".amdhsa_kernarg_size 164" in transformed
    assert [int(value) for value in re.findall(r"\.offset:\s*(\d+)", transformed)] == [
        0,
        40,
        80,
        128,
    ]
    assert hashlib.sha256(transformed.encode()).hexdigest() == (
        "2759206039c58f8c14cac7749e3d8b591feb29485e7f0281b871d58c8d5ab2f9"
    )
