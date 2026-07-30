#!/usr/bin/env python3
"""测量attention VALU吞吐，以及MFMA/VALU的intra和inter co-issue容量。

默认外层运行时循环1000次，每个segment静态展开1000组：

* 吞吐：分别测量空循环和1,000,000条目标指令，二者相减后除以1,000,000。
* intra co-issue：一个wave内交替发射``MFMA + N * VALU``，workgroup为1 wave。
* inter co-issue：8-wave workgroup中，wave 0..3和wave 4..7通过条件barrier错开一个phase；
    同一SIMD上的wave 0/4分别执行MFMA segment和tested-op segment，然后交换角色。

两种co-issue均测试N=0..4，并区分：

* fully hidden by anchor：加入tested-op后总时间相对anchor不增加；
* full co-issue：总时间接近``max(anchor, tested-op stream)``；
* partial co-issue：总时间小于两段完全串行之和，但没有达到full co-issue。

新增测试指令只需在 ``VALU_TESTS`` 中增加一项，并使用已有寄存器字段。
"""

import argparse
import json
import statistics

import torch

from pyhip.core.asmjit import JIT, jit

DEFAULT_INNER_UNROLL = 1000
DEFAULT_OUTER_LOOPS = 1000
INTER_WAVES = 8
MFMA_TO_VALU_START_CYCLES = 4.0
MAX_VALU_COUNT = 4
VOID_POINTER = "void*"


def emit_none(jit, regs, slot):
    del jit, regs, slot


# 新增指令只需在这里增加一个 ``(opcode, lambda)``。
VALU_TESTS = [
    ("v_add_f32", lambda j, r, i: j.v_add_f32(r["dst"][i], r["src0"][i], r["src1"][i])),
    (
        "v_add_f32_e64",
        lambda j, r, i: j.v_add_f32_e64(r["dst"][i], r["src0"][i], r["src1"][i]),
    ),
    ("v_sub_f32", lambda j, r, i: j.v_sub_f32(r["dst"][i], r["src0"][i], r["src1"][i])),
    ("v_mul_f32", lambda j, r, i: j.v_mul_f32(r["dst"][i], r["src0"][i], r["src1"][i])),
    (
        "v_mul_f32_e64",
        lambda j, r, i: j.v_mul_f32_e64(r["dst"][i], r["src0"][i], r["src1"][i]),
    ),
    (
        "v_fma_f32",
        lambda j, r, i: j.v_fma_f32(
            r["dst"][i], r["src0"][i], r["src1"][i], r["src2"][i]
        ),
    ),
    (
        "v_fmac_f32",
        lambda j, r, i: j.v_fmac_f32(r["dst"][i], r["src0"][i], r["src1"][i]),
    ),
    ("v_max_f32", lambda j, r, i: j.v_max_f32(r["dst"][i], r["src0"][i], r["src1"][i])),
    (
        "v_max3_f32",
        lambda j, r, i: j.v_max3_f32(
            r["dst"][i], r["src0"][i], r["src1"][i], r["src2"][i]
        ),
    ),
    ("v_exp_f32", lambda j, r, i: j.v_exp_f32(r["dst"][i], r["src0"][i])),
    ("v_rcp_f32", lambda j, r, i: j.v_rcp_f32(r["dst"][i], r["src0"][i])),
    (
        "v_pk_add_f32",
        lambda j, r, i: j.v_pk_add_f32(
            r["pk_dst"][i], r["pk_src0"][i], r["pk_src1"][i]
        ),
    ),
    (
        "v_pk_mul_f32",
        lambda j, r, i: j.v_pk_mul_f32(
            r["pk_dst"][i], r["pk_src0"][i], r["pk_src1"][i]
        ),
    ),
    (
        "v_cmp_gt_f32",
        lambda j, r, i: j.v_cmp_gt_f32_e32("vcc", r["src0"][i], r["src1"][i]),
    ),
    (
        "v_cndmask_b32",
        lambda j, r, i: j.v_cndmask_b32_e32(
            r["u32_dst"][i], r["u32_src0"][i], r["u32_src1"][i], "vcc"
        ),
    ),
    (
        "v_add_u32",
        lambda j, r, i: j.v_add_u32(
            r["u32_dst"][i], r["u32_src0"][i], r["u32_src1"][i]
        ),
    ),
    (
        "v_perm_b32",
        lambda j, r, i: j.v_perm_b32(
            r["u32_dst"][i], r["u32_src0"][i], r["u32_src1"][i], r["perm_sel"]
        ),
    ),
]
VALU_EMITTERS = dict(VALU_TESTS)


def make_registers(jit, register_chains):
    regs = {
        "src0": jit.gpr(register_chains, "vf32", 0.25, align=4),
        "src1": jit.gpr(register_chains, "vf32", 0.5, align=4),
        "src2": jit.gpr(register_chains, "vf32", 0.75, align=4),
        "dst": jit.gpr(register_chains, "vf32", 1.0, align=4),
        "exp_src": jit.gpr(register_chains, "vf32", 0.25, align=4),
        "exp_dst": jit.gpr(register_chains, "vf32", 1.0, align=4),
        "pk_src0": jit.gpr(register_chains, 2, "vf32", 0.25, align=2),
        "pk_src1": jit.gpr(register_chains, 2, "vf32", 0.5, align=2),
        "pk_dst": jit.gpr(register_chains, 2, "vf32", 1.0, align=2),
        "u32_src0": jit.gpr(register_chains, "vu32", 0x12345678, align=4),
        "u32_src1": jit.gpr(register_chains, "vu32", 0x89ABCDEF, align=4),
        "u32_dst": jit.gpr(register_chains, "vu32", 1, align=4),
        "perm_sel": jit.get_sgpr_const(0x05040100),
    }
    jit.SetMask("vcc", True)
    return regs


def read_clock(jit):
    value = jit.gpr(2, "su32", align=2)
    jit.s_memtime(value)
    return value


def store_elapsed(jit, start, output, byte_offset):
    stop = read_clock(jit)
    jit.s_waitcnt(mod="lgkmcnt(0)")
    jit.s_sub_u32(stop[0], stop[0], start[0])
    jit.s_subb_u32(stop[1], stop[1], start[1])
    jit.s_store_dwordx2(stop, output, byte_offset, mod="glc")


def make_mfma_registers(jit):
    if jit.gfx >= 950:
        mfma_a = jit.gpr(8, "vu32", align=2)
        mfma_b = jit.gpr(8, "vu32", align=2)
        mfma_a[...] = 0x01010101
        mfma_b[...] = 0x01010101
    else:
        mfma_a = jit.gpr(2, "vu32", align=2)
        mfma_b = jit.gpr(2, "vu32", align=2)
        mfma_a[...] = 0x3F803F80
        mfma_b[...] = 0x3F803F80
    mfma_d = jit.gpr(4, 4, "vf32", align=4)
    mfma_d[...] = 0.0
    return mfma_d, mfma_a, mfma_b


def emit_mfma(jit, mfma_d, mfma_a, mfma_b, slot):
    if jit.gfx >= 950:
        jit.v_mfma_f32_16x16x128_f8f6f4(mfma_d[slot], mfma_a, mfma_b, 0)
    else:
        jit.v_mfma_f32_16x16x16_bf16(mfma_d[slot], mfma_a, mfma_b, 0)


def emit_exp(jit, regs, slot):
    jit.v_exp_f32(regs["exp_dst"][slot], regs["exp_src"][slot])


def emit_alignment_padding(jit, alignment_nops):
    for _ in range(alignment_nops):
        jit.s_nop(0)


@jit(no_pass=["pass_dse", "pass_dce"])
def measure_instruction(
    jit: JIT,
    opcode,
    outer_loops,
    inner_unroll,
    register_chains,
    alignment_nops,
    output: VOID_POINTER,  # pyright: ignore[reportInvalidTypeForm]
):
    assert opcode in VALU_EMITTERS or opcode == "none"
    emit = emit_none if opcode == "none" else VALU_EMITTERS[opcode]
    regs = make_registers(jit, register_chains)

    emit_alignment_padding(jit, alignment_nops)
    start = read_clock(jit)
    loop = jit.gpr("su32", 0)
    with jit.While(loop[0] < outer_loops):
        for index in range(inner_unroll):
            emit(jit, regs, index % register_chains)
        loop[0] += 1
    store_elapsed(jit, start, output, 0)

    sink = jit.gpr("vf32")
    jit.v_add_f32(sink, regs["dst"][0], regs["pk_dst"][0, 0])
    sink_s = jit.gpr("su32")
    jit.v_readfirstlane_b32(sink_s, sink)
    jit.s_store_dword(sink_s, output, 8, mod="glc")


@jit(no_pass=["pass_dse", "pass_dce"])
def measure_coissue(
    jit: JIT,
    opcode,
    valu_count,
    outer_loops,
    inner_unroll,
    register_chains,
    alignment_nops,
    output: VOID_POINTER,  # pyright: ignore[reportInvalidTypeForm]
):
    assert opcode in VALU_EMITTERS
    assert 0 <= valu_count <= MAX_VALU_COUNT
    emit = VALU_EMITTERS[opcode]
    regs = make_registers(jit, register_chains)

    mfma_d, mfma_a, mfma_b = make_mfma_registers(jit)

    emit_alignment_padding(jit, alignment_nops)
    start = read_clock(jit)
    loop = jit.gpr("su32", 0)
    with jit.While(loop[0] < outer_loops):
        for index in range(inner_unroll):
            slot = index % register_chains
            emit_mfma(jit, mfma_d, mfma_a, mfma_b, slot)
            for valu_slot in range(valu_count):
                emit(jit, regs, (slot + valu_slot) % register_chains)
        loop[0] += 1
    store_elapsed(jit, start, output, 0)

    sink = jit.gpr("vf32")
    jit.v_add_f32(sink, mfma_d[0, 0], regs["dst"][0])
    jit.v_add_f32(sink, sink, regs["pk_dst"][0, 0])
    sink_s = jit.gpr("su32")
    jit.v_readfirstlane_b32(sink_s, sink)
    jit.s_store_dword(sink_s, output, 8, mod="glc")


@jit(no_pass=["pass_dse", "pass_dce"])
def measure_exp_coissue(
    jit: JIT,
    opcode,
    valu_count,
    outer_loops,
    inner_unroll,
    register_chains,
    alignment_nops,
    output: VOID_POINTER,  # pyright: ignore[reportInvalidTypeForm]
):
    """测量同一wave内EXP anchor与N条tested-op的intra co-issue。"""
    assert opcode in VALU_EMITTERS
    assert 0 <= valu_count <= MAX_VALU_COUNT
    emit = VALU_EMITTERS[opcode]
    regs = make_registers(jit, register_chains)

    emit_alignment_padding(jit, alignment_nops)
    start = read_clock(jit)
    loop = jit.gpr("su32", 0)
    with jit.While(loop[0] < outer_loops):
        for index in range(inner_unroll):
            slot = index % register_chains
            emit_exp(jit, regs, slot)
            for valu_slot in range(valu_count):
                emit(jit, regs, (slot + valu_slot) % register_chains)
        loop[0] += 1
    store_elapsed(jit, start, output, 0)

    sink = jit.gpr("vf32")
    jit.v_add_f32(sink, regs["exp_dst"][0], regs["dst"][0])
    jit.v_add_f32(sink, sink, regs["pk_dst"][0, 0])
    sink_s = jit.gpr("su32")
    jit.v_readfirstlane_b32(sink_s, sink)
    jit.s_store_dword(sink_s, output, 8, mod="glc")


@jit(no_pass=["pass_dse", "pass_dce"])
def measure_mfma_exp_alu_bundle(
    jit: JIT,
    alu_count,
    exp_first,
    outer_loops,
    inner_unroll,
    register_chains,
    alignment_nops,
    output: VOID_POINTER,  # pyright: ignore[reportInvalidTypeForm]
):
    """测量 ``MFMA + EXP + N*ADD`` 与 ``MFMA + N*ADD + EXP``。"""
    assert 0 <= alu_count <= 3
    regs = make_registers(jit, register_chains)
    mfma_d, mfma_a, mfma_b = make_mfma_registers(jit)

    emit_alignment_padding(jit, alignment_nops)
    start = read_clock(jit)
    loop = jit.gpr("su32", 0)
    with jit.While(loop[0] < outer_loops):
        for index in range(inner_unroll):
            slot = index % register_chains
            emit_mfma(jit, mfma_d, mfma_a, mfma_b, slot)
            if exp_first:
                emit_exp(jit, regs, slot)
            for alu_slot in range(alu_count):
                VALU_EMITTERS["v_add_f32"](
                    jit, regs, (slot + alu_slot) % register_chains
                )
            if not exp_first:
                emit_exp(jit, regs, slot)
        loop[0] += 1
    store_elapsed(jit, start, output, 0)

    sink = jit.gpr("vf32")
    jit.v_add_f32(sink, mfma_d[0, 0], regs["exp_dst"][0])
    jit.v_add_f32(sink, sink, regs["dst"][0])
    sink_s = jit.gpr("su32")
    jit.v_readfirstlane_b32(sink_s, sink)
    jit.s_store_dword(sink_s, output, 8, mod="glc")


@jit(no_pass=["pass_dse", "pass_dce"])
def measure_two_mfma_exp_alu_bundle(
    jit: JIT,
    alu_count,
    order,
    outer_loops,
    inner_unroll,
    register_chains,
    alignment_nops,
    output: VOID_POINTER,  # pyright: ignore[reportInvalidTypeForm]
):
    """测量两条独立MFMA共享一组EXP与N条ADD的三种排布。"""
    assert 0 <= alu_count <= 3
    assert order in (0, 1, 2)
    regs = make_registers(jit, register_chains)
    mfma_d, mfma_a, mfma_b = make_mfma_registers(jit)

    def emit_alu(slot):
        for alu_slot in range(alu_count):
            VALU_EMITTERS["v_add_f32"](jit, regs, (slot + alu_slot) % register_chains)

    emit_alignment_padding(jit, alignment_nops)
    start = read_clock(jit)
    loop = jit.gpr("su32", 0)
    with jit.While(loop[0] < outer_loops):
        for index in range(inner_unroll):
            slot = (2 * index) % register_chains
            emit_mfma(jit, mfma_d, mfma_a, mfma_b, slot)
            if order == 1:
                emit_alu(slot)
            elif order == 2:
                emit_exp(jit, regs, slot)
            emit_mfma(
                jit,
                mfma_d,
                mfma_a,
                mfma_b,
                (slot + 1) % register_chains,
            )
            if order == 0:
                emit_alu(slot)
                emit_exp(jit, regs, slot)
            elif order == 1:
                emit_exp(jit, regs, slot)
            else:
                emit_alu(slot)
        loop[0] += 1
    store_elapsed(jit, start, output, 0)

    sink = jit.gpr("vf32")
    jit.v_add_f32(sink, mfma_d[0, 0], mfma_d[1, 0])
    jit.v_add_f32(sink, sink, regs["exp_dst"][0])
    jit.v_add_f32(sink, sink, regs["dst"][0])
    sink_s = jit.gpr("su32")
    jit.v_readfirstlane_b32(sink_s, sink)
    jit.s_store_dword(sink_s, output, 8, mod="glc")


@jit(no_pass=["pass_dse", "pass_dce"])
def measure_inter_coissue(
    jit: JIT,
    opcode,
    valu_count,
    outer_loops,
    inner_unroll,
    register_chains,
    alignment_nops,
    output: VOID_POINTER,  # pyright: ignore[reportInvalidTypeForm]
):
    """让同一SIMD上的两个wave反相执行MFMA和tested-op segment。"""
    assert opcode in VALU_EMITTERS
    assert 0 <= valu_count <= MAX_VALU_COUNT
    emit = VALU_EMITTERS[opcode]
    regs = make_registers(jit, register_chains)
    mfma_d, mfma_a, mfma_b = make_mfma_registers(jit)

    emit_alignment_padding(jit, alignment_nops)
    start = read_clock(jit)

    # 后4 waves先多等待一个barrier。第一次物理barrier由低4 waves的common
    # barrier和高4 waves的extra barrier配对，从而建立一个phase的偏移。
    with jit.If(jit.warp_id[0] >= 4):
        jit.s_barrier()
    jit.s_barrier()

    loop = jit.gpr("su32", 0)
    with jit.While(loop[0] < outer_loops):
        for index in range(inner_unroll):
            emit_mfma(jit, mfma_d, mfma_a, mfma_b, index & 3)
        jit.s_barrier()

        for index in range(inner_unroll):
            slot = index % register_chains
            for valu_slot in range(valu_count):
                emit(jit, regs, (slot + valu_slot) % register_chains)
        jit.s_barrier()
        loop[0] += 1

    # 低4 waves领先一个barrier，出口补一次以排空高4 waves的最后一个tested-op segment。
    with jit.If(jit.warp_id[0] < 4):
        jit.s_barrier()

    wave_byte_offset = jit.gpr("su32", jit.warp_id[0] * 16)
    store_elapsed(jit, start, output, wave_byte_offset)

    simd_id = jit.gpr("su32")
    jit.s_getreg_b32(simd_id, mod="hwreg(HW_REG_HW_ID, 4, 2)")
    jit.s_store_dword(simd_id, output, wave_byte_offset + 8, mod="glc")

    sink = jit.gpr("vf32")
    jit.v_add_f32(sink, mfma_d[0, 0], regs["dst"][0])
    jit.v_add_f32(sink, sink, regs["pk_dst"][0, 0])
    sink_s = jit.gpr("su32")
    jit.v_readfirstlane_b32(sink_s, sink)
    jit.s_store_dword(sink_s, output, wave_byte_offset + 12, mod="glc")
    jit.s_waitcnt(mod="lgkmcnt(0)")


@jit(no_pass=["pass_dse", "pass_dce"])
def measure_inter_exp_coissue(
    jit: JIT,
    opcode,
    valu_count,
    outer_loops,
    inner_unroll,
    register_chains,
    alignment_nops,
    output: VOID_POINTER,  # pyright: ignore[reportInvalidTypeForm]
):
    """让同一SIMD上的两个wave反相执行EXP和tested-op segment。"""
    assert opcode in VALU_EMITTERS
    assert 0 <= valu_count <= MAX_VALU_COUNT
    emit = VALU_EMITTERS[opcode]
    regs = make_registers(jit, register_chains)

    emit_alignment_padding(jit, alignment_nops)
    start = read_clock(jit)
    with jit.If(jit.warp_id[0] >= 4):
        jit.s_barrier()
    jit.s_barrier()

    loop = jit.gpr("su32", 0)
    with jit.While(loop[0] < outer_loops):
        for index in range(inner_unroll):
            emit_exp(jit, regs, index % register_chains)
        jit.s_barrier()

        for index in range(inner_unroll):
            slot = index % register_chains
            for valu_slot in range(valu_count):
                emit(jit, regs, (slot + valu_slot) % register_chains)
        jit.s_barrier()
        loop[0] += 1

    with jit.If(jit.warp_id[0] < 4):
        jit.s_barrier()

    wave_byte_offset = jit.gpr("su32", jit.warp_id[0] * 16)
    store_elapsed(jit, start, output, wave_byte_offset)

    simd_id = jit.gpr("su32")
    jit.s_getreg_b32(simd_id, mod="hwreg(HW_REG_HW_ID, 4, 2)")
    jit.s_store_dword(simd_id, output, wave_byte_offset + 8, mod="glc")

    sink = jit.gpr("vf32")
    jit.v_add_f32(sink, regs["exp_dst"][0], regs["dst"][0])
    jit.v_add_f32(sink, sink, regs["pk_dst"][0, 0])
    sink_s = jit.gpr("su32")
    jit.v_readfirstlane_b32(sink_s, sink)
    jit.s_store_dword(sink_s, output, wave_byte_offset + 12, mod="glc")
    jit.s_waitcnt(mod="lgkmcnt(0)")


def run_kernel(kernel, arguments, samples, warmup):
    output = torch.zeros(2, dtype=torch.uint64, device="cuda")
    for _ in range(warmup):
        kernel([1], [64], *arguments, output.data_ptr())
    torch.cuda.synchronize()

    values = []
    for _ in range(samples):
        kernel([1], [64], *arguments, output.data_ptr())
        torch.cuda.synchronize()
        values.append(int(output[0].item()))
    return statistics.median(values), values


def run_inter_kernel(kernel, arguments, samples, warmup):
    output = torch.zeros((INTER_WAVES, 2), dtype=torch.uint64, device="cuda")
    for _ in range(warmup):
        kernel([1], [INTER_WAVES * 64], *arguments, output.data_ptr())
    torch.cuda.synchronize()

    critical_values = []
    wave_samples = []
    simd_id_samples = []
    for _ in range(samples):
        kernel([1], [INTER_WAVES * 64], *arguments, output.data_ptr())
        torch.cuda.synchronize()
        host = output.cpu()
        elapsed = [int(value) for value in host[:, 0].tolist()]
        current_simd_ids = [int(value) & 0xFFFFFFFF for value in host[:, 1].tolist()]
        for wave in range(4):
            if current_simd_ids[wave] != current_simd_ids[wave + 4]:
                raise RuntimeError(
                    "inter co-issue要求wave i与wave i+4位于同一SIMD；"
                    f"实际SIMD映射为{current_simd_ids}"
                )
        simd_id_samples.append(current_simd_ids)
        critical_values.append(max(elapsed))
        wave_samples.append(elapsed)

    wave_medians = [
        statistics.median(sample[wave] for sample in wave_samples)
        for wave in range(INTER_WAVES)
    ]
    return {
        "cycles": statistics.median(critical_values),
        "samples": critical_values,
        "wave_median_cycles": wave_medians,
        "max_wave_spread_cycles": max(wave_medians) - min(wave_medians),
        "simd_ids": simd_id_samples[0],
        "simd_id_samples": simd_id_samples,
    }


def finalize_coissue(rows, tolerance, tested_op_cycles):
    baseline = rows[0]["cycles_per_group"]
    max_fully_hidden = 0
    max_full_coissue = 0
    for row in rows:
        tested_stream_cycles = row["valu_count"] * tested_op_cycles
        serial_cycles = baseline + tested_stream_cycles
        ideal_full_coissue_cycles = max(baseline, tested_stream_cycles)
        max_overlap_cycles = min(baseline, tested_stream_cycles)
        row["delta_cycles_per_group"] = row["cycles_per_group"] - baseline
        row["tested_stream_cycles"] = tested_stream_cycles
        row["serial_cycles"] = serial_cycles
        row["ideal_full_coissue_cycles"] = ideal_full_coissue_cycles
        row["overlap_cycles"] = max(0.0, serial_cycles - row["cycles_per_group"])
        row["overlap_ratio"] = (
            0.0
            if max_overlap_cycles == 0
            else min(1.0, row["overlap_cycles"] / max_overlap_cycles)
        )
        row["fully_hidden_by_anchor"] = row["delta_cycles_per_group"] <= tolerance
        row["full_coissue"] = (
            row["cycles_per_group"] - ideal_full_coissue_cycles <= tolerance
        )
        row["partial_coissue"] = (
            row["overlap_cycles"] > tolerance and not row["full_coissue"]
        )
        if (
            row["valu_count"] > 0
            and row["fully_hidden_by_anchor"]
            and row["valu_count"] == max_fully_hidden + 1
        ):
            max_fully_hidden = row["valu_count"]
        if (
            row["valu_count"] > 0
            and row["full_coissue"]
            and row["valu_count"] == max_full_coissue + 1
        ):
            max_full_coissue = row["valu_count"]
    rows[0]["delta_cycles_per_group"] = 0.0
    rows[0]["fully_hidden_by_anchor"] = True
    rows[0]["full_coissue"] = True
    rows[0]["partial_coissue"] = False
    # 兼容旧JSON字段；旧fully_hidden实际语义是fully hidden by anchor。
    for row in rows:
        row["fully_hidden"] = row["fully_hidden_by_anchor"]
    return baseline, max_fully_hidden, max_full_coissue


def measure_one(
    opcode,
    outer_loops,
    inner_unroll,
    register_chains,
    alignment_nops,
    samples,
    warmup,
    tolerance,
    throughput_only=False,
):
    instruction_count = outer_loops * inner_unroll
    empty_cycles, _ = run_kernel(
        measure_instruction,
        ("none", outer_loops, inner_unroll, register_chains, alignment_nops),
        samples,
        warmup,
    )
    instruction_cycles, throughput_samples = run_kernel(
        measure_instruction,
        (opcode, outer_loops, inner_unroll, register_chains, alignment_nops),
        samples,
        warmup,
    )
    throughput = max(0.0, instruction_cycles - empty_cycles) / instruction_count
    throughput_result = {
        "opcode": opcode,
        "instruction_count": instruction_count,
        "empty_cycles": empty_cycles,
        "instruction_cycles": instruction_cycles,
        "throughput_cycles_per_instruction": throughput,
        "throughput_instructions_per_cycle": (
            0.0 if throughput == 0 else 1.0 / throughput
        ),
        "throughput_samples": throughput_samples,
    }
    if throughput_only:
        return throughput_result

    intra_coissue = []
    for valu_count in range(MAX_VALU_COUNT + 1):
        cycles, cycle_samples = run_kernel(
            measure_coissue,
            (
                opcode,
                valu_count,
                outer_loops,
                inner_unroll,
                register_chains,
                alignment_nops,
            ),
            samples,
            warmup,
        )
        intra_coissue.append(
            {
                "valu_count": valu_count,
                "cycles": cycles,
                "cycles_per_group": cycles / instruction_count,
                "samples": cycle_samples,
            }
        )

    intra_baseline, max_intra_fully_hidden, max_intra_coissue = finalize_coissue(
        intra_coissue, tolerance, throughput
    )

    inter_coissue = []
    inter_instruction_count = instruction_count
    for valu_count in range(MAX_VALU_COUNT + 1):
        measurement = run_inter_kernel(
            measure_inter_coissue,
            (
                opcode,
                valu_count,
                outer_loops,
                inner_unroll,
                register_chains,
                alignment_nops,
            ),
            samples,
            warmup,
        )
        # 同一SIMD上的两个wave都执行一遍MFMA和tested-op segment。steady state
        # 每个runtime loop包含两个反相phase，因此按2*group_count归一。
        measurement.update(
            {
                "valu_count": valu_count,
                "cycles_per_group": measurement["cycles"]
                / (2 * inter_instruction_count),
            }
        )
        inter_coissue.append(measurement)
    inter_baseline, max_inter_fully_hidden, max_inter_coissue = finalize_coissue(
        inter_coissue, tolerance, throughput
    )

    exp_intra_coissue = []
    for valu_count in range(MAX_VALU_COUNT + 1):
        cycles, cycle_samples = run_kernel(
            measure_exp_coissue,
            (
                opcode,
                valu_count,
                outer_loops,
                inner_unroll,
                register_chains,
                alignment_nops,
            ),
            samples,
            warmup,
        )
        exp_intra_coissue.append(
            {
                "valu_count": valu_count,
                "cycles": cycles,
                "cycles_per_group": cycles / instruction_count,
                "samples": cycle_samples,
            }
        )
    (
        exp_intra_baseline,
        max_exp_intra_fully_hidden,
        max_exp_intra_coissue,
    ) = finalize_coissue(exp_intra_coissue, tolerance, throughput)

    exp_inter_coissue = []
    for valu_count in range(MAX_VALU_COUNT + 1):
        measurement = run_inter_kernel(
            measure_inter_exp_coissue,
            (
                opcode,
                valu_count,
                outer_loops,
                inner_unroll,
                register_chains,
                alignment_nops,
            ),
            samples,
            warmup,
        )
        measurement.update(
            {
                "valu_count": valu_count,
                "cycles_per_group": measurement["cycles"]
                / (2 * inter_instruction_count),
            }
        )
        exp_inter_coissue.append(measurement)
    (
        exp_inter_baseline,
        max_exp_inter_fully_hidden,
        max_exp_inter_coissue,
    ) = finalize_coissue(exp_inter_coissue, tolerance, throughput)

    available_shadow = max(0.0, intra_baseline - MFMA_TO_VALU_START_CYCLES)
    throughput_capacity = (
        0
        if throughput == 0
        else min(
            MAX_VALU_COUNT,
            int((available_shadow + tolerance) / throughput),
        )
    )
    return {
        **throughput_result,
        "mfma_baseline_cycles_per_instruction": intra_baseline,
        "inter_mfma_baseline_cycles_per_instruction": inter_baseline,
        "mfma_to_valu_start_cycles": MFMA_TO_VALU_START_CYCLES,
        "throughput_predicted_valu": throughput_capacity,
        "max_fully_hidden_valu": max_intra_fully_hidden,
        "max_intra_fully_hidden_by_anchor": max_intra_fully_hidden,
        "max_inter_fully_hidden_by_anchor": max_inter_fully_hidden,
        "max_intra_coissue": max_intra_coissue,
        "max_inter_coissue": max_inter_coissue,
        "max_intra_full_coissue": max_intra_coissue,
        "max_inter_full_coissue": max_inter_coissue,
        "exp_intra_baseline_cycles_per_instruction": exp_intra_baseline,
        "exp_inter_baseline_cycles_per_instruction": exp_inter_baseline,
        "max_exp_intra_fully_hidden_by_anchor": max_exp_intra_fully_hidden,
        "max_exp_inter_fully_hidden_by_anchor": max_exp_inter_fully_hidden,
        "max_exp_intra_full_coissue": max_exp_intra_coissue,
        "max_exp_inter_full_coissue": max_exp_inter_coissue,
        "coissue": intra_coissue,
        "intra_coissue": intra_coissue,
        "inter_coissue": inter_coissue,
        "exp_intra_coissue": exp_intra_coissue,
        "exp_inter_coissue": exp_inter_coissue,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ops",
        default="all",
        help="逗号分隔的opcode；all表示VALU_TESTS中的全部指令",
    )
    parser.add_argument("--outer-loops", type=int, default=DEFAULT_OUTER_LOOPS)
    parser.add_argument("--inner-unroll", type=int, default=DEFAULT_INNER_UNROLL)
    parser.add_argument("--register-chains", type=int, default=4)
    parser.add_argument(
        "--alignment-nops",
        type=int,
        default=0,
        help="在计时开始前插入的4-byte s_nop数量，用于控制hot loop PC对齐",
    )
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--throughput-only",
        action="store_true",
        help="只测单指令吞吐，跳过MFMA/EXP intra/inter co-issue",
    )
    parser.add_argument(
        "--mfma-exp-alu-bundle",
        action="store_true",
        help="只测MFMA、EXP与0..3条独立v_add_f32的两种顺序",
    )
    parser.add_argument(
        "--two-mfma-exp-alu-bundle",
        action="store_true",
        help="只测两条MFMA共享EXP与0..3条独立v_add_f32的三种顺序",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.25,
        help="full co-issue判定允许的cycles/group误差",
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--json", help="将完整结果写入JSON")
    args = parser.parse_args()

    if (
        args.outer_loops <= 0
        or args.inner_unroll <= 0
        or args.register_chains <= 0
        or args.alignment_nops < 0
        or args.samples <= 0
        or args.warmup < 0
    ):
        parser.error("循环和样本参数必须为正数")
    if args.tolerance < 0:
        parser.error("--tolerance不能为负")

    if args.ops == "all":
        opcodes = list(VALU_EMITTERS)
    else:
        opcodes = [item.strip() for item in args.ops.split(",") if item.strip()]
        unknown = [opcode for opcode in opcodes if opcode not in VALU_EMITTERS]
        if unknown:
            parser.error(f"未知opcode：{', '.join(unknown)}")

    torch.cuda.set_device(args.device)
    arch = torch.cuda.get_device_properties(args.device).gcnArchName
    if not ("gfx94" in arch or "gfx950" in arch):
        raise RuntimeError(f"当前只支持gfx94x/gfx950，检测到{arch}")

    results = []
    print(
        f"平台：{arch}，每次{args.outer_loops}×{args.inner_unroll}条指令，"
        f"register_chains={args.register_chains}，alignment_nops={args.alignment_nops}"
    )
    if args.two_mfma_exp_alu_bundle:
        instruction_count = args.outer_loops * args.inner_unroll
        order_names = (
            "MFMA2+ALU+EXP",
            "MFMA+ALU+MFMA+EXP",
            "MFMA+EXP+MFMA+ALU",
        )
        bundle_rows = []
        print("order                       ALU  cycle/group  samples")
        for order, order_name in enumerate(order_names):
            for alu_count in range(4):
                cycles, cycle_samples = run_kernel(
                    measure_two_mfma_exp_alu_bundle,
                    (
                        alu_count,
                        order,
                        args.outer_loops,
                        args.inner_unroll,
                        args.register_chains,
                        args.alignment_nops,
                    ),
                    args.samples,
                    args.warmup,
                )
                row = {
                    "order": order_name,
                    "alu_count": alu_count,
                    "cycles": cycles,
                    "cycles_per_group": cycles / instruction_count,
                    "samples": cycle_samples,
                }
                bundle_rows.append(row)
                print(
                    f"{order_name:27s} {alu_count:3d}"
                    f" {row['cycles_per_group']:12.3f}  {cycle_samples}"
                )
        report = {
            "arch": arch,
            "mode": "two-mfma-exp-alu-bundle",
            "outer_loops": args.outer_loops,
            "inner_unroll": args.inner_unroll,
            "register_chains": args.register_chains,
            "alignment_nops": args.alignment_nops,
            "instruction_count": instruction_count,
            "samples": args.samples,
            "warmup": args.warmup,
            "results": bundle_rows,
        }
        if args.json:
            with open(args.json, "w") as stream:
                json.dump(report, stream, indent=2, sort_keys=True)
                stream.write("\n")
            print(f"JSON：{args.json}")
        return

    if args.mfma_exp_alu_bundle:
        instruction_count = args.outer_loops * args.inner_unroll
        bundle_rows = []
        print("order                 ALU  cycle/group  samples")
        for exp_first in (False, True):
            order = "MFMA+ALU+EXP" if not exp_first else "MFMA+EXP+ALU"
            for alu_count in range(4):
                cycles, cycle_samples = run_kernel(
                    measure_mfma_exp_alu_bundle,
                    (
                        alu_count,
                        exp_first,
                        args.outer_loops,
                        args.inner_unroll,
                        args.register_chains,
                        args.alignment_nops,
                    ),
                    args.samples,
                    args.warmup,
                )
                row = {
                    "order": order,
                    "alu_count": alu_count,
                    "cycles": cycles,
                    "cycles_per_group": cycles / instruction_count,
                    "samples": cycle_samples,
                }
                bundle_rows.append(row)
                print(
                    f"{order:21s} {alu_count:3d}"
                    f" {row['cycles_per_group']:12.3f}  {cycle_samples}"
                )
        report = {
            "arch": arch,
            "mode": "mfma-exp-alu-bundle",
            "outer_loops": args.outer_loops,
            "inner_unroll": args.inner_unroll,
            "register_chains": args.register_chains,
            "alignment_nops": args.alignment_nops,
            "instruction_count": instruction_count,
            "samples": args.samples,
            "warmup": args.warmup,
            "results": bundle_rows,
        }
        if args.json:
            with open(args.json, "w") as stream:
                json.dump(report, stream, indent=2, sort_keys=True)
                stream.write("\n")
            print(f"JSON：{args.json}")
        return

    if args.throughput_only:
        print("opcode                 cycle/inst  instruction cycles  empty cycles")
    else:
        print(
            "opcode                 cycle/inst  MFMA hidden I/I  MFMA full I/I  "
            "EXP full I/I  EXP intra/inter N=1..4 total cycle(group)"
        )
    for opcode in opcodes:
        result = measure_one(
            opcode,
            args.outer_loops,
            args.inner_unroll,
            args.register_chains,
            args.alignment_nops,
            args.samples,
            args.warmup,
            args.tolerance,
            args.throughput_only,
        )
        results.append(result)
        if args.throughput_only:
            print(
                f"{opcode:22s} {result['throughput_cycles_per_instruction']:>10.6f}"
                f" {result['instruction_cycles']:>19.0f} {result['empty_cycles']:>13.0f}"
            )
            continue
        exp_intra_cycles = ",".join(
            f"{row['cycles_per_group']:.3f}" for row in result["exp_intra_coissue"][1:]
        )
        exp_inter_cycles = ",".join(
            f"{row['cycles_per_group']:.3f}" for row in result["exp_inter_coissue"][1:]
        )
        print(
            f"{opcode:22s} {result['throughput_cycles_per_instruction']:>10.3f}"
            f" {result['max_intra_fully_hidden_by_anchor']:>3d}/"
            f"{result['max_inter_fully_hidden_by_anchor']:<3d}"
            f" {result['max_intra_full_coissue']:>3d}/{result['max_inter_full_coissue']:<3d}"
            f" {result['max_exp_intra_full_coissue']:>3d}/{result['max_exp_inter_full_coissue']:<3d}  "
            f"{exp_intra_cycles} / {exp_inter_cycles}"
        )

    report = {
        "arch": arch,
        "mode": "throughput-only" if args.throughput_only else "full-coissue",
        "outer_loops": args.outer_loops,
        "inner_unroll": args.inner_unroll,
        "register_chains": args.register_chains,
        "alignment_nops": args.alignment_nops,
        "inter_waves": INTER_WAVES,
        "instruction_count": args.outer_loops * args.inner_unroll,
        "samples": args.samples,
        "warmup": args.warmup,
        "mfma_to_valu_start_cycles": MFMA_TO_VALU_START_CYCLES,
        "max_valu_count": MAX_VALU_COUNT,
        "tolerance_cycles_per_group": args.tolerance,
        "inter_protocol": {
            "low_wave_group": [0, 1, 2, 3],
            "high_wave_group": [4, 5, 6, 7],
            "entry_extra_barrier_group": "high_wave_group",
            "exit_drain_barrier_group": "low_wave_group",
            "phases_per_runtime_loop": 2,
            "segment_instruction_count": args.inner_unroll,
            "simd_pairs": [[0, 4], [1, 5], [2, 6], [3, 7]],
        },
        "anchors": ["mfma", "v_exp_f32"],
        "results": results,
    }
    if args.json:
        with open(args.json, "w") as stream:
            json.dump(report, stream, indent=2, sort_keys=True)
            stream.write("\n")
        print(f"JSON：{args.json}")


if __name__ == "__main__":
    main()
