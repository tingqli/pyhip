#!/usr/bin/env python3
"""测量 attention VALU 指令的吞吐，以及每条 MFMA 最多可共发多少条 VALU。

每次只启动一个64线程workgroup。默认外层运行时循环100次，内层静态展开1000次：

* 吞吐：分别测量空循环和100,000条目标指令，二者相减后除以100,000。
* 共发：在同一次kernel中依次测量纯MFMA，以及 ``MFMA + N * VALU``，N=1..4。
  若相对纯MFMA的每组增量不超过容差，则N条VALU视为可以完全藏在MFMA shadow中。

新增测试指令只需在 ``VALU_TESTS`` 中增加一项，并使用已有寄存器字段。
"""

import argparse
import json
import statistics

import torch

import pyhip

INNER_UNROLL = 1000
MFMA_TO_VALU_START_CYCLES = 4.0
MAX_VALU_COUNT = 4


def emit_none(jit, regs, slot):
    del jit, regs, slot


# 新增指令只需在这里增加一个 ``(opcode, lambda)``。
VALU_TESTS = [
    ("v_add_f32", lambda j, r, i: j.v_add_f32(r["dst"][i], r["src0"][i], r["src1"][i])),
    ("v_sub_f32", lambda j, r, i: j.v_sub_f32(r["dst"][i], r["src0"][i], r["src1"][i])),
    ("v_mul_f32", lambda j, r, i: j.v_mul_f32(r["dst"][i], r["src0"][i], r["src1"][i])),
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


def make_registers(jit):
    regs = {
        "src0": jit.gpr(4, "vf32", 0.25, align=4),
        "src1": jit.gpr(4, "vf32", 0.5, align=4),
        "src2": jit.gpr(4, "vf32", 0.75, align=4),
        "dst": jit.gpr(4, "vf32", 1.0, align=4),
        "pk_src0": jit.gpr(4, 2, "vf32", 0.25, align=2),
        "pk_src1": jit.gpr(4, 2, "vf32", 0.5, align=2),
        "pk_dst": jit.gpr(4, 2, "vf32", 1.0, align=2),
        "u32_src0": jit.gpr(4, "vu32", 0x12345678, align=4),
        "u32_src1": jit.gpr(4, "vu32", 0x89ABCDEF, align=4),
        "u32_dst": jit.gpr(4, "vu32", 1, align=4),
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


def emit_mfma(jit, mfma_d, mfma_a, mfma_b, slot):
    if jit.gfx >= 950:
        jit.v_mfma_f32_16x16x128_f8f6f4(mfma_d[slot], mfma_a, mfma_b, 0)
    else:
        jit.v_mfma_f32_16x16x16_bf16(mfma_d[slot], mfma_a, mfma_b, 0)


@pyhip.jit(no_pass=["pass_dse", "pass_dce"])
def measure_instruction(
    jit: pyhip.JIT,
    opcode,
    outer_loops,
    output: "void*",  # noqa: F722 - PyHIP JIT ABI注解
):
    assert opcode in VALU_EMITTERS or opcode == "none"
    emit = emit_none if opcode == "none" else VALU_EMITTERS[opcode]
    regs = make_registers(jit)

    start = read_clock(jit)
    loop = jit.gpr("su32", 0)
    with jit.While(loop[0] < outer_loops):
        for index in range(INNER_UNROLL):
            emit(jit, regs, index & 3)
        loop[0] += 1
    store_elapsed(jit, start, output, 0)

    sink = jit.gpr("vf32")
    jit.v_add_f32(sink, regs["dst"][0], regs["pk_dst"][0, 0])
    sink_s = jit.gpr("su32")
    jit.v_readfirstlane_b32(sink_s, sink)
    jit.s_store_dword(sink_s, output, 8, mod="glc")


@pyhip.jit(no_pass=["pass_dse", "pass_dce"])
def measure_coissue(
    jit: pyhip.JIT,
    opcode,
    valu_count,
    outer_loops,
    output: "void*",  # noqa: F722 - PyHIP JIT ABI注解
):
    assert opcode in VALU_EMITTERS
    assert 0 <= valu_count <= MAX_VALU_COUNT
    emit = VALU_EMITTERS[opcode]
    regs = make_registers(jit)

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

    start = read_clock(jit)
    loop = jit.gpr("su32", 0)
    with jit.While(loop[0] < outer_loops):
        for index in range(INNER_UNROLL):
            slot = index & 3
            emit_mfma(jit, mfma_d, mfma_a, mfma_b, slot)
            for valu_slot in range(valu_count):
                emit(jit, regs, (slot + valu_slot) & 3)
        loop[0] += 1
    store_elapsed(jit, start, output, 0)

    sink = jit.gpr("vf32")
    jit.v_add_f32(sink, mfma_d[0, 0], regs["dst"][0])
    jit.v_add_f32(sink, sink, regs["pk_dst"][0, 0])
    sink_s = jit.gpr("su32")
    jit.v_readfirstlane_b32(sink_s, sink)
    jit.s_store_dword(sink_s, output, 8, mod="glc")


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


def measure_one(opcode, outer_loops, samples, warmup, tolerance):
    instruction_count = outer_loops * INNER_UNROLL
    empty_cycles, _ = run_kernel(
        measure_instruction,
        ("none", outer_loops),
        samples,
        warmup,
    )
    instruction_cycles, throughput_samples = run_kernel(
        measure_instruction,
        (opcode, outer_loops),
        samples,
        warmup,
    )
    throughput = max(0.0, instruction_cycles - empty_cycles) / instruction_count

    coissue = []
    for valu_count in range(MAX_VALU_COUNT + 1):
        cycles, cycle_samples = run_kernel(
            measure_coissue,
            (opcode, valu_count, outer_loops),
            samples,
            warmup,
        )
        coissue.append(
            {
                "valu_count": valu_count,
                "cycles": cycles,
                "cycles_per_group": cycles / instruction_count,
                "samples": cycle_samples,
            }
        )

    baseline = coissue[0]["cycles_per_group"]
    max_coissue = 0
    for row in coissue[1:]:
        row["delta_cycles_per_group"] = row["cycles_per_group"] - baseline
        row["fully_hidden"] = row["delta_cycles_per_group"] <= tolerance
        if row["fully_hidden"] and row["valu_count"] == max_coissue + 1:
            max_coissue = row["valu_count"]

    coissue[0]["delta_cycles_per_group"] = 0.0
    coissue[0]["fully_hidden"] = True
    available_shadow = max(0.0, baseline - MFMA_TO_VALU_START_CYCLES)
    throughput_capacity = (
        0
        if throughput == 0
        else min(
            MAX_VALU_COUNT,
            int((available_shadow + tolerance) / throughput),
        )
    )
    return {
        "opcode": opcode,
        "instruction_count": instruction_count,
        "empty_cycles": empty_cycles,
        "instruction_cycles": instruction_cycles,
        "throughput_cycles_per_instruction": throughput,
        "throughput_instructions_per_cycle": (
            0.0 if throughput == 0 else 1.0 / throughput
        ),
        "throughput_samples": throughput_samples,
        "mfma_baseline_cycles_per_instruction": baseline,
        "mfma_to_valu_start_cycles": MFMA_TO_VALU_START_CYCLES,
        "throughput_predicted_valu": throughput_capacity,
        "max_fully_hidden_valu": max_coissue,
        "coissue": coissue,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ops",
        default="all",
        help="逗号分隔的opcode；all表示VALU_TESTS中的全部指令",
    )
    parser.add_argument("--outer-loops", type=int, default=100)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.25,
        help="判定完全隐藏允许的cycles/group误差",
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--json", help="将完整结果写入JSON")
    args = parser.parse_args()

    if args.outer_loops <= 0 or args.samples <= 0 or args.warmup < 0:
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
    print(f"平台：{arch}，每次{args.outer_loops}×{INNER_UNROLL}条指令")
    print(
        "opcode                 cycle/inst  inst/cycle  理论数  实测数  N=1..4增量(cycle/group)"
    )
    for opcode in opcodes:
        result = measure_one(
            opcode,
            args.outer_loops,
            args.samples,
            args.warmup,
            args.tolerance,
        )
        results.append(result)
        deltas = ", ".join(
            f"{row['delta_cycles_per_group']:+.3f}" for row in result["coissue"][1:]
        )
        print(
            f"{opcode:22s} {result['throughput_cycles_per_instruction']:>10.3f}"
            f" {result['throughput_instructions_per_cycle']:>11.4f}"
            f" {result['throughput_predicted_valu']:>7d}"
            f" {result['max_fully_hidden_valu']:>9d}  {deltas}"
        )

    report = {
        "arch": arch,
        "outer_loops": args.outer_loops,
        "inner_unroll": INNER_UNROLL,
        "instruction_count": args.outer_loops * INNER_UNROLL,
        "samples": args.samples,
        "mfma_to_valu_start_cycles": MFMA_TO_VALU_START_CYCLES,
        "max_valu_count": MAX_VALU_COUNT,
        "tolerance_cycles_per_group": args.tolerance,
        "results": results,
    }
    if args.json:
        with open(args.json, "w") as stream:
            json.dump(report, stream, indent=2, sort_keys=True)
            stream.write("\n")
        print(f"JSON：{args.json}")


if __name__ == "__main__":
    main()
