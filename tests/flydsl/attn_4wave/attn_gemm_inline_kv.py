"""Fly外壳 + JIT主流程大块inline asm的attention实验实现。"""

import hashlib
import re
from pathlib import Path

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.expr import arith

from attn_4wave.fly_isa_priority import (  # pyright: ignore[reportMissingImports]
    convert_accvgprs_to_vgprs,
    convert_jit_attention_to_fly_abi,
)

BM = 128
BN = 32
D = 128
THREADS = 256
LDS_BYTES = 16 * 1024
ARCHIVED_ISA_SHA256 = "18e3fe8e48e9eaa2bc62ba6ac82e7f41c5019e216b6638c0bd8decb452139c3b"


def _archive_path():
    return (
        Path(__file__).resolve().parent
        / "isa/attn-gemm-jit-setprio-best-gfx942-m40960-n40960-237p1t.s"
    )


def _instruction_text(line):
    text = line.split(";", 1)[0].strip()
    if not text or text.startswith((".loc", ".file", ".cfi", ";;")):
        return None
    return text


def _extract_jit_main_inline_asm():
    source = _archive_path().read_text(encoding="utf-8")
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    if digest != ARCHIVED_ISA_SHA256:
        raise ValueError(
            "archived JIT ISA changed: "
            f"expected {ARCHIVED_ISA_SHA256}, found {digest}"
        )

    # 保留已验证的JIT指令顺序，转换为Fly的全V寄存器形式和tensor kernarg ABI。
    source = convert_jit_attention_to_fly_abi(convert_accvgprs_to_vgprs(source))
    lines = source.splitlines()
    begin = next(
        index
        for index, line in enumerate(lines)
        if line.strip().startswith("_jit_main:")
    )
    end = next(
        index
        for index, line in enumerate(lines[begin:], begin)
        if line.strip().startswith("s_endpgm")
    )

    archived_body = []
    for line in lines[begin + 1 : end]:
        instruction = _instruction_text(line)
        if instruction is not None:
            archived_body.append(instruction)

    descriptor_begin = archived_body.index("s_mov_b32 s19,0x20000")
    archived_prefix = archived_body[:descriptor_begin]
    if sum(item.startswith("s_load_dwordx2") for item in archived_prefix) != 4:
        raise ValueError("expected four archived kernarg pointer loads")
    for required in (
        "s_lshl_b32 s14,s2,0x7",
        "s_mul_i32 s14,0xa00000,s3",
        "s_add_u32 s10,s10,s5",
        "s_addc_u32 s11,s11,0x0",
    ):
        if required not in archived_prefix:
            raise ValueError(
                f"archived JIT address prologue changed: missing {required}"
            )

    # block/head基址由Fly计算；JIT地址prologue只保留wave内行偏移和lane设置。
    body = [
        "v_lshrrev_b32 v1,0x6,v0",
        "s_nop 0x1",
        "v_readfirstlane_b32 s5,v1",
        "v_and_b32 v1,0x3f,v0",
        "s_waitcnt lgkmcnt(0)",
        "v_and_b32 v2,0xf,v1",
        "v_lshrrev_b32 v3,0x4,v1",
        "v_xor_b32 v4,0x20,v1",
        "v_lshlrev_b32 v4,0x2,v4",
        "v_xor_b32 v5,0x30,v1",
        "v_lshlrev_b32 v5,0x2,v5",
        "s_nop 0x1",
        "s_lshl_b32 s5,s5,0x5",
        "s_lshl_b32 s5,s5,0x8",
        "s_add_u32 s6,s6,s5",
        "s_addc_u32 s7,s7,0x0",
        "s_add_u32 s12,s12,s5",
        "s_addc_u32 s13,s13,0x0",
        *archived_body[descriptor_begin:],
    ]

    labels = [
        match.group(1)
        for instruction in body
        if (match := re.fullmatch(r"([A-Za-z_.$][A-Za-z0-9_.$]*):", instruction))
    ]
    if len(labels) != len(set(labels)):
        raise ValueError(f"duplicate labels in archived JIT body: {labels}")
    label_map = {
        label: ".Lfly_inline_attn_" + re.sub(r"[^A-Za-z0-9_]", "_", label)
        for label in labels
    }
    for index, instruction in enumerate(body):
        for label, replacement in label_map.items():
            instruction = re.sub(
                rf"(?<![A-Za-z0-9_.$]){re.escape(label)}(?![A-Za-z0-9_.$])",
                replacement,
                instruction,
            )
        body[index] = instruction

    mfma_count = sum(item.startswith("v_mfma_") for item in body)
    priority_count = sum(item.startswith("s_setprio") for item in body)
    if mfma_count != 128 or priority_count != 4:
        raise ValueError(
            f"unexpected archived JIT body: {mfma_count=} {priority_count=}"
        )
    if any(re.search(r"(?<![A-Za-z0-9_])a(?:\d+|\[)", item) for item in body):
        raise ValueError("AGPR operand remains in all-V inline body")

    return "\n\t".join(body)


JIT_MAIN_INLINE_ASM = _extract_jit_main_inline_asm()

JIT_REGISTER_CONSTRAINTS = (
    "{s[6:7]},{s[8:9]},{s[10:11]},{s[12:13]}," "~{v219},~{s27},~{memory},~{vcc},~{scc}"
)


def build(M, N, D_, BM_, BN_, H=1):
    """构建固定40960 attention，将JIT主流程放入单个asm块。"""
    if (M, N, D_, BM_, BN_) != (40960, 40960, D, BM, BN):
        raise ValueError(
            "jit_body_inline is specialized for "
            f"M=N=40960,D=128,BM=128,BN=32; got {(M, N, D_, BM_, BN_)}"
        )

    @flyc.kernel(known_block_size=[THREADS, 1, 1])
    def attn_kernel(
        query: fx.Tensor,
        key: fx.Tensor,
        value: fx.Tensor,
        output: fx.Tensor,
    ):
        block = fx.block_idx.x
        head = fx.block_idx.y
        q_offset = head * (M * D) + block * (BM * D)
        kv_offset = head * (N * D)
        q_base = fx.ptrtoint(fx.get_iter(query) + q_offset)
        k_base = fx.ptrtoint(fx.get_iter(key) + kv_offset)
        v_base = fx.ptrtoint(fx.get_iter(value) + kv_offset)
        o_base = fx.ptrtoint(fx.get_iter(output) + q_offset)

        # Fly负责tensor ABI、block/head offset和launch；asm负责wave内offset、
        # descriptor、prologue、含两次kv_step的pair-loop及epilogue/store。
        llvm.inline_asm(
            None,
            [
                arith.unwrap(q_base),
                arith.unwrap(k_base),
                arith.unwrap(v_base),
                arith.unwrap(o_base),
            ],
            JIT_MAIN_INLINE_ASM,
            JIT_REGISTER_CONSTRAINTS,
            has_side_effects=True,
        )

    @flyc.jit
    def launch(
        query: fx.Tensor,
        key: fx.Tensor,
        value: fx.Tensor,
        output: fx.Tensor,
        stream: fx.Stream,
    ):
        attn_kernel(query, key, value, output).launch(
            grid=(M // BM, H, 1),
            block=(THREADS, 1, 1),
            smem=LDS_BYTES,
            stream=stream,
        )

    return launch
