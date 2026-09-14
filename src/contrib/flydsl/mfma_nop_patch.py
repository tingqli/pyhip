# SPDX-License-Identifier: MIT

"""Experimental post-codegen MFMA removal for FlyDSL kernels.

The patch is applied to the embedded HSACO before the MLIR execution engine
loads it. Every MFMA instruction is replaced in place by the same number of
four-byte ``s_nop 0`` instructions, preserving all other code-object bytes.
"""

from __future__ import annotations

import os
import re
import shutil
import struct
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

from flydsl._mlir import ir
from flydsl.compiler import jit_executor, jit_function

_MLIR_BINARY_RE = re.compile(r'bin = "((?:[^"\\]|\\.)*)"', re.DOTALL)
_HEX_DIGITS = frozenset("0123456789abcdefABCDEF")
_TARGETS: set[str] = set()
_INSTALLED = False
_ORIGINAL_INIT = jit_executor.CompiledArtifact.__init__


def _decode_mlir_bytes(value: str) -> bytes:
    result = bytearray()
    index = 0
    while index < len(value):
        if (
            value[index] == "\\"
            and index + 2 < len(value)
            and value[index + 1] in _HEX_DIGITS
            and value[index + 2] in _HEX_DIGITS
        ):
            result.append(int(value[index + 1 : index + 3], 16))
            index += 3
        elif value[index : index + 2] == "\\\\":
            result.append(ord("\\"))
            index += 2
        else:
            result.extend(value[index].encode())
            index += 1
    return bytes(result)


def _encode_mlir_bytes(value: bytes) -> str:
    return "".join(f"\\{byte:02X}" for byte in value)


def _text_section(binary: bytes) -> tuple[int, int, int]:
    if binary[:4] != b"\x7fELF" or binary[4] != 2 or binary[5] != 1:
        raise RuntimeError("MFMA nop patch expects a little-endian ELF64 HSACO")

    section_offset = struct.unpack_from("<Q", binary, 0x28)[0]
    section_entry_size = struct.unpack_from("<H", binary, 0x3A)[0]
    section_count = struct.unpack_from("<H", binary, 0x3C)[0]
    string_section_index = struct.unpack_from("<H", binary, 0x3E)[0]

    def section(index: int) -> tuple[int, int, int, int]:
        offset = section_offset + index * section_entry_size
        name = struct.unpack_from("<I", binary, offset)[0]
        address = struct.unpack_from("<Q", binary, offset + 0x10)[0]
        file_offset = struct.unpack_from("<Q", binary, offset + 0x18)[0]
        size = struct.unpack_from("<Q", binary, offset + 0x20)[0]
        return name, address, file_offset, size

    _, _, strings_offset, strings_size = section(string_section_index)
    strings = binary[strings_offset : strings_offset + strings_size]
    for index in range(section_count):
        name_offset, address, file_offset, size = section(index)
        name_end = strings.find(b"\0", name_offset)
        if strings[name_offset:name_end] == b".text":
            return address, file_offset, size
    raise RuntimeError("HSACO has no .text section")


def _llvm_objdump() -> str:
    candidates = (
        os.environ.get("LLVM_OBJDUMP"),
        "/opt/rocm/llvm/bin/llvm-objdump",
        shutil.which("llvm-objdump"),
    )
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return candidate
    raise RuntimeError("llvm-objdump was not found; set LLVM_OBJDUMP")


@lru_cache(maxsize=4)
def _nop_bytes(size: int) -> bytes:
    llvm_mc = str(Path(_llvm_objdump()).with_name("llvm-mc"))
    instructions = ["v_nop_e64"] * (size // 8)
    if size % 8 == 4:
        instructions.append("s_nop 0")
    source = ".text\n" + "\n".join(instructions) + "\n"
    result = subprocess.run(
        [llvm_mc, "-triple=amdgcn-amd-amdhsa", "-mcpu=gfx950", "-filetype=obj"],
        input=source.encode(),
        check=True,
        capture_output=True,
    ).stdout
    text_address, text_offset, text_size = _text_section(result)
    del text_address
    return result[text_offset : text_offset + text_size]


def _patch_hsaco(binary: bytes) -> tuple[bytes, int]:
    with tempfile.NamedTemporaryFile(suffix=".hsaco") as hsaco:
        hsaco.write(binary)
        hsaco.flush()
        disassembly = subprocess.run(
            [_llvm_objdump(), "-d", "--mcpu=gfx950", hsaco.name],
            check=True,
            capture_output=True,
            text=True,
        ).stdout

    text_address, text_offset, text_size = _text_section(binary)
    patched = bytearray(binary)
    count = 0
    for line in disassembly.splitlines():
        if "\tv_mfma" not in line:
            continue
        match = re.search(
            r"//\s*([0-9A-Fa-f]+):\s*((?:[0-9A-Fa-f]{8}(?:\s+|$))+)",
            line,
        )
        if match is None:
            raise RuntimeError(f"cannot parse MFMA encoding: {line}")
        address = int(match.group(1), 16)
        instruction_bytes = 4 * len(match.group(2).split())
        file_offset = text_offset + address - text_address
        if not text_offset <= file_offset < text_offset + text_size:
            raise RuntimeError(f"MFMA address 0x{address:x} is outside .text")
        patched[file_offset : file_offset + instruction_bytes] = _nop_bytes(
            instruction_bytes
        )
        count += 1
    return bytes(patched), count


def _patch_ir_text(text: str) -> tuple[str, int, bytes | None]:
    total = 0
    last_binary = None

    def replace(match: re.Match[str]) -> str:
        nonlocal total, last_binary
        binary = _decode_mlir_bytes(match.group(1))
        patched, count = _patch_hsaco(binary)
        total += count
        last_binary = patched
        return f'bin = "{_encode_mlir_bytes(patched)}"'

    return _MLIR_BINARY_RE.sub(replace, text), total, last_binary


def install_mfma_nop_patch(kernel_name: str) -> None:
    """Arrange to remove MFMA instructions from one named FlyDSL kernel."""
    global _INSTALLED
    _TARGETS.add(kernel_name)
    if _INSTALLED:
        return

    def patched_init(self, compiled_module, func_name, *args, **kwargs):
        text = str(compiled_module)
        if any(target in text for target in _TARGETS):
            patched_text, count, binary = _patch_ir_text(text)
            if count == 0:
                raise RuntimeError("MFMA nop patch matched the kernel but found no MFMA")
            dump_path = os.environ.get("PYHIP_FLYDSL_NOP_MFMA_DUMP")
            if dump_path and binary is not None:
                Path(dump_path).write_bytes(binary)
            compiled_module = ir.Module.parse(
                patched_text, context=compiled_module.context
            )
            print(f"[flydsl] replaced {count} MFMA instructions with size-matched NOPs")
        _ORIGINAL_INIT(self, compiled_module, func_name, *args, **kwargs)

    jit_executor.CompiledArtifact.__init__ = patched_init
    jit_function.CompiledArtifact.__init__ = patched_init
    _INSTALLED = True
