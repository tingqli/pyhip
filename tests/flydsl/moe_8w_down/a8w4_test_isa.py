# SPDX-License-Identifier: MIT
"""Read-only ISA checks for already-compiled A8W4 kernels; no launch caching."""

import hashlib
import re
import subprocess
import tempfile


def launch_of(down):
    if hasattr(down, "launch"):
        return down.launch
    cells = dict(zip(down.__code__.co_freevars, (cell.cell_contents for cell in down.__closure__)))
    return cells["launch"]


def _disassemble(down, *, with_metadata=False):
    from pyhip.contrib.flydsl.mfma_nop_patch import _decode_mlir_bytes

    artifact = launch_of(down)._cf._keepalive
    if isinstance(artifact, tuple):
        artifact = next(x for x in artifact if hasattr(x, "_ir_text"))
    binary_text = re.findall(r'"(\\7FELF(?:[^"\\]|\\.)*)"', artifact._ir_text, re.DOTALL)
    assert len(binary_text) == 1, "expected one emitted GPU module"
    binary = _decode_mlir_bytes(binary_text[0])
    metadata = ""
    with tempfile.NamedTemporaryFile(suffix=".hsaco") as file:
        file.write(binary)
        file.flush()
        text = subprocess.run(
            ["/opt/rocm/llvm/bin/llvm-objdump", "-d", "--mcpu=gfx950", file.name],
            check=True, capture_output=True, text=True).stdout
        if with_metadata:
            metadata = subprocess.run(
                ["/opt/rocm/llvm/bin/llvm-readobj", "--notes", file.name],
                check=True, capture_output=True, text=True).stdout
    instructions = [line.split("//")[0].strip() for line in text.splitlines()
                    if re.search(r'//\s*[0-9A-Fa-f]+:', line)]
    return binary, instructions, metadata


def _audit(binary, instructions, policy):
    stores = [line for line in instructions if line.startswith("buffer_store")]
    assert stores and all(line.startswith("buffer_store_dwordx4") for line in stores)
    if policy == 18:
        assert all("sc1" in line.split() and "nt" in line.split() for line in stores), stores[:4]
    elif policy == 2:
        assert all("sc1" not in line.split() and "nt" in line.split() for line in stores), stores[:4]
    elif policy == 16:
        assert all("sc1" in line.split() and "nt" not in line.split() for line in stores), stores[:4]
    elif policy in (None, 0):
        assert all("sc1" not in line.split() and "nt" not in line.split() for line in stores), stores[:4]
    normalized = [re.sub(r'\s+(?:sc1|nt)\b', '', line) if line.startswith("buffer_store") else line
                  for line in instructions]
    return {
        "elf_sha256": hashlib.sha256(binary).hexdigest(),
        "instructions_sha256": hashlib.sha256('\n'.join(instructions).encode()).hexdigest(),
        "store_flags_removed_sha256": hashlib.sha256('\n'.join(normalized).encode()).hexdigest(),
        "instruction_count": len(instructions),
        "static_output_store_count": len(stores),
        "static_mfma_count": sum(line.startswith('v_mfma') for line in instructions),
        "store_examples": stores[:2],
        "b_dma_examples": [line for line in instructions
                           if line.startswith('buffer_load') and 'lds' in line.split()][:2],
    }


def audit(down, policy):
    """Assert output-store policy and report hashes/counts of the current ELF."""
    binary, instructions, _ = _disassemble(down)
    return _audit(binary, instructions, policy)


def inspect_isa(kernel, policy):
    """Audit plus scratch/DPP/DMA counts and code-object resource metadata."""
    binary, instructions, metadata = _disassemble(kernel, with_metadata=True)
    result = _audit(binary, instructions, policy)
    result['scratch_instructions'] = sum('scratch_' in line for line in instructions)
    result['dpp_instructions'] = sum('quad_perm:[1,0,3,2]' in line for line in instructions)
    result['dma_instructions'] = sum(line.startswith('buffer_load') and 'lds' in line.split()
                                     for line in instructions)
    result['resources'] = {name: int(value) for name, value in re.findall(
        r'\.(group_segment_fixed_size|private_segment_fixed_size|sgpr_count|vgpr_count|agpr_count|sgpr_spill_count|vgpr_spill_count):\s*(\d+)',
        metadata)}
    return result