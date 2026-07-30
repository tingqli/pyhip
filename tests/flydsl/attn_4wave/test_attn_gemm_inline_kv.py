import hashlib
import re

from attn_4wave.attn_gemm_inline_kv import (  # pyright: ignore[reportMissingImports]
    ARCHIVED_ISA_SHA256,
    JIT_MAIN_INLINE_ASM,
    JIT_REGISTER_CONSTRAINTS,
    _archive_path,
)


def test_jit_main_inline_body_invariants():
    instructions = [line.strip() for line in JIT_MAIN_INLINE_ASM.splitlines()]

    assert (
        hashlib.sha256(_archive_path().read_bytes()).hexdigest() == ARCHIVED_ISA_SHA256
    )
    assert sum(line.startswith("v_mfma_") for line in instructions) == 128
    assert sum(line.startswith("s_setprio") for line in instructions) == 4
    assert not any(
        re.search(r"(?<![A-Za-z0-9_])a(?:\d+|\[)", line) for line in instructions
    )
    assert not any(line.startswith("s_load_dwordx2") for line in instructions)
    assert any(".Lfly_inline_attn_attn_pair_loop:" == line for line in instructions)
    assert "{s[6:7]}" in JIT_REGISTER_CONSTRAINTS
    assert "{s[12:13]}" in JIT_REGISTER_CONSTRAINTS
    assert "~{v219}" in JIT_REGISTER_CONSTRAINTS
