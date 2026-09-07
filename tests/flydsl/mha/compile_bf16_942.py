"""CPU-only gfx942 BF16 resource compilation with metadata-only tensors.

No device allocation, execution engine, GPU queries or kernel launch. The
launcher sees the same shapes/strides/dtypes as native execution.
"""

import argparse
import contextlib
import hashlib
import itertools
import json
import os
from pathlib import Path
import re

import torch
import flydsl.compiler as flyc
from flydsl.utils import env

if __package__:
    from . import mha_pa_bf16_942 as module
    from ._runner import resource_fields, save
else:
    import mha_pa_bf16_942 as module
    from _runner import resource_fields, save


def compile_case(directory, *, dq, dv, page, causal, with_lse, heads=16, kv_heads=1, q=10240, kv=10240,
                 batch=1, mode="per-token", hints=None):
    def tensor(shape, dtype=torch.bfloat16):
        return torch.empty(shape, dtype=dtype, device="meta")
    pages = batch * ((kv + page - 1) // page)
    query = tensor((q * batch, heads, dq))
    key = tensor((pages, kv_heads, dq // 8, page, 8))
    value = tensor((pages, kv_heads, page // 8, dv, 8))
    cq, ck, indptr = (tensor((batch + 1,), torch.int32) for _ in range(3))
    indices, last = tensor((pages,), torch.int32), tensor((batch,), torch.int32)
    qs = tensor((q * batch * heads if mode == "per-token" else 1,), torch.float32)
    ks, vs = tensor((1,), torch.float32), tensor((1,), torch.float32)
    out, lse = tensor((q * batch, heads, dv)), tensor((q * batch, heads), torch.float32)
    counter = tensor((81,), torch.int32)
    launch = module._build_attention(heads, kv_heads, dq, dv, page, causal, mode, with_lse=with_lse)
    saved_hints = launch.compile_hints
    if hints:
        launch.compile_hints = {**launch.compile_hints, **hints}
    args = (query, key, value, cq, ck, indptr, indices, qs, ks, vs, last, out, lse if with_lse else ks, counter, 80, None)
    directory.mkdir(parents=True, exist_ok=True)
    saved = (env.compile.arch, env.compile.compile_only, env.debug.dump_ir, env.debug.dump_dir, env.runtime.enable_cache)
    log_path = directory / "compile.log"
    try:
        env.compile.arch, env.compile.compile_only = "gfx942", True
        env.debug.dump_ir, env.debug.dump_dir, env.runtime.enable_cache = True, str(directory), False
        # LLVM can print a register-allocation error yet return a partial ISA.
        # Capture native stderr too; an emitted file/exit0 alone is not success.
        with log_path.open("w") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            stderr_fd = os.dup(2)
            try:
                os.dup2(log.fileno(), 2)
                assert flyc.compile(launch, *args) is None
            finally:
                os.dup2(stderr_fd, 2)
                os.close(stderr_fd)
    finally:
        env.compile.arch, env.compile.compile_only, env.debug.dump_ir, env.debug.dump_dir, env.runtime.enable_cache = saved
        launch.compile_hints = saved_hints
    reject_codegen_errors(log_path.read_text())
    files = list(directory.rglob("*final_isa.s"))
    assert len(files) == 1, files
    text = files[0].read_text()
    return {"dq": dq, "dv": dv, "page": page, "causal": causal, "with_lse": with_lse,
            "mode": mode, "heads": heads, "kv_heads": kv_heads, "batch": batch, "q": q, "kv": kv,
            "executed": False, "gpu_queried": False, "tensor_device": "meta", "resources": resource_fields(text),
            "scratch_instructions": sum("scratch_" in line for line in text.splitlines()),
            "readfirstlane_instructions": text.count("v_readfirstlane"),
            "isa": str(files[0]), "isa_sha256": hashlib.sha256(text.encode()).hexdigest()}


def reject_codegen_errors(text):
    errors = [line for line in text.splitlines() if re.search(r"(?:^|\s)(?:error:|LLVM ERROR:)", line)]
    if errors:
        raise RuntimeError("invalid code generation: " + "; ".join(errors[:4]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dq", type=int, nargs="+", default=[128, 192])
    parser.add_argument("--dv", type=int, nargs="+", default=[128])
    parser.add_argument("--page", type=int, nargs="+", default=[64])
    parser.add_argument("--causal", type=int, nargs="+", default=[0])
    parser.add_argument("--lse", type=int, nargs="+", default=[0])
    parser.add_argument("--mode", choices=("per-token", "per-tensor"), default="per-token")
    parser.add_argument("--hints", type=json.loads, default={})
    parser.add_argument("--require-no-scratch", action="store_true")
    parser.add_argument("--retain-isa", action="store_true")
    parser.add_argument("--dump-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # Any accidental query/allocation path through torch.cuda is a test failure.
    def forbidden(*_args, **_kwargs):
        raise RuntimeError("CPU-only compilation must not initialize or query a GPU")
    for name in ("_lazy_init", "get_device_properties", "current_stream", "synchronize", "is_available"):
        setattr(torch.cuda, name, forbidden)
    result = {"executed": False, "gpu_queried": False, "source_sha256": hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
              "hints": args.hints, "records": [], "complete": False}
    save(args.output, result)
    for dq, dv, page, causal, with_lse in itertools.product(args.dq, args.dv, args.page, args.causal, args.lse):
        directory = args.dump_root / f"d{dq}_v{dv}_p{page}_c{causal}_lse{with_lse}_{args.mode}"
        row = compile_case(directory, dq=dq, dv=dv, page=page, causal=bool(causal), with_lse=bool(with_lse),
                           mode=args.mode, hints=args.hints)
        if args.retain_isa:
            retained = args.output.parent / (args.output.stem + "_isa") / (directory.name + ".s")
            retained.parent.mkdir(parents=True, exist_ok=True)
            retained.write_bytes(Path(row["isa"]).read_bytes())
            row["retained_isa"] = str(retained)
        result["records"].append(row)
        save(args.output, result)
        if args.require_no_scratch:
            fields = row["resources"]
            assert fields["private_segment_fixed_size"] == fields["vgpr_spill_count"] == row["scratch_instructions"] == 0, row
        print("BF16_CPU_COMPILE", dq, dv, page, causal, with_lse, row["resources"], flush=True)
    result["complete"] = True
    save(args.output, result)


if __name__ == "__main__":
    main()