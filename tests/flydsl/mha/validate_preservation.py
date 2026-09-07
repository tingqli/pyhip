"""Matched original/refactored validation, without modifying either kernel.

Original modules are loaded from the pinned Git revisions, not another test
directory or an installed checkout. Native timing uses ABBA order and the same
input, compiler, process and hardware policy. Cross-target audit never launches
code. No clocks, power or PTL settings are changed.
"""

import argparse
from collections import Counter
import contextlib
import hashlib
import importlib.util
import itertools
from pathlib import Path
import re
import statistics
import subprocess
import sys
import tempfile

import torch

if __package__:
    from ._runner import environment, profile_round, resource_fields, save
    from ._testing import FP8, FP8_REG, BF16_942, BF16_950, BF16_950_PERSISTENT, SWA
    from ._testing import make_case, make_call, assert_close, gpu_arch
else:
    from _runner import environment, profile_round, resource_fields, save
    from _testing import FP8, FP8_REG, BF16_942, BF16_950, BF16_950_PERSISTENT, SWA
    from _testing import make_case, make_call, assert_close, gpu_arch


HERE = Path(__file__).resolve().parent
GIT_ROOT = HERE.parents[2]
SOURCE_BRANCH = "23cc6d1e95b1611493e21232bef5d9962b7b73c9"
SOURCE_MAIN = "ebc533488b3d6a55e1dd386da2cc8c04293432ab"
SOURCES = {
    FP8.module: (SOURCE_BRANCH, "tests/flydsl/pa_8wave/tests/flydsl/pa_8wave/new_pa_8wave_942.py",
                 "ed955f256ebd596328b81e1a69731d7774e21579a38f16917cdcc0e28f62d89e"),
    BF16_942.module: (SOURCE_MAIN, "tests/flydsl/pa_8wave/pa_prefill_8w32x32.py",
                      "620209a023ccb5ea566489774d19edae880dfbcee298613233ed6f01f3b59849"),
    BF16_950.module: (SOURCE_BRANCH, "tests/flydsl/pa_8wave/pa_8wave_950.py",
                      "975757800802f8b4d30ebd325a0fc0763a8b0d5346c329836b2e7c2b8b553c69"),
    SWA.module: (SOURCE_BRANCH, "tests/flydsl/pa_1wave/swa_1wave.py",
                 "44b32d3bc5feb29e8088686397babb4552bfe5c3ed18653c5041e1657b07bf01"),
}


def load_original(name, directory):
    revision, relative, expected = SOURCES[name]
    source = subprocess.check_output(["git", "show", f"{revision}:{relative}"], cwd=GIT_ROOT)
    assert hashlib.sha256(source).hexdigest() == expected, (revision, relative)
    # FlyDSL uses inspect.getsource; a temporary exact copy preserves source
    # locations for tracing without changing the original checkout.
    path = directory / f"original_{name}.py"
    path.write_bytes(source)
    spec = importlib.util.spec_from_file_location(f"original_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def original_call(case, backend, module, causal):
    options = {"memory_mode": backend.memory_mode} if backend.fp8 else {}
    factory = module.PagedAttention(case.heads, case.kv_heads, case.dq, case.dv, case.page,
                                   causal, case.mode, **options)
    out = torch.empty(case.q.shape[0], case.heads, case.dv, dtype=torch.bfloat16, device=case.q.device)

    def call():
        return factory(case.q, case.k, case.v, case.cq, case.ck, case.indptr, case.indices,
                       max(case.q_lens), max(case.kv_lens), causal, case.qs, case.ks, case.vs, case.last, out=out)
    return call


def native_comparison(args, originals):
    if gpu_arch() != "gfx942":
        raise RuntimeError("native preservation matrix requires gfx942")
    result = {"environment": environment(), "originals": SOURCES, "records": []}
    output = args.output or HERE / "results" / "preservation_942.json"
    for backend, dq, causal in itertools.product((FP8, FP8_REG, BF16_942), (128, 192), (False, True)):
        q, kv = (1024, 1024) if causal else (10240, 2560)
        case = make_case((q,), (kv,), dtype=backend.dtype, dq=dq, heads=16, poison_tail=False)
        calls = {"original": original_call(case, backend, originals[backend.module], causal),
                 "refactored": make_call(case, backend, causal)[0]}
        values = {name: fn().clone() for name, fn in calls.items()}
        for value in values.values():
            assert_close(case, backend, value, None, causal)
        torch.testing.assert_close(values["refactored"], values["original"], rtol=0, atol=0)
        for _ in range(3):
            for name, fn in calls.items():
                torch.testing.assert_close(fn(), values[name], rtol=0, atol=0)
        for _ in range(args.warmup):
            for fn in calls.values():
                fn()
        pairs = []
        for trial in range(args.rounds):
            order = ("original", "refactored", "refactored", "original")
            samples = [profile_round(calls[name], iterations=args.iterations) for name in order]
            def paired(metric, indexes):
                return statistics.mean(samples[index][metric]["mean_us"] for index in indexes)
            row = {"round": trial, "order": order, "samples": samples,
                   "attention_ratio": paired("attention", (1, 2)) / paired("attention", (0, 3)),
                   "total_ratio": paired("total", (1, 2)) / paired("total", (0, 3)),
                   "original_drift_pct": 100 * (samples[3]["attention"]["mean_us"] / samples[0]["attention"]["mean_us"] - 1)}
            pairs.append(row)
            print("PRESERVATION_ABBA", backend.name, dq, causal, trial,
                  row["attention_ratio"], row["original_drift_pct"], flush=True)
        entry = {"backend": backend.name, "dq": dq, "q": q, "kv": kv, "causal": causal,
                 "batch": 1, "heads": 16, "kv_heads": 1, "page": 64,
                 "output_bit_exact": True, "pairs": pairs,
                 "attention_delta_pct": 100 * (statistics.median(pair["attention_ratio"] for pair in pairs) - 1),
                 "total_gpu_delta_pct": 100 * (statistics.median(pair["total_ratio"] for pair in pairs) - 1),
                 "attention_us": {name: statistics.median(sample["attention"]["mean_us"] for pair in pairs
                     for label, sample in zip(pair["order"], pair["samples"]) if label == name) for name in calls},
                 "protocol": {"initial_warmup": args.warmup, "rounds": args.rounds, "iterations": args.iterations,
                              "order": "ABBA", "per_sample": "20 warmup; drop first; 1.5IQR mean"}}
        result["records"].append(entry)
        save(output, result)
        print("PRESERVATION_RESULT", backend.name, dq, causal, entry["attention_delta_pct"], flush=True)
    return result


def compile_one(module, backend, case, *, with_lse, causal, qt, bn, original, directory):
    from flydsl.utils import env
    out = torch.empty(case.q.shape[0], case.heads, case.dv, device="cuda", dtype=torch.bfloat16)
    lse = torch.empty(case.q.shape[:2], device="cuda") if with_lse else case.ks
    common = (case.q.view(-1), case.k.view(-1), case.v.view(-1), out.view(-1), lse.view(-1),
              case.cq, case.indptr, case.indices, case.last, case.qs.view(-1), case.ks, case.vs)
    stream = torch.cuda.current_stream()
    if backend == SWA:
        values = (*common, case.sinks, 16, 1, case.k.shape[0], 1, 10240, case.dq,
                  16*case.dq, case.dq, 16*128, 128, 128, True, True, with_lse, case.dq**-0.5, bn, qt)
        launch = module._launch
        values = (*values, stream) if original else (*values, 950, stream)
    else:
        values = (*common, case.ks, 16, 1, case.k.shape[0], 1, 10240, case.dq, -1, False,
                  16*case.dq, case.dq, 16*128, 128, True, causal, with_lse, case.dq**-0.5)
        if backend.persistent:
            grid = torch.cuda.get_device_properties(0).multi_processor_count
            counter = torch.zeros(2, device="cuda", dtype=torch.int32)
            launch, values = module._launch_persistent, (*values, counter, grid, stream)
        else:
            launch, values = module._launch_attention, (*values, stream)
    directory.mkdir(parents=True, exist_ok=True)
    env.debug.dump_dir = str(directory)
    with (directory / "compile.log").open("w") as log, contextlib.redirect_stdout(log):
        assert launch(*values) is None
    files = list(directory.rglob("*final_isa.s"))
    assert len(files) == 1, files
    text = files[0].read_text()
    instructions = [line.strip() for line in text.splitlines() if re.match(r"\s+(?:v_|s_|ds_|buffer_|global_|scratch_)\w+", line)]
    return {"resources": resource_fields(text), "isa": str(files[0]),
            "isa_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "instruction_sha256": hashlib.sha256("\n".join(instructions).encode()).hexdigest(),
            "opcodes": dict(Counter(line.split()[0] for line in instructions))}


def cross_comparison(args, originals):
    from flydsl.utils import env
    result = {"environment": environment(), "originals": SOURCES, "target": "gfx950", "executed": False, "records": []}
    output = args.output or HERE / "results" / "preservation_950_compile.json"
    saved = (env.compile.arch, env.compile.compile_only, env.debug.dump_ir, env.debug.dump_dir, env.runtime.enable_cache)
    try:
        env.compile.arch, env.compile.compile_only = "gfx950", True
        env.debug.dump_ir, env.runtime.enable_cache = True, False
        for backend in (BF16_950, BF16_950_PERSISTENT, SWA):
            for dq, causal, with_lse in itertools.product((128, 192), (True,) if backend == SWA else (False, True), (False, True)):
                tiles = itertools.product((16, 32), (16, 32, 64)) if backend == SWA else ((None, None),)
                for qt, bn in tiles:
                    case = make_case((10240,), (2560,), dq=dq, heads=16, poison_tail=False,
                                     window_left=128 if backend == SWA else -1, has_sink=backend == SWA)
                    tag = f"{backend.name}_d{dq}_c{int(causal)}_lse{int(with_lse)}_q{qt}_bn{bn}"
                    record = {"backend": backend.name, "dq": dq, "causal": causal, "with_lse": with_lse,
                              "query_tile": qt, "block_n": bn}
                    for name, module in (("original", originals[backend.module]), ("refactored", backend.load())):
                        record[name] = compile_one(module, backend, case, with_lse=with_lse, causal=causal,
                            qt=qt, bn=bn, original=name == "original", directory=args.dump_root / tag / name)
                    record["resources_equal"] = record["original"]["resources"] == record["refactored"]["resources"]
                    record["instructions_equal"] = record["original"]["instruction_sha256"] == record["refactored"]["instruction_sha256"]
                    record["opcode_counts_equal"] = record["original"]["opcodes"] == record["refactored"]["opcodes"]
                    result["records"].append(record)
                    save(output, result)
                    print("PRESERVATION_COMPILE", tag, record["resources_equal"], record["instructions_equal"], flush=True)
    finally:
        env.compile.arch, env.compile.compile_only, env.debug.dump_ir, env.debug.dump_dir, env.runtime.enable_cache = saved
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("native", "cross-compile"), required=True)
    parser.add_argument("--warmup", type=int, default=1200)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dump-root", type=Path, default=Path("/tmp/mha_preservation_950"))
    args = parser.parse_args()
    if args.warmup < 0 or args.rounds < 1 or args.iterations < 2:
        parser.error("warmup >= 0, rounds >= 1 and iterations >= 2 are required")
    args.output = args.output.resolve() if args.output is not None else None
    args.dump_root = args.dump_root.resolve()
    required = (FP8.module, BF16_942.module) if args.mode == "native" else (BF16_950.module, SWA.module)
    with tempfile.TemporaryDirectory(prefix="mha_pinned_sources_") as directory:
        # Original main enables a relative "my_ir_dumps" directory at import.
        # Isolate that legacy debug side effect instead of editing/patching the
        # hash-verified source or overwriting the user's existing IR dumps.
        with contextlib.chdir(directory):
            originals = {name: load_original(name, Path(directory)) for name in required}
            (native_comparison if args.mode == "native" else cross_comparison)(args, originals)


if __name__ == "__main__":
    main()