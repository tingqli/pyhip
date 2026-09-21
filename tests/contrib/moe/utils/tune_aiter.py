# SPDX-License-Identifier: MIT
"""Run the official Aiter tuner, then measure moe_driver.aiter against ref.

Use one fixed output directory and run only one instance at a time. Import,
help and dry-run are CPU-only; real measurement runs in the calling process.
"""

from __future__ import annotations

import argparse
import ast
from contextlib import redirect_stdout
import csv
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

__all__ = ["tune", "ensure_tuned"]

_KEYS = (
    "gfx", "cu_num", "token", "model_dim", "inter_dim", "expert", "topk",
    "act_type", "dtype", "q_dtype_a", "q_dtype_w", "q_type", "use_g1u1",
    "doweight_stage1",
)
_RESULTS = (
    "block_m", "ksplit", "us1", "kernelName1", "err1", "us2", "kernelName2",
    "err2", "us", "run_1stage", "xbf16", "flat", "tflops", "bw",
)
_ACTIVATIONS = {"silu": "Silu", "gelu": "Gelu", "gelu_tanh": "GeluTanh",
                "swiglu": "Swiglu", "situv2": "Situv2"}
_QUANTS = ("no_quant", "bf16", "fp8_ptpc", "fp8_per_tensor", "fp8_blockscale",
           "a16w4", "a8w4", "a4w4")
_DRIVER_KWARGS = ("preshuffle", "swiglu_limit", "beta", "linear_beta", "output_dtype")
_BACKENDS = ("asm", "cktile", "flydsl", "flydslv2", "flydsli4", "opus")
_DEFAULT_TOKENS = tuple(1 << power for power in range(17))
# Named *_args model configurations in ../test_moe.py; no import-time GPU setup.
# Preserve global I and numerical policy, not kernel tiles or batch scheduling.
MODEL_CONFIGS = {
    "hy3": "8,4096,1536,193,9,fp8_per_tensor,silu",
    "hy3_pad": "8,4096,2048,193,9,fp8_per_tensor,silu", # 1536/8=192 which is not supported by aiter
    "qwen35_397B": "8,4096,4096,512,10,fp8_ptpc,silu",
    "qwen35_397B_k256": "8,4096,2048,512,10,fp8_ptpc,silu",
    "qwen35_35B": "1,2048,512,256,8,fp8_ptpc,silu",
    "qwen35_35B_k256": "1,2048,256,256,8,fp8_ptpc,silu",
    "xiaomi": "8,6144,2048,384,8,fp8_ptpc,silu",
    "h3": "8,6144,3072,128,4,fp8_ptpc,silu",
}
_TUNER = Path("csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py")
_CACHE_DIRS = {"AITER_JIT_DIR": "jit", "FLYDSL_RUNTIME_CACHE_DIR": "flydsl-cache",
               "FLYDSL_AUTOTUNE_CACHE_DIR": "flydsl-autotune", "TRITON_CACHE_DIR": "triton-cache",
               "TORCHINDUCTOR_CACHE_DIR": "inductor-cache"}


def _positive_int(name, value, *, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _spec(TP, model_dim, inter_dim, experts, topk, tokens, quant_scheme_str,
          activation, driver_kwargs=None):
    for key, value in (("TP", TP), ("model_dim", model_dim), ("inter_dim", inter_dim),
                       ("experts", experts), ("topk", topk), ("tokens", tokens)):
        _positive_int(key, value)
    if inter_dim % TP:
        raise ValueError("inter_dim is global/pre-TP and must be divisible by TP")
    if topk > experts or topk > 255 or tokens >= 2**24:
        raise ValueError("require topk <= experts, topk <= 255 and tokens < 2**24")
    quant = quant_scheme_str.replace("-", "_")
    act = activation.lower().replace("-", "_")
    act = {"situ": "situv2", "gelutanh": "gelu_tanh", "none": "no", "identity": "no"}.get(act, act)
    extra = dict(driver_kwargs or {})
    unknown = extra.keys() - set(_DRIVER_KWARGS)
    if unknown:
        raise ValueError(f"unknown driver kwargs: {', '.join(sorted(unknown))}")
    spec = dict(TP=TP, model_dim=model_dim, inter_dim=inter_dim, experts=experts,
                topk=topk, tokens=tokens, quant_scheme_str=quant, activation=act,
                preshuffle=True,
                swiglu_limit=7.0 if act == "swiglu" else None,
                beta=4.0 if act == "situv2" else 1.0,
                linear_beta=25.0 if act == "situv2" else 1.0)
    spec.update(extra)
    if not isinstance(spec["preshuffle"], bool):
        raise TypeError("preshuffle must be bool")
    for name in ("swiglu_limit", "beta", "linear_beta"):
        value = spec[name]
        if name == "swiglu_limit" and value is None:
            continue
        if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                or (value < 0 if name == "swiglu_limit" else value <= 0)):
            raise ValueError(f"invalid {name}: {value!r}")
    if "output_dtype" in spec:
        spec["output_dtype"] = str(spec["output_dtype"]).removeprefix("torch.")
        if spec["output_dtype"] not in ("float16", "bfloat16", "float32"):
            raise ValueError("output_dtype must be float16, bfloat16 or float32")
    return spec


def _check_tuning_spec(spec, gfx=None):
    """The official CSV/tuner cannot describe arbitrary driver overrides."""
    quant, act = spec["quant_scheme_str"], spec["activation"]
    if quant not in _QUANTS or act not in _ACTIVATIONS:
        raise NotImplementedError(f"unsupported quantization/activation: {quant}/{act}")
    mx = quant in ("a16w4", "a8w4", "a4w4")
    alignment = 256 if mx else (128 if quant == "fp8_blockscale" else 32)
    if spec["model_dim"] % alignment or (spec["inter_dim"] // spec["TP"]) % alignment:
        raise NotImplementedError(f"H and local I must be divisible by {alignment}; no implicit padding")
    if mx:
        if act not in ("silu", "swiglu", "situv2"):
            raise NotImplementedError("MXFP4 requires silu/swiglu/situv2 and stage2 routing weights")
        if quant == "a16w4" and act != "situv2":
            raise NotImplementedError("standard A16W4 tuning supports SiTUv2 only")
        if quant == "a8w4" and act not in ("swiglu", "situv2"):
            raise NotImplementedError("standard A8W4 tuning requires SwiGLU or SiTUv2")
    elif act not in ("silu", "gelu", "gelu_tanh"):
        raise NotImplementedError("BF16/FP8 tuning requires silu/gelu/gelu_tanh")
    standard = dict(preshuffle=True, output_dtype="bfloat16",
                    swiglu_limit=7.0 if act == "swiglu" else None,
                    beta=4.0 if act == "situv2" else 1.0,
                    linear_beta=25.0 if act == "situv2" else 1.0)
    for name, expected in standard.items():
        actual = spec.get(name, expected)
        if name == "swiglu_limit" and act == "swiglu" and actual is None:
            actual = 7.0
        if actual != expected:
            raise NotImplementedError(f"standard Aiter tuned CSV requires {name}={expected!r}; "
                                      "use moe_bench with a compatible driver for custom settings")


def _token_key(tokens):
    if tokens < 32768:
        return 1 << (tokens - 1).bit_length()
    return 131072 if tokens >= 131072 else 32768


def _quant_contract(quant, gfx):
    if gfx not in ("gfx942", "gfx950"):
        raise NotImplementedError(f"unsupported architecture {gfx!r}")
    fp8 = "torch.float8_e4m3fnuz" if gfx == "gfx942" else "torch.float8_e4m3fn"
    if quant in ("no_quant", "bf16"):
        return "QuantType.No", "torch.bfloat16", "torch.bfloat16"
    if quant.startswith("fp8_"):
        qtype = {"fp8_ptpc": "per_Token", "fp8_per_tensor": "per_Tensor",
                 "fp8_blockscale": "per_1x128"}[quant]
        return f"QuantType.{qtype}", fp8, fp8
    if gfx != "gfx950":
        raise NotImplementedError("MXFP4 tuning requires gfx950")
    fp4 = "torch.float4_e2m1fn_x2"
    return "QuantType.per_1x32", {"a16w4": "torch.bfloat16", "a8w4": fp8, "a4w4": fp4}[quant], fp4


def _untuned_row(spec, gfx, cu_num):
    _check_tuning_spec(spec, gfx)
    qtype, qa, qw = _quant_contract(spec["quant_scheme_str"], gfx)
    return dict(zip(_KEYS, (
        gfx, cu_num, _token_key(spec["tokens"]), spec["model_dim"],
        spec["inter_dim"] // spec["TP"], spec["experts"], spec["topk"],
        f"ActivationType.{_ACTIVATIONS[spec['activation']]}", "torch.bfloat16",
        qa, qw, qtype, True, False,
    )))


def _write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def _write_csv(path, fields, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _csv_int(value):
    number = float(value)
    if not number.is_integer():
        raise ValueError(f"not an integer CSV value: {value!r}")
    return int(number)


def _csv_bool(value):
    value = str(value).strip().lower()
    if value in ("true", "1", "1.0"):
        return True
    if value in ("false", "0", "0.0"):
        return False
    raise ValueError(f"not a boolean CSV value: {value!r}")


def _same_key(row, expected):
    for key, value in expected.items():
        actual = row[key].strip()
        if isinstance(value, bool):
            actual = _csv_bool(actual)
        elif isinstance(value, int):
            actual = _csv_int(actual)
        if actual != value:
            return False
    return True


def _read_winner(path, expected, *, history=False):
    """Read the official winner, or the latest full-key untagged history row."""
    path = Path(path)
    if history and (not path.exists() or path.stat().st_size == 0):
        return None
    text = path.read_text(encoding="utf-8")
    reader = csv.DictReader(io.StringIO(text), strict=True)
    fields = reader.fieldnames or []
    if len(fields) != len(set(fields)) or not set(_KEYS + _RESULTS[:10]).issubset(fields):
        raise RuntimeError(f"unrecognized CSV columns in {path}")
    lines, matches = text.split("\n"), []
    for row in reader:
        if None in row or any(v is None or "\n" in v or "\r" in v for v in row.values()):
            raise RuntimeError(f"malformed CSV record in {path}")
        if not row.get("_tag", "").strip() and _same_key(row, expected):
            matches.append((row, lines[reader.line_num - 1]))
    if not matches and history:
        return None
    if not matches or (not history and len(matches) != 1):
        raise RuntimeError(f"expected one full-key winner in {path}, got {len(matches)}")
    row, raw = matches[-1]
    us = float(row["us"])
    if not math.isfinite(us) or us <= 0:
        raise RuntimeError(f"no valid candidate in {path}: us={row['us']}")
    if _csv_int(row["block_m"]) <= 0 or _csv_int(row["ksplit"]) < 0:
        raise RuntimeError("invalid block_m/ksplit in tuned row")
    for key in ("kernelName1",) if _csv_bool(row["run_1stage"]) else ("kernelName1", "kernelName2"):
        if row[key].strip().lower() in ("", "none", "nan", "null", "0"):
            raise RuntimeError(f"tuned row has no {key}")
    return dict(csv_header=lines[0], tuned_row=raw, tuned_config=row)


def _append_csv(path, fields, row):
    """Append in place. The caller owns serialization and write-failure recovery."""
    path = Path(path)
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    if existing:
        header = next(csv.reader(io.StringIO(existing)))
        if not set(fields).issubset(header):
            raise RuntimeError(f"incompatible CSV columns in {path}")
        fields = header
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        if not existing:
            writer.writeheader()
        elif not existing.endswith(("\n", "\r")):
            stream.write("\n")
        writer.writerow(row)


def _aiter_root(value):
    if value is None:
        found = importlib.util.find_spec("aiter")
        if found is None or found.origin is None:
            raise RuntimeError("Aiter is not installed; supply an importable aiter_root")
        value = Path(found.origin).resolve().parent.parent
    root = Path(value).expanduser().resolve()
    if not (root / _TUNER).is_file():
        raise RuntimeError(f"Aiter tuning script not found under {root}")
    return root


def _environment(output_dir, root, device, spec, backends, kernel_regex):
    env = os.environ.copy()
    visible = env.get("HIP_VISIBLE_DEVICES", env.get("CUDA_VISIBLE_DEVICES"))
    selected = visible.split(",")[device].strip() if visible is not None else str(device)
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env.update(HIP_VISIBLE_DEVICES=selected, PYTHONDONTWRITEBYTECODE="1",
               PYTHONPATH=os.pathsep.join(filter(None, (str(root), env.get("PYTHONPATH", "")))),
               AITER_META_DIR=str(root), AITER_CONFIG_FMOE=str(output_dir / "work/tuned.csv"),
               CU_NUM="0", GPU_ARCHS="native", AITER_REBUILD="0",
               TUNE_ONLY=backends, OPUS_ONLY="0", OPUS_SKIP_CKTILE="0",
               TUNE_MOE_EXPERT_BALANCE="False", AITER_ONLINE_TUNE="0", AITER_BYPASS_TUNE_CONFIG="0",
               AITER_KSPLIT="0", AITER_AOT_IMPORT="0", AITER_TRITON_ONLY="0", AITER_FLYDSL_FORCE="1",
               AITER_MOE_A8W4_BYPASS_QUANT="0", AITER_MXFP4_INTERMEDIATE="0",
               AITER_FLYDSL_STAGE2_FP8="0", AITER_FORCE_A8W4="0", AITER_FLYDSL_FORCE_REDUCE="0",
               AITER_XBFLOAT16="0", AITER_USE_NT="-1", AITER_MOE_SORT_BACKEND="auto",
               AITER_USE_CK_MOE_SORTING="0", AITER_USE_FLYDSL_MOE_SORTING="0",
               AITER_SITUV2_A8W4="1" if spec["quant_scheme_str"] == "a8w4" else "0",
               AITER_SITUV2_A4W4="1" if spec["quant_scheme_str"] == "a4w4" else "0",
               AITER_BF16_FP8_MOE_BOUND="0" if spec["quant_scheme_str"] == "a8w4" else "256",
               GPTOSS_SWIGLU_MXFP4_BF16_BOUND="0" if spec["quant_scheme_str"] == "a4w4" else "256")
    for key, name in _CACHE_DIRS.items():
        env.setdefault(key, str(output_dir / name))
    env.pop("AITER_GPU_ARCHS", None)
    env.pop("TUNE_MOE_KERNEL_REGEX", None)
    if kernel_regex is not None:
        env["TUNE_MOE_KERNEL_REGEX"] = kernel_regex
    return env


def _run_tuner(root, output_dir, env, timeout_s):
    work = output_dir / "work"
    command = [sys.executable, str(root / _TUNER),
               "-i", str(work / "untuned.csv"), "-o", str(work / "tuned.csv"),
               "-o2", str(work / "profile.csv"), "--mp", "1", "--all", "--timeout", str(timeout_s)]
    log = output_dir / "tune.log"
    with log.open("w") as stream:
        done = subprocess.run(command, cwd=output_dir, env=env, stdout=stream, stderr=subprocess.STDOUT)
    if done.returncode:
        with log.open(errors="replace") as stream:
            no_candidate = any("no valid candidate found for " in line for line in stream)
        detail = "; no valid kernel combination for this shape" if no_candidate else ""
        raise RuntimeError(f"Aiter tuner exited {done.returncode}{detail}; see {log}")


def _device_info(device):
    import torch

    if not torch.version.hip or not torch.cuda.is_available():
        raise RuntimeError("Aiter tuning requires ROCm Torch and an AMD GPU")
    with torch.cuda.device(device):
        torch.cuda.empty_cache()  # Make idle allocations available to the tuner child.
        p = torch.cuda.get_device_properties(device)
    return dict(name=p.name, gfx=p.gcnArchName.split(":")[0], cu_num=p.multi_processor_count)


def _benchmark(spec, root, config_dir, device, *, seed, warmup, iters, rounds,
               buffer_count, diff_threshold):
    """Measure the selected config using the same helpers as cross_compare."""
    import torch

    if __package__:
        from . import moe_driver, moe_bench
    else:
        import moe_driver
        import moe_bench

    with torch.cuda.device(device), torch.no_grad():
        import aiter

        if Path(aiter.__file__).resolve().parent.parent != root:
            raise RuntimeError("benchmark imports a different Aiter checkout; set PYTHONPATH before starting Python")
        target = torch.device("cuda", device)
        info = _device_info(device)
        row = _read_winner(config_dir / "tuned.csv", _untuned_row(spec, info["gfx"], info["cu_num"]))
        config = moe_bench.driver_config(spec)
        prepare, run = moe_driver.aiter(config, tuned_config=row["tuned_config"])
        data = moe_bench.make_data(spec, seed, device=target)
        buffer = moe_bench.prepare_buffer(config, prepare, data)
        moe_bench.launch(run, buffer)
        metadata = run.metadata[(target, _token_key(spec["tokens"]))]
        intermediate = None if metadata.fuse_quant in ("fp4", "fp8") else torch.bfloat16
        reference = moe_bench.make_reference(spec, data, intermediate_dtype=intermediate, output_dtype=torch.bfloat16)
        del data
        measured = moe_bench.measure(run, buffer, reference, warmup=warmup, iters=iters,
                                     rounds=rounds, buffer_count=buffer_count, diff_threshold=diff_threshold)
        del buffer, reference
        torch.cuda.empty_cache()
    if measured["status"] != "OK":
        raise RuntimeError(f"{measured['reason']}: calc_diff={measured['diff']}")
    return dict(**measured, dispatch_verified=True, warmup=warmup, iters=iters, rounds=rounds,
                reference=dict(driver="ref", intermediate_dtype=str(intermediate), output_dtype="torch.bfloat16"))


def ensure_tuned(spec, root, output, device, info, *, seed=43, timeout_s=120,
                 tune_backends=None, kernel_regex=None, force=False, no_tune=False):
    """Get one official winner without generating inputs, reference or timings."""
    output = Path(output)
    expected = _untuned_row(spec, info["gfx"], info["cu_num"])
    winner = None if force else _read_winner(output / "tuned.csv", expected, history=True)
    skipped = winner is not None
    if winner is None:
        if no_tune:
            raise ValueError(f"no exact tuned row for tokens={spec['tokens']} in {output}")
        work = output / "work"
        work.mkdir(parents=True, exist_ok=True)
        _append_csv(output / "untuned.csv", _KEYS, expected)
        _write_csv(work / "untuned.csv", _KEYS, [expected])
        _write_csv(work / "tuned.csv", _KEYS + _RESULTS, [])
        (work / "profile.csv").unlink(missing_ok=True)
        env = _environment(output, root, device, spec, tune_backends or "", kernel_regex)
        env.update(TUNE_MOE_ROUTING_SEED=str(seed), GPU_ARCHS=info["gfx"], CU_NUM=str(info["cu_num"]))
        _run_tuner(root, output, env, timeout_s)
        winner = _read_winner(work / "tuned.csv", expected)
        _append_csv(output / "tuned.csv", tuple(winner["tuned_config"]), winner["tuned_config"])
        winner = _read_winner(output / "tuned.csv", expected, history=True)
    return dict(**winner, tuning_skipped=skipped, source_csv=str(output / "tuned.csv"))


def tune(
    TP: int, model_dim: int, inter_dim: int, experts: int, topk: int, tokens: int,
    quant_scheme_str: str = "fp8_ptpc", activation: str = "silu", *,
    device: int | None = None,
    aiter_root=None, output_dir="./tuned_aiter", seed: int = 43, warmup: int = 5,
    iters: int = 20, rounds: int = 5, buffer_count: int = 10,
    diff_threshold: float = 0.02, timeout_s: int = 120,
    tune_backends: str | None = None, kernel_regex: str | None = None, force: bool = False,
    driver_kwargs=None,
) -> dict:
    """Tune/reuse one CSV key, then measure actual tokens through moe_driver.

    output_dir is persistent and cannot be None. The caller guarantees single
    execution and owns directory/GPU concurrency risks. timeout_s is the
    official tuner's per-task watchdog, not an overall wall-clock timeout.
    """
    if output_dir is None:
        raise ValueError("output_dir cannot be None; default is ./tuned_aiter")
    spec = _spec(TP, model_dim, inter_dim, experts, topk, tokens, quant_scheme_str,
                 activation, driver_kwargs)
    _check_tuning_spec(spec)
    for name, value in (("warmup", warmup), ("iters", iters), ("rounds", rounds),
                        ("buffer_count", buffer_count), ("timeout_s", timeout_s)):
        _positive_int(name, value)
    _positive_int("seed", seed, minimum=0)
    if not math.isfinite(diff_threshold) or not 0 <= diff_threshold <= 2:
        raise ValueError("diff_threshold must be finite and in [0,2]")
    backends = tune_backends or ""
    if any(item not in _BACKENDS for item in backends.split(",") if item):
        raise ValueError(f"tune_backends must be a comma-list from {_BACKENDS}")
    if kernel_regex is not None:
        re.compile(kernel_regex)
    if device is None:
        torch = sys.modules.get("torch")
        device = torch.cuda.current_device() if torch is not None and torch.cuda.is_initialized() else 0
    _positive_int("device", device, minimum=0)
    root = _aiter_root(aiter_root)
    output = Path(output_dir).expanduser().resolve()
    work = output / "work"
    work.mkdir(parents=True, exist_ok=True)
    (output / "result.json").unlink(missing_ok=True)
    info = _device_info(device)
    winner = ensure_tuned(spec, root, output, device, info, seed=seed, timeout_s=timeout_s,
                          tune_backends=backends, kernel_regex=kernel_regex, force=force)
    tuning_skipped = winner["tuning_skipped"]
    (work / "tuned.csv").write_text(winner["csv_header"] + "\n" + winner["tuned_row"] + "\n")
    with (output / "bench.log").open("w") as log, redirect_stdout(log):
        measured = _benchmark(spec, root, work, device, seed=seed, warmup=warmup, iters=iters,
                              rounds=rounds, buffer_count=buffer_count, diff_threshold=diff_threshold)
    result = dict(**winner, **measured, spec=spec, device=info,
                  output_dir=str(output), tuned_csv=str(output / "tuned.csv"),
                  untuned_csv=str(output / "untuned.csv"),
                  tune_backends=None if tuning_skipped else backends or "all",
                  kernel_regex=None if tuning_skipped else kernel_regex)
    _write_json(output / "result.json", result)
    return result


def _parse_config(value):
    names = ("TP", "model_dim", "inter_dim", "experts", "topk", "quant_scheme_str", "activation")
    fields = [field.strip() for field in value.split(",")]
    if fields[0] in MODEL_CONFIGS:
        fields = MODEL_CONFIGS[fields[0]].split(",") + fields[1:]
    elif re.fullmatch(r"[A-Za-z_]\w*", fields[0]):
        raise argparse.ArgumentTypeError(f"unknown model alias {fields[0]!r}; see --help for available aliases")
    if fields and not fields[-1]:
        fields.pop()  # A single trailing comma is harmless.
    if len(fields) < 7 or not all(fields):
        raise argparse.ArgumentTypeError("expected " + ",".join(names) + "[,key=value...][,tokens...]")
    if any("=" in field for field in fields[:7]):
        raise argparse.ArgumentTypeError("driver kwargs must follow activation and precede tokens")
    config = dict(zip(names, fields[:7]))
    try:
        config.update({name: int(config[name]) for name in names[:5]})
    except ValueError as exc:
        raise argparse.ArgumentTypeError("model dimensions must be integers") from exc
    tokens, kwargs = [], {}
    for field in fields[7:]:
        if "=" not in field:
            try:
                tokens.append(int(field))
            except ValueError as exc:
                raise argparse.ArgumentTypeError(f"expected a key=value option or integer tokens, got {field!r}") from exc
            continue
        if tokens:
            raise argparse.ArgumentTypeError("driver kwargs must precede all tokens")
        name, text = (part.strip() for part in field.split("=", 1))
        if name not in _DRIVER_KWARGS:
            raise argparse.ArgumentTypeError(f"unknown driver kwarg {name!r}; choose from {', '.join(_DRIVER_KWARGS)}")
        if name in kwargs:
            raise argparse.ArgumentTypeError(f"duplicate driver kwarg: {name}")
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            if not re.fullmatch(r"[A-Za-z_]\w*(?:\.\w+)*", text):
                raise argparse.ArgumentTypeError(f"{name} requires a scalar literal or name, got {text!r}") from None
            parsed = text
        if not isinstance(parsed, (str, bool, int, float, type(None))):
            raise argparse.ArgumentTypeError(f"{name} requires a scalar value")
        kwargs[name] = parsed
    tokens = tokens or list(_DEFAULT_TOKENS)
    if len(tokens) != len(set(tokens)) or any(not 0 < token < 2**24 for token in tokens):
        raise argparse.ArgumentTypeError("tokens must be unique positive integers < 2**24")
    config["tokens"] = tokens
    if kwargs:
        config["driver_kwargs"] = kwargs
    return config


def _config_help():
    return ("Model config aliases (from test_moe.py; case-sensitive):\n"
            + "\n".join(f"  {name:<20} = {config}" for name, config in MODEL_CONFIGS.items())
            + "\n\nUse NAME or NAME[,key=value...][,tokens...], e.g. hy3,8192,16384.\n"
              "Omitted tokens use powers of two from 1 to 65536; aliases do not select kernels or pad shapes.")


def _parser():
    parser = argparse.ArgumentParser(description=__doc__, epilog=_config_help(),
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=_parse_config,
                        help="model alias (listed below) or TP,H,global-I,experts,topk,quant,activation[,key=value...][,tokens...]")
    parser.add_argument("--output-dir", default="./tuned_aiter", help="persistent CSVs, work files and tuner caches")
    parser.add_argument("--device", type=int)
    parser.add_argument("--aiter-root")
    parser.add_argument("--force", action="store_true", help="retune and append instead of using a cached row")
    for flag, default in (("seed", 43), ("warmup", 5), ("iters", 20), ("rounds", 5),
                          ("buffer-count", 10), ("timeout-s", 120)):
        parser.add_argument(f"--{flag}", type=int, default=default)
    parser.add_argument("--diff-threshold", type=float, default=0.02)
    parser.add_argument("--tune-backends", help="restrict new tuning sources; default: all")
    parser.add_argument("--kernel-regex", help="restrict new tuning kernels; default: no filter")
    parser.add_argument("--dry-run", action="store_true", help="validate/print model specs without GPU discovery or files")
    return parser


def main(argv=None):
    parser = _parser()
    options = vars(parser.parse_args(argv))
    options.update(options.pop("config"))
    token_counts, dry = options.pop("tokens"), options.pop("dry_run")
    try:
        names = ("TP", "model_dim", "inter_dim", "experts", "topk", "quant_scheme_str", "activation")
        specs = [_spec(tokens=m, **{name: options[name] for name in names},
                       driver_kwargs=options.get("driver_kwargs")) for m in token_counts]
        for spec in specs:
            _check_tuning_spec(spec)
        results = ([dict(spec=spec, executed=False) for spec in specs] if dry else
                   [tune(tokens=m, **options) for m in token_counts])
        result = results[0] if len(results) == 1 else dict(requested_tokens=token_counts, results=results, executed=not dry)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (ValueError, TypeError, NotImplementedError, re.error) as exc:
        parser.error(str(exc))
    except (RuntimeError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())