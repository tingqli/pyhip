"""使用相同的模型 shape、权重布局和输入，对比 Aiter 与 tuned_moe 的性能。

脚本统一使用默认 CUDA 设备，不为某个后端单独 padding，也不创建 benchmark
worker 或 buffer 池。可先运行官方 Aiter tuner；run_perftest 负责轮换 buffer，
计时覆盖完整的 eager 调用，而非 graph replay 或各阶段耗时之和。用法见同目录 README.md。
"""

import argparse
from contextlib import contextmanager, nullcontext
import csv
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys

from pyhip.testing.moe_shapes import MOE_MODELS
from pyhip.testing.moe import check_output, measure_moe, moe_types, prepare_moe, torch_reference


@contextmanager
def environment(name, value):
    """临时修改调优环境变量，结束后恢复原值。"""
    previous = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


def _moe_types(model, args):
    return moe_types(model, dtype=args.dtype, quant=args.quant,
                     activation=args.activation, gate_mode=args.gate_mode)


def _aiter_shape(model, tokens, args):
    """CSV 使用 Aiter 的 lookup key，不给实际输入做 padding。"""
    import aiter.fused_moe as fm
    from aiter import dtypes

    kind, dtype, quant, activation = _moe_types(model, args)
    q_dtype_a = dtype
    if kind == "mxfp4":
        # Aiter 按实际 M 和 gate mode 选择激活精度，再按 bucket 查表。
        if args.activation == "situv2":
            q_dtype_a = (dtypes.fp8 if os.getenv("AITER_SITUV2_A8W4", "0") == "1" else
                         dtypes.fp4x2 if os.getenv("AITER_SITUV2_A4W4", "0") == "1" else dtypes.bf16)
        elif args.activation == "swiglu" and args.gate_mode == "separated":
            q_dtype_a = dtypes.bf16 if tokens < fm._SWIGLU_MXFP4_BF16_BOUND else dtypes.fp4x2
        elif args.activation == "swiglu" or args.gate_mode == "interleave":
            q_dtype_a = (dtypes.bf16 if fm.get_gfx() != "gfx950" or
                         tokens < int(os.getenv("AITER_BF16_FP8_MOE_BOUND", "256")) else dtypes.fp8)
    h, i = model["HIDDEN_SIZE"], model["INTER_SIZE"] // model["TP"]
    if kind == "block" and (h % 128 or i % 128):
        raise ValueError("block scales require H and I_tp divisible by 128; no implicit padding is allowed")
    return dict(token=fm.get_padded_M(tokens), model_dim=h, inter_dim=i,
                expert=model["E"], topk=model["TOPK"], act_type=activation, dtype=dtypes.bf16,
                q_dtype_a=q_dtype_a, q_dtype_w=dtype, q_type=fm.quant_remap.get(quant, quant),
                use_g1u1=args.activation != "gelu", doweight_stage1=False)


@contextmanager
def tune_aiter(models, args):
    """一次收集全部 shape，官方 tuner 写入最佳行，确认 dispatcher 命中后再计时。"""
    import aiter.fused_moe as fm
    from aiter.jit.core import AITER_CONFIGS, AITER_CSRC_DIR

    shapes = [_aiter_shape(MOE_MODELS[name], tokens, args) for name in models for tokens in args.tokens]
    shapes = list({tuple(shape.values()): shape for shape in shapes}.values())
    if any(not shape["use_g1u1"] for shape in shapes):
        raise ValueError("the official Aiter tuner currently supports G1U1 only, not non-gated GELU (G1U0)")
    directory = args.tune_aiter.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    untuned, tuned = directory / "untuned.csv", directory / "tuned.csv"
    with untuned.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(shapes[0]))
        writer.writeheader()
        writer.writerows(shapes)
    script = Path(AITER_CSRC_DIR) / "ck_gemm_moe_2stages_codegen/gemm_moe_tune.py"
    print(f"Aiter tuning: {len(shapes)} shapes, {untuned} -> {tuned}", flush=True)
    # 当前 tuner 默认只搜 flydslv2；显式打开全部候选，并限制为同一张 GPU。
    env = dict(os.environ, TUNE_ONLY="", OPUS_ONLY="0", OPUS_SKIP_CKTILE="0", TUNE_MOE_KERNEL_REGEX="",
               TUNE_MOE_EXPERT_BALANCE=str(args.routing == "balanced"), TUNE_MOE_ROUTING_SEED=str(args.seed))
    subprocess.run([sys.executable, str(script), "-i", str(untuned), "-o", str(tuned), "--all", "--mp", "1", "--timeout", "300"],
                   cwd=script.parents[2], env=env, check=True)

    def clear_configs():
        AITER_CONFIGS.get_config_file.cache_clear()
        fm.cfg_2stages = None
        fm.get_2stage_cfgs.cache_clear()

    clear_configs()
    try:
        with environment("AITER_CONFIG_FMOE", str(tuned)), environment("AITER_BYPASS_TUNE_CONFIG", "0"), \
                environment("AITER_KSPLIT", "0"):
            for shape in shapes:
                lookup = {"activation" if name == "act_type" else name: value for name, value in shape.items()}
                beta, linear_beta, _ = fm._normalize_mxfp4_activation_params(
                    shape["act_type"], args.beta, args.linear_beta, args.swiglu_limit)
                metadata = fm.get_2stage_cfgs(**lookup, hidden_pad=0, intermediate_pad=0,
                                             gate_mode=args.gate_mode, situ_beta=beta,
                                             situ_linear_beta=linear_beta, swiglu_limit=args.swiglu_limit)
                key = (fm.get_gfx_runtime(), fm.get_cu_num(), *(value if isinstance(value, (int, bool)) else str(value)
                                                             for value in shape.values()))
                config = fm.cfg_2stages[0].get(key)
                if config is None or not 0 < float(config["us"]) < math.inf:
                    raise RuntimeError(f"Aiter has no valid tuned config for {shape}")
                # 查到 CSV 不等于使用它：拒绝被 dispatcher 丢弃后退回 heuristic 的情况。
                actual = []
                for stage, op in enumerate((metadata.stage1, metadata.stage2), 1):
                    kw = getattr(op, "keywords", {})
                    name = kw.get("kernelName", kw.get(f"kernelName{stage}", ""))
                    if getattr(op, "func", None) is fm.cktile_moe_stage2:
                        name = f"cktile_a8w4_bm{metadata.block_m}"
                    actual.append(name)
                count = 1 if metadata.run_1stage else 2
                if (actual[:count] != [config[f"kernelName{stage}"] for stage in range(1, count + 1)] or
                        (metadata.block_m, metadata.ksplit, metadata.run_1stage) !=
                        (config["block_m"], config["ksplit"], bool(config["run_1stage"]))):
                    raise RuntimeError(f"Aiter did not select the tuned kernels for {shape}: {actual}")
            print(f"Aiter: verified {len(shapes)} tuned configs from {tuned}", flush=True)
            yield str(tuned)
    finally:
        clear_configs()


def prepare(model, tokens, args):
    return prepare_moe(model, tokens, dtype=args.dtype, quant=args.quant, activation=args.activation,
                       gate_mode=args.gate_mode, preshuffle=args.preshuffle, routing=args.routing,
                       seed=args.seed, beta=args.beta, linear_beta=args.linear_beta,
                       swiglu_limit=args.swiglu_limit)


def measure(op, call, reference, args, *, allow_incorrect=False):
    return measure_moe(op, call, reference, iters=args.iters, warmup=args.warmup,
                       copies=args.copies, allow_incorrect=allow_incorrect)


def run_case(name, model, tokens, args):
    """运行一个模型和 batch size：准备数据、调优并校验，再对比性能。"""
    import torch
    from pyhip.ops.moe import tuned_moe as tm

    row = dict(model=name, tokens=tokens, model_dim=model["HIDDEN_SIZE"],
               inter_dim=model["INTER_SIZE"], tp=model["TP"], inter_dim_tp=model["INTER_SIZE"] // model["TP"],
               experts=model["E"], topk=model["TOPK"], dtype=args.dtype, gate_mode=args.gate_mode,
               use_g1u1=args.activation != "gelu",
               preshuffle=args.preshuffle, routing=args.routing, seed=args.seed,
               activation=args.activation, swiglu_limit=args.swiglu_limit, beta=args.beta,
               linear_beta=args.linear_beta, correctness={}, rounds={"aiter": [], "tuned": []}, winner=None)
    previous_record = tm.record_dispatch
    try:
        tm.record_dispatch = False
        call, kind = prepare(model, tokens, args)
        row["quant"] = kind
        reference = torch_reference(call)
        if not torch.isfinite(reference).all().item():
            raise ValueError("non-finite reference")
        ops = {"aiter": tm._aiter_fused_moe, "tuned": tm.fused_moe}
        for backend, op in ops.items():
            try:
                tm.record_dispatch = backend == "tuned"
                if tm.record_dispatch:
                    tm.last_dispatch = None
                call["output"].fill_(float("nan"))
                # Validate current FlyDSL source without reusing disk-compiled kernels.
                with environment("FLYDSL_RUNTIME_ENABLE_CACHE", "0"), environment("FLYDSL_AUTOTUNE", "1" if backend == "tuned" and (args.retune or args.tune_aiter) else "0"):
                    result = op(**call)
                row["correctness"][backend] = check_output(result, call["output"], reference)
                if backend == "tuned":
                    row["winner"] = None if tm.last_dispatch is None else tm.last_dispatch.copy()
            except Exception as error:
                torch.cuda.synchronize()  # 检查异步 GPU 错误；非法访存等致命错误继续向外抛出。
                row["correctness"][backend] = dict(status="ERROR", reason=str(error))
            finally:
                tm.record_dispatch = False  # 正式计时不记录 dispatch。
        if row["correctness"]["aiter"]["status"] == "ERROR" or row["correctness"]["tuned"]["status"] != "PASS":
            row["status"] = "NOT_COMPARABLE"
            row["reason"] = "Performance comparison skipped: Aiter must execute with valid output metadata and winner must pass accuracy validation."
            return row
        if args.check_only:
            row["status"] = "CHECK_PASS" if row["correctness"]["aiter"]["status"] == "PASS" else "AITER_INCORRECT"
            return row

        # 相同参数已经完成调优；关闭强制搜索，计时只运行已选中的配置。
        with environment("FLYDSL_AUTOTUNE", "0"):
            for round_id in range(args.rounds):
                # 轮流采用 A→T 和 T→A 的顺序，减小固定先后顺序对结果的影响。
                order = ("aiter", "tuned") if round_id % 2 == 0 else ("tuned", "aiter")
                for backend in order:
                    stats = measure(ops[backend], call, reference, args, allow_incorrect=backend == "aiter")
                    row["rounds"][backend].append(stats)
                    check = stats["correctness"]
                    if check.get("diff", math.inf) >= row["correctness"][backend].get("diff", math.inf):
                        row["correctness"][backend] = check
        flops = (6 if row["use_g1u1"] else 4) * tokens * model["TOPK"] * model["HIDDEN_SIZE"] * row["inter_dim_tp"]
        for backend in ops:
            # 汇总所有轮次的样本后再取中位数，不单独挑最快的一轮。
            samples = [t for run in row["rounds"][backend] for t in run["samples_us"]]
            median = statistics.median(samples)
            row[backend] = dict(median_us=median, mean_us=statistics.mean(samples),
                                min_us=min(samples), max_us=max(samples),
                                effective_tflops=flops / (median * 1e6))
        row["speedup"] = row["aiter"]["median_us"] / row["tuned"]["median_us"]
        row["status"] = "PASS" if row["correctness"]["aiter"]["status"] == "PASS" else "AITER_INCORRECT"
        if row["status"] == "AITER_INCORRECT":
            row["reason"] = "Aiter accuracy failed; speedup compares timings only, not equivalent correct results."
    except Exception as error:
        row.update(status="ERROR", reason=f"{type(error).__name__}: {error}")
        torch.cuda.synchronize()
    finally:
        tm.record_dispatch = previous_record
    return row


def print_table(rows, *, file=None):
    """并列展示精度与时延；Aiter 数值失败的比较明确标注。"""
    print("\n| model | M | H / I_tp / E / topk | check A/T | Aiter diff | winner diff | Aiter us | tuned us | speedup | winner | winner TFLOPS | status | winner config |", file=file)
    print("|---|---:|---|---|---:|---:|---:|---:|---:|---|---:|---|---|", file=file)
    for row in rows:
        times = [f"{row[b]['median_us']:.2f}" if b in row else "—" for b in ("aiter", "tuned")]
        checks = "/".join(row["correctness"].get(b, {}).get("status", "—") for b in ("aiter", "tuned"))
        diffs = []
        for backend in ("aiter", "tuned"):
            check = row["correctness"].get(backend, {})
            diffs.append(f"{check['diff']:.6g}" if "diff" in check else
                         "NaN/Inf" if check.get("status") == "INCORRECT" else "—")
        speed = f"{row['speedup']:.3f}x" if row.get("speedup") is not None else "—"
        winner = (row.get("winner") or {}).get("_impl", "—")
        tflops = row.get("tuned", {}).get("effective_tflops")
        throughput = f"{tflops:.3f}" if tflops is not None else "—"
        config = json.dumps(row["winner"], sort_keys=True) if row.get("winner") else "—"
        dims = " / ".join(str(row[k]) for k in ("model_dim", "inter_dim_tp", "experts", "topk"))
        print(f"| {row['model']} | {row['tokens']} | {dims} | {checks} | {diffs[0]} | {diffs[1]} | {times[0]} | {times[1]} | {speed} | {winner} | {throughput} | {row['status']} | {config} |", file=file)
    print(file=file)
    for row in rows:
        label = f"{row['model']} M={row['tokens']}"
        if row.get("reason"):
            print(f"[{label}] {row['reason'].splitlines()[0]}", file=file)
        for backend, check in row["correctness"].items():
            if check["status"] != "PASS":
                detail = check.get("reason", f"calc_diff={check.get('diff')}")
                print(f"[{label}] {backend}: {check['status']}: {detail.splitlines()[0]}", file=file)


def print_markdown(rows, file):
    """按模型分组，复用屏幕表格，不重新计算或测量性能。"""
    print("# MoE performance", file=file)
    print("\nWinner TFLOPS uses the measured full tuned-call median latency.", file=file)
    groups = {}
    for row in rows:
        groups.setdefault(row["model"], []).append(row)
    if not groups:
        print("\nNo completed cases.", file=file)
    for group in groups.values():
        row = group[0]
        print(f"\n### {row['model']} ({row['dtype']}/{row.get('quant', '—')}, "
              f"{row['activation']}, TP={row['tp']})", file=file)
        fields = ("model_dim", "inter_dim", "inter_dim_tp", "experts", "topk", "gate_mode",
                  "preshuffle", "routing", "seed", "swiglu_limit", "beta", "linear_beta")
        print("\n" + ", ".join(f"{name}={row[name]}" for name in fields if row.get(name) is not None), file=file)
        print_table(group, file=file)


def main(argv=None):
    """解析命令行参数，只在运行测试时设置默认 CUDA 设备。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=[*MOE_MODELS, "all"], default=["all"])
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 4, 16, 64, 256, 1024, 4096, 8192, 16384, 32768])
    parser.add_argument("--dtype", choices=("bf16", "fp8", "mxfp4"), default="fp8")
    parser.add_argument("--quant", choices=("model", "ptpc", "per_tensor", "block"), default="model")
    parser.add_argument("--activation", choices=("silu", "swiglu", "situv2", "gelu"), default="silu",
                        help="gelu selects non-gated (G1U0) BF16 MoE; other activations use G1U1")
    parser.add_argument("--swiglu-limit", type=float)
    parser.add_argument("--beta", type=float)
    parser.add_argument("--linear-beta", type=float)
    parser.add_argument("--gate-mode", choices=("separated", "interleave"), default="separated")
    parser.add_argument("--preshuffle", choices=("on", "off"), default="on")
    parser.add_argument("--routing", choices=("balanced", "random"), default="balanced")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--retune", action="store_true", help="force a search before timing each exact M")
    parser.add_argument("--tune-aiter", nargs="?", type=Path, const=Path("tuned_aiter_moe"), metavar="DIR",
                        help="run the official Aiter tuner first; save untuned.csv/tuned.csv in DIR (default: tuned_aiter_moe)")
    parser.add_argument("--copies", type=int, default=0, help="run_perftest copies; 0 keeps its automatic ~4GB cap")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=2, help="alternate A/T then T/A blocks")
    parser.add_argument("--output", type=Path, help="new JSON file; refuses to overwrite an existing report")
    parser.add_argument("--md", type=Path, metavar="FILE",
                        help="new Markdown performance report grouped by model; refuses to overwrite")
    args = parser.parse_args(argv)
    if args.list_models:
        print("model                     model_dim  inter_dim  TP  inter_dim_tp  experts  topk  fp8_quant")
        for name, shape in MOE_MODELS.items():
            print(f"{name:25} {shape['HIDDEN_SIZE']:9} {shape['INTER_SIZE']:10} {shape['TP']:3} "
                  f"{shape['INTER_SIZE']//shape['TP']:13} {shape['E']:8} {shape['TOPK']:5} {shape['quant_type']}")
        return 0
    if any(n <= 0 for n in args.tokens) or args.iters <= 0 or args.rounds <= 0 or min(args.warmup, args.copies) < 0:
        parser.error("tokens/iters/rounds must be positive; warmup/copies must be nonnegative")
    if args.gate_mode == "interleave" and (args.dtype != "mxfp4" or args.preshuffle != "on"):
        parser.error("interleave requires preshuffled MXFP4 weights")
    if args.dtype != "fp8" and args.quant != "model":
        parser.error("--quant only overrides FP8 quantization")
    if args.activation == "gelu":
        if args.dtype != "bf16" or args.gate_mode != "separated":
            parser.error("--activation gelu requires --dtype bf16 --gate-mode separated (non-gated MoE)")
        if any(value is not None for value in (args.swiglu_limit, args.beta, args.linear_beta)):
            parser.error("GELU does not use swiglu-limit/beta/linear-beta")
        if args.tune_aiter:
            parser.error("--tune-aiter does not support GELU G1U0; the official tuner only tunes G1U1")
    if args.tune_aiter and args.preshuffle != "on":
        parser.error("--tune-aiter requires --preshuffle on; the official tuner benchmarks shuffled weights")
    if args.output is not None and args.output.exists():
        parser.error("report already exists; use a new --output path")
    if args.md is not None and args.md.exists():
        parser.error("report already exists; use a new --md path")
    if args.output is not None and args.md is not None and args.output.resolve() == args.md.resolve():
        parser.error("--output and --md must use different paths")

    import aiter
    import torch
    from pyhip.testing import misc

    if not torch.cuda.is_available() or torch.version.hip is None:
        parser.error("ROCm PyTorch and an AMD GPU are required")
    if not args.check_only and misc.CUDAPERF is not None:
        parser.error("unset CUDAPERF so the timing context cannot be disabled")
    # 测试不需要 CPU tensor；在这里设置默认设备，避免 import 或查看帮助时产生副作用。
    torch.set_default_device("cuda")
    aiter.logger.setLevel("ERROR")
    models = list(MOE_MODELS) if "all" in args.models else list(dict.fromkeys(args.models))
    rows = []
    report = dict(settings={name: str(value) if isinstance(value, Path) else value
                            for name, value in vars(args).items()},
                  rows=rows)
    try:
        with environment("AITER_ONLINE_TUNE", "0"), environment("FLYDSL_AUTOTUNE", "0"), \
                (tune_aiter(models, args) if args.tune_aiter else nullcontext()) as tuned_file:
            if tuned_file is not None:
                report["aiter_tuned_file"] = tuned_file
            for name in models:
                for tokens in dict.fromkeys(args.tokens):
                    print(f"\n=== {name}, M={tokens} ===", flush=True)
                    row = run_case(name, MOE_MODELS[name], tokens, args)
                    rows.append(row)
    finally:
        print_table(rows)
        if args.md is not None:
            args.md.parent.mkdir(parents=True, exist_ok=True)
            with args.md.open("x", encoding="utf-8") as stream:
                print_markdown(rows, stream)
            print(f"Markdown report: {args.md}")
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x") as stream:
                json.dump(report, stream, indent=2, allow_nan=False)
                stream.write("\n")
            print(f"Report: {args.output}")
    if args.tune_aiter:
        print(f"Tuned Aiter folder: {args.tune_aiter}")
    return 0 if rows and all(row["status"] in ("CHECK_PASS", "PASS") for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())