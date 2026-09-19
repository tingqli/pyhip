# SPDX-License-Identifier: MIT
"""GR read胜出prefill配置的基础功能与性能测试。

--rows 64K --gpu 4：单batch；--sweep --gpu 4：4K/8K/16K/32K/64K。
--check-only：仅功能检查。默认10 buffers、2次预热、10次采样取中位数。
性能测试只读检查空闲GPU和PTL Enabled/VECTOR,F8，不改硬件设置。
"""

import argparse
import gc
import json
import os
from pathlib import Path
import shutil
from statistics import median
import subprocess
import sys


DEFAULT_BATCHES = (4096, 8192, 16384, 32768, 65536)


def parse_batch(value):
    text = value.strip().lower()
    try:
        rows = int(text[:-1]) * 1024 if text.endswith("k") else int(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("batch must be an integer or an integer followed by K") from error
    if not 1 <= rows <= 65536:
        raise argparse.ArgumentTypeError("batch must be in 1..65536")
    return rows


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--rows", type=parse_batch, default=65536)
    mode.add_argument("--sweep", action="store_true")
    parser.add_argument("--batches", nargs="+", type=parse_batch, help="optional --sweep batch list")
    parser.add_argument("--gpu", type=int, default=4, help="physical ROCm GPU index")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--seed", type=int, default=131)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--buffers", type=int, default=10)
    parser.add_argument("--up-block-m", type=int, choices=(128, 256), default=128, help="Up M tile; default M128")
    parser.add_argument("--amd-smi", type=Path, help="AMD SMI executable or Python CLI with PTL reporting")
    # 兼容此前的单命令调用；两个名称现在都只选择同一个胜出配置。
    parser.add_argument("--implementation", choices=("prefill", "tuned"), default="prefill", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.gpu < 0 or args.warmup < 0 or args.iters < 1 or args.buffers < 1:
        parser.error("gpu/warmup must be nonnegative; iters/buffers must be positive")
    if args.batches and (not args.sweep or len(set(args.batches)) != len(args.batches)):
        parser.error("--batches requires --sweep and distinct batch sizes")
    args.batches = (args.batches or list(DEFAULT_BATCHES)) if args.sweep else [args.rows]
    return args


def prepare_cli_environment(args):
    """选卡在Torch导入前完成；只复用现有容器依赖，不安装或改硬件。"""
    repo = Path(__file__).resolve().parents[3]
    paths = [repo / "src", Path("/opt/aiter"),
             Path(f"/usr/local/lib/python{sys.version_info.major}.{sys.version_info.minor}/dist-packages")]
    for path in paths:
        if path.is_dir() and str(path) not in sys.path:
            sys.path.append(str(path))
    for key in ("ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL", "CUDAPERF", "COMPILE_ONLY",
                "FLYDSL_DUMP_IR", "FLYDSL_DEBUG_DUMP_ASM", "FLYDSL_DUMP_DIR", "FLYDSL_RUNTIME_RUN_ONLY"):
        os.environ.pop(key, None)
    os.environ["HIP_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["FLYDSL_RUNTIME_ENABLE_CACHE"] = "1"


if __name__ == "__main__":
    _CLI_ARGS = parse_args()
    prepare_cli_environment(_CLI_ARGS)

import pytest
import torch
import torch.nn.functional as F
from pyhip.misc import cudaPerf

if __package__:
    from .combined_host import CombinedPaddedGRRead
    from .kernel import C, H, K, R, MAX_ROWS
else:
    from combined_host import CombinedPaddedGRRead
    from kernel import C, H, K, R, MAX_ROWS


CHECK_ROWS = 1024
# 两种GEMM累加顺序在BF16中点可选相邻值，经SiLU和再次舍入后P可差2 ULP。
# 采用2*BF16 eps相对容差；最终输出容差保持原值，不要求库GEMM逐位一致。
DOWN_TOLERANCE = dict(rtol=2 * torch.finfo(torch.bfloat16).eps, atol=2e-5)
OUTPUT_TOLERANCE = dict(rtol=1e-2, atol=5e-3)


@torch.compile(fullgraph=True)
def _mix_reference(x, w_down, w_up):
    """原sglang _mix_compute表达式；保留原torch.compile的BF16边界和逐元素融合。"""
    p = F.silu(F.linear(x, w_down) / C)
    logits = F.linear(p, w_up)
    gates = torch.sigmoid(logits).unflatten(-1, (C, H))
    return p, (gates * x.unflatten(-1, (C, H))).mean(dim=-2)


def reference_bf16(x, w_down, w_up):
    """按行分块覆盖全部原始BF16输入；参考计算和编译不计入kernel时延。"""
    activation = torch.empty((x.shape[0], R), device=x.device, dtype=torch.bfloat16)
    output = torch.empty((x.shape[0], H), device=x.device, dtype=torch.bfloat16)
    with torch.autocast("cuda", enabled=False):
        for begin in range(0, x.shape[0], CHECK_ROWS):
            end = min(begin + CHECK_ROWS, x.shape[0])
            p, y = _mix_reference(x[begin:end], w_down, w_up)
            assert p.dtype == y.dtype == torch.bfloat16
            activation[begin:end], output[begin:end] = p, y
    return activation, output


def check_close(actual, expected, tolerance):
    """基础逐元素容差检查，同时输出真实relative L2，不混用calc_diff。"""
    assert actual.shape == expected.shape
    error_squared = expected_squared = 0.0
    for begin in range(0, actual.shape[0], CHECK_ROWS):
        a, e = actual[begin:begin + CHECK_ROWS].double(), expected[begin:begin + CHECK_ROWS].double()
        torch.testing.assert_close(a, e, **tolerance)
        error_squared += (a - e).square().sum().item()
        expected_squared += e.square().sum().item()
    return (error_squared / expected_squared if expected_squared else error_squared) ** 0.5


def require_reference_hardware(gpu, amd_smi=None):
    """只读性能门禁；功能测试不要求空闲GPU或PTL。"""
    query = subprocess.run(["rocm-smi", "-d", str(gpu), "--showuse", "--showmemuse", "--showperflevel",
                            "--showmaxpower", "--json"], text=True, capture_output=True, check=True, timeout=30)
    card = json.loads(query.stdout)[f"card{gpu}"]
    use, vram = int(card["GPU use (%)"]), int(card["GPU Memory Allocated (VRAM%)"])
    if use > 5 or vram > 20:
        raise RuntimeError(f"GPU{gpu} busy: use={use}%, VRAM={vram}%; stop without retry")
    bundled = Path("/tmp/amd-smi-lib-26.2.2-rocm-7.2.3/opt/rocm-7.2.3/libexec/amdsmi_cli/amdsmi_cli.py")
    cli = amd_smi or (bundled if bundled.is_file() else shutil.which("amd-smi"))
    if cli is None:
        raise RuntimeError("PTL reporting requires a compatible --amd-smi")
    cli = Path(cli).resolve()
    env = os.environ.copy()
    command = [str(cli)]
    if cli.suffix == ".py":
        command = [sys.executable, str(cli)]
        base = cli.parents[2]
        if (base / "share/amd_smi").is_dir():
            env["PYTHONPATH"] = os.pathsep.join((str(base / "share/amd_smi"), str(cli.parent)))
            env["LD_LIBRARY_PATH"] = os.pathsep.join((str(base / "lib"), str(base / "share/amd_smi/amdsmi"), env.get("LD_LIBRARY_PATH", "")))
    ptl = subprocess.run(command + ["static", "-g", str(gpu), "--limit", "--json"],
                         env=env, text=True, capture_output=True, check=True, timeout=30)
    limit = next(item["limit"] for item in json.loads(ptl.stdout)["gpu_data"] if str(item["gpu"]) == str(gpu))
    formats = {part.strip().upper() for part in str(limit.get("ptl_format")).split(",")}
    if str(limit.get("ptl_state")).lower() != "enabled" or formats != {"VECTOR", "F8"}:
        raise RuntimeError("Timing requires PTL Enabled/VECTOR,F8; no hardware settings were changed")
    print(f"GPU{gpu}: PTL Enabled/VECTOR,F8, use={use}%, VRAM={vram}%, "
          f"perf={card['Performance Level']}, cap={card['Max Graphics Package Power (W)']} W; unchanged")


@torch.no_grad()
def run_test(rows=129, seed=131, *, benchmark=True, num_warmup=2, num_iters=10, num_buffers=10, up_block_m=128):
    """基础Down/Output正确性检查，再可选计时Down、Up和直接Total。"""
    if not 1 <= rows <= MAX_ROWS or num_warmup < 0 or num_iters < 1 or num_buffers < 1:
        raise ValueError("invalid rows or timing parameters")
    generator = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn((rows, K), device="cuda", dtype=torch.bfloat16, generator=generator)
    wd = torch.randn((R, K), device="cuda", dtype=torch.bfloat16, generator=generator) * 0.02
    wu = torch.randn((K, R), device="cuda", dtype=torch.bfloat16, generator=generator) * 0.02
    reader = CombinedPaddedGRRead(rows, wd, wu, implementation="prefill", up_block_m=up_block_m)
    assert reader.partial.dtype == torch.bfloat16 and reader.partial.element_size() == 2
    down_expected, expected = reference_bf16(x, wd, wu)

    def validate(instance, scope):
        assert instance.partial.dtype == instance.output.dtype == torch.bfloat16
        if scope == "down":
            values = instance.partial.view(-1, R)
            error = check_close(values[:rows], down_expected, DOWN_TOLERANCE)
            assert bool((values[rows:] == 0).all().item()), "Down padding must be zero"
            return error
        return check_close(instance.output, expected, OUTPUT_TOLERANCE)

    # 投毒后执行真实两kernel路径，避免未写区域误用旧结果。
    reader.partial.fill_(torch.nan)
    reader.output.fill_(torch.nan)
    assert reader(x) is reader.output
    down_error, output_error = validate(reader, "down"), validate(reader, "up")
    props = torch.cuda.get_device_properties(x.device)
    print(f"T={rows}, Up M{up_block_m}; {props.name} ({props.gcnArchName}); Down rel_l2={down_error:.6g}, Output rel_l2={output_error:.6g}: PASS")
    print("Reference: 原BF16 torch.compile算子链；kernel P=BF16，Up FP32 logits直接sigmoid，Y整数helper舍入；按原容差验收。")
    timings = {}
    if not benchmark:
        return timings

    # 所有实际计时的X/已shuffle权重/P/Y都轮换；JIT、准备、校验均在计时外。
    buffers = [(x, reader)] + [(x.clone(), CombinedPaddedGRRead(rows, wd, wu, implementation="prefill", up_block_m=up_block_m))
                               for _ in range(num_buffers - 1)]
    for input_x, instance in buffers:
        instance.partial.fill_(torch.nan)
        instance.output.fill_(torch.nan)
        instance(input_x)
        validate(instance, "down")
        validate(instance, "up")

        padded = reader.padded_rows  # Down/Up共用所选Up tile对齐的workspace。
        up_tasks = padded // up_block_m * 2
        user_slots = props.multi_processor_count * 2
        print(f"Up任务数={up_tasks}，按{props.multi_processor_count}CU*2={user_slots}份额："
            f"{up_tasks // user_slots}轮+{up_tasks % user_slots}尾任务；仅任务数模型，不是CU归属实测。")
    effective = {"down": 2 * rows * K * R, "up": 2 * rows * K * R, "total": 4 * rows * K * R}
    executed = {"down": 2 * padded * K * R, "up": 2 * padded * K * R, "total": 4 * padded * K * R}
    for scope in ("down", "up", "total"):
        perf = cudaPerf(name=f"gr_read_{scope}", verbose=0)
        if not perf.enable:
            raise RuntimeError("CUDAPERF disables GR read timing")
        for index in range(num_warmup + num_iters):
            input_x, instance = buffers[index % num_buffers]
            launch = instance.run_down if scope == "down" else instance.run_up if scope == "up" else instance
            with perf:
                launch(input_x)
        samples = [seconds * 1e6 for seconds in perf.latencies[num_warmup:]]
        elapsed = median(samples)
        timings[scope] = {"elapsed_us": elapsed, "samples_us": samples,
                          "executed_tflops": executed[scope] / elapsed / 1e6,
                          "effective_tflops": effective[scope] / elapsed / 1e6}
        if scope == "up":
            # 每logit：sigmoid缩放乘/加1、gate乘、从零累加各1 FLOP；每Y再乘1/C。
            # W-first唯一有效X/Y；P被两个N分片读取，W每所选M tile CTA各读一次。
            # 这是源码请求模型，不是PMC实测HBM流量，不包含LDS或OOB请求。
            request_bytes = 4 * rows * R + 2 * K * R * (padded // up_block_m) + 2 * rows * K + 2 * rows * H
            timings[scope].update(up_gate_tflops=(executed[scope] + 4 * padded * K + padded * H) / elapsed / 1e6,
                                  exp2_gops_per_s=padded * K / elapsed / 1e3, rcp_gops_per_s=padded * K / elapsed / 1e3,
                                  requested_bytes=request_bytes, request_tbps=request_bytes / elapsed / 1e6)
        for _, instance in buffers:
            validate(instance, scope)  # 直接检查计时输出，不重跑combined覆盖它。

    print(f"cudaPerf: warmup={num_warmup}, iters={num_iters}, buffers={num_buffers}, median; Total直接计时，不相加。")
    print("MFMA FLOPs: Down=Up=2*Tpad*K*R, Total=4*Tpad*K*R；有效FLOPs分别2/2/4*T*K*R。")
    print("Up+gate另加4*Tpad*K+Tpad*H；exp2/rcp各Tpad*K次单列，不计普通FLOPs。")
    print("| Scope | ms | MFMA TFLOPS | Effective TFLOPS | Up+gate TFLOPS |")
    print("|---|---:|---:|---:|---:|")
    for scope, stats in timings.items():
        fused = f"{stats['up_gate_tflops']:.3f}" if scope == "up" else "-"
        print(f"| {scope} | {stats['elapsed_us'] / 1000:.6f} | {stats['executed_tflops']:.3f} | {stats['effective_tflops']:.3f} | {fused} |")
    print(f"Up exp2={timings['up']['exp2_gops_per_s']:.3f}, rcp={timings['up']['rcp_gops_per_s']:.3f} Gop/s（完整Up时延）。")
    print(f"Up请求带宽={timings['up']['request_tbps']:.3f} TB/s，模型字节数={timings['up']['requested_bytes']}；非实测HBM带宽。")
    return timings


def test_gr_read():
    """仅一个基础用例：两种M各检查1行全padding CTA和129行尾块。"""
    if torch.version.hip is None or not torch.cuda.is_available():
        pytest.skip("ROCm GPU required")
    if not torch.cuda.get_device_properties().gcnArchName.startswith("gfx942"):
        pytest.skip("gfx942 required")
    for up_block_m in (128, 256):
        run_test(rows=1, benchmark=False, up_block_m=up_block_m)
        run_test(rows=129, benchmark=False, up_block_m=up_block_m)


def main(args):
    results = []
    for rows in args.batches:
        if not args.check_only:
            require_reference_hardware(args.gpu, args.amd_smi)
        timings = run_test(rows, args.seed, benchmark=not args.check_only,
                           num_warmup=args.warmup, num_iters=args.iters, num_buffers=args.buffers, up_block_m=args.up_block_m)
        results.append((rows, timings))
        # reader的bound-method dispatch有引用环；batch之间释放，不进入计时区。
        gc.collect()
        torch.cuda.empty_cache()
    if args.sweep and not args.check_only:
        print("\n| Batch | Down ms | Up ms | Total ms | Up+gate TFLOPS |")
        print("|---:|---:|---:|---:|---:|")
        for rows, timings in results:
            print(f"| {rows} | {timings['down']['elapsed_us'] / 1000:.6f} | {timings['up']['elapsed_us'] / 1000:.6f} | "
                  f"{timings['total']['elapsed_us'] / 1000:.6f} | {timings['up']['up_gate_tflops']:.3f} |")
    return results


if __name__ == "__main__":
    main(_CLI_ARGS)