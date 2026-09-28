# SPDX-License-Identifier: MIT
"""Check decode/prefill accuracy, then print separate performance tables.

Direct execution checks all selected batches before running the existing
samplers in bench_gr_read_compare.py. --check-only skips performance.
No result file is created unless --output is supplied; pytest checks accuracy only.
"""
import argparse
from contextlib import nullcontext
from functools import cache
import gc
import json
import os
from pathlib import Path
import sys
import warnings

C, H, R = 4, 2560, 320
CHECK_ROWS = 1024
DOWN_TOLERANCE = dict(rtol=0.015625, atol=2e-5)
OUTPUT_TOLERANCE = dict(rtol=1e-2, atol=5e-3)
DEFAULT_BATCHES = (33, 64, 128, 256, 512) + tuple(k * 1024 for k in (1, 2, 4, 8, 10, 12, 16, 20, 24, 28, 30, 32, 36, 48, 60, 64))
SCOPES = ("down", "up", "total")


def parse_batch(value):
    text = value.strip().lower()
    try:
        rows = int(text[:-1]) * 1024 if text.endswith("k") else int(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("batch must be an integer or an integer followed by K") from error
    if rows < 1:
        raise argparse.ArgumentTypeError("batch must be a positive integer")
    return rows


def prepare_cli_environment(args):
    """Select the GPU before importing Torch without installing dependencies or changing hardware."""
    repo = Path(__file__).resolve().parents[3]
    paths = (repo, repo / "src", Path("/opt/aiter"),
             Path(f"/usr/local/lib/python{sys.version_info.major}.{sys.version_info.minor}/dist-packages"))
    for path in paths:
        if path.is_dir() and str(path) not in sys.path:
            sys.path.append(str(path))
    if os.environ.get("HSA_CU_MASK") or os.environ.get("ROC_GLOBAL_CU_MASK"):
        raise RuntimeError("GRRead reference tests require an unmasked GPU")
    for key in ("ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL", "CUDAPERF", "COMPILE_ONLY",
                "FLYDSL_DUMP_IR", "FLYDSL_DEBUG_DUMP_ASM", "FLYDSL_DUMP_DIR", "FLYDSL_RUNTIME_RUN_ONLY"):
        os.environ.pop(key, None)
    os.environ["HIP_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["FLYDSL_RUNTIME_ENABLE_CACHE"] = "1"


def use_checkout_package():
    import importlib.util
    repo = Path(__file__).resolve().parents[3]
    expected = repo / "src/pyhip/__init__.py"
    module = sys.modules.get("pyhip")
    if module is not None:
        if Path(module.__file__).resolve() != expected:
            raise RuntimeError(f"wrong PyHIP checkout: {module.__file__}")
        return
    spec = importlib.util.spec_from_file_location("pyhip", expected,
                                                  submodule_search_locations=[str(repo / "src/pyhip")])
    module = importlib.util.module_from_spec(spec)
    sys.modules["pyhip"] = module
    spec.loader.exec_module(module)


def warn_architecture(torch, device=None):
    arch = torch.cuda.get_device_properties(device).gcnArchName
    if arch.split(":", 1)[0] != "gfx942":
        warnings.warn(f"GR read was validated on gfx942; running on {arch}",
                      RuntimeWarning, stacklevel=2)


@cache
def dependencies():
    # 延迟导入：CLI先选GPU；--help和参数解析不初始化Torch，也不依赖pytest。
    use_checkout_package()
    import torch
    import torch.nn.functional as F
    import flydsl.compiler as flyc
    from pyhip.testing import cudaPerf
    if torch.version.hip is None or not torch.cuda.is_available():
        raise RuntimeError("ROCm GPU required")
    warn_architecture(torch, 0)

    @torch.compile(fullgraph=True)
    def _mix_reference(x, w_down, w_up):
        p = F.silu(F.linear(x, w_down) / C)
        logits = F.linear(p, w_up)
        gates = torch.sigmoid(logits).unflatten(-1, (C, H))
        return p, (gates * x.unflatten(-1, (C, H))).mean(dim=-2)

    return torch, flyc, _mix_reference, cudaPerf


@cache
def baseline_modules():
    """Load this checkout's test-only module in script and pytest import modes."""
    import importlib.util
    name = '_pyhip_gr_read_baselines'
    path = Path(__file__).with_name('baselines.py')
    if name in sys.modules:
        module = sys.modules[name]
        if Path(module.__file__).resolve() != path.resolve():
            raise RuntimeError('baseline module belongs to a different checkout')
        return module
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@cache
def torch_compile_mix():
    """Full-row performance baseline; independent from the P/Y accuracy reference."""
    dependencies()
    return baseline_modules().torch_compiled('prefill')


def prepare_reader(x, w_down, w_up):
    dependencies()
    from pyhip.ops.gr_read.flydsl import GRReadPrefill, prepare_weights
    packed_down, packed_up = prepare_weights(w_down, w_up)
    reader = GRReadPrefill(x.shape[0], packed_down, packed_up)
    reader._check_input(x)
    return reader


def prepare_gr_read(x, w_down, w_up):
    """Compatibility tuple for historical scripts; current tests use the public object."""
    reader = prepare_reader(x, w_down, w_up)
    return (reader.w_down, reader.w_up, reader.partial, reader.output,
            reader.down, reader.up, reader.down_config, reader.up_config[1])


def run_down(x, w_down, partial, down):
    """Launch Down with prepared buffers and return the BF16 intermediate."""
    torch, _, _, _ = dependencies()
    if x.shape[0]:
        with torch.cuda.device(x.device):
            down(x, w_down, partial, x.shape[0], torch.cuda.current_stream(x.device))
    return partial


def run_up(x, w_up, partial, output, up):
    """Launch Up with prepared buffers and return the BF16 output."""
    torch, _, _, _ = dependencies()
    if x.shape[0]:
        with torch.cuda.device(x.device):
            up(x, w_up, partial, output, x.shape[0], torch.cuda.current_stream(x.device))
    return output


def run_gr_read(x, w_down, w_up, partial, output, down, up):
    """Launch Down then Up on the current stream, without host row splitting."""
    torch, _, _, _ = dependencies()
    if x.shape[0]:
        with torch.cuda.device(x.device):
            stream = torch.cuda.current_stream(x.device)
            down(x, w_down, partial, x.shape[0], stream)
            up(x, w_up, partial, output, x.shape[0], stream)
    return output


def reference_bf16(x, w_down, w_up):
    torch, _, mix, _ = dependencies()
    activation = torch.empty((x.shape[0], R), device=x.device, dtype=torch.bfloat16)
    output = torch.empty((x.shape[0], H), device=x.device, dtype=torch.bfloat16)
    with torch.autocast("cuda", enabled=False):
        for begin in range(0, x.shape[0], CHECK_ROWS):
            end = min(begin + CHECK_ROWS, x.shape[0])
            p, y = mix(x[begin:end], w_down, w_up)
            assert p.dtype == y.dtype == torch.bfloat16
            activation[begin:end], output[begin:end] = p, y
    return activation, output


def check_close(actual, expected, tolerance):
    torch, _, _, _ = dependencies()
    assert actual.shape == expected.shape
    error_squared = expected_squared = 0.0
    for begin in range(0, actual.shape[0], CHECK_ROWS):
        a, e = actual[begin:begin + CHECK_ROWS].double(), expected[begin:begin + CHECK_ROWS].double()
        torch.testing.assert_close(a, e, **tolerance)
        error_squared += (a - e).square().sum().item()
        expected_squared += e.square().sum().item()
    return (error_squared / expected_squared if expected_squared else error_squared) ** 0.5


def make_inputs(torch, rows, seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn((rows, 10240), device="cuda", dtype=torch.bfloat16, generator=generator)
    wd = torch.randn((320, 10240), device="cuda", dtype=torch.bfloat16, generator=generator) * 0.02
    wu = torch.randn((10240, 320), device="cuda", dtype=torch.bfloat16, generator=generator) * 0.02
    return x, wd, wu


def check_batch(rows, args):
    torch, _, _, _ = dependencies()
    with torch.no_grad():
        x, wd, wu = make_inputs(torch, rows, args.seed)
        reader = prepare_reader(x, wd, wu)
        partial, output = reader.partial, reader.output
        dm, dw, dn, dk = reader.down_config
        um, un = reader.up_config
        p_expected, y_expected = reference_bf16(x, wd, wu)
        assert partial.dtype == output.dtype == torch.bfloat16
        assert partial.shape == (rows, R)
        scope = getattr(args, "scope", "all")
        partial.fill_(torch.nan)
        assert reader.run_down(x) is partial
        p_error = check_close(partial, p_expected, DOWN_TOLERANCE)
        y_error = None
        if scope in ("up", "all"):
            output.fill_(torch.nan)
            assert reader.run_up(x) is output
            y_error = check_close(output, y_expected, OUTPUT_TOLERANCE)
        if scope in ("total", "all"):
            partial.fill_(torch.nan); output.fill_(torch.nan)
            assert reader(x) is output
            check_close(partial, p_expected, DOWN_TOLERANCE)
            y_error = check_close(output, y_expected, OUTPUT_TOLERANCE)
        if scope in ("total", "all"):
            from pyhip.ops.gr_read import gr_read
            actual = gr_read(x, reader.w_down, reader.w_up)
            if 0 < rows <= 32:
                assert_decode_close(actual, decode_reference(x, wd, wu), 'public exact-row decode')
            else:
                check_close(actual, y_expected, OUTPUT_TOLERANCE)
            assert gr_read(x, reader.w_down, reader.w_up, output=output) is output
            torch.testing.assert_close(output, actual, rtol=0, atol=0)
        result = {"complete": True, "rows": rows, "down_n_splits": dn,
                  "down_block_m": dm, "down_num_waves": dw, "down_block_k": dk,
                  "up_block_m": um, "n_splits": un, "P_rel_l2": p_error, "Y_rel_l2": y_error,
                  "P_tolerance": DOWN_TOLERANCE, "Y_tolerance": OUTPUT_TOLERANCE,
                  "partial_shape": list(partial.shape), "checked_scopes": list(SCOPES) if scope == "all" else [scope]}
        if getattr(args, "verbose", False):
            print(f"T={rows} prefill {scope}: P rel_l2={p_error:.6g}, Y rel_l2={y_error} PASS", flush=True)
        return result


def decode_reference(x, w_down, w_up):
    import torch
    (x, wd, wu) = (x.double(), w_down.double(), w_up.double())
    hidden = torch.nn.functional.silu(x @ wd.T * 0.25)
    gates = torch.sigmoid(hidden @ wu.T).reshape(-1, 4, 2560)
    return (gates * x.reshape(-1, 4, 2560)).mean(dim=1)


def make_decode_inputs(rows, seed):
    import torch
    gen = torch.Generator(device='cuda').manual_seed(seed)

    def randn(*shape):
        return torch.randn(*shape, dtype=torch.bfloat16, device='cuda', generator=gen)
    return (randn(rows, 10240), randn(320, 10240) * 0.02, randn(10240, 320) * 0.02)


def capture_decode(calls):
    import torch
    for _ in range(2):
        for call in calls:
            call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with warnings.catch_warnings(record=True) as messages:
        warnings.simplefilter('always')
        with torch.cuda.graph(graph):
            for call in calls:
                call()
    for message in messages:
        if 'graph is empty' in str(message.message).lower():
            raise RuntimeError('empty graph: launch did not use the capture stream')
        warnings.warn(str(message.message), message.category)
    return graph


def guarded(shape, dtype, device):
    import math
    import torch
    storage = torch.full((math.prod(shape) + 32,), 97, dtype=dtype, device=device)
    return (storage[16:-16].view(shape), storage)


def assert_decode_close(actual, expected, label):
    import torch
    assert torch.allclose(actual.double(), expected, rtol=0.01, atol=0.005), label


def check_decode_rows(rows, wd, wu, pd, pu, seed):
    import torch
    use_checkout_package()
    from pyhip.ops.gr_read.flydsl import GRReadDecode
    from pyhip.ops.gr_read import gr_read
    from pyhip.ops.gr_read.flydsl.common import K, R, H as HS
    reader = GRReadDecode(rows, pd, pu)
    assert reader.w_down.data_ptr() == pd.data_ptr()
    assert reader.w_up.data_ptr() == pu.data_ptr()
    assert reader.partial.shape == (4 * rows * R,), "decode P must be compact"
    (x, sx) = guarded((rows, K), torch.bfloat16, pd.device)
    (reader.partial, sp) = guarded(reader.partial.shape, torch.float32, pd.device)
    (reader.output, sy) = guarded((rows, HS), torch.bfloat16, pd.device)
    (api_output, api_storage) = guarded((rows, HS), torch.bfloat16, pd.device)
    gen = torch.Generator(device=pd.device).manual_seed(seed)
    x.normal_(generator=gen)
    assert_decode_close(reader(x), decode_reference(x, wd, wu), f'T={rows}: eager FP64 mismatch')
    assert gr_read(x, pd, pu, output=api_output) is api_output
    torch.testing.assert_close(api_output, reader.output, rtol=0, atol=0)
    graph = capture_decode([lambda : reader(x), lambda : gr_read(x, pd, pu, output=api_output)])
    (count, maximum) = (0, 0.0)
    for tail in ('zero', 'stale', 'nan'):
        x.normal_(generator=gen)
        for live in list(range(rows, -1, -1)) + list(range(1, rows + 1)):
            x[:live].normal_(generator=gen)
            if tail == 'zero':
                x[live:].zero_()
            elif tail == 'nan':
                x[live:].fill_(float('nan'))
            before = x.clone()
            reader.partial.fill_(float('nan'))
            reader.output.fill_(float('nan'))
            api_output.fill_(float('nan'))
            graph.replay()
            expected = decode_reference(x[:live], wd, wu)
            actual = reader.output[:live].double()
            label = f'T={rows}, live={live}, tail={tail}'
            assert_decode_close(actual, expected, label)
            assert_decode_close(api_output[:live], expected, label + ': public API')
            torch.testing.assert_close(api_output[:live], reader.output[:live], rtol=0, atol=0)
            if live:
                error = ((actual - expected).abs() / (0.005 + 0.01 * expected.abs())).max().item()
                maximum = max(maximum, error)
            partial = reader.partial.view(4, rows, R)
            assert torch.isfinite(partial[:, :live]).all(), label + ': live scratch'
            if tail == 'zero':
                assert torch.count_nonzero(partial[:, live:]) == 0, label + ': zero scratch tail'
                assert torch.count_nonzero(reader.output[live:]) == 0, label + ': zero output tail'
            elif tail == 'stale':
                assert torch.isfinite(partial).all(), label + ': stale scratch'
                assert torch.isfinite(reader.output).all(), label + ': stale output'
            assert torch.allclose(x, before, rtol=0, atol=0, equal_nan=True), label + ': input mutated'
            for storage in (sx, sp, sy, api_storage):
                assert torch.all(storage[:16] == 97) and torch.all(storage[-16:] == 97), label + ': guard overwritten'
            count += 1
    return {'rows': rows, 'replays': count, 'public_api_replays': count, 'max_scaled_error': maximum,
            'partial_shape': [4, rows, R], 'compact_partial_guards_passed': True, 'passed': True}


def check_decode_stages(rows, wd, wu, pd, pu, seed, scope="all"):
    import torch
    import flydsl.compiler as flyc
    from pyhip.ops.gr_read.flydsl import GRReadDecode
    from pyhip.ops.gr_read.flydsl.down import make_decode_down
    from pyhip.ops.gr_read.flydsl.up import make_decode_up

    x = make_decode_inputs(rows, seed)[0]
    reader = GRReadDecode(rows, pd, pu)
    assert reader.partial.shape == (4 * rows * 320,), "decode P must be compact"
    reader.partial, partial_storage = guarded(reader.partial.shape, torch.float32, pd.device)
    stream = torch.cuda.current_stream()
    down = flyc.compile(make_decode_down(rows), x.view(-1), pd, reader.partial, stream)
    up = flyc.compile(make_decode_up(rows), x.view(-1), pu, reader.partial, reader.output.view(-1), stream)
    for state in range(2):
        if state: x.mul_(0.99).add_(0.015625)
        before = x.clone()
        reader.partial.fill_(float("nan"))
        down(x.view(-1), pd, reader.partial, stream)
        partial = reader.partial.view(4, rows, 320)
        if scope in ("down", "all"):
            for split in range(4):
                begin, end = split * 2560, (split + 1) * 2560
                expected = x[:, begin:end].double() @ wd[:, begin:end].double().T
                torch.testing.assert_close(partial[split, :rows].double(), expected, rtol=2e-5, atol=1e-5)
        assert torch.all(partial_storage[:16] == 97) and torch.all(partial_storage[-16:] == 97), "Down overwrote compact P guard"
        if scope in ("up", "all"):
            hidden = torch.nn.functional.silu(partial[:, :rows].double().sum(0) * .25)
            expected = (torch.sigmoid(hidden @ wu.double().T).reshape(rows, 4, 2560)
                        * x.double().reshape(rows, 4, 2560)).mean(1)
            reader.output.fill_(float("nan"))
            up(x.view(-1), pu, reader.partial, reader.output.view(-1), stream)
            assert_decode_close(reader.output, expected, "isolated decode Up from actual P")
        assert torch.all(partial_storage[:16] == 97) and torch.all(partial_storage[-16:] == 97), "Up overwrote compact P guard"
        assert torch.equal(before, x)
    return {"rows": rows, "scope": scope, "input_states": 2, "passed": True}


def release_buffers():
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def selected_rows(parser, args):
    if args.rows is not None:
        if args.phase == "all": parser.error("use --decode-rows / --prefill-rows with --phase all")
        if args.decode_rows is not None or args.prefill_rows is not None:
            parser.error("--rows cannot be combined with phase-specific rows")
        if args.phase == "decode": args.decode_rows = args.rows
        else: args.prefill_rows = args.rows
    args.decode_rows = args.decode_rows or list(range(1, 33))
    args.prefill_rows = args.prefill_rows or list(DEFAULT_BATCHES)
    if any(not 1 <= r <= 32 for r in args.decode_rows): parser.error("decode rows must be in 1..32")
    if any(r < 33 for r in args.prefill_rows): parser.error("prefill benchmark rows start at 33")
    if any(len(set(rs)) != len(rs) for rs in (args.decode_rows, args.prefill_rows)):
        parser.error("rows must be distinct")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("all", "decode", "prefill"), default="all")
    parser.add_argument("--scope", choices=(*SCOPES, "all"), default="all",
                        help="accuracy scope; a single scope implies --check-only")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--rows", "--batches", nargs="+", type=parse_batch)
    parser.add_argument("--decode-rows", nargs="+", type=parse_batch)
    parser.add_argument("--prefill-rows", nargs="+", type=parse_batch)
    parser.add_argument("--decode-weights", type=int, default=2, help="accuracy weight pairs; performance retains 100")
    parser.add_argument("--decode-seed", type=int, default=303, help="accuracy seed; performance retains seed 707")
    parser.add_argument("--prefill-seed", type=int, default=131)
    parser.add_argument("--seed", type=int, help="override the selected single phase")
    parser.add_argument("--weights", type=int, help="alias for --decode-weights in decode-only mode")
    parser.add_argument("--check-only", action="store_true", help="check accuracy only; skip hardware gates and timing")
    parser.add_argument("--no-baselines", action="store_true", help="skip frozen performance baselines in both phases; accuracy references still run")
    parser.add_argument("--amd-smi", type=Path, help="compatible amd-smi CLI for performance hardware gates")
    parser.add_argument("--output", type=Path, help="optional new JSONL file; default: console only")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    selected_rows(parser, args)
    if args.seed is not None:
        if args.phase == "all": parser.error("use --decode-seed / --prefill-seed with --phase all")
        setattr(args, args.phase + "_seed", args.seed)
    if args.weights is not None:
        if args.phase != "decode": parser.error("--weights requires --phase decode")
        args.decode_weights = args.weights
    if args.gpu < 0 or args.decode_weights < 1 or not __debug__:
        parser.error("nonnegative GPU and positive weight count required; do not use python -O")
    args.check_only = args.check_only or args.scope != "all"
    return args


def run_correctness(args, emit):
    """Keep the original accuracy workloads, completing both phases before timing."""
    use_checkout_package()
    import torch
    from pyhip.ops.gr_read.flydsl import prepare_weights
    if torch.version.hip is None or not torch.cuda.is_available():
        raise RuntimeError("ROCm GPU required")
    if args.phase in ("all", "decode"):
        print(f"Accuracy: decode {len(args.decode_rows)} batches, {args.decode_weights} weight pairs, scope={args.scope}", flush=True)
        total = 0
        with torch.inference_mode():
            for i in range(args.decode_weights):
                _, wd, wu = make_decode_inputs(1, args.decode_seed + i)
                pd, pu = prepare_weights(wd, wu)
                before_down, before_up = pd.clone(), pu.clone()
                for index, rows in enumerate(args.decode_rows, 1):
                    print(f"[Decode accuracy {i + 1}/{args.decode_weights}, {index}/{len(args.decode_rows)}] T={rows}", flush=True)
                    seed = args.decode_seed + i * 10000 + rows
                    if args.scope in ("down", "up", "all"):
                        result = check_decode_stages(rows, wd, wu, pd, pu, seed, args.scope)
                        emit({"type": "stage_check", "phase": "decode", "weight": i, **result})
                    if args.scope in ("total", "all"):
                        result = check_decode_rows(rows, wd, wu, pd, pu, seed)
                        total += result["replays"]
                        emit({"type": "check", "phase": "decode", "weight": i, **result})
                    assert torch.equal(pd, before_down) and torch.equal(pu, before_up), "packed weights mutated"
        emit({"type": "summary", "phase": "decode", "complete": True, "rows": args.decode_rows, "replays": total})
        print(f"Decode: {len(args.decode_rows)} shapes, {args.decode_weights} weight pairs, "
              f"{total} Graph replays with prepared + functional calls PASS", flush=True)
    if args.phase in ("all", "prefill"):
        for index, rows in enumerate(args.prefill_rows, 1):
            print(f"[Prefill accuracy {index}/{len(args.prefill_rows)}] T={rows}, scope={args.scope}", flush=True)
            result = check_batch(rows, argparse.Namespace(seed=args.prefill_seed, scope=args.scope, verbose=args.verbose))
            emit({"type": "check", "phase": "prefill", **result})
        emit({"type": "summary", "phase": "prefill", "complete": True, "rows": args.prefill_rows})
        print(f"Prefill: {len(args.prefill_rows)} shapes, {args.scope} PASS", flush=True)


@cache
def benchmarking():
    import importlib.util
    path = Path(__file__).with_name("bench_gr_read_compare.py")
    spec = importlib.util.spec_from_file_location("gr_read_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def benchmark_options(args):
    # Preserve the independent decode accuracy (2/303) and performance (100/707)
    # workloads. Other sampling options remain on the standalone benchmark CLI.
    options = ["--phase", args.phase, "--gpu", str(args.gpu),
               "--decode-rows", *map(str, args.decode_rows),
               "--prefill-rows", *map(str, args.prefill_rows),
               "--prefill-seed", str(args.prefill_seed)]
    for name in ("amd_smi",):
        value = getattr(args, name)
        if value is not None:
            options.extend(("--" + name.replace("_", "-"), str(value)))
    if args.no_baselines:
        options.append("--no-baselines")
    if args.verbose:
        options.append("--verbose")
    return benchmarking().parse_args(options)


def main(argv=None):
    args = parse_args(argv)
    performance = None if args.check_only else benchmark_options(args)
    prepare_cli_environment(args)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
    with (args.output.open("x") if args.output else nullcontext()) as out:
        def emit(record):
            if out:
                out.write(json.dumps({"stage": "accuracy", **record}, allow_nan=False) + "\n")
                out.flush()
        if performance is not None:
            bench = benchmarking()
            def emit_performance(phase, record):
                bench.emit_record(out, phase, {"stage": "performance", **record})
            bench.benchmark_entry_gate(performance, emit_performance)
        print(f"[1/{1 if args.check_only else 2}] Accuracy checks for all selected batches", flush=True)
        run_correctness(args, emit)
        release_buffers()
        if performance is not None:
            print("\n[2/2] All selected batches passed accuracy; starting performance.", flush=True)
            bench.run_benchmarks(
                performance,
                emit_performance,
                prefill_checked=True,
                hardware_checked=True,
            )
        else:
            print("Accuracy complete; performance skipped (--check-only or single --scope).", flush=True)
    return 0


if "pytest" in sys.modules:
    import pytest


    @pytest.mark.parametrize("rows", (0, 1, 31, 32, 47, 48, 49, 63, 65, 127, 129, 255, 257, 511, 513,
                                      1023, 1025, 2047, 2049, 2560, 2561, 3072, 3073,
                                      4095, 4097, 65537, *DEFAULT_BATCHES))
    def test_gr_read(rows):
        torch = pytest.importorskip("torch")
        if torch.version.hip is None or not torch.cuda.is_available():
            pytest.skip("ROCm GPU required")
        warn_architecture(torch)
        try:
            check_batch(rows, argparse.Namespace(seed=131))
        finally:
            release_buffers()


    def test_shared_weights_and_decode_devices():
        """Public decode/prefill instances share packing and keep device-local launches."""
        import pytest
        torch = pytest.importorskip('torch')
        if torch.version.hip is None or not torch.cuda.is_available():
            pytest.skip('ROCm required')
        warn_architecture(torch, 0)
        use_checkout_package()
        from pyhip.ops.gr_read.flydsl import GRReadDecode, GRReadPrefill, prepare_weights
        from pyhip.ops.gr_read.flydsl.common import prepare_weights as shared_prepare
        assert prepare_weights is shared_prepare
        with torch.no_grad(), torch.cuda.device(0):
            (x, wd, wu) = make_decode_inputs(64, 4917)
            (pd, pu) = prepare_weights(wd, wu)
            saved_weights = (pd.clone(), pu.clone())
            (decode, prefill) = (GRReadDecode(16, pd, pu), GRReadPrefill(64, pd, pu))
            assert decode.w_down.data_ptr() == prefill.w_down.data_ptr() == pd.data_ptr()
            assert decode.w_up.data_ptr() == prefill.w_up.data_ptr() == pu.data_ptr()
            assert decode.partial.dtype == torch.float32 and prefill.partial.dtype == torch.bfloat16
            graph = capture_decode([lambda : decode(x[:16])])
            for state in range(2):
                if state:
                    x.mul_(0.99).add_(0.015625)
                graph.replay()
                assert_decode_close(decode.output, decode_reference(x[:16], wd, wu), 'decode shared packing')
                assert_decode_close(prefill(x), decode_reference(x, wd, wu), 'prefill shared packing')
            if torch.cuda.device_count() >= 2:
                warn_architecture(torch, 1)
                second = GRReadDecode(16, pd.to('cuda:1'), pu.to('cuda:1'))
                actual = second(x[:16].to('cuda:1')).to('cuda:0')
                torch.testing.assert_close(actual, decode.output, rtol=0, atol=0)
                assert torch.cuda.current_device() == 0
                with torch.cuda.device(1):
                    decode(x[:16])
                    assert torch.cuda.current_device() == 1
            assert torch.equal(pd, saved_weights[0]) and torch.equal(pu, saved_weights[1])


    def api_inputs(rows=17, seed=1729):
        torch = pytest.importorskip('torch')
        pytest.importorskip('flydsl')
        if torch.version.hip is None or not torch.cuda.is_available():
            pytest.skip('ROCm required')
        use_checkout_package()
        from pyhip.ops.gr_read import prepare_weights
        x, wd, wu = make_decode_inputs(rows, seed)
        return x, wd, wu, *prepare_weights(wd, wu)


    def test_api_cache_and_output_lifetime(monkeypatch):
        import weakref
        import torch
        x, wd, wu, pd, pu = api_inputs()
        from pyhip.ops.gr_read.flydsl import host
        monkeypatch.setattr(host, '_compiled_calls', {})
        calls = []
        compile_call = host._compile_call

        def observed(rows, *args):
            calls.append(rows)
            return compile_call(rows, *args)

        monkeypatch.setattr(host, '_compile_call', observed)
        y = host.gr_read(x, pd, pu)
        assert_decode_close(y, decode_reference(x, wd, wu), 'cold call')
        before = y.clone()
        out = torch.empty_like(y)
        x.mul_(.75)
        assert host.gr_read(x, pd, pu, output=out) is out
        assert_decode_close(out, decode_reference(x, wd, wu), 'changed input')
        assert torch.equal(y, before) and y.data_ptr() != out.data_ptr()
        assert calls == [17]
        host.gr_read(x[:12], pd, pu)
        assert calls == [17, 12], 'do not compile all T below the first input'
        x2, wd2, wu2, pd2, pu2 = api_inputs(seed=77)
        assert_decode_close(host.gr_read(x2, pd2, pu2), decode_reference(x2, wd2, wu2), 'current weights')
        assert calls == [17, 12], 'compiled code must be shared across weight pairs'
        refs = [weakref.ref(t) for t in (x2, pd2, pu2)]
        del x2, pd2, pu2
        gc.collect()
        assert all(ref() is None for ref in refs), 'compiled cache must not retain caller tensors'


    def test_api_prefill_runtime_rows(monkeypatch):
        import torch
        x, wd, wu, pd, pu = api_inputs(128)
        from pyhip.ops.gr_read.flydsl import host
        monkeypatch.setattr(host, '_compiled_calls', {})
        expected = host.GRReadPrefill(33, pd, pu)(x[:33]).clone()
        torch.testing.assert_close(host.gr_read(x[:33], pd, pu), expected, rtol=0, atol=0)
        assert len(host._compiled_calls) == 1

        def unexpected_compile(*args, **kwargs):
            pytest.fail('same prefill config must reuse code for different runtime rows')

        monkeypatch.setattr(host, '_compile_call', unexpected_compile)
        for rows in (64, 65, 127, 128):
            reference = host.GRReadPrefill(rows, pd, pu)(x[:rows]).clone()
            out = torch.empty_like(reference)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                result = host.gr_read(x[:rows], pd, pu, output=out)
            graph.replay()
            assert result is out
            torch.testing.assert_close(out, reference, rtol=0, atol=0)
            assert len(host._compiled_calls) == 1


    @pytest.mark.parametrize('rows', (17, 24, 33, 513))
    def test_api_graph_workspaces(monkeypatch, rows):
        import torch
        x, wd, wu, pd, pu = api_inputs(rows)
        from pyhip.ops.gr_read.flydsl import host
        x2 = x.clone().mul_(.5)
        host.gr_read(x, pd, pu)
        allocations = []
        native_empty = torch.empty

        def guarded_empty(shape, *args, **kwargs):
            if torch.cuda.is_current_stream_capturing():
                tensor, storage = guarded(shape, kwargs['dtype'], kwargs['device'])
                allocations.append((tensor, storage))
                return tensor
            return native_empty(shape, *args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(torch, 'empty', guarded_empty)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                y = host.gr_read(x, pd, pu)
                y2 = host.gr_read(x2, pd, pu)
        assert len(allocations) == 4
        assert len({t.data_ptr() for t, _ in allocations}) == 4
        p_shape = (4 * rows * R,) if rows <= 32 else (rows, R)
        partials = [t for t, _ in allocations if tuple(t.shape) == p_shape]
        assert len(partials) == 2
        assert all(p.dtype == (torch.float32 if rows <= 32 else torch.bfloat16) for p in partials)
        for live in (rows, min(rows, 17), 0, rows):
            x[:live].mul_(.9).add_(.015625)
            x[live:].fill_(float('nan'))
            if live == rows:
                x.nan_to_num_(.125)
            before = x.clone()
            for t, _ in allocations:
                t.fill_(float('nan'))
            graph.replay()
            reference = (decode_reference(x[:live], wd, wu) if rows <= 32
                         else reference_bf16(x[:live], wd, wu)[1].double())
            assert_decode_close(y[:live], reference, 'fixed T and valid prefix')
            reference2 = (decode_reference(x2, wd, wu) if rows <= 32
                          else reference_bf16(x2, wd, wu)[1].double())
            assert_decode_close(y2, reference2, 'independent P/Y for second call')
            torch.testing.assert_close(x, before, rtol=0, atol=0, equal_nan=True)
            for _, storage in allocations:
                assert torch.all(storage[:16] == 97) and torch.all(storage[-16:] == 97)


    def test_api_empty_and_contracts(monkeypatch):
        import torch
        x, wd, wu, pd, pu = api_inputs()
        from pyhip.ops.gr_read.flydsl import host
        monkeypatch.setattr(host, '_compiled_calls', {})
        empty = x[:0]
        out = torch.empty((0, H), dtype=x.dtype, device=x.device)
        assert host.gr_read(empty, pd, pu, output=out) is out
        assert host.gr_read(empty, pd, pu).shape == (0, H)
        assert not host._compiled_calls
        output = torch.full((17, H), 13, dtype=x.dtype, device=x.device)
        bad_calls = [
            (x.float(), pd, pu, output), (x[:, ::2], pd, pu, output),
            (x, wd, pu, output), (x, pd, pu.float(), output),
            (x, pd, pu, output[:16]), (x, pd, pu, output.float()),
            (x, pd, pu, x.view(-1)[:17 * H].view(17, H)),
            (x, pd, pu, pd[:17 * H].view(17, H)),
        ]
        misaligned = torch.empty(x.numel() + 1, dtype=x.dtype, device=x.device)[1:].view_as(x)
        bad_calls.append((misaligned, pd, pu, output))
        for xi, pdi, pui, yi in bad_calls:
            with pytest.raises(ValueError):
                host.gr_read(xi, pdi, pui, output=yi)
        assert torch.all(output == 13) and not host._compiled_calls
        with pytest.raises(ValueError, match='inference-only'):
            host.gr_read(x.detach().requires_grad_(True), pd, pu)
        graph = torch.cuda.CUDAGraph()
        with pytest.raises(RuntimeError, match='warmed'), torch.cuda.graph(graph):
            host.gr_read(x, pd, pu)
        assert not host._compiled_calls


    def test_api_streams_and_devices():
        import torch
        x, wd, wu, pd, pu = api_inputs()
        from pyhip.ops.gr_read import gr_read
        gr_read(x, pd, pu)
        producer = torch.cuda.current_stream()
        streams = [torch.cuda.Stream(), torch.cuda.Stream()]
        outputs = []
        for stream in streams:
            stream.wait_stream(producer)
            with torch.cuda.stream(stream):
                outputs.append(gr_read(x, pd, pu))
        for stream in streams:
            producer.wait_stream(stream)
        assert outputs[0].data_ptr() != outputs[1].data_ptr()
        torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
        if torch.cuda.device_count() > 1:
            x1, pd1, pu1 = [t.to('cuda:1') for t in (x, pd, pu)]
            assert torch.cuda.current_device() == 0
            result = gr_read(x1, pd1, pu1)
            assert torch.cuda.current_device() == 0
            torch.testing.assert_close(result.to(x.device), outputs[0], rtol=0, atol=0)


    def test_baseline_raw_weights():
        torch = pytest.importorskip('torch')
        baselines = baseline_modules()
        wd = torch.arange(32, dtype=torch.float32).view(4, 8).to(torch.bfloat16)
        wu = wd.T.contiguous()
        a, b = baselines.prepare_weights(wd, wu)
        c, d = baselines.prepare_weights(wd, wu)
        assert all(t.is_contiguous() for t in (a, b, c, d))
        assert len({t.data_ptr() for t in (wd, wu, a, b, c, d)}) == 6
        assert torch.equal(a, wd) and torch.equal(c, wd)
        assert torch.equal(b, wu) and torch.equal(d, wu)
        with pytest.raises(ValueError, match='original 2D'):
            baselines.prepare_weights(wd.flatten(), wu.flatten())


    def test_baseline_cli_without_framework(monkeypatch):
        import importlib.util
        original = importlib.util.find_spec

        def no_framework(name, *args, **kwargs):
            if name == 'sglang' or name.startswith('sglang.'):
                pytest.fail('CLI must not discover a framework installation')
            return original(name, *args, **kwargs)

        monkeypatch.setattr(importlib.util, 'find_spec', no_framework)
        for parser in (parse_args, benchmarking().parse_args):
            assert not parser([]).no_baselines
            assert parser(['--no-baselines']).no_baselines
            assert not hasattr(parser([]), 'sglang_root')
            with pytest.raises(SystemExit):
                parser(['--sglang-root', '/does/not/exist'])
            with pytest.raises(SystemExit):
                parser(['--no-sglang'])
        assert benchmark_options(parse_args(['--no-baselines'])).no_baselines


    @pytest.mark.parametrize('rows', (1, 3, 8, 12, 16, 17, 24, 33, 65))
    def test_frozen_baselines_graph(monkeypatch, rows):
        import builtins
        import torch
        x, wd, wu, pd, pu = api_inputs(rows, seed=907)
        original_import = builtins.__import__

        def without_framework(name, *args, **kwargs):
            if name == 'sglang' or name.startswith('sglang.'):
                pytest.fail('frozen baselines must not import SGLang')
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, '__import__', without_framework)
        baselines = baseline_modules()
        triton = baselines
        twd, twu = baselines.prepare_weights(wd, wu)
        rwd, rwu = baselines.prepare_weights(wd, wu)
        assert all(a.data_ptr() != b.data_ptr() for a, b in ((pd, twd), (pu, twu), (twd, rwd), (twu, rwu)))
        supported = triton.fused_hc_mix_supported(x, rwd, rwu)
        assert supported == (rows <= 16)
        assert not triton.fused_hc_mix_supported(x, rwd, rwu, deterministic=True)
        compiled = baselines.torch_compiled('decode' if rows <= 32 else 'prefill')
        calls = [('torch.compile', lambda: compiled(x, twd, twu, C, H))]
        if supported:
            calls.append(('triton', lambda: triton.fused_hc_mix(x, rwd, rwu, C, H)))
        with torch.inference_mode():
            for label, call in calls:
                expected = decode_reference(x, wd, wu)
                assert_decode_close(call(), expected, label + ' initial')
                call()  # all first-use setup must complete before capture
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    output = call()
                for _ in range(2):
                    x.mul_(.99).add_(.015625)
                    output.fill_(float('nan'))
                    graph.replay()
                    assert_decode_close(output, decode_reference(x, wd, wu), label + ' replay')
                    if label == 'triton':
                        assert torch.count_nonzero(triton._get_counters(x.device)) == 0
        assert torch.equal(twd, wd) and torch.equal(rwd, wd)
        assert torch.equal(twu, wu) and torch.equal(rwu, wu)


if __name__ == "__main__":
    raise SystemExit(main())
