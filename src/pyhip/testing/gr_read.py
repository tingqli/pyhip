# SPDX-License-Identifier: MIT
"""Shared GR read inputs, references and accuracy checks for tests and benchmarks.

GPU/compiler imports remain lazy. This module does not import repository test
scripts, benchmark scripts, frozen baselines or an inference framework.
"""
import argparse
from functools import cache
import gc
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



def validate_report_paths(parser, output, markdown):
    for option, path in (('--output', output), ('--md', markdown)):
        if path is not None and path.exists():
            parser.error(f'{option} already exists; choose a new report path: {path}')
    if output is not None and markdown is not None and output.resolve() == markdown.resolve():
        parser.error('--output and --md must use different paths')


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
    timer_module = sys.modules.get("pyhip.testing.misc")
    if timer_module is not None:
        timer_module.CUDAPERF = None



def warn_architecture(torch, device=None):
    arch = torch.cuda.get_device_properties(device).gcnArchName
    if arch.split(":", 1)[0] != "gfx942":
        warnings.warn(f"GR read was validated on gfx942; running on {arch}",
                      RuntimeWarning, stacklevel=2)



@cache
def dependencies():
    # 延迟导入：CLI先选GPU；--help和参数解析不初始化Torch，也不依赖pytest。
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



def run_correctness(args, emit):
    """Keep the original accuracy workloads, completing both phases before timing."""
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
