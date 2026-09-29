# SPDX-License-Identifier: MIT
"""Print separate decode Graph and prefill eager benchmark tables.

Defaults preserve the two PR #36 protocols. Frozen test baselines are optional.
Progress is printed during setup
and between measurements, followed by the final tables. No result files are
created unless --output or --md is supplied. See readme.md for integration and
the measured MI308X performance snapshot.
"""
import argparse
from contextlib import contextmanager, nullcontext, redirect_stdout
from datetime import datetime, timezone
from functools import cache
import gc
import hashlib
import importlib.util
from importlib import metadata as package_metadata
import io
import json
import math
import os
from pathlib import Path
from statistics import median
from types import SimpleNamespace
import shutil
import subprocess
import sys
import time
import warnings
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'src'))
C, H, R = 4, 2560, 320
SCOPES = ('down', 'up', 'total', 'baseline')
TIMING_SCOPES = ('down', 'up', 'total', 'torch_compile')


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@cache
def testing():
    from pyhip.testing import gr_read
    if Path(gr_read.__file__).resolve() != REPO / 'src/pyhip/testing/gr_read.py':
        raise RuntimeError(f'wrong PyHIP checkout: {gr_read.__file__}')
    return gr_read


@cache
def baseline_modules():
    """Load this checkout's benchmark baseline module in script and pytest import modes."""
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
    testing().dependencies()
    return baseline_modules().torch_compiled('prefill')


def source_hashes(no_baselines):
    paths = [Path(__file__), Path(testing().__file__),
             REPO / 'src/pyhip/ops/gr_read/__init__.py',
             *(REPO / 'src/pyhip/ops/gr_read/flydsl').glob('*.py')]
    if not no_baselines:
        paths.append(Path(__file__).with_name('baselines.py'))
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def dependencies(no_baselines=False):
    prefill = testing()
    import torch
    import flydsl.compiler as flyc
    from pyhip.testing import cudaPerf
    from pyhip.ops.gr_read.flydsl import GRReadDecode, prepare_weights
    from pyhip.ops.gr_read.flydsl.down import make_decode_down
    from pyhip.ops.gr_read.flydsl.up import make_decode_up
    decode = SimpleNamespace(GRReadDecode=GRReadDecode, prepare_weights=prepare_weights,
                             down_launcher=make_decode_down, up_launcher=make_decode_up)
    check = SimpleNamespace(reference=reference, assert_close=assert_close, make_inputs=make_inputs, capture=capture)
    if no_baselines:
        return torch, flyc, cudaPerf, decode, check, prefill, None, None
    baselines = baseline_modules()
    return (torch, flyc, cudaPerf, decode, check, prefill,
            baselines.torch_compiled('decode'), baselines)


def reference(x, w_down, w_up):
    import torch
    x, wd, wu = x.double(), w_down.double(), w_up.double()
    hidden = torch.nn.functional.silu((x @ wd.T) * 0.25)
    gates = torch.sigmoid(hidden @ wu.T).reshape(-1, 4, 2560)
    return (gates * x.reshape(-1, 4, 2560)).mean(dim=1)


def assert_close(actual, expected, label):
    import torch
    torch.testing.assert_close(actual.double(), expected.double(), rtol=1e-2, atol=5e-3, msg=label)


def make_inputs(rows, seed):
    import torch
    generator = torch.Generator(device='cuda').manual_seed(seed)
    def randn(*shape):
        return torch.randn(*shape, dtype=torch.bfloat16, device='cuda', generator=generator)
    return randn(rows, 10240), randn(320, 10240) * 0.02, randn(10240, 320) * 0.02


def capture(calls):
    import torch
    for _ in range(2):
        for call in calls: call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with warnings.catch_warnings(record=True) as messages:
        warnings.simplefilter('always')
        with torch.cuda.graph(graph):
            for call in calls: call()
    for message in messages:
        if 'graph is empty' in str(message.message).lower():
            raise RuntimeError('empty graph: launch did not use the capture stream')
        warnings.warn(str(message.message), message.category)
    return graph


def graph_samples(torch, graph, calls, count):
    # Same timer/divisor as the existing H64 benchmark: 2 passes, 3 replays/sample.
    for _ in range(3):
        graph.replay()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    values = []
    for _ in range(count):
        start.record()
        for _ in range(3):
            graph.replay()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end) * 1000 / (calls * 3))
    return values


def address(tensor):
    return {'pointer': tensor.data_ptr(), 'storage_offset': tensor.storage_offset(),
            'storage_base': tensor.untyped_storage().data_ptr(),
            'mod256': tensor.data_ptr() % 256, 'mod4096': tensor.data_ptr() % 4096,
            'shape': list(tensor.shape), 'dtype': str(tensor.dtype)}


def read_hardware(gpu, amd_smi=None):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("ROCPROF", "ROCP_", "AQLPROFILE")) and k not in (
        "LD_PRELOAD", "HSA_TOOLS_LIB", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL")}
    query = subprocess.run(["rocm-smi", "-d", str(gpu), "--showuse", "--showmemuse", "--showperflevel",
                            "--showmaxpower", "--showbus", "--json"],
                           env=env, text=True, capture_output=True, check=True, timeout=30)
    card = json.loads(query.stdout)[f"card{gpu}"]
    bundled = Path("/tmp/amd-smi-lib-26.2.2-rocm-7.2.3/opt/rocm-7.2.3/libexec/amdsmi_cli/amdsmi_cli.py")
    cli = amd_smi or (bundled if bundled.is_file() else shutil.which("amd-smi"))
    if cli is None:
        raise RuntimeError("PTL reporting requires a compatible --amd-smi")
    cli = Path(cli).resolve()
    command = [str(cli)]
    if cli.suffix == ".py":
        command = [sys.executable, str(cli)]
        base = cli.parents[2]
        if (base / "share/amd_smi").is_dir():
            env["PYTHONPATH"] = os.pathsep.join((str(base / "share/amd_smi"), str(cli.parent)))
            env["LD_LIBRARY_PATH"] = os.pathsep.join((str(base / "lib"), str(base / "share/amd_smi/amdsmi"), env.get("LD_LIBRARY_PATH", "")))
    query = subprocess.run(command + ["static", "-g", str(gpu), "--limit", "--json"],
                           env=env, text=True, capture_output=True, check=True, timeout=30)
    limit = next(item["limit"] for item in json.loads(query.stdout)["gpu_data"] if str(item["gpu"]) == str(gpu))
    return {"gpu": gpu, "card": card, "limit": limit, "settings_written": False}


def validate_hardware(snapshot):
    card, limit = snapshot["card"], snapshot["limit"]
    use, vram = int(card["GPU use (%)"]), int(card["GPU Memory Allocated (VRAM%)"])
    if use > 5 or vram > 20:
        raise RuntimeError(f"GPU{snapshot['gpu']} busy: use={use}%, VRAM={vram}%; stop without retry")
    formats = {part.strip().upper() for part in str(limit.get("ptl_format")).split(",")}
    if str(limit.get("ptl_state")).lower() != "enabled" or formats != {"VECTOR", "F8"}:
        warnings.warn(f"Expected PTL Enabled/VECTOR,F8; continuing with "
                      f"{limit.get('ptl_state')}/{limit.get('ptl_format')}",
                      RuntimeWarning, stacklevel=2)


def tensor_address(tensor, *, output=False):
    pointer, base = tensor.data_ptr(), tensor.untyped_storage().data_ptr()
    if output:
        assert tensor.is_contiguous() and tensor.storage_offset() == 0 and pointer == base
        assert pointer % 256 == 0, "performance Y must start at its aligned allocation base"
    return {"pointer": pointer, "storage_base": base, "storage_offset": tensor.storage_offset(),
            "mod256": pointer % 256, "mod4096": pointer % 4096}


def benchmark_entry_gate(args, emit):
    # Check once, before correctness, allocations or warmup. SMI utilization can
    # lag behind our own GPU work, so it is not an idle gate between batches.
    time.sleep(args.settle_seconds)
    snapshot = read_hardware(args.gpu, args.amd_smi)
    emit('setup', {'type': 'hardware', 'event_phase': 'entry',
                   'hardware_policy': 'entry_only', 'settle_seconds': args.settle_seconds,
                   **snapshot})
    print(f"GPU{args.gpu} startup check: use={snapshot['card']['GPU use (%)']}%, "
          f"VRAM={snapshot['card']['GPU Memory Allocated (VRAM%)']}%; "
          "hardware checked once before GPU work.", flush=True)
    if args.verbose:
        print(snapshot, flush=True)
    validate_hardware(snapshot)
    return snapshot


class Case:

    def __init__(self, rows, x, wd, wu, args, dep, packed=None):
        (torch, flyc, _, decode, _, prefill, compiled_mix, triton_mix) = dep
        (self.torch, self.x, self.wd, self.wu) = (torch, x, wd, wu)
        (self.phase, self.rows, self.prefill) = (args.phase, rows, prefill)
        (pd, pu) = packed if packed is not None else decode.prepare_weights(wd, wu)
        reader = decode.GRReadDecode(rows, pd, pu)
        from pyhip.ops.gr_read import gr_read
        self.gr_read = gr_read
        self.reader = reader
        (self.pd, self.pu, self.p, self.y) = (pd, pu, reader.partial, reader.output)
        self.down = flyc.compile(decode.down_launcher(rows), x.view(-1), pd, self.p, torch.cuda.current_stream(x.device))
        self.up = flyc.compile(decode.up_launcher(rows), x.view(-1), pu, self.p, self.y.view(-1), torch.cuda.current_stream(x.device))
        self.baseline_backend = self.baseline_call = None
        self.baseline_wd = self.baseline_wu = None
        if triton_mix is not None:
            self.baseline_wd, self.baseline_wu = baseline_modules().prepare_weights(wd, wu)
            self.baseline_backend = 'triton' if triton_mix.fused_hc_mix_supported(x, self.baseline_wd, self.baseline_wu) else 'torch.compile'
            self.baseline_call = triton_mix.fused_hc_mix if self.baseline_backend == 'triton' else compiled_mix
        self.baseline_y = None

    def run(self, scope):
        torch = self.torch
        if scope == 'baseline':
            self.baseline_y = self.baseline_call(self.x, self.baseline_wd, self.baseline_wu, 4, 2560)
            return self.baseline_y
        stream = torch.cuda.current_stream(self.x.device)
        if scope == 'down':
            self.down(self.x.view(-1), self.pd, self.p, stream)
        elif scope == 'up':
            self.up(self.x.view(-1), self.pu, self.p, self.y.view(-1), stream)
        else:
            self.gr_read(self.x, self.pd, self.pu, output=self.y)
        return self.p if scope == 'down' else self.y


@contextmanager
def observe_partial_allocations(shape, dtype, device):
    """Temporarily observe native allocations, retaining only address integers."""
    import torch
    native_empty = torch.empty
    partials = []

    def observed_empty(*args, **kwargs):
        tensor = native_empty(*args, **kwargs)
        if tensor.device == device and tensor.dtype == dtype and tuple(tensor.shape) == shape:
            # No tensor/storage references, metadata dictionaries or GPU work.
            partials.append((tensor.data_ptr(), tensor.untyped_storage().data_ptr(),
                             tensor.storage_offset()))
        return tensor

    with patch.object(torch, 'empty', observed_empty):
        yield partials


def partial_address_record(partials, shape, dtype, x, packed_down, packed_up, output):
    """Build JSON metadata after the call; eager callers must finish timing first."""
    if len(partials) != 1:
        raise RuntimeError(f'expected one Total P allocation, observed {len(partials)}')
    pointer, storage_base, storage_offset = partials[0]
    return {'P_total': {'pointer': pointer, 'storage_base': storage_base,
                        'storage_offset': storage_offset, 'mod256': pointer % 256,
                        'mod4096': pointer % 4096, 'shape': list(shape), 'dtype': str(dtype)},
            'relative_bytes': {'P_minus_' + name: pointer - tensor.data_ptr()
                               for name, tensor in (('X', x), ('WD_packed', packed_down),
                                                    ('WU_packed', packed_up), ('Y', output))}}


def capture_total_call(case, buffer_index, buffer_count, records):
    """Observe the native P allocation only during this Total graph call."""
    torch = case.torch
    if not torch.cuda.is_current_stream_capturing():
        return case.run('total')
    shape = (4 * case.rows * R,)
    with observe_partial_allocations(shape, torch.float32, case.x.device) as partials:
        output = case.run('total')
    call_index = len(records)
    records.append({'call_index': call_index, 'pass_index': call_index // buffer_count,
                    'buffer': buffer_index,
                    'logical_shape': [4, case.rows, R],
                    **partial_address_record(partials, shape, torch.float32,
                                             case.x, case.pd, case.pu, case.y)})
    return output


def output_check(torch, actual, expected, label):
    torch.testing.assert_close(actual.double(), expected.double(), rtol=1e-2, atol=5e-3,
                               msg=label)


def fp64_check(case, reference):
    # Chunking keeps the large-prefill reference memory bounded.
    for begin in range(0, case.rows, 1024):
        end = min(begin + 1024, case.rows)
        expected = reference(case.x[begin:end], case.wd, case.wu)
        output_check(case.torch, case.y[begin:end], expected, 'PyHIP vs FP64')
        if case.baseline_call is not None:
            output_check(case.torch, case.baseline_y[begin:end], expected, 'frozen baseline vs FP64')


def decode_stage_check(case):
    torch = case.torch
    rows = case.rows
    x, wd, wu = case.x.double(), case.wd.double(), case.wu.double()
    partial = case.p.view(4, rows, 320)
    case.run('down')
    maximum = 0.0
    for split in range(4):
        expected = x[:, split * 2560:(split + 1) * 2560] @ wd[:, split * 2560:(split + 1) * 2560].T
        actual = partial[split, :rows].double()
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=1e-5)
        maximum = max(maximum, (actual - expected).abs().max().item())
    hidden = torch.nn.functional.silu(partial[:, :rows].double().sum(0) * 0.25)
    expected = (torch.sigmoid(hidden @ wu.T).reshape(rows, 4, 2560)
                * x.reshape(rows, 4, 2560)).mean(1)
    case.y.fill_(float('nan'))
    case.run('up')
    output_check(torch, case.y, expected, 'isolated Up vs FP64 from actual P')
    return maximum


def check_timed(cases, scope):
    for c in cases:
        if scope == 'down':
            c.torch.testing.assert_close(c.p, c.saved_p, rtol=0, atol=0)
        elif scope in ('up', 'total'):
            c.torch.testing.assert_close(c.y, c.saved_y, rtol=0, atol=0)
        else:
            # SGLang's persistent atomic reduction is not required to be bitexact.
            output_check(c.torch, c.baseline_y, c.saved_baseline, 'timed frozen baseline output')


def benchmark_decode_rows(rows, pairs, args, dep, emit, packed_pairs=None):
    (torch, _, cudaPerf, _, check, prefill, _, _) = dep
    scopes = SCOPES if dep[-1] is not None else SCOPES[:-1]
    gen = torch.Generator(device='cuda').manual_seed(args.seed + rows)
    cases = [Case(rows, torch.randn(rows, 10240, device='cuda', dtype=torch.bfloat16, generator=gen), wd, wu, args, dep, packed_pairs[i] if packed_pairs is not None else None) for (i, (wd, wu)) in enumerate(pairs)]
    stage_errors = []
    for (index, c) in enumerate(cases):
        c.run('total')
        if c.baseline_call is not None:
            c.run('baseline')
        fp64_check(c, check.reference)
        if args.phase == 'decode' and index < 2:
            stage_errors.append(decode_stage_check(c))
        (c.saved_p, c.saved_y, c.saved_baseline) = (c.p.clone(), c.y.clone(), c.baseline_y.clone() if c.baseline_y is not None else None)
    graphs = {}
    total_calls = []
    for scope in scopes:
        if scope == 'total' and args.record_partials:
            calls = [lambda c=c, bi=bi: capture_total_call(c, bi, len(cases), total_calls)
                     for bi, c in enumerate(cases)]
        else:
            calls = [lambda c=c, scope=scope: c.run(scope) for c in cases]
        graphs[scope] = check.capture(calls + calls)
        graphs[scope].replay()
        check_timed(cases, scope)
    if args.record_partials:
        assert len(total_calls) == 2 * len(cases), 'record every Total call in both graph passes'
    emit({'type': 'correctness', 'phase': 'initial', 'rows': rows, 'pairs': len(pairs), 'fp64_passed': True, 'isolated_down_max_abs': stage_errors, 'isolated_up_fp64_passed': bool(stage_errors)})
    if args.record_partials:
        emit({'type': 'addresses', 'rows': rows, 'total_entry': 'gr_read',
              'total_partial': 'actual capture allocations in total_calls; P_stages belongs to Down/Up stage graphs',
              'total_calls': total_calls,
              'buffers': [{k: address(v) for (k, v) in (('X', c.x), ('WD_raw', c.wd), ('WU_raw', c.wu), ('WD_packed', c.pd), ('WU_packed', c.pu), ('P_stages', c.p), ('Y', c.y), ('Baseline_WD', c.baseline_wd), ('Baseline_WU', c.baseline_wu), ('Baseline_Y', c.baseline_y)) if v is not None} for c in cases]})
    torch.cuda.synchronize()
    values = {scope: [] for scope in scopes}
    for round_index in range(args.rounds):
        order = scopes if round_index % 2 == 0 else scopes[::-1]
        for scope in order:
            samples = graph_samples(torch, graphs[scope], len(cases) * 2, args.samples)
            values[scope].extend(samples)
            emit({'type': 'samples', 'rows': rows, 'round': round_index, 'scope': scope, 'us': samples, 'mode': 'cuda_graph'})
            check_timed(cases, scope)
    for c in cases:
        c.x.mul_(0.99).add_(0.015625)
    if graphs:
        graphs['total'].replay()
        if 'baseline' in graphs:
            graphs['baseline'].replay()
    else:
        for c in cases:
            c.run('total')
            if c.baseline_call is not None:
                c.run('baseline')
    for c in cases:
        fp64_check(c, check.reference)
    torch.cuda.synchronize()
    timings = {scope: median(samples) for (scope, samples) in values.items()}
    result = {'type': 'result', 'rows': rows, 'phase': args.phase, 'mode': 'cuda_graph' if graphs else 'eager', 'baseline_backend': cases[0].baseline_backend, 'median_us': timings, 'speedup_total': timings['baseline'] / timings['total'] if 'baseline' in timings else None, 'changed_input_fp64_passed': True, 'effective_tflops': {s: (2 if s in ('down', 'up') else 4) * rows * 10240 * 320 / (v * 1000000.0) for (s, v) in timings.items()}}
    emit(result)
    comparison = (f" Frozen({cases[0].baseline_backend})={timings['baseline']:.3f} us speedup={result['speedup_total']:.3f}x"
                  if 'baseline' in timings else " Baseline: not run")
    print(f"T={rows:5d} {result['mode']:10s} Down={timings['down']:.3f} Up={timings['up']:.3f} Total={timings['total']:.3f}" + comparison, flush=True)
    return result


def benchmark_prefill_batch(rows, args, emit):
    prefill = testing()
    torch, _, _, cuda_perf = prefill.dependencies()
    compiled_mix = None if args.no_baselines else torch_compile_mix()
    scopes = TIMING_SCOPES[:-1] if args.no_baselines else TIMING_SCOPES
    from pyhip.ops.gr_read import gr_read
    with torch.no_grad():
        x, wd, wu = prefill.make_inputs(torch, rows, args.seed)
        inputs = [x] + [x.clone() for _ in range(args.buffers - 1)]
        # Both implementations rotate over independent weight allocations.
        # Each pair has identical logical values; packing happens only here.
        weights = [(wd, wu)] + [(wd.clone(), wu.clone()) for _ in range(args.buffers - 1)]
        baseline_weights = ([baseline_modules().prepare_weights(*pair) for pair in weights]
                            if compiled_mix is not None else [])
        readers = [prefill.prepare_reader(input_x, *pair) for input_x, pair in zip(inputs, weights)]
        expected_p = expected_y = None
        expected_torch = None
        torch_outputs = [None] * args.buffers
        addresses = []
        calls = {scope: [] for scope in scopes}
        for bi, (input_x, reader, (raw_down, raw_up)) in enumerate(zip(inputs, readers, weights)):
            dm, dw, dn, dk = reader.down_config
            um, un = reader.up_config
            partial, output = reader.partial, reader.output
            addresses.append({name: tensor_address(tensor, output=name == "Y") for name, tensor in zip(
                ("X", "W_down", "W_up", "P_stages", "Y"),
                (input_x, reader.w_down, reader.w_up, partial, output))})
            calls["down"].append(lambda x=input_x, r=reader: r.run_down(x))
            calls["up"].append(lambda x=input_x, r=reader: r.run_up(x))
            calls["total"].append(lambda x=input_x, r=reader: gr_read(x, r.w_down, r.w_up, output=r.output))
            if compiled_mix is not None:
                baseline_down, baseline_up = baseline_weights[bi]
                calls["torch_compile"].append(
                    lambda x=input_x, wd=baseline_down, wu=baseline_up: compiled_mix(x, wd, wu, C, H))
            partial.fill_(torch.nan); output.fill_(torch.nan)
            reader(input_x)
            if expected_p is None:
                expected_p, expected_y = partial.clone(), output.clone()
            torch.testing.assert_close(partial, expected_p, rtol=0, atol=0)
            torch.testing.assert_close(output, expected_y, rtol=0, atol=0)
            if compiled_mix is not None:
                # Compile/check every raw-layout baseline buffer before timing.
                torch_outputs[bi] = calls["torch_compile"][bi]()
                if expected_torch is None:
                    expected_torch = torch_outputs[bi].clone()
                torch.testing.assert_close(torch_outputs[bi], expected_torch, rtol=0, atol=0)
                prefill.check_close(output, torch_outputs[bi], prefill.OUTPUT_TOLERANCE)
                addresses[-1].update({name: tensor_address(tensor) for name, tensor in (
                    ("Torch_X", input_x), ("Torch_W_down", baseline_down), ("Torch_W_up", baseline_up),
                    ("Torch_Y_prepared", torch_outputs[bi]))})
        assert all(len({row[name]["pointer"] for row in addresses}) == args.buffers for name in addresses[0])
        emit({"type": "addresses", "rows": rows, "buffers": addresses,
              "total_entry": "gr_read", "total_partial": "actual allocations recorded as P_total on each Total sample; P_stages belongs to stage measurements"})
        for scope in scopes:
            for index in range(args.warmup): calls[scope][index % args.buffers]()
        torch.cuda.synchronize()
        timings = {}
        # Retain the original stage-by-stage sampling order; append Torch.
        # These are whole-scope medians, not an interleaved A/B experiment.
        for scope in scopes:
            perf = cuda_perf(name=f"gr_read_{scope}", verbose=0)
            if not perf.enable:
                raise RuntimeError("CUDAPERF disables GRRead timing")
            for index in range(args.iters):
                bi = index % args.buffers
                partials = None
                if scope == 'total' and args.record_partials:
                    # Install/remove the observer outside the original timer.
                    # Only raw address integers are collected inside the call.
                    with observe_partial_allocations((rows, R), torch.bfloat16, inputs[bi].device) as partials:
                        with perf: actual = calls[scope][bi]()
                else:
                    with perf: actual = calls[scope][bi]()
                us = perf.latencies[-1] * 1e6
                assert math.isfinite(us) and us > 0
                record = {"scope": scope, "sample": index, "buffer": bi, "us": us}
                if partials is not None:
                    reader = readers[bi]
                    record.update(partial_address_record(partials, (rows, R), torch.bfloat16,
                                                         inputs[bi], reader.w_down, reader.w_up, actual))
                if scope == "torch_compile":
                    torch_outputs[bi] = actual
                    record["output_address"] = tensor_address(actual)
                emit({"type": "sample", "rows": rows, **record})
                # Validate each timed Torch output before a later call can
                # release/reuse it. No validation is inside the timer.
                if scope == "torch_compile":
                    torch.testing.assert_close(actual, expected_torch, rtol=0, atol=0)
            samples = [value * 1e6 for value in perf.latencies]
            assert len(samples) == args.iters
            for bi in sorted({index % args.buffers for index in range(args.iters)}):
                reader = readers[bi]
                actual = (torch_outputs[bi] if scope == "torch_compile" else
                          reader.partial if scope == "down" else reader.output)
                expected = (expected_torch if scope == "torch_compile" else
                            expected_p if scope == "down" else expected_y)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            flops = (2 if scope in ("down", "up") else 4) * rows * 10240 * 320
            elapsed = median(samples)
            timings[scope] = {"elapsed_us": elapsed, "samples_us": samples, "gemm_FLOPs": flops,
                              "effective_TFLOPS": flops / elapsed / 1e6}
            print(f"  T={rows:5d} Down M{dm}/W{dw}/N{dn}/BK{dk} / Up M{um}/N{un} "
                  f"{scope}: {elapsed:.3f} us", flush=True)
        return {"complete": True, "rows": rows, "down_n_splits": dn,
                "down_block_m": dm, "down_num_waves": dw, "down_block_k": dk,
                "up_block_m": um, "n_splits": un, "timings": timings,
                "buffers": args.buffers, "warmup_each": args.warmup, "samples_each": args.iters,
                "timed_outputs_bitexact": True, "all_samples_retained": True,
                "Total_measured_directly": True, "settings_written": False,
                "speedup_vs_torch_compile": (timings["torch_compile"]["elapsed_us"] / timings["total"]["elapsed_us"]
                                             if compiled_mix is not None else None),
                "torch_compile": {"full_rows": rows, "returns": "Y", "options": "default", "cuda_graph": False,
                                  "shared_X": True, "independent_raw_weight_buffers": args.buffers,
                                  "output_allocation": "native", "Y_tolerance": prefill.OUTPUT_TOLERANCE}
                                  if compiled_mix is not None else None,
                "sampling_order": list(scopes)}


def print_prefill_table(results):
    print("\n| Batch | Down | Up | Down us / TFLOPS | Up us / TFLOPS | Total us / TFLOPS | Frozen Torch compile us / TFLOPS | Speedup |", flush=True)
    print("|---:|---:|---:|---:|---:|---:|---:|---:|", flush=True)
    for result in results:
        cells = [(f"{result['timings'][scope]['elapsed_us']:.3f} / {result['timings'][scope]['effective_TFLOPS']:.3f}"
                  if scope in result['timings'] else '—') for scope in TIMING_SCOPES]
        speedup = f"{result['speedup_vs_torch_compile']:.3f}x" if result['speedup_vs_torch_compile'] is not None else '—'
        down = (f"M{result['down_block_m']}/W{result['down_num_waves']}"
                f"/N{result['down_n_splits']}/BK{result['down_block_k']}")
        print(f"| {result['rows']} | {down} | M{result['up_block_m']}/N{result['n_splits']} | "
              f"{' | '.join(cells)} | {speedup} |", flush=True)
    print("Speedup = Torch compile Total / PyHIP Total; eager cudaPerf, full-row calls, no CUDA Graph.", flush=True)


def hardware_snapshot():
    physical = os.getenv("HIP_VISIBLE_DEVICES") or os.getenv("CUDA_VISIBLE_DEVICES") or "0"
    if not physical.isdigit() or os.getenv("ROCR_VISIBLE_DEVICES"):
        raise RuntimeError("select one physical GPU with HIP_VISIBLE_DEVICES")
    if os.getenv("CUDA_VISIBLE_DEVICES", physical) != physical:
        raise RuntimeError("HIP_VISIBLE_DEVICES and CUDA_VISIBLE_DEVICES must match")
    if os.getenv("HSA_CU_MASK") or os.getenv("ROC_GLOBAL_CU_MASK"):
        raise RuntimeError("benchmark requires an unmasked GPU")
    env = {k: v for k, v in os.environ.items() if k not in (
        "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL")}
    def query(command):
        return json.loads(subprocess.run(command, env=env, text=True, capture_output=True,
                                         check=True, timeout=30).stdout)
    card = query(["rocm-smi", "-d", physical, "--showuse", "--showmemuse", "--showbus", "--json"])[f"card{physical}"]
    limit = query(["amd-smi", "static", "-g", physical, "--limit", "--json"])["gpu_data"][0]["limit"]
    return {"physical_gpu": int(physical), "card": card, "limit": limit}


def check_hardware(snapshot, *, entry=False):
    card, limit = snapshot["card"], snapshot["limit"]
    use, memory = int(card["GPU use (%)"]), int(card["GPU Memory Allocated (VRAM%)"])
    if (entry and (use != 0 or memory != 0)) or use > 5 or memory > 20:
        raise RuntimeError(f"GPU occupied: {snapshot}; benchmark stopped")
    if limit.get("ptl_state") != "Enabled" or limit.get("ptl_format") != "VECTOR,F8":
        warnings.warn(f"Expected PTL Enabled/VECTOR,F8; continuing with "
                      f"{limit.get('ptl_state')}/{limit.get('ptl_format')}",
                      RuntimeWarning, stacklevel=2)


def time_graph(graph, calls, samples):
    # Same CUDA Event timer as be1555f support.time_graph; no Python launch timing.
    import torch
    for _ in range(3):
        graph.replay()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    values = []
    for _ in range(samples):
        start.record()
        for _ in range(3):
            graph.replay()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end) * 1000 / (calls * 3))
    return values


def benchmark_decode_total(rows, pairs, args, emit):
    import torch
    from pyhip.ops.gr_read.flydsl import GRReadDecode, prepare_weights
    gen = torch.Generator(device="cuda").manual_seed(args.seed + rows)
    xs = [torch.randn(rows, 10240, device="cuda", dtype=torch.bfloat16, generator=gen) for _ in pairs]
    readers = [GRReadDecode(rows, *prepare_weights(wd, wu)) for wd, wu in pairs]
    emit({"type": "addresses", "rows": rows, "buffers": [
        {name: {"pointer": t.data_ptr(), "storage_offset": t.storage_offset(), "mod4096": t.data_ptr() % 4096}
         for name, t in zip(("X", "WD", "WU", "P", "Y"), (x, r.w_down, r.w_up, r.partial, r.output))}
        for x, r in zip(xs, readers)]})
    calls = [lambda x=x, r=r: r(x) for x, r in zip(xs, readers)]
    graph = capture(calls + calls)
    graph.replay()
    for x, r, (wd, wu) in zip(xs, readers, pairs):
        assert_close(r.output, reference(x, wd, wu), f"T={rows}: initial FP64")
    torch.cuda.synchronize()
    # The caller checks hardware once before preparing the benchmark inputs.
    timings = []
    for round_index in range(args.rounds):
        values = time_graph(graph, len(pairs) * 2, args.samples)
        timings.extend(values)
        emit({"type": "samples", "rows": rows, "round": round_index, "us": values})
    for x in xs:
        x.mul_(0.99).add_(0.015625)
    graph.replay()
    for x, r, (wd, wu) in zip(xs, readers, pairs):
        assert_close(r.output, reference(x, wd, wu), f"T={rows}: changed-input FP64")
    latency = median(timings)
    result = {"type": "result", "rows": rows, "median_us": latency, "fp64_passed": True,
              "effective_tflops": 4 * rows * 10240 * 320 / (latency * 1e6)}
    emit(result)
    print(f"T={rows:2}  H64 final Graph: {latency:8.3f} us  {result['effective_tflops']:.3f} effective TFLOPS", flush=True)
    return result


def parse_args(argv=None):
    checks = testing()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('all', 'decode', 'prefill'), default='all')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--no-baselines', action='store_true', help='skip frozen performance baselines in both phases; accuracy references still run')
    parser.add_argument('--rows', '--batches', nargs='+', type=checks.parse_batch)
    parser.add_argument('--decode-rows', nargs='+', type=checks.parse_batch)
    parser.add_argument('--prefill-rows', nargs='+', type=checks.parse_batch)
    parser.add_argument('--decode-weights', type=int, default=100)
    parser.add_argument('--decode-rounds', type=int, default=3)
    parser.add_argument('--decode-samples', type=int, default=7)
    parser.add_argument('--decode-seed', type=int, default=707)
    parser.add_argument('--prefill-buffers', type=int, default=10)
    parser.add_argument('--prefill-warmup', type=int, default=2)
    parser.add_argument('--prefill-iters', type=int, default=10)
    parser.add_argument('--prefill-seed', type=int, default=131)
    parser.add_argument('--settle-seconds', type=float, default=2.0, help='wait once before the startup hardware check')
    parser.add_argument('--amd-smi', type=Path)
    parser.add_argument('--output', type=Path, help='optional new JSONL file; default: progress and tables on console')
    parser.add_argument('--md', type=Path, help='optional new Markdown performance report')
    parser.add_argument('--verbose', action='store_true', help='also print hardware and detailed correctness diagnostics')
    for name in ('weights', 'rounds', 'samples', 'buffers', 'warmup', 'iters', 'seed'):
        parser.add_argument('--' + name, type=int, help='single-phase compatibility option')
    args = parser.parse_args(argv)
    checks.selected_rows(parser, args)
    for name in ('weights', 'rounds', 'samples', 'buffers', 'warmup', 'iters', 'seed'):
        value = getattr(args, name)
        if value is None: continue
        if args.phase == 'all': parser.error(f'use phase-specific options instead of --{name} with --phase all')
        mapping = ({'weights': 'weights', 'rounds': 'rounds', 'samples': 'samples', 'seed': 'seed'}
                   if args.phase == 'decode' else
                   {'weights': 'buffers', 'buffers': 'buffers', 'warmup': 'warmup', 'iters': 'iters', 'samples': 'iters', 'seed': 'seed'})
        if name not in mapping: parser.error(f'--{name} is not an option for {args.phase}')
        setattr(args, args.phase + '_' + mapping[name], value)
    counts = (args.decode_weights, args.decode_rounds, args.decode_samples, args.prefill_buffers, args.prefill_iters)
    if min(counts) < 1 or args.gpu < 0 or args.prefill_warmup < 0 or args.settle_seconds < 0 or not __debug__:
        parser.error('positive counts, nonnegative GPU/warmup/settling required; do not use python -O')
    checks.validate_report_paths(parser, args.output, args.md)
    return args


def emit_record(stream, phase, record):
    if stream is None:
        return
    record = dict(record)
    if 'phase' in record and record['phase'] != phase:
        record['event_phase'] = record.pop('phase')
    record['phase'] = phase
    stream.write(json.dumps(record, allow_nan=False) + '\n')
    stream.flush()


def run_decode(args, emit):
    local = SimpleNamespace(phase='decode', gpu=args.gpu, seed=args.decode_seed,
                            weights=args.decode_weights, rounds=args.decode_rounds,
                            samples=args.decode_samples, warmup=2,
                            amd_smi=args.amd_smi, verbose=args.verbose,
                            record_partials=args.output is not None)
    runtime = 'PyHIP runtime' if args.no_baselines else 'PyHIP and frozen baselines'
    print(f"\nDecode: {len(args.decode_rows)} batches, {local.weights} weight pairs, "
          f"{local.rounds}x{local.samples} Graph samples/scope. Loading {runtime}...", flush=True)
    if args.no_baselines:
        print("Frozen performance baselines disabled (--no-baselines).", flush=True)
    dep = dependencies(args.no_baselines)
    torch, _, _, _, check, prefill, _, _ = dep
    if torch.version.hip is None or not torch.cuda.is_available():
        raise RuntimeError('decode comparison requires a ROCm GPU')
    prefill.warn_architecture(torch, 0)
    props = torch.cuda.get_device_properties(0)
    emit({'type': 'environment', 'torch': torch.__version__, 'hip': torch.version.hip,
          'gpu': props.name, 'arch': props.gcnArchName, 'compute_units': props.multi_processor_count,
          'protocol': {'weights': local.weights, 'rounds': local.rounds, 'samples': local.samples,
                       'seed': local.seed, 'graph_passes': 2, 'replays_per_sample': 3, 'mode': 'cuda_graph',
                       'record_total_partials': local.record_partials},
          'sources': source_hashes(args.no_baselines),
          'baseline': None if args.no_baselines else baseline_modules().metadata(),
          'settings_written': False})
    results = []
    with torch.inference_mode():
        print(f"Decode: preparing {local.weights} shared weight pairs...", flush=True)
        pairs = [check.make_inputs(1, local.seed + i)[1:] for i in range(local.weights)]
        from pyhip.ops.gr_read.flydsl import prepare_weights
        packed_pairs = [prepare_weights(wd, wu) for wd, wu in pairs]
        emit({'type': 'weight_preparation', 'pairs': len(pairs), 'packing_calls': len(pairs),
              'shared_across_rows': True, 'baseline_layout': 'independent contiguous raw matrices' if not args.no_baselines else None})
        for index, rows in enumerate(args.decode_rows, 1):
            print(f"[Decode {index}/{len(args.decode_rows)}] T={rows}: "
                  "preparing kernels, checking outputs and capturing graphs...", flush=True)
            results.append(benchmark_decode_rows(rows, pairs, local, dep, emit, packed_pairs))
            gc.collect()
            torch.cuda.empty_cache()
    emit({'type': 'summary', 'complete': True, 'rows': args.decode_rows})
    return results


def run_prefill(args, emit, *, check=True):
    prefill = testing()
    local = SimpleNamespace(gpu=args.gpu, seed=args.prefill_seed, buffers=args.prefill_buffers,
                            warmup=args.prefill_warmup, iters=args.prefill_iters,
                            amd_smi=args.amd_smi, verbose=args.verbose, no_baselines=args.no_baselines,
                            record_partials=args.output is not None)
    print(f"\nPrefill: {len(args.prefill_rows)} batches, {local.buffers} buffers, "
          f"{local.warmup} warmup calls, {local.iters} eager samples/scope. Loading runtime...", flush=True)
    torch, _, _, _ = prefill.dependencies()
    props = torch.cuda.get_device_properties(0)
    emit({'type': 'environment', 'torch': torch.__version__, 'hip': torch.version.hip,
          'gpu': props.name, 'arch': props.gcnArchName, 'compute_units': props.multi_processor_count,
          'protocol': {'buffers': local.buffers, 'warmup': local.warmup, 'iters': local.iters,
                       'seed': local.seed, 'scope_order': list(TIMING_SCOPES[:-1] if args.no_baselines else TIMING_SCOPES), 'mode': 'eager',
                       'record_total_partials': local.record_partials},
          'sources': source_hashes(args.no_baselines),
          'baseline': None if args.no_baselines else baseline_modules().metadata(),
          'settings_written': False})
    # Keep the original prefill flow: all selected correctness checks before timing.
    for index, rows in enumerate(args.prefill_rows if check else (), 1):
        try:
            print(f"[Prefill check {index}/{len(args.prefill_rows)}] T={rows}: checking correctness (JIT may compile)...", flush=True)
            result = prefill.check_batch(rows, local)
            emit({'type': 'correctness', **result})
            if not args.verbose:
                print(f"  PASS: P rel_l2={result['P_rel_l2']:.6g}, Y rel_l2={result['Y_rel_l2']:.6g}", flush=True)
        finally:
            prefill.release_buffers()
    results = []
    print("Prefill: correctness passed; starting eager timing.", flush=True)
    for index, rows in enumerate(args.prefill_rows, 1):
        try:
            print(f"[Prefill {index}/{len(args.prefill_rows)}] T={rows}: preparing buffers and timing Down/Up/Total/Torch compile...", flush=True)
            result = benchmark_prefill_batch(rows, local, emit)
            results.append(result)
            emit({'type': 'result', **result})
        finally:
            prefill.release_buffers()
    emit({'type': 'summary', 'complete': True, 'rows': args.prefill_rows})
    return results


def print_decode_table(results):
    print('\nDecode (CUDA Graph)')
    print('\n| T | Decode Down us | Decode Up us | Decode Total us | Frozen baseline backend | Frozen baseline Total us | Speedup |')
    print('|---:|---:|---:|---:|---|---:|---:|')
    for r in results:
        t = r['median_us']
        baseline_time = f"{t['baseline']:.3f}" if 'baseline' in t else '—'
        speedup = f"{r['speedup_total']:.3f}x" if r['speedup_total'] is not None else '—'
        print(f"| {r['rows']} | {t['down']:.3f} | {t['up']:.3f} | {t['total']:.3f} | "
              f"{r['baseline_backend'] or 'not run'} | {baseline_time} | {speedup} |")


def write_markdown_report(path, args, results, environments, hardware, started_at):
    """Render completed results with the same tables printed by the benchmark."""
    stream = io.StringIO()
    with redirect_stdout(stream):
        print('# GR read performance report\n')
        print(f'Run started: {started_at}. Dimensions: BF16 X[T,10240], C=4, H=2560, R=320.\n')
        print('Total calls `gr_read(..., output=out)` with an internally allocated P. '
              'Packing, compilation and correctness checks are outside timing.\n')
        if hardware is not None:
            card, limit = hardware['card'], hardware['limit']
            print(f"Physical GPU {hardware['gpu']}, PCI {card.get('PCI Bus', 'unknown')}; "
                  f"startup use={card['GPU use (%)']}%, VRAM={card['GPU Memory Allocated (VRAM%)']}%; "
                  f"PTL={limit.get('ptl_state')}/{limit.get('ptl_format')}.\n")
        print('Hardware checked once at startup; device settings were not changed.\n')
        try:
            print(f"FlyDSL: {package_metadata.version('flydsl')}.\n")
        except package_metadata.PackageNotFoundError:
            print('FlyDSL package version unavailable.\n')
        for phase, rows in results.items():
            env = environments[phase]
            print(f'## {phase.capitalize()}\n')
            print(f"GPU: {env['gpu']}; {env['arch']}; {env['compute_units']} CUs. "
                  f"Torch: {env['torch']}; HIP: {env['hip']}.\n")
            print('Sampling protocol:\n```json\n' + json.dumps(env['protocol'], indent=2) + '\n```\n')
            if env['baseline'] is None:
                print('Frozen performance baselines disabled.\n')
            else:
                baseline = env['baseline']
                print(f"Frozen SGLang baseline checkout: `{baseline['source_checkout_commit']}`; "
                      f"Triton tuning commit: `{baseline['triton_last_change_commit']}`.\n")
                print('Baseline compiler versions: `' + json.dumps(baseline['runtime_versions']) + '`.\n')
            print(f'{len(rows)} selected shapes completed with output checks.\n')
            if phase == 'decode':
                print_decode_table(rows)
            else:
                print_prefill_table(rows)
            print('\nSource SHA256:\n```json\n' + json.dumps(
                {str(Path(p).relative_to(REPO)): digest for p, digest in env['sources'].items()},
                indent=2) + '\n```\n')
        if args.output is not None:
            raw = os.path.relpath(args.output.resolve(), path.resolve().parent)
            print(f'[Raw samples and actual P addresses](<{raw}>)\n')
        else:
            print('Raw samples and internal P addresses were not exported; use `--output` to retain them.\n')
        print('Decode uses CUDA Graph replay; prefill uses eager cudaPerf. '
              'The two timing modes are reported separately. Effective TFLOPS count GEMM work only.')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as out:
        out.write(stream.getvalue())


def run_benchmarks(args, emit, *, prefill_checked=False, hardware_checked=False, startup_hardware=None):
    """Run the existing samplers and print both final tables after all phases."""
    started_at = datetime.now(timezone.utc).isoformat(timespec='seconds')
    if not hardware_checked:
        startup_hardware = benchmark_entry_gate(args, emit)
    checks = testing()
    environments = {}

    def phase_emit(phase, record):
        if args.md is not None and record['type'] == 'environment':
            environments[phase] = record
        emit(phase, record)

    results_by_phase = {}
    if args.phase in ('all', 'decode'):
        results_by_phase['decode'] = run_decode(args, lambda r: phase_emit('decode', r))
        checks.release_buffers()
    if args.phase in ('all', 'prefill'):
        results_by_phase['prefill'] = run_prefill(args, lambda r: phase_emit('prefill', r), check=not prefill_checked)
    print('\nBenchmark complete. Final results:', flush=True)
    if 'decode' in results_by_phase:
        print_decode_table(results_by_phase['decode'])
    if 'prefill' in results_by_phase:
        print('\nPrefill (eager cudaPerf)')
        print_prefill_table(results_by_phase['prefill'])
    sys.stdout.flush()
    if args.md is not None:
        write_markdown_report(args.md, args, results_by_phase, environments, startup_hardware, started_at)
        print(f'Markdown report: {args.md}', flush=True)
    return results_by_phase


def main(argv=None):
    args = parse_args(argv)
    testing().prepare_cli_environment(args)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
    with (args.output.open('x') if args.output else nullcontext()) as stream:
        run_benchmarks(args, lambda phase, record: emit_record(stream, phase, record))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
