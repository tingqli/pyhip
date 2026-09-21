# SPDX-License-Identifier: MIT
"""Compare PyHIP GR read with an explicit, current SGLang checkout.

Decode: CUDA Graph Down / Up / Total and SGLang Total, T=1..32.
Prefill: ordinary cudaPerf Down / Up / Total and SGLang Total.
No model is loaded. Raw BF16 weights and X values are shared by both paths.
"""

import argparse
import ast
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from statistics import median
from types import SimpleNamespace
import subprocess
import sys
import time
import warnings

REPO = Path(__file__).resolve().parents[3]
PREFILL_ROWS = (32, 64, 128, 256, 512) + tuple(
    k * 1024 for k in (1, 2, 4, 8, 10, 12, 16, 20, 24, 28, 30, 32, 36, 48, 60, 64)
)
SCOPES = ('down', 'up', 'total', 'sglang')


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


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


def dependencies(sglang_root, *, with_decode=True):
    # Explicit package path avoids accidentally benchmarking another editable install.
    if 'pyhip' not in sys.modules:
        load_file('pyhip', REPO / 'src/__init__.py')
    import pyhip
    if Path(pyhip.__file__).resolve() != REPO / 'src/__init__.py':
        raise RuntimeError(f'wrong PyHIP checkout: {pyhip.__file__}')
    import torch
    import flydsl.compiler as flyc
    from pyhip.misc import cudaPerf

    decode = None
    if with_decode:
        path = REPO / 'tests/contrib/gr_read_decode/kernel.py'
        if not path.is_file():
            raise RuntimeError('decode example is not installed; use --phase prefill')
        decode = load_file('gr_compare_decode', path)
    check = SimpleNamespace(reference=reference, assert_close=assert_close, make_inputs=make_inputs, capture=capture)
    prefill = load_file('gr_compare_prefill', Path(__file__).with_name('test_gr_read.py'))
    source = sglang_root / 'python/sglang/srt/layers/hyperconnection.py'
    tree = ast.parse(source.read_text())
    functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == '_mix_compute']
    if len(functions) != 1:
        raise RuntimeError('expected one SGLang _mix_compute function')
    namespace = {'torch': torch, 'F': torch.nn.functional}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), 'exec'), namespace)
    # Same default torch.compile call as GatedResidual.__init__, using its exact AST.
    compiled_mix = torch.compile(namespace['_mix_compute'])
    sys.path.insert(0, str(sglang_root / 'python'))
    triton_source = sglang_root / 'python/sglang/srt/layers/hc_mix_triton.py'
    triton_mix = load_file('gr_compare_sglang_triton', triton_source)
    return torch, flyc, cudaPerf, decode, check, prefill, compiled_mix, triton_mix


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
            'mod256': tensor.data_ptr() % 256, 'mod4096': tensor.data_ptr() % 4096,
            'shape': list(tensor.shape), 'dtype': str(tensor.dtype)}


def gate(prefill, args, phase, emit, rows=None):
    # Fixed before any run: allow this process's own work to leave the SMI window.
    # One query per gate; failed gates are retained and never retried in this run.
    time.sleep(args.settle_seconds)
    snapshot = prefill.read_hardware(args.gpu)
    emit({'type': 'hardware', 'phase': phase, 'rows': rows, **snapshot})
    prefill.validate_hardware(snapshot)


class Case:
    def __init__(self, rows, x, wd, wu, args, dep, packed=None):
        torch, flyc, _, decode, _, prefill, compiled_mix, triton_mix = dep
        self.torch, self.x, self.wd, self.wu = torch, x, wd, wu
        self.phase, self.rows, self.prefill = args.phase, rows, prefill
        if args.phase == 'decode':
            pd, pu = packed if packed is not None else decode.prepare_weights(wd, wu)
            reader = decode.GRReadDecode(rows, pd, pu)
            self.reader = reader
            self.pd, self.pu, self.p, self.y = pd, pu, reader.partial, reader.output
            self.down = flyc.compile(decode.down_launcher(rows), x.view(-1), pd, self.p,
                                     torch.cuda.current_stream(x.device))
            self.up = flyc.compile(decode.up_launcher(rows), x.view(-1), pu, self.p,
                                   self.y.view(-1), torch.cuda.current_stream(x.device))
        else:
            from pyhip.contrib.flydsl.gr_read import GRReadPrefill, prepare_weights
            self.pd, self.pu = packed if packed is not None else prepare_weights(wd, wu)
            self.reader = GRReadPrefill(rows, self.pd, self.pu)
            self.p, self.y = self.reader.partial, self.reader.output
            self.down, self.up = self.reader.down, self.reader.up
        # On gfx942 the preceding CuTe/SM100 branch in GatedResidual.mix is unavailable.
        self.sg_backend = ('triton' if triton_mix.fused_hc_mix_supported(x, wd, wu)
                           else 'torch.compile')
        self.sg_call = triton_mix.fused_hc_mix if self.sg_backend == 'triton' else compiled_mix
        self.sg_y = None

    def run(self, scope):
        torch = self.torch
        if scope == 'sglang':
            self.sg_y = self.sg_call(self.x, self.wd, self.wu, 4, 2560)
            return self.sg_y
        if self.phase == 'decode':
            stream = torch.cuda.current_stream(self.x.device)
            if scope == 'down':
                self.down(self.x.view(-1), self.pd, self.p, stream)
            elif scope == 'up':
                self.up(self.x.view(-1), self.pu, self.p, self.y.view(-1), stream)
            else:
                self.reader(self.x)
        elif scope == 'down':
            self.reader.run_down(self.x)
        elif scope == 'up':
            self.reader.run_up(self.x)
        else:
            self.reader(self.x)
        return self.p if scope == 'down' else self.y


def output_check(torch, actual, expected, label):
    torch.testing.assert_close(actual.double(), expected.double(), rtol=1e-2, atol=5e-3,
                               msg=label)


def fp64_check(case, reference):
    # Chunking keeps the large-prefill reference memory bounded.
    for begin in range(0, case.rows, 1024):
        end = min(begin + 1024, case.rows)
        expected = reference(case.x[begin:end], case.wd, case.wu)
        output_check(case.torch, case.y[begin:end], expected, 'PyHIP vs FP64')
        output_check(case.torch, case.sg_y[begin:end], expected, 'SGLang vs FP64')


def decode_stage_check(case):
    torch = case.torch
    rows = case.rows
    x, wd, wu = case.x.double(), case.wd.double(), case.wu.double()
    partial = case.p.view(4, case.reader.padded_rows, 320)
    case.run('down')
    maximum = 0.0
    for split in range(4):
        expected = x[:, split * 2560:(split + 1) * 2560] @ wd[:, split * 2560:(split + 1) * 2560].T
        actual = partial[split, :rows].double()
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=1e-5)
        maximum = max(maximum, (actual - expected).abs().max().item())
    assert torch.count_nonzero(partial[:, rows:]) == 0
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
            output_check(c.torch, c.sg_y, c.saved_sg, 'timed SGLang output')


def benchmark_rows(rows, pairs, args, dep, emit, packed_pairs=None):
    torch, _, cudaPerf, _, check, prefill, _, _ = dep
    gen = torch.Generator(device='cuda').manual_seed(args.seed + rows)
    cases = [Case(rows, torch.randn(rows, 10240, device='cuda', dtype=torch.bfloat16,
                                   generator=gen), wd, wu, args, dep,
                  packed_pairs[i] if packed_pairs is not None else None) for i, (wd, wu) in enumerate(pairs)]
    stage_errors = []
    for index, c in enumerate(cases):
        c.run('total')
        c.run('sglang')
        fp64_check(c, check.reference)
        if args.phase == 'decode' and index < 2:
            stage_errors.append(decode_stage_check(c))
        c.saved_p, c.saved_y, c.saved_sg = c.p.clone(), c.y.clone(), c.sg_y.clone()
    graphs = {}
    if args.phase == 'decode':
        for scope in SCOPES:
            calls = [lambda c=c, scope=scope: c.run(scope) for c in cases]
            graphs[scope] = check.capture(calls + calls)
            graphs[scope].replay()
            check_timed(cases, scope)
    else:
        for scope in SCOPES:
            for i in range(args.warmup):
                cases[i % len(cases)].run(scope)
    emit({'type': 'correctness', 'phase': 'initial', 'rows': rows, 'pairs': len(pairs),
          'fp64_passed': True, 'isolated_down_max_abs': stage_errors,
          'isolated_up_fp64_passed': bool(stage_errors)})
    emit({'type': 'addresses', 'rows': rows, 'buffers': [
        {k: address(v) for k, v in (('X', c.x), ('WD_raw', c.wd), ('WU_raw', c.wu),
                                    ('WD_packed', c.pd), ('WU_packed', c.pu),
                                    ('P', c.p), ('Y', c.y), ('SGLang_Y', c.sg_y))}
        for c in cases]})
    torch.cuda.synchronize()
    gate(prefill, args, 'before_samples', emit, rows)
    values = {scope: [] for scope in SCOPES}
    if args.phase == 'decode':
        for round_index in range(args.rounds):
            order = SCOPES if round_index % 2 == 0 else SCOPES[::-1]
            for scope in order:
                samples = graph_samples(torch, graphs[scope], len(cases) * 2, args.samples)
                values[scope].extend(samples)
                emit({'type': 'samples', 'rows': rows, 'round': round_index,
                      'scope': scope, 'us': samples, 'mode': 'cuda_graph'})
                check_timed(cases, scope)
    else:
        timers = {s: cudaPerf(name='gr_compare_' + s, verbose=0) for s in SCOPES}
        if not all(timer.enable for timer in timers.values()):
            raise RuntimeError('cudaPerf is disabled')
        for index in range(args.samples):
            order = SCOPES if index % 2 == 0 else SCOPES[::-1]
            for scope in order:
                with timers[scope]:
                    cases[index % len(cases)].run(scope)
                value = timers[scope].latencies[-1] * 1e6
                values[scope].append(value)
                emit({'type': 'samples', 'rows': rows, 'sample': index,
                      'buffer': index % len(cases), 'scope': scope, 'us': [value], 'mode': 'eager'})
        for scope in SCOPES:
            check_timed(cases, scope)
    # Changed inputs must be observed by the actual captured graphs / eager calls.
    for c in cases:
        c.x.mul_(0.99).add_(0.015625)
    if graphs:
        graphs['total'].replay()
        graphs['sglang'].replay()
    else:
        for c in cases:
            c.run('total')
            c.run('sglang')
    for c in cases:
        fp64_check(c, check.reference)
    torch.cuda.synchronize()
    gate(prefill, args, 'after_samples', emit, rows)
    timings = {scope: median(samples) for scope, samples in values.items()}
    result = {'type': 'result', 'rows': rows, 'phase': args.phase,
              'mode': 'cuda_graph' if graphs else 'eager',
              'sglang_backend': cases[0].sg_backend, 'median_us': timings,
              'speedup_total': timings['sglang'] / timings['total'],
              'changed_input_fp64_passed': True,
              'effective_tflops': {s: (2 if s in ('down', 'up') else 4) * rows * 10240 * 320 / (v * 1e6)
                                   for s, v in timings.items()}}
    emit(result)
    print(f"T={rows:5d} {result['mode']:10s} Down={timings['down']:.3f} Up={timings['up']:.3f} "
          f"Total={timings['total']:.3f} SGLang({cases[0].sg_backend})={timings['sglang']:.3f} us "
          f"speedup={result['speedup_total']:.3f}x", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('decode', 'prefill'), required=True)
    parser.add_argument('--sglang-root', type=Path, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--rows', '--batches', nargs='+', type=int)
    parser.add_argument('--weights', type=int, help='default: decode 100, prefill 10')
    parser.add_argument('--rounds', type=int, default=3, help='decode Graph sample rounds')
    parser.add_argument('--samples', type=int, help='default: decode 7 per round, prefill 10')
    parser.add_argument('--warmup', type=int, default=2, help='prefill warmup calls per scope')
    parser.add_argument('--settle-seconds', type=float, default=2.0)
    parser.add_argument('--seed', type=int, default=707)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.rows = args.rows or (list(range(1, 33)) if args.phase == 'decode' else list(PREFILL_ROWS))
    args.weights = args.weights if args.weights is not None else (100 if args.phase == 'decode' else 10)
    args.samples = args.samples if args.samples is not None else (7 if args.phase == 'decode' else 10)
    if (not __debug__ or min(args.rows + [args.weights, args.rounds, args.samples]) < 1
            or args.gpu < 0 or args.warmup < 0 or args.settle_seconds < 0):
        parser.error('positive counts/rows and nonnegative GPU/warmup/settling required; no python -O')
    if args.phase == 'decode' and max(args.rows) > 32:
        parser.error('decode supports T=1..32')
    if len(set(args.rows)) != len(args.rows):
        parser.error('rows must be distinct')
    if os.getenv('HSA_CU_MASK') or os.getenv('ROC_GLOBAL_CU_MASK'):
        parser.error('benchmark requires an unmasked GPU')
    for key in ('ROCR_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES', 'GPU_DEVICE_ORDINAL'):
        os.environ.pop(key, None)
    os.environ['HIP_VISIBLE_DEVICES'] = str(args.gpu)
    args.sglang_root = args.sglang_root.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as out:
        def emit(value):
            out.write(json.dumps(value, allow_nan=False) + '\n')
            out.flush()
        dep = dependencies(args.sglang_root, with_decode=args.phase == 'decode')
        torch, _, _, _, check, prefill, _, _ = dep
        props = torch.cuda.get_device_properties(0)
        if torch.version.hip is None or props.gcnArchName.split(':')[0] != 'gfx942':
            raise RuntimeError('this comparison targets ROCm gfx942')
        paths = [Path(__file__),
                 args.sglang_root / 'python/sglang/srt/layers/hyperconnection.py',
                 args.sglang_root / 'python/sglang/srt/layers/hc_mix_triton.py']
        paths.extend((REPO / 'src/contrib/flydsl/gr_read').glob('*.py'))
        if args.phase == 'decode':
            paths.append(REPO / 'tests/contrib/gr_read_decode/kernel.py')
        emit({'type': 'environment', 'torch': torch.__version__, 'hip': torch.version.hip,
              'gpu': props.name, 'arch': props.gcnArchName, 'compute_units': props.multi_processor_count,
              'args': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              'sources': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
              'sglang_head': subprocess.check_output(['git', '-C', str(args.sglang_root), 'rev-parse', 'HEAD'], text=True).strip(),
              'scope': 'normalized-input GR read only; excludes RMSNorm, TP collectives and model execution',
              'settings_written': False})
        gate(prefill, args, 'entry', emit)
        with torch.inference_mode():
            pairs = [check.make_inputs(1, args.seed + i)[1:] for i in range(args.weights)]
            from pyhip.contrib.flydsl.gr_read import prepare_weights
            packed_pairs = [prepare_weights(wd, wu) for wd, wu in pairs]
            emit({'type': 'weight_preparation', 'pairs': len(pairs), 'packing_calls': len(pairs),
                  'shared_across_rows': True})
            for rows in args.rows:
                benchmark_rows(rows, pairs, args, dep, emit, packed_pairs)
                gc.collect()
                torch.cuda.empty_cache()
        gate(prefill, args, 'exit', emit)
        emit({'type': 'summary', 'complete': True, 'rows': args.rows})


if __name__ == '__main__':
    main()
