# SPDX-License-Identifier: MIT
"""GR read regression tests and the compatible accuracy-then-performance CLI.

Performance implementation and end-user documentation live in benchmarks/gr_read.
--check-only skips timing; pytest runs correctness checks only.
"""
import argparse
from contextlib import nullcontext
from functools import cache
import gc
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'src'))
from pyhip.testing import gr_read as checks
from pyhip.testing.gr_read import (
    C,
    H,
    R,
    CHECK_ROWS,
    DOWN_TOLERANCE,
    OUTPUT_TOLERANCE,
    DEFAULT_BATCHES,
    SCOPES,
    parse_batch,
    prepare_cli_environment,
    warn_architecture,
    dependencies,
    prepare_reader,
    prepare_gr_read,
    run_down,
    run_up,
    run_gr_read,
    reference_bf16,
    check_close,
    make_inputs,
    check_batch,
    decode_reference,
    make_decode_inputs,
    capture_decode,
    guarded,
    assert_decode_close,
    check_decode_rows,
    check_decode_stages,
    release_buffers,
    selected_rows,
    run_correctness
)


def use_checkout_package():
    if Path(checks.__file__).resolve() != REPO / 'src/pyhip/testing/gr_read.py':
        raise RuntimeError(f'wrong PyHIP checkout: {checks.__file__}')


def baseline_modules():
    return benchmarking().baseline_modules()


def torch_compile_mix():
    return benchmarking().torch_compile_mix()


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
    parser.add_argument("--md", type=Path, help="optional new Markdown performance report")
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
    if args.check_only and args.md is not None:
        parser.error('--md requires performance measurement; remove --check-only / single --scope')
    checks.validate_report_paths(parser, args.output, args.md)
    return args




@cache
def benchmarking():
    import importlib.util
    path = REPO / "benchmarks/gr_read/bench_gr_read_compare.py"
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
    for name in ("amd_smi", "output", "md"):
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
            startup_hardware = bench.benchmark_entry_gate(performance, emit_performance)
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
                startup_hardware=startup_hardware,
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


    @pytest.mark.parametrize('rows', (1, 17))
    def test_decode_total_graph_addresses(monkeypatch, rows):
        import weakref
        import torch

        bench = benchmarking()
        dep = bench.dependencies(no_baselines=True)
        from pyhip.ops.gr_read.flydsl import host
        cases = []
        with torch.inference_mode():
            for seed in (901, 902):
                x, wd, wu, pd, pu = api_inputs(rows, seed=seed)
                case = bench.Case(rows, x, wd, wu, argparse.Namespace(phase='decode'),
                                  dep, packed=(pd, pu))
                case.run('total')
                cases.append(case)
            key = next(k for k in host._compiled_calls if k[0] == x.device.index and k[2] == rows)
            native_pair, = host._compiled_calls[key]
            launch_partials, partial_refs, records = [], [], []

            def observed_pair(*args):
                if torch.cuda.is_current_stream_capturing():
                    # Independent check at the launch boundary: P is argument 3.
                    launch_partials.append(bench.address(args[3]))
                    partial_refs.append(weakref.ref(args[3]))
                return native_pair(*args)

            native_empty = torch.empty
            with monkeypatch.context() as patch:
                patch.setitem(host._compiled_calls, key, (observed_pair,))
                calls = [lambda c=c, bi=bi: bench.capture_total_call(c, bi, len(cases), records)
                         for bi, c in enumerate(cases)]
                graph = bench.capture(calls + calls)
            assert torch.empty is native_empty
            assert len(records) == len(launch_partials) == 4  # Excludes eager warmups.
            assert [r['P_total'] for r in records] == launch_partials
            assert [(r['call_index'], r['pass_index'], r['buffer']) for r in records] == [
                (0, 0, 0), (1, 0, 1), (2, 1, 0), (3, 1, 1)]
            assert all(ref() is None for ref in partial_refs), 'logging must not retain P tensors'
            assert json.loads(json.dumps(records)) == records
            for record in records:
                case = cases[record['buffer']]
                partial = record['P_total']
                assert partial['pointer'] != case.p.data_ptr()
                assert partial['shape'] == [4 * rows * R]
                assert record['logical_shape'] == [4, rows, R]
                for name, tensor in (('X', case.x), ('WD_packed', case.pd),
                                     ('WU_packed', case.pu), ('Y', case.y)):
                    assert record['relative_bytes']['P_minus_' + name] == partial['pointer'] - tensor.data_ptr()
            for _ in range(2):
                for case in cases:
                    case.x.mul_(.9).add_(.015625)
                    case.y.fill_(torch.nan)
                graph.replay()
                for case in cases:
                    assert_decode_close(case.y, decode_reference(case.x, case.wd, case.wu),
                                        'Total graph with recorded native workspace')
                assert len(records) == 4  # Replay does not run the observer or allocate P.


    def test_partial_logging_output_option(tmp_path):
        output = tmp_path / 'samples.jsonl'
        assert benchmark_options(parse_args([])).output is None
        assert benchmark_options(parse_args(['--output', str(output)])).output == output
        assert benchmarking().parse_args(['--output', str(output)]).output == output


    def test_benchmark_report_paths(tmp_path):
        output, markdown = tmp_path / 'samples.jsonl', tmp_path / 'report.md'
        assert benchmark_options(parse_args(['--md', str(markdown)])).md == markdown
        for parser in (parse_args, benchmarking().parse_args):
            with pytest.raises(SystemExit):
                parser(['--output', str(output), '--md', str(output)])
            markdown.write_text('keep this report')
            with pytest.raises(SystemExit):
                parser(['--md', str(markdown)])
            assert markdown.read_text() == 'keep this report'
            markdown.unlink()
        with pytest.raises(SystemExit):
            parse_args(['--check-only', '--md', str(markdown)])


    @pytest.mark.parametrize('with_baselines', (False, True))
    def test_benchmark_markdown_report(tmp_path, with_baselines):
        bench = benchmarking()
        path = tmp_path / 'report.md'
        baseline = ({'source_checkout_commit': 'source-version',
                     'triton_last_change_commit': 'tuning-version',
                     'runtime_versions': {'torch': 'torch-version', 'triton': 'triton-version'}}
                    if with_baselines else None)
        env = {'gpu': 'MI308X', 'arch': 'gfx942', 'compute_units': 80,
               'torch': 'torch-version', 'hip': 'hip-version',
               'protocol': {'record_total_partials': with_baselines}, 'baseline': baseline,
               'sources': {str(bench.REPO / 'src/pyhip/testing/gr_read.py'): 'source-hash'}}
        timings = {'down': 4., 'up': 6., 'total': 10.}
        prefill_timings = {scope: {'elapsed_us': value, 'effective_TFLOPS': 1.}
                           for scope, value in timings.items()}
        if with_baselines:
            timings['baseline'] = 20.
            prefill_timings['torch_compile'] = {'elapsed_us': 20., 'effective_TFLOPS': .5}
        results = {'decode': [{'rows': 1, 'median_us': timings,
                              'baseline_backend': 'triton' if with_baselines else None,
                              'speedup_total': 2. if with_baselines else None}],
                   'prefill': [{'rows': 33, 'timings': prefill_timings, 'down_block_m': 16,
                                'down_num_waves': 2, 'down_n_splits': 10, 'down_block_k': 1024,
                                'up_block_m': 64, 'n_splits': 40,
                                'speedup_vs_torch_compile': 2. if with_baselines else None}]}
        hardware = {'gpu': 2, 'card': {'PCI Bus': '0000:A4:00.0', 'GPU use (%)': '0',
                                     'GPU Memory Allocated (VRAM%)': '0'},
                    'limit': {'ptl_state': 'Enabled', 'ptl_format': 'VECTOR,F8'}}
        args = argparse.Namespace(output=tmp_path / 'raw.jsonl' if with_baselines else None)
        bench.write_markdown_report(path, args, results, {'decode': env, 'prefill': env},
                                    hardware, '2026-09-29T00:00:00+00:00')
        report = path.read_text()
        for expected in ('## Decode', '## Prefill', 'MI308X', '80 CUs', '0000:A4:00.0',
                         'M16/W2/N10/BK1024', 'src/pyhip/testing/gr_read.py', '10.000'):
            assert expected in report
        if with_baselines:
            assert 'source-version' in report and '2.000x' in report and '<raw.jsonl>' in report
        else:
            assert 'Frozen performance baselines disabled' in report and 'not exported' in report
        with pytest.raises(FileExistsError):
            bench.write_markdown_report(path, args, results, {'decode': env, 'prefill': env},
                                        hardware, 'later')
        assert path.read_text() == report


    @pytest.mark.parametrize('rows,record_partials', ((33, True), (513, True), (33, False)))
    def test_prefill_total_sample_addresses(monkeypatch, rows, record_partials):
        import weakref
        import torch

        bench = benchmarking()
        prefill = bench.testing()
        dep = prefill.dependencies()
        from pyhip.ops.gr_read.flydsl import host
        x, wd, wu, pd, pu = api_inputs(rows)
        host.gr_read(x, pd, pu)
        config = host._prefill_config(rows, torch.cuda.get_device_properties(x.device).multi_processor_count)
        key = next(k for k in host._compiled_calls if k[0] == x.device.index and k[2] == config)
        kernels = host._compiled_calls[key]
        launches, partial_refs, records = [], [], []
        active_scope = None

        # Exercise sample boundaries without doing performance measurement in pytest.
        class AccuracyTimer:
            enable = True

            def __init__(self, name, verbose):
                self.name, self.latencies = name, []

            def __enter__(self):
                nonlocal active_scope
                active_scope = self.name

            def __exit__(self, *args):
                nonlocal active_scope
                active_scope = None
                self.latencies.append(1e-6)

        def observed_kernel(kernel):
            def launch(*args):
                if active_scope == 'gr_read_total':
                    partial = args[3] if len(kernels) == 1 else args[2]
                    launches.append(bench.address(partial))
                    partial_refs.append(weakref.ref(partial))
                return kernel(*args)
            return launch

        make_record = bench.partial_address_record

        def checked_record(*args):
            assert active_scope is None, 'build address metadata only after timing'
            return make_record(*args)

        def unexpected_observer(*args, **kwargs):
            pytest.fail('console-only prefill must not install a P observer')

        native_empty = torch.empty
        monkeypatch.setitem(host._compiled_calls, key, tuple(observed_kernel(k) for k in kernels))
        monkeypatch.setattr(prefill, 'dependencies', lambda: (*dep[:3], AccuracyTimer))
        monkeypatch.setattr(bench, 'partial_address_record', checked_record)
        if not record_partials:
            monkeypatch.setattr(bench, 'observe_partial_allocations', unexpected_observer)
        args = argparse.Namespace(seed=131, buffers=2, warmup=2, iters=3,
                                  no_baselines=True, record_partials=record_partials)
        result = bench.benchmark_prefill_batch(rows, args, records.append)
        assert result['timed_outputs_bitexact'] and torch.empty is native_empty
        assert all(ref() is None for ref in partial_refs)
        buffers = next(r['buffers'] for r in records if r['type'] == 'addresses')
        samples = [r for r in records if r['type'] == 'sample' and r['scope'] == 'total']
        assert [r['buffer'] for r in samples] == [0, 1, 0]
        assert len(launches) == len(samples) * len(kernels)
        for index, sample in enumerate(samples):
            assert ('P_total' in sample) == record_partials
            if not record_partials:
                continue
            partial = sample['P_total']
            assert all(p == partial for p in launches[index * len(kernels):(index + 1) * len(kernels)])
            assert partial['shape'] == [rows, R] and partial['dtype'] == 'torch.bfloat16'
            buffer = buffers[sample['buffer']]
            assert partial['pointer'] != buffer['P_stages']['pointer']
            for name, field in (('X', 'X'), ('WD_packed', 'W_down'), ('WU_packed', 'W_up'), ('Y', 'Y')):
                assert sample['relative_bytes']['P_minus_' + name] == partial['pointer'] - buffer[field]['pointer']
        assert not any('P_total' in r for r in records if r.get('scope') in ('down', 'up'))
        json.dumps(records)


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
