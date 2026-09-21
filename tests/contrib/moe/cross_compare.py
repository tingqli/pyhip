# SPDX-License-Identifier: MIT
"""Compare complete MoE drivers, using a verified tuned Aiter baseline when available.

CLI config: TP,H,global_I,E,topk,quant,activation[,key=value...][,tokens...].
Omit config to sweep MODEL_CONFIGS and all default token counts.
Candidates run serially in the main process, sharing one batch and reference.
Only the official Aiter tuner uses a separate process; benchmarks have no timeout.
Missing rows are tuned without an extra benchmark.
stdout contains Markdown, while compiler/kernel diagnostics remain in logs.
Default output is one Aiter/best comparison row per config and batch; -v expands it.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import sys
import tempfile
import time
import traceback

from tqdm import tqdm

if __package__:
    from .utils import moe_tuned, tune_aiter
else:
    from utils import moe_tuned, tune_aiter

__all__ = ["compare", "compare_all", "render_markdown", "render_html", "main"]

_TEMPLATE = Path(__file__).with_name("utils") / "moe_compare.html"
_PROTOCOL = "eager_moe_driver_v3"
_SUMMARY_HEADER = (
    "| Config | Tokens | Aiter status | Aiter μs ↓ | Best μs ↓ | Aiter / best ↑ | "
    "Aiter diff ↓ | Best diff ↓ | Aiter kernel | Best driver / kernel |\n"
    "|---|---:|---|---:|---:|---:|---:|---:|---|---|\n"
)


def _print(text, stream, *, end="\n"):
    """Keep Markdown intact while tqdm redraws its stderr progress bars."""
    if stream is not None:
        tqdm.write(text, file=stream, end=end)
        stream.flush()


def render_html(reports):
    """One offline HTML file; escape '<' so embedded JSON cannot end its script."""
    template = _TEMPLATE.read_text(encoding="utf-8")
    data = json.dumps(dict(benchmark_protocol=_PROTOCOL, reports=reports),
                      ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    return template.replace("__MOE_DATA__", data)


def _write_dashboard(reports, directory):
    directory.mkdir(parents=True, exist_ok=True)
    tune_aiter._write_json(directory / "comparison.json", dict(benchmark_protocol=_PROTOCOL, reports=reports))
    (directory / "comparison.html").write_text(render_html(reports), encoding="utf-8")


def _driver_module():
    if __package__:
        from .utils import moe_driver
    else:
        from utils import moe_driver
    return moe_driver


def _preflight(spec, key, params):
    """Host-only shape/contract rejection before allocating GPU test data."""
    if __package__:
        from .utils.moe_bench import driver_config
    else:
        from utils.moe_bench import driver_config
    module = _driver_module()
    try:
        module.registry[key](driver_config(spec))
    except (NotImplementedError, ValueError) as error:
        return _row(key, params, "UNSUPPORTED", reason=str(error))
    except Exception as error:
        return _row(key, params, "ERROR", reason=f"factory error: {type(error).__name__}: {error}")
    return None


def _row(key, params, status, **extra):
    return {**dict(candidate=key, driver=params["name"], params=dict(params), status=status,
                   diff=None, e2e_latency_s=None, speedup=None, samples_s=[], kernel_name=key), **extra}


def _benchmark(key, params, spec, options, data, reference, log_path, *, baseline=None, tuned_config=None):
    """Call a driver directly; log ordinary errors without a retry or worker."""
    if __package__:
        from .utils import moe_bench
    else:
        from utils import moe_bench

    start = time.perf_counter()
    with moe_bench.log_output(log_path):
        try:
            result = moe_bench.measure_driver(key, spec, options, data, reference, tuned_config=tuned_config)
        except NotImplementedError as error:
            result = dict(status="UNSUPPORTED", reason=str(error))
        except Exception as error:
            traceback.print_exc()
            result = dict(status="ERROR", reason=f"{type(error).__name__}: {error}")
    row = _row(key, params, **result, log=str(log_path), wall_s=time.perf_counter() - start)
    if row["status"] == "OK":
        row["speedup"] = 1.0 if key == "aiter" else baseline / row["e2e_latency_s"] if baseline is not None else None
    return row


def _ensure_tuned(spec, options, device_info):
    tuned = tune_aiter.ensure_tuned(
        spec, Path(options["aiter_root"]), Path(options["output_dir"]), options["device"], device_info,
        seed=options["seed"], timeout_s=options["tune_timeout_s"], tune_backends=options["tune_backends"],
        kernel_regex=options["kernel_regex"], force=options["force"], no_tune=options["no_tune"],
    )
    return dict(**tuned,
                search_scope=("cached row; original search scope unknown" if tuned["tuning_skipped"] else
                              f"official catalog; backends={options['tune_backends'] or 'all'}, regex={options['kernel_regex'] or 'none'}"))


def _cell(value):
    return str(value).replace("|", "\\|").replace("\r", " ").replace("\n", "<br>")


def _model_name(spec):
    return next((alias for alias, value in tune_aiter.MODEL_CONFIGS.items()
                 if all(spec[name] == item for name, item in tune_aiter._parse_config(value).items()
                        if name not in ("tokens", "driver_kwargs"))),
                f"TP{spec['TP']}-H{spec['model_dim']}-I{spec['inter_dim']}-E{spec['experts']}-"
                f"K{spec['topk']}-{spec['quant_scheme_str']}-{spec['activation']}")


def render_markdown(batch, verbose=0, *, model=None, header=True):
    """0: compact Aiter/best row; 1: Aiter + OK candidates; >1: all rows."""
    spec = batch["spec"]
    if verbose == 0:
        baseline = next((row for row in batch["rows"] if row["candidate"] == "aiter"), None)
        best = moe_tuned.best_row({"rows": [row for row in batch["rows"] if row["candidate"] != "aiter"]})
        latencies = ["—" if row is None or row.get("e2e_latency_s") is None else f"{row['e2e_latency_s'] * 1e6:.3f}"
                     for row in (baseline, best)]
        diffs = ["—" if row is None or row.get("diff") is None else f"{row['diff']:.6g}"
                 for row in (baseline, best)]
        speedup = (f"{baseline['e2e_latency_s'] / best['e2e_latency_s']:.3f}x"
                   if baseline is not None and baseline["status"] == "OK" and best is not None else "—")
        cells = (model or _model_name(spec), spec["tokens"], baseline["status"] if baseline else "NOT_RUN",
                 *latencies, speedup, *diffs,
                 baseline.get("kernel_name", "—") if baseline else "—", best["kernel_name"] if best else "—")
        return (_SUMMARY_HEADER if header else "") + "| " + " | ".join(map(_cell, cells)) + " |\n"
    lines = [f"### tokens={spec['tokens']} (Aiter token key={tune_aiter._token_key(spec['tokens'])})", ""]
    if batch.get("tuning"):
        tuned = batch["tuning"]
        mode = "reused exact row" if tuned["tuning_skipped"] else "tuned missing/forced row"
        lines.extend([f"Aiter: {mode}; {_cell(tuned['search_scope'])}.", ""])
    preparation = batch.get("preparation", {})
    if preparation.get("status") == "OK":
        lines.extend([f"GPU batch: data {preparation['make_data_s']:.3f}s, "
                      f"reference {preparation['make_reference_s']:.3f}s (once); "
                      f"{preparation['retained_bytes'] / 2**30:.3f} GiB retained. Preparation is outside E2E timing.", ""])
    if batch["rows"] and batch["rows"][0]["status"] != "OK":
        lines.extend(["No valid Aiter baseline; candidate accuracy/latency are independent, speedups are unavailable.", ""])
    lines.extend(["| Candidate | Status | calc_diff ↓ | E2E μs ↓ | Aiter / candidate ↑ | Kernel / configuration / reason |",
                  "|---|---|---:|---:|---:|---|"])
    for row in batch["rows"]:
        if verbose == 1 and row["status"] != "OK" and row["candidate"] != "aiter":
            continue
        diff = "—" if row.get("diff") is None else f"{row['diff']:.6g}"
        latency = "—" if row.get("e2e_latency_s") is None else f"{row['e2e_latency_s'] * 1e6:.3f}"
        speed = "—" if row.get("speedup") is None else f"{row['speedup']:.3f}x"
        detail = row.get("kernel_name", row["driver"])
        if row["driver"] != "aiter":
            detail = f"{row['driver']}({', '.join(f'{k}={v!r}' for k, v in row['params'].items() if k != 'name')})"
            if row.get("activation_path"):
                detail += f"; A={row['activation_path']}"
        if row.get("reason"):
            reason = " ".join(str(row["reason"]).split())
            detail += "; " + (reason if len(reason) <= 350 else reason[:350] + "… (see log)")
        lines.append("| " + " | ".join(map(_cell, (row["candidate"], row["status"], diff, latency, speed, detail))) + " |")
    return "\n".join(lines) + "\n"


def _validate(TP, model_dim, inter_dim, experts, topk, tokens, quant_scheme_str, activation, options):
    if tokens is None:
        tokens = list(tune_aiter._DEFAULT_TOKENS)
    elif isinstance(tokens, int) and not isinstance(tokens, bool):
        tokens = [tokens]
    else:
        tokens = list(tokens)
    if not tokens or len(tokens) != len(set(tokens)):
        raise ValueError("tokens must be a nonempty list of unique positive integers")
    for name in ("warmup", "iters", "rounds", "buffer_count", "tune_timeout_s", "ref_chunk_size"):
        tune_aiter._positive_int(name, options[name])
    tune_aiter._positive_int("seed", options["seed"], minimum=0)
    tune_aiter._positive_int("device", options["device"], minimum=0)
    tune_aiter._positive_int("verbose", options["verbose"], minimum=0)
    if options["seed"] >= 2**63 - 1:
        raise ValueError("seed must be < 2**63-1")
    threshold = options["diff_threshold"]
    if isinstance(threshold, bool) or not math.isfinite(threshold) or not 0 <= threshold <= 2:
        raise ValueError("diff_threshold must be finite and in [0,2]")
    for name in ("force", "no_tune", "dry_run"):
        if not isinstance(options[name], bool):
            raise TypeError(f"{name} must be bool")
    if options["force"] and options["no_tune"]:
        raise ValueError("--force and --no-tune are mutually exclusive")
    for name in ("ref_intermediate_dtype", "ref_route_dtype"):
        if options[name] not in ("none", "bf16"):
            raise ValueError(f"{name} must be none or bf16")
    specs = [tune_aiter._spec(TP, model_dim, inter_dim, experts, topk, m, quant_scheme_str, activation,
                             options["driver_kwargs"]) for m in tokens]
    for spec in specs:
        tune_aiter._check_tuning_spec(spec)
    if options["tune_backends"] is not None and any(
        value not in tune_aiter._BACKENDS for value in options["tune_backends"].split(",") if value
    ):
        raise ValueError(f"tune_backends must be a comma-list from {tune_aiter._BACKENDS}")
    if options["kernel_regex"] is not None:
        import re
        try:
            re.compile(options["kernel_regex"])
        except re.error as error:
            raise ValueError(f"invalid kernel_regex: {error}") from error
    return specs


def compare(
    TP, model_dim, inter_dim, experts, topk, tokens=None,
    quant_scheme_str="fp8_ptpc", activation="silu", *,
    output_dir="./tuned_aiter", device=0, aiter_root=None, no_tune=False, force=False,
    seed=43, warmup=5, iters=20, rounds=5, buffer_count=10,
    diff_threshold=0.02, tune_timeout_s=120, tune_backends=None, kernel_regex=None,
    ref_intermediate_dtype="none", ref_route_dtype="none", ref_chunk_size=256,
    dry_run=False, stream=None, verbose=0, driver_kwargs=None, model=None, _previous_reports=None,
):
    """Compare all registered local candidates and optionally print Markdown.

    Default: reuse latest exact tuned rows, tune only missing keys. --no-tune
    never searches or changes Aiter's CSV. Reference rounding is fixed across ALL rows
    (default FP32, no intermediate/route cast), never adapted per candidate.
    Aiter failure does not block candidate measurements; only relative speedups
    require a valid baseline. Every timing uses the same complete-call protocol.
    verbose changes only Markdown output, never measurements or JSON/HTML/CSV.
    """
    options = dict(output_dir=output_dir, device=device, aiter_root=aiter_root,
                   no_tune=no_tune, force=force, seed=seed, warmup=warmup, iters=iters, rounds=rounds,
                   buffer_count=buffer_count, diff_threshold=diff_threshold,
                   tune_timeout_s=tune_timeout_s, tune_backends=tune_backends, kernel_regex=kernel_regex,
                   ref_intermediate_dtype=ref_intermediate_dtype, ref_route_dtype=ref_route_dtype,
                   ref_chunk_size=ref_chunk_size, dry_run=dry_run, verbose=verbose, driver_kwargs=driver_kwargs)
    if output_dir is None:
        raise ValueError("cross_compare requires output_dir for tuned baseline selection")
    specs = _validate(TP, model_dim, inter_dim, experts, topk, tokens, quant_scheme_str, activation, options)
    overrides = {name: specs[0][name] for name in (driver_kwargs or {})}
    options["driver_kwargs"] = overrides
    selected = {name: {"name": name} for name in _driver_module().registry}

    # Validate all static combinations before any tuning or GPU allocation.
    preflight = [{key: _preflight(spec, key, param) for key, param in selected.items()} for spec in specs]
    if dry_run:
        result = dict(executed=False, model=model, specs=specs, candidates=selected,
                      rejected=[{key: row["reason"] for key, row in item.items() if row} for item in preflight])
        if stream is not None:
            print(json.dumps(result, ensure_ascii=False, indent=2), file=stream)
        return result

    if verbose == 0 and _previous_reports is None:
        _print(_SUMMARY_HEADER, stream, end="")

    import torch

    with torch.cuda.device(device), torch.no_grad():
        return _compare_specs(specs, selected, preflight, options, model,
                              _previous_reports or (), stream)


def _compare_specs(specs, selected, preflight, options, model, previous_reports, stream):
    import torch

    if __package__:
        from .utils import moe_bench
    else:
        from utils import moe_bench

    root = tune_aiter._aiter_root(options["aiter_root"])
    output = Path(options["output_dir"]).expanduser().resolve()
    device = options["device"]
    verbose = options["verbose"]
    first = specs[0]
    output.mkdir(parents=True, exist_ok=True)
    if (output / "moe_tuned.csv").exists():
        moe_tuned.read_rows(output / "moe_tuned.csv")  # Reject old schemas before tuning/GPU work.
    run = Path(tempfile.mkdtemp(prefix="cross-compare-", dir=output))
    options.update(output_dir=str(output), aiter_root=str(root))
    tune_aiter._write_json(run / "plan.json", dict(options=options, specs=specs, candidates=selected))
    _print(f"{model or 'MoE'}: probing GPU (log: {run / 'setup.log'})",
           sys.stderr if stream is not None else None)
    with moe_bench.log_output(run / "setup.log"):
        import aiter

        if Path(aiter.__file__).resolve().parent.parent != root:
            raise RuntimeError("benchmark imports a different Aiter checkout; set PYTHONPATH before starting Python")
        device_info = tune_aiter._device_info(device)
    model = model or _model_name(first)
    report = dict(executed=True, success=True, model=model, artifact_dir=str(run), output_dir=str(output),
                  device=device_info, benchmark_protocol=_PROTOCOL, options=options, batches=[])
    heading = (f"# MoE cross comparison\n\n"
               f"TP={first['TP']}, H={first['model_dim']}, global I={first['inter_dim']}, "
               f"local I={first['inter_dim'] // first['TP']}, E={first['experts']}, topk={first['topk']}; "
               f"{specs[0]['quant_scheme_str']}, {specs[0]['activation']}\n\n"
               f"Reference: {specs[0].get('output_dtype', 'float32')} output, "
               f"intermediate={options['ref_intermediate_dtype']}, route={options['ref_route_dtype']}; "
               f"seed={options['seed']}, buffers={options['buffer_count']}, warmup={options['warmup']}, "
               f"iters={options['iters']}, rounds={options['rounds']}. "
               f"calc_diff is not a percentage; speedup = freshly measured Aiter / candidate.\n\n"
               f"GPU: {device_info['name']} ({device_info['gfx']}); external load is not excluded.\n\n")
    markdown = heading if verbose else _SUMMARY_HEADER
    if verbose:
        _print(heading, stream, end="")
    for index, spec in enumerate(specs):
        batch_dir = run / f"tokens-{spec['tokens']}"
        batch_dir.mkdir()
        batch = dict(spec=spec, rows=[])
        baseline_params = {"name": "aiter"}
        data = reference = None
        with tqdm(total=1 + len(selected), desc=f"{model} | batch {index + 1}/{len(specs)} | tokens={spec['tokens']}",
                  unit="driver", file=sys.stderr, dynamic_ncols=True, leave=False, disable=stream is None) as progress:
            try:
                baseline = tuned = None
                try:
                    progress.set_postfix_str("aiter: tune/cache lookup")
                    tuned = _ensure_tuned(spec, options, device_info)
                    batch["tuning"] = tuned
                except (OSError, ValueError, RuntimeError) as error:
                    baseline = _row("aiter", baseline_params, "ERROR", reason=str(error))
                progress.set_postfix_str("prepare GPU data + reference (once)")
                with moe_bench.log_output(batch_dir / "prepare.log"):
                    try:
                        data, reference, preparation = moe_bench.prepare_batch(spec, options)
                    except Exception as error:
                        traceback.print_exc()
                        preparation = dict(status="ERROR", reason=f"{type(error).__name__}: {error}")
                batch["preparation"] = dict(preparation, log=str(batch_dir / "prepare.log"))
                ready = preparation["status"] == "OK"
                if not ready:
                    batch["error"] = f"batch preparation failed: {preparation.get('reason')}"
                if tuned is not None and ready:
                    progress.set_postfix_str("aiter: benchmark")
                    baseline = _benchmark("aiter", baseline_params, spec, options, data, reference,
                                          batch_dir / "aiter.log", tuned_config=tuned["tuned_config"])
                if baseline is None:
                    baseline = _row("aiter", baseline_params, "NOT_RUN", reason=batch["error"])
                batch["rows"].append(baseline)
                progress.set_postfix_str(f"aiter: {baseline['status']}", refresh=False)
                progress.update(1)
                baseline_latency = baseline["e2e_latency_s"] if baseline["status"] == "OK" else None
                for candidate_index, (key, param) in enumerate(selected.items()):
                    row = preflight[index][key]
                    if row is None and not ready:
                        row = _row(key, param, "NOT_RUN", reason=batch["error"])
                    if row is None:
                        progress.set_postfix_str(f"{key}: benchmark")
                        row = _benchmark(key, param, spec, options, data, reference,
                                         batch_dir / f"candidate-{candidate_index:03d}.log", baseline=baseline_latency)
                    batch["rows"].append(row)
                    progress.set_postfix_str(f"{key}: {row['status']}", refresh=False)
                    progress.update(1)
                    tune_aiter._write_json(batch_dir / "comparison.json", batch)
            except (OSError, ValueError, RuntimeError) as error:
                batch["error"] = str(error)
                if not batch["rows"]:
                    batch["rows"].append(_row("aiter", baseline_params, "ERROR", reason=str(error)))
                existing = {row["candidate"] for row in batch["rows"]}
                for key, param in selected.items():
                    if key not in existing:
                        batch["rows"].append(preflight[index][key] or _row(key, param, "NOT_RUN", reason="batch setup/reference failed"))
                progress.set_postfix_str("batch setup/reference failed", refresh=False)
                progress.update(len(batch["rows"]) - progress.n)
            finally:
                # Free the batch before the next official tuner runs; keep imports/JIT warm.
                data = reference = None
                moe_bench.moe_driver._ACTIVE_AITER = None
                moe_bench.moe_driver._AITER_CONFIG_ENV = None
                gc.collect()
                torch.cuda.empty_cache()
        batch["success"] = not batch.get("error") and all(row["status"] in ("OK", "UNSUPPORTED") for row in batch["rows"])
        # An unsupported BASELINE is a batch failure, not a skipped experiment.
        batch["success"] = batch["success"] and batch["rows"][0]["status"] == "OK"
        report["success"] = report["success"] and batch["success"]
        report["batches"].append(batch)
        table = render_markdown(batch, verbose, model=model, header=False) + ("\n" if verbose else "")
        markdown += table
        _print(table, stream, end="")
        tune_aiter._write_json(batch_dir / "comparison.json", batch)
        tune_aiter._write_json(run / "comparison.json", report)
        (run / "comparison.md").write_text(markdown)
        moe_tuned.update_csv(output / "moe_tuned.csv", report, batch)
        # Checkpoint the small scalar dashboard; large reference tensors are already gone.
        _write_dashboard([*previous_reports, report], output)
    (run / "comparison.html").write_text(render_html([report]), encoding="utf-8")
    _print(f"Artifacts: {run}", sys.stderr if verbose == 0 and stream is not None else stream)
    return report


def compare_all(*, stream=None, **options):
    """Run each model serially in this process, checkpointing completed batches."""
    reports = []
    dry = options.get("dry_run", False)
    verbose = tune_aiter._positive_int("verbose", options.get("verbose", 0), minimum=0)
    detail_stream = sys.stderr if verbose == 0 and stream is not None else stream
    if not dry and verbose == 0:
        _print(_SUMMARY_HEADER, stream, end="")
    with tqdm(total=len(tune_aiter.MODEL_CONFIGS), desc="Models", unit="model", file=sys.stderr,
              dynamic_ncols=True, disable=dry or stream is None) as progress:
        for name in tune_aiter.MODEL_CONFIGS:
            progress.set_postfix_str(name)
            if verbose:
                _print(f"\n## Model: {name}", stream)
            try:
                report = compare(**{**options, **tune_aiter._parse_config(name)}, model=name, stream=stream,
                                 _previous_reports=reports)
            except (OSError, RuntimeError) as error:
                report = dict(model=name, executed=True, success=False, error=str(error), batches=[])
                _print(f"{name}: {error}", detail_stream)
            reports.append(report)
            if not dry:
                output = Path(options.get("output_dir", "./tuned_aiter")).expanduser().resolve()
                _write_dashboard(reports, output)
            progress.update(1)
    result = dict(executed=not dry, success=all(report.get("success", True) for report in reports), reports=reports)
    if stream is not None and not dry:
        _print(f"Dashboard: {output / 'comparison.html'}\nWinners: {output / 'moe_tuned.csv'}", detail_stream)
    return result


def _parser():
    parser = argparse.ArgumentParser(description=__doc__, epilog=tune_aiter._config_help(),
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", nargs="?", type=tune_aiter._parse_config,
                        help="model/config with optional tokens; omitted: all MODEL_CONFIGS and tokens 1,2,...,65536")
    parser.add_argument("--output-dir", default="./tuned_aiter", help="Aiter history, comparison.html and moe_tuned.csv directory")
    parser.add_argument("-v", "--verbose", type=int, default=0, metavar="LEVEL",
                        help="Markdown output: 0=Aiter/best columns per batch (default), 1=Aiter + all OK, >1=all candidates")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--aiter-root")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--no-tune", action="store_true", help="cached-only: missing exact CSV rows are errors")
    modes.add_argument("--force", action="store_true", help="force tuning and append new rows; keep compiled caches")
    for flag, default in (("seed", 43), ("warmup", 5), ("iters", 20), ("rounds", 5), ("buffer-count", 10),
                          ("tune-timeout-s", 120), ("ref-chunk-size", 256)):
        parser.add_argument(f"--{flag}", type=int, default=default)
    parser.add_argument("--diff-threshold", type=float, default=0.02)
    parser.add_argument("--tune-backends", help="restrict new official tuning; does not replace cached rows")
    parser.add_argument("--kernel-regex", help="restrict new official tuning (not an exhaustive search)")
    parser.add_argument("--ref-intermediate-dtype", choices=("none", "bf16"), default="none")
    parser.add_argument("--ref-route-dtype", choices=("none", "bf16"), default="none")
    parser.add_argument("--dry-run", action="store_true", help="CPU-only candidate/shape validation; no files, tuning or GPU initialization")
    return parser


def main(argv=None):
    parser = _parser()
    options = vars(parser.parse_args(argv))
    config = options.pop("config")
    try:
        result = (compare(**{**options, **config}, stream=sys.stdout) if config is not None
                  else compare_all(**options, stream=sys.stdout))
        return 0 if not result["executed"] or result["success"] else 1
    except (ValueError, TypeError, NotImplementedError) as error:
        parser.error(str(error))
    except (OSError, RuntimeError) as error:
        print(str(error), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())