# SPDX-License-Identifier: MIT
"""Shared in-process MoE measurement helpers and direct benchmarks.

CLI: moe_bench.py TP,H,global_I,E,topk,quant,activation[,key=value...][,tokens...] --driver 'NAME_OR_REGEX'
cross_compare reuses the same data, reference and measurement helpers.
"""

import argparse
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import math
import os
from pathlib import Path
import re
import statistics
import sys
import time

import torch
from torch.utils._pytree import tree_flatten, tree_unflatten

if __package__:
    from . import moe_driver, moe_ref, moe_tuned
else:
    import moe_driver
    import moe_ref
    import moe_tuned

PROTOCOL = "eager_moe_driver_v3"


def driver_config(spec):
    """Convert the CLI's global I/TP to the driver's local, immutable config."""
    values = moe_tuned.config_values(spec)
    values["output_dtype"] = getattr(torch, values["output_dtype"])
    return moe_driver.MOEconfig(**values)


def make_data(spec, seed, *, device="cuda"):
    """Deterministic inputs and identical weights across token counts."""
    M, H, I, E, K = (spec["tokens"], spec["model_dim"], spec["inter_dim"] // spec["TP"],
                     spec["experts"], spec["topk"])
    wg = torch.Generator(device=device).manual_seed(seed)
    ag = torch.Generator(device=device).manual_seed(seed + 1)
    w1 = torch.empty((E, 2 * I, H), device=device, dtype=torch.bfloat16)
    w2 = torch.empty((E, H, I), device=device, dtype=torch.bfloat16)
    for e in range(E):  # Bound the FP32 initialization temporary to one expert.
        w1[e].copy_(torch.randn((2 * I, H), generator=wg, device=device) / math.sqrt(H))
        w2[e].copy_(torch.randn((H, I), generator=wg, device=device) / math.sqrt(I))
    x = torch.randn((M, H), generator=ag, device=device, dtype=torch.bfloat16)
    scores, ids = torch.randn((M, E), generator=ag, device=device).topk(K, dim=-1)
    return dict(hidden_states=x, weight1=w1, weight2=w2,
                topk_weight=scores.softmax(-1), topk_ids=ids.int())


def prepare_buffer(config, prepare, data):
    """One private input set; run_perftest owns the timing copies."""
    moe_driver.validate_routes(config, data["topk_ids"], data["topk_weight"])
    prepared = prepare(data["weight1"], data["weight2"])
    return (data["hidden_states"].clone(), prepared,
            data["topk_weight"].clone(), data["topk_ids"].clone(),
            torch.full_like(data["hidden_states"], float("nan"), dtype=config.output_dtype))


def launch(run, buffer):
    x, weights, tw, ti, out = buffer
    result = run(x, weights, ti, tw, out)
    if result is not out:
        raise RuntimeError("driver did not honor the output buffer")
    return result


def make_reference(spec, data, *, intermediate_dtype=None, route_dtype=None,
                   output_dtype=None, token_chunk_size=256):
    config = driver_config(spec)
    dtype = output_dtype or (config.output_dtype if "output_dtype" in spec else torch.float32)
    config = config._replace(preshuffle=False, output_dtype=dtype)
    prepare, run = moe_driver.ref(config, intermediate_dtype=intermediate_dtype,
                                  route_dtype=route_dtype, token_chunk_size=token_chunk_size)
    weights = prepare(data["weight1"], data["weight2"])
    output = torch.empty_like(data["hidden_states"], dtype=dtype)
    return run(data["hidden_states"], weights, data["topk_ids"], data["topk_weight"], output)


def measure(op, buffer, reference, *, warmup=5, iters=20, rounds=5, buffer_count=10,
            diff_threshold=0.02, time_failed=False):
    """Delegate copying/warmup/timing to run_perftest; check every used output copy."""
    import pyhip

    # run_perftest clones top-level tensors, including leaves of tuned weight trees.
    args, tree = tree_flatten(buffer)
    outputs = {}

    def invoke(*args):
        output = launch(op, tree_unflatten(args, tree))
        outputs[output.data_ptr()] = output
        return output

    def diff():
        if time_failed and any(not bool(torch.isfinite(output).all()) for output in outputs.values()):
            return math.inf
        return max(moe_ref.calc_diff(output, reference) for output in outputs.values())

    invoke(*args)
    error = diff()
    if error > diff_threshold and not time_failed:
        return dict(status="FAIL", diff=error, reason="pre-timing accuracy failed")
    samples = []
    for _ in range(rounds):
        outputs.clear()
        buffer[-1].fill_(float("nan"))  # Copies must not inherit a previously valid output.
        _, us = pyhip.run_perftest(invoke, *args, num_warmup=warmup, num_iters=iters,
                                  num_copies=buffer_count)
        samples.append(us * 1e-6)
        error = max(error, diff())
    if not all(math.isfinite(value) and value > 0 for value in samples):
        raise RuntimeError(f"invalid event timing samples: {samples}")
    result = dict(status="OK" if error <= diff_threshold else "FAIL", diff=error if math.isfinite(error) else None,
                  e2e_latency_s=statistics.median(samples), samples_s=samples,
                  benchmark_protocol=PROTOCOL, buffer_count=buffer_count)
    if result["status"] == "FAIL":
        result["reason"] = "accuracy failed; latency is diagnostic only"
    return result


@torch.no_grad()
def prepare_batch(spec, options):
    """Generate one batch and reference, shared read-only by all candidates."""
    start = time.perf_counter()
    data = make_data(spec, options["seed"], device=torch.device("cuda", options["device"]))
    torch.cuda.synchronize()
    data_s = time.perf_counter() - start
    start = time.perf_counter()
    reference = make_reference(
        spec, data,
        intermediate_dtype=torch.bfloat16 if options["ref_intermediate_dtype"] == "bf16" else None,
        route_dtype=torch.bfloat16 if options["ref_route_dtype"] == "bf16" else None,
        token_chunk_size=options["ref_chunk_size"],
    )
    torch.cuda.synchronize()
    return data, reference, dict(status="OK", make_data_s=data_s,
                                 make_reference_s=time.perf_counter() - start,
                                 retained_bytes=sum(t.numel() * t.element_size() for t in (*data.values(), reference)))


def measure_driver(name, spec, options, data, reference, *, tuned_config=None):
    """Prepare and measure one driver against the batch's shared reference."""
    config = driver_config(spec)
    prepare, run = (moe_driver.aiter(config, tuned_config=tuned_config) if name == "aiter" else
                    moe_driver.registry[name](config))
    with torch.no_grad():
        start = time.perf_counter()
        buffer = prepare_buffer(config, prepare, data)
        torch.cuda.synchronize()
        prepare_s = time.perf_counter() - start
        result = measure(run, buffer, reference, **{
            name: options[name] for name in ("warmup", "iters", "rounds", "buffer_count", "diff_threshold")})
        result["prepare_buffers_s"] = prepare_s
        if name == "aiter":
            metadata = next(iter(run.metadata.values()))
            names = [moe_driver._kernel_name(getattr(metadata, f"stage{stage}"), stage)
                     for stage in ((1,) if metadata.run_1stage else (1, 2))]
            result.update(dispatch_verified=True, kernel_name=" → ".join(names), kernel_names=names)
        else:
            result["kernel_name"] = name
        result["activation_path"] = getattr(run, "activation_path", "aiter_public")
        return result


@contextmanager
def log_output(path):
    """Capture Python, native HIP and compiler output during a serial call."""
    sys.stdout.flush()
    sys.stderr.flush()
    saved = os.dup(1), os.dup(2)
    try:
        with Path(path).open("w", buffering=1) as log:
            os.dup2(log.fileno(), 1)
            os.dup2(log.fileno(), 2)
            with redirect_stdout(log), redirect_stderr(log):
                yield
    finally:
        for fd, original in zip((1, 2), saved):
            os.dup2(original, fd)
            os.close(original)


def main(argv=None):
    if __package__:
        from . import tune_aiter
    else:
        import tune_aiter

    parser = argparse.ArgumentParser(description=__doc__, epilog=tune_aiter._config_help(),
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=tune_aiter._parse_config,
                        help="model alias (listed below) or TP,H,global_I,E,topk,quant,activation[,key=value...][,tokens...]")
    parser.add_argument("--driver", required=True, help="registered function/ref/aiter/tuned name or full-match regex")
    parser.add_argument("--output-dir", default="./tuned_aiter", help="results directory; also tuned CSV source for Aiter")
    parser.add_argument("--device", type=int, default=0)
    for flag, default in (("seed", 43), ("warmup", 5), ("iters", 20), ("rounds", 5), ("buffer-count", 10)):
        parser.add_argument(f"--{flag}", type=int, default=default)
    parser.add_argument("--diff-threshold", type=float, default=0.02)
    args = parser.parse_args(argv)

    drivers = sorted(set(moe_driver.registry) | {"ref", "aiter", "tuned"})
    config = dict(args.config)
    tokens = config.pop("tokens")
    try:
        pattern = re.compile(args.driver)
        selected = [name for name in drivers if pattern.fullmatch(name)]
        if not selected:
            raise ValueError(f"--driver matched no drivers: {args.driver!r}")
        specs = [tune_aiter._spec(**config, tokens=m) for m in tokens]
        for name in ("warmup", "iters", "rounds", "buffer_count"):
            tune_aiter._positive_int(name, getattr(args, name))
        for name in ("seed", "device"):
            tune_aiter._positive_int(name, getattr(args, name), minimum=0)
        if not math.isfinite(args.diff_threshold) or not 0 <= args.diff_threshold <= 2:
            raise ValueError("diff_threshold must be finite and in [0,2]")
    except (ValueError, TypeError, NotImplementedError, re.error) as exc:
        parser.error(str(exc))

    baseline_driver = next(iter(selected))
    output = Path(args.output_dir).expanduser().resolve()
    settings = {name: getattr(args, name) for name in ("seed", "warmup", "iters", "rounds", "buffer_count", "diff_threshold")}
    results = {name: [] for name in selected}
    try:
        info = tune_aiter._device_info(args.device)
        output.mkdir(parents=True, exist_ok=True)
        for name in selected:
            (output / f"bench-{name}.json").unlink(missing_ok=True)
        print(f"# {args.driver} — {info['name']} — device={args.device}", flush=True)
        print(f"Baseline: {baseline_driver}", flush=True)
        print("| Tokens | Driver | Status | calc_diff ↓ | E2E μs ↓ | Speedup ↑ |\n|---:|---|---|---:|---:|---:|", flush=True)
        with torch.cuda.device(args.device), torch.no_grad():
            for spec in specs:
                data = reference = None
                baseline = None
                for name in selected:
                    config = driver_config(spec)
                    if name == "aiter" and config.preshuffle:
                        winner = tune_aiter._read_winner(output / "tuned.csv", tune_aiter._untuned_row(
                            spec, info["gfx"], info["cu_num"]), history=True)
                        if winner is None:
                            raise RuntimeError(f"no tuned Aiter row for tokens={spec['tokens']} in {output}; run tune_aiter first")
                        prepare, run = moe_driver.aiter(config, tuned_config=winner["tuned_config"])
                    elif name == "aiter":
                        prepare, run = moe_driver.aiter(config)
                    elif name == "tuned":
                        prepare, run = moe_driver.tuned(config, tuned_csv=output / "moe_tuned.csv", tokens=spec["tokens"])
                    elif name == "ref":
                        prepare, run = moe_driver.ref(config._replace(preshuffle=False))
                    else:
                        prepare, run = moe_driver.registry[name](config)
                    if data is None:
                        data = make_data(spec, args.seed, device=torch.device("cuda", args.device))
                        reference = make_reference(spec, data)  # One reference with the same numerical config for all drivers.
                    buffer = prepare_buffer(config, prepare, data)
                    result = measure(run, buffer, reference, warmup=args.warmup, iters=args.iters,
                                     rounds=args.rounds, buffer_count=args.buffer_count,
                                     diff_threshold=args.diff_threshold, time_failed=True)
                    latency = result["e2e_latency_s"]
                    if baseline is None:
                        baseline = latency
                    result["speedup"] = baseline / latency
                    results[name].append(dict(spec=spec, **result))
                    tune_aiter._write_json(output / f"bench-{name}.json", dict(
                        driver=name, device=info, settings=settings,
                        baseline_driver=baseline_driver, results=results[name]))
                    diff = "nonfinite" if result["diff"] is None else f"{result['diff']:.6g}"
                    print(f"| {spec['tokens']} | {name} | {result['status']} | {diff} | "
                          f"{latency * 1e6:.3f} | {result['speedup']:.3f}x |", flush=True)
                    del buffer, prepare, run
                del data, reference
        for name in selected:
            print(f"Results: {output / f'bench-{name}.json'}", flush=True)
        return 0 if all(result["status"] == "OK" for rows in results.values() for result in rows) else 1
    except (RuntimeError, ValueError, TypeError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())