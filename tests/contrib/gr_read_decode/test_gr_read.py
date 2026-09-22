# SPDX-License-Identifier: MIT
"""Run BF16/FP64 correctness and CUDA Graph replay checks without a model or pytest."""

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import warnings


def use_checkout_package():
    """Make direct CLI runs use this checkout, even with another editable install."""
    import importlib.util
    import sys
    repo = Path(__file__).resolve().parents[3]
    expected = repo / "src/__init__.py"
    if "pyhip" in sys.modules:
        if Path(sys.modules["pyhip"].__file__).resolve() != expected:
            raise RuntimeError("a different PyHIP checkout is already imported")
        return
    spec = importlib.util.spec_from_file_location("pyhip", expected,
                                                  submodule_search_locations=[str(repo / "src")])
    module = importlib.util.module_from_spec(spec)
    sys.modules["pyhip"] = module
    spec.loader.exec_module(module)


def reference(x, w_down, w_up):
    import torch
    x, wd, wu = x.double(), w_down.double(), w_up.double()
    hidden = torch.nn.functional.silu((x @ wd.T) * 0.25)
    gates = torch.sigmoid(hidden @ wu.T).reshape(-1, 4, 2560)
    return (gates * x.reshape(-1, 4, 2560)).mean(dim=1)


def make_inputs(rows, seed):
    import torch
    gen = torch.Generator(device="cuda").manual_seed(seed)
    def randn(*shape):
        return torch.randn(*shape, dtype=torch.bfloat16, device="cuda", generator=gen)
    return randn(rows, 10240), randn(320, 10240) * 0.02, randn(10240, 320) * 0.02


def capture(calls):
    import torch
    for _ in range(2):
        for call in calls:
            call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with warnings.catch_warnings(record=True) as messages:
        warnings.simplefilter("always")
        with torch.cuda.graph(graph):
            for call in calls:
                call()
    for message in messages:
        if "graph is empty" in str(message.message).lower():
            raise RuntimeError("empty graph: launch did not use the capture stream")
        warnings.warn(str(message.message), message.category)
    return graph


def guarded(shape, dtype, device):
    import math
    import torch
    storage = torch.full((math.prod(shape) + 32,), 97, dtype=dtype, device=device)
    return storage[16:-16].view(shape), storage


def assert_close(actual, expected, label):
    import torch
    assert torch.allclose(actual.double(), expected, rtol=1e-2, atol=5e-3), label


def check_rows(rows, wd, wu, pd, pu, seed):
    import torch
    use_checkout_package()
    from pyhip.contrib.flydsl.gr_read import GRReadDecode
    from pyhip.contrib.flydsl.gr_read.common import K, R, H as HS

    reader = GRReadDecode(rows, pd, pu)
    assert reader.w_down.data_ptr() == pd.data_ptr()
    assert reader.w_up.data_ptr() == pu.data_ptr()
    x, sx = guarded((rows, K), torch.bfloat16, pd.device)
    reader.partial, sp = guarded(reader.partial.shape, torch.float32, pd.device)
    reader.output, sy = guarded((rows, HS), torch.bfloat16, pd.device)
    gen = torch.Generator(device=pd.device).manual_seed(seed)
    x.normal_(generator=gen)
    assert_close(reader(x), reference(x, wd, wu), f"T={rows}: eager FP64 mismatch")
    graph = capture([lambda: reader(x)])
    count, maximum = 0, 0.0
    for tail in ("zero", "stale", "nan"):
        x.normal_(generator=gen)
        # Reuse one captured graph while the live prefix shrinks and grows.
        for live in list(range(rows, -1, -1)) + list(range(1, rows + 1)):
            x[:live].normal_(generator=gen)
            if tail == "zero":
                x[live:].zero_()
            elif tail == "nan":
                x[live:].fill_(float("nan"))
            before = x.clone()
            reader.partial.fill_(float("nan"))
            reader.output.fill_(float("nan"))
            graph.replay()
            expected = reference(x[:live], wd, wu)
            actual = reader.output[:live].double()
            label = f"T={rows}, live={live}, tail={tail}"
            assert_close(actual, expected, label)
            if live:
                error = ((actual - expected).abs() / (0.005 + 0.01 * expected.abs())).max().item()
                maximum = max(maximum, error)
            partial = reader.partial.view(4, reader.padded_rows, R)
            assert torch.isfinite(partial[:, :live]).all(), label + ": live scratch"
            assert torch.count_nonzero(partial[:, rows:]) == 0, label + ": internal padding"
            if tail == "zero":
                assert torch.count_nonzero(partial[:, live:]) == 0, label + ": zero scratch tail"
                assert torch.count_nonzero(reader.output[live:]) == 0, label + ": zero output tail"
            elif tail == "stale":
                assert torch.isfinite(partial).all(), label + ": stale scratch"
                assert torch.isfinite(reader.output).all(), label + ": stale output"
            assert torch.allclose(x, before, rtol=0, atol=0, equal_nan=True), label + ": input mutated"
            for storage in (sx, sp, sy):
                assert torch.all(storage[:16] == 97) and torch.all(storage[-16:] == 97), label + ": guard overwritten"
            count += 1
    return {"rows": rows, "replays": count, "max_scaled_error": maximum, "passed": True}


def parse_rows(text):
    rows = int(text)
    if not 1 <= rows <= 32:
        raise argparse.ArgumentTypeError("rows must be in 1..32")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=parse_rows, nargs="+", default=list(range(1, 33)))
    parser.add_argument("--weights", type=int, default=2, help="independent synthetic weight pairs")
    parser.add_argument("--seed", type=int, default=303)
    parser.add_argument("--output", type=Path, help="optional new JSONL file")
    args = parser.parse_args()
    if args.weights < 1:
        parser.error("--weights must be positive")
    if not __debug__:
        parser.error("run without python -O; checks use ordinary asserts")
    import torch
    use_checkout_package()
    from pyhip.contrib.flydsl.gr_read import prepare_weights
    if torch.version.hip is None or not torch.cuda.is_available():
        raise RuntimeError("ROCm PyTorch and a GPU are required")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode(), (args.output.open("x") if args.output else nullcontext()) as log:
        total = 0
        for i in range(args.weights):
            _, wd, wu = make_inputs(1, args.seed + i)
            pd, pu = prepare_weights(wd, wu)
            before_down, before_up = pd.clone(), pu.clone()
            for rows in args.rows:
                result = check_rows(rows, wd, wu, pd, pu, args.seed + i * 10000 + rows)
                assert torch.equal(pd, before_down) and torch.equal(pu, before_up), "packed weights mutated"
                result["weight"] = i
                total += result["replays"]
                if log:
                    log.write(json.dumps(result) + "\n")
                    log.flush()
                print(f"weight={i} T={rows:2}: eager + {result['replays']} graph replays PASS", flush=True)
        print(f"PASS: {args.weights} weight pairs, {len(args.rows)} shapes, {total} graph replays")


if __name__ == "__main__":
    main()


def test_shared_weights_and_decode_devices():
    """Public decode/prefill instances share packing and keep device-local launches."""
    import pytest
    torch = pytest.importorskip("torch")
    if torch.version.hip is None or not torch.cuda.is_available():
        pytest.skip("ROCm required")
    if not torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx942"):
        pytest.skip("gfx942 required")
    use_checkout_package()
    from pyhip.contrib.flydsl.gr_read import GRReadDecode, GRReadPrefill, prepare_weights
    from pyhip.contrib.flydsl.gr_read.common import prepare_weights as shared_prepare

    assert prepare_weights is shared_prepare
    with torch.no_grad(), torch.cuda.device(0):
        x, wd, wu = make_inputs(64, 4917)
        pd, pu = prepare_weights(wd, wu)
        saved_weights = pd.clone(), pu.clone()
        decode, prefill = GRReadDecode(16, pd, pu), GRReadPrefill(64, pd, pu)
        assert decode.w_down.data_ptr() == prefill.w_down.data_ptr() == pd.data_ptr()
        assert decode.w_up.data_ptr() == prefill.w_up.data_ptr() == pu.data_ptr()
        assert decode.partial.dtype == torch.float32 and prefill.partial.dtype == torch.bfloat16
        graph = capture([lambda: decode(x[:16])])
        for state in range(2):
            if state: x.mul_(0.99).add_(0.015625)
            graph.replay()
            assert_close(decode.output, reference(x[:16], wd, wu), "decode shared packing")
            assert_close(prefill(x), reference(x, wd, wu), "prefill shared packing")
        if torch.cuda.device_count() >= 2 and torch.cuda.get_device_properties(1).gcnArchName.startswith("gfx942"):
            second = GRReadDecode(16, pd.to("cuda:1"), pu.to("cuda:1"))
            actual = second(x[:16].to("cuda:1")).to("cuda:0")
            torch.testing.assert_close(actual, decode.output, rtol=0, atol=0)
            assert torch.cuda.current_device() == 0
            with torch.cuda.device(1):
                decode(x[:16])
                assert torch.cuda.current_device() == 1
        assert torch.equal(pd, saved_weights[0]) and torch.equal(pu, saved_weights[1])
