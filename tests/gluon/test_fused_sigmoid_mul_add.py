"""
Kernel: final_hidden_states += sigmoid(gate) * shared_output
Shapes: gate [num_tokens, 1], shared_output / final_hidden_states [num_tokens, hidden_size]
bench perf
python -m pytest test_fused_sigmoid_mul_add.py -v -k benchmark -s
"""

import os
import sys

import pytest
import torch

_tests_dir = os.path.dirname(os.path.abspath(__file__))
if _tests_dir not in sys.path:
    sys.path.insert(0, _tests_dir)

from fused_sigmoid_mul_add_gluon import fused_sigmoid_mul_add_gluon

# Only these shapes: accuracy + benchmark
CASES = [
    pytest.param(1, 4096, torch.bfloat16, id="1x4096_bf16"),
    pytest.param(2, 4096, torch.bfloat16, id="2x4096_bf16"),
    pytest.param(8000, 4096, torch.bfloat16, id="8000x4096_bf16"),
]


def _ref(gate: torch.Tensor, shared: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    return out + torch.sigmoid(gate) * shared


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("num_tokens, hidden_size, dtype", CASES)
def test_accuracy(num_tokens, hidden_size, dtype):
    """Kernel output vs PyTorch reference."""
    device = torch.device("cuda", 0)
    gate = torch.randn(num_tokens, 1, device=device, dtype=dtype)
    shared = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)
    out = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)

    ref = _ref(gate, shared, out)
    out_kernel = out.clone()
    fused_sigmoid_mul_add_gluon(gate, shared, out_kernel)

    torch.testing.assert_close(out_kernel, ref, rtol=2e-2, atol=2e-2)


def _bench(run_fn, num_tokens: int, hidden_size: int, dtype: torch.dtype, warmup: int = 10, repeat: int = 100):
    for _ in range(warmup):
        run_fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        run_fn()
    end.record()
    torch.cuda.synchronize()
    mean_ms = start.elapsed_time(end) / repeat
    es = torch.empty(0, dtype=dtype).element_size()
    bytes_run = (num_tokens + num_tokens * hidden_size + 2 * num_tokens * hidden_size) * es
    gbps = (bytes_run / 1e9) / (mean_ms / 1000) if mean_ms > 0 else 0
    return mean_ms, gbps


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("num_tokens, hidden_size, dtype", CASES)
def test_benchmark(num_tokens, hidden_size, dtype):
    """Fused kernel vs PyTorch ref: time and GB/s."""
    device = torch.device("cuda", 0)
    gate = torch.randn(num_tokens, 1, device=device, dtype=dtype)
    shared = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)
    z = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)

    def run_kernel():
        out = z.clone()
        fused_sigmoid_mul_add_gluon(gate, shared, out)

    def run_ref():
        _ref(gate, shared, z)

    fused_ms, fused_gbps = _bench(run_kernel, num_tokens, hidden_size, dtype)
    ref_ms, ref_gbps = _bench(run_ref, num_tokens, hidden_size, dtype)
    speedup = ref_ms / fused_ms if fused_ms > 0 else 0.0

    print(
        f"\n[benchmark] {num_tokens}x{hidden_size} {dtype}\n"
        f"  fused: {fused_ms*1000:.2f} us/iter, {fused_gbps:.2f} GB/s\n"
        f"  ref:   {ref_ms*1000:.2f} us/iter, {ref_gbps:.2f} GB/s  speedup {speedup:.2f}x"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
