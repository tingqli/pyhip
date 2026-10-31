# Explicit benchmarks

This directory holds standalone operator timing/model-matrix scripts. It is not
part of default pytest collection. Some filenames retain `test_` to minimize
churn; their function arguments are CLI inputs, not pytest fixtures.

- [gemm](gemm/): FP8/blockscale comparisons, standalone linear checks, and trace inputs.
- [moe](moe/): Aiter vs tuned MoE comparison, sorting checks, and trace config.
- [conv](conv/): depthwise convolution correctness and timing CLI.
- [attention](attention/): paged-attention correctness/timing script.

Run with the same Python environment used to install PyHIP:

```bash
python benchmarks/moe/bench_tuned_moe.py --help
python benchmarks/moe/bench_tuned_moe.py --list-models
python benchmarks/conv/test_conv_depthwise.py --help
python benchmarks/gemm/test_w8a8_block_fp8_linear.py
python benchmarks/gemm/tune_gemm_a8w4.py --help
```

The MoE comparison runs both APIs in one process. Fixed BF16 8-wave and MXFP4
kernel checks live in the [MoE pytest suite](../tests/ops/moe/README.md), with
explicit `perf` groups for large shapes. The GEMM trace script's external
profiler/decoder requirements are unchanged.

For same-shape Aiter vs autotuned PyHIP MoE comparisons, use
[bench_tuned_moe.py](moe/bench_tuned_moe.py). It shares the seven model presets
with the fixed-kernel pytest suite, validates both backends independently, and
exports raw timings without backend-specific shape padding. See the
[MoE benchmark guide](moe/README.md) for the protocol and result statuses.

For MXFP8 x MXFP4 GEMM, [tune_gemm_a8w4.py](gemm/tune_gemm_a8w4.py) uses the
same runtime autotuner as the public API, supports JSON config export/reuse,
and compares the best Aiter candidate with automatic Aiter/PyHIP dispatch.
See the [A8W4 GEMM guide](../src/pyhip/ops/gemm/README.md) for layout constraints,
cache controls and the rotating-buffer timing protocol.

GRRead keeps its benchmark CLI in
[bench_gr_read_compare.py](../tests/ops/gr_read/bench_gr_read_compare.py), beside
the pytest checks and default accuracy-then-performance CLI in
[test_gr_read.py](../tests/ops/gr_read/test_gr_read.py) (`--check-only` skips timing).
Comparisons that depend on relocated GEMM kernels now live in the opt-in tests
under [tests/ops/gemm](../tests/ops/gemm/).

Benchmarks may allocate large buffers, compile kernels, or require external
tools. There is no new scheduling, process isolation, hardware management, or
automatic tuning framework introduced by this directory move.
