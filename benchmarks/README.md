# Explicit benchmarks

This directory holds standalone operator timing/model-matrix scripts. It is not
part of default pytest collection. Some filenames retain `test_` to minimize
churn; their function arguments are CLI inputs, not pytest fixtures.

- [gemm](gemm/): FP8/blockscale comparisons, standalone linear checks, and trace inputs.
- [moe](moe/): fused MoE CLI, comparison script, MXFP4/sorting checks, and trace config.
- [conv](conv/): depthwise convolution correctness and timing CLI.
- [attention](attention/): paged-attention correctness/timing script.

Run with the same Python environment used to install PyHIP:

```bash
python benchmarks/moe/test_fused_moe.py --help
python benchmarks/moe/cmp_perf_fused_moe.py --help
python benchmarks/moe/bench_tuned_moe.py --list-models
python benchmarks/conv/test_conv_depthwise.py --help
python benchmarks/gemm/test_w8a8_block_fp8_linear.py
```

The MoE comparison script locates its sibling CLI relative to itself and uses
the current Python interpreter. The GEMM trace script points to the relocated
GEMM regression file; its external profiler/decoder requirements are unchanged.

For same-shape Aiter vs autotuned PyHIP MoE comparisons, use
[bench_tuned_moe.py](moe/bench_tuned_moe.py). It shares the seven model presets
with the interactive regression, validates both backends independently, and
exports raw timings without backend-specific shape padding. See the
[MoE benchmark guide](moe/README.md) for the protocol and result statuses.

GRRead deliberately keeps its combined CLI and pytest checks in
[test_gr_read.py](../tests/ops/gr_read/test_gr_read.py). Comparisons that depend on
experimental kernels remain with those kernels, for example
[compare_gemm_950.py](../experiments/gemm/flydsl/compare_gemm_950.py).

Benchmarks may allocate large buffers, compile kernels, or require external
tools. There is no new scheduling, process isolation, hardware management, or
automatic tuning framework introduced by this directory move.