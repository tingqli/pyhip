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
python benchmarks/conv/test_conv_depthwise.py --help
python benchmarks/gemm/test_w8a8_block_fp8_linear.py
```

The MoE comparison script locates its sibling CLI relative to itself and uses
the current Python interpreter. The GEMM trace script points to the relocated
GEMM regression file; its external profiler/decoder requirements are unchanged.

GRRead keeps its benchmark CLI in
[bench_gr_read_compare.py](../tests/ops/gr_read/bench_gr_read_compare.py), beside
the correctness CLI and pytest checks in
[test_gr_read.py](../tests/ops/gr_read/test_gr_read.py). Comparisons that depend on
relocated GEMM kernels now live in the opt-in tests under
[tests/ops/gemm](../tests/ops/gemm/).

Benchmarks may allocate large buffers, compile kernels, or require external
tools. There is no new scheduling, process isolation, hardware management, or
automatic tuning framework introduced by this directory move.
