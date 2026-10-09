# Explicit benchmarks

This directory holds standalone operator timing/model-matrix scripts. It is not
part of default pytest collection. Some filenames retain `test_` to minimize
churn; their function arguments are CLI inputs, not pytest fixtures.

- [gemm](gemm/): FP8/blockscale comparisons, standalone linear checks, and trace inputs.
- [moe](moe/): Aiter vs tuned MoE comparison, sorting checks, and trace config.
- [conv](conv/): depthwise convolution correctness and timing CLI.
- [attention](attention/): paged-attention correctness/timing script.
- [gr_read](gr_read/readme.md): decode/prefill performance, frozen baselines, integration examples and Markdown reports.
- [gr_write](gr_write/readme.md): GR write + per-branch RMSNorm prefill timing, gate/apply breakdown and design notes.

Run with the same Python environment used to install PyHIP:

```bash
python benchmarks/moe/bench_tuned_moe.py --help
python benchmarks/moe/bench_tuned_moe.py --list-models
python benchmarks/conv/test_conv_depthwise.py --help
python benchmarks/gemm/test_w8a8_block_fp8_linear.py
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

GRRead's benchmark CLI is [bench_gr_read_compare.py](gr_read/bench_gr_read_compare.py).
The [GR read guide](gr_read/readme.md) includes measured performance, integration
and warmup examples, and `--output` / `--md` report options. Inputs and accuracy
references are shared through `pyhip.testing.gr_read`; the benchmark does not
load repository test scripts. Pytest checks and the compatible
accuracy-then-performance CLI remain in
[test_gr_read.py](../tests/ops/gr_read/test_gr_read.py) (`--check-only` skips timing).
Comparisons that depend on relocated GEMM kernels now live in the opt-in tests
under [tests/ops/gemm](../tests/ops/gemm/).

Benchmarks may allocate large buffers, compile kernels, or require external
tools. There is no new scheduling, process isolation, hardware management, or
automatic tuning framework introduced by this directory move.
