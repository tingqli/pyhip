# Regression tests

`tests` contains the default regression suite, organized by what is tested:

```text
tests/
  codegen/asm/       Assembly JIT, IR, instruction, and layout regression tests
  ops/
    conv/           Installed pointwise convolution
    gemm/           Installed ASM/FlyDSL GEMM variants and split-K
    gr_read/        GRRead correctness tests and its existing benchmark CLI
    moe/            Installed MoE implementations and cross-backend tests
```

Do not infer CPU-only behavior from `codegen`: these tests launch GPU kernels.
GRRead requires gfx942; other cases have their existing architecture gates.

## Running

Install PyHIP in the selected ROCm Python environment and run from the repository
root. [pytest.ini](../pytest.ini) selects `tests`, `test_*.py`, and importlib mode.
The allocator environment setup in [conftest.py](../conftest.py) applies both
here and to explicitly selected experiments, as it did before the move.

```bash
python -m pytest --collect-only -q
python -m pytest tests/codegen/asm -q
python -m pytest tests/ops/gemm/test_cdna4.py -q
python -m pytest tests/ops/moe/test_moe.py -m perf -s
```

Explicitly performance-only test functions in the GEMM, MLP, and MoE suites
are marked `perf` and excluded by default. `-m perf` opts in; `-m ""` removes the
default marker filter. Correctness tests that also record timing remain intact;
the marker does not split or rewrite existing test bodies.

The FlyDSL [8-wave block-scale FP8](ops/gemm/test_gemm_fp8_blockscale_8w.py)
and [4-wave MXFP8/MXFP4](ops/gemm/test_gemm_mxfp8_4w.py) suites import kernels
from [pyhip.ops.gemm.flydsl](../src/pyhip/ops/gemm/flydsl/). Their GPU tests
require ROCm CDNA4 (`gfx950`) and skip on other devices. The kernel factories
also reject non-CDNA4 compilation targets; explicit offline `gfx950` targets
remain supported. Factories cache launchers by compilation target and static
configuration, expose `cache_info()` / `cache_clear()`, and validate the target
even on cache hits. MX quantization and weight-preshuffle cases additionally
require AIter. Select either file directly for correctness, or add `-m perf -s`
to run its rotating-buffer benchmarks.

The inherited shell runners remain beside their suites. They still clear the
existing PyHIP JIT cache; use direct pytest commands above when that is unwanted.

## Relocation map

| Former group | Current home |
|---|---|
| Core tests | [codegen/asm](codegen/asm/) |
| Parameterized GEMM regressions | [ops/gemm](ops/gemm/) |
| Experimental gfx950 8-wave block-scale / 4-wave MXFP8 GEMM | [kernels](../src/pyhip/ops/gemm/flydsl/) and [tests](ops/gemm/) |
| Pointwise convolution | [test_conv_pointwise.py](ops/conv/test_conv_pointwise.py) |
| GRRead, README, and local collection config | [ops/gr_read](ops/gr_read/) |
| Cross-backend MoE regression suite | [test_moe.py](ops/moe/test_moe.py) |
| CLI-only operator timing/model matrices | [benchmarks](../benchmarks/README.md) |
| Kernel prototypes, local checks, and profiler groups | [experiments](../experiments/README.md) |
| Gluon debug-buffer timing helper | [pyhip.testing.timing](../src/pyhip/testing/timing.py) |

The benchmark scripts that take ordinary function arguments without pytest
fixtures remain executable scripts, not newly fabricated parametrized tests.
GRRead uses [test_gr_read.py](ops/gr_read/test_gr_read.py) for correctness and
[bench_gr_read_compare.py](ops/gr_read/bench_gr_read_compare.py) for timing.
Both cover decode and prefill; the benchmark prints a separate table for each.

## Existing limitations

This organization does not repair historical kernels or relax their numerical
checks. For example, the MoE mixed-dispatch test still refers to the already
removed `gemm2_8x1_k192` / `gemm2_8x1_k320` modules. Those failures remain visible
when selected. Optional Gluon convolution tests require a compatible Triton
installation with Gluon support.

New reusable kernels belong under `src/pyhip/ops`; tests should consume the
installed package. Small instruction/layout probes can live in a regression
test. Larger prototypes stay with their own validation scripts under experiments
until they are deliberately promoted to the installed operator library.
