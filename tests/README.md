# Regression tests

`tests` contains the default regression suite, organized by what is tested:

```text
tests/
  codegen/asm/       Assembly JIT, IR, instruction, and layout regression tests
  ops/
    conv/           Installed pointwise convolution
    gemm/           Installed ASM GEMM variants and split-K
    gr_read/        GRRead correctness tests and its existing benchmark CLI
    mlp/            Installed Gluon MLP
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

Six explicitly performance-only test functions in the GEMM, MLP, and MoE suites
are marked `perf` and excluded by default. `-m perf` opts in; `-m ""` removes the
default marker filter. Correctness tests that also record timing remain intact;
the marker does not split or rewrite existing test bodies.

The inherited shell runners remain beside their suites. They still clear the
existing PyHIP JIT cache; use direct pytest commands above when that is unwanted.

## Relocation map

| Former group | Current home |
|---|---|
| Core tests | [codegen/asm](codegen/asm/) |
| Parameterized GEMM regressions | [ops/gemm](ops/gemm/) |
| Pointwise convolution | [test_conv_pointwise.py](ops/conv/test_conv_pointwise.py) |
| GRRead, README, and local collection config | [ops/gr_read](ops/gr_read/) |
| Cross-backend MoE regression suite | [test_moe.py](ops/moe/test_moe.py) |
| Gluon MLP regression suite | [test_fused_mlp.py](ops/mlp/test_fused_mlp.py) |
| CLI-only operator timing/model matrices | [benchmarks](../benchmarks/README.md) |
| Kernel prototypes, local checks, and profiler groups | [experiments](../experiments/README.md) |
| Gluon debug-buffer timing helper | [pyhip.testing.timing](../src/pyhip/testing/timing.py) |

The benchmark scripts that take ordinary function arguments without pytest
fixtures remain executable scripts, not newly fabricated parametrized tests.
GRRead keeps its single-file CLI/test design; there is no new shared runner.
No regression or experimental test body was deleted to narrow collection.

## Existing limitations

This organization does not repair historical kernels or relax their numerical
checks. For example, the MoE mixed-dispatch test still refers to the already
removed `gemm2_8x1_k192` / `gemm2_8x1_k320` modules. Those failures remain visible
when selected. Some experimental Gluon scripts require an older compiler API;
Triton linear-attention experiments need the SGLang FLA modules.

New reusable kernels belong under `src/pyhip/ops`; tests should consume the
installed package. Small instruction/layout probes can live in a regression
test. Larger prototypes stay with their own validation scripts under experiments
until they are deliberately promoted to the installed operator library.