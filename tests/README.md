# Regression tests

`tests` contains the default regression suite, organized by what is tested:

```text
tests/
  codegen/asm/       Assembly JIT, IR, instruction, and layout regression tests
  ops/
    conv/           Installed pointwise convolution
    gemm/           Installed ASM GEMM variants and split-K
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

Explicitly performance-only test functions in the GEMM and MoE suites
are marked `perf` and excluded by default. `-m perf` opts in; `-m ""` removes the
default marker filter. MoE now separates fixed-kernel correctness from timing;
see [MoE pytest usage](ops/moe/README.md) for individual configurations and node IDs.

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
| CLI-only operator timing/model matrices | [benchmarks](../benchmarks/README.md) |
| Kernel prototypes, local checks, and profiler groups | [experiments](../experiments/README.md) |
| Gluon debug-buffer timing helper | [pyhip.testing.timing](../src/pyhip/testing/timing.py) |

The benchmark scripts that take ordinary function arguments without pytest
fixtures remain executable scripts, not newly fabricated parametrized tests.
GRRead keeps its existing single-file CLI/test design.

## Existing limitations

This organization does not repair historical kernels or relax their numerical
checks. MoE fixed-kernel tests consume the installed launchers and shared
[input/reference helpers](../src/pyhip/testing/moe.py); they do not depend on
autotune candidate selection. Some other suites still require GPU initialization
during collection. Optional experiments retain their own dependencies.

New reusable kernels belong under `src/pyhip/ops`; tests should consume the
installed package. Small instruction/layout probes can live in a regression
test. Larger prototypes stay with their own validation scripts under experiments
until they are deliberately promoted to the installed operator library.