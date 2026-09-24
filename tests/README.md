# Regression tests

`tests` contains regression tests and historical diagnostics, organized by what is tested:

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
The default collection is an explicit file list in [pytest.ini](../pytest.ini),
not every test under this directory. Existing test source, parameter matrices,
assertions, tolerances, and skip rules are unchanged.

## Running

Install PyHIP in the selected ROCm Python environment and run from the repository
root. [pytest.ini](../pytest.ini) selects the curated regression files,
`test_*.py`, importlib mode, and `not perf`.
The allocator environment setup in [conftest.py](../conftest.py) applies both
here and to explicitly selected experiments, as it did before the move.

```bash
# Default regression set after broad code changes
python3 -m pytest -q

# Inspect exactly what the default set will run
python3 -m pytest --collect-only -q

# Explicit paths override the file list; historical failures remain visible
python3 -m pytest tests -q
python3 -m pytest tests/codegen/asm/test_transpose.py -q

# Performance remains opt-in
python3 -m pytest tests/ops/moe/test_moe.py -m perf -s
```

**Use bare `pytest` / `python3 -m pytest` for the default set.** Passing `tests`
explicitly requests the whole directory and bypasses the curated file list.
`-m perf` changes only marker selection, not the file list. To include every
historical test and performance case, use an explicit directory and `-m ""`;
that is not the routine regression command and can allocate large buffers.

## Default coverage

The list includes 20 existing modules; it does not copy tests into a new runner
or add collection hooks:

- **ASM codegen/runtime:** basic memory operations, scalar/vector expressions,
  integer division, control flow, SIMT, CSE/DCE/DSE, debug logging, LDS tensors,
  workgroup loads, swizzle, reduction, softmax, and the score/value MFMA check.
- **Installed GEMM:** all existing correctness cases in
  [test_cdna4.py](ops/gemm/test_cdna4.py) and
  [test_a4w4_mxfp4.py](ops/gemm/test_a4w4_mxfp4.py).
- **MoE:** all non-`perf` cases in [test_moe.py](ops/moe/test_moe.py) and
  [test_tuned_moe.py](ops/moe/test_tuned_moe.py), including the original full
  split-K batch sweep, fixed ASM/FlyDSL paths, independent references, dispatch,
  autotune/cache, current-stream, and graph replay regressions.
- **GRRead:** [test_gr_read.py](ops/gr_read/test_gr_read.py), retaining its
  existing gfx942 gate and full correctness matrix.

Validated on **gfx950 with ROCm Torch, Aiter, and FlyDSL** on 2026-09-24:
**2433 passed, 65 skipped, 170 performance cases deselected**, in about four
minutes with the existing compilation caches. The skips were 44 gfx942-only
GRRead cases and 21 unsupported MoE tile combinations. Cold compilation can take
longer. This is not evidence of a passing gfx942 or CPU-only suite: some unchanged
tests initialize the GPU during collection and the CDNA4/MXFP4 GEMM tests have no
architecture skip. Dependency and architecture requirements still apply.

This set is a routine regression baseline, not a guarantee for every operator,
shape, architecture, or performance change. Run the affected module's additional
tests and benchmarks when changing code outside its coverage. New test modules
must be explicitly added to the list after checking assertions, dependencies,
runtime, and supported hardware.

## Not in the default set

These tests remain unchanged and runnable by explicit path. They are not marked
`xfail`, and their failures have not been converted into skips.

| Existing module | Reason / follow-up |
|---|---|
| [test_mfma.py](codegen/asm/test_mfma.py) | Six of eight template/layout cases failed numerical checks in the gfx950 baseline. Keep the failures available for a dedicated investigation; the default set still covers MFMA through other codegen/GEMM/MoE tests. |
| [test_transpose.py](codegen/asm/test_transpose.py) | Five of eight cases failed, including BF16 8×4 and FP8 cases. Per-lane transpose coverage is incomplete until those failures are resolved. |
| [test_4wave_cdna4_slicing.py](ops/gemm/test_4wave_cdna4_slicing.py) | The original call omits `PROFILE_CYCLE` and fails with `IndexError`. A temporary diagnosis also exposed a non-power-of-two swizzle divisor for N1536; no source fix is retained here. |
| [test_mfma_minimal.py](codegen/asm/test_mfma_minimal.py) | Numerical mismatches only print diagnostics; pytest does not fail. Not counted as correctness coverage. |
| [test_execmask.py](codegen/asm/test_execmask.py) | Diagnostic probes with no output assertions; SIMT/control-flow checks remain in the default set. |
| [test_gemm.py](codegen/asm/test_gemm.py) | Large benchmark-style matrix with 40 tensor copies and repeated timing. |
| [test_sum.py](codegen/asm/test_sum.py) | Memory-bandwidth experiment allocating about 20 GB of input copies and running timing loops. |
| [test_jit_gemm_splitk.py](ops/gemm/test_jit_gemm_splitk.py) | A single correctness node hides hundreds of batch cases across three precisions and allocates 32 copies. Run explicitly for changes to this standalone GEMM path. |
| [test_conv_pointwise.py](ops/conv/test_conv_pointwise.py) | Large ASM/Gluon timing comparison that prints diff without asserting it; the Gluon layout probe also looks like a pytest test. The default set does not claim pointwise-convolution correctness coverage. |

Explicitly performance-only test functions in the GEMM and MoE suites
are marked `perf` and excluded by default. `-m perf` opts in; `-m ""` removes the
default marker filter within the selected files. MoE separates fixed-kernel correctness from timing;
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