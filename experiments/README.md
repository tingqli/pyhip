# Kernel experiments

This tree preserves prototypes together with their local correctness tests,
benchmark scripts, native sources, analysis tools, figures, and notes. Moving a
group here does not remove its tests or declare its kernel production-ready.

```text
experiments/
  attention/
    flydsl/         MHA, PA 4/8-wave, Flash Attention API, and attention studies
  gemm/
    flydsl/         GEMM iterations, gfx950 matrix and FP8 comparisons
  moe/
    flydsl/         A8W4 and eight-wave down-projection groups
    down/           Standalone down-projection scripts using saved input data
    smoothquant/    INT8/SmoothQuant implementations, HIP sources, and plots
    analysis/       Cost-model scratch work
  codegen/flydsl/   Explicit compiler/instruction experiments
```

## Explicit execution only

Default pytest collection is restricted to `tests`. Do not run pytest over this
entire tree: some historical scripts execute GPU work, read saved tensors, or
generate plots at import time. Their code and launch behavior were deliberately
not rewritten during relocation.

Select a known test file or node, for example:

```bash
python -m pytest experiments/moe/flydsl/moe_8w_down/test_blockscaled.py -k test_cli_scope -q
python -m pytest experiments/gemm/flydsl/test_gemm.py -k gemm_950 -q
python experiments/gemm/flydsl/compare_gemm_950.py --help
```

Some plain modules provide their own script entry points rather than pytest
fixtures. Run those directly, for example:

```bash
python experiments/gemm/flydsl/gemm4.py
```

Experiments retain their own performance selection rules; formal
regression-suite performance-only functions are marked `perf`. MHA still uses
`PYHIP_MHA_PERF` as before. Do not assume a broad experimental invocation is cheap.

The gfx950 8-wave block-scale FP8 and 4-wave MXFP8/MXFP4 implementations now
live in [pyhip.ops.gemm.flydsl](../src/pyhip/ops/gemm/flydsl/), with correctness
and opt-in performance tests in [tests/ops/gemm](../tests/ops/gemm/). Their
remaining comparison script imports the library kernel, not a sibling test.

Sibling imports support both explicit pytest/importlib execution and existing
direct script entry points. Flash Attention and the PA 4/8-wave groups remain
siblings; the MoE down kernels and reducer remain together. Shared `gen_timing`
now comes from `pyhip.testing.timing`, not an ambiguous top-level `common` module.

Old external URLs and frozen profiler/results references in historical reports
are retained; untracked measurement artifacts were not copied or deleted. Use
the new paths for new executions and keep historical provenance distinct.