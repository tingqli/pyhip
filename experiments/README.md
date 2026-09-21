# Kernel experiments

This tree preserves prototypes together with their local correctness tests,
benchmark scripts, native sources, analysis tools, figures, and notes. Moving a
group here does not remove its tests or declare its kernel production-ready.

```text
experiments/
  attention/
    flydsl/         MHA, PA 4/8-wave, Flash Attention API, and attention studies
    triton/         Linear-attention / GDN studies and compute_br checks
  gemm/
    flydsl/         GEMM iterations, gfx950 matrix and FP8 comparisons
    gluon/          GEMM variants and compiler-specific studies
  moe/
    flydsl/         A8W4 and eight-wave down-projection groups
    gluon/          MoE GEMM variants
    down/           Standalone down-projection scripts using saved input data
    smoothquant/    INT8/SmoothQuant implementations, HIP sources, and plots
    analysis/       Cost-model scratch work
  codegen/          Explicit compiler/instruction experiments
  elementwise/     Fused sigmoid/multiply/add implementation and tests
  reduction/       Reduction prototypes
  softmax/         Softmax prototypes
```

## Explicit execution only

Default pytest collection is restricted to `tests`. Do not run pytest over this
entire tree: some historical scripts execute GPU work, read saved tensors, or
generate plots at import time. Their code and launch behavior were deliberately
not rewritten during relocation.

Select a known test file or node, for example:

```bash
python -m pytest experiments/elementwise/gluon/test_fused_sigmoid_mul_add.py -q
python -m pytest experiments/moe/flydsl/moe_8w_down/test_blockscaled.py -k test_cli_scope -q
python -m pytest experiments/gemm/flydsl/test_gemm.py -k gemm_950 -q
python experiments/gemm/flydsl/compare_gemm_950.py --help
```

Plain modules with embedded test functions keep their original names. For an
explicit pytest invocation of such a file, override the filename rule:

```bash
python -m pytest experiments/gemm/gluon/gemm_splitk.py -o python_files=*.py -k test_acc -q
```

Experiments retain their own performance selection rules; only the six formal
regression-suite performance functions were newly marked `perf`. MHA still uses
`PYHIP_MHA_PERF` as before. Do not assume a broad experimental invocation is cheap.

Sibling imports support both explicit pytest/importlib execution and existing
direct script entry points. Flash Attention and the PA 4/8-wave groups remain
siblings; the MoE down kernels and reducer remain together. Shared `gen_timing`
now comes from `pyhip.testing.timing`, not an ambiguous top-level `common` module.

Old external URLs and frozen profiler/results references in historical reports
are retained; untracked measurement artifacts were not copied or deleted. Use
the new paths for new executions and keep historical provenance distinct.