# PyHIP

**AMDGPU kernels, development tools, and evaluation in one repository.**

PyHIP is a workspace for developing, integrating, and comparing GPU kernels
written in **HIP, FlyDSL, Gluon, or Python-generated assembly**. Reusable
implementations are shipped as Python modules under `pyhip.ops` and can be
imported directly after installation.

The repository separates three responsibilities:

- **Use kernels:** import an implementation or an existing operator wrapper.
- **Develop kernels:** use the language and compiler suited to the problem;
    assembly JIT is an optional tool, not the center of every implementation.
- **Evaluate kernels:** keep correctness regressions, explicit benchmarks, and
    experimental implementations separate, with reusable measurement helpers.

There is no mandatory backend class hierarchy, common compiler, or universal
"fastest kernel" dispatcher. Each implementation retains its own interface and
supported shapes, layouts, dtypes, and GPU architectures.

## Installation

Use a Linux environment with an AMD GPU, ROCm, and a matching ROCm PyTorch build.
A [ROCm PyTorch container](https://rocm.docs.amd.com/projects/install-on-linux/en/develop/install/3rd-party/pytorch-install.html)
is a convenient starting point.

```bash
# Install from GitHub
python -m pip install git+https://github.com/tingqli/pyhip.git

# Or, from a checkout, install the current branch for development
python -m pip install -e .
```

Install the dependencies required by the selected implementation in the same
environment: ROCm toolchain for HIP/assembly, FlyDSL for FlyDSL kernels,
Triton with Gluon support for optional convolution kernels, and Aiter or other
integration dependencies where used. The base package does not install all GPU
backends.

`import pyhip` does not import the optional GPU backends or initialize a GPU.
Importing a specific implementation may load its dependencies. First calls can
compile kernels; installation does not precompile all architectures or remove
the corresponding compiler requirements.

For an existing editable installation from the old layout, reinstall it so it
points to the package under [src/pyhip](src/pyhip).

## Using installed kernels

Choose the implementation explicitly. Packaged kernels do not require imports
from repository tests or a test-directory `PYTHONPATH`.

| Operator family | Implementations currently present |
|---|---|
| [GEMM / Linear](src/pyhip/ops/gemm) | Assembly and an ASM/Aiter quantized Linear wrapper |
| [MoE](src/pyhip/ops/moe) | Assembly, FlyDSL, and an Aiter-compatible autotuned API |
| [Attention](src/pyhip/ops/attention) | Assembly paged attention |
| [Convolution](src/pyhip/ops/conv) | HIP depthwise and assembly/Gluon pointwise implementations |
| [GRRead](src/pyhip/ops/gr_read) | FlyDSL down/up projections |

Availability in this table does not imply support for every GPU or input shape.
Check the implementation and its associated tests for the required contract.

Quantized Linear accepts `method="auto"` or `"jit"` for ASM, and `"aiter"` for
Aiter. MoE uses `pyhip.ops.moe.tuned_moe.fused_moe`, with Aiter's signature;
supported inputs are autotuned across validated Aiter, ASM and FlyDSL candidates.
Other API features are forwarded to Aiter. Fixed kernel tests use
`pyhip.testing.moe.make_moe_runner`, without autotuning or fallback. The old
MoE wrappers and their `method="auto"/"jit"` interface have been removed.
Optional Gluon convolution implementations remain available.

For example, this BF16 grouped pointwise convolution uses the Gluon path:

```python
import torch
from pyhip.ops.conv.conv_pointwise import conv_pointwise

x = torch.randn((1, 16, 3, 4, 64), device="cuda", dtype=torch.bfloat16)
weight = torch.randn((4, 4, 4), device="cuda", dtype=torch.bfloat16)
bias = torch.randn((16,), device="cuda", dtype=torch.bfloat16)

y = conv_pointwise(x, weight, bias, groups=4, use_gluon=True)
assert y.shape == x.shape
```

Other entry points include `pyhip.ops.moe.tuned_moe.fused_moe`,
`pyhip.ops.moe.asm.moe.moe_2stage_splitk`, and
`pyhip.ops.moe.flydsl.moe_gemm_2stage.compile_moe_gemm1`. Some are complete
Tensor operators; others are low-level kernels or launcher factories. Refer to
their signatures rather than assuming identical call conventions.

## Repository layout

```text
src/pyhip/                  # Installed Python package
├── ops/                    # Operators, grouped by operation and backend
│   ├── gemm/               # asm/ and existing wrappers
│   ├── moe/                # asm/, flydsl/, autotuned MoE API
│   ├── attention/          # asm/
│   ├── conv/               # Wrappers and packaged hip/ sources
│   └── gr_read/flydsl/      # Down/up projections
├── codegen/                # Shared asm/ and flydsl/ authoring tools
├── runtime/                # HIP compilation, code-object loading, and launch
├── testing/                # Timing, independent references, and trace helpers
└── tools/                  # Explicit code-inspection and hardware-probing tools

tests/                      # Default correctness/regression suite
├── codegen/asm/            # Assembly JIT and instruction tests (includes GPU work)
└── ops/                    # Tests of installed operators
benchmarks/                 # Explicit operator timing and model-matrix scripts
experiments/                # Prototypes with their own tests, benchmarks, and notes
docs/                       # Usage, debugging, and optimization documentation
archive/                    # Historical code and reference material
.github/skills/             # Reusable development and validation guidance
```

**Dependency direction:** tests and benchmarks consume the installed package;
the installed package must not depend on them. Experimental groups can contain
their own kernels without becoming part of the published API.

Existing multi-backend wrappers remain at the operation level. Native sources
are packaged with their wrappers; compilation outputs belong in user caches,
not the installation directory. Package discovery and resource inclusion are
defined in [pyproject.toml](pyproject.toml).

## Testing and benchmarking

Run from the repository root using the environment in which PyHIP is installed.

```bash
# Inspect the default test collection
python -m pytest --collect-only -q

# Run a selected operator regression suite
python -m pytest tests/ops/gemm/test_cdna4.py -q

# Opt in to performance-only tests
python -m pytest tests/ops/moe/test_moe.py -m perf -s

# Inspect a standalone benchmark's options
python benchmarks/moe/bench_tuned_moe.py --help

# Run an experimental check explicitly
python -m pytest experiments/gemm/flydsl/test_gemm.py -k gemm_950 -q
```

[pytest.ini](pytest.ini) limits default collection to test-named Python modules
under [tests](tests), uses importlib mode, and excludes the `perf` marker.
Many tests require a GPU, including some during collection. Architecture and
dependency constraints still apply; the directory layout is not a guarantee
that all historical tests pass in every environment.

- [tests/README.md](tests/README.md): regression entry points and collection rules.
- [benchmarks/README.md](benchmarks/README.md): standalone timing scripts.
- [experiments/README.md](experiments/README.md): opt-in experimental validation.

Do not collect the entire experimental tree: some scripts perform GPU work or
load saved input data at import time. GRRead deliberately keeps its combined
test/benchmark CLI in [test_gr_read.py](tests/ops/gr_read/test_gr_read.py).

Shared helpers are available from `pyhip.testing`, including `calc_diff`,
`cudaPerf`, and `run_perftest`. The latter returns `(output, latency_us)` and
handles warmup and tensor-buffer rotation; correctness checks and the choice of
what is inside the timed call remain the caller's responsibility.

## Developing and integrating kernels

1. Keep a new prototype with its local checks and profiling scripts in the
     appropriate experimental operation/backend group.
2. When integrating it for direct use, put the kernel and wrapper under the
     corresponding `pyhip.ops` module. Document shape, dtype, layout, architecture,
     output-buffer, and weight-preparation requirements.
3. Use backend-specific helpers from `pyhip.codegen` or the backend's own
     compiler/runtime. Do not route FlyDSL or Triton through the assembly JIT.
4. Add correctness coverage under the matching operator test group and an
     explicit benchmark where useful. Keep reference calculations and compilation
     outside the intended timing region, and state whether preparation is timed.

Preserve existing operator interfaces when reorganizing code. Shared helpers
should have a concrete use across implementations; no plugin registry or new
framework is required just to add a kernel.

Useful starting points: [FlyDSL examples](docs/learn_flydsl),
[FlyDSL debugging](docs/debug-flydsl.md),
[GPU profiling](docs/profile-gpu.md), and
[development/validation guidance](.github/skills/README.md).

### Migrating older imports

The old `pyhip.contrib.*` and `pyhip.core.*` paths have moved to operator,
codegen, and runtime modules. Existing `pyhip.jit`, `pyhip.JIT`, `pyhip.module`,
timing helpers, and lazy `pyhip.fly` / `pyhip.printv` remain available.
Use `pyhip.codegen.flydsl` for shared FlyDSL authoring helpers.

## Appendix: low-level kernel authoring

### Assembly JIT

The original assembly toolkit is retained in
[src/pyhip/codegen/asm](src/pyhip/codegen/asm):

- `@pyhip.jit()` defines a kernel; its first argument, `J`, generates instructions.
- `J.gpr` allocates logical SGPR/VGPR/AccVGPR values with automatic lifetime and
    physical-register allocation, without automatic spilling.
- `J.If` / `J.While` describe control flow; instruction methods expose explicit
    loads, stores, MFMA, and wait counts.
- String-annotated parameters describe runtime HIP arguments; unannotated
    parameters specialize the kernel at compile time.

The pipeline generates assembly IR, applies optimization/register-allocation
passes, and uses ROCm to build and load a code object. Existing compiled-artifact
caching is retained (`PYHIP_CACHE_DIR`, default `~/.pyhip`).

See [basic examples](tests/codegen/asm/test_basic.py),
[control flow](tests/codegen/asm/test_jump.py), and
[MFMA tests](tests/codegen/asm/test_mfma_minimal.py) for executable examples.

### HIP sources and code objects

`pyhip.module` compiles HIP C++ or assembly sources and wraps kernel launches
through the HIP runtime. Relative source paths are resolved against the calling
Python module. See the [depthwise wrapper](src/pyhip/ops/conv/conv_depthwise.py),
its [packaged HIP sources](src/pyhip/ops/conv/hip), and the
[runtime implementation](src/pyhip/runtime/hiptools.py).

`python -m pyhip` retains the embedded-Python HIP/Markdown runner.
`python -m pyhip.tools.exts` extracts assembly from trace output;
`python -m pyhip.tools.probe` runs hardware probes and is not part of the default
test suite.



