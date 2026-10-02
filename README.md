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

### Validate against an Aiter ref

[scripts/validate_aiter.sh](scripts/validate_aiter.sh) accepts an Aiter branch,
tag, or commit and tests it against the current PyHIP working tree:

```bash
# Default suite, then an activated debug shell
bash scripts/validate_aiter.sh main

# A tag and a focused test; pytest arguments follow --
bash scripts/validate_aiter.sh v0.1.12.post1 -- -q -k test_aiter_signature

# Use refs from an existing clone without changing that checkout
bash scripts/validate_aiter.sh --repo /path/to/aiter COMMIT_SHA

# Noninteractive run; optionally choose the outer Python and a new work directory
bash scripts/validate_aiter.sh --python /path/to/python3 \
        --work-dir /path/to/new-run --no-shell main -- -q

# remove test environment
rm -rf -- /root/.cache/pyhip/aiter-envs/run.zDkPok
```

- **Torch is reused**, not downloaded or reinstalled, from `python3` on the
    calling shell's `PATH` (or `--python`). That interpreter must already have
    ROCm Torch; ROCm, its compiler, Git, and Python venv support must also be
    available. Incompatible Torch requirements fail instead of replacing Torch.
- The new venv shares outer site packages as fallbacks, including when the
    outer Python is itself in a venv. This is not a fully isolated dependency
    environment. Compatible installed dependencies may be reused; new installs
    stay in the new venv.
- Aiter is cloned separately, checked out at the resolved commit, and its
    submodules and declared build/runtime dependencies are installed. **FlyDSL's
    version comes only from that Aiter ref**; the runner adds no independent
    FlyDSL version. The ref's Triton installer runs when present. Aiter and the
    current PyHIP checkout are installed editable, including local PyHIP edits.
- Pytest runs from this repository root and honors [pytest.ini](pytest.ini).
    The venv, Aiter checkout, separate compilation caches, activation script,
    and exit-status summary are retained in the printed run directory. By
    default this is a unique directory under `~/.cache/pyhip/aiter-envs`;
    `--work-dir` must name a directory that does not yet exist.
- Success or failure opens an activated **child Bash** when a terminal is
    available. It starts in PyHIP without reading the user's Bash startup file.
    `exit` returns to the original shell without activating the parent. Use the
    printed `source` command to reopen the same environment and caches later.
    `--no-shell` or noninteractive input skips the child shell. After it exits,
    the runner returns the installation/pytest status, not the debug shell's
    status.

Environments are not automatically deleted. Allow disk space for dependency
wheels, submodules, and cold compilation. An old Aiter ref may genuinely be
incompatible with the current Python, Torch, ROCm, GPU, or PyHIP; the script
retains failures for debugging rather than changing tests to pass.

## Using installed kernels

Choose the implementation explicitly. Packaged kernels do not require imports
from repository tests or a test-directory `PYTHONPATH`.

| Operator family | Implementations currently present |
|---|---|
| [GEMM / Linear](src/pyhip/ops/gemm) | Assembly and an ASM/Aiter quantized Linear wrapper |
| [MoE](src/pyhip/ops/moe) | Assembly, FlyDSL, and an Aiter-compatible autotuned API |
| [Attention](src/pyhip/ops/attention) | Assembly paged attention |
| [Convolution](src/pyhip/ops/conv) | Shared FP16/BF16 HIP depthwise Conv3D implementations |
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
# Run the curated default regression set after broad code changes
python3 -m pytest -q

# Inspect the default test collection
python3 -m pytest --collect-only -q

# Run a selected operator regression suite
python -m pytest tests/ops/gemm/test_cdna4.py -q

# Opt in to performance-only tests
python -m pytest tests/ops/moe/test_moe.py -m perf -s

# Inspect a standalone benchmark's options
python benchmarks/moe/bench_tuned_moe.py --help

# Run an experimental check explicitly
python -m pytest experiments/gemm/flydsl/test_gemm.py -k gemm_950 -q
```

[pytest.ini](pytest.ini) lists the existing modules in the default regression
set, uses importlib mode, and excludes the `perf` marker. Passing `tests`
explicitly overrides that list and includes historical diagnostics/failures.
No existing test code is changed for selection. The default set was validated
on gfx950; many tests require a GPU, including some during collection.
See [tests/README.md](tests/README.md) for coverage, exclusions, and hardware limits.

- [tests/README.md](tests/README.md): regression entry points and collection rules.
- [benchmarks/README.md](benchmarks/README.md): standalone timing scripts.
- [experiments/README.md](experiments/README.md): opt-in experimental validation.

Do not collect the entire experimental tree: some scripts perform GPU work or
load saved input data at import time. GRRead checks accuracy before printing
decode and prefill performance tables via [test_gr_read.py](tests/ops/gr_read/test_gr_read.py)
(`--check-only` skips timing). Its standalone benchmark CLI is
[bench_gr_read_compare.py](benchmarks/gr_read/bench_gr_read_compare.py); performance
data, integration examples and report commands are in the
[GR read benchmark guide](benchmarks/gr_read/readme.md).

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
