---
name: flydsl-host-launch
description: The required way to write the host side of FlyDSL kernels in pyhip (a8w4 included) — `@flydsl_kernel` from `pyhip.contrib.flydsl.host_launch`, a Triton-style kernel object called as `kern(*runtime_args, **compile_key)` that replaces the `functools.cache` compile function + `_run_compiled(launch, x.data_ptr(), ..., torch.cuda.current_stream())` boilerplate; gives compile-key normalization, heuristics, per-device launchers, raw-stream launches, CUDA-graph safety, preload and a CHECK mode. Use whenever adding a new FlyDSL kernel / op, writing or rewriting a compile function that returns a `@flyc.jit` launcher, a torch-facing wrapper around a FlyDSL kernel, or touching `_run_compiled` / `data_ptr()` argument lists / `torch.cuda.current_stream()` launches (新增 FlyDSL kernel / 写 host 封装 / launch 封装 / 包装成 torch 算子 / 编译函数 / 迁移 _run_compiled).
---

# FlyDSL host wrappers: `@flydsl_kernel`

The module is `src/contrib/flydsl/host_launch.py`; its docstring is the API reference. Tests
are in `tests/flydsl/test_host_launch.py`.

Use it for **every new FlyDSL kernel in pyhip**, a8w4 included. Also use it when a task
already rewrites an existing wrapper (section 5). Kernel-side rules (disk-cache key,
control flow, LDS, ...) are in the `flydsl-gfx942-pitfalls` skill.

## 1. The pattern

This example is verified bit-exact against `a8w4/_vendor/a8w4_ops.channel_exponents`. It is
written for a8w4 (the `.._vendor` imports); elsewhere, use
`fx.inttoptr(fx.PointerType.get(dtype.ir_type, fx.AddressSpace.Global, align), fx.Int64(addr))`
as the pointer helper.

```python
import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import gpu, range_constexpr

from pyhip.contrib.flydsl.host_launch import cdiv, flydsl_kernel, kernel_name, ptr

from .._vendor.a8w4_ops import _pointer, _scale_index


@flydsl_kernel(heuristics={"rows_per_wave": lambda k: 4 if k["cols"] >= 4096 else 16})
def channel_exponents_kernel(*, cols, rows_per_wave):   # keyword-only = the compile key
    groups = cols // 32                                  # derived constants: here (closure ints)
    partitions = 64 // rows_per_wave
    if cols % 32 or groups % partitions:
        raise ValueError(f"unsupported scale group geometry: cols={cols}")
    rounds = groups // partitions
    rows_per_block = 4 * rows_per_wave

    @flyc.kernel(name=kernel_name("a8w4_chan_exp", c=cols, rpw=rows_per_wave),
                 known_block_size=(256, 1, 1))
    def kernel(scales: fx.Int64, output: fx.Int64, i32_rows: fx.Int32):
        tid = fx.Int32(gpu.thread_id("x"))
        row = (fx.Int32(gpu.block_id("x")) * rows_per_block + (tid // 64) * rows_per_wave
               + tid % rows_per_wave)
        part = (tid % 64) // rows_per_wave
        if row < i32_rows:                               # runtime size, not a compile key
            largest = fx.Int32(0)
            for j in range(rounds):
                group = part + j * partitions
                code = fx.ptr_load(fx.add_offset(_pointer(scales, fx.Uint8, 1),
                                                 fx.Int64(_scale_index(row, group, groups))))
                largest = fx.max(largest, code.to(fx.Int32) & 255)
            for bit in range_constexpr(partitions.bit_length() - 1):
                largest = fx.max(largest, largest.shuffle_xor(rows_per_wave << bit, 64))
            if part == 0:
                fx.ptr_store(largest, fx.add_offset(_pointer(output, fx.Int32, 4), row))

    @flyc.jit
    def launch(scales: fx.Int64, output: fx.Int64, i32_rows: fx.Int32, stream: fx.Stream):
        grid = (i32_rows + (rows_per_block - 1)) // rows_per_block   # grid from runtime size
        kernel(scales, output, i32_rows).launch(grid=(fx.Int64(grid), 1, 1), block=(256, 1, 1),
                                                stream=stream)

    return launch


def channel_exponents(packed, scales):                   # the op: plain Python
    experts, n, kp = packed.shape
    rows, cols = experts * n, kp * 2
    out = torch.empty((experts, n), device=packed.device, dtype=torch.int32)
    if rows:  # numel: shuffle_scale pads rows to 32; CHECK mode then catches short buffers
        channel_exponents_kernel(ptr(scales, torch.uint8, numel=cdiv(rows, 32) * 32 * (cols // 32)),
                                 ptr(out, torch.int32), rows, cols=cols)
    return out
```

## 2. Rules

### Builder (the decorated compile function)

- **Import absolutely**: `from pyhip.contrib.flydsl.host_launch import ...`. a8w4 also runs
  as the top-level package `a8w4` (from `src/contrib/flydsl`), where a relative
  `from ...host_launch` fails.
- **Builder signature**: keyword-only compile-time parameters; it returns the `@flyc.jit`
  launcher.
  - No `functools.cache`: the decorator caches.
  - No tensors and no torch calls.
  - Geometry checks raise `ValueError` here; they run once per variant.
- **Compile key = only what changes the generated code**: unroll counts, tile shapes,
  optional paths, dtype names.
  - Runtime sizes (rows, tokens, M) are `fx.Int32` launch parameters. Compute the grid
    inside `launch` from them, or pass it as an `i32_grid` parameter.
  - A size in the key recompiles per shape. The object warns at 32 variants (`warn_after=`;
    `None` turns the warning off, e.g. for tuning sweeps).
- **Key values**: int / float / bool / str / None / Enum / tuple only; lists become tuples.
  Tensors, dicts and config objects are rejected: FlyDSL's disk-cache key ignores closure
  objects, which would give stale binaries (pitfalls §1).
- **Derived constants belong in the builder**, as above, not in the kernel body. Python ints
  computed inside the kernel can be promoted to `Int32` by the AST rewriter inside a
  runtime `if`.
- **Defaults and tuning choices** go in `heuristics={"name": f(key_dict)}`.
  - Entries are evaluated in order, only on a cache miss.
  - A caller can override one by passing the name.
- **Kernel name**: `kernel_name(base, **key)`, so ISA dumps and profiles tell variants apart.
- **Launch signature**: every parameter annotated.
  - `fx.Int64` for each pointer. The kernel converts it with `fx.inttoptr`, a8w4
    `_pointer(addr, dtype, align)`, or a buffer resource.
  - `fx.Int32` for sizes, prefixed `i32_`.
  - `fx.Float32` for float scalars.
  - Last parameter: `stream: fx.Stream`.

  `fx.Pointer`, `fx.Tensor`, `fx.Constexpr` and unannotated parameters raise `TypeError` at
  the first call. An unannotated parameter would otherwise be passed as Int32 and truncate
  addresses.
- **Per-variant compile hints**: set `launch.compile_hints[...]` in the builder before
  returning.

### The op and the call site

- **The op is plain Python**:
  1. check shapes and dtypes with cheap `if`s;
  2. allocate outputs;
  3. return early on empty input (`if rows:`);
  4. call `kern(ptr(t, dtype, align, numel), ..., sizes, **key)`.
- **Positional arguments** map to the launch parameters in order: tensor → `data_ptr()`,
  `None` → null pointer. `ptr()` documents the kernel's assumptions and costs nothing
  unless CHECK is on.
- **Never write `torch.cuda.current_stream()`, `.data_ptr()` or `_run_compiled` in a new
  wrapper.** The object launches on the first tensor's device and that device's current
  stream, through a raw handle (~0.1 µs instead of ~4 µs). CUDA graph capture works.
- **Never call the `@flyc.jit` launcher from torch code.** It costs ~70 µs per call and only
  ever uses the current device.
- **Hot loops and persistent objects** (megamoe-style): bind `self._k = kern.get(**key)`
  once, then call `self._k(*args)`; this skips the key lookup.

## 3. API

| call | does |
|---|---|
| `kern(*args, **key)` | launches; the first call per (key, device) compiles or loads from the disk cache |
| `kern.get(**key)` | returns the specialization `exe`, called as `exe(*args)`. `exe.launch` is the JitFunction (bound to `exe.dev` once called); `exe.cf` is its CompiledFunction |
| `kern.preload(**key)` | compiles into the disk cache without launching (warm-up) |
| `kern.variants` / `kern.clear()` | normalized key → specialization / drops them |
| `kern.builder` | the undecorated compile function |
| `cdiv`, `kernel_name`, `ptr`, `Ptr`, `CHECK` | helpers; `CHECK` mirrors `PYHIP_FLYDSL_CHECK=1` |

`kern(**key)` with no runtime arguments raises. That is the typical migration slip: replace
`compile_x(...)` with `kern.get(...)`.

Cost per launch on MI300X, 5 arguments (`python tests/flydsl/test_host_launch.py`):

| path | µs |
|---|---|
| bare CompiledFunction, raw stream | ~4.3 |
| `kern(...)` / bound `exe(...)` | ~6 |
| `_run_compiled` style, `torch.cuda.current_stream()` | ~9 |
| `@flyc.jit` called directly | ~70+ |

The machine is shared, so only compare numbers from the same run.

## 4. Debugging

- **`PYHIP_FLYDSL_CHECK=1`** (or `monkeypatch.setattr(host_launch, "CHECK", True)`) validates
  every launch against the annotations:
  - argument count;
  - a tensor at a non-`Int64` parameter ("argument order?");
  - Int32 overflow;
  - CPU tensors, mixed devices, non-contiguous tensors;
  - `ptr()` dtype, alignment and `numel`.

  Run a new op's tests once with it.
- **Compile log**:
  `logging.getLogger("pyhip.contrib.flydsl.host_launch").setLevel(logging.INFO)` prints
  `compiled name[key] on cuda:N in T ms`.
  - ~15 ms is a disk-cache hit; ≥50 ms is a real compile.
  - Use it to confirm a migrated kernel reuses the old binary (same source, same artifact).
- **Stale results after an edit**: see pitfalls §1 and use a fresh
  `FLYDSL_RUNTIME_CACHE_DIR=$(mktemp -d)`. Within a process, `kern.variants` keeps the old
  launcher, as `functools.cache` did; start a new process.

## 5. Migrating an existing compile function

Only do this when the task rewrites that wrapper anyway. Never touch `a8w4/experiments/`
(frozen copies) or `a8w4/_vendor/` (kept diffable against aiter).

1. Replace `@cache` with `@flydsl_kernel(...)`; its parameters are now passed as keywords.
   - Alternatively, wrap it without editing:
     `kern = flydsl_kernel(_compile_x.__wrapped__)`.
   - Move tuning defaults into `heuristics`.
2. Replace `_run_compiled(_compile_x(a, b), x.data_ptr(), ..., torch.cuda.current_stream())`
   with `kern(x, ..., a=a, b=b)`. Keep the `launch` argument order; drop the stream.
3. Switch the other callers of `_compile_x(...)` (bench, tune, megamoe objects) to
   `kern.get(a=..., b=...)`. If they need the raw JitFunction, use `.launch` or
   `.builder`. Find them with `grep -rn "_compile_x" src/contrib/flydsl`.
4. Verify:
   - bit-exact against the old wrapper, using the template
     `test_wraps_an_existing_compile_function_bit_exact` in the tests;
   - the a8w4 tests;
   - timing, if the op is on a hot path (`gpu-timing-protocol`).

## 6. When not to use it

- **Launchers taking `fx.Pointer` (`flyc.from_c_void_p`), `fx.Tensor` (`flyc.from_dlpack`) or
  `fx.Constexpr`**, as in aiter and pyhip `moe_gemm_splitk.py`: leave them as they are.
  New code uses the Int64 style above.
- **`torch.compile` / FakeTensor tracing / AOT export**: the kernel object is opaque. If you
  need these, register the *op* with `torch.library.custom_op` plus a fake impl;
  `host_launch` does not do this.
