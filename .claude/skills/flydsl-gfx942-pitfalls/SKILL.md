---
name: flydsl-gfx942-pitfalls
description: Verified pitfalls when writing or modifying FlyDSL kernels for gfx942 (MI300X), from the pyhip a8w4 work — the exact JIT disk-cache key rules of FlyDSL 0.3.2 (which edits do NOT invalidate a cached kernel), runtime control-flow limits, divergent ifs in loops, VGPR hoisting and occupancy, LDS 64 KB launch failure, release fences that write back L2, cross-XCD coherence, atomic serialization, graph-safe counters, and the bit-exact verification recipe. Use when editing kernels under a8w4/kernels or a8w4/_vendor (or any FlyDSL kernel), adding compile-time switches, or when a kernel change "has no effect" or results look stale.
---

# FlyDSL on gfx942: pitfalls that cost real time

## 1. The JIT disk cache key (FlyDSL 0.3.2; verified with a probe kernel, fresh process per run)

Re-verify after a FlyDSL upgrade (pyhip, from `src/contrib/flydsl`):
`HIP_VISIBLE_DEVICES=0 python a8w4/experiments/20261115_flydsl_cache_key/run_probe.py`
(`lds_probe.py` next to it for the LDS limit; results of 0.3.2 in `result.txt`).

A kernel is reused from `~/.flydsl/cache` (or `FLYDSL_RUNTIME_CACHE_DIR`) unless the change is
part of its key. Key = toolchain hash + source of the `@flyc.jit` function + sources of its deps
+ closure scalars + a snapshot of the module globals the code reads + argument signature.

| change | new binary? |
|---|---|
| source of the `@flyc.jit` / `@flyc.kernel` function (also a comment inside it) | yes |
| plain helper in the SAME directory as the jit function, called by bare name | yes |
| plain helper in ANOTHER directory (e.g. a shared `utils/*.py` used by `kernels/*.py`) | **no: stale** |
| plain helper called as a module attribute (`mod.helper(...)`): the name lookup finds the module | **no: stale** |
| plain helper called only inside a nested function (a `def` inside a helper or kernel body): only each scanned function's own bytecode names are read | **no: stale** |
| a `@flyc.jit` / `@flyc.kernel` dependency anywhere | yes (its own key) |
| closure value int / float / bool / str / None / tuple / Enum | yes |
| closure value list / dict / set / object | **no: stale** |
| module global the kernel (or a jit / same-dir dep) reads by name | yes for scalars and list / dict / set / tuple (by value), callables by qualname; **other objects only by type** |
| a value read at trace time from runtime state (tensor address, env var read in a helper) | **no: stale** |
| text outside function bodies (module comments) | no rebuild, none needed |

The module-attribute and nested-function rows come from `jit_function._collect_dependency_sources`
(it reads `co_names` and closures of each function, not nested code objects), checked by listing
the dependencies of a8w4 `compile_ep_gemm1`, not from the probe.

Rules:
- Compile-time switches: int / tuple arguments of the (usually `functools.cache`d) compile
  function, also folded into the kernel name (then dumps and profiles tell variants apart).
  Never a list / dict / set / callable / config object in the closure. New host wrappers use
  `@flydsl_kernel` (skill `flydsl-host-launch`), which rejects such keys.
- Import helpers by name into the kernel's directory (a8w4: `kernels/_mem.py` etc.). After
  editing any helper, or whenever a change "does nothing": run with a fresh cache,
  `FLYDSL_RUNTIME_CACHE_DIR=$(mktemp -d) python ...`. An ablation that reuses a stale binary
  reports "no effect" (this produced a wrong a8w4 gemm2 epilogue conclusion).
- Env-var knobs: read them once into a closure int; better, freeze them as module constants.
- In-process, `functools.cache` on the compile function also returns the old launcher: start
  a new process after editing.

## 2. Control flow

- Runtime `if` / `while` only inside `@flyc.jit` / `@flyc.kernel` bodies (AST rewrite);
  `range()` becomes `scf.for` only there. Elsewhere use `range_constexpr` / `const_expr`.
- "compile-time `if` followed by a runtime `elif`" is unreliable: nest the ifs.
- A nested `@flyc.jit` does not re-specialize on Python int arguments: derive sizes from
  list lengths or the closure.
- No lane-divergent `if` (e.g. `if tx == 0:` taking a ticket) inside a runtime `while`: LLVM
  runs the body under a partial exec mask -> wrong results or a hang. Branch per wave, select
  per lane.

## 3. Registers, LDS, occupancy

- Check every kernel change: `python -m a8w4.prof isa --grep <kernel> -- <cmd that compiles it>`
  (fresh cache automatically; VGPR / AGPR / SGPR, spills, LDS, scratch, waves per SIMD).
- Waves per SIMD from VGPRs (512 per lane, granule 8): <= 128 -> 4, <= 168 -> 3, <= 256 -> 2.
  Prefetch at BM >= 64 or two row variants via `dyn_m` pushed a8w4 kernels over 168 and made
  large batches slower.
- LLVM hoists per-lane address math out of loops and keeps it live in VGPRs (a16w4 got 73-83
  spills): break it with an opaque value (`_opaque_tx`).
- LDS > 64 KB does not raise anywhere (verified, FlyDSL 0.3.2): LLVM prints `local memory (N)
  exceeds limit (65536)` to stderr, the launch prints `hipErrorIllegalState`, Python goes on
  and the output is simply never written. Watch stderr, check outputs (`bench check` flags
  it as BAD), keep a Python assert on the LDS size (a8w4 gemm2 fp8 l128 uses exactly 64 KB).

## 4. Memory model and synchronization

- A release fence emits `buffer_wbl2 sc0 sc1`: the whole L2 is written back (7-14 us). Write
  the data write-through (sc0 | sc1), `s_waitcnt vmcnt(0)`, then the flag.
- The 8 XCD L2s are not coherent: data another WG wrote (maybe another XCD) needs sc0 | sc1
  loads (`_gload_sys`).
- All WGs hitting one atomic address serialize (~40 ns each): split counters, one cache line
  each (a8w4 ep1k: 32 sub-counters).
- Graph safety: no host-side epochs as flags (tp1kq is not graph-safe); the last WG resets
  counters in-kernel (ep_prep, ep1k).
- Persistent kernels and grid barriers need all their WGs resident at once: fewer free CUs
  (another process) deadlocks instead of slowing down. 304 CU / 8 XCD are hard-coded in a8w4
  (`kernels/_hw.py`: `NUM_CU`, `NUM_XCD`, `LDS_BYTES_MAX`, cache-policy bits `CM_*`).

## 5. Verifying a kernel change (before any timing)

1. Bit-exact first: with topk 1 every output element is written once, so old and new kernels
   must match bit for bit: `python -m a8w4.bench check --shape 4096,2048,32,256,1 -t 64 4096
   -v <old> <new> --bitwise`. Compare intermediates (q8s, gemm1 output) bitwise as well.
2. Then cos vs the fp32 reference and stability / graph replay: `bench check ... --graph`.
3. Poison output buffers (`a8w4.bench.check.poison_`) so skipped rows show up; split timing and
   bitwise comparison also caught a double-processed odd tail row that left results unchanged
   but doubled the traffic.
4. Then resources (section 3), then time (`gpu-timing-protocol`, `python -m a8w4.bench ab`),
   then retune the affected shapes: tile changes reorder the configs.

## 6. Clock and power

MI300X at the 750 W cap runs ~1.05-1.4 GHz and reads 3.6-4.0 TB/s: compare against bounds
measured on this machine, not datasheet peaks. A 39% VALU cut (fd2 decode) gave ~10%: the
bottleneck moved to power. Better balanced work can be slower (EP per-XCD queues: hot tiles
together, higher power, lower clock, +12-18% at <= 1024 tokens).
