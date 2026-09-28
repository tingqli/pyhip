# Autotuned MoE

[tuned_moe.py](tuned_moe.py) provides an Aiter-compatible `fused_moe` API.
It selects among Aiter, PyHIP ASM, and FlyDSL implementations for supported
BF16-input inference workloads on gfx942/gfx950. Other API features are forwarded
to Aiter; that path is not validated by the local tuner.

See the [performance snapshot](../../../../benchmarks/moe/perf-snapshot-350.md)
for recorded results, including failed cases. These are run-specific measurements.

## Standard workflow

### 1. Collect the workload

Add model shapes to [moe_shapes.py](../../testing/moe_shapes.py):
`HIDDEN_SIZE`, `INTER_SIZE`, `TP`, `E`, `TOPK`, and the default FP8 `quant_type`.
`INTER_SIZE` is global; each kernel uses `INTER_SIZE / TP`. The benchmark covers
one TP shard, without communication or backend-specific padding.

Choose representative token counts, dtype, quantization, activation, gate mode,
weight layout, and routing. Match the deployment GPU and numerical settings.
Run commands below from the repository root in the ROCm environment with PyHIP,
Aiter, and FlyDSL installed.

```bash
python3 benchmarks/moe/bench_tuned_moe.py --list-models
```

### 2. Use the API

For existing BF16 SiLU inputs:

```python
from pyhip.ops.moe.tuned_moe import fused_moe

y = fused_moe(x, w1, w2, topk_weight, topk_ids, output=out)
```

Quantized calls also pass `quant_type`, `w1_scale`, and `w2_scale`. The caller
prepares weight/scale layouts and preserves each weight's `is_shuffled` flag;
the wrapper does not infer layout from `Parameter` or convert weights at runtime.
Native calls require `int32` expert IDs in `[0, w1.shape[0])` on every call and
graph replay. The router must enforce this: cached execution does not scan ID
values, and invalid IDs can cause out-of-bounds access. The tuning-time reference
check is not a per-call safety check. Use a non-overlapping output matching the
input's shape, dtype, and device. Most local paths require shuffled weights.

On a cache miss, the tuner checks candidates against an independent Torch
reference, then times valid candidates. Cache hits reuse the config with the
current tensors and stream. Set `FLYDSL_AUTOTUNE=1` to force a fresh search;
turn it off after tuning and warm up before graph capture.

Compressed weights do not require quantized activations in every candidate.
The following paths dequantize weights in the kernel and keep BF16 inputs and
intermediates; both weights must be shuffled, and activation/shape limits apply:

| Weight format | `jit_splitk` | `fly_decode` (direct / sorted) |
|---|---|---|
| FP8 PTPC | Yes | Yes |
| FP8 per-tensor | No | Yes |
| FP8 block (W128×128) | Yes, H divisible by 512 | No |
| MXFP4 | Yes, H divisible by 1024 | Yes |

These A16W8/A16W4 candidates compete with activation-quantized paths. They still
must pass the same Torch reference and `calc_diff <= 0.02` check; no per-candidate
reference or relaxed tolerance is used. Skipping activation quantization can help
small batches, but does not guarantee higher accuracy, lower latency, or exact
equivalence to the quantized computation. Cache hits do not repeat this check.
Block-mode caches from the previous, activation-quantized-only policy are retuned.

### 3. Validate, tune, and compare

[bench_tuned_moe.py](../../../../benchmarks/moe/bench_tuned_moe.py) uses shared
inputs and references for Aiter and the tuned API. Start with a small check:

```bash
python3 benchmarks/moe/bench_tuned_moe.py \
	--models qwen35_35B_k256 --tokens 1 4 64 --dtype fp8 --check-only
```

Then tune the Aiter baseline and PyHIP candidates, export configs, and measure:

```bash
FLYDSL_AUTOTUNE_CONFIG_DIR="$PWD/moe_configs" \
python3 benchmarks/moe/bench_tuned_moe.py \
	--models qwen35_35B_k256 --tokens 1 4 64 1024 --dtype fp8 \
	--tune-aiter "$PWD/tuned_aiter_moe" --output /tmp/moe-tune.json
```

- `--tune-aiter` runs the official tuner once for deduplicated shapes, verifies
	CSV dispatch, then forces a new PyHIP search. It requires shuffled weights
	and does not support non-gated GELU. Without it, `--retune` refreshes only the
	PyHIP selection, including Aiter with its current config.
- `--check-only` skips comparison timing, not tuning on a cache miss.
	The benchmark controls `FLYDSL_AUTOTUNE`; use `--retune` or `--tune-aiter`
	rather than setting that variable externally to force a search.
- Rows show diff, median latency, speedup, winner TFLOPS, and the recorded config.
	Timing covers the full eager MoE call, not isolated GEMMs or graph replay.
	`AITER_INCORRECT` keeps timing results but is not a valid correctness pass.
- Use `--md FILE` for a Markdown report grouped by model, or `--output FILE`
	for JSON. Both require new file paths. Test deployment-specific precision
	and layout options separately; see the [benchmark guide](../../../../benchmarks/moe/README.md).

### 4. Reuse offline configs

Aiter CSVs and PyHIP config artifacts are separate. To reuse the export above
without requesting another search:

```bash
AITER_CONFIG_FMOE="$PWD/tuned_aiter_moe/tuned.csv" \
AITER_BYPASS_TUNE_CONFIG=0 AITER_KSPLIT=0 \
FLYDSL_AUTOTUNE_CONFIG_DIR="$PWD/moe_configs" \
python3 benchmarks/moe/bench_tuned_moe.py \
	--models qwen35_35B_k256 --tokens 1 4 64 1024 --dtype fp8 \
	--output /tmp/moe-reuse.json
```

Use the same settings in the application, with `AITER_ONLINE_TUNE=0` and
`FLYDSL_AUTOTUNE=0`. Set environment variables before importing the backends.
Keep the Aiter CSV path and relevant `AITER_*`, `MOE_*`, and `PYHIP_*` values
consistent: they are part of the model key.

- `FLYDSL_AUTOTUNE_CACHE_DIR` selects the normal persistent winner cache.
	`FLYDSL_AUTOTUNE_CONFIG_DIR` enables artifact lookup and export during forced
	tuning. Normal cache hits take precedence over artifacts.
- Keys include a power-of-two token bucket, model metadata, layout, device, and
	numerical settings. Several token counts can share one winner. `--retune`
	tests each requested count but may overwrite the same bucket's config.
- **`FLYDSL_AUTOTUNE=0` is not a strict offline mode.** A missing cache/artifact
	still triggers tuning. Prewarm all required keys before latency-sensitive
	execution or graph capture.
- Config artifacts store choices, not weights or compiled kernels. They do not
	remove first-use compilation. Revalidate after code or numerical changes;
	cache hits do not repeat the accuracy check.

## Integrate a new implementation

1. **Add the kernel and launcher.** Use the existing ASM or FlyDSL backend.
	 Write into and return the caller's output, honor each weight/scale layout,
	 and use the input device's current stream. Keep replay state graph-safe.
2. **Wire dispatch.** Add an `_impl` branch in `_fmoe_wrapper()` and a small
	 launcher in [tuned_moe.py](tuned_moe.py). Record any new config fields in
	 `last_dispatch`. Extend `_native_kind()` only when supporting a new API case.
3. **Enumerate candidates.** Add explicit `Config` entries in `_configs()` with
	 the required dtype, activation, layout, architecture, and launch-safety gates.
	 Avoid duplicating FlyDSL compiler constraints. Keep the shared output/finite
	 checks and `calc_diff <= 0.02`; this is not a per-element 2% tolerance.
4. **Test the fixed implementation.** Add small parameterized cases to
	 [test_moe.py](../../../../tests/ops/moe/test_moe.py), using
	 [shared helpers](../../testing/moe.py): `prepare_moe()`, `torch_reference()`,
	 and `make_moe_runner(config)`. Do not select tests through `_configs()` or
	 allow fallback. Cover tails and supported layouts. Put large timing cases
	 in a separate `perf` test sharing the same helper.
5. **Test integration.** Extend
	 [test_tuned_moe.py](../../../../tests/ops/moe/test_tuned_moe.py) for candidate
	 gates, cache/artifact reuse, streams, and graph replay. Update the model-key
	 version and artifact name when old configs are no longer valid.
6. **Compare end to end.** Rerun the benchmark with `--retune` (or
	 `--tune-aiter`) on relevant workloads. Inspect the winner config and both
	 diffs; enumeration alone does not prove the new path was selected. Keep
	 fixed-kernel selection in pytest, not in a new benchmark mode.

```bash
python3 -m pytest tests/ops/moe/test_moe.py tests/ops/moe/test_tuned_moe.py -q
# Example: explicit performance cases for one implementation
python3 -m pytest tests/ops/moe/test_moe.py -m perf -k jit_8wave -s
```

See the [fixed-kernel test guide](../../../../tests/ops/moe/README.md) for
individual cases and measurement details.


