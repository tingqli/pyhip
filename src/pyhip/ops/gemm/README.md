# Autotuned A8W4 GEMM

[tuned_gemm_a8w4.py](tuned_gemm_a8w4.py) provides the same ops-level signature
and defaults as `aiter.gemm_a8w4_mxfp8`, using the `flydsl.autotune` framework
already used by [tuned_moe.py](../moe/tuned_moe.py).

```python
# from aiter import gemm_a8w4_mxfp8
from pyhip.ops.gemm.tuned_gemm_a8w4 import gemm_a8w4_mxfp8

y = gemm_a8w4_mxfp8(a, b, a_scale, b_scale, a_preshuffle=False)
```

Only the import changes; installing PyHIP does not monkey-patch Aiter.
Off gfx950, the call and all its arguments are forwarded to Aiter unchanged.
The installed Aiter public A8W4 op is gfx1250-only, so on gfx950 an adapter uses
its native FlyDSL MXFP8 x MXFP4 implementation as the Aiter candidate.

## gfx950 input contract

| Operand | Shape / type | Preparation |
|---|---|---|
| `A` | `[M, K]` FP8 E4M3 | row-major; `a_preshuffle=False` |
| `B` | `[N, K/2]` packed MXFP4 | `shuffle_mxfp8fp4_b`, equivalent to `shuffle_weight(B, (16, 16))` |
| `ScaleA` | `[pad256(M), pad8(K/32)]` E8M0 | `per_1x32_mx_quant_hip(..., shuffle=True)` or `shuffle_scale` |
| `ScaleB` | `[pad256(N), pad8(K/32)]` E8M0 | `shuffle_scale` |

These are Aiter's **gfx950** scale bytes, not the gfx1250
`shuffle_mxfp8fp4_scale` layout. The wrapper does not guess layouts from values
or custom tensor attributes. Both candidates read the same A/B data.

Native execution is inference-only, returns a new BF16 `[M, N]` tensor, and
requires contiguous tensors on one device, A/B aligned to 16 bytes and scales
to 4 bytes. The current kernels have 32-bit buffer bounds. PyHIP requires
`K % 256 == 0` and `N % 16 == 0`; Aiter reads whole 32-row B-scale chunks and
therefore requires `N % 32 == 0`. Invalid shapes or options fail explicitly.
The API retains Aiter's `a_preshuffle=True` default, but callers on gfx950 must
pass `False`; neither native candidate consumes preshuffled A.

## Selection and correctness

1. An explicit `kernelName` bypasses autotuning. It may name an Aiter
   `flydsl_mxpsh_*_F8_F4_B16_*` configuration or a PyHIP
   `pyhip_mxfp8_4w_*` configuration.
2. Automatic calls use FlyDSL's normal winner cache, then optional JSON
   configuration artifacts. A miss enumerates legal Aiter tile/split/WPE/XCD
   configurations and the PyHIP variants.
3. **Every candidate, including Aiter, is checked against an independent
   full-K FP32 reference rounded to BF16 before benchmarking.**
   Nonfinite outputs, wrong output identity, and `calc_diff > 1e-5` are rejected.
   No valid candidate means an error, not an unchecked fallback.
4. FlyDSL benchmarks the valid candidates, caches the winner, and executes it
   with the current tensors and stream. PyHIP's scale repack and Aiter's
   Split-K reduction are included. Common output allocation happens outside
   candidate ranking; the comparison benchmark times the complete public call.

`splitk` is a literal count. It restricts automatic candidates or overrides
an explicit Aiter configuration; PyHIP does not split K.
`record_dispatch = True` enables `last_dispatch` diagnostics, including
`_impl`, `kernelName` and `splitK`. Cached execution does not recheck tensor
values, and execution errors propagate rather than silently switching backend.

Keys include **exact M/N/K**, operand dtype/shape/stride, scale-layout version,
implementation source hashes, package versions, device and numerical settings.
Unlike the MoE token buckets, this first GEMM integration does not extrapolate
across M values. There is no manual CSV dispatcher or largest-M fallback.

## Tune and export

The first automatic call compiles, validates and times candidates; it can be
expensive. Tune on an idle GPU before deployment:

```bash
FLYDSL_AUTOTUNE_CONFIG_DIR="$PWD/a8w4_configs" \
FLYDSL_AUTOTUNE_CACHE_DIR="$PWD/a8w4_cache" \
python benchmarks/gemm/tune_gemm_a8w4.py \
    --shapes 32,64,512 4096,4096,8192 --retune \
    --output /tmp/a8w4-tune.json
```

[The benchmark](../../../../benchmarks/gemm/tune_gemm_a8w4.py) uses the runtime
autotuner for both an Aiter-only baseline and the combined API. Their cache
identities differ, so the baseline cannot overwrite the combined winner.
No second winner search or CSV dispatcher is maintained by the script.

`-i shapes.csv` accepts M,N,K columns. `--check-only` skips comparison timing,
**not tuning on a cache miss**. Default comparison timing is the original
`pyhip.cudaPerf` with 50 independent input buffers and 50 alternating
Aiter/auto samples. It records raw times, current choices, output pointers,
correctness and effective TFLOPS; it checks the actual timed outputs.
Every timed output is retained until validation, so increasing `--samples`
also increases memory use.
Small working sets may fit in cache even with 50 buffers.

## Reuse and graph capture

```bash
FLYDSL_AUTOTUNE=0 \
FLYDSL_AUTOTUNE_CONFIG_DIR="$PWD/a8w4_configs" \
FLYDSL_AUTOTUNE_CACHE_DIR="$PWD/a8w4_cache" \
python benchmarks/gemm/tune_gemm_a8w4.py \
    --shapes 32,64,512 4096,4096,8192 --output /tmp/a8w4-reuse.json
```

- `FLYDSL_AUTOTUNE=1` forces search; if a config directory is set, the winner
  is exported as `pyhip_gemm_a8w4_mxfp8_v1-*.json`.
- `FLYDSL_AUTOTUNE_CACHE_DIR` selects the normal persistent cache. Set it
  before import. Normal cache hits take precedence over artifacts.
- `FLYDSL_AUTOTUNE_CONFIG_DIR` enables artifact lookup and forced-tuning export.
  Artifacts store choices, not weights or compiled kernels.
- `FLYDSL_AUTOTUNE=0` is **not strict offline mode**: a missing entry tunes.
  Warm up every required shape/config in eager mode before latency-sensitive
  execution or CUDA graph capture, and turn forced tuning off for capture.
- Under `torch.compile`, the API is the opaque
  `pyhip::gemm_a8w4_mxfp8` operator, with a fake implementation for shape tracing.
- The native libraries currently cache device-loaded FlyDSL functions
  process-wide. Use one GPU per process. A call on a second device is rejected
  before launch, rather than reusing the first device's code object.

## Tests

```bash
python -m pytest tests/ops/gemm/test_tuned_gemm_a8w4.py \
    tests/ops/gemm/test_gemm_mxfp8_4w.py
```
