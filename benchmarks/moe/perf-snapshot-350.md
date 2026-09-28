# MoE performance

Winner TFLOPS uses the measured full tuned-call median latency.

```bash
python3 bench_tuned_moe.py --tune-aiter --md perf-snapshot-350.md
```

### hy3 (fp8/per_tensor, silu, TP=8)

model_dim=4096, inter_dim=1536, inter_dim_tp=192, experts=193, topk=9, gate_mode=separated, preshuffle=on, routing=balanced, seed=0

| model | M | H / I_tp / E / topk | check A/T | Aiter diff | winner diff | Aiter us | tuned us | speedup | winner | winner TFLOPS | status | winner config |
|---|---:|---|---|---:|---:|---:|---:|---:|---|---:|---|---|
| hy3 | 1 | 4096 / 192 / 193 / 9 | ERROR/PASS | — | 0.000856079 | — | — | — | fly_decode | — | NOT_COMPARABLE | {"_impl": "fly_decode", "block_n": 0, "decode_alg": "batch1", "down_path": "default", "num_oc_splits": 1, "padding": null, "tile_m_down": 16, "tile_m_gate": 16, "tile_n_down": 64, "tile_n_gate": 64} |
| hy3 | 4 | 4096 / 192 / 193 / 9 | ERROR/PASS | — | 0.00101768 | — | — | — | fly_decode | — | NOT_COMPARABLE | {"_impl": "fly_decode", "block_n": 0, "decode_alg": "batch1", "down_path": "default", "num_oc_splits": 1, "padding": null, "tile_m_down": 16, "tile_m_gate": 16, "tile_n_down": 64, "tile_n_gate": 64} |
| hy3 | 16 | 4096 / 192 / 193 / 9 | ERROR/PASS | — | 0.00101401 | — | — | — | fly_decode | — | NOT_COMPARABLE | {"_impl": "fly_decode", "block_n": 0, "decode_alg": "batch1", "down_path": "default", "num_oc_splits": 1, "padding": null, "tile_m_down": 16, "tile_m_gate": 16, "tile_n_down": 64, "tile_n_gate": 32} |
| hy3 | 64 | 4096 / 192 / 193 / 9 | ERROR/PASS | — | 0.00103808 | — | — | — | fly_decode | — | NOT_COMPARABLE | {"_impl": "fly_decode", "block_n": 0, "decode_alg": "splitk", "down_path": "default", "num_oc_splits": 1, "padding": null, "tile_m_down": 32, "tile_m_gate": 32, "tile_n_down": 64, "tile_n_gate": 64} |
| hy3 | 256 | 4096 / 192 / 193 / 9 | ERROR/PASS | — | 0.00108143 | — | — | — | fly_decode | — | NOT_COMPARABLE | {"_impl": "fly_decode", "block_n": 0, "decode_alg": "splitk", "down_path": "default", "num_oc_splits": 1, "padding": null, "tile_m_down": 32, "tile_m_gate": 32, "tile_n_down": 64, "tile_n_gate": 64} |
| hy3 | 1024 | 4096 / 192 / 193 / 9 | ERROR/PASS | — | 0.000120211 | — | — | — | fly_prefill | — | NOT_COMPARABLE | {"_impl": "fly_prefill", "block_n": 0, "decode_alg": "splitk", "down_path": "1x4_64x256", "num_oc_splits": 1, "padding": 128, "tile_k_gate": 256, "tile_m_down": 64, "tile_m_gate": 64, "tile_n_down": 256, "tile_n_gate": 128} |
| hy3 | 4096 | 4096 / 192 / 193 / 9 | ERROR/PASS | — | 0.000118167 | — | — | — | fly_prefill | — | NOT_COMPARABLE | {"_impl": "fly_prefill", "block_n": 0, "decode_alg": "splitk", "down_path": "default", "num_oc_splits": 1, "padding": null, "tile_k_gate": 128, "tile_m_down": 128, "tile_m_gate": 128, "tile_n_down": 128, "tile_n_gate": 128} |
| hy3 | 8192 | 4096 / 192 / 193 / 9 | ERROR/PASS | — | 0.00033335 | — | — | — | fly_prefill | — | NOT_COMPARABLE | {"_impl": "fly_prefill", "block_n": 0, "decode_alg": "splitk", "down_path": "default", "num_oc_splits": 1, "padding": null, "tile_k_gate": 128, "tile_m_down": 128, "tile_m_gate": 128, "tile_n_down": 128, "tile_n_gate": 128} |
| hy3 | 16384 | 4096 / 192 / 193 / 9 | ERROR/PASS | — | 0.000102513 | — | — | — | fly_prefill | — | NOT_COMPARABLE | {"_impl": "fly_prefill", "block_n": 0, "decode_alg": "splitk", "down_path": "default", "num_oc_splits": 1, "padding": null, "tile_k_gate": 128, "tile_m_down": 128, "tile_m_gate": 128, "tile_n_down": 128, "tile_n_gate": 128} |
| hy3 | 32768 | 4096 / 192 / 193 / 9 | ERROR/PASS | — | 0.000116739 | — | — | — | fly_prefill | — | NOT_COMPARABLE | {"_impl": "fly_prefill", "block_n": 0, "decode_alg": "splitk", "down_path": "8x1", "num_oc_splits": 1, "padding": 128, "tile_k_gate": 256, "tile_m_down": 256, "tile_m_gate": 64, "tile_n_down": 128, "tile_n_gate": 128} |

[hy3 M=1] Performance comparison skipped: Aiter must execute with valid output metadata and winner must pass accuracy validation.
[hy3 M=1] aiter: ERROR: wrong! device_gemm with the specified compilation parameters does not support this GEMM problem
[hy3 M=4] Performance comparison skipped: Aiter must execute with valid output metadata and winner must pass accuracy validation.
[hy3 M=4] aiter: ERROR: wrong! device_gemm with the specified compilation parameters does not support this GEMM problem
[hy3 M=16] Performance comparison skipped: Aiter must execute with valid output metadata and winner must pass accuracy validation.
[hy3 M=16] aiter: ERROR: wrong! device_gemm with the specified compilation parameters does not support this GEMM problem
[hy3 M=64] Performance comparison skipped: Aiter must execute with valid output metadata and winner must pass accuracy validation.
[hy3 M=64] aiter: ERROR: wrong! device_gemm with the specified compilation parameters does not support this GEMM problem
[hy3 M=256] Performance comparison skipped: Aiter must execute with valid output metadata and winner must pass accuracy validation.
[hy3 M=256] aiter: ERROR: wrong! device_gemm with the specified compilation parameters does not support this GEMM problem
[hy3 M=1024] Performance comparison skipped: Aiter must execute with valid output metadata and winner must pass accuracy validation.
[hy3 M=1024] aiter: ERROR: wrong! device_gemm with the specified compilation parameters does not support this GEMM problem
[hy3 M=4096] Performance comparison skipped: Aiter must execute with valid output metadata and winner must pass accuracy validation.
[hy3 M=4096] aiter: ERROR: wrong! device_gemm with the specified compilation parameters does not support this GEMM problem
[hy3 M=8192] Performance comparison skipped: Aiter must execute with valid output metadata and winner must pass accuracy validation.
[hy3 M=8192] aiter: ERROR: wrong! device_gemm with the specified compilation parameters does not support this GEMM problem
[hy3 M=16384] Performance comparison skipped: Aiter must execute with valid output metadata and winner must pass accuracy validation.
[hy3 M=16384] aiter: ERROR: wrong! device_gemm with the specified compilation parameters does not support this GEMM problem
[hy3 M=32768] Performance comparison skipped: Aiter must execute with valid output metadata and winner must pass accuracy validation.
[hy3 M=32768] aiter: ERROR: wrong! device_gemm with the specified compilation parameters does not support this GEMM problem

### qwen35_397B (fp8/ptpc, silu, TP=8)

model_dim=4096, inter_dim=4096, inter_dim_tp=512, experts=512, topk=10, gate_mode=separated, preshuffle=on, routing=balanced, seed=0

| model | M | H / I_tp / E / topk | check A/T | Aiter diff | winner diff | Aiter us | tuned us | speedup | winner | winner TFLOPS | status | winner config |
|---|---:|---|---|---:|---:|---:|---:|---:|---|---:|---|---|
| qwen35_397B | 1 | 4096 / 512 / 512 / 10 | PASS/PASS | 8.1597e-05 | 0.00112279 | 39.18 | 23.28 | 1.683x | jit_batch1 | 5.405 | PASS | {"_impl": "jit_batch1", "block_n": 0, "decode_alg": "splitk", "down_path": "default", "num_oc_splits": 1, "padding": null, "tile_m_down": 16, "tile_m_gate": 16, "tile_n_down": 32, "tile_n_gate": 32} |
| qwen35_397B | 4 | 4096 / 512 / 512 / 10 | PASS/PASS | 0.00010965 | 0.0011035 | 66.80 | 51.62 | 1.294x | fly_decode | 9.750 | PASS | {"_impl": "fly_decode", "block_n": 0, "decode_alg": "batch1", "down_path": "default", "num_oc_splits": 1, "padding": null, "tile_m_down": 16, "tile_m_gate": 16, "tile_n_down": 64, "tile_n_gate": 32} |
| qwen35_397B | 16 | 4096 / 512 / 512 / 10 | PASS/PASS | 0.000141109 | 0.00106789 | 220.56 | 202.32 | 1.090x | fly_decode | 9.951 | PASS | {"_impl": "fly_decode", "block_n": 0, "decode_alg": "batch1", "down_path": "default", "num_oc_splits": 1, "padding": null, "tile_m_down": 16, "tile_m_gate": 16, "tile_n_down": 64, "tile_n_gate": 32} |
| qwen35_397B | 64 | 4096 / 512 / 512 / 10 | PASS/PASS | 0.000134026 | 0.00104426 | 644.41 | 567.15 | 1.136x | jit_splitk | 14.199 | PASS | {"_impl": "jit_splitk", "block_n": 0, "decode_alg": "splitk", "down_path": "default", "num_oc_splits": 1, "padding": null, "tile_m_down": 32, "tile_m_gate": 32, "tile_n_down": 64, "tile_n_gate": 64} |
| qwen35_397B | 256 | 4096 / 512 / 512 / 10 | PASS/PASS | 0.000148564 | 0.00105818 | 662.81 | 595.13 | 1.114x | jit_splitk | 54.127 | PASS | {"_impl": "jit_splitk", "block_n": 0, "decode_alg": "splitk", "down_path": "default", "num_oc_splits": 1, "padding": null, "tile_m_down": 16, "tile_m_gate": 16, "tile_n_down": 64, "tile_n_gate": 64} |
| qwen35_397B | 1024 | 4096 / 512 / 512 / 10 | PASS/PASS | 0.000145156 | 0.000145175 | 719.95 | 713.47 | 1.009x | aiter | 180.595 | PASS | {"_impl": "aiter"} |
| qwen35_397B | 4096 | 4096 / 512 / 512 / 10 | PASS/PASS | 0.000145842 | 0.000132282 | 1937.13 | 1050.75 | 1.844x | fly_prefill | 490.501 | PASS | {"_impl": "fly_prefill", "block_n": 0, "decode_alg": "splitk", "down_path": "default", "num_oc_splits": 1, "padding": null, "tile_k_gate": 128, "tile_m_down": 128, "tile_m_gate": 128, "tile_n_down": 128, "tile_n_gate": 256} |
| qwen35_397B | 8192 | 4096 / 512 / 512 / 10 | PASS/PASS | 0.000145964 | 0.000132337 | 3213.88 | 1542.24 | 2.084x | fly_prefill | 668.372 | PASS | {"_impl": "fly_prefill", "block_n": 0, "decode_alg": "splitk", "down_path": "1x4_64x256", "num_oc_splits": 1, "padding": 0, "tile_k_gate": 128, "tile_m_down": 64, "tile_m_gate": 64, "tile_n_down": 256, "tile_n_gate": 128} |
| qwen35_397B | 16384 | 4096 / 512 / 512 / 10 | PASS/PASS | 0.000145688 | 0.000131971 | 5110.33 | 2384.97 | 2.143x | fly_prefill | 864.406 | PASS | {"_impl": "fly_prefill", "block_n": 0, "decode_alg": "splitk", "down_path": "1x4_64x256", "num_oc_splits": 1, "padding": 0, "tile_k_gate": 128, "tile_m_down": 64, "tile_m_gate": 64, "tile_n_down": 256, "tile_n_gate": 256} |
| qwen35_397B | 32768 | 4096 / 512 / 512 / 10 | PASS/PASS | 0.000145881 | 0.00013217 | 6846.39 | 4603.52 | 1.487x | fly_prefill | 895.655 | PASS | {"_impl": "fly_prefill", "block_n": 0, "decode_alg": "splitk", "down_path": "1x4_64x256", "num_oc_splits": 1, "padding": 0, "tile_k_gate": 128, "tile_m_down": 64, "tile_m_gate": 64, "tile_n_down": 256, "tile_n_gate": 256} |


### qwen35_397B_k256 (fp8/ptpc, silu, TP=8)

model_dim=4096, inter_dim=2048, inter_dim_tp=256, experts=512, topk=10, gate_mode=separated, preshuffle=on, routing=balanced, seed=0

| model | M | H / I_tp / E / topk | check A/T | Aiter diff | winner diff | Aiter us | tuned us | speedup | winner | winner TFLOPS | status | winner config |
|---|---:|---|---|---:|---:|---:|---:|---:|---|---:|---|---|
| qwen35_397B_k256 | 1 | 4096 / 256 / 512 / 10 | PASS/PASS | 0.000157611 | 0.000792223 | 36.50 | 19.56 | 1.866x | jit_batch1 | 3.216 | PASS | {"_impl": "jit_batch1", "block_n": 0, "decode_alg": "splitk", "down_path": "default", "num_oc_splits": 1, "padding": null, "tile_m_down": 16, "tile_m_gate": 16, "tile_n_down": 32, "tile_n_gate": 32} |
| qwen35_397B_k256 | 4 | 4096 / 256 / 512 / 10 | PASS/PASS | 0.000113885 | 0.000931371 | 52.30 | 32.64 | 1.602x | fly_decode | 7.710 | PASS | {"_impl": "fly_decode", "block_n": 0, "decode_alg": "batch1", "down_path": "default", "num_oc_splits": 1, "padding": null, "tile_m_down": 16, "tile_m_gate": 16, "tile_n_down": 64, "tile_n_gate": 64} |
| qwen35_397B_k256 | 16 | 4096 / 256 / 512 / 10 | PASS/PASS | 0.000145205 | 0.00103779 | 127.70 | 107.54 | 1.187x | fly_decode | 9.360 | PASS | {"_impl": "fly_decode", "block_n": 0, "decode_alg": "batch1", "down_path": "default", "num_oc_splits": 1, "padding": null, "tile_m_down": 16, "tile_m_gate": 16, "tile_n_down": 64, "tile_n_gate": 32} |
| qwen35_397B_k256 | 64 | 4096 / 256 / 512 / 10 | PASS/PASS | 0.000145818 | 0.00102638 | 334.34 | 296.06 | 1.129x | jit_splitk | 13.600 | PASS | {"_impl": "jit_splitk", "block_n": 0, "decode_alg": "splitk", "down_path": "default", "num_oc_splits": 1, "padding": null, "tile_m_down": 32, "tile_m_gate": 32, "tile_n_down": 64, "tile_n_gate": 64} |
| qwen35_397B_k256 | 256 | 4096 / 256 / 512 / 10 | PASS/PASS | 0.000140612 | 0.0010433 | 347.18 | 314.94 | 1.102x | jit_splitk | 51.140 | PASS | {"_impl": "jit_splitk", "block_n": 0, "decode_alg": "splitk", "down_path": "default", "num_oc_splits": 1, "padding": null, "tile_m_down": 16, "tile_m_gate": 16, "tile_n_down": 64, "tile_n_gate": 64} |
| qwen35_397B_k256 | 1024 | 4096 / 256 / 512 / 10 | PASS/PASS | 0.000144811 | 0.000144805 | 388.14 | 386.87 | 1.003x | aiter | 166.529 | PASS | {"_impl": "aiter"} |
| qwen35_397B_k256 | 4096 | 4096 / 256 / 512 / 10 | PASS/PASS | 0.00014338 | 0.000131198 | 964.45 | 669.07 | 1.441x | fly_prefill | 385.159 | PASS | {"_impl": "fly_prefill", "block_n": 0, "decode_alg": "splitk", "down_path": "default", "num_oc_splits": 1, "padding": null, "tile_k_gate": 128, "tile_m_down": 128, "tile_m_gate": 128, "tile_n_down": 128, "tile_n_gate": 256} |
| qwen35_397B_k256 | 8192 | 4096 / 256 / 512 / 10 | PASS/PASS | 0.000143948 | 0.000130357 | 1627.18 | 1142.78 | 1.424x | fly_prefill | 451.004 | PASS | {"_impl": "fly_prefill", "block_n": 0, "decode_alg": "splitk", "down_path": "default", "num_oc_splits": 1, "padding": null, "tile_k_gate": 128, "tile_m_down": 128, "tile_m_gate": 128, "tile_n_down": 128, "tile_n_gate": 256} |
| qwen35_397B_k256 | 16384 | 4096 / 256 / 512 / 10 | PASS/PASS | 0.000143716 | 0.00012944 | 2691.20 | 1792.36 | 1.501x | fly_prefill | 575.102 | PASS | {"_impl": "fly_prefill", "block_n": 0, "decode_alg": "splitk", "down_path": "1x4_64x256", "num_oc_splits": 1, "padding": 0, "tile_k_gate": 256, "tile_m_down": 64, "tile_m_gate": 64, "tile_n_down": 256, "tile_n_gate": 128} |
| qwen35_397B_k256 | 32768 | 4096 / 256 / 512 / 10 | PASS/PASS | 0.000143417 | 0.000129951 | 4347.68 | 3113.58 | 1.396x | fly_prefill | 662.126 | PASS | {"_impl": "fly_prefill", "block_n": 0, "decode_alg": "splitk", "down_path": "8x1_compact", "num_oc_splits": 1, "padding": 0, "tile_k_gate": 128, "tile_m_down": 64, "tile_m_gate": 64, "tile_n_down": 128, "tile_n_gate": 256} |


### qwen35_35B (fp8/ptpc, silu, TP=1)

model_dim=2048, inter_dim=512, inter_dim_tp=512, experts=256, topk=8, gate_mode=separated, preshuffle=on, routing=balanced, seed=0

| model | M | H / I_tp / E / topk | check A/T | Aiter diff | winner diff | Aiter us | tuned us | speedup | winner | winner TFLOPS | status | winner config |
|---|---:|---|---|---:|---:|---:|---:|---:|---|---:|---|---|
| qwen35_35B | 1 | 2048 / 512 / 256 / 8 | PASS/PASS | 0.000126998 | 0.00098896 | 31.08 | 17.44 | 1.782x | jit_batch1 | 2.886 | PASS | {"_impl": "jit_batch1", "block_n": 0, "decode_alg": "splitk", "down_path": "default", "num_oc_splits": 1, "padding": null, "tile_m_down": 16, "tile_m_gate": 16, "tile_n_down": 32, "tile_n_gate": 32} |
| qwen35_35B | 4 | 2048 / 512 / 256 / 8 | PASS/PASS | 0.000202602 | 0.00100774 | 45.04 | 28.26 | 1.594x | fly_decode | 7.124 | PASS | {"_impl": "fly_decode", "block_n": 0, "decode_alg": "batch1", "down_path": "default", "num_oc_splits": 1, "padding": null, "tile_m_down": 16, "tile_m_gate": 16, "tile_n_down": 64, "tile_n_gate": 64} |
| qwen35_35B | 16 | 2048 / 512 / 256 / 8 | PASS/PASS | 0.0001713 | 0.00101726 | 109.32 | 87.88 | 1.244x | fly_decode | 9.164 | PASS | {"_impl": "fly_decode", "block_n": 0, "decode_alg": "batch1", "down_path": "default", "num_oc_splits": 1, "padding": null, "tile_m_down": 16, "tile_m_gate": 16, "tile_n_down": 64, "tile_n_gate": 32} |
| qwen35_35B | 64 | 2048 / 512 / 256 / 8 | PASS/PASS | 0.000157317 | 0.00103141 | 175.48 | 157.50 | 1.114x | jit_splitk | 20.452 | PASS | {"_impl": "jit_splitk", "block_n": 0, "decode_alg": "splitk", "down_path": "default", "num_oc_splits": 1, "padding": null, "tile_m_down": 32, "tile_m_gate": 32, "tile_n_down": 64, "tile_n_gate": 64} |
| qwen35_35B | 256 | 2048 / 512 / 256 / 8 | PASS/PASS | 0.000141436 | 0.00104189 | 182.58 | 174.28 | 1.048x | jit_batch | 73.931 | PASS | {"_impl": "jit_batch", "block_n": 1024, "decode_alg": "splitk", "down_path": "default", "num_oc_splits": 1, "padding": null, "tile_m_down": 16, "tile_m_gate": 16, "tile_n_down": 64, "tile_n_gate": 32} |
| qwen35_35B | 1024 | 2048 / 512 / 256 / 8 | PASS/PASS | 0.000141325 | 0.000141348 | 196.96 | 198.06 | 0.994x | aiter | 260.218 | PASS | {"_impl": "aiter"} |
| qwen35_35B | 4096 | 2048 / 512 / 256 / 8 | PASS/PASS | 0.000143774 | 0.000132004 | 626.59 | 335.10 | 1.870x | fly_prefill | 615.206 | PASS | {"_impl": "fly_prefill", "block_n": 0, "decode_alg": "splitk", "down_path": "default", "num_oc_splits": 1, "padding": null, "tile_k_gate": 128, "tile_m_down": 128, "tile_m_gate": 128, "tile_n_down": 128, "tile_n_gate": 256} |
| qwen35_35B | 8192 | 2048 / 512 / 256 / 8 | PASS/PASS | 0.00014418 | 0.000131727 | 1191.24 | 535.87 | 2.223x | fly_prefill | 769.439 | PASS | {"_impl": "fly_prefill", "block_n": 0, "decode_alg": "splitk", "down_path": "1x4_64x256", "num_oc_splits": 1, "padding": 0, "tile_k_gate": 128, "tile_m_down": 64, "tile_m_gate": 64, "tile_n_down": 256, "tile_n_gate": 256} |
| qwen35_35B | 16384 | 2048 / 512 / 256 / 8 | PASS/PASS | 0.000144263 | 0.000131769 | 1367.28 | 958.99 | 1.426x | fly_prefill | 859.896 | PASS | {"_impl": "fly_prefill", "block_n": 0, "decode_alg": "splitk", "down_path": "1x4_64x256", "num_oc_splits": 1, "padding": 0, "tile_k_gate": 128, "tile_m_down": 64, "tile_m_gate": 64, "tile_n_down": 256, "tile_n_gate": 256} |
| qwen35_35B | 32768 | 2048 / 512 / 256 / 8 | PASS/PASS | 0.000144296 | 0.000131613 | 2559.93 | 1900.01 | 1.347x | fly_prefill | 868.033 | PASS | {"_impl": "fly_prefill", "block_n": 0, "decode_alg": "splitk", "down_path": "1x4_64x256", "num_oc_splits": 1, "padding": 0, "tile_k_gate": 128, "tile_m_down": 64, "tile_m_gate": 64, "tile_n_down": 256, "tile_n_gate": 256} |


### qwen35_35B_k256 (fp8/ptpc, silu, TP=1)

model_dim=2048, inter_dim=256, inter_dim_tp=256, experts=256, topk=8, gate_mode=separated, preshuffle=on, routing=balanced, seed=0

| model | M | H / I_tp / E / topk | check A/T | Aiter diff | winner diff | Aiter us | tuned us | speedup | winner | winner TFLOPS | status | winner config |
|---|---:|---|---|---:|---:|---:|---:|---:|---|---:|---|---|
| qwen35_35B_k256 | 1 | 2048 / 256 / 256 / 8 | PASS/PASS | 3.50533e-05 | 0.000941647 | 36.32 | 16.32 | 2.225x | jit_batch1 | 1.542 | PASS | {"_impl": "jit_batch1", "block_n": 0, "decode_alg": "splitk", "down_path": "default", "num_oc_splits": 1, "padding": null, "tile_m_down": 16, "tile_m_gate": 16, "tile_n_down": 32, "tile_n_gate": 32} |
| qwen35_35B_k256 | 4 | 2048 / 256 / 256 / 8 | PASS/PASS | 7.37696e-05 | 0.00104095 | 41.08 | 20.24 | 2.030x | fly_decode | 4.973 | PASS | {"_impl": "fly_decode", "block_n": 0, "decode_alg": "batch1", "down_path": "default", "num_oc_splits": 1, "padding": null, "tile_m_down": 16, "tile_m_gate": 16, "tile_n_down": 64, "tile_n_gate": 64} |
| qwen35_35B_k256 | 16 | 2048 / 256 / 256 / 8 | PASS/PASS | 9.57271e-05 | 0.00102565 | 71.88 | 48.74 | 1.475x | fly_decode | 8.261 | PASS | {"_impl": "fly_decode", "block_n": 0, "decode_alg": "batch1", "down_path": "default", "num_oc_splits": 1, "padding": null, "tile_m_down": 16, "tile_m_gate": 16, "tile_n_down": 64, "tile_n_gate": 32} |
| qwen35_35B_k256 | 64 | 2048 / 256 / 256 / 8 | PASS/PASS | 0.000141429 | 0.00102441 | 99.20 | 87.80 | 1.130x | jit_splitk | 18.344 | PASS | {"_impl": "jit_splitk", "block_n": 0, "decode_alg": "splitk", "down_path": "default", "num_oc_splits": 1, "padding": null, "tile_m_down": 32, "tile_m_gate": 32, "tile_n_down": 64, "tile_n_gate": 64} |
| qwen35_35B_k256 | 256 | 2048 / 256 / 256 / 8 | PASS/PASS | 0.000146829 | 0.00102944 | 102.84 | 98.00 | 1.049x | jit_batch | 65.738 | PASS | {"_impl": "jit_batch", "block_n": 1024, "decode_alg": "splitk", "down_path": "default", "num_oc_splits": 1, "padding": null, "tile_m_down": 16, "tile_m_gate": 16, "tile_n_down": 64, "tile_n_gate": 32} |
| qwen35_35B_k256 | 1024 | 2048 / 256 / 256 / 8 | PASS/PASS | 0.000144038 | 0.00014404 | 113.96 | 114.96 | 0.991x | aiter | 224.160 | PASS | {"_impl": "aiter"} |
| qwen35_35B_k256 | 4096 | 2048 / 256 / 256 / 8 | PASS/PASS | 0.000141667 | 0.000130029 | 338.26 | 217.50 | 1.555x | fly_prefill | 473.921 | PASS | {"_impl": "fly_prefill", "block_n": 0, "decode_alg": "splitk", "down_path": "default", "num_oc_splits": 1, "padding": null, "tile_k_gate": 256, "tile_m_down": 128, "tile_m_gate": 128, "tile_n_down": 128, "tile_n_gate": 128} |
| qwen35_35B_k256 | 8192 | 2048 / 256 / 256 / 8 | PASS/PASS | 0.000143119 | 0.000129982 | 626.37 | 381.88 | 1.640x | fly_prefill | 539.844 | PASS | {"_impl": "fly_prefill", "block_n": 0, "decode_alg": "splitk", "down_path": "default", "num_oc_splits": 1, "padding": null, "tile_k_gate": 128, "tile_m_down": 128, "tile_m_gate": 128, "tile_n_down": 128, "tile_n_gate": 256} |
| qwen35_35B_k256 | 16384 | 2048 / 256 / 256 / 8 | PASS/PASS | 0.00014179 | 0.000129834 | 870.77 | 677.47 | 1.285x | fly_prefill | 608.614 | PASS | {"_impl": "fly_prefill", "block_n": 0, "decode_alg": "splitk", "down_path": "8x1", "num_oc_splits": 1, "padding": 0, "tile_k_gate": 128, "tile_m_down": 256, "tile_m_gate": 64, "tile_n_down": 128, "tile_n_gate": 256} |
| qwen35_35B_k256 | 32768 | 2048 / 256 / 256 / 8 | PASS/PASS | 0.000142508 | 0.000130428 | 1665.88 | 1333.16 | 1.250x | fly_prefill | 618.557 | PASS | {"_impl": "fly_prefill", "block_n": 0, "decode_alg": "splitk", "down_path": "8x1", "num_oc_splits": 1, "padding": 0, "tile_k_gate": 128, "tile_m_down": 256, "tile_m_gate": 64, "tile_n_down": 128, "tile_n_gate": 256} |


### mimo_ptpc (fp8/ptpc, silu, TP=8)

model_dim=6144, inter_dim=2048, inter_dim_tp=256, experts=384, topk=8, gate_mode=separated, preshuffle=on, routing=balanced, seed=0

| model | M | H / I_tp / E / topk | check A/T | Aiter diff | winner diff | Aiter us | tuned us | speedup | winner | winner TFLOPS | status | winner config |
|---|---:|---|---|---:|---:|---:|---:|---:|---|---:|---|---|
| mimo_ptpc | 1 | 6144 / 256 / 384 / 8 | PASS/PASS | 0.000137444 | 0.000941675 | 46.88 | 22.72 | 2.063x | jit_batch1 | 3.323 | PASS | {"_impl": "jit_batch1", "block_n": 0, "decode_alg": "splitk", "down_path": "default", "num_oc_splits": 1, "padding": null, "tile_m_down": 16, "tile_m_gate": 16, "tile_n_down": 32, "tile_n_gate": 32} |
| mimo_ptpc | 4 | 6144 / 256 / 384 / 8 | PASS/PASS | 0.000125359 | 0.00095861 | 72.58 | 35.88 | 2.023x | fly_decode | 8.417 | PASS | {"_impl": "fly_decode", "block_n": 0, "decode_alg": "batch1", "down_path": "default", "num_oc_splits": 1, "padding": null, "tile_m_down": 16, "tile_m_gate": 16, "tile_n_down": 64, "tile_n_gate": 64} |
| mimo_ptpc | 16 | 6144 / 256 / 384 / 8 | PASS/PASS | 0.00013204 | 0.00103241 | 155.04 | 129.66 | 1.196x | fly_decode | 9.316 | PASS | {"_impl": "fly_decode", "block_n": 0, "decode_alg": "batch1", "down_path": "default", "num_oc_splits": 1, "padding": null, "tile_m_down": 16, "tile_m_gate": 16, "tile_n_down": 64, "tile_n_gate": 64} |
| mimo_ptpc | 64 | 6144 / 256 / 384 / 8 | PASS/PASS | 0.000146725 | 0.00101579 | 392.17 | 331.82 | 1.182x | jit_splitk | 14.561 | PASS | {"_impl": "jit_splitk", "block_n": 0, "decode_alg": "splitk", "down_path": "default", "num_oc_splits": 1, "padding": null, "tile_m_down": 32, "tile_m_gate": 32, "tile_n_down": 64, "tile_n_gate": 64} |
| mimo_ptpc | 256 | 6144 / 256 / 384 / 8 | PASS/PASS | 0.000147531 | 0.00102775 | 407.57 | 355.44 | 1.147x | jit_splitk | 54.375 | PASS | {"_impl": "jit_splitk", "block_n": 0, "decode_alg": "splitk", "down_path": "default", "num_oc_splits": 1, "padding": null, "tile_m_down": 16, "tile_m_gate": 16, "tile_n_down": 128, "tile_n_gate": 128} |
| mimo_ptpc | 1024 | 6144 / 256 / 384 / 8 | PASS/PASS | 0.000138804 | 0.000138816 | 452.55 | 453.45 | 0.998x | aiter | 170.493 | PASS | {"_impl": "aiter"} |
| mimo_ptpc | 4096 | 6144 / 256 / 384 / 8 | PASS/PASS | 0.000142529 | 0.000130266 | 1152.52 | 761.95 | 1.513x | fly_prefill | 405.850 | PASS | {"_impl": "fly_prefill", "block_n": 0, "decode_alg": "splitk", "down_path": "default", "num_oc_splits": 1, "padding": null, "tile_k_gate": 128, "tile_m_down": 128, "tile_m_gate": 128, "tile_n_down": 128, "tile_n_gate": 256} |
| mimo_ptpc | 8192 | 6144 / 256 / 384 / 8 | PASS/PASS | 0.000141865 | 0.000129545 | 2214.11 | 1327.14 | 1.668x | fly_prefill | 466.022 | PASS | {"_impl": "fly_prefill", "block_n": 0, "decode_alg": "splitk", "down_path": "default", "num_oc_splits": 1, "padding": null, "tile_k_gate": 128, "tile_m_down": 128, "tile_m_gate": 128, "tile_n_down": 128, "tile_n_gate": 256} |
| mimo_ptpc | 16384 | 6144 / 256 / 384 / 8 | PASS/PASS | 0.000142612 | 0.000130514 | 3305.18 | 2114.15 | 1.563x | fly_prefill | 585.082 | PASS | {"_impl": "fly_prefill", "block_n": 0, "decode_alg": "splitk", "down_path": "default", "num_oc_splits": 1, "padding": null, "tile_k_gate": 128, "tile_m_down": 128, "tile_m_gate": 128, "tile_n_down": 128, "tile_n_gate": 256} |
| mimo_ptpc | 32768 | 6144 / 256 / 384 / 8 | PASS/PASS | 0.00014078 | 0.000129522 | 5359.17 | 3754.79 | 1.427x | fly_prefill | 658.865 | PASS | {"_impl": "fly_prefill", "block_n": 0, "decode_alg": "splitk", "down_path": "8x1_compact", "num_oc_splits": 1, "padding": 128, "tile_k_gate": 128, "tile_m_down": 64, "tile_m_gate": 64, "tile_n_down": 128, "tile_n_gate": 256} |


### mimo_block (fp8/block, silu, TP=8)

model_dim=6144, inter_dim=2048, inter_dim_tp=256, experts=384, topk=8, gate_mode=separated, preshuffle=on, routing=balanced, seed=0

| model | M | H / I_tp / E / topk | check A/T | Aiter diff | winner diff | Aiter us | tuned us | speedup | winner | winner TFLOPS | status | winner config |
|---|---:|---|---|---:|---:|---:|---:|---:|---|---:|---|---|
| mimo_block | 1 | 6144 / 256 / 384 / 8 | PASS/PASS | 7.69393e-06 | 3.64983e-06 | 37.94 | 83.62 | 0.454x | jit_blockscale | 0.903 | PASS | {"_impl": "jit_blockscale", "block_n": 0, "decode_alg": "splitk", "down_path": "default", "num_oc_splits": 1, "padding": null, "tile_m_down": 128, "tile_m_gate": 128, "tile_n_down": 256, "tile_n_gate": 256} |
| mimo_block | 4 | 6144 / 256 / 384 / 8 | PASS/PASS | 1.44203e-05 | 3.54731e-06 | 58.20 | 103.54 | 0.562x | jit_blockscale | 2.917 | PASS | {"_impl": "jit_blockscale", "block_n": 0, "decode_alg": "splitk", "down_path": "default", "num_oc_splits": 1, "padding": null, "tile_m_down": 128, "tile_m_gate": 128, "tile_n_down": 128, "tile_n_gate": 256} |
| mimo_block | 16 | 6144 / 256 / 384 / 8 | PASS/PASS | 8.8356e-06 | 8.80083e-06 | 135.44 | 135.18 | 1.002x | aiter | 8.936 | PASS | {"_impl": "aiter"} |
| mimo_block | 64 | 6144 / 256 / 384 / 8 | PASS/PASS | 8.94516e-05 | 8.94152e-05 | 379.75 | 380.87 | 0.997x | aiter | 12.686 | PASS | {"_impl": "aiter"} |
| mimo_block | 256 | 6144 / 256 / 384 / 8 | PASS/PASS | 8.27049e-05 | 8.27432e-05 | 398.85 | 397.01 | 1.005x | aiter | 48.683 | PASS | {"_impl": "aiter"} |
| mimo_block | 1024 | 6144 / 256 / 384 / 8 | PASS/PASS | 7.99407e-05 | 7.99468e-05 | 452.77 | 452.81 | 1.000x | aiter | 170.734 | PASS | {"_impl": "aiter"} |
| mimo_block | 4096 | 6144 / 256 / 384 / 8 | PASS/PASS | 8.11659e-05 | 4.35524e-06 | 1087.87 | 737.67 | 1.475x | jit_blockscale | 419.209 | PASS | {"_impl": "jit_blockscale", "block_n": 0, "decode_alg": "splitk", "down_path": "persistent", "num_oc_splits": 2, "padding": null, "tile_m_down": 256, "tile_m_gate": 256, "tile_n_down": 64, "tile_n_gate": 256} |
| mimo_block | 8192 | 6144 / 256 / 384 / 8 | PASS/PASS | 8.08592e-05 | 4.32391e-06 | 2069.25 | 1003.05 | 2.063x | jit_blockscale | 616.593 | PASS | {"_impl": "jit_blockscale", "block_n": 0, "decode_alg": "splitk", "down_path": "persistent", "num_oc_splits": 2, "padding": null, "tile_m_down": 256, "tile_m_gate": 256, "tile_n_down": 64, "tile_n_gate": 256} |
| mimo_block | 16384 | 6144 / 256 / 384 / 8 | PASS/PASS | 8.06159e-05 | 4.30074e-06 | 3449.13 | 1762.72 | 1.957x | jit_blockscale | 701.727 | PASS | {"_impl": "jit_blockscale", "block_n": 0, "decode_alg": "splitk", "down_path": "persistent", "num_oc_splits": 2, "padding": null, "tile_m_down": 256, "tile_m_gate": 256, "tile_n_down": 64, "tile_n_gate": 256} |
| mimo_block | 32768 | 6144 / 256 / 384 / 8 | PASS/PASS | 8.06384e-05 | 4.35538e-06 | 5879.12 | 3449.37 | 1.704x | jit_blockscale | 717.205 | PASS | {"_impl": "jit_blockscale", "block_n": 0, "decode_alg": "splitk", "down_path": "persistent", "num_oc_splits": 1, "padding": null, "tile_m_down": 256, "tile_m_gate": 256, "tile_n_down": 64, "tile_n_gate": 256} |


### h3 (fp8/ptpc, silu, TP=8)

model_dim=6144, inter_dim=3072, inter_dim_tp=384, experts=128, topk=4, gate_mode=separated, preshuffle=on, routing=balanced, seed=0

| model | M | H / I_tp / E / topk | check A/T | Aiter diff | winner diff | Aiter us | tuned us | speedup | winner | winner TFLOPS | status | winner config |
|---|---:|---|---|---:|---:|---:|---:|---:|---|---:|---|---|
| h3 | 1 | 6144 / 384 / 128 / 4 | ERROR/PASS | — | 0.00108021 | — | — | — | jit_batch1 | — | NOT_COMPARABLE | {"_impl": "jit_batch1", "block_n": 0, "decode_alg": "splitk", "down_path": "default", "num_oc_splits": 1, "padding": null, "tile_m_down": 16, "tile_m_gate": 16, "tile_n_down": 32, "tile_n_gate": 32} |
| h3 | 4 | 6144 / 384 / 128 / 4 | ERROR/PASS | — | 0.00109037 | — | — | — | fly_decode | — | NOT_COMPARABLE | {"_impl": "fly_decode", "block_n": 0, "decode_alg": "batch1", "down_path": "default", "num_oc_splits": 1, "padding": null, "tile_m_down": 16, "tile_m_gate": 16, "tile_n_down": 64, "tile_n_gate": 64} |
| h3 | 16 | 6144 / 384 / 128 / 4 | ERROR/PASS | — | 0.00107367 | — | — | — | fly_decode | — | NOT_COMPARABLE | {"_impl": "fly_decode", "block_n": 0, "decode_alg": "batch1", "down_path": "default", "num_oc_splits": 1, "padding": null, "tile_m_down": 16, "tile_m_gate": 16, "tile_n_down": 64, "tile_n_gate": 32} |
| h3 | 64 | 6144 / 384 / 128 / 4 | PASS/PASS | 0.000145288 | 0.00103737 | 205.98 | 174.70 | 1.179x | jit_splitk | 20.743 | PASS | {"_impl": "jit_splitk", "block_n": 0, "decode_alg": "splitk", "down_path": "default", "num_oc_splits": 1, "padding": null, "tile_m_down": 32, "tile_m_gate": 32, "tile_n_down": 64, "tile_n_gate": 64} |
| h3 | 256 | 6144 / 384 / 128 / 4 | PASS/PASS | 0.000136409 | 0.00103722 | 219.92 | 194.66 | 1.130x | jit_splitk | 74.465 | PASS | {"_impl": "jit_splitk", "block_n": 0, "decode_alg": "splitk", "down_path": "default", "num_oc_splits": 1, "padding": null, "tile_m_down": 16, "tile_m_gate": 16, "tile_n_down": 64, "tile_n_gate": 64} |
| h3 | 1024 | 6144 / 384 / 128 / 4 | PASS/PASS | 0.000142958 | 0.000143 | 240.62 | 238.92 | 1.007x | aiter | 242.681 | PASS | {"_impl": "aiter"} |
| h3 | 4096 | 6144 / 384 / 128 / 4 | PASS/PASS | 0.000141088 | 0.000131323 | 711.41 | 431.99 | 1.647x | fly_prefill | 536.888 | PASS | {"_impl": "fly_prefill", "block_n": 0, "decode_alg": "splitk", "down_path": "default", "num_oc_splits": 1, "padding": null, "tile_k_gate": 128, "tile_m_down": 128, "tile_m_gate": 128, "tile_n_down": 128, "tile_n_gate": 256} |
| h3 | 8192 | 6144 / 384 / 128 / 4 | PASS/PASS | 0.000139954 | 0.000132031 | 1348.44 | 674.61 | 1.999x | fly_prefill | 687.593 | PASS | {"_impl": "fly_prefill", "block_n": 0, "decode_alg": "splitk", "down_path": "default", "num_oc_splits": 1, "padding": null, "tile_k_gate": 128, "tile_m_down": 128, "tile_m_gate": 128, "tile_n_down": 128, "tile_n_gate": 256} |
| h3 | 16384 | 6144 / 384 / 128 / 4 | PASS/PASS | 0.000140426 | 0.000130993 | 1786.90 | 1356.60 | 1.317x | fly_prefill | 683.852 | PASS | {"_impl": "fly_prefill", "block_n": 0, "decode_alg": "splitk", "down_path": "default", "num_oc_splits": 1, "padding": null, "tile_k_gate": 128, "tile_m_down": 128, "tile_m_gate": 128, "tile_n_down": 128, "tile_n_gate": 256} |
| h3 | 32768 | 6144 / 384 / 128 / 4 | PASS/PASS | 0.000140815 | 0.000132039 | 3348.66 | 2657.08 | 1.260x | fly_prefill | 698.296 | PASS | {"_impl": "fly_prefill", "block_n": 0, "decode_alg": "splitk", "down_path": "8x1", "num_oc_splits": 1, "padding": 128, "tile_k_gate": 128, "tile_m_down": 256, "tile_m_gate": 64, "tile_n_down": 128, "tile_n_gate": 256} |

[h3 M=1] Performance comparison skipped: Aiter must execute with valid output metadata and winner must pass accuracy validation.
[h3 M=1] aiter: ERROR: wrong! device_gemm with the specified compilation parameters does not support this GEMM problem
[h3 M=4] Performance comparison skipped: Aiter must execute with valid output metadata and winner must pass accuracy validation.
[h3 M=4] aiter: ERROR: wrong! device_gemm with the specified compilation parameters does not support this GEMM problem
[h3 M=16] Performance comparison skipped: Aiter must execute with valid output metadata and winner must pass accuracy validation.
[h3 M=16] aiter: ERROR: wrong! device_gemm with the specified compilation parameters does not support this GEMM problem
