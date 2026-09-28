# MoE performance

Winner TFLOPS uses the measured full tuned-call median latency.

### hy3 (fp8/per_tensor, silu, TP=8)

model_dim=4096, inter_dim=1536, inter_dim_tp=192, experts=193, topk=9, gate_mode=separated, preshuffle=on, routing=balanced, seed=0

| model | M | H / I_tp / E / topk | check A/T | Aiter diff | winner diff | Aiter us | tuned us | speedup | winner | winner TFLOPS | status | winner config |
|---|---:|---|---|---:|---:|---:|---:|---:|---|---:|---|---|
| hy3 | 1 | 4096 / 192 / 193 / 9 | INCORRECT/PASS | 0.95666 | 0.00104633 | 56.10 | 26.20 | 2.141x | fly_decode | 1.621 | AITER_INCORRECT | {"_impl": "fly_decode", "block_n": 0, "decode_alg": "batch1", "down_path": "default", "num_oc_splits": 1, "padding": null, "tile_m_down": 16, "tile_m_gate": 16, "tile_n_down": 64, "tile_n_gate": 64} |
| hy3 | 4 | 4096 / 192 / 193 / 9 | INCORRECT/PASS | 0.93435 | 0.001187 | 77.06 | 35.10 | 2.195x | fly_decode | 4.840 | AITER_INCORRECT | {"_impl": "fly_decode", "block_n": 0, "decode_alg": "batch1", "down_path": "default", "num_oc_splits": 1, "padding": null, "tile_m_down": 16, "tile_m_gate": 16, "tile_n_down": 64, "tile_n_gate": 64} |
| hy3 | 16 | 4096 / 192 / 193 / 9 | INCORRECT/PASS | 0.940583 | 0.00109114 | 197.84 | 107.06 | 1.848x | fly_decode | 6.347 | AITER_INCORRECT | {"_impl": "fly_decode", "block_n": 0, "decode_alg": "batch1", "down_path": "default", "num_oc_splits": 1, "padding": null, "tile_m_down": 16, "tile_m_gate": 16, "tile_n_down": 64, "tile_n_gate": 64} |
| hy3 | 64 | 4096 / 192 / 193 / 9 | INCORRECT/PASS | 0.938015 | 0.00106375 | 661.12 | 337.24 | 1.960x | fly_decode | 8.059 | AITER_INCORRECT | {"_impl": "fly_decode", "block_n": 0, "decode_alg": "splitk", "down_path": "default", "num_oc_splits": 1, "padding": null, "tile_m_down": 16, "tile_m_gate": 16, "tile_n_down": 128, "tile_n_gate": 128} |
| hy3 | 256 | 4096 / 192 / 193 / 9 | INCORRECT/PASS | 0.936552 | 0.00107777 | 392.14 | 406.66 | 0.964x | fly_decode | 26.734 | AITER_INCORRECT | {"_impl": "fly_decode", "block_n": 0, "decode_alg": "splitk", "down_path": "default", "num_oc_splits": 1, "padding": null, "tile_m_down": 16, "tile_m_gate": 16, "tile_n_down": 128, "tile_n_gate": 128} |
| hy3 | 1024 | 4096 / 192 / 193 / 9 | INCORRECT/PASS | 0.93695 | 7.29223e-05 | 809.56 | 670.88 | 1.207x | fly_prefill | 64.820 | AITER_INCORRECT | {"_impl": "fly_prefill", "block_n": 0, "decode_alg": "splitk", "down_path": "1x4_64x256", "num_oc_splits": 1, "padding": 128, "tile_k_gate": 256, "tile_m_down": 64, "tile_m_gate": 64, "tile_n_down": 256, "tile_n_gate": 128} |
| hy3 | 4096 | 4096 / 192 / 193 / 9 | INCORRECT/PASS | 0.937608 | 0.00010365 | 1843.35 | 1347.93 | 1.368x | fly_prefill | 129.047 | AITER_INCORRECT | {"_impl": "fly_prefill", "block_n": 0, "decode_alg": "splitk", "down_path": "1x4_64x256", "num_oc_splits": 1, "padding": 128, "tile_k_gate": 256, "tile_m_down": 64, "tile_m_gate": 64, "tile_n_down": 256, "tile_n_gate": 128} |
| hy3 | 8192 | 4096 / 192 / 193 / 9 | INCORRECT/PASS | 0.937209 | 0.000105146 | 3110.89 | 2274.21 | 1.368x | fly_prefill | 152.973 | AITER_INCORRECT | {"_impl": "fly_prefill", "block_n": 0, "decode_alg": "splitk", "down_path": "8x1_compact", "num_oc_splits": 1, "padding": 128, "tile_k_gate": 256, "tile_m_down": 64, "tile_m_gate": 64, "tile_n_down": 128, "tile_n_gate": 128} |
| hy3 | 16384 | 4096 / 192 / 193 / 9 | INCORRECT/PASS | 0.937443 | 0.000118918 | 6203.94 | 4078.10 | 1.521x | fly_prefill | 170.615 | AITER_INCORRECT | {"_impl": "fly_prefill", "block_n": 0, "decode_alg": "splitk", "down_path": "8x1", "num_oc_splits": 1, "padding": 128, "tile_k_gate": 256, "tile_m_down": 256, "tile_m_gate": 64, "tile_n_down": 128, "tile_n_gate": 128} |
| hy3 | 32768 | 4096 / 192 / 193 / 9 | INCORRECT/PASS | 0.937116 | 0.00011883 | 12518.01 | 8039.61 | 1.557x | fly_prefill | 173.089 | AITER_INCORRECT | {"_impl": "fly_prefill", "block_n": 0, "decode_alg": "splitk", "down_path": "8x1_compact", "num_oc_splits": 1, "padding": 0, "tile_k_gate": 256, "tile_m_down": 64, "tile_m_gate": 64, "tile_n_down": 128, "tile_n_gate": 128} |

[hy3 M=1] Aiter accuracy failed; speedup compares timings only, not equivalent correct results.
[hy3 M=1] aiter: INCORRECT: calc_diff=0.9566600121507384
[hy3 M=4] Aiter accuracy failed; speedup compares timings only, not equivalent correct results.
[hy3 M=4] aiter: INCORRECT: calc_diff=0.9343504904620873
[hy3 M=16] Aiter accuracy failed; speedup compares timings only, not equivalent correct results.
[hy3 M=16] aiter: INCORRECT: calc_diff=0.9405834567579081
[hy3 M=64] Aiter accuracy failed; speedup compares timings only, not equivalent correct results.
[hy3 M=64] aiter: INCORRECT: calc_diff=0.9380154227133377
[hy3 M=256] Aiter accuracy failed; speedup compares timings only, not equivalent correct results.
[hy3 M=256] aiter: INCORRECT: calc_diff=0.9365516197736059
[hy3 M=1024] Aiter accuracy failed; speedup compares timings only, not equivalent correct results.
[hy3 M=1024] aiter: INCORRECT: calc_diff=0.9369502694953032
[hy3 M=4096] Aiter accuracy failed; speedup compares timings only, not equivalent correct results.
[hy3 M=4096] aiter: INCORRECT: calc_diff=0.9376083046397582
[hy3 M=8192] Aiter accuracy failed; speedup compares timings only, not equivalent correct results.
[hy3 M=8192] aiter: INCORRECT: calc_diff=0.9372094542290121
[hy3 M=16384] Aiter accuracy failed; speedup compares timings only, not equivalent correct results.
[hy3 M=16384] aiter: INCORRECT: calc_diff=0.9374431300709943
[hy3 M=32768] Aiter accuracy failed; speedup compares timings only, not equivalent correct results.
[hy3 M=32768] aiter: INCORRECT: calc_diff=0.9371159199299413

### qwen35_397B (fp8/ptpc, silu, TP=8)

model_dim=4096, inter_dim=4096, inter_dim_tp=512, experts=512, topk=10, gate_mode=separated, preshuffle=on, routing=balanced, seed=0

| model | M | H / I_tp / E / topk | check A/T | Aiter diff | winner diff | Aiter us | tuned us | speedup | winner | winner TFLOPS | status | winner config |
|---|---:|---|---|---:|---:|---:|---:|---:|---|---:|---|---|
| qwen35_397B | 1 | 4096 / 512 / 512 / 10 | PASS/PASS | 0.000144612 | 0.00116139 | 42.58 | 30.34 | 1.403x | jit_batch1 | 4.147 | PASS | {"_impl": "jit_batch1", "block_n": 0, "decode_alg": "splitk", "down_path": "default", "num_oc_splits": 1, "padding": null, "tile_m_down": 16, "tile_m_gate": 16, "tile_n_down": 32, "tile_n_gate": 32} |
| qwen35_397B | 4 | 4096 / 512 / 512 / 10 | PASS/PASS | 0.000135958 | 0.00107923 | 88.78 | 65.44 | 1.357x | fly_decode | 7.691 | PASS | {"_impl": "fly_decode", "block_n": 0, "decode_alg": "batch1", "down_path": "default", "num_oc_splits": 1, "padding": null, "tile_m_down": 16, "tile_m_gate": 16, "tile_n_down": 64, "tile_n_gate": 64} |
| qwen35_397B | 16 | 4096 / 512 / 512 / 10 | PASS/PASS | 0.000147817 | 0.00107132 | 307.76 | 293.56 | 1.048x | fly_decode | 6.858 | PASS | {"_impl": "fly_decode", "block_n": 0, "decode_alg": "batch1", "down_path": "default", "num_oc_splits": 1, "padding": null, "tile_m_down": 16, "tile_m_gate": 16, "tile_n_down": 64, "tile_n_gate": 64} |
| qwen35_397B | 64 | 4096 / 512 / 512 / 10 | PASS/PASS | 0.00014458 | 0.00105585 | 952.44 | 936.30 | 1.017x | jit_splitk | 8.601 | PASS | {"_impl": "jit_splitk", "block_n": 0, "decode_alg": "splitk", "down_path": "default", "num_oc_splits": 1, "padding": null, "tile_m_down": 16, "tile_m_gate": 16, "tile_n_down": 64, "tile_n_gate": 64} |
| qwen35_397B | 256 | 4096 / 512 / 512 / 10 | PASS/PASS | 0.000143285 | 0.00105269 | 967.54 | 1046.02 | 0.925x | jit_splitk | 30.795 | PASS | {"_impl": "jit_splitk", "block_n": 0, "decode_alg": "splitk", "down_path": "default", "num_oc_splits": 1, "padding": null, "tile_m_down": 16, "tile_m_gate": 16, "tile_n_down": 128, "tile_n_gate": 128} |
| qwen35_397B | 1024 | 4096 / 512 / 512 / 10 | PASS/PASS | 0.000147099 | 0.000147077 | 1039.06 | 1209.34 | 0.859x | aiter | 106.545 | PASS | {"_impl": "aiter"} |
| qwen35_397B | 4096 | 4096 / 512 / 512 / 10 | PASS/PASS | 0.000143906 | 0.000130052 | 2633.91 | 2576.51 | 1.022x | fly_prefill | 200.037 | PASS | {"_impl": "fly_prefill", "block_n": 0, "decode_alg": "splitk", "down_path": "default", "num_oc_splits": 1, "padding": null, "tile_k_gate": 128, "tile_m_down": 128, "tile_m_gate": 128, "tile_n_down": 128, "tile_n_gate": 128} |
| qwen35_397B | 8192 | 4096 / 512 / 512 / 10 | PASS/PASS | 0.000144682 | 0.000132704 | 4093.62 | 3720.61 | 1.100x | fly_prefill | 277.049 | PASS | {"_impl": "fly_prefill", "block_n": 0, "decode_alg": "splitk", "down_path": "default", "num_oc_splits": 1, "padding": null, "tile_k_gate": 128, "tile_m_down": 64, "tile_m_gate": 64, "tile_n_down": 128, "tile_n_gate": 256} |
| qwen35_397B | 16384 | 4096 / 512 / 512 / 10 | PASS/PASS | 0.00014461 | 0.000131761 | 7475.23 | 6260.74 | 1.194x | fly_prefill | 329.287 | PASS | {"_impl": "fly_prefill", "block_n": 0, "decode_alg": "splitk", "down_path": "default", "num_oc_splits": 1, "padding": null, "tile_k_gate": 128, "tile_m_down": 64, "tile_m_gate": 64, "tile_n_down": 128, "tile_n_gate": 256} |
| qwen35_397B | 32768 | 4096 / 512 / 512 / 10 | PASS/PASS | 0.000143662 | 0.000131372 | 14399.57 | 12213.63 | 1.179x | fly_prefill | 337.588 | PASS | {"_impl": "fly_prefill", "block_n": 0, "decode_alg": "splitk", "down_path": "default", "num_oc_splits": 1, "padding": null, "tile_k_gate": 128, "tile_m_down": 64, "tile_m_gate": 64, "tile_n_down": 128, "tile_n_gate": 256} |


### qwen35_397B_k256 (fp8/ptpc, silu, TP=8)

model_dim=4096, inter_dim=2048, inter_dim_tp=256, experts=512, topk=10, gate_mode=separated, preshuffle=on, routing=balanced, seed=0

| model | M | H / I_tp / E / topk | check A/T | Aiter diff | winner diff | Aiter us | tuned us | speedup | winner | winner TFLOPS | status | winner config |
|---|---:|---|---|---:|---:|---:|---:|---:|---|---:|---|---|
| qwen35_397B_k256 | 1 | 4096 / 256 / 512 / 10 | PASS/PASS | 0.000127828 | 0.00112289 | 40.20 | 21.14 | 1.902x | jit_batch1 | 2.976 | PASS | {"_impl": "jit_batch1", "block_n": 0, "decode_alg": "splitk", "down_path": "default", "num_oc_splits": 1, "padding": null, "tile_m_down": 16, "tile_m_gate": 16, "tile_n_down": 32, "tile_n_gate": 32} |
| qwen35_397B_k256 | 4 | 4096 / 256 / 512 / 10 | PASS/PASS | 0.000158943 | 0.0011387 | 66.08 | 41.08 | 1.609x | fly_decode | 6.126 | PASS | {"_impl": "fly_decode", "block_n": 0, "decode_alg": "batch1", "down_path": "default", "num_oc_splits": 1, "padding": null, "tile_m_down": 16, "tile_m_gate": 16, "tile_n_down": 64, "tile_n_gate": 64} |
| qwen35_397B_k256 | 16 | 4096 / 256 / 512 / 10 | PASS/PASS | 0.000120808 | 0.00106991 | 198.44 | 161.08 | 1.232x | fly_decode | 6.249 | PASS | {"_impl": "fly_decode", "block_n": 0, "decode_alg": "batch1", "down_path": "default", "num_oc_splits": 1, "padding": null, "tile_m_down": 16, "tile_m_gate": 16, "tile_n_down": 64, "tile_n_gate": 64} |
| qwen35_397B_k256 | 64 | 4096 / 256 / 512 / 10 | PASS/PASS | 0.000137455 | 0.0010629 | 549.60 | 586.78 | 0.937x | jit_batch | 6.862 | PASS | {"_impl": "jit_batch", "block_n": 1024, "decode_alg": "splitk", "down_path": "default", "num_oc_splits": 1, "padding": null, "tile_m_down": 16, "tile_m_gate": 16, "tile_n_down": 64, "tile_n_gate": 32} |
| qwen35_397B_k256 | 256 | 4096 / 256 / 512 / 10 | PASS/PASS | 0.000140189 | 0.00103035 | 563.08 | 631.86 | 0.891x | jit_splitk | 25.490 | PASS | {"_impl": "jit_splitk", "block_n": 0, "decode_alg": "splitk", "down_path": "default", "num_oc_splits": 1, "padding": null, "tile_m_down": 16, "tile_m_gate": 16, "tile_n_down": 128, "tile_n_gate": 128} |
| qwen35_397B_k256 | 1024 | 4096 / 256 / 512 / 10 | PASS/PASS | 0.000138937 | 0.000138912 | 606.82 | 790.90 | 0.767x | aiter | 81.457 | PASS | {"_impl": "aiter"} |
| qwen35_397B_k256 | 4096 | 4096 / 256 / 512 / 10 | PASS/PASS | 0.000140242 | 0.000127486 | 1518.73 | 1632.67 | 0.930x | fly_prefill | 157.839 | PASS | {"_impl": "fly_prefill", "block_n": 0, "decode_alg": "splitk", "down_path": "default", "num_oc_splits": 1, "padding": null, "tile_k_gate": 128, "tile_m_down": 128, "tile_m_gate": 128, "tile_n_down": 128, "tile_n_gate": 128} |
| qwen35_397B_k256 | 8192 | 4096 / 256 / 512 / 10 | PASS/PASS | 0.000141216 | 0.000128654 | 2438.65 | 2443.13 | 0.998x | fly_prefill | 210.957 | PASS | {"_impl": "fly_prefill", "block_n": 0, "decode_alg": "splitk", "down_path": "default", "num_oc_splits": 1, "padding": null, "tile_k_gate": 128, "tile_m_down": 64, "tile_m_gate": 64, "tile_n_down": 128, "tile_n_gate": 256} |
| qwen35_397B_k256 | 16384 | 4096 / 256 / 512 / 10 | PASS/PASS | 0.000139797 | 0.000128204 | 4617.68 | 3750.33 | 1.231x | fly_prefill | 274.853 | PASS | {"_impl": "fly_prefill", "block_n": 0, "decode_alg": "splitk", "down_path": "8x1_compact", "num_oc_splits": 1, "padding": 128, "tile_k_gate": 128, "tile_m_down": 64, "tile_m_gate": 64, "tile_n_down": 128, "tile_n_gate": 256} |
| qwen35_397B_k256 | 32768 | 4096 / 256 / 512 / 10 | PASS/PASS | 0.000141137 | 0.000129235 | 9165.31 | 7380.99 | 1.242x | fly_prefill | 279.310 | PASS | {"_impl": "fly_prefill", "block_n": 0, "decode_alg": "splitk", "down_path": "8x1_compact", "num_oc_splits": 1, "padding": 128, "tile_k_gate": 128, "tile_m_down": 64, "tile_m_gate": 64, "tile_n_down": 128, "tile_n_gate": 256} |


### qwen35_35B (fp8/ptpc, silu, TP=1)

model_dim=2048, inter_dim=512, inter_dim_tp=512, experts=256, topk=8, gate_mode=separated, preshuffle=on, routing=balanced, seed=0

| model | M | H / I_tp / E / topk | check A/T | Aiter diff | winner diff | Aiter us | tuned us | speedup | winner | winner TFLOPS | status | winner config |
|---|---:|---|---|---:|---:|---:|---:|---:|---|---:|---|---|
| qwen35_35B | 1 | 2048 / 512 / 256 / 8 | PASS/PASS | 0.000111937 | 0.00102513 | 35.30 | 20.72 | 1.704x | jit_batch1 | 2.429 | PASS | {"_impl": "jit_batch1", "block_n": 0, "decode_alg": "splitk", "down_path": "default", "num_oc_splits": 1, "padding": null, "tile_m_down": 16, "tile_m_gate": 16, "tile_n_down": 32, "tile_n_gate": 32} |
| qwen35_35B | 4 | 2048 / 512 / 256 / 8 | PASS/PASS | 0.000108648 | 0.00100963 | 60.44 | 39.52 | 1.529x | fly_decode | 5.094 | PASS | {"_impl": "fly_decode", "block_n": 0, "decode_alg": "batch1", "down_path": "default", "num_oc_splits": 1, "padding": null, "tile_m_down": 16, "tile_m_gate": 16, "tile_n_down": 64, "tile_n_gate": 64} |
| qwen35_35B | 16 | 2048 / 512 / 256 / 8 | PASS/PASS | 0.000133737 | 0.00106182 | 255.82 | 127.32 | 2.009x | fly_decode | 6.325 | PASS | {"_impl": "fly_decode", "block_n": 0, "decode_alg": "batch1", "down_path": "default", "num_oc_splits": 1, "padding": null, "tile_m_down": 16, "tile_m_gate": 16, "tile_n_down": 64, "tile_n_gate": 64} |
| qwen35_35B | 64 | 2048 / 512 / 256 / 8 | PASS/PASS | 0.000146101 | 0.00107717 | 314.20 | 346.00 | 0.908x | jit_splitk | 9.310 | PASS | {"_impl": "jit_splitk", "block_n": 0, "decode_alg": "splitk", "down_path": "default", "num_oc_splits": 1, "padding": null, "tile_m_down": 16, "tile_m_gate": 16, "tile_n_down": 64, "tile_n_gate": 64} |
| qwen35_35B | 256 | 2048 / 512 / 256 / 8 | PASS/PASS | 0.000148075 | 0.00105954 | 319.26 | 387.56 | 0.824x | jit_splitk | 33.246 | PASS | {"_impl": "jit_splitk", "block_n": 0, "decode_alg": "splitk", "down_path": "default", "num_oc_splits": 1, "padding": null, "tile_m_down": 16, "tile_m_gate": 16, "tile_n_down": 128, "tile_n_gate": 128} |
| qwen35_35B | 1024 | 2048 / 512 / 256 / 8 | PASS/PASS | 0.000141536 | 0.000141552 | 337.40 | 530.06 | 0.637x | aiter | 97.233 | PASS | {"_impl": "aiter"} |
| qwen35_35B | 4096 | 2048 / 512 / 256 / 8 | PASS/PASS | 0.000143057 | 0.000130314 | 916.06 | 968.04 | 0.946x | fly_prefill | 212.964 | PASS | {"_impl": "fly_prefill", "block_n": 0, "decode_alg": "splitk", "down_path": "default", "num_oc_splits": 1, "padding": null, "tile_k_gate": 128, "tile_m_down": 64, "tile_m_gate": 64, "tile_n_down": 128, "tile_n_gate": 256} |
| qwen35_35B | 8192 | 2048 / 512 / 256 / 8 | PASS/PASS | 0.000142085 | 0.000131208 | 1565.95 | 1566.33 | 1.000x | fly_prefill | 263.238 | PASS | {"_impl": "fly_prefill", "block_n": 0, "decode_alg": "splitk", "down_path": "default", "num_oc_splits": 1, "padding": null, "tile_k_gate": 128, "tile_m_down": 64, "tile_m_gate": 64, "tile_n_down": 128, "tile_n_gate": 256} |
| qwen35_35B | 16384 | 2048 / 512 / 256 / 8 | PASS/PASS | 0.000141337 | 0.000130587 | 3071.73 | 2710.59 | 1.133x | fly_prefill | 304.227 | PASS | {"_impl": "fly_prefill", "block_n": 0, "decode_alg": "splitk", "down_path": "default", "num_oc_splits": 1, "padding": null, "tile_k_gate": 128, "tile_m_down": 64, "tile_m_gate": 64, "tile_n_down": 128, "tile_n_gate": 256} |
| qwen35_35B | 32768 | 2048 / 512 / 256 / 8 | PASS/PASS | 0.000142041 | 0.000130919 | 5697.64 | 5174.92 | 1.101x | fly_prefill | 318.704 | PASS | {"_impl": "fly_prefill", "block_n": 0, "decode_alg": "splitk", "down_path": "default", "num_oc_splits": 1, "padding": null, "tile_k_gate": 128, "tile_m_down": 64, "tile_m_gate": 64, "tile_n_down": 128, "tile_n_gate": 256} |


### qwen35_35B_k256 (fp8/ptpc, silu, TP=1)

model_dim=2048, inter_dim=256, inter_dim_tp=256, experts=256, topk=8, gate_mode=separated, preshuffle=on, routing=balanced, seed=0

| model | M | H / I_tp / E / topk | check A/T | Aiter diff | winner diff | Aiter us | tuned us | speedup | winner | winner TFLOPS | status | winner config |
|---|---:|---|---|---:|---:|---:|---:|---:|---|---:|---|---|
| qwen35_35B_k256 | 1 | 2048 / 256 / 256 / 8 | PASS/PASS | 0.000155563 | 0.00103166 | 37.80 | 16.42 | 2.302x | jit_batch1 | 1.533 | PASS | {"_impl": "jit_batch1", "block_n": 0, "decode_alg": "splitk", "down_path": "default", "num_oc_splits": 1, "padding": null, "tile_m_down": 16, "tile_m_gate": 16, "tile_n_down": 32, "tile_n_gate": 32} |
| qwen35_35B_k256 | 4 | 2048 / 256 / 256 / 8 | PASS/PASS | 0.000140072 | 0.00101737 | 47.38 | 25.32 | 1.871x | fly_decode | 3.976 | PASS | {"_impl": "fly_decode", "block_n": 0, "decode_alg": "batch1", "down_path": "default", "num_oc_splits": 1, "padding": null, "tile_m_down": 16, "tile_m_gate": 16, "tile_n_down": 64, "tile_n_gate": 64} |
| qwen35_35B_k256 | 16 | 2048 / 256 / 256 / 8 | PASS/PASS | 0.000137601 | 0.00107915 | 104.30 | 72.12 | 1.446x | fly_decode | 5.583 | PASS | {"_impl": "fly_decode", "block_n": 0, "decode_alg": "batch1", "down_path": "default", "num_oc_splits": 1, "padding": null, "tile_m_down": 16, "tile_m_gate": 16, "tile_n_down": 64, "tile_n_gate": 64} |
| qwen35_35B_k256 | 64 | 2048 / 256 / 256 / 8 | PASS/PASS | 0.000125391 | 0.00104546 | 184.72 | 246.48 | 0.749x | jit_splitk | 6.534 | PASS | {"_impl": "jit_splitk", "block_n": 0, "decode_alg": "splitk", "down_path": "default", "num_oc_splits": 1, "padding": null, "tile_m_down": 16, "tile_m_gate": 16, "tile_n_down": 64, "tile_n_gate": 64} |
| qwen35_35B_k256 | 256 | 2048 / 256 / 256 / 8 | PASS/PASS | 0.000136972 | 0.00103191 | 189.16 | 276.66 | 0.684x | jit_splitk | 23.286 | PASS | {"_impl": "jit_splitk", "block_n": 0, "decode_alg": "splitk", "down_path": "default", "num_oc_splits": 1, "padding": null, "tile_m_down": 16, "tile_m_gate": 16, "tile_n_down": 128, "tile_n_gate": 128} |
| qwen35_35B_k256 | 1024 | 2048 / 256 / 256 / 8 | PASS/PASS | 0.00014196 | 0.000141936 | 206.12 | 395.46 | 0.521x | aiter | 65.164 | PASS | {"_impl": "aiter"} |
| qwen35_35B_k256 | 4096 | 2048 / 256 / 256 / 8 | PASS/PASS | 0.000137825 | 0.000127961 | 558.94 | 731.68 | 0.764x | fly_prefill | 140.880 | PASS | {"_impl": "fly_prefill", "block_n": 0, "decode_alg": "splitk", "down_path": "default", "num_oc_splits": 1, "padding": null, "tile_k_gate": 128, "tile_m_down": 64, "tile_m_gate": 64, "tile_n_down": 128, "tile_n_gate": 256} |
| qwen35_35B_k256 | 8192 | 2048 / 256 / 256 / 8 | PASS/PASS | 0.000138304 | 0.000129779 | 1024.86 | 1058.90 | 0.968x | fly_prefill | 194.690 | PASS | {"_impl": "fly_prefill", "block_n": 0, "decode_alg": "splitk", "down_path": "8x1_compact", "num_oc_splits": 1, "padding": 128, "tile_k_gate": 128, "tile_m_down": 64, "tile_m_gate": 64, "tile_n_down": 128, "tile_n_gate": 256} |
| qwen35_35B_k256 | 16384 | 2048 / 256 / 256 / 8 | PASS/PASS | 0.00013804 | 0.000128361 | 2063.29 | 1746.77 | 1.181x | fly_prefill | 236.046 | PASS | {"_impl": "fly_prefill", "block_n": 0, "decode_alg": "splitk", "down_path": "8x1", "num_oc_splits": 1, "padding": 128, "tile_k_gate": 128, "tile_m_down": 256, "tile_m_gate": 64, "tile_n_down": 128, "tile_n_gate": 256} |
| qwen35_35B_k256 | 32768 | 2048 / 256 / 256 / 8 | PASS/PASS | 0.000138462 | 0.000128042 | 3754.03 | 3161.27 | 1.188x | fly_prefill | 260.855 | PASS | {"_impl": "fly_prefill", "block_n": 0, "decode_alg": "splitk", "down_path": "8x1", "num_oc_splits": 1, "padding": 128, "tile_k_gate": 128, "tile_m_down": 256, "tile_m_gate": 64, "tile_n_down": 128, "tile_n_gate": 256} |


### mimo_ptpc (fp8/ptpc, silu, TP=8)

model_dim=6144, inter_dim=2048, inter_dim_tp=256, experts=384, topk=8, gate_mode=separated, preshuffle=on, routing=balanced, seed=0

| model | M | H / I_tp / E / topk | check A/T | Aiter diff | winner diff | Aiter us | tuned us | speedup | winner | winner TFLOPS | status | winner config |
|---|---:|---|---|---:|---:|---:|---:|---:|---|---:|---|---|
| mimo_ptpc | 1 | 6144 / 256 / 384 / 8 | PASS/PASS | 0.000145645 | 0.000874092 | 52.96 | 25.04 | 2.115x | jit_batch1 | 3.015 | PASS | {"_impl": "jit_batch1", "block_n": 0, "decode_alg": "splitk", "down_path": "default", "num_oc_splits": 1, "padding": null, "tile_m_down": 16, "tile_m_gate": 16, "tile_n_down": 32, "tile_n_gate": 32} |
| mimo_ptpc | 4 | 6144 / 256 / 384 / 8 | PASS/PASS | 0.000143591 | 0.000944051 | 83.96 | 53.40 | 1.572x | fly_decode | 5.655 | PASS | {"_impl": "fly_decode", "block_n": 0, "decode_alg": "batch1", "down_path": "default", "num_oc_splits": 1, "padding": null, "tile_m_down": 16, "tile_m_gate": 16, "tile_n_down": 64, "tile_n_gate": 64} |
| mimo_ptpc | 16 | 6144 / 256 / 384 / 8 | PASS/PASS | 0.000169016 | 0.00101894 | 244.48 | 190.36 | 1.284x | fly_decode | 6.346 | PASS | {"_impl": "fly_decode", "block_n": 0, "decode_alg": "batch1", "down_path": "default", "num_oc_splits": 1, "padding": null, "tile_m_down": 16, "tile_m_gate": 16, "tile_n_down": 64, "tile_n_gate": 64} |
| mimo_ptpc | 64 | 6144 / 256 / 384 / 8 | PASS/PASS | 0.000156549 | 0.00102072 | 571.30 | 644.00 | 0.887x | jit_splitk | 7.503 | PASS | {"_impl": "jit_splitk", "block_n": 0, "decode_alg": "splitk", "down_path": "default", "num_oc_splits": 1, "padding": null, "tile_m_down": 16, "tile_m_gate": 16, "tile_n_down": 64, "tile_n_gate": 64} |
| mimo_ptpc | 256 | 6144 / 256 / 384 / 8 | PASS/PASS | 0.000142033 | 0.00103079 | 594.60 | 698.90 | 0.851x | jit_splitk | 27.654 | PASS | {"_impl": "jit_splitk", "block_n": 0, "decode_alg": "splitk", "down_path": "default", "num_oc_splits": 1, "padding": null, "tile_m_down": 16, "tile_m_gate": 16, "tile_n_down": 128, "tile_n_gate": 128} |
| mimo_ptpc | 1024 | 6144 / 256 / 384 / 8 | PASS/PASS | 0.000134795 | 0.000134785 | 648.68 | 834.42 | 0.777x | aiter | 92.650 | PASS | {"_impl": "aiter"} |
| mimo_ptpc | 4096 | 6144 / 256 / 384 / 8 | PASS/PASS | 0.000138856 | 0.000130912 | 1720.95 | 1774.25 | 0.970x | fly_prefill | 174.292 | PASS | {"_impl": "fly_prefill", "block_n": 0, "decode_alg": "splitk", "down_path": "default", "num_oc_splits": 1, "padding": null, "tile_k_gate": 128, "tile_m_down": 128, "tile_m_gate": 128, "tile_n_down": 128, "tile_n_gate": 128} |
| mimo_ptpc | 8192 | 6144 / 256 / 384 / 8 | PASS/PASS | 0.000137526 | 0.000128448 | 3073.77 | 2637.05 | 1.166x | fly_prefill | 234.533 | PASS | {"_impl": "fly_prefill", "block_n": 0, "decode_alg": "splitk", "down_path": "1x4_64x256", "num_oc_splits": 1, "padding": 0, "tile_k_gate": 128, "tile_m_down": 64, "tile_m_gate": 64, "tile_n_down": 256, "tile_n_gate": 256} |
| mimo_ptpc | 16384 | 6144 / 256 / 384 / 8 | PASS/PASS | 0.000139123 | 0.000129942 | 5556.76 | 4738.74 | 1.173x | fly_prefill | 261.030 | PASS | {"_impl": "fly_prefill", "block_n": 0, "decode_alg": "splitk", "down_path": "8x1_compact", "num_oc_splits": 1, "padding": 128, "tile_k_gate": 128, "tile_m_down": 64, "tile_m_gate": 64, "tile_n_down": 128, "tile_n_gate": 256} |
| mimo_ptpc | 32768 | 6144 / 256 / 384 / 8 | PASS/PASS | 0.00013814 | 0.000128895 | 10988.26 | 8877.67 | 1.238x | fly_prefill | 278.666 | PASS | {"_impl": "fly_prefill", "block_n": 0, "decode_alg": "splitk", "down_path": "8x1_compact", "num_oc_splits": 1, "padding": 128, "tile_k_gate": 128, "tile_m_down": 64, "tile_m_gate": 64, "tile_n_down": 128, "tile_n_gate": 256} |


### mimo_block (fp8/block, silu, TP=8)

model_dim=6144, inter_dim=2048, inter_dim_tp=256, experts=384, topk=8, gate_mode=separated, preshuffle=on, routing=balanced, seed=0

| model | M | H / I_tp / E / topk | check A/T | Aiter diff | winner diff | Aiter us | tuned us | speedup | winner | winner TFLOPS | status | winner config |
|---|---:|---|---|---:|---:|---:|---:|---:|---|---:|---|---|
| mimo_block | 1 | 6144 / 256 / 384 / 8 | PASS/PASS | 0.000164 | 0.00100299 | 45.72 | 158.22 | 0.289x | jit_splitk | 0.477 | PASS | {"_impl": "jit_splitk", "block_n": 0, "decode_alg": "splitk", "down_path": "default", "num_oc_splits": 1, "padding": null, "tile_m_down": 16, "tile_m_gate": 16, "tile_n_down": 128, "tile_n_gate": 128} |
| mimo_block | 4 | 6144 / 256 / 384 / 8 | PASS/PASS | 0.000132756 | 0.0011263 | 79.52 | 178.52 | 0.445x | jit_splitk | 1.692 | PASS | {"_impl": "jit_splitk", "block_n": 0, "decode_alg": "splitk", "down_path": "default", "num_oc_splits": 1, "padding": null, "tile_m_down": 16, "tile_m_gate": 16, "tile_n_down": 64, "tile_n_gate": 64} |
| mimo_block | 16 | 6144 / 256 / 384 / 8 | PASS/PASS | 6.04264e-05 | 0.0012865 | 202.10 | 352.36 | 0.574x | jit_splitk | 3.428 | PASS | {"_impl": "jit_splitk", "block_n": 0, "decode_alg": "splitk", "down_path": "default", "num_oc_splits": 1, "padding": null, "tile_m_down": 16, "tile_m_gate": 16, "tile_n_down": 64, "tile_n_gate": 64} |
| mimo_block | 64 | 6144 / 256 / 384 / 8 | PASS/PASS | 8.61789e-05 | 8.62704e-05 | 517.16 | 734.52 | 0.704x | aiter | 6.578 | PASS | {"_impl": "aiter"} |
| mimo_block | 256 | 6144 / 256 / 384 / 8 | PASS/PASS | 9.09258e-05 | 9.09705e-05 | 539.32 | 744.18 | 0.725x | aiter | 25.971 | PASS | {"_impl": "aiter"} |
| mimo_block | 1024 | 6144 / 256 / 384 / 8 | PASS/PASS | 9.03743e-05 | 9.03844e-05 | 626.70 | 825.18 | 0.759x | aiter | 93.688 | PASS | {"_impl": "aiter"} |
| mimo_block | 4096 | 6144 / 256 / 384 / 8 | PASS/PASS | 8.82945e-05 | 8.82918e-05 | 1650.75 | 1785.67 | 0.924x | aiter | 173.178 | PASS | {"_impl": "aiter"} |
| mimo_block | 8192 | 6144 / 256 / 384 / 8 | PASS/PASS | 8.94833e-05 | 8.94833e-05 | 3168.73 | 3170.45 | 0.999x | aiter | 195.075 | PASS | {"_impl": "aiter"} |
| mimo_block | 16384 | 6144 / 256 / 384 / 8 | PASS/PASS | 8.95935e-05 | 8.95937e-05 | 5742.72 | 5735.72 | 1.001x | aiter | 215.657 | PASS | {"_impl": "aiter"} |
| mimo_block | 32768 | 6144 / 256 / 384 / 8 | PASS/PASS | 8.99188e-05 | 8.99187e-05 | 11585.46 | 11564.14 | 1.002x | aiter | 213.929 | PASS | {"_impl": "aiter"} |


### h3 (fp8/ptpc, silu, TP=8)

model_dim=6144, inter_dim=3072, inter_dim_tp=384, experts=128, topk=4, gate_mode=separated, preshuffle=on, routing=balanced, seed=0

| model | M | H / I_tp / E / topk | check A/T | Aiter diff | winner diff | Aiter us | tuned us | speedup | winner | winner TFLOPS | status | winner config |
|---|---:|---|---|---:|---:|---:|---:|---:|---|---:|---|---|
| h3 | 1 | 6144 / 384 / 128 / 4 | PASS/PASS | 6.6843e-05 | 0.00096877 | 42.88 | 23.02 | 1.863x | jit_batch1 | 2.460 | PASS | {"_impl": "jit_batch1", "block_n": 0, "decode_alg": "splitk", "down_path": "default", "num_oc_splits": 1, "padding": null, "tile_m_down": 16, "tile_m_gate": 16, "tile_n_down": 32, "tile_n_gate": 32} |
| h3 | 4 | 6144 / 384 / 128 / 4 | PASS/PASS | 9.33826e-05 | 0.000972866 | 68.30 | 45.36 | 1.506x | fly_decode | 4.993 | PASS | {"_impl": "fly_decode", "block_n": 0, "decode_alg": "batch1", "down_path": "default", "num_oc_splits": 1, "padding": null, "tile_m_down": 16, "tile_m_gate": 16, "tile_n_down": 64, "tile_n_gate": 64} |
| h3 | 16 | 6144 / 384 / 128 / 4 | PASS/PASS | 0.000120006 | 0.00104836 | 167.50 | 143.14 | 1.170x | fly_decode | 6.329 | PASS | {"_impl": "fly_decode", "block_n": 0, "decode_alg": "batch1", "down_path": "default", "num_oc_splits": 1, "padding": null, "tile_m_down": 16, "tile_m_gate": 16, "tile_n_down": 64, "tile_n_gate": 64} |
| h3 | 64 | 6144 / 384 / 128 / 4 | PASS/PASS | 0.000141601 | 0.00103207 | 305.74 | 381.62 | 0.801x | jit_splitk | 9.496 | PASS | {"_impl": "jit_splitk", "block_n": 0, "decode_alg": "splitk", "down_path": "default", "num_oc_splits": 1, "padding": null, "tile_m_down": 16, "tile_m_gate": 16, "tile_n_down": 128, "tile_n_gate": 128} |
| h3 | 256 | 6144 / 384 / 128 / 4 | PASS/PASS | 0.000137121 | 0.00104073 | 323.98 | 435.30 | 0.744x | jit_splitk | 33.300 | PASS | {"_impl": "jit_splitk", "block_n": 0, "decode_alg": "splitk", "down_path": "default", "num_oc_splits": 1, "padding": null, "tile_m_down": 16, "tile_m_gate": 16, "tile_n_down": 128, "tile_n_gate": 128} |
| h3 | 1024 | 6144 / 384 / 128 / 4 | PASS/PASS | 0.000140384 | 0.000140385 | 368.94 | 556.46 | 0.663x | aiter | 104.198 | PASS | {"_impl": "aiter"} |
| h3 | 4096 | 6144 / 384 / 128 / 4 | PASS/PASS | 0.000140354 | 0.000133576 | 1116.72 | 1099.18 | 1.016x | fly_prefill | 211.000 | PASS | {"_impl": "fly_prefill", "block_n": 0, "decode_alg": "splitk", "down_path": "1x4_64x256", "num_oc_splits": 1, "padding": 128, "tile_k_gate": 128, "tile_m_down": 64, "tile_m_gate": 64, "tile_n_down": 256, "tile_n_gate": 256} |
| h3 | 8192 | 6144 / 384 / 128 / 4 | PASS/PASS | 0.000138189 | 0.000130756 | 1956.17 | 1738.15 | 1.125x | fly_prefill | 266.869 | PASS | {"_impl": "fly_prefill", "block_n": 0, "decode_alg": "splitk", "down_path": "default", "num_oc_splits": 1, "padding": null, "tile_k_gate": 256, "tile_m_down": 64, "tile_m_gate": 64, "tile_n_down": 128, "tile_n_gate": 128} |
| h3 | 16384 | 6144 / 384 / 128 / 4 | PASS/PASS | 0.000138205 | 0.000131394 | 3844.79 | 3064.77 | 1.255x | fly_prefill | 302.702 | PASS | {"_impl": "fly_prefill", "block_n": 0, "decode_alg": "splitk", "down_path": "default", "num_oc_splits": 1, "padding": null, "tile_k_gate": 128, "tile_m_down": 64, "tile_m_gate": 64, "tile_n_down": 128, "tile_n_gate": 256} |
| h3 | 32768 | 6144 / 384 / 128 / 4 | PASS/PASS | 0.000138083 | 0.00013097 | 7284.67 | 6081.98 | 1.198x | fly_prefill | 305.069 | PASS | {"_impl": "fly_prefill", "block_n": 0, "decode_alg": "splitk", "down_path": "default", "num_oc_splits": 1, "padding": null, "tile_k_gate": 128, "tile_m_down": 64, "tile_m_gate": 64, "tile_n_down": 128, "tile_n_gate": 256} |

