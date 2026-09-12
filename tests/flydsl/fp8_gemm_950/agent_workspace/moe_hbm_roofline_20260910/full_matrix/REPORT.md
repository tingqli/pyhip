# PyHIP A8W4 MoE Roofline Results

## 2026-09-12 Native A-scale follow-up

The [paired-row DS64 expanded regression report](../../native_scale_perf_20260912/expanded_paired_rows/README.md)
covers 25 configurations, two complete six-round paired batches, and 25/25
independent reference-accuracy passes. Pooled kernel latency changes have a
median of +0.55%, with 24/25 configurations <=1%; worst pooled +1.14%, worst
single-batch median +1.39%. These are **kernel-only paired comparisons**, not an
end-to-end speedup or an all-shape <=1% guarantee. The exact tested snapshot is
identified in the linked report; the later live experimental kernel is different.
The historical roofline/PMC/ATT results below have not been replaced or reprofiled.

All event timings are minimums from the unprofiled rotating-clone run. PMC columns are medians of isolated profiled dispatches.

- Shape: gate_up=512, K=6144, topk=8, experts=384
- Practical ceilings used: 5000 GB/s HBM, 3500 TFLOP/s A8W4 matrix
- Ridge point: 700.0 FLOP/byte
- Saturation threshold: 70%
- HBM % = profiled HBM GB/s / practical HBM ceiling
- Compute % = profiled padded TFLOP/s / practical A8W4 ceiling
- HBM bytes = 32 * (DRAM read 32B equivalents + normal-write 32B equivalents + atomic-write 32B equivalents)
- Padded FLOPs = 2 * padded_tokens * gate_up_size * hidden_size

## PyHIP Roofline Summary

| M | Padded rows | Pad | Event us | Padded TF/s | HBM GB/s | HBM % | Compute % | AI F/B | MFMA % | Wait % | Roof | Observed |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|:---|
| 8192 | 98304 | 1.500 | 227.482 | 2718.79 | 4016.05 | 80.3 | 67.9 | 592.0 | 66.1 | 12.4 | memory | memory-bandwidth |
| 12288 | 98304 | 1.000 | 249.762 | 2476.26 | 4779.99 | 95.6 | 68.2 | 499.0 | 61.0 | 13.2 | memory | memory-bandwidth |
| 16384 | 196608 | 1.500 | 423.924 | 2917.86 | 2681.33 | 53.6 | 63.0 | 823.0 | 68.5 | 12.1 | compute | issue/sync/occupancy |
| 24576 | 196608 | 1.000 | 468.605 | 2639.64 | 3272.17 | 65.4 | 60.3 | 645.4 | 65.8 | 12.1 | memory | memory-side-unsaturated |
| 32768 | 294912 | 1.125 | 670.247 | 2768.27 | 3454.42 | 69.1 | 74.2 | 750.7 | 68.2 | 12.1 | compute | compute |
| 49152 | 393216 | 1.000 | 925.329 | 2673.54 | 3195.08 | 63.9 | 68.4 | 749.4 | 68.8 | 12.2 | compute | issue/sync/occupancy |
| 65536 | 589824 | 1.125 | 1320.013 | 2811.22 | 2969.78 | 59.4 | 71.9 | 847.8 | 69.8 | 12.0 | compute | compute |

## AIter Roofline Summary

| Tokens | Padded rows | AI F/B | HBM GB/s | Event padded TF/s | Profile padded TF/s | Compute % | Bound |
|---:|---:|---:|---:|---:|---:|---:|:---|
| 8192 | 98304 | 576.5 | 3671.16 | 2445.70 | 2116.59 | 60.5 | memory-bandwidth |
| 12288 | 98304 | 483.9 | 4038.40 | 2251.93 | 1953.72 | 55.8 | memory-bandwidth |
| 16384 | 147456 | 581.9 | 3130.45 | 2358.89 | 1821.60 | 52.0 | memory-side-unsaturated |
| 24576 | 196608 | 629.2 | 3229.61 | 2359.85 | 2032.17 | 58.1 | memory-side-unsaturated |
| 32768 | 294912 | 717.3 | 2930.08 | 2432.75 | 2101.73 | 60.0 | issue/sync/occupancy |
| 49152 | 393216 | 740.9 | 2811.16 | 2386.07 | 2077.35 | 59.4 | issue/sync/occupancy |
| 65536 | 540672 | 731.2 | 2862.90 | 2418.37 | 2094.57 | 59.8 | issue/sync/occupancy |

## PyHIP vs AIter MoE Throughput

Both ratios are PyHIP / AIter. Effective throughput uses routed tokens; padded throughput includes rows executed by the MFMA pipe.
Effective compute % = routed rows / padded rows; it measures padding efficiency, not hardware compute utilization.

### Throughput

| Tokens | AIter effective TF/s | PyHIP effective TF/s | AIter padded TF/s | PyHIP padded TF/s |
|---:|---:|---:|---:|---:|
| 8192 | 1630.47 | 1812.53 | 2445.70 | 2718.79 |
| 12288 | 2251.93 | 2476.26 | 2251.93 | 2476.26 |
| 16384 | 2096.79 | 1945.24 | 2358.89 | 2917.86 |
| 24576 | 2359.85 | 2639.64 | 2359.85 | 2639.64 |
| 32768 | 2162.44 | 2460.69 | 2432.75 | 2768.27 |
| 49152 | 2386.07 | 2673.54 | 2386.07 | 2673.54 |
| 65536 | 2345.08 | 2498.87 | 2418.37 | 2811.22 |

### Ratios And Padding Efficiency

| Tokens | AIter padded rows | PyHIP padded rows | AIter effective compute % | PyHIP effective compute % | Effective tput ratio | Padded tput ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 8192 | 98304 | 98304 | 66.7 | 66.7 | 1.112 | 1.112 |
| 12288 | 98304 | 98304 | 100.0 | 100.0 | 1.100 | 1.100 |
| 16384 | 147456 | 196608 | 88.9 | 66.7 | 0.928 | 1.237 |
| 24576 | 196608 | 196608 | 100.0 | 100.0 | 1.119 | 1.119 |
| 32768 | 294912 | 294912 | 88.9 | 88.9 | 1.138 | 1.138 |
| 49152 | 393216 | 393216 | 100.0 | 100.0 | 1.120 | 1.120 |
| 65536 | 540672 | 589824 | 97.0 | 88.9 | 1.066 | 1.162 |

### Token-Weighted Summary

Weighted mean = sum(tokens * metric) / sum(tokens). Ratios are PyHIP / AIter; for latency, values below 1 are better.

| Metric | AIter weighted mean | PyHIP weighted mean | PyHIP/AIter |
|:---|---:|---:|---:|
| Event latency (us) | 923.456 | 848.975 | 0.919 |
| Effective TF/s | 2274.84 | 2478.87 | 1.090 |
| Padded TF/s | 2392.76 | 2736.94 | 1.144 |
| HBM GB/s | 3026.23 | 3259.28 | 1.077 |

## PyHIP MoE vs PyHIP Dense A8W4 GEMM

Each dense GEMM uses M equal to the matched MoE padded rows, with the same N, K, datatypes, scales, and GEMM FLOPs. Dense uses one shared weight matrix while MoE accesses expert-indexed weights. The dense GEMM writes N BF16 outputs per row; fused MoE gate/up writes N/2 BF16 outputs after activation, so HBM bytes are measured rather than assumed equal.

Dense coverage: 7/7 unique padded-row shapes (complete).

`TF/s ratio` and `PMC ratio` are PyHIP MoE / PyHIP dense; values above 1 mean MoE has higher throughput.

| Tokens | Padded M | MoE TF/s | Dense TF/s | TF/s ratio | MoE PMC TF/s | Dense PMC TF/s | PMC ratio | HBM bytes ratio | MoE bound | Dense bound |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|:---|
| 8192 | 98304 | 2718.79 | 2702.62 | 1.006 | 2377.64 | 2553.12 | 0.931 | 1.368 | memory-bandwidth | compute |
| 12288 | 98304 | 2476.26 | 2702.62 | 0.916 | 2385.33 | 2553.12 | 0.934 | 1.623 | memory-bandwidth | compute |
| 16384 | 196608 | 2917.86 | 2787.65 | 1.047 | 2206.61 | 2725.73 | 0.810 | 0.984 | issue/sync/occupancy | compute |
| 24576 | 196608 | 2639.64 | 2787.65 | 0.947 | 2111.97 | 2725.73 | 0.775 | 1.255 | memory-side-unsaturated | compute |
| 32768 | 294912 | 2768.27 | 2788.91 | 0.993 | 2595.99 | 2689.15 | 0.965 | 1.079 | compute | compute |
| 49152 | 393216 | 2673.54 | 2804.59 | 0.953 | 2393.73 | 2680.37 | 0.893 | 1.080 | issue/sync/occupancy | compute |
| 65536 | 589824 | 2811.22 | 2751.36 | 1.022 | 2517.79 | 2663.06 | 0.945 | 0.955 | compute | compute |

## ATT Mainloop MFMA Efficiency

These results are calculated from decoded ATT wave clocks and are distinct from
the dispatch-level PMC `MFMA %` column above. The measured interval starts at the
first mainloop entry and ends at the first epilogue entry. Each of the 23
mainloop iterations contains 128
`v_mfma_scale_f32_16x16x128_f8f6f4` instructions at 32 cycles each, giving 4096
theoretical MFMA cycles per iteration.

`MFMA efficiency = 4096 / mean measured cycles per iteration`. The mean gives
equal weight to every valid decoded wave across the two captured dispatches.

| MoE tokens | Matched GEMM M | MoE valid waves | MoE cycles/iter | MoE MFMA efficiency | GEMM valid waves | GEMM cycles/iter | GEMM MFMA efficiency | MoE - GEMM |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 12288 | 98304 | 48/48 | 4787.456522 | 85.556913% | 48/48 | 4503.793478 | 90.945556% | -5.388643 pp |
| 16384 | 196608 | 96/96 | 4292.257246 | 95.427645% | 96/96 | 4517.317029 | 90.673291% | +4.754354 pp |
| 24576 | 196608 | 96/96 | 4447.780797 | 92.090869% | 96/96 | 4475.617754 | 91.518093% | +0.572777 pp |
| 32768 | 294912 | 144/144 | 4312.907005 | 94.970747% | 144/144 | 4558.710145 | 89.849977% | +5.120770 pp |
| 49152 | 393216 | 192/192 | 4375.557065 | 93.610938% | 192/192 | 4421.855072 | 92.630806% | +0.980132 pp |
| 65536 | 589824 | 288/288 | 4337.845411 | 94.424757% | 288/288 | 4448.084541 | 92.084581% | +2.340177 pp |

All 1728 decoded waves passed loop validation: every wave contained all 23
iterations, with one back-edge and all 128 static MFMA instruction indices
present exactly once per iteration. Across the six shapes, the unweighted mean
mainloop MFMA efficiency is 92.680312% for MoE and 91.283717% for the matched
dense-shaped gate/up runs, a mean difference of +1.396594 percentage points.

### Refreshed 24576/196608 Phase Breakdown

The current `--warmup 3` captures contain two dispatches and 96 valid waves per
side. Phase boundaries use `wave.begin -> first mainloop entry -> first epilogue
entry -> wave.end`, so the three phase durations add exactly to the decoded wave
lifetime. This whole-wave result is an ATT-derived kernel-body metric, not the
dispatch-level PMC `MfmaUtil` counter.

| Phase | MFMA/wave | Theoretical MFMA cycles | MoE mean cycles | MoE time share | MoE MFMA efficiency | GEMM mean cycles | GEMM time share | GEMM MFMA efficiency | MoE - GEMM |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Prologue | 0 | 0 | 8912.625 | 6.490% | N/A | 8310.792 | 6.045% | N/A | N/A |
| Mainloop | 2944 | 94208 | 102298.958 | 74.495% | 92.090869% | 102939.208 | 74.880% | 91.518093% | +0.572777 pp |
| Epilogue | 128 | 4096 | 26111.542 | 19.015% | 15.686550% | 26222.542 | 19.075% | 15.620149% | +0.066401 pp |
| Whole wave | 3072 | 98304 | 137323.125 | 100.000% | 71.585904% | 137472.542 | 100.000% | 71.508098% | +0.077806 pp |

Relative to GEMM, MoE spends 601.833 more cycles in the prologue (+7.242%),
640.250 fewer cycles in the mainloop (-0.622%), 111.000 fewer cycles in the
epilogue (-0.423%), and 149.417 fewer cycles over the whole wave (-0.109%). The
two per-dispatch mainloop efficiencies are 91.408907% and 92.783084% for MoE,
versus 92.697646% and 90.368181% for GEMM; this repeat-to-repeat spread is larger
than the aggregate 0.572777-point difference.

The full-device `att_shader_engine_mask=0xffffffff` results, with one SE0-SE31
table for each target CU1-CU7, are in
[MoE Full-Device ATT Per-SE Results](../../moe_full_se_att_20260911/REPORT.md).

### Legacy Four-SE Per-Target-CU Phase Breakdown

The following earlier tables use `att_shader_engine_mask=0xf` and
`att_target_cu=1..7`, respectively. Each target
was captured independently at matching-kernel iterations 5, 8, and 14. Every
iteration produced one decoded dispatch with 96 valid waves, so each table gives
equal weight to 288 waves across three dispatches. With
`att_shader_engine_mask=0xf`, one target index selects the CU with that local
index in each of SE0-SE3; a table therefore aggregates four physical CUs, not
one physical CU. All four SIMDs are included for every selected CU.

#### ATT Target CU 1

| Phase | MFMA/wave | Theoretical MFMA cycles | Mean cycles | Time share | MFMA efficiency |
|:---|---:|---:|---:|---:|---:|
| Prologue | 0 | 0 | 8515.319 | 6.220% | N/A |
| Mainloop | 2944 | 94208 | 102380.681 | 74.786% | 92.017361% |
| Epilogue | 128 | 4096 | 26003.014 | 18.994% | 15.752020% |
| Whole wave | 3072 | 98304 | 136899.014 | 100.000% | 71.807676% |

#### ATT Target CU 2

| Phase | MFMA/wave | Theoretical MFMA cycles | Mean cycles | Time share | MFMA efficiency |
|:---|---:|---:|---:|---:|---:|
| Prologue | 0 | 0 | 8703.847 | 6.317% | N/A |
| Mainloop | 2944 | 94208 | 103070.778 | 74.810% | 91.401270% |
| Epilogue | 128 | 4096 | 26002.000 | 18.873% | 15.752634% |
| Whole wave | 3072 | 98304 | 137776.625 | 100.000% | 71.350274% |

#### ATT Target CU 3

| Phase | MFMA/wave | Theoretical MFMA cycles | Mean cycles | Time share | MFMA efficiency |
|:---|---:|---:|---:|---:|---:|
| Prologue | 0 | 0 | 8676.458 | 6.327% | N/A |
| Mainloop | 2944 | 94208 | 102464.792 | 74.713% | 91.941826% |
| Epilogue | 128 | 4096 | 26002.375 | 18.960% | 15.752407% |
| Whole wave | 3072 | 98304 | 137143.625 | 100.000% | 71.679599% |

#### ATT Target CU 4

| Phase | MFMA/wave | Theoretical MFMA cycles | Mean cycles | Time share | MFMA efficiency |
|:---|---:|---:|---:|---:|---:|
| Prologue | 0 | 0 | 8950.694 | 6.431% | N/A |
| Mainloop | 2944 | 94208 | 104147.917 | 74.832% | 90.455962% |
| Epilogue | 128 | 4096 | 26077.819 | 18.737% | 15.706835% |
| Whole wave | 3072 | 98304 | 139176.431 | 100.000% | 70.632649% |

#### ATT Target CU 5

| Phase | MFMA/wave | Theoretical MFMA cycles | Mean cycles | Time share | MFMA efficiency |
|:---|---:|---:|---:|---:|---:|
| Prologue | 0 | 0 | 8883.889 | 6.467% | N/A |
| Mainloop | 2944 | 94208 | 102451.875 | 74.584% | 91.953417% |
| Epilogue | 128 | 4096 | 26027.903 | 18.948% | 15.736958% |
| Whole wave | 3072 | 98304 | 137363.667 | 100.000% | 71.564776% |

#### ATT Target CU 6

| Phase | MFMA/wave | Theoretical MFMA cycles | Mean cycles | Time share | MFMA efficiency |
|:---|---:|---:|---:|---:|---:|
| Prologue | 0 | 0 | 8603.722 | 6.229% | N/A |
| Mainloop | 2944 | 94208 | 103509.861 | 74.937% | 91.013551% |
| Epilogue | 128 | 4096 | 26015.333 | 18.834% | 15.744561% |
| Whole wave | 3072 | 98304 | 138128.917 | 100.000% | 71.168299% |

#### ATT Target CU 7

| Phase | MFMA/wave | Theoretical MFMA cycles | Mean cycles | Time share | MFMA efficiency |
|:---|---:|---:|---:|---:|---:|
| Prologue | 0 | 0 | 8594.875 | 6.213% | N/A |
| Mainloop | 2944 | 94208 | 103723.653 | 74.982% | 90.825957% |
| Epilogue | 128 | 4096 | 26013.486 | 18.805% | 15.745679% |
| Whole wave | 3072 | 98304 | 138332.014 | 100.000% | 71.063810% |

Across target indices 1-7, the unweighted mean mainloop MFMA efficiency is
91.372763%, ranging from 90.455962% on target CU 4 to 92.017361% on target CU 1
(1.561399 percentage-point spread). Whole-wave MFMA efficiency averages
71.323869%, with a 70.632649%-71.807676% range. All 2016 decoded waves passed
the 23-iteration mainloop, phase-boundary, dynamic-MFMA-count, and duration-sum
checks; no ATT data-loss warning was reported.

## Quantitative Conclusions

- PyHIP wins effective throughput on 6/7 shapes and padded throughput on 7/7 shapes.
- Tokens-weighted PyHIP/AIter ratios are 0.919x latency, 1.090x effective TF/s, 1.144x padded TF/s, and 1.077x measured HBM GB/s.
- The weakest effective result is tokens=16384: 0.928x effective versus 1.237x padded throughput.
- The largest PyHIP padding-efficiency deficit is tokens=16384: 66.7% versus AIter 88.9%.
- Observed PyHIP bounds: compute=2, issue/sync/occupancy=2, memory-bandwidth=2, memory-side-unsaturated=1. Observed AIter bounds: issue/sync/occupancy=3, memory-bandwidth=2, memory-side-unsaturated=2.
- Against same-padded-FLOP dense GEMM, PyHIP MoE wins unprofiled event TF/s on 3/7 shapes; event ratios span 0.916x to 1.047x.
- Profiled MoE/dense TF/s ratios span 0.775x to 0.965x; these isolated PMC durations are used for roofline classification, while event timings are the primary performance comparison.
- In the six-shape ATT sweep, MoE mainloop MFMA efficiency is 85.56% to 95.43%, while matched dense-shaped gate/up efficiency is 89.85% to 92.63%. MoE is higher in 5/6 comparisons by 0.57 to 5.12 percentage points and lower for tokens=12288 by 5.39 percentage points.

## Interpretation

`Roof` is the arithmetic-intensity prediction. `Observed` uses HBM and padded-TFLOPS utilization relative to the practical ceilings. The PMC `MFMA %` value is a dispatch-level hardware pipeline diagnostic; ATT mainloop MFMA efficiency measures theoretical MFMA cycles divided by elapsed cycles only inside the decoded mainloop. A low-bandwidth, high-wait result is labeled memory-latency/cache rather than HBM-bandwidth bound. If neither subsystem is saturated, the report does not force a false memory-versus-compute binary classification.

Nominal tensor bytes are intentionally excluded from measured HBM traffic.
