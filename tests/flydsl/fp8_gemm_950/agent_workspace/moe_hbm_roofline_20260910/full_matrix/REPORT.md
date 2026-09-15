# PyHIP A8W4 MoE Roofline Results

## 2026-09-14 B-scale Commit Double-check: ATT vs Actual Throughput

**T24576: `866d403` lowers sampled ATT mainloop efficiency, but no unprofiled MoE throughput regression was reproduced.** Direct parent is `16f3101`; both revisions use SiTUv2. Current's SwiGLU is listed separately, not used to explain the scale-commit before/after result.

GPU6, gate_up512/K6144/topk8/E384, seed0, XCD8/groupM4. Unprofiled original-driver test:7 alternating rounds, warmup3/clones20/Event samples20, median of seven per-run minima. All21 runs completed; common inputs match, and the14 parent/commit outputs match byte-for-byte. Scale preparation/conversion is outside kernel timing.

| Revision | Activation | Unprofiled latency µs | Padded TF/s | Initial paired ATT mainloop efficiency |
|:---|:---|---:|---:|---:|
| 16f3101, before B-scale commit | SiTUv2 | 473.684 | 2611.34 | 90.749526% |
| 866d403, B-scale commit | SiTUv2 | 471.845 | 2621.52 | 86.972244% |
| Current 477a7ce kernel snapshot | SwiGLU | 439.324 | 2815.58 | 86.869116% |

The B-scale commit changes median latency by **−0.388%** and throughput by **+0.390%**, a small effect within observed variation. Paired latency changes span−1.179% to+0.261%; this does not support an actual tput regression at this shape.

ATT is a separate measurement: CU1 on SE0–3/allSIMDs, matching iteration8,4counterordered rounds/version. Initial parent→B-commit mainloop cycles increase **+4.343%** (all four pairs positive), efficiency−3.777pp. Follow-up control comparisons reproduce positive aggregate increases of+1.967% to+3.621%, with individual reversals. Extra initial cycles localize mostly to G2S issue+stall (**+3782.531 of+4508.604 cycles/wave**), especially A-data G2S; explicit waitcnt time and MFMA work barely change.

**Cause boundary:** MFMA nesting, B-data row mapping and B-scale global-contiguity ablations did not recover the gap. Widening the parent's B-scale transport plus its LDS slots produces a smaller+1.031% mainloop increase, implicating the transport path but not proving a unique cache/queue/LDS/occupancy mechanism. Actual MFMA work and G2S request counts are unchanged; B-scale transfers widen from DWORD to DWORDx4. No PMC counters were collected.

Sampled mean mainloop cycles are not whole-device unprofiled dispatch time: fixed eighth launch versus best-of20 Event launches, sampled CUs versus dispatch makespan, and cycles versus time without locked/logged clocks. ATT perturbation and cache/scheduling effects are not isolated. **SwiGLU's shorter epilogue is relevant only to current, not the two SiTUv2 scale revisions.**

No production fix merged. A native-layout producer-balancing candidate preserves DWORDx4 and the pipeline, but its first GPU test was blocked by119.6GB VRAM already in use on GPU6; only syntax/integer mapping checks passed. It must pass GPU correctness, ATT and unprofiled throughput before adoption. Historical tables below are unchanged.

[Detailed conclusions, failed attempts and candidate status](../../scale_commit_att_20260914/CONCLUSIONS.md) · [Unprofiled results](../../scale_commit_att_20260914/unprofiled_perf/REPORT.md) · [Initial commit ATT comparison](../../scale_commit_att_20260914/REPORT.md)

## 2026-09-14 Current SwiGLU ATT and Performance

**Activation: SiLU(gate) × up, without clamping.** All new tables in this section use the same frozen source, including the user-added epilogue wait/barrier before consuming the final up operands.
Kernel SHA256: `477a7ce297519d6a3ebaa4b25eb45d9f0f205732280c6b1b78caf30a62bf9f2e`.
Driver SHA256: `fa4e2de725d94e718f8e2d8ca40d8b34c3e831f71acadd7c9726312255898c97`.

- Common shape: intermediate=256 (gate_up=512), K=6144, topk=8, experts=384; GPU6.
- AIter: tile M128/N256/K256, act=silu, swiglu_limit=None, XCD8. PyHIP: four waves, sort block M256, XCD enabled/groupM4. Native A/B scale layouts retained.
- Kernel source differs from the earlier 695be7... snapshot below. T24576 was recaptured alongside T12288 and T49152; previous ATT and performance results are not relabeled as current.

### ATT Phase Breakdown: T12288 / T24576 / T49152

Scope: target CU1 on SE0–SE3 (four physical CUs), all four SIMDs, two dispatches per shape, warmup3/clones20. Six dispatches and24 raw SE traces contain **672 complete waves**:96/192/384 for T12288/T24576/T49152. No trace-loss warning was reported; all three capture output checks have max_abs=0 against AIter.
Phase boundaries: wave.begin → first mainloop entry → first epilogue entry → wave.end. Every wave passed23-iteration validation, all128 static mainloop MFMA indices exactly once per iteration, dynamic MFMA counts0/2944/128, and num_insts==num_stitched==instruction-list length. Phase cycles sum exactly to each wave lifetime.
Time share = summed phase cycles / summed whole-wave cycles. Mean cycles use equal wave weights within each shape; shapes are not pooled. The epilogue includes two-K-tile drain, SwiGLU, packing and stores.

| Tokens | Valid waves | Phase | MFMA/wave | Theoretical MFMA cycles | Mean cycles/wave | Time share | Theoretical MFMA efficiency |
|---:|---:|:---|---:|---:|---:|---:|---:|
| 12288 | 96 | Prologue | 0 | 0 | 10122.917 | 7.085% | N/A |
| 12288 | 96 | Mainloop | 2944 | 94208 | 121625.750 | 85.124% | 77.457282% |
| 12288 | 96 | Epilogue | 128 | 4096 | 11132.583 | 7.791% | 36.792898% |
| 12288 | 96 | Whole wave | 3072 | 98304 | 142881.250 | 100.000% | 68.801190% |
| 24576 | 192 | Prologue | 0 | 0 | 9272.896 | 7.358% | N/A |
| 24576 | 192 | Mainloop | 2944 | 94208 | 105623.542 | 83.807% | 89.192237% |
| 24576 | 192 | Epilogue | 128 | 4096 | 11135.375 | 8.835% | 36.783674% |
| 24576 | 192 | Whole wave | 3072 | 98304 | 126031.812 | 100.000% | 77.999354% |
| 49152 | 384 | Prologue | 0 | 0 | 8586.396 | 6.568% | N/A |
| 49152 | 384 | Mainloop | 2944 | 94208 | 110975.031 | 84.889% | 84.891168% |
| 49152 | 384 | Epilogue | 128 | 4096 | 11168.312 | 8.543% | 36.675192% |
| 49152 | 384 | Whole wave | 3072 | 98304 | 130729.740 | 100.000% | 75.196356% |

Theoretical MFMA efficiency = theoretical MFMA cycles / phase elapsed cycles; it is **not** the dispatch-level PMC MfmaUtil counter or additive instruction issue/stall share. Full-device occupancy and HBM bandwidth cannot be inferred from these sampled waves.

#### ATT Per-dispatch Variability

| Tokens | Dispatch | Waves | Prologue share | Mainloop share | Epilogue share | Whole-wave cycles |
|---:|:---|---:|---:|---:|---:|---:|
| 12288 | 434 | 48 | 7.207% | 84.963% | 7.831% | 142103.000 |
| 12288 | 440 | 48 | 6.964% | 85.283% | 7.753% | 143659.500 |
| 24576 | 437 | 96 | 7.393% | 83.899% | 8.708% | 127869.458 |
| 24576 | 443 | 96 | 7.321% | 83.713% | 8.966% | 124194.167 |
| 49152 | 437 | 192 | 6.385% | 85.240% | 8.375% | 133548.792 |
| 49152 | 443 | 192 | 6.759% | 84.522% | 8.719% | 127910.688 |

### SwiGLU Performance Method

Unprofiled Event benchmark after ATT completed: three full seven-shape runs of the frozen test_moe driver, seed0,5warmups,20samples,20rotating clones; original token order8192,16384,32768,65536,12288,24576,49152. Each shape runs AIter then PyHIP, not alternating paired order. GPU6 idle was checked before the sequence; exclusive access is not guaranteed.
Each reported latency is the **median of three per-run minima**, not the global minimum or the median of all60 samples. Throughputs are recomputed from that latency. All21/21 checks passed with max_abs=0 and finite outputs; source hashes matched before/after.
Routed rows=tokens×topk. Effective FLOPs=2×routed_rows×gate_up×K; padded FLOPs replace routed_rows with the implementation’s executed padded rows. Throughput counts GEMM FLOPs, not activation operations. Both throughput ratios below are **PyHIP/AIter**.

###  Throughput data PYHIIP and AITER

| Tokens | AIter effective TF/s | PyHIP effective TF/s | AIter padded TF/s | PyHIP padded TF/s |
|---:|---:|---:|---:|---:|
| 8192 | 1614.38 | 1963.77 | 2421.57 | 2945.65 |
| 12288 | 2256.20 | 2671.81 | 2256.20 | 2671.81 |
| 16384 | 2096.15 | 2119.43 | 2358.17 | 3179.14 |
| 24576 | 2353.74 | 2822.77 | 2353.74 | 2822.77 |
| 32768 | 2174.19 | 2652.38 | 2445.96 | 2983.93 |
| 49152 | 2394.20 | 2911.95 | 2394.20 | 2911.95 |
| 65536 | 2305.87 | 2676.75 | 2377.93 | 3011.34 |

### Throughput Ratios(pyhip/aiter) And Padding Efficiency

Effective compute % below means routed_rows/padded_rows, i.e. **padding efficiency**, not hardware utilization.

| Tokens | AIter padded rows | PyHIP padded rows | AIter effective compute % | PyHIP effective compute % | Effective tput ratio | Padded tput ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 8192 | 98304 | 98304 | 66.7 | 66.7 | 1.216 | 1.216 |
| 12288 | 98304 | 98304 | 100.0 | 100.0 | 1.184 | 1.184 |
| 16384 | 147456 | 196608 | 88.9 | 66.7 | 1.011 | 1.348 |
| 24576 | 196608 | 196608 | 100.0 | 100.0 | 1.199 | 1.199 |
| 32768 | 294912 | 294912 | 88.9 | 88.9 | 1.220 | 1.220 |
| 49152 | 393216 | 393216 | 100.0 | 100.0 | 1.216 | 1.216 |
| 65536 | 540672 | 589824 | 97.0 | 88.9 | 1.161 | 1.266 |


### SwiGLU PyHIP MoE vs Equal-padded Single-expert GEMM (2026-09-14)

Both workloads use the same unmodified, source-frozen **test_moe.py** driver and the PyHIP result from that driver. MoE: topk=8, experts=384; GEMM: topk=1, experts=1, tokens set to the MoE **actual PyHIP padded rows**. This is the single-expert fused gate/up SwiGLU path, not a different standalone dense-GEMM implementation.
Common settings: GPU6, gate_up=512 (intermediate=256), K=6144, XCD enabled/groupM4, seed0,5 warmups,20 rotating clones,20 Event samples. **No profiler or PMC counters.** Input preparation, quantization and routing are outside the timed kernel.
Three rounds per pair, alternating MoE/GEMM order by pair and round. Each invocation retains the driver’s AIter check and AIter-then-PyHIP measurement order; only **PyHIP padded TF/s** is compared below. Each reported latency is the median of three per-run minima. Repeated matched GEMM sizes are rerun for each pair, not reused from another row.
Kernel SHA256: `477a7ce297519d6a3ebaa4b25eb45d9f0f205732280c6b1b78caf30a62bf9f2e`. Driver SHA256: `fa4e2de725d94e718f8e2d8ca40d8b34c3e831f71acadd7c9726312255898c97`.
Validation: **42/42 invocations passed**, with matched actual padded rows and unchanged live/source snapshots. Every workload passed the driver’s AIter/PyHIP finite-output and closeness checks; all max_abs values zero: **True**. MoE and single-expert GEMM outputs are not compared to each other, since their inputs/weights differ.

Padded FLOPs = 2 × padded_M × 512 × 6144. Padded TF/s = padded_FLOPs / (latency_us × 10^6). Ratio below is **MoE padded TF/s / GEMM padded TF/s**; >1 means higher measured MoE throughput.

| MoE tokens | Matched GEMM tokens / padded M | MoE padded TF/s | GEMM padded TF/s | MoE/GEMM tput ratio |
|---:|---:|---:|---:|---:|
| 8192 | 98304 | 2926.14 | 2766.46 | 1.058 |
| 12288 | 98304 | 2650.75 | 2779.39 | 0.954 |
| 16384 | 196608 | 3143.59 | 2832.34 | 1.110 |
| 24576 | 196608 | 2815.32 | 2837.01 | 0.992 |
| 32768 | 294912 | 2998.20 | 2869.48 | 1.045 |
| 49152 | 393216 | 2875.53 | 2875.53 | 1.000 |
| 65536 | 589824 | 3043.95 | 2877.85 | 1.058 |

gemm的tput 使用 expert = 1, topk=1 moe kernel 模拟的。
12288低于gemm的原因是每个expert只能分到256 tokens, B 完全没有办法复用。 24576每个expert可以分到512个tokens就已经基本相同。
8192, 16384, 32768, 65536， MOE 反而比 gemm高的原因是对于sortte table里面的padding token, MOE 输入buffer load是不会又Vmem load,而且MFMA是A的寄存器输入时0， 所以MFMA 效率反而比padded等效token要高。

#### Repeat evidence for the equal-padded comparison

| MoE tokens | MoE median-min µs | GEMM median-min µs | MoE run-min range µs | GEMM run-min range µs |
|---:|---:|---:|:---|:---|
| 8192 | 211.362 | 223.562 | 209.202–212.482 | 222.362–224.082 |
| 12288 | 233.321 | 222.522 | 232.962–237.562 | 222.282–222.722 |
| 16384 | 393.483 | 436.724 | 393.123–393.604 | 433.804–437.405 |
| 24576 | 439.364 | 436.005 | 437.364–440.564 | 435.964–436.404 |
| 32768 | 618.846 | 646.606 | 618.325–618.966 | 643.366–648.446 |
| 49152 | 860.328 | 860.328 | 860.048–865.688 | 856.528–862.648 |
| 65536 | 1219.090 | 1289.451 | 1215.411–1221.651 | 1280.572–1292.251 |

The ranges are three observed per-run minima, not confidence intervals. GPU6 was checked idle before the sequence; exclusive device access is not guaranteed. Historical PMC/dense-GEMM tables are unchanged and are not combined with these Event timings.
[Summary](../../swiglu_moe_equal_gemm_20260914/summary.json) · [All42 run results and exact commands](../../swiglu_moe_equal_gemm_20260914/results.json) · [Run log](../../swiglu_moe_equal_gemm_20260914/run.log)

### Single-expert Simulated GEMM vs Standalone Scaled A8W4 GEMM (2026-09-14)

Both sides were freshly measured on **GPU0**, not mixed with the GPU6 results above. GPU6 was occupied at preflight and no benchmark ran there for this comparison. No PMC counters or hardware profiler were used.
- Simulated GEMM: test_moe.py, experts=1, topk=1, tokens=M; use its PyHIP result, not AIter throughput.
- Standalone GEMM: unmodified test_mxfp8_gemm_4w.py run_test API, with_scale=True, B_MXFP4=True, A LDS padding/B LDS swizzle, preshuffle=False, permlane=True, store_overlap=False. Calling the API selects all M sizes without editing the hardcoded single-shape __main__.
- Both: N/gate_up=512, K=6144, tile256×256×128 matrix work per block, four waves, XCD8/groupM4; 20 clones,20 warmups,20 Event samples, seed0. Three alternating workload rounds per unique M; report median of three per-run minima.
- Five unique M sizes cover the seven MoE-token rows. Rows mapping to the same M intentionally share that freshly measured result.
- Host-only adapter: standalone A is passed as a zero-copy [M,K] view at compilation/launch rather than a flattened shape. Its original flat tensor shape exceeds signed int32 at M393216 and M589824, causing FlyDSL argument packing to fail. The adapter is applied to every M; pointer, bytes, scale tensors and original kernel source are unchanged. No source change to the standalone or simulated kernel.

**Scale contract:** both use one E8M0 scale per32 K elements, but different physical layouts. Simulated uses native AIter routed A and gate/up-interleaved B scales; standalone uses its legacy K-group-major scale permutation. Each uses its own preparation outside timing; neither scale preparation nor format conversion is timed.
**Other differences:** simulated includes SwiGLU and writes M×256 BF16, whereas standalone is linear and writes M×512 BF16. The original test APIs also generate different input distributions/clone contents. Matrix work2×M×512×6144 matches, but this is not identical-output or scale-layout-only A/B.

| Original MoE tokens | GEMM M | Simulated GEMM padded TF/s | Standalone A8W4 TF/s | Simulated/standalone tput |
|---:|---:|---:|---:|---:|
| 8192 | 98304 | 2851.14 | 2707.36 | 1.053 |
| 12288 | 98304 | 2851.14 | 2707.36 | 1.053 |
| 16384 | 196608 | 2935.04 | 2772.66 | 1.059 |
| 24576 | 196608 | 2935.04 | 2772.66 | 1.059 |
| 32768 | 294912 | 2968.46 | 2790.93 | 1.064 |
| 49152 | 393216 | 2980.58 | 2835.33 | 1.051 |
| 65536 | 589824 | 2991.54 | 2838.32 | 1.054 |

Ratio is simulated/standalone; >1 means greater measured simulated-GEMM throughput. Both throughput numerators count matrix FLOPs only, not activation operations.

#### Repeat evidence: simulated versus standalone

| M | Simulated median-min µs | Standalone median-min µs | Simulated run-min range µs | Standalone run-min range µs |
|---:|---:|---:|:---|:---|
| 98304 | 216.922 | 228.442 | 215.362–217.522 | 227.722–228.722 |
| 196608 | 421.443 | 446.124 | 421.163–422.484 | 445.964–447.284 |
| 294912 | 625.046 | 664.805 | 624.405–627.006 | 664.566–665.166 |
| 393216 | 830.007 | 872.527 | 828.007–830.647 | 870.608–880.768 |
| 589824 | 1240.450 | 1307.410 | 1240.131–1240.851 | 1305.091–1327.172 |

Validation: 30/30 runs completed. Simulated runs passed the original AIter/PyHIP output check. All15 standalone runs passed the original calc_diff≤1e-5 check; strict elementwise allclose failed in 0/15 standalone runs (reported separately, not hidden). No cross-implementation output equality is asserted because epilogues differ.
All three source snapshots matched live hashes before/after measurement. Standalone timing samples are captured at host run_test return using sys.setprofile; GPU timing remains its existing cudaPerf Events. This is not rocprofiler/ATT/PMC. Ranges are observed per-run minima, not confidence intervals; idle preflight is not exclusive GPU ownership.

Source SHA256:
- [test_moe.py](../../simulated_vs_dense_a8w4_20260914/matrix_a_view/data/test_moe.py): `fa4e2de725d94e718f8e2d8ca40d8b34c3e831f71acadd7c9726312255898c97`
- [test_moe_mxfp8_mxfp4_gateup_4w.py](../../simulated_vs_dense_a8w4_20260914/matrix_a_view/data/test_moe_mxfp8_mxfp4_gateup_4w.py): `477a7ce297519d6a3ebaa4b25eb45d9f0f205732280c6b1b78caf30a62bf9f2e`
- [test_mxfp8_gemm_4w.py](../../simulated_vs_dense_a8w4_20260914/matrix_a_view/data/test_mxfp8_gemm_4w.py): `61502281047dff30a9a744a1d2bb19751e4d36cb92e95b6a7171964f3aa94a56`

[Summary](../../simulated_vs_dense_a8w4_20260914/matrix_a_view/summary.json) · [Raw results and commands](../../simulated_vs_dense_a8w4_20260914/matrix_a_view/results.json)

### Current SwiGLU Artifacts

- [ATT capture/phase summary](../../swiglu_phase_sweep_20260914/summary.json)
- [Performance summary and all per-run results](../../swiglu_phase_sweep_20260914/performance/summary.json)
- [Unprofiled run0](../../swiglu_phase_sweep_20260914/performance/run_0.log), [run1](../../swiglu_phase_sweep_20260914/performance/run_1.log), [run2](../../swiglu_phase_sweep_20260914/performance/run_2.log)
- Per-wave phase data: [T12288](../../swiglu_phase_sweep_20260914/tokens12288/phases.json), [T24576](../../swiglu_phase_sweep_20260914/tokens24576/phases.json), [T49152](../../swiglu_phase_sweep_20260914/tokens49152/phases.json)
- [Frozen current kernel](../../swiglu_phase_sweep_20260914/snapshot/test_moe_mxfp8_mxfp4_gateup_4w.py)

## User-supplied MoE and GEMM ATT

These statistics use the user-provided captures, **not** the GPU6/CU1 sweep
above. No kernels were rerun. The GEMM command uses tokens=24576, topk=1,
experts=1, gate_up=512, K=6144: it is the single-expert fused gate/up path,
with 24576 routed rows, not a matched-workload 196608-row dense run.

### User GEMM Phase Breakdown

Dispatches 418 and 420 each contain four complete decoded waves, all on
SE0/CU0: **8 waves total**. Raw trace files exist for SE0–3, but no wave
JSON was present for SE1–3; the statistics do not fabricate missing samples.

| Phase | MFMA/wave | Theoretical MFMA cycles | Mean cycles/wave | Time share | Theoretical MFMA efficiency |
|:---|---:|---:|---:|---:|---:|
| Prologue | 0 | 0 | 9653.000 | 6.853% | N/A |
| Mainloop | 2944 | 94208 | 119973.500 | 85.174% | 78.524007% |
| Epilogue | 128 | 4096 | 11231.000 | 7.973% | 36.470483% |
| Whole wave | 3072 | 98304 | 140857.500 | 100.000% | 69.789681% |

Mainloop cycles/iteration = 119973.5 / 23 = **5216.239130**;
efficiency = 4096 / 5216.239130 = **78.524007%**. Individual dispatch
efficiencies are **81.065639% / 76.136905%**. This is a small sample with
visible variation, not a whole-device MFMA busy measurement.

### Why this differs from the preceding captures

The supplied MoE has 432 complete waves on CU0 across 18 decoded SEs:
mainloop efficiency **85.095088%**, or **84.007713%** when restricted to
its SE0–3 samples. The earlier agent result **89.192237%** uses CU1 on
SE0–3 and two other dispatches. The physical GPU ordinal and capture-time
source hash are not established from the user-provided wave files.

**The user MoE, user GEMM and preceding agent captures have exactly the same
mainloop disassembly, including operands.** The mainloop efficiency formula
is also the same. These observations do not establish a SwiGLU-caused
mainloop regression: CU/SE selection, dispatches and workload are not
controlled A/B samples. The exact cause of the measured cycle differences
has not been isolated.

All provided waves passed stitching, 23-iteration/backedge/MFMA-index and
phase-duration-sum validation. A capture log was not supplied, so this is not
a claim of independently verified absence of raw packet loss.
Details: [user capture analysis and per-SE tables](../../user_att_20260914/REPORT.md),
[per-wave results and input hashes](../../user_att_20260914/summary.json).

## 2026-09-14 Earlier SwiGLU Snapshot ATT Phase Breakdown

The following 695be7... capture predates the user-added epilogue wait/barrier;
it is retained as earlier-snapshot data, not the current three-shape sweep.

Current activation is **SiLU(gate) × up, without clamping**. This update uses
the source-frozen SwiGLU capture for tokens=24576, intermediate=256
(gate_up=512), K=6144, topk=8, experts=384: GPU6, target CU1 on SE0–SE3,
all four SIMDs, two dispatches (437/443), **192 complete waves**.
Kernel SHA256: `695be7a39a2a34bfdc8024fdfa2d8c63109e94c07343d615f9bc4fc001484c89`.

Boundaries are `wave.begin -> first mainloop entry -> first epilogue entry -> wave.end`.
Every wave passed 23-iteration/MFMA-count validation and the three elapsed
durations sum exactly to its lifetime. Time share is summed phase cycles divided
by summed whole-wave cycles; mean cycles give equal weight to each wave.
The epilogue includes the final two K tiles, activation, packing and stores.

| Phase | MFMA/wave | Theoretical MFMA cycles | Mean cycles/wave | Time share | Theoretical MFMA efficiency |
|:---|---:|---:|---:|---:|---:|
| Prologue | 0 | 0 | 9287.292 | 7.388% | N/A |
| Mainloop | 2944 | 94208 | 105428.708 | 83.867% | 89.357066% |
| Epilogue | 128 | 4096 | 10993.208 | 8.745% | 37.259368% |
| Whole wave | 3072 | 98304 | 125709.208 | 100.000% | 78.199522% |

Compared with the preceding same-scope SiTUv2 capture (not the older MoE/GEMM
tables below), phase shares changed from **6.748% / 75.339% / 17.913%** to
**7.388% / 83.867% / 8.745%**. Mainloop absolute cycles changed only +0.027%;
its higher share primarily reflects the epilogue shrinking by 56.133%.
These are separate captures, not interleaved ATT A/B samples.

Theoretical MFMA efficiency is theoretical MFMA cycles / phase elapsed cycles,
not the PMC `MfmaUtil` counter or an additive instruction issue/stall share.
No new matched dense-GEMM capture or PMC roofline run was performed; **all
older roofline, MoE/GEMM and per-CU tables below remain historical**.

Details: [current phase comparison and per-dispatch shares](../../swiglu_20260914/REPORT.md#att-prologue--mainloop--epilogue-breakdown),
[SwiGLU ATT report](../../swiglu_att_20260914/REPORT.md), and
[per-wave phase validation](../../swiglu_att_20260914/phases.json).

## 2026-09-12 Native A-scale follow-up

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

## Historical ATT Mainloop MFMA Efficiency

**Historical MoE/GEMM comparison, not the 2026-09-14 SwiGLU sweep or the
user-supplied captures above.** The cycles and efficiencies below are retained
from their original captures. The formula is the same; the underlying samples
are different. No current matched dense-GEMM comparison has replaced this table.

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
| 24576 | 196608 | 96/96 | 4447.780797 | 92.090869% | 96/96 | 4475.617754 | 91.518093% | +0.572777 pp |
| 49152 | 393216 | 192/192 | 4375.557065 | 93.610938% | 192/192 | 4421.855072 | 92.630806% | +0.980132 pp |
| 16384 | 196608 | 96/96 | 4292.257246 | 95.427645% | 96/96 | 4517.317029 | 90.673291% | +4.754354 pp |
| 32768 | 294912 | 144/144 | 4312.907005 | 94.970747% | 144/144 | 4558.710145 | 89.849977% | +5.120770 pp |
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


