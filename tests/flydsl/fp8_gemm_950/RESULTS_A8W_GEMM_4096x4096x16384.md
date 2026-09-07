# A8W8/A8W4 ATT results on gfx950

Date: 2026-09-07

Shape: `M=4096, N=4096, K=16384`

## Current `test_mxfp8_gemm_4w.py` snapshot: with scale vs without scale

This section profiles the current kernel directly through
`test_mxfp8_gemm_4w.py`. It is newer than the historical FlyDSL/Gluon
comparison below, so its no-scale result must not be compared as though it were
the same kernel snapshot as the historical 96.34% result.

Both modes use `USE_SWIZZLE=0`, `PRESHUFFLE_B=0`, `B_MXFP4=False`, BF16 output,
50 data clones, and 50 timed launches. The scaled mode uses MXFP8 E4M3 inputs
with E8M0 scales and `SCALE_G2R=1`; the no-scale mode uses E4M3 inputs. FlyDSL's
runtime cache was disabled so the traces reflect the current source.

### Results

Mainloop efficiency covers only the 63 complete steady-state K-loop
iterations. Both modes issue 128 MFMAs per iteration, and each MFMA accounts
for 32 cycles, so the theoretical work is 4096 cycles per iteration. Event
latency and TFLOPS are end-to-end best-of-50 measurements from
`pyhip.cudaPerf` (`torch.cuda.Event`).

| Metric | Without scale | With scale | With scale - without scale |
|---|---:|---:|---:|
| MFMA instructions / iteration | 128 | 128 | 0 |
| Theoretical MFMA cycles / iteration | 4096 | 4096 | 0 |
| Observed mainloop cycles / iteration | 4401.905 | 4753.079 | +351.175 (+7.98%) |
| Non-MFMA overhead / iteration | 305.905 | 657.079 | +351.175 (+114.80%) |
| Mainloop MFMA efficiency | 93.05% | 86.18% | -6.87 pp |
| Best Event latency | 182.321 us | 193.002 us | +10.681 us (+5.86%) |
| Best Event throughput | 3015.32 TFLOPS | 2848.45 TFLOPS | -166.87 TFLOPS (-5.53%) |
| Benchmark validation (`is_correct`) | `True` | `True` | - |
| Reported `diff` | 2.3599e-08 | 2.5713e-08 | +2.1135e-09 |

The no-scale trace uses `v_mfma_f32_16x16x128_f8f6f4`; the scaled trace uses
`v_mfma_scale_f32_16x16x128_f8f6f4`. In this snapshot, scale handling more than
doubles the non-MFMA mainloop overhead. This ATT result describes one sampled
wave's steady-state loop; the Event results describe the complete kernel.

### Profile commands and traces

For each command, only the corresponding `with_scale` invocation under
`__main__` was enabled. The ATT configuration selects the 15th matching
`gemm_kernel` invocation on CU 0.

Without scale:

```bash
cd /mywork/pyhip/tests/flydsl/fp8_gemm_950

HIP_VISIBLE_DEVICES=5 \
ROCPROF_ATT_LIBRARY_PATH=/opt/rocm/lib \
FLYDSL_RUNTIME_ENABLE_CACHE=0 \
SCALE_G2R=1 \
PYTHONPATH=/mywork/pyhip \
/opt/rocm/bin/rocprofv3 --att \
	-i /mywork/pyhip/tests/flydsl/att_matmul.json \
	-d /tmp/a8w8-noscale-test-file-att-20260907 \
	-- python ./test_mxfp8_gemm_4w.py
```

With scale, freshly rerun for this comparison:

```bash
cd /mywork/pyhip/tests/flydsl/fp8_gemm_950

HIP_VISIBLE_DEVICES=5 \
ROCPROF_ATT_LIBRARY_PATH=/opt/rocm/lib \
FLYDSL_RUNTIME_ENABLE_CACHE=0 \
SCALE_G2R=1 \
PYTHONPATH=/mywork/pyhip \
/opt/rocm/bin/rocprofv3 --att \
	-i /mywork/pyhip/tests/flydsl/att_matmul.json \
	-d /tmp/a8w8-scaled-test-file-att-20260907-rerun \
	-- python ./test_mxfp8_gemm_4w.py
```

Decode the steady-state mainloops:

```bash
python /mywork/gfx950-gluon-tutorials/scripts/process_json.py \
	/tmp/a8w8-noscale-test-file-att-20260907/ui_output_agent_63787_dispatch_315

python /mywork/gfx950-gluon-tutorials/scripts/process_json.py \
	/tmp/a8w8-scaled-test-file-att-20260907-rerun/ui_output_agent_4579_dispatch_864
```

Fresh direct-test traces:

- Without scale: `/tmp/a8w8-noscale-test-file-att-20260907/ui_output_agent_63787_dispatch_315`
- With scale: `/tmp/a8w8-scaled-test-file-att-20260907-rerun/ui_output_agent_4579_dispatch_864`

## Current A8W4 with-scale snapshot

This profile uses MXFP8 E4M3 A, MXFP4 E2M1 B, E8M0 block scales, BF16 output,
A padding (`USE_SWIZZLE=0`), and the default B LDS swizzle
(`B_LDS_SWIZZLE=True`) at `M=N=4096, K=16384`. The tuned source issues each
G2R scale load before the data DMA for the same A or B phase.

| Metric | Before tuning | Scale-first tuned | Change |
|---|---:|---:|---:|
| Benchmark validation (`is_correct`) | `True` | `True` | unchanged |
| Reported `diff` | 7.4168e-09 | 7.4168e-09 | unchanged |
| Best Event latency (uninstrumented) | 191.441 us | 183.961 us | -3.91% |
| Median Event latency (50 runs) | 194.402 us | 186.162 us | -4.24% |
| Best Event throughput (uninstrumented) | 2871.67 TFLOPS | 2988.44 TFLOPS | +4.07% |
| Complete steady-state K iterations | 63 | 63 | unchanged |
| MFMA instructions / iteration | 128 | 128 | unchanged |
| Theoretical MFMA cycles / iteration | 4096 | 4096 | unchanged |
| Observed mainloop cycles / iteration | 5494.286 | 4966.000 | -9.62% |
| Non-MFMA overhead / iteration | 1398.286 | 870.000 | -37.78% |
| `s_waitcnt` stall cycles / iteration | 587.1 | 196.5 | -66.53% |
| Mainloop MFMA efficiency | 74.55% | 82.48% | +7.93 pp |

### A8W4 with-scale waitcnt audit

The explicit VMEM wait values are dependency-tight and were not relaxed by the
tuning. Each A phase issues four data VMEM operations and one scale VMEM
operation, while each B phase issues two data VMEM operations and one scale
VMEM operation. One tile therefore issues `B_l(3), A_t(5), A_b(5), B_r(3)`, or
16 VMEM operations; the two-tile prologue has 32 outstanding operations.

On gfx950, `vmcnt(N)` permits at most N newer VMEM operations to remain
outstanding. The prologue `vmcnt(24)` therefore completes exactly the oldest
eight operations needed by the first `B_l + A_t` consumers. The emitted
steady-state sequence `vmcnt(19), vmcnt(19), vmcnt(21), vmcnt(21)` advances the
same 3/5-operation phase boundaries before each LDS consumer. The drain values
`19, 16, 13, 8, 3, 0` follow the remaining 3/5-operation groups. Reducing any
of these values would cross a producer-consumer boundary.

The dominant baseline bubble instead came from compiler-generated loop-latch
waits: scale VMEM operations were issued after their same-phase data DMA, and
the final `vmcnt(3)` stalled about 383 cycles per iteration. Issuing each scale
load first lets LLVM use `vmcnt(15)` and `vmcnt(7)` at the loop latch; the latter
stalls only about 7 cycles per iteration. The manual phase waits above remain
unchanged.

A scheduler-position sweep (`SCALE_VMEM_POS=3,5,7,9,11,13`, plus scheduling
disabled) found no stable Event-time winner, so the default position remains 7.
The B-phase VMEM accounting is retained: removing it increased the ATT mainloop
from 4966.000 to 5046.603 cycles per iteration and reduced efficiency from
82.48% to 81.16%.

The final ISA confirms that register allocation does not spill:

| ISA resource | Value |
|---|---:|
| VGPRs (`.amdhsa_next_free_vgpr`) | 464 |
| SGPRs (`.amdhsa_next_free_sgpr`) | 96 |
| AGPR offset (`.amdhsa_accum_offset`) | 208 |
| Private segment | 0 B |
| Dynamic stack | disabled |
| `scratch_load` / `scratch_store` instructions | 0 |

The ISA dump is
`/tmp/a8w4-scaled-scale-first-restored-isa-20260907/gemm_kernel_0/21_final_isa.s`.
It is byte-identical to the ISA used for the tuned ATT and PMC runs.

A separate PMC run reports no LDS bank conflicts or LDS data-FIFO stalls for
the complete kernel:

| Hardware counter | Samples | Minimum | Maximum | Sum | Nonzero samples |
|---|---:|---:|---:|---:|---:|
| `SQ_LDS_BANK_CONFLICT` | 64 | 0 | 0 | 0 | 0 |
| `SQ_LDS_DATA_FIFO_FULL` | 64 | 0 | 0 | 0 | 0 |

PMC database:
`/tmp/a8w4-scaled-scale-first-pmc-20260907/smci355-ccs-aus-m09-09/740400_results.db`.

ATT trace and analyzer command:

```bash
python /mywork/gfx950-gluon-tutorials/scripts/process_json.py \
	/tmp/a8w4-scaled-scale-first-att-20260907/ui_output_agent_20341_dispatch_848
```

The ATT analyzer found 128 scaled MFMAs in each of 63 complete K-loop
iterations. The 4096 theoretical MFMA cycles divided by the observed 4966.000
cycles per iteration gives 82.48% steady-state MFMA efficiency. Event results
from ATT-instrumented runs are not used as the normal performance values above.

## Current no-scale snapshot: A8W8 vs A8W4

This comparison uses the same shape, direct test entry point, ATT settings,
50 data clones, and 50 timed launches as the current results above. Both modes
use E4M3 A without scale. A8W8 uses E4M3 B. A8W4 uses E2M1 FP4 B packed two
elements per byte, with A padding (`USE_SWIZZLE=0`) and independent B LDS
swizzle (`B_LDS_SWIZZLE=True`).

Both traces execute 128 unscaled `v_mfma_f32_16x16x128_f8f6f4` instructions
per iteration. The A8W4 encoding additionally uses `cbsz:4`. The theoretical
MFMA work is therefore 4096 cycles per iteration for both modes.

| Metric | A8W8 without scale | A8W4 without scale | A8W4 - A8W8 |
|---|---:|---:|---:|
| MFMA instructions / iteration | 128 | 128 | 0 |
| Theoretical MFMA cycles / iteration | 4096 | 4096 | 0 |
| Observed mainloop cycles / iteration | 4401.905 | 4552.857 | +150.952 (+3.43%) |
| Non-MFMA overhead / iteration | 305.905 | 456.857 | +150.952 (+49.35%) |
| Mainloop MFMA efficiency | 93.05% | 89.97% | -3.08 pp |
| Best Event latency | 182.321 us | 174.281 us | -8.040 us (-4.41%) |
| Best Event throughput | 3015.32 TFLOPS | 3154.42 TFLOPS | +139.10 TFLOPS (+4.61%) |
| Benchmark validation (`is_correct`) | `True` | `True` | - |
| Reported `diff` | 2.3599e-08 | 6.3912e-11 | -2.3535e-08 |

A8W4 has 3.08 percentage points lower steady-state MFMA efficiency because its
sampled mainloop spends 150.952 more cycles per iteration outside the MFMA
accounting interval. Its complete kernel is nevertheless 4.41% faster because
the packed FP4 B operand halves B storage and traffic. Mainloop efficiency and
end-to-end latency therefore rank these two kernels differently.

This is a comparison of the current kernel paths, not a scheduler-controlled
datatype experiment. A8W8 without scale selects `hot_loop_a8w8_noscale`, while
A8W4 selects `hot_loop_scheduler_mainloop`.

Profile A8W4 without scale with only the
`with_scale=False, B_MXFP4=True, B_LDS_SWIZZLE=True` invocation enabled under
`__main__`:

```bash
cd /mywork/pyhip/tests/flydsl/fp8_gemm_950

HIP_VISIBLE_DEVICES=5 \
ROCPROF_ATT_LIBRARY_PATH=/opt/rocm/lib \
FLYDSL_RUNTIME_ENABLE_CACHE=0 \
PYTHONPATH=/mywork/pyhip \
/opt/rocm/bin/rocprofv3 --att \
	-i /mywork/pyhip/tests/flydsl/att_matmul.json \
	-d /tmp/a8w4-noscale-a-pad-b-swizzle-att-20260907 \
	-- python ./test_mxfp8_gemm_4w.py

python /mywork/gfx950-gluon-tutorials/scripts/process_json.py \
	/tmp/a8w4-noscale-a-pad-b-swizzle-att-20260907/ui_output_agent_34773_dispatch_271
```

No-scale comparison traces:

- A8W8: `/tmp/a8w8-noscale-test-file-att-20260907/ui_output_agent_63787_dispatch_315`
- A8W4, A padding + B swizzle: `/tmp/a8w4-noscale-a-pad-b-swizzle-att-20260907/ui_output_agent_34773_dispatch_271`

### A8W4 B padding vs B swizzle

An alternating no-profiler test used `test_mxfp8_gemm_4w.py` with the same 50
clones and 50 Event measurements per round. Only `B_LDS_SWIZZLE` changed.

| Best-of-50 result | A padding + B padding | A padding + B swizzle |
|---|---:|---:|
| Round 1 latency | 173.002 us | 172.481 us |
| Round 1 throughput | 3177.74 TFLOPS | 3187.34 TFLOPS |
| Round 2 latency | 173.242 us | 173.082 us |
| Round 2 throughput | 3173.34 TFLOPS | 3176.27 TFLOPS |
| Mean of per-round best latency | 173.122 us | 172.782 us |
| Mean of per-round best throughput | 3175.54 TFLOPS | 3181.81 TFLOPS |
| Median of all 100 Event samples | 175.302 us | 175.322 us |

The best-of-50 summaries favor B swizzle by about 0.20%, while the aggregate
median differs by only 0.012% in the opposite direction. The two layouts are
therefore performance-equivalent at this shape within run-to-run noise; there
is no statistically meaningful performance winner from these samples.

MXFP4 now defaults to B LDS swizzle when `B_LDS_SWIZZLE` is omitted. Passing
`B_LDS_SWIZZLE=False` remains available for explicit B-padding comparisons;
non-MXFP4 paths continue to inherit the A LDS layout.

Matched hardware-counter runs used the current `test_mxfp8_gemm_4w.py` source
and changed only `B_LDS_SWIZZLE`. B padding reports conflicts in every sampled
counter instance, whereas B swizzle reports none:

```bash
HIP_VISIBLE_DEVICES=5 \
FLYDSL_RUNTIME_ENABLE_CACHE=0 \
PYTHONPATH=/mywork/pyhip \
/opt/rocm/bin/rocprofv3 \
	--pmc 'SQ_LDS_BANK_CONFLICT,SQ_LDS_DATA_FIFO_FULL' \
	--kernel-include-regex 'gemm_kernel' \
	-d /tmp/a8w4-noscale-a-pad-b-<layout>-pmc-20260907 \
	-- python -c '<single run_test call with B_LDS_SWIZZLE=False or True>'
```

| B LDS layout | Hardware counter | Samples | Value / instance | Sum | Nonzero samples |
|---|---|---:|---:|---:|---:|
| Padding | `SQ_LDS_BANK_CONFLICT` | 64 | 131072 | 8388608 | 64 |
| Swizzle | `SQ_LDS_BANK_CONFLICT` | 64 | 0 | 0 | 0 |
| Padding | `SQ_LDS_DATA_FIFO_FULL` | 64 | 0 | 0 | 0 |
| Swizzle | `SQ_LDS_DATA_FIFO_FULL` | 64 | 0 | 0 | 0 |

Matched profile databases:

- B padding: `/tmp/a8w4-noscale-a-pad-b-padding-pmc-20260907/smci355-ccs-aus-m09-09/711403_results.db`
- B swizzle: `/tmp/a8w4-noscale-a-pad-b-swizzle-pmc-matched-20260907/smci355-ccs-aus-m09-09/711924_results.db`

## Historical FlyDSL/Gluon no-scale snapshot

The remaining results predate the direct-test snapshot above and compare an
older FlyDSL no-scale kernel with Gluon.

Compared kernels:

- FlyDSL: E4M3 x E4M3, BF16 output, no scale.
- Gluon: E5M2 x E5M2, FP16 output, `scale=None`.

The host does not provide a separate `rocm_trace` command, so the traces were
collected with ROCm Advanced Thread Trace through `rocprofv3 --att`.

## Commands

### CUDA/HIP Event performance comparison

This command uses `pyhip.cudaPerf` (`torch.cuda.Event`) rather than profiler
timestamps. The explicit arguments are also the script defaults: 50 FP8 clones,
50 measurements, and seed 0.

```bash
cd /mywork/pyhip/tests/flydsl/fp8_gemm_950

PYTHONPATH=/tmp/triton-gfx950-v11-site:/mywork/pyhip \
LLVM_PASS_PLUGIN_PATH=/mywork/gfx950-gluon-tutorials/plugins/llir_scheduler/libLlirSched.so \
LLVM_PASS_PLUGIN_KEEP_TARGET_MACHINE=1 \
TRITON_FORCE_MFMA_AGPR=1 \
TRITON_AMDGCNAS_PLUGIN=1 \
TRITON_CACHE_DIR=/tmp/a8w8_noscale_compare_cache \
python compare_a8w8_noscale_cold.py \
	--m 4096 --n 4096 --k 16384 --clones 50 --runs 50 --seed 0
```

The benchmark creates one FP32 source for each of A and B, converts each source
to E4M3 and E5M2, deletes the FP32 source, and clones only the FP8 tensors.
FlyDSL and Gluon therefore receive values derived from the same source, in the
native FP8 format expected by each unchanged kernel.

### ATT configurations

FlyDSL configuration, used as
`/mywork/pyhip/tests/flydsl/att_matmul.json`:

```json
{
	"jobs": [
		{
			"kernel_include_regex": "gemm_kernel",
			"kernel_exclude_regex": "",
			"kernel_iteration_range": "[15]",
			"advanced_thread_trace": true,
			"att_target_cu": 0,
			"att_shader_engine_mask": "0xF",
			"att_simd_select": "0xF",
			"att_buffer_size": "0x60000000"
		}
	]
}
```

Gluon configuration, used as `/tmp/att_a8w8_gluon.json`:

```json
{
	"jobs": [
		{
			"kernel_include_regex": "a8w8_kernel",
			"kernel_exclude_regex": "",
			"kernel_iteration_range": "[15]",
			"advanced_thread_trace": true,
			"att_target_cu": 0,
			"att_shader_engine_mask": "0xF",
			"att_simd_select": "0xF",
			"att_buffer_size": "0x60000000"
		}
	]
}
```

Separate regexes are required because the emitted kernel symbols are
`gemm_kernel_0` and `a8w8_kernel`, respectively. `[15]` selects the 15th
matching invocation, after compilation and warmup-related activity.

### ROCm ATT profile commands

FlyDSL E4M3 A8W8 without scale:

```bash
cd /mywork/pyhip/tests/flydsl

ROCPROF_ATT_LIBRARY_PATH=/opt/rocm/lib \
PYTHONPATH=/mywork/pyhip \
rocprofv3 --att \
	-i att_matmul.json \
	-d /tmp/a8w8-noscale-att-rocmtrace-20260907 \
	-- python /tmp/compare_scaled_gemms.py \
		--target flydsl-a8w8-noscale --mode att --att-runs 16
```

Gluon E5M2 A8W8 without scale:

```bash
cd /mywork/pyhip/tests/flydsl

ROCPROF_ATT_LIBRARY_PATH=/opt/rocm/lib \
PYTHONPATH=/tmp/triton-gfx950-v11-site:/mywork/pyhip \
LLVM_PASS_PLUGIN_PATH=/mywork/gfx950-gluon-tutorials/plugins/llir_scheduler/libLlirSched.so \
LLVM_PASS_PLUGIN_KEEP_TARGET_MACHINE=1 \
TRITON_FORCE_MFMA_AGPR=1 \
TRITON_AMDGCNAS_PLUGIN=1 \
TRITON_CACHE_DIR=/tmp/gluon-a8w8-att-cache-20260907 \
rocprofv3 --att \
	-i /tmp/att_a8w8_gluon.json \
	-d /tmp/gluon-a8w8-att-rocmtrace-20260907-v2 \
	-- python /tmp/compare_scaled_gemms.py \
		--target gluon-a8w8 --mode att --att-runs 16
```

Decode and calculate the steady-state mainloop results:

```bash
python /mywork/gfx950-gluon-tutorials/scripts/process_json.py \
	/tmp/a8w8-noscale-att-rocmtrace-20260907/ui_output_agent_32169_dispatch_21

python /mywork/gfx950-gluon-tutorials/scripts/process_json.py \
	/tmp/gluon-a8w8-att-rocmtrace-20260907-v2/ui_output_agent_54098_dispatch_20
```

## MFMA efficiency

Efficiency is calculated only over the 63 complete steady-state mainloop
iterations of a traced wave:

```text
MFMA efficiency = theoretical MFMA cycles / observed mainloop cycles
```

Both loops execute 128 MFMA instructions per iteration. Each tested
`16x16x128_f8f6f4` instruction accounts for 32 cycles, giving 4096 theoretical
MFMA cycles per iteration.

| Metric | FlyDSL | Gluon | FlyDSL - Gluon |
|---|---:|---:|---:|
| MFMA instructions / iteration | 128 | 128 | 0 |
| Theoretical MFMA cycles / iteration | 4096 | 4096 | 0 |
| Observed mainloop cycles / iteration | 4251.746 | 4115.746 | +136.000 |
| Non-MFMA overhead / iteration | 155.746 | 19.746 | +136.000 |
| Mainloop MFMA efficiency | 96.34% | 99.52% | -3.18 pp |

Gluon removes 87.3% of the non-MFMA mainloop overhead present in FlyDSL.

### Raw mainloop profile results

| `process_json.py` field | FlyDSL | Gluon |
|---|---:|---:|
| `loop_first_index` | 582 | 621 |
| `loop_last_index` | 859 | 903 |
| `epilogue_first_index` | 860 | 904 |
| `loop_hitcount` | 504 | 504 |
| `epilogue_hitcount` | 8 | 8 |
| `num_iterations` | 63 | 63 |
| `mfma_count_in_loop` | 128 | 128 |
| `total_mfma_cycles_in_loop` | 4096 | 4096 |
| SE0/SM0 mainloop duration | 267,840 cycles | 259,292 cycles |
| SE2/SM0 mainloop duration | 267,880 cycles | 259,292 cycles |
| Average mainloop duration | 267,860 cycles | 259,292 cycles |
| Average mainloop duration / iteration | 4251.746 cycles | 4115.746 cycles |
| Mainloop MFMA efficiency | 96.34% | 99.52% |

The mainloop calculation excludes prologue and epilogue. It uses the average of
the two decoded SM0 waves and divides by 63 complete loop iterations.

## Where the performance gap is

### 1. Mainloop scheduling: 70.3% of the traced-wave gap

The two loops perform essentially the same amount of work per iteration:

| Dynamic instruction class / iteration | FlyDSL | Gluon |
|---|---:|---:|
| MFMA | 128 | 128 |
| LDS read | 64 | 64 |
| Global buffer load | 32 | 32 |
| `s_waitcnt` | 8 | 8 |
| `s_barrier` | 8 | 8 |
| SALU, excluding branch | 36 | 42 |

FlyDSL does not lose efficiency because it executes more MFMA, LDS, or global
load instructions. The difference is their placement. In FlyDSL, each group of
16 MFMAs ends with a roughly 72-cycle MFMA-to-MFMA interval containing
`s_waitcnt`, `s_barrier`, and the first LDS reads of the next phase. Its loop
back-edge interval is about 132 cycles. Gluon distributes LDS, global-load, and
scalar work through the MFMA stream; its largest common internal intervals are
44-48 cycles and its loop-back interval is about 60 cycles.

Relative to an ideal 32-cycle interval, the eight 16-MFMA regions contribute:

| Region | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | Total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FlyDSL extra cycles | 12 | 11.94 | 12 | 12 | 12 | 12 | 12 | 72 | 155.94 |
| Gluon extra cycles | -20 | 4 | 4 | 0 | 0 | 12 | -12 | 32 | 20.00 |
| Difference | 32 | 7.94 | 8 | 12 | 12 | 0 | 24 | 40 | 135.94 |

Negative regional values mean neighboring MFMAs were issued closer than the
32-cycle accounting interval; they are not negative work. The total across all
regions is the relevant quantity.

The decoder's stall tags support the same diagnosis:

| Stall-tagged cycles / iteration | FlyDSL | Gluon |
|---|---:|---:|
| LDS read | 224.00 | 36.00 |
| Global buffer load | 129.55 | 0.19 |
| Barrier | 159.55 | 104.00 |
| Explicit waitcnt | 32.00 | 32.00 |

These categories overlap with useful MFMA execution and therefore must not be
added as a causal cycle breakdown. They show that FlyDSL hides LDS/global issue
latency less effectively, while the explicit wait count itself is identical.

### 2. Epilogue: 25.2% of the traced-wave gap

| Phase | FlyDSL cycles | Gluon cycles | Difference | Share of total gap |
|---|---:|---:|---:|---:|
| Prologue | 4,844 | 4,296 | +548 | 4.5% |
| Mainloop | 267,860 | 259,292 | +8,568 | 70.3% |
| Epilogue | 15,676 | 12,608 | +3,068 | 25.2% |
| Total traced wave | 288,380 | 276,196 | +12,184 | 100.0% |

FlyDSL's epilogue contains 356 `v_accvgpr_read_b32`, 100
`v_accvgpr_write_b32`, 128 BF16 conversions, and 64 `v_permlane16_swap`
instructions. Gluon contains 324 AGPR reads, 68 AGPR writes, and 128 FP16
conversions, with a different LDS-based output path. Thus FlyDSL has 64 more
AGPR transfer instructions and a longer output path in this trace.

This epilogue comparison is not format-neutral: FlyDSL writes BF16 while Gluon
writes FP16. The 3,068-cycle difference cannot be attributed solely to the
compiler or mainloop implementation.

### 3. Overall interpretation

The primary comparable gap is mainloop instruction interleaving. Gluon's
LLIR scheduler plus force-AGPR and post-assembly peephole keep memory and scalar
operations interleaved with MFMA execution. FlyDSL's generated loop clusters
the wait/barrier/LDS transition work at each 16-MFMA boundary, leaving 136 more
cycles per iteration despite having the same memory-operation counts.

The fresh ATT trace makes the Gluon mainloop only 3.20% shorter, while its MFMA
efficiency is 3.18 percentage points higher. Whole-kernel performance can have
additional differences from the output dtype, epilogue, workgroup scheduling,
occupancy, and memory-system behavior; a single-wave ATT result should not be
used as a complete end-to-end latency model.

## Trace details

ATT settings:

- 15th matching kernel invocation
- target CU 0
- shader-engine mask `0xF`
- SIMD select `0xF`
- ATT buffer size `0x60000000`
- loop duration averaged over the traced SM0 wave from SE0 and SE2

Fresh decoded traces:

- FlyDSL: `/tmp/a8w8-noscale-att-rocmtrace-20260907/ui_output_agent_32169_dispatch_21`
- Gluon: `/tmp/gluon-a8w8-att-rocmtrace-20260907-v2/ui_output_agent_54098_dispatch_20`

No kernel source was modified for this profiling comparison.
