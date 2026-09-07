# A8W8 no-scale ATT comparison on gfx950

Date: 2026-09-07

Shape: `M=4096, N=4096, K=16384`

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
