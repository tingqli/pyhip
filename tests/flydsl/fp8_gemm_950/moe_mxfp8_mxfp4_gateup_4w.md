# MXFP8 x MXFP4 MoE Gate/Up 4-Wave Development Log

Date: 2026-09-01  
Device: gfx950 (`HIP_VISIBLE_DEVICES=4`)

## Scope

The new FlyDSL kernel computes raw expert gate and up projections. It does not
apply SiLU and does not multiply gate by up.

- Input A: MXFP8, logical `[tokens, hidden_size]`
- Expert B: MXFP4, logical `[experts, 2 * intermediate_size, hidden_size]`
- Output C: BF16, `[tokens, topk, 2 * intermediate_size]`
- Required case: tokens 8192, topk 8, experts 256, intermediate size 512,
  hidden size 6144
- Sorting block: 256 rows
- Tile: `256 x 256 x 128`, four waves
- LDS: padding layout, ping-pong buffers

Each workgroup computes one 256-row sorted routing block and one pair of
128-column projections. Gate occupies the first half of the output's last
dimension and up occupies the second half.

## Routing And Bounds

The balanced test routing maps choice `(token, slot)` to
`(token * topk + slot) % experts`. For the required case every expert receives
exactly 256 choices, so there are 256 expert blocks. The small accuracy case
uses 96 tokens, topk 2, and 3 experts; each expert has 64 valid choices and 192
padding rows.

A, B, and C use buffer resources.

- Sorted IDs encode `(slot << 24) | token`.
- Padding encodes `(topk << 24) | tokens`.
- An A padding row addresses the first byte after the bounded A resource, so
  `raw_ptr_buffer_load_lds` supplies zero.
- C stores require valid sorted row, token, and slot predicates. Invalid writes
  are suppressed with the buffer-store mask.
- A scales are gathered into sorted-row order on the host. Padding scales use
  E8M0 unity (`0x7f`).

A dedicated 16-byte routing probe separately verified indirect buffer-resource
loads and masked no-op stores.

## Accuracy

Commands:

```bash
cd /mywork/pyhip
HIP_VISIBLE_DEVICES=4 python tests/flydsl/fp8_gemm_950/test_moe_mxfp8_mxfp4_gateup_4w.py --routing-probe
HIP_VISIBLE_DEVICES=4 python tests/flydsl/fp8_gemm_950/test_moe_mxfp8_mxfp4_gateup_4w.py --small-accuracy
HIP_VISIBLE_DEVICES=4 python tests/flydsl/fp8_gemm_950/test_moe_mxfp8_mxfp4_gateup_4w.py --accuracy
```

Results:

| Case | Output shape | Padding | allclose | max abs | calc_diff |
|---|---:|---:|---:|---:|---:|
| Small | `[96, 2, 256]` | 192 rows/expert | true | 0.5 | `4.73e-9` |
| Required | `[8192, 8, 1024]` | none | true | 4.0 | `7.45e-9` |

The comparison uses dequantized MXFP8/MXFP4 expert-wise PyTorch matmul,
BF16 output, `rtol=0.02`, and `atol=0.01`.

## Performance

The equal-work GEMM is `M=N=8192, K=6144`. Both launches use 1024 workgroups
and perform the same useful FLOPs with the same A8W4 MFMA and tile shape.

Paired command:

```bash
HIP_VISIBLE_DEVICES=4 python tests/flydsl/fp8_gemm_950/test_moe_mxfp8_mxfp4_gateup_4w.py --benchmark --warmup 5 --iterations 20
```

Final paired hot-data result:

| Kernel | Best | Median | Best throughput |
|---|---:|---:|---:|
| Optimized MoE | 0.456 ms | 0.475 ms | 1806.65 TFLOPS |
| Equal-work GEMM | 0.308 ms | 0.335 ms | 2680.14 TFLOPS |

The optimized MoE is 1.483x the GEMM latency and reaches 67.4% of GEMM
throughput.

### XCD Launch-Order Sweep (Intermediate Size 256)

Date: 2026-09-03

Configuration: tokens 8192, topk 8, experts 256, intermediate size 256,
hidden size 6144, MXFP8 activations, MXFP4 weights, BF16 output, padding LDS,
20 data clones, 5 warmups, and 20 measured iterations. Balanced routing gives
each expert exactly 256 rows.

The MoE output is `[8192, 8, 512]`. The paired standalone GEMM used by the
benchmark is `M=8192, N=4096, K=6144`, whose `[8192, 4096]` output is the MoE
output with its topk and projection dimensions flattened. Both execute
`2 * 8192 * 8 * 512 * 6144` useful FLOPs. This is an equal-output-shape and
equal-FLOP comparison; it is not `M=65536, N=512, K=6144`.

Commands:

```bash
python tests/flydsl/fp8_gemm_950/test_moe_mxfp8_mxfp4_gateup_4w.py \
  --benchmark --tokens 8192 --intermediate-size 256 --hidden-size 6144 \
  --topk 8 --num-experts 256 --data-clones 20 --warmup 5 --iterations 20

for group in 4 8 16 32; do
  python tests/flydsl/fp8_gemm_950/test_moe_mxfp8_mxfp4_gateup_4w.py \
    --benchmark --tokens 8192 --intermediate-size 256 --hidden-size 6144 \
    --topk 8 --num-experts 256 --data-clones 20 --warmup 5 --iterations 20 \
    --xcd-swizzle --group-size-m "$group"
done
```

MoE results:

| Launch order | Best (ms) | Median (ms) | Best TFLOPS | Median TFLOPS |
|---|---:|---:|---:|---:|
| No XCD swizzle | 0.172921 | 0.176362 | 2384.42 | 2337.90 |
| XCD, `GROUP_SIZE_M=4` | **0.171601** | 0.175361 | **2402.76** | 2351.25 |
| XCD, `GROUP_SIZE_M=8` | 0.172202 | **0.175121** | 2394.38 | **2354.47** |
| XCD, `GROUP_SIZE_M=16` | 0.172841 | 0.175962 | 2385.53 | 2343.22 |
| XCD, `GROUP_SIZE_M=32` | 0.177002 | 0.179961 | 2329.45 | 2291.15 |

Paired `8192 x 4096 x 6144` standalone GEMM results from the same processes:

| MoE launch paired with GEMM | GEMM best (ms) | GEMM median (ms) | GEMM best TFLOPS | GEMM median TFLOPS | MoE/GEMM latency | MoE/GEMM throughput |
|---|---:|---:|---:|---:|---:|---:|
| No XCD swizzle | 0.157081 | 0.160202 | 2624.87 | 2573.73 | 1.101x | 90.840% |
| XCD, `GROUP_SIZE_M=4` | 0.156282 | 0.160241 | 2638.29 | 2573.10 | 1.098x | 91.073% |
| XCD, `GROUP_SIZE_M=8` | 0.158321 | 0.161522 | 2604.31 | 2552.70 | 1.088x | 91.939% |
| XCD, `GROUP_SIZE_M=16` | 0.157562 | 0.162802 | 2616.85 | 2532.63 | 1.097x | 91.160% |
| XCD, `GROUP_SIZE_M=32` | 0.157802 | 0.162522 | 2612.87 | 2536.99 | 1.122x | 89.153% |

`GROUP_SIZE_M=4` has the best MoE sample and improves best throughput by
0.77% over no XCD swizzle. `GROUP_SIZE_M=8` has the best MoE median, but the
difference from group 4 is only 0.14%. Group 32 regresses best throughput by
2.31%. Use group 4 as the default for this shape.

## Optimization History

| Version | Best latency | Throughput | Observation |
|---|---:|---:|---|
| Serialized correctness baseline | 0.542 ms | 1522 TFLOPS | Full VMEM wait per K tile |
| Hoist sorted-ID loads | 0.541 ms | 1525 TFLOPS | Routing metadata was not the main cost |
| Ping-pong padding LDS | 0.461 ms | 1790 TFLOPS | About 15% lower latency |
| Issue expert B before A | 0.458 ms | about 1800 TFLOPS | Small improvement for cold weights |
| Final paired run | 0.456 ms | 1807 TFLOPS | 67.4% of equal-work GEMM |

The ping-pong version prefetches K tiles 0 and 1, computes two tiles per loop,
and loads tiles `k+2` and `k+3` into the released LDS buffers. This overlaps
future VMEM traffic with current MFMA work.

## Profile Findings

Dispatch resource profile:

| Kernel | VGPR | SGPR | LDS | Scratch |
|---|---:|---:|---:|---:|
| Serialized MoE | 192 | 112 | 50,688 B | 0 |
| Ping-pong MoE | 208-216 | 112 | 101,376 B | 0 |
| Equal-work GEMM | 236 | 112 | 101,376 B | 0 |

There is no scratch spill. PID/XCD swizzle was also excluded as the controlling
factor: GEMM measured about 0.321 ms with either setting.

SQ counters from one full dispatch:

| Counter | Optimized MoE | GEMM |
|---|---:|---:|
| `SQ_INSTS_MFMA` | 12,582,912 | 12,582,912 |
| `SQ_VALU_MFMA_BUSY_CYCLES` | 402,653,184 | 402,653,184 |
| `SQ_WAIT_ANY` | about 82.4M | about 20.7M |
| `SQ_WAVE_CYCLES` | about 247.9M | about 143.3M |

The useful MFMA work is identical. MoE spends substantially more wave cycles
waiting.

Memory counters:

| Counter | Optimized MoE | GEMM |
|---|---:|---:|
| L2 reads | about 25.58M | about 25.83M |
| L2 hit ratio | 62.1% | 85.5% |
| DRAM read requests | about 10.24M | about 3.30M |

The remaining gap is primarily the workload's weight reuse, not spill or grid
mapping. The standard GEMM repeatedly uses a roughly 25 MiB B matrix across 32
M tiles. The MoE case traverses about 768 MiB of expert weights with little
cross-workgroup reuse. A control with the same MoE FLOPs but one 3 MiB expert
reached 2126 TFLOPS, versus 1845 TFLOPS with 256 experts, confirming the reuse
sensitivity.

## Profiler Commands

```bash
rocprofv3 --kernel-trace --stats -d /tmp/moe_trace -o trace -- \
  python tests/flydsl/fp8_gemm_950/test_moe_mxfp8_mxfp4_gateup_4w.py --accuracy

rocprofv3 \
  --pmc SQ_INSTS_MFMA SQ_VALU_MFMA_BUSY_CYCLES SQ_WAIT_ANY SQ_WAVE_CYCLES \
  --kernel-include-regex 'moe_gateup_kernel' -f csv -d /tmp/moe_sq -o moe_sq -- \
  python tests/flydsl/fp8_gemm_950/test_moe_mxfp8_mxfp4_gateup_4w.py --accuracy

rocprofv3 \
  --pmc TCC_READ_sum TCC_HIT_sum TCC_MISS_sum TCC_EA0_RDREQ_DRAM_sum \
  --kernel-include-regex 'moe_gateup_kernel' -f csv -d /tmp/moe_mem -o moe_mem -- \
  python tests/flydsl/fp8_gemm_950/test_moe_mxfp8_mxfp4_gateup_4w.py --accuracy
```

`rocprof-compute` was not used because its installed frontend is missing Python
packages such as `astunparse`, `dash`, and `sqlalchemy`. Raw rocprofv3 counters
were sufficient for the comparison.

## Remaining Opportunities

- Add more K stages or finer `sched_group_barrier` placement to reduce
  `SQ_WAIT_ANY`; gains may be limited by cold expert B reads.
- Experiment with expert/N workgroup ordering to improve short-range B reuse
  when an expert has multiple sorted M blocks.
- Integrate sorted A-scale construction with the production sorting/quantization
  path instead of preparing it in this standalone test driver.
- Add SiLU and gate-by-up only after preserving this raw gate/up accuracy
  baseline.

## gaps with MOE in aiter.
For mxfp8 and mxfp4,  aiter fused MOE would sort the A scale as sorted_id table. [R, G], `R` means routed token ID. `G` means quantization groups. 
R would padded to 32 alignment, G would be padded to 8 aligned.
Also after A scale is sorted, [R, G] would be permuted    

[R, G] =  [R//32, 2r1, 16r0, G//8, 2g1, 4g0] would be permuted to 

[R//32, G//8, 4g0,16r0,2g1,2r1] 

current gemmA8w4 perfer the layout:


```
        scale_u8.view(rows // 128, 4, 32, groups)
        .permute(3, 0, 2, 1)
```

[R//128, 4r1, 32r0, G] -> [G, R//128, 32r0, 4r1]

把 这个是fused_dynamic_mxfp8_quant_moe_sort的 preshuffle之后的scale转化成test_moe_mxfp8_mxfp4_gateup_4w.py里面的A

1. 这个是fused_dynamic_mxfp8_quant_moe_sort scale的layout [R//32, G//8,4g0 ,16r0,2g1,2r1]  
  view as  [R//128, 4r2, G//8,4g0 ,16r0,2g1,2r1],


2. permute:
[R//128, 4r2, G//8,4g0 ,16r0,2g1,2r1] -[0, 1, 2, 3, 4, 5,6]    -> [G, R//128, 32r0, 4r1]

permute to 
[]
[G//8, 4g0, 2g1, R//128, 2r1, 16r0, 4r2]


permute:  [2， 5， 3， 0， 6， 4， 1]
reshape:



可以把moe_mxfp8_mxfp4_gatup_4w.md A scale sorted routing这部分用fused_moe.py来代替，.md文件中有关于如何通过permute以及reshape实现，