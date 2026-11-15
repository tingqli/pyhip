---
name: gpu-timing-protocol
description: How to time GPU kernels on MI300X-class AMD GPUs so that A/B results hold inside a model — a cold-cache protocol that also defeats the 256 MB MALL (rotate >= 8 weight copies), read-only L2 flush, host-overhead exclusion, interleaved ABBA rounds with a duplicated-baseline noise floor, power-capped clocks, routing-sample effects, multi-GPU slowest-rank timing, and when a <2% difference is real. Use whenever benchmarking or tuning kernels (aiter, pyhip, FlyDSL, sglang ops), writing a timing loop, or judging a small speedup / a cache-policy knob.
---

# Timing GPU kernels so the result holds in the model

Reference implementation: pyhip `src/contrib/flydsl/a8w4/bench/core.py` (`Timer`,
`interleaved`, `noise_band`, `check_protocols`) and `a8w4/ref.py` (`flush_cache`,
`weight_copies`, `time_cold`). Evidence: `a8w4/experiments/log_20261115.md` §3.3.

## 1. Cold caches: flush L2 AND do not let the MALL keep the weights

- MI300X: 4 MB L2 per XCD, then the 256 MB Infinity Cache (MALL). Reading 512 MB (even 2 GB)
  between calls evicts L2 but NOT lines every call re-reads: the same weights stay in the MALL.
- Effect: a config that loads weights through the cache (gfx942 cache modifier 0 / 1) looked
  5-13% faster than nt loads; in a model every layer reads other weights, so the gain is not
  there (it reversed to +2-10% at small batches). rocprof L2 hit rate and DRAM read requests
  were identical for both: the difference sat after L2.
- Protocol: flush by READING a large buffer (a read-modify-write flush leaves dirty lines whose
  write-back steals bandwidth and makes cached configs look 9-48% slower), and make every
  timed call read the next of >= 8 copies of the weights (2 copies were not enough).
- Rotation without a flush is not enough either: the result then depends on what the
  previous call left in the MALL.
- The same applies to any reused input: large activations that the model would produce fresh
  should not sit in the MALL from the previous iteration either.

## 2. Exclude host overhead and first-call effects

- Before the start event, put a ~1 ms GPU spin (`torch.cuda._sleep(2_000_000)`) after the
  flush: the host enqueues the whole call meanwhile, so launch overhead is excluded, as in a
  CUDA graph replay. Record start / end events around the call, synchronize every iteration.
- Warm up (compile, allocate, first-touch) before timing; the first call after a JIT is slow.
- Median of >= 10 calls per measurement.

## 3. Compare in the same process, interleaved

- Clock / power drift and placement effects are large: the same config differed ~15% between
  runs; the same data in a different buffer changed a GEMM by 2.5%. Never compare across runs
  or across processes for effects of a few %.
- Rounds: each round times every variant once (median of its calls); reverse the variant order
  every other round (ABBA); report the median of the round medians and the min-max.
- Time the baseline twice ("base#2"): its difference is the noise floor. Call a difference
  only beyond max(|base#2 delta|, half the base's round spread, ~0.3%).
- Large batches (>= 8192 rows): treat < 1% e2e as noise unless it repeats over rounds and
  samples; at the biggest sizes even 3% needed repeats.

## 4. The workload itself varies

- MoE routing samples change the per-rank / per-expert work: with ~1 block of rows per expert
  (EP8 at 4096 tokens), the block count moved +-9% between samples and a table point differed
  7% from a verify run. Compare configs on the same samples; use several samples when the
  difference is < 3%.
- Tiny EP batches: with uniform routing ~1/3 of the ranks get no local expert at 1 token; use a
  routing with a fixed number of local experts per token to measure the kernel.

## 5. Machine state

- Idle GPUs only: check `rocm-smi --showuse`; coordinate with `/tmp/gpu-<i>.lock` (flock),
  the convention of aiter's validate-kernel-pr.
- One process per GPU on all 8 GPUs at once did not bias single-GPU timings (ratio 1.003), so
  parallel tuning across GPUs is fine; sharing one GPU is not.
- 750 W cap: clocks ~1.05-1.4 GHz, reads 3.6-4.0 TB/s. Use bounds measured on the machine
  (a plain read, hipBLASLt fp8) instead of datasheet peaks.

## 6. Multi-GPU

- Align all ranks before the timed region: flush, host barrier, GPU spin, then a device-side
  barrier (MORI shmem barrier or a 1-element all-reduce); e2e = the slowest rank per
  iteration, median over iterations; per-stage times = the slowest rank's median.
- Interleave variants inside one job (one process group): separate jobs cannot be interleaved;
  if they must be, use ABBA job order and repeat.
- To measure exposed communication, swap the peer targets for local scratch (same work, no
  remote traffic) and compare in the same job.

## 7. Before believing a speedup

- Name the mechanism (fewer bytes, better overlap, fewer waits, higher occupancy) and the point
  where it should vanish; check that point.
- Cache-policy or rasterization knobs (anything that changes data reuse): measure one point
  under two protocols first (deployment-like vs single copy); if they disagree, only the
  deployment-like one counts.
- Ablations that skip code are not additive (dead-code elimination, changed write patterns):
  confirm with a parameter change.
