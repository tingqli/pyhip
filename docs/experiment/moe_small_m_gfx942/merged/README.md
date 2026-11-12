# Option A: one merged GateUp and one merged Down launch (`merged`), measured against option B (`split`)

Record: [docs/record/moe_small_m_gfx942.md §8](../../../record/moe_small_m_gfx942.md). Option B (the 4-wave small-M
kernels on every block, dense task list, in `src`) is described in [../split/](../split/README.md); this directory
holds option A and the session that measures both.

Option A now lives in `src/contrib/flydsl/moe_merged_a8w8_blockscale_gfx942.py` and is what
`fused_moe_a8w8_blockscale_gfx942_sort` / `_mixed` run (record §8.7, §10). The code below and the tools here are the
version measured in run 4 (commit 806d8fe, branch `luwei/942-fp8-blockwise-backup-b`).

## Code

- `variants/moe_small_m_8wave_pair.py`
  - Step 1: the 4-wave small-M kernels as 8-wave workgroups. A workgroup takes one sort block and two adjacent N tiles.
    Half h (waves 4h..4h+3) computes N tile 2p+h exactly like a workgroup of the 4-wave kernel. The A tile and its
    scales are loaded once for both halves, and all 8 waves share the barriers.
    `compile_moe_small_m(...).emit_pair` is the task body, the same code as the standalone pair kernel.
  - Step 2 (`merged_plugin`): the `merged=` argument of the BM256 compile functions in `src`. Those functions run the
    BM256 body unchanged; the 14 default BM256 configurations are ISA-identical with and without the hook.
  - Step 2 `task`, the dense task list with the BM256 tasks first:
    - Every workgroup scans the row-128 IDs of the valid blocks (3 per thread, bit-plane ballots and `mbcnt`).
    - Each XCD (`pid % 8`) gets a contiguous range of BM256 tasks (rank of the big block, N tile), then a range of
      pair tasks (rank of the small block, pair).
    - The owner of a task's block writes the block index to LDS. There are 3 barriers: the last one protects the scan
      words before a task overwrites that LDS.
    - Workgroups past the last task slot skip the scan. Blocks past `num_valid` issue no probe load (out-of-bounds
      offset).
  - Step 2 `body`: the pair task. It uses an LDS view in the BM256 kernel's 64 KiB and an opaque tid copy, so the two
    arms' indices do not merge.
  - Step 2 `grid`: `8 * (2 * num_blocks + 2)`, sized from the sort capacity.
  - Step 2 `key`: a hash of the plugin's sources (see "Kernel cache" below).
  - Entry points: `moe_gateup_merged`, `moe_down_merged`, and `fused_moe_merged` (sort, merged GateUp, merged Down).
- Resources (fresh compile, `FLYDSL_RUNTIME_ENABLE_CACHE=0`):
  - Merged GateUp and Down: 250 VGPR, 64 KiB LDS, no spill.
  - Standalone pair kernels: 246 / 244 VGPR.

## Tools (`tools/`)

- `check_pair.py`: the pair kernels against the current 4-wave kernels (bitwise).
- `check_merged.py`: checks the merged path for correctness.
  - Merged GateUp is bitwise equal to the two-launch GateUp.
  - Fused merged passes against the asm reference for T=64..32768 (calc_diff 1.4–2.6e-6).
- `ab_merged.py`: kernel A/B on synthetic sort blocks. Impls:
  - `bm256`
  - `merged`
  - `direct`: BM256 mapping, no scan.
  - `nobody`: direct with an empty pair arm.
  - `exact`: nobody with the grid cut to the valid tasks.
  - `small`
- `make_bench_merged.py` generates `bench_merged.py`. It is `tests/contrib/moe/bench_moe_small_m_gfx942.py` plus path
  `merged` and kernel variants `k: gateup_merged` / `k: down_merged`.
- `run_ab.sh`: idle-gated runs on GPU `$GPU` with paths asm, bm256, small, split and merged. Each run covers
  T=256..8192 sustained and burst, plus T=12288/16384/32768 (`final_large_*`).
- `summarize_ab.py` writes `summary.md`.

Reproduce:

```bash
python tools/make_bench_merged.py
GPU=1 tools/run_ab.sh
python tools/summarize_ab.py > summary.md
```

## Kernel cache

Importing aiter points the FlyDSL disk cache at `aiter/jit/flydsl_cache` with caching on. The JIT key covers only two
things:

- the function sources next to the root file (here the BM256 files in `src`);
- closure scalars (int, float, bool, str, None, tuple, enum).

The plugin object and its code in this directory are not part of the key. Until `MergedPlugin.key` existed, every
merged compile of a shape therefore reused the first one. That first compile was at 08:10, the full-scan version of
run 1, and it was reused across processes.

| Data | What was actually measured |
|---|---|
| `run1_full_scan/` | The full-scan version, as intended |
| `run2_cached/`, `run3_cached/` | The same run-1 kernel again (merged differs by ≤1–2 % between the three runs) |
| Lines of `ab_merged_full_blocks.log` before the "# after the plugin cache-key fix" marker | `direct` / `nobody` / `exact` were that kernel too |

The BM256 compile functions now put `merged.key` (a str) into the launch closure. The pair kernels, the small-M
kernels and the split kernels were not affected, because their sources are next to their own root files and their
options are scalars.

## Results (run 4, current code, top-level `*.json` / `*.log`, `summary.md`)

Setup:

- `GPU=1 tools/run_ab.sh`, 2026-11-10 10:15–10:25 UTC. Every run passed its amdsmi gates on the first attempt.
- Merged and split kernels were compiled fresh with the new plugin key.
- Paths include the sort.
- small = the current two-kernel path (small-M + BM256 kernels, each skipping the other's blocks).
- bm256 = the BM256 path: sort without zeroing, route counts, zeroing of the multi-route rows, and Down `single_store`.

Path speedup vs tuned asm (the fastest of 6 asm kernels). Burst for T ≤ 8192, sustained for the large T. The best
path per row is in bold.

| T | routing | small | split (B) | merged (A) | bm256 |
|---|---|---|---|---|---|
| 256 | ep_uniform | 1.292 | **1.310** | 1.280 | 0.538 |
| 512 | ep_uniform | 1.310 | **1.339** | 1.316 | 0.574 |
| 1024 | ep_uniform | 1.315 | **1.332** | 1.330 | 0.586 |
| 2048 | ep_uniform | 1.269 | 1.288 | **1.306** | 0.651 |
| 4096 | ep_uniform | 1.354 | **1.401** | 1.319 | 0.973 |
| 6144 | ep_uniform | 1.222 | 1.105 | 1.259 | **1.280** |
| 8192 | ep_uniform | 1.424 | 1.402 | 1.460 | **1.534** |
| 12288 | ep_uniform | 1.310 | 1.358 | 1.362 | **1.370** |
| 16384 | ep_uniform | 1.419 | 1.353 | 1.448 | **1.580** |
| 32768 | ep_uniform | 1.499 | 1.227 | 1.522 | **1.614** |
| 256 | random | 1.301 | **1.318** | 1.302 | 0.558 |
| 512 | random | 1.280 | **1.314** | 1.307 | 0.584 |
| 1024 | random | 1.263 | 1.314 | **1.322** | 0.589 |
| 2048 | random | 1.423 | **1.456** | 1.449 | 0.763 |
| 4096 | random | 0.923 | **1.129** | 1.061 | 1.031 |
| 6144 | random | **1.243** | 1.105 | 1.233 | 1.236 |
| 8192 | random | 1.054 | 1.195 | **1.313** | 1.157 |
| 12288 | random | 1.207 | 1.224 | **1.394** | 1.337 |
| 16384 | random | 1.367 | 1.305 | 1.506 | **1.524** |
| 32768 | random | 1.324 | 1.148 | **1.449** | 1.428 |

Kernel medians (µs, sustained runs). "small+big" is the two-kernel path; at T ≥ 6144 it is the BM256 kernels alone.

| T | routing | GateUp small+big / split / merged | Down small+big / split / merged |
|---|---|---|---|
| 1024 | ep_uniform | 148.9 / 144.7 / 143.6 | 84.6 / 79.4 / 76.9 |
| 4096 | ep_uniform | 225.9 / 219.8 / 232.0 | 130.8 / 122.2 / 123.2 |
| 6144 | ep_uniform | 325.1 / 392.5 / 317.1 | 187.1 / 211.7 / 184.2 |
| 8192 | ep_uniform | 356.3 / 371.5 / 348.3 | 211.3 / 204.2 / 207.5 |
| 32768 | ep_uniform | 1156.3 / 1480.5 / 1153.5 | 719.3 / 803.1 / 703.2 |
| 4096 | random | 371.9 / 282.7 / 312.1 | 192.2 / 152.5 / 157.3 |
| 8192 | random | 497.0 / 439.6 / 374.1 | 294.0 / 240.8 / 219.2 |
| 32768 | random | 1279.8 / 1558.8 / 1191.9 | 766.3 / 778.7 / 703.5 |

### Option B (split)

- Best or within 1 % of the best for every T ≤ 2048, at T=4096 ep_uniform, and at T=4096 random (1.129, against 1.061
  for merged and 1.031 for bm256).
- Worst at T ≥ 6144: 10–12 % below merged at 6144, 9 % at random 8192, and 21 % at random 32768. At ep_uniform 32768
  it is 24 % below bm256.
- Cause: a big block becomes two 128-row tasks.
  - Partial blocks (6144: 128 + 64 rows) have halves of unequal length that stop sharing the weights in L2.
  - Full blocks at large T run the 4-wave tile, which is slower than the BM256 tile when compute bound.

### Option A (merged)

- At most 8 % below the best path. The worst point is ep_uniform 16384 against bm256, a path-level difference (see the
  last bullet); every other point is within 6 %.
- Best at T=2048 ep_uniform, random 1024, random 8192 (1.313, +10 % over split, +13 % over bm256), random 12288 and
  random 32768.
- Behind split at T ≤ 4096 by 0–6 %. The pair task at S=8 loses the phase shift between two 4-wave workgroups
  (`ab_pair_b32.log`). Random 4096 has 12 blocks with 129–149 routes, which run as full 256-row BM256 tiles.
- The merged kernels are as fast as the BM256 kernels on full blocks (T=32768 ep_uniform: GateUp 1153.5 vs 1156.3 µs,
  Down 703.2 vs 719.3).
  - `ab_merged_full_blocks.log` (after the marker) has two kernel A/B runs: 128 full blocks, 928 spare capacity
    blocks. merged / bm256 is GateUp 1.005 / 0.970 and Down 0.978 / 1.037, i.e. within 4 % in both directions.
  - The monitor flagged most of those rounds as having foreign kernels.
- The remaining large-T gap to the bm256 path (5–8 % at ep_uniform 8192–32768) is in the path, not the kernels. The
  bm256 path's Down uses `single_store` and its sort does not zero `out`. The merged path zeroes all of `out` and adds
  every route with atomics.
