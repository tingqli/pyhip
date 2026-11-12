# Option B: the small-M kernels on every block (`all_blocks`, `big_blocks="split"`)

Record: [docs/record/moe_small_m_gfx942.md §8](../../../record/moe_small_m_gfx942.md). The dense-task-list version
(v2) is measured together with option A in [../merged/](../merged/README.md); this directory holds the first
version (v1) and the A/B runs that led to v2. Option B and the small-M module were removed from `src` afterwards
(record §10); v2 is in commit da764f5.

- **v1** (source no longer in the tree, hashes in `sources.sha256`): `compile_moe_small_m(all_blocks=True)` with tasks
  (block, N tile, row half), the row half fastest; a half-1 task exits at once unless the block has a route at row 128.
  `run_split.sh` → `final_*`, `burst_*`, `final_large_*` (`.json`, `.log`, all amdsmi gates PASS, GPU 1, 2026-11-10):
  asm / bm256 / small (current two-kernel path) / split.
- **v1 diagnosis** (`ab_split_empty.log`, `tools/ab_split.py`): on 32 blocks of 128 or 64 routes (all small) the split
  kernels are 21–24 % slower than the small-M kernels doing exactly the same work, with or without 128 capacity
  blocks past `num_valid` (`--extra 0 / 128`). The workgroups of the empty half-1 tasks, interleaved with the real
  ones, hold CU slots for their probe loads; with 2 slots per CU, later real tasks wait for them. Workgroups past the
  last task (the sort's spare capacity) do not matter.
- **v2** (in `src`): a dense task list (block b owns `n_tiles * (1 + big_b)` tasks), found by each workgroup with a
  scan of the blocks' row-128 IDs (5 blocks per thread, bit-plane ballots + `mbcnt`, two barriers).
  `ab_split_dense.log` (same A/B, 128 capacity blocks): split / small GateUp 1.015× / 1.026× slower, Down 1.023× /
  1.053× (rows 128 / 64), against 1.31× / 1.27× for v1. Skipping the scan in workgroups past every task came later
  (see ../merged).

v1 results (µs with sort; tuned asm = fastest asm kernel of the run): `python tools/summarize_split.py`.

| T | routing | small | split v1 | bm256 | tuned asm |
|---|---|---|---|---|---|
| 1024 | ep_uniform, sustained | 252.4 | 245.2 | 435.8 | 292.7 |
| 4096 | ep_uniform, sustained | 370.7 | 446.1 | 479.8 | 494.8 |
| 4096 | random, sustained | 594.0 | 499.2 | 490.1 | 532.9 |
| 8192 | random, sustained | 846.1 | 742.9 | 723.6 | 882.6 |
| 6144 | ep_uniform, sustained | 551.9 | 626.1 | 521.9 | 699.1 |
| 32768 | ep_uniform, sustained | 1950.7 | 2394.1 | 1816.6 | 2952.5 |

Kernel times at T=8192 ep_uniform (every block full, both halves S=8): split GateUp 349.5 vs BM256 347.8 µs, Down 206.3
vs 203.1 (the second read of the weights comes from L2). At T=6144 (half 1 has 64 rows, S=4) split GateUp is 390.5 vs
326.9: the halves run at different speeds and no longer share the weights in L2.

Tools: `run_split.sh` (idle-gated runs), `summarize_split.py`, `ab_split.py` (kernel A/B, synthetic blocks).
