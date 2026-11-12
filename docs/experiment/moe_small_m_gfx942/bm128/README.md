# 128-row branch inside the BM256 kernels (`bm128`)

Record: [docs/record/moe_small_m_gfx942.md §9](../../../record/moe_small_m_gfx942.md). The aiter path
(`fused_moe_a8w8_blockscale_gfx942`: GateUp 2x4_pf1 FP8 + Down 4x2_pf1 `single_store`) keeps its grids, workgroup
shape and task order; a workgroup whose 256-row sort block has no route at row 128 (at most 128 routes) runs a 128-row
pipeline instead of the BM256 one. Same idea as luocheng25/aiter 3bc0b50 ("gemm1: per-block BM256/BM128 dual
pipeline in the GateUp kernel", gfx950), here for both gfx942 kernels.

## Code

- `src/contrib/flydsl/moe_gateup_a8w8_blockscale_gfx942_lds.py` / `moe_down_a8w8_blockscale_gfx942_lds.py`: `bm128`
  (GateUp 2x4_pf1, Down 4x2_pf1 with atomics). Each workgroup reads the sorted ID of row 128 of its block (one uniform
  load) and runs `big_body()` or `big_body(128)`.
  - GateUp (A streamed, B held): 2 stages per K block instead of 4 (A packets 0 and 1, M slices 0..3 of every wave).
    The four B packets of the next K block are loaded in stage 0 and written to LDS in stage 1. The FP8 epilogue
    writes rows 0..127; rows 128..255 of such a block are not written (Down never reads padding rows).
  - Down (A held, B streamed): A packets 0 and 1 only, 2 of the 4 MFMA pairs per stage (M slices 0 and 1), still
    4 stages per K block.
  - The two pipelines end with different empty inline asm. Without it LLVM's SimplifyCFG sank their common tail
    (the Down epilogue) into one block; the atomics' SGPR base (`"s"` operand) became a phi held in VGPRs and
    `v_readfirstlane` read a clobbered value: memory fault at 0x14000.
- `src/contrib/flydsl/moe_a8w8_blockscale_gfx942.py`: `fused_moe_a8w8_blockscale_gfx942(..., bm128=True)` by default.
- ISA (MiMo shape, `FLYDSL_RUNTIME_ENABLE_CACHE=0`):
  - `bm128=False` (all other configurations): `22_final_isa.s` byte-identical to the previous commit for GateUp
    2x4_pf1 FP8, Down 4x2_pf1 atomic sorted (with and without `single_store`), route counts and zero rows.
  - `bm128=True`: the BM256 K loop is opcode-identical to the kernel without it (GateUp 743, Down 708 instructions
    per 2 K blocks); the 128-row K loop has 399 / 406. VGPR 250 / 246 as before, no spills; SGPR 54 → 72 / 61 → 69.

## Tests

- `tests/contrib/moe/test_gateup_a8w8_blockscale_gfx942.py`: config `lds-2x4-bm128` (BF16 output, 5 shapes).
- `tests/contrib/moe/test_down_a8w8_blockscale_gfx942.py`:
  - configs `4x2-atomic-bm128`, `4x2-atomic-sorted-single-bm128` (6 shapes);
  - `test_gateup_down_bm128` (every block <= 128 routes, exactly 128, exactly 129, mixed with Down K/128 = 3, MiMo
    EP8 ranks of ~64 and ~128 routes per expert): the bm128 GateUp equals the BM256 GateUp bit for bit on every row it
    computes and leaves the other rows untouched; Down with and without bm128 (and `single_store`) on that output;
  - `test_fused_entry`: `fused_moe_a8w8_blockscale_gfx942` on aiter's EP sort (no zeroing), T = 64 / 1000 / 4096 /
    8192, bm128 off and on, against the asm test's reference.
- 199 passed, 1 skipped (`test_down_*`, `test_gateup_*`, `test_moe_small_m_gfx942.py`).
- After 1a4fc73 the BM256 kernels keep only the aiter configuration (record §11): the BF16-output GateUp test and the
  2x4 / kernel-reduce / unsorted-A2 Down configurations are gone; bm128 is tested by the `test_down` configs `bm128`,
  `single-bm128`, `test_gateup_down_bm128` and `test_fused_entry`.

## Tools (`tools/`)

- `run_bm128.sh`: protocol of `../merged/tools/run_ab.sh` (each run starts after GPU `$GPU` was idle and is redone
  until all its amdsmi gates pass) with `tests/contrib/moe/bench_moe_small_m_gfx942.py` itself: paths asm (6 kernels),
  bm256 (`bm128=False`), bm128, small, split, merged; T = 256..8192 sustained (`--kernels`) and burst, T = 12288 /
  16384 / 32768 sustained (`--kernels`). `sources.sha256` holds the measured sources.
- `summarize_bm128.py` writes [summary.md](summary.md).

```bash
GPU=1 tools/run_bm128.sh && python tools/summarize_bm128.py > summary.md
```

`run_bm128.sh` needs the tree of commit da764f5: the small / split paths and the small-M module were removed afterwards
(record §10).

## Results (2026-11-11, GPU 1, all gates PASS)

MI300X, MiMo-V2.6-Flash EP8 rank (I 2048, H 4096, 32 local experts, top-8), cold weights, medians with sort, speedup
over the tuned asm (the fastest of 6 asm kernels in the same session). `burst_ep_uniform` passed on attempt 3 (the
first two overlapped an external job).

| run | T | 256 | 512 | 1024 | 2048 | 4096 | 6144 | 8192 |
|---|---|---|---|---|---|---|---|---|
| ep_uniform sustained | bm256 | 0.619x | 0.631x | 0.685x | 0.710x | 1.041x | 1.355x | 1.641x |
| | bm128 | 0.841x | 0.859x | 0.878x | 0.911x | 1.277x | 1.353x | 1.655x |
| | merged | 1.250x | 1.139x | 1.251x | 1.213x | 1.360x | 1.290x | 1.545x |
| random sustained | bm256 | 0.619x | 0.648x | 0.647x | 0.792x | 1.081x | 1.321x | 1.231x |
| | bm128 | 0.836x | 0.858x | 0.877x | 1.007x | 1.044x | 1.331x | 1.234x |
| | merged | 1.275x | 1.278x | 1.180x | 1.340x | 1.030x | 1.310x | 1.393x |
| ep_uniform burst | bm256 | 0.554x | 0.573x | 0.599x | 0.658x | 0.966x | 1.265x | 1.532x |
| | bm128 | 0.842x | 0.857x | 0.888x | 0.924x | 1.263x | 1.258x | 1.563x |
| | merged | 1.288x | 1.325x | 1.348x | 1.289x | 1.412x | 1.201x | 1.428x |
| random burst | bm256 | 0.552x | 0.560x | 0.581x | 0.741x | 1.008x | 1.219x | 1.148x |
| | bm128 | 0.826x | 0.859x | 0.839x | 1.030x | 1.043x | 1.241x | 1.196x |
| | merged | 1.276x | 1.323x | 1.319x | 1.436x | 1.049x | 1.255x | 1.327x |

| run | T | 12288 | 16384 | 32768 |
|---|---|---|---|---|
| ep_uniform sustained | bm256 / bm128 / merged | 1.388x / 1.611x / 1.380x | 1.578x / 1.569x / 1.442x | 1.623x / 1.618x / 1.507x |
| random sustained | bm256 / bm128 / merged | 1.330x / 1.433x / 1.413x | 1.547x / 1.560x / 1.508x | 1.443x / 1.458x / 1.466x |

All paths (small, split as well) and the kernel tables are in [summary.md](summary.md).

- bm128 against bm256 (time): 0.65-0.79 for T <= 2048, 0.76-0.82 at ep_uniform T=4096 (every block exactly 128
  routes), 0.86 at ep_uniform T=12288 (each expert 256 + 128 routes), 0.93-1.005 elsewhere. The one exception is
  random T=4096 sustained, 1.036 at path level although both kernels are faster on their own (GateUp 289.5 vs
  298.4 us, Down 159.5 vs 163.2 us): the path samples depend on the variant timed before (forward rounds 506-526 us,
  reversed rounds 491-504 us), and an interleaved A/B of the two kernels inside the path on GPU 2 (`/tmp` script, no
  idle gate) gave GateUp 300.0 vs 305.0 us and Down 164.6 vs 173.3 us. In the burst run the same point is 0.967.
- bm128 against the best path of each point: 0.64-0.75 for T <= 2048 (small, split and merged are 1.15-1.44x the
  asm there), 0.88-0.95 at T=4096, within 1.2 % for T >= 6144 except random T=8192 (0.886 / 0.901, merged).
- Kernels at T <= 2048 (sustained): GateUp 190-213 us with bm128 against 261-289 us without, 127-167 us for the
  small-M / merged kernels; Down 109-123 us against 144-156 us, 66-96 us.
