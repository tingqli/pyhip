# mega_large (20261118)

Full write-up: `../log_20261118c.md`.

- goal / hypothesis: one kernel per EP call for prefill-sized batches (aiter
  `mega_moe_large_m.py`, ported to gfx942): x quant, the count exchange, the plan, the
  dispatch, gemm1 / gemm2 and the combine in one launch, instead of `mega`'s 5 launches and
  two cross-rank waits.
- change: `kernels/ep_mega_large.py`, `megamoe.MegaMoeLargeEpA8W4`, `kernels/_protocol.py`
  (`large_*` layout), `moe_ep` solver `mega_large` (knobs comb x bcm), the EP table.
  Follow-up (20261119, log section 7): the x quant is the `moe_ep` problem dim `x_quant`
  (per_1x128 / per_token; it was the knob qx), supported by mega / mega_small / mega_large
  (kernel key QX: 1 per (row, 128), 0 per row, names `_xrow`).
  The experiment switches (TL timeline, LAG, LCOMB, COMB 1) are not kept in the kernel:
  `pre_cleanup.diff` turns the landed kernel back into the experiment kernel (see
  `timeline_large.py`'s docstring for the host side).
- commands (from src/contrib/flydsl, 8 GPUs):
  - correctness: `python a8w4/experiments/20261118_mega_large/check_large.py -t 2048 32768 --uneven [--act situv2] [-k comb=fp8 x_quant=per_token]`,
    `python -m pytest -q -s a8w4/tests/test_a8w4_megamoe_ep8.py` (`-k large`: mega_large only),
    `python -m kernelkit check a8w4.tune:EP_TUNING mimo-flash2.6-ep8 -t 2048 32768 -v @table ... --repeats 2 --graph` (`check_graph.md`)
  - precision of the x quant: `precision_qx.py -t 2048 8192 [--outliers] [--act situv2]`
  - A/B: the `kernelkit ab` command line is the second line of each ab*.md (ab1-ab7 use the
    experiment knob names: `comb=1` bf16 stores, `comb=2` = today's `comb=fp8`)
  - timeline: apply `pre_cleanup.diff` (regenerated after the x_quant change), run
    `timeline_large.py`
  - table: `python -m kernelkit tune-all a8w4.tune:EP_TUNING --gpus 0-7 --out /tmp/a8w4_ep_large`
    (logs `mimo-flash2.6-ep8-*_0.log`; `tune_v1_situv2_xrow_bad.log`: the first pass, before
    qx became a knob), `tune_compare.txt` (old vs new table), `tune_show.md` (best per solver)
  - table with x_quant: the same command (`--out /tmp/a8w4_ep_xq`), logs
    `tune_xq_<act>-<x_quant>_0.log`, silu per_1x128 128 / 512 / 1024 re-tuned and merged
    (`tune_xq_silu-per_1x128_retune.log`: some mori configs were 1.3-4.6x slow there in the
    first pass),
    `tune_xq_show.md`, `tune_xq_compare.md` (per_token vs per_1x128; vs the previous table),
    `tune_xq_verify.log` (`kernelkit verify ... -t 20 100 300 1000 1500 3000 6000 20000`)
- protocol: kernelkit cold timing (L2 flush + 8 weight copies), slowest rank, ABBA, `<base>#2`.
- results: ab1.md-ab8.md, tune_compare.txt, tune_show.md, check_graph.md; x_quant:
  tune_xq_show.md, tune_xq_compare.md, tune_xq_verify.log.
- conclusion: `mega_large` is the best EP config from 1024 (silu) / 2048 (situv2) tokens:
  silu 7.0-9.6% faster than `mega` at 2048-32768 (per-row x scales, as `a8w4_moe`); situv2
  0.4-3.7% (per-(row, 128) scales: per-row ones miss the op's 5e-3 tolerance, 7.3e-3).
  Mechanism: one launch, no host-side stages, per-row x quant without gemm1 rescaling; the
  fp8 combine halves the XGMI bytes of the combine at 4096 / 8192 / 32768 tokens. Limits:
  XGMI ~200 GB/s per GPU for the dispatch, gemm1 units W-decode bound (~106 us for 64 or
  128 rows), 100-170 us spread between ranks at 32768.
- conclusion (x_quant): per_1x128 and per_token share the crossovers (mega_small <= 1024
  tokens, mega_large 2048-16384; 32768: mega_large for per_1x128, mega for per_token, 3.0-3.6%
  ahead). per_token is 1.2-4.9% faster at <= 1024 tokens, 4.8-7.3% at >= 2048 (no gemm1
  rescaling); per_1x128 equals the previous table's per-128 configs within +-0.5%.
