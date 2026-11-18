# mega_small (20261118)

Full write-up: `../log_20261118.md`.

- goal / hypothesis: one kernel per EP call for decode-sized batches (aiter `mega_moe_small_m.py`,
  ported to gfx942). It has no count exchange, plan kernel or finish kernel, so the 5 launches
  and two cross-rank waits of `mega` go away.
- change: `kernels/ep_mega_small.py`, `megamoe.MegaMoeSmallEpA8W4`, `moe_ep` solver
  `mega_small`. The timeline switch is not kept in the kernel: `timeline_instrumentation.diff`.
- commands (from src/contrib/flydsl, 8 GPUs):
  - correctness: `python a8w4/experiments/20261118_mega_small/check_small.py -t 64 512 [--uneven] [--act situv2] [-k bm=64]`,
    `python -m pytest -q a8w4/tests/test_a8w4_megamoe_ep8.py -k small`
  - A/B: the `kernelkit ab` command line is the second line of each ab*.md
  - timeline: apply `timeline_instrumentation.diff` (see `timeline.py`'s docstring), run `timeline.py`;
    output timeline_scm19.txt / timeline_scm0.txt
  - table: `python -m kernelkit tune-all a8w4.tune:EP_TUNING --gpus 0-7 --out /tmp/a8w4_ep`
    (logs `mimo-flash2.6-ep8-*_0.log`), re-tunes of the outlier points `re_ep.log` / `re_ep2.log`,
    then `python -m kernelkit merge`
- protocol: kernelkit cold timing (L2 flush + 8 weight copies), slowest rank, ABBA, `<base>#2`.
- results: ab1.md–ab5.md, tune_compare.txt (old vs new table), tune_show.md (best per solver).
- conclusion: `mega_small` is the best EP config at 64–1024 tokens: 15.5–19.1% faster than the
  old table's best at <= 512 tokens, 6.0–6.3% at 1024. `mega` stays best from 2048. Mechanism:
  one launch with no count exchange, plan or finish. Each gemm unit still loses ~5 us to the W
  prefetch fill (README 10.27).
