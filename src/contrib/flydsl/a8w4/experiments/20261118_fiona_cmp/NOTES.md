# fiona_cmp (20261118)

Full write-up: `../log_20261118b.md`.

- goal: compare FionaZZ92/aiter 28e6b92 ("backport gfx942 MiMo A8W4 decode to FlyDSL 0.3.2":
  aiter's MegaMoEV2 small-M single kernel ported to gfx942, `SmallMRuntime`) with pyhip
  `mega_small` (`MegaMoeSmallEpA8W4`, the same aiter design on the a8w4 tiles) and `mega`:
  design differences, speed and accuracy on the same machine, inputs and weights.
- setup: FionaZZ92's aiter checked out at /tmp/fiona_aiter (not vendored, see the log §2.1),
  imported with `AITER_TRITON_ONLY=1`; its kernels unchanged; its sglang env (UNPACK reuse,
  OPERANDS native_ab_early, PROFILE register_pressure) unless `--fiona-env default`.
- commands (from src/contrib/flydsl, 8 GPUs, all GPU locks held):
  - sweep: `python a8w4/experiments/20261118_fiona_cmp/cmp_fiona.py -m 12 24 48 80 96 128 192 256 304 418 512 --out <dir>/sglang_env.json`
    (`sglang_env.log`), `... -m 12 48 96 192 512 --fiona-env default --no-mega --protocols cold graph8w`
    (`default_env.*`), `... -m 1 2 4 --no-mega --protocols cold graph8` (`tiny.*`)
  - tables of the log: `python a8w4/experiments/20261118_fiona_cmp/report.py`
  - resources / loop mix: `python -m a8w4.prof isa --grep megamoe -- python <dir>/cmp_fiona.py -m 12 96 128 --protocols`,
    then `loop_mix.py <dump>/<kernel>/21_final_isa.s` (`isa_resources_loop_mix.txt`)
  - counters: `python -m a8w4.prof counters --kernel megamoe --pmc "SQ_INSTS_VALU SQ_INSTS_MFMA SQ_INSTS_VMEM_RD SQ_INSTS_LDS" ... -- python <dir>/cmp_fiona.py -m 48 --protocols cold --iters 2 --no-mega`
    (the tool's parser crashed; summary parsed by hand: `counters_m48.txt`)
- protocol: kernelkit cold timing (L2 flush + 8 weight copies, host barrier, GPU spin, MORI
  device barrier, slowest rank, ABBA, `fiona#2`), plus CUDA graphs of 8 calls on 8 copies
  (`graph8`) or on one copy (`graph8w`, FionaZZ92's own protocol).
- result: FionaZZ92 is 2.15-2.86x slower than pyhip's fastest (cold; 2.3-3.3x graph) at
  1-512 tokens per rank, with the same accuracy; graph8w reproduces its published numbers
  (+0.4..-12.6%). Cause: its GEMM loops issue ~27 VALU per MFMA (emulated scaled MFMA) vs ~5 and
  run one wave per SIMD.
- conclusion: nothing to port from its GEMM / protocol; ideas: mega_small tmax 2048 (1536
  tokens -4.8% vs mega, one point), a runtime W-cache switch for mega_small past 1024 tokens.
