---
name: a8w4-perf-experiment
description: Workflow for performance experiments on the pyhip a8w4 MoE kernels (src/contrib/flydsl/a8w4; TP / EP / megamoe for MiMo-V2.6-Flash on MI300X) — finding what was already tried, timing A/B variants without the MALL bias, per-kernel breakdowns, 8-GPU megamoe sweeps, counters / ISA checks, retuning the table, landing a change and logging it. Use for any a8w4 optimization experiment, knob sweep, tuned-table work, experiments/log_*.md, or when a8w4/program.md drives the session (a8w4 优化实验 / 调参 / 调优表 / 实验日志).
---

# a8w4 performance experiments

All commands run from `src/contrib/flydsl` (the directory holding `a8w4/`). Every tool
prints `-h`. Before editing kernels also read the `flydsl-gfx942-pitfalls` skill; host
wrappers of new kernels follow `flydsl-host-launch`; for the timing rules behind the tools,
`gpu-timing-protocol`.

## The expensive mistakes of earlier sessions (and what now prevents them)

| mistake (log) | cost | guard |
|---|---|---|
| timed one weight copy: the 256 MB MALL kept W, cached-W configs looked 5-13% faster (1115 §3.3) | a full EP8 retune and CFGS change thrown away; a shipped table with wrong TP picks | the tools time one way only (flush + the next of 8 weight copies); `bench ab` refuses cached-W variants with fewer copies |
| believed a gain with no mechanism (t16 "faster" while each expert's W is read once) | as above | rule: no mechanism, suspect the measurement; `prof counters` to locate the level |
| started outside the focus range (split-K at <=16 tokens, 1114b §4) | a session's budget | step 1 below: baseline + breakdown on 1024-32768 first |
| swept a knob before reading the scheduler (lag only matters at 3-4 groups per queue, 1114b §3.4) | wasted 8-GPU jobs | step 2: derive where the knob can matter, then pick points |
| added up SKIP ablations (hist/scatter, 1114b §3.2) | wrong bottleneck | ablations are not additive (DCE, write patterns): cross-check by changing a parameter |
| cross-run comparisons; <1% e2e claims at large T | false conclusions | same-process interleaving, the noise floor `<base>#2`, routing samples |
| a non-scalar compile switch (callable / list) in a SKIP ablation: FlyDSL reused a stale binary | wrong ablation result | int / tuple switches; fresh `FLYDSL_RUNTIME_CACHE_DIR` after helper edits |
| script named `profile.py`; `--local` read as a rank | broken import; a redone run | `exp lint`; `--local-experts-per-token` |

## 0. Orient cheaply (do not read every log)

```bash
python a8w4/exp.py ls                    # experiments/index.csv: one row per experiment / idea
python a8w4/exp.py grep 'bcm|cache'      # regex over all columns; --area ep mega --verdict pending idea
python a8w4/exp.py grep lag --wide       # with notes
```
`python a8w4/exp.py` (script form) is instant; `python -m a8w4.exp` imports torch first.
Open a `log_*.md` section only for the rows you need. Never cite numbers of rows whose verdict
is `invalidated`. From the README read §4 (kernel choice), §6 (table / tuner), §10 (pitfalls),
§11 (next steps) — not the whole file. Do not read raw `*.txt` results in old experiment dirs.

## 1. Baseline and breakdown on the focus range

```bash
python -m a8w4.bench breakdown --case ep8 -t 1024 2048 4096 8192 16384 32768 --out a8w4/experiments/<dir>
python -m a8w4.bench breakdown --case tp8 -t 1024 4096 32768          # kernels per family, e2e, cos
python -m a8w4.bench breakdown --case ep8 -t 1 2 4 --local-experts-per-token 1   # tiny EP batches
```
Cases: `tp8` (I 256, 256 experts), `tp4` (I 512), `ep8` (I 2048, 32 of 256 experts); or
`--shape H,I,E_local,E_global,topk`. Decide the target from where the time goes.

## 2. Before a sweep: where can the knob matter?

Read the code path of the knob first and derive the token counts where it changes the work:
EP8 rows per expert = T*8/256 = T/32 (one 128-row block at 4096); megamoe lead = min(groups
per queue, lag); W re-read only with >= ~1.5 blocks per expert. Time only those points (+ one
control). Cheap hypotheses first: `python a8w4/exp.py grep <knob>` may already answer it.

## 3. A/B timing (one GPU)

```bash
python -m a8w4.bench ab --case ep8 -t 1024 2048 -v "ep3k m64" "pf2=m64:g2.pf=2" l128 \
    --out a8w4/experiments/<dir> --name pf
python -m a8w4.bench ab --case tp8 -t 16 64 -v "tp2k bm32 g1bcm2 g2nt4" "tp2k bm32 g1bcm1 g2nt4"
python -m a8w4.bench ab --case ep8 -t 4096 --samples 3 -v l128 l128c     # EP8: work depends on the routing draw
```
- Variant spec: `[label=]<config name | bare CFGS name | @table>[:override,...]`;
  ep3k overrides `g1.<k>=v`, `g2.<k>=v` (pf, bcm, acm, sch, st, prio, nw, ni/no, arb, tr, wpe),
  `bm=`, `rowv=32/64/128`, `sched=`, `sst=`, `-` drops a key; a variant the kernels cannot run
  raises at its first call (listed as FAIL with the exception). Custom
  pipelines: `label=py:<file.py | module>:<func>` with
  `func(x, W, tw, ti, *, expert_offset, num_experts, act, situ) -> out` — put an experiment's
  kernel variant behind such a function instead of writing a new timing script.
- Read the output: `us` = median of round medians; `delta %` vs the first variant;
  `<base>#2` = the noise floor; verdict `faster` / `slower` only beyond the noise band, `~`
  otherwise; `DIFF` = cos vs base > 1e-4 (a numerics change, not just order).
- Paste the generated `<name>.md` tables into the log; never hand-compute deltas.
- Defaults: 5 rounds x 16 calls. For effects < 2% at large T use `--rounds 7`; still treat
  < 1% at >= 8192 tokens as noise unless repeated with `--samples`.

## 4. A kernel variant

- Prefer a compile-time switch in the kernel (int / tuple key argument of the `@flydsl_kernel`
  builder, part of the kernel name) over a copied 1700-line file; keep only a diff in
  the experiment dir, record the commit hash in the log.
- Correctness before timing: `python -m a8w4.bench check --case ep8 -t 1 77 600 4096 -v l128
  "x=l128:g1.pf=1" --graph`; bit-exact checks need one route per token:
  `--shape 4096,2048,32,256,1 --bitwise`.
- Resources: `python -m a8w4.prof isa --grep ep_gemm -- python -m a8w4.bench check --case ep8
  -t 1024 -v m64 --repeats 0` (VGPRs > 128 / 168 cost waves per SIMD; LDS > 64 KB fails at launch).

## 5. Explain the result

- Counters per kernel: `python -m a8w4.prof counters --group l2 ea --kernel gemm -- python -m
  a8w4.bench ab --case ep8 -t 1024 -v m64 "c=m64:g2.bcm=0" --rounds 1 --iters 3 --no-dup --no-ref`.
  Same L2 hits / EA reads but different time: the difference is after L2 (MALL) or in clocks.
- A gain needs a mechanism (fewer bytes, more overlap, fewer waits, better occupancy). If the
  mechanism predicts where it should vanish, check that point too.

## 6. megamoe / 8 GPUs

```bash
python -m a8w4.bench dist -t 4096 8192 32768 -v default lag=4 "c=bcm=0/1" producers=8 --out <dir>
python -m a8w4.bench dist -t 4096 32768 -v default mode=ldisp mode=lcomb mode=local   # exposed comm
```
One job, variants interleaved (ABBA), 8 weight copies per rank, slowest-rank e2e + stages.
Needs all 8 GPUs idle (`rocm-smi --showuse`). Full-layer comparisons vs MORI: the test scripts
`python a8w4/tests/test_a8w4_megamoe_ep8.py`, `test_a8w4_mori_ep8.py`, `test_a8w4_mori_tp8.py`.

## 7. Landing checklist

1. Kernel, `ep_gemm.CFGS` / `AUTO`, `moe.EP_CFGS`, and the `moe_<solver>` function in `moe.py`
   (`@op_register` outermost, then the `@knob` values, `@no_capture` if not graph-safe,
   `@meta`; limits are asserts in the kernel with a message) as needed (README §9.3).
2. `python -m pytest -q a8w4/tests/test_a8w4_ops.py a8w4/tests/test_a8w4_moe.py` (the ops file
   runs the tuned configs of every problem it lists: tune a new config first); for megamoe / MORI /
   TP-layer changes also the three 8-GPU tests.
3. Selection changed: `python -m a8w4.tune tune-all --gpus 0-7 --out /tmp/a8w4_tune`
   (idle check, locks, BAD/FAIL report, merged table), `tune compare`, copy the table,
   `tune verify <problem or group> -t ...` (`python -m a8w4.tune problems` lists them).
4. README: `python -m a8w4.tune readme --write` (§7.2-7.4), then §4 / §5.6 / §6 / §10 / §11
   by hand; `configs/README.md` if the table changed; `tune readme --check` passes.

## 8. Archive

- `python a8w4/exp.py new <name>`: `experiments/<date>_<name>/NOTES.md`; tools write there via
  `--out a8w4/experiments/<date>_<name>`.
- One index row per result or idea: `python a8w4/exp.py add --dir <dir> --area ep --knob
  "m64 g2.pf=1" --tokens 1024-2048 --protocol coldrot8 --delta -1.7% --verdict landed --log
  "log_<date>.md 3.1" --note "..."`. Rows found biased later: add a row with `invalidated`.
- `experiments/log_<date>.md`: environment; survey table (source | idea | relevance); the list
  of tries vs the index; per experiment: change, command, the generated table, mechanism,
  verdict; reflection (what cost time, which guard would have caught it); next directions
  ranked for 1024-32768; landing table (scenario | proposal | status).
- `python a8w4/exp.py lint` (no ERROR; WARN on old scripts is expected) and
  `python a8w4/exp.py clean --inplace <file>` for noisy legacy outputs.
