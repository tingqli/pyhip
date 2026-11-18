# configs：a8w4 调优表

`a8w4_moe_tuned.csv` 是 `a8w4_moe` / `a8w4_moe_dyn` 选配置时查的表，由 `pyhip.kit.tune`（`python -m a8w4.tune`）生成。查表和计时的机制见上级 README 第 6 节；本文件记录表里有什么，以及生成它的命令。

## 格式

列：`model_dim, inter_dim, experts, experts_global, topk, routing, act, ~num_tokens, config, us, cos`

- **key**：`(model_dim, inter_dim, experts, experts_global, topk, routing, act)`，精确匹配。其中 `experts` 是本卡的专家数，`experts_global` 是全局专家数。
- **分桶列** `~num_tokens`（`~` 标出分桶维）：调用时取第一个 ≥ token 数的点。
- **每个 token 点保存全部候选**，不只保存最快的那个：
  - `us` 是各轮的中位数（冷 cache，每次调用换一份权重，不计 host 开销，见 `pyhip.kit.timing`）；
  - `cos` 是各轮与 fp32 参考之间最大的 cos 差；
  - cos ≥ 5e-3 或运行时抛异常（kernel 不支持）的候选不写入。
- **查表**：
  - 只用调用时那个 `act` 的行；没调过的 act 抛 `ValueError`（先 tune）；
  - 没有 `act` 列的表里 act 不是 key。
- **行的顺序**：按 (shape, routing, act, ~num_tokens, us) 排序，每次 `merge` 都会重新排（`pyhip.kit.table`）。

## 当前内容

| problem（`python -m a8w4.tune problems`） | shape H/I/E/E_global/topk | act | 点数 | 行数 |
|---|---|---|---:|---:|
| `tp8-silu` | 4096/256/256/256/8 | silu | 35 | 385 |
| `tp8-situv2` | 4096/256/256/256/8 | situv2 | 35 | 385 |
| `tp4-silu` | 4096/512/256/256/8 | silu | 35 | 280 |
| `tp4-situv2` | 4096/512/256/256/8 | situv2 | 35 | 280 |
| `ep8-silu` | 4096/2048/32/256/8 | silu | 35 | 510 |
| `ep8-situv2` | 4096/2048/32/256/8 | situv2 | 35 | 510 |

- 共 2350 行，routing 都是 uniform。
- token 点为 1, 2, 3, 4, 6, 8, 12, … , 24576, 32768（`tune.TOKENS`，`TUNING` 的 points）。
- `tp4-situv2` 是 20261118 补调的（`tune-all tp4-situv2 --gpus 0-7 --per-job 8`，无 BAD），此前 TP4 的 situv2 调用借用 silu 行。35 个点里 4 个点选中的配置与 silu 不同，相差 0.2–0.7%。
- 20261115 生成时 problem 叫 `mimo-v2.6-flash-<shape>`（旧 `tune.CASES`），EP8 的 expert_offset 是 0；现在 problem 来自 `ref.SHAPES`，EP8 用 rank 3（expert_offset 96）。expert_offset 不进表。

**来源**：20261115 用下面的命令整表重新生成（脚本、日志和各卡的 CSV 在 `experiments/20261115_retune2/`）。
- 原因是计时方式改了：每次调用换一份权重（`--copies`，默认 8 份）。刷 cache 挤不掉 MALL 里每次调用都读的权重，旧表在同一份权重上反复计时，W 走 cache 的配置显得快 5–13%（上级 README 10 节第 15 条）；旧表因此在 TP8 / TP4 的 6–128 tokens 选了 `g1bcm1`，新方式下比 `g1bcm2` 慢 5–7.7%。
- EP8 的候选多了 `ep3k l128c`（W 走 cache 的 l128），≥6144 tokens 选中它。
- 与旧表逐点的对比见 `experiments/20261115_retune2/compare_vs_head.txt`，总结见 `experiments/log_20261115.md`。

**situv2 行**：
- SiTUv2 参数取 `ref.SITU`（β 4、β_l 25、clamp 16）。
- β 和 clamp 是 launch 参数，不影响耗时，所以这组行对任意 SiTUv2 参数都适用。
- 与 silu 行的逐点对比见上级 README 6.3 节。

**环境**：8× MI300X（gfx942，304 CU，功耗上限 750 W），ROCm 7.2，torch 2.9.1+rocm7.2.0，FlyDSL 0.3.2，aiter tq/mimo-moe。

## 生成命令

在包含 `a8w4/` 的目录下运行（pyhip 里是 `src/contrib/flydsl`）。

- 每个 problem 用 4 张卡，每张卡跑 1/4 的 token 点；
- 8 张卡同时跑两个 problem，谁先跑完谁接下一个；
- 每个进程先把权重复制 8 份（EP8 / TP8 约 3.3 GB），计时时轮流使用；
- 机器要空闲：`tune-all` 开始前用 `rocm-smi --showuse` 检查（忙就报错，`--force` 跳过），并拿住 `/tmp/gpu-N.lock`（与 aiter validate-kernel-pr 同一约定）。

```bash
cd pyhip/src/contrib/flydsl
# 默认即上表的 6 个 problem（tune.JOBS）；token 点默认 tune.TOKENS
python -m a8w4.tune tune-all --gpus 0-7 --out /tmp/a8w4_tune
# 结束时打印被剔除的候选（BAD，正常为空；kernel 不支持的是 n/a，只在日志里计数），并合并成 /tmp/a8w4_tune/a8w4_moe_tuned.new.csv
python -m a8w4.tune compare a8w4/configs/a8w4_moe_tuned.csv /tmp/a8w4_tune/a8w4_moe_tuned.new.csv
cp /tmp/a8w4_tune/a8w4_moe_tuned.new.csv a8w4/configs/a8w4_moe_tuned.csv
python -m a8w4.tune readme --write                 # 上级 README 7.2–7.4
```

如果 FlyDSL 缓存（`~/.flydsl/cache`）里已经有这些 kernel，每一组 8 个进程约 1 分钟。缓存为空时要先编译，每个配置 5–13 s。

- **只更新一部分**（例如只重调 situv2）：`tune-all tp8-situv2 ep8-situv2`，再 `python -m a8w4.tune merge /tmp/a8w4_tune/*_[0-9].csv`。`merge` 只覆盖同一行身份的行（同一 shape、routing、act、token 点和 config），其余行保持不变。
- **整表重新生成**：用 `tune-all` 合并出的新文件整个替换（如上），不要 `merge` 进旧表，否则旧表里已经不再是候选的配置会留在表里。
- **测量误差**：8 个进程同时跑时，大 batch 的结果和事后在空闲卡上单独复测相比相差 ±5–8%，方向不固定（见上级 README 6.3 节）。20261115 的对照实验显示这主要是运行之间的差异，不是并发造成的（`experiments/20261115_retune2/concurrent/`：并发 / 单独的比值均值 1.003）。

## 检查

```bash
python -m a8w4.tune show   ep8-situv2   # 每点最快的配置、各 solver 最快的配置、capture 时的选择
python -m a8w4.tune verify ep8-situv2 -t 1 20 100 300 1000 3000 20000
python -m pytest a8w4/tests/test_a8w4_moe.py -q -p no:cacheprovider
```

`verify` 走的是 `ops.moe(...)` 的查表路径，token 数可以不在调优点上。每个 token 数打印：选中的配置、实测耗时、表中耗时和 cos。

## 新 shape、routing 或 act

- **新 shape**：
  - 从一个 problem 出发用 `--set` 改 dim：`tune tp8-silu --set model_dim=6144 experts=384 experts_global=384`，EP 再改 `expert_offset`；
  - 或者在 `ref.SHAPES`（`moe_test` 的 `@dim` shape）里加一项，它就成为 `TUNING` 的 problem（`<name>-silu` 等），也进了生成的测试；
  - 之后同样 `tune`（或 `tune-all <problem>`）再 `merge`。
- **`--set routing=hot`**：结果写成另一组行，调用时用 `a8w4_moe(..., routing="hot")` 选这组。
- **act**：可选值见 `ref.ACTS`（problem 名的后半段）。每个 shape 的每个 act 都要调：act 是精确匹配的 key。
