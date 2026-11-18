# a8w4 重构：敏捷执行计划

从 2026-11-16 起，本文件是这次重构的执行依据。

- **思维草图**：`refactory.md`（Q1–Q32）、`refactory2.md`（用户的新想法）和第 5–7 轮讨论（摘要见第 5 节）。它们只提供方向和事实，不是承诺。做到某一步时再从草图里取用，发现草图不对就改掉。
- **原则**：
  1. 越确定、越能立刻开始、越不会返工的事越先做。
  2. 每个迭代要小而完整：能验证（ISA 不变或 pytest 通过）、能提交、能 review。
  3. 做当前迭代时不为远处的设计预留结构。后面发现问题，就回头重构前面的工作。
  4. 远处的目标只当作朦胧的方向，不提前细化。

## 0. 续接：新 session 从这里开始

### 0.1 当前状态（每次停下来前更新）

- **迭代**：I1、I1b、I2、I3、I4 已完成。I3 的范围是用户看过 I2 后改的（见 3.4 节）：`_vendor` 只留真正用到的代码，迁到 `kernels/` 下的共享模块，删掉 `_vendor/`。用户看过 I4 后决定先跳过 I4b（NaN 传播，1.1 节），先做 I5（3.2 节）。I5 已完成。之后讨论出函数级 kit 的设计（`refactor6.md`），用户 2026-11-17 让开始实现：迭代 K（3.1 节）已完成。用户看过 K 后讨论了 `refactor7.md`：A 部分（op 注册表 `@op_register`）的迭代 R（3.0a 节）已完成。之后用户委托按推荐定下 B、C 部分的问题，并说明 sglang 现有代码不再考虑（冻结 API 取消）：迭代 D（3.0 节，查表分派进 kit）已完成。用户看过 D 后决定：只保证调优表里的 key，限制交给 kernel 的 assert，迭代 E（3.0e 节，去掉 `@requires`）已完成；用户让 E 之后直接做完 T1–T4（通用 tune / bench），每完成一个就提交并 push 到 tq2：T1–T4 已完成（3.0t 节）。之后用户重新梳理了术语和分工（2026-11-18）：problem 的维度叫 dim、solver 的旋钮叫 knob，测试和调优各自集中声明：迭代 V（3.0v 节）已完成。用户看过 V 后要求一个概念一个名字：迭代 W（3.0w 节，kind → solver、`<op>_test` → `<op>_problem` 等）已完成，等用户看过。I4b、I6 排在这些之后；另有一个待定问题：`ref_moe(raw)` 的调用方（`bench/core.py`、`bench/dist.py` 的 megamoe 参考、两个 MORI 测试）要不要改用 `moe_ref(W)`。
- **分支**：`/sgl-workspace/pyhip` 的 `refactor/a8w4`。
- **已有提交**：
  - 迭代 0，安全网：`64ce77c`、`d17f502`（ISA 基线在这个提交上生成）、`4e256d9`。
  - 设计文档：`1383e41`、`0c50a56`、`ab05750`、`6e2c2ac`（本文件）。
  - I1：`0b0e3c9`、`725209f`、`51817b1`、`f9d75f3`、`cb2e8f0`（收尾：文档和实验脚本）。
  - I1b：`aa91396`、`2b2efaa`、`4f478b4`、`8399c9d`（文档）、`bb7f40e`。
  - I2：`892d7f8`（`TL`、`DBG`）、`87fd7c8`（a16w4）、`7dbb85e`（实验脚本），`aac9bcf`（文档）。
  - I3：`18f243f`（原地精简）、`cc44955`（迁到 `kernels/`，删除 `_vendor/`）、`bf95fe8`（文档）。
  - I4：`47bde29`（计划）、`6c8434c`、`4b9eb2f`、`5e5d9e3`、`e565657`（合并 fp8x4 pack）、`f9d0ca1`（`tp_fmoe` 的 `NUM_XCD`）、`f95a9e6`（文档）。
  - I5：`97eb73e`（计划）、`d306c51`、`3067a2d`、`af9f077`、`89bf081`、`06d3875`（删 `kernels/_launch.py`）、`ff82c3d`（删无用 kernel 参数，有意改 ISA）、`0e7eaee`（文档）、`1d9f8e6`（skills）。
  - K：`77aa77d`（计划）、`3361672`（K0）、`a6862e8`（K1）、`6acef65`（K2）、`b92ecc4`（K3）、`2a8e774`（修 `test_cuda_graph`）、`9859360`（K4）、`2ee2a19`（K5）、`4cb1ded`（文档、skill）。
  - R：`716cc46`（计划）、`8101410`（R0）、`689e112`（R1）、`8735b35`（文档、skill）。
  - D：`056028b`（计划）、`5d0b861`（D0）、`4f51e94`（D1 的 kit 部分：host 开销）、`23a3e8f`（D1）、`e60470a`（D2 文档）。
  - E：`e522ebe`。
  - T1：`bebaff3`。
  - T2：`3637665`。
  - T3：`ef107ba`。
  - T4：`712fc05`。
  - V：`af50811`（V0）、`9adc2f0`（V1）、`893e3f2`（V2）、V3（文档，本文件所在的提交）。
- **用户的文件**：未跟踪的 `refactory2.md` 属于用户，不要改，也不要提交。`refactor3.md`–`refactor7.md`（设计稿）同样不提交。

### 0.2 工作方式

- 和用户用中文交流。需要用户决定时，用 `ask_user` 一次问一个问题；不再用 grill 的方式一问一串。
- **一个迭代的流程**：
  1. 在第 3 节写清本迭代的任务和验收标准；
  2. 实现，每个小步一个提交；
  3. 验证；
  4. 在第 4 节记一行，更新 0.1；
  5. 停下来等用户 review，用户同意后再开始下一个迭代。
- 提交信息里写 ISA 对比结论，结尾加 `Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>`。
- 环境、GPU 规矩、安全网命令、基线位置和重新生成方法，都沿用 `refactory.md` 的 §0.3 和 §0.4，内容仍然有效。

### 0.3 不变量（每个迭代都要守住）

- **纯结构性的改动**（删除走不到的分支、搬文件、改名、换 host 封装）：`prof isa-diff` 对基线必须输出 `ISA IDENTICAL`。kernel 名字可以变。
- **ISA 基线**：`/sgl-workspace/a8w4_refactor_state/isa_base` 从 I5 第 6 步起是 `ff82c3d` 的快照。更早的留在同目录：`isa_base_e565657`（I4 第 4 步到 I5 第 5 步）、`isa_base_d17f502`（I4 第 4 步之前）。基线丢了就按 `refactory.md` §0.4 重新生成，但 worktree 开在 `ff82c3d`；以后有意改变 ISA 的提交，也要这样换基线并在这里记下。
- **中途不重新调优**：哪个改动需要重新调优，就说明它不是纯结构性的（Q6）。
- **公开 API**：冻结（Q8、Q17）已由用户 2026-11-17 取消（sglang 将重新集成）。`tests/test_a8w4_api.py` 从 D 起守护新的公开接口（`ops`、`a8w4_moe`、`a8w4_moe_dyn`、`prepare_weights`、`ref`），不用 GPU，约 3 秒跑完。
- **用 GPU 前**：先看 `rocm-smi --showuse`，再拿 `flock /tmp/gpu-N.lock`。卡被占用时停下来问用户（Q29）。

## 1. Backlog（按确定程度分三档）

### 1.1 近期：确定、可以立刻做（按顺序）

| # | 内容 | 来源 | 验收 |
|---|---|---|---|
| I1 | 删除 TP kernel（`tp_gemm1`、`tp_gemm2`、`tp_fmoe`、`tp_fmoe_q`）里走不到的选项和冻结常量 | Q4，Phase 1 | ISA IDENTICAL；API 测试；单卡 pytest |
| I1b | 删除 TP kernel 里调用方不用的模式取值（fast_dec 0/1、gemm1 的 bf16 epilogue 等，规则见 3.6 节） | 用户在 I1 review 时决定 | 同上 |
| I2 | 删除 EP kernel 里走不到的选项（`TL` 等）和 a16w4 路径，包括 bench 的 `adt=bf16` 覆盖 | Q4，Q11 | 同上 |
| I3 | 精简 `_vendor`：没人读的环境变量、类、函数、参数直接删；剩下的迁到 `kernels/` 下按主题分的共享模块；删除 `_vendor/`（规则见 3.4 节） | 原 I3 + 原 I4 前半；用户 2026-11-16 | ISA IDENTICAL；API 测试；单卡 pytest |
| I4 | kernel 文件之间不再互相 import（共享的 device helper 搬进 I3 建的模块，按名字导入；现在的 `G1.x`、`EP.x` 这类模块属性调用不进 FlyDSL 缓存 key）；`_fp4x8_to_fp8x8`、`_fp8x4_pack` / `_pack_activation` 只留一份；硬件常量集中定义 | Q12，Q14，Phase 2 | ISA IDENTICAL（第 4 步除外：有意改动，输出逐位比较 + 性能 A/B） |
| I4b | 量化的 amax 改成传播 NaN（PyTorch 要求算子尽量传播 NaN）。I4 第 4 步实测：输入 x 的一个元素是 NaN 时，ep3k 的 3 个 tr 配置（`m64`、`l128`、`l128c`）、`ep1k`，以及对应的 dyn 配置输出有限的错误值；ref 和其他配置是整行 NaN。原因是 gemm1→gemm2 的重量化用 `maxnumf` 求 amax，忽略了 NaN：`ep_gemm` 的 tr epilogue（约 376–396 行）、`_kloop.quant_chunk`（ep1k 的 gemm2 输入）。同样写法的还有 megamoe 的 `ep_mega` epilogue 和 `ep_mega_comm.mega_quant_kernel`（量化原始 x），多卡，测试脚本还没覆盖。改成 `fx.max`，pack 函数不用动 | 用户 2026-11-16 | 每个配置注入 NaN / ±Inf 后输出和 ref 一样整行 NaN、其他行不受影响（加进 pytest）；ISA 只在这几处变；性能 A/B；单卡 pytest |
| I5 | host 封装改用 `@flydsl_kernel`，替代 `_run_compiled` 和 `functools.cache`；每个 kernel 文件一个公开入口；同时删掉 I1 留下的无用 kernel 参数（标了 `# unused` 的那些）。已完成，见 3.2 节 | Q13，Phase 3 | ISA IDENTICAL（kernarg 偏移变化要逐个解释）；pytest |
| K | 函数级 kit：op 写成一族函数（`moe_ref`、`moe_test`、`moe_tp1k` …），配置空间和限制写在函数的装饰器上，测试和 tune 由 `pyhip.kit` 生成或驱动。已完成，见 3.1 节 | `refactor6.md`（Q66–Q76）；用户 2026-11-17 | 见 3.1 节 |
| R | op 注册表：`@op_register` 逐个注册，取代 `collect(globals())`；每个包一个 `Registry`（`a8w4.ops.moe`）。已完成，见 3.0a 节 | `refactor7.md` A 部分；用户 2026-11-17 | 见 3.0a 节 |
| D | 查表分派进 kit（`Op.pick`、`Op.__call__`、`<op>_fallback`、`Registry(table=, bucket=, relax=)`）；取消冻结 API；moe_dyn 词汇表改名。见 3.0 节 | `refactor7.md` B 部分和“决定”；用户 2026-11-17 | 见 3.0 节 |
| E | 去掉 `@requires` 和兜底：限制只在 kernel 的 assert 里，只保证调优表里的 key；保留 `@no_capture`。已完成，见 3.0e 节 | 用户 2026-11-17 | 见 3.0e 节 |
| T1–T4 | 通用 tune / bench：`kit.table`（T1）、`kit.timing`（T2）、`kit.tune` + CLI（T3）、`kit.bench`（T4） | `refactor7.md` C 部分 | 每步输出与改前相同（tune 的候选、CSV、show / verify；bench 的 resolve），计时同卡 ABAB 在噪声内 |
| I6 | 新建 `workspace.py`，统一管理每张卡一份的缓冲区 | Q19 | ISA IDENTICAL；pytest |

### 1.2 中期：方向清楚，细节做到时再定

- kernel 内部按术语表改名，长函数拆成按阶段命名的函数（Q20、Q21）。可以在碰到某个文件时顺手做。
- `table.py`：把调优表 CSV 的读、写、合并收拢到一处，供 dispatch 和 tune 共用。配置字符串的 parse / format 加往返测试（Q10）。
- 族模块加注册表，目标是“加一个族只改一处”（Q1a、Q9、Q18）。迭代 K 已用函数族 + `collect` 实现：加一个族就是在 `moe.py` 写一个 `moe_<kind>` 函数，配置名、tune 候选、测试、bench 都从它导出（README 9.3 节）。
- 统一命令行入口，去掉 `sys.path` hack（Q7）；工具搬进 `dev/`（Q16、Q25）。
- 测试整理（Q22）、文档拆分（Q24）。
- megamoe 套用同样的分层（Q15）。

### 1.3 远期：朦胧目标，不提前计划

- 与具体 kernel 无关的公共工具层，比如 `pyhip.kit`，包含 isa、timing、tune、bench 等（refactory2 §1）。迭代 K 先建了 `pyhip.kit` 的 op 族和测试生成部分。
- 把“功能”和“实现”分开：共享测试、预先算好 dispatch 计划、离线 autotune（refactory2 §2）。
- 一条命令尝试一个优化思路（refactory2 §3）：
  - 用 tune 和基线比较，先过精度门；
  - 结果好就更新调优表里相关的行，并在 `perf_log.md` 里加一行。
- 降低 host 开销：I5 后每次调用约 40–67 µs（I5 前 45–79 µs，见 3.2 节）。
- 已知问题，只记录不处理：`ep3k l128_v1` 配合 fp8 每 128 列一个 scale 的输入时 LDS 超限，`why_not` 没有拒绝这种组合。

## 2. 删除规则（I1–I3 共用）

- 一个选项只有在 `a8w4_moe`、`a8w4_moe_dyn`、`MegaMoeEpA8W4`、`tune`、`bench` 都走不到时才能删。
- bench 的 variant 覆盖只作用于 ep3k：`g1.*` 和 `g2.*` 会作为 `ep_gemm1` / `ep_gemm2` 的参数传入，此外还有 `bm`、`rowv`、`sched`、`sst`（语法见 `bench/core.py` 的 `VARIANT_HELP`；K5 起由 `moe.ep_preset` 经 `moe_ep3k(**over)` 应用；`adt` 覆盖已在 I2 删掉）。TP kernel 的选项只能经 `moe.py`、`tune`、megamoe 走到。
- 删完后，剩下的 kernel 的 ISA 必须完全不变。不在快照矩阵里的配置，本来就走不到。

## 3. 当前迭代

### 3.0w W：一个概念一个名字（已完成）

用户 2026-11-18：统一成 solver，概念越少越好；检查有没有其他同一个东西两个名字的；`<op>_test` 改叫 `<op>_problem`（问题侧和解法侧对称：`<op>_problem` 的参数是 dim、`@dim` 给取值，`<op>_<solver>` 的仅限关键字参数是 knob、`@knob` 给取值）。

- [x] W：只改名。
  - solver：配置 dict 的键 `"kind"` → `"solver"`，`Op.impls` → `Op.solvers`，`<op>_<kind>` → `<op>_<solver>`；a8w4 里指 solver 的 “kernel family” / `FAMILIES` / `KINDS` → solver / `SOLVERS`（`breakdown.py` 的 kernel family 是 kernel 名前缀，另一回事，不改；README 的“TP 族 / EP 族”是一组 solver，不改）。配置名不变，调优表不动。
  - problem：`<op>_test` → `<op>_problem`（`Op.test` → `Op.problem`，a8w4 `moe_test` / `moe_dyn_test` → `moe_problem` / `moe_dyn_problem`）；`Op` 各方法的 dict 参数 `dims` → `problem`（它就是 problem），`kit.tune` / `kit.bench` 的 `c` → `problem`；a8w4 `shape_dims` → `shape_problem`，bench 的 `Case.dims()` → `Case.problem()`。`<op>_dims` 不改：它返回一次调用看得出来的那些 dim 值。
  - 其他同义词：README 的 data → inputs（代码一直叫 inputs）；candidates → configs（CLI 命令 `candidates` → `configs`，和 `op.configs` 同名；日志 “N configs”、“BAD configs”）；“网格点 / grid”指配置时改叫配置；`Op.meta(cfg, key, dims)` → `meta(cfg, name, problem)`（key 只指精确匹配的 dim）；内部 `extras` → `over`。
  - 对照 HEAD：360 个测试 ID 相同；600 次查表（三个 shape × act × routing × tokens × capture）选中的配置名相同；`configs` / `show` / `problems` 输出与旧 `candidates` / `show` / `problems` 逐字相同；`readme --check` 通过；ISA IDENTICAL；kit 18、单卡 413、8 卡 3 passed。API 测试 `test_pick_kind` 改名 `test_pick_solver`。

待用户定的同类问题（没改）：

- a8w4 bench 的 `Case`（`--case tp8`、`case_from_args`）和 ISA 快照任务名 `moe:<case>:...`、`test_a8w4_moe.py` 的 `case` 参数：都是 shape 预设的名字（加上 act / routing / expert_offset 就是不带 bucket 的 problem）。
- `spec` 两个意思：命令行的 `<模块>:<Tuning 名>` 和 bench 的 variant spec。

### 3.0v V：术语 dim / knob，测试和调优集中声明（已完成）

用户 2026-11-18：调优表本质上是 problem 到 solver 配置的映射；problem 的维度叫 dim（原 vocabulary）、solver 的旋钮叫 knob（原 `@param`），同一个东西只有一个名字；bucket 是允许非精确匹配的那个 dim，其余 dim 精确匹配；问题相关的放在 `<op>_test`，solver 相关的放在 `<op>_<kind>`，测试和调优各自集中、显式传入。按推荐定下：表头用 `~num_tokens` 标出 bucket；去掉 relax、补调 tp4-situv2；`seed` 是 `<op>_test` 的保留参数、不是 dim；多个 dim 一起的 `@dim` 要有轴名；命令行的第一个参数是 `<模块>:<Tuning 名>`。

- [x] V0 `af50811`：改名。`param` 拆成 `dim`（只用在 `<op>_test` 上）和 `knob`（只用在 solver 上），用错位置注册时报错；`Op.vocab` → `Op.dim_names`，各方法的 `vocab` 参数 → `dims`；`seed` 必须是带默认值的仅限关键字参数；“implementation” → “solver”；a8w4 `shape_vocab` → `shape_dims`。语义不变。对照 HEAD（用 HEAD 快照遮住可编辑安装的 pyhip）：tune cases / candidates / show 相同，360 个测试 ID 相同；ISA IDENTICAL；kit 18、单卡 413 passed。
- [x] V1 `9adc2f0`：problem 显式化。`<op>_test` 上的 `@dim` 只给 dim 的取值命名（shape 预设、act、routing），不再决定测什么、调什么；`op.problems(*axes)` 组合 problem；`op_tests(op, problems)` 必须传入要测的 problem；`kit.tune.Tuning(op, problems=, bucket=, points=, keys=, rotate=, table=)` 放下全部调优设置，`Registry` 去掉 points / unkeyed / rotate；命令行 `python -m pyhip.kit <cmd> <模块>:<Tuning 名>`，`cases` 改名 `problems`；bench 也吃 Tuning。a8w4：`tune.py` 有 `TOKENS` 和 `TUNING`，测试文件写明 problem（`test_moe` / `test_moe_dyn`）。对照 HEAD：problem = 旧 case，candidates / show / FIELDS / 360 个测试 ID 相同；EP8 16 / 600 tokens 新旧 tune 两张卡并排：表头、行、配置相同，cos 在 atomics 噪声内，时间比 1.008；ISA IDENTICAL；413 passed。
- [x] V2 `893e3f2`：表头说明 bucket（`~<dim>`，`kit.table.bucket`），分派从表头读 key 和 bucket，`Registry(name, table=)`；去掉 relax。a8w4 表头改为 `~num_tokens`（行不变），补调 tp4-situv2 280 行（tune-all 8 卡，无 BAD；35 点里 4 点的 winner 与 silu 不同，差 0.2–0.7%）。分派对照 HEAD：600 次查表里只有 10 次不同，全是 tp4-situv2 用上自己的行；测试 ID 不变；ISA IDENTICAL；单卡 413、8 卡 3 passed。
- [x] V3：文档。kit README 按新术语重写（术语表、四处分工、axpy 例子分 op / 调优脚本 / 测试三个文件，输出都是实跑的；用户 `3b30b0a` 加的“调优时的值从哪来”改写成 8.2 节）；a8w4 README、configs/README、skill a8w4-perf-experiment；`tune.JOBS` 加上 tp4-situv2。

### 3.0t T1–T4：通用 tune / bench（已完成，设计见 `refactor7.md` C 部分）

用户 2026-11-17：E 之后直接做完 T1–T4，每完成一个就提交并 push 到 tq2。

- [x] T1 `bebaff3`：`pyhip.kit.table`（标准库）——`read`、`identity`（`config` 及之前的列：同 identity 的行互相替换）、`write`（按 identity 去掉 config 后排序，数字按数值，再按 us；原子替换）、`merge`、`compare`。分派的读表、a8w4 tune 的 merge、`tables.compare` 都改用它。对照被替换的代码（原样拷贝）：把表打乱拆成 4 份再合并、把 300 行改时间后合并进表，得到的文件逐字节相同；compare 输出的 177 行相同；分派结果不变；`readme --check` 通过；kit 16、API 22 passed。去掉了旧部分文件“没有 act 列就当 silu”的补列（没有这样的文件了）。
- [x] T2 `3637665`：`pyhip.kit.timing`（torch）——`flush_cache`、`copies`（原 `ref.weight_copies`，也接受单个 tensor）、`time_cold`、协议 `Timer`、ABBA 的 `interleaved`、`noise_band`、`verdict` 和常量。从 `ref.py`、`bench/core.py` 原样搬过来（AST 只差 docstring 和 `R.flush_cache` → `flush_cache`），旧名字在原处重新导出（实验脚本不用改）。冒烟：bench ab EP8 600 tokens（ep1k 被 kernel 的 assert 丢弃）、EP8 一个点的 tune；kit 16、API 22 passed。
- [x] T3 `ef107ba`：`pyhip.kit.tune` + CLI `python -m pyhip.kit <cmd> <module>:<op>`（cases / candidates / tune / merge / show / verify / tune-all / compare）。声明在 `Registry(points=, unkeyed=, rotate=)`：调优点、不进表的词汇、计时轮换的输入。调优 case = `<op>_test` 的网格去掉分桶轴（`tp8-silu` 等），`--set` 改词汇；表头由词汇推出（为此 `moe_test` 把 routing 放到 act 前面）。a8w4 的 `tune.py` 变成薄包装（加 compare / readme / stats 报告），`tune.CASES` 并进 `ref.SHAPES`（EP8 的 case 用 rank 3，expert_offset 96；bench 仍认旧名字；ISA 快照仍用 0）。对照改前的包：candidates 三个 shape 相同；表头 = 旧 `FIELDS`；TP8 三个点新旧 tune 同卡先后跑，33 行相同、cos 相同、us 中位比 +0.00%（-1.3..+2.7%）；show 的 point / best / us / capture 选择在三个 shape 上相同（格式改为每个实现一列）；verify 选择和表中时间相同（EP8 加 `--set expert_offset=0` 复现旧结果）；tune-all 冒烟。ISA IDENTICAL；kit 17、API 22、单卡 413 passed。
- [x] T4 `712fc05`：`pyhip.kit.bench`（torch）——变体写法（`[label=]base[:k=v,...]`、`py:`、`@table`）、`resolve`（覆盖项成为配置项，`@meta` 读 tile_m / cached_loads）、cached-load 协议检查、运行记录（`Recorder`、`run_meta`、`md_table`）和两个引擎 `ab`、`check`；指标取 `<op>_check` 的第一项；CLI `python -m pyhip.kit ab|check <module>:<op> <case>`。a8w4 的 `bench/core.py` 只留用例、输入（`--local-experts-per-token`）、本地路由 / m-block 注释、裸 ep3k 预设名（`alias`）和 breakdown 用的 `resolve` 适配（`run(x, W, tw, ti)`）；`ab.py`、`check.py` 调 kit 引擎，命令行不变；记录字段 `wcopies` 改名 `copies`。对照改前的包（同一 EP8 shape、600 tokens、s32 / m64 / m64:g2.pf=1、`--bitwise`）：配置、本地路由、block 数、cos、verdict 相同，两张卡上时间差 1–2%；`check` tp8 1 / 77 / 600 加 `--graph` 结果相同；breakdown、py variant、2 个 sample × 2 个协议、`--out`、通用 CLI 冒烟；kit 18、单卡 413、8 卡 3 passed。只改 host 端。

### 3.0e E：去掉 `@requires`（已完成）

**背景**：用户问为什么要在实现函数上重复 kernel 里已有的检查，并定下假定：只有命中调优表的配置保证调用成功；没调过的 key、或者强制一个 kernel 跑不了的配置，直接抛异常；tune 用异常表示“跑不了”。实测（session 的 `files/req/probe.py`，8 卡，崩溃 / 卡死单独记录）：`@requires` 拒绝的 164 种组合全部由 kernel 代码抛 `AssertionError`，没有 fault、hang 或错误结果；另有 7 种其实能跑（TP 在 H = 3840，`model_dim % 512` 写严了）。另一个方向的漂移：l128_v1 配 fp8 分组 scale 的 LDS 限制 `@requires` 没写。

**改动**（一个提交 `e522ebe`）：

- kit：删 `requires`、`why_not`、`candidates`、`<op>_fallback` 角色；加 `@no_capture(reason)`、`Op.capturable`、`Op.tuned`（调优点的全部配置）；`pick` = `tuned` 里最快的（capture 时跳过 `@no_capture`），没有行抛 `ValueError`；`configs` 是完整网格。`kit.testing`：有表时每个用例测 `tuned` 的配置，没表时测完整网格。
- a8w4：`moe.py` 删全部限制和兜底，tp1kq 加 `@no_capture`；变成用户可见报错的几处 kernel assert 补上消息；tune 候选 = 完整网格，抛异常的记 n/a；verify 去掉“未调优默认值”；bench 去掉 `@auto`；isa_snapshot 编译每个 token 数所在调优点的配置（加网格外的 ep3k 预设）。

**验证**：

- 生成测试的 ID = 原来没有 skip 的那 360 项，逐项相同；
- lookup / dyn 对 D 的 golden：调过的 shape、uniform 路由、≤ 32768 tokens 全部相同；没调过的 shape 和 routing "hot" 改为抛 `ValueError`；EP8 ≥ 65536 tokens 选到 ep3k l128c（运行时在 ep_prep 抛异常）；
- TP4 一个点的 tune 记下的 8 个配置与表里相同（12 个 n/a）；
- ISA IDENTICAL；kit 15、API 22 passed；单卡 413 passed（不再有 skip）；8 卡 3 passed。

### 3.0 D：查表分派进 kit，取消冻结 API（已完成，设计见 `refactor7.md` B 部分和“决定”）

**目标**：

- kit 做分派：`Registry(name, table=, bucket=, relax=)` 声明调优表；`op.pick(vocab, table=, capturing=)` 选配置；`op(*args, config=, table=, **hints)` 读词汇表、选配置、运行；新角色 `<op>_fallback`。
- a8w4 只留声明和策略：`act → silu`、两个兜底函数、`moe_dyn` 的词汇表改名和容量限制。
- 取消冻结 API：元组形式的 7 个函数删掉，工具和测试改用 op 的方法；`a8w4_moe` / `a8w4_moe_dyn` 只剩入参检查、`M == 0` 和 `MOE(...)`。

**步骤**：

| 步 | 内容 | 验收 |
|---|---|---|
| D0 | kit：读表（按路径 + mtime 缓存）、`pick`、`points`、`__call__`、`<op>_fallback`、`Registry` 的声明 | `tests/kit` 用玩具 op 和 tmp CSV：精确 key、第一个 ≥ 的点和超过最大点、relax、旧表缺列、None 的 key 取默认值、不认识的 kind 跳过、capturing 过滤、兜底、都不能跑时的 `ValueError`、mtime 重读、hints、`config=`；不碰 a8w4 |
| D1 | a8w4：声明、兜底、moe_dyn 改名、薄包装；删元组形式的函数和分派段；tune / tables / bench / isa_snapshot / 测试改用 op 的方法；`test_a8w4_api.py` 改写；用到旧接口的实验脚本加说明 | K2 golden 的同样内容改用新接口计算，和改前逐项相同，只有有意差异（B1、≥ 65536 token 的 dyn、未知 ep3k 预设的解析），逐类列出；op 级 golden：`moe` 相同，`moe_dyn` 只有改名；生成测试的 ID 不变；ISA IDENTICAL；API + kit 测试；单卡 pytest；8 卡 MORI / megamoe；`a8w4_moe` 的 host 时间同卡 ABAB 不变差 |
| D2 | 文档：README（2 节接口、6.2 查表规则、8 节、9.3 节）、skill | — |

改前的基线和脚本在 session 的 `files/d/`（`golden_d.py` 用新接口算 K2 golden 并按类列出差异；`host4.py` 是 host 时间的 ABBA），改前的包在 `/tmp/d1/pre/a8w4_pre`。

**实现时定下的细节**（不对再改）：

- 表的 key 列 = 表头 ∩ 词汇（去掉 bucket 列），缺的列就不是 key；词汇里为 None 的 key 按 `<op>_test` 的默认值匹配。
- 薄包装只留 M == 0：kernel 在 M == 0 时会失败（tp1kq、ep3k 报 hipErrorInvalidValue，ep3k 还有访存错误，已实测）。入参检查搬进 `moe_dims` / `moe_dyn_dims`；kit 的 `op(...)` 每次调用（包括 `config=`）都调 dims。
- bench 的 `parse_variant` 仍然拒绝不认识的 ep3k 预设：`MOE.parse("ep3k nope")` 能解析，由 `why_not` 报。
- host 时间：D1 第一版的查表路径比改前慢约 3.6 µs（实现换成空函数时），真实 launch 时慢约 7 µs。来源是 `dims`、词汇合并、`pick` 和 `run` 多出的一层调用，以及每次的 `os.stat`（约 1.2 µs，改前也有）。真实 launch 时 Python 的开销大约翻倍。kit 因此改了三处（`4f51e94`）：
  - `op(...)` 按（表、mtime、capturing、dims 的值、hints）记住（实现, knobs）；
  - 每张表的 mtime 最多一秒读一次，表更新后一秒内生效；
  - 入参检查只在 dims 里做一次。

  改完后同卡 ABBA（改前的包计两次，差值作噪声底）：查表路径 −0.8 到 +0.8 µs（共 63–72 µs）；指定配置名时 +1 µs，因为 dims 现在也会做检查，这条只是开发工具走的路径。
- `test_a8w4_moe.py` 的 `test_dyn_config` 改名为 `test_dyn_pick`。`test_a8w4_api.py` 改写后是 22 项。

**结果**：

- [x] D0 `5d0b861`：kit 读表、`pick`、`points`、`__call__`、`<op>_fallback`、`Registry(name, table=, bucket=, relax=)`。`tests/kit` 15 passed；a8w4 没有改动。
- [x] D1 `4f51e94`（kit 部分）、`23a3e8f`：
  - a8w4 改动：
    - `_ops.py` 声明调优表；
    - 加了 `moe_fallback` 和 `moe_dyn_fallback`；
    - `moe_dyn_ep3k` 要求 capacity < 65536；
    - `moe_dyn_test` 的词汇改为 `capacity` / `num_tokens`，并加了 `routing`；
    - 删掉 7 个元组形式的函数、`FIELDS` 和表代码；
    - tune、tables、bench、isa_snapshot、五个测试文件都改用 op 的方法；
    - 用到旧接口的 5 个实验脚本在文件头注明“在 `5d0b861` 或更早的提交上跑”。
  - 验证：
    - 用新接口算 K2 golden，和改前相比：`supported` 20088 点、候选、名字、解析全部相同。只有三类有意差异：
      - 没有能跑的配置的 shape (1000,256,256,256,8)、(4096,384,256,256,8)：lookup 和 default 改为抛 `ValueError`（各 384 / 96 项），原来返回一个跑不了的 tp2k；
      - dyn 在 ≥ 65536 token 时改为抛 `ValueError`（20 项），原来返回超过 ep_prep 上限的 l128c；
      - `"ep3k nope"` 现在能解析。
    - op 级 golden：`moe` 相同；`moe_dyn` 只差改名（加 `routing`），20 个用例改名后逐项相同。
    - `test_a8w4_ops` 的 ID 不变（506 项）。
    - ISA IDENTICAL（558 次编译、177 个 ISA、4 个已有错误）。
    - API 22、kit 15 passed；单卡 moe + ops + api 413 passed、146 skipped；8 卡 MORI EP8 / TP8 + megamoe 3 passed。
    - host 时间见上。
- [x] D2 `e60470a`：README 的 2.1 / 2.2 / 2.4、3、6.2、8、9.1 / 9.3 节。skill 里没有提到被删的接口，不用改。
- **验收**：都已满足。host 时间只有一点偏差：查表路径在噪声内，指定配置名时 +1 µs（见上）。

### 3.0a R：op 注册表（已完成，设计见 `refactor7.md` A 部分）

**目标**：用装饰器 `@op_register` 逐个注册 op 的函数，取代一次性的 `collect(globals())`。

- 每个包一个注册表（`pyhip.kit.Registry`）。a8w4 的在 `_ops.py`：`ops.moe`、`ops.moe_dyn`。
- ref、test、check、dims 和实现函数全部注册；`@op_register` 写在最外层。
- 角色仍由名字推断，校验内容和 `collect` 相同，只是改成在每个函数注册时做。
- 不用全局的 `pyhip.kit.ops`：同一个包会被导入两次（`a8w4` 和 `pyhip.contrib.flydsl.a8w4`，已实测），`moe` 这个名字在 pyhip 里也太泛。

**步骤**（每步一个提交）：

| 步 | 内容 | 验收 |
|---|---|---|
| R0 | kit：`Registry`、`Registry.register`；`Op` 改成逐个追加；标签检查“已注册”；删 `collect`；`tests/kit` 改写，并加注册规则的测试 | `pytest tests/kit` 通过；不碰 a8w4 |
| R1 | a8w4：`_ops.py`；`ref.py` 8 个、`moe.py` 6 个函数加 `@op_register`；`moe.py` 去掉 `collect` 和 F401 导入；`__init__` 导出 `ops`；`test_a8w4_ops.py` 用 `op_tests(ops)`；README | 改前存的 golden（K2 的冻结函数 golden，加 op 级 golden：词汇表、实现顺序、用例、每个用例的配置和原因、meta）逐项相同；生成测试的 ID 不变（`test_a8w4_ops` 加 API 共 529 项）；API 测试；kit 测试；ISA IDENTICAL；单卡 pytest |

改前的基线和脚本在 session 的 `files/r/`（`golden_ops.py`；冻结函数用 `files/k2/golden.py`）。

**实现时定下的细节**（不对再改）：

- R0 留了一个过渡用的 `collect`（建在 `Registry` 上：先 ref、再 test、再其余），让 R0 的提交里 a8w4 仍能用；R1 删掉。
- `@op_register` 只注册普通函数（`inspect.isfunction`）。注册后函数上有 `kit_op`（op 名）；标签遇到它就报“不是最外层”。
- 实现名不能以 `_test` / `_check` / `_dims` 结尾：这样 `moe_dyn_test` 早于 `moe_dyn_ref` 时报“先注册 `moe_dyn_ref`”，而不是签名不符。
- 生成测试的顺序变了：`moe` 的在前（ref.py 里 `moe_ref` 先注册），原来 `collect` 按 `moe.py` 的 import 列表把 `moe_dyn` 放在前面。ID 集合不变。

**结果**：

- [x] R0 `8101410`：`Registry`、`Registry.register`；`Op` 拆成 `_add_test` / `_add_helper` / `_add_impl`，校验失败时回滚。`tests/kit` 13 passed（新增注册规则、两个互不影响的注册表）。a8w4 在过渡的 `collect` 上：K2 golden 和 op 级 golden 都与改前相同；API 测试通过。
- [x] R1 `689e112`：`_ops.py`；`ref.py` 8 个、`moe.py` 6 个函数加 `@op_register`；删 `collect` 和 `moe.py` 的 F401 导入。验证：
  - K2 golden（`supported` 20088 点、候选、名字、`lookup_config` 4608、`default_config` 1152、`dyn_config` 1152）与 op 级 golden（词汇表、实现顺序和函数、用例、每个用例的配置和原因、meta）都与改前相同；
  - 生成测试的 ID 集合相同（529 项）；三个测试文件在一个进程里收集（包被导入两次）得 560 项；
  - ISA IDENTICAL（558 次编译、177 个 ISA，4 个已有错误）；
  - API 23 + kit 13 passed；单卡 `test_a8w4_moe.py` + `test_a8w4_ops.py` 391 passed、146 skipped；`tune readme --check` 通过。
- [x] 文档 `8735b35`：README（文件表、复制到其他工程、8 节、9.3 节）、skill `a8w4-perf-experiment`。设计并进 `refactor7.md` A 部分（不提交）。
- **验收**：都已满足。

### 3.1 K：函数级 kit（已完成，设计见 `refactor6.md`）

**目标**：同一个 op 的参考实现、各个实现、测试输入和配置空间写成一族普通函数，靠名字前缀归到一起：

- 通用机制放进新的 `pyhip.kit`（`src/kit/`），和 a8w4 无关，只在 host 端；
- `ref.py` 定义问题：`moe_ref`、`moe_test`、`moe_check`、`moe_dims`；`moe_dyn_*` 同理；
- `moe.py` 放实现：`moe_tp2k` … `moe_ep1k`、`moe_dyn_ep3k`，配置空间和限制用 `@param`、`@requires`、`@meta` 写在函数上；
- 冻结的 API 签名和行为都不变：`supported`、`config_name`、`parse_config`、`run_config`、`lookup_config`、`dyn_config`、`a8w4_moe`、`a8w4_moe_dyn` 变成薄包装。

**步骤**（每步一个提交）：

| 步 | 内容 | 验收 |
|---|---|---|
| K0 | `pyhip.kit`：`param`、`requires`、`meta`、`stack`、`collect`、`Op`；`pyhip.kit.testing.op_tests`；`tests/kit/` 下用玩具 op 写的 CPU 测试，包括用 `pytester` 跑生成的测试 | `pytest tests/kit` 通过；不碰 a8w4 |
| K1 | `ref.py`：参考实现改为直接吃 `prepare_weights` 的输出（`decode_fd2` 解码 fd2 权重），`ref_moe(raw)` 和 `moe_ref(W)` 共用一个核心；加 `moe_dyn_ref` | `decode_fd2` 往返测试；`moe_ref(W)` 和 `ref_moe(raw)` 逐位相同（TP8、EP8）；API 测试；单卡 pytest |
| K2 | `moe.py` 的 runner 改成实现函数，`OPS = collect(...)`；`ref.py` 加 `moe_test`、`moe_check`、`moe_dims` 和 `moe_dyn_*`；`EP_CFGS` 从 `tune.py` 搬进 `moe.py` | 改之前存 golden JSON（`supported` 网格、tune 候选、CSV 名字往返、`lookup_config` / `dyn_config` / `default_config`），改完全部相等；ISA IDENTICAL；API 测试；单卡 pytest；`a8w4_moe` 每次调用的 host 时间不变差 |
| K3 | 生成的测试 `tests/test_a8w4_ops.py`（`test_ops = op_tests(OPS)`），删掉被它取代的手写测试 | 新测试覆盖旧测试的（形状、配置、token 数）组合；单卡 pytest 全过 |
| K4 | `tune` 改用 op：候选、参考结果、精度检查都走 `MOE` | `tune candidates` 输出逐字相同；CSV 格式不变；小规模 tune 能跑通 |
| K5 | bench 的 `tile_m`、`cached_loads` 改从 `@meta` 取 | bench 的 `--list` / 配置解析输出不变 |

**实现时定下的细节**（不对再改）：

- tune 保留自己的 `CASES`（生产形状 `mimo-v2.6-flash-*`），自己拼用例 dict，不用测试网格；所以 `Op.cases()` 先不支持过滤和替换轴（refactor6.md §3、§5.2 里的 `only` / `**axes` 暂不做）。
- `lookup_config` / `dyn_config` 按（shape、tokens、act、capturing、routing、表路径、表的 mtime）记住结果：通用的 `@requires` 检查比原来手写的 `supported()` 慢（每次 1.5–2.6 µs，原来 0.2–0.3 µs），记住结果后查表路径反而更快。
- bench 的 ep3k 覆盖项是 cfg 里的额外项（knob 以外的 key），只有带 `**over` 的实现（`moe_ep3k`、`moe_dyn_ep3k`）接受；`@requires` / `@meta` 也看得到它们。

**结果**：

- [x] K0 `3361672`：`src/kit/__init__.py`（533 行）、`src/kit/testing.py`（76 行）；`tests/kit` 用玩具 op 写的 CPU 测试 12 passed，包括用 `pytester` 跑生成的测试。
- [x] K1 `a6862e8`：`decode_fd2` 解码 prepare 后的权重；`moe_ref` / `moe_dyn_ref` 的参数与 `a8w4_moe` / `a8w4_moe_dyn` 相同；`ref_moe` 与改之前逐位相同（TP8 / EP8，1 / 77 / 600 tokens，silu / situv2）。新测试：`decode_fd2` 往返、`moe_ref` 与 `ref_moe` 逐位相同。
- [x] K2 `6acef65`：6 个实现函数和共用的限制 `tp_limits` / `ep_limits` / `ep3k_space`；`ref.py` 加 `moe_test` / `moe_check` / `moe_dims` 和 `moe_dyn_*`。验证：
  - 冻结函数的 golden JSON 全部相等：`supported` 20088 点、候选 12、名字往返 31、解析错误 8、`lookup_config` 4608、`default_config` 1152、`dyn_config` 1152；
  - ISA IDENTICAL（558 次编译）；
  - 每次 `a8w4_moe` 调用的 host 时间（eager，同卡 ABAB）：指定配置不变；查表路径 tp8 67.2 → 66.2 µs，ep8 76.2 → 73.2 µs。
- [x] K3 `b92ecc4`：`tests/test_a8w4_ops.py` 生成 506 项（360 passed，146 skipped，skip 原因来自 `@requires`），取代 4 个手写测试和 `recv_buffer`。旧的每个（shape、配置、token 数）组合都在里面；新增 tp4、situv2 的全部配置、`moe_dyn` 的全部 ep3k 预设。`2a8e774` 修了 K 之前就有的 `test_cuda_graph` 偶发失败：eager 选 tp1kq、capture 选 tp1k，两者之差恰好跨 1e-5 的阈值；现在 eager 参考用 capture 的配置。
- [x] K4 `9859360`：tune / verify 改用 `MOE.candidates`、`moe_test`、`moe_ref`、`MOE.run`、`moe_check`；删 `_situ`。验证：
  - `tune candidates` 的输出 tp8 / tp4 / ep8 逐字相同；
  - 4 个 token 数的 8 卡 tune，新旧 CSV 的表头和 key 相同，cos 相同或差 ≤ 3e-6；
  - 同卡 ABAB 计时，新 / 旧中位数 0.998 / 1.000（波动 ±2.5%）；
  - verify 的选择和 cos 相同。
- [x] K5 `2ee2a19`：bench 的 `resolve()` 改用 `MOE.why_not` / `run` / `meta`；删 `ep_cfg`、`_ep_cached`、`cached_loads()`、`block_m()`。验证：
  - `resolve()` 的输出（17 个 spec × tp8 / tp4 / ep8 silu / ep8 situv2 × 1 / 77 / 600 / 4096 tokens）：label、config、bm、cached 和 `ep_moe` 的参数完全相同，计算结果在噪声内；
  - CLI（`ab` tp8 / ep8、`breakdown`、协议拒绝、`check` tp8 / ep8）只差有意的改动：skip 行写出 `@requires` 的原因；ep8 的 `t16:g2.arb=512` 原来 resolve 之后在 kernel 里 assert 失败（check 报 3 FAILED），现在是 n/a 并给出原因。
- [x] 文档：README（文件表、复制到其他工程、8 节的参考实现和 bench check、9.1、9.3 加一个族）、skill `a8w4-perf-experiment`（landing checklist）、本文件。
- **验收**：都已满足。最后一次全量：API 23 passed；单卡 `test_a8w4_moe.py` + `test_a8w4_ops.py` 391 passed、146 skipped；kit 12 passed；`tune readme --check` 通过。

### 3.2 I5：host 封装改用 `@flydsl_kernel`（已完成）

**现状**（I4 之后）：16 个编译函数用 `functools.cache` 缓存，返回 `@flyc.jit` launcher；op 函数用 `_run_compiled(launch, x.data_ptr(), ..., torch.cuda.current_stream())` 启动。

| 文件 | 编译函数 → kernel 对象 | op（启动方） |
|---|---|---|
| `row_quant` | `_compile_row_quant` → `row_quant_kernel` | `quantize_rows` |
| `channel_exp` | `_compile_channel_exponents` → `channel_exp_kernel` | `channel_exponents` |
| `tp_prep` | `compile_tp_prep` → `tp_prep_kernel` | `tp_prep` |
| `tp_gemm1` / `tp_gemm2` | `compile_a8w4_gemm1/2` → `tp_gemm1_kernel` / `tp_gemm2_kernel` | `a8w4_gemm1/2` |
| `tp_fmoe` / `tp_fmoe_q` | `compile_a8w4_fmoe_bm64(q)` → `tp_fmoe_kernel` / `tp_fmoe_q_kernel` | `a8w4_fmoe_bm64(q)` |
| `ep_prep` | `compile_ep_prep` → `ep_prep_kernel` | `ep_prep` |
| `ep_gemm` | `compile_ep_gemm1/2` → `ep_gemm1_kernel` / `ep_gemm2_kernel` | `ep_gemm1/2` |
| `ep_small` | `compile_ep_small` → `ep_small_kernel` | `ep_moe_small` |
| `ep_mega` | `compile_ep_mega` → `ep_mega_kernel` | `megamoe.MegaMoeEpA8W4` |
| `ep_mega_comm` | `compile_quant/prep/plan/finish` → `mega_quant/prep/plan/finish_kernel` | 同上 |

其他调用方：`isa_snapshot.py`（megamoe 只编译不启动）、`bench/dist.py`（`compile_ep_mega`）；`experiments/` 里 6 个脚本 import `_run_compiled`，`20261115_mega_bcm/sweep.py` 调 `compile_ep_mega`。

**规则**：

- kernel 对象叫 `<文件名>_kernel`（用户 2026-11-16 定）；`ep_mega_comm` 的 4 个叫 `mega_<名>_kernel`。
- builder 函数体不改。`@flyc.kernel` 的名字不改，ISA、rocprof trace、perf 日志都对得上。编译 key 的参数名（`D_HIDDEN`、`BM` 等）也不改，留给 Q21。
- launch 的参数顺序不改。op 里直接传 tensor（`None` 表示空指针），去掉 `.data_ptr()` 和 stream；op 的签名不改（公开 API）。
- 默认值和 builder 的 keyword 默认值保持原样，不搬进 `heuristics`。省略某个 key 和显式传默认值会得到两个 variant，但它们是同一个二进制。
- `channel_exp` 的 `rows` 仍在 key 里：它只在准备权重时调用，每个模型一个形状。改成运行时参数会改 ISA，不在本迭代做。
- 常驻对象（megamoe）在 `__init__` 里用 `kern.get(**key)` 绑定一次，热路径直接调用。
- `_run_compiled` 搬到 `experiments/_launch.py`，实验脚本只改 import；调用 `compile_x(...)` 的地方改成 `x_kernel.builder(...)`（用户 2026-11-16 定）。例外：`20261115_mega_bcm/sweep.py` 把结果赋给 `MegaMoeEpA8W4.launch`，而它现在存的是 specialization，所以改成 `ep_mega_kernel.get(...)`。
- 无用的 kernel 参数（`tp_gemm1` 的 `tbl` / `rank` / `epoch` / `tiles`，`tp_gemm2` 的 `dbg` / `tbl` / `rank` / `epoch`，`tp_fmoe` 的 `dbg`）最后一步单独删。会改 kernarg 布局，ISA 的差异要逐个解释。

**任务**（每步一个提交；每步都跑 ISA 快照、API 测试和单卡 pytest，并用 `PYHIP_FLYDSL_CHECK=1` 跑一次 pytest，检查参数顺序、dtype 和对齐）：

- [x] 第 1 步（试点）：`row_quant`、`channel_exp`。`d306c51`
- [x] 第 2 步：TP，即 `tp_prep`、`tp_gemm1`、`tp_gemm2`、`tp_fmoe`、`tp_fmoe_q`。`3067a2d`
- [x] 第 3 步：EP，即 `ep_prep`、`ep_gemm`、`ep_small`。`af9f077`。`ep_small` 的 builder 原来返回 `(launch, info)`，现在只返回 launcher；容量改由 builder 和 op 共用的 `_capacity()` 计算。
- [x] 第 4 步：megamoe，即 `ep_mega`、`ep_mega_comm`、`megamoe.py`、`isa_snapshot.py`、`bench/dist.py`；多卡 pytest。`89bf081`。`isa_snapshot` 改用 `kern.preload(**key)`，删掉 `_dummy_args`；8 卡 mori ep8 / tp8 和 megamoe 通过，megamoe 在 `PYHIP_FLYDSL_CHECK=1` 下也通过。
- [x] 第 5 步：删除 `kernels/_launch.py`；新建 `experiments/_launch.py`，改实验脚本的 import 和 `.builder(...)`。`06d3875`。实验脚本都能 import，`20261115_flydsl_cache_key/run_probe.py` 跑通、结论不变；`exp.py lint` 输出不变。
- [x] 第 6 步：删无用的 kernel 参数，ISA 的差异逐个解释。`ff82c3d`，结果见下。
- [x] 文档：README（文件表、复制到其他工程、加 kernel 的写法、9.5 节）、本文件。
- **验收**：第 1–5 步 ISA IDENTICAL；第 6 步只有 kernarg 相关的差异；API 测试；单卡和多卡 pytest；`grep -rn "_run_compiled\|current_stream()\|functools.cache\|@cache" kernels/ megamoe.py` 没有结果（已满足）。

**第 6 步的 ISA 差异**（177 个不同 ISA 里变了 52 个，都是 `a8w4_gemm1`、`a8w4_gemm2`、`a8w4_fmoe_bm64`；按指令助记符逐对比较，脚本 `kclass2.py` 在 session 的 `files/i5_ab/`）：

| kernel | 变化 |
|---|---|
| `a8w4_gemm2`（8 个）、silu 的 `a8w4_fmoe_bm64`（10 个） | 只有 kernarg 元数据（`kernarg_size`、参数 offset）；指令完全相同（silu 不读 situ 的 f32 参数） |
| `a8w4_gemm1`（24 个） | kernarg 变短后标量 load 合并：silu 是 2 个 `s_load_dword` → 1 个 `s_load_dwordx2`，或 2 个 `s_load_dword` + 1 个 `s_load_dwordx2` → 1 个 `s_load_dwordx4`；situv2 是 `s_load_dwordx4` + 3 个 `s_load_dword` → 1 个 `s_load_dwordx8`。随之调度有小的重排，多 / 少 1 个 `s_mov`、`s_nop`、`s_waitcnt`；VGPR、SGPR、LDS 不变 |
| situv2 的 `a8w4_fmoe_bm64`（10 个） | 同上：2 个 `s_load_dword` → 1 个 `s_load_dwordx2`；寄存器分配变了，主循环里 MFMA、`v_perm` 等的顺序有小的重排 |

**第 6 步的性能 A/B**（`bench ab`，coldrot，5 轮；base 是 `06d3875` 的副本，以 `py:` variant 跑调优表的选择，new 是 `@table`；结果表在 session 的 `files/i5_ab/`）：

| case | act | tokens | new 相对 old | 噪声带 |
|---|---|---|---|---|
| TP8 | silu | 1、16、64、512 | +0.2、-0.2、+0.0、+0.1% | ≤1.0% |
| TP8 | situv2 | 1、16、64、512、1024、2048、8192 | -0.2、+0.0、-0.0、-0.0、+0.3、-0.1、+1.3% | ≤2.5% |
| TP4 | silu | 1、2、64、512 | +0.0、+0.1、+0.2、-0.0% | ≤0.4% |
| TP4 | situv2 | 1、2、64、512、2048、8192 | -0.2、+0.1、-0.2、-0.1、+0.8、-0.4% | ≤1.3% |

全部在噪声带内（verdict 都是 “~”）。

**host 开销**（每次 `a8w4_moe` 调用的 CPU 入队时间，200 次不同步取中位数，脚本 `host.py` 在 session 的 `files/i5_ab/`；I5 前 = `97eb73e`）：

| case | config | I5 前 | I5 后 |
|---|---|---|---|
| TP8 1 / 64 tokens | `tp2k bm32 g1bcm2 g2base` | 71–74 µs | 61 µs |
| TP8 1 / 64 tokens | `tp1k bcm1 ks1` | 45–47 µs | 40 µs |
| EP8 1 / 64 tokens | `ep3k m64` | 78–80 µs | 67 µs |

省下的主要是每次 launch 的 `torch.cuda.current_stream()` 和 `.data_ptr()`。剩下的大头在 op 的 Python 逻辑（查表、分配、`moe_sorting` 等），留给远期的“降低 host 开销”。

### 3.3 I4：kernel 文件之间不再互相 import（已完成）

**现状**（I3 之后）：kernel 文件之间的引用都是模块属性调用（`G1.x`、`EP.x`、`PR.x`、`F.x`），这些 helper 不进 FlyDSL 缓存 key；还有逐字节相同的重复定义：

- `tp_gemm1` 和 `tp_gemm2` 各有一份 `_perm_u32`、`_fp8_lut_predelta`、`_fp4x8_to_fp8x8_fast`、`_channel_factor`、`_fp8_frag8`、`_mma_fp8`；`tp_fmoe`、`tp_fmoe_q`、`ep_gemm`、`ep_mega`、`ep_small` 经 `G1.` 用 `tp_gemm1` 的那份；
- `ep_mega`、`ep_small` 经 `EP.` 用 `ep_gemm` 的 K 循环 `_make_core` 和几个小 helper；`tp_fmoe_q` 经 `F.` 用 `tp_fmoe` 的 helper；`ep_small`、`tp_prep`、`ep_mega_comm` 从 `ep_prep` 拿原子操作和 `NUM_CU`；`ep_mega_comm` 从 `ep_mega` 拿 flag 布局常量；
- `_rsrc` 有 3 个版本（`tp_fmoe`、`ep_small` 的相同，`ep_mega` 的多一个 `nbytes`）；`NUM_CU = 304` 定义了 2 次，`NXCD = 8` 3 次，LDS 上限 65536 写了 4 次，cache 策略位 17（sc0|sc1）有两个名字。

**范围**：只动 device 代码。host 端的组合（`ep_gemm.ep_moe` 调 `ep_prep`）留给 I5。规则：

- 被两个以上 kernel 文件用到的 helper 和常量，搬进共享模块，按名字导入（这样也进缓存 key）；只有一个文件用的留在原处。
- 函数体不改，名字不改；逐字节相同的重复定义只留一份。实现不同的（`_fp8x4_pack` 和 `_pack_activation`）先比 ISA，相同才合并；不同就问用户（结果见第 4 步）。
- 顺手删掉没用的模块级名字（`ep_gemm`、`ep_small` 的 `HERE`）。

**目标布局**（新模块都在 `kernels/`）：

| 模块 | 搬进来的内容 |
|---|---|
| `_cvt.py` | `_perm_u32`、`_fp8_lut_predelta`、`_fp4x8_to_fp8x8_fast`、`_channel_factor`（`tp_gemm1/2`）；`_bf16x2_rne`、`_fp8x4_pack`（`ep_gemm`）；`_rnu_hi`（`tp_fmoe`） |
| `_mma.py`（新） | `_fp8_frag8`、`_mma_fp8`（`tp_gemm1/2`） |
| `_act.py` | `situ_args`（host 端，`tp_gemm1`） |
| `_hw.py`（新） | `NUM_CU`、`NUM_XCD`、`LDS_BYTES_MAX`、cache 策略位 `CM_*`、`DPP_QUAD_SWAP`、`_u`（readfirstlane）、`_opaque_tx` |
| `_atomic.py`（新） | `_gatomic_add`、`_gatomic_xchg`、`_gload_sys`、`_latomic_add`（`ep_prep`）；`_atomic_bf16x4`（`ep_gemm`）；`_atomics4_saddr`、`_atomics_mi`（`tp_fmoe`） |
| `_mem.py` | `_rsrc`（3 个版本合一，`nbytes` 可选）、`_ld`、`_store`（`ep_mega`）、`_lds_f32_store_masked`（`ep_gemm`） |
| `_kloop.py`（新） | `_make_core`（流式读 W 的 K 循环，ep3k / ep1k / megamoe 共用）、`_frag`、`_masked_lds_store`、`_dispatch_rows` |
| `_protocol.py`（新） | 一个 kernel 写、另一个 kernel 读的 buffer 格式：EP 调度表的 `SCHED_BORDER`（`ep_prep` → `ep_gemm`）；megamoe 的 flag / 队列布局（`ep_mega` ↔ `ep_mega_comm`） |

**任务**（每步一个提交，每步都跑 ISA 快照）：

- [x] 第 1 步：TP 的重复 helper → `_cvt`、`_mma`；`situ_args` → `_act`。改 `tp_gemm1/2`、`tp_fmoe(_q)`、`ep_gemm`、`ep_mega`、`ep_small`、`megamoe.py`。
- [x] 第 2 步：`_hw`、`_atomic`、`_mem` 的新增；改 `ep_prep`、`tp_prep`、`ep_small`、`tp_fmoe(_q)`、`ep_mega(_comm)`、`tp_gemm1/2`（`NXCD`）。`_getreg` 只有 `tp_fmoe_q` 用，搬到那里；`CM_COH` / `_CM_SYS` 合成 `CM_SYS`，`ep_mega` 的 `P2P_AUX` / `WT_AUX` 和 `tp_fmoe_q` 的 `CM_SC1` 改用 `CM_*` 位；`ep_prep` 的 `NX`、`ep_gemm` 的 XCD 轮转里的字面量 8 改成 `NUM_XCD`；`experiments/20261114_tp_prep` 的 3 个副本跟着改 import。`prof.py` 的 65536 不改（它不导入 flydsl）。
- [x] 第 3 步：`_kloop`、`_protocol`；`ep_mega`、`ep_small`、`ep_mega_comm` 不再 import 别的 kernel 文件，`ep_gemm` 只剩 host 端的 `ep_prep`。`megamoe.py` 的 `K.NPES` 这类常量改成从 `_protocol` 按名字导入；`experiments/20261114_splitk_gemm1/ep_gemm_sk.py` 跟着改 import。改写工具误报过两次（元组赋值里定义的名字、从非 kernel 模块导入的同名常量），已修。
- [x] 第 4 步：`_fp8x4_pack` 和 `_pack_activation` 的 ISA 不同（后者用传播 NaN 的 `minimumf` / `maximumf`，NaN 打成 0；前者用 `minnumf` / `maxnumf`，NaN 打成 240），不能无损合并。用户先按“正常使用中激活和权重不含 NaN”定为全部改用 `_fp8x4_pack`、删掉 `_pack_activation`；之后又提出 PyTorch 要求算子尽量传播 NaN，想回滚。回滚前实测了 NaN / ±Inf 注入（见下），NaN 传播由 amax 决定，和 pack 函数无关，用户决定保留合并。这是本迭代唯一有意改变 ISA 的一步，验证如下：
  - ISA：只有原来调用 `_pack_activation` 的 kernel 变了，指令数都减少，VGPR 和 spill 不变或更少。按族：`tp_prep` 266 个全变，−22% 到 −27%；`ep_prep` −12% 到 −19%；TP 的 gemm1 −4% 到 −8%；`fmoe` 约 −4%；`ep_small` −6.5%；`ep_gemm1` 的非 tr epilogue −3% 到 −4%；`row_quant` −33%。megamoe、gemm2、`ep_gemm1` 的 tr epilogue 不变。
  - 输出逐位相同：topk 1 下用 HEAD 的 worktree 和改后的代码各跑一遍，覆盖 tp8 / tp4 / ep8 × silu / situv2 的全部族，以及 dyn 的 bf16 / fp8g128 / fp8tok。380 个确定性输出全部相同。另有 43 个没比：split-K 原子累加本身不确定的 `tp1k ks4/ks8`，以及已知 LDS 超限的 `l128_v1` fp8g128。
  - NaN / ±Inf 注入（改前 `5e5d9e3` 对改后，tp8 / tp4 / ep8 + dyn bf16，64 和 1024 token，共 83 个配置；在一个路由到本地专家的 token 的 `x[r, 100]` 放 NaN、+Inf 或 −Inf）。PyTorch 参考 `ref_moe` 在三种情况下都是 `out[r]` 整行 NaN，其他行不受影响。
    - NaN：改前改后 83 个全部相同。原来调用 `_pack_activation` 的地方，amax 都用 `fx.max`（传播 NaN），scale 已经是 NaN，字节是 0 还是 240 不影响结果。
    - ±Inf：70 个相同。另外 13 个（ep3k 的 tr 配置 `m64`、`l128`、`l128c`，`ep1k m512`，以及对应的 dyn）改前输出有限的错误值，改后是整行 NaN，和 ref 一致。改前是 `Inf × 0 = NaN` 被打成 0，整行变成 `0 × Inf`，再被下游 `maxnumf` 的 amax 丢掉。
    - 改前改后都有的问题：这 7 个配置在 NaN 输入时也输出有限的错误值，根因是重量化的 amax 用 `maxnumf`。已列为 I4b（1.1 节）。
    - 脚本：`/tmp/i4/nan.py`、`/tmp/i4/naninf.py`（临时，I4b 时写进 pytest）。
  - 性能（coldrot，同进程交替，HEAD 为 base）：tp8、tp4、ep8 共 20 个点，加 tp8 situv2 的 3 个点，没有变慢的。tp8 和 tp4 在 64–256 token、tp8 situv2 在 16 和 1024 token 超出噪声，快 0.4%–1.6%，其余在噪声内。
  - API 测试 23 passed；单卡 pytest 94 passed、1 skipped。
  - ISA 基线从这一步的提交起换成新快照（见 0.3 节）。
- [x] 第 2 步补漏：`tp_fmoe` 的 XCD 轮转里还有一个字面量 8，改成 `NUM_XCD`。ISA IDENTICAL。
- [x] 文档：README 的目录结构（新共享模块、kernel 文件之间不互相 import）、`_make_core` 的位置、9.2 节硬件常量（`_hw.py`）、12 节来源表；`flydsl-gfx942-pitfalls` skill 指向 `_hw.py`；`_fp8x4_pack` 的 docstring 写明 NaN 靠 scale 传播；本文件。
- **验收**：ISA IDENTICAL（第 4 步除外）；API 测试；单卡 pytest；`grep "from \. import\|from \.[a-z]" kernels/*.py` 只剩共享模块（`_` 开头）和 `ep_gemm` 的 `ep_prep`。全部满足。

### 3.4 I3：精简 `_vendor`，迁到 `kernels/` 的共享模块（已完成）

**背景**：`_vendor/` 是为了不依赖 aiter，从 aiter 原样复制来的 9 个文件（2268 行）。用 AST 从包内所有调用点（含 `experiments/`）做可达性分析，约 1170 行走不到：

- `dpp_utils.py`、`layout_utils.py` 整个没用；`kernels_common.py` 只用到 `LOG2E`；
- `tensor_shim.py` 只用到 `_run_compiled`、`_preload_compiled`、`_to_raw`；原 I3 的 3 个环境变量（`AITER_FLYDSL_KERNARG_PRELOAD`、`..._COUNT`、`AITER_FLYDSL_MOE_EXPERT_SCHEDULING_MODE`）没有任何代码读，`TensorView` / `GTensor` 等类也没用；
- `mxfp4_gemm_common.py` 60 个定义只用到 8 个；`buffer_ops.py` 的 `BufferResourceDescriptor`、`create_buffer_resource` 没用。

**规则**：

- 删：走不到的定义；用到的函数里没有调用方传的参数和分支（`buffer_load` 的 `is_scalar` / `soffset_bytes`，`buffer_store` 的 `offset_is_bytes`，`get_element_ptr` 的静态偏移，`gate_up_act` 的 `swiglu`，`sigmoid_batch` 的 `alpha`）。
- 冻结：只在 gfx942 和 FlyDSL 0.3.2 上取一个值的兼容分支（V# flags 读 `FLYDSL_GPU_ARCH` 判断 RDNA；`RawPtrBufferLoadOp` 的 `aux` 是属性还是操作数）。
- 函数名暂不改（按术语表改名是中期的事），只改模块位置和 import。唯一的例外：`_gep1` 和 `get_element_ptr` 签名相同，合并成 `get_element_ptr`。

**结果布局**（共 464 行；helper 都按名字导入，和 kernel 同一目录，大部分进了 FlyDSL 缓存 key）：

| 模块 | 行数 | 内容 | 来源 |
|---|---|---|---|
| `kernels/_mem.py` | 232 | IR 值转换 `_raw` / `_unwrap_value`；`_udiv` / `_umod`；global / LDS 指针、`get_element_ptr`；buffer resource、`buffer_load` / `buffer_store`；i32 buffer view | `buffer_ops`、`mxfp4_gemm_common`、`moe_utils`、`tensor_shim._to_raw`、`a8w4_ops._pointer` |
| `kernels/_act.py` | 119 | gate / up 激活（silu、situv2） | `act`、`kernels_common.LOG2E` |
| `kernels/_cvt.py` | 19 | `_pack_activation`（4 个 f32 打包成 fp8，饱和到 ±240，-0 → +0）；I4 去重的 fp4 解码等也放这里 | `a8w4_ops` |
| `kernels/_launch.py` | 24 | host 端 `_run_compiled`、`_preload_compiled`：只是 `pyhip.contrib.flydsl.host_launch` 的 `_compile_and_run` / `_compile_only` 的薄封装，不再自带一份 `COMPILE_ONLY` 加锁逻辑（I5 换成 `@flydsl_kernel` 后删除） | `tensor_shim` |
| `kernels/channel_exp.py` | 70 | W 的通道指数扫描 kernel 和 host 入口 `channel_exponents`（`prepare_weights` 用） | `a8w4_ops` |

**任务**：

- [x] 第 1 步（`18f243f`）：在 `_vendor` 里原地删掉走不到的代码和参数，冻结兼容分支。2268 → 553 行，删掉 `dpp_utils`、`layout_utils`、`kernels_common` 三个文件。ISA IDENTICAL。
- [x] 第 2 步（`cc44955`）：迁到上表的模块；`bo.X` / `G1._buffer_i32_scalar_read` 这类模块属性调用改成按名字导入，import 只留用到的名字；顺手删掉 kernel 文件和 `ep_gemm_sk` 里 25 个没用的 import；`experiments/`、`megamoe.py`、`isa_snapshot.py`、`weights.py` 跟着改；删除 `_vendor/` 和 `pyproject.toml` 里的子包。
- [x] 文档：README（简介、目录结构、依赖表、复制到其他工程、缓存 key 规则、pyproject 子包、来源表）；skill `flydsl-gfx942-pitfalls`（缓存 key 表加两行）、`flydsl-host-launch`（示例的 import）、`a8w4-perf-experiment`。
- **验收**：ISA IDENTICAL（177 个 ISA）；`channel_exponents` 不在快照里，单独比对了新旧 ISA 和输出，逐字节相同；API 测试 23 passed；单卡 pytest 94 passed、1 skipped。

**经验**：

- FlyDSL 0.3.2 的缓存 key 只收集“按名字引用、和 jit 文件同一目录”的普通 helper（`jit_function._collect_dependency_sources`：读每个函数的 `co_names` → `__globals__`，以及闭包）。所以：
  - 经模块属性调用（`bo.buffer_load`、`G1.x`）的 helper 不进 key，迁移时一律改成按名字导入；
  - `@dsl_loc_tracing` 把函数包成 FlyDSL 目录下的 wrapper，也会把 helper 挡在 key 外面。它只加 MLIR location，`.s` 里没有 `.loc`，所以去掉了；
  - 不进入嵌套函数：`ep_gemm` 的 `_body` 里 `_epilogue` 调用的 `gate_up_act`、`lds_vec_load` 等仍然不在 key 里。这是 FlyDSL 的限制，不是这次引入的（原来 `_vendor` 的 helper 全都不在 key 里）。改 helper 后照旧用空的 `FLYDSL_RUNTIME_CACHE_DIR`。
- `_raw` 和 `_unwrap_value` 不能合并：后者让 Python int 原样通过，`buffer_load` 等的偏移参数靠它接受常量。
- 第 1 步删 `get_element_ptr` 的 `elem_type` 参数时漏了 `_gep1` 的调用点，ISA 快照的编译失败直接暴露了问题：快照每次用新的 `FLYDSL_RUNTIME_CACHE_DIR`，结论可信。
- 引入 `host_launch` 后，导入 a8w4 会加载 pyhip 的这个模块（多约 0.03 s）。`kernels/_launch.py` 是包内唯一的 pyhip 绝对导入，README 的“复制到其他工程”已写明（I5 删了它，改为各 kernel 文件导入 `flydsl_kernel`）。

### 3.5 I2：删除 EP kernel 的调试开关和 a16w4 路径（已完成）

**范围**（a16w4 依据 Q11：以后要上线时作为独立的族加回来）：

- [x] 调试开关（`892d7f8`）：
  - `ep_gemm` 的 `TL`（per-tile 时间线），连同 `tl_buffer`、`_tl_now`、`_tl_record`、`_sreg`；
  - `ep_small` 的 `DBG`（只跑 gemm1 / 什么都不跑），由 `cfg["dbg"]` 传入，没有调用方设置它。
- [x] `ep_gemm` 的 a16w4（`87fd7c8`）：
  - `compile_ep_gemm1/2` 的 `ADT` 参数，`ep_gemm1/2`、`ep_moe` 的 `adt` 参数；
  - `_make_core` 的 `FP8` 参数（`ep_mega` 和 `ep_small` 也传了它）；
  - `_fp8x4_to_bf16x4`、gemm1 的 bf16 输出、`CFGS` 里的 `bf16` 覆盖项；
  - 模块 docstring。
- [x] `ep_small` 的 a16w4（`87fd7c8`）：`ADT` 和 `adt`，连同 a16w4 专用的“第一个任务之后再做 setup”的顺序；`_scratch` 的缓存 key 去掉 `adt`。
- [x] 调用方（`87fd7c8`、`7dbb85e`）：
  - `moe.py` 不再传 `adt="fp8"`；
  - `bench/core.py` 删掉 variant 的 `adt=` 覆盖；
  - `experiments/` 里传 `adt="fp8"` 的脚本去掉这个参数；
  - `experiments/20261115_ep_prio/a16w4_ab.py` 只能在 I2 之前的提交上跑，文件头加说明。
- **不动**：`ref.ref_moe` 的 `quant_x` 参数（冻结 API）；`AQ8`（A 是 bf16 源、在 staging 时量化成 fp8，属于 a8w4）；ISA 快照里 `dyn:*:bf16` 是 x 的输入格式，和 a16w4 无关。
- kernel 名字照旧（`ep_gemm1_fp8_…`、`ep_small_fp8_…_d0`）。
- **验收**：和 I1 相同。
- **验收结果**：
  - 每个提交都是 ISA IDENTICAL（558 次编译，177 份 ISA；快照里有 ep1k、ep3k、mega、dyn 的作业）；kernel 名字逐字不变；
  - API 测试 23 passed；单卡 pytest：94 passed、1 skipped，与基线一致；
  - `ep_gemm` 1707 行减到 1545 行，`ep_small` 609 行减到 592 行。

**经验**：

- `_make_core` 是 `ep_gemm`、`ep_small`、`ep_mega` 共用的，改它的签名要在同一个提交里改完所有调用方。
- `experiments/20261114_splitk_gemm1/ep_gemm_sk.py` 是独立的副本，`SKG.*` 调用照旧带 `adt`，不要跟着改。
- `ep_small` 的 gemm1 中间结果本来就是 bf16（两种模式都是），`_bf16x2_rne` 仍在用，不属于 a16w4。

### 3.6 I1b：删除 TP kernel 里调用方不用的模式取值（已完成）

**规则**：

- **删**：模式开关里调用方不用的取值，也就是另一种实现方式，连同只为它们存在的辅助函数和参数。
- **留**：数值型调优参数，比如 tile 尺寸、`BM`、`b_cache_mod`、`xcd_swizzle`、`waves_per_eu`、`NTILE`、`ksplit`、`n_ctas`。即使调用方只用一个值也留着。
- **留**：调用方确实会走到的取值。gemm1 的 `dyn_m=False` 对应调优表里的 bm32；`pipeline` 的自动规则（`rows <= 32`）两条路径都会走到，只删手动覆盖的参数。
- **kernel 名字不变**：被折叠的标签（`_ftz`、`_fd2`、`_q8s` 等）写成常量，方便对照 perf 日志、rocprof trace 和 ISA 基线。

**任务**：

- [x] `kernels/tp_gemm1.py`（`aa91396`，1044 行减到 915 行）：
  - `epilog` 只留 `"q8s"`，删掉 bf16 epilogue（连同 `_body` 的 `arg_cumsum` 参数）；
  - `fast_dec` 只留 2，删掉 `_fp8_lut`、`_fp4x8_to_fp8x8`、`_fp8_lut_fast`；
  - 删掉 `flush_tiny` 参数和 `pipeline` 覆盖参数（现在固定为 `rows <= 32 and K_TILES > 1`）；
  - `prep_w1_fast(w, s, exp)` 只做 fd2，去掉 `fast_dec` 参数；
  - host 的 `inter_dim` 改成必填。
- [x] `kernels/tp_gemm2.py`（`2b2efaa`，1225 行减到 1034 行）：
  - `fast_dec` 只留 2，删掉 `flush_tiny`；
  - `A_GROUP` 只留 128，`dyn_rows` 只留 True；
  - `epilog` 由 `NTILE` 决定，删掉这个参数；NTILE 时 `ROW_VGPR` 恒为真，LDS 里放行数据的那条路径删掉了。
- [x] `kernels/tp_fmoe.py`（`4f478b4`）：`dyn_rows`、`xcd` 恒为真。
- [x] 同步修改调用方：`moe.py`、`weights.py`，以及 `experiments/20261114_tp_prep/run.py`、`experiments/20261114_splitk_gemm1/run_tp.py`（`order.py` 没传这些参数）。
- **验收结果**：
  - 每个提交都是 ISA IDENTICAL（558 次编译，177 份 ISA）；kernel 名字逐字不变；
  - API 测试 23 passed；
  - 单卡 pytest：94 passed、1 skipped，与基线一致。

**经验**：

- 折叠掉的选项在 kernel 名字里的标签，直接写成字符串常量，并加注释说明“这些选项现在恒开”。
- 只服务于被删路径的 buffer、copy atom 等，删掉后 ISA 不变（本来就会被 DCE）。

### 3.7 I1：删除 TP kernel 里走不到的代码（已完成）

host 端（`moe._run_tp2k`，G1/G2 唯一的调用方）只用到 gemm1 的 `epilog="q8s"`（`moe.py:170`），以及 gemm2 的 `lds` 和 `epilog="direct64"`（`moe.py:178`）；megamoe 只用到 `G1.situ_args`。

- [x] `kernels/tp_gemm1.py`（`0b0e3c9`）：
  - 删除 `epilog` 的 `"q8"`、`"q8push"`、`"q8pushs"`，以及只服务于它们的参数和代码（`tbl`、`rank`、`epoch`、`npes`、`SCALE_RUN` 等）；
  - 删除 `w_layout="tile"`；
  - 删除 `PUSH_VAR`。
- [x] `kernels/tp_gemm2.py`（`725209f`，1741 行减到 1224 行）：
  - 删除 `TILE_ORDER`、`PROBE`、`CHUNK_N`、`AR`、`AR_SEP`、`STAGE_ROWS`。AR 分支 import 了一个不存在的 `gemm2_ar_fused`，本来就跑不了；
  - 把冻结常量 `EPI_DRAIN`、`BF16_FAST`、`WPERM`、`PERSIST`、`AR_HOOK` 折叠掉；
  - 删除走不到的 `epilog` 分支：现在 NTILE=0 只用 `lds`，NTILE>0 只用 `direct64`。
- [x] `kernels/tp_fmoe.py`（`51817b1`）：把冻结常量 `ABL`、`LDS_PAD`、`DELAY`、`G1_SPREAD`、`PROBE` 折叠掉，`NO_ATOMIC` 也一起删了。`tp_fmoe_q` 的 kernel 名原来读 `F.NO_ATOMIC`，在同一个提交里改掉了。
- [x] `kernels/tp_fmoe_q.py`（`f9d75f3`）：把 `DBG` 折叠掉，连同 trace 辅助函数和 `TRACE_N`。
- **验收结果**：
  - 每个提交都跑了 `isa-snapshot` 和 `isa-diff`，结果都是 IDENTICAL（558 次编译，177 份不同的 ISA）；
  - 每个提交都跑了 `test_a8w4_api.py`：23 passed；
  - 单卡 pytest：94 passed、1 skipped，与基线一致。

**这一轮的经验**（以后的迭代也适用）：

- 删 kernel 参数会改变 kernarg 偏移，从而改变 ISA。所以没用的参数先保留，标成 `# unused`，留到 I5 一起删。
- 删除没用的 IR 运算（比如 `+0`、算了不用的值）不影响 ISA，会被 DCE 掉。
- 恒为真的条件不一定能删：`tp_gemm2` 里的 `bx >= 0` 删掉后指令调度变了，ISA 不同，所以保留并加了注释。
- 跨文件引用要查全：`tp_fmoe_q` 通过 `F.` 引用 `tp_fmoe` 的常量。删一个模块级名字之前，先 grep 所有 `模块别名.名字` 的用法。
- `experiments/` 里的脚本也会调用 kernel 的 host 函数，ISA 快照和 pytest 覆盖不到。删 host 参数后要 grep 这些脚本（`20261114_tp_prep/run.py` 原来给 gemm1 传了 `topk=`，已改）。
- README 9.5 节原来列着这些实验选项，已经同步改了。

当时列出的 I1b 候选（`fast_dec=0/1`、`flush_tiny`、`A_GROUP=0`、`ROW_VGPR=False`、`dyn_rows=False`、gemm1 的 `epilog="bf16"` 和 `pipeline`），用户决定删，已在 I1b 完成（见 3.6 节）。`dyn_m=False` 和 `xcd_swizzle` 按规则保留。

## 4. 迭代记录

| 日期 | 迭代 | 结果 | 提交 |
|---|---|---|---|
| 2026-11-15 | 0 安全网 | `prof isa-snapshot` / `isa-diff` / `isa-cover`；API 契约测试和 CSV 往返测试；ISA 和 pytest 基线 | `64ce77c` `d17f502` `4e256d9` |
| 2026-11-16 | I1 删除 TP kernel 的死代码 | 4 个文件净减约 830 行；ISA IDENTICAL；API 测试 23 passed；单卡 pytest 94 passed、1 skipped | `0b0e3c9` `725209f` `51817b1` `f9d75f3` |
| 2026-11-16 | I1b 删除 TP kernel 里不用的模式取值 | `tp_gemm1` 1044→915 行，`tp_gemm2` 1225→1034 行，`tp_fmoe` 去掉两个开关；kernel 名字不变；ISA IDENTICAL；API 测试 23 passed；单卡 pytest 94 passed、1 skipped | `aa91396` `2b2efaa` `4f478b4` `8399c9d` |
| 2026-11-16 | I2 删除 EP 的调试开关和 a16w4 路径 | `ep_gemm` 1707→1545 行，`ep_small` 609→592 行；`_make_core` 去掉 `FP8`；bench 去掉 `adt=` 覆盖；kernel 名字不变；ISA IDENTICAL；API 测试 23 passed；单卡 pytest 94 passed、1 skipped | `892d7f8` `87fd7c8` `7dbb85e` |
| 2026-11-16 | I3 精简 `_vendor`，迁到 `kernels/` | `_vendor/` 9 个文件 2268 行 → `kernels/` 5 个共享模块 464 行；helper 按名字导入，大部分进了 FlyDSL 缓存 key；`_run_compiled` 改用 pyhip `host_launch`；删 25 个无用 import；ISA IDENTICAL；`channel_exponents` ISA 和输出逐字节相同；API 测试 23 passed；单卡 pytest 94 passed、1 skipped | `18f243f` `cc44955` |
| 2026-11-16 | I4 kernel 文件之间不再互相 import | 新增共享模块 `_mma`、`_hw`、`_atomic`、`_kloop`、`_protocol`，扩充 `_cvt`、`_mem`、`_act`；kernel 之间只剩 host 端 `ep_gemm` → `ep_prep`；重复的 helper 和硬件常量各只一份。第 1–3 步和补漏 ISA IDENTICAL；第 4 步有意改 ISA（合并 fp8x4 pack，指令更少，输出逐位相同，NaN 不变、±Inf 更接近 ref，快 0–1.6%）；API 测试 23 passed；单卡 pytest 94 passed、1 skipped | `6c8434c` `4b9eb2f` `5e5d9e3` `e565657` `f9d0ca1` |
| 2026-11-16 | I5 host 封装改用 `@flydsl_kernel` | 16 个编译函数改成 kernel 对象（`<文件名>_kernel`），op 直接传 tensor，去掉 `_run_compiled`、`current_stream()`、`.data_ptr()`、`functools.cache`；megamoe 构造时绑定 `kern.get()`；删除 `kernels/_launch.py`（实验脚本改用 `experiments/_launch.py`）；删除 TP kernel 的 9 个无用参数。第 1–5 步 ISA IDENTICAL；第 6 步有意改 ISA（kernarg 变短、标量 load 合并），TP4 / TP8 A/B 在噪声内；host 开销每次调用少 6–13 µs；API 测试 23 passed；单卡 pytest 94 passed、1 skipped（含 CHECK 模式）；8 卡 pytest 3 passed | `d306c51` `3067a2d` `af9f077` `89bf081` `06d3875` `ff82c3d` |
| 2026-11-17 | K 函数级 kit | 新增 `pyhip.kit`（op = 一族同前缀的函数，`@param` / `@requires` / `@meta` / `stack`、`collect`、`op_tests`）；`ref.py` 定义问题（`moe_ref` 读 prepare 后的权重、`moe_test`、`moe_check`、`moe_dims`、`moe_dyn_*`），`moe.py` 的 runner 表改成 6 个实现函数；测试由 op 生成（506 项），tune / verify / bench 都改走 op。冻结 API 的 golden JSON 全部相等；ISA IDENTICAL；查表路径 host 时间少 1–3 µs；tune 候选逐字相同、计时不变；bench 只有 skip 原因变具体、覆盖项受 `@requires` 检查。API 测试 23 passed；单卡 pytest 391 passed、146 skipped；kit 12 passed | `3361672` `a6862e8` `6acef65` `b92ecc4` `2a8e774` `9859360` `2ee2a19` `4cb1ded` |
| 2026-11-17 | R op 注册表 | `pyhip.kit.Registry` + `@op_register` 逐个注册（名字定角色，注册时校验，失败回滚），取代 `collect`；a8w4 一包一个注册表 `_ops.ops`，ref.py 8 个、moe.py 6 个函数注册，`MOE = ops.moe`。冻结函数 golden 和 op 级 golden 全部相等；生成测试 ID 集合不变（顺序 moe 在前）；包被导入两次时两份注册表互不影响；ISA IDENTICAL；API 23 passed；kit 13 passed；单卡 pytest 391 passed、146 skipped | `716cc46` `8101410` `689e112` `8735b35` |
| 2026-11-17 | D 查表分派进 kit，取消冻结 API | kit：`Registry(table=, bucket=, relax=)`、`Op.pick` / `points` / `__call__`、`<op>_fallback`；`op(...)` 按调用记住实现，mtime 每秒最多读一次。a8w4：`_ops.py` 声明表；两个兜底函数；`moe_dyn` 词汇改名（capacity / num_tokens）；删 7 个元组形式的函数；薄包装只留 M == 0，入参检查进 dims；工具和测试改用 op 的方法。K2 golden 只有三类有意差异；op golden 中 moe 相同、moe_dyn 只差改名；测试 ID 不变；ISA IDENTICAL；API 22、kit 15 passed；单卡 413 passed、146 skipped；8 卡 3 passed；查表路径的 host 时间在噪声内 | `056028b` `5d0b861` `4f51e94` `23a3e8f` `e60470a` |
| 2026-11-17 | E 去掉 `@requires` | 限制只在 kernel 的 assert 里（164 种被拒绝的组合实测都是干净的 `AssertionError`）；只保证调优表里的 key，没调过的抛 `ValueError`；删兜底；`@no_capture`、`Op.tuned`、`Op.capturable`；生成的测试测调优过的配置（与原来没 skip 的 360 项相同）；ISA IDENTICAL；单卡 413、8 卡 3 passed | `e522ebe` |
| 2026-11-17 | T1–T4 通用 tune / bench | `pyhip.kit.table`（表的读写、合并、比较）、`kit.timing`（冷 cache 协议、ABBA、噪声带）、`kit.tune` + `python -m pyhip.kit`（cases / candidates / tune / merge / show / verify / tune-all / compare）、`kit.bench`（ab / check）；a8w4 的 tune、bench 变成薄包装。表文件逐字节相同；tune 的候选、表头、选择相同，计时在噪声内；bench 新旧输出相同；ISA IDENTICAL；kit 18、单卡 413、8 卡 3 passed | `bebaff3` `3637665` `ef107ba` `712fc05` |
| 2026-11-18 | V 术语 dim / knob，测试和调优集中声明 | `@dim`（problem 的取值）/ `@knob`（solver 的网格）；`op.problems`、`op_tests(op, problems)`、`Tuning`；表头 `~num_tokens` 标出 bucket，`Registry(name, table)`；去掉 relax，补调 tp4-situv2。测试 ID、候选、show 不变；分派只在 tp4-situv2 上变；ISA IDENTICAL；kit 18、单卡 413、8 卡 3 passed | `af50811` `9adc2f0` `893e3f2` |
| 2026-11-18 | W 一个概念一个名字 | kind → solver（配置键、`Op.solvers`）；`<op>_test` → `<op>_problem`；dict 参数 `dims` → `problem`；data → inputs；CLI `candidates` → `configs`。测试 ID、分派、CLI 输出不变；ISA IDENTICAL；kit 18、单卡 413、8 卡 3 passed | 见 git log |
| 2026-11-18 | kit 迁出 pyhip | `pyhip.kit` 迁到独立仓库 [tingqli/kernelkit](https://github.com/tingqli/kernelkit)（包 `kernelkit`，CLI `python -m kernelkit`，v0.1.0）；pyhip 删掉 `src/kit`、`tests/kit`，`pyproject.toml` 依赖 `kernelkit @ git+https://github.com/tingqli/kernelkit.git@v0.1.0`；a8w4 的 import 和文档改指 kernelkit。测试 ID 不变；单卡 413 passed；kernelkit 18 passed | 见 git log |
| 2026-11-18 | EP 整层算子 moe_ep（kernelkit v0.2.0） | kernelkit 0.2.0：solver 上的 `@multi_gpu(world, init=)`（卡数可以是 problem 的 dim，进表的 key），tune / verify / ab / check 自动起 world 个 rank（`kernelkit.spmd`：首次调用和 check 的结果各 rank 一致，计时取最慢的 rank，rank 0 写表），tune-all 按卡数分配 GPU。a8w4：`moe_ep.py`（`EpComm`；solver mega / mori / mori_eager，30 个配置），`_ops.ep_ops` 和表 `configs/a8w4_moe_ep_tuned.csv`，`tune.EP_TUNING`（EP8，64–32768 tokens，600 行）：≤256 tokens 选 mori，≥512 选 mega。不改 kernel；8 卡 check（含 `--graph`）、verify 通过；kernelkit 22 passed | 见 git log |

**I4 的经验**：

- 自动改写 import 的工具误报过两次（元组赋值里定义的名字、从非 kernel 模块导入的同名常量）。改写后要靠 ISA 快照和 pyflakes 兜底，不能只看工具输出。
- 实验脚本（`experiments/`）也按名字从 kernel 文件导入 helper，搬 helper 时要一起改。
- GPU 偶尔被别的容器占用，ISA 快照报 “GPUs busy”。用每 5 秒重试一次的包装脚本（`/tmp/i4/snap.sh`）等空闲时再跑。
- “两个函数要不要合并”这类问题，先实测端到端行为再让用户决定：NaN 传播看的是 amax，不是 pack 函数。

**I5 的经验**：

- 迁移时要找出所有把 launcher 存起来再调用的地方（megamoe 的 `self.launch`、bench / 实验脚本里替换它的代码）。存的东西从 JitFunction 变成 specialization 后，替换方也要改成 `.get(...)`。
- 删 kernel 参数即使只改 kernarg 布局，也可能让编译器合并标量 load 并重排调度。ISA 差异要按指令助记符逐对比较，不能只看 offset；会重排的 kernel 要做性能 A/B。
- A/B 新旧代码：把旧提交 `git archive` 成另一个包名（`/tmp/i5/oldpkg/a8w4_old`），用 `bench ab` 的 `py:` variant 跑它，不用另写计时脚本。

**K 的经验**：

- “换实现、不换行为”的重构，先把冻结函数在整个输入网格上的输出存成 golden JSON，改完逐项比较，比挑几个例子可靠。
- TP / EP kernel 都用 atomic，同一输入两次运行不逐位相同（TP 的 cos 差到 2e-5，EP 到 7e-7）。比较新旧输出前先量出“旧对旧”的噪声，cos 用 float64 算：float32 的 `cosine_similarity` 在 16M 个元素上会给出错误的值，甚至负数。
- 生成的测试把原来的偶发失败暴露了出来（`test_cuda_graph` 比的其实是两个不同的 kernel）。偶发失败先查它比较的是不是同一个东西。
- 限制写在实现函数上以后，原来绕过检查的路径（bench 的覆盖项）也被检查到了；这类行为变化要在提交信息里单独列出。

**T1–T4 的经验**：

- 工具类重构的新旧对照：把改前的提交 `git archive` 到 `/tmp`，同一条命令在两张空闲卡上并排跑。用例的默认值有意改了（EP8 的 expert_offset 0 → 96）时，对照要用 `--shape` / `--set` 把它固定住，否则差异来自输入而不是代码。
- 分桶查表在没调过的点上会落到最近的点；“没有调优配置”只在 key（非分桶的词汇）不在表里时才出现。

## 5. 草图摘要：第 5–7 轮讨论（远期目标的素材，不是承诺）

- **公共工具层**：
  - 只放 host 端代码，kernel 不 import 它，所以不影响 FlyDSL 缓存 key 和 ISA；
  - 只在内部使用，不承诺 API 稳定。
- **Op / Impl 草图**：
  - `Impl` 提供：`name`、`Config`（frozen dataclass）、`graph_safe`、`candidates(p)`、`supports(p, cfg, capturing) -> 原因或 None`、`max_tokens`、`bind(p, cfg) -> 热路径函数`。
  - `Op` 提供：`Problem`、`cases`、`make_inputs`、`reference`、`check`、`fallbacks`、`default`。
  - 按问题和是否在 capture 预先算好 Plan，热路径目标 ≤ 1 µs。用穷举把 Plan 和旧的 `lookup_config` 对比，作为正确性证据。
  - 只做离线 autotune。
- **实测的 host 开销**：
  - `lookup_config` 每次 3.7 到 4.6 µs；
  - `a8w4_moe` 每次调用的 host 时间：强制指定配置约 75 µs，查表约 86 到 90 µs。主要花在约 10 次 launch、8 次 `torch.empty` 和 3 次 `current_stream` 上。
- **基线版本和新版本可以在同一个进程里加载**，做法是：
  - 用 `spec_from_file_location(别名, dir/__init__.py, submodule_search_locations=[dir])` 加载 worktree 里的目录；
  - 在 `exec_module` 之前把模块放进 `sys.modules[别名]`。

  原因是 a8w4 只用相对 import；FlyDSL 0.3.2 的缓存 key 是源码文本，源码相同的 kernel 两边共用缓存。
- **topk=8 的输出不确定**：同一版本跑两次，结果会差 1 个 bf16 ULP（原子加的顺序不同）。所以逐位比较只适用于 topk=1。
- **运行时可以取到 ISA 指纹**：
  - `CompiledArtifact._ir_text` 里嵌着 HSACO；
  - 处理流程：反转义 → `llvm-objdump` → 规范化 → sha256；
  - 一个配置的 5 个 kernel 约 0.18 秒，缓存命中时也能用。
- **调优和计时的事实**：
  - FlyDSL 缓存是热的时候，一个 tune 任务组约 1 分钟，全部 5 组约 3 分钟；新配置每个要编译 5 到 13 秒；
  - 大 batch 下不同 run 之间相差 ±5% 到 8%，所以只能在同一个 run 里交错比较；
  - `tables.compare` 用“旧赢家在新表里的耗时”来算提升，但原地修改 kernel 时看不出差异；
  - `time_cold` 用 GPU sleep 排除 host 开销。
- **“一条命令尝试优化思路”的草案**：`a8w4 tune try NAME -m 说明`。
  - 选定的 job 跑完整 tune，并把 `@base`（基线 rev 的公开 API）和 `@base#2`（噪声参照）作为额外候选交错计时。
  - 每个点和 `@base` 比，判为 win、持平或 loss；BAD/FAIL 一票否决。
  - 结果被接受时：只替换 win 点在调优表里的行，在 `perf_log.md` 加一行，再写报告目录和 `experiments/index.csv`。
  - 退出码：0 接受，1 拒绝，2 精度失败。
