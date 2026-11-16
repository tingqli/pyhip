# a8w4 重构：设计与计划

本文件是这次重构的执行依据，记录 4 轮讨论定下的 32 个决定和阶段计划，每完成一个阶段就打勾。
全部完成后，长期有用的内容并入 `docs/design.md` 和 `docs/dev.md`，然后删除本文件（Q31）。

分支：从 `tq/moe-a8w4` 拉出的 `refactor/a8w4`（Q28）。

## 1. 目标（Q1）

验收标准：

- **(a) 加一个新配置或 kernel 族只改一处**：新建一个族模块，再在注册表里加一行。现在要改 5 处：`kernels/`、`moe.RUNNERS`、`config_name/parse_config`、`supported()`、`tune.candidates()`。
- **(b) 分层清楚，依赖只能向下**：公开 API → dispatch（查表、能力过滤）→ 族插件 → kernel 入口（`@flydsl_kernel`）→ 共享 device helper。
- **(c) 包里不留走不到的代码**。
- **(d) 每个 kernel 文件只有一个公开的编译 / launch 入口**，参数写明、有类型。

## 2. 不变量与验证

- **数值和性能（Q2）**：
  - 纯结构性的步骤（搬文件、改名、删编译期走不到的分支、换 host 封装、拆阶段函数），要求生成的 ISA 逐字节相同，kernel 名字可以不同。
  - 确实会改变代码生成的步骤降一级：topk=1 时输出逐位一致，并且 `bench ab` 的差异在噪声内，提交信息里要写明。
  - 这次不为去重接受这一级的改动（Q27）。
- **ISA 对比工具（Q23）**：`prof isa-snapshot --out DIR` 和 `prof isa-diff A B`。
  - 用空的 FlyDSL 缓存编译一个固定矩阵：3 个调优 shape × 2 种激活 × `tune` 的全部候选；`dyn` 的变体（bf16、fp8 每 128 列一个 scale、fp8 每 token 一个 scale；prep 单 WG / 多 WG）；`row_quant`、`tp_prep`、`weights`；megamoe。
  - 规范化 final ISA（去掉 kernel 名和注释），每个 kernel 记一个哈希。
  - 每一步都跑全量矩阵，8 张卡并行；单次超过约 30 分钟就改成每步只跑改动涉及的子集，到里程碑再跑全量。
- **中途不重新调优（Q6）**：哪一步要求重新调优，就说明它违反了上面的约束。
- **最后的性能核对（Q30）**：用 git worktree 检出基线版本，和新代码一起用 `bench ab` 的 `py:` 变体交错计时。ep8、tp8、tp4 每个 case 测 8、1024、4096、32768 tokens 四个点，要求差异在噪声内；另外跑一遍 `tune verify`。
- **GPU 使用（Q29）**：开始前用 `rocm-smi --showuse` 确认空闲并拿住 `/tmp/gpu-N.lock`；单卡任务只用一张空闲卡；8 卡测试只在阶段末跑；卡被占用时先停下来问，不用 `--force`。

## 3. 范围（Q3）

- **要改的**：
  - 运行时：`moe.py`、`megamoe.py`、`weights.py`、`kernels/`、`_vendor/`；
  - 工具：`tune`、`tables`、`bench`、`prof`、`ref`、`exp`；
  - `tests/`；
  - 文档：README、`configs/README.md`、skill 里的命令。
- **`experiments/`** 冻结成历史档案，脚本不再保证能跑新代码。加一个 `experiments/README` 写明它对应的最后一个可运行 commit（Q25）。
- **sglang 不用改**：冻结的 API 和 `ref` 保持不变。

## 4. 决定

### 4.1 包身份、入口和公开 API

- **Q4**：包内代码就是正式版本，断开与 aiter `doc-mimo/kernels/` 实验原文件的逐位 / 逐行对应。`a8w4_moe`、`a8w4_moe_dyn`、`MegaMoeEpA8W4`、`tune`、`bench` 都走不到的选项一律删除，需要时从 git 历史找回。
- **Q5**：`pyhip.contrib.flydsl.a8w4` 是唯一导入路径，不再要求“整个目录能复制到别的工程”，可以直接依赖 pyhip 的公共模块。
- **Q7**：命令行只有一个入口 `python -m pyhip.contrib.flydsl.a8w4 <tune|bench|prof|exp>`，另注册 console script `a8w4`。删除在 `src/contrib/flydsl` 下作为顶层包运行的方式和 `sys.path` hack。
- **Q8**：以下 7 个函数在顶层保持名字和签名不变：`prepare_weights`、`shuffle_mxfp4`、`a8w4_moe`、`a8w4_moe_dyn`、`dyn_config`、`supported`、`lookup_config`。`config_name`、`parse_config`、`run_config`、`default_config`、`DEFAULT_CSV` 移到子模块，不留兼容别名。
- **Q17**：`a8w4.ref` 是对外稳定的数值参考（`ref_moe`、`act_ref`、`make_weights`、`make_routing`、`cos_diff`、`quant_fp8_groups`）。计时部分（`flush_cache`、`weight_copies`、`time_cold`、`WCOPIES`）移到 `dev/timing.py`。

### 4.2 kernel 族插件和配置

- **Q9**：每个族（`tp2k`、`tp1k`、`tp1kq`、`ep3k`、`ep1k`）是一个模块，提供：
  - frozen dataclass `Config`；
  - `parse` / `format`；
  - `supports(shape, ctx)`，`ctx` 包含 act、capturing、tokens，不支持时返回原因；
  - `candidates(shape)`；
  - `run(cfg, inputs, ctx)`；
  - `graph_safe`。

  所有族登记在注册表里。`dispatch`、`tune`、`bench` 只通过注册表访问各族，不再写 `if kind == ...`。
- **Q10**：族 ID 和配置字符串逐字节不变，CSV 不迁移。加测试：表里每一行都能 parse 再 format 回原字符串。“2k/1k”这类名字的含义只在 docstring 里解释。
- **Q18**：
  - `Ep3kConfig` 是有类型的 dataclass，`g1` / `g2` 是嵌套的 knob dataclass，缺省值全部显式写出。
  - `t16`、`s16`、`m64` 等是命名 preset，也就是这个 dataclass 的实例。
  - `ep3k m64:g2.pf=1` 本身就是合法的配置字符串；`format` 时如果和某个 preset 相同就输出 preset 名。
  - 这套覆盖写法通过插件协议统一提供，TP 族也能用；`bench` 的 variant 解析直接调用插件的 `parse`。

### 4.3 device 代码

- **Q11**：删除 a16w4（`adt="bf16"`）路径。以后要上线 a16w4，作为独立的族重新加回来。
- **Q12**：
  - 共享的 device 代码放在 `kernels/` 同一目录的私有模块 `kernels/_*.py` 里，按主题分：`_fp4`、`_sync`、`_pack`、`_sched`、`_mem` 等。
  - `_vendor/` 里用到的部分并进来，没用到的删掉。
  - kernel 文件之间不再互相 import。
  - 不建子目录：FlyDSL 0.3.2 的缓存 key 不包含其他目录里的 helper，放在子目录里，改了 helper 进程会拿到旧二进制。
- **Q13**：所有 host 封装改用 `pyhip.contrib.flydsl.host_launch` 的 `@flydsl_kernel`，包括 `weights`、`row_quant` 和 megamoe 的 5 个 kernel。删除 `_run_compiled` 和 `functools.cache` 编译函数。
- **Q14**：304 CU、8 XCD、64 KB LDS 集中定义成编译期常量，不改成运行时从设备读取。
- **Q19**：`workspace.py` 统一负责“每张卡一份、第一次调用时分配”的缓冲区（计数器、workspace、expert mask），取代 `ep_prep._WS`、`ep_small._BUF`、`tp_fmoe_q._WS`、`moe._MASKS`。仍然是每张卡一份，不是每个 stream 一份。
- **Q20**：
  - 删掉走不到的分支，抽出共享 helper；超过约 300 行的 kernel 函数体，在同一个文件里拆成按阶段命名的普通函数（trace 时内联，ISA 不变）。
  - 不把 TP 和 EP 两族的 gemm 合并。
- **Q21**：
  - 在 `docs/design.md` 放一张术语表。维度用 `H`、`I`、`E`、`topk`，其余编译 key 一律小写下划线（`act`、`dyn_rows`、`block_m`、`bcm`、`pf`……）。
  - 配置字段名和配置字符串里的 token 一一对应。
- **Q27**：“差不多”的重复代码（例如 `row_quant`、`ep_prep`、`tp_prep` 的行量化内循环），只有合并后每个调用方的 ISA 都不变时才合并；做不到就保留几份，并在注释里写明相似和差异。

### 4.4 megamoe（Q15）

- 作为正式模块：从 `__init__` 导出 `MegaMoeEpA8W4`，写进文档，套用同样的分层和 `@flydsl_kernel`，不再 import EP 族 kernel 文件里的私有函数。
- 用 host 端 epoch，不能放进 CUDA graph。这一点在文档里写成已知限制，这次不修。

### 4.5 测试（Q22）

- `tests/` 下只放 pytest。8 卡的用例加 `gpu8` marker，GPU 不够 8 张时自动跳过。
- `test_a8w4_mori_{ep8,tp8}.py` 里的性能命令行部分移到 `bench layer --case ep8|tp8`。
- 新增测试：
  - CSV 往返；
  - 插件契约：`candidates` 给出的配置都满足 `supports`，`parse(format(c)) == c`；
  - import 规则：kernel 文件之间不互相 import，运行时代码不 import `dev/`；
  - 公开 API 签名快照。

### 4.6 文档和工具

- **Q24**：
  - `README.md` 只写给使用者，约 250 行。其余拆到 `docs/design.md`（含分层图和术语表）、`docs/tuning.md`（原 §6 和 `configs/README.md` 合并）、`docs/perf.md`（原 §7，表格由 `tune readme` 生成）、`docs/dev.md`（原 §9）。
  - README §10 里 FlyDSL、内存模型、计时三类坑，以 skill `flydsl-gfx942-pitfalls` / `gpu-timing-protocol` 为准；MORI 和数值相关的坑放进 `design.md`。
  - `program.md` 并入 skill `a8w4-perf-experiment`；`explain.md` 已被 skill `explain-code` 覆盖，删除。
  - `*.explain/` 在重构完成后按新结构重新生成。
- **Q25**：`exp.py` 移到 `dev/`，作为 `exp` 子命令；FlyDSL 缓存 key 探针移到 `dev/`，作为 `prof cache-probe` 子命令。

### 4.7 默认约定

- 文档用中文，代码、注释和 docstring 用英文。
- 类型注解只加在公开 API 和插件协议上。
- 不新增 linter，沿用现有的 `# noqa` 风格。
- TP 族在专家切片上仍依赖 aiter 的 `moe_sorting`，不动。
- `pyproject.toml` 的子包列表和 package-data 跟随新结构更新。

## 5. 目标布局（Q16）

```
a8w4/
  __init__.py      冻结的 7 个函数 + MegaMoeEpA8W4
  __main__.py      命令行入口：tune / bench / prof / exp
  api.py           a8w4_moe / a8w4_moe_dyn（参数校验 → dispatch）
  dispatch.py      注册表、supported / lookup_config / dyn_config / default_config
  table.py         调优表 CSV 的读、写、合并（dispatch 和 tune 共用）
  workspace.py     每张卡一份的缓冲区
  families/        _base.py（插件协议）、_tp_prep.py、tp2k.py tp1k.py tp1kq.py ep3k.py ep1k.py
  megamoe.py  weights.py  ref.py
  kernels/         device 代码 + @flydsl_kernel 封装；_*.py 是共享 helper（同一目录，不建子目录）
  configs/         a8w4_moe_tuned.csv
  dev/             tune.py tables.py prof.py exp.py timing.py bench/
  tests/  docs/  experiments/（冻结）
```

依赖规则（由测试保证）：

- 运行时代码不 import `dev/`；
- `families/` 只调用 kernel 的公开入口；
- `kernels/*.py` 之间不互相 import，只能 import `kernels/_*.py`。

## 6. 阶段计划（Q26）

每个小步一个提交，提交信息里写 ISA 对比结论；每个阶段结束停下来等 review（Q28）。

- [ ] **0 安全网**：
  - [ ] `prof isa-snapshot` / `prof isa-diff`；
  - [ ] 在 HEAD 上生成 ISA 基线；
  - [ ] API 签名快照测试、CSV 往返测试；
  - [ ] 现有单卡和 8 卡 pytest 的基线结果。
- [ ] **1 删除走不到的代码**：
  - 实验选项：`q8push`/`q8pushs`、`w_layout="tile"`、`AR`/`AR_SEP`、`TILE_ORDER`、`PROBE`、`TL` 等；
  - 冻结的环境变量开关；
  - a16w4 路径。

  验证：剩下的 kernel ISA 不变。
- [ ] **2 device 共享库**：
  - `_vendor` 并入 `kernels/_*.py`；
  - kernel 文件之间不再互相 import；
  - 去掉重复的 `_fp4x8_to_fp8x8`；
  - 硬件常量集中；
  - 清理 import 样板。

  验证：ISA 不变。
- [ ] **3 kernel 入口**：改用 `@flydsl_kernel`，每个 kernel 文件一个公开入口；新建 `workspace.py`。验证：ISA 不变。
- [ ] **4 kernel 内部**：按术语表改名；长函数拆成阶段函数。验证：ISA 不变。
- [ ] **5 host 层**：`table.py`、`families/` 插件和有类型的配置、`dispatch.py`、`api.py`、`__init__`、megamoe 改用新入口。验证：ISA 不变，pytest 通过。
- [ ] **6 工具**：
  - `dev/`、`__main__` 和 console script；
  - bench 改用插件的 parse；
  - `bench layer`；
  - `exp` 和 `prof cache-probe`；
  - `pyproject` 子包列表。
- [ ] **7 测试**：`gpu8` marker、插件契约测试、import 规则测试。
- [ ] **8 文档**：README 拆分、`docs/*`、更新 skill 里的命令、`experiments/README`、重新生成 `*.explain/`。
- [ ] **收尾**：最后的性能核对（Q30），把本文件并入 `docs` 后删除。
