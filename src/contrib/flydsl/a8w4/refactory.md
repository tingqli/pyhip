# a8w4 重构：设计与计划

> **2026-11-16 起本文件降级为思维草图**：执行依据改为 `agile_refactor.md`（按确定程度分迭代推进）。本文件保留 4 轮讨论的决定和事实，供迭代时参考，不再作为必须遵守的计划；下面的“唯一依据”“阶段计划”等表述以 `agile_refactor.md` 为准。

本文件记录 4 轮讨论定下的 32 个决定和阶段计划。
全部完成后，长期有用的内容并入 `docs/design.md` 和 `docs/dev.md`，然后删除本文件（Q31）。

分支：从 `tq/moe-a8w4` 拉出的 `refactor/a8w4`（Q28）。

## 0. 续接：新 session 从这里开始

本节写给从空 context 接手的人：读完本节和第 6 节（阶段计划）就能继续。第 1 到 5 节是已经定下的设计，不要重新讨论，除非用户提出。

### 0.1 当前状态（每次停下来前更新）

- **阶段**：Phase 0 已完成，**正在等用户 review Phase 0 的提交**。用户明确说了“先等我 review 提交”，没有得到用户同意前不要开始 Phase 1。
- **分支和提交**（`/sgl-workspace/pyhip`，分支 `refactor/a8w4`）：
  - `1383e41` 设计和计划（本文件）
  - `64ce77c` `prof isa-snapshot` / `isa-diff`
  - `d17f502` `tests/test_a8w4_api.py`（**ISA 基线就是在这个提交上生成的**）
  - `4e256d9` `prof isa-cover`
  - `0c50a56` 本文件：Phase 0 打勾、基线结果
  - 之后的提交：本节（续接说明）
- **待用户决定**：
  - `ep3k l128_v1` 配合 fp8 每 128 列一个 scale 的输入时 LDS 超限（见第 6 节 Phase 0 基线）。我的建议是 Phase 1 不改，只记录；用户还没回复。
- **工作区**：`/sgl-workspace/pyhip` 根目录下未跟踪的 `0.txt`、`1.txt` 是用户的文件，不要动、不要提交。

### 0.2 工作方式（用户已确认）

- 和用户用中文交流；需要用户决定的事用 `ask_user` 工具问，一次问一个问题，给出选项和推荐项。
- 每个小步一个提交，提交信息写 ISA 对比结论，结尾加 `Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>`。
- **每个阶段结束后停下来，等用户 review**，用户同意后才进入下一阶段。
- 本文件是计划的唯一依据：完成一项就打勾，并更新 0.1；停下来之前也要更新 0.1。
- 设计里没写到的问题，先问用户，不要自己决定。
- 不要为了“差不多”的去重接受 ISA 变化（Q27）；不要中途重新调优（Q6）。

### 0.3 环境

- Python：`/opt/venv/bin/python`（3.10）。pyhip 是 editable 安装，`pyhip` 包映射到 `/sgl-workspace/pyhip/src`，所以 a8w4 的导入路径是 `pyhip.contrib.flydsl.a8w4`，对应目录 `src/contrib/flydsl/a8w4`。
- FlyDSL 0.3.2：`/opt/venv/lib/python3.10/site-packages/flydsl`。ISA 快照工具 hook 了它的内部函数（`flydsl.compiler.jit_function` 里的 `_dump_ir`、`_dump_isa`、`_extract_llvm_ir`、`_infer_kernel_names_from_asm`），换 FlyDSL 版本后要重新检查。
- 8 张 MI300X（gfx942），224 核。磁盘 `/` 已用 99%，只剩约 15 GB；一次快照约 40 MB。
- GPU 规矩（Q29）：
  - 用前先看 `rocm-smi --showuse`，再拿 `/tmp/gpu-N.lock`（`flock -n`）。
  - 卡被占用时停下来问用户。
  - `isa-snapshot` 自己会检查占用、加锁。
- sglang 里调用 a8w4 的地方：
  - `/sgl-workspace/sglang/python/sglang/srt/layers/quantization/mxfp4_aiter_a16w4_moe.py`
  - `/sgl-workspace/sglang/test/registered/unit/layers/quantization/test_mxfp4_aiter_a16w4_moe.py`
- 可参考的 skill（`/sgl-workspace/pyhip/.claude/skills/`）：
  - `flydsl-gfx942-pitfalls`：缓存 key 规则、LDS 64 KB 等；
  - `flydsl-host-launch`：`@flydsl_kernel` 的写法，Phase 3 要用；
  - `a8w4-perf-experiment`、`gpu-timing-protocol`：收尾时的性能核对要用。

### 0.4 安全网用法

每一步的验证顺序：ISA 快照 → 对比 → 必要时跑 pytest。

```bash
cd /tmp   # 在任何目录都可以，用包路径运行
P=pyhip.contrib.flydsl.a8w4.prof
python -m $P isa-snapshot --out /tmp/a8w4_isa/new --gpus 0-7 --workers-per-gpu 4   # 约 105 秒
python -m $P isa-diff /sgl-workspace/a8w4_refactor_state/isa_base /tmp/a8w4_isa/new   # 退出码 0 = 一致
python -m $P isa-snapshot --list          # 列出 26 个 job
python -m $P isa-snapshot --out DIR --jobs 'moe:tp8' --gpus 7   # 只跑部分 job（正则）
python -m $P isa-cover BASE -- CMD...     # CMD 编译的 kernel 是否都在快照里
```

- **基线位置**：
  - `/sgl-workspace/a8w4_refactor_state/isa_base`（持久）；
  - `/tmp/a8w4_isa/base`（同一份内容，机器重启后可能丢失）。
  - 同一目录下还有基线 pytest 日志 `pytest_moe.log`、`pytest_gpu8.log`。
- **基线丢了怎么重新生成**：在 `d17f502` 上开 worktree，用 `PYTHONPATH` 让 `pyhip` 指向它的 `src`。PYTHONPATH 优先于 editable finder，这个方法已验证可行；收尾的性能核对（Q30）也用同样的办法。

  ```bash
  git -C /sgl-workspace/pyhip worktree add --detach /tmp/a8w4_base_wt d17f502
  mkdir -p /tmp/a8w4_base_path && ln -sfn /tmp/a8w4_base_wt/src /tmp/a8w4_base_path/pyhip
  cd /tmp && PYTHONPATH=/tmp/a8w4_base_path python -m pyhip.contrib.flydsl.a8w4.prof \
      isa-snapshot --out /sgl-workspace/a8w4_refactor_state/isa_base
  ```

- **`isa-diff` 的输出怎么读**：
  - `ISA IDENTICAL`：两次快照的规范化 ISA 哈希集合相同，结构性步骤的验收就是这个。kernel 名不同没关系，比较时已经忽略。
  - `changed ... code`：指令变了，不合格。
  - `changed ... metadata only`：指令相同，只有元数据变了（比如 kernarg 布局）。Phase 3 改用 `@flydsl_kernel` 时可能出现，要逐个解释原因后提交。
  - `errors only in A/B`：某个矩阵项开始报错或不再报错。
  - 已有的 4 个错误（`l128_v1` + fp8g128）两边都会出现，属于正常。
- **矩阵是从当前代码生成的**：job 和配置来自 `tune.CASES`、`tune.candidates()`、`moe.supported()`、`EG.CFGS`。Phase 1 删掉的选项本来就不在矩阵里。Phase 5、6 移动 `tune` 和 `moe` 后，`isa_snapshot.py` 要同步改成用新模块，但**矩阵内容必须不变**（job 名、配置、token 列表都一样）。改完先对基线跑一次 diff，确认还是 `ISA IDENTICAL`。
- **pytest**：
  - 现有测试仍然用 `sys.path` hack 和 `from a8w4 import`，要在 `src/contrib/flydsl` 目录下运行：
    - `python -m pytest -q a8w4/tests/test_a8w4_moe.py`：单卡，约 5 分钟；
    - `python -m pytest -q a8w4/tests/test_a8w4_mori_ep8.py a8w4/tests/test_a8w4_mori_tp8.py a8w4/tests/test_a8w4_megamoe_ep8.py`：8 卡，约 4 分钟，只在阶段末跑。
  - `tests/test_a8w4_api.py` 用包路径导入，在任何目录都能跑，不需要 GPU，约 3 秒。

### 0.5 下一步：Phase 1（删除走不到的代码）的准备

动手前的原则：

- 一个选项只有在 `a8w4_moe`、`a8w4_moe_dyn`、`MegaMoeEpA8W4`、`tune`、`bench` 都走不到时才能删。
- `bench` 的 variant 覆盖写法要专门检查，在 `bench/core.py` 约 160 到 260 行。ep3k 的 knob 要继续能从 bench 覆盖（Q18）。
- 删完以后，剩下的 kernel 的 ISA 必须不变。

已经查到的位置（行号对应 `0c50a56`）：

- `kernels/tp_gemm1.py`：
  - `epilog` 的 `"q8"` / `"q8push"`，以及 `tbl`、`rank`、`epoch`、`npes`。`moe.py` 只用到 `"q8s"` 和 `"bf16"`，要先确认 `"q8"` 是否被别处用到。
  - `w_layout`（`"tile"`）。
  - `PUSH_VAR = ()`（第 308 行，冻结）。
- `kernels/tp_gemm2.py`：
  - `TILE_ORDER`、`PROBE`、`AR`、`AR_SEP`：编译函数在 613 到 617 行，host 端在 1701 到 1705 行。
  - 冻结常量：`EPI_DRAIN`（58）、`BF16_FAST`（61）、`WPERM`（66）、`PERSIST`（69）、`AR_HOOK`（705）。
  - `moe.py:178` 用了 `epilog="direct64"`；`epilog` 默认是 `"lds"`。
- `kernels/tp_fmoe.py`：冻结常量 `ABL`、`LDS_PAD`、`DELAY`、`G1_SPREAD`、`PROBE`（81 到 90 行）。
- `kernels/tp_fmoe_q.py`：`DBG = ''`（67 行）。
- `kernels/ep_gemm.py`：`TL = 0`（276 行，以及 785、790、1134、1139 行）。
- a16w4（Q11）：
  - `ADT in ("fp8", "bf16")`：`ep_gemm.py` 751、1269 行，`ep_small.py` 140 行；
  - `ep_moe` 的 `adt` 参数（`ep_gemm.py` 约 1618 行）；
  - 各处 `ADT == "bf16"` 分支。
  - 注意：`ref.ref_moe` 的 `quant_x` 参数（False 表示不模拟 fp8 激活）属于冻结 API，sglang 的测试按位置传了这个参数，不能随 a16w4 一起删。
- `_vendor/tensor_shim.py` 30 到 40 行读了 3 个环境变量：
  - `AITER_FLYDSL_KERNARG_PRELOAD`、`AITER_FLYDSL_KERNARG_PRELOAD_COUNT`、`AITER_FLYDSL_MOE_EXPERT_SCHEDULING_MODE`；
  - 它们属于“冻结的环境开关”，要先查清楚影响了什么，再决定冻结成常量还是删除。
- 每删完一组就跑一次快照和对比，然后提交。

为后续阶段记下的事实：

- **探索 sub-agent 说 `dpp_utils`、`layout_utils` 没人用，这是错的**：`_vendor/mxfp4_gemm_common.py` import 了这两个模块。Phase 2 要按实际调用的函数来判断。
- kernel 文件之间的 import：
  - ep_gemm → ep_prep、tp_gemm1
  - ep_mega → ep_gemm、tp_gemm1
  - ep_mega_comm → ep_gemm、ep_mega、ep_prep
  - ep_small → ep_gemm、tp_gemm1、ep_prep
  - tp_fmoe → tp_gemm1
  - tp_fmoe_q → tp_fmoe、tp_gemm1
  - tp_prep → ep_prep（`NUM_CU`、`_latomic_add`）
  - weights → tp_gemm1.`prep_w1_fast`
- `_fp4x8_to_fp8x8(_fast)` 在 `tp_gemm1` 和 `tp_gemm2` 里各有一份。
- **编译 key 依赖 token 数**：
  - tp_prep 的 `EPT` / `HU` 来自 `_route_range(M*K)` / `_hist_unroll(M*K)`；
  - ep_prep 的 `MULTI = M*K > 2048`；
  - ep_gemm 的 `ZRC`（由 M、H、grid 决定）、`XG`、`DYN`；
  - ep1k 的 `maxm`。

  快照的 token 列表已经覆盖了这些切换点（`isa_snapshot.isa_tokens()`）。
- 共享 device helper 不能放进子目录（Q12，FlyDSL 0.3.2 缓存 key 的规则）。
- 这几个时间是实测的：单卡 pytest 约 303 秒；8 卡 pytest 约 244 秒；全量快照约 104 秒，CPU 时间约 29 分钟。

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
  - 实测全量矩阵约 105 秒（26 个 job，8 卡 × 每卡 4 个 worker），所以每一步都跑全量。
  - `prof isa-cover SNAPSHOT -- CMD`：检查一个命令（包括它 spawn 的子进程）编译的 kernel 是否都在快照里，用来证明矩阵覆盖了测试。
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
  - sglang 的用法还要求（Phase 0 查到，`tests/test_a8w4_api.py` 固定下来）：`supported` 接受 dict 配置（`dict(kind="tp2k", block_m=64, g1_bcm=1, g2="base")`）；`lookup_config(...)["kind"]` 可以取下标；`dyn_config` 对没有 EP kernel 的 shape 抛 `ValueError`；`ref.ref_moe` 按位置传 8 个参数。所以 Phase 5 引入有类型的 Config 后，公开 API 边界上仍然要收 dict、返回能按 `["kind"]` 取值的对象。
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

- [x] **0 安全网**：
  - [x] `prof isa-snapshot` / `prof isa-diff` / `prof isa-cover`（`isa_snapshot.py`）；
  - [x] 在 HEAD 上生成 ISA 基线；
  - [x] API 签名快照测试、CSV 往返测试（`tests/test_a8w4_api.py`，只跑 host 代码，23 个用例）；
  - [x] 现有单卡和 8 卡 pytest 的基线结果。

  基线（commit d17f502，FlyDSL 0.3.2）：
  - ISA 快照：558 次编译，177 个不同的 ISA，4 个错误，104 秒；连续两次快照完全相同。快照放在 `/tmp/a8w4_isa/base`。
  - 4 个错误是已有问题：`a8w4_moe_dyn` 用 `ep3k l128_v1` 且 x 是每 128 列一个 scale 的 fp8 时，gemm1 的 LDS 需要 82432 B，超过 64 KB，`compile_ep_gemm1` 里的 assert 报错（silu / situv2 × 容量 256 / 4096）。`dyn_config` 不会选 `l128_v1`，只有显式指定才会碰到，但 `supported()` 没有拒绝这种组合。
  - `tests/test_a8w4_moe.py`：94 passed、1 skipped，303 秒；它编译的 74 个 kernel 都在快照里。
  - 8 卡 `test_a8w4_mori_ep8.py`、`test_a8w4_mori_tp8.py`、`test_a8w4_megamoe_ep8.py`：3 passed，244 秒；它们编译的 31 个不同 kernel 都在快照里。
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
