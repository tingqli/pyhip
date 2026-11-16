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

- **迭代**：I1、I1b、I2、I3 已完成。I3 的范围是用户看过 I2 后改的（见 3.1 节）：`_vendor` 只留真正用到的代码，迁到 `kernels/` 下的共享模块，删掉 `_vendor/`。现在等用户 review I3，下一个是 I4。
- **分支**：`/sgl-workspace/pyhip` 的 `refactor/a8w4`。
- **已有提交**：
  - 迭代 0，安全网：`64ce77c`、`d17f502`（ISA 基线在这个提交上生成）、`4e256d9`。
  - 设计文档：`1383e41`、`0c50a56`、`ab05750`、`6e2c2ac`（本文件）。
  - I1：`0b0e3c9`、`725209f`、`51817b1`、`f9d75f3`、`cb2e8f0`（收尾：文档和实验脚本）。
  - I1b：`aa91396`、`2b2efaa`、`4f478b4`、`8399c9d`（文档）、`bb7f40e`。
  - I2：`892d7f8`（`TL`、`DBG`）、`87fd7c8`（a16w4）、`7dbb85e`（实验脚本），`aac9bcf`（文档）。
  - I3：`18f243f`（原地精简）、`cc44955`（迁到 `kernels/`，删除 `_vendor/`），以及文档提交。
- **用户的文件**：未跟踪的 `refactory2.md` 属于用户，不要改，也不要提交。

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
- **中途不重新调优**：哪个改动需要重新调优，就说明它不是纯结构性的（Q6）。
- **冻结的公开 API**：7 个函数和 `ref`（Q8、Q17），由 `tests/test_a8w4_api.py` 守护，不用 GPU，约 3 秒跑完。
- **用 GPU 前**：先看 `rocm-smi --showuse`，再拿 `flock /tmp/gpu-N.lock`。卡被占用时停下来问用户（Q29）。

## 1. Backlog（按确定程度分三档）

### 1.1 近期：确定、可以立刻做（按顺序）

| # | 内容 | 来源 | 验收 |
|---|---|---|---|
| I1 | 删除 TP kernel（`tp_gemm1`、`tp_gemm2`、`tp_fmoe`、`tp_fmoe_q`）里走不到的选项和冻结常量 | Q4，Phase 1 | ISA IDENTICAL；API 测试；单卡 pytest |
| I1b | 删除 TP kernel 里调用方不用的模式取值（fast_dec 0/1、gemm1 的 bf16 epilogue 等，规则见 3.3 节） | 用户在 I1 review 时决定 | 同上 |
| I2 | 删除 EP kernel 里走不到的选项（`TL` 等）和 a16w4 路径，包括 bench 的 `adt=bf16` 覆盖 | Q4，Q11 | 同上 |
| I3 | 精简 `_vendor`：没人读的环境变量、类、函数、参数直接删；剩下的迁到 `kernels/` 下按主题分的共享模块；删除 `_vendor/`（规则见 3.1 节） | 原 I3 + 原 I4 前半；用户 2026-11-16 | ISA IDENTICAL；API 测试；单卡 pytest |
| I4 | kernel 文件之间不再互相 import（共享的 device helper 搬进 I3 建的模块，按名字导入；现在的 `G1.x`、`EP.x` 这类模块属性调用不进 FlyDSL 缓存 key）；`_fp4x8_to_fp8x8`、`_fp8x4_pack` / `_pack_activation` 只留一份；硬件常量集中定义 | Q12，Q14，Phase 2 | ISA IDENTICAL |
| I5 | host 封装改用 `@flydsl_kernel`，替代 `_run_compiled` 和 `functools.cache`；每个 kernel 文件一个公开入口；同时删掉 I1 留下的无用 kernel 参数（标了 `# unused` 的那些）。到时按文件再拆成几个迭代 | Q13，Phase 3 | ISA IDENTICAL（kernarg 偏移变化要逐个解释）；pytest |
| I6 | 新建 `workspace.py`，统一管理每张卡一份的缓冲区 | Q19 | ISA IDENTICAL；pytest |

### 1.2 中期：方向清楚，细节做到时再定

- kernel 内部按术语表改名，长函数拆成按阶段命名的函数（Q20、Q21）。可以在碰到某个文件时顺手做。
- `table.py`：把调优表 CSV 的读、写、合并收拢到一处，供 dispatch 和 tune 共用。配置字符串的 parse / format 加往返测试（Q10）。
- 族模块加注册表，目标是“加一个族只改一处”（Q1a、Q9、Q18）。接口形状可以参考第 5 节的 Op/Impl 草图，但只按 a8w4 当时的实际需要来定。
- 统一命令行入口，去掉 `sys.path` hack（Q7）；工具搬进 `dev/`（Q16、Q25）。
- 测试整理（Q22）、文档拆分（Q24）。
- megamoe 套用同样的分层（Q15）。

### 1.3 远期：朦胧目标，不提前计划

- 与具体 kernel 无关的公共工具层，比如 `pyhip.kit`，包含 isa、timing、tune、bench 等（refactory2 §1）。
- 把“功能”和“实现”分开：共享测试、预先算好 dispatch 计划、离线 autotune（refactory2 §2）。
- 一条命令尝试一个优化思路（refactory2 §3）：
  - 用 tune 和基线比较，先过精度门；
  - 结果好就更新调优表里相关的行，并在 `perf_log.md` 里加一行。
- 降低 host 开销：现在每次调用约 75 µs。
- 已知问题，只记录不处理：`ep3k l128_v1` 配合 fp8 每 128 列一个 scale 的输入时 LDS 超限，`supported()` 没有拒绝这种组合。

## 2. 删除规则（I1–I3 共用）

- 一个选项只有在 `a8w4_moe`、`a8w4_moe_dyn`、`MegaMoeEpA8W4`、`tune`、`bench` 都走不到时才能删。
- bench 的 variant 覆盖只作用于 ep3k：`g1.*` 和 `g2.*` 会作为 `ep_gemm1` / `ep_gemm2` 的参数传入，此外还有 `bm`、`rowv`、`sched`、`sst`（见 `bench/core.py` 约 160 到 260 行；`adt` 覆盖已在 I2 删掉）。TP kernel 的选项只能经 `moe.py`、`tune`、megamoe 走到。
- 删完后，剩下的 kernel 的 ISA 必须完全不变。不在快照矩阵里的配置，本来就走不到。

## 3. 当前迭代

### 3.1 I3：精简 `_vendor`，迁到 `kernels/` 的共享模块（已完成，等 review）

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
- 引入 `host_launch` 后，导入 a8w4 会加载 pyhip 的这个模块（多约 0.03 s）。`kernels/_launch.py` 是包内唯一的 pyhip 绝对导入，README 的“复制到其他工程”已写明。

### 3.2 I2：删除 EP kernel 的调试开关和 a16w4 路径（已完成）

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

### 3.3 I1b：删除 TP kernel 里调用方不用的模式取值（已完成）

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

### 3.4 I1：删除 TP kernel 里走不到的代码（已完成）

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

当时列出的 I1b 候选（`fast_dec=0/1`、`flush_tiny`、`A_GROUP=0`、`ROW_VGPR=False`、`dyn_rows=False`、gemm1 的 `epilog="bf16"` 和 `pipeline`），用户决定删，已在 I1b 完成（见 3.3 节）。`dyn_m=False` 和 `xcd_swizzle` 按规则保留。

## 4. 迭代记录

| 日期 | 迭代 | 结果 | 提交 |
|---|---|---|---|
| 2026-11-15 | 0 安全网 | `prof isa-snapshot` / `isa-diff` / `isa-cover`；API 契约测试和 CSV 往返测试；ISA 和 pytest 基线 | `64ce77c` `d17f502` `4e256d9` |
| 2026-11-16 | I1 删除 TP kernel 的死代码 | 4 个文件净减约 830 行；ISA IDENTICAL；API 测试 23 passed；单卡 pytest 94 passed、1 skipped | `0b0e3c9` `725209f` `51817b1` `f9d75f3` |
| 2026-11-16 | I1b 删除 TP kernel 里不用的模式取值 | `tp_gemm1` 1044→915 行，`tp_gemm2` 1225→1034 行，`tp_fmoe` 去掉两个开关；kernel 名字不变；ISA IDENTICAL；API 测试 23 passed；单卡 pytest 94 passed、1 skipped | `aa91396` `2b2efaa` `4f478b4` `8399c9d` |
| 2026-11-16 | I2 删除 EP 的调试开关和 a16w4 路径 | `ep_gemm` 1707→1545 行，`ep_small` 609→592 行；`_make_core` 去掉 `FP8`；bench 去掉 `adt=` 覆盖；kernel 名字不变；ISA IDENTICAL；API 测试 23 passed；单卡 pytest 94 passed、1 skipped | `892d7f8` `87fd7c8` `7dbb85e` |
| 2026-11-16 | I3 精简 `_vendor`，迁到 `kernels/` | `_vendor/` 9 个文件 2268 行 → `kernels/` 5 个共享模块 464 行；helper 按名字导入，大部分进了 FlyDSL 缓存 key；`_run_compiled` 改用 pyhip `host_launch`；删 25 个无用 import；ISA IDENTICAL；`channel_exponents` ISA 和输出逐字节相同；API 测试 23 passed；单卡 pytest 94 passed、1 skipped | `18f243f` `cc44955` |

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
