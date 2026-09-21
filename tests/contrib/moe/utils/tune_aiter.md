# Aiter MoE 调优与 E2E 测量

[tune_aiter.py](tune_aiter.py) 只负责一条串行流程：

1. 生成官方 CSV key，查询输出目录中的最新匹配配置。
2. 缺行或 `force=True` 时，用 `subprocess.run()` 直接调用官方 tuner。
3. 把 winner 追加到历史，将单行字典交给 `moe_driver.aiter(moe_config, tuned_config=...)`，取得 `(prepare, run)`。
4. 通过共用的 [moe_bench.py](moe_bench.py) 生成数据/参考并用GPU event测量完整调用。

不修改 Aiter 代码或优化 kernel，不维护独立 worker、自定义异常、文件/GPU 锁、随机临时目录或审计框架。
**由用户保证单例执行，并承担目录并发写入、GPU 竞争和中断风险。**
导入、帮助和 dry-run 只用标准库；真实调用在当前进程加载 Torch/Aiter 并初始化 GPU。

## 使用

在本目录运行：

```bash
python tune_aiter.py 2,4096,512,4,2,fp8_ptpc,silu,32
```

位置参数为 `TP,model_dim,inter_dim,experts,topk,quant_scheme_str,activation[,key=value...][,tokens...]`。
省略 tokens 时依次处理 1、2、4、……、65536，共17项；显式列表只处理指定项，按输入顺序串行执行。
也接受模型简称，如 `qwen35_397B_k256,8192`、`xiaomi,preshuffle=True,8192,16384`；简称区分大小写。
`--help`列出全部简称及完整config，来源与映射见[模型预设](moe_driver.md#driver-优化与对比工具)。
可选kwargs只放在activation和tokens之间，支持末尾一个逗号；键和值的规则见[共享config格式](moe_driver.md#driver-优化与对比工具)。

```bash
python tune_aiter.py 8,4096,2048,512,10,fp8_blockscale,silu,1,4,8 \
  --output-dir ./tuned_aiter --force
```

```bash
python tune_aiter.py "2,4096,512,4,2,fp8_ptpc,silu,preshuffle=True,32," \
  --dry-run
```

本入口始终使用官方tuned CSV；其key不记录自定义gate布局/clamp/SiTU常量或非标准weight layout。
因此显式参数必须符合下文标准组合，否则在调优或文件写入前报错，不忽略参数、不把不同数值问题混进同一条历史。
自定义参数实验使用 [moe_bench.py](moe_bench.py) 与支持该组合的driver。新driver固定stage2路由加权、设备原生FP8；gate/up物理布局由候选处理。

Python API：

```python
import tune_aiter

result = tune_aiter.tune(
    TP=2, model_dim=4096, inter_dim=512, experts=4, topk=2, tokens=32,
    quant_scheme_str="fp8_ptpc", activation="silu",
    driver_kwargs={"preshuffle": True},
)
print(result["tuned_row"], result["e2e_latency_s"], result["diff"])
```

从父目录可 `from utils import tune_aiter`，或 `python -m utils.tune_aiter ...`。
`inter_dim` 是 **TP 切分前**的宽度，权重与 CSV 使用 `inter_dim // TP`；不执行通信、隐式切片或 padding。

`ensure_tuned(spec, root, output, device, info, ...)`只获取/搜索配置，不生成输入、参考或计时；
供cross_compare使用，避免缺配置时先测一次、进入比较后再测一次。`tune()`仍组合配置获取和E2E，CLI不变。

## 固定输出目录

API 和 CLI 的 `output_dir` 默认均为 **`./tuned_aiter`**，相对于调用时的当前目录。
API 显式传 `None` 会报错，不再创建临时目录，也不设置/维护 `TMPDIR`。

```text
tuned_aiter/
  untuned.csv             # 新调优请求的追加历史
  tuned.csv               # 官方有效 winner 的追加历史
  work/
    untuned.csv           # 本次单 shape 输入
    tuned.csv             # 本次单行 winner，helper 读取后显式传给 driver
    profile.csv           # 官方 -o2 候选组合结果
  tune.log                # 最近一次真正调优的 stdout/stderr
  bench.log               # 当前测量的 Python stdout
  result.json             # 最近一次成功结果；实际调用开始时删除旧结果
  jit/                    # 未设置相应环境变量时使用的持久缓存
  flydsl-cache/
  flydsl-autotune/
  triton-cache/
  inductor-cache/
```

- 不生成 `aiter-moe-*`，不扫描旧工作区或兄弟目录。旧 CSV 若只在旧目录中，直接把 `output_dir` 指向该目录，或自行迁移到根目录。
- 不创建锁文件、不防止同时运行、不拦截符号链接、不做 append 回滚/fsync 事务。不要把输出目录指向不希望修改的位置。
- 缓存位置优先沿用已有 `AITER_JIT_DIR` / FlyDSL / Triton / Inductor 环境变量，否则使用上图中的固定目录；`force` 不清理编译缓存。
- E2E 使用当前进程已有的后端缓存，不在已导入模块后强制切换路径。需要指定缓存位置时，在启动 Python 前设置环境。
- 不再保留 request/plan/source/failure/dispatch 审计 JSON。失败直接抛标准异常，运行失败使用 `RuntimeError`，文件错误保持 `OSError` 等原类型。
- 缓存命中不改写 `tune.log` 或上一次调优输入/profile；以 `tuning_skipped` 判断本次是否执行调优。

### 缓存和失败行为

完整 key 包括 GPU arch/CU、token 查表桶、H/local I/E/topk、激活、A/W dtype、量化方式和 route-weight 阶段。
历史有多条同 key 时取最后一条无 tag 的有效记录。无效最新行明确报错，不静默重调。

| 情况 | 行为 |
|---|---|
| 命中 | 不启动 tuner，每次重新测量 E2E |
| 缺行 | 追加 untuned，调用官方脚本，检查结果后追加 tuned |
| `force=True` | 不读缓存，重调并追加，旧行保留 |
| 调优失败或没有有效 winner | 不追加成功配置 |
| E2E 失败 | 已追加 winner 保留，下次可只重试测量 |

调优依据 token 桶，测量依据实际 tokens。例如17与31共享 key32；32768与65536共享 key32768。
复用 kernel 不复用旧 latency/diff。官方 CSV 的 `us/us1/us2` 是微秒阶段指标，`err1/err2` 不是本工具的 `calc_diff`。

## 参数

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `output_dir` / `--output-dir` | `./tuned_aiter` | 固定输出目录，不接受 None |
| `device` / `--device` | 当前已初始化设备，否则0 | 继承可见设备中的编号 |
| `aiter_root` / `--aiter-root` | 已安装的 checkout | 官方脚本所在目录；当前进程须能导入同一份 Aiter |
| `driver_kwargs` / config中的 `key=value` | 无显式覆盖 | 仅preshuffle、swiglu_limit、beta、linear_beta、output_dtype |
| `force` / `--force` | False | 强制重调 |
| `seed` / `--seed` | 43 | 调优路由与 E2E 输入 seed |
| `warmup` / `--warmup` | 5 | 每组 buffer 预热次数 |
| `iters` / `--iters` | 20 | 每轮计时的完整调用次数 |
| `rounds` / `--rounds` | 5 | 多轮取中位数 |
| `buffer_count` / `--buffer-count` | 10 | 独立地址的输入/权重/路由/输出轮换组数 |
| `diff_threshold` / `--diff-threshold` | 0.02 | E2E 精度门槛 |
| `timeout_s` / `--timeout-s` | 120 | 传给官方 `--timeout` 的**单任务 watchdog** |
| `tune_backends` / `--tune-backends` | 全部 | 限制新搜索来源，例如 `cktile` |
| `kernel_regex` / `--kernel-regex` | 不过滤 | 限制新搜索 kernel |
| `--dry-run` | False | 只打印规范化规格，不查GPU、不生成CSV/文件 |

参数精简：

- 新config/CLI不接受 `doweight_stage1`、`fp8_dtype`、`gate_mode`、`max_tokens` 或旧拼写 `preshuffled`；gate_mode只在相关工厂中作为packing超参，不作旧CLI兼容转换。
- 不为每个driver参数增加独立CLI flag；通过config kwargs或Python `driver_kwargs`传入，官方tuner路径仍只允许下表标准组合。
- 删除第二套 `tune_timeout_s` / `--tune-timeout-s`；保留的 `timeout_s` 不再是外层进程 wall-clock 限时，不覆盖编译、全程或 E2E。
- 删除 dry-run 的 `--gfx` / `--cu-num`。真实运行查询 GPU；dry-run 预览规格而不是伪造设备相关的 CSV。
- 搜索过滤、轮换/计时参数有实际用途，保留。它们不会改变命中的历史行；要改变搜索范围请使用 `force`。

没有外层 watchdog 或专门的子进程组清理器；需要强制限时/进程清理时由运行脚本的外层负责。
旧版不一致的 API/CLI 超时默认值已统一。CLI 参数错误退出2，运行/文件失败退出1，中断退出130。

## 数值与计时

| 量化 | 激活 | 自动选择 |
|---|---|---|
| BF16/no-quant、三种FP8 | silu / gelu / gelu_tanh | separated gate/up |
| A16W4 | situv2 | separated，beta=4、linear_beta=25 |
| A8W4 | swiglu / situv2 | interleave；SwiGLU clamp=7 或 SiTUv2 4/25 |
| A4W4 | silu / swiglu / situv2 | separated；同上固定激活参数 |

MX要求gfx950、H/local I为256的倍数；FP8 blockscale为128，其它为32。
FP8 PTPC、per-tensor、blockscale对应 `per_Token`、`per_Tensor`、`per_1x128`。
gfx942使用E4M3FNUZ，gfx950使用E4M3FN。
INT8不接入：虽然参考量化已复用 Aiter，当前 driver/tuner 尚未接入对应完整 MoE 路径。

输入和源权重为BF16，路由为logits topk+softmax；数据生成与比较脚本共用，权重seed与激活/路由seed分开。
`prepare`返回普通权重字典，每driver只量化/shuffle一次。外部仅准备一份私有输入，嵌套权重展开后由 `pyhip.run_perftest` 复制为 `buffer_count` 份并轮换，保留 `is_shuffled`。标准两阶段参考显式舍入BF16中间值；融合MX输出先不额外舍入，仍执行A2量化。

协议 **`eager_moe_driver_v3`**：静态W量化/shuffle、参考、输入复制、首次编译/dispatch检查和预热不计时。
计时包含sorting、动态A1/A2量化、两阶段GEMM/激活/归约等完整调用，不再逐次验证权重归属或录制Aiter dispatch。
每轮 `run_perftest` 执行 `warmup` 次预热和 `iters` 次逐调用event计时/同步，返回平均微秒，换算成秒后多轮取中位数；首次调用及每轮所有使用过的输出均检查精度与非有限结果。
v3不再用一对event包住整轮，不能直接把新旧数字当作kernel加速比。
这是 eager 接口延迟，会包含 host 提交造成的设备空闲，不等于kernel时长之和或CUDA Graph。

不做显存预估/空闲GPU检查；用户按模型选择 `buffer_count` 并承担OOM/资源竞争风险，不自动缩减参数。
测量前后释放PyTorch空闲allocator cache，便于下一次子进程使用显存；不释放调用者仍持有的tensor或编译缓存。
不改GPU时钟。Aiter配置为进程级串行切换，不再承诺环境和缓存隔离；不要与绕过driver的调用交错或并发运行。

返回核心字段为 `tuned_row`、`csv_header`、`tuned_config`、`e2e_latency_s`、`samples_s`、`diff`、
`dispatch_verified`、`tuning_skipped`、`spec`、`device`、`output_dir`、两份历史CSV路径及搜索过滤信息。
删除 `artifact_dir/artifact_reused/cache_dirs`、source/hash、重复selection/search标签等管理字段。
只返回官方catalog winner的实测结果，不保证它是所有候选中E2E最快的组合。

## 已知上游副作用

官方脚本除了 `-o2` 指定的候选组合输出，还会把原始stage profile合并写回
`AITER_ROOT_DIR/aiter/configs/profile_fmoe.csv`。**`-i/-o/-o2` 无法重定向这处输出。**
因此即使输出目录在源码之外，仍可能改动该文件，并要求上游目录可写。

这里接受官方行为，不修改Aiter、不monkey-patch、不备份或恢复profile。
不同GPU/输出目录也可能写到同一文件，用户负责串行执行；缓存命中不启动tuner，不产生这项调优副作用。
上游部分JIT命令不引用路径，源码/缓存目录应避免空格和shell特殊字符；本工具不再额外验证路径。

当前gfx950 catalog的FP8 stage2没有可用的K64候选，因此Hy3的local I192会得到`no valid candidate found`；helper会在错误中说明无有效kernel组合，仍保留完整日志，不放宽精度门槛、不伪造winner、不自动padding为I256。
这属于当前上游kernel覆盖限制；[cross_compare.py](../cross_compare.py)会保留Aiter失败，但继续使用独立参考测试本地候选（无Aiter相对加速比）。

## 建议验证目的（不维护专用测试文件）

原13个用例的目的保留为以下建议，按需要手工或由外层测试；不是本次已执行的完整验证声明：

1. **调优→命中→force→新shape**：历史只追加，命中仍测实际tokens，新结果与CSV阶段时间不混淆，编译缓存不被清空。
2. **完整key/最新行**：排除其他GPU、量化和tagged行，保留CSV内引号与kernel名称。
3. **官方调优失败**：不发布新的成功配置，不把旧结果文件当成本次成功。
4. **退出0但输出为空**：必须报错，不能继续使用陈旧winner。
5. **E2E失败重试**：调优winner保留，修复后可以只重测而不再次搜索。
6. **无效历史行**：明确失败，不静默重调或返回假指标。
7. **子进程边界**：只使用官方现有参数，传递所选设备/搜索范围，非零退出可定位日志。
8. **超时语义**：核实官方 `--timeout` 收到正确值；原外层进程组终止用例已随该机制删除，不再作为工具承诺。
9. **CPU-only入口**：import/help/dry-run不初始化GPU、不创建输出；省略tokens使用17项默认列表。
10. **CLI批量行为**：显式列表保序，参数有效后才运行，某项失败不继续后续项；全部使用同一个固定输出目录。
11. **driver测量**：准备不计时、计实际batch、参考使用相同输入，event换算后单位为秒并取中位数。
12. **计时后精度**：计时中产生错误结果也应失败，不能只检查第一次输出。
13. **无效计时**：零、负或非有限event结果不得作为有效latency返回。

本次接口精简还建议验证默认 `./tuned_aiter`、显式 `None` 拒绝、无锁/无随机目录，以及删除的参数不再被CLI接受。
现有 [test_moe_driver.py](test_moe_driver.py)、[test_moe_ref.py](test_moe_ref.py)、[test_quantizer.py](test_quantizer.py)
覆盖各自实现；不把已删除的调优测试搬到这些文件。