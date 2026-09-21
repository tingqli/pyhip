# 完整 MoE 横向比较

[cross_compare.py](cross_compare.py) 使用 [utils/moe_driver.py](utils/moe_driver.py)
和 [utils/tune_aiter.py](utils/tune_aiter.py)，为每个 tokens 输出一张 Markdown 表：

- Aiter 基准优先显示：相对参考的 `calc_diff`、新测完整调用耗时、实际选中的 stage1/stage2 kernel 名。
- 后续按 `moe_driver.registry` 注册顺序遍历config-only工厂（包括其它 Aiter 布局变体）；`ref`、`tuned` 和单独测量的 `aiter` 基准不在候选注册表内。
- 终端和Markdown默认每个config＋tokens一行，并列展示Aiter和最快有效候选；`-v 1`展开全部 `OK` 候选，`-v 2`展示全部状态。Aiter基准始终保留，JSON/HTML保留完整结果，预检不支持的候选不执行。
- 每行包含配置名、状态、`calc_diff`、绝对 E2E 延迟（微秒）、相对 Aiter 加速比，以及超参/失败原因。
- 不以调优 CSV 中的 `us` 代替实测，不以单 GEMM 性能代替完整算子。

[utils/moe_bench.py](utils/moe_bench.py) 提供普通的 `prepare_batch()` / `measure_driver()` 函数，
数据生成、参考计算、Aiter基准和全部候选均在**主进程**串行执行。每个batch只准备一次数据和参考，跨候选复用；没有Ray、worker、进程隔离或故障重建。
只有官方Aiter调优保留原有子进程机制。

## CLI

在本目录、使用安装有 ROCm Torch/Aiter/PyHIP/FlyDSL 的 Python，无需额外调度框架依赖：

```bash
python cross_compare.py
```

不传配置时，按 `MODEL_CONFIGS` 的顺序遍历所有模型，每个模型遍历默认17项 tokens：1、2、4、……、65536。
每个 batch 仍先复用/调优 Aiter、验证实际 dispatch，再比较全部实际候选；不启动并行调度器。
`python cross_compare.py --dry-run` 可只检查全部模型与候选而不使用GPU。

指定配置时保留原来的单模型行为：

```bash
python cross_compare.py 1,4096,256,4,2,bf16,silu,1,17,128 \
  --output-dir tuned_aiter
```

### 输出详细程度

`-v LEVEL` / `--verbose LEVEL` 接受非负整数，默认 `0`；单模型和全模型模式一致：

| LEVEL | 终端和Markdown表格 |
|---|---|
| `0` | 每个测试（config缩写＋实际batch-size）一行；列出Aiter状态、Aiter/Best耗时（μs）、`Aiter / best`、双方 `calc_diff`、Aiter选中kernel和Best driver/kernel配置名 |
| `1` | 每个batch展开逐kernel表格，只显示Aiter基准与全部 `OK` 候选 |
| `>1` | 每个batch展开全部候选，包括 `FAIL`、`ERROR`、`UNSUPPORTED`、`NOT_RUN` |

Best只在**非Aiter基准**的 `OK` 候选中选实测E2E最小者，同延迟保留注册顺序；它也可能慢于Aiter。没有有效候选时Best相关列为 `—`；Aiter失败仍显示其状态，但相对性能比为 `—`。
具名模型使用简称，无匹配简称时使用 `TP/H/I/E/K/quant/activation` 的紧凑名称；JSON保留完整数值配置。Aiter显示stage kernel名，Best显示当前driver报告的kernel/注册配置名。
默认模式整轮只打印一次表头，不再逐batch插入元数据段落；进度、错误提示和artifact路径走stderr。`verbose`只控制表格，不改变测试范围、失败退出码、JSON/HTML完整结果或最佳driver CSV的选择（CSV仍允许Aiter基准获胜）。

### 测试进度

- 进度条使用 `tqdm`，输出到 **stderr**；stdout 仍是原来的 Markdown 结果表，可单独重定向保存。
- 全模型模式显示已处理模型数；每个 batch 显示模型、`batch i/N`、tokens、已处理/总 driver 数、耗时和估计剩余时间。
- 长任务开始前显示当前阶段：GPU probe、Aiter `tune/cache lookup`、GPU data/reference 准备、Aiter benchmark、当前候选 benchmark。阶段内不解析上游日志或伪造百分比，详细日志按阶段保存。
- `UNSUPPORTED`、失败和 `NOT_RUN` 也计为已处理，100%不代表全部精度通过。Aiter调优与JIT耗时差异很大，ETA仅供参考。
- `--dry-run` 不显示进度条；Python API 的 `stream=None` 保持静默。正常结束、异常或 Ctrl-C 都会关闭进度条。

位置参数与调优助手相同：

```text
TP,model_dim,inter_dim,experts,topk,quant_scheme_str,activation[,key=value...][,tokens...]
```

`inter_dim` 是 TP 划分前宽度，本地 I 为 `inter_dim // TP`。
省略 tokens 时使用调优助手的17项默认列表：1、2、4、……、65536。
显式 tokens 必须是无重复的正整数、小于 `2**24`；按给定顺序执行，不自动补其它 batch。
也接受 [test_moe.py](test_moe.py#L1663-L1760) 的模型简称，如 `xiaomi,8192,16384`，或 `h3,preshuffle=True,8192`。
简称区分大小写，只展开为完整config；`--help`列出全部简称及对应配置，详见[模型预设](utils/moe_driver.md#driver-优化与对比工具)。
kwargs只能位于activation之后、tokens之前；支持末尾一个逗号。
支持键、字面量/dtype写法与覆盖规则见[共享config格式](utils/moe_driver.md#driver-优化与对比工具)。
配置在入口转换为使用local I的 `MOEconfig`，进入preflight、driver工厂和统一参考；工厂返回 `(prepare, run)`。物理布局/tile是注册工厂固定的超参，不由config字典覆盖。

```bash
python cross_compare.py "1,4096,256,4,2,bf16,silu,preshuffle=True,1,17,128," \
  --dry-run
```

只复用已有调优结果，不触发 tuner：

```bash
python cross_compare.py 8,4096,2048,512,10,fp8_blockscale,silu,1,4,8 \
  --output-dir tuned_aiter --no-tune
```

CLI 始终遍历 `moe_driver.registry` 中实际候选，不提供候选筛选选项。`tuned` 是结果消费者，不参加自身的调优比较。
不支持的组合由预检或运行阶段跳过，`verbose>1` 时仍在表格中显示其状态和原因。

CPU-only 预览：

```bash
python cross_compare.py 1,4096,256,4,2,fp8_blockscale,silu,1,17 --dry-run
```

显示规范化模型参数、候选列表和静态拒绝原因。不查询/初始化 GPU，不创建输出文件、不调优。
比较脚本的普通导入和 `--help` 不导入 Torch/Aiter；dry-run 和 moe_bench 导入会加载 Torch，但不初始化 CUDA。

### 选项

| 参数 | 默认 | 含义 |
|---|---|---|
| `--output-dir` | `tuned_aiter` | 复用/追加调优历史；比较日志与结果也保存在此目录 |
| `-v` / `--verbose` | 0 | 表格详细程度：0为Aiter/Best并列汇总，1为Aiter＋全部OK，>1为全部候选 |
| `--no-tune` | False | 只读已有精确 tuned 行；缺行时报错 |
| `--force` | False | 每个请求 batch 强制重调并追加；与 no-tune 互斥 |
| `--device` | 0 | 继承的可见设备列表中的编号；主进程使用 `torch.cuda.device(device)`，不改可见设备环境变量 |
| `--aiter-root` | 安装的 checkout | 与实际导入源码核对 |
| `--seed` | 43 | 独立随机generator；所有候选使用同一生成函数和seed，不做全量输入哈希 |
| `--warmup` | 5 | 每轮 `run_perftest` 的预热调用次数 |
| `--iters` | 20 | 每轮逐调用event计时的次数 |
| `--rounds` | 5 | 多轮平均延迟取中位数 |
| `--buffer-count` | 10 | 轮换独立的激活、权重、路由和输出存储，数值保持相同 |
| `--diff-threshold` | 0.02 | 所有行统一精度阈值；超阈值不产生有效加速比 |
| `--tune-timeout-s` | 120 | 传给官方 tuner 的任务上限 |
| `--tune-backends` / `--kernel-regex` | 不限制 | 只影响新调优；受限结果不称为全 catalog 最优 |
| `--ref-intermediate-dtype` | `none` | 统一参考在激活后是否舍入到 BF16；可选 none/bf16 |
| `--ref-route-dtype` | `none` | 统一参考在路由加权后是否舍入到 BF16；可选 none/bf16 |
| `--ref-chunk-size` | 256 | 参考的 token chunk，大模型可显式减小 |

新driver固定stage2路由加权与设备原生FP8；已删除 `--doweight-stage1`。config kwargs仅保留 `preshuffle, swiglu_limit, beta, linear_beta, output_dtype`，Python用 `driver_kwargs={...}`。
Aiter内部为A8W4选择interleave packing，其余separated；源W1始终自然 `[gate;up]`。SwiGLU clamp=7，SiTUv2 `beta=4, linear_beta=25`。
脚本会尝试建立官方tuned Aiter baseline，所以显式kwargs仍须符合该CSV可表示的标准组合；不兼容的布局、clamp、输出dtype或SiTU常量在开始前报错。
需要不同参数时，用 [utils/moe_bench.py](utils/moe_bench.py) 测量支持它的driver，不将自定义问题与标准调优历史混用。
若某本地 kernel 需要不同 clamp 或量化策略，该配置报告 `UNSUPPORTED`，不改契约使它勉强通过。
可比较的数值策略是 Aiter 调优助手支持的子集，不是所有15种参考 quantizer 都有 Aiter baseline。

## 调优结果复用与实际 dispatch

1. 按完整 key 查询输出根目录的 tuned CSV；包含 GPU arch/CU、padded token key、模型维度、A/W dtype、quant、activation 等。
2. 同 key 选择最后一条无 tag 的有效记录。只查指定目录根CSV；不扫描旧artifact，需要旧记录时显式指向它所在的目录。
3. 调用 `tune_aiter.ensure_tuned()` 只获取/搜索配置，不生成输入或E2E计时；`--force` 强制搜索，`--no-tune` 只读。
4. 将源历史选中的单行字典直接传入基准driver并保存在报告中，不再生成额外配置目录。
5. 通过 `moe_driver.aiter(moe_config, tuned_config=单行字典)` 得到prepare/run，调用public `fused_moe`；首次实际batch核对metadata、stage调用和A dtype，稳态不重复审计。
6. 在比较报告中保存源 CSV、原始行、搜索/复用口径与实际 kernel 名；不使用官方 tuner 的阶段 `us` 计算横向加速比。

默认和 cached-only 比较不修改已有历史 CSV；新调优仍沿用 helper 的 append-only 行为和编译缓存。
同 padded token key 可以复用 kernel，但参考和性能总是按实际 batch 分别运行。
调优失败、cached-only缺行、最新行无效或 public dispatch 不符合数值契约时，该 batch 的 Aiter 行保留失败，不回退到 Aiter 内置 heuristic。
这不阻止本地候选：统一参考独立于Aiter准备，基准失败仍复用同一份GPU数据/参考；无有效Aiter基准时，相对加速比显示`—`，JSON为`null`，不拿首个候选冒充Aiter。
例如 FP8 key 下选中的 `xbf16` kernel 不等于要求的 FP8 A 策略，driver 会拒绝它，而非给出误导的 speedup。

官方调优的“最佳”仅指其已搜索 catalog 的 winner，不等同于所有实现的全局最优。
缓存历史通常没有原始搜索范围，表格明确显示 `original search scope unknown`。

### Hy3 的 I192 / gfx950 限制

`hy3,8192`的本地I是`1536 // 8 = 192`。当前上游gfx950的FP8 CK stage2候选使用K128/K256，K64候选被禁用，实测所有stage2候选均不支持此shape，因此官方tuner无法组成有效的两阶段winner并退出1。
stage1的部分ASM候选也有tile不整除警告，但仅过滤这些候选不会补出缺失的stage2。
旧 [test_moe.py](test_moe.py#L1222-L1237) 会把Aiter本地I向上补到128的倍数（192→256）；这里不悄悄改模型shape或FP8量化策略，也不修改上游catalog。
需要对比加宽后的性能时，显式使用 `hy3_pad,8192`（global I=2048、local I=256）；原 `hy3` 保持不变。
因此当前环境的Aiter行仍会报无有效组合，但本地JIT/FlyDSL候选可以继续测量；无需Aiter时也可直接使用 [utils/moe_bench.py](utils/moe_bench.py)。

## 同数据、统一参考

- 每个 batch 使用 BF16 `[M,H]` 输入和自然 BF16 `[E,2I,H]`、`[E,H,I]` 权重。
- W1/W2 来自 FP32 正态分布、分别按 `sqrt(H)`/`sqrt(I)` 缩放后转 BF16；初始化临时张量限于一个专家。
- 独立 RNG 生成 logits，经 topk 和 softmax 得到 Int32 IDs / FP32 路由权重，每个 token 的专家互异。
- 权重 RNG 与激活/路由 RNG 分离，因此同一 seed 下不同 tokens 也使用相同权重。
- 数据生成、参考和event计时共用 [utils/moe_bench.py](utils/moe_bench.py)。每个batch调用一次 `make_data()` 和 `make_reference()`；Aiter与全部候选复用，不读写大Tensor文件。
- GPU数据和默认FP32参考都是主进程的局部变量。只保留当前batch，结束时释放数据、参考、上一个Aiter闭包和allocator空闲缓存，再运行下一batch的官方tuner；不重建进程或GPU上下文。
- 每个候选构造prepare/run闭包，只准备一份私有输入，再由 `pyhip.run_perftest` 复制和轮换。权重由各driver独立prepare为字典，参考只用于只读比较。实验kernel不会接收共享源权重/激活/路由作为可写工作区。显式 `output_dtype` 同样作用于统一参考。
- 默认参考中间值/route 不强制 BF16 舍入，保持 FP32 计算策略；两个显式舍入选项对整个 batch 的所有行一致，不按 kernel 的输出选择有利参考。
- W quant/shuffle 每个driver执行一次；嵌套prepared weights展开为顶层Tensor参数，确保 `run_perftest` 也复制静态权重并保留 `is_shuffled` 标记，不只是轮换激活。

### 准备开销与剩余成本

`batch.preparation` 记录 `make_data_s`、`make_reference_s`、`retained_bytes`；候选结果的 `prepare_buffers_s` 记录单份私有输入与权重准备，不含 `run_perftest` 内部复制。
候选的 `wall_s` 记录准备、编译、检查和测量的总墙钟；不再记录PID、RPC等待或恢复字段。以上都**不混入 E2E latency**。

跨候选/batch/模型复用当前进程的导入、GPU上下文与JIT缓存；每driver只量化/shuffle一次，再复制轮换buffer。
剩余成本是首次导入/上下文初始化、新kernel的JIT、每batch的参考及各driver自己的静态准备。可继续直接使用 `moe_bench --driver 正则` 测少量候选。

`calc_diff` 定义为：

$$1 - \frac{2\sum xy}{\sum(x^2+y^2)}.$$

它不是错误元素百分比，也不是最大相对误差。拒绝非有限结果；优化输出 BF16，参考默认输出 FP32。

## 完整调用计时

所有行与调优工具共用 `eager_moe_driver_v3`，计时委托给 `pyhip.run_perftest`：

1. 数据/参考生成、每driver一次 `prepare()`、独立buffer复制和路由校验在计时外。
2. 原始私有输入先启动一次，完成 JIT 并检查精度；失败时不进入计时。
3. 每轮由 `run_perftest` 复制 `buffer_count` 份输入，执行 `warmup` 次预热和 `iters` 次计时调用；输出复制前重新填 NaN，可发现未写完的输出。
4. `run_perftest` 对每次完整调用使用current stream event及同步，返回平均微秒；换算为秒后，多轮取中位数。
5. 每轮结束复查所有实际使用过的输出副本，精度失败行不显示有效加速比。

排序、动态 A1/A2 quant、两阶段 GEMM、激活、必要清零、任务表、reduction 和运行 workspace 都在完整调用内。
buffer 轮换也包含静态 W 的独立存储；不会只轮换 activations 却声称冷权重测量。
默认10份可能较占显存，由用户选择 `--buffer-count` 并承担OOM风险；不再做显存预估，不自动改变协议。

**这是 eager 完整接口延迟，不是纯 GPU kernel 累计时长或 CUDA Graph 延迟。**
event时间可能包含Python提交不足造成的GPU空闲；Aiter首次dispatch验证在预热前完成，不再计入每次调用。
v3改为逐调用event计时和同步，不再用一对event包住整轮；新旧数字不能直接作为kernel加速比。
因此表格加速比只适用于这个协议，不能直接当作模型吞吐提升。对极短 kernel 要增加 iters/rounds；本脚本不调整时钟，不宣称排除了外部 GPU 负载。

$$\text{speedup}=\frac{\text{本次实测 Aiter E2E}}{\text{本次实测 candidate E2E}}.$$

## 主进程执行与失败处理

`compare()` / `compare_all()` 都是普通同步函数，没有runner、phase请求分发或后台测试服务。

- 正常、精度失败、不支持和可捕获的Python异常都在当前进程处理；候选异常保存 traceback、状态和原因后继续下一候选，不重试、不重新生成batch。
- 初始batch准备失败时，本来可执行的候选标记 `NOT_RUN`。Aiter调优或baseline失败不阻止本地候选；只有有效Aiter行才能提供相对加速比。
- 主进程使用调用者当前可见设备列表中的 `device`，不覆盖 `HIP_VISIBLE_DEVICES` / `CUDA_VISIBLE_DEVICES`。官方tuner仍使用其原有环境配置和独立子进程。
- 采用已安装的Aiter和当前进程的编译缓存；`--aiter-root`只核对源码位置，不在运行中更换导入路径。Aiter driver负责串行切换配置/缓存，不通过重启进程切换数值profile。
- **没有GPU故障隔离**：非法访存、native崩溃、设备卡死可能终止或阻塞整轮测试；用户负责停止/重启，不承诺继续运行或恢复损坏的GPU上下文。

缺行/force仍先调用独立的**官方调优子进程**，之后在主进程准备数据、执行E2E。前一batch的数据在下一次调优前释放。
比较模式首次精度失败不进入计时，每轮计时后继续检查所有使用过的输出；不启用直接 `--driver` 模式的诊断计时或首项加速比。
删除测试 `timeout_s` / `--timeout-s`；仅保留 `tune_timeout_s` / `--tune-timeout-s`，语义仍是官方tuner的单任务watchdog。

### 注意事项

- 模块全局状态、环境和编译缓存会保留。这里是可信实验kernel的串行工具，不提供内存沙箱、环境恢复、动态模块重载或并发Aiter状态保证。
- 每次串行使用一张GPU；支持进程已初始化CUDA的情况，也支持已有可见设备列表中的非零device编号。
- 不用CSV/GPU合作锁，用户负责单例执行与GPU/输出目录资源分配。每batch保留一份源数据/参考，加上轮换权重；不自动缩小工作量来规避OOM。
- 正常结束、Python异常和Ctrl-C会释放batch引用、关闭日志和进度条。不承诺回收所有编译器派生进程，也无法保证恢复GPU reset、硬件故障或主进程被外部SIGKILL。

| 状态 | 含义 |
|---|---|
| `OK` | 精度、调用、数据一致性和计时验证通过 |
| `UNSUPPORTED` | 工厂/运行时明确不支持数值或 shape/tile 组合 |
| `FAIL` | 有限输出但 `calc_diff` 超阈值；无有效 speedup |
| `ERROR` | 编译、运行、OOM、非有限输出、无效证据或 baseline 调优失败 |
| `NOT_RUN` | 初始参考生成或batch准备失败，无法运行候选；并非单纯Aiter失败 |

普通候选失败继续比较后续候选及 tokens；native/GPU故障不保证继续。Aiter失败后继续测候选，只缺少相对加速比；只有初始参考/准备失败才保留`NOT_RUN`。
失败始终保存在JSON/HTML中，终端/Markdown按 `verbose` 选择显示；baseline 不支持时也保留原因且该batch判为失败，不伪造基准。
退出码：全部 batch 有有效 baseline 且没有 FAIL/ERROR 为0；否则1；CLI 参数错误2；中断130。
单纯存在不支持的本地候选不算整体失败。

## 输出目录和 Python API

stdout打印结果表；默认 `verbose=0` 时artifact路径走stderr，展开模式沿用stdout路径提示。进度走stderr，编译器/测量输出写到日志。每次比较独立保存报告，不覆盖历史比较：

```text
output_dir/
  tuned.csv
  moe_tuned.csv                     # 每个精确模型/GPU/tokens 的最佳有效候选
  comparison.html                   # 本次调用的累计单文件交互总览
  comparison.json                   # 与HTML同源的全部标量报告
  untuned.csv                       # 仅新调优时写入
  work/                             # tune_aiter 固定单shape工作目录
  jit/, flydsl-cache/, ...           # 官方tuner缓存；主进程沿用自己的环境设置
  cross-compare-.../
    plan.json
    comparison.md
    comparison.json
    comparison.html                 # 该模型本次运行的独立离线报告
    setup.log                       # Aiter导入与GPU信息
    tokens-17/
      comparison.json
      prepare.log                   # 一次性数据/参考准备
      aiter.log
      candidate-000.log             # 各候选测量日志；完整状态保存在comparison.json
```

Python 调用仍为同一流程：

```python
import sys
import cross_compare

report = cross_compare.compare(
    TP=1, model_dim=4096, inter_dim=256, experts=4, topk=2,
    tokens=[1, 17], quant_scheme_str="bf16", activation="silu",
    output_dir="tuned_aiter", no_tune=True,
    stream=sys.stdout, verbose=0,
)
```

省略 `stream` 不打印表格，仍返回并保存spec/seed、samples、参考设置、status和错误文本。
Python API 与 CLI 一样，按注册顺序遍历实际候选，不再提供筛选参数或 glob 匹配。`compare_all(output_dir=..., stream=...)` 串行运行全部预设模型，其余选项与 `compare()` 相同。
`compare()`、`compare_all()`接受 `verbose=0/1/2/...`；`render_markdown(batch, verbose=..., model=...)`也可用同一规则重新渲染已有batch结果。
不维护进程生命周期、任务请求文件、恢复记录、请求UUID、设备身份审计或输入哈希。

## HTML 与最佳 driver CSV

HTML 把数据、CSS、JS 全部内嵌，直接双击打开，无服务器/CDN/npm依赖。支持模型和tokens切换、跨batch延迟/加速比曲线、driver筛选/开关、线性/对数坐标、延迟排名、表格排序及错误详情；Aiter基准在详情表中始终排第一。
每完成一个batch更新根目录总览和CSV，全模型运行中保留已完成模型；不是实时网页，继续运行后需刷新页面。

`moe_tuned.csv` 的核心列：

- key：`gfx,cu_num,model_dim,inter_dim_tp,experts,topk,quant_scheme,activation,preshuffle,swiglu_limit,beta,linear_beta,output_dtype,tokens`。
- `inter_dim_tp` 是 **local I**；global I/TP仍在报告spec中。`tokens` 是 **实测 M**，不是 Aiter 的 padded token key。
- `model` 是阅读用别名，不参与查找；同数值配置不同别名不会成为两条调度记录。
- `driver` 是config-only注册函数名，例如 `prefill_fp8_1x4_64x256`；普通候选 `params={}`，超参由函数固定。Aiter基准获胜时 `params.tuned_config` 包含已验证的官方单行快照，不依赖以后变化的 tuned CSV。
- `us,diff,benchmark_protocol,aiter_status,report` 保存实测微秒、误差、协议、基准状态及原始报告位置。

只在 `status=OK` 的行中选最小 E2E latency；FAIL/ERROR/UNSUPPORTED/NOT_RUN 永远不获胜。
同延迟按原顺序选择（Aiter优先）；再次测量替换同key，其它key保留。若重测后没有有效候选，删除该key旧记录，不留下假winner。
CSV不是append-only调优历史；Aiter自己的 tuned CSV 仍保留原append-only行为。`--no-tune`只禁止Aiter搜索，仍会输出新的实测winner。
旧类接口的最佳driver CSV列不兼容，在调优/GPU工作前明确报错；需移走旧表或选新输出目录后重测，不自动转换或删除历史。官方Aiter tuned CSV可继续复用。

## 注意事项：刻意保持简单

1. **“最佳”仅指本次协议下、当前GPU上、精度通过的已测候选最小值**，不保证未执行的候选或其它软件版本、路由分布更慢。生产前用足够的iters/rounds重跑，不使用验证时的极短计时作最终调优。
2. Aiter固定第一且必须通过tuned配置验证才算有效。原Hy3 I192当前无上游可用组合：保留Aiter错误、继续测本地候选；CSV可记录本地winner，但 `aiter_status` 标为失败，HTML不显示相对加速比，整体退出1。不伪造未调优baseline。
3. 全量8模型×17tokens×全部候选可能耗时很长；默认10份权重buffer也可能OOM。记录可捕获的OOM，不自动降batch、少测候选、改buffer-count或估算资源；测试不再有外层超时。用户按机器资源设置选项。
4. 精确tokens无记录就报错，不向上取整、不插值、不自动在线调优。需要新的M时单独运行相应配置。
5. CSV单写者，不做锁、事务、版本哈希、热更新、自动续跑。被重新测量的key才会更新；中断后未访问的旧key仍保留。修改kernel/registry/环境后应重测，过期快照不保证继续有效。
6. HTML/根JSON表示本次调用，下一次会覆盖；每模型run目录保留历史。CSV保留其它key，故它可能覆盖比当前HTML更多的历史模型。
7. `tuned(config, tokens=...)` 返回的prepare可能为不同winner准备多份布局，显存开销由用户承担。可显式限制tokens子集；详见 [moe_driver 的 tuned 用法](utils/moe_driver.md#tuned按实测-csv-分派)。运行期不量化权重、不shuffle、不读文件。

## 验证

建议保留的集成检查：每batch只生成一次数据/参考；数据、参考、Aiter和候选都在调用者PID执行；普通异常后不重建数据；无baseline不产生speedup；初始准备失败才跳过候选；每driver只prepare一次但buffer地址独立；跨batch先释放数据再调优；非零可见GPU和Ctrl-C清理。
2026-09-21主进程改造验证：511项数值测试通过（driver 163项）；无参dry-run覆盖8模型/136个batch。模拟验证普通异常/精度失败/不支持继续、缺基准无speedup、准备失败、中断释放与双模型汇总。真实MI350X GPU 1、预先初始化CUDA后运行Hy3-pad的1/4096 tokens，各44行分别为26/25个OK、18/19个UNSUPPORTED，Aiter均使用已有调优行并验证dispatch；确认数据/参考/测量同PID、每batch仅准备一次、设备/可见列表不变、日志和HTML/CSV输出正常。短计时只用于冒烟；未执行完整全模型调优或注入非法GPU访问，旧崩溃/超时恢复不再是功能承诺。
不恢复已删除的专用比较测试文件；kernel数值边界仍在各driver/参考测试中覆盖。
新增功能建议检查：无参dry-run覆盖所有模型/tokens；失败行不能获胜；重测同key替换而非重复；缺M/错误GPU报错；tuned复用各自布局且Aiter快照被验证；离线HTML筛选和失败状态可见。不另建专用测试框架。