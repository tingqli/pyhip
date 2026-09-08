# Paged Attention / SWA

更新：2026-09-08。唯一入口为 [test_mha_pa.py](test_mha_pa.py)：CLI运行性能集，普通pytest运行边界正确性测试。
先逐buffer检查FP32参考O与重复逐位一致性，再报告 **acc、时间、TFLOPS、逻辑GB/s**；全程不测LSE。
正常运行不打印选卡、准备、校验、轮次或阶段耗时等流程信息；性能表及`--verbose-runs`逐样本性能输出保持不变。
出错仍输出异常诊断；阶段耗时、设备映射、dispatch和跳过原因保存在JSON，选卡采样保存在idle JSONL。
平台环境、当前范围及历史证据见 [README.308.md](README.308.md) 和 [README.355.md](README.355.md)。
本机gfx950低负载重测见 [README.355.md](README.355.md)：按用户允许的gfx/UMC低于2%条件启动，CLI与性能pytest各自48候选全部通过；实际设备为MI350X，非独占。

**已启用编译缓存，在排查问题时需要检查缓存是否出现问题**。沿用FlyDSL原生默认缓存，无MHA私有持久缓存层或额外缓存开关；缓存不替代正确性检查。

## 三个性能函数与显式参数集

参数集合、输入/FP32参考、gather/AITER、选卡和报告代码全部位于 [test_mha_pa.py](test_mha_pa.py)，CLI与pytest共用候选选择规则。

| suite | pytest函数 | 参数集 | case数 | gfx942候选数 |
|---|---|---|---:|---:|
| `bf16-mha` | `test_perf_bf16_mha` | `BF16_MHA_PERF_CASES` | 9 | 16 |
| `fp8-mha` | `test_perf_fp8_mha` | `FP8_MHA_PERF_CASES` | 7 | 7 |
| `swa` | `test_perf_swa` | `SWA_PERF_CASES` | 8 | 20 |
| **all** | 三类合并 | `PERF_SUITES` | **24** | **43** |

- BF16/SWA小shape只测自有kernel；其余BF16必须包含AITER，FP8全部只测自有LDS，长SWA必须包含prepared AITER与gather+CK。
	声明的参考缺依赖、编译失败或数值错误均失败，不自动删掉比较项，不探测无关参考。
- **BF16 dense适配已删除，三个H8/HK8、P32/P64的原MHA形状保留**，仅在gfx942运行并与AITER比较。
- **FP8外部BN32参考及其加载/SHA适配已删除，全部7个FP8场景保留**；各类独立FP32正确性参考不受影响。
- SWA的两个KV128K重复case已删除，独立shape覆盖不变；通用`--repeat`参数仍保留。
- gfx950已按低于2%启动条件完成14个可运行case、48个候选、2400样本；其余10个case因架构跳过，完整值见[README.355.md](README.355.md)。

MI308单文件合并后CLI与性能pytest各自完整通过：**24 case / 43候选 / 2150 event样本**；功能pytest为12通过、2跳过，性能pytest为24通过。
完整性能数据及计时直接列于 [README.308.md](README.308.md)。

## 当前CLI

从仓库根目录运行，先按平台文档准备已有ROCm环境和`PY`。以下输出目录每轮新建：

```bash
PY="${PY:-python}"
MHA=tests/flydsl/mha/test_mha_pa.py
"$PY" "$MHA" --list
OUT=$(mktemp -d "$PWD/tests/flydsl/mha/results/mha-three-suites.XXXXXX")
"$PY" "$MHA" --suite all --output "$OUT/all.json"

# 可选：精确选择同一suite中的两个case，不改变它们的输入或参考。
"$PY" "$MHA" --suite bf16-mha --case bf16-full-d128 bf16-full-d192 --output "$OUT/bf16-full.json"
```

| 参数 | 当前含义 |
|---|---|
| `--suite all` / `bf16-mha` / `fp8-mha` / `swa` | 默认`all`；同进程、同一选定GPU依次运行 |
| `--case ID ...` | 可选的场景筛选：不传则运行所选suite全部场景；传入一个或多个精确且不重复的ID，只运行这些场景，但不修改其shape或参考候选 |
| `--list` | 列出所选case及完整参数，不初始化GPU、不加载原生kernel/AITER；包含架构限定信息 |
| `--output` | 可选的新JSON文件，成功后生成同名Markdown；所有case直接放在根JSON的`records`，无嵌套manifest/每组子目录 |
| `--buffers 10 --run-count 5 --warmup 10 --repeat 1` | 默认10个独立buffer、5个完整轮次，每候选50个event；`repeat`逐调用轮转并归一化为每调用时间 |
| `--gpu auto` / `current` / `N` | CLI默认`auto`（可由`PYHIP_MHA_GPU`覆盖）；自动选卡 / 保留现有可见设备 / 等待物理SMI编号N |
| `--gpu-pool 1,2,3` | 限制自动选择的物理卡池；不能与数字`--gpu`并用，配合`current`会转为自动选择 |
| `--required-ptl current` / `VECTOR,F8` / `VECTOR,BF16` | 默认`current`不筛策略；选卡时只筛选**已启用**指定策略的卡，绝不设置硬件 |
| `--max-gpu-utilization 2` | 显式允许低负载启动：gfx/UMC均严格低于2%，允许驻留进程；默认0保持原严格空闲规则。pytest用`PYHIP_MHA_MAX_GPU_UTILIZATION` |
| `--verbose-runs` | 打印每个计时样本的acc/时间/TFLOPS/带宽，不恢复流程打印；不指定也会保存全部原始样本 |

例如`--suite bf16-mha --case bf16-full-d128`只运行该Full场景，而不是整个BF16集合；在MI308仍比较自有BF16与AITER。
用`--suite bf16-mha --list`查询ID；`--case`是CLI参数，不是pytest参数，执行顺序仍以参数集合为准。

`buffers/run-count/repeat`必须为正数，`warmup`可为0；CLI始终检查精度。自定义输入请编辑显式`Workload`行，
不再通过旧的后端、preset、shape或参考开关拼接场景。只做正确性检查应运行普通pytest。
不支持的架构记为skip/`unavailable`；若一个候选也没运行，CLI失败，不将空报告当通过。

自动/指定物理卡选择默认只读检查：连续3次、5秒间隔的gfx/UMC空闲且无其他进程；忙时无截止时间等待。
显式正门槛模式则连续3次gfx/UMC均严格小于门槛即可启动，不要求进程为空；case前后仍记录利用率/进程，
但不以自身测量产生的活动拒绝结果。阈值只约束开始时刻，结果标为非独占，不宣称整个测量期间低负载。
选中后按ROCr UUID映射并核验BDF，整轮固定该卡。自动选择会覆盖可见设备变量，调度器分配场景须限定获分配卡池。
`--gpu current`不进行选卡/PTL筛选或稳定空闲等待，但性能case前后仍检查其他进程。
正利用率门槛必须配合auto或物理GPU编号，不能用current绕过起始负载检查。
有输出时自动选卡另存同名idle JSONL日志；不覆盖已有报告或选卡日志。用户级锁不等于调度器独占预约。

## pytest：默认功能，性能显式启用

普通pytest默认仅收集14项功能测试；只有`PYHIP_MHA_PERF=1`时才收集额外24项性能测试。没有新增CPU测试或独立gather测试。
MI308本次功能回归12通过/2个gfx950 persistent跳过；性能pytest24项全部通过，各项可包含多个候选。

```bash
PYHIP_MHA_GPU=auto "$PY" -m pytest "$MHA" --import-mode=importlib -q
PYHIP_MHA_GPU=auto PYHIP_MHA_PERF=1 PYHIP_MHA_OUTPUT="$OUT/pytest-perf" "$PY" -m pytest "$MHA" --import-mode=importlib -k test_perf_ -q
```

`PYHIP_MHA_PERF=1`须在收集前设置，`-k test_perf_`进一步只选择性能；省略`-k`则一起运行功能与性能。
`PYHIP_MHA_OUTPUT`可省略；指定时使用新输出目录，每case保存一对JSON/Markdown，已有同名文件拒绝覆盖，不生成整轮manifest。
原pytest自定义开关及marker已移除，改用环境变量和标准pytest筛选，不再需要本目录的conftest插件。
pytest选卡使用`PYHIP_MHA_GPU`（未设置时为`current`，与CLI默认不同）及`PYHIP_MHA_REQUIRED_PTL`，
可用`PYHIP_MHA_SELECTION_LOG`保存选卡日志。pytest需指定物理卡时设置`PYHIP_MHA_GPU=N`；候选池选项只属于CLI。
低负载pytest使用`PYHIP_MHA_GPU=auto PYHIP_MHA_MAX_GPU_UTILIZATION=2`；默认0仍采用原严格空闲条件。
收集阶段和CLI列表均不初始化GPU。

## 实现与职责

| 实现 | 报告中的backend | 范围 |
|---|---|---|
| [mha_pa_bf16_950.py](mha_pa_bf16_950.py) | `bf16_950` / `bf16_950_persistent` | gfx950 BF16，static/persistent；D128/192、V128、page64，full/causal/SWA/sink |
| [mha_pa_swa_bf16.py](mha_pa_swa_bf16.py) | `swa_bf16` | gfx950/gfx942 BF16，单wave causal SWA；D128/192、V128、page64 |
| [mha_pa_bf16_942.py](mha_pa_bf16_942.py) | `bf16_942` | gfx942 BF16，8-wave；保留原有输入与padding限制 |
| [mha_pa_fp8_942.py](mha_pa_fp8_942.py) | `fp8_942` | gfx942 FNUZ FP8，仅LDS；register实现已移除 |

- [test_mha_pa.py](test_mha_pa.py)：全部测试实现，按参数/输入与oracle/参考/gather/硬件与报告/计时/pytest分区；原6个测试辅助模块及conftest已合并删除。
- [_dsl.py](_dsl.py)：生产kernel共用FlyDSL适配，仍由kernel直接导入，不属于测试框架，独立保留。
- [__init__.py](__init__.py)：保留kernel包结构；4个生产kernel未因合并而修改。

gather实现已经迁入主文件，`aiter_gather`对照仍逐次执行它；删除的是原模块文件，不是该性能路径。
已删除未使用的旧单卡等待、Case.pack、量化前padding兼容路径、BACKENDS/single_dispatch字段和未使用的LSE/unchecked测试分支。

逻辑GB/s不是HBM流量计数器；SWA仍按完整逻辑KV计字节，gather+CK另计完整KV读写。
跨机修改记录见 [changes.md](changes.md)。历史JSON、JUnit、日志、ISA和自动生成表不改写，也不冒充当前范围的新实测。