# Paged Attention / SWA

更新：2026-09-09。默认统一入口为 [test_mha_pa.py](test_mha_pa.py)：CLI运行性能集，普通pytest运行边界正确性测试。
MI308计算/访存分离pipeline已完成68项回归。用户确认250T目标为D128/D192 MHA long、含两种调度；persistent分别266.633/266.633T，普通grid分别264.072/267.176T，四项达标。平台文档第2节更新为同一完整进程的61条结果，full/short仍为237.897–248.660T，不扩大达标结论。结果为低负载启动、非独占，不宣称全程无干扰。
先逐buffer检查FP32参考O与重复逐位一致性，再报告 **acc、时间、TFLOPS、逻辑GB/s**；性能集不测LSE，BF16功能契约另校验自然对数LSE。
正常运行不打印选卡、准备、校验、轮次或阶段耗时等流程信息；性能表及`--verbose-runs`逐样本性能输出保持不变。
出错仍输出异常诊断；阶段耗时、设备映射、dispatch和跳过原因保存在JSON，选卡采样保存在idle JSONL。
平台环境、当前范围及历史证据见 [README.308.md](README.308.md) 和 [README.355.md](README.355.md)。
本机gfx950低负载重测见 [README.355.md](README.355.md)：按用户允许的gfx/UMC低于2%条件启动，CLI与性能pytest各自48候选全部通过；实际设备为MI350X，非独占。

**2026-09-23 D256 扩展：**gfx942 BF16 新增 `DQ=DV=256`，目标 `H=24/HK=2`，
使用独立 BM128/BN64 M16 内核；支持原 page32/64/128、两种调度及 LSE 契约。
新增6个显式性能case与10个功能参数。**220 TFLOPS 尚未达成**，不能沿用D128/D192的历史250T结论；
最终BF16功能回归41通过，D256完整600样本原精度/重复检查通过；GPU2 Full为
**189.626/189.319T**（persistent/grid），Short-p64为193.899/193.575T，均为10buffer/50样本中位数。
全部成功/失败优化、环境门禁和最终验证见 [opt.md](opt.md)。下文68项/61行成绩属于2026-09-09历史完整轮。

**当前D256默认已切换为v73组合（用户要求）：**公开`PagedAttention`不再需要额外参数，即启用
**Q保持寄存器加载、K/V early DMA、逐条交织、页表提前/ready合并、S4 LGKM后移**。
同时适用于persistent/grid及page32/64/128；D128/D192分派和设备函数不变。
默认Full/persistent的实际ELF与此前`both`候选相同，原FP32精度及重复检查通过；本次默认切换不重测性能。
公开D256功能子集11项通过，详见[默认启用验收与文件清单](results/d256_default_20260923/README.md)。
上面的v48六shape成绩及下面的实验数字是各自历史轮次，不改写为新默认成绩，220T目标状态不变。
实验模块保留原参数默认值以支持显式对照；公共入口显式选择全部组合选项。

**可选 M32 P-exchange 实验（2026-09-23）：**新增独立
[mha_pa_bf16_256_pexchange_942.py](mha_pa_bf16_256_pexchange_942.py)，显式导入其中的`PagedAttention`选择，
不替换公共默认后端。QK按key列分半、LDS交换P/行统计、两wave各累加DV128；64KiB LDS，已测Full零scratch/spill。
独立[回归](test_mha_pa_pexchange.py)21项通过。2buffer/4样本Full为133.437/132.598T，
同轮v48为190.322/189.862T；**新版本当前更慢**，不加入默认CLI候选。
详情与资源/原始证据见[P-exchange记录](results/d256_pexchange_20260923/README.md)。

同日[CK S-shuffle对照](results/ck_sshuffle_20260923/README.md)已完成10buffer/50样本：
两个显式实例29.391/36.267T，同轮自有P-exchange133.430T、v48 190.241T。
这些shuffle配置不是gfx942默认生成实例，且交换FP32 S而非BF16 P；不作为CK最优性能结论。

**可选4B/lane direct-to-LDS实验：**[独立DMA4模块](mha_pa_bf16_256_dma_942.py)保留Q-only/K-only/Q+K，
Q/K阶段最好方案为`dma_query=False, dma_key="early", interleave=True`，K写LDS的VMEM与S4 V读取逐条交织。
Q/K/QK的78项及补测DS写/DMA写混排的26项功能检查通过；后者未优于S4读写交织。
10buffer/50样本同轮相对v48约+4%；绝对吞吐存在运行中频率波动，两个完整50样本轮均保留，
这些Q/K单因素实验当时未替换默认v48，历史性能矩阵保留。详见[结果与限制](results/d256_dma4_20260923/README.md)。

**后续V DMA实验：**同一模块显式设`dma_query=False, dma_key="early", dma_value="early"`，
V在S0与K DS读取交织、K仍在S4交织，逐tile完整64-bit地址构造V descriptor；实验factory的V默认仍`off`。
V-only/K+V共52项功能检查通过。Full探索4样本K+V为209.449T；完整50样本遇到共同降速，
全样本中位K+V为141.596T、同轮K-only133.577T/v48 128.279T，分别提升6.00%/10.38%。
实际已观测时钟下降，**不能宣称稳定209T或达到220T**；Full VGPR224/SGPR89、LDS64KiB、零scratch/spill。
详见[V DMA原始证据与限制](results/d256_vdma4_20260923/README.md)及[优化记录](opt.md)。

**v73 Memory阶段试验：**在显式K+V参数上新增`early_pages`与`late_s4_wait`。
实验factory中这两个开关默认关闭；当前公共D256入口已默认选择两者同时开启的组合。
页表提前/别名ready合并、S4 LGKM移到barrier后首消费前，三配置78项功能通过；S4-only实际仅交换两对wait/barrier，资源不变。
五候选各50样本中，纯S4后移/组合相对v72的同round/buffer配对比分别约+1.51%/+1.62%；
本轮又有公共降速与恢复，组合全样本148.349T，探索213.042T并非稳定成绩。详见[对照结果](results/d256_memstage_20260923/README.md)。

**已启用编译缓存，在排查问题时需要检查缓存是否出现问题**。沿用FlyDSL原生默认缓存，无MHA私有持久缓存层或额外缓存开关；缓存不替代正确性检查。

## 三个性能函数与显式参数集

参数集合、输入/FP32参考、gather/AITER、选卡和报告代码全部位于 [test_mha_pa.py](test_mha_pa.py)，CLI与pytest共用候选选择规则。

| suite | pytest函数 | 参数集 | case数 | gfx942候选数 |
|---|---|---|---:|---:|
| `bf16-mha` | `test_perf_bf16_mha` | `BF16_MHA_PERF_CASES` | 18 | 51 |
| `fp8-mha` | `test_perf_fp8_mha` | `FP8_MHA_PERF_CASES` | 7 | 7 |
| `swa` | `test_perf_swa` | `SWA_PERF_CASES` | 8 | 20 |
| **all** | 三类合并 | `PERF_SUITES` | **33** | **78** |

- BF16/SWA小shape只测自有kernel；其余BF16必须包含AITER，FP8全部只测自有LDS，长SWA必须包含prepared AITER与gather+CK。
	声明的参考缺依赖、编译失败或数值错误均失败，不自动删掉比较项，不探测无关参考。
- gfx942 BF16以FP8为蓝本重写为8-wave/8-stage：每个BF16性能shape均测默认persistent及普通网格，不保留旧计算内核作备用。
- **BF16 dense适配已删除，三个H8/HK8、P32/P64的原MHA形状保留**，仅在gfx942运行并与AITER比较。
- D192对齐任务在三个MHA形状各增加一个D192/V128配对；Q/KV/H/page不变，避免跨形状比较吞吐。
- **FP8外部BN32参考及其加载/SHA适配已删除，全部7个FP8场景保留**；各类独立FP32正确性参考不受影响。
- SWA的两个KV128K重复case已删除，独立shape覆盖不变；通用`--repeat`参数仍保留。
- gfx950已按低于2%启动条件完成14个可运行case、48个候选、2400样本；其余10个case因架构跳过，完整值见[README.355.md](README.355.md)。

MI308阶段分离后的单文件合并pytest：**68通过、0跳过**；41项功能、27项性能、61候选/3050 event样本，全部采用同一轮数据而非混合较快试验。suite1364.283秒、进程1366.68秒，含选卡11.353秒；54次case边界gfx最高2%、UMC0%，未见其他进程，但仍非独占。所有原始记录保留。
完整性能数据及计时直接列于 [README.308.md](README.308.md)。

## 当前CLI

从仓库根目录运行，先按平台文档准备已有ROCm环境和`PY`。以下输出目录每轮新建：

```bash
PY="${PY:-python}"
MHA=experiments/attention/flydsl/mha/test_mha_pa.py
"$PY" "$MHA" --list
OUT=$(mktemp -d "$PWD/mha-results.XXXXXX")
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

例如`--suite bf16-mha --case bf16-full-d128`只运行该Full场景，而不是整个BF16集合；在MI308比较自有BF16 persistent、普通网格与AITER。
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

默认统一入口单文件pytest收集52项功能测试（含新增D256默认factory回归）；只有`PYHIP_MHA_PERF=1`时才收集额外33项性能测试。没有独立gather测试。
新M32实验的21项回归在[test_mha_pa_pexchange.py](test_mha_pa_pexchange.py)，需显式选择该文件；不改变上述默认suite。
MI308本次数值检查41项功能及27项性能全部通过。BF16覆盖块数边界、所有Dq/Dv128/192和page32/64/128组合、两种调度、无LSE热路径/LSE、NaN尾页、前缀guard及并发stream/graph。

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
| [mha_pa_bf16_942.py](mha_pa_bf16_942.py) | `bf16_942` / `bf16_942_grid` | gfx942 BF16，Dq/Dv128/192 使用 BM256/BN64；Dq=Dv256 默认分派到 [mha_pa_bf16_256_dma_942.py](mha_pa_bf16_256_dma_942.py) 的v73组合（BM128/BN64）；两调度、page32/64/128、LSE/NaN尾页 |
| [mha_pa_bf16_256_pexchange_942.py](mha_pa_bf16_256_pexchange_942.py) | 独立驱动 `pexchange` / `pexchange_grid` | 可选实验，M32 key-split/P交换/DV128；仅D256，两调度、page32/64/128、LSE；未加入默认CLI候选 |
| [mha_pa_fp8_942.py](mha_pa_fp8_942.py) | `fp8_942` | gfx942 FNUZ FP8，仅LDS；register实现已移除 |

- [test_mha_pa.py](test_mha_pa.py)：默认三类suite与共用测试辅助实现，按参数/输入与oracle/参考/gather/硬件与报告/计时/pytest分区；原6个测试辅助模块及conftest已合并删除。
- [test_mha_pa_pexchange.py](test_mha_pa_pexchange.py)：独立M32实验的factory隔离、LDS/P布局及GPU功能回归，复用统一入口的输入/参考工具。
- [_dsl.py](_dsl.py)：生产kernel共用FlyDSL适配，仍由kernel直接导入，不属于测试框架，独立保留。
- [__init__.py](__init__.py)：保留kernel包结构。

gfx942 BF16的`persistent=None/True`采用固定驻留CTA的grid-stride调度，不是旧版atomic ticket；无需共享counter和初始化dispatch。`persistent=False`采用FP8式普通网格，二者共用计算核心。
D128/D192的V保持64-bit GLOBAL加载；D256默认K/V DMA，其中V逐tile重建完整64-bit地址的buffer描述符。
公共SHUFFLE-5D张量格式和小于2GiB的跨度边界保留。内部K/V LDS和输出C-shuffle来自新蓝本，旧BF16实现已删除。
D128/D192的V128无LSE热路径均使用完整K片段、单K+双V槽；S0/2/4/6集中VMEM/DS，S1/3/5/7集中MFMA/VALU，上一块exp按24+8分配到两次QK旁。V在S0读取/S2发布，下一块K在S4读取/S6发布；cross32归约在memory阶段与独立DS共用等待。D128静态双phase、D192动态单phase，保持8-wave/8-stage。
同源persistent long ATT已逐wave验证稳态分离；入口初始化/重排例外单独列出，普通grid ATT正在补充。D128/D192实际persistent long产物VGPR228/254、LDS49,920/58,240字节，scratch及VGPR/SGPR spill均0。LSE/V192仍使用已验证K64流式路径，不外推阶段纯度或性能。

gather实现已经迁入主文件，`aiter_gather`对照仍逐次执行它；删除的是原模块文件，不是该性能路径。
已删除未使用的旧单卡等待、Case.pack、量化前padding兼容路径和BACKENDS/single_dispatch字段；LSE功能回归复用同一FP32参考，未进入性能计时。

逻辑GB/s不是HBM流量计数器；SWA仍按完整逻辑KV计字节，gather+CK另计完整KV读写。
跨机修改记录见 [changes.md](../../../../tests/flydsl/mha/changes.md)。历史JSON、JUnit、日志、ISA和自动生成表不改写，也不冒充当前范围的新实测。