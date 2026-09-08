# gfx950 Attention：运行、测试与性能

更新：2026-09-08。适用 **MI350X / gfx950，BF16，Dqk128/192，Dv128，page64**。
唯一入口 [test_mha_pa.py](test_mha_pa.py) 已改为三套显式性能集；当前gfx950展开规则仅完成契约核对，
**没有新gfx950原生测试或性能结果**。第4节保留的是2026-09-07旧单buffer记录，不是当前范围验收。
新CLI的MI308结果单独见 [README.308.md](README.308.md)，不可代替gfx950验证。

**已启用编译缓存，在排查问题时需要检查缓存是否出现问题**。使用FlyDSL原生默认缓存，无MHA私有持久缓存层、
自定义目录管理或额外缓存开关；所有buffer仍检查FP32 O和重复逐位一致性，不测LSE。

## 1. 快速开始

在gfx950机器上从Git根执行，使用已有ROCm/AITER环境，不为测试重装GPU依赖，也不沿用MI308的临时JIT目录。
旧MI350记录的环境为Python3.10.12、PyTorch2.9.1+rocm7.2.0.git7e1940d4、
HIP7.2.26015-fc0010cf6a、FlyDSL0.3.1、AITER `4529fa9c72f06aa7144b2ebed3eb5ff01e3499d1`；这不是新环境验证。

```bash
PY=/opt/venv/bin/python
MHA=tests/flydsl/mha/test_mha_pa.py
OUT=$(mktemp -d "$PWD/tests/flydsl/mha/results/gfx950-three-suites.XXXXXX")
export PATH="$(dirname "$PY"):$PATH"
export GPU_ARCHS=gfx950 PYHIP_MHA_GPU=auto PYHIP_MHA_REQUIRED_PTL=current
unset HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
unset FLYDSL_COMPILE_ONLY FLYDSL_COMPILE_ARCH CUDAPERF

# GPU-free：列出三套显式case，包含各自架构限制。
"$PY" "$MHA" --list

# 当前整轮；all是默认suite，不再只运行一个D192 full形状。
"$PY" "$MHA" --suite all --gpu auto --required-ptl current --buffers 10 --run-count 5 --warmup 10 --repeat 1 --output "$OUT/all.json"

# 可选：单独运行BF16或精确选择一对长SWA形状。
"$PY" "$MHA" --suite bf16-mha --output "$OUT/bf16-mha.json"
"$PY" "$MHA" --suite swa --case swa-kv32768-d128 swa-kv32768-d192 --output "$OUT/swa-32k.json"
```

上述为当前CLI命令，未在gfx950执行。多架构混合机器须用`--gpu-pool`限定已分配的gfx950物理SMI编号，
或用`--gpu N`选择指定卡；`GPU_ARCHS`只是编译环境变量，不是运行时选卡过滤器。
自动选择只读检查连续3次、5秒间隔的gfx/UMC空闲与无其他进程，随后按ROCr UUID映射、核验BDF；不设置PTL/时钟/功耗，不抢占任务。
`--required-ptl current`不筛策略；其他PTL值只筛选已经启用该策略的卡，不能强加MI308的F8策略。
`--gpu current`保留现有可见设备、不做自动选卡/PTL筛选，性能case前后仍检查其他进程。

## 2. 当前参数与gfx950候选规则

- `--suite all|bf16-mha|fp8-mha|swa`选择显式集合；可选`--case ID ...`只筛选其中的场景，不修改输入或参考候选。不传时运行所选集合全部场景；ID必须精确且不能重复。
  `--list`仅列参数、不初始化GPU，不加载kernel/AITER。完整CLI表见 [README.md](README.md)。
- `--output`可选，一份根JSON保存全部所选case的`records`，成功后生成一份同名Markdown；无嵌套manifest。
  用新文件避免覆盖；自动选卡另有同名idle JSONL日志。`--verbose-runs`显示每个样本，JSON无论如何保留全部样本。
- 正常运行不打印选卡/准备/校验/轮次/阶段流程；性能表和逐样本性能输出不变。异常诊断仍可见，详情保留于JSON/idle JSONL；静默等待仍执行原空闲检查。
- CLI始终检查正确性，`buffers/run-count/repeat`必须为正数，`warmup`非负；只做功能检查用pytest。
  自定义输入请编辑主测试文件三个参数集合中的显式`Workload`行，不再使用旧的后端、preset、shape或参考开关。

三个性能函数是`test_perf_bf16_mha`（`BF16_MHA_PERF_CASES`，9项）、
`test_perf_fp8_mha`（`FP8_MHA_PERF_CASES`，7项）及`test_perf_swa`（`SWA_PERF_CASES`，8项）。
全部精确ID/参数见 [README.308.md](README.308.md)，下面仅列gfx950按当前架构规则展开的候选数：

| suite / 场景 | case执行数 | gfx950每case候选 | 候选合计 |
|---|---:|---|---:|
| BF16 smoke，Q65/KV129，D128/192 | 2 | `bf16_950`、`bf16_950_persistent`，仅自有 | 4 |
| BF16 full/causal，D128/192 | 4 | static、persistent、AITER | 12 |
| BF16原dense比较的H8/HK8 MHA形状 | 3 | gfx942限定，全部skip（含P64形状） | 0 |
| FP8全套 | 7 | gfx942限定，全部skip，无fallback | 0 |
| SWA W0+sink小shape，D128/192 | 2 | `swa_bf16`，仅自有 | 2 |
| SWA长shape，KV32K/64K/128K，D128/192 | 6 | static、persistent、`swa_bf16`、prepared AITER、gather+CK | 30 |

因此BF16为6个可运行case/16候选，SWA为8个case/32候选；`all`合计**14个可运行case、48候选，10个架构skip**。
默认协议若全部成功将有**2400个event样本**；这些是规则计数，不是本机新实测。单选FP8时gfx950没有可运行候选，CLI失败，不能视为通过。
SWA的两个KV128K重复标签已删除，独立shape覆盖不变；通用`--repeat 1`默认不变。

BF16 dense及FP8外部BN32适配均已删除；FP8在gfx942也仅测自有LDS。三个原BF16形状仅在gfx942保留为自有+AITER；其余非smoke BF16和长SWA的声明参考是**必需项**。
缺依赖、JIT失败或数值错误必须失败，不能自动去掉AITER/gather或缩小输入来凑通过。

## 3. 当前计时与比较口径

默认10个独立Q/K/V、scale/metadata和O/workspace，seed依次为20260905+i；warmup10、run-count5、repeat1。
每轮遍历全部buffer，每候选50个event，不是只测5个buffer。`repeat`增大时逐调用轮转并归一化为每调用时间。
所有buffer都先检查FP32 O、有限值及两次逐位重复；BF16逐元素`rtol=atol=0.02`，acc为逐buffer归一化平方误差的最大值。

- 采用`pyhip.cudaPerf` GPU event的全部样本中位数，不删慢样本或挑最快值；辅助launch与间隙在event内，GPU spin在起始event前。
  JIT、FP32参考、量化、workspace分配和prepared布局转换在计时外；不测LSE。
- 有效FLOPs为 $2H_q\sum_b N_{visible,b}(D_{qk}+D_v)$，只计可见QK/PV，不计mask/padding和sink。
  带宽按逻辑Q/K/V/O计字节，SWA仍含完整逻辑KV；**不是实测HBM流量**。
- AITER full使用公开varlen路由，不强制OPUS；窗口/sink显式用CK。prepared AITER的线性KV提前准备，不能称作分页端到端性能。
- `aiter_gather`每次完整读取分页KV并写出线性workspace，再执行CK；一对event覆盖两个dispatch，不缓存gather、不裁前缀。
  其逻辑字节在Q/K/V/O之上另加完整KV读+写，因此不能用不同分子的GB/s大小代替同shape延迟比较。

若只选择旧表中的6个逻辑形状，可使用当前ID，但现在会额外包含2条gather+CK，按默认协议为22候选/1100样本，**不是旧20候选/100 run复现**：

```bash
"$PY" "$MHA" --suite all --case bf16-full-d128 bf16-full-d192 bf16-causal-d128 bf16-causal-d192 swa-kv131072-d128 swa-kv131072-d192 --output "$OUT/historical-shapes-current-protocol.json"
```

## 4. 历史MI350性能记录（旧6形状 / 20候选 / 100 run）

以下数值原样保留自旧文档：2026-09-07 11:53 UTC，MI350X/gfx950、256 CU，代码起点`2bdffaa`。
当时单buffer、warmup10/run-count5/repeat1；B1/H16/HK1/V128/page64/unit descales/per-token、seed20260905。
Full Q10240/KV2583、causal Q32768/KV32768各有static/persistent/AITER；SWA Q16384/KV131072/W128+sink另加单wave，**没有gather+CK**。
旧记录记载20候选精度通过，是非独占快速诊断，保留auto-DPM/功耗策略、PTL=N/A。
**当前工作树未附带该轮原始文件，本文未重新核验这些历史数值，更未将其记作新gfx950结果。**

后端：`bf16_950`=8-wave static，`bf16_950_persistent`=8-wave persistent，`swa_bf16`=单wave。

| case | backend | acc | 时间 µs | TFLOPS | 带宽 GB/s |
|---|---|---:|---:|---:|---:|
| full_d128 | bf16_950 | 2.75148e-6 | 229.448 | 944.344 | 371.363 |
| full_d128 | bf16_950_persistent | 2.75148e-6 | 234.408 | 924.362 | 363.505 |
| full_d128 | aiter | 2.56306e-6 | 223.407 | 969.879 | 381.405 |
| causal_d128 | bf16_950 | 2.39042e-6 | 4208.627 | 1045.039 | 67.769 |
| causal_d128 | bf16_950_persistent | 2.39042e-6 | 4160.666 | 1057.086 | 68.550 |
| causal_d128 | aiter | 2.26225e-6 | 4050.822 | 1085.750 | 70.409 |
| swa_d128 | bf16_950 | 2.71154e-6 | 93.883 | 184.422 | 2144.441 |
| swa_d128 | bf16_950_persistent | 2.71154e-6 | 92.083 | 188.027 | 2186.360 |
| swa_d128 | swa_bf16 | 2.75048e-6 | 68.603 | 252.381 | 2934.662 |
| swa_d128 | aiter | 2.75048e-6 | 112.444 | 153.980 | 1790.461 |
| full_d192 | bf16_950 | 2.78096e-6 | 260.729 | 1038.807 | 408.511 |
| full_d192 | bf16_950_persistent | 2.78096e-6 | 275.090 | 984.577 | 387.185 |
| full_d192 | aiter | 2.78098e-6 | 255.969 | 1058.125 | 416.108 |
| causal_d192 | bf16_950 | 2.33660e-6 | 4884.171 | 1125.621 | 72.994 |
| causal_d192 | bf16_950_persistent | 2.33660e-6 | 4887.971 | 1124.746 | 72.937 |
| causal_d192 | aiter | 2.33662e-6 | 4809.449 | 1143.109 | 74.128 |
| swa_d192 | bf16_950 | 2.70907e-6 | 114.124 | 189.641 | 2205.130 |
| swa_d192 | bf16_950_persistent | 2.70907e-6 | 113.484 | 190.711 | 2217.566 |
| swa_d192 | swa_bf16 | 2.74844e-6 | 79.523 | 272.155 | 3164.597 |
| swa_d192 | aiter | 2.74844e-6 | 123.964 | 174.588 | 2030.091 |

原文历史链接保留（当前工作树缺失，无法本地复核）：[历史JSON](results/readme-repro.BqIaWy/performance.json)、
[历史逐run日志](results/readme-repro.BqIaWy/performance.log)。旧JSON的`config`、`protocol`、`environment`记录当时参数、环境和源码身份。
仅设为单buffer也不能把新候选集合还原为旧集合。

| AITER场景 | 旧文档记录的kernel（非当前命中验证） |
|---|---|
| D128 full / causal | `aiter::fmha_fwd_hd128_bf16_group` / `aiter::fmha_fwd_hd128_bf16_causal_group`（ASM） |
| D192 full / causal | `gqa_d192_v128_kernel<...>`（OPUS） |
| D128 SWA | CK `BlockFmhaPipelineQRKSVSAsyncTrload` |
| D192 SWA | CK `BlockFmhaPipelineQRKSVSAsync` |

不要求AITER必须命中OPUS，也不要求自有实现所有case都快于AITER；以实际输出的kernel与样本为准。
更换AITER/编译器后应按当前ID重新测试，不能沿用旧列名认定同一实现，也不能跨计时协议拼接性能结论。

## 5. 基本功能测试与通过标准

全部测试辅助代码已合入主文件，原6个辅助模块和本目录conftest已删除；生产kernel和共用DSL适配保持独立。
性能case本身检查主shape正确性；普通pytest默认仅收集14个边界功能项，`PYHIP_MHA_PERF=1`才额外收集24个性能项。
当前gfx950架构适用性如下，**不是本次通过结果**：

| 测试 | 范围 | gfx950架构适用性 |
|---|---|---|
| `test_bf16_mha` | D128/192×static/persistent；ragged、空Q/KV、NaN尾页、padded布局、非单位scale | 4项可运行 |
| `test_swa` | D128/192单wave；同类边界加W128/sink | 2项可运行 |
| `test_fp8_mha` | D128/192×C/NC×两种Q scale，仅gfx942 | 8项skip |

按第1节环境运行普通功能测试，或显式启用性能集：

```bash
PYHIP_MHA_SELECTION_LOG="$OUT/pytest-functional-gpu.jsonl" "$PY" -m pytest "$MHA" --import-mode=importlib -q --junitxml="$OUT/tests.xml"
PYHIP_MHA_PERF=1 PYHIP_MHA_OUTPUT="$OUT/pytest-perf" PYHIP_MHA_SELECTION_LOG="$OUT/pytest-perf-gpu.jsonl" "$PY" -m pytest "$MHA" --import-mode=importlib -k test_perf_ -q
```

性能pytest共24参数项，在gfx950按规则14项可运行、10项架构skip；每项可含多个候选，不等于48个pytest项。
`PYHIP_MHA_PERF=1`在收集前启用性能函数，`-k test_perf_`只选性能；省略`-k`则同时执行功能与性能。
`PYHIP_MHA_OUTPUT`可省略，指定时使用新目录，每case一对JSON/Markdown，已有同名文件拒绝覆盖；不再使用原pytest自定义参数/marker。
pytest选卡使用`PYHIP_MHA_GPU`/`PYHIP_MHA_REQUIRED_PTL`，未设置时GPU默认为current，收集阶段不初始化GPU。
混合架构机器的pytest应设置`PYHIP_MHA_GPU=N`限定已分配的gfx950物理卡；pytest不接受CLI的候选池选项。
没有新增CPU测试或独立gather测试；gather的运行时检查仍在SWA性能对照内。

**历史功能证据链接保留，当前工作树缺失**：[JUnit](results/cleanup.jML3Sd/tests.xml)、[日志](results/cleanup.jML3Sd/tests.log)。
旧文档记载6项通过；当时尚无当前8项gfx942 FP8功能项，不能再写成当前“6通过/0跳过”。
旧Q65/KV129检查的[full](results/cleanup.jML3Sd/full.json)及
[W0+sink SWA](results/cleanup.jML3Sd/swa.json)曾包含AITER，共14候选；它们不是现在仅自有的smoke范围。

**新gfx950验收仍待原生运行**：6个适用功能项需实际通过，8个FP8项应明确skip；
性能报告需完整、所有已声明候选正确且计时有效。缺参考不是通过，不以共享设备单次快测判断小比例优化；正式结论另需负载/独占证据。

## 6. 开发参考与保留事项

- **8-wave**：BM256/BN64、两组4-wave错开执行；packed-V避免重复搬运，PV的
  `traversal_order="mnk"`保持交错顺序。改wait/barrier必须保持DMA依赖与跨任务同步。
- **单wave**：W≤16默认QT16/BN16，其余QT32/BN32；两个16行子tile共享K/V，先QK再PV，
  控制寄存器生命周期；lazy-max和lane-local sum不能改变sink只加一次的语义。
- **边界安全**：只mask概率不足以消除V尾部NaN，0×NaN仍为NaN；保持offset/stride正确。
  单wave并非所有宽窗都最优；新形状应作为显式`Workload`记录，不据窄窗推断全部窗口。
- **选卡**：自动模式会覆盖可见设备变量，调度器环境须限定已分配的物理卡池；逻辑GPU0不是固定SMI编号。
  用户级锁不是系统预约，整个流程不改硬件策略或终止他人任务。
- **缓存**：核对FlyDSL版本、`env.runtime.enable_cache`、`env.runtime.cache_dir`和残留环境覆盖；
  沿用原生默认，不叠加私有缓存，也不通过减少10-buffer或跳过精度检查掩盖问题。
- **历史资料**：旧A/B、ISA、OPUS与跨机器数据保持原值及原协议；缺失的历史附件不新建或伪造，不把旧数量当新验收。

目录职责见 [README.md](README.md)，跨机记录见 [changes.md](changes.md)。换机需携带实际工作树和依赖，
只复制commit可能遗漏未提交修改；所有新结果另存，不覆盖历史JSON/JUnit/ISA。