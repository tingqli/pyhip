# MI325X / gfx942 新机验收

日期：2026-09-07 UTC（机器本地起始日期为 2026-09-06）。

**交付方式**：按本轮用户最新选择，只保留工作树改动和证据，**不提交、不推送**。
这覆盖原交接文档中的本地commit要求；不擅自配置缺失的Git身份。

> 用户随后重新指定了BF16 dense LKG/V-global及FP8 BN32 prefill性能参考，详见
> [REQUESTED_REFERENCES.md](REQUESTED_REFERENCES.md)。本页旧对照仍是历史证据，不能重标为新参考已达标。

> **当前阻塞**：FP8原版对照运行未进入计时，随后GPU状态读取超时。
> 已停止本轮自己的benchmark/监测进程，未重置GPU、未改硬件设置。后续GPU测试暂停，
> 需管理员先确认设备/驱动健康。本页及[状态清单](results/newmachine_mi325_20260907T035439Z/run-status.json)
> 明确区分已通过项目和未完成项目，不能把本轮标为全部完成。

## 机器与历史边界

- 本机：**AMD Instinct MI325X，gfx942，304 CU，256 GiB HBM**；逻辑 GPU0 对应
  `0000:83:00.0`。ROCm 系统工具链 7.2.3。
- 源机：MI308X，gfx942，80 CU。**架构相同不等于吞吐、调度占用或功率策略相同**。
- MI325X 的 `amd-smi` 返回 `ptl_state=N/A`、`ptl_format=N/A`。
  本轮仅 `--ptl current`，不执行 PTL、时钟、功率、NUMA 设置，不终止其他人的进程。
- MI308X / PTL VECTOR,F8 的历史 413.984T / 400T gate **不适用于本机**。
  本机性能只建立 MI325X 自己的原版/当前版对照，不能按 304/80 简单换算验收门槛。
- 无 gfx950 硬件：gfx950 native 仍 blocked。生成 gfx950 ISA 不算 native 验收。

本轮独立证据目录：
[results/newmachine_mi325_20260907T035439Z](results/newmachine_mi325_20260907T035439Z)。
可移植[证据清单](results/newmachine_mi325_20260907T035439Z/validation-manifest.json)逐文件记录SHA、
各次JUnit及104份保留ISA，全部ISA哈希核验通过；同一测试的多轮运行不重复计为唯一通过数。
该清单为生成时快照；交付前又验证了离线清单工具，后续清单保留独立文件，不覆盖本快照。
最终[依赖/双编译器文件指纹](results/newmachine_mi325_20260907T035439Z/environment-final-inventory.json)
只读metadata和磁盘文件，没有再次导入ROCm。
历史 JSON/JUnit/ISA 保留原样。起始 commit 为
`f06791491c3b157793500252a0127d8f45c7776a`，分支 `tmp-main`；两个固定原版 Git 对象均在本地。
起始四个生产 kernel SHA256 与 [CONTEXT_HANDOFF.md](CONTEXT_HANDOFF.md) 一致。

## 已重建并实际验证的环境

| 组件 | 本机配置 |
|---|---|
| Python | 3.11.11，用户独立 venv；保留原 Python3.10 venv |
| PyTorch / HIP | 2.12.1+rocm7.2 / 7.2.53211，官方 ROCm wheel |
| FlyDSL | 0.3.1；另建 0.2.2 独立控制环境，未覆盖主环境 |
| Triton | triton-rocm 3.7.1，随 ROCm torch 安装；不是 CUDA wheel |
| pytest / NumPy | 9.1.1 / 2.4.6 |
| pyhip | editable 明确绑定此工作树的 src，而非旧 checkout |
| C++ ABI | 系统 libstdc++ 提供 GLIBCXX_3.4.30，无需替换系统库 |
| AITER | `bde46043bcf08e41ac40395a18369ab6309153ca`，版本 0.1.21.dev11+gbde46043b |
| AITER CK | `15e12dd7f25ee583617c78f66cb502ff9916585f` |

主解释器为 `$HOME/.venvs/pyhip-mha-mi325/bin/python`；0.2.2 控制环境为
`$HOME/.venvs/pyhip-mha-mi325-flydsl022/bin/python`。环境构建使用用户级 `uv`，不提权。
主运行设置为三个 visible-device 变量均为 `0`、`GPU_ARCHS=gfx942`、`MAX_JOBS=4`、
`FLYDSL_RUNTIME_ENABLE_CACHE=0`，清除两个 compile-only 环境变量。
独立 `AITER_JIT_DIR=$HOME/.cache/pyhip-mha-mi325/aiter-jit`；缓存不进入 Git 交付。

**AITER 来源差异**：交接中的 `83faabaa4bf077713c0c71546a8935313c01640b`
无法从公开 ROCm/aiter 获取（Git `not our ref`；提交页面 404）。本轮选择上游 main
升级 FlyDSL 0.3.2 之前的固定提交 `bde46043...`，其自身要求正是 FlyDSL0.3.1。
以 `AITER_USE_SYSTEM_TRITON=1 PREBUILD_KERNELS=0` 安装，保留已验证的 torch/Triton，
启用 CK，而非 Triton-only 模式；没有修改 AITER 源码。**不称为源机 AITER 精确复现**。

编辑器的旧 Python 环境缓存曾使依赖安装工具把包装入旧 venv；已通过显式
`uv pip --python` 和实际 import 路径检查纠正。不能只看编辑器选择或安装成功提示。
工作区解释器设置现已指向新3.11环境，Pylance重新确认选择后相关import诊断已消失。
ROCm 初始化在此工程样片 CPU 上输出 `Invalid processor info`；保留 stderr，未屏蔽错误。
GPU 架构/CU 检查、最小 GPU 运算和 native 测试是独立验证依据。

## 已完成的初始验证

| 检查 | 结果 | 证据 |
|---|---|---|
| 未改源码 CPU 契约 | 78 passed | [CPU JUnit](results/newmachine_mi325_20260907T035439Z/cpu-initial.xml) |
| 初始完整收集 | 2289 collected；不是 2289 passed | [收集记录](results/newmachine_mi325_20260907T035439Z/collect-initial.log) |
| 普通 / gather 性能计划 | 44 / 6 个完整输入 | [普通计划](results/newmachine_mi325_20260907T035439Z/documented-plan.json)、[gather 计划](results/newmachine_mi325_20260907T035439Z/gather-linear-plan.json) |
| BF16/gfx942 native | 205 passed / 46 skipped；含24项固定原版逐位对照 | [BF16 JUnit](results/newmachine_mi325_20260907T035439Z/bf16-942-initial.xml) |
| 初始完整native回归 | **979 passed / 1310 skipped / 0 failed**，3511.97s（含AITER首次构建） | [完整JUnit](results/newmachine_mi325_20260907T035439Z/functional-all-initial.xml) |
| 完整KV gather / CK native | 12 passed / 12 gfx950 skipped | [gather JUnit](results/newmachine_mi325_20260907T035439Z/gather-native-initial.xml) |
| 304 CTA BF16 资源 | 96项无scratch/VGPR spill/AGPR；12项各2个SGPR lane转存 | [per-token](results/newmachine_mi325_20260907T035439Z/resources-bf16-304-per-token.json)、[per-tensor](results/newmachine_mi325_20260907T035439Z/resources-bf16-304-per-tensor.json) |
| 80/304 CTA 同编译器控制 | D128/192 × page32/64：资源与指令流均相同 | [80 CTA 控制](results/newmachine_mi325_20260907T035439Z/resources-bf16-metadata80.json) |
| 原版304 CTA资源控制 | D128/page32,64分别56/104B scratch；D192分别120/176B；当前均0 | [原版资源/ISA](results/newmachine_mi325_20260907T035439Z/resources-bf16-original-304.json) |
| 新增MI325工具契约 | 88 passed，纯CPU | [CPU JUnit](results/newmachine_mi325_20260907T035439Z/cpu-mi325.xml) |
| 双stream长graph / CU counter | **14 passed / 4 gfx950 skipped**，49,152次graph内attention调用 | [stress JUnit](results/newmachine_mi325_20260907T035439Z/stress-native-v2.xml) |

初始完整回归的1310个skip中，1059个为架构不匹配（gfx950 SWA409、BF16327、persistent323）；
其余包含后端专属能力和缺失参考instance，完整原因保留于JUnit。初始压力测试失败是
`torch`不提供FP8 `mul_cuda`的数据准备问题，已改为FP32乘法后cast/copy，保留
[首次失败](results/newmachine_mi325_20260907T035439Z/stress-native.xml)，未更改kernel或容差。

## MI325 同机原版/当前 BF16 对照

四项均同进程、同输入、同编译器、原计时协议；全部输出逐位一致。
“历史形状”仍比较8wave，不把它写成4wave250T复现。

| Workload | 计时 | 原8wave µs | 当前8wave µs | 原TFLOPS | 当前TFLOPS | 延迟变化 |
|---|---|---:|---:|---:|---:|---:|
| D128/H16/Q10240/KV2583/page64 NC | profiler attention | 598.492 | 563.731 | 362.039 | 384.364 | −5.81% |
| D192，同上 | profiler attention | 816.882 | 701.094 | 331.562 | 386.321 | −14.17% |
| D128/H1/Q=KV40960/page32 NC | 10buffer events | 2539.917 | 2450.320 | 338.197 | 350.564 | −3.53% |
| D192/H16/Q10240/KV2583/page32 NC | 10buffer events | 812.853 | 775.335 | 333.206 | 349.329 | −4.62% |

[四项原始样本/完整协议](results/newmachine_mi325_20260907T035439Z/bf16-original-current.json)。
包含首轮重复在内的全部10条原/当前记录已导出为
[TFLOPS表](results/newmachine_mi325_20260907T035439Z/performance-summary.md)、
[CSV](results/newmachine_mi325_20260907T035439Z/performance-summary.csv)、
[JSON](results/newmachine_mi325_20260907T035439Z/performance-summary.json)，每条列出GPU/CU、FlyDSL和计时口径。
本轮有GPU metrics/process监测；CLI检查无其他进程，但仍无外部调度预约证明，不升级为历史正式验收。

## 首轮 D192/page32 诊断

保持原 H16 / Q10240 / KV2583 / D192 / V128 / page32 / NC / per-token / 无LSE，
10 buffer、10 warmup、50 event 样本中位数，输入/源码/编译器均相同。

| 原8-wave µs | 当前8-wave µs | 原8-wave TFLOPS | 当前8-wave TFLOPS | 当前相对延迟 |
|---:|---:|---:|---:|---:|
| 816.714 | 793.714 | 331.630 | 341.240 | −2.82% |

[首轮原始样本](results/newmachine_mi325_20260907T035439Z/bf16-page32-initial.json)。
同时提供[含TFLOPS的可读表](results/newmachine_mi325_20260907T035439Z/performance-initial-summary.md)、
[CSV数据](results/newmachine_mi325_20260907T035439Z/performance-initial-summary.csv)及
[JSON汇总](results/newmachine_mi325_20260907T035439Z/performance-initial-summary.json)。
原版/当前输出逐位相同。**MI308X 的 +10.95% 回退未在此首轮 MI325X 测量重现**；
这不是“已经修复旧机回退”的结论，也不是四wave历史204T基线验收。
GPU进程前后检查为空不等于调度层独占预约，首轮未有全程进程监测。

### TFLOPS 统一口径

所有性能表同时列出每个实际运行后端的 **µs 和有效 TFLOPS**，原始JSON中保留
`effective_flops`、`tflops`、完整shape、计时protocol及raw samples。计算为：

$$
\mathrm{TFLOPS}=\frac{2H_q\sum_b N_{\mathrm{visible},b}(D_{qk}+D_v)}{t_{\mu s}\times10^6}.
$$

本例实际运算量为270,847,180,800 FLOPs。causal/SWA仅计可见QK/PV工作，W128最多129key，
sink不额外计QK/PV。profiler列使用attention主kernel时间，historical events列使用完整event
interval；两者不混算。gather+linear用**包含两段及间隙的单event总时间**计算有效attention
TFLOPS，单独gather的TFLOPS为不适用，不填0。CPU契约、ISA编译、未运行/缺instance的条目也不填吞吐。

[export_performance.py](export_performance.py) 只读指定报告，核对已有TFLOPS与原始运算量/耗时，
输出独立CSV/JSON/Markdown，保留源文件SHA和partial/unavailable状态，拒绝覆盖旧结果。
吞吐导出与性能契约CPU回归共68项通过，见
[throughput-cpu.xml](results/newmachine_mi325_20260907T035439Z/throughput-cpu.xml)。

## 本轮代码变更范围

- 四个生产 kernel **没有修改**；没有 fallback、容差或同步语义变更。
- [compile_bf16_942.py](compile_bf16_942.py) 新增显式 `--workgroups`：默认80保留源机控制，
  MI325用304，counter为305个int32；仍纯meta tensor、不查询GPU。报告新增编译器版本和CTA元数据。
- [_hardware.py](_hardware.py) 对 PTL N/A/缺失/未知提前给出不支持错误，任何 setter 之前失败。
- CPU回归确保 MI325 的高TFLOPS也不能误标成 MI308 gate passed。
- [test_mha_stress.py](test_mha_stress.py)：每个backend/维度双stream、每graph三次调用、
  四轮live metadata/cache更新、每轮256次replay，共6144次launch；检查逐位一致、LSE、
  空请求guard、编译缓存稳定。另检查各page/D配置counter按实际CU分配。
- [export_performance.py](export_performance.py)、[validation_manifest.py](validation_manifest.py)：
  纯离线TFLOPS导出和证据清单，不导入torch，不把原版/CK的baseline标签写成当前版标签。
  最终纯离线测试[15项通过](results/newmachine_mi325_20260907T035439Z/offline-final-v2.xml)。

## FP8阻塞与安全停止

05:22 UTC启动 `reproduce_baselines.py --backend fp8 --ptl current`；超过15分钟没有
`BASELINE_ROUND`。该[报告](results/newmachine_mi325_20260907T035439Z/fp8-original-current-flydsl031.json)
保留 `complete=false`、`records=[]`，**没有FP8 TFLOPS数据**。
不能仅凭日志缺失确定卡在原版/当前版、首次launch、正确性或warmup的哪个阶段。

OS观察显示本轮FP8进程持续约200% CPU，新amd-smi读取进程处于D状态，
等待点为 `amddrm_sched_entity_flush`。确认PID/命令后只向本轮benchmark发送SIGTERM，
并停止本轮两个监测器和阻塞读取进程。随后一次15秒限时健康读取仍exit124；
未执行GPU reset、sudo、功率/时钟/PTL变更，也未杀其他人的进程。最终检查无本轮GPU进程残留。
`dmesg`读取权限不足，未提权，因此**驱动/运行时/kernel根因尚未定位**。

之后一次隐藏GPU的meta审计尝试也在导入阶段进入driver wait，尚未生成任何ISA/JSON；
已停止并删除未经验证的临时工具，保留
[失败日志](results/newmachine_mi325_20260907T035439Z/preservation950-meta.log)。
因此本轮不再执行任何导入ROCm的“CPU-only”命令。前面96项真实编译已在设备正常期间完成，证据不变。

## 完成项与剩余边界

| 交接项 | 本轮结论 |
|---|---|
| 新机环境 | 完成，另保留FlyDSL0.2.2独立环境；AITER使用明确记录的新固定提交 |
| 现有完整native + 新增压力 | 初始979/1310；新增14/4，均0失败；修改后的完整CLI未再做一次合并native运行，不虚构新总通过数 |
| BF16资源 | 96项零scratch；原版4项304CTA资源也重新生成，page64原版104/176B与旧证据一致 |
| D192/page32回退 | MI325两轮当前比原8wave快2.82%/4.62%；未改kernel，不能宣称MI308问题已修复 |
| FP8 400T | MI308/PTL gate在本机不适用；本机FP8计时因阻塞没有完成 |
| BF16 250T/4wave comparator | 没有运行（串行队列因FP8失败中止）；上表350.564T仍是当前8wave的MI325结果 |
| 全44项性能 | 计划完整；仅4个BF16控制case及一次重复有数据，不宣称全矩阵已跑 |
| gather+linear | 完整KV与CK native正确性完成；实现总路径确实调用gather+CK且计时单interval；6项长shape实际计时未完成，不填写TFLOPS |
| gfx950/OPUS | 无原生硬件；本轮新交叉ISA也未完成，旧机compile报告仍只是历史 |
| 独占/硬件恢复 | 无调度层预约；未设置硬件，不涉及PTL恢复；需管理员确认GPU/driver健康后才恢复测试 |

按用户最新选择保留未提交改动，不commit、不push。恢复测试使用新的绝对输出目录，不覆盖本轮失败或历史数据。