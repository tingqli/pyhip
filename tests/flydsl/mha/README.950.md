# gfx950 Attention：运行、测试与性能

更新：2026-09-07。适用 **MI350X / gfx950，BF16，Dqk128/192，Dv128，page64**。
Full、causal、单wave SWA都用 [test_mha_pa.py](test_mha_pa.py)，默认**先检查输出O，再测性能**。
本文合并覆盖、性能与开发说明；性能数据来自注明时间的实测，当前入口可复测相同输入与流程。

## 1. 快速开始

以下命令均从Git仓库根目录执行。使用已有ROCm环境，不要为运行测试重装GPU依赖。
本机已验证的环境：Python3.10.12、PyTorch2.9.1+rocm7.2.0.git7e1940d4、
HIP7.2.26015-fc0010cf6a、FlyDSL0.3.1；AITER提交`4529fa9c72f06aa7144b2ebed3eb5ff01e3499d1`。

```bash
PY=/opt/venv/bin/python
export HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0
export FLYDSL_RUNTIME_ENABLE_CACHE=0
unset FLYDSL_COMPILE_ONLY FLYDSL_COMPILE_ARCH CUDAPERF

# 默认：8-wave + AITER，D192，Q10240/KV2583，noncausal。
"$PY" tests/flydsl/mha/test_mha_pa.py

# 单wave SWA + AITER；默认Q16K/KV128K，W128，开启sink。
"$PY" tests/flydsl/mha/test_mha_pa.py --backend swa --window 128 --sink

# 自定义causal，比较8-wave static/persistent与AITER。
"$PY" tests/flydsl/mha/test_mha_pa.py \
  --q 32768 --kv 32768 --dq 128 --causal --backend 8wave persistent
```

每次运行显示`acc=... time=... us tflops=... bw=... GB/s`，最后输出中位数表。
`--output`保存全部样本、环境、源码hash及AITER kernel；使用新文件，不能覆盖旧结果。

## 2. 常用参数

| 参数 | 含义 |
|---|---|
| `--check 1` / `--check 0` | 默认1：计时前检查FP32参考。0：只测性能，显示`acc=unchecked`，不算精度通过 |
| `--run-count 0` | 只检查正确性，不计时 |
| `--run-count 5 --warmup 10 --repeat 1` | 计时次数、预热次数、每个event区间的调用数；repeat>1仍报告每次调用时间 |
| `--backend 8wave persistent swa` | 选一个或多个实现；默认8-wave，单wave需显式选`swa` |
| `--q 128 --kv 256 --batch 3` | 三条等长序列，每条Q128/KV256 |
| `--q-lens 33,129 --kv-lens 65,193` | 两条不等长序列；列表优先于对应的`--q/--kv`，两侧列表项数须一致 |
| `--dq 128 192 --heads 16 --kv-heads 1` | Q/K head维度、Q head数、KV head数；Q head数须为KV head数的整数倍 |
| `--causal` | 只允许当前位置及之前的key，Q/KV不等长时按右下角对齐 |
| `--window 128 --sink` | 非负窗口自动开启causal；W128最多129个key，W0仅当前位置。sink只加一次分母 |
| `--aiter auto/on/off` | auto允许缺依赖/不支持时显示N/A；on要求参考可用；off只测自有实现。数值错误始终失败 |
| `--layout padded --poison-tail` | 同一case兼测非连续Q/O和NaN尾页；另可选`head-major` |
| `--query-tile 16 --block-n 32` | 单wave调优参数；默认不展开全tile扫描 |

- `W-1`表示不限制滑动窗口，是否causal由`--causal`决定；`custom_d192`只是“自定义D192 case”的名字。
- 单case选择`swa`或显式非负`--window`时，未指定的Q/KV默认16384/131072；其余默认10240/2583。
- 更多配置可看`--help`，包括scale与输出路径。非单位descales的AITER比较不支持时明确N/A，
  可用`--aiter off`检查自有kernel；不为参考偷偷改变输入值。

## 3. 一条命令复测本文性能表

沿用§1环境。下面是**生成§4全部数据的实际命令**，输出目录名随机，因此可重复执行：

```bash
OUT=$(mktemp -d "$PWD/tests/flydsl/mha/results/readme-repro.XXXXXX")
"$PY" tests/flydsl/mha/test_mha_pa.py \
  --preset basic --dq 128 192 --backend 8wave persistent swa \
  --aiter on --check 1 --run-count 5 --warmup 10 --repeat 1 \
  --output "$OUT/performance.json"
```

统一B1/H16/HK1、BF16/V128/page64、预分配O、unit descales、per-token Q scale、零尾页；
固定随机种子20260905。`basic`只包含以下三类，每类测D128与D192：

| 场景 | 每序列Q / KV | 模式 | 实际候选 |
|---|---|---|---|
| Full | 10240 / 2583 | noncausal | 8-wave static、persistent、AITER |
| Causal | 32768 / 32768 | causal | 8-wave static、persistent、AITER |
| SWA | 16384 / 131072 | W128 + sink | 8-wave static、persistent、单wave、AITER |

共**6个shape、20个候选结果、100个计时run**。只选`--backend swa --preset basic --dq 128 192`
则只跑两个SWA shape。其它batch、窗口、长序列用§2参数直接配置，不再增加另一套功能测试。

### 数据口径

- `acc`是`pyhip.calc_diff`的归一化平方误差，约为 $\sum(ref-out)^2/\sum(ref^2+out^2)$，
  **越小越好，0表示一致**，不是准确率百分比。同时逐元素检查`rtol=atol=0.02`及有限值。
  每个shape的所有候选检查通过后才计时；失败打印`acc`并停止，不能把坏结果当快结果。
- 时间使用`pyhip.cudaPerf` GPU event，每个run单独记录，最终取所有run的中位数。
  区间内包含launch间隙，计时器自带的GPU spin在起始event之前；不包含JIT、FP32参考和布局准备。
- 有效FLOPs为 $2H_q\sum_b N_{visible,b}(D_{qk}+D_v)$，只计真实可见QK/PV，不计mask/padding和sink。
  $\mathrm{TFLOPS}=F/(t_{\mu s}10^6)$；带宽为逻辑Q/K/V/O字节除时间，单位GB/s。
  **带宽不是实测HBM流量**，尤其SWA的逻辑KV字节包含未访问前缀。
- AITER输入是提前准备的linear KV，自有kernel直接读取5D分页KV；转换不计时，不能称为
  完整分页端到端对照。full走公开varlen路由，窗口/sink显式走CK，避免W0/sink被错误路由。

## 4. 性能实测记录

2026-09-07 11:53 UTC，MI350X/gfx950、256 CU；代码起点`2bdffaa`，实际工作树源码hash记录在JSON。
**全部20个候选精度通过**。这是一轮非独占快速诊断，保留原auto-DPM/功耗策略、PTL=N/A。
命令可以复现输入和流程，**不保证共享GPU上的微秒数完全一致**；部分SWA轮次波动明显，
请查看全部样本，不挑最快一轮，也不与旧profiler计时拼接比较。

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

来源：[本次JSON](results/readme-repro.BqIaWy/performance.json)、
[逐run日志](results/readme-repro.BqIaWy/performance.log)。自动生成的结果表与本文来自同一轮，
JSON的`config`、`protocol`、`environment`包含复测所需参数、环境和源码身份。
后续删除旧脚本、精简helper未改内核、输入或计时方式；本表及JSON保留原测量值和源码hash。

| AITER场景 | 本轮实际命中的kernel |
|---|---|
| D128 full / causal | `aiter::fmha_fwd_hd128_bf16_group` / `aiter::fmha_fwd_hd128_bf16_causal_group`（ASM） |
| D192 full / causal | `gqa_d192_v128_kernel<...>`（OPUS） |
| D128 SWA | CK `BlockFmhaPipelineQRKSVSAsyncTrload` |
| D192 SWA | CK `BlockFmhaPipelineQRKSVSAsync` |

不要求AITER必须命中OPUS，也不要求自有实现所有case都快于AITER；以实际输出的kernel与样本为准。
更换AITER/编译器后应重新跑相同命令，不能沿用旧列名认定同一实现。

## 5. 基本功能测试与通过标准

性能命令本身已经检查主shape正确性。日常只运行[test_mha_pa.py](test_mha_pa.py)，
独立的测试工具回归文件已移除，主文件仅保留以下6项边界检查：

| 测试 | 范围 | 本次结果 |
|---|---|---:|
| `test_page_boundaries` | D128/192×8-wave static/persistent；短ragged、空KV、NaN尾页、padded布局、非单位scale | 4通过 |
| `test_swa_page_boundaries` | D128/192单wave；同类边界加W128/sink | 2通过 |

复现当前**6通过、0失败、0跳过**：

```bash
"$PY" -m pytest tests/flydsl/mha/test_mha_pa.py --import-mode=importlib -q --junitxml="$OUT/tests.xml"
```

清理后证据：[JUnit](results/cleanup.jML3Sd/tests.xml)、[日志](results/cleanup.jML3Sd/tests.log)。
目录级pytest也仅收集这6项；旧专项脚本及其测试已删除，不再需要收集排除配置。
另以Q65/KV129、D128/192验证了[full](results/cleanup.jML3Sd/full.json)和
[W0+sink SWA](results/cleanup.jML3Sd/swa.json)：含AITER的14个候选均通过精度检查并输出逐run指标。
这些小形状只验证入口与指标，不作为性能对比；§4数据仍是原始完整性能轮次。

**日常OK**：主shape精度检查通过，以上6项通过，计时有效且报告完整。`--check 0`不是精度通过，
缺依赖的N/A不是已验证。不以单次快测判断小比例性能变化；若要确认优化/回退，在同机同配置下
保留修改前后样本逐case比较。正式性能结论另需独占/负载证据。

## 6. 开发参考与保留事项

- **8-wave**：BM256/BN64、两组4-wave错开执行；packed-V避免重复搬运，PV的
  `traversal_order="mnk"`保持交错顺序。改wait/barrier必须保持DMA依赖与跨任务同步。
- **单wave**：W≤16默认QT16/BN16，其余QT32/BN32；两个16行子tile共享K/V，先QK再PV，
  控制寄存器生命周期；lazy-max和lane-local sum不能改变sink只加一次的语义。
- **边界安全**：只mask概率不足以消除V尾部NaN，0×NaN仍为NaN；保持offset/stride正确。
  单wave并非所有宽窗都最优，新增优化用相应`--window`复测，不据窄窗推断全部窗口。
- **环境**：逻辑GPU0不一定是SMI索引0，本机对应PCI `0000:75:00.0` / SMI索引3。
  不自动设置时钟/功耗/PTL，不终止其它任务；需要空闲检查加`--require-idle`。
- **历史资料**：旧A/B、ISA、OPUS与其它机器的原始结果保留，但不再把多轮表格混入本文。
  原文件hash与当时协议仍以各自JSON为准，旧数量不代表当前小套件已重新运行。
- **gfx942 / MI325**：不属于上面的本机验收；BF16 page32历史差距仍需在对应机器确认，
  FP8与指定参考的验收曾受设备健康问题阻塞，不能视为通过。相关内核仍由主入口选择，
  本次未重新做gfx942原生验证；不要套用gfx950数据或修改共享机器策略来凑门槛。

目录只保留主入口、4个内核、必要helper和两份README，职责见[目录说明](README.md)。
换机需携带当前工作树与依赖；只复制commit会遗漏未提交修改。每次结果另存新目录，
不覆盖历史JSON/JUnit/ISA。