# MI308X Attention：三个性能集与10-buffer复现

更新：2026-09-09。**MI308X / gfx942 / 80 CU；PTL Enabled/VECTOR,F8；常规集不测LSE。**
**计算/访存阶段分离版本已完成68项回归。按用户确认的验收范围，D128/D192 MHA long的persistent及普通grid四项均超过250TFLOPS：266.633 / 264.072 / 266.633 / 267.176T。第2节更新为同一完整进程的全部61条性能结果，不拼接试验最优值；full/short仍为237.897–248.660T，不宣称所有shape达250T。**
唯一入口及测试实现均在 [test_mha_pa.py](test_mha_pa.py)。BF16旧实现已删除，以FP8为蓝本重建**8-wave、8-stage、BM256/BN64**内核，普通网格与persistent共用计算核心。当前范围为41项功能、27项性能，61候选、3050个event；新增3个D192 MHA配对shape用于同Q/KV/H/page比较。
BF16 dense和FP8外部BN32适配均已移除，但所有原始shape保留。FP8仅自有LDS；BF16/SWA小shape只测自有kernel。
按用户授权在现有入口扩展BF16功能覆盖及普通网格候选；LSE仅在功能测试校验，不进入性能计时。gfx/UMC均低于3%即可启动、允许驻留进程，结果标为低负载启动/非独占；不把这一门槛改为代码默认值。本次另按授权采集ATT核验阶段排布，没有新增独立测试框架或PMC采集。

## 1. 环境与结果

| 项目 | 现有复现环境 |
|---|---|
| GPU | MI308X / gfx942 / 80 CU / 约192 GiB |
| Python | 3.11.11，使用第6节的已有venv |
| torch / HIP / 系统ROCm | 2.12.1+rocm7.2 / 7.2.53211 / 7.2.1 |
| FlyDSL / Triton | 0.3.1 / triton-rocm 3.7.1 |
| AITER / CK | `bde46043bcf08e41ac40395a18369ab6309153ca` / `15e12dd7f25ee583617c78f66cb502ff9916585f` |
| 本轮设备 | 物理GPU0，BDF `0000:0a:00.0`，选卡后映射为逻辑GPU0并核验 |
| 本轮启动条件 | gfx/UMC均<3%，连续3次、5秒间隔；允许驻留进程，非独占 |

AITER是临时目录中的editable安装；第6节JIT目录的同级aiter源码目录被清理后，需要按上述固定commit恢复，不能静默省略参考。
报告记录实际解释器、torch/HIP/FlyDSL、源码SHA、设备映射和PTL。计时不改变PTL、时钟、功率或NUMA，不reset或终止其他任务。

## 2. 完整性能数据与测试耗时

第2.1–2.3节全部来自同一个功能+性能pytest进程的27份逐case报告，结果标识为mi308_stage_split_20260908.tYs3PU的final。BF16源码SHA256为`ecd7598a97c8af3eae841650366f864ea5b48fd54edda6db39d5001d4baa13c8`，测试SHA256为`2342b520a9481a4da6397498451a4a0c0213170f51f21768444f3828d6e4d60b`。后续回退试验均撤销，当前源码与该轮及最终ATT逐字节相同。
完整列出全部61条候选：BF16 34条、FP8 7条、SWA 20条，包括全部小shape；全部正确性通过。
每行时间为10个独立buffer、5轮共50个event的每调用中位数，`acc`为10组最大值；时间/TFLOPS/逻辑GB/s保留三位小数。
逻辑GB/s不是实测HBM带宽；AITER prepared不含gather，`aiter_gather`包含完整gather+CK。FP8仅自有LDS，无外部参考。

### 2.1 BF16 MHA：全部34条

`bf16_942`为默认persistent固定驻留网格；`bf16_942_grid`为普通head/batch/query-tile网格。两者均为新实现，不是旧版对照或不同计算后端。

| case | backend | acc | 时间µs | TFLOPS | 逻辑GB/s |
|---|---|---:|---:|---:|---:|
| `bf16-smoke-d128` | `bf16_942` | 2.62256e-06 | 18.640 | 3.685 | 32.110 |
| `bf16-smoke-d128` | `bf16_942_grid` | 2.62256e-06 | 18.641 | 3.685 | 32.109 |
| `bf16-smoke-d192` | `bf16_942` | 2.58211e-06 | 20.920 | 4.104 | 35.763 |
| `bf16-smoke-d192` | `bf16_942_grid` | 2.58211e-06 | 21.120 | 4.065 | 35.424 |
| `bf16-full-d128` | `bf16_942` | 2.81277e-06 | 910.806 | 237.897 | 93.553 |
| `bf16-full-d128` | `bf16_942_grid` | 2.81277e-06 | 896.766 | 241.621 | 95.018 |
| `bf16-full-d128` | `aiter` | 2.61054e-06 | 1312.588 | 165.077 | 64.916 |
| `bf16-full-d192` | `bf16_942` | 2.89132e-06 | 1105.747 | 244.945 | 96.325 |
| `bf16-full-d192` | `bf16_942_grid` | 2.89132e-06 | 1123.407 | 241.094 | 94.810 |
| `bf16-full-d192` | `aiter` | 2.67837e-06 | 1602.310 | 169.035 | 66.473 |
| `bf16-causal-d128` | `bf16_942` | 2.39431e-06 | 17841.617 | 246.512 | 15.986 |
| `bf16-causal-d128` | `bf16_942_grid` | 2.39431e-06 | 17891.759 | 245.822 | 15.941 |
| `bf16-causal-d128` | `aiter` | 2.24777e-06 | 26113.932 | 168.423 | 10.922 |
| `bf16-causal-d192` | `bf16_942` | 2.38919e-06 | 21692.624 | 253.438 | 16.435 |
| `bf16-causal-d192` | `bf16_942_grid` | 2.38919e-06 | 21566.103 | 254.924 | 16.531 |
| `bf16-causal-d192` | `aiter` | 2.24661e-06 | 31103.164 | 176.758 | 11.462 |
| `bf16-mha-long-p32` | `bf16_942` | 2.76838e-06 | 6443.263 | 266.633 | 26.038 |
| `bf16-mha-long-p32` | `bf16_942_grid` | 2.76838e-06 | 6505.743 | 264.072 | 25.788 |
| `bf16-mha-long-p32` | `aiter` | 2.68604e-06 | 9952.405 | 172.620 | 16.857 |
| `bf16-mha-short-p32` | `bf16_942` | 2.76931e-06 | 438.783 | 244.709 | 119.487 |
| `bf16-mha-short-p32` | `bf16_942_grid` | 2.76931e-06 | 438.963 | 244.609 | 119.438 |
| `bf16-mha-short-p32` | `aiter` | 2.57288e-06 | 650.205 | 165.139 | 80.634 |
| `bf16-mha-short-p64` | `bf16_942` | 2.76789e-06 | 442.603 | 242.597 | 118.456 |
| `bf16-mha-short-p64` | `bf16_942_grid` | 2.76789e-06 | 438.802 | 244.698 | 119.482 |
| `bf16-mha-short-p64` | `aiter` | 2.57363e-06 | 651.104 | 164.911 | 80.523 |
| `bf16-mha-long-p32-d192` | `bf16_942` | 2.77004e-06 | 8054.074 | 266.633 | 26.038 |
| `bf16-mha-long-p32-d192` | `bf16_942_grid` | 2.77004e-06 | 8037.713 | 267.176 | 26.091 |
| `bf16-mha-long-p32-d192` | `aiter` | 2.68957e-06 | 12083.800 | 177.716 | 17.355 |
| `bf16-mha-short-p32-d192` | `bf16_942` | 2.78746e-06 | 542.564 | 247.377 | 120.790 |
| `bf16-mha-short-p32-d192` | `bf16_942_grid` | 2.78746e-06 | 539.763 | 248.660 | 121.416 |
| `bf16-mha-short-p32-d192` | `aiter` | 2.59008e-06 | 793.485 | 169.150 | 82.593 |
| `bf16-mha-short-p64-d192` | `bf16_942` | 2.78692e-06 | 546.124 | 245.764 | 120.002 |
| `bf16-mha-short-p64-d192` | `bf16_942_grid` | 2.78692e-06 | 540.723 | 248.219 | 121.201 |
| `bf16-mha-short-p64-d192` | `aiter` | 2.59231e-06 | 793.185 | 169.214 | 82.624 |

**本次阶段分离前→最终回归，默认persistent全部12个case。**优化前为此前已发布的D192对齐版本，SHA256为`5ad625b586010220847cb9b99c04b8120a53140d62425c820b24012c535111f8`，结果标识mi308_d192_parity_20260908.DDEkqP的perf-retest。两轮均为GPU0、相同PTL、10-buffer/50样本、低负载启动非独占；分别测量，不混合样本。

| case | 优化前µs | 优化后µs | 延迟下降 | 优化前TFLOPS | 优化后TFLOPS |
|---|---:|---:|---:|---:|---:|
| `bf16-smoke-d128` | 19.000 | 18.640 | 1.89% | 3.615 | 3.685 |
| `bf16-smoke-d192` | 22.401 | 20.920 | 6.61% | 3.833 | 4.104 |
| `bf16-full-d128` | 1002.546 | 910.806 | 9.15% | 216.127 | 237.897 |
| `bf16-full-d192` | 1254.908 | 1105.747 | 11.89% | 215.830 | 244.945 |
| `bf16-causal-d128` | 19470.104 | 17841.617 | 8.36% | 225.894 | 246.512 |
| `bf16-causal-d192` | 24169.751 | 21692.624 | 10.25% | 227.463 | 253.438 |
| `bf16-mha-long-p32` | 7180.425 | 6443.263 | 10.27% | 239.260 | 266.633 |
| `bf16-mha-short-p32` | 486.463 | 438.783 | 9.80% | 220.724 | 244.709 |
| `bf16-mha-short-p64` | 488.123 | 442.603 | 9.33% | 219.974 | 242.597 |
| `bf16-mha-long-p32-d192` | 9135.617 | 8054.074 | 11.84% | 235.067 | 266.633 |
| `bf16-mha-short-p32-d192` | 617.984 | 542.564 | 12.20% | 217.186 | 247.377 |
| `bf16-mha-short-p64-d192` | 610.444 | 546.124 | 10.54% | 219.869 | 245.764 |

全部12个默认persistent case均快于前次发布轮；D128/D192 long延迟分别下降**10.27% / 11.84%**。不将原始基线、各版调优或ATT时长合入当前性能样本。

**同一次最终重测、同shape、同调度的D128/D192吞吐对比。**Dv均128；Dq192使QK+PV算量增加25%，故目标是TFLOPS接近，而不是相同延迟。差值为D192/D128−1。

| 场景（Q/KV，H/HK，page） | persistent D128 T | persistent D192 T | 差值 | 普通网格 D128 T | 普通网格 D192 T | 差值 |
|---|---:|---:|---:|---:|---:|---:|
| full（10240/2583，16/1，64） | 237.897 | 244.945 | +2.96% | 241.621 | 241.094 | −0.22% |
| causal（32768/32768，16/1，64） | 246.512 | 253.438 | +2.81% | 245.822 | 254.924 | +3.70% |
| MHA long（20480/20480，8/8，32） | 266.633 | 266.633 | +0.00% | 264.072 | 267.176 | +1.18% |
| MHA short（10240/2560，8/8，32） | 244.709 | 247.377 | +1.09% | 244.609 | 248.660 | +1.66% |
| MHA short（10240/2560，8/8，64） | 242.597 | 245.764 | +1.31% | 244.698 | 248.219 | +1.44% |

**250T验收范围经用户明确为D128/D192 MHA long、含两种调度，四项全部达标。**两种D的默认persistent吞吐在三位小数处相同，不声称数学上完全相等。若额外把所有非smoke BF16都按250T统计，则只有6/20项达到；full/short仍为237.897–248.660T，不能扩大达标结论。性能结论限定于已测形状和低负载启动/非独占条件。

### 2.2 FP8 MHA：全部7条

| case | backend | acc | 时间µs | TFLOPS | 逻辑GB/s |
|---|---|---:|---:|---:|---:|
| `fp8-full-token-d128` | `fp8_942` | 0.000367516 | 648.885 | 333.923 | 97.977 |
| `fp8-full-token-d192` | `fp8_942` | 0.000387074 | 678.344 | 399.277 | 109.424 |
| `fp8-causal-token-d128` | `fp8_942` | 0.000273883 | 16449.409 | 267.376 | 12.749 |
| `fp8-causal-token-d192` | `fp8_942` | 0.000272198 | 17447.156 | 315.107 | 14.063 |
| `fp8-full-tensor-d128` | `fp8_942` | 0.000363142 | 313.083 | 342.958 | 102.569 |
| `fp8-full-tensor-d192` | `fp8_942` | 0.000383093 | 652.744 | 411.242 | 113.704 |
| `fp8-causal-tensor-d192` | `fp8_942` | 0.000273586 | 17399.356 | 315.973 | 14.102 |

### 2.3 SWA：全部20条（已去重）

已删除两个KV128K重复执行项；保留D128/D192各一个原用例，独立shape覆盖不变。
当前SWA为8个case、20候选，整套为27个case、61候选、3050样本；下表为本轮完整重测，不从旧测量中挑较快值。

| case | backend | acc | 时间µs | TFLOPS | 逻辑GB/s |
|---|---|---:|---:|---:|---:|
| `swa-w0-d128` | `swa_bf16` | 2.86756e-06 | 6.200 | 0.086 | 96.537 |
| `swa-w0-d192` | `swa_bf16` | 2.83579e-06 | 6.320 | 0.105 | 118.380 |
| `swa-kv32768-d128` | `swa_bf16` | 2.76076e-06 | 241.022 | 71.836 | 626.478 |
| `swa-kv32768-d128` | `aiter` | 1.89381e-05 | 510.423 | 33.921 | 295.823 |
| `swa-kv32768-d128` | `aiter_gather` | 1.89381e-05 | 592.804 | 29.207 | 311.316 |
| `swa-kv32768-d192` | `swa_bf16` | 2.75763e-06 | 277.141 | 78.092 | 681.037 |
| `swa-kv32768-d192` | `aiter` | 1.89335e-05 | 567.884 | 38.111 | 332.363 |
| `swa-kv32768-d192` | `aiter_gather` | 1.89335e-05 | 648.964 | 33.349 | 355.469 |
| `swa-kv65536-d128` | `swa_bf16` | 2.75856e-06 | 240.602 | 71.962 | 697.303 |
| `swa-kv65536-d128` | `aiter` | 1.89501e-05 | 511.384 | 33.857 | 328.075 |
| `swa-kv65536-d128` | `aiter_gather` | 1.89501e-05 | 668.465 | 25.901 | 351.374 |
| `swa-kv65536-d192` | `swa_bf16` | 2.76895e-06 | 275.422 | 78.580 | 761.434 |
| `swa-kv65536-d192` | `aiter` | 1.89464e-05 | 568.523 | 38.068 | 368.877 |
| `swa-kv65536-d192` | `aiter_gather` | 1.89464e-05 | 726.425 | 29.793 | 404.173 |
| `swa-kv131072-d128` | `swa_bf16` | 2.76235e-06 | 241.081 | 71.819 | 835.099 |
| `swa-kv131072-d128` | `aiter` | 1.89506e-05 | 509.263 | 33.998 | 395.329 |
| `swa-kv131072-d128` | `aiter_gather` | 1.89506e-05 | 821.946 | 21.065 | 408.232 |
| `swa-kv131072-d192` | `swa_bf16` | 2.75848e-06 | 277.581 | 77.969 | 906.612 |
| `swa-kv131072-d192` | `aiter` | 1.89467e-05 | 570.064 | 37.965 | 441.456 |
| `swa-kv131072-d192` | `aiter_gather` | 1.89467e-05 | 886.366 | 24.417 | 473.203 |

### 2.4 整轮与分类耗时

| suite | case执行数 | 候选数 | event样本数 | case总wall time之和（秒） |
|---|---:|---:|---:|---:|
| `bf16-mha` | 12 | 34 | 1700 | 231.767 |
| `fp8-mha` | 7 | 7 | 350 | 28.209 |
| `swa` | 8 | 20 | 1000 | 6.780 |
| **合计** | **27** | **61** | **3050** | **266.756** |

当前源码同一轮完成41项功能及27项性能：**68通过、0失败、0跳过**，suite **1364.283秒**、进程 **1366.68秒**，其中选卡 **11.353秒**。全部61行、3050个样本、10组输入/输出/workspace独立性、逐buffer精度/逐位复测、指标计算和源码身份均完成CPU审计。BF16 case wall time包含新特化首调/编译；本轮并非纯热缓存计时。
物理GPU0三次启动采样gfx/UMC均0%/0%，间隔5秒；54次case边界均未见其他进程，gfx最高2%、UMC均0%。按授权规则仍标为**低负载启动、非独占**，离散快照不是独占预约或全程无干扰证明。
正门槛只约束开始，不以测试期间采样超过3%为由删样本。ATT另行采集，不用其dispatch时长替换event时间；基线、各次调优和历史受干扰样本均原样保留，不混入本轮表格。
case wall time之和不含功能测试、选卡、组外守卫、环境查询与报告保存，不能再与进程时间相加；也不等于kernel event时间之和。

## 3. 默认10-buffer协议

- `--buffers 10`为缺省值；独立随机输入`seed+i`、Q/K/V、metadata、scale，每候选每组独立O。
  AITER prepared KV和gather workspace也按buffer独立；JSON记录地址和seed。
- `--warmup 10`依次预热10组；`--run-count 5`表示**5个完整buffer轮次**，每轮测完全部10组，
  默认每候选**50个event样本**。不是只分配10组却只计时5组。
- `--repeat 1`为默认；增大时在一对event内逐调用轮转buffer，再按调用次数归一化。
  各候选使用同一索引序列，候选先后顺序交替；重复case标签已删除，不影响通用`--repeat`参数。
- 每buffer独立FP32 O检查、有限值检查、两次额外逐位重复；BF16的`rtol=atol=0.02`，FP8为0.1。
  汇总acc为10组最大值，逐组值记录于`acc_per_buffer`。
- 总结取全部event样本中位数，不选最快值、不删慢样本；小shape的dispatch抖动也保留。
- JIT、FP32参考、量化和预先布局转换在计时外；gather总路径的每次转换在计时内。
  计时覆盖完整调用及launch间隙。新版BF16 persistent无counter分配/初始化或辅助dispatch；旧版基线计时包含这些开销，未人为扣除。`cudaPerf` GPU spin在起始event前，10-buffer不等于强制清cache。

## 4. 参考与SWA路径

| 候选 | 含义 | 计时范围 |
|---|---|---|
| `bf16_942` / `bf16_942_grid` | 同一8-wave/8-stage核心，persistent / 普通网格 | 直接读取SHUFFLE-5D分页KV的完整调用 |
| `fp8_942` / `swa_bf16` | 自有FP8 LDS / BF16单wave | 直接读取SHUFFLE-5D分页KV的完整调用 |
| `aiter` BF16 Full/Causal/MHA | public varlen，当前MI308报告命中ASM | prepared linear KV，不含转换 |
| `aiter` SWA | prepared CK varlen | 不含gather |
| `aiter_gather` SWA | **完整KV gather+CK** | 每次重新gather全部KV再执行CK，**一对event覆盖两段** |

选择规则明确写在 [test_mha_pa.py](test_mha_pa.py) 的三个候选函数中：
BF16每shape均测persistent和普通网格，小shape无外部参考，其余必须AITER；FP8仅自有LDS，不加载任何外部性能参考；SWA小shape无外部参考，长shape必须prepared CK及gather+CK。
这些声明是必需比较项，缺依赖或首调失败会停止，不静默跳过；数值错误始终失败。
BF16 dense及FP8 BN32适配已删除；主文件保留每buffer独立FP32 O正确性检查。
gather不裁SWA前缀，不用两个独立均值相加，不进入生产dispatch；性能路径仍检查完整gather内容、独立workspace及两个dispatch。
gather实现及AITER适配已合入主文件，原辅助模块已删除；`aiter_gather`仍每次执行完整KV搬运，不进入生产dispatch。

## 5. 指标规则

`acc`是归一化平方误差 $\sum(ref-out)^2/\sum(ref^2+out^2)$，越小越好，不是准确率百分比；同时必须通过逐元素检查。
`passed`仅指正确性，不表示达到性能门槛。时间越低越好，TFLOPS只计可见QK/PV，不计mask/padding或sink。

- direct/prepared按逻辑Q/K/V/O计字节，SWA也计完整逻辑KV；FP8输入1字节，BF16输入/O为2字节。
- gather+CK在此基础上加**完整KV读+完整workspace写**，除以总路径时间，不是只算有效窗口。
- GB/s是逻辑字节除对应event时间，**不是实测HBM流量/饱和率**，不包含所有物理重读、metadata或缓存行为。
- 各路径字节分子不同，优劣优先比较同shape延迟，不能只看GB/s大小。
- FP8源为BF16随机数经scale量化为FP8，量化不计时；与旧native-cast/FlyDSL0.2.2/profiler gate不同。

## 6. 当前CLI复现

从Git根、同一个bash终端执行，使用已有环境，无需重装GPU依赖。先完成下面的环境准备，之后每条测试命令均为可直接复制的单行；每轮创建新`OUT`，不覆盖旧证据。

```bash
PY="$HOME/.venvs/pyhip-mha-mi308/bin/python"
MHA=tests/flydsl/mha/test_mha_pa.py
OUT=$(mktemp -d "$PWD/tests/flydsl/mha/results/mi308-three-suites.XXXXXX")
export PYHIP_MHA_GPU=auto PYHIP_MHA_REQUIRED_PTL=VECTOR,F8 PYHIP_MHA_MAX_GPU_UTILIZATION=3
unset HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
export GPU_ARCHS=gfx942 MAX_JOBS=4
export AITER_JIT_DIR=/tmp/cheluo-mi308-full-20260907/aiter-jit
export PATH="$(dirname "$PY"):$PATH"
unset FLYDSL_COMPILE_ONLY FLYDSL_COMPILE_ARCH CUDAPERF

# GPU-free：仅列出显式case、参数和架构限制。
"$PY" "$MHA" --list

# 可选CLI性能整轮：默认suite就是all，同进程执行原有三套case。
"$PY" "$MHA" --suite all --gpu auto --required-ptl VECTOR,F8 --buffers 10 --run-count 5 --warmup 10 --repeat 1 --output "$OUT/all.json"

# 完整回归：同一入口复测全部功能（含授权扩展）及全部性能shape。
PYHIP_MHA_PERF=1 PYHIP_MHA_OUTPUT="$OUT/performance" PYHIP_MHA_SELECTION_LOG="$OUT/selection.idle.jsonl" "$PY" -m pytest "$MHA" -q --junitxml="$OUT/tests.xml"
```

也可单独运行某类或精确ID；下面是互相独立的选择示例，完整ID见第7节：

```bash
"$PY" "$MHA" --suite bf16-mha --output "$OUT/bf16-mha.json"
"$PY" "$MHA" --suite fp8-mha --output "$OUT/fp8-mha.json"
"$PY" "$MHA" --suite swa --output "$OUT/swa.json"

# 原dense比较形状仍在；当前为BF16 persistent、普通网格和AITER三条候选。
"$PY" "$MHA" --suite bf16-mha --case bf16-mha-long-p32 bf16-mha-short-p32 bf16-mha-short-p64 --output "$OUT/mha-pages.json"

# 四个小shape，仅自有kernel，无AITER、BN32或gather。
"$PY" "$MHA" --case bf16-smoke-d128 bf16-smoke-d192 swa-w0-d128 swa-w0-d192 --output "$OUT/own-smoke.json"

# 单个FP8案例及完整逐buffer/样本日志。
"$PY" "$MHA" --suite fp8-mha --case fp8-full-tensor-d192 --verbose-runs --output "$OUT/fp8-tensor-d192.json"
```

**`--case`是可选的场景筛选器**：`--suite`选类别，`--case`从该类别中挑选已定义的场景，不设置shape、不选择单个实现，也不修改参考逻辑。
- 不传`--case`：运行所选suite全部场景；例如`--suite bf16-mha`当前运行12个场景、34条候选。
- 传`--case bf16-full-d128`：只运行Q10240/KV2583、Dq128、H16/HK1、page64、非causal这一场景；在MI308包含BF16 persistent、普通网格与AITER三条候选。
- 多个ID用空格分隔，必须精确、不能重复且属于所选suite；`--suite all`（默认）可跨类别选取。执行顺序沿用参数集合，不随ID输入顺序改变。
- `--list`只列出参数、不运行GPU；可先用它查ID。`--case`属于CLI，pytest单项选择使用pytest节点ID或`-k`，不是此参数。

```bash
# 单场景复现：无需运行整套BF16测试。
"$PY" "$MHA" --suite bf16-mha --case bf16-full-d128 --output "$OUT/bf16-full-d128.json"
# 只查看该类别的可选ID和参数。
"$PY" "$MHA" --suite bf16-mha --list
```

按用户授权在主文件扩展功能回归与BF16普通网格候选；D192性能对齐任务又为已有3个MHA shape各补一个D192配对，原有shape不变，不另建专项测试入口；后续无关扩展仍需确认。
CLI始终执行正确性检查，`buffers/run-count/repeat`必须为正数，`warmup`非负。完整参数表见 [README.md](README.md)。

`--output`可省略；指定时用新的JSON文件。**一份根JSON的`records`包含全部所选case，成功后生成一份同名Markdown**，
没有嵌套manifest或每组子目录。自动选卡另有同名idle JSONL日志，已有输出/日志拒绝覆盖。

CLI默认`--gpu auto`；`--required-ptl VECTOR,F8`只筛选已Enabled/VECTOR,F8的卡，不更改策略。
自动扫描所有合格卡，连续3次、5秒间隔检查。代码默认门槛0要求gfx/UMC均0%且无其他进程；本轮显式设为3，gfx/UMC均严格低于3%即可启动，允许驻留进程。
正门槛只限制选卡/开始，之后记录负载而不因自身GPU活动拒绝结果；标注非独占，不保证测试期间一直低于3%。全部不满足门槛时等待。
用`--gpu-pool 1,2,3`限制物理SMI候选池，或`--gpu 2`固定等待物理GPU2；选中后按ROCr UUID映射并核验BDF。
自动选择会覆盖可见设备变量，调度器环境须限定已分配的卡；用户级选卡锁不是独占预约。
`--gpu current`保留现有可见设备，不做自动等待/PTL筛选，但性能case前后仍有只读进程检查。

## 7. 显式case与pytest范围

以下以主文件中的三个参数集合为准：统一B1、Dv128、contiguous Q/O、零尾页、seed20260905。
除表内注明外，H16/HK1、page64、Q scale为per-token；BF16使用unit descales，FP8从BF16源量化。
NC为full/noncausal，C为bottom-right causal；BF16/FP8 full与causal无窗口、无sink。

### 7.1 BF16 MHA：12 case / 34候选

`test_perf_bf16_mha`使用`BF16_MHA_PERF_CASES`；前两项各2个自有候选，其余10项各`bf16_942`+`bf16_942_grid`+AITER。

| 精确case ID | Q / KV | Dq | H / HK | page | 模式 |
|---|---|---:|---|---:|---|
| `bf16-smoke-d128` | 65 / 129 | 128 | 16 / 1 | 64 | NC，仅自有 |
| `bf16-smoke-d192` | 65 / 129 | 192 | 16 / 1 | 64 | NC，仅自有 |
| `bf16-full-d128` | 10240 / 2583 | 128 | 16 / 1 | 64 | NC |
| `bf16-full-d192` | 10240 / 2583 | 192 | 16 / 1 | 64 | NC |
| `bf16-causal-d128` | 32768 / 32768 | 128 | 16 / 1 | 64 | C |
| `bf16-causal-d192` | 32768 / 32768 | 192 | 16 / 1 | 64 | C |
| `bf16-mha-long-p32` | 20480 / 20480 | 128 | 8 / 8 | 32 | NC，gfx942限定 |
| `bf16-mha-short-p32` | 10240 / 2560 | 128 | 8 / 8 | 32 | NC，gfx942限定 |
| `bf16-mha-short-p64` | 10240 / 2560 | 128 | 8 / 8 | 64 | NC，gfx942限定 |
| `bf16-mha-long-p32-d192` | 20480 / 20480 | 192 | 8 / 8 | 32 | NC，gfx942限定 |
| `bf16-mha-short-p32-d192` | 10240 / 2560 | 192 | 8 / 8 | 32 | NC，gfx942限定 |
| `bf16-mha-short-p64-d192` | 10240 / 2560 | 192 | 8 / 8 | 64 | NC，gfx942限定 |

H8/HK8三个D128形状为原dense比较形状；只删dense参考，没有删形状或改变参数。D192配对仅改变Dq，Dv仍为128，其有效算量是D128的1.25倍；比较TFLOPS而非要求相同延迟。

### 7.2 FP8 MHA：7 case / 7候选

`test_perf_fp8_mha`使用`FP8_MHA_PERF_CASES`；全部gfx942限定，每项仅运行`fp8_942`，不再包含指定BN32参考。
统一HK1/page64，Q scale模式如下，K/V为per-tensor scale。

| 精确case ID | Q / KV | Dq | H | 模式 | Q scale |
|---|---|---:|---:|---|---|
| `fp8-full-token-d128` | 10240 / 2583 | 128 | 16 | NC | per-token |
| `fp8-full-token-d192` | 10240 / 2583 | 192 | 16 | NC | per-token |
| `fp8-causal-token-d128` | 32768 / 32768 | 128 | 16 | C | per-token |
| `fp8-causal-token-d192` | 32768 / 32768 | 192 | 16 | C | per-token |
| `fp8-full-tensor-d128` | 10240 / 2560 | 128 | 8 | NC | per-tensor |
| `fp8-full-tensor-d192` | 10240 / 2560 | 192 | 16 | NC | per-tensor |
| `fp8-causal-tensor-d192` | 32768 / 32768 | 192 | 16 | C | per-tensor |

Full D128两种scale行的H和KV也不同，不能把性能差异仅归因于scale模式。

### 7.3 SWA：8 case / 20候选

`test_perf_swa`使用`SWA_PERF_CASES`；统一BF16、C+sink、H16/HK1/page64/per-token。
前两项只测`swa_bf16`；6个长shape执行各包含`swa_bf16`、`aiter`、`aiter_gather`。

| 精确case ID | Q / KV | Dq | window |
|---|---|---:|---:|
| `swa-w0-d128` | 65 / 129 | 128 | 0 |
| `swa-w0-d192` | 65 / 129 | 192 | 0 |
| `swa-kv32768-d128` | 16384 / 32768 | 128 | 128 |
| `swa-kv32768-d192` | 16384 / 32768 | 192 | 128 |
| `swa-kv65536-d128` | 16384 / 65536 | 128 | 128 |
| `swa-kv65536-d192` | 16384 / 65536 | 192 | 128 |
| `swa-kv131072-d128` | 16384 / 131072 | 128 | 128 |
| `swa-kv131072-d192` | 16384 / 131072 | 192 | 128 |

两个KV128K重复参数项已删除，当前8个case均为独立shape；仍默认`--repeat 1`。
W128最多可见129个key，W0只含当前位置；sink只加一次分母。

### 7.4 普通pytest与性能pytest

本目录的conftest插件已删除。主文件通过性能函数的`__test__`属性控制收集：默认41项功能，`PYHIP_MHA_PERF=1`时额外收集27项性能。
不再需要自定义pytest参数或marker；`-k test_perf_`使用标准pytest功能只选择性能用例。
下表是当前范围；本次功能pytest与性能pytest均已实际运行：

| 功能函数 | 参数项数 | MI308适用性 | 主要检查 |
|---|---:|---|---|
| `test_bf16_mha` | D128/192 × 普通网格/persistent = 4 | 4可运行 | ragged、空Q、尾页、causal、GQA、非单位scale |
| `test_bf16_stage_boundaries` | D128/192 × 5种KV长度 = 10 | 10可运行 | 1/2/3个BN64块的序言、交接、排空 |
| `test_bf16_stage_operands` | 3 | 3可运行 | 均匀Q/V和随机V定向回归，防止跨块页号/搬运错误 |
| `test_bf16_layout_contract` | Dq/Dv四组合 × page32/64/128 = 12 | 12可运行 | 每项覆盖C/NC、两种调度、无LSE热路径与自然对数LSE、NaN尾页、前缀guard、非单位scale、caller O |
| `test_bf16_streams_and_graph` | 2 | 2可运行 | 普通/persistent、超过80个任务、并发stream与graph replay |
| `test_fp8_mha` | D128/192 × C/NC × 两种Q scale = 8 | 8可运行 | ragged、空Q、NaN尾页、量化scale |
| `test_swa` | D128/192 = 2 | 2可运行 | 空KV、W128+sink、padded Q/O、NaN尾页、非单位scale |

最新指定入口的功能+性能合并pytest：**68通过、0跳过**（1364.283秒）；功能41通过、性能27通过，本轮全部性能见第2节。LSE只在功能契约中校验，性能集仍不请求LSE。
本次pipeline没有修改测试入口或shape；此前D192任务已补齐3个配对case。没有新增独立测试框架或恢复独立gather测试；失败和较慢试验的全部原始样本保留。

按第6节准备环境后，选择普通功能测试或显式性能测试：

```bash
PYHIP_MHA_SELECTION_LOG="$OUT/pytest-functional-gpu.jsonl" "$PY" -m pytest "$MHA" --import-mode=importlib -q --junitxml="$OUT/tests.xml"

PYHIP_MHA_PERF=1 PYHIP_MHA_OUTPUT="$OUT/pytest-perf" PYHIP_MHA_SELECTION_LOG="$OUT/pytest-perf-gpu.jsonl" "$PY" -m pytest "$MHA" --import-mode=importlib -k test_perf_ -q
```

性能pytest与CLI共用三套参数和候选规则；收集前设置`PYHIP_MHA_PERF=1`启用性能项，`-k test_perf_`只运行性能项；省略`-k`则同时运行功能测试。
`PYHIP_MHA_OUTPUT`是可选的输出目录，每case一对JSON/Markdown，已有同名文件拒绝覆盖；省略则不落盘，不生成整轮manifest。
pytest用`PYHIP_MHA_GPU`/`PYHIP_MHA_REQUIRED_PTL`选卡；第6节已设为auto/VECTOR,F8，未设置GPU变量时默认current。
列表与收集阶段不初始化GPU。普通功能测试不做性能event计时；性能小shape是27项中的4项，不能另加到41项功能数里。

## 8. 原生缓存、输出与跨机边界

**已启用编译缓存，在排查问题时需要检查缓存是否出现问题**。当前直接使用FlyDSL原生默认缓存，
没有MHA私有持久缓存层、目录重定向或额外CLI开关，不需包装原生缓存。
核对实际FlyDSL版本、`env.runtime.enable_cache`、`env.runtime.cache_dir`以及shell残留的
`FLYDSL_RUNTIME_ENABLE_CACHE`/`FLYDSL_RUNTIME_CACHE_DIR`覆盖，避免旧环境继续禁用或重定向缓存。
源码/依赖变化后的异常输出、加载错误或反复编译，应检查缓存兼容性和失效；首次/新shape编译正常。
实测发现FlyDSL0.3.1的AST改写会将分支移入局部函数，原生建键可能漏掉分支内的全局helper；BF16在分支外显式捕获`body = _body`，并捕获phase内的rescale、预计算地址/访存和归约helper依赖。原生依赖键、实际执行产物IR与ISA均核验。
只记录源文件SHA不足以证明运行了新helper，调优时还需核对实际artifact；不以关闭缓存或添加私有缓存层代替依赖修复。
不缓存正确性结论，不加载不可信缓存，不通过减少10-buffer或跳过检查掩盖问题。

新版以FP8的`_body`/八阶段phase/C-shuffle为蓝本，旧`kv_step`及旧调度框架已删除；BF16适配MFMA32×32×8、P的软件round-half-up和O的软件RNE。
D128/V128用单K槽+双V槽、LDS 49,920字节；D192/V128无LSE热路径为完整K片段、双V槽58,240字节。D128静态双phase展开，D192动态槽号单phase循环。D192 LSE和V192保留已验证K64流式路径（41,600或49,920字节），不外推热路径阶段纯度/性能到这些功能特化。

本次参照gfx950的排布，将稳态循环明确拆分；lane地址提前生成，V保留64-bit SGPR基址+VGPR lane offset的真实GLOBAL读取。上一块softmax随当前QK计算，当前mask/max随上一块PV计算，VALU处理落后相应MFMA数据一块。

| stage | 类别 | 工作 |
|---|---|---|
| S0 | memory | K(t).lo LDS，V(t) GLOBAL，标量页号lookahead |
| S1 | compute | QK(t).lo，上一块24个exp，动态V槽LDS地址 |
| S2 | memory | 等待并发布V(t)，K(t).hi LDS |
| S3 | compute | QK(t).hi，其余8个exp/local sum/P打包，预计算K(t+1)的page32向量地址 |
| S4 | memory | cross32(sum)，V(t−1).lo LDS，K(t+1) BUFFER预取 |
| S5 | compute | PV(t−1).lo，当前mask/local max，完成上一块sum |
| S6 | memory | cross32(max)，V(t−1).hi LDS，等待并发布K(t+1) |
| S7 | compute | PV(t−1).hi，完成max/center/lazy rescale及max寄存器交接 |

`ds_bpermute`从compute移到S4/S6，与独立DS访问共用最后的LGKM等待，下一compute阶段才消费；不是直接删掉等待。单K槽写入仍在两个wave组退役当前K读取之后；`_stage_end`只有编译器栅栏和CTA同步，不替代数据就绪等待。
`persistent=None/True`默认固定最多80个驻留CTA，按grid-stride领取后续工作；**不是旧版动态atomic ticket**，无counter工作区及初始化dispatch。`persistent=False`使用普通网格；计算核心和公共张量格式相同，两种调度均验证并发stream和graph replay。
最终同源persistent long ATT各32个wave完整拼接，所有稳定回边迭代的四个memory阶段0 VALU/MFMA，四个compute阶段0 VMEM/DS。每次工作项首次进入S0前仍有初始化/表示重排：D128为1条VALU，D192为97条（48个shift、48个perm及1个置零），严格按barrier包围统计时共256/24,832条；这些入口指令全部保留报告，不能称序言/排空/输出也纯净。普通grid的独立ATT正在补充。
两种D的persistent trace各包含163,328次主循环cross32；bpermute与共享等待间分别有11–12/12–13条指令，其中8–10/8–11条独立DS，等待stall中位数78/62周期。配对wave的32周期MFMA理想发射窗口占比为93.54%/92.83%，仅为模型，不是PMC busy计数，也不把累计wave stall当作CU空闲。
同源persistent long实际产物：D128 **VGPR228、SGPR78、LDS49,920字节**，D192 **VGPR254、SGPR80、LDS58,240字节**；两者scratch及VGPR/SGPR spill均0。计数来自本次实际编译IR/HSACO，不沿用前次版本；其他形状及LSE/V192不作零spill外推。

V使用64-bit GLOBAL地址协同加载后进入LDS，不创建V buffer描述符。buffer的32-bit相对offset/跨度限制并不要求GPU基地址低于4GiB。
当前gfx942 BF16公开API仍限制单个Q/K/V张量byte span小于2GiB；该限制早于本项优化，本轮没有解除，不能将撤销buffer等同于完整支持>4GiB单张量。

MI308不能验证gfx950原生执行；gfx950另机实测范围见 [README.355.md](README.355.md)（实际MI350X、低负载启动，非独占），不能把skip视作pass。
10-buffer增加显存用量，OOM应如实报告，不自动缩小shape或buffer数。跨机记录见 [changes.md](changes.md)。

JSON的`wall_time_s`区分选卡和执行，每case另列输入准备、FP32参考、参考设置/首调、输出校验、dispatch profiling、warmup和measurement阶段。
首调可能含缓存加载/JIT，不是纯编译时间；`measurement_wall_s`含spin与Python开销，不等于event样本之和。
正常运行不再打印选卡、环境、`PREPARED`/`VALIDATED`/`MEASURED`/`WALL_TIME`或AITER dispatch流程信息。
性能表和`--verbose-runs`逐样本性能信息保持不变；阶段耗时、dispatch、跳过原因仍在JSON中，空闲采样仍在idle JSONL中。
不打印等待进度不代表不等待，仍需3次满足显式启动门槛；完整精度检查不变，`--list`/`--help`显式查询仍有输出。
AITER默认使用ERROR日志级别（显式`AITER_LOG_LEVEL`可覆盖）；profiler只过滤启动/停止及单周期提示，失败时回放原生诊断，不屏蔽其他异常警告/错误。
原测试辅助模块及pytest配置已合并删除；正式目录仍仅保留主测试文件、4个生产kernel、生产共用DSL适配和包初始化。本轮重写gfx942 BF16、在主测试文件增加必要回归；FP8/SWA/gfx950及DSL未因本轮修改，原生缓存不重定向。
临时实验源码打包保存，不在正式目录交付额外脚本。全部原始JSON、JUnit、日志、ISA和旧版基线保留。
历史报告保持原样；第2.1–2.3节当前性能表来自同一最终完整pytest，第2.1独立前后表明确标注阶段分离前基线，同形状对齐表只取最终同一轮，不混合样本。

旧单buffer、dense、register、ISA及日志继续保留在各历史结果中；不得用其数值或计数冒充新范围复测。
