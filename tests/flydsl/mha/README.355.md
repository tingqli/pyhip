# MI350X Attention：三个性能集与10-buffer复现

更新：2026-09-08。**实际设备为MI350X / gfx950 / 256 CU；PTL N/A；不测LSE。**
本文件按要求保留355命名，不是MI355X原生证据。格式与[README.308.md](README.308.md)一致，数据仅使用本机实测。
基线为`lc/tmp-main@4adaf81eeb7d7fe334b4d256c47473a2422aa60f`，唯一入口为[test_mha_pa.py](test_mha_pa.py)。
CLI与性能pytest各自完成**14个适用case、48候选、2400个event样本，全部正确性通过**；完整性能数据直接列于第2节。
功能pytest为**6通过、8跳过**；性能pytest为**14通过、10跳过**。用户允许gfx/UMC低于2%即可开始，
本轮是**低负载启动、允许驻留进程的非独占测试**，不等同于MI308的默认严格空闲条件。

## 1. 环境与结果

| 项目 | 本轮复现环境 |
|---|---|
| GPU | AMD Instinct MI350X / gfx950 / 256 CU |
| Python | 3.10.12，使用第6节的已有venv |
| torch / HIP | 2.9.1+rocm7.2.0.git7e1940d4 / 7.2.26015-fc0010cf6a |
| FlyDSL / Triton | 0.3.1 / 3.6.0+git42270451 |
| AITER / CK | `4529fa9c72f06aa7144b2ebed3eb5ff01e3499d1` / `15e12dd7f25ee583617c78f66cb502ff9916585f` |
| 本轮设备 | SMI GPU0 / PCI `0000:05:00.0` / 逻辑GPU0，三个进程均核验通过 |
| 功耗 / PTL | socket limit 1000 W / PTL N/A，保持已有设置 |
| 负载条件 | gfx和UMC连续3次、5秒间隔均严格低于2%；允许驻留进程 |
| 实测时间 | 2026-09-08 03:50:57–03:53:45 UTC |

AITER使用本机已有安装，不沿用MI308的临时源码/JIT目录，不静默省略参考。
选中时gfx/UMC均0%，各有2条其他进程记录，按新条件允许并记录，不伪称进程为空。
每case前后保存利用率与进程快照；阈值只约束开始时刻，不保证整个测量期间无争用。
不设置PTL、时钟、功率或NUMA，不reset或终止其他任务；用户级选卡锁不是独占预约。

## 2. 完整性能数据与测试耗时

以下全部来自同一轮CLI实测，不混入性能pytest、旧非独占报告或不同轮次的较快样本。
完整列出48条候选：BF16 16条、SWA 32条；FP8仅gfx942适用，本机没有FP8性能数据。
每行是10个独立buffer、5轮共50个event的每调用中位数，`acc`取10组最大值；时间/TFLOPS/逻辑GB/s保留三位小数。
全部48条均正确性通过；`passed`不等于性能达标，逻辑GB/s也不是实测HBM带宽。

### 2.1 BF16 MHA：全部16条

B1/H16/HK1/Dv128/page64/per-token/unit descales；smoke为Q65/KV129，仅自有；
Full为Q10240/KV2583，Causal为Q32768/KV32768，两类均必须比较AITER。

| case | backend | acc | 时间µs | TFLOPS | 逻辑GB/s |
|---|---|---:|---:|---:|---:|
| `bf16-smoke-d128` | `bf16_950` | 2.62256e-06 | 12.521 | 5.486 | 47.802 |
| `bf16-smoke-d128` | `bf16_950_persistent` | 2.62256e-06 | 13.200 | 5.204 | 45.343 |
| `bf16-smoke-d192` | `bf16_950` | 2.58826e-06 | 13.281 | 6.465 | 56.333 |
| `bf16-smoke-d192` | `bf16_950_persistent` | 2.58826e-06 | 14.360 | 5.979 | 52.100 |
| `bf16-full-d128` | `bf16_950` | 2.83967e-06 | 221.348 | 978.903 | 384.954 |
| `bf16-full-d128` | `bf16_950_persistent` | 2.83967e-06 | 228.348 | 948.893 | 373.152 |
| `bf16-full-d128` | `aiter` | 2.63913e-06 | 216.268 | 1001.895 | 393.995 |
| `bf16-full-d192` | `bf16_950` | 2.79416e-06 | 255.689 | 1059.284 | 416.564 |
| `bf16-full-d192` | `bf16_950_persistent` | 2.79416e-06 | 265.990 | 1018.261 | 400.431 |
| `bf16-full-d192` | `aiter` | 2.79417e-06 | 252.209 | 1073.900 | 422.311 |
| `bf16-causal-d128` | `bf16_950` | 2.40834e-06 | 4107.284 | 1070.825 | 69.441 |
| `bf16-causal-d128` | `bf16_950_persistent` | 2.40834e-06 | 4117.584 | 1068.146 | 69.267 |
| `bf16-causal-d128` | `aiter` | 2.27852e-06 | 4071.903 | 1080.129 | 70.044 |
| `bf16-causal-d192` | `bf16_950` | 2.3923e-06 | 4874.211 | 1127.921 | 73.143 |
| `bf16-causal-d192` | `bf16_950_persistent` | 2.3923e-06 | 4864.631 | 1130.142 | 73.287 |
| `bf16-causal-d192` | `aiter` | 2.39228e-06 | 4815.149 | 1141.756 | 74.040 |

### 2.2 FP8 MHA：本机不适用

`FP8_MHA_PERF_CASES`的7个场景均限定gfx942；本机gfx950全部记架构不适用，不计通过或性能值。
没有改成BF16、替换shape或使用compile-only冒充原生验证。具体场景见第7.2节。

### 2.3 SWA：全部32条（已去重）

统一B1/H16/HK1/Dv128/page64/per-token/causal+sink；W0为Q65/KV129，仅单wave。
长shape为Q16384/W128，KV为32768/65536/131072，分别比较static、persistent、单wave、prepared CK和完整gather+CK。

| case | backend | acc | 时间µs | TFLOPS | 逻辑GB/s |
|---|---|---:|---:|---:|---:|
| `swa-w0-d128` | `swa_bf16` | 2.86742e-06 | 6.360 | 0.084 | 94.108 |
| `swa-w0-d192` | `swa_bf16` | 2.87153e-06 | 6.320 | 0.105 | 118.380 |
| `swa-kv32768-d128` | `bf16_950` | 2.72183e-06 | 101.584 | 170.442 | 1486.412 |
| `swa-kv32768-d128` | `bf16_950_persistent` | 2.72183e-06 | 88.284 | 196.119 | 1710.342 |
| `swa-kv32768-d128` | `swa_bf16` | 2.76338e-06 | 71.263 | 242.960 | 2118.841 |
| `swa-kv32768-d128` | `aiter` | 2.76339e-06 | 112.464 | 153.952 | 1342.607 |
| `swa-kv32768-d128` | `aiter_gather` | 2.76339e-06 | 143.705 | 120.484 | 1284.224 |
| `swa-kv32768-d192` | `bf16_950` | 2.72542e-06 | 119.824 | 180.620 | 1575.174 |
| `swa-kv32768-d192` | `bf16_950_persistent` | 2.72542e-06 | 112.564 | 192.269 | 1676.768 |
| `swa-kv32768-d192` | `swa_bf16` | 2.76497e-06 | 80.943 | 267.381 | 2331.810 |
| `swa-kv32768-d192` | `aiter` | 2.76496e-06 | 123.524 | 175.210 | 1527.992 |
| `swa-kv32768-d192` | `aiter_gather` | 2.76496e-06 | 161.706 | 133.839 | 1426.581 |
| `swa-kv65536-d128` | `bf16_950` | 2.71792e-06 | 101.804 | 170.074 | 1648.000 |
| `swa-kv65536-d128` | `bf16_950_persistent` | 2.71792e-06 | 88.403 | 195.854 | 1897.811 |
| `swa-kv65536-d128` | `swa_bf16` | 2.75836e-06 | 71.682 | 241.539 | 2340.490 |
| `swa-kv65536-d128` | `aiter` | 2.75836e-06 | 111.984 | 154.612 | 1498.180 |
| `swa-kv65536-d128` | `aiter_gather` | 2.75836e-06 | 174.526 | 99.206 | 1345.823 |
| `swa-kv65536-d192` | `bf16_950` | 2.72396e-06 | 117.825 | 183.685 | 1779.895 |
| `swa-kv65536-d192` | `bf16_950_persistent` | 2.72396e-06 | 113.244 | 191.115 | 1851.888 |
| `swa-kv65536-d192` | `swa_bf16` | 2.76427e-06 | 80.883 | 267.579 | 2592.822 |
| `swa-kv65536-d192` | `aiter` | 2.76426e-06 | 124.004 | 174.532 | 1691.197 |
| `swa-kv65536-d192` | `aiter_gather` | 2.76426e-06 | 194.566 | 111.235 | 1509.002 |
| `swa-kv131072-d128` | `bf16_950` | 2.7254e-06 | 99.323 | 174.320 | 2026.978 |
| `swa-kv131072-d128` | `bf16_950_persistent` | 2.7254e-06 | 88.583 | 195.456 | 2272.745 |
| `swa-kv131072-d128` | `swa_bf16` | 2.76839e-06 | 71.742 | 241.337 | 2806.239 |
| `swa-kv131072-d128` | `aiter` | 2.76837e-06 | 111.944 | 154.668 | 1798.466 |
| `swa-kv131072-d128` | `aiter_gather` | 2.76837e-06 | 235.949 | 73.381 | 1422.108 |
| `swa-kv131072-d192` | `bf16_950` | 2.7224e-06 | 116.364 | 185.991 | 2162.681 |
| `swa-kv131072-d192` | `bf16_950_persistent` | 2.7224e-06 | 112.864 | 191.758 | 2229.748 |
| `swa-kv131072-d192` | `swa_bf16` | 2.76336e-06 | 80.663 | 268.309 | 3119.872 |
| `swa-kv131072-d192` | `aiter` | 2.76334e-06 | 123.865 | 174.728 | 2031.722 |
| `swa-kv131072-d192` | `aiter_gather` | 2.76334e-06 | 259.410 | 83.430 | 1616.866 |

来源：[本轮CLI原始JSON](results/gfx950-lowload2-20260908.yFE1Zz/all.json#L1)，保留全部样本及实际dispatch。

### 2.4 整轮与分类耗时

| suite | case执行数 | 候选数 | event样本数 | case总wall time之和（秒） |
|---|---:|---:|---:|---:|
| `bf16-mha` | 6 | 16 | 800 | 21.634 |
| `fp8-mha` | 0 | 0 | 0 | 0.000 |
| `swa` | 8 | 32 | 1600 | 5.258 |
| **合计** | **14** | **48** | **2400** | **26.891** |

另有3个BF16 MHA和7个FP8架构不适用case；这里的0不是FP8性能结果。总wall time由未舍入值求和。

| 步骤 | 进程总时间（秒） | 内部计时（秒） |
|---|---:|---|
| CLI | 46.429 | 选卡+激活11.232；执行33.420 |
| 功能pytest | 24.682 | JUnit suite23.440，含选卡setup |
| 性能pytest | 97.001 | JUnit suite95.016，含选卡setup |
| 三个成功进程合计 | 168.113 | 不包含之前1266.585秒的旧严格等待或文档整理 |

分类wall time之和不含case外环境查询、负载快照和保存；进程时间已包含选卡，不能再次累加。
性能pytest逐case查询环境并保存报告，开销与CLI不同，不能将其当作kernel性能差异；上述不是开发会话总耗时。

- Full/Causal static比AITER慢约**0.87%–2.35%**，本轮接近；persistent无普遍收益，Full慢约5.46%–5.59%。
- 6组长SWA单wave比prepared AITER时延低**34.47%–36.64%**，比完整gather+AITER低**49.94%–69.59%**。
- 这些是本轮低负载启动结果；不保证后续始终低于2%，不据小幅差异给稳定性能或独占验收结论。

## 3. 默认10-buffer协议

- `--buffers 10`为缺省值；独立随机输入`seed+i`、Q/K/V、metadata、scale，每候选每buffer独立O。
  AITER prepared KV和gather workspace也按buffer独立；JSON记录地址和seed。
- `--warmup 10`依次预热10组；`--run-count 5`是5个完整轮次，每轮测完buffer0–9，
  每候选50个event样本，不是只分配10组却只测5组。
- `--repeat 1`为默认；增大时在一对event内逐调用轮转buffer，再除以调用数。
  各候选使用同一索引序列，候选先后顺序交替；不恢复已移除的独立repeat3专项。
- 每buffer检查独立FP32 O、有限值及BF16 `rtol=atol=0.02`，再做两次额外逐位重复。
  汇总acc为10组最大值，逐组值保存在`acc_per_buffer`；不测LSE。
- 取全部event的每调用中位数，不选最快值、不删慢样本。10-buffer不是强制清cache，不宣称完全冷cache。
- JIT、FP32参考、布局准备及workspace分配在计时外；gather的逐次转换、调用内counter初始化、辅助launch和间隙在event内。
  `cudaPerf`自带GPU spin在起始event之前，不把阶段wall time当成kernel延迟。

## 4. 参考与SWA路径

| 候选 | 含义 | 计时范围 |
|---|---|---|
| `bf16_950` / `bf16_950_persistent` / `swa_bf16` | 自有static / persistent / 单wave | 直接读取SHUFFLE-5D分页KV的完整调用 |
| `aiter` BF16 Full/Causal | AITER public varlen | prepared linear KV，不含转换 |
| `aiter` SWA | prepared CK varlen | 不含gather |
| `aiter_gather` SWA | 完整KV gather+CK | 每次重新gather全部KV后执行CK，一对event覆盖两段 |

候选由主文件中的`bf16_mha_candidates`和`swa_candidates`决定：小shape仅自有，长shape的AITER参考为必需项。
缺依赖、首调或数值失败时停止，不静默减少候选或更换shape。BF16 dense/FP8 BN32性能适配已删除，独立FP32 O仍保留。
gather不裁SWA前缀、不缓存输出，不把两个独立均值相加；运行时检查完整gather内容、独立workspace及两个dispatch。
gather只用于测试参考，不进入生产dispatch。

| 场景 | 本轮实际命中的AITER kernel |
|---|---|
| D128 Full / Causal | `aiter::fmha_fwd_hd128_bf16_group` / `aiter::fmha_fwd_hd128_bf16_causal_group`（ASM） |
| D192 Full / Causal | `gqa_d192_v128_kernel<...>`（OPUS） |
| D128 SWA W128 | CK `BlockFmhaPipelineQRKSVSAsyncTrload` |
| D192 SWA W128 | CK `BlockFmhaPipelineQRKSVSAsync` |
| `aiter_gather` | `_gather_kv`加对应CK，共2个dispatch |

完整名称保存在JSON/自动结果表，不将所有AITER统称为OPUS。prepared参考不是完整分页端到端性能；
只有`aiter_gather`包含这里定义的完整gather+CK路径。

## 5. 指标规则

`acc`是归一化平方误差 $\sum(ref-out)^2/\sum(ref^2+out^2)$，越小越好，不是准确率百分比；同时必须通过逐元素检查。
`passed`仅指正确性，不表示达到性能门槛。时间越低越好，TFLOPS只计可见QK/PV，不计mask/padding或sink。

- 有效FLOPs为 $2H_q\sum_b N_{visible,b}(D_{qk}+D_v)$，$\mathrm{TFLOPS}=F/(t_{\mu s}10^6)$。
- direct/prepared按逻辑Q/K/V/O计字节，BF16输入和O均2字节；SWA也计完整逻辑KV。
- gather+CK在此基础上加完整KV读和完整workspace写，除以总路径时间；不是只算可见窗口。
- GB/s是逻辑字节除时间，不是实测HBM流量/饱和率，不包含所有物理重读、metadata或缓存行为。
  各路径字节分子不同，优劣优先比较同shape延迟，不能只看GB/s大小。

## 6. 当前CLI复现

从Git根、同一个bash终端执行，使用已有ROCm环境，无需重装GPU依赖。先准备环境，再复制单行测试命令；
每轮新建OUT，不覆盖旧证据。本机PTL为N/A，使用`current`保留策略，不套用MI308的Enabled/VECTOR,F8条件。

```bash
PY=/opt/venv/bin/python
MHA=tests/flydsl/mha/test_mha_pa.py
OUT=$(mktemp -d "$PWD/tests/flydsl/mha/results/gfx950-lowload2-20260908.XXXXXX")
export PATH="$(dirname "$PY"):$PATH"
export GPU_ARCHS=gfx950 MAX_JOBS=4 PYHIP_MHA_GPU=auto PYHIP_MHA_REQUIRED_PTL=current
export PYHIP_MHA_MAX_GPU_UTILIZATION=2
unset HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
unset FLYDSL_COMPILE_ONLY FLYDSL_COMPILE_ARCH CUDAPERF FLYDSL_RUNTIME_ENABLE_CACHE FLYDSL_RUNTIME_CACHE_DIR
unset PYHIP_MHA_COMPILE_CACHE PYHIP_MHA_PERF PYHIP_MHA_OUTPUT PYHIP_MHA_SELECTION_LOG

# GPU-free：仅列出显式case、参数和架构限制。
"$PY" "$MHA" --list
# 推荐整轮：选卡一次，同进程执行所有适用case。
"$PY" "$MHA" --suite all --gpu auto --required-ptl current --max-gpu-utilization 2 --buffers 10 --run-count 5 --warmup 10 --repeat 1 --output "$OUT/all.json"
```

也可按类别或精确ID选择，以下是独立的选择示例，不必在整轮之后重复执行：

```bash
"$PY" "$MHA" --suite bf16-mha --output "$OUT/bf16-mha.json"
"$PY" "$MHA" --suite swa --output "$OUT/swa.json"
"$PY" "$MHA" --suite bf16-mha --case bf16-full-d128 --output "$OUT/bf16-full-d128.json"
"$PY" "$MHA" --suite swa --case swa-kv32768-d128 swa-kv32768-d192 --output "$OUT/swa-32k.json"
"$PY" "$MHA" --case bf16-smoke-d128 bf16-smoke-d192 swa-w0-d128 swa-w0-d192 --output "$OUT/own-smoke.json"
"$PY" "$MHA" --suite bf16-mha --list
```

**`--case`是可选筛选器**：`--suite`选择类别，不传`--case`则运行该类别全部预定义场景；传入ID只筛选场景，
不改变shape、候选或参考。ID必须精确、互不重复且属于所选suite，执行顺序仍按参数集合。
例如`bf16-full-d128`在本机包含static/persistent/AITER三条候选，不是只选某个实现。
自定义shape应编辑显式`Workload`，不再使用旧backend/preset/shape开关；CLI始终检查正确性。

`--output`可省略；指定时一份根JSON的`records`保存全部所选case，成功后生成同名Markdown。
自动选卡另存同名idle JSONL。已有输出/日志拒绝覆盖，详细阶段、设备映射、实际kernel和全部样本均保留。

本轮显式`--max-gpu-utilization 2`：扫描任意合格GPU，每卡连续3次、5秒间隔gfx/UMC均严格低于2%后启动，
允许驻留进程，阈值只约束开始时刻。case前后记录利用率/进程，不以自身测试负载触发重新等待。
未设门槛时默认0，仍要求零利用率且无其他进程；CLI可用`--max-gpu-utilization 0`，pytest用`PYHIP_MHA_MAX_GPU_UTILIZATION=0`恢复严格模式。

`--gpu-pool`可限定授权物理SMI卡池，`--gpu N`固定等待某卡；正门槛不能配合`current`绕过起始负载检查。
选中后按ROCr UUID映射并核验BDF，进程内不换卡。`GPU_ARCHS`仅是编译设置，不代替调度器设备池授权。
用户级文件锁不是独占预约；命令复现输入和流程，不保证共享GPU上的微秒数完全一致。

## 7. 显式case与pytest范围

以主文件三个参数集为准：统一B1、Dv128、contiguous Q/O、零尾页、seed20260905；除注明外H16/HK1/page64/per-token。
NC为full/noncausal，C为bottom-right causal；BF16 full/causal无窗口、无sink，SWA为C+sink。

### 7.1 BF16 MHA：9个声明case，6个适用 / 16候选

`test_perf_bf16_mha`使用`BF16_MHA_PERF_CASES`；smoke各static/persistent两条，常规full/causal各另加AITER。

| 精确case ID | Q / KV | Dq | H / HK | page | gfx950适用性 |
|---|---|---:|---|---:|---|
| `bf16-smoke-d128` | 65 / 129 | 128 | 16 / 1 | 64 | NC，仅自有 |
| `bf16-smoke-d192` | 65 / 129 | 192 | 16 / 1 | 64 | NC，仅自有 |
| `bf16-full-d128` | 10240 / 2583 | 128 | 16 / 1 | 64 | NC，含AITER |
| `bf16-full-d192` | 10240 / 2583 | 192 | 16 / 1 | 64 | NC，含AITER |
| `bf16-causal-d128` | 32768 / 32768 | 128 | 16 / 1 | 64 | C，含AITER |
| `bf16-causal-d192` | 32768 / 32768 | 192 | 16 / 1 | 64 | C，含AITER |
| `bf16-mha-long-p32` | 20480 / 20480 | 128 | 8 / 8 | 32 | gfx942限定，跳过 |
| `bf16-mha-short-p32` | 10240 / 2560 | 128 | 8 / 8 | 32 | gfx942限定，跳过 |
| `bf16-mha-short-p64` | 10240 / 2560 | 128 | 8 / 8 | 64 | gfx942限定，跳过 |

最后三项按当前集合均限定gfx942，包括page64项；没有为了本机增加额外dense-page64或改变架构限制。

### 7.2 FP8 MHA：7个声明case，本机全部跳过

`test_perf_fp8_mha`使用`FP8_MHA_PERF_CASES`；全部仅gfx942的自有LDS，无外部BN32性能参考。

| 精确case ID | Q / KV | Dq | H | 模式 | Q scale |
|---|---|---:|---:|---|---|
| `fp8-full-token-d128` | 10240 / 2583 | 128 | 16 | NC | per-token |
| `fp8-full-token-d192` | 10240 / 2583 | 192 | 16 | NC | per-token |
| `fp8-causal-token-d128` | 32768 / 32768 | 128 | 16 | C | per-token |
| `fp8-causal-token-d192` | 32768 / 32768 | 192 | 16 | C | per-token |
| `fp8-full-tensor-d128` | 10240 / 2560 | 128 | 8 | NC | per-tensor |
| `fp8-full-tensor-d192` | 10240 / 2560 | 192 | 16 | NC | per-tensor |
| `fp8-causal-tensor-d192` | 32768 / 32768 | 192 | 16 | C | per-tensor |

统一HK1/Dv128/page64。单选FP8时本机没有可运行候选，CLI失败，不把空报告当作通过。

### 7.3 SWA：8 case / 32候选

`test_perf_swa`使用`SWA_PERF_CASES`；W0各仅单wave，6个长shape各包含static/persistent/单wave/prepared CK/gather+CK。

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

W128最多可见129个key，W0仅当前位置；sink只加一次分母。重复KV128K标签已删除，不增加独立shape覆盖。

### 7.4 普通pytest与性能pytest

默认仅收集14项功能；`PYHIP_MHA_PERF=1`额外启用24项性能，`-k test_perf_`只选择性能。
普通功能测试不做event计时；4个性能小shape不能另加到功能用例数里。

| 功能函数 | 参数项数 | gfx950本轮结果 | 主要检查 |
|---|---:|---|---|
| `test_bf16_mha` | 4 | 4通过 | D128/192×static/persistent；ragged、空Q/KV、NaN尾页、padded、GQA、非单位scale |
| `test_fp8_mha` | 8 | 8跳过 | 仅gfx942的D128/192×C/NC×两种Q scale |
| `test_swa` | 2 | 2通过 | D128/192单wave；W128/sink、空KV、padded、NaN尾页、非单位scale |

功能合计6通过/8跳过/0失败；性能pytest14通过/10架构跳过/0失败，包含48候选和2400个event。
按第6节环境运行，三个测试进程依次执行，不与CLI同时启动：

```bash
PYHIP_MHA_SELECTION_LOG="$OUT/functional-gpu.jsonl" "$PY" -m pytest "$MHA" --import-mode=importlib -q --durations=0 --junitxml="$OUT/functional.xml"
PYHIP_MHA_PERF=1 PYHIP_MHA_OUTPUT="$OUT/pytest-perf" PYHIP_MHA_SELECTION_LOG="$OUT/performance-pytest-gpu.jsonl" "$PY" -m pytest "$MHA" --import-mode=importlib -k test_perf_ -q --durations=0 --junitxml="$OUT/performance-pytest.xml"
```

pytest使用`PYHIP_MHA_GPU`、`PYHIP_MHA_REQUIRED_PTL`及`PYHIP_MHA_MAX_GPU_UTILIZATION`，第6节已设为auto/current/2。
每个进程独立选卡，实际设备必须以日志为准；固定卡可设`PYHIP_MHA_GPU=N`。收集阶段不初始化GPU。
`PYHIP_MHA_OUTPUT`可选，指定时每case一对JSON/Markdown；已有同名文件拒绝覆盖，没有额外插件或自定义pytest参数。

## 8. 原生缓存、输出与跨机边界

**已启用编译缓存，在排查问题时需要检查缓存是否出现问题**。当前使用FlyDSL原生默认缓存，
没有MHA私有持久缓存层、目录重定向或额外CLI缓存开关。核对版本、`env.runtime.enable_cache`、`env.runtime.cache_dir`
及shell的`FLYDSL_RUNTIME_ENABLE_CACHE`/`FLYDSL_RUNTIME_CACHE_DIR`覆盖；不要通过关闭缓存、减少buffer或跳过正确性掩盖问题。
本轮确认缓存开启，不宣称全部命中；首调仍可能含缓存加载或JIT，不等于纯编译时间。

正常运行只显示性能表；`--verbose-runs`显示每个样本，阶段耗时/设备/dispatch仍保存在JSON。
不打印等待进度不代表没有等待；错误诊断继续输出。所有shape使用原规模，未因显存或参考支持情况自动缩小。

### 为完成本轮测试实际需要的修改

相对`4adaf81`，运行代码只需在主测试文件支持用户指定的“低于2%即可开始”；不是为了修复kernel精度或性能：

| 位置 | 修改目的 |
|---|---|
| `parse_args` / `native_gpu_selection` | 接入CLI `--max-gpu-utilization`和pytest环境变量；默认0仍保留严格空闲 |
| `_gpu_activity` / `_gpu_load_ready` / `select_idle_gpu` | 读取并校验gfx/UMC，连续3次严格低于门槛时允许驻留进程；保留PTL/BDF/锁验证 |
| `ensure_idle` / `require_idle_device` / `activate_selected_gpu` | 低负载模式不再因驻留进程被旧守卫拒绝；记录启动及case边界负载 |
| `run` | 保存门槛、其他进程和前后快照，JSON/Markdown明确标记低负载启动、非独占 |

生产4个kernel和[_dsl.py](_dsl.py)无需修改，已与基线逐字节核对；case集合、输入/FP32 O、gather、精度和event计时循环均未变。
无需恢复已删除的旧helper，也不需要改用例架构、重装依赖或设置硬件策略；剩余改动只是平台文档和[changes.md](changes.md)同步。

原始JSON、JUnit、选卡日志与[完整验证记录](results/gfx950-lowload2-20260908.yFE1Zz/validation.json#L1)保留，
[进程耗时](results/gfx950-lowload2-20260908.yFE1Zz/run-status.json#L1)不含之前已中断的严格等待。
旧单buffer、dense、repeat3及其它轮次原始结果仍保留，不再把其表格和失效命令混入当前README。
本次文档整理不重跑未修改的GPU代码、不改写历史数值；低负载启动不等于独占，MI355X与gfx942不在本机原生验证范围。