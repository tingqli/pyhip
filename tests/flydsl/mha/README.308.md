# MI308X Attention：三个性能集与10-buffer复现

更新：2026-09-08。**MI308X / gfx942 / 80 CU；PTL Enabled/VECTOR,F8；常规集不测LSE。**
唯一常规入口及测试实现均在 [test_mha_pa.py](test_mha_pa.py)。生产kernel已应用页号异步预取和V地址偏移复用；本轮仅用该入口复测原有功能与性能项，**36通过、2跳过**，其中24个性能case、43候选、2150个event全部通过。完整本轮性能表见第2.1–2.3节。
BF16 dense和FP8外部BN32适配均已移除，但所有原始shape保留。FP8仅自有LDS；BF16/SWA小shape只测自有kernel。
按用户授权，gfx/UMC均低于3%即可启动、允许驻留进程，结果标为低负载启动/非独占；不把这一门槛改为代码默认值。**后续扩展测试范围必须先确认**，不自行增加shape、参考、LSE专项或ATT。

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

第2.1–2.3节来自本轮同一个pytest进程、指定入口生成的24份逐case报告，未混入前轮CLI或专项数值。生产源码SHA256为`84feac1fb38f834adedd48327998cf582af36803f2cb01cbba320dc0c772804f`。
完整列出全部43条候选：BF16 16条、FP8 7条、SWA 20条，包括全部小shape；全部正确性通过。
每行时间为10个独立buffer、5轮共50个event的每调用中位数，`acc`为10组最大值；时间/TFLOPS/逻辑GB/s保留三位小数。
逻辑GB/s不是实测HBM带宽；AITER prepared不含gather，`aiter_gather`包含完整gather+CK。FP8仅自有LDS，无外部参考。

### 2.1 BF16 MHA：全部16条

| case | backend | acc | 时间µs | TFLOPS | 逻辑GB/s |
|---|---|---:|---:|---:|---:|
| `bf16-smoke-d128` | `bf16_942` | 2.69056e-06 | 29.360 | 2.340 | 20.386 |
| `bf16-smoke-d192` | `bf16_942` | 2.68794e-06 | 35.260 | 2.435 | 21.218 |
| `bf16-full-d128` | `bf16_942` | 2.81978e-06 | 1049.347 | 206.488 | 81.202 |
| `bf16-full-d128` | `aiter` | 2.61054e-06 | 1311.809 | 165.175 | 64.955 |
| `bf16-full-d192` | `bf16_942` | 2.89635e-06 | 1733.331 | 156.258 | 61.449 |
| `bf16-full-d192` | `aiter` | 2.67837e-06 | 1602.411 | 169.025 | 66.469 |
| `bf16-causal-d128` | `bf16_942` | 2.4229e-06 | 20315.752 | 216.491 | 14.039 |
| `bf16-causal-d128` | `aiter` | 2.24777e-06 | 26093.030 | 168.558 | 10.931 |
| `bf16-causal-d192` | `bf16_942` | 2.41583e-06 | 34057.159 | 161.426 | 10.468 |
| `bf16-causal-d192` | `aiter` | 2.24661e-06 | 31095.223 | 176.803 | 11.465 |
| `bf16-mha-long-p32` | `bf16_942` | 2.77029e-06 | 7538.348 | 227.900 | 22.256 |
| `bf16-mha-long-p32` | `aiter` | 2.68604e-06 | 9943.124 | 172.781 | 16.873 |
| `bf16-mha-short-p32` | `bf16_942` | 2.77385e-06 | 520.804 | 206.170 | 100.669 |
| `bf16-mha-short-p32` | `aiter` | 2.57288e-06 | 649.765 | 165.251 | 80.689 |
| `bf16-mha-short-p64` | `bf16_942` | 2.77397e-06 | 519.403 | 206.726 | 100.940 |
| `bf16-mha-short-p64` | `aiter` | 2.57363e-06 | 650.904 | 164.962 | 80.548 |

当前生产已应用page+4加载/消费分离及D128无LSE的V lane偏移复用；V保持64-bit global加载，未恢复buffer特化。BF16本表最高227.900TFLOPS，尚未达到250T。
D192 full/causal仍慢于AITER，不将全套正确性通过等同于所有路径性能达标。

### 2.2 FP8 MHA：全部7条

| case | backend | acc | 时间µs | TFLOPS | 逻辑GB/s |
|---|---|---:|---:|---:|---:|
| `fp8-full-token-d128` | `fp8_942` | 0.000367516 | 649.245 | 333.738 | 97.923 |
| `fp8-full-token-d192` | `fp8_942` | 0.000387074 | 678.404 | 399.242 | 109.414 |
| `fp8-causal-token-d128` | `fp8_942` | 0.000273883 | 16445.247 | 267.444 | 12.752 |
| `fp8-causal-token-d192` | `fp8_942` | 0.000272198 | 17444.152 | 315.162 | 14.066 |
| `fp8-full-tensor-d128` | `fp8_942` | 0.000363142 | 313.202 | 342.827 | 102.530 |
| `fp8-full-tensor-d192` | `fp8_942` | 0.000383093 | 652.725 | 411.254 | 113.707 |
| `fp8-causal-tensor-d192` | `fp8_942` | 0.000273586 | 17406.851 | 315.837 | 14.096 |

### 2.3 SWA：全部20条（已去重）

已删除两个KV128K重复执行项；保留D128/D192各一个原用例，独立shape覆盖不变。
当前SWA为8个case、20候选，整套为24个case、43候选、2150样本；下表为本轮完整重测，不从两次旧测量中挑较快值。

| case | backend | acc | 时间µs | TFLOPS | 逻辑GB/s |
|---|---|---:|---:|---:|---:|
| `swa-w0-d128` | `swa_bf16` | 2.86756e-06 | 6.220 | 0.086 | 96.226 |
| `swa-w0-d192` | `swa_bf16` | 2.83579e-06 | 6.320 | 0.105 | 118.380 |
| `swa-kv32768-d128` | `swa_bf16` | 2.76076e-06 | 240.961 | 71.854 | 626.636 |
| `swa-kv32768-d128` | `aiter` | 1.89381e-05 | 509.723 | 33.968 | 296.229 |
| `swa-kv32768-d128` | `aiter_gather` | 1.89381e-05 | 594.463 | 29.126 | 310.447 |
| `swa-kv32768-d192` | `swa_bf16` | 2.75763e-06 | 276.982 | 78.137 | 681.429 |
| `swa-kv32768-d192` | `aiter` | 1.89335e-05 | 569.784 | 37.984 | 331.255 |
| `swa-kv32768-d192` | `aiter_gather` | 1.89335e-05 | 652.864 | 33.150 | 353.346 |
| `swa-kv65536-d128` | `swa_bf16` | 2.75856e-06 | 243.402 | 71.134 | 689.280 |
| `swa-kv65536-d128` | `aiter` | 1.89501e-05 | 510.543 | 33.913 | 328.615 |
| `swa-kv65536-d128` | `aiter_gather` | 1.89501e-05 | 670.105 | 25.838 | 350.514 |
| `swa-kv65536-d192` | `swa_bf16` | 2.76895e-06 | 276.341 | 78.318 | 758.899 |
| `swa-kv65536-d192` | `aiter` | 1.89464e-05 | 568.103 | 38.096 | 369.150 |
| `swa-kv65536-d192` | `aiter_gather` | 1.89464e-05 | 726.165 | 29.804 | 404.318 |
| `swa-kv131072-d128` | `swa_bf16` | 2.76235e-06 | 241.801 | 71.605 | 832.613 |
| `swa-kv131072-d128` | `aiter` | 1.89506e-05 | 510.283 | 33.930 | 394.539 |
| `swa-kv131072-d128` | `aiter_gather` | 1.89506e-05 | 825.206 | 20.982 | 406.619 |
| `swa-kv131072-d192` | `swa_bf16` | 2.75848e-06 | 277.022 | 78.126 | 908.441 |
| `swa-kv131072-d192` | `aiter` | 1.89467e-05 | 570.663 | 37.925 | 440.992 |
| `swa-kv131072-d192` | `aiter_gather` | 1.89467e-05 | 885.706 | 24.435 | 473.555 |

### 2.4 整轮与分类耗时

| suite | case执行数 | 候选数 | event样本数 | case总wall time之和（秒） |
|---|---:|---:|---:|---:|
| `bf16-mha` | 9 | 16 | 800 | 35.667 |
| `fp8-mha` | 7 | 7 | 350 | 28.263 |
| `swa` | 8 | 20 | 1000 | 6.851 |
| **合计** | **24** | **43** | **2150** | **70.781** |

同一pytest进程完成已有14项功能及24项性能：**36通过、2跳过**，suite **127.480秒**、进程 **129.12秒**；选卡 **11.884秒**已包含在内。
功能部分为12通过、2个gfx950跳过；性能部分24项全部通过。没有追加独立CLI复测、专项shape、LSE扩展或ATT。
启动时gfx/UMC均0%、无其他进程；48次case前后快照也均0%且无其他进程，但采样不是连续监控，结果仍按授权规则标为**低负载启动、非独占**。
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
- JIT、FP32参考、量化、预先布局转换与workspace分配在计时外；gather总路径的每次转换在计时内。
  counter初始化、辅助launch和间隙在event内，`cudaPerf` GPU spin在起始event前。10-buffer不等于强制清cache。

## 4. 参考与SWA路径

| 候选 | 含义 | 计时范围 |
|---|---|---|
| `bf16_942` / `fp8_942` / `swa_bf16` | 自有8-wave / LDS / 单wave | 直接读取SHUFFLE-5D分页KV的完整调用 |
| `aiter` BF16 Full/Causal/MHA | public varlen，当前MI308报告命中ASM | prepared linear KV，不含转换 |
| `aiter` SWA | prepared CK varlen | 不含gather |
| `aiter_gather` SWA | **完整KV gather+CK** | 每次重新gather全部KV再执行CK，**一对event覆盖两段** |

选择规则明确写在 [test_mha_pa.py](test_mha_pa.py) 的三个候选函数中：
BF16小shape无外部参考，其余必须AITER；FP8仅自有LDS，不加载任何外部性能参考；SWA小shape无外部参考，长shape必须prepared CK及gather+CK。
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

# 本轮实际方式：同一指定入口同时复测原有功能与性能项，不扩展范围。
PYHIP_MHA_PERF=1 PYHIP_MHA_OUTPUT="$OUT/performance" PYHIP_MHA_SELECTION_LOG="$OUT/selection.idle.jsonl" "$PY" -m pytest "$MHA" -q --junitxml="$OUT/tests.xml"
```

也可单独运行某类或精确ID；下面是互相独立的选择示例，完整ID见第7节：

```bash
"$PY" "$MHA" --suite bf16-mha --output "$OUT/bf16-mha.json"
"$PY" "$MHA" --suite fp8-mha --output "$OUT/fp8-mha.json"
"$PY" "$MHA" --suite swa --output "$OUT/swa.json"

# 原dense比较形状仍在；当前只有自有BF16与AITER两条候选。
"$PY" "$MHA" --suite bf16-mha --case bf16-mha-long-p32 bf16-mha-short-p32 bf16-mha-short-p64 --output "$OUT/mha-pages.json"

# 四个小shape，仅自有kernel，无AITER、BN32或gather。
"$PY" "$MHA" --case bf16-smoke-d128 bf16-smoke-d192 swa-w0-d128 swa-w0-d192 --output "$OUT/own-smoke.json"

# 单个FP8案例及完整逐buffer/样本日志。
"$PY" "$MHA" --suite fp8-mha --case fp8-full-tensor-d192 --verbose-runs --output "$OUT/fp8-tensor-d192.json"
```

**`--case`是可选的场景筛选器**：`--suite`选类别，`--case`从该类别中挑选已定义的场景，不设置shape、不选择单个实现，也不修改参考逻辑。
- 不传`--case`：运行所选suite全部场景；例如`--suite bf16-mha`运行9个场景、16条候选。
- 传`--case bf16-full-d128`：只运行Q10240/KV2583、Dq128、H16/HK1、page64、非causal这一场景；在MI308仍包含自有BF16与AITER两条候选。
- 多个ID用空格分隔，必须精确、不能重复且属于所选suite；`--suite all`（默认）可跨类别选取。执行顺序沿用参数集合，不随ID输入顺序改变。
- `--list`只列出参数、不运行GPU；可先用它查ID。`--case`属于CLI，pytest单项选择使用pytest节点ID或`-k`，不是此参数。

```bash
# 单场景复现：无需运行整套BF16测试。
"$PY" "$MHA" --suite bf16-mha --case bf16-full-d128 --output "$OUT/bf16-full-d128.json"
# 只查看该类别的可选ID和参数。
"$PY" "$MHA" --suite bf16-mha --list
```

新增/修改shape、候选或扩展测试前必须先确认；本轮只使用主文件既有三个参数集合，不另建专项测试入口。
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

### 7.1 BF16 MHA：9 case / 16候选

`test_perf_bf16_mha`使用`BF16_MHA_PERF_CASES`；前两项各1个自有候选，其余7项各`bf16_942`+AITER。

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

最后三个是原dense比较形状；只删dense参考，没有删形状或改变参数。

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

本目录的conftest插件已删除。主文件通过性能函数的`__test__`属性控制收集：默认仅14项功能，`PYHIP_MHA_PERF=1`时额外收集24项性能。
不再需要自定义pytest参数或marker；`-k test_perf_`使用标准pytest功能只选择性能用例。
下表是当前范围；本次功能pytest与性能pytest均已实际运行：

| 功能函数 | 参数项数 | MI308适用性 | 主要检查 |
|---|---:|---|---|
| `test_bf16_mha` | D128/192 × 8wave/persistent = 4 | 2可运行、2个gfx950 persistent跳过 | ragged、空Q、尾页、causal、GQA、非单位scale |
| `test_fp8_mha` | D128/192 × C/NC × 两种Q scale = 8 | 8可运行 | ragged、空Q、NaN尾页、量化scale |
| `test_swa` | D128/192 = 2 | 2可运行 | 空KV、W128+sink、padded Q/O、NaN尾页、非单位scale |

本轮只执行指定入口的功能+性能合并pytest：**36通过、2跳过**（127.480秒）；对应功能12通过/2跳过、性能24通过。默认功能收集14项，启用性能后共38项。
没有新增CPU测试、恢复独立gather测试或加入额外shape；进一步扩范围先确认。

按第6节准备环境后，选择普通功能测试或显式性能测试：

```bash
PYHIP_MHA_SELECTION_LOG="$OUT/pytest-functional-gpu.jsonl" "$PY" -m pytest "$MHA" --import-mode=importlib -q --junitxml="$OUT/tests.xml"

PYHIP_MHA_PERF=1 PYHIP_MHA_OUTPUT="$OUT/pytest-perf" PYHIP_MHA_SELECTION_LOG="$OUT/pytest-perf-gpu.jsonl" "$PY" -m pytest "$MHA" --import-mode=importlib -k test_perf_ -q
```

性能pytest与CLI共用三套参数和候选规则；收集前设置`PYHIP_MHA_PERF=1`启用性能项，`-k test_perf_`只运行性能项；省略`-k`则同时运行功能测试。
`PYHIP_MHA_OUTPUT`是可选的输出目录，每case一对JSON/Markdown，已有同名文件拒绝覆盖；省略则不落盘，不生成整轮manifest。
pytest用`PYHIP_MHA_GPU`/`PYHIP_MHA_REQUIRED_PTL`选卡；第6节已设为auto/VECTOR,F8，未设置GPU变量时默认current。
列表与收集阶段不初始化GPU。普通功能测试不做性能event计时；性能小shape是24项中的4项，不能另加到14项功能数里。

## 8. 原生缓存、输出与跨机边界

**已启用编译缓存，在排查问题时需要检查缓存是否出现问题**。当前直接使用FlyDSL原生默认缓存，
没有MHA私有持久缓存层、目录重定向或额外CLI开关，不需包装原生缓存。
核对实际FlyDSL版本、`env.runtime.enable_cache`、`env.runtime.cache_dir`以及shell残留的
`FLYDSL_RUNTIME_ENABLE_CACHE`/`FLYDSL_RUNTIME_CACHE_DIR`覆盖，避免旧环境继续禁用或重定向缓存。
源码/依赖变化后的异常输出、加载错误或反复编译，应检查缓存兼容性和失效；首次/新shape编译正常。
本轮实测发现FlyDSL0.3.1可能漏掉只在嵌套函数中使用的全局helper依赖；BF16工厂已显式闭包捕获softmax，原生依赖键与新产物IR均核验。
只记录源文件SHA不足以证明运行了新helper，调优时还需核对实际artifact；不以关闭缓存或添加私有缓存层代替依赖修复。
不缓存正确性结论，不加载不可信缓存，不通过减少10-buffer或跳过检查掩盖问题。

V已恢复64-bit global地址加载，不再创建D128 V buffer描述符。buffer的32-bit相对offset/跨度限制并不要求GPU基地址低于4GiB。
当前gfx942 BF16公开API仍限制单个Q/K/V张量byte span小于2GiB；该限制早于本项优化，本轮没有解除，不能将撤销buffer等同于完整支持>4GiB单张量。

MI308不能验证gfx950原生执行；gfx950另机实测范围见 [README.355.md](README.355.md)（实际MI350X、低负载启动，非独占），不能把skip视作pass。
10-buffer增加显存用量，OOM应如实报告，不自动缩小shape或buffer数。跨机记录见 [changes.md](changes.md)。

JSON的`wall_time_s`区分选卡和执行，每case另列输入准备、FP32参考、参考设置/首调、输出校验、dispatch profiling、warmup和measurement阶段。
首调可能含缓存加载/JIT，不是纯编译时间；`measurement_wall_s`含spin与Python开销，不等于event样本之和。
正常运行不再打印选卡、环境、`PREPARED`/`VALIDATED`/`MEASURED`/`WALL_TIME`或AITER dispatch流程信息。
性能表和`--verbose-runs`逐样本性能信息保持不变；阶段耗时、dispatch、跳过原因仍在JSON中，空闲采样仍在idle JSONL中。
不打印等待进度不代表不等待，仍需3次满足显式启动门槛；完整精度检查不变，`--list`/`--help`显式查询仍有输出。
AITER默认使用ERROR日志级别（显式`AITER_LOG_LEVEL`可覆盖）；profiler只过滤启动/停止及单周期提示，失败时回放原生诊断，不屏蔽其他异常警告/错误。
原测试辅助模块及pytest配置已合并删除；仅保留主测试文件、4个生产kernel、生产共用DSL适配和包初始化。本轮仅修改gfx942 BF16生产kernel，其余kernel/DSL及常规测试实现未改，原生缓存不重定向。
删除旧单卡等待、无调用的Case.pack及后端字段、历史量化前padding路径、未使用的LSE/unchecked测试分支；实际输入和O数值不变。
历史报告保持原样；第2.1–2.3节为本轮同一pytest进程的完整性能数据，不混入历史专项或前后对照的样本。

旧单buffer、dense、register、ISA及日志继续保留在各历史结果中；不得用其数值或计数冒充新范围复测。
