# MI325X Attention：三个性能集与10-buffer复现

更新：2026-09-08。**MI325X / gfx942 / 304 CU；PTL N/A；不测LSE。**
按 [README.308.md](README.308.md) 的三个性能集完成本轮验证：CLI与性能pytest各自 **24 case、43候选、2150 event样本，全部正确性通过**；完整CLI数据直接列于第2节。
BF16 dense和FP8外部BN32适配均已移除，原始shape保留；FP8仅自有LDS，BF16/SWA小shape只测自有kernel。
功能pytest **12通过、2个gfx950 persistent跳过**。性能pytest子进程成功，外围脚本收尾错误独立记录于第8节，不改写原始证据。

## 1. 环境与结果

| 项目 | 现有复现环境 |
|---|---|
| GPU | MI325X / gfx942 / 304 CU / 约256 GiB |
| 主机 | quanta-ccs-aus-E04-10，Linux 6.8.0-38-generic |
| Python | 3.11.11，使用第6节的已有venv |
| torch / HIP / 系统ROCm | 2.12.1+rocm7.2 / 7.2.53211 / 7.2.0 |
| FlyDSL / Triton | 0.3.1 / triton-rocm 3.7.1 |
| AITER / CK | `bde46043bcf08e41ac40395a18369ab6309153ca` / `15e12dd7f25ee583617c78f66cb502ff9916585f` |
| 本轮设备 | 物理GPU3，BDF `0000:79:00.0`，ROCr UUID `GPU-9ff0c5f642c689c2`；映射为逻辑GPU0并核验 |
| PTL / 功率限制 | N/A / 1000W，只读；不套用MI308的Enabled/VECTOR,F8 |

**版本边界：本轮实测为 `4adaf81` 加MI325本地选卡/守卫/证据适配，源码已保存为 [实测快照](results/mi325_singlefile_20260908T031247Z/tested-source/test_mha_pa.py)。**
测试结束后，当前分支`tmp-main`已快进到`lc/tmp-main`的 **`1f741754ab9821c935bf98f6b8190f8983a4f443`**；[当前入口](test_mha_pa.py)与远端一致，尚未按这个新提交做MI325原生复测。
下文复现命令明确使用已测快照，不把旧数值重标为最新远端验证。

AITER使用已有固定commit的editable安装，工作盘JIT副本见第6节；源码/依赖清理后须按固定版本恢复，不能静默省略参考。
沿用用户授权：**只使用GPU3，gfx/UMC均低于2%，允许驻留进程；不是独占测试。**
根盘满时按授权使用工作盘HOME/临时/缓存存储，不修改PTL、时钟、功率或NUMA，不reset、不终止其他任务，不commit/push。

## 2. 完整性能数据与测试耗时

以下为 **2026-09-08 03:49:34–04:07:11 UTC** 同一轮CLI实测，未混入旧轮次或pytest的性能数值。
完整列出全部43条候选：BF16 16条、FP8 7条、SWA 20条，包括全部小shape；全部正确性通过。
每行时间为10个独立buffer、5轮共50个event的每调用中位数，`acc`为10组最大值；时间/TFLOPS/逻辑GB/s保留三位小数。
逻辑GB/s不是实测HBM带宽；AITER prepared不含gather，`aiter_gather`包含完整gather+CK。FP8仅自有LDS，无外部参考。

### 2.1 BF16 MHA：全部16条

| case | backend | acc | 时间µs | TFLOPS | 逻辑GB/s |
|---|---|---:|---:|---:|---:|
| `bf16-smoke-d128` | `bf16_942` | 2.690568e-06 | 29.265 | 2.347 | 20.452 |
| `bf16-smoke-d192` | `bf16_942` | 2.675029e-06 | 32.529 | 2.640 | 23.000 |
| `bf16-full-d128` | `bf16_942` | 2.868857e-06 | 549.629 | 394.225 | 155.029 |
| `bf16-full-d128` | `aiter` | 2.654881e-06 | 411.421 | 526.657 | 207.108 |
| `bf16-full-d192` | `bf16_942` | 2.877927e-06 | 705.124 | 384.113 | 151.052 |
| `bf16-full-d192` | `aiter` | 2.664986e-06 | 461.296 | 587.144 | 230.895 |
| `bf16-causal-d128` | `bf16_942` | 2.424525e-06 | 9928.882 | 442.968 | 28.726 |
| `bf16-causal-d128` | `aiter` | 2.249573e-06 | 7859.980 | 559.566 | 36.287 |
| `bf16-causal-d192` | `bf16_942` | 2.425113e-06 | 12386.871 | 443.835 | 28.782 |
| `bf16-causal-d192` | `aiter` | 2.247466e-06 | 8585.956 | 640.316 | 41.523 |
| `bf16-mha-long-p32` | `bf16_942` | 2.780988e-06 | 4475.493 | 383.865 | 37.487 |
| `bf16-mha-long-p32` | `aiter` | 2.697695e-06 | 3369.104 | 509.924 | 49.797 |
| `bf16-mha-short-p32` | `bf16_942` | 2.772983e-06 | 369.998 | 290.202 | 141.700 |
| `bf16-mha-short-p32` | `aiter` | 2.571839e-06 | 256.887 | 417.981 | 204.092 |
| `bf16-mha-short-p64` | `bf16_942` | 2.776314e-06 | 377.430 | 284.488 | 138.910 |
| `bf16-mha-short-p64` | `aiter` | 2.572041e-06 | 255.446 | 420.340 | 205.244 |

七个非小 shape 的自有 BF16 比本轮 AITER prepared ASM **慢 26.32%–52.86%**；没有指定 dense 参考，不能沿用旧 dense gate 的通过/失败结论。

### 2.2 FP8 MHA：全部7条

| case | backend | acc | 时间µs | TFLOPS | 逻辑GB/s |
|---|---|---:|---:|---:|---:|
| `fp8-full-token-d128` | `fp8_942` | 0.000381311 | 264.379 | 819.572 | 240.472 |
| `fp8-full-token-d192` | `fp8_942` | 0.0003829935 | 288.776 | 937.914 | 257.040 |
| `fp8-causal-token-d128` | `fp8_942` | 0.0002756518 | 5807.743 | 757.296 | 36.110 |
| `fp8-causal-token-d192` | `fp8_942` | 0.0002730074 | 6478.657 | 848.590 | 37.873 |
| `fp8-full-tensor-d128` | `fp8_942` | 0.0003780793 | 162.085 | 662.456 | 198.122 |
| `fp8-full-tensor-d192` | `fp8_942` | 0.0003830103 | 279.362 | 960.889 | 265.676 |
| `fp8-causal-tensor-d192` | `fp8_942` | 0.0002732739 | 6464.434 | 850.457 | 37.956 |

所有 FP8 仅与独立 FP32 O 检查正确性，**没有外部性能参考**，不宣称本轮相对 BN32 加速。token/tensor 两个 D128 Full 的 H/KV 也不同，不能只归因于 scale 模式。

### 2.3 SWA：全部20条（已去重）

两个KV128K重复参数项已删除；保留D128/D192各一个原用例，独立shape覆盖不变。
当前SWA为8个case、20候选，整套24个case、43候选、2150样本；不从重复测量中挑较快值。

| case | backend | acc | 时间µs | TFLOPS | 逻辑GB/s |
|---|---|---:|---:|---:|---:|
| `swa-w0-d128` | `swa_bf16` | 2.867565e-06 | 5.929 | 0.090 | 100.958 |
| `swa-w0-d192` | `swa_bf16` | 2.871528e-06 | 6.049 | 0.110 | 123.683 |
| `swa-kv32768-d128` | `swa_bf16` | 2.760462e-06 | 96.906 | 178.669 | 1558.159 |
| `swa-kv32768-d128` | `aiter` | 1.89355e-05 | 192.230 | 90.070 | 785.491 |
| `swa-kv32768-d128` | `aiter_gather` | 1.89355e-05 | 224.879 | 76.993 | 820.659 |
| `swa-kv32768-d192` | `swa_bf16` | 2.765669e-06 | 114.032 | 189.794 | 1655.182 |
| `swa-kv32768-d192` | `aiter` | 1.894092e-05 | 199.922 | 108.255 | 944.087 |
| `swa-kv32768-d192` | `aiter_gather` | 1.894092e-05 | 247.934 | 87.292 | 930.436 |
| `swa-kv65536-d128` | `swa_bf16` | 2.762626e-06 | 97.487 | 177.604 | 1720.970 |
| `swa-kv65536-d128` | `aiter` | 1.894159e-05 | 192.751 | 89.826 | 870.409 |
| `swa-kv65536-d128` | `aiter_gather` | 1.894159e-05 | 255.325 | 67.812 | 919.930 |
| `swa-kv65536-d192` | `swa_bf16` | 2.760535e-06 | 113.131 | 191.306 | 1853.738 |
| `swa-kv65536-d192` | `aiter` | 1.895294e-05 | 200.884 | 107.737 | 1043.964 |
| `swa-kv65536-d192` | `aiter_gather` | 1.895294e-05 | 282.247 | 76.680 | 1040.230 |
| `swa-kv131072-d128` | `swa_bf16` | 2.766486e-06 | 95.984 | 180.384 | 2097.491 |
| `swa-kv131072-d128` | `aiter` | 1.894081e-05 | 189.166 | 91.529 | 1064.285 |
| `swa-kv131072-d128` | `aiter_gather` | 1.894081e-05 | 318.420 | 54.375 | 1053.779 |
| `swa-kv131072-d192` | `swa_bf16` | 2.760524e-06 | 111.909 | 193.395 | 2248.776 |
| `swa-kv131072-d192` | `aiter` | 1.89498e-05 | 200.603 | 107.888 | 1254.509 |
| `swa-kv131072-d192` | `aiter_gather` | 1.89498e-05 | 359.883 | 60.138 | 1165.463 |

六个长 shape 的 direct 比完整 gather+CK **延迟低 54.01%–69.86%**；prepared CK 不含 gather，仅作分解参考。
gather 逐次搬运完整 KV，不裁去窗口外前缀；实际为一个 Triton gather 加一个 CK dispatch，一对 event 覆盖两段。
每组 20 个 K/V workspace 地址及 10 个 slot mapping 地址独立，prepared 与 gather 的 workspace 也不复用。

### 2.4 整轮与分类耗时

| suite | case执行数 | 候选数 | event样本数 | case总wall time之和（秒） |
|---|---:|---:|---:|---:|
| `bf16-mha` | 9 | 16 | 800 | 155.662 |
| `fp8-mha` | 7 | 7 | 350 | 147.748 |
| `swa` | 8 | 20 | 1000 | 121.747 |
| **合计** | **24** | **43** | **2150** | **425.157** |

CLI进程总耗时 **1056.893秒**；其中选卡及激活 **11.149秒**、执行 **1043.253秒**。
功能pytest **12通过、2跳过**，suite **183.127秒**、进程 **184.370秒**；性能pytest **24通过**，suite **1058.326秒**、进程 **1060.068秒**。
进程时间包含选卡/执行，不能再次累加；分类wall time之和不含case外守卫、环境查询和保存。
三次Python成功子进程共2301.331秒，不是含排障和文档整理的会话总耗时，也不将外围脚本收尾错误称为成功。

本机已测适配比MI308多做case前后和计时前的稳定等待：CLI有72个守卫窗口、260条采样，其中44条不满足阈值后继续等待。
守卫累计956.695秒；其中计时前守卫345.601秒，外层同步/读取计入阶段后为348.890秒，已包含在425.157秒case合计中。
性能pytest也有72个窗口、260条采样、44条阈值拒绝；功能pytest有12个窗口、36条采样。
近期自身计算也会影响采样，不能把拒绝样本全归因于他人任务；等待均在event外，不把wall time差异当作kernel性能变化。
全部case阶段和pytest逐项耗时见 [验证汇总](results/mi325_singlefile_20260908T031247Z/SUMMARY.md) 及同目录CSV，正文不重复展开24行阶段表。

## 3. 默认10-buffer协议

- `--buffers 10`为缺省值；独立随机输入`seed+i`、Q/K/V、metadata、scale，每候选每组独立O。
	AITER prepared KV和gather workspace也按buffer独立；JSON记录地址和seed。
- `--warmup 10`依次预热10组；`--run-count 5`表示**5个完整buffer轮次**，每轮测完全部10组，默认每候选**50个event样本**。
- `--repeat 1`为默认；增大时在一对event内逐调用轮转buffer，再按调用次数归一化。
	各候选使用同一索引序列，候选先后顺序交替；不恢复独立repeat3专项。
- 每buffer独立FP32 O检查、有限值检查、两次额外逐位重复；BF16的`rtol=atol=0.02`，FP8为0.1。
	汇总`acc`为10组最大值，逐组值记录于`acc_per_buffer`。
- 总结取全部event样本中位数，不选最快值、不删慢样本；小shape的dispatch抖动也保留。
- JIT、FP32参考、量化、预先布局转换与workspace分配在计时外；gather总路径每次转换在计时内。
	counter初始化、辅助launch和间隙在event内，`cudaPerf` GPU spin在起始event前。10-buffer不等于强制清cache。

两轮性能已分别审计：每轮430个buffer精度检查、860次额外逐位检查、2150个event，所有buffer地址、每轮0–9索引和中位数匹配。
这不是额外GPU pytest项，不把离线审计计入功能测试通过数。

## 4. 参考与SWA路径

| 候选 | 含义 | 计时范围 |
|---|---|---|
| `bf16_942` / `fp8_942` / `swa_bf16` | 自有8-wave / LDS / 单wave | 直接读取SHUFFLE-5D分页KV的完整调用 |
| `aiter` BF16 Full/Causal/MHA | public varlen，本轮命中ASM | prepared linear KV，不含转换 |
| `aiter` SWA | prepared CK varlen | 不含gather |
| `aiter_gather` SWA | **完整KV gather+CK** | 每次重新gather全部KV再执行CK，**一对event覆盖两段** |

候选规则与MI308一致：BF16小shape无外部参考，其余必须AITER；FP8仅自有LDS；SWA小shape无参考，长shape必须prepared CK及gather+CK。
声明参考缺依赖或首调失败会停止，不静默省略；数值错误始终失败。BF16 dense/FP8 BN32适配已删除，独立FP32 O检查仍覆盖全部buffer。
gather不裁SWA前缀，不用两个独立均值相加，不进入生产dispatch；运行时检查完整gather内容、独立workspace及两个dispatch。

本轮BF16实际命中ASM `fmha_fwd_hd128_bf16_rtna_group` / `fmha_fwd_hd192x128_bf16_rtna_group`及对应causal变体；SWA命中CK `BlockFmhaPipelineQRKSVSAsync`。
完整dispatch名称保存在JSON，不能仅凭AITER入口名推断具体实现。BF16完整调用中的counter初始化未从event中扣除。

## 5. 指标规则

`acc`是归一化平方误差 $\sum(ref-out)^2/\sum(ref^2+out^2)$，越小越好，不是准确率百分比；同时必须通过逐元素检查。
`passed`仅指正确性，不表示达到性能门槛。时间越低越好，TFLOPS只计可见QK/PV，不计mask/padding或sink。

- direct/prepared按逻辑Q/K/V/O计字节，SWA也计完整逻辑KV；FP8输入1字节，BF16输入/O为2字节。
- gather+CK在此基础上加**完整KV读+完整workspace写**，除以总路径时间，不是只算有效窗口。
- GB/s是逻辑字节除对应event时间，**不是实测HBM流量/饱和率**，不包含所有物理重读、metadata或缓存行为。
- 各路径字节分子不同，优劣优先比较同shape延迟，不能只看GB/s大小。
- FP8源为BF16随机数经scale量化为FP8，量化不计时；与旧native-cast/FlyDSL0.2.2/profiler gate不同。

## 6. 当前CLI复现

从Git根、同一个专用bash终端执行，使用已有环境，无需重装GPU依赖。环境准备后，每条测试命令均为单行；每轮新建`OUT`，不覆盖旧证据。
**本节复现第2节已测版本，因此`MHA`明确指向实测源码快照，而不是更新后的远端入口。**
远端`1f74175`使用`--max-gpu-utilization` / `PYHIP_MHA_MAX_GPU_UTILIZATION`，与快照的参数/等待语义不同；当前远端尚未在MI325复测，本地旧补丁没有重放到它上面。

```bash
PY=/root/.venvs/pyhip-mha-mi325-quanta-20260907/bin/python
MHA=tests/flydsl/mha/results/mi325_singlefile_20260908T031247Z/tested-source/test_mha_pa.py
OUT=$(mktemp -d "$PWD/tests/flydsl/mha/results/mi325-three-suites.XXXXXX")
export HOME=/host_lc/pyhip-mha-mi325-storage-20260908/home
export XDG_CACHE_HOME="$HOME/.cache" TMPDIR=/host_lc/pyhip-mha-mi325-storage-20260908/tmp
export TMP="$TMPDIR" TEMP="$TMPDIR"
export PYHIP_MHA_GPU=3 PYHIP_MHA_REQUIRED_PTL=current PYHIP_MHA_IDLE_MAX_UTILIZATION=2
export ROCR_VISIBLE_DEVICES=GPU-9ff0c5f642c689c2 HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0
export GPU_ARCHS=gfx942 MAX_JOBS=4
export AITER_JIT_DIR="$XDG_CACHE_HOME/pyhip-mha-mi325-quanta/aiter-jit-gpu3"
export PATH="$(dirname "$PY"):$PATH" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH COMPILE_ONLY ARCH CUDAPERF PYHIP_MHA_COMPILE_CACHE
unset FLYDSL_RUNTIME_ENABLE_CACHE FLYDSL_RUNTIME_CACHE_DIR FLYDSL_COMPILE_ONLY FLYDSL_COMPILE_ARCH FLYDSL_DUMP_IR FLYDSL_DUMP_DIR FLYDSL_RUN_ONLY
unset TRITON_CACHE_DIR TORCH_EXTENSIONS_DIR PYHIP_MHA_PERF PYHIP_MHA_OUTPUT PYHIP_MHA_SELECTION_LOG

# GPU-free：仅列出显式case、参数和架构限制。
"$PY" "$MHA" --list

# 推荐整轮：固定授权GPU3，只选卡一次，同进程执行三套case。
"$PY" "$MHA" --suite all --gpu 3 --required-ptl current --idle-max-utilization 2 --buffers 10 --run-count 5 --warmup 10 --repeat 1 --output "$OUT/all.json"
```

也可单独运行某类或精确ID；以下是互相独立的选择示例，完整ID见第7节：

```bash
"$PY" "$MHA" --suite bf16-mha --output "$OUT/bf16-mha.json"
"$PY" "$MHA" --suite fp8-mha --output "$OUT/fp8-mha.json"
"$PY" "$MHA" --suite swa --output "$OUT/swa.json"

# 原dense比较shape仍在，仅自有BF16与AITER两条候选。
"$PY" "$MHA" --suite bf16-mha --case bf16-mha-long-p32 bf16-mha-short-p32 bf16-mha-short-p64 --output "$OUT/mha-pages.json"

# 四个小shape，仅自有kernel，无AITER、BN32或gather。
"$PY" "$MHA" --case bf16-smoke-d128 bf16-smoke-d192 swa-w0-d128 swa-w0-d192 --output "$OUT/own-smoke.json"

# 单个FP8案例及完整逐buffer/样本日志。
"$PY" "$MHA" --suite fp8-mha --case fp8-full-tensor-d192 --verbose-runs --output "$OUT/fp8-tensor-d192.json"
```

**`--case`是可选场景筛选器**：`--suite`选类别，`--case`从该类别挑选已定义场景，不设置shape、不选择单个实现，也不修改参考逻辑。
- 不传`--case`：运行所选suite全部场景；例如BF16为9个case、16条候选。
- `--case bf16-full-d128`：只选Q10240/KV2583、Dq128、H16/HK1、page64、非causal场景，包含自有BF16与AITER。
- 多个ID用空格分隔，必须精确、不重复且属于所选suite；执行顺序沿用参数集合。`--suite all`可跨类别选取。
- `--list`只列参数、不初始化GPU；pytest单项选择使用节点ID或`-k`，不是CLI的`--case`。

```bash
"$PY" "$MHA" --suite bf16-mha --case bf16-full-d128 --output "$OUT/bf16-full-d128.json"
"$PY" "$MHA" --suite bf16-mha --list
```

自定义输入应修改显式`Workload`集合，不恢复旧backend/preset/shape猜测或参考开关。CLI始终检查正确性，buffers/run-count/repeat为正、warmup非负。
一份根JSON的`records`包含所有选定case，成功后生成同名Markdown；选卡和守卫另存同名idle JSONL，拒绝覆盖已有输出。

已测快照的严格默认仍要求gfx/UMC为0且无其他进程；本机显式阈值2允许驻留进程，要求连续3次、5秒间隔均严格低于2%。
除选卡外，每case前后及准备/校验/预热后的正式计时前也等待；健康但忙时无总截止时间，读数失效则失败关闭。
本机仅授权GPU3，不使用不限池的auto选卡；用户级锁不是独占预约，采样不能保证测量期间完全无争用。
快照按KFD `domain/location_id`匹配BDF取得ROCr UUID，并核验实际torch BDF；PCI sysfs的ASIC serial不能直接当此机ROCr UUID。

## 7. 显式case与pytest范围

统一B1、Dv128、contiguous Q/O、零尾页、seed20260905；除表内注明外，H16/HK1、page64、Q scale为per-token。
BF16使用unit descales，FP8从BF16源量化；NC为full/noncausal，C为bottom-right causal，BF16/FP8 full和causal无窗口、无sink。

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

后三个原dense比较shape不变；long固定Q/KV20480，不按MI325的304 CU放大为77824。

### 7.2 FP8 MHA：7 case / 7候选

`test_perf_fp8_mha`使用`FP8_MHA_PERF_CASES`；全部gfx942限定，每项仅`fp8_942`。统一HK1/page64，K/V为per-tensor scale。

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
前两项只测`swa_bf16`，6个长shape各包含`swa_bf16`、`aiter`、`aiter_gather`。

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

两个KV128K重复参数项已删除；当前8个case均为独立shape，默认repeat1。W128最多可见129个key，W0只含当前位置，sink只加一次分母。

### 7.4 普通pytest与性能pytest

没有目录conftest插件；性能函数的`__test__`属性控制收集：默认仅14项功能，`PYHIP_MHA_PERF=1`时额外收集24项性能。
标准`-k test_perf_`仅选性能项，不需自定义pytest参数或marker。本轮两类pytest子进程均已实际运行：

| 功能函数 | 参数项数 | MI325适用性 | 主要检查 |
|---|---:|---|---|
| `test_bf16_mha` | D128/192 × 8wave/persistent = 4 | 2可运行、2个gfx950 persistent跳过 | ragged、空Q、尾页、causal、GQA、非单位scale |
| `test_fp8_mha` | D128/192 × C/NC × 两种Q scale = 8 | 8可运行 | ragged、空Q、NaN尾页、量化scale |
| `test_swa` | D128/192 = 2 | 2可运行 | 空KV、W128+sink、padded Q/O、NaN尾页、非单位scale |

功能 **12通过、2跳过**（183.127秒），性能 **24项全部通过**（1058.326秒）；默认及显式性能收集分别14项和24项。
没有新增CPU测试、独立gather测试或repeat3专项；性能小shape是24项中的4项，不能另加到14项功能数里。

按第6节准备环境后，选择普通功能测试或显式性能测试：

```bash
PYHIP_MHA_SELECTION_LOG="$OUT/pytest-functional-gpu.jsonl" "$PY" -m pytest "$MHA" --import-mode=importlib -q --junitxml="$OUT/tests.xml"

PYHIP_MHA_PERF=1 PYHIP_MHA_OUTPUT="$OUT/pytest-perf" PYHIP_MHA_SELECTION_LOG="$OUT/pytest-perf-gpu.jsonl" "$PY" -m pytest "$MHA" --import-mode=importlib -k test_perf_ -q
```

性能pytest与CLI共用三套参数和候选规则；省略`-k`则同时执行功能测试。`PYHIP_MHA_OUTPUT`可省略，指定时每case保存一对JSON/Markdown，已有文件拒绝覆盖。
已测快照用`PYHIP_MHA_GPU=3`、`PYHIP_MHA_REQUIRED_PTL=current`及显式低利用率变量；未设置GPU变量时pytest默认current，与CLI默认auto不同。
列表和收集不初始化GPU，普通功能测试不做性能event计时。最新远端低负载接口不同，不能直接将快照变量用于当前入口。

## 8. 原生缓存、输出与跨机边界

**已启用编译缓存，在排查问题时需要检查缓存是否出现问题**。使用FlyDSL原生缓存，默认开启，不增加MHA私有持久层或额外缓存CLI开关。
本机根overlay写入`ENOSPC`，用户授权使用工作盘：仅验证子进程的HOME、XDG缓存和TMPDIR改到工作盘，原根目录/全局环境不动。
已有FlyDSL/Triton/AITER缓存复制后分别48/66/916个文件SHA256一致；原缓存保留，没有清空或移动，测试继续使用这些副本。
FlyDSL仍按默认HOME相对目录解析，`env.runtime.enable_cache=True`，无`FLYDSL_*`覆盖；**实际物理目录已变化，这是本机显式存储适配**。
不修改原生key、编译锁或失效规则，不把“开启”当作全部命中，不称纯冷/全热。异常输出、加载失败或反复编译时检查版本、实际缓存目录和shell残留覆盖，不能减少buffer或跳过正确性。

本轮已测源码只改选卡/守卫和证据字段；四个生产kernel、DSL、输入/FP32参考及event循环已与`4adaf81`逐字节/AST核对。
JSON补齐metadata/scale、prepared/gather workspace及slot mapping地址并检查去别名；两轮各43候选/2150样本全部审计通过。
本机PCI sysfs unique_id是ASIC serial，首次错误映射被实际设备核验拦截；KFD映射修复后完整测试通过，错误日志保留，不把14项fixture error当pass。
这些本地适配在更新远端前已存入stash **`0e53a6947dedff0f44a33d459fd91d2863f3b845`**，未覆盖当前最新源码；[更新与备份记录](results/mi325_singlefile_20260908T031247Z/remote-update.txt)保存完整版本及归档摘要。

JSON的wall time区分选卡/执行，每case另列输入、FP32参考、参考设置/首调、校验、dispatch profiling、warmup、计时前等待和measurement。
首调可能含缓存加载/JIT，不是纯编译时间；measurement wall含spin/Python开销，不等于event样本之和。
正常流程不打印选卡/准备/校验进度，性能表和`--verbose-runs`保留；错误诊断、dispatch及idle采样独立保存。CLI一条ROCTracer重复flow警告未隐藏。
系统没有GNU time，使用标准库记录子进程耗时。性能pytest已经完整结束、JUnit24通过、子进程exit0，但外层bash在收尾时因运行中脚本编辑报EOF并exit2；当前脚本语法已检查，原日志不改写，也未为此重测或筛选样本。
后续修改执行脚本须等进程退出，或用不可变脚本副本启动，避免同类收尾错误。

最后重新等待GPU3三次稳定样本后完成独立tensor smoke，实际BDF/UUID正确，ECC corrected/uncorrected/deferred均0。
smoke后的只读瞬时负载为gfx2%/UMC0%，不把它表述为持续空闲；该时刻未再启动其他GPU工作。本轮所有自有测试和监控已退出。
MI325不能验证gfx950原生执行；对应平台见 [README.355.md](README.355.md)，不能把skip当pass。10-buffer OOM须如实报告，不自动缩小规模。

原始结果、JUnit、逐项耗时及源码快照保存在 [本轮目录](results/mi325_singlefile_20260908T031247Z)，完整审计见 [audit.json](results/mi325_singlefile_20260908T031247Z/audit.json)。
第2节始终只用同一轮CLI数据；旧单buffer、dense、register、ISA及失败日志保留，不能冒充当前远端或其他机器的新范围验证。