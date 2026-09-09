# MI325X Attention：三个性能集与10-buffer复现

更新：2026-09-08。**MI325X / gfx942 / 304 CU；PTL N/A；不测LSE。**
代码已快进至远端`lc/tmp-main`的 **`4f3820e`**。本轮仅使用 [test_mha_pa.py](test_mha_pa.py) 的`test_bf16_mha`和`test_perf_bf16_mha`，**16项全部通过、0失败、0跳过，52项未选择**。
其中BF16性能 **12 case、34候选、1700 event样本全部正确性通过**，新数据见第2.1节；功能部分4项通过，覆盖persistent和普通grid。
远端新版已加入grid候选及3个D192 MHA配对shape，本轮直接使用这些既有参数；没有自行新增测试函数、shape或参考。
**FP8/SWA本轮未重测**，第2.2–2.3节明确保留历史数据；新增的stage/layout/LSE/streams/graph专项也未选择，不追加CLI、ATT或GPU smoke。

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

**本轮BF16实测提交：`4f3820ec5df6339dc5b9fe63dd56515ad6a60292`。**
生产 [mha_pa_bf16_942.py](mha_pa_bf16_942.py) SHA256为 `ecd7598a97c8af3eae841650366f864ea5b48fd54edda6db39d5001d4baa13c8`，与该提交一致。
仅在指定测试入口修正本机KFD/PCI UUID映射；没有改动生产kernel、输入、参考、参数集合或计时循环。
第2.2–2.3节仍来自`4adaf81`加当时MI325适配的历史轮次，不代表FP8/SWA在`4f3820e`上重新验证。

AITER使用已有固定commit的editable安装，工作盘JIT副本见第6节；源码/依赖清理后须按固定版本恢复，不能静默省略参考。
沿用用户授权：**只使用GPU3，连续3次gfx/UMC均低于2%后启动，允许驻留进程；不是独占测试。**
继续使用已准备的工作盘HOME/临时/原生缓存，不清空缓存或重新安装依赖；不修改PTL、时钟、功率或NUMA，不reset、不终止其他任务，不commit/push。

## 2. 完整性能数据与测试耗时

**第2.1节为2026-09-08 23:51:59–23:54:54 UTC同一个pytest进程生成的12份BF16报告，共34条候选；第2.2–2.3节是保留的历史FP8/SWA数据。**
本轮只更新BF16，不拼接新旧样本、不选择较快轮次；历史值不计入本轮测试计数。
每行时间为10个独立buffer、5轮共50个event的每调用中位数，`acc`为10组最大值；时间/TFLOPS/逻辑GB/s保留三位小数。
逻辑GB/s不是实测HBM带宽；AITER prepared不含gather，`aiter_gather`包含完整gather+CK。FP8仅自有LDS，无外部参考。

### 2.1 BF16 MHA：全部34条

`bf16_942`为默认persistent固定驻留网格，`bf16_942_grid`为普通head/batch/query-tile网格；二者使用同一新版8-wave/8-stage、BM256/BN64计算核心，不是新旧kernel对照。

| case | backend | acc | 时间µs | TFLOPS | 逻辑GB/s |
|---|---|---:|---:|---:|---:|
| `bf16-smoke-d128` | `bf16_942` | 2.622561e-06 | 16.946 | 4.053 | 35.320 |
| `bf16-smoke-d128` | `bf16_942_grid` | 2.622561e-06 | 17.026 | 4.034 | 35.154 |
| `bf16-smoke-d192` | `bf16_942` | 2.5879e-06 | 18.909 | 4.541 | 39.566 |
| `bf16-smoke-d192` | `bf16_942_grid` | 2.5879e-06 | 19.109 | 4.493 | 39.152 |
| `bf16-full-d128` | `bf16_942` | 2.861705e-06 | 383.419 | 565.121 | 222.234 |
| `bf16-full-d128` | `bf16_942_grid` | 2.861705e-06 | 378.291 | 572.781 | 225.246 |
| `bf16-full-d128` | `aiter` | 2.654881e-06 | 413.304 | 524.258 | 206.164 |
| `bf16-full-d192` | `bf16_942` | 2.872502e-06 | 464.822 | 582.690 | 229.143 |
| `bf16-full-d192` | `bf16_942_grid` | 2.872502e-06 | 468.027 | 578.700 | 227.574 |
| `bf16-full-d192` | `aiter` | 2.664986e-06 | 463.399 | 584.479 | 229.846 |
| `bf16-causal-d128` | `bf16_942` | 2.393107e-06 | 7434.683 | 591.576 | 38.362 |
| `bf16-causal-d128` | `bf16_942_grid` | 2.393107e-06 | 7378.994 | 596.041 | 38.652 |
| `bf16-causal-d128` | `aiter` | 2.249573e-06 | 7823.789 | 562.155 | 36.455 |
| `bf16-causal-d192` | `bf16_942` | 2.389287e-06 | 8929.439 | 615.686 | 39.926 |
| `bf16-causal-d192` | `bf16_942_grid` | 2.389287e-06 | 8759.943 | 627.598 | 40.698 |
| `bf16-causal-d192` | `aiter` | 2.247466e-06 | 8605.850 | 638.836 | 41.427 |
| `bf16-mha-long-p32` | `bf16_942` | 2.780976e-06 | 2878.425 | 596.850 | 58.286 |
| `bf16-mha-long-p32` | `bf16_942_grid` | 2.780976e-06 | 2915.681 | 589.223 | 57.541 |
| `bf16-mha-long-p32` | `aiter` | 2.697695e-06 | 3333.071 | 515.437 | 50.336 |
| `bf16-mha-short-p32` | `bf16_942` | 2.766049e-06 | 234.294 | 458.288 | 223.774 |
| `bf16-mha-short-p32` | `bf16_942_grid` | 2.766049e-06 | 233.413 | 460.018 | 224.618 |
| `bf16-mha-short-p32` | `aiter` | 2.571839e-06 | 256.727 | 418.242 | 204.220 |
| `bf16-mha-short-p64` | `bf16_942` | 2.768248e-06 | 235.415 | 456.106 | 222.708 |
| `bf16-mha-short-p64` | `bf16_942_grid` | 2.768248e-06 | 232.932 | 460.969 | 225.082 |
| `bf16-mha-short-p64` | `aiter` | 2.572041e-06 | 256.747 | 418.209 | 204.204 |
| `bf16-mha-long-p32-d192` | `bf16_942` | 2.786838e-06 | 3689.849 | 581.998 | 56.836 |
| `bf16-mha-long-p32-d192` | `bf16_942_grid` | 2.786838e-06 | 3677.091 | 584.017 | 57.033 |
| `bf16-mha-long-p32-d192` | `aiter` | 2.703956e-06 | 3667.336 | 585.570 | 57.185 |
| `bf16-mha-short-p32-d192` | `bf16_942` | 2.797057e-06 | 287.755 | 466.431 | 227.750 |
| `bf16-mha-short-p32-d192` | `bf16_942_grid` | 2.797057e-06 | 285.050 | 470.856 | 229.910 |
| `bf16-mha-short-p32-d192` | `aiter` | 2.600875e-06 | 263.819 | 508.750 | 248.413 |
| `bf16-mha-short-p64-d192` | `bf16_942` | 2.798525e-06 | 290.238 | 462.440 | 225.801 |
| `bf16-mha-short-p64-d192` | `bf16_942_grid` | 2.798525e-06 | 285.070 | 470.823 | 229.894 |
| `bf16-mha-short-p64-d192` | `aiter` | 2.601774e-06 | 263.738 | 508.906 | 248.489 |

本轮默认persistent最高 **615.686 TFLOPS**，普通grid最高 **627.598 TFLOPS**，均为D192 causal场景。
相对同shape AITER prepared ASM，D128五个非smoke场景的两种调度均更快，延迟下降 **4.97%–13.64%**；D192五个非smoke场景均略慢至较慢，延迟增加 **0.27%–10.05%**。
MHA long的D128/D192 persistent为 **596.850 / 581.998T**，普通grid为 **589.223 / 584.017T**；这些是MI325本轮数据，不套用MI308的250T验收或跨机roofline结论。
正确性通过不等于达到roofline或性能门槛；本轮没有指定dense参考，也没有追加前后kernel对照测试。

### 2.2 FP8 MHA：全部7条

**历史保留，未在本轮运行。** 以下为`4adaf81`轮次2026-09-08 03:49:34–04:07:11 UTC的CLI数值，来源保存在 [历史原始报告](results/mi325_bf16_e7b4be8_20260908T081608Z/history/all.json)。

| case | backend | acc | 时间µs | TFLOPS | 逻辑GB/s |
|---|---|---:|---:|---:|---:|
| `fp8-full-token-d128` | `fp8_942` | 0.000381311 | 264.379 | 819.572 | 240.472 |
| `fp8-full-token-d192` | `fp8_942` | 0.0003829935 | 288.776 | 937.914 | 257.040 |
| `fp8-causal-token-d128` | `fp8_942` | 0.0002756518 | 5807.743 | 757.296 | 36.110 |
| `fp8-causal-token-d192` | `fp8_942` | 0.0002730074 | 6478.657 | 848.590 | 37.873 |
| `fp8-full-tensor-d128` | `fp8_942` | 0.0003780793 | 162.085 | 662.456 | 198.122 |
| `fp8-full-tensor-d192` | `fp8_942` | 0.0003830103 | 279.362 | 960.889 | 265.676 |
| `fp8-causal-tensor-d192` | `fp8_942` | 0.0002732739 | 6464.434 | 850.457 | 37.956 |

历史FP8仅与独立FP32 O检查正确性，**没有外部性能参考**；本轮不宣称FP8重测或相对BN32加速。两个D128 Full的H/KV也不同，不能只归因于scale模式。

### 2.3 SWA：全部20条（已去重）

**历史保留，未在本轮运行。** 数值与第2.2节来自同一份历史CLI，不改写原样本。
两个KV128K重复参数项已删除，仍为8个独立case、20候选；这些不计入本轮BF16的12 case/34候选/1700样本。

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

历史六个长shape的direct比完整gather+CK **延迟低54.01%–69.86%**；不是本轮SWA性能结论。prepared CK不含gather，仅作分解参考。
gather 逐次搬运完整 KV，不裁去窗口外前缀；实际为一个 Triton gather 加一个 CK dispatch，一对 event 覆盖两段。
每组 20 个 K/V workspace 地址及 10 个 slot mapping 地址独立，prepared 与 gather 的 workspace 也不复用。

### 2.4 整轮与分类耗时

| suite | case执行数 | 候选数 | event样本数 | case总wall time之和（秒） |
|---|---:|---:|---:|---:|
| `bf16-mha` | 12 | 34 | 1700 | 129.662 |
| **本轮合计** | **12** | **34** | **1700** | **129.662** |

本轮同一个pytest进程执行两个指定BF16函数：**16通过、0失败、0跳过，52项未选择**，suite **172.245秒**、进程 **174.236秒**，退出码0。
功能4项通过，性能12项通过；新版gfx942功能同时支持grid和persistent，已没有上一轮的2个persistent架构跳过。没有独立CLI二次计时、FP8/SWA或专项测试。
选卡 **10.695秒**已包含在进程时间内：3条采样gfx/UMC均0%，无拒绝样本；健康设备忙时仍无总等待截止时间。
性能case前后共24条负载记录，最大gfx11%/UMC0%，存在驻留进程；门槛只限制选卡/开始，不把自身计算活动作为争用错误，不保证测量期间持续低于2%。
case wall合计不含功能测试、选卡、环境查询和报告保存，不能与进程时间再次相加，也不等于event时间之和。
129.662秒包含新specialization首调/编译，不能称为纯热缓存耗时。完整逐case阶段、原始样本与JUnit见 [本轮BF16结果](results/mi325_bf16_4f3820e_20260908T234624Z/bf16-audit.json) 和 [pytest日志](results/mi325_bf16_4f3820e_20260908T234624Z/pytest.log)。

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
	计时覆盖完整调用及launch间隙；新版BF16两种调度均无共享counter分配/初始化，每次实测只有一个attention dispatch，旧版counter开销未人为扣除。
	`cudaPerf` GPU spin在起始event前。10-buffer不等于强制清cache。

本轮BF16性能已离线审计：**340个buffer精度检查、680次额外逐位检查、1700个event**，输入Q/K/V/scale/indices及各候选O地址独立，每轮0–9索引和所有中位数匹配。
这不是额外GPU pytest项，不把离线审计计入功能测试通过数。

## 4. 参考与SWA路径

| 候选 | 含义 | 计时范围 |
|---|---|---|
| `bf16_942` / `bf16_942_grid` | 同一8-wave/8-stage核心，persistent / 普通grid | 直接读取SHUFFLE-5D分页KV的完整调用 |
| `fp8_942` / `swa_bf16` | 自有FP8 LDS / BF16单wave；本轮未测 | 直接读取SHUFFLE-5D分页KV的完整调用 |
| `aiter` BF16 Full/Causal/MHA | public varlen，本轮命中ASM | prepared linear KV，不含转换 |
| `aiter` SWA | prepared CK varlen | 不含gather |
| `aiter_gather` SWA | **完整KV gather+CK** | 每次重新gather全部KV再执行CK，**一对event覆盖两段** |

候选规则与新版MI308一致：所有BF16场景均含persistent和普通grid，小shape无外部参考，其余必须AITER；FP8/SWA规则仅保留说明，本轮未执行。
声明参考缺依赖或首调失败会停止，不静默省略；数值错误始终失败。BF16 dense/FP8 BN32适配已删除，独立FP32 O检查仍覆盖全部buffer。
gather不裁SWA前缀，不用两个独立均值相加，不进入生产dispatch；运行时检查完整gather内容、独立workspace及两个dispatch。

本轮BF16实际命中ASM `fmha_fwd_hd128_bf16_rtna_group` / `fmha_fwd_hd192x128_bf16_rtna_group`及对应causal变体；SWA的CK `BlockFmhaPipelineQRKSVSAsync`记录仅属于历史轮次。
完整dispatch名称保存在JSON，不能仅凭AITER入口名推断具体实现。新版BF16不再使用旧atomic ticket counter，34条候选均实际核对为单dispatch。

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
**本轮实际使用指定主入口的BF16功能+性能pytest，不建立另一套测试入口。**
当前使用`--max-gpu-utilization` / `PYHIP_MHA_MAX_GPU_UTILIZATION`的低负载启动语义，不恢复旧快照的逐case稳定等待。

```bash
PY=/root/.venvs/pyhip-mha-mi325-quanta-20260907/bin/python
MHA=/host_lc/pyhip/tests/flydsl/mha/test_mha_pa.py
OUT=$(mktemp -d "$PWD/tests/flydsl/mha/results/mi325-bf16.XXXXXX")
export HOME=/host_lc/pyhip-mha-mi325-storage-20260908/home
export XDG_CACHE_HOME="$HOME/.cache" TMPDIR=/host_lc/pyhip-mha-mi325-storage-20260908/tmp
export TMP="$TMPDIR" TEMP="$TMPDIR"
export PYHIP_MHA_GPU=3 PYHIP_MHA_REQUIRED_PTL=current PYHIP_MHA_MAX_GPU_UTILIZATION=2
export ROCR_VISIBLE_DEVICES=GPU-9ff0c5f642c689c2 HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0
export GPU_ARCHS=gfx942 MAX_JOBS=4
export AITER_JIT_DIR="$XDG_CACHE_HOME/pyhip-mha-mi325-quanta/aiter-jit-gpu3"
export PATH="$(dirname "$PY"):$PATH" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH COMPILE_ONLY ARCH CUDAPERF PYHIP_MHA_COMPILE_CACHE PYHIP_MHA_IDLE_MAX_UTILIZATION
unset FLYDSL_RUNTIME_ENABLE_CACHE FLYDSL_RUNTIME_CACHE_DIR FLYDSL_COMPILE_ONLY FLYDSL_COMPILE_ARCH FLYDSL_DUMP_IR FLYDSL_DUMP_DIR FLYDSL_RUN_ONLY FLYDSL_RUNTIME_RUN_ONLY
unset TRITON_CACHE_DIR TORCH_EXTENSIONS_DIR PYHIP_MHA_PERF PYHIP_MHA_OUTPUT PYHIP_MHA_SELECTION_LOG

# 可选GPU-free查询：只列出已有BF16场景。
"$PY" "$MHA" --suite bf16-mha --list

# 本轮实际方式：仅既有BF16功能与性能项，默认10buffer/5轮/warmup10/repeat1。
PYHIP_MHA_PERF=1 PYHIP_MHA_OUTPUT="$OUT/performance" PYHIP_MHA_SELECTION_LOG="$OUT/selection.idle.jsonl" "$PY" -m pytest "$MHA" --import-mode=importlib -k 'test_bf16_mha or test_perf_bf16_mha' -q --durations=0 --junitxml="$OUT/tests.xml"
```

`-k`只选择主文件中的`test_bf16_mha`和`test_perf_bf16_mha`：新版共16个参数项（4功能+12性能），其他52项不运行。
12个性能case及persistent/grid/AITER候选直接来自更新后的`BF16_MHA_PERF_CASES`和候选规则，不在本机手工增删参数或引入额外参考。
性能pytest为每case保存一对JSON/Markdown，独立保存JUnit和选卡日志；不覆盖旧文件。
CLI的`--suite`/`--case`仅用于选择预定义场景，不改变shape或参考；pytest使用标准`-k`或节点ID，不使用CLI的`--case`。
本轮没有另外运行CLI；新增/修改shape、候选、参考或专项前必须先确认，不能在复测中自行扩展。

默认门槛0仍要求gfx/UMC为0且无其他进程；本机显式2允许驻留进程，连续3次、5秒间隔均低于2%才启动，健康但忙时无总等待截止时间。
正门槛只限制选卡/开始，之后case前后记录负载，不重新等待，也不因自身GPU活动拒绝结果。
本机仅授权GPU3，不使用不限池的auto选卡；用户级锁不是独占预约，采样不能保证测量期间完全无争用。
本机适配按KFD `domain/location_id`匹配BDF取得ROCr UUID，并核验实际torch BDF；PCI sysfs的ASIC serial不能直接当此机ROCr UUID。

## 7. 显式case与pytest范围

统一B1、Dv128、contiguous Q/O、零尾页、seed20260905；除表内注明外，H16/HK1、page64、Q scale为per-token。
BF16使用unit descales，FP8从BF16源量化；NC为full/noncausal，C为bottom-right causal，BF16/FP8 full和causal无窗口、无sink。
**本轮只执行第7.1节及BF16功能项；第7.2–7.3节仅保留既有集合说明。**

### 7.1 BF16 MHA：12 case / 34候选

`test_perf_bf16_mha`使用远端已有`BF16_MHA_PERF_CASES`；前两项各persistent+grid两个自有候选，其余10项各`bf16_942`+`bf16_942_grid`+AITER，共34条。

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

原三个MHA shape及其远端新增D192配对的Q/KV/H/page均固定；long仍Q/KV20480，不按MI325的304 CU放大为77824。
这是新版已有集合的变化，不新增本机专项；D192算量比同shape D128多25%，吞吐比较不等于延迟应相同。

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

没有目录conftest插件；性能函数的`__test__`属性控制收集：新版默认41项功能，`PYHIP_MHA_PERF=1`时额外收集27项性能，共68项。
不需自定义pytest参数或marker。本轮仅用`-k 'test_bf16_mha or test_perf_bf16_mha'`，在同一进程运行BF16子集：

| 功能函数 | 参数项数 | MI325适用性 | 主要检查 |
|---|---:|---|---|
| `test_bf16_mha` | D128/192 × 8wave/persistent = 4 | 本轮4项通过；gfx942中8wave标签对应普通grid | ragged、空Q、尾页、causal、GQA、非单位scale |
| `test_bf16_stage_boundaries` / `test_bf16_stage_operands` | 10 + 3 | 本轮未选择 | 新版stage边界与定值operand检查 |
| `test_bf16_layout_contract` / `test_bf16_streams_and_graph` | 12 + 2 | 本轮未选择 | 新版layout/LSE、streams/graph专项 |
| `test_fp8_mha` | D128/192 × C/NC × 两种Q scale = 8 | 本轮未选择 | ragged、空Q、NaN尾页、量化scale |
| `test_swa` | D128/192 = 2 | 本轮未选择 | 空KV、W128+sink、padded Q/O、NaN尾页、非单位scale |

本轮 **16通过、0跳过、52项未选择**（172.245秒）：指定BF16功能4项通过，性能12项通过。
未选择的52项包括27个新版BF16专项、10个FP8/SWA功能项及15个FP8/SWA性能项；不把远端MI308的68项通过计作本机结果。
本机没有新增测试、独立gather或repeat3专项；BF16性能小shape是12项中的2项，不能另加到4个功能参数项里。
准确单行命令见第6节。性能pytest与CLI共享集合和候选规则，正利用率门槛须配合auto或物理GPU编号，不能用current绕过起始负载检查。
列表/收集不初始化GPU，普通功能测试不做性能event计时；后续扩展测试范围须先确认。

## 8. 原生缓存、输出与跨机边界

**已启用编译缓存，在排查问题时需要检查缓存是否出现问题**。使用FlyDSL原生缓存，默认开启，不增加MHA私有持久层或额外缓存CLI开关。
本机根overlay写入`ENOSPC`，用户授权使用工作盘：仅验证子进程的HOME、XDG缓存和TMPDIR改到工作盘，原根目录/全局环境不动。
已有FlyDSL/Triton/AITER缓存复制后分别48/66/916个文件SHA256一致；原缓存保留，没有清空或移动，测试继续使用这些副本。
FlyDSL仍按默认HOME相对目录解析，`env.runtime.enable_cache=True`，无`FLYDSL_*`覆盖；**实际物理目录已变化，这是本机显式存储适配**。
不修改原生key、编译锁或失效规则，不把“开启”当作全部命中，不称纯冷/全热。异常输出、加载失败或反复编译时检查版本、实际缓存目录和shell残留覆盖，不能减少buffer或跳过正确性。

最新BF16以FP8为蓝本重建BM256/BN64、8-wave/8-stage核心，访存与计算分阶段；V使用64-bit GLOBAL地址，默认persistent为固定驻留grid-stride，普通grid复用同一核心，无旧ticket计数器。
本轮没有修改生产优化或开展额外IR/ISA/LSE/ATT专项；只移植选卡必需的KFD UUID解析，输入/FP32参考/候选/测量等既有函数AST与更新后的HEAD一致。
BF16生产源码及测试源码SHA与12份报告逐一核对；原生编译缓存保持开启，不清理/切换缓存、不以私有层替代原生依赖机制，不声称所有编译产物均命中。
本机PCI unique_id为ASIC serial，KFD unique_id才是已核验ROCr标识；修正只影响选卡，严格默认、3次启动采样、PTL及实际BDF核验不变。

JSON的wall time区分选卡/执行，每case另列输入、FP32参考、参考设置/首调、校验、dispatch profiling、warmup和measurement。
首调可能含缓存加载/JIT，不是纯编译时间；measurement wall含spin/Python开销，不等于event样本之和。
正常流程不打印选卡/准备/校验进度，性能表和`--verbose-runs`保留；错误诊断、dispatch及idle采样独立保存。
本轮用bash内建`time`记录完整pytest子进程，退出码0；测试期间没有修改执行脚本或源码。
测试结束后的只读ECC corrected/uncorrected/deferred均0，瞬时gfx/UMC均0%；这是离散快照，不是持续空闲或独占证明。
此后未追加GPU smoke、CLI、专项或其他类别测试，也未终止其他进程；本轮自有pytest及选卡监控均已退出。
MI325不能验证gfx950原生执行；对应平台见 [README.355.md](README.355.md)，不能把skip当pass。10-buffer OOM须如实报告，不自动缩小规模。

本轮证据见 [BF16结果目录](results/mi325_bf16_4f3820e_20260908T234624Z)、[JUnit](results/mi325_bf16_4f3820e_20260908T234624Z/tests.xml)、[离线审计](results/mi325_bf16_4f3820e_20260908T234624Z/bf16-audit.json)。
快进更新前，原暂存的选卡修复和未暂存变更记录已完整归档并保存到stash **`884b96fbcee56cb5e3394f7625351ee44de27505`**；[更新基线](results/mi325_bf16_4f3820e_20260908T234624Z/baseline.txt)记录归档路径和SHA256，没有盲目重放旧stash。
更新后的原MI325正文另备份为 [更新前报告](results/mi325_bf16_4f3820e_20260908T234624Z/README.325.before.md)。上一轮`e7b4be8`报告与历史FP8/SWA来源仍保留在原结果目录，数值未改写。
第2.1节只用本次指定入口BF16 pytest数据，第2.2–2.3节明确为历史；不能将两种轮次合并宣称全套最新验证。