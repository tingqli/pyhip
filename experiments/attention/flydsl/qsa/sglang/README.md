# 临时SGLang对照与接入

此目录未来可整体移除；QSA运行时不依赖它。

| 文件 | 用途 |
|---|---|
| [baseline.py](baseline.py) | 两个原SGLang Triton kernel、原launch表、来源/SHA/版权、tensor adapter合一，仅供测试 |
| [plugin.py](plugin.py) | 原生entry-point五个hook，单调用QSA、原精度校验、可选真实输入采集；同时提供`--build-target`构建入口 |
| [profile_model.py](profile_model.py) | 一次完成新包构建、原TP2启动/warmup/profile、双rank结果验证和自身进程组清理 |

插件只替换gfx942、BF16 D256、Q12/6/3 KV1的main-runner eager EXTEND；decode、draft/verify、graph/compile、CP/DCP及不支持的attention选项仍走原实现。
原backend继续负责KV写入/gather、有效query裁剪和padding。只替换最终attention，不替换indexer。
启用须同时设置`PYHIP_QSA_PREFILL=1`与`SGLANG_PLUGINS=pyhip_flydsl_qsa`；三个上游文件SHA必须匹配。

构建产物是包含dist-info的独立`pyhip_qsa_runtime`包，版本0.2.0；不需pip、不修改SGLang、不把experiments放入server路径。
父包惰性导入，禁用时不导入Torch/FlyDSL。仅打包QSA四模块、MHA两个依赖与本插件，附来源hash和许可证。
`--build-target`要求新mytest/mydata子目录；打包目标本身作为独立源码快照保留。

## 1. 运行前准备

以下命令针对本机已有ROCm环境、Qwen3.8模型和**TP2 / GPU0、1 / 端口9080**。不要激活缺少Torch/FlyDSL的PyHIP虚拟环境，也不需要重新安装依赖。
每个新终端先执行：

```bash
ROOT=/opt/lc/pyhip
DATA="$ROOT/mytest/mydata"
QSA_SGLANG="$ROOT/experiments/attention/flydsl/qsa/sglang"
export PATH="/usr/bin:/usr/local/bin:/bin:/opt/rocm-7.14/bin:$PATH"
export PYTHONDONTWRITEBYTECODE=1
unset HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL
unset HSA_CU_MASK ROC_GLOBAL_CU_MASK
cd "$ROOT"
```

原启动脚本内部设置可见卡为0、1、2、3，TP2实际用0、1；下面不是任意GPU映射或TP8启动范例。
请先确认两卡空闲、9080没有其他服务。三个上游文件SHA若变化，需重新检查接入兼容性，不能只删除校验。

## 2. 一键构建、启动并profile（推荐）

执行[profile_model.py](profile_model.py)即可，**不必先手动构建插件或启动服务**：

```bash
OUT="$DATA/qsa_profile_$(date -u +%Y%m%dT%H%M%SZ)_$$"
/bin/python3 "$QSA_SGLANG/profile_model.py" --output "$OUT"
```

输出目录必须尚不存在，不要预先`mkdir "$OUT"`。流程为：

1. 从当前源码构建独立插件，设置`PYTHONPATH`、`SGLANG_PLUGINS`和`PYHIP_QSA_PREFILL=1`。
2. 调用原TP2启动脚本；等日志中的ready事件，不轮询或sleep。
3. 调用用户的warmup脚本，另预热11888/12000行；开启`PYHIP_QSA_VALIDATE=1`，在profile外以原`.02`容差校验。
4. 调用原profile脚本：4条名义12000→5请求、并发1，验证两rank trace和每rank48次QSA替换。
5. 保存命令、源码身份、实际server配置、调用统计和GPU状态；清理本轮自身进程组。

**这是一次性采集命令，结束后不会留下常驻服务。** 要保留服务，用第3节。输出检查：

```bash
cat "$OUT/capture_status.json"
cat "$OUT/qsa/qsa_tp0.json" "$OUT/qsa/qsa_tp1.json"
find "$OUT/profiles" -name '*-TP-*.trace.json.gz'
```

成功时终端打印`PROFILE_COMPLETE=...`，状态中的`complete`、`cleanup_verified`、`external_sources_unchanged`均为true。
GPU门禁失败就停止，不轮询等待、不修改PTL/时钟/功率；新数据、日志与运行时缓存都定向到mytest/mydata。

可选变体，每次使用新的输出目录，**顺序执行，不要并发运行**：

```bash
# 同时捕获两层真实Q/K/V、indices和output；默认行数11888/12000。
/bin/python3 "$QSA_SGLANG/profile_model.py" \
	--output "$DATA/qsa_inputs_$(date -u +%Y%m%dT%H%M%SZ)_$$" \
	--dump-layers 3 47

# 不启用插件，采集原SGLang基线。
/bin/python3 "$QSA_SGLANG/profile_model.py" \
	--output "$DATA/qsa_baseline_$(date -u +%Y%m%dT%H%M%SZ)_$$" \
	--baseline
```

输入仅在profile期间克隆，CPU转存发生在原profiler停止/导出之后；文件含tensor SHA与布局hash。
**输入克隆会扰动trace**；正常profile不传`--dump-layers`，脚本也会清除继承的dump环境变量。

## 3. 仅启动集成QSA的常驻服务

### 3.1 构建并设置服务端环境

接第1节，在启动终端执行。缓存及AITER配置与一键脚本一致，不向SGLang源码目录写运行产物：

```bash
RUN="$DATA/qsa_serve_$(date -u +%Y%m%dT%H%M%SZ)_$$"
mkdir -p "$RUN/tmp" "$RUN/profiles"
/bin/python3 "$QSA_SGLANG/plugin.py" --build-target "$RUN/plugin"

export PYTHONPATH="$RUN/plugin"
export SGLANG_PLUGINS=pyhip_flydsl_qsa PYHIP_QSA_PREFILL=1
export PYHIP_QSA_VALIDATE=1 PYHIP_QSA_REPORT_DIR="$RUN/qsa"
unset PYHIP_QSA_DUMP_LAYERS PYHIP_QSA_DUMP_ROWS PYHIP_QSA_DUMP_DIR
export LOG_FILE="$RUN/server.log" TMPDIR="$RUN/tmp"
export SGLANG_TORCH_PROFILER_DIR="$RUN/profiles"
export SGLANG_PROFILE_WITH_STACK=1 SGLANG_PROFILE_RECORD_SHAPES=1

CACHE="$DATA/.cache"
export XDG_CACHE_HOME="$CACHE"
export SGLANG_CACHE_DIR="$CACHE/sglang" SGLANG_JIT_CACHE_DIR="$CACHE/sglang_jit"
export TORCHINDUCTOR_CACHE_DIR="$CACHE/sglang/inductor"
export TRITON_CACHE_DIR="$CACHE/triton" FLYDSL_RUNTIME_CACHE_DIR="$CACHE/flydsl"
export TORCH_EXTENSIONS_DIR="$CACHE/torch_extensions" AITER_JIT_DIR="$CACHE/aiter"
CFG="$ROOT/mytest/sglang_tp2_base_20260925_01/aiter_configs_snapshot"
export AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE="$CFG/a8w8_bpreshuffle_tuned_gemm.csv"
export AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE="$CFG/a8w8_blockscale_bpreshuffle_tuned_gemm.csv"
export AITER_CONFIG_GEMM_BF16="$CFG/bf16_tuned_gemm.csv" AITER_CONFIG_FMOE="$CFG/tuned_fmoe.csv"
```

`PYTHONPATH`指向构建目标根目录，不是本源码目录，更不能指向这个名为sglang的子目录。
修改QSA源码后须构建新target并重启服务；已构建target是冻结副本，不自动跟随源码变动。
`PYHIP_QSA_VALIDATE=1`只在profile外对每layer/layout首次调用做对照校验，确认稳定后常驻服务可设0；不是开关插件。

### 3.2 启动TP2服务

```bash
printf 'RUN=%s\n' "$RUN"
cd "$RUN"
TP_SIZE=2 HOST=127.0.0.1 PORT=9080 \
MODEL_PATH=/models/Qwen3.8-Flash-Next-PTPC-FP8 \
SERVED_MODEL_NAME=Qwen/Qwen3.8-Flash-Next-PTPC-FP8 \
MEM_FRACTION_STATIC=0.95 CHUNKED_PREFILL_SIZE=16384 \
MAX_RUNNING_REQUESTS=32 CUDA_GRAPH_MAX_BS_DECODE=32 \
PLE_OFFLOAD_EMBEDDING=0 AITER_MOE_PADDING_SIZE=64 \
bash /opt/sglang/scripts/launch_qwen38_flash_next_fp8_mi308x_pure_tp_4_or_8_or_2.sh
```

保持此终端运行；等待`The server is fired up and ready to roll!`。日志应出现
`PyHIP QSA enabled: eager EXTEND, auto4/dense2051/sortedBN32`，且没有ABI mismatch或hook加载错误。
启用日志表示注册成功；实际替换还要看请求后的profile调用统计。仅服务ready不能证明QSA已接入，因为SGLang会记录插件加载异常后继续启动。
需要停止时在该前台终端按Ctrl-C；不要全局pkill其他SGLang服务。

## 4. 对上述常驻服务发请求、profile

另开终端，先执行第1节，再把`RUN`设为启动终端打印的真实路径；**不要再运行第2节，它会另起服务并占用同一端口**。

```bash
RUN="/opt/lc/pyhip/mytest/mydata/qsa_serve_..."  # 替换为启动终端打印的RUN
cd "$RUN"
/bin/python3 /opt/evaluation7/check_acc_long_oai.py > "$RUN/warmup.log" 2>&1
cat "$RUN/warmup.log"

BENCH_MODEL=/models/Qwen3.8-Flash-Next-PTPC-FP8 \
BENCH_HOST=127.0.0.1 BENCH_PORT=9080 \
INPUT_TOKENS=12000 OUTPUT_TOKENS=5 NUM_PROMPTS=4 MAX_CONCURRENCY=1 DATASET_NAME=random \
PROFILE_LOG_FILE="$RUN/profile.log" SGLANG_TORCH_PROFILER_DIR="$RUN/profiles" \
bash -o pipefail /opt/evaluation7/run_pure_text_profile.sh

cat "$RUN/qsa/qsa_tp0.json" "$RUN/qsa/qsa_tp1.json"
find "$RUN/profiles" -name '*-TP-*.trace.json.gz'
```

这只向已运行服务发送请求，**不会停止服务**。服务端的profiler变量必须在启动前设置，仅在客户端设置不能代替第3.1节。
报告的`calls_per_layer`应非空；固定工作负载通常为12层各4次。manual流程没有一键脚本的精确11888/12000行预热与门禁校验，首次trace可能含JIT；正式对比优先用第2节。
当前插件报告用排他创建，**同一服务/报告目录只采集一次**；再次profile应换新的`RUN`并重启，不能复用旧报告冒充新结果。

## 验证范围

0.2.0独立包的原生5hook、惰性加载和两份真实输入执行已通过，见[整理记录](../../../../../mytest/mydata/qsa_consolidation_20260925_01/README.md)。
本页命令与当前CLI/环境逐项核对；此次只补文档，没有重新启动整模型，不把旧模型trace标作新插件的端到端验收。
profiler trace不是无profiler吞吐测试；bench duration包含trace导出等待，不能直接解释为常规请求耗时。

真实输入加载、FP32参考、性能、summary均已并入[../test_qsa.py](../test_qsa.py)，没有第二套回放或插件测试文件。
原长期上下文/模型指标测试未由这个快速profile流程替代。