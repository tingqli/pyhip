# 临时SGLang对照与接入

此目录未来可整体移除；QSA运行时不依赖它。

| 文件 | 用途 |
|---|---|
| [attention_baseline.py](attention_baseline.py) | 两个原SGLang Triton kernel、原launch表、来源/SHA/版权、tensor adapter合一，仅供attention测试 |
| [plugin.py](plugin.py) | 原生entry-point五个3D hook＋可选prefill indexer hook＋可选decode indexer hook，单调用QSA、全量FP32参考校验、可选真实输入采集；同时提供`--build-target`构建入口 |
| [attention_validation.py](attention_validation.py) | opt-in attention服务校验的独立selected-token FP32参考；QK后缩放、FP32概率/PV，不依赖QSA规划器，不替换服务输出 |
| [profile_model.py](profile_model.py) | 一次完成新包构建、原TP2启动/warmup/profile、双rank结果验证和自身进程组清理 |

插件只替换gfx942、BF16 D256、Q12/6/3 KV1的main-runner eager EXTEND；decode、draft/verify、graph/compile、CP/DCP及不支持的attention选项仍走原实现。
原backend继续负责KV写入、有效query裁剪、padding及3D prefix gather；attention只消费3D K/V。默认不替换indexer。
启用须同时设置`PYHIP_QSA_PREFILL=1`与`SGLANG_PLUGINS=pyhip_flydsl_qsa`；三个上游源码SHA必须匹配（backend、kernel、qsa_indexer）。

**命名整理（2026-09-29）。** 总包仍为`qsa`，同时包含attention和indexer；插件现在惰性导入`qsa.attention`并调用`attention()`。attention侧六个运行时文件统一为`attention_*`命名（公共入口与私有packed例外见[映射](../opt.md#qsa-attention-names)），参考模块也加attention前缀。共享插件/报告包含两部分，故保留`pyhip_qsa_runtime`、`pyhip_flydsl_qsa`、`PYHIP_QSA_*`及`qsa_tp{rank}.json`，不错误地把indexer或SGLang QSA协议改名。新profile的attention分段为`pyhip_qsa.attention.prepare`与`pyhip_qsa.attention.compute`，设备符号为`attention_*`；旧trace、包和来源SHA不回写。部署更新须新建target，旧包不会自动获得新名称。

**QSA-T13校验修复（2026-09-29）。** 原TP2 C32负载已复现layer47、`query_lens=(11936,4416)`、`prefix_lens=(64,0)`的同一元素差异；错误来自原SGLang参考把`Q*scale*log2(e)`提前量化为BF16。独立FP64约−0.000000656，QSA为−0.001220703，原参考为+0.020141602（原参考自身不满足`.02/.02`）。因此不改变attention输出，而改正验证oracle：`PYHIP_QSA_VALIDATE=1`仍对每个首次layer/layout的**全部元素**做`rtol=atol=.02`检查，比较QSA转FP32与独立FP32 selected-token结果；同跑旧SGLang比较并在`legacy_close/legacy_error`保留差异。新参考失败仍抛错，不能以旧参考相同为由放行；不做输出替换或额外路由。新模块只在验证分支惰性导入，关闭校验或profile时原热路径不变。

失败时先把Q/K/V、indices、output、FP32 expected、legacy expected与layer/rank/长度/scale/tensor hash保存到`PYHIP_QSA_DUMP_DIR`，否则使用`PYHIP_QSA_REPORT_DIR/failures`；然后重新抛出原assert。未配置目录只记录位置；落盘失败也不会吞掉数值断言。默认capture仍只在profile期间克隆，**数值失败落盘不受profile/row过滤器限制**。真实失败的全50,233,344元素、新参考63行CPU FP64及独立包TP2/4/8检查见[参考证据](../../../../../mytest/mydata/qsa_t13_20260929_01/reference_proof/result.json)和[独立包结果](../../../../../mytest/mydata/qsa_t13_20260929_01/package_check_same_shape/result.json)。原[入口门禁失败](../../../../../mytest/mydata/qsa_t13_20260929_01/fixed_service/status.json)保留；用户授权续测后，修复后的实际TP2服务现已完成C1 128请求、C32预热32＋3轮96请求和双rank profile，全部成功。两rank各77布局×12层=924项全量FP32 `.02/.02`通过，原失败布局及同一旧参考差异再次出现但准确检查通过；TP0另有1条旧对照差异，均保留日志。见[服务审计](../../../../../mytest/mydata/qsa_t13_20260929_02/analysis.json)与[闭环记录](../opt.md#qsa-t13-service-closure)。此运行启用昂贵校验及新layout JIT，不作为生产性能；服务已清理，新部署仍须显式构建包含验证模块的14源码target。

**可选prefill indexer（2026-09-28）。** `PYHIP_QSA_INDEXER=1`（默认0）时再AROUND hook `QSAIndexer.forward_cuda`，只接eager EXTEND、gfx942、BF16、4×128头、ratio 4、top512/2048、NeoX rotary 64、单请求≤65536个压缩key（262144 token，模型上限）的调用，其余走原实现；另校验9个上游模块SHA（metadata、mqa、qsa_kv_pool、mrope、rotary utils、layernorm、minimax rmsnorm、fast_topk、qwen4_exp）。`PYHIP_QSA_INDEXER_GEMM=hipblaslt|sglang`（默认hipblaslt）：前者在行数>5120时用固定hipBLASLt solution 90517，按kernel名SHA256核验、不符则回退，GEMM舍入与SGLang不同；后者保持SGLang GEMM，prep逐bit一致。`PYHIP_QSA_VALIDATE=1`时每个(layer,长度)首次调用用SGLang GEMM同跑原路径对照。`PYHIP_QSA_INDEXER_DUMP_LAYERS=3,47`在profile中采集indexer真实输入。logits/top-k为FlyDSL kernel，全部形状共用一次编译；每进程首次调用JIT，空`FLYDSL_RUNTIME_CACHE_DIR`约1.0 s、已有缓存约0.06 s。性能、正确性与系统结果见[../opt.md](../opt.md)。

**可选decode indexer（2026-09-29）。** `PYHIP_QSA_INDEXER_DECODE=0|select|1`（默认0，与`PYHIP_QSA_INDEXER`独立），使用同一组上游SHA校验。

- `select`（第一阶段）：AROUND hook `QSAIndexer.select_decode_tokens`，eager decode和CUDA graph capture都接。要求为gfx942；module为4×128头、ratio 4、top512/2048；q为连续的BF16 [rows,4或8,128]（8头是fused prep的零padding，只读前4头）且rows<65536；压缩cache为连续的BF16 [pages,16,1,128]且小于2 GiB；页表为连续的int32 [rows,P]、宽度为16P；lengths为int32。不满足时走原实现。只替换logits：FlyDSL分页kernel按设备端长度读每行自己的压缩key；SGLang原`fast_topk`（row_starts改用常驻零buffer）和Triton expand不变，每层graph从16个kernel减到3个。grid由静态形状决定，没有host同步，由SGLang capture前的两次eager warmup完成FlyDSL编译（冷缓存约0.5–0.6 s）。
- `1`（第二阶段）：包含`select`，另由`QSAIndexer.forward_cuda` hook整段接管CUDA graph decode（`ForwardMode.DECODE`、`is_cuda_graph`、graph buffer齐全）。SGLang自己的`index_qk_proj` GEMM之后，一个Triton kernel `_indexer_decode_prep`完成q norm/RoPE、pending ring写入（key与三轴rope位置）、组边界均值压缩＋k norm/RoPE及压缩cache写入（非边界行写slot 0，与SGLang相同），再接第一阶段的分页logits、fast_topk与expand；SGLang每层43个prep kernel变为1个。额外要求：MRotaryEmbedding（非GLM，0或3段）或RotaryEmbedding、NeoX、rotary 64、BF16连续cos/sin cache、int64 positions/ring slots/rope缓冲、BF16 key_state/compressed [.,1,128]、int32 group_locs [rows,4]与write_locs；不满足或eager decode时走SGLang `forward_cuda`（仍经`select` hook）。q、ring、rope与压缩写入与SGLang逐bit一致。

**decode的服务内校验（2026-09-29新增）。** 以前`PYHIP_QSA_VALIDATE=1`不检查decode（capture中不能同步）。现在同时开`PYHIP_QSA_INDEXER_DECODE=1`时，每个graph decode步的每层都在同一CUDA graph内做对照：

- 先跑SGLang自己的`forward_cuda`作参考（其中的`select` hook直通SGLang），记下它写的ring行和压缩行；再用SGLang的`project_qk`重算q；最后跑PyHIP forward，覆盖同样的行。
- 要求逐bit一致：q；真实请求的ring key/rope行（padding行用保留请求0，ring行号<4，排除）；组边界的压缩行（非边界行写slot 0，排除）。
- 两份token选择都要满足：是PyHIP FP32 logits的top-512（相对近并列容差1e-5，与离线FP64标准相同）；完整block数为min(length,512)，每个block 4个token；因果尾token相同。expand输出的布局是：选中block的token，紧接着尾token，其余为−1。
- SGLang自己的decode选择在近并列时不可复现：同一输入连续6次调用中，有1次在4行里有1行集合不同。所以“集合不同”只计入`different_token_sets`，不算失败。
- 结果累加在每层的设备计数器里（int64计数＋最大相对边界间隙），不同步，可capture。下一次eager prefill读取计数，任一失败即抛`PyHIP QSA decode differs from SGLang`。profile停止时写入报告的`indexer_decode_validation`（键`forward:<layer>`）。
- `select`模式的调用同样计数（键`select:<layer>`）。
- 开启后每步decode多跑一遍SGLang indexer和检查，计时不能当性能。

真实TP2验收已通过（2026-09-29，decode=1、VALIDATE=1，原C1/C32协议256请求加profile）：每rank 12层×10300次调用、25644个真实行、6408个边界压缩行，q/ring/压缩行不一致和选择违例全为0；集合不同约0.015%，都是近并列（最大相对间隙1.66e-7）；attention与prefill indexer各336项通过。见[分析](../../../../../mytest/mydata/qsa_indexer_decode_system_20260929_04/analysis_075332.json)和[opt.md](../opt.md#qsa-decode-stage2-validate)。

MTP/target-verify在结构上也会命中`select` hook，但未测试（当前启动脚本未开speculative）。

构建产物是包含dist-info的独立`pyhip_qsa_runtime`包，恢复3D版本0.2.0；不需pip、不把experiments放入server路径。2026-09-28已按用户要求撤回T01的SGLang prefill/decode 5D改动、页表边界及专属测试；既有MHA/cache writer不动。
父包惰性导入，禁用时不导入Torch/FlyDSL。当前打包attention六模块（公共入口、prepare、dense、union、direct、私有packed direct）、indexer四个模块（`indexer.py`、`indexer_logits.py`、`indexer_topk.py`、`indexer_decode.py`）、MHA两个依赖、本插件与attention独立验证模块，共14个源码文件，无预编译二进制，附来源hash和许可证。旧13源码包只作历史，不自动获得T13修复。
`--build-target`要求新mytest/mydata子目录；打包目标本身作为独立源码快照保留。

当前3D packed direct保留17/10填充工作比分流，raw保留rho4；没有撤销3D优化。5D因Vvec8与四token选块的布局成本退化而撤回，性能及Vvec4/S4原型限制见[../opt.md](../opt.md)。当前接入不支持QSA 5D，不应为该路径启用vectorized_5d。旧0.3.0/clean2包与5D回放只作历史证据，不可直接套用回退后的SGLang；新使用需构建新target。

## 1. 运行前准备

以下命令针对Qwen3.8模型和**TP2 / GPU0、1 / 端口9080**，后续默认使用PyHIP的.venv。2026-09-28该环境已独立安装匹配ROCm的Torch/FlyDSL/Triton及SGLang依赖，整服务启动时另补齐镜像sglang-kernel0.4.6.post1的39个精确载荷，仍关闭system-site-packages；不要静默切回系统Python。最新实际TP2原生/当前模型测试已经完成并清理，见[系统审计](../../../../../mytest/mydata/qsa_system_20260928_01/final_analysis.json)；不是常驻部署，版本和来源见[../opt.md](../opt.md)。
每个新终端先执行：

```bash
ROOT=/opt/lc/pyhip
DATA="$ROOT/mytest/mydata"
QSA_SGLANG="$ROOT/experiments/attention/flydsl/qsa/sglang"
PYTHON="$ROOT/.venv/bin/python"
export PATH="$ROOT/.venv/bin:/opt/rocm-7.14/bin:$PATH"
export PYTHONDONTWRITEBYTECODE=1
unset HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL
unset HSA_CU_MASK ROC_GLOBAL_CU_MASK
cd "$ROOT"
```

原启动脚本内部设置可见卡为0、1、2、3，TP2实际用0、1；下面不是任意GPU映射或TP8启动范例。
请先确认两卡空闲、9080没有其他服务。三个上游源码SHA若变化，需重新检查接入兼容性，不能只删除校验。

## 2. 一键构建、启动并profile（推荐）

执行[profile_model.py](profile_model.py)即可，**不必先手动构建插件或启动服务**：

```bash
OUT="$DATA/qsa_profile_$(date -u +%Y%m%dT%H%M%SZ)_$$"
"$PYTHON" "$QSA_SGLANG/profile_model.py" --output "$OUT"
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
"$PYTHON" "$QSA_SGLANG/profile_model.py" \
	--output "$DATA/qsa_inputs_$(date -u +%Y%m%dT%H%M%SZ)_$$" \
	--dump-layers 3 47

# 不启用插件，采集原SGLang基线。
"$PYTHON" "$QSA_SGLANG/profile_model.py" \
	--output "$DATA/qsa_baseline_$(date -u +%Y%m%dT%H%M%SZ)_$$" \
	--baseline

# 同时启用PyHIP prefill indexer（PYHIP_QSA_INDEXER=1）。
# --dump-indexer-layers 3 47采集indexer输入及当时实际路径的输出/pool行；test_indexer的capture不加--indexer（SGLang原indexer）。
"$PYTHON" "$QSA_SGLANG/profile_model.py" \
	--output "$DATA/qsa_indexer_$(date -u +%Y%m%dT%H%M%SZ)_$$" \
	--indexer
```

输入仅在profile期间克隆，CPU转存发生在原profiler停止/导出之后；文件含3D Q/K/V、indices、output、tensor SHA与请求布局hash，不再采集5D缓存或页表。
**输入克隆会扰动trace**；正常profile不传`--dump-layers`，脚本也会清除继承的dump环境变量。

## 3. 仅启动集成QSA的常驻服务

### 3.1 构建并设置服务端环境

接第1节，在启动终端执行。缓存及AITER配置与一键脚本一致，不向SGLang源码目录写运行产物：

```bash
RUN="$DATA/qsa_serve_$(date -u +%Y%m%dT%H%M%SZ)_$$"
mkdir -p "$RUN/tmp" "$RUN/profiles"
"$PYTHON" "$QSA_SGLANG/plugin.py" --build-target "$RUN/plugin"

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

2026-09-26：当前源码已将恢复校验的排序结果复用于direct，移除了独立direct排序，并改进direct计算；linear新增B1非分页causal配平。
本轮仅本地功能/内核验收，未重新部署独立包或采集SGLang整模型profile。新部署须按上节重新build target；旧日志中的排序kernel名称属于历史版本。
`PYHIP_QSA_VALIDATE=1`对attention和prefill indexer只在profile外对每layer/layout首次调用做对照校验；开decode indexer时，每个graph decode步都在graph内对照（见上文“decode的服务内校验”）。确认稳定后常驻服务可设0；它不是插件开关。

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
`PyHIP QSA enabled: eager EXTEND, 3D only, dense2051;packed=pad1.7;raw=rho4`，且没有ABI mismatch或hook加载错误。
启用日志表示注册成功；实际替换还要看请求后的profile调用统计。仅服务ready不能证明QSA已接入，因为SGLang会记录插件加载异常后继续启动。
需要停止时在该前台终端按Ctrl-C；不要全局pkill其他SGLang服务。

## 4. 对上述常驻服务发请求、profile

另开终端，先执行第1节，再把`RUN`设为启动终端打印的真实路径；**不要再运行第2节，它会另起服务并占用同一端口**。

```bash
RUN="/opt/lc/pyhip/mytest/mydata/qsa_serve_..."  # 替换为启动终端打印的RUN
cd "$RUN"
"$PYTHON" /opt/evaluation7/check_acc_long_oai.py > "$RUN/warmup.log" 2>&1
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

T01、clean2及Vvec4/S4的功能/性能记录保留为撤销前历史，见[../opt.md](../opt.md)。当前仅3D，插件仍不接管SGLang graph/compile。单wave恢复/分片mask后QSA65项（含3D SGLang backend）通过；54组完整链378个实际dispatch三字段均0。真实链仍8kernel；36组普通kernel性能及逐kernel分解完成，旧A3停止记录不回填。此前MHA8192慢阶段未解决。

prefill indexer（`PYHIP_QSA_INDEXER=1`）实际TP2系统测试：同负载各128个无profiler请求，TTFT中位955.13→879.42/881.26ms（默认hipBLASLt，两次独立服务）和879.75/888.39ms（`PYHIP_QSA_INDEXER_GEMM=sglang`）；服务间波动2–9ms，两种GEMM在TTFT上不可分辨。每rank 72个校验调用全部通过，trace中每调用5502→525µs。见[../opt.md](../opt.md)与[系统分析](../../../../../mytest/mydata/qsa_indexer_system_20260928_01/final_analysis_185655.json)。

decode indexer第二阶段实际TP2三臂系统测试（顺序服务，`PYHIP_QSA_VALIDATE=0`，三臂都开prefill indexer），decode分别为`0`、`select`、`1`：

| 指标 | `0` | `select` | `1` |
|---|---|---|---|
| C1 ITL中位 | 13.404ms | 12.253ms | 10.852ms |
| C1请求时延中位 | 1730.0ms | 1659.7ms | 1570.4ms |
| C32 ITL中位 | 43.644ms | 23.443ms | 21.186ms |
| profile每层prep | 197µs（44个kernel） | 190µs（43个kernel） | 5.3–5.7µs（1个kernel） |

TTFT均约887ms。见[分析](../../../../../mytest/mydata/qsa_indexer_decode_system_20260929_03/analysis_054756.json)。本次插件包在freeze时一次构建，三臂共用。

第一阶段测试（当时`PYHIP_QSA_INDEXER_DECODE=1`，即现在的`select`；两臂都开prefill indexer）实际TP2系统测试，顺序服务，`PYHIP_QSA_VALIDATE=0`。C1（约12k prompt、输出64、128请求）的ITL中位从13.440ms降到12.209ms，请求时延从1730.6ms降到1660.7ms。C32因mamba状态池实际同时31路decode，ITL中位从43.261ms降到23.491ms。profile中每层decode MQA从106.7µs降到4.5µs。见[分析](../../../../../mytest/mydata/qsa_indexer_decode_system_20260929_02/analysis_041225.json)。首次运行开了`PYHIP_QSA_VALIDATE=1`，在C32第1轮因attention校验超容差而中止（[记录](../../../../../mytest/mydata/qsa_indexer_decode_system_20260929_01)）。

最新实际TP2原生/当前系统各128个无profiler请求（固定10条约12k输入、输出5、并发1），TTFT中位1037.031→955.764ms，请求时延1090.343→1009.214ms；顺序启动对照，不是同址交错或饱和吞吐。双rank各12层×4次profile替换核验通过，144项attention原`.02/.02`检查通过；配对生成文本114/128相同，差异集中于两条内部也不稳定的prompt，未做整模型质量认证。见[系统报告](../../../../../mytest/mydata/qsa_system_20260928_01/final_analysis.json)。两个测试服务已清理；未测真实TP4/8系统。

已重建[当前九源码独立包manifest](../../../../../mytest/mydata/qsa_prepare_latency_20260928_01/plugin_optimized/pyhip_qsa_runtime/source_manifest.json)，包含单wave恢复和分片mask准备实现，0.2.0、三个ABI/五hook注册均核验；包内L3/L47×TP2/4/8六例、48次实际dispatch零spill，未导入experiments，禁用entry-point仍惰性加载。动态LDS与固定字段分开核验；旧target是冻结源码，使用更新须构建新target。该验证不是整模型部署，下方早期包结果仍为历史。

本次模型服务另构建[实际使用的系统包](../../../../../mytest/mydata/qsa_system_20260928_01/current_v2/plugin/pyhip_qsa_runtime/source_manifest.json)，九源码与当前实现一致。系统trace证明每次4准备＋4attention launch，但未重新审计整个模型所有kernel的spill；不能将此前QSA三零外推到整模型。

2026-09-26核心union已更新，性能与限制见[QSA说明](../README.md#L1)。本页构建命令会复制当前核心；旧target仍是冻结副本，须新建target并重启才能使用更新。
该优化轮完成了源码backend回归和真实输入本地重放，**没有重新构建/部署独立插件或运行整模型profile**；下面0.2.0包的验证属于此前整理轮。

0.2.0独立包的原生5hook、惰性加载和两份真实输入执行已通过，见[整理记录](../../../../../mytest/mydata/qsa_consolidation_20260925_01/README.md)。
本页命令已使用新的默认Python环境，兼容ROCm依赖已就绪；仍须检查服务端设备、空闲状态及三个ABI哈希。上方2026-09-28系统结果使用新建包与本次模型trace，旧模型/早期包数据只作历史，不能互换标签。
profiler trace不是无profiler吞吐测试；bench duration包含trace导出等待，不能直接解释为常规请求耗时。

真实attention输入加载、FP32参考、性能、summary均已并入[../test_attention.py](../test_attention.py)，没有第二套attention回放或插件测试文件。
原长期上下文/模型指标测试未由这个快速profile流程替代。