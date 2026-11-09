# MiMo-V2.6-Flash端到端测试：FlyDSL FP8 block-scale MoE替换asm（gfx942，SGLang DP2 + attention TP4 + EP8）

2026-11-05，8×MI300X。先按[MiMo-V2.6-Flash评测指南](https://github.com/sammysun0711/sglang/blob/mimo-opt-mxfp4-dflash/evaluation/mimo_v26_flash/mimo-v2.6-flash-mopd-evaluation-guide.md)搭baseline，记录prefill性能和GSM8K精度；再把本仓库的FlyDSL GateUp + Down（见[moe_a8w8_blockscale_gfx942.md](moe_a8w8_blockscale_gfx942.md)）接到baseline的pyhip和aiter上，用同样的方法测，和baseline比较。

## 1. 结论

- prefill吞吐：8K–256K输入+6.6%到+10.3%，平均TTFT减少6.3–8.6%；512K +1.8%，983K持平（长上下文以attention为主）。
- 对照组（asm + 同样的MORI接收上限）和baseline相差−1.1%到+1.6%，所以提升来自MoE kernel，不是上限本身。
- GSM8K 97.50%（1286/1319），baseline 97.57%（1287/1319），差1题。DFlash接受长度5.62/5.53，baseline 5.64/5.54，都过了≥4的门槛。
- FlyDSL MoE必须同时设`SGLANG_MORI_MOE_MAX_INPUT_TOKENS=32768`，否则每次调用都回退到asm（原因见4.3）。

## 2. 环境

镜像`rocm/sgl-dev:v0.5.11-rocm720-mi30x-20260510`，按指南第2节安装：

| 组件 | 来源 | 版本 |
|---|---|---|
| sglang | sammysun0711/sglang `mimo-opt-mxfp4-dflash` | 6b84c84717，editable |
| sgl-kernel | 上面仓库的`sgl-kernel`，`setup_rocm.py install` | 0.4.3 |
| aiter | sammysun0711/aiter `integration/mimo-fp4-dflash` | 74cca902b，CK打`mimo_page64_qk192_v128_batch_prefill.patch`，editable |
| FlyDSL | pip | 0.3.2 |
| MORI | pip `amd_mori` | 1.2.4 |
| pyhip | aiter的`setup.py`依赖：tingqli/pyhip `xiaomi-355-opt` | ee5260d |

- 和指南唯一的不同：镜像自带的sglang、aiter、mori没有`rm -rf`，移到`/sgl-workspace/_img0510/`备份；原来的pip环境存了一份`pip freeze`。
- 模型`/models/MiMo-V2.6-Flash-MOPD`。checkpoint里的MXFP4 expert在加载时转成FP8 128×128 block-scale（launcher设了`SGLANG_MXFP4_TO_FP8_128BLOCK=1`），所以prefill的MoE走aiter的FP8 block-scale `fused_moe`，gfx942上是1-stage asm `fmoe_fp8_blockscale_g1u1`，正是FlyDSL要替换的路径。每卡32个本地expert：w13 `(32, 4096, 4096)`，w2 `(32, 4096, 2048)`。

## 3. 改动

### 3.1 pyhip

baseline环境已经有pyhip，所以把MoE kernel cherry-pick到它上面：分支`mimo/xiaomi-355-opt-moe-ep8`，基于`xiaomi-355-opt`的ee5260d。

| commit | 内容 |
|---|---|
| a7fd50e | 从`luwei/942-fp8-blockwise`的c8ec0ba拿来`moe_a8w8_blockscale_gfx942.py`、GateUp/Down两个LDS kernel、`gemm_a8w8_blockscale_gfx942_common.py`，以及`tests/contrib/moe`下4个测试和FlyDSL对asm的benchmark。`helpers.py`只新增`run_compiled`和`BufferTensor`，原有函数不动。FlyDSL 0.3.2没有`rocdl.s_buffer_load`，wave内一致的int32读取改成`raw_ptr_buffer_load` + `readfirstlane`（`load_uniform_i32`）。 |
| 1a2d755 | 这个aiter的`moe_sorting`按`tokens*topk + experts*block - topk`分配，不向上取整到整block，而GateUp/Down按整block检查边界。先用补齐的办法，已被下一条替代。 |
| fd7bc92 | 有效block总在buffer以内，改成截断block数（不复制）。benchmark的256行sort同样处理。 |

安装：`cd <pyhip_baseline> && pip install --no-deps .`，替换aiter装的那份。

### 3.2 aiter

`/sgl-workspace/aiter`上的分支`mimo-fp4-dflash-flydsl-moe`，基于74cca902b：

| commit | 内容 |
|---|---|
| 5dfb93c62 | cherry-pick ac2b48fe（`AITER_FMOE_FP8_BLOCKSCALE_FLYDSL=1`开关），没有冲突。用AST检查过钩子里引用的名字在这个版本的`_fused_moe_impl`里都有定义；`moe_sorting`的`output`参数和per-1x128量化的`num_rows`参数也都在。 |
| c7cf91cfd | 这个分支的`fused_moe`多了`a1_scale_is_transposed`参数；钩子只接受token-major的`[tokens, H/128]`，scale已经转置时回退到asm。 |

开关默认关闭，不设环境变量时和baseline行为相同。

### 3.3 sglang

不改代码，只加环境变量。

## 4. 测试方法

### 4.1 单元测试

```bash
cd <pyhip_baseline>/tests/contrib/moe
HIP_VISIBLE_DEVICES=0 PYTHONPATH=$PWD python -m pytest --import-mode=importlib -q \
    test_gateup_a8w8_blockscale_gfx942.py test_down_a8w8_blockscale_gfx942.py \
    test_fmoe_a8w8_blockscale_asm_gfx942.py test_aiter_fused_moe_flydsl_gfx942.py
```

- 测的是安装好的pyhip，不把`src`加进`PYTHONPATH`。`PYTHONPATH=$PWD`让测试之间能互相import。
- 要用`--import-mode=importlib`：默认模式会把仓库里的`tests/triton`目录当成`triton`包，torch导入时报`module 'triton' has no attribute 'language'`。
- 结果：167个用例，166个通过。失败的是aiter的asm参考kernel `novs_32x256`（在`test_fmoe_a8w8_blockscale_asm_gfx942.py`里），时好时坏，前一次运行167个全过。FlyDSL的用例全部通过。

### 4.2 kernel级性能

```bash
cd <pyhip_baseline>/tests/contrib/moe
HIP_VISIBLE_DEVICES=0 python bench_moe_a8w8_blockscale_gfx942_vs_asm.py --tokens 16384 --routing random --json vs_asm_16k.json
HIP_VISIBLE_DEVICES=0 python bench_moe_a8w8_blockscale_gfx942_vs_asm.py --tokens 32768 --routing random --json vs_asm_32k.json
```

一个EP8 rank（32个本地expert，top-8）。计时方法见[MoE文档5.4](moe_a8w8_blockscale_gfx942.md)：5份clone轮换（冷权重）、queued计时、amdsmi空闲检查。GPU必须空闲，所以在服务停掉以后测。

中位数，µs，含moe_sorting：

| tokens | asm（调优，vs_64x256） | asm（默认） | FlyDSL（single_store） | 加速比（对默认 / 对调优） |
|---:|---:|---:|---:|---|
| 16384 | 1669.3 | 1654.6 | 1064.0 | 1.56x / 1.57x |
| 32768 | 2871.1 | 3259.4 | 1968.6 | 1.66x / 1.46x |

calc_diff：asm 1.3e-5，FlyDSL 2.6e-6。

### 4.3 端到端prefill

服务用指南第3节"DP2 + attention TP4 + EP8"的命令，三种配置只差额外的环境变量：

| 配置 | 额外环境变量 |
|---|---|
| baseline | 无 |
| FlyDSL | `AITER_FMOE_FP8_BLOCKSCALE_FLYDSL=1 SGLANG_MORI_MOE_MAX_INPUT_TOKENS=32768` |
| 对照（asm + 上限） | `SGLANG_MORI_MOE_MAX_INPUT_TOKENS=32768` |

```bash
cd /sgl-workspace/sglang/evaluation/mimo_v26_flash
<额外环境变量> CHUNKED_PREFILL_SIZE=32768 MEM_FRACTION_STATIC=0.75 SWA_FULL_TOKENS_RATIO=0.03 \
MAX_RUNNING_REQUESTS=192 SGLANG_USE_PYHIP_MIMO_PREFILL=1 DISABLE_RADIX_CACHE=1 ENABLE_DFLASH=1 \
PERFORMANCE_MODE=1 LOG_DIR=<server日志目录> \
setsid nohup ./launch_mimo_v26_flash_dp2_tp4_ep8_dflash_accuracy_baseline.sh > launch.out 2>&1 < /dev/null &

cd /sgl-workspace/sglang/evaluation
BENCHMARK_PRESET=mimo_v26_flash HOST=127.0.0.1 LOG_DIR=<bench日志目录> ./run_benchmark_mimo_pro_prefill.sh
```

为什么要设`SGLANG_MORI_MOE_MAX_INPUT_TOKENS=32768`：

- launcher把每个rank的MORI dispatch容量设成每个DP的chunk 16384。MORI按`max_num_tokens_to_recv()`分配接收buffer，即8×16384 = 131072行，SGLang把整个buffer交给`fused_moe`，有效行数在`num_local_tokens`里。
- FlyDSL排序后的FP8中间结果按最坏情况分配：`(ceil(rows*8/256)+32)*256`行 × I。131072行时是2.02 GiB，超过buffer descriptor的2 GiB上限，钩子回退到asm。
- 这个配置下MoE输入在attention TP内是scatter的（`layers/communicator.py`里的`_compute_mlp_mode`）。每一步全局最多2个DP × 16384 = 32768个token，每个token对同一个rank最多发一次，所以一个rank最多收32768行。上限设32768不会丢token；`moe_runner/aiter.py`按这个值截断接收buffer，中间结果降到553 MB。

客户端用`mimo_v26_flash`预设：随机数据，输入长度固定，输出1个token，每个点先跑4个预热请求并清cache。`PERFORMANCE_MODE=1`让服务端用fake路由（`--ep-dispatch-algorithm fake`）。

| 输入长度 | 请求数 | 并发 |
|---:|---:|---:|
| 8192 / 16384 | 1024 | 32 |
| 32768 | 512 | 32 |
| 65536 | 256 | 32 |
| 131072 | 128 | 16 |
| 262144 | 64 | 8 |
| 524288 | 32 | 4 |
| 983040 | 16 | 4 |

确认FlyDSL确实在跑：

- 服务日志里没有`AITER_FMOE_FP8_BLOCKSCALE_FLYDSL=1 but ...; using the default path`。钩子每种回退原因记一次warning。
- 用SGLang的profiler抓一次prefill（`POST /start_profile`，`output_dir`指定目录，发一个请求后`POST /stop_profile`），数rank 0 trace里的kernel：47层每层各有一个`gateup_kernel_0`、`down_kernel_0`、`count_kernel_0`、`zero_kernel_0`，另有4个moe_sorting kernel，没有asm `fmoe`。

汇总：`python tests/contrib/moe/summarize_sglang_prefill_bench.py baseline=<dir> asm_cap=<dir> flydsl=<dir>`。

### 4.4 精度

服务换成`PERFORMANCE_MODE=0`（真实路由）重启，额外环境变量和对应的性能配置相同，然后跑指南第8节：

```bash
cd /sgl-workspace/sglang/evaluation
MODEL_PATH=/models/MiMo-V2.6-Flash-MOPD/ MIN_ACCEPT_LENGTH=4.0 OUTPUT_DIR=<结果目录> python3 ./run_evalscope_gsm8k_accuracy.py
```

GSM8K main的完整测试集（1319题），4-shot，EvalScope 1.11.0（脚本自动装在`/sgl-workspace/.venvs`下的独立venv里）。脚本同时检查每个worker的DFlash平均接受长度≥4。

### 4.5 注意事项

- 钩子按buffer行数（`topk_ids.shape[0]`）和`AITER_FMOE_FP8_BLOCKSCALE_FLYDSL_MIN_TOKENS`（默认4096）比较。截断后恒为32768，所以decode和DFlash verify的MoE也走FlyDSL。小M时FlyDSL比asm慢（MoE文档第6节，2048个有效token以下约0.8x），decode吞吐这次没有测。
- GSM8K这次服务开着decode CUDA graph，FlyDSL在graph里跑，精度正常。
- 每种配置都要用全部8张卡，开始前确认GPU空闲。第一次启动要JIT编译aiter模块，之后每次启动约2分钟。
- 服务和benchmark用`setsid nohup`起，避免终端断开时被杀。这次06:16容器重启过一次，FlyDSL的512K点被打断，512K和983K在新起的服务上重测。

## 5. 结果

prefill输入吞吐（tok/s，整机）：

| 输入长度 | 并发 | baseline | asm + 上限 | FlyDSL + 上限 | FlyDSL / baseline | FlyDSL / asm + 上限 |
|---:|---:|---:|---:|---:|---:|---:|
| 8192 | 32 | 82,545 | 83,877 | 91,069 | +10.3% | +8.6% |
| 16384 | 32 | 83,087 | 82,166 | 89,180 | +7.3% | +8.5% |
| 32768 | 32 | 77,582 | 77,219 | 83,337 | +7.4% | +7.9% |
| 65536 | 32 | 69,397 | 68,714 | 74,392 | +7.2% | +8.3% |
| 131072 | 16 | 57,947 | 57,491 | 61,778 | +6.6% | +7.5% |
| 262144 | 8 | 39,407 | 未测 | 42,623 | +8.2% | |
| 524288 | 4 | 26,658 | 未测 | 27,140 | +1.8% | |
| 983040 | 4 | 17,156 | 未测 | 17,091 | −0.4% | |

平均TTFT（ms）：

| 输入长度 | baseline | FlyDSL | 变化 |
|---:|---:|---:|---:|
| 8192 | 3,098 | 2,832 | −8.6% |
| 16384 | 6,222 | 5,796 | −6.8% |
| 32768 | 13,096 | 12,191 | −6.9% |
| 65536 | 28,360 | 26,473 | −6.7% |
| 131072 | 34,198 | 32,052 | −6.3% |
| 262144 | 50,610 | 46,834 | −7.5% |
| 524288 | 75,934 | 74,592 | −1.8% |
| 983040 | 213,757 | 214,258 | +0.2% |

精度：

| | GSM8K | 答对 | DFlash接受长度（DP0 / DP1） |
|---|---:|---:|---:|
| baseline | 97.57% | 1287/1319 | 5.64 / 5.54 |
| FlyDSL | 97.50% | 1286/1319 | 5.62 / 5.53 |

本机原始数据在`/mywork/mimo_eval/`：`baseline/`、`flydsl/`、`asm_cap/`下是服务日志、benchmark日志和GSM8K结果，`compare_prefill.md`是上面的对比表，`kernel_vs_asm_*`和`pyhip_unit_tests*.log`是4.1、4.2的输出。
