# QSA：整体正确性、性能与外部集成

QSA分为 **indexer（选token）** 与 **attention（计算输出）**。本目录验证整个算子；单kernel功能与性能测试位于[tests/ops/qsa](../../tests/ops/qsa)。实现已经迁入可安装的PyHIP包，参考[MoE的目录组织](../../src/pyhip/ops/moe)。

## 目录和依赖

| 内容 | 入口 |
|---|---|
| 完整attention：恢复/校验/分流/构表/dense/union/direct | [attention.py](../../src/pyhip/ops/qsa/flydsl/attention.py#L54) |
| prefill indexer、decode select、decode forward | [indexer.py](../../src/pyhip/ops/qsa/flydsl/indexer.py) |
| 共用MHA helper与BF16 D256线性内核 | [_common.py](../../src/pyhip/ops/mha/flydsl/_common.py)、[mha_pa_bf16_256_linear_942.py](../../src/pyhip/ops/mha/flydsl/mha_pa_bf16_256_linear_942.py) |
| 整体attention回归/真实回放/性能 | [test_attention.py](test_attention.py) |
| 整体indexer回归/真实回放/性能 | [test_indexer.py](test_indexer.py) |
| 单kernel attention功能/性能 | [test_attention.py](../../tests/ops/qsa/test_attention.py#L1)、[test_attention_kernels.py](../../tests/ops/qsa/test_attention_kernels.py#L1) |
| 单kernel indexer功能/性能 | [test_indexer.py](../../tests/ops/qsa/test_indexer.py#L1) |
| 共用数据/参考、门禁和JSON/CSV导出 | [_attention.py](../../tests/ops/qsa/_attention.py)、[_indexer.py](../../tests/ops/qsa/_indexer.py)、[_benchmark.py](../../tests/ops/qsa/_benchmark.py) |

目标为 **ROCm gfx942（MI308X）、BF16**。使用与ROCm匹配的PyTorch、FlyDSL、Triton，以及NumPy、msgspec和pytest；当前验证环境为Python3.10.12、Torch2.12/ROCm7.14、FlyDSL0.3.2、Triton3.8。基础PyHIP安装不会自动安装全部可选GPU依赖。

attention和prefill indexer的已安装计算实现不依赖experiments/tests/benchmarks。**decode indexer仍调用SGLang的fast_topk和expand**；本轮只搬迁，不重写这些依赖。完整indexer基准还使用SGLang QSAIndexer、AITER及原临时adapter，需匹配的SGLang/AITER环境。只有测试和基准从源码checkout引用临时参考。

```bash
# 在已配置ROCm/PyTorch/FlyDSL的环境安装；不要用CUDA版PyTorch替换ROCm版。
python -m pip install -e .
```

本机后续示例统一使用PyHIP的.venv解释器。新数据目录必须在mytest/mydata下且尚不存在；该目录允许是大容量卷的符号链接。GPU编号是未屏蔽的物理编号，先自行确认没有其他任务。

## 1. 整体正确性与性能入口

从仓库根目录运行；默认pytest不计时，`-m perf`才计时。显式选择benchmarks目录；它不在默认`testpaths=tests`中。

```bash
QSA_REPLAY_GPU=6 .venv/bin/python -m pytest -q \
  benchmarks/qsa/test_attention.py benchmarks/qsa/test_indexer.py \
  tests/ops/qsa/test_attention.py tests/ops/qsa/test_attention_kernels.py \
  tests/ops/qsa/test_indexer.py

# 无真实capture也可验证完整算子。每条命令使用不同新目录。
.venv/bin/python benchmarks/qsa/test_attention.py --gpu 6 \
  --synthetic 64 2051 12000 --tp-sizes 2 4 8 --check-only \
  --output mytest/mydata/qsa_attention_check_run1
.venv/bin/python benchmarks/qsa/test_indexer.py --gpu 6 --synthetic --check-only \
  --output mytest/mydata/qsa_indexer_check_run1
.venv/bin/python benchmarks/qsa/test_indexer.py --gpu 6 --decode-forward \
  --rows 1 --lengths 12000 --check-only --output mytest/mydata/qsa_decode_check_run1

# 完整attention性能：有capture时用--inputs指定，否则可用--synthetic 12000。
.venv/bin/python benchmarks/qsa/test_attention.py --gpu 6 --synthetic 12000 \
  --tp-sizes 2 4 8 --buffers 10 --samples 128 \
  --output mytest/mydata/qsa_attention_perf_run1
.venv/bin/python benchmarks/qsa/test_indexer.py --gpu 6 --synthetic \
  --buffers 10 --samples 128 --output mytest/mydata/qsa_indexer_perf_run1
.venv/bin/python benchmarks/qsa/test_indexer.py --gpu 6 --decode \
  --rows 1 8 32 --keys 3000 --output mytest/mydata/qsa_decode_select_perf_run1
.venv/bin/python benchmarks/qsa/test_indexer.py --gpu 6 --decode-forward \
  --rows 1 8 32 --lengths 12000 --output mytest/mydata/qsa_decode_forward_perf_run1
```

数据与校验：

- attention默认读取原8个TP2 capture，支持`QSA_REAL_INPUT_DIR`或`--inputs`覆盖；indexer支持`QSA_INDEXER_INPUT_DIR`或`--inputs`。缺少显式输入或没有任何用例会报错，不输出“0例成功”。历史capture来源与hash保持不变；T13真实失败张量缺失时该专属回归明确skip。
- attention验证完整输出对原参考、选行对独立FP32、输出guard、只读输入、重复逐bit、动态selection/graph、分流和工作区；T13及构造反例保留独立FP32/FP64判据，不把旧BF16预缩放参考当精确数学结果。原`.02/.02`不放宽。
- indexer验证完整token ABI、因果/无重复四token块及尾token、q/ring/compressed写入、FP64近并列top-k判据、SGLang adapter与decode图内校验。SGLang-GEMM和hipBLASLt-GEMM两条路径分别检查，舍入差异不伪称逐bit。
- `--check-only`在prefill、decode select、decode forward都不构造计时器、不执行性能门禁；decode图重放正确性另由pytest覆盖。

输出：每例保留结果JSON（完整raw、shape、设备、源码hash、校验、失败原因），以及硬件快照；整组生成 **summary.json、summary.csv、raw.csv**。CSV按case/scope区分，attention带dense/union/direct行数，原始样本全保留。读取`complete`，不能只看有CSV就认为通过。

计时边界与协议：

- 默认10个独立输入/输出buffer、2warmup、128samples，原`cudaPerf`，AB/BA或组件scope轮换；不筛首尾和长尾。
- attention完整调用含恢复、校验、分流、构表、每次必要KV pack及全部attention；不含JIT、输出分配、indexer或SGLang的KV gather。
- prefill indexer是hidden states→token selections，**包含index_qk_proj GEMM、adapter、norm/RoPE、pool写入、压缩、logits、top-k和分配**。`pyhip_exact`用SGLang GEMM，`pyhip`用现有hipBLASLt策略。
- decode select/forward测CUDA graph replay；forward包含GEMM与完整prep/selection，metadata更新和capture不计入。
- 单次检查use≤5%、VRAM≤20%、PTL Enabled/VECTOR,F8；门禁失败立即停止并保留raw，不等待、不重采到通过、不改硬件。
- TP2/4/8为attention本地H12/H6/H3形状；TP4/8由TP2激活裁剪时明确标记derived，**不是新多卡TP4/8服务**。indexer4头和权重在各rank复制，TP2/4/8每rank工作相同，不伪造三份独立分布式测量。

## 2. 单kernel功能与性能

准备、构表、guard与参考在计时区间之外，检查的是刚完成的实际输出。**组件中位数不能相加当作完整调用时延。**

```bash
.venv/bin/python tests/ops/qsa/test_attention_kernels.py --gpu 6 --check-only \
  --tp-sizes 2 4 8 --output mytest/mydata/qsa_attention_kernels_check1
.venv/bin/python tests/ops/qsa/test_attention_kernels.py --gpu 6 --perf \
  --tp-sizes 2 4 8 --buffers 10 --samples 128 \
  --output mytest/mydata/qsa_attention_kernels_perf1
.venv/bin/python tests/ops/qsa/test_indexer.py --gpu 6 --perf \
  --buffers 10 --samples 128 --output mytest/mydata/qsa_indexer_kernels_perf1

QSA_REPLAY_GPU=6 QSA_REPLAY_OUTPUT=mytest/mydata/qsa_kernels_pytest1 \
  .venv/bin/python -m pytest -q -m perf \
  tests/ops/qsa/test_attention_kernels.py tests/ops/qsa/test_indexer.py
```

| attention scope | 边界 |
|---|---|
| `recover`、`compact`、`order_masks_validate`、`scatter_prepared` | 各一次对应Triton kernel，含原校验或mask逻辑，不含测试reset |
| `dense`、`union`、`raw_direct` | 各一次完整设备kernel；union强制开启，raw用合法非packed计划 |
| `pack`、`packed_direct` | 将原双kernel launcher分为两次测试调用，分别计时，不改设备函数 |

indexer scopes：`q_prep`、`k_compress`、`gather_prefix`、`indexer_logits`、`indexer_topk`、`decode_prep`、`decode_logits`。支持`--scopes`筛选，生产计划没有的prefix/logits调用不会伪造。功能覆盖小尾部、511/512/513压缩key边界、ragged、padding、近并列/重复分数和page乱序。实际编译资源要求private、VGPR spill、SGPR spill都为0；Triton动态LDS单独记录。

## 3. 整体性能数据（既有已完成测量）

以下是迁移前已完成、保留原身份的基线，不标成迁移后新实测，也不把不同场次的indexer与attention中位数相加。完整历史见[优化记录](../../experiments/attention/flydsl/qsa/opt.md)。

**本轮新性能验证未通过入口门禁，0 raw。** [执行收据](../../mytest/mydata/qsa_package_refactor_20260929_01/measurements/execution.json)：首个L3/TP2的入口快照为GPU6 use0%、VRAM47%（门限20%），计时器尚未执行；其余性能任务未启动。原失败JSON/CSV保留，不改门限、没有等待重试，也不根据稍后GPU空闲反推当时通过。新组件计时与整体indexer计时循环尚未在本轮GPU采样完成；下面数字仅为历史基线。

**后续GPU0–3快速验证也未采到新样本。** 用户指定四卡、10buffers/2warmup/20samples；初始四卡均use0%/VRAM0%、PTL Enabled/VECTOR,F8，实际采样前分别为GPU0 69%/47%、GPU1 64%/47%、GPU2 18%/2%、GPU3 9%/1%（use/VRAM），全触发门禁、0raw，原失败不重试。GPU0后续独立进程在自身分配0B时已见171.0GB设备占用（82.96%），自身检查最高reserved仅0.998GB；不能把该设备占用解释为本基准临时缓存。GPU2/3的利用率可能含自身预热窗口，未证实且不豁免。见[四卡审计](../../mytest/mydata/qsa_quick_perf_20260929_01/analysis.json)和[显存来源诊断](../../mytest/mydata/qsa_quick_perf_20260929_01/setup_memory.json)。

**去掉packed文件名前导下划线后的授权重测：** [本轮审计](../../mytest/mydata/qsa_quick_perf_20260929_02/analysis.json)确认47个相关Python文件仅发生模块名替换，packed计算正文逐字不变，4项packed数值/scratch/graph回归通过。GPU0–3初始门禁全过，但各卡在自身检查/预热后的`before_samples`分别为20%/2%、23%/2%、19%/2%、9%/1%（use/VRAM），再次以原5%利用率门槛停止，0raw。低显存不能证明无外部任务；采样窗也可能包含自身预热，不能据此推断内核性能回退，未跳过门禁或反复采样。

### 完整attention

gfx942/MI308X、10buffers/128samples。单位µs，包含prepare＋自动分流＋attention。来源：[完整54形状矩阵](../../mytest/mydata/qsa_tp_kernels_20260929_02/branches_table.txt)、[行数/launch](../../mytest/mydata/qsa_tp_kernels_20260929_02/routes_table.txt)。

| TP0真实输入 | TP2/H12 | TP4/H6派生 | TP8/H3派生 |
|---|---:|---:|---:|
| L3 M12000 | 2725.875 | 1879.170 | 1434.848 |
| L47 M12000 | 2818.115 | 1817.030 | 1409.808 |
| L3 M11888 | 2674.194 | 1672.749 | 1320.807 |
| L47 M11888 | 2707.274 | 1811.109 | 1419.128 |

L3 M12000的dense/union/direct行数：TP2=2051/3929/6020，TP4/8=2051/9949/0；均8个launch（空gated direct/pack仍launch）。路由并非所有输入最优，例如share75/TP8自动1218.765µs、full_direct601.383µs；本轮不改路由掩盖该限制。

### 完整indexer

Prefill是真实输入完整forward、包含GEMM，单位µs。来源：[L3](../../mytest/mydata/qsa_indexer_flydsl_20260929_01/formal_v1/indexer_tp0_layer3_m12000/result.json)、[L47](../../mytest/mydata/qsa_indexer_flydsl_20260929_01/formal_v1/indexer_tp0_layer47_m12000/result.json)。indexer在TP2/4/8每rank相同。

| TP0输入 | SGLang | PyHIP FlyDSL（默认投影） |
|---|---:|---:|
| L3 M12000 | 6482.516 | 495.803 |
| L47 M12000 | 6533.055 | 496.503 |

Decode完整forward图重放、GPU3、10buffers/128samples，包含GEMM与prep，单位µs。来源：[最终9形状报告](../../mytest/mydata/qsa_indexer_decode_20260929_01/formal_forward_summary.json)。

| 请求行×序列token数 | SGLang | 仅替换select | 完整PyHIP forward |
|---|---:|---:|---:|
| 1×12000 | 221.70 | 128.82 | 31.24 |
| 8×12000 | 672.7 | 189.5 | 35.7 |
| 32×12000 | 2022.7 | 212.9 | 43.3 |

实际TP2历史服务、`VALIDATE=0`：完整decode优化C1 ITL13.404→10.852ms，C32 ITL43.644→21.186ms，TTFT基本不变；[服务对照](../../mytest/mydata/qsa_indexer_decode_system_20260929_03/analysis_054756.json)。这是不同臂顺序服务，不是微kernel时间；不代表模型质量、饱和吞吐或TP4/8服务验收。

## 4. 外部集成示例

### 已有合法选中token时调用attention

外部只导入已安装包；不需要把experiments或tests加进PYTHONPATH。下面短上下文示例可直接执行。长上下文的indices必须来自合法indexer/调用方选择，**不能用随机整数填满indices**。

```python
import torch
from pyhip.ops.qsa.flydsl.attention import attention

torch.cuda.set_device(0)  # 每进程一个GPU；TP使用独立worker进程
device = torch.device("cuda", 0)
m, h, hk, d = 64, 12, 1, 256
q = torch.randn(m, h, d, device=device, dtype=torch.bfloat16)
k = torch.randn(m, hk, d, device=device, dtype=torch.bfloat16)
v = torch.randn_like(k)
tokens = torch.arange(2051, device=device)[None, :]
positions = torch.arange(m, device=device)[:, None]
indices = torch.where(tokens <= positions, tokens, -1).to(torch.int32).contiguous()
out = torch.empty_like(q)
attention(q, k, v, indices, query_lens=(m,), prefix_lens=(0,), out=out)
```

约束：Q/O为连续BF16 `[M,H,256]`，K/V为连续BF16 `[N,HK,256]`，`H/HK≤16`，16B对齐、inference-only；QSA不接受5D KV。`indices`是连续int32 `[M,2051]`，每行最多512个完整唯一四token块（块可乱序），接该query的0–3因果尾token，再填−1；ID为请求内逻辑token。`query_lens`/`prefix_lens`是host长度，packed请求的K/V按请求拼接；默认单请求prefix=`N-M`。默认scale=1/16，`out`不与输入重叠。

### 接入prefill indexer，再把选择交给attention

下面是调用方已有projection和缓存metadata时的适配函数，变量是参数而非隐藏全局。**indexer Q/K的D128与attention的D256是不同投影**；indexer的compressed pool不是attention的V缓存。

```python
from pyhip.ops.qsa.flydsl.indexer import prefill_indexer
from pyhip.ops.qsa.flydsl.attention import attention

def qsa_prefill(q, k, v, projected_index_qk, indexer_metadata,
                query_lens, prefix_lens, out=None):
    indices = prefill_indexer(projected_index_qk, **indexer_metadata)
    return attention(q, k, v, indices,
                     query_lens=query_lens, prefix_lens=prefix_lens, out=out)
```

`projected_index_qk`是调用方GEMM输出的连续BF16 `[M,640]`（4×128 query＋1×128 key）。`indexer_metadata`必须包含[原始接口](../../src/pyhip/ops/qsa/flydsl/indexer.py#L285)的以下键，所有tensor在同一GPU：

| 参数 | 契约 |
|---|---|
| `heads`、`seq_lens`、`extend_lens` | 4；host最终序列长度和本次query长度；prefix按4token组对齐 |
| `positions`、`logical_positions` | int64 `[M]`或三轴`[3,M]`；int64 `[M]`，请求内prefix+i |
| `state_slots`、`key_state`、`rope_state` | int64 `[M]` ring槽；BF16 `[slots,1,128]`；int64 `[slots,3]`，原地更新 |
| `write_locs`、`member_rows`、`group_sequences`、`group_ends` | int32组写槽、int64本次QK首成员行、请求编号、请求内组末位置；保留原slot0 padding规则 |
| `rope_matrix`、`compressed` | int64 `[M,3]`；BF16 `[compressed_slots,1,128]`，原地更新 |
| `token_slot_table` | int32 `[requests,max_seq]`，每token物理槽，组首物理槽/4定位compressed key |
| `cos_sin_cache`、`axis_map` | 连续BF16/FP32 `[positions,64]`；int32 `[32]`选择三轴 |
| `q_weight`、`k_weight`、`q_eps`、`k_eps` | BF16 `[128]` Gemma RMSNorm增量权重；host epsilon |

单请求最多65536个compressed key/262144token；返回int32 `[M,2051]`。slot0与ring保留行是惰性写入区域，不能作为有效数据读取。不要把mean、RoPE或slot规划隐去当成免费预处理。

Decode使用`decode_indexer(q, cache, page_table, lengths, query_positions, sequence_lengths)`，或`decode_forward(projected_index_qk, **decode_metadata)`；实际签名和graph要求见[indexer.py](../../src/pyhip/ops/qsa/flydsl/indexer.py#L374)。保留SGLang的top-k/expand依赖；完整decode每行必须属于不同请求，page table是16-key compressed page，长度是compressed key数。prefill入口不是通用decode替代。

### attention CUDA graph

复用同一stream/layout先eager预热，再capture；不要并发重放共享同一工作区的graph。图内只调用已经预热的attention；维度、长度和scale不能在capture时首次出现。

```python
stream = torch.cuda.Stream(device=q.device)
stream.wait_stream(torch.cuda.current_stream(q.device))
with torch.cuda.stream(stream):
    attention(q, k, v, indices, out=out)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        attention(q, k, v, indices, out=out)
torch.cuda.current_stream(q.device).wait_stream(stream)
graph.replay()  # 重放前可原地更新同形状Q/K/V和合法indices
```

## 5. 临时SGLang接入与迁移范围

[attention_direct_packed.py](../../src/pyhip/ops/qsa/flydsl/attention_direct_packed.py)已按用户要求去掉文件名前导下划线，调用与测试引用同步更新。它仍是[attention_direct.run](../../src/pyhip/ops/qsa/flydsl/attention_direct.py#L620)的packed分支；计算正文、GPU符号、布局及工作区不变。对外完整算子入口仍为`attention()`。为不修改留待下一轮删除的SGLang临时目录，实验目录旧名仅保留6行转发，指向新的正式模块；正式src目录不保留旧名模块。

[experiments/attention/flydsl/qsa/sglang](../../experiments/attention/flydsl/qsa/sglang)本轮**文件内容不变**，下一轮再清理。旧QSA/MHA运行时路径只留薄模块转发，canonical实现和编译缓存位于src，没有复制两套算法。

重要过渡限制：原`build_target`仍按旧清单复制文件，因此**本轮以后新生成的临时插件target包含转发模块，依赖同版本已安装的PyHIP**；不再承诺target单独包含全部kernel。既有冻结独立包完全不改。部署时必须同时安装本分支PyHIP并重新构建target；原子发布PyHIP和target，避免版本错配。直接使用上方PyHIP API不需要临时插件。

迁移证据：[运行时逐字证明](../../mytest/mydata/qsa_package_refactor_20260929_01/runtime_proof.json)、[原测试定义保留](../../mytest/mydata/qsa_package_refactor_20260929_01/test_relocation_proof.json)、[新旧设备等价](../../mytest/mydata/qsa_package_refactor_20260929_01/equivalence/result.json)、[隔离wheel打包](../../mytest/mydata/qsa_package_refactor_20260929_01/package/manifest.json)、[外部API运行](../../mytest/mydata/qsa_package_refactor_20260929_01/package/gpu.json)。本轮不改SGLang源码、attention数学/路由、indexer精度或历史数据；不做模型质量或真实多卡服务声明。

验证结果：原152项回归＋52项新kernel/host检查 **204 passed、42 perf deselected、0失败/跳过**；共用MHA另29项通过。21个新旧同输入用例一致，77对对象/83对完整函数机器码、ABI及资源相同，三类private/spill全0。六个最终check-only CLI共37例通过且生成JSON/CSV，无计时raw/门禁；见[回归](../../mytest/mydata/qsa_package_refactor_20260929_01/regression/execution.json)、[MHA](../../mytest/mydata/qsa_package_refactor_20260929_01/mha_regression/execution.json)、[CLI结果](../../mytest/mydata/qsa_package_refactor_20260929_01/cli_checks/result.json)。随后报告逻辑补充了异常/未运行用例汇总，另有CPU专测；没有重跑GPU到获得性能样本。