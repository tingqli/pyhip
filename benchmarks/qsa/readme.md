# QSA：使用说明与最终性能对比

QSA分为indexer（选token）与attention（计算输出）。本页只维护当前使用说明和最终有效性能；优化过程、中间版本、失败诊断统一追加到[源码readme](../../src/pyhip/ops/qsa/flydsl/readme.md)。已安装实现不依赖实验目录或临时插件。

## 目录和依赖

| 内容 | 入口 |
|---|---|
| 完整attention：恢复/校验/分流/构表/union/direct | [attention.py](../../src/pyhip/ops/qsa/flydsl/attention.py#L179) |
| prefill indexer、decode select、decode forward | [indexer.py](../../src/pyhip/ops/qsa/flydsl/indexer.py) |
| 共用MHA helper与BF16 D256线性内核 | [_common.py](../../src/pyhip/ops/mha/flydsl/_common.py)、[mha_pa_bf16_256_linear_942.py](../../src/pyhip/ops/mha/flydsl/mha_pa_bf16_256_linear_942.py) |
| 合成attention三分支/完整调用 | [test_attention.py](test_attention.py) |
| 合成indexer完整调用 | [test_indexer.py](test_indexer.py) |
| Attention/indexer基本功能 | [test_attention.py](../../tests/ops/qsa/test_attention.py)、[test_indexer.py](../../tests/ops/qsa/test_indexer.py) |
| 当前逐kernel性能 | 随上述两个benchmark默认输出，无独立对照脚本 |
| 共用数据/参考、记录和JSON/CSV导出 | [_attention.py](../../tests/ops/qsa/_attention.py)、[_indexer.py](../../tests/ops/qsa/_indexer.py)、[_benchmark.py](../../tests/ops/qsa/_benchmark.py) |

目标为 **ROCm gfx942（MI308X）、BF16**。使用与ROCm匹配的PyTorch、FlyDSL、Triton，以及NumPy、msgspec和pytest；当前验证环境为Python3.10.12、Torch2.12/ROCm7.14、FlyDSL0.3.2、Triton3.8。基础PyHIP安装不会自动安装全部可选GPU依赖。

Attention、prefill indexer和decode indexer的计算实现均不依赖SGLang、AITER或experiments/tests/benchmarks。Decode的top-k和token展开使用PyHIP自身的FlyDSL实现，与prefill共享选择核心。测试使用独立tensor夹具和精度参考，不创建模型或临时adapter；只有第5节的服务集成需要SGLang。

```bash
# 在已配置ROCm/PyTorch/FlyDSL的环境安装；不要用CUDA版PyTorch替换ROCm版。
python -m pip install -e .
```

本机后续示例统一使用PyHIP的.venv解释器。新数据目录必须在mytest/mydata下且尚不存在；该目录允许是大容量卷的符号链接。`--gpu`使用当前进程可见的设备编号，`--gpu 0`是第一张可见卡；无需清除GPU/CU屏蔽环境变量。性能测试前自行确认设备空闲。

FlyDSL的持久编译缓存在源码只改了嵌套helper时可能返回旧kernel。升级或修改PyHIP后，先清空该缓存，或为测试设置新的空目录：`export FLYDSL_RUNTIME_CACHE_DIR=$(mktemp -d)`。

所有kernel都在各入口的第一次调用时编译，之后任何批次组合都不再编译：每种head形状（H, HK, scale）第一次调用attention时，编译该形状可能用到的全部变体（不建计划、每种union tile行数）；prefill indexer和decode的kernel在各自第一次调用时编译。Triton和FlyDSL缓存都为空时，第一次attention在H12/H6/H3（TP2/4/8）约需36/39/68秒，第一次prefill indexer约2秒，第一次decode约3秒。服务应在启动或预热阶段完成这些调用；CUDA graph capture中不编译，capture前仍须eager预热。

## 1. 整体正确性与合成性能

```bash
QSA_REPLAY_GPU=2 .venv/bin/python -m pytest -q \
  tests/ops/qsa/test_attention.py tests/ops/qsa/test_indexer.py

.venv/bin/python benchmarks/qsa/test_attention.py --gpu 2 \
  --rows 64 12000 --tp-sizes 2 4 8 --check-only \
  --output mytest/mydata/qsa_attention_check_new
.venv/bin/python benchmarks/qsa/test_attention.py --gpu 2 \
  --output mytest/mydata/qsa_attention_perf_new
.venv/bin/python benchmarks/qsa/test_indexer.py --gpu 2 \
  --output mytest/mydata/qsa_indexer_perf_new
.venv/bin/python benchmarks/qsa/test_indexer.py --gpu 2 --mode prefill \
  --lengths 12000 --output mytest/mydata/qsa_prefill_perf_new
.venv/bin/python benchmarks/qsa/test_indexer.py --gpu 2 --mode decode \
  --rows 1 32 --keys 3000 --perf --output mytest/mydata/qsa_decode_perf_new
.venv/bin/python benchmarks/qsa/test_indexer.py --gpu 2 --mode decode-forward \
  --rows 1 32 --lengths 12000 --perf --output mytest/mydata/qsa_decode_forward_perf_new
```

不依赖真实capture或数据集。CLI默认执行整体性能和当前逐kernel计时；`--check-only`仅验证，`--perf`作为兼容参数保留。省略`--output`时自动创建mytest/mydata下的新目录。Attention默认M12000、TP2/4/8；indexer默认prefill12000、decode的1/32行x3000key和decode-forward的1/32行x12000token。Attention的qsa/forced_direct/forced_union比较相同选择；prefix_*三项只比较共同的2051行完整因果前缀，不能与长稀疏选择当作同语义。

Attention完整计时含恢复/校验/构表/分流、必要KV pack及计算，不含公共入口检查、workspace查找、输出分配、JIT、indexer或服务KV gather。Indexer三模式从投影后输入开始，不含projection GEMM；prefill为eager调用，decode/decode-forward为单次graph replay，后者包含完整prep和selection。

原cudaPerf、10独立buffer、2warmup、128samples，保留首尾和慢样本。Attention基本精度`.02/.02`不变，indexer按完整token ABI、FP64 top-k边界`1e-5`及逐bit缓存状态校验；decode返回顺序不要求固定。输出为summary.json/CSV、raw.csv、逐例JSON、源码和地址证据。`complete=false`的数据不能作为有效性能。

Kernel/算子benchmark不检查GPU利用率、显存占用或PTL，不限制GPU/CU屏蔽环境变量，也不生成硬件快照；入口、单case、单kernel、采样循环和结束时均无GPU状态门禁。正确性校验与源码一致性检查保留，失败即停止并保留已完成数据。`complete`表示执行和校验是否完成，不证明测量期间设备空闲。TP2/4/8 kernel形状对应单卡H12/H6/H3，不是分布式服务；indexer在rank间复制，不能伪造三份不同TP测量。

## 2. 逐kernel性能

直接运行正式benchmark即可得到当前kernel耗时，不加载旧commit、不做旧新对照：

```bash
.venv/bin/python benchmarks/qsa/test_attention.py --gpu 2

.venv/bin/python benchmarks/qsa/test_indexer.py --gpu 2
```

原cudaPerf、10buffers/2warmup/128samples，kernel顺序轮转。捕获实际生产HIP节点；reset和前置依赖在计时外执行，计时后运行后置节点并校验实际输出。整体调用另测，不能将独立kernel中位数相加。自动路由下的空gated kernel也记录其实际启动开销。

控制台及输出目录的`kernels.txt`、`kernels.csv`列出`Case / Kernel / Median_us / Status`；逐case的`kernels/result.json`包含raw、原始符号和grid/block/shared。整轮有效性以顶层`summary.json`和`matrix_status.json`为准。

输出格式示例（尖括号是字段占位符，不是实测值）：

```text
Case               Kernel                              Median_us   Status
m12000_tp2         0:attention_recover_scatter            <实测值>    valid/invalid
m12000_tp2         3:attention_union                      <实测值>    valid/invalid
prefill_n12000     0:indexer_q_prep                       <实测值>    valid/invalid
decode_r1_k3000    0:qsa_indexer_decode_logits            <实测值>    valid/invalid
```

### 当前实测

2026-10-02，MI308X/gfx942，原cudaPerf、10buffers、2warmup、128samples，每次运行使用新建的FlyDSL缓存目录，均在GPU2（PCI 0000:a4:00.0）上实测。Attention和indexer默认入口是union合并KV尾部变体、indexer去掉整数特化和首次调用预编译之后重测的；长行decode select用到的kernel此后没有改动，沿用同日较早的测量。每次开始前rocm-smi显示8张卡use均为0%；本轮测量期间，本会话在GPU4–6上同时运行编译清单，没有使用GPU2。按当前协议未做硬件门禁。各组均通过正确性与源码哈希检查。单位为微秒，取全部样本中位数；单节点图重放包含该launch的固定开销，不能把这些中位数相加作为整体调用时间。

Attention：M=N=12000、D256、KV1，默认合成选择及自动分流。TP2/4/8是单卡H12/H6/H3形状。完整因果前缀行也和其它行一样在union/direct间选择；总行数少于`max(384, 64*H/HK)`（H12为768，H6/H3为384）时不建union计划，全部走direct。各请求KV按4-token对齐后PK+PV不超过64MiB时direct走packed（含多请求和KV长度非4倍数），超出时走raw。逐tile公式之后，attention_order_masks再做一次全局路由检查。Union tile为H12 10行、H6 16行、H3 32行；大单请求无prefix且行数足够时H6取21行、H3取42行，本例TP4/TP8即为21/42。本例union/direct行数为TP2 6330/5670、TP4 10053/1947、TP8 12000/0。合成选择的块号乱序，attention_recover_scatter逐行排序；PyHIP prefill indexer的升序输出同样排序，只有块号恰为0,1,2,…的完整因果前缀行跳过排序。

| Kernel | TP2/H12 | TP4/H6 | TP8/H3 |
|---|---:|---:|---:|
| attention_recover_scatter（含逐行校验） | 99.921 | 99.060 | 101.281 |
| attention_compact | 27.080 | 19.600 | 17.480 |
| attention_order_masks（含全局路由） | 38.320 | 31.600 | 42.000 |
| attention_union | 944.845 | 1218.666 | 952.165 |
| attention_pack_kv | 17.600 | 15.681 | 9.600 |
| attention_direct（packed） | 1428.267 | 542.883 | 14.720 |

TP8全部12000行走union，pack/direct只测得gated空分支启动开销，不是实际direct耗时。整体auto QSA另测为2518.232 / 1891.609 / 1101.565微秒；2051行完整因果前缀（prefix_qsa，KV长度不是4的倍数）为205.841 / 139.241 / 136.620微秒。

Prefill indexer：M12000、4个D128 Q头、ratio4、top512；投影GEMM不在计时范围。logits按token_slot_table直接从压缩key池读取所有key（含prefix），不另拷key。

| Kernel | 中位耗时 |
|---|---:|
| indexer_q_prep | 39.760 |
| indexer_k_compress | 13.920 |
| qsa_indexer_logits | 136.381 |
| qsa_indexer_topk（含位置校验） | 123.320 |

整体prefill_indexer另测为291.061微秒，包含公开入口内的分配与准备。

Decode indexer：每请求3000个压缩key（12000 token）；select只选块，forward包含投影后的prep和选块，两者均不含GEMM。Decode top-k每行一个512线程CTA，与prefill共用同一选择实现。

| Kernel | select B1 | select B32 | forward B1 | forward B32 |
|---|---:|---:|---:|---:|
| indexer_decode_prep | 不调用 | 不调用 | 9.960 | 10.920 |
| qsa_indexer_decode_logits | 9.880 | 16.320 | 9.840 | 16.680 |
| qsa_indexer_decode_topk（含展开） | 14.280 | 15.160 | 14.640 | 15.320 |

整体select图重放B1/B32为18.720 / 25.441微秒，forward为22.240 / 31.400微秒。

长行decode select（`--mode decode --rows 1 8 32 --keys 16384 32768 65536`）：

| 压缩key | logits B1/B8/B32 | top-k（含展开）B1/B8/B32 | 整体B1/B8/B32 |
|---:|---:|---:|---:|
| 16384 | 9.960 / 20.540 / 53.400 | 19.760 / 20.240 / 20.540 | 24.120 / 34.800 / 66.361 |
| 32768 | 12.200 / 31.280 / 85.161 | 26.281 / 27.080 / 27.360 | 32.600 / 52.040 / 117.521 |
| 65536 | 13.960 / 47.260 / 183.501 | 39.340 / 40.221 / 42.200 | 47.640 / 88.020 / 217.421 |

B8的32768/65536 key和全部B32长行读取的压缩key最多，10个独立buffer稳定地分成快慢两组：同一buffer始终在同一组，差别只出现在logits（各buffer的top-k一致），应来自buffer所在的显存位置。两组的logits相差13–19%，整体相差8–16%（如B32、65536 key的logits约154/184微秒，整体约191/221微秒）。全样本中位数取决于两组样本的比例，换卡或重新分配后可能在两组之间移动。

以上为当前版本绝对性能，不包含旧新对照；attention共3个case、18行kernel结果、2304条整体raw，indexer默认5个case、14行kernel结果、640条整体raw，长行decode 9个case、18行kernel结果、1152条整体raw，均未筛除慢样本。记录索引保留在[源码readme](../../src/pyhip/ops/qsa/flydsl/readme.md)，复现只需本页已跟踪入口。

公开入口的CPU下发时间（GPU保持忙，60次中位数，TP2/4/8相同）：attention建union计划时约87µs，不建计划时约54µs；prefill_indexer在M12000时约195µs，3请求时约247µs。图重放不含这部分；eager prefill中GPU时间短于它的调用（如2051行完整前缀、64行长prefix）受host限制。

## 3. 最终服务性能对比

2026-10-02，PyHIP为本页描述的当前实现（HEAD `d780b7d`之上的工作区改动），SGLang HEAD `c3c4bc8`，gfx942/MI308X，模型Qwen3.8-Flash-Next-PTPC-FP8。PyHIP使用独立FlyDSL decode top-k/展开。原生与PyHIP在相同TP及配置下测量C1/C2/C4/C8，共16场、512请求；每场32请求、名义12000输入/350输出、服务seed42，计时关闭数值校验和profiler。源码哈希、请求量、配置一致性及整轮入口/出口检查均通过。

服务配置为chunked prefill 16384、最大运行请求32、decode graph最大batch 32、关闭radix cache、AITER attention/MoE backend。两组的实际`mem_fraction_static`一致：TP2为0.8075、TP4为0.7225；服务和客户端沿用4个CPU线程。

| TP | 并发 | 原生token/s | PyHIP token/s | 吞吐变化 | TTFT中位数(ms，原生/PyHIP) | ITL中位数(ms，原生/PyHIP) | TTFT p99(ms，原生/PyHIP) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2 | 1 | 61.141 | 74.839 | +22.40% | 1039.570 / 883.639 | 13.415 / 10.868 | 1054.290 / 893.162 |
| 2 | 2 | 95.198 | 119.095 | +25.10% | 2034.246 / 1409.038 | 15.196 / 11.866 | 2054.692 / 1739.677 |
| 2 | 4 | 128.284 | 162.252 | +26.48% | 4199.038 / 3525.682 | 18.930 / 14.347 | 4820.141 / 4138.424 |
| 2 | 8 | 169.642 | 223.941 | +32.01% | 6801.593 / 5696.669 | 22.822 / 15.394 | 8541.590 / 7137.231 |
| 4 | 1 | 65.639 | 81.696 | +24.46% | 842.008 / 681.608 | 12.859 / 10.321 | 864.678 / 691.672 |
| 4 | 2 | 104.367 | 134.065 | +28.46% | 1635.777 / 1101.515 | 14.495 / 11.160 | 1660.019 / 1330.662 |
| 4 | 4 | 147.897 | 197.560 | +33.58% | 3404.067 / 2711.783 | 16.642 / 12.028 | 5228.649 / 3974.000 |
| 4 | 8 | 195.228 | 271.694 | +39.17% | 5512.537 / 4373.875 | 21.132 / 13.689 | 6958.959 / 5529.009 |

各并发档在服务内共享warm cache，smoke及11888/12000预热不变；两种实现顺序运行，各自按C1、C2、C4、C8测量，不是交错同址或饱和吞吐测试。整轮仅入口和全部服务退出后各检查一次GPU状态，不证明采样期间持续隔离。默认`SGLANG_USE_PYHIP_QSA=0`保持不变。

PyHIP的kernel都在各入口的第一次调用时编译（见“目录和依赖”末段）：decode的kernel在SGLang启动时的CUDA graph capture中编译，prefill indexer和attention在SGLang自带的预热请求中编译，都在服务打印就绪之前完成；若用`--skip-server-warmup`启动，这次编译会落在第一个真实请求上。上表的Triton缓存复制自较早一轮服务（已含SGLang原生kernel在这些负载下的变体），FlyDSL缓存清空后重建，服务启动后的每一次Triton编译都记录。两组16场测量中都没有Triton编译，PyHIP的FlyDSL kernel全部在数值验收服务的启动阶段编完；测量中只有原生TP4 C4遇到一次AITER FlyDSL MoE kernel的新档位编译，日志中没有可见停顿。缓存为空时，SGLang原生kernel（如apply_interleaved_rope_kernel、_sparse_gqa_chunk_prefill）和AITER MoE也会在服务中按新形状编译：一次冷缓存测量中原生TP2在C4/C8各编译19/12次（每rank合计6.0/3.4秒），吞吐低约5%–6%。新环境应先用代表性负载预热，或保留编译缓存。

本轮独立数值验收TP2/TP4各35请求全部通过，TEST=1，覆盖24k分块和32并发；每个rank的12个QSA层都完成attention与prefill indexer校验（各372次）。这不是模型质量或生成文本bitexact验收。12份独立rank trace确认各PyHIP rank运行当前decode top-k（4请求profile中每rank 240次），原生组无该kernel；profile计时不混入本表。原始证据与过程由[源码readme](../../src/pyhip/ops/qsa/flydsl/readme.md)索引，本页命令不依赖本地未提交研究脚本或报告。

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

约束：Q/O为连续BF16 `[M,H,256]`，K/V为连续BF16 `[N,HK,256]`，`H/HK≤16`，16B对齐、inference-only；QSA不接受5D KV。`indices`是连续int32 `[M,2051]`，每行最多512个完整唯一四token块（块可乱序），接该query的0–3因果尾token，再填−1；ID为请求内逻辑token。`query_lens`/`prefix_lens`是host长度，packed请求的K/V按请求拼接；默认单请求prefix=`N-M`。默认scale=1/16，`out`不与输入重叠。违反上述选择布局的行会在GPU上trap，进程中止。

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

`projected_index_qk`是调用方GEMM输出的连续BF16 `[M,640]`（4×128 query＋1×128 key）。`indexer_metadata`必须包含[原始接口](../../src/pyhip/ops/qsa/flydsl/indexer.py#L289)的以下键，所有tensor在同一GPU：

| 参数 | 契约 |
|---|---|
| `heads`、`seq_lens`、`extend_lens` | 4；host最终序列长度和本次query长度；prefix按4token组对齐 |
| `positions`、`logical_positions` | int64 `[M]`或三轴`[3,M]`；int64 `[M]`，请求内prefix+i，不符时GPU trap、进程中止 |
| `state_slots`、`key_state`、`rope_state` | int64 `[M]` ring槽；BF16 `[slots,1,128]`；int64 `[slots,3]`，原地更新 |
| `write_locs`、`member_rows`、`group_sequences`、`group_ends` | int32组写槽、int64本次QK首成员行、请求编号、请求内组末位置；保留原slot0 padding规则 |
| `rope_matrix`、`compressed` | int64 `[M,3]`；BF16 `[compressed_slots,1,128]`，原地更新 |
| `token_slot_table` | int32 `[requests,max_seq]`，每token物理槽，组首物理槽/4定位compressed key |
| `cos_sin_cache`、`axis_map` | 连续BF16/FP32 `[positions,64]`；int32 `[32]`选择三轴 |
| `q_weight`、`k_weight`、`q_eps`、`k_eps` | BF16 `[128]` Gemma RMSNorm增量权重；host epsilon |

单请求最多65536个compressed key/262144token；返回int32 `[M,2051]`，每行块号升序，可原样交给attention（attention对每行照常校验和排序，复制或改写后的indices结果相同）。slot0与ring保留行是惰性写入区域，不能作为有效数据读取。不要把mean、RoPE或slot规划隐去当成免费预处理。

Decode使用`decode_indexer(q, cache, page_table, lengths, query_positions, sequence_lengths)`，或`decode_forward(projected_index_qk, **decode_metadata)`；实际签名和graph要求见[indexer.py](../../src/pyhip/ops/qsa/flydsl/indexer.py)。全部算子由PyHIP实现，无需SGLang/AITER。完整decode每行必须属于不同请求，page table是16-key compressed page，长度是compressed key数，表宽最多4096页（65536个compressed key）。先eager预热logits及top-k，再capture；graph重放可原地更新长度和位置。prefill入口不是通用decode替代。

### attention CUDA graph

复用同一stream/layout先eager预热，再capture。同一stream上的各layout共享一份按需增长的scratch，capture时固定当前缓冲；因此在同一stream上capture的graph不能彼此并发重放，也不能与该stream上的eager调用并发。图内只调用已经预热的attention；维度、长度和scale不能在capture时首次出现。

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

## 5. 真实TP2/TP4服务测试

SGLang直接依赖已安装PyHIP，`SGLANG_USE_PYHIP_QSA=0/1`选择原生/PyHIP，`SGLANG_TEST_PYHIP_QSA=1`仅用于数值验收。性能计时必须关闭数值检查，GEMM保持原SGLang路径，不使用临时插件。

以下直接使用SGLang已提交的launcher和Python benchmark模块。先在一个终端启动服务；TP_SIZE取2或4，SGLANG_USE_PYHIP_QSA取0或1。四种配置依次运行，不并行占用相同GPU。两种实现使用相同模型、seed、预热、请求长度及内存配置。

```bash
cd /opt/sglang
PATH="/opt/lc/pyhip/.venv/bin:$PATH" TP_SIZE=2 \
  HOST=127.0.0.1 PORT=9080 SGLANG_USE_PYHIP_QSA=1 SGLANG_TEST_PYHIP_QSA=0 \
  LOG_FILE=/opt/lc/pyhip/mytest/mydata/qsa_serving_new/tp2_pyhip/server.log \
  bash scripts/launch_qwen38_flash_next_fp8_mi308x_pure_tp_4_or_8_or_2.sh --random-seed 42
```

服务就绪并完成统一预热后，在另一终端执行。替换输出目录标识以匹配当前TP/实现；每种配置依次测C1/C2/C4/C8：

```bash
cd /opt/lc/pyhip
OUT="$PWD/mytest/mydata/qsa_serving_new/tp2_pyhip"
for concurrency in 1 2 4 8; do
  mkdir -p "$OUT/c${concurrency}"
  pushd "$OUT/c${concurrency}" >/dev/null
  /opt/lc/pyhip/.venv/bin/python -m sglang.bench_serving \
    --backend sglang --model /models/Qwen3.8-Flash-Next-PTPC-FP8 \
    --dataset-name random --host 127.0.0.1 --port 9080 --num-prompts 32 \
    --random-input 12000 --random-output 350 --random-range-ratio 1.0 \
    --max-concurrency "$concurrency"
  popd >/dev/null
done
```

正式测试在整个任务开始和结束各记录一次GPU状态，不在每个并发/单kernel之间检查。验证运行使用TEST=1且不作为性能，计时使用TEST=0；只清理自己启动的服务，不停止其他工作负载。