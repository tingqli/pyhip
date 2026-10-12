# QSA：使用说明与最终性能对比

QSA分为indexer（选token）与attention（计算输出）。本页只维护当前使用说明和最终有效性能；优化过程、中间版本、失败诊断统一追加到[源码readme](../../src/pyhip/ops/qsa/flydsl/readme.md)。已安装实现不依赖实验目录或临时插件。

## 目录和依赖

| 内容 | 入口 |
|---|---|
| 完整attention：恢复/校验/分流/构表/union/direct | [attention.py](../../src/pyhip/ops/qsa/flydsl/attention.py#L179) |
| prefill indexer、decode indexer（含MTP verify） | [indexer.py](../../src/pyhip/ops/qsa/flydsl/indexer.py) |
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

所有kernel都在各入口的第一次调用时编译，之后任何批次组合都不再编译：每种head形状（H, HK, scale）第一次调用attention时，编译该形状可能用到的全部变体（不建计划、每种union tile行数）；prefill indexer、decode和MTP verify（`decode_indexer(verify=True)`）的kernel在各自第一次调用时编译。Triton和FlyDSL缓存都为空时，第一次attention在H12/H6/H3（TP2/4/8）约需36/39/68秒，第一次prefill indexer约2秒，第一次decode约3秒。服务应在启动或预热阶段完成这些调用；CUDA graph capture中不编译，capture前仍须eager预热。

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
  --rows 1 32 --lengths 12000 --perf --output mytest/mydata/qsa_decode_perf_new
```

不依赖真实capture或数据集。CLI默认执行整体性能和当前逐kernel计时；`--check-only`仅验证，`--perf`作为兼容参数保留。省略`--output`时自动创建mytest/mydata下的新目录。Attention默认M12000、TP2/4/8；indexer默认prefill12000和decode的1/32行x12000token。Attention的qsa/forced_direct/forced_union比较相同选择；prefix_*三项只比较共同的2051行完整因果前缀，不能与长稀疏选择当作同语义。

Attention完整计时含恢复/校验/构表/分流、必要KV pack及计算，不含公共入口检查、workspace查找、输出分配、JIT、indexer或服务KV gather。Indexer两种模式都从投影后输入开始，不含projection GEMM；prefill为eager调用，decode为单次graph replay，包含完整prep和selection。

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
decode_r1_n12000   1:qsa_indexer_decode_logits            <实测值>    valid/invalid
```

### 当前实测

2026-10-02/03，MI308X/gfx942，原cudaPerf、10buffers、2warmup、128samples，每次运行使用新建的FlyDSL缓存目录，均在GPU2（PCI 0000:a4:00.0）上实测。Attention是2026-10-04把K3路由改为单一步数阈值之后重测的；indexer是10-02去掉整数特化和首次调用预编译之后测的。每次开始前8张卡use均为0%。Attention测量期间，本会话在GPU5、GPU6上同时运行check-only和pytest；indexer测量期间，本会话在GPU4–6上运行编译清单；都没有使用GPU2。按当前协议未做硬件门禁。各组均通过正确性与源码哈希检查。单位为微秒，取全部样本中位数；单节点图重放包含该launch的固定开销，不能把这些中位数相加作为整体调用时间。

Attention：M=N=12000、D256、KV1，默认合成选择及自动分流。TP2/4/8是单卡H12/H6/H3形状。完整因果前缀行也和其它行一样在union/direct间选择；总行数少于`max(384, 64*H/HK)`（H12为768，H6/H3为384）时不建union计划，全部走direct。各请求KV按4-token对齐后PK+PV不超过64MiB时direct走packed（含多请求和KV长度非4倍数），超出时走raw。逐tile公式之后，attention_order_masks再做一次全局路由：union受最长任务限制时选一个步数阈值，把更长的union tile改走direct，最多全部改走direct。Union tile为H12 10行、H6 16行、H3 32行；大单请求无prefix且行数足够时H6取21行、H3取42行，本例TP4/TP8即为21/42。本例union/direct行数为TP2 6330/5670、TP4 10053/1947、TP8 12000/0。合成选择的块号乱序，attention_recover_scatter逐行排序；PyHIP prefill indexer的升序输出同样排序，只有块号恰为0,1,2,…的完整因果前缀行跳过排序。

| Kernel | TP2/H12 | TP4/H6 | TP8/H3 |
|---|---:|---:|---:|
| attention_recover_scatter（含逐行校验） | 100.601 | 99.140 | 101.481 |
| attention_compact | 27.320 | 19.360 | 17.720 |
| attention_order_masks（含全局路由） | 36.301 | 30.000 | 41.820 |
| attention_union | 944.925 | 1219.287 | 951.305 |
| attention_pack_kv | 17.740 | 15.600 | 9.640 |
| attention_direct（packed） | 1427.908 | 542.623 | 14.601 |

TP8全部12000行走union，pack/direct只测得gated空分支启动开销，不是实际direct耗时。整体auto QSA另测为2515.173 / 1895.651 / 1100.786微秒；2051行完整因果前缀（prefix_qsa，KV长度不是4的倍数）为204.981 / 142.181 / 135.921微秒。

Prefill indexer：M12000、4个D128 Q头、ratio4、top512；投影GEMM不在计时范围。logits按token_slot_table直接从压缩key池读取所有key（含prefix），不另拷key。

| Kernel | 中位耗时 |
|---|---:|
| indexer_q_prep | 39.760 |
| indexer_k_compress | 13.920 |
| qsa_indexer_logits | 136.381 |
| qsa_indexer_topk（含位置校验） | 123.320 |

整体prefill_indexer另测为291.061微秒，包含公开入口内的分配与准备。

Decode indexer：每请求3000个压缩key（12000 token），包含投影后的prep和选块，不含GEMM。Decode top-k每行一个512线程CTA，与prefill共用同一选择实现。

| Kernel | B1 | B32 |
|---|---:|---:|
| indexer_decode_prep | 9.960 | 10.920 |
| qsa_indexer_decode_logits | 9.840 | 16.680 |
| qsa_indexer_decode_topk（含展开） | 14.640 | 15.320 |

整体图重放B1/B32为22.240 / 31.400微秒。

以上为当前版本绝对性能，不包含旧新对照；attention共3个case、18行kernel结果、2304条整体raw，indexer为prefill 1个case、decode 2个case，共10行kernel结果、384条整体raw，均未筛除慢样本。记录索引保留在[源码readme](../../src/pyhip/ops/qsa/flydsl/readme.md)，复现只需本页已跟踪入口。

公开入口的CPU下发时间（GPU保持忙，60次中位数，TP2/4/8相同）：attention建union计划时约87µs，不建计划时约54µs；prefill_indexer在M12000时约195µs，3请求时约247µs。图重放不含这部分；eager prefill中GPU时间短于它的调用（如2051行完整前缀、64行长prefix）受host限制。

## 3. 最终服务性能对比

2026-10-04，PyHIP为本页描述的当前实现（HEAD `601c01e`之上的工作区改动），SGLang HEAD `c3c4bc8`，gfx942/MI308X，模型Qwen3.8-Flash-Next-PTPC-FP8。PyHIP使用独立FlyDSL decode top-k/展开。原生与PyHIP在相同TP及配置下测量C1/C2/C4/C8，共16场、512请求；每场32请求、名义12000输入/350输出、服务seed42，计时关闭数值校验和profiler。源码哈希、请求量、配置一致性及整轮入口/出口检查均通过。

服务配置为chunked prefill 16384、最大运行请求32、decode graph最大batch 32、关闭radix cache、AITER attention/MoE backend。两组的实际`mem_fraction_static`一致：TP2为0.8075、TP4为0.7225；服务和客户端沿用4个CPU线程。

| TP | 并发 | 原生token/s | PyHIP token/s | 吞吐变化 | TTFT中位数(ms，原生/PyHIP) | ITL中位数(ms，原生/PyHIP) | TTFT p99(ms，原生/PyHIP) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2 | 1 | 61.066 | 75.029 | +22.86% | 1046.033 / 881.239 | 13.416 / 10.840 | 1061.528 / 894.048 |
| 2 | 2 | 94.867 | 119.243 | +25.69% | 2040.948 / 1398.219 | 15.257 / 11.856 | 2061.682 / 1743.694 |
| 2 | 4 | 127.894 | 162.101 | +26.75% | 4207.044 / 3529.918 | 19.002 / 14.361 | 4844.628 / 4137.701 |
| 2 | 8 | 169.357 | 223.683 | +32.08% | 6820.034 / 5705.494 | 22.866 / 15.425 | 8571.052 / 7141.292 |
| 4 | 1 | 65.860 | 81.790 | +24.19% | 844.297 / 680.019 | 12.800 / 10.313 | 860.221 / 689.080 |
| 4 | 2 | 104.564 | 134.445 | +28.58% | 1639.680 / 1099.245 | 14.446 / 11.121 | 1660.176 / 1330.014 |
| 4 | 4 | 148.010 | 197.280 | +33.29% | 3412.374 / 2712.255 | 16.622 / 12.031 | 5236.036 / 4027.899 |
| 4 | 8 | 195.293 | 271.719 | +39.13% | 5517.120 / 4376.158 | 21.111 / 13.681 | 6976.845 / 5531.324 |

各并发档在服务内共享warm cache，smoke及11888/12000预热不变；两种实现顺序运行，各自按C1、C2、C4、C8测量，不是交错同址或饱和吞吐测试。整轮仅入口和全部服务退出后各检查一次GPU状态，不证明采样期间持续隔离。默认`SGLANG_USE_PYHIP_QSA=0`保持不变。

PyHIP的kernel都在各入口的第一次调用时编译（见“目录和依赖”末段）：decode的kernel在SGLang启动时的CUDA graph capture中编译，prefill indexer和attention在SGLang自带的预热请求中编译，都在服务打印就绪之前完成；若用`--skip-server-warmup`启动，这次编译会落在第一个真实请求上。上表的Triton缓存复制自较早一轮服务（已含SGLang原生kernel在这些负载下的变体），FlyDSL缓存清空后重建，服务启动后的每一次Triton编译都记录。两组16场测量中都没有Triton编译，PyHIP的FlyDSL kernel全部在数值验收服务的启动阶段编完；测量中只有原生TP4 C4遇到一次AITER FlyDSL MoE kernel的新档位编译，日志中没有可见停顿。缓存为空时，SGLang原生kernel（如apply_interleaved_rope_kernel、_sparse_gqa_chunk_prefill）和AITER MoE也会在服务中按新形状编译：一次冷缓存测量中原生TP2在C4/C8各编译19/12次（每rank合计6.0/3.4秒），吞吐低约5%–6%。新环境应先用代表性负载预热，或保留编译缓存。

本轮独立数值验收TP2/TP4各35请求全部通过，TEST=1，覆盖24k分块和32并发；每个rank的12个QSA层都完成attention与prefill indexer校验（各372次）。这不是模型质量或生成文本bitexact验收。12份独立rank trace确认各PyHIP rank运行当前decode top-k（4请求profile中每rank 240次），原生组无该kernel；profile计时不混入本表。原始证据与过程由[源码readme](../../src/pyhip/ops/qsa/flydsl/readme.md)索引，本页命令不依赖本地未提交研究脚本或报告。

### MTP（EAGLE 3/1/4）

2026-10-08，模型、launcher、服务配置和32请求、名义12000输入/350输出的协议同上（实际`mem_fraction_static`仍为TP2 0.8075、TP4 0.7225），SGLang另加`--speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4`（QSA要求topk为1、草稿token不超过4）。两组都开MTP；PyHIP组的indexer在prefill和CUDA graph TARGET_VERIFY中由PyHIP计算，verify的attention、draft模型和draft extend仍是SGLang原生。SGLang的pending ring为每个请求2×ratio个槽：verify窗口跨过压缩边界时，不会再覆盖该组在窗口之前的成员。PyHIP为HEAD `cd07379`之上的工作区改动，SGLang为HEAD `c3c4bc8`之上的工作区改动。

| TP | 并发 | 原生token/s | PyHIP token/s | 吞吐变化 | TTFT中位数(ms，原生/PyHIP) | ITL中位数(ms，原生/PyHIP) | 接受长度(原生/PyHIP) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2 | 1 | 85.565 | 100.522 | +17.48% | 1111.322 / 952.165 | 7.472 / 6.323 | 3.52 / 3.50 |
| 2 | 2 | 123.624 | 152.574 | +23.42% | 1118.417 / 966.886 | 8.411 / 6.583 | 3.51 / 3.51 |
| 2 | 4 | 161.350 | 206.102 | +27.74% | 1131.475 / 976.806 | 10.421 / 7.324 | 3.51 / 3.51 |
| 2 | 8 | 190.011 | 246.494 | +29.73% | 2541.675 / 2040.588 | 14.152 / 8.590 | 3.51 / 3.49 |
| 4 | 1 | 94.603 | 114.947 | +21.50% | 910.442 / 730.136 | 6.883 / 5.766 | 3.47 / 3.50 |
| 4 | 2 | 138.183 | 177.980 | +28.80% | 933.431 / 746.129 | 7.959 / 6.130 | 3.47 / 3.51 |
| 4 | 4 | 181.547 | 242.920 | +33.81% | 959.175 / 755.141 | 9.879 / 6.733 | 3.49 / 3.51 |
| 4 | 8 | 220.971 | 312.808 | +41.56% | 2057.562 / 1673.210 | 13.188 / 7.641 | 3.50 / 3.52 |

16场测量中都没有服务中的Triton编译。C8的TTFT主要由排队决定（解码更快时请求到达更密，与prefill交错更多），不宜单独比较。C1 profile中每个verify graph的indexer由约5.6ms降到0.35ms，verify graph在TP2由21.56ms降到16.18ms、TP4由19.74ms降到14.21ms（profiler给每个kernel多算约1µs）。与上表非MTP的PyHIP相比，MTP+PyHIP的吞吐在TP2高10%–34%、TP4高15%–41%，并发越高收益越小。

接受长度两边相当。ring修复之前的同一协议中PyHIP低1%–4%，原因已查明（[逐请求研究](../../mytest/mydata/qsa_accept_study_20261008_01/summary.json)：64个同协议prompt，C1贪心，四种组合各跑两轮）：两边生成的文本相同时，每一步接受的草稿相同，同一段文本的teacher-forced似然也相同；差距全部来自生成的文本不同。prefill的微小数值差在near-tie处翻转贪心选择，原生自己两轮之间也有45%的prompt输出不同、接受长度差2.3%；基准的32个prompt里有2个在当时的PyHIP下稳定走到更难预测的分支，占了约75%的差距。所以这里的接受长度只能在几个百分点内比较，不反映精度高低。

数值验收TP2/TP4各35请求全部通过（TEST=1）：每个rank的12个QSA层各完成234/247次graph verify校验，q、ring和压缩key与原生逐bit一致、选择合法。ring修复之前的数字见[源码readme](../../src/pyhip/ops/qsa/flydsl/readme.md)的优化历史。

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

### Indexer接口

[prefill_indexer](../../src/pyhip/ops/qsa/flydsl/indexer.py#L323)和[decode_indexer](../../src/pyhip/ops/qsa/flydsl/indexer.py#L401)都从调用方`index_qk_proj`之后的`qk`开始，返回同一种token选择，可原样交给attention。除`qk`外都是关键字参数，两个入口中同名参数含义相同；只依赖PyHIP，不需要SGLang/AITER。

共同约定：

- 只支持gfx942。Indexer固定4个Q头、1个K头、D=128、压缩比4、每行选512个压缩块。**indexer的D128投影与attention的D256是不同投影**，compressed池也不是attention的KV缓存。
- 所有tensor在同一GPU，除表中注明外必须连续。`key_state`、`rope_state`、`compressed`原地更新。
- 槽号全部由调用方给出。SGLang的pending ring每请求8个槽（压缩比的2倍）：请求r（从1开始）的位置p在槽`8r + p%8`；请求0从不分配，`[0,8)`是dump行。首token物理槽为s的4-token组，压缩key在`compressed[s//4]`；slot 0是惰性写入位置。dump行和slot 0会被并发写入，不能当数据读。
- 返回int32 `[rows,2051]`：每行先是选中的压缩块，按块号升序、每块展开成4个token（最多2048个）；紧接着是0–3个因果尾token（最后一个完整组之后、到query本身为止）；其余填−1。Token ID是请求内逻辑位置。位置p的query可见`(p+1)//4`个压缩key，不超过512个时全选；分数相同时取块号小的。
- 第一次调用时编译（缓存为空时prefill约2秒、decode约3秒），之后任何长度组合都不再编译。

下面的示例共用这组参数（普通RoPE，rotary_dim=64）：

```python
import torch

device = torch.device("cuda", 0)
D = 128
inv_freq = 1.0 / 10000 ** (torch.arange(0, 64, 2, device=device) / 64)
angles = torch.arange(262144 + 64, device=device)[:, None] * inv_freq
cos_sin_cache = torch.cat((angles.cos(), angles.sin()), 1).to(torch.bfloat16)  # [positions, 64]：前半cos，后半sin
axis_map = torch.zeros(32, device=device, dtype=torch.int32)                  # 普通RoPE：每个旋转对都用轴0
q_weight = torch.zeros(D, device=device, dtype=torch.bfloat16)                # Gemma RMSNorm：x·rstd·(1+w)
k_weight = torch.zeros(D, device=device, dtype=torch.bfloat16)
```

#### prefill_indexer

```python
indices = prefill_indexer(qk, *, heads, positions, logical_positions, state_slots, key_state, rope_state,
                          write_locs, member_rows, group_sequences, group_ends, rope_matrix, compressed,
                          token_slot_table, cos_sin_cache, axis_map, q_weight, k_weight, q_eps, k_eps,
                          seq_lens, extend_lens, q_out=None)           # → int32 [T, 2051]
```

T是本批query行数（各请求的行按请求顺序拼接），R是请求数，G是本批写压缩key的组数（可含padding），S、C分别是ring和压缩池的槽数，N是RoPE表的行数。

| 参数 | 类型与shape | 说明 |
|---|---|---|
| `qk` | BF16 `[T,640]` | `index_qk_proj`输出：每行4个query头×128，后接1个key×128 |
| `heads` | int | 必须是4 |
| `seq_lens` | host int序列 `[R]` | 各请求本批之后的总长度（prefix + 本批行数） |
| `extend_lens` | host int序列 `[R]` | 各请求本批的行数（可为0），合计T且大于0；prefix = `seq_lens − extend_lens`必须是4的倍数 |
| `positions` | int64 `[T]`或`[3,T]`，末维stride为1 | RoPE位置；一维时三个轴相同，`[3,T]`是MRoPE的三个轴 |
| `logical_positions` | int64 `[T]` | 请求内位置，必须等于prefix + i；不符时top-k在GPU上trap，进程中止 |
| `state_slots` | int64 `[T]` | 每行写进pending ring的槽：请求最后一个不完整组的行写`8r + p%8`，其余行写dump行`p%8` |
| `key_state` | BF16 `[S,1,128]` | pending ring中的原始key，在`state_slots`处写入 |
| `rope_state` | int64 `[S,3]` | pending ring中的三轴位置，与`key_state`同槽号 |
| `write_locs` | int32 `[G]` | 本批凑满的组（4个成员都在本批）写入的压缩槽；padding填0 |
| `member_rows` | int64 `[G]` | 每组首成员在`qk`中的行号；padding填0 |
| `group_sequences`、`group_ends` | int64 `[G]` | 每组所属请求、组末token的请求内位置；只检查形状，kernel不读 |
| `rope_matrix` | int64 `[T,3]` | 每行的三轴位置（即`positions`转置）；压缩key用首成员那一行的位置 |
| `compressed` | BF16 `[C,1,128]`，小于2GiB | 压缩key池：本批在`write_locs`写入；logits经`token_slot_table`读取全部可见key（含prefix） |
| `token_slot_table` | int32 `[R, ≥max(seq_lens)]`，列stride为1 | 各请求每个token的物理槽；请求s的第j个压缩key在`token_slot_table[s,4j]//4` |
| `cos_sin_cache` | BF16或FP32 `[N,rotary_dim]` | RoPE表，每行前半cos、后半sin，须覆盖所有位置；要求`rotary_dim%4==0`，`rotary_dim/2`与`128−rotary_dim`是2的幂（模型为64）；只有BF16表与SGLang逐bit一致 |
| `axis_map` | int32 `[rotary_dim/2]` | 每个旋转对用哪个位置轴（0/1/2，按MRoPE分段）；普通RoPE全为0 |
| `q_weight`、`k_weight` | BF16 `[128]` | Gemma RMSNorm权重，按`x·rstd·(1+w)`作用于q和k |
| `q_eps`、`k_eps` | float | RMSNorm的epsilon |
| `q_out` | 可选，BF16 `[T,4,128]` | 给定时写入归一化并加RoPE后的index Q（SGLang数值校验用） |

注意事项：

- 只用于eager调用：host按`(seq_lens, extend_lens)`规划布局（缓存最近8种），一次pinned上传，不同步stream。
- 单请求最多65536个压缩key（262144 token）；logits临时缓冲每块不超过256MiB。
- Chunked prefill时，前面各块写入的压缩key由logits经`token_slot_table`读取；每块只压缩4个成员都在本块的组，所以prefix必须按4对齐。
- 每次调用返回新分配的indices，之后的调用不会改写。

示例：单请求、没有prefix。真实调用时`qk`来自`index_qk_proj`，缓存和槽号来自调用方的KV管理。

```python
from pyhip.ops.qsa.flydsl.indexer import prefill_indexer

T, RING, r = 12000, 8, 1                    # 请求槽1；请求0是dump
pos = torch.arange(T, device=device)        # int64
table = (64 + pos).to(torch.int32)[None]    # token → 物理槽；4-token组按4对齐
groups = torch.arange(T // 4, device=device)
key_state = torch.zeros((r + 1) * RING, 1, D, device=device, dtype=torch.bfloat16)
rope_state = torch.zeros((r + 1) * RING, 3, device=device, dtype=torch.int64)
compressed = torch.zeros((64 + T) // 4 + 1, 1, D, device=device, dtype=torch.bfloat16)
indices = prefill_indexer(
    torch.randn(T, 5 * D, device=device).to(torch.bfloat16),       # index_qk_proj输出
    heads=4, positions=pos, logical_positions=pos,
    state_slots=torch.where(pos >= T // 4 * 4, r * RING + pos % RING, pos % RING),
    key_state=key_state, rope_state=rope_state,
    write_locs=(table[0, groups * 4] // 4).int(), member_rows=groups * 4,
    group_sequences=torch.zeros_like(groups), group_ends=groups * 4 + 3,
    rope_matrix=pos[:, None].expand(-1, 3).contiguous(), compressed=compressed,
    token_slot_table=table, cos_sin_cache=cos_sin_cache, axis_map=axis_map,
    q_weight=q_weight, k_weight=k_weight, q_eps=1e-6, k_eps=1e-6,
    seq_lens=(T,), extend_lens=(T,))
# int32 [12000, 2051]；例如第3000行可见750个压缩key，选512块共2048个token，再接1个尾token
# 交给attention（q/k/v是attention自己的D256投影）：attention(q, k, v, indices, query_lens=(T,), prefix_lens=(0,))
```

#### decode_indexer

```python
indices = decode_indexer(qk, *, positions, logical_positions, state_slots, key_state, rope_state, write_locs,
                         group_locs, compressed, page_table, lengths, cos_sin_cache, axis_map, q_weight,
                         k_weight, q_eps, k_eps, seq_lens, verify=False, q_out=None, logits_out=None)
# → int32 [B, 2051]
```

B是行数（每行一个query token，含CUDA graph的padding行），P是页表宽度（页数）。`key_state`、`rope_state`、`cos_sin_cache`、`axis_map`、`q_weight`、`k_weight`、`q_eps`、`k_eps`与prefill相同，下表不再重复。

| 参数 | 类型与shape | 说明 |
|---|---|---|
| `qk` | BF16 `[B,640]` | 每行一个新token的`index_qk_proj`输出 |
| `positions` | int64 `[B]`或`[3,B]` | RoPE位置，同prefill |
| `logical_positions` | int32（也可int64）`[B]` | 本行query的请求内位置（= `seq_lens − 1`）；decode不校验位置 |
| `seq_lens` | int32 `[B]` | 本行所在序列含当前token的长度 |
| `state_slots` | int64 `[B]` | 当前token写进ring的槽`8r + p%8`；padding行写dump行 |
| `group_locs` | int32 `[B,4]` | 以本位置结尾的组的4个成员（位置p−3..p，最早的在前，小于0时按0）在ring中的槽`8r + 成员位置%8` |
| `write_locs` | int32 `[B]` | 本token补满一组（`seq_lens%4==0`）时该组的压缩槽，否则0 |
| `compressed` | BF16 `[C,1,128]`，小于2GiB | 压缩key池，按每页16个key读取；本步补满的组先写入，再参与本步选择 |
| `page_table` | int32 `[B,P]`，P ≤ 4096 | 本行第p页对应压缩key `16·page_table[row,p] + [0,16)`；logits宽16·P |
| `lengths` | int32 `[B]` | 本行可见的压缩key数，= `seq_lens // 4` |
| `verify` | bool | False：各行必须属于不同请求；True：MTP的TARGET_VERIFY窗口，一个请求占连续W（≤4）行 |
| `q_out` | 可选，BF16 `[B,4,128]` | 写入归一化并加RoPE后的index Q |
| `logits_out` | 可选，FP32 `[B,16·P]` | 写入logits；其存储须在末尾多留512个值（top-k按512个值对齐读取） |

注意事项：

- 所有metadata都在设备上，可在CUDA graph中capture。必须先eager调用一次以完成编译（第一次`verify=True`另编译verify用的两个kernel），未预热就在capture中调用会报错。重放前原地更新各张量的内容，地址不能变。
- `verify=False`时prep只看得到本行自己的ring写入，所以同一请求不能占多行。`verify=True`时压缩在所有行写完ring之后另起一个launch（与SGLang顺序一致）；ring每请求必须有8个槽，窗口才不会覆盖组内窗口之前的成员。
- 调用前，prefill和之前各步写入的`key_state`、`rope_state`、`compressed`必须已就绪。
- padding行用请求0：`state_slots`、`group_locs`指向dump行，`write_locs`和`lengths`为0；其输出忽略。

示例：CUDA graph decode，每请求一行，按每页64 token分配物理槽。

```python
from pyhip.ops.qsa.flydsl.indexer import decode_indexer

B, CONTEXT, RING = 4, 16384, 8
pages = CONTEXT // 64                       # 每页64 token = 16个压缩key
requests = torch.arange(1, B + 1, device=device)
base = 64 * (1 + (requests - 1) * pages)    # 请求r占物理槽base[r] + [0, CONTEXT)
key_state = torch.zeros((B + 1) * RING, 1, D, device=device, dtype=torch.bfloat16)
rope_state = torch.zeros((B + 1) * RING, 3, device=device, dtype=torch.int64)
compressed = torch.zeros((1 + B * pages) * 16, 1, D, device=device, dtype=torch.bfloat16)
page_table = (base[:, None] // 64 + torch.arange(pages, device=device)).int()
# graph中地址固定的输入
qk = torch.empty(B, 5 * D, device=device, dtype=torch.bfloat16)
positions = torch.empty(B, device=device, dtype=torch.int64)
state_slots = torch.empty_like(positions)
seq_lens = torch.empty(B, device=device, dtype=torch.int32)
logical_positions, lengths, write_locs = (torch.empty_like(seq_lens) for _ in range(3))
group_locs = torch.empty(B, 4, device=device, dtype=torch.int32)

def step(seq):                              # int64 [B]：各请求含当前token的长度
    p = seq - 1
    qk.copy_(torch.randn_like(qk))          # 新token的index_qk_proj输出
    positions.copy_(p)
    logical_positions.copy_(p)
    seq_lens.copy_(seq)
    lengths.copy_(seq // 4)
    state_slots.copy_(requests * RING + p % RING)
    members = (p[:, None] - torch.arange(3, -1, -1, device=device)).clamp_min(0)
    group_locs.copy_(requests[:, None] * RING + members % RING)
    write_locs.copy_(torch.where(seq % 4 == 0, (base + p) // 4, 0))

def run():
    return decode_indexer(
        qk, positions=positions, logical_positions=logical_positions, state_slots=state_slots,
        key_state=key_state, rope_state=rope_state, write_locs=write_locs, group_locs=group_locs,
        compressed=compressed, page_table=page_table, lengths=lengths, cos_sin_cache=cos_sin_cache,
        axis_map=axis_map, q_weight=q_weight, k_weight=k_weight, q_eps=1e-6, k_eps=1e-6, seq_lens=seq_lens)

seq = torch.tensor([3001, 12000, 5, 9000], device=device)
step(seq)
run()                                       # eager预热，完成编译
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    out = run()                             # out在graph的内存池里
step(seq + 1)                               # 下一个token：原地更新内容
graph.replay()                              # out各行有效token数：2050、2049、6、2049
```

MTP verify：一个请求占连续W（≤4）行，每行的metadata按该行自己的位置计算，再传`verify=True`（capture方式同上）：

```python
W = 4
requests_w, base_w = requests.repeat_interleave(W), base.repeat_interleave(W)
seq_w = (torch.tensor([3001, 12000, 5, 9000], device=device).repeat_interleave(W)
         + torch.arange(W, device=device).repeat(B))                    # 窗口内各行的序列长度
p = seq_w - 1
members = (p[:, None] - torch.arange(3, -1, -1, device=device)).clamp_min(0)
out = decode_indexer(
    torch.randn(B * W, 5 * D, device=device).to(torch.bfloat16), positions=p, logical_positions=p.int(),
    state_slots=requests_w * RING + p % RING, key_state=key_state, rope_state=rope_state,
    write_locs=torch.where(seq_w % 4 == 0, (base_w + p) // 4, 0).int(),
    group_locs=(requests_w[:, None] * RING + members % RING).int(), compressed=compressed,
    page_table=page_table.repeat_interleave(W, 0).contiguous(), lengths=(seq_w // 4).int(),
    cos_sin_cache=cos_sin_cache, axis_map=axis_map, q_weight=q_weight, k_weight=k_weight,
    q_eps=1e-6, k_eps=1e-6, seq_lens=seq_w.int(), verify=True)   # int32 [16, 2051]
```

以上示例已在gfx942上运行：[检查脚本](../../mytest/qsa_readme_indexer_examples.py)按顺序直接执行本节的代码块，并核对注释中的结果。

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

MTP对照在上面的launcher命令后追加`--speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4`，其余不变。

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