# QSA：使用说明与最终性能对比

QSA分为indexer（选token）与attention（计算输出）。本页只维护当前使用说明和最终有效性能；优化过程、中间版本、失败诊断统一追加到[源码readme](../../src/pyhip/ops/qsa/flydsl/readme.md)。已安装实现不依赖实验目录或临时插件。

## 目录和依赖

| 内容 | 入口 |
|---|---|
| 完整attention：恢复/分流/构表/union/direct | [attention.py](../../src/pyhip/ops/qsa/flydsl/attention.py#L219)（入口）及同目录的`sparse_prefill_*.py`（kernel） |
| prefill indexer、decode indexer（含MTP verify） | [indexer.py](../../src/pyhip/ops/qsa/flydsl/indexer.py) |
| indexer共用的MHA helper | [_common.py](../../src/pyhip/ops/mha/flydsl/_common.py) |
| 合成attention三分支/完整调用 | [test_attention.py](test_attention.py) |
| 合成indexer完整调用 | [test_indexer.py](test_indexer.py) |
| Attention/indexer基本功能 | [test_attention.py](../../tests/ops/qsa/test_attention.py)、[test_indexer.py](../../tests/ops/qsa/test_indexer.py) |
| AITER op test：精度与耗时，对照SGLang原生Triton | [sparse_prefill_op_test.py](../../tests/ops/qsa/sparse_prefill_op_test.py) |
| 当前逐kernel性能 | 随上述两个benchmark默认输出，无独立对照脚本 |
| 共用数据/参考、记录和JSON/CSV导出 | [_attention.py](../../tests/ops/qsa/_attention.py)、[_indexer.py](../../tests/ops/qsa/_indexer.py)、[_benchmark.py](../../tests/ops/qsa/_benchmark.py) |

目标为 **ROCm gfx942（MI308X）、BF16**。使用与ROCm匹配的PyTorch、FlyDSL、Triton，以及NumPy和pytest；AITER op test另需AITER（`aiter.test_common`）、pandas和tabulate。当前验证环境为Python3.10.12、Torch2.12/ROCm7.14、FlyDSL0.4.0（与AITER `2544b2927`固定的版本相同）、Triton3.8。基础PyHIP安装不会自动安装全部可选GPU依赖。

Attention沿用AITER [#6576](https://github.com/ROCm/aiter/pull/6576)的源码结构和kernel名（`sparse_prefill_*`），K1–K3、union和direct全部是FlyDSL；Triton只用于indexer的I1、I2、D0、D0′四个prep kernel。Attention、prefill indexer和decode indexer的计算实现均不依赖SGLang、AITER或experiments/tests/benchmarks。Decode的top-k和token展开使用PyHIP自身的FlyDSL实现，与prefill共享选择核心。测试使用独立tensor夹具和精度参考，不创建模型或临时adapter；只有第5节的服务集成需要SGLang。

```bash
# 在已配置ROCm/PyTorch/FlyDSL的环境安装；不要用CUDA版PyTorch替换ROCm版。
python -m pip install -e .
```

本机后续示例统一使用PyHIP的.venv解释器。新数据目录必须在mytest/mydata下且尚不存在；该目录允许是大容量卷的符号链接。`--gpu`使用当前进程可见的设备编号，`--gpu 0`是第一张可见卡；无需清除GPU/CU屏蔽环境变量。性能测试前自行确认设备空闲。

FlyDSL的持久编译缓存在源码只改了嵌套helper时可能返回旧kernel。升级或修改PyHIP后，先清空该缓存，或为测试设置新的空目录：`export FLYDSL_RUNTIME_CACHE_DIR=$(mktemp -d)`。

Attention的kernel变体在第一次用到时编译（与AITER相同）：head形状、union tile行数、选择宽度T、是否传slot表、池是否超过4GiB等编译期参数每出现一种新组合编译一次，之后同一组合的任何长度都不再编译。FlyDSL缓存为空时，第一个建union计划的调用（如12k prefill）约14–15秒，不建计划的小调用约1.7秒，H6/H3第一次用到16/32行tile的带prefix chunk约7–12秒，第一次传slot表约13秒，池超过4GiB的WIDE约1秒；持久缓存命中时0.02–0.4秒。Prefill indexer、decode和MTP verify（`decode_indexer(verify=True)`）的kernel在各自第一次调用时编译（top-k一次编译uint16/int32块号、长行共行和decode长行多CTA的全部变体），缓存为空时第一次prefill indexer约6秒、第一次decode约6.5秒。服务应保留FlyDSL持久缓存，或在启动预热中发出实际会出现的布局（含带prefix的长请求），否则编译落在触发它的请求上；CUDA graph capture中不编译，capture前仍须eager预热。

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

# AITER op test：功能、graph和1k–32k prefill用例的精度与耗时，对照SGLang原生Triton
HIP_VISIBLE_DEVICES=2 .venv/bin/python tests/ops/qsa/sparse_prefill_op_test.py
HIP_VISIBLE_DEVICES=2 .venv/bin/python tests/ops/qsa/sparse_prefill_op_test.py -c prefill_12000 -g 12 6 3
```

不依赖真实capture或数据集。CLI默认执行整体性能和当前逐kernel计时；`--check-only`仅验证，`--perf`作为兼容参数保留。省略`--output`时自动创建mytest/mydata下的新目录。Attention默认M12000、TP2/4/8；indexer默认prefill12000和decode的1/32行x12000token。Attention的qsa/forced_direct/forced_union比较相同选择；prefix_*三项只比较共同的2051行完整因果前缀，不能与长稀疏选择当作同语义。AITER op test（`-g`是每个KV头的Q头数，12/6/3即TP2/4/8）在控制台打印markdown表：PyHIP与Triton的耗时、TFLOPS、TB/s和与torch参考的不符数（`.02/.02`）。

Attention完整计时含恢复/构表/分流、必要KV pack及计算，不含公共入口检查、workspace查找、输出分配、JIT、indexer或服务KV gather。Indexer两种模式都从投影后输入开始，不含projection GEMM；prefill为eager调用，decode为单次graph replay，包含完整prep和selection。

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
m12000_tp2         0:sparse_prefill_recover_scatter       <实测值>    valid/invalid
m12000_tp2         3:sparse_prefill_union                 <实测值>    valid/invalid
prefill_n12000     0:indexer_q_prep                       <实测值>    valid/invalid
decode_r1_n12000   1:qsa_indexer_decode_logits            <实测值>    valid/invalid
```

### 当前实测

Attention为2026-10-30同步AITER #6576并重新应用之后重测：MI308X/gfx942，GPU2，原cudaPerf、10buffers、2warmup、128samples，两轮（每轮新建FlyDSL缓存目录），表中为两轮各自中位数的平均。Indexer源码在同步中未改，数据为10-29在同一GPU、同一协议下的测量。按当前协议未做硬件门禁。各组均通过正确性与源码哈希检查。单位为微秒；单节点图重放包含该launch的固定开销，不能把这些中位数相加作为整体调用时间。

Attention：M=N=12000、D256、KV1，默认合成选择及自动分流。TP2/4/8是单卡H12/H6/H3形状。完整因果前缀行也和其它行一样在union/direct间选择；总行数少于`max(384, 64*H/HK)`（H12为768，H6/H3为384）时不建union计划，全部走direct。各请求KV按4-token对齐后PK+PV不超过64MiB时direct走packed（含多请求和KV长度非4倍数）；超出时若上下文token数不超过行数的32倍且PK+PV不超过1GiB仍走packed，否则走raw。逐tile公式之后，sparse_prefill_order_masks再做一次全局路由：union受最长任务限制时选一个步数阈值，把更长的union tile改走direct，最多全部改走direct。Union tile为H12 10行、H6 16行、H3 32行；大单请求无prefix且行数足够时H6取21行、H3取42行，本例TP4/TP8即为21/42。本例union/direct行数为TP2 6330/5670、TP4 10053/1947、TP8 12000/0。合成选择的块号乱序，sparse_prefill_recover_scatter逐行排序；PyHIP prefill indexer的升序输出同样排序，只有块号恰为0,1,2,…的完整因果前缀行跳过排序。选择不校验。

| Kernel | TP2/H12 | TP4/H6 | TP8/H3 |
|---|---:|---:|---:|
| sparse_prefill_recover_scatter | 77.451 | 76.381 | 75.900 |
| sparse_prefill_compact | 27.010 | 21.721 | 23.380 |
| sparse_prefill_order_masks（含全局路由） | 24.080 | 31.120 | 38.630 |
| sparse_prefill_union | 937.145 | 1196.716 | 939.865 |
| sparse_prefill_pack_kv | 17.540 | 15.570 | 9.501 |
| sparse_prefill_direct（packed） | 1429.547 | 542.253 | 14.540 |

TP8全部12000行走union，pack/direct只测得gated空分支启动开销，不是实际direct耗时。整体auto QSA另测为2474.413 / 1849.579 / 1065.995微秒；2051行完整因果前缀（prefix_qsa，KV长度不是4的倍数）为199.211 / 136.881 / 134.011微秒。sparse_prefill_compact只读tile各行选中的块，耗时随选中块数增长，不随上下文长度线性增长。

AITER op test的性能用例（单请求完整prefill，T=512，模拟真实indexer相邻行约共享一半块的correlated选择；AITER `run_perftest`，10组轮换输入，torch.profiler统计设备时间）：PyHIP与SGLang原生Triton（无prefix时调用`sparse_gqa_fwd_interface_triton`）的每次调用耗时，GPU3，两轮各新建FlyDSL缓存目录、取较小值，单位微秒；两者与torch参考在`.02/.02`下全部一致。

| 长度 | H12（TP2）PyHIP / Triton | H6（TP4） | H3（TP8） |
|---:|---|---|---|
| 1024 | 84.6 / 284.4（3.4×） | 84.7 / 277.0（3.3×） | 81.8 / 276.4（3.4×） |
| 2048 | 198.4 / 983.1（5.0×） | 135.2 / 930.3（6.9×） | 132.2 / 928.7（7.0×） |
| 4096 | 676.7 / 2751.3（4.1×） | 334.8 / 2723.5（8.1×） | 245.2 / 2718.6（11.1×） |
| 8192 | 1609.4 / 6435.9（4.0×） | 999.5 / 6379.0（6.4×） | 632.2 / 6381.8（10.1×） |
| 12000 | 2472.8 / 9957.9（4.0×） | 1849.4 / 9854.6（5.3×） | 1068.9 / 9859.2（9.2×） |
| 16384 | 4222.5 / 13944.9（3.3×） | 2829.2 / 13816.6（4.9×） | 1952.5 / 13816.2（7.1×） |
| 32768 | 9916.0 / 29013.6（2.9×） | 8490.6 / 28740.6（3.4×） | 7108.7 / 28730.5（4.0×） |

Prefill indexer：M12000、4个D128 Q头、ratio4、top512；投影GEMM不在计时范围。logits按token_slot_table直接从压缩key池读取所有key（含prefix），不另拷key。

| Kernel | 中位耗时 |
|---|---:|
| indexer_q_prep | 39.780 |
| indexer_k_compress | 14.080 |
| qsa_indexer_logits | 136.341 |
| qsa_indexer_topk（含位置校验） | 123.561 |

整体prefill_indexer另测为292.862微秒，包含公开入口内的分配与准备。

Decode indexer：每请求3000个压缩key（12000 token），包含投影后的prep和选块，不含GEMM。Decode top-k每行一个512线程CTA，与prefill共用同一选择实现；页表宽度超过32768个压缩key时另加两个launch，把每行分给多个CTA（见下面的1M）。

| Kernel | B1 | B32 |
|---|---:|---:|
| indexer_decode_prep | 10.000 | 10.840 |
| qsa_indexer_decode_logits | 9.840 | 16.800 |
| qsa_indexer_decode_topk（含展开） | 14.600 | 15.200 |

整体图重放B1/B32为22.120 / 31.320微秒。

1M上下文（262144个压缩key）：decode按本协议B1/B8整体图重放50.420 / 197.261微秒，FP64选择校验全部通过（`--mode decode --rows 1 8 --lengths 1048576`）。其中prep 10.640/11.080、logits 25.680/175.981；top-k分三个kernel：qsa_indexer_decode_topk_hist（每行分给多个CTA，各统计一段的直方图）10.040/13.160，qsa_indexer_decode_topk_gather（找到第512大所在的bin，各段写出更高bin的块和该bin的候选）13.020/16.240，qsa_indexer_decode_topk（每行一个CTA排名候选、归并并展开）13.760/13.721。Prefill indexer按CUDA事件计时整次调用（3轮、每轮随机buffer位置；多chunk调用中torch.profiler会丢kernel记录，不用于此处）：上下文末尾16384行每层约46.1ms，前缀245760时约9.43ms。Attention同样按事件计时整次调用（上下文末尾16384行、每层，H12/H6/H3，GPU1，与冻结的同步提交同进程交替）：每32行共享448块、另各选最近64块时全部走union，为3.46/1.89/1.14ms（同步提交4.28/2.50/1.55ms）；均匀分散的选择全部走direct，为10.56/10.36/10.06ms（同步提交11.89/11.48/10.66ms）。下面slot表测量中同样两种选择用连续K/V调用的一列（GPU5）为3.40/1.90/1.13ms与10.21–10.34/10.20/10.08–10.14ms，与重排前同协议相差−3.7%～+0.0%。

K/V按slot表原地读取（`attention(..., kv_slots=...)`，见第4节）：SGLang的K/V在按页分配的token池里，此前每层先用`index_select`把整段上下文拷成连续K/V再调用attention。下表为2026-10-30按当前源码在GPU5上测，上下文末尾16384行的chunk、每层，同一输入三种调用交替测（3轮、顺序正反交替、每轮3次，CUDA事件计时整次调用，中位数），输出逐bit相同：拷贝 = SGLang原做法（两次`index_select`加attention），原地 = 传池和slot表。池为上下文的2倍、64-token页，页序连续（如服务刚启动时一个长请求）或随机打乱；其余行填NaN。表中为三次独立运行的中位数，单位微秒，正数表示原地更快，H12/H6/H3（TP2/4/8单卡形状）：

| 上下文 | 共享选择（union） | 分散选择（direct），页序连续 | 分散选择（direct），页序随机 |
|---:|---|---|---|
| 32768 | 59–67 / 41–46 / 37–39 | 36 / 55 / 18 | 42 / 6 / 41 |
| 65536 | 118–130 / 83–104 / 65–66 | 57 / 56 / 47 | −4 / 33 / 34 |
| 131072 | 148–176 / 87–216 / 155–267 | 168 / 133 / 129 | 162 / 150 / 119 |
| 262144 | 336–462 / 321–350 / 310–330 | 326 / 253 / 288 | 341 / 300 / 312 |
| 524288 | 671–709 / 670–698 / 654–707 | 791 / 671 / 638 | 907 / 673 / 607 |
| 1048576 | 1351–1438 / 1323–1445 / 1336–1447 | 1467 / 1477 / 1393 | 1717 / 1704 / 1698 |

池超过4GiB时（4.5GiB的池，上下文的页随机散在整个池里，同样的协议）：

| 上下文 | 共享选择（union） | 分散选择（direct） |
|---:|---|---|
| 131072 | 179 / 169 / 157 | 143 / 146 / 144 |
| 262144 | 348 / 350 / 333 | 333 / 295 / 292 |
| 1048576 | 1426 / 1409 / 1419 | 1001 / 1041 / 579 |

原地读省掉了拷贝（1M时每层约1.4ms和1GiB临时显存）。上下文不超过行数的32倍（16384行的chunk即512k以内）时direct先按slot表打包整段上下文，与读连续K/V相当；1M时raw direct直接按池地址读，与读连续K/V相差−0.9%～+3.1%。池超过4GiB时raw direct改用64位global读，1M分散选择比读连续K/V慢0.3%（H6）、2.0%（H12）、9.5%（H3），仍比拷贝快；union和打包与4GiB以内的池相同。64k以内两种方式都打包，差值在测量波动内。同一组内三种调用交替测量，组间受buffer位置影响，只比较同一行内的数。

以上为当前版本绝对性能，不包含旧新对照；attention每轮3个case、18行kernel结果、2304条整体raw，indexer为prefill 1个case、decode 2个case，共10行kernel结果、384条整体raw，均未筛除慢样本。记录索引保留在[源码readme](../../src/pyhip/ops/qsa/flydsl/readme.md)，复现只需本页已跟踪入口。

公开入口的CPU下发时间（GPU保持忙，60次中位数，TP2/4/8相近）：attention建union计划时约95–99µs，不建计划时约58–60µs；prefill_indexer在M12000和3请求时都约201–202µs。图重放不含这部分；eager prefill中GPU时间短于它的调用（如2051行完整前缀、64行长prefix）受host限制。

## 3. 最终服务性能对比

2026-10-30，PyHIP为提交`14592cb`（attention同步AITER #6576）的下一个提交（重新应用按slot表直读K/V、4GiB以上的池、按行数打包和窗口版K2；测量时是与之相同的工作区），SGLang为HEAD `4475222`之上的工作区改动（带prefix的chunk把该层的K/V池和slot表交给PyHIP），AITER `2544b2927`（本机按当前源码重建），FlyDSL 0.4.0，gfx942/MI308X，模型Qwen3.8-Flash-Next-PTPC-FP8。原生与PyHIP在相同TP及配置下测量C1/C2/C4/C8，共16场、512请求；每场32请求、名义12000输入/350输出、服务seed 42，计时关闭数值校验和profiler。源码哈希（与本页数值验收的服务相同）、请求量和配置一致性检查均通过。

服务配置为chunked prefill 16384、最大运行请求32、decode graph最大batch 32、关闭radix cache、AITER attention/MoE backend。两组的实际`mem_fraction_static`一致：TP2为0.8075、TP4为0.7225；服务和客户端沿用4个CPU线程。“原生”与“PyHIP”是同一SGLang服务只改`SGLANG_USE_PYHIP_QSA`（0/1）：原生组在12个QSA层用SGLang自带的indexer（prefill和decode选块）与prefill稀疏attention，PyHIP组把这三部分换成PyHIP的kernel。decode阶段的稀疏attention、indexer的投影GEMM以及模型其余部分（GEMM、MoE、线性attention层等）两组相同；两组也都用launcher默认的PyHIP GR read（`SGLANG_GR_READ_FLYDSL=1`），所以原生组并非完全不含PyHIP。AITER的BF16 GEMM调优表固定为09-25快照（`AITER_CONFIG_GEMM_BF16`，原因见第5节），其余调优表用AITER默认；`SGLANG_PLUGINS=none`。

| TP | 并发 | 原生token/s | PyHIP token/s | 吞吐变化 | TTFT中位数(ms，原生/PyHIP) | ITL中位数(ms，原生/PyHIP) | TTFT p99(ms，原生/PyHIP) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2 | 1 | 84.512 | 113.384 | +34.16% | 950.872 / 787.858 | 9.086 / 6.500 | 986.961 / 809.017 |
| 2 | 2 | 122.868 | 162.060 | +31.90% | 1856.319 / 1290.308 | 10.921 / 7.616 | 1886.102 / 1576.600 |
| 2 | 4 | 167.511 | 228.931 | +36.67% | 3585.215 / 2485.449 | 13.191 / 8.716 | 3762.511 / 3070.482 |
| 2 | 8 | 203.707 | 285.806 | +40.30% | 6001.831 / 4049.230 | 17.831 / 10.636 | 7527.835 / 6078.533 |
| 4 | 1 | 95.050 | 132.866 | +39.79% | 736.990 / 569.227 | 8.431 / 5.915 | 751.010 / 576.042 |
| 4 | 2 | 139.644 | 200.704 | +43.73% | 1439.870 / 1096.675 | 10.148 / 6.811 | 1493.548 / 1116.888 |
| 4 | 4 | 192.887 | 288.675 | +49.66% | 2831.525 / 1824.651 | 12.137 / 7.583 | 2976.432 / 2214.936 |
| 4 | 8 | 242.649 | 368.849 | +52.01% | 4653.291 / 2982.716 | 16.240 / 8.916 | 5872.715 / 4436.445 |

每个服务就绪后先做smoke和11888/12000两次预热，再用同样参数跑一轮不计时的8请求C8，然后在同一服务内依次测C1、C2、C4、C8。两种实现顺序运行，不是交错同址或饱和吞吐测试；整轮仅在入口和全部服务退出后各检查一次GPU状态，不证明采样期间持续隔离。本轮TP2用GPU2、3，TP4用GPU2–5（GPU6、7当时被其它进程占用）。C4、C8的TTFT主要由排队决定，分布成簇，中位数可能落在两簇之间，不宜单独比较。默认`SGLANG_USE_PYHIP_QSA=0`保持不变。

表中TP2 C2的PyHIP（162.1 token/s）比同配置的其它测量低约2.7%：之后在同样的GPU2、3上把TP2两组各重测一遍（期间没有其它测量），PyHIP为114.595/166.693/229.428/286.176 token/s（C1–C8），相对原生+35.2%/+34.9%/+37.4%/+41.7%，C2为166.7；同一负载C2的两个prefill chunk（两请求12000+4384行；带4384 token prefix的7616行，经slot表读池）单卡逐例计时也没有变慢，所以是服务测量的波动。

PyHIP attention的kernel变体在第一次用到时编译（见“目录和依赖”末段）：decode的kernel在SGLang启动时的CUDA graph capture中编译；SGLang自带的预热请求很短，只编译不建计划的变体，12k prefill和带prefix的chunk用到的变体在第一个这样的请求中编译。本轮FlyDSL缓存从空开始：数值验收服务的第一条11888 token预热请求在TP2/TP4用了17.1/12.8秒（缓存命中时约1.1秒）；性能服务复用验收服务编好的缓存（验收负载含24000/100000 token请求和32并发，覆盖带prefix的chunk），Triton等缓存复制自10-30上一轮服务。16场测量中服务日志没有Triton编译或AITER构建，计时窗口内FlyDSL/Triton缓存没有新写入。缓存为空时，SGLang原生kernel（如apply_interleaved_rope_kernel、_sparse_gqa_chunk_prefill）和AITER MoE也会在服务中按新形状编译：10-02的一次冷缓存测量中原生TP2在C4/C8各编译19/12次（每rank合计6.0/3.4秒），吞吐低约5%–6%。新环境应先用代表性负载预热，或保留编译缓存。

数值验收（TEST=1，2026-10-30按当前源码）TP2/TP4/TP8各36请求全部通过：12000与24000 token各一条、32条并发、1条9 token短请求和1条100000 token请求（chunk 16384，最后一个chunk带98304 token的prefix，attention经slot表直接读K/V池）。TP2/TP4用默认配置，每层K池2.49/3.30GiB；TP8把`MEM_FRACTION_STATIC`设为0.93（实际0.7905），每层K池9437696行、4.50GiB，验证池超过4GiB时的原地读（100000 token请求的最后一个chunk为1696行、上下文超过行数的32倍，raw direct用WIDE变体）。每个rank的12个QSA层都有带prefix的chunk；每个rank完成attention校验444次，每层prefill indexer校验37次、graph decode校验293次；日志无异常，没有设备端断言。这不是模型质量或生成文本bitexact验收。原始证据与过程由[源码readme](../../src/pyhip/ops/qsa/flydsl/readme.md)索引，本页命令不依赖本地未提交研究脚本或报告。

### MTP（EAGLE 3/1/4）

2026-10-08测，早于上表，环境不同：当时的AITER尚未重建、四类AITER调优表（a8w8 bpreshuffle、a8w8 blockscale bpreshuffle、BF16 GEMM、FMoE）都固定为09-25快照，FlyDSL 0.3.2，launcher还没有PyHIP GR read；绝对值不能与上表直接比较。模型、服务配置和32请求、名义12000输入/350输出的协议与上表相同（实际`mem_fraction_static`仍为TP2 0.8075、TP4 0.7225），SGLang另加`--speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4`（QSA要求topk为1、草稿token不超过4）。两组都开MTP；PyHIP组的indexer在prefill和CUDA graph TARGET_VERIFY中由PyHIP计算，verify的attention、draft模型和draft extend仍是SGLang原生。SGLang的pending ring为每个请求2×ratio个槽：verify窗口跨过压缩边界时，不会再覆盖该组在窗口之前的成员。PyHIP为HEAD `cd07379`之上的工作区改动，SGLang为HEAD `c3c4bc8`之上的工作区改动。

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

16场测量中都没有服务中的Triton编译。C8的TTFT主要由排队决定（解码更快时请求到达更密，与prefill交错更多），不宜单独比较。C1 profile中每个verify graph的indexer由约5.6ms降到0.35ms，verify graph在TP2由21.56ms降到16.18ms、TP4由19.74ms降到14.21ms（profiler给每个kernel多算约1µs）。与同一环境下10-04测的非MTP PyHIP相比，MTP+PyHIP的吞吐在TP2高10%–34%、TP4高15%–41%，并发越高收益越小。

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

约束：Q/O为连续BF16 `[M,H,256]`，K/V为连续BF16 `[N,HK,256]`，`H/HK≤16`，16B对齐、inference-only；QSA不接受5D KV。Q和每个请求的K（V同）的字节数都不超过4GiB−1MiB（HK=1时一个请求约838万token），整批packed K/V不受此限。`indices`是连续int32 `[M,4T+3]`，T为256/512/768/1024（由宽度决定，indexer输出的2051宽即T=512）：因果窗口有v个token的行，前`4·min(v//4, T)`列恰为min(v//4, T)个不同的完整四token块（块内4个token依次写出，块可乱序），接v%4个因果尾token，其余填−1；ID为请求内逻辑token。选择不校验：块数不对、缺尾token或块重复时结果错误，不报错。K/V须为有限值。`query_lens`/`prefix_lens`是host长度，packed请求的K/V按请求拼接；默认单请求prefix=`N-M`。默认scale=1/16，`scale·log2(e)`须是有限的正FP32；`out`不与输入重叠。

K/V在按页分配的token池里时，可传`kv_slots`原地读取，不必先拷成连续K/V：`kv_slots`是同设备连续int32 `[N]`，第i项是packed上下文第i个token（请求按顺序拼接）在池中的行号；K/V为池`[P,HK,256]`，大小不限，池超过4GiB−1MiB时raw direct改用64位global读（比buffer读慢约0–10%）。每个请求的每个四token块（请求内token 4j..4j+3，含请求末尾不满4个token的块）须占池中连续4行且起始行是4的倍数，页大小为4的倍数的分页池满足这一点；不满块多出的行值任意，slot值不校验。结果与先按`kv_slots`拷出连续K/V再调用逐bit相同。SGLang的chunked prefill对带prefix的chunk这样调用。

```python
# 接上例：把K/V放进64-token页的池，页序打乱
pool_rows = 4 * 64
pages = torch.randperm(pool_rows // 64 - 1, device=device)[: (m + 63) // 64] + 1
slots = (pages[:, None] * 64 + torch.arange(64, device=device)).flatten()[:m].int().contiguous()
k_pool = torch.zeros(pool_rows, hk, d, device=device, dtype=torch.bfloat16)
v_pool = torch.zeros_like(k_pool)
k_pool[slots.long()], v_pool[slots.long()] = k, v
in_place = attention(q, k_pool, v_pool, indices, query_lens=(m,), prefix_lens=(0,), kv_slots=slots)
assert torch.equal(in_place, out)
```

### Indexer接口

[prefill_indexer](../../src/pyhip/ops/qsa/flydsl/indexer.py#L338)和[decode_indexer](../../src/pyhip/ops/qsa/flydsl/indexer.py#L417)都从调用方`index_qk_proj`之后的`qk`开始，返回同一种token选择，可原样交给attention。除`qk`外都是关键字参数，两个入口中同名参数含义相同；只依赖PyHIP，不需要SGLang/AITER。

共同约定：

- 只支持gfx942。Indexer固定4个Q头、1个K头、D=128、压缩比4、每行选512个压缩块。**indexer的D128投影与attention的D256是不同投影**，compressed池也不是attention的KV缓存。
- 所有tensor在同一GPU，除表中注明外必须连续。`key_state`、`rope_state`、`compressed`原地更新。
- 槽号全部由调用方给出。SGLang的pending ring每请求8个槽（压缩比的2倍）：请求r（从1开始）的位置p在槽`8r + p%8`；请求0从不分配，`[0,8)`是dump行。首token物理槽为s的4-token组，压缩key在`compressed[s//4]`；slot 0是惰性写入位置。dump行和slot 0会被并发写入，不能当数据读。
- 返回int32 `[rows,2051]`：每行先是选中的压缩块，按块号升序、每块展开成4个token（最多2048个）；紧接着是0–3个因果尾token（最后一个完整组之后、到query本身为止）；其余填−1。Token ID是请求内逻辑位置。位置p的query可见`(p+1)//4`个压缩key，不超过512个时全选；分数相同时取块号小的。
- 第一次调用时编译（缓存为空时prefill约6秒、decode约6.5秒），之后任何长度组合都不再编译。

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
- 单请求最多262144个压缩key（1M token）；宽度超过65536个key的logits行用32位块号的top-k变体，两种变体在第一次调用时一起编译。logits临时缓冲按512MiB分块，长行时每块至少1024行（1M末尾约1.1GiB）。
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
| `page_table` | int32 `[B,P]`，P ≤ 16384 | 本行第p页对应压缩key `16·page_table[row,p] + [0,16)`；logits宽16·P，`B·16·P`个FP32须小于2GiB |
| `lengths` | int32 `[B]` | 本行可见的压缩key数，= `seq_lens // 4` |
| `verify` | bool | False：各行必须属于不同请求；True：MTP的TARGET_VERIFY窗口，一个请求占连续W（≤4）行 |
| `q_out` | 可选，BF16 `[B,4,128]` | 写入归一化并加RoPE后的index Q |
| `logits_out` | 可选，FP32 `[B,16·P]` | 写入logits；其存储须在末尾多留512个值（top-k按512个值对齐读取） |

注意事项：

- 所有metadata都在设备上，可在CUDA graph中capture。必须先eager调用一次以完成编译（第一次`verify=True`另编译verify用的两个kernel），未预热就在capture中调用会报错。重放前原地更新各张量的内容，地址不能变。
- 页表宽度超过32768个压缩key（131072 token）时，top-k另加两个launch，把实际长度也超过它的行分给多个CTA统计直方图、收集候选；所需统计与临时缓冲在调用内分配，可以capture。
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

以下直接使用SGLang已提交的launcher和Python benchmark模块。先在一个终端启动服务；TP_SIZE取2或4，SGLANG_USE_PYHIP_QSA取0或1。四种配置依次运行，不并行占用相同GPU。两种实现使用相同模型、seed、预热、请求长度及内存配置。launcher不再固定设备，用`CUDA_VISIBLE_DEVICES`/`HIP_VISIBLE_DEVICES`选卡；其默认模型已改为PLE-BF16版本，第3节的表用`MODEL_PATH`指定PTPC-FP8。

本机当前环境另需两处设置（原因见[源码readme](../../src/pyhip/ops/qsa/flydsl/readme.md)第5节10-29条目）：AITER `2544b2927`的`model_configs/qwen38_flash_next_bf16_tuned_gemm.csv`给BF16 GEMM的hipBLASLt解在本机会选到FP8 kernel，启动预热时GPU访问错误，所以`AITER_CONFIG_GEMM_BF16`指向此前各轮用的09-25调优表；venv里登记的`pyhip_monokernel`插件已无模块，`SGLANG_PLUGINS=none`跳过它（不设只多打印traceback）。

```bash
cd /opt/sglang
PATH="/opt/lc/pyhip/.venv/bin:$PATH" TP_SIZE=2 CUDA_VISIBLE_DEVICES=0,1 HIP_VISIBLE_DEVICES=0,1 \
  MODEL_PATH=/models/Qwen3.8-Flash-Next-PTPC-FP8 SERVED_MODEL_NAME=Qwen/Qwen3.8-Flash-Next-PTPC-FP8 \
  HOST=127.0.0.1 PORT=9080 SGLANG_USE_PYHIP_QSA=1 SGLANG_TEST_PYHIP_QSA=0 SGLANG_PLUGINS=none \
  AITER_CONFIG_GEMM_BF16=/opt/lc/pyhip/mytest/sglang_tp2_base_20260925_01/aiter_configs_snapshot/bf16_tuned_gemm.csv \
  LOG_FILE=/opt/lc/pyhip/mytest/mydata/qsa_serving_new/tp2_pyhip/server.log \
  bash scripts/launch_qwen38_flash_next_fp8_mi308x_pure_tp_4_or_8_or_2.sh --random-seed 42
```

MTP对照在上面的launcher命令后追加`--speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4`，其余不变。

服务就绪并完成统一预热后（第3节的表另用同样参数先跑一轮不计时的8请求C8），在另一终端执行。替换输出目录标识以匹配当前TP/实现；每种配置依次测C1/C2/C4/C8：

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