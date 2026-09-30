# QSA：gfx942 多-query、多-head 稀疏 attention

将 SGLang QSA 实验包的输入契约、两个 prefill 用例、冻结 baseline 和 FP32 reference 迁入 PyHIP；
当前采用 **dense / block-native direct / local union 三路精确分流**，attention 计算均为 FlyDSL。
Triton 只用于 union 建表和可选 block 排序；未使用的旧 fallback/wave 实验已从当前源码删除，历史快照保持不动。
仅针对 **gfx942、BF16、D=256、压缩比 4、block Top-K=512 + causal tail**；不是生产服务接入。

主验收形状已改为 **TP2（Q12/KV1）和 TP4（Q6/KV1）**，TP8（Q3/KV1）保留功能覆盖，不设性能门槛。
**约 200 TFLOPS 目标尚未达到。** 当前 independent 完整建表+run 对比同场 baseline 加速约
1.25–1.48 倍；shared 对照加速约4.22–7.42倍。shared 是合成高重合输入，不代表真实模型或随机选择。
清理前完整238项正确性测试通过；清理后64项CPU、18项direct/graph回归通过，TP2/4两个主形状direct特化ELF与原计时产物逐字一致。
本次只做机械清理、验证和文档交接，没有重跑性能；正式八场历史raw、源码、ELF、资源和门禁仍可独立审计。
全部优化尝试、被拒绝的实验和修复按时间追加在 [opt.md](opt.md)，旧 TP1 数据仅为历史记录。
**迁移前先读 [opt.md](opt.md) 最后的“迁移交接与清理状态”章节：当前状态、必需文件、外部依赖、审计限制和下一轮任务均集中在那里。**

## 1. 用例、模型与输入契约

模型字段见 [model_config.json](model_config.json)，来源为
[Qwen/Qwen3.8-Flash-Next 固定配置](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/34567a4712bc9766c4449e2e98e4468bfa24d915/config.json)，
revision 为 `34567a4712bc9766c4449e2e98e4468bfa24d915`；不下载权重、不运行 indexer。
最终 attention 是 Q24/KV2/D256；索引器的 D128 不是这里的 head dimension。

| 主用例 | `no_prefix` | `chunk_prefill` |
| --- | --- | --- |
| 新 query 数 M | 12000 | 12000 |
| 缓存 prefix 数 P | 0 | 12000 |
| 完整 KV 长度 L=P+M | 12000 | 24000 |
| TP2 Q / output，BF16 contiguous | `[12000,12,256]` | `[12000,12,256]` |
| TP4 Q / output，BF16 contiguous | `[12000,6,256]` | `[12000,6,256]` |
| K、V 各自形状，TP2/4/8 | `[12000,1,256]` | `[24000,1,256]` |
| query 请求内绝对位置 | `0..11999` | `12000..23999` |

12000 是十进制 token 数；P=12000 是实验参数，不是模型固定值。
TP1/2/4/8 的本地 `(Hq,Hkv,G=Hq/Hkv)` 分别为 `(24,2,12)`、`(12,1,12)`、`(6,1,6)`、`(3,1,3)`；
只在单 GPU 模拟本地头数，不含 TP/DCP 通信；TP2/4为主要性能矩阵，TP8正确性覆盖，TP1少量兼容测试保留。

- `indices[M,2051]`、`block_indices[M,512]` 为 int32，所有 heads 共享选择；有效前缀后填 `-1`。
- 对 query 位置 p，完整块数 `C=(p+1)//4`；选择 `min(C,512)` 个不同块 b，展开 `4*b+[0,1,2,3]`，再追加 `[4*C,p]` 的 0–3 个尾 token。
- 块可以无序，但不得重复、越界或指向未来；真实 query 至少有一个有效 token，允许零 query 的请求段。
- `cu_q/cu_k` 分别是 packed Q/KV 的 token 起点；prefix-only 请求也必须推进 `cu_k`。
- `query_positions` 是请求内位置，`query_sequence_ids` 是请求编号；token t 访问 `K/V[cu_k[s]+t,hkv,:]`，不是物理 cache slot。
- 对 Q head h，`hkv=h//G`；每个 `(query,head)` 独立 softmax，不能让所有行无条件 attend 整个并集。
- `scale=1/sqrt(256)=0.0625`；精确指保留选择集合，不承诺不同浮点实现逐位相同。

`independent` 默认 seed=17，每行独立无放回随机选块；`shared` 每组共享随机优先级但保留各自因果边界；
`recent` 选择最近完整块。三者均为合成负载，BF16 Q/K/V 也是随机值，不是模型 Top-K/激活或模型精度验证。
基准 `--selection-group` 固定默认32，与 `--query-tile` **解耦**；扫BQ不再改变输入。
输入生成函数的 `selection_group` 默认仍是8，基准显式传32并记录两份索引的SHA-256。

## 2. 实现与调度

[kernel.py](kernel.py) 基于 [../mha/mha_pa_bf16_256_linear_942.py](../mha/mha_pa_bf16_256_linear_942.py)
的 LDS、QK/PV MFMA 和 staggered pipeline，并复用相邻 MHA helpers；直接消费普通线性 Q/K/V/O，无额外 KV 转换。

### Dense：可见上下文不超过2051

[dense.py](dense.py) 处理每请求开头 `min(M,max(0,2051-P))` 个query。
这些行的Top-K覆盖全部可见token，直接dense causal，不读取indices、不建union。
每请求使用原Q/O的前t行及原K/V的前P+t行，保持bottom-right因果偏移，零复制。
KV长度是64的倍数时复用相邻native实现，否则使用本地full-VOFFSET有界DMA变体。
默认no_prefix覆盖前2051行；P12000场景没有dense行。`dense_limit=0`可禁用。

### Direct：低重合度直接读取原始块选择

[direct.py](direct.py) 每CTA四个wave，各处理一个query及其GQA heads。
直接读取512个块编号，四-token packet内复用block地址，再单独处理0..3个causal尾token。
支持BN32/64，默认32；Q/K在寄存器，V按PV需要的布局直接global加载并在寄存器做BF16转置，
只有output shuffle占32KiB LDS。K和V分别有受控预取、明确wait和compiler fence。
默认先将各query块编号排序以改善访问局部性，排序成本计入rebuild；原两份输入索引不改写。
`sort_blocks=False`可禁用排序。全部query归union的direct CTA通过uniform gate在计算前退出。

### Local union：仅为一个M128计算tile共享K/V

1. 先排除dense行，再按请求原始query边界分组。effective BQ为requested BQ与不超过`128/G`的最大2幂的较小值；默认TP2=8、TP4=16、TP8=32。
2. dense切割后的首组可以不满BQ，后续分组不整体平移；不跨请求、不根据synthetic selection标签改算法。
3. GPU清零dense membership、atomic OR写入完整块及尾块，先计算cost gate；只有active组做compact前缀扫描与写表。
4. 全体query共有且最早query已经完整可见的块排在前面，完整64-token common tile免mask；其余预计算逐query score masks。
5. `[BQ,G]`展平到M轴，每份局部并集只服务一个M128/BN64计算tile；不再让三个M tile遍历同一32-query总并集。
6. 每CTA512线程/8waves、64KiB LDS，BF16 MFMA/FP32 online softmax；persistent grid上限为CU数×multiplier，默认2。

设 tile 内实际 query 数为 r、并集块数为 U，每行所需块数（含尾块）为 $s_i$，GPU 判定：

$$\rho=\frac{U\,r}{\sum_i s_i},\qquad \rho\le1.5\Rightarrow\text{union active}.$$

这是块级工作膨胀启发式，不是实测 HBM 流量或精确 token 成本。
默认 `--algorithm auto`：dense先写其覆盖行；active稀疏组走union，其余走direct。
各路径写入区域互斥，不在CPU读取active总数，也不改变selection集合。
`--algorithm direct` 完全跳过union建表，但仍可使用dense和可选块排序。
`--algorithm union` 对非dense行强制局部并集，不运行direct。
旧 fallback 和 wave launcher 已删除；direct 使用的三个算术 helper 已原样移入 [direct.py](direct.py)，无遗留运行时导入。
仅在历史结果的冻结源码中保留旧实现，用于复现过去实验；当前CLI不提供wave模式。

## 3. API、计划生命周期和单位

输入定义见 [contract.py](contract.py)，入口见 [implementation.py](implementation.py)，GPU 建表见 [plan.py](plan.py)。

- `prepare(inputs=..., mode="auto", query_tile=32, grid_multiplier=2, max_union_inflation=1.5, dense_limit=2051, block_n=32, sort_blocks=True) -> DispatchPlan`：构造dense/direct/union计划并初始化动态表。
- `DispatchPlan` 包含 `dense: DensePlan`、可空的 `direct: DirectPlan`、可空的 `union: SparsePlan` 和 `mode`；没有稀疏行时两种稀疏计划均为空。
- `run(inputs=..., prepared=plan, out=...) -> None`：写 caller 预分配的 contiguous output，shape/dtype/device 与 Q 一致；每次取得当前 PyTorch stream。
- `implementation.rebuild_plan(inputs=..., plan=...) -> None`：在输入GPU上重建union/gate，再仅排序真正走direct的行；不分配静态metadata。direct且sort=false时为无动态工作。
- 改 Q/K/V 值但选择不变时可复用计划；选择改变时须同步更新 `indices` 与 `block_indices` 并 rebuild，不能复用过期 membership。
- 改请求长度、prefix、位置布局、形状、设备或 BQ 时重新 prepare；metadata 内含旧位置，rebuild 不是任意变长布局的重建接口。
- Graph 前在足量真实 buffers 上 prepare 并 warm run（含所有所需特化）；capture 内使用 rebuild + run，保持地址/形状固定，replay 可读取原位更新后的选择。
- Graph replay已验证shared→independent→shared gate翻转和direct排序重建。图外同布局替换inputs时，sort=false读取当前block tensor，不沿用旧引用。
- out不得与Q/K/V重叠，即使没有dense行也检查。全部物理读取边界按byte验证；padding wave只能查询当前tile有效query的gate。
- 首次 FlyDSL compile 会真实 launch；未 warm 的特化在 capture 中报错。编译缓存缓存代码，不缓存前次 attention 结果或索引值。

以下 device 表均为 int32；T 为 query tile 数，CAP 为按 16 块向上对齐的容量，只能读取 counts 指定的有效前缀。

| 字段 | 形状 / 单位与语义 |
| --- | --- |
| `metadata` | `[T,5] = [q_first,q_rows,kv_first,kv_len,position0]`；起点/长度都是 **token**，`q_first/kv_first` 分别相对 packed Q/KV，`position0` 相对请求 |
| `dense_membership` | `[T,max_blocks]`，请求内 block-id 寻址的 query 位图；`max_blocks=ceil(max_seqlen_k/4)` |
| `blocks`, `membership` | `[T,CAP]`，请求内 **4-token 块编号**与位图；bit j 对应 tile 的第 j 个 query，BQ32 须按 unsigned 32-bit 解释 |
| `counts` | `[T,2]`：第0项为并集**块数**U，第1项为免mask前缀的64-token tile数；inactive组第二项为0，compact内容不读取 |
| `score_masks` | `[T,CAP/16,BQ,4]`，每个 int32 的低 16 bits 对应一个 lane quarter 的 score 保留位；common 前缀不读取 masks |
| `active`, `query_tiles` | `[T]` 的union gate（0/1）；`[M]` query→tile映射供direct使用，dense行是-1且不得据此解引用active |

DirectPlan另外保存四-query静态metadata、BN、可选sorted block buffer，复用union的gate与query映射。
DensePlan保存每请求的eligible query数、原packed起点及两元素CU表；run不执行`.item()`。

`group_padded` 当前等于 G，并未对 G 补齐。raw-buffer descriptor extent、VOFFSET/SOFFSET 和 LDS 地址均以 **byte** 计；
typed BF16 pointer 的加法以元素计（每元素 2 bytes），不可混用 token、block、element 和 byte 单位。

## 4. 正确性修复与测试节点

当前验证包含以下关键边界：

- **非对齐 KV 的 raw-buffer 越界**：原 DMA 将块起点放在 `SOFFSET`，但 gfx942 raw-buffer 范围检查不包含它；物理末块不足 4 token 时可能读入 guard/邻接数据，`0 × NaN V` 仍为 NaN。
  [kernel.py](kernel.py) 在任一请求 `(P+M)%4 != 0` 时选择 `_bounded_dma`，把完整块/packet/channel 字节偏移放入 `VOFFSET`，`SOFFSET=0`，用请求/head extent 截断；drain 无效 token 零填充。所有长度对齐时保留原快路径。
- **FP32缩放精度**：新direct在FP32 QK累积后应用scale，抵消构造验证不因提前BF16舍入破坏结果；冻结baseline不改写。
- **分流与生命周期**：输出alias拒绝、补齐wave gate边界、sort=false替换输入、非当前GPU重建、graph中active切换和排序结果重建。

清理前的完整记录为 **238 passed、0 skipped、0 failed**，
见 [results/tp24_v12_final_correctness.xml](results/tp24_v12_final_correctness.xml)。
清理后运行64项CPU和18项direct/graph定向回归；未重跑完整矩阵或性能，不能把旧JUnit称为新的验收结果。
所有输出检查 shape/dtype/contiguity 与 finite；候选和 baseline 全输出比较，另对 FP32 oracle 做边界加均匀抽样（ragged 全行）。
统一 `rtol=0.02, atol=0.02`；relative L2 是补充指标，不能替代逐元素门槛。

[test_qsa.py](test_qsa.py) 的节点清单：

| 节点 | 覆盖 |
| --- | --- |
| `test_frozen_baseline_snapshot_cpu` | 两个冻结函数含装饰器的 SHA-256 |
| `test_default_model_contract_cpu` | 模型字段、M/P、默认选择与 TP |
| `test_causal_block_and_tail_contract_cpu`, `test_ragged_prefix_contract_cpu` | 4-token 边界、2048/2051 附近、prefix-only 请求、packed offsets |
| `test_invalid_causal_contract_rejected_cpu`, `test_reference_masks_padding_and_ragged_prefix_cpu` | future/padding/重复块/错误映射拒绝；可解析 FP32 参考 |
| `test_implementation_matches_baseline_and_fp32` | TP2/4主矩阵，两个prefix场景、三种selection、ragged/M257/M12000；另有TP8大形状功能和TP1兼容节点 |
| `test_zero_prefix_chunk_matches_baseline_prefill` | baseline 的 P=0 chunk 与普通 prefill 一致 |
| `test_unaligned_union_nan_guards_and_all_masked_tiles` | BQ8/10/32 强制 union、L=65、NaN K/V guards、输出前后 guard、局部全 mask |
| `test_direct_cancellation_scales_after_fp32_dot` | TP2/4/8、BN32/64、强制direct，验证抵消输入 |
| `test_union_graph_rebuild_preserves_current_indices` | Graph replay 前原位替换两份索引，重建计划并校验当前选择 |
| `test_forced_union_disjoint_blocks_and_softmax_rescale` | TP2/4/8、强制union、完全不相交块、早期全mask、较大logits、重复运行逐位一致 |
| `test_dense_dispatch_boundaries_match_fp32`, `test_dense_only_writes_eligible_ragged_prefixes` | P0/2047/2050/2051/12000，原packed地址、tail poison和只写eligible行 |
| `test_auto_graph_gate_flips_rebuild_sorted_direct` | BN32/64及TP2/4/8，shared/direct/union切换时重建排序计划 |
| `test_sparse_only_rejects_output_alias`, `test_direct_unsorted_replacement_input` | 输出重叠拒绝、无排序计划不复用过期输入 |
| `test_rebuild_uses_input_device_not_ambient_device` | 输入GPU与当前GPU不同，重建正确且恢复当前设备 |

模块fixture检查已编译union/direct/dense native及bounded特化的private segment和VGPR/SGPR spill为零。
`perf`只标记TP2/4大规模**正确性**节点，pytest本身不计时；TP8无性能要求但保留大形状功能覆盖。
无可用gfx942/FlyDSL时GPU节点skip，不能视为通过GPU验证。
仓库 [../../../../pytest.ini](../../../../pytest.ini) 默认 `-m "not perf"` 且只收集 tests；本实验必须显式指定文件。
从 PyHIP 根目录运行：

```bash
PYTHONPATH=src python -m pytest experiments/attention/flydsl/qsa/test_qsa.py -q -k cpu
PYTHONPATH=src python -m pytest experiments/attention/flydsl/qsa/test_qsa.py -q -m 'not perf'
PYTHONPATH=src python -m pytest experiments/attention/flydsl/qsa/test_qsa.py -q -m perf
PYTHONPATH=src python -m pytest experiments/attention/flydsl/qsa/test_qsa.py -q -m 'perf or not perf'
```

## 5. 基准口径、门禁与命令

使用已有匹配 ROCm 的 PyTorch/Triton、FlyDSL、NumPy、msgspec、pytest；不依赖 SGLang runtime、模型权重或外部工作区的绝对导入。
迁入的 QSA 模块使用 package-relative imports，不从原 SGLang checkout 动态导入；仍须保留 PyHIP 的相邻 MHA helpers、计时器和只读硬件查询，不能只复制本目录就运行。
使用物理 `--gpu`，不得设置 `ROCR_VISIBLE_DEVICES`、`HIP_VISIBLE_DEVICES`、`CUDA_VISIBLE_DEVICES` 或其他 device remap；CLI 拒绝前三种非空变量，并核对 runtime PCI BDF。

- [bench.py](bench.py) 在入口、准备/预热完成后采样前、结束时分别读取门禁：**GPU use≤5%、VRAM≤20%、PTL Enabled / VECTOR,F8**。
- 只查询状态，**不写 PTL、频率、功率、NUMA 或其他 GPU 调优控制**；任一门禁失败保存已有 raw/errors 并停止，不轮询等空闲、不重采挑快段。
- 复用 [../../../../src/pyhip/testing/misc.py](../../../../src/pyhip/testing/misc.py) 的原 `cudaPerf`；默认 10 组独立 allocation（相同输入值）、每组每 scope warmup 2 次、每 scope 10 samples，轮换 buffer、逐 sample 正反交错 scope 顺序。
- 保存全部 event us、sample/buffer 编号、含计时器前导的 wall us；汇总取全部样本中位数。记录地址、storage offset、mod256/mod4096，性能输出从 allocation 起点开始。
- 每组先对 baseline/FP32 校验并审核精确 membership；采样后再次检查输出和计划。失败不得因性能较好而忽略。

| scope | 实际计时范围 |
| --- | --- |
| `candidate_run` | 预建计划的run：dense/direct/union FlyDSL分支；包含各branch发射和空gate成本 |
| `plan_and_run` | 动态GPU建表和direct排序 + run；预分配scratch，不含静态分配或host metadata构造 |
| `baseline_run` | caller 预分配 output 的原单-query Triton baseline；没有 union 建表 |

旧报告的`flydsl_run`字段是历史口径，新报告使用`candidate_run`，不能混用不同版本结果。
direct模式没有union建表，但sort=true仍需排序，相关成本不能从完整调用中扣掉。
三者均不含 JIT、输入生成/拷贝、indexer/Top-K、block-to-token 展开、分页 KV gather/write、QKV/O projection、服务调度或多卡通信；不是 TTFT/整模型吞吐。
baseline 适配还用 host `max_seqlen_q` 避免 GPU `.max().item()`，保留原 H20/non-H20 launch tables，未声称为 ROCm 调优；仅暴露共享二维 indices 和连续 BF16 输入。

有效工作量只按原始有效选择计数（QK 与 PV 两次乘加），不奖励并集膨胀、M/BN padding 或 softmax 附加操作：

$$F_{\mathrm{useful}}=\sum_{i,j}[\mathrm{indices}_{i,j}\ge0]\times4\times H_q\times256,\qquad
\mathrm{TFLOPS}_{\mathrm{effective}}=F_{\mathrm{useful}}/(t_{\mu s}\times10^6).$$

TP2两用例有效FLOPs为 **276416102400 / 302211072000**；TP4为 **138208051200 / 151105536000**。
TP8和baseline的目标字段为空，不设性能要求。先完成正确性再选择空闲卡；下例GPU1不是空闲保证。
输出必须是新的 case 子目录，不覆盖已冻结结果。当前容器默认的 `HIP_VISIBLE_DEVICES`
即使是完整序列也会被严格拒绝；只在子进程中移除映射并用 `--gpu` 选卡，不改变设备配置：

```bash
env -u HIP_VISIBLE_DEVICES -u ROCR_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES PYTHONPATH=src python -m experiments.attention.flydsl.qsa.bench --gpu 1 --tp-list 2 4 --case all --algorithm auto --selection independent --output experiments/attention/flydsl/qsa/results/local_independent
env -u HIP_VISIBLE_DEVICES -u ROCR_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES PYTHONPATH=src python -m experiments.attention.flydsl.qsa.bench --gpu 1 --tp-list 2 4 --case all --algorithm auto --selection shared --selection-group 32 --output experiments/attention/flydsl/qsa/results/local_shared
env -u HIP_VISIBLE_DEVICES -u ROCR_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES PYTHONPATH=src python -m experiments.attention.flydsl.qsa.bench --gpu 1 --tp-list 2 4 --case all --algorithm direct --dense-limit 0 --block-n 32 --output experiments/attention/flydsl/qsa/results/local_direct
```

单个`--attention-tp`默认2，`--tp-list 2 4`显式顺序测两种形状。
可用`--query-tokens`、`--prefix-tokens`缩小检查，`--block-n 32/64`切direct tile；
`--query-tile`不改变`--selection-group`。`--amd-smi`指定支持PTL查询的工具。
正式协议保留默认`--buffers 10 --warmup 2 --samples 10`。

## 6. 最终结果与未达标说明

以下八份报告为 **2026-09-25、三路实现**的 `auto/requested_BQ32/SG32/seed17`，TP2/4 effective BQ分别8/16；
按raw重算的中位数与summary一致。清理只改变direct中三个helper的位置和绑定，其他计算实现与冻结源一致；
源码不再整体相同，迁移复现证明和两个主形状的ELF对照见 [opt.md](opt.md)。
设备为 MI308X/gfx942、物理 GPU1、PCI `0000:80:00.0`；三个门禁均通过且无设置写入。
环境记录为 Python 3.10.12、PyTorch `2.12.0+rocm7.2.4.gitcf5ea6e.post2`、HIP `7.2.53211`；不同 selection 为分开场次。

| 最终报告 | run ms / T | plan+run ms / T | baseline ms | 完整调用加速 |
| --- | ---: | ---: | ---: | ---: |
| [TP2 independent/no_prefix](results/tp24_final_independent/no_prefix_tp2_independent_bq32/result.json) | 6.453 / 42.83 | 6.635 / 41.66 | 9.458 | 1.43x |
| [TP2 independent/chunk](results/tp24_final_independent/chunk_prefill_tp2_independent_bq32/result.json) | 8.142 / 37.12 | 8.448 / 35.77 | 10.568 | 1.25x |
| [TP4 independent/no_prefix](results/tp24_final_independent/no_prefix_tp4_independent_bq32/result.json) | 6.183 / 22.35 | 6.396 / 21.61 | 9.448 | 1.48x |
| [TP4 independent/chunk](results/tp24_final_independent/chunk_prefill_tp4_independent_bq32/result.json) | 8.110 / 18.63 | 8.392 / 18.01 | 10.573 | 1.26x |
| [TP2 shared/no_prefix](results/tp24_final_shared/no_prefix_tp2_shared_bq32/result.json) | 1.985 / 139.27 | 2.187 / 126.37 | 9.238 | 4.22x |
| [TP2 shared/chunk](results/tp24_final_shared/chunk_prefill_tp2_shared_bq32/result.json) | 2.089 / 144.66 | 2.402 / 125.82 | 10.223 | 4.26x |
| [TP4 shared/no_prefix](results/tp24_final_shared/no_prefix_tp4_shared_bq32/result.json) | 1.073 / 128.76 | 1.251 / 110.46 | 9.262 | 7.40x |
| [TP4 shared/chunk](results/tp24_final_shared/chunk_prefill_tp4_shared_bq32/result.json) | 1.106 / 136.63 | 1.372 / 110.12 | 10.187 | 7.42x |

八份报告均为`complete=true`、无errors、每scope10raw样本。**所有TP2/4 candidate目标均未达到200**；baseline目标为空。
各计时 scope 使用独立 output allocation，检查实际被计时的输出，不用其他 scope 重跑覆盖。
保存完整源码快照、相邻 MHA 依赖哈希、FlyDSL/Triton 版本、实际执行的 ELF/IR 与三阶段门禁。
independent的chunk全部走新FlyDSL direct，完整调用比同场baseline快约25–26%；shared的所有稀疏组走local union。
这些是与冻结Triton实现的比较，不代表优于所有attention库，也不是整模型TTFT收益。
[wave 无 prefix](results/wave_bq4/no_prefix_tp1_independent_bq4/result.json#L1) / [wave chunk](results/wave_bq4/chunk_prefill_tp1_independent_bq4/result.json#L1)
的探索记录为约 26.23/26.35 TFLOPS，低于同场 baseline 29.38/28.68；仅 2 buffers、6 samples，不与正式协议混用，保留但不作为默认。

未采用或失败的探索也保留：早期 GPU0/GPU1 空闲门禁失败、四阶段展开的 scratch/spill 拒绝、
初版 ELF 导出转义解析失败均不计为达标样本。初版 formal_auto 结果早于修复，不替代上表。
参考源码的 reserved-M0 编译 warning 未被屏蔽；最终被测 FlyDSL 产物无 private/spill。

## 7. 文件、来源与后续证据

| 文件 | 职责 |
| --- | --- |
| [contract.py](contract.py)、[inputs.py](inputs.py)、[reference.py](reference.py) | 迁入的契约、确定性生成/校验、独立 FP32 oracle |
| [baseline_kernels.py](baseline_kernels.py)、[baseline.py](baseline.py) | SGLang 冻结 kernel 与 allocation-free host adapter |
| [implementation.py](implementation.py)、[plan.py](plan.py) | DispatchPlan、三路调度、对齐局部并集和GPU gate |
| [dense.py](dense.py)、[direct.py](direct.py)、[kernel.py](kernel.py) | 当前FlyDSL dense、block-native direct、local union |
| [test_qsa.py](test_qsa.py)、[bench.py](bench.py) | 迁移/新增正确性节点与门禁计时、raw/IR 输出 |
| [audit.py](audit.py) | 无 GPU 重算中位数/有效 FLOPs，复核门禁、冻结源码、ELF 和资源 |
| [opt.md](opt.md) | 本轮逐版本优化日志，包含失败、保留方案、正确性修复及正式结果 |

可独立重算最终八份冻结报告，不启动GPU：

```bash
python -m experiments.attention.flydsl.qsa.audit experiments/attention/flydsl/qsa/results/tp24_final_*/*/result.json
```

`--check-current` 仍严格比较当前源码AST；本次将helper移入direct后，它对旧报告应当失败，
不能放宽检查或修改旧快照来“通过”。清理前后机械等价的复现方法在 [opt.md](opt.md)，
下一台机器产生新结果后再使用`--check-current`检查其对应源码。

SGLang 来源、文件哈希和适配差异见 [source_manifest.json](source_manifest.json)，记录的 workspace HEAD 为 `540d564c19436f28f2644e2247350da56c124452`。
这是源函数的副本，不是生产代码 relocation；迁入文件保留 SGLang Team / Apache-2.0 标识，不能以 PyHIP 的 [../../../../LICENSE](../../../../LICENSE) 覆盖其归属。
冻结校验范围是含装饰器及最终换行的函数文本，不是整个迁移文件逐字相同：

- `_sparse_gqa_prefill`：`215b956fac4755f39de59b04c60b3e867f15aa44b27cd55055423b6f29f2f7f3`。
- `_sparse_gqa_chunk_prefill`：`050a46369a586a4310b988acd3b6fa7b80369ad813874a00c8fe272c152694d6`。

当前正式产物：union VGPR251/SGPR73/LDS65536，direct BN32 VGPR214/SGPR67/LDS32768，
dense bounded VGPR230/SGPR60；所有private/spill=0。各特化真实ELF与哈希保存在对应结果目录。
产物列表来自进程缓存，可能包含同进程较早的TP/case特化，不冒充逐case profiler trace。
全部正确性特化另通过zero-spill fixture；资源数不直接等于occupancy。
旧final_v2和formal_auto目录是TP1历史实现，不能作为当前TP2/4性能；无`--check-current`可仅审计其冻结证据。
本次保留基准、FP32 reference、测试、计时与审计文件，因为它们是继续优化的验收工具，不是待删除的旧实现。

后续重点是低重合direct的加载/调度、部分query共享与更精细局部union，而不是把扩大并集作为无条件优化。
对于默认 independent，要达到 200 需要改变当前计算分解或找到真实 workload 的可复用结构，
不能从现有结果保证该目标可达。在目标机器复测后，针对**对应最终特化**的 ELF/ISA/descriptor
检查 byte bounds、寄存器、scratch、MFMA/VMEM/LDS 等待；不能拿旧 IR 当新机器码。
需要时参考 [GPU profiling](../../../../docs/profile-gpu.md#L1) 做独立 ATT/PMC，区分 VMEM issue stall 与完成等待；若 profiler 改变 PTL，不能冒充同门禁性能证据。
最后仍用原 cudaPerf、轮换地址和全部样本验收有效 TFLOPS；**不得修改数值容差、FLOPs 分母或用 shared 代替 independent 来宣称达到 200。**
