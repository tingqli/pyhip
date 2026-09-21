---
name: moe-fp8-tiling
description: 'Use when optimizing gfx942 FlyDSL MoE 8x1 K192/K320 or BK128 tiling, expert-row padding, compact M256 full and M64 tail tasks, routing imbalance, PTPC/per-tensor scales, or Down/Combined/Full path selection.'
---

# MoE FP8分块与compact尾块

目标：按实际有效工作量、专家分布和完整调用成本选择分块，而不是把单CTA吞吐当作全链性能。这里针对[FlyDSL两阶段MoE](../../../src/pyhip/ops/moe/flydsl/moe_gemm_2stage/README.md)；不要直接套用gfx950块量化或GRRead的数值规则。

## 1. 先核实现行支持范围

从[dispatcher](../../../src/pyhip/ops/moe/flydsl/moe_gemm_2stage/gemm2.py)与[8x1 builder](../../../src/pyhip/ops/moe/flydsl/moe_gemm_2stage/gemm2_8x1.py)读取真实约束：

- 当前Down路径为`default`、`1x4_64x256`、`8x1`、`8x1_compact`。旧分支名称和历史调参开关不是现行API。
- 此8x1是M256/N128、512线程，FP8 E4M3FNUZ输入/权重、BF16输出；支持K192/256/320/384/512/640，N为正的128倍数。
- 当前权重/激活组合为PTPC/PTPC、per-tensor/per-tensor、per-tensor/PTPC；量化布局和scale位置必须与调用方一致。
- K192使用完整192，K320使用128+192，其余使用BK128。K192/K320的`tile_k=128`兼容值不代表实际退回旧分块。
- 这些是本版本的实现约束，不是硬件通用限制。修改前重新读源码，不根据历史矩阵恢复已删除模块。

## 2. K分块要与数值及流水一起设计

1. 列出实际K宽度与offset，证明覆盖恰好K且无重复。K320不能将128+192误记为两个128或通过填充增加MFMA。
2. 区分归约块和量化块。本实现的PTPC/per-tensor scale路径应以`pack_8x1_record`和`row_scale`构造为准，不能照搬另一kernel的K128独立缩放公式。
3. 固定FP32累加、weight scale、activation/routing scale与最终BF16转换顺序。现行gfx942 MoE打包与GRRead整数BF16 helper不是同一个合同，不在整理代码时顺便替换。
4. 对每种K记录B槽大小、每packet MFMA、VMEM请求宽度、scale load、pack与store的消费者。K192搬运的每lane24B是16B+8B两条请求，不能作为一条VM事件。
5. 独立处理短N、首N、过渡N、稳态和最后N。K192只有L/H两拍，K320的128/192非对称槽不能照抄BK128倍数的等待表。
6. 检查C/scale生存期与最大B暂存重叠。延后BF16打包会保留更多FP32结果与scale，少几条指令也可能提高寄存器峰值。

## 3. 先量化专家行padding成本

设专家e的有效route行数为$r_e$，M块大小为B：

$$
R_{exec}(B)=\sum_e B\left\lceil r_e/B\right\rceil,\qquad
\rho(B)=R_{exec}(B)/\sum_e r_e.
$$

- 记录有效行、执行行、分配容量三个不同量；有效TFLOPS使用有效工作量，不能用padding后的MFMA数提高分子。
- 重点测试专家行数在64/256倍数附近的台阶。例如每expert从512变513行，M256执行行从512变768；这是确定的工作量变化，不自动证明全部时延增幅由padding造成。
- 只屏蔽输出store不会省掉已发出的MFMA。要减少无效计算，需改变任务粒度或满块/尾块分工。
- 均衡、偏斜、少量活跃专家等routing分布分别评估；单seed均衡合成分布不能证明生产分布最优。

## 4. compact任务表的正确性条件

参考[compact实现](../../../src/pyhip/ops/moe/flydsl/moe_gemm_2stage/gemm2_8x1_compact.py)：

1. 保留M64 sorting metadata；每任务为`[physical_row_begin, expert_id]`，不重写原始route物理行号。每个expert必须是唯一连续run，run排列可不按expert编号。
2. 对一个含$b_e$个M64块的run，分为$\lfloor b_e/4\rfloor$个M256 full和$b_e\bmod4$个M64 tail。证明每个有效物理块恰好被处理一次。
3. GPU构表后同stream执行full、tail；counts由device消费，不做Host计数回读或同步来决定grid。workspace只依据Host可知容量和CU信息预分配。
4. 当前可将全局full表不足CU利用率阈值的末轮后缀拆成M64；每拆一个full增加4个tail，需同时更新容量、计数和跨expert后的输出位置。阈值0.6是当前策略，不是普遍最优。
5. full=0也可能仍发构表、空full和tail三个launch，不能将其成本按零处理。
6. N能被256整除时tail用1x4，否则使用支持该宽度的default；独立Down的N128支持不代表gateup或reduce整链也支持它。
7. 同指针的routing内容可以改变，不能按pointer缓存任务表。Graph重放和连续调用要重新构表并验证零任务、全full、全tail、混合、容量边界。
8. 大expert权重地址在乘法前升到Int64；counts单位是行数还是任务数必须与消费者约定一致。

## 5. 按完整执行边界选择路径

- Down计时包含compact的构表/full/tail；Combined包含对应Down与reduce；Full需查看实际sorting、quant、gateup、Down、inverse、sum是否在同一event内。
- 三个独立阶段中位数不能相减推算sum，也不能各取最快路径拼成一个“Full结果”。
- 单Down获益可能被构表、padding传播或TOPK归约抵消。检查真正被计时的所有轮换权重均已shuffle，校验真正的timed output，不只校验预热结果。
- 选择规则以预先确定的同址配对统计为准；历史推荐不是自动selector，也不能跨输入分布/设备搬用。
- 正确性先于性能；按[性能验证skill](../gpu-benchmark-validation/SKILL.md)保持原timer、地址条件和门禁。本文是工作方法，不授权自动重跑全矩阵或修改硬件。

## 6. 证据入口

- [现行分发、支持范围与历史测量身份](../../../src/pyhip/ops/moe/flydsl/moe_gemm_2stage/README.md)。
- [任务表覆盖回归](../../../tests/contrib/moe/test_compact_m64_tasks.py)、[compact历史验收](../../../tests/contrib/moe/results/compact_m64/summary.json)。
- [六K流水与资源历史审计](../../../tests/contrib/moe/results/readme_latest_20260911/audited.json)。
- [打包后移的Memory/Compute反例](../../../tests/contrib/moe/results/k256_memory_pack_20260909/att_comparison.json)。
- [计时路径权重准备错误的诊断](../../../tests/contrib/moe/results/qwen397_perf_gap_20260911/suite.json)。

历史证据用于定位条件和反例，不自动证明当前源码仍有同样问题或性能。