---
name: flydsl-mfma-pipeline
description: 'Use when tuning FlyDSL AMD FP8/A8W4 GEMM or MoE Memory/Compute scheduling, direct-global-to-LDS B rings, MFMA packing overlap, register spills, independent four-wave CTA occupancy, pairwise publication, or wave staggering while preserving exact arithmetic.'
---

# FlyDSL MFMA流水与生存期优化

## 先核对基线

不要把已有的MFMA指令、A寄存器复用、B直接进LDS、四槽ring、4＋4 wave错相或寄存器BF16/permlane包装误写成“新增”。本仓库PyHIP down已具备这些；改进点是**工作粒度、发射顺序、重叠范围和等待/复用协议**。

## 实施步骤

1. **固定数据/算术契约。** 明确FP8/FP4格式、K分块、scale布局、routing乘法位置与BF16舍入。blockscale K256实例先算独立K128 dot，再缩放并用FP32 FMA合入；A8W4原生scaled-MFMA则保留各K步的累积顺序、E8M0与opsel。不能将一种内核的算术模板套到另一种，也不能重结合scale或routing。
2. **以实际packet预算资源。** 本例N64×K256的FP8 B packet=16KiB，四槽共64KiB；不是四个64KiB槽。加上ID/scale/queue LDS，再结合VGPR/SGPR/AGPR限制估算驻留。CTA数、waves/CTA、waves/SIMD、同时驻留必须分开。
3. **画出producer/consumer表。** 每个slot记录B DMA、各wave LDS读、对应MFMA和允许覆写的时刻。DMA完成等待只解决生产者；覆写前仍需确认所有消费者读完。不要仅看当前wave已读完便删CTA barrier。
4. **延迟退休MFMA结果。** 将已发射MFMA的FP32 partial放入短队列；先发后续独立MFMA，再做旧partial缩放/FMA，避免立即读结果造成依赖等待。队列过长会增长寄存器生存期，必须以机器码和时延验收。
5. **把前一个packet的包装放入当前Compute。** routing乘法、BF16 conversion、permlane/DPP逐项穿插到MFMA间；store可推迟到后续Memory或再后一包Compute。包装必须消费已完成的FP32结果，保持原RNE顺序；详见[packed epilogue方法](../moe-packed-epilogue/SKILL.md)。
6. **分散有用的请求，而不是填充空转。** 把下一B packet的若干DMA turn分散到当前LDS读之间，或把旧C store分散到MFMA间。绝不添加无意义VMEM指令改善“连续性”图表。
7. **控制热段地址计算。** 在prologue准备lane/row/slot地址；短生存期数据在必要时pin到VGPR。uniform值用标量，但防止大量衍生N地址被提到persistent外层长期占SGPR。只对已观测spill做局部rematerialization，不能假设一律hoist更快。
8. **按短N/稳态/收尾分别验证。** FlyDSL动态循环的向量状态显式`init`/`yield`；末尾禁止预取下一任务地址。所有带任务复用的路径在下一次metadata/B覆写前排空必要DMA/LDS消费者。

## 可选协议：两packet一起发布

适用于Q为偶数且四槽读写协议可配对的路径；**不要求相邻两包共享scale**。不能盲目替代所有PF3流水。

- 四槽0…3；启动至多预取q0…q3。
- q0只等待q0/q1完成，q2/q3可继续飞行。
- 偶数q≥2先等待当前pair完成，CTA同步退休前一pair读，再向已退休的两个槽填q+2/q+3。
- 奇数q使用前一偶数阶段已发布的B；不再重复CTA publication barrier。
- 仅当q+2<Q才refill，最后一pair不越界；仍逐packet等待LDS读进入寄存器。
- blockscale实例中，可在偶数包读共享的N128 scale、奇数包复用；A8W4每个N行的scale不同，必须保留每包自己的scale。不得旋转两包时拆散既定配对。

这是**减少发布频次**，不是取消必要同步或减少矩阵数据量。测量更少barrier是否真正缩短Down；如果别的访存/VALU瓶颈占主导，收益可能很小。

## A8W4实验的实用经验

- **提前打包的价值首先是缩短生存期。** BN128先完成左N64的全部K，在右N64的MFMA间打包左半；每个输出自身的K顺序不变。已测M256/8wave的K384消除了spill，但无spill的K256近乎持平。更细地拆散打包操作也未稳定更快，BN64不能直接套用左右N64方案。
- **区分CTA数量和wave数量。** 一个8wave CTA换成两个4wave CTA，每CU总wave数可以不变，变化是它们不再共用同一个barrier。这里只提供独立推进的机会，不保证两个CTA始终错开执行。
- **小M不是免费提高驻留。** M128要原生sort128；padding可能减少，但M块、B请求和队列开销可能增加。只把M256线程数减半，还会增大每wave工作量和寄存器压力。
- **用同机器码隔离驻留因素。** 从实际ELF查询LDS、寄存器和最大CTA/CU；额外预留dynamic LDS可限制驻留，再确认GPU指令未变。静态扩大LDS可能改变编译结果，不能直接当作纯驻留对照。
- **保留形状边界。** K256/BN128可容纳两个四wave CTA，K384的LDS预算可能只容纳一个；BN64虽提高驻留，也可能增加循环开销。小batch和短任务必须单独复测，不能按K或occupancy一个数字选赢家。

证据：[计算/打包交错](../../../tests/flydsl/moe_8w_down/A8W4_OPTIMIZATION.md#compute-overlap)、[独立四wave与驻留对照](../../../tests/flydsl/moe_8w_down/A8W4_OPTIMIZATION.md#four-wave)。

## 编译器与ISA注意事项

- 先看安装的FlyDSL API；不要把其他环境的intrinsic签名直接照搬。
- 位宽、addrspace和字节偏移要精确。当前gfx950 direct-LDS intrinsic支持16B/lane；不要因营销说明假设32B/lane可编译。
- `s_setprio`改变同SIMD wave仲裁，不改变GPU频率，也不保证全局优先级。
- 4＋4组的错相barrier属于一个配对协议：prologue与epilogue必须闭合；不能只保留其中一端。
- `sched_barrier`、空inline-asm pin、标记helper可能改变编译器调度，不是可任意删的“无作用注释”。
- `remaining=7`是源码调度预算，不等于所有实际MFMA间都有7条机器VALU；startup、drain、copy和DPP均需看ISA。
- 使用编译器识别的DPP intrinsic暴露跨lane读hazard；opaque inline asm不一定得到正确间隔调度。
- 静态MFMA/store条数不是动态执行量；布局改变可能影响循环展开或尾部代码复制，不能把静态条数减少直接解释为少做了计算。

## 验收

- 精度/graph：短N、padding、跨expert边界、valid0、invalid→恢复、原始严格相消测试。
- ISA/资源：MFMA opcode/次数、native FMA、无意外scratch、AGPR及SGPR spill都核对；不要只看profiler CSV的寄存器列。
- 普通timing：同址正反对照、保持原timer，Down和Full分别实测。ATT是局部追踪，不是普通耗时；各wave的wait不能求和成独占kernel时间。
- 只有在有假设时采ATT/PMC，不因文档修改或纯格式化重跑GPU。

## 本仓库代码

- [PyHIP已有的wave错相与Memory/Compute序列](../../../src/contrib/moe_gemm_8wave.py#L1121-L1225)。
- [M256 DMA/scale](../../../tests/flydsl/moe_8w_down/moe_multistage_down.py#L278-L327)、[延迟退休与包装交错](../../../tests/flydsl/moe_8w_down/moe_multistage_down.py#L340-L448)、[错相首尾闭合](../../../tests/flydsl/moe_8w_down/moe_multistage_down.py#L474-L511)。
- [M128 pair/PF3 publication与refill](../../../tests/flydsl/moe_8w_down/moe_multistage_down_m128.py#L221-L261)、[延迟store/包装/退休](../../../tests/flydsl/moe_8w_down/moe_multistage_down_m128.py#L263-L339)。
- [worker N相位rematerialization](../../../tests/flydsl/moe_8w_down/moe_multistage_down_m128.py#L139-L150)。