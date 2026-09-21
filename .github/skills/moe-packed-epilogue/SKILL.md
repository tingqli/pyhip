---
name: moe-packed-epilogue
description: 'Use when optimizing MoE GEMM routed or packed output, 64B/128B per-row store coalescing, BF16/permlane/DPP reordering, NT/SC1 cache-policy experiments, inverse route indices, or TOPK reduction without changing numerical semantics.'
---

# MoE写出合并、packed布局与归约

## 目标与前提

先区分两个独立选择：**跨线程重排，让同一行写得更连续**；**改成packed布局，改变后续归约的读取方式**。DPP重排不依赖packed，routed输出也可以做到128B/行，并继续使用原来的`torch.sum`。

只有选择packed时，才将不规则routed写回改为排序行写出，再交给inverse＋gather reduce。消费者必须能直接读packed；如果还需restore到routed，应把转换成本纳入比较。

原PyHIP已经有寄存器BF16/permlane且无C LDS。因此新增点是packed地址、额外DPP合并和消费者设计，不是笼统的“去掉PyHIP的C LDS”。

## routed 64B→128B：先试不改输出格式的方案

1. **按一条wave store、同一有效行计算连续宽度。** 原A8W4每lane写16B，四lane写同一行，共64B。BN128每行最终写256B，是多条store的总量，不是单次合并宽度；4/8wave也不决定这个宽度。
2. **交换行和列的一位。** 令`lr=lane%16`。先完成原route乘法、BF16舍入和permlane，再用DPP交换`lr`低位与BF16列的bit5，使八lane各写16B，共同覆盖一行128B。输出仍为`[tokens, topk, N]`。
3. **数据和目标有效性一起重排。** 广播对应偶/奇目标route，不能沿用源lane的mask。无效目标必须落到buffer真正OOB；靠buffer边界抑制写入，不需要额外可写的sentinel行。测试相邻行一真一假、全无效、空任务和恢复。
4. **把布局和cache拆成四组。** 固定调度，测64/128B × aux0/aux18。当前gfx950的NT=2、SC1=16、两者=18；检查实际计时ELF，去掉store标志后，同宽度的完整指令文本应相同。原Tensor赋值与显式copy可能生成不同代码，先保留aux0作桥接控制。
5. **以Full决定是否采用。** 两seed复测中，常用M128路径在8K/16K/32K受益，128B下NT+SC1通常还有额外Full收益；但4K不宜统一开启，8K无width转置的分片路径在aux0下DPP还会使Down变慢。改布局后要重新测任务顺序。

128B只是逻辑连续地址覆盖，**不等于证明一个128B硬件事务或HBM流量减半**。DPP也有指令和寄存器成本；记录scratch和驻留是否变化，不能只看地址更整齐。

实现及反例：[A8W4写出实验](../../../tests/flydsl/moe_8w_down/A8W4_OPTIMIZATION.md#coalescing)、[轻量四组性能对照](../../../tests/flydsl/moe_8w_down/experiments/bench.py)、[主四wave边界回归](../../../tests/flydsl/moe_8w_down/test_a8w4.py)。

## 选择packed布局时的步骤

1. **固定数值顺序。** blockscale实例每route：K128 partial→FP32 factor乘法→下一K128 partial显式FMA→routing乘法→BF16 RNE；A8W4保留原生scaled-MFMA的K累积顺序。然后TOPK的BF16值扩到FP32顺序相加→最终BF16。不能提前跨route合并FP32、重结合scale或放宽容差。
2. **写出物理ABI。** 设Block M=B、排序位置l、完整输出列n、完整N。布局为`[m_block, global_N64, row_in_block, col64]`，元素索引：

   $$
   E(l,n)=\lfloor l/B\rfloor BN+\lfloor n/64\rfloor B64+(l\bmod B)64+(n\bmod64).
   $$

   2D张量只是存储容器，并非普通行主序。buffer分配是`[capacity_blocks*B, N]`；OC只分列，**不能再乘OC扩行数**。
3. **匹配sorting。** down、expert索引、valid行长度、inverse位置及reducer全部用同一B。M128必须sort128；不能只是把GEMM的M改128而继续沿用sort256测成原生路径。
4. **从lane布局推导寄存器重排。** 先列出每lane持有的M/N值、BF16 pair顺序及最终目标字节地址。用permlane恢复MFMA结果布局，再交换row/column低位，使地址实际连续；不要只看lane内col，不计row stride就宣称coalesced。
5. **保护输出有效性。** 输入valid与目标输出行valid不同。若DPP交换row bit，必须同时交换目标mask；推荐识别的intrinsic以保留hazard信息。无效buffer地址要落到真正OOB且不溢出32-bit sentinel；大buffer需另行支持，不能简单截断。
6. **每次重建inverse。** `inverse[token,slot]=sorted_position`；先fill(-1)，只读valid前缀。reducer将-1或OOB route贡献置0。相同pointer不代表routing内容相同，禁止按pointer缓存inverse。
7. **向量化消费者。** 当前示例256线程×每线程8个BF16覆盖2048列；提前读取TOPK位置与各route片段，FP32累加。选择列块大小时同时看读合并、寄存器压力和小N浪费。
8. **分别调读写策略。** streaming packed中间值可测试NT store/read，最终输出仍可cached。当前gfx950 B读取aux16、M256 C aux2、M128 C aux18是已选配置，不是所有GPU最佳值；必须核对实际ISA标志。

## 测量与边界

- 分别报告Down、inverse fill/build、reduce以及直接计时的Full；Full不能由组件数相加替代。
- warm reduce单独计时不能代表刚完成Down后的cache状态；Full的收益不能未经测量就全归给reduce。
- 与routed＋`torch.sum`比较时，共享A/B/scales和输出地址，原始routing语义一致；解码/参考/投毒放在计时外。
- 流量模型注明“实际有效输出”与“分配的padding容量”。默认有效BF16 route写出是1.5GiB，不是全部capacity行。
- 使用PMC解释HBM时，DRAM32B计数乘32，用**同dispatch**时间作分母；`_sum`不再乘XCD。NT不等于写流量减少或一定更快。
- 全零routing、valid0、专家尾块、恶意有效-looking capacity tail、连续graph replay都要测试。包含接近BF16中点的跨K相消，不只用随机allclose。
- 上述完整边界用于维护中的主实现；历史性能探索先做一次普通精确检查，再专注同址配对计时，不重复主实现的鲁棒性矩阵。
- 一个独立reduce workspace不能隔离缓存down callable的内部persistent queue；遵守[队列并发契约](../gpu-persistent-work-queues/SKILL.md)。

## 本仓库代码与回归

- [PyHIP routed地址和store](../../../src/contrib/moe_gemm_8wave.py#L1080-L1110)。
- [BF16/permlane与row/column DPP交换](../../../tests/flydsl/moe_8w_down/moe_multistage_down.py#L64-L115)。
- [M128 packed输出地址及DPP mask](../../../tests/flydsl/moe_8w_down/moe_multistage_down_m128.py#L177-L214)。
- [packed reducer索引、load与顺序sum](../../../tests/flydsl/moe_8w_down/moe_multistage_reduce.py#L40-L59)。
- [workspace与完整pipeline](../../../tests/flydsl/moe_8w_down/moe_multistage_pipeline.py#L12-L82)、[inverse valid-prefix保护](../../../src/contrib/flydsl/moe_gemm_2stage/moe_reduce.py#L110-L136)。
- [严格相消](../../../tests/flydsl/moe_8w_down/test_blockscaled.py#L515-L555)、[专家边界/容量尾部](../../../tests/flydsl/moe_8w_down/test_blockscaled.py#L558-L599)。