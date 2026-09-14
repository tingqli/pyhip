---
name: moe-packed-epilogue
description: 'Use when optimizing MoE GEMM output stores and TOPK reduction with packed sorted-row output, register BF16/permlane/DPP reordering, inverse route indices, native sorting alignment, or cache hints without changing FP32/FMA/BF16 numerical semantics.'
---

# MoE packed输出与专用reduce

## 目标与前提

将不规则routed写回从大GEMM的epilogue移到更小的inverse＋gather reduce，必要时用寄存器bit交换让相邻lane合并写事务。**只在消费者能直接读packed格式时成立**；如果还需要单独restore到routed再sum，应把restore成本纳入比较。

原PyHIP已经有寄存器BF16/permlane且无C LDS。因此新增点是packed地址、额外DPP合并和消费者设计，不是笼统的“去掉PyHIP的C LDS”。

## 实施步骤

1. **固定数值顺序。** 每route：K128 partial→FP32 factor乘法→下一K128 partial显式FMA→routing乘法→BF16 RNE；然后TOPK的BF16值扩到FP32顺序相加→最终BF16。不能提前跨route合并FP32、重结合scale或放宽容差。
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
- 与routed＋`torch.sum`比较时，共享A/B/scales和输出地址，原始routing语义一致；解码/参考/投毒放在计时外。
- 流量模型注明“实际有效输出”与“分配的padding容量”。默认有效BF16 route写出是1.5GiB，不是全部capacity行。
- 使用PMC解释HBM时，DRAM32B计数乘32，用**同dispatch**时间作分母；`_sum`不再乘XCD。NT不等于写流量减少或一定更快。
- 全零routing、valid0、专家尾块、恶意有效-looking capacity tail、连续graph replay都要测试。包含接近BF16中点的跨K相消，不只用随机allclose。
- 一个独立reduce workspace不能隔离缓存down callable的内部persistent queue；遵守[队列并发契约](../gpu-persistent-work-queues/SKILL.md)。

## 本仓库代码与回归

- [PyHIP routed地址和store](../../../src/contrib/moe_gemm_8wave.py#L1080-L1110)。
- [BF16/permlane与row/column DPP交换](../../../tests/flydsl/moe_8w_down/moe_multistage_down.py#L64-L115)。
- [M128 packed输出地址及DPP mask](../../../tests/flydsl/moe_8w_down/moe_multistage_down_m128.py#L177-L214)。
- [packed reducer索引、load与顺序sum](../../../tests/flydsl/moe_8w_down/moe_multistage_reduce.py#L40-L59)。
- [workspace与完整pipeline](../../../tests/flydsl/moe_8w_down/moe_multistage_pipeline.py#L12-L82)、[inverse valid-prefix保护](../../../src/contrib/flydsl/moe_gemm_2stage/moe_reduce.py#L110-L136)。
- [严格相消](../../../tests/flydsl/moe_8w_down/test_blockscaled.py#L515-L555)、[专家边界/容量尾部](../../../tests/flydsl/moe_8w_down/test_blockscaled.py#L558-L599)。