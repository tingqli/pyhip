---
name: flydsl-cshuffle-layout
description: 'Use when optimizing FlyDSL gfx942 LDS CShuffle, ds_read_b64 versus ds_read_b128 or ds_read2st64_b64, plane layouts, lane payload mapping, LDS bank-conflict models, output coalescing, or precomputed buffer store offsets.'
---

# FlyDSL LDS CShuffle布局与读取合并

目标：在保持每lane输出payload、数值打包和读写生命周期不变的前提下减少LDS指令与热段地址计算。先分辨B读和CShuffle读，不能看到64-bit LDS读取就统一改成128-bit。

## 1. 从真实producer/consumer表达式建立模型

1. 从[现行8x1代码](../../../src/pyhip/ops/moe/flydsl/moe_gemm_2stage/gemm2_8x1.py)找`cshuffle_plane_offset`、`shuffle_8x1_c_r2s2r`、`store_8x1_c_r2g`与B读取原语。
2. 给每个BF16元素附上`(wave,lane,record,row,half,element)`标签；按真实写地址放入模拟LDS，再按真实读地址收集输出。
3. 覆盖所有wave/lane/half/输出行和N偏移，验证：无非法重叠、无漏读、每次输出payload与原布局逐元素相同。
4. 尽量执行源码中提取的地址表达式，而非另写一个“看起来相同”的模型。全局输出地址和mask也属于证明范围。

## 2. 先证明能否合并，再选指令

- 两个8B来源要构成一个`ds_read_b128`，必须确实相邻、满足对齐并按预期顺序消费；两个原始offset不同不代表可直接合并。
- 已验证的plane思路将同一输出的两段8B放在固定间距的平面中，以一条`ds_read2st64_b64`读取两个地址。该指令仍是两个64-bit子操作，不是连续128-bit读取。
- gfx942历史实现的plane间距是2048B，对应`offset1:4`的编码单位；不要把4当作4B，也不要把不同输出行的基址差误当作read2两来源的间距。换ISA/位宽时重新核对编码。
- `fx.copy`可能被编译器跨输出行重新配对；当前实现用适当的`sched_barrier`限定配对区域。是否真的发出预期read2，需看最终ISA。
- 保留原128-bit写入与全局store宽度。若为减少读指令引入额外move、wait、scratch或更多bank冲突，不能只按指令条数判断胜出。

## 3. bank模型必须按指令的实际服务组

1. gfx942相关静态模型使用32个DWORD bank：`bank=(byte_address//4)%32`。
2. 把一条宽读取/写入展开成该指令实际服务的lane子组和DWORD子操作；读b64、读b128、写b128的lane分组不能混用。
3. 对read2分别检查两个b64子操作；对write2同样分开建模。注明假设，不将模型冲突数冒充PMC实测。
4. 检查所有row、half、wave和地址相位，不只验证lane0或一个N tile。
5. 若需要硬件验证，查本机rocprofiler支持的真实counter及单位。缺失counter不能当零；LDS issue-stall也不能直接等同bank conflict。

## 4. 配套处理地址与等待

- uniform N偏移可走SGPR `soffset`，将复杂lane地址提前到有空隙的Compute准备；不要在load/store wrapper里隐式生成新的VALU。
- `fx.copy(..., soffset=...)`在当前接口中使用元素单位；raw buffer接口与`BufferTensor`显式偏移使用字节，详见[接口与编译验证](../flydsl-codegen-validation/SKILL.md)。
- 读取位宽改变后，重新计算LGKM请求序列；不能保留基于原指令数的等待预算。
- LDS写入完成不等于消费者完成。覆盖旧输出scratch之前，需要确认所有旧读者结束；opaque inline asm的异步依赖也要纳入。
- 先检查短N/首N/末N与尾行，再看稳态。不要以“Compute里没有内存指令”推断Memory末尾任意wait都可删。

## 5. 验收分层

1. CPU真实表达式payload与地址覆盖；可加入会漏行/错行的负例。
2. 离线编译检查实际读写opcode、offset、请求数、寄存器峰值、LDS及spill；完整提取所有出口和kernel。
3. GPU原数值参考、同址对照、NaN/guard和必要的graph生命周期检查。
4. 最小阶段性能确认；只有请求才扩大整链/矩阵。降低指令数、bank风险或某类stall不等于Full改善。

## 证据与区分

- [真实标签与bank模型回归](../../../tests/contrib/moe/test_k192_cshuffle.py)。
- [K192合并读取的历史验收](../../../tests/contrib/moe/results/k192_read2_merge_20260908/final_audited_verified.json)。
- [K320读合并与soffset历史验收](../../../tests/contrib/moe/results/k320_read2_soffset_20260908/final_audited.json)。

本skill针对LDS CShuffle；[packed epilogue](../moe-packed-epilogue/SKILL.md)是另一种全局输出ABI/消费者设计，不应因为名称相似而混称同一个优化。