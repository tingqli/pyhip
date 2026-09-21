---
name: gpu-task-swizzle
description: 'Use when optimizing GPU GEMM or MoE task order, OC splitting, grid swizzle, XCD/L2 locality, padding/SE aliasing, or N-packet phase rotation. Prove valid-prefix coverage and distinguish logical mappings from measured hardware placement.'
---

# GPU任务swizzle与局部性

## 适用范围

优化**任务到输入／权重片段的访问顺序**，而不是更改矩阵数值。适用于独立CTA，也可作为分片persistent队列的逻辑映射。先区分三件不同的事：

- 权重pre-shuffle：改变B的物理存储，适配MFMA lane布局。
- task/grid swizzle：对逻辑`(M块, OC分片)`编号做置换，不搬张量。
- packet N phase：改变一个任务内遍历N的起点，B、scale、输出地址必须一起旋转。

## 实施步骤

1. **还原真实调用。** 记录完整N、Block M、OC分片数S、排序粒度、valid前缀和capacity。不要把tile N当作一个persistent claim覆盖的全部N。
2. **定义任务域。** 若L为padded有效排序行数，B为Block M，则$T=(L/B)S$；`t`解码为$m=\lfloor t/S\rfloor$、$s=t\bmod S$。不能用真实route数或未初始化capacity尾部替代L。
3. **选择前缀转置而非随意取模。** 对宽度w、$c=\lfloor T/w\rfloor$：

   $$
   F_w(v)=\begin{cases}
   (v\bmod w)c+\lfloor v/w\rfloor,&v<wc,\\
   v,&v\ge wc.
   \end{cases}
   $$

   `v >= T`的capacity CTA仍保持无效；不足w的尾部identity映射。T=0或T<w也必须正确。
4. **先做CPU证明。** 整分区是一个w×c矩阵的转置；当c>0时，逆映射为$v=(t\bmod c)w+\lfloor t/c\rfloor$。验证有效任务不重不漏、尾部不进入有效域，再上GPU。可运行[CPU检查脚本](./scripts/check_swizzle.py)。
5. **分析消费域，而不是只看相邻task。** 用映射后的`(m,s)`统计同一A块、同一B`(expert,s)`由哪些XCD消费，兼顾时间复用距离与每XCD工作集。OC分片增加A重复读取；不同OC读的是不同B列，B总量不能再乘S。
6. **测量物理归属。** 只有在该设备/launch上读回XCC证明`XCD(block)=block%8`，才能把CPU模型解释为真实XCD分布。w不是物理XCD数，模型不是pinning API；改变grid/block/LDS/资源限制后需重新验证。
7. **单独测试N相位。** 若每OC有偶数Q个N64 packet，保持每个N128的两半相邻：

   $$p(q)=2((\lfloor q/2\rfloor+r)\bmod(Q/2))+(q\bmod2).$$

   对同一p读取B、使用`p//2`的scale、写回对应p。当前实例$r=\lfloor worker/8\rfloor$；独立CTA的worker是原block ID，不是转置后task；persistent是resident worker，不是queue rank。
8. **配对验收。** 同数据/地址、原timer、正反顺序测Down和完整流程；profiling在另一次采集。新旧routing语义相同，但不同Block M必须各自原生排序。

## 选择宽度的具体实例

默认MoE案例L128=148096、S=8、T=9256：

- width2：$F_2(8j+x)=4628(x\bmod2)+4j+\lfloor x/2\rfloor$。
- width4：$F_4(8j+x)=2314(x\bmod4)+2j+\lfloor x/4\rfloor$。

在`XCD=block%8`模型下，同一M的A消费域通常从identity的8个缩到width2的4个、width4的2个；分段边界可跨更多域，不能省略尾部/边界。**更少复制不保证更快**，资源竞争、N相位和工作集也会改变。

当前M256、S=4、T=3072的width8满足$F_8(8j+x)=384x+j$；每4个j覆盖同一M的四个OC。当前独立M128用width2，persistent M128用width4，不用旧README中的width4替代现状。

## 禁止的推论

- 不能从task编号直接认定SE/CU编号或同时驻留；SE相位可能是每XCD的排列。
- 不能从旧sort256的padding分布推断原生sort128仍存在相同SE倾斜。
- 不能把单CU无VMEM发射区间说成HBM空闲；不能把聚合TCC命中率说成B专属命中率。
- 所有resident CTA从一个全局队列取rank后再做swizzle，**不等于**独立CTA的固定归属；需要[分片队列方法](../gpu-persistent-work-queues/SKILL.md)。

## 本仓库代码与证据

- [M256 width8前缀转置](../../../experiments/moe/flydsl/moe_8w_down/moe_multistage_down.py#L188-L211)。
- [M128 width2/4及queue到virtual task](../../../experiments/moe/flydsl/moe_8w_down/moe_multistage_down_m128.py#L45-L98)。
- [N相位同时进入B地址](../../../experiments/moe/flydsl/moe_8w_down/moe_multistage_down_m128.py#L139-L160)、[输出地址](../../../experiments/moe/flydsl/moe_8w_down/moe_multistage_down_m128.py#L211-L214)。
- [旧sort256与原生sort128的区别](../../../experiments/attention/flydsl/attn_4wave/tools/se-dispatch.md#L186-L292)；其中历史width4实测不可重标为当前width2实测。