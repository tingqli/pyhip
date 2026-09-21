---
name: gpu-persistent-work-queues
description: 'Use when implementing or reviewing GPU persistent GEMM/MoE scheduling, global versus sharded atomic queues, task swizzle affinity, graph-safe counter lifecycle, or last-worker self-reset without inter-CTA spinning. Separate correctness from locality and performance assumptions.'
---

# Persistent任务队列与安全自复位

## 先拆开三个问题

1. resident CTA反复做任务，是否减少每个逻辑任务的硬件WG接纳成本？
2. 谁领取哪个逻辑任务，能否保持需要的A/B缓存消费域？
3. 队列在下一次调用／graph replay之前，是否一定回到合法状态？

PyHIP MoE down已有全局atomic persistent队列。分片与自复位才是本例新增内容；persistent本身不是天然的性能优势。

## 实施步骤

1. **固定有效域。** 设总任务T、逻辑shard数X、worker数P，先保证每个shard确有worker；例如P是X的正倍数。空任务、少于P的任务、shape改变均在契约中。
2. **选调度方式。** 全局`atomicAdd(head,1)`容易负载均衡，但task与物理XCD不固定；静态`v=worker+kP`可保持模X同余，但无动态平衡；分片可在一个复用域内兼顾动态调度与局部性。不要仅给全局rank做swizzle就认为恢复了独立grid归属。
3. **逻辑分片保证覆盖。** 令$h=worker\bmod X$，$r=\operatorname{atomicAdd}(head[h],1)$，$v=Xr+h$，仅$v<T$时执行，再用[task swizzle](../gpu-task-swizzle/SKILL.md)得到真正`(M,OC)`。只读真实XCC建队列则还必须证明所有shard都有worker，不能依赖未验证的launch分布。
4. **分开correctness和affinity。** 逻辑方案不依赖实际物理XCD就不重不漏；性能上的同XCD复用，仍需测量`XCD(worker)=worker%X`等假设。当前8个128B间距head用于隔离热点，128B不是所有设备cache-line大小的声明。
5. **只保留一个claim代码站点。** 一个leader atomic→LDS发布→CTA barrier→一致读取/退出。曾见prologue和latch两个相同claim被LLVM合并成lane-divergent多入口CFG；诊断store还会掩盖问题。一定验证无诊断的普通kernel和graph。
6. **切任务前退休资源。** 当前任务的B DMA、LDS读者和必要global请求必须按协议完成，才复用metadata/ring。没有跨CTA同步并不意味着CTA内barrier可以删。
7. **从含reset的完整调用比较。** 先使用显式每次清零作安全基线；只有证明下述条件后才尝试kernel内自复位。常驻P要与LDS、寄存器、CU数一起测，不把“launch P=512”等价于512个CTA同时驻留。

## 可复用的末退出worker自复位证明

适用于**无work stealing、固定shard成员、每次领取恰好1任务、越界立即退出且不再claim**的队列。

每shard有效任务数：

$$T_h=\max(0,\lceil(T-h)/X\rceil).$$

该shard有W个worker，head初始为0。全局原子的唯一rank中，0…$T_h-1$为有效任务，随后每个worker恰好领到一个terminal rank退出。terminal rank恰好为$T_h$…$T_h+W-1$。

取得最后rank $T_h+W-1$ 的worker可以置`head[h]=0`：W个不同worker已经各领取terminal rank，因此没有worker还能产生后续claim。它们可能尚未完成函数退出，但不再访问这个head。下一launch在同stream上等整个kernel完成，故看到已恢复状态；不需要跨CTA自旋或grid barrier。

当前T总能被8整除，P=512、X=8、W=64，条件简化为`rank == T/8+63`。**63不是通用常量。** T=0仍需要64次terminal claim/shard才能安全复位。若有bundle、stealing、worker提前退出、失败/取消或多个kernel并发共享状态，必须重新证明，不能沿用该公式。

## 生命周期与验证

- 仅首次分配置零；从一开始即valid0也必须归零。捕获前完成分配/编译预热。
- callable持有私有head，调用者不应投毒内部状态；公开scratch/counter的约定不可暗改。
- 同一缓存callable禁止跨stream并发；单独的输出workspace不能隔离同一内部head。要并发需独立队列拥有者或明确的跨stream同步。
- 测试同callable不同tokens/valid前缀、连续两个launch同graph、多次replay、空/invalid→恢复，检查最终全head及padding为零、公开counter不变。
- CPU模拟任意claim/消费交错验证无重复/遗漏；GPU验证原始非诊断路径。新增placement记录只用于证据，不把它的时延当性能。
- 若性能目标未到，保留负结果。原子小字节数不证明延迟无关；L2读取少也不证明kernel更快。

## 本仓库实现

- [PyHIP全局claim](../../../src/pyhip/ops/moe/asm/moe_gemm_8wave.py#L833-L864)、[M256全局claim与独立width8](../../../experiments/moe/flydsl/moe_8w_down/moe_multistage_down.py#L188-L211)。
- [M128逻辑分片/单header claim](../../../experiments/moe/flydsl/moe_8w_down/moe_multistage_down_m128.py#L72-L98)。
- [任务结束同步与末terminal自复位](../../../experiments/moe/flydsl/moe_8w_down/moe_multistage_down_m128.py#L369-L387)、[首次分配与私有状态](../../../experiments/moe/flydsl/moe_8w_down/moe_multistage_down_m128.py#L400-L436)。
- [graph/shape生命周期与CPU交错回归](../../../experiments/moe/flydsl/moe_8w_down/test_blockscaled.py#L602-L681)。