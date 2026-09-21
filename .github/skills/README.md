# PyHIP GPU优化skills

本目录使用仓库级Agent Skills布局：每个子目录包含带`name`／`description`的SKILL文件，由支持该布局的工具按需发现；不需要额外的全局指令或GPU环境改动。

这些skills记录**可复用的推导、实施和验证方法**，不是无条件套用的最佳参数。gfx950 MoE down的实例入口见 [四个case及PyHIP基线对照](../../tests/flydsl/moe_8w_down/README.md)。

## 目前学到的优化经验

以下来自gfx950上的MoE实验。先理解原因，再决定是否用于其他形状或设备。

1. **先找真正的瓶颈。** 已有的A复用、B直入共享内存、流水预取不是新优化；优先检查寄存器溢出、等待和输出写入是否连续。
2. **算完一部分，就尽早打包。** 把已完成的FP32结果转成BF16，同时计算另一部分，可缩短寄存器占用时间。消除溢出时收益很大；本来不溢出时可能持平或略慢。
3. **更多独立CTA提供重叠机会，不保证更快。** 两个四wave CTA可各自推进，但小M也会增加任务数和B读取。要同时算收益与代价，不能只看驻留数量。
4. **连续写出不一定要改输出格式。** DPP跨线程重排可将同一行的连续覆盖从64B扩大到128B，仍保留routed布局和原归约。重排数据时，目标行和有效性必须一起变。
5. **先定写出布局，再调cache策略。** NT+SC1在64B布局下可能变慢，在128B布局下却有收益。用“64/128B × 开/关策略”四组对照，别把两个因素混在一起。
6. **局部性优化要随布局重新测。** 分片队列和任务重排没有固定赢家；写出布局变化后，原来最快的任务顺序可能不再最快。
7. **最终看完整调用链。** Down更快不一定Full更快；必须包含队列清零、布局转换和归约的实际成本，不能将分项耗时相加代替端到端计时。
8. **让结论可复现，也保留反例。** 每次只改一个因素；同数据、同地址、交替顺序、多轮和多seed复测。检查数值、graph、边界和实际机器码；没有硬件计数证据，就不把提速说成“HBM事务减半”。

详细数据见 [A8W4综合优化记录](../../tests/flydsl/moe_8w_down/A8W4_OPTIMIZATION.md)。技能正文保留方法，不复制完整性能表；实验赢家也不自动成为默认配置。

| Skill | 适用场景 |
|---|---|
| [gpu-task-swizzle/SKILL.md](gpu-task-swizzle/SKILL.md) | OC任务拆分、grid转置、XCD消费域、独立的N相位、尾部双射检查 |
| [flydsl-mfma-pipeline/SKILL.md](flydsl-mfma-pipeline/SKILL.md) | Memory/Compute交错、提前打包减少溢出、四wave独立CTA驻留、pair publication |
| [moe-packed-epilogue/SKILL.md](moe-packed-epilogue/SKILL.md) | routed/packed的64B→128B写出、DPP重排、cache策略对照、inverse＋TOPK reducer |
| [gpu-persistent-work-queues/SKILL.md](gpu-persistent-work-queues/SKILL.md) | 全局／分片persistent队列、保持swizzle归属、单claim站点、可证明的末退出worker自复位 |

共同要求：先核实基线已经具备什么；以真实输出布局和调用链测量；保留参考精度、同步和失败记录；普通计时、ATT、PMC分开报告。旧实验目录是只读证据，不在其中写源码、日志或字节码缓存。