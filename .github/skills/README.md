# PyHIP GPU优化skills

本目录使用仓库级Agent Skills布局：每个子目录包含带`name`／`description`的SKILL文件，由支持该布局的工具按需发现；不需要额外的全局指令或GPU环境改动。

这些skills记录**可复用的推导、实施和验证方法**，不是无条件套用的最佳参数。gfx950 MoE down的实例入口见 [四个case及PyHIP基线对照](../../tests/flydsl/moe_8w_down/README.md)。

| Skill | 适用场景 |
|---|---|
| [gpu-task-swizzle/SKILL.md](gpu-task-swizzle/SKILL.md) | OC任务拆分、grid转置、XCD消费域、独立的N相位、尾部双射检查 |
| [flydsl-mfma-pipeline/SKILL.md](flydsl-mfma-pipeline/SKILL.md) | direct global→LDS ring、Memory/Compute交错、MFMA延迟退休、pair publication、寄存器生存期 |
| [moe-packed-epilogue/SKILL.md](moe-packed-epilogue/SKILL.md) | 用packed输出避免routed scatter、寄存器DPP重排、inverse＋TOPK reducer、精确舍入与cache policy |
| [gpu-persistent-work-queues/SKILL.md](gpu-persistent-work-queues/SKILL.md) | 全局／分片persistent队列、保持swizzle归属、单claim站点、可证明的末退出worker自复位 |
| [grread-bf16-prefill/SKILL.md](grread-bf16-prefill/SKILL.md) | gfx942 GRRead的BF16边界、H64/stream权重排列、X128与W2、MFMA后处理交织、N分片与精确尾块 |
| [gpu-benchmark-validation/SKILL.md](gpu-benchmark-validation/SKILL.md) | 空闲GPU/PTL门禁、原cudaPerf多buffer、地址对齐条件、有效TFLOPS口径、ATT稳态和VMEM归因 |
| [moe-fp8-tiling/SKILL.md](moe-fp8-tiling/SKILL.md) | gfx942 MoE K192/128+192分块、专家padding、M256 full＋M64 tail任务、整链路径选择 |
| [flydsl-cshuffle-layout/SKILL.md](flydsl-cshuffle-layout/SKILL.md) | LDS CShuffle逐元素映射、plane布局、read2合并、bank子操作模型、地址和等待验证 |
| [flydsl-codegen-validation/SKILL.md](flydsl-codegen-validation/SKILL.md) | 实际Python/FlyDSL环境、元素与字节offset、对齐、离线/JIT缓存、完整ELF/ISA/ABI等价性 |

共同要求：先核实基线已经具备什么；以真实输出布局和调用链测量；保留参考精度、同步和失败记录；普通计时、ATT、PMC分开报告。旧实验目录是只读证据，不在其中写源码、日志或字节码缓存。