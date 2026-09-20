---
name: gpu-benchmark-validation
description: 'Use when benchmarking or diagnosing PyHIP GPU kernels, GRRead latency or effective TFLOPS, cudaPerf rotating buffers, PTL and idle-GPU gates, alignment-sensitive regressions, ATT steady-state VMEM issue stalls, or differences between ordinary timing and PMC/ATT evidence.'
---

# GPU性能测量与归因验证

目标：得到可复现、口径明确的性能数据，避免把采样环境、地址变化或局部trace指标误判为kernel优化。以下门禁是本仓库gfx942参考环境的约定；其他平台需先明确等价条件，不能直接放宽规则。

## 1. 冻结对象和测试范围

1. 读取当前kernel、调用方和原[计时器](../../../src/misc.py)，记录shape、dtype、归约长度、grid/block、分片、源码哈希。存在编译产物时记录ISA/ELF身份，不能给旧产物换标签冒充新版本。
2. 明确是在测基础阶段、完整调用还是构造准备。GRRead默认不计shuffle、分配、JIT、参考和校验；Full是直接计Down＋Up，不相加两个独立中位数。
3. 正确性必须先于性能：逐元素容差、shape和边界检查不变，rel_l2是附加指标。不能通过放宽容差换取性能。
4. 预先确定shape、buffer数量、预热、样本数和候选顺序。小改动先最小目标阶段验证；只有用户要求或预先规定条件满足后，才扩大矩阵。

## 2. 查空闲GPU，但不要阻塞其他工作

- 一次读取各卡利用率、显存、PCI地址及型号，选择空闲卡；用户允许切换时可以另选空闲GPU，无合格GPU则继续CPU分析或正确性，不sleep、不轮询等待。
- 本仓库性能门禁：GPU use≤5%、VRAM≤20%、PTL **Enabled／VECTOR,F8**。在入口、buffer准备后/采样前、结束后分别核验，保留每次快照。
- 不写PTL、频率、功率或NUMA来让测试通过。不能将一个卡上的before和另一个卡上的after当同场对比。
- 门禁失败先保留已完成结果和全部raw，再停止性能；不能反复采到合格样本或隐去结束门禁失败。正确性检查不要求空闲。
- PTL和powercap相同不等于实际动态频率相同。只有测到kernel期间相关遥测，才能讨论时钟因素，空闲频率不能解释运行时长尾。

## 3. 固定实际地址与执行路径

1. 采用[现有测试入口](../../../tests/contrib/gr_read/test_gr_read.py)中的原`cudaPerf`，默认10个独立X/权重/P/Y buffers、每阶段2warmup、10sample，全样本中位数。不要为了“更稳定”偷偷换timer。
2. 记录各tensor实际data pointer、storage base、storage offset、mod256及mod4096。性能Y用原生allocation起点；guard view用于正确性时不要直接替代性能buffer。
3. 对照尽量使用同一批输入和地址，交错AB/BA或预先固定的ABBA/BAAB，并保持buffer轮换。先后两个整轮数据可以报告，但必须标明非交错、非同址配对的限制。
4. 相同ELF也会因X/Y相对地址变化而变慢。Y加128B仍然满足16B自然对齐，不能据此声称硬件要求256B对齐；256B对齐也不保证任意相对地址都快。
5. 检查实际被计时的输出，不能重跑另一个实现覆盖它后再验正确。所有轮换权重都必须执行同样的shuffle/量化准备，不能只准备预热用的第0组。保留同版重复输入的一致性检查，数值变更另与原参考比较；[历史诊断](../../../tests/contrib/moe/results/qwen397_perf_gap_20260911/suite.json)记录过“只校验预热输出、计时却使用未shuffle权重”的问题，不据此断言当前入口仍有该bug。
6. Graph捕获中每次取得当前stream，不能把捕获前的外层stream固定在闭包里。普通FlyDSL编译会真实launch一次，编译用张量也必须足量。

## 4. 保存样本并使用正确分母

- 每条raw保存scope、sample序号、buffer编号和us；包含第一次、最后一次及所有长尾。汇总从raw重算中位数，不挑快段、不按频率筛样本。
- 每条性能结果同时给出时延和有效TFLOPS，并写明工作量。GRRead固定维度下：

$$
F_D=F_U=2T\times10240\times320,\qquad
F_{Total}=4T\times10240\times320,\qquad
TFLOPS_{effective}=F/(t_{us}\times10^6).
$$

- F不含gate/激活等附加操作，是有效GEMM工作量；不要偷偷换成包含额外VALU的另一口径。1k=1024，不将60k与64k混用。
- 配对ratio中位与“两个完整样本中位数之比”分别报告，不互相替代。声明改善范围，不用一个小batch结论代表全矩阵。
- 通过全样本门槛与通过正确性是两件事。若原验收阈值失败，保留失败，不事后以另一个统计口径宣布通过。
- MoE的Down、Combined、Full边界由实际harness确定：compact构表和空full仍有成本，Full不一定固定中间地址；不能用Combined−Down推算独立sum，也不能拿Down收益代替整链收益。
- `calc_diff`与`rel_l2`不是同一指标。本仓库相关`calc_diff`采用平方误差相对两者能量之和的形式，`rel_l2`为误差范数除参考范数；必须读实际实现、保持原阈值，不换名后沿用数值。
- IQR是样本分位，不是置信区间；跨0不证明等价。全1权重的性能、随机权重正确性和真实routing的端到端表现分别说明，不跨条件拼接。

## 5. 先分清ATT/PMC回答的是什么

| 数据 | 可以回答 | 不能直接回答 |
|---|---|---|
| 原cudaPerf多buffer | 当前环境的真实调用时延 | 某个硬件队列为何阻塞 |
| ATT局部采样 | 指令发射、等待和wave时间关系 | 整卡普通时延、精确HBM响应时间 |
| PMC计数器 | 该次计数范围内的流量、事件 | 未测场次或不同PTL下的因果关系 |
| 请求字节模型 | 算法/代码发出的逻辑数据量 | 实测HBM流量或带宽 |

只有有明确机制假设时才采ATT/PMC，不能因文档整理反复profile。若profiler改变PTL状态，必须记录其实际采样状态，不能把Disabled的PMC结论直接当Enabled的性能结论。

## 6. 针对稳态stall建立可证伪解释

1. 用户问稳态时，先隔离所有相关resident wave都处于内部loop的共同窗口；startup/drain另列，不能用它们解释内部stall。
2. 校验实际code object、PC对应指令、完整wave和stitch质量。`pc_index=0`误指向注释等trace不能用于精确归因。
3. 按physical `(SE,CU,SIMD)`合并resident waves的MFMA执行区间，使用不重叠分类统计剩余时间。不要把各wave stall累加当墙钟；tick和MFMA窗口长度应由目标架构/分析约定明确给出。
4. 区分预取距离、正常VMEM service、issue-stall和completion-wait。`issue = attempt + stall`；`vmcnt`约束给出的是未完成指令上界，不是实际硬件队列占用。
5. issue-stall表示请求路径不能立即接受发射，不等于HBM响应延迟；completion-wait也包含完整访问路径。热缓存或越界零返回的反事实可以否定“必然来自HBM”的过强解释，但需独立标注其ELF和语义。
6. 预取距离与任务映射用2×2单因素实验分开；源码仅改一处仍可能改变寄存器分配/补充wait。没有ISA验证，不宣称机器码只改一个立即数。
7. 物理MFMA union busy乘roof得到的是**模型TFLOPS**。即使局部稳态改善，重复任务的prologue/epilogue或其他CU也可能使普通整体时延不变。

## 7. 交付检查

- 给出精确源/产物、shape、设备与PCI、实际PTL、采样协议、全部raw和失败状态。
- 从原始JSON而非执行代理摘要核对是否真正完成；stdout被重定向不等于未执行，JUnit counts位于子`testsuite`。
- 改代码用当前基线；历史结果只读，成功的exclusive动作不重复运行。Git index可能被用户更新，只记录，不恢复或写入。
- 汇总不混用不同设备、地址条件、时钟状态、PTL或工作量；没有证据时写清尚未确定的原因。

方法来源：[Y地址偏移的同址对照](../../../tests/contrib/gr_read/results/n2_baseline_reconcile_20260918/timing.json)、[VMEM issue与完成等待反例](../../../tests/contrib/gr_read/results/n4_vmem_issue_20260919/summary.json)、[预取与映射2×2](../../../tests/contrib/gr_read/results/n4_prefetch_mapping_20260919/comparison.json)。这些是方法证据，不是当前版本的最新性能表。