# MI308X FP8 Block-Scale GEMM 优化记录

日期：2026-10-31。目标：8 waves、256x256 输出 tile、固定 M split=1，FP8 A8W8 + FP32 block scale，BF16 输出，cold-cache 吞吐约 500 TFLOPS。

## 主线与测量口径

当前主线是 **v1 -> v2 -> v8 -> v11 -> v25 -> v32**。v32 仅优化预重排 B 的 N=4096 分支，其他路径保持 v25；其余多数是分支实验，不是逐版叠加。

- v2 拆开 MFMA/DQ 依赖，分散预取，B-scale 改走 SMEM。
- v8/v11 改善输出合并和预取分布；v11 的后续 ATT 中，采样稳态 MFMA 忙碌比例达到 94.9%。
- v25 仅对预重排 B 微调阶段 barrier 位置，最终在目标验收 shape 上达到 cold-cache 513.7 TFLOPS。
- v32 对 N=4096 做四相位 N64 包置换，保持主循环 ISA 不变；同范围 ATT 将4096活跃 mainloop 从85.85%提升到94.25%，8192参照为94.06%，同址冷吞吐408.1T提升到433.7T。
- 开发主要使用 `M=16384, N=3584, K=6144`；最终验收使用 `8192x8192x16384`。不同 shape、布局和测次的绝对 TFLOPS 不能直接排名。
- 表中百分比是各自同轮基线的配对中位时延变化，负数更快。没有独立 ATT 的候选，只报告测量结果，不把推测当作瓶颈结论。
- v11、v8、v12 性能接近，排序会随测次变化；三版源码、机器码及同轮排名已[冻结保存](../gfx942_blockscale_top3/manifest.json)。

## 已确认有收益的改进

| 改进 | 有效机制 | 证据与采用范围 |
|---|---|---|
| v2：MFMA/DQ 解耦与搬运重排 | partial 与最终 C 分离；B-scale 用 SMEM；预计算地址，分散 A 预取；每条 MFMA 后穿插独立标量 DQ | 组合改动比 v1 快约 25%；ATT 的 VMEM issue 从 18.4% 降至 5.0%，VALU 可见空档从 4.8% 降至 0%。没有单项消融，不能分别声称各项贡献 |
| v8：16B 合并输出 | BF16 打包后通过 lane XOR-16 交换，合并相邻 N16 子块；每 lane 的 store 从 32 次 8B 减为 16 次 16B | 相对 v2 配对时延约 -3.0%；最终保留，数值转换不变 |
| v11：均衡预取 | 四个 packet 分别预取 `[A0,SA0]`、`[A2,SA1]`、`[A1]`、`[A3]`，避免请求集中在单个阶段 | 与 v8 的差距接近噪声；后续同轮排名和 ATT 支持将其作为基线，但不视为稳定、可叠加的百分比收益 |
| v12：Memory 内 VMEM/LDS 交错 | 在 LDS read 之间发出未来 VMEM，减少队首发射停顿向后传播 | 相对 v8 某轮配对时延约 -1.7%；保留为备选，没有叠加到最终 v25 |
| v25：预重排 B 的窄 barrier 调整 | 仅把阶段切换提前到最后一条寄存器操作数 MFMA 前；不提前 LDS 覆写、不减少 barrier 总数 | 相对 v11 预重排 B 的配对时延约 -1.9%；普通 B 不改动。提前 4 条 MFMA 的 v3 已回退，不能扩大解释 |
| v32：跨 WG 的 B 请求错相 | N=4096预重排路径按 `p ^ ((blockIdx.x // 4) & 3)` 访问 N64 包，B-scale和输出同步置换；不改K累加顺序、MFMA/DQ或wait/barrier | 同址冷时延 -5.90%，活跃 mainloop 提升约8.41pp；240 VGPR、32 KiB LDS、零scratch，其他代表路径设备代码逐字节不变 |

上述收益来自不同对照，不能相加或相乘估算最终吞吐。v1 的四槽 ring 和 M split 是正确性/流水线基础，没有独立测量相对最早原型的收益。

## 版本汇总

| 版本 | 主要改变 | 测量或 profile 结论 |
|---|---|---|
| 初版 | gfx942 MFMA；A/B 经 VGPR 到单份 LDS；DQ 紧跟 MFMA | 正确性原型，尚无有效跨 phase 重叠 |
| v1 | 8x1 wave 分工；A 寄存器双缓冲、四槽 B-LDS ring；固定 M split、延迟 DQ | 约 289-318T；ATT：MFMA 忙碌 62.6%，VMEM issue 占稳态 18.4% |
| **v2** | B-scale 改为 SMEM；分散 A 预取；预计算地址；MFMA 间穿插独立 DQ | 比 v1 快约 25%，约 422T；ATT：MFMA 忙碌 79.2%，VMEM issue 降至 5.0%，VALU 可见空档降至 0% |
| v3 | Compute 结束 barrier 提前 4 条 MFMA | 回退至约 377T，撤回 |
| v4 | 未来 A/B 预取移入 MFMA 间隙 | 比 v2 慢 2.4%，撤回 |
| v5 | 取消 Compute 高优先级 | 基本中性，未保留 |
| v6 | 两包共用阶段切换；六槽 48 KiB LDS | 比 v2 慢 14%；ATT：barrier 降至 4.1%，但 LDS 发射/等待合计升至 16.7% |
| v7 | 增加 8 VGPR，提前读取第二包首个 B 子块 | 仍比 v2 慢 11.3%，撤回 |
| **v8** | 相邻 N16 输出跨 lane 拼接，8B store 改为 16B store | 比 v2 快 3.0%，约 442T；保留 |
| v9 | 缩短 Compute 高优先级窗口 | 无稳定收益 |
| v10 | 32x32 FP8 MFMA、4x2 wave 分工、部分 A 原位预取 | 约 237T；正确性通过但吞吐回退，未独立确认具体 stall 原因 |
| **v11** | A 数据与 scale 预取更均匀地分布到四个 packet | 与 v8 接近；冻结同轮排名约 457T，成为后续基线；采样 ATT 的 MFMA 忙碌达 94.9% |
| v12 | Memory 阶段内交错 VMEM 与 LDS read | 比 v8 快约 1.7%，约 448-454T；作为备选冻结 |
| v13 | B 写 LDS 提前到前一 Compute 尾部 | 比 v12 慢 3.3%，撤回 |
| v14 | 取消强制 4+4 错相，每 packet 只保留一次 barrier | 比 v12 慢 13.4%，撤回 |
| v15 | 16x16 MFMA 改为 4x2 wave 映射，减少 B 广播 | 约 297T；减少 B 广播未转化为整体收益 |
| v16 | A/B 均协作加载，经 64 KiB LDS 分阶段复用 | 比 v11 慢 4.8%，未采纳 |
| v17 | 分散 A 写 LDS，补足跨组读取退休等待 | 略有改善，仍比 v11 慢 3.9% |
| v18 | M 分组从 8 改为 4 | 无稳定收益 |
| v19 | 两套 partial，增加到四条独立 MFMA 累加链 | 256 VGPR，基本中性 |
| v20 | 两包 B 完整驻寄存器，A 单份原位预取 | 比 v11 慢 16.8%，未采纳 |
| v21 | 最后 K 块的 BF16 打包前移到 MFMA/DQ 尾部 | 无稳定收益 |
| v22 | 整 N tile 删除输出 mask 分支 | 无稳定收益 |
| v23 | 外层 task 分组宽度从 8 改为 4 | 无稳定收益 |
| v24 | 输出 XOR-16 交换改用 ds_swizzle_b32 | 无稳定收益 |
| **v25** | 基于 v11，仅预重排 B 将 barrier 提前到最后一条 MFMA 之前；普通 B 保持原调度 | 同址对比快 1.9%；最终 cold-cache 预重排 B 达 **513.7T** |
| v26 | 当前B的LDS读提前到下一B的VMEM等待前，发布前完整退休DS | 4096同址约快0.6%，不足以解决主要缺口，未保留 |
| v27 | B预取q+3改为q+5，暂存bank从2增到4 | 4096约408.1T，无收益，排除单纯预取距离不足的解释 |
| v28/v29 | 外层task groups从8改为2/1 | 分别约405.5T/417.7T；v29同址约415.9T，仅有限收益 |
| v30 | v29基础上M分组8改为16，增强B复用 | 约405.9T，更多B复用并不保证更快 |
| v31 | 每4个WG交错两个N128半块 | 同址425.8T；匹配SE0/CU0活跃mainloop 86.98%提升到89.82%，方向有效 |
| **v32** | 每4个WG交错四个N64包，保留原task映射 | **同址433.7T，时延 -5.90%；广范围无丢包对照94.25%，达到8192的94.06%水平** |
| v33 | v32叠加v29的task groups=1 | 约432.3T，无额外收益，未保留 |
| v34 | 四相位改为相邻WG轮换，即 `blockIdx.x & 3` | 约412.1T，错相粒度重要，撤回到v32 |

## 当前设计

- **算术契约**：A 每行每 K128 一个 FP32 scale；B 每 N128xK128 一个 FP32 scale。先完成一个 K128 的 FP32 dot，再执行 `C = fma(partial, scale_A * scale_B, C)`，不跨 scale 块合并点积。
- **计算分工**：WG 为 512 threads / 8 waves，输出 256x256；每 wave 负责 32x256，M 拆为两个 16 行片段。每个 K128 依次处理四个 N64 packet，每个 M-split phase 为 16x64x128。
- **数据路径**：A 直接进入两套操作数 VGPR；B 用两套预取 VGPR 中转，再 `ds_write_b128` 到四个 N64xK128 LDS 槽，每槽 8 KiB，总计 32 KiB。A-scale 双缓冲进 VGPR，B-scale 双缓冲经 `s_load_dword` 进 SGPR。
- **重叠方式**：4+4 waves 错开 Memory/Compute；Memory 读取当前 `Q[q]`、提交 `Q[q+1]`、预取 `Q[q+3]`。wait 预算来自实际请求顺序，不用无条件排空代替流水线。
- **N=4096预重排路径**：逻辑packet仍为0..3，实际列包为 `packet ^ phase`，`phase=(blockIdx.x // 4)&3`。scale行取对应N128半块，最终store取回对应列；每个输出仍按原K0..K127顺序累加。
- **DQ 与寄存器复用**：N64 的四个 N16 partial 分成两对；计算当前 P0/P1 时退休上一 phase 的 P2/P3，计算当前 P2/P3 时退休当前 P0/P1。仅用 16 个 partial VGPR，最终 C 占 128 VGPR；总资源实测 240 VGPR、零 scratch。
- **安全边界**：槽复用要等落后 wave group 的读取也退休，首尾 barrier 必须配对。v25 提前的最后一条 MFMA/DQ 只访问寄存器，不能据此继续提前 LDS 写入。排空最后 partial 后做原 BF16 转换、lane 交换和 16B store。

### 为什么 A/B 不采用相同 load

两边的 global 指令都是 `buffer_load_dwordx4`，不同的是加载后的共享方式。当前 8x1 wave 分工中，每个 wave 拥有独立的 32 行 A，A 可留在本 wave 的 VGPR 并复用于四个 N64 packet；同一包 B 则要给全部 8 个 wave 使用，协作加载后经 LDS 广播能避免八份 global 读取。每个 WG、每 K128 的逻辑 A/B 数据各为 32 KiB，这不是 A/B 数据量不同，而是跨 wave 的复用域不同。

| 方案 | 已有实测范围 | 结论 |
|---|---|---|
| v11/v25：A 直入 VGPR，B 经 LDS | 四组目标 shape、两种 B 布局 | 当前保留基线；不保证其他 M/N 下仍最优 |
| v16：A/B 都协作加载，经 64 KiB LDS | `16384x3584x6144`、普通 B | 同址对照比 v11 慢约 4.8% |
| v17：同上，分散 A 写 LDS并修正退休等待 | 同上 | 比 v11 慢约 3.9%，仍未超过基线 |
| v15：4x2 wave 分工，减少 B 广播 | 同上 | 回退约 48%；同时改变了 wave 映射和 A 预取，不是纯 load 消融 |

上述是早期单 shape 对照。以下补充本轮跨 shape 的同路加载比较，以及 4096 方阵差距的实际 CU 分配证据。

### 同路加载的跨 shape 冷缓存对照

最佳 v25 已提交为 `4437044`。候选保留 256x256 输出、8 waves、M split、MFMA/DQ 与 BF16 转换；两边直接进 VGPR 或两边协作进 LDS，必然同时改变内存请求量、资源占用和相应 wait。每版均独立完成 32-buffer 预热后计时 32 次，再按反向版本顺序重复，共 64 样本；版本共享同一批输入地址，但不连续计时同一地址，避免热缓存偏差。全部候选通过 36 个正确性/guard/Graph 场景。

| MxNxK | B 布局 | v25：A直入VGPR/B经LDS | A/B均直入VGPR | A/B均经LDS |
|---|---|---:|---:|---:|
| 4096x4096x16384 | plain | 389.0T | 84.7T | 388.9T |
| 4096x4096x16384 | preshuffle | 408.0T | 172.5T | **420.3T** |
| 8192x8192x16384 | plain | **483.2T** | 106.1T | 471.1T |
| 8192x8192x16384 | preshuffle | **513.0T** | 220.9T | 510.0T |
| 16384x3584x6144 | plain | **430.2T** | 97.9T | 418.3T |
| 16384x3584x6144 | preshuffle | **484.5T** | 206.5T | 468.1T |

全 VGPR 版本每个 wave 都读取同一包 B，增加重复 global 请求；这个原型明显回退，不能据此断言所有 direct-load pipeline 都无效。全 LDS 版本在 4096 方阵预重排上有小幅收益，在 8192 方阵也达到约 510T，但不是跨 shape 普遍更优；其资源为 249 VGPR、64 KiB LDS，v25 为 240 VGPR、32 KiB LDS。当前保留候选与[完整对照](../gfx942_symmetric_load/comparison.json)，不全局替换 v25。

### 4096 方阵为何比 8192 方阵低

256x256 tile 下，4096 方阵启动 256 个 WG，8192 方阵启动 1024 个 WG。不能只比较单个 WG 的计算量：在 80 CU 上，256 个 WG 的尾轮可能只有少数 CU 工作。此次用同一份 `N=4096,K=16384` code object，仅改变运行时 M，再采集全部 16 个 SE 的 occupancy，确认：

| Shape | WG总数 | 实际80 CU的WG分布 | v25预重排中位耗时 | 对称LDS预重排中位耗时 |
|---|---:|---|---:|---:|
| 4096x4096x16384 | 256 | 64个CU各3个，16个CU各4个 | 1347.1 us | 1310.3 us |
| 5120x4096x16384 | 320 | 80个CU各4个 | 1380.1 us | 1347.6 us |

后者多 25% 算术工作，耗时却只多约 2%-3%。这是两种加载方式都存在的 WG 尾轮损失，不是 A/B load 不对称独有的问题。当前资源下每 CU 驻留一个8-wave WG；均匀等耗时模型的容量利用率分别为 `256/(80*4)=80%` 与 `1024/(80*13)=98.46%`，吞吐比约 81.25%，与方阵间的实测差距相近。模型不等于全 GPU 的严格性能上限，剩余差异仍包含访存、缓存、时钟和 WG 耗时变化。

单纯 swizzle 不能让一个 WG 同时占用多个 CU，也不能消除任务数除不尽造成的尾部。若要求 4096 方阵也接近大方阵吞吐，需要进一步测试该 shape 专用的小 tile、split-K 或工作切分；这些会改变当前固定256x256的约束或增加额外开销，不能未经验证直接替换。[同code-object分轮对照](../gfx942_symmetric_load/round_boundary.json)、[实际CU分布](../gfx942_symmetric_load/physical_distribution.json)已保存。

### 4096 的 MFMA 效率与真实尾部时间

进一步分析相同 code object、预重排 B、32-buffer cold-cache 的 4096/5120 ATT。occupancy 覆盖全部 80 CU；指令分析覆盖实际有 trace 的 14 个 CU、56 个 SIMD，分别有 392/448 个完整 wave。

| 指标 | 4096x4096x16384 | 5120x4096x16384 |
|---|---:|---:|
| 活跃 wave 共同主循环的 MFMA 忙碌比例 | 83.30% | 89.52% |
| 主循环 barrier 空档 | 7.91% | 4.96% |
| 主循环 VMEM 发射/等待空档 | 5.35% | 2.75% |
| 主循环 LDS 发射/等待空档 | 3.29% | 2.61% |
| 无 resident wave 的 CU 时间 | 14.86% | 2.43% |
| 其中，CU 提前结束形成的尾部空闲 | 14.85% | 1.40% |

4096 中只分到三个 WG 的 64 个 CU，中位数在所属 SE 的时间轴 82.35% 处就结束；剩余时间继续等待分到四个 WG 的 CU。实际记录中每 CU 最多驻留 8 个 wave，即当前一个 WG。这里的无任务比例来自每个 SE 内目标 kernel 首个 wave 进入到最后一个 wave 退出的 occupancy 积分，不假设跨 SE 时间戳对齐。

这里有两类不同分母的观测：**整次 kernel 的尾轮不满**，以及 **4096 活跃 mainloop 内更多的同步/访存空档**。前者不能解释后者，“少量慢 WG”也只是分布现象，不是活跃 mainloop 效率低的原因。256 个 WG 分成 64个CU各3个、16个CU各4个，已经是计数上最均匀的分配，不代表 swizzle 有错误。不能说差距全部来自理想模型中的20%损失，也不能把活跃 MFMA 忙碌比例与无任务 CU 时间直接相加或相乘成精确全GPU利用率，因为两者分母不同。

4096 的 prologue/mainloop/epilogue 生命周期占比分别约为 1.72%/96.14%/2.14%，对应阶段 MFMA 效率约 41.70%/83.32%/29.76%。prologue 包含首 K 块，epilogue 包含末 K 块及写回。仅选取 SE0/CU0 时，4096 主循环忙碌比例为86.21%；此前8192的94.0%也来自SE0/CU0，但属于另一次采集，不能把两者差值直接解释为唯一的shape效应。

结论：compute-bound 只描述活跃计算阶段，不保证整卡全程满载；小 tile 是改善任务粒度的待测候选，不是既定要求。目前保留256x256 tile与两种加载路径。[4096效率报告](../gfx942_symmetric_load/profile4096/mfma_report.json)、[阶段统计](../gfx942_symmetric_load/profile4096/phase_efficiency.json)、[5120控制组](../gfx942_symmetric_load/profile5120/mfma_report.json)、[真实CU空闲积分](../gfx942_symmetric_load/cu_idle_analysis.json)已保存。

### 4096/8192 的 L1/L2 PMC 对照

2026-10-31，冻结 v25、预重排 B、K16384、256x256 tile、8 waves。固定 GPU0 (`0000:0a:00.0`)，每次使用32份独立 A/B/scale clone，预热0..31后仅采目标 kernel 第33次调用，即再次使用 buffer0。两种 shape 各采 L1、L2、memory 三组 counter，每组一个目标 dispatch；六次输出均通过原阈值校验。

**环境限制：rocprofv3 PMC 自动将 PTL 置为 Disabled，formats 查询返回 NOT_SUPPORTED。** 六次采集记录的 PTL 状态一致；没有手动更改 PTL、频率或功率。采集前 profiler 外确认 PTL Enabled；采集后确认恢复为 Enabled/VECTOR,F8，前后均GPU空闲且无其他进程。原始入口失败的日志也保留。普通计时门禁未放宽，PMC 使用显式诊断模式，不能将其约2.424/7.909 ms当成普通 cold benchmark。

L1 指 TCP/vL1D，使用本机 ROCm 7.2.4 rocprof-compute 的公式：`100 * (1 - (TCP_TCC_READ_REQ_sum + TCP_TCC_WRITE_REQ_sum) / TCP_TOTAL_CACHE_ACCESSES_sum)`；目标 ISA 无原子操作，省略两个原子请求项。L2 为 `100 * TCC_HIT_sum / (TCC_HIT_sum + TCC_MISS_sum)`，包括工具定义的读写等访问，不是单独的 read-hit 指标。

| 整次 kernel、全部硬件实例 | 4096x4096 | 8192x8192 |
|---|---:|---:|
| L1/vL1D 命中率 | 70.60946% | 70.60954% |
| L2 命中率 | 39.42065% | 39.41931% |
| L2 miss / WG | 61,292.36 | 61,273.65 |
| Fabric read 平均周期 (`RDREQ_LEVEL/RDREQ`) | 378.26 | 365.01 |
| TCP 等待 L2 数据的累计 stall cycles / WG | 269,430.78 | 249,004.95 |
| TCP read-tag 冲突累计 cycles / WG | 194,304 | 194,304 |
| TCP TCR stall / DRAM credit stall | 0 / 0 | 0 / 0 |

这次 PMC **不支持4096的 L1/L2 命中率更低**。Fabric read 平均周期约高3.6%，每WG的 TCP pending stall 约高8.2%；它们提示后续检查请求服务、排队和等待隐藏，但不证明 HBM 带宽、缓存容量或任务顺序是原因。Fabric 指标不是纯 HBM 延迟，跨实例累计 stall 也不是可直接相加的墙钟空档。以 TCP clocks 为分母，pending stall 反而是25.92%/29.50%，不能混用分母归因。

此前 ATT 在匹配 SE0/CU0、两条 resident wave 均处于 K2..125 的窗口内，MFMA busy 为86.2111%/93.9894%；4096额外空档主要是 barrier +3.74pp、VMEM issue +2.22pp、VMEM wait +0.85pp。Compute 内连续 MFMA 的中位间隔均为16 cycles，暴露的 VALU 空档均为0。**这些 Enabled-PTL 的活跃窗口，与本次 Disabled-PTL 的整 kernel PMC 不可直接作因果等同。** 每组仅一个采样、各组地址独立，未估计重复测量方差；仍需在原条件下追踪内存 producer 到 wait/barrier 的传播，不能声称已找到或修复7.78pp差距。

[PMC 数值与公式汇总](../gfx942_symmetric_load/cache_pmc_summary.json)、[完整原始数据与复算脚本](../gfx942_symmetric_load/cache_pmc_raw.tar.gz)已保存，后者包含 counter CSV、run JSON、日志、实际 code objects、本机公式配置和失败记录。

### v32：修复4096活跃 mainloop 的供数空档

针对性实验否定了“只是B预取太近”的解释：提前LDS读取仅约0.6%收益，增加两包B预取没有收益，单独改变task groups也不足。改变B包访问相位则在**731条主循环指令完全相同**、资源与K累加顺序相同的条件下，同时改善冷时延和活跃mainloop。可确认的问题位于跨WG的B请求顺序及其VMEM等待向barrier传播；不是DQ变慢，也不能用尾轮空闲解释。具体TCC bank、fabric分区或硬件队列仍未独立证明，不能把软件的4-WG分组称为已知XCD映射。

实现仅需在prologue为4个B包设置置换后的地址、调整对应B-scale行，在epilogue写回对应列。`phase=0,1,2,3`分别对应包顺序 `0123 / 1032 / 2301 / 3210`，每个N128内的两包仍相邻。LDS逻辑槽、partial/DQ依赖、4+4 wave生命周期和所有barrier保持v25。

最终验收固定GPU0、PTL Enabled/VECTOR,F8，32份不重叠输入，预热0..31后采第33次调用。为避免大shape的ATT丢包，三组均使用16-SE mask、每CU只采SIMD0、1GiB trace buffer；实际为**相同14个物理SE/CU/SIMD位置**，98/94/364个wave全部完整且stitched一致。只统计两条resident wave均处于K2..125的窗口，包含所有完整批次，不含CU无任务时间。

| 匹配活跃 mainloop | v25 4096 | v32 4096 | v25 8192参照 |
|---|---:|---:|---:|
| MFMA busy | 85.847% | **94.252%** | 94.058% |
| VMEM issue空档 | 1.634% | 0.301% | 0.705% |
| VMEM wait空档 | 0.341% | 0.024% | 0.064% |
| barrier空档 | 10.531% | 4.106% | 3.874% |
| LDS issue+wait空档 | 1.339% | 0.962% | 0.907% |

另一次全4-SIMD的4096对照覆盖56个物理SIMD，v25/v32为82.72%/92.01%，同样显示广范围改善；对应8192采集出现data loss，已排除并保存失败日志。单SE四SIMD对照为86.98%/95.50%，8192为94.95%。这些采样范围不能混用。最终有效对照说明4096已达到8192约94%的水平，不把单次采集的0.19pp差值解释为统计显著优势，也不称其为整卡利用率。

原 `cudaPerf`、同址32-buffer正反完整轮次、每版64样本：4096预重排B时延 **1347.093 -> 1267.556 us**，下降 **5.904%**；吞吐 **408.105 -> 433.713 TFLOPS**，提高6.275%。两版实际计时输出 `calc_diff=8.985e-6`。固定256x256/8 waves/M split=1，未修改输入格式、timer、PTL或数值容差。

24项pipeline回归通过，覆盖N=4096的短K、奇数K块、单行和跨256行尾块，两种B布局、NaN输出、guard与Graph；另5项未改路径guard/Graph控制通过。最终源码编译的 `.text` 与profiled v32相同。8192普通/预重排B、4096普通B、N3584/3392预重排B等五条代表路径与冻结v25设备代码逐字节相同。

邻近M3840/M5120性能对照在入口发现6个其他GPU进程而终止，没有重试或放宽门禁。N4096分支的其他M/K虽有正确性覆盖，尚无通用性能保证；不能声称该项性能回归已通过。生产实现已切换v32，冻结v25与原两种加载路径保留。

证据：[v32源码](../gfx942_blockscale_v32/v32.py)、[验收汇总](../gfx942_blockscale_v32/summary.json)、[同址冷样本](../gfx942_blockscale_v32/cold_pair.json)、[回归结果](../gfx942_blockscale_v32/correctness.xml)、[复现脚本和设备代码](../gfx942_blockscale_v32/validation_artifacts.tar.gz)。

## 与 gfx950 参考实现的区别

下表针对本次[本地 gfx950 参考 kernel](../../../../flydsl/fp8_gemm_950/test_gemm_fp8_8w_blockscale.py)的 `split_m=True, with_scale=True` 路径，不泛指所有 gfx950 kernel，也不声明与远端 PR 最新 revision 完全一致。

| 方面 | gfx950 参考路径 | 当前 gfx942 / MI308X |
|---|---|---|
| 共性 | 8 waves、256x256 输出、K128 FP32 block scale、M split、延迟 DQ | 保留相同目标和数学契约 |
| 单条 FP8 MFMA | `16x16x128_f8f6f4`，一次覆盖 K128 | `16x16x32_fp8_fp8`，四次累加覆盖 K128 |
| 输入 FP8 格式 | 本次参考用 E4M3FN | 本次测试用 E4M3FNUZ；数据编码不能原样混用 |
| Wave/phase 形状 | 象限内采用 2x4 wave 分工，split 后每 phase 为 32x32x128 | 8x1 wave 分工，每 phase 为 16x64x128 |
| global 到 LDS | 支持 128-bit/lane direct-to-LDS 搬运，参考代码使用 `BufferCopyLDS128b` 等接口 | 不使用 gfx950 的 `buffer_load_dwordx4 ... lds`；B 必须先进入 VGPR 再写 LDS，A 直接留在 VGPR |
| LDS 与 ping-pong | A/B 的 top/bottom、left/right 都在 LDS 做双缓冲；声明的 A+B 数据区约 132 KiB，另有 scale 空间 | 64 KiB LDS 预算下只放 B 四槽 ring，实际用 32 KiB；A 双缓冲移到寄存器 |
| LDS 布局 | 分组 padding | 128-bit 块的 row/column XOR swizzle，无 A/scale LDS 占用 |
| Scale 路径 | A-scale 经 LDS ping-pong；split 路径 B-scale 用 `s_buffer_load_dword` 取标量 | A-scale 进双套 VGPR；B-scale 用 `s_load_dword` 进双套 SGPR |
| DQ 调度粒度 | 每条 K128 MFMA 周围安排四条标量 FMA，复用单 FIFO | 按四次 K32 MFMA 和两对 partial 编排，每条 MFMA 后主要安排一条独立 DQ FMA |
| 输出 | 原生 BF16 packed conversion 与 `permlane16_swap`，16B store | 保留 PyHIP 的 gfx942 BF16 转换，经 `ds_bpermute_b32`/select 做 lane 交换，再 16B store；不声称跨架构 bit-exact 舍入 |

注意：gfx950 参考的公共 prologue 还声明并预装完整 B-scale LDS 区，但 `split_m=True` 的热路径实际取数走 SMEM，不能把它概括成“B-scale 每步都从 LDS 读”。LDS 容量按源码声明区分，不能把寄存器里的逻辑 fragment 数与 LDS 分配相加。

核心差异不是换一个 MFMA opcode，而是 **gfx950 用较大的 LDS 和宽 direct-to-LDS 搬运换取寄存器余量；gfx942 必须重新分配 VGPR/LDS 的职责，并重新安排每个 K128 内的 MFMA/DQ 和搬运节奏**。

## 最终验证

### Cold-cache 吞吐

MI308X / gfx942，GPU0 独占，PTL 为 Enabled / VECTOR,F8。使用原 `cudaPerf`，A/B 各 32 份不重叠 clone，A+B 工作集 8 GiB；同一地址复用前经过 7.75 GiB 输入。64 个样本覆盖两轮轮换，不额外 flush 缓存；允许单次 GEMM 内部正常的缓存复用。

| 8192x8192x16384 | 中位耗时 | 中位 TFLOPS | calc_diff |
|---|---:|---:|---:|
| 普通 B | 4.536 ms | 484.8 | 9.16e-6 |
| 预重排 B | 4.281 ms | **513.7** | 9.17e-6 |

预重排 B 两轮分别为 514.4、512.9 TFLOPS。目标在该 shape 的预重排路径达成，不代表所有 shape/布局都达到 500T。

### 生命周期与效率

- 240 VGPR、32 KiB LDS、零 scratch。源调度审计覆盖 25 种 K 长度和约 103 万次全局地址检查；动态 ISA 审计覆盖 104 个完整 wave，未发现异步 load 未完成就读取或覆写。12 项尾块、guard、Graph 回归通过。
- 安全结论限定于已验证的输入/launch 契约及 gfx942 wait-counter 语义，不等同于通用硬件 race detector。
- 最终 cold-cache ATT 的主循环占生命周期 95.9%，采样物理 SIMD 的 MFMA 忙碌比例为 94.0%；包含首 K 块的 prologue、包含末 K 块及写出的 epilogue 分别占 1.8%、2.3%。这是 SE0/CU0 的采样，不是全 GPU 绝对利用率。
- 保留相同 MFMA、内存、wait、barrier 和资源，移除 DQ 的诊断副本耗时由 4.281 ms 降至 4.204 ms，仅减少 1.79%；主循环 MFMA 忙碌比例由 94.0% 升至 94.4%。去 DQ 版本不产生正确 GEMM 输出，不能计为达标性能；删除指令也会改变 PC 对齐与动态仲裁。

结论：最终主循环接近 compute-bound，DQ 大部分已隐藏。收益来自依赖拆分、合适的预取节奏、输出合并和细粒度同步调整，而不是更多缓冲、更少 barrier 或更大的 MFMA 就一定更快。

## 是否还有优化空间

**有，但最终预重排大 shape 已主要是小幅挤压，不再有证据支持一次大改获得几十个百分点。** 这个判断只基于最终 cold-cache 采样，不外推到普通 B、小 M 或其他 N/K。

| 优先级 | 方向 | 依据与约束 |
|---|---|---|
| 较高 | 普通 B 路径、静态权重预重排 | 同一验收 shape 的普通 B 484.8T、预重排 B 513.7T，吞吐相差约 6%；应对普通 B 单独 profile。计时不含预重排成本，只在可摊销准备成本的工作流中讨论端到端收益 |
| 较高 | 小 M、短 K、N 尾块的 shape 专用配置 | 大 shape 的计算占比不能代表这些场景；WG 分轮尾部、padding 和 prologue/epilogue 更可能主导。task/grid 参数需重新证明覆盖并实测物理分布 |
| 中等 | 细粒度同步、LDS 发射和内存请求节奏 | 最终稳态空档约 6.0%：barrier 2.59%、LDS 发射/等待 1.84%、VMEM 发射/等待 1.43%、其他 0.14%。这些是互斥归因，不等于可直接删掉的成本 |
| 较低 | 再优化 DQ 或扩大缓冲 | 去 DQ 的非正确 GEMM 诊断只减少 1.79% 耗时；已有 240 VGPR。v6/v7/v19/v20 表明更多 LDS、partial 或 B buffer 不自动更快，必须重新核对生存期和占用率 |

将最终采样窗口视为频率、访存与调度完全不变的模型：主循环占比 `f=0.95857`、MFMA 效率 `u=0.93990`，假设能消除全部主循环空档，理想加速为 `1 / (1 - f * (1-u)) - 1 = 6.1%`；假设整个生命周期的所有 MFMA 空档都能消除，按整体效率 `0.91617` 得到约 `9.2%`。

这两个数只是该采样模型的乐观上限，不是全 GPU 的硬上限或可承诺收益；内存搬运、barrier 发布、输出写回都不能归零，时钟与请求仲裁也可能随改动变化。合理预期是先寻找少量、可重复的增益，并优先改善尚未接近 compute-bound 的 shape/布局。

后续每个候选应保留当前冻结基线，按“同址多 buffer 正反对照 -> 原阈值正确性与生命周期检查 -> head 匹配的 ATT”验收。不要重复采用已否决的宽泛方向：整体提前 barrier、机械合并 packet、扩大寄存器状态或仅因指令更大而更换 MFMA。

## GEMM归档与MoE交接

本次GEMM开发在v32处归档，下一阶段从该实现开发MoE，不在本次归档中新增MoE内核。v25基线为 `4437044`，v32及FlyDSL版本在其后提交。

### 文件入口

| 内容 | 入口 |
|---|---|
| 当前可调用GEMM | [gemm_fp8_gfx942.py](../../../../../src/contrib/gemm_fp8_gfx942.py)，符号 `gemm_a8w8_blockscale_8wave_gfx942` |
| v32冻结代码及验收 | [v32.py](../gfx942_blockscale_v32/v32.py)、[summary.json](../gfx942_blockscale_v32/summary.json) |
| 原接口、参考及回归 | [test_a8w8_f32_blockscale.py](../../test_a8w8_f32_blockscale.py)，`run_gfx942` / `run_torch` / `test_gfx942_pipeline` |
| A/B均经LDS的候选 | [gemm_fp8_gfx942_lds.py](../gfx942_symmetric_load/gemm_fp8_gfx942_lds.py)，保留对照用途，尚未应用v32错相 |
| v25与前三版 | [v25.py](v25.py)、[top3/manifest.json](../gfx942_blockscale_top3/manifest.json) |
| 优化过程与被否定方案 | 本文的版本汇总、v32供数空档修复、生命周期审计与PMC限制 |

### 必须保留的GEMM契约

- gfx942 FP8 E4M3FNUZ输入、BF16输出；B逻辑布局为 `[N,K]`，预重排仅指 AITER `shuffle_weight(B, layout=(16,16))`，不是完整AITER接口兼容承诺。
- A-scale连续布局为 `[K/128,M]`，B-scale为 `[ceil(N/128),K/128]`；每K128先完成FP32 dot，再用原FP32 scale乘积和FMA累加，最后按原规则转BF16。
- `N % 16 == 0`、`K >= 128`、`K % 128 == 0`；M为正、grid覆盖完整输出，字节偏移必须落在现有32-bit地址计算范围内。
- 256x256输出、8 waves、M split=1。A为两套操作数VGPR，计算K块k时预取k+1，没有同时保存完整k/k+1/k+2的三缓冲。
- B为两套暂存VGPR加四槽LDS ring；每槽N64xK128、8KiB，共32KiB。容量相当于两个128x128，但不是同一B块的两份副本，也不是固定当前K/下一K两套LDS。
- 4+4 wave发布/读取/覆写协议、请求顺序推导的vmcnt、DS退休和首尾barrier必须一起保留；gfx942不能照搬gfx950的128-bit/lane direct-to-LDS加载。
- v32只在预重排N4096分支做N64包XOR置换；B-scale和输出列必须同步映射。这是已验证的GEMM请求错相，不是可直接沿用的MoE专家调度策略。

### MoE起步顺序

1. 先固定真实调用契约：GEMM1/激活/GEMM2/加权归并的边界、专家权重与scale布局、top-k路由、sorted row和原token的映射，以及输出dtype与累加顺序。尚未确定的项不预设成已有实现。
2. 从单专家、恒等路由接入v32计算主体，先与归档GEMM逐元素对照；然后增加按专家分段的任务和路由行加载，不同时引入persistent queue或更换tile。
3. 显式处理空专家、少量行、尾tile及padding。路由/scale偏移必须按所属专家和有效行重新推导，不能只替换A指针后沿用连续GEMM的buffer边界。
4. 在正确性通过后再加入真实输出scatter或packed输出、routing weight和top-k reduction；明确在哪个精度阶段乘权重和归并，不能擅自改变FP32/FMA/BF16语义。
5. 最后测任务顺序、N64错相、A/B混合或全LDS路径对真实专家负载的影响。当前240 VGPR已很高，新增路由元数据后先核对spill、占用率和生存期，再考虑融合或加深预取。
6. 验收使用原timer、独立输入冷轮换、实际输出校验及Graph；分开报告各GEMM、归并和端到端耗时。GEMM的433.7T/513.7T不能当作MoE吞吐承诺。

本次归档包含源码、调用测试、公共映射helper、版本快照、配置、原始性能/ATT/PMC、失败记录和逐文件SHA-256清单。它是基于上述仓库的工作成果快照，不是完整Python/ROCm环境镜像；不包含无关的core/test工作树修改。恢复后应先检查manifest，再在实际环境做目标验证。N4096邻近M性能门禁失败、PMC自动关闭PTL和ATT采样范围的限制均保留，不补写为通过。

## FlyDSL版本：8x1、2x4-LDS pf1与2x4-LDS pf2

2026-11-01，三个kernel均以FlyDSL独立实现，原PyHIP汇编源码保持不变：

| Kernel | FlyDSL源码 | 来源 | 资源 |
|---|---|---|---|
| 8x1 | [gemm_a8w8_blockscale_gfx942_8x1.py](../../../../../src/contrib/flydsl/gemm_a8w8_blockscale_gfx942_8x1.py) | PyHIP v32移植 | 242 VGPR、32 KiB LDS、0 scratch |
| 2x4-LDS pf1 | [gemm_a8w8_blockscale_gfx942_2x4_lds_pf1.py](../../../../../src/contrib/flydsl/gemm_a8w8_blockscale_gfx942_2x4_lds_pf1.py) | PyHIP 4x2-LDS移植 | 240 VGPR、64 KiB LDS、0 scratch |
| 2x4-LDS pf2 | [gemm_a8w8_blockscale_gfx942_2x4_lds_pf2.py](../../../../../src/contrib/flydsl/gemm_a8w8_blockscale_gfx942_2x4_lds_pf2.py) | FlyDSL原生r3 | 250 VGPR、64 KiB LDS、0 scratch |

2026-11-04改名：两者wave排布相同（M向2个×N向4个；PyHIP的"4x2"按gfx950的(N, M)顺序），区别在global load比写LDS早1个stage（`_2x4_lds_pf1`，即4x2-LDS）还是2个stage（`_2x4_lds_pf2`，即2x4 r3）。

两个移植版保留PyHIP的tile映射、MFMA/DQ顺序、barrier与4+4错相协议，以及gfx942 BF16打包（`v_add_f32`加`0x8000`位型后取高半）。12个shape×两种B布局各24例与PyHIP输出**逐位一致**，含NaN填充、guard与Graph。性能为GPU4同址32-buffer冷输入、正反两轮各64样本：

| Shape / B | PyHIP 8x1 v32 | FlyDSL 8x1 | PyHIP 4x2-LDS | FlyDSL 2x4-LDS pf1 | FlyDSL 2x4-LDS pf2 (r3) |
|---|---:|---:|---:|---:|---:|
| 4096 预重排 | 432.9T | 429.8T | 423.7T | 422.5T | 414.3T |
| 4096 普通 | 388.1T | 385.1T | 406.5T | 407.8T | - |
| 8192 预重排 | 514.8T | 513.1T | 515.0T | 513.7T | 506.7T |
| 8192 普通 | 484.2T | 476.7T | 502.8T | 503.6T | - |

2x4 r3的数值来自另一轮同协议测量（`gfx942_flydsl_2x4/summary.json`），不是与移植版同轮配对。

移植中确认的FlyDSL问题：

- 末K块放在动态循环外时，LLVM把其DQ FMA下沉到epilogue条件barrier之后，形成256 VGPR加50个溢出；改为所有K块在循环内、下一块下标钳位到KB-1后为242 VGPR、零scratch。
- 小rmem tensor的`fill(0)`会使`fly-layout-lowering`无诊断失败，改用显式`Vec.filled`存储。
- 只在每组`{MFMA, FMA}`后放`sched_barrier`时，LLVM把FMA排到同组MFMA前，FMA与产生partial的MFMA只隔一条MFMA，触发XDL冒险插`s_nop 4`。ATT显示VALU空档2.29%、MFMA忙碌90.97%（PyHIP 93.82%），8192预重排慢2.4%。在MFMA与FMA之间再加一道fence恢复严格交替后，差距缩小到0.3%-1.6%。
- B-scale沿用gfx950参考的内联`s_buffer_load_dword`；LLVM不跟踪其结果，所有使用都位于`lgkmcnt(0)`之后，由ISA等待审计确认。

### 2x4 r3上限消融

在2x4 r3上做ATT与同布局ELF补丁消融（被移除的指令替换为等长`s_nop`，其余地址不变）。8192预重排稳态MFMA忙碌91.86%，最大空档为LDS发射3.93%。同址冷计时（相对r3）：

| 变体 | 8192 | 4096 |
|---|---:|---:|
| 去DQ | +0.78% | +0.90% |
| 所有K块读K0数据（缓存驻留） | -0.24% | -0.17% |
| 去全部barrier | -8.55% | -8.43% |
| 去LDS读写 | +1.02% | +0.75% |
| 去VMEM | +2.84% | +2.20% |
| 仅保留MFMA与barrier | +4.10% | +3.29% |

结论：DQ与数据来源都不是r3主循环瓶颈，循环内全部辅助指令的上限约4%；barrier的4+4错相同步不可删除。逐WG `s_memtime`/`s_memrealtime`探针显示负载下中位SCLK为1.786 GHz而非1.85 GHz，主循环每K块4449周期（理想4096，92.1%）；4096方阵另受256个WG在80 CU上分4轮的20%尾轮损失。消融变体不是GEMM，只检查完整写出。

证据：[移植汇总](../gfx942_flydsl_ports/summary.json)、[8x1性能](../gfx942_flydsl_ports/perf_8x1_vs_pyhip.json)、[4x2-LDS性能](../gfx942_flydsl_ports/perf_4x2_lds_vs_pyhip.json)、[逐位比对](../gfx942_flydsl_ports/bitwise_8x1_vs_pyhip.json)、[ISA审计](../gfx942_flydsl_ports/audit/)、[2x4消融](../gfx942_flydsl_ports/ablation_2x4_r3/)。

## 证据索引

- [v25 源码快照](v25.py)及[版本与性能索引](manifest.json)
- [v11/v25 同址对比](v11_v25_3584_preshuffle.json)
- [冻结前三版本](../gfx942_blockscale_top3/manifest.json)
- [Cold-cache、生命周期与 DQ 消融汇总](../gfx942_blockscale_cold_audit/summary.json)
- [64 样本原始计时](../gfx942_blockscale_cold_audit/cold_audit_8192_64.json)
- [分阶段 MFMA 效率](../gfx942_blockscale_cold_audit/baseline/phase_efficiency.json)
- [动态 ISA 寄存器审计](../gfx942_blockscale_cold_audit/baseline/register_lifetime_audit.json)
- [A/B同路加载对照](../gfx942_symmetric_load/comparison.json)及[4096尾轮硬件证据](../gfx942_symmetric_load/physical_distribution.json)
- [4096/8192 L1/L2 PMC 汇总](../gfx942_symmetric_load/cache_pmc_summary.json)及[原始采集包](../gfx942_symmetric_load/cache_pmc_raw.tar.gz)
- [v32修复验收](../gfx942_blockscale_v32/summary.json)及[源码快照](../gfx942_blockscale_v32/v32.py)