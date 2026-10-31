# MI308X FP8 Block-Scale GEMM 优化记录

日期：2026-10-31。目标：8 waves、256x256 输出 tile、固定 M split=1，FP8 A8W8 + FP32 block scale，BF16 输出，cold-cache 吞吐约 500 TFLOPS。

## 主线与测量口径

最终主线是 **v1 -> v2 -> v8 -> v11 -> v25**。其余多数是分支实验，不是逐版叠加。

- v2 拆开 MFMA/DQ 依赖，分散预取，B-scale 改走 SMEM。
- v8/v11 改善输出合并和预取分布；v11 的后续 ATT 中，采样稳态 MFMA 忙碌比例达到 94.9%。
- v25 仅对预重排 B 微调阶段 barrier 位置，最终在目标验收 shape 上达到 cold-cache 513.7 TFLOPS。
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

## 当前设计

- **算术契约**：A 每行每 K128 一个 FP32 scale；B 每 N128xK128 一个 FP32 scale。先完成一个 K128 的 FP32 dot，再执行 `C = fma(partial, scale_A * scale_B, C)`，不跨 scale 块合并点积。
- **计算分工**：WG 为 512 threads / 8 waves，输出 256x256；每 wave 负责 32x256，M 拆为两个 16 行片段。每个 K128 依次处理四个 N64 packet，每个 M-split phase 为 16x64x128。
- **数据路径**：A 直接进入两套操作数 VGPR；B 用两套预取 VGPR 中转，再 `ds_write_b128` 到四个 N64xK128 LDS 槽，每槽 8 KiB，总计 32 KiB。A-scale 双缓冲进 VGPR，B-scale 双缓冲经 `s_load_dword` 进 SGPR。
- **重叠方式**：4+4 waves 错开 Memory/Compute；Memory 读取当前 `Q[q]`、提交 `Q[q+1]`、预取 `Q[q+3]`。wait 预算来自实际请求顺序，不用无条件排空代替流水线。
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

因此已有 A/B 同路加载比较，但尚未完成这类候选的 M/N 系统扫描，不能只凭单个 shape 否定对称方案。M/N 改变会影响 WG 数、分轮尾部、跨 WG 的 A/B 复用距离与缓存工作集。4096 方阵较慢的原因也不能直接归结为不对称 load 或 swizzle；需要保持 tile/K/算术不变，用相邻整轮 WG 数和相同 cold-cache 口径区分这些因素。

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

## 证据索引

- [v25 源码快照](v25.py)及[版本与性能索引](manifest.json)
- [v11/v25 同址对比](v11_v25_3584_preshuffle.json)
- [冻结前三版本](../gfx942_blockscale_top3/manifest.json)
- [Cold-cache、生命周期与 DQ 消融汇总](../gfx942_blockscale_cold_audit/summary.json)
- [64 样本原始计时](../gfx942_blockscale_cold_audit/cold_audit_8192_64.json)
- [分阶段 MFMA 效率](../gfx942_blockscale_cold_audit/baseline/phase_efficiency.json)
- [动态 ISA 寄存器审计](../gfx942_blockscale_cold_audit/baseline/register_lifetime_audit.json)