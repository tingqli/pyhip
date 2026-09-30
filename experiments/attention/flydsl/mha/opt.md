# BF16 DQ=DV=256 优化记录

> 2026-09-25整理：当前代码仅保留v73分页胜出路径与独立linear D256路径；P-exchange、旧v48及未采纳的实验开关已删除。
> helper抽到`_common`/`_d256`，公共分页ABI不变；下文v98“可选”等说明为当时历史状态。
> 本次保留路径13个实际ELF与清理前相同；[清理验证](../../../../mytest/mydata/qsa_cleanup_20260925_01/README.md)。

> 当前默认：按用户要求启用**v73 K+V DMA＋页表提前/ready合并＋S4 LGKM后移**组合，
> Q不做DMA；两种调度及page32/64/128均沿用公开入口。2026-09-23新目标为 **230 TFLOPS，尚未达到**。
> 本轮v74–99探索候选约221.5T；正式50样本遇到共同变速，配对中位改善3.85%，但全量中位验收未通过。
> 保留v98为显式可选流水线，**不替换v73默认、不声称230T达标**。
> 下述方案48成绩为历史完整验收，不是新默认成绩：
> BM128/BN64、8wave、M16、64KiB LDS，GPU2 Full的50样本中位数为
> **189.626T / 189.319T**（persistent/grid），Short-p64为193.899T / 193.575T。
> BF16功能回归41通过；完整6shape×2调度共600样本通过原数值检查。详情见文末。
> 失败模块已从可执行目录移除，冻结源、
> 原始JSON、ISA与ATT证据保留在本轮结果目录，未改写或挑选快样本。

## 2026-09-22：目标、基线与验收口径

- 目标：gfx942 / MI308X，BF16，`DQ=DV=256, H=24, HK=2`，有效吞吐至少 **220 TFLOPS**。
- 用户未额外指定序列长度；沿用现有性能集的 Full (`Q=10240, KV=2583, page64`)、Causal (`32768/32768, page64`)、Long (`20480/20480, page32`)、Short (`10240/2560, page32/64`)。小 shape 仅验证功能，不承诺 220T。
- 同时验证默认 persistent 与普通 grid；有效工作量为 `2*H*有效Q-K对数*(DQ+DV)`，causal 仅计可见三角区域，分母为完整 attention 调用时间。
- 保留原 `cudaPerf`：10 个独立 buffer、10 次预热、5 轮共 50 个样本，汇总全部样本的中位数；每个 buffer 先与独立 FP32 oracle 按 `rtol=atol=0.02` 检查，再验证重复逐位一致。不改变容差、计时边界和 AITER 候选。
- 原基线 Git `5f6544b502f6cb5dc0a3b47ba589ed0a859299ca`，BF16 源 SHA256 `ecd7598a97c8af3eae841650366f864ea5b48fd54edda6db39d5001d4baa13c8`。`PagedAttention(24,2,256,256,64,False)` 原先抛出 `NotImplementedError`，因此目标形状无可执行性能基线。
- Python：已有 `pyhip-mha-mi308` venv，Python 3.11.11、Torch 2.12.1+rocm7.2、FlyDSL 0.3.1；保留原生编译缓存。
- GPU0：MI308X / 80 CU / `0000:0a:00.0`。初查所有卡 PTL Disabled，未在此状态采性能。用户明确指定 `amd-smi set -F VECTOR,F8 -g 0`，随后只读核验 GPU0 为 **Enabled / VECTOR,F8**。用户补充开关命令为 `sudo amd-smi set --ptl-status 1/0`；已启用，不重复设置，也不改其他卡或频率/功率。

### 方案 1：BM128/BN32、MFMA16、单 K + 双 V

- 原 BM256/BN64 在 256 维下单 K+单 V padded LDS 已达 66,560 B，超过 64 KiB；每 lane 完整 O 需 128 个 FP32 寄存器，不能仅放宽 factory 检查。
- 新增独立 256 维设备 specialization，公共参数检查与调用 ABI 仍复用原包装层，128/192 内核不改。
- BM128/BN32、8 wave、原生 `16x16x16bf16_1k`，每 wave 16 行；O 降为 64 FP32/lane，Q 与 K 各 32 DWORD/lane。
- K 每 32-token/8-channel packet 使用 576 B pitch（512 B payload + 64 B bank-phase padding），总 18,432 B；V 使用两个 16,384 B 槽。总 LDS **51,200 B**。
- 保持 8-stage / 4+4 wave 错相：S0 读 K-low/V-global，S2 发布 V/读 K-high，S4 读 V-low/下一块 K，S6 读 V-high/发布 K；MFMA 与 softmax 放在交替计算阶段。
- V 继续使用真实 64-bit GLOBAL 地址，不改为 buffer descriptor；四个 16-lane 组的 softmax 使用独立 cross-lane 请求，与 V LDS 读取共同等待。
- 新增目标 GQA 性能项、page32/64/128 的 ragged/NaN-tail/前缀/LSE 回归以及两种调度的 stream/graph 回归。
- 状态：待编译、数值及性能实测；以上资源数为源码布局预算，不是已验证的机器码资源或性能结论。

### 编译检查 1

- 首次 256 specialization 编译失败：`global_load_dwordx4 offset:8192` 超出 gfx942 指令的 signed 13-bit immediate 范围。没有成功执行 GPU kernel，也没有性能成绩。
- 改为第二个 64-bit SGPR base 加 8192，GLOBAL 指令 immediate 保持 0；保留完整地址及原两次 128-bit 请求。
- 原始失败日志保留在 [v1-smoke.log](results/d256_20260922/v1-smoke.log)，失败报告 [v1-smoke.json](results/d256_20260922/v1-smoke.json)。

### 数值检查 2

- GLOBAL 修复后成功编译，但 smoke 的逐元素检查失败，`acc=0.87255686`（此指标为归一化平方误差，越小越好），未进入正式性能测试；[日志](results/d256_20260922/v2-smoke.log)、[失败报告](results/d256_20260922/v2-smoke.json)保留。
- 分离操作数检查：整 BN 下 `Q=0,V=1` 和 `V=1` 通过，`Q=0,V=random` 失败，先定位 PV 的 V 片段而非改变 softmax 容差。
- V 的 packed 连续加载先存为紧凑 `(4,2,8)` 寄存器张量，再以 `(4,8,2)` stride `(1,8,4)` 的 view 交给 MFMA；不能对 strided view 直接 store 后误认为原始 packed 顺序未变。待重验。

### 数值检查 3：smoke 通过，普通 grid 重复一致性失败

- V view 修复后 smoke 两调度均通过，`acc=2.6236e-6 / 2.6564e-6`；[报告](results/d256_20260922/v3-smoke.json)。小形状的单样本时间不用于 220T 验收。
- 新增 5 项功能回归：stream/graph 两项通过；三种 page 的 ragged/LSE 检查中，O/LSE 原容差通过，但普通 grid 重复逐位一致性失败。仍保留原 bit-exact 要求，不进入性能验收。[JUnit](results/d256_20260922/v3-contracts.xml)。
- 导出真实执行产物：persistent LSE `VGPR188, SGPR77`，grid LSE `VGPR180, SGPR61`，均 `LDS51200, scratch0, spill0`。见 [grid ISA](results/d256_20260922/v3-bf16_942_grid-lse.s)；非源文件相似性推断。
- 诊断：Q=0/V=random 重复稳定，Q=random/V=1 的 O 稳定，但随机 Q/V 普通 grid 有 BF16 ULP 级差异，LSE 差异约 1e-6。ISA prologue 的 opaque `v_max3/v_max` 在最后 QK MFMA 后很快读累加器；M16 短归约没有 M32 的独立计算间隔。
- 修复假设：把本地 max 改为编译器可见的 `maximumf` 树，让 MFMA→VALU hazard 能由 LLVM 插入间隔；不改变有限 score 的 max 语义。待重验，不把原因假设写成已证实结论。

### 数值检查 4 与环境检查

- 改为 `maximumf` 后，新增 **5/5 功能测试全部通过**，无 skip，96.27 秒；[JUnit](results/d256_20260922/v4-contracts.xml)。覆盖三页大小、ragged/prefix/NaN、两调度、causal/noncausal、LSE、重复 bit-exact、stream/graph。先前普通 grid 非确定性未再出现。
- 首次完整 Full 性能入口因 `aiter` 导入缺失而停止，尚未产生 kernel 性能样本。旧环境保留 `amd-aiter` 元数据，但 editable finder 指向已被清理的临时 checkout。[失败报告](results/d256_20260922/v4-full-a.json)。不通过删除 public candidate 或放宽检查掩盖该失败。
- 调优期间使用证据目录内的显式 `--native-only` 诊断范围：仅两种自有调度，独立 FP32 oracle/逐buffer检查/bit-exact/原 cudaPerf 均保留；结果明确标注未执行 AITER。正式入口候选规则不变。

### 性能基线与方案 5：扩大 BN，减少每 FLOP 的阶段开销

- 首个正确版本 Full、2 buffer/2 轮诊断：persistent **3822.444 us / 170.057T**，grid **3808.904 us / 170.661T**，未达到 220T。[原始结果](results/d256_20260922/v4-full-native.json)。固定门禁均通过；这是调优小样本，非最终 50 样本验收。
- 后续 [v4-baseline](results/d256_20260922/v4-baseline.json) 保存源快照及真实编译产物，用于可核验对照，不改写原结果。
- 将 BN32 改为 BN64，BM128/8-wave/MFMA16 不变；每计算阶段 MFMA 数翻倍，减少相同工作量下的循环、barrier、标量页表开销。
- LDS 改为 unpadded K32 KiB + V32 KiB，刚好 64 KiB；K packet 奇偶位 XOR 到 bank-phase 位，代替 padding。V 单槽，上一块 V 在下一轮 S0 发布，当前 V 暂存在寄存器；覆写在两组最后 V-high 读取之后。
- 页32显式处理一个 BN 横跨两个物理页；score/P/V 的 K32 packet 顺序及 tail mask 一起扩展。预计寄存器峰值增加，是否 spill/达到目标以接下来实际产物和运行验收为准。

### 方案 5 实测与方案 6

- BN64 smoke/Full 原检查通过，新增5项边界功能全通过（142.46秒）。Full 2×2诊断 **182.103T / 182.063T**；LDS65,536，VGPR256/252，scratch/spill0。[数据](results/d256_20260922/v5-bn64.json)。
- Long page32 2×2诊断仅 **154.719T / 180.594T**，仍未达标；不能用 Full 改善代替长序列结论。[数据](results/d256_20260922/v5-long.json)。
- 机器码显示 `maximumf` 生成 NaN 传播用的 compare/select，最大值归约大于必要的有限/已mask score 操作。下一步改用编译器可见的 `llvm.amdgcn.fmax3.f32`，保留 hazard 建模而不用 opaque 汇编；沿用有限 score/-inf 语义，继续检查 poisoned-tail。
- 编译器不提供 `llvm.amdgcn.fmax3.f32`，方案6未执行GPU；保留失败日志。改用标准 `llvm.maxnum.f32` 归约树，由 LLVM 选择可合并的硬件 max 指令，不假定专用 intrinsic 存在。

### 方案 6 结果与方案 7：直接 V 操作数

- `maxnum` 版本 Full 通过原精度/重复检查，约 **186.938T / 186.385T**，较 BN64 的 `maximumf` 小幅改善，仍未达到220T。[结果](results/d256_20260922/v6-maxnum.json)。
- 新候选回到 BN32：K 仍协作 GLOBAL→LDS，V 改为按 MFMA lane 直接 GLOBAL→寄存器，省去 V LDS store/read。S0 预取 V-low，S4 预取 V-high；请求均在消费前显式等待。
- 这是 LDS 流量与重复 GLOBAL/L2 请求的权衡，非已证实收益；保持公共 V64-bit 地址和 FP32/BF16 语义。当前候选 LDS18,432B，实际 VGPR/耗时待测。

### 方案 7 回退与方案 8：MFMA32 / K128 两阶段

- 直接 V 方案正确，但 Full 降到 **148.513T / 148.590T**，拒绝保留；[失败优化数据](results/d256_20260922/v7-directv.json)。减少 LDS 不代表总性能提升，重复 GLOBAL 请求成本不能忽略。
- 新方案使用 BM256/BN32、MFMA32，QK 的 K256 按相同归约顺序分为两个 K128 Compute；Q128 和 K128 在对应 Memory 阶段一起取数，避免 Q64 DWORD 与 O128 DWORD 全程重叠。增加 Q 的缓存读取，换取更低的 K/V 片段复制量及每wave32行。
- LDS 恢复单 K + 双 V ring，K 16,896B、V 2×16,384B，总49,664B；公共 Q/K/V/O 语义与 API 不变。此版本待实测。
- 方案8首测正确，但 Full 仅 **59.588T / 65.085T**。真实代码显示 `scratch48/16B`、`VGPR spill11/3`，原因之一是 LLVM 把固定 Q128 请求提升到循环外，使两个 Q 半片段同时长期存活；不是可接受的性能版本。
- 方案9在每次 Q128 阶段入口 pin row address，阻止循环不变量提升并缩短 Q 生命周期；继续核验实际 ISA 是否去 spill。
- 方案9仍只有 **74.414T / 74.053T**；persistent scratch44B/spill10，grid scratch0/spill0仍慢。按原始产物拒绝，不把 group LDS49,664B误写成scratch。恢复方案6的已验证 M16/BN64 源码，快照 SHA256 `726278645018a66f7f044df07d8c7425d1b4cca821710774e2ab57eefe0c1970`。

### 方案 10：V LDS bank-phase XOR

- 恢复版本的 V 布局中，四个16-lane组的 token-plane 基址相差4096B，在32 DWORD bank模型下同相位；这是可直接从地址验证的冲突风险，不当作实测counter。
- 对 V packet 的 channel 低位 XOR `(token_plane%4)*4`，同步修改协作写和 MFMA 读。每个32-key/256-channel平面仍完整覆盖，tile/page/数值顺序不变。用实测检验 LDS bank布局是否为主要限制。
- 方案10正确但 Full 降到 **142.564T / 142.503T**，与 bank-phase 风险模型不能直接等同净收益；回退该布局。[数据](results/d256_20260922/v10-vswizzle.json)。

### 方案 11：常驻 Q、K64 流式 M32

- 再试 M32：Q256 常驻，但每个 QK stage 内按 K64 顺序加载/消费/退休 K；不再每个BN重新读取Q。目的为去掉方案8/9的重复Q请求和全K128生存峰，保持原累加次序。
- K-high 最后块在 compute 内读取，prologue 在覆写单K槽前补全两组rendezvous；循环下一块K仍S6发布。
- 方案11仍spill53/51，Full **133.067T / 132.938T**，不保留。M16方案6源恢复为对照。

### 方案 12：QK 与 PV 的 wave 专用化

- 将一个 BM128 CTA 的4个wave专用于QK/softmax、另4个wave专用于PV，使用MFMA32；Q的64DWORD与O的128FP32不再由同一wave同时持有。
- QK(t)与PV(t-1)并行，K/V各32KiB；QK结束后K槽临时放P及FP32 correction/sum，PV读回后才发布下一K；旧V高半部读完才发布当前V。
- 每组stage的barrier次数相同，不采用spin或跨CTA通信。QK累计sum/max通过LDS传给对应PV行，保留原lazy-rescale和概率BF16舍入。
- 此为新候选，性能与功能未验收，不从资源预算推断达标。
- 方案12修复AST循环状态后数值通过且无scratch/spill，但Full **128.616T / 128.655T**，P跨组传递与阶段同步成本高；不采用。[结果](results/d256_20260922/v12-warp-state.json)。

### 方案 13：M16 同相、四次 CTA rendezvous

- 恢复最佳M16/BN64主体，取消4+4错相，所有wave同相执行；Memory之后只保留wait和compiler fence，每个Compute之后执行CTA barrier，循环barrier从8降到4。
- 单K/V槽的覆写仍晚于全CTA前次读取，S0 V发布在S3 barrier之前完成，K发布在回边barrier之前完成。测试同步减少是否超过原错相隐藏收益，不通过删必要wait实现。
- 方案13正确但仅 **146.015T / 145.331T**，拒绝保留并恢复错相版本。[结果](results/d256_20260922/v13-fourbar.json)。

### 方案 14：QK 两条独立 MFMA 链交错

- 最佳M16/BN64的ISA中存在先完成一个N16的全部K再算另一个N16的连续依赖链。改用显式ROCDL MFMA intrinsic，并用仅MFMA的scheduler fence保持每个K步的两条独立N16链交错。
- K顺序和FP32累加次序在每个输出内不变；不使用不透明MFMA汇编，保留hazard识别。
- 方案14 Full **186.421T / 185.986T**，无明显收益；仍低于目标。

### 方案 15：计算阶段 wave priority

- 单因素试验：Compute阶段设置 `s_setprio(3)`，Memory阶段还原0；仅改变wave仲裁，不改GPU频率/功率/PTL。检查是否缓解Memory伙伴对MFMA发射的干扰；不预设收益。
- 方案15 Full **183.991T / 183.452T**，低于方案6，去掉priority。

### 方案 16：四wave、小BM与单V槽

- 回到BN32，缩BM128→64、8wave→4wave，每wave仍16行；K/V搬运按256线程扩为4轮，V单槽。验证更小CTA与任务调度对吞吐的影响，不假定自动双CTA驻留（当前带padding LDS34,816B，仍大于半CU LDS）。
- 方案16验证正确，LDS34,816、VGPR210/204，编译器明确报告occupancy1，Full **102.659T / 102.556T**。下一步去掉K padding并采用 XOR 保持读写映射，LDS降到32,768B；persistent grid增加到2×CU，为两个4wave CTA提供任务，不声称物理驻留必然为2。
- 方案17双CTA条件下 Full **124.292T / 130.830T**，仍退化，未采用。

### 方案 18：专用 QK/PV wave 阶段错开

- 方案12的两个wave组在相同stage同时compute/同时memory，没有隐藏DS等待。重排为QK-low与V-low读重叠、PV-low与K-high读重叠、QK-high与V-high读重叠、PV-high与softmax重叠；P传递/nextK发布仍有独立CTA同步。
- 同一输出的数学顺序保持不变，意在减少wave专用化方案的空闲间隔；待测。
- 方案18正确，Full **153.797T / 153.254T**，未达标，不保留。

### 方案 19：四wave M32、显式 AGPR PV

- 利用仓库已有 `=a` MFMA内联约束，把宽O累加器放到AGPR；4wave BM128，occupancy目标1，不强制8wave block的寄存器预算。
- 保留Q常驻与K64顺序流式读取，验证是否消除M32的scratch，同时检查AGPR读写开销。只在真实机器码和运行通过后评价，不把AGPR类别本身视作性能保证。
- 方案19AGPR128、总VGPR计数368、零scratch/spill，但Full只有 **113.441T / 113.445T**，不采用。

### 方案 20：单 rendezvous 的四wave软件流水

- 针对已测的barrier和直接V路径代价，建立BM64/BN32的顺序online-softmax软件流水：K双LDS槽，V-low/V-high直接GLOBAL预取，QK与PV之间保持明确VM请求年龄，下一K的DS读与当前PV-high重叠。
- 每BN仅一次CTA rendezvous；不使用8stage的全阶段同步。K双槽32KiB，persistent发出2×CU个CTA。仍使用相同公开包装和严格参考检查。
- 方案20 Full **133.531T / 143.627T**，persistent仍spill2，未采用。恢复方案6作后续对照。

### bank模型纠正与方案21：M16调度预算

- 按gfx942 `ds_read_b128`实际lane服务组（0–3与20–23等）离线枚举：原V布局每bank1次，方案10 XOR后每bank2次。此前“不同16-lane组同bank即冲突”的简化模型不成立；方案10的性能退化有明确地址模型反证。原K XOR4布局在该模型下无冲突，不再盲目改swizzle。
- M16 MFMA时间窗短于旧M32。试验缩短每计算stage的VALU调度预算，EXP按8组×2而不是32组×1，避免过量hint限制LLVM排程。保留数学次序和barrier。
- 方案21 Full **184.421T / 184.283T**，恢复原hint。

### 方案22：页地址无符号算术

- 有效tile/page范围由接口非空KV与循环边界保证非负；在页号索引和页内余数处使用Uint32，避免有符号除2次幂/取余展开为多条SALU符号修正。物理tensor基址仍64位，不截断。
- 方案22 Full **186.908T / 186.632T**，基本中性。原128/192对照仍为239.946/246.492T，原Long128为268.157T，当前设备未出现普遍性能下降；不将D256低吞吐归咎于未测的时钟因素。

### 方案23：顺序 M32、严格短生存期

- 四wave BM128/BN32，Q常驻、K64分段，取消lagging softmax使前一score不与当前QK同时存活；V-low/V-high分别读取/消费/退休，避免两半V同时占寄存器。
- 这是无多重预取的资源下界候选；确认零scratch后再决定是否添加局部预取。不放宽精度或直接忽略spill。
- 方案23数值通过但仍spill17/12，Full **152.969T / 160.425T**。方案24把Q也改成K64同时消费的小片段，并pin每chunk的Q地址，消除Q64DWORD长期占用；是否有利待测，重复Q请求仍是真实计时代价。
- 方案24零scratch，但重复Q加载导致 **68.468T / 69.276T**，拒绝。方案25恢复常驻Q顺序版本，PV显式AGPR/occupancy1，先看无spill是否能补偿驻留降低。
- 方案25 Full **110.502T / 105.984T**，虽无spill但AGPR+VGPR总量过高，不采用。

### 方案26：M16 双查询组与四分段 PV

- 每wave处理两个16行组（BM256/8wave），共享K和V寄存器片段，提高访存复用；QK分两次N16，PV按四个64维块依次处理，避免一次持有全V128片段。
- 前一score不跨下一QK，online-softmax先完成，保持O总128FP32/lane、Q总64DWORD/lane。此资源预算必须由编译产物验证；仍以原容差/重复一致性为前置门槛。
- 方案26仍spill40/150，Full **128.332T / 117.676T**，不采用。

### 方案27：V 暂存不跨 softmax

- 方案23真实ISA的spill对象是Q的前三个128-bit packet；V-low提前读取与FP32 scores/exp/pack临时量重叠。将V-low移到P打包完成后，再以compiler fence阻止下一V-high提前，测试是否可去掉该具体寄存器峰值。
- 方案27只减grid spill12→10，Full **151.061T / 157.801T**，仍拒绝。

### 方案28：双查询组 K128 小片段

- 在方案26中把每个K256片段拆成两个K128顺序消费，保持两query组共享K，同时减少16个DWORD临时寄存器；Q和O仍常驻。验证是否足以跨过VGPR256的spill界限。
- 方案28仍spill39/45，Full **125.595T / 127.694T**，不采用。

### 方案29：四阶段、专用组搬运、独立 P

- QK/PV专用wave改为BN32，K16KiB、V双槽32KiB、独立P8KiB与metadata2KiB，共59,392B。
- 仅QK组搬K，仅PV组搬V；四phase中QK(t)与PV(t-1)的LDS读交叠，P写与PV计算交叠，nextK/V发布在phase末统一同步。减少原专用组方案的重复协作搬运与P/K别名同步成本。
- 全组barrier数一致，末尾PV排空与LSE保留；仍需GPU验收。
- 方案29多tile数值失败（单tile通过），后续pin诊断又触发连续VGPR约束不足；未计性能，也未将其作为备用路径。

### 方案30：双查询组 O 原位更新

- 将softmax的correction计算与整向量O更新解耦，先完成两组P打包再rescale；PV使用tied-accumulator MFMA以减少旧/新O重叠生存区间。只作为资源试验，正确性必须重新检查。
- 方案30仍spill42/39，Full **110.929T / 123.685T**，不采用。

### 方案31：Q 小片段 AGPR 缓存

- 不再把全部O放到AGPR；仅将Q的一半32DWORD放AGPR，每K64前读回16DWORD，代替方案23的Q scratch spill。保留2wave预算与原总算量，待检查AGPR合计是否仍允许目标驻留。
- 方案31非有限输出，未进入计时，拒绝保留。

### 方案32：M16 compute窗口分散 VMEM

- 回到无spill M16/BN64；把4个V请求从S0移到QK-low、4个nextK请求从S4移到PV-low，按8MFMA:1VMEM分组。检验较短M16 compute窗口是否更适合混合发射，而非沿用M32的严格阶段分离。
- 方案32 Full **185.856T / 185.513T**，未改善。

### 方案33：M32 顺序 V64

- 方案23的V128片段占32DWORD、再叠加Q64/O128和地址变量，超过256预算。把PV按V64×4分段，每段只16DWORD V，再测试是否能在Q常驻条件下零spill。额外GLOBAL等待的成本完整计入。
- 首次方案33在入口发现其他用户PID4154272驻留而被门禁拒绝，未采性能、不终止他人进程。用户随后明确允许 **gfx/UMC<3%、允许驻留进程的非独占协议**；后续单独标签记录，不与此前无其他进程样本混合。PTL仍Enabled/VECTOR,F8，VRAM<20%，每场保留进程/利用率快照。
- 方案33低负载轮正确，grid零spill、persistent spill1；Full **156.078T / 162.452T**。V64可降寄存器，但直接GLOBAL延迟仍高。

### 方案34：M32 K64/V64 的 LDS 细粒度错相

- BM256/BN32、8wave，Q常驻、K64四段QK、V64四段PV；当前tile softmax，不把前一score带过下一QK。K16KiB+双V32KiB共48KiB。
- 各段memory/compute错相，保持每时刻窄操作数，V仍协作GLOBAL→LDS，避免方案33每wave独立GLOBAL读；nextK/alternateV仅在所有旧读者结束后发布。
- 资源/性能尚待验证。
- 方案34 persistent smoke通过，但普通grid精度失败；追加MFMA退休间隔仍同样失败，反证不是该间隔能修复的问题。未计性能，不作为交付路径。

### 方案35：跨 wave 分摊 DQ/DV

- 两组wave负责同一组查询行的两个DQ128部分，LDS交换FP32部分score后相加，再各自负责不同的DV128。没有重复QK；每wave只需Q32DWORD、O64FP32，避开DQ/DV同时256的寄存器峰。
- K16KiB读完后与32KiB部分score区复用，V独立16KiB，共48KiB；所有覆写都在相关消费者barrier之后。QK的两个K128部分相加改变关联顺序，须通过原FP32元素容差，不宣称bitexact于其他实现。
- 方案35正确但Full **93.670T / 94.115T**，部分score LDS交换成本高，未采用。

### 方案36：GQA head launch次序

- 恢复最佳M16/BN64主体，把逻辑head映射成 `(head%HK)*(H/HK)+head/HK`。对任意H可整除HK的输入，这是H域双射，输出仍用原public head编号。
- 对H24/HK2使相邻launch交替KV头，可能改变各XCD的KV复制与复用；不假定blockid等于物理XCD，不改变K/V布局或有效FLOP分母。
- 方案36 Full **186.167T / 186.177T**，Long **160.252T / 197.089T**，仍未达标。任务顺序不是Full瓶颈的充分修复。

### 方案37：细粒度 LDS 的概率打包约束

- 方案34全一操作数显示persistent正确，grid在KV64固定输出0.875、KV129固定0.9375，单tile正确；这不是随机噪声。回退无效退休延时，改用已有稳健的`v_perm_b32`概率打包，禁止编译器跨循环重新组合BF16概率寄存器。
- 方案37仍失败。导出实际grid ISA后发现明确hazard：`global_load_dwordx4 v[130:133]` 后紧接 `scratch_store_dwordx4 v[130:133]`，在 `vmcnt(0)` **之前**。opaque异步结果被RA提前spill，固定丢值不应归咎于MFMA数学或timer；拒绝所有有此hazard的候选，不通过放宽容差规避。

### 方案38：跨 wave score 的 plane 布局

- 方案35无spill，但score以每lane64B stride存放，使b128服务组重复命中同bank。改成4个8192B plane，每lane相邻16B，paired-wave读取也按plane。保持FP32score以及工作量完全不变，只修复该交换布局。
- 方案38正确且比方案35快，但 Full **111.650T / 112.233T**，仍拒绝。

### 方案39：PV 分片时 rescale

- 双查询组试验把整O64向量的条件rescale拆入每个4元素MFMA累加片段；correction=1时乘1保持值，避免大向量分支合流引起旧/新O同时存活。使用compiler-visible MFMA，不再opaque tiedMFMA。新增乘法成本仍计入。
- 方案39仍spill且 Full **115.099T / 106.436T**，不采用。

### 方案40：专用 PV 波分两次 V128

- 方案29避免将128个BF16片段跨runtime分支整体携带；在PV计算phase内分两次读取/消费V128。保持同P/同score/同总O，目的为消除大V生命周期和opaque spill危险。
- 方案40精度恢复且零spill，但Full **121.547T / 121.899T**，未采用。

### 方案41：四个独立 O32 状态

- 细粒度M32路径不再每次PV把quarter插回整个O64向量，而是四个独立32FP32累加器，PV直接更新对应状态；消除vector insert造成的老/新宽O干涉。总存储仍128FP32/lane，未降精度。
- 方案41仍非有限输出，拒绝。

### 方案42：四wave M32、协作 V64 LDS

- 以方案33（普通grid零spill）为基础，把每wave四次V64 GLOBAL同步读改成协作GLOBAL→单V LDS槽，再四次V64 LDS消费。K单槽16KiB、V单槽16KiB；下一K/V只在当前所有读者完成后加载/发布，两个CTA同步保证边界。
- 保持Q常驻、K64分段、V64分段、原容差和FP32累加。尚待实测。
- 方案42 Full **162.279T / 172.011T**；grid零spill，但每个K64/V64即时DS等待暴露。方案43将下一片段DS读前移到当前MFMA前，等待留在消费前；观察增加一片段预取生存期是否仍能控制spill。
- 方案43 Full **164.507T / 173.090T**，仍未达目标。

### 方案44：Q64 LDS 缓存替代 scratch

- 细粒度M32：把Q256中的64维放入32KiB LDS（每lane16DWORD），仅其余192维48DWORD常驻。K16KiB、单V16KiB，总64KiB；每轮QK-low从LDS读取Q64，不从GLOBAL反复读取。
- V-next预取延到PV阶段，显式等待后作为loop状态到下次S0发布，降低QK/softmax峰值；O保持四个独立32FP32状态。目标为彻底去spill，是否达到由实际ISA检验。
- 方案44终于两调度零scratch/spill并数值通过；Full **166.381T / 166.366T**，未达标。方案45在相同阶段边界多准备一个K64/V64片段，验证较长LDS预取距离是否改善。
- 方案45又产生spill27/18，Full **129.381T / 138.663T**，退回44。

### 方案46：Q-cache M32 三次同步

- 将44改为同相执行，仅V发布后、K全部读完后、loop回边三个CTA barrier；K64/V64内部仅显式wait+compiler fence。K/V覆写不越过对应全组读者，不保留不必要的段间CTA同步。
- 方案46 Full **161.051T / 161.316T**，未优于M16最佳，恢复M16无spill基线。

### ATT 定向分析与方案47：V 发布移入 QK-low

- 最佳M16 ordinarygrid的ATT-only采集完成。初次因Torch wheel与系统两份profiler SDK重复注册失败；只用用户缓存目录内root别名与`--realpath`统一到wheel SDK后成功，未使用PMC/未改变PTL。详见 [ATT分析](results/d256_20260922/att-analysis/README.md)。
- 一组稳态物理SIMD配对wave每BN周期5,916cycles，128MFMA/wave；16cycle发射模型覆盖69.236%，并非实测硬件busy计数。未覆盖1,820cycles，其中S0/S6发布相关handoff约1,284；steady VMEM completion wait仅4cycles/次，不能称HBM饱和。
- V/K LDS写只有31.25%落入伙伴MFMA窗口，LDS读88.3%已被覆盖。将V(t-1)发布从S0移至S1，按8MFMA:1DS-write交织；S2保留LGKM等待，S4读V前两组均已完成写入。单V覆写仍晚于前轮S6读者。
- 方案47 Full **177.030T / 177.314T**，移到Compute反而与伙伴DS争用，不采纳。

### 方案48：Memory 内按4read:1write分散发布

- 把S0的4个V写均匀插入16个K读之间，S6的4个K写均匀插入16个V读之间，避免先读满再集中发布造成的DS队列尾部；不增加请求、不改数据量，明确在K写之前等待VMEM。
- 方案48 Full改善到 **190.197T / 189.873T**，零spill，但仍未达220。

### 方案49：V 发布分摊 S0/S2

- V的4个packet写分为S0两次、S2两次，每8个K读后发一个写；S4读V在两组S2及S3同步之后，保证完整发布。目的使原S0过长、S2偏短的两个访存阶段更平衡，V请求和算术均不变。
- 方案49为 **189.032T / 189.042T**，未优于48，恢复48布局。

### 方案50：半轮 phase rotation

- 在QK/PV交界和回边各加一次双方都执行的rendezvous，交换哪组wave先进入下一半轮，保持一拍错相和所有读写边界。该额外同步用于检验长期phase失衡，不承诺性能；不改计算、请求数或FLOP。
- 方案50 Full **162.870T / 163.009T**，回退到48。

### 方案51：V GLOBAL 直接到 LDS

- 离线assembler确认gfx942支持 `global_load_lds_dword`（完整64-bit SGPR base），不假定gfx950的x4DMA支持。
- 每tile16个DWORD-DMA/线程，分S0/S2各8条并穿插K LDS读，替代4个GLOBAL128bit+4个DS-write，降低ATT指向的LDS发布压力；VMEM在S2明确排空，S4前全组发布完成。尾页及page32两物理页地址按原tile处理。
- 请求数增多可能抵消省去DS写的收益；保持未达标状态直到实测。
- 方案51正确但 **186.070T / 179.138T**，DMA请求/地址开销抵消收益，恢复48（190T）。

### 方案52：K 发布分摊 S4/S6

- nextK在S2读取，四个packet在S4/S6各发布两次。M16 full K片段已在两组S2全部读完，伙伴S3只消费寄存器，因此S4覆写安全；不适用于在S3内部继续读LDS的流式K变体。
- 预取状态会更长，必须核验spill；该布局/阶段合同仅用于本候选。
- 方案52 Full **182.756T / 182.494T**，回退48。

### 方案53：LDS 发布位宽诊断

- 只把热点packet的b128写拆成四个b32，地址/payload一致；用真实计时确认ATT的b128发布92–96cycle stall能否通过更细服务粒度改变。请求数增多完整计入，不把较窄指令数视为收益。
- 方案53 Full **158.448T / 158.176T**，拆窄退化，拒绝。

### 方案54：K/V 双 DMA

- 在V DMA试验基础上，K改为buffer DWORD直接LDS，按目标XOR布局反算每DWORD全局地址；S4/S6各8请求，S6等待所有DMA完成，再进入下一QK。
- 保持K/V数据布局、概率/累加语义与原timer。逻辑请求数显著增加，仍须实测决定，不据“去DS写”宣称加速。
- 方案54正确但 **173.184T / 180.176T**，未达标，不采用DMA。

### 方案55：Q-cache M32 合并为八阶段

- 基于正确零spill的44，把四个K64计算合成两个K128、四个V64合成两个V128，O状态恢复两段64FP32，期望每Compute窗口从8MFMA扩大到16MFMA。
- Q64仍放LDS、Q192常驻，QK用compiler-visible M32 intrinsic，保持K累加顺序；singleV-next在下一S0发布，K-next仍S6发布。是否因宽片段再次spill需实测。
- 方案55再次spill24/18，Full **165.735T / 165.628T**，不采用。

### 方案56：四wave、大寄存器池软件流水

- BM128/BN64，4wave、occupancy1，Q常驻VGPR，O分两段AGPR；K双LDS槽64KiB，V直接GLOBAL预取，单次tile发布同步。
- 与之前AGPR/BN32方案不同，BN64扩大每次QK/PV工作窗口，软件预取争取隐藏访存；仍检查实际AGPR/总寄存器与无scratch。该候选未验收。
- 方案56smoke数值失败，未计性能。

### 方案57：八wave、四独立O32、每tile两次同步

- 从方案42正确顺序M32改为BM256/8wave，协作K/V各8DWORD，当前K64/V64片段16DWORD；O四段独立32FP32避免宽向量插入，Q256常驻。
- P打包完成后才发下一K/V预取，与当前四次PV重叠；所有旧读者结束后统一发布，两个CTA同步/BN。保持原数值顺序，测实际live-range是否符合预算。
- 方案57仍spill84/11，Full **151.336T / 166.140T**，拒绝。

### 方案58：MFMA 直接读取 AGPR Q64

- 离线assembler确认gfx942 BF16 MFMA的A/B操作数可来自AGPR。将Q64通过显式`v_accvgpr_write_b32`写入16个AGPR，并直接作为MFMA source，不再像失败方案31用空asm伪造搬运。
- Q192仍VGPR，O全FP32；释放Q-cache32KiB LDS。以真实产物检验总寄存器、spill和数值，不把指令可编译等同可用。
- 方案58编译器仅分配4AGPR且重新spill39/46，Full **125.258T / 115.153T**，拒绝并恢复最佳M16。

### 方案59：LDS 写优先

- 从48改为先发布4个packet再发16个operand读；与先读后写、4:1交织构成同请求数的顺序对照。依据ATT写尾部主导证据，测试提前退休staging寄存器和请求先后顺序的效果。
- GPU0首次59入口显存153675/196592MB，门禁拒绝；后续单次全卡快照GPU0/1繁忙，GPU2–7已Enabled/VECTOR,F8且gfx0/UMC0、VRAM284MB。切换GPU2的新场次单独记录BDF，不与GPU0结果作同址或同卡配对比较。
- 方案59 GPU2 Full **189.397T / 189.949T**，仍未达目标。

### 方案60：12wave QK/PV-low/PV-high 专用化

- 在正确专用组40基础上，把PV拆为两组各4wave，每组只保持O64FP32/V128片段；4QK+4PVlow+4PVhigh，共12wave。V只有low组协作搬运，不重复global请求。
- 理论需≤170总寄存器才能三wave/SIMD；明确以actualVGPR/spill为准。仍四阶段、P/meta单槽/双V，数值/API不变。
- 方案60 GPU2 Full **126.376T / 126.878T**，未达标。

### 方案61：BN64 split-DQ/DV

- 将正确但同步成本高的35/38扩大BN64：K32KiB、V32KiB，K读完后分两轮在32KiB plane交换low/high部分score。PV32MFMA窗口，QK不重复；实际关联顺序按原容差验收。
- 六个明确的CTA读写边界，每laneQ32/O64，预取nextKV放在score/P退休之后。验证增大工作粒度能否覆盖交换成本。
- 方案61正确但Full **107.142T / 107.759T**，不采用。

### 方案62：operand read2st64_b64

- 恢复48的M16主体，对成对16B operand packet分别用两条`ds_read2st64_b64`读取低/高8B并恢复原DWORD顺序，ST64-B64 offset单位为512B。
- 不减少payload或声称一条read2等于连续b128。CPU枚举所有512线程/两half共131072DWORD映射一致；检查实际服务组和wait成本是否不同，仍需GPU原精度验收。
- 方案62 Full **84.288T / 84.232T**，严重退化，不采用。

### 方案63：AGPR 内原位 rescale

- 之前AGPR方案的总寄存器高，是rescale/epilogue把整O搬回VGPR而扩展live range。新候选O四段C32保持AGPR，rare rescale逐标量用一个scratch VGPR读/乘/写回，正常路径仅AGPR MFMA；输出也只读4个值再RNE存出。
- Q64DWORD常驻VGPR，K/V窄片段16DWORD，LDS32KiB，8wave目标2驻留；实际AGPR/总寄存器由ISA验收，保留零spill硬性要求。
- 方案63smoke元素检查失败，未计性能，不采用。

### 方案64：QK/PV 专用 wave 完整错相交接

- BM128/BN64，4QK+4PV。QK-low对应V发布，QK-high对应V-low读取，PV-low对应nextK发布，下一tile的K-low读取对应前一tile的PV-high；P/meta在K区短暂复用，PV读完后才覆写nextK。
- 明确八个双方相同的phase，singleV覆写前已将前一high全部读到寄存器；不让Q与O共占同一wave。完整数据契约仍按原oracle验证。
- 方案64正确但spill35/19，GPU2 Full **114.988T / 124.725T**，不采用。

## 交付收敛（2026-09-23）

- 恢复方案48设备主体，删除八个未采纳的D256实验模块，公开包装仅分派到
	[mha_pa_bf16_256_942.py](mha_pa_bf16_256_942.py)。D128/D192设备核心保持原样。
- 原v48冻结源SHA256 `ff447622e52f62a5a3b73921481a5fb9b9786e42e0c759b8e946fd818b4a8b15`。
	收敛版仅清理未用import、明确包/脚本导入、修正文档措辞；最终实际ISA仍须复核。
- 正式功能范围增加5个D256 stage-boundary参数、3个D256 layout参数、2个D256 stream/graph参数；
	H24/HK2、DQ=DV256明确进入测试，不降低元素容差或重复逐位一致性要求。
- AITER在已有环境中的editable路径已失效；公共CLI仍明确报错而不是静默删参考。
	本轮性能证据使用显式native-only驱动，独立FP32参考、所有buffer校验、原cudaPerf全部保留。
- 历史artifact收据若同一factory同时缓存smoke/full，原导出标签可能沿用首个case名；
	原始ELF/IR未改动。最终驱动按真实compiled signature的MAX_Q筛选并记录page_count，避免错误标签。
- 最终性能范围仍为用户指定的BF16维度/头数下明确列出的Full/Causal/Long/Short；
	`passed`表示数值正确，不表示达到220T。不同GPU、不同负载政策的样本不合并。
- 已把缺失AITER源码恢复到用户缓存目录的固定commit `bde46043bcf08e41ac40395a18369ab6309153ca`，
	CK submodule `15e12dd7f25ee583617c78f66cb502ff9916585f`。仅对参考进程设置PYTHONPATH及venv PATH，
	保留系统/venv Torch、FlyDSL、Triton版本，不修改旧editable finder或全局Python配置；import已通过。
	最终参考比较若该D256 specialization不可用会明确记录失败，不修改native数值/吞吐口径。
- [CPU布局审计](results/d256_20260922/final-layout-audit.json)通过：V写精确覆盖LDS字节32768–65535，无重叠/越界；
	M16的query坐标是`lane&15`（不是M32的`lane&31`），四个lane组恰好覆盖BN64的全部key；
	remaining1–64尾mask、page32/64/128投机page clamp及bottom-right causal predicate枚举一致。
	此为地址/谓词证明，不替代GPU数值及同步回归。
- 最终BF16功能回归 [JUnit](results/d256_20260922/final-bf16-tests.xml)：**41通过、0失败、0跳过**，1193.881秒，
	包含原128/192与新增256的10个参数。源码设备路径没有使用失败实验模块。
- 首次正式10-buffer/50样本轮完成smoke与Full后，在Causal采样前显存门禁失败：39430/196592MB，
	略超20%，因此未继续计时；[不完整报告](results/d256_20260922/final-native-gpu2.json)与所有已完成样本保留，不宣称整轮通过。
	诊断为FP32参考/clone验证留下的unused PyTorch allocator块。后续协议只在验证/预热后、采样前调用一次
	`empty_cache()`释放未使用块，不释放活跃输入/输出、不改变地址、计时器或样本数，也不放宽20%门槛。

### 最终完整 native-only 验收

来源：[完整JSON](results/d256_20260922/final-native-clean-gpu2.json)、
[离线审计](results/d256_20260922/final-native-clean-gpu2-audit.json)。
**不是AITER对比轮**；每buffer独立FP32参考及两次额外bit-exact复测、每候选10buffer×5轮=50event、
warmup10、repeat1、原cudaPerf完整调用；所有600样本保留，包括长尾，不筛选最快区间。

设备GPU2 / MI308X / gfx942 / 80CU / BDF `0000:a4:00.0`，PTL全场快照均Enabled/VECTOR,F8。
用户授权低负载非独占协议；14次entry/prepared/sampling/exit快照gfx0–2%、UMC0–1%、VRAM≤15.736%，
不把离散快照当全程无干扰证明。长序列的慢样本没有丢弃，也不在未采运行中频率的情况下归因于时钟。

| 场景 (Q/KV/page) | persistent µs | persistent TFLOPS | grid µs | grid TFLOPS | 最大 acc |
|---|---:|---:|---:|---:|---:|
| Smoke (65/129/64) | 21.560 | 9.558 | 21.440 | 9.611 | 2.548e-6 |
| Full (10240/2583/64) | 3427.981 | **189.626** | 3433.542 | **189.319** | 2.779e-6 |
| Causal (32768/32768/64) | 94245.068 | 140.002 | 94671.368 | 139.372 | 2.379e-6 |
| Long (20480/20480/32) | 75064.301 | 137.321 | 75016.422 | 137.409 | 2.786e-6 |
| Short-p32 (10240/2560/32) | 3361.861 | 191.633 | 3366.761 | 191.355 | 2.797e-6 |
| Short-p64 (10240/2560/64) | 3322.581 | **193.899** | 3328.141 | **193.575** | 2.796e-6 |

结论：数值/重复性/门禁审计通过，但**10个非smoke候选均未达到220T**。
Full persistent距离220T仍需约16.0%吞吐提升（约13.8%时延降低），不能标记性能目标完成。

- 最终D256源码SHA256 `8f8e976d55f89ffae69c65d70e60aedbd98edf834d3d2bb3176e08b4dcf68279`。
- 12个实际artifact全部scratch0、VGPR/SGPR spill0、AGPR0、LDS65536B；VGPR250–256。
	Full两调度ELF SHA与方案48完全相同，证明收敛清理未改变Full设备机器码。
- 本轮旧export虽按MAX_Q筛选，Full/Short-p64还共享MAX_Q，NP41/40被归到首个case名；
	**原收据/ELF不改写**，独立审计的`verified_artifact_mapping`按MAX_Q+NP完成唯一匹配。
	驱动后续导出已同时筛选NP，避免再出现该标签问题。

### 恢复 AITER 后的独立 Full 对照轮

[完整报告](results/d256_20260922/final-aiter-full-gpu2.json)、
[独立审计](results/d256_20260922/final-aiter-full-gpu2-audit.json)。
同GPU2/PTL、同Full输入形状、10buffer/50样本、原FP32元素检查与重复bit-exact。
三候选在同一轮交替顺序测量；AITER为公共`flash_attn_varlen_func`，prepared linear KV，布局准备不计时，
自有内核直接读取SHUFFLE-5D分页KV。

| backend | 中位数 µs | 有效 TFLOPS | acc |
|---|---:|---:|---:|
| 默认 persistent | 3417.163 | **190.226** | 2.779e-6 |
| 普通 grid | 3422.542 | **189.927** | 2.779e-6 |
| AITER public router | 6424.662 | 101.178 | 1.885e-5 |

- 自有Full分别约为本轮AITER的1.880×/1.877×吞吐，但**仍未达到220 TFLOPS**。
	220是TFLOPS目标，绝非220微秒门槛。
- 此独立对照轮不替换前述完整6shape表，也不从两轮中拼接较快行。
- AITER/CK固定SHA均与环境恢复记录一致；所有候选50个原始样本、buffer地址、dispatch与门禁保存。
- 全部任务结束时未提交/推送Git，未终止他人进程、未变更GPU频率/功率。

### 当前 v48 ATT 采集与 UI 交付（2026-09-23）

- 按用户要求采集当前保留的v48，Full Q10240/KV2583、DQ=DV256、H24/HK2、page64、普通grid，
	无LSE。只采第3次dispatch，SE0/CU1/四SIMD，256MiB ATT buffer；不启用PMC/ATT计数器。
- GPU0/1正在繁忙，使用GPU2/BDF `0000:a4:00.0`；entry/prepared/exit均gfx0%、UMC0%，
	VRAM284/2229/4629 MB，PTL Enabled/VECTOR,F8，未设置任何频率/功率/PTL。
- 实际捕获code object 6的SHA与当前Full/grid验收ELF完全相同；VGPR252、SGPR53、LDS65536B，
	AGPR/scratch/spill均0。原FP32元素检查前后通过，`acc=2.727154653547892e-6`，三次重复逐位一致。
- UI完整复制到当前MHA目录，入口：[filenames.json](ui_output_agent_63554_dispatch_393/filenames.json)，
	同目录附带[统计CSV](stats_ui_output_agent_63554_dispatch_393.csv)。200文件/113,796,321bytes，逐文件SHA一致。
- 192/192wave全部拼接完整，3,062,016条动态指令与code-map hits一致；无DataLost/溢出诊断。
	当前UI为ISA映射，没有源码行映射。原始ATT、产物和遥测另保留，不覆写旧v6 trace。
- [采集说明](results/d256_v48_att_20260923/README.md)、
	[数值/身份](results/d256_v48_att_20260923/driver-result.json)、
	[完整性/复制审计](results/d256_v48_att_20260923/ui-copy-audit.json)。
	此次是capture-only，不是新性能验收，**220 TFLOPS仍未达到**。

### v48 bank-conflict 离线核验

- 对本次dispatch393对应的冻结源AST地址枚举所有512线程/half/packet，按gfx942 b128实际服务分组检查。
	K读2048组、V读2048组、K写256组、V写256组、C-shuffle读512组，每bank最多1个不同DWORD，均无模型冲突。
- C-shuffle b64写512组在连续16-lane模型下同样无冲突；没有独立硬件验证该写指令分组，保留条件结论。
	仅移除K XOR的CPU负例全部为2-way冲突，证明模型能识别冲突，实际内核未改动。
- 当前ATT内部循环平均发射stall：S0 K读12.85/V写27.98、S2 K读21.09、S4 V读22.68、
	S6 V读17.21/K写27.51 cycles/次。高stall常在LDS写后第一条读，不能直接解释为bank conflict。
- 全192wave动态指令、code-map与CSV统计一致；没有bank/replay counters，因此不宣称硬件实测零冲突。
	未采PMC、未运行GPU、未修改kernel/策略。详见[bank检查](results/d256_v48_att_20260923/bank-conflict.md)及
	[原始枚举/统计](results/d256_v48_att_20260923/bank-conflict-audit.json)。

### LDS访问量与16-stage建议（离线分析，非新候选实测）

- 本次ATT核对每wave/BN：64条b128读、8条b128写、6条bpermute、128条MFMA、8次barrier；
	每CTA为512KiB读+64KiB写。各wave负责16行query，所以唯一K/V各32KiB被8个wave各读一次，
	不是单wave内冗余读，也不是scratch spill。
- 当前v48选定32个内部周期的每物理SIMD wave对平均5783.25–5783.375cycles/BN；
	16cycle MFMA服务窗模型覆盖约70.825%，不是硬件busy counter。
- 将16R/32MFMA拆成8R/16MFMA不改变流量/计算比；若8→16同时翻倍CTA barrier/wait，
	不保证改善。真正收益条件是窄operand降低活跃寄存器并改善发布/消费重叠，不能只拆函数名。
- 可验证的两个首尾精简点：prologue的V0重复发布、末循环无后继QK的K[last]预取/发布；
	它们各只省一个tile的请求，不是每BN都省。未在本次修改。
- 更大收益需减少8wave重复读取，如提高每wave query复用，但已有M32/双query组受Q/O寄存器压力限制。
	已退化的b32拆写、DWORD DMA、把写机械搬入Compute不作为已证实改善。
- 详见[LDS与16-stage分析](results/d256_v48_att_20260923/lds-pressure.md)、
	[核算数据](results/d256_v48_att_20260923/lds-pressure-audit.json)。未执行GPU/改内核，220T仍未达到。

### M32 双wave DV128与P交换设计（尚未实现）

- 用户提出两个wave完成同一组M32 query的DV256，各自仅保留DV128的O：理论O由128降至64FP32/lane，
	输出按列拼接、不做O归约；必须共享同一softmax尺度/normalizer，不能各自独立归一化P。
- 更均衡的分工是QK沿BN的key列切半、每个score仍完整归约DQ256；交换行max/sum及BF16 P后，
	PV改为沿DV切半。BN64时每wave32QK+32PV M32 MFMA。单producerQK+PV128则与纯PV伙伴约3:1失衡。
- BM128/8wave时P唯一payload16KiB；理论K/V读量相对当前M16可从512降到256KiB/CTA/BN，
	另付P写16KiB、P读16–32KiB及metadata/同步，以上是模型而非实测。完整M32 Q仍占64DWORD/lane。
- 当前K/V已用满64KiB，P必须在证明旧K读完、P消费完成边界后与LDS复用，或重新分块。
- 已核对成熟参考：CUTLASS示例41的softmax-P→shared memory→PV；CK的N-warp S-shuffle通过LDS重排
	softmax前S再供不同PV布局，两者不能混称。旧v35/v38/v61交换的是DQ部分FP32 score，v60则12wave专用角色，
	都不等于新的8wave key-split/P-exchange映射。
- 详见[M32 P交换设计与源码参考](results/d256_v48_att_20260923/m32-p-exchange-design.md)。本次未改内核/跑GPU，未产生新性能结果。

### 方案65–67：独立 M32 key-split/P-exchange 实现（2026-09-23）

- 按用户要求新增[mha_pa_bf16_256_pexchange_942.py](mha_pa_bf16_256_pexchange_942.py)，
	QK在成对wave之间切BN的32-key部分，每个score仍完整归约DQ256；交换BF16 P及FP32行max/sum后，
	PV按DV128切分。BM128/BN64，8wave配对(w,w+4)，每wave O64FP32、Q64DWORD。
- K32KiB消费结束后复用为P16KiB+max2KiB+sum2KiB，V独立32KiB；共64KiB。
	当前同相执行，每BN五次CTA rendezvous，不是v48的4+4错相。对方P读取与第一V64 operand共用等待，
	后续KV预取在第一个PV64之后；原scale/lazy-rescale/P舍入/O舍入与公开契约保留。
- v65完整K256版本先通过smoke元素检查，但persistent为scratch36B/spill8，资源门禁停止计时；
	[原报告](results/d256_pexchange_20260923/v65-smoke.json)保留。
- v66改顺序K128×2，产物确实变化但仍scratch36B/spill8，
	[原报告](results/d256_pexchange_20260923/v66-smoke.json)保留，不把源改动当资源成功。
- v67真实ISA显示长期地址表达式被spill；改为使用点附近重建LDS地址并pin tid，
	smoke及Full的两调度均去除scratch/spill。最终源SHA
	`5ef99b640d69fcf2d94c58f5e30202756c7cc71d3c7ed979d6588aee0c54bfa6`。
- [独立回归](test_mha_pa_pexchange.py)最终[JUnit](results/d256_pexchange_20260923/v67-functional-fixed.xml)
	**21通过/0失败/0跳过**：2项CPU factory/布局证明，19项GPU参数覆盖尾块、三页、causal/noncausal、
	LSE、prefix/NaN/scales、stream/graph及跨half重缩放。首次回归20通过，剩1项是测试字段名错误，
	修正`Case.page_order`后全通过，不改容差或绕过检查。

Full Q10240/KV2583/page64，GPU2/BDF `0000:a4:00.0`，2buffer×2round=4样本/候选、10warmup、repeat1，
原cudaPerf及独立FP32 `.02`元素检查/两次逐位重复；同轮四候选交替测量。**这是小样本诊断，不是50样本验收或AITER比较**。

| 候选 | 中位数 µs | TFLOPS | VGPR/SGPR | LDS B | scratch/spill |
|---|---:|---:|---|---:|---|
| M32 P-exchange persistent | 4871.472 | **133.437** | 246/64 | 65536 | 0/0 |
| M32 P-exchange grid | 4902.292 | **132.598** | 244/44 | 65536 | 0/0 |
| 同轮v48 persistent | 3415.442 | 190.322 | 256/72 | 65536 | 0/0 |
| 同轮v48 grid | 3423.723 | 189.862 | 252/53 | 65536 | 0/0 |

- [原始Full结果](results/d256_pexchange_20260923/v67-full.json)、
	[数值/资源/ISA/门禁审计](results/d256_pexchange_20260923/v67-audit.json)、
	[实现与证据说明](results/d256_pexchange_20260923/README.md)。新acc最大2.73324e-6。
- entry/prepared/sampling/exit分别gfx0/0/2/1%、UMC全0%，VRAM最高2.725%，PTL全程快照Enabled/VECTOR,F8。
	允许驻留进程的非独占协议；不设置PTL/频率/功率，不采新ATT/PMC。
- 结论：**独立实现功能完成，但当前性能更慢，220T未达到**。按用户要求保留新文件，默认仍为v48。
	原v48设备源、原wrapper与统一测试入口SHA不变；没有修改默认candidate集或历史性能表。

### CK 类似 S-shuffle 方案的性能测试（2026-09-23）

- 用户要求测试CK类似方案。实际核对固定CK `15e12dd7f25ee583617c78f66cb502ff9916585f`：
	gfx9 BF16/D256默认SplitKV tile为M64/N128、`r4x1x1`，生成器因此选择`qr`而非`qr_nwarp_sshuffle`。
	**模板存在不等于gfx942默认有对应调优配置**；没有将AITER普通QR结果冒充S-shuffle。
- 在[隔离wrapper](results/ck_sshuffle_20260923/ck_sshuffle.cpp)中显式实例化未修改的CK模板：
	M32/N64/2wave（各DV128）、M32/N128/4wave（各DV64），以及M64/N128/4wave QR对照。
	默认policy要求MWarp1；两shuffle实例均从实际dispatch/ELF确认。
- CK交换的是softmax前FP32 S，再按PV布局读回；自有方案交换softmax后BF16 P与行统计，非完全相同算法。
	`num_splits=1`、直接BF16输出、无LSE，每调用一个attention kernel，无需且未漏计combine。
- GPU2/BDF `0000:a4:00.0`，Full Q10240/KV2583/H24/HK2/DQ=DV256，noncausal；
	10buffer×5round=50event/候选、六候选共300event，10warmup/repeat1/原cudaPerf，按buffer/round交替顺序。
	原FP32 `.02`元素检查、逐buffer两次精确重复通过；全样本中位数，不筛长尾。
	CK/AITER使用同值prepared线性KV、准备在计时外；自有仍读SHUFFLE-5D分页KV，不声称包括gather成本。

| 同轮候选 | 中位数 µs | TFLOPS |
|---|---:|---:|
| CK S-shuffle M32/N64，2wave，DV128/wave | 22116.703 | **29.391** |
| CK S-shuffle M32/N128，4wave，DV64/wave | 17923.757 | **36.267** |
| CK QR M64/N128，4wave，单split | 10808.710 | **60.140** |
| AITER public（实际CK QR M128/N128） | 6437.281 | **100.979** |
| 自有P-exchange persistent | 4871.732 | **133.430** |
| 自有v48 persistent | 3416.902 | **190.241** |

- 实际CK metadata：S-shuffle2wave `vgpr_count348/agpr_count92/SGPR55/LDS35296`，
	4wave `300/48/54/35296`，QR `288/64/54/18432`；三者scratch和VGPR/SGPR spill均0，实际ISA无scratch指令。
	字段不重复相加；未采occupancy counter。35,296B LDS大于半CU容量，不能双CTA驻留；2wave配置利用率限制需另行分析。
- 全部边界gfx/UMC0，VRAM最高7.505%，PTL Enabled/VECTOR,F8；非独占协议，未设置GPU策略、未采新ATT/PMC。
	CK保留原编译BF16 truncate设置，未改变参考容差；没有修改CK/AITER或自有设备源码。
- [详细说明](results/ck_sshuffle_20260923/README.md)、[完整原始结果](results/ck_sshuffle_20260923/full-50.json)、
	[300样本/dispatch/ELF审计](results/ck_sshuffle_20260923/full-50-audit.json)。
	结论仅限两个自定义实例：**明显更慢，不代表CK最优能力或S/P重分布必然无效；220T仍未达到**。

### AITER 其他 D256 路径及 ASM 离线调查（2026-09-23）

- 固定AITER `bde46043` / CK `15e12dd7`，只读源码、manifest和已有二进制；没有GPU运行、重新编译、ATT/PMC或新性能成绩。
- **gfx942 BF16 FMHA ASM没有DQ=DV256行**：仅D128/V128、D192/V128；PA ASM的256出现在block/page字段，不是head_dim。
	真正D256 ASM位于gfx950，QKV为FP8、O为BF16；公共FP8路由还要求power-of-two GQA，当前H24/HK2=12不满足。
- 当前Full可新增的源码候选是Triton默认MHA和DAO/FA3共享prefill（后两者同一底层实现，不重复计数）。
	gfx942默认均BM128/BN64/4wave，常规Q-resident online softmax；D256源码可参数化，但本次未声称编译/精度/性能通过。
- **找到更接近BF16 P-exchange的现有实现：Gluon `pa_decode_gluon`及HIP `paged_attention_rocm`**。
	两者QK按KV-token跨4wave分工，softmax行统计协作，BF16 P经LDS重分布，PV按DV列分工；D256时各wave负责DV64，主要用M16。
	Gluon公共入口`query_length≤4`、`query_length×GQA≤64`，多token自动causal；不是当前Q10240 noncausal Full替代。
	HIP为KV partition256+reduce的decode/MTP路径。Unified虽有BF16 D256配置/测试，但只允许causal；Lean paged是deprecated decode、GQA TODO。
- 已用LLVM完整反汇编gfx950 FP8 D256及MI308 BF16 D128/D192，未执行：
	- gfx950 D256是BM256/BN64/8wave，每wave32query×完整DV256，O占128FP32/lane，**没有P交换**。
		全函数无`ds_write*`；QKV走`buffer_load_dwordx4 … lds`，V用`ds_read_b64_tr_b8`，P寄存器打包FP8直接PV。
		4+4两组分担K/V预取并错相，两组都做QK/PV；raw LDS=163840B，不能直接迁到gfx942/BF16。
	- gfx942 D192/V128 ASM用4wave/M32，Q/K走4B/lane direct-LDS，V经VGPR/permute再b128写LDS；
		两套score/operand寄存器交错QK、旧P exp/pack及旧PV，新score统计穿插PV，未见跨wave P交换。
		这是调度参考，不是D256实现。MI308产物选择依据PCI chip ID而非仅CU数。
- 当前缓存中与已测AITER公共dispatch同符号的CK D256：BM128/BN128/4个M-wave，Q load-once，DQ32/KV32分块，寄存器P。
	metadata `vgpr_count458/agpr_count202/SGPR48/LDS18432/private528`，spill字段0且ISA无`scratch_*`；不能把private528直接认作动态spill。
	旧公共AITER报告未保存库hash，故只记为当前缓存审计，不追加伪造历史artifact收据。
- [完整路径与算法报告](results/aiter_d256_inventory_20260923/README.md)、
	[源码证据](results/aiter_d256_inventory_20260923/source-evidence.txt)、
	[ASM/CK资源摘要](results/aiter_d256_inventory_20260923/summary.json)、
	[完整离线审计v2](results/aiter_d256_inventory_20260923/inventory-v2.json)。
	原kernel/hash与历史性能保持不变；**220T仍未达到**。

### 68–69. Q/K 4B/lane direct-to-LDS 与 DS/VMEM 交织（2026-09-23）

- 按用户要求分别尝试Q-only、K-only、Q+K，并对照连续DMA、S4/S6拆分、晚发射。
	新增独立[mha_pa_bf16_256_dma_942.py](mha_pa_bf16_256_dma_942.py)，默认v48不重定向。
- 4B DMA的LDS目标固定为`m0+lane*4`，因此在global源lane反置原K XOR；CPU逐元素证明与v48原32KiB K图像完全一致。
	保留SHUFFLE-5D/page32/64/128与原buffer extent；V仍走64-bit GLOBAL，数值/舍入不变。
- Q使用序言64KiB临时区，32DMA/wave、8个b128片段读回，低D128读取与高D128 DMA交织，全部Q读完才复用K/V LDS。
	K每tile/wave由4条b128 VMEM+4条DS写改为16条DMA；写入字节相同，不误称减少VMEM请求数。
- v68首次Q-only出现private16B/3VGPR spill，资源门禁拒绝；[失败原始记录](results/d256_dma4_20260923/v68-smoke.json)保留。
	v69将Q-only地址就近生成，constexpr移除无用K地址，零scratch/spill恢复；未放宽精度或资源门禁。
- 安全发射点：领先wave组S3结束与落后组S2 K-high读完成barrier配对，因此S4开始可覆盖K。
	**最好方案在S4把16个V-low `ds_read_b128`与16个K DMA按1:1交织**，S6 `vmcnt(0)`退休DMA，
	下次S0读新K前双方已经经过完成barrier。真实ISA确认`RD×16`，非只改变源码次序。

同轮小样本探索：GPU2/a4，Full Q10240/KV2583/H24/HK2/D256/page64/noncausal，2buffer×2round=4样本/候选。

| 候选 | TFLOPS | µs | VGPR / SGPR |
|---|---:|---:|---|
| v48 | 189.927 | 3422.542 | 256 / 72 |
| Q-only | 188.657 | 3445.583 | 256 / 100 |
| **K-only S4交织** | **197.735** | **3287.401** | **242 / 87** |
| Q+K S4交织 | 194.857 | 3335.942 | 244 / 106 |
| K-only S4连续DMA | 173.190 | 3753.304 | 242 / 87 |
| K-only S4/S6各半 | 183.184 | 3548.522 | 244 / 87 |
| K-only S6晚DMA | 177.803 | 3655.924 | 244 / 87 |

- [完整功能矩阵](results/d256_dma4_20260923/v69-functional.json)：Q/K/QK三配置×两调度，共78项通过，含page/tail/causal/LSE/prefix/nonunit scales及stream/graph。
	已导出的smoke/Full均LDS65536、AGPR/private/spill0；K-only消掉14个VGPR，但不因此宣称occupancy提升。
- **七候选50样本完整轮出现共同变慢，必须保留**：v48/Q/K/QK/burst/split/late分别
	`128.348/127.652/133.439/131.838/116.934/126.073/122.945 T`，K仍领先同轮v48 **3.97%**。
	10buffer逐元素`.02`、两次精确重复、每buffer与v48逐位一致全部通过；没有筛掉慢样本。
- 为调查该现象，另做一次固定v48/K双候选50样本对照，并异步读取4次SMI clock/power/temp；没有插入冷却等待或改变原cudaPerf。
	完整中位数v48 **3425.122µs/189.784T**，K-only **3295.160µs/197.269T**，吞吐 **+3.94%**、时延约−3.79%。
	配对ratio中位数1.03921。两个候选后段也共同变慢，计时区间内读到gfx约1745→1378MHz变化；
	无逐kernel时钟或节流原因证明，**197T不是持续稳定吞吐保证**，两轮结果各自独立保留。
- [350样本原始轮](results/d256_dma4_20260923/v69-full-50.json)、
	[100样本遥测轮](results/d256_dma4_20260923/v69-pair-50-telemetry.json)、
	[全部样本/ISA/功能审计](results/d256_dma4_20260923/v69-pair-50-telemetry-audit.json)、
	[跨轮核对](results/d256_dma4_20260923/reconciled.json)、[完整说明](results/d256_dma4_20260923/README.md)。
	新源SHA `bdea83646f0f3ba32074b53da9074d2b4781dc87f3a7cba24a66d4cda5f7043c`，
	K-only Full ELF `b2f41435b7bd00db33ecc341fadc5c623e417ffd354859a7cc92eff62db078f9`。
- 保留独立实验，推荐显式`dma_query=False, dma_key="early", interleave=True`；默认v48及历史六shape/AITER成绩不改。
	所有边界gfx/UMC<3%、VRAM≤20%、PTL Enabled/VECTOR,F8；非独占，不设置GPU策略、不采ATT/PMC。
	**本次相对收益约4%，220T仍未达到。**

### 70. 补测 S2 的 DS写 / DMA写直接混排

- 增加`dma_key="write_mix"`，不是仅混排V的DS读取：S0退休旧K-low，S2读取旧K-high时，
	将V的4次b128 DS写与next K-low的8次DMA交织，S4再发next K-high的8次DMA。
	ISA确认S2为`(R,R,D,R,R,W,D)×4`、S4为`(R,R,D)×8`；地址区域与跨组barrier安全性分别保留。
- 同轮2buffer/4样本Full：v48 **190.306T/3415.722µs**，此前K-only S4读写交织 **197.647T/3288.861µs**，
	新写写混排 **190.408T/3413.902µs**。原精度/重复/与v48逐位检查通过，VGPR256/SGPR87、LDS65536、private/spill0。
	没有优于早S4方案，保留诊断配置但不扩大性能验收。
- 新模式[26项功能检查](results/d256_dma4_20260923/v70-write-functional.json)通过，与v69共104条；
	[Full原始样本](results/d256_dma4_20260923/v70-write-full.json)、
	[S2/S4 ISA审计](results/d256_dma4_20260923/v70-write-full-audit.json)、
	[最终汇总](results/d256_dma4_20260923/reconciled-final.json)。
- 当前新源SHA `f2cee7e1ec118ea3082a8adf94979e03467fe3e383cd667313a775973e390979`，
	最佳K-only的ELF仍为`b2f41435...`、与全部v69 Full轮次恒等；新模式ELF为`f4f87dee...`。
	默认v48/原wrapper/统一入口及GPU策略不变，仍推荐只改K、S4逐条交织，不启用Q DMA；220T未达到。

### 71. V 4B/lane buffer direct-to-LDS：布局通过，SGPR spill门禁停止（2026-09-23）

- 按用户“尝试V使用dma”扩展同一独立DMA实验模块，不替换默认v48；新增`dma_value=off/early/split/late`与V交织开关。
	每wave每tile的4次V GLOBAL b128 +4次DS b128写改为16次DMA4，仍使用原SHUFFLE-5D、K XOR、M16及64KiB LDS。
- 与历史51/54的GLOBAL-LDS V和拆分阶段不同，本次V使用编译器识别的buffer-LDS intrinsic；
	**page stride乘法前升64位，完整V地址逐tile构造descriptor**。page32使用两个16KiB descriptor，page64/128使用一个32KiB。
	36组CPU逐元素映射证明与原V LDS图像相同，包括page128奇偶、两个物理页相同/不同和合成64-bit高地址进位。
- V(t−1)在S0与K-low DS读取逐条交织，S2 vmcnt等待、跨组barrier之后S4消费；单tile与最后tile有独立drain。
	不增加CTA barrier；K仍可在S4与V-low DS读取交织。
- [v71 smoke](results/d256_vdma4_20260923/v71-smoke.json)的V-only首buffer精度通过、VG238/SG100，无spill；
	K+V为VG225/SG106，产生**8个SGPR-to-VGPR-lane spill**，private0但不满足零spill门禁，未进入计时。
	实际ISA有`v_writelane/v_readlane v224`；多组V m0目的地址和SOFFSET常量被长期保留。失败产物不删除，不放宽门禁。

### 72. V DMA标量局部重建、发射位置比较及50样本复核

- 只将V DMA的m0目的地址/SOFFSET在每条请求附近重建，避免SGPR hoisting；V-only SG100→74，K+V SG106→89、spill8→0。
	默认v48、K-only Full ELF不变，Q保持原寄存器加载。V-only VG238，K+V VG224；全部已导出smoke/Full为LDS65536、AGPR/private/spills0。
- 同轮GPU2/Full Q10240/KV2583、H24/HK2、DQ=DV256、page64/noncausal，2buffer×2round探索：

| 候选 | µs | TFLOPS |
|---|---:|---:|
| v48 | 3422.182 | 189.947 |
| K-only，S4交织 | 3289.861 | 197.587 |
| V-only，S0交织 | 3229.001 | 201.311 |
| **K+V，S0/S4分别逐条交织** | **3103.540** | **209.449** |
| V-only，V burst | 3728.785 | 174.328 |
| K+V，V burst | 3601.064 | 180.511 |
| K+V，V分S0/S2各8条 | 3380.362 | 192.297 |
| K+V，V全部晚到S2 | 3532.363 | 184.022 |

- 上表仅每候选4样本诊断。随后固定v48/K/V/KV，原cudaPerf、10独立buffer×5轮=200 events，
	10warmup/repeat1、原FP32 `.02`、两次精确重复、全部buffer与v48逐位一致均通过，完整50样本中位数：

| 候选 | µs | TFLOPS | 同轮相对v48 |
|---|---:|---:|---:|
| v48 | 5067.353 | 128.279 | 1.0000× |
| K-only | 4866.351 | 133.577 | 1.0413× |
| V-only | 4787.650 | 135.773 | 1.0584× |
| **K+V** | **4590.749** | **141.596** | **1.1038×** |

- **K+V比同轮K-only吞吐+6.0034%、时延−5.66%；配对时延比中位1.060113。**
	但本轮再次出现共同降速：v48前两轮约3422/3424µs，后三轮约5072/5070/5076µs；K+V约3102/3105→4597/4598/4599µs。
	四次有界异步只读SMI均与计时wrapper重叠，gfx0–3从1754–1759→1347–1351→1228–1233→1214–1219MHz，
	power448→354→324→320W、hotspot均54°C。已观测动态时钟下降，**未证明具体限频原因或单kernel精确时钟**。
	所有慢样本保留，不以快两轮替代全样本，不重跑至快；不能声称稳定209T，更未达到220T。
- V-only/K+V[52项功能检查](results/d256_vdma4_20260923/v72-functional.json)全部通过：page32/64/128、两调度、
	KV1…129、变长/空query、前缀guard/scales/NaN尾页、causal/LSE `.002`，含4个stream/graph记录。
	其他V调度只验证了探索Full，不冒充完整功能矩阵；零spill只声明已导出smoke/Full，不外推LSE产物。
- 两组128 MFMA回边均核对：K+V每wave/BN64仍有64次DS b128读，显式DS b128写8→0，DMA4为32次；
	S0/S4均`(R,D)×16`。DMA仍写LDS，**不是LDS写字节归零**；输出CShuffle仍有DS b64写。
- GPU2/BDF a4，四个正式边界gfx0/1/2/2、UMC0，VRAM≤6.1783%，PTL Enabled/VECTOR,F8；低负载非独占。
	不改PTL/频率/功率/NUMA，不采ATT/PMC、不关缓存。默认公共后端、原测试与timer保持不变；实验V默认仍`off`。
- [完整结果/算法/限制](results/d256_vdma4_20260923/README.md)、
	[探索32events](results/d256_vdma4_20260923/v72-full.json)、[正式200events](results/d256_vdma4_20260923/v72-full-50.json)、
	[正式独立审计](results/d256_vdma4_20260923/v72-full-50-audit.json)。
	当前源码SHA `a7979a426e16d20abc8fbcad06d7fcfe5b25dff3dc3b51f6b6dcbd2ecea6be5f`；
	Full K+V ELF `20594498d686a6b6c384a59ac572780a7c8f0475b29b7d3c31bb84fed64d1555`。

### v72 K+V DMA Full/persistent ATT与UI交付（2026-09-23）

- 按用户“att”请求采集上述当前K+V候选，Q DMA关闭、V S0/K S4交织；不改内核、默认后端或编译选项。
	Full Q10240/KV2583、H24/HK2、D256/page64/noncausal/noLSE，**persistent grid80×1×1、block512**。
	GPU2/BDF a4，第三次调用（dispatch393），SE0/CU1/四SIMD、256MiB ATT buffer；不采PMC/ATT activity。
- 完整UI已复制到当前MHA目录：[manifest](ui_output_agent_11400_dispatch_393/filenames.json)，
	[统计表](stats_ui_output_agent_11400_dispatch_393.csv)；16文件135,382,632bytes，全部SHA与原件相同。
	8/8长驻wave完整拼接且到达`s_endpgm`，每SIMD两wave、每wave24任务；不应套用旧grid的192-wave数量。
- 捕获code object6与正式50样本的ELF `20594498...`完全相同，源IR身份也相同；VG224/SG89、LDS65536、private/spill0。
	3,814,152动态指令全部映射/hit/CSV一致，MFMA1,007,616、DMA4 254,976；DS b128写0，CShuffle b64写3,072。
	这些是完整性计数，不代表DMA不写LDS，也没有据此宣称bank conflict或stall归因。
- 原FP32 `.02`前后检查通过，acc均`2.727154653547892e-6`，第2/3/4次调用逐位一致。
	入口/准备/退出gfx与UMC均0，VRAM283/2229/4629MB，PTL始终Enabled/VECTOR,F8；低负载非独占。
	未改PTL/时钟/功率/NUMA，缓存保持启用；日志无DataLost/溢出/拼接失败。UI提供ISA，Source字段为空。
- [本次范围与证据](results/d256_v72_kv_att_20260923/README.md)、
	[完整性与复制审计](results/d256_v72_kv_att_20260923/ui-copy-audit.json)。
	只采ATT，不重新测TFLOPS；persistent与旧v48 grid trace的总时长不能直接作速度对照，220T目标状态不变。

### v72 Memory阶段SALU/VALU与LGKM后移检查（离线，未改内核）

- 按用户要求核对上述实际ATT/ELF：8wave×24任务，每任务固定`t=5…36`，共6144次/阶段；
	S0每wave/BN64有**52SALU/1VALU**，S4有**33SALU/0VALU**，S2/S6无ALU。
	S0唯一VALU是后续mask bound的递推，不是lane/LDS地址；这些不变地址已经在循环外。
- 公共计算优先项：页表SMEM由S0尾提前发射，ready仍延后；page64/128的两次同源`_page_ready`合并一次；
	按PAGE32/64/128简化tile到page索引并提前table完整基址。V两DWORD动态基址可尝试前一Compute准备，
	但不要把16个M0地址/SOFFSET全部长期保存：v71已因此产生SGPR spill8，v72局部生成才消除。
- 实测等待span中位（shader cycles，领先/落后）：S0首次LGKM32/32、重复LGKM4/4；
	S2合并VM/LGKM148/148；S4 LGKM58/88；S6前VM4/4、末LGKM148/148。
	不是可直接节省周期，S2合并等待也不能从现有trace拆成VM和LGKM各占多少。
- **S4末LGKM两组可以作为跨barrier、移到S5首PV前的试验。** 领先组S2/S6的LGKM也可单独评估后移，
	但保留VM发布并保证首MFMA之前ready；领先S2当前已有Compute伙伴制约，单挪wait未必有收益。
	**落后组S2/S6不能单独后移**：前者会合后领先S4覆盖K，后者会合后领先下一S0覆盖V，必须先保证旧DS读退休。
- CDNA3 `s_barrier`不清零访存counter；gfx942有BackOffBarrier，所以也不能套用旧架构的“任何访存都必须先清空”禁令。
	具体依据跨wave读/覆写和本wave首consumer判断；raw DS inline asm不能依赖编译器自动修补被删除的等待。
- [详细依赖/建议](results/d256_v72_kv_att_20260923/memory-stage-analysis.md)、
	[实际PC与ATT计数](results/d256_v72_kv_att_20260923/memory-stage-audit.json)。仅CPU分析，没有新优化版本或新性能成绩。

### 73. 页表提前/别名ready合并与S4等待后移的2×2试验

- 按用户“尝试”新增两个默认关闭选项：`early_pages`、`late_s4_wait`；仅允许Q DMA关闭、K/V均early的诊断。
	公共v48不变，原v72 K+V控制ELF仍为`20594498...`，未启用选项的编译行为不变。
- 页优化只改S0页表SMEM的发射位置并合并page64/128同源ready，page32仍等待两个page；
	页索引/clamp及算术不变。未来page结果在S0尾wait之后才复制，不提前读取待定SGPR。
- S4只将LGKM移到原barrier之后、S5首个PV/cross-sum消费之前，加compiler fence避免消费者越过；
	**S2/S6的VM发布与旧LDS读退休等待全部不动**，没有删除或增加CTA barrier。
- [实际Full ISA审计](results/d256_memstage_20260923/v73-full-audit.json)确认：
	`wait_s4`整个函数只有PC `0x3b88/0x3b8c`、`0x6840/0x6844`两对barrier/wait机器字换位，**其余全部指令机器字相同**。
	`pages`两组SMEM实际在首DS/DMA之前，S0 wait2→1。四DMA候选VG224/SG89、LDS65536、AGPR/private/spill0。
- GPU2/Full/persistent，2buffer×2round=4样本探索，原FP32 `.02`、两次精确重复、与同轮v48逐buffer位同：

| 候选 | µs | TFLOPS | 相对v72 K+V |
|---|---:|---:|---:|
| v48 | 3416.003 | 190.291 | 0.90724× |
| v72 K+V控制 | 3099.120 | 209.748 | 1.00000× |
| 页表提前/ready合并 | 3096.080 | 209.954 | 1.00098× |
| S4等待后移 | 3057.400 | 212.610 | 1.01365× |
| 两者组合 | 3051.201 | 213.042 | 1.01571× |

- 此为小样本诊断，尚不声称稳定213T；页优化单独约0.10%不足以断言稳定收益。
	后续功能及完整50样本结果见下，不将此表替换为较快片段。
- [smoke](results/d256_memstage_20260923/v73-smoke.json)、[Full20样本](results/d256_memstage_20260923/v73-full.json)、
	[实验说明](results/d256_memstage_20260923/README.md)。源码SHA `5feaea7ced508de30f313a11b12a24258eb13cff29304429158003b04c06445c`。
	没有新ATT/PMC、调频/PTL/功率/NUMA修改或缓存关闭；220T仍未达到。

#### v73完整功能与50样本复核

- 三种新配置[78项功能检查](results/d256_memstage_20260923/v73-functional.json)全部通过，各26项、共6个stream/graph记录。
	覆盖page32/64/128、KV边界、两调度、causal、LSE `.002`、prefix/NaN/scales、ragged/空query；原FP32 `.02`与两次精确重复不变。
- 固定上述五候选、10独立buffer×5round=250events；全10buffer原精度/精确重复/与v48逐位相同均通过。
	全样本中位数与同round/buffer配对时延比分别列出，不混为同一统计量：

| 候选 | 全50样本µs | TFLOPS | 全中位数比/v72 | 配对比中位/v72 |
|---|---:|---:|---:|---:|
| v48 | 5060.591 | 128.450 | 0.90518× | 0.90629× |
| v72 K+V | 4580.748 | 141.905 | 1.00000× | 1.00000× |
| 页表优化 | 4575.548 | 142.067 | 1.00114× | 1.00144× |
| S4等待后移 | 4511.948 | 144.069 | 1.01525× | 1.01512× |
| 组合 | 4381.787 | 148.349 | 1.04541× | 1.01622× |

- 本轮同样有公共降速且末轮耗时恢复：v72各轮3099.579/4574.349/4594.229/4592.208/3099.759µs，
	组合3050.718/4295.727/4521.188/4525.748/3054.179µs；v48也同向变化。
	四次只读遥测与计时wrapper重叠，gfx1744–1750→1353–1357→1227–1233→1212–1218MHz，
	power446/355/323/319W、hotspot50/51/51/51°C；没有逐kernel时钟、末轮时钟或确切限频原因证明。
- **S4单因素配对收益约1.51%，组合约1.62%。** 组合全中位数比+4.54%受快慢区间切换影响，
	不能全部归因为代码、也不能将探索213.042T作为稳定吞吐；所有250原始样本保留，不筛快段，不重试至快。
- 正式与探索轮ELF逐候选相同。S4-only仍严格只有两对wait/barrier换位；所有DMA Full VG224/SG89、LDS65536、private/spill0。
	该资源结论仅限已导出smoke/Full，不冒充所有LSE产物均无spill。新开关均默认False，公共v48不替换。
- 门禁GPU2/a4，gfx0/2/2/2、UMC0、VRAM≤6.7887%、PTL Enabled/VECTOR,F8；非独占，没有GPU策略修改或新profile。
	[250样本原始结果](results/d256_memstage_20260923/v73-full-50.json)、
	[ISA/功能/遥测审计](results/d256_memstage_20260923/v73-full-50-audit.json)。**220T目标仍未达到。**

### v73最快组合默认启用（2026-09-23，用户明确要求）

- 公开`PagedAttention(..., DQ=DV=256, ...)`默认选择已有v73 `both`组合：
	Q DMA关闭，K/V early DMA、两处逐条交织、`early_pages=True`、`late_s4_wait=True`。
	支持persistent/grid和page32/64/128；D128/D192的分派及所有原设备函数AST保持不变。
- 默认wrapper直接取得组合launcher，不调用实验public factory，避免循环验证/构造；
	与显式factory使用相同位置参数cache key。显式实验仍有独立wrapper/compiled缓存，不污染公共默认实例。
	实验factory保留原默认参数，便于明确控制变量；改变的是实际公共D256分派，而非GPU策略。
- [默认Full验证](results/d256_default_20260923/default-promotion.json)的两调度原FP32 `.02`、两次精确重复、
	与显式组合逐位一致全部通过。persistent ELF仍为`d9294dfc824afe4738eb076093430148fa0678ffefb1d2ce143e766842a0f26c`；
	grid ELF为`6378a0a0fa39a54a183fc339d09b556f7ac8340f99e8ab64a18606cfc7bc148e`，也与对应显式组合恒等。
	Full资源persistent VG224/SG89，grid VG218/SG72；均LDS65536、AGPR/private/spills0。
- 新增默认factory回归，验证三页/None-True-False调度、实验缓存隔离和D128/D192分派未变；
	公开D256功能子集[JUnit](results/d256_default_20260923/default-functional.xml) **11通过、0失败、0跳过**，183.907秒；
	含5项KV边界、3项分页layout（两调度/full/causal/O/LSE/guard）及2项stream/graph。
	此次默认启用没有重跑性能或ATT/PMC，不将213T探索值改写为稳定验收成绩。
- 本次源文件修改仅为[mha_pa_bf16_942.py](mha_pa_bf16_942.py)、[test_mha_pa.py](test_mha_pa.py)，
	同步[README.md](README.md)及本优化记录；DMA设备源码未改。
	另新增[默认验收驱动](results/d256_default_20260923/validate.py)、[交付说明/文件清单](results/d256_default_20260923/README.md)
	及本次验证证据，未提交/推送Git。

## 2026-09-23：继续优化，目标提高到230 TFLOPS（v74–98）

- 沿用Full：Q10240/KV2583、DQ=DV256、H24/HK2、page64，完整调用有效工作量650033233920 FLOPs。
	达到230T需时延不超过2826.231µs；不改变FP32 `.02`、LSE `.002`、概率half-up或输出RNE。
- GPU2 / `0000:a4:00.0`，原生JIT缓存启用，gfx/UMC严格小于3%、VRAM不超过20%、PTL Enabled/VECTOR,F8；
	低负载非独占，未修改GPU频率/功率/PTL/NUMA，未使用PMC。所有raw、失败和编译源码快照保留。
- 本轮每项探索为同场2独立buffer×2轮、每候选4事件、正反顺序交替；原timer不变。
	v73显式对照Full ELF一直为`d9294dfc824afe4738eb076093430148fa0678ffefb1d2ce143e766842a0f26c`，
	不能再把旧脚本调用公共入口的`v48`标签当成v48执行；新驱动禁止该标签。

### 探索记录（均为4样本，不是50样本验收）

| 轮次 | 假设与结果 | 同轮v73 / 候选TFLOPS | 证据 |
|---|---|---:|---|
| v74 | S0等待跨barrier；仅领先S2/S6后移，落后WAR等待不动 | 212.906 / 216.073 | [六路等待](results/d256_pipeline230_20260923/v74-waits-full.json) |
| v75 | S4、领先S6的V操作数按N32逐批等LGKM12/8/4/0，其他数学不变 | 212.993 / 217.535 | [PV宽度](results/d256_pipeline230_20260923/v75-pv-full.json) |
| v76–77 | M0→已有DS→DMA填hazard间隔，K-only约217.7；QK分批等待退化 | 212.940 / 217.700 | [QK与fused-K](results/d256_pipeline230_20260923/v77-qk-full.json) |
| v78 | 退休当前操作数后提前读下一半K/V，虽无spill但争用/调度成本增加 | 213.264 / 186.076、201.697、173.666 | [prefetch](results/d256_pipeline230_20260923/v78-prefetch-full.json) |
| v79 | 免LDS输出64/128位，不如原coalesced CShuffle | 213.272 / 212.724、213.536 | [直接输出](results/d256_pipeline230_20260923/v79-output-full.json) |
| v80、v87 | QK双链与scheduler mask调整均未优于已有组合 | 212.732 / 219.202以下 | [正确mask对照](results/d256_pipeline230_20260923/v87-schedmask-full.json) |
| v81 | DMA立即数误当只偏移GLOBAL，精度失败，未计时 | 无候选成绩 | [原失败](results/d256_pipeline230_20260923/v81-immediate-full.json) |
| v82 | 成对删非数据handoff barrier，正确但失去有益错相 | 212.736 / 211.492、214.965、198.517 | [同步精简](results/d256_pipeline230_20260923/v82-prune-full.json) |
| v83 | V descriptor基址减32KiB，M0兼作SOFFSET，省每packet独立偏移SALU | 213.092 / 218.883 | [V M0](results/d256_pipeline230_20260923/v83-m0-full.json) |
| v84 | query-block优先或有效前缀转置，不改输出/算术 | 212.519 / 219.847、219.643 | [任务序列](results/d256_pipeline230_20260923/v84-task-full.json) |
| v85 | page64 K也共用M0，增加descriptor/live range后不如V-only | 212.723 / 217.861 | [K M0](results/d256_pipeline230_20260923/v85-m0-kv-full.json) |
| v86 | 页请求移S2以允许S0非零LGKM，QK/EXP重排退化 | 212.730 / 215.159 | [S0 progression](results/d256_pipeline230_20260923/v86-s0progress-full.json) |
| v88 | CShuffle用32/64KiB平面减少尾部同步，32KiB仅小幅 | 212.877 / 219.979、218.540 | [输出平面](results/d256_pipeline230_20260923/v88-outputplane-full.json) |
| v89 | 领先K/V半发布移入compute使落后等待可后移，正确但减慢 | 212.574 / 211.453、205.360、199.047 | [分半发布](results/d256_pipeline230_20260923/v89-publication-full.json) |
| v90–91 | **K偶D32面先读，落后S2在交接前只等LGKM8，消费前归零** | 212.636 / 221.167 | [部分WAR](results/d256_pipeline230_20260923/v91-war-overlap-full.json) |
| v92 | 领先S7每8MFMA内穿插max/center，保持逐输出关联顺序 | 212.417 / 221.603 | [compute overlap](results/d256_pipeline230_20260923/v92-compute-full.json) |
| v93 | compute `s_setprio`1/3，仅wave仲裁，均退化 | 212.721 / 218.507、218.193 | [priority](results/d256_pipeline230_20260923/v93-priority-full.json) |
| v94 | 部分WAR后QK逐packet等待，额外调度开销抵消 | 213.081 / 219.201 | [K stream](results/d256_pipeline230_20260923/v94-warstream-full.json) |
| v95–96 | 最后组合：32KiB输出无收益；order1/order2近似；PV2保留 | 212.873 / 221.421、221.523 | [组合](results/d256_pipeline230_20260923/v95-combo-full.json) |
| v97 | 修正DMA立即数同时偏移LDS，M0相减后精度通过，但不如V M0 | 213.042 / 217.368、217.048 | [立即数修正](results/d256_pipeline230_20260923/v97-immediate-corrected-full.json) |

- v79两次copy谓词接口编译失败已保留；修为bounded-buffer屏蔽后才进行计时，未放宽检查。
- `sched_barrier(mask)`中的mask是**允许跨越**的类型，不是阻止的类型；v87按文档另测，未篡改早期记录。
- GFX9 DMA的立即数同时加到GLOBAL及LDS，v81遗漏LDS补偿，v97已修复并验证；拒绝把失败版本留作可调用分支。
- v90 [逐字节WAR证明](results/d256_pipeline230_20260923/v90-war-proof.json)：领先wave0–3 K DMA仅写偶D32面；
	将受威胁的8次read先发、LGKM8退休后才能放行，剩余8次read与领先写集合不相交。落后组消费前仍LGKM0。
	V对应部分等待虽然也有不相交证明，但实测退化，没有采纳。所有CTA barrier保留。

### 当前候选ATT及收敛验证

- 只采一次qfast ATT（**不是最终部分K/PV7组合trace**），实际ELF
	`0d670f61ead38a8cda35dbc41408ac3b10d22ede54141b7b3f35b9fae5e5bb98`；
	[前后原参考/第2、3、4次逐位重复](results/d256_pipeline230_20260923/att-qfast/driver-result.json)通过。
	[独立审计](results/d256_pipeline230_20260923/att-qfast/analysis.json)：8完整驻留波、3,711,632条动态指令，
	全部code-map/CSV命中一致，1,007,616 MFMA不变。UI和原ATT保留在该独立证据目录，未覆盖旧UI。
- 全8波×24任务×tile5…36窗口，领先S0后LGKM中位100cycle；落后S2/S6末LGKM中位148cycle。
	PV渐进12/8/4/0等待各4cycle。只是本采集的wave跨度，不求和当作整卡耗时，也不由此断言bank冲突。
- [初步离线审计](results/d256_pipeline230_20260923/preliminary-audit.json)核对37份当时已完成报告，
	含全部失败；V M0地址54组（含32位进位）、有效前缀转置8196组CPU证明通过。
- v98清理未采纳分支，只保留等待后移、PV progression、V M0、任务序列、部分K WAR和S7 overlap；
	原Q/K/V DMA诊断入口保留。[清理后Full](results/d256_pipeline230_20260923/v98-clean-full.json)原参考、
	两buffer/v73逐位一致均通过，ELF与清理前相同：order2
	`774d2eaba3e4726e2d2ece80ea1b52ac7a9b21d7a01ae44cbab0caf5a224c889`，
	order1 `ee9715acaf68c91aaea9658250de4ff65c045be7aff453e352b47c88b6f3ad05`。
	Full资源VGPR224/SGPR92、LDS65536、AGPR/private/spills0。
- [完整功能验收](results/d256_pipeline230_20260923/v98-functional.json) **26项记录全部通过**，
	包含两调度、三页、ragged/空Q/前缀/scales/NaN-tail、causal/full、O/LSE、两项stream/graph。
	[定向重缩放验收](results/d256_pipeline230_20260923/rescale-check.json)另 **12项通过**：递增logit强制触发lazy-rescale，
	两个O半部及LSE仍通过原容差、两次逐位重复和v73逐位对照。
- [factory回归](results/d256_pipeline230_20260923/v98-factory.xml)2项通过、51项未选择；
	其中新增流水线选项类型/同步前提/缓存隔离回归，公共默认回归不变。未重新运行整个D128/D192 GPU矩阵。

### v98正式150事件验收与未达标结论

[原始50样本/候选](results/d256_pipeline230_20260923/v98-full-50.json)，10个独立buffer×5轮，
3候选共150事件；每buffer原FP32和两次逐位重复/v73一致检查全部通过。
最大acc均`2.7790490331192075e-6`。保留每个startup/慢尾，不以较快轮次替换全量中位数。

| 候选 | 全量中位µs | 全量TFLOPS | 全中位比/v73 | 同round/buffer配对比中位 |
|---|---:|---:|---:|---:|
| v73默认 | 4064.386 | 159.934 | 1.000000 | 1.000000 |
| v98，order2 | 4097.646 | 158.636 | 0.991883 | **1.038459** |
| v98，order1 | 4174.227 | 155.725 | 0.973686 | **1.039051** |

- v73各轮中位3050.799 / 3053.159 / 4064.386 / 4521.868 / 4524.369µs；
	order2各轮2941.039 / 2942.279 / 4097.646 / 4355.227 / 4350.488µs。
	全中位跨过共同变速的第3轮，不能把它与配对比混为同一统计量。
- 四个只读SMI请求区间与timer wrapper分别重叠56/46/41/10次，gfx有效时钟范围
	1734–1740 → 1373–1377 → 1229–1233 → 1383–1390MHz，socket451/372/333/217W。
	最后一份是覆盖收尾的请求区间，不是最后每个kernel的确定运行频率；throttle_status为N/A，具体降速原因未证明。
- entry/prepared/sampling/exit gfx0/0/2/0，UMC均0，VRAM最多5.5679%，PTL一直Enabled/VECTOR,F8。
	门禁通过不等于固定运行频率或全程独占。
- **230T未达成；正式全样本中位也未证明候选优于v73。** 因此公共默认保持v73，
	v98作为有配对改善证据的显式可选实现保留，不用探索221T宣称稳定成绩。
- 正式验收后新增唯一v99假设：V每2条/K每4条DMA共用M0，利用立即数同时补偿两端地址；
	同场v73 212.681T、v98 220.825T、batch V/K/KV 219.768/219.100/218.647T，未改善，删除该可执行分支。
	[v99证据](results/d256_pipeline230_20260923/v99-dmabatch-full.json)全部保留，不是丢弃正式慢轮后重试达标。
- 最终[交付说明](results/d256_pipeline230_20260923/README.md)记录可选参数、文件和证据。
	本轮未修改public wrapper、v48共享算术或原timer；未stage/commit/push，旧ATT/UI和其他工作区文件保留。
- [最终独立审计](results/d256_pipeline230_20260923/final-audit.json)40份报告全部核对通过；
	当前DMA源SHA`c260fca5fbf4dc4bc07195d8f676c1c5e499d093dbc3e11a02866c454bb599b4`，
	仅最后模块说明/注释与正式源不同，设备AST相同；[最后factory回归](results/d256_pipeline230_20260923/final-factory.xml)2通过。

## 2026-09-24：v98最快候选身份复核与ATT采集

- 按用户“v98是最快的版本吗，如果是采集att”复核现有v95/v98/v99原始样本：
	v98为本轮最快保留候选组，order1/order2接近；order2探索221.523T，正式配对比1.038459。
	正式全量中位未胜v73，不称稳定最快或230T达标。未重跑性能，默认仍v73。
- 新采集**v98 persistent/order2**，明确启用部分K等待、S7 PV/center融合；不是旧qfast。
	Full Q10240/KV2583/H24/HK2/D256/page64，grid80、block512，GPU2/a4，SE0/CU1/allSIMD，
	第3次调用dispatch393，256MiB ATT buffer，无PMC/activity counters。
- [实际ELF](results/d256_v98_att_20260924/artifact.json)与正式计时版本完全一致：
	`774d2eaba3e4726e2d2ece80ea1b52ac7a9b21d7a01ae44cbab0caf5a224c889`；VG224/SG92、LDS65536、AGPR/private/spills0。
	源IR SHA不同且分别留档，未误称IR恒等；当前源码/默认/原timer均未变。
- [前后数值](results/d256_v98_att_20260924/driver-result.json)原FP32 `.02`通过，acc均2.727154653547892e-6；
	第2/3/4次逐位重复通过。入口/准备/退出gfx/UMC均0，VRAM283/2229/4629MB，PTL Enabled/VECTOR,F8。
- [独立ATT审计](results/d256_v98_att_20260924/ui-copy-audit.json)：8/8完整波，每波24任务，
	3,718,544动态指令、1,007,616 MFMA、254,976 DMA4；CSV/code-map计数一致，全部endpgm，无数据丢失。
- UI已复制到当前MHA目录：[manifest](ui_output_agent_18599_dispatch_393/filenames.json)、
	[统计CSV](stats_ui_output_agent_18599_dispatch_393.csv)。16文件/132,888,374bytes，逐文件SHA一致，
	不覆盖旧UI；为ISA映射，Source为空。详见[本次说明](results/d256_v98_att_20260924/README.md)。
- 保留reserved-M0 warning；没有修改GPU/PTL/时钟/功率/NUMA，没有Git写操作；不以ATT周期冒充性能成绩。

## 2026-09-24：原生D256 linear，KV page1/4（l01–l38）

### 目标、合同与最终状态

- 用户要求Q/K/V/O均为linear `[T,H,D]`，首批支持KV page1/4，接口参考`flash_attn_varlen_func`，V从LDS读后用`v_perm`转置；性能差距不超过10%。
- 新增独立 [API](../flash_attn_api/flash_attn_varlen_d256.py) 与 [原生内核](mha_pa_bf16_256_linear_942.py)，保留原D128 adapter及D256 SHUFFLE默认v73不变。
	支持BF16/gfx942/D256/GQA、无表连续KV及任意物理page表、bottom-right causal、FP32自然log LSE、out/stream/graph；热调用仅一个attention dispatch，无KV预转换。
- 采用原Full B1/Q10240/KV2583/H24/HK2，650033233920有效FLOPs；GPU2/a4，严格gfx/UMC<3%、VRAM≤20%、PTL Enabled/VECTOR,F8；不改GPU策略、不用PMC/ATT、不写Git。
- **功能完成；吞吐口径相对v98均在10%内。** 但预先声明更严格的全中位时延≤1.10倍v98，最终随机page4为**+10.33%**，仍未通过该门槛，不能改用配对比或四舍五入宣布通过。

### 实现与已采纳优化

- BM128/BN64、8wave/4+4错相、M16 MFMA；K/V各32KiB LDS，Q/O和逻辑算术保持原合同。
- K逆XOR DMA；V保留row-major LDS，b128读取后采用CK BF16 2×2相同选择器`0x01000504/0x03020706`。
	用编译器可见`llvm.amdgcn.perm`暴露hazard，不使用gfx950硬件transpose或外部转换kernel。
- 页表每tile一次向量lookahead，K/V复用同组scalar字节偏移；page4每wave连续16token，只缓存4个页基址。
- 两组wave分别写低/高D128；K低D读先退休，V高D读跨barrier时与领先低D DMA不相交。CPU逐字节镜像/WAR/bank服务组/`v_perm`/O payload/任务覆盖证明通过。
- 首K16 V转置放Memory末尾，后续转置与MFMA交错；S5行和/max与PV末段重叠，S7 max/center分段重叠；O采用64KiB CShuffle批量8条LDS read。
- noncausal/noLSE四tile展开降低页偏移队列PHI搬运；causal/LSE只双展开，避免四展开引起的SGPR spill。

### 主要试验（探索均为每候选4样本，不是正式结论）

| 版本 | 改动/结果 | 实测或失败证据 |
|---|---|---|
| l01–l04 | 局部scalar重物化消除12个SGPR spill；opaque perm错误→可见intrinsic；l03缓存仍复用旧ELF，l04实际新ELF后通过 | [l01](results/d256_linear_20260924/l01-smoke.json)、[l04](results/d256_linear_20260924/l04-linear-full.json) 连续177T |
| l05 | 只有末Vtile需OOB归零；删除稳态32次tail比较/select | [连续204T](results/d256_linear_20260924/l05-linear-full.json)；[随机page1](results/d256_linear_20260924/l05-paged1-full.json)146T、[page4](results/d256_linear_20260924/l05-paged4-full.json)96T |
| l06–l09 | SMEM查表/有符号除法→向量页请求→三tile预取→预计算byte offset→成组readlane填依赖间距 | [l09 page1](results/d256_linear_20260924/l09-lane-batch-p1.json)185T、[page4](results/d256_linear_20260924/l09-lane-batch-p4.json)185T |
| l10–l14 | wave低/高D分工与WAR分批退休；偏移准备重叠compute；M0→DS→DMA | [l11](results/d256_linear_20260924/l11-compute-offsets-p1.json)188T；order2及局部PV配对无改善 |
| l15–l19 | 转置提前一步与MFMA交错，首步移Memory，S7 center交错；head-major任务序 | [l19 page1](results/d256_linear_20260924/l19-headmajor-p1.json)197T、[page4](results/d256_linear_20260924/l19-headmajor-p4.json)199T |
| l20–l24 | 显式QK交错无收益；O批量read、K/V scalar偏移复用、S5末PV与行统计交错 | [l23 page1](results/d256_linear_20260924/l23-reuse-offsets-p1.json)203T；[l24](results/d256_linear_20260924/l24-summary-overlap.json) dense218T/paged203T |
| l25–l30 | S1/S3偏移分摊无收益；双展开→page4页内复用→四展开；causal/LSE按资源保留双展开 | [l29](results/d256_linear_20260924/l29-four-unroll.json) paged约213T；[l30资源回归](results/d256_linear_20260924/l30-resource-functional.xml)98通过 |
| l31–l38 | 八展开、M0四复用、末K裁剪、直接O、DS中途perm、三展开、早exp、分散max均未优于保留版本 | [完整68报告审计](results/d256_linear_20260924/final-audit.json)，全部保留，当前恢复精确l30源码 |

- l17误用只拼FP32的`_join`拼Int32位串，ISA出现有损int→float→int，修复为Int32拼接后通过；l38每step解包覆盖已更新row_sum，修复后仍较慢，均保留原失败。
- l13 page4 prepared gfx5%门禁失败，未计时，不重试绕过门禁。l32汇编offset修饰符位置错误，两次未计时编译失败后按完整日志修复；M0复用微小收益但VGPR255未采用。
- **早期l01–l04的GB/s无效：驱动`logical_io_bytes=1`占位错误。** TFLOPS分母和原始时延未变，l05起字节数修为256948224；历史JSON不改写。

### 正式50样本/候选，全量300事件

[l30-all-full-50.json](results/d256_linear_20260924/l30-all-full-50.json)，10独立buffer×5轮、warmup10、repeat1，6候选按round+buffer正反交替，原`cudaPerf`不变。
构造各原生输入格式、JIT、分配和独立FP32参考均在事件外；linear完整热API调用含在事件内，实际只有一个attention kernel。

| 候选 | 全中位µs | TFLOPS | 时延/v98增幅 | TFLOPS/v98降幅 | 严格时延≤1.10 |
|---|---:|---:|---:|---:|---|
| v73/default | 4233.508 | 153.545 | +0.95% | — | 对照 |
| v98/order2 | 4193.687 | 155.003 | 基准 | 基准 | 对照 |
| dense page1 | 4374.608 | 148.592 | +4.31% | 4.14% | 通过 |
| dense page4 | 4481.209 | 145.058 | +6.86% | 6.42% | 通过 |
| 随机page1 | 4603.270 | 141.211 | +9.77% | 8.90% | 通过 |
| 随机page4 | 4627.090 | 140.484 | **+10.33%** | **9.37%** | **未通过** |

- 相对当前默认v73，随机page1/page4时延+8.73%/+9.30%。无表page1/4是相同ELF；两组原生allocation与采样条件不同，不能把时差归因为page算法。
- 同round/buffer配对时延比1.040160/1.038087，不等于全中位数比1.097667/1.103346，不替代验收。
- 两个早期完整正式轮：[l23](results/d256_linear_20260924/l23-all-full-50.json) paged+10.50%/+10.99%；
	[l27](results/d256_linear_20260924/l27-all-full-50.json)+9.65%/+10.36%；均保留所有慢尾。后续重测有实质代码变化，不重试同版直到快。
- 最终原v73/v98及linear都出现共同变速；4次只读遥测gfx1731–1739→1321–1324→1195–1196→1180–1182MHz，wrapper重叠56/43/38/39次。
	不证明逐kernel时钟或确切降速原因。entry/prepared/sampling/exit gfx0/2/2/2、UMC0、VRAM≤6.3797%、PTL不变，门禁通过不等于固定频率/独占。

### 最终数值、资源与源码身份

- [98项JUnit](results/d256_linear_20260924/l30-resource-functional.xml)0失败/0跳过，445.210秒，包含causal/LSE、page1/4真实随机/逆序、ragged/空Q、KV1…129、多tile、不同GQA、NaN尾部/guard、softmax尺度、rare rescale、stream/graph及metadata重验/拒绝。
	O原容差`.02`、LSE`.002`、两次逐位重复不变；所有本次测试实际编译特化的private/VGPR spill/SGPR spill均0。
- 正式60组候选/buffer原FP32和v98逐位对照通过。Full dense VG235/SG62、paged1 VG242/SG106、paged4 VG247/SG68，全部LDS65536、AGPR/private/spills0；未隐藏reserved-M0编译warning。
- 当前内核逐字节等于正式冻结源，SHA`ac3fcf440f737f7c7f7e4ec439b99d0254efa266726d6ccd0722cd62c5b59e77`。
	Full ELF dense `b38f7dff1f57a739fe0478a25552be839be9194dbd8f2ce85159cda4d536fff1`、paged1 `a94eda5e4ee23ac6c20ce7ed4ac0c74d3870749938af978e5ede65ce7ce891fc`、paged4 `ef9800a0842c6437797c201492f6a504c7141c5379f56c34e8bbc9b43316178b`。
- [独立CPU审计](results/d256_linear_20260924/final-audit.json)核对68份报告、1684个保留事件、ELF/资源/哈希、正式候选顺序、门禁、遥测、JUnit及布局/WAR/页寻址/循环覆盖。
	[接口与使用说明](../flash_attn_api/README.md)、[本轮完整交付](results/d256_linear_20260924/README.md)明确支持范围、metadata warmup与inference-mode限制，以及严格page4时延门槛仍未达成。