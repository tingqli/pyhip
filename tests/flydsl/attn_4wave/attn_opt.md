# 融合 Attention 双 GEMM 优化总表（当前维护）

状态：**ACTIVE / 唯一持续维护入口**
更新时间：2026-07-30
当前实现：[`../test_attn_gemm.py`](../test_attn_gemm.py)
JIT实现：[`test_attn_gemm_jit.py`](test_attn_gemm_jit.py)
co-issue工具：[`tools/test-coissue.py`](tools/test-coissue.py)
co-issue说明：[`tools/mfma-valu-coissue.md`](tools/mfma-valu-coissue.md)

以后所有新的优化方法、失败反证、ISA资源和严格性能结果都写入本文，不再更新归档文档。每次实验至少更新：

1. 对应路线的决策表；
2. 第 7 节最终性能快照；
3. 第 8 节增量实验日志；
4. 若改变当前实现，同时更新正确性、ISA资源和ATT结论。

## 使用与复现

以下命令均从仓库根目录执行。性能复现前先用`rocm-smi --showuse`选择空闲GPU，并固定使用新的
`PYTHONPYCACHEPREFIX`/`PYHIP_CACHE_DIR`，避免旧缓存混入结果。

### Fly stage-antiphase

小shape正确性：

```bash
HIP_VISIBLE_DEVICES=0 \
H=1 MULT=2 \
PYTHONPYCACHEPREFIX=/tmp/attn-fly-small-pyc \
FLYDSL_RUNTIME_ENABLE_CACHE=0 \
python3 -B tests/flydsl/test_attn_gemm.py
```

预期`rel_l2≈0.00315`。40960性能：

```bash
HIP_VISIBLE_DEVICES=0 \
H=1 MULT=320 \
PYTHONPYCACHEPREFIX=/tmp/attn-fly-40960-pyc \
FLYDSL_RUNTIME_ENABLE_CACHE=0 \
python3 -B tests/flydsl/test_attn_gemm.py
```

当前gfx942空闲卡复现约`3907--3912 us / 219.7--219.9 TFLOPS`，`rel_l2≈0.00319`。

### PyHIP JIT

两个入口的小shape正确性：

```bash
HIP_VISIBLE_DEVICES=0 H=1 MULT=2 CHECK=1 \
ATTN_JIT_KERNEL=production PYHIP_CACHE_DIR=/tmp/attn-jit-production-small \
python3 -B tests/flydsl/attn_4wave/test_attn_gemm_jit.py

HIP_VISIBLE_DEVICES=0 H=1 MULT=2 CHECK=1 \
ATTN_JIT_KERNEL=setprio_best PYHIP_CACHE_DIR=/tmp/attn-jit-setprio-small \
python3 -B tests/flydsl/attn_4wave/test_attn_gemm_jit.py
```

40960性能会自动跳过显存过大的torch参考；也可显式设置`CHECK=0`：

```bash
HIP_VISIBLE_DEVICES=0 H=1 MULT=320 CHECK=0 \
ATTN_JIT_KERNEL=production PYHIP_CACHE_DIR=/tmp/attn-jit-production-40960 \
python3 -B tests/flydsl/attn_4wave/test_attn_gemm_jit.py

HIP_VISIBLE_DEVICES=0 H=1 MULT=320 CHECK=0 \
ATTN_JIT_KERNEL=setprio_best PYHIP_CACHE_DIR=/tmp/attn-jit-setprio-40960 \
python3 -B tests/flydsl/attn_4wave/test_attn_gemm_jit.py
```

历史稳定口径分别为`208.6--208.8T`与`236.5--237.1T`。性能测试使用10套buffer、10次预热和50次
CUDA-event样本的中位数。

### MFMA/VALU co-issue

快速执行检查：

```bash
HIP_VISIBLE_DEVICES=0 PYHIP_CACHE_DIR=/tmp/attn-coissue-smoke \
python3 -B tests/flydsl/attn_4wave/tools/test-coissue.py \
	--ops v_add_f32 --outer-loops 2 --inner-unroll 8 \
	--samples 1 --warmup 0 --throughput-only
```

正式gfx942复现：

```bash
HIP_VISIBLE_DEVICES=0 PYHIP_CACHE_DIR=/tmp/attn-coissue-formal \
python3 -B tests/flydsl/attn_4wave/tools/test-coissue.py \
	--ops all --outer-loops 1000 --inner-unroll 1000 \
	--samples 5 --warmup 1 --tolerance 0.25 \
	--json /tmp/attn-valu-intra-inter-coissue.json
```

完整指标定义、PC对齐要求和gfx942结果见[`tools/mfma-valu-coissue.md`](tools/mfma-valu-coissue.md)。

## 1. 问题、口径与状态定义

目标是一个kernel内完成：

```text
GEMM1: S = Q[M,D] @ K[N,D]^T
softmax: P = softmax(S / sqrt(D))
GEMM2: O = P[M,N] @ V[N,D]
```

当前主验证shape：

```text
gfx942 / MI308X
H=1, M=N=40960, D=128, BM=128, BN=32, bf16
```

性能口径：空闲GPU、多个buffer轮换、CUDA event计时。正式A/B使用`C-X-X-C`，相邻control漂移门限0.5%。
不同shape的绝对TFLOPS不能直接横比；650W功耗墙导致共同slow态时，只采纳同组时间比。

状态定义：

- **当前采纳**：保留在当前219T源码路径。
- **历史采纳**：曾进入对应阶段主线，后被更优版本取代。
- **仅验证**：作为机器码oracle或机制证据，不是当前高层实现。
- **失败**：精度、资源或严格性能已否决。
- **中性**：严格结果在噪声内，不保留复杂度。
- **待验证**：只有局部证据，没有完成正确性、资源、严格A/B和必要ATT闭环。

## 2. 无softmax双GEMM路线

该路线建立了当前kernel的数据布局、register trick、宽访存和K流水基础。早期数据使用8192，v8以后主要使用20480，v13同时验证40960。

| 阶段 | 优化方法 | 资源/精度 | 性能 | 决策与结论 |
|---|---|---|---:|---|
| v1 | `S`经LDS中转；V非合并global读；2个barrier | `rel_l2≈1.3e-4` | 40.3T @8192 | 历史基线；窄访存、LDS往返和无流水共同限制 |
| v2 | register-resident `S`；`K@Q^T` register trick；4-wave M-split | 326V，1 wave/SIMD | 37.6T @8192 | 历史采纳；避免S落全局，但VGPR悬崖使其暂时更慢 |
| v3 | fragment错峰复用；BN 128→64 | 202V，2 waves/SIMD | 38.9T @8192 | 历史采纳；证明必须先释放VGPR预算 |
| v4 | GEMM1 K维加入`k_perm`，窄LDS读改128-bit | 196V | 72.1T @8192 | 历史采纳；首次关键提速 |
| path-b | 放弃register trick，S经LDS，让V走128-bit | 196V | 68.8T @8192 | 失败；S的LDS写读超过V宽读收益 |
| v5 | GEMM1 M维加入`perm_M`，S累加器直接匹配GEMM2 | 200V，`rel_l2≈1.3e-4` | 91.7T @8192 | 历史采纳；register trick与K/Q/V全128-bit可以共存 |
| v6 | V改为协作global→LDS→register | 200V | 90.0T @8192 | 失败；额外LDS和barrier抵消合并收益 |
| v7 | f32→bf16改为`+0x8000`后截断 | 200V，`rel_l2≈1.5e-4` | 97.1T @8192 | 历史采纳；每tile约省96条转换VALU |
| v8 | K/V软件预取，global load与当前计算重叠 | 184V | 163.6T @20480 | 历史采纳；建立软件流水 |
| 显式双套reg ping-pong | 手工展开并保留两组K/V和临时fragment | 234V | 155.3T @20480 | 失败；live range扩张抵消流水收益 |
| v9 | K/V LDS加入`swizzle(3,3,3)` | `rel_l2≈1.9e-4` | 168.6T @20480 | 历史采纳；降低bank conflict |
| 弱/其他swizzle | `(1,3,3)`或其他参数 | 指令形态变化 | 最差94.5T | 失败；弱swizzle显著增加bank conflict |
| v10 | GEMM2转置为`O^T`，输出改64-bit store | 同占用率 | 170.5T @20480 | 历史采纳；输出连续化的小幅稳定收益 |
| v11 | V绕过LDS，paged global直接进入fragment | 去掉V LDS链 | 183.0T @20480 | 历史采纳；V无需像广播K一样经LDS |
| K/V全直读 | 广播K也绕过LDS | K读取4倍冗余 | 49.7--69.3T | 失败；广播K必须走协作加载和LDS |
| v12 | BN 64→32释放VGPR，再加入K LDS双缓冲 | 190V，2 waves/SIMD | 229.8T @20480 | 历史采纳；“先省寄存器，再装流水” |
| BN64 K双缓冲 | 大BN直接增加双缓冲 | 265V，1 wave/SIMD | 115T | 失败；跨occupancy阈值 |
| V ping-pong @BN64 | V跨tile双缓冲 | 344V，1 wave/SIMD | 132T | 失败；VGPR过高 |
| V预取 @BN32 | 增加loop-carried V状态 | 222V，仍2 waves | 210.8T | 失败；未降occupancy仍破坏调度自由度 |
| 2-stage计算流水 @BN32 | 两套计算状态 | 200V，仍2 waves | 208.5T | 失败；live range与全局排程回退 |
| 手写hot-loop分组 | 强制GEMM1/GEMM2调度组 | 190V | 185.6T | 失败；不如LLVM原调度 |
| v13a | `perm_M`从MMA挪到全局K物理布局 | 为后续腾VGPR | 197T @20480 | 单独失败、组合采纳 |
| v13b | KV循环展开2次，LDS stage变编译期常量 | 中间态 | 194T @20480 | 单独失败、组合采纳 |
| v13c | K LDS读跨迭代prefetch，GEMM1消费已就绪fragment | 204V，2 waves，`rel_l2≈0.00021` | 235.8T @20480；266.0T @40960 | 无softmax历史终态；三步组合缺一不可 |

主线经验：增加预取状态前必须先释放VGPR；广播K走LDS，V可直接global；静态合法重排不等于动态ready更优。

## 3. 在线softmax与高层FlyDSL路线

| 阶段/方法 | 机器变化 | 性能/精度 | 决策 | 结论 |
|---|---|---:|---|---|
| 完整multi-head + flash softmax | 在线`m/l/O`，EXP用`v_exp_f32` | 160--164T；`rel_l2≈0.00311` | 历史采纳 | 瓶颈转为VALU/EXP和长期相位 |
| LOG2E折进scale | `exp`改`exp2`并合并常量乘 | +6.6%；160.4T @8192 | 历史采纳 | 删除逐元素LOG2E乘 |
| Q预缩放 | BF16 Q提前乘scale | 158.5T；`rel_l2=0.00358` | 失败 | BF16量化超精度门限 |
| 条件correction/O-rescale | 仅rebase时修正O | 阶段收益+5.2% | 历史采纳 | 降低常见路径工作 |
| lazy rebase `Δ=8` | `tmax > m+8`才更新reference | 166.9T @8192；181.2T @40960 | 历史采纳 | 大幅减少O-rescale |
| 精确two-pass | 先求全局max，再重算QK/PV | 93.0T | 失败 | 重复MFMA与访存代价过大 |
| BN64/双PV pipeline | 更大BN或两套GEMM2 | 128T / 161.2T | 失败 | VGPR、重复工作和occupancy回退 |
| PMC/ATT co-issue建模 | MFMA busy-window与VALU/EXP overlap | VALU覆盖MFMA busy约13.29% | 分析采纳 | 优化目标是有效重叠而非只减指令 |
| scheduler hint/sleep/局部priority | 强制错相或分组 | 150.7--165.1T | 失败 | 破坏VMEM/LDS/MFMA原交错 |
| GEMM1按两个mt拆分 | 两条独立query-row accumulator链 | 166.7T @8192；183.1T @40960 | 历史采纳 | dependency wait小幅下降 |
| inline scalar MUL | opaque `v_mul_f32`替代packed MUL | 162.7T | 失败 | inline边界阻碍全局重排 |
| running-sum标准FMA | `l=fma(l,corr,ts)` | 170.6T @8192；`rel_l2=0.00316` | 历史采纳 | scalar FMA可进入MFMA窗口 |
| packed FMA/延后scale/BF16截断 | 局部减指令 | 161.7T、158.9T或精度失败 | 失败 | packed依赖、调度迁移或舍入误差 |
| softmax(mt0)细粒度穿插 | 穿插进GEMM1(mt1)/GEMM2(mt0) | 157.2--157.9T | 失败 | 扩live range并打断MFMA ILP |
| 8-wave双pipeline | 两组4-wave显式错相 | 150.5T | 失败 | gfx942缺少低成本子组barrier |
| JIT调度移植到高层Fly | K写中置、半量转换、长序列scheduler | 194.4--194.7T @40960；240V | 历史高层终态 | 高层codegen稳定主路径 |
| shape分派 | `N<32768`恢复短序列调度 | 170.6--170.7T @8192 | 历史采纳 | 长序列相位不能机械用于短shape |

## 4. PyHIP JIT、后ISA与机器级调度路线

| 阶段/方法 | 机器变化 | 性能/精度 | 决策 | 结论 |
|---|---|---:|---|---|
| PyHIP JIT双窗口交织 | 精确控制MFMA/VMEM/DS/VALU顺序 | 171.7T | 历史采纳 | 机器级调度可超过高层路径 |
| prepare/center/finish重排 + K写填DS窗 | 移动独立工作，不扩live range | 188.1→194.5T | 历史采纳 | 减少物理no-issue |
| 三路DS fanout | max/sum串行wait改一次wait | 200.3T | 历史采纳 | physical no-issue下降 |
| 半量BF16 pack填窗 | 前移一个n-block pack | 203.1--203.3T | 历史采纳 | 12条/tile接近填满窗口 |
| 全量提前pack | 两个n-block都提前 | 199.1--199.2T | 失败 | 过量填窗延迟sum/FMA |
| e32常量 + rolling offset | 缩短编码并调整长期相位 | 205.3--205.5T | 历史采纳 | 仍2 waves、零spill |
| next-K LDS读入GEMM2 shadow | `MFMA→DSRD→MFMA` | 206.4--206.5T | 历史采纳 | 减少LDS暴露 |
| max覆盖VMEM wait | max放入`vmcnt(10/2)`之间 | 206.5T | 历史采纳 | 减少VMEM暴露 |
| 两MFMA间3 ALU+1 EXP | 局部bundle | 207.9--208.3T | 历史采纳 | 微基准与kernel均有效 |
| 全局复制6组bundle | 机械扩大局部最优 | 200.7--201.0T | 失败 | probability EXP与双wave相位恶化 |
| GEMM1区整体填窗 | K地址XOR、V offset滚动等 | 208.6--208.8T | JIT production历史终态 | 生产JIT归档入口 |
| 零延迟DAG wavefront重排 | 追求更高静态重叠 | 最快204.1T | 失败 | 静态合法不等于物理ready |
| 四阶段多次setprio | 多窗口priority切换 | 不稳定或回退 | 失败 | 切换成本和断续窗口破坏流水 |
| 单次连续priority窗口 | GEMM1 mt0中段升权，跨softmax后降权 | 3623.5us / 237.1T；`rel_l2=0.00318646` | 历史最佳JIT | 长期resident-wave反相成立 |
| A/V同ISA消融 | 156V+64A机械改220V+0A | 236.56T，两者相同 | 仅验证 | A/V类别不是差距根因 |
| JIT ISA oracle（Fly ABI） | 改为Fly 164-byte ABI，热循环不变 | 3630.9us / 236.6T | 仅验证 | ABI不是差距根因 |
| strict Fly后ISA | identity max、pack、绝对priority三变换 | 205.2T | 历史采纳 | 形态严格绑定当前Fly ISA |
| raw/formal p0接管 | 接管scale/max/DS/SUB/EXP | 相对205T -5.18%至-9.67% | 失败 | shadow外空洞增加 |
| 对称pair p0接管 | 对称搬迁局部链 | -6.91% | 失败 | 未处理完整release/backedge |
| wait/fanout/priority微消融 | 删wait、旋转RAW链等 | 0至-7.73% | 失败/中性 | wait与局部链共同塑造相位 |
| Fly源码局部inline asm | max、pack、priority、MFMA块 | -4.31%至-10.88% | 失败 | opaque SSA改变寄存器和全局排程 |
| 完整JIT body inline | Fly外壳 + 单asm主流程 | 3633.1us / 236.4T | 仅验证 | 大边界可保持JIT节奏，但非高层codegen |

## 5. running-sum、packed指令与周期priority路线

| 方法 | ISA/资源 | 严格性能 | 决策 | 根因 |
|---|---|---:|---|---|
| 高层base | 240V；packed add/mul + scalar FMA | 194.1--194.6T | 历史默认 | 稳定基线 |
| 只后移跨lane shuffle | 254V；新增packed FMA | 181.7--182.0T | 失败 | packed FMA位于MFMA关键区 |
| shuffle后移 + scalar inline FMA | 232V；目标packed清零 | 189.5--189.7T | 失败 | 比base慢2.54% |
| 完整后移sum | 268V，1 wave/SIMD | 143.0--143.6T | 失败 | packed算术增加且越过256阈值 |
| 完整后移 + inline FMA | 246V，恢复2 waves | 191.1T | 失败 | 相对base仍慢1.96% |
| shuffle-inline late-scale | 234V，删除packed scale/SUB | 189.8--190.1T | 局部有效但未采纳 | 相对自身control约+0.3%，仍慢于base |
| full-inline late-scale | 248V | 约189.4T | 单独失败 | 慢slot暴露K LDS延迟 |
| full-late + `56:0,88:2` | 248V，只增2条setprio | 约198.5T；相对full-late +4.92% | 历史实验采纳 | slot完成差976.9→7.8 cycles/tile |
| V-load填phase窗 | 7个候选 | 未长测 | 正确性失败 | 目标VGPR仍是MFMA活值 |
| K-read提前 | 9个逐元素一致候选 | 全部回退0.27%--1.55% | 失败 | 破坏DS/MFMA请求年龄和slot相位 |

### 5.1 专项失败、中性与待验证完整索引

下表逐条迁移归档文档第23.5节。它们没有形成独立主线里程碑，但都约束了后续决策；“待验证”只表示历史状态，不代表当前代码仍保留入口。

| 专项思路 | 结果/性能 | 决策 | 可复用结论 |
|---|---:|---|---|
| V跨tile寄存器双缓冲 | 154V+96A，仍2 waves；196.9--197.0T | 失败 | occupancy不变也不代表live range和VMEM相位无成本 |
| DPP `wave_rol:1`列归约 | 384 rotate + 392 hazard NOP；114.6T | 失败 | gfx942缺少一次rotate16/32，DS fanout更合适 |
| 三次`wave_shr` + `readlane` | lane结果与目标列归约不符 | 精度失败 | 相邻lane窗口不等于目标`q/q+16/q+32/q+48`布局 |
| lane内max/sum平衡树 | max稳定回退；sum在噪声内 | 失败/中性 | 更短依赖树未必改善跨lane与双wave调度 |
| 两个probability n-block全部提前pack | 199.1--199.2T | 失败 | 24条/tile超过DS等待窗，延迟后续sum/FMA |
| 一个n-block提前pack | 203.1--203.3T | 历史采纳 | 12条/tile基本填满可用DS窗口 |
| correction寄存器别名 | 152V；202.7--202.9T | 失败 | 少move/少VGPR不等于更短物理关键路径 |
| pack跨MFMA拆分 | 202.2--202.8T | 失败 | 局部邻接变化破坏既有填窗相位 |
| 提前round-add | 202.2--202.8T | 失败 | 缩短编码不足以抵消调度变化 |
| 8-byte对齐 | 202.2--202.8T | 失败 | 对齐变化没有形成吞吐收益 |
| V load批量前压2条 | 201.9T | 失败 | 缩短VMEM成熟距离并增加队列竞争 |
| V load批量前压4条 | 195.7T | 失败 | 批量前压进一步恶化VMEM队列竞争 |
| SGPR stride缩码 | 204.4--204.6T | 失败 | 机器码更小不是吞吐充分条件 |
| 循环上界缩码 | 204.4--204.6T | 失败 | SALU编码减少未改变关键路径 |
| `J.emit(center)`预算10→11 | 169.1T | 失败 | 预算11完整取第三条5-cycle FMA并挤占下一MFMA |
| 全局6组`MFMA→3ALU→MFMA→EXP` | 200.7--201.0T | 失败 | 微基准局部最优不能机械复制到全循环 |
| 第二个局部ALU/EXP pair | 206.3T | 失败 | 提前p0 EXP破坏后续probability EXP相位 |
| V load中置 | 回退约0.6--1.2T | 失败 | V到GEMM2的成熟距离变短 |
| V load lookahead | 回退约0.6--1.2T | 失败 | 请求年龄与消费点关系恶化 |
| future-K进一步提前 | 203--204.6T | 失败 | VMEM队列压力和请求年龄顺序恶化 |
| threshold预计算填GEMM1空窗 | 206.2T或中性 | 失败/中性 | 增加VGPR或读取未完成n-block，不是真独立工作 |
| max预计算填GEMM1空窗 | 206.2T或中性 | 失败/中性 | 提前计算受数据ready约束 |
| 四个offset全部滚动 | 无稳定收益 | 中性 | 只保留曾与K地址XOR协同的V-offset滚动 |
| GEMM1 split-K双accumulator | 约203T；增加8--16V和8--16 ADD | 失败 | 缩短RAW链的代价超过收益 |
| 完整n-block wavefront | 198.3--204.1T | 失败 | 真实release链不等于零延迟DAG |
| 半n-block wavefront | 最快204.1T，仍-1.80% | 失败 | EXP/sum/pack/rescale与live range决定动态ready |
| 半wavefront功耗态复测 | fast +0.76%，全体-0.23%，slow -0.29% | 失败 | DPM双态下必须依赖紧邻control，局部正样本不足采纳 |
| 粗四阶段setprio | -15.57%；去priority后-21.16% | 失败但机制有效 | priority本身约+6.47%，主损失来自粗阶段切分 |
| 15种fine priority mask | 最佳`0x7`仍-1.23%，全开-9.28% | 失败 | 多次切换和最后K-read区提权破坏互补工作 |
| `start=3,end=15`早期样本 | 初看+1.06%，严格门限后-0.03% | 假阳性 | 必须使用`C-X-X-C`和0.5%漂移门限 |
| 单次连续priority窗口`7→15` | 237.1T | 历史JIT采纳 | 减少切换并维持跨softmax长期反相 |
| Fly JIT式长priority窗口 | 194.2→182.3T，-6.14% | 失败 | 边界不能脱离JIT精确机器顺序移植 |
| Fly mt分片、wait 49→41 | 226V；184.2T，-5.4% | 失败 | wait/VGPR下降未改变occupancy，长期相位更差 |
| raw/formal p0接管 | 相对205T -5.18%至-9.67% | 失败 | 单边搬迁增加shadow外空洞和双wave长团重叠 |
| 对称pair p0接管 | -6.91%；shadow外+114.829 cycles/tile | 失败 | 对称局部链仍未处理完整release/backedge |
| priority终点96→104 | -4.57% | 失败 | priority窗口长度与resident-wave相位高度耦合 |
| 独立M16/M17交换 | +0.023% | 中性 | 提前一条MFMA释放不足以改变墙钟 |
| 回边GEMM2 RAW链旋转 | -0.29% | 失败 | 单条chain重排不能改善全局ready关系 |
| strict Fly max-only三路fanout | 240V→244V，-7.73% | 失败 | 减wait同时改变寄存器和双wave相位 |
| GEMM1 K wait压缩6条 | -0.42% | 失败 | 数值冗余wait仍参与塑造物理调度相位 |
| 局部inline max | -4.31%至-10.88% | 失败 | opaque inline改变SSA、寄存器和LLVM全局调度 |
| 局部inline pack | -4.31%至-10.88% | 失败 | 局部指令固定化破坏全局排程 |
| 局部inline priority | -4.31%至-10.88% | 失败 | native/inline priority边界均会约束调度 |
| 局部inline MFMA块 | -4.31%至-10.88% | 失败 | 小inline边界不足以保留完整JIT节奏 |
| 完整JIT body inline | 236.4T；与oracle时间比1.00025 | 仅验证 | 足够大边界可保机器节奏，但不是高层原生codegen |
| softmax1 prepare预执行31/35/43 cycles | 小shape精度/资源通过，未完成严格A/B | 历史待验证，入口已删 | 只有保持DS fanout、threshold和K写成熟距离时才值得重建 |
| persistent grid（有尾批shape） | 40960的320 WG整除80 CU | 历史待验证，入口已删 | 只应在实际尾批shape评估 |
| xor32地址复用 | 未独立严格A/B | 历史待验证，入口已删 | 不与已回退的raw-FMA/max-fanout混合判断 |

## 6. 最新219T旋转反相流水（当前唯一实现）

### 6.1 数据流与阶段边界

当前源码复用原有`kv_step()`的7项loop state，不跨回边携带`frag_V`。每个静态偶/奇substep：

1. **stage1续段**：读取`V(i)`，8条global load与`GEMM1(i)`的32条MFMA交织；
2. `s_setprio(0)`，结束stage1；
3. **stage0**：global预取`K(i+2)`，执行lazy softmax、running-sum和O correction；
4. probability转BF16；`s_setprio(2)`结束stage0；
5. **stage1起段**：`K(i+1)`写LDS，与`GEMM2(i)`交织；barrier后从LDS读K到`frag_K`；
6. stage1跨runtime loop回边，延续到下一substep的V load + GEMM1结束。

这与旧高性能流水的生产/消费关系一致：K global预取放在softmax段，V预取放在其消费GEMM1的开头，K LDS写读围绕GEMM2并跨到下一次GEMM1。

### 6.2 失败的中间版本

| 版本 | 精度/资源 | 性能 | 结论 |
|---|---|---:|---|
| flat `vector<8xf32>` + 旧专用stage helper | small `rel_l2=33.37368` | 无效 | 失败；不能修复旧stage状态结构 |
| 专用stage helper，跨回边携带V | 288V | 139.0T | 失败；V live range过长，1 wave/SIMD |
| 旋转到原`kv_step`，原生逐元素vector FMA | 260V | 约139T | 失败；越过2-wave门槛 |
| 同一旋转流水，priority=0 | 268V | 140.5T | 失败；无反相且寄存器更高 |
| 旋转到原`kv_step`，inline vector FMA | 244V | 219T级 | **当前采纳** |

### 6.3 正确性、资源与性能

```text
M=N=256:   rel_l2 = 0.00315
M=N=40960: rel_l2 = 0.00319

private_segment_fixed_size = 0
next_free_vgpr = 244
accum_offset = 244
static MFMA = 128
static s_setprio = 6
```

128条静态MFMA是完整计算：runtime loop body有两个静态substep，每个substep包含32条GEMM1和32条GEMM2。

最终24组`C-X-X-C`复测：

```text
valid = 22/24
control interleaved = 4554.8 us / 188.6T
current rotated     = 3919.6 us / 219.2T
time_ratio = 0.86095
speedup = 16.15%
candidate rel_l2 = 0.00319
```

单跑复验为3911.3us / 219.7T；ATT采集时为3908.7us / 219.8T。

### 6.4 ATT阶段结果

Trace：`/tmp/attn-stage-final-rotated/ui_output_agent_6508_dispatch_84`

| 指标（cycles/tile） | slot0 | slot1 |
|---|---:|---:|
| stage0 | 1220.127 | 1214.690 |
| stage1稳态 | 1323.231 | 1487.797 |
| wave总时长 | 2544.894 | 2702.900 |

```text
异stage重叠占比 = 87.094%
异stage重叠     = 2214.101 cycles/tile
同stage重叠     = 327.918 cycles/tile
completion skew = 160.881 cycles/tile
```

旧专用stage的stage1约2469--2530 cycles/tile，新stage1降至1323--1488；stage0/stage1由约1:2.4改善到约1:1.09--1.22。剩余问题是slot1额外约165 cycles/tile，而不是stage1整体过长。

## 7. 最终性能快照

“当前保留”只指当前源码中的219T旋转流水；其余行是已经迁移进本文的历史终点，不再保留对应代码分支。

| 路径 | shape与口径 | 时间 | TFLOPS | 资源 | 精度 | 定位 |
|---|---|---:|---:|---|---:|---|
| **当前Fly旋转反相** | softmax，40960，严格A/B | **3919.6us** | **219.2T** | 244V，2 waves，零scratch | `0.00319` | **当前唯一保留实现** |
| 当前Fly旋转反相单跑 | softmax，40960 | 3911.3us | 219.7T | 同上 | `0.00319` | 单跑复验 |
| Fly无softmax v13c | 40960 | 约3230us | 265.2--266.0T | 204V | `≈0.00021` | 历史无softmax终态 |
| 高层Fly旧默认 | softmax，40960 | 约4415--4425us | 194.4--194.7T | 240V | `≈0.00319` | 历史基线 |
| full-late + periodic setprio | softmax，40960 | fast约4328us | 约198.5T | 248V | `≈0.00319` | 历史实验终态 |
| strict Fly后ISA | softmax，40960 | 约4185us | 205.2T | 2 waves，零spill | `≈0.00319` | 历史严格变换 |
| PyHIP JIT production | softmax，40960 | 约4115us | 208.6--208.8T | 156V+64A | `≈0.00319` | 历史生产JIT |
| PyHIP JIT setprio_best | softmax，40960 | 3623.5us | 237.1T | 156V+64A | `0.00318646` | 历史全项目最高性能 |
| JIT ISA oracle / body inline | softmax，40960 | 3630--3633us | 236.4--236.6T | 220V | `≈0.00319` | 仅验证机器目标 |

注意：早期“189.1T→209.1T”结果是control覆盖候选输出造成的假阳性，候选实际精度错误；**209.1T绝对不能引用或采纳**。

## 8. 增量实验日志

以后从这里按时间追加，不删除失败项。建议模板：

```markdown
### YYYY-MM-DD：实验名

- 假设：
- 改动：
- 候选自身精度：
- ISA：VGPR / scratch / MFMA / setprio：
- 严格A/B：control / candidate / valid pairs / speedup：
- ATT或PMC：
- 决策：采纳 / 失败 / 中性 / 待验证：
- 产物路径：
```

### 2026-07-30：stage边界旋转与旧pipeline预取风格统一

- 假设：把K global预取移入softmax段，把V load放在GEMM1开头，并让K LDS写读围绕GEMM2且跨回边延续到下一GEMM1，可缩短stage1并保持2 waves/SIMD。
- 改动：复用原`kv_step`状态；stage1跨回边；不携带V；保留原VMEM/DS scheduler。
- 正确性：256为`0.00315`，40960为`0.00319`。
- ISA：244V、零scratch、128 MFMA、6 setprio。
- 严格A/B：188.6T→219.2T，22/24有效，+16.15%。
- ATT：stage1 2.5K→1.32--1.49K cycles/tile；异stage重叠提升到87.094%。
- 决策：**采纳为当前唯一实现**。

### 2026-07-30：删除219T以外的运行分支

- 删除：无softmax、interleaved control、7种sum-reduce、周期post-ISA、JIT ISA oracle、JIT body inline、backend compare及`C-X-X-C`实验分发。
- 保留：219T kernel、`H/MULT`尺寸参数、小尺寸精度检查和多buffer中位数性能测试；scheduler固定为219T长序列排法。
- 代码规模：`test_attn_gemm.py`从约1300行收敛到595行；旧JIT/ISA源码和归档ISA只作为历史资料保留，不再由当前入口导入。
- 机器码：清理前后均为1015条可执行指令，逐条0差异；244V、零scratch、128 MFMA、6 setprio不变。
- 清理后复验：256 `rel_l2=0.00315`；40960 `rel_l2=0.00319`，3907.3us / 219.8T。
- 静态检查：Black与Ruff通过；Pylance仍会对FlyDSL动态Tensor/fragment/`range(init=...)` API报告类型误报，以实际编译和GPU运行结果为准。
- 决策：**采纳；当前源码只保留219T运行路径**。

### 2026-07-30：V global load与GEMM1交织ATT分析

- 分析区域：`fx.copy(cp_vg, ... frag_V)`、`frag_St.fill(0)`和随后两组`gemm1_mt()`。
- Trace：`/tmp/attn-stage-final-rotated/ui_output_agent_6508_dispatch_84`；其40960可执行ISA与清理后kernel逐条一致。
- 实际机器顺序：源码虽然先写V copy再写GEMM1，最终ISA已经将8条`buffer_load_dwordx4`分散进32条GEMM1 MFMA；当前scheduler配额为8组`VMEM1 + MFMA3`，剩余MFMA留在尾部。
- 首批resident wave每tile统计：8条V load合计stall约9--11 cycles；32条GEMM1 MFMA约293--420 cycles。选区内独立`vmcnt`约0--24 cycles，渐进K `lgkmcnt`固定约32 cycles。V load本身不是首批wave主瓶颈。
- 首批resident wave物理SIMD issue-union：选区约1215.233 cycles/tile，物理no-issue为372.940 cycles/tile（30.69%）。主要组合是`MFMA+VALU` 146.229、`MFMA+MFMA` 83.642、`MFMA+scheduler/ready` 74.778 cycles/tile，而不是两条wave同时卡V。
- 后续wave物理no-issue为608.908 / 1431.814 cycles/tile（42.53%）；其中`VMEM-load+VMEM-wait`为212.689 cycles/tile。热点集中在第5/6条V load与另一wave的`vmcnt(9)`，说明后续wave存在VMEM队列背压，但静态scheduler同时作用于首批/后续wave。
- 成熟距离：最后一条V load到当前GEMM1结束仍约175--237 cycles，之后还有完整stage0才由GEMM2消费，当前V数据依赖没有暴露等待。
- 可证伪实验：把scheduler改为8组`VMEM1 + MFMA4`，试图将8条load均匀铺满32条MFMA。256/40960精度仍为0.00315/0.00319，资源仍244V、零scratch、128 MFMA、6 setprio，但性能降为4152.4us / 206.9T。
- 恢复验证：原`VMEM1 + MFMA3`回到3907.3us / 219.8T。
- 决策：**不采纳更均匀交织**。尾部连续MFMA为V请求提供成熟距离并参与resident-wave反相；物理空洞主要是MFMA/VALU ready关系。若继续，只能围绕第5/6条load做单点、相位感知的机器调度实验，不能全局增加MFMA配额或机械lookahead。

## 9. 后续方向与硬约束

下一步先分析最终ATT中slot1额外约165 cycles/tile：

1. K LDS write→barrier→K LDS read；
2. K LDS read→GEMM1的`lgkmcnt`等待；
3. V global load与GEMM1交织段的`vmcnt`等待；
4. 两个静态substep是否存在不对称机器调度。

硬约束：

- 保持两个`vector<8xf32>` running-sum，不做scalar提前归约。
- 保持7项loop state，不跨回边携带`frag_V`。
- 当前完整静态MFMA门槛是128，scratch必须为0，VGPR必须不超过256。
- 不插NOP补延迟，不机械提前K read，不只压缩scale/EXP链。
- 每次顺序：候选自身精度 → ISA资源 → 40960严格A/B → 必要时ATT/PMC。