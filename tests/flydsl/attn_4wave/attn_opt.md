# 融合 Attention 双 GEMM 优化总表（当前维护）

状态：**ACTIVE / 唯一持续维护入口**
更新时间：2026-08-01
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
H=1 BM=128 MULT=2 D=128 \
PYTHONPYCACHEPREFIX=/tmp/attn-fly-small-pyc \
FLYDSL_RUNTIME_ENABLE_CACHE=0 \
python3 -B tests/flydsl/test_attn_gemm.py
```

预期`rel_l2≈0.00315`。40960性能：

```bash
HIP_VISIBLE_DEVICES=0 \
H=1 BM=128 MULT=320 D=128 \
PYTHONPYCACHEPREFIX=/tmp/attn-fly-40960-pyc \
FLYDSL_RUNTIME_ENABLE_CACHE=0 \
python3 -B tests/flydsl/test_attn_gemm.py
```

当前gfx942空闲卡复现约`3543.5 us / 242.4 TFLOPS`，`rel_l2≈0.00319`。

BM64、每wave 16行、D192性能：

```bash
HIP_VISIBLE_DEVICES=0 \
H=1 BM=64 MULT=640 D=192 \
PYTHONPYCACHEPREFIX=/tmp/attn-fly-bm64-d192-pyc \
FLYDSL_RUNTIME_ENABLE_CACHE=0 \
python3 -B tests/flydsl/test_attn_gemm.py
```

当前gfx942空闲卡两次为`11482.6--11559.7 us / 112.1--112.4 TFLOPS`，`rel_l2≈0.00315`。

FP8 Q/K/V（gfx942原生E4M3FNUZ，BF16输出）：

```bash
HIP_VISIBLE_DEVICES=0 \
H=1 BM=128 MULT=320 D=128 QKV_DTYPE=fp8 \
PYTHONPYCACHEPREFIX=/tmp/attn-fly-fp8-pyc \
FLYDSL_RUNTIME_ENABLE_CACHE=0 \
python3 -B tests/flydsl/test_attn_gemm.py
```

当前gfx942空闲卡为`3592.1 us / 239.1 TFLOPS`。40960下相对等价online-FP8 reference的
`rel_l2=0.00027`；小shape相对标准未量化attention为`0.02204`。

FP8 D192：

```bash
HIP_VISIBLE_DEVICES=0 \
H=1 BM=128 MULT=320 D=192 QKV_DTYPE=fp8 \
PYTHONPYCACHEPREFIX=/tmp/attn-fly-fp8-d192-pyc \
FLYDSL_RUNTIME_ENABLE_CACHE=0 \
python3 -B tests/flydsl/test_attn_gemm.py
```

当前gfx942空闲卡为`4115.5 us / 313.1 TFLOPS`，相对等价online-FP8 reference的
`rel_l2=0.00030`。

32x32 MFMA（仅`BM=128`）：

```bash
HIP_VISIBLE_DEVICES=0 \
H=1 BM=128 MULT=2 D=128 MMA_MN=32 QKV_DTYPE=bf16 \
FLYDSL_RUNTIME_ENABLE_CACHE=0 \
python3 -B tests/flydsl/test_attn_gemm.py
```

BF16使用`MFMA(32,32,8)`，FP8使用`MFMA(32,32,16)`。`BM=64 + MMA_MN=32`
会在构建阶段直接拒绝。GPU 0计算利用率为0%时的40960测试使用`CHECK=0`跳过大矩阵reference；
该卡仍有外部进程常驻94% VRAM，因此结果作为共享显存环境数据单独记录。

MMA32 D128保持与MMA16相同的stage语义。BF16读取`HW_ID.WAVE_ID`给两个resident slot分配
`1/3`与`0/2`四级priority；FP8保持原priority。D128专用scheduler使用GEMM1
`VMEM:MFMA=1:1`和GEMM2 `DSRD:MFMA=1:1`，BF16/FP8分别在DSRD前放3/0条MFMA。
BF16 D128最终为`3714.5--3723.9us / 230.7--231.3T`；D192恢复原scheduler。完整实验和ATT见第8节。

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

- **当前采纳**：保留在当前222T源码路径。
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

## 6. 最新222T旋转反相流水（当前唯一实现）

### 6.1 数据流与阶段边界

当前源码复用原有`kv_step()`的7项loop state，不跨回边携带`frag_V`。每个静态偶/奇substep：

1. **stage1续段**：读取`V(i)`，8条global load与`GEMM1(i)`的32条MFMA交织；第一个静态substep使用统一`3×MFMA/load`，第二个substep使用`3,3,3,4,4,3,2,2`消除第5/6条load附近的VMEM队列背压；
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
| 旋转到原`kv_step`，inline vector FMA | 244V | 219T级 | 历史采纳；建立当前数据流和反相边界 |
| 第二substep相位感知V-load配额 | 244V | 221.3--222.1T | **当前采纳；纯机器调度重排** |
| runtime循环展开2→4 | 精度不变 | 原K写位置209.4T；K写前移后236.7--236.8T，均慢于各自control | 失败；两种K写位置均稳定回退 |

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

stage边界旋转的历史24组`C-X-X-C`复测：

```text
valid = 22/24
control interleaved = 4554.8 us / 188.6T
current rotated     = 3919.6 us / 219.2T
time_ratio = 0.86095
speedup = 16.15%
candidate rel_l2 = 0.00319
```

第二substep相位感知V-load配额的3轮`C-X-X-C`复测：

```text
control median = 3904.4 us / 220.0T级
candidate median = 3874.5 us / 221.85T
time_ratio = 0.992342
speedup = 0.772%
逐轮 speedup = 0.738%, 0.764%, 0.816%
candidate rel_l2 = 0.00319
```

候选ISA仍为244V、零scratch、128 MFMA、6 setprio和17条NOP；全部opcode计数与基线一致。ATT采集时为3877.3us / 221.6T。

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

### 6.5 相位感知V-load配额ATT结果

基线Trace：`/tmp/attn-current-stage-antiphase/ui_output_agent_12897_dispatch_84`

候选Trace：`/tmp/attn-vload-stagger-att/ui_output_agent_41880_dispatch_84`

两份Trace均为H=1、M=N=40960、48条完整wave。按物理SIMD合并两个resident wave的issue区间，避免对同时发生的per-wave stall重复计数：

| raw物理指标（cycles/tile） | 基线 | 候选 | 变化 |
|---|---:|---:|---:|
| trace task cycles | 1418.922 | 1386.716 | -32.207 (-2.27%) |
| physical no-issue | 558.554 | 529.365 | -29.188 (-5.23%) |
| MFMA shadow内no-issue | 342.070 | 329.891 | -12.179 |
| shadow外no-issue | 216.484 | 199.474 | -17.010 |
| `VMEM-load + VMEM-wait` | 66.157 | 35.526 | **-30.632 (-46.3%)** |
| `MFMA + MFMA` | 35.589 | 26.524 | -9.064 |
| 单独`MFMA` blocker | 17.043 | 5.874 | -11.169 |

候选将第二substep的实际V-load间MFMA间隔从`3,3,3,3,3,3,3`改为`3,3,3,4,4,3,2`；第一substep保持全3不变。代价是`LDS/SMEM-wait + MFMA`增加约3.98 cycles/tile、`LDS/SMEM-wait + LDS/crosslane`增加约4.31 cycles/tile，但净物理空洞和墙钟都下降，因此采纳。

## 7. 最终性能快照

“当前保留”只指当前源码中的222T旋转流水；其余行是已经迁移进本文的历史终点，不再保留对应代码分支。

| 路径 | shape与口径 | 时间 | TFLOPS | 资源 | 精度 | 定位 |
|---|---|---:|---:|---|---:|---|
| MMA32 FP8 BM128 | softmax，40960，D192，GPU0共享VRAM | 3969.1us | 324.6T | 158V级，零scratch | `0.00003`² | 实验支持配置 |
| **MMA32 FP8 BM128** | softmax，40960，D128，GPU0 | **3932.5--3934.0us** | **218.4T** | 资源待重采，8KB LDS，零scratch | `0.00005`² | **当前D128精确scheduler** |
| MMA32 BF16 BM128 | softmax，40960，D192，GPU0共享VRAM | 7373.3us | 174.8T | 212V级，零scratch | `0.00315`² | 实验支持配置 |
| **MMA32 BF16 BM128** | softmax，40960，D128，GPU0 | **3714.5--3723.9us** | **230.7--231.3T** | 104V+128A，112 SGPR，16KB LDS，零scratch | `0.00315`² | **当前HW-slot+精确scheduler** |
| **FP8 BM128 / 每wave 32行** | softmax，40960，D192 | **4115.5us** | **313.1T** | 220V，12KB LDS，零scratch | `0.00030`¹ | **当前支持配置** |
| FP8 BM64 / 每wave 16行 | softmax，40960，D192 | 6168.0us | 208.9T | 134V，12KB LDS，零scratch | `0.00030`¹ | 当前支持配置 |
| **FP8 BM128 / 每wave 32行** | softmax，40960，D128 | **3592.1us** | **239.1T** | 166V，8KB LDS，零scratch | `0.00027`¹ | **当前支持配置** |
| FP8 BM64 / 每wave 16行 | softmax，40960，D128 | 4171.7us | 205.9T | 100V，8KB LDS，零scratch | `0.00027`¹ | 当前支持配置 |
| **BM64 / 每wave 16行** | softmax，40960，D192，独立复测 | **11559.7us** | **112.1T** | 218V，24KB LDS，零scratch | `0.00315` | **当前支持配置** |
| BM128 / 每wave 32行 | softmax，40960，D192，对照 | 12099.6us | 106.5T | 256V+73A，24KB LDS，零scratch | `0.00315` | 当前支持配置 |
| **当前Fly旋转反相（K LDS写前移）** | softmax，40960，D128，最后独立复测 | **3543.5us** | **242.4T** | ISA资源待重采 | `0.00319` | **当前源码，2次展开** |
| K LDS写前移前的Fly旋转反相 | softmax，40960，3轮严格A/B | 3874.5us | 221.85T | 244V，2 waves，零scratch | `0.00319` | 历史前序版本 |
| K LDS写前移前的ATT采集 | softmax，40960 | 3877.3us | 221.6T | 同上 | `0.00319` | 48-wave ATT复验 |
| Fly无softmax v13c | 40960 | 约3230us | 265.2--266.0T | 204V | `≈0.00021` | 历史无softmax终态 |
| 高层Fly旧默认 | softmax，40960 | 约4415--4425us | 194.4--194.7T | 240V | `≈0.00319` | 历史基线 |
| full-late + periodic setprio | softmax，40960 | fast约4328us | 约198.5T | 248V | `≈0.00319` | 历史实验终态 |
| strict Fly后ISA | softmax，40960 | 约4185us | 205.2T | 2 waves，零spill | `≈0.00319` | 历史严格变换 |
| PyHIP JIT production | softmax，40960 | 约4115us | 208.6--208.8T | 156V+64A | `≈0.00319` | 历史生产JIT |
| PyHIP JIT setprio_best | softmax，40960 | 3623.5us | 237.1T | 156V+64A | `0.00318646` | 历史全项目最高性能 |
| JIT ISA oracle / body inline | softmax，40960 | 3630--3633us | 236.4--236.6T | 220V | `≈0.00319` | 仅验证机器目标 |

注意：早期“189.1T→209.1T”结果是control覆盖候选输出造成的假阳性，候选实际精度错误；**209.1T绝对不能引用或采纳**。

¹ FP8精度对比等价kernel语义：Q/K/V为E4M3FNUZ，online softmax每32-token tile将未归一化
probability乘240后量化到FP8，GEMM2使用原生FP8 MFMA，输出为BF16。小shape相对标准未量化
`softmax(QK)@V`的算法级`rel_l2=0.02204`，相对等价online-FP8 reference为`5.39e-5`。

² MMA32的40960性能使用`CHECK=0`，精度数字来自相同代码的小shape独立验证。GPU 0在测试前后
`gfx=0%`，但外部进程曾常驻约94% VRAM；不可与独占空闲卡结果直接混用。初始性能日志为
`/tmp/attn-mma32-gpu0-perf.log`；BF16 D128原始stage基线ATT为
[`../ui_output_agent_27126_dispatch_82`](../ui_output_agent_27126_dispatch_82)，最终硬件wave-slot
与精确scheduler ATT为[`../ui_output_agent_61472_dispatch_82`](../ui_output_agent_61472_dispatch_82)。

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

### 2026-07-31：第二substep相位感知V-load配额

- 假设：物理ledger显示第二静态substep的`vmcnt(9)`与第5/6条V load形成66.157 cycles/tile的`VMEM-load + VMEM-wait`共同空洞；只调整该substep可避免破坏第一substep已经稳定的相位。
- 改动：第一substep保持`3,3,3,3,3,3,3,3`；第二substep改为`3,3,3,4,4,3,2,2`，两者总MFMA配额均为24。
- 正确性：256为`0.00315`，40960为`0.00319`。
- ISA：244V、零scratch、128 MFMA、6 setprio、17 NOP；opcode计数与基线完全相同。
- 严格A/B：3轮`C-X-X-C`的control/candidate中位数为3904.4/3874.5us，逐轮加速0.738%/0.764%/0.816%，总体+0.772%。
- ATT：`VMEM-load + VMEM-wait` 66.157→35.526 cycles/tile（-46.3%）；总物理no-issue 558.554→529.365（-5.23%）；trace task cycles 1418.922→1386.716（-2.27%）。
- 决策：**采纳**。
- 产物：`/tmp/attn-vstagger-paired.log`、`/tmp/attn-vload-stagger-isa`、`/tmp/attn-vload-stagger-att/ui_output_agent_41880_dispatch_84`、`/tmp/attn-vload-stagger-physical-ledger.json`。

### 2026-07-31：K LDS读写交织反证

- 全部8条K读从`DSRD1+MFMA1`改成`DSRD1+MFMA2`：240V、精度不变，但220T→207.5T。
- 只将每个substep最热的最后一条`ds_read_b128`后移跨过1条MFMA：资源/指令数不变，但220.0T→212.0T。
- 把barrier固定到完整GEMM2后，使K写后有31条MFMA：232V、NOP减少2条，但220T→186.7T。
- 在第12条GEMM2 MFMA设置局部栅栏，使K写后有13条MFMA：228V，但220T→196.6T。
- 在第8条GEMM2 MFMA设置局部栅栏，使K写后有9条MFMA且barrier位置接近基线：244V，但220T→206.5T。
- 在第二条DS写后强制4条已有VALU：机器码出现4条`v_perm_b32`和1条MFMA，VGPR 244→232、wait 55→44，但219.8T→205.1T。
- 结论：高per-wave `ds_read_b128`/`lgkmcnt` stall是当前双wave相位的一部分，不是可独立搬动的局部根因。任何DS读写重排必须以物理SIMD共同空洞和严格墙钟为准；当前不再改变K读节奏或K写/barrier位置。

### 2026-07-31：runtime KV循环展开4次

- 假设：把每次runtime循环的静态`kv_step`从2个增加到4个，可减少一半SCF回边和loop-carried state开销，同时保持偶/奇K LDS ping-pong不变。
- 改动：循环上界从`N/BN/2`改为`N/BN/4`，`kv0`步长从2改为4，并追加第二组偶/奇substep；fragment和7项loop state不变。
- 正确性：256为`rel_l2=0.00315`；40960两次均为`rel_l2=0.00319`。
- 性能：40960候选连续两次为4103.0us / 209.4T、4103.1us / 209.4T；恢复二次展开后的同卡相邻control为3864.1us / 222.3T。
- 对比：四次展开耗时增加6.18%，吞吐下降5.80%；回退幅度远超采样噪声。
- 决策：**失败并回退**。静态body加倍节省的回边不足以抵消代码布局和resident-wave长期相位恶化；当前源码保持二次展开。

同日在K LDS写前移到softmax后、O rescale前的新版本上重新验证：

- 二次展开control：3537.4us / 242.8T、3541.9us / 242.5T；两次均为`rel_l2=0.00319`。
- 四次展开候选：3627.7us / 236.8T、3630.0us / 236.7T；两次均为`rel_l2=0.00319`，256小shape为`0.00315`。
- 两组均值：control 3539.65us，候选3628.85us；四次展开耗时增加2.520%，等价吞吐下降2.458%。
- 恢复二次展开后的最后独立复测：3543.5us / 242.4T，`rel_l2=0.00319`；与前两次control一致。
- 决策仍为**失败并回退**；保留K LDS写前移，只恢复二次展开。未采集该候选ATT，不进一步归因具体微观stall。

### 2026-07-31：BM64与每wave 16行支持

- 目标：支持`BM=64`，仍使用4 waves / 256线程，每wave只处理一个16行query group；保留`BM=128`每wave两个16行group的路径。
- 实现：将query group数参数化为`BM/(4*16)`；softmax、O rescale和epilogue按该数展开。BF16 probability pack仍按BN方向的两个16行half执行。
- D192修复：原K cooperative copy在D192时隐含需要384线程，但launch只有256线程。现固定256线程覆盖完整`[BN,D]`，D128/D192时每线程分别搬2/3个128-bit向量；`D`增加环境参数，默认192。
- 小shape正确性：BM64/BM128、D128/D192四种组合均为`rel_l2=0.00315`。
- BM64正式结果：`H=1, M=N=40960, D=192`两次为11482.6us / 112.4T、11559.7us / 112.1T，`rel_l2=0.00315`。
- BM64 ISA：4 waves / 256线程，218V、96 SGPR、24KB LDS、零scratch，静态96 MFMA、6 setprio。
- 同shape BM128对照：12099.6us / 106.5T，256V+73A、24KB LDS、零scratch；BM64吞吐高5.54%。
- 用法：`HIP_VISIBLE_DEVICES=3 H=1 BM=64 MULT=640 D=192 python3 -B tests/flydsl/test_attn_gemm.py`。
- 决策：**增加为当前支持配置**。

### 2026-07-31：FP8 Q/K/V支持

- 接口：`QKV_DTYPE=bf16|fp8`，gfx942的FP8映射为`torch.float8_e4m3fnuz` / `fx.Float8E4M3FNUZ`；输出保持BF16。
- GEMM：GEMM1/GEMM2均使用原生`MFMA(16,16,32,FP8)`；BF16继续使用`MFMA(16,16,16,BF16)`。
- probability：为避免E4M3FNUZ下溢，online softmax在FP8路径使用精确rebase；未归一化probability乘240后由`v_cvt_pk_fp8_f32`量化，epilogue除去该scale。
- 布局：FP8 K LDS必须使用plain row-major，不能复用按BF16元素宽度设计的`SwizzleType(3,3,3)`；V host preshuffle补偿GEMM1 score的32-row物理置换。
- 正确性：BM64/BM128与D128/D192四种小shape相对等价online-FP8 reference均为`rel_l2=3e-5--5e-5`；BF16四种组合保持`0.00315`。
- 标准attention口径：BM64/D128小shape相对未量化`softmax(QK)@V`为`rel_l2=0.02204`，属于FP8输入和中间probability量化误差。
- 正式BM128/D128：`H=1, M=N=40960`为3592.1us / 239.1T，`rel_l2=0.00027`；166V、19 SGPR、8KB LDS、零scratch，64条原生FP8 MFMA、16条FP8 pack。
- 正式BM64/D128：4171.7us / 205.9T，`rel_l2=0.00027`；100V、19 SGPR、8KB LDS、零scratch，32条原生FP8 MFMA、8条FP8 pack。
- 正式BM128/D192：4115.5us / 313.1T，`rel_l2=0.00030`；220V、96 SGPR、12KB LDS、零scratch，96条原生FP8 MFMA、16条FP8 pack。
- 正式BM64/D192：6168.0us / 208.9T，`rel_l2=0.00030`；134V、96 SGPR、12KB LDS、零scratch，48条原生FP8 MFMA、8条FP8 pack。
- D192中BM128相对BM64吞吐高49.88%；BM128 D192相对D128耗时只增加14.6%，计算量增加50%，TFLOPS提高30.9%。
- BF16正式回归：BM128/D128为3537.1us / 242.9T，`rel_l2=0.00319`，与改前一致。
- 决策：**增加为当前支持配置**。

### 2026-07-31：32x32 MFMA与128-bit grouped K读取

- 接口：`MMA_MN=16|32`；`MMA_MN=32`只支持`BM=128`，`BM=64`以`MMA_MN=32 only supports BM=128`在构建前失败。
- Atom：BF16使用`MFMA(32,32,8)`，FP8使用`MFMA(32,32,16)`；仍固定4 waves，每wave负责32行query。
- GEMM1读取：BF16 `k_perm1=(4,2,2):(1,8,4)`，FP8为`(8,2,2):(1,16,8)`。每个128-bit K group显式按nested坐标`(ki,kg)`调用两次atom，BF16每次消费4个、FP8每次消费8个。
- 全局K布局：BF16 MMA32使用`(4,2,2,2):(D,8D,4D,16D)`补偿32-row score置换；MMA16与FP8原布局保持不变。
- GEMM2：MMA32使用同一grouped K permutation。BF16 V为128-bit；FP8 V的单atom operand是8 FP8，FlyDSL generic `make_tiled_copy_A`对128-bit retile会留下`ub.poison`，因此仍使用合法的64-bit V读取。Q/K保持128-bit。
- FP8 probability：32x32 C fragment的16个score按线性顺序打包为两个8-FP8 K16 atom；MMA32使用独立的host V row inverse permutation。
- 小shape正确性：BF16 D128/D192均为`rel_l2=0.00315`；FP8 D128/D192相对等价online-FP8 reference分别为`0.00005/0.00003`。默认MMA16 BF16/FP8回归不变。
- ISA（D128）：BF16为64条`v_mfma_f32_32x32x8_bf16`、212V、96 SGPR、16KB LDS、零scratch，buffer load全部128-bit；FP8为32条`v_mfma_f32_32x32x16_fp8_fp8`、158V、68 SGPR、8KB LDS、零scratch，Q/K为128-bit、V为64-bit。
- GPU0性能环境：测试触发与结束时`gfx=0%`，但外部进程常驻约94% VRAM；全尺寸reference因显存不足，使用`CHECK=0`。BF16 D128/D192为4429.2us / 194.0T、7373.3us / 174.8T；FP8 D128/D192为4297.9us / 199.9T、3969.1us / 324.6T。
- 对比：D128 FP8相对BF16耗时低2.96%、吞吐高3.04%；D192 FP8相对BF16耗时低46.17%、吞吐高85.70%。该比例仍受共享GPU环境影响。
- 产物：`/tmp/attn-mma32-gpu0-perf.log`。
- 决策：**增加为实验支持配置**。

### 2026-08-01：BF16 D128 MMA32移动setprio边界（已取消）

- 根因ATT：[`../ui_output_agent_27126_dispatch_82`](../ui_output_agent_27126_dispatch_82)。两个resident wave只有`40.08%`时间priority相反，`46.34%`时间同时为`prio2`；首批slot完成skew中位数为`1,384,316 cycles`。原窗口的快slot约为`p2=1324 / p0=840 cycles`，慢slot的`p2`又被放大到约1796 cycles，形成priority正反馈。
- 对照：同一套分析在历史MMA16 ATT中得到`84.55%--87.51%`物理反相，证明静态setprio次数正确不代表MMA32也能自然反相。
- 失败反证：`prio2→prio1`为`4428us`级，与原方案等价；全部`prio0`为`4347us`级，反而快约1.9%；仅将prologue初始priority改0仍为`4432us`级；反转窗口方向为`4484us`级；在GEMM2后、barrier前升权为`4159--4161us`，仍慢于barrier后升权的`4095us`级。
- 历史候选边界：BF16 D128 MMA32让`prio2`跨过V load、GEMM1、K prefetch、probability EXP和running-sum，在probability写回后降为`prio0`；K LDS写、O rescale、BF16 pack、GEMM2和workgroup barrier完成后恢复`prio2`，再执行K LDS read并跨回边进入下一次GEMM1。
- 严格性能：同机器边界的`priority=0` control为`4440.3--4441.1us / 193.4--193.5T`，最终候选为`4026.3--4032.0us / 213.1--213.4T`。移除实验开关后的最终源码复测为`4020.6--4041.0us / 212.6--213.7T`；相对最初`4429.2us / 194.0T`，中位时间下降约8.99%。
- 历史候选ATT位于`/tmp/attn-mma32-prio-scoped-final/ui_output_agent_21317_dispatch_82`。物理反相提升到`93.11%`，同时`prio2`降到`1.43%`，同时`prio0`为`5.46%`；slot完成skew中位数降到`78,682 cycles`，相对原始下降94.32%；`p2/p0`窗口中位数为`1256 / 1360 cycles`。
- ATT stall：相对原始trace，总stall下降4.42%，MFMA stall下降3.11%，LDS指令stall下降82.74%，barrier stall下降5.37%，但LDS wait上升213.48%。动态MFMA和setprio工作量不变；资源从`84V+132A`调整为`88V+128A`，combined VGPR仍为216，另有112 SGPR、16KB LDS、零scratch，保持2 waves/SIMD。
- 跨shape反证：直接将新边界推广到BF16 D192、FP8 D128、FP8 D192会分别回退到`163.8T / 188.2--188.5T / 291.3--291.4T`。最终用编译期条件将其限定为`MMA_MN=32 && BF16 && D=128`；其他MMA32配置恢复原边界后分别回到`175.0T / 199.8T / 324.4--324.5T`。
- 决策：虽然墙钟和ATT均改善，但该方法改变了stage0/stage1定义，不符合“与MMA16相同完整stage反相”的目标，**取消并恢复原stage边界**。

### 2026-08-01：`HW_ID.WAVE_ID`四级priority与MMA32精确scheduler

- 目标：保持MMA16的完整stage边界不变。stage0为K global预取/LDS写与softmax；stage1为GEMM2并跨回边覆盖下一tile的V预取/GEMM1。用`s_getreg_b32 ..., hwreg(HW_REG_HW_ID, 0, 4)`读取每个SIMD内的物理wave slot。
- 指定映射：slot0使用`stage0=1, stage1=3`，slot1使用`stage0=0, stage1=2`。最终ISA只有1条`s_getreg_b32`，结果常驻`s19`；ATT确认`sl0`只执行`1/3`、`sl1`只执行`0/2`。每个stage边界因`s_setprio`只有立即数形式，需要`s_cmp + s_cbranch + s_setprio + s_branch`选择路径。gfx942不支持gfx12+的`s_setprio_inc_wg`，无法用该指令消除热循环分支。
- 正确性/资源：小shape`rel_l2=0.00315`；40960 ATT CSV为`84V+132A`、112 SGPR、16KB LDS、零scratch，combined VGPR allocation为216，仍为2 waves/SIMD。小shape ISA dump为212 VGPR、74 SGPR，两种口径不可混用。
- 初始性能：只加指定映射时为`4519.2--4522.0us / 190.0--190.1T`，ATT运行`4523.1us / 189.9T`；比原stage control `4432.1--4433.0us / 193.8--193.9T`慢约2.0%。初始ATT位于`/tmp/attn-mma32-hwslot/ui_output_agent_25427_dispatch_82`：完整stage反相仅从原MMA32的`40.08%`升到`44.55%`，首批slot完成skew从`1,384,316`恶化到`2,034,602 cycles`。
- 相位漂移：两个slot到达同一stage边界的差值从首tile约`3--4K cycles`近似线性增长到末尾约`2.03M cycles`。slot0的stage0/stage1中位耗时约`860/1324 cycles`，slot1约`1160/2592 cycles`；固定给slot0高一级priority放大了原有速度差，不能建立负反馈。
- 派生反证：将固定高一级priority交换给slot1（slot0 `0/2`、slot1 `1/3`）更慢，为`4657.0--4662.9us / 184.3--184.5T`。令两个异相方向都保持2级差（slot0 `1/2`、slot1 `0/3`）也只有`4525.1--4529.3us / 189.7--189.8T`。
- scheduler根因：MMA32每个D128 GEMM只有BF16 16条/FP8 8条MFMA，但旧`hot_loop_scheduler`仍使用MMA16配额。BF16 GEMM2的前14条MFMA配额先耗尽，8条DSRD仅前2条与MFMA配对，末6条连续聚团；越靠后的`ds_read_b128` stall越高，单条最高约0.48M cycles。
- 第一修复：BF16 D128改为GEMM1 `VMEM1:MFMA2`，GEMM2前置4条MFMA后`(DSRD1:MFMA1)×8`。ISA中DSRD最大连续团从6降为1，DSRD stall从3.176M降为2.176M（-31.5%），总stall下降11.6%，stage反相升到65.99%；性能`4005.1--4005.8us / 214.5T`。
- VMEM比例：同时修改GEMM1/GEMM2配额的早期`1:1`实验为213.9--214.0T，存在混淆；隔离后仅将GEMM1改为`VMEM1:MFMA1`，GEMM2保持DSRD 1:1，性能提升到`3847.1--3852.1us / 223.0--223.3T`。stage反相达到85.99%--86.92%，与MMA16的87.51%基本一致；slot skew降到10--30K cycles。
- DSRD位置：DSRD从GEMM2调度区起点开始虽然反相最高，但DSRD/DSWR发出竞争较高。保持VMEM 1:1后，BF16在DSRD链前放3条MFMA达到最终`3714.5--3723.9us / 230.7--231.3T`。完整离散扫描为lead0≈223T、lead1≈218T、lead2≈214T、lead3≈231T、lead4≈226T、lead5≈228T、lead6≈211T、lead7≈205T、lead8≈204T；lead3是明确的长期相位峰值。
- 其他反证：`MFMA→DSRD`顺序为217.8T，必须保持`DSRD→MFMA`请求先发；第二组DSWR插入深度0/1/2/4均为3847--3852us，墙钟不可区分，采用结构最简单的连续DSWR。`VMEM:MFMA=1:2`最终为221.3--221.7T，低于1:1。
- 最终ATT：[`../ui_output_agent_61472_dispatch_82`](../ui_output_agent_61472_dispatch_82)，单SE1、单dispatch、16 waves、891条ISA、源码映射99.89%、无截断，snapshot与最终源码逐字一致。完整stage反相为`80.18%`，同时stage1/0为`18.37%/1.45%`，slot完成skew中位数仅`17,304 cycles`；stage中位时长为slot0 `1160/1352`、slot1 `888/1596 cycles`。相对初始HW-slot ATT，总stall约下降13%，MFMA stall约下降31%，DSRD stall`3.176M→2.773M`且单条最大`0.484M→0.312M`；总stall率`64.91%→60.9%`。LDS wait约0.505M，消费端`lgkmcnt`不在主关键路径。资源为104V+128A、112 SGPR、16KB LDS、零scratch，combined VGPR 232，保持2 waves/SIMD。
- 无`s_getreg`消融：保持完全相同的D128 scheduler，只恢复统一`stage0=0/stage1=2`。ISA中`s_getreg_b32`从1条降为0，VGPR/occupancy不变；小shape`rel_l2=0.00315`。40960三次为`3971.7--3978.5us / 216.0--216.3T`，相对有`s_getreg`的`3718.2--3719.7us / 231.0--231.1T`耗时增加约6.92%。ATT见[`../ui_output_agent_47082_dispatch_82`](../ui_output_agent_47082_dispatch_82)：完整stage反相`80.18%→61.63%`，slot skew`17,304→754,750 cycles`，总stall增加4.84%，MFMA stall增加19.19%，VMEM-load stall增加63.79%；LDS stall虽下降38.14%，没有转化为关键路径收益。结论：当前scheduler仍需要HW-slot priority，最终代码已恢复`s_getreg`。
- K LDS写位置消融：仅对BF16 D128 MMA32移动`fx.copy(cp_cs, ld_cur, ...)`，精度均保持`rel_l2=0.00315`、资源仍232 combined VGPR/零scratch。移到BF16 `_cvt_f32_to_bf16(frag_St)`之后时，ISA中两条`ds_write_b128`落在rounding ADD之后、pack `v_perm`之前，性能回退到`4000.7--4001.1us / 214.7T`。移到O rescale/state更新之后、BF16 pack之前时，两个静态substep被后端排成不对称写入位置，性能为`3974.5--4020.1us / 213.8--216.1T`。恢复原位置（probability完成后、O rescale之前）立即回到`3720.3--3725.3us / 230.6--230.9T`。结论：后移K写破坏lead3 scheduler的长期相位，原位置保留。
- K LDS写拆分消融：BF16 D128的per-thread copy fragment为`((8,2),1,1)`，内层`2`恰对应两条128-bit atom。用nested slice拆成两个`(8,1,1)`子片后，ISA仍是每substep两条`ds_write_b128`，没有降成窄写；combined VGPR从232增到236，仍为2 waves、零scratch。`C-X-X-C`中control两端为`3715.9/3716.2us`（231.2T），拆分模式均明显回退：half0原位、half1在O-rescale后为`4106.1--4109.4us / 209.1--209.2T`；half1移到BF16 conversion后或pack后分别为`4088.0/4086.2us`（210.2T）；交换half身份为`4090.7us / 210.1T`。因此回退与half身份和后移距离无关，根因是把一次完整copy拆成两个独立调度边界并破坏长期相位。拆分候选ATT为[`../ui_output_agent_47059_dispatch_82`](../ui_output_agent_47059_dispatch_82)：单SE1、单dispatch、16 waves、891条ISA、源码映射99.89%、无截断，正式资源108V+132A、112 SGPR、16KB LDS、零scratch，ATT运行`4109.7us / 209.0T`。代码已恢复整片原位写，复测`3715.9--3725.3us / 230.6--231.2T`。
- softmax EXP/DSWR交织：基线源码虽然在probability EXP后写K，后端会把整片两条`ds_write_b128`提前到16条probability `v_exp_f32`之前。使用callback并在两侧加入`sched_barrier(0)`可强制真实机器序列`EXP×8 → DSWR×2 → EXP×8`，不拆两个128-bit atom；但combined VGPR从232增到236，`C-X-X-C`从control `3718.8/3722.7us`回退到`4009.7--4012.1us / 214.1--214.2T`。去掉硬barrier后，后端把DSWR重新移出EXP块。源码切点4/8/12中，4和12生成逐字相同ISA，说明普通`sched_barrier`不能按TRANS数量建立流水。空SSA anchor、side-effect inline `v_exp_f32`和`~{memory}`也无法阻止EXP跨越LDS写；8标量struct anchor还触发LLVM限制，均已撤销。
- `sched_group_barrier`机制验证：LLVM IGroupLP原生支持`VALU=0x002`、`DS_WRITE=0x200`和`TRANS=0x400`。独立gfx942 MIR验证从“8 VALU + 2个连续DSWR”精确重排为`VALU2 → DSWR1 → VALU2 → DSWR1 → VALU4`；attention中同一次完整`fx.copy`产生的两条`ds_write_b128`也能用两个`DS_WRITE(size=1)` group分别定位，无需拆copy。mask、独立`syncid=1`及`_sched_valu/_sched_trans/_sched_ds_write` helper均局部定义在`test_attn_gemm.py`，不修改FlyDSL目录。
- 对称`EXP8/DSWR2/EXP8`反证：使用独立`syncid=1`发出`TRANS(size=9) → DS_WRITE(size=2) → TRANS(size=8)`可精确生成“1条running-max correction EXP + 8条probability EXP → 2条DSWR → 8条probability EXP”；首组为9是因为同一调度区更早还有1条correction EXP。资源仍232 combined VGPR/零scratch，但性能从`3715.6/3716.1us / 231.3T`回退到`4124.1--4145.8us / 207.2--208.5T`。这只否定该位置，不否定group机制。
- resident-wave LDS冲突分析：ATT文件需按同一SIMD的`slot0/slot1`配对，`wv0/wv1`是前后两代wave，不能互配。基线K写几乎没有`DS_WRITE↔DS_WRITE`冲突（32 cycles内为0），真正冲突是`DS_WRITE↔DS_READ`：40960次K写中22583次（55.1%）距对方slot的DS read不超过32 cycles，最近距离中位数28 cycles；近冲突写stall均值64.5 cycles，远离时43.8 cycles。slot0/slot1写stall均值严重不对称，为102.2/8.2 cycles。
- 分散两条写的位置扫描：保持一次完整copy，分别用`DS_WRITE(size=1)`把两条写插入16条probability EXP。粗扫`(0,4)/(0,8)/(0,12)/(2,6)/(2,10)/(2,14)/(4,8)/(4,12)/(6,10)/(6,14)/(8,12)/(10,14)`和局部细扫确认`(2,6)`为稳定峰值；邻域`(2,5)=248.2T`、`(2,7)=245.1T`、`(1,5)=245.4T`，而`(2,6)=249.1--249.3T`。最终机器序列为“1 correction EXP + 2 probability EXP → DSWR1 → 4 EXP → DSWR1 → 10 EXP”，资源仍104V+128A、112 SGPR、16KB LDS、零scratch、2 waves/SIMD，`rel_l2=0.00315`。
- 最终`(2,6)` ATT：[`../ui_output_agent_22766_dispatch_82`](../ui_output_agent_22766_dispatch_82)，单SE1、单dispatch、16 waves、891条ISA、源码映射99.89%、无截断，snapshot与最终源码逐字一致，ATT运行`3451.1us / 248.9T`。相对[`../ui_output_agent_61472_dispatch_82`](../ui_output_agent_61472_dispatch_82)，总stall`29.73M→25.67M`（-13.7%）、MFMA stall`14.59M→12.62M`（-13.5%）、barrier stall`3.28M→0.92M`（-72.0%）、K写stall`2.26M→1.82M`（-19.7%）。跨slot 32-cycle `W↔R`近冲突`22583→5057`（55.1%→12.3%），最近距离中位数`28→276 cycles`，两条写间距中位数`68→220 cycles`，slot0/slot1写stall均值从`102.2/8.2→46.5/42.2 cycles`。
- 额外物理slot错位消融：在`(2,6)`基础上将已有stage0 priority分支融合4/8/16-cycle `s_nop`，分别延迟slot0或slot1。无延迟为`249.3T`；slot0延迟4/8/16为`247.0/235.6/241.2T`，slot1延迟4/8/16为`249.1/238.0/245.4T`。因此单wave`(2,6)`已经通过现有反相调度自然错开resident waves；额外slot延迟无收益，最终不保留NOP。
- shape/dtype分派：精确scheduler仅对`MMA_MN=32 && D=128`启用；BF16/FP8的DSRD前置MFMA分别为3/0条，EXP/DSWR `(2,6)`交织进一步仅对BF16 D128启用。BF16/FP8 D128分别达到`249.1--249.3T / 218.4T`；D192保持旧scheduler，BF16/FP8分别为`174.8T / 324.4T`。
- 决策：**采纳D128硬件wave-slot priority、精确`VMEM 1:1 / DSRD 1:1` scheduler，以及BF16 D128的完整copy `EXP2/DSWR1/EXP4/DSWR1/EXP10`交织；D192保持原scheduler。**

## 9. 后续方向与硬约束

当前已处理两个静态substep的VMEM不对称。后续优先分析候选ATT中剩余的物理共同空洞：

1. `MFMA + TRANS`约127.3 cycles/tile；
2. `LDS/SMEM-wait + MFMA`约83.5 cycles/tile；
3. `MFMA + scheduler/ready`约57.6 cycles/tile；
4. 剩余`VMEM-load + VMEM-wait`约35.5 cycles/tile。

硬约束：

- 保持两个`vector<8xf32>` running-sum，不做scalar提前归约。
- 保持7项loop state，不跨回边携带`frag_V`。
- 当前完整静态MFMA门槛是128，scratch必须为0，VGPR必须不超过256。
- 不插NOP补跨slot延迟，不机械提前K read，不只压缩scale/EXP链。
- 不再全局增加DSRD/MFMA间隔，不移动K写/barrier位置，不在DS写后强制VALU分组；这些路线已严格回退。
- MMA32 D128保持`VMEM:MFMA=1:1`与`DSRD:MFMA=1:1`；BF16/FP8分别前置3/0条MFMA。BF16相位对前置MFMA数量高度敏感。
- MMA32 BF16 D128源码仍保持一次完整K LDS copy，位置在probability完成后、O rescale之前；机器级仅用schedule groups采用已验证的`EXP2/DSWR1/EXP4/DSWR1/EXP10`，不要后移源码copy到BF16 pack附近。
- 不拆分MMA32 BF16 D128的两个128-bit K LDS写atom；分片会稳定回退约9--10%。
- EXP/DSWR位置高度敏感：不要复用失败的对称`EXP8/DSWR2/EXP8`或中后段写位置；BF16 D128只保留已验证的`EXP2/DSWR1/EXP4/DSWR1/EXP10`。
- MMA32 D192保持旧scheduler，不得机械复用D128配额。
- 每次顺序：候选自身精度 → ISA资源 → 40960严格A/B → 必要时ATT/PMC。