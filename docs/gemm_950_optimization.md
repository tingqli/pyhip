# GFX950 FlyDSL GEMM 优化记录

本文记录 `tests/flydsl/test_gemm.py` 中 `gemm_4wave_950` 和
`gemm_8wave_950` 的实现、指令调度、汇编检查与性能结果。

## 1. 测试范围

- 目标架构：AMD CDNA4 `gfx950`，测试机器为 256 CU。
- 输入：BF16 或 FP8 E4M3FN 的 A/B 矩阵。
- B 布局：AIter/aiter 的 16 行 preshuffle 布局。
- 输出：BF16。
- FP8 暂不接收外部 scale；scaled MFMA 的 A/B scale 都设置为 E8M0 单位值
  `0x7f7f7f7f`。
- 性能表使用 `M=N=K=4096`、输出 tile `256 x 256`。
- 最新性能数据采集于 2026-07-25。BF16最终对比预热30轮，4/8-wave交替执行
  12组、每组80次并取中位数；FP8最终数据采用同进程交替执行，降低温度和频率漂移
  带来的顺序偏差。

## 2. GFX950 指令选择

GFX950 使用 CDNA4 原生指令宽度：

| 输入 | FlyDSL atom | 最终 ISA |
|---|---|---|
| BF16 | `MFMA(16, 16, 32, BFloat16)` | `v_mfma_f32_16x16x32_bf16` |
| FP8 | `cdna4.MFMA_Scale(16, 16, 128, Float8E4M3FN)` | `v_mfma_scale_f32_16x16x128_f8f6f4` |

BF16 的 K permutation 为 `(8, 4):(1, 8)`；FP8 为
`(32, 4):(1, 32)`，都与单条硬件指令消费的操作数宽度一致。

检查最终 ISA：

```bash
FLYDSL_DUMP_IR=1 FLYDSL_DUMP_DIR=/tmp/gemm950_ir \
  pytest -q tests/flydsl/test_gemm.py::test_gemm_950_correctness -s
grep -R -E "v_mfma(_scale)?_f32_16x16x(32|128)" /tmp/gemm950_ir
```

## 3. 两个 K tile 展开的流水

每个 kernel 都为四个矩阵象限分配两级 LDS：

- `AT`：A 的上半块；
- `AB`：A 的下半块；
- `BL`：B 的左半块；
- `BR`：B 的右半块。

GMem 通过 `BufferCopyLDS128b` 直接写 LDS。Prologue 先装入 K tile 0 和 1，
再把 `AT0`、`BL0` 从 LDS 预取到寄存器。

Prologue 的 DMA 发射顺序与 `bf16_3stage_4wave` 完全一致：每个 stage 都是
`BL -> AT -> AB -> BR`，两个 stage 合计八组。每个逻辑 async group 实际生成 4 条
`buffer_load_dwordx4 ... lds`，所以 `wait_group(4)` 在 ISA 中对应 `vmcnt(16)`，不是
`vmcnt(4)`。

运行时循环每次前进两个 K tile。stage 选择和边界在编译期固定，因此生成代码没有
`k % 2` 或“是否存在 next tile”的动态分支。当前要求 K tile 数量至少为 2 且是偶数。

一个循环 body 固定为八个区域：

| 区域 | 计算 | 为下一次计算读取 LDS | GMem 直接预取到 LDS |
|---|---|---|---|
| 0 | `TL(stage 0)` | `AB0` | `BL(k+2)` |
| 1 | `BL(stage 0)` | `BR0` | `AT(k+2)` |
| 2 | `TR(stage 0)` | `BL1` | `AB(k+2)` |
| 3 | `BR(stage 0)` | `AT1` | `BR(k+2)` |
| 4 | `TL(stage 1)` | `AB1` | `BL(k+3)` |
| 5 | `BL(stage 1)` | `BR1` | `AT(k+3)` |
| 6 | `TR(stage 1)` | `BL0(k+2)` | `AB(k+3)` |
| 7 | `BR(stage 1)` | `AT0(k+2)` | `BR(k+3)` |

最后固定处理两个 K tile 的 tail，不再发出未来 tile 的预取。区域顺序参考 Gluon
`bf16_3stage_4wave`，同时保留 FlyDSL 的 LDS ping-pong 表达。

区域 3 和 7 末尾与参考实现一样使用 `wait_group(4)`，最终 ISA 都是
`s_waitcnt vmcnt(16) lgkmcnt(0)` 加 `s_barrier`。Tail 的六次逻辑等待为
`wait_group(5,4,3,2,1,0)`，对应物理 VMEM 计数 `20,16,12,8,4,0`。

为了让 LLVM 正确区分在途 DMA 写和独立 LDS read，BF16 路径给 DMA 写添加统一
`alias.scope`，给 LDS read 添加本地 `alias.scope` 和对应 `noalias`。没有这对 metadata
时，`SIInsertWaitcnts` 会保守插入额外 `vmcnt(4)`，并导致部分子块无法按 scheduler
模板交织。

另外复用了 Gluon 的两个 B-fragment live-range anchor，并在区域 3 后延长 `AB/BR`
fragment live range，避免 LDS read 目标 VGPR 与仍被 MFMA 消费的输入发生 WAR 复用。

### 3.1 LDS padding 与连续 B 布局

4-wave BF16 的 A padding 与 `tests/flydsl/test_async_copy.py::asycn_copy_padding`
使用完全相同的两个 view。对于 `128 x 64` 半 tile：

```text
write: ((8,16),64):((64,528),1)
read:  ((16,8),(32,2)):((528,64),(1,32))
```

即每 8 行（512 个 BF16）插入 16 个 BF16。写入前还按
`physical_row = (logical_row % 8) * 16 + logical_row // 8` 重排行 mode；只给写、读
使用同一个 view 会保留一半 bank conflict。

B 不做 padding 或 XOR swizzle。GMem-to-LDS 直接按 host preshuffle 顺序线性复制：
第 `tid` 个线程每轮负责从 `tid * 16` 字节开始的一个 16-byte vector，下一轮前进
`256 * 16` 字节。LDS read 使用同一 preshuffle物理地址规则，因此同一 wave 的连续
线程自然访问连续 bank。

4-wave FP8沿用相同的双view和连续B策略，但A的 `kWidth=32` 需要采用a8w8教程的
双padding规则 `[[1024, 16], [2048, 32]]`。每1024个FP8元素插入16个元素，并在每
2048个元素处额外插入32个元素。A的DMA写view和MFMA读view分别表达物理写入顺序与
operand读取顺序；B保持host preshuffle字节序，DMA直接复制，LDS读取按FP8 tiled-copy
对应的lane坐标重组两个连续K16向量。

`M=N=K=4096`、`256 x 256 x 64`、256 workgroups 的 PMC 结果：

| 配置 | `SQ_LDS_BANK_CONFLICT` | 每 workgroup | LDS |
|---|---:|---:|---:|
| 无 padding/连续 B | 25,165,824 | 98,304 | 131,072 B |
| A/B 共用简单 padding view | 8,388,608 | 32,768 | 135,168 B |
| 仅修正 A 的双 view | 4,194,304 | 16,384 | 135,168 B |
| A 双 view + B 连续布局 | 0 | 0 | 133,120 B |

参考 `asycn_copy_padding` 在相同访问负载下从 `3,670,016` 降到 `0`，证明这套 A
参数本身能完全移除 bank conflict。

FP8使用 `4096^3`、`256 x 256 x 128`、256 workgroups采集PMC。只加入A双padding时，
`SQ_LDS_BANK_CONFLICT`从无padding版本的`12,582,912`降到`6,291,456`；剩余冲突来自
B。加入连续B写入/读取后，三次dispatch的结果均为：

| Counter | A双padding | A双padding + B连续布局 |
|---|---:|---:|
| `SQ_LDS_BANK_CONFLICT` | 6,291,456 | **0** |
| `SQ_LDS_DATA_FIFO_FULL` | 0 | **0** |

因此a8w8双padding和连续B布局共同消除了4-wave FP8热循环的全部LDS bank conflict。

8-wave BF16也需要A/B两侧共同变换。A沿用BF16的512元素间隔padding，但针对
`(4,2)` wave布局使用转置read view，并同步重排GMem源行：

```text
write: ((8,16),64):((64,528),1)
read:  ((16,8),(32,2)):((528,64),(1,32))
source row: (row % 8) * 16 + row // 8
```

每个线程负责的第二个A chunk在重排后只前进8个源行。B仍保持host preshuffle顺序，
但每512个BF16元素插入16个元素；raw DMA目标地址和手工LDS reader应用同一个物理
offset变换。每个B stage因此从8192增长到8448个BF16，kernel总LDS为135,168 B。

最终`4096^3`、256 workgroups、每个workgroup 8 waves的新PMC采集结果为：

| Counter | FlyDSL 8-wave BF16 | Gluon 8-wave BF16 |
|---|---:|---:|
| `SQ_LDS_BANK_CONFLICT` | **0** | 393,216 |
| `SQ_LDS_DATA_FIFO_FULL` | **0** | 162,787 |
| `SQ_WAVE_CYCLES` | 82,904,344 | 76,449,929 |
| `SQ_BUSY_CYCLES` | 5,239,993 | 4,800,956 |

FlyDSL四次采样的bank conflict和FIFO full均为0；cycle列取多次dispatch中位数。

### 3.2 XCD-aware PID 映射

GFX950 有 8 个 XCD，每个 XCD 有独立 L2。`compile_gemm_950` 默认使用从 Gluon
`get_pids` 移植的映射：先按 `pid % 8` 把连续硬件 block分配到 8 个 XCD，再以
`GROUP_SIZE_M=4` 对 M tile分组。访问相同 B tile的四个 M block因此更可能落在同一
XCD并复用 L2。`pid_swizzle=False` 可恢复原 row-major映射用于 A/B测试。

`M=N=K=4096`、4-wave、`256 x 256 x 64` 的 PMC：

| 映射 | L2 hit rate | `TCC_MISS_sum` | DRAM read requests |
|---|---:|---:|---:|
| FlyDSL row-major | 72.32% | 2,622,656 | 2,361,576 |
| FlyDSL `get_pids` | 80.63% | 1,836,212 | 1,575,164 |
| Gluon `get_pids` | 79.33% | 1,872,159 | 1,602,284 |

总 `TCC_READ_sum` 基本不变。FlyDSL `get_pids` 相对 row-major减少 30.0% L2 miss和
33.3% DRAM read request，证明映射生效。不过该尺寸已经偏 compute-bound，交替 12 轮、
每轮 100 次的中位延迟只从 `0.100815 ms` 降到 `0.100640 ms`，提升 0.17%。

## 4. `hot_loop_scheduler` 的目标排布

每个区域由 `rocdl.sched_barrier(0)` 限定 scheduling region。对于 4-wave BF16
`256 x 256 x 64`，每区有 32 条 MFMA、8 条 DSRD、4 条 VMEM。代码使用 Gluon
中的具体组数：

1. 8 次 `sched_mfma(1)` + `sched_dsrd(1)`；
2. 4 次 `sched_mfma(4)` + `sched_vmem(1)`；
3. 最后 `sched_mfma(8)`。

带 LDS read 的 tail 使用 `8 x (MFMA1 + DSRD1) + MFMA24`；纯计算 tail 使用
`MFMA32`。其他 tile/dtype 的指令数量不同，继续使用按实际数量生成的 fallback 调度。

FlyDSL wrapper 对应的 mask 为：

```text
MFMA     0x008
VMEM_RD  0x020
DS_READ  0x100
DS_WRITE 0x200
```

它们在 LLVM IR 中会变为：

```llvm
call void @llvm.amdgcn.sched.group.barrier(i32 mask, i32 count, i32 group)
```

## 5. 为什么最终汇编没有严格等于 scheduler 列表

### 5.1 `sched.group.barrier` 不是汇编模板

`rocdl.sched_*` 不会在源码当前位置直接生成一条等待指令，也不保证最终汇编逐条等于
Python 调用列表。AMDGPU 的 `IGroupLPDAGMutation` 会在一个 scheduling region
（通常是同一 basic block、两个 `sched.barrier` 之间）内：

1. 按 mask 搜索可能匹配的 machine instruction；
2. 用 `PipelineSolver` 给这些候选指令分组；
3. 尝试增加 `Order/Artificial` 调度依赖；
4. 如果依赖与现有数据依赖成环，则放弃相应约束以保证正确性。

因此它本质上是 best-effort 的 machine-scheduler 约束，不是最终 ISA 的严格脚本。
常见 VALU 容易误匹配；真实 RAW/WAR/WAW 依赖、waitcnt 和寄存器压力也会限制重排。

### 5.2 tied MFMA inline asm 不能可靠匹配 MFMA mask

最初为了让 accumulator 固定在 AGPR，BF16 K32 使用 tied inline asm：

```python
llvm.inline_asm(
    T.vec(4, T.f32),
    [a, b, acc],
    "v_mfma_f32_16x16x32_bf16 $0, $1, $2, $0",
    "=a,v,v,0",
    has_side_effects=False,
)
```

LLVM IR 中它是一个 inline-asm call，而不是
`llvm.amdgcn.mfma.f32.16x16x32.bf16` intrinsic。进入 Machine IR 后，它按
`INLINEASM` 处理；`sched_mfma` 的 `0x008` mask 面向 AMDGPU 原生 MFMA opcode，
无法可靠地把这些 inline asm 当作“精确的 MFMA 组”计数。

使用匹配版本的 `llc -stop-after=machine-scheduler` 检查 MIR 后，可以直接看到：

```text
INLINEASM "v_mfma_f32_16x16x32_bf16 ..."
...
SCHED_GROUP_BARRIER 8,   1, group
SCHED_GROUP_BARRIER 256, 1, group
```

即调度阶段看到的是 `INLINEASM`，不是 `V_MFMA_F32_16X16X32_BF16`。因此
`SCHED_GROUP_BARRIER 8,...` 找不到完整的原生 MFMA 候选集合。

这解释了 tied-inline 对照路径的现象：

- LLVM IR 中 8/4/8 的 `sched.group.barrier` 数量和顺序完全正确；
- 最终 ISA 确实发生了交织，但不严格等于目标列表；
- 实际交织主要来自普通 machine scheduler 对 pure inline asm 数据依赖的重排，而不是
  group solver 对 MFMA mask 的完整匹配。

### 5.3 group id 不是主因

Gluon 八个区域分别使用 group id `0..7`。FlyDSL 原先的便捷函数
`sched_mfma/sched_dsrd/sched_vmem` 把 group 固定为 0。为排除此差异，当前 4-wave
固定调度改为直接调用 `sched_group_barrier(mask, count, region_id)`，八个区域传入
`0..7`。

实验结果：

- LLVM IR 中第三参数正确变为 0、1、2……7；
- 规范化（去掉 debug `.loc`）后的最终 ISA 与 group 全为 0 的版本完全一致；
- 指令类别连续段数量仍为 136；
- 60 次基准为 `0.1417 ms / 970.3 TFLOPS`，属于正常波动。

原因是每个区域已经由 `sched_barrier(0)` 分隔，group id 不负责把 inline asm 重新分类
为 MFMA。当前偏离的主因仍是 MFMA opcode 分类和真实数据依赖。

### 5.4 `has_side_effects` 对交织的影响

在 tied-inline 对照路径中，MFMA 的全部架构状态都已由 A/B 输入、tied accumulator
输入和输出建模，没有隐藏的
内存或控制副作用，因此可以安全使用 `has_side_effects=False`。结果仍被后续计算消费，
不会被 DCE。

使用 `True` 时 LLVM IR 为 `asm sideeffect`，32 条 MFMA 会基本成团，随后才集中出现
DSRD/VMEM。改为 `False` 后：

| 指标 | `True` | `False` |
|---|---:|---:|
| 热循环 MFMA | 256 | 256 |
| 热循环 DSRD | 64 | 64 |
| 热循环 VMEM | 32 | 32 |
| 指令类别连续段数量 | 40 | 136 |
| 热循环 `v_accvgpr_read_b32` | 0 | 0 |
| 热循环 `v_accvgpr_write_b32` | 0 | 0 |

`False` 恢复了明显交织，但仍不能让 group solver 严格识别 inline-asm MFMA。不要把
`has_side_effects=False` 用于 barrier、wait、atomic、写内存 asm，或结果未使用但必须保留
的 asm。

### 5.5 如果必须严格控制顺序

可选方案按侵入程度排序：

1. **当前采用的本地 hybrid 方案**：只用空 `=a,0` asm 约束 accumulator live range，
  计算仍使用原生 ROCDL MFMA。它同时保留 MFMA 调度分类和零热循环 AGPR 搬运，但
  `sched_group` 仍是 best-effort，不保证逐位模板。
2. **整段区域写成一个 inline asm**：把 32 MFMA、8 DSRD、4 VMEM 全部写入一个 asm
   模板，顺序最严格，但地址、waitcnt、寄存器约束和可维护性成本很高，编译器也无法再优化。
3. **扩展 FlyDSL/LLVM**：新增能同时表达“原生 MFMA opcode”和“accumulator tied AGPR”
   的 lowering，使 Machine IR 保留真正的 `V_MFMA_*` opcode，随后
   `sched.group.barrier(mask_mfma, ...)` 才有机会严格匹配。这是结构上最正确、但需要修改
   FlyDSL/LLVM 后端的方案。

### 5.6 独立验证脚本

仓库提供了 native、tied inline asm 和 native-anchor 的自动对照脚本：

```bash
python3 tests/flydsl/verify_inline_asm_scheduling.py
```

脚本从同一个 4-wave BF16 kernel 生成三版代码，随后自动检查 LLVM IR、
machine-scheduler 后 MIR 和最终 ISA：

- native 版：LLVM intrinsic 和 MIR 原生 `V_MFMA_*` opcode；
- inline 版：LLVM inline asm 和 MIR `INLINEASM` opcode；
- native-anchor 版：空 `=a,0` asm 加原生 MFMA；
- 三版都包含相同的 `SCHED_GROUP_BARRIER`；
- 比较热循环实际 `M/D/V` 类别序列与目标交织序列；
- 统计 `v_accvgpr_read/write`。

当前输出的关键部分：

```text
[native]        MIR native_opcode=512, inlineasm=0,   agpr_anchor=0
[inline]        MIR native_opcode=0,   inlineasm=512, agpr_anchor=0
[native-anchor] MIR native_opcode=512, inlineasm=0,   agpr_anchor=256

PASS: inline asm 在 MIR 中仍是 INLINEASM，MFMA mask 无法精确匹配它。
PASS: native-anchor 保留原生 MFMA 分类并消除了热循环 AGPR 搬运。
INFO: native-anchor 恢复了 MFMA mask 匹配，但 sched_group 仍是 best-effort。
```

native 版和 native-anchor 版仍可能因真实数据依赖而偏离理想序列，这体现
`sched.group.barrier` 的 best-effort 语义；inline 版的额外问题是 opcode 分类阶段已经
无法作为 MFMA 候选参与精确分组。所有 dump 默认保存在
`/tmp/flydsl_inline_asm_scheduling`。

## 6. AGPR accumulator 固定

4-wave launch 使用：

```python
value_attrs = {"passthrough": [["amdgpu-agpr-alloc", "256,256"]]}
launch.compile_hints["llvm_options"] = {"amdgpu-mfma-vgpr-form": False}
```

通用 `fx.gemm` 会让每个象限成为 loop-carried `vector<64xf32>` SSA 值。Layout
lowering 将其拆成 32 条独立 `f32x4` MFMA 链，但后端在循环回边给部分 SSA 值分配了
不同 AGPR 槽。解决这些 AGPR 置换环时，每个两-tile 循环会插入 96 次
`v_accvgpr_read_b32` 和 96 次 `v_accvgpr_write_b32`。

在 FlyDSL 目录中可以组合出一个不修改 FlyDSL 的解决方案：

- `kernels/attention/flash_attn_utils.py` 使用空 tied asm 锚定 SSA live range；
- `kernels/gemm/hgemm_splitk.py` 使用原生 `rocdl.mfma_f32_16x16x32_bf16`；
- `kernels/gemm/fp8_gemm_4wave.py` 证明 `=a,...,0` 能把 accumulator 固定到 AGPR。

当前实现只在每条 accumulator 链入口发出一次空 asm：

```python
acc = llvm.inline_asm(
  T.vec(4, T.f32),
  [arith._to_raw(acc)],
  "",
  "=a,0",
  has_side_effects=False,
)
for k in range_constexpr(TILE_K // 32):
  acc = rocdl.mfma_f32_16x16x32_bf16(..., acc, ...)
```

`=a` 把 live range 约束到 AGPR，`0` 把输入和输出绑定到同一寄存器。空 asm 在 LLVM IR
和 MIR 中存在以约束寄存器分配，但最终不生成 ISA；真正计算仍是原生
`V_MFMA_F32_16X16X32_BF16`，所以 `mask_mfma` 可以识别。

修复后的保存汇编中：

```text
hot loop: 256 MFMA, 0 accvgpr_read, 0 accvgpr_write
```

初始化、epilogue 和其他非热循环边界仍可包含 AGPR/VGPR 传输；关键是循环回边不再出现
原生 `fx.gemm` 路径的 96/96 次置换搬运。

## 7. 8-wave 同步

`sched_barrier`只是编译器调度fence，不是workgroup同步。8-wave kernel把8个wave
分为两组，每组4个wave交错执行：prologue中`wave_id >= 4`的组先执行一次条件
`s_barrier`，tail结束时由`wave_id < 4`的组补上对应barrier。

每个计算区域使用以下协议：

```text
sched_barrier -> s_setprio(1) -> MFMA -> s_setprio(0) -> raw s_barrier
```

每个memory区域显式发出`s_waitcnt`再执行raw `rocdl.s_barrier()`。普通区域只等待
`lgkmcnt(0)`；区域3和7与Gluon一样使用`vmcnt(8) lgkmcnt(0)`，保留8个VMEM操作
在途并跨半循环重叠。主循环最终包含128 MFMA、48 `ds_read_b128`、16 GMem-to-LDS
VMEM和16个`s_barrier`，与Gluon数量一致。

这里必须使用raw `s_barrier`。`gpu.barrier()`携带内存语义，LLVM会在其前面保守插入
`vmcnt(0)`，把下一tile的DMA全部等完；早期版本因此只有约628 TFLOPS。改用显式
waitcnt + raw barrier后，最终ISA保留`vmcnt(8)`，性能先恢复到约1082 TFLOPS，随后
再通过padding、连续B和地址预计算继续提升。

## 8. Epilogue

累加器使用 FlyDSL 标准转换：

```python
c_frag_bf16.store(c_frag.load().to(fx.BFloat16))
```

它替代手工加 `0x8000`、移位和截断的实现，使用正常浮点转换语义。

最新K sweep中FlyDSL和Gluon已进入约1.2%以内的波动区间。对
`K=1024..16384` 拟合 `latency = fixed + slope * K`：

| 实现 | fixed | 每增加 1024 K 的延迟 |
|---|---:|---:|
| FlyDSL `get_pids` | 12.62 us | 21.16 us |
| Gluon | 11.72 us | 21.42 us |

两边K斜率相差约1.2%，拟合固定项相差约0.90 us。此前较大的固定开销差异对应到明确的
epilogue差异：
旧 FlyDSL C fragment产生 64 条 `buffer_store_dwordx2`，Gluon经过
`convert_layout(..., mem_c_layout)` 后产生 32 条 `buffer_store_dwordx4`。直接把
`make_tiled_copy_C` 的 atom换成 `BufferCopy128b` 会在 lowering 中产生越界
`vector.extract_strided_slice`，因为相邻两个 64-bit fragment在线程内并不对应相邻地址。

最终实现沿用 pyhip CDNA4 GEMM和 FlyDSL gfx950 FlashAttention的 proven pattern：每次取
相邻两个 N accumulator slice，分别用 `cvt_pk_bf16_f32` 打成四个 dword，再调用
`rocdl.permlane16_swap` 交换相差 16 lane的数据，重组成一个 `i32x4`，最后用
`buffer_ops.buffer_store`直接发出 128-bit store。由于 C tile在本 kernel中先做过
`transposed_c_layout`，地址计算使用 `repeat * 2 + wave` 的 16x16 block顺序。

最终 epilogue ISA为：

```text
64 x v_permlane16_swap_b32
32 x buffer_store_dwordx4
```

同进程交替 12 轮、每轮 100 次，`4096^3` 中位延迟从 `0.10060 ms` 降到
`0.09839 ms`，提升 2.25%，吞吐从 `1366.2` 提升到 `1396.9 TFLOPS`。

8-wave采用natural-pipeline epilogue：在TL、BL、TR、BR各自最后一次MFMA后立即转换并
写出，避免四个象限同时存活到统一epilogue。该调整把private segment从76 B降到44 B；
DMA目标地址预计算后最终为20 B，且热循环内没有scratch load/store。

当前8-wave C fragment相邻的16列由不同wave持有，不能只靠wave内`permlane16`合并。
Gluon的`convert_layout`实际生成跨wave的`ds_write_b128 -> barrier -> ds_read_b128`交换。
FlyDSL中试验了同类LDS CShuffle，ISA从32条`buffer_store_dwordx2`变为16条
`buffer_store_dwordx4`，但受控性能从约1348.5降到1330.4 TFLOPS，因此没有保留。

## 9. 性能结果

同一`benchmark_gemm_950`配置的前后结果。BF16当前值来自12组×80次交替测试：

| Kernel | 初始延迟 | 初始性能 | 当前延迟 | 当前性能 | 总加速 |
|---|---:|---:|---:|---:|---:|
| 4-wave BF16 | 0.2861 ms | 480.4 TFLOPS | 0.09736 ms | 1411.6 TFLOPS | 2.94x |
| 8-wave BF16 | 0.2366 ms | 581.0 TFLOPS | 0.10215 ms | 1345.5 TFLOPS | 2.32x |
| 4-wave FP8 | 0.1500 ms | 916.2 TFLOPS | 0.05209 ms | 2638.5 TFLOPS | 2.88x |
| 8-wave FP8 | 0.1269 ms | 1083.2 TFLOPS | 0.08424 ms | 1631.6 TFLOPS | 1.51x |

最终Q1/Q3区间为：4-wave BF16 `0.09728/0.09751 ms`、8-wave BF16
`0.10195/0.10227 ms`。8-wave延迟比同批4-wave高4.9%，吞吐为4-wave的95.3%，达到
相同性能量级。单独测得Gluon 8-wave为`0.09607 ms / 1430.6 TFLOPS`，当前FlyDSL
仍慢约6.3%。4-wave FP8最终结果来自
10组交替顺序、每组80次的受控测试，中位数为`0.05209 ms / 2638.5 TFLOPS`；同一
进程中的无padding版本为`0.05206 ms / 2640.3 TFLOPS`，说明零冲突padding没有引入
可测性能回退。

4-wave BF16 的分阶段结果：

- 固定 scheduler、未固定 AGPR：`0.1657 ms / 829.2 TFLOPS`；
- tied AGPR、`has_side_effects=True`：`0.1488 ms / 923.5 TFLOPS`；
- tied AGPR、`has_side_effects=False`：三组 60 次运行中位数
  `0.1412 ms / 973.4 TFLOPS`，比上一阶段再提升约 5.4%。

最终用轮换执行顺序重新比较三种 BF16 路径，避免预热顺序偏差：

| 路径 | MIR MFMA 形式 | 热循环 AGPR read/write | 中位延迟 | 性能 |
|---|---|---:|---:|---:|
| native `fx.gemm` | `V_MFMA_*` | 96/96 | 0.1360 ms | 1010.8 TFLOPS |
| tied-inline | `INLINEASM` | 0/0 | 0.1205 ms | 1140.6 TFLOPS |
| native-anchor（默认） | `V_MFMA_*` | 0/0 | 0.1179 ms | 1165.6 TFLOPS |

native-anchor 比 tied-inline 延迟低约 2.2%，并恢复了 MFMA mask 分类。完成 alias、
live-range 和同步形式对齐后，FlyDSL 与真实 Gluon ISA 的八区 `M/D/V` 类别串达到
`352/352` 逐位一致，同步位置也完全一致。

同一组 `4096^3` BF16 输入、4-wave、`256x256` tile。最终对比采用预热20次、
7组×60次并丢弃前2组后取中位数：

| 实现 | 中位延迟 | 性能 | Q1/Q3 |
|---|---:|---:|---:|
| Gluon `get_pids` | 0.09663 ms | 1422.3 TFLOPS | 0.09649/0.09666 ms |
| FlyDSL，无 padding | 0.10890 ms | 1262.0 TFLOPS | - |
| FlyDSL，A padding + B 连续 + row-major PID | 0.10081 ms | 1363.3 TFLOPS | - |
| FlyDSL，A padding + B 连续 + `get_pids` | 0.10060 ms | 1366.2 TFLOPS | - |
| FlyDSL，以上配置 + permlane16 epilogue | 0.09839 ms | 1396.9 TFLOPS | - |
| FlyDSL，以上配置 + DMA地址预计算（最终） | 0.09724 ms | 1413.5 TFLOPS | 0.09719/0.09745 ms |

最终FlyDSL相比自身无padding基线快10.7%，与同批Gluon结果相差0.60 us（0.62%）。
核心交织、LDS bank conflict和L2命中率都已对齐；permlane16 epilogue和DMA地址预计算
分别消除了固定存储开销和热循环地址开销。

最终 `get_pids` K sweep采用相同的预热20次、7组×60次、丢弃前2组方法：

| K | FlyDSL | Gluon | 绝对差值 | 相对差值 |
|---:|---:|---:|---:|---:|
| 1024 | 0.03332 ms | 0.03356 ms | -0.24 us | -0.72% |
| 2048 | 0.05416 ms | 0.05447 ms | -0.31 us | -0.57% |
| 4096 | 0.09724 ms | 0.09663 ms | +0.60 us | +0.62% |
| 8192 | 0.18416 ms | 0.18369 ms | +0.47 us | +0.25% |
| 16384 | 0.35016 ms | 0.35439 ms | -4.24 us | -1.20% |

正差值表示FlyDSL较慢，负差值表示FlyDSL较快。五个K点均在约1.2%以内，已没有旧版本
随K稳定存在的数微秒差距。

### 9.1 DMA地址优化前的PMC定位

对完成permlane16 epilogue、尚未进行DMA地址预计算的版本和Gluon使用相同
`4096^3`、256 workgroups、每个workgroup 4 waves，按不超过4个counter一组分别采集
PMC。该表用于定位后续DMA地址优化，关键结果如下：

| Counter | FlyDSL | Gluon | FlyDSL/Gluon |
|---|---:|---:|---:|
| `SQ_INSTS_MFMA` | 8,388,608 | 8,388,608 | 1.000 |
| `SQ_VALU_MFMA_BUSY_CYCLES` | 134,217,728 | 134,217,728 | 1.000 |
| `SQ_INSTS_VMEM` | 1,081,344 | 1,091,584 | 0.991 |
| `SQ_ACTIVE_INST_VMEM` | 1,081,344 | 1,081,344 | 1.000 |
| `SQ_INSTS_LDS` | 2,097,152 | 2,162,688 | 0.970 |
| `SQ_WAIT_ANY` | 3,902,282 | 5,052,678 | 0.772 |
| `SQ_WAIT_INST_LDS` | 1,634,659 | 2,051,756 | 0.797 |
| `SQ_INSTS` | 18,859,008 | 17,259,520 | 1.093 |
| `SQ_INSTS_VALU` | 13,338,624 | 11,058,176 | 1.206 |
| `SQ_INSTS_VALU_INT32` | 2,744,320 | 1,068,032 | **2.570** |
| `SQ_THREAD_CYCLES_VALU` | 791,806,976 | 707,725,312 | 1.119 |
| `SQ_WAVE_CYCLES` | 42,228,324 | 41,718,814 | 1.012 |
| `SQ_BUSY_CYCLES` | 5,293,318 | 5,236,743 | 1.011 |

这些 counter排除了 MFMA吞吐、VMEM数量、LDS bank conflict、L2命中率和等待时间：
FlyDSL的 wait反而更少，MFMA busy完全相同。差异集中在普通 VALU，尤其 INT32地址运算。
按 1024 waves和 64 个 K tile归一化，FlyDSL每 wave、每 K tile多约 25.6 条 INT32
VALU，并多约 7.8 wave cycles。

最终 ISA的热循环也给出相同结论：FlyDSL有 116 条普通 VALU，Gluon只有 29 条。
FlyDSL热点集中在 `copy_bf16_gmem_to_lds`：

- B preshuffle地址每次 DMA都重新做 `//`、`%` 和 byte offset组合；
- A padding路径每次重新计算 row permutation和 `row * K + k`；
- 每个动态 LDS目标指针都会生成 `v_readfirstlane_b32 -> s_mov_b32 m0`。

Gluon则在 prologue建立 `mem_a_offsets` / `mem_b_offsets`，主循环只做统一递增。对应的
FlyDSL优化已经完成：prologue预计算 lane-local A/B源 offset，主循环仅叠加 K tile和
chunk常量；每组4条 `raw_ptr_buffer_load_lds`只对第一个LDS目标地址执行一次
`readfirstlane`，后3个地址在SGPR中按固定stride递增。

### 9.2 DMA地址预计算结果

最终ISA保持352条 `MFMA/DSRD/VMEM` 类别逐位不变，热循环
`v_readfirstlane_b32` 从32条降到8条，VGPR/AGPR/SGPR allocation仍为
`220/256/96`。相对仅预计算源 `voffset` 的版本，`4096^3` PMC如下：

| Counter | 源 `voffset` 预计算 | + LDS目标scalarize | 变化 |
|---|---:|---:|---:|
| `SQ_INSTS_VALU_INT32` | 1,256,448 | 1,231,872 | -2.0% |
| `SQ_INSTS_VALU` | 11,738,112 | 10,927,104 | -6.9% |
| `SQ_WAVE_CYCLES` | 41,471,251 | 40,931,398 | -1.3% |
| `SQ_BUSY_CYCLES` | 5,197,080 | 5,129,224 | -1.3% |

同一进程内交替运行两版，12组、每组80次的配对测试中，LDS目标scalarize版12/12
更快；配对中位延迟降低0.354 us。两版独立中位数为0.097570 ms和0.097227 ms，
对应1408.6和1413.6 TFLOPS。

也测试了把AT/AB/BL/BR象限基址分别编码到4个buffer descriptor。该方案将热循环
`v_add_u32` 从36条降到16条、`SQ_INSTS_VALU_INT32`降到579,584，但没有保留：
`K=8192` 的16组ABBA配对测试全部回退，中位慢0.562 us。说明减少地址指令不能替代
依赖链和descriptor切换的实际cycle验证。

指令前端不是独立瓶颈：FlyDSL `SQ_IFETCH` 比 Gluon多 5.3%，但
`SQ_IFETCH_LEVEL/SQ_IFETCH` 分别约为 0.232和 0.230，平均取指等待几乎相同；只是较多
地址指令带来的自然取指增量。

资源压力是次要因素。最终 ISA元数据为：

| 实现 | arch VGPR | AGPR | combined | SGPR |
|---|---:|---:|---:|---:|
| FlyDSL | 220 | 256 | 476 | 49 |
| Gluon | 168 | 256 | 424 | 87 |

两者 combined VGPR都大于256且不超过512，因此都处于1 wave/SIMD档，不存在 occupancy
跳档；`SQ_WAVE_CYCLES/SQ_BUSY_CU_CYCLES` 两边也都为1.0。FlyDSL多52个 arch VGPR
不会进一步降低 occupancy，但会压缩调度和地址临时值的空间，与额外 INT32 VALU来自同一
组复杂地址表达式。

运行基准：

```bash
python3 -c "from tests.flydsl.test_gemm import benchmark_gemm_950 as bench; \
print({f'{w}wave_{dtype}': bench(w, dtype) for w, dtype in \
((4, 'bf16'), (8, 'bf16'), (4, 'fp8'), (8, 'fp8'))})"
```

## 10. 正确性与汇编验证

参数化测试覆盖 4/8-wave × BF16/FP8，并使用四个 K tile，因此会执行一轮完整展开
主循环和两个 tile 的 tail：

```bash
pytest -q tests/flydsl/test_gemm.py::test_gemm_950_correctness -s
```

gfx950 上期望结果为 `4 passed`。

检查热循环交织和 AGPR 搬运：

```bash
asm=tests/flydsl/gemm_4wave_950_bf16.s
awk '/^\.LBB0_1:/{inloop=1} /s_cbranch_vccnz \.LBB0_1/{inloop=0} inloop' "$asm" \
  | grep -E 'v_mfma|ds_read|buffer_load|v_accvgpr_(read|write)'
```

严格检查 prologue、八个子块、prefetch、主循环等待和 tail 等待：

```bash
python3 tests/flydsl/verify_gemm_950_pipeline.py
```

脚本同时解析 Gluon/FlyDSL Python AST 和两份保存的最终 ISA，并要求：

- prologue DMA 顺序为两次 `BL -> AT -> AB -> BR`；
- 每个子块恰好包含对应来源的 32 MFMA、8 LDS read 和 4 GMem-to-LDS prefetch；
- 八个子块的 352 个 `MFMA/DSRD/VMEM` 类别位置逐位一致；
- 两份主循环的 `vmcnt` 都为 `[16,16]`，barrier 都为 2，AGPR 搬运都为 `0/0`；
- 同步相对位置一致：`vmcnt@168, barrier@169, vmcnt@344, barrier@345`；
- tail 逻辑等待顺序为 `5,4,3,2,1,0`。
