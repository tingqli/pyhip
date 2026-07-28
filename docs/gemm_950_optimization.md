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
- 测试环境：`gfx950:sramecc+:xnack-`、256 CU、288 GiB VRAM，PyTorch
  `2.11.0+gitd0c8b1f`，ROCm `7.2.53211`。
- 最新性能数据采集于 2026-07-26，源码基于提交 `1c51019` 加当前工作树修改。
- 除特别说明外，性能表使用 `M=N=K=4096`、输出 tile `256 x 256`、默认
  XCD-aware `get_pids`。同一表的候选先全部编译并预热20轮，再在同一进程中轮换执行
  20组、每组100次，报告中位数和Q1/Q3。case顺序逐轮循环shift，并在完成一整轮位置
  循环后整体反向，保证每个case在每个计时位置出现次数相同。
- 不同表来自独立进程，绝对延迟会受GPU温度和频率轻微影响；选择结论只比较同一表内
  的轮换结果。

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

Gluon参考在区域3和7等待；FlyDSL为支持短tile，把两次等待前移到区域2和6的LDS read
之前，确保下一stage的DMA已经完成。新鲜ISA中FlyDSL形成两组
`s_waitcnt vmcnt(8) lgkmcnt(0)`加`s_barrier`，位于分类指令索引89和265；Gluon参考
仍为`vmcnt(16)`，位于168和344。Tail的六次逻辑等待为
`wait_group(5,4,3,2,1,0)`，对应物理VMEM计数`20,16,12,8,4,0`。

当前实现把 `AT/AB/BL/BR` 的两个 stage 拆成 8 个独立 `SharedStorage950` leaf。
`SharedAllocator(static=True)` 因此生成 8 个独立 LDS global，而不是从 4 个大数组用
`add_offset`切出第二个 stage。DMA目标指针统一表达为：

```text
独立 stage global root + wave-relative byte offset + chunk offset
```

wave-relative offset只在prologue计算一次，DMA仍生成`buffer_load_dwordx4 ... lds`。
最终LLVM IR中没有`ptrtoint/inttoptr`，也没有`alias.scope/noalias`。LLVM可以利用不同
global的对象身份排除跨stage/跨象限别名；同一对象的真实生命周期依赖仍由现有
wait/barrier表达。相比旧metadata方案，后端会保留少量更严格的`vmcnt`，但受控测试未
测得性能回退。

BF16 路径复用了 Gluon 的两个 B-fragment live-range anchor，并在区域 3 后延长 `AB/BR`
fragment live range，避免 LDS read 目标 VGPR 与仍被 MFMA 消费的输入发生 WAR 复用。
独立消融确认这些 B anchor 不影响正确性，但会稳定影响调度和性能，因此正式路径继续保留。

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

8-wave FP8在相同两组4-wave调度框架上移植了A8W8的双padding A布局。A的write/read
view继续使用`[[1024,16],[2048,32]]`，GMem源行按8行分组重排；B按host preshuffle
字节序通过raw DMA直接复制，并由每个wave按`row = lane%16 + (wave%4)*16`读取两个
连续K16向量。生产`256x256` tile还包含第二个64-row N repeat。

`4096^3` PMC显示优化过程为：

| 8-wave FP8配置 | `SQ_LDS_BANK_CONFLICT` | `SQ_WAVE_CYCLES` | `SQ_BUSY_CYCLES` |
|---|---:|---:|---:|
| 原generic A/B LDS布局 | 18,874,368 | 78,417,450 | 4,968,092 |
| A8W8双padding A | 6,291,456 | 65,184,150 | 4,136,550 |
| + 连续preshuffle B（最终） | **0** | **47,986,713** | **3,056,110** |

最终四次dispatch的`SQ_LDS_BANK_CONFLICT`和`SQ_LDS_DATA_FIFO_FULL`均为0。ISA使用
133,120 B LDS。移除8-wave的MFMA调度提示后、采用permlane epilogue之前，FP8 ISA
使用256个VGPR、52 B private segment，并在tail中有3对
`scratch_store/load_dwordx4`；BF16 ISA使用244个VGPR、0 B private segment且没有
scratch。第6.4节记录了这项历史资源变化，当前permlane资源见第8节和第9.11节。

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
`MFMA32`。

8-wave 不再把 MFMA 数量作为 memory scheduler 的时间轴，也不再发出
`sched_mfma`。设当前区域有 `D` 条 DSRD、`V` 条 VMEM，循环值取
`S = max(D, V)`；第 `i` 步的累计目标分别为 `ceil((i + 1) * D / S)` 和
`ceil((i + 1) * V / S)`。较密集类别因此每步恰好发一条，较稀疏类别按比例分布，
既没有空步，也严格保持 `D`/`V` 总数。同一步先发 DSRD 再发 VMEM，保留了旧调度
删除 MFMA 后的 memory 相对顺序。

BF16 与 FP8 的 tile stage 字节数相同，因此当前实际循环值一致：

| Tile | A/B DSRD | VMEM | A/B `S` | A/B 目标序列 |
|---|---:|---:|---:|---|
| `256x256` | `8 / 4` | `2` | `8 / 4` | `DVDDDDVDDD / DVDDVD` |
| `128x128` | `4 / 2` | `1` | `4 / 2` | `DVDDD / DVD` |

这里 `D` 表示一个 `sched_dsrd(1)`，`V` 表示一个 `sched_vmem(1)`。VMEM-first
和把 VMEM 放在完整事件序列中点的候选都做过相同输入的轮转对照；DSRD-first 在四组
`tile x dtype` 的首轮筛选中有三组最快，另一组与最快值只差 0.015%。

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

1. **当前采用的显式 accumulator-chain 方案**：`gemm_bf16_agpr` 按 `f32x4` slice
  构造原生 ROCDL MFMA SSA 链。它同时保留 MFMA 调度分类和零热循环 AGPR 搬运；当前
  LLVM 后端不再需要额外的 C-fragment 回边 anchor。`sched_group` 仍是 best-effort，
  不保证逐位模板。
2. **整段区域写成一个 inline asm**：把 32 MFMA、8 DSRD、4 VMEM 全部写入一个 asm
   模板，顺序最严格，但地址、waitcnt、寄存器约束和可维护性成本很高，编译器也无法再优化。
3. **扩展 FlyDSL/LLVM**：新增能同时表达“原生 MFMA opcode”和“accumulator tied AGPR”
   的 lowering，使 Machine IR 保留真正的 `V_MFMA_*` opcode，随后
   `sched.group.barrier(mask_mfma, ...)` 才有机会严格匹配。这是结构上最正确、但需要修改
   FlyDSL/LLVM 后端的方案。

### 5.6 历史验证脚本

此前曾使用本地脚本对照 native、tied inline asm 和 native-anchor。该脚本依赖后来删除的
实验参数 `native_bf16_agpr_anchor`，以下内容仅保留为历史定位记录，不是当前可执行接口。

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

该脚本记录的是引入显式 accumulator chain 之前的定位过程。当前正式路径已经通过
第 6.1 节的独立消融删除 C-fragment 回边 anchor；其中 native-anchor 对照及其断言应
视为历史实验，不再代表默认实现。

## 6. AGPR accumulator 固定

4-wave launch 使用：

```python
value_attrs = {"passthrough": [["amdgpu-agpr-alloc", "256,256"]]}
launch.compile_hints["llvm_options"] = {"amdgpu-mfma-vgpr-form": False}
```

通用 `fx.gemm` 会让每个象限成为 loop-carried `vector<64xf32>` SSA 值。Layout
lowering 虽然仍生成相同的原生 K32 MFMA opcode，但后端需要解决宽 fragment 在循环回边
的 AGPR 槽置换。当前实现改为按 `f32x4` slice 显式构造 accumulator chain：

```python
for m in range_constexpr(...):
  for n in range_constexpr(...):
    c_slice = c_frag[None, m, n]
    acc = arith._to_raw(c_slice.load())
    for k in range_constexpr(TILE_K // 32):
      acc = rocdl.mfma_f32_16x16x32_bf16(..., acc, ...)
    c_slice.store(acc)
```

真正计算仍是原生 `V_MFMA_F32_16X16X32_BF16`，所以 `mask_mfma` 可以识别。显式
slice chain 已足以让当前 LLVM 22 后端在热循环保持稳定 AGPR 分配，不再需要额外的
空 `=a` C-fragment 回边 anchor。

修复后的保存汇编中：

```text
hot loop: 256 MFMA, 0 accvgpr_read, 0 accvgpr_write
```

初始化、epilogue 和其他非热循环边界仍可包含 AGPR/VGPR 传输；关键是循环回边不再出现
原生 `fx.gemm` 路径的 96/96 次置换搬运。

### 6.1 B/C anchor 与 `fx.gemm` 独立消融

在 gfx950 上分别控制 B-fragment anchor、C-fragment 回边 anchor 和 BF16 GEMM helper，
并检查短 K/长 K 正确性、LLVM/MIR、最终 ISA、寄存器资源与同进程轮换性能。结论如下：

2026-07-26 使用 `/host_lc/FlyDSL` 当前源码重新构建并 editable 安装
`flydsl 0.2.4.dev747`（LLVM/MLIR commit `7f77ca0d`），关闭 JIT disk cache 后重新验证：

| 变体 | 正确性 | 最终 ISA / 资源变化 | 性能结果 | 决策 |
|---|---|---|---|---|
| 删除 `anchor_b_frag` | K=512/4096/8192 通过 | M/D/V 连续段 `193 -> 156`；`next_free_vgpr 460 -> 464`，`accum_offset 204 -> 208` | K=2048/4096/8192 延迟分别回退 1.49%/1.33%/1.85% | 保留 |
| 恢复 `anchor_c_frag_agpr` | K=512/4096/8192 通过 | 可执行指令流、M/D/V 序列和资源完全不变；MIR 仅多 4 个空 `INLINEASM` | 三个 K 点差异为 +0.04%/-0.04%/+0.16%，属于噪声 | 继续删除 |
| 全量改用 `fx.gemm` | K=512/4096/8192 通过 | 热循环增加 100/100 次 AGPR read/write；`next_free_vgpr 460 -> 512`，private segment `0 -> 44 B`，并有 10/10 次 scratch store/load | K=2048/4096/8192 延迟分别回退 11.96%/12.83%/13.73%；K=4096 为 0.09731 -> 0.10979 ms | 拒绝 |

因此 `anchor_b_frag` 是性能约束而不是正确性约束；`anchor_c_frag_agpr` 对当前显式
accumulator-chain 路径冗余；`gemm_bf16_agpr` 不能在不损失性能和寄存器质量的前提下
直接替换为 `fx.gemm`。上述 C-anchor 结论只针对当前 helper 和 LLVM 后端，不应泛化为
所有宽 fragment lowering 都不需要 live-range 约束。

### 6.2 删除 `native_bf16_agpr_anchor`

该参数并不控制当前需要保留的 AGPR anchor，而是选择两种 BF16 MFMA 表达：默认值
`True` 生成原生 `llvm.amdgcn.mfma` intrinsic，`False` 则生成 tied inline asm。
在当前 FlyDSL/LLVM 上重新验证后：

- 两条路径在 K=512/4096/8192 都正确；
- tied-inline 路径把 512 条原生 MIR `V_MFMA_*` 全部变成 `INLINEASM`，使
  `sched_group` 的 MFMA mask 无法识别；
- tied-inline 路径的热循环 M/D/V 连续段从 193 降到 137，`s_waitcnt` 从 31 增到 37；
- K=2048/4096/8192 的性能差异仅为 -0.11%/-0.08%/+0.04%，没有可保留的收益。

因此正式路径固定使用原生 MFMA intrinsic，并删除 `native_bf16_agpr_anchor` 参数及
tied-inline 分支。删除前后最终可执行 ISA 逐条一致，寄存器资源和热循环 AGPR 搬运均不变。

### 6.3 三个手工 copy helper 的必要性

在 4-wave BF16/FP8、`128x128`/`256x256` tile、K=512/4096/8192 上分别替换三个
helper，并检查正确性、最终 ISA、寄存器资源、同进程轮换性能及 PMC：

| Helper | 替代方案 | 结果 | 决策 |
|---|---|---|---|
| `copy_lds_to_bf16_frag` | 直接使用 `fx.copy(lds_copy_atom, src, dst)` | 全部正确；`ds_read_b128` 数量、M/D/V 交织、VGPR、wait、scratch不变；性能差异在噪声内 | 删除 |
| `copy_contiguous_b_lds_to_frag` | 直接使用普通 B view 的 `fx.copy` | BF16/FP8 均错误，普通 view 未表达 host preshuffle 物理地址 | 保留 |
| `copy_contiguous_b_lds_to_frag` | 改用完整 preshuffle LDS view 后 `fx.copy` | 正确且 bank conflict/FIFO full 均为0，但 BF16 增加4个架构VGPR和额外wait；K=8192 的16组配对仅2组更快，中位回退0.33% | 保留 |
| `copy_bf16_gmem_to_lds` | 直接使用原 src/dst view 的 async `fx.copy` | A/B 两侧在BF16/FP8中均错误，缺少A分组行重排或B preshuffle原样复制 | 保留 |
| `copy_bf16_gmem_to_lds` | 把A分组行重排编码进global source view后async `fx.copy` | 正确且bank conflict/FIFO full均为0，但BF16架构VGPR增加16个；BF16/FP8的K=4096/8192均稳定回退约0.2%~0.3% | 保留 |

因此只有 `copy_lds_to_bf16_frag` 是冗余的手工展开。其余两个helper并非DSL语义上绝对
无法用layout表达，但在当前编译器上仍是保持正确物理映射、寄存器质量和长K性能所必需的
实现形式。六组PMC（BF16/FP8各三版）的`SQ_LDS_BANK_CONFLICT`和
`SQ_LDS_DATA_FIFO_FULL`均为0。

### 6.4 8-wave copy helper 与 scheduler 消融

对8-wave BF16/FP8分别隔离A DMA、B DMA/reader、A reader和`hot_loop_scheduler`，覆盖
`128x128`/`256x256` tile及K=512/4096/8192：

| 项目 | 替代方案 | 结果 | 决策 |
|---|---|---|---|
| `copy_a_gmem_to_lds_raw_8wave` / `copy_a_gmem_to_lds_8wave` | 直接generic async copy | BF16/带paddingFP8均错误，缺少8行分组重排 | 保留 |
| 同上 | 把A分组重排编码进global source view后generic copy | 正确；FP8性能接近，但BF16产生48 B private segment和22条scratch指令，K=4096/8192回退24%/18% | 保留 |
| `copy_lds_to_frag_bf16_8wave` | `fx.copy(lds_copy_atom, src, dst)` | 全部正确；DS read数量、VGPR、M/D/V交织和性能均等价 | 删除 |
| `copy_b_gmem_to_lds_8wave` | 仅DMA改generic | BF16/FP8错误，写入格式与手工reader不一致 | 保留协议 |
| `copy_b_lds_to_frag_bf16_8wave` / `copy_b_lds_to_frag_fp8_8wave` | 仅reader改generic | BF16/FP8错误，读取格式与手工DMA不一致 | 保留协议 |
| 上述B DMA与reader成对generic化 | 两端统一使用logical tiled-copy格式 | 全部正确且无scratch，但重新引入BF16 `4,194,304`、FP8 `6,291,456`次bank conflict；性能分别回退约6.4%和8.6%~9.1% | 保留手工协议 |
| `hot_loop_scheduler` | 删除全部`sched_mfma/dsrd/vmem`提示，保留region fence | 正确；256 BF16回退约0.18%~0.21%，128 tile的BF16/FP8回退约1.4%~3.3%，M/D/V连续段从48降到31 | 拒绝全删 |
| `hot_loop_scheduler` | 只删除`sched_mfma`，以`max(DSRD, VMEM)`为循环值并保留全部memory提示 | 4/8-wave x BF16/FP8正确；8-wave 256 BF16的LLVM调度组由356减至100，MFMA `256 -> 0`而DSRD/VMEM仍为`84/16`；最终热循环的`128M/48D/16V`和48个连续段逐项不变，`next_free_vgpr 254 -> 244`且仍无scratch。FP8则由36 B/8 spills/2对scratch变为52 B/12 spills/3对scratch | 采用 |

最终memory-only版本与旧MFMA版本又覆盖`128x128`/`256x256`、BF16/FP8、
`K=2048/4096/8192`做20轮、每轮100次的ABBA配对。12组中9组更快，变化范围为
`-0.423%`到`+0.209%`；三组轻微回退没有随K增长，长K不存在稳定退化。这个结果说明
旧版本的性能职责来自DSRD/VMEM提示，不需要为已经独立成compute phase的MFMA再发一遍
组约束。三个无MFMA的D/V顺序候选都产生相同的FP8 spill增加，说明它来自MFMA组约束
消失后的寄存器分配变化，而不是所选DSRD-first顺序；实测收益已包含这项代价。

删除A reader后的正式8-wave BF16/FP8再次用PMC确认：
`SQ_LDS_BANK_CONFLICT=0`、`SQ_LDS_DATA_FIFO_FULL=0`。A source-layout DMA候选也保持
零冲突，但因BF16 spill和长K回退被淘汰。因此8-wave中只有
`copy_lds_to_frag_bf16_8wave`属于冗余手工展开；其余copy helper和memory scheduler
仍有明确的正确性或性能职责。

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
后续绝对地址预计算阶段进一步降到20 B。permlane优化前的memory-only scheduler版本中，
BF16为0 B且没有scratch，FP8为52 B并在tail中有3对scratch load/store。采用
permlane16 + 128-bit epilogue后，当前BF16 ISA为40 B和3对scratch，FP8为24 B和2对
scratch；两者都由32条`buffer_store_dwordx2`降为16条`buffer_store_dwordx4`。
热循环本身的MFMA/DSRD/VMEM类别序列保持不变。

早期尝试误把相邻16列作为8-wave epilogue的打包单位，因此转而测试了Gluon式
`ds_write_b128 -> barrier -> ds_read_b128`跨wave CShuffle。该方案虽把32条
`buffer_store_dwordx2`变为16条`buffer_store_dwordx4`，受控性能却从约1348.5降到
1330.4 TFLOPS，因而没有保留。最终实现配对的是同一wave内相隔64列的两个N repeat，
再用`permlane16.swap`重排16-lane group，不需要跨wave LDS交换。

## 9. 性能结果

### 9.1 当前主结果

同一`benchmark_gemm_950`配置的前后结果。当前值来自本轮K sweep中
`M=N=K=4096`、`256x256` tile的20组x100次平衡轮换测试：

| Kernel | 初始延迟 | 当前延迟 | Q1/Q3 | 当前性能 | 总加速 |
|---|---:|---:|---:|---:|---:|
| 4-wave BF16 | 0.2861 ms | 0.097076 ms | 0.096830/0.097275 ms | 1415.8 TFLOPS | 2.95x |
| 8-wave BF16 | 0.2366 ms | 0.099465 ms | 0.099172/0.099698 ms | 1381.8 TFLOPS | 2.38x |
| 4-wave FP8 | 0.1500 ms | 0.048150 ms | 0.048111/0.048405 ms | 2854.4 TFLOPS | 3.12x |
| 8-wave FP8 | 0.1269 ms | 0.047725 ms | 0.047671/0.047860 ms | 2879.8 TFLOPS | 2.66x |

本表当前值为独立stage object + 显式wave-relative GEP + 无alias metadata，并包含
8-wave permlane16 + 128-bit epilogue。与旧实现和历史阶段的比较保留在后续小节。
本轮同表比较中，8-wave BF16比4-wave慢2.46%，8-wave FP8则快0.88%。

### 9.2 K sweep

固定`M=N=4096`、`256x256` tile。表中20个路径点均先以NaN预填输出并通过全输出
数值检查，再进入计时：

| DType | K | 4-wave ms（Q1/Q3） | 4-wave TFLOPS | 8-wave ms（Q1/Q3） | 8-wave TFLOPS | 8-wave延迟差 |
|---|---:|---:|---:|---:|---:|---:|
| BF16 | 1024 | 0.031661（0.031589/0.031908） | 1085.2 | 0.031951（0.031835/0.032048） | 1075.4 | +0.92% |
| BF16 | 2048 | 0.053268（0.053215/0.053321） | 1290.1 | 0.054218（0.054126/0.054262） | 1267.5 | +1.78% |
| BF16 | 4096 | 0.097076（0.096830/0.097275） | 1415.8 | 0.099465（0.099172/0.099698） | 1381.8 | +2.46% |
| BF16 | 8192 | 0.184099（0.183617/0.184283） | 1493.1 | 0.189637（0.189073/0.189943） | 1449.5 | +3.01% |
| BF16 | 16384 | 0.353041（0.352164/0.353984） | 1557.2 | 0.360476（0.358888/0.361844） | 1525.1 | +2.11% |
| FP8 | 1024 | 0.019525（0.019403/0.019870） | 1759.7 | 0.019319（0.019188/0.019656） | 1778.5 | -1.06% |
| FP8 | 2048 | 0.029589（0.029465/0.029751） | 2322.5 | 0.028843（0.028811/0.029176） | 2382.5 | -2.52% |
| FP8 | 4096 | 0.048150（0.048111/0.048405） | 2854.4 | 0.047725（0.047671/0.047860） | 2879.8 | -0.88% |
| FP8 | 8192 | 0.086214（0.086079/0.086429） | 3188.3 | 0.086075（0.085763/0.086370） | 3193.5 | -0.16% |
| FP8 | 16384 | 0.163020（0.162404/0.164293） | 3372.3 | 0.162902（0.162786/0.163164） | 3374.8 | -0.07% |

按`latency = fixed + slope * (K / 1024)`拟合：

| 路径 | fixed | 每增加1024 K的延迟 |
|---|---:|---:|
| 4-wave BF16 | 10.94 us | 21.43 us |
| 8-wave BF16 | 11.26 us | 21.92 us |
| 4-wave FP8 | 10.09 us | 9.55 us |
| 8-wave FP8 | 9.61 us | 9.58 us |

BF16中4-wave同时有更低固定项和更低斜率，五个K点均领先。permlane epilogue使FP8
8-wave固定项从旧拟合的12.92 us降至9.61 us，比4-wave低0.48 us；两者斜率只差0.3%，
因此8-wave在五个K点均持平或领先0.07%到2.52%。

### 9.3 Tile sweep

固定`4096^3`，只统计第10节全尺寸数值验证通过的配置：

| DType | 路径 / tile | Grid | 中位延迟 | Q1/Q3 | TFLOPS |
|---|---|---:|---:|---:|---:|
| BF16 | 4-wave `256x256` | 256 | 0.096645 ms | 0.096329/0.097040 ms | 1422.1 |
| BF16 | 8-wave `128x128` | 1024 | 0.133658 ms | 0.133378/0.134654 ms | 1028.3 |
| BF16 | 8-wave `128x256` | 512 | 0.121745 ms | 0.120051/0.123390 ms | 1128.9 |
| BF16 | 8-wave `256x128` | 512 | 0.118902 ms | 0.118546/0.120265 ms | 1155.9 |
| BF16 | 8-wave `256x256` | 256 | 0.099342 ms | 0.098940/0.099704 ms | 1383.5 |
| FP8 | 4-wave `256x256` | 256 | 0.048090 ms | 0.048066/0.048220 ms | 2857.9 |
| FP8 | 8-wave `128x128` | 1024 | 0.065071 ms | 0.064738/0.065232 ms | 2112.1 |
| FP8 | 8-wave `128x256` | 512 | 0.062209 ms | 0.061825/0.062721 ms | 2209.3 |
| FP8 | 8-wave `256x128` | 512 | 0.060349 ms | 0.060033/0.060753 ms | 2277.4 |
| FP8 | 8-wave `256x256` | 256 | 0.047582 ms | 0.047535/0.047678 ms | 2888.5 |

8-wave中`256x256`在两种dtype都最快。相对该配置，BF16三个较小tile慢19.7%到
34.5%，FP8慢26.8%到36.8%；更多workgroup没有抵消更小tile带来的重复加载与固定开销。
`256x256`下8-wave BF16比4-wave慢2.79%，8-wave FP8则快1.06%。

### 9.4 Problem size与PID映射

固定`K=4096`、`256x256` tile的方形问题规模。16个规模/路径组合均通过全输出
数值检查：

| M=N | Grid | 4w BF16 ms / TFLOPS | 8w BF16 ms / TFLOPS | 4w FP8 ms / TFLOPS | 8w FP8 ms / TFLOPS |
|---:|---:|---:|---:|---:|---:|
| 1024 | 16 | 0.067208 / 127.8 | **0.066804 / 128.6** | 0.038889 / 220.9 | **0.035724 / 240.5** |
| 2048 | 64 | 0.068242 / 503.5 | **0.067771 / 507.0** | 0.039294 / 874.4 | **0.037612 / 913.5** |
| 4096 | 256 | 0.097126 / 1415.1 | 0.100063 / 1373.5 | 0.048249 / 2848.5 | **0.047753 / 2878.1** |
| 8192 | 1024 | 0.373751 / 1470.9 | 0.386015 / 1424.2 | 0.183202 / 3000.8 | **0.180909 / 3038.9** |

BF16在`1024^2`和`2048^2`欠填充规模由8-wave快0.60%/0.69%，在`4096^2`和`8192^2`
则由4-wave快3.02%/3.28%。FP8四个规模均由8-wave领先，优势从欠填充`1024^2`的
8.14%收窄到其余规模的1.03%到4.28%。

`4096^3`下默认`get_pids`相对row-major的16组x100次配对结果：

| 路径 | row-major | `get_pids` | 延迟变化 |
|---|---:|---:|---:|
| 4-wave BF16 | 0.096978 ms | 0.096757 ms | -0.228% |
| 8-wave BF16 | 0.099822 ms | 0.099761 ms | -0.061% |
| 4-wave FP8 | 0.048534 ms | 0.048204 ms | -0.679% |
| 8-wave FP8 | 0.081099 ms | 0.048303 ms | -40.440% |

四条路径均受益，因此继续默认启用XCD-aware映射。8-wave FP8的幅度不是单纯cache
locality：独立进程和手工ABBA均复现约81 us对48 us，最终ISA显示row-major路径产生
84 B private segment，而`get_pids`仅24 B。其余三条路径收益为0.06%到0.68%。

### 9.5 与Gluon和本地FP8 8-wave统一对比

`tests/flydsl/compare_gemm_950.py`把所有实现包装为只发起kernel launch的无参闭包；
输入转换、B转置/preshuffle、输出分配、编译和正确性检查均在计时区外完成。所有闭包
统一使用`pyhip.cudaPerf`，预热20轮后执行20组、每组100次，并输出CSV。case顺序逐轮
循环shift，并在完成一整轮位置循环后整体反向；当round数是case数的整数倍时，每个
case在每个计时位置出现次数相同。旧的“每轮shift后再按奇偶反转”在只有两个case时
会退化为固定A->B顺序，相关数据已重测。

上游使用`ROCm/gfx950-gluon-tutorials`提交`8686f59`与Triton
`gfx950-tutorial-v1.1`（`b51f761`）：

- 4-wave Gluon使用官方`full`配置：LLIR scheduler + force-AGPR + amdgcnas；
- 8-wave Gluon按官方要求使用`base`/no-AGPR，不加载4-wave插件；
- 脚本默认`--gluon-config auto`，自动拆成两个隔离子进程并合并CSV，避免force-AGPR
  污染8-wave；
- FlyDSL和本地pyhip JIT的FP8是E4M3FN输入、BF16输出；Gluon BF8是E5M2输入、FP16
  输出。两者硬件指令宽度相同，但数值格式不同，因此只比较kernel吞吐，不声称逐位
  等价。

`M=N=K=4096`正式结果：

| 路径 | 输入/输出 | 中位延迟 | Q1/Q3 | TFLOPS | 相对同wave FlyDSL |
|---|---|---:|---:|---:|---:|
| FlyDSL 4-wave BF16 | BF16/BF16 | 0.096564 ms | 0.096292/0.096933 ms | 1423.3 | - |
| Gluon 4-wave BF16 full | BF16/BF16 | 0.107574 ms | 0.107365/0.108125 ms | 1277.6 | +11.40%延迟 |
| FlyDSL 4-wave FP8 | E4M3FN/BF16 | 0.050544 ms | 0.049827/0.051056 ms | 2719.2 | - |
| Gluon 4-wave BF8 full | E5M2/FP16 | 0.106751 ms | 0.106472/0.107275 ms | 1287.5 | +111.20%延迟 |
| FlyDSL 8-wave BF16 | BF16/BF16 | 0.099292 ms | 0.099010/0.099439 ms | 1384.2 | - |
| Gluon 8-wave BF16 base | BF16/BF16 | 0.095732 ms | 0.095369/0.095874 ms | 1435.7 | -3.59%延迟 |
| FlyDSL 8-wave FP8 | E4M3FN/BF16 | 0.047487 ms | 0.047405/0.047653 ms | 2894.3 | - |
| Gluon 8-wave BF8 base | E5M2/FP16 | 0.050111 ms | 0.050081/0.050183 ms | 2742.7 | +5.53%延迟 |
| pyhip JIT 8-wave preshuffle | E4M3FN/BF16 | 0.044933 ms | 0.044860/0.045021 ms | 3058.8 | -5.38%延迟 |
| pyhip JIT 8-wave row-major | E4M3FN/BF16 | 0.044924 ms | 0.044855/0.045010 ms | 3059.3 | -5.40%延迟 |

4-wave Gluon把256个accumulator AGPR的初始化/回读成本集中在固定项，短K下并不占优；
缓存元数据确认`amdgpu-agpr-alloc=256`、LLIR scheduler barrier和零scratch均已生效。
在教程用于报告峰值的长K形状，Gluon重新领先：

| 形状 | FlyDSL | Gluon full | Gluon延迟优势 |
|---|---:|---:|---:|
| BF16 `4096x4096x8192` | 0.184416 ms / 1490.5 TFLOPS | 0.178148 ms / 1543.0 TFLOPS | 3.40% |
| FP8/BF8 `4096x4096x16384` | 0.162135 ms / 3390.7 TFLOPS | 0.157202 ms / 3497.1 TFLOPS | 3.04% |

本地对比项直接调用`tests/contrib/gemm/test_fp8_8wave.py`使用的
`pyhip.contrib.gemm_fp8.gemm_8wave_fp8bf16fp16`，分别覆盖`bpreshuffle=True/False`。
两条路径都通过相同输入的正确性检查；preshuffle与row-major自身只差0.02%，属于同一
性能档。采用permlane16 + 128-bit epilogue后，FlyDSL与pyhip JIT在`4096^3`下的差距
已由优化前约11%缩小到5.38%/5.40%；同时FlyDSL 8-wave FP8从略慢于Gluon 8-wave转为
低5.24%延迟。

按Gluon BF8官方headline形状`M=N=4096, K=16384`补测同一组8-wave实现：

| 路径 | 输入/输出 | 中位延迟 | Q1/Q3 | TFLOPS | 相对FlyDSL | 相对Gluon |
|---|---|---:|---:|---:|---:|---:|
| FlyDSL 8-wave FP8 | E4M3FN/BF16 | 0.162738 ms | 0.162517/0.163264 ms | 3378.2 | - | -2.27%延迟 |
| Gluon 8-wave BF8 base | E5M2/FP16 | 0.166518 ms | 0.165886/0.166940 ms | 3301.5 | +2.32%延迟 | - |
| pyhip JIT 8-wave preshuffle | E4M3FN/BF16 | 0.158808 ms | 0.158276/0.159076 ms | 3461.8 | -2.42%延迟 | -4.63%延迟 |
| pyhip JIT 8-wave row-major | E4M3FN/BF16 | 0.159153 ms | 0.158633/0.159540 ms | 3454.3 | -2.20%延迟 | -4.42%延迟 |

长K下FlyDSL 8-wave比Gluon 8-wave低2.27%延迟；pyhip JIT仍领先，但相对FlyDSL的
优势进一步收窄为2.20%到2.42%。preshuffle比row-major低0.22%延迟，差异仍在很小的
范围内。该表延续前述格式边界：Gluon使用E5M2/FP16，另外三条路径使用E4M3FN/BF16。

把同形状下4-wave结果也纳入排名，Gluon 4-wave full以0.157202 ms最快；pyhip JIT
preshuffle/row-major只慢1.02%/1.24%，同时比FlyDSL 4-wave快2.05%/1.84%，比FlyDSL
8-wave快2.42%/2.20%。FlyDSL 4-wave比8-wave低0.37%延迟。因此在该官方长K形状，
pyhip JIT仍位于4-wave Gluon之后、两条FlyDSL和Gluon 8-wave之前。

复现一次完整对比：

```bash
PYTHONPATH=/path/to/triton-gfx950-v1.1/python \
python3 tests/flydsl/compare_gemm_950.py \
  --gluon-repo /path/to/gfx950-gluon-tutorials \
  --gluon-config auto --m 4096 --n 4096 --k 4096 \
  --warmup 20 --rounds 20 --iterations 100 \
  --csv /tmp/gemm950_compare.csv
```

Triton必须按上游要求使用`nanobind==2.10.2`和`TRITON_EXT_ENABLED=1`构建；4-wave
full配置的LLIR插件与Triton LLVM pin有ABI绑定。脚本会先编译FlyDSL，再global-load
`libtriton`供LLIR插件解析LLVM符号。

### 9.6 4-wave非标准tile修复与性能回归

4-wave原实现只有`256x256`方形tile可靠，存在两个独立根因：

1. C fragment在`transposed_c_layout`后按`[value, N-repeat, M-repeat]`组织，但
   `gemm_bf16_agpr`和permlane epilogue按`[M-repeat, N-repeat]`索引。方形tile掩盖
   了mode交换；矩形BF16会在`vector.extract_strided_slice`越界，FP8会写错或只写
   一半输出。
2. Region 2/6首次读取下一stage的LDS fragment时，尚未等待对应GMem-to-LDS DMA；
   `256x256`靠较长MFMA段偶然隐藏延迟，`128x128`多block主循环会非确定性读到未完成
   数据。修复将`wait_vmem_barrier(a_vmem_count + b_vmem_count)`前移到Region 2/6的
   LDS read之前，Region 3/7不再做滞后的等待。

新增`test_gemm_950_4wave_multiblock`，覆盖4种tile x BF16/FP8，使用至少2x2
workgroup、四个K tile并以NaN预填输出。加上原4项smoke测试共`12 passed`；进一步在
`4096^3`上对8种tile/dtype组合做全输出检查也全部通过。

修复前后标准`256x256`路径采用同进程16轮x100次ABBA。BF16三个K点变化均在
±0.10%噪声内；FP8三个K点稳定提升0.79%到1.30%，因此没有性能回退：

| DType | K | 修复前 | 修复后 | 变化 |
|---|---:|---:|---:|---:|
| BF16 | 2048 | 0.053491 ms | 0.053466 ms | -0.05% |
| BF16 | 4096 | 0.097504 ms | 0.097579 ms | +0.08% |
| BF16 | 8192 | 0.184494 ms | 0.184667 ms | +0.09% |
| FP8 | 2048 | 0.029786 ms | 0.029550 ms | -0.79% |
| FP8 | 4096 | 0.048722 ms | 0.048089 ms | -1.30% |
| FP8 | 8192 | 0.087832 ms | 0.086730 ms | -1.26% |

### 9.7 历史分阶段结果

4-wave BF16 的分阶段结果：

- 固定 scheduler、未固定 AGPR：`0.1657 ms / 829.2 TFLOPS`；
- tied AGPR、`has_side_effects=True`：`0.1488 ms / 923.5 TFLOPS`；
- tied AGPR、`has_side_effects=False`：三组 60 次运行中位数
  `0.1412 ms / 973.4 TFLOPS`，比上一阶段再提升约 5.4%。

当时用轮换执行顺序重新比较三种 BF16 路径，避免预热顺序偏差：

| 路径 | MIR MFMA 形式 | 热循环 AGPR read/write | 中位延迟 | 性能 |
|---|---|---:|---:|---:|
| native `fx.gemm` | `V_MFMA_*` | 96/96 | 0.1360 ms | 1010.8 TFLOPS |
| tied-inline | `INLINEASM` | 0/0 | 0.1205 ms | 1140.6 TFLOPS |
| native-anchor（当时默认） | `V_MFMA_*` | 0/0 | 0.1179 ms | 1165.6 TFLOPS |

native-anchor 比 tied-inline 延迟低约 2.2%，并恢复了 MFMA mask 分类。当前实现已按
第 6.1 节删除冗余 C-fragment anchor。完成当时的
alias metadata、live-range 和同步形式对齐后，FlyDSL 与真实 Gluon ISA 的八区 `M/D/V` 类别串达到
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
| FlyDSL，以上配置 + DMA绝对地址预计算（历史） | 0.09724 ms | 1413.5 TFLOPS | 0.09719/0.09745 ms |
| FlyDSL，独立stage object + wave-relative GEP（当前） | 0.09756 ms | 1408.8 TFLOPS | 0.09741/0.09761 ms |

当前FlyDSL相比自身无padding基线仍快约10%。核心交织、LDS bank conflict和L2命中率
均已对齐；permlane16 epilogue消除了固定存储开销，wave-relative GEP在保留stage对象
身份的同时避免了热循环重复scalarize。

绝对地址预计算阶段的 `get_pids` K sweep采用相同的预热20次、7组×60次、丢弃前2组方法：

| K | FlyDSL | Gluon | 绝对差值 | 相对差值 |
|---:|---:|---:|---:|---:|
| 1024 | 0.03332 ms | 0.03356 ms | -0.24 us | -0.72% |
| 2048 | 0.05416 ms | 0.05447 ms | -0.31 us | -0.57% |
| 4096 | 0.09724 ms | 0.09663 ms | +0.60 us | +0.62% |
| 8192 | 0.18416 ms | 0.18369 ms | +0.47 us | +0.25% |
| 16384 | 0.35016 ms | 0.35439 ms | -4.24 us | -1.20% |

正差值表示FlyDSL较慢，负差值表示FlyDSL较快。五个K点均在约1.2%以内，已没有旧版本
随K稳定存在的数微秒差距。

### 9.8 DMA地址优化前的PMC定位

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

### 9.9 DMA绝对地址预计算结果（历史阶段）

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

### 9.10 独立stage对象与wave-relative GEP（当前）

绝对地址预计算会把LDS pointer变成`ptrtoint -> readfirstlane -> inttoptr`，对象来源随之
丢失，因此必须依赖scoped alias metadata。当前实现将每个ping-pong stage拆成独立
LDS global，并从global root加显式wave-relative offset构造DMA pointer：

- BF16 A：`wave_id * (8 * TILE_K + 16)`个元素；
- FP8 A：`(wave_id % 2) * (8 * TILE_K + 16) + (wave_id // 2) * (2 * (8 * TILE_K + 16) + 32)`个元素；
- B：`wave_id * wave_stride_bytes`，再叠加chunk offset。

最终4/8-wave、BF16/FP8 IR均有8个`@__shared_alloc_*`，并满足：

```text
ptrtoint/inttoptr = 0
alias.scope/noalias = 0
buffer_load_dwordx4 ... lds 保留
```

`4096^3`、每条路径4次PMC的结果：

| Kernel | `SQ_LDS_BANK_CONFLICT` | `SQ_LDS_DATA_FIFO_FULL` | `SQ_WAVE_CYCLES`中位数 | `SQ_BUSY_CYCLES`中位数 |
|---|---:|---:|---:|---:|
| 4-wave BF16 | 0 | 0 | 41,246,727 | 5,168,714 |
| 4-wave FP8 | 0 | 0 | 24,604,033 | 3,087,710 |
| 8-wave BF16 | 0 | 0 | 81,223,768 | 5,133,184 |
| 8-wave FP8 | 0 | 0 | 45,595,300 | 2,907,140 |

也测试了把AT/AB/BL/BR象限基址分别编码到4个buffer descriptor。该方案将热循环
`v_add_u32` 从36条降到16条、`SQ_INSTS_VALU_INT32`降到579,584，但没有保留：
`K=8192` 的16组ABBA配对测试全部回退，中位慢0.562 us。说明减少地址指令不能替代
依赖链和descriptor切换的实际cycle验证。

指令前端不是独立瓶颈：FlyDSL `SQ_IFETCH` 比 Gluon多 5.3%，但
`SQ_IFETCH_LEVEL/SQ_IFETCH` 分别约为 0.232和 0.230，平均取指等待几乎相同；只是较多
地址指令带来的自然取指增量。

资源压力是次要因素。绝对地址预计算阶段的ISA元数据为：

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

### 9.11 与本地pyhip FP8 8-wave的PMC差异定位（permlane优化前）

以下PMC用于定位旧64-bit epilogue的瓶颈。对FlyDSL 8-wave FP8和
`tests/contrib/gemm/test_fp8_8wave.py`调用的pyhip JIT
`gemm_8wave_fp8bf16fp16`做同输入PMC对比。两边均使用E4M3FN输入、BF16输出、
preshuffled B、`256x256x128` tile、512线程和256个workgroup；分别测试
`M=N=4096, K=4096/16384`。`rocprofv3`每组最多采4个counter，只过滤目标kernel，
预热2次后对4次dispatch取中位数。FlyDSL的`flyc.compile`会额外执行一次kernel，统计时
已按dispatch ID排除该首发样本。

code object和trace资源如下。trace中的VGPR allocation单位为128/100，换算到元数据的
物理VGPR数即256/200；两边都受单个约128 KiB LDS workgroup限制，均为2 waves/SIMD，
因此不是occupancy档位差异。

| 资源/ISA | FlyDSL | pyhip JIT |
|---|---:|---:|
| 物理VGPR | 256 | 200 |
| 实际SGPR计数 | 42 | 38 |
| private segment | 52 B | 0 B |
| LDS | 133120 B | 131072 B |
| MFMA形式 | `v_mfma_scale_f32_16x16x128_f8f6f4` | `v_mfma_f32_16x16x128_f8f6f4` |
| C写回 | 32条`buffer_store_dwordx2` | 16条`buffer_store_dwordx4` |
| scratch指令 | 3条store + 3条load | 0 |

`K=4096`的关键PMC如下。wait和wave counter的单位是硬件定义的quad-cycle；这些counter
标记为nondeterministic，但4次目标dispatch的结果稳定，表中使用中位数。

| Counter | FlyDSL | pyhip JIT | FlyDSL/pyhip |
|---|---:|---:|---:|
| rocprof目标kernel延迟 | 53.821 us | 48.561 us | 1.108 |
| `SQ_WAVE_CYCLES` | 47,808,424 | 40,334,864 | 1.185 |
| `SQ_BUSY_CYCLES` | 3,060,039 | 2,600,556 | 1.177 |
| `SQ_INSTS_MFMA` | 2,097,152 | 2,097,152 | 1.000 |
| `SQ_VALU_MFMA_BUSY_CYCLES` | 67,108,864 | 67,108,864 | 1.000 |
| `SQ_INSTS_LDS` | 1,572,864 | 1,572,864 | 1.000 |
| `SQ_LDS_BANK_CONFLICT` / `SQ_LDS_DATA_FIFO_FULL` | 0 / 0 | 0 / 0 | - |
| `SQ_WAIT_ANY` | 18,752,693 | 11,822,208 | 1.586 |
| `SQ_WAIT_INST_ANY` | 20,013,362 | 20,554,326 | 0.974 |
| `SQ_THREAD_CYCLES_VALU` | 352,851,968 | 230,952,960 | 1.528 |
| `SQ_INSTS` | 8,013,824 | 8,697,856 | 0.921 |
| `SQ_INSTS_VMEM_RD`，每wave | 259 | 270 | 0.959 |
| `SQ_INSTS_VMEM_WR`，每wave | 35 | 16 | 2.188 |
| L2 hit rate | 79.47% | 78.22% | - |
| `TCC_EA0_RDREQ_DRAM_sum` | 836,584 | 829,944 | 1.008 |

MFMA数量和MFMA busy cycle完全相同，LDS指令数相同且无bank conflict/FIFO full，FlyDSL
的L2命中率还略高。FlyDSL动态总指令少7.9%，`SQ_WAIT_INST_ANY`也没有增加，因此差距
不是MFMA吞吐、LDS、DRAM流量或指令前端造成的。最明显的信号是
`SQ_WAIT_ANY`增加58.6%，而不是更多指令等待issue。`SQ_THREAD_CYCLES_VALU`虽然增加
52.8%，但其随K增长的差值几乎恰好等于每条`v_mfma_scale`比普通`v_mfma`多记一个
VALU-accounted cycle；结合完全相同的MFMA busy cycle，它只能作为opcode分类或issue
压力信号，不能单独证明普通VALU算术是瓶颈。

VMEM读写拆分进一步定位了固定尾部。`SQ_INSTS_VMEM_RD/WR`包含scratch：FlyDSL每wave
有256条输入read、3条scratch read、32条C write和3条scratch write；pyhip有270条
输入read、16条C write且无scratch。pyhip虽然多14条输入read，但其16-byte写回和零
spill仍把总VMEM降到286条/wave，FlyDSL为294条/wave。FlyDSL的3条
`scratch_store_dwordx4`位于tail MFMA之间，稍后3条`scratch_load_dwordx4`之后又出现
`vmcnt(2/1/0)`再消费最后3组MFMA accumulator，直接拉长了固定critical path。

长K用于区分固定开销与主循环斜率：

| 指标 | `K=4096`比值 | `K=16384`比值 | 增加96个K tile的比值 |
|---|---:|---:|---:|
| 目标kernel延迟 | 1.108 | 1.037 | 1.007 |
| `SQ_WAVE_CYCLES` | 1.185 | 1.062 | 1.014 |
| `SQ_WAIT_ANY` | 1.586 | 1.287 | 1.168 |
| `SQ_INSTS_MFMA` | 1.000 | 1.000 | 1.000 |
| `SQ_INSTS_LDS` | 1.000 | 1.000 | 1.000 |
| `SQ_INSTS_VMEM` | 1.028 | 1.010 | 1.003 |
| `TCC_EA0_RDREQ_DRAM_sum` | 1.008 | 1.004 | 1.003 |

按两个K点对`SQ_WAVE_CYCLES`做线性分解，`K=4096`时两边相差7,473,560；其中按
主循环斜率推算的32个K tile差值约468,643，剩余固定差值约7,004,917，占**93.7%**。
kernel延迟做同样分解得到约5.0 us固定差值。主循环并非完全对齐：每增加一个K tile，
FlyDSL约为531.83 wave quad-cycles，pyhip约为524.68，FlyDSL仍慢1.36%；但当前短K
约11%的差距主要来自prologue/tail/epilogue，而不是热循环MFMA吞吐。

pyhip将编译期K循环完全展开，并手工固定每阶段的DSRD/VMEM/wait顺序；FlyDSL保留
运行时循环并依赖LLVM和memory-only scheduler排布。FlyDSL使用scale全为1的
`v_mfma_scale`，pyhip使用普通`v_mfma`，但MFMA busy cycle完全一致，所以现有PMC不
支持“scaled MFMA计算吞吐更低”这一结论；它更可能通过寄存器生命周期和周边调度间接
影响资源。主循环残余1.36%应在固定尾部修复后再单独优化。

按照PMC优先级，8-wave已改用4-wave验证过的`permlane16.swap`打包和16-byte store。
8-wave的wave layout是`4M x 2N`，因此C fragment按`[value, N/128, M/64]`解释；每两组
N repeat打包成4个i32后发出一条`buffer_store_dwordx4`。`TILE_N`不能成对打包时仍回退
到原64-bit copy路径。

最终FP8 ISA中C写回从32条`buffer_store_dwordx2`降为16条
`buffer_store_dwordx4`，新增32条`v_permlane16_swap_b32`。private segment从52 B降为
24 B，scratch从3 store + 3 load降为2 + 2；因此写回已显著缩短accumulator live range，
但尚未完全消除spill。同进程交替A/B、丢弃前2轮后的结果为：

| 形状/类型 | 64-bit fallback | permlane + 128-bit | 配对中位变化 |
|---|---:|---:|---:|
| FP8 `4096x4096x4096` | 50.926 us | 47.832 us | -2.977 us（-6.1%） |
| BF16 `4096x4096x4096` | 101.172 us | 100.206 us | -0.894 us（-1.0%） |
| FP8 `4096x4096x16384` | 166.550 us | 163.852 us | -2.711 us（-1.6%） |

新增`test_gemm_950_8wave_permlane_multiblock`以`256x256` tile、2x2 workgroups和NaN预填
覆盖BF16/FP8新路径。后续优先级变为：先定位剩余24 B tail spill，再为gfx950补充普通
`v_mfma_f32_16x16x128_f8f6f4` atom做A/B，最后测试K循环部分展开和DSRD/VMEM/wait
重排。旧PMC显示每K tile约1.36%的`SQ_WAVE_CYCLES`斜率差，但当前wall-time拟合只差
0.3%；继续优化热循环前应先用permlane版本重新采集PMC确认。

## 10. 正确性与汇编验证

参数化smoke测试覆盖`128x128`单block下的4/8-wave x BF16/FP8；新增多block测试
覆盖4-wave的4种tile x BF16/FP8，并使用四个K tile执行主循环和tail：

```bash
pytest -q \
  tests/flydsl/test_gemm.py::test_gemm_950_correctness \
  tests/flydsl/test_gemm.py::test_gemm_950_4wave_multiblock \
  tests/flydsl/test_gemm.py::test_gemm_950_8wave_permlane_multiblock
```

gfx950上期望结果为`14 passed`。多block测试以NaN预填输出，既检查数值误差也能捕获
未写区域；它覆盖4-wave矩形mode交换和短tile DMA竞态，以及8-wave permlane16 +
128-bit epilogue的`256x256` BF16/FP8路径。

当前全尺寸支持矩阵：

| Waves | Tile | BF16 | FP8 | 结论 |
|---:|---:|---|---|---|
| 4 | `128x128` | 通过 | 通过 | 支持 |
| 4 | `128x256` | 通过 | 通过 | 支持 |
| 4 | `256x128` | 通过 | 通过 | 支持 |
| 4 | `256x256` | 通过 | 通过 | 当前最优4-wave路径 |
| 8 | `128x128` | 通过 | 通过 | 支持 |
| 8 | `128x256` | 通过 | 通过 | 支持 |
| 8 | `256x128` | 通过 | 通过 | 支持 |
| 8 | `256x256` | 通过 | 通过 | 当前最优8-wave路径 |

除2x2 workgroup回归外，4-wave和8-wave的4种tile x BF16/FP8都在`4096^3`上用
NaN预填C做过全输出检查，全部没有未写元素或容差外结果。第9.6节记录4-wave修复前的
具体失败模式、根因和性能回归数据。

标准`256x256`路径的tail-only和长K检查覆盖：

- BF16 `K=128/8192`，4-wave和8-wave均通过；
- FP8 `K=256/8192`，4-wave和8-wave均通过；
- 4-wave四种tile的BF16/FP8多block`4096^3`均通过；
- 8-wave四种tile的BF16/FP8多block`4096^3`均通过。

第9节所有K sweep、tile sweep和problem-size sweep性能点都遵循相同准入规则：先用
NaN预填C，运行一次当前kernel，要求没有未写元素且全部输出满足BF16/FP8对应容差，
再进行预热和轮换计时。PID映射对照中的标准`256x256`路径也分别验证了row-major和
`get_pids`结果。

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
- FlyDSL源码的两次主循环wait位于Region 2/6的LDS read之前；Gluon参考源码仍在
  Region 3/7等待，二者分别报告而不再强求源码位置相同；
- 4-wave参考的每个子块恰好包含对应来源的32 MFMA、8 LDS read和4 GMem-to-LDS
  prefetch；
- 4-wave八个子块的352个`MFMA/DSRD/VMEM`类别位置逐位一致；
- FlyDSL主循环的`vmcnt`为`[8,8]`、位置为`89/265`；Gluon参考为`[16,16]`、
  位置为`168/344`；两边barrier都为2，AGPR搬运都为`0/0`；
- tail 逻辑等待顺序为 `5,4,3,2,1,0`。
