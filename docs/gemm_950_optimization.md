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
- 基准默认预热 5 次，正式数据另外使用了 3 组、每组 60 次运行取中位数。

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

`M=N=K=4096`、`256 x 256 x 64`、256 workgroups 的 PMC 结果：

| 配置 | `SQ_LDS_BANK_CONFLICT` | 每 workgroup | LDS |
|---|---:|---:|---:|
| 无 padding/连续 B | 25,165,824 | 98,304 | 131,072 B |
| A/B 共用简单 padding view | 8,388,608 | 32,768 | 135,168 B |
| 仅修正 A 的双 view | 4,194,304 | 16,384 | 135,168 B |
| A 双 view + B 连续布局 | 0 | 0 | 133,120 B |

参考 `asycn_copy_padding` 在相同访问负载下从 `3,670,016` 降到 `0`，证明这套 A
参数本身能完全移除 bank conflict。

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

`sched_barrier` 只是编译器调度 fence，不是 workgroup 同步。8-wave kernel 每个 MFMA
区域之后仍需要 `gpu.barrier()`，与 Gluon 的 8-wave 同步结构一致。遗漏该 barrier 时，
只跑两个 K tile 的 tail 测试可能通过，但进入展开主循环后会产生错误结果。

## 8. Epilogue

累加器使用 FlyDSL 标准转换：

```python
c_frag_bf16.store(c_frag.load().to(fx.BFloat16))
```

它替代手工加 `0x8000`、移位和截断的实现，使用正常浮点转换语义。

## 9. 性能结果

同一 `benchmark_gemm_950` 配置的前后结果：

| Kernel | 初始延迟 | 初始性能 | 当前延迟 | 当前性能 | 总加速 |
|---|---:|---:|---:|---:|---:|
| 4-wave BF16 | 0.2861 ms | 480.4 TFLOPS | 0.10083 ms | 1363.1 TFLOPS | 2.84x |
| 8-wave BF16 | 0.2366 ms | 581.0 TFLOPS | 0.1865 ms | 737.1 TFLOPS | 1.27x |
| 4-wave FP8 | 0.1500 ms | 916.2 TFLOPS | 0.0754 ms | 1822.4 TFLOPS | 1.99x |
| 8-wave FP8 | 0.1269 ms | 1083.2 TFLOPS | 0.0951 ms | 1445.2 TFLOPS | 1.33x |

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

同一组 `4096^3` BF16 输入、4-wave、`256x256` tile，预热 20 次，7 轮、每轮 100 次：

| 实现 | 中位延迟 | 性能 | Q1/Q3 |
|---|---:|---:|---:|
| Gluon | 0.09521 ms | 1443.5 TFLOPS | - |
| FlyDSL，无 padding | 0.10890 ms | 1262.0 TFLOPS | - |
| FlyDSL，A padding + B 连续 | 0.10083 ms | 1363.1 TFLOPS | - |

修正 LDS 地址后，FlyDSL 相比自身无 padding 基线快 8.0%，与 Gluon 的延迟差距从
约 14% 缩小到 5.9%。核心交织和 LDS bank conflict 都已经对齐，剩余差异来自额外的
地址计算、普通 VALU 和寄存器压力。

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
