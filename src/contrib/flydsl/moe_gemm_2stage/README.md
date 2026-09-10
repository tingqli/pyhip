# FlyDSL MoE down `8x1` 设计

## 当前支持范围（2026-09-08）

- `down_path="8x1"`和`"8x1_compact"`仅支持 **K=192、256、320、384、512、640**。
- K=128及其rolling/pure实现、独立Nloop和专用VMEM账本已移除；请求该K会明确报错，不静默切换算法。
- **BK128分块仍保留**：K256/384/512/640使用BK128，K320使用128+192，K192使用整192。
- 其他down算法的K128支持、历史源码快照、测试数据和ATT UI不受此删除影响。

下文是历史设计与实验记录；其中旧K128支持和BK64方案不代表当前接口。

本 README 摘录自[根设计文档](../../../../../../../../../../../design_moe_gemm_8wave_down.md)第0节；第1–22节旧 JIT 设计仍在原文。
原章节号、日期和实验数据保留，下文“后文第1–22节”指原文。
这是设计/实现演进记录，不自动等于本目录 `matrix_smoke` 源码快照或工作区后续 K192/K320 候选的当前状态。

## 0. FlyDSL `8x1` / BN128 双阶段实验设计（2026-09-04）

本节定义下一版 FlyDSL down kernel 的实验规范。后文第1--22节记录旧 JIT
`moe_gemm_8wave_down` 的 BN64、四级 LDS ring 设计，只作为历史参考，不是新实现的
资源或流水约束。新实现计划独立为 `moe_gemm_2stage/gemm2_8x1.py`，先支持 FP8 PTPC
和K256，不修改当前生产`4x1`。

参考实现：

- `src/contrib/flydsl/moe_gemm_2stage/gemm2_2x4.py`：gfx942上的512-thread启动、
  conditional-displacement barrier、`s_setprio`和raw buffer store；
- `/opt/aiter/csrc/include/fmha_fwd_hd192_v128_bf16_opus_kernel.hpp`：两个4-wave组
  相差一个barrier代次的严格错相；只借鉴barrier代次，不移植gfx942不支持的
  `sched_group_barrier`调度提示；
- `tests/flydsl/attn_4wave/tools/batched-gemm-core-ceiling.md`：co-issue上界和4-wave
  cooperative-B-load结果。该探针不含真实LDS、依赖、scale和epilogue，只能作为上界。

### 0.1 固定几何与首版范围

```text
WG threads       = 512 = 8 waves
WG tile          = BM256 x N
inner N tile     = BN128
inner K tile     = BK64 or BK128
wave topology    = 8x1
wave output      = M32 x N128
dtype            = A/B FP8 E4M3FNUZ, C BF16
quantization     = PTPC
first case       = K256
```

8个wave只沿M分工；每个wave持有`A[32, K]`，遍历完整N。一个WG沿N每次处理128列，
在K内每次处理128点：

```text
for n0 in range(0, N, 128):
    for k0 in range(0, K, 128):
        C[0:256, n0:n0+128] +=
            A[0:256, k0:k0+128] @ B[n0:n0+128, k0:k0+128].T
```

K128/256/384/512使用BK128，分别有1/2/3/4个K core；K192使用3个BK64 core，避免
补零执行。Qwen35的N2048在K256时对应16个N tile、32个K core；Qwen397的N4096
对应32个N tile、64个K core。A只在prologue gather一次，随后跨完整N循环常驻寄存器。

当时范围（历史）：

- K128、K192、K256、K384和K512均已支持；
- K192固定使用BK64，其余K固定使用BK128；
- K256保留rolling epilogue优化快路径；其他K使用同一反相B流水和pure epilogue；
- BF16：见LDS分析，它不能直接复用首版双BK128 ping-pong；
- 自动selector、非MI308X映射和低Batch边界均等K256原型通过后再处理。

### 0.2 每wave工作量

对`BM256 x BN128 x BK128`：

```text
wave_M x wave_N                  = 32 x 128
MFMA / wave / BK128              = (32/16)*(128/16)*(128/32) = 64
A bytes / wave / full K256       = 32*256 = 8192B = 32 VGPR
B bytes / WG / BK128             = 128*128 = 16384B
B global bytes / wave / BK128    = 16384/8 = 2048B
B buffer_load_dwordx4 / wave     = 2
B ds_read_b128 / wave / BK128    = 16
B operand / wave / full N128     = 64 VGPR
C FP32 accumulator / wave        = 32*128/64 = 64 VGPR
C ds_write_b128 / wave / N tile  = 8
C ds_read_b128 / wave / N tile   = 8
C buffer_store_dwordx4 / wave    = 8
```

每wave输出仍为4096点，与当前BM256/BN64 `4x1`的`M64 x N64`相同；每个BK128也仍为
64条MFMA。BN翻倍后B operand从32增至64 VGPR，但A的M条带减半，A从64降至32 VGPR。
因此主要数据寄存器总量近似守恒，这是8x1有机会保持在256 VGPR以内的原因。

完整N上，8x1的N tile数和K core数相对BN64 4x1减半，因而barrier代次数也近似减半；
代价是同一个B元素需广播给8个M-wave而不是4个，单wave每BK128的B LDS读取由8条增至
16条，WG总B-LDS读流量约为4x1的两倍。反相主要是在用更高LDS读压力换更少barrier和
更稳定的MFMA/non-MFMA互补，不能只按全局B读取量判断收益。

### 0.3 LDS账本

B ping/pong必须是两个独立声明，不能从一个`2*BN*BK`数组切片：

```text
b_ping : FP8[128*128], align 16 = 16384B
b_pong : FP8[128*128], align 16 = 16384B
```

CShuffle按一个正在执行memory stage的4-wave半组分配。每wave一次只处理一个16-row
row-pair，需要`16*128*2 = 4096B`；4个memory wave共用：

```text
cshuffle_scratch = 4*16*128*2 = 16384B
```

两个半组严格反相，不会同时进入CShuffle。前一半组必须在stage barrier前完成全部
LDS read，后一半组才能复用同一16KB scratch。`sorted_ids[256]`和
`sorted_weights[256]`共2KB，只在prologue存活；建议把它们作为
`cshuffle_scratch`的临时视图，A和row scale进入寄存器并经过WG barrier后再将该区域
改作CShuffle。weight scale直接从global读入寄存器，不占LDS。

| LDS对象 | 字节 |
| --- | ---: |
| `b_ping` | 16,384 |
| `b_pong` | 16,384 |
| metadata/CShuffle复用scratch | 16,384 |
| **峰值** | **49,152 (48KiB)** |

即使metadata暂时独立分配，总量也只有51,200B；但首版仍应显式复用，以保留调试余量。
48KiB在gfx942的64KiB/WG上限内，只允许每CU驻留一个8-wave WG，即每个
SIMD约2 waves（Q2）。

无错相/普通pipeline对照不能复用上述16KB半组scratch，因为8个wave可能同时进入
CShuffle；该对照应为两个4-wave group各声明16KB，共32KB。它与两个B槽合计正好
64KB，metadata仍须在prologue后与CShuffle别名，且allocator不能再产生额外padding。

BF16的一个B tile为32KB，两个独立ping/pong已经占满64KB，无法再容纳CShuffle。
因此BF16不是简单替换dtype：后续必须改为BK64、单B槽、BN64或不经LDS的输出重排之一。

### 0.4 VGPR上界

gfx942按combined VGPR计算occupancy。8-wave WG需要约2 waves/SIMD，单wave硬上限为
256 combined VGPR；实现目标应为`<=240`，`241--248`只能进入性能试验，`>256`直接
判失败。

K256稳态的显式活跃值预算：

| 对象 | 32-bit VGPR/wave | 约束 |
| --- | ---: | --- |
| 完整A `32x256` | 32 | 跨完整N循环存活 |
| 当前C `32x128` FP32 | 64 | 不允许双C tile |
| 当前BK128的B operand | 64 | 16条`ds_read_b128`完整读取N128 |
| 下一B半片global staging | 8 | 两条dwordx4；只保留一套staging |
| row id、routing/A scale | 4--8 | 每次只保留两个M atom所需值 |
| weight scale | 2--4 | 8个N atom逐个消费，禁止C-shape fragment |
| C pack、vector offset、临时值 | 12--20 | row-pair流式复用 |
| **显式小计** | **186--200** | 不含LLVM分配碎片和循环状态 |

以已编译的N4096/BM128 `4x1`（164 VGPR）为锚点，BN64扩到BN128会分别增加约32个C
和32个B operand寄存器，得到约228 VGPR；以当前BM256 `4x1`的240 VGPR为另一锚点，
8x1的A减32而B增32，仍约240。故首版合理预测为`228--248 VGPR`，必须以fresh ISA和
driver occupancy为准。

以下任一情况都会大概率越过256：

- 保留上一N tile的完整C以跨tile延迟epilogue：`+64`；
- 同时保留两套完整B LDS fragment：`+64`；
- 用`make_fragment_C`构造完整PTPC weight-scale fragment：最坏`+64`；
- 一次物化完整BF16 packed-C而不是按row-pair流式复用：约`+24`以上。

因此首版禁止双C、双B-reg fragment和完整scale fragment。若2-stage需要这些对象才能
平衡时序，应改成更多细粒度stage，而不是接受spill。最终门禁为0 private/scratch。

SGPR预计接近现有kernel的96--112，不是Q2限制项，但N/K/wave级基址应全部保持标量，
避免为了地址计算把它们重新物化为VGPR。

### 0.5 B的8-wave协作与两槽依赖

严格反相后，同一物理时刻只有4个wave处于memory stage，所以一个`B[128,128]`不能在
单个物理memory phase内由8个wave完整产生。首版必须把每个B tile拆成两个8KB半片：

```text
wave 0..3 在自己的 memory(t) 阶段写 B[t+1].half0
wave 4..7 在下一物理阶段的 memory(t) 写 B[t+1].half1
```

两半跨相邻两个物理阶段完成。B至少提前一个逻辑K core开始预取；`B[t+1]`完成后，
group0和group1分别在各自的`memory(t+1)`读取它。ping/pong足够，因为写
`B[t+1]`时，两组都已把同奇偶的`B[t-1]`读入寄存器。

为避免混淆，后文使用四个不同动作：

```text
P(t, g): global B[t].half[g] -> 该wave的8个staging VGPR（真正的预取，不等待）
W(t, g): 等待P(t,g)后，把staging VGPR以2条ds_write_b128提交到Bslot(t)
R(t):    从已完整的Bslot(t)发出16条ds_read_b128 -> 64个B operand VGPR
C(t):    只执行使用Areg和Breg的64条MFMA
```

后文`M(t)`与`S0(t)`同义，表示memory stage；`C(t)`与`S1(t)`同义，表示纯MFMA
compute stage。

K/N边界统一使用扁平core编号；因此“最后一个K预取下一N的K0”不是特殊分支：

```text
core(n, k) = n * K_TILES + k
next_core(n, k):
    if k + 1 < K_TILES: return (n,     k + 1)
    else:               return (n + 1, 0)
```

prologue由全部8个wave共同执行`P(0, all) -> W(0, all)`，完整填好B0。进入错相区后，
每个memory stage一开始先发`P(t+1, group)`，再执行`R(t)`、前一N的epilogue等独立工作，
最后才等待并执行`W(t+1, group)`。因此global-load latency位于整个memory stage背后，
不是“读完当前B以后才同步加载下一B”。

前四个barrier代次如下，其中`M(t)`包含`P(t+1,g) + R(t) + W(t+1,g)`：

| barrier代次前的物理阶段 | group0（wave0--3） | group1（wave4--7） | 下一B状态 |
| --- | --- | --- | --- |
| prologue | `P(0,all), W(0,all)` | `P(0,all), W(0,all)` | B0完整 |
| 0 | `M(0): P(1,0), R(0), W(1,0)` | 额外stagger barrier，阻塞 | B1 half0完成 |
| 1 | `C(0)` | `M(0): P(1,1), R(0), W(1,1)` | B1完整 |
| 2 | `M(1): P(2,0), R(1), W(2,0)` | `C(0)` | B2 half0完成 |
| 3 | `C(1)` | `M(1): P(2,1), R(1), W(2,1)` | B2完整 |

所以group0进入`M(1)`并执行`R(1)`前，group1已经完成`W(1,1)`；同理后续每个slot在
消费前都已由两个group写完整。这个代次关系必须用最终ISA和正确性压力测试验证，不能
仅由源码中的barrier数量推断。

如果要求每个B tile在一个memory phase内完成，则只能让当前4-wave半组重复加载完整
16KB；这会把每waveB读取从2KB增至4KB，已不再是“8-wave协作加载”，不作为首选。

### 0.6 gfx942地址生成

核心循环使用`fx.buffer_ops.create_buffer_resource`和raw buffer load/store。FlyDSL当前
API支持：

```text
fx.rocdl.readfirstlane(i32, value)
fx.buffer_ops.buffer_load(..., soffset_bytes=scalar_offset)
fx.buffer_ops.buffer_store(..., soffset_bytes=scalar_offset,
                           offset_is_bytes=True)
```

建议把`expert/N/K/wave/load-round`共同决定的字节基址通过`readfirstlane`固定到SGPR，
每lane只保留一个由`lane_id`决定的32-bit vector offset。B的两个load round复用同一组
8个payload VGPR。output按row-pair现算一个row vector offset，N tile基址走
`soffset_bytes`；不要让所有行地址跨核心循环存活。

旧JIT使用`buffer_load_dwordx4 ... lds`直接写LDS；当前FlyDSL首版应保守按
`buffer_load_dwordx4 -> 8 VGPR payload -> UniversalCopy128b -> LDS`建模。只有实际
lowering和ISA证明支持安全的direct-to-LDS，才能把这8个staging VGPR从预算中删除。

fresh ISA门禁：

- N/K主循环内不应重复出现大整数乘法或64-bit vector地址构造；
- B路径目标是每wave每BK128两条`buffer_load_dwordx4`，随后两条
    `ds_write_b128`；
- output为每wave每N tile八条`buffer_store_dwordx4 nt`；
- 只允许prologue中的A gather保留必要的per-row vector地址；
- 若raw buffer路径没有减少VGPR或反而增加SALU/wait，应保留`fx.copy`版本作对照，
  不能只凭源码形式判断。

### 0.7 严格反相流水

定义扁平core编号：

```text
t = n_tile * K_TILES + k_tile
Bslot(t) = b_ping if t is even else b_pong
group = wave_id // 4        # group0: wave0..3, group1: wave4..7
```

每个group执行相同的`memory(t) -> compute(t)`代码。group1在prologue多执行一个
`stage_end()`，group0在epilogue补一个，令两个group始终相差一个barrier代次：

```text
stage_end():
    sched_barrier(0)
    s_barrier()
    sched_barrier(0)
```

`wave_id/group`必须先经过`readfirstlane`，确保条件分支是scalar branch。进入错相区后
禁止只有部分wave提前退出；两组执行的barrier总数必须完全相同。

#### Stage目录

与Opus参考的写法一致，这里把prologue、稳态stage和drain逐项列出。`setprio`和
`stage_end()`视为stage边界控制，不计入payload；`S1`两次边界之间只允许MFMA。

**P0：metadata与首块weight prologue（8 waves同步）**

```text
1. 映射task并读取expert id；所有WG-uniform退出都在错相前完成。
2. sorted_ids/sorted_weights -> scratch；WG barrier。
3. P_full_B(0)：全部8个wave先发B(n0,k0)的两条buffer_load_dwordx4。
4. gather每wave的A[32,K]、routing weight和activation scale到VGPR。
5. partial vmcnt：只等待较老的B0，允许较新的A/scale继续在飞。
6. W_full_B(0)：两条ds_write_b128/wave -> b_ping。
7. 等待A/scale；WG barrier。此时B0完整可见，scratch可改作CShuffle。
```

**P1：错相启动**

```text
group0: 直接进入S0(n0,k0)
group1: 额外执行一次stage_end()，阻塞一个barrier代次
```

**S0：memory stage，禁止MFMA**

共同顺序如下，必须把下一块weight预取放在stage最前面：

```text
1. P(next,g)：预取B[next].half[g] -> 8 staging VGPR，不等待。
2. 若当前是K0且n>0，发出上一N tile的PTPC weight-scale load。
3. R(cur)：16条ds_read_b128，从完整Bslot(cur)读取当前B -> 64 Breg。
4. 若当前是K0且n>0，写回本group上一N tile的128行：
      Creg * weight_scale * row_scale
      -> BF16 pack
      -> 8条ds_write_b128
      -> 8条ds_read_b128
      -> 8条buffer_store_dwordx4 nt
5. 等待步骤1的两个较老B load，保留更年轻的output store在飞。
6. W(next,g)：两条ds_write_b128，把B[next].half[g]提交到下一ping/pong槽。
7. 等待R(cur)和W(next,g)所需的lgkmcnt；stage_end()。
```

K256下，S0在两个K位置的具体职责不同：

| S0位置 | 下一weight预取 | output epilogue | 当前B读取 |
| --- | --- | --- | --- |
| `S0(n,k0,g)` | `P(B[n,k1].half[g])` | 写回`D[n-1]`，`n=0`时跳过 | `R(B[n,k0])` |
| `S0(n,k1,g)` | `P(B[n+1,k0].half[g])` | 无；当前N尚未完成 | `R(B[n,k1])` |

因此K1末尾显式预取的是下一N的K0，而不是等到下一轮N循环才开始读取weight。

**S1：compute stage，payload只包含MFMA**

```text
S1(n,k0): 64条MFMA，C输入使用0，开始新的C[n]
S1(n,k1): 64条MFMA，C输入使用Creg，完成C[n]
```

这里定义的是基线`S1P`。基线S1内禁止B/A/scale load、LDS读写、global store、地址计算、BF16转换和显式清零。
K0通过MFMA的`c=0`覆盖旧Creg。实现时可在进入S1后的边界处设置`setprio(3)`，在离开
S1的边界处恢复`setprio(0)`；两者不能漂入64条MFMA序列中间。后文可选`S1E`只放宽为
MFMA加独立scalar FMA/permute，仍禁止DS/VMEM和global store。

K256稳态stage的每wave静态主体为：

| Stage | B VMEM | B LDS | Epilogue | MFMA |
| --- | ---: | ---: | ---: | ---: |
| 首个`S0(n0,k0)` | 2 load | 16 read + 2 write | 无 | 0 |
| 轻`S0(n,k1)` | 2 load | 16 read + 2 write | 无 | 0 |
| 重`S0(n>0,k0)` | 2 load | 16 B-read + 2 B-write + 8 C-write + 8 C-read | scale/pack + 8 store | 0 |
| `S1(n,k0/k1)` | 0 | 0 | 0 | 64 |

表中B VMEM和B LDS write属于下一core，B LDS read属于当前core；这正是预取与消费的
跨stage关系。

#### VALU集中位置

现有Qwen35 K256/BM256 `4x1` final ATT的13,417条指令中，排除MFMA后共有6,244条
`v_*` VALU。按opcode语义分类如下；内联helper的debug location大量归到同一行，因此
这里不使用源码行号分类：

| VALU类别 | 静态指令数 | 占非MFMA VALU | 主要opcode |
| --- | ---: | ---: | --- |
| output scale与BF16 pack | 5,124 | 82.06% | 2,048 `v_fma_f32`、2,047 `v_fmaak_f32`、1,024 `v_perm_b32`及少量mul/fmac |
| 地址、索引与lane变换 | 593 | 9.50% | `v_add_lshl`、`v_lshl_or`、`v_and/or`、`v_mad` |
| move/materialization | 519 | 8.31% | `v_mov_b32` |
| 其他 | 8 | 0.13% | 少量杂项 |

当前4x1的N2048按BN64展开32个N tile，所以上述主epilogue平均每tile、每wave约为：

```text
64  x weight-scale FMA
64  x row-scale + BF16-bias FMA
32  x v_perm_b32
----------------------------
约160条主要VALU / N tile / wave
```

8x1每wave输出`M32 x N128 = 4096`点，与4x1的`M64 x N64 = 4096`点相同，因此每个
完成N128 tile的每wave epilogue仍预计约160条主要VALU，不会因BN翻倍而翻倍。它们的
stage归属为：

| Stage | VALU密度 | 主要VALU |
| --- | --- | --- |
| `P0` | 低，一次性 | sorted-id解码、A gather地址、routing与activation scale合并 |
| 首个`S0(n0,k0)` | 很低 | 少量lane/vector offset；主要是VMEM/LDS |
| 轻`S0(n,k1)` | 很低 | 下一N/K0的weight地址；目标是由SGPR `soffset`承担 |
| 重`S0(n>0,k0)` | **最高** | 上一N的约128条scale/FMA、32条BF16 pack/permute，以及少量output/CShuffle地址 |
| `S1(n,k0/k1)` | **0条普通VALU** | 只允许64条MFMA；MFMA按独立issue类别统计 |
| `D0` | **最高** | 最后N tile的同一套约160条epilogue VALU |
| `D1` | 0 | 只有barrier代次补偿 |

weight预取本体是两条`buffer_load_dwordx4`和两条`ds_write_b128`，不是VALU。若核心循环
仍出现大量地址VALU，说明`readfirstlane + raw-buffer soffset`没有成功把N/K/wave基址
标量化。允许保留少量`v_readfirstlane_b32`完成VGPR到SGPR的转换，但不应在每个N atom
或每个store重复执行。

在物理时序中，主要VALU出现在：

```text
T4: group0 heavy S0 = epilogue VALU + CShuffle/store
    || group1 S1 = 64 MFMA
T5: group0 S1 = 64 MFMA
    || group1 heavy S0 = epilogue VALU + CShuffle/store

T8/T9、T12/T13……重复相同模式；D0只在尾部出现一次。
```

因此反相设计真正要遮盖的是重`S0`中的epilogue VALU，而不是两条weight load。ATT验收
时应分别统计`VALU issue`、`VALU dependency stall`及其与peer MFMA的重叠；只看到
MFMA union提高不足以证明约160条epilogue VALU已经被隐藏。

#### 可选实验：把epilogue滚入后续MFMA

可以把epilogue VALU延迟到后续MFMA序列中尝试同wave co-issue，但不能保留“上一N的
完整C + 下一N的完整C”两个fragment；这会额外增加64 VGPR，使预计`228--248 VGPR`
直接越过256。正确做法是按C fragment小单元滚动退休和复用。

一个wave的`M32 x N128` C包含`2 M atom x 8 N atom`，共16个4-FP32 fragment。
把“同一M atom、相邻两个N atom”定义为一个retire record：

```text
records / N tile / wave       = 2 * (8/2) = 8
C VGPR / record               = 2 fragments * 4 = 8
MFMA / record / BK128         = 2 N atoms * 4 K atoms = 8
epilogue VALU / record        = 8 weight-scale FMA
                              + 8 row-scale/BF16-bias FMA
                              + 4 v_perm
                              = 20
CShuffle/memory / record      = 1 ds_write_b128
                              + 1 ds_read_b128
                              + 1 buffer_store_dwordx4 nt
```

把同一N-pair在两个M atom上的record组成一个super-record：

```text
super-record / N tile / wave  = 4
C VGPR / super-record         = 16
MFMA / super-record / BK128   = 16
epilogue VALU / super-record  = 40
40 / 16                       = 2.5 VALU/MFMA
```

super-record包含4个独立C accumulator，每个accumulator在BK128内更新4次。MFMA按
`K atom -> 4个C fragment`排列，同一accumulator两次写之间相隔4条MFMA，与正式
co-issue微基准的四条独立accumulator链接近。

因此“每条MFMA后最多3条scalar VALU”在静态数量上可以容纳40条epilogue VALU；但
3条是容量上限，不是要求每次填满。CShuffle的DS读写、global store、wait和scale读取
不属于这3条scalar VALU，必须放在packet边界或S0，不能冒充免费co-issue。

gfx942正式微基准为该比例提供了机制证据：以16-cycle BF16 MFMA为anchor，独立的
scalar `v_fma_f32`和`v_perm_b32`在同wave及peer-wave两种模式下均可fully hide 3条；
第4条会增加约4 cycles。普通scalar VALU约4 cycles，MFMA开始约4 cycles后可进入VALU
pipeline，剩余shadow约12 cycles，恰好容纳3条。packed `v_pk_mul_f32/v_pk_add_f32`
容量为0，因此本方案必须继续使用scalar FMA和`v_perm_b32`，不能为减少指令数改回packed
FP32。

该正式微基准的anchor是`v_mfma_f32_16x16x16_bf16`，8x1使用
`v_mfma_f32_16x16x32_fp8_fp8`。两者在当前设计中均按16-cycle执行窗建模，但FP8组合仍
应先用同一微基准补测`N=0..4`；在FP8实测完成前，“3条可完全隐藏”是强假设而非生产
保证。

##### 推荐V1：同N K1内滚动，隐藏75% scalar VALU

当前N的C在最终K stage中按super-record逐步完成。V1在计算`SR(r+1)`时退休`SR(r)`：

```text
S0(n,k1):
    prefetch/commit B(n+1,k0)
    read B(n,k1)
    fill scale_lds[group][N128]            # 512B/group
    preload scale for SR0

S1E(n,k1):
    16 final-K MFMA -> SR0                 # 无旧SR可退休
    16 final-K MFMA -> SR1 || scalar_epilogue(SR0) -> packed SR0
    16 final-K MFMA -> SR2 || scalar_epilogue(SR1) -> packed SR1
    16 final-K MFMA -> SR3 || scalar_epilogue(SR2) -> packed SR2
    # packed SR0..2占24 VGPR；SR3仍为16个FP32，总C-family从64降到40

S0(n+1,k0):
    scalar_epilogue(SR3) -> packed SR3     # 最先执行，遮盖peer K1开头的纯MFMA
    P(B[n+1,k1], group)
    R(B[n+1,k0]) -> Breg
    for packed SR0..3:
        ds_write -> ds_read -> 2 x dwordx4 NT store
    # 旧C全部死亡，64个C槽可由下一N的K0 MFMA以c=0覆盖
    wait/commit B[n+1,k1].half[group]

S1P(n+1,k0):
    64 pure K0 MFMA
```

V1每个N tile将`3 * 40 = 120`条scalar epilogue VALU滚入K1的48条carrier MFMA，
平均2.5条/MFMA；剩余SR3的40条VALU和全部CShuffle/store仍在下一S0，由peer group的
MFMA做inter-wave遮盖。这样不会把global store插入MFMA train，也不需要双C。

之所以首版只隐藏75%而不是把160条全部塞进K1，是因为SR0的最终结果必须先由前16条
K1 MFMA产生；只有后48条MFMA可安全作为完整super-record carrier。其scalar容量为
`48 * 3 = 144`，覆盖120条后尚余24 slot，但不足以再容纳完整SR3的40条。更细的
fragment级滚动可能利用SR0内部后段MFMA，但RAW和寄存器复用更难，留作V2。

packed SR必须就地占用已经死亡的FP32 C槽；三个packed SR共24 VGPR，加仍为FP32的SR3
16 VGPR，低于原64 C VGPR。若LLVM另分配24个VGPR而不复用死槽，候选可能超过256，
应先通过缩短live range/inline asm约束修复，不能接受spill。

##### 每16条MFMA的40-VALU模板

旧SR包含4个C fragment。设其最终MFMA按fragment 0..3结束；新carrier SR的MFMA记为
`F0..F15`。假设FP8 MFMA accumulator到scalar FMA需要至少约16 cycles，frag0..3可
分别在`F0..F3`之后达到四条MFMA的依赖距离。一个依赖感知模板为：

| carrier MFMA | 随后的最多3条scalar VALU |
| ---: | --- |
| `F0` | `W(f0,0..2)` |
| `F1` | `W(f0,3), W(f1,0..1)` |
| `F2` | `W(f1,2..3), W(f2,0)` |
| `F3` | `W(f2,1..3)` |
| `F4` | `W(f3,0..2)` |
| `F5` | `W(f3,3), R(f0,0..1)` |
| `F6` | `R(f0,2..3), R(f1,0)` |
| `F7` | `R(f1,1..3)` |
| `F8` | `R(f2,0..2)` |
| `F9` | `R(f2,3), R(f3,0..1)` |
| `F10` | `R(f3,2..3), P0` |
| `F11` | `P1, P2, P3` |
| `F12` | `P4, P5, P6` |
| `F13` | `P7, spare, spare` |
| `F14` | `spare, spare, spare` |
| `F15` | `spare, spare, spare` |

`W`是weight-scale FMA，`R`是row-scale/BF16-bias FMA，`P`是`v_perm_b32`。元素顺序
必须保证W/R/P不是同一值上的连续RAW链。该表有40条真实epilogue VALU和8个空slot；
空slot留给少量move/地址或scheduler，不能填packed FP32。

FP8 accumulator RAW距离尚未由当前微基准验证。若ISA/功能测试显示`F0..F3`开始W过早，
先实现保守V0：`SR0/SR1`只计算，`SR2`退休SR0，`SR3`退休SR1，仅隐藏80/160=50%的
scalar VALU；SR2/SR3在下一S0退休。V0提供完整16-MFMA super-record间隔。

##### scale与调度约束

当前实现不为PTPC weight scale增加LDS。K1对应的S0直接发出8条
`global_load_dwordx4`，在进入S1E前保留scale fragment；S1E通过逐级`vmcnt(7..2)`让
每个super-record仅在对应scale就绪后执行。scale VMEM不计入每个MFMA后的3条scalar
VALU容量。这个选择使rolling版本比原先在S1E内发scale load多占8个VGPR，但保证所有
compute段都没有VMEM/DS/store。总LDS为：

```text
32KB B ping/pong + 16KB shared CShuffle = 48KB
```

S1E只允许MFMA、scalar FMA/permute及必要move。CShuffle DS和global store全部留在下一
S0；strict stagger保证同一物理时刻只有一个group使用共享16KB CShuffle。

三条VALU不能是同一个值上的`W -> R -> P`连续依赖链。gfx942/ROCDL支持
`sched_group_barrier`，可在自然basic block内用`MFMA 1 -> VALU 3`约束machine
scheduler；但它可能误匹配普通VALU并延长live range。首版先显式生成顺序并检查ISA，
只有LLVM重新聚团时才加局部group barrier。

FlyDSL调度提示的伪代码为：

```text
# 先在同一个natural basic block生成16条MFMA和40条独立scalar VALU。
emit_super_record_mfma_and_epilogue()

# 然后给machine scheduler描述目标节奏；0x8=MFMA，0x2=VALU。
for slot in 0..15:
    sched_group_barrier(mask=0x8, count=1, group=0)
    sched_group_barrier(mask=0x2, count=3, group=0)
sched_barrier(0)
```

最后两个slot实际只有8个空位中的一部分真实VALU，`count=3`可能误匹配packet外地址指令；
更稳妥的实现是按上表真实数量生成`3,...,3,1,0,0`，或把16-MFMA packet拆成两个自然
basic block。每个block内统计必须精确，不能用`count=100`通配。

`v_fma_f32`和`v_perm_b32`是对PC对齐敏感的8-byte指令；微基准中跨8-byte边界会把
约4 cycles抬到约5 cycles。fresh ISA必须检查rolling区域的实际PC对齐，且避免无意插入
4-byte指令令后续整段VOP3错位。若对齐无法稳定控制，应把实际5-cycle成本纳入容量，
此时每MFMA稳定隐藏3条的结论可能不再成立。

##### 与双group反相的组合

```text
T3: group0 S1E(n0,k1) || group1 S0(n0,k1)
T4: group0 S0(n1,k0)  || group1 S1E(n0,k1)
T5: group0 S1P(n1,k0) || group1 S0(n1,k0)
T6: group0 S0(n1,k1)  || group1 S1P(n1,k0)
T7: group0 S1E(n1,k1) || group1 S0(n1,k1)
```

因此V1同时利用两种重叠：S1E内是同wave MFMA/scalar-VALU co-issue；另一group的S0
提供VMEM/LDS/CShuffle/store。下一K0 S0先完成scalar SR3的40条VALU，目标是与peer K1
开头尚未插入epilogue VALU的16条MFMA重叠；随后才发B VMEM并执行LDS read、CShuffle和
store。这是最终ISA顺序保证，实际跨wave重叠程度仍由ATT确认。

注意同一SIMD只有共享的VALU发射资源；intra与inter容量不能相加。以`T4`为例，group1
的S1E携带120条scalar VALU，group0的S0携带剩余40条，合计仍是原tile的160条，面对
64条peer MFMA即`2.5 VALU/MFMA`，低于微基准的3条容量。V1的潜在收益来自让VALU更早
ready、缩短重S0及避免大段VALU聚团，不是把总容量从3变成6；若ATT显示VALU issue
contention或MFMA密度下降，应保留纯S1P基线。

##### 风险与晋级门禁

现有ceiling的“每16条MFMA插1条global store”实验回退5.93%，说明打断MFMA train和
让读写VMEM长期并存可能得不偿失。它没有测试独立标量VALU，不能直接否证本方案，但
要求先保留“纯MFMA S1 + peer-wave重S0”作为control。

候选只有同时满足以下条件才继续：

- fresh ISA `<=256 VGPR`、目标`<=240`，0 scratch；不能出现双C fragment；
- 每个carrier 16条MFMA对应40条scale/pack VALU，整tile总数不增加；
- MFMA之间是2--3条独立scalar VALU，而不是依赖链、packed FP32、DS/VMEM或成片VALU；
- ATT中MFMA issue密度不低于纯S1 control，VALU与MFMA重叠增加；
- shared CShuffle无跨group覆盖，总LDS为48KB，physical/reduced output逐bit一致；
- 先用同进程10-buffer ABBA4比较pure-S1与rolling-S1E，再决定是否采ATT。

**D0：最终N tile drain**

```text
1. 最后一个S1(n_last,k1)完成后，不再执行P/W。
2. load最后N tile的weight scale。
3. 对本group的Creg执行scale、BF16 pack、CShuffle和8条dwordx4 NT store。
4. 等待CShuffle LDS读取完成；output store仅按kernel完成语义所需程度drain。
5. stage_end()。
```

**D1：barrier代次补偿**

```text
group0: 额外执行一次stage_end()，补偿group1在P1执行的额外barrier
group1: 无额外stage
```

#### K256稳态并行时序

记号：

```text
M_g(n,k) = group g执行S0(n,k,g)
C_g(n,k) = group g执行S1(n,k)
E_g(n)   = group g写回N tile n中自己负责的128行
H_g(n,k) = group g预取/提交B[n,k].half[g]
```

下表列出prologue后连续十个物理时刻。每一行的group0和group1同时运行；表尾给出该
时刻结束后新weight tile的完成状态。

| 物理时刻 | group0：wave0--3 | group1：wave4--7 | weight流水状态 | output状态 |
| ---: | --- | --- | --- | --- |
| `T-1` | P0：`P/W B(0,0)`并gather A | P0：`P/W B(0,0)`并gather A | `B(0,0)`完整于ping | 无 |
| `T0` | `M_0(0,0)`：`H_0(0,1)`、`R(0,0)` | P1：额外barrier，阻塞 | `B(0,1).half0`于pong | 无 |
| `T1` | `C_0(0,0)`：64 MFMA | `M_1(0,0)`：`H_1(0,1)`、`R(0,0)` | `B(0,1)`完整于pong | 无 |
| `T2` | `M_0(0,1)`：`H_0(1,0)`、`R(0,1)` | `C_1(0,0)`：64 MFMA | `B(1,0).half0`于ping | 无 |
| `T3` | `C_0(0,1)`：64 MFMA，完成`C_0(0)` | `M_1(0,1)`：`H_1(1,0)`、`R(0,1)` | `B(1,0)`完整于ping | group0的`C(0)`就绪 |
| `T4` | `M_0(1,0)`：`H_0(1,1)`、`R(1,0)`、`E_0(0)` | `C_1(0,1)`：64 MFMA，完成`C_1(0)` | `B(1,1).half0`于pong | group0写`D(0)` |
| `T5` | `C_0(1,0)`：64 MFMA | `M_1(1,0)`：`H_1(1,1)`、`R(1,0)`、`E_1(0)` | `B(1,1)`完整于pong | group1写`D(0)` |
| `T6` | `M_0(1,1)`：`H_0(2,0)`、`R(1,1)` | `C_1(1,0)`：64 MFMA | `B(2,0).half0`于ping | 无 |
| `T7` | `C_0(1,1)`：64 MFMA，完成`C_0(1)` | `M_1(1,1)`：`H_1(2,0)`、`R(1,1)` | `B(2,0)`完整于ping | group0的`C(1)`就绪 |
| `T8` | `M_0(2,0)`：`H_0(2,1)`、`R(2,0)`、`E_0(1)` | `C_1(1,1)`：64 MFMA，完成`C_1(1)` | `B(2,1).half0`于pong | group0写`D(1)` |
| `T9` | `C_0(2,0)`：64 MFMA | `M_1(2,0)`：`H_1(2,1)`、`R(2,0)`、`E_1(1)` | `B(2,1)`完整于pong | group1写`D(1)` |

从`T1`开始，每个稳态时刻都是一组纯MFMA与另一组纯非MFMA并行：

```text
T1: group0 MFMA(n0,k0) || group1 prefetch B(n0,k1) + LDS read/write
T2: group0 prefetch B(n1,k0) + LDS read/write || group1 MFMA(n0,k0)
T3: group0 MFMA(n0,k1) || group1 prefetch下一N的B(n1,k0) + LDS read/write
T4: group0 prefetch B(n1,k1) + write D(n0) || group1 MFMA(n0,k1)
T5: group0 MFMA(n1,k0) || group1 prefetch B(n1,k1) + write D(n0)
```

`T2/T3`明确展示K1阶段预取下一N的K0；`T4/T5`展示最重的output epilogue分别与另一个
group的K1/K0 MFMA重叠。每个SIMD期望形成`wave g`与`wave g+4`的一对，但最终物理配对
必须由ATT确认。

#### 尾部并行时序

令`L=(n_last,k1)`为最后一个core：

| 物理时刻 | group0 | group1 |
| ---: | --- | --- |
| `TL-1` | `C_0(L)`，完成最后C | `M_1(L)`，无下一B预取 |
| `TL` | `D0_0`：写回最后D | `C_1(L)`，完成最后C |
| `TL+1` | `D1_0`：补偿barrier并阻塞 | `D0_1`：写回最后D |

两组完成相同的逻辑stage数和barrier代数后才能退出；不得让某组在最后一次store后直接
return，否则另一组可能永久等待。

#### 伪代码

```text
kernel_8x1(...):
    tid       = threadIdx.x                 # 0..511
    lane      = tid % 64
    wave      = readfirstlane(tid / 64)     # 0..7, SGPR
    group     = readfirstlane(wave / 4)     # 0 or 1
    local_w   = wave % 4

    task = map_task(blockIdx.y)
    if task * 256 >= valid_rows:            # WG-uniform exit, before stagger
        return

    # Three independent LDS objects. Do not slice one 32KB B allocation.
    shared b_ping[128 * 128] : fp8 align(16)
    shared b_pong[128 * 128] : fp8 align(16)
    shared scratch[16 KiB]   : byte align(16)

    # Prologue metadata uses scratch[0:2KiB].
    cooperative_load(sorted_ids[task, 0:256], scratch.ids)
    cooperative_load(sorted_weights[task, 0:256], scratch.route)
    barrier()

    # Weight prologue: issue B0 BEFORE A gather, so A loads/address work hide B0 latency.
    b_stage = P_full_B(0):
        issue_two_raw_buffer_load_dwordx4(
            B[core=0].stripe[wave],
            vector_offset = lane_B_offset,
            scalar_offset = readfirstlane(B_wave_base(core=0, wave)),
        )

    # Each wave owns 32 rows and keeps all K in registers. These VMEM loads are
    # younger than P_full_B(0), which permits a partial vmcnt wait for B0.
    for m_atom in 0..1:
        encoded_row[m_atom] = scratch.ids[wave*32 + m_atom*16 + lane%16]
        row_scale[m_atom] = scratch.route[...] * activation_scale[...]
        for a_chunk in full K:
            Areg[m_atom, a_chunk] = raw_buffer_load_A(encoded_row, a_chunk)

    # Wait only the two older B0 loads; keep younger A/scale loads in flight.
    wait P_full_B(0), keeping A/scale VMEM outstanding
    W_full_B(0):
        commit_two_ds_write_b128(Bslot(0), b_stage)
    wait A and row scale
    barrier()                              # B0 visible; scratch may become CShuffle

    K_TILES   = K / 128                    # first version: 2
    N_TILES   = N / 128
    TOTAL     = N_TILES * K_TILES
    Creg.fill(0)

    if group == 1:
        stage_end()                        # one-generation displacement

    for t in 0 .. TOTAL-1:
        n_tile = t / K_TILES
        k_tile = t % K_TILES

        # -------- memory stage M(t): no MFMA --------
        # 1. Issue the NEXT weight global prefetch first. K1 naturally points
        #    to the next N tile's K0 through flattened core numbering.
        if t + 1 < TOTAL:
            b_stage = P(t + 1, group):
                issue_two_raw_buffer_load_dwordx4(
                B[t+1].half[group],
                vector_offset = lane_B_offset,
                scalar_offset = readfirstlane(B_tile_wave_base(t+1, wave)),
            )

        # 2. Read the CURRENT complete weight tile from ping/pong to Breg.
        Breg = R(t):
            issue_16_ds_read_b128(Bslot(t))

        # 3. Retire a completed output while P(t+1,g) remains in flight.
        if k_tile == 0 and t > 0:
            wscale = raw_load_128_channel_scales(n_tile - 1)
            for row_pair in 0..1:
                # Stream one row-pair: scale -> BF16 pack -> 128-bit LDS.
                scaled = Creg[row_pair] * row_scale * wscale
                ds_write_b128(scratch.wave_local[local_w], pack_bf16(scaled))
                wait required LDS writes
                row_major = ds_read_b128(scratch.wave_local[local_w])
                wait required LDS reads
                buffer_store_dwordx4_nt(
                    row_major,
                    vector_offset = current_row_offset,
                    scalar_offset = readfirstlane(output_n_base(n_tile - 1)),
                )
            Creg.fill(0)

        # 4. Only now wait for the prefetched NEXT half and commit it to LDS.
        if t + 1 < TOTAL:
            wait P(t + 1, group), keeping newer output stores in flight
            W(t + 1, group):
                commit_two_ds_write_b128(
                    Bslot(t + 1).half[group], b_stage
                )

        wait until R(t) and W(t+1,group) LDS operations are done
        setprio(0)
        stage_end()

        # -------- compute stage C(t): MFMA only --------
        setprio(3)
        repeat 64 MFMA in dependency-spread order:
            Creg = mfma_fp8(Areg[k_tile], Breg, Creg)
        setprio(0)
        stage_end()

    # Retire the final N tile. At runtime group0 drains while group1 computes,
    # then group1 drains while group0 waits at the compensating barrier.
    wscale = raw_load_128_channel_scales(N_TILES - 1)
    cshuffle_and_store_final_C(Creg, row_scale, wscale, scratch)
    wait final LDS reads and required output stores
    stage_end()

    if group == 0:
        stage_end()                        # balance group1 prologue displacement
```

首版不保留previous-C；K1完成后的完整epilogue放在下一N的K0 memory stage。这样不会
增加64个C寄存器，但memory stage呈现轻/重交替。若重stage无法被64条MFMA遮盖，先把
epilogue按两个row-pair细分为更多stage；不要直接双缓冲C。

### 0.8 反相是否能遮盖非MFMA工作

每个compute stage固定64条MFMA。当前Qwen35 K256 `4x1` fresh ATT中同量K0段的中位
跨度约1024 cycles，可作为第一版阴影预算。8x1的重memory stage每wave约包含：

```text
下一B半片:       2 buffer load + 2 ds_write
当前B fragment: 16 ds_read
PTPC scale:      8个N atom的scale load（标量/向量化以ISA为准，2--4个live）
输出缩放:       约128个FP32 mul/FMA
BF16 pack:       约32个VALU/permute
CShuffle/store:  8 ds_write + 8 ds_read + 8 buffer store
```

总量约200条非MFMA指令。B预取应最先发出，VMEM等待与输出缩放、CShuffle和当前B的
LDS读取重叠；输出store保持异步。按纯发射槽和约1024-cycle计算窗，容量上可以遮盖，
且每个SIMD预期同时驻留一个compute wave和一个memory wave。

但这不是性能保证。48KiB LDS把实际形态限制在Q2；ceiling扫描中，W8 synthetic
`2stage_barrier`相对`2stage_0`明显回退：Qwen397 K256/Q2约
`268.19 -> 185.09T`，Qwen35 K256/Q2约`265.44 -> 186.96T`。该探针没有真实memory
工作可供遮盖，不能直接否证本设计，却证明hard barrier本身代价很高。原型必须同时
保留无错相或普通pipeline对照；只有ATT证明
每个SIMD上的一对resident wave长期呈现“一个MFMA、一个非MFMA”，且阶段总时长接近
`max(compute, memory)`而非两者之和，才算反相成功。

推荐ATT门禁：

- 8个wave按`0/4、1/5、2/6、3/7`形成每SIMD一对；若实际物理映射不同，分组必须调整；
- steady区MFMA/non-MFMA反相，不在同一时段一起等待barrier；
- barrier stall、`lgkmcnt`和`vmcnt`分别统计，不能只看MFMA union；
- 核心循环无scratch、无大整数vector地址链；
- B ping/pong覆盖前必须证明两个group都已读取旧slot。

### 0.9 两个Qwen K256 case的胜率

#### Qwen3.5 35B K256

32K时每expert平均1024行，BM256无额外expert padding；实际valid rows为262,144。
Down有效工作量为：

```text
2 * 32768 * 8 * 2048 * 256 = 274,877,906,944 FLOP
```

当前1x4为`0.819303ms / 335.50T`，当前生产4x1为
`0.766544ms / 358.59T`。4-wave cooperative-B-load ceiling为`473.10T`；8x1每wave的
MFMA、B-load和D-store工作量与该形态相同，可先作为乐观代理。8x1超过1x4只需达到
该代理上界的70.9%，超过当前4x1需75.8%，因此该case有现实胜率，应作为首个实现和
ABBA目标。

#### Qwen3.5 397B K256

32K时每expert平均640行，BM256按expert补到768行；实际4x1 BM256 valid rows为
393,216，而useful rows只有327,680，useful/executed效率为`5/6 = 83.33%`：

```text
useful FLOP   = 2 * 327680 * 4096 * 256 = 687,194,767,360
executed FLOP = 2 * 393216 * 4096 * 256 = 824,633,720,832
```

当前1x4最终验收为`1.866767ms / 368.12 useful TFLOPS`。沿用`452.90 executed T`的
cooperative ceiling作乐观代理，先乘padding效率后只有`377.42 useful T`，仅比1x4高
2.53%。8x1必须达到`441.74 executed T`，即代理上界的97.54%，才能刚好追平1x4；
真实kernel还必须支付LDS、barrier、scale、metadata和CShuffle成本。因此在固定BM256
约束下，Qwen397超越1x4的概率很低。它适合作为第二个正确性/性能对照，不应作为首个
晋级目标。

上述`452.90/473.10T`来自BN64、4-wave协作加载的无依赖ceiling，不是8x1实测。
正式实现前应先扩展/运行精确的`WMxWN=8x1、BM256、BN128、BK128、B cooperation=8、
LDS=48KiB、full-N/WG` skeleton，分别测普通pipeline和strict stagger；这里的折算只做
go/no-go排序。

### 0.10 实施顺序与门禁

1. 先做K256 compile-only skeleton：A/C/B fragment、两个独立B LDS变量和16KB scratch；
   不做正确性前先验证`<=240 VGPR`目标、`<=256`硬门禁、48KiB LDS、0 scratch。
2. 验证B地址多重集：每个BK128恰好覆盖`128*128`个FP8元素，两个group各覆盖不相交
   的8KB；ping/pong覆盖前无未完成reader。
3. 加入metadata/A gather和PTPC scale，禁止完整scale fragment；再次检查VGPR。
4. 加入row-pair CShuffle，逐bit/column-code验证128-bit LDS到dwordx4输出映射。
5. 先跑无错相参考，再开conditional-displacement stagger；功能门禁通过后使用空闲GPU、
    PTL `Enabled / VECTOR,F8`、1800MHz、NUMA off和10 buffers执行ABBA4。无错相参考使用
    32KB CShuffle和总计64KB LDS，strict stagger使用16KB CShuffle和总计48KB LDS。
6. Qwen35若稳定超过当前4x1，升ABBA12/24并采fresh ATT；Qwen397只在Qwen35流水成立后
   补测。所有结果同时报告ms和有效TFLOPS，并区分useful/executed工作量。

#### 当前实现状态

实现位于`src/contrib/flydsl/moe_gemm_2stage/gemm2_8x1.py`，源码SHA256为
`2a8dac91c072d074ed5b4cc8e8e713c2392bbbdb5e975f30613d6126ef4d1788`。固定约束为
BM256、BN128、BK128、K256、512 threads和PTPC FP8；dispatcher使用
`down_path="8x1"`。

compile-only的gfx942最终ISA验证如下：

| 版本 | ISA SHA256 | VGPR | SGPR | LDS | scratch |
| --- | --- | ---: | ---: | ---: | ---: |
| pure-S1 | `40fe50ada9010b9f3b97e5a78016e495bce07ebf5b56a1c109ceb7487a87e439` | 230 | 96 | 49152 B | 0 |
| rolling-S1E | `8aed9d0a89b4496bc629391c83af44731dbfb53e7120fedafdfd6534684c4d1c` | 232 | 96 | 49152 B | 0 |

rolling最终ISA共有2048条MFMA和128条`buffer_store_dwordx4`，无packed FP32；32个
compute段均为64条MFMA，K0段无普通VALU，K1段约120条scalar VALU且无VMEM、DS或
store。每个MFMA后的普通VALU不超过3条。P0中两条B0 load位于全部8条A gather load
之前，`vmcnt(4)`后提交B0到LDS，再以`vmcnt(0)`收完A。

2026-09-04的最终ISA审计还确认：16个K0 compute段都只有64条MFMA；16个K1 compute
段各有64条MFMA、48条weight-scale FMA、48条row-scale/BF16-bias FMA、24条
`v_perm_b32`和6条递减scale `vmcnt` wait，无地址VALU或内存指令。将4个独立fragment
改为统一按`W -> R -> P`阶段生成后，后端NOP由76条降至32条；再将SR3的40条VALU
提前到下一K0 memory stage头部后，最终NOP为20条且资源不变。

K1先发2条next-B和8条较新的weight-scale load，随后以`vmcnt(8)`只等待next-B；带8条
output store的K0也以`vmcnt(8)`等待next-B。SR3前移后，每个稳态K0先发两条当前B，
此时VMEM队列按“2笔旧scale + 2笔新B”排列；随后以`vmcnt(3) -> 4 W -> vmcnt(2)`只消费
旧scale，两条B保持在途，再执行剩余36条`W/R/P`。因此scale等待不会drain当前B预取；
完整kernel仍只有4处边界`vmcnt(0)`，steady K0没有`vmcnt(0)`。

B LDS的`ds_write_b128`和`ds_read_b128`、CShuffle的`ds_write_b128`和
`ds_read2st64_b64`按CDNA3 lane-group静态模拟均为0冲突，每组访问恰好覆盖32个bank。
GPU7上的目标dispatch（grid 4608、workgroup 512、LDS 49152 B、scratch 0）进一步测得
`SQ_LDS_BANK_CONFLICT=0`。

后续地址审计发现旧rolling ISA仍有316条整数VALU。它们不在compute段，但输出地址在
每个N tile的K0 memory stage重复生成，B tile地址也在每个core重复做VGPR加法。当前
版本将8个lane相关输出destination offset提前到循环前，后续N tile增量折叠为
`buffer_store_dwordx4 offset:*`；B的两个lane offset只计算一次并固定在两个VGPR中，
`n*32768 + k*2048`通过MUBUF的SGPR `soffset`传入。32个B core offset已逐项与理论序列
核对一致。最终整数VALU由316降为136；新增31条SALU用于
物化展开后的scalar offset，rolling VGPR从230升到232，仍低于240目标。32个compute
段中的整数VALU保持为0。剩余136条集中在metadata/A gather、CShuffle首次地址初始化
和一次64-bit scale地址进位，不在稳态重复计算。

当前最终ISA共有6774条指令。15个稳态K0 memory stage都严格以
`16 v_fma_f32 + 16 v_fmaak_f32 + 8 v_perm_b32`开头，随后才发两条B
`buffer_load_dwordx4`。第一组CShuffle的4写/4读先入LDS队列，再发16条B
`ds_read_b128`；`lgkmcnt(14)`只等待较老CShuffle结果并保留大部分B读在途，进入MFMA前
仍以`lgkmcnt(0)`保证B fragment完整。总数学、VMEM和DS指令数不变。

当前版本重新采集`SQ_LDS_BANK_CONFLICT=0`；静态CDNA3 lane-group模拟也确认全部B
`ds_read_b128` offset每组恰好覆盖32个bank。旧ATT中单条B读的successful issue成本
p50/p95均为4 cycles，相邻读issue delta的p50/p90为20/32 cycles；该约32-cycle现象来自
同CU四SIMD、双resident wave共享LDS发射/队列，而不是bank replay。原始紧邻B burst的
`lgkmcnt(0)`实测stall p50为152 cycles，因此立即full wait并非必要。

Qwen35 K256 B32768、PTL `Enabled / VECTOR,F8`、1800MHz、10-buffer短ABBA4中，原始
full-wait与partial-wait版本分别为0.900644ms / 305.20有效TFLOPS和
0.879963ms / 312.37有效TFLOPS。候选中位ratio为0.9690，但IQR为
`[0.8646, 1.0805]`且仅2/4轮获胜，因此只视为方向性信号，不作为稳定性能结论。

fresh rolling端到端测试无死锁和非法访问，结果为
`test_acc_fly_splitk_2s_down_8x1: 1 passed`、`diff=0.00018656`。测试在auto DPM且NUMA
balancing开启的环境进行，只用于正确性和bank counter，不作为正式性能结果；正式
多buffer性能仍待验证。地址前移版
已在GPU7直接复跑同一正确性测试，仍为`1 passed`、`diff=0.00018656`；正确性测试无需
等待GPU空闲，只有正式性能测试使用空闲GPU门禁。bank-counter结果来自地址前移前版本，
但B/CShuffle LDS地址公式和DS指令未变化。

#### 2026-09-04 fresh ATT：K0-head SR3反相

> 以下至“当前K256 ATT入口”是实现演进记录，使用过旧geometry或旧owner口径，仅用于
> 解释候选取舍。当前统计定义、数值和复现命令只认
> [stall_analysis.md](../../../../../../../../../../flydsl/attn_4wave/tools/stall_analysis.md)。

最终B-first源码（SHA256 `fb296aeb...b65167`）在GPU7以1800MHz determinism、PTL
`Enabled / VECTOR,F8`、650W和NUMA off抓取dispatch 16。Qwen35 K256 B32768单次
dispatch为0.819003ms；有效工作量为274,877,906,944 FLOP，对应335.625有效TFLOPS。
该数值来自ATT采集运行，不替代多buffer ABBA性能结论。

四个SE共得到536条完整wave trace，其中408条active wave均恰好执行2048条MFMA，128条
为uniform early-exit。16个物理SIMD都同时驻留两个wave slot；分析204个重叠wave pair
和6120个稳态K0-head窗口后得到：

- B第二条load到首条SR3 VALU的p50距离为12 cycles；
- scale `vmcnt(3)`和`vmcnt(2)`在全部6120个窗口中都仅为4 cycles，无观测到data stall；
- 40条SR3 VALU的p50跨度为216 cycles，p95为332 cycles；
- 99.395%的SR3 VALU issue落在peer MFMA执行窗内，96.285%落在peer纯MFMA区域；
- group1纯MFMA命中为100%，group0为92.569%（任意MFMA命中98.791%）；
- B在`vmcnt(8)`前的p50 lead time为1752 cycles，`vmcnt(8)`的p50/p95均为4 cycles；
- 仅89/6120（1.454%）个`vmcnt(8)`超过4 cycles，7/6120（0.114%）超过1024 cycles。

因此scale partial wait没有破坏B预取，K0-head SR3与peer MFMA的反相重叠已被ATT直接验证。
剩余问题是group0相位不如group1完整，以及少量B VMEM长尾。MFMA-lifecycle steady union为
65.941%；trim掉每slot一个replacement wave后的central two-slot envelope MFMA union为
58.927%，双slot active比例为99.313%，说明仍有明显非MFMA/tail暴露。

完整解码UI和专项分析位于`ui_output_moe_8x1_k0head_dispatch_16/`，入口报告为
`ATT_ANALYSIS.md`。采集结束后GPU7已恢复为auto、PTL Disabled和NUMA balancing on。

#### 2026-09-04：32-MFMA stage、跨stage B预取与`lgkmcnt`实验

当前8x1已将每个BK128的64条MFMA拆成两个32-MFMA compute stage；K256每个N tile
形成8个memory/compute stage。B quarter使用单套staging跨过一个完整32-MFMA窗口后，
再由下一memory stage等待并提交到LDS。生产shape最终ISA为178 VGPR、96 SGPR、48KiB
LDS、0 scratch，保持2048条MFMA、72条B buffer load、128条scale load、512条B
`ds_read_b128`和128条output store不变。N2048专项正确性为`diff=0.00017510`。

carry版本fresh ATT位于`ui_output_moe_8x1_vmcnt_carry_dispatch_16/`。B load到
`vmcnt(4)`尝试的p50提前量由472增至1176 cycles；wait stall由p50/p95
`48/276`降至`4/4` cycles，98.94%的稳态wait仅4 cycles。dispatch 16为
`0.694283ms / 395.92 useful TFLOPS`，但该值是单次ATT，不替代多buffer正式结论。

在carry基础上隔离测试两种`lgkmcnt`调整。以下均为`n>0`且存在下一B预取的稳态
memory stage；每段之后都是`barrier -> 32 MFMA + 40 VALU`。

##### Carry：两次full LDS wait（历史基线）

```text
2 global_load_dwordx4                 # 当前scale quarter

2 ds_write_b128                       # CShuffle写
2 ds_read2st64_b64                    # CShuffle读
s_waitcnt lgkmcnt(0)                  # 等CShuffle结果
整理CShuffle结果
2 buffer_store_dwordx4 nt             # 上一N的一个output quarter

8 ds_read_b128                        # 当前B half -> MFMA operand
s_waitcnt vmcnt(4)                    # 等跨stage pending-B；保留2 scale + 2 store
1 ds_write_b128                       # pending-B提交到下一LDS slot
1 buffer_load_dwordx4                 # 发下一个B quarter，跨32 MFMA
s_waitcnt lgkmcnt(0)                  # 等8 B-read + 1 pending-B write
s_barrier
```

第一次full wait保证CShuffle结果可由VALU整理并送入output store；第二次full wait保证
当前B fragment和下一LDS slot在barrier前完成。生产ISA为178 VGPR、205条
`s_waitcnt`，其中133条`lgkmcnt(0)`。

##### Partial：B读越过CShuffle partial wait

```text
2 global_load_dwordx4                 # 当前scale quarter

2 ds_write_b128                       # CShuffle写
2 ds_read2st64_b64                    # CShuffle读
8 ds_read_b128                        # 当前B half先入队
s_waitcnt lgkmcnt(8)                  # 源码意图：只收较老CShuffle，保留B读
整理CShuffle结果
2 buffer_store_dwordx4 nt

s_waitcnt vmcnt(4)                    # 等pending-B；保留2 scale + 2 store
1 ds_write_b128                       # pending-B提交到下一LDS slot
1 buffer_load_dwordx4                 # 下一个B quarter
s_waitcnt lgkmcnt(0)                  # 等B读和pending-B write
s_barrier
```

最终ISA没有简单保留“一条`lgkmcnt(8)`”；LLVM按两个CShuffle结果的实际寄存器依赖生成
`lgkmcnt(9)`和`lgkmcnt(8)`。完整N2048 ISA共有91条`lgkmcnt(8)`、60条
`lgkmcnt(9)`和73条`lgkmcnt(0)`，总`s_waitcnt`反而增至296条，VGPR增至182。

##### Merged：源码合并为一次full LDS wait

```text
2 global_load_dwordx4                 # 当前scale quarter

2 ds_write_b128                       # CShuffle写
2 ds_read2st64_b64                    # CShuffle读，结果暂不消费
8 ds_read_b128                        # 当前B half
s_waitcnt vmcnt(2)                    # 等pending-B；只保留当前2条scale load
1 ds_write_b128                       # pending-B提交到下一LDS slot
s_waitcnt lgkmcnt(0)                  # 源码唯一full wait：CShuffle + B读 + B写
整理CShuffle结果
2 buffer_store_dwordx4 nt
1 buffer_load_dwordx4                 # 下一个B quarter，放在stage最后
s_barrier
```

源码只有一次显式`lgkmcnt(0)`，但机器调度器仍需分别等待两个CShuffle结果寄存器；代表性
最终ISA为`lgkmcnt(10) -> lgkmcnt(9) -> lgkmcnt(0)`。完整ISA共有1条
`lgkmcnt(10)`、32条`lgkmcnt(8)`、60条`lgkmcnt(9)`和73条`lgkmcnt(0)`；因此
“源码合并”并不等于机器码只剩一个wait。总`s_waitcnt`为238条，VGPR为182。

##### 三版本对比

| 项目 | carry | partial | merged |
| --- | --- | --- | --- |
| CShuffle结果等待 | 立即`lgkmcnt(0)` | B读后源码`lgkmcnt(8)` | 延迟到stage尾统一等待 |
| stage尾等待 | `lgkmcnt(0)` | `lgkmcnt(0)` | 同一个源码`lgkmcnt(0)` |
| pending-B VMEM wait | `vmcnt(4)` | `vmcnt(4)` | `vmcnt(2)` |
| 下一个B load位置 | stage尾、full LDS wait前 | stage尾、full LDS wait前 | output store后、barrier前 |
| 最终ISA partial LDS wait | 无 | 91×`lgkmcnt(8)` + 60×`lgkmcnt(9)` | 32×`lgkmcnt(8)` + 60×`lgkmcnt(9)` + 1×`lgkmcnt(10)` |
| 最终ISA `lgkmcnt(0)` | 133 | 73 | 73 |
| 总`s_waitcnt` | 205 | 296 | 238 |
| VGPR / SGPR / LDS / scratch | 178 / 96 / 48KiB / 0 | 182 / 96 / 48KiB / 0 | 182 / 96 / 48KiB / 0 |
| 数学、VMEM、DS、store总量 | 基准 | 与基准相同 | 与基准相同 |
| N2048物理输出 | 基准 | 逐bit相同，relative-L2=0 | 逐bit相同，relative-L2=0 |
| 独立ABBA12中位 | 0.765723ms / 358.98T | 0.765463ms / 359.10T | 0.801003ms / 343.17T |
| 候选/carry配对ratio中位 | 1.00000 | 1.00073，6/12胜 | 1.04704，6/12胜 |
| 结论 | **保留** | 与carry持平，不合入 | 回退约4.29%，不合入 |

三版本N2048物理输出逐bit一致、relative-L2为0。partial绝对中位只比同轮carry快
0.034%，配对IQR大幅跨1，视为持平；merged相对carry回退约4.29%。因此当前工作区
当时保留carry版本，不合入基础partial或merged。

merged隔离源码`5709ca09...`另行抓取fresh ATT。GPU7 dispatch 16为
`0.731723ms / 375.66 useful TFLOPS`，trace资源为56 regular + 128 accum VGPR、
112 SGPR、48KiB LDS、0 scratch。四个SE共生成536个完整wave JSON，UI位于
`ui_output_moe_8x1_lgkm_merged_dispatch_16/`；UI内嵌`source_2_gemm2_8x1.py`的
SHA256与merged隔离源码完全一致。该单次ATT数值不改变ABBA12中merged回退4.29%的
正式判断。

Partial还测试了late-wait变体：删除memory stage中`s_barrier`前的源码
`s_waitcnt lgkmcnt(0)`，并将其移动到随后32-MFMA compute stage末尾。隔离源码SHA256
为`17f2e541...`，N2048输出与carry逐bit一致、relative-L2为0。最终ISA没有把一条
full wait原样放到第32条MFMA后；LLVM按各组B operand首次消费自动拆成
`lgkmcnt(8/6/5/4/2/1/0)`并穿插在MFMA序列中。生产ISA为182 VGPR、96 SGPR、48KiB
LDS、0 scratch，数学、VMEM、DS和store总量不变，但`s_waitcnt`增至612条。

late-wait fresh ATT dispatch 16为`0.695163ms / 395.42 useful TFLOPS`。400个active
wave中，compute内自动wait的p50/p90/p95均为4 cycles；`lgkmcnt(8)`全部为4 cycles，
`lgkmcnt(0)`有97.91%为4 cycles。四个SE共生成656个完整wave JSON，UI位于
`ui_output_moe_8x1_partial_latewait_dispatch_16/`。该结果说明LDS完成延迟已被分摊到
MFMA消费点，但仍需独立多buffer ABBA才能判断大量自动wait是否改善稳定性能。该
late-wait源码当时已合入工作区，源码SHA256为`17f2e541...`；工作区fresh N2048 ISA
与ATT前隔离候选逐字节一致，ISA SHA256为`7a564947...`。

##### Memory-stage `v_mov`的两种实验

late-wait生产ISA的每个稳态memory stage有4条`v_mov_b32`，来自CShuffle两条
`ds_read2st64_b64`结果向两个连续`buffer_store_dwordx4`源的重排，不是B staging或
地址生成。N2048稳态memory区共有约240条、完整kernel共有262条`v_mov_b32`。late-wait
ATT中全部动态`v_mov` stall为p50 4、p95 44 cycles，部分CShuffle拼装site的p90达到
88--96 cycles。

**实验1：删除move。** 每个output row先分配连续8-BF16 rmem fragment，将两个64-bit
LDS copy直接写入前后half；在copy间加入`sched_barrier(0)`阻止LLVM跨row融合
`ds_read2st64_b64`。最终ISA把128条`ds_read2st64_b64`改成256条`ds_read_b64`，完整
`v_mov_b32`由262降到6，VGPR由182降到172，总指令由7449降到7203；代价是总DS issue
从832增至960。N2048输出与late-wait逐bit一致、relative-L2=0。独立10-buffer ABBA12：

```text
late-wait: 0.765223ms / 359.21T
novmov:    0.759783ms / 361.78T
ratio median 0.99822，IQR [0.87408, 1.12670]，6/12胜
```

绝对中位快约0.71%，但配对结果完全不稳定；增加的128条DS issue基本抵消move/VGPR
减少，因此不合入。

**实验2：让四个compute stage都从头部交织VALU。** 原late-wait中phase0/2的40条
epilogue VALU位于MFMA 1--14，phase1/3位于MFMA 17--30。实验版改为phase0/1分别退休
上一N的SR2/SR3，phase2/3退休当前N已完成的SR0/SR1，最终四个phase均把40条VALU放在
MFMA 1--14；最后N的SR2/SR3在tail补齐。N2048逐bit一致、relative-L2=0，但跨tile
延长packed-record生命周期使VGPR由182升到200。独立10-buffer ABBA12：

```text
late-wait: 0.768604ms / 357.63T
headvalu:  0.769923ms / 357.02T
ratio median 1.00505，IQR [0.87889, 1.13443]，6/12胜
```

该版本略慢且配对不稳定，也不合入。当时工作区继续保留late-wait源码`17f2e541...`。

##### Corrected physical-SIMD ATT与4+4 B-read调度

旧K0-head ATT按每core 64 MFMA、每N tile 2 core分析，得到的65.941%/58.927% union
不能代表当前32-MFMA half-stage实现。将分析器改为每core 32 MFMA、每N tile 4 core，
并排除0-MFMA uniform early-exit wave后，严格按`stall_analysis.md`执行：动态事件使用
`code[pc_index]`映射，successful issue取`first_attempt + stall`，每条MFMA标记16-cycle
执行窗，再按`(shader_engine, cu, simd)`合并两个resident wave。N2--N13的late-wait
physical MFMA-union账本为：

```text
physical total:                        11,715,980 cycles
MFMA-union busy:                        9,728,000 cycles = 83.0319%
MFMA-union idle:                        1,987,980 cycles = 16.9681%
```

每个4-cycle union-idle tick按两个active wave等权归入互斥owner，owner之和严格等于idle。
late-wait owner构成为：

| exclusive owner | cycles | idle占比 | union总周期占比 |
| --- | ---: | ---: | ---: |
| normal issue exposure | 604,826 | 30.424% | 5.162% |
| other dependency stall | 488,546 | 24.575% | 4.170% |
| structural tail | 289,804 | 14.578% | 2.474% |
| DS issue stall | 232,464 | 11.693% | 1.984% |
| LDS completion wait | 123,340 | 6.204% | 1.053% |
| VMEM issue stall | 102,974 | 5.180% | 0.879% |
| VMEM completion wait | 94,638 | 4.761% | 0.808% |
| scheduler ready | 51,032 | 2.567% | 0.436% |
| MFMA issue unavailable | 356 | 0.018% | 0.003% |

这里`normal issue exposure`表示union空洞内仍在成功发射普通VALU/DS/VMEM/SALU的服务
成本，不是stall；`other dependency stall`主要包含barrier、VALU和SALU/control依赖。
单wave的MFMA unavailable不能直接相加到physical账本，本trace中它在physical union里
近似为0。

进一步按静态PC分类133条`s_barrier`，`barrier + barrier`的397,104 cycles全部闭合：
381,644 cycles（96.1%）来自`compute_end + memory_end`，15,460 cycles来自
`memory_end + prologue phase compensation`。主循环不存在可直接删除的同类重复barrier；
这些barrier同时维持两个4-wave组反相、B quarter可见性和16KiB CShuffle复用。gfx942
也不支持gfx12的`s_barrier_signal/s_barrier_wait`拆分，因此不采用提前signal方案。

隔离测试将memory-stage优先级从0提高到1或2。两版N2048均逐bit一致，最终ISA除128处
`s_setprio`立即数外与late-wait相同。`setprio=1`独立10-buffer ABBA12结果为：

```text
late-wait: 0.768383ms / 357.74 useful TFLOPS
setprio=1: 0.770503ms / 356.75 useful TFLOPS
ratio median 1.00761，IQR [0.87935, 1.13381]，6/12胜
```

结果不稳定且略慢；`setprio=2`在ABBA4中也无优势，二者均不合入。

随后直接针对`VALU issue + DS-read stall`做4+4 B-read调度。每个N32 quarter使用独立
LDS视图和fragment，最终ISA的稳态memory stage为：

```text
2 ds_read2st64_b64                    # CShuffle读
4 ds_read_b128                        # 当前B quarter 0
s_waitcnt lgkmcnt(5/4)
4 v_mov_b32 + 2 buffer_store_dwordx4  # 消费CShuffle结果
4 ds_read_b128                        # 当前B quarter 1
```

调度屏障只约束LLVM不能把第二组B读重新上提，不生成额外机器指令。与late-wait相比，
完整N2048 ISA仍为2048 MFMA、512条B `ds_read_b128`、128条CShuffle
`ds_read2st64_b64`、192条`ds_write_b128`、612条`s_waitcnt`和133条`s_barrier`；
资源仍为182 VGPR、96 SGPR、48KiB LDS、0 scratch，也未增加`vmcnt(0)`。

10-buffer、GPU7、1800MHz determinism、PTL `Enabled / VECTOR,F8`、650W、NUMA off，
每轮先对两个版本各做一次不计时prime以消除同步后首dispatch固定长尾。有效工作量为
`2*32768*8*2048*256 = 274,877,906,944 FLOP`：

| 测试 | late-wait | 4+4 B-read | 配对ratio | 胜场 |
| --- | ---: | ---: | ---: | ---: |
| primed ABBA4 | 0.772723ms / 355.73T | 0.768924ms / 357.48T | 0.99232，IQR [0.98886, 0.99602] | 4/4 |
| primed ABBA12 #1 | 0.764463ms / 359.57T | 0.760224ms / 361.58T | 0.99200，IQR [0.98690, 1.00023] | 9/12 |
| primed ABBA12 #2 | 0.773843ms / 355.21T | 0.768003ms / 357.91T | 0.98931，IQR [0.98512, 0.99344] | 12/12 |
| clean primed ABBA24 | 0.778023ms / 353.30T | 0.768463ms / 357.70T | 0.98752，IQR [0.98581, 0.99231] | 24/24 |

clean ABBA24的配对中位改善为1.25%，24/24轮胜；每版本48个绝对样本。测试初始全机
0% busy，GPU7为auto、PTL Disabled、NUMA on；测试中为1800MHz determinism、PTL
`Enabled / VECTOR,F8`、NUMA off；结束后恢复auto、PTL Disabled、NUMA on。fresh ATT
dispatch 16为`0.695283ms / 395.35 useful TFLOPS`，该单次ATT时延不替代ABBA结论。

bread44的N2--N13 physical MFMA-union为：

```text
physical total:                        11,748,900 cycles
MFMA-union busy:                        9,922,560 cycles = 84.4552%
MFMA-union idle:                        1,826,340 cycles = 15.5448%
lifecycle MFMA-union busy:                                  70.20%
single-wave cycles/N:                  4930.63 -> 4849.51 (-1.65%)
```

其exclusive owner账本为：

| exclusive owner | cycles | idle占比 | union总周期占比 |
| --- | ---: | ---: | ---: |
| normal issue exposure | 620,918 | 33.998% | 5.285% |
| other dependency stall | 506,370 | 27.726% | 4.310% |
| structural tail | 230,952 | 12.646% | 1.966% |
| DS issue stall | 172,444 | 9.442% | 1.468% |
| LDS completion wait | 132,498 | 7.255% | 1.128% |
| VMEM completion wait | 69,962 | 3.831% | 0.595% |
| VMEM issue stall | 51,114 | 2.799% | 0.435% |
| scheduler ready | 42,082 | 2.304% | 0.358% |
| MFMA issue unavailable | 0 | 0% | 0% |

按各trace自己的physical total归一，late-wait到bread44的owner转移为：

| owner | late-wait | bread44 | 变化 |
| --- | ---: | ---: | ---: |
| normal issue exposure | 5.1624% | 5.2849% | +0.1225 pp |
| other dependency stall | 4.1699% | 4.3099% | +0.1400 pp |
| structural tail | 2.4736% | 1.9657% | -0.5078 pp |
| DS issue stall | 1.9842% | 1.4677% | -0.5164 pp |
| LDS completion wait | 1.0528% | 1.1277% | +0.0750 pp |
| VMEM completion wait | 0.8078% | 0.5955% | -0.2123 pp |
| VMEM issue stall | 0.8789% | 0.4351% | -0.4439 pp |
| scheduler ready | 0.4356% | 0.3582% | -0.0774 pp |
| MFMA issue unavailable | 0.0030% | 0% | -0.0030 pp |

下降项合计1.7608个百分点，转移到normal issue、other dependency和LDS wait共
0.3375个百分点，净减少idle 1.4233个百分点，恰好等于MFMA union busy增量，账本闭合。

bread44热点进一步拆分如下，百分比均为union总周期占比：

- normal issue 5.285%：普通VALU 3.843%，DS write 0.687%，DS read 0.416%，
    VMEM load 0.226%，VMEM store 0.063%，SALU 0.048%。其中`v_fmaak_f32`为
    1.700%，`v_fma_f32`为1.633%，`v_perm_b32`为0.348%。
- other dependency 4.310%：`s_barrier` 2.878%，SALU/control 0.929%，VALU依赖
    0.503%；其中`s_setprio` 0.497%、`s_nop` 0.433%。
- DS issue 1.468%：`ds_read_b128` 0.857%、`ds_read2st64_b64` 0.311%、
    `ds_write_b128` 0.300%。
- LDS completion wait 1.128%：全部为`lgkmcnt`；主要是compute消费B fragment时的
    `lgkmcnt(8/6/5/2/1)`，memory中的`lgkmcnt(5/4)`只占较小部分。
- VMEM issue 0.435%：scale `global_load_dwordx4` 0.304%、output store 0.118%、
    B `buffer_load_dwordx4` 0.013%。VMEM completion wait 0.595%全部来自
    `vmcnt(4)`。

joint reason×phase只作为定位见证，不能与owner重复相加。最大组合为
`normal issue@core3 + structural tail@tail`，占idle 3.934%（总周期0.611%）；其次是
`DS issue@boundary23 + normal issue@core2` 3.530%和
`DS issue@boundary12 + normal issue@core2` 3.509%。`all_waves_same_reason`也只是
witness：other dependency占总周期2.119%，normal issue占1.836%，双tail占0.498%；
不存在all-waves VMEM issue或VMEM wait见证，strict VMEM-wait floor为0。

ATT共536条完整wave：408条active wave均执行2048 MFMA，128条uniform early-exit；
四SE active分布为104/96/104/104。16个采样physical SIMD中12个捕获26条active wave、
4个捕获24条。dispatch grid为`(512, 1280, 1)`、每WG 512 threads，资源为56 regular
VGPR + 128 AGPR、112 trace SGPR、48KiB LDS、0 scratch。该采样只解释目标CU内部，
不覆盖80 CU间dispatch-tail；整卡结论由上述clean ABBA24给出。

因此4+4的主收益是降低B `ds_read_b128` issue、B/output VMEM issue/wait和structural
tail；部分时间转移成普通FMA发射、VALU依赖与`lgkmcnt`，但physical union和墙钟同时
改善，不是单纯的stall转移。rolling和
pure N2048物理输出均逐bit一致、relative-L2=0；仓库专项pytest为`1 passed`。该4+4
基线源码SHA256为`c0626336...`，fresh N2048 ISA SHA256为`edf9d2f9...`。ATT UI保存在
`ui_output_moe_8x1_bread44_dispatch_16/`。ATT隔离源码SHA256为`0f00d257...`；它与
工作区只存在缩进格式差异，二者fresh ISA逐字节一致。

##### Prologue额外half-B预取：两级VMEM carry

4+4 B-read版本的稳态`vmcnt(4)`前方有2条scale load和2条output store，等待更老的
pending-B quarter完成。ATT中24,072次动态`vmcnt(4)`的p50/p90/p95均为4 cycles，
730次（3.03%）超过4 cycles；strict physical owner中VMEM completion wait占union
总周期0.5955%。

本实验不是把同一条load在memory stage内前移，而是在prologue额外发出下一份half-B
对应的quarter request，并把单级carry扩成两级FIFO。每个4-wave group的序列为：

```text
prologue: issue L0 -> staging0; issue L1 -> staging1; vmcnt(1)
peeled M0: commit L0; issue L2 -> staging0
steady Mh: commit Lh from staging[h&1]; issue L(h+2) into freed staging
```

每个`Lh`是每wave一条`buffer_load_dwordx4`；两个4-wave group分别贡献一个quarter，
合起来形成一个N64 half-B。稳态等待点同时保留一个更年轻的B request，因此rolling
主路径由56处`vmcnt(4)`变成`vmcnt(5)`。静态producer/consumer审计确认62/62个B
load到`ds_write_b128` commit边均有counter-valid wait覆盖；ATT中load successful issue
到commit successful issue的p50由1156增至2340 cycles，主要跨越的MFMA数由32增至64。

资源预算与fresh ISA结果：

| 资源/工作量 | 4+4基线 | 两级carry | 变化 |
| --- | ---: | ---: | ---: |
| 编译器next-free VGPR | 182 | 186 | +4 |
| ATT regular + accum allocation | 56 + 128 | 60 + 132 | +8 combined allocation |
| SGPR | 96 | 96 | 0 |
| LDS | 48KiB | 48KiB | 0 |
| scratch | 0 | 0 | 0 |
| B buffer load / B DS read / MFMA | 72 / 512 / 2048 | 72 / 512 / 2048 | 0 |

48KiB LDS已经把kernel限制为1 WG/CU，即2 waves/SIMD；192 combined allocation仍低于
256硬门槛，occupancy不变，并保留64个combined register余量。rolling/pure的N2048
以及单tile N128 rolling/pure均逐bit一致、relative-L2=0；专项pytest为`1 passed`。

10-buffer clean primed ABBA结果：

| 测试 | 4+4基线 | 两级carry | 配对ratio | 胜场 |
| --- | ---: | ---: | ---: | ---: |
| ABBA4 | 0.774323ms / 354.99T | 0.754624ms / 364.26T | 0.97334，IQR [0.96365, 0.97519] | 4/4 |
| ABBA12 | 0.779004ms / 352.86T | 0.752303ms / 365.38T | 0.96697，IQR [0.96552, 0.97089] | 12/12 |
| ABBA24 | 0.774583ms / 354.87T | 0.751384ms / 365.83T | 0.96702，IQR [0.96372, 0.96969] | 24/24 |

fresh ATT dispatch 16为`0.677603ms / 405.66 useful TFLOPS`。N2--N13 strict
physical MFMA-union由84.4552%升至85.2412%，single-wave cycles/N由4849.51降至
4807.02。exclusive owner按union总周期归一的转移为：

| owner | 4+4基线 | 两级carry | 变化 |
| --- | ---: | ---: | ---: |
| normal issue exposure | 5.2849% | 5.3555% | +0.0706 pp |
| other dependency stall | 4.3099% | 3.8368% | -0.4732 pp |
| structural tail | 1.9657% | 1.6409% | -0.3248 pp |
| DS issue stall | 1.4677% | 1.5085% | +0.0407 pp |
| LDS completion wait | 1.1277% | 1.1486% | +0.0208 pp |
| VMEM completion wait | 0.5955% | 0.5225% | -0.0730 pp |
| VMEM issue stall | 0.4351% | 0.4492% | +0.0141 pp |
| scheduler ready | 0.3582% | 0.2969% | -0.0613 pp |

因此增加预取没有造成显著VMEM issue反弹；但约3.3%的墙钟改善也不只来自减少
`vmcnt`等待。更深carry改变了memory/compute到达相位，主要减少barrier依赖0.4830
个百分点和structural tail 0.3248个百分点。该深预取基线源码SHA256为
`3a4936d9...`，fresh ISA SHA256为`73961767...`；ATT UI位于
`ui_output_moe_8x1_bprefetch2_dispatch_16/`。

##### K128/K192/K384/K512泛化

多K支持没有复制新的builder。K256专用rolling路径保持原样；其余K通过编译期常量
选择BK、fragment rank和pure epilogue。B流水由原先隐含的两K-stage关系改为全局core
坐标：

```text
core(n, k)        = n * K_STAGES + k
Bslot(core)       = core & 1
pending(q)        = core(source(q)) + 1
future_pending(q) = pending(q + 2)
```

这使奇数K-stage的N边界也保持正确ping/pong parity。K192的BK64 fragment会折叠
`k_iter`维，因此A/B operand使用`k_atom`索引；每个4-wave group只让前128线程参与
quarter load和LDS commit。K256/BK128的编译期分支被完全裁掉，fresh ISA与多K改动前
逐字节一致，SHA256仍为`73961767...`。

N2048 fresh ISA资源和算法工作量：

| K | BK / K stages | VGPR / SGPR | LDS / scratch | MFMA | B load / B LDS read / B LDS write |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 128 / 1 | 169 / 96 | 48KiB / 0 | 1,024 | 36 / 256 / 160 |
| 192 | 64 / 3 | 171 / 96 | 32KiB / 0 | 1,536 | 101 / 384 / 223 |
| 256 | 128 / 2 | 186 / 96 | 48KiB / 0 | 2,048 | 72 / 512 / 192 |
| 384 | 128 / 3 | 192 / 96 | 48KiB / 0 | 3,072 | 108 / 768 / 224 |
| 512 | 128 / 4 | 210 / 96 | 48KiB / 0 | 4,096 | 144 / 1,024 / 256 |

K128/192/384/512在N128和N2048隔离运行中均finite；正式torch-reference
`test_acc_fly_splitk_2s_down_8x1`以N512参数化覆盖五种K并得到`5 passed`。K512最高为
210 VGPR，仍低于256硬门槛且0 scratch。当前多K工作区源码SHA256为`85a13a74...`。
各shape的fresh ISA SHA256分别为：

```text
K128  039590ebfce1f5723e18aee765e3ee134d270dd2a4681511177aeb031336ac8e
K192  f1c64a5ac95fc54b03c3e233cd9e7848d568db27e60ad2460bf4c8792e23800a
K256  739617672892eb8035e997d493298e5617c0e1ddfa5f4350011f7aee2777c7df
K384  a540995a14875f1441b98014b1de622e5f7e921dbf038db8ace6d7396d56e5bc
K512  331231503ee1b4f02be1d52e445a1596b77b026bd78044d3745638df99e11fff
```

##### 当前K256 ATT入口

当前K256 ATT的方法、唯一复算命令、七层账本、98540记录交集和N2/core1具体stage均已
收敛到[stall_analysis.md](../../../../../../../../../../flydsl/attn_4wave/tools/stall_analysis.md)。统一分析器为
[analyze_mfma_stall.py](../../../../../../../../../../flydsl/attn_4wave/tools/analyze_mfma_stall.py)，raw trace和
生成报告位于`ui_output_moe_8x1_k256_current_dispatch_16/`。本设计文档不再复制ATT
数字，避免方法或trace更新后出现两个权威版本。

统一性能测试使用B32768、N2048、TOPK8、E256、BM256、BN128、10个轮换buffer；
正序和反序各12轮，每种K共48个样本。GPU7固定1800MHz determinism、PTL
`Enabled / VECTOR,F8`、650W、NUMA off，测试结束后恢复原状态。有效工作量和吞吐为：

$$
F_{useful}=2\times32768\times8\times2048\times K,
\qquad
T_{effective}=F_{useful}/t.
$$

| K | BK | useful FLOP | 中位时延 | P25--P75 | 有效TFLOPS |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 128 | 137,438,953,472 | 0.840743ms | 0.838783--0.842133ms | 163.47 |
| 192 | 64 | 206,158,430,208 | 1.402705ms | 1.399016--1.409205ms | 146.97 |
| 256 | 128 | 274,877,906,944 | 0.738483ms | 0.729823--0.745843ms | 372.22 |
| 384 | 128 | 412,316,860,416 | 1.344445ms | 1.340965--1.346465ms | 306.68 |
| 512 | 128 | 549,755,813,888 | 1.611647ms | 1.610086--1.613047ms | 341.11 |

这些均为wall-time有效TFLOPS，不是ATT union乘roof得到的模型TFLOPS。K192使用BK64，
stage/barrier数与K384相同但每stage只有一半MFMA，因此有效TFLOPS最低；K256仍使用
专用rolling epilogue，不能直接用它与其他K的pure epilogue效率做线性比较。

同一carry源码`2e08d1fd...`随后在GPU7、PTL `Enabled / VECTOR,F8`、1800MHz、
650W、NUMA off下重新执行10-buffer A/A ABBA12。两个独立标签共48个样本，合并中位为
`0.772963ms / 355.62 useful TFLOPS`，P25/P75为`0.768243/0.951284ms`；两标签
ratio中位为1.00639且各6/12胜，说明没有代码差异，但运行中仍存在周期性慢样本。
physical输出逐bit一致、relative-L2为0。

同版本fresh ATT dispatch 16为`0.739683ms / 371.62 useful TFLOPS`，资源为
52 regular + 132 accum VGPR、112 trace SGPR、48KiB LDS、0 scratch。四个SE共生成
480个完整wave JSON，UI位于`ui_output_moe_8x1_carry_fresh_dispatch_16/`。该ATT时延
低于ABBA12中位，但仍是单次采集结果，不能替代多buffer性能结论。测试结束后GPU7已
恢复auto、PTL Disabled，NUMA balancing恢复开启。

