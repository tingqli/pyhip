# gfx94x/gfx950 MFMA/VALU intra与inter co-issue微基准

统一工具为
[`tools/analyze-kernel-mfma-valu-coissue.py`](tools/analyze-kernel-mfma-valu-coissue.py)。
它直接测量attention softmax所用指令的吞吐，并在一次运行中同时测量：

- **intra co-issue:**同一wave内的`MFMA + N * VALU`;
- **inter co-issue:**同一SIMD上两个wave分别执行MFMA segment和tested-op segment。

两种模式都比较anchor与N=1..4 tested-op stream的总时间,分别给出fully-hidden容量、full-coissue容量和
partial-overlap比例。

脚本自动读取当前 GPU 架构并选择 MFMA：

| 平台 | 微基准 MFMA |
|---|---|
| gfx940/gfx941/gfx942 | `v_mfma_f32_16x16x16_bf16` |
| gfx950 | `v_mfma_f32_16x16x128_f8f6f4` |

本轮正式结果来自 gfx942。gfx950 的自动选择和 JIT 路径已保留，但重构后尚未在 gfx950
实机复验，因此不能把下面的 gfx942 容量直接外推到 gfx950。

## 测量方法

默认执行1000次运行时循环,每个segment静态展开1000组,共测量1,000,000组（旧正式结果的10倍）。
计时区间使用`s_memtime`,每个配置预热一次并采集五个样本,报告中位数。

### 单指令吞吐

脚本分别测量空循环和目标指令循环：

$$
C_{\mathrm{op}}
=\frac{T_{\mathrm{op}}-T_{\mathrm{empty}}}{N_{\mathrm{outer}}N_{\mathrm{inner}}}
$$

$$
R_{\mathrm{op}}=\frac{1}{C_{\mathrm{op}}}
$$

其中$C_{\mathrm{op}}$为`cycle/inst`，$R_{\mathrm{op}}$为`inst/cycle`。`--inner-unroll`控制静态展开,
`--register-chains`控制独立寄存器组数。

### 8-byte指令的PC对齐

旧100k结果中`v_fma_f32`、`v_max3_f32`、`v_pk_add_f32`和`v_pk_mul_f32`约为5 cycles,看似违反gfx942
普通VALU约4-cycle的规律。将计算量提高10倍后仍分别为5.012/5.012/5.004/5.004,因此不是采样噪声。
最终机器码没有任何NOP:计时区间是连续1000条目标指令加每segment五条SALU循环控制。

这些指令都是8-byte VOP3/VOP3P。旧kernel首条目标指令位于$PC\bmod8=4$,每条8-byte指令都跨8-byte
边界。`--alignment-nops 1`只在开始计时前增加一条4-byte`s_nop`,把hot loop移到$PC\bmod8=0$;
NOP不在计时区间内。结果为：

| opcode | $PC\bmod8=4$ | $PC\bmod8=0$ | 修正 |
|---|---:|---:|---:|
| `v_add_f32_e64` | 5.012 | 4.016 | -0.996 |
| `v_mul_f32_e64` | 5.012 | 4.016 | -0.996 |
| `v_fma_f32` | 5.012 | 4.016 | -0.996 |
| `v_max3_f32` | 5.012 | 4.016 | -0.996 |
| `v_pk_add_f32` | 5.004 | 4.008 | -0.996 |
| `v_pk_mul_f32` | 5.004 | 4.008 | -0.996 |

同一数学运算的4-byte `v_add_f32_e32`/`v_mul_f32_e32`始终约4 cycles。PMC进一步确认这是PC对齐罚时,
不是算术吞吐：

| 1M ADD stream | `SQ_IFETCH` | active | dependency wait | issue wait | wave cycles |
|---|---:|---:|---:|---:|---:|
| e32 | 127,006 | 1,004,029 | 8,526 | 0 | 1,012,555 |
| e64,$PC\bmod8=4$ | 252,006 | 1,004,029 | 258,610 | 0 | 1,262,639 |
| e64,$PC\bmod8=0$ | 252,006 | 1,004,029 | 9,537 | 0 | 1,013,566 |

e64未对齐相对对齐版本多出的约249k wave cycles全部进入`SQ_WAIT_ANY`,active和issue wait不变。
因此canonical吞吐取不跨8-byte边界的结果;5-cycle值保留为生产ISA可能遇到的对齐敏感成本,不能再当作
FMA/MAX3/packed FP32的固有吞吐。

### intra co-issue

对每条目标指令分别生成五个 kernel：

```text
MFMA
MFMA + 1 * VALU
MFMA + 2 * VALU
MFMA + 3 * VALU
MFMA + 4 * VALU
```

MFMA 使用四组独立 accumulator，VALU 也轮换四组寄存器。对 $N=1,2,3,4$：

$$
\Delta_N=C_{\mathrm{MFMA}+N\mathrm{VALU}}-C_{\mathrm{MFMA}}
$$

旧版只用$\Delta_N\le0.25$ cycles/group判断tested-op是否被MFMA完全覆盖。该指标现在明确命名为
**fully hidden by anchor**,不能把不满足它的组合表述为“完全无法co-issue”。脚本同时计算：

$$
C_{serial}=C_{anchor}+N C_{op},\qquad
C_{full}=\max(C_{anchor},NC_{op})
$$

$$
C_{overlap}=C_{serial}-C_{measured},\qquad
R_{overlap}=\frac{C_{overlap}}{\min(C_{anchor},NC_{op})}.
$$

- $C_{measured}\approx C_{full}$:full co-issue;
- $C_{full}<C_{measured}<C_{serial}$:partial co-issue;
- $C_{measured}\approx C_{serial}$:基本无overlap。

最大full co-issue数仍要求从N=1开始连续满足,避免把噪声中的孤立点误判为容量。

这种定义直接检验：

$$
T(\mathrm{MFMA})\approx T(\mathrm{MFMA}+N\times\mathrm{VALU})
$$

不依赖 gfx942 上不存在的 `SQ_VALU_MFMA_COEXEC_CYCLES` PMC，也不需要固定 gap 或
`s_nop` 扫描。

### inter co-issue

inter kernel只启动一个512线程workgroup（8 waves）。gfx942把wave i和wave i+4放到同一SIMD。
kernel使用条件barrier建立并维持一个phase偏移：

```text
wave 4..7: extra entry barrier
all waves: common entry barrier

repeat 100 times:
  1000 x MFMA
  barrier
  1000 x (N * tested-op)
  barrier

wave 0..3: extra drain barrier
```

第一次物理barrier由低4 waves的common barrier和高4 waves的extra barrier配对。steady state中,
同一SIMD上的wave 0/4恰好反相:一条wave执行MFMA segment时,另一条执行tested-op segment;下一phase
交换角色。入口/出口条件barrier保证barrier总数配平且不会死锁。

每个wave独立写回elapsed和SIMD ID。host逐样本强制验证：

```text
SIMD(wave i) == SIMD(wave i+4), i=0..3
```

不同launch可以映射到不同SIMD排列,但4对wave必须始终同SIMD。本轮全部正式样本零配对错误,
8-wave elapsed离散度小于0.048%。inter结果按
$2N_{\mathrm{outer}}N_{\mathrm{inner}}$组归一,因为每个runtime loop包含两个反相phase。正式结果的
1000次循环把pipeline fill/drain误差进一步压低。

### throughput-predicted capacity

gfx942 上 BF16 MFMA 的实测 busy 时间为 16 cycles，普通 VALU 约在 MFMA 开始后
4 cycles 进入发射，因此可用于隐藏 VALU 的 shadow 约为 12 cycles。脚本同时给出仅由吞吐
推导的容量：

$$
N_{\mathrm{theory}}
=\min\left(4,
\left\lfloor
\frac{C_{\mathrm{MFMA}}-4+\epsilon}{C_{\mathrm{op}}}
\right\rfloor\right),
\qquad \epsilon=0.25
$$

throughput-predicted capacity只表示时间窗口能容纳多少条指令,没有考虑硬件never-coissue规则。
它与intra/inter co-issue不一致本身就是重要结果。

### EXP anchor

为直接回答EXP与其他VALU能否co-issue,脚本新增两组kernel：

```text
intra: EXP + N * tested-op
inter: wave 0..3执行EXP segment，wave 4..7执行tested-op segment，然后交换角色
```

EXP使用独立`exp_src/exp_dst`寄存器,tested-op使用原有四组寄存器,两者没有RAW依赖。测量参数、
8-wave条件barrier协议和N=0..4归一方法与MFMA anchor完全相同。

## gfx942正式intra/inter结果

测试环境为gfx942,参数为`1000 * 1000`组、五个样本、一个预热样本、容差0.25 cycles/group。
8-byte指令按每opcode选择不跨8-byte边界的hot-loop对齐。
完整JSON位于：

```text
/tmp/coissue-canonical-10x-gfx942.json
```

仓库内可追踪的精简摘要为
[`data/mfma-valu-intra-inter-coissue-gfx942.json`](data/mfma-valu-intra-inter-coissue-gfx942.json)。

MFMA、EXP和普通ALU的三元组合及“双MFMA一组”顺序实测保存在
[`data/mfma-exp-alu-bundle-gfx942.json`](data/mfma-exp-alu-bundle-gfx942.json)。关键结果是
`MFMA -> 3 ALU -> MFMA -> EXP`约36.053 cycles/group,相对同顺序0 ALU的36.040只增加约0.013 cycle；
`MFMA -> MFMA -> 3 ALU -> EXP`则为48.052 cycles/group。

| Opcode | cycle/inst | hidden I/I | full I/I | intra $\Delta_{1..4}$ | inter $\Delta_{1..4}$ |
|---|---:|---:|---:|---|---|
| `v_add_f32` | 4.012 | 3/3 | 3/3 | .020/.024/.028/4.028 | -.014/-.012/-.010/3.966 |
| `v_sub_f32` | 4.012 | 3/3 | 3/3 | .020/.024/.028/4.028 | -.014/-.012/-.010/3.966 |
| `v_mul_f32` | 4.012 | 3/3 | 3/3 | .020/.024/.028/4.029 | -.014/-.012/-.010/3.966 |
| `v_fma_f32` | **4.016** | **3/3** | **3/3** | .020/.025/.029/4.029 | -.014/-.011/-.009/4.968 |
| `v_fmac_f32` | 4.012 | 3/3 | 3/3 | .020/.024/.028/4.028 | -.014/-.012/-.010/3.966 |
| `v_max_f32` | 4.012 | 3/3 | 3/3 | .020/.024/.028/4.028 | -.014/-.012/-.010/3.966 |
| `v_max3_f32` | **4.016** | **3/3** | **3/3** | .020/.024/.029/4.029 | -.014/-.011/-.008/4.968 |
| `v_exp_f32` | 16.000 | 0/0 | 0/0 | 4.028/20.028/36.028/52.028 | 3.990/19.966/35.966/51.966 |
| `v_rcp_f32` | 16.000 | 0/0 | 0/0 | 4.028/20.028/36.028/52.028 | 3.990/19.966/35.966/51.966 |
| `v_pk_add_f32` | **4.008** | 0/0 | 0/0 | 12.012/16.012/20.013/24.013 | 4.964/9.964/14.964/19.964 |
| `v_pk_mul_f32` | **4.008** | 0/0 | 0/0 | 12.012/16.012/20.013/24.013 | 4.964/9.964/14.964/19.965 |
| `v_cmp_gt_f32` | 4.016 | 3/3 | 3/3 | .020/.024/.028/4.028 | .002/.005/.007/3.967 |
| `v_cndmask_b32` | 4.008 | 3/3 | 3/3 | .008/.016/.024/4.024 | .002/.004/.006/3.966 |
| `v_add_u32` | 4.008 | 3/3 | 3/3 | .008/.016/.024/4.024 | .003/.010/.006/3.966 |
| `v_perm_b32` | 4.012 | 3/3 | 3/3 | .008/.017/.025/4.025 | .002/.005/.008/4.963 |

`I/I`表示intra/inter。`hidden`要求组合时间不超过MFMA baseline;`full`只要求组合时间不超过两条
stream中较慢者。

结果可分为五类：

1. 普通4-cycle scalar VALU的intra/inter fully-hidden和full-coissue容量都是3。
2. `v_fma_f32`和`v_max3_f32`的canonical吞吐也是约4 cycles,fully-hidden/full-coissue容量均为
  intra 3/inter 3。旧intra容量2来自8-byte指令跨界的5-cycle对齐罚时。
3. `v_exp_f32`和`v_rcp_f32`没有被MFMA fully hidden,但不是“完全无法co-issue”。一条MFMA和一条EXP
  独立串行应为$16.026+16.000=32.026$ cycles,实测仅20.056 cycles,即重叠11.970 cycles,
  overlap ratio为**74.8%**。这是明显的partial co-issue。
4. packed FP32 ADD/MUL本身canonical吞吐约4 cycles,但MFMA intra下首条仍增加约12 cycles,inter约
  5 cycles/条。吞吐修正没有改变其MFMA fully-hidden/full-coissue容量0,说明这是pipeline冲突而非慢吞吐。
5. inter不是所有VALU容量都无限增加:N=4时普通scalar仍增加约4 cycles/group,容量仍为3。

因此,attention优化应优先把独立scalar VALU、地址计算、比较和`v_perm_b32`安排进MFMA窗口。
FMA/MAX3和普通scalar一样可在同wave MFMA shadow中容纳3条;packed FP32仍不能被MFMA fully hidden。
实际kernel还必须计入
barrier、数据依赖、VGPR live range和occupancy;本微基准只测steady-state pipeline容量。

## EXP与其他VALU的co-issue

下表为N=1正式结果。`overlap`按相对完全串行时间节省的周期计算：

| tested-op | op cycle | EXP+op intra | overlap | EXP+op inter | overlap |
|---|---:|---:|---:|---:|---:|
| `v_add_f32` | 4.012 | 20.036 | 0.3% | 20.008 | 1.2% |
| `v_sub_f32` | 4.012 | 20.036 | 0.3% | 20.008 | 1.2% |
| `v_mul_f32` | 4.012 | 20.036 | 0.3% | 20.008 | 1.2% |
| `v_fma_f32` | 4.016 | 20.036 | 0.8% | 20.511 | 0.0% |
| `v_fmac_f32` | 4.012 | 20.036 | 0.3% | 20.008 | 1.2% |
| `v_max_f32` | 4.012 | 20.036 | 0.3% | 20.008 | 1.2% |
| `v_max3_f32` | 4.016 | 20.036 | 0.8% | 20.511 | 0.0% |
| `v_exp_f32` | 16.000 | 32.036 | 0.0% | 32.008 | 0.2% |
| `v_rcp_f32` | 16.000 | 32.036 | 0.0% | 32.008 | 0.2% |
| `v_pk_add_f32` | 4.008 | 20.052 | 0.2% | 20.511 | 0.0% |
| `v_pk_mul_f32` | 4.008 | 20.052 | 0.2% | 20.510 | 0.0% |
| `v_cmp_gt_f32` | 4.016 | 20.052 | 0.0% | 20.008 | 1.3% |
| `v_cndmask_b32` | 4.008 | 20.052 | 0.0% | 20.008 | 1.1% |
| `v_add_u32` | 4.008 | 20.052 | 0.0% | 20.009 | 1.1% |
| `v_perm_b32` | 4.012 | 20.052 | 0.0% | 20.506 | 0.0% |

结论：

1. EXP与普通4-cycle ADD/SUB/MUL等基本串行,总时间约$16+4=20$ cycles,overlap接近0。
2. 对齐修正后,EXP与FMA/MAX3/packed FP32也基本串行,overlap仅0–0.8%。旧约20%来自tested-op吞吐
  被跨界PC对齐误报为5 cycles,并不是真实overlap。
3. EXP与EXP或RCP约为$16+16=32$ cycles,基本无overlap。
4. inter没有改善EXP+FMA/MAX3/packed FP32;部分组合还增加约0.5 cycle。
5. 这些是pairwise测试。不能由`MFMA+EXP`和`EXP+FMA`分别有partial overlap,推导三者能同时以相同比例overlap。

## 使用方法

运行全部指令：

```bash
cd /root/workspace/luocheng/pyhip
HIP_VISIBLE_DEVICES=2 \
PYHIP_CACHE_DIR=/tmp/pyhip-coissue-all \
python3 tests/flydsl/attn_4wave/tools/analyze-kernel-mfma-valu-coissue.py \
  --ops all \
  --outer-loops 1000 \
  --inner-unroll 1000 \
  --samples 5 \
  --warmup 1 \
  --tolerance 0.25 \
  --json /tmp/attn-valu-intra-inter-coissue.json
```

只测部分指令：

```bash
HIP_VISIBLE_DEVICES=2 \
python3 tests/flydsl/attn_4wave/tools/analyze-kernel-mfma-valu-coissue.py \
  --ops v_add_f32,v_fma_f32,v_exp_f32,v_pk_mul_f32
```

JSON 为每条指令保存：

- 空循环和目标循环的原始中位数、全部样本；
- `throughput_cycles_per_instruction` 和 `throughput_instructions_per_cycle`；
- `mfma_baseline_cycles_per_instruction`；
- `throughput_predicted_valu`、`max_intra_coissue`和`max_inter_coissue`;
- `intra_coissue`和`inter_coissue`中N=0..4的总cycles、fully-hidden/full/partial co-issue判定、
  overlap cycles和overlap ratio;
- `exp_intra_coissue`和`exp_inter_coissue`中的EXP anchor测试结果;
- inter每个wave的中位elapsed、最大wave离散度和每个样本的SIMD映射。

## 增加测试指令

tested-op集中在脚本顶部的`VALU_TESTS`列表。若已有寄存器字段能够表达操作数,只需增加
一个 `(opcode, lambda)`：

```python
VALU_TESTS = [
    # ...
    ("v_new_op", lambda j, r, i: j.v_new_op(r["dst"][i], r["src0"][i])),
]
```

若新指令需要不同类型或数量的操作数，再在`make_registers()`中增加对应寄存器。
对8-byte指令必须同时用`--alignment-nops 0/1 --throughput-only`测两种PC半字对齐;canonical吞吐取
不跨8-byte边界的结果,生产kernel分析则应使用其实际PC对齐。
其余throughput、intra/inter co-issue、CLI和JSON逻辑无需修改。

## gfx950 历史交叉验证

重构前的 gfx950 实验使用 PMC 和等工作量控制组，结论与新方法要验证的问题一致：

| gfx950 指标 | 两条 scalar `v_fmac_f32` | 一条 packed `v_pk_fma_f32` |
|---|---:|---:|
| Co-exec cycles/MFMA | 7.999 | 0.000 |
| Device cycles/MFMA | 285.022 | 317.018 |

生产 kernel 中，packed FMA 版本的 median 从 509.645 us 回退到 579.166 us，MFMA 数量和
busy 时间不变，而 co-exec/MFMA 从 16.920 降到 2.465。该历史数据说明 packed FP32 的
never-coissue 问题同时存在于 gfx950，但仍需用当前脚本在 gfx950 上重新采样，才能报告与上表
同口径的吞吐和 `N=1..4` 容量。

## 参考资料

- [ROCm Compute Profiler MFMA pipeline](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/conceptual/cdna/pipeline-descriptions.html#matrix-fused-multiply-add-mfma)
- [LLVM `SIPreEmitPeephole.cpp`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/SIPreEmitPeephole.cpp)
- [attention 优化记录](attn_gemm_optimization.md)
