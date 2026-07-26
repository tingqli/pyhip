# gfx94x/gfx950 MFMA 与 VALU 共发微基准

统一工具为
[`archive/gemm/analyze-kernel-mfma-valu-coissue.py`](../archive/gemm/analyze-kernel-mfma-valu-coissue.py)。
它直接测量 attention softmax 所用指令的吞吐，并比较纯 MFMA 与
`MFMA + N * VALU` 的总时间，给出每条 MFMA 能完全隐藏的 VALU 数量。

脚本自动读取当前 GPU 架构并选择 MFMA：

| 平台 | 微基准 MFMA |
|---|---|
| gfx940/gfx941/gfx942 | `v_mfma_f32_16x16x16_bf16` |
| gfx950 | `v_mfma_f32_16x16x128_f8f6f4` |

本轮正式结果来自 gfx942。gfx950 的自动选择和 JIT 路径已保留，但重构后尚未在 gfx950
实机复验，因此不能把下面的 gfx942 容量直接外推到 gfx950。

## 测量方法

每个 kernel 只启动一个 64 线程 workgroup。默认执行 100 次运行时循环，每轮静态展开
1000 组，共测量 100,000 组。计时区间使用 `s_memtime`，每个配置预热一次并采集五个样本，
报告中位数。

### 单指令吞吐

脚本分别测量空循环和目标指令循环：

$$
C_{\mathrm{op}}
=\frac{T_{\mathrm{op}}-T_{\mathrm{empty}}}{100\times1000}
$$

$$
R_{\mathrm{op}}=\frac{1}{C_{\mathrm{op}}}
$$

其中 $C_{\mathrm{op}}$ 为 `cycle/inst`，$R_{\mathrm{op}}$ 为 `inst/cycle`。
四组独立寄存器循环使用，避免把单一寄存器的 RAW latency 当成指令吞吐。

### 实测共发数

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

当 $\Delta_N\le 0.25$ cycles/group 时，判定这 $N$ 条 VALU 被 MFMA 完全隐藏。
最终共发数要求从 $N=1$ 开始连续满足，避免把噪声中的孤立点误判为容量。

这种定义直接检验：

$$
T(\mathrm{MFMA})\approx T(\mathrm{MFMA}+N\times\mathrm{VALU})
$$

不依赖 gfx942 上不存在的 `SQ_VALU_MFMA_COEXEC_CYCLES` PMC，也不需要固定 gap 或
`s_nop` 扫描。

### 吞吐理论数

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

理论数只表示时间窗口能容纳多少条指令，没有考虑硬件 never-coissue 规则。理论数与实测数
不一致本身就是重要结果。

## gfx942 正式结果

测试环境为 gfx942，参数为 `100 * 1000` 组、五个样本、一个预热样本、容差 0.25
cycles/group。完整 JSON 位于本次测试机的
`/tmp/attn-valu-throughput-coissue-gfx942-final.json`。

| Opcode | cycle/inst | inst/cycle | 吞吐理论数 | 实测共发数 | $\Delta_{1..4}$ cycles/group |
|---|---:|---:|---:|---:|---|
| `v_add_f32` | 4.0132 | 0.2492 | 3 | 3 | +0.022, +0.026, +0.032, +4.033 |
| `v_sub_f32` | 4.0129 | 0.2492 | 3 | 3 | +0.022, +0.027, +0.032, +4.032 |
| `v_mul_f32` | 4.0129 | 0.2492 | 3 | 3 | +0.022, +0.026, +0.032, +4.033 |
| `v_fma_f32` | 5.0142 | 0.1994 | 2 | 2 | +0.023, +0.030, +4.033, +9.035 |
| `v_fmac_f32` | 4.0128 | 0.2492 | 3 | 3 | +0.023, +0.026, +0.031, +4.033 |
| `v_max_f32` | 4.0131 | 0.2492 | 3 | 3 | +0.022, +0.027, +0.032, +4.033 |
| `v_max3_f32` | 5.0142 | 0.1994 | 2 | 2 | +0.023, +0.030, +4.032, +9.034 |
| `v_exp_f32` | 16.0000 | 0.0625 | 0 | 0 | +4.030, +20.029, +36.028, +52.028 |
| `v_rcp_f32` | 16.0000 | 0.0625 | 0 | 0 | +4.030, +20.029, +36.029, +52.029 |
| `v_pk_add_f32` | 5.0062 | 0.1998 | 2 | 0 | +12.014, +16.016, +20.019, +25.021 |
| `v_pk_mul_f32` | 5.0062 | 0.1998 | 2 | 0 | +12.014, +16.016, +20.019, +25.022 |
| `v_cmp_gt_f32` | 4.0169 | 0.2489 | 3 | 3 | +0.022, +0.028, +0.032, +4.033 |
| `v_cndmask_b32` | 4.0090 | 0.2494 | 3 | 3 | +0.010, +0.018, +0.027, +4.029 |
| `v_add_u32` | 4.0089 | 0.2494 | 3 | 3 | +0.010, +0.018, +0.028, +4.029 |
| `v_perm_b32` | 4.0142 | 0.2491 | 3 | 3 | +0.010, +0.020, +0.032, +4.034 |

结果可分为四类：

1. 普通 4-cycle scalar VALU 的理论数和实测数都是 3，符合 $(16-4)/4=3$。
2. `v_fma_f32` 和 `v_max3_f32` 约为 5 cycles，理论和实测都只能隐藏 2 条。
3. `v_exp_f32` 和 `v_rcp_f32` 为 16-cycle TRANS，理论和实测都是 0。
4. packed FP32 ADD/MUL 的吞吐约为 5 cycles，时间窗口理论上可容纳 2 条，但实测一条也
   不能隐藏。这证明限制来自 packed FP32 与 MFMA 的硬件发射冲突，而不是裸吞吐。

因此，attention 优化应优先把独立 scalar VALU、地址计算、比较和 `v_perm_b32` 安排进 MFMA
shadow；TRANS 无法靠重排获得共发，packed FP32 则应移出 MFMA busy window 或在收益允许时拆成
scalar 指令。实际 kernel 仍需同时检查数据依赖、VGPR live range 和 occupancy。

## 使用方法

运行全部指令：

```bash
cd /root/workspace/luocheng/pyhip
HIP_VISIBLE_DEVICES=2 \
PYHIP_CACHE_DIR=/tmp/pyhip-coissue-all \
python3 archive/gemm/analyze-kernel-mfma-valu-coissue.py \
  --ops all \
  --outer-loops 100 \
  --samples 5 \
  --warmup 1 \
  --tolerance 0.25 \
  --json /tmp/attn-valu-throughput-coissue.json
```

只测部分指令：

```bash
HIP_VISIBLE_DEVICES=2 \
python3 archive/gemm/analyze-kernel-mfma-valu-coissue.py \
  --ops v_add_f32,v_fma_f32,v_exp_f32,v_pk_mul_f32
```

JSON 为每条指令保存：

- 空循环和目标循环的原始中位数、全部样本；
- `throughput_cycles_per_instruction` 和 `throughput_instructions_per_cycle`；
- `mfma_baseline_cycles_per_instruction`；
- `throughput_predicted_valu` 和 `max_fully_hidden_valu`；
- $N=0..4$ 的总 cycles、cycles/group、相对增量和完全隐藏判定。

## 增加测试指令

待测试指令集中在脚本顶部的 `VALU_TESTS` 列表。若已有寄存器字段能够表达操作数，只需增加
一个 `(opcode, lambda)`：

```python
VALU_TESTS = [
    # ...
    ("v_new_op", lambda j, r, i: j.v_new_op(r["dst"][i], r["src0"][i])),
]
```

若新指令需要不同类型或数量的操作数，再在 `make_registers()` 中增加对应的四组独立寄存器。
其余吞吐、共发、CLI 和 JSON 逻辑无需修改。

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
