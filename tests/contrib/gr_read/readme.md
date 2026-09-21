# GRRead：性能与优化方法

## 1. 测试与最新性能

唯一入口：[test_gr_read.py](test_gr_read.py)。在仓库根目录运行：

```bash
python tests/contrib/gr_read/test_gr_read.py --gpu 3
python tests/contrib/gr_read/test_gr_read.py --gpu 3 --check-only
```

默认先检查全部16档的Down、Up和完整调用，再测试三个阶段的性能；可用`--rows`或`--batches`限定范围。pytest仅执行基础正确性，不启动性能矩阵。

### 测量条件

- **测量日期：2026-09-20**。AMD Instinct MI308X，gfx942，80CU；物理GPU3，PCI `0000:C8:00.0`；Torch 2.9.1／ROCm 7.2。
- BF16：X `[T,10240]`、P `[T,320]`、Y `[T,2560]`；P无行padding。非空完整调用固定Down＋Up两次launch。
- 原`cudaPerf`，**10组独立buffer、每阶段2次预热、10条样本取中位数**。准备、权重shuffle、JIT、参考和校验不计时；Total直接测量，不相加Down／Up中位数。
- 性能入口、采样前、出口要求GPU use≤5%、VRAM≤20%、PTL **Enabled／VECTOR,F8**。本轮48份门禁均通过，最大GPU use1%、VRAM9%，auto650W，无设备设置修改。
- 16档正确性全部通过，原逐元素容差保持：P `rtol=0.015625, atol=2e-5`；Y `rtol=0.01, atol=0.005`。全部**480条性能样本**保留，未过滤长尾。

每格为 **中位时延μs／有效GEMM TFLOPS**，1k＝1024行。工作量和换算为：

$$
F_D=F_U=2T\times10240\times320,\qquad
F_{Total}=4T\times10240\times320,\qquad
TFLOPS_{effective}=\frac{F}{t_{\mu s}\times10^6}.
$$

### 最新性能表

| Batch | Down | Up | Down us / TFLOPS | Up us / TFLOPS | Total us / TFLOPS |
|---:|---:|---:|---:|---:|---:|
| 1k | N2 | N8 | 136.481 / 49.171 | 76.660 / 87.541 | 196.261 / 68.387 |
| 2k | N2 | N8 | 134.800 / 99.568 | 78.881 / 170.152 | 199.541 / 134.526 |
| 4k | N1 | N4 | 163.841 / 163.839 | 144.481 / 185.794 | 293.802 / 182.733 |
| 8k | N1 | N2 | 297.521 / 180.448 | 275.581 / 194.814 | 552.662 / 194.285 |
| 10k | N1 | N2 | 324.021 / 207.113 | 279.621 / 239.999 | 561.282 / 239.127 |
| 12k | N1 | N8 | 426.281 / 188.914 | 382.961 / 210.284 | 796.183 / 202.292 |
| 16k | N1 | N8 | 550.362 / 195.097 | 532.382 / 201.686 | 1079.344 / 198.962 |
| 20k | N1 | N2 | 558.042 / 240.516 | 557.502 / 240.748 | 1110.864 / 241.646 |
| 24k | N1 | N4 | 690.962 / 233.097 | 715.643 / 225.058 | 1414.785 / 227.683 |
| 28k | N1 | N2 | 825.203 / 227.708 | 825.543 / 227.614 | 1657.486 / 226.735 |
| 30k | N1 | N2 | 831.363 / 242.164 | 830.643 / 242.374 | 1666.745 / 241.581 |
| 32k | N1 | N8 | 959.143 / 223.896 | 992.363 / 216.401 | 1958.046 / 219.350 |
| 36k | N1 | N2 | 1092.583 / 221.120 | 1096.704 / 220.289 | 2199.208 / 219.708 |
| 48k | N1 | N2 | 1371.565 / 234.858 | 1370.385 / 235.060 | 2771.510 / 232.453 |
| 60k | N1 | N2 | 1649.445 / 244.114 | 1669.966 / 241.115 | 3348.571 / 240.493 |
| 64k | N1 | N4 | 1785.165 / 240.592 | 1886.286 / 227.694 | 3702.832 / 231.983 |

数据来源：[完整结果与各阶段样本](results/readme_refresh_20260920/run/summary.json)、[原始测试日志](results/readme_refresh_20260920/benchmark.log)、[样本／门禁／地址复核](results/readme_refresh_20260920/summarize.json)。这是当前版本的一轮完整测量，不与不同场次的历史最佳值拼接，也不是优化前后的同址配对比较。

## 2. 有效优化方法

1. **先固定数值边界，再减少算术。** Down完整K10240 FP32累加后先舍入BF16，再做FP32缩放和SiLU，写BF16 P。Up完整R320归约、直接FP32 logits、stream0→1→2→3顺序FMA，最后乘0.25并使用既有整数BF16 helper。浮点FMA加偏置不能替代整数位模式舍入；“容差通过”与“逐位等价”分开验收。[数值helpers](../../../src/pyhip/ops/gr_read/flydsl/helpers.py)

2. **按实际CTA轮数独立选择Down／Up的N分片。** Down N1/N2对应320／160列，Up选择N2/N4/N8；比较$\lceil B_MN/U\rceil c_N$而不是只看单CTA工作量。Down用$B_D=\lceil T/64\rceil$，Up用$B_U=\lceil T/256\rceil$；更多分片有利于小batch填满CU，却可能增加大batch轮数和重复读取。成本是历史校准模型，不是跨设备最优保证。[选择规则](../../../src/pyhip/ops/gr_read/flydsl/common.py)

3. **让权重顺序匹配归约顺序和输出lane。** Up先将权重排列为`[H64 tile, stream, H32 half, ...]`，相邻8个H32 packet完成同一H64的四路归约；内层重排使同一lane取得连续8个BF16，适配X和16B Y store。随后N16/K32 preshuffle负责MFMA输入打包，不能混淆两层排列。变更前先用CPU标签证明索引双射，再同步修改消费地址。[准备函数](test_gr_read.py#L110)

4. **交错当前MFMA与上一packet的后处理。** 在Compute(q)计算当前logits，同时推进q−1的scale／exp／rcp／FMA／BF16打包，拆开后处理自身依赖链。当前每10条MFMA推进两个旧元素，SFU独占间隔，普通间隔控制VALU数量；实际收益取决于机器调度和生存期，不是让指令间隔形式上均匀。保留FIRST／LOOP／LAST和drain的完整覆盖。[Up流水](../../../src/pyhip/ops/gr_read/flydsl/prefill_up.py)

5. **联合调整预取深度、寄存器占用与等待。** 当前W2在Memory(q)发布W(q+1)、预读W(q+2)，少保留一个W搬运包；没有减少总W字节。任何预取或发射顺序改动都要重算W/X/Y的VMEM完成账本，检查LDS覆写和实际ISA。4＋4wave错相必须首尾闭合，barrier不能替代完成等待，VGPR下降也不自动提高占用率。

6. **按缓存行组织X协作读取，再恢复计算布局。** 同行8个lane各读16B形成X128，覆盖H64并供相邻两个H32消费；通过每wave3KiB LDS恢复MFMA布局。两个20KiB W槽＋24KiB X区共64KiB。X/P读和Y写当前使用NT、W保持default；合并请求能减少重复取读，但增加LDS工作，不能只凭HBM字节下降宣称加速。

7. **64位tile基址＋局部buffer边界，避免Host分块和P padding。** CTA入口先以64位元素偏移重设X/P/Y，descriptor只覆盖当前tile有效行。Down N1输出范围为`valid_rows*R*2`；N2保留stride R，范围为`((valid_rows-1)*R+BN)*2`，valid0为0。二维tensor避免展平动态shape超i32；rows运行时传入，工厂仅按N分片缓存。正常`flyc.compile`会执行一次，必须提供足量张量。[Down尾块处理](../../../src/pyhip/ops/gr_read/flydsl/prefill_down.py)

8. **把性能条件和归因口径固定下来。** 正确性guard view与性能原生allocation分开，记录X/P/Y相对地址；同一ELF也会对地址偏移敏感。保留长尾，用同址交错对照确认小收益。ATT只分析共同稳态窗口，区分issue-stall和completion-wait，不把跨wave stall求和当墙钟，也不把MFMA union×roof模型TFLOPS当实测。逻辑M-major或swizzle不保证物理XCD归属，需单独验证。

可复用流程已整理为项目skills：
- [grread-bf16-prefill](../../../.github/skills/grread-bf16-prefill/SKILL.md)：数值边界、权重／lane布局、N分片、X128/W2流水、精确尾块。
- [gpu-benchmark-validation](../../../.github/skills/gpu-benchmark-validation/SKILL.md)：空闲GPU与PTL、原计时器和地址条件、原始样本、ATT/PMC归因边界。