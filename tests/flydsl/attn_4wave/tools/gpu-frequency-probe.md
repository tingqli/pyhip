# GPU降频分析：简单Pattern与实际Kernel

[`probe-gpu-frequency.py`](probe-gpu-frequency.py) 使用PyHIP JIT生成可控负载，并可离线分析rocprofv3
ATT。本文分两部分：先用简单pattern建立经验边界，再用DPM遥测、ATT和受控code-object对照解释
实际H3 kernel。

## 结论

### 基于简单pattern

1. 密集MFMA会触发非PPT主导的降频：纯MFMA在仅242 W时，3秒整窗SCLK已降至约1132 MHz。
2. 纯MFMA约150 ms后离开1.8 GHz，但该状态会跨无间隙kernel边界累积；150 ms不是单kernel上限。
3. BF16 16x16 pattern在约6.6 scalar VALU/MFMA处恢复高平均频率，但该比例不能跨opcode复用。
4. 换成production同款32x32 MFMA后，FP8和BF16都在约60 cycles/MFMA附近跨越1800 MHz边界。

| MFMA opcode | 降频侧已验证至 | 高频侧已验证自 |
|---|---:|---:|
| `v_mfma_f32_32x32x16_fp8_fp8` | 60.356 cycles/MFMA | 60.982 cycles/MFMA |
| `v_mfma_f32_32x32x8_bf16` | 59.718 cycles/MFMA | 60.978 cycles/MFMA |

### 基于实际kernel

| 精度 | 实现 | DPM结果 | cycles/MFMA | 结论 |
|---|---|---|---:|---|
| FP8 | Triton | 不循环 | 105.879 | MFMA时间密度低，保持高频 |
| FP8 | ASM MI308 | 不循环 | 63.450 | 位于FP8 pattern高频侧 |
| FP8 | ASM MI300 | 约1 s循环 | 49.028 | 位于FP8 pattern降频侧 |
| FP8 | FlyDSL 8-wave | 约1 s循环 | 53.870 | 位于FP8 pattern降频侧 |
| FP8 | FlyDSL 4-wave | 约1 s循环 | 54.468 | 位于FP8 pattern降频侧 |
| BF16 | Triton | 不循环 | 62.303 | 位于BF16 pattern高频侧 |
| BF16 | ASM MI308 | 不循环 | 54.241 | 密度会误判；高stall稀释有效执行压力 |
| BF16 | ASM MI300 | 约1 s循环 | 38.959 | stall更低、有效issue更密集 |

除BF16 ASM MI308外，实际kernel结果都可由同opcode的`cycles/MFMA`边界直接解释。BF16 ASM
MI308需要结合`stall/MFMA`、issue density和MI300 `.co`受控对照，结论仅标为机制候选。

## 第一部分：基于简单pattern的降频推导

推导路径为：区分MFMA降频与PPT降频，测量持续时间，检验VALU比例，最后使用production同款
opcode得到可与实际kernel比较的密度边界。

### 1. 指标与判定

kernel同时读取固定100 MHz的`s_memrealtime`和随shader clock累加的`s_memtime`：

$$
f_{\mathrm{SCLK}} = 100\,\mathrm{MHz}
\frac{\Delta\mathrm{s\_memtime}}{\Delta\mathrm{s\_memrealtime}}.
$$

含MFMA的pattern还报告aggregate SIMD `cycles/MFMA`：

$$
\mathrm{cycles/MFMA} =
\frac{\text{该SIMD的采样timeline span}}{\text{该SIMD上的MFMA总数}}.
$$

数值越小表示MFMA时间密度越高。正式判定使用DPM `auto`、空闲GPU、至少1秒窗口和完整SCLK/PPT
轨迹；1 ms结果只用于观察启动瞬态。

### 2. 区分MFMA与PPT降频

以下为MI308X上的3秒端点。`effective SCLK`是完整窗口双时钟平均，`steady SCLK`是后半段sysfs
中位数。

| pattern | effective SCLK | steady SCLK | PPT | 推导 |
|---|---:|---:|---:|---|
| 纯MFMA | 1131.5 MHz | 1014 MHz | 242 W | 非PPT主导降频 |
| MFMA + 3 VALU | 1131.6 MHz | 1011 MHz | 256 W | 少量可共发VALU不改变档位 |
| MFMA + EXP | 1342.7 MHz | 1210 MHz | 265 W | MFMA仍触发降频 |
| 纯EXP | 1834.0 MHz | 1853 MHz | 296 W | 基本不降频 |
| 纯VALU | 1837.2 MHz | 1850 MHz | 316 W | 基本不降频 |
| 纯HBM load | 1592.0 MHz | 1588 MHz | 650 W | PPT上限主导降频 |

因此，密集MFMA与纯HBM load是两类不同机制：前者低功耗但低频，后者触及650 W PPT后降频。

### 3. 约150 ms高频平台

kernel内timeline以1800 MHz为阈值。两种分辨率都显示纯BF16 MFMA在约150 ms后离开高频平台：

| timeline bin | 三次离开1.8 GHz的时间 | 中位数 |
|---:|---:|---:|
| 0.5 ms | 149.120 / 147.614 / 149.122 ms | 149.120 ms |
| 0.25 ms | 150.895 / 151.135 / 152.252 ms | 151.135 ms |

但`4 x 100 ms`无间隙纯MFMA dispatch train显示该状态跨kernel边界累积：

| dispatch | 累计时间 | SCLK中位数 [三次范围] |
|---:|---:|---:|
| 0 | 100 ms | 1833.0 [1702.8, 1833.4] MHz |
| 1 | 200 ms | 1471.5 [1468.7, 1587.7] MHz |
| 2 | 300 ms | 1000.0 [999.8, 1000.0] MHz |
| 3 | 400 ms | 999.9 [999.8, 1000.2] MHz |

**结论**：150 ms描述持续密集MFMA状态，不是单kernel时长规则。

### 4. 约6.6 VALU/MFMA规则

简单pattern连续执行$M$条`v_mfma_f32_16x16x16_bf16`，再执行$N$条`v_fmac_f32`。三种burst
长度的1800 MHz整窗边界为：

| MFMA burst | 最后失败点 | 首个通过点 |
|---:|---:|---:|
| 32 | 6.59375 VALU/MFMA | 6.625 VALU/MFMA |
| 64 | 6.5625 VALU/MFMA | 6.578125 VALU/MFMA |
| 128 | 6.5 VALU/MFMA | 6.5625 VALU/MFMA |

约6.6:1能恢复3秒平均频率，但更保守的后半段p10标准需要约10.5:1。

**结论**：6.6:1只适用于该BF16 16x16 opcode、scalar-FMA filler和launch配置，不是架构常数。

### 5. Production同款MFMA边界

为得到可与实际kernel比较的边界，微探针改用production同款MFMA，固定16条MFMA后接scalar
FMA；测试使用2 blocks/CU，即每SIMD两个resident wave，3秒、三次重复。

| 精度 | 最后失败点 | 首个通过点 | 经验边界 |
|---|---|---|---:|
| FP8 | 13.25 VALU/MFMA；60.356 cycles/MFMA；1792.6 MHz | 13.50 VALU/MFMA；60.982 cycles/MFMA；1805.3 MHz | 60.356--60.982 |
| BF16 | 13.00 VALU/MFMA；59.657 [59.620, 59.718] cycles/MFMA；1778.6 MHz | 13.50 VALU/MFMA；60.989 [60.978, 61.029] cycles/MFMA；1805.2 MHz | 59.718--60.978 |

由此得到最终pattern判据：与实际kernel比较时，应使用同opcode、同resident-wave口径的
`cycles/MFMA`，而不是固定VALU比例。FP8与BF16边界都接近60只是本机实测结果，不能视为跨opcode
架构常数。

## 第二部分：基于实际kernel的降频推导

实际kernel按三步分析：DPM遥测确认是否循环，ATT与同opcode的pattern边界比较；若两者矛盾，再用
受控code-object对照检查stall和有效issue密度。

### 1. FP8实际kernel

#### 观察

| 实现 | DPM结果 | cycles/MFMA | scalar VALU/MFMA | 全VALU/MFMA | max MFMA run |
|---|---|---:|---:|---:|---:|
| Triton | 不循环 | 105.879 | 9.829 | 13.923 | 2 |
| ASM MI308 | 不循环 | 63.450 | 1.674 | 4.658 | 16 |
| ASM MI300 | 约1 s循环 | 49.028 | 3.611 | 5.612 | 16 |
| FlyDSL 8-wave | 约1 s循环 | 53.870 | 5.393 | 6.393 | 5 |
| FlyDSL 4-wave | 约1 s循环 | 54.468 | 5.453 | 6.516 | 5 |

#### 推导

1. **排除150 ms**：ASM MI308单dispatch为111.292 ms，但相邻host gap仅0.095 ms；限制器状态不会
   被kernel边界重置。
2. **排除6.6和静态burst**：ASM MI308的scalar/full VALU比例只有1.674/4.658，却不循环；MI308与
   MI300的最长MFMA run同为16，DPM行为仍相反。
3. **使用FP8 exact-op边界**：MI300/FlyDSL的49.028--54.468均低于60.356，全部循环；MI308的
   63.450和Triton的105.879均高于60.982，均不循环。

> FP8五个实际kernel被同opcode的`cycles/MFMA`边界完全分开。Triton和ASM MI308保持高频的共同原因
> 是MFMA时间密度较低，而不是kernel边界、固定VALU比例或静态MFMA burst长度。

### 2. BF16实际kernel

#### 观察

| 实现 | DPM结果 | 平均耗时 | cycles/MFMA | stall/MFMA | issue density |
|---|---|---:|---:|---:|---:|
| Triton | 不循环 | 191.049 ms | 62.303 | 78.699 | 0.073208 |
| ASM MI308 | 不循环 | 191.377 ms | 54.241 | 70.764 | 0.072331 |
| ASM MI300 | 约1 s循环 | 168.773 ms | 38.959 | 34.166 | 0.118643 |

#### 推导

1. **排除150 ms和6.6**：Triton/MI308单次都约191 ms，scalar/full VALU比例分别只有
   5.265/6.282和2.555/4.539；两条局部规则都不能解释它们保持高频。
2. **Triton直接由密度解释**：62.303高于BF16 pattern高频侧边界60.978，DPM也不循环。
3. **MI308是密度例外**：54.241落在pattern降频侧，却不循环，因此需要检查实际issue行为。

MI308与MI300的symbol、输入、launch、64 KiB LDS、SGPR/VGPR/AGPR和6912个occupancy event均
相同，只替换`.co`：

| 指标 | MI308 | MI300 | MI300相对变化 |
|---|---:|---:|---:|
| cycles/MFMA | 54.241 | 38.959 | -28.17% |
| stall/MFMA | 70.764 | 34.166 | -51.72% |
| issue density | 0.072331 | 0.118643 | +64.03% |

MI308的大量周期停在MFMA、barrier和依赖stall上，持续有效issue与切换活动较低。工具将其标记为
`stall_diluted_candidate`：这是受控对照支持的机制候选，不是硬件公布阈值。

> Triton BF16因MFMA时间密度较低而保持高频；ASM MI308 BF16则由code-object调度造成的高stall
> 稀释持续有效执行压力。两者都不能由单次时长或固定VALU比例解释。

### ATT有效性

DPM遥测用于证明是否出现频率循环；ATT只用于解释执行机制。所有纳入分析的wave均满足
`num_stitched == num_insts`，capture log没有`Stitch Incomplete`、`Wave incomplete`、cutoff或
parser mismatch。BF16 trace使用约2 GiB buffer；首个被截断的512 MiB Triton trace已作废。

## 附录：复现与证据

运行前确认目标GPU空闲且DPM为`auto`，以下命令从仓库根目录执行。

### Exact-op密度扫描

```bash
HIP_VISIBLE_DEVICES=4 python3 -B tests/flydsl/attn_4wave/tools/probe-gpu-frequency.py \
  --workloads fp8_mfma_valu_burst --duration-ms 3000 --repeats 3 \
  --blocks-per-cu 2 --mfma-burst 16 --fp8-valu-scan 212,216 \
  --sample-interval-ms 10 --cooldown-ms 3000 \
  --json /tmp/gpu-frequency-fp8-density-boundary.json

HIP_VISIBLE_DEVICES=4 python3 -B tests/flydsl/attn_4wave/tools/probe-gpu-frequency.py \
  --workloads bf16_32 --duration-ms 3000 --repeats 3 \
  --blocks-per-cu 2 --mfma-burst 16 --bf16-valu-scan 208,216 \
  --sample-interval-ms 10 --cooldown-ms 3000 \
  --json /tmp/gpu-frequency-bf16-density-boundary.json
```

### Production ATT离线归类

```bash
python3 -B tests/flydsl/attn_4wave/tools/probe-gpu-frequency.py \
  --att-traces triton=/tmp/h3-att-triton,mi308=/tmp/h3-att-mi308,\
mi300=/tmp/h3-att-mi300,flydsl8=/tmp/h3-att-flydsl8-1simd,\
flydsl4=/tmp/h3-att-flydsl4-1simd \
  --json /tmp/h3-fp8-att-validation.json

python3 -B tests/flydsl/attn_4wave/tools/probe-gpu-frequency.py \
  --att-traces triton_bf16=/tmp/h3-att-bf16-triton,\
asm_mi308_bf16=/tmp/h3-att-bf16-mi308,asm_mi300_bf16=/tmp/h3-att-bf16-mi300 \
  --json /tmp/h3-bf16-att-validation.json
```

### 机器可读证据

- [FP8分析与源数据哈希](../../../../artifacts/h3-fp8-frequency-rule/analysis.json)：dispatch train、
  exact-op完整八点扫描和五项production ATT分类。
- [BF16分析与源数据哈希](../../../../artifacts/h3-bf16-frequency-rule/analysis.json)：exact-op边界、
  三项production ATT、受控资源身份和70次auto-DPM汇总。

所有边界均限于当前MI308X、软件栈、温度、occupancy、filler和DPM状态，不是架构规格。
