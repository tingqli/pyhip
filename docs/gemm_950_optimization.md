# GFX950 FlyDSL GEMM 优化与测试

本文只记录当前`tests/flydsl/test_gemm.py::compile_gemm_950`仍在使用的优化、可复现测试
方法和最终数据。历史失败方案、外部实现对照和已淘汰阶段数据不属于本文。

## 1. 测试范围

- GPU：AMD CDNA4 `gfx950`，测试进程固定`HIP_VISIBLE_DEVICES=0`。
- Kernel：4-wave与8-wave GFX950 GEMM。
- 输入/输出：BF16→BF16；FP8 E4M3FN→BF16。
- 输出tile：`128x128`、`128x256`、`256x128`、`256x256`。
- 默认性能形状：`M=N=K=4096`。
- K sweep：`K=1024,2048,4096,8192,16384`，固定`M=N=4096`与`256x256` tile。
- 本地对照：pyhip JIT 8-wave FP8，覆盖preshuffle与row-major B布局。
- 工具链：PyTorch `2.11.0+gitd0c8b1f`，FlyDSL `0.2.4`。

## 2. 可复现测试方法

### 2.1 环境

```bash
cd /host_lc/pyhip-gemm
source .venv/bin/activate
export HIP_VISIBLE_DEVICES=0
export FLYDSL_RUNTIME_ENABLE_CACHE=0
export PYTHONPATH=/host_lc/FlyDSL/build-fly/python_packages:/host_lc/pyhip-gemm
```

性能测试前必须确认目标GPU连续两次空闲：

```bash
rocm-smi -d 0 \
  --showuse --showmemuse --showmeminfo vram \
  --showtemp --showclocks --showpower --showpids
```

准入条件：连续两次GPU use为0%、VRAM activity为0%、没有本地竞争计算进程，并且可用
显存足以运行测试；若有外部显存驻留，必须记录占用和可用容量。本轮全量重测期间外部
任务驻留89% VRAM，但开始前连续采样及每组测试后均为0% use/activity，仍有约32 GiB
可用，且没有本地benchmark/profiler进程。tile、默认JIT和K sweep的最大Q1/Q3跨度分别
为1.75%、0.35%和2.58%。

### 2.2 功能测试

运行pytest、完整长循环矩阵和两K-tile prologue/tail矩阵：

```bash
python3 tests/flydsl/compare_gemm_950.py validate \
  --full-size 4096 --full-k 4096 \
  --full-launches 10 --two-tile-launches 20 \
  --csv /tmp/gemm950_functional.csv
```

该命令执行：

1. `pytest tests/flydsl/test_gemm.py -k gemm_950 -q`；
2. 16个`wave x tile x dtype`配置的`4096^3`长循环，每项10次；
3. 相同16个配置的两K-tile版本，BF16 K=128、FP8 K=256，每项20次。

每次launch前用NaN填充C，并对整个输出与Torch reference做比较：BF16使用
`rtol=0.1, atol=0.03`，FP8使用`rtol=0.05, atol=0.5`。

### 2.3 汇编与同步验证

```bash
python3 tests/flydsl/verify_gemm_950_pipeline.py
python3 tests/flydsl/compare_gemm_950.py verify-8wave
sha256sum -c tests/flydsl/asm/gfx950_current/SHA256SUMS
sha256sum -c tests/flydsl/asm/gfx950_8wave_tiles/SHA256SUMS
```

默认`256x256`汇编合同：

- 4-wave：prologue `vmcnt(24)`；主循环八次`vmcnt(20)`；tail
  `20,16,12,8,4,0`；相邻wait之间4条DMA。
- 8-wave：prologue `vmcnt(12)`；主循环八次`vmcnt(10)`；tail
  `10,8,6,4,2,0`；相邻wait之间2条DMA；每区相位为
  `wait -> setprio(1) -> barrier -> MFMA -> setprio(0) -> barrier -> LDS read/DMA`。

完整8-wave tile汇编、抽取值和SHA256保存在
`tests/flydsl/asm/gfx950_8wave_tiles/`。

### 2.4 性能测试

计时闭包只发起kernel；输入构造、preshuffle、分配、编译和正确性检查均在计时区外。
所有case先通过完整输出检查，再预热20轮，执行24组、每组100次的位置平衡计时；case
每轮循环移位，完成一轮位置集合后反向。

完整tile sweep：

```bash
python3 tests/flydsl/compare_gemm_950.py benchmark \
  --m 4096 --n 4096 --k-values 4096 \
  --waves all --dtype all \
  --tiles 128x128,128x256,256x128,256x256 \
  --jit-layout none --warmup 20 --rounds 24 --iterations 100 \
  --csv /tmp/gemm950_tile_sweep.csv
```

默认FlyDSL/JIT对照：

```bash
python3 tests/flydsl/compare_gemm_950.py benchmark \
  --m 4096 --n 4096 --k-values 4096 \
  --waves 8 --dtype fp8 --tiles 256x256 \
  --jit-layout both --warmup 20 --rounds 24 --iterations 100 \
  --csv /tmp/gemm950_default_vs_jit.csv
```

K sweep：

```bash
python3 tests/flydsl/compare_gemm_950.py benchmark \
  --m 4096 --n 4096 \
  --k-values 1024,2048,4096,8192,16384 \
  --waves all --dtype all --tiles 256x256 \
  --jit-layout both --warmup 20 --rounds 24 --iterations 100 \
  --csv /tmp/gemm950_k_sweep.csv
```

## 3. 当前采用的优化方法

### 3.1 指令与寄存器

- BF16使用原生`v_mfma_f32_16x16x32_bf16`。
- FP8复用现有stateful `MFMA_Scale` atom，将A/B state设为0，由LLVM选择plain
  `v_mfma_f32_16x16x128_f8f6f4`；不修改FlyDSL atom实现。
- 4-wave BF16按`f32x4` slice构造原生MFMA accumulator chain，使累加器保持在AGPR且
  保留MFMA scheduler分类。
- 保留B fragment live-range anchor，避免LDS reader过早复用仍被MFMA消费的VGPR。

### 3.2 LDS与数据布局

- `AT/AB/BL/BR`的两个stage使用8个独立`SharedStorage950` leaf，保留对象身份并避免
  alias metadata。
- 4-wave BF16 A每512元素padding 16个元素；FP8 A使用1024/2048双层padding。
- 8-wave A使用对应的分组行重排与padding read/write view。
- B在LDS中保持host preshuffle物理顺序；DMA与reader成对实现，连续访问消除bank
  conflict。
- DMA目标采用独立stage root加wave-relative GEP；源offset在prologue预计算，主循环只
  叠加K tile和chunk常量。

### 3.3 Pipeline与同步

令一个A/B逻辑组分别展开为`A=a_vmem_count`、`B=b_vmem_count`条物理DMA：

- prologue：`3A+3B`；
- region 0/1/4/5：`2A+3B`；
- region 2/3/6/7：`3A+2B`；
- tail：`2A+3B,2A+2B,2A+B,A+B,B,0`。

4-wave在每次LDS read前执行wait/barrier。8-wave把每个`rocdl.s_waitcnt`直接写在对应
region中，不再隐藏在`begin_compute_phase`内；最终相位保持
`wait -> setprio(1) -> barrier -> MFMA -> setprio(0) -> barrier -> LDS read/DMA`。
这种写法让每区的VMEM账本和物理wait位置能在源码中直接核对，同时保持compute高优先级。
每区scheduler把DSRD和VMEM按实际数量均匀分布；4-wave BF16使用MFMA/DSRD/VMEM分组
提示，8-wave只保留memory提示以降低调度组和寄存器压力。

上一轮显式wait重构相对其重构前保存的数据，默认`256x256` 8-wave BF16延迟从0.097349 ms降至
0.096863 ms（-0.50%），FP8从0.045700 ms变为0.045816 ms（+0.25%）。五点K sweep中，
8-wave BF16变化范围为-0.82%到-0.20%，FP8为-0.25%到+0.14%；结合当轮最大2.58%的
Q1/Q3跨度，没有观察到显式wait和compute高优先级导致的稳定性能回退。

### 3.4 输出与调度

- permlane16 epilogue把两个64-bit store合并为128-bit store，不需要跨wave CShuffle。
- XCD-aware `get_pids`按8个XCD和`GROUP_SIZE_M=4`映射workgroup，提高B的L2复用。
- 默认保留`256x256` tile；本轮完整tile sweep确认它在两种精度和两种wave数下均最快。

tile sweep延迟变化范围为-0.43%到+0.26%，默认JIT对比为
-0.02%到+0.06%，K sweep为-0.62%到+0.31%。三组最大绝对变化分别为0.43%、0.06%和
0.62%，均小于对应运行内Q1/Q3跨度，未观察到rebase性能回退。49行原始对比保存在
`tests/flydsl/results/gfx950_current/rebase_comparison.csv`。

## 4. 最终测试数据

### 4.1 功能

| 测试 | 覆盖 | 结果 |
|---|---|---:|
| Pytest | 单block、多block、permlane、小tile重复launch | 18/18通过 |
| 长循环矩阵 | 16配置，`4096^3`，每项10次 | 160/160通过 |
| 两K-tile矩阵 | 16配置，每项20次 | 320/320通过 |
| 总计 | 当前代码 | 498次全部通过 |

### 4.2 Tile性能，`4096^3`

| Waves / Tile | BF16 ms / TFLOPS | FP8 ms / TFLOPS |
|---|---:|---:|
| 4w `128x128` | 0.112675 / 1219.8 | 0.056216 / 2444.8 |
| 4w `128x256` | 0.109554 / 1254.5 | 0.051581 / 2664.5 |
| 4w `256x128` | 0.106075 / 1295.7 | 0.050945 / 2697.8 |
| 4w `256x256` | **0.097037 / 1416.4** | **0.047158 / 2914.4** |
| 8w `128x128` | 0.134772 / 1019.8 | 0.063684 / 2158.1 |
| 8w `128x256` | 0.126054 / 1090.3 | 0.058890 / 2333.8 |
| 8w `256x128` | 0.121099 / 1134.9 | 0.057743 / 2380.2 |
| 8w `256x256` | **0.097042 / 1416.3** | **0.045704 / 3007.1** |

`256x256`中，4-wave与8-wave BF16延迟相差0.01%，8-wave FP8比4-wave快3.08%。全部
16个case在计时前通过正确性检查，Q1/Q3跨度最大为2.25%。原始数据：
`tests/flydsl/results/gfx950_current/tile_sweep.csv`。

### 4.3 默认8-wave FP8与JIT，`4096^3`

| 路径 | 中位延迟 | Q1/Q3 | TFLOPS | 相对FlyDSL |
|---|---:|---:|---:|---:|
| FlyDSL | 0.045852 ms | 0.045786/0.045940 ms | 2997.5 | - |
| JIT preshuffle | 0.044920 ms | 0.044884/0.044972 ms | 3059.7 | -2.03% |
| JIT row-major | 0.044984 ms | 0.044927/0.045061 ms | 3055.3 | -1.89% |

三条路径均先通过同输入正确性检查，Q1/Q3跨度最大为0.34%。原始数据：
`tests/flydsl/results/gfx950_current/default_vs_jit.csv`。

### 4.4 K sweep

K sweep开始前，外部任务保留89% VRAM，但连续采样显示GPU use与memory activity均为0%，
仍有约32 GiB可用；测试后仍为0%。所有case先通过正确性检查，再按20次预热、24轮、
每轮100次位置平衡计时。Q1/Q3跨度最大为中位数的2.76%。

BF16：

| K | 4-wave ms / TFLOPS | 8-wave ms / TFLOPS | 8-wave延迟差 |
|---:|---:|---:|---:|
| 1024 | 0.031703 / 1083.8 | 0.031195 / 1101.4 | -1.60% |
| 2048 | 0.053385 / 1287.2 | 0.053137 / 1293.2 | -0.46% |
| 4096 | 0.097259 / 1413.1 | 0.097498 / 1409.7 | +0.25% |
| 8192 | 0.184687 / 1488.3 | 0.186331 / 1475.2 | +0.89% |
| 16384 | 0.355703 / 1545.5 | 0.355995 / 1544.3 | +0.08% |

FP8与JIT：

| K | FlyDSL 4w ms / TFLOPS | FlyDSL 8w ms / TFLOPS | JIT preshuffle ms / TFLOPS | JIT row-major ms / TFLOPS |
|---:|---:|---:|---:|---:|
| 1024 | 0.019367 / 1774.2 | 0.018116 / 1896.6 | 0.017037 / 2016.8 | 0.017000 / 2021.1 |
| 2048 | 0.029045 / 2366.0 | 0.027352 / 2512.4 | 0.026233 / 2619.6 | 0.026136 / 2629.3 |
| 4096 | 0.047115 / 2917.1 | 0.045782 / 3002.1 | 0.044767 / 3070.1 | 0.044828 / 3065.9 |
| 8192 | 0.084289 / 3261.1 | 0.082726 / 3322.8 | 0.082083 / 3348.8 | 0.082227 / 3342.9 |
| 16384 | 0.159456 / 3447.7 | 0.156791 / 3506.3 | 0.158525 / 3467.9 | 0.158384 / 3471.0 |

在短K下JIT领先；随着K增加，差距持续缩小。K=16384时FlyDSL 8-wave FP8达到
3506.3 TFLOPS，分别比JIT preshuffle和row-major快1.09%与1.01%。原始30行数据保存在
`tests/flydsl/results/gfx950_current/k_sweep.csv`。
