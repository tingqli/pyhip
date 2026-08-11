# GPU workload frequency probe

[`probe-gpu-frequency.py`](probe-gpu-frequency.py) 使用 PyHIP JIT 生成持续满载的 AMDGPU kernel，比较不同
指令组合触发的 SCLK、PPT 功耗和温度变化。当前支持 gfx94x 与 gfx950。

## Workloads

| 名称 | 热循环 | 默认参数 |
|---|---|---|
| `mfma` | 纯 MFMA，4 组独立 accumulator | gfx94x BF16 `16x16x16`；gfx950 FP8/F6/F4 `16x16x128` |
| `mfma_valu` | 每条 MFMA 后放可共发的独立 scalar VALU | `3 x v_add_f32` |
| `mfma_valu_burst` | 连续 MFMA 段后接连续 VALU 段；仅显式选择，不属于 `all` | `--mfma-burst`、`--valu-burst` |
| `mfma_exp` | MFMA 与独立 EXP 严格交替 | 每组 `MFMA -> v_exp_f32` |
| `exp` | 纯 EXP，4 组独立目标寄存器轮转 | `v_exp_f32` |
| `valu` | 4 条独立依赖链轮转 | `v_fmac_f32` |
| `mfma_mem` | MFMA 与延迟消费的 non-temporal HBM load 交错 | `load -> 64 x MFMA -> vmcnt(0)` |
| `mfma_ds_mem` | MFMA、LDS read 与 HBM load 交错 | `load -> ds_read -> 16 x MFMA -> lgkmcnt(0) -> 48 x MFMA -> vmcnt(0)` |
| `mem` | 纯 non-temporal HBM 流式读取 | 每组 1 条 128-bit/lane load |

延迟保证模式按 MFMA 最小 latency 16 cycles 计算。`mfma_mem` 和 `mfma_ds_mem` 的 64 条 MFMA
使用同一 accumulator 依赖链，因此 HBM load 发出到 `vmcnt(0)`/消费之间至少相隔 1024 cycles；
`mfma_ds_mem` 的 LDS read 发出到 `lgkmcnt(0)`/消费之间先执行同一链上的 16 条 MFMA，至少相隔
256 cycles。gfx950 的 FP8 `16x16x128` MFMA latency 可能为 32 cycles，因此实际距离只会更长。

默认 launch 为 `4 x CU` 个 256-thread workgroup。每个 wave 写回硬件位置，host 会要求全部 wave 均有
记录，并报告实际覆盖的 CU/SIMD 数。HBM workload 使用 512 MiB、2 的幂大小的环形缓冲区；可通过
`--buffer-mib` 修改。

`mfma_exp` 在 gfx942 实测每 CU 超过 2 个 workgroup 时会分两批执行，使整次 launch 时长约为每 wave
目标时长的两倍；程序因此将该模式自动限制为最多 `2 x CU`，仍要求覆盖全部 CU。其他模式继续使用
`--blocks-per-cu`。JSON 每条结果记录实际 `grid_blocks` 与 `blocks_per_cu`，若 event 时长明显超过
每 wave 的硬件时长，程序会拒绝该结果。

## Timing and frequency

kernel 同时读取两种硬件计数器：

- `s_memrealtime`：固定 100 MHz，与 DPM 无关；kernel 按它自行运行到目标时长；
- `s_memtime`：按 shader clock 累加。

整段有效频率为：

$$
f_{\mathrm{SCLK}} = 100\,\mathrm{MHz}
\frac{\Delta\mathrm{s\_memtime}}{\Delta\mathrm{s\_memrealtime}}.
$$

因此 1 ms 测试不依赖刷新较慢的 sysfs。程序仍用独立 CPU 线程采集以下轨迹，默认间隔 10 ms：

- `freq*_input`，标签为 `sclk`/`gfxclk`；
- `power*_input`，标签为 `PPT`/`socket power`；
- junction 与 HBM 温度；
- `gpu_busy_percent`。

结果同时报告硬件目标时长误差、CUDA event 相对硬件时长的额外开销、全窗口有效 SCLK，以及后半程
sysfs 稳态中位数。`drop` 以 `pp_dpm_sclk` 中的最高档为基准；它不是相对 idle 的频率变化。

## Usage

从仓库根目录运行。先用 `rocm-smi --showuse --showmemuse` 选择空闲卡，并保持 DPM 为 `auto`：

```bash
cd /root/workspace/luocheng/pyhip
HIP_VISIBLE_DEVICES=4 \
PYHIP_CACHE_DIR=/tmp/pyhip-gpu-frequency \
python3 -B tests/flydsl/attn_4wave/tools/probe-gpu-frequency.py \
  --workloads all \
  --duration-ms 1,10,100,1000,3000 \
  --repeats 3 \
  --json /tmp/gpu-frequency.json
```

只检查两个端点：

```bash
HIP_VISIBLE_DEVICES=4 \
python3 -B tests/flydsl/attn_4wave/tools/probe-gpu-frequency.py \
  --workloads mfma,valu \
  --duration-ms 1,3000 \
  --cooldown-ms 2000
```

单次持续时间范围为 1--30000 ms。小于 sysfs 采样间隔的运行可能没有有效的 sysfs 稳态点，此时以
kernel 内双时钟算出的 `effective_sclk_mhz` 为准。`--cooldown-ms` 控制不同 workload 之间的冷却时间；
正式横向比较应保留足够冷却，并检查 JSON 中的完整频率/功耗轨迹。

## JSON

JSON 保留：

- GPU 名称、架构、BDF、CU 数、DPM 模式、传感器路径和最高 SCLK 档；
- launch、循环、内存、采样及冷却配置；
- MFMA latency 假设，以及 VMEM/DS wait 的最小 cycle 与 MFMA 数；
- 每个 wave 的 realtime ticks、shader cycles、硬件 ID、XCC ID 和循环批次数；
- CUDA event/host/hardware 时长、CU/SIMD 覆盖、有效 SCLK 分布及估算吞吐；
- 每个 sysfs 原始采样点，以及全窗口/后半程的 SCLK、功耗和温度统计。

程序默认拒绝启动时 `gpu_busy_percent != 0` 或 DPM 非 `auto` 的 GPU。`--allow-busy` 只绕过前者；
共享卡结果会混入其他进程的负载，不适合作为降频结论。

## MI308X 正式结果（2026-08-10）

### 环境与协议

- GPU：AMD Instinct MI308X OAM，`gfx942`，80 CU，BDF `0001:0b:00.0`，NUMA node 1；
- SCLK DPM：`auto`，最高档 1850 MHz；PPT cap 650 W；
- 软件：ROCm 7.2.0、PyTorch `2.9.1+rocm7.2.0.git7e1940d4`、Python 3.10.12；
- 仓库：`luocheng/try-mha-308`，HEAD `6ad9261f7ccdd880d9d1965283e97e86d9707f5e`；
- 脚本 SHA256：`77346b18a4f99aaa50c2b7fdd567f4d4e0079e36d2b29bf2801ad288b838c208`；
- 启动前：GPU busy 0%，SCLK 90/91 MHz，DPM `auto`，PPT 162/163 W；
- 参数：每档 3 次，时长 `1,10,100,1000,3000 ms`，workload 之间冷却 2000 ms，sysfs 每 10 ms
  采样，`inner_unroll=64`，HBM buffer 512 MiB；
- grid：默认 4 blocks/CU；`mfma_exp` 自动使用 2 blocks/CU。原矩阵 105 个样本及纯 EXP 补充的
  6 个样本均覆盖 80 CU 和对应 grid 的全部 wave，没有分批调度。

短窗与稳态分别独立保存，防止长时间矩阵中断时丢失已完成结果：

| 原始数据 | 时间戳（UTC） | 样本数 | SHA256 |
|---|---|---:|---|
| `/tmp/gpu-frequency-short-20260810.json` | 2026-08-10 12:56:13 | 63 | `aa68647f81acf0fbae5c70a3dcd4dd5c022cf5c81d360007e6cc20ae5dca3d3c` |
| `/tmp/gpu-frequency-steady-20260810.json` | 2026-08-10 12:59:25 | 42 | `9009972221c5f8290e4feb855550f350becc02cef7be413b70dd40a2399cf595` |
| `/tmp/gpu-frequency-exp-steady-20260810.json` | 2026-08-10 13:48:32 | 6 | `b41615ece6cb5288bd1e55b9666f4e6a1a59d586c472c6e86d15704c23a31a5a` |

原 105 样本矩阵不包含后来新增的纯 EXP，复现时必须显式指定原七种 workload：

```bash
HIP_VISIBLE_DEVICES=4 PYHIP_CACHE_DIR=/tmp/pyhip-gpu-frequency-formal-20260810 \
python3 -B tests/flydsl/attn_4wave/tools/probe-gpu-frequency.py \
  --device 0 --workloads mfma,mfma_valu,mfma_exp,valu,mfma_mem,mfma_ds_mem,mem \
  --duration-ms 1,10,100 --repeats 3 \
  --blocks-per-cu 4 --inner-unroll 64 --valu-per-mfma 3 --loads-per-group 1 \
  --buffer-mib 512 --sample-interval-ms 10 --cooldown-ms 2000 \
  --json /tmp/gpu-frequency-short-20260810.json

HIP_VISIBLE_DEVICES=4 PYHIP_CACHE_DIR=/tmp/pyhip-gpu-frequency-formal-20260810 \
python3 -B tests/flydsl/attn_4wave/tools/probe-gpu-frequency.py \
  --device 0 --workloads mfma,mfma_valu,mfma_exp,valu,mfma_mem,mfma_ds_mem,mem \
  --duration-ms 1000,3000 --repeats 3 \
  --blocks-per-cu 4 --inner-unroll 64 --valu-per-mfma 3 --loads-per-group 1 \
  --buffer-mib 512 --sample-interval-ms 10 --cooldown-ms 2000 \
  --json /tmp/gpu-frequency-steady-20260810.json
```

纯 EXP 补充结果使用当前脚本 SHA256
`d64c07553275fb217261901e30010a9a5ca43feedc8bdbd8e2573ce453ba0c4b`：

```bash
HIP_VISIBLE_DEVICES=4 PYHIP_CACHE_DIR=/tmp/pbexp-final \
python3 -B tests/flydsl/attn_4wave/tools/probe-gpu-frequency.py \
  --device 0 --workloads exp --duration-ms 1000,3000 --repeats 3 \
  --blocks-per-cu 4 --sample-interval-ms 10 --cooldown-ms 2000 \
  --json /tmp/gpu-frequency-exp-steady-20260810.json
```

### 不同时长的整窗有效 SCLK

每格为 3 次运行的中位数 `[最小值, 最大值]`，单位 MHz。该值来自每个 wave 的
`s_memtime / s_memrealtime`，不是单个 sysfs 瞬时点。

| workload | 1 ms | 10 ms | 100 ms | 1 s | 3 s |
|---|---:|---:|---:|---:|---:|
| `mfma` | 1748.6 [1639.1, 1836.3] | 1834.1 [1815.9, 1835.7] | 1831.7 [1829.9, 1832.9] | 1132.2 [1131.7, 1135.9] | 1131.5 [1131.4, 1132.0] |
| `mfma_valu` | 1789.1 [1754.1, 1829.9] | 1818.1 [1815.6, 1822.5] | 1823.8 [1734.6, 1823.9] | 1131.9 [1130.4, 1133.8] | 1131.6 [1131.5, 1132.3] |
| `mfma_exp` | 1752.6 [1735.9, 1831.3] | 1823.4 [1811.4, 1826.8] | 1828.2 [1827.9, 1829.1] | 1343.6 [1342.8, 1348.3] | 1342.7 [1342.3, 1342.8] |
| `valu` | 1765.2 [1635.1, 1840.8] | 1832.2 [1819.4, 1836.2] | 1836.3 [1800.2, 1838.7] | 1836.9 [1826.6, 1837.9] | 1837.2 [1836.7, 1839.1] |
| `mfma_mem` | 1735.2 [1729.7, 1781.2] | 1835.2 [1826.8, 1835.4] | 1832.4 [1729.0, 1833.0] | 1132.3 [1130.4, 1132.8] | 1132.1 [1131.2, 1132.2] |
| `mfma_ds_mem` | 1832.6 [1304.6, 1834.2] | 1827.1 [1820.0, 1830.4] | 1832.1 [1830.3, 1833.5] | 1133.4 [1133.3, 1133.9] | 1132.1 [1131.9, 1132.7] |
| `mem` | 1738.6 [1686.0, 1843.3] | 1734.9 [1723.3, 1753.2] | 1604.1 [1572.5, 1646.7] | 1643.3 [1636.5, 1644.2] | 1592.0 [1585.9, 1594.8] |

纯 EXP 是后续补充测试，仅正式采集 1/3 秒三次重复：

| workload | 1 s | 3 s |
|---|---:|---:|
| `exp` | 1830.1 [1818.0, 1832.5] | 1834.0 [1833.6, 1837.2] |

1 ms 结果依赖 workload 启动时的 DPM 相位，三次范围可达 530 MHz，不能用单个 1 ms 样本判断稳态
降频。10 ms 开始明显收敛；3 秒时计算类 workload 的三次范围不超过 2.4 MHz，纯 HBM load 的范围为
8.8 MHz。硬件目标时长误差在 1 ms 档最高 0.84%（纯 `mem` 的静态批次粒度），10 ms 以上最高
0.066%，100 ms 以上最高 0.0069%。

### 3 秒稳态端点

`effective SCLK` 是完整 3 秒窗口的双时钟平均；`steady sysfs` 是后半段 sysfs 样本的中位数，因此
计算类 workload 在前半段从高档切换到低档时，前者会高于后者。

| workload | effective SCLK | 相对 1850 MHz 降幅 | steady sysfs | PPT | 最高 junction | 最高 HBM | 吞吐 |
|---|---:|---:|---:|---:|---:|---:|---|
| `mfma` | 1131.5 MHz | 38.84% | 1014 MHz | 242 W | 52 C | 47 C | 185.060 TFLOPS |
| `mfma_valu` | 1131.6 MHz | 38.83% | 1011 MHz | 256 W | 53 C | 46 C | 184.988 TFLOPS MFMA + 4.336 TFLOPS VALU |
| `mfma_exp` | 1342.7 MHz | 27.42% | 1210 MHz | 265 W | 53 C | 46 C | 175.792 TFLOPS + 1373.379 GOPS EXP |
| `exp` | 1834.0 MHz | 0.87% | 1853 MHz | 296 W | 50 C | 44 C | 2344.462 GOPS EXP |
| `valu` | 1837.2 MHz | 0.69% | 1850 MHz | 316 W | 54 C | 46 C | 18.808 TFLOPS |
| `mfma_mem` | 1132.1 MHz | 38.80% | 1010 MHz | 304.5 W | 53 C | 49 C | 185.016 TFLOPS + 0.361 TB/s |
| `mfma_ds_mem` | 1132.1 MHz | 38.80% | 1012 MHz | 305 W | 53 C | 49 C | 185.061 TFLOPS + 0.361 TB/s HBM + 0.361 TB/s LDS |
| `mem` | 1592.0 MHz | 13.95% | 1588 MHz | 650 W | 56 C | 68 C | 4.296 TB/s |

### 结论

1. 纯 MFMA 是最强的非 PPT 降频触发器：3 秒整窗约 1132 MHz，后半段约 1010 MHz，但 PPT 只有
  242 W。加入三条可共发 VALU 不改变频率档位，只把 PPT 提高约 14 W。
2. 纯 EXP 基本不降频：3 秒整窗约 1834 MHz、后半段约 1853 MHz、PPT 约 296 W。`MFMA + EXP`
  却稳定在整窗约 1343 MHz、后半段约 1210 MHz，说明该低频档由 MFMA 段触发，不是 EXP 本身。
3. 纯 VALU 同样几乎不降频：整窗 1837 MHz，后半段保持 1850 MHz，PPT 约 316 W。
4. 在当前低带宽延迟隐藏配置中，给 MFMA 加 HBM load 或 LDS+HBM 不改变 MFMA 主导的降频档位；
  `mfma_mem` 与 `mfma_ds_mem` 的 3 秒频率和 MFMA 吞吐几乎相同。
5. 纯 HBM load 触及 650 W PPT，3 秒整窗降到约 1592 MHz，HBM 温度升至 68 C。它的降频由功耗
  上限主导，与纯 MFMA 的低功耗/低频机制不同。
6. 对生产 kernel 判断降频时，至少使用 1 秒窗口并同时报告整窗有效 SCLK、后半段 sysfs SCLK 和
  PPT；1 ms 只适合观察启动瞬态，不能代表稳态档位。

## 纯 MFMA 连续维持 1.8 GHz 的时间（2026-08-10）

### 测量方法

sysfs SCLK 约每 10 ms 更新且有明显滞后，不能直接用来定位降频起点。脚本新增 kernel 内 timeline
模式：每个 bin 同时读取随 shader clock 累加的 `s_memtime` 和固定 100 MHz 的
`s_memrealtime`，直接计算该 bin 的有效 SCLK。判定条件为：

- 阈值 1800 MHz；
- 首次连续 1.5 ms 的 bin 中位频率低于 1800 MHz，视为离开 1.8 GHz 高频平台；
- 2 blocks/CU、256 threads/block。该配置的 100 ms 纯 MFMA 吞吐中位数为 300.15 TFLOPS，与
  4 blocks/CU 的约 299.9 TFLOPS 一致，仍是满载；4 blocks/CU 的记录型 kernel 会分两批调度，不能
  用于绝对时间测量。

正式测试使用两种分辨率交叉验证：

| bin | 连续低频 bin | 三次离开 1.8 GHz 的时间 | 中位数 |
|---:|---:|---:|---:|
| 0.5 ms | 3 | 149.120 / 147.614 / 149.122 ms | 149.120 ms |
| 0.25 ms | 6 | 150.895 / 151.135 / 152.252 ms | 151.135 ms |

两种分辨率都观察到阶跃：阈值前约为 1824--1831 MHz，随后在约 1--2 ms 内快速落到约
1500--1600 MHz，并在约 175--200 ms 后稳定到约 1000 MHz。分箱本身包含计数器读取和每-bin写回，
不同分辨率及运行初态会造成约 2 ms 差异，因此最终结论取交叉验证范围而不是伪精确的单点：

> **在本机 MI308X、DPM `auto`、满载纯 BF16 MFMA 下，约可连续维持 1.8 GHz 150 ms；实测离开
> 高频平台的范围为 147.6--152.3 ms。**

复现命令：

```bash
HIP_VISIBLE_DEVICES=4 PYHIP_CACHE_DIR=/tmp/pbtlfinal \
python3 -B tests/flydsl/attn_4wave/tools/probe-gpu-frequency.py \
  --device 0 --duration-ms 400 --repeats 3 --blocks-per-cu 2 --inner-unroll 64 \
  --timeline-bin-ms 0.5 --timeline-threshold-mhz 1800 --timeline-low-bins 3 \
  --sample-interval-ms 5 --cooldown-ms 3000 \
  --json /tmp/gpu-frequency-mfma-timeline-20260810.json

HIP_VISIBLE_DEVICES=4 PYHIP_CACHE_DIR=/tmp/pbtl025 \
python3 -B tests/flydsl/attn_4wave/tools/probe-gpu-frequency.py \
  --device 0 --duration-ms 200 --repeats 3 --blocks-per-cu 2 --inner-unroll 64 \
  --timeline-bin-ms 0.25 --timeline-threshold-mhz 1800 --timeline-low-bins 6 \
  --sample-interval-ms 5 --cooldown-ms 3000 \
  --json /tmp/gpu-frequency-mfma-timeline-025ms-20260810.json
```

- timeline 脚本 SHA256：`15b5d83dedc37ce3d08015b4818bce6e29209c6bf2208f2b73a8a575d3ceb9b1`；
- 0.5 ms JSON SHA256：`be94e7ebc6ae4f894c336901e1fac3aebadcc9f991db5a6f2706dab7f70a70c0`；
- 0.25 ms JSON SHA256：`a44f3c0d4813197fd86c8c65cde0b3f5dd6b4342cc8d41b454c535ab2788f6b0`。

## 连续 MFMA 后接连续 VALU 扫描（2026-08-10）

### 定义

该测试与 `mfma_valu` 的逐条交织不同。每个运行时循环固定执行：

```text
M 条连续 v_mfma_f32_16x16x16_bf16
N 条连续 v_fmac_f32
```

MFMA 和 VALU 都轮转 4 条独立依赖链。最终 ISA 已逐点验证为完整连续段，没有被后端重排成交织序列。
测试使用 MI308X、DPM `auto`、3 秒 kernel、3 次重复、4 blocks/CU、每次 workload 间冷却 3 秒。

“避免降频”不是一个硬件公布的二元状态，本节使用两个明确口径：

1. **整窗通过**：3 秒双时钟有效 SCLK 不低于 1800 MHz；
2. **保守 p10 通过**：后半段 10 ms sysfs 样本的 p10 不低于 1800 MHz，即至少约 90% 的后半段
  样本处于 1800 MHz 以上。

### 64 条 MFMA 的精确边界

每行均为 3 次独立运行；effective 列为中位数 `[最小值, 最大值]`。

| 连续 MFMA | 连续 VALU | VALU/MFMA | effective SCLK | steady sysfs 中位数 | steady p10 中位数 | PPT | 三次整窗均 >=1800 |
|---:|---:|---:|---:|---:|---:|---:|---|
| 64 | 416 | 6.500000 | 1791.5 [1790.1, 1791.8] | 1827.0 | 1733.5 | 359 W | 否 |
| 64 | 418 | 6.531250 | 1793.7 [1790.8, 1795.0] | 1819.0 | 1738.4 | 360 W | 否 |
| 64 | 419 | 6.546875 | 1799.4 [1798.6, 1799.8] | 1824.0 | 1752.8 | 359 W | 否 |
| 64 | 420 | 6.562500 | 1800.2 [1799.6, 1803.3] | 1830.0 | 1765.5 | 359 W | 否 |
| 64 | 421 | 6.578125 | 1809.9 [1809.4, 1811.0] | 1833.5 | 1787.0 | 360 W | 是 |
| 64 | 422 | 6.593750 | 1813.8 [1811.2, 1814.7] | 1831.0 | 1775.0 | 360 W | 是 |
| 64 | 448 | 7.000000 | 1826.4 [1822.4, 1830.1] | 1842.0 | 1818.0 | 361 W | 是 |
| 64 | 512 | 8.000000 | 1824.6 [1817.7, 1826.4] | 1843.0 | 1817.2 | 355 W | 是 |
| 64 | 1024 | 16.000000 | 1834.4 [1822.1, 1836.8] | 1847.0 | 1828.4 | 333 W | 是 |

因此对主测试配置 `M=64`：

- 419 条 VALU 三次都未达到 1800 MHz；
- 420 条处在边界，中位数刚过线但有一次为 1799.6 MHz；
- **421 条是三次整窗都超过 1800 MHz 的最小实测值**，即 `421/64 = 6.578125`；
- 工程上可取 **约 6.6 条连续 scalar FMA VALU / 1 条 BF16 MFMA** 作为恢复高平均频率的经验比例。

### burst 长度验证

阈值并非只由比例决定，短 burst 有轻微额外开销；32/128 条 MFMA 的相邻验证如下：

| 连续 MFMA | 连续 VALU | VALU/MFMA | effective SCLK | 三次整窗均 >=1800 |
|---:|---:|---:|---:|---|
| 32 | 210 | 6.562500 | 1796.3 [1796.0, 1798.4] | 否 |
| 32 | 211 | 6.593750 | 1803.7 [1796.9, 1805.0] | 否 |
| 32 | 212 | 6.625000 | 1806.5 [1800.5, 1808.3] | 是 |
| 128 | 832 | 6.500000 | 1792.0 [1788.3, 1792.6] | 否 |
| 128 | 840 | 6.562500 | 1804.7 [1802.6, 1805.0] | 是 |

三种 burst 长度的稳定整窗阈值落在 `6.5625--6.625 VALU/MFMA`，支持使用 6.6:1 作为近似比例，
但不能把它当成架构保证。

### 低频尾部

421 条 VALU 能恢复 3 秒平均频率，但后半段 sysfs p10 仍约 1787 MHz，说明仍偶尔落入较低档位。
按“3 次运行的后半段 p10 都 >=1800 MHz”这一更保守标准：

- 656 条仍失败，其中一组 p10 为 1797 MHz；
- 672 条通过，三组 p10 为 1820.4/1820.2/1821.2 MHz；
- 即保守 p10 阈值在 656--672 条之间，约为 **10.5:1 VALU/MFMA**；
- 即使 1024 条 VALU，个别 10 ms 瞬时最小值仍可低于 1800 MHz，因此不能声称所有采样点绝不降频。

### 复现与原始数据

显式选择 burst 模式，例如：

```bash
HIP_VISIBLE_DEVICES=4 PYHIP_CACHE_DIR=/tmp/pb421 \
python3 -B tests/flydsl/attn_4wave/tools/probe-gpu-frequency.py \
  --device 0 --workloads mfma_valu_burst --duration-ms 3000 --repeats 3 \
  --blocks-per-cu 4 --mfma-burst 64 --valu-burst 421 \
  --sample-interval-ms 10 --cooldown-ms 3000 \
  --json /tmp/gpu-frequency-valu-burst-final-20260810/valu-421.json
```

- 扫描脚本 SHA256：`2d651bdb3715a3a1988dd9671d0bf1525bdb5db5b6897b296530011a85688d70`；
- 汇总：`/tmp/gpu-frequency-valu-burst-summary-20260810.json`；
- 汇总 SHA256：`65ca2d923422946fa9b72793316898e327e88ebfe5ce2fc5b22178415da5cb64`；
- 每个汇总项包含三次原始值、源 JSON 路径及其 SHA256。