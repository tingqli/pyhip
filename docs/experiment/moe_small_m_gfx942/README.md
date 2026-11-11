# gfx942 FP8 block-scale MoE：每 expert ≤128 routes 的 small-M 分支 —— 实验数据

过程与结论见 [docs/record/moe_small_m_gfx942.md](../../record/moe_small_m_gfx942.md)。本目录只放数据、复现命令和由数据直接得出的表。

4-wave small-M kernel、`big_blocks` 选项、BM256 的 `big_blocks_only` 和 bench 的 small / split 路径已从代码中删除（record
§10）：本目录和各子目录里用到它们的工具要在 commit da764f5 上运行。

**本页是第一版**（8 wave、128 行 sort、host 端按 shape 选择）的数据。当前代码是第三版（4 wave、256 行 sort 表内分支、每个
workgroup 128×256 输出 tile），数据在 [4wave_128/README.md](4wave_128/README.md)；第二版（GateUp 64 列）与两版的同会话对照在
[4wave/README.md](4wave/README.md)。第一版源码保存在
`4wave/variants/moe_small_m_8wave_sort128.py`；下面的 `tools/sweep_blocks.py`、`tools/wg_timeline.py` 按第一版的 API 写成，
复现命令中的 bench `small` 路径现在指第二版的 mixed 路径。

## 环境

- 8× MI300X（gfx942:sramecc+:xnack-，304 CU，750 W），测量在 GPU0（PCI 0000:0c:00.0），探针/开发实验在 GPU1。
- torch 2.11.0+rocm7.2，HIP 7.2.26015，FlyDSL 0.4.0，aiter 285579b62（`/sgl-workspace/aiter`），rocprofv3 1.1.0。
- 形状：MiMo-V2.6-Flash 一个 EP8 rank：hidden H=4096，moe_inter I=2048，256 个全局 expert / 32 个本地，top-8；
  权重 shuffle_weight(16,16) FP8，per-128×128 block scale；激活 per-(token, 128) scale。每次调用读本地全部权重 768 MiB
  （W1 512 MiB + W2 256 MiB）。
- **GPU 争用**：另一个容器的作业间歇性占满 8 张卡（~190 GB 显存、~95% GFX activity、~570 W）。所有计时脚本都用
  `tests/contrib/moe/gpu_contention.py` 采样外部进程的 CU 占用，被污染的计时块重测；bench 前后与每轮还记录 amdsmi gate
  （activity ≤5%）。最终运行在外部作业的空闲窗口里逐个执行（`/tmp` 下的脚本等 GPU0 连续空闲后再跑、gate 里出现外部
  显存就重跑）：`final_random`、`burst_*`、`final_ep_uniform_t4096` 的 gate 全部 PASS；`final_ep_uniform` 的 T=4096
  第 1–3 轮记录到外部进程（activity 12–29%），该点用单独重测替换；`baseline_ep_uniform` 的 gate 因本进程尾部 activity
  采样 >5% 判 FAIL，但显存只有本进程。争用期间的开发数据没有收入本目录。

## 文件

| 文件 | 内容 |
|---|---|
| `baseline_ep_uniform.{json,log}` | 优化前基线：aiter asm 全部 11 个 1-stage kernel + BM256 路径，T=256..4096 |
| `final_ep_uniform.{json,log}` | 最终（持续态）：asm / BM256 / small 路径 + 单 kernel（`k:`），ep_uniform 路由 |
| `final_ep_uniform_t4096.{json,log}` | 上一行 T=4096 的干净重测（主运行该点有外部负载） |
| `final_random.{json,log}` | 同上，random 路由（每 expert routes 不均匀） |
| `burst_{ep_uniform,random}.{json,log}` | 突发态（`--warmup-ms 0`）asm 与 small 路径 |
| `superseded/final_*_pow2_slices.*` | 只有 1/2/4/8 个 16 行 slice 变体、ring=4 时的同一组测量（random 不达标的那一版） |
| `superseded/{final,burst}_*_ring4.*` | slice 1..8、ring=4 时的同一组测量（之后默认改为 ring=2） |
| `roofline_probe.{json,log}` | HBM 读带宽、FP8 MFMA 持续吞吐、`torch._scaled_mm` 探针（GPU0） |
| `stream_probe.json` | 每 WG 1 MiB、不同 stream 数/块长的 HBM 读探针（GPU1） |
| `tools/summarize.py` | 由 bench JSON 生成本文的表与 roofline |
| `tools/stream_probe_gfx942.{cpp,py}` | 多 stream 读带宽探针 |
| `tools/sweep_blocks.py` | 合成 sorted block（每块 `--rows` 个 route）扫描：时间/带宽 vs block 数、slice 数、ring |
| `tools/wg_timeline.py` | 插桩版 kernel（s_memrealtime + HW_ID/XCC_ID）给出每个 workgroup 的起止时间与 CU |
| `tools/att_phases.py` | ATT trace 中每个 K block 的 MFMA 跨度/屏障/尾部 |

## 复现

```bash
pip install -e .                                     # pyproject: package-dir {"pyhip" = "src"}，提供 pyhip.contrib.flydsl
cd tests/contrib/moe
python -m pytest -q -p no:cacheprovider --import-mode=importlib test_moe_small_m_gfx942.py
HIP_VISIBLE_DEVICES=0 python bench_moe_small_m_gfx942.py --tokens 256 512 1024 2048 4096 --paths asm bm256 small \
    --kernels --no-abort [--routing random] --json <out.json>
HIP_VISIBLE_DEVICES=0 python roofline_probe_gfx942.py --json <roofline_probe.json>
cd <this dir>/tools && python summarize.py ../final_ep_uniform.json ../final_random.json
python stream_probe_gfx942.py --json ../stream_probe.json
python sweep_blocks.py --rows 32 --blocks 19 32 38
python wg_timeline.py --rows 128 --blocks 32
```

bench 方法：每个变体排队执行（先 `torch.cuda._sleep` 垫起，≥50 ms 不计时预热，然后 32 次计时调用，每次一对 event），
4 轮、每轮反转变体顺序；读权重的变体轮换 2 份权重/激活副本，避免 256 MB MALL 命中；每个变体跑完后与 asm 测试的参考结果
对比（`check`）。表中为中位数。

## 1. Roofline 输入（探针）

GPU0 `roofline_probe.json`（GPU1 结果相同：4.94 TB/s、1157 TFLOPS）：

| 探针 | 结果 |
|---|---|
| HBM 读 1 GiB，nontemporal，304 WG×256 线程，4 load 在途 | **4.95 TB/s**（608 WG 4.87，1216/2432 WG 4.58） |
| FP8 MFMA `v_mfma_f32_16x16x32_fp8_fp8` 突发（4 / 8 wave/SIMD） | 1151 / 1205 TFLOPS |
| FP8 MFMA 持续 2 s | **1151 TFLOPS**（峰值 2615 的 44%：750 W 下有效时钟 ≈0.93 GHz） |
| `torch._scaled_mm` FP8 M=N=K=8192 / M=4096 | 1124 / 1074 TFLOPS |

时钟实测：ATT 中相邻 MFMA 发射间隔 16 tick（= SCLK 周期），trace 时钟 1.30 GHz（S=8 GateUp，profiler 串行化 dispatch）；
PMC `GRBM_GUI_ACTIVE`/XCD/时长：Down S=2 1.54–1.63 GHz，S=8 1.37–1.39 GHz。背靠背持续运行时功耗墙把时钟压得更低
（S=8 GateUp 单 task 117 µs，对应 ≈0.9 GHz 量级），与 MFMA 持续探针一致：**计算上限取 1157 TFLOPS**。

## 2. 权重流带宽：任务粒度和 stream 形态

§2–§3 的开发测量在 GPU1 上、用当时的默认 ring=4、短块计时（16 次预热 + 8～30 次计时，突发态），合成输入只有一份权重。

small-M kernel 每个 workgroup（8 wave，每 CU 1 个）读一个 expert 的 1 MiB（GateUp：256 行×4096）或 512 KiB（Down）
权重，形态是 16 个 16 行 group（相距 64 KiB）× 每 K block 2 KiB。`stream_probe.json`（每 WG 1 MiB，128 KiB 在途）：

| WG 数（轮） | 16×2 KiB | 16×4 KiB | 8×2 KiB | 8×8 KiB | 4×8 KiB | 1×32 KiB |
|---|---|---|---|---|---|---|
| 304（1.00） | 3.04* | 3.92 | 4.16 | 4.17 | 4.24 | 4.27 |
| 512（1.68） | 4.25 | 4.40 | 4.48 | 4.34 | 4.47 | 4.51 |
| 608（2.00） | 4.35 | 4.32 | 4.53 | 4.32 | 4.54 | 4.57 |

单位 TB/s；*第一个运行的 kernel，含冷启动。结论：**每 WG 1 MiB 的任务即使单 stream 也只有 4.3–4.6 TB/s**（大块连续读 3.4 MiB/WG
才到 4.94）；16 stream 形态只再损失约 5%。

`tools/sweep_blocks.py`，32 行/块（S=2），GateUp / Down（TB/s 按权重字节）：

| 有效 block | task 数 | 每 XCD 轮数 | GateUp µs | TB/s | Down µs | TB/s |
|---|---|---|---|---|---|---|
| 1 | 16 | 0.05 | 22.9 | 0.73 | 13.0 | 0.64 |
| 4 | 64 | 0.21 | 27.8 | 2.42 | 15.4 | 2.18 |
| 8 | 128 | 0.42 | 40.2 | 3.34 | 19.6 | 3.42 |
| 16 | 256 | 0.84 | 65.2 | 4.12 | 34.9 | 3.85 |
| 19 | 304 | 1.00 | 74.3–76.3 | 4.18–4.29 | 38.8–39.8 | 4.01–4.11 |
| 24 | 384 | 1.26 | 106.5 | 3.78 | 55.4 | 3.63 |
| 32 | 512 | 1.68 | 129.1–134.6 | 3.99–4.16 | 68.0–71.5 | 3.75–3.95 |
| 38 | 608 | 2.00 | 152.5–155.2 | 4.11–4.18 | 78.0–78.8 | 4.04–4.08 |
| 76 | 1216 | 4.00 | 353.2–357.6 | 3.57–3.61 | 156.8–160.9 | 3.96–4.07 |

单 CU 不受争用时能拉 ~50 GB/s（16 task：1 MiB/20 µs），满载时每 CU ~14 GB/s，所以非整轮的尾部不浪费一整轮
（1.68 轮的时间是 1 轮的 1.76 倍），**带宽上限由权重流本身决定，与 1 MiB 任务的探针相差 ≤4%**。

不改变上限的尝试（同一合成输入，S=2，1/2/4 轮）：

| 变体 | 1 轮 GateUp | 2 轮 GateUp | 4 轮 GateUp | 1 轮 Down | 2 轮 Down |
|---|---|---|---|---|---|
| ring=4（基准） | 76.3 | 155.2 | 356.9 | 39.8 | 78.2 |
| ring=2 | 75.0 | 154.7 | 356.5 | 38.8 | 78.0 |
| ring=8 | 74.8 | 156.4 | 354.8 | 40.6 | 82.2 |
| 去掉 MFMA 和 A | 75.1 | 154.7 | 353.6 | 39.6 | 78.9 |
| 再去掉 barrier | 75.5 | 155.9 | 357.6 | 40.0 | 78.8 |
| 每 task 旋转 K 起点（k_rot = slot×{1,4,5,13} mod KB） | 78.4–88.4 | 160.5–176.0 | – | 41.6–44.6 | 82.2–89.3 |

旋转 K 起点（让并发 task 落在不同 HBM 通道相位）反而慢 3–18%：同相位的并发流对 DRAM 更友好。

空 workgroup（真实 sort 的 grid 按最大 block 数开，只有有效 block 干活）：32 个有效 block 时 grid 32/96/288 block →
GateUp 129.1/130.1/131.4 µs，Down 70.3/70.3/70.3 µs，可忽略。

## 3. slice 数（每块用到的 16 行数）与计算侧

`sweep_blocks.py --blocks 32`（512 task，1.68 轮），每块 rows 个 route，K 循环/epilogue 取 S=ceil(rows/16) 个 slice：

| rows | S | GateUp µs | Down µs |
|---|---|---|---|
| 16 | 1 | 129.3 | 67.0 |
| 32 | 2 | 129.1–133.6 | 69.5–70.3 |
| 48 | 3 | 141.9 | 75.9 |
| 64 | 4 | 141.5–146.6 | 78.7–80.4 |
| 80 | 5 | 167.4 | 90.1 |
| 96 | 6 | 185.8 | 99.0 |
| 112 | 7 | 211.1 | 110.3 |
| 128 | 8 | 214.3–240.5 | 120.4–124.4 |

S≥4 后每多一个 slice GateUp +~20 µs、Down +~10 µs：边际成本等于 MFMA 工作量（每 slice 每 K block 每 SIMD 256 周期）在功耗墙时钟下的耗时，
即 S≥5 的 task 受 MFMA 吞吐限制。S=8 的消融（旧版 kernel，32 block）：去掉 B 加载 237.7、去掉 A 294.2*、去掉 barrier 233.8、
同时去掉 B/A/barrier 240.0、原版 237.9 µs（*该次被外部负载污染）——只要 MFMA 在，时间就不变。

S=8 每 workgroup 时间线（`wg_timeline.py`，GateUp）：

| 配置 | task 时长 µs（min/中位/max） | 第 1 轮结束 | 第 2 轮 | WG 跨度 |
|---|---|---|---|---|
| 19 block = 304 task（1 轮） | 113.7 / 118.2 / 122.4 | 115–123 µs | – | 122.8 µs |
| 32 block = 512 task | 93.5 / 113.4 / 117.2 | 112–117 µs | 208 task 在 117 µs 开始，~97 µs/task | 216.0 µs |
| S=2，32 block | 48.1 / 69.8 / 72.2 | 67–73 µs | ~50 µs/task | 123.4 µs |

8 个 XCD 的结束时间相差 ≤3 µs（task 分配均衡）。第 2 轮只有 208/304 个 CU 在跑，单 task 更快（功耗/带宽余量），
但 1.68 轮仍比"可整除"多付 ~16%。ATT（S=8 GateUp，单 CU 全部 SIMD）：循环内每 K block 中位 2508 周期（MFMA 2048，82%），
每 task 序言 7–16k 周期（依赖链：num_valid → sorted_ids/expert → A 行 → 首个 B 块）、epilogue 8–9k 周期（SiLU 的
exp/rcp 是 1/4 速率 VALU）。

K 循环重排（A 用 3 个 LDS buffer 提前两块写入、下一块 slice 0 在 barrier 前读入寄存器、块边界 LDS 工作放到最后一个
slice 的 MFMA 之下、访存放在每个 slice 的 MFMA 之后）与原版交替 A/B（32 block，6 轮）：

| rows | GateUp 原版 → 重排 | Down 原版 → 重排 |
|---|---|---|
| 128 | 214.3 → 215.6（0.993×） | 121.6 → 120.4（1.008×） |
| 64 | 143.7 → 144.7（0.997×） | 81.0 → 78.7（1.029×） |
| 32 | 130.6 → 133.6（0.978×） | 70.1 → 70.3（0.997×） |
| 16 | 129.3 → 129.3（0.999×） | 67.0 → 67.0（1.002×） |

无收益，保留原版（更少 LDS、更简单）。

## 4. 最终结果（ring=2，slice 1..8，sort 清零 + Down 原子加）

`tools/summarize.py ... --hbm-tbps 4.95 --mfma-tflops 1151`（GPU0 探针，`roofline_probe.json`）。所有路径都含 sort，
单位 µs，中位数。"tuned asm" = 11 个 asm kernel 中含 sort 最快的那条路径（括号内为 kernel）；"per-round" = 4 轮各自的中位数
（从小到大），以及对应的 tuned asm / small 比值范围。

### 4.1 持续态（bench 默认：每个变体先跑 ≥50 ms 再计时）

ep_uniform（`final_ep_uniform.json`；T=4096 一行取 `final_ep_uniform_t4096.json`：主运行的 T=4096 第 1–3 轮 gate 记录到
外部进程 12–29% activity，单独重测 gate PASS）：

| T | routes/expert | asm auto | tuned asm | BM256 | **small** | **small vs tuned asm** | vs asm auto | small w/o sort | per-round | 比值范围 |
|---|---|---|---|---|---|---|---|---|---|---|
| 256 | 8 | 261.7 | 260.4 (vs_ps_32x256) | 429.0 | 215.5 | **1.209×** | 1.215× | 200.7 | 211/214/218/245 | 1.06–1.23× |
| 512 | 16 | 274.8 | 272.4 (vs_32x256) | 431.9 | 220.2 | **1.237×** | 1.248× | 201.7 | 217/218/219/250 | 1.09–1.26× |
| 1024 | 32 | 290.2 | 290.2 (auto) | 443.0 | 248.6 | **1.167×** | 1.167× | 229.0 | 222/236/250/258 | 1.12–1.31× |
| 2048 | 64 | 381.3 | 332.4 (vs_psx_64x256) | 461.0 | 288.4 | **1.152×** | 1.322× | 251.3 | 282/287/289/330 | 1.01–1.18× |
| 4096 | 128 | 556.8 | 494.3 (vs_64x256) | 481.1 | 371.6 | **1.330×** | 1.498× | 332.2 | 370/373/374/377 | 1.31–1.33× |

random（`final_random.json`，gate 全部 PASS）：

| T | routes/expert | asm auto | tuned asm | BM256 | **small** | **small vs tuned asm** | vs asm auto | small w/o sort | per-round | 比值范围 |
|---|---|---|---|---|---|---|---|---|---|---|
| 256 | 3–13 | 261.5 | 260.1 (novs_ps_32x256) | 422.3 | 220.3 | **1.181×** | 1.187× | 206.0 | 212/215/245/252 | 1.03–1.23× |
| 512 | 11–24 | 271.8 | 269.2 (vs_ps_32x256) | 426.7 | 242.6 | **1.109×** | 1.120× | 224.5 | 219/221/243/245 | 1.10–1.23× |
| 1024 | 22–44 | 313.2 | 287.7 (vs_64x256) | 446.1 | 250.4 | **1.149×** | 1.251× | 229.4 | 225/227/256/262 | 1.10–1.28× |
| 2048 | 47–77 | 352.2 | 352.2 (auto) | 452.6 | 285.2 | **1.235×** | 1.235× | 249.0 | 282/284/288/291 | 1.21–1.25× |
| 4096 | 96–149 | 561.8 | 524.2 (vs_64x256) | 489.0 | 464.1 | **1.129×** | 1.210× | 427.3 | 458/459/465/474 | 1.11–1.14× |

small 路径在持续态有两个稳定档（每轮内 8 次一致、轮与轮之间跳变，差 ~12–16%，GateUp+Down 部分变化，sort 不变），
asm 各轮稳定（±2%）。同一组 kernel 在上一次 ring=4 的正式运行（`superseded/final_*_ring4.*`）里 random T=512 为 1.232×、
T=1024 为 1.159×：中位数落在哪一档决定了 1.11× 还是 1.23×。原因是功耗墙（§1 的 728 W / 1237 MHz），见
`docs/record/moe_small_m_gfx942.md` §3.7。

单 kernel（持续态，ep_uniform / random）：

| T | sort128 清零 | GateUp small | Down small | GateUp+Down | sort256 | GateUp BM256 | Down BM256 | 最快 asm kernel（不含 sort） |
|---|---|---|---|---|---|---|---|---|
| 256 | 11.0 / 10.1 | 131.6 / 130.9 | 68.2 / 68.1 | 199.7 / 199.0 | 10.4 / 9.6 | 264.6 / 261.9 | 150.6 / 144.1 | 246.4 / 247.8 |
| 512 | 12.7 / 12.0 | 146.8 / 151.7 | 68.2 / 76.6 | 215.0 / 228.3 | 12.3 / 11.7 | 275.5 / 271.6 | 149.5 / 147.6 | 256.2 / 253.7 |
| 1024 | 13.9 / 13.0 | 150.1 / 150.9 | 81.0 / 80.8 | 231.1 / 231.8 | 12.7 / 11.7 | 279.6 / 281.7 | 145.8 / 149.2 | 270.6 / 267.1 |
| 2048 | 21.8 / 21.0 | 185.2 / 189.2 | 98.9 / 94.0 | 284.1 / 283.2 | 18.7 / 18.0 | 293.8 / 290.4 | 150.6 / 148.9 | 306.6 / 339.9 |
| 4096 | 26.3 / 25.5 | 228.3 / 283.7 | 129.6 / 154.3 | 357.9 / 438.0 | 19.0 / 18.4 | 301.5 / 300.5 | 164.6 / 166.2 | 477.2 / 493.9 |

### 4.2 突发态（`--warmup-ms 0`：每块 8 次预热）

| T | ep_uniform: tuned asm | small | **vs tuned asm** | random: tuned asm | small | **vs tuned asm** |
|---|---|---|---|---|---|---|
| 256 | 259.5 (vs_ps_32x256) | 210.8 | **1.231×** | 259.4 (vs_ps_32x256) | 208.4 | **1.245×** |
| 512 | 268.9 (vs_psx_64x256) | 213.9 | **1.257×** | 270.2 (vs_psx_64x256) | 213.6 | **1.265×** |
| 1024 | 287.0 (vs_psx_64x256) | 221.5 | **1.296×** | 286.7 (vs_psx_64x256) | 224.0 | **1.280×** |
| 2048 | 328.8 (vs_psx_64x256) | 272.4 | **1.207×** | 338.1 (vs_32x256) | 266.0 | **1.271×** |
| 4096 | 524.8 (vs_psx_64x256) | 385.2 | **1.362×** | 553.0 (vs_64x256) | 484.9 | **1.140×** |

突发态 ep_uniform T=2048 各轮 257/290/255/290：反序轮（small 紧跟 60 ms 空闲之后）慢，正序轮（跟在 asm 之后）快。

### 4.3 roofline 对照（持续态 GateUp+Down kernel 之和，不含 sort）

| T | routes | GFLOP | MiB | t_mem | t_mfma | 类型 | max | 和 | ep_uniform | /max | /和 | random | /max | /和 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 256 | 256 | 12.9 | 774 | 164.0 | 11.2 | memory | 164.0 | 175.2 | 199.7 | 1.22 | 1.14 | 199.0 | 1.21 | 1.13 |
| 512 | 512 | 25.8 | 780 | 165.2 | 22.4 | memory | 165.2 | 187.6 | 215.0 | 1.30 | 1.15 | 228.3 | 1.38 | 1.21 |
| 1024 | 1024 | 51.5 | 792 | 167.8 | 44.8 | memory | 167.8 | 212.6 | 231.1 | 1.38 | 1.09 | 231.8 | 1.38 | 1.09 |
| 2048 | 2048 | 103.1 | 816 | 172.9 | 89.6 | memory | 172.9 | 262.4 | 284.1 | 1.64 | 1.08 | 283.2 | 1.64 | 1.09 |
| 4096 | 4096 | 206.2 | 864 | 183.0 | 179.1 | 拐点 | 183.0 | 362.1 | 357.9 | 1.96 | **0.99** | 438.0 | 2.39 | 1.21 |

（random 行的 routes/GFLOP 与 ep_uniform 相差 <2%，t_mem/t_mfma 取 ep_uniform 的。）
