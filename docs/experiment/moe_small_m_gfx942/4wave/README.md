# 4-wave small-M + 256 行 sort 表内分支 —— 实验数据

对应记录：[docs/record/moe_small_m_gfx942.md](../../../record/moe_small_m_gfx942.md) §6。上一版（8 wave、128 行 sort、host 端按
shape 选择）的数据在上一级目录（[../README.md](../README.md)），其源码作为对照保存在 `variants/`。

**本页是第二版**（GateUp 64 列、A2 按 64 列量化）的数据，源码保存在 `variants/moe_small_m_4wave_64col.py`。当前代码是第三版
（每个 workgroup 128×256 输出 tile），数据在 [../4wave_128/README.md](../4wave_128/README.md)；`tools/` 由两版共用（第三版
加了 `run_128.sh`、`summarize128.py`，`ab_kernels.py` 加了按 GFX activity 的空闲门控，`make_bench_ab.py` 加了 c64 路径）。

## 环境

- AMD Instinct MI300X（gfx942:sramecc+:xnack-），304 CU，750 W 功耗墙；本组数据用 GPU 1（BDF 0000:22:00.0），kernel A/B 的部分
  对照用 GPU 2。
- torch 2.11.0+rocm7.2，HIP 7.2.26015，FlyDSL 0.4.0，aiter（`/sgl-workspace/aiter`），rocprofv3（ATT 用
  `librocprof-trace-decoder`）。
- 另一个容器的进程间歇性占用全部 8 张卡（常驻 ~133 GB 显存，GFX activity 30–100%）。因此：bench 每个计时块由
  `gpu_contention.ExternalLoadMonitor` 监视，期间有外部 kernel 就重测该块；`tools/run_4w.sh` 只在连续 3 次（间隔 10 s）
  activity ≤3% 后开始一次 bench。下表 5 个正式 JSON 的所有 gate 记录 activity ≤5%、显存 ≤4.7 GB（`gates PASS`），
  没有被重测的块。

## 文件

| 文件 | 内容 |
|---|---|
| `final_ep_uniform.{log,json}` | 持续态（每个变体先跑 ≥50 ms），ep_uniform，T=256..4096：asm 6 个、bm256、small、old8 路径和单 kernel |
| `final_random.{log,json}` | 同上，random 路由 |
| `burst_ep_uniform.{log,json}`、`burst_random.{log,json}` | 突发态（`--warmup-ms 0`），asm 6 个、small、old8 |
| `final_large.{log,json}` | 持续态 random T=6144/8192（大部分 block 走 BM256）：asm、bm256、small |
| `att_summary.txt` | ATT 分析输出：lane 版与当前版 GateUp 128 行（S=8），当前版 64 行（S=4） |
| `variants/moe_small_m_8wave_sort128.py` | 上一版 small-M（8 wave、128 行 sort、per-128 A2 scale），A/B 的 old8 |
| `variants/moe_small_m_4wave_lane_scales.py` | 4-wave 第一版：每个 wave 自己读 A scale（16 VMEM/wave/K block），A/B 的 lane |
| `tools/make_bench_ab.py` | 由仓库 bench 生成 `bench_ab.py`：多一个 old8 路径（sort128 清零 + 8-wave GateUp/Down 原子加）和 old8 单 kernel |
| `tools/run_4w.sh` | 依次跑上面 5 个 bench（等空闲窗口） |
| `tools/summarize4.py` | 从 JSON 打印本页的表（含每轮中位数） |
| `tools/ab_kernels.py` | 合成 block 上的 kernel 级交替 A/B（old8 / lane / cur / 任意模块文件，`name@ring`） |
| `tools/check_equal.py` | 当前 kernel 与 lane 版逐位比较、20 次调用的确定性检查（LDS 竞争就是这样发现的） |
| `tools/run_one.py` | 单个 kernel 的 rocprofv3（PMC/ATT）驱动 |
| `tools/att_phases2.py` | 两 barrier K 循环的每 K block 阶段、按指令类的 stall 份额、两 wave 同驻时每 SIMD 的 MFMA 管线占用 |

## 复现

```bash
export PYTHONPATH=<含 pyhip -> src 链接的目录>:$PYTHONPATH
cd tests/contrib/moe
python -m pytest -q test_moe_small_m_gfx942.py                      # 32 passed, 4 skipped
D=../../../docs/experiment/moe_small_m_gfx942/4wave
GPU=1 $D/tools/run_4w.sh                                           # 5 个 bench → $D/*.{log,json}
python $D/tools/summarize4.py $D/final_ep_uniform.json $D/final_random.json $D/burst_*.json $D/final_large.json
HIP_VISIBLE_DEVICES=1 python $D/tools/ab_kernels.py --impls old8 lane cur --rows 16 32 64 128
HIP_VISIBLE_DEVICES=1 python $D/tools/check_equal.py 256 ep_uniform
rocprofv3 --att --att-library-path <decoder> --att-target-cu 1 --att-shader-engine-mask 0x1 --att-simd-select 0xf \
    --att-buffer-size 0x10000000 --kernel-include-regex small_m -d att -o att -- \
    python $D/tools/run_one.py --impl cur --rows 128 --calls 1
python $D/tools/att_phases2.py att/ui_output_* 32
```

## 1. kernel 级 A/B（合成：32 个 block × rows 行，每 expert 一个，同一份权重反复调用）

`tools/ab_kernels.py`，8 轮交替取中位数（µs）。同一权重反复读，绝对值偏乐观，比的是比例。old8 用 128 行 block，lane / cur
用 256 行 block 的前 rows 行。

| rows (S) | GateUp old8 | lane | cur | Down old8 | lane | cur |
|---|---|---|---|---|---|---|
| 16 (1) | 125.7 | 122.5 | 122.6 | 65.7 | 66.0 | 65.8 |
| 32 (2) | 126.7 | 130.0 | 128.4 | 70.0 | 70.9 | 69.4 |
| 64 (4) | 139.7 | 167.9 | 145.8 | 78.2 | 99.2 | 79.2 |
| 128 (8) | 213.4 | 267.7 | 222.1 | 120.4 | 164.3 | 120.1 |

- lane（第一版）在 S≥4 时比 old8 慢 16–27%；cur（A scale 改为每 WG 读一次、经 LDS 共享，A tile 单缓冲）在 S=8 时
  GateUp −17%、Down −27%，与 old8 持平（GateUp 0.96×，Down 1.00×）。
- 均无收益、已撤销的尝试（同一工具，交替测量）：
  - ring=4（GateUp 230 / Down 246 VGPR 仍可 2 wave/SIMD）：128 行 GateUp 224.5 vs 221.9，Down 126.3 vs 120.4。
  - S≤6 时 A tile 双缓冲（每 K block 1 个 barrier，S=7/8 仍单缓冲；LDS 32 KiB）：S=2/4 差 <1%。
  - 每 K block 先读 slice 0 的 LDS 再发 B load：64 行 +0.7%，128 行 −1.1%。
  - B load 全部移到 memory phase（第二个 barrier 前）：16/32 行 ±1%，64 行 −3%，128 行 −2%（GateUp）/ −4.5%（Down）。

## 2. 第一版（lane）的瓶颈：VMEM 指令发射

- ISA：每 wave 每 K block 发 16 条 VMEM（B 4 × dwordx4、A 4 × dwordx4、A scale 8 × dword，每条 dword load 16 个不同
  cache line）；每 CU（8 wave）128 条。
- ATT（`att_summary.txt` 第一段，128 行，GateUp）：每 K block 中位 2616 周期，其中 MFMA 跨度 1116，**buffer_load 的
  发射 stall 中位 894（p90 2857）**，`s_waitcnt vmcnt` 只有 72——数据按时到达，是发射端（TA 输入队列）排队。
- PMC（rocprofv3，128 行 GateUp，265 µs）：TCC_EA0_RDREQ 4.356 M × 128 B = 531 MiB（= 512 MiB 权重 + 首次读 A），
  A 在 L2 命中，不是 HBM 流量问题；SQ_WAIT_INST_ANY 占 wave 周期 47%（old8 为 34%）。TA_* 计数器会让 rocprofv3 挂起，未采到。
- 修正：A scale 每 WG 每 K block 只读一次（每线程 1 条 dword，GateUp 两半线程读同一行），写入 LDS 供 4 个 wave 读；
  为把 LDS 留在 32 KiB 内，A tile 改为单缓冲、每 K block 两个 barrier。每 wave 每 K block 的 VMEM 降到 4+4+1（S=8）。
  同时把 Down 的 A2 scale 改为 [I/64, capacity]（每 K block 半段 128 行连续），GateUp 的写也连续。

## 3. 当前版 ATT（`att_summary.txt` 第二、三段）

时钟：`realtime.json` 给出 gfx clock 1.235 GHz（功耗墙下）。

- S=8（128 行 GateUp）：每 K block 中位 2014 周期，compute 1536（MFMA 跨度 1144，64 个 MFMA 几乎背靠背），barrier 1 等待 60，
  memory phase ~216。两 wave 同驻窗口内每个 SIMD 的 MFMA 管线占用 **74.5–75.7%**。stall 分布：MFMA 47%（两 wave 抢 MFMA，
  属正常）、LDS 17%、waitcnt 16%、VMEM 14%。
- 同一 CU 的 workgroup 时间线（occupancy 事件，9 个 CU）：两个同驻 WG 一个 74 µs 结束、另一个 ~100 µs 结束（MFMA 仲裁偏向
  老 wave，WG 确实相互错开）；第 3 个 WG 74→142 µs，第 4 个 100→176 µs。1024 个任务、608 个槽：有 4 个 WG 的 CU 在 176 µs
  结束，只有 3 个的在 ~142 µs 结束——尾轮量化约 20%。
- S=4（64 行，GPU 2）：每 K block 1412 周期，MFMA 跨度 564，MFMA 占用 52%；stall 中 VMEM 占 38.6%，几乎全在每个 K block 的
  第一条 B load（中位 64、均值 173、p90 569 周期）和第一条 A load（均值 60）。两个 wave 在 barrier 后同时发 B load。

## 4. 正确性

- `test_moe_small_m_gfx942.py`：32 passed、4 skipped（GateUp：per-64 行 scale 与 FP32 参考、FP8 重量化失配 <1%、BM256 block 与
  capacity 不被写；GateUp+Down（ring 2/4）；aiter sort 上的 mixed 路径 T=64..8192（只有 small、混合、只有 BM256）；
  `fused_moe_a8w8_blockscale_gfx942_sort`）。fused 测试的参考对 small block 的 route 按 (行, 64) 重量化，对 BM256 block 按
  (行, 128)（asm 测试的参考固定按 128，per-64 的结果与它的 calc_diff 是 1.4e-4–2.3e-4，超过它的 1e-4 阈值）。
- bench 中 small 路径的 calc_diff：ep_uniform 1.38–1.42e-6，random 2.53–2.69e-6，与 old8 相同量级（asm 8e-6–1.3e-5）。
- **LDS 竞争（已修复）**：单缓冲 A tile 第一次实现时，`check_equal.py` 显示 GateUp 20 次调用中 8 次与 lane 版不一致、Down
  10 次中 9 次不一致（fused 测试偶发 1–15 个元素超差）。S=1 的 ISA 里，compute(k) 的 MFMA 被指令选择（SelectionDAG）挪到了
  两个 barrier 之后（`sched_barrier` 只约束 machine scheduler），于是 `ds_read` 在 barrier 1 时还没返回，别的 wave 已开始写
  A(k+1)。在 barrier 1 前加显式 `s_waitcnt lgkmcnt(0)` 后，4 种 T 下 GateUp 与 lane 版逐位一致、Down 一致。
- prologue 的发射顺序：LLVM 的 waitcnt 在循环头合并 prologue 与回边两条路径，取更严格者。prologue 若最后才发 B（或调度器
  把它排到 A 之后），循环里每个 K block 第一次等 B 时会连带等刚发出的 A；prologue 按循环顺序发（并用 `sched_barrier` 固定）
  后，循环内的 vmcnt 只覆盖 B。

## 5. 路径级结果（含 sort，中位数 µs；tuned asm = 6 个 asm kernel 中含 sort 最快者）

small = aiter `moe_sorting(256)`（清零 out）+ `fused_moe_a8w8_blockscale_gfx942_mixed`：small-M GateUp、BM256 GateUp
（big_blocks_only）、small-M Down、BM256 Down（big_blocks_only），T>128 时 5 个 kernel；old8 = 上一版（sort128 清零 + 8-wave
GateUp/Down，3 个 kernel）。每轮中位数见 `summarize4.py` 输出。

### 5.1 持续态

| T | ep: tuned asm | small | old8 | small/asm | old8/asm | random: tuned asm | small | old8 | small/asm | old8/asm |
|---|---|---|---|---|---|---|---|---|---|---|
| 256 | 260.4 | 218.1 | 213.3 | **1.194** | 1.221 | 261.1 | 219.1 | 219.0 | **1.191** | 1.192 |
| 512 | 274.8 | 237.4 | 217.4 | **1.158** | 1.264 | 272.4 | 229.0 | 247.1 | **1.189** | 1.102 |
| 1024 | 292.3 | 242.1 | 255.7 | **1.207** | 1.143 | 289.6 | 258.4 | 226.4 | 1.121 | 1.280 |
| 2048 | 332.3 | 304.0 | 285.4 | 1.093 | 1.164 | 356.8 | 309.1 | 286.3 | **1.154** | 1.246 |
| 4096 | 497.9 | 396.0 | 376.7 | **1.258** | 1.322 | 521.9 | 621.6 | 461.6 | 0.840 | 1.131 |

持续态有功耗档切换（每轮中位数，例如 ep T=512 small 256/223/254/219、old8 215/218/249/215），单个点的比值可差 ~10%。
同一会话前一次（gate 未通过、但无重测块）的 ep_uniform：small 1.195/1.178/1.110/1.097/1.231，old8 1.226/1.198/1.134/1.172/1.357。

### 5.2 突发态

| T | ep: tuned asm | small | old8 | small/asm | old8/asm | random: tuned asm | small | old8 | small/asm | old8/asm |
|---|---|---|---|---|---|---|---|---|---|---|
| 256 | 258.5 | 211.6 | 211.3 | **1.222** | 1.223 | 260.1 | 212.1 | 210.4 | **1.226** | 1.236 |
| 512 | 272.0 | 217.7 | 214.6 | **1.250** | 1.268 | 273.4 | 221.3 | 214.5 | **1.235** | 1.274 |
| 1024 | 287.5 | 230.0 | 222.9 | **1.250** | 1.290 | 287.4 | 242.2 | 227.2 | **1.186** | 1.265 |
| 2048 | 333.1 | 288.5 | 262.9 | **1.154** | 1.267 | 365.4 | 323.2 | 272.6 | 1.131 | 1.341 |
| 4096 | 528.9 | 420.5 | 382.7 | **1.258** | 1.382 | 557.3 | 628.0 | 492.0 | 0.887 | 1.133 |

### 5.3 单 kernel（持续态 ep_uniform / random，中位数 µs）

| T | sort256Z | sort128Z | GateUp small | Down small | GateUp old8 | Down old8 | BM256 GateUp (big only) | BM256 Down (big only) |
|---|---|---|---|---|---|---|---|---|
| 256 | 10.8 | 10.8 | 145.7 / 146.0 | 69.3 / 69.7 | 148.0 / 132.5 | 70.0 / 69.5 | 5.9 / 5.9 | 5.9 / 5.9 |
| 512 | 12.6 | 12.7 | 144.7 / 148.1 | 70.7 / 82.5 | 135.5 / 148.1 | 69.8 / 77.5 | 5.9 / 5.9 | 5.8 / 5.9 |
| 1024 | 13.9 | 13.8 | 149.4 / 157.1 | 83.1 / 85.2 | 150.6 / 152.8 | 82.1 / 82.1 | 5.9 / 5.9 | 5.9 / 5.9 |
| 2048 | 21.9 | 21.7 | 166.4 / 180.1 | 95.8 / 97.9 | 184.1 / 171.9 | 91.6 / 92.8 | 5.8 / 5.8 | 5.8 / 5.8 |
| 4096 | 26.3 | 26.1 | 235.2 / 178.2 | 132.7 / 101.6 | 220.1 / 289.1 | 128.3 / 150.8 | 6.4 / **213.6** | 6.5 / **103.2** |

- 没有大 block 时，两个 BM256 kernel 每个 5.8–6.5 µs（每个 WG 只读 num_valid 和第 128 行就退出）；路径里它们与相邻 kernel
  的启动/收尾一起，使 small 路径比 old8 多 ~5–20 µs。
- random T=4096：每 expert 96–149 个 route，12 个 expert 超过 128，它们的 block 走 BM256：BM256 GateUp/Down 只处理这 12 个
  block 却用了 213.6 / 103.2 µs（同一 kernel 处理全部 block 时为 298.2 / 163.8 µs——任务少时尾轮和 256 行 tile 的固定开销
  占主导），small 路径因此只有 asm 的 0.84–0.89×。old8 把这些 expert 拆成 128 + 尾块（权重读两遍）仍有 1.13×。

### 5.4 大 T（持续态 random，大部分 block 走 BM256）

| T | tuned asm | small（mixed） | bm256（原 BM256 路径，single_store） | small/asm | bm256/asm |
|---|---|---|---|---|---|
| 6144 | 710.7 | 543.5 | 531.0 | 1.308 | 1.338 |
| 8192 | 873.8 | 850.0 | 714.4 | 1.028 | 1.223 |

T=6144：32 个 expert 都有 159–226 个 route，全部 block 走 BM256，small 路径只多两个空的 small-M launch 和 Down 的原子加
（bm256 路径用 single_store）。T=8192：16 个 expert 有 257–293 个 route（一个满 block 加 1–37 行的尾 block，尾 block 走
small-M），另 16 个是一个 224–256 行的 block。两类 block 分成两次串行 launch、各自有尾轮，比原 BM256 路径（一次 launch
处理全部 block）慢 19%。

### 5.5 侧流（未采用）

把两个 BM256 kernel 放到侧流上（sort 后 fork、Down 后 join）与 small kernel 并行（GPU 2，交替 8 轮）：T=512 ep 227.1 vs
213.8 µs、T=2048 ep 286.3 vs 272.5、T=6144 random 614.9 vs 590.5（变慢 4–6%），只有 T=4096 random 595.4 vs 623.4（快 4.5%）。
跨流同步的开销大于省下的空 launch，未采用。
