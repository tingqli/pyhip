# gfx942 FP8 block-scale MoE：每 expert ≤128 routes 的 small-M 分支（分析、实现、优化记录）

本文记录三版实现、一次运行时选路的尝试和 BM256 kernel 内的 128 行分支。**现在的代码只有两条路径**（§10）：
aiter 入口 `fused_moe_a8w8_blockscale_gfx942`（BM256 + `single_store`，GateUp / Down 内的 128 行分支 `bm128` 默认开，
§9）；`fused_moe_a8w8_blockscale_gfx942_sort` / `_mixed` 的方案 A（merged，§8），每个 stage 一次 launch，有第 128 行的
block 走 BM256 tile，其余走 8-wave pair 任务（`src/contrib/flydsl/moe_merged_a8w8_blockscale_gfx942.py`）。BM256 GateUp / Down
本身也只保留这两条路径用到的配置（§11）。第一至第三版
和方案 B 的 4-wave small-M kernel 已从 `src` 删除，最后一版在 commit da764f5。§7 是第三版（4 wave、每 CU 2 个
workgroup、128×256 输出 tile 的 small-M kernel 加 BM256 kernel，两次 launch），§6 是第二版（4 wave、与 BM256 共用 256 行 sort 表、
在 kernel 内按 block 选择路径），§1–§5 是第一版（8 wave、128 行 sort、host 端按 shape 选择）的分析与优化过程，roofline
部分各版通用；前两版源码作为对照保存在
[docs/experiment/moe_small_m_gfx942/4wave/variants/](../experiment/moe_small_m_gfx942/4wave/variants/)。

数据与复现命令：第一版 [docs/experiment/moe_small_m_gfx942/README.md](../experiment/moe_small_m_gfx942/README.md)，
第二版 [docs/experiment/moe_small_m_gfx942/4wave/README.md](../experiment/moe_small_m_gfx942/4wave/README.md)，
第三版 [docs/experiment/moe_small_m_gfx942/4wave_128/README.md](../experiment/moe_small_m_gfx942/4wave_128/README.md)，
§8 [docs/experiment/moe_small_m_gfx942/merged/README.md](../experiment/moe_small_m_gfx942/merged/README.md)、
[docs/experiment/moe_small_m_gfx942/split/README.md](../experiment/moe_small_m_gfx942/split/README.md)，
§9 [docs/experiment/moe_small_m_gfx942/bm128/README.md](../experiment/moe_small_m_gfx942/bm128/README.md)。
代码：

- [src/contrib/flydsl/moe_merged_a8w8_blockscale_gfx942.py](../../src/contrib/flydsl/moe_merged_a8w8_blockscale_gfx942.py)：
  方案 A 的 pair 任务、任务扫描 plugin 和 `moe_gateup_merged`、`moe_down_merged`。
- [src/contrib/flydsl/moe_a8w8_blockscale_gfx942.py](../../src/contrib/flydsl/moe_a8w8_blockscale_gfx942.py)：
  aiter 入口 `fused_moe_a8w8_blockscale_gfx942`（`bm128=True`）、`fused_moe_a8w8_blockscale_gfx942_mixed`（256 行 sort 表上的
  merged kernel，超过任务扫描容量时退回 BM256 + bm128）、`fused_moe_a8w8_blockscale_gfx942_sort`（sort + mixed）。
- [src/contrib/flydsl/moe_gateup_a8w8_blockscale_gfx942_lds.py](../../src/contrib/flydsl/moe_gateup_a8w8_blockscale_gfx942_lds.py)、
  [src/contrib/flydsl/moe_down_a8w8_blockscale_gfx942_lds.py](../../src/contrib/flydsl/moe_down_a8w8_blockscale_gfx942_lds.py)：
  BM256 kernel 的 `bm128`（§9），以及方案 A 用的 `merged=` hook。
- [tests/contrib/moe/test_moe_small_m_gfx942.py](../../tests/contrib/moe/test_moe_small_m_gfx942.py)、
  [tests/contrib/moe/bench_moe_small_m_gfx942.py](../../tests/contrib/moe/bench_moe_small_m_gfx942.py)、
  [tests/contrib/moe/roofline_probe_gfx942.py](../../tests/contrib/moe/roofline_probe_gfx942.py)、
  [tests/contrib/moe/gpu_contention.py](../../tests/contrib/moe/gpu_contention.py)。

## 0. 结论

### 精简（§10、§11）

- 只保留 BM256 + bm128（aiter 路径）和 merged（`_sort` / `_mixed`）。4-wave small-M kernel、方案 B（split）、第三版的两次
  launch 路径和 BM256 的 `big_blocks_only` 已删除；保留的 kernel 机器码不变。
- BM256 GateUp / Down 只保留 aiter 路径的配置：GateUp 2x4 + 排序后的 FP8 输出，Down 4x2 + 排序后的 A2 + BF16 atomic，
  可选 `single_store`、`bm128`、`merged`。`layout`、`xcd_swizzle`、`pair_blocks`、GateUp 的 BF16 输出、Down 的未排序 A2 和
  `reduce="kernel"` 已删除（最后一版 1a4fc73）。GateUp 机器码不变；Down 少一个 kernel 参数，只有 kernarg 偏移和 SGPR 分配不同。

### BM256 kernel 内的 128 行分支（§9，aiter 路径）

- grid、workgroup、任务顺序与 aiter 路径的 BM256 相同；block 的第 128 行无 route 时，GateUp / Down 跑 128 行
  流水线。`fused_moe_a8w8_blockscale_gfx942` 默认开启，aiter 侧不用改。BM256 的 K 循环 opcode 不变。
- 相对 tuned asm（含 sort）：T ≤ 2048 从 0.55–0.79× 提到 0.83–1.03×；ep_uniform T=4096 从 1.04× 提到 1.28×；
  T ≥ 6144 与 BM256 持平或更好（ep_uniform T=12288 1.61×）。
- T ≤ 2048 的速度只有各点最好路径（small / split / merged，asm 的 1.15–1.44×）的 0.64–0.75 倍；random T=8192 为
  merged 的 0.89–0.90 倍。

### 运行时按 block 选路（§8）

- 要求：kernel 内按每个 256 行 block 的实际 route 数选择，每个 stage 一次 launch。先试 B，再试 A。
  - B（split，`src`）：全部用 4-wave small-M kernel，有第 128 行的 block 拆成两个 128 行任务。
  - A（merged，`src`，现为默认）：一个 8-wave kernel，大 block 走 BM256 tile，其余走 pair 任务。
  - 两者都用 kernel 内的稠密任务表（ballot + `mbcnt` 扫描 sort 表）。
- 结果（相对 tuned asm，含 sort，§8.5）：
  - B 在 T≤4096 最好或接近最好，random T=4096 为 1.129×；T≥6144 比 A 慢 0–21%（10 个点中 7 个慢 9% 以上）。
  - A 在 20 个点里最差比最好的路径慢 8%，在 random T=8192（1.313×）和大 T random 最好。
  - 两者都消除了第三版在 random T=4096 / 8192 的短板（0.923× / 1.054×）和空 launch。
- 采用 A 作为默认运行时路径（§8.7）。A 在大 T ep_uniform 与 bm256 路径差 5–8%，差别来自路径（`single_store` / 清零），
  不来自 kernel。
- 方法上的教训：FlyDSL 磁盘缓存（import aiter 后开启）的键不含 plugin 对象，A 的前三次正式运行测的都是第一版
  kernel（§8.4 第 3 点）；现在 plugin 带源码哈希 key。

### 第三版（§7，`big_blocks="bm256"`）

- 每个 workgroup 输出 128×256 的 MFMA tile：GateUp 128 个 gate + 128 个对应 up 权重行（128 个输出列），Down 256 个输出列；
  每个 wave 4×16 行权重。仍是 4 wave + occupancy 2（GateUp 256、Down 254 VGPR，不 spill；LDS 18 / 16.5 KiB）。
- 第二版只算 64 列是按寄存器估算选的：128×256 的 accumulator 在 S=8 时就是 128 VGPR，直接实现 S=8 要 ~300。
  实际做法是按 slice 数选布局：S≤6 用原来的（两套 partial、下一 slice 的 A 预读、A 两个暂存槽），S=7/8 用
  light 布局（一套 partial、A 按半个 slice 读、一个暂存槽），再把 B 的行偏移放进 soffset、拦住 LLVM 把各变体共同的
  prologue 提到分支外（§7.3、§7.4）。
- A2 回到按 (行, 128 列) 量化，与 BM256 同一布局、同一个 buffer；fused 路径的参考与 asm 测试相同。
- kernel 级（交替 A/B）：比 8-wave 版 GateUp 快 2–7%、Down 快 4–10%，比第二版 GateUp 快 5–12%。
- 与 tuned asm 比（含 sort）：ep_uniform 突发态 1.286–1.337×、持续态 1.157–1.366×；random T≤2048 突发态
  1.263–1.297×、持续态 1.159–1.307×。路径上与 8-wave 版持平到快 5%（突发态）：kernel 的领先被两个空的 BM256 launch
  （各 ~6 µs）抵掉一部分。**random T=4096 仍是 0.88–0.91×**，T=8192 random 比 BM256 路径慢 18%：混合 block
  的问题在 256 行表内分支，不在 tile（§6.5）。

### 第二版（§6）

- 4 wave + occupancy 2：GateUp 198 VGPR / 16.5 KiB LDS，Down 208 VGPR / 17 KiB LDS，`"amdgpu-waves-per-eu"="2,2"`，每 CU 两个
  workgroup、每 SIMD 两个 wave。每 K block 分 compute phase（LDS 读 A、MFMA、上一 slice 的 dequant）和 memory phase（VMEM
  发射、两个 barrier 之间写 LDS，无 VALU）。ATT：两个 WG 同驻时确实互相错开，S=8 时每 SIMD MFMA 管线占用 75%。
- 256 行 sort 表 + kernel 内分支：small-M kernel 与 BM256 kernel 都按 256 行 block 划分任务，各自读 block 的第 128 行：
  无效（≤128 个 route）走 small-M，有效走 BM256，另一方的 workgroup 直接退出。
- 精度：small block 的 A2 改为按 (行, 64 列) 量化（4-wave GateUp 的 N tile 是 64 列），calc_diff 与第一版同量级。
- 与 tuned asm 比（含 sort）：ep_uniform 突发态 1.154–1.258×、持续态 1.093–1.258×；random 突发态 1.131–1.235×、持续态
  1.121–1.191×（T≤2048）。**random T=4096 只有 0.84–0.89×**：12 个超过 128 route 的 expert 的 block 走 BM256，BM256 kernel
  只处理 12 个 block 时效率很低。T=8192 random（满 block + 尾 block 混合）比原 BM256 路径慢 19%。
- 与第一版比：kernel 级（同一会话）S≤2 持平，S=4/8 时 GateUp 慢 ~4%（持续态 T=4096 慢 7%）、Down 持平到慢 3%；路径级多两个
  （多数时候是空的）BM256 launch。small/old8 的时间比（>1 表示第二版快）：ep_uniform 0.91–1.00（持续态 T=1024 为 1.06，
  功耗档），random T≤1024 0.88–1.08，random T=2048 0.84–0.93，random T=4096 0.74–0.78。4-wave 的 N tile 是 64 列，A 的全局读、
  LDS 写、barrier 每输出列都是 8-wave 版的 2 倍，两个 WG 错开带来的重叠没有抵消这部分开销。

### 第一版（§1–§5）

- **实验分析（roofline）**：整个每 expert ≤128 routes 的区间（MiMo EP8 rank，T=256..4096）都是 memory bound：每次调用
  读 768 MiB 本地权重，t_mem = 164–183 µs（4.95 TB/s），t_mfma = 11–179 µs（1151 TFLOPS，功耗墙下的 FP8 持续吞吐），
  T=4096（每 expert 128）正好在拐点。实测 small-M GateUp+Down（持续态）为 t_mem 的 1.22–1.38×（T≤1024），
  T=2048 为 1.64×，T=4096 为 357.9 µs = t_mem + t_mfma 的 0.99×。memory bound 部分与按任务粒度（每 WG 1 MiB）测的
  同形态读探针只差 ~4%；与 4.95 TB/s 的差距来自任务粒度（探针本身只有 4.3–4.6 TB/s）和 750 W 功耗墙。
- **分支**：`fused_moe_a8w8_blockscale_gfx942_sort` 在期望每 expert route 数 ≤128 时走 128 行 sort + small-M GateUp/Down
  （新 kernel，运行时按每个 block 的实际行数选 1..8 个 16 行 slice），否则走原 BM256 路径。
- **与调优后 asm 比（含 sort，中位数）**：

| T | 256 | 512 | 1024 | 2048 | 4096 |
|---|---|---|---|---|---|
| ep_uniform 持续态 | 1.209× | 1.237× | 1.167× | 1.152× | 1.330× |
| ep_uniform 突发态 | 1.231× | 1.257× | 1.296× | 1.207× | 1.362× |
| random 持续态 | 1.181× | 1.109× | 1.149× | 1.235× | 1.129× |
| random 突发态 | 1.245× | 1.265× | 1.280× | 1.271× | 1.140× |
| BM256（原路径）持续态 ep_uniform | 0.61× | 0.63× | 0.66× | 0.72× | 1.03× |

  突发态下所有每 expert ≤128 的点 ≥1.207×。持续态下 small 路径有两个功耗档（每轮内稳定、轮间跳变，快档 1.18–1.31×、
  慢档 1.01–1.12×），asm 只有一档；中位数落在哪一档决定了 random T=512/1024 是 1.11/1.15× 还是（上一次正式运行的）
  1.23/1.16×。random T=4096 有 40% 的 expert 超过 128 route（最多 149），已不属于 ≤128 的情形，只有 1.13–1.14×。
  结论：突发态全部达到 15%，持续态 10 个点中 7 个达到，random T=512/1024 的中位数受功耗档影响低于 1.15×，
  random T=4096（含 >128 的 expert）未达到。

## 1. 问题与 roofline

### 1.1 背景

当前 gfx942 FlyDSL blockscale MoE（`fused_moe_a8w8_blockscale_gfx942`）按 256 行一个 sort block（BM=256）计算。
MiMo-V2.6-Flash 一个 EP8 rank（H=4096，I=2048，32 个本地 expert / 256 全局，top-8）在 decode 和小 batch prefill 时，
每个 expert 只分到几到一百多个 route（T 个 token 时平均 T·8/256 = T/32）。≤128 时 BM256 tile 至少一半是 padding：
MFMA 白算、A 行白读，而权重仍然每个 expert 读一遍。

对照基线：aiter gfx942 1-stage asm `fmoe_fp8_blockscale_g1u1`（`hsa/gfx942/fmoe/silu/fmoe_bf16_blockscaleFp8_g1u1_silu.csv`
中 11 个非 flat kernel：auto、vs/novs、ps/psx、32x128/32x256/64x256）。"tuned asm" 取每个 T 下含 sort 最快的那个。
要求：small-M 分支比 tuned asm 快 ≥15%。

### 1.2 roofline 模型

每次调用：

- 字节：本地权重 768 MiB（W1 512 + W2 256）+ A1 T·H FP8 + A2 写读 2·routes·I FP8 + out 清零与写回 2·T·H·2 B
  = 774 MiB（T=256）… 864 MiB（T=4096）。权重占 89–99%。
- FLOP：6·routes·I·H（GateUp 4·routes·I·H + Down 2·routes·I·H）= 12.9 GFLOP（T=256）… 206 GFLOP（T=4096）。
- 带宽：1 GiB nontemporal 读探针 **4.95 TB/s**；计算：FP8 MFMA 持续 2 s 探针 **1151 TFLOPS**（峰值 2615 的 44%，
  750 W 功耗墙下有效时钟约 0.93 GHz；`_scaled_mm` 8192³ 为 1.07–1.12 PF，与之一致）。

| T | routes | GFLOP | MiB | t_mem µs | t_mfma µs | 类型 | roofline = max | 串行 = 和 |
|---|---|---|---|---|---|---|---|---|
| 256 | 256 | 12.9 | 774 | 164.0 | 11.2 | memory | 164.0 | 175.2 |
| 512 | 512 | 25.8 | 780 | 165.2 | 22.4 | memory | 165.2 | 187.6 |
| 1024 | 1024 | 51.5 | 792 | 167.8 | 44.8 | memory | 167.8 | 212.6 |
| 2048 | 2048 | 103.1 | 816 | 172.9 | 89.6 | memory | 172.9 | 262.4 |
| 4096 | 4096 | 206.2 | 864 | 183.0 | 179.1 | 拐点 | 183.0 | 362.1 |

ep_uniform 路由（每 expert 恰好 T/32 个 route）；random 路由的 routes 相差 <1%。结论：**整个 ≤128 区间都是 memory
bound，T=4096（每 expert 128）正好在拐点**。目标：memory bound 时接近 t_mem，拐点处至少低于 t_mem + t_mfma。

### 1.3 实测上限比探针低

探针的 4.94 TB/s 是"每 WG 连续读 3.4 MiB"。按 MoE 任务的粒度（每 WG 读一个 expert 的 1 MiB 权重块）同样的读探针只有
4.25–4.57 TB/s（`stream_probe.json`），small-M kernel 的 16 stream × 2 KiB 形态再低 ~5%；kernel 达到同形态探针的 96%
（512 task：kernel 126–134 µs vs 探针 126 µs）。另外持续调用时 GPU 撞 750 W 功耗墙（GateUp T=1024 持续运行实测 728 W、
1237 MHz），时间比短突发多 10–16%，见 §3.7。

## 2. 设计

### 2.1 分支与 sort

- `use_small_m(tokens, topk, global_experts)`：期望每 expert route 数 tokens·topk/global_experts ≤ 128 走 small-M。
  在 host 端按 shape 决定（不读 device 数据，不同步）。
- small-M：aiter `moe_sorting(block_size=128, moe_buf=out)`，sort kernel 顺带把 out 清零；GateUp 写 A2；Down 用 BF16
  原子加把每个 route 加到 out。整条路径 3 个 kernel（sort、GateUp、Down），与 asm 路径（sort 清零 + asm）同口径。
- BM256 路径不变（空 moe_buf + route count + 清零 + single_store）。

### 2.2 kernel 结构（GateUp 与 Down 同一个模板）

- 一个 workgroup = 8 wave（512 线程）= 一个 128 行 sort block × 一个 N tile：GateUp 128 个输出列（每 wave 16 gate 行 +
  对应 16 up 行），Down 256 个输出列（每 wave 2×16 行）。task 在 8 个 XCD 间按"每 XCD 一段连续 task"分配
  （`task = (pid & 7)·chunk + (pid >> 3)`，chunk 由运行时有效 block 数算出），同一 block 的 N tile 共享一个 L2。
- B（aiter shuffle_weight(16,16)）：每 lane 一次 16 B load 就是 MFMA 的操作数布局，直接进 VGPR，经 `ring` 个 K block
  的寄存器环（`ring−1` 块在途，nt），不经 LDS、不需要 barrier。越过最后一个 K block 的 load 指向描述符之外。
- A：GateUp 按 token gather，Down 读 GateUp 写的排序后 FP8 行；提前两个 K block 读进 VGPR，写入 LDS 双缓冲
  （16 KiB/块，XOR swizzle，`ds_read_b128` 无 bank conflict），每 K block 一个 barrier；A scale 与 A 一起进 LDS。
- **运行时行数**：每个 block 用 ballot+ctlz 算出最高有效行，K 循环与 epilogue 按 S∈{1..8} 个 16 行 slice 各编译一份，
  block 取覆盖其 route 的最小 S。MFMA、A 流量、写回都随 route 数走（粒度 16 行），而不是 128 行。
- block scale：每个（slice, side）每个 K block 4 个链式 `mfma_f32_16x16x32_fp8_fp8` 得到 K128 部分和，在下一个 slice
  的 MFMA 阴影下做 `acc += partial · (scale_a[row] · scale_b)`；B scale 一个 VGPR 装下 task 所有 K block，`v_readlane` 取。
- GateUp epilogue：SiLU(gate)·up，行内 128 列的 amax（`ds_bpermute` + LDS 跨 wave），按 (行, 128 列) FP8 量化，
  amax/240 为 scale；只写用到的行（Down 不读 padding 行）。
- Down epilogue：routing weight × acc → BF16（RNE），每 wave 经 LDS 转置成每 lane 4 行 × 64 B 连续，
  `global_atomic_pk_add_bf16`（可选 single_store：单 route token 直接 store）。
- 资源（ring=2）：GateUp 204 VGPR、Down 186 VGPR，无 spill，LDS 33 KiB，每 CU 1 个 workgroup（每 SIMD 2 wave）。

## 3. 优化过程

### 3.1 第一版

正确性测试全过后第一次测量（GPU0 空闲）：T=1024 small 247 µs vs tuned asm 291（1.18×），BM256 461；T=4096 414 vs
485（1.17×）。GateUp 在 T=1024 只有 ~3.7 TB/s，T=4096（S=8）约 2 TB/s、MFMA 效率 ~50%。

### 3.2 寄存器与 vmcnt

- A 的下一 slice 在当前 slice 的 MFMA 之前从 LDS 读（每 slice 提前一拍），A scale 和 A 一起进 LDS。
- 回归：在展开的 K 循环里写 `if tid < 128:`（只让前 128 线程写 A scale）使 GateUp 变成 256 VGPR + 256 个 spill，
  scratch 访问让所有 vmcnt 等待都变成 vmcnt(0)，S=8 从 260 µs 掉到 1009 µs。改成所有线程无条件写同一个值后恢复
  （212/220 VGPR，0 spill）。教训：展开循环里不要放按 lane 的动态分支。
- vmcnt 按发射顺序计数：每轮先发 A、再发 B，这样等 A 时不会连带等最新的 B 预取（测量上无明显收益，保留）。

### 3.3 路径级：去掉 route count / 清零 kernel

第一版沿用 BM256 的做法：空 moe_buf 的 sort + `moe_route_counts`（8.4 µs）+ `moe_zero_rows`（5.8 µs）+ Down
single_store。改为让 sort 直接清零 out（sort128 带清零只比不带多 0.6/1.0/7.3 µs @T=256/1024/4096），Down 全部原子加
（与 single_store 时间相同：69.3 vs 68.9 µs @T=256，134.3 vs 131.3 µs @T=4096）。T≤1024 每次省 ~13 µs 和两次 launch。

### 3.4 带宽上限在哪里

- `sweep_blocks.py`：S=2 时 1 轮（304 task）GateUp 4.2–4.3 TB/s，ring 2/4/8、去掉 MFMA 和 A、去掉 barrier 都不变；
  1.68 轮（512 task）只比 1 轮慢 1.76 倍（尾轮 CU 少时单 CU 带宽上升，单 CU 空载可达 ~50 GB/s）；空 workgroup 1–2 µs。
- 假设"并发 task 在同一 K 相位读权重造成 HBM 通道冲突"，按 task 旋转 K 起点：**反而慢 3–18%**，撤销。
- `stream_probe`：每 WG 1 MiB 的任务即使单 stream 也只有 4.25–4.57 TB/s；16×2 KiB 形态只再损失 ~5%。上限主要来自
  任务粒度（每个 WG 只读 1 MiB、ramp 与尾部），不是访存形态。要再往上需要跨 task 连续预取（persistent）。

### 3.5 S=8（T=4096）

- ATT（单 CU，4 SIMD）：相邻 MFMA 发射间隔 16 tick，即 trace 时钟就是 SCLK（该次 1.30 GHz）。循环内每 K block 中位
  2508 周期，MFMA 2048（82%）；每 task 序言 7–16k 周期（num_valid → sorted_ids/expert → A 行 → 首个 B 块的依赖链）、
  epilogue 8–9k 周期（SiLU 的 exp/rcp 是 1/4 速率）。
- 消融：去掉 B、去掉 A、去掉 barrier、三者都去掉，S=8 时间都不变（~235–240 µs）——MFMA 在就不变。
- slice 扫描：S≥4 后每多一个 slice，GateUp +~20 µs、Down +~10 µs，等于该 slice 的 MFMA 量在功耗墙时钟下的耗时。
- `wg_timeline.py`（插桩 s_memrealtime）：304 个 S=8 task 同时跑时每个 117 µs，8 个 XCD 结束时间差 ≤3 µs；512 task 时
  第 2 轮 208 个 task 每个 ~97 µs（CU 少、功耗余量大），1.68 轮比可整除多付 ~16%。
- 结论：S=8 是功耗墙下的 MFMA 吞吐受限 + 尾轮 + 每 task 固定开销。
- 尝试（负结果）：A 用 3 个 LDS buffer 提前两块写入、下一块 slice 0 在 barrier 前读进寄存器、块边界的 LDS 工作放在最后
  一个 slice 的 MFMA 之下、访存放在每个 slice 的 MFMA 之后——交替 A/B 在 S=1/2/4/8 全部在 ±3% 内，撤销（代码更复杂、
  LDS 50 KiB）。另试过 burst 发射全部 load、ring 深度，均无收益。

### 3.6 slice 粒度：1/2/4/8 → 1..8

只有 2 的幂 slice 时，random 路由下 routes 刚过 2 的幂的 expert 白算最多近一倍：T=2048 random（每 expert 47–77）
65–77 行的 block 只能用 S=8。正式测量 random：T=1024 1.106×、T=2048 **1.037×**、T=4096 1.100×（不达标）。
加上 S=3,5,6,7 后（16 行粒度，`sweep_blocks`：80 行 GateUp 167 µs vs S=8 232 µs），random T=1024/2048 变为
1.159×/1.272×。

### 3.7 功耗墙与 ring 深度

bench 每个变体先跑 ≥50 ms 预热再计时（持续态）。同一 kernel 在 bench 里比短突发慢：GateUp T=1024 突发 129 µs、
bench 148 µs、连续 3000 次 150 µs，此时 728 W / 1237 MHz（空闲 182 W / 2098 MHz）。memory bound 的 kernel 也随时钟变慢，
说明持续态由功耗决定：HBM 流量 + MFMA + 静态功耗把 750 W 用满。因此：

- 用 MFMA 对 padding 少算、少搬数据的设计在持续态收益更大；
- 持续态 A/B：ring=2 比 4 快 0–4.3%（T=256/1024/2048/4096 ep_uniform、4096 random 全部不慢），ring=8 慢 3–5%；
  B cache 策略 nt（aux=2）比默认快 8–9%，aux=3 与 nt 相同。默认改为 ring=2（VGPR 也降到 204/186）。
- bench 另跑 `--warmup-ms 0` 的突发态作为对照。

## 4. 结果

完整表格（含每轮中位数、BM256、单 kernel、roofline 对照）见实验目录 README §4。以下为持续态 ep_uniform（µs，含 sort）：

| T | tuned asm | small | sort128（清零） | GateUp | Down | GateUp+Down / t_mem | GateUp+Down / (t_mem+t_mfma) |
|---|---|---|---|---|---|---|---|
| 256 | 260.4 | 215.5 | 11.0 | 131.6 | 68.2 | 1.22 | 1.14 |
| 512 | 272.4 | 220.2 | 12.7 | 146.8 | 68.2 | 1.30 | 1.15 |
| 1024 | 290.2 | 248.6 | 13.9 | 150.1 | 81.0 | 1.38 | 1.09 |
| 2048 | 332.4 | 288.4 | 21.8 | 185.2 | 98.9 | 1.64 | 1.08 |
| 4096 | 494.3 | 371.6 | 26.3 | 228.3 | 129.6 | 1.96 | 0.99 |

精度：所有路径的输出都用 asm 测试的参考实现检查（`check`）；small 路径 calc_diff 1.4e-6（ep_uniform）/ 2.6e-6（random），
asm 为 8e-6–1.4e-5。`test_moe_small_m_gfx942.py` 46 个用例通过（GateUp、GateUp+Down 两种写回、ring 2/4、aiter sort 三种
清零方式、分支选择）。

## 5. 未达标点、限制与后续

- 持续态功耗档：small 路径把 GPU 推到 750 W（GateUp 持续 728 W），控制器在两个时钟档之间切换，慢档比快档慢 12–16%。
  asm 更慢、功率更低，不受影响。要在慢档也保住 15% 需要降低每次调用的能耗（片上数据搬运：A 在 LDS 里被 8 个 wave
  各读一遍、B 经 L2→L1→VGPR）。
- random T=4096：超过 128 route 的 expert 会分成满块 + 尾块，尾块只有 1–21 行却把该 expert 的权重再读一遍，任务数
  从 512 变成 ~720（GateUp 283.7 vs ep_uniform 228.3 µs）。
- asm 正确性：基线与最终运行中 `novs_32x256`（T=512/1024）和 `vs_ps_32x256`（T=256）偶发 FAIL（与 csv 注释
  "非 ps 的 32x256 在 32K 偶尔出错"一致），tuned asm 仍按时间取最快，不排除这些 kernel。
- 分支判断在 host 端按 shape：a2a 接收缓冲区按容量开 shape（远大于实际 token 数）时会走 BM256 路径。
- 后续方向：(1) persistent 融合 GateUp→Down（按 block 计数依赖），消除 GateUp 尾轮、Down launch 和每 task 序言，
  并在 task 之间连续预取权重以接近 4.9 TB/s；(2) 一个 expert 超过 128 route 时把尾块并入同一 task（共享 B，S 到 10），
  避免权重读两遍；(3) 按 block 大小从大到小调度（LPT），缓解混合大小 block 的尾部。

## 6. 第二版：4 wave + 256 行 sort 表内分支

数据：[docs/experiment/moe_small_m_gfx942/4wave/README.md](../experiment/moe_small_m_gfx942/4wave/README.md)。

### 6.1 要求

1. small-M 改为 4 wave 一个 workgroup；MFMA 与 DQ 仍错开一拍（DQ 在下一 slice 的 MFMA 下做）；每个 K block 分 memory 和
   compute 两个 phase，memory phase 尽量不放 VALU；occupancy=2，让每个 SIMD 同时有两个 wavefront，靠硬件访存限制自然地
   让两个 wave 的 memory / compute 错开。
2. 只用 256 行的 sort 表；GateUp、Down 都按 256 行 block 划分 workgroup，kernel 内判断：block 的第 128 个 token 无效
   （≤128 个 route）走 small-M，否则走 256 路径。

### 6.2 设计

- **任务**：small-M GateUp 一个 WG = 一个 256 行 block 的前 128 行 × 64 个输出列（每 wave 16 gate + 16 up 行），
  Down = 128 个输出列（每 wave 2×16 行）；grid 按 256 行 block 数开，task 映射与第一版相同（每 XCD 一段连续 task）。
  WG 先读 `sorted_ids[block·256 + 128]`：有效就退出（该 block 归 BM256）。
- **BM256 kernel**：`big_blocks_only=True` 时做相反的判断（第 128 行无效就退出）；Down 用原子加（out 由 sort 清零）。
  mixed 路径：sort256（清零）→ small GateUp →（tokens>128 时）BM256 GateUp → small Down →（tokens>128 时）BM256 Down。
  每个 expert 的 route 数不超过 token 数，tokens≤128 时不可能有大 block，不发 BM256 launch。
- **资源**：GateUp 198 VGPR、Down 208 VGPR（≤256），LDS 16.5 / 17 KiB，`amdgpu-waves-per-eu=2,2`：每 CU 2 个 WG。
- **K block**：compute phase = 发下一块 B（直接进 VGPR 的 ring，不等）+ 每个 slice 的 LDS 读 A、8 个链式 MFMA、上一 slice
  的 `acc += partial·(sa·sb)`；memory phase = 发 k+2 的 A 行与 A scale（进 VGPR 暂存）→ `lgkmcnt(0)` → barrier →
  把 k+1 的 A tile 与 scale 写进 LDS → `lgkmcnt(0)` → barrier。memory phase 没有 VALU。
- **A2 scale 改为每 64 列**：4-wave GateUp 的 N tile 只有 64 列，一个 WG 拿不到 128 列的 amax，于是 small block 的 A2 按
  (行, 64 列) 量化，scale 存成 [I/64, capacity]（GateUp 每个 WG 写连续 128 个 float）；Down 把每个 K128 拆成两条 K64 的
  MFMA 链，各用自己的 scale 退休。BM256 block 仍是 per-128（[I/128, capacity]）。测试参考按 route 所在 block 选 64 / 128。

### 6.3 问题与修正

1. **第一版 4-wave 比 8-wave 慢 16–27%（S≥4）**，原因是 VMEM 指令发射（[4wave README §2](../experiment/moe_small_m_gfx942/4wave/README.md)）：
   起初为了把 LDS 留在 32 KiB（A 双缓冲 2×16 KiB），A scale 由每个 wave 自己按 lane 读（每 slice 一条 dword load，
   每条 16 个不同 cache line），每 wave 每 K block 16 条 VMEM，每 CU 128 条。ATT：每 K block 2616 周期中 buffer_load 的
   发射 stall 中位 894、p90 2857，而 vmcnt 等待只有 72。改为每 WG 每 K block 读一次 A scale（每线程 1 条）、经 LDS 给 4 个
   wave，A tile 改单缓冲（两个 barrier）；每 wave 每 K block 降到 9 条 VMEM（S=8），S=8 GateUp 268→222 µs、Down 164→120 µs。
2. **prologue 与 waitcnt 合并**：LLVM 在循环头合并 prologue 和回边两条路径的 VMEM 计数并取更严格者。prologue 若最后发 B
   （调度器也会把 B 排到 A 之后），循环里每次等 B 都会连带等刚发的 A。prologue 改为按循环的顺序发（A/scale(j+2) 与
   B(j+ring−1) 交替，`sched_barrier` 固定）后，循环内的 vmcnt 只覆盖 B。
3. **LDS 竞争**：单缓冲后，S=1 的 ISA 里 compute(k) 的 MFMA 被 SelectionDAG 排到了两个 barrier 之后（`sched_barrier`
   只约束 machine scheduler），等待 `ds_read` 的 `lgkmcnt` 也随之到了 barrier 之后，别的 wave 已在写 A(k+1)：GateUp 20 次
   调用中 8 次与 lane 版不同。barrier 1 前加显式 `s_waitcnt lgkmcnt(0)` 后逐位一致。教训：单缓冲 LDS 的 WAR 不能依赖
   "消费者在 barrier 前"，要显式等 LDS 读返回。
4. **fused 测试的参考**：asm 测试的参考按 per-128 重量化中间结果，per-64 的 small 路径对它 calc_diff 2.3e-4（阈值 1e-4）。
   改为按 route 所在 block 选 64 / 128 的参考后 calc_diff 1.4e-6（ep_uniform）/ 2.6e-6（random），与 8-wave 版相同。

### 6.4 没有收益的尝试（交替 A/B，已撤销）

- ring=4（VGPR 230 / 246 仍是 2 wave/SIMD）：128 行 GateUp −1%、Down −5%。
- S≤6 时 A tile 双缓冲（每 K block 1 个 barrier；S=7/8 仍单缓冲，LDS 32 KiB）：<1%。
- 每 K block 先读 slice 0 再发 B；把 B 全部移到 memory phase：0 到 −4.5%。S=4 的 ATT 里 VMEM stall 占 39%，集中在每个
  K block 的第一条 B load（两 barrier 后 4 个 wave 同时发），但把它挪进 memory phase 会拉长两个 barrier 之间的时间。
- 两个 BM256 kernel 放到侧流（fork/join）：T=512/2048/6144 慢 4–6%，只有 random T=4096 快 4.5%。

### 6.5 结果与分析

- ATT（S=8，1.235 GHz）：每 K block 2014 周期，MFMA 跨度 1144（64 个，几乎背靠背）；两 WG 同驻时 MFMA 管线占用 75%。
  同 CU 的两个 WG 一个 74 µs、一个 100 µs 结束（MFMA 仲裁偏向老 wave），接着第 3、4 个 WG 补上：有 4 个 WG 的 CU 176 µs、
  有 3 个的 ~142 µs 结束（1024 task / 608 槽的尾轮）。S=4 时占用 52%，VMEM 发射 stall 是最大的一项。
- 路径（含 sort，tuned asm 为 6 个 asm kernel 中最快者）：

| T | ep 持续 | ep 突发 | random 持续 | random 突发 |
|---|---|---|---|---|
| 256 | 1.194× | 1.222× | 1.191× | 1.226× |
| 512 | 1.158× | 1.250× | 1.189× | 1.235× |
| 1024 | 1.207× | 1.250× | 1.121× | 1.186× |
| 2048 | 1.093× | 1.154× | 1.154× | 1.131× |
| 4096 | 1.258× | 1.258× | 0.840× | 0.887× |

  同一会话第一版（old8）：ep 持续 1.221/1.264/1.143/1.164/1.322×、突发 1.223/1.268/1.290/1.267/1.382×；random 持续
  1.192/1.102/1.280/1.246/1.131×、突发 1.236/1.274/1.265/1.341/1.133×。
- 空的 BM256 launch：每个 5.8–6.5 µs，在 tokens>128 而没有大 block 时（T=256..2048 的常见情形）每次调用多两个。
- 大 block 与小 block 共存时的代价：random T=4096 的 12 个大 block 让 BM256 GateUp/Down 用了 213.6 / 103.2 µs（处理全部
  block 时 298.2 / 163.8 µs），small 路径 0.84–0.89×；T=8192 random（满 block + 1–37 行尾 block）比原 BM256 路径慢 19%。
  "第 128 行有效就走 BM256" 在 route 数刚过 128 或尾 block 很小时并不划算。
- 结论：4-wave + occupancy 2 的错开机制在硬件上成立（WG 自然错相、MFMA 管线 75%），但每输出列的 A 读、LDS 写、barrier
  加倍，S≥4 时整体不如 8-wave 版；256 行表内分支省掉了 host 端按 shape 的判断和第二张 sort 表，代价是多两次 launch，
  以及混合 block 时 BM256 处理少量 block 的低效。

### 6.6 可选的后续

- 混合 block：把阈值从第 128 行改为更靠后的行（如 192），129..191 行的 block 由 small-M 分两个 128 行半块处理（权重读两遍，
  与第一版的拆分相同），只有足够满的 block 才走 BM256；或反过来，有满 block 时让 BM256 处理该 expert 的尾 block。
- 空 launch：BM256 kernel 改为 persistent（grid = CU 数），无大 block 时每个 WG 只做一次判断。
- 4-wave 的 A 流量：让同一 block 相邻 N tile 的两个 WG 落在同一 CU 才能共享 L1，目前的调度无法保证。

## 7. 第三版：每个 workgroup 128×256 输出 tile（`big_blocks="bm256"`）

数据：[docs/experiment/moe_small_m_gfx942/4wave_128/README.md](../experiment/moe_small_m_gfx942/4wave_128/README.md)。

### 7.1 要求与第二版只算 64 列的原因

- 要求：4 wave 的 workgroup 也算 128 列（occupancy 2 不变）；随后明确为每个 kernel（GateUp 和 Down）的 workgroup 都输出
  128×256（M 128 行 × N 256 个权重行；8-wave 版的 tile 也是这么大：GateUp 128 个输出列、Down 256 个）。
- 第二版每个 wave 持有 2×16 行权重，GateUp 的 2 组是一组 gate 和对应的 up，所以 4 wave × 16 = 64 个输出列（Down 2×16×4 =
  128 列）。128×256 要求每个 wave 4×16 行权重：accumulator 4 组 × 4 × S，S=8 时 128 VGPR，B ring（2 个 K block）64，
  再加两套 partial 32、A 两个暂存槽 32、A 读 16、地址。当时按这个估算选了 64 列，没有编译验证。
- 编译实测（`vgpr.txt`）：直接实现 GateUp S≤5 放得下（248），S=6 起 spill，S=8 溢出 64 个（需要 ~320）。根本原因：每 CU 两个
  workgroup 各有一份 128×256 的 accumulator，占了每个 SIMD 寄存器堆（512）的一半；8-wave 版每 CU 只有一份。

### 7.2 设计

- **tile**：GateUp 一个 WG = 一个 256 行 block 的前 128 行 × 128 个输出列，wave w 持有 gate 行 n·128 + 32w + {0, 16} 起的
  两组和对应的 up 行；Down = 256 个输出列，wave w 持有 n·256 + 64w 起的 4 组。Down 的 B scale 组按 wave 取
  （WG 的 256 行跨两个 128 行 scale 组）。Down 的 epilogue 按组对做两次 LDS 转置（与原来的一次相同）。
- **A2 按 (行, 128 列)**：[I/128, capacity]，与 BM256 GateUp 相同布局，mixed 路径两边共用一个 scale buffer；Down 每个
  K block 一条 K128 的 MFMA 链、一个 scale（第二版拆成两条 K64）。fused 测试直接用 asm 测试的参考。
- **两种寄存器布局**（`compile_moe_small_m.schedule`，按 slice 数在编译期选，GateUp 与 Down 都是 S≥7 用 light）：
  - full：K step 外层、组内层，每 slice 16 个 MFMA；两套 partial（按 slice 奇偶），上一 slice 的 16 个 partial 元素在本
    slice 的 16 个 MFMA 下逐个退休；下一 slice 的 A 在本 slice 开头读；A 提前两个 K block 进两个暂存槽。
  - light：K 半块 → 组对 → K 四分之一 → 组的顺序，每条 4 个 MFMA 的链最后一步后 2+e 个 MFMA 退休元素 e，正好在该组
    下一条链开始前退完，于是只要一套 partial（省 16 + 4）；下一 slice 的 A 按 K 半块读，本 slice 半块 0 用完就读下一
    slice 的半块 0（省 8）；A 只提前一个 K block、一个暂存槽，在第二个 barrier 之后发（省 16）。
  - 两种布局每个 accumulator 的累加顺序相同，结果与直接实现逐位一致（GateUp 输出与 scale 全同；Down 只差原子加顺序）。
- **通用的省寄存器**：B 每组的行偏移（wave 内一致）放进 soffset，voffset 只剩 lane·16（j·1024 进立即数）；各变体与 epilogue
  的线程下标从带不同注释的 `v_mov_b32` 不透明 tid 拷贝算（§7.4）；K 循环的 trip count 也经一条 `s_mov_b32` 变成不透明值。
- **资源**：GateUp 256 VGPR、Down 254 VGPR，都不 spill；LDS 18 KiB（row max 放在 A scale 区，2 KiB）/ 16.5 KiB。

### 7.3 寄存器（GateUp，单个变体，不限 occupancy 编译）

| slices | full | light | 直接实现 |
|---|---|---|---|
| 1 | 142 | 126 | 158 |
| 4 | 207 | 178 | 223 |
| 5 | 226 | 200 | 248 |
| 6 | 242 | 216 | 256 + 2 spill |
| 7 | 256 + 19 spill | 238 | 256 + 38 spill |
| 8 | 256 + 46 spill | 254 | 256 + 64 spill |

Down（4 组）与 GateUp 基本相同：full S=6 242、light S=7/8 236 / 252。

### 7.4 问题与修正

1. **多变体 kernel 的 spill**：S=8 单独编译只要 254，1..8 一起编译却 spill 18–20。S=7 和 S=8 的 A 拷贝数都是 4，两条分支开头
   （token 读、A(0) 的 load）完全相同，SimplifyCFG 把两个分支的相同前缀提到 dispatch 之前；AMDGPU 结构化后的 CFG 让 S=8
   分支从 S=7 分支的末尾经 flow block 进入，提上来的 A(0) 数据于是在 S=7 的整个 K 循环里都活着。每个变体开头加一个带不同
   注释的 `v_mov_b32` 不透明 tid 拷贝（同样的 inline asm 也会被提出去，所以注释必须不同），下标都从它算，spill 消失。
2. **epilogue 的下标被 GVN 合并**：GateUp epilogue 的输出行（tid/8 + 32c）与 prologue 的 A 拷贝行是同一个表达式，被合并后
   跨 K 循环存活，S=8 溢出 7 个。epilogue 改用自己的不透明 tid 拷贝。
3. **light 布局不是免费的**：所有 slice 都用 light 时 S≤5 比 full 慢 3–5%（一套 partial 的退休窗口窄、A 按半块读只提前
   8 个 MFMA、单暂存槽的 A 只有一个 compute phase 的时间）。于是只在放不下的 S=7/8 用 light。
4. **K=256 的 Down 访存越界**：测试形状 inter=256（Down 只有 2 个 K block，K 循环 1 次）时 LLVM 把循环展平，直线代码
   spill 118 个 VGPR（468 B scratch），而这些 launch 不分配 scratch，于是 `HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION`。
   K 循环的 trip count 改为不透明值后保持循环形态，254 VGPR、无 scratch。K≥512 本来不受影响。
5. **干扰检测失效**：外部作业运行时 KFD 不报告它的 CU occupancy（`ExternalLoadMonitor` 读到 0，GFX activity 30–100%），
   第一次路径级运行在负载下跑完，只被 bench 自己的 amdsmi 门限发现。改为：kernel A/B 每轮前要求 3 s 空闲（activity、VRAM），
   结束后再查一次；路径 bench 每次运行前等 30 s 空闲，门限不过整次重跑。

### 7.5 结果

kernel 级交替 A/B（µs，6 轮中位数；c64 = 第二版，down128 = 第三版改回 128 列 Down）：

| rows | GateUp c64 | GateUp cur | GateUp old8 | Down c64 | Down down128 | Down cur | Down old8 |
|---|---|---|---|---|---|---|---|
| 16 | 122.5 | 116.9 | 125.5 | 65.6 | 66.0 | 63.6 | 66.8 |
| 32 | 128.3 | 120.8 | 125.2 | 69.4 | 69.0 | 65.6 | 70.1 |
| 64 | 146.1 | 131.5 | 139.2 | 78.4 | 77.3 | 74.3 | 77.5 |
| 96 | 176.2 | 158.0 | 167.1 | 97.3 | 93.5 | 87.0 | 95.1 |
| 112 | 194.7 | 182.8 | 186.9 | 109.4 | 105.8 | 99.6 | 107.5 |
| 128 | 209.7 | 198.3 | 205.8 | 118.9 | 115.1 | 108.7 | 119.0 |

路径（含 sort，相对 tuned asm 的加速比；同一会话的 old8、c64）：

| T | ep 持续 small / old8 / c64 | ep 突发 small / old8 / c64 | random 持续 small / old8 / c64 | random 突发 small / old8 / c64 |
|---|---|---|---|---|
| 256 | 1.254 / 1.210 / 1.181 | 1.286 / 1.230 / 1.221 | 1.250 / 1.202 / 1.182 | 1.294 / 1.238 / 1.219 |
| 512 | 1.247 / 1.243 / 1.230 | 1.308 / 1.273 / 1.253 | 1.250 / 1.216 / 1.200 | 1.297 / 1.280 / 1.235 |
| 1024 | 1.157 / 1.306 / 1.123 | 1.309 / 1.312 / 1.259 | 1.159 / 1.140 / 1.144 | 1.263 / 1.237 / 1.176 |
| 2048 | 1.179 / 1.190 / 1.087 | 1.289 / 1.277 / 1.153 | 1.307 / 1.302 / 1.211 | 1.278 / 1.284 / 1.122 |
| 4096 | 1.366 / 1.342 / 1.259 | 1.337 / 1.305 / 1.226 | 0.879 / 1.080 / 0.839 | 0.907 / 1.115 / 0.876 |

- 持续态在两个功耗档间切换（ep T=1024 small 每轮 252/226/253/254，old8 227/219/229/223），单点可差 ~10%；突发态更稳。
- 第三版对第二版：每个 T 都快（突发 3–14%）。对 8-wave 版：突发态 ep 0.997–1.046×、random（T≤2048）0.995–1.046×。
  kernel 的 3–10% 领先在路径上只剩一部分，差额主要是 T>128 时两个空的 BM256 launch（各 ~6 µs）。
- random T=4096（12 个 >128 route 的 expert 走 BM256）与 T=8192 random（843 vs BM256 路径 711.5 µs）与第二版同样：这是
  256 行表内分支的问题（§6.5），与 tile 无关。
- 中间版本（Down 仍是 128 列）的同一组数据在 `4wave_128/down_n128/`：Down 改为 256 列后突发态 ep T=2048 从 1.180 升到 1.289，
  其余在 ±3% 内。

### 7.6 可选的后续

- 空 BM256 launch：persistent 的 BM256 kernel，或由 small-M kernel 统计大 block 数、BM256 只在有大 block 时做事（launch 本身
  仍在）。这是现在 small 路径对 8-wave 版的主要差距。
- 混合 block：见 §6.6。
- GateUp light 布局已用满 256 VGPR；再加任何每 slice 的状态都会 spill，改动 S=7/8 的路径前先看 `vgpr.txt` 的余量。

### 7.7 为什么 128×256 的 4-wave 版最快

三版读的 HBM 一样（PMC，GateUp 32 行：HBM 读请求都是 4.25 M），差别在 K 循环的指令开销和延迟能否被另一个 wave 盖住。
每个 MFMA 摊到的 K 循环指令（[`isa_mix.txt`](../experiment/moe_small_m_gfx942/4wave_128/isa_mix.txt)，每 wave 每 K block，
单独编译一个 slice 数；c64 / old8 / cur）：

| | VMEM | LDS 读 | LDS 写 | barrier | VALU（不含 MFMA） | 全部 |
|---|---|---|---|---|---|---|
| GateUp S=1 | 0.75 / 0.75 / 0.63 | 0.38 / 0.38 / 0.19 | 0.25 / 0.25 / 0.13 | 0.25 / 0.13 / 0.13 | 2.50 / 2.25 / 1.31 | 6.94 / 6.44 / 5.03 |
| GateUp S=8 | 0.14 / 0.11 / 0.10 | 0.36 / 0.36 / 0.19 | 0.08 / 0.05 / 0.04 | 0.031 / 0.016 / 0.016 | 1.28 / 1.28 / 1.14 | 3.44 / 3.27 / 2.95 |
| Down S=8 | 0.14 / 0.11 / 0.10 | 0.38 / 0.36 / 0.19 | 0.08 / 0.05 / 0.04 | 0.031 / 0.016 / 0.016 | 2.28 / 1.15 / 1.08 | 4.41 / 3.12 / 2.87 |

1. **每个 wave 4 组权重行**：一次 LDS 读到的 A 片段喂 4 条 MFMA 链（前两版都是 2 条），LDS 读 / MFMA 减半（PMC：0.23 vs
   0.47）。前两版每个 wave 都只有 2×16 行。
2. **对 64 列版**：每个 K block 的固定开销（A 行的全局读、A scale、LDS 写、两个 barrier、waitcnt）和每个 task 的 prologue /
   epilogue 摊到 2 倍的 MFMA 上：VMEM / MFMA −17%（PMC 0.33 vs 0.41），L2 请求 / MFMA 2.99 → 2.52，wave 数减半；Down 去掉
   K64 拆分后 VALU / MFMA 从 2.3 降到 1.1。每 MFMA 的 K 循环指令少 14–39%。
3. **对 8-wave 版**：tile 大小和 A、B 的全局流量相同（L2 请求 / MFMA 2.51 vs 2.52），差别是 (a) 第 1 点的 LDS 读减半，每 MFMA
   的 K 循环指令少 8–22%；(b) 每 CU 两个独立的 4-wave WG 取代一个 8-wave WG：每个 SIMD 上的两个 wave 属于不同的 WG，一个在
   barrier、memory phase、prologue 或 epilogue 时另一个的 MFMA 在跑（第二版的 ATT：同驻的两个 WG 自然错开，两个 wave 同在
   K 循环时 MFMA 管线 75%）。8-wave 版两个 wave 停在同一个 barrier，同一个 CU 上的 task 一个接一个跑（512 task / 304 CU），
   每个 task 的启动（第一批 HBM 读）和收尾都在关键路径上。
4. **代价与前提**：每个 SIMD 要放两份 128×256 的 accumulator（S=8 时每 wave 128 VGPR），只有 light 布局才在 occupancy 2
   下放得下（§7.2–7.4）。第一版有 128×256 的 tile，但每个 wave 只有 2 组、没有 3(b)；第二版有 3(b)，但 tile 只有 128×128；
   第三版三者都有。

动态计数（等待 / 发射周期、MFMA 忙）需要空闲的 GPU，外部作业持续占用期间没有拿到，`tools/pmc_compare.py` 可补测。

## 8. 运行时按 block 选路：方案 B（split）与方案 A（merged）

数据：[split/README.md](../experiment/moe_small_m_gfx942/split/README.md)（B 的第一版与诊断）、
[merged/README.md](../experiment/moe_small_m_gfx942/merged/README.md)（A 的代码、工具，以及 A、B 同一会话的测量）。

### 8.1 要求

- 第三版的 mixed 路径是两次 launch：small-M kernel 和 BM256 kernel 各读每个 256 行 block 的第 128 行，跳过对方的 block。
  它的两个问题（§7.5）：
  - 大小 block 混合时慢：random T=4096 只有 0.88–0.91×，random T=8192 比 BM256 路径慢 18%。
  - T>128 时多两个空 launch。
- 要求：**运行时**在 kernel 内按每个 block 的实际 route 数选择，不在编译期，也不在 host 端按 T 选。先试 B，再试 A。
  - B：所有 block 都走 4-wave small-M kernel（occupancy 2）。有第 128 行的 block（"大 block"）拆成两个 128 行任务，
    每个 stage 一次 launch。
  - A：每个 stage 一个 8-wave kernel。大 block 走 BM256 的 256×256 tile；其余 block 走 pair 任务：两个 4-wave 半组算
    相邻两个 N tile，共享 A。

### 8.2 方案 B（`src`：`compile_moe_small_m(all_blocks=True)`，`fused_moe_a8w8_blockscale_gfx942_mixed(big_blocks="split")`）

- **任务**：(block, N tile, 行半)。行半 1 只在 block 有第 128 行时存在，row_base = 256·block + 128·行半；不再 launch
  BM256 kernel。
- **v1**：grid = 容量 × n_tiles × 2，没有行半 1 的任务立即退出。
  - 问题：这些空 WG 夹在真实 WG 之间。每 CU 只有 2 个槽，空 WG 等探测 load 时占着槽，后面的真实任务要等。
  - 结果：全是小 block 时 split kernel 比做同样工作的 small-M kernel 慢 21–24%（`split/ab_split_empty.log`）；
    容量之外的尾部空 WG 不影响。
- **v2（当前）**：稠密任务表，block b 有 n_tiles·(1+big_b) 个任务。
  - 每个 WG 扫描 valid block 的第 128 行：每线程 5 个 block，按位平面做 ballot + `mbcnt` 前缀，两个 barrier。
  - 每个 XCD 取连续的一段任务；超出全部任务槽的 WG 不扫描。
  - SCAN_BLOCKS=5，即最多 1280 个 block，T=32768 够用。
  - 同一 A/B 下 split/small 为 1.015–1.053×（v1 为 1.27–1.31×）。资源不变：GateUp 256、Down 254 VGPR。
- **测试**：`test_moe_small_m_gfx942.py` 36 passed、1 skipped。新增用例：SPLIT_SHAPES、SPLIT_DOWN_SHAPES、fused 的
  `big_blocks=bm256/split`、dispatch T=8192 split。

### 8.3 方案 A（`src/contrib/flydsl/moe_merged_a8w8_blockscale_gfx942.py`，§8.7；run4 测的是实验目录 `merged/variants/moe_small_m_8wave_pair.py`）

- **第 1 步（pair kernel）**：8 个 wave，一个 WG 取一个 block 和两个相邻 N tile。半组 h 的计算与 4-wave kernel 的一个 WG
  相同；A 及其 scale 只读一次、写 LDS 一次，8 个 wave 共用 barrier。
  - 与当前 4-wave 逐位相同（`check_pair.py`）。
  - 时间（`ab_pair_b32.log`）：rows ≤64 在 ±3% 内；rows 96/128 时 GateUp 慢 4%、Down 慢 9%。原因是两个半组同步执行，
    失去了 4-wave 双 WG 的错相（§7.7 第 3(b) 点）。
- **第 2 步（merged）**：
  - BM256 GateUp / Down（`src`）加 `merged=` hook：主体包成 `big_body()`，LDS 分配移到 kernel 作用域。14 个默认配置的
    ISA 与改动前相同。
  - plugin 提供四样东西：
    - `task`：稠密任务表，大 block 的任务在前。每个 XCD 先一段 BM256 任务，再一段 pair 任务。每线程 3 个 block，
      最多 1536 个 block。3 个 barrier，最后一个保证扫描字读完后任务才覆盖这块 LDS。超出任务槽的 WG 不扫描；
      num_valid 之后的 block 不发探测 load。
    - `body`：pair 任务。它在 BM256 的 64 KiB LDS 内取视图，并用带注释的不透明 tid，避免两个分支的下标被 GVN 合并。
    - `grid`：8·(2·num_blocks+2)，按 sort 容量算。
    - `key`：见 §8.4 第 3 点。
  - 资源：GateUp / Down 都是 250 VGPR、64 KiB LDS，不 spill。
  - 正确性（`check_merged.py`）：GateUp 与两次 launch 的结果逐位相同；fused 对 asm 参考 T=64..32768 均通过，
    calc_diff 1.4–2.6e-6。

### 8.4 问题与修正

1. **B v1 的交错空 WG**：见 8.2，改为稠密任务表。
2. **B 的两个行半不等长**（例如 T=6144，每个 expert 192 个 route，即 128 + 64 行）：两个半块各读一遍权重，速度不同、
   不再同时运行，L2 共享不到 B。split GateUp 393–398 µs，BM256 324 µs。这是方案 B 本身的代价。
3. **FlyDSL 内核缓存键不含 plugin（影响 A 的全部早期数据）**：
   - 起因：import aiter 会把 FlyDSL 磁盘缓存设到 `aiter/jit/flydsl_cache` 并开启。缓存键只含两样东西：root 文件同目录的
     函数源码，以及闭包里的标量（int/float/bool/str/None/tuple/enum）。plugin 对象和它在实验目录里的代码都不在键里。
   - 后果：同一形状的所有 merged 编译都复用了第一次的结果，也就是 08:10 编译的 run1 版本（每个 WG 扫描全部容量、
     没有提前退出）。
   - 被影响的数据：
     - 之后改 plugin（提前退出、valid 之外不探测）的两次正式运行（`run2_cached/`、`run3_cached/`）测到的仍是 run1 的
       kernel，三次运行 merged 相差 ≤1–2%。
     - `ab_merged.py` 的 direct / nobody / exact 变体也都是同一个 kernel。
   - 怎么发现的：rocprofv3 显示 "exact" 的 grid 是 8·(2·nb+2)，而不是设定的 2048。
   - 修正：BM256 的 compile 函数把 `merged.key`（plugin 源码的哈希）放进 launch 的闭包。
   - 未受影响：small-M、split 和 pair kernel 的源码与各自的 root 文件同目录，选项都是标量。
4. **A 的大 T 变慢（run1 版本）**：128 个满 block、1056 个 block 的容量，Down 比 BM256 慢 12–17%。
   - grid 按容量算：T=32768 时 16912 个 WG，真实任务只有 2048 个。run1 版本让每个尾部 WG 也扫描全部容量。
   - 慢的程度随容量变化：容量 129 → 1056 个 block，kernel 从 585 µs 变到 735 µs（rocprofv3）。
   - 当前版本（提前退出 + 只探测 valid block）在同一 A/B 下重测两次，merged / BM256 为：GateUp 1.005× / 0.970×，
     Down 0.978× / 1.037×，即 ±4% 以内、方向不一。监视器把大多数轮标成有外来 kernel，以 8.5 节的路径级数据为准。
5. **FlyDSL AST 改写**：动态 `if` 体内（包括嵌套函数里）赋值的名字，如果在 kernel 作用域也出现过（即使在未选中的
   const_expr 分支里），就会报 "initialized as None"。
   - 修正：辅助逻辑放进函数，冲突的名字改名（`chunk` → `xcd_chunk`）。
   - 在 kernel 外定义、含动态 `if` 的函数要先过 `ASTRewriter.transform`。

### 8.5 结果（run4）

- 运行：`merged/tools/run_ab.sh`，GPU 1，所有运行的 amdsmi 门限第一次就通过。merged 与 split 都是新 key 下重新编译的。
- 路径都含 sort。各路径：
  - small：第三版的两次 launch 路径。
  - split：方案 B。
  - merged：方案 A。
  - bm256：原 BM256 路径。它的 sort 不清零 `out`，只清零多 route 的行，Down 用 `single_store`。
- 数值是相对 tuned asm 的加速比：T≤8192 取突发态，大 T 取持续态，每行最好的加粗。

| T | 路由 | small | split（B） | merged（A） | bm256 |
|---|---|---|---|---|---|
| 256 | ep_uniform | 1.292 | **1.310** | 1.280 | 0.538 |
| 512 | ep_uniform | 1.310 | **1.339** | 1.316 | 0.574 |
| 1024 | ep_uniform | 1.315 | **1.332** | 1.330 | 0.586 |
| 2048 | ep_uniform | 1.269 | 1.288 | **1.306** | 0.651 |
| 4096 | ep_uniform | 1.354 | **1.401** | 1.319 | 0.973 |
| 6144 | ep_uniform | 1.222 | 1.105 | 1.259 | **1.280** |
| 8192 | ep_uniform | 1.424 | 1.402 | 1.460 | **1.534** |
| 12288 | ep_uniform | 1.310 | 1.358 | 1.362 | **1.370** |
| 16384 | ep_uniform | 1.419 | 1.353 | 1.448 | **1.580** |
| 32768 | ep_uniform | 1.499 | 1.227 | 1.522 | **1.614** |
| 256 | random | 1.301 | **1.318** | 1.302 | 0.558 |
| 512 | random | 1.280 | **1.314** | 1.307 | 0.584 |
| 1024 | random | 1.263 | 1.314 | **1.322** | 0.589 |
| 2048 | random | 1.423 | **1.456** | 1.449 | 0.763 |
| 4096 | random | 0.923 | **1.129** | 1.061 | 1.031 |
| 6144 | random | **1.243** | 1.105 | 1.233 | 1.236 |
| 8192 | random | 1.054 | 1.195 | **1.313** | 1.157 |
| 12288 | random | 1.207 | 1.224 | **1.394** | 1.337 |
| 16384 | random | 1.367 | 1.305 | 1.506 | **1.524** |
| 32768 | random | 1.324 | 1.148 | **1.449** | 1.428 |

kernel 中位数（µs，持续态）。"两次 launch" 是 small-M + BM256 两个 kernel 之和；T≥6144 时只有 BM256 kernel 在做事：

| T | 路由 | GateUp 两次 launch / split / merged | Down 两次 launch / split / merged |
|---|---|---|---|
| 1024 | ep_uniform | 148.9 / 144.7 / 143.6 | 84.6 / 79.4 / 76.9 |
| 4096 | ep_uniform | 225.9 / 219.8 / 232.0 | 130.8 / 122.2 / 123.2 |
| 6144 | ep_uniform | 325.1 / 392.5 / 317.1 | 187.1 / 211.7 / 184.2 |
| 8192 | ep_uniform | 356.3 / 371.5 / 348.3 | 211.3 / 204.2 / 207.5 |
| 32768 | ep_uniform | 1156.3 / 1480.5 / 1153.5 | 719.3 / 803.1 / 703.2 |
| 4096 | random | 371.9 / 282.7 / 312.1 | 192.2 / 152.5 / 157.3 |
| 8192 | random | 497.0 / 439.6 / 374.1 | 294.0 / 240.8 / 219.2 |
| 32768 | random | 1279.8 / 1558.8 / 1191.9 | 766.3 / 778.7 / 703.5 |

**B（split）**：

- T≤2048 与 T=4096 ep_uniform 时最好或与最好的差 ≤1%。
- random T=4096 最好（1.129×）。这一点的 12 个刚过 128 route 的 block，在 B 里是一个满 S=8 的半块加一个 1–21 行的小半块，
  任务粒度细。
- T≥6144 最差：比 merged 慢 10–12%（6144）、9%（random 8192）、21%（random 32768）；ep T=32768 比 bm256 慢 24%。原因有两个：
  - 不满的大 block 两个半块不等长，L2 共享不到权重（§8.4 第 2 点）。
  - 大 T 计算受限时，4-wave tile 不如 BM256 tile（§6.5：两 WG 同驻时 MFMA 管线 75%）。

**A（merged）**：

- 最差比最好的路径慢 8%（ep T=16384，对 bm256 路径，是路径级差异，见本节最后一点），其余点都在 6% 以内。
- 在 ep T=2048、random T=1024、random T=8192、random T=12288、random T=32768 最好。random T=8192 为 1.313×，比 split 快 10%、
  比 bm256 快 13%：满 block 和尾 block 在同一个 launch 里，尾 block 不再单独占一次 BM256 launch。
- T≤4096 比 split 慢 0–6%，有两个原因：
  - pair 任务在 S=8 时比两个 4-wave WG 慢（第 1 步的 A/B）。
  - random T=4096 的刚过 128 route 的 block 按 256 行整块计算。
- 满 block 上 merged kernel 与 BM256 kernel 一样快，例如 ep T=32768：GateUp 1153.5 vs 1156.3 µs，Down 703.2 vs 719.3 µs。
- 大 T 时 ep_uniform 比 bm256 路径慢 5–8%（T=8192..32768），差别在路径而不在 kernel：bm256 路径的 sort 不清零 `out`，
  Down 对单 route 的 token 直接 store（`single_store`）；merged 路径清零整个 `out`，所有 route 都用原子加。

### 8.6 结论与后续

- 两个方案都满足"运行时按 block 选择、每个 stage 一次 launch"，都比第三版的两次 launch 路径（small）好。
  - small 在 random T=4096 / 8192 只有 0.923× / 1.054×，A、B 都消除了这个问题。
  - 空的 BM256 launch 也没有了。
- 单一的运行时路径建议用 **A（merged）**：
  - A 在 20 个点里最差比最好的路径慢 8%，B 最差慢 24%。
  - A 在大 T 和混合 block 时最好或接近最好。
  - 代价是 T≤4096 比 B 慢 0–6%。
- 若 host 端可以按 T 选择（不符合本次"运行时"的要求）：T≤4096 用 B，其余用 A。
- 后续：
  1. merged Down 支持 `single_store`：BM256 分支已有，需给 pair 分支加上，sort 也改为只清零多 route 的行。
     这应能补上大 T ep_uniform 与 bm256 路径 5–8% 的差。
  2. 刚过 128 route 的 block 不再按 256 行整块算：例如改为第 192 行有效才算大 block，129..191 route 的 block 拆成两个
     pair 任务（行半）。这是 B 的做法，只用在 A 的这一类 block 上。
  3. pair 任务在 S=8 时的 4–9% 差距：两个半组错开半个 K block，或给每个半组单独的 barrier。
- 方案 B 已进 `src`（`big_blocks="split"`）。方案 A 已进 `src` 并成为默认，见 §8.7。

### 8.7 A 进入 `src`

- 实验目录的 `merged/variants/moe_small_m_8wave_pair.py`（run4 测的版本，commit 806d8fe，备份分支
  `luwei/942-fp8-blockwise-backup-b`）搬到 `src/contrib/flydsl/moe_merged_a8w8_blockscale_gfx942.py`，内容如下：
  - pair 任务：`compile_pair_tasks` 只返回 emitter 和它的 LDS 区域，去掉了实验用的独立 pair kernel。
  - 插件：`merged_plugin`、`MergedPlugin`、`source_key`。
  - 入口：`moe_gateup_merged`、`moe_down_merged`。
- 相对 run4 版本的改动，MiMo 形状下的任务分配不变：
  - grid 和扫描上界按每个 block 的 tile 数（记为 t，MiMo 为 16）计算，不再写死 16。grid =
    `8·(⌈t·num_blocks/8⌉ + 1)`，扫描上界为每个 XCD `⌈t·valid/8⌉ + 1` 个 slot。旧上界是
    `⌈t·valid/8⌉ + ⌈(t/2)·valid/8⌉`（约 3·valid），全为大 block 时超出 2·valid 的 slot（受 grid 限制）只扫描不干活；
    现在每个 XCD 最多 1 个。
  - 断言：BM256 与 pair 的 tile 数一致；pair 任务装得进 BM256 kernel 的 64 KiB LDS。pair 的参数检查本来就要求
    tile 数为偶数。
  - `SCAN_BLOCKS` 由 3 改为 4：最多覆盖 2048 个 sort block，topk=8、32 个本地 expert 时 T ≤ 64512。
- `fused_moe_a8w8_blockscale_gfx942_mixed` / `_sort` 加 `big_blocks="merged"` 并设为默认。sort 容量超过
  `MAX_BLOCKS` 时退回 `"bm256"`。aiter 入口 `fused_moe_a8w8_blockscale_gfx942`（BM256 + `single_store`）不变。
- 测试（`test_moe_small_m_gfx942.py`，50 passed、1 skipped）：
  - `test_gateup_down_merged`：6 个形状，含 sort 容量之外的垃圾 block。merged GateUp 与 small-M + BM256 两次 launch
    逐位相同，Down 对全部 route 的参考通过。
  - fused / dispatch 测试加 `merged`。
  - `test_merged_capacity_fallback`：把 `MAX_BLOCKS` 设成 1，检查退回路径。
  - BM256 的 GateUp / Down 测试（`test_gateup_a8w8_blockscale_gfx942.py`、`test_down_a8w8_blockscale_gfx942.py`）：118 passed。
- `bench_moe_small_m_gfx942.py` 加 `--paths merged` 和 `k: gateup_merged` / `k: down_merged`。实验目录的 tools 仍对应
  run4 的版本。
- `src` 版本的性能在 §9 的会话（2026-11-11）中复测：与 run4 相比，tuned asm 的倍数在 −5%…+7% 之内（例如 ep T=256
  突发态 1.288× vs 1.280×，random T=8192 突发态 1.327× vs 1.313×，ep T=4096 突发态 1.412× vs 1.319×）。

## 9. BM256 kernel 内的 128 行分支（`bm128`，aiter 路径）

数据与复现：[docs/experiment/moe_small_m_gfx942/bm128/README.md](../experiment/moe_small_m_gfx942/bm128/README.md)。

### 9.1 动机

- §6–§8 的 small-M、split、merged 都有自己的 workgroup 形状或 grid：4-wave、每 CU 两个 workgroup，或者 kernel 内
  的任务扫描。接入 aiter 要改 aiter 那一侧的路径。
- aiter 的 GateUp 已有同类做法：luocheng25/aiter 3bc0b50（"gemm1: per-block BM256/BM128 dual pipeline in the GateUp
  kernel"，gfx950）。
- 本节把这个做法用到 gfx942 的 GateUp 和 Down：grid、workgroup（8 wave、64 KiB LDS）、任务顺序都与 aiter 路径的
  BM256 相同，只在 kernel 内多一个 128 行分支。

### 9.2 实现（`bm128=True`：GateUp 2x4_pf1，Down 4x2_pf1 atomic）

- 每个 workgroup 读自己 block 第 128 行的 sorted ID（一次 uniform load）。该行无效时 block 至多 128 个 route，跑
  128 行流水线；否则跑原 BM256 流水线。代码上是 `big_body()` 和 `big_body(128)` 两份。
- GateUp（A 流式、B 驻留）：
  - 每个 K block 2 个 stage（原来 4 个），只读 A packet 0、1；每个 wave 只算 M slice 0..3。
  - 下一个 K block 的 4 个 B packet 在 stage 0 发出，stage 1 写入 LDS。
  - FP8 epilogue 只写 0..127 行；128..255 行不写（Down 从不读 padding 行）。
- Down（A 驻留、B 流式）：只读 A packet 0、1；每个 stage 4 个 MFMA pair 中只算 2 个（M slice 0、1）；每个 K block
  仍是 4 个 stage。
- 两条流水线末尾各放一条不同的空 inline asm。没有它时，LLVM SimplifyCFG 把两份 Down epilogue 的公共尾部下沉到同一个
  block：atomic 的 SGPR 基址（`"s"` 操作数）变成存在 VGPR 里的 phi，`v_readfirstlane` 读到被覆盖的值，GPU 访存错误
  （地址 0x14000）。
- `fused_moe_a8w8_blockscale_gfx942` 默认 `bm128=True`，aiter 侧的调用不用改。
- 机器码（MiMo 形状）：
  - `bm128=False` 时 `22_final_isa.s` 与上一个 commit 逐字节相同：GateUp FP8、Down atomic sorted（有无
    `single_store`）、route counts、zero rows。
  - `bm128=True` 时 BM256 的 K 循环与原 kernel opcode 相同（GateUp 743 条、Down 708 条 / 2 个 K block）；128 行 K 循环
    分别为 399 条和 406 条。
  - VGPR 250 / 246 不变，无 spill；SGPR 54 → 72、61 → 69。
- 测试（共 199 passed、1 skipped）：
  - GateUp BF16 输出加 bm128 配置；Down 加两个 bm128 配置。
  - `test_gateup_down_bm128`：六种 block 构成（全部 ≤128、恰好 128、恰好 129、混合、MiMo EP8 每 expert ~64 / ~128）。
    bm128 GateUp 计算的行与 BM256 逐位相同，其余行不写；Down 开或关 bm128 都通过参考。
  - `test_fused_entry`：aiter sort 加 `fused_moe_a8w8_blockscale_gfx942`，T = 64 / 1000 / 4096 / 8192，bm128 开、关。

### 9.3 结果（相对 tuned asm，含 sort；2026-11-11 GPU1，gates 全部通过）

| | T | 256 | 512 | 1024 | 2048 | 4096 | 6144 | 8192 |
|---|---|---|---|---|---|---|---|---|
| ep_uniform 持续态 | bm256 | 0.619× | 0.631× | 0.685× | 0.710× | 1.041× | 1.355× | 1.641× |
| | bm128 | 0.841× | 0.859× | 0.878× | 0.911× | 1.277× | 1.353× | 1.655× |
| | merged | 1.250× | 1.139× | 1.251× | 1.213× | 1.360× | 1.290× | 1.545× |
| random 持续态 | bm256 | 0.619× | 0.648× | 0.647× | 0.792× | 1.081× | 1.321× | 1.231× |
| | bm128 | 0.836× | 0.858× | 0.877× | 1.007× | 1.044× | 1.331× | 1.234× |
| | merged | 1.275× | 1.278× | 1.180× | 1.340× | 1.030× | 1.310× | 1.393× |
| ep_uniform 突发态 | bm128 | 0.842× | 0.857× | 0.888× | 0.924× | 1.263× | 1.258× | 1.563× |
| random 突发态 | bm128 | 0.826× | 0.859× | 0.839× | 1.030× | 1.043× | 1.241× | 1.196× |

大 T（持续态）：ep_uniform T=12288 / 16384 / 32768 为 1.611× / 1.569× / 1.618×（bm256 1.388× / 1.578× / 1.623×）；
random 为 1.433× / 1.560× / 1.458×（bm256 1.330× / 1.547× / 1.443×）。

- 与 bm256 比（时间）：
  - T ≤ 2048：0.65–0.79。
  - ep_uniform T=4096（每个 block 恰好 128 个 route）：0.76–0.82；ep_uniform T=12288（每个 expert 256 + 128）：0.86。
  - 其余点 0.93–1.005。
  - 例外是 random T=4096 持续态：路径上 1.036，但两个 kernel 单测都更快（GateUp 289.5 vs 298.4 µs，Down 159.5 vs
    163.2 µs）。这一点的路径样本与前一个被测变体有关；突发态同一点是 0.967。
- 与各点最好的路径比：
  - T ≤ 2048：0.64–0.75。small、split、merged 在这一段是 asm 的 1.15–1.44×。
  - T = 4096：0.88–0.95。
  - T ≥ 6144：差距 ≤ 1.2%，只有 random T=8192 例外（0.886 / 0.901，merged 最好）。
- kernel（T ≤ 2048，持续态）：
  - GateUp：bm128 190–213 µs，BM256 261–289 µs，small-M / merged 127–167 µs。
  - Down：bm128 109–123 µs，BM256 144–156 µs，small-M / merged 66–96 µs。
- T ≤ 2048 剩下差距的可能来源（推断，未单独测量）：
  - 同一 grid 下每个 CU 只驻留一个 8-wave workgroup（64 KiB LDS）。
  - ≤ 128 个 route 的 block 仍按 128 行计算：T=256 时每个 expert 8 行，94% 是 padding。
  - B 每个 K block 经 LDS 读入寄存器。
  - 对照之下，small-M / merged 每 CU 两个 4-wave workgroup，并按 16 行 slice 在 1..8 个之间选。

### 9.4 对 aiter 的意义

- aiter hook 调用的入口没变，默认即走 bm128。
- 与 asm 的交点：
  - aiter 的 `AITER_FMOE_FP8_BLOCKSCALE_FLYDSL_MIN_TOKENS` 默认 4096。
  - bm128 在 T=2048 时：random 是 asm 的 1.007× / 1.030×，ep_uniform 是 0.911× / 0.924×。
  - T ≤ 1024 时为 0.83–0.89×。
  - 因此阈值保持 4096；random 类负载可以降到 2048。

## 10. 精简：只保留 BM256 + bm128 和 merged

- 删除：
  - `src/contrib/flydsl/moe_small_m_a8w8_blockscale_gfx942.py`：第一至第三版的 4-wave small-M GateUp / Down，包括方案 B
    的 `all_blocks`。
  - `fused_moe_a8w8_blockscale_gfx942_mixed` / `_sort` 的 `big_blocks`（`"bm256"`、`"split"`）和 `ring` 参数。
  - BM256 GateUp / Down 的 `big_blocks_only`（只有 `"bm256"` 选项用到）。
  - 删除前的最后一版在 commit da764f5。
- 保留：
  - aiter 入口 `fused_moe_a8w8_blockscale_gfx942`：BM256 + `bm128`（默认开）。
  - `_sort` / `_mixed`：merged。sort block 超过 `MAX_BLOCKS`（2048）时退回 BM256 + bm128（以前退回 small-M + BM256）。
- merged 的 pair 任务本来就是独立的一份代码，不 import small-M 模块。机器码（MiMo 形状）：
  - merged GateUp / Down 与 da764f5 逐字节相同。
  - aiter 路径的 GateUp FP8 与 Down atomic sorted（有无 `single_store`），在 bm128 开、关两种情况下都逐字节相同。
- 测试：`test_moe_small_m_gfx942.py` 去掉 small-M kernel 的测试，fused / dispatch 只测 merged，容量退回改测 BM256 + bm128。
  merged GateUp 在有第 128 行的 block 上与 BM256 GateUp 逐位相同，其余 block 的 route 行与参考比较。
  三个测试文件（merged、Down、GateUp）共 165 passed。
- bench（`bench_moe_small_m_gfx942.py`）的 paths 只剩 asm、bm256、bm128、merged。
- 实验目录里的工具用到了已删的代码（small-M 模块、`big_blocks`、`big_blocks_only`、bench 的 small / split 路径），
  要在 da764f5 上运行。

## 11. 精简：BM256 GateUp / Down 只保留 aiter 路径的配置

- 删除的是只给测试和对照用的编译期选项，aiter 路径和 merged 都不用：
  - GateUp：`layout`（只留 2x4_pf1）、`xcd_swizzle`（总是开）、`pair_blocks`、BF16 输出 `out[T, topk, I]`。现在只写排序后的
    FP8 A2，`out_scale_t`、`topk` 成为必填参数。
  - Down：`layout`（只留 4x2_pf1）、`xcd_swizzle`、未排序的 A2（按 `[T, topk, I]` gather）、`reduce="kernel"`（写 rows 和
    inverse，再由 `moe_down_reduce` 求和；包括 `moe_down_rows_a8w8_blockscale_gfx942_lds`、`compile_moe_down_reduce`、
    kernel 参数 `pInverse`）。H ≤ 16384 的限制只来自 reduce kernel（每个 token 一个 H/16 线程的 workgroup），一并去掉。
  - 删除前的最后一版在 commit 1a4fc73。
- 现在的接口：
  - `moe_gateup_a8w8_blockscale_gfx942_lds(a, w1, a_scale_t, w1_scale, sorted_ids, expert_ids, num_valid_ids, out,
    out_scale_t, topk, bm128=False)`。
  - `moe_down_a8w8_blockscale_gfx942_lds(a2, w2, a2_scale_t, w2_scale, sorted_ids, sorted_weights, expert_ids,
    num_valid_ids, out, topk, zero_out=True, single_store=False, route_counts=None, bm128=False)`，a2 为
    `[blocks*256, I]`。
  - compile 函数只剩 `merged`、`bm128`，Down 另有 `single_store`。
- 机器码（MiMo 形状，与 1a4fc73 比，`FLYDSL_RUNTIME_ENABLE_CACHE=0`）：
  - GateUp（bm128 开、关）和 merged GateUp 逐字节相同。
  - Down 去掉 `pInverse` 后 kernarg 从 96 字节变为 88 字节，num_tokens / num_blocks 的偏移从 0x58 变为 0x50。
    不带 `single_store` 的 Down（bm128 开、关）和 merged Down 只有这个立即数和 kernarg 元数据不同。
  - 带 `single_store` 时 pNumValid 与 pRouteCounts 变成相邻参数，两条 `s_load_dwordx2` 合成一条 `s_load_dwordx4`，
    SGPR 重新分配：K 循环 opcode 相同，只是 SGPR 编号不同。
  - VGPR（Down 246、merged Down 250）、SGPR、LDS 都不变；route counts、zero rows kernel 逐字节相同。
  - `audit_moe_pipeline_gfx942.py`（只剩 gateup_2x4_pf1、down_4x2_pf1）两个 kernel 都 PASS。
- 测试：
  - GateUp 只剩 `test_gateup_fp8_sorted`（6 个 shape），BF16 输出的 `test_gateup` 及 4x2、`pair_blocks`、linear 配置已删除。
  - Down：`test_down` 有 4 个配置（atomic、single_store、bm128、single_store + bm128），各跑 6 个 shape。
    `test_gateup_down_fp8_pipeline` 只测 GateUp → Down（atomic、single_store）。`test_validate_extents` 改为检查 A2 的
    描述符上限。
  - 三个文件（GateUp、Down、merged）共 66 passed。`test_aiter_fused_moe_flydsl_gfx942.py` 的 5 例因本机 aiter 没有 hook
    被跳过，aiter 入口由 `test_fused_entry` 覆盖。
- bench 和工具：
  - `bench_moe_a8w8_blockscale_gfx942_vs_asm.py` 去掉 `--gemm1-layout` / `--gemm2-layout`。
  - `bench_fused_moe_a2a_gfx942.py` 去掉 FlyDSL+pair 路径，只比 asm 和 FlyDSL。
  - Down 测试文件的 `benchmark()` 去掉 layout、未排序 A2、reduce kernel 的变体；GateUp 的 `benchmark()` 改为 FP8 输出，
    去掉 `--no-xcd-swizzle`。
- 文档中用到这些选项的数据，要在 1a4fc73 或更早的版本上复现：
  - [moe_gateup_a8w8_blockscale_gfx942.md](../moe_gateup_a8w8_blockscale_gfx942.md) 的 4x2、`pair_blocks`、BF16 输出；
  - [moe_a8w8_blockscale_gfx942.md](../moe_a8w8_blockscale_gfx942.md) §3、§6.1 的 `pair_blocks`。

