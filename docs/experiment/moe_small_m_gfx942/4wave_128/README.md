# 4-wave small-M，每个 workgroup 128×256 输出 tile（第三版）—— 实验数据

过程与结论见 [docs/record/moe_small_m_gfx942.md](../../../record/moe_small_m_gfx942.md) §7。第二版（GateUp 64 列）的数据在
[../4wave/README.md](../4wave/README.md)，工具和对照源码也在那里（`../4wave/tools`、`../4wave/variants`）。

## 文件

| 文件 | 内容 |
|---|---|
| `final_ep_uniform.{log,json}`、`final_random.{log,json}` | 持续态（每个变体先跑 ≥50 ms），T=256..4096：asm 6 个、bm256、small、old8、c64 路径和单 kernel |
| `burst_ep_uniform.{log,json}`、`burst_random.{log,json}` | 突发态（`--warmup-ms 0`），asm 6 个、small、old8、c64 |
| `final_large.{log,json}` | 持续态 random T=6144/8192（大部分 block 走 BM256），asm、bm256、small、c64 |
| `final_random.attempt1.log` | 门限没过、重跑前的那次（保留） |
| `run_128.wait` | 每次运行的开始 / 结束 / 重跑记录 |
| `ab_kernels.log` | kernel 级交替 A/B（c64、down128、cur、old8） |
| `vgpr.txt` | 每个 slice 数的 VGPR（full / light 布局、直接实现），及踩到的 spill |
| `isa_mix.txt` | K 循环每 wave 每 K block 的指令构成（`tools/isa_mix.py`，c64 / old8 / cur，S=1/2/4/8），record §7.7 |
| `pmc_gateup32.txt` | GateUp 32 行的硬件计数（`tools/pmc_compare.py` 的第一版门控，在一个空闲窗口里测得），record §7.7 |
| `down_n128/` | 中间版本（GateUp 128×256、Down 128×128）的同一组数据与 A/B（`ab_kernels.log` 的 cur 即该版本，按 activity 门控；`ab_kernels_light.log` 是所有 slice 都用 light 布局的版本，当时只有 KFD occupancy 门控，可能受外部作业干扰） |

路径名：small = 当前代码（aiter `moe_sorting(256)` 清零 out + `fused_moe_a8w8_blockscale_gfx942_mixed`，T>128 时
5 个 kernel）；old8 = 第一版（sort128 + 8-wave GateUp/Down）；c64 = 第二版（4 wave、GateUp 64 列、A2 按 64 列量化，
经它自己的 mixed 分支）；bm256 = 原 BM256 路径；tuned asm = 6 个 asm kernel 中含 sort 最快者。

## 复现

```bash
D=docs/experiment/moe_small_m_gfx942
export PYTHONPATH=<含 pyhip -> src 链接的目录>:$PYTHONPATH
# 路径级，GPU 1：每次运行前等 30 s 空闲（activity <= 3 %、VRAM <= 20 %），门限不过就整次重跑
GPU=1 $D/4wave/tools/run_128.sh
python $D/4wave/tools/summarize128.py $D/4wave_128/final_*.json $D/4wave_128/burst_*.json
# kernel 级交替 A/B（每轮前 3 s 空闲、结束后再查一次，被打扰的轮重跑）
HIP_VISIBLE_DEVICES=2 python $D/4wave/tools/ab_kernels.py --impls c64 down128 cur old8 --rows 16 32 64 96 112 128
# 正确性
cd tests/contrib/moe && python -m pytest -q -p no:cacheprovider --import-mode=importlib test_moe_small_m_gfx942.py
```

## 1. kernel 级 A/B

`ab_kernels.py`，32 个 block × rows 行（每 expert 一个），同一份权重反复调用，6 轮交替取中位数（µs）。down128 =
当前代码把 Down 改回 128 列（`variants/moe_small_m_4wave_down128.py`），其 GateUp 与 cur 相同。

| rows (S) | GateUp c64 | cur | old8 | cur/old8 | Down c64 | down128 | cur | old8 | cur/old8 |
|---|---|---|---|---|---|---|---|---|---|
| 16 (1) | 122.5 | 116.9 | 125.5 | 1.074× | 65.6 | 66.0 | 63.6 | 66.8 | 1.050× |
| 32 (2) | 128.3 | 120.8 | 125.2 | 1.036× | 69.4 | 69.0 | 65.6 | 70.1 | 1.069× |
| 64 (4) | 146.1 | 131.5 | 139.2 | 1.059× | 78.4 | 77.3 | 74.3 | 77.5 | 1.043× |
| 96 (6) | 176.2 | 158.0 | 167.1 | 1.058× | 97.3 | 93.5 | 87.0 | 95.1 | 1.093× |
| 112 (7) | 194.7 | 182.8 | 186.9 | 1.022× | 109.4 | 105.8 | 99.6 | 107.5 | 1.079× |
| 128 (8) | 209.7 | 198.3 | 205.8 | 1.038× | 118.9 | 115.1 | 108.7 | 119.0 | 1.095× |

（×：old8 时间 / cur 时间。）GateUp 128 列比 64 列快 5–12%，Down 256 列比 128 列快 4–8%；两个 kernel 都比 8-wave 版快
2–10%。`down_n128/ab_kernels.log` 另有直接实现（plain128）：S≤6 与 full 布局相当，S=7/8 spill，GateUp 只有 64 列版的
0.82× / 0.22×；`down_n128/ab_kernels_light.log`：所有 slice 都用 light 布局时 S≤5 比 full 慢 3–5%（门控较弱，只作方向参考）。

## 2. 寄存器（`vgpr.txt`）

128×256 tile 的 accumulator 在 S=8 时是 128 VGPR（4 组 × 4 × 8）。full 布局 S≤6 放得下（242），S=7/8 需要 ~275 / ~300；
light 布局 238 / 254。按 slice 数选择后：GateUp 256、Down 254 VGPR，都不 spill。

## 3. 正确性

- `test_moe_small_m_gfx942.py`：21 passed、1 skipped（K/128 为奇数的形状）。small 与 BM256 都按 (行, 128 列) 量化，fused
  路径直接用 asm 测试的参考。
- 与直接实现逐位比较（`variants/moe_small_m_4wave_128col_plain.py`，T=256/1024/2048/4096）：GateUp 输出与 scale 20 次
  调用全部一致；Down 在 ep_uniform 下一致，random 下只差 BF16 原子加的求和顺序（最大 0.125）。
- bench 中 small 路径对 asm 参考的 calc_diff：ep_uniform 1.37–1.41e-6，random 2.53–2.67e-6，与 old8、bm256 相同。

## 4. 路径级（含 sort，中位数 µs，括号内为相对 tuned asm 的加速比）

全部运行的门限通过（activity ≤5%，VRAM ≤4.6 GB）；final_random 第一次门限没过，重跑。

### 4.1 持续态

| T | ep: tuned asm | small | old8 | c64 | random: tuned asm | small | old8 | c64 |
|---|---|---|---|---|---|---|---|---|
| 256 | 261.7 | 208.7 (1.254) | 216.4 (1.210) | 221.7 (1.181) | 261.9 | 209.6 (1.250) | 217.9 (1.202) | 221.5 (1.182) |
| 512 | 271.8 | 217.9 (1.247) | 218.6 (1.243) | 221.0 (1.230) | 272.7 | 218.2 (1.250) | 224.3 (1.216) | 227.2 (1.200) |
| 1024 | 292.7 | 253.0 (1.157) | 224.0 (1.306) | 260.6 (1.123) | 294.7 | 254.4 (1.159) | 258.5 (1.140) | 257.6 (1.144) |
| 2048 | 333.1 | 282.5 (1.179) | 279.9 (1.190) | 306.4 (1.087) | 373.1 | 285.5 (1.307) | 286.5 (1.302) | 308.0 (1.211) |
| 4096 | 499.7 | 365.9 (1.366) | 372.5 (1.342) | 397.1 (1.259) | 525.3 | 597.5 (0.879) | 486.5 (1.080) | 626.2 (0.839) |

持续态在两个功耗档之间切换，每轮中位数例如 ep T=1024 small 252/226/253/254、old8 227/219/229/223，random T=1024 small
256/260/224/226、old8 263/227/244/263；单点比值可差 ~10%，同一 T 的突发态更稳定。

### 4.2 突发态

| T | ep: tuned asm | small | old8 | c64 | random: tuned asm | small | old8 | c64 |
|---|---|---|---|---|---|---|---|---|
| 256 | 259.8 | 202.0 (1.286) | 211.2 (1.230) | 212.8 (1.221) | 261.5 | 202.0 (1.294) | 211.3 (1.238) | 214.6 (1.219) |
| 512 | 270.7 | 206.9 (1.308) | 212.6 (1.273) | 216.0 (1.253) | 273.8 | 211.1 (1.297) | 214.0 (1.280) | 221.6 (1.235) |
| 1024 | 287.6 | 219.7 (1.309) | 219.1 (1.312) | 228.5 (1.259) | 282.0 | 223.2 (1.263) | 227.9 (1.237) | 239.9 (1.176) |
| 2048 | 328.7 | 254.9 (1.289) | 257.3 (1.277) | 285.0 (1.153) | 351.1 | 274.9 (1.278) | 273.6 (1.284) | 313.0 (1.122) |
| 4096 | 533.2 | 398.8 (1.337) | 408.5 (1.305) | 434.9 (1.226) | 550.8 | 607.2 (0.907) | 494.2 (1.115) | 628.4 (0.876) |

### 4.3 大 T（持续态 random）

| T | tuned asm | small | c64 | bm256 |
|---|---|---|---|---|
| 6144 | 708.9 | 538.5 (1.316) | 546.7 (1.297) | 526.7 (1.346) |
| 8192 | 877.2 | 843.0 (1.041) | 851.7 (1.030) | 711.5 (1.233) |

### 4.4 单 kernel（持续态 ep_uniform / random，中位数 µs）

| T | GateUp small | Down small | GateUp old8 | Down old8 | BM256 GateUp (big only) | BM256 Down (big only) |
|---|---|---|---|---|---|---|
| 256 | 122.2 / 138.8 | 65.8 / 65.5 | 133.3 / 132.0 | 69.9 / 68.2 | 5.9 / 6.2 | 5.9 / 6.1 |
| 512 | 137.5 / 140.8 | 67.0 / 75.8 | 151.1 / 149.9 | 69.6 / 69.5 | 5.8 / 5.8 | 5.8 / 5.9 |
| 1024 | 144.8 / 146.4 | 77.8 / 81.8 | 149.9 / 152.0 | 82.1 / 82.8 | 5.9 / 6.0 | 5.8 / 6.0 |
| 2048 | 152.1 / 160.8 | 86.3 / 89.3 | 159.4 / 168.3 | 91.8 / 90.6 | 5.9 / 5.8 | 5.8 / 5.8 |
| 4096 | 225.4 / 165.9 | 123.6 / 87.7 | 222.5 / 287.2 | 130.0 / 151.7 | 6.4 / 213.6 | 6.5 / 103.1 |

- 单 kernel 在持续态里受功耗档影响（small 对 old8：GateUp −5%..+10%、Down −8%..+6%），交替 A/B（§1）更可靠：GateUp
  +2..7%、Down +4..10%。这部分领先在路径上被两个空的 BM256 launch（每个 ~6 µs，T>128 时总有）抵掉一部分：small 路径相对
  old8 是 ep 突发 0.997–1.046×（T≤4096）、random 突发 0.995–1.046×（T≤2048）。
- random T=4096 仍是 0.88–0.91×：12 个超过 128 route 的 expert 的 block 走 BM256，只处理 12 个 block 的 BM256
  GateUp/Down 要 213.6 / 103.1 µs。这一点与 tile 无关，是 256 行表内分支的问题（record §6.5、§7.6）。

## 5. 中间版本（`down_n128/`：GateUp 128×256、Down 128×128）

同样的协议（门限通过，ep_uniform 第一次运行时外部作业占着 131 GB 显存但空闲，`--max-vram-pct 100`）。small 相对 tuned
asm：ep 持续 1.153/1.261/1.153/1.168/1.347、突发 1.275/1.295/1.299/1.180/1.313；random 持续 1.237/1.233/1.242/1.279/0.864、
突发 1.275/1.277/1.270/1.309/0.909。Down 改为 256 列后，突发态 ep T=2048 从 1.180 到 1.289，其余在 ±3% 内；持续态受功耗档
影响，变化在 −7%..+9% 之间。

## 6. 测量注意

- 外部作业期间 KFD 不报告它的 CU occupancy（`gpu_contention.ExternalLoadMonitor` 读到 0，而 GFX activity 30–100%）。
  第一次路径级运行因此在负载下跑完了，只有 bench 自己的 amdsmi 门限（activity）发现；已作废。之后：kernel A/B 每轮前
  要求 3 s 内 activity ≤2%、VRAM ≤ `--max-vram-pct`，结束 0.3 s 后再查一次；路径 bench 每次运行前等 30 s 空闲，门限
  不过就整次重跑（`run_128.sh`）。
- 外部作业空闲时也可能占着 ~134 GB 显存，此时 VRAM 门限永远不过；`--max-vram-pct` / `MAX_VRAM_PCT` 可放宽（记录在 JSON
  的 args 里）。本目录顶层的数据都在 VRAM ≤20% 下测得。
