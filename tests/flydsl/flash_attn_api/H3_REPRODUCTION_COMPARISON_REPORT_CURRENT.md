# H3 Attention srdc-52 与 srdc-7 对比报告

日期：2026-08-15

本报告对比两台不同服务器上的数据：

- `srdc-52`（`hjbog-srdc-52.amd.com`）：历史报告
	[H3_FIVE_KERNEL_PERFORMANCE_REPORT.md](https://github.com/luocheng25/pyhip/blob/luocheng/try-mha-308-h3/tests/flydsl/H3_FIVE_KERNEL_PERFORMANCE_REPORT.md)
	中的数据；
- `srdc-7`（`hjbog-srdc-7.amd.com`）：2026-08-15 BF16重测数据。

两台服务器均使用MI308X，AITER code object的SHA256也相同，但软件栈、驱动、固件和
系统设置并不完全一致。因此这是跨服务器实测对比，不能视为同一服务器上的单变量软件
版本消融。

`srdc-7`本次测试5个BF16实现。全部测试均执行正确性检查、每项
`3 warmup + 70 dispatch`，并以1 ms间隔采样GPU SCLK、PPT功耗和温度。全部测试使用
auto DPM，未锁频，未修改650 W power cap。PTL保持启用，但两次数据使用的格式不同：

- 2026-08-13：`sudo amd-smi set --ptl-format F16,BF16`；
- 2026-08-15最新数据：`sudo amd-smi set --ptl-format VECTOR,F8`。

本次受控复现仅在物理GPU 4上切换相同格式，分别重跑五个实现；profiler在每个进程的
preflight和结束快照中都记录并校验`ptl_state/ptl_format`。

测试结构和统计口径沿用原
[`H3_REPRODUCTION_COMPARISON_REPORT.md`](https://github.com/luocheng25/pyhip/blob/luocheng/try-mha-308-h3/tests/flydsl/H3_REPRODUCTION_COMPARISON_REPORT.md)，
并作以下实现替换：

- 删除原报告的dense近似和逐segment varlen行；
- 新增当前linear K/V公开接口；4-wave在接口内各执行一次paged K/V转换，再执行一次
	whole-batch paged MHA；
- 新增相同wrapper和数据布局下的whole-batch paged 8-wave MHA；
- 保留并重测 `ASM MI300`、`Triton`、`ASM MI308`。

当前4-wave和8-wave在srdc-52都没有相同wrapper、数据布局和single-MHA调度的历史数据，
因此srdc-52列保持为空，不借用旧4-wave或dense结果。性能参考基线统一设为本轮srdc-7
的ASM MI300 `193.558T`；srdc-52历史列只用于跨服务器背景。

本次只测试BF16五项。原报告FP8部分不复制，也不更新。

## 第一部分：BF16

### 性能对比

| 实现 | srdc-52耗时 | srdc-7耗时 | srdc-52 TFLOPS | srdc-7 TFLOPS | srdc-7相对srdc-52 | 相对ASM MI300 | srdc-7 CV | srdc-7稳态SCLK | 周期 srdc-52 / srdc-7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| new flydsl 8wave | — | **137.396 ms** | — | **210.980** | — | **+9.00%** | 10.815% | 1348-1761 MHz | — / 是 |
| new flydsl 4wave | — | **138.401 ms** | — | **208.725** | — | **+7.84%** | 8.931% | 1360-1723 MHz | — / 是 |
| ASM MI300 | 168.773 ms | 148.508 ms | 172.129 | 193.558 | +12.45% | 基线 | 5.556% | 1452-1747 MHz | 是 / 是 |
| Triton | 191.049 ms | 182.957 ms | 149.988 | 156.613 | +4.42% | -19.09% | **0.044%** | 1784-1795 MHz | 否 / 否 |
| ASM MI308 | 191.377 ms | 191.764 ms | 149.727 | 149.420 | -0.21% | -22.80% | 0.058% | 1782-1791 MHz | 否 / 否 |

本次BF16排名为：

```text
new flydsl 8wave > new flydsl 4wave > ASM MI300 > Triton > ASM MI308
```

以本轮ASM MI300 `193.558T`为性能基线：

- new flydsl 8wave：`+9.00%`；
- new flydsl 4wave：`+7.84%`；
- Triton：`-19.09%`；
- ASM MI308：`-22.80%`。

8-wave比4-wave的70-dispatch平均吞吐高`1.08%`。跨服务器只对存在同实现历史数据的
三项计算：ASM MI300 `+12.45%`、Triton `+4.42%`、ASM MI308 `-0.21%`。当前4-wave和
8-wave均无srdc-52同实现数据，不计算跨服务器加速。

两个FlyDSL实现的高吞吐burst间隔为7或8个dispatch，对应约`0.96-1.11 s`，严格算法
均给出`cycle_detected=True`。8-wave的70次均值/中值为`210.980T/207.039T`，最快
dispatch为`119.628 ms / 239.520T`；4-wave为`208.725T/210.169T`，最快dispatch为
`124.321 ms / 230.479T`。`VECTOR,F8`下auto-DPM高档占比明显高于`F16,BF16`，因此
完整均值也接近指定历史参考值。

### PTL格式性能对比

除PTL格式外，两组都使用相同GPU、代码、shape、数据、auto DPM、650 W power cap和
`3+70`统计口径。下表“目标”是原报告数据，“重测”是本次重新执行的数据；单位均为
TFLOPS。

| 实现 | 8月13日数据 `F16,BF16` | `F16,BF16`重测 | 偏差 | 最新目标 `VECTOR,F8` | `VECTOR,F8`重测 | 偏差 |
|---|---:|---:|---:|---:|---:|---:|
| new flydsl 8wave | 182.305 | 182.813 | +0.28% | 210.980 | 210.951 | -0.01% |
| new flydsl 4wave | 181.456 | 181.386 | -0.04% | 208.725 | 208.691 | -0.02% |
| ASM MI300 | 172.071 | 172.105 | +0.02% | 193.558 | 193.526 | -0.02% |
| Triton | 156.113 | 156.122 | +0.01% | 156.613 | 156.622 | +0.01% |
| ASM MI308 | 149.284 | 149.255 | -0.02% | 149.420 | 149.350 | -0.05% |

两组各含5个实现、350条有效dispatch，正确性全部通过；JSON中的preflight和结束快照
分别保持`Enabled / F16,BF16`与`Enabled / VECTOR,F8`。`F16,BF16`重测相对8月13日
目标的最大绝对偏差为`0.28%`，`VECTOR,F8`重测相对最新目标的最大绝对偏差为`0.05%`，
因此两组数据均可复现。

在两组本次重测之间，切换为`VECTOR,F8`后，4-wave、8-wave和ASM MI300分别提高
`15.05%`、`15.39%`和`12.45%`，Triton与ASM MI308只变化`+0.32%/+0.06%`。
对应稳态SCLK范围也从`1158-1701 / 1147-1717 / 1241-1739 MHz`提高到
`1361-1724 / 1347-1760 / 1450-1747 MHz`。结果确认PTL格式是两次数据差异的关键
测试条件：它改变了功率敏感实现的auto-DPM分布，而不是kernel源码或AITER code object。

### `VECTOR,F8`当前耗时分布

下表全部来自本轮70条event记录；“最快TFLOPS”按该实现最短单次event时间计算，不作为
稳态均值使用。

| 实现 | 均值耗时 | 中值耗时 | 最快耗时 | 最慢耗时 | 均值TFLOPS | 中值TFLOPS | 最快TFLOPS | CV |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| new flydsl 8wave | 137.396 ms | 138.396 ms | 119.628 ms | 155.163 ms | 210.980 | 207.039 | 239.520 | 10.815% |
| new flydsl 4wave | 138.401 ms | 136.339 ms | 124.321 ms | 156.141 ms | 208.725 | 210.169 | 230.479 | 8.931% |
| ASM MI300 | 148.508 ms | 141.822 ms | 140.806 ms | 165.317 ms | 193.558 | 202.037 | 203.496 | 5.556% |
| Triton | 182.957 ms | 182.940 ms | 182.834 ms | 183.272 ms | 156.613 | 156.627 | 156.718 | 0.044% |
| ASM MI308 | 191.764 ms | 191.801 ms | 191.568 ms | 192.252 ms | 149.420 | 149.391 | 149.572 | 0.058% |

Triton和MI308均为稳定高频状态，CV分别仅`0.044%/0.058%`，不检测为DPM周期。

### srdc-7 `VECTOR,F8` GPU频率数据

稳态范围排除每个实现最初两个dispatch的正常爬频。P05/中值/P95和标准差基于每次
dispatch内1 ms SCLK样本的算术平均值。

| 实现 | 全程观测范围 | 稳态观测范围 | 稳态均值P05 / 中值 / P95 | 稳态均值标准差 | 频率点数 | 周期 |
|---|---:|---:|---:|---:|---:|---|
| new flydsl 8wave | 1348-1761 MHz | 1348-1761 MHz | 1377.0 / 1505.0 / 1746.1 MHz | 139.9 MHz | 7725 | 是，约1 s |
| new flydsl 4wave | 1360-1723 MHz | 1360-1723 MHz | 1395.1 / 1564.7 / 1709.2 MHz | 117.8 MHz | 7794 | 是，约1 s |
| ASM MI300 | 1439-1747 MHz | 1452-1747 MHz | 1542.5 / 1692.3 / 1740.0 MHz | 73.2 MHz | 8324 | 是，1.042-1.044 s |
| Triton | 1444-1795 MHz | 1784-1795 MHz | 1792.8 / 1793.3 / 1794.0 MHz | 0.7 MHz | 10263 | 否 |
| ASM MI308 | 1442-1791 MHz | 1782-1791 MHz | 1788.3 / 1788.9 / 1790.4 MHz | 0.8 MHz | 10784 | 否 |

频率结论：

1. 4-wave、8-wave和ASM MI300均有明显auto-DPM高低档切换；
2. 按4-wave/8-wave/MI300顺序，吞吐与SCLK相关系数为`0.8062/0.8037/0.7932`；
3. 三者峰谷吞吐降幅分别为`20.38%/22.90%/14.83%`；
4. 三者`ppt_accumulated`增量分别为`3614/737/5713`，Triton/MI308为`1/0`；
5. MI300严格复现每7个dispatch一个高吞吐区间，周期为`1.042-1.044 s`；
6. Triton和MI308总体稳定，CV仅为`0.044%/0.058%`。

### srdc-7正确性

参考为segment-wise BF16 PyTorch SDPA。63225-token主段与7-token尾段独立计算，tail单独
打分。

| 实现 | whole cosine | relative L2 | max abs | tail cosine | 结果 |
|---|---:|---:|---:|---:|---|
| new flydsl 8wave | 0.999997258 | 0.002417377 | 0.000244141 | 1.000000119 | 通过 |
| new flydsl 4wave | 0.999997258 | 0.002417369 | 0.000244141 | 1.000000119 | 通过 |
| Triton | 0.999999940 | 0.000502826 | 0.000244141 | 1.000000119 | 通过 |
| ASM MI308 RTNA | 0.999999881 | 0.000568197 | 0.000244141 | 1.000000119 | 通过 |
| ASM MI300 RTNA | 0.999999881 | 0.000568194 | 0.000244141 | 1.000000119 | 通过 |

五项结果均finite，max-abs相同；7-token tail cosine均为`1.000000119`。

## 第二部分：FP8

本次范围明确限定为上述五项BF16实现，未重测FP8。原报告FP8数据不复制到本报告，避免
造成FP8也已随当前4/8-wave varlen实现重新验证的误解。

## 公共测试口径与环境

- 服务器映射：`srdc-52`为`hjbog-srdc-52.amd.com`，`srdc-7`为
	`hjbog-srdc-7.amd.com`；
- shape：segments `(63225,7)`，`Hq=Hkv=14`，`Dq=Dv=128`，seed `1101`；
- dtype：BF16，non-causal，scale=`1/sqrt(128)`；
- 真实FLOPs：`28,653,368,031,232`；
- 目标：物理GPU 4，MI308X，`gfx942`，80 CU，650 W；
- DPM：`auto`，未锁频；
- PTL：`Enabled`；8月13日格式为`F16,BF16`，最新数据格式为`VECTOR,F8`；
- 每项`3 warmup + 70 dispatch`，CUDA event只覆盖被测接口；
- 以1 ms采样SCLK、PPT、junction和HBM温度；
- `srdc-7`：Python `3.12.13`、Torch `2.10.0+git8514f05`、HIP `7.2.53211`、
	ROCm `7.2.3`、AMDGPU driver `6.16.13`、FlyDSL `0.3.0`、Triton `3.6.0`；
- AITER源码：`v0.1.14-rc0-238-g31c4b3e64-dirty`，提交
	`31c4b3e64343eaef40477b339c36eed511bf94d4`；
- 被测kernel/API提交：`1c3058dc981d923f7d5c58d23adaab2db2374065`；原始最新数据测试
	前后工作树干净，PTL A/B只增加profiler环境取证，不修改被测路径；
- `srdc-52`：Python 3.10、Torch 2.9、ROCm 7.0、AITER 0.1.14、FlyDSL 0.3.0；
- `srdc-7`的AITER BF16 MI308、MI300 `.co` SHA256与`srdc-52`相同；
- `srdc-7`当前`kernel.numa_balancing=0`；
- 两台服务器的软件栈、驱动、固件和系统设置不完全相同，不能把性能或周期差异归因于
	单一组件。

### Code object

| 实现 | 实际源文件 | SHA256 |
|---|---|---|
| ASM MI308 | `hsa/gfx942/fmha_v3_fwd/MI308/fwd_hd128_bf16_rtna_group.co` | `3687c5610a454572e4a615ec58f05e707fdf3995e4dc932cf2219ad2fa0052ff` |
| ASM MI300 | `hsa/gfx942/fmha_v3_fwd/MI300/fwd_hd128_bf16_rtna_group.co` | `f8d7e1dfc5301edeb83e5520e8d710798c7641a52040c33dbed77c18115813c5` |

两项SHA与上游报告相同。当前PCI ID会让AITER选择MI308目录；profiler使用临时
`AITER_ASM_DIR`映射，并通过`Path.resolve()`验证实际源文件，不覆盖安装目录中的活动`.co`。
MI308和MI300分进程测试，避免AITER按kernel名缓存已加载code object。

### 环境偏差

原始最新数据和PTL A/B共9个正式profile进程开始时GPU均为`busy=0`、DPM=`auto`，有
14个零CU占用resident context，初始VRAM为`152.89 GiB`。PTL A/B的preflight和结束
快照还分别确认格式始终保持`F16,BF16`或`VECTOR,F8`。严格preflight默认会拒绝resident
context；本次
使用显式`ATTN_PROFILE_ALLOW_RESIDENT_PROCESSES=1`和200000 MiB初始VRAM阈值放行，并把
完整进程列表写入schema-v2 JSON。

每个profile进程结束快照均为DPM=`auto`、`gpu_busy_percent=0-1`，15个context、约
`156.82-157.67 GiB` VRAM。启动快照没有并发外部CU活动，但profiler没有在dispatch期间持续
采样逐进程CU占用；高驻留VRAM环境也与srdc-52及此前低驻留重测明显不同，结果不应解释为
纯kernel单变量消融。

## 数据与复现

原始最新数据产物：

```text
/tmp/h3-reproduction-20260815-1c3058d
```

- `profile-flydsl-4w8w.json/.log`；
- `profile-triton-mi308.json/.log`；
- `profile-mi300.json/.log`；
- `SHA256SUMS`。

PTL受控复现产物：

```text
/tmp/h3-reproduction-20260815-ptl-f16-bf16-1c3058d
/tmp/h3-reproduction-20260815-ptl-vector-f8-1c3058d
```

每个目录均包含相同的三份JSON、三份stdout日志和`SHA256SUMS`。全部profile为schema v2，
每项70条dispatch；所有event时间、TFLOPS和sensor count均为正。三组产物各自执行
`sha256sum -c SHA256SUMS`均全部通过。

PTL条件设置如下。无`--gpu`时命令作用于所有GPU；本次在共享服务器上使用
`--gpu 4`限定到被测卡，读回格式与下列命令相同。

```bash
# 复现2026-08-13数据
sudo amd-smi set --ptl-format F16,BF16

# 复现2026-08-15最新数据
sudo amd-smi set --ptl-format VECTOR,F8
```

设置其中一种PTL格式后，令`PTL_FORMAT`和`OUTPUT_DIR`与之对应，再执行三个独立profile
进程：

```bash
# F16,BF16
PTL_FORMAT=F16,BF16
OUTPUT_DIR=/tmp/h3-reproduction-20260815-ptl-f16-bf16-1c3058d

# 或VECTOR,F8
# PTL_FORMAT=VECTOR,F8
# OUTPUT_DIR=/tmp/h3-reproduction-20260815-ptl-vector-f8-1c3058d

# new flydsl 4wave + 8wave
HIP_VISIBLE_DEVICES=4 \
ATTN_PROFILE_IMPLS=new_flydsl_4wave,new_flydsl_8wave \
ATTN_PROFILE_WARMUP=3 ATTN_PROFILE_ITERS=70 \
ATTN_PROFILE_SENSOR_INTERVAL_MS=1 \
ATTN_PROFILE_ALLOW_RESIDENT_PROCESSES=1 \
ATTN_PROFILE_MAX_INITIAL_VRAM_MIB=200000 \
ATTN_PROFILE_EXPECT_PTL_FORMAT="$PTL_FORMAT" \
ATTN_PROFILE_OUTPUT="$OUTPUT_DIR/profile-flydsl-4w8w.json" \
FLYDSL_RUNTIME_ENABLE_CACHE=0 \
.venv/bin/python -B tests/flydsl/flash_attn_api/profile_h3_reproduction_current.py

# Triton + MI308 ASM
HIP_VISIBLE_DEVICES=4 \
ATTN_PROFILE_IMPLS=triton,asm_mi308 \
ATTN_PROFILE_WARMUP=3 ATTN_PROFILE_ITERS=70 \
ATTN_PROFILE_SENSOR_INTERVAL_MS=1 \
ATTN_PROFILE_ALLOW_RESIDENT_PROCESSES=1 \
ATTN_PROFILE_MAX_INITIAL_VRAM_MIB=200000 \
ATTN_PROFILE_EXPECT_PTL_FORMAT="$PTL_FORMAT" \
ATTN_PROFILE_OUTPUT="$OUTPUT_DIR/profile-triton-mi308.json" \
.venv/bin/python -B tests/flydsl/flash_attn_api/profile_h3_reproduction_current.py

# MI300 ASM必须使用独立进程
HIP_VISIBLE_DEVICES=4 \
ATTN_PROFILE_IMPLS=asm_mi300 \
ATTN_PROFILE_WARMUP=3 ATTN_PROFILE_ITERS=70 \
ATTN_PROFILE_SENSOR_INTERVAL_MS=1 \
ATTN_PROFILE_ALLOW_RESIDENT_PROCESSES=1 \
ATTN_PROFILE_MAX_INITIAL_VRAM_MIB=200000 \
ATTN_PROFILE_EXPECT_PTL_FORMAT="$PTL_FORMAT" \
ATTN_PROFILE_OUTPUT="$OUTPUT_DIR/profile-mi300.json" \
.venv/bin/python -B tests/flydsl/flash_attn_api/profile_h3_reproduction_current.py
```

当GPU 4完全无KFD context且初始VRAM低于1 GiB时，应删除两个`ATTN_PROFILE_ALLOW_*`
覆盖，恢复与上游报告完全一致的严格preflight。