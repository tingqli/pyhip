# H3 Attention srdc-52 与 srdc-7 对比报告

日期：2026-08-13

本报告对比两台不同服务器上的数据：

- `srdc-52`（`hjbog-srdc-52.amd.com`）：历史报告
	[H3_FIVE_KERNEL_PERFORMANCE_REPORT.md](https://github.com/luocheng25/pyhip/blob/luocheng/try-mha-308-h3/tests/flydsl/H3_FIVE_KERNEL_PERFORMANCE_REPORT.md)
	中的数据；
- `srdc-7`（`hjbog-srdc-7.amd.com`）：2026-08-13 BF16重测数据。

两台服务器均使用MI308X，AITER code object的SHA256也相同，但软件栈、驱动、固件和
系统设置并不完全一致。因此这是跨服务器实测对比，不能视为同一服务器上的单变量软件
版本消融。

`srdc-7`本次测试5个BF16实现。全部测试均执行正确性检查、每项
`3 warmup + 70 dispatch`，并以1 ms间隔采样GPU SCLK、PPT功耗和温度。全部测试使用
auto DPM，未锁频，未修改650 W power cap。

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
的ASM MI300 `172.071T`；srdc-52历史列只用于跨服务器背景。

本次只测试BF16五项。原报告FP8部分不复制，也不更新。

## 第一部分：BF16

### 性能对比

| 实现 | srdc-52耗时 | srdc-7耗时 | srdc-52 TFLOPS | srdc-7 TFLOPS | srdc-7相对srdc-52 | 相对ASM MI300 | srdc-7 CV | srdc-7稳态SCLK | 周期 srdc-52 / srdc-7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| new flydsl 8wave | — | **161.139 ms** | — | **182.305** | — | **+5.95%** | 16.680% | 1154-1728 MHz | — / 是 |
| new flydsl 4wave | — | **161.298 ms** | — | **181.456** | — | **+5.45%** | 15.199% | 1155-1699 MHz | — / 是 |
| ASM MI300 | 168.773 ms | 169.210 ms | 172.129 | 172.071 | -0.03% | 基线 | 12.942% | 1241-1733 MHz | 是 / 是 |
| Triton | 191.049 ms | 183.546 ms | 149.988 | 156.113 | +4.08% | -9.27% | 0.464% | 1730-1793 MHz | 否 / 否* |
| ASM MI308 | 191.377 ms | 191.939 ms | 149.727 | 149.284 | -0.30% | -13.24% | **0.058%** | 1783-1790 MHz | 否 / 否 |

本次BF16排名为：

```text
new flydsl 8wave > new flydsl 4wave > ASM MI300 > Triton > ASM MI308
```

以本轮ASM MI300 `172.071T`为性能基线：

- new flydsl 8wave：`+5.95%`；
- new flydsl 4wave：`+5.45%`；
- Triton：`-9.27%`；
- ASM MI308：`-13.24%`。

8-wave比4-wave的70-dispatch平均吞吐高`0.47%`。跨服务器只对存在同实现历史数据的
三项计算：ASM MI300 `-0.03%`、Triton `+4.08%`、ASM MI308 `-0.30%`。当前4-wave和
8-wave均无srdc-52同实现数据，不计算跨服务器加速。

两个FlyDSL实现的高吞吐burst间隔均为6或7个dispatch，对应约`0.97-1.12 s`，严格算法
均给出`cycle_detected=True`。8-wave的70次均值/中值为`182.305T/167.104T`，最快
dispatch为`120.665 ms / 237.462T`；4-wave为`181.456T/165.753T`，最快dispatch为
`124.538 ms / 230.078T`。二者高频状态均超过210T，但auto-DPM未在完整70-dispatch
窗口内持续保持该状态。

### 与指定参考8-wave的对照

指定报告中的BF16参考值为8-wave `137.099 ms / 212.753T`、4-wave
`137.983 ms / 210.785T`。需要注意，参考8-wave是
`test_attn_8wave_32x32_lkgv.py`的单段dense近似；参考4-wave才是真varlen paged实现。

为区分源码回归和运行环境差异，2026-08-12曾把参考branch的原始源码隔离到
`/tmp/h3-reference-run`，并在相同GPU/软件环境直接执行`3+70`计时：

| 实现 | 历史报告 | 原始参考源码在当前宿主 | 当前linear K/V公开接口 | 当前相对同机参考源码 |
|---|---:|---:|---:|---:|
| 8-wave | 212.753T | 184.629T | 182.305T | -1.26% |
| 4-wave | 210.785T | 183.040T | 181.456T | -0.87% |

4-wave当前pipeline从block 0开始，不存在负prologue block，相关的无效编译期裁剪已删除。
4-wave和8-wave的page-table lookahead都使用按当前sequence真实页数限制的buffer
descriptor，越界load由硬件返回0，调用方不再需要追加guard page ID。此前guarded版本
在同进程`reference -> current -> current -> reference`夹心A/B中与
参考源码打平；本轮bounded-buffer正式进程相对此前同机参考源码为4-wave `-0.87%`、
8-wave `-1.26%`。该对照跨进程且受auto-DPM影响，不作为单变量消融。公开接口仍只接受
linear K/V：当4-wave选择vectorized K内部路径时，K转换在
`flash_attn_varlen_func`内执行并计入event。vectorized模式允许
`cu_seqlens_k=None`，此时复用`cu_seqlens_q`，因此只表示Q/K边界一致的self-attention。

相对本报告上一轮正式profile，本轮4-wave从`181.552T`变为`181.456T`（`-0.05%`），
8-wave从`182.307T`变为`182.305T`（`-0.00%`）。同轮重测的MI300、Triton和MI308基线
变化分别为`+0.03%/+0.04%/-0.02%`。五项均在`±0.05%`内，说明当前代码和resident
环境下的完整重测稳定复现上一轮结果。

此前微基准中，H3上K转换均值约`0.391 ms`，V转换约`0.375 ms`，合计不到总耗时的
`0.5%`，不是历史
均值缺口来源。8-wave测试了减少resident workgroup、静态grid、grid-stride、direct VMEM
epilogue和旧vectorized K路径；这些方案均未改善完整多周期均值，当前dynamic ticket +
C-shuffle + linear K仍是当前宿主上的最快组合。

历史参考测试结束时GPU为“无KFD进程、约284 MiB驱动保留显存”；本轮8张MI308X均有
约152-153 GiB驻留显存，GPU 4有10个零CU占用但宿主PID namespace不可见的resident
context。原始参考源码在该环境也比历史报告低约13%，所以当前容器内无法完成“70次均值
达到历史212.753T/210.785T”的环境级验收。本轮当前实现相对同机参考源码，4-wave相差
约0.9%、8-wave相差约1.3%；两者单次高频burst均超过历史平均目标。

### 当前耗时分布

下表全部来自本轮70条event记录；“最快TFLOPS”按该实现最短单次event时间计算，不作为
稳态均值使用。

| 实现 | 均值耗时 | 中值耗时 | 最快耗时 | 最慢耗时 | 均值TFLOPS | 中值TFLOPS | 最快TFLOPS | CV |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| new flydsl 8wave | 161.139 ms | 171.471 ms | 120.665 ms | 182.792 ms | 182.305 | 167.104 | 237.462 | 16.680% |
| new flydsl 4wave | 161.298 ms | 172.868 ms | 124.538 ms | 182.812 ms | 181.456 | 165.753 | 230.078 | 15.199% |
| ASM MI300 | 169.210 ms | 168.845 ms | 141.118 ms | 190.054 ms | 172.071 | 169.702 | 203.046 | 12.942% |
| Triton | 183.546 ms | 183.171 ms | 182.928 ms | 187.353 ms | 156.113 | 156.430 | 156.637 | 0.464% |
| ASM MI308 | 191.939 ms | 191.947 ms | 191.678 ms | 192.191 ms | 149.284 | 149.278 | 149.487 | 0.058% |

\* Triton的启发式周期字段为`true`，但CV仅`0.464%`、峰谷降幅仅`2.36%`，且所谓高区间
在后半程连续覆盖42个dispatch；本报告按“无实质周期”解释，不把阈值附近的小幅噪声视为
与FlyDSL/MI300相同的DPM周期。

### srdc-7 GPU频率数据

稳态范围排除每个实现最初两个dispatch的正常爬频。P05/中值/P95和标准差基于每次
dispatch内1 ms SCLK样本的算术平均值。

| 实现 | 全程观测范围 | 稳态观测范围 | 稳态均值P05 / 中值 / P95 | 稳态均值标准差 | 频率点数 | 周期 |
|---|---:|---:|---:|---:|---:|---|
| new flydsl 8wave | 1154-1734 MHz | 1154-1728 MHz | 1168.8 / 1242.5 / 1631.8 MHz | 165.6 MHz | 9068 | 是，约1 s |
| new flydsl 4wave | 1155-1699 MHz | 1155-1699 MHz | 1173.2 / 1257.0 / 1606.7 MHz | 160.2 MHz | 9057 | 是，约1 s |
| ASM MI300 | 1241-1733 MHz | 1241-1733 MHz | 1307.0 / 1448.6 / 1682.9 MHz | 142.9 MHz | 9575 | 是，1.021-1.022 s |
| Triton | 1442-1793 MHz | 1730-1793 MHz | 1771.2 / 1789.3 / 1792.0 MHz | 6.7 MHz | 10318 | 无实质周期* |
| ASM MI308 | 1442-1790 MHz | 1783-1790 MHz | 1786.3 / 1787.2 / 1789.2 MHz | 0.8 MHz | 10800 | 否 |

频率结论：

1. 4-wave、8-wave和ASM MI300均有明显auto-DPM高低档切换；
2. 按4-wave/8-wave/MI300顺序，吞吐与SCLK相关系数为`0.7925/0.7736/0.8204`；
3. 三者峰谷吞吐降幅分别为`31.88%/33.99%/25.75%`；
4. 三者`ppt_accumulated`增量分别为`2048/418/3722`，Triton/MI308为`3/0`；
5. MI300严格复现每6个dispatch一个高吞吐区间，周期为`1.021-1.022 s`；
6. Triton和MI308总体稳定，CV仅为`0.464%/0.058%`。

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

当前提交的直接测试同时全部通过：4-wave `3 passed`、8-wave `36 passed`、公开API
`11 passed`，合计`50 passed`。

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
- 每项`3 warmup + 70 dispatch`，CUDA event只覆盖被测接口；
- 以1 ms采样SCLK、PPT、junction和HBM温度；
- `srdc-7`：Python `3.12.13`、Torch `2.10.0+git8514f05`、HIP `7.2.53211`、
	ROCm `7.2.3`、AMDGPU driver `6.16.13`、FlyDSL `0.3.0`、Triton `3.6.0`；
- AITER源码：`v0.1.14-rc0-238-g31c4b3e64-dirty`，提交
	`31c4b3e64343eaef40477b339c36eed511bf94d4`；
- pyhip提交：`0948eab4d73789046d40c87d62d18c1a290d6f61`，测试时工作树干净；
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

三个正式profile进程开始时GPU均为`busy=0`、DPM=`auto`，但10个KFD context驻留约
153.3 GiB VRAM，`amd-smi`报告其`cu_occupancy=0`。正式测试前同一宿主任务曾达到
`busy=83-100%`，profiler按设计拒绝启动；本轮等待其回到上述低占用窗口后才开始。严格
preflight默认会拒绝resident context；本次
使用显式`ATTN_PROFILE_ALLOW_RESIDENT_PROCESSES=1`和200000 MiB初始VRAM阈值放行，并把
完整进程列表写入schema-v2 JSON。

每个profile进程结束快照均为DPM=`auto`、`gpu_busy_percent=0-1`，11个context、约
157.2-158.0 GiB VRAM。启动快照没有并发外部CU活动，但profiler没有在dispatch期间持续
采样逐进程CU占用；高驻留VRAM环境也与srdc-52及此前低驻留重测明显不同，结果不应解释为
纯kernel单变量消融。

## 数据与复现

测试产物：

```text
/tmp/h3-reproduction-20260813-0948eab
```

- `profile-flydsl-4w8w.json/.log`；
- `profile-triton-mi308.json/.log`；
- `profile-mi300.json/.log`；
- `SHA256SUMS`。

全部profile为schema v2，每项70条dispatch；所有event时间、TFLOPS和sensor count均为正。
`sha256sum -c SHA256SUMS`全部通过。

当前分支profile入口：

```bash
# new flydsl 4wave + 8wave
HIP_VISIBLE_DEVICES=4 \
ATTN_PROFILE_IMPLS=new_flydsl_4wave,new_flydsl_8wave \
ATTN_PROFILE_WARMUP=3 ATTN_PROFILE_ITERS=70 \
ATTN_PROFILE_SENSOR_INTERVAL_MS=1 \
ATTN_PROFILE_ALLOW_RESIDENT_PROCESSES=1 \
ATTN_PROFILE_MAX_INITIAL_VRAM_MIB=200000 \
ATTN_PROFILE_OUTPUT=/tmp/h3-reproduction-20260813-0948eab/profile-flydsl-4w8w.json \
FLYDSL_RUNTIME_ENABLE_CACHE=0 \
.venv/bin/python -B tests/flydsl/flash_attn_api/profile_h3_reproduction_current.py

# Triton + MI308 ASM
HIP_VISIBLE_DEVICES=4 \
ATTN_PROFILE_IMPLS=triton,asm_mi308 \
ATTN_PROFILE_WARMUP=3 ATTN_PROFILE_ITERS=70 \
ATTN_PROFILE_SENSOR_INTERVAL_MS=1 \
ATTN_PROFILE_ALLOW_RESIDENT_PROCESSES=1 \
ATTN_PROFILE_MAX_INITIAL_VRAM_MIB=200000 \
ATTN_PROFILE_OUTPUT=/tmp/h3-reproduction-20260813-0948eab/profile-triton-mi308.json \
.venv/bin/python -B tests/flydsl/flash_attn_api/profile_h3_reproduction_current.py

# MI300 ASM必须使用独立进程
HIP_VISIBLE_DEVICES=4 \
ATTN_PROFILE_IMPLS=asm_mi300 \
ATTN_PROFILE_WARMUP=3 ATTN_PROFILE_ITERS=70 \
ATTN_PROFILE_SENSOR_INTERVAL_MS=1 \
ATTN_PROFILE_ALLOW_RESIDENT_PROCESSES=1 \
ATTN_PROFILE_MAX_INITIAL_VRAM_MIB=200000 \
ATTN_PROFILE_OUTPUT=/tmp/h3-reproduction-20260813-0948eab/profile-mi300.json \
.venv/bin/python -B tests/flydsl/flash_attn_api/profile_h3_reproduction_current.py
```

当GPU 4完全无KFD context且初始VRAM低于1 GiB时，应删除两个`ATTN_PROFILE_ALLOW_*`
覆盖，恢复与上游报告完全一致的严格preflight。