# GR read：接入、性能与框架预热

公共入口为 `gr_read(x, packed_down, packed_up, output=None)`。它按输入实际行数选择 T1–32 decode 或 T33 起的 prefill，内部申请中间工作区并缓存编译产物。输入不额外补行，非空计算保持 Down＋Up 两个 GPU kernel。

固定维度为 C=4、H=2560、R=320、K=10240；当前实现与性能配置在 ROCm gfx942、MI308X / 80CU 上验证。其他 ROCm 架构发出 warning 后继续尝试执行；实际编译、执行或精度错误正常报错。`GRReadPrefill` 与 `GRReadDecode` 保留供分阶段检查及已有调用兼容；业务调用使用统一函数。

## 最小接入

本轮函数接口验证使用 **FlyDSL 0.3.1、ROCm PyTorch 2.12.0**。原 kernel 的历史验证使用过 FlyDSL 0.3.2。先在已安装 ROCm PyTorch/FlyDSL 的环境中，从仓库根目录运行 `python3 -m pip install -e .`。

```python
from pyhip.ops.gr_read import gr_read, prepare_weights

# 模型加载 / 准备阶段，每对原始 BF16 权重只做一次。
# w_down: [320, 10240]，w_up: [10240, 320]，同一张 ROCm GPU。
packed_down, packed_up = prepare_weights(w_down, w_up)

# x 是已归一化、连续且 16B 对齐的 BF16 [T,10240]。
# 首次调用编译当前 T/配置；后续相同配置复用编译结果。
y = gr_read(x, packed_down, packed_up)

# 调用方已有 BF16 [T,2560] 输出时可直接写入，返回值就是 out。
y = gr_read(x, packed_down, packed_up, output=out)
assert y is out
```

每层每个 device 保留一份 packed 权重，所有 T 共用其指针。Down 为 N16/K32 preshuffle；Up 先按 H64 做 `reshape(4,40,2,4,2,4,320).permute(1,0,2,4,3,5,6)`，再做同一 N16/K32 preshuffle。这与现有 decode 的权重布局一致。将来接入 SGLang 时仍在加载阶段做一次，不能放进每次 forward。

函数默认每次申请独立输出，仍持有的旧输出不会被下一次调用覆盖。中间 P 由 PyHIP 管理：decode 为紧凑 FP32 `[4,T,320]`，prefill 为 BF16 `[T,320]`。缓存只保留编译产物，不持有调用方输入、权重或工作区。并发 stream 可共享只读权重和代码，工作区由 Torch 分配器隔离；调用方遵循正常的 Torch stream 依赖规则。

Graph capture 前，用同一函数预热需要的 shape/配置和 device；缓存未命中时在 capture 内调用会明确报错。Graph replay 复用捕获时的地址，输出随 replay 更新。框架传入 X24、实际只有 17 行有效时，编译/执行仍是 T24，P 始终为 `[4,24,320]`；有效前缀由框架消费。普通 X17 则只编译并执行 T17，不自动编译 1..17。

外部 output 必须与输入 T/device 匹配、连续且 16B 对齐，不能与 X 或 packed 权重重叠。T=0 返回空输出且不编译、不 launch。接口用于推理，梯度输入需在 `torch.no_grad()` / `inference_mode()` 中使用。

诊断时仍可构造 `GRReadPrefill(T, packed_down, packed_up, partial=p, output=y)`，通过 `run_down(x)` / `run_up(x)` 查看 P 或分阶段计时。准备对象持有的 Y 会被下一次对象调用覆盖；这一旧对象语义与函数的默认输出生命周期不同。

## 性能数据

以下为 **2026-09-29 迁移后完整矩阵实测**：物理 GPU2，AMD Instinct MI308X / gfx942 / 80CU，PTL Enabled/VECTOR,F8；FlyDSL 0.3.1，ROCm PyTorch `2.12.0+rocm7.2.4.gitcf5ea6e.post2`，Triton `3.7.1`。默认 32 档 Decode 与 21 档 Prefill 均完成原正确性校验和性能采样。本轮检查阻止加载仓库 `tests/` 脚本及 SGLang，实际访问次数为 0。

Total 测量 `gr_read(..., output=out)`，P 由函数内部申请。Decode：100 对权重、图内两轮、每样本三次 replay、3×7 样本/阶段；Prefill：10 个 buffer、2 次预热、10 样本/阶段。两种测法分别列出，Speedup = 固定对照 Total / PyHIP Total。有效 TFLOPS 仅计算 GEMM 工作量。

本轮同时使用 `--output` 与 `--md`，记录了实际 P 地址；纯打印模式不安装地址观察器。数据代表本次硬件/软件与分配条件，不是跨机器性能保证。复现方式见下一节。被测 benchmark SHA256：`bc13256d9496b31a71c806e445e7295bdcbe4c2fb47587da151cedcf42fff5e3`。

### Decode（CUDA Graph）


Decode (CUDA Graph)

| T | Decode Down us | Decode Up us | Decode Total us | Frozen baseline backend | Frozen baseline Total us | Speedup |
|---:|---:|---:|---:|---|---:|---:|
| 1 | 4.823 | 6.529 | 11.225 | triton | 31.625 | 2.817x |
| 2 | 4.899 | 6.573 | 11.464 | triton | 31.927 | 2.785x |
| 3 | 4.947 | 6.658 | 11.580 | triton | 32.801 | 2.833x |
| 4 | 4.991 | 6.661 | 11.679 | triton | 32.979 | 2.824x |
| 5 | 5.009 | 6.680 | 11.729 | triton | 33.994 | 2.898x |
| 6 | 5.070 | 6.697 | 11.729 | triton | 34.085 | 2.906x |
| 7 | 5.138 | 6.840 | 11.736 | triton | 34.148 | 2.910x |
| 8 | 5.220 | 6.872 | 11.773 | triton | 34.256 | 2.910x |
| 9 | 5.226 | 7.630 | 12.346 | triton | 35.740 | 2.895x |
| 10 | 5.313 | 7.828 | 12.519 | triton | 35.763 | 2.857x |
| 11 | 5.412 | 7.829 | 12.660 | triton | 35.857 | 2.832x |
| 12 | 5.513 | 7.828 | 12.731 | triton | 35.867 | 2.817x |
| 13 | 5.521 | 7.829 | 12.780 | triton | 35.909 | 2.810x |
| 14 | 5.636 | 7.840 | 12.911 | triton | 35.906 | 2.781x |
| 15 | 5.762 | 7.890 | 13.108 | triton | 35.992 | 2.746x |
| 16 | 5.914 | 7.832 | 13.272 | triton | 36.004 | 2.713x |
| 17 | 6.882 | 9.781 | 16.289 | torch.compile | 36.689 | 2.252x |
| 18 | 7.061 | 9.781 | 16.491 | torch.compile | 32.737 | 1.985x |
| 19 | 7.227 | 9.783 | 16.603 | torch.compile | 32.729 | 1.971x |
| 20 | 7.384 | 9.782 | 16.824 | torch.compile | 32.790 | 1.949x |
| 21 | 7.411 | 9.782 | 16.858 | torch.compile | 32.864 | 1.949x |
| 22 | 7.579 | 9.785 | 16.985 | torch.compile | 32.933 | 1.939x |
| 23 | 7.829 | 9.788 | 17.338 | torch.compile | 32.921 | 1.899x |
| 24 | 8.130 | 9.787 | 17.629 | torch.compile | 32.981 | 1.871x |
| 25 | 8.337 | 9.790 | 17.898 | torch.compile | 27.154 | 1.517x |
| 26 | 8.891 | 9.811 | 18.493 | torch.compile | 27.216 | 1.472x |
| 27 | 9.055 | 9.838 | 18.624 | torch.compile | 27.164 | 1.459x |
| 28 | 9.286 | 9.850 | 18.851 | torch.compile | 27.316 | 1.449x |
| 29 | 9.298 | 9.782 | 18.859 | torch.compile | 27.316 | 1.448x |
| 30 | 9.452 | 9.782 | 19.007 | torch.compile | 27.355 | 1.439x |
| 31 | 9.585 | 9.782 | 19.159 | torch.compile | 27.337 | 1.427x |
| 32 | 9.782 | 9.779 | 19.337 | torch.compile | 26.783 | 1.385x |

### Prefill（eager cudaPerf）


| Batch | Down | Up | Down us / TFLOPS | Up us / TFLOPS | Total us / TFLOPS | Frozen Torch compile us / TFLOPS | Speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 33 | M16/W2/N10/BK1024 | M64/N40 | 15.840 / 13.653 | 16.600 / 13.028 | 31.280 / 13.828 | 46.500 / 9.302 | 1.487x |
| 64 | M16/W2/N10/BK1024 | M64/N40 | 16.120 / 26.019 | 17.300 / 24.245 | 32.320 / 25.955 | 32.240 / 26.019 | 0.998x |
| 128 | M16/W2/N10/BK1024 | M64/N40 | 16.300 / 51.464 | 17.460 / 48.045 | 32.820 / 51.119 | 42.140 / 39.813 | 1.284x |
| 256 | M16/W2/N10/BK512 | M128/N40 | 24.300 / 69.041 | 20.601 / 81.441 | 43.040 / 77.961 | 53.300 / 62.953 | 1.238x |
| 512 | M32/W4/N5/BK512 | M256/N40 | 29.700 / 112.978 | 26.000 / 129.056 | 54.060 / 124.137 | 83.400 / 80.466 | 1.543x |
| 1024 | M32/W4/N5/BK128 | M256/N20 | 58.241 / 115.227 | 39.720 / 168.953 | 94.121 / 142.601 | 161.301 / 83.209 | 1.714x |
| 2048 | M32/W4/N5/BK128 | M256/N10 | 107.141 / 125.273 | 66.700 / 201.224 | 168.201 / 159.592 | 279.062 / 96.192 | 1.659x |
| 4096 | M64/W4/N1/BK64 | M256/N10 | 155.321 / 172.826 | 130.481 / 205.728 | 279.822 / 191.862 | 494.422 / 108.585 | 1.767x |
| 8192 | M64/W4/N1/BK64 | M256/N2 | 282.241 / 190.217 | 276.961 / 193.844 | 550.263 / 195.132 | 945.785 / 113.529 | 1.719x |
| 10240 | M64/W4/N1/BK64 | M256/N2 | 291.161 / 230.487 | 283.821 / 236.447 | 560.103 / 239.630 | 1752.370 / 76.592 | 3.129x |
| 12288 | M64/W4/N1/BK64 | M256/N8 | 422.542 / 190.586 | 386.402 / 208.412 | 794.725 / 202.663 | 1898.710 / 84.827 | 2.389x |
| 16384 | M64/W4/N1/BK64 | M256/N8 | 555.303 / 193.361 | 534.843 / 200.758 | 1079.666 / 198.903 | 3065.977 / 70.042 | 2.840x |
| 20480 | M64/W4/N1/BK64 | M256/N2 | 565.383 / 237.393 | 560.023 / 239.664 | 1111.346 / 241.541 | 3389.418 / 79.198 | 3.050x |
| 24576 | M64/W4/N1/BK64 | M256/N4 | 698.844 / 230.468 | 718.364 / 224.206 | 1413.868 / 227.831 | 3713.560 / 86.742 | 2.627x |
| 28672 | M64/W4/N1/BK64 | M256/N2 | 833.425 / 225.461 | 829.565 / 226.510 | 1657.189 / 226.775 | 4869.726 / 77.173 | 2.939x |
| 30720 | M64/W4/N1/BK64 | M256/N2 | 839.545 / 239.805 | 833.964 / 241.409 | 1674.969 / 240.394 | 5045.767 / 79.800 | 3.012x |
| 32768 | M64/W4/N1/BK64 | M256/N8 | 968.486 / 221.736 | 995.385 / 215.744 | 1963.511 / 218.739 | 5214.669 / 82.363 | 2.656x |
| 36864 | M64/W4/N1/BK64 | M256/N2 | 1100.446 / 219.540 | 1098.306 / 219.968 | 2203.952 / 219.235 | 6348.074 / 76.115 | 2.880x |
| 49152 | M64/W4/N1/BK64 | M256/N2 | 1382.468 / 233.006 | 1380.947 / 233.262 | 2772.935 / 232.333 | 7356.820 / 87.571 | 2.653x |
| 61440 | M64/W4/N1/BK64 | M256/N2 | 1663.130 / 242.106 | 1683.569 / 239.166 | 3386.238 / 237.817 | 6677.156 / 120.606 | 1.972x |
| 65536 | M64/W4/N1/BK64 | M256/N4 | 1795.910 / 239.153 | 1885.570 / 227.781 | 3718.020 / 231.035 | 7284.519 / 117.920 | 1.959x |
Speedup = Torch compile Total / PyHIP Total; eager cudaPerf, full-row calls, no CUDA Graph.

## 运行 benchmark 与生成报告

在仓库根目录、安装好 PyHIP / ROCm PyTorch / FlyDSL 的环境中运行。脚本不依赖 SGLang 安装或 `tests/` 脚本；输入、参考与精度检查来自 `pyhip.testing.gr_read`。每个被计时的配置先完成必要的正确性校验。当前使用固定经验选型，无运行时 autotune；本入口不额外运行 tuner。

```bash
# 默认完整矩阵，控制台打印 Decode 和 Prefill 性能表。
python3 benchmarks/gr_read/bench_gr_read_compare.py --gpu 2

# 选择一个阶段或若干行数。
python3 benchmarks/gr_read/bench_gr_read_compare.py --gpu 2 --phase decode --rows 1 12 17 24 32
python3 benchmarks/gr_read/bench_gr_read_compare.py --gpu 2 --phase prefill --rows 33 512 2049 4096

# 生成可分享的 Markdown 报告；单独 --md 不启用 P 地址观察器。
python3 benchmarks/gr_read/bench_gr_read_compare.py --gpu 2 --md /tmp/gr_read_report.md

# 同时保存原始样本和实际 P 地址。
python3 benchmarks/gr_read/bench_gr_read_compare.py --gpu 2 \
  --output /tmp/gr_read_samples.jsonl --md /tmp/gr_read_report_with_samples.md

# 跳过固定性能对照，保留原精度检查。
python3 benchmarks/gr_read/bench_gr_read_compare.py --gpu 2 --no-baselines
```

`--output` 和 `--md` 都要求新的文件路径，不覆盖已有记录。Markdown 报告包含运行时间、设备/软件环境、采样协议、性能表、对照来源和源码 SHA256；指定 `--output` 时附原始 JSONL 链接。只在整次启动时检查一次硬件，后续不反复检查空闲或重试。

原命令 `python3 tests/ops/gr_read/test_gr_read.py --gpu 2` 保留完整精度检查后再调用新 benchmark 的兼容行为；`--check-only` 和 pytest 只检查正确性。测试命令和数值边界见 [tests README](../../tests/ops/gr_read/readme.md)。

## 配置与框架预热

| 文件 | 职责 |
| --- | --- |
| [host.py](../../src/pyhip/ops/gr_read/flydsl/host.py) | `gr_read` 分派、编译缓存、内部工作区与当前 stream/device；保留旧准备对象和分阶段调用 |
| [common.py](../../src/pyhip/ops/gr_read/flydsl/common.py) | 公共权重预处理入口、维度、固定配置选择 |
| [down.py](../../src/pyhip/ops/gr_read/flydsl/down.py) | Prefill Down 与调优后的 tile/BK/swizzle 参数，以及 decode split-K Down |
| [up.py](../../src/pyhip/ops/gr_read/flydsl/up.py) | Prefill M256/M64/M128 Up 编译期特化，以及 decode Up |
| [helpers.py](../../src/pyhip/ops/gr_read/flydsl/helpers.py) | 原数值与底层辅助函数 |

Decode 对全部 T1–32 按真实 T 编译和缓存，不采用固定七档表。Prefill 保留按真实 T 的现有配置选择，相同配置跨 T 复用编译产物；没有运行时 autotune。
本轮曾对 prefill 的 pow2 配置选型做同址测试，T2049 回退 34.67%，部分大 batch 回退约 5.2%，因此保留原选型。配置缓存无需以牺牲真实行数的选型为代价。

80CU 上的小 prefill 配置如下：

| T | Down M / waves / N splits / BK | Up M / N splits | Down swizzle shift |
| --- | --- | --- | --- |
| 33–128 | 16 / 2 / 10 / 1024 | 64 / 40 | 7 |
| 129–256 | 16 / 2 / 10 / 512 | 128 / 40 | 6 |
| 257–512 | 32 / 4 / 5 / 512 | 256 / 40 | 6 |
| 其余 T，或其它 CU 数 | 原 prefill 配置选择 | 原 M256 Up | 3 |

所有非空 decode/prefill 调用都通过一个已编译 host 入口，依次提交 Down、Up 两个 GPU kernel。Host 提交方式不再按 T512 或 CU 数分支；上表中的算法配置选择保持独立。单独的 Down/Up 入口用于分阶段检查和计时，其它 CU 数的配置没有在本机做性能验收。

测试辅助函数集中在 `pyhip.testing.gr_read`；业务接入使用公共 `gr_read` 函数。

### 框架侧编译预热：MI308X / 80CU 的建议档位

**以下建议对应当前 MI308X、gfx942、80CU、BF16 固定维度版本，覆盖本地输入 T33–65536。** `common.py` 中的配置选择使用该机器上实测校准的 CTA 轮数成本和 tile 参数，是经验公式；中间 T 通过公式选择配置，不表示每个 T 都已实测最优。更换 GPU 型号、可用 CU 数或机器配置后，应重新做正确性与性能实测，再确认选型及预热档位。ROCm、Torch/FlyDSL 版本变化也应复验，不能直接把这张表当作跨机器的最优配置。

框架以最终传给 `gr_read` 的 `x.shape[0]` 为准安排预热：decode 的 T1–32 按实际需要的行数分别预热；prefill 按配置组合复用编译产物。当前 80CU 选型在 T33–65536 内共有 **11 种 Down/Up 配置组合**，可在加载权重后、Graph capture 前，选择每组一个代表 T 调用 `gr_read()`：

| 建议预热 T | Down M / waves / N splits / BK | Up M / N splits |
| ---: | --- | --- |
| 128 | 16 / 2 / 10 / 1024 | 64 / 40 |
| 256 | 16 / 2 / 10 / 512 | 128 / 40 |
| 512 | 32 / 4 / 5 / 512 | 256 / 40 |
| 1024 | 32 / 4 / 5 / 128 | 256 / 20 |
| 2048 | 32 / 4 / 5 / 128 | 256 / 10 |
| 2560 | 64 / 4 / 2 / 64 | 256 / 8 |
| 3072 | 64 / 4 / 1 / 64 | 256 / 20 |
| 4096 | 64 / 4 / 1 / 64 | 256 / 10 |
| 5120 | 64 / 4 / 1 / 64 | 256 / 4 |
| 7680 | 64 / 4 / 1 / 64 | 256 / 8 |
| 10240 | 64 / 4 / 1 / 64 | 256 / 2 |

这些是**编译预热代表行数**，不要求框架使用同一组 Graph 档位，也不要求输入补齐到这些行数。例如预热 T2560 后，T2049–2560 共用该编译结果；T4096 属于另一组配置。大 T 的配置可能在多个不连续区间重复出现，上表不是按相邻行划分的 T 区间表。

只预热 64、128、256……65536 这些 pow2 行数会覆盖 9 种组合，漏掉 T2049–2560 和 T2561–3072 对应的两组；已有 pow2 预热流程可额外加入 **2560、3072**。如需按较小的最大 T 缩减预热范围，应按 `select_prefill_config(T, compute_units)` 去重并重新选取范围内的代表，不能简单删除表中大于最大 T 的行，否则可能漏掉边界处已经需要的配置。

下面示例只预热算子，使用模型加载阶段已经通过公共 `prepare_weights()` 得到的一对权重，不运行整个模型：

```python
import torch
from pyhip.ops.gr_read import gr_read

# 仅适用于上述 MI308X / 80CU 版本；覆盖 prefill T33–65536。
prefill_warmup_rows = [
    128, 256, 512, 1024, 2048,
    2560, 3072, 4096, 5120, 7680, 10240,
]
with torch.cuda.device(packed_down.device), torch.inference_mode():
    for rows in prefill_warmup_rows:
        x = torch.zeros((rows, 10240), dtype=torch.bfloat16,
                        device=packed_down.device)
        gr_read(x, packed_down, packed_up)
    torch.cuda.synchronize()
```

每个 worker 进程、每个 device 分别预热。同一进程/device 内，相同配置的代码跨层权重共享，用一对权重即可完成这些配置的编译；各层仍各自准备 packed 权重。P 和默认 Y 由 PyHIP 分配，预热不替代框架正常的逐 Graph shape 预热与 capture，捕获时仍使用该图实际的 tensor shape 和地址。

仅使用 Graph 路径时，可以跟随框架实际 capture shape 逐档预热；相同配置不会重复编译。若 eager 或回退路径也会出现任意 T，并希望避免服务期间首次编译，则应覆盖上述完整配置集合。例如框架把 live=2049 放进 X2304 的图，GR read 选择 T2304 的配置；放进 X4096 的图则选择 T4096 的配置。未来若封装按最大 T 自动枚举配置的预热辅助接口，应由 PyHIP 管理选型和去重，框架传入实际范围即可；当前通过上面的 `gr_read()` 调用完成预热。

## 测量与固定对照

默认使用 [baselines.py](baselines.py) 中固定的 Torch compile / Triton 对照，环境不需要安装 SGLang，也不会探测或读取外部 checkout。`--sglang-root` 和 `--no-sglang` 已移除；`--no-baselines` 跳过 decode 和 prefill 两阶段的性能对照，相关表格显示 `—`，独立精度参考仍运行。对照所需的 PyTorch/Triton 依赖缺失会正常报错，不静默更换实现。

`--check-only` 或单独指定 `--scope down/up/total` 时只做精度检查；pytest 也只检查正确性。精度失败会在进入性能阶段前停止。性能采样继续复用 `bench_gr_read_compare.py`：decode 精度默认为 2 对权重 / seed303，性能仍为 100 对权重 / seed707；`test_gr_read.py` 的 `--decode-weights` / `--decode-seed` 只作用于精度。需要调整性能采样参数时使用单独的 benchmark 入口。显式提供 `--output` 时，同一 JSONL 用 `stage=accuracy/performance` 区分默认流程的两部分记录。两个入口都仅在提供 `--output` 时采集 Total 内部 P 的地址；普通 `python3 tests/ops/gr_read/test_gr_read.py --gpu 2` 不安装这些观察器，继续原来的纯打印测量流程。

包含性能测量的运行只在**整次启动时检查一次硬件**，位于正确性测试、GPU buffer 准备与预热之前；decode/prefill 切换、各 batch 和结束时不再查询或判断空闲，避免 `rocm-smi` 滞后的利用率包含本次测试自身的活动。入口默认静置 2 秒，独立 benchmark 的 `--settle-seconds` 只调整这一次等待。GPU use≤5%、VRAM≤20% 仍为入口条件，失败立即停止，不重试；纯精度运行不做硬件检查。PTL Enabled/VECTOR,F8 作为已验证环境的提示，状态或格式不同只发 warning；查询失败仍正常报错。显式提供 `--output` 时保存一条 `phase=setup`、`event_phase=entry`、`hardware_policy=entry_only` 的原始快照，prefill result 不再包含逐 batch 的 `hardware_before*` / `hardware_after` 字段。入口快照只描述启动状态，不能证明整个测量期间独占 GPU。

Benchmark 的 prefill 表分别测 Down、Up、Total 和 Torch compile 完整调用。使用原 `cudaPerf`，默认 10 组独立 buffer、各阶段 2 次预热、10 个样本取中位数；保留逐阶段采样顺序，Torch compile 接在 Total 后面。这张表使用各阶段整组样本的中位数，不是交错 A/B 测量。Total 直接调用 `gr_read(..., output=预分配的Y)`，P 由函数内部申请；Down/Up 单阶段仍使用准备对象的 P。两者工作区地址可能不同，不能用阶段中位数之和代替 Total。权重打包、对象构造、首次编译、参考与校验不计时。没有修改设备设置或剔除长尾。

Prefill 的地址表用 `P_stages` 标识 `reader.partial`。启用 `--output` 后，每条 `scope=total` 的 `sample` 另存当次原生申请的 BF16 `[T,320]` `P_total`，包含 pointer、storage base/offset、mod256/mod4096、shape、dtype，以及相对 X、packed Down/Up 权重、Y 的 `relative_bytes`；通过 `sample` 和 `buffer` 下标关联到这次真实计时调用，不以预热地址代替。观察器在原 `cudaPerf` 上下文外安装/卸载，调用内只收集地址整数，完整元数据和 JSON 在计时结束后生成，不保留 tensor/storage 引用。Eager 采样中的地址读取仍有少量 CPU 工作，导出模式不承诺零观察开销；环境记录的 `protocol.record_total_partials` 标明是否开启。纯打印模式不执行这部分采集。

表格新增 `Torch compile us / TFLOPS` 和 `Speedup`，其中 **Speedup = Torch compile Total / PyHIP Total**，大于 1 表示 PyHIP 更快；显式传入 `--output` 时，结果及全部样本写入 JSONL。默认显示阶段初始化、逐 batch 准备/检查/时延进度，最后打印两张完整表，不自动创建结果文件或目录。进度实时刷新，均在计时区间外；`--verbose` 可额外显示硬件和详细正确性信息。TFLOPS 两边都按 `4*T*10240*320` 的有效 GEMM 工作量计算。

两套性能对照集中在一个 `baselines.py` 文件中：Torch 部分保留原输出 Y 的公式和默认 `torch.compile` 选项，对完整 T 一次调用；decode/prefill 使用独立根包装函数保持编译缓存隔离。原按 1024 行分块、返回 P/Y 的正确性参考仍单独保留，不用于计时。

Triton 部分固定本轮已测的 SGLang checkout `1b5e190695300821eff9c393af5cc07a6df949a7`，包含 `8cf5501b6913f57a2e7c8dcee52b625fc8ab23c3` 的 MI308X/80CU 调优，不能称为当前 upstream main。Kernel、launcher、支持范围和 counter 重置逻辑保留，确定性模式改为显式参数，默认 False，不再导入框架配置。来源、原始 SHA256、修改说明放在文件头注释，Apache-2.0 许可证全文放在文件尾注释；本地辅助函数沿用仓库 MIT 许可证。`metadata()` 记录来源版本、当前单文件 SHA256 和 Torch/Triton 版本，无需额外 JSON 或目录。Triton counter 由同设备串行调用共享，基准不并发执行它；atomic 累加按原容差检查，不要求逐位确定性。

权重从同一组原始逻辑值出发：PyHIP 使用自己的 packed Down/Up；每个活动对照实例使用独立的原始连续 BF16 矩阵 `[320,10240]`、`[10240,320]`。两套对照均不接受 PyHIP 的 packed 布局。复制和 packing 均在计时外，X 保持共享；记录各自权重/P/Y 地址，检查所有实际计时输出。移入固定源码和权重副本后的数据作为新一轮基线，原始历史结果不改写。

默认 BF16 固定形状下，decode T1–16 使用 Triton persistent，T17–32 使用 Torch compile；prefill 使用 Torch compile。结果标为 Frozen baseline，并记录源码及 Torch/Triton 版本。测量从已归一化的 X 开始，不包含 RMSNorm、TP 通信或整个模型。

两个入口都支持 `--phase all/decode/prefill`，默认 all。指定一个 phase 时可用 `--rows` / `--batches`；同时运行两阶段时用 `--decode-rows` 和 `--prefill-rows`。Decode 默认覆盖全部 T1–32，benchmark 的 Down/Up/Total/Frozen baseline 分别捕获 Graph；prefill 的 Graph 开关与采样方式保持原样。


### Decode 与 prefill 的测法不同

Decode 使用 CUDA Graph 测量，prefill 使用原 `cudaPerf` 普通调用，两者分别出表。以下为默认参数；每次 GR read 调用的 batch 都是该行的 T。

| 项目 | Decode | Prefill |
| --- | --- | --- |
| 执行方式 | CUDA Graph | 普通调用，无 CUDA Graph |
| 工作集 | 100 组独立权重和对应输入实例 | 10 组独立 buffer |
| 一个图的内容 | `calls + calls`：100 个实例执行两遍，共 200 次调用 | 没有图 |
| 一个计时样本 | Event 包住 3 次 graph replay，共 600 次调用 | 一个 `with cudaPerf` 包住一次调用 |
| 换算为每次调用的 µs | `elapsed_ms * 1000 / (100 * 2 * 3)` | `perf.latencies[-1] * 1e6`，由秒转换为 µs |
| 显式采样前预热 | 每轮先 replay 图 3 次，预热不计时 | 每个 scope 普通调用 2 次，预热不计时 |
| 最终统计 | 3 轮 × 7＝21 个均摊时延样本，取中位数 | 10 个单次调用时延样本，取中位数 |

Decode 公式中的 `1000` 是毫秒转微秒，`100` 是独立测试实例数，`2` 来自图中的 `calls + calls`，`3` 来自计时区间的三次 replay。实际代码使用实例数 `N * 2 * 3` 作分母，100 是默认的 N。对于 Total，一次调用已经包含 Down＋Up；外层 3×7 只负责收集 21 个样本，再取中位数。

图构造前的预执行、packing、首次编译和正确性检查均在计时区外。以上差异沿用各自原有基准；同一张表里的 PyHIP 与固定基线 使用该表对应的计时方式。



### 实际中间工作区的地址记录

Decode 的 `addresses` JSON 记录同时保留分阶段和 Total Graph 的真实工作区地址：`buffers[i].P_stages` 是 Down/Up 单独测量使用的准备对象 P；`total_calls` 则逐次记录 Total Graph 捕获期间 `gr_read()` 原生申请的内部 P。默认 100 组 buffer、图内两轮，共 200 条 Total 调用记录：

- `call_index` 为图内调用顺序（从 0 开始），`pass_index` 为图内第 0/1 轮，`buffer` 对应 `buffers` 数组下标。
- `P_total` 包含实际指针、storage base/offset、地址模 256/4096、物理 shape 和 dtype；`logical_shape` 为 decode 的 `[4,T,320]`。
- `relative_bytes` 记录有符号字节差 `P_minus_X`、`P_minus_WD_packed`、`P_minus_WU_packed`、`P_minus_Y`，便于比较实际工作区布局。

每次捕获调用分别记录，允许原生分配器复用地址，不要求 200 个不同指针。仅在启用 `--output` 时，采集在 Total capture 的该次调用内观察原生分配，只保存数值，不保存 tensor/storage 引用，不改变 P 分配或增加 GPU 操作；预热与计时 replay 不采集。Replay 使用捕获时的地址。纯打印模式直接捕获原调用，不启用地址观察器。旧 JSON 未记录的 Total P 地址无法通过 `P_stages` 推导，不事后补写历史记录。

固定对照在本机 gfx942、显式非确定性模式下，T1–16 为 Triton persistent，T17–32 为 Torch compile；两边均使用原 decode Graph 协议。表中列出实际后端，独立于 prefill 表。X 和逻辑权重值相同，物理权重 buffer 按各自布局独立准备。Triton counter 在同一设备的串行调用间共享，测试不并发执行这些基线。


<details>
<summary>实现与历史调优方法</summary>


1. **先固定数值边界，再减少算术。** Down完整K10240 FP32累加后先舍入BF16，再做FP32缩放和SiLU，写BF16 P。Up完整R320归约、直接FP32 logits、stream0→1→2→3顺序FMA，最后乘0.25并使用既有整数BF16 helper。浮点FMA加偏置不能替代整数位模式舍入；“容差通过”与“逐位等价”分开验收。[数值helpers](../../src/pyhip/ops/gr_read/flydsl/helpers.py)

2. **按实际CTA轮数独立选择Down／Up的N分片。** Down N1/N2对应320／160列，Up选择N2/N4/N8；比较$\lceil B_MN/U\rceil c_N$而不是只看单CTA工作量。Down用$B_D=\lceil T/64\rceil$，Up用$B_U=\lceil T/256\rceil$；更多分片有利于小batch填满CU，却可能增加大batch轮数和重复读取。成本是历史校准模型，不是跨设备最优保证。[选择规则](../../src/pyhip/ops/gr_read/flydsl/common.py)

3. **让权重顺序匹配归约顺序和输出lane。** Up先将权重排列为`[H64 tile, stream, H32 half, ...]`，相邻8个H32 packet完成同一H64的四路归约；内层重排使同一lane取得连续8个BF16，适配X和16B Y store。随后N16/K32 preshuffle负责MFMA输入打包，不能混淆两层排列。变更前先用CPU标签证明索引双射，再同步修改消费地址。[准备与参考辅助函数](../../src/pyhip/testing/gr_read.py)

4. **交错当前MFMA与上一packet的后处理。** 在Compute(q)计算当前logits，同时推进q−1的scale／exp／rcp／FMA／BF16打包，拆开后处理自身依赖链。当前每10条MFMA推进两个旧元素，SFU独占间隔，普通间隔控制VALU数量；实际收益取决于机器调度和生存期，不是让指令间隔形式上均匀。保留FIRST／LOOP／LAST和drain的完整覆盖。[Up流水](../../src/pyhip/ops/gr_read/flydsl/up.py)

5. **联合调整预取深度、寄存器占用与等待。** 当前W2在Memory(q)发布W(q+1)、预读W(q+2)，少保留一个W搬运包；没有减少总W字节。任何预取或发射顺序改动都要重算W/X/Y的VMEM完成账本，检查LDS覆写和实际ISA。4＋4wave错相必须首尾闭合，barrier不能替代完成等待，VGPR下降也不自动提高占用率。

6. **按缓存行组织X协作读取，再恢复计算布局。** 同行8个lane各读16B形成X128，覆盖H64并供相邻两个H32消费；通过每wave3KiB LDS恢复MFMA布局。两个20KiB W槽＋24KiB X区共64KiB。X/P读和Y写当前使用NT、W保持default；合并请求能减少重复取读，但增加LDS工作，不能只凭HBM字节下降宣称加速。

7. **64位tile基址＋局部buffer边界，避免Host分块和P padding。** CTA入口先以64位元素偏移重设X/P/Y，descriptor只覆盖当前tile有效行。Down N1输出范围为`valid_rows*R*2`；N2保留stride R，范围为`((valid_rows-1)*R+BN)*2`，valid0为0。二维tensor避免展平动态shape超i32；rows运行时传入，工厂仅按N分片缓存。正常`flyc.compile`会执行一次，必须提供足量张量。[Down尾块处理](../../src/pyhip/ops/gr_read/flydsl/down.py)

8. **把性能条件和归因口径固定下来。** 正确性guard view与性能原生allocation分开，记录X/P/Y相对地址；同一ELF也会对地址偏移敏感。保留长尾，用同址交错对照确认小收益。ATT只分析共同稳态窗口，区分issue-stall和completion-wait，不把跨wave stall求和当墙钟，也不把MFMA union×roof模型TFLOPS当实测。逻辑M-major或swizzle不保证物理XCD归属，需单独验证。

可复用流程已整理为项目skills：
- [grread-bf16-prefill](../../.github/skills/grread-bf16-prefill/SKILL.md)：数值边界、权重／lane布局、N分片、X128/W2流水、精确尾块。
- [gpu-benchmark-validation](../../.github/skills/gpu-benchmark-validation/SKILL.md)：空闲GPU与PTL、原计时器和地址条件、原始样本、ATT/PMC归因边界。



</details>
