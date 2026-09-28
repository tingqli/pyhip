# GR read prefill / decode：正式接口与接入说明

公共入口为 `gr_read(x, packed_down, packed_up, output=None)`。它按输入实际行数选择 T1–32 decode 或 T33 起的 prefill，内部申请中间工作区并缓存编译产物。输入不额外补行，非空计算保持 Down＋Up 两个 GPU kernel。

固定维度为 C=4、H=2560、R=320、K=10240；当前实现与性能配置在 ROCm gfx942、MI308X / 80CU 上验证。其他 ROCm 架构发出 warning 后继续尝试执行；实际编译、执行或精度错误正常报错。`GRReadPrefill` 与 `GRReadDecode` 保留供分阶段检查及已有调用兼容；业务调用使用统一函数。

## 1. 安装与最小接入

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

## 2. 正式源码与分派

| 文件 | 职责 |
| --- | --- |
| [host.py](../../../src/pyhip/ops/gr_read/flydsl/host.py) | `gr_read` 分派、编译缓存、内部工作区与当前 stream/device；保留旧准备对象和分阶段调用 |
| [common.py](../../../src/pyhip/ops/gr_read/flydsl/common.py) | 公共权重预处理入口、维度、固定配置选择 |
| [down.py](../../../src/pyhip/ops/gr_read/flydsl/down.py) | Prefill Down 与调优后的 tile/BK/swizzle 参数，以及 decode split-K Down |
| [up.py](../../../src/pyhip/ops/gr_read/flydsl/up.py) | Prefill M256/M64/M128 Up 编译期特化，以及 decode Up |
| [helpers.py](../../../src/pyhip/ops/gr_read/flydsl/helpers.py) | 原数值与底层辅助函数 |

Decode 对全部 T1–32 按真实 T 编译和缓存，不采用固定七档表。Prefill 保留按真实 T 的现有配置选择，相同配置跨 T 复用编译产物；没有运行时 autotune。
本轮曾对 prefill 的 pow2 配置选型做同址测试，T2049 回退 34.67%，部分大 batch 回退约 5.2%，因此保留原选型。配置缓存无需以牺牲真实行数的选型为代价。

80CU 上的小 prefill 配置如下：

| T | Down M / waves / N splits / BK | Up M / N splits | Down swizzle shift |
| --- | --- | --- | --- |
| 33–128 | 16 / 2 / 10 / 1024 | 64 / 40 | 7 |
| 129–256 | 16 / 2 / 10 / 512 | 128 / 40 | 6 |
| 257–512 | 32 / 4 / 5 / 512 | 256 / 40 | 6 |
| 其余 T，或其它 CU 数 | 原 prefill 配置选择 | 原 M256 Up | 3 |

T33–512 将两个 GPU launch 收在一个已编译 host 调用中，仍然是两个 GPU kernel。此处未捕获 CUDA Graph。范围外保留原两个 launcher；其它 CU 数的回退没有在本机做性能验收。

旧测试脚本的 `prepare_gr_read` / `run_gr_read` tuple 接口仍保留兼容；业务接入使用公共 `gr_read` 函数。

## 3. 数值边界与验收范围

Down 保持完整 K 的 FP32 累加 → BF16 舍入 → 转回 FP32 除 4、SiLU → BF16 P。Up 保持原 FP32 logits、四路顺序 FMA 及最终 BF16 舍入。这次调整 tile、调度和准备接口，未删除 SiLU 之前的 BF16 舍入边界。

2026-09-21 的调优及正式目录迁移分别验证了 100 对真实 HC 权重 × 30 档 T（48,64,…,512）× 初始/改变 X，共 6000 例 P/Y 逐位相同。X 仍为随机 BF16，未作端到端模型质量评估。正式入口的 50 项 pytest 全部通过，包含空输入、尾块、配置边界、工作区、stream 和双 GPU。

“逐位相同”指保留同事原 prefill 的结果，不表示解决已有参考误差。之前真实权重 T33/T512 的 400 例检查中，原 prefill 有 16 例 P、1 例 Y 未通过原 BF16 Torch reference 容差；本次没有放宽容差，也没有宣称修复这些舍入边界案例。

## 4. 正确性与性能入口

在仓库根目录直接运行 `test_gr_read.py`，默认使用 GPU0，先检查全部常用 batch 的精度，全部通过后再测性能，最后输出 Decode 和 Prefill 两张独立表。Decode 覆盖 T1–32；prefill 覆盖 T33/64/128/256/512 及原 1K–64K 矩阵，共 21 档。过程中实时显示进度，默认不创建结果文件。

```bash
# 默认：全部 batch 先 acc，再输出全部性能；GPU0。
python3 tests/ops/gr_read/test_gr_read.py

# 选择 GPU；或只检查精度，不进入性能门禁和计时。
python3 tests/ops/gr_read/test_gr_read.py --gpu 2
python3 tests/ops/gr_read/test_gr_read.py --gpu 2 --check-only

# 只测 PyHIP 性能；两阶段的固定性能对照均跳过，精度检查仍保留。
python3 tests/ops/gr_read/test_gr_read.py --gpu 2 --no-baselines

# Pytest 保留原数值、边界、共享权重和 device 回归检查。
HIP_VISIBLE_DEVICES=0,1 python3 -m pytest -q tests/ops/gr_read/test_gr_read.py

# 也可单独运行原 benchmark 入口，保留其内部输出校验。
python3 tests/ops/gr_read/bench_gr_read_compare.py --gpu 3

# 只测 prefill：33/64/128/256/512 及 1K–64K 共 21 档，无需 SGLang 安装。
python3 tests/ops/gr_read/bench_gr_read_compare.py --phase prefill --gpu 3

# 检查 prefill Up：先运行并校验 Down 的 P，再检查 Up；不做 Total 验证或性能计时。
python3 tests/ops/gr_read/test_gr_read.py --phase prefill --scope up --gpu 3 \
  --batches 33 48 64 128 129 256 257 512 513

# 仅在需要保留原始样本时显式指定输出文件。
python3 tests/ops/gr_read/bench_gr_read_compare.py --gpu 3 \
  --output /tmp/gr_read_compare.jsonl
```

默认使用 [baselines.py](baselines.py) 中固定的 Torch compile / Triton 对照，环境不需要安装 SGLang，也不会探测或读取外部 checkout。`--sglang-root` 和 `--no-sglang` 已移除；`--no-baselines` 跳过 decode 和 prefill 两阶段的性能对照，相关表格显示 `—`，独立精度参考仍运行。对照所需的 PyTorch/Triton 依赖缺失会正常报错，不静默更换实现。

`--check-only` 或单独指定 `--scope down/up/total` 时只做精度检查；pytest 也只检查正确性。精度失败会在进入性能阶段前停止。性能采样继续复用 `bench_gr_read_compare.py`：decode 精度默认为 2 对权重 / seed303，性能仍为 100 对权重 / seed707；`test_gr_read.py` 的 `--decode-weights` / `--decode-seed` 只作用于精度。需要调整性能采样参数时使用单独的 benchmark 入口。显式提供 `--output` 时，同一 JSONL 用 `stage=accuracy/performance` 区分默认流程的两部分记录。

包含性能测量的运行只在**整次启动时检查一次硬件**，位于正确性测试、GPU buffer 准备与预热之前；decode/prefill 切换、各 batch 和结束时不再查询或判断空闲，避免 `rocm-smi` 滞后的利用率包含本次测试自身的活动。入口默认静置 2 秒，独立 benchmark 的 `--settle-seconds` 只调整这一次等待。GPU use≤5%、VRAM≤20% 仍为入口条件，失败立即停止，不重试；纯精度运行不做硬件检查。PTL Enabled/VECTOR,F8 作为已验证环境的提示，状态或格式不同只发 warning；查询失败仍正常报错。显式提供 `--output` 时保存一条 `phase=setup`、`event_phase=entry`、`hardware_policy=entry_only` 的原始快照，prefill result 不再包含逐 batch 的 `hardware_before*` / `hardware_after` 字段。入口快照只描述启动状态，不能证明整个测量期间独占 GPU。

Benchmark 的 prefill 表分别测 Down、Up、Total 和 Torch compile 完整调用。使用原 `cudaPerf`，默认 10 组独立 buffer、各阶段 2 次预热、10 个样本取中位数；保留逐阶段采样顺序，Torch compile 接在 Total 后面。这张表使用各阶段整组样本的中位数，不是交错 A/B 测量。Total 直接调用 `gr_read(..., output=预分配的Y)`，P 由函数内部申请；Down/Up 单阶段仍使用准备对象的 P。两者工作区地址可能不同，不能用阶段中位数之和代替 Total。权重打包、对象构造、首次编译、参考与校验不计时。没有修改设备设置或剔除长尾。

表格新增 `Torch compile us / TFLOPS` 和 `Speedup`，其中 **Speedup = Torch compile Total / PyHIP Total**，大于 1 表示 PyHIP 更快；显式传入 `--output` 时，结果及全部样本写入 JSONL。默认显示阶段初始化、逐 batch 准备/检查/时延进度，最后打印两张完整表，不自动创建结果文件或目录。进度实时刷新，均在计时区间外；`--verbose` 可额外显示硬件和详细正确性信息。TFLOPS 两边都按 `4*T*10240*320` 的有效 GEMM 工作量计算。

两套性能对照集中在一个 `baselines.py` 文件中：Torch 部分保留原输出 Y 的公式和默认 `torch.compile` 选项，对完整 T 一次调用；decode/prefill 使用独立根包装函数保持编译缓存隔离。原按 1024 行分块、返回 P/Y 的正确性参考仍单独保留，不用于计时。

Triton 部分固定本轮已测的 SGLang checkout `1b5e190695300821eff9c393af5cc07a6df949a7`，包含 `8cf5501b6913f57a2e7c8dcee52b625fc8ab23c3` 的 MI308X/80CU 调优，不能称为当前 upstream main。Kernel、launcher、支持范围和 counter 重置逻辑保留，确定性模式改为显式参数，默认 False，不再导入框架配置。来源、原始 SHA256、修改说明放在文件头注释，Apache-2.0 许可证全文放在文件尾注释；本地辅助函数沿用仓库 MIT 许可证。`metadata()` 记录来源版本、当前单文件 SHA256 和 Torch/Triton 版本，无需额外 JSON 或目录。Triton counter 由同设备串行调用共享，基准不并发执行它；atomic 累加按原容差检查，不要求逐位确定性。

权重从同一组原始逻辑值出发：PyHIP 使用自己的 packed Down/Up；每个活动对照实例使用独立的原始连续 BF16 矩阵 `[320,10240]`、`[10240,320]`。两套对照均不接受 PyHIP 的 packed 布局。复制和 packing 均在计时外，X 保持共享；记录各自权重/P/Y 地址，检查所有实际计时输出。移入固定源码和权重副本后的数据作为新一轮基线，原始历史结果不改写。

默认 BF16 固定形状下，decode T1–16 使用 Triton persistent，T17–32 使用 Torch compile；prefill 使用 Torch compile。结果标为 Frozen baseline，并记录源码及 Torch/Triton 版本。测量从已归一化的 X 开始，不包含 RMSNorm、TP 通信或整个模型。

两个入口都支持 `--phase all/decode/prefill`，默认 all。指定一个 phase 时可用 `--rows` / `--batches`；同时运行两阶段时用 `--decode-rows` 和 `--prefill-rows`。Decode 默认覆盖全部 T1–32，benchmark 的 Down/Up/Total/Frozen baseline 分别捕获 Graph；prefill 的 Graph 开关与采样方式保持原样。

T48–512 的正式优化前后对照已在 30 档、普通调用下验证：选中的两段实现全部快于该版 SGLang torch.compile，范围为 1.06–4.19×。这是特定硬件和固定协议的测量，不能将少量样本 smoke 的时延替代完整数据。迁入公共接口后的同址配对复测中，30 档时延变化中位数为 −0.095%，最大增加 0.451%；完整报告和原始数据记录在本机 `qwen3.8-flash-next-doc` 的 45/46 号文档。

### 公共函数回归

`test_gr_read.py` 同时检查准备对象和 `gr_read()`。Decode 的 6528 次 Graph replay 每次包含两种完整调用，比较相同 T 的输出逐位一致；另有按需编译、跨权重缓存、默认输出生命周期、Graph 工作区污染/guard、空输入、契约检查及 stream/device 回归。
固定对照独立化后，全量 pytest 为 67 项，其中新增 11 项检查原始权重副本、框架独立性、Torch/Triton 精度和 counter 的 Graph replay 行为。

```bash
HIP_VISIBLE_DEVICES=2,3 python3 -m pytest -q tests/ops/gr_read/test_gr_read.py -k test_api
HIP_VISIBLE_DEVICES=2 python3 -m pytest -q tests/ops/gr_read/test_gr_read.py -k baseline
python3 tests/ops/gr_read/test_gr_read.py --gpu 2 --phase decode --rows 1 12 17 24 28 32 --check-only
```

## 5. 有效优化方法

1. **先固定数值边界，再减少算术。** Down完整K10240 FP32累加后先舍入BF16，再做FP32缩放和SiLU，写BF16 P。Up完整R320归约、直接FP32 logits、stream0→1→2→3顺序FMA，最后乘0.25并使用既有整数BF16 helper。浮点FMA加偏置不能替代整数位模式舍入；“容差通过”与“逐位等价”分开验收。[数值helpers](../../../src/pyhip/ops/gr_read/flydsl/helpers.py)

2. **按实际CTA轮数独立选择Down／Up的N分片。** Down N1/N2对应320／160列，Up选择N2/N4/N8；比较$\lceil B_MN/U\rceil c_N$而不是只看单CTA工作量。Down用$B_D=\lceil T/64\rceil$，Up用$B_U=\lceil T/256\rceil$；更多分片有利于小batch填满CU，却可能增加大batch轮数和重复读取。成本是历史校准模型，不是跨设备最优保证。[选择规则](../../../src/pyhip/ops/gr_read/flydsl/common.py)

3. **让权重顺序匹配归约顺序和输出lane。** Up先将权重排列为`[H64 tile, stream, H32 half, ...]`，相邻8个H32 packet完成同一H64的四路归约；内层重排使同一lane取得连续8个BF16，适配X和16B Y store。随后N16/K32 preshuffle负责MFMA输入打包，不能混淆两层排列。变更前先用CPU标签证明索引双射，再同步修改消费地址。[准备函数](test_gr_read.py#L110)

4. **交错当前MFMA与上一packet的后处理。** 在Compute(q)计算当前logits，同时推进q−1的scale／exp／rcp／FMA／BF16打包，拆开后处理自身依赖链。当前每10条MFMA推进两个旧元素，SFU独占间隔，普通间隔控制VALU数量；实际收益取决于机器调度和生存期，不是让指令间隔形式上均匀。保留FIRST／LOOP／LAST和drain的完整覆盖。[Up流水](../../../src/pyhip/ops/gr_read/flydsl/up.py)

5. **联合调整预取深度、寄存器占用与等待。** 当前W2在Memory(q)发布W(q+1)、预读W(q+2)，少保留一个W搬运包；没有减少总W字节。任何预取或发射顺序改动都要重算W/X/Y的VMEM完成账本，检查LDS覆写和实际ISA。4＋4wave错相必须首尾闭合，barrier不能替代完成等待，VGPR下降也不自动提高占用率。

6. **按缓存行组织X协作读取，再恢复计算布局。** 同行8个lane各读16B形成X128，覆盖H64并供相邻两个H32消费；通过每wave3KiB LDS恢复MFMA布局。两个20KiB W槽＋24KiB X区共64KiB。X/P读和Y写当前使用NT、W保持default；合并请求能减少重复取读，但增加LDS工作，不能只凭HBM字节下降宣称加速。

7. **64位tile基址＋局部buffer边界，避免Host分块和P padding。** CTA入口先以64位元素偏移重设X/P/Y，descriptor只覆盖当前tile有效行。Down N1输出范围为`valid_rows*R*2`；N2保留stride R，范围为`((valid_rows-1)*R+BN)*2`，valid0为0。二维tensor避免展平动态shape超i32；rows运行时传入，工厂仅按N分片缓存。正常`flyc.compile`会执行一次，必须提供足量张量。[Down尾块处理](../../../src/pyhip/ops/gr_read/flydsl/down.py)

8. **把性能条件和归因口径固定下来。** 正确性guard view与性能原生allocation分开，记录X/P/Y相对地址；同一ELF也会对地址偏移敏感。保留长尾，用同址交错对照确认小收益。ATT只分析共同稳态窗口，区分issue-stall和completion-wait，不把跨wave stall求和当墙钟，也不把MFMA union×roof模型TFLOPS当实测。逻辑M-major或swizzle不保证物理XCD归属，需单独验证。

可复用流程已整理为项目skills：
- [grread-bf16-prefill](../../../.github/skills/grread-bf16-prefill/SKILL.md)：数值边界、权重／lane布局、N分片、X128/W2流水、精确尾块。
- [gpu-benchmark-validation](../../../.github/skills/gpu-benchmark-validation/SKILL.md)：空闲GPU与PTL、原计时器和地址条件、原始样本、ATT/PMC归因边界。


## 6. Decode T1–32：独立正确性、原 Graph 基准和固定对照

实现已按方向收拢：[down.py](../../../src/pyhip/ops/gr_read/flydsl/down.py) 包含 prefill Down 和 decode split-K Down，[up.py](../../../src/pyhip/ops/gr_read/flydsl/up.py) 包含所有 prefill Up 和 decode Up。两后端共用 `common.prepare_weights()`；模型加载时 prepare 一次，同一份 packed Down/Up 权重传给统一 `gr_read()`，也可用于准备对象的分阶段检查。本轮函数接口验证使用 **FlyDSL 0.3.1**。

Decode 默认覆盖全部 T1–32，独立测量、独立出表。Prefill 继续普通调用，默认性能矩阵首档现为 T33；T33–64 全部使用之前验证的小 M 配置（同上文 T33–128 的配置，包括 T48），其余 prefill 配置保持。

Decode 的 P 现在紧凑存储为 FP32 `[4,T,320]`（对象内为 flat tensor），不再补齐到 16/32 行；`reader.padded_rows` 已移除，查看 P 用 `reader.partial.view(4, reader.rows, 320)`。Down 只写 `row < T`；Up 使用一个有界 buffer，将无效行的读取地址映射到整个 P 末尾之外，由硬件返回零，避免读进下一份 partial，也不需要额外的 EXEC 行掩码。M tile 和向上取整的 grid 保留，没有旧 padded-P 回退或 `skip_padding` 分段。

CUDA Graph 中 T 是图的固定输入行数。例如图档位 T24、实际 live=17 时，X/Y 仍是 24 行，P 固定为 `[4,24,320]`，地址和 split 间距不随 replay 改变；图内补齐的 `[17,24)` 行照常计算，框架只消费有效输出。普通调用 X17 时才准备 `[4,17,320]`。

在仓库根目录运行：

```bash
# Decode 正确性：原 2 对权重、全部 T1–32、6528 次完整调用 Graph replay，加分阶段检查。
python3 tests/ops/gr_read/test_gr_read.py --phase decode --gpu 2 --check-only

# 只排查 decode Down 或 Up。
python3 tests/ops/gr_read/test_gr_read.py --phase decode --scope down --rows 1 16 17 32 --gpu 2
python3 tests/ops/gr_read/test_gr_read.py --phase decode --scope up --rows 1 16 17 32 --gpu 2

# Decode 的固定基线 Graph 对照表：原 100 权重 / 3 rounds / 7 samples，默认不写文件。
python3 tests/ops/gr_read/bench_gr_read_compare.py \
  --phase decode --gpu 2
```

Decode 对照保持原 PR 中的 100 组独立权重工作集、buffer 准备、capture、计时器和检查流程：图内依次执行所有实例两轮，每个样本 replay 三次，Event 时间除以 600；3 轮 × 7 个样本全部保留并取中位数。Packing、编译和参考不计时。原独立 Total 基准的计时核心保留在 benchmark 文件的 `benchmark_decode_total()` 中供回归使用，默认不额外输出第三张表。

固定对照在本机 gfx942、显式非确定性模式下，T1–16 为 Triton persistent，T17–32 为 Torch compile；两边均使用原 decode Graph 协议。表中列出实际后端，独立于 prefill 表。X 和逻辑权重值相同，物理权重 buffer 按各自布局独立准备。Triton counter 在同一设备的串行调用间共享，测试不并发执行这些基线。

原 FP64 容差 `rtol=1e-2, atol=5e-3`、live rows 缩小/恢复、zero/stale/NaN 尾行、P/Y 预污染、输入及权重不变检查保留。P 不再有内部 padding，改为检查准确的紧凑 shape、P 两端 guard 及各 split 的数值；完整 Graph replay 继续检查 X/P/Y 两端 guard。

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


## 性能数据

2026-09-28，GPU 2 / MI308X / 80CU / PTL Enabled,VECTOR,F8，FlyDSL 0.3.1、ROCm PyTorch 2.12.0。下表来自屏蔽 SGLang 探测/导入后的完整精度与性能流程，使用 tests 内固定对照及独立的原始权重副本。Total 使用函数式入口并传入预分配 Y。

额外的同场对照区分了工作区布局影响：32 档 decode 编译产物均与原实现逐字节相同；相同 X/WD/WU/P/Y 地址下，新函数与准备对象的差异在约 0.13% 内。真实内部工作区分配时，T≤16 的中位增量约 0.17 µs，外部 output 模式最大 +2.27%，默认分配输出模式最大 +2.22%；T17–32 的增量较小。该差异与工作区地址/复用方式相关，不能把同址结果当成所有分配布局都无回退。

### Decode (CUDA Graph)

| T | Decode Down us | Decode Up us | Decode Total us | Frozen baseline backend | Frozen baseline Total us | Speedup |
|---:|---:|---:|---:|---|---:|---:|
| 1 | 4.804 | 6.530 | 11.340 | triton | 31.601 | 2.787x |
| 2 | 4.883 | 6.581 | 11.381 | triton | 31.851 | 2.799x |
| 3 | 4.930 | 6.664 | 11.660 | triton | 32.927 | 2.824x |
| 4 | 4.972 | 6.662 | 11.686 | triton | 32.968 | 2.821x |
| 5 | 4.987 | 6.687 | 11.730 | triton | 34.142 | 2.911x |
| 6 | 5.049 | 6.695 | 11.729 | triton | 34.159 | 2.912x |
| 7 | 5.111 | 6.843 | 11.733 | triton | 34.232 | 2.917x |
| 8 | 5.164 | 6.866 | 11.805 | triton | 34.302 | 2.906x |
| 9 | 5.184 | 7.621 | 12.348 | triton | 35.681 | 2.890x |
| 10 | 5.264 | 7.830 | 12.482 | triton | 35.746 | 2.864x |
| 11 | 5.389 | 7.830 | 12.604 | triton | 35.755 | 2.837x |
| 12 | 5.498 | 7.830 | 12.737 | triton | 36.012 | 2.827x |
| 13 | 5.512 | 7.831 | 12.795 | triton | 35.905 | 2.806x |
| 14 | 5.625 | 7.848 | 12.922 | triton | 35.921 | 2.780x |
| 15 | 5.747 | 7.912 | 13.141 | triton | 36.010 | 2.740x |
| 16 | 5.908 | 7.836 | 13.262 | triton | 36.082 | 2.721x |
| 17 | 6.876 | 9.780 | 16.246 | torch.compile | 36.677 | 2.258x |
| 18 | 7.048 | 9.781 | 16.488 | torch.compile | 32.633 | 1.979x |
| 19 | 7.196 | 9.784 | 16.613 | torch.compile | 32.793 | 1.974x |
| 20 | 7.367 | 9.784 | 16.775 | torch.compile | 32.753 | 1.952x |
| 21 | 7.394 | 9.782 | 16.835 | torch.compile | 32.737 | 1.945x |
| 22 | 7.561 | 9.787 | 16.937 | torch.compile | 32.714 | 1.932x |
| 23 | 7.802 | 9.787 | 17.241 | torch.compile | 32.861 | 1.906x |
| 24 | 8.114 | 9.789 | 17.591 | torch.compile | 32.858 | 1.868x |
| 25 | 8.324 | 9.789 | 17.891 | torch.compile | 27.304 | 1.526x |
| 26 | 8.856 | 9.815 | 18.364 | torch.compile | 27.371 | 1.490x |
| 27 | 8.992 | 9.844 | 18.610 | torch.compile | 27.373 | 1.471x |
| 28 | 9.254 | 9.853 | 18.863 | torch.compile | 27.337 | 1.449x |
| 29 | 9.274 | 9.784 | 18.864 | torch.compile | 27.385 | 1.452x |
| 30 | 9.431 | 9.780 | 19.002 | torch.compile | 27.473 | 1.446x |
| 31 | 9.565 | 9.781 | 19.122 | torch.compile | 27.360 | 1.431x |
| 32 | 9.778 | 9.781 | 19.309 | torch.compile | 27.097 | 1.403x |

### Prefill (eager cudaPerf)

| Batch | Down | Up | Down us / TFLOPS | Up us / TFLOPS | Total us / TFLOPS | Frozen Torch compile us / TFLOPS | Speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 33 | M16/W2/N10/BK1024 | M64/N40 | 16.580 / 13.044 | 16.560 / 13.060 | 31.060 / 13.926 | 46.340 / 9.334 | 1.492x |
| 64 | M16/W2/N10/BK1024 | M64/N40 | 16.980 / 24.701 | 17.340 / 24.189 | 31.880 / 26.313 | 32.360 / 25.922 | 1.015x |
| 128 | M16/W2/N10/BK1024 | M64/N40 | 16.880 / 49.696 | 17.360 / 48.321 | 32.580 / 51.495 | 41.920 / 40.022 | 1.287x |
| 256 | M16/W2/N10/BK512 | M128/N40 | 26.080 / 64.329 | 20.580 / 81.522 | 42.620 / 78.729 | 53.380 / 62.860 | 1.252x |
| 512 | M32/W4/N5/BK512 | M256/N40 | 29.920 / 112.147 | 26.181 / 128.166 | 54.020 / 124.229 | 82.860 / 80.991 | 1.534x |
| 1024 | M32/W4/N5/BK128 | M256/N20 | 58.860 / 114.014 | 39.560 / 169.638 | 93.661 / 143.302 | 160.040 / 83.865 | 1.709x |
| 2048 | M32/W4/N5/BK128 | M256/N10 | 111.681 / 120.180 | 66.121 / 202.990 | 169.101 / 158.743 | 279.401 / 96.075 | 1.652x |
| 4096 | M64/W4/N1/BK64 | M256/N10 | 158.741 / 169.103 | 130.061 / 206.392 | 280.382 / 191.479 | 495.422 / 108.366 | 1.767x |
| 8192 | M64/W4/N1/BK64 | M256/N2 | 286.501 / 187.389 | 276.042 / 194.489 | 550.983 / 194.877 | 946.725 / 113.416 | 1.718x |
| 10240 | M64/W4/N1/BK64 | M256/N2 | 293.442 / 228.696 | 281.001 / 238.820 | 559.963 / 239.690 | 1751.429 / 76.633 | 3.128x |
| 12288 | M64/W4/N1/BK64 | M256/N8 | 416.843 / 193.192 | 382.422 / 210.581 | 795.544 / 202.454 | 1900.530 / 84.745 | 2.389x |
| 16384 | M64/W4/N1/BK64 | M256/N8 | 549.903 / 195.260 | 530.943 / 202.233 | 1078.425 / 199.131 | 3065.656 / 70.050 | 2.843x |
| 20480 | M64/W4/N1/BK64 | M256/N2 | 558.003 / 240.532 | 557.663 / 240.679 | 1107.026 / 242.483 | 3392.918 / 79.116 | 3.065x |
| 24576 | M64/W4/N1/BK64 | M256/N4 | 691.043 / 233.070 | 714.824 / 225.316 | 1407.767 / 228.818 | 3716.799 / 86.667 | 2.640x |
| 28672 | M64/W4/N1/BK64 | M256/N2 | 823.245 / 228.249 | 825.345 / 227.668 | 1651.349 / 227.577 | 4876.746 / 77.062 | 2.953x |
| 30720 | M64/W4/N1/BK64 | M256/N2 | 827.845 / 243.194 | 826.165 / 243.688 | 1664.568 / 241.896 | 5046.306 / 79.792 | 3.032x |
| 32768 | M64/W4/N1/BK64 | M256/N8 | 958.065 / 224.148 | 988.245 / 217.303 | 1953.850 / 219.821 | 5202.287 / 82.559 | 2.663x |
| 36864 | M64/W4/N1/BK64 | M256/N2 | 1089.306 / 221.785 | 1091.646 / 221.310 | 2187.132 / 220.921 | 6348.953 / 76.104 | 2.903x |
| 49152 | M64/W4/N1/BK64 | M256/N2 | 1364.727 / 236.034 | 1373.207 / 234.577 | 2761.334 / 233.309 | 7366.139 / 87.460 | 2.668x |
| 61440 | M64/W4/N1/BK64 | M256/N2 | 1647.088 / 244.464 | 1667.729 / 241.438 | 3358.297 / 239.796 | 6656.535 / 120.980 | 1.982x |
| 65536 | M64/W4/N1/BK64 | M256/N4 | 1775.450 / 241.909 | 1867.910 / 229.934 | 3690.140 / 232.781 | 7254.199 / 118.413 | 1.966x |
Speedup = Torch compile Total / PyHIP Total; eager cudaPerf, full-row calls, no CUDA Graph.
