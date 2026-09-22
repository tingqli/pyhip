# GR read prefill：正式接口与接入说明

当前实现是两阶段 BF16 prefill：Down GEMM + SiLU 写入 P，Up GEMM + sigmoid + 四路加权归约写入 Y。T48–512 的调优已进入正式源码；调用方无需导入实验目录，也无需手动选择 kernel。

支持 ROCm gfx942。固定维度为 C=4、H=2560、R=320、K=10240；本轮性能配置在 MI308X / 80CU 上验证。此入口负责 prefill，独立 decode 实现及上层 prefill/decode 分派另行接入。


```python
from pyhip.contrib.flydsl.gr_read import GRReadPrefill, prepare_weights

# 模型加载 / 准备阶段，每对原始 BF16 权重只做一次。
# w_down: [320, 10240]，w_up: [10240, 320]，同一张 ROCm GPU。
packed_down, packed_up = prepare_weights(w_down, w_up)

# 按实际 T 提前准备并缓存对象：分配 P/Y、编译均发生在构造时。
read64 = GRReadPrefill(64, packed_down, packed_up)
read512 = GRReadPrefill(512, packed_down, packed_up)

# 运行阶段：x64 为已经归一化的 contiguous BF16 [64, 10240]。
# 在当前 stream 上启动 Down、Up；返回对象持有的 BF16 [64, 2560]。
y64 = read64(x64)

# 分阶段检查 / 测量；Up 消费本次 Down 生成的 P。
p64 = read64.run_down(x64)
y64 = read64.run_up(x64)
```

每层每个 device 保留一份 packed 权重，所有 T 共用其指针。Down 为 N16/K32 preshuffle；Up 先按 H64 做 `reshape(4,40,2,4,2,4,320).permute(1,0,2,4,3,5,6)`，再做同一 N16/K32 preshuffle。这与现有 decode 的权重布局一致。将来接入 SGLang 时仍在加载阶段做一次，不能放进每次 forward。

每个对象固定 T/device，独占 P `[T,320]` 和 Y `[T,2560]`；下一次调用覆盖返回值。需要保留结果时由调用方复制。并发 stream 各用独立对象，可共享只读 packed 权重；准备阶段与执行 stream 的依赖由调用方按正常 Torch stream 规则管理。

已有工作区时可使用 `GRReadPrefill(T, packed_down, packed_up, partial=p, output=y)`。P/Y 必须与 X、权重及彼此独立，shape/dtype/device 正确、连续、起始地址 16B 对齐。构造会执行编译样例并写入工作区，不要在已有数据仍被消费时重新构造。T=0 返回空工作区，不 launch。

## 2. 正式源码与分派

| 文件 | 职责 |
| --- | --- |
| [host.py](../../../src/contrib/flydsl/gr_read/host.py) | `GRReadPrefill` 准备、工作区、当前 stream/device、完整或分阶段调用 |
| [common.py](../../../src/contrib/flydsl/gr_read/common.py) | 公共权重预处理入口、维度、固定配置选择 |
| [prefill_down.py](../../../src/contrib/flydsl/gr_read/down.py) | 原 Down 与调优后的 tile/BK/swizzle 参数 |
| [prefill_up.py](../../../src/contrib/flydsl/gr_read/up.py) | 原 M256 Up，按参数分派小 M 版本 |
| [prefill_up_small.py](../../../src/contrib/flydsl/gr_read/up.py) | 合并后的 M64/M128 Up 编译期特化 |
| [helpers.py](../../../src/contrib/flydsl/gr_read/helpers.py) | 原数值与底层辅助函数 |

80CU 上的配置如下，选择发生在准备阶段，无运行时 autotune：

| T | Down M / waves / N splits / BK | Up M / N splits | Down swizzle shift |
| --- | --- | --- | --- |
| 48–128 | 16 / 2 / 10 / 1024 | 64 / 40 | 7 |
| 129–256 | 16 / 2 / 10 / 512 | 128 / 40 | 6 |
| 257–512 | 32 / 4 / 5 / 512 | 256 / 40 | 6 |
| 其余 T，或其它 CU 数 | 原 prefill 配置选择 | 原 M256 Up | 3 |

T48–512 将两个 GPU launch 收在一个已编译 host 调用中，仍然是两个 GPU kernel。此处未捕获 CUDA Graph。范围外保留原两个 launcher；其它 CU 数的回退没有在本机做性能验收。

旧测试脚本的 `prepare_gr_read` / `run_gr_read` tuple 接口仍保留兼容；业务接入使用上面的公共对象，才能直接使用最终的配置与 host 调用路径。

## 3. 数值边界与验收范围

Down 保持完整 K 的 FP32 累加 → BF16 舍入 → 转回 FP32 除 4、SiLU → BF16 P。Up 保持原 FP32 logits、四路顺序 FMA 及最终 BF16 舍入。这次调整 tile、调度和准备接口，未删除 SiLU 之前的 BF16 舍入边界。

2026-09-21 的调优及正式目录迁移分别验证了 100 对真实 HC 权重 × 30 档 T（48,64,…,512）× 初始/改变 X，共 6000 例 P/Y 逐位相同。X 仍为随机 BF16，未作端到端模型质量评估。正式入口的 50 项 pytest 全部通过，包含空输入、尾块、配置边界、工作区、stream 和双 GPU。

“逐位相同”指保留同事原 prefill 的结果，不表示解决已有参考误差。之前真实权重 T33/T512 的 400 例检查中，原 prefill 有 16 例 P、1 例 Y 未通过原 BF16 Torch reference 容差；本次没有放宽容差，也没有宣称修复这些舍入边界案例。

## 4. 正确性与性能入口

```bash
# 全部基础正确性，包括大 T 与边界；单 GPU 时双 GPU 项跳过。
HIP_VISIBLE_DEVICES=0,1 python3 -m pytest -q tests/contrib/gr_read/test_gr_read.py

# 原入口默认 21 档：32/64/128/256/512 及 1K–64K。
python3 tests/contrib/gr_read/test_gr_read.py --gpu 3 --check-only
# 性能表同时输出 Down、Up、Total、Torch compile Total 和加速比。
python3 tests/contrib/gr_read/test_gr_read.py --gpu 3

# 重点检查新的配置边界。
python3 tests/contrib/gr_read/test_gr_read.py --gpu 3 --check-only \
  --batches 47 48 49 128 129 256 257 512 513

# 与当前 SGLang checkout 的 prefill 对照，普通调用，无 CUDA Graph。
python3 tests/contrib/gr_read/bench_gr_read_compare.py \
  --phase prefill --sglang-root /opt/sglang --gpu 3 \
  --rows 48 64 128 192 256 384 512 1024 \
  --output /tmp/gr_read_prefill_compare.jsonl
```

原性能入口分别测 Down、Up、Total，并新增 Torch compile 完整调用。使用原 `cudaPerf`，默认 10 组独立 buffer、各阶段 2 次预热、10 个样本取中位数；保留逐阶段采样顺序，Torch compile 接在 Total 后面。这张表使用各阶段整组样本的中位数，不是交错 A/B 测量。Total 直接测量。权重打包、对象构造、首次编译、参考与校验不计时。门禁前固定静置 2 秒，单次查询要求 GPU use≤5%、VRAM≤20%、PTL Enabled/VECTOR,F8；失败停止并保留数据。没有修改设备设置或剔除长尾。

表格新增 `Torch compile us / TFLOPS` 和 `Speedup`，其中 **Speedup = Torch compile Total / PyHIP Total**，大于 1 表示 PyHIP 更快；结果及全部样本也写入原 JSON/JSONL 输出。TFLOPS 两边都按 `4*T*10240*320` 的有效 GEMM 工作量计算。

这里的 Torch compile 对照在脚本内保留 SGLang `2843214f6ed923e992a74ee4d7a0cda5d7deddbf` 的 `_mix_compute` 公式，使用默认 `torch.compile`，只返回 Y，无需安装 SGLang。每次对完整 T 调用一次；原来按 1024 行分块、返回 P/Y 的正确性参考仍单独保留，不用于计时。两边使用同一个 X；Torch 轮换 10 对原始 BF16 权重，PyHIP 使用对应的 packed 权重，均在准备阶段生成。Torch 保留原生中间张量和输出分配，地址单独记录，内部工作区不强制与 PyHIP 相同。所有实际被计时的 Torch 输出都会检查；prefill 全程普通调用，无 CUDA Graph。`--check-only` 仍只运行原正确性检查。

[SGLang 对照脚本](bench_gr_read_compare.py)读取指定 checkout 的原 `_mix_compute` AST 并按原方式 `torch.compile`，同时加载 `hc_mix_triton.py` 的实现及支持判断。两边共用原始权重值和 X；PyHIP 的 packed 权重每对只准备一次、跨所有 T 复用。JSONL 记录版本、源码 hash、地址、原始样本、初始/改变输入 FP64 检查；`--output` 必须是新文件。

当前核对的 SGLang `2843214f6ed923e992a74ee4d7a0cda5d7deddbf` 在 gfx942 非确定性推理下，T1–16 为 Triton persistent，T≥17 为 torch.compile。对照脚本按实际支持函数选择后端。测量从已经归一化的 X 开始，不含 RMSNorm、TP 通信和整个模型。

本 prefill 补丁可独立运行上述对照；若仓库同时有 `tests/contrib/gr_read_decode/kernel.py`，同一脚本的 `--phase decode` 默认覆盖 T1–32，Down/Up/Total 分别捕获 CUDA Graph。独立 decode 源码未包含在本 prefill 补丁中。

T48–512 的正式优化前后对照已在 30 档、普通调用下验证：选中的两段实现全部快于该版 SGLang torch.compile，范围为 1.06–4.19×。这是特定硬件和固定协议的测量，不能将少量样本 smoke 的时延替代完整数据。迁入公共接口后的同址配对复测中，30 档时延变化中位数为 −0.095%，最大增加 0.451%；完整报告和原始数据记录在本机 `qwen3.8-flash-next-doc` 的 45/46 号文档。

## 5. 有效优化方法

1. **先固定数值边界，再减少算术。** Down完整K10240 FP32累加后先舍入BF16，再做FP32缩放和SiLU，写BF16 P。Up完整R320归约、直接FP32 logits、stream0→1→2→3顺序FMA，最后乘0.25并使用既有整数BF16 helper。浮点FMA加偏置不能替代整数位模式舍入；“容差通过”与“逐位等价”分开验收。[数值helpers](../../../src/contrib/flydsl/gr_read/helpers.py)

2. **按实际CTA轮数独立选择Down／Up的N分片。** Down N1/N2对应320／160列，Up选择N2/N4/N8；比较$\lceil B_MN/U\rceil c_N$而不是只看单CTA工作量。Down用$B_D=\lceil T/64\rceil$，Up用$B_U=\lceil T/256\rceil$；更多分片有利于小batch填满CU，却可能增加大batch轮数和重复读取。成本是历史校准模型，不是跨设备最优保证。[选择规则](../../../src/contrib/flydsl/gr_read/common.py)

3. **让权重顺序匹配归约顺序和输出lane。** Up先将权重排列为`[H64 tile, stream, H32 half, ...]`，相邻8个H32 packet完成同一H64的四路归约；内层重排使同一lane取得连续8个BF16，适配X和16B Y store。随后N16/K32 preshuffle负责MFMA输入打包，不能混淆两层排列。变更前先用CPU标签证明索引双射，再同步修改消费地址。[准备函数](test_gr_read.py#L110)

4. **交错当前MFMA与上一packet的后处理。** 在Compute(q)计算当前logits，同时推进q−1的scale／exp／rcp／FMA／BF16打包，拆开后处理自身依赖链。当前每10条MFMA推进两个旧元素，SFU独占间隔，普通间隔控制VALU数量；实际收益取决于机器调度和生存期，不是让指令间隔形式上均匀。保留FIRST／LOOP／LAST和drain的完整覆盖。[Up流水](../../../src/contrib/flydsl/gr_read/up.py)

5. **联合调整预取深度、寄存器占用与等待。** 当前W2在Memory(q)发布W(q+1)、预读W(q+2)，少保留一个W搬运包；没有减少总W字节。任何预取或发射顺序改动都要重算W/X/Y的VMEM完成账本，检查LDS覆写和实际ISA。4＋4wave错相必须首尾闭合，barrier不能替代完成等待，VGPR下降也不自动提高占用率。

6. **按缓存行组织X协作读取，再恢复计算布局。** 同行8个lane各读16B形成X128，覆盖H64并供相邻两个H32消费；通过每wave3KiB LDS恢复MFMA布局。两个20KiB W槽＋24KiB X区共64KiB。X/P读和Y写当前使用NT、W保持default；合并请求能减少重复取读，但增加LDS工作，不能只凭HBM字节下降宣称加速。

7. **64位tile基址＋局部buffer边界，避免Host分块和P padding。** CTA入口先以64位元素偏移重设X/P/Y，descriptor只覆盖当前tile有效行。Down N1输出范围为`valid_rows*R*2`；N2保留stride R，范围为`((valid_rows-1)*R+BN)*2`，valid0为0。二维tensor避免展平动态shape超i32；rows运行时传入，工厂仅按N分片缓存。正常`flyc.compile`会执行一次，必须提供足量张量。[Down尾块处理](../../../src/contrib/flydsl/gr_read/down.py)

8. **把性能条件和归因口径固定下来。** 正确性guard view与性能原生allocation分开，记录X/P/Y相对地址；同一ELF也会对地址偏移敏感。保留长尾，用同址交错对照确认小收益。ATT只分析共同稳态窗口，区分issue-stall和completion-wait，不把跨wave stall求和当墙钟，也不把MFMA union×roof模型TFLOPS当实测。逻辑M-major或swizzle不保证物理XCD归属，需单独验证。

可复用流程已整理为项目skills：
- [grread-bf16-prefill](../../../.github/skills/grread-bf16-prefill/SKILL.md)：数值边界、权重／lane布局、N分片、X128/W2流水、精确尾块。
- [gpu-benchmark-validation](../../../.github/skills/gpu-benchmark-validation/SKILL.md)：空闲GPU与PTL、原计时器和地址条件、原始样本、ATT/PMC归因边界。


## 6. Decode T1–32：独立正确性、原 Graph 基准和 SGLang 对照

实现已按方向收拢：[down.py](../../../src/contrib/flydsl/gr_read/down.py) 包含 prefill Down 和 decode split-K Down，[up.py](../../../src/contrib/flydsl/gr_read/up.py) 包含所有 prefill Up 和 decode Up。上文旧文件名对应的实现均迁入这两个文件。两后端共用 `common.prepare_weights()`；模型加载时 prepare 一次，同一份 packed Down/Up 权重可传给 `GRReadDecode` 和 `GRReadPrefill`。FlyDSL 验证版本为 **0.3.2**。

Decode 默认覆盖全部 T1–32，独立测量、独立出表。Prefill 继续普通调用，默认性能矩阵首档现为 T33；T33–64 全部使用之前验证的小 M 配置（同上文 T48–128 的配置，包括 T48），其余 prefill 配置保持。

在仓库根目录运行：

```bash
# 原 decode 正确性：默认 2 对权重、全部 T1–32、6528 次 Graph replay。
HIP_VISIBLE_DEVICES=2 python3 tests/contrib/gr_read_decode/test_gr_read.py

# 原 decode 带宽型负载基准：100 对独立权重，全部 T1–32。
HIP_VISIBLE_DEVICES=2 python3 tests/contrib/gr_read_decode/bench_gr_read.py \
  --weights 100 --rounds 3 --samples 7 --seed 707 \
  --output /tmp/gr_read_decode_graph.jsonl

# Decode 与当前 SGLang 单独对照，Down/Up/Total 分别捕获 Graph。
python3 tests/contrib/gr_read/bench_gr_read_compare.py \
  --phase decode --sglang-root /opt/sglang --gpu 2 \
  --output /tmp/gr_read_decode_sglang.jsonl
```

`--output` 使用新文件名。原 decode 基准的 100 组独立权重工作集、buffer 准备、capture、计时器和检查流程保持：图内依次执行所有实例两轮，每个样本 replay 三次，Event 时间除以 600；3 轮 × 7 个样本全部保留并取中位数。Packing、编译和参考不计时。输出仍使用原微秒时延与有效 TFLOPS 指标，未把逻辑字节数当成实测 HBM 流量。

SGLang 对照读取指定 checkout 的原实现和支持判断。在本机 gfx942、默认非确定性推理下，T1–16 为 Triton persistent，T17–32 为 Torch compile；两边均使用 decode 的 Graph 协议。对照表包含 T1–32 每一行和实际后端，独立于 prefill 表。两边使用相同 X 与逻辑权重，PyHIP packing 在准备阶段完成。

原 FP64 容差 `rtol=1e-2, atol=5e-3`、live rows 缩小/恢复、zero/stale/NaN 尾行、P/Y 预污染、内部 padding、guard、输入及权重不变检查全部保留。
