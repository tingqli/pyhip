# 同配置对照 Aiter 与 tuned MoE

入口：[bench_tuned_moe.py](bench_tuned_moe.py)。在主进程内调用已安装的
`aiter.fused_moe.fused_moe` 和 `pyhip.ops.moe.tuned_moe.fused_moe`，不启动
benchmark worker 或 Ray。默认使用 Aiter 已有配置；`--tune-aiter` 可在对照前
统一运行官方 Aiter tuner。

已有一次运行的[性能快照](perf-snapshot-350.md)，包含失败状态和完整 winner 配置；
只作该次运行的参考，不代表所有设备和输入的性能保证。

## 共享模型配置

[moe_shapes.py](../../src/pyhip/testing/moe_shapes.py) 保存八个模型/量化配置；固定 kernel
pytest 和 benchmark 都读取该表，避免 shape 漂移。这里是原测试预设，不是模型所有
checkpoint/TP 的通用定义。只运行单个 TP shard 的 MoE，不包含通信。

| 名称 | model_dim | inter_dim（全局） | TP | inter_dim_tp（实际 kernel） | experts | topk | 默认 FP8 量化 |
|---|---:|---:|---:|---:|---:|---:|---|
| hy3 | 4096 | 1536 | 8 | 192 | 193 | 9 | per_tensor |
| qwen35_397B | 4096 | 4096 | 8 | 512 | 512 | 10 | ptpc |
| qwen35_397B_k256 | 4096 | 2048 | 8 | 256 | 512 | 10 | ptpc |
| qwen35_35B | 2048 | 512 | 1 | 512 | 256 | 8 | ptpc |
| qwen35_35B_k256 | 2048 | 256 | 1 | 256 | 256 | 8 | ptpc |
| mimo_ptpc | 6144 | 2048 | 8 | 256 | 384 | 8 | ptpc |
| mimo_block | 6144 | 2048 | 8 | 256 | 384 | 8 | block |
| h3 | 6144 | 3072 | 8 | 384 | 128 | 4 | ptpc |

**双方始终使用相同的实际维度，不做后端专属 padding。** 已移除的旧交互测试
`entry_common()` 曾将 FP8 Aiter 的 I_tp 按 128 对齐、其他后端按 64 对齐；
因此 Hy3 可能变成 Aiter I_tp=256 对 PyHIP I_tp=192。这种数据不能直接当作
同配置 speedup。新入口保留 I_tp=192；若某后端不支持，记录失败而不是改 shape。

输入生成、独立 Torch 参考、输出检查和多 buffer 测量统一放在
[pyhip.testing.moe](../../src/pyhip/testing/moe.py)，由 benchmark 和 pytest 共用。
固定 FlyDSL/ASM kernel 的正确性和性能测试见
[MoE pytest 说明](../../tests/ops/moe/README.md)，不在本脚本新增后端选择功能。

## 使用

在仓库根目录、ROCm Python 环境中运行。建议先小 batch 精度，再扩大矩阵。

```bash
# 只列模型，不导入 Torch/Aiter 或初始化 GPU
python benchmarks/moe/bench_tuned_moe.py --list-models

# 精度检查；首次无缓存时仍会搜索候选，但不做双方性能对照
python benchmarks/moe/bench_tuned_moe.py \
  --models hy3 qwen35_35B_k256 --tokens 1 4 64 --dtype fp8 \
  --check-only --output /tmp/moe-accuracy.json

# 每个实际 M 重新调优，再关闭强制搜索，进行对照
python benchmarks/moe/bench_tuned_moe.py \
  --models qwen35_35B_k256 --tokens 1 4 64 1024 --dtype fp8 \
  --retune --warmup 2 --iters 10 --rounds 2 --output /tmp/moe-perf.json

# 全模型同 batch 矩阵；默认沿用各模型的 FP8 量化模式
python benchmarks/moe/bench_tuned_moe.py \
  --models all --tokens 1 4 64 1024 --dtype fp8 \
  --output /tmp/moe-all.json --md /tmp/moe-all.md

# 先批量调优 Aiter，再重新搜索 PyHIP 候选并比较；目录参数可省略
python3 benchmarks/moe/bench_tuned_moe.py \
  --models qwen35_35B qwen35_35B_k256 --tokens 1 4 64 1024 --dtype fp8 \
  --tune-aiter /tmp/moe-aiter-tuning --output /tmp/moe-after-tuning.json

# FP8 block-scales：W128×128，A1×128；小 batch 和 prefill 都参与搜索
python3 benchmarks/moe/bench_tuned_moe.py \
  --models qwen35_35B_k256 --tokens 1 64 1024 --dtype fp8 --quant block \
  --retune --output /tmp/moe-block.json

# BF16 非 gated GELU（G1U0）：沿用模型维度，不代表原模型使用 GELU
python3 benchmarks/moe/bench_tuned_moe.py \
  --models qwen35_35B_k256 --tokens 1 64 1024 --dtype bf16 --activation gelu \
  --retune --check-only --output /tmp/moe-gelu.json

# 独立检查 BF16，或 MXFP4 的 caller 布局与激活参数
python benchmarks/moe/bench_tuned_moe.py \
  --models qwen35_35B_k256 --tokens 4 --dtype bf16 --check-only
python benchmarks/moe/bench_tuned_moe.py \
  --models qwen35_35B_k256 --tokens 4 --dtype mxfp4 \
  --gate-mode interleave --activation situv2 --beta 0.5 --linear-beta 2 \
  --check-only
```

- `main()` 调用 `torch.set_default_device("cuda")`，统一在当前 CUDA/HIP 设备
  创建测试张量；导入模块、`--help` 和 `--list-models` 不改变默认设备。
  `torch.Generator` 仍显式选择 CUDA，因为它不遵循默认张量设备设置。
  可通过 `HIP_VISIBLE_DEVICES` 限制可见设备；张量默认不需要梯度。
- `--quant ptpc|per_tensor|block` 只覆盖 FP8。block 要求 H/I_tp 同时按128
  对齐，不符合则报错；`--preshuffle off` 不改变 shape，只改变共同输入布局。
  gfx950 的 block-scale SiLU 候选包括原生 FP8 8-wave 两阶段路径及小 K
  persistent Down，激活在两次 GEMM 前分别量化，scale 使用转置存储。
  普通 8-wave 路径支持 raw/shuffled 权重；persistent Down 只用于 shuffled w2。
  `jit_splitk` 也可作为 A16W8 block 候选：两份权重已 shuffle、SiLU/separated、
  H 按512对齐时，kernel 反量化权重并直接消费 BF16 输入及中间激活，不执行
  A1×128 激活量化。gfx942 可搜索该路径和 Aiter，但不加入 gfx950 专用 8-wave。
  两种路径都须通过相同的独立参考和原误差门槛。
- gfx950 的 BF16 SiLU 也加入 `jit_8wave`，共用普通/persistent 8-wave 流程，
  不量化激活；两份权重分别遵循 `is_shuffled`。persistent Down 支持小 K 和
  OC split1/2/4，counter 每次调用及 graph replay 都在输入设备清零。
- MXFP4 SiLU 增加 `jit_mxfp4_4wave`：专用 4-wave Gate/Up 搭配通用 MXFP4 Down；
  `jit_mxfp4` 保留通用 Gate/Up，两条路径均采用 A4W4、sorted activation scales
  和运行时选择的 final reduce：H 按1024对齐时用 JIT `moe_gemm_final_reduce_bf16`，
  否则用 `torch.sum` 写入同一 output，不因此排除 GEMM 候选。GEMM 要求 separated、
  两份权重已 shuffle、H/I_tp 按256对齐、相关 tensor 小于4 GiB；
  专用 Gate/Up 搜索 M/N128/256，Down N128。
  **A4W4 候选仍需通过公共 A16W4 Torch 参考的原精度门槛**，不能因为支持该 kernel
  就保证它在每个 shape 上成为有效候选。固定 kernel pytest 单独检查 A4W4 数值语义。
- FlyDSL direct decode 将输出清零融合进 Gate/Up，BF16/FP8/MXFP4 都支持，
  大 H、小 I_tp 时用多轮清零覆盖全部输出。MXFP4 的 B1 同时搜索 Down DN32/64。
  `fly_decode` 的 direct/sorted 路径均保留 BF16 激活，支持 PTPC、per-tensor 和
  MXFP4 权重反量化，不支持 block scale；`jit_splitk` 支持 PTPC、block、MXFP4，
  不直接支持 per-tensor scales。省去激活量化可能降低小 batch 开销，但不保证
  所有输入都更快或更接近量化参考；加入候选不等于一定通过 prune 或成为 winner。
  prefill 的 Gate/Up BN 搜索128/256，BK 为 BF16 的64/128或 FP8 的128/256；
  三条专用 Down 路径也覆盖这两组 BN，输出 padding 搜索0/128B。
  默认 FP8 Down 不再排除 I192/I320。候选层只枚举配置并尊重 caller 的 M 设置，
  不重复维护 FlyDSL 的 K/E 白名单、对齐和 LDS 公式；kernel 编译失败的配置由
  prune 排除，其余配置仍须通过原精度检查。winner 的 `tile_k_gate` 保存在 JSON。
  Gate/Up 和 Down 的 FP32→BF16 沿用公共转换函数的 RTA/RTE 选项，当前默认 RTA；
  如需 RTE，在进程启动前设置 `AITER_FLYDSL_MOE_BF16_RTA_SIMPLIFIED=0`。
  舍入选项应在导入 kernel 前设置；改变选项后重新调优，不在运行中切换。
  inverse/sorted reduce 的编译缓存按 tensor 所在设备隔离，每次使用该设备当前 stream。
- `--activation gelu --dtype bf16` 使用 G1U0：W1 为 `[E, I_tp, H]`，不含 gate；
  W2 为 `[E, H, I_tp]`。`jit_gelu` 复用现有 8-wave kernel，固定 M/N tile=256，
  要求 gfx950、H/I_tp 按256对齐、`--gate-mode separated`，并满足32-bit buffer
  地址范围；两份权重分别支持 raw/shuffled。实际 token 数不必是256的倍数。
  GELU kernel 使用 tanh 近似，仍对独立的默认 Torch GELU 参考检查，不放宽阈值。
  不加入旧 wrapper 中的 MXFP4 Torch fallback，也不把 gated kernel 当作 G1U0。
  **当前官方 Aiter tuner 只支持 G1U1，所以 GELU 不可与 `--tune-aiter` 同用。**
  当前 Aiter 的 BF16 G1U0 路径在已测 shape 上不通过精度；benchmark 仍会搜索和
  验证 `jit_gelu`，并显示双方 diff、时延和加速比，状态标记 `AITER_INCORRECT`。
  此时加速比仅比较执行时间，不代表两条实现都给出了正确结果。
- `--routing balanced` 沿用旧测试的随机 expert permutation 循环分配；
  `random` 对每个 token 使用随机 top-k。双方共用相同路由和 routing weights。
- `--copies 0` 使用 `pyhip.run_perftest` 内部约4GB复制预算；可显式指定份数。
  **外部只准备一份 tensor，不重复建立 buffer 池。** 大模型仍需足够显存。
- `--retune` 在计时前逐个 M 强制搜索。不开启时，相同 power-of-two bucket
  可能共用已有 winner，不能把 bucket 复用称为“每个 M 都单独调优”。
  缓存目录沿用 `FLYDSL_AUTOTUNE_CACHE_DIR`；
  `FLYDSL_AUTOTUNE_CONFIG_DIR` 可启用强制调优时的离线配置导出。
  本次迁移使用 v6 缓存，不复用 v5 及更早配置；首次调用会重新调优。
  block 模式的 key 记录同时允许 A1×128/BF16 激活的候选策略，之前仅搜索
  A1×128 的 winner/cache/artifact 不再命中；其它模式不受影响。
- `--tune-aiter [DIR]` 默认目录为 `./tuned_aiter_moe`。先把整个模型/token 矩阵
  按 Aiter 的 lookup key 去重写入 `untuned.csv`，再调用已安装 Aiter 的
  `csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py`，最佳 kernel 保存到
  `tuned.csv`。使用 `--all --mp 1` 重新调优本次 shape，不覆盖 Aiter 自带配置；
  开启官方普通 tuner 的全部候选来源，不继承 `TUNE_ONLY`/kernel regex 筛选。
  tuning 使用首张可见 GPU，运行前可用 `HIP_VISIBLE_DEVICES` 选卡。
  - token 使用 Aiter 的 `get_padded_M()` bucket，例如 3/4 共用 M=4 的配置；
    H/I_tp 保持实际维度，正式对照仍用用户指定的实际 M，不给输入 padding。
    MXFP4 的激活类型按实际 M、gate mode 和 Aiter 环境开关选择。
  - 设置 `AITER_CONFIG_FMOE`、刷新配置缓存，并核对 dispatcher 的 kernel 名称、
    block_m、ksplit 与 CSV 一致。tuner 失败、缺少最佳行或退回 heuristic 时停止，
    不把未调优的 Aiter 当成最佳基线；官方 tuner 只测 shuffled 权重，所以要求
    `--preshuffle on`。
  - 自动重新搜索 PyHIP 候选，避免复用 Aiter 调优前的 winner；不必再加 `--retune`。
    调优和配置验证都在正式计时之前，JSON 记录本次 Aiter 配置路径。
- 输出路径必须是新文件，防止覆盖历史报告。非通过/不可比状态返回非零退出码。

## 对照口径

1. 一次生成 H/I_tp/E/topk、随机输入、routing、量化权重和 scales。权重按专家
   生成，减少准备阶段峰值显存；两条调用路径共用相同数值和布局。
2. 分别对独立 Torch 参考验证两条 API，使用现有 `calc_diff <= 0.02` 加
   finite/output identity 检查。该指标是能量归一化误差，**不是逐元素2%误差**。
   Aiter 本身也可能不通过；不会用 Aiter 输出当唯一参考。
3. 调优、编译、参考和 weight shuffle 均在对照计时之外。计时包含完整 eager
  API 的 sorting、activation quant、GEMM1、GEMM2、reduction、内部 workspace
  分配/launch，以及 tuned API 的缓存查找；不是裸 kernel 或 graph replay。
  两边共同使用的输出跟踪 wrapper 的 Python 开销也在 eager event 区间内。
4. 计时前关闭 `FLYDSL_AUTOTUNE`，使用上一步已选好的配置；关闭
  `AITER_ONLINE_TUNE`。只有 `--tune-aiter` 的前置阶段运行官方 tuner，
  正式计时不会触发调优。原环境值在结束时恢复。
5. 统一调用 `pyhip.run_perftest`，传入所有顶层 tensor（包括 output/scales），
   保留 clone 后的 `is_shuffled`。默认按 Aiter→tuned、tuned→Aiter 两个 block
   交错；每个实现各自使用内部复制池，**不是同地址逐样本配对测量**。
6. 每轮检查所有实际使用的输出副本，输出在 clone 前填 NaN，不重跑另一个
  结果代替校验。Aiter 数值失败（包括 NaN/Inf）不阻止计时，仍记录检查结果；
  winner 数值失败或任何一方执行/输出元数据错误时不生成加速比。
  `run_perftest` 返回均值，摘要从全部原始样本算中位数。
7. `speedup = Aiter_median_us / tuned_median_us`；大于1表示本次 tuned API 更快。
  gated 模式有效计算量为 `6*M*topk*model_dim*inter_dim_tp`；非 gated GELU 为
  `4*M*topk*model_dim*inter_dim_tp`，均不含激活/quant/通信 FLOPs。
   表内 winner 可能是 Aiter；这种情况仍测整个 tuned API 的调度开销。

FlyDSL 调优器自己的计时与最终多buffer测量不完全相同。因此 winner 是“调优器
选中的配置”，不保证最终所有地址/路由分布中都最快。Aiter 精度失败时仍比较
时延，但保留失败标记；winner 仍须通过原精度阈值。未支持形状不擅自 padding、换 dtype。
同样，Aiter 的最佳行来自官方 tuner 的输入生成、精度门槛和计时方式；即使开启
`--tune-aiter`，仍执行这里独立的 Torch 参考和实际计时输出检查。

## 输出

运行结束打印一张汇总表，包括 `Aiter diff`、`winner diff`、`Aiter us`、`tuned us`、
`speedup`、`winner`、`winner TFLOPS`、状态和完整 winner config。
`winner TFLOPS` 紧跟 `winner` 列，使用实测 tuned API 的中位时延：
`effective_FLOPs / (tuned_median_us * 1e6)`，gated/非 gated 的有效计算量见上文。
即使 winner 是 Aiter，也使用 tuned API 的实测时延；未计时则显示 `—`。
diff 取计时前检查及各轮输出副本中的最大值；非有限输出
显示 `NaN/Inf`，无法检查显示 `—`，JSON 不写入非标准的 NaN/Infinity 数值。
空、非有限或非正的计时样本记为 `ERROR`，不计算 speedup/TFLOPS，也不写入报告样本。
可选 JSON 只保存运行参数、模型维度、精度结果、配置和计时样本。

`--md FILE` 将相同的最终表格写入 Markdown，不重跑计时，也不收录调优日志。
每个模型使用三级标题，包含模型名、dtype/quant、activation 和 TP；标题下记录
H/I/I_tp、experts/topk、gate mode、shuffle、routing、seed 和激活参数，随后列出各个 M。
失败状态及原因也保留。可与 `--output` 同用；两者必须使用不同的新文件，
不会覆盖已有报告，缺失的父目录自动创建。

winner 来自 tuned MoE 的实际 dispatch，不读取 FlyDSL 私有缓存。模块的
`record_dispatch` 默认关闭；开启后，`last_dispatch` 只覆盖保存最后一次配置，
不持有 tensor。benchmark 仅在计时前的 tuned 调用开启记录，成功返回后取快照，
正式计时关闭。该诊断接口仅供串行使用；graph replay 不会更新记录。

不采集 GPU 状态，不检查利用率/显存门槛，不扫描源码或记录 tensor 地址。
请自行选择合适的设备；延迟和加速比只是本次运行的实测结果。

旧 MoE wrappers、逐阶段调试脚本、subprocess 比较入口和清缓存启动脚本已删除。
不提供旧 `method` 参数的兼容转发，也不保留 MXFP4 GELU 的 Torch 执行 fallback；
GELU 本地候选仍只支持 BF16。EP 等公共 API 功能继续转交 Aiter，不承诺旧本地路径行为。
自定义 shape 可在固定 pytest 中添加模型字典；benchmark CLI 仍使用共享模型表。

| 状态 | 含义 |
|---|---|
| CHECK_PASS | 两条实现精度通过，只做正确性，不做性能对照 |
| PASS | 两条实现和实际计时输出校验通过，显示延迟和加速比 |
| AITER_INCORRECT | Aiter 数值检查失败，winner 通过；除 check-only 外仍显示延迟和 speedup，退出码保持非零 |
| NOT_COMPARABLE | 执行/输出元数据检查失败，或 winner 精度失败，不生成 speedup |
| ERROR | 数据准备、参考或调用错误；reason 中保留异常 |