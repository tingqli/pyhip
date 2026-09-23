# 同配置对照 Aiter 与 tuned MoE

入口：[bench_tuned_moe.py](bench_tuned_moe.py)。在主进程内调用已安装的
`aiter.fused_moe.fused_moe` 和 `pyhip.ops.moe.tuned_moe.fused_moe`，不启动
benchmark worker 或 Ray。默认使用 Aiter 已有配置；`--tune-aiter` 可在对照前
统一运行官方 Aiter tuner。

## 共享模型配置

[moe_shapes.py](../../src/pyhip/testing/moe_shapes.py) 保存从
[旧 MoE 测试](../../tests/ops/moe/test_moe.py) 抽出的七个模型配置；固定 kernel
pytest 和 benchmark 都读取该表，避免 shape 漂移。这里是原测试预设，不是模型所有
checkpoint/TP 的通用定义。只运行单个 TP shard 的 MoE，不包含通信。

| 名称 | model_dim | inter_dim（全局） | TP | inter_dim_tp（实际 kernel） | experts | topk | 默认 FP8 量化 |
|---|---:|---:|---:|---:|---:|---:|---|
| hy3 | 4096 | 1536 | 8 | 192 | 193 | 9 | per_tensor |
| qwen35_397B | 4096 | 4096 | 8 | 512 | 512 | 10 | ptpc |
| qwen35_397B_k256 | 4096 | 2048 | 8 | 256 | 512 | 10 | ptpc |
| qwen35_35B | 2048 | 512 | 1 | 512 | 256 | 8 | ptpc |
| qwen35_35B_k256 | 2048 | 256 | 1 | 256 | 256 | 8 | ptpc |
| xiaomi | 6144 | 2048 | 8 | 256 | 384 | 8 | ptpc |
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
  --models all --tokens 1 4 64 1024 --dtype fp8 --output /tmp/moe-all.json

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
  gfx942 不加入这些 gfx950 专用 kernel，仍使用已有 split-K/Aiter 候选。
- FlyDSL direct decode 将输出清零融合进 Gate/Up，BF16/FP8/MXFP4 都支持，
  大 H、小 I_tp 时用多轮清零覆盖全部输出。MXFP4 的 B1 同时搜索 Down DN32/64。
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
  本次候选补齐使用 v5 缓存，不复用 v4 及更早配置；首次调用会重新调优。
- `--tune-aiter [DIR]` 默认目录为 `./tuned_aiter`。先把整个模型/token 矩阵
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
`speedup`、winner 和状态。diff 取计时前检查及各轮输出副本中的最大值；非有限输出
显示 `NaN/Inf`，无法检查显示 `—`，JSON 不写入非标准的 NaN/Infinity 数值。
可选 JSON 只保存运行参数、模型维度、精度结果、配置和计时样本。

winner 来自 tuned MoE 的实际 dispatch，不读取 FlyDSL 私有缓存。模块的
`record_dispatch` 默认关闭；开启后，`last_dispatch` 只覆盖保存最后一次配置，
不持有 tensor。benchmark 仅在计时前的 tuned 调用开启记录，成功返回后取快照，
正式计时关闭。该诊断接口仅供串行使用；graph replay 不会更新记录。

不采集 GPU 状态，不检查利用率/显存门槛，不扫描源码或记录 tensor 地址。
请自行选择合适的设备；延迟和加速比只是本次运行的实测结果。

| 状态 | 含义 |
|---|---|
| CHECK_PASS | 两条实现精度通过，只做正确性，不做性能对照 |
| PASS | 两条实现和实际计时输出校验通过，显示延迟和加速比 |
| AITER_INCORRECT | Aiter 数值检查失败，winner 通过；除 check-only 外仍显示延迟和 speedup，退出码保持非零 |
| NOT_COMPARABLE | 执行/输出元数据检查失败，或 winner 精度失败，不生成 speedup |
| ERROR | 数据准备、参考或调用错误；reason 中保留异常 |