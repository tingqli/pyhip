# 工作流程

理解问题 -> 调研优化思路 -> 逐个尝试 -> 存档

细节（计时规则、工具用法、落地清单）见 skill `a8w4-perf-experiment`（pyhip/.claude/skills/），改 kernel 前读 `flydsl-gfx942-pitfalls`。命令都在 `src/contrib/flydsl` 下运行。

## 理解问题
阅读 README.md 明确：
 - 本目录kernel的优化目标（§4 kernel 选择、§6 调优、§10 踩过的坑、§11 后续方向）
 - 目前尝试过的优化方案，包括有效和无效的：先查登记表 `python a8w4/exp.py ls` / `python a8w4/exp.py grep <关键词>`（experiments/index.csv，一行一个实验或想法），需要细节时再打开对应 experiments/log_*.md 的那一节；verdict 为 invalidated 的数字不要引用
 - 在重点范围做基线和分解：`python -m a8w4.bench breakdown --case ep8 -t 1024 4096 16384 32768`（tp8 同理）

## 调研优化思路

到网上搜索相关优化（包括但不局限于AMDGPU/ROCM的），学习优化方法，评估使用该方法，是否可能产生进一步优化效果
 - 根据对 README.md 的理解总结搜索关键词
 - vllm-project/humming, rocm/aiter, DeepEP, DeepGEMM、ByteDance Flux / COMET, MegaBlocks（Stanford / Databricks 开源）, TensorRT-LLM，SGLang / vLLM，Megatron-LM，FlashInfer 以及 TokenSpeed 等项目的当前代码以及open PR
 - 分析理解如何适配到a8w4 moe tp/ep/mega 的优化
 - 提示：目前主要关注的total_num_tokens范围是1024~32768

## 逐个尝试
整理本次收集到的优化思路，对照 experiments/index.csv（及 experiments/log_xxx.md）中已经尝试过的实验 (xxx是年月日)，形成本次值得尝试的列表，并逐项尝试，尝试产出和实验结果都记录到experiments子目录下的对应子目录中：
 - `python a8w4/exp.py new <名字>` 建目录；扫参前先读调度代码，确定哪些 token 点会有差异
 - 计时只用 `python -m a8w4.bench ab`（单卡）/ `python -m a8w4.bench dist`（megamoe 8 卡），`--out` 写到实验目录；日志里贴工具生成的表，不手抄数字、不手算百分比
 - 正确性 `python -m a8w4.bench check`，资源 `python -m a8w4.prof isa`，机制 `python -m a8w4.prof counters`
 - 收益要有机制解释，解释不通先怀疑测量

## 存档
在 experiments/log_xxx.md 记录本次运行的全部实验的汇总(xxx是年月日)，并反思本次调研+分析+实验的收获和未来值得深入的方向。给出值得替换/补充当前 kernels 实现的建议
 - 每个实验和新想法用 `python a8w4/exp.py add ...` 写一行 index；`python a8w4/exp.py lint` 不能有 ERROR
 - 落地的改动按 skill 的落地清单同步调优表（`python -m a8w4.tune tune-all` / `compare` / `readme --write`）、README 和测试


