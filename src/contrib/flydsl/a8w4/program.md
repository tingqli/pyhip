# 工作流程

理解问题 -> 调研优化思路 -> 逐个尝试 -> 存档

## 理解问题
阅读 README.md 明确：
 - 本目录kernel的优化目标
 - 目前尝试过的优化方案，包括有效和无效的

## 调研优化思路

到网上搜索相关优化（包括但不局限于AMDGPU/ROCM的），学习优化方法，评估使用该方法，是否可能产生进一步优化效果
 - 根据对 README.md 的理解总结搜索关键词
 - vllm-project/humming, rocm/aiter, DeepEP, DeepGEMM、ByteDance Flux / COMET, MegaBlocks（Stanford / Databricks 开源）, TensorRT-LLM，SGLang / vLLM，Megatron-LM，FlashInfer 以及 TokenSpeed 等项目的当前代码以及open PR
 - 分析理解如何适配到a8w4 moe tp/ep/mega 的优化

## 逐个尝试
整理本次收集到的优化思路，对照 experiments/log_xxx.md 中已经尝试过的实验 (xxx是年月日)，形成本次值得尝试的列表，并逐项尝试，尝试产出和实验结果都记录到experiments子目录下的对应子目录中。

## 存档
在 experiments/log_xxx.md 记录本次运行的全部实验的汇总(xxx是年月日)，并反思本次调研+分析+实验的收获和未来值得深入的方向。给出值得替换/补充当前 kernels 实现的建议


