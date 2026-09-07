# 原生FP8/gfx942基线摘录（迁移副本）

这是一份**历史文档摘录，不是当前版本的新测量**。用于在只携带新MHA目录和Git历史时，
仍能明确400T验收的原始条件。旧工作区独立报告的全文SHA256为
`393f703e0c77eb632bf709d8bc573b69515abbdbc4bbc52efe8cc9d96512442c`。

原文件定位（旧工作区的未跟踪实验目录，不要求目标机器存在）：

```text
tests/flydsl/pa_8wave/tests/flydsl/pa_8wave/new_native_gfx942_results.md
日期：2026-09-06
```

## 原文关键摘录

> 最新更新：保留S0优化，S5在最大值 `ds_bpermute` 与首次消费之间插入两条独立PV MFMA。
> 当前公开接口独立验收 **648.421µs / 413.984T**，相对前一409T版本5组ABBA配对延迟下降 **1.022%**。

> **重要硬件前提：验收时 GPU0 的 PTL 为 Enabled / VECTOR,F8。**
> 此配置经用户明确授权临时设置；未修改时钟、650W功率上限或其他GPU。
> 完成验收后已恢复 GPU0 原始 **PTL Disabled**，并读取确认。

## 口径

| 项目 | 历史条件 |
|---|---|
| GPU | AMD Instinct MI308X / gfx942 / 80 CU |
| 输入 | B1/Hq16/Hkv1/Q10240/KV2560/Dq192/Dv128/page64，NC，无LSE |
| 类型/布局 | Q/K/V FNUZ E4M3，BF16 O，SHUFFLE-5D，LDS mode |
| 输入生成 | FP32随机数直接cast为FP8，Q逐token/head unit scale，K/V scalar unit scale |
| 有效FLOPs | 268435456000 |
| 历史S5测量 | 648.420663043478µs / 413.983500680022 TFLOPS |
| 明确验收门槛 | ≥400 TFLOPS；对应≤671.08864µs |
| Python / PyTorch / HIP | 3.11.11 / 2.12.1+rocm7.2 / 7.2.53211 |
| FlyDSL | **0.2.2**；当前统一MHA使用0.3.1，须记录该差别 |
| 硬件策略 | PTL Enabled / VECTOR,F8，auto-DPM，未锁频，原650W上限 |
| 计时 | profiler，1200共同预热，5轮交替；每轮20warmup/100samples；丢首样本、1.5IQR均值，轮次中位数 |
| 正确性 | 全输出FP32参考，rtol=atol=0.1；重复输出逐位一致 |

该内核原源码来自Git提交`23cc6d1e95b1611493e21232bef5d9962b7b73c9`，路径及源码
SHA由 [validate_preservation.py](../../validate_preservation.py) 的 `SOURCES` 固定。
旧报告正文保留了更早的S0数字402.805T，不能用它取代最新S5数字，也不能把Enabled与Disabled
结果混在一起。目标机器不是MI308X或策略/输入/计时不匹配时只能作为诊断比较。

本摘录不复制旧报告中指向未打包ATT/UI目录的链接，不宣称那些旧原始分析文件包含在本次提交。
当前可移植的测量与开发状态见 [CONTEXT_HANDOFF.md](../../CONTEXT_HANDOFF.md)。