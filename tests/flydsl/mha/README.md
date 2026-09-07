# Paged Attention / SWA

唯一测试入口：[test_mha_pa.py](test_mha_pa.py)。同一个case先检查输出正确性，再逐run显示
**acc、时间、TFLOPS、带宽**；AITER参考同时记录实际kernel名称。
日常功能测试也只运行该文件的6项边界检查，不再需要独立的测试工具回归文件。

**运行命令、参数、测试范围及可复现数据统一见 [README.950.md](README.950.md)。**

| 实现 | backend | 范围 |
|---|---|---|
| [mha_pa_bf16_950.py](mha_pa_bf16_950.py) | `8wave` / `persistent`（gfx950） | BF16、D128/192、V128、page64；full/causal/SWA/sink |
| [mha_pa_swa_bf16.py](mha_pa_swa_bf16.py) | `swa` | gfx950/gfx942 BF16，单wave causal SWA；D128/192、V128、page64 |
| [mha_pa_bf16_942.py](mha_pa_bf16_942.py) | `bf16_942` | gfx942 BF16，8-wave；保留原有输入与padding限制 |
| [mha_pa_fp8_942.py](mha_pa_fp8_942.py) | `fp8_942` / `fp8_942_register` | gfx942 FNUZ FP8，LDS/register两种路径 |

除主入口和4个内核外，只保留包初始化及实际依赖的helper：
- [_dsl.py](_dsl.py)：内核使用的FlyDSL适配。
- [_testing.py](_testing.py)、[_perf_cases.py](_perf_cases.py)：输入、FP32参考、调用和shape/FLOPs。
- [_references.py](_references.py)：AITER参考；[_runner.py](_runner.py)：环境与结果保存；
	[_hardware.py](_hardware.py)：可选的只读GPU空闲检查。

本目录只维护这两份顶层文档。原覆盖、性能、优化和交接说明已去重整理到
[README.950.md](README.950.md)；旧基准、审计、导出、监测脚本及其测试已删除。
已有原始JSON、JUnit、日志、ISA及自动生成的结果表保留，不改写历史证据。