# GR read 正确性回归

性能入口、完整性能数据、公共接口接入示例和 MI308X/80CU 预热档位见 [benchmark README](../../../benchmarks/gr_read/readme.md)。正式性能脚本为 [bench_gr_read_compare.py](../../../benchmarks/gr_read/bench_gr_read_compare.py)，固定 Torch/Triton 对照集中在同目录的单文件 `baselines.py`。

本目录保留 `test_gr_read.py`、pytest 配置与兼容 CLI。输入生成、参考计算、guard 和共用正确性检查位于 [pyhip.testing.gr_read](../../../src/pyhip/testing/gr_read.py)；benchmark 不加载本目录的测试脚本。

## 运行

在仓库根目录、已安装 ROCm PyTorch/FlyDSL 的环境中运行：

```bash
# 整个 GR read pytest：只检查正确性，不做性能采样。
HIP_VISIBLE_DEVICES=2 PYTHONPATH=src python3 -m pytest -q tests/ops/gr_read/test_gr_read.py

# API、工作区和 Graph/stream/device 回归；两张可见 GPU 时包含跨设备检查。
HIP_VISIBLE_DEVICES=2,3 PYTHONPATH=src python3 -m pytest -q tests/ops/gr_read/test_gr_read.py -k test_api

# 固定对照的精度和 Graph replay。
HIP_VISIBLE_DEVICES=2 PYTHONPATH=src python3 -m pytest -q tests/ops/gr_read/test_gr_read.py -k baseline

# 实际 P 地址及报告生成的定向回归。
HIP_VISIBLE_DEVICES=2 PYTHONPATH=src python3 -m pytest -q tests/ops/gr_read/test_gr_read.py \
  -k 'addresses or partial_logging or benchmark_report or benchmark_markdown'

# 保留原 CLI：完整精度检查后，调用 benchmarks/gr_read 的采样实现。
python3 tests/ops/gr_read/test_gr_read.py --gpu 2

# 仅精度：decode 原 2 对权重、T1–32、6528 次 Graph replay。
python3 tests/ops/gr_read/test_gr_read.py --gpu 2 --phase decode --check-only
python3 tests/ops/gr_read/test_gr_read.py --gpu 2 --phase prefill --check-only

# 单阶段核验；Up 检查会先生成并校验 Down 的 P。
python3 tests/ops/gr_read/test_gr_read.py --gpu 2 --phase decode --scope down --rows 1 16 17 32
python3 tests/ops/gr_read/test_gr_read.py --gpu 2 --phase prefill --scope up --rows 33 128 129 512 513
```

兼容 CLI 的 `--decode-weights` / `--decode-seed` 仅控制精度工作集，默认 2 对权重 / seed303；性能仍使用原 100 对权重 / seed707。调整性能采样参数应使用独立 benchmark。兼容 CLI 也支持 `--no-baselines`、`--output` 和 `--md`；单独 `--scope` 或 `--check-only` 不做性能测量。

## 覆盖范围

- 准备对象 `GRReadDecode`、`GRReadPrefill` 与函数式 `gr_read()` 的数值、shape、空输入和尾行。
- Decode 紧凑 FP32 P `[4,T,320]`；Prefill BF16 P `[T,320]`。P/Y 污染、两端 guard、输入/权重不变及改变输入后的 replay。
- 编译缓存复用、输出生命周期、Graph 工作区、stream/device 和接口契约。
- 固定 Torch/Triton 对照的独立原始权重副本、无框架依赖和 counter replay。
- Decode capture / Prefill sample 的真实 P 地址、无 tensor 引用残留、纯打印禁用观察器和输出选项传递。
- Markdown 表格、对照关闭时的报告内容，以及已有文件/路径冲突的拒绝。

当前 GPU 正确性基线是 MI308X / gfx942 / 80CU。其他 ROCm 架构发出 warning 后继续执行，实际编译或精度错误正常失败；不能将本机通过推断为其他机器也通过。pytest 不要求空闲 GPU，不运行性能入口的硬件检查。新增 kernel 时可在此目录添加对应回归，现有测试保持原精度阈值。

## 数值与历史验收说明


Down 保持完整 K 的 FP32 累加 → BF16 舍入 → 转回 FP32 除 4、SiLU → BF16 P。Up 保持原 FP32 logits、四路顺序 FMA 及最终 BF16 舍入。这次调整 tile、调度和准备接口，未删除 SiLU 之前的 BF16 舍入边界。

2026-09-21 的调优及正式目录迁移分别验证了 100 对真实 HC 权重 × 30 档 T（48,64,…,512）× 初始/改变 X，共 6000 例 P/Y 逐位相同。X 仍为随机 BF16，未作端到端模型质量评估。正式入口的 50 项 pytest 全部通过，包含空输入、尾块、配置边界、工作区、stream 和双 GPU。

“逐位相同”指保留同事原 prefill 的结果，不表示解决已有参考误差。之前真实权重 T33/T512 的 400 例检查中，原 prefill 有 16 例 P、1 例 Y 未通过原 BF16 Torch reference 容差；本次没有放宽容差，也没有宣称修复这些舍入边界案例。
