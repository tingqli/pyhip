# GR write 正确性回归测试

性能入口、完整性能数据和接入示例见 [benchmark readme](../../../benchmarks/gr_write/readme.md)。
输入生成、参考实现与共用误差检查位于 [pyhip.testing.gr_write](../../../src/pyhip/testing/gr_write.py)；
benchmark 不加载本目录脚本。

## 运行

在仓库根目录、已安装 ROCm PyTorch/FlyDSL 的环境中运行：

```bash
# 整个 GR write pytest：只检查正确性，不做性能采样。
HIP_VISIBLE_DEVICES=2 PYTHONPATH=src python3 -m pytest -q tests/ops/gr_write/test_gr_write.py

# 单个行数 / Graph replay / 参数检查。
HIP_VISIBLE_DEVICES=2 PYTHONPATH=src python3 -m pytest -q tests/ops/gr_write/test_gr_write.py -k "12000 or graph or rejects"

# CLI：先做精度检查，再用 cudaPerf 测 Total（默认 --buffers 10 --warmup 2 --iters 10，取中位数）。
python3 tests/ops/gr_write/test_gr_write.py --gpu 2 --rows 1 33 4097 12000

# 只测精度。
python3 tests/ops/gr_write/test_gr_write.py --gpu 2 --rows 12000 --check-only
```

## 覆盖范围

- `prepare_weights` 的打包布局（MFMA A 操作数分片、按 apply 列归属排列的 FP32 `1 + w`）。
- `ROW_SIZES` 中的 T = 1, 2, 3, 4, 5, 33, 129, 1000, 4097, 12000 与参考实现对比；包括不足 4 行的 quad 尾部、
  行数少于 persistent workgroup 数的情况，以及输入不被修改。
- 外部 `out` / `normed` 前后各留一行 guard，检查越界写；重复调用逐位一致。
- 非默认 `eps` 的单独编译、T=0、参数/对齐/重叠/梯度拒绝。
- 预热后 CUDA Graph capture，并在更换输入后多次 replay（apply 的行计数器由 gate kernel 每次清零）。

## 数值语义与容差

符号（y、r、n、W、w、a、C、H、eps）与上游公式的来源见
[benchmark readme 的“符号与来源”](../../../benchmarks/gr_write/readme.md#符号与来源)。参考实现
`pyhip.testing.gr_write.reference` 按 kernel 的舍入点计算：

- `dot = n[m] · W[c]` 用 FP64 计算后转 FP32；`a = 2 / (1 + exp(−dot / C))`，FP32。
- `out = bf16(fp32(r + a·y))`：`r + a·y` 用 FP64 计算后转 FP32，模拟一次 FP32 FMA，再按最近偶数舍入到 BF16。
- `rstd = fp32(rsqrt(Σ_i out² / H + eps))`（FP64 计算），`normed = bf16(fp32(fp32(out · rstd) · fp32(1 + w)))`；
  每个 H = 2560 宽的 branch 单独归一化。

Kernel 的 gate 点积按 MFMA 顺序做 FP32 累加（gate.py），rstd 用硬件近似的 `v_rsq_f32`（apply.py 中的
`rocdl.rsq`），因此个别元素会在舍入边界上差 1 个 BF16 ULP。`check_close` 同时要求（常量都在
`pyhip.testing.gr_write`）：

- `out` 满足 `OUT_TOLERANCE = dict(rtol=2**-7, atol=1e-5)`，`normed` 满足
  `NORMED_TOLERANCE = dict(rtol=2**-6, atol=1e-4)`；
- 两者与参考逐位相同的比例都不低于 `MIN_EXACT_FRACTION = 0.999`。

2026-10-09 在 GPU2（MI308X）上运行 `python3 tests/ops/gr_write/test_gr_write.py --gpu 2 --rows 12000 --check-only`，
输出 `T=12000: accuracy OK (exact out 0.999986, normed 0.999982)`。

当前 GPU 正确性基线是 MI308X / gfx942 / 80CU；其他架构会给出 warning 后继续执行（host.py）。pytest 不要求空闲 GPU。
