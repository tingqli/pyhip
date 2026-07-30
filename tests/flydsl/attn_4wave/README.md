# 4-Wave Attention Bundle

本目录集中保存 `test_attn_gemm.py` 的 4-wave attention 优化闭包。当前唯一运行入口是上一级的 [`test_attn_gemm.py`](../test_attn_gemm.py)，只保留219T旋转反相流水。

当前及以后所有优化记录统一写入 [`attn_gemm_optimization_current.md`](attn_gemm_optimization_current.md)。

## 内容

- `attn_gemm_optimization_current.md`：当前维护的完整决策表、219T实现和增量实验日志。
- `attn_gemm_optimization.md`：早期逐步优化长文，已归档，只读。
- `attn_gemm_inline_kv.py`、`fly_isa_priority.py`、`test_attn_gemm_jit.py`：历史JIT/ISA实验资料，不再由当前入口导入。
- `test_*.py`、`isa/`、`tools/`、`data/`：历史回归、归档ISA及分析产物，用于追溯旧结论。
- `images/`：cycle-axis和resident-slot图。
- `mfma-valu-coissue.md`：MFMA/VALU/EXP共发微基准说明。
- `TODO.md`：历史任务记录；新工作以当前优化总表为准。

## 验证

```bash
cd tests/flydsl
python3 -m pytest \
  attn_4wave/test_fly_isa_priority.py \
  attn_4wave/test_attn_gemm_inline_kv.py -q

HIP_VISIBLE_DEVICES=0 H=1 MULT=16 \
  FLYDSL_RUNTIME_ENABLE_CACHE=0 python3 test_attn_gemm.py
```

绘图工具从任意工作目录运行时，默认读写本目录的 `data/` 和 `images/`：

```bash
python3 tests/flydsl/attn_4wave/tools/render-attn-cycle-axis.py
python3 tests/flydsl/attn_4wave/tools/render-attn-slot-functions.py
```
