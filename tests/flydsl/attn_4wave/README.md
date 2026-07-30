# 4-Wave Attention Bundle

本目录集中保存 `test_attn_gemm.py` 的 4-wave attention 优化闭包，主运行入口仍为上一级的 [`test_attn_gemm.py`](../test_attn_gemm.py)。

## 内容

- `attn_gemm_inline_kv.py`：Fly外壳与完整JIT主体inline实验。
- `fly_isa_priority.py`：严格attention ISA变换、静态HSACO构建与加载。
- `test_attn_gemm_jit.py`：PyHIP JIT production及`setprio_best`实现。
- `test_*.py`：inline、ISA变换和DPP归约回归测试。
- `isa/`：受SHA保护的gfx942归档ISA。
- `tools/`：ATT账本、共发分析和SVG渲染工具。
- `data/`：结构化性能、ISA和ATT分析结果。
- `images/`：cycle-axis和resident-slot图。
- `attn_gemm_optimization.md`：完整优化记录、采纳/失败路线和最终性能表。
- `mfma-valu-coissue.md`：MFMA/VALU/EXP共发微基准说明。
- `TODO.md`：该优化主线的完成项与后续工作。

## 验证

```bash
cd tests/flydsl
python3 -m pytest \
  attn_4wave/test_fly_isa_priority.py \
  attn_4wave/test_attn_gemm_inline_kv.py -q

HIP_VISIBLE_DEVICES=0 H=1 MULT=16 SOFTMAX=1 \
  FLYDSL_RUNTIME_ENABLE_CACHE=0 python3 test_attn_gemm.py
```

绘图工具从任意工作目录运行时，默认读写本目录的 `data/` 和 `images/`：

```bash
python3 tests/flydsl/attn_4wave/tools/render-attn-cycle-axis.py
python3 tests/flydsl/attn_4wave/tools/render-attn-slot-functions.py
```
